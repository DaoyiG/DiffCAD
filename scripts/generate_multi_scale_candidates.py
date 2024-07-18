import argparse
import os
import sys
import importlib
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
import torchvision
from models.diffusion.ddim_scale import DDIMSampler
import cv2
import json
import kornia as kn


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=True)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.cuda()
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--category",
        type=str,
        default="02818832"
    )

    parser.add_argument(
        "--config_path",
        type=str,
        help="path to the model config file"
    )

    parser.add_argument(
        "--model_path",
        type=str,
        help="path to the model checkpoint"
    )

    parser.add_argument(
        "--data_path",
        type=str,
        help="path to the data"
    )
    
    parser.add_argument(
        "--split_path",
        type=str,
        help="read the test split",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to"
    )

    parser.add_argument(
        "--num_iters",
        type=int,
        default=5,
        help="number of different input subsampled pointcloud",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    
    parser.add_argument(
        "--gt_scale",
        type=bool,
        default=False,
        help="whether have access to ground truth scale",
    )


    opt = parser.parse_args()

    config = OmegaConf.load(opt.config_path)

    print("evaluate model with parameterization of {}".format(config.model.params.parameterization))

    model_path = opt.model_path

    ckpt_name = model_path.split('/')[-1].split('.')[0]

    print("running generation on {}".format(model_path))
    model = load_model_from_config(config, model_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    with open(opt.split_path, 'r') as f:
        split_lines = f.read().splitlines()

    sampler = DDIMSampler(model)

    output_path = os.path.join(opt.outdir, ckpt_name)
    print("Saving predictions to {}".format(output_path))
    os.makedirs(output_path, exist_ok=True)

    all_samples = list()

    prediction = {}

    min_diffs = []
    mean_diffs = []

    scale_infos = {}
    for line in tqdm(split_lines):
        scale_infos[line] = {}
        
    rz = torchvision.transforms.Resize(size=(30, 40))

    for i in range(opt.num_iters):
        with torch.no_grad():
            with model.ema_scope():
                for line in tqdm(split_lines):
                    # read the target frame
                    scene_id, frame_idx = line.split()
                    scene_info = scene_id + '_' + frame_idx
                    frame_id, mesh_id, inst_id = frame_idx.split('_')
                    prediction[scene_info] = {}
                    
                    if opt.gt_scale:
                        sensor_depth_path = os.path.join(opt.data_path, 'SensorDepth', scene_id, frame_id + '.png')
                        sensor_depth = cv2.imread(sensor_depth_path, -1) / 1000.0

                        sensor_depth = torch.from_numpy(sensor_depth).unsqueeze(0).unsqueeze(0)
                        sensor_depth = torch.nn.functional.interpolate(sensor_depth, size=(360, 480))
                        sensor_depth = sensor_depth.squeeze(0).squeeze(0).detach().cpu().numpy()
                        
                        depth_input = torch.from_numpy(sensor_depth)[None]
                        depth_input = rz(depth_input)

                    depth_fname = os.path.join(opt.data_path, "ZoeDepthPredictions", scene_id, "{}_pred_dmap{}".format(frame_id, '.npy'))
                    depth = np.load(depth_fname)

                    depth_pred_input = torch.from_numpy(depth)[None]
                    depth_pred_input = rz(depth_pred_input).to(device)

                    mask_fname = os.path.join(opt.data_path, "ODISEPredictions_NEW", scene_id, opt.category, frame_idx+'.png')
                    mask_full = cv2.imread(mask_fname, -1) / 255

                    mask = torch.from_numpy(mask_full).unsqueeze(0).unsqueeze(0).float()
                    kernel = torch.ones(3, 3)
                    mask = kn.morphology.erosion(mask, kernel)
                    mask_ero = mask.squeeze(0).squeeze(0).detach().cpu().numpy()
                    
                    pred_target_depth = depth * mask_ero
                    
                    if opt.gt_scale:
                        target_depth = sensor_depth * mask_ero
                        gt_scale = np.mean(pred_target_depth) / np.mean(target_depth)
                        scale_infos[line]['gt'] = gt_scale
                    
                    cond = model.get_learned_conditioning(depth_pred_input.unsqueeze(0)).to(device)

                    samples, _ = model.sample_log(cond=cond, batch_size=1, ddim=True, ddim_steps=200, eta=1.)

                    pred_scales = model.decode_first_stage(samples).squeeze(0).squeeze(0)

                    pred_scales = pred_scales.detach().cpu().numpy()

                    pred_scale_mean = np.mean(pred_scales)

                    scale_infos[line][str(int(i))] = pred_scale_mean + 1.1
                    

with open('{}/predictions.json'.format(output_path), 'w', encoding='utf-8') as f:
    json.dump(scale_infos, f, ensure_ascii=False, indent=2)
