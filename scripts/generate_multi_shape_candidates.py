import argparse
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from util import instantiate_from_config
import cv2
import json
import open3d as o3d


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
    )

    parser.add_argument(
        "--model_path",
        type=str,
    )
    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument(
        "--ply_path",
        type=str,
        help="path to the predicted NOCs, which are used as input to the model"
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=3,
        help="number of different input NOCs",
    )

    parser.add_argument(
        "--latent_root",
        type=str,
        help="path to the latent codes for retrieval"
    )


    parser.add_argument(
        "--split_path",
        type=str,
        help="read the split",
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
    )


    opt = parser.parse_args()

    config = OmegaConf.load(opt.config_path)

    print("evaluate model with parameterization of {}".format(config.model.params.parameterization))

    model_basename = opt.model_path.split('/')[-3]
    model_path = opt.model_path

    ckpt_name = model_path.split('/')[-1].split('.')[0]

    print("running generation on {}".format(model_path))
    model = load_model_from_config(config, model_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    with open(opt.split_path, 'r') as f:
        split_lines = f.read().splitlines()

    with open(os.path.join(opt.data_path, 'CAD_Pools/scan2cad_cad_pool_{}.json'.format(opt.category)), 'r') as f:
        cad_pool_s2c = json.load(f)

    latent_joint = []
    latent_joint_ids = []
    for cad_id in cad_pool_s2c['train']:
        latent_joint_ids.append(cad_id)
        latent_joint.append(torch.load(
            os.path.join(opt.latent_root, cad_id+'.pt'), map_location='cpu').squeeze(0))
    latent_joint = torch.stack(latent_joint).to(device)

    print("Loaded total {} latents".format(latent_joint.shape[0]))

    output_path = os.path.join(opt.outdir, model_basename, ckpt_name)
    os.makedirs(output_path, exist_ok=True)

    all_samples = list()

    cham_dists_l1 = []
    cham_dists_l2 = []
    cham_dists_l1_eu = []
    cham_dists_l2_eu = []
    inference_results = {}

    for line in tqdm(split_lines):
        scene_id, frame_idx = line.split()
        scene_info = scene_id + '_' + frame_idx
        inference_results[scene_info] = {}

    for line in tqdm(split_lines):
        cond_batch = []
        for i in range(opt.num_iters):
            scene_id, frame_info = line.split()
            scene_info = scene_id + '_' + frame_info
            frame_id, latent_gt_idx, instance_id = frame_info.split('_')

            nocs_fname = os.path.join(opt.ply_path, scene_info + '_nocs_pred_{}.ply'.format(i))
            nocs_ply = o3d.io.read_point_cloud(nocs_fname)
            points3d = np.asarray(nocs_ply.points)

            sample_num = 1024
            indices = np.random.randint(points3d.shape[0], size=sample_num)
            nocs_pc = points3d[indices, :]
            nocs_pc = torch.from_numpy(nocs_pc)  # num_points, 3
            nocs_pc = nocs_pc[None].to(device).float()
            cond_batch.append(nocs_pc)

        cond_batch = torch.cat(cond_batch, dim=0)
        with torch.no_grad():
            with model.ema_scope():
                c = model.get_learned_conditioning(cond_batch).float()
                samples, _ = model.sample(cond=c, batch_size=opt.num_iters, return_intermediates=True) # B 256 1 1
                x_samples = model.decode_first_stage(samples)

                dists = torch.zeros(latent_joint.shape[0]).to(device)

                for x in range(opt.num_iters):
                    for j in range(latent_joint.shape[0]):
                        dists[j] = torch.nn.functional.cosine_similarity(x_samples[x].squeeze(-1).squeeze(-1), latent_joint[j], dim=0)
                    top_idx = latent_joint_ids[torch.argmax(dists).item()]
                    retrieved_id = top_idx
                    gt_id = latent_gt_idx

                    query_lat_eu = x_samples[x].unsqueeze(0).squeeze(-1).squeeze(-1)
                    eu_dists = torch.cdist(latent_joint, query_lat_eu, p=2).squeeze(-1)  # N
                    ret_id_eu = latent_joint_ids[torch.argmin(eu_dists).item()]

                    inference_results[scene_info]['gt_latent_idx'] = gt_id
                    inference_results[scene_info]['retrieved_latent_idx_{}'.format(x)] = retrieved_id # retrieval based on cosine similarity
                    inference_results[scene_info]['retrieved_latent_idx_{}_eu'.format(x)] = ret_id_eu # retrieval based on euclidean distance

    print("Saving results to {}".format(output_path))

    with open(os.path.join(output_path, 'inference_results_.json'), 'w', encoding='utf-8') as f:
        json.dump(inference_results, f, ensure_ascii=False, indent=2)
