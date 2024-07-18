import argparse
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from util import instantiate_from_config
import cv2
import json
import kornia as kn
import open3d as o3d
import quaternion


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

def make_M_from_tqs(t: list, q: list, s: list, center=None) -> np.ndarray:
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M

def decompose_mat4(M: np.ndarray) -> tuple:
    R = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])
    s = np.abs(s)

    R[:, 0] /= sx
    R[:, 1] /= sy
    R[:, 2] /= sz

    q = quaternion.as_float_array(quaternion.from_rotation_matrix(R[0:3, 0:3]))
    # q = quaternion.from_float_array(quaternion_from_matrix(M, False))

    t = M[0:3, 3]
    return t, q, s, R


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--category",
        type=str,
        default=None
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
        "--normalized_depth",
        type=bool,
        default=False,
        help="whether using normalized depth as input, this should be consistent with the training setting"
    )

    parser.add_argument(
        "--data_path",
        type=str,
    )
    parser.add_argument(
        "--pose_gt_root",
        type=str,
    )

    parser.add_argument(
        "--mesh_root",
        type=str,
        help="to get the centroid offset for the canonicalized mesh to align with the GT pose"
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

    parser.add_argument(
        "--num_iters",
        type=int,
        default=3,
        help="number of different input subsampled pointcloud",
    )
    
    parser.add_argument(
        "--gt_pose",
        type=bool,
        default=False,
        help="whether have access to ground truth pose&rendered depth",
    )
    
    parser.add_argument(
        "--pred_scale_dir",
        type=str,
    )

    opt = parser.parse_args()

    with open(opt.pred_scale_dir, 'r') as f:
        scales_fullset = json.load(f)

    config = OmegaConf.load(opt.config_path)

    print("evaluate model with parameterization of {}".format(config.model.params.parameterization))
    
    model_path = opt.model_path
    
    if opt.gt_pose:
        with open(opt.pose_gt_root, 'r') as f:
            pose_gts = json.load(f)

    ckpt_name = model_path.split('/')[-1].split('.')[0]

    print("running generation on {}".format(model_path))
    model = load_model_from_config(config, model_path)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    with open(opt.split_path, 'r') as f:
        split_lines = f.read().splitlines()

    output_path = os.path.join(opt.outdir, ckpt_name)
    print("Saving visuals to {}".format(output_path))
    os.makedirs(output_path, exist_ok=True)

    for line in tqdm(split_lines):
        ori_cloud_cond_batch = []

        # read the target frame
        scene_id, frame_idx = line.split()
        scene_info = scene_id + '_' + frame_idx
        frame_id, mesh_id, inst_id = frame_idx.split('_')

        scales_subset = []

        for x in range(opt.num_iters):
            scales_subset.append(scales_fullset[scene_id + ' ' + frame_idx][str(x)][0])

        scales_subset = sorted(scales_subset)
        print('number of scales per-scene: ', len(scales_subset))

        for i, depth_aug_scalar in enumerate(scales_subset):
            if opt.gt_pose:
                depth_gt_fname = os.path.join(opt.data_path, "Rendering", scene_id, 'depth', "{}{}".format(frame_id, '.png'))
                depth_gt = cv2.imread(depth_gt_fname, -1)
                gt_pose_ws = np.asarray(pose_gts[scene_info]).reshape(4, 4)
                t_gt, q_gt, s_gt, R_gt = decompose_mat4(gt_pose_ws)
                scale_gt = np.linalg.norm(gt_pose_ws[:3, :3], axis=0)
                with open(os.path.join(opt.mesh_root, mesh_id, 'mesh_info.json'), 'r') as f:
                    centroid_offset = json.load(f)["centroid_offset"]
                centroid_offset = np.asarray(centroid_offset)
                offset = np.eye(4)
                offset[:3, 3] = -centroid_offset
                pose_recalib = gt_pose_ws @ offset
                
            depth_fname = os.path.join(opt.data_path, 'ZoeDepthPredictions', scene_id, "{}_pred_dmap.npy".format(frame_id))
            depth = np.load(depth_fname)
            depth = (depth * 1000)
            depth = (depth / depth_aug_scalar).astype(np.uint16)

            mask_fname = os.path.join(opt.data_path, "ODISEPredictions_NEW", scene_id, opt.category, frame_idx+'.png')
            mask = cv2.imread(mask_fname, -1) / 255

            mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
            kernel = torch.ones(7, 7)
            mask = kn.morphology.erosion(mask, kernel)

            mask_ero = mask.squeeze(0).squeeze(0).detach().cpu().numpy()

            target_depth = depth * mask_ero.astype(np.uint8)
            
            if opt.gt_pose:
                target_depth_gt = depth_gt * mask_ero.astype(np.uint8)
                target_depth_gt = o3d.geometry.Image(target_depth_gt)
                pcd_trans_gt = o3d.geometry.PointCloud.create_from_depth_image(target_depth_gt, intr, pose_recalib)
                gt_nocs_rendered = np.asarray(pcd_trans_gt.points)

            target_depth = o3d.geometry.Image(target_depth)
            target_depth_full = o3d.geometry.Image(depth)
            cam_K = np.asarray([434.98, 0.0, 239.36, 0.0, 434.05, 182.2, 0.0, 0.0, 1.0], dtype=np.float32).reshape(3, 3)
            intr = o3d.camera.PinholeCameraIntrinsic(480, 360, cam_K)
            pcd_orig = o3d.geometry.PointCloud.create_from_depth_image(target_depth, intr)

            ori_cloud = np.asarray(pcd_orig.points)

            ori_cloud_normalized = (ori_cloud - ori_cloud.min(axis=0)) / ((ori_cloud.max(axis=0) - ori_cloud.min(axis=0)).max() + 1e-15)

            sample_num_pc = 1024
            indices_pc = np.random.choice(ori_cloud.shape[0], size=sample_num_pc, replace=False)
            ori_cloud = ori_cloud[indices_pc, :]
            ori_cloud_normalized = ori_cloud_normalized[indices_pc, :]

            if opt.normalized_depth:
                ori_cloud_cond = torch.from_numpy(ori_cloud_normalized).float()
            else:
                ori_cloud_cond = torch.from_numpy(ori_cloud).float()

            ori_cloud_cond = ori_cloud_cond.unsqueeze(0).to(device)
            ori_cloud_cond_batch.append(ori_cloud_cond)

            # save the original depth for pose solver
            depth_fname = os.path.join(output_path, scene_info + '_depth_input_{}.ply'.format(i))
            pcd_depth = o3d.geometry.PointCloud()
            pcd_depth.points = o3d.utility.Vector3dVector(ori_cloud)

            o3d.io.write_point_cloud("{}".format(depth_fname), pcd_depth, write_ascii=False)
        ori_cloud_cond_batch = torch.cat(ori_cloud_cond_batch, dim=0)
        with torch.no_grad():
            with model.ema_scope():
                cond = model.get_learned_conditioning(ori_cloud_cond_batch)

                samples, _ = model.sample(cond=cond, batch_size=opt.num_iters, return_intermediates=True)
                samples = model.decode_first_stage(samples)

                for j in range(opt.num_iters):
                    pred_nocs = samples[j].permute(1, 0)
                    pred_nocs = pred_nocs.detach().cpu().numpy()
                    pred_fname = os.path.join(output_path, scene_info + '_nocs_pred_{}.ply'.format(j))
                    pcd_pred = o3d.geometry.PointCloud()
                    pcd_pred.points = o3d.utility.Vector3dVector(pred_nocs)

                    o3d.io.write_point_cloud("{}".format(pred_fname), pcd_pred, write_ascii=False)
