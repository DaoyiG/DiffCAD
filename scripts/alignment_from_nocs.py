import argparse
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from util import instantiate_from_config
from data.data_util import estimate9DTransform, to_homo
import json
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


def calc_rotation_diff(q: np.quaternion, q00: np.quaternion) -> float:
    np.seterr(all='raise')
    # rotation_dot = np.dot(quaternion.as_float_array(q00), quaternion.as_float_array(q))
    rotation_dot = q00[0] * q[0] + q00[1] * q[1] + q00[2] * q[2] + q00[3] * q[3]

    rotation_dot_abs = np.abs(rotation_dot)
    try:
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)

    return error_rotation


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

    R[:, 0] /= sx
    R[:, 1] /= sy
    R[:, 2] /= sz

    q = quaternion.as_float_array(quaternion.from_rotation_matrix(R[0:3, 0:3]))
    # q = quaternion.from_float_array(quaternion_from_matrix(M, False))

    t = M[0:3, 3]
    return t, q, s, R


def re(R_est, R_gt):
    """Rotational Error.

    :param R_est: 3x3 ndarray with the estimated rotation matrix.
    :param R_gt: 3x3 ndarray with the ground-truth rotation matrix.
    :return: The calculated error.
    """
    assert R_est.shape == R_gt.shape == (3, 3)
    rotation_diff = np.dot(R_est, R_gt.T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    # Avoid invalid values due to numerical errors
    error_cos = min(1.0, max(-1.0, 0.5 * (trace - 1.0)))
    rd_deg = np.rad2deg(np.arccos(error_cos))

    return rd_deg


def te(t_est, t_gt):
    """Translational Error.

    :param t_est: 3x1 ndarray with the estimated translation vector.
    :param t_gt: 3x1 ndarray with the ground-truth translation vector.
    :return: The calculated error.
    """
    t_est = t_est.flatten()
    t_gt = t_gt.flatten()
    assert t_est.size == t_gt.size == 3
    error = np.linalg.norm(t_gt - t_est)
    return error


def se(s_est, s_gt):
    """Scale Error.

    :param s_est: 3x1 ndarray with the estimated scale vector.
    :param s_gt: 3x1 ndarray with the ground-truth scale vector.
    :return: The calculated error.
    """
    s_est = s_est.flatten()
    s_gt = s_gt.flatten()
    assert s_est.size == s_gt.size == 3
    # error = np.abs(np.mean(s_est/s_gt) - 1)
    error = np.mean(np.abs((s_est/s_gt) - 1))
    return error


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prediction_path",
        type=str,
        help="path to the predictions of NOCs of a category",
    )

    parser.add_argument(
        "--pose_gt_root",
        type=str,
        help="path to the ground truth pose json",
    )

    parser.add_argument(
        "--mesh_root",
        type=str,
        help="original shapenet meshes are not canonicalized (they have slight misalignment), the original ground truth pose is defined based on those meshes. We canonicalize those meshes and save the centroid offset into the mesh root",
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
        default=2,
        help="number of different input candidates",
    )
    
    parser.add_argument(
        "--visualize",
        type=bool,
        default=False,
        help="visualize the alignment results (object point clouds with transformed NOCs)",
    )

    opt = parser.parse_args()

    with open(opt.pose_gt_root, 'r') as f:
        pose_gts = json.load(f)

    with open(opt.split_path, 'r') as f:
        split_lines = f.read().splitlines()

    os.makedirs(opt.outdir, exist_ok=True)

    res = []
    tes = []
    ses = []
    invalid_pose = []
    prediction = {}

    for line in tqdm(split_lines):

        # read the target frame
        scene_id, frame_idx = line.split()
        scene_info = scene_id + '_' + frame_idx
        frame_id, mesh_id, inst_id = frame_idx.split('_')
        prediction[scene_info] = {}

        gt_pose_ws = np.asarray(pose_gts[scene_info]).reshape(4, 4)
        gt_pose_ws = np.asarray(gt_pose_ws).reshape(4, 4)

        with open(os.path.join(opt.mesh_root, mesh_id, 'mesh_info.json'), 'r') as f:
            centroid_offset = json.load(f)["centroid_offset"]

        centroid_offset = np.asarray(centroid_offset)
        offset = np.eye(4)
        offset[:3, 3] = -centroid_offset

        cam_K = np.asarray([434.98, 0.0, 239.36, 0.0, 434.05, 182.2, 0.0, 0.0, 1.0], dtype=np.float32).reshape(3, 3)

        pose_recalib = gt_pose_ws @ offset

        t_gt, q_gt, s_gt, R_gt = decompose_mat4(pose_recalib)

        scale_gt_frompose = np.linalg.norm(pose_recalib[:3, :3], axis=0)
        assert np.allclose(s_gt, scale_gt_frompose)

        best_transforms = []
        best_ratios = []
        best_nocss = []
        best_errs = []
        for i in tqdm(range(opt.num_iters)):

            prediction[scene_info]['gt_pose'] = pose_recalib.tolist()

            prediction[scene_info]['{}'.format(i)] = {}

            ori_cloud_ply = o3d.io.read_point_cloud(os.path.join(opt.prediction_path, scene_info + '_depth_input_{}.ply'.format(i)))
            pred_nocs_ply = o3d.io.read_point_cloud(os.path.join(opt.prediction_path, scene_info + '_nocs_pred_{}.ply'.format(i)))

            ori_cloud = np.asarray(ori_cloud_ply.points)

            pred_nocs = np.asarray(pred_nocs_ply.points)

            min_scale = [0.8, 0.8, 0.8]
            max_scale = [4.0, 4.0, 4.0]
            max_dimensions = np.array([1.2, 1.2, 1.2])

            best_ratio = 0
            best_transform = None
            best_err = np.inf
            for thres in [0.001, 0.005, 0.01, 0.05, 0.1]:
                use_kdtree_for_eval = False
                kdtree_eval_resolution = 1
                transform, inliers = estimate9DTransform(source=pred_nocs, target=ori_cloud, PassThreshold=thres, max_iter=5000,
                                                         use_kdtree_for_eval=use_kdtree_for_eval, kdtree_eval_resolution=kdtree_eval_resolution,
                                                         max_scale=max_scale, min_scale=min_scale, max_dimensions=max_dimensions)

                if transform is not None:
                    transformed = (transform@to_homo(pred_nocs).T).T[:, :3]
                    errs = np.linalg.norm(transformed-ori_cloud, axis=1)
                    total_err = errs.mean()
                    if total_err < best_err:
                        best_transform = transform.copy()
                        best_err = total_err.copy()
                        best_nocs = pred_nocs.copy()

            if best_transform is not None:
                best_transforms.append(best_transform)
                best_errs.append(best_err)
                best_nocss.append(best_nocs)
                prediction[scene_info]['{}'.format(i)]['best_transform'] = best_transform.tolist()
                
        if len(best_errs) != 0:
            # pick one with the best ratio
            best_rat_idx = np.argmin(np.asarray(best_errs))
            best_transform_selected = best_transforms[best_rat_idx]
            best_pred_nocs = best_nocss[best_rat_idx]

            # save the depth
            depth_fname = os.path.join(opt.outdir, scene_info + '_depth_input.ply')

            o3d.io.write_point_cloud("{}".format(depth_fname), ori_cloud_ply, write_ascii=False)

            pred_fname = os.path.join(opt.outdir, scene_info + '_best_pred.ply')
            pcd_pred = o3d.geometry.PointCloud()
            pcd_pred.points = o3d.utility.Vector3dVector(best_pred_nocs)

            o3d.io.write_point_cloud("{}".format(pred_fname), pcd_pred, write_ascii=False)

            transformed = (best_transform_selected@to_homo(best_pred_nocs).T).T[:, :3]
            pred_trans_fname = os.path.join(opt.outdir, scene_info + '_best_pred_transformed.ply')
            pcd_pred_trans = o3d.geometry.PointCloud()
            pcd_pred_trans.points = o3d.utility.Vector3dVector(transformed)

            o3d.io.write_point_cloud("{}".format(pred_trans_fname), pcd_pred_trans, write_ascii=False)
            
            if opt.visualize:
                pcd_pred_trans.paint_uniform_color([1, 0.706, 0])
                ori_cloud_ply.paint_uniform_color([0, 0.651, 0.929])
                o3d.visualization.draw_geometries([pcd_pred_trans, ori_cloud_ply], window_name=scene_info)

            t_pred, q_pred, s_pred, R_pred = decompose_mat4(best_transform_selected)

            rot_err = re(R_pred, R_gt)
            rot_err_1 = calc_rotation_diff(q_pred, q_gt)
            trans_err = te(t_pred.reshape(3, 1), t_gt.reshape(3, 1))
            scale_err = se(s_pred, s_gt)

            print("{} rot err {}; trans err {} ; scale err {} ".format(scene_info, rot_err, trans_err, scale_err))

            res.append(rot_err)
            tes.append(trans_err)
            ses.append(scale_err)


            prediction[scene_info]['predicted_pose'] = best_transform_selected.tolist()
            prediction[scene_info]['gt_pose'] = pose_recalib.tolist()
            prediction[scene_info]['rot_err'] = rot_err.tolist()
            prediction[scene_info]['trans_err'] = trans_err.tolist()
            prediction[scene_info]['scale_err'] = scale_err.tolist()


    with open(os.path.join(opt.outdir, 'pose_predictions.json'), 'w', encoding='utf-8') as f:
        json.dump(prediction, f, ensure_ascii=False, indent=2)
