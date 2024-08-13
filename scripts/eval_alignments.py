import os
import numpy as np
import quaternion
import json
from tqdm import tqdm
import cv2
import pycocotools.mask as mask_util
import argparse



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

    t = M[0:3, 3]
    return t, q, s, R

def rotation_diff(q: np.quaternion, q_gt: np.quaternion, sym='') -> float:
    if sym == "__SYM_ROTATE_UP_2":
        m = 2
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_4":
        m = 4
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    elif sym == "__SYM_ROTATE_UP_INF":
        m = 36
        tmp = [calc_rotation_diff(q, q_gt*quaternion.from_rotation_vector([0, (i*2.0/m)*np.pi, 0])) for i in range(m)]
        error_rotation = np.min(tmp)
    else:
        error_rotation = calc_rotation_diff(q, q_gt)
    return error_rotation

def calc_rotation_diff(q: np.quaternion, q00: np.quaternion) -> float:
    np.seterr(all='raise')
    rotation_dot = q00[0] * q[0] + q00[1] * q[1] + q00[2] * q[2] + q00[3] * q[3]

    rotation_dot_abs = np.abs(rotation_dot)
    try:
        error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    except:
        return 0.0
    error_rotation_rad = 2 * np.arccos(rotation_dot_abs)
    error_rotation = np.rad2deg(error_rotation_rad)

    return error_rotation

def rotation_error(R_est, R_gt, sym=''):
    if sym == "__SYM_ROTATE_UP_2":
        m = 2
        tmp = [re(R_est, R_gt @ np.array([[np.cos((i*2.0/m)*np.pi), 0, np.sin((i*2.0/m)*np.pi)], [0, 1, 0], [-np.sin((i*2.0/m)*np.pi), 0, np.cos((i*2.0/m)*np.pi)]])) for i in range(m)]
        error_rotation = np.min(tmp)
    
    elif sym == "__SYM_ROTATE_UP_4":
        m = 4
        tmp = [re(R_est, R_gt @ np.array([[np.cos((i*2.0/m)*np.pi), 0, np.sin((i*2.0/m)*np.pi)], [0, 1, 0], [-np.sin((i*2.0/m)*np.pi), 0, np.cos((i*2.0/m)*np.pi)]])) for i in range(m)]
        error_rotation = np.min(tmp)
        
    elif sym == "__SYM_ROTATE_UP_INF":
        m = 36
        tmp = [re(R_est, R_gt @ np.array([[np.cos((i*2.0/m)*np.pi), 0, np.sin((i*2.0/m)*np.pi)], [0, 1, 0], [-np.sin((i*2.0/m)*np.pi), 0, np.cos((i*2.0/m)*np.pi)]])) for i in range(m)]
        error_rotation = np.min(tmp)
        
    else:
        error_rotation = re(R_est, R_gt)
    return error_rotation

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

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def annotate_roi( mask):
    rle = mask_util.encode(np.array(
        mask[:, :, None], order='F', dtype='uint8'
    ))[0]
    bbox = mask_util.toBbox(rle) # x1, y1, h, w
    area = mask_util.area(rle)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle, bbox.tolist(), area.tolist()
    
def xywh_to_xyxy( xywh):
    """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format.
    https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/boxes.py
    """
    if isinstance(xywh, (list, tuple)):
        # Single box given as a list of coordinates
        assert len(xywh) == 4
        x1, y1 = xywh[0], xywh[1]
        x2 = x1 + np.maximum(0., xywh[2]) # oringal version x2 = x1 + np.maximum(0., xywh[2] - 1.)
        y2 = y1 + np.maximum(0., xywh[3]) # oringal version y2 = y1 + np.maximum(0., xywh[3] - 1.)
        return (x1, y1, x2, y2)
    elif isinstance(xywh, np.ndarray):
        # Multiple boxes given as a 2D ndarray
        return np.hstack(
            (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
        )
    else:
        raise TypeError('Argument xywh must be a list, tuple, or numpy array.')
    
def eval_alignment(opts, splits, gt_annos, pose_predictions, pose_gts):
    total_re = []
    total_te = []
    total_se = []
    tps = []
    missing_preds = []

    for totalinfo in tqdm(splits):
        scene_id, frame_mesh_inst_info = totalinfo.split()
        frame_id, obj_id, inst_id = frame_mesh_inst_info.split('_')
        scene_info = scene_id + '_' + frame_mesh_inst_info
        
        sym_i = None
        for anno in gt_annos:
            if anno['id_scan'] == scene_id:
                for anno_obj in anno['aligned_models']:
                    if anno_obj['id_cad'] == obj_id:
                        sym_i = anno_obj['sym']
        
        # check whether the prediction exists
        if scene_info not in pose_predictions:
            missing_preds.append(scene_info)
            continue
        
        poses = []

        gt_pose = np.asarray(pose_gts[scene_info]).reshape(4, 4) # note that this is without offset recalib!

        with open(os.path.join(opts.mesh_data_path, obj_id, 'mesh_info.json'), 'r') as f:
            mesh_info = json.load(f)

        centroid_offset = mesh_info['centroid_offset']
        centroid_offset = np.asarray(centroid_offset)
        offset = np.eye(4)
        offset[:3, 3] = -centroid_offset

        pose_recalib = gt_pose @ offset
        t_gt, q_gt, s_gt, R_gt = decompose_mat4(pose_recalib)
        try:
            pose = np.asarray(pose_predictions[scene_info]["pose"]).reshape(4, 4)
            poses.append(pose)
            t_pred, q_pred, s_pred, R_pred = decompose_mat4(pose)
            rot_err = rotation_error(R_pred, R_gt, sym=sym_i)
            trans_err = te(t_pred.reshape(3, 1), t_gt.reshape(3, 1))
            scale_err = se(s_pred, s_gt)
        except:
            missing_preds.append(scene_info)
            continue
        
        total_re.append(rot_err)
        total_te.append(trans_err)
        total_se.append(scale_err)
        
        if rot_err < 20 and trans_err < 0.2 and scale_err < 0.2:
            tps.append(1)
        else:
            tps.append(0)
            
    for _ in range(len(missing_preds)):
        tps.append(0)

    print('missing pred', len(missing_preds))
    avg_res = np.asarray(total_re).mean()
    avg_tes = np.asarray(total_te).mean()
    avg_ses = np.asarray(total_se).mean()
    print("valid preds {}; RE {}; TE {}; SE {}".format(len(total_re), avg_res, avg_tes, avg_ses))
    print("total item {}; TP: {}".format(len(tps), np.asarray(tps).mean()))


def eval_alignment_roca(opts, all_scene_ids, roca_per_frame, target_class, gt_annos, gt_poses):
    
    id_name_mapping = {'04256520':'sofa',
                '03001627':'chair',
                '02818832':'bed',
                '02747177':'bin',
                '02933112':'cabinet',
                '03211117':'display',
                '04379243':'table',
                '02871439':'bookcase'
    }
    
    missing_preds = []
    rot_errs = []
    trans_errs = []
    scale_errs = []
    tps = []
    
    for sfid in tqdm(all_scene_ids):
        scene_id, frame_id_mesh_id_inst_id = sfid.split()
        frame_id, mesh_id, inst_id = frame_id_mesh_id_inst_id.split('_')
        scene_info = scene_id + '_' + frame_id_mesh_id_inst_id
        
        gt_pose = np.asarray(gt_poses[scene_info]).reshape(4, 4) # roca learns from the gt that was annotated by raw shapenet meshes
        t_gt, q_gt, s_gt, R_gt = decompose_mat4(gt_pose)
        
        sym_i = None
        for anno in gt_annos:
            if anno['id_scan'] == scene_id:
                for anno_obj in anno['aligned_models']:
                    if anno_obj['id_cad'] == mesh_id:
                        sym_i = anno_obj['sym']
        try:
            roca_frame_info = roca_per_frame['{}/color/{}.jpg'.format(scene_id, frame_id)] # list of dicts
        except:
            missing_preds.append(sfid)
            continue
        
        target_obj_infos = []
        pred_obj = []
        for objinfo in roca_frame_info:
            pred_obj.append(objinfo["category"])
            if objinfo["category"] == id_name_mapping[target_class]:
                target_obj_infos.append(objinfo)
        
        if len(target_obj_infos) == 0:
            missing_preds.append(sfid)
            continue
        
        gt_mask_dir = os.path.join(opts.data_path, 'GTMASK', target_class, scene_id, 'visib_mask/{}.png'.format(frame_id_mesh_id_inst_id))
        gt_visib_mask = cv2.imread(gt_mask_dir, -1) / 255
        _, bbox_xywh, _ = annotate_roi(gt_visib_mask.astype(np.uint8))
        bbox_xyxy = xywh_to_xyxy(bbox_xywh)
        
        max_iou = 0
        matched_pred_info = None
        
        # find registration between the predicted and gt bbox
        for pred_target_info in target_obj_infos:
            pred_bbox = pred_target_info["bbox"]
            
            # find the most matching gt bbox to associate the pose
            # calculate the iou of bbox
            iou = bb_intersection_over_union(pred_bbox, bbox_xyxy)
            if iou > max_iou:
                matched_pred_info = pred_target_info
                
        if matched_pred_info is None:
            missing_preds.append(sfid)
            continue
            
        pred_bbox = matched_pred_info["bbox"]
        
        t_pred = np.asarray(matched_pred_info["t"]).reshape(t_gt.shape)
        q_pred = np.asarray(matched_pred_info["q"]).reshape(q_gt.shape)
        s_pred = np.asarray(matched_pred_info["s"]).reshape(s_gt.shape)
        pred_pose = make_M_from_tqs(t_pred, q_pred, s_pred)
        t_pred1, q_pred1, s_pred1, R_pred = decompose_mat4(pred_pose)
        
        assert np.allclose(t_pred1, t_pred, atol=1e-5), print(t_pred1, t_pred)
        assert np.allclose(s_pred, s_pred1, atol=1e-5), print(s_pred, s_pred1)
        
        rot_err = rotation_error(R_pred, R_gt, sym=sym_i)
        rot_err_q_sym = rotation_diff(q_pred, q_gt, sym=sym_i)
        trans_err = te(t_pred, t_gt)
        scale_err = se(s_pred, s_gt)
        
        rot_errs.append(rot_err)
        trans_errs.append(trans_err)
        scale_errs.append(scale_err)
        
        if rot_err_q_sym < 20 and trans_err < 0.2 and scale_err < 0.2:
            tps.append(1)
        else:
            tps.append(0)

    avg_res = np.asarray(rot_errs).mean()
    avg_tes = np.asarray(trans_errs).mean()
    avg_ses = np.asarray(scale_errs).mean()
    print("valid preds {}; RE {}; TE {}; SE {}".format(len(rot_errs), avg_res, avg_tes, avg_ses))

    for _ in range(len(missing_preds)):
        tps.append(0)

    print("TP: {}; TP: {}".format(len(tps), np.asarray(tps).mean()))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--category",
        type=str,
        default="04379243"
    )

    parser.add_argument(
        "--roca_prediction",
        type=str,
        help="path to the per-frame predictions of roca"
    )
    
    parser.add_argument(
        "--prediction_path",
        type=str,
        help="path to the per-frame predictions of ours"
    )
        

    parser.add_argument(
        "--data_path",
        type=str,
        help="path to the scannet25k data"
    )
    
    parser.add_argument(
        "--pose_gt_path",
        type=str,
        help="path to the saved gt poses"
    )
    
    parser.add_argument(
        "--split_path",
        type=str,
        help="read the test split",
    )

    parser.add_argument(
        "--mesh_data_path",
        type=str,
        help="path to read the centroid offset of the mesh",
    )
    
    parser.add_argument(
        "--full_annotation_path",
        type=str,
        help="path to read the original annotation",
    )


    opt = parser.parse_args()

    with open(opt.prediction_path, 'r') as f:
        pose_predictions = json.load(f)

    with open(opt.pose_gt_path, 'r') as f:
        pose_gts = json.load(f)

    with open(opt.split_path, 'r') as f:
        splits = f.read().splitlines()
    
    with open(opt.full_annotation_path, 'r') as f:
        full_annos = json.load(f)
    
    eval_alignment(opt, splits, full_annos, pose_predictions, pose_gts)
    
    with open(opt.roca_prediction, 'r') as f:
        roca_predictions = json.load(f)
        
    eval_alignment_roca(opt, splits, roca_predictions, opt.category, full_annos, pose_gts)