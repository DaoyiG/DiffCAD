import numpy as np
import torch
import cv2
import open3d as o3d
from scipy.spatial import cKDTree
import itertools
import quaternion
import torch.nn.functional as F

# batch*n
def normalize_vector(v):
    v = F.normalize(v, p=2, dim=1)
    return v


# u, v batch*n
def cross_product(u, v):
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # batch*3

    return out


# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(ortho6d):
    x_raw = ortho6d[:, 0:3]  # batch*3
    y_raw = ortho6d[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix


def mat_to_ortho6d_batch(rots):
    """
    bx3x3
    ---
    bx6
    """
    x = rots[:, :, 0]  # col x
    y = rots[:, :, 1]  # col y
    ortho6d = torch.cat([x, y], 1)  # bx6
    return ortho6d


def mat_to_ortho6d_np(rot):
    """
    3x3
    ---
    (6,)
    """
    x = rot[:3, 0]  # col x
    y = rot[:3, 1]  # col y
    ortho6d = np.concatenate([x, y])  # (6,)
    return ortho6d

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

def correct_pcd_normal_direction(pcd, view_port=np.zeros((3),dtype=float)):
  view_dir = view_port.reshape(-1,3)-np.asarray(pcd.points)   #(N,3)
  view_dir = view_dir/np.linalg.norm(view_dir,axis=1).reshape(-1,1)
  normals = np.asarray(pcd.normals)/(np.linalg.norm(np.asarray(pcd.normals),axis=1)+1e-10).reshape(-1,1)
  dots = (view_dir*normals).sum(axis=1)
  indices = np.where(dots<0)
  normals[indices,:] = -normals[indices,:]
  pcd.normals = o3d.utility.Vector3dVector(normals)
  return pcd

def evaluateModel(OutTransform, SourceHom, TargetHom, PassThreshold):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    InlierIdx = np.where(ResidualVec < PassThreshold)
    nInliers = np.count_nonzero(InlierIdx)
    InlierRatio = nInliers / SourceHom.shape[1]
    return Residual, InlierRatio, InlierIdx[0]

def evaluateModelNoThresh(OutTransform, SourceHom, TargetHom):
    Diff = TargetHom - np.matmul(OutTransform, SourceHom)
    ResidualVec = np.linalg.norm(Diff[:3, :], axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual

def evaluateModelNonHom(source, target, Scales, Rotation, Translation):
    RepTrans = np.tile(Translation, (source.shape[0], 1))
    TransSource = (np.diag(Scales) @ Rotation @ source.transpose() + RepTrans.transpose()).transpose()
    Diff = target - TransSource
    ResidualVec = np.linalg.norm(Diff, axis=0)
    Residual = np.linalg.norm(ResidualVec)
    return Residual

def estimateRestrictedAffineTransform(source: np.array, target: np.array, verbose=False):
    
    SourceHom = np.transpose(np.hstack([source, np.ones([source.shape[0], 1])]))
    TargetHom = np.transpose(np.hstack([target, np.ones([source.shape[0], 1])]))

    RetVal, AffineTrans, Inliers = cv2.estimateAffine3D(source, target)
    # We assume no shear in the affine matrix and decompose into rotation, non-uniform scales, and translation
    Translation = AffineTrans[:3, 3]
    NUScaleRotMat = AffineTrans[:3, :3]
    # NUScaleRotMat should be the matrix SR, where S is a diagonal scale matrix and R is the rotation matrix (equivalently RS)
    # Let us do the SVD of NUScaleRotMat to obtain R1*S*R2 and then R = R1 * R2
    R1, ScalesSorted, R2 = np.linalg.svd(NUScaleRotMat, full_matrices=True)

    if verbose:
        print('-----------------------------------------------------------------------')
    # Now, the scales are sort in ascending order which is painful because we don't know the x, y, z scales
    # Let's figure that out by evaluating all 6 possible permutations of the scales
    ScalePermutations = list(itertools.permutations(ScalesSorted))
    MinResidual = 1e8
    Scales = ScalePermutations[0]
    OutTransform = np.identity(4)
    Rotation = np.identity(3)
    for ScaleCand in ScalePermutations:
        CurrScale = np.asarray(ScaleCand)
        CurrTransform = np.identity(4)
        CurrRotation = (np.diag(1 / CurrScale) @ NUScaleRotMat).transpose()
        CurrTransform[:3, :3] = np.diag(CurrScale) @ CurrRotation
        CurrTransform[:3, 3] = Translation
        # Residual = evaluateModel(CurrTransform, SourceHom, TargetHom)
        Residual = evaluateModelNonHom(source, target, CurrScale,CurrRotation, Translation)
        if verbose:
            # print('CurrTransform:\n', CurrTransform)
            print('CurrScale:', CurrScale)
            print('Residual:', Residual)
            print('AltRes:', evaluateModelNoThresh(CurrTransform, SourceHom, TargetHom))
        if Residual < MinResidual:
            MinResidual = Residual
            Scales = CurrScale
            Rotation = CurrRotation
            OutTransform = CurrTransform

    if verbose:
        print('Best Scale:', Scales)

    if verbose:
        print('Affine Scales:', Scales)
        print('Affine Translation:', Translation)
        print('Affine Rotation:\n', Rotation)
        print('-----------------------------------------------------------------------')

    return Scales, Rotation, Translation, OutTransform

def toOpen3dCloud(points, colors=None, normals=None):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if colors is not None:
        if colors.max() > 1:
            colors = colors/255.0
        cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    if normals is not None:
        cloud.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    return cloud


def to_homo(pts):
    '''
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    '''
    assert len(pts.shape) == 2, f'pts.shape: {pts.shape}'
    homo = np.concatenate((pts, np.ones((pts.shape[0], 1))), axis=-1)
    return homo


def estimateAffine3D(source, target, PassThreshold):
    '''
    @source: (N,3)
    '''
    ret, transform, inliers = cv2.estimateAffine3D(
        source, target, confidence=0.999, ransacThreshold=PassThreshold, )
    tmp = np.eye(4)
    tmp[:3] = transform
    transform = tmp
    inliers = np.where(inliers > 0)[0]
    return transform, inliers


def estimate9DTransform_worker(cur_src, cur_dst, source, target, PassThreshold, use_kdtree_for_eval=False, kdtree_eval_resolution=None, max_scale=np.array([99, 99, 99]), min_scale=np.array([0, 0, 0]), max_dimensions=None):
    bad_return = None, None, None
    transform, inliers = estimateAffine3D(
        source=cur_src, target=cur_dst, PassThreshold=PassThreshold)
    new_transform = transform.copy()
    scales = np.linalg.norm(transform[:3, :3], axis=0)
    if (scales > max_scale).any() or (scales < min_scale).any():
        return bad_return

    R = transform[:3, :3]/scales.reshape(1, 3)
    u, s, vh = np.linalg.svd(R)

    if s.min() < 0.8 or s.max() > 1.2:
        return bad_return

    R = u@vh
    if np.linalg.det(R) < 0:
        return bad_return

    new_transform[:3, :3] = R@np.diag(scales)
    transform = new_transform.copy()

    if max_dimensions is not None:
        cloud_at_canonical = (np.linalg.inv(transform) @
                              to_homo(target).T).T[:, :3]
        dimensions = cloud_at_canonical.max(
            axis=0)-cloud_at_canonical.min(axis=0)
        if (dimensions > max_dimensions).any():
            return bad_return

    src_transformed = (transform@to_homo(source).T).T[:, :3]

    if not use_kdtree_for_eval:
        errs = np.linalg.norm(src_transformed-target, axis=-1)
        ratio = np.sum(errs <= PassThreshold)/len(errs)
        inliers = np.where(errs <= PassThreshold)[0]
    else:
        pcd = toOpen3dCloud(target)
        pcd = pcd.voxel_down_sample(voxel_size=kdtree_eval_resolution)
        kdtree = cKDTree(np.asarray(pcd.points).copy())
        dists1, indices1 = kdtree.query(src_transformed)
        pcd = toOpen3dCloud(src_transformed)
        pcd = pcd.voxel_down_sample(voxel_size=kdtree_eval_resolution)
        kdtree = cKDTree(np.asarray(pcd.points).copy())
        dists2, indices2 = kdtree.query(target)
        errs = np.concatenate((dists1, dists2), axis=0).reshape(-1)
        ratio = np.sum(errs <= PassThreshold)/len(errs)
        inliers = np.where(dists1 <= PassThreshold)[0]

    return ratio, transform, inliers


def estimate9DTransform(source, target, PassThreshold, max_iter=3000, use_kdtree_for_eval=False, kdtree_eval_resolution=None, max_scale=np.array([99, 99, 99]), min_scale=np.array([0, 0, 0]), max_dimensions=None):
    best_transform = None
    best_ratio = 0
    inliers = None

    n_iter = 0
    srcs = []
    dsts = []
    for i in range(max_iter):
        ids = np.random.choice(len(source), size=4, replace=False)
        cur_src = source[ids]
        cur_dst = target[ids]
        srcs.append(cur_src)
        dsts.append(cur_dst)

    outs = []
    for i in range(len(srcs)):
        out = estimate9DTransform_worker(srcs[i], dsts[i], source, target, PassThreshold, use_kdtree_for_eval,
                                         kdtree_eval_resolution=kdtree_eval_resolution, max_scale=max_scale, min_scale=min_scale, max_dimensions=max_dimensions)
        if out[0] is None:
            continue
        outs.append((out))
    if len(outs) == 0:
        return None, None

    ratios = []
    transforms = []
    inlierss = []
    for out in outs:
        ratio, transform, inliers = out
        ratios.append(ratio)
        transforms.append(transform)
        inlierss.append(inliers)

    best_id = np.array(ratios).argmax()
    best_transform = transforms[best_id]
    inliers = inlierss[best_id]
    return best_transform, inliers

def crop_resize_by_warp_affine(img, center, scale, output_w, output_h, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_w, int) and isinstance(output_h, int):
        output_size = (output_w, output_h)
    else:
        raise ValueError("the output size should be integer!")
    trans = get_affine_transform(center, scale, rot, output_w, output_h)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img

def get_affine_transform(center, scale, rot, output_w, output_h, shift=np.array([0, 0], dtype=np.float32), inv=False):
    """
    adapted from CenterNet: https://github.com/xingyizhou/CenterNet/blob/master/src/lib/utils/image.py
    center: ndarray: (cx, cy)
    scale: (w, h)
    rot: angle in deg
    output_size: int or (w, h)
    """
    if isinstance(center, (tuple, list)):
        center = np.array(center, dtype=np.float32)

    if isinstance(scale, (int, float)):
        scale = np.array([scale, scale], dtype=np.float32)

    if isinstance(output_w, (int, float)) and isinstance(output_h,  (int, float)):
        output_size = (output_w, output_h)
    else:
        raise ValueError("the output size should be integer!")

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result
