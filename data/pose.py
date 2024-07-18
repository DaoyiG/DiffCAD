import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import json
import cv2
import pycocotools.mask as mask_util
import open3d as o3d
import kornia as kn


class DiffCADposeBase(Dataset):
    def __init__(self,
                 category,
                 txt_file,
                 data_root,
                 scene_config_root,
                 real_data_root,
                 mesh_root,
                 mesh_root_aug,
                 is_train=False,
                 ):
        self.category = category
        self.data_paths = txt_file  # split file
        self.data_root = data_root
        self.real_data_path = real_data_root
        self.augment = is_train
        self.scene_config_root = scene_config_root

        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)

        self.mesh_dir_all = mesh_root
        self.s2c_aug_mesh_dir = mesh_root_aug
        self.s2c_pose_gt_root = os.path.join(self.real_data_path, 'val_pose_gt/scan2cad_val_{}.json'.format(self.category))
        with open(self.s2c_pose_gt_root, 'r') as f:
            self.s2c_pose_gts = json.load(f)

    def __len__(self):
        return self._length

    def __getitem__(self, i):

        scene_idx, frame_idx = self.image_paths[i].split()
        scene_info = scene_idx + '_' + frame_idx

        dataset_label = self.get_dataset_name(i)

        latent_idx = None
        if dataset_label == '3DF':
            scene_id = '-'.join(scene_idx.split('-')[:5])
            latent_idx = '-'.join(scene_idx.split('-')[5:])
        elif dataset_label == 'S2C':
            frame_idx_s2c, latent_idx, instance_id = frame_idx.split('_')
        else:
            raise ValueError

        assert latent_idx is not None

        if dataset_label == '3DF':
            example = dict(scene_idx=scene_id, frame_idx=frame_idx, scene_info=scene_info, latent_idx=latent_idx,
                           dataset_label=dataset_label, mesh_dir=self.mesh_dir_all)
        elif dataset_label == 'S2C':
            example = dict(scene_idx=scene_idx, frame_idx=frame_idx_s2c, scene_info=scene_info, latent_idx=latent_idx,
                           dataset_label=dataset_label, mesh_dir=self.mesh_dir_all)

        depth_full = self.get_depth(i, dataset_label, ext='.png')

        depth_pred = self.get_depth_pred(i, dataset_label)
        if dataset_label == '3DF':
            init_scale, additionial_scale, centroid_offset, pose, cam_K = self.get_pose_3df(i)
            scale_gt = np.asarray(init_scale * additionial_scale)
            centroid_offset_and_scale = np.eye(4)
            centroid_offset_and_scale[:3, :3] *= scale_gt
            centroid_offset_and_scale[:3, 3] = -centroid_offset
            pose_recalib = pose @ centroid_offset_and_scale

        elif dataset_label == 'S2C':
            scale_gt, offset, pose, cam_K = self.get_pose_s2c(i)
            pose_recalib = pose @ offset

        mask_full = self.get_obj_mask(i, dataset_label, ext='.png')
        mask = torch.from_numpy(mask_full).unsqueeze(0).unsqueeze(0).float()

        kernel = torch.ones(3, 3)
        mask = kn.morphology.erosion(mask, kernel)
        mask_ero = mask.squeeze(0).squeeze(0).detach().cpu().numpy()

        if mask_ero.sum() <= 0.05 * 360 * 480:
            print("mask portion is less than 5% ", scene_info)

        target_depth = depth_full * mask_ero.astype(np.uint8)
        pred_target_depth = depth_pred * mask_ero.astype(np.uint8)

        pred_target_depth = pred_target_depth.astype(np.uint16)

        target_depth = o3d.geometry.Image(target_depth)
        pred_target_depth = o3d.geometry.Image(pred_target_depth)

        intr = o3d.camera.PinholeCameraIntrinsic(480, 360, cam_K)
        pcd_orig = o3d.geometry.PointCloud.create_from_depth_image(target_depth, intr)
        pcd_orig_pred = o3d.geometry.PointCloud.create_from_depth_image(pred_target_depth, intr)
        pcd_trans = o3d.geometry.PointCloud.create_from_depth_image(target_depth, intr, pose_recalib)

        # # vis purpose
        # if self.augment:
        #     pcd_orig.paint_uniform_color([1, 0.706, 0])
        #     pcd_orig_pred.paint_uniform_color([0, 0.651, 0.929])
        #     o3d.visualization.draw_geometries([pcd_orig, pcd_orig_pred], window_name=scene_info)

        ori_cloud = np.asarray(pcd_orig.points).copy()
        ori_cloud_pred = np.asarray(pcd_orig_pred.points).copy()

        ori_cloud_normalized = (ori_cloud - ori_cloud.min(axis=0)) / \
            ((ori_cloud.max(axis=0) - ori_cloud.min(axis=0)).max() + 1e-15)
        ori_cloud_pred_normalized = (ori_cloud_pred - ori_cloud_pred.min(axis=0)) / \
            ((ori_cloud_pred.max(axis=0) - ori_cloud_pred.min(axis=0)).max() + 1e-15)

        gt_nocs = np.asarray(pcd_trans.points)

        sample_num_pc = 1024
        if ori_cloud.shape[0] > 1024:
            indices_pc = np.random.choice(
                ori_cloud.shape[0], size=sample_num_pc, replace=False)
        else:
            indices_pc = np.random.choice(
                ori_cloud.shape[0], size=sample_num_pc, replace=True)

        ori_cloud = ori_cloud[indices_pc, :]
        ori_cloud_pred = ori_cloud_pred[indices_pc, :]
        ori_cloud_normalized = ori_cloud_normalized[indices_pc, :]
        ori_cloud_pred_normalized = ori_cloud_pred_normalized[indices_pc, :]
        gt_nocs = gt_nocs[indices_pc, :]

        # # # vis purpose
        # gt_nocs_vis = o3d.geometry.PointCloud()
        # gt_nocs_vis.points = o3d.utility.Vector3dVector(gt_nocs)
        # gt_nocs_vis.paint_uniform_color([1, 0.706, 0])  # yellow

        # normalized_mesh = trimesh.load(os.path.join(self.mesh_dir_all, latent_idx, 'model_normalized.obj'))
        # pointcloud_frommesh, _ = trimesh.sample.sample_surface(normalized_mesh, 5000)
        # pcd_mesh = o3d.geometry.PointCloud()
        # pcd_mesh.points = o3d.utility.Vector3dVector(pointcloud_frommesh)
        # pcd_mesh.paint_uniform_color([0, 0.651, 0.929])  # blue
        # o3d.visualization.draw_geometries([pcd_mesh, gt_nocs_vis], window_name=scene_info)


        if self.augment:
            # point shift
            add_t = np.random.uniform(-0.01, 0.01, (1, 3))
            add_t = add_t + np.clip(0.002*np.random.randn(ori_cloud.shape[0], 3), -0.01, 0.01)
            ori_cloud = np.add(ori_cloud, add_t)
            ori_cloud_pred = np.add(ori_cloud_pred, add_t)
            ori_cloud_normalized = np.add(ori_cloud_normalized, add_t)
            ori_cloud_pred_normalized = np.add(ori_cloud_pred_normalized, add_t)
            
        # this is the gt pose without processing with centroid offset and final scale
        pose = torch.from_numpy(pose)
        pose_recalib = torch.from_numpy(pose_recalib)

        ori_cloud = torch.from_numpy(ori_cloud)
        ori_cloud_pred = torch.from_numpy(ori_cloud_pred)

        ori_cloud_normalized = torch.from_numpy(ori_cloud_normalized)
        ori_cloud_pred_normalized = torch.from_numpy(ori_cloud_pred_normalized)
        gt_nocs = torch.from_numpy(gt_nocs)

        # ori_cloud_stacked = torch.cat([ori_cloud_pred, ori_cloud_pred_normalized], dim=1)

        example.update(
            depth_input=ori_cloud,
            # depth_input=ori_cloud_pred,
            # depth_input=ori_cloud_normalized,
            nocs_gt=gt_nocs,

        )

        return example

    def generate_occ_mask(self, orig_mask):
        h, w = orig_mask.shape
        h_start = np.random.uniform(0.1, 0.5)
        w_start = np.random.uniform(0.1, 0.5)
        occ_mask = np.zeros_like(orig_mask)
        occ_mask[int(h_start*h):, int(w_start*w):] = 1.0
        return occ_mask

    def get_gaussian_blur(self):
        kernel_size = 5
        transform = transforms.RandomApply(
            torch.nn.ModuleList(
                [transforms.GaussianBlur(kernel_size=kernel_size, sigma=(1.0, 2.0)),]),
            p=0.7)
        return transform

    def get_dataset_name(self, idx):
        # naive way of identifying the dataset name by checking whether the scene id has '-' or '_'
        line = self.image_paths[idx].split()
        scene_id, _ = line

        if '-' in scene_id:
            return '3DF'
        elif '_' in scene_id:
            return 'S2C'
        else:
            raise ValueError

    def annotate_roi(self, mask):
        rle = mask_util.encode(np.array(
            mask[:, :, None], order='F', dtype='uint8'
        ))[0]
        bbox = mask_util.toBbox(rle)  # x1, y1, h, w
        area = mask_util.area(rle)
        rle['counts'] = rle['counts'].decode('utf-8')
        return rle, bbox.tolist(), area.tolist()

    def xywh_to_xyxy(self, xywh):
        """Convert [x1 y1 w h] box format to [x1 y1 x2 y2] format.
        https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/boxes.py
        """
        if isinstance(xywh, (list, tuple)):
            # Single box given as a list of coordinates
            assert len(xywh) == 4
            x1, y1 = xywh[0], xywh[1]
            # oringal version x2 = x1 + np.maximum(0., xywh[2] - 1.)
            x2 = x1 + np.maximum(0., xywh[2])
            # oringal version y2 = y1 + np.maximum(0., xywh[3] - 1.)
            y2 = y1 + np.maximum(0., xywh[3])
            return (x1, y1, x2, y2)
        elif isinstance(xywh, np.ndarray):
            # Multiple boxes given as a 2D ndarray
            return np.hstack(
                (xywh[:, 0:2], xywh[:, 0:2] + np.maximum(0, xywh[:, 2:4] - 1))
            )
        else:
            raise TypeError(
                'Argument xywh must be a list, tuple, or numpy array.')

    def get_depth(self, idx, dataset_label, ext='.png'):
        try:
            # Parse the split text
            line = self.image_paths[idx].split()
            scene_id, frame_id = line

            if dataset_label == "3DF":
                file = "{}{}".format(frame_id, ext)
            elif dataset_label == "S2C":
                file = "{}{}".format(frame_id.split('_')[0], ext)

        except Exception as e:
            print(line)
            raise e

        assert dataset_label in ["3DF", "S2C"]

        if dataset_label == "3DF":
            # gt rendered depth
            depth_fname = os.path.join(self.data_root, scene_id, 'bop_data/train_pbr/000000/depth', file)
            depth = cv2.imread(depth_fname, -1)  # 360, 480

        else:
            # gt rendered depth
            depth_fname = os.path.join(self.real_data_path, "Rendering", scene_id, 'depth', file)

            depth = cv2.imread(depth_fname, -1)  # 360, 480 millimiter uint16

        assert depth.dtype == np.uint16

        return depth

    def get_depth_pred(self, idx, dataset_label):
        try:
            line = self.image_paths[idx].split()
            scene_id, frame_id = line
            if dataset_label == "3DF":
                file = "{}_pred_dmap.npy".format(frame_id)
            elif dataset_label == "S2C":
                file = "{}_pred_dmap.npy".format(frame_id.split('_')[0])

        except Exception as e:
            print(line)
            raise e

        assert dataset_label in ["3DF", "S2C"]

        if dataset_label == "3DF":
            # pred depth
            depth_fname = os.path.join(self.data_root, scene_id, 'bop_data/train_pbr/000000/zoedepth', file)

        else:
            # pred depth
            depth_fname = os.path.join(self.real_data_path, "ZoeDepthPredictions", scene_id, file)

        depth = np.load(depth_fname)
        depth = (depth * 1000).astype(np.uint16)

        assert depth.dtype == np.uint16

        return depth

    def get_obj_mask(self, idx, dataset_label, ext='.png'):
        try:
            # Parse the split text
            line = self.image_paths[idx].split()
            scene_id, frame_id = line

            file = "{}{}".format(frame_id, ext)

        except Exception as e:
            print(line)
            raise e

        assert dataset_label in ["3DF", "S2C"]
        if dataset_label == "3DF":
            # gt mask
            # mask_fname = os.path.join(self.data_root, scene_id, 'bop_data/train_pbr/000000/visib_mask', file)

            # pred mask from odise
            mask_fname_odsie = os.path.join(self.data_root, scene_id, 'odise_preds', file)
            if os.path.exists(mask_fname_odsie):
                mask_fname = mask_fname_odsie
            else:
                mask_fname = os.path.join(self.data_root, scene_id, 'bop_data/train_pbr/000000/visib_mask', file)
        else:
            # gt mask from rendering
            # mask_fname = os.path.join(self.real_data_path, "Rendering", scene_id, 'mask_sofa', file)

            # pred mask from odise
            mask_fname = os.path.join(self.real_data_path, "ODISEPredictions_NEW", scene_id, self.category, file)

        mask = cv2.imread(mask_fname, -1) / 255
        mask = mask.astype(np.uint8)

        return mask

    def bbox_augmentation(self, bbox_xyxy, dzi=False):
        if dzi:
            bbox_center, scale = self.aug_bbox(bbox_xyxy)  # DZI

        else:
            x1, y1, x2, y2 = bbox_xyxy
            bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])

            bw = max(bbox_xyxy[2] - bbox_xyxy[0], 1)
            bh = max(bbox_xyxy[3] - bbox_xyxy[1], 1)

            scale = max(bh, bw) * 1.5
            scale = min(scale, self.orig_h, self.orig_w) * 1.0

        return bbox_center, scale

    # DZI: aug bbox during training
    def aug_bbox(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        bh = y2 - y1
        bw = x2 - x1

        scale_ratio = 1 + 0.25 * \
            (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
        shift_ratio = 0.25 * \
            (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
        bbox_center = np.array(
            [cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
        scale = max(y2 - y1, x2 - x1) * scale_ratio * 1.5
        scale = min(scale, max(self.orig_h, self.orig_w)) * 1.0
        return bbox_center, scale

    def get_scale_3df(self, idx, obj_uids):
        line = self.image_paths[idx].split()
        scene_id = line[0]
        mesh_id = '-'.join(line[0].split('-')[5:])

        # this is the scale between the 'raw_model' and the processed normalized model
        if '-' in mesh_id:
            initial_scale_dir = os.path.join(self.mesh_dir_all, mesh_id, 'mesh_info.json')
        else:
            # then this is actually a shapenet object but scaled by its NN 3DF object scale
            initial_scale_dir = os.path.join(self.s2c_aug_mesh_dir, mesh_id, 'mesh_info.json')

        with open(initial_scale_dir, 'r') as f:
            mesh_info = json.load(f)
        initial_scale = np.asarray(mesh_info['scale'])
        centroid_offset = torch.from_numpy( np.asarray(mesh_info["centroid_offset"]))

        # 3DF scale the raw model again to construct the scene
        scene_config_dir = os.path.join(self.scene_config_root, scene_id + '.json')
        with open(scene_config_dir, 'r') as f:
            scene_config = json.load(f)

        for room_id, room in enumerate(scene_config["scene"]["room"]):
            for child in room["children"]:
                if "furniture" in child["instanceid"]:
                    if obj_uids[0] == child["ref"]:
                        additional_scale = np.asarray(child["scale"])

        return initial_scale, additional_scale, centroid_offset

    def get_pose_3df(self, idx):
        line = self.image_paths[idx].split()
        scene_id = line[0]
        frame_idx = str(int(line[1]))

        mesh_id = '-'.join(line[0].split('-')[5:])

        pose_fname = os.path.join(self.data_root, scene_id, 'bop_data/train_pbr/000000/scene_gt.json')
        with open(pose_fname, 'r') as f:
            pose_gt = json.load(f)

        cam_K = None

        scene_info_fname = os.path.join(self.data_root, scene_id, 'bop_data/train_pbr/000000/scene_camera.json')
        with open(scene_info_fname, 'r') as f:
            scene_info_gt = json.load(f)

        cam_K = scene_info_gt[frame_idx]["cam_K"]
        cam_K = np.asarray(cam_K, dtype=np.float32).reshape(3, 3)

        anno = pose_gt[frame_idx]  # all objects of a frame

        rot = None
        tra = None
        obj_uid = None

        rots = []
        tras = []
        obj_uids = []
        for obj_pose in anno:
            if obj_pose['obj_jid'] == mesh_id:
                obj_uid = obj_pose['obj_uid']
                rot = np.asarray(obj_pose['cam_R_m2c']).reshape(3, 3)
                tra = np.asarray(obj_pose['cam_t_m2c']).reshape(3,)  # in millimeters
                rots.append(rot)
                tras.append(tra)
                obj_uids.append(obj_uid)

        assert len(rots) is not None, print("does not find the pose of the target object")
        assert len(tras) is not None, print("does not find the pose of the target object")
        assert len(obj_uids) is not None, print("does not find the pose of the target object")

        rot_y = np.array([[np.cos(np.pi), 0, np.sin(np.pi)], [0, 1, 0], [-np.sin(np.pi), 0, np.cos(np.pi)]])

        pose = np.empty((4, 4))
        # the original bop pose gt is defined on raw_model, which is y-180 different than the processed normalized model
        pose[:3, :3] = rots[0] @ rot_y
        pose[:3, 3] = tras[0] / 1000  # in meter
        pose[3] = [0, 0, 0, 1]

        init_scale, additionial_scale, centroid_offset = self.get_scale_3df(idx, obj_uids)

        return init_scale, additionial_scale, centroid_offset, pose, cam_K

    def get_pose_s2c(self, idx):

        line = self.image_paths[idx].split()
        scene_id, frame_idx = line
        scene_info = scene_id + '_' + frame_idx

        gt_pose_ws = self.s2c_pose_gts[scene_info]
        gt_pose_ws = np.asarray(gt_pose_ws).reshape(4, 4)

        scale_gt = np.linalg.norm(gt_pose_ws[:3, :3], axis=0)
        scale_gt = np.asarray(scale_gt)

        with open(os.path.join(self.mesh_dir_all, frame_idx.split('_')[-2], 'mesh_info.json'), 'r') as f:
            mesh_info = json.load(f)

        centroid_offset = mesh_info['centroid_offset']
        centroid_offset = np.asarray(centroid_offset)
        offset = np.eye(4)
        offset[:3, 3] = -centroid_offset

        intr_dir = os.path.join(self.real_data_path, 'Images/tasks/scannet_frames_25k', scene_id, 'intrinsics_color.txt')
        with open(intr_dir) as f:
            cam_K = np.array([
                [float(w) for w in l.strip().split()]
                for l in f
            ])
        cam_K = cam_K.reshape(4, 4)[:3, :3]

        return scale_gt, offset, gt_pose_ws, cam_K

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long)
        distance = torch.ones(B, N) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long)
        batch_indices = torch.arange(B, dtype=torch.long)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1).float()
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids



class DiffCADposeTrain(DiffCADposeBase):
    def __init__(self, **kwargs):
        super().__init__(
            category="category id",
            txt_file="path to train split",
            data_root="path to synthetic rendering",
            scene_config_root="path to scene config",
            real_data_root="path to real data (Scan2CAD25k)",
            mesh_root="path to mesh",
            mesh_root_aug="path to augmented mesh",
            is_train=True,
            **kwargs)


class DiffCADposeValidation(DiffCADposeBase):
    def __init__(self, **kwargs):
        super().__init__(
            category="category id",
            txt_file="path to train split",
            data_root="path to synthetic rendering",
            scene_config_root="path to scene config",
            real_data_root="path to real data (Scan2CAD25k)",
            mesh_root="path to mesh",
            mesh_root_aug="path to augmented mesh",
            is_train=False,
            **kwargs)
