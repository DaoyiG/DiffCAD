import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import json
from glob import glob
import cv2
import pycocotools.mask as mask_util
import open3d as o3d
import kornia as kn

class DiffCADshapeBase(Dataset):
    def __init__(self,
                 category,
                 txt_file,
                 data_root,
                 scene_config_root,
                 latent_root,
                 real_data_root,
                 mesh_root,
                 mesh_root_aug,
                 is_train=False,
                 ):
        self.category = category
        self.data_paths = txt_file  # split file
        self.data_root = data_root
        self.real_data_path = real_data_root
        self.raw_H, self.raw_W = 256, 256
        self.augment = is_train
        self.scene_config_root = scene_config_root

        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()

        self._length = len(self.image_paths)

        self.mesh_dir_all = mesh_root
        self.s2c_aug_mesh_dir = mesh_root_aug

        self.latent_dir = latent_root

        self.s2c_pose_gt_root = os.path.join(self.real_data_path, 'val_pose_gt/scan2cad_val_{}.json'.format(self.category))

        with open(self.s2c_pose_gt_root, 'r') as f:
            self.s2c_pose_gts = json.load(f)

        self.latent_dir_all = latent_root

        latent_joint = []
        self.latent_joint_ids = []
        latent_joint_dirs = sorted(
            glob(os.path.join(self.latent_dir_all, '*.pt')))
        for latent_joint_dir in latent_joint_dirs:
            self.latent_joint_ids.append(
                latent_joint_dir.split('/')[-1].split('.')[0])
            latent_joint.append(torch.load(
                latent_joint_dir, map_location='cpu').squeeze(0))
        self.latent_joint = torch.stack(latent_joint)

        print("{} loaded total {} latents".format(txt_file, self.latent_joint.shape[0]))

    def __len__(self):
        return self._length

    def __getitem__(self, i):

        if len(self.image_paths[i]) <= 40:
            # this is just object id
            latent_idx = self.image_paths[i]
            latent_gt = self.get_latent(latent_idx, ext='.pt')
            scene_info = 'single_object_{}'.format(latent_idx)
            example = dict(scene_idx=latent_idx, frame_idx=latent_idx, scene_info=scene_info, latent_idx=latent_idx,
                           dataset_label='SYN', mesh_dir=self.mesh_dir_all)
            full_pc_fname = os.path.join(
                self.mesh_dir_all, latent_idx, 'pointcloud.npz')
            points_full = np.load(full_pc_fname)['points'].astype(np.float32)
            pointcloud_gt = torch.Tensor(points_full)

            # crop the point cloud
            xmin = np.random.uniform(-0.45, 0.05)
            x = np.random.uniform(0.25, 0.65)
            y = np.random.uniform(0.25, 0.65)
            z = np.random.uniform(0.25, 0.65)
            roi_min = np.array([xmin, xmin, xmin])
            roi_max = np.array([max(min(xmin + x, 0.4), -0.4), max(min(xmin + y, 0.4), -0.4), max(min(xmin + z, 0.4), -0.4)])
            partial_pc = points_full.copy()
            partial_pc = points_full[
                (points_full[:, 0] >= roi_min[0]) & (points_full[:, 0] <= roi_max[0]) &
                (points_full[:, 1] >= roi_min[1]) & (points_full[:, 1] <= roi_max[1]) &
                (points_full[:, 2] >= roi_min[2]) & (
                    points_full[:, 2] <= roi_max[2])
            ]
            sample_num_pc = 1024
            if partial_pc.shape[0] > 1024:
                indices_pc = np.random.choice(partial_pc.shape[0], size=sample_num_pc, replace=False)
            elif partial_pc.shape[0] > 0:
                indices_pc = np.random.choice(partial_pc.shape[0], size=sample_num_pc, replace=True)
            else:
                indices_pc = np.random.choice(points_full.shape[0], size=sample_num_pc, replace=False)

            if partial_pc.shape[0] > 0:
                gt_nocs = partial_pc[indices_pc, :]
            else:
                gt_nocs = points_full[indices_pc, :]

            add_t = np.random.uniform(-0.05, 0.05, (1, 3))
            add_t = add_t + np.clip(0.01*np.random.randn(gt_nocs.shape[0], 3), -0.05, 0.05)
            gt_nocs = np.add(gt_nocs, add_t)
            gt_nocs = torch.from_numpy(gt_nocs)

            example.update(
                nocs_pc=gt_nocs,  # this is from transformed depth pointcloud
                latent_gt=latent_gt,
                pointcloud_gt=pointcloud_gt,
                latent_all=self.latent_joint,
                latent_all_ids=self.latent_joint_ids,
            )

        else:
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

            mask_full = self.get_obj_mask(i, dataset_label, ext='.png')

            if dataset_label == '3DF':
                init_scale, additionial_scale, centroid_offset, pose, cam_K = self.get_pose_3df(i)
                scale_gt = np.asarray(init_scale * additionial_scale)
                centroid_offset_and_scale = np.eye(4)
                centroid_offset_and_scale[:3, :3] *= scale_gt
                centroid_offset_and_scale[:3, 3] = -centroid_offset
                pose_recalib = pose @ centroid_offset_and_scale

            sample_num_pc = 1024

            if dataset_label == '3DF':
                depth_full = self.get_depth(i, dataset_label, ext='.png')

                mask = torch.from_numpy(mask_full).unsqueeze(0).unsqueeze(0).float()
                kernel = torch.ones(5, 5)
                mask = kn.morphology.erosion(mask, kernel)
                mask_ero = mask.squeeze(0).squeeze(0).detach().cpu().numpy()

                target_depth = depth_full * mask_ero.astype(np.uint8)

                target_depth = o3d.geometry.Image(target_depth)
                intr = o3d.camera.PinholeCameraIntrinsic(480, 360, cam_K)
                pcd_trans = o3d.geometry.PointCloud.create_from_depth_image(target_depth, intr, pose_recalib)

                gt_nocs = np.asarray(pcd_trans.points)

                if gt_nocs.shape[0] > 1024:
                    indices_pc = np.random.choice(gt_nocs.shape[0], size=sample_num_pc, replace=False)
                elif gt_nocs.shape[0] > 0:
                    indices_pc = np.random.choice(gt_nocs.shape[0], size=sample_num_pc, replace=True)
                else:
                    print('No point cloud found for {}'.format(scene_info))

                gt_nocs = gt_nocs[indices_pc, :]
                gt_nocs.clip(-0.55, 0.55)

                # # for vis purpose
                # # if '-' not in latent_idx:
                # pc_dir = os.path.join(self.mesh_dir_all, example['latent_idx'], 'pointcloud.npz')
                # pointcloud_dict_gt = np.load(pc_dir)
                # points_gt = pointcloud_dict_gt['points'].astype(np.float32)
                # indices_pc_cond = np.random.randint(points_gt.shape[0], size=sample_num_pc)
                # pc_gt = points_gt[indices_pc_cond, :]
                # pc_gt_vis = o3d.geometry.PointCloud()
                # pc_gt_vis.points = o3d.utility.Vector3dVector(pc_gt)

                # gt_nocs_vis = o3d.geometry.PointCloud()
                # gt_nocs_vis.points = o3d.utility.Vector3dVector(gt_nocs)

                # pc_gt_vis.paint_uniform_color([1, 0.706, 0])
                # gt_nocs_vis.paint_uniform_color([0, 0.651, 0.929])
                # o3d.visualization.draw_geometries([gt_nocs_vis, pc_gt_vis], window_name=scene_info)

            else:
                # directly load gt nocs from rendered depth: rendered depth + gt pose --> gt nocs
                gt_nocs = np.asarray(o3d.io.read_point_cloud(os.path.join(self.real_data_path, 'NOCS_PLY', self.category, scene_info + '.ply')).points)

                if gt_nocs.shape[0] > 1024:
                    indices_pc = np.random.choice(gt_nocs.shape[0], size=sample_num_pc, replace=False)
                elif gt_nocs.shape[0] > 0:
                    indices_pc = np.random.choice(gt_nocs.shape[0], size=sample_num_pc, replace=True)
                gt_nocs = gt_nocs[indices_pc, :]

            # add data augmentation for gt nocs point cloud
            if self.augment:
                # point shift
                add_t = np.random.uniform(-0.05, 0.05, (1, 3))
                add_t = add_t + np.clip(0.01*np.random.randn(gt_nocs.shape[0], 3), -0.05, 0.05)
                gt_nocs = np.add(gt_nocs, add_t)

                # add chance for pseudo classifier-free guidance - enabling more nocs hallucination?
                prob = torch.randint(low=0, high=10, size=(1,))
                partial_hallucination = 1
                if prob <= partial_hallucination:
                    a = np.random.randint(low=0, high=2, size=(gt_nocs.shape[0], ))
                    random_mask = np.stack((a, a, a), axis=-1)
                    gt_nocs = gt_nocs * random_mask


            gt_nocs = torch.from_numpy(gt_nocs)  # 1024 3

            latent_gt = self.get_latent(latent_idx, ext='.pt')


            # get gt pc
            pc_dir = os.path.join(self.mesh_dir_all, example['latent_idx'], 'pointcloud.npz')
            pointcloud_dict_gt = np.load(pc_dir)
            points_gt = pointcloud_dict_gt['points'].astype(np.float32)
            pointcloud_gt = torch.Tensor(points_gt)

            example.update(
                nocs_pc=gt_nocs,
                latent_gt=latent_gt,
                pointcloud_gt=pointcloud_gt,
                latent_all=self.latent_joint,
                latent_all_ids=self.latent_joint_ids,
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
                file = "{}_pred_dmap.npy".format(frame_id.split('_')[0])
        except Exception as e:
            print(line)
            raise e

        assert dataset_label in ["3DF", "S2C"]

        if dataset_label == "3DF":
            # gt rendered depth
            depth_fname = os.path.join(self.data_root, scene_id, 'bop_data/train_pbr/000000/depth', file)
            depth = cv2.imread(depth_fname, -1)  # 360, 480

            assert depth.dtype == np.uint16

        else:
            # predicted depth from ZoeDepth
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
            # pred mask from odise
            mask_fname_odise = os.path.join(self.data_root, scene_id, 'odise_preds', file)

            if os.path.exists(mask_fname_odise):
                mask_fname = mask_fname_odise
            else:
                # gt mask
                mask_fname = os.path.join(self.data_root, scene_id, 'bop_data/train_pbr/000000/visib_mask', file)

        else:
            mask_fname = os.path.join(self.real_data_path, "ODISEPredictions_NEW", scene_id, self.category, file)

        if os.path.isfile(mask_fname):
            mask = cv2.imread(mask_fname, -1) / 255
            mask = mask.astype(np.uint8)

        else:
            print(line)

        return mask

    def get_latent(self, latent_idx, ext='.pt'):

        file = "{}{}".format(latent_idx, ext)

        latent_fname = os.path.join(self.latent_dir, file)
        latent = torch.load(latent_fname, map_location='cpu')
        latent = latent.squeeze(0)

        return latent


    def get_scale_3df(self, idx, obj_uids):
        line = self.image_paths[idx].split()
        scene_id = '-'.join(line[0].split('-')[:5])
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
        centroid_offset = np.asarray(mesh_info["centroid_offset"])

        # 3DF scale the raw model again to construct the scene
        scene_config_dir = os.path.join(
            self.scene_config_root, line[0] + '.json')
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
        scene_id = '-'.join(line[0].split('-')[:5])
        mesh_id = '-'.join(line[0].split('-')[5:])
        # scene_id = line[0]
        # mesh_id = self.sceneid2objid[scene_id]
        frame_idx = str(int(line[1]))

        pose_fname = os.path.join(self.data_root, line[0], 'bop_data/train_pbr/000000/scene_gt.json')
        with open(pose_fname, 'r') as f:
            pose_gt = json.load(f)

        cam_K = None

        scene_info_fname = os.path.join(self.data_root, line[0], 'bop_data/train_pbr/000000/scene_camera.json')
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

        intr_dir = os.path.join(self.real_data_path,'Images/tasks/scannet_frames_25k', scene_id, 'intrinsics_color.txt')
        with open(intr_dir) as f:
            cam_K = np.array([
                [float(w) for w in l.strip().split()]
                for l in f
            ])
        cam_K = cam_K.reshape(4, 4)[:3, :3]

        return scale_gt, offset, gt_pose_ws, cam_K


class DiffCADshapeTrain(DiffCADshapeBase):
    def __init__(self, **kwargs):
        super().__init__(
            category='category id',
            txt_file="path to train split file",
            data_root="path to synthetic rendering",
            scene_config_root="path to scene config",
            latent_root="path to latent",
            real_data_root="path to real data (Scan2CAD25k)",
            mesh_root="path to mesh",
            mesh_root_aug="path to augmented mesh",
            is_train=True,
            **kwargs)


class DiffCADshapeValidation(DiffCADshapeBase):
    def __init__(self, **kwargs):
        super().__init__(
            category='category id',
            txt_file="path to val split file",
            data_root="path to synthetic rendering",
            scene_config_root="path to scene config",
            latent_root="path to latent",
            real_data_root="path to real data (Scan2CAD25k)",
            mesh_root="path to mesh",
            mesh_root_aug="path to augmented mesh",
            is_train=False,
            **kwargs)