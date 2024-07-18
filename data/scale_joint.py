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
from .data_util import crop_resize_by_warp_affine
import torchvision
import kornia as kn


class DiffCADscaleBase(Dataset):
    def __init__(self,
                 category,
                 txt_file,
                 data_root,
                 real_data_root,
                 is_train=False,
                 ):
        self.category = category
        self.data_paths = txt_file  # split file
        self.data_root = data_root
        self.real_data_path = real_data_root
        self.augment = is_train

        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)


        # load pre-computed scales
        with open(os.path.join(self.real_data_path, 'val_zoedepth_scales', '02818832.json'), 'r') as f:
            s2c_02818832_scales = json.load(f)
        with open(os.path.join(self.real_data_path, 'val_zoedepth_scales', '02871439.json'), 'r') as f:
            s2c_02871439_scales = json.load(f)
        with open(os.path.join(self.real_data_path, 'val_zoedepth_scales', '02933112.json'), 'r') as f:
            s2c_02933112_scales = json.load(f)
        with open(os.path.join(self.real_data_path, 'val_zoedepth_scales', '03001627.json'), 'r') as f:
            s2c_03001627_scales = json.load(f)
        with open(os.path.join(self.real_data_path, 'val_zoedepth_scales', '04256520.json'), 'r') as f:
            s2c_04256520_scales = json.load(f)
        with open(os.path.join(self.real_data_path, 'val_zoedepth_scales', '04379243.json'), 'r') as f:
            s2c_04379243_scales = json.load(f)

        # load pre-computed scales
        with open(os.path.join(self.data_root, 'train_zoedepth_scales', '02818832.json'), 'r') as f:
            syn_02818832_scales = json.load(f)
        with open(os.path.join(self.data_root, 'train_zoedepth_scales', '02871439.json'), 'r') as f:
            syn_02871439_scales = json.load(f)
        with open(os.path.join(self.data_root, 'train_zoedepth_scales', '02933112.json'), 'r') as f:
            syn_02933112_scales = json.load(f)
        with open(os.path.join(self.data_root, 'train_zoedepth_scales', '03001627.json'), 'r') as f:
            syn_03001627_scales = json.load(f)
        with open(os.path.join(self.data_root, 'train_zoedepth_scales', '04256520.json'), 'r') as f:
            syn_04256520_scales = json.load(f)
        with open(os.path.join(self.data_root, 'train_zoedepth_scales', '04379243.json'), 'r') as f:
            syn_04379243_scales = json.load(f)

        self.train_scales = {
            '02818832': syn_02818832_scales,
            '02871439': syn_02871439_scales,
            '02933112': syn_02933112_scales,
            '03001627': syn_03001627_scales,
            '04256520': syn_04256520_scales,
            '04379243': syn_04379243_scales,
        }

        self.s2c_scales = {
            '02818832': s2c_02818832_scales,
            '02871439': s2c_02871439_scales,
            '02933112': s2c_02933112_scales,
            '03001627': s2c_03001627_scales,
            '04256520': s2c_04256520_scales,
            '04379243': s2c_04379243_scales,
        }

    def __len__(self):
        return self._length

    def __getitem__(self, i):

        scene_idx, frame_idx = self.image_paths[i].split()
        scene_info = scene_idx + '_' + frame_idx

        dataset_label = self.get_dataset_name(i)

        latent_idx = None
        if dataset_label == '3DF':
            category_id = scene_idx.split('-')[0]
            scene_id = '-'.join(scene_idx.split('-')[1:6])
            latent_idx = '-'.join(scene_idx.split('-')[6:])
        elif dataset_label == 'S2C':
            category_id = scene_idx.split('_')[0]
            scene_id = '_'.join(scene_idx.split('_')[1:])
            frame_idx_s2c, latent_idx, instance_id = frame_idx.split('_')
        else:
            raise ValueError

        assert latent_idx is not None

        if dataset_label == '3DF':
            example = dict(category_id=category_id, scene_idx=scene_id, frame_idx=frame_idx, scene_info=scene_info, latent_idx=latent_idx,
                           dataset_label=dataset_label)
        elif dataset_label == 'S2C':
            example = dict(category_id=category_id, scene_idx=scene_id, frame_idx=frame_idx_s2c, scene_info=scene_info, latent_idx=latent_idx,
                           dataset_label=dataset_label)

        depth_full = self.get_depth(i, dataset_label, ext='.png') / 1000.0
        self.orig_h, self.orig_w = depth_full.shape

        depth_pred = self.get_depth_pred(i, dataset_label) / 1000.0

        rz = self.get_resize(size=(30, 40))

        depth_input = torch.from_numpy(depth_full)[None]
        depth_input = rz(depth_input)

        depth_pred_input = torch.from_numpy(depth_pred)[None]
        depth_pred_input = rz(depth_pred_input)

        if self.augment:
            scale_depth = self.train_scales[category_id]['-'.join(scene_idx.split('-')[1:])+'_'+frame_idx]["best_scale_from_renddepth"]
        else:
            scale_depth = self.s2c_scales[category_id][scene_id + '_'+frame_idx]["best_scale_from_sensordepth"]

        if self.augment:
            # horizontal flip
            aug_flip = self.get_flip(p=0.5)
            depth_pred_input = aug_flip(depth_pred_input)

            # augmentation using the scale
            shift_prob = np.random.uniform(0.0, 1.0)
            if shift_prob > 0.5:
                rand_scale_shift = np.random.uniform(scale_depth-0.2, scale_depth+0.3)

                depth_pred_input = depth_pred_input / rand_scale_shift
                scale_depth = rand_scale_shift

            # rotation prob
            rot_prob = np.random.uniform(0.0, 1.0)
            if rot_prob > 0.7:
                aug_rot = self.get_roration(degrees=45)
                depth_pred_input = aug_rot(depth_pred_input)
            scale_depth = torch.tensor(scale_depth).float() - 1.0
        else:
            scale_depth = torch.tensor(scale_depth).float() - 1.0

        depth_scale_img = torch.ones_like(depth_pred_input) * scale_depth

        example.update(
            depth_input=depth_pred_input,
            depth_scale_img=depth_scale_img,
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
        return torchvision.transforms.GaussianBlur(kernel_size=kernel_size, sigma=(1.0, 2.0))
    
    def get_resize(self, size):
        return torchvision.transforms.Resize(size=size)

    def get_flip(self, p):
        return torchvision.transforms.RandomHorizontalFlip(p=p)

    def get_roration(self, degrees):
        return torchvision.transforms.RandomRotation(degrees=degrees)

    def get_dataset_name(self, idx):
        line = self.image_paths[idx].split()
        scene_id, _ = line

        if '-' in scene_id:
            return '3DF'
        elif '_' in scene_id:
            return 'S2C'
        else:
            raise ValueError


    def get_depth(self, idx, dataset_label, ext='.png'):
        try:
            # Parse the split text
            line = self.image_paths[idx].split()
            scene_id, frame_id = line

            if dataset_label == "3DF":
                category_id = scene_id.split('-')[0]
                file = "{}{}".format(frame_id, ext)
            elif dataset_label == "S2C":
                category_id = scene_id.split('_')[0]
                file = "{}{}".format(frame_id.split('_')[0], ext)

        except Exception as e:
            print(line)
            raise e

        assert dataset_label in ["3DF", "S2C"]

        if dataset_label == "3DF":
            data_root = self.data_root + '/3D-FRONT-RENDER-{}'.format(category_id)
            # gt rendered depth
            depth_fname = os.path.join(data_root, '-'.join(scene_id.split('-')[1:]), 'bop_data/train_pbr/000000/depth', file)
            depth = cv2.imread(depth_fname, -1)

        else:
            # gt rendered depth
            depth_fname = os.path.join(self.real_data_path, "Rendering", '_'.join(scene_id.split('_')[1:]), 'depth', file)

            depth = cv2.imread(depth_fname, -1)  # 360, 480 millimiter uint16

        assert depth.dtype == np.uint16

        return depth

    def get_depth_pred(self, idx, dataset_label):
        try:
            line = self.image_paths[idx].split()
            scene_id, frame_id = line
            if dataset_label == "3DF":
                category_id = scene_id.split('-')[0]
                file = "{}_pred_dmap.npy".format(frame_id)
            elif dataset_label == "S2C":
                category_id = scene_id.split('_')[0]
                file = "{}_pred_dmap.npy".format(frame_id.split('_')[0])

        except Exception as e:
            print(line)
            raise e

        assert dataset_label in ["3DF", "S2C"]

        if dataset_label == "3DF":
            data_root = self.data_root + '/3D-FRONT-RENDER-{}'.format(category_id)
            # gt rendered depth
            depth_fname = os.path.join(data_root, '-'.join(scene_id.split('-')[1:]), 'bop_data/train_pbr/000000/zoedepth', file)

        else:
            # predicted depth
            depth_fname = os.path.join(self.real_data_path, "ZoeDepthPredictions", '_'.join(scene_id.split('_')[1:]), file)

        depth = np.load(depth_fname)
        depth = (depth * 1000).astype(np.uint16)

        assert depth.dtype == np.uint16

        return depth


class DiffCADscaleTrain(DiffCADscaleBase):
    def __init__(self, **kwargs):
        super().__init__(
            category="joint",
            txt_file="path to train split",
            data_root="path to synthetic data with gt scale defined",
            real_data_root="path to real data with gt scale defined",
            is_train=True,
            **kwargs)


class DiffCADscaleValidation(DiffCADscaleBase):
    def __init__(self, **kwargs):
        super().__init__(
            category="joint",
            txt_file="path to val split",
            data_root="path to synthetic data with gt scale defined",
            real_data_root="path to real data with gt scale defined",
            is_train=False,
            **kwargs)
