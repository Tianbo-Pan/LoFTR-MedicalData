import imp
from os import path as osp
from typing import Dict
from unicodedata import name
import cv2
import os
import numpy as np
import torch
import torch.utils as utils
import pdb
from numpy.linalg import inv
from src.utils.dataset import (
    read_scannet_gray,
    read_scannet_depth,
    read_scannet_pose,
    read_scannet_intrinsic,
    read_homo
)


class HomoDataset(utils.data.Dataset):
    def __init__(self,
                 root_dir,
                 npz_path,
                 intrinsic_path,
                 mode='train',
                 min_overlap_score=0.4,
                 augment_fn=None,
                 homo_dir=None,
                 **kwargs):
        """Manage one scene of ScanNet Dataset.
        Args:
            root_dir (str): ScanNet root directory that contains scene folders.
            npz_path (str): {scene_id}.npz path. This contains image pair information of a scene.
            intrinsic_path (str): path to depth-camera intrinsic file.
            mode (str): options are ['train', 'val', 'test'].
            augment_fn (callable, optional): augments images with pre-defined visual effects.
            pose_dir (str): ScanNet root directory that contains all poses.
                (we use a separate (optional) pose_dir since we store images and poses separately.)
        """
        super().__init__()
        self.root_dir = root_dir
        self.homo_dir = pose_dir if pose_dir is not None else root_dir
        self.mode = mode

        # prepare data_names, intrinsics and extrinsics(T)
        with np.load(npz_path) as data:
            self.data_names = data['name']
            if 'score' in data.keys() and mode not in ['val' or 'test']:
                kept_mask = data['score'] > min_overlap_score
                self.data_names = self.data_names[kept_mask]
        self.intrinsics = dict(np.load(intrinsic_path))

        # for training LoFTR
        self.augment_fn = augment_fn if mode == 'train' else None

    def __len__(self):
        return len(self.data_names)

    def __read_gt_homo__(self, gt_homo):
        pth = osp.join(os.readlink(osp.join(self.root_dir, scene_name)), f'{gt_homo}.txt')
        return read_homo(pth)

    def __getitem__(self, idx):
        data_name = self.data_names[idx]
        img, img_warped, gt_homo = data_name
        scene_name = f'scene{scene_name:04d}_{scene_sub_name:02d}'

        # read the grayscale image which will be resized to (1, 480, 640)
        img_name0 = osp.join(os.readlink(osp.join(self.root_dir, scene_name)), f'{img}.png')
        img_name1 = osp.join(os.readlink(osp.join(self.root_dir, scene_name)), f'{img_warped}.jpg')
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', img_name0)
        img = cv2.imread(img_name0)
        # TODO: Support augmentation & handle seeds for each worker correctly.
        image0 = read_scannet_gray(img_name0, resize=(640, 480), augment_fn=None)
        #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        image1 = read_scannet_gray(img_name1, resize=(640, 480), augment_fn=None)
        #    augment_fn=np.random.choice([self.augment_fn, None], p=[0.5, 0.5]))
        homo = self.__read_gt_homo__(gt_homo)

        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'gt_homo': homo,
            'dataset_name': 'Homo',
            'scene_id': scene_name,
            'pair_id': idx,
            'pair_names': (osp.join(scene_name, f'{img}.jpg'),
                           osp.join(scene_name, f'{img_warped}.jpg'))
        }

        return data
