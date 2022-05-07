import os
import numpy as np
import torch
from PIL import Image
from time import time
from .base import BaseDataset


# Potsdam Dataset Information
# 0: Impervious surfaces (RGB: 255, 255, 255)
# 1: Building (RGB: 0, 0, 255)
# 2: Low vegetation (RGB: 0, 255, 255)
# 3: Tree (RGB: 0, 255, 0)
# 4: Car (RGB: 255, 255, 0)
# 5: Clutter/background (RGB: 255, 0, 0)
class Potsdam(BaseDataset):
    CLASSES = [
        "Impervious surfaces",
        "Building", "Low vegetation",
        "Tree", "Car", "Clutter/background"
    ]
    NUM_CLASS = 6
    BASE_DIR = "potsdam"

    def __init__(self, root="../dataset/", split='train', mode=None, transform=None, target_transform=None,
                 **kwargs):
        super(Potsdam, self).__init__(root, split, mode, transform, target_transform, kwargs)
        time_s = time()
        data_path = os.path.join(root, self.BASE_DIR)
        image_path = os.path.join(data_path, "2_Ortho_RGB_split_1024")
        label_path = os.path.join(data_path, "5_Labels_all_split_1024_fine")
        # check if path exists
        assert os.path.isdir(image_path), "image path not exists in {}".format(image_path)
        assert os.path.isdir(label_path), "label path not exists in {}".format(label_path)
        self.image_path = image_path
        self.label_path = label_path
        # split files
        split_root = os.path.join(data_path, "split_1024")
        if self.split == "train":
            split_path = os.path.join(split_root, "train.txt")
        elif self.split == "val":
            split_path = os.path.join(split_root, "val.txt")
        elif self.split == "test":
            split_path = os.path.join(split_root, "test.txt")
        else:
            raise RuntimeError("unknown split: {}".format(self.split))
        self.images = []
        self.masks = []
        # check split file
        assert os.path.isfile(split_path)
        with open(split_path, 'r') as file:
            images = file.readlines()
            file.close()
        for i in images:
            i = i.rstrip('\n')
            i = i.split(',')
            image = i[0]
            label = i[1]
            image_p = os.path.join(image_path, image)
            label_p = os.path.join(label_path, label)
            assert os.path.isfile(image_p), "{} not a file".format(image_p)
            assert os.path.isfile(label_p), "{} not a file".format(label_p)
            self.images.append(image_p)
            self.masks.append(label_p)
        if self.mode != "test":
            assert len(self.images) == len(self.masks)
        time_e = time()
        print("Init Dataset time: {}s".format(time_e - time_s))

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = self.masks[index]
        image = Image.open(image_path).convert("RGB")
        if self.mode == "test":
            if self.transform is not None:
                image = self.transform(image)
            return image, os.path.basename(image_path)
        label = Image.open(label_path)
        if self.mode == "train":
            image, label = self._sync_transform(image, label)
        elif self.mode == "val":
            image, label = self._val_sync_transform(image, label)
        else:
            assert self.mode == "testval"
            label = self._mask_transform(label)
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label

    def _mask_transform(self, mask):
        label = np.array(mask)
        return torch.from_numpy(label).long()

    def __len__(self):
        return len(self.images)

    @property
    def pred_offset(self):
        return 0
