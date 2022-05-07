#!/usr/bin/python
# -*- encoding: utf-8 -*-


import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import os
from PIL import Image
import numpy as np
import json
from tqdm import tqdm

from transform import *


# Potsdam Dataset Information
# 0: Impervious surfaces (RGB: 255, 255, 255)
# 1: Building (RGB: 0, 0, 255)
# 2: Low vegetation (RGB: 0, 255, 255)
# 3: Tree (RGB: 0, 255, 0)
# 4: Car (RGB: 255, 255, 0)
# 5: Clutter/background (RGB: 255, 0, 0)

class Potsdam(Dataset):
    # change the cropsize to fit potsdam images
    def __init__(self, rootpth, cropsize=(768, 768), mode='train',
                 randomscale=(0.125, 0.25, 0.375, 0.5, 0.675, 0.75, 0.875, 1.0, 1.25, 1.5), *args, **kwargs):
        super(Potsdam, self).__init__()
        assert mode in ('train', 'val', 'test', 'trainval')
        self.mode = mode
        print('self.mode', self.mode)
        self.ignore_lb = 255  # TODO: may be not used

        # read potsdam information, same as the comments at the begining
        # as the formatt: {(255, 255, 255): 0, ...}
        with open('./potsdam_info.json', 'r') as fr:
            labels_info = json.load(fr)
        self.lb_map = {tuple(el['color']): el['trainId'] for el in labels_info}

        # reading the split txt files(train, test, val)
        index_path = os.path.join(rootpth, 'split_1024')
        train_index_path = os.path.join(index_path, 'train_full.txt')
        test_index_path = os.path.join(index_path, 'test.txt')
        val_index_path = os.path.join(index_path, 'val.txt')
        assert os.path.exists(train_index_path)
        assert os.path.exists(test_index_path)
        assert os.path.exists(val_index_path)
        self.train_index, self.test_index, self.val_index = list(), list(), list()
        with open(train_index_path, 'r') as train_index_file:
            with open(test_index_path, 'r') as test_index_file:
                with open(val_index_path, 'r') as val_index_file:
                    train_index = list(map(lambda x: x.rstrip(
                        '\n'), train_index_file.readlines()))
                    test_index = list(map(lambda x: x.rstrip(
                        '\n'), test_index_file.readlines()))
                    val_index = list(map(lambda x: x.rstrip('\n'),
                                         val_index_file.readlines()))
                    self.train_index = list(
                        map(lambda x: tuple(x.split(',')), train_index))
                    self.test_index = list(
                        map(lambda x: tuple(x.split(',')), test_index))
                    self.val_index = list(
                        map(lambda x: tuple(x.split(',')), val_index))
        # check index integraty
        self.image_path = os.path.join(rootpth, '2_Ortho_RGB_split_1024')
        self.label_path = os.path.join(rootpth, '5_Labels_all_split_1024_fine')
        assert os.path.isdir(self.image_path)
        assert os.path.isdir(self.label_path)
        image_list = os.listdir(self.image_path)
        label_list = os.listdir(self.label_path)
        if 'index.txt' in image_list:
            image_list.remove('index.txt')
        if 'index.txt' in label_list:
            label_list.remove('index.txt')
        # assert len(image_list) == len(label_list)
        assert len(self.train_index) + len(self.test_index) == len(
            image_list), "with len of train_index is {}, test_index is {} and image_list is {}".format(
            len(self.train_index), len(self.test_index), len(image_list))
        for image, label in tqdm(self.train_index + self.test_index + self.val_index):
            assert image in image_list, "{} not in image list".format(image)
            assert label in label_list, "{} not in label list".format(label)

        # handle __len__
        if self.mode == "train":
            self.len = len(train_index)
            self.index = self.train_index
        elif self.mode == "test":
            self.len = len(test_index)
            self.index = self.test_index
        else:
            self.len = len(val_index)
            self.index = self.val_index

        # pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.3392922, 0.36275864, 0.33674471), (0.14043275, 0.1387484, 0.1442598)),
        ])
        self.trans_train = Compose([
            # ColorJitter(
            #     brightness=0.5,
            #     contrast=0.5,
            #     saturation=0.5),
            HorizontalFlip(),
            RandomScale(randomscale),
            RandomCrop(cropsize)
        ])

    def __getitem__(self, idx):
        image_name, label_name = self.index[idx]
        image_path = os.path.join(self.image_path, image_name)
        label_path = os.path.join(self.label_path, label_name)
        assert os.path.isfile(image_path)
        assert os.path.isfile(label_path)
        image = Image.open(image_path).convert("RGB")
        label = Image.open(label_path)
        if self.mode == "train" or self.mode == "trainval":
            im_lb = dict(im=image, lb=label)
            im_lb = self.trans_train(im_lb)
            image, label = im_lb['im'], im_lb['lb']
        image = self.to_tensor(image)
        label = np.array(label)
        label = torch.LongTensor(label)
        # label = self.convert_labels(label)
        return image, label

    def __len__(self):
        return self.len

    def convert_labels(self, image):
        image = np.array(image)
        image_shape = image.shape
        new = np.zeros((image_shape[0], image_shape[1]), dtype=np.int64)
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                tmp = tuple(image[i, j])
                if tmp in self.lb_map:
                    new[i, j] = self.lb_map[tmp]
                else:
                    new[i, j] = -1
        return new


if __name__ == "__main__":
    from tqdm import tqdm

    ds = Potsdam("G:\\utf-8' 'Potsdam\\Potsdam", n_classes=6, mode='val')
    uni = []
    loader = DataLoader(ds)
    for image, label in tqdm(loader):
        uni_label = np.unique(label).tolist()
        uni.append(uni_label)
    # print(uni)
    # print(set(uni))
