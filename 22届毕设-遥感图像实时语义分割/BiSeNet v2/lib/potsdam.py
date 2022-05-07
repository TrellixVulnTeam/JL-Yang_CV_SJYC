import os
from torch.utils.data import DataLoader, Dataset
import torch
import cv2
import numpy as np
import lib.transform_cv2 as T
import time
from PIL import Image


# Potsdam Dataset Information
# 0: Impervious surfaces (RGB: 255, 255, 255)
# 1: Building (RGB: 0, 0, 255)
# 2: Low vegetation (RGB: 0, 255, 255)
# 3: Tree (RGB: 0, 255, 0)
# 4: Car (RGB: 255, 255, 0)
# 5: Clutter/background (RGB: 255, 0, 0)
class Potsdam(Dataset):
    def __init__(self, data_root, split_file_path, trans_func, mode='train'):
        super(Potsdam, self).__init__()
        time_start_init = time.time()
        # initializing parameters
        assert mode in ('train', 'test', 'val'), "mode wrong with: {}".format(mode)
        self.mode = mode
        self.lb_map = {
            (255, 255, 255): 0,
            (0, 0, 255): 1,
            (0, 255, 255): 2,
            (0, 255, 0): 3,
            (255, 255, 0): 4,
            (255, 0, 0): 5
        }
        self.trans_func = trans_func  # trans_func is in get_dataloader.py
        self.to_tensor = T.ToTensor(
            mean=(0.3392992, 0.36275864, 0.33674471),
            std=(0.14043275, 0.1387484, 0.1442598)
        )
        self.n_cats = 6
        self.lb_ignore = 255
        # reading the corresponding split file (train, test, val)
        assert os.path.isfile(split_file_path), "split file path wrong: {}".format(split_file_path)
        with open(split_file_path, 'r') as file:
            split_list = file.readlines()
        self.image_label_list = []
        image_root = os.path.join(data_root, '2_Ortho_RGB_split_1024')
        label_root = os.path.join(data_root, '5_Labels_all_split_1024_fine')
        for i in split_list:
            i = i.rstrip('\n')
            assert '\n' not in i, "\\n in {}".format(i)
            i_list = i.split(',')  # i_list = [image name, label name]
            image_path = os.path.join(image_root, i_list[0])
            label_path = os.path.join(label_root, i_list[1])
            assert os.path.exists(image_path), "image path not exists: {}".format(image_path)
            assert os.path.exists(label_path), "label path not exists: {}".format(label_path)
            self.image_label_list.append((image_path, label_path))
        # checking
        # now image_label_list contains the path of all images and labels corresponding to the specific split file
        assert len(self.image_label_list) == len(split_list)
        time_end_init = time.time()
        print("Dataset init time: {}s".format(time_end_init - time_start_init))

    def convert_labels(self, image):
        image_shape = image.shape
        new = np.zeros((image_shape[0], image_shape[1]), dtype=np.int64)
        for i in range(image_shape[0]):
            for j in range(image_shape[1]):
                tmp = tuple(image[i, j])
                if tmp in self.lb_map:
                    new[i, j] = self.lb_map[tmp]
                else:
                    new[i, j] = 255
        return new

    def __getitem__(self, index):
        image_path = self.image_label_list[index][0]
        label_path = self.image_label_list[index][1]
        # check again
        os.path.isfile(image_path), "image path not exists: {}".format(image_path)
        os.path.isfile(label_path), "label path not exists: {}".format(label_path)
        # reading image and label
        # image = cv2.imread(image_path)[:, :, ::-1]  # transform the channel from "BGR" to "RGB"
        # label = cv2.imread(label_path)[:, :, ::-1]
        image = np.array(Image.open(image_path))
        label = np.array(Image.open(label_path))
        # transform the label
        # label = self.convert_labels(label)
        im_lb = dict(im=image, lb=label)
        if self.trans_func is not None:
            im_lb = self.trans_func(im_lb)
        im_lb = self.to_tensor(im_lb)
        image, label = im_lb['im'], im_lb['lb']
        return image.detach(), label.detach()

    def __len__(self):
        return len(self.image_label_list)


if __name__ == '__main__':
    from tqdm import tqdm
    from torch.utils.data import DataLoader

    dataset = Potsdam('/root/bisenet-1024/datasets/potsdam', '/root/bisenet-1024/datasets/potsdam/split_1024/val.txt',
                      trans_func=None, mode='val')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True)
    for imgs, label in dataloader:
        print(len(imgs))
        for el in imgs:
            print(el.size())
        break
    # for index, (image, label) in enumerate(tqdm(dataloader)):
    #     assert image.shape == (3, 1024, 1024), "image shape wrong {}, {}".format(index, image.shape)
    #     assert label.shape == (1024, 1024), "label shape wrong {}, {}".format(index, label.shape)
    print("DONE")
