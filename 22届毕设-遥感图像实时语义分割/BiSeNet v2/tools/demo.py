import sys

sys.path.insert(0, '.')
import argparse
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

import lib.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file

torch.set_grad_enabled(False)

# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2.py', )
parse.add_argument('--weight-path', type=str, default='./res/model_final.pth', )
parse.add_argument('--img-path', dest='img_path', type=str, default='./example.png', )
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)
output_root = "output/potsdam-1024"
input_root = "datasets/potsdam/2_Ortho_RGB_split_1024"
test_index_path = "datasets/potsdam/split_1024/test.txt"
if not os.path.exists(output_root):
    os.mkdir(output_root)
assert os.path.isdir(input_root)
assert os.path.isfile(test_index_path)
with open(test_index_path, 'r') as test_index:
    image_list = test_index.readlines()
image_list = list(map(lambda x: x.split(',')[0], image_list))
for i in image_list:
    assert os.path.isfile(os.path.join(input_root, i))

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='pred')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()
net.cuda()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3392922, 0.36275864, 0.33674471),  # city, rgb
    std=(0.14043275, 0.1387484, 0.1442598),
)
for i in tqdm(image_list):
    # preprocessing
    im = np.array(Image.open(os.path.join(input_root, i)))
    im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0).cuda()
    # inference
    out = net(im).squeeze().detach().cpu().numpy()
    out = Image.fromarray(np.uint8(out))
    out.save(os.path.join(output_root, i))
