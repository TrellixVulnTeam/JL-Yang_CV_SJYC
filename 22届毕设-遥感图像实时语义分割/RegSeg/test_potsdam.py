import os
import torch
import torchvision as vision
from PIL import Image
from model import RegSeg
from tqdm import tqdm
import numpy as np

split_path = "/root/dataset/potsdam/split_1024/test.txt"
image_root = "/root/dataset/potsdam/2_Ortho_RGB_split_1024"
output_path = "./output/potsdam-1024"
transform = vision.transforms.Compose([
    vision.transforms.ToTensor(),
    vision.transforms.Normalize(
        mean=(0.3392922, 0.36275864, 0.33674471),
        std=(0.14043275, 0.1387484, 0.1442598)
    )
])

with open(split_path, 'r') as file:
    l = file.readlines()
    file.close()
image_label_list = []
for i in l:
    i = i.rstrip('\n')
    i = i.split(',')
    image_label_list.append((i[0], i[1]))
# check
for image_name, label_name in image_label_list:
    assert os.path.isfile(os.path.join(image_root, image_name)), f"{image_name} not exists"
# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
md = RegSeg(name="exp48_decoder26", num_classes=6,
            pretrained="./checkpoints/potsdam1024_exp48_decoder26_500_epochs_run4")
md.eval()
md.to(device)

# test
# test_image_path = "./top_potsdam_2_13_RGB_15.tif"
# test_image = Image.open(test_image_path).convert("RGB")
# test_image = transform(test_image)
# test_image = test_image.unsqueeze(0).cuda()
# output = md(test_image)
# output = torch.argmax(output, dim=1)
# print(output)
# print(output.shape)

# go
with torch.no_grad():
    for image_name, _ in tqdm(image_label_list):
        image_path = os.path.join(image_root, image_name)
        assert os.path.isfile(image_path)
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        image = image.unsqueeze(0).cuda()
        output = md(image)
        output = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        output = Image.fromarray(np.uint8(output))
        output.save(os.path.join(output_path, image_name))
