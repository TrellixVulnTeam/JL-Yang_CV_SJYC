import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from models.model_stages import BiSeNet
import torch

# prepare data
tmp_test_image_path = "./images/top_potsdam_2_13_RGB_25.tif"
tmp_test_image_output = "./images/top_potsdam_2_13_pred_25.tif"
image_root = "./data/potsdam/2_Ortho_RGB_split_1024"
output_root = "./output/potsdam-1024"
split_file_path = "./data/potsdam/split_1024/test.txt"
with open(split_file_path, 'r') as file:
    image_list = file.readlines()
    file.close()
image_list = list(map(lambda x: x.split(',')[0]), image_list)
for i in image_list:
    assert os.path.isfile(os.path.join(image_root, i)
                          ), "{} not exists".format(i)

# load model
net = BiSeNet(backbone='STDCNet1446', n_classes=6, use_boundary_16=False,
              use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_conv_last=False)
net.load_state_dict(torch.load(
    './checkpoints/train_STDC2-Seg-1024/pths/model_maxmIOU100.pth'))
net.eval()
net.cuda()

# preprocess parameters
to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.3392922, 0.36275864, 0.33674471),
                         (0.14043275, 0.1387484, 0.1442598)),
])
# load image
with torch.no_grad():
    for i in image_list:
        image_path = os.path.join(image_root, i)
        image = Image.open(image_path)
        image = to_tensor(image).unsqueeze(0).cuda()
        out = net(image)[0].squeeze().detach()
        out = torch.argmax(out, dim=0).cpu().numpy()
        out = Image.fromarray(np.uint8(out))
        out.save(os.path.join(output_root, i))

