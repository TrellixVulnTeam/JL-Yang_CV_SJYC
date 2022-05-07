import numpy as np
from tqdm import tqdm
from models.model_stages import BiSeNet
import torch
from time import time

# load model
net = BiSeNet(backbone='STDCNet1446', n_classes=6, use_boundary_16=False,
              use_boundary_2=False, use_boundary_4=False, use_boundary_8=True, use_conv_last=False)
net.load_state_dict(torch.load(
    './checkpoints/train_STDC2-Seg-1024/pths/model_maxmIOU100.pth'))
net.eval()
net.cuda()

x = torch.tensor(np.random.random((1, 3, 1024, 1024)) - 0.5, dtype=torch.float).cuda()
n_it = 2000
N_it = 15
results = []

# load image
with torch.no_grad():
    for _ in tqdm(range(N_it)):
        st = time()
        for _ in range(n_it):
            out = net(x)
        et = time()
        results.append(n_it / (et - st))
print(results)
print(np.mean(results))
