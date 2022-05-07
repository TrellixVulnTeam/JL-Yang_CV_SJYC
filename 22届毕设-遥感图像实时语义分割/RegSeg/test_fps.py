import torch
from time import time
from model import RegSeg
from tqdm import tqdm
import numpy as np

# load model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
md = RegSeg(name="exp48_decoder26", num_classes=6,
            pretrained="./checkpoints/potsdam1024_exp48_decoder26_500_epochs_run4")
md.eval()
md.to(device)

a = torch.tensor(np.random.random((1, 3, 1024, 1024)) - 0.5, dtype=torch.float).cuda()
N_it = 15
n_it = 2000
results = []

# go
with torch.no_grad():
    for _ in tqdm(range(N_it)):
        st = time()
        for _ in range(n_it):
            out = md(a)
        et = time()
        results.append(n_it / (et - st))

print(results)
print(np.mean(results))
