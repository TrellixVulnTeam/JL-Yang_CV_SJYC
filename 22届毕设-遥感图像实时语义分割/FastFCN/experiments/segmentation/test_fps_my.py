import os
from tqdm import tqdm

import torch
from torch.nn import BatchNorm2d
import numpy as np
from time import time

from encoding.models import get_segmentation_model

from .option import Options


def test_fps(args):
    # model
    model = get_segmentation_model(args.model, dataset=args.dataset,
                                   backbone=args.backbone, dilated=args.dilated,
                                   lateral=args.lateral, jpu=args.jpu, aux=args.aux,
                                   se_loss=args.se_loss, norm_layer=BatchNorm2d,
                                   base_size=args.base_size, crop_size=args.crop_size)

    print(model)
    num_total = sum([l.numel() for l in model.pretrained.parameters()])
    num_train = sum([l.numel() for l in model.head.parameters()])
    print(f"pretrained: {num_total}")
    print(f"head: {num_train}")
    model.cuda()
    model.eval()
    # virtualize
    N_it = 10
    n_it = 1000
    x = torch.tensor(np.random.random((1, 3, 1024, 1024)) - 0.5, dtype=torch.float).cuda()
    # process
    with torch.no_grad():
        result = []
        for _ in tqdm(range(N_it)):
            st = time()
            for _ in range(n_it):
                out = model(x)
            et = time()
            result.append(n_it / (et - st))  # fps

    print(f"result: {result}")
    print("mean: {} and std: {}".format(np.mean(result), np.std(result)))


if __name__ == "__main__":
    option = Options()
    args = option.parse()
    torch.manual_seed(args.seed)
    test_fps(args)
