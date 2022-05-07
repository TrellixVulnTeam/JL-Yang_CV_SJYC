import os
from tqdm import tqdm

import torch
from torch.nn import BatchNorm2d
import torchvision.transforms as transform

import encoding.utils as utils

from PIL import Image

from encoding.datasets import datasets
from encoding.models import get_model, get_segmentation_model, MultiEvalModule

from .option import Options


def test(args):
    # data transforms
    input_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize([0.3392992, 0.36275864, 0.33674471], [0.14043275, 0.1387484, 0.1442598])])
    # model
    if args.model_zoo is not None:
        model = get_model(args.model_zoo, pretrained=True)
    else:
        model = get_segmentation_model(args.model, dataset=args.dataset,
                                       backbone=args.backbone, dilated=args.dilated,
                                       lateral=args.lateral, jpu=args.jpu, aux=args.aux,
                                       se_loss=args.se_loss, norm_layer=BatchNorm2d,
                                       base_size=args.base_size, crop_size=args.crop_size)
        # resuming checkpoint
        if args.resume is None or not os.path.isfile(args.resume):
            raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        # strict=False, so that it is compatible with old pytorch saved models
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))

    print(model)
    scales = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25] if args.dataset == 'citys' else \
        [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    if not args.ms:
        scales = [1.0]
    num_classes = datasets[args.dataset.lower()].NUM_CLASS
    evaluator = MultiEvalModule(model, num_classes, scales=scales, flip=args.ms).cuda()
    evaluator.eval()

    # read images
    dataset_root = "../dataset/potsdam"
    split_path = os.path.join(dataset_root, "split_1024/test.txt")
    images_root = os.path.join(dataset_root, "2_Ortho_RGB_split_1024")
    output_root = "./output/potsdam-1024"
    with open(split_path, 'r') as file:
        image_list = file.readlines()
        file.close()
    image_list = list(map(lambda x: x.split(',')[0], image_list))
    # process
    for i in tqdm(image_list):
        image_path = os.path.join(images_root, i)
        assert os.path.isfile(image_path), "{} not a file".format(i)
        img = input_transform(Image.open(image_path).convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            output = evaluator.parallel_forward(img)[0]
            predict = torch.max(output, 1)[1].cpu().numpy()
        mask = utils.get_mask_pallete(predict, args.dataset)
        mask.save(os.path.join(output_root, i))


if __name__ == "__main__":
    option = Options()
    args = option.parse()

    torch.manual_seed(args.seed)

    test(args)
