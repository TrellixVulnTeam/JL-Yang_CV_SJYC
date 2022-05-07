import transforms as T
from data_utils import *
from datasets.potsdam1024 import Potsdam1024


def build_val_transform(val_input_size, val_label_size):
    mean = (0.3392922, 0.36275864, 0.33674471)
    std = (0.14043275, 0.1387484, 0.1442598)
    transforms = []
    transforms.append(
        T.ValResize(val_input_size, val_label_size)
    )
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)


def build_train_transform(train_min_size, train_max_size, train_crop_size, aug_mode, ignore_value):
    mean = (0.3392922, 0.36275864, 0.33674471)
    std = (0.14043275, 0.1387484, 0.1442598)
    fill = tuple([int(v * 255) for v in mean])
    # ignore_value = 255
    edge_aware_crop = False
    resize_mode = "uniform"
    transforms = []
    transforms.append(
        T.RandomResize(train_min_size, train_max_size, resize_mode)
    )
    if isinstance(train_crop_size, int):
        crop_h, crop_w = train_crop_size, train_crop_size
    else:
        crop_h, crop_w = train_crop_size
    transforms.append(
        T.RandomCrop(crop_h, crop_w, fill, ignore_value, random_pad=True, edge_aware=edge_aware_crop)
    )
    transforms.append(T.RandomHorizontalFlip(0.5))
    if aug_mode == "baseline":
        pass
    elif aug_mode == "randaug":
        transforms.append(T.RandAugment(2, 0.2, "full", prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode == "randaug_reduced":
        transforms.append(T.RandAugment(2, 0.2, "reduced", prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode == "randaug_reduced2":
        transforms.append(T.RandAugment(2, 0.3, "reduced2", prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode == "randaug_reduced3":
        transforms.append(T.RandAugment(2, 0.3, "reduced", prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
    elif aug_mode == "colour_jitter":
        transforms.append(T.ColorJitter(0.3, 0.3, 0.3, 0, prob=1))
    elif aug_mode == "rotate":
        transforms.append(T.RandomRotation((-10, 10), mean=fill, ignore_value=ignore_value, prob=1.0, expand=False))
    elif aug_mode == "noise":
        transforms.append(T.AddNoise(10, prob=1.0))
    elif aug_mode == "custom1":
        transforms.append(T.RandAugment(2, 0.2, "reduced", prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
        transforms.append(T.AddNoise(10, prob=0.2))
    elif aug_mode == "custom2":
        transforms.append(T.RandAugment(2, 0.2, "reduced2", prob=1.0, fill=fill,
                                        ignore_value=ignore_value))
        transforms.append(T.AddNoise(10, prob=0.1))
    elif aug_mode == "custom3":
        transforms.append(T.ColorJitter(0.3, 0.4, 0.5, 0, prob=1))
    else:
        raise NotImplementedError()
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(
        mean,
        std
    ))
    return T.Compose(transforms)


def get_potsdam1024(root, batch_size, train_min_size, train_max_size, train_crop_size, val_input_size, val_label_size,
                    aug_mode, train_split, val_split, num_workers, ignore_value):
    train_transform = build_train_transform(train_min_size, train_max_size, train_crop_size, aug_mode, ignore_value)
    val_transform = build_val_transform(val_input_size, val_label_size)
    train = Potsdam1024(root, train_split, transforms=train_transform)
    val = Potsdam1024(root, val_split, transforms=val_transform)
    train_loader = get_dataloader_train(train, batch_size, num_workers)
    val_loader = get_dataloader_val(val, num_workers)
    return train_loader, val_loader, train
