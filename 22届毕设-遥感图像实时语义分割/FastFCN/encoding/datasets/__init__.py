from .base import *
from .potsdam_1024 import Potsdam

datasets = {
    'potsdam': Potsdam,
}


def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
