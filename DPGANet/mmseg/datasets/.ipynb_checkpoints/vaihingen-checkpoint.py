from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class Vaihingen(CustomDataset):
  CLASSES = ("Clutter","Impervious_surface","Building","Low_vegetation","Tree","Car")
  PALETTE = [[255,0,0],[255,255,255],[0,0,255],[0,255,255],[0,255,0],[255,255,0]]
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png', 
                      **kwargs)
                     
    #assert osp.exists(self.img_dir) and self.split is not None
