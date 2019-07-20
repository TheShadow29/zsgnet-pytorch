from yacs.config import CfgNode as CN
import json

ds_info = json.load(open('./configs/ds_info.json'))
def_cfg = json.load(open('./configs/cfg.json'))

cfg = CN(def_cfg)
cfg.ds_info = CN(ds_info)

# DATASET
cfg.FIXED_W = 300
cfg.FIXED_H = 300

cfg.flickr30k = CN({'img_dir': './data/flickr30k/flickr30k_images'})

# Training
cfg.local_rank = 0
cfg.do_dist = False


key_maps = {}
