from yacs.config import CfgNode as CN
import json

ds_info = json.load(open('./configs/ds_info.json'))
def_cfg = json.load(open('./configs/cfg.json'))

cfg = CN(def_cfg)
cfg.ds_info = CN(ds_info)


# Device
# setting default device
cfg.device = 'cuda'

# Training
cfg.local_rank = 0
cfg.do_dist = False

# Testing
cfg.only_val = False
cfg.only_test = False

key_maps = {}
