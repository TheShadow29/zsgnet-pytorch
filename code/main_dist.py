"""
Main file for distributed training
"""
from dat_loader import get_data
from mdl import get_default_net
from loss import get_default_loss
from evaluator import get_default_eval
from utils import update_from_dict, Learner, synchronize

import numpy as np
import torch
import fire
from functools import partial

from extended_config import (cfg as conf, key_maps, CN)


def learner_init(uid: str, cfg: CN) -> Learner:
    device = torch.device('cuda')
    data = get_data(cfg)

    # Ugly hack because I wanted ratios, scales
    # in fractional formats
    if type(cfg['ratios']) != list:
        ratios = eval(cfg['ratios'], {})
    else:
        ratios = cfg['ratios']
    if type(cfg['scales']) != list:
        scales = cfg['scale_factor'] * np.array(eval(cfg['scales'], {}))
    else:
        scales = cfg['scale_factor'] * np.array(cfg['scales'])

    num_anchors = len(ratios) * len(scales)
    mdl = get_default_net(num_anchors=num_anchors, cfg=cfg)
    mdl.to(device)
    mdl = torch.nn.parallel.DistributedDataParallel(
        mdl, device_ids=[cfg.local_rank],
        output_device=cfg.local_rank, broadcast_buffers=True,
        find_unused_parameters=True)
    loss_fn = get_default_loss(ratios, scales, cfg)
    # loss_fn.to(device)
    eval_fn = get_default_eval(ratios, scales, cfg)
    # eval_fn.to(device)
    opt_fn = partial(torch.optim.Adam, betas=(0.9, 0.99))

    learn = Learner(uid=uid, data=data, mdl=mdl, loss_fn=loss_fn,
                    opt_fn=opt_fn, eval_fn=eval_fn, device=device, cfg=cfg)
    return learn


def main_dist(uid: str, **kwargs):
    """
    uid is a unique identifier for the experiment name
    Can be kept same as a previous run, by default will start executing
    from latest saved model
    **kwargs: allows arbit arguments of cfg to be changed
    """
    cfg = conf
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        assert 'local_rank' in kwargs
        torch.cuda.set_device(kwargs['local_rank'])
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    # Update the config file depending on the command line args
    cfg = update_from_dict(cfg, kwargs, key_maps)
    cfg.num_gpus = num_gpus
    # Freeze the cfg, can no longer be changed
    cfg.freeze()
    # print(cfg)
    # Initialize learner
    learn = learner_init(uid, cfg)
    learn.fit(epochs=cfg.epochs, lr=cfg.lr)


if __name__ == '__main__':
    fire.Fire(main_dist)
