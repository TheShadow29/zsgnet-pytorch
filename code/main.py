"""
QNet: Main SetUp Code
Author: Arka Sadhu
"""
import pandas as pd
from dat_loader import get_data
from mdl import get_default_net
# from qnet_model import get_default_net
from loss import get_default_loss
import torch
import fire
from evaluator import Evaluator
# from evaluate import Evaluator
import json
from functools import partial
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
from utils import Learner
# import logging
from extended_config import cfg as conf

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
# logger = logging.getLogger(__name__)


def sanity_check(learn):
    qnet = learn.model
    qlos = learn.loss_func
    db = learn.data
    x, y = next(iter(db.train_dl))
    opt = torch.optim.Adam(qnet.parameters(), lr=2e-4)
    met = learn.metrics[0]
    for i in range(1000):
        opt.zero_grad()
        out = qnet(*x)
        loss = qlos(out, y)
        loss.backward()
        opt.step()
        met_out = met(out, y)
        # print(i, loss.item(), met_out.item())
        print(i, qlos.box_loss_smooth.smooth.item(), qlos.cls_loss_smooth.smooth.item(),
              met_out.item(), met.best_possible_result.item())


def test_for_entities(learn):
    learn.model.eval()
    full_eval_lists = []
    with torch.no_grad():
        for xb, yb in tqdm(learn.data.test_dl):
            out = learn.model(*xb)
            evl = learn.metrics[0](out, yb)
            eval_list = learn.metrics[0].fin_results
            full_eval_lists.append(eval_list)
        full_eval_tensor = torch.cat(full_eval_lists, dim=0)
        # import pdb
        # pdb.set_trace()
        print(f'Acc: {full_eval_tensor.float().mean()}')
        full_eval_df = pd.DataFrame(full_eval_tensor.tolist())
        full_eval_df.to_csv('./vgg_flickr_eval_test.lst',
                            header=False, index=False)


def learner_init(uid, cfg):
    device_count = torch.cuda.device_count()
    device = torch.device('cuda')

    if type(cfg['ratios']) != list:
        ratios = eval(cfg['ratios'], {})
    else:
        ratios = cfg['ratios']
    if type(cfg['scales']) != list:
        scales = cfg['scale_factor'] * np.array(eval(cfg['scales'], {}))
    else:
        scales = cfg['scale_factor'] * np.array(cfg['scales'])

    num_anchors = len(ratios) * len(scales)
    qnet = get_default_net(num_anchors=num_anchors, cfg=cfg)
    qnet = qnet.to(device)
    qnet = torch.nn.DataParallel(qnet)

    qlos = get_default_loss(
        ratios, scales, cfg)
    qlos = qlos.to(device)
    qeval = Evaluator(ratios, scales, cfg)
    # db = get_data(bs=cfg['bs'] * device_count, nw=cfg['nw'], bsv=cfg['bsv'] * device_count,
    #               nwv=cfg['nwv'], devices=cfg['devices'], do_tfms=cfg['do_tfms'],
    #               cfg=cfg, data_cfg=data_cfg)
    # db = get_data(cfg, ds_name=cfg['ds_to_use'])
    db = get_data(cfg)
    opt_fn = partial(torch.optim.Adam, betas=(0.9, 0.99))

    # Note: Currently using default optimizer
    learn = Learner(uid=uid, data=db, mdl=qnet, loss_fn=qlos,
                    opt_fn=opt_fn, eval_fn=qeval, device=device, cfg=cfg)
    return learn


def main(uid, del_existing=False, resume=True, **kwargs):
    # cfg = json.load(open('cfg.json'))
    cfg = conf
    cfg['resume'] = resume
    cfg['del_existing'] = del_existing
    cfg.update(kwargs)

    cfg.device_count = torch.cuda.device_count()
    # data_cfg = json.load(open('./ds_info.json'))
    if cfg.do_dp:
        cfg.bs = cfg.bs * cfg.device_count
        cfg.nw = cfg.nw * cfg.device_count

        cfg.bsv = cfg.bsv * cfg.device_count
        cfg.nwv = cfg.nwv * cfg.device_count

    learn = learner_init(uid, cfg)
    if not cfg['test_only']:
        learn.fit(epochs=int(cfg['epochs']), lr=cfg['lr'])
    elif cfg['get_predictions']:
        ofile = './odict.json'
        if 'ofile' in kwargs:
            ofile = kwargs['ofile']
        out_dict = learn.get_predictions()
        json.dump(out_dict, open(ofile, 'w'))
    else:
        print(cfg)
        if cfg['valid_also']:
            val_loss, val_acc = learn.validate(db=learn.data.valid_dl)
            for k in val_acc:
                print(val_acc[k])
        # if isinstance(learn.data.test_dl, list):
        #     for i, t in enumerate(learn.data.test_dl):
        #         print('For dl ', i)
        #         test_loss, test_acc = learn.validate(db=t)
        #         for k in test_acc:
        #             print(test_acc[k])

        # else:
        #     test_loss, test_acc = learn.validate(db=learn.data.test_dl)
        #     for k in test_acc:
        #         print(test_acc[k])
        # learn.validate()
        # sanity_check(learn)


if __name__ == '__main__':
    fire.Fire(main)
