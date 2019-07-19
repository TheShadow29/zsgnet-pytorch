from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import pandas as pd
from utils import DataWrap
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import re
import PIL
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import pickle
import ast
import logging
from torchvision import transforms
import spacy
from extended_config import cfg as conf


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

nlp = spacy.load('en_core_web_md')


def pil2tensor(image, dtype: np.dtype):
    "Convert PIL style `image` array to torch style image tensor."
    a = np.asarray(image)
    if a.ndim == 2:
        a = np.expand_dims(a, 2)
    a = np.transpose(a, (1, 0, 2))
    a = np.transpose(a, (2, 1, 0))
    return torch.from_numpy(a.astype(dtype, copy=False))


class ImgQuDataset(Dataset):
    """
    Any Grounding dataset.
    Args:
        train_file (string): CSV file with annotations
        The format should be: img_file, bbox, queries
        Can have same img_file on multiple lines
    """

    def __init__(self, cfg, csv_file, ds_name, split_type='train'):
        self.cfg = cfg
        self.ann_file = csv_file
        self.ds_name = ds_name
        self.split_type = split_type

        # self.image_data = pd.read_csv(csv_file)
        self.image_data = self._read_annotations(csv_file)
        self.image_data = self.image_data.iloc[:200]
        self.img_dir = Path(self.cfg.ds_info[self.ds_name]['img_dir'])
        self.phrase_len = 50
        self.item_getter = getattr(self, 'simple_item_getter')
        # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        # std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        return self.item_getter(idx)

    def simple_item_getter(self, idx):
        img_file, annot, q_chosen = self.load_annotations(idx)
        img = PIL.Image.open(img_file).convert('RGB')

        h, w = img.height, img.width

        q_chosen = q_chosen.strip()
        qtmp = nlp(str(q_chosen))
        if len(qtmp) == 0:
            logger.error('Empty string provided')
            raise NotImplementedError
        qlen = len(qtmp)
        q_chosen = q_chosen + ' PD'*(self.phrase_len - qlen)
        q_chosen_emb = nlp(q_chosen)
        if not len(q_chosen_emb) == self.phrase_len:
            q_chosen_emb = q_chosen_emb[:self.phrase_len]

        q_chosen_emb_vecs = np.array([q.vector for q in q_chosen_emb])
        # qlen = len(q_chosen_emb_vecs)
        # Annot is in x1y1x2y2 format
        target = np.array(annot)
        # img = self.resize_fixed_transform(img)
        img = img.resize((self.cfg.FIXED_W, self.cfg.FIXED_H))
        # Now target is in y1x1y2x2 format which is required by the model
        # The above is because the anchor format is created
        # in row, column format
        target = np.array([target[1], target[0], target[3], target[2]])
        # Resize target to range 0-1
        target = np.array([
            target[0] / h, target[1] / w,
            target[2] / h, target[3] / w
        ])
        # Target in range -1 to 1
        target = 2 * target - 1

        # img = self.img_transforms(img)
        # img = Image(pil2tensor(img, np.float_).float().div_(255))
        img = pil2tensor(img, np.float_).float().div_(255)
        out = {
            'img': img,
            'idxs': torch.tensor(idx),
            'qvec': torch.from_numpy(q_chosen_emb_vecs),
            'qlens': torch.tensor(qlen),
            'annot': torch.from_numpy(target).float(),
            'orig_annot': torch.tensor(annot).float(),
            'img_size': torch.tensor([h, w])
        }

        return out

    def load_annotations(self, idx):
        annotation_list = self.image_data.iloc[idx]
        img_file, x1, y1, x2, y2, queries = annotation_list
        img_file = self.img_dir / f'{img_file}'
        if isinstance(queries, list):
            query_chosen = np.random.choice(queries)
        else:
            assert isinstance(queries, str)
            query_chosen = queries
        if '_' in query_chosen:
            query_chosen = query_chosen.replace('_', ' ')
        # annotations = np.array([y1, x1, y2, x2])
        annotations = np.array([x1, y1, x2, y2])
        return img_file, annotations, query_chosen

    def _read_annotations(self, trn_file):
        trn_data = pd.read_csv(trn_file)
        trn_data['bbox'] = trn_data.bbox.apply(
            lambda x: ast.literal_eval(x))
        if self.split_type == 'train':
            trn_data['query'] = trn_data['query'].apply(
                lambda x: ast.literal_eval(x))

        trn_data['x1'] = trn_data.bbox.apply(lambda x: x[0])
        trn_data['y1'] = trn_data.bbox.apply(lambda x: x[1])
        trn_data['x2'] = trn_data.bbox.apply(lambda x: x[2])
        trn_data['y2'] = trn_data.bbox.apply(lambda x: x[3])
        if self.ds_name == 'flickr30k':
            trn_data = trn_data.assign(
                image_fpath=trn_data.img_id.apply(lambda x: f'{x}.jpg'))
            trn_df = trn_data[['image_fpath',
                               'x1', 'y1', 'x2', 'y2', 'query']]
        elif self.ds_name == 'refclef':
            trn_df = trn_data[['img_id',
                               'x1', 'y1', 'x2', 'y2', 'query']]
        return trn_df


def collater(batch):
    qlens = torch.Tensor([i['qlens'] for i in batch])
    max_qlen = int(qlens.max().item())
    # query_vecs = [torch.Tensor(i['query'][:max_qlen]) for i in batch]
    out_dict = {}
    for k in batch[0]:
        out_dict[k] = torch.stack([b[k] for b in batch]).float()
    out_dict['qvec'] = out_dict['qvec'][:, :max_qlen]

    return out_dict


def get_data(cfg, ds_name):
    # trn_csv_file = Path('./data/flickr30k/csv_dir/train.csv')
    trn_csv_file = cfg.ds_info[ds_name]['trn_csv_file']
    trn_ds = ImgQuDataset(cfg=cfg, csv_file=trn_csv_file,
                          ds_name=ds_name, split_type='train')

    trn_dl = DataLoader(trn_ds,
                        batch_size=cfg.bs,
                        # batch_size=5,
                        num_workers=cfg.nw,
                        shuffle=True, collate_fn=collater)

    # val_csv_file = Path('./data/flickr30k/csv_dir/val.csv')
    val_csv_file = cfg.ds_info[ds_name]['val_csv_file']
    val_ds = ImgQuDataset(cfg=cfg, csv_file=val_csv_file,
                          ds_name=ds_name, split_type='valid')
    val_dl = DataLoader(val_ds,
                        batch_size=cfg.bsv,
                        # batch_size=5,
                        num_workers=cfg.nwv,
                        shuffle=False, collate_fn=collater)
    data = DataWrap(path='./tmp', train_dl=trn_dl, valid_dl=val_dl)
    return data
# def get_data(bs=30, nw=10, bsv=30, nwv=10, devices=0,
#              do_tfms=False, cfg=None, data_cfg=None):
#     ds_to_use = cfg['ds_to_use']
#     data_dir = Path(data_cfg[ds_to_use]['data_dir'])
#     img_dir = Path(data_cfg[ds_to_use]['img_dir'])
#     tmp_path = cfg['tmp_path']
#     csvs_tdir = data_dir / 'csvs'
#     if ds_to_use == 'vg_split':
#         if not cfg['train_balanced_set']:
#             trn_csv = csvs_tdir / 'train_unseen.csv'
#         else:
#             trn_csv = csvs_tdir / 'train_balanced_unseen.csv'
#             val_csv = csvs_tdir / 'val_balanced_unseen.csv'
#         if not cfg['test_balanced_set']:
#             test1_csv = csvs_tdir / 'tp1_unseen.csv'
#             test2_csv = csvs_tdir / 'tp2_unseen.csv'
#         else:
#             test1_csv = csvs_tdir / 'tp1_balanced_unseen.csv'
#             test2_csv = csvs_tdir / 'tp2_balanced_unseen.csv'

#         print('Using custom unseen csv files. Training is {}, Testing is {}'.format(
#             cfg['train_balanced_set'], cfg['test_balanced_set']))
#     else:
#         csv_suffix = cfg['csv_suffix']
#         print(f'Using {csv_suffix} files')
#         trn_csv = csvs_tdir / f'train_{csv_suffix}.csv'
#         val_csv = csvs_tdir / f'val_{csv_suffix}.csv'
#         test_csv = csvs_tdir / f'test_{csv_suffix}.csv'

#     if do_tfms:
#         tfms = get_transforms(do_flip=cfg['do_flip'], max_rotate=1,
#                               max_lighting=0.2, max_zoom=1,
#                               max_warp=0)
#         tfms = [tfms[0], None]
#     else:
#         tfms = [None, None]

#     trn_ds = ImgQuDataset(trn_csv, tdir=data_dir, img_dir=img_dir,
#                           transform=tfms[0], cfg=cfg)
#     trn_dl = DataLoader(trn_ds, batch_size=bs, num_workers=nw, collate_fn=collater,
#                         shuffle=True, drop_last=True)
#     val_ds = ImgQuDataset(val_csv, tdir=data_dir, img_dir=img_dir,
#                           transform=tfms[1], cfg=cfg)
#     val_dl = DataLoader(val_ds, batch_size=bsv, collate_fn=collater,
#                         num_workers=nwv, drop_last=False)
#     if cfg['ds_to_use'] == 'vg_split':
#         test1_ds = ImgQuDataset(test1_csv, tdir=data_dir, img_dir=img_dir,
#                                 transform=tfms[1], cfg=cfg)
#         test1_dl = DataLoader(test1_ds, batch_size=bsv, collate_fn=collater,
#                               num_workers=nwv, drop_last=False)
#         test2_ds = ImgQuDataset(test2_csv, tdir=data_dir, img_dir=img_dir,
#                                 transform=tfms[1], cfg=cfg)
#         test2_dl = DataLoader(test2_ds, batch_size=bsv, collate_fn=collater,
#                               num_workers=nwv, drop_last=False)
#         db = DataWrap(train_dl=trn_dl, valid_dl=val_dl,
#                       test_dl=[test1_dl, test2_dl], path=tmp_path)

#     else:
#         test_ds = ImgQuDataset(test_csv, tdir=data_dir, img_dir=img_dir,
#                                transform=tfms[1], cfg=cfg)
#         test_dl = DataLoader(test_ds, batch_size=bsv, collate_fn=collater,
#                              num_workers=nwv, drop_last=False)
#         db = DataWrap(train_dl=trn_dl, valid_dl=val_dl,
#                       test_dl=test_dl, path=tmp_path)

#     # test_ds = ImgQuDataset(test_csv, tdir=data_dir, img_dir=img_dir,
#     # transform=tfms[1], cfg=cfg)
#     # test_dl = DataLoader(test_ds, batch_size=bsv, collate_fn=collater,
#     # num_workers=nwv, drop_last=False)

#     # db = DataBunch(trn_dl, val_dl, test_dl, path=tmp_path,
#     # collate_fn=collater, device=devices)


#     return db
if __name__ == '__main__':
    cfg = conf
    data = get_data(cfg, ds_name='refclef')

    # cfg = json.load(open('cfg.json', 'r'))
    # data_cfg = json.load(open('ds_info.json', 'r'))
    # cfg['ds_to_use'] = 'vg_split'
    # db = get_data(
    #     60, 14, 60, 14, cfg=cfg, data_cfg=data_cfg)
    # for x in tqdm(db.train_dl):
    #     pass
    # for x in tqdm(db.valid_dl):
    #     pass
