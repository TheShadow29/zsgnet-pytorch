"""
Python file to create required csvs for ReferIt datasets
Author: Arka Sadhu
Adapted from https://github.com/lichengunc/refer
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
from collections import defaultdict
import numpy as np
from typing import Dict, List, Any
from ds_prep_utils import Cft, ID, BaseCSVPrepare, DF
import pickle


class ReferItCSVPrepare(BaseCSVPrepare):
    """
    Data preparation class
    """

    def after_init(self):
        self.splitBy = 'berkeley'
        self.data_dir = self.ds_root / 'refclef'
        self.ref_ann_file = self.data_dir / f'refs({self.splitBy}).p'
        self.ref_instance_file = self.data_dir / 'instances.json'

        self.ref_ann = pickle.load(self.ref_ann_file.open('rb'))
        self.ref_inst = json.load(self.ref_instance_file.open('r'))
        self.ref_inst_ann = self.ref_inst['annotations']

    def get_annotations(self):
        self.instance_dict_by_ann_id = {
            v['id']: ind for ind, v in enumerate(self.ref_inst_ann)}
        out_dict_list = []
        for rj in self.ref_ann:
            spl = rj['split']
            sents = rj['sentences']
            ann_id = rj['ann_id']
            inst_bbox = self.ref_inst_ann[self.instance_dict_by_ann_id[ann_id]]['bbox']
            # Saving in [x0, y0, x1, y1] format
            inst_bbox = [inst_bbox[0], inst_bbox[1],
                         inst_bbox[2] + inst_bbox[0], inst_bbox[3]+inst_bbox[1]]

            sents = [s['raw'] for s in sents]
            sents = [t.strip().lower() for t in sents]
            out_dict = {}
            out_dict['img_id'] = f"{rj['image_id']}.jpg"
            out_dict['bbox'] = inst_bbox
            out_dict['split'] = spl
            out_dict['query'] = sents
            out_dict_list.append(out_dict)
        return out_dict_list

    def get_dfmask_from_ids(self, ids: List[Any], annots: DF):
        return ids

    def get_trn_val_test_ids(self, output_annot: DF):
        trn_ids_mask = output_annot.split.apply(lambda x: x == 'train')
        val_ids_mask = output_annot.split.apply(lambda x: x == 'val')
        test_ids_mask = output_annot.split.apply(lambda x: x == 'test')
        return trn_ids_mask, val_ids_mask, test_ids_mask


if __name__ == '__main__':
    ds_cfg = json.load(open('./data/ds_prep_config.json'))
    ref = ReferItCSVPrepare(ds_cfg['refclef'])
    ref.save_annot_to_format()
