"""
Create CSV file annotations for Flickr30k
"""

import pandas as pd
from xml.etree import ElementTree as et
from pathlib import Path
from tqdm import tqdm
import json
import re
from collections import defaultdict
import numpy as np
from ds_prep_utils import Cft, ID, BaseCSVPrepare, union_of_rects


class Flickr_one_img_info:
    def __init__(self, ds_cfg: Cft, img_id: ID, results_out):
        self.img_id = img_id
        self.rw = results_out
        ann_path = Path(ds_cfg.ann_path)
        sen_path = Path(ds_cfg.sen_path)
        self.ann_fname = ann_path / f'{img_id}.xml'
        self.sen_fname = sen_path / f'{img_id}.txt'
        self.ann_file = et.parse(self.ann_fname).getroot()
        # list of lists
        self.cid_bbox_dict = defaultdict(list)
        self.cid_sc_nobnd = dict()
        self.cid_entity_dict = dict()
        self.cid_text_dict = defaultdict(list)

        self.p1 = re.compile(r'\[.*?\]')
        self.p2 = re.compile(r'\[/EN#(\d*)/(\w*)\s(.*)\]')
        self.p3 = re.compile(r'\[/EN#(\d*)/(\w*)/(\w*)\s(.*)\]')

        self.cid_dict = dict()
        self.get_full_ann()

    def get_img_dim(self):
        tmp = self.ann_file.find('size')
        self.img_w = int(tmp.find('width').text)
        self.img_h = int(tmp.find('height').text)
        self.img_depth = int(tmp.find('depth').text)

    def get_bbox(self, bbx):
        xmin = int(bbx.find('xmin').text)
        ymin = int(bbx.find('ymin').text)
        xmax = int(bbx.find('xmax').text)
        ymax = int(bbx.find('ymax').text)
        return [xmin, ymin, xmax, ymax]

    def get_ann(self):
        assert str(self.img_id) == self.ann_file.findall(
            'filename')[0].text[:-4]
        tmp = self.ann_file.findall('object')

        for o in tmp:
            nlist = o.findall('name')
            bbox = o.find('bndbox')
            sc = o.find('scene')
            nbnd = o.find('nobndbox')

            if bbox is not None:
                for n in nlist:
                    bnd_bbox = self.get_bbox(bbox)
                    self.cid_bbox_dict[int(n.text)].append(bnd_bbox)
            if sc is None:
                sc = -1
            else:
                sc = int(sc.text)
            if nbnd is None:
                nbnd = -1
            else:
                nbnd = int(nbnd.text)

            for n in nlist:
                self.cid_sc_nobnd[int(n.text)] = {'scene': sc, 'nobnd': nbnd}
        return

    def get_sen_ann(self):
        """
        Some complicated logic
        """
        with self.sen_fname.open('r') as g:
            for l in g.readlines():
                tmp = self.p1.findall(l)
                for t in tmp:
                    tmp2 = self.p2.findall(t)
                    if len(tmp2) == 0:
                        tmp2 = self.p3.findall(t)
                    if len(tmp2) != 1:
                        tmp2 = self.p3.findall(t)
                        assert len(tmp2) == 1
                        tmp3 = tmp2[0]
                        if int(tmp3[0]) != 0:
                            self.cid_entity_dict[int(tmp3[0])] = [
                                tmp3[1], tmp3[2]]
                            self.cid_text_dict[int(tmp3[0])].append(tmp3[3])

                    tmp3 = tmp2[0]
                    if int(tmp3[0]) != 0:
                        self.cid_entity_dict[int(tmp3[0])] = tmp3[1]
                        self.cid_text_dict[int(tmp3[0])].append(tmp3[-1])
        return

    def get_full_ann(self):
        self.get_ann()
        self.get_sen_ann()
        rw = self.rw
        for k in self.cid_entity_dict.keys():
            bbox = self.cid_bbox_dict[k]
            if len(bbox) == 0:
                continue
            if len(bbox) > 1:
                bbox = union_of_rects(np.array(bbox)).tolist()
            else:
                bbox = bbox[0]
            tmpl = {'bbox': bbox,
                    'scene': self.cid_sc_nobnd[k]['scene'],
                    'nobnd': self.cid_sc_nobnd[k]['nobnd'],
                    'entity': self.cid_entity_dict[k],
                    'query': self.cid_text_dict[k],
                    'full_txt': rw[str(self.img_id)]}
            self.cid_dict[k] = tmpl
        return self.cid_dict


class FlickrCSVPrepare(BaseCSVPrepare):
    def get_annotations(self):
        results_out = self.ds_root / 'results.json'
        rw = json.load(results_out.open('r'))
        full_img_cid_dict = {}
        for img_id in tqdm(rw.keys(), total=len(rw)):
            f = Flickr_one_img_info(self.ds_prep_cfg, img_id, rw)
            full_img_cid_dict[f.img_id] = f.cid_dict
        json.dump(full_img_cid_dict, open(
            self.ds_root / 'all_ann_2.json', 'w'))

        out_dict_list = []
        for img_id in tqdm(full_img_cid_dict):
            for cid in full_img_cid_dict[img_id]:
                out_dict = full_img_cid_dict[img_id][cid]
                out_dict['img_id'] = img_id
                out_dict_list.append(out_dict)

        return out_dict_list

    def get_trn_val_test_ids(self, output_annot=None):
        trn_ids = list(pd.read_csv(
            self.ds_prep_cfg.trn_img_ids, header=None)[0])
        val_ids = list(pd.read_csv(
            self.ds_prep_cfg.val_img_ids, header=None)[0])
        test_ids = list(pd.read_csv(
            self.ds_prep_cfg.test_img_ids, header=None)[0])

        return trn_ids, val_ids, test_ids


if __name__ == '__main__':
    ds_cfg = json.load(open('./data/ds_prep_config.json'))
    fl = FlickrCSVPrepare(ds_cfg['flickr30k'])
    fl.save_annot_to_format()
