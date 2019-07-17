"""
Utility functions to prepare dataset in a common format
"""
from typing import List, Dict, Union, Any
from yacs.config import CfgNode as CN
from pathlib import Path
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from dataclasses import dataclass
import json
from tqdm import tqdm
import copy

Fpath = Union[Path, str]
Cft = Union[Dict, CN]
DF = pd.DataFrame
ID = Union[int, str]


def union_of_rects(rects):
    """
    Calculates union of two rectangular boxes
    Assumes both rects of form N x [xmin, ymin, xmax, ymax]
    """
    xA = np.min(rects[:, 0])
    yA = np.min(rects[:, 1])
    xB = np.max(rects[:, 2])
    yB = np.max(rects[:, 3])
    return np.array([xA, yA, xB, yB], dtype=np.int32)


@dataclass
class BaseCSVPrepare(ABC):
    """
    Abstract class to prepare CSV files
    to be used for data loading
    ds_root: Path to root directory of the dataset.
    This can be a symbolic path as well
    """
    ds_prep_cfg: Cft

    def __post_init__(self):
        """
        Initializes stuff from the dataset preparation
        configuration.
        """
        # Convert to CN type if not already
        if isinstance(self.ds_prep_cfg, dict):
            self.ds_prep_cfg = CN(self.ds_prep_cfg)

        #  Set the dataset root (resolve symbolic links)
        self.ds_root = Path(self.ds_prep_cfg.root).resolve()
        self.ann_file = self.ds_root / 'all_annot_new.json'
        self.csv_root = self.ds_root / 'csv_dir'
        self.csv_root.mkdir(exist_ok=True)
        self.after_init()

    def after_init(self):
        pass

    def load_annotations(self) -> DF:
        if self.ann_file.exists():
            return pd.DataFrame(json.load(open(self.ann_file)))
        else:
            output = self.get_annotations()
            assert isinstance(output, (dict, list))
            json.dump(output, self.ann_file.open('w'))
            return pd.DataFrame(output)

    @abstractmethod
    def get_annotations(self):
        """
        Getting the annotations, specific to dataset.
        The output should be of the format:
        output_annot = List[grnd_dict]
        grnd_dict = {'bbox': [x1,y1,x2,y2], 'img_id': img_id,
        'queries': List[query], 'entity_name': optional, 'full_sentence': optional}
        bbox should be in x1y1x2y2 format
        """
        raise NotImplementedError

    @abstractmethod
    def get_trn_val_test_ids(self, output_annot=None) -> List[Any]:
        """
        Obtain training, validation and testing ids.
        Depends on the dataset
        """
        raise NotImplementedError

    def get_dfmask_from_ids(self, ids: List[Any], annots: DF):
        ids_set = set(ids)
        return annots.img_id.apply(lambda x: x in ids_set)

    def get_df_from_ids(self, ids: List[Any], annots: DF, split_type='val'):
        """
        Return the df with ids. Basically for train we can directly return.
        For validation, testing each query is separate row.
        """

        msk1 = self.get_dfmask_from_ids(ids, annots)
        annots_to_use = annots[msk1]
        if split_type == 'train':
            return annots_to_use
        else:
            out_dict_list = []
            for ind, row in tqdm(annots_to_use.iterrows(),
                                 total=len(annots_to_use)):
                for query in row['query']:
                    out_dict = copy.deepcopy(row)
                    out_dict['query'] = query
                    out_dict_list.append(out_dict)
            return pd.DataFrame(out_dict_list)

    def save_annot_to_format(self):
        """
        Saves the annotations to the following csv format
        img_name,x1,y1,x2,y2,query(ies)
        """
        output_annot = self.load_annotations()
        trn_ids, val_ids, test_ids = self.get_trn_val_test_ids(output_annot)
        output_annot = output_annot[['img_id', 'bbox', 'query']]
        trn_df = self.get_df_from_ids(
            trn_ids, output_annot, split_type='train')
        trn_df.to_csv(self.csv_root / 'train.csv', index=False, header=True)

        val_df = self.get_df_from_ids(val_ids, output_annot)
        val_df.to_csv(self.csv_root / 'val.csv', index=False, header=True)

        if test_ids is not None:
            test_df = self.get_df_from_ids(test_ids, output_annot)
            test_df.to_csv(self.csv_root / 'test.csv',
                           index=False, header=True)
