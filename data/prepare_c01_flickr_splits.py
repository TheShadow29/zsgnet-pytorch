"""
Creating zero-shot splits from Flickr Annotations
"""
from typing import Dict, List, Any
from ds_prep_utils import Cft, ID, BaseCSVPrepare, DF
from dataclasses import dataclass
from yacs.config import CfgNode as CN
from pathlib import Path
import json
import pandas as pd
import spacy
from tqdm import tqdm
from collections import Counter
import numpy as np
import copy
import pickle
nlp = spacy.load("en_core_web_sm")

np.random.seed(5)


class FlickrUnseenWordsCSVPrepare(BaseCSVPrepare):

    def after_init(self):
        self.flickr_ann_file = self.ds_root.parent / 'all_annot_new.json'
        self.flickr_ann = None
        self.load_annotations()

    def load_annotations(self):
        if self.flickr_ann is None:
            self.flickr_ann = json.load(open(self.flickr_ann_file))
        return pd.DataFrame(self.flickr_ann)

    def get_annotations(self):
        return

    def get_query_word_list(self):
        self.query_word_lemma_file = self.ds_root / 'query_word_lemma_counter.json'
        if not self.query_word_lemma_file.exists():
            query_word_list = []
            for ind, grnd_dict in enumerate(tqdm(self.flickr_ann)):
                queries = grnd_dict['query']
                for query in queries:
                    tmp_query = nlp(query)
                    query_word_list += [t.lemma_ for t in tmp_query]
            query_word_counter = Counter(query_word_list)
            json.dump(query_word_counter, open(
                self.query_word_lemma_file, 'w'))
        return Counter(json.load(open(self.query_word_lemma_file)))

    def create_exclude_include_list(self):
        self.exclude_include_list_file = self.ds_root / 'inc_exc_word_list.json'
        if not self.exclude_include_list_file.exists():
            self.load_annotations()
            queries_lemma_count = self.get_query_word_list()

            # create include list
            qmost_common = queries_lemma_count.most_common(500)
            include_list = [q[0] for q in qmost_common]

            # exclude list
            remaining_list = [
                r for r in queries_lemma_count if r not in set(include_list)]

            to_include_prob = 0.7
            num_to_incl = int(to_include_prob * len(remaining_list))

            id_list = np.random.permutation(len(remaining_list))
            to_include = id_list[:num_to_incl]
            to_exclude = id_list[num_to_incl:]

            include_list += [remaining_list[t] for t in to_include]
            exclude_list = [remaining_list[t] for t in to_exclude]

            out_dict = {'exclude_list': exclude_list,
                        'include_list': include_list}
            json.dump(out_dict, self.exclude_include_list_file.open('w'))
        return json.load(self.exclude_include_list_file.open('r'))

    def get_trn_val_test_ids(self, output_annot=None):
        inc_excl_lists = self.create_exclude_include_list()
        incl_set = inc_excl_lists['include_list']
        excl_set = inc_excl_lists['exclude_list']

        test_ids_file = self.ds_root / 'test_ids.pkl'
        new_output_annot_file = self.ds_root / 'test_output_annot.pkl'
        if not test_ids_file.exists():
            test_ids = []
            new_output_annot = []
            for ind, grnd_dict in enumerate(tqdm(self.flickr_ann)):
                queries = grnd_dict['query']
                qs_to_use = []
                for query in queries:
                    tmp_query = nlp(query)
                    last_idx = -1
                    qu = tmp_query[last_idx]
                    while not len(qu.text) > 1:
                        print('why', qu.text)
                        try:
                            last_idx -= 1
                            qu = tmp_query[last_idx]
                        except IndexError:
                            print('noope')
                            break
                    if not (qu.lemma_ in incl_set):
                        assert qu.lemma_ in excl_set
                        qs_to_use.append(query)
                if len(qs_to_use) > 0:
                    qs_to_use = list(set(qs_to_use))
                    grnd_dict1 = copy.deepcopy(grnd_dict)
                    grnd_dict1['query'] = qs_to_use
                    grnd_dict1['split_type'] = 'test'
                    new_output_annot.append(grnd_dict1)
                    test_ids.append(grnd_dict1['img_id'])
            pickle.dump(test_ids, test_ids_file.open('wb'))
            pickle.dump(new_output_annot, new_output_annot_file.open('wb'))
        test_ids = pickle.load(test_ids_file.open('rb'))
        new_output_annot = pickle.load(new_output_annot_file.open('rb'))

        flickr_df = pd.DataFrame(self.flickr_ann)
        all_ids = set(list(flickr_df.img_id))
        trn_val_ids = list(all_ids - set(test_ids))

        to_include_prob = 0.7
        num_to_incl = int(to_include_prob * len(trn_val_ids))

        id_list = np.random.permutation(len(trn_val_ids))
        trids = id_list[:num_to_incl]
        vlids = id_list[num_to_incl:]

        trn_ids = [trn_val_ids[trid] for trid in trids]
        val_ids = [trn_val_ids[vlid] for vlid in vlids]

        for ind, grnd_dict in enumerate(tqdm(self.flickr_ann)):
            if grnd_dict['img_id'] in trn_val_ids:
                queries = grnd_dict['query']
                # if not all([nlp(q)[-1].lemma_ in incl_set for q in queries]):
                # continue
                new_output_annot.append(grnd_dict)

        return trn_ids, val_ids, test_ids, pd.DataFrame(new_output_annot)

    def save_annot_to_format(self):
        """
        Saves the annotations to the following csv format
        img_name,x1,y1,x2,y2,query(ies)
        """
        output_annot = self.load_annotations()
        trn_ids, val_ids, test_ids, output_annot = self.get_trn_val_test_ids(
            output_annot)
        output_annot = output_annot[['img_id', 'bbox', 'query']]
        # trn_df = self.get_df_from_ids(
        #     trn_ids, output_annot, split_type='train')
        # trn_df.to_csv(self.csv_root / 'train.csv', index=False, header=True)

        # val_df = self.get_df_from_ids(val_ids, output_annot)
        # val_df.to_csv(self.csv_root / 'val.csv', index=False, header=True)

        if test_ids is not None:
            import pdb
            pdb.set_trace()
            test_df = self.get_df_from_ids(test_ids, output_annot)
            test_df.to_csv(self.csv_root / 'test.csv',
                           index=False, header=True)


if __name__ == '__main__':
    ds_cfg = json.load(open('./data/ds_prep_config.json'))
    fl0 = FlickrUnseenWordsCSVPrepare(ds_cfg['flickr_unseen_words'])
    # fl0.create_exclude_include_list()
    fl0.save_annot_to_format()
