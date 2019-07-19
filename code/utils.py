"""
Utility functions
"""
from typing import Dict, List, Optional, Union, Any, Callable
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
from collections import Counter
from tqdm import tqdm
import time
import shutil
import json
from fastprogress.fastprogress import master_bar, progress_bar
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)


@dataclass
class DataWrap:
    path: Union[str, Path]
    train_dl: DataLoader
    valid_dl: DataLoader
    test_dl: Optional[Union[DataLoader, List]] = None


class SmoothenValue():
    """
    Create a smooth moving average for a value(loss, etc) using `beta`.
    Adapted from fastai(https://github.com/fastai/fastai)
    """

    def __init__(self, beta: float):
        self.beta, self.n, self.mov_avg = beta, 0, 0
        self.smooth = 0

    def add_value(self, val: float) -> None:
        "Add `val` to calculate updated smoothed value."
        self.n += 1
        self.mov_avg = self.beta * \
            self.mov_avg + (1 - self.beta) * val
        self.smooth = self.mov_avg / (1 - self.beta ** self.n)


class SmoothenDict:
    "Converts list to dicts"

    def __init__(self, keys: List[str], val: int):
        self.keys = keys
        self.smooth_vals = {k: SmoothenValue(val) for k in keys}

    def add_value(self, val: Dict[str, torch.tensor]):
        for k in self.keys:
            self.smooth_vals[k].add_value(val[k].detach())

    @property
    def smooth(self):
        return {k: self.smooth_vals[k].smooth for k in self.keys}

    @property
    def smooth1(self):
        return self.smooth_vals[self.keys[0]].smooth


def compute_avg(inp: List, nums: torch.tensor) -> float:
    "Computes average given list of torch.tensor and numbers corresponding to them"
    return (torch.stack(inp) * nums).sum() / nums.sum()


def compute_avg_dict(inp: Dict[str, List],
                     nums: torch.tensor) -> Dict[str, float]:
    "Takes dict as input"
    out_dict = {}
    for k in inp:
        out_dict[k] = compute_avg(inp[k], nums)

    return out_dict


def good_format_stats(names, stats) -> str:
    "Format stats before printing."
    str_stats = []
    for name, stat in zip(names, stats):
        t = str(stat) if isinstance(stat, int) else f'{stat.item():.4f}'
        t += ' ' * (len(name) - len(t))
        str_stats.append(t)
    return '  '.join(str_stats)


@dataclass
class Learner:
    uid: str
    data: DataWrap
    mdl: nn.Module
    loss_fn: nn.Module
    cfg: Dict
    eval_fn: nn.Module
    opt_fn: Callable
    device: torch.device = torch.device('cuda:0')

    def __post_init__(self):
        "Setup log file, load model if required"
        self.logger = logging.getLogger(__name__)
        if not isinstance(self.data.test_dl, list):
            self.data.test_dl = [self.data.test_dl]

        self.loss_keys = self.loss_fn.loss_keys
        self.met_keys = self.eval_fn.met_keys
        self.log_keys = ['epochs']

        for k in self.loss_keys:
            self.log_keys += [f'trn_{k}', f'val_{k}']
        for k in self.met_keys:
            self.log_keys += [f'trn_{k}', f'val_{k}']

        if self.cfg['test_at_runtime']:
            acc_met_key = self.met_keys[0]
            if isinstance(self.data.test_dl, list):
                for i in range(len(self.data.test_dl)):
                    self.log_keys += [f'test{i+1}_{acc_met_key}']
            else:
                self.log_keys += [f'test_{acc_met_key}']

        self.log_file = Path(self.data.path) / 'logs' / f'{self.uid}.txt'
        self.log_dir = self.log_file.parent / f'{self.uid}'

        self.model_file = Path(self.data.path) / 'models' / f'{self.uid}.pth'
        self.model_file.parent.mkdir(exist_ok=True, parents=True)

        self.prepare_log_file()

        # Set the number of iterations to 0. Updated in loading if required
        self.num_it = 0

        if self.cfg['resume']:
            self.load_model_dict(
                resume_path=self.cfg['resume_path'], load_opt=self.cfg['load_opt'])
        self.best_met = 0

    def prepare_log_file(self):
        "Prepares the log files depending on arguments"
        if self.log_file.exists():
            if self.cfg['del_existing']:
                self.logger.info(
                    f'removing existing log with same name {self.log_dir.stem}')
                shutil.rmtree(self.log_dir)
                f = self.log_file.open('w')
            else:
                f = self.log_file.open('a')
        else:
            self.log_file.parent.mkdir(exist_ok=True, parents=True)
            f = self.log_file.open('w')

        cfgtxt = json.dumps(self.cfg)
        f.write(cfgtxt)
        f.write('\n\n')
        f.write('  '.join(self.log_keys) + '\n')
        f.close()

    def update_log_file(self, towrite: str):
        "Updates the log files as and when required"
        with self.log_file.open('a') as f:
            f.write(towrite + '\n')

    def do_test(self, mb=None) -> List[torch.tensor]:
        test_accs = []
        for t in self.data.test_dl:
            test_loss, test_acc = self.validate(mb=mb, db=t)
            test_accs += [test_acc[self.met_keys[0]]]
        return test_accs

    def get_predictions(self, db=None):
        if db is None:
            db = self.data.test_dl
        else:
            if not isinstance(db, list):
                db = [db]
        out_dict = {}
        with torch.no_grad():
            for tidx, t in enumerate(db):
                results_dict = {'Acc': [], 'MaxPos': []}
                strt_idx = 0
                for batch in tqdm(t):
                    for k in batch:
                        batch[k] = batch[k].to(self.device)
                    out = self.mdl(batch)
                    metric = self.eval_fn(out, batch)
                    remember_info = self.eval_fn.remember_info
                    for k in remember_info:
                        results_dict[k] += remember_info[k].tolist()

                out_dict[f'test_{tidx}'] = results_dict
        return out_dict

    def validate(self, db: Optional[DataLoader] = None,
                 mb=None) -> List[torch.tensor]:
        "Validation loop, done after every epoch"
        self.mdl.eval()
        if db is None:
            db = self.data.valid_dl
        with torch.no_grad():
            val_losses = {k: [] for k in self.loss_keys}
            eval_metrics = {k: [] for k in self.met_keys}
            nums = []
            for batch in progress_bar(db, parent=mb):
                for b in batch.keys():
                    batch[b] = batch[b].to(self.device)
                out = self.mdl(batch)
                out_loss = self.loss_fn(out, batch)

                metric = self.eval_fn(out, batch)
                for k in self.loss_keys:
                    val_losses[k].append(out_loss[k].detach().cpu())
                for k in self.met_keys:
                    eval_metrics[k].append(metric[k].detach().cpu())
                nums.append(batch[next(iter(batch))].shape[0])

            del batch
            nums = torch.tensor(nums).float()
            val_loss = compute_avg_dict(val_losses, nums)
            eval_metric = compute_avg_dict(eval_metrics, nums)
            return val_loss, eval_metric

    def train_epoch(self, mb) -> List[torch.tensor]:
        "One epoch used for training"
        self.mdl.train()
        # trn_loss = SmoothenValue(0.9)
        trn_loss = SmoothenDict(self.loss_keys, 0.9)
        trn_acc = SmoothenDict(self.met_keys, 0.9)

        for batch_id, batch in enumerate(progress_bar(self.data.train_dl, parent=mb)):
            # for batch_id, batch in progress_bar(QueueIterator(batch_queue), parent=mb):
            # for batch_id, batch in QueueIterator(batch_queue):
            # Increment number of iterations
            self.num_it += 1
            for b in batch.keys():
                batch[b] = batch[b].to(self.device)
            self.optimizer.zero_grad()
            out = self.mdl(batch)
            out_loss = self.loss_fn(out, batch)
            loss = out_loss[self.loss_keys[0]]
            loss = loss.mean()
            loss.backward()
            self.optimizer.step()
            metric = self.eval_fn(out, batch)
            trn_loss.add_value(out_loss)
            trn_acc.add_value(metric)
            mb.child.comment = (
                f'LossB {loss: .4f} | SmLossB {trn_loss.smooth1: .4f} | AccB {trn_acc.smooth1: .4f}')
            del out_loss
            del loss
            # print(f'Done {batch_id}')
        del batch
        self.optimizer.zero_grad()
        return trn_loss.smooth, trn_acc.smooth

    def load_model_dict(self, resume_path: Optional[str] = None, load_opt: bool = False):
        "Load the model and/or optimizer"

        if resume_path == "":
            mfile = self.model_file
        else:
            mfile = Path(resume_path)

        if not mfile.exists():
            self.logger.info(
                f'No existing model in {mfile}, starting from scratch')
            return
        try:
            checkpoint = torch.load(open(mfile, 'rb'))
            self.logger.info(f'Loaded model from {mfile} Correctly')
        except OSError as e:
            self.logger.error(
                f'Some problem with resume path: {resume_path}. Exception raised {e}')
            raise e
        if self.cfg['load_normally']:
            self.mdl.load_state_dict(
                checkpoint['model_state_dict'], strict=self.cfg['strict_load'])
        # self.logger.info('Added model file correctly')
        if 'num_it' in checkpoint.keys():
            self.num_it = checkpoint['num_it']

        if load_opt:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def save_model_dict(self):
        "Save the model and optimizer"
        checkpoint = {'model_state_dict': self.mdl.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'num_it': self.num_it}
        torch.save(checkpoint, self.model_file.open('wb'))

    def prepare_to_write(self, epoch: int,
                         train_loss: torch.tensor,
                         val_loss: torch.tensor,
                         train_acc: Dict[str, torch.tensor],
                         val_acc: Dict[str, torch.tensor],
                         test_accs: List = None) -> List[torch.tensor]:

        out_list = [epoch]
        # , train_loss, val_loss]
        for k in self.loss_keys:
            out_list += [train_loss[k], val_loss[k]]
        for k in self.met_keys:
            out_list += [train_acc[k], val_acc[k]]
        if test_accs is not None:
            for t in test_accs:
                out_list += [t]
        assert len(out_list) == len(self.log_keys)
        return out_list

    @property
    def lr(self):
        return self.cfg['lr']

    @property
    def epoch(self):
        return self.cfg['epochs']

    def fit(self, epochs: int, lr: float, params_opt_dict: Optional[Dict] = None):
        "Main training loop"
        self.logger.info(self.cfg)
        mb = master_bar(range(epochs))
        self.optimizer = self.prepare_optimizer(params_opt_dict)
        self.lr_scheduler = self.prepare_scheduler(self.optimizer)
        # Loop over epochs
        mb.write(self.log_keys, table=True)
        exception = False
        met_to_use = None
        st_time = time.time()
        try:
            for epoch in mb:
                train_loss, train_acc = self.train_epoch(mb)
                valid_loss, valid_acc = self.validate(self.data.valid_dl, mb)

                valid_loss_to_use = valid_loss[self.loss_keys[0]]
                valid_acc_to_use = valid_acc[self.met_keys[0]]
                test_accs = None
                if self.cfg['test_at_runtime']:
                    test_accs = self.do_test(mb)
                if 'sfn' in self.cfg and self.cfg['sfn'] == 'ReduceLROnPlateau':
                    # self.lr_scheduler.step(valid_loss_to_use)
                    self.lr_scheduler.step(valid_acc_to_use)
                else:
                    self.lr_scheduler.step()

                to_write = self.prepare_to_write(
                    epoch, train_loss, valid_loss,
                    train_acc, valid_acc, test_accs)
                mb.write([str(stat) if isinstance(stat, int)
                          else f'{stat:.4f}' for stat in to_write], table=True)
                self.update_log_file(
                    good_format_stats(self.log_keys, to_write))
                met_to_use = valid_acc[self.met_keys[0]]
                if self.best_met < met_to_use:
                    self.best_met = met_to_use
                    self.save_model_dict()
        except Exception as e:
            exception = e
            raise e
        finally:
            end_time = time.time()
            self.update_log_file(
                f'epochs done {epoch}. Exited due to exception {exception}. Total time taken {end_time - st_time: 0.4f}')

            if met_to_use:
                if self.best_met < met_to_use:
                    self.save_model_dict()

    def prepare_optimizer(self, params=None):
        "Prepare a normal optimizer"
        if not params:
            params = self.mdl.parameters()
        opt = self.opt_fn(params, lr=self.lr)
        return opt

    def prepare_scheduler(self, opt: torch.optim):
        "Prepares a LR scheduler on top of optimizer"
        if 'sfn' in self.cfg:
            sfn = self.cfg['sfn']
            lr_sched = getattr(torch.optim.lr_scheduler,
                               self.cfg['sfn'])
        else:
            lr_sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda epoch: 1)
            # lr_sched = torch.optim.lr_scheduler.StepLR(
            # opt, step_size=10, gamma=0.1)
            # lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            # opt, )

            # self.cfg['sfn'] = 'ReduceLROnPlateau'
            # lr_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            #     opt, patience=2, mode='max', factor=0.5, verbose=True)

        return lr_sched

    def overfit_batch(self, epochs: int, lr: float):
        "Sanity check to see if model overfits on a batch"
        batch = next(iter(self.data.train_dl))
        for b in batch.keys():
            batch[b] = batch[b].to(self.device)
        self.mdl.train()
        opt = self.prepare_optimizer(epochs, lr)

        for i in range(1000):
            opt.zero_grad()
            out = self.mdl(batch)
            loss = self.loss_fn(out, batch)
            loss.backward()
            opt.step()
            met = self.eval_fn(out, batch)
            print(f'Iter {i} | loss {loss: 0.4f} | acc {met: 0.4f}')
