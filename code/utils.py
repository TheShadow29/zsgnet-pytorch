"""
Utility functions
"""
from typing import Dict, List, Optional, Union, Any, Callable
import torch
from torch import nn
from torch.utils.data import DataLoader
from dataclasses import dataclass
from pathlib import Path
import sys
import re
import numpy as np
from collections import Counter
from tqdm import tqdm
import time
import shutil
import json
from fastprogress.fastprogress import master_bar, progress_bar
import logging
import pickle
from torch.utils.tensorboard import SummaryWriter
from torch import distributed as dist
from yacs.config import CfgNode as CN


def update_from_dict(cfg: CN, dct: Dict[str, Any],
                     key_maps: Dict[str, str] = None) -> CN:
    """
    Given original CfgNode (cfg) and input dictionary allows changing
    the cfg with the updated dictionary values
    Optional key_maps argument which defines a mapping between
    same keys of the cfg node. Only used for convenience
    Adapted from:
    https://github.com/rbgirshick/yacs/blob/master/yacs/config.py#L219
    """
    # Original cfg
    root = cfg

    # Change the input dictionary using keymaps
    # Now it is aligned with the cfg
    full_key_list = list(dct.keys())
    for full_key in full_key_list:
        if full_key in key_maps:
            cfg[full_key] = dct[full_key]
            new_key = key_maps[full_key]
            dct[new_key] = dct.pop(full_key)

    # Convert the cfg using dictionary input
    for full_key, v in dct.items():
        if root.key_is_deprecated(full_key):
            continue
        if root.key_is_renamed(full_key):
            root.raise_key_rename_error(full_key)
        key_list = full_key.split(".")
        d = cfg
        for subkey in key_list[:-1]:
            # Most important statement
            assert subkey in d, f'key {full_key} doesnot exist'
            d = d[subkey]

        subkey = key_list[-1]
        # Most important statement
        assert subkey in d, f'key {full_key} doesnot exist'

        value = cfg._decode_cfg_value(v)

        assert isinstance(value, type(d[subkey]))
        d[subkey] = value

    return cfg


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def exec_func_if_main_proc(func: Callable):
    def wrapper(*args, **kwargs):
        if is_main_process():
            func(*args, **kwargs)
    return wrapper


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
    device: torch.device = torch.device('cuda')

    def __post_init__(self):
        "Setup log file, load model if required"
        # ToDo: shift this to DataWrap and make it Dict
        # Makes the test file correspondence explicit
        if not isinstance(self.data.test_dl, list):
            self.data.test_dl = [self.data.test_dl]

        self.init_log_dirs()

        self.prepare_log_keys()

        self.prepare_log_file()

        self.logger = self.init_logger()

        # Set the number of iterations, epochs, best_met to 0.
        # Updated in loading if required
        self.num_it = 0
        self.num_epoch = 0
        self.best_met = 0

        # Resume if given a path
        if self.cfg['resume']:
            self.load_model_dict(
                resume_path=self.cfg['resume_path'],
                load_opt=self.cfg['load_opt'])

        # self.writer.add_text(tag='cfg', text_string=json.dumps(self.cfg),
            # global_step=self.num_epoch)

    def init_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        if not is_main_process():
            return logger
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s %(name)s %(levelname)s: %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        fh = logging.FileHandler(str(self.extra_logger_file))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # logging.basicConfig(
        #     filename=self.extra_logger_file,
        #     filemode='a',
        #     format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        #     datefmt='%m/%d/%Y %H:%M:%S',
        #     level=logging.INFO
        # )
        return logger

    def init_log_dirs(self):
        """
        Convenience function to create the following:
        1. Log dir to store log file in txt format
        2. Extra Log dir to store the logger output
        3. Tb log dir to store tensorboard files
        4. Model dir to save the model files
        5. Predictions dir to store the predictions of the saved model
        6. [Optional] Can add some 3rd party logger
        """
        # Saves the text logs
        self.txt_log_file = Path(
            self.data.path) / 'txt_logs' / f'{self.uid}.txt'

        # Saves the output of self.logger
        self.extra_logger_file = Path(
            self.data.path) / 'ext_logs' / f'{self.uid}.txt'

        # Saves SummaryWriter outputs
        self.tb_log_dir = Path(self.data.path) / 'tb_logs' / f'{self.uid}'

        # Saves the trained model
        self.model_file = Path(self.data.path) / 'models' / f'{self.uid}.pth'

        # Saves the output predictions
        self.predictions_dir = Path(
            self.data.path) / 'predictions' / f'{self.uid}'

        self.create_log_dirs()

    @exec_func_if_main_proc
    def create_log_dirs(self):
        """
        Creates the directories initialized in init_log_dirs
        """
        self.txt_log_file.parent.mkdir(exist_ok=True, parents=True)
        self.extra_logger_file.parent.mkdir(exist_ok=True)
        self.tb_log_dir.mkdir(exist_ok=True, parents=True)
        self.model_file.parent.mkdir(exist_ok=True)
        self.predictions_dir.mkdir(exist_ok=True, parents=True)

    def prepare_log_keys(self):
        """
        Creates the relevant keys to be logged.
        Mainly used by the txt logger to output in a good format
        """
        def _prepare_log_keys(keys_list: List[List[str]],
                              prefix: List[str]) -> List[str]:
            """
            Convenience function to create log keys
            keys_list: List[loss_keys, met_keys]
            prefix: List['trn', 'val']
            """
            log_keys = []
            for keys in keys_list:
                for key in keys:
                    log_keys += [f'{p}_{key}' for p in prefix]
            return log_keys

        self.loss_keys = self.loss_fn.loss_keys
        self.met_keys = self.eval_fn.met_keys

        # When writing Training and Validation together
        self.log_keys = ['epochs'] + _prepare_log_keys(
            [self.loss_keys, self.met_keys],
            ['trn', 'val']
        )

    @exec_func_if_main_proc
    def prepare_log_file(self):
        "Prepares the log files depending on arguments"
        f = self.txt_log_file.open('a')
        cfgtxt = json.dumps(self.cfg)
        f.write(cfgtxt)
        f.write('\n\n')
        f.write('  '.join(self.log_keys) + '\n')
        f.close()

    @exec_func_if_main_proc
    def update_log_file(self, towrite: str):
        "Updates the log files as and when required"
        with self.txt_log_file.open('a') as f:
            f.write(towrite + '\n')

    def get_predictions_list(self, predictions: Dict[str, List]) -> List[Dict]:
        "Converts dictionary of lists to list of dictionary"
        keys = list(predictions.keys())
        num_preds = len(predictions[keys[0]])
        out_list = [{k: predictions[k][ind] for k in keys}
                    for ind in range(num_preds)]
        return out_list

    def validate(self, db: Optional[DataLoader] = None,
                 mb=None) -> List[torch.tensor]:
        "Validation loop, done after every epoch"
        self.mdl.eval()
        if db is None:
            db = self.data.valid_dl

        predicted_box_dict_list = []
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
                prediction_dict = {
                    'id': metric['idxs'].tolist(),
                    'pred_boxes': metric['pred_boxes'].tolist(),
                    'pred_scores': metric['pred_scores'].tolist()
                }
                predicted_box_dict_list += self.get_predictions_list(
                    prediction_dict)
            nums = torch.tensor(nums).float()
            val_loss = compute_avg_dict(val_losses, nums)
            eval_metric = compute_avg_dict(eval_metrics, nums)
            return val_loss, eval_metric, predicted_box_dict_list

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

            # self.writer.add_scalar(
            #     tag='trn_loss', scalar_value=out_loss[self.loss_keys[0]],
            #     global_step=self.num_it)
            comment_to_print = f'LossB {loss: .4f} | SmLossB {trn_loss.smooth1: .4f} | AccB {trn_acc.smooth1: .4f}'
            mb.child.comment = comment_to_print
            if self.num_it % 2 == 0:
                self.logger.debug(f'Num_it {self.num_it} {comment_to_print}')
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

        if 'num_epoch' in checkpoint.keys():
            self.num_epoch = checkpoint['num_epoch']

        if 'best_met' in checkpoint.keys():
            self.best_met = checkpoint['best_met']

        if load_opt:
            self.optimizer = self.prepare_optimizer()
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint:
                self.lr_scheduler = self.prepare_scheduler()
                self.lr_scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])

    @exec_func_if_main_proc
    def save_model_dict(self):
        "Save the model and optimizer"
        # if is_main_process():
        checkpoint = {
            'model_state_dict': self.mdl.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'num_it': self.num_it,
            'num_epoch': self.num_epoch,
            'cfgtxt': json.dumps(self.cfg),
            'best_met': self.best_met
        }
        torch.save(checkpoint, self.model_file.open('wb'))

    @exec_func_if_main_proc
    def update_prediction_file(self, predictions, pred_file):
        pickle.dump(predictions, pred_file.open('wb'))

    # @exec_func_if_main_proc
    def prepare_to_write(
            self,
            train_loss: Dict[str, torch.tensor],
            train_acc: Dict[str, torch.tensor],
            val_loss: Dict[str, torch.tensor] = None,
            val_acc: Dict[str, torch.tensor] = None
    ) -> List[torch.tensor]:

        epoch = self.num_epoch
        out_list = [epoch]

        for k in self.loss_keys:
            out_list += [train_loss[k]]
            if val_loss is not None:
                out_list += [val_loss[k]]

        for k in self.met_keys:
            out_list += [train_acc[k]]
            if val_acc is not None:
                out_list += [val_acc[k]]

        assert len(out_list) == len(self.log_keys)
        return out_list

    @property
    def lr(self):
        return self.cfg['lr']

    @property
    def epoch(self):
        return self.cfg['epochs']

    @exec_func_if_main_proc
    def master_bar_write(self, mb, **kwargs):
        mb.write(**kwargs)

    def fit(self, epochs: int, lr: float,
            params_opt_dict: Optional[Dict] = None):
        "Main training loop"
        # Print logger at the start of the training loop
        self.logger.info(self.cfg)
        # Initialize the progress_bar
        mb = master_bar(range(epochs))
        # Initialize optimizer
        # Prepare Optimizer may need to be re-written as per use
        self.optimizer = self.prepare_optimizer(params_opt_dict)
        # Initialize scheduler
        # Prepare scheduler may need to re-written as per use
        self.lr_scheduler = self.prepare_scheduler(self.optimizer)

        # Write the top row display
        # mb.write(self.log_keys, table=True)
        self.master_bar_write(mb, line=self.log_keys, table=True)
        exception = False
        met_to_use = None
        # Keep record of time until exit
        st_time = time.time()
        try:
            # Loop over epochs
            for epoch in mb:
                self.num_epoch += 1
                train_loss, train_acc = self.train_epoch(mb)

                valid_loss, valid_acc, predictions = self.validate(
                    self.data.valid_dl, mb)

                valid_acc_to_use = valid_acc[self.met_keys[0]]
                # Depending on type
                self.scheduler_step(valid_acc_to_use)

                # Now only need main process
                # Decide to save or not
                met_to_use = valid_acc[self.met_keys[0]]
                if self.best_met < met_to_use:
                    self.best_met = met_to_use
                    self.save_model_dict()
                    self.update_prediction_file(
                        predictions,
                        self.predictions_dir / f'val_preds_{self.uid}.pkl')

                # Prepare what all to write
                to_write = self.prepare_to_write(
                    train_loss, train_acc,
                    valid_loss, valid_acc
                )

                # Display on terminal
                assert to_write is not None
                mb_write = [str(stat) if isinstance(stat, int)
                            else f'{stat:.4f}' for stat in to_write]
                self.master_bar_write(mb, line=mb_write, table=True)

                # for k, record in zip(self.log_keys, to_write):
                #     self.writer.add_scalar(
                #         tag=k, scalar_value=record, global_step=self.num_epoch)
                # Update in the log file
                self.update_log_file(
                    good_format_stats(self.log_keys, to_write))

        except Exception as e:
            exception = e
            raise e
        finally:
            end_time = time.time()
            self.update_log_file(
                f'epochs done {epoch}. Exited due to exception {exception}. '
                f'Total time taken {end_time - st_time: 0.4f}\n\n'
            )
            # Decide to save finally or not
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
        self.sched_using_val_metric = False
        if 'sfn' in self.cfg:
            self.sched_using_val_metric = self.cfg['sfn'] == 'ReduceLROnPlateau'
            lr_sched = getattr(torch.optim.lr_scheduler,
                               self.cfg['sfn'])
        else:
            lr_sched = torch.optim.lr_scheduler.LambdaLR(
                opt, lambda epoch: 1)

        return lr_sched

    def scheduler_step(self, val_metric):
        if self.sched_using_val_metric:
            self.lr_scheduler.step(val_metric)
        else:
            self.lr_scheduler.step()
        return

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
