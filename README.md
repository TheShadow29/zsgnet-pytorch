# zsgnet-pytorch
This is the official repository for ICCV19 oral paper [Zero-Shot Grounding of Objects from Natural Language Queries](https://arxiv.org/abs/1908.07129). It contains the code and the datasets to reproduce the numbers for our model ZSGNet in the paper. 

The code has been refactored from the original implementation and now supports Distributed learning (see [pytorch docs](https://pytorch.org/docs/stable/nn.html#torch.nn.parallel.DistributedDataParallel)) for significantly faster training (around 4x speedup from pytorch [Dataparallel](https://pytorch.org/docs/stable/nn.html#dataparallel))

The code is fairly easy to use and extendable for future work. Feel free to open an issue in case of queries.

## ToDo
- [ ] Add colab demo.
- [ ] Add installation guide.
- [ ] Pretrained models
- [ ] Add hubconfig 
- [ ] Add tensorboard

## Install
(Coming soon!)

## Data Preparation
Look at DATA_README.MD for data-prepration.

## Training
Basic usage is `python code/main_dist.py "experiment_name" --arg1=val1 --arg2=val2` and the arg1, arg2 can be found in `configs/cfg.yaml`. This trains using the DataParallel mode.

For distributed learning use `python -m torch.distributed.launch --nproc_per_node=$ngpus code/main_dist.py` instead. This trains using the DistributedDataParallel mode. (Also see [caveat in using distributed training](#caveats-in-distributeddataparallel) below)

An example to train on ReferIt dataset (note you must have prepared referit dataset) would be:

```
python code/main_dist.py "referit_try" --ds_to_use='refclef' --bs=16 --nw=4
```

Similarly for distributed learning (need to set npgus as the number of gpus)
```
python -m torch.distributed.launch --nproc_per_node=$npgus code/main_dist.py "referit_try" --ds_to_use='refclef' --bs=16 --nw=4
```

### Logging
Logs are stored inside `tmp/` directory. When you run the code with $exp_name the following are stored:
- `txt_logs/$exp_name.txt`: the config used and the training, validation losses after ever epoch.
- `models/$exp_name.pth`: the model, optimizer, scheduler, accuracy, number of epochs and iterations completed are stored. Only the best model upto the current epoch is stored.
- `ext_logs/$exp_name.txt`: this uses the `logging` module of python to store the `logger.debug` outputs printed. Mainly used for debugging.
- `tb_logs/$exp_name`: this is still wip, right now just creates a directory and nothing more, ideally want to support the tensorboard logs.
- `predictions`: the validation outputs of current best model.

## Evaluation
There are two ways to evaluate. 

1. For validation, it is already computed in the training loop. If you just want to evaluate on validation or testing on a model trained previously ($exp_name) you can do:
```
python code/main_dist.py $exp_name --ds_to_use='refclef' --resume=True --only_valid=True --only_test=True
```
or you can use a different experiment name as well and pass `--resume_path` argument like:
```
python code/main_dist.py $exp_name --ds_to_use='refclef' --resume=True --resume_path='./tmp/models/referit_try.pth' 
```
After this, the logs would be available inside `tmp/txt_logs/$exp_name.txt`

2. If you have some other model, you can output the predictions in the following structure into a pickle file say `predictions.pkl`:
```
[
    {'id': annotation_id,
 	'pred_boxes': [x1,y1,x2,y2]},
    .
    .
    .
]
```

Then you can evaluate using `code/eval_script.py` using:
```
python code/eval_script.py predictions_file gt_file
```
For referit it would be
```
python code/eval_script.py ./tmp/predictions/$exp_name/val_preds_$exp_name.pkl ./data/referit/csv_dir/val.csv
```

### Caveats in DistributedDataParallel
When training using DDP, there is no easy way to get all the validation outputs into one process (that works only for tensors). As a result one has to save the predictions of each separate process and then read again to combine them in the main process. Current implementation doesn't do this for simplicity, as a result the validation results obtained during training are slight different from the actual results. 

To get the correct results, one can follow the steps in [Evaluation](#evaluation) as is (the point to note is **NOT** use `torch.distributed.launch` for evaluation). Thus, you would get correct results when using simply dataparallel.


## Pre-trained Models
(Coming soon!)

# Acknowledgements
We thank:
1. [@yhenon](https://github.com/yhenon) for their repository on retina-net (https://github.com/yhenon/pytorch-retinanet).
1. [@amdegroot](https://github.com/amdegroot) for their repsository on ssd using vgg (https://github.com/amdegroot/ssd.pytorch)
1. [fastai](https://github.com/fastai/fastai) repository for helpful logging, anchor box generation and convolution functions.

# Citation

If you find the code or dataset useful, please cite us:

```
@inproceedings{sadhu2017zero,
author = {Sadhu, Arka and Chen, Kan and Nevatia, Ram}, 
title = {Zero-Shot Grounding of Objects from Natural Language Queries},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
year = {2019} 
}
```

