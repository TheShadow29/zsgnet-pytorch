"""
Creates anchor based on the backbone
Based on code from https://github.com/fastai/fastai_docs/blob/master/dev_nb/102a_coco.ipynb
Author: Arka Sadhu
"""
import torch
import numpy as np
from torch import nn


def cthw2tlbr(boxes):
    "Convert center/size format `boxes` to top/left bottom/right corners."
    top_left = boxes[..., :2] - boxes[..., 2:]/2
    bot_right = boxes[..., :2] + boxes[..., 2:]/2
    return torch.cat([top_left, bot_right], dim=-1)


def tlbr2cthw(boxes):
    "Convert top/left bottom/right format `boxes` to center/size corners."
    center = (boxes[..., :2] + boxes[..., 2:])/2
    sizes = boxes[..., 2:] - boxes[..., :2]
    return torch.cat([center, sizes], dim=-1)


def tlbr2tlhw(boxes):
    "Convert tl br format `boxes` to tl hw format"
    top_left = boxes[:, :2]
    height_width = boxes[:, 2:] - boxes[:, :2]
    return torch.cat([top_left, height_width], 1)


def tlhw2tlbr(boxes):
    "Convert tl br format `boxes` to tl hw format"
    top_left = boxes[..., :2]
    bottom_right = boxes[..., 2:] + boxes[..., :2]
    return torch.cat([top_left, bottom_right], -1)


def x1y1x2y2_to_y1x1y2x2(boxes):
    "Convert xy boxes to yx boxes and vice versa"
    box_tmp = boxes.clone()
    box_tmp[..., 0], box_tmp[..., 1] = boxes[..., 1], boxes[..., 0]
    box_tmp[..., 2], box_tmp[..., 3] = boxes[..., 3], boxes[..., 2]
    return box_tmp


def create_grid(size, flatten=True):
    "Create a grid of a given `size`."
    if isinstance(size, tuple):
        H, W = size
    else:
        H, W = size, size

    grid = torch.FloatTensor(H, W, 2)
    linear_points = torch.linspace(-1+1/W, 1-1/W,
                                   W) if W > 1 else torch.tensor([0.])
    grid[:, :, 1] = torch.ger(torch.ones(
        H), linear_points).expand_as(grid[:, :, 0])
    linear_points = torch.linspace(-1+1/H, 1-1/H,
                                   H) if H > 1 else torch.tensor([0.])
    grid[:, :, 0] = torch.ger(
        linear_points, torch.ones(W)).expand_as(grid[:, :, 1])
    return grid.view(-1, 2) if flatten else grid


def create_anchors(sizes, ratios, scales, flatten=True, device=torch.device('cuda')):
    "Create anchor of `sizes`, `ratios` and `scales`."
    # device = torch.device('cuda')
    aspects = [[[s*np.sqrt(r), s*np.sqrt(1/r)]
                for s in scales] for r in ratios]
    aspects = torch.tensor(aspects).to(device).view(-1, 2)
    anchors = []
    for h, w in sizes:
        if type(h) == torch.Tensor:
            h = int(h.item())
            w = int(w.item())

        sized_aspects = (
            aspects * torch.tensor([2/h, 2/w]).to(device)).unsqueeze(0)
        base_grid = create_grid((h, w)).to(device).unsqueeze(1)
        n, a = base_grid.size(0), aspects.size(0)
        ancs = torch.cat([base_grid.expand(n, a, 2),
                          sized_aspects.expand(n, a, 2)], 2)
        anchors.append(ancs.view(h, w, a, 4))
    anchs = torch.cat([anc.view(-1, 4)
                       for anc in anchors], 0) if flatten else anchors
    return cthw2tlbr(anchs) if flatten else anchors


def intersection(anchors, targets):
    """
    Compute the sizes of the intersections of `anchors` by `targets`.
    Assume both anchors and targets are in tl br format
    """
    ancs, tgts = anchors, targets
    a, t = ancs.size(0), tgts.size(0)
    ancs, tgts = ancs.unsqueeze(1).expand(
        a, t, 4), tgts.unsqueeze(0).expand(a, t, 4)
    top_left_i = torch.max(ancs[..., :2], tgts[..., :2])
    bot_right_i = torch.min(ancs[..., 2:], tgts[..., 2:])

    sizes = torch.clamp(bot_right_i - top_left_i, min=0)
    return sizes[..., 0] * sizes[..., 1]


def IoU_values(anchors, targets):
    """
    Compute the IoU values of `anchors` by `targets`.
    Expects both in tlbr format
    """
    inter = intersection(anchors, targets)
    ancs, tgts = tlbr2cthw(anchors), tlbr2cthw(targets)
    anc_sz, tgt_sz = ancs[:, 2] * \
        ancs[:, 3], tgts[:, 2] * tgts[:, 3]
    union = anc_sz.unsqueeze(1) + tgt_sz.unsqueeze(0) - inter
    return inter/(union+1e-8)


def simple_iou(box1, box2):
    """
    Simple iou between box1 and box2
    """
    def simple_inter(ancs, tgts):
        top_left_i = torch.max(ancs[..., :2], tgts[..., :2])
        bot_right_i = torch.min(ancs[..., 2:], tgts[..., 2:])
        sizes = torch.clamp(bot_right_i - top_left_i, min=0)
        return sizes[..., 0] * sizes[..., 1]

    inter = intersection(box1, box2)
    ancs, tgts = tlbr2tlhw(box1), tlbr2tlhw(box2)
    anc_sz, tgt_sz = ancs[:, 2] * \
        ancs[:, 3], tgts[:, 2] * tgts[:, 3]
    union = anc_sz + tgt_sz - inter
    return inter / (union + 1e-8)


def match_anchors(anchors, targets, match_thr=0.5, bkg_thr=0.4):
    """
    Match `anchors` to targets. -1 is match to background, -2 is ignore.
    """
    ious = IoU_values(anchors, targets)
    matches = anchors.new(anchors.size(0)).zero_().long() - 2
    vals, idxs = torch.max(ious, 1)
    matches[vals < bkg_thr] = -1
    matches[vals > match_thr] = idxs[vals > match_thr]
    # Overwrite matches with each target getting the anchor that has the max IoU.
    vals, idxs = torch.max(ious, 0)
    # If idxs contains repetition, this doesn't bug and only the last is considered.
    matches[idxs] = targets.new_tensor(list(range(targets.size(0)))).long()
    return matches


def simple_match_anchors(anchors, targets, match_thr=0.4, bkg_thr=0.1):
    """
    Match `anchors` to targets. -1 is match to background, -2 is ignore.
    Note here:
    anchors are fixed
    targets are from a batch
    """
    # ious = IoU_values(anchors, targets)
    ious = IoU_values(targets, anchors)
    matches = ious.new(ious.shape).zero_().long() - 2
    matches[ious < bkg_thr] = -1
    matches[ious > match_thr] = 1
    return matches


def bbox_to_reg_params(anchors, boxes):
    """
    Converts boxes to corresponding reg params
    Assume both in rchw format
    """
    boxes = tlbr2cthw(boxes)
    anchors = tlbr2cthw(anchors)
    anchors = anchors.expand(boxes.size(0), anchors.size(0), 4)
    boxes = boxes.unsqueeze(1)
    trc = (boxes[..., :2] - anchors[..., :2]) / (anchors[..., 2:] + 1e-8)
    thw = torch.log(boxes[..., 2:] / (anchors[..., 2:] + 1e-8))
    return torch.cat((trc, thw), 2)


def reg_params_to_bbox(anchors, boxes, std12=[1, 1]):
    """
    Converts reg_params to corresponding boxes
    Assume anchors in r1c1r2c2 format
    Boxes in standard form r*, c*, h*, w*
    """
    anc1 = anchors.clone()
    anc1 = tlbr2cthw(anc1)
    b1 = boxes[..., :2] * std12[0]
    a111 = anc1[..., 2:] * b1 + anc1[..., :2]

    b2 = boxes[..., 2:] * std12[1]
    a222 = torch.exp(b2) * anc1[..., 2:]
    af = torch.cat([a111, a222], dim=2)
    aft = cthw2tlbr(af)
    return aft
