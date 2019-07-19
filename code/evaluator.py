import torch
from torch import nn
from anchors import (create_anchors, reg_params_to_bbox,
                     IoU_values, x1y1x2y2_to_y1x1y2x2)
from typing import Dict
from functools import partial
# from simple_utils import (
# rcrc2tlbr, get_pil_images_from_batch, save_img_with_boxes_easy)


def reshape(box, new_size):
    """
    box: (N, 4) in y1x1y2x2 format
    new_size: (N, 2) stack of (h, w)
    """
    box[:, :2] = new_size * box[:, :2]
    box[:, 2:] = new_size * box[:, 2:]
    return box


class Evaluator(nn.Module):
    """
    To get the accuracy. Operates at training time.
    """

    def __init__(self, ratios, scales, cfg):
        super().__init__()
        self.cfg = cfg

        self.ratios = ratios
        self.scales = scales

        self.alpha = cfg['alpha']
        self.gamma = cfg['gamma']
        self.use_focal = cfg['use_focal']
        self.use_softmax = cfg['use_softmax']
        self.use_multi = cfg['use_multi']

        self.lamb_reg = cfg['lamb_reg']

        self.met_keys = ['Acc', 'MaxPos']
        self.anchs = None
        self.get_anchors = partial(
            create_anchors, ratios=self.ratios,
            scales=self.scales, flatten=True)

        self.acc_iou_threshold = self.cfg['acc_iou_threshold']

    def forward(self, out: Dict[str, torch.tensor],
                inp: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:

        annot = inp['annot']
        att_box = out['att_out']
        reg_box = out['bbx_out']
        feat_sizes = out['feat_sizes']
        num_f_out = out['num_f_out']

        device = att_box.device

        if len(num_f_out) > 1:
            num_f_out = int(num_f_out[0].item())
        else:
            num_f_out = int(num_f_out.item())

        feat_sizes = feat_sizes[:num_f_out, :]

        if self.anchs is None:
            feat_sizes = feat_sizes[:num_f_out, :]
            anchs = self.get_anchors(feat_sizes)
            anchs = anchs.to(device)
            self.anchs = anchs
        else:
            anchs = self.anchs

        att_box_sigmoid = torch.sigmoid(att_box).squeeze(-1)
        att_box_best, att_box_best_ids = att_box_sigmoid.max(1)
        # self.att_box_best = att_box_best

        ious1 = IoU_values(annot, anchs)
        gt_mask, expected_best_ids = ious1.max(1)

        actual_bbox = reg_params_to_bbox(
            anchs, reg_box)

        best_possible_result, _ = self.get_eval_result(
            actual_bbox, annot, expected_best_ids)

        msk = None
        actual_result, pred_boxes = self.get_eval_result(
            actual_bbox, annot, att_box_best_ids, msk)

        out_dict = {}
        out_dict['Acc'] = actual_result
        out_dict['MaxPos'] = best_possible_result
        out_dict['idxs'] = inp['idxs']

        reshaped_boxes = x1y1x2y2_to_y1x1y2x2(reshape(
            (pred_boxes + 1)/2, (inp['img_size'])))
        out_dict['pred_boxes'] = reshaped_boxes
        out_dict['pred_scores'] = att_box_best
        # orig_annot = inp['orig_annot']
        # Sanity check
        # iou1 = (torch.diag(IoU_values(reshaped_boxes, orig_annot))
        #         >= self.acc_iou_threshold).float().mean()
        # assert actual_result.item() == iou1.item()
        return out_dict

    def get_eval_result(self, actual_bbox, annot, ids_to_use, msk=None):
        best_boxes = torch.gather(
            actual_bbox, 1, ids_to_use.view(-1, 1, 1).expand(-1, 1, 4))
        best_boxes = best_boxes.view(best_boxes.size(0), -1)
        if msk is not None:
            best_boxes[msk] = 0
        # self.best_boxes = best_boxes
        ious = torch.diag(IoU_values(best_boxes, annot))
        # self.fin_results = ious
        return (ious >= self.acc_iou_threshold).float().mean(), best_boxes
