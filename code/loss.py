import torch
from torch import nn
import torch.nn.functional as F
from anchors import (create_anchors, simple_match_anchors,
                     bbox_to_reg_params, IoU_values)
from typing import Dict
from functools import partial
# from utils import reduce_dict


class ZSGLoss(nn.Module):
    """
    Criterion to be minimized
    Requires the anchors to be used
    for loss computation
    """

    def __init__(self, ratios, scales, cfg):
        super().__init__()
        self.cfg = cfg

        self.ratios = ratios
        self.scales = scales

        self.alpha = cfg['alpha']
        self.gamma = cfg['gamma']

        # Which loss fucntion to use
        self.use_focal = cfg['use_focal']
        self.use_softmax = cfg['use_softmax']
        self.use_multi = cfg['use_multi']

        self.lamb_reg = cfg['lamb_reg']

        self.loss_keys = ['loss', 'cls_ls', 'box_ls']
        self.anchs = None
        self.get_anchors = partial(
            create_anchors, ratios=self.ratios,
            scales=self.scales, flatten=True)

        self.box_loss = nn.SmoothL1Loss(reduction='none')

    def forward(self, out: Dict[str, torch.tensor],
                inp: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        """
        inp: att_box, reg_box, feat_sizes
        annot: gt box (r1c1r2c2 form)
        """
        annot = inp['annot']
        att_box = out['att_out']
        reg_box = out['bbx_out']
        feat_sizes = out['feat_sizes']
        num_f_out = out['num_f_out']

        device = att_box.device

        # get the correct number of output features
        # in the case of DataParallel
        if len(num_f_out) > 1:
            num_f_out = int(num_f_out[0].item())
        else:
            num_f_out = int(num_f_out.item())

        # Computes Anchors only once since size is kept fixed
        # Needs to be changed in case size is not fixed
        if self.anchs is None:
            feat_sizes = feat_sizes[:num_f_out, :]
            anchs = self.get_anchors(feat_sizes)
            anchs = anchs.to(device)
            self.anchs = anchs
        else:
            anchs = self.anchs
        matches = simple_match_anchors(
            anchs, annot, match_thr=self.cfg['matching_threshold'])
        bbx_mask = (matches >= 0)
        ious1 = IoU_values(annot, anchs)
        _, msk = ious1.max(1)

        bbx_mask2 = torch.eye(anchs.size(0))[msk]
        bbx_mask2 = bbx_mask2 > 0
        bbx_mask2 = bbx_mask2.to(device)
        top1_mask = bbx_mask2

        if not self.use_multi:
            bbx_mask = bbx_mask2
        else:
            bbx_mask = bbx_mask | bbx_mask2

        # all clear
        gt_reg_params = bbox_to_reg_params(anchs, annot)
        box_l = self.box_loss(reg_box, gt_reg_params)
        # box_l_relv = box_l.sum(dim=2)[bbx_mask]
        box_l_relv = box_l.sum(dim=2) * bbx_mask.float()
        box_l_relv = box_l_relv.sum(dim=1) / bbx_mask.sum(dim=-1).float()
        box_loss = box_l_relv.mean()

        if box_loss.cpu() == torch.Tensor([float("Inf")]):
            # There is a likely bug with annot box
            # being very small
            import pdb
            pdb.set_trace()

        att_box = att_box.squeeze(-1)
        att_box_sigm = torch.sigmoid(att_box)

        if self.use_softmax:
            assert self.use_multi is False
            gt_ids = msk
            clas_loss = F.cross_entropy(att_box, gt_ids, reduction='none')
        else:
            if self.use_focal:
                encoded_tgt = bbx_mask.float()
                ps = att_box_sigm
                weights = encoded_tgt * (1-ps) + (1-encoded_tgt) * ps
                alphas = ((1-encoded_tgt) * self.alpha +
                          encoded_tgt * (1-self.alpha))
                weights.pow_(self.gamma).mul_(alphas)
                weights = weights.detach()
            else:
                weights = None

            clas_loss = F.binary_cross_entropy_with_logits(
                att_box, bbx_mask.float(), weight=weights, reduction='none')

        clas_loss = clas_loss.sum() / bbx_mask.sum()
        # clas_loss = clas_loss.sum() / clas_loss.size(0)

        if torch.isnan(box_loss) or torch.isnan(clas_loss):
            # print('Nan Loss')
            box_loss = box_loss.new_ones(box_loss.shape) * 0.01
            box_loss.requires_grad = True
            clas_loss = clas_loss.new_ones(clas_loss.shape)
            clas_loss.requires_grad = True

        out_loss = self.lamb_reg * box_loss + clas_loss
        # + self.lamb_rel * rel_loss
        out_dict = {}
        out_dict['loss'] = out_loss
        out_dict['cls_ls'] = clas_loss
        out_dict['box_ls'] = box_loss
        # out_dict['rel_ls'] = rel_loss

        return out_dict
        # return reduce_dict(out_dict)


def get_default_loss(ratios, scales, cfg):
    return ZSGLoss(ratios, scales, cfg)
