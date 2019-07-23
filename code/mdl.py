"""
Model file for zsgnet
Author: Arka Sadhu
"""
import torch
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.models as tvm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from fpn_resnet import FPN_backbone
from anchors import x1y1x2y2_to_y1x1y2x2, create_grid
import ssd_vgg
from typing import Dict, Any
from extended_config import cfg as conf
from dat_loader import get_data


# conv2d, conv2d_relu are adapted from
# https://github.com/fastai/fastai/blob/5c4cefdeaf11fdbbdf876dbe37134c118dca03ad/fastai/layers.py#L98
def conv2d(ni: int, nf: int, ks: int = 3, stride: int = 1,
           padding: int = None, bias=False) -> nn.Conv2d:
    "Create and initialize `nn.Conv2d` layer. `padding` defaults to `ks//2`."
    if padding is None:
        padding = ks//2
    return nn.Conv2d(ni, nf, kernel_size=ks, stride=stride,
                     padding=padding, bias=bias)


def conv2d_relu(ni: int, nf: int, ks: int = 3, stride: int = 1, padding: int = None,
                bn: bool = False, bias: bool = False) -> nn.Sequential:
    """
    Create a `conv2d` layer with `nn.ReLU` activation
    and optional(`bn`) `nn.BatchNorm2d`: `ni` input, `nf` out
    filters, `ks` kernel, `stride`:stride, `padding`:padding,
    `bn`: batch normalization.
    """
    layers = [conv2d(ni, nf, ks=ks, stride=stride,
                     padding=padding, bias=bias), nn.ReLU(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)


class BackBone(nn.Module):
    """
    A general purpose Backbone class.
    For a new network, need to redefine:
    --> encode_feats
    Optionally after_init
    """

    def __init__(self, encoder: nn.Module, cfg: dict, out_chs=256):
        """
        Make required forward hooks
        """
        super().__init__()
        self.device = torch.device(cfg.device)
        self.encoder = encoder
        self.cfg = cfg
        self.out_chs = out_chs
        self.after_init()

    def after_init(self):
        pass

    def num_channels(self):
        raise NotImplementedError

    def concat_we(self, x, we, only_we=False, only_grid=False):
        """
        Convenience function to concat we
        Expects x in the form B x C x H x W (one feature map)
        we: B x wdim (the language vector)
        Output: concatenated word embedding and grid centers
        """
        # Both cannot be true
        assert not (only_we and only_grid)

        # Create the grid
        grid = create_grid((x.size(2), x.size(3)),
                           flatten=False).to(self.device)
        grid = grid.permute(2, 0, 1).contiguous()

        # TODO: Slightly cleaner implementation?
        grid_tile = grid.view(
            1, grid.size(0), grid.size(1), grid.size(2)).expand(
            we.size(0), grid.size(0), grid.size(1), grid.size(2))

        # In case we only need the grid
        # Basically, don't use any image/language information
        if only_grid:
            return grid_tile

        # Expand word embeddings
        word_emb_tile = we.view(
            we.size(0), we.size(1), 1, 1).expand(
                we.size(0), we.size(1), x.size(2), x.size(3))

        # In case performing image blind (requiring only language)
        if only_we:
            return word_emb_tile

        # Concatenate along the channel dimension
        return torch.cat((x, word_emb_tile, grid_tile), dim=1)

    def encode_feats(self, inp):
        raise NotImplementedError

    def forward(self, inp, we=None,
                only_we=False, only_grid=False):
        """
        expecting word embedding of shape B x WE.
        If only image features are needed, don't
        provide any word embedding
        """
        feats = self.encode_feats(inp)
        # If we want to do normalization of the features
        if self.cfg['do_norm']:
            feats = [
                feat / feat.norm(dim=1).unsqueeze(1).expand(*feat.shape)
                for feat in feats
            ]

        # For language blind setting, can directly return the features
        if we is None:
            return feats

        if self.cfg['do_norm']:
            b, wdim = we.shape
            we = we / we.norm(dim=1).unsqueeze(1).expand(b, wdim)

        out = [self.concat_we(
            f, we, only_we=only_we, only_grid=only_grid) for f in feats]

        return out


class RetinaBackBone(BackBone):
    def after_init(self):
        self.num_chs = self.num_channels()
        self.fpn = FPN_backbone(self.num_chs, cfg, feat_size=self.out_chs)

    def num_channels(self):
        return [self.encoder.layer2[-1].conv3.out_channels,
                self.encoder.layer3[-1].conv3.out_channels,
                self.encoder.layer4[-1].conv3.out_channels]

    def encode_feats(self, inp):
        x = self.encoder.conv1(inp)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        x = self.encoder.maxpool(x)
        x1 = self.encoder.layer1(x)
        x2 = self.encoder.layer2(x1)
        x3 = self.encoder.layer3(x2)
        x4 = self.encoder.layer4(x3)

        feats = self.fpn([x2, x3, x4])
        return feats


class SSDBackBone(BackBone):
    """
    ssd_vgg.py already implements encoder
    """

    def encode_feats(self, inp):
        return self.encoder(inp)


class ZSGNet(nn.Module):
    """
    The main model
    Uses SSD like architecture but for Lang+Vision
    """

    def __init__(self, backbone, n_anchors=1, final_bias=0., cfg=None):
        super().__init__()
        assert isinstance(backbone, BackBone)
        self.backbone = backbone

        # Assume the output from each
        # component of backbone will have 256 channels
        self.device = torch.device(cfg.device)

        self.cfg = cfg

        # should be len(ratios) * len(scales)
        self.n_anchors = n_anchors

        self.emb_dim = cfg['emb_dim']
        self.bid = cfg['use_bidirectional']
        self.lstm_dim = cfg['lstm_dim']

        # Calculate output dimension of LSTM
        self.lstm_out_dim = self.lstm_dim * (self.bid + 1)

        # Separate cases for language, image blind settings
        if self.cfg['use_lang'] and self.cfg['use_img']:
            self.start_dim_head = self.lstm_dim*(self.bid+1) + 256 + 2
        elif self.cfg['use_img'] and not self.cfg['use_lang']:
            # language blind
            self.start_dim_head = 256
        elif self.cfg['use_lang'] and not self.cfg['use_img']:
            # image blind
            self.start_dim_head = self.lstm_dim*(self.bid+1)
        else:
            # both image, lang blind
            self.start_dim_head = 2

        # If shared heads for classification, box regression
        # This is the config used in the paper
        if self.cfg['use_same_atb']:
            bias = torch.zeros(5 * self.n_anchors)
            bias[torch.arange(4, 5 * self.n_anchors, 5)] = -4
            self.att_reg_box = self._head_subnet(
                5, self.n_anchors, final_bias=bias,
                start_dim_head=self.start_dim_head
            )
        # This is not used. Kept for historical purposes
        else:
            self.att_box = self._head_subnet(
                1, self.n_anchors, -4., start_dim_head=self.start_dim_head)
            self.reg_box = self._head_subnet(
                4, self.n_anchors, start_dim_head=self.start_dim_head)

        self.lstm = nn.LSTM(self.emb_dim, self.lstm_dim,
                            bidirectional=self.bid, batch_first=False)
        self.after_init()

    def after_init(self):
        "Placeholder if any child class needs something more"
        pass

    def _head_subnet(self, n_classes, n_anchors, final_bias=0., n_conv=4, chs=256,
                     start_dim_head=256):
        """
        Convenience function to create attention and regression heads
        """
        layers = [conv2d_relu(start_dim_head, chs, bias=True)]
        layers += [conv2d_relu(chs, chs, bias=True) for _ in range(n_conv)]
        layers += [conv2d(chs, n_classes * n_anchors, bias=True)]
        layers[-1].bias.data.zero_().add_(final_bias)
        return nn.Sequential(*layers)

    def permute_correctly(self, inp, outc):
        """
        Basically square box features are flattened
        """
        # inp is features
        # B x C x H x W -> B x H x W x C
        out = inp.permute(0, 2, 3, 1).contiguous()
        out = out.view(out.size(0), -1, outc)
        return out

    def concat_we(self, x, we, append_grid_centers=True):
        """
        Convenience function to concat we
        Expects x in the form B x C x H x W
        we: B x wdim
        """
        b, wdim = we.shape
        we = we / we.norm(dim=1).unsqueeze(1).expand(b, wdim)
        word_emb_tile = we.view(we.size(0), we.size(1),
                                1, 1).expand(we.size(0),
                                             we.size(1),
                                             x.size(2), x.size(3))

        if append_grid_centers:
            grid = create_grid((x.size(2), x.size(3)),
                               flatten=False).to(self.device)
            grid = grid.permute(2, 0, 1).contiguous()
            grid_tile = grid.view(1, grid.size(0), grid.size(1), grid.size(2)).expand(
                we.size(0), grid.size(0), grid.size(1), grid.size(2))

            return torch.cat((x, word_emb_tile, grid_tile), dim=1)
        return torch.cat((x, word_emb_tile), dim=1)

    def lstm_init_hidden(self, bs):
        """
        Initialize the very first hidden state of LSTM
        Basically, the LSTM should be independent of this
        """
        if not self.bid:
            hidden_a = torch.randn(1, bs, self.lstm_dim)
            hidden_b = torch.randn(1, bs, self.lstm_dim)
        else:
            hidden_a = torch.randn(2, bs, self.lstm_dim)
            hidden_b = torch.randn(2, bs, self.lstm_dim)

        hidden_a = hidden_a.to(self.device)
        hidden_b = hidden_b.to(self.device)

        return (hidden_a, hidden_b)

    def apply_lstm(self, word_embs, qlens, max_qlen, get_full_seq=False):
        """
        Applies lstm function.
        word_embs: word embeddings, B x seq_len x 300
        qlen: length of the phrases
        Try not to fiddle with this function.
        IT JUST WORKS
        """
        # B x T x E
        bs, max_seq_len, emb_dim = word_embs.shape
        # bid x B x L
        self.hidden = self.lstm_init_hidden(bs)
        # B x 1, B x 1
        qlens1, perm_idx = qlens.sort(0, descending=True)
        # B x T x E (permuted)
        qtoks = word_embs[perm_idx]
        # T x B x E
        embeds = qtoks.permute(1, 0, 2).contiguous()
        # Packed Embeddings
        packed_embed_inp = pack_padded_sequence(
            embeds, lengths=qlens1, batch_first=False)
        # To ensure no pains with DataParallel
        # self.lstm.flatten_parameters()
        lstm_out1, (self.hidden, _) = self.lstm(packed_embed_inp, self.hidden)

        # T x B x L
        lstm_out, req_lens = pad_packed_sequence(
            lstm_out1, batch_first=False, total_length=max_qlen)

        # TODO: Simplify getting the last vector
        masks = (qlens1-1).view(1, -1, 1).expand(max_qlen,
                                                 lstm_out.size(1), lstm_out.size(2))
        qvec_sorted = lstm_out.gather(0, masks.long())[0]

        qvec_out = word_embs.new_zeros(qvec_sorted.shape)
        qvec_out[perm_idx] = qvec_sorted
        # if full sequence is needed for future work
        if get_full_seq:
            lstm_out_1 = lstm_out.transpose(1, 0).contiguous()
            return lstm_out_1
        return qvec_out.contiguous()

    def forward(self, inp: Dict[str, Any]):
        """
        Forward method of the model
        inp0 : image to be used
        inp1 : word embeddings, B x seq_len x 300
        qlens: length of phrases

        The following is performed:
        1. Get final hidden state features of lstm
        2. Get image feature maps
        3. Concatenate the two, specifically, copy lang features
        and append it to all the image feature maps, also append the
        grid centers.
        4. Use the classification, regression head on this concatenated features
        The matching with groundtruth is done in loss function and evaluation
        """
        inp0 = inp['img']
        inp1 = inp['qvec']
        qlens = inp['qlens']
        max_qlen = int(qlens.max().item())
        req_embs = inp1[:, :max_qlen, :].contiguous()

        req_emb = self.apply_lstm(req_embs, qlens, max_qlen)

        # TODO: use ssd via backbone to simplify code
        if self.cfg['use_model'] == 'retina':
            # image blind
            if self.cfg['use_lang'] and not self.cfg['use_img']:
                # feat_out = self.backbone(inp0)
                feat_out = self.backbone(inp0, req_emb, only_we=True)
                # feat_out = [f[:, 256:, :, :] for f in feat_out]
            # language blind
            elif self.cfg['use_img'] and not self.cfg['use_lang']:
                feat_out = self.backbone(inp0)
                # feat_out = self.backbone(inp0, req_emb, only_we=True)
            elif not self.cfg['use_img'] and not self.cfg['use_lang']:
                feat_out = self.backbone(inp0, req_emb, only_grid=True)
            # see full language + image (happens by default)
            else:
                feat_out = self.backbone(inp0, req_emb)

            if self.cfg['use_same_atb']:
                att_bbx_out = torch.cat([self.permute_correctly(
                    self.att_reg_box(feature), 5) for feature in feat_out], dim=1)
                att_out = att_bbx_out[..., [-1]]
                bbx_out = att_bbx_out[..., :-1]
            else:
                att_out = torch.cat(
                    [self.permute_correctly(self.att_box(feature), 1)
                     for feature in feat_out], dim=1)
                bbx_out = torch.cat(
                    [self.permute_correctly(self.reg_box(feature), 4)
                     for feature in feat_out], dim=1)

        ########################################################
        # For SSD300
        elif self.cfg['use_model'] == 'ssd_vgg':
            feats = self.backbone(inp0)
            if not all([not torch.any(torch.isnan(f)) for f in feats]):
                import pdb
                pdb.set_trace()
            feat_out = [self.concat_we(f, req_emb) for f in feats]
            if self.cfg['use_same_atb']:
                att_bbx_out = torch.cat([self.permute_correctly(
                    self.att_reg_box(feature), 5) for feature in feat_out], dim=1)

                att_out = att_bbx_out[..., [-1]]
                bbx_out = att_bbx_out[..., :-1]
            else:
                att_out = torch.cat(
                    [self.permute_correctly(self.att_box(feature), 1)
                     for feature in feat_out], dim=1)
                bbx_out = torch.cat(
                    [self.permute_correctly(self.reg_box(feature), 4)
                     for feature in feat_out], dim=1)

        feat_sizes = torch.Tensor([[f.size(2), f.size(3)]
                                   for f in feat_out]).to(self.device)

        num_f_out = torch.Tensor([len(feat_out)]).to(self.device)

        out_dict = {}
        out_dict['att_out'] = att_out
        out_dict['bbx_out'] = bbx_out
        out_dict['feat_sizes'] = feat_sizes
        out_dict['num_f_out'] = num_f_out

        return out_dict


def get_default_net(num_anchors=1, cfg=None):
    if cfg['use_model'] == 'retina':
        encoder = tvm.resnet50(True)
    elif cfg['use_model'] == 'ssd_vgg':
        encoder = ssd_vgg.build_ssd('train', cfg=cfg)
        encoder.vgg.load_state_dict(
            torch.load('./weights/vgg16_reducedfc.pth'))
        print('loaded pretrained vgg backbone')

    backbone = BackBone(encoder, cfg)
    qnet = ZSGNet(backbone, num_anchors, cfg=cfg)
    return qnet


if __name__ == '__main__':
    # torch.manual_seed(0)
    cfg = conf
    data = get_data(cfg, ds_name='refclef')

    zsg_net = get_default_net()
    batch = next(iter(data.train_dl))
    out = zsg_net(batch)
