"""
Model file for qnet
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
    """Create a `conv2d` layer with `nn.ReLU` activation and optional(`bn`) `nn.BatchNorm2d`: `ni` input, `nf` out
    filters, `ks` kernel, `stride`:stride, `padding`:padding, `bn`: batch normalization."""
    layers = [conv2d(ni, nf, ks=ks, stride=stride,
                     padding=padding, bias=bias), nn.ReLU(inplace=True)]
    if bn:
        layers.append(nn.BatchNorm2d(nf))
    return nn.Sequential(*layers)


class BackBone(nn.Module):
    """
    A general purpose Backbone class.
    """

    def __init__(self, encoder: nn.Module, cfg: dict, out_chs=256):
        """
        Make required forward hooks
        """
        super().__init__()
        self.device = torch.device('cuda')
        self.encoder = encoder
        self.num_chs = self.num_channels()
        self.fpn = FPN_backbone(self.num_chs, cfg, feat_size=out_chs)
        self.cfg = cfg
        self.encode_feats = self.encode_feats_default

    def num_channels(self):
        return [self.encoder.layer2[-1].conv3.out_channels,
                self.encoder.layer3[-1].conv3.out_channels,
                self.encoder.layer4[-1].conv3.out_channels]

    def concat_we(self, x, we, only_we=False, only_grid=False):
        """
        Convenience function to concat we
        Expects x in the form B x C x H x W
        we: B x wdim
        """
        grid = create_grid((x.size(2), x.size(3)),
                           flatten=False).to(self.device)
        grid = grid.permute(2, 0, 1).contiguous()
        grid_tile = grid.view(1, grid.size(0), grid.size(1), grid.size(2)).expand(
            we.size(0), grid.size(0), grid.size(1), grid.size(2))
        word_emb_tile = we.view(we.size(0), we.size(1),
                                1, 1).expand(we.size(0),
                                             we.size(1),
                                             x.size(2), x.size(3))
        if not only_we and not only_grid:
            return torch.cat((x, word_emb_tile, grid_tile), dim=1)
        elif only_we:
            # return torch.cat((word_emb_tile, grid_tile), dim=1)
            return word_emb_tile
        elif only_grid:
            return grid_tile

    def encode_feats_default(self, inp):
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

    def forward(self, inp, we=None, additional=None,
                only_feats=False, only_we=False, only_grid=False):
        """
        expecting word embedding of shape B x WE
        """
        feats = self.encode_feats_default(inp)
        if self.cfg['do_norm']:
            feats = [
                feat / feat.norm(dim=1).unsqueeze(1).expand(*feat.shape) for feat in feats]
        if we is None:
            return feats
        else:
            if self.cfg['do_norm_feats']:
                b, wdim = we.shape
                we = we / we.norm(dim=1).unsqueeze(1).expand(b, wdim)

            if not only_feats:
                out = [self.concat_we(
                    f, we, only_we=only_we, only_grid=only_grid) for f in feats]
            else:
                out = feats
            if additional:
                new_func = getattr(self, additional)
                out2 = [new_func(f, we) for f in feats]
                out = [out, out2]
            return out


class ZSGNet(nn.Module):
    """
    The main model
    Uses SSD like architecture
    """

    def __init__(self, backbone, n_anchors=1, final_bias=0., cfg=None):
        super().__init__()
        self.backbone = backbone
        # Assume the output from each
        # component of backbone will have 256 channels
        self.device = torch.device('cuda')
        if cfg is None:
            self.cfg = {"emb_dim": 300,
                        "use_bidirectional": True, "lstm_dim": 256}
        else:
            self.cfg = cfg
        self.n_anchors = n_anchors
        if self.cfg['use_model'] == 'retina_pretrained':
            self.n_anchors = 9

        self.emb_dim = cfg['emb_dim']
        self.bid = cfg['use_bidirectional']
        self.lstm_dim = cfg['lstm_dim']
        self.lstm_out_dim = self.lstm_dim * (self.bid + 1)
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
        # elif self.cfg['use_lang']:
        # self.start_dim_head = self.lstm_dim*(self.bid+1)

        if self.cfg['use_model'] == 'retina'\
                or self.cfg['use_model'] == 'retina_pretrained' \
                or self.cfg['use_model'] == 'ssd_vgg':
            if self.cfg['use_same_atb']:
                bias = torch.zeros(5 * self.n_anchors)
                bias[torch.arange(4, 5 * self.n_anchors, 5)] = -4
                self.att_reg_box = self._head_subnet(
                    5, self.n_anchors, final_bias=bias, start_dim_head=self.start_dim_head)
            else:
                self.att_box = self._head_subnet(
                    1, self.n_anchors, -4., start_dim_head=self.start_dim_head)
                self.reg_box = self._head_subnet(
                    4, self.n_anchors, start_dim_head=self.start_dim_head)
        elif self.cfg['use_model'] == 'yolo':
            pass

        self.lstm = nn.LSTM(self.emb_dim, self.lstm_dim,
                            bidirectional=self.bid, batch_first=False)
        self.after_init()

    def after_init(self):
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

        masks = (qlens1-1).view(1, -1, 1).expand(max_qlen,
                                                 lstm_out.size(1), lstm_out.size(2))
        qvec_sorted = lstm_out.gather(0, masks.long())[0]

        qvec_out = word_embs.new_zeros(qvec_sorted.shape)
        qvec_out[perm_idx] = qvec_sorted
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
        """
        inp0 = inp['img']
        inp1 = inp['qvec']
        qlens = inp['qlens']
        max_qlen = int(qlens.max().item())
        req_embs = inp1[:, :max_qlen, :].contiguous()

        req_emb = self.apply_lstm(req_embs, qlens, max_qlen)

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
            # see full
            else:
                feat_out = self.backbone(inp0, req_emb)
            # import pdb
            # pdb.set_trace()
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
        elif self.cfg['use_model'] == 'yolo':
            feat_out = self.backbone(inp0, req_emb)
            att_bbx_out = torch.cat([self.permute_correctly(
                feature, 85) for feature in feat_out], dim=1)
            att_out = att_bbx_out[..., [4]]
            bbx_out = att_bbx_out[..., :4]

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

        # if self.cfg['use_model'] == 'yolo' or \
        # self.cfg['use_model'] == 'retina' or \
        # self.cfg['use_model'] == 'ssd_vgg':
        # return att_out, bbx_out, feat_sizes, num_f_out, locp
        out_dict = {}
        out_dict['att_out'] = att_out
        out_dict['bbx_out'] = bbx_out
        out_dict['feat_sizes'] = feat_sizes
        out_dict['num_f_out'] = num_f_out

        if self.cfg['use_model'] == 'retina_pretrained':
            out_dict['ret_pret_reg'] = x1y1x2y2_to_y1x1y2x2(reg)
            out_dict['ret_pret_cls'] = cls
            # return (att_out, bbx_out, feat_sizes, num_f_out, cls,
            # x1y1x2y2_to_y1x1y2x2(reg), locp)

        return out_dict


def get_default_net(num_anchors=1, cfg=None):
    if cfg['use_model'] == 'retina':
        # if cfg['use_default_encoder']:
        encoder = tvm.resnet50(True)
        backbone = BackBone(encoder, cfg)

    elif cfg['use_model'] == 'ssd_vgg':
        backbone = ssd_vgg.build_ssd('train', cfg=cfg)
        backbone.vgg.load_state_dict(
            torch.load('./weights/vgg16_reducedfc.pth'))
        print('loaded pretrained vgg backbone')

    qnet = ZSGNet(backbone, num_anchors, cfg=cfg)
    return qnet


if __name__ == '__main__':
    # torch.manual_seed(0)
    cfg = conf
    data = get_data(cfg, ds_name='refclef')

    zsg_net = get_default_net()
    batch = next(iter(data.train_dl))
    out = zsg_net(batch)
