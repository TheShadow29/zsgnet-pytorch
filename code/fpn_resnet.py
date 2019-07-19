"""
Taken from the wonderful repository: https://github.com/yhenon/pytorch-retinanet/blob/master/model.py
"""

import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    """
    standard Basic block
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Standard Bottleneck block
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def pad_out(k):
    "padding to have same size"
    return (k-1)//2


class FPN_backbone(nn.Module):
    """
    A different fpn, doubt it will work
    """

    def __init__(self, inch_list, cfg, feat_size=256):
        super().__init__()

#         self.backbone = backbone

        # expects c3, c4, c5 channel dims
        self.inch_list = inch_list
        self.cfg = cfg
        c3_ch, c4_ch, c5_ch = self.inch_list
        self.feat_size = feat_size

        self.P7_2 = nn.Conv2d(in_channels=self.feat_size,
                              out_channels=self.feat_size, stride=2,
                              kernel_size=3,
                              padding=1)
        self.P6 = nn.Conv2d(in_channels=c5_ch,
                            out_channels=self.feat_size,
                            kernel_size=3, stride=2, padding=pad_out(3))
        self.P5_1 = nn.Conv2d(in_channels=c5_ch,
                              out_channels=self.feat_size,
                              kernel_size=1, padding=pad_out(1))

        self.P5_2 = nn.Conv2d(in_channels=self.feat_size, out_channels=self.feat_size,
                              kernel_size=3, padding=pad_out(3))

        self.P4_1 = nn.Conv2d(in_channels=c4_ch,
                              out_channels=self.feat_size, kernel_size=1,
                              padding=pad_out(1))

        self.P4_2 = nn.Conv2d(in_channels=self.feat_size,
                              out_channels=self.feat_size, kernel_size=3,
                              padding=pad_out(3))

        self.P3_1 = nn.Conv2d(in_channels=c3_ch,
                              out_channels=self.feat_size, kernel_size=1,
                              padding=pad_out(1))

        self.P3_2 = nn.Conv2d(in_channels=self.feat_size,
                              out_channels=self.feat_size, kernel_size=3,
                              padding=pad_out(3))

    def forward(self, inp):
        # expects inp to be output of c3, c4, c5
        c3, c4, c5 = inp
        p51 = self.P5_1(c5)
        p5_out = self.P5_2(p51)

        # p5_up = F.interpolate(p51, scale_factor=2)
        p5_up = F.interpolate(p51, size=(c4.size(2), c4.size(3)))
        p41 = self.P4_1(c4) + p5_up
        p4_out = self.P4_2(p41)

        # p4_up = F.interpolate(p41, scale_factor=2)
        p4_up = F.interpolate(p41, size=(c3.size(2), c3.size(3)))
        p31 = self.P3_1(c3) + p4_up
        p3_out = self.P3_2(p31)

        p6_out = self.P6(c5)

        p7_out = self.P7_2(F.relu(p6_out))
        if self.cfg['resize_img'] == [600, 600]:
            return [p4_out, p5_out, p6_out, p7_out]

        # p8_out = self.p8_gen(F.relu(p7_out))
        p8_out = F.adaptive_avg_pool2d(p7_out, 1)
        return [p3_out, p4_out, p5_out, p6_out, p7_out, p8_out]


class PyramidFeatures(nn.Module):
    """
    Pyramid Features, especially for Resnet
    """

    def __init__(self, C3_size, C4_size, C5_size, feature_size=256):
        super(PyramidFeatures, self).__init__()

        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=1, padding=1)
        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=1, padding=1)
        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size,
                              kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=1, padding=1)
        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size,
                            kernel_size=3, stride=2, padding=1)
        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size,
                              kernel_size=3, stride=2, padding=1)

    def forward(self, inputs):
        """
        Inputs should be from layer2,3,4
        """
        C3, C4, C5 = inputs
        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)
        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)
        P6_x = self.P6(C5)
        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)
        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class ResNet(nn.Module):
    """
    Basic Resnet Module
    """

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if block == BasicBlock:
            fpn_sizes = [self.layer2[layers[1]-1].conv2.out_channels, self.layer3[layers[2] -
                                                                                  1].conv2.out_channels, self.layer4[layers[3]-1].conv2.out_channels]
        elif block == Bottleneck:
            fpn_sizes = [self.layer2[layers[1]-1].conv3.out_channels, self.layer3[layers[2] -
                                                                                  1].conv3.out_channels, self.layer4[layers[3]-1].conv3.out_channels]

        self.freeze_bn()
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        prior = 0.01

    def _make_layer(self, block, planes, blocks, stride=1):
        """
        Convenience function to generate layers given blocks and
        channel dimensions
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        """
        inputs should be images
        """
        img_batch = inputs

        x = self.conv1(img_batch)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        features = self.fpn([x2, x3, x4])
        return features


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(
            model_urls['resnet50'], model_dir='.'), strict=False)
    return model
