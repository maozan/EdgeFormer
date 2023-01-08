import torch.nn as nn
import torch
from re import T
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler

import functools
from einops import rearrange, repeat

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d

class DR(nn.Module):
    def __init__(self, in_d, out_d):
        super(DR, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.conv1 = nn.Conv2d(self.in_d, self.out_d, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.out_d)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, ratio = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels,in_channels//ratio,1,bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels//ratio, in_channels,1,bias=False)
        self.sigmod = nn.Sigmoid()
    def forward(self,x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmod(out)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class SICDNet(torch.nn.Module):
    def __init__(self, input_nc, output_nc, feat_nc, backbone='resnet18',
                 output_sigmoid=True, if_upsample_2x=True,
                 is_cam=False, is_ecam=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(SICDNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,False,False])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = nn.Conv2d(64, output_nc, kernel_size=1)

        self.if_upsample_2x = if_upsample_2x
        layers = 512 * expand
        self.conv_pred = nn.Conv2d(layers, 32, kernel_size=3, padding=1)

        self.dr1 = DR(64, feat_nc)
        self.dr2 = DR(128, feat_nc)
        self.dr3 = DR(256, feat_nc)
        self.dr4 = DR(512, feat_nc)

        self.drop = nn.Dropout2d(p=0.2)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

        self.is_ecam = is_ecam
        self.is_cam = is_cam
        self.cam = ChannelAttention(feat_nc * 4, ratio=16)
        self.cam1 = ChannelAttention(feat_nc, ratio=16 // 4)

    def forward_aligned(self, x1, x2):
        x1_r = repeat(x1, 'b c h w -> b d c h w', d = 1)
        x1_d = rearrange(x1_r, 'b d (c c1) h w -> b (d c1) c h w', c1 = 8)

        x2_r = repeat(x2, 'b c h w -> b d c h w', d = 1)
        x2_d = rearrange(x2_r, 'b d (c c1) h w -> b (d c1) c h w', c1 = 8)

        return x1_d, x2_d

    def forward(self, x1, x2):
        x_A1, x_A2, x_A3, x_A4 = self.forward_single(x1)
        x_B1, x_B2, x_B3, x_B4 = self.forward_single(x2)

        ## aligned
        # print(x_A4.shape)
        x_A4, x_B4 = self.forward_aligned(x_A4, x_B4)
        # print(x_A4.shape, x_B4.shape)
        x_pa = x_A4 * x_B4
        # print(x_pa.shape)
        x_a = torch.sum(x_pa, dim = 1)
        # print(x_a.shape)

        ## decoder
        x = self.upsamplex4(self.upsamplex4(self.upsamplex2(x_a)))    
       
        x = self.classifier(x)

        if self.output_sigmoid:
            x = self.sigmoid(x)
        output = []
        output.append(x)
        return output

    def forward_single(self, x):
        # resnet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x1 = self.resnet.layer1(x) # 1/4, in=64, out=64
        x2 = self.resnet.layer2(x1) # 1/8, in=64, out=128
        x3 = self.resnet.layer3(x2) # 1/16, in=128, out=256
        x4 = self.resnet.layer4(x3) # 1/32, in=256, out=512
    
        return x1, x2, x3, x4

