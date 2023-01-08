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

class ResFE(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(pretrained=True, replace_stride_with_dilation=[False,True,True])

    def forward(self, x1, x2):
        pass

class Conv(nn.Module):
    def __init__(self, in_d = 3 + 12):
        super(Conv, self).__init__()
        self.in_d = in_d
        
        # first conv operation
        self.conv_64_1 = nn.Conv2d(in_channels=self.in_d, out_channels=64, kernel_size=3, padding=1)
        self.bn_64_1 = nn.BatchNorm2d(64)
        self.conv_64_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1)
        self.bn_64_2 = nn.BatchNorm2d(64)

        # second conv operation
        self.conv_128_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn_128_1 = nn.BatchNorm2d(128)
        self.conv_128_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)
        self.bn_128_2 = nn.BatchNorm2d(128)

        # third conv operation
        self.conv_256_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn_256_1 = nn.BatchNorm2d(256)
        self.conv_256_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.bn_256_2 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.relu = nn.ReLU()

    def forward(self, input):
        x1 = self.bn_64_1(self.conv_64_1(input))
        x1 = self.relu(self.bn_64_2(self.conv_64_2(x1)))

        x2 = self.pool(x1)
        x2 = self.bn_128_1(self.conv_128_1(x2))
        x2 = self.relu(self.bn_128_2(self.conv_128_2(x2)))

        x3 = self.pool(x2)
        x3 = self.bn_256_1(self.conv_256_1(x3))
        x3 = self.relu(self.bn_256_2(self.conv_256_2(x3)))

        return x3

class HMNet(nn.Module):
    def __init__(self, input_nc, output_nc, feat_nc, image_level=True):
        super(HMNet, self).__init__()
        # cat之后的特征提取器
        self.FEC = Conv(in_d = 3 + 12)
        self.FE1 = Conv(in_d = 3)
        self.FE2 = Conv(in_d = 3)
        self.FEE1 = Conv(in_d = 12)
        self.FEE2 = Conv(in_d = 12)

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.classifier = self.classifier = nn.Sequential(
                        nn.Conv2d(256, feat_nc, kernel_size=3,
                                            padding=1, stride=1, bias=False),
                        nn.BatchNorm2d(feat_nc),
                        nn.ReLU(),
                        nn.Conv2d(feat_nc, output_nc, kernel_size=1)
                        )

        self.image_level = image_level

    def forward(self, x1, x2, edge1=None, edge2=None):
        
        if edge1 is not None and edge2 is not None:
            if self.image_level:
                print("===== image =====")
                x = self.forward_image_level(x1, x2, edge1, edge2)
            else:
                print("===== feature =====")
                x = self.forward_feature_level(x1, x2, edge1, edge2)
        
        if edge1 is None and edge2 is None:
            print('======= pure =======')
            x_1 = self.FE1(x1)
            x_2 = self.FE1(x2)            
            x = torch.abs(x_1 - x_2)
            x = self.upsamplex2(self.upsamplex2(x))
            # classifier
            x = self.classifier(x)

        output = []
        output.append(x)
        return output

    def forward_image_level(self, x1, x2, edge1, edge2):
        # oringin and edge image concat
        x_1 = torch.cat((x1, edge1), dim=1)
        x_2 = torch.cat((x2, edge2), dim=1)

        print(x_1.shape)

        f_1 = self.FEC(x_1)
        f_2 = self.FEC(x_2)

        # diffrential
        x = torch.abs(f_1 - f_2)

        # decoder upsample
        x = self.upsamplex2(self.upsamplex2(x))

        # classifier
        x = self.classifier(x)

        return x

    def forward_feature_level(self, x1, x2, edge1, edge2):

        f_1 = self.FE1(x1)
        f_2 = self.FE1(x2)

        fe_1 = self.FEE1(edge1)
        fe_2 = self.FEE2(edge2)

        x_1 = f_1 + fe_1
        x_2 = f_2 + fe_2
        
        x = torch.abs(x_1 - x_2)

        x = self.upsamplex2(self.upsamplex2(x))

        # classifier
        x = self.classifier(x)

        return x