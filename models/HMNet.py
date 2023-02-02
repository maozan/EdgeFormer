import torch.nn as nn
import torch
from re import T
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch import einsum

import functools
from einops import rearrange, repeat

import numpy as np

import models
from models.help_funcs import Transformer, TransformerDecoder, TwoLayerConv2d

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

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x  # [N, H//downscaling_factor, W//downscaling_factor, out_channels]

class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))

def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding, cross_attn):
        super().__init__()
        inner_dim = head_dim * heads

        self.heads = heads
        self.scale = head_dim ** -0.5
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted
        self.cross_attn = cross_attn

        if self.shifted:
            displacement = window_size // 2
            self.cyclic_shift = CyclicShift(-displacement)
            self.cyclic_back_shift = CyclicShift(displacement)
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)

        if not self.cross_attn:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        else:
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
            self.to_q = nn.Linear(dim, inner_dim, bias=False)

        if self.relative_pos_embedding:
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, y=None):
        if self.shifted:
            x = self.cyclic_shift(x)
            if self.cross_attn:
                y = self.cyclic_shift(y)

        b, n_h, n_w, _, h = *x.shape, self.heads
        # print('forward-x: ', x.shape)   # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
        if not self.cross_attn:
            qkv = self.to_qkv(x).chunk(3, dim=-1)
            # [N, H//downscaling_factor, W//downscaling_factor, head_dim * head] * 3
        else:
            kv = self.to_kv(x).chunk(2, dim=-1)
            qkv = (self.to_q(y), kv[0], kv[1])

        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)
        # print('forward-q: ', q.shape)   # [N, num_heads, num_win, win_area, hidden_dim/num_heads]
        # print('forward-k: ', k.shape)
        # print('forward-v: ', v.shape)

        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale  # q * k / sqrt(d)

        if self.relative_pos_embedding:
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            dots += self.pos_embedding

        if self.shifted:
            dots[:, :, -nw_w:] += self.upper_lower_mask
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        attn = dots.softmax(dim=-1)

        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        # [N, H//downscaling_factor, W//downscaling_factor, head_dim * head]
        out = self.to_out(out)
        # [N, H//downscaling_factor, W//downscaling_factor, dim]
        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding, cross_attn):
        super().__init__()
        self.attention_block = Residual(PreNorm(dim, WindowAttention(dim=dim,
                                                                     heads=heads,
                                                                     head_dim=head_dim,
                                                                     shifted=shifted,
                                                                     window_size=window_size,
                                                                     relative_pos_embedding=relative_pos_embedding,
                                                                     cross_attn=cross_attn)))
        self.mlp_block = Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x, y=None):
        x = self.attention_block(x, y=y)
        x = self.mlp_block(x)
        return x

class SwinModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, cross_attn):
        r"""
        Args:
            in_channels(int): 输入通道数
            hidden_dimension(int): 隐藏层维数，patch_partition提取patch时有个Linear学习的维数
            layers(int): swin block数，必须为2的倍数，连续的，regular block和shift block
            downscaling_factor: H,W上下采样倍数
            num_heads: multi-attn 的 attn 头的个数
            head_dim:   每个attn 头的维数
            window_size:    窗口大小，窗口内进行attn运算
        """
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                            downscaling_factor=downscaling_factor)

        self.layers = nn.ModuleList([])
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn),
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding,
                          cross_attn=cross_attn),
            ]))

    def forward(self, x, y=None):
        if y is None:
            x = self.patch_partition(x)  # [N, H//downscaling_factor, W//downscaling_factor, hidden_dim]
            for regular_block, shifted_block in self.layers:
                x = regular_block(x)
                x = shifted_block(x)
            return x.permute(0, 3, 1, 2)
            # [N, hidden_dim,  H//downscaling_factor, W//downscaling_factor]
        else:
            x = self.patch_partition(x)
            y = self.patch_partition(y)
            for regular_block, shifted_block in self.layers:
                x = regular_block(x, y)
                x = shifted_block(x, y)
            return x.permute(0, 3, 1, 2)

class HMNet(nn.Module):
    def __init__(self, input_nc, output_nc, feat_nc, 
                        n_feats=256, n_heads=4, head_dim=64, win_size=4,
                        image_level=True,
                        is_trans=False):
        super(HMNet, self).__init__()
        # cat之后的特征提取器
        self.FEC = Conv(in_d = 3 + 12)
        self.FE1 = Conv(in_d = 3)
        self.FE2 = Conv(in_d = 3)
        self.FEE1 = Conv(in_d = 12)
        self.FEE2 = Conv(in_d = 12)

        self.cross_attn = SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                window_size=win_size, relative_pos_embedding=True, cross_attn=True)
        
        self.self_attn = SwinModule(in_channels=n_feats, hidden_dimension=n_feats, layers=2,
                                                downscaling_factor=1, num_heads=n_heads, head_dim=head_dim,
                                                window_size=win_size, relative_pos_embedding=True, cross_attn=False) 

        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')

        self.classifier = self.classifier = nn.Sequential(
                        nn.Conv2d(256, feat_nc, kernel_size=3,
                                            padding=1, stride=1, bias=False),
                        nn.BatchNorm2d(feat_nc),
                        nn.ReLU(),
                        nn.Conv2d(feat_nc, output_nc, kernel_size=1)
                        )

        self.image_level = image_level
        self.is_trans = is_trans

    def forward(self, x1, x2, edge1=None, edge2=None):
        
        if edge1 is not None and edge2 is not None:
            if self.image_level:
                print("===== image =====")
                x = self.forward_image_level(x1, x2, edge1, edge2)
            else:
                print("===== feature =====")
                if self.is_trans:
                    print('----transformer----')
                    x = self.forward_feature_level_cross_attention(x1, x2, edge1, edge2)
                else:
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

        # 应用transformer是在这儿的特征融合前还是融合后，应该都有考量
        x_1 = f_1 + fe_1 
        x_2 = f_2 + fe_2 
        
        x = torch.abs(x_1 - x_2)

        x = self.upsamplex2(self.upsamplex2(x))

        # classifier
        x = self.classifier(x)

        return x

    def forward_feature_level_cross_attention(self, x1, x2, edge1, edge2):
        '''
            edge_feature: q,
            image_feature: kv,
        '''

        f_1 = self.FE1(x1)
        f_2 = self.FE1(x2)

        fe_1 = self.FEE1(edge1)
        fe_2 = self.FEE2(edge2)

        # insert transformer module
        # x:kv, y:q
        t_1 = self.cross_attn(f_1, fe_1)
        t_2 = self.cross_attn(f_2, fe_2)

        x_1 = f_1 + fe_1 + t_1
        x_2 = f_2 + fe_2 + t_2
        # x_1 = t_1
        # x_2 = t_2
        
        x = torch.abs(x_1 - x_2)

        x = self.upsamplex2(self.upsamplex2(x))

        # classifier
        x = self.classifier(x)

        return x
    
    def forward_feature_level_cross_attention_cascade(self, x1, x2, edge1, edge2):
        '''
            edge_feature: q,
            image_feature: kv,
        '''

        f_1 = self.FE1(x1)
        f_2 = self.FE1(x2)

        fe_1 = self.FEE1(edge1)
        fe_2 = self.FEE2(edge2)

        # insert transformer module
        # x:kv, y:q
        t_1 = self.cross_attn(f_1, fe_1)
        t_2 = self.cross_attn(f_2, fe_2)

        ## before feature fusion, cascade transfomer module be perform
        # sab sab; sab cab; cab sab; cab cab (sab: self_attn, cab: cross_attn)
        t_1 = self.self_attn(t_1) # plain fusion
        t_2 = self.self_attn(t_2)

        a = t_1.clone()
        b = t_2.clone()
        # t_1 = self.cross_attn(t_1, b)
        # t_2 = self.cross_attn(t_2, a)

        x_1 = f_1 + fe_1 + t_1
        x_2 = f_2 + fe_2 + t_2
        # x_1 = t_1
        # x_2 = t_2
        
        x = torch.abs(x_1 - x_2)

        x = self.upsamplex2(self.upsamplex2(x))

        # classifier
        x = self.classifier(x)

        return x


class HMNet_res(torch.nn.Module):
    def __init__(self, input_nc, output_nc, feat_nc, backbone='resnet18',
                 output_sigmoid=True, if_upsample_2x=True,
                 is_cam=False, is_ecam=False):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(HMNet_res, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[True,True,True])
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
        self.bn   = nn.BatchNorm2d(input_nc)
        self.upsamplex2 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsamplex4 = nn.Upsample(scale_factor=4, mode='bilinear')

        self.classifier = nn.Sequential(
                        nn.Conv2d(feat_nc, feat_nc, kernel_size=3,
                                            padding=1, stride=1, bias=False),
                        nn.BatchNorm2d(feat_nc),
                        nn.ReLU(),
                        nn.Conv2d(feat_nc, output_nc, kernel_size=1)
                        )

        self.feat_nc = feat_nc
        self.if_upsample_2x = if_upsample_2x
        layers = 512 * expand
        self.conv_edge = nn.Sequential(
                        nn.Conv2d(12, output_nc, kernel_size=1,),
                        nn.BatchNorm2d(output_nc),
                        nn.ReLU(),
                        )

        self.conv_32 = nn.Conv2d(512, feat_nc, kernel_size=1)

        self.drop = nn.Dropout2d(p=0.2)

        self.output_sigmoid = output_sigmoid
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x1_edge = None, x2_edge = None):
        if x1_edge is not None and x2_edge is not None:
            x1_edge_feat = self.conv_edge(x1_edge)
            x2_edge_feat = self.conv_edge(x2_edge)
            # x_A = torch.cat((x1, x1_edge), dim=1)
            # x_B = torch.cat((x2, x2_edge), dim=1)

            # x1 = self.conv_pre(x_A)
            # x2 = self.conv_pre(x_B)

        x_A1, x_A2, x_A3, x_A4 = self.forward_single(x1)
        x_B1, x_B2, x_B3, x_B4 = self.forward_single(x2)

        x_A4 = self.conv_32(x_A4)
        x_B4 = self.conv_32(x_B4)

        ## decoder
        if x1_edge is not None and x2_edge is not None:
            x_A4 = self.upsamplex4(x_A4) + x1_edge_feat.repeat(1, self.feat_nc // 2, 1, 1)
            x_B4 = self.upsamplex4(x_B4) + x2_edge_feat.repeat(1, self.feat_nc // 2, 1, 1)
        x = torch.abs(x_A4 - x_B4)
        if x1_edge is None and x2_edge is None:
            x = self.upsamplex4(x)  
       
        x = self.classifier(x)

        # if x1_edge is not None and x2_edge is not None:
        #     edge = torch.abs(x1_edge_feat -  x2_edge_feat)
        #     x = x + edge

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
