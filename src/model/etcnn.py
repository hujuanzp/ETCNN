## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from turtle import forward
from model import common
from model import attention

import torch.nn as nn
import torch

def make_model(args, parent=False):
    return ETCNN(args)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, args, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        # modules_body = [attention.NonLocalSparseAttention(hannels=n_feat, chunk_size=args.chunk_size, n_hashes=args.n_hashes, reduction=4, res_scale=args.res_scale)]
        modules_body = []

        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class ResidualGroup1(nn.Module):
    def __init__(self, args, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        # modules_body = [attention.NonLocalSparseAttention(hannels=n_feat, chunk_size=args.chunk_size, n_hashes=args.n_hashes, reduction=4, res_scale=args.res_scale)]
        modules_body = []

        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        # res += x
        return res

class RCS(nn.Module):
    def __init__(self, args, conv, n_feats, kernel_size):
        super(RCS, self).__init__()
        m_rb = [
            conv(args.n_colors, n_feats, kernel_size),
            nn.ReLU(True),
            conv(n_feats, args.n_colors, kernel_size)
        ]
        self.res_scale = args.res_scale
        self.conv2 = conv(n_feats, args.n_colors, kernel_size)
        self.conv3 = conv(n_feats, args.n_colors, kernel_size)
        self.rb = nn.Sequential(*m_rb)
         
    def forward(self, input, x):
        x1 = self.conv2(x)
        rs = input - x1
        c = torch.gt(input, 0).float()
        rs = rs * c
        rs1 = self.rb(rs)
        rs1 = rs1+rs*self.res_scale       
        x2 = self.conv3(x)
        x3 = rs1 + x2
        return x3

class ETCNN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ETCNN, self).__init__()
        
        #RG  Para-----#
        n_resgroups = 10
        n_resblocks = 20
        n_feats = 64
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        act = nn.ReLU(True)
        #-----------------
        # NLSA Para---#
        chunk_size = 144
        n_hashes  = 4
        #-----------------

        
        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]
        modules_body =[attention.NonLocalSparseAttention(channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=args.res_scale)]
        for i in range(3):
            modules_body.append(ResidualGroup(args, conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks))
        modules_body.append(attention.NonLocalSparseAttention(channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=args.res_scale))
        for i in range(2):
            modules_body.append(ResidualGroup(args, conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks))
        modules_body.append(attention.NonLocalSparseAttention(channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=args.res_scale))
        for i in range(3):
            modules_body.append(ResidualGroup(args, conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks))
        modules_body.append(attention.NonLocalSparseAttention(channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=args.res_scale))
        for i in range(2):
            modules_body.append(ResidualGroup(args, conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks))
        modules_body.append(attention.NonLocalSparseAttention(channels=n_feats, chunk_size=chunk_size, n_hashes=n_hashes, reduction=4, res_scale=args.res_scale))

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        self.body = nn.Sequential(*modules_body)

        self.rcs = RCS(args, conv, n_feats, kernel_size)
        self.dc = common.Dc()

    def forward(self, x):
        input = x
        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        x = self.rcs(input, res) * 1

        x = self.add_mean(x)
        x = self.dc(input, x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
