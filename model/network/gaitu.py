import torch 
import torch.nn as nn
import torch.nn.init as init
import numpy as np 
import torch.nn.functional as F 
import random
import copy

def conv1d(in_planes, out_planes, kernel_size, has_bias=False, **kwargs):
    return nn.Conv1d(in_planes, out_planes, kernel_size, bias=has_bias, **kwargs)

def conv_bn(in_planes, out_planes, kernel_size, **kwargs):
    return nn.Sequential(conv1d(in_planes, out_planes, kernel_size, **kwargs),
                            nn.BatchNorm1d(out_planes))

def cal_key(in_planes, out_planes, kernel_size, **kwargs):
    return nn.Sequential(conv1d(in_planes, in_planes//16, kernel_size, **kwargs),
                            nn.LeakyReLU(inplace=True),
                            conv1d(in_planes//16, out_planes, kernel_size, **kwargs),
                            nn.Sigmoid())


class TFE(nn.Module):
    def __init__(self, in_planes, out_planes, part_num):
        super(TFE, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.part_num = part_num

        self.dynamic1 = nn.ModuleList([conv_bn(in_planes*part_num, out_planes*part_num, 3, padding=1, groups=part_num), 
                                conv_bn(in_planes*part_num, out_planes*part_num, 3, padding=1, groups=part_num)])
        self.dynamic2 = nn.ModuleList([conv_bn(in_planes*part_num, out_planes*part_num, 3, padding=2, dilation = 2, groups=part_num), 
                                conv_bn(in_planes*part_num, out_planes*part_num, 3, padding=2, dilation = 2, groups=part_num)])
        self.dynamic3 = nn.ModuleList([conv_bn(in_planes*part_num, out_planes*part_num, 3, padding=3, dilation = 3, groups=part_num), 
                                conv_bn(in_planes*part_num, out_planes*part_num, 3, padding=3, dilation = 3, groups=part_num)])

    def get_dynamic(self, x):
        n, s, c, h = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(n, -1, s)
        temp1 = self.dynamic1[0](x)
        dynamic1_feature = temp1 + self.dynamic1[1](temp1)
        temp2 = self.dynamic2[0](x)
        dynamic2_feature = temp2 + self.dynamic2[1](temp2)
        temp3 = self.dynamic3[0](x)
        dynamic3_feature = temp3 + self.dynamic3[1](temp3)
        return (dynamic1_feature + dynamic2_feature + dynamic3_feature).view(n, h, c, s).permute(0, 3, 2, 1).contiguous() 

    def get_static(self, x):
        n, s, c, h = x.size()
        x = x.permute(0, 3, 2, 1).contiguous()
        long_term_feature = x.mean(-1)
        long_term_feature = long_term_feature.unsqueeze(1).repeat(1, s, 1, 1)
        return long_term_feature.permute(0, 1, 3, 2).contiguous()

    def forward(self, x):
        multi_scale_temporal = [x, self.get_dynamic(x), self.get_static(x)]
        return multi_scale_temporal


class C_TA(nn.Module):
    def __init__(self, in_planes, part_num):
        super(C_TA, self).__init__()
        self.in_planes = in_planes
        self.part_num = part_num

        self.key = cal_key(in_planes*part_num*2, part_num, 1, groups=part_num)
        self.value = conv1d(in_planes*part_num*2, in_planes*part_num, 1, groups=part_num)
    
    def forward(self, y, gt):
        n, s, c, h = y.size()
        y = y.unsqueeze(-1)
        gt = gt.unsqueeze(-1) + y
        t = torch.cat([y, gt], -1)
        cat_feature = t.permute(0, 3, 4, 2, 1).contiguous().view(n, h*2*c, s)

        part_score = self.key(cat_feature).view(n, h, 1, s)
        cat_feature = self.value(cat_feature).view(n, h, c, s)
        weighted_sum = ((cat_feature * part_score).sum(-1) / part_score.sum(-1)).permute(1, 0, 2).contiguous()

        return weighted_sum


class F_TA(nn.Module):
    def __init__(self, in_planes, part_num):
        super(F_TA, self).__init__()
        self.in_planes = in_planes
        self.part_num = part_num

        self.key = cal_key(in_planes*part_num*2, part_num, 1, groups=part_num)
        self.value = conv1d(in_planes*part_num*2, in_planes*part_num, 1, groups=part_num)

    def forward(self, x, gt):
        n, s, c, p = x.size()        
        cat_feature = torch.cat([x, gt+x], 2)
        
        part_score = self.key(cat_feature.permute(0, 3, 2, 1).contiguous().view(n, -1, s)).view(n, p, 1, s)
        cat_feature = self.value(cat_feature.permute(0, 3, 2, 1).contiguous().view(n, -1, s)).view(n, p, c, s)
        weighted_sum = (cat_feature * part_score).sum(-1) / part_score.sum(-1) #nxpxc

        max_fea = (cat_feature * part_score).max(-1)[0] / part_score.mean(-1)

        weighted_sum = torch.cat([weighted_sum, max_fea], 2)
        weighted_sum = weighted_sum.permute(1, 0, 2).contiguous()

        return weighted_sum

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

