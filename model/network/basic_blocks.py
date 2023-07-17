import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv2d(x)
        return F.leaky_relu(x, inplace=True)

class BasicConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv1d(x)
        return F.leaky_relu(x, inplace=True)

class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
            
    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1, c, h, w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)
        
        
class BasicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(BasicConv3d, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=bias, **kwargs)

    def forward(self, ipts):
        '''
            ipts: [n, c, s, h, w]
            outs: [n, c, s, h, w]
        '''
        outs = self.conv3d(ipts)
        return outs

class FConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(FConv3d, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.local_conv3d = BasicConv3d(in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feat = x.split(split_size, 3)
            lcl_feat = torch.cat([self.local_conv3d(_) for _ in lcl_feat], 3)
        return lcl_feat

class DConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(DConv3d, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.local_conv3d = BasicConv3d(in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            fea = []
            feaout = []
            for i in range(2**self.halving - 1):
                fea.append(self.local_conv3d(x[:, :, :, i*split_size:(i+2)*split_size, :]))
            feaout.append(fea[0][:, :, :, 0:split_size, :])
            for i in range(0, 2**self.halving - 2):
                feaout.append((fea[i][:, :, :, split_size:, :] + fea[i+1][:, :, :, :split_size, :])/2)
            feaout.append(fea[2**self.halving - 2][:, :, :, split_size:, :])
        return torch.cat(feaout, 3)


class TSB(nn.Module):
    def __init__(self, in_channels, use_gpu=False, **kwargs):
        super(TSB, self).__init__()
        self.in_channels = in_channels
        self.use_gpu = use_gpu
        self.patch_size = 2

        self.W = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                    kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.in_channels)
        )

        self.pool = nn.AvgPool3d(kernel_size=(1, self.patch_size, self.patch_size), 
                stride=(1, 1, 1), padding=(0, self.patch_size//2, self.patch_size//2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        nn.init.constant_(self.W[1].weight.data, 0.0)
        nn.init.constant_(self.W[1].bias.data, 0.0)


    def forward(self, x):
        b, c, t, h, w = x.size()
        inputs = x

        query = x.view(b, c, t, -1).mean(-1) 
        query = query.permute(0, 2, 1) 
        memory = self.pool(x) 
        if self.patch_size % 2 == 0:
            memory = memory[:, :, :, :-1, :-1]

        memory = memory.contiguous().view(b, 1, c, t * h * w) 
        query = F.normalize(query, p=2, dim=2, eps=1e-12)
        memory = F.normalize(memory, p=2, dim=2, eps=1e-12)
        f = torch.matmul(query.unsqueeze(2), memory) * 5
        f = f.view(b, t, t, h * w) 

        # mask the self-enhance
        mask = torch.eye(t).type(x.dtype) 
        if self.use_gpu: mask = mask.cuda()
        mask = mask.view(1, t, t, 1)
        f = (f - mask * 1e8).view(b, t, t * h * w)
        f = F.softmax(f, dim=-1)
        y = x.view(b, c, t * h * w)
        y = torch.matmul(f, y.permute(0, 2, 1)) 
        y = self.W(y.view(b * t, c, 1, 1))
        y = y.view(b, t, c, 1, 1)
        y = y.permute(0, 2, 1, 3, 4)
        z = y + inputs

        return z

class MAConv(nn.Module):
    def __init__(self, in_channels, out_channels, halving, fm_sign=False, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False, **kwargs):
        super(MAConv, self).__init__()
        self.halving = halving
        self.fm_sign = fm_sign
        self.local_conv3da = BasicConv3d(in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)
        self.local_conv3db = BasicConv3d(in_channels, out_channels, kernel_size, stride, padding, bias, **kwargs)

    def forward(self, x):
        '''
            x: [n, c, s, h, w]
        '''
        if self.halving == 0:
            lcl_feat = self.local_conv3d(x)
        else:
            h = x.size(3)
            split_size = int(h // 2**self.halving)
            lcl_feata = x.split(split_size, 3)
            lcl_feata = torch.cat([self.local_conv3da(_) for _ in lcl_feata], 3)

            xb = x[:, :, :, split_size+split_size//2:h-split_size-split_size//2, :]
            xbsplit = xb.split(split_size, 3)
            lcl_featb = []
            lcl_featb.append(self.local_conv3db(x[:, :, :, :split_size+split_size//2, :]))
            lcl_featb.append(torch.cat([self.local_conv3db(_) for _ in xbsplit], 3))
            lcl_featb.append(self.local_conv3db(x[:, :, :, h-split_size-split_size//2:, :]))
            lcl_featb = torch.cat(lcl_featb, 3)

        return lcl_feata + lcl_featb
