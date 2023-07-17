import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
import random

from .basic_blocks import BasicConv3d, MAConv
from .gaitu import TFE, C_TA, F_TA, clones
from .motion import MyMotion2


class GaitU(nn.Module):
    def __init__(self, hidden_dim, class_num):
        super(GaitU, self).__init__()
        self.hidden_dim = hidden_dim

        _in_channels = 1
        _channels = [32, 64, 128]
        high=32
        
        
        self.conv3d_1f = nn.Sequential(
            BasicConv3d(_in_channels, _channels[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3d_2f = nn.Sequential(
            MAConv(_channels[0], _channels[1], 3, kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.MaxPoolf = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3d_3f = nn.Sequential(
            MAConv(_channels[1], _channels[2], 3, kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3d_4f = nn.Sequential(
            BasicConv3d(_channels[2], _channels[2], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        
        self.conv3d_1c = nn.Sequential(
            BasicConv3d(_in_channels, _channels[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3d_2c = nn.Sequential(
            BasicConv3d(_channels[0], _channels[1], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.MaxPoolc = nn.MaxPool3d(
            kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv3d_3c = nn.Sequential(
            BasicConv3d(_channels[1], _channels[2], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.conv3d_4c = nn.Sequential(
            BasicConv3d(_channels[2], _channels[2], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )
        
        temporal_channels = _channels[2]+_channels[2]+_channels[1]

        self.MyMotion = MyMotion2()
        self.conv3d_motionc = nn.Sequential(
            BasicConv3d(temporal_channels, _channels[0], kernel_size=(3, 3, 3),
                        stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.LeakyReLU(inplace=True)
        )

        self.gf_tfe = TFE(temporal_channels, temporal_channels, high)
        self.gf_gd = F_TA(temporal_channels, high)
        self.gf_gs = F_TA(temporal_channels, high)
        self.gc_tfe = TFE(temporal_channels+_channels[0], temporal_channels+_channels[0], high//2)
        self.gc_gd = C_TA(temporal_channels+_channels[0], high//2)
        self.gc_gs = C_TA(temporal_channels+_channels[0], high//2)
        self.clsf=nn.Linear(in_features=temporal_channels, out_features=11)
        self.clsc=nn.Linear(in_features=temporal_channels+_channels[0], out_features=11)

        # separate FC
        self.fc_binf = nn.Parameter(
                init.xavier_uniform_(
                    torch.zeros(high, temporal_channels*4, hidden_dim)))
        self.fc_binc = nn.Parameter(
                init.xavier_uniform_(
                    torch.zeros(high//2, (temporal_channels+_channels[0])*2, hidden_dim)))       
        self.fc_binclass = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.zeros(high + high//2, hidden_dim, class_num)))
        
        self.bn = nn.ModuleList()
        for i in range(high + high//2):
            self.bn.append(nn.BatchNorm1d(hidden_dim))
        # initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.Conv2d, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.constant(m.bias.data, 0.0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                m.bias.data.zero_()

    def forward(self, silho, train):
        n = silho.size(0)
        x = silho.unsqueeze(2)
        del silho

        n, s, c, h, w = x.size()
        if s < 3:
            repeat = 3 if s == 1 else 2
            x = x.repeat(1, repeat, 1, 1, 1)
        if s == 3:
            x = x.repeat(1, 2, 1, 1, 1)
        if train and random.random() > 0.8:
            for i in range(n):
                self.randnum = random.randint(0, 10)
                for j in range(self.randnum):
                    size = random.randint(1, 3)
                    h = random.randint(10, 50)
                    w = random.randint(10, 34)
                    x[i, :, :, h-size:h+size, w-size:w+size] = 0
                    h1 = random.randint(2, 61)
                    w1 = random.randint(10, 34)
                    x[i, :, :, h1-size:h1+size, w1-size:w1+size] = 1
        
        y = x         
        x = self.conv3d_1f(x.permute(0, 2, 1, 3, 4))   
        x = self.conv3d_2f(x)
        x = self.MaxPoolf(x)
        xr = x
        x = self.conv3d_3f(x)
        x = torch.cat([xr, x, self.conv3d_4f(x)], 1).permute(0, 2, 1, 3, 4)

        x_view = torch.max(x, 1)[0]
        x_feat = F.avg_pool2d(x_view, (x_view.size(-2), x_view.size(-1))).squeeze(-1).squeeze(-1)
        angle_probef=self.clsf(x_feat) # n 11
             
        x = x.max(-1)[0] + x.mean(-1) #n, s, c, h
        x, t_d, t_s = self.gf_tfe(x)
        gfd = self.gf_gd(x, t_d)
        gfs = self.gf_gs(x, t_s)
        featuref = torch.cat([gfd, gfs], -1) #h, n, c*2
        
        triple_feature = featuref.matmul(self.fc_binf) # h, n, 256
        triple_feature = triple_feature.permute(1, 0, 2).contiguous() # n, h, 256
        
        
        y = self.conv3d_1c(y.permute(0, 2, 1, 3, 4))
        y = self.conv3d_2c(y)
        y = self.MaxPoolc(y)
        yr = y
        y = self.conv3d_3c(y)
        y = torch.cat([yr, y, self.conv3d_4c(y)], 1)
        y = self.MaxPoolc(y)
        y = torch.cat([y,  self.conv3d_motionc(self.MyMotion(y))], 1).permute(0, 2, 1, 3, 4)


        y_view = torch.max(y, 1)[0]
        y_feat = F.avg_pool2d(y_view, (y_view.size(-2), y_view.size(-1))).squeeze(-1).squeeze(-1)
        angle_probec=self.clsc(y_feat) # n 11


        y = y.max(-1)[0] + y.mean(-1) #n, s, c, h
        y, t_dy, t_sy = self.gc_tfe(y)    
        gcd = self.gc_gd(y, t_dy)
        gcs = self.gc_gs(y, t_sy)
        featurec =  torch.cat([gcd, gcs], 2).contiguous() #h, n, 2*c
        
        triple_featurey = featurec.matmul(self.fc_binc) # h, n, 256
        triple_featurey = triple_featurey.permute(1, 0, 2).contiguous() # n, h, 256
        triple_feature = torch.cat([triple_feature, triple_featurey], 1).contiguous()
               
        part_feature = []
        for idx, block in enumerate(self.bn):
            part_feature.append(block(triple_feature[:, idx, :]).unsqueeze(1))
        part_feature = torch.cat(part_feature, 1).contiguous()
        
        logits = part_feature.permute(1, 0, 2).matmul(self.fc_binclass) # [p, n, c]

        return triple_feature, logits.permute(1,0,2).contiguous(), angle_probef, angle_probec
