import torch
import torch.nn.functional as F
import torch.nn as nn
def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:] #b*h*w, 1, h, w
    xgrid, ygrid = coords.split([1,1], dim=-1) #b*h*w, 7, 7, 1
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1).contiguous()
    img = F.grid_sample(img.contiguous(), grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)
def cal_corr(f0,f1):
    b,c,h,w=f0.shape
    f0=f0.view(b,c,h*w).permute(0,2,1).contiguous()
    f1=f1.view(b,c,h*w).contiguous()
    #print(f0.shape,f1.shape)
    mp=torch.bmm(f0,f1)/c
    mp=mp.view(b,h*w,h,w).contiguous()
    return mp

class CorrBlock(nn.Module):
    def __init__(self,num_levels=1, radius=3, input_dim=320):
        super(CorrBlock, self).__init__()
        self.num_levels = num_levels
        self.radius = radius
        self.fc1=nn.Conv2d(input_dim,input_dim//2,kernel_size=1,bias=False)#nn.Linear(input_dim,input_dim//4,bias=False)
        self.fc0=nn.Conv2d(input_dim,input_dim//2,kernel_size=1,bias=False)
    def forward(self,x):
        self.corr_pyramid = []
        device=x.device
        #print('x:de:',x.device)
        r=self.radius
        x=x.permute(0,2,1,3,4).contiguous()
        # x:b,t,c,h,w
        f0=x[:,:-1].clone()
        f1=x[:,1:].clone()
        b,t,c,h,w=f0.shape
        f0=f0.view(b*t,c,h,w)
        f1=f1.view(b*t,c,h,w)
        f0=self.fc0(f0)
        f1=self.fc1(f1)
        f0=f0.view(b,t,c//2,h,w)
        f1=f1.view(b,t,c//2,h,w)

        b,t,c,h,w=f0.shape # 16 11

        f0=f0.view(b*t,c,h,w)
        f1=f1.view(b*t,c,h,w)
        
        b,c,h,w=f0.shape
        
        
        coords = coords_grid(b, h, w).to(device) #b*t, 2, h, w
        coords = coords.permute(0, 2, 3, 1).contiguous() #b*t, h, w, 2
        
        
        corr=cal_corr(f0,f1) #b*t, h*w, h, w
        corr=corr.reshape(b*h*w, 1, h, w)
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)
        out_pyramid = []
        for i in range(self.num_levels):
            
            corr=self.corr_pyramid[i]

            #print(coords.shape)
            dx = torch.linspace(-r, r, 2*r+1) # tensor([-3., -2., -1.,  0.,  1.,  2.,  3.])
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(device) # 7, 7, 2
            centroid_lvl = coords.reshape(b*h*w, 1, 1, 2) / (2**i)
            #centroid_lvl = coords.view(b*h*w, 1, 1, 2) / (2**i)
            #print(centroid_lvl)
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            #print(delta_lvl)
            coords_lvl = centroid_lvl + delta_lvl # b*h*w, 7, 7, 2
            corr = bilinear_sampler(corr, coords_lvl)
            #print('corr:',corr.shape)
            corr = corr.view(b, h, w, -1) 
            out_pyramid.append(corr)
            
        out_fea=torch.cat(out_pyramid,3)
        b,h,w,c=out_fea.shape
        out_fea=out_fea.view(b//t,t,h,w,c)
        return out_fea.permute(0, 4, 1, 2, 3).contiguous().float() # b c t h w

class MyMotion(nn.Module):
    def __init__(self):
        super(MyMotion, self).__init__()

    def forward(self,x):
        lcl_feat = x.split(2, dim=1)
        lcl_feat = torch.cat([_ - _.mean(1).unsqueeze(1) for _ in lcl_feat], 1)
        out = (torch.abs(lcl_feat) * x).sigmoid() + x
        return out

class MyMotion2(nn.Module):
    def __init__(self):
        super(MyMotion2, self).__init__()

    def forward(self,x):
        f01=x[:,:,:-2].clone()
        f11=x[:,:,2:].clone()
        out = torch.cat([torch.zeros_like(x[:, :, 0]).unsqueeze(2), torch.zeros_like(x[:, :, 0]).unsqueeze(2), f11-f01], 2)
        f00=x[:,:,:-1].clone()
        f10=x[:,:,1:].clone()
        out0 = torch.cat([torch.zeros_like(x[:, :, 0]).unsqueeze(2),f10-f00], 2)
        return torch.abs(out) + torch.abs(out0)
    
