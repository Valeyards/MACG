import torch.nn.functional as F
import math
import numpy as np
from typing import Any, Sequence, Tuple
import functools
import ml_collections
from torch import nn
import torch
from torchvision.transforms import Resize
from torchvision.transforms.functional import InterpolationMode
import einops
import torchvision
from torch.distributions.normal import Normal
import torch.utils.checkpoint as checkpoint
import time

class dw_sep_conv(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel)
        self.pwconv = nn.Conv2d(in_channel, in_channel, kernel_size=1)
        self.id = nn.Identity()
    def forward(self, x):
        b,c,g,w = x.shape
        if g==1 or w==1:
            y = self.id(x)
        else:
            y = self.dwconv(x)
        y = self.pwconv(x)
        return y

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ID(nn.Module):
    def __init__(self,in_channel=None):
        super().__init__()
        
    def forward(self, x):
        return x

class GridTokenMixLayer(nn.Module):   
    # grid size 5 6 7
    def __init__(self,in_channel,grid_size: Sequence[int], tokenmixer=None, bias = True,factor = 2,dropout_rate= 0.0):
        super().__init__()
        self.in_channel=in_channel    
        self.grid_size= grid_size
        self.bias=bias
        self.factor=factor
        self.dropout_rate=dropout_rate
        self.tokenmixer = tokenmixer
        self.linear1=nn.Linear(self.in_channel,self.in_channel* self.factor, bias=self.bias)
        self.linear2=nn.Linear(self.in_channel,self.in_channel, bias=self.bias)
              
        if tokenmixer is None:
            self.gridgatingunit=GridGatingUnit(self.grid_size[0]*self.grid_size[1]*self.grid_size[2])         
        else:
            self.tokenmixer = tokenmixer(in_channel)
  
    def forward(self, x, deterministic=True):
        n, c, d, h, w = x.shape      
        gd, gh, gw = self.grid_size
        fd, fh, fw = d//gd, h // gh, w // gw 
        x = block_images_einops(x, patch_size=(fd, fh, fw)) # b c g p
        # gMLP1: Global (grid) mixing part, provides global grid communication.
        _n, _num_channels, _g, _p = x.shape
        if self.tokenmixer:
            x = self.tokenmixer(x) + x
        else:
            y = nn.LayerNorm([_num_channels, _g, _p]).to(x.device)(x)     
            y=torch.swapaxes(y, -1, -3)   
            y = self.linear1(y)
            y=torch.swapaxes(y, -1, -3)
            y = F.gelu(y)

            y = self.gridgatingunit(y) 

            y=torch.swapaxes(y, -1, -3) 
            y = self.linear2(y)
            y=torch.swapaxes(y, -1, -3)  
            #y = F.dropout(y,self.dropout_rate,deterministic)   
            x = x + y
        x = unblock_images_einops(x, grid_size=(gd, gh, gw), patch_size=(fd, fh, fw))
        return x # b c d h w

class BlockTokenMixLayer(nn.Module):  
    def __init__(self,in_channel,block_size, tokenmixer=None, bias=True,factor=2,dropout_rate=0.0):
        super().__init__()
        self.in_channel=in_channel
        self.block_size=block_size
        self.factor=factor
        self.dropout_rate=dropout_rate
        self.bias=bias
        self.tokenmixer = tokenmixer
        self.linear1=nn.Linear(self.in_channel,self.in_channel* self.factor, bias=self.bias) 
        self.linear2=nn.Linear(self.in_channel,self.in_channel, bias=self.bias)
        
        if tokenmixer is None:
            self.blockgatingunit=BlockGatingUnit(self.block_size[0]*self.block_size[1]*self.block_size[2])  
        else:
            self.tokenmixer = tokenmixer(in_channel)
            #self.tokenmixer = tokenmixer(in_channel, flag=0)
  
    def forward(self, x, deterministic=True):
        n,c,d,h,w = x.shape
        fd, fh, fw = self.block_size
        gd, gh, gw = d // fd, h // fh, w // fw
        x = block_images_einops(x, patch_size=(fd, fh, fw))
        # MLP2: Local (block) mixing part, provides within-block communication.
        if self.tokenmixer:
            x = self.tokenmixer(x) + x
        else:         
            y = nn.LayerNorm([x.shape[1],x.shape[2],x.shape[3]]).to(x.device)(x)
            y=torch.swapaxes(y,-1,-3)
            y = self.linear1(y)
            y=torch.swapaxes(y,-1,-3)
            y = F.gelu(y)
            y = self.blockgatingunit(y)
            y=torch.swapaxes(y,-1,-3)
            y = self.linear2(y)
            y=torch.swapaxes(y,-1,-3)
            #y = F.dropout(y,self.dropout_rate,deterministic)
            x = x + y
        x = unblock_images_einops(x, grid_size=(gd, gh, gw), patch_size=(fd, fh, fw))
        return x    

class Conv1x1(nn.Module):   
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, pad=0,bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias)
    def forward(self, x):
        return self.conv(x)
class Conv3x3(nn.Module):   
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pad=1,bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias)
    def forward(self, x):
        return self.conv(x)
class ConvT_up(nn.Module):  
    def __init__(self, in_channels, out_channels, pad=0,bias=True):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, padding=pad, bias=bias)
    def forward(self, x):
        return self.conv(x)
class Conv_down(nn.Module):  
    def __init__(self, in_channels, out_channels, pad=1,bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=4, stride=2, padding=pad, bias=bias)
    def forward(self, x):
        return self.conv(x)

def block_images_einops(x, patch_size):
    """Image to patches."""
    batch, channels,depth, height, width = x.shape
    x=x.permute(0,2,3,4,1)
    grid_depth = depth // patch_size[0]
    grid_height = height // patch_size[1]
    grid_width = width // patch_size[2]
    
    # b g p c
    x = einops.rearrange(
      x, "n (gd fd) (gh fh) (gw fw) c -> n (gd gh gw) (fd fh fw) c",
      gd=grid_depth, gh=grid_height, gw=grid_width, fd = patch_size[0], fh=patch_size[1], fw=patch_size[2])
    return x.permute(0,3,1,2) # b c g p

def unblock_images_einops(x, grid_size, patch_size):
    """patches to images."""
    # b c g p 
    x=x.permute(0,2,3,1) 
    # b g p c
    x = einops.rearrange(
      x, "n (gd gh gw) (fd fh fw) c -> n (gd fd) (gh fh) (gw fw) c",
      gd=grid_size[0], gh=grid_size[1], gw=grid_size[2], fd=patch_size[0], fh=patch_size[1], fw=patch_size[2])
    return x.permute(0,4,1,2,3) #  b c d h w

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, D, H, W)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, window_size, C)
    """
    x = x.permute(0,2,3,4,1)
    B, C, D, H, W = x.shape

    x = x.view(B, D // window_size[0], window_size[0], H // window_size[1], window_size[1], W // window_size[2], window_size[2], C)

    windows = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0], window_size[1], window_size[2], C)
    return windows

def window_reverse(windows, window_size, H, W, L):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, window_size, C)
        window_size (int): Window size
     ]   H (int): Height of image
        W (int): Width of image
        L (int): Length of image
    Returns:
        x: (B, H, W, L, C)
    """
    B = int(windows.shape[0] / (H * W * L / window_size[0] / window_size[1] / window_size[2]))
    x = windows.view(B, H // window_size[0], W // window_size[1], L // window_size[2], window_size[0], window_size[1], window_size[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(B, H, W, L, -1)
    return 

class MlpBlock(nn.Module):   
    """A 1-hidden-layer MLP block, applied over the last dimension."""
    def __init__(self,in_channel,mlp_dim,dropout_rate=0.0,bias=True):
        super().__init__()
        self.in_channel=in_channel
        self.mlp_dim = mlp_dim
        self.bias = bias
        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.linear1=nn.Linear(self.in_channel,self.mlp_dim,bias=self.bias)
        self.linear2=nn.Linear(self.mlp_dim,self.in_channel,bias=self.bias)
  
    def forward(self, x):
        x = einops.rearrange( x, "n c d h w-> n d h w c")
        x = self.linear1(x)
        x = F.gelu(x)
        #x = F.dropout(x,p=self.dropout_rate) 
        x = self.dropout(x)
        x = self.linear2(x)
        return einops.rearrange( x, "n d h w c -> n c d h w")
 
   
class GridGatingUnit(nn.Module):  
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the second last.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self,h_size,bias=True):   
        super().__init__()
        self.h_size=h_size
        self.bias=bias
        self.linear=nn.Linear(self.h_size,self.h_size,bias=self.bias)
    def forward(self, x): 
        u,v = np.split(x, 2, axis=1)     
        v_b, v_c, v_h, v_w = v.shape
        v_fun = nn.LayerNorm([v_c, v_h, v_w]).to(x.device)
        v=v_fun(v)
        v = torch.swapaxes(v, -1, -2)  
        v = self.linear(v) 
        v = torch.swapaxes(v, -1, -2)
        return u * (v + 1.)
    
class BlockGatingUnit(nn.Module):   
    """A SpatialGatingUnit as defined in the gMLP paper.
    The 'spatial' dim is defined as the **second last**.
    If applied on other dims, you should swapaxes first.
    """
    def __init__(self,w_size,bias=True):
        super().__init__()
        self.w_size=w_size    
        self.bias=bias
        self.linear=nn.Linear(self.w_size, self.w_size,bias=self.bias)
  
    def forward(self, x):
        u, v = np.split(x, 2, axis=1)
        v = nn.LayerNorm([v.shape[1],v.shape[2],v.shape[3]]).to(x.device)(v)
        v = self.linear(v)
        return u * (v + 1.)
     
class CALayer(nn.Module):    
    """Squeeze-and-excitation block for channel attention.
    ref: https://arxiv.org/abs/1709.01507
    """
    def __init__(self,in_channel,features,reduction=4,bias=True): 
        super().__init__()
        self.in_channel=in_channel
        self.features=features
        self.reduction=reduction
        self.bias=bias
        self.conv_1=Conv1x1(self.in_channel,self.features//self.reduction, bias=self.bias)
        self.conv_2=Conv1x1(self.features//self.reduction,self.features, bias=self.bias)
    def forward(self, x):
        # 2D global average pooling
        y = torch.mean(x, dim=[2,3,4],keepdim=True)  
        # Squeeze (in Squeeze-Excitation)
        y = self.conv_1(y)
        y_fun1 = nn.ReLU()
        y=y_fun1(y)
        # Excitation (in Squeeze-Excitation)
        y=self.conv_2(y)
        y_fun2 = nn.Sigmoid()
        y=y_fun2(y)
        return x * y 

class RCAB(nn.Module):
    def __init__(self,features,reduction=4,lrelu_slope=0.2,bias=True):
        super().__init__()
        self.features=features
        self.reduction=reduction
        self.lrelu_slope=lrelu_slope
        self.bias=bias
        self.conv3_1=Conv3x3(in_channels=self.features,out_channels=self.features,bias=self.bias)   
        #self.conv3_2=Conv3x3(in_channels=self.features,out_channels=self.features,bias=self.bias)
        self.calayer=CALayer(in_channel=self.features,features=self.features, reduction=self.reduction,bias=self.bias)

    def forward(self, x):
        shortcut = x
        n,c,d,h,w=x.shape
        x=nn.LayerNorm([c,d,h,w]).to(x.device)(x)
        x = self.conv3_1(x)    
        x = nn.functional.leaky_relu(x, negative_slope=self.lrelu_slope)  
        #x = self.conv3_2(x)
        x = self.calayer(x)
        return x + shortcut   

class RDCAB(nn.Module):     
    """Residual dense channel attention block. Used in Bottlenecks."""
    def __init__(self,in_channel,features,reduction= 4,bias= True,dropout_rate = 0.0):   # reduction used to be 16
        super().__init__()
        self.in_channel=in_channel
        self.features = features
        self.reduction = reduction
        self.bias=bias
        self.dropout_rate=dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.mlpblock=MlpBlock(in_channel=self.in_channel,mlp_dim=self.features,dropout_rate=self.dropout_rate,bias=self.bias)
        self.calayer=CALayer(in_channel=self.in_channel,features=self.features,reduction=self.reduction,bias=self.bias)
  
    def forward(self, x, deterministic=True):
        y = nn.LayerNorm([x.shape[1],x.shape[2],x.shape[3], x.shape[4]]).to(x.device)(x)
        y = self.mlpblock(y)
        y = self.dropout(y)
        y = self.calayer(y)
        x = x + y
        return x
    
class GetSpatialGatingWeights(nn.Module):  
    """Get gating weights for cross-gating MLP block."""
    def __init__(self,in_channel,block_size,grid_size,input_proj_factor=2,dropout_rate=0.0,bias=True):
        super().__init__()
        self.in_channel=in_channel
        self.block_size=block_size
        self.grid_size=grid_size
        self.input_proj_factor=input_proj_factor
        self.dropout_rate=dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)
        self.bias=bias
        self.linear1=nn.Linear(self.in_channel,self.in_channel * self.input_proj_factor,bias=self.bias)
        self.linear2= nn.Linear(self.grid_size[0]*self.grid_size[1]*self.grid_size[2],self.grid_size[0]*self.grid_size[1]*self.grid_size[2], bias=self.bias)
        self.linear3= nn.Linear(self.block_size[0]*self.block_size[1]*self.block_size[2],self.block_size[0]*self.block_size[1]*self.block_size[2], bias=self.bias)
        self.linear4=nn.Linear(self.in_channel * self.input_proj_factor,self.in_channel,bias=self.bias)
  
    def forward(self, x):
        n, c ,d, h, w = x.shape

        # input projection
        x = nn.LayerNorm([c,d,h,w]).to(x.device)(x)
        x= torch.swapaxes(x,-1,-4)
        x = self.linear1(x)
        x= torch.swapaxes(x,-1,-4)
        x = F.gelu(x)
        u, v = np.split(x, 2, axis=1)

        # Get grid weights
        gd, gh, gw = self.grid_size
        fd, fh, fw = d // gd, h // gh, w // gw
        # u: b c g w
        u = block_images_einops(u, patch_size=(fd, fh, fw))
        u = torch.swapaxes(u, -1, -2)
        u = self.linear2(u) 
        u = torch.swapaxes(u, -1, -2)

        u = unblock_images_einops(u, grid_size=(gd, gh, gw), patch_size=(fd, fh, fw))
        # Get Block weights
        fd, fh, fw = self.block_size
        gd, gh, gw = d // fd, h // fh, w // fw

        v = block_images_einops(v, patch_size=(fd, fh, fw))

        v = self.linear3(v)

        v = unblock_images_einops(v, grid_size=(gd, gh, gw), patch_size=(fd, fh, fw))
        x = torch.cat([u, v], dim=1)

        x = torch.swapaxes(x, -1, -4)
        x = self.linear4(x)
        x = torch.swapaxes(x, -1, -4)
        #x = F.dropout(x,p=self.dropout_rate)
        x = self.dropout(x)
        
        return x
    
class CrossGatingBlock(nn.Module):   
    """Cross-gating MLP block."""
    def __init__(self, features,grid_size,block_size, use_checkpoint=False, dropout_rate = 0.0,input_proj_factor = 2,upsample_y = False,bias = True):
        super().__init__()
        self.features=features
        self.grid_size=grid_size
        self.block_size=block_size
        self.dropout_rate = dropout_rate
        self.input_proj_factor = input_proj_factor
        self.upsample_y = upsample_y
        self.bias = bias
        self.use_checkpoint = use_checkpoint

        self.out = self.outputs(self.start_channel * 2, self.out_dim, kernel_size=3, stride=1, padding=1, bias=False)

        self.up1 = self.decoder(self.start_channel * 8, self.start_channel * 8)
        self.up2 = self.decoder(self.start_channel * 4, self.start_channel * 4)
        self.up3 = self.decoder(self.start_channel * 2, self.start_channel * 2)
        self.up4 = self.decoder(self.start_channel * 2, self.start_channel * 2)
    
    def encoder(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.PReLU())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.PReLU())
        return layer
    
    def decoder(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.PReLU())
        return layer
    
    def outputs(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                bias=False, batchnorm=False):
        if batchnorm:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.BatchNorm3d(out_channels),
                nn.Tanh())
        else:
            layer = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
                nn.Softsign())
        return layer
    
    def forward(self,x, y):
        # cgb x(skip_half+enc) y(dec out_half)
        
        x_in = torch.cat((x, y), 1)
        e0 = self.inputenc(x_in)
        e0 = self.enc1(e0)

        e1 = self.down1(e0)
        e1 = self.enc2(e1)

        e2 = self.down2(e1)
        e2 = self.enc3(e2)

        e3 = self.down3(e2)
        e3 = self.enc4(e3)
        
        e4 = self.down4(e3)
        e4 = self.bottleneck(e4)

        d0 = torch.cat((self.up1(e4), e3), 1)

        d0 = self.dc1(d0)
        d0 = self.dc2(d0)

        d1 = torch.cat((self.up2(d0), e2), 1)

        d1 = self.dc3(d1)
        d1 = self.dc4(d1)

        d2 = torch.cat((self.up3(d1), e1), 1)

        d2 = self.dc5(d2)
        d2 = self.dc6(d2)
        
        f_xy = self.out(d2)
        f_xy_0 = torch.nn.functional.interpolate(f_xy[:,0:1,...], size=[160, 192, 224], mode='trilinear')
        f_xy_1 = torch.nn.functional.interpolate(f_xy[:,1:2,...], size=[160, 192, 224], mode='trilinear')
        f_xy_2 = torch.nn.functional.interpolate(f_xy[:,2:3,...], size=[160, 192, 224], mode='trilinear')
        f_xy = torch.cat((f_xy_0, f_xy_1, f_xy_2), 1)

        return f_xy
      
class SpatialTransform(nn.Module):
    def __init__(self):
        super(SpatialTransform, self).__init__()
    def forward(self, mov_image, flow, mod = 'bilinear'):
        d2, h2, w2 = mov_image.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.to(mov_image.device).float()
        grid_d = grid_d.to(mov_image.device).float()
        grid_w = grid_w.to(mov_image.device).float()
#         grid_d = nn.Parameter(grid_d, requires_grad=False)
#         grid_w = nn.Parameter(grid_w, requires_grad=False)
#         grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow_d = flow[:,:,:,:,0]
        flow_h = flow[:,:,:,:,1]
        flow_w = flow[:,:,:,:,2]
        
        # Remove Channel Dimension
        disp_d = (grid_d + (flow_d)).squeeze(1)
        disp_h = (grid_h + (flow_h)).squeeze(1)
        disp_w = (grid_w + (flow_w)).squeeze(1)
        sample_grid = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
        warped = torch.nn.functional.grid_sample(mov_image, sample_grid, mode = mod, align_corners = True)
        
        return warped
    # submission to l2r task3 with 
    #disp_field = flow.detach().cpu().numpy()[0]
    #disp_field = np.array([zoom(disp_field[i], 0.5, order=2) for i in range(3)])
    
    # l2r original
    #D,H,W = fixed_seg.shape
    #identity = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    #warped_seg = map_coordinates(moving_seg, identity + disp_field.transpose(3,0,1,2), order=0)
    
class DiffeomorphicTransform(nn.Module):
    def __init__(self, time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        self.time_step = time_step

    def forward(self, flow):
    
        # print(flow.shape)
        d2, h2, w2 = flow.shape[-3:]
        grid_d, grid_h, grid_w = torch.meshgrid([torch.linspace(-1, 1, d2), torch.linspace(-1, 1, h2), torch.linspace(-1, 1, w2)])
        grid_h = grid_h.float().to(flow.device)
        grid_d = grid_d.float().to(flow.device)
        grid_w = grid_w.float().to(flow.device)
        grid_d = nn.Parameter(grid_d, requires_grad=False)
        grid_w = nn.Parameter(grid_w, requires_grad=False)
        grid_h = nn.Parameter(grid_h, requires_grad=False)
        flow = flow / (2 ** self.time_step)
        
        
        for i in range(self.time_step):
            flow_d = flow[:,0,:,:,:]
            flow_h = flow[:,1,:,:,:]
            flow_w = flow[:,2,:,:,:]
            disp_d = (grid_d + flow_d).squeeze(1)
            disp_h = (grid_h + flow_h).squeeze(1)
            disp_w = (grid_w + flow_w).squeeze(1)
            
            deformation = torch.stack((disp_w, disp_h, disp_d), 4)  # shape (N, D, H, W, 3)
            flow = flow + torch.nn.functional.grid_sample(flow, deformation, mode='bilinear', padding_mode="border", align_corners = True)
        return flow

def smoothloss(y_pred):
    d2, h2, w2 = y_pred.shape[-3:]
    dy = torch.abs(y_pred[:,:,1:, :, :] - y_pred[:,:, :-1, :, :]) / 2 * d2
    dx = torch.abs(y_pred[:,:,:, 1:, :] - y_pred[:,:, :, :-1, :]) / 2 * h2
    dz = torch.abs(y_pred[:,:,:, :, 1:] - y_pred[:,:, :, :, :-1]) / 2 * w2
    return (torch.mean(dx * dx)+torch.mean(dy*dy)+torch.mean(dz*dz))/3.0

"""
Normalized local cross-correlation function in Pytorch. Modified from https://github.com/voxelmorph/voxelmorph.
"""
class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """
    def __init__(self, win=9, eps=1e-5):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 3
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        conv_fn = F.conv3d

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size/2))
        J_sum = conv_fn(J, weight, padding=int(win_size/2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size/2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size/2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size/2))

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return -1.0 * torch.mean(cc)

class DiceLoss(nn.Module):
    """Dice and Xentropy loss"""

    def __init__(self, num_class=36):
        super().__init__()
        self.num_class = num_class

    def forward(self, y_pred, y_true):
        #y_pred = torch.round(y_pred)
        #y_pred = nn.functional.one_hot(torch.round(y_pred).long(), num_classes=7)
        #y_pred = torch.squeeze(y_pred, 1)
        #y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=self.num_class)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
        intersection = y_pred * y_true
        intersection = intersection.sum(dim=[2, 3, 4])
        union = torch.pow(y_pred, 2).sum(dim=[2, 3, 4]) + torch.pow(y_true, 2).sum(dim=[2, 3, 4])
        dsc = (2.*intersection) / (union + 1e-5)
        dsc = (1-torch.mean(dsc))
        return dsc

class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

class SAD:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean(torch.abs(y_true - y_pred))

def encoder(in_channels, out_channels, kernel_size=3, stride=1, padding=1,
            bias=False, batchnorm=False):
    if batchnorm:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nn.GELU())
    else:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.GELU())
    return layer

def decoder(in_channels, out_channels, kernel_size=2, stride=2, padding=0,
                output_padding=0, bias=True):
        layer = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                               padding=padding, output_padding=output_padding, bias=bias),
            nn.GELU())
        return layer

# dropout modified to 0
class MAXLayer(nn.Module):
    def __init__(self, in_channels, depth, block_size, grid_size, use_bias, use_checkpoint=False, dropout_rate=0.0, updown=None, last=None, token=None, split_head=True):
        # down 1 ->downsample, down 0 -> upsample, else None
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.dropout_rate = dropout_rate
        self.blks = nn.ModuleList()
        self.token = token
        for i in range(depth):
            if last is not None:
                self.blks.append(nn.Sequential(
                    MAgMLP_tokens(in_channel=in_channels, block_size=block_size, grid_size=grid_size, dropout_rate=self.dropout_rate, mixer=self.token, split_head=split_head),
                    RCAB(features = in_channels)))
            else:
                self.blks.append(nn.Sequential(
                    MAgMLP_tokens(in_channel=in_channels, block_size=block_size, grid_size=grid_size, dropout_rate=self.dropout_rate, mixer=self.token, split_head=split_head),
                    RDCAB(in_channel =in_channels, features = in_channels, reduction=4, dropout_rate = self.dropout_rate)))

        
        self.cross = CrossGatingBlock(features=in_channels,grid_size=grid_size,block_size=block_size, use_checkpoint=self.use_checkpoint, dropout_rate=self.dropout_rate)
        
        self.updown = updown
        if updown == 1:
            self.updown = encoder(in_channels, in_channels*2, stride=2, bias=use_bias)
        if updown == 0:
            self.updown = decoder(in_channels, in_channels//2, kernel_size=2, stride=2, bias=use_bias)
            
    def forward_func(self,x,y):
        for blk in self.blks:
            x = blk(x)
            y = blk(y)
        x, y = self.cross(x,y)
        x_feature = x
        y_feature = y
        if self.updown is not None:
            x = self.updown(x)
            y = self.updown(y)
        return x, x_feature, y, y_feature
    def forward(self, x,y):
        if self.use_checkpoint:
            x,x_feature,y,y_feature = checkpoint.checkpoint(self.forward_func,x,y)
            return x,x_feature,y,y_feature
        else:
            x,x_feature,y,y_feature = self.forward_func(x,y)
            return x,x_feature,y,y_feature
    
class MACG(nn.Module):
    def __init__(self,in_channel=1, out_dim=3,start_channel=8, use_checkpoint=False, use_bias=True,depth = [2,2,6,2],
                 lrelu_slope= 0.2,use_global_mlp = True, block_size = (5,6,7), grid_size = (5, 6, 7),
                 num_bottleneck_blocks = 1,block_gmlp_factor= 2,
                 grid_gmlp_factor= 2,input_proj_factor= 2, num_outputs = 3,dropout_rate = 0.0, token=dw_sep_conv, split_head=True):
        super().__init__()
        self.in_channel=in_channel
        self.start_channel = start_channel
        self.depth = depth
        self.use_bias=use_bias
        self.lrelu_slope= lrelu_slope
        self.use_global_mlp = use_global_mlp 
        self.block_size = block_size
        self.grid_size = grid_size
        self.block_gmlp_factor= block_gmlp_factor
        self.grid_gmlp_factor= grid_gmlp_factor
        self.input_proj_factor= input_proj_factor
        self.num_outputs = num_outputs
        self.dropout_rate = dropout_rate
        self.out_dim = out_dim
        self.bottleneck_reduction = 8
        self.use_checkpoint = use_checkpoint
        self.token = token
        self.split = split_head
        self.patch_embed = nn.Conv3d(self.in_channel, start_channel, kernel_size=4, stride=4)
        
        
        # stages
        self.enc1 = MAXLayer(start_channel, depth[0], self.block_size, self.grid_size, use_bias, updown=1, use_checkpoint=self.use_checkpoint, token=token,split_head=self.split)
        self.enc2 = MAXLayer(start_channel*2, depth[1], self.block_size, self.grid_size, use_bias, updown=1, use_checkpoint=self.use_checkpoint, token=token,split_head=self.split)  
        self.enc3 = MAXLayer(start_channel*4, depth[2], self.block_size, self.grid_size, use_bias, updown=1, use_checkpoint=self.use_checkpoint, token=token,split_head=self.split)
        self.bottleneck = MAXLayer(start_channel*8, depth[3], self.block_size, self.grid_size, use_bias, updown=None, last=True, token=token,split_head=self.split)
        # self.up1 = nn.ConvTranspose3d(start_channel*8, start_channel*8, kernel_size=2, stride=2, padding=0)
        
        self.dec1 = MAXLayer(start_channel*8, depth[3], self.block_size, self.grid_size, use_bias, updown=0, token=token,split_head=self.split)
        self.cat1 = nn.Conv3d(start_channel*8, start_channel*4, kernel_size=3, stride=1, padding=1)
        self.dec2 = MAXLayer(start_channel*4, depth[2], self.block_size, self.grid_size, use_bias, updown=0, token=token,split_head=self.split)
        self.cat2 = nn.Conv3d(start_channel*4, start_channel*2, kernel_size=3, stride=1, padding=1)
        self.dec3 = MAXLayer(start_channel*2, depth[1], self.block_size, self.grid_size, use_bias, updown=0,use_checkpoint=self.use_checkpoint, token=token,split_head=self.split)
        self.cat3 = nn.Conv3d(start_channel*2, start_channel, kernel_size=3, stride=1, padding=1)
        self.dec4 = MAXLayer(start_channel, depth[0], self.block_size, self.grid_size, use_bias, updown=None, use_checkpoint=self.use_checkpoint, token=token,split_head=self.split)
        self.reverse_patch_embedding = nn.ConvTranspose3d(2*start_channel, start_channel//2, (4, 4, 4), stride=4)
        
        self.outconv = nn.Conv3d(start_channel, 3, 3, stride=1, padding=1)
        self.outconv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.outconv.weight.shape))
        self.outconv.bias = nn.Parameter(torch.zeros(self.outconv.bias.shape))
        self.head = nn.Sequential(self.outconv, nn.Softsign())
        
        self.convskip = Conv3dReLU(2, start_channel//2, 3, 1, use_batchnorm=False)
        
        
    def forward(self, x, y):
        x1 = self.patch_embed(x)
        y1 = self.patch_embed(y)
        skip = self.convskip(torch.cat([x,y],dim=1))
        
        x2, x2_f, y2, y2_f = self.enc1(x1, y1)
        x3, x3_f, y3, y3_f = self.enc2(x2, y2)
        x4, x4_f, y4, y4_f = self.enc3(x3, y3)
#         y4, y4_f = self.enc3(y3)
#         x4, y4 = self.cross3(x4, y4)
        
        x5, _, y5, _= self.bottleneck(x4, y4)

        #y5, _ = self.bottleneck(y4)
        
        x_4, _, y_4, _= self.dec1(x5, y5)
        x_4 = self.cat1(torch.cat([x_4, x4_f], dim=1))
#         y_4, _ = self.dec1(y5)
        y_4 = self.cat1(torch.cat([y_4, y4_f], dim=1))
#         x_4, y_4 = self.cross4(x_4, y_4)

        
        x_3,_, y_3, _= self.dec2(x_4, y_4)
        x_3 = self.cat2(torch.cat([x_3, x3_f], dim=1))
#         y_3,_ = self.dec2(y_4)
        y_3 = self.cat2(torch.cat([y_3, y3_f], dim=1))
#         x_3, y_3 = self.cross5(x_3, y_3)

        x_2, _, y_2, _ = self.dec3(x_3, y_3)
        x_2 = self.cat3(torch.cat([x_2, x2_f], dim=1))
#         y_2, _ = self.dec3(y_3)
        y_2 = self.cat3(torch.cat([y_2, y2_f], dim=1))

        x_1, _, y_1, _ = self.dec4(x_2, y_2)
#         y_1, _ = self.dec4(y_2)
        x_y = self.reverse_patch_embedding(torch.cat([x_1,y_1], dim=1))
            
        out = self.head(torch.cat([x_y, skip], dim=1))

    
        return out

class Conv3dIN(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dIN, self).__init__(conv, nm)    

class Conv3dINReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=False,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dINReLU, self).__init__(conv, nm, relu)
        
class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=False,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, relu)
        
