import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

from einops import rearrange
from monai.networks.blocks.unetr_block import UnetrUpBlock

class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, dropout=0, max_len:int=5000) -> None:
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # size=(1, L, d_model)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + nn.Parameter(self.pe[:, :x.size(1)],requires_grad=False) #size = [batch, L, d_model]
        return self.dropout(x) # size = [batch, L, d_model]
    
class LFM(nn.Module):
    def __init__(self, num_channels):
        super(LFM, self).__init__()
        
    def make_gaussian(self, y_idx, x_idx, height, width, sigma=7):
        yv, xv = torch.meshgrid([torch.arange(0, height), torch.arange(0, width)])
        yv = yv.unsqueeze(0).float().cuda()
        xv = xv.unsqueeze(0).float().cuda()
        g = torch.exp(- ((yv - y_idx) ** 2 + (xv - x_idx) ** 2) / (2 * sigma ** 2))
        return g.unsqueeze(0)       #1, 1, H, W
    
    def forward(self, x, sigma):
        b, c, h, w = x.shape
        x = x.float()
        y = torch.fft.fft2(x.cpu()).to(x.device)
        h_idx, w_idx = h // 2, w // 2
        high_filter = self.make_gaussian(h_idx, w_idx, h, w, sigma=sigma)
        
        y = y * (1 - high_filter)
        y_imag = y.imag
        y_real = y.real
        
        y = torch.complex(y_real, y_imag)
        y = torch.fft.ifft2(y.cpu(), s=(h, w)).to(y.device).float()
        return x + y
    
class DecoderLayer(nn.Module):
    def __init__(self, in_channels:int, output_text_len:int, input_text_len:int=24, embed_dim:int=768):
        super(DecoderLayer, self).__init__()
        self.in_channels = in_channels
        self.lfm = LFM(in_channels)
        self.sig = nn.Parameter(torch.tensor(5.0),requires_grad=True) 
        self.self_attn_norm = nn.LayerNorm(in_channels)
        self.cross_attn_norm = nn.LayerNorm(in_channels)
        self.self_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=8,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(embed_dim=in_channels,num_heads=8,batch_first=True)
        self.text_project = nn.Sequential(
            nn.Conv1d(input_text_len,output_text_len,kernel_size=1,stride=1),
            nn.GELU(),
            nn.Linear(embed_dim,in_channels),
            nn.LeakyReLU(),
        )
        self.vis_pos = PositionalEncoding(in_channels)
        self.txt_pos = PositionalEncoding(in_channels,max_len=output_text_len)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.scale1 = nn.Parameter(torch.tensor(1.421),requires_grad=True)
        self.scale2 = nn.Parameter(torch.tensor(1.421),requires_grad=True)
        self.scale3 = nn.Parameter(torch.tensor(1.421),requires_grad=True)
        self.lin1 = nn.Linear(in_channels, in_channels // 2)
        self.lin2 = nn.Linear(in_channels // 2, in_channels)
        self.norm = nn.LayerNorm(in_channels)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self,x,txt):
        '''
        x:[B N C1]
        txt:[B,L,C]
        '''
        txt = self.text_project(txt)
        
        B, N, C = x.shape
        x = x.permute(0,2,1).view(B,C,int(np.sqrt(N)),int(np.sqrt(N)))
        x = self.lfm(x,self.sig)
        x = x.view(B,C,-1).permute(0,2,1)
        
        # Cross-Attention
        vis1 = self.norm2(x)
        vis1 = self.cross_attn(query=self.vis_pos(vis1),
                                key=self.txt_pos(txt),
                                value=txt)[0]
        vis1 = self.cross_attn_norm(vis1)
        vis1 = x + self.scale1 * vis1
        
        # Self-Attention
        vis2 = self.norm1(vis1)
        vis2 = self.self_attn(query=self.vis_pos(vis2),
                                key=self.vis_pos(vis2),
                                value=vis2)[0]
        vis2 = self.self_attn_norm(vis2)
        vis2 = vis1 + self.scale2 * vis2
        
        out = self.lin2(self.dropout(F.relu(self.lin1(vis2))))
        out = vis2 + self.scale3 * out
        return out

class Decoder(nn.Module):
    def __init__(self,in_channels, out_channels, spatial_size, text_len) -> None:
        super().__init__()
        self.guide_layer = DecoderLayer(in_channels,text_len)   # for skip
        self.spatial_size = spatial_size
        self.decoder = UnetrUpBlock(2,in_channels,out_channels,3,2,norm_name='BATCH')
        
    def forward(self, vis, skip_vis, txt):
        if txt is not None:
            vis =  self.guide_layer(vis, txt)
        vis = rearrange(vis,'B (H W) C -> B C H W',H=self.spatial_size,W=self.spatial_size)
        skip_vis = rearrange(skip_vis,'B (H W) C -> B C H W',H=self.spatial_size*2,W=self.spatial_size*2)
        output = self.decoder(vis,skip_vis)
        output = rearrange(output,'B C H W -> B (H W) C')
        return output