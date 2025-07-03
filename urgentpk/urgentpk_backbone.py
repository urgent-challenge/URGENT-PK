from typing import Any
import lightning as L
from torch.optim.optimizer import Optimizer
# from transformers import AdamW, get_linear_schedule_with_warmup
import torch
# from wavLM import WavLM, WavLMConfig
# import torchaudio
# import utmos
# from utmos.lightning_module import BaselineLightningModule
# from utmos.model import Projection
from urgentpk.resnet import ResNet34, ResNet18
import os
from tqdm import tqdm
import torch.nn.functional as F
from torch import nn

def choose_model(model_name="resnet34",
                 feat_dim=1024):
    if model_name == "mlp":
        return mlp(feat_dim=feat_dim)
    elif model_name == "resnet34":
        return resnet34(feat_dim=feat_dim)
    elif model_name == "resnet34_mlp":
        return resnet34_mlp(feat_dim=feat_dim)
    elif model_name == "resnet18":
        return resnet18(feat_dim=feat_dim)
    elif model_name == "att_resnet34":
        return att_resnet34(feat_dim=feat_dim)
    elif model_name == "att_resnet34_mlp":
        return att_resnet34_mlp(feat_dim=feat_dim)
    elif model_name == "att_resnet18":
        return att_resnet18(feat_dim=feat_dim)
    else:
        print("model not supported yet, using default mlp!")
        return mlp(feat_dim=feat_dim)

'''
input: [B, 2, L, N]
output: [B, 3]
'''

class mlp(torch.nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * feat_dim, 512),
                                 nn.ReLU(),
                                 nn.Linear(512, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 3),
                                 nn.ReLU())
    
    def forward(self, input):
        B, C, L, N = input.shape
        feat = input.permute(0, 2, 1, 3).contiguous() # [B, L, 2, N]
        feat = feat.view(B, L, -1)
        feat = self.mlp(feat).mean(dim=1)
        return feat

class resnet34(torch.nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.resnet = ResNet34(input_channels=2, m_channels=32, feat_dim=feat_dim, emb_dim=3, dropout=0.3)
    
    def forward(self, input):
        B, C, L, N = input.shape
        input = input.permute(0, 1, 3, 2).contiguous() # [B, C, N, L]
        feat = self.resnet(input)
        assert feat.shape[-1] == 3
        return feat

class resnet34_mlp(torch.nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.resnet = ResNet34(input_channels=2, m_channels=32, feat_dim=feat_dim, emb_dim=256, dropout=0.3)
        self.mlp = Projection(hidden_dim=512, activation=torch.nn.ReLU(), range_clipping=False, input_dim=256, output_dim=3)
    
    def forward(self, input):
        B, C, L, N = input.shape
        input = input.permute(0, 1, 3, 2).contiguous() # [B, C, N, L]
        feat = self.resnet(input)
        feat = self.mlp(feat, None)
        assert feat.shape[-1] == 3
        return feat
    
class resnet18(torch.nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.resnet = ResNet18(input_channels=2, m_channels=32, feat_dim=feat_dim, emb_dim=3, dropout=0.3)
    
    def forward(self, input):
        B, C, L, N = input.shape
        input = input.permute(0, 1, 3, 2).contiguous() # [B, C, N, L]
        feat = self.resnet(input)
        assert feat.shape[-1] == 3
        return feat
    
class att_resnet34(torch.nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=feat_dim, num_heads=2, bias=True)
        self.resnet = ResNet34(input_channels=2, m_channels=32, feat_dim=feat_dim, emb_dim=3, dropout=0.3)
    
    def forward(self, input):
        B, C, L, N = input.shape
        feat_1 = input[:, 0, :, :] # [B, L, N]
        feat_2 = input[:, 1, :, :]
        cross_1 = self.attention(feat_1, feat_2, feat_2)[0].unsqueeze(1).permute(0, 1, 3, 2).contiguous() # [B, 1, N, L]
        cross_2 = self.attention(feat_2, feat_1, feat_1)[0].unsqueeze(1).permute(0, 1, 3, 2).contiguous() # [B, 1, N, L]
        feat = torch.concat([cross_1, cross_2], dim=1) # [B, 2, N, L]
        feat = self.resnet(feat)
        return feat
    
class att_resnet34_mlp(torch.nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=feat_dim, num_heads=2, bias=True)
        self.resnet = ResNet34(input_channels=2, m_channels=32, feat_dim=feat_dim, emb_dim=256, dropout=0.3)
        self.mlp = Projection(hidden_dim=512, activation=torch.nn.ReLU(), range_clipping=False, input_dim=256, output_dim=3)
    
    def forward(self, input):
        B, C, L, N = input.shape
        feat_1 = input[:, 0, :, :] # [B, L, N]
        feat_2 = input[:, 1, :, :]
        cross_1 = self.attention(feat_1, feat_2, feat_2)[0].unsqueeze(1).permute(0, 1, 3, 2).contiguous() # [B, 1, N, L]
        cross_2 = self.attention(feat_2, feat_1, feat_1)[0].unsqueeze(1).permute(0, 1, 3, 2).contiguous() # [B, 1, N, L]
        feat = torch.concat([cross_1, cross_2], dim=1) # [B, 2, N, L]
        feat = self.resnet(feat)
        feat = self.mlp(feat, None)
        return feat

class att_resnet18(torch.nn.Module):
    def __init__(self, feat_dim=1024):
        super().__init__()
        self.attention = torch.nn.MultiheadAttention(embed_dim=feat_dim, num_heads=2, bias=True)
        self.resnet = ResNet18(input_channels=2, m_channels=32, feat_dim=feat_dim, emb_dim=3, dropout=0.3)
    
    def forward(self, input):
        B, C, L, N = input.shape
        feat_1 = input[:, 0, :, :] # [B, L, N]
        feat_2 = input[:, 1, :, :]
        cross_1 = self.attention(feat_1, feat_2, feat_2)[0].unsqueeze(1).permute(0, 1, 3, 2).contiguous() # [B, 1, N, L]
        cross_2 = self.attention(feat_2, feat_1, feat_1)[0].unsqueeze(1).permute(0, 1, 3, 2).contiguous() # [B, 1, N, L]
        feat = torch.concat([cross_1, cross_2], dim=1) # [B, 2, N, L]
        feat = self.resnet(feat)
        return feat
