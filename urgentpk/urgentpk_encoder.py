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

class stft_encoder(torch.nn.Module):
    def __init__(self, n_fft=320, hop_length=160, win_length=320):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, input):
        B, C, L = input.shape
        input = input.contiguous().view(-1, L) # [B * C, L]
        # print(input.shape, flush=True)
        spec = torch.stft(input, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length, return_complex=True)
        spec = torch.view_as_real(spec) # [B * C, N, T, 2]
        # print(spec.shape, flush=True)
        spec = torch.concat((spec[:, :, :, 0], spec[:, :, :, 1]), dim=1) # [B * C, 2 * N, T]
        # spec = spec.view(B, C, -1, spec.shape[-1]).permute(0, 1, 3, 2).contiguous() # [B, C, T, 2 * N]
        spec = spec.contiguous().view(B, C, -1, spec.shape[-1]) # [B, C, 2 * N, T]
        # print(spec.shape, flush=True)
        return spec

if __name__ == "__main__":
    input = torch.randn((8, 2, 10000))
    model = stft_encoder()
    out = model(input)
    print(out.shape)