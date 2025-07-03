from typing import Any
import lightning as L
from torch.optim.optimizer import Optimizer
# from transformers import AdamW, get_linear_schedule_with_warmup
import torch
# from urgentpk.wavLM import WavLM, WavLMConfig
import torchaudio
# import utmos
# from utmos.lightning_module import BaselineLightningModule
# from utmos.model import Projection, SSL_model
from urgentpk.resnet import ResNet34, ResNet18
import os
from tqdm import tqdm
import torch.nn.functional as F
from urgentpk.urgentpk_encoder import stft_encoder
from urgentpk.urgentpk_backbone import choose_model

class Config:
    def __init__(
        self,
        learning_rate=1e-4,
        batch_size=4,
        weight_decay=1e-6,
        adam_epsilon=1e-8,
        warmup_steps=2,
        num_worker=4,
        num_train_epochs=150,
        gradient_accumulation_steps=1,
        device="cuda",
        train_version=0,
        train_tag="debug",
        train_name='ab_test',
        val_check_interval=1000,
        save_top_k=3,
        resume=True,
        seed=1996,
        fs=16000,
        gradient_clip=0.5,
        lr_step_size=1,
        lr_gamma=0.85,
        dataset="/home/wangyou.zhang/urgent2024_challenge/submissions/",
        delta=0.0,
        # data_path="/home/wangyou.zhang/urgent2024_challenge/submissions/", # local/virtual_team_vctk_train
        encoder="utmos",
        backbone="resnet34",
        tune_utmos=False,
        choose_model="AbModel",
        init_ckpt="None"
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.adam_epsilon = adam_epsilon
        self.warmup_steps = warmup_steps
        self.num_worker = num_worker
        self.num_train_epochs = num_train_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.train_version = train_version
        self.train_tag = train_tag
        self.train_name = train_name
        self.val_check_interval = val_check_interval
        self.save_top_k = save_top_k
        self.resume = resume
        self.seed = seed
        self.dataset = dataset
        self.delta = delta
        # self.data_path = data_path
        self.encoder = encoder
        self.backbone = backbone
        self.fs = fs
        self.gradient_clip = gradient_clip
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.tune_utmos = tune_utmos
        self.choose_model = choose_model
        self.init_ckpt = init_ckpt

class AbModel(torch.nn.Module):
    def __init__(self, cfg: Config=None):
        super().__init__()
        self.encoder_name = cfg.encoder
        self.backbone = cfg.backbone
        assert self.encoder_name in ["mel", "stft"], "encoder should be 'mel' or 'stft'!"
        if self.encoder_name == "mel":
            self.encoder = torch.nn.Sequential(torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=320, win_length=160, n_mels=120),
                                               torchaudio.transforms.AmplitudeToDB())
            self.feat_dim = 120
        elif self.encoder_name == "stft":
            self.encoder = stft_encoder(n_fft=320, hop_length=160, win_length=320)
            self.feat_dim = 322
        '''
        elif self.encoder_name == "utmos":
            self.utmos_model = BaselineLightningModule.load_from_checkpoint('/home/chenda.li/workspace/urgent26/abtest_model/utmos/model.ckpt')
            if cfg is not None and not cfg.tune_utmos:
                for p in self.utmos_model.parameters():
                    p.requires_grad = False
            else:
                pass
            del self.utmos_model.output_layers[-1]
            self.encoder = self.utmos_model.forward_feature
            self.feat_dim = 1024
        '''
        # encoder output: [B, 2, L, N]

        self.model = choose_model(model_name=cfg.backbone, feat_dim=self.feat_dim)
        # model output: [B, 3]

        print("### AbModel init ###")
        print("encoder:", self.encoder_name, flush=True)
        print("backbone:", self.backbone, flush=True)
        print("feature dim:", self.feat_dim, flush=True)
        # print("tune utmos:", cfg.tune_utmos)
    
    def forward(self, input):
        B, C, L = input.shape
        '''
        if self.encoder_name == "utmos":
            batch = {'wav': input.reshape(B*C, L),
                     'domains': torch.tensor([0]*B*C),
                     'judge_id': torch.tensor([288]*B*C)}
            feat = self.utmos_model.forward_feature(batch) # self.encoder(batch)
            feat = feat.view(B, C, feat.shape[-2], feat.shape[-1])
        else:
        '''
        feat = self.encoder(input).permute(0, 1, 3, 2).contiguous()

        sp = feat.shape
        assert sp[-1] == self.feat_dim and sp[0] == B and sp[1] == 2 and len(sp) == 4, "feature shape should be [B, 2, L, N]!"

        feat = self.model(feat)
        assert feat.shape == (B, 3), "output shape should be [B, 3]!"

        out = feat[:, 0:1]
        scores = feat[:, 1:3]
        return out, scores

class AbTestModel(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()

        self.save_hyperparameters()
        self.cfg = cfg
        self.nn = torch.nn.Linear(100, 100)
        print("### AbTestModel init ###")
        print("dataset", cfg.dataset)
        print("init checkpoint:", cfg.init_ckpt)
        print("lr:", cfg.learning_rate)

        self.model = AbModel(cfg=cfg)
        
        self.loss = torch.nn.BCELoss()
        self.mse = torch.nn.MSELoss()
        self.kl = torch.nn.KLDivLoss()


    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:

        return
    
    def forward_step(self, batch, stage='train'):

        audios, score = batch

        label = (score[:, 0] > score[:, 1]).float().detach()

        loss = 0
        acc = 0

        batch_size = len(audios)

        predicted, p_scores = self.model(audios)
        predicted = predicted.view(-1)

        logits = torch.sigmoid(predicted)

        loss = self.loss(logits, label)
        if p_scores is not None:
            mse_loss = self.mse(score, p_scores)
        else:
            mse_loss = 0

        loss += loss + mse_loss

        acc = (((logits > 0.5) == label).sum() / batch_size).item()


        self.log(f'{stage}_loss', loss.detach().item(), on_step=True, prog_bar=True, batch_size=batch_size)
        self.log(f'{stage}_mse_loss', mse_loss if isinstance(mse_loss, int) else mse_loss.detach().item(), on_step=True, prog_bar=True, batch_size=batch_size)
        self.log(f'{stage}_acc', acc, on_step=True, prog_bar=True,batch_size=batch_size )
        self.log(f'lr', self.optimizers().param_groups[0]["lr"], on_step=True, prog_bar=True,batch_size=batch_size )
        return loss, acc


    def training_step(self, batch):
        
        loss, acc = self.forward_step(batch)
        
        return loss

    def validation_step(self, batch):
        loss, acc = self.forward_step(batch, stage='val')

        return {'loss': loss.detach()}


    def configure_optimizers(self):


        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            eps=self.cfg.adam_epsilon,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.cfg.lr_step_size, gamma=self.cfg.lr_gamma)

        return [optimizer], [scheduler]

if __name__ == "__main__":
    encoder = "mel"
    backbone = "resnet34"
    config = Config()
    input = torch.randn((8, 2, 10000)).to("cuda:0")
    config.backbone = backbone
    config.encoder = encoder
    model = AbModel(config).to("cuda:0")
    out = model(input)
    assert out[0].shape == (8, 1) and out[1].shape == (8, 2)
