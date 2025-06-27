#!/bin/env python

import os
from torch import optim, nn, utils, Tensor
# from torchvision.datasets import MNIST
# from torchvision.transforms import ToTensor
import lightning as L
import torch
import argparse
from argparse import ArgumentParser
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import  ModelCheckpoint
import torch.multiprocessing as mp
# from model import AbTestModel, Config
# from model_resnet import AbTestModel, Config
# from model_mod import AbTestModel, Config
from urgentpk_model import AbTestModel, Config
from prepare_dataset import ABDataset_urgent24, ABDataset_urgent25, MyDataModule, ABDataset_vctk, ABDataset_urgent25_integrate


def config_parser():
    cfg = Config()
    parameters = vars(cfg)

    parser = ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    for par in parameters:
        default = parameters[par]
        parser.add_argument(f"--{par}", type=str2bool if isinstance(default, bool) else type(default), default=default)
    args = parser.parse_args()
    return args

def prepare_data_module(cfg):
    print("### prepare data module ###")
    # if cfg.data_path.endswith("vctk_train") or cfg.data_path.endswith("vctk"):
    if cfg.dataset == "vctk":
        print("[in prepare data module] using vctk dataset!", flush=True)
        train_set = ABDataset_vctk("/home/jiahe.wang/workspace/urgent26/local/virtual_team_vctk_train", 'tr', fs=cfg.fs, gap_thres=cfg.score_diff_thres)
        val_set = ABDataset_vctk("/home/jiahe.wang/workspace/urgent26/local/virtual_team_vctk_train", 'cv', fs=cfg.fs, gap_thres=cfg.score_diff_thres)
    elif cfg.dataset == "urgent" or cfg.dataset == "urgent24":
        print("[in prepare data module] using urgent24 dataset!", flush=True)
        train_set = ABDataset_urgent24("/home/wangyou.zhang/urgent2024_challenge/submissions/", 'tr', fs=cfg.fs, gap_thres=cfg.score_diff_thres)
        val_set = ABDataset_urgent24("/home/wangyou.zhang/urgent2024_challenge/submissions/", 'cv', fs=cfg.fs, gap_thres=cfg.score_diff_thres)
    elif cfg.dataset == "urgent25":
        print("[in prepare data module] using urgent25 dataset!", flush=True)
        train_set = ABDataset_urgent25("/home/chenda.li/workspace/urgent26/urgent25_submissions", 'tr', fs=cfg.fs, gap_thres=cfg.score_diff_thres)
        val_set = ABDataset_urgent25("/home/chenda.li/workspace/urgent26/urgent25_submissions", 'cv', fs=cfg.fs, gap_thres=cfg.score_diff_thres)
    elif cfg.dataset == "urgent25_int":
        print("[in prepare data module] using urgent25 integrate dataset!", flush=True)
        train_set = ABDataset_urgent25_integrate("/home/chenda.li/workspace/urgent26/urgent25_submissions", 'tr', fs=cfg.fs, gap_thres=cfg.score_diff_thres)
        val_set = ABDataset_urgent25_integrate("/home/chenda.li/workspace/urgent26/urgent25_submissions", 'cv', fs=cfg.fs, gap_thres=cfg.score_diff_thres)
    else:
        print("[in prepare data module] dataset unknown! using default urgent24!", flush=True)
        train_set = ABDataset_urgent24("/home/wangyou.zhang/urgent2024_challenge/submissions/", 'tr', fs=cfg.fs, gap_thres=cfg.score_diff_thres)
        val_set = ABDataset_urgent24("/home/wangyou.zhang/urgent2024_challenge/submissions/", 'cv', fs=cfg.fs, gap_thres=cfg.score_diff_thres)

    print("train set len:", len(train_set), flush=True)
    print("val set len:", len(val_set), flush=True)
    data_module = MyDataModule(train_set=train_set, val_set=val_set, cfg=cfg)
    return data_module


def prepare_call_backs(cfg):

    best_metrics = [
        ('val_acc', 'max'),
        ('val_mse_loss', 'min'),
        ('val_loss', 'min'),]
    call_backs = []
    for i, (metric, min_or_max) in enumerate(best_metrics):
        call_back = ModelCheckpoint(
            filename="best_{epoch:02d}-{step:06d}-{"+ metric + ":.3f}",
            save_top_k=cfg.save_top_k,
            monitor=metric,
            every_n_train_steps=cfg.val_check_interval,
            mode=min_or_max,
            save_weights_only=(metric != "val_loss"),
            save_last=(metric == "val_loss"),
        )
        call_backs.append(call_back)


    return call_backs

if __name__ == "__main__":
    # mp.set_start_method('spawn')
    torch.set_float32_matmul_precision('medium')
    args = config_parser()
    cfg = Config(**vars(args))
    print("seed:", cfg.seed, flush=True)
    L.seed_everything(seed=cfg.seed)

    model = AbTestModel(cfg=cfg)
    if os.path.exists(cfg.init_ckpt):
        # model = AbTestModel.load_from_checkpoint(cfg.init_ckpt)
        ckpt = torch.load(cfg.init_ckpt)
        model.load_state_dict(ckpt["state_dict"])
        print("init from:", cfg.init_ckpt)

    print(model, flush=True)

    logger = TensorBoardLogger(save_dir=f"./exp/{cfg.train_tag}", version=cfg.train_version, name=cfg.train_name)
    data_module = prepare_data_module(cfg)
    call_backs = prepare_call_backs(cfg=cfg)
    
    last_ckpt = f"./exp/{cfg.train_tag}/{cfg.train_name}/version_{cfg.train_version}/checkpoints/last.ckpt"
    last_ckpt = last_ckpt if cfg.resume and os.path.exists(last_ckpt) else None

    trainer = L.Trainer(
        max_epochs=cfg.num_train_epochs,
        accelerator=cfg.device,
        gradient_clip_val=cfg.gradient_clip,
        logger=logger,
        val_check_interval=cfg.val_check_interval,
        callbacks=call_backs,
    )
    trainer.fit(model=model, datamodule=data_module, ckpt_path=last_ckpt,)
