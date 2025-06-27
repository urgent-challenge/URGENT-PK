from prepare_dataset import MOSDataset, MOSDataset25, ABDataset_urgent24, ABDataset_urgent25, ABDataset_vctk, ABDataset_urgent25_integrate, ABDataset_chime
# from model import AbTestModel
from model_mod import AbTestModel, AbTestModel_old
import itertools
import torchaudio
import torch
from scipy.stats import kendalltau, spearmanr
# from dnsmos import get_dns_mos
from tqdm import tqdm
import os
import fire

device = torch.device("cuda")

def read_wav(wav):
    wav, sr = torchaudio.load(wav)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)    

    return wav 

def pk_ab_model(ab_model, wav1, wav2):

    wav1 = read_wav(wav1)
    wav2 = read_wav(wav2)

    audios = torch.stack([wav1, wav2], dim=1)
    predicted, p_scores = ab_model.model(audios.to(device))
    predicted = predicted.view(-1)
    logits = torch.sigmoid(predicted)
    rtv = int(logits > 0.5)
    return rtv, logits.item(), p_scores[0][0].item(), p_scores[0][1].item()

def get_abtest(ckpt="none"):
    ckpt_list = ["/home/jiahe.wang/workspace/urgent26/abtest_model/exp/FNL_mel_resnet34_urgent24_1e-4_0.25/ab_test/version_0/checkpoints/best_epoch=11-step=092000-val_mse_loss=0.204.ckpt",
                 "/home/jiahe.wang/workspace/urgent26/abtest_model/exp/FNL_utmos_fix_resnet34_urgent24_1e-5_0.25/ab_test/version_0/checkpoints/best_epoch=06-step=168000-val_mse_loss=0.313.ckpt",
                 "/home/jiahe.wang/workspace/urgent26/abtest_model/exp/FNL_utmos_ft_resnet34_urgent24_1e-5_0.25/ab_test/version_0/checkpoints/best_epoch=00-step=024000-val_acc=0.714.ckpt"]
    # ckpt = "/home/jiahe.wang/workspace/urgent26/abtest_model/exp/FNL_mel_resnet34_urgent24_1e-4_0.25/ab_test/version_0/checkpoints/best_epoch=11-step=092000-val_mse_loss=0.204.ckpt"
    # ckpt = "/home/jiahe.wang/workspace/urgent26/abtest_model/exp/FNL_utmos_fix_resnet34_urgent24_1e-5_0.25/ab_test/version_0/checkpoints/best_epoch=06-step=168000-val_mse_loss=0.313.ckpt"
    # ckpt = "/home/jiahe.wang/workspace/urgent26/abtest_model/exp/FNL_utmos_ft_resnet34_urgent24_1e-5_0.25/ab_test/version_0/checkpoints/best_epoch=00-step=024000-val_acc=0.714.ckpt"
    for ckpt in ckpt_list:
        print("checkpoint:", ckpt, flush=True)

        assert os.path.exists(ckpt), "checkpoint does not exists!"
        ab_model = AbTestModel.load_from_checkpoint(ckpt)
        ab_model = ab_model.to(device)
        ab_model.eval()

        abtest_dir = "/home/jiahe.wang/workspace/urgent26/local/real_abtest_20"
        group_num = 6
        pair_num = 20

        for group in range(group_num):
            group_name = "group" + str(group)
            group_dir = os.path.join(abtest_dir, group_name)
            log_path = os.path.join(group_dir, "log.txt")
            oracle_comp = {}
            with open(log_path, "r") as f:
                for line in f:
                    line = line.strip().split()
                    # print(line)
                    oracle_comp[line[0]] = line[1]
                # print(oracle_comp)
            cnt = 0
            print("group:", group)
            for pair in range(1, pair_num + 1):
                wav1_path = os.path.join(group_dir, "pair" + str(pair) + "_0.wav")
                wav2_path = os.path.join(group_dir, "pair" + str(pair) + "_1.wav")
                rtv, logit, score1, score2 = pk_ab_model(ab_model, wav1_path, wav2_path)
                if int(rtv) + int(oracle_comp["pair" + str(pair)]) == 1:
                    cnt += 1
            print("acc:", round(cnt / pair_num, 3))
        print("")
        del ab_model

if __name__ == "__main__":
    fire.Fire(get_abtest)
