from urgentpk.PKDataset import PKDataset
from urgentpk.urgentpk_model import AbTestModel
import itertools
import torchaudio
import torch
from scipy.stats import kendalltau, spearmanr
import tqdm
import os
import fire
import random
from argparse import ArgumentParser

device = torch.device("cuda")

def pearson_correlation(list1, list2):
    mean1 = sum(list1) / len(list1)
    mean2 = sum(list2) / len(list2)
    numerator = sum((x - mean1) * (y - mean2) for x, y in zip(list1, list2))
    denominator = (sum((x - mean1)**2 for x in list1) * sum((y - mean2)**2 for y in list2)) ** 0.5
    return numerator / denominator

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

def get_teams_mos(teams_mos):
    rtv = {}
    for team, all_mos in teams_mos.items():
        mos = sum([ float(all_mos[k]) for k in all_mos]) / len(all_mos)
        rtv[team] = mos
    return rtv

def get_ab_rank(teams_wav,
                ab_model,
                rank_func=pk_ab_model,
                vote_mode="logit_binary_vote",
                noisy_team=None,):

    teams = list(teams_wav.keys())
    team_score = {k:0 for k in teams}
    sample_ids = teams_wav[teams[0]].keys()
    
    if not vote_mode.startswith("get_mos"):
        all_perm = itertools.combinations(teams, 2)
        for a, b in tqdm.tqdm(list(all_perm)):
        # for a, b in list(all_perm):
            for sample in sample_ids:
                wav_a = teams_wav[a][sample]
                wav_b = teams_wav[b][sample]
                rtv, logits, a_score, b_score = rank_func(ab_model, wav_a, wav_b)
                # print("rtv and logits:", rtv, logits, flush=True)
                if vote_mode == "logit_binary_vote":
                    if rtv:
                        team_score[a] += 1
                    else:
                        team_score[b] += 1
                elif vote_mode == "logit_non_binary_vote":
                    team_score[a] += logits
                    team_score[b] += (1.0 - logits)
                elif vote_mode == "p_score":
                    team_score[a] += a_score
                    team_score[b] += b_score
                elif vote_mode == "p_score_compare":
                    if a_score > b_score:
                        team_score[a] += 1
                    else:
                        team_score[b] += 1
                else:
                    print("vote mode unknown, using default logit_binary_vote!")
                    if rtv:
                        team_score[a] += 1
                    else:
                        team_score[b] += 1
    else:
        for team in teams:
            for sample in sample_ids:
                wav = teams_wav[team][sample]
                wav_noisy = teams_wav[noisy_team][sample]
                if vote_mode == "get_mos_dup":
                    rtv, logits, a_score, b_score = rank_func(ab_model, wav, wav)
                    team_score[team] += (a_score + b_score) / 2
                elif vote_mode == "get_mos_noise":
                    rtv, logits, a_score, _ = rank_func(ab_model, wav, wav_noisy)
                    rtv, logits, _, b_score = rank_func(ab_model, wav_noisy, wav)
                    team_score[team] += (a_score + b_score) / 2
            team_score[team] /= len(sample_ids)

    return team_score

def get_rank(ckpt="none",
             subset="test",
             dataset=None,
             vote_mode="logit_binary_vote",
             noisy_team=None,
             ):
    
    # print("### get rank inf ###", flush=True)
    # print("checkpoint:", ckpt, flush=True)
    # print("dataset:", dataset, flush=True)
    # print("subset:", subset, flush=True)
    # print("vote_mode:", vote_mode, flush=True)

    assert vote_mode in ["logit_binary_vote",
                         "logit_non_binary_vote",
                         "get_mos_dup",
                         "get_mos_noise",], "vote mode should be in [logit_binary_vote, logit_non_binary_vote, get_mos_dup, get_mos_noise]!"
    assert os.path.exists(ckpt), "checkpoint does not exists!"
    assert os.path.exists(dataset), "dataset does not exists!"

    ab_model = AbTestModel.load_from_checkpoint(ckpt)
    ab_model = ab_model.to(device)
    ab_model.eval()

    if subset == "test":
        teams_wav, teams_mos = PKDataset(root_path=dataset, fs=16000).load_wavs_tt()
    else:
        teams_wav, teams_mos = PKDataset(root_path=dataset, fs=16000).load_wavs_cv()

    mos = get_teams_mos(teams_mos)
    teams = list(teams_mos.keys())
    if vote_mode == "get_mos_noise":
        assert noisy_team in teams, "the virtual noisy team does not exists!"

    assert len(teams_wav) == len(teams_mos), "number of teams error!"
    assert len(teams_wav[teams[0]]) == len(teams_mos[teams[0]]), "number of utterances error!"

    ab_rank = get_ab_rank(teams_wav, ab_model, vote_mode=vote_mode, noisy_team=noisy_team)

    krcc, p_value = kendalltau([mos[k] for k in teams], [ab_rank[k] for k in teams])
    srcc, p_value = spearmanr([mos[k] for k in teams], [ab_rank[k] for k in teams])
    lcc = pearson_correlation([mos[k] for k in teams], [ab_rank[k] for k in teams])
    
    print("krcc:", krcc, "srcc:", srcc, "lcc:", lcc, flush=True)
    return krcc, srcc, lcc

def main():
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="path to the checkpoint")
    parser.add_argument("--subset", type=str, default="test", help="valid or test")
    parser.add_argument("--dataset", type=str, default="unknown", help="dataset directory")
    parser.add_argument("--score_mode", type=str, default="logit_binary_vote", help="logit_binary_vote / logit_non_binary_vote / get_mos_dup / get_mos_noise")
    parser.add_argument("--noisy_team", type=str, default="unknown", help="the virtual team that consists of noisy speeches, if you need to do the ablation study of predicted MOS")
    args = parser.parse_args()
    krcc, srcc, lcc = get_rank(args.ckpt_path, args.subset, args.dataset, args.score_mode, args.noisy_team)

if __name__ == "__main__":
    main()
