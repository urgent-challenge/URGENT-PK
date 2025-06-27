from prepare_dataset import MOSDataset, MOSDataset25, ABDataset_urgent24, ABDataset_urgent25, ABDataset_vctk, ABDataset_urgent25_integrate, ABDataset_chime
# from model import AbTestModel
# from model_mod import AbTestModel, AbTestModel_old
from urgentpk_model import AbTestModel
import itertools
import torchaudio
import torch
from scipy.stats import kendalltau, spearmanr
# from dnsmos import get_dns_mos
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

'''
def get_team_dnsmos(team_audios):
    rtv = {}
    for team, all_audios in team_audios.items():
        score = 0
        for k in all_audios:
            score += get_dns_mos(all_audios[k])
        rtv[team] = score / len(all_audios)
    return rtv
'''

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

'''
def pk_ab_dnsmos(wav1, wav2):
    dnsmos1 = get_dns_mos(wav1)
    dnsmos2 = get_dns_mos(wav2)
    return dnsmos1 > dnsmos2
'''

def get_team_mos(team_mos):
    rtv = {}
    for team, all_mos in team_mos.items():
        mos = sum([ float(all_mos[k]) for k in all_mos]) / len(all_mos)
        rtv[team] = mos
    return rtv

def scores_to_rank(teams_score):
    sorted_teams = sorted(teams_score.items(), key=lambda item: item[1], reverse=True)
    ab_rank_dic = {}
    for i, (team, score) in enumerate(sorted_teams):
        ab_rank_dic[team] = i + 1
    return ab_rank_dic

def get_official_rank(dataset):
    team_24_rank = ["526", "505", "524", "478", "481", "518", "520", "488", "489", "495", "455", "477", "523", "492", "474", "427", "422", "509", "511", "456", "508", "506", "426"]
    team_25_rank = ["795", "783", "791", "761", "806", "796", "729", "740", "776", "719", "809", "734", "799", "753", "757", "738", "804", "747", "794", "763", "774", "716"]
    if dataset.endswith("24"):
        rank_dic = {team: index + 1 for index, team in enumerate(team_24_rank)}
    elif dataset.endswith("25"):
        rank_dic = {team: index + 1 for index, team in enumerate(team_25_rank)}
    return rank_dic

def get_ab_rank(team_audios,
                ab_model,
                rank_func=pk_ab_model,
                vote_mode="logit_binary_vote",
                noisy_team=None,):

    teams = list(team_audios.keys())
    team_score = {k:0 for k in teams}
    sample_ids = team_audios[teams[0]].keys()
    
    print("### get team score start! ###", flush=True)
    if not vote_mode.startswith("get_mos"):
        all_perm = itertools.combinations(teams, 2)
        for a, b in tqdm.tqdm(list(all_perm)):
        # for a, b in list(all_perm):
            for sample in sample_ids:
                wav_a = team_audios[a][sample]
                wav_b = team_audios[b][sample]
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
                wav = team_audios[team][sample]
                wav_noisy = team_audios[noisy_team][sample]
                if vote_mode == "get_mos_dup":
                    rtv, logits, a_score, b_score = rank_func(ab_model, wav, wav)
                    team_score[team] += (a_score + b_score) / 2
                elif vote_mode == "get_mos_noise":
                    rtv, logits, a_score, _ = rank_func(ab_model, wav, wav_noisy)
                    rtv, logits, _, b_score = rank_func(ab_model, wav_noisy, wav)
                    team_score[team] += (a_score + b_score) / 2
            team_score[team] /= len(sample_ids)

    print("### get team score finish! ###", flush=True)
    return team_score

def get_rank(ckpt="none",
             mode="test",
             dataset="urgent25",
             vote_mode="logit_binary_vote",
             compare_with="MOS",
             ):
    
    print("### get rank inf ###", flush=True)
    print("checkpoint:", ckpt, flush=True)
    print("dataset:", dataset, flush=True)
    print("subset:", mode, flush=True)
    print("vote_mode:", vote_mode, flush=True)
    print("compare_with:", compare_with, flush=True)

    assert vote_mode in ["logit_binary_vote",
                         "logit_non_binary_vote",
                         "p_score",
                         "p_score_compare",
                         "get_mos_dup",
                         "get_mos_noise",], "vote mode should be in [logit_binary_vote, logit_non_binary_vote, p_score, p_score_compare, get_mos_dup, get_mos_noise]!"
    assert compare_with in ["MOS", "rank"]
    assert os.path.exists(ckpt), "checkpoint does not exists!"

    ab_model = AbTestModel.load_from_checkpoint(ckpt)
    ab_model = ab_model.to(device)
    ab_model.eval()

    if mode == "test":
        if dataset == "urgent25" or dataset == "urgent25_en":
            team_audios, team_mos = MOSDataset25().load_wavs()
        elif dataset == "urgent24":
            team_audios, team_mos = ABDataset_urgent24("/home/wangyou.zhang/urgent2024_challenge/submissions/", 'tr', fs=16000).load_wavs_for_test()
        elif dataset == "urgent25_de":
            team_audios, team_mos = MOSDataset25().load_wavs_by_lang("de")
        elif dataset == "urgent25_jp":
            team_audios, team_mos = MOSDataset25().load_wavs_by_lang("jp")
        elif dataset == "urgent25_zh":
            team_audios, team_mos = MOSDataset25().load_wavs_by_lang("zh")
        elif dataset == "chime":
            team_audios, team_mos = ABDataset_chime().load_wavs_for_test()
        else:
            print("dataset unknown! using default urgent25 data for test!", flush=True)
            team_audios, team_mos = MOSDataset25().load_wavs()
    else:
        if dataset == "urgent25":
            team_audios, team_mos = ABDataset_urgent25("/home/chenda.li/workspace/urgent26/urgent25_submissions", 'tr', fs=16000).load_wavs_for_valid()
        elif dataset == "urgent24":
            team_audios, team_mos = ABDataset_urgent24("/home/wangyou.zhang/urgent2024_challenge/submissions/", 'tr', fs=16000).load_wavs_for_valid()
        elif dataset == "urgent25_int":
            team_audios, team_mos = ABDataset_urgent25_integrate("/home/chenda.li/workspace/urgent26/urgent25_submissions", 'tr', fs=16000).load_wavs_for_valid()
        else:
            print("dataset unknown! using default urgent25 data for valid!", flush=True)
            team_audios, team_mos = ABDataset_urgent25("/home/chenda.li/workspace/urgent26/urgent25_submissions", 'tr', fs=16000).load_wavs_for_valid()

    team_24_rank = ["526", "505", "524", "478", "481", "518", "520", "488", "489", "495", "455", "477", "523", "492", "474", "427", "422", "509", "511", "456", "508", "506", "426"]
    team_25_rank = ["795", "783", "791", "761", "806", "796", "729", "740", "776", "719", "809", "734", "799", "753", "757", "738", "804", "747", "794", "763", "774", "716"]
    noisy_team = None
    if dataset.startswith("urgent24"):
        noisy_team = "426"
    elif dataset.startswith("urgent25"):
        noisy_team = "716"

    mos = get_team_mos(team_mos)
    teams = list(team_mos.keys())

    assert len(team_audios) == len(team_mos), "number of teams error!"
    assert len(team_audios[teams[0]]) == len(team_mos[teams[0]]), "number of utterances error!"

    ab_rank = get_ab_rank(team_audios, ab_model, vote_mode=vote_mode, noisy_team=noisy_team)

    if compare_with == "rank":
        # predicted rank compare with oracle rank
        if dataset.startswith("urgent24"):
            rank_dic = {team: index + 1 for index, team in enumerate(team_24_rank)}
        elif dataset.startswith("urgent25"):
            rank_dic = {team: index + 1 for index, team in enumerate(team_25_rank)}
        
        sorted_teams = sorted(ab_rank.items(), key=lambda item: item[1], reverse=True)
        ab_rank_dic = {}
        for i, (team, score) in enumerate(sorted_teams):
            ab_rank_dic[team] = i + 1

        krcc, p_value = kendalltau([rank_dic[k] for k in teams], [ab_rank_dic[k] for k in teams])
        srcc, p_value = spearmanr([rank_dic[k] for k in teams], [ab_rank_dic[k] for k in teams])
        lcc = pearson_correlation([rank_dic[k] for k in teams], [ab_rank_dic[k] for k in teams])
    else:
        # predicted MOS compare with oracle MOS
        krcc, p_value = kendalltau([mos[k] for k in teams], [ab_rank[k] for k in teams])
        srcc, p_value = spearmanr([mos[k] for k in teams], [ab_rank[k] for k in teams])
        lcc = pearson_correlation([mos[k] for k in teams], [ab_rank[k] for k in teams])
    
    sum = krcc + srcc + lcc

    '''
    try:
        ckpt_name = os.path.basename(ckpt)
        basename = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(ckpt))))
        config = os.path.basename(basename)
    except:
        ckpt_name = ckpt
        config = "unknown"
        print("fail to get config!", flush=True)
    '''

    # print("krcc:", krcc, "srcc:", srcc, "lcc:", lcc, "sum:", sum, flush=True)
    print("krcc:", krcc, "srcc:", srcc, "lcc:", lcc, flush=True)
    
    '''
    if mode == "test":
        save_file = "test_results.txt"
    else:
        save_file = "valid_results.txt"

    vote_mode = vote_mode + "_" + compare_with
        
    with open(save_file, "a") as f:
            f.write(mode + " " + config + " " + ckpt_name + " " + dataset + " " + str(vote_mode) + " " + str(round(krcc, 3)) + " " + str(round(srcc, 3)) + " " + str(round(lcc, 3)) + " " + str(round(sum, 3)) + "\n")
    exp_folder = os.path.dirname(os.path.dirname(ckpt))
    with open(os.path.join(exp_folder, save_file), "a") as f:
        f.write(mode + " " + config + " " + ckpt_name + " " + dataset + " " + str(vote_mode) + " " + str(round(krcc, 3)) + " " + str(round(srcc, 3)) + " " + str(round(lcc, 3)) + " " + str(round(sum, 3)) + "\n")
    '''

    return krcc, srcc, lcc

if __name__ == "__main__":
    # fire.Fire(get_rank)
    parser = ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, help="path to the checkpoint")
    parser.add_argument("--subset", type=str, default="test", help="valid or test")
    parser.add_argument("--dataset", type=str, default="urgent25", help="on which dataset to perform inference")
    parser.add_argument("--score_mode", type=str, default="logit_binary_vote", help="logit_binary_vote, logit_non_binary_vote, p_score, p_score_compare, get_mos_dup, get_mos_noise")
    args = parser.parse_args()
    krcc, srcc, lcc = get_rank(args.ckpt_path, args.subset, args.dataset, args.score_mode)
