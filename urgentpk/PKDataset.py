import torch
import glob
import itertools
import soundfile as sf
import torchaudio
import lightning
import os
import pandas as pd
import numpy as np
import librosa
import random
import json
from torch.nn.utils.rnn import pad_sequence

def check_two_lists(list1, list2):
    cnt = 0
    for i in list1:
        for j in list2:
            if i == j:
                cnt += 1
    return cnt

def read_scp(scp_file):
    rtv = {}
    with open(scp_file) as f:
        for line in f:
            k, v = line.strip().split()
            rtv[k] = v
    return rtv

def check_keys(key_list, dic):
    state = True
    for key in key_list:
        if not key in dic:
            print(key)
            state = False
    return state

def get_len_from_dic(dic):
    tot = 0
    for key in dic:
        path = dic[key]
        dua = librosa.get_duration(filename=path)
        tot = tot + dua
    return tot

def trim_teams_wavs_mos(teams_mos, teams_wav):
    teams = teams_mos.keys()
    utts = set()
    assert not utts, "utts is not empty!"
    
    for team in teams:
        team_mos = teams_mos[team]
        team_wav = teams_wav[team]
        utt_from_mos = set(team_mos.keys())
        utt_from_wav = set(team_wav.keys())
        team_utt = utt_from_mos.intersection(utt_from_wav)
        if not utts:
            utts = team_utt
        else:
            utts = utts.intersection(team_utt)
    
    for team in teams:
        team_mos = teams_mos[team]
        team_wav = teams_wav[team]
        teams_mos[team] = {utt: team_mos[utt] for utt in utts}
        teams_wav[team] = {utt: team_wav[utt] for utt in utts}

    return teams_mos, teams_wav

def check_teams_wavs_mos(teams_mos, teams_wav):
    assert len(teams_mos) == len(teams_wav), (len(teams_mos), len(teams_wav))
    assert set(teams_mos.keys()) == set(teams_wav.keys()), "teams ID error!"
    keys_len = -1
    keys_set = None
    for team in teams_mos:
        team_mos = teams_mos[team]
        team_wav = teams_wav[team]
        assert len(team_mos) == len(team_wav) and len(team_mos) > 0, (len(team_mos), len(team_wav))
        assert set(team_mos.keys()) == set(team_wav.keys()), "utt ID error!"
        if keys_len == -1:
            keys_len = len(team_mos)
            keys_set = set(team_mos.keys())
        else:
            assert keys_len == len(team_mos)
            assert keys_set == set(team_mos.keys())
    print("PKDataset checked!", "teams:", len(teams_mos), "utts:", keys_len)

'''
PKDataset be like:

dataset/
    split.json
        {
            tr: [...]
            cv: [...]
            tt: [...]
        }
    team_1/
        wav.scp
        mos.scp
    team_2/
        wav.scp
        mos.scp
    .
    .
    .
    team_k/
        wav.scp
        mos.scp
'''

class PKDataset(torch.utils.data.Dataset):
    def __init__(self, root_path=None, folder='tr', fs=16000, delta=0.0):
        super().__init__()

        self.root_path = root_path
        self.folder = folder
        self.fs = fs
        self.delta = delta
        if self.delta == 0.0:
            self.delta = -0.1
        
        self.split = None
        if os.path.exists(f'{self.root_path}/split.json'):
            with open(f'{self.root_path}/split.json', 'r') as f:
                self.split = json.load(f)
        assert self.split is not None and 'tr' in self.split, "'split.json' is required to load the dataset, see README.md for more details!"
        # print(self.split)

        teams_wav, teams_mos, self.wav_pairs, self.mos_pairs = self.load_wavs()
        check_teams_wavs_mos(teams_wav, teams_mos)

    def __len__(self):
        return len(self.wav_pairs)
    
    def __getitem__(self, index):
        wav_pair = self.wav_pairs[index]
        mos_pair = self.mos_pairs[index]

        audio = []
        for wav in wav_pair:
            wav, sr = torchaudio.load(wav)
            if sr != self.fs:
                wav = torchaudio.functional.resample(wav, sr, self.fs)
            audio.append(wav)

        audio = torch.cat(audio)
        mos_pair = torch.tensor(mos_pair)
        return audio, mos_pair

    def load_wavs(self,):
        team_scp = glob.glob(f'{self.root_path}/*/wav.scp')
        teams = [scp.split('/')[-2] for scp in team_scp]
        teams = list(set(teams))
        perms = list(itertools.permutations(teams, 2))

        teams_wav = {}
        teams_mos = {}

        for team in teams:
            wav_scp = f'{self.root_path}/{team}/wav.scp'
            teams_wav[team] = read_scp(wav_scp)
            mos_scp = f'{self.root_path}/{team}/mos.scp'
            teams_mos[team] = read_scp(mos_scp)
        
        teams_mos, teams_wav = trim_teams_wavs_mos(teams_mos, teams_wav)
        wav_pairs = []
        mos_pairs = []
        utt_keys = self.split[self.folder]
        
        for perm in perms:
            for key in utt_keys:
                mos_diff = abs(float(teams_mos[perm[0]][key]) - float(teams_mos[perm[1]][key]))
                if self.folder == 'tr' and mos_diff <= self.delta:
                    continue
                wav_pairs.append((teams_wav[perm[0]][key], teams_wav[perm[1]][key]))
                mos_pairs.append((float(teams_mos[perm[0]][key]), float(teams_mos[perm[1]][key])))

        return teams_wav, teams_mos, wav_pairs, mos_pairs

    def load_wavs_cv(self,):
        team_scp = glob.glob(f'{self.root_path}/*/wav.scp')
        teams = [scp.split('/')[-2] for scp  in team_scp]
        teams = list(set(teams))

        teams_wav = {}
        teams_mos = {}
        
        if 'cv' in self.split:
            keys = self.split['cv']
        else:
            keys = teams_mos[teams[0]].keys()
            keys = [key for key in keys if key not in self.split['tr']]

        for team in teams:
            wav_scp = f'{self.root_path}/{team}/wav.scp'
            teams_wav[team] = read_scp(wav_scp)
            teams_wav[team] = {k: teams_wav[team][k] for k in keys}
            mos_scp = f'{self.root_path}/{team}/mos.scp'
            teams_mos[team] = read_scp(mos_scp)
            teams_mos[team] = {k: teams_mos[team][k] for k in keys}
        
        teams_mos, teams_wav = trim_teams_wavs_mos(teams_mos, teams_wav)
        return teams_wav, teams_mos

    def load_wavs_tt(self,):
        team_scp = glob.glob(f'{self.root_path}/*/wav.scp')
        teams = [scp.split('/')[-2] for scp  in team_scp]
        teams = list(set(teams))

        teams_wav = {}
        teams_mos = {}

        if 'tt' in self.split:
            keys = self.split['tt']
        else:
            keys = teams_mos[teams[0]].keys()

        for team in teams:
            wav_scp = f'{self.root_path}/{team}/wav.scp'
            teams_wav[team] = read_scp(wav_scp)
            teams_wav[team] = {k: teams_wav[team][k] for k in keys}
            mos_scp = f'{self.root_path}/{team}/mos.scp'
            teams_mos[team] = read_scp(mos_scp)
            teams_mos[team] = {k: teams_mos[team][k] for k in keys}

        teams_mos, teams_wav = trim_teams_wavs_mos(teams_mos, teams_wav)
        return teams_wav, teams_mos

def padding_collect_fn(feature):
    audios = [x[0].T for x in feature]
    scores = torch.stack([x[1] for x in feature])
    audios = pad_sequence(audios, batch_first=True, padding_value=0)
    audios = audios.permute(0, 2, 1)
    return audios, scores

class MyDataModule(lightning.LightningDataModule):
    def __init__(self, train_set, val_set,  cfg, test_set=None):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.cfg = cfg
        
    def train_dataloader(self,):
        return torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            drop_last=True,
            shuffle=True,
            num_workers=self.cfg.num_worker,
            collate_fn=padding_collect_fn,
        )

    def val_dataloader(self,):
        return torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.cfg.batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=self.cfg.num_worker,
            collate_fn=padding_collect_fn,
            # persistent_workers=self.cfg.num_worker>0,    
        )

    def test_dataloader(self,):

        if self.test_set:
            return torch.utils.data.DataLoader(
            self.test_set,
            batch_size=1,
            drop_last=False,
            shuffle=False,
            num_workers=0,
            collate_fn=padding_collect_fn,
            # persistent_workers=self.cfg.num_worker>0,    
         )
        else:
            return None

if __name__ == "__main__":
    urgent25data = PKDataset(root_path="/home/jiahe.wang/workspace/urgentpk/urgentpk/local/PKDataset24", folder="tr", fs=16000, delta=0.3)
    teams_wav, teams_mos, wav_pairs, mos_pairs = urgent25data.load_wavs()
    check_teams_wavs_mos(teams_mos, teams_wav)
    teams_wav, teams_mos = urgent25data.load_wavs_cv()
    check_teams_wavs_mos(teams_mos, teams_wav)
    teams_wav, teams_mos = urgent25data.load_wavs_tt()
    check_teams_wavs_mos(teams_mos, teams_wav)