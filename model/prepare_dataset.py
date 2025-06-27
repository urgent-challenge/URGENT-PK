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
from torch.nn.utils.rnn import pad_sequence

# /home/wangyou.zhang/urgent2024_challenge/submissions
# /home/chenda/workspace/urgent26/urgent24/submissions
# /home/chenda.li/workspace/urgent26/urgent25_submissions

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
    print("checked!")
    print("team number:", len(teams_mos))
    print("utt number:", keys_len)

class ABDataset(torch.utils.data.Dataset):
    def __init__(self, root_path=None, folder="tr", fs=16000, gap_thres=0.0):
        super().__init__()

        self.root_path = root_path
        self.folder = folder
        self.fs = fs
        self.gap_thres = gap_thres
        if self.gap_thres == 0.0:
            self.gap_thres = -0.1
            
        self.wav_pairs, self.mos_pairs = self.load_wavs()

    def load_wavs(self,): # if used as training set, complete this
        wav_pairs = None
        mos_pairs = None
        return wav_pairs, mos_pairs # [(speech1, speech2)], [(MOS1, MOS2)]

    def load_wavs_for_valid(self,): # if used as training set, complete this
        teams_wav = None
        teams_mos = None
        return teams_wav, teams_mos # {teamID: {uttID: speech}}, {teamID: {uttID: MOS}}

    def load_wavs_for_test(self,): # if used as testing set, complete this
        teams_wav = None
        teams_mos = None
        return teams_wav, teams_mos # {teamID: {uttID: speech}}, {teamID: {uttID: MOS}}

class ABDataset_urgent24(torch.utils.data.Dataset):
    def __init__(self, root_path="/home/wangyou.zhang/urgent2024_challenge/submissions/", folder='tr', fs=16000, gap_thres=0.0):
        super().__init__()

        self.root_path = root_path
        self.cv_keys = ['fileid_224','fileid_793','fileid_438','fileid_810', 'fileid_24', 'fileid_985', 'fileid_734', 'fileid_11']
        self.tt_keys = []

        self.folder = folder
        self.fs = fs
        self.gap_thres = gap_thres
        if self.gap_thres == 0.0:
            self.gap_thres = -0.1

        self.wav_pairs, self.mos_pairs = self.load_wavs()
        assert len(self.wav_pairs) == len(self.mos_pairs)

        print("### ABDataset_urgent24 init ###")
        print("root path:", self.root_path)
        print("score diff threshold:", self.gap_thres)

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
        teams = [scp.split('/')[-2] for scp  in team_scp]
        perms = list(itertools.permutations(teams, 2))

        teams_wav = {}
        teams_mos = {}

        for team in teams:
            wav_scp = f'{self.root_path}/{team}/wav.scp'
            teams_wav[team] = read_scp(wav_scp)
            mos_scp = f'{self.root_path}/{team}/score/mos/MOS.scp'
            teams_mos[team] = read_scp(mos_scp)

        wav_pairs = []
        mos_pairs = []

        utt_keys = read_scp(mos_scp).keys()
        utt_keys = self.cv_keys if self.folder == 'cv' else utt_keys
        utt_keys = self.tt_keys if self.folder == 'tt' else utt_keys
        
        for perm in perms:
            for key in utt_keys:
                if self.folder == 'tr' and (key in self.cv_keys or key in self.tt_keys):
                    continue
                mos_gap = abs(float(teams_mos[perm[0]][key]) - float(teams_mos[perm[1]][key]))
                if self.folder == 'tr' and mos_gap <= self.gap_thres:
                    continue
                wav_pairs.append((
                    teams_wav[perm[0]][key],
                    teams_wav[perm[1]][key],
                )
                )
                mos_pairs.append((
                    float(teams_mos[perm[0]][key]),
                    float(teams_mos[perm[1]][key]),
                ))

        return wav_pairs, mos_pairs
    
    def load_wavs_for_test(self,):
        team_scp = glob.glob(f'{self.root_path}/*/wav.scp')
        teams = [scp.split('/')[-2] for scp  in team_scp]

        teams_wav = {}
        teams_mos = {}

        for team in teams:
            wav_scp = f'{self.root_path}/{team}/wav.scp'
            teams_wav[team] = read_scp(wav_scp)
            mos_scp = f'{self.root_path}/{team}/score/mos/MOS.scp'
            teams_mos[team] = read_scp(mos_scp)

        teams = list(teams_wav.keys())
        keys = teams_mos[team].keys()

        for team in teams:
            teams_wav[team] = { k: teams_wav[team][k] for k in keys}
            teams_mos[team] = { k: teams_mos[team][k] for k in keys}

        return teams_wav, teams_mos
    
    def load_wavs_for_valid(self,):
        team_scp = glob.glob(f'{self.root_path}/*/wav.scp')
        teams = [scp.split('/')[-2] for scp  in team_scp]

        teams_wav = {}
        teams_mos = {}

        for team in teams:
            wav_scp = f'{self.root_path}/{team}/wav.scp'
            teams_wav[team] = read_scp(wav_scp)
            mos_scp = f'{self.root_path}/{team}/score/mos/MOS.scp'
            teams_mos[team] = read_scp(mos_scp)

        teams = list(teams_wav.keys())
        keys = self.cv_keys

        for team in teams:
            teams_wav[team] = { k: teams_wav[team][k] for k in keys}
            teams_mos[team] = { k: teams_mos[team][k] for k in keys}

        return teams_wav, teams_mos

class ABDataset_urgent25(torch.utils.data.Dataset):
    def __init__(self, root_path="/home/chenda.li/workspace/urgent26/urgent25_submissions", folder='tr', fs=16000, gap_thres=0.0):
        super().__init__()

        self.root_path = root_path
        self.cv_keys = ['fileid_105','fileid_205','fileid_350','fileid_417', 'fileid_106', 'fileid_229', 'fileid_351', 'fileid_418']
        self.tt_keys = []

        self.folder = folder
        self.fs = fs
        self.gap_thres = gap_thres
        if self.gap_thres == 0.0:
            self.gap_thres = -0.1

        self.wav_pairs, self.mos_pairs = self.load_wavs()
        assert len(self.wav_pairs) == len(self.mos_pairs)

        print("### ABDataset_urgent25 init ###")
        print("root path:", self.root_path)
        print("score diff threshold:", self.gap_thres)

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
        team_scp = glob.glob(f'{self.root_path}/mos/MOS8_*')
        teams = [scp.split('/')[-1].split('_')[-1].split('.')[0] for scp  in team_scp]

        teams_wav = {}
        teams_mos = {}

        for team in teams:
            wav_scp = f'{self.root_path}/wavs/{team}.scp'
            mos_scp = f'{self.root_path}/mos/MOS8_{team}.scp'

            if not (os.path.exists(wav_scp) and os.path.exists(mos_scp)):
                continue

            teams_wav[team] = read_scp(wav_scp)
            teams_mos[team] = read_scp(mos_scp)

        teams = list(teams_wav.keys())
        keys = teams_mos[team].keys()

        for team in teams:
            teams_wav[team] = { k: teams_wav[team][k] for k in keys}
            teams_mos[team] = { k: teams_mos[team][k] for k in keys}
        perms = list(itertools.permutations(teams, 2))

        wav_pairs = []
        mos_pairs = []

        utt_keys = read_scp(mos_scp).keys()
        utt_keys = self.cv_keys if self.folder == 'cv' else utt_keys
        utt_keys = self.tt_keys if self.folder == 'tt' else utt_keys

        for perm in perms:
            for key in utt_keys:
                if self.folder == 'tr' and (key in self.cv_keys or key in self.tt_keys):
                    continue
                mos_gap = abs(float(teams_mos[perm[0]][key]) - float(teams_mos[perm[1]][key]))
                if self.folder == 'tr' and mos_gap <= self.gap_thres:
                    continue
                wav_pairs.append((
                    teams_wav[perm[0]][key],
                    teams_wav[perm[1]][key],
                )
                )
                mos_pairs.append((
                    float(teams_mos[perm[0]][key]),
                    float(teams_mos[perm[1]][key]),
                ))

        return wav_pairs, mos_pairs
    
    def load_wavs_for_valid(self,):
        team_scp = glob.glob(f'{self.root_path}/mos/MOS8_*')
        teams = [scp.split('/')[-1].split('_')[-1].split('.')[0] for scp  in team_scp]

        teams_wav = {}
        teams_mos = {}

        for team in teams:
            wav_scp = f'{self.root_path}/wavs/{team}.scp'
            mos_scp = f'{self.root_path}/mos/MOS8_{team}.scp'

            if not (os.path.exists(wav_scp) and os.path.exists(mos_scp)):
                continue

            teams_wav[team] = read_scp(wav_scp)
            teams_mos[team] = read_scp(mos_scp)

        teams = list(teams_wav.keys())
        keys = self.cv_keys

        for team in teams:
            teams_wav[team] = { k: teams_wav[team][k] for k in keys}
            teams_mos[team] = { k: teams_mos[team][k] for k in keys}

        return teams_wav, teams_mos

class ABDataset_urgent25_integrate(torch.utils.data.Dataset):
    def __init__(self, root_path="/home/chenda.li/workspace/urgent26/urgent25_submissions", folder='tr', fs=16000, gap_thres=0.0):
        super().__init__()

        self.root_path = root_path
        self.cv_keys = ['fileid_105','fileid_205','fileid_350','fileid_417', 'fileid_106', 'fileid_229', 'fileid_351', 'fileid_418',
                        'fileid_169','fileid_170','fileid_171','fileid_172','fileid_173','fileid_174','fileid_175','fileid_176',
                        'fileid_10','fileid_11','fileid_12','fileid_13','fileid_14','fileid_15','fileid_16','fileid_17',
                        'fileid_451','fileid_452','fileid_453','fileid_454','fileid_455','fileid_456','fileid_457','fileid_458']
        self.tt_keys = []

        self.folder = folder
        self.fs = fs
        self.gap_thres = gap_thres
        if self.gap_thres == 0.0:
            self.gap_thres = -0.1

        self.wav_pairs, self.mos_pairs = self.load_wavs()
        assert len(self.wav_pairs) == len(self.mos_pairs)

        print("### ABDataset_urgent25_integrate init ###")
        print("root path:", self.root_path)
        print("score diff threshold:", self.gap_thres)

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
        team_scp = glob.glob(f'{self.root_path}/mos/MOS8_*')
        teams = [scp.split('/')[-1].split('_')[-1].split('.')[0] for scp  in team_scp]

        teams_wav = {}
        teams_mos = {}

        csv_path = "/home/jiahe.wang/workspace/urgent26/local/URGENT2025_MOS_8Votes_EN_DE_ZH_JP.csv"
        df = pd.read_csv(csv_path)

        for team in teams:
            wav_scp = f'{self.root_path}/wavs/{team}.scp'
            if not os.path.exists(wav_scp):
                continue
            teams_wav[team] = read_scp(wav_scp)

        teams = list(teams_wav.keys())

        utt_keys = []
        for team in teams:
            teams_mos[team] = {}
        
        for i, line in df.iterrows():
            if str(line["Team ID"]) in teams:
                utt_keys.append(line["File ID"])
                teams_mos[str(line["Team ID"])][line["File ID"]] = float(line["MOS"])

        utt_keys = np.sort(list(set(utt_keys)))
        keys = teams_mos[team].keys()

        for team in teams:
            teams_wav[team] = {k: teams_wav[team][k] for k in keys}
            teams_mos[team] = {k: teams_mos[team][k] for k in keys}
            assert len(teams_wav[team]) == len(teams_mos[team])
            assert set(teams_wav[team].keys()) == set(teams_mos[team].keys())
        assert len(teams_wav) == len(teams_mos), (len(teams_wav), len(teams_mos))

        perms = list(itertools.permutations(teams, 2))

        wav_pairs = []
        mos_pairs = []

        utt_keys = self.cv_keys if self.folder == 'cv' else utt_keys
        utt_keys = self.tt_keys if self.folder == 'tt' else utt_keys
    
        for perm in perms:
            for key in utt_keys:
                if self.folder == 'tr' and (key in self.cv_keys or key in self.tt_keys):
                    continue
                mos_gap = abs(float(teams_mos[perm[0]][key]) - float(teams_mos[perm[1]][key]))
                if self.folder == 'tr' and mos_gap <= self.gap_thres:
                    continue
                wav_pairs.append((
                    teams_wav[perm[0]][key],
                    teams_wav[perm[1]][key],
                )
                )
                mos_pairs.append((
                    float(teams_mos[perm[0]][key]),
                    float(teams_mos[perm[1]][key]),
                ))

        return wav_pairs, mos_pairs
    
    def load_wavs_for_valid(self,):
        team_scp = glob.glob(f'{self.root_path}/mos/MOS8_*')
        teams = [scp.split('/')[-1].split('_')[-1].split('.')[0] for scp  in team_scp]

        teams_wav = {}
        teams_mos = {}

        csv_path = "/home/jiahe.wang/workspace/urgent26/local/URGENT2025_MOS_8Votes_EN_DE_ZH_JP.csv"
        df = pd.read_csv(csv_path)

        for team in teams:
            wav_scp = f'{self.root_path}/wavs/{team}.scp'
            if not os.path.exists(wav_scp):
                continue
            teams_wav[team] = read_scp(wav_scp)

        teams = list(teams_wav.keys())

        utt_keys = []
        for team in teams:
            teams_mos[team] = {}
        
        for i, line in df.iterrows():
            if str(line["Team ID"]) in teams:
                utt_keys.append(line["File ID"])
                teams_mos[str(line["Team ID"])][line["File ID"]] = float(line["MOS"])

        utt_keys = np.sort(list(set(utt_keys)))

        keys = self.cv_keys

        for team in teams:
            teams_wav[team] = {k: teams_wav[team][k] for k in keys}
            teams_mos[team] = {k: teams_mos[team][k] for k in keys}
            assert len(teams_wav[team]) == len(teams_mos[team])
            assert set(teams_wav[team].keys()) == set(teams_mos[team].keys())
        assert len(teams_wav) == len(teams_mos)

        return teams_wav, teams_mos

class ABDataset_chime(torch.utils.data.Dataset):
    def __init__(self, root_path="/home/jiahe.wang/workspace/urgent26/local/listening_test", fs=16000):
        super().__init__()
        self.root_path = root_path
        self.fs = fs
        self.csv_path = "/home/jiahe.wang/workspace/urgent26/local/listening_test/MOS_results_listening_test.csv"
        self.data_path = "/home/jiahe.wang/workspace/urgent26/local/listening_test/data"

    def __len__(self):
        return len(self.wav_pairs)
    
    def __getitem__(self, index):
        return None
    
    def load_wavs_for_test(self,):
        teams = ["C0", "C1", "C2", "C3", "C4"]

        df = pd.read_csv(self.csv_path)
        utt_list = []
        teams_mos = {}
        teams_wav = {}

        for team in teams:
            teams_mos[team] = {}
            teams_wav[team] = {}
        
        for i, line in df.iterrows():
            if str(line["scale"]) == "OVRL" and str(line["condition"]) in teams:
                tm = str(line["condition"])
                basename = str(line["sample"])
                utt = str(line["sample"]).split(".")[0]
                mos = float(line["MOS"])
                teams_mos[tm][utt] = mos
                teams_wav[tm][utt] = os.path.join(self.data_path, tm, basename)

        return teams_wav, teams_mos

class ABDataset_vctk(torch.utils.data.Dataset):
    def __init__(self, root_path='', folder='tr', fs=16000, gap_thres=0.0):
        super().__init__()

        self.root_path = root_path
        self.cv_keys = ['p226_001', 'p226_002', 'p226_003', 'p226_004', 'p227_001', 'p227_002', 'p227_003', 'p227_004', 'p228_001', 'p228_002', 'p228_003', 'p228_004', 'p226_005', 'p226_006', 'p227_005', 'p227_006', 'p228_005', 'p228_006']
        self.tt_keys = []

        self.folder = folder
        self.fs = fs
        self.gap_thres = gap_thres # only pairs with > 0.25 mos gap will be loaded for training
        if self.gap_thres == 0.0:
            self.gap_thres = -0.1

        self.wav_pairs, self.mos_pairs = self.load_wavs()
        assert len(self.wav_pairs) == len(self.mos_pairs)

        print("### ABDataset_vctk init ###")
        print("root path:", self.root_path)
        print("score diff threshold:", self.gap_thres)

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
        teams = [scp.split('/')[-2] for scp  in team_scp]
        perms = list(itertools.permutations(teams, 2))

        print("teams:", len(teams))

        teams_wav = {}
        teams_mos = {}

        for team in teams:
            wav_scp = f'{self.root_path}/{team}/wav.scp'
            teams_wav[team] = read_scp(wav_scp)
            mos_scp = f'{self.root_path}/{team}/score.scp'
            teams_mos[team] = read_scp(mos_scp)

        wav_pairs = []
        mos_pairs = []

        utt_keys = read_scp(mos_scp).keys()
        utt_keys = self.cv_keys if self.folder == 'cv' else utt_keys
        utt_keys = self.tt_keys if self.folder == 'tt' else utt_keys
        
        for perm in perms:
            for key in utt_keys:
                if self.folder == 'tr' and (key in self.cv_keys or key in self.tt_keys):
                    continue
                mos_gap = abs(float(teams_mos[perm[0]][key]) - float(teams_mos[perm[1]][key]))
                if self.folder == 'tr' and mos_gap <= self.gap_thres:
                    continue
                wav_pairs.append((
                    teams_wav[perm[0]][key],
                    teams_wav[perm[1]][key],
                )
                )
                mos_pairs.append((
                    float(teams_mos[perm[0]][key]),
                    float(teams_mos[perm[1]][key]),
                ))

        return wav_pairs, mos_pairs

class MOSDataset25(torch.utils.data.Dataset):
    def __init__(self, root_path='/home/chenda.li/workspace/urgent26/urgent25_submissions', fs=16000):
        super().__init__()

        self.root_path = root_path
        self.fs = fs
        self.csv_path = "/home/jiahe.wang/workspace/urgent26/local/URGENT2025_MOS_8Votes_EN_DE_ZH_JP.csv"

        self.wav_pairs, self.mos_pairs = self.load_wavs()
        assert len(self.wav_pairs) == len(self.mos_pairs)

    def __len__(self):
        return len(self.wav_pairs)
    
    def __getitem__(self, index):
        return None

    def load_wavs(self,):
        team_scp = glob.glob(f'{self.root_path}/mos/MOS8_*')
        teams = [scp.split('/')[-1].split('_')[-1].split('.')[0] for scp  in team_scp] 

        teams_wav = {}
        teams_mos = {}

        for team in teams:
            wav_scp = f'{self.root_path}/wavs/{team}.scp'
            mos_scp = f'{self.root_path}/mos/MOS8_{team}.scp'

            if not (os.path.exists(wav_scp) and os.path.exists(mos_scp)):
                continue

            teams_wav[team] = read_scp(wav_scp)
            teams_mos[team] = read_scp(mos_scp)

        teams = list(teams_wav.keys())

        keys = teams_mos[team].keys()

        for team in teams:
            teams_wav[team] = { k: teams_wav[team][k] for k in keys}
            teams_mos[team] = { k: teams_mos[team][k] for k in keys}

        return teams_wav, teams_mos
    
    def load_wavs_by_lang(self, lang="en"):

        assert lang in ["en", "zh", "jp", "de"], "language should be in [en, zh, jp, de]!"

        team_scp = glob.glob(f'{self.root_path}/mos/MOS8_*')
        teams = [scp.split('/')[-1].split('_')[-1].split('.')[0] for scp  in team_scp] 
        teams_wav = {}
        for team in teams:
            wav_scp = f'{self.root_path}/wavs/{team}.scp'
            if not os.path.exists(wav_scp):
                continue
            teams_wav[team] = read_scp(wav_scp)
        teams = list(teams_wav.keys())

        csv_path = "/home/jiahe.wang/workspace/urgent26/local/URGENT2025_MOS_8Votes_EN_DE_ZH_JP.csv"
        df = pd.read_csv(csv_path)
        utt_list = []
        teams_mos = {}
        for team in teams:
            teams_mos[team] = {}
        
        for i, line in df.iterrows():
            if line["lang"] == lang and str(line["Team ID"]) in teams:
                utt_list.append(line["File ID"])
                teams_mos[str(line["Team ID"])][line["File ID"]] = float(line["MOS"])

        utt_list = np.sort(list(set(utt_list)))
        
        for team in teams:
            teams_wav[team] = {k: teams_wav[team][k] for k in utt_list}
            teams_mos[team] = {k: teams_mos[team][k] for k in utt_list}
            assert len(teams_wav[team]) == len(teams_mos[team])
        assert len(teams_wav) == len(teams_mos)

        return teams_wav, teams_mos

class MOSDataset(torch.utils.data.Dataset):
    def __init__(self, root_path='', fs=16000):
        super().__init__()

        self.root_path = root_path
        self.utt_keys = ['fileid_224','fileid_793','fileid_438','fileid_810', 'fileid_24', 'fileid_985', 'fileid_734', 'fileid_11']
        self.fs = fs

        self.wav_pairs, self.mos_pairs = self.load_wavs()
        assert len(self.wav_pairs) == len(self.mos_pairs)

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
        teams = [scp.split('/')[-2] for scp  in team_scp] 
        teams_wav = {}
        teams_mos = {}

        for team in teams:
            wav_scp = f'{self.root_path}/{team}/wav.scp'
            teams_wav[team] = read_scp(wav_scp)
            mos_scp = f'{self.root_path}/{team}/score/mos/MOS.scp'
            teams_mos[team] = read_scp(mos_scp)

        keys = self.utt_keys if self.utt_keys else teams_wav[team].keys()
        
        for team in teams:
            teams_wav[team] = { k: teams_wav[team][k] for k in keys}
            teams_mos[team] = { k: teams_mos[team][k] for k in keys}

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

    # /home/wangyou.zhang/urgent2024_challenge/submissions
    # /home/chenda/workspace/urgent26/urgent24/submissions
    # /home/chenda.li/workspace/urgent26/urgent25_submissions

    '''
    # prepare real A/B test

    dir = "/home/jiahe.wang/workspace/urgent26/local"
    tag = "real_abtest_4"
    test_dir = os.path.join(dir, tag)
    os.system("mkdir -p " + test_dir + " || true")
    for bin in range(6):
        group_name = "group" + str(bin)
        group_dir = os.path.join(test_dir, group_name)
        log_path = os.path.join(group_dir, "log.txt")
        os.system("mkdir -p " + group_dir + " || true")
        os.system("touch " + log_path + " || true")

    data = ABDataset_urgent25()
    wav_pairs, mos_pairs = data.wav_pairs, data.mos_pairs
    assert len(wav_pairs) == len(mos_pairs)
    l = len(wav_pairs)
    wav_group = {i:[] for i in range(6)}
    mos_group = {i:[] for i in range(6)}

    for i in range(l):
        gap = abs(mos_pairs[i][0] - mos_pairs[i][1])
        bin = gap // 0.4
        if bin >= 6:
            continue
        wav_group[bin].append(wav_pairs[i])
        mos_group[bin].append(mos_pairs[i])

    for bin in range(6):
        print("bin", bin, len(mos_group[bin]))
        index = range(len(mos_group[bin]))
        chosen_idx = random.sample(index, 20)
        print(chosen_idx)
        group_name = "group" + str(bin)
        group_dir = os.path.join(test_dir, group_name)
        log_path = os.path.join(group_dir, "log.txt")
        for i, idx in enumerate(chosen_idx):
            wav_pair = wav_group[bin][idx]
            mos_pair = mos_group[bin][idx]
            wav_1, sr_1 = librosa.load(wav_pair[0], sr=None)
            wav_2, sr_2 = librosa.load(wav_pair[1], sr=None)
            sf.write(os.path.join(group_dir, "pair" + str(i + 1) + "_0.wav"), wav_1, sr_1)
            sf.write(os.path.join(group_dir, "pair" + str(i + 1) + "_1.wav"), wav_2, sr_2)
            with open(log_path, "a") as f:
                if mos_pair[0] > mos_pair[1]:
                    ans = "0"
                else:
                    ans = "1"
                f.write("pair" + str(i + 1) + " " + ans + " " + str(mos_pair[0]) + " " + str(mos_pair[1]) + "\n")
    '''

    teams_wav, teams_mos = ABDataset_urgent24().load_wavs_for_test()
    check_teams_wavs_mos(teams_mos=teams_mos, teams_wav=teams_wav)
    teams_wav, teams_mos = ABDataset_urgent24().load_wavs_for_valid()
    check_teams_wavs_mos(teams_mos=teams_mos, teams_wav=teams_wav)

    teams_wav, teams_mos = MOSDataset25().load_wavs()
    check_teams_wavs_mos(teams_mos=teams_mos, teams_wav=teams_wav)
    teams_wav, teams_mos = MOSDataset25().load_wavs_by_lang("de")
    check_teams_wavs_mos(teams_mos=teams_mos, teams_wav=teams_wav)
    teams_wav, teams_mos = MOSDataset25().load_wavs_by_lang("zh")
    check_teams_wavs_mos(teams_mos=teams_mos, teams_wav=teams_wav)
    teams_wav, teams_mos = MOSDataset25().load_wavs_by_lang("jp")
    check_teams_wavs_mos(teams_mos=teams_mos, teams_wav=teams_wav)

    teams_wav, teams_mos = ABDataset_chime().load_wavs_for_test()
    check_teams_wavs_mos(teams_mos=teams_mos, teams_wav=teams_wav)
