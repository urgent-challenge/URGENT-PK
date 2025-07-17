import numpy as np
import os
from tqdm import tqdm
import soundfile as sf
from datasets import load_dataset
from collections import defaultdict
import argparse
import json
import random

'''
    huggingface url: https://huggingface.co/datasets/urgent-challenge/urgent2024_mos
    samples be like:
    {
        'id': 'P426-fileid_184',
        'audio': <datasets.features._torchcodec.AudioDecoder object at 0x760de4d4f010>,
        'sampling_rate': 22050,
        'mos': 2.375,
        'raw_ratings': [3, 3, 4, 2, 2, 2, 1, 2]
    }
'''

def get_valid_uids(teams_dict):
    valid_uids = set()
    for team in teams_dict:
        valid_uids = valid_uids.intersection(set(teams_dict[team]['uids'])) if valid_uids else set(teams_dict[team]['uids'])
    valid_uids = list(valid_uids)
    return valid_uids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, default='none', help='the folder of the PKDataset, excluding the audio files.')
    parser.add_argument('--wav_dir', type=str, default='none', help='The folder to save the audio files.')
    parser.add_argument('--cv_num', type=int, default=10, help='Number of utterances in the validation subset.')
    args = parser.parse_args()
    write_dir = args.dataset_dir
    save_dir = args.wav_dir
    cv_num = args.cv_num

    if write_dir == 'none':
        write_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PKDataset')
    if save_dir == 'none':
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'PKDataset_wavs')
    os.makedirs(write_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    data = load_dataset('urgent-challenge/urgent2024_mos')

    teams_dict = dict()

    for i, sample in tqdm(enumerate(data['test'])):
        id = sample['id']
        audio = sample['audio']['array']
        fs = sample['sampling_rate']
        mos = sample['mos']
        # raw_mos = sample['raw_ratings']
        
        team, uid = id.split('-')
        if team not in teams_dict:
            teams_dict[team] = {'uids': [], 'file_path': [], 'mos': []}
            os.makedirs(os.path.join(write_dir, team), exist_ok=True)
            os.makedirs(os.path.join(save_dir, team), exist_ok=True)

        file_path = os.path.join(save_dir, team, uid + '.wav')
        sf.write(file_path, audio, fs)
        teams_dict[team]['uids'].append(uid)
        teams_dict[team]['file_path'].append(file_path)
        teams_dict[team]['mos'].append(mos)

    valid_uids = get_valid_uids(teams_dict)

    for team in teams_dict:
        with open(os.path.join(write_dir, team, 'wav.scp'), 'w') as f:
            for i, uid in enumerate(teams_dict[team]['uids']):
                if uid in valid_uids:
                    path = teams_dict[team]['file_path'][i]
                    f.write(f'{uid} {path}\n')
        with open(os.path.join(write_dir, team, 'mos.scp'), 'w') as f:
            for i, uid in enumerate(teams_dict[team]['uids']):
                if uid in valid_uids:
                    mos = teams_dict[team]['mos'][i]
                    f.write(f'{uid} {mos}\n')

    cv_uids = random.sample(valid_uids, cv_num)
    tt_uids = valid_uids
    tr_uids = [uid for uid in valid_uids if uid not in cv_uids]
    split = {
        'tr': tr_uids,
        'cv': cv_uids,
        'tt': tt_uids,
    }
    with open(os.path.join(write_dir, 'split.json'), 'w') as f:
        json.dump(split, f)
    
    print('### Building PKDataset from Urgent2024_mos finish! ###')
    print('total teams:', len(teams_dict))
    print('utts in each team:', len(valid_uids))
    print('PKDataset folder:', write_dir)
    print('wavs saved in:', save_dir)


if __name__ == '__main__':
    main()