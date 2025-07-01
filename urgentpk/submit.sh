#!/bin/bash
#SBATCH -o logs/job.%j.out
#SBATCH -p 4090
#SBATCH --qos=qnormal
#SBATCH -J train
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32G

python train_urgentpk.py \
    --train_tag mel_resnet34_urgent24 \
    --batch_size 12 \
    --resume true \
    --learning_rate 0.0001 \
    --dataset /home/jiahe.wang/workspace/urgentpk/urgentpk/local/PKDataset24 \
    --delta 0.30 \
    --encoder mel \
    --backbone resnet34 \
    --tune_utmos False \
    --seed 1996 \
    # --init_ckpt /home/jiahe.wang/workspace/urgent26/abtest_model/exp/FNL_mel_resnet34_vctk_1e-4_1.0/ab_test/version_0/checkpoints/last.ckpt