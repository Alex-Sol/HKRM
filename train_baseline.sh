#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=1 python trainval_baseline.py --dataset dior --bs 4 --lr 0.004 --lr_decay_step 8  --nw 1 \
              --epochs 200 --log_dir log --save_dir /home/zengli/HKRM/checkpoints/
