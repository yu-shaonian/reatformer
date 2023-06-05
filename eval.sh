#!/usr/bin/env bash
DTU_TESTING="/data-1/leiguojun/data/dtu/"
CKPT_FILE="./checkpoints/d20/model_000015.ckpt"
python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE $@
