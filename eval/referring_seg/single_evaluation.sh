#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="./:$PYTHONPATH"
MASTER_PORT=22999

CKPT_PATH=$1
REF_SEG_DATASET=$2
RESULT_PATH=$3
RETAIN_TOKENS=$4

deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py \
    --version "$CKPT_PATH" \
    --refer_seg_data "$REF_SEG_DATASET" \
    --results_path "$RESULT_PATH" \
    --num_retain_tokens $RETAIN_TOKENS \
    --pretrained