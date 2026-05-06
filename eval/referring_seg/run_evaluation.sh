#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="./:$PYTHONPATH"
MASTER_PORT=22999

# Positional arguments for the bash scripts
CKPT_PATH=$1
RESULT_PATH=$2
RETAIN_TOKENS=$3

# RefCOCO
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco|val" --results_path "$RESULT_PATH" --num_retain_tokens $RETAIN_TOKENS --pretrained
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco|testA" --results_path "$RESULT_PATH" --num_retain_tokens $RETAIN_TOKENS --pretrained
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco|testB" --results_path "$RESULT_PATH" --num_retain_tokens $RETAIN_TOKENS --pretrained

# RefCOCO+
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|val" --results_path "$RESULT_PATH" --num_retain_tokens $RETAIN_TOKENS --pretrained
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|testA" --results_path "$RESULT_PATH" --num_retain_tokens $RETAIN_TOKENS --pretrained
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcoco+|testB" --results_path "$RESULT_PATH" --num_retain_tokens $RETAIN_TOKENS --pretrained

# RefCOCOg
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcocog|val" --results_path "$RESULT_PATH" --num_retain_tokens $RETAIN_TOKENS --pretrained
deepspeed --master_port="$MASTER_PORT" eval/referring_seg/infer_and_evaluate.py --version "$CKPT_PATH" --refer_seg_data "refcocog|test" --results_path "$RESULT_PATH" --num_retain_tokens $RETAIN_TOKENS --pretrained
