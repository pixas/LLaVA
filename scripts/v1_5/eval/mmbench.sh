#!/bin/bash

SPLIT="mmbench_dev_20230712"
CKPT="llava-v1.5-7b-qformer-lora2"
python -m llava.eval.model_vqa_mmbench \
    --model-path /remote-home/syjiang/checkpoints/$CKPT \
    --model-base /remote-home/share/models/vicuna-7b-v1.5 \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/${CKPT}_llava_v1.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode llava_v1

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment ${CKPT}_llava_v1
