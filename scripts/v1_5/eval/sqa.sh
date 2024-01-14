#!/bin/bash

CKPT="$1"
python -m llava.eval.model_vqa_science \
    --model-path /remote-home/yushengliao/syjiang/checkpoints/$CKPT \
    --model-base /remote-home/yushengliao/syjiang/checkpoints/vicuna-7b-v1.5 \
    --question-file /remote-home/yushengliao/syjiang/datasets/ScienceQA/data/scienceqa/llava_test_CQM-A.json \
    --image-folder /remote-home/yushengliao/syjiang/datasets/ScienceQA/data/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /remote-home/yushengliao/syjiang/datasets/ScienceQA/data/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/$CKPT.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/${CKPT}_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/${CKPT}_result.json
