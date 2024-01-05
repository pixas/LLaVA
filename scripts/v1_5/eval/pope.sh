#!/bin/bash
CKPT="llava-v1.5-7b-qformer-3moe-lora"
python -m llava.eval.model_vqa_pope \
    --model-path /remote-home/syjiang/checkpoints/$CKPT \
    --model-base /remote-home/share/models/vicuna-7b-v1.5 \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --image-folder ./playground/data/eval/pope/val2014 \
    --answers-file ./playground/data/eval/pope/answers/$CKPT.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_pope.py \
    --annotation-dir ./playground/data/eval/pope/coco \
    --question-file ./playground/data/eval/pope/llava_pope_test.jsonl \
    --result-file ./playground/data/eval/pope/answers/$CKPT.jsonl
