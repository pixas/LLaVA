#!/bin/bash

CKPT="$1"
python -m llava.eval.model_vqa_loader \
    --model-path /remote-home/yushengliao/syjiang/checkpoints/$CKPT \
    --model-base /remote-home/yushengliao/syjiang/checkpoints/vicuna-7b-v1.5 \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${CKPT}.jsonl \
    --temperature 0 \
    --conv-mode llava_v1 \

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT}.jsonl
