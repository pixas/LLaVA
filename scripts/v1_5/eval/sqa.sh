#!/bin/bash

python -m llava.eval.model_vqa_science \
    --model-path /remote-home/syjiang/checkpoints/llava-v1.5-7b-4moe-lora \
    --model-base /remote-home/share/models/vicuna-7b-v1.5 \
    --question-file /remote-home/syjiang/datasets/ScienceQA/data/scienceqa/llava_test_CQM-A.json \
    --image-folder /remote-home/syjiang/datasets/ScienceQA/data/scienceqa/images/test \
    --answers-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-4moe-lora.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1

python llava/eval/eval_science_qa.py \
    --base-dir /remote-home/syjiang/datasets/ScienceQA/data/scienceqa \
    --result-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-4moe-lora.jsonl \
    --output-file ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-4moe-lora_output.jsonl \
    --output-result ./playground/data/eval/scienceqa/answers/llava-v1.5-7b-4moe-lora_result.json
