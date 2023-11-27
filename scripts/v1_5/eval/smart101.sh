#!/bin/bash

python -m llava.eval.model_vqa_smart \
    --model-path /remote-home/syjiang/.cache/huggingface/hub/models--liuhaotian--llava-v1.5-13b \
    --question-file /remote-home/syjiang/repo/InternLM-XComposer/datasets/smart101_qa_mini1000_random.json \
    --answers-file results/llava-v1.5-13b-smart101-1000random_cot.json \
    --temperature 0 \
    --conv-mode vicuna_v1