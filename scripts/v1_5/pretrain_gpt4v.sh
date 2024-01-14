#!/bin/bash

python -u -m torch.distributed.run \
    --nproc_per_node 4 \
    --nnodes 1 \
    --rdzv_id 29572 --rdzv_backend c10d --rdzv_endpoint localhost:29572 \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /remote-home/yushengliao/syjiang/checkpoints/vicuna-7b-v1.5 \
    --version llava_llama_2 \
    --data_path /remote-home/yushengliao/syjiang/datasets/share_gpt4v/sharegpt4v_mix665k_cap23k_coco-ap9k_lcs3k_sam9k_div2k.json \
    --image_folder /remote-home/yushengliao/syjiang/datasets/share_gpt4v \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /remote-home/syjiang/checkpoints/llava-v1.5-7b-pretrain-sharegpt4v \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
