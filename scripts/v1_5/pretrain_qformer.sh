#!/bin/bash

#SBATCH -J FT-qformer-llava7b
#SBATCH --partition=x090
#SBATCH --nodes=1
#SBATCH --gres=gpu:4  
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=4G  
#SBATCH --output=logs/llava7b_qformer-pretrain-pretrain_5e4.out
###SBATCH --kill-on-bad-exit=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

GPUS_PER_NODE=4
NNODES=$SLURM_NNODES

echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号

export LOGLEVEL=INFO
export NCCL_DEBUG=ERROR
export NCCL_SOCKET_IFNAME="eth0"
export MASTER_PORT=29579

srun --jobid $SLURM_JOBID python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_id $MASTER_PORT --rdzv_backend c10d --rdzv_endpoint $head_node_ip:$MASTER_PORT \
    --node_rank $SLURM_PROCID \
    llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /remote-home/share/models/vicuna-7b-v1.5 \
    --version plain \
    --data_path /remote-home/syjiang/datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /remote-home/syjiang/datasets/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type qformer \
    --qformer_text_input True \
    --qformer_use_pretrained True \
    --mm_projector_experts 4 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /remote-home/syjiang/checkpoints/llava-v1.5-7b-pretrain-qformer-pretrained_5e4 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
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
