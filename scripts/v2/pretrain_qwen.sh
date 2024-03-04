#!/bin/bash

#SBATCH -J pretrain_qwen
#SBATCH --partition=x090
#SBATCH --nodes=1
#SBATCH --gres=gpu:2 
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1    
#SBATCH --mem-per-cpu=8G  
#SBATCH --output=logs/qwen1.8b_pretrain.out
###SBATCH --kill-on-bad-exit=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

GPUS_PER_NODE=2
NNODES=$SLURM_NNODES

echo Node IP: $head_node_ip nodes_array: $nodes_array
srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号

export LOGLEVEL=INFO
export NCCL_DEBUG=ERROR
export NCCL_SOCKET_IFNAME="eth0"
export MASTER_PORT=29333

srun --jobid $SLURM_JOBID python -u -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_id $MASTER_PORT --rdzv_backend c10d --rdzv_endpoint $head_node_ip:$MASTER_PORT \
    --node_rank $SLURM_PROCID \
    llava/train/train_mem_moe.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path /remote-home/share/models/Qwen1.5-1.8B-Chat \
    --version plain \
    --data_path /remote-home/syjiang/datasets/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder /remote-home/syjiang/datasets/LLaVA-Pretrain/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir /remote-home/syjiang/checkpoints/qwen1.5-1.8b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
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
