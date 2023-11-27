#!/bin/bash

#SBATCH --partition=x090

#SBATCH --job-name=multinode-example
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=2G

###SBATCH --chdir=/remote-home/syjiang/repo/LLaVA
###SBATCH --output=logs/output.txt
###SBATCH --kill-on-bad-exit=1


head_node_ip=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)

echo Node IP: $head_node_ip nodes_array: $SLURM_NODELIST

export LOGLEVEL=ERROR
export NCCL_DEBUG=INFO #TRACE # 可以改成 ERROR，减少输出量
#export CUDA_LAUNCH_BLOCKING=1

srun bash -c 'echo $SLURMD_NODENAME-$SLURM_JOB_GPUS' # 打印出不同机器上分配的显卡编号

conda activate llava
srun scripts/v1_5/pretrain.sh 

sync && echo "success"