#!/bin/bash
#
# The "#" before the "SBATCH" parameters do not comment it out! Use triple "###" to comment something out.
# Check our wiki for valid QOS / PARTITION / ACCOUNT combinations and resource limits!
# https://wiki.ukp.informatik.tu-darmstadt.de/bin/view/Services/Campus/ComputeCluster
# You can shorten this example script and adapt to create your own one.
#
# Give your job a proper name
#SBATCH --job-name=llamaipo
#
# How many cpus to request
#SBATCH --cpus-per-task=16
#
# How much memory to request
#SBATCH --mem=1TB
#
# How many gpus to request
#SBATCH --gres=gpu:2
#
# Limit runtime d-hh:mm:ss - here limited to 1min
#SBATCH --time=2-00:00:00
#
# PARTITION to run in (athene-only people need to specify partition "gpu-athene" - otherwise the default "gpu" partition, which can only be used by UKP members, is selected leading to errors during job submission!)
#SBATCH --partition=yolo
#
# ACCOUNT to use (default account for athene-only people is "athene-researcher" and therefore does not need to be specified - check your accounts with command: "sshare -U")
###SBATCH --account=athene-researcher
#
# QOS to use (default QOS for everyone is "gpu" and therefore does not need to be specified)
###SBATCH --qos=yolo
#
# Define standard output files - make sure those files exist
#SBATCH --output=/storage/athene/work/sakharova/sft_1ep_dpo_10ep_1000_2e-4_classic_AGAIN.output
#SBATCH --error=/storage/athene/work/sakharova/sft_1ep_dpo_10ep_1000_2e-4_classic_AGAIN.error

#module load cuda/12.2
path_to_dataset=data/APPS/llama_dpo_100.json
model_path="exps/CodeLlama-13B-Python-hf-finetuned"
output_dir="outputs/llama_dpo_models/sft_1ep_dpo_1ep_100"
tokenizer_name="codellama/CodeLlama-13b-Python-hf"
beta=0.1
epochs=1
loss_type=sigmoid
lr=2e-4

python llama_dpo_trainer.py --path_to_dataset $path_to_dataset --model_path $model_path \
    --output_dir $output_dir --tokenizer_name $tokenizer_name \
    --beta $beta --epochs $epochs --loss_type $loss_type --lr $lr
