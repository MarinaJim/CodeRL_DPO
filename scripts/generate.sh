#!/bin/bash
#
# The "#" before the "SBATCH" parameters do not comment it out! Use triple "###" to comment something out.
# Check our wiki for valid QOS / PARTITION / ACCOUNT combinations and resource limits!
# https://wiki.ukp.informatik.tu-darmstadt.de/bin/view/Services/Campus/ComputeCluster
# You can shorten this example script and adapt to create your own one.
#
# Give your job a proper name
#SBATCH --job-name=sm_generate
#
# How many cpus to request
#SBATCH --cpus-per-task=10
#
# How much memory to request
#SBATCH --mem=64GB
#
# How many gpus to request
#SBATCH --gres=gpu:2
#
# Limit runtime d-hh:mm:ss - here limited to 1min
#SBATCH --time=0-23:00:00
#
# PARTITION to run in (athene-only people need to specify partition "gpu-athene" - otherwise the default "gpu" partition, which can only be used by UKP members, is selected leading to errors during job submission!)
#SBATCH --partition=gpu-athene
#
# ACCOUNT to use (default account for athene-only people is "athene-researcher" and therefore does not need to be specified - check your accounts with command: "sshare -U")
###SBATCH --account=athene-student
#
# QOS to use (default QOS for everyone is "gpu" and therefore does not need to be specified)
###SBATCH --qos=gpu
#
# Define standard output files - make sure those files exist
#SBATCH --output=/storage/athene/work/sakharova/generate.output
#SBATCH --error=/storage/athene/work/sakharova/generate.error

#model_name=Salesforce/codet5-large-ntp-py
model_name=exps/finetuned-codet5-large-ntp-py/checkpoint-29200
#model_path=/storage/athene/work/sakharova/CodeRL_DPO/models/codet5_large_ntp_py
tokenizer_name=Salesforce/codet5-large-ntp-py
tokenizer_path=models/codet5_tokenizer/
test_path=data/APPS/test/ 

#path_to_problems=/storage/athene/work/sakharova/CodeRL_DPO/data/dpo_indexes.json
path_to_problems=None
start=0
end=50
num_seqs_per_iter=10
num_seqs=10
temp=0.6
max_len=1024
output_path=outputs/codes/

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_name $model_name \
    --tokenizer_name $tokenizer_name \
    --tokenizer_path $tokenizer_path \
    --test_path $test_path \
    --output_path $output_path \
    -s $start -e $end \
    --max_len $max_len \
    --num_seqs $num_seqs --num_seqs_per_iter $num_seqs_per_iter \
    --temperature $temp \
#    --path_to_problems $path_to_problems
#     --model_path $model_path \