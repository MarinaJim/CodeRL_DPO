#!/bin/bash
#
# The "#" before the "SBATCH" parameters do not comment it out! Use triple "###" to comment something out.
# Check our wiki for valid QOS / PARTITION / ACCOUNT combinations and resource limits!
# https://wiki.ukp.informatik.tu-darmstadt.de/bin/view/Services/Campus/ComputeCluster
# You can shorten this example script and adapt to create your own one.
#
# Give your job a proper name
#SBATCH --job-name=sm_run_dpo
#
# How many cpus to request
#SBATCH --cpus-per-task=8
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
###SBATCH --account=athene-student
#
# QOS to use (default QOS for everyone is "gpu" and therefore does not need to be specified)
###SBATCH --qos=yolo
#
# Define standard output files - make sure those files exist
#SBATCH --output=/storage/athene/work/sakharova/generate_codet5.output
#SBATCH --error=/storage/athene/work/sakharova/generate_codet5.error

model_name=exps/outputs/rl_models/1ep-actor-codet5-ft-critic-codet5-base-se-1ep/final_checkpoint
#tokenizer_name=codellama/CodeLlama-13b-Python-hf
output_path=outputs/results_for_presentation/codet5-actor/1ep_2e-5_CodeT5-base-SE_all-data/codes
tokenizer_name=Salesforce/codet5-large-ntp-py
temp=0.6

test_path=data/APPS/test/

path_to_problems=None
start=0
end=2500
num_seqs_per_iter=10
num_seqs=10
max_len=1024

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_name $model_name \
    --tokenizer_name $tokenizer_name \
    --test_path $test_path \
    --output_path $output_path \
    -s $start -e $end\
    --max_len $max_len \
    --num_seqs $num_seqs --num_seqs_per_iter $num_seqs_per_iter \
    --temperature $temp\
