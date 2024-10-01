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
#SBATCH --mem=16GB
#
# How many gpus to request
#SBATCH --gres=gpu:1
#
# Limit runtime d-hh:mm:ss - here limited to 1min
#SBATCH --time=1-00:00:00
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
#SBATCH --output=/storage/athene/work/sakharova/create_dpo_dataset.output
#SBATCH --error=/storage/athene/work/sakharova/create_dpo_dataset.error

path_to_apps=data/APPS/train
# clean means without appended "import" commands
# bestworst means rejected code is the one with the lowest score
# fixed - only 1 and lower than 1
path_to_dpo=data/APPS/mixed_dataset_5000.json
path_to_results_1=/storage/athene/work/sakharova/CodeRL_DPO/outputs/warmup_codes_for_dpo/llama13b_python_ft_train/test_results
path_to_codes_1=/storage/athene/work/sakharova/CodeRL_DPO/outputs/warmup_codes_for_dpo/llama13b_python_ft_train/codes

path_to_results_2=/storage/athene/work/sakharova/CodeRL_DPO/outputs/warmup_codes_for_dpo/codet5_1ep-4bs-1ga-2e-5/train/test_results
path_to_codes_2=/storage/athene/work/sakharova/CodeRL_DPO/outputs/warmup_codes_for_dpo/codet5_1ep-4bs-1ga-2e-5/train/codes

best_threshold=1
worst_threshold=-2
max_len=5000
samples_per_task=1

python create_preference_dataset/create_dpo_dataset_both_models.py \
    --path_to_apps $path_to_apps --path_to_dpo $path_to_dpo \
    --path_to_results_1 $path_to_results_1 --path_to_codes_1 $path_to_codes_1 \
    --path_to_results_2 $path_to_results_2 --path_to_codes_2 $path_to_codes_2 \
    --best_threshold $best_threshold --worst_threshold $worst_threshold\
    --max_len $max_len --samples_per_task $samples_per_task