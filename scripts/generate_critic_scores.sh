#!/bin/bash
#
# The "#" before the "SBATCH" parameters do not comment it out! Use triple "###" to comment something out.
# Check our wiki for valid QOS / PARTITION / ACCOUNT combinations and resource limits!
# https://wiki.ukp.informatik.tu-darmstadt.de/bin/view/Services/Campus/ComputeCluster
# You can shorten this example script and adapt to create your own one.
#
# Give your job a proper name
#SBATCH --job-name=ll_val_unit_tests
#
# How many cpus to request
#SBATCH --cpus-per-task=16
#
# How much memory to request
#SBATCH --mem=512GB
#
# How many gpus to request
#SBATCH --gres=gpu:2
#
# Limit runtime d-hh:mm:ss - here limited to 1min
#SBATCH --time=1-00:00:00
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
#SBATCH --output=/storage/athene/work/sakharova/generate_critic_scores_train.output
#SBATCH --error=/storage/athene/work/sakharova/generate_critic_scores_train.error

critic_path=exps/critic_models/codet5-base-all-1ep/final_checkpoint
tokenizer_name=Salesforce/codet5-base
test_path=data/APPS/critic_train_se_only

output_path=outputs/results_for_presentation/codet5-critic/train/codet5-base-REAL-all-1ep

CUDA_VISIBLE_DEVICES=0 python generate.py \
    --model_name ${critic_path} \
    --tokenizer_name ${tokenizer_name} \
    --test_path ${test_path} \
    --output_path ${output_path} \
    --critic_scores --use_output_path