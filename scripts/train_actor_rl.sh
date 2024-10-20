#!/bin/bash
#
# The "#" before the "SBATCH" parameters do not comment it out! Use triple "###" to comment something out.
# Check our wiki for valid QOS / PARTITION / ACCOUNT combinations and resource limits!
# https://wiki.ukp.informatik.tu-darmstadt.de/bin/view/Services/Campus/ComputeCluster
# You can shorten this example script and adapt to create your own one.
#
# Give your job a proper name
#SBATCH --job-name=sm_train_actor
#
# How many cpus to request
#SBATCH --cpus-per-task=16
#
# How much memory to request
#SBATCH --mem=1TB
#
# How many gpus to request
#SBATCH --gres=gpu:4
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
#SBATCH --output=/storage/athene/work/sakharova/train_actor_relative_returns.output
#SBATCH --error=/storage/athene/work/sakharova/train_actor_relative_returns.error

python \
    train_orig.py \
    --batch-size-per-replica 4 --grad-acc-steps 4 \
    --epochs 1 --lr 2e-6 \
    --save-freq 100 --log-freq 10 --save_total_limit 5 \
    --fp16 \
    --tuning_mode rl --model codet5-large \
    --model_path exps/codet5-large-ntp-py-2e-5-epoch0-traineval/checkpoint-14654 \
    --critic_scores_root outputs/results_for_presentation/codet5-critic/train/codet5-finetuned-critic-se-1ep \
    --tokenizer Salesforce/codet5-large-ntp-py --train-path data/APPS/critic_train_se_only \
    --include_gt True --save_dir outputs/rl_models/1ep-4rl-2gt-2e-6-critic-coderl-ft-se-1ep \
    --max_rl_per_task 4 --max_gt_per_task 2