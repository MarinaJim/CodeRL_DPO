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
#SBATCH --output=/storage/athene/work/sakharova/codet5-base-true.output
#SBATCH --error=/storage/athene/work/sakharova/codet5-base-true.error

module load cuda/12.2

model=codet5-base
#model_path=exps/coderl-codet5-funetuned-critic
tokenizer=Salesforce/codet5-base
save_dir=critic_models/todelete
train_path=data/APPS/critic_train_se_only
include_gt=True
python \
    train_orig.py \
    --batch-size-per-replica 4 --grad-acc-steps 1 \
    --epochs 10 --lr 2e-5 \
    --save-freq 1000 --log-freq 10 --save_total_limit 5 \
    --tuning_mode critic --model $model \
    --fp16 --tokenizer $tokenizer \
    --save_dir $save_dir --train-path $train_path \
    --include_gt $include_gt