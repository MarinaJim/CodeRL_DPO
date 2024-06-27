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
#SBATCH --time=0-16:00:00
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

preference_dir=data/APPS/preference_test
train_dir=data/APPS/train

python create_preference_dataset/transfer_create_sample_calls.py -pp $preference_dir -tp $train_dir

for task in "$preference_dir"/*
do
    if [ -e "$task/sample_call.py" ]; then
        cd $task
        monkeytype run sample_call.py
        monkeytype apply solution
        cd ../../../..
    fi
    python create_preference_dataset/create_inputs_for_task.py -t $task
    python create_preference_dataset/create_outputs_for_task.py -t $task
    python create_preference_dataset/bring_data_into_preference.py --tr $train_dir -t $task
done
