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
#SBATCH --mem=256GB
#
# How many gpus to request
#SBATCH --gres=gpu:2
#
# Limit runtime d-hh:mm:ss - here limited to 1min
#SBATCH --time=0-23:00:00
#
# PARTITION to run in (athene-only people need to specify partition "gpu-athene" - otherwise the default "gpu" partition, which can only be used by UKP members, is selected leading to errors during job submission!)
#SBATCH --partition=gpu-athene

# ACCOUNT to use (default account for athene-only people is "athene-researcher" and therefore does not need to be specified - check your accounts with command: "sshare -U")
###SBATCH --account=gpu-large
#
# QOS to use (default QOS for everyone is "gpu" and therefore does not need to be specified)
###SBATCH --qos=gpu-large
#
# Define standard output files - make sure those files exist
#SBATCH --output=/storage/athene/work/sakharova/download_llama.output
#SBATCH --error=/storage/athene/work/sakharova/download_llama.error


python download_llama.py