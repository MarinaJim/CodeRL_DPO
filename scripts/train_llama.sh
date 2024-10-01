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
#SBATCH --gres=gpu:2
#
# Limit runtime d-hh:mm:ss - here limited to 1min
#SBATCH --time=3-00:00:00
#
# PARTITION to run in (athene-only people need to specify partition "gpu-athene" - otherwise the default "gpu" partition, which can only be used by UKP members, is selected leading to errors during job submission!)
#SBATCH --partition=yolo

# ACCOUNT to use (default account for athene-only people is "athene-researcher" and therefore does not need to be specified - check your accounts with command: "sshare -U")
###SBATCH --account=athene-researcher
#
# QOS to use (default QOS for everyone is "gpu" and therefore does not need to be specified)
###SBATCH --qos=yolo
#
# Define standard output files - make sure those files exist
#SBATCH --output=/storage/athene/work/sakharova/train_llama.output
#SBATCH --error=/storage/athene/work/sakharova/train_llama.error

module load cuda/12.2
python -m llama_recipes.finetuning \
    --use_peft --peft_method lora \
    --model_name codellama/CodeLlama-13b-Python-hf \
    --dataset "custom_dataset" --custom_dataset.file "datasets_apps/apps_dataset.py" \
    --output_dir exps/CodeLlama-13B-Python-hf-peft-evaluated-test \
    --batch_size_training 4 --gradient_accumulation_steps 1 \
    --run_validation True --num_epochs 10 --num_workers_dataloader 4 