"""
Creates a bash scripts to run unit tests for all solutions in the outputs/codes folder.
"""
import os
import sys
import json

script = """#!/bin/bash
#
# The "#" before the "SBATCH" parameters do not comment it out! Use triple "###" to comment something out.
# Check our wiki for valid QOS / PARTITION / ACCOUNT combinations and resource limits!
# https://wiki.ukp.informatik.tu-darmstadt.de/bin/view/Services/Campus/ComputeCluster
# You can shorten this example script and adapt to create your own one.
#
# Give your job a proper name
#SBATCH --job-name=sakharova_testscript
#
# Where to send job start / end messages to - comment in to use!
###SBATCH --mail-user=marina.sakharova@stud.tu-darmstadt.de
#SBATCH --mail-type=ALL
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
#SBATCH --time=0-05:00:00
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
#SBATCH --output=/storage/athene/work/sakharova/run_unit_tests.output
#SBATCH --error=/storage/athene/work/sakharova/run_unit_tests.error

code_path=/storage/athene/work/sakharova/CodeRL_DPO/outputs/codes/
output_path=/storage/athene/work/sakharova/CodeRL_DPO/outputs/test_results
test_path=/storage/athene/work/sakharova/CodeRL_DPO/data/APPS/train
example_tests=0
threads=10
"""

dpo_indexes =[index.replace(".json", "") for index in os.listdir("outputs/codes")]
dpo_indexes_str = "( " + str.join(" ", [str(i) for i in dpo_indexes]) + ")"
script += f"""

if [ ! -d $output_path ] 
then
    echo "Directory DOES NOT exists." 
    mkdir $output_path
fi
dpo_indexes={dpo_indexes_str}
index=0
for i in "${{!dpo_indexes[@]}}"; do 
    echo 'testing sample index #' ${{i}}
    ((index++))   
    (
    python test_one_solution.py \\
        --code_path ${{code_path\}} \\
        --output_path ${{output_path}} \\
        --test_path $test_path \\
        --example_tests $example_tests \\
        --i $i
    ) &        
    if (( $index % $threads == 0 )); then wait; fi 
done 

wait 

"""
with open("/storage/athene/work/sakharova/CodeRL_DPO/scripts/run_unit_tests_train.sh", "w") as f:
    f.write(script)