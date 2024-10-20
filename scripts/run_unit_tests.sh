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
#SBATCH --mem=16GB
#
# How many gpus to request
#SBATCH --gres=gpu:1
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
#SBATCH --output=/storage/athene/work/sakharova/run_unit_tests.output
#SBATCH --error=/storage/athene/work/sakharova/run_unit_tests.error

code_path=/storage/athene/work/sakharova/CodeRL_DPO/outputs/results_for_presentation/llama/sft_1ep_dpo_1ep_100_2e-4_0.1_classic/codes
output_path=/storage/athene/work/sakharova/CodeRL_DPO/outputs/results_for_presentation/llama/sft_1ep_dpo_1ep_100_2e-4_0.1_classic/test_results_check
test_path=data/APPS/test

example_tests=0 # 0: run hidden unit tests; 1: run example unit tests 
threads=10
start=0
end=2500

if [ ! -d $output_path ] 
then
    echo "Directory DOES NOT exists." 
    mkdir $output_path
fi

index=0
for (( i=$start;i<$end;i++ )) ; do 
    echo 'testing sample index #' ${i}
    ((index++))   
    (
    python test_one_solution.py \
        --code_path ${code_path} \
        --output_path ${output_path} \
        --test_path $test_path \
        --example_tests $example_tests \
        --i $i 
    ) &        
    if (( $index % $threads == 0 )); then wait; fi 
done 

wait 
