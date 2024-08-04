import pandas as pd
import json
import os
import matplotlib.pyplot as plt

all_outputs = []
model_dir = "outputs/dpo_results/checkpoint-1ep-4bs-1ga-2e-5-originalapps-0.1-1ep"
results_dir = os.path.join(model_dir, "test_results")
distribution_plot_path = os.path.join(model_dir, "test_results_dist.jpg")
results = sorted(os.listdir(results_dir))
errors = 0
for i in results:
    path = f"{results_dir}/{i}"
    i = int(i.replace(".pkl", ""))
    try:
        output = pd.read_pickle(path)
        all_outputs.append(output[i]["results"])
    except EOFError:
        errors += 1
        

sum_outputs = sum([sum(outputs) for outputs in all_outputs])
total_outputs = sum(len(outputs) for outputs in all_outputs)
mean_acc =  sum_outputs / total_outputs 

with open(os.path.join(model_dir, "mean_accuracy.txt"), "w") as f:
    f.write(f"Mean accuracy = {mean_acc}")

mean_accuracies = [sum(output) / len(output) for output in all_outputs if output != []]
plt.hist(mean_accuracies, bins=20, color="steelblue", edgecolor="black")
plt.savefig(distribution_plot_path)

print(errors)



    