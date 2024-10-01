import json
import os
import math
import matplotlib.pyplot as plt
from plot_model_loss import plot_and_save

def get_losses(path, key):
    with open(path, "r") as f:
        data = json.load(f)

    arr = data[key]
    n = int(len(arr)/10)

    sliced_arr = [arr[i:i+n] for i in range(0, len(arr), n)]

    n_nan = []
    averages = []
    for epoch in sliced_arr:
        epoch_without_nan = [d_point for d_point in epoch if not math.isnan(d_point)]
        averages.append(sum(epoch_without_nan) / len(epoch_without_nan))
        n_nan.append(len(epoch) - len(epoch_without_nan))
    return averages, [n / len(epoch) for n in n_nan]

path = "/storage/athene/work/sakharova/CodeRL_DPO/exps/CodeLlama-13B-Python-hf-peft-evaluated/metrics_data_None-2024-09-22_16-38-05.json"
val_loss, n_nan_val = get_losses(path, "val_step_loss")
train_loss, n_nan_train = get_losses(path, "train_step_loss")

print(n_nan_val)
print(n_nan_train)
plot_and_save(list(range(1, len(train_loss)+1)), 
              train_loss,
              val_loss, 
              "presentation_scripts_and_pics/pics/llama_warmup_loss.jpg",
              "CodeLlama: Train and validation loss")