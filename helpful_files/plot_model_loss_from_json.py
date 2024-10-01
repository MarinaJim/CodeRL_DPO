import json
from ast import literal_eval
import matplotlib.pyplot as plt
import math

with open("helpful_files/output_with_validation/train_losses.json", "r") as f:
    train_output_0_9 = json.load(f)

with open("helpful_files/output_with_validation/val_losses.json", "r") as f:
    val_losses = json.load(f)[:10]


losses = []
for line in train_output_0_9:
        line["epoch"] = math.floor(line["epoch"])
        if line["epoch"] < 10:
            losses.append(line)

grouped_losses = {}
for loss in losses:
    if loss["epoch"] + 1 in grouped_losses:
        grouped_losses[loss["epoch"] + 1].append(loss["loss"])
    else:
        grouped_losses[loss["epoch"] + 1] = [loss["loss"]]

avg_losses = [sum(losses) / len(losses) for losses in grouped_losses.values()]


plt.figure(figsize=(6, 5))
plt.plot(grouped_losses.keys(), avg_losses, label="Train loss")
plt.plot(grouped_losses.keys(), val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.xticks(list(grouped_losses.keys()))
plt.title("CodeT5: Train and validation loss")
plt.legend()
plt.savefig("helpful_files/output_without_validation/loss_plot_0-9.png")

