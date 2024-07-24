import json
from ast import literal_eval
import matplotlib.pyplot as plt
import math

with open("helpful_files/output_files/losses0-9.json", "r") as f:
    train_output_0_9 = json.load(f)

with open("helpful_files/output_files/losses10-19.json", "r") as f:
    train_output_10_19 = json.load(f)


losses = []

for line in train_output_0_9:
        line["epoch"] = math.floor(line["epoch"])
        losses.append(line)

for line in train_output_10_19:
    line["epoch"] = math.floor(line["epoch"])
    losses.append(line)

grouped_losses = {}
for loss in losses:
    if loss["epoch"] in grouped_losses:
        grouped_losses[loss["epoch"]].append(loss["loss"])
    else:
        grouped_losses[loss["epoch"]] = [loss["loss"]]

avg_losses = [sum(losses) / len(losses) for losses in grouped_losses.values()]


plt.plot(grouped_losses.keys(), avg_losses)
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.xticks(list(grouped_losses.keys()))
plt.savefig("helpful_files/output_files/loss_plot_0-19.png")

