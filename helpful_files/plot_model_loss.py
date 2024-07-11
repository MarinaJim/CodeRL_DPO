import json
from ast import literal_eval
import matplotlib.pyplot as plt
import math

MIN_EPOCH = 10
with open("/storage/athene/work/sakharova/train_actor.output", "r") as f:
    trai_output = f.read()

train_output = trai_output.split("\n")

losses = []

for line in train_output:
    if "\'loss\'" in line:
        line = literal_eval(line)
        line["epoch"] = MIN_EPOCH + math.floor(line["epoch"])
        losses.append(line)
#losses = losses[:146000]

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
plt.savefig("helpful_files/output_files/loss_plot10-19.png")

with open("helpful_files/output_files/losses10-19.json", "w") as f:
    json.dump(losses, f)
