import json
from ast import literal_eval
import matplotlib.pyplot as plt

with open("/storage/athene/work/sakharova/train_actor.output", "r") as f:
    trai_output = f.read()

train_output = trai_output.split("\n")

losses = []

for line in train_output:
    if "\'loss\'" in line:
        line = literal_eval(line)
        losses.append(line)

losses = [output["loss"] for output in losses]
#losses = [loss for index, loss in enumerate(losses) if index % 10 == 0]
plt.plot(range(len(losses)), losses)
plt.savefig("loss_plot.png")