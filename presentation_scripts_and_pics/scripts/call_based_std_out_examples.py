import os
import json

folder = "data/APPS/train"
std = []

for task in os.listdir(folder):
    with open(os.path.join(folder, task, "solutions.json"), "r") as f:
        data = json.load(f)
    data = [d for d in data if "def " in d]
    if data == []:
        continue
    minim = sorted(data, key=len)[0]
    if "starter_code.py" in os.listdir(os.path.join(folder, task)):
        std.append(minim)

std = sorted(std, key=len)[600:630]
for s in std:
    print(s)
    print("--------")