import os
import shutil
import pickle
import numpy as np
import json

root = "outputs/results_for_presentation/codet5-actor"
    

for model in os.listdir(root):
    n_all_0 = 0
    n_total = 0
    path = os.path.join(root, model, "codes")
    for task in os.listdir(path):
        with open(os.path.join(path, task), "r") as f:
            data = json.load(f)
            codes = list(data.values())[0]["code"]
            n_all_0 += 1 if all([code == "" for code in codes]) else 0
            n_total += 1
    print(model)
    print(n_all_0 / n_total)
    print()
