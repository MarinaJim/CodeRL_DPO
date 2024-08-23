import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def pass_at_k(n, c, k):
    """
    n: total #samples
    c: #correct samples
    k: k
    """
    if n - c < k:
        return 1.0
    return 1 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

orig_dir = "data/APPS/orig_test"
model_dir = "outputs/codet5_python_evaluation_on_apps"
k = 5
results_dir = os.path.join(model_dir, "test_results")
results = sorted(os.listdir(results_dir))
difficulties = {}
for i in results:
    path = f"{results_dir}/{i}"
    i = int(i.replace(".pkl", ""))
    output = pd.read_pickle(path)
    scores = [score if score == 1 else 0 for score in output[i]["results"]]

    i = "0"*(4 - len(str(i))) + str(i)
    with open(os.path.join(orig_dir, i, "metadata.json"), "r") as f:
        data = json.load(f)
        difficulty = data["difficulty"]
    if difficulty not in difficulties.keys():
        difficulties[difficulty] = scores
    else:
        difficulties[difficulty].extend(scores)

for difficulty, scores in difficulties.items():
    n = len(scores)
    c = sum(scores)
    difficulties[difficulty] = pass_at_k(n, c, k)

print(difficulties)


    

        


    