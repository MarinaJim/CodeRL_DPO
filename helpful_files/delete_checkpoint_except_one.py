import os
import json
import pickle
import shutil

dir = "outputs/dpo_models/original1ep_0.1_10ep_threshold1_1000_fixed/"
lst = os.listdir(dir)
lst = [int(point.replace("checkpoint-", "")) for point in lst]
for i, point in enumerate(sorted(lst)):
    if i != len(lst) - 1:
        shutil.rmtree(f"{dir}/checkpoint-{point}")