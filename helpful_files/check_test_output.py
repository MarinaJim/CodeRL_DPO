import os
import pickle
import json

with open("data/APPS/dpo_dataset.json", "r") as f:
    dpo_dataset = json.load(f)

print(len(dpo_dataset["prompt"]))
