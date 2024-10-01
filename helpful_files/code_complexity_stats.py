import json
import os
from collections import Counter

def get_stats(folder):
    difficulties = []
    for task in os.listdir(folder):
        with open(os.path.join(folder, task, "metadata.json"), "r") as f:
            data = json.load(f)
        difficulties.append(data["difficulty"])
    return difficulties


train = get_stats("data/APPS/train")
test = get_stats("data/APPS/orig_test")

print(Counter(train))
train = {key: value / len(train) for key, value in Counter(train).items()}
test = {key: value / len(test) for key, value in Counter(test).items()}

print(train)
print(test)