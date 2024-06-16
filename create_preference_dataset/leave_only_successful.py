import shutil
import os

for task in os.listdir("data/APPS/preference"):
    path = os.path.join("data/APPS/preference", task, "parameters.json")
    if not os.path.exists(path):
        shutil.rmtree(os.path.join("data/APPS/preference", task))