import shutil
import os

train_root = "data/APPS/train"
preference_root = "data/APPS/preference"
dpo_tasks = os.listdir(preference_root)
for task in dpo_tasks:

    train_path = os.path.join(train_root, task)
    preference_path = os.path.join(preference_root, task)
    """
    shutil.copy(os.path.join(train_path, "question.txt"), preference_path)
    if os.path.exists(os.path.join(train_path, "starter_code.py")):
        shutil.copy(os.path.join(train_path, "starter_code.py"), preference_path)"""
    if os.path.exists(os.path.join(preference_path, "monkeytype_io.json")):
        print(preference_path)
    else:
        print("not found")