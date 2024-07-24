import shutil
import os
import random

TEST_PATH = "data/APPS/test"
VALIDATION_PATH = "data/APPS/validation"
test_tasks = os.listdir(TEST_PATH)

random.seed(9054295)
k = 2500
validation_tasks = random.sample(test_tasks, k = k)

for task in validation_tasks:
    task_test_path = os.path.join(TEST_PATH, task)
    task_valid_path = os.path.join(VALIDATION_PATH, task)
    if os.path.exists(task_test_path):
        shutil.copytree(src=task_test_path, dst=task_valid_path)
        shutil.rmtree(task_test_path)
    else:
        shutil.copytree(src=task_test_path, dst=task_valid_path)

print(len(os.listdir(TEST_PATH)))
print(len(os.listdir(VALIDATION_PATH)))

"""
validation_tasks = os.listdir(VALIDATION_PATH)
for task in validation_tasks:
        task_test_path = os.path.join(TEST_PATH, task)
        task_valid_path = os.path.join(VALIDATION_PATH, task)
        if not os.path.exists(task_test_path):
                shutil.copytree(src=task_valid_path, dst=task_test_path)
        shutil.rmtree(task_valid_path)
"""