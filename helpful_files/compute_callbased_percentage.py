"""
Outputs percentages of call-based tasks in the train and test data respectively.
"""
import os

def compute_callbased_percentage(folder: str):
    total = len(os.listdir(folder))
    count = 0
    for task in os.listdir(folder):
        if "starter_code.py" in os.listdir(folder + "/" + task):
            count += 1
    return count / total
            

train_percentage = compute_callbased_percentage("data/APPS/train")
test_percentage = compute_callbased_percentage("data/APPS/test") 
print("Call-based percentage in train data: ", train_percentage)
print("Call-based percentage in test data: ", test_percentage)