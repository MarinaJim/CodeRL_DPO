"""
Outputs percentages of call-based tasks in the train and test data respectively.
"""
import os
import json
from collections import Counter
import sys
from helper_functions import get_call_based, get_class_tasks, get_number_inputs, get_one_class_one_def, get_solution_class

def compute_percentage(tasks, folder):
    total = os.listdir(folder)
    return len(tasks) / len(total)

def main():
    # First print general information about the dataset
    print("---General stats about dataset---")
    call_based_train = get_call_based("data/APPS/train")
    call_based_test = get_call_based("data/APPS/test") 
    cb_train_percentage = compute_percentage(call_based_train, "data/APPS/train")
    cb_test_percentage = compute_percentage(call_based_test, "data/APPS/test")
    print(f"# call-based in train data: {len(call_based_train)}")
    print(f"# call-based in test data: {len(call_based_test)}")
    print("Call-based percentage in train data: ", cb_train_percentage)
    print("Call-based percentage in test data: ", cb_test_percentage)
    print()

    # Print information about the number of inputs in the input_output.json file
    cb_problems_ncounts = get_number_inputs(call_based_train)
    n_count_counter = Counter(cb_problems_ncounts.values())
    print("# available inputs for ALL call-based tasks")
    for key, value in sorted(n_count_counter.items(), key=lambda x: x[1], reverse=True):
        print(f" {key}: {value}")
    print()

    # Then info about tasks without classes
    class_tasks = get_class_tasks(call_based_train)
    non_class_tasks = [task for task in call_based_train if task not in class_tasks]
    print(f"# class tasks: {len(class_tasks)}, # non-class tasks: {len(non_class_tasks)}")

    # First investigate tasks without classes
    print("---Non-class tasks---")
    nc_problems_ncounts = get_number_inputs(non_class_tasks)
    n_count_counter = Counter(nc_problems_ncounts.values())
    print("# available inputs non-class tasks")
    for key, value in sorted(n_count_counter.items(), key=lambda x: x[1], reverse=True):
        print(f" {key}: {value}")
    print()

    # Then investigate tasks with one class and one function
    print("---Class Tasks---")
    oneclassonedef = get_one_class_one_def(class_tasks)
    print(f"- {len(oneclassonedef)} tasks with one class and one function")
    other = [task for task in class_tasks if task not in oneclassonedef]
    print(f"- {len(other)} other tasks. These will not be considered at the moment")    

if __name__ == "__main__":
    main()
