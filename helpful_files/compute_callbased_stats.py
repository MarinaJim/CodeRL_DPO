"""
Outputs percentages of call-based tasks in the train and test data respectively.
"""
import os
import json
from collections import Counter
import sys

def get_call_based(folder):
    call_based = []
    for task in os.listdir(folder):
        if "starter_code.py" not in os.listdir(folder + "/" + task):
            continue
        with open(folder + "/" + task + "/starter_code.py", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "def " in line:
                    call_based.append(task)
                    break
    return call_based

def get_number_inputs(folder):
    """
    For each call-based problem in the folder, returns number of inputs in the file input_output.json.
    """
    sys.set_int_max_str_digits(0)
    ns_inputs = {}
    for task in os.listdir(folder):
        if "starter_code.py" not in os.listdir(folder + "/" + task):
            continue

        # make sure the problem is call-based
        call_based = False
        with open(folder + "/" + task + "/starter_code.py", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "def " in line:
                    call_based = True
                    break
        if not call_based:
            continue
        
        # retrieve the number of outputs
        try:
            with open(folder + "/" + task + "/input_output.json", "r") as f:
                inputs = json.load(f)
                inputs = inputs["inputs"]
                if len(inputs) < 10:
                    ns_inputs[task] = len(inputs)
                else:
                    ns_inputs[task] = "10+"
        except FileNotFoundError as e:
            ns_inputs[task] = 0
    return ns_inputs

def get_input_output_not_available(folder):
    """
    Returns ids of all problems for which the input_output.json file is not available.
    """
    sys.set_int_max_str_digits(0)
    zero_input = []
    for task in os.listdir(folder):
        if "starter_code.py" not in os.listdir(folder + "/" + task):
            continue
        path = folder + "/" + task + "/input_output.json"
        if(not os.path.exists(path)):
            zero_input.append(task)
    return zero_input


def get_cluster(problems):
    """
    Often the problems with similar structure are grouped together in ca. 100er groups.
    This method returns the groups of the given problems.
    """
    problems = [problem[:2] for problem in problems]
    return Counter(problems)

def get_class_tasks(tasks, folder="data/APPS/train"):
    """
    Returns a list of problems which starters code contains a class.
    """
    class_tasks = []
    for task in tasks:
        if "starter_code.py" not in os.listdir(folder + "/" + task):
            continue
        with open(folder + "/" + task + "/starter_code.py", "r") as f:
            lines = f.readlines()
            for line in lines:
                if "class " in line:
                    class_tasks.append(task)
                    break
    return class_tasks

def get_two_plus_classes(tasks, folder="data/APPS/train"):
    """
    Returns tasks that have more than one class in the starter code.
    """
    tasks = os.listdir(folder)
    class_tasks = get_class_tasks(tasks)
    more_than_two = []
    for task in class_tasks:
        code = os.path.join(folder, str(task), "starter_code.py")
        n_class = 0
        with open(code, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "class" in line:
                    n_class += 1
            if n_class > 1:
                more_than_two.append(task)
    return more_than_two

def compute_percentage(tasks, folder):
    total = os.listdir(folder)
    return len(tasks) / len(total)

def main():
    # First print general information about the dataset
    print("---General stats---")
    call_based_train = get_call_based("data/APPS/train")
    call_based_test = get_call_based("data/APPS/test") 
    cb_train_percentage = compute_percentage(call_based_train, "data/APPS/train")
    cb_test_percentage = compute_percentage(call_based_test, "data/APPS/test")
    print("Call-based percentage in train data: ", cb_train_percentage)
    print("Call-based percentage in test data: ", cb_test_percentage)
    print()

    # Print information about the number of inputs in the input_output.json file
    cb_problems_ncounts = get_number_inputs("data/APPS/train")
    n_count_counter = Counter(cb_problems_ncounts.values())
    print("Number of inputs in input_output.json")
    for key, value in sorted(n_count_counter.items(), key=lambda x: x[1], reverse=True):
        print(f" {key}: {value}")
    print()

    # Go in-depth for problems without file input_output.json
    io_not_available = get_input_output_not_available("data/APPS/train")
    io_not_available_cluster = get_cluster(io_not_available)
    n_class_no_io = len(get_class_tasks(io_not_available))
    print("---Tasks without input_output file---")
    print(f"Amount: {len(io_not_available)}, class-tasks percentage: {n_class_no_io / len(io_not_available)}")
    print()

    # Go in-depth for problems that do have input_output.json file, but len(input) is zero.
    print("---Tasks with 0 length of input array---")
    other = [problem for problem in cb_problems_ncounts.keys() if problem not in io_not_available and cb_problems_ncounts[problem] == 0]
    other_cluster = get_cluster(other)
    print(f"Amount: {len(other)}")
    print()

    print("---Tasks with classes---")
    class_problems = list(get_class_tasks(cb_problems_ncounts.keys()))
    print(f"Out of {len(cb_problems_ncounts.keys())} call-based problems, {len(class_problems)} contain classes, or {len(class_problems) / len(cb_problems_ncounts.keys())}%")
    two_plus_classes_problems = get_two_plus_classes(cb_problems_ncounts.keys())
    print(f"From all class-based problems, {len(two_plus_classes_problems)}, or {len(two_plus_classes_problems) / len(cb_problems_ncounts.keys())} contain more than one class.")
if __name__ == "__main__":
    main()
