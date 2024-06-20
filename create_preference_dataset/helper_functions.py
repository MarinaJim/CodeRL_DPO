import os
import sys
import json

def get_call_based(folder):
    sys.set_int_max_str_digits(0)
    call_based = []
    for task in os.listdir(folder):
        if "input_output.json" not in os.listdir(folder + "/" + task):
            continue
        with open(folder + "/" + task + "/input_output.json", "r") as f:
            in_outs = json.load(f)
            if in_outs.get("fn_name") is not None:
                call_based.append(task)
    return call_based

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

def get_one_class_one_def(tasks):
    """
    Returns tasks whose starters code has one class with one function in it.
    """
    one_class_one_def = []
    for task in tasks:
        code = os.path.join("data/APPS/train", str(task), "solutions.json")
        with open(code, "r") as f:
            solutions = json.load(f)
            for solution in solutions:
                n_class = 0
                n_def = 0
                for line in solution.split("\n"):
                    if "class " in line:
                        n_class += 1
                    if "def " in line:
                        n_def += 1
                if n_class == 1 and n_def == 1:
                    one_class_one_def.append(task)
                    break
    return one_class_one_def

def get_solution_class(tasks):
    """
    Returns tasks whose starters code has two classes, one of which is Solution.
    """
    solution_classes = []
    for task in tasks:
        code = os.path.join("data/APPS/train", str(task), "starter_code.py")
        n_classes = 0
        with open(code, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "class" in line:
                    n_classes += 1
                    class_name = line.split(" ")[1]
            if n_classes > 1:
                if "Solution" in class_name:
                    solution_classes.append(task)
    return solution_classes

def get_number_inputs(tasks, folder="data/APPS/train"):
    """
    For each call-based problem in the folder, returns number of inputs in the file input_output.json.
    """
    sys.set_int_max_str_digits(0)
    ns_inputs = {}
    for task in tasks:
        # retrieve the number of outputs
        io_path = folder + "/" + task + "/input_output.json"
        if os.path.exists(io_path):
            with open(io_path, "r") as f:
                inputs = json.load(f)
                inputs = inputs["inputs"]
                if len(inputs) < 5:
                    ns_inputs[task] = len(inputs)

                elif len(inputs) >=5 and len(inputs) <= 10:
                    ns_inputs[task] = "5-10"

                else:
                    ns_inputs[task] = "> 10"
        else:
            ns_inputs[task] = 0
    return ns_inputs

def get_tasks_with_generated_inputs(folder):
    successful_tasks = []
    all_tasks = os.listdir(folder)
    for task in all_tasks:
        path = os.path.join(folder, task, "parameters.json")
        if os.path.exists(path):
            successful_tasks.append(task)
    return successful_tasks
