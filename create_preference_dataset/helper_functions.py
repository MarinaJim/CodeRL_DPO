import os

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
        code = os.path.join("data/APPS/train", str(task), "starter_code.py")
        n_def = 0
        n_class = 0
        with open(code, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "class " in line:
                    n_class += 1
                if "def " in line:
                    n_def += 1
            if n_class == 1 and n_def == 1:
                one_class_one_def.append(task)
    return one_class_one_def

def get_solution_class(tasks):
    """
    Returns tasks whose starters code has two classes, one of which is Solution.
    """
    solution_classes = []
    for task in tasks:
        code = os.path.join("data/APPS/train", str(task), "starter_code.py")
        n_def = 0
        with open(code, "r") as f:
            lines = f.readlines()
            for line in lines:
                if "class Solution" in line:
                    solution_classes.append(task)
                    break
    return solution_classes