import os
import json
import re
import shutil

from helper_functions import (get_call_based, 
                              get_class_tasks, 
                              get_one_class_one_def, 
                              get_solution_class, 
                              get_number_inputs)

REPLACEMENT_PATTERN = r'\bdef\s+\w+\(.*?\):'

def save_1c1d_annotated(one_class_one_def_tasks):
    for task in one_class_one_def_tasks:
        # get a solution code
        solution_path = os.path.join("data/APPS/train", str(task), "solutions.json")
        with open(solution_path, "r") as f:
            solutions = json.load(f)
        solution = solutions[0]
        solution = solution.split("\n")

        # get method head from the starter code
        starter_code_path = os.path.join("data/APPS/train", str(task), "starter_code.py")
        with open(starter_code_path, "r") as f:
            starter_code = f.readlines()
            for line in starter_code:
                if "def " in line:
                    method_head = line.strip()
                    break
        for index, line in enumerate(solution):
            if "def " in line:
                solution[index] = re.sub(REPLACEMENT_PATTERN, method_head, solution[index])
                break
        
        path_to_annotated_solution = os.path.join(path_to_preference, str(task))
        if not os.path.exists(path_to_annotated_solution):
            os.mkdir(path_to_annotated_solution)
        
        code = "import sys\nimport time\nimport itertools\nfrom itertools import accumulate, product, permutations, combinations\nimport collections\nfrom collections import Counter, OrderedDict, deque, defaultdict, ChainMap\nfrom functools import lru_cache\nimport math\nfrom math import sqrt, sin, cos, tan, ceil, fabs, floor, gcd, exp, log, log2\nimport fractions\nfrom typing import List, Tuple\nimport numpy as np\nimport random\nimport heapq\nfrom heapq import *\n\n"
        for line in solution:
            code += line + "\n"

        with open(os.path.join(path_to_annotated_solution, "solution.py"), "w") as f:
            f.write(code)

"""
save_1c1d_annotated(one_class_one_def_tasks)
"""

def save_non_class_annotated(non_class_tasks):
    for task in non_class_tasks:
        # load one sample solution
        solution_path = os.path.join("data/APPS/train", str(task), "solutions.json")
        with open(solution_path, "r") as f:
            solutions = json.load(f)
        solution = ""
        for sol in solutions:
            if sol.count("def ") == 1:
                solution = sol
                break
        if solution == "":
            continue
    
        # create a directory for the task
        path_to_task = os.path.join(path_to_preference, str(task))
        if not os.path.exists(path_to_task):
            os.mkdir(path_to_task)

        # write the solution so that it can be annotated
        with open(os.path.join(path_to_task, "solution.py"), "w") as f:
            f.write(solution)

        # get inputs and outputs for the code
        io_path = os.path.join("data/APPS/train", str(task), "input_output.json")
        with open(io_path) as f:
            in_outs = json.load(f)

        # write a function call to a file to run monkeytype over it later
        fn_name = in_outs["fn_name"]
        sample_input = in_outs["inputs"][0]
        sample_call = f"from solution import {fn_name}\n"
        sample_input = [repr(sample) for sample in sample_input]
        sample_input_str = ", ".join(sample_input)
        sample_call += f"{fn_name}({sample_input_str})"
        with open(os.path.join(path_to_task, "sample_call.py"), "w") as f:
            f.write(sample_call)
    

path_to_preference = "data/APPS/preference"
if not os.path.exists(path_to_preference):
    os.mkdir(path_to_preference)
    
call_based_tasks = get_call_based("data/APPS/train")
class_tasks = get_class_tasks(call_based_tasks)

# Below, we define two main types of tasks:
# 1) non-class-based
# 2) class-based with one class and one function

one_class_one_def_tasks = get_one_class_one_def(class_tasks)

# Get non-class tasks that have at least one test input
non_class_tasks = [task for task in call_based_tasks if task not in class_tasks]
n_inputs_not_class_tasks = get_number_inputs(non_class_tasks)
non_class_tasks = [task for task in non_class_tasks if n_inputs_not_class_tasks[task] != 0]

save_1c1d_annotated(one_class_one_def_tasks)
save_non_class_annotated(non_class_tasks)