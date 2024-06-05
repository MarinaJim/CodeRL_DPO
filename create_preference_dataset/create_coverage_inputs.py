import os
import json
from helper_functions import get_call_based, get_class_tasks, get_one_class_one_def, get_solution_class


call_based_tasks = get_call_based("data/APPS/train")
class_tasks = get_class_tasks(call_based_tasks)

# Below, we define three main types of tasks:
# 1) non-class-based
# 2) class-based with one class and one function
# 3) class-based with two classes, one of which is Solution

not_class_tasks = [task for task in call_based_tasks if task not in class_tasks]
one_class_one_def_tasks = get_one_class_one_def(class_tasks)
solution_class_tasks = get_solution_class(class_tasks)
print(len(solution_class_tasks))