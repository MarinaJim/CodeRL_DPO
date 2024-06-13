import os
import json
import subprocess
from ast import literal_eval
import sys
sys.set_int_max_str_digits(0)

def save_outputs_call_based(parameters, fn_name, path_to_task, task_id):
    monkeytype_io = {}
    inputs = []
    outputs = []

    for params in parameters:
        call = f"from solution import {fn_name}\n"
        sample_input = [repr(sample) for sample in params]
        sample_input_str = ", ".join(sample_input)
        call += "print(" + f"{fn_name}({sample_input_str})" + ")"
        with open(os.path.join(path_to_task, "call.py"), "w") as f:
            f.write(call)

        try:
            output = subprocess.check_output(["python", f"{path_to_task}/call.py"])
            output = literal_eval(output.decode("utf-8"))
            inputs.append(params)
            outputs.append(output)
        except Exception as e:
            print("Exception was raised for ", task_id)
    monkeytype_io["inputs"] = inputs
    monkeytype_io["outputs"] = outputs

    with open(os.path.join(path_to_task, "monkeytype_io.json"), "w") as f:
        json.dump(monkeytype_io, f)


def save_outputs_std_output(parameters, fn_name, path_to_task, task_id):
    monkeytype_io = {}
    inputs = []
    outputs = []

    for params in parameters:
        call = f"from solution import Solution\n"
        call += "solution_object = Solution()\n"
        sample_input = [repr(sample) for sample in params]
        sample_input_str = ", ".join(sample_input)
        call += "print(" + f"solution_object.{fn_name}({sample_input_str})" + ")"
        with open(os.path.join(path_to_task, "call.py"), "w") as f:
            f.write(call)

        try:
            output = subprocess.check_output(["python", f"{path_to_task}/call.py"])
            output = literal_eval(output.decode("utf-8"))
            inputs.append(params)
            outputs.append(output)
        except Exception as e:
            print("Exception was raised for ", task_id)
            
    monkeytype_io["inputs"] = inputs
    monkeytype_io["outputs"] = outputs

    with open(os.path.join(path_to_task, "monkeytype_io.json"), "w") as f:
        json.dump(monkeytype_io, f)

def create_outputs_for_monkeytype_inputs(path_to_preference):
    tasks = os.listdir(path_to_preference)
    for task in tasks:
        if not os.path.exists(os.path.join(path_to_preference, task, "monkeytype_io.json")):
            print(task)
        else:
            print("exists")
            
        continue
        path_to_task = os.path.join(path_to_preference, str(task))
        parameters_path = os.path.join(path_to_task, "parameters.json")
        with open(parameters_path) as f:
            parameters = json.load(f)

        io_path = os.path.join("data/APPS/train", str(task), "input_output.json")
        with open(io_path) as f:
            in_outs = json.load(f)
        if "fn_name" not in in_outs:
            print(f"No function name for {task}")

        fn_name = in_outs["fn_name"]

        if os.path.exists(os.path.join(path_to_task, "sample_call.py")):
            save_outputs_call_based(parameters, fn_name, path_to_task, task)
        else:
            save_outputs_std_output(parameters, fn_name, path_to_task, task)
            
if __name__ == "__main__":
    path_to_preference = "data/APPS/preference"
    create_outputs_for_monkeytype_inputs(path_to_preference)