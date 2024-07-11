import os
import json
import subprocess
from ast import literal_eval
import shutil
import argparse
import sys
sys.set_int_max_str_digits(0)

def save_outputs_call_based(parameters, fn_name, path_to_task, task_id):
    monkeytype_io = {}
    inputs = []
    outputs = []

    for params in parameters:
        call = "import json\n"
        call += f"from solution import {fn_name}\n"
        sample_input = [repr(sample) for sample in params]
        sample_input_str = ", ".join(sample_input)
        call += f"output = {fn_name}({sample_input_str})\n"
        call += f"with open('{path_to_task}/method_output.json', 'w') as f:\n  json.dump(output, f)"
        with open(os.path.join(path_to_task, "call.py"), "w") as f:
            f.write(call)
        try:
            subprocess.call(["python", f"{path_to_task}/call.py"])
            with open(os.path.join(path_to_task, "method_output.json")) as f:
                output = json.load(f)
            inputs.append(params)
            outputs.append(output)

        except Exception as e:
            print("Exception was raised for ", task_id)
            print(f"Exception: {e}")
            print()
    if outputs == []:
        shutil.rmtree(path_to_task)

    monkeytype_io["fn_name"] = fn_name
    monkeytype_io["inputs"] = inputs
    monkeytype_io["outputs"] = outputs

    with open(os.path.join(path_to_task, "input_output.json"), "w") as f:
        json.dump(monkeytype_io, f)


def save_outputs_class_based(parameters, fn_name, path_to_task, task_id):
    monkeytype_io = {}
    inputs = []
    outputs = []

    for params in parameters:
        call = "import json\n"
        call += f"from solution import Solution\n"
        call += "solution_object = Solution()\n"
        sample_input = [repr(sample) for sample in params]
        sample_input_str = ", ".join(sample_input)
        call += f"output = solution_object.{fn_name}({sample_input_str})\n"
        call += f"with open('{path_to_task}/method_output.json', 'w') as f:\n  json.dump(output, f)"
        with open(os.path.join(path_to_task, "call.py"), "w") as f:
            f.write(call)
        try:
            subprocess.call(["python", f"{path_to_task}/call.py"])
            with open(os.path.join(path_to_task, "method_output.json")) as f:
                output = json.load(f)
            inputs.append(params)
            outputs.append(output)

        except Exception as e:
            print("Exception was raised for ", task_id)
            print(f"Exception: {e}")
            print()
    if outputs == []:
        shutil.rmtree(path_to_task)
    
    monkeytype_io["inputs"] = inputs
    monkeytype_io["outputs"] = outputs
    monkeytype_io["fn_name"] = fn_name

    with open(os.path.join(path_to_task, "input_output.json"), "w") as f:
        json.dump(monkeytype_io, f)

def create_outputs_for_monkeytype_inputs(path_to_task):
        if not os.path.exists(path_to_task):
            return
        task = path_to_task.split("/")[-1]
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
            save_outputs_class_based(parameters, fn_name, path_to_task, task)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing generated programs with unit testing")
    parser.add_argument("-t","--task_path", type=str, help="Path to the task crosshair will run for")
    args = parser.parse_args()
    
    create_outputs_for_monkeytype_inputs(args.task_path)