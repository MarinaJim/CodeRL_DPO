import subprocess
import os
import json
import re
import shutil
from ast import literal_eval
import argparse

def main(args):
    task = args.task_path
    print(f"Task is: {task}")
    all_params = []
    path_to_code = os.path.join(task, "solution.py")
    # run crosshair to get test parameters
    try:
        parameters = subprocess.check_output(["crosshair", "cover", 
                                              "--coverage_type", "path", 
                                              "--max_uninteresting_iterations", "3", 
                                              "--per_path_timeout", "5",
                                              path_to_code], timeout=120)
        parameters = parameters.splitlines()
        if parameters == []:
            print("Crosshair did not find any parameters for task", task)
            shutil.rmtree(task)
        # check parameter options one by one
        for params in parameters:
            # print original crosshair output

            # convert bytes to string
            params = params.decode("utf-8")
            if ":=" in params or "<lambda>" in params:
                print(f"skipping: {params}")
                continue
            # get rid of ")" and method name
            params = params[:-1].split("(")
            # non-class case (there is no self parameter)
            if len(params) == 2:
                params = params[1]
            else:
                # class case
                params = params[2]
                params = params[2:]

            pattern = re.compile(r'\b\w+=([^,]+)')
            params = "[" + pattern.sub(r'\1', params) + "]"
            params = literal_eval(params)
            all_params.append(params)
        with open(os.path.join(task, "parameters.json"), "w") as f:
            json.dump(all_params, f)

    except Exception as e:
        print("Crosshair failed for task", task)
        shutil.rmtree(task)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing generated programs with unit testing")
    parser.add_argument("-t","--task_path", type=str, help="Path to the task crosshair will run for")
    args = parser.parse_args()
    
    main(args)

