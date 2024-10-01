import os
import shutil
import pickle
import json

test = "data/APPS/test"
src_codes = "outputs/results_for_presentation/codet5/sft_1ep/codes"
src_results = "outputs/results_for_presentation/codet5/sft_1ep/test_results"
critic_path = "data/APPS/test"
# if  true, use only symbolic execution data for training. Else use symbolic execution data + ground truth results for all tasks
se_only = False

for task in os.listdir(test):
    task_id = str(int(task))
    code_path = os.path.join(src_codes, f"{task_id}.json")
    results_path = os.path.join(src_results, f"{task_id}.pkl")
    if not os.path.exists(results_path):
        continue
    with open(results_path, "rb") as f:
        try:
            results = pickle.load(f)
        except:
            print("Cant load for", task)
            continue
        results = results[int(task_id)]
        
        arr_of_err = []
        for errors in results["errors"]:
            if all([error is None for error in errors]):
                arr_of_err.append("None")
            else:
                found_error = False
                for err in errors:
                    if isinstance(err, tuple):
                        arr_of_err.append(err[0].__class__.__name__)
                        found_error = True
                        break
                if not found_error:
                    print("error!")
        results["errors"] = arr_of_err
    with open(code_path, "r") as f:
        codes = json.load(f)
        codes = codes[task_id]
    arr = []
    for (code, result, error) in zip(codes["code"], results["results"], results["errors"]):
        if result >= 0 and result < 1:
            result = 0
        arr.append({"code": code, "result": result, "error_type": error})
    with open(os.path.join(critic_path, task, f"gen_solutions.json"), "w") as f:
        json.dump(arr, f)

print("saved codes")

    
        


