import os
import shutil
import pickle
import json

tasks = "data/APPS/preference"
src_codes = "outputs/warmup_codes_for_dpo/checkpoint-1ep-4bs-1ga-2e-5/train/codes"
src_results = "outputs/warmup_codes_for_dpo/checkpoint-1ep-4bs-1ga-2e-5/train/test_results"

for task in os.listdir(tasks):
    print(task)
    task_id = str(int(task))
    code_path = os.path.join(src_codes, f"{task}.json")
    results_path = os.path.join(src_results, f"{task}.pkl")
    with open(results_path, "rb") as f:
        results = pickle.load(f)
        results = results[int(task_id)]
    with open(code_path, "r") as f:
        codes = json.load(f)
        codes = codes[task]
    formatted = {"code": codes["code"], "result": results["results"], "error_type": results["errors"]}
    with open(os.path.join(tasks, task, f"gen_solutions.json"), "w") as f:
        json.dump(formatted, f)
print("saved codes")
    
        


