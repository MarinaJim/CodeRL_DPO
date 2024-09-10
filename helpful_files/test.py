import os
import shutil

tasks = "data/APPS/preference"
src_codes = "outputs/warmup_codes_for_dpo/checkpoint-1ep-4bs-1ga-2e-5/train/codes"
src_results = "outputs/warmup_codes_for_dpo/checkpoint-1ep-4bs-1ga-2e-5/train/test_results"

for task in os.listdir(tasks):
    task_id = str(int(task))
    code_path = os.path.join(src_codes, f"{task}.json")
    results_path = os.path.join(src_results, f"{task}.pkl")
    

