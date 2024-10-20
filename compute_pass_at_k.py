import pandas as pd
import json
import os
import numpy as np
import pickle
import sys

def pass_at_k(n, c, k):
    """
    n: total #samples
    c: #correct samples
    k: k
    """
    if n - c < k: return 1.0
    pass_atk= 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    return pass_atk 

def compute_for_folder(task_folder, score_folder, k):
    n_errors = 0
    difficulties = {}
    for task in sorted(os.listdir(score_folder)):
        path = f"{score_folder}/{task}"
        try:
            with open(path, "r") as f:
                output = json.load(f)
        except IndexError:
            n_errors += 1
            continue

        index = list(output.keys())[0]
        scores = [score if score == 1 else 0 for score in output[index]["results"]]

        task = task.replace(".json", "")
        task = "0"*(4 - len(task)) + str(task)
        with open(os.path.join(task_folder, task, "metadata.json"), "r") as f:
            data = json.load(f)
            difficulty = data["difficulty"]
        if difficulty not in difficulties.keys():
            difficulties[difficulty] = scores
        else:
            difficulties[difficulty].extend(scores)

    for difficulty, scores in difficulties.items():
        n = len(scores)
        c = sum(scores)
        difficulties[difficulty] = pass_at_k(n, c, k)

    return difficulties, n_errors


def compute_for_folder_with_pkl(task_folder, score_folder, k):
    n_errors = 0
    difficulties = {"total": []}

    for task in sorted(os.listdir(score_folder)):
        path = f"{score_folder}/{task}"
        try:
            with open(path, "rb") as f:
                output = pickle.load(f)
        except EOFError:
            n_errors += 1
            continue

        index = list(output.keys())[0]
        scores = [score if score == 1 else 0 for score in output[index]["results"]]

        task = task.replace(".pkl", "")
        task = "0"*(4 - len(task)) + str(task)
        with open(os.path.join(task_folder, task, "metadata.json"), "r") as f:
            data = json.load(f)
            difficulty = data["difficulty"]
        if difficulty not in difficulties.keys():
            difficulties[difficulty] = [scores]
        else:
            difficulties[difficulty].append(scores)
        difficulties["total"].append(scores)

    for difficulty, scores in difficulties.items():
        pass_at_k_arr = []
        for score in scores:
            n = len(score)
            c = sum(score)
            pass_at_5 = pass_at_k(n, c, 1)
            pass_at_k_arr.append(pass_at_5)
        difficulties[difficulty] = sum(pass_at_k_arr) / len(pass_at_k_arr)

    return difficulties, n_errors

def compute_for_base_model(folder):
    task_folder = "data/APPS/test"
    all_results = {"model": [], "introductory": [], "interview": [], "competition": [], "total": []}
    for model in os.listdir(folder):

        score_folder = os.path.join(folder, model, "test_results")
        
        scores, n_errors = compute_for_folder_with_pkl(task_folder, score_folder, 5)
        model = model.replace("sft_1ep_", "")
        all_results["model"].append(model)
        for key in scores.keys():
            all_results[key].append(scores[key])

    return all_results

def compute_and_save_for_rl(folder, model_name, k):
    task_folder = "data/APPS/test"
    all_results = {"critic": [], "epochs": [], "lr":[], "dataset":  [], "introductory": [], "interview": [], "competition": [], "total": []}
    for model in os.listdir(folder):
        score_folder = os.path.join(folder, model, "test_results")
        if not os.path.exists(score_folder):
            continue
        scores, n_errors = compute_for_folder_with_pkl(task_folder, score_folder, k)
        model = model.split("_")
        all_results["epochs"].append(int(model[0].replace("ep", "")))
        all_results["lr"].append(model[1])
        all_results["critic"].append(model[2])
        all_results["dataset"].append(model[3])

        for key in scores.keys():
            all_results[key].append(scores[key])

    df = pd.DataFrame(all_results)
    df = df.round(4)
    df = df.sort_values(by=['total'], ascending=False)
    df.to_csv(os.path.join(f"outputs/results_for_presentation/evaluation_metrics/pass_at_k/pass_at_{k}_{model_name}.csv"), index=False)


def compute_and_save_for_dpo(folder, model_name, k):
    task_folder = "data/APPS/test"
    all_results = {"method": [], "epochs": [], "samples":[],"lr":[], "beta":[], "dataset":  [], "introductory": [], "interview": [], "competition": [], "total": []}
    for model in os.listdir(folder):

        score_folder = os.path.join(folder, model, "test_results")
        if not os.path.exists(score_folder):
            continue
        scores, n_errors = compute_for_folder_with_pkl(task_folder, score_folder, k)
        if model not in ["original", "sft_1ep", "sft_10ep"]:
            model = model.split("_")
            all_results["method"].append(model[2])
            all_results["epochs"].append(int(model[3].replace("ep", "")))
            all_results["samples"].append(model[4])
            all_results["lr"].append(model[5])
            all_results["beta"].append(model[6])
            all_results["dataset"].append(model[7])
        elif model == "sft_1ep":
            all_results["method"].append("sft")
            all_results["epochs"].append(" ")
            all_results["samples"].append(" ")
            all_results["lr"].append(" ")
            all_results["beta"].append(" ")
            all_results["dataset"].append(" ")
        elif model == "original":
            all_results["method"].append("original")
            all_results["epochs"].append(" ")
            all_results["samples"].append(" ")
            all_results["lr"].append(" ")
            all_results["beta"].append(" ")
            all_results["dataset"].append(" ")
        else:
            continue
        for key in scores.keys():
            all_results[key].append(scores[key])

    df = pd.DataFrame(all_results)
    df = df.round(4)
    df = df.sort_values(by=['total'], ascending=False)
    df.to_csv(os.path.join(f"outputs/results_for_presentation/evaluation_metrics/pass_at_k/pass_at_{k}_{model_name}.csv"), index=False)

    

def main():
    k = 5
    for model_name in ["llama", "codet5"]:
        compute_and_save_for_dpo(f"/storage/athene/work/sakharova/CodeRL_DPO/outputs/results_for_presentation/{model_name}", model_name, k)
    model_name = "codet5-actor"
    compute_and_save_for_rl(f"/storage/athene/work/sakharova/CodeRL_DPO/outputs/results_for_presentation/{model_name}", model_name, k)

if __name__ == "__main__":
    
    main()
    