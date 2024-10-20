import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import pickle


def get_difficulties_for_folder(model_dir, task_folder):
    results_dir = os.path.join(model_dir, "test_results")
    results = sorted(os.listdir(results_dir))
    errors = 0
    difficulties =  {"introductory": [], "interview": [], "competition": [], "total": []}
    for task in results:
        path = f"{results_dir}/{task}"
        task = task.replace(".pkl", "")
        task = "0"*(4 - len(task)) + str(task)
        with open(os.path.join(task_folder, task, "metadata.json"), "r") as f:
            data = json.load(f)
            difficulty = data["difficulty"]
        try:
            with open(path, "rb") as f:
                output = pickle.load(f)
            index = list(output.keys())[0]
            scores = output[index]["results"]
            difficulties[difficulty].extend(scores)
            difficulties["total"].extend(scores)
        except  EOFError:
            errors += 1
    return difficulties

def plot_difficulties(plot_folder, difficulties):

    for model, total in zip(difficulties["method"], difficulties["total"]):
        plot_path = os.path.join(plot_folder, f"{model}.jpg")
        plt.hist([output for output in total], color="steelblue", edgecolor="black")
        plt.savefig(plot_path)
        plt.clf()

def get_mean_values(all_results):

    for key in ["introductory", "interview", "competition", "total"]:
        mean_values = []
        for k in all_results[key]:
            mean_values.append(sum(k) / len(k))
        
        all_results[key] = mean_values
    return all_results

def get_difficulties_for_all_folders_rl(folder, task_folder):
    
    all_results = {"critic": [], "epochs": [], "lr":[], "dataset":  [], "introductory": [], "interview": [], "competition": [], "total": []}
    for model in os.listdir(folder):
        difficulties = get_difficulties_for_folder(os.path.join(folder, model), task_folder)

        
        model = model.split("_")
        all_results["epochs"].append(int(model[0].replace("ep", "")))
        all_results["lr"].append(model[1])
        all_results["critic"].append(model[2])
        all_results["dataset"].append(model[3])


        for key in difficulties.keys():
            #mean_score = sum(difficulties[key]) / len(difficulties[key])
            all_results[key].append(difficulties[key])
    return all_results

def get_difficulties_for_all_folders_dpo(folder, task_folder):
    
    all_results = {"method": [], "epochs": [], "samples":[],"lr":[], "beta":[], "dataset":  [], "introductory": [], "interview": [], "competition": [], "total": []}
    for model in os.listdir(folder):
        difficulties = get_difficulties_for_folder(os.path.join(folder, model), task_folder)

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
        else:
            continue

        for key in difficulties.keys():
            #mean_score = sum(difficulties[key]) / len(difficulties[key])
            all_results[key].append(difficulties[key])
    return all_results

def save_for_folder(results, model_name):

    df = pd.DataFrame(results)
    df = df.round(4)
    df = df.sort_values(by=['total'], ascending=False)
    df.to_csv(os.path.join(f"outputs/results_for_presentation/evaluation_metrics/mean_score/{model_name}.csv"), index=False)



def main():
    task_folder = "data/APPS/test"

    for model_name in ["codet5-actor"]:
        if model_name == "codet5-actor":
            difficulties = get_difficulties_for_all_folders_rl(f"outputs/results_for_presentation/{model_name}", task_folder)
        else:   
            difficulties = get_difficulties_for_all_folders_dpo(f"outputs/results_for_presentation/{model_name}", task_folder)
        #plot_difficulties("/storage/athene/work/sakharova/CodeRL_DPO/outputs/results_for_presentation/evaluation_metrics/mean_score_distribution", difficulties)
        difficulties = get_mean_values(difficulties)
        save_for_folder(difficulties, model_name)

if __name__ == "__main__":
    main()