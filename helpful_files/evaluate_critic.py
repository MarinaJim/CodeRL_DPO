import os
import pickle
import pandas as pd

def get_scores_for_one_code(file, scores_name):
    with open(file, "rb") as f:
        return pickle.load(f)[scores_name]


def get_scores_for_model(folder, scores_name):
    predictions = []
    for file in os.listdir(folder):
        pred = get_scores_for_one_code(os.path.join(folder, file), scores_name)
        predictions.append(pred)
    return predictions

def compute_mae(gt, pred):
    mae = 0
    for gt_answers, pred_answers in zip(gt, pred):
        for gt_answer, pred_answer in zip(gt_answers, pred_answers):
                mae += abs(gt_answer - pred_answer)
    
    return mae / sum(len(answer) for answer in gt)


def compute_accuracy(gt, pred):
    correct = 0
    for gt_answers, pred_answers in zip(gt, pred):
        for gt_answer, pred_answer in zip(gt_answers, pred_answers):
            if gt_answer == pred_answer:
                correct += 1
    return correct / sum(len(x) for x in gt)
    
def get_all_metrics(folder):
    gt_error_types = get_scores_for_model(os.path.join(folder, os.listdir(folder)[0]), 
                                            "gt_error_type")
    all_predictions = {}
    for model_folder in os.listdir(folder):
        model_predictions = get_scores_for_model(os.path.join(folder, model_folder), "pred_error_type")
        all_predictions[model_folder] = model_predictions

    accuracies = {}
    maes = {}
    for key, values in all_predictions.items():
        accuracy = compute_accuracy(gt_error_types, values)
        accuracies[key] = accuracy

        mae = compute_mae(gt_error_types, values)
        maes[key] = mae
    

    return accuracies, maes
    
def get_metrics_df(accuracies, maes):
        exp_names = accuracies.keys()
        dataset = ["SE" if "-se-" in name else "SE + GT" if "-all-" in name else "-" for name in exp_names]
        model_names = ["CodeT5-base" if "-base-" in name else "CodeT5-finetuned-critic" for name in exp_names]
        df = pd.DataFrame(
            {
            'Model': model_names,
            'Dataset': dataset,
            'Accuracy': list(accuracies.values()),
            'MAE': list(maes.values())
            }
        )
        df = df.sort_values(by=["MAE"])
        return df

def main():
    folder = "outputs/results_for_presentation/codet5-critic/test/"
    accuracies, maes = get_all_metrics(folder)
    df = get_metrics_df(accuracies, maes)
    df.to_csv("outputs/results_for_presentation/codet5-critic/evaluation_results/test_results.csv", index=False)

    folder = "outputs/results_for_presentation/codet5-critic/train/"
    accuracies, maes = get_all_metrics(folder)
    df = get_metrics_df(accuracies, maes)
    df.to_csv("outputs/results_for_presentation/codet5-critic/evaluation_results/train_results.csv", index=False)
    

if __name__ == "__main__":
    main()