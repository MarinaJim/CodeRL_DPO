import os
import pickle
import random
import json

def sample_chosen_rejected(output):
    """
    Returns a dictionary of the following form:
    {
     "chosen": "*code with the highest score*",
     "rejected": "*a random code that does not have the highest score*"
    }
    """
    results_and_sols = []
    for i, (result, sol) in enumerate(zip(output["results"], output["sols"])):
        results_and_sols.append({"results": result, "code": sol})

    results_and_sols.sort(key=lambda x: x["results"], reverse=True)
    chosen = results_and_sols[0]["code"]
    rejected = random.choice(results_and_sols[1:])["code"]
    return {"chosen": chosen, "rejected": rejected}


def create_dpo_dataset_dict(path_to_preference):
    """
    Creates a preference dataset in the DPO-conform format.
    """
    prompts = []
    chosen_output = []
    rejected_output = []

    for problem in os.listdir(path_to_preference):
        try:
            with open(
                os.path.join(path_to_preference, problem, "question.txt"),
                "r",
                encoding="utf-8",
            ) as f:
                prompt = f.read()
                
            problem = str.lstrip(problem, "0")
            with open(
                os.path.join("outputs/test_results", f"{problem}.pkl"),
                "rb"
            ) as f:
                output = pickle.load(f)
                output = output[int(problem)]
                sample = sample_chosen_rejected(output)
                chosen_output.append(sample["chosen"])
                rejected_output.append(sample["rejected"])
                prompts.append(prompt)
        except Exception as e:
            print(f"Exception occured for {problem}: {e}")
            
    dpo_dataset_dict = {
        "prompt": prompts,
        "chosen": chosen_output,
        "rejected": rejected_output,
    }
    return dpo_dataset_dict


def main():
    """
    Creates a DPO dataset and saves it into a file.
    """
    path_to_dataset = "data/APPS/preference"
    dpo_dataset_dict = create_dpo_dataset_dict(path_to_dataset)
    with open("data/APPS/dpo_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dpo_dataset_dict, f)


if __name__ == "__main__":
    main()
