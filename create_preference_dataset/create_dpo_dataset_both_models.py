from create_dpo_dataset import create_dpo_dataset_dict
import random
import argparse
import json

def create_dpo_dataset_dict_both_models(path_to_codes_1, path_to_codes_2,
                                        path_to_results_1, path_to_results_2,
                                        path_to_apps, samples_per_task,
                                        max_len, best_threshold,
                                        worst_threshold):
    random.seed(940542582)
    dataset_dict_1 = create_dpo_dataset_dict(path_to_codes_1,
                                                path_to_results_1,
                                                path_to_apps,
                                                samples_per_task,
                                                max_len,
                                                best_threshold,
                                                worst_threshold)
    print(list(dataset_dict_1.values()))
    print("length:")
    print(len(list(dataset_dict_1.values())))
    print()
    print(dataset_dict_1)
    dataset_dict_2 = create_dpo_dataset_dict(path_to_codes_2,
                                                path_to_results_2,
                                                path_to_apps,
                                                samples_per_task,
                                                max_len,
                                                best_threshold,
                                                worst_threshold)


    for prompt, chosen, rejected in zip(*dataset_dict_1.values()):
        if prompt in dataset_dict_2["prompt"]:
            # same as selecting a random boolean
            if random.random() < 0.5:
                continue
            index_prompt = dataset_dict_2["prompt"].index(prompt)
            dataset_dict_2["prompt"].remove(prompt)
            dataset_dict_2["chosen"].pop(index_prompt)
            dataset_dict_2["rejected"].pop(index_prompt)


        dataset_dict_2["prompt"].append(prompt)
        dataset_dict_2["chosen"].append(chosen)
        dataset_dict_2["rejected"].append(rejected)
    
    
    return dataset_dict_2


def main(args):
    dataset = create_dpo_dataset_dict_both_models(args.path_to_codes_1, args.path_to_codes_2,
                                        args.path_to_results_1, args.path_to_results_2, 
                                        args.path_to_apps, args.samples_per_task,
                                        args.max_len,
                                        args.best_threshold,
                                        args.worst_threshold)
    print(len(dataset["prompt"]))
    print(len(dataset["chosen"]))
    with open(args.path_to_dpo, "w") as f:
        json.dump(dataset, f)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_codes_1", type=str, help="Path to codes of the first model")
    parser.add_argument("--path_to_codes_2", type=str, help="Path to codes of the second model")
    parser.add_argument("--path_to_results_1", type=str, help="Path to test results of the first model")
    parser.add_argument("--path_to_results_2", type=str, help="Path to test results of the second model")
    parser.add_argument("--path_to_dpo", type=str, help="path to future dpo dataset")

    parser.add_argument("-pd", "--path_to_apps", help="Path to APPS dataset")
    parser.add_argument("-bt", "--best_threshold", type=float, help="Minimum threshold for the best code. Value from -2 to 1")
    parser.add_argument("-wt", "--worst_threshold", type=float, default=-2, help="Maximum threshold for the worst code. Value from -2 to 1")
    parser.add_argument("--max_len", type=int, help="Length of the DPO dataset")
    parser.add_argument("--samples_per_task", type=int, help="How many examples to take per one task")
    args = parser.parse_args()
    main(args)