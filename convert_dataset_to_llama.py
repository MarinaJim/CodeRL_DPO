import os
import json
import argparse

def main(args):
    with open(args.path_to_dataset, "r") as f:
        dataset = json.load(f)
    llama_dataset = []
    for (prompt, chosen, rejected) in zip(dataset["prompt"], dataset["chosen"], dataset["rejected"]):
        entry = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        llama_dataset.append(entry)
    with open(args.path_to_llama_dataset, "w") as f:
        json.dump(llama_dataset, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset")
    parser.add_argument("--path_to_llama_dataset")
    args = parser.parse_args()
    main(args)