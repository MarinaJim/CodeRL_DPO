import torch
from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer
from transformers import T5ForConditionalGeneration, RobertaTokenizer
import json
import gc

gc.collect()
torch.cuda.empty_cache()

def main():
    print("Starting the DPO training")
    path_to_dataset = "data/APPS/dpo_dataset_original.json"
    model_path = "/storage/athene/work/sakharova/CodeRL_DPO/models/finetuned-codet5-large-ntp-py/checkpoint-10ep-4bs-1ga"
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    with open(path_to_dataset, "r") as f:
        train_dataset = json.load(f)
        train_dataset = Dataset.from_dict(train_dataset)
    print(train_dataset)
    training_args = DPOConfig(beta=0.1, 
        output_dir="outputs/dpo_outputs/original",
        max_prompt_length=1024,
        max_length=1024,
        max_target_length=1024,
        remove_unused_columns=False,
        per_device_train_batch_size=1)
    
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    print("Initialized the DPO trainer")
    dpo_trainer.train()
    print("Trained the model")

if __name__ == "__main__":
    
    main()