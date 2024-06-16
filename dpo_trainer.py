import torch
from trl import DPOConfig, DPOTrainer
from transformers import T5ForConditionalGeneration, RobertaTokenizer
import json

def main():
    path_to_dataset = "data/APPS/dpo_dataset.json"
    model_path = "exps/finetuned-codet5-large-ntp-py/checkpoint-146000"
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    with open(path_to_dataset, "r") as f:
        train_dataset = json.load(f)
    print(type(train_dataset))
    training_args = DPOConfig(beta=0.1, 
        output_dir="outputs/dpo_outputs")
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
    )
    dpo_trainer.train()

if __name__ == "__main__":
    
    main()