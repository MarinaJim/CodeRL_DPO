import torch
from datasets import load_dataset, Dataset
from trl import DPOConfig, DPOTrainer
from transformers import T5ForConditionalGeneration, AutoTokenizer, AutoModelForCausalLM
import json
import gc
import argparse

def main(args):
    path_to_dataset = args.path_to_dataset
    model_path = args.model_path
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    with open(path_to_dataset, "r") as f:
        train_dataset = json.load(f)
        train_dataset = Dataset.from_dict(train_dataset)
    training_args = DPOConfig(beta=float(args.beta), 
        output_dir=args.output_dir,
        max_prompt_length=1024,
        max_length=1024,
        max_target_length=1024,
        remove_unused_columns=False,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        save_steps=20,
        logging_steps=20,
        save_total_limit=5,
        num_train_epochs=int(args.epochs),
        loss_type=args.loss_type)
    training_args.set_save(steps=20)
    
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset", help="path to DPO dataset json file")
    parser.add_argument("--model_path", help="Path to model")
    parser.add_argument("--output_dir", help="path to output directory where checkpoints will be stored")
    parser.add_argument("--tokenizer_name", help="Name of tokenizer")
    parser.add_argument("--beta", help="the value of beta")
    parser.add_argument("--epochs", help="Number of training epochs")
    parser.add_argument("--loss_type", default="sigmoid", help="Loss function")
    args = parser.parse_args()
    main(args)