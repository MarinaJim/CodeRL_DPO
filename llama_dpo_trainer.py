from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
from datasets import load_dataset, Dataset
PatchDPOTrainer()
from transformers import AutoTokenizer
import torch
from transformers import TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from peft.peft_model import PeftModel
from trl import DPOTrainer, DPOConfig
import argparse
import json
import gc

gc.collect()
torch.cuda.empty_cache()

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    path_to_dataset = args.path_to_dataset

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
    )
    model.config.use_cache = False
    model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(
        model,
        args.peft_path,
        is_trainable=True,
        adapter_name="dpo",
    )
    model.load_adapter(args.peft_path, adapter_name="reference")

    with open(path_to_dataset, "r") as f:
            train_dataset = json.load(f)
            train_dataset = Dataset.from_dict(train_dataset)
    
    training_args = DPOConfig(beta=float(args.beta), 
        model_adapter_name="dpo",
        ref_adapter_name="reference",
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
        model = model,
        args = training_args,
        train_dataset = train_dataset,
        tokenizer = tokenizer,
    )
    print("Initialized the DPO trainer")
    dpo_trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset", help="path to DPO dataset json file")
    parser.add_argument("--model_path", help="Path to model")
    parser.add_argument("--output_dir", help="path to output directory where checkpoints will be stored")
    parser.add_argument("--tokenizer_name", help="Name of tokenizer")
    parser.add_argument("--beta", help="the value of beta")
    parser.add_argument("--epochs", help="Number of training epochs")
    parser.add_argument("--loss_type", default="sigmoid", help="Loss function")
    parser.add_argument("--peft_path")
    args = parser.parse_args()
    main(args)