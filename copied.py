import os
import torch
import argparse

from transformers import AutoTokenizer, TrainingArguments, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def main(args):
    output_dir=args.output_dir
    model_name = args.model_path

    dataset = load_dataset("json", data_files=args.path_to_dataset,split="train")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    model_ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, quantization_config=bnb_config)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token

    def return_prompt_and_responses(samples):
        return {
            "prompt": [
                f"You are a helpful coding assistant. You generate Python 3 code based on a task description in natural language.\n### TASK: ```{input}```\n ### CODE: "
                for input in samples["prompt"]
            ],
            "chosen": samples["chosen"],
            "rejected": samples["rejected"],
        }

    original_columns = dataset.column_names

    dataset = dataset.map(
        return_prompt_and_responses,
        batched=True,
        remove_columns=original_columns
    )

    training_args = DPOConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        gradient_checkpointing =True,
        max_grad_norm= 0.3,
        num_train_epochs=args.epochs, 
        save_steps= 100,
        learning_rate=2e-4,
        bf16=True,
        save_total_limit=3,
        logging_steps=1,
        output_dir=output_dir,
        optim="paged_adamw_32bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        remove_unused_columns=False,
        max_prompt_length=1024,
        max_length=1024,
    )

    peft_config = LoraConfig(
        r=32,
        lora_alpha=16,
        target_modules=find_all_linear_names(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=model_ref,
        args=training_args,
        beta=args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer
    )


    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)


    output_dir = os.path.join(output_dir, "final_checkpoint")
    dpo_trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_dataset", help="path to DPO dataset json file")
    parser.add_argument("--model_path", help="Path to model")
    parser.add_argument("--output_dir", help="path to output directory where checkpoints will be stored")
    parser.add_argument("--tokenizer_name", help="Name of tokenizer")
    parser.add_argument("--beta", type=float, help="the value of beta")
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    args = parser.parse_args()
    main(args)