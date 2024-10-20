#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import io
import logging
import math
import os
import pprint
import sys
import time
import json
import pdb 
from tqdm import tqdm
from datetime import datetime

import transformers
import torch

from datasets_apps.apps_critic import APPSBaseDataset
from trainers.trainer_rl_orig import Trainer_RL
import transformers.src
from transformers.src.transformers import Trainer  

import torch.multiprocessing

import transformers.src.transformers
torch.multiprocessing.set_sharing_strategy('file_system')

def run_training(args, train_data):
    if args.model in ['codet5-base', 'codet5-large']:
        model_path = args.model_path if args.model_path is not None else 'Salesforce/{}'.format(args.model)        
        print("Loading model from {}...".format(model_path))
        sys.stdout.flush()
        model = transformers.src.transformers.T5ForConditionalGeneration.from_pretrained(
            model_path,
            tuning_mode=args.tuning_mode, 
            clone_rl_head=args.clone_rl_head) 
        
        if args.clone_rl_head:
            # Optional: clone a seperate RL head and initialize the model weights from finetuned LM head 
            print("Initializing RL head with finetuned LM head...")
            lm_head_params = model.lm_head.weight.detach().numpy()
            model.rl_head.weight = torch.nn.Parameter(torch.tensor(lm_head_params))
                
    print('Finished loading model {}'.format(args.model))
    sys.stdout.flush()
    start_iteration = 0
    train_data.start_iteration = start_iteration

    training_args = transformers.src.transformers.training_args.TrainingArguments(
        output_dir=args.save_dir,
        overwrite_output_dir=True, 
        
        do_train=True,
        do_eval=False,
        do_predict=True,
        evaluation_strategy='no',
        eval_steps=0, 

        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size_per_replica,
        gradient_accumulation_steps=args.grad_acc_steps,

        learning_rate=args.lr,
        weight_decay=0.05,
        lr_scheduler_type='constant_with_warmup',

        logging_dir=args.save_dir, 
        logging_first_step=True,
        logging_steps=args.log_freq,
        save_steps=args.save_freq,
        save_total_limit=args.save_total_limit,

        dataloader_drop_last=True,
        dataloader_num_workers=0 if args.db else 8,

        local_rank=args.local_rank,

        deepspeed=args.deepspeed,
        fp16=args.fp16,
        
    )
    include_gt = args.include_gt == "True"
    if args.tuning_mode in ['critic', 'rl']:
        trainer = Trainer_RL(
            model=model,
            args=training_args,
            train_dataset=train_data,
            tuning_mode=args.tuning_mode        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
        )
    trainer.train()
    
    model.save_pretrained(os.path.join(args.save_dir, "final_checkpoint"))


def get_dataset(args): 
    
    fnames = os.listdir(args.train_path) 
    

    if args.model in ['codet5-base', 'codet5-large']:
        max_tokens = 512 
        max_src_tokens = 600
    else:
        max_tokens = 1024
        max_src_tokens = -1
    include_gt = args.include_gt == "True"
    train_data = APPSBaseDataset(
        dataroot=args.train_path, 
        problem_dirs=fnames,
        model=args.model,
        tokenizer=args.tokenizer,
        max_tokens=max_tokens,
        max_src_tokens=max_src_tokens,
        sample_mode=args.sample_mode,
        tuning_mode=args.tuning_mode,
        include_gt=include_gt,
        relative_returns=args.relative_returns,
        critic_scores_root = args.critic_scores_root,
        max_gt_per_task=args.max_gt_per_task,
        max_rl_per_task=args.max_rl_per_task
    )

    return train_data


def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataset 
    train_data = get_dataset(args)

    # Save args to file
    json.dump(argsdict, open(os.path.join(args.save_dir, "args.json"), 'w'))

    # Load and train model; save model checkpoints 
    run_training(args, train_data)


if __name__ == "__main__":
    from configs.train_configs import *
    
    main(args)