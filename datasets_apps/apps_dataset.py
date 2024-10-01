#
# Copyright (c) 2022, salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#

import torch
import glob
import logging
import random
import fnmatch
import sys
import numpy as np
import gc
import os
from tqdm import tqdm 
from collections import Counter
import pickle as pkl 
import json, pdb 

from multiprocessing import Manager
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import datasets_apps.utils as dsutils

class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problem_dirs, model, max_tokens, sample_mode, 
                 tuning_mode, max_src_tokens, relative_returns, tokenizer):
        self.dataroot = dataroot
        self.problem_dirs = problem_dirs 

        self.model = model
        self.sample_mode = sample_mode
        self.tuning_mode = tuning_mode
        self.relative_returns = relative_returns
        
        self.max_tokens = max_tokens
        self.max_src_tokens = max_src_tokens

        self.samples = []           
        self.all_error_types, self.all_error_subtypes, self.all_baseline_error_types = [], [], []
        self.initialize()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)


    def load_gen_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        info = []
        
        for idx, sol in enumerate(sols):
            sol_str = sol['code']  
            sample = (question_str, starter_code, sol_str, answer_type)
            samples.append(sample) 
            
            result = sol['result']
            error_type = sol['error_type']
            
            info.append((result, error_type))
            
        return samples, info 
    
    def load_rl_samples(self, sols, baseline_error_type): 
        samples = []
        info = []
        
        for idx, code in enumerate(sols['code']):   
            samples.append((sols['prompt'][idx], code))
            info.append((sols['gt_error_type'][idx], sols['error_hidden_states'][idx], baseline_error_type))
            
        return samples, info 
     
    def load_gt_samples(self, sols, answer_type, starter_code, question_str):
        samples = []
        for sol_str in sols:
            sol_str = dsutils.reindent_code(sol_str)
            sample = (question_str, starter_code, sol_str, answer_type)
            samples.append(sample)
        
        return samples 
    
    def get_gt_info(self):
        return (1, None)

    def get_baseline_error_type(self, sols): 
        return dsutils.get_error_type(sols[0]['result'])
    
    def update_error_stat(self, info):
        for i in info:
            error_type = dsutils.get_error_type(i[0])
            error_subtype = i[1]
            self.all_error_types.append(error_type)
            self.all_error_subtypes.append(error_subtype) 
            
    def update_error_stat_rl(self, info):
        for i in info:
            error_type = i[0]
            baseline_error_type = i[-1]
            self.all_error_types.append(error_type)
            self.all_baseline_error_types.append(baseline_error_type) 
    
    def initialize(self):
        all_samples = []
        skipped_problems = []
        samples_info = [] 
        gen_samples = [] 
        
        all_samples_dict = {} 
        print(f"Loading {len(self.problem_dirs)} problems from {self.dataroot}.")

        for problem_name in tqdm(self.problem_dirs):
            question_fname = os.path.join(self.dataroot, problem_name, "question.txt")
            sols_fname = os.path.join(self.dataroot, problem_name, "solutions.json")
            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue
                
            # Read the question description
            with open(question_fname, 'r') as f:
                question_str = f.read()

            starter_code = os.path.join(self.dataroot, problem_name, "starter_code.py")
            if (os.path.isfile(starter_code)):
                answer_type = "\nUse Call-Based format\n"
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                answer_type = "\nUse Standard Input format\n"
                starter_code = ""

            sols_str_list = json.load(open(sols_fname, 'r'))
            # get ground truth samples from the list of solutions, answer type, started code and question
            # one sample here is one answer to the question
            gt_samples = self.load_gt_samples(sols_str_list, answer_type, starter_code, question_str)
            if len(gt_samples) > 5:
                gt_samples = gt_samples[:5]
            all_samples += gt_samples 
            
            # Read all the solutions
           
        print(f"Samples:\n{all_samples[:5]}")
        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")
        
        print(f"Loaded {len(all_samples)} samples from {self.dataroot}.")
            
        self.samples = all_samples
        self.samples_info = samples_info 
        self.gen_samples = gen_samples 
        
        if self.relative_returns: 
            self.all_error_types = np.array(self.all_error_types)
            self.all_baseline_error_types = np.array(self.all_baseline_error_types)
            print('Sampled Error > Baseline error: {}/{}'.format((self.all_error_types>self.all_baseline_error_types).sum(),
                                                                  len(self.all_error_types)))
            print('Sampled Error = Baseline error: {}/{}'.format((self.all_error_types==self.all_baseline_error_types).sum(),
                                                                  len(self.all_error_types)))
            print('Sampled Error < Baseline error: {}/{}'.format((self.all_error_types<self.all_baseline_error_types).sum(),
                                                                  len(self.all_error_types)))
            
            sample_rewards = np.array([dsutils.get_reward_from_error_type(e) for e in self.all_error_types])
            baseline_rewards = np.array([dsutils.get_reward_from_error_type(e) for e in self.all_baseline_error_types])
            print("Relative returns distribution: {}".format(sorted(Counter(sample_rewards-baseline_rewards).items())))

            
    def __len__(self):
        if self.tuning_mode in ['critic', 'rl']:
            return len(self.gen_samples)
        return len(self.samples)

    def __getitem__(self, idx):
        
        if self.tuning_mode in ['critic']:
            raw_samples = self.pack_samples(idx, 'gen')
            inputs = self.sample_task(raw_samples, 'gen')
         
        elif self.tuning_mode in ['rl']:
            gt_sample_idx = random.randint(0, len(self.samples)-1)
            raw_gt_samples = self.pack_samples(gt_sample_idx)
            inputs = self.sample_task(raw_gt_samples)
            
            item = self.gen_samples[idx]
            info = self.samples_info[idx]

            gen_inputs = self.sample_rl_task(item, info)
            for k,v in gen_inputs.items():
                inputs['rl_{}'.format(k)] = v 
            
        else:
            raw_samples = self.pack_samples(idx)
            inputs = self.sample_task(raw_samples)

        gc.collect()
        return inputs

    def pack_samples(self, idx, sample_type=None):
        """
        Repeatedly pick question, answer pairs from self.dataroot until we hit max_tokens.
        This will not include the tokens for the QUESTION and ANSWER prompt, as well as the  
        self.question_prefix. These will be added later and the total input will be 
        truncated if necessary.

        Always include the sample at idx at the beginning.
        """
        curr_num_tokens = 0
        curr_samples = [] 
        
        if sample_type == 'gen':
            sample_pool = self.gen_samples
        else:
            sample_pool = self.samples
        
        if self.sample_mode == 'uniform_sol':
            curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[idx]
            if self.tuning_mode in ['critic'] and sample_type=='gen': 
                curr_result, curr_error_subtype = self.samples_info[idx] 
                
        elif self.sample_mode == 'uniform_prob':
            raise NotImplementedError()

        while curr_num_tokens < self.max_tokens:

            curr_q = curr_q[:150000]
            curr_s = curr_s[:150000]
            curr_a = curr_a[:150000]

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))            
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))
            
            if self.tuning_mode in ['critic'] and sample_type=='gen': 
                curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix, 
                                     curr_result, curr_error_subtype))
                break 
     
            else:
                curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix))
                
                # only pack 1 sample each sequence for codeT5 
                if self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py']:
                    break 

            if self.sample_mode == 'uniform_sol':
                new_idx = random.randint(0, len(sample_pool)-1)
                curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[new_idx] 
            elif self.sample_mode == 'uniform_prob':
                raise NotImplementedError()

        return curr_samples

    
    def sample_task(self, samples, sample_type=None):

        input_ids = []
        label_ids = []
                    
        if self.tuning_mode in ['critic'] and sample_type == 'gen': 
            error_types = [] 
        
        for sample in samples:
            if self.tuning_mode in ['critic'] and sample_type == 'gen': 
                q_str, s_str, a_str, answer_type, result, error_subtype = sample
            else:
                q_str, s_str, a_str, answer_type = sample
            
            q_str =  "\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"

            question_token_ids = self.tokenizer.encode(q_str, verbose=False)
            input_ids.extend(question_token_ids)
             
            answer_token_ids = self.tokenizer.encode(a_str, verbose=False)

            if self.model not in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py']:
                label_ids.extend([-100] * len(question_token_ids))
                answer_token_ids.append(self.tokenizer.eos_token_id)
                input_ids.extend(answer_token_ids)
            label_ids.extend(answer_token_ids)

            if self.tuning_mode in ['critic'] and sample_type == 'gen':
                error_types.append(dsutils.get_error_type(result))
                
        # Sanity checks and padding 
        input_ids_max_len = self.max_src_tokens if self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py'] else self.max_tokens 
        if len(input_ids) < input_ids_max_len: 
            new_input_ids = [self.tokenizer.eos_token_id] * input_ids_max_len
            new_input_ids[:len(input_ids)] = input_ids
            input_ids = new_input_ids 
            
            if self.model not in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py']:
                new_label_ids = [-100] * input_ids_max_len 
                new_label_ids[:len(label_ids)] = label_ids
                label_ids = new_label_ids
                
        if self.model in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py'] and len(label_ids) < self.max_tokens:
            new_label_ids = [-100] * self.max_tokens 
            new_label_ids[:len(label_ids)] = label_ids
            label_ids = new_label_ids
        
        if self.model not in ['codet5-base', 'codet5-large', 'codet5-large-ntp-py'] and len(input_ids) != len(label_ids): pdb.set_trace()
            
        if self.tuning_mode in ['critic'] and sample_type == 'gen': 
            assert len(error_types) == 1 
            
        # Cut off the excess
        input_ids = input_ids[:input_ids_max_len]
        attention_mask = [1]*len(input_ids)
        label_ids = label_ids[:self.max_tokens]
        
        out_sample = {
            "input_ids" : input_ids,
            "attention_mask" : attention_mask,
            "labels" :  label_ids
        }
        
        if self.tuning_mode in ['critic'] and sample_type == 'gen': 
            out_sample['error_types'] = torch.LongTensor(error_types)            
            
        return out_sample 

    def sample_rl_task(self, item, info):
        input_ids, labels = item
        gt_error, error_logit, baseline_error = info 
        
        softmax_fn = torch.nn.Softmax(dim=-1)
        rewards = softmax_fn(torch.tensor(error_logit))[:,gt_error]
        
        if self.relative_returns: 
            curr_reward = dsutils.get_reward_from_error_type(gt_error)
            baseline_reward = dsutils.get_reward_from_error_type(baseline_error) if baseline_error!=-1 else 0 
            relative_reward = curr_reward - baseline_reward
            rewards *= relative_reward
        else:
            rewards *= dsutils.get_reward_from_error_type(gt_error)
        
        # masking rewards 
        reward_mask = (error_logit == -np.float('Inf'))[:,0]
        rewards[reward_mask] = 0.0
        rl_label_ids = np.array(labels)
        rl_label_ids[reward_mask] = -100 

        assert len(labels) == len(rewards)
            
        if len(input_ids) < self.max_src_tokens:
            new_input_ids = [self.tokenizer.eos_token_id] * self.max_src_tokens
            new_input_ids[:len(input_ids)] = input_ids
            input_ids = new_input_ids
            
        if len(rl_label_ids) < self.max_tokens: 
            new_rl_label_ids = np.array([-100] * self.max_tokens)
            new_rl_label_ids[:len(rl_label_ids)] = rl_label_ids
            rl_label_ids = new_rl_label_ids
            
            new_rewards = torch.zeros(self.max_tokens)
            new_rewards[:len(rewards)] = rewards 
            rewards = new_rewards 
        
        input_ids = input_ids[:self.max_src_tokens]
        rewards = rewards[:self.max_tokens]
        rl_label_ids = rl_label_ids[:self.max_tokens]
        
        out_sample = {
            "input_ids" : torch.LongTensor(input_ids),
            "rewards": rewards,
            'label_ids': torch.LongTensor(rl_label_ids)
        }
        
        return out_sample 
    

def get_custom_dataset(dataset_config, tokenizer, split:str):

    if split == "train":
        split_path = "data/APPS/train/"
    else:
        split_path = "data/APPS/validation"

    return APPSBaseDataset(dataroot=split_path, 
        problem_dirs=os.listdir(split_path),
        model="codellama/CodeLlama-13b-Python-hf",
        max_tokens=512,
        max_src_tokens=600,
        sample_mode='uniform_sol',
        tuning_mode='none',
        relative_returns=False,
        tokenizer="codellama/CodeLlama-13b-Python-hf")
    