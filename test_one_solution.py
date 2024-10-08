import json
import numpy as np
import os
import os.path 
import pprint
import glob 
from tqdm import tqdm
import pdb
import traceback 
import pickle as pkl 
from typing import List

from utils.testing_util import run_test

def eval_and_save_problems(args):

    problems = sorted(glob.glob(args.test_path + '/*'))
    test_indices = [] 
    # enumerate through sorted problems
    for problem_idx, problem in enumerate(problems): 
        # get the real id
        problem_id = int(problem.split('/')[-1])
        # get real file path
        code_file_path = args.code_path + '/{}.json'.format(problem_id)
        if os.path.exists(code_file_path):
            # append the list index
            test_indices.append(problem_idx)

    real_index = test_indices[args.index] 
    problem = problems[real_index]
    real_index = int(problem.split('/')[-1])

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print('Testing sample {}'.format(problem))
    
    if args.example_tests:
        print("Using example tests") 
    
    codes_loc = args.code_path + '/{}.json'.format(real_index)
    if not os.path.isfile(codes_loc):
        print(f"codes_loc: {codes_loc} is not a file")
        exit() 
    with open(codes_loc, "r") as file: 
        gen_codes = json.load(file)[str(real_index)]['code']

    test_file = os.path.join(problem, "input_output.json")
    tests = json.load(open(test_file, 'r'))
    nb_tests = len(tests['inputs'])
    if args.max_tests!=-1 and nb_tests > args.max_tests: 
        print("args.max_tests!=-1 and nb_tests > args.max_tests")
        exit() 

    if os.path.isfile(args.output_path + '/{}.pkl'.format(real_index)):
        exit()
        
    print("Saving to {}".format(args.output_path + '/{}.pkl'.format(real_index)))

    all_results, all_errors, all_sols = [], [], []

    for o_idx, o in tqdm(enumerate(gen_codes), total=len(gen_codes), ncols=0, leave=False):

        curr_results = []
        curr_errors = []
        curr_sol = None
        try:
            curr_results, curr_errors, _, curr_sol = run_test(prob_path=problem, test=o, debug=args.debug, 
                                          example_tests=args.example_tests)

            curr_errors = [(e, traceback.format_tb(e.__traceback__)) if e is not None else e for e in curr_errors]
            all_results.append(curr_results)
            all_errors.append(curr_errors)
            all_sols.append(curr_sol)

        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            break

        save_results = {real_index : {'results': all_results, 'errors': all_errors, 'sols': all_sols}} 
        with open(args.output_path + '/{}.pkl'.format(real_index), "wb") as file:
            pkl.dump(save_results, file)  

    '''
    How to read results:
    [-2] = compile error, 
    [-1] = runtime error 
    [False] = failed test case 
    [True] = passed test case
    '''

    save_results = {real_index : {'results': all_results, 'errors': all_errors, 'sols': all_sols}} 
    pkl.dump(save_results,  open(args.output_path + '/{}.pkl'.format(real_index), "wb"))                    

def main(args):    
    argsdict = vars(args)    
    eval_and_save_problems(args)

if __name__ == "__main__":
    from configs.unit_test_configs import * 
    main(args)