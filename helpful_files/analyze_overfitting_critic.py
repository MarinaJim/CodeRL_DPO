import os
import pickle
from evaluate_critic import get_scores_for_model

path = "/storage/athene/work/sakharova/CodeRL_DPO/outputs/results_for_presentation/codet5-critic/test/CodeT5-base-all-1ep"
gt = get_scores_for_model(path, "gt_error_type")
pred = get_scores_for_model(path, "pred_error_type")

num_no_3 = 0
num_3 = 0
for result in pred:
    for res in result:
        if res != 3:
            num_no_3 += 1
        else:
            num_3 += 1

print(num_no_3)
print(num_3)
