import pandas as pd
import json
import os
all_outputs = ""

for i in os.listdir("/storage/athene/work/sakharova/CodeRL_DPO/outputs/test_results/"):
    path = f"/storage/athene/work/sakharova/CodeRL_DPO/outputs/test_results/{i}"
    i = int(i.replace(".pkl", ""))
    output = pd.read_pickle(path)
    all_outputs += str(output[i]["results"])
    all_outputs += "\n"

with open(f"/storage/athene/work/sakharova/CodeRL_DPO/helpful_files/output_files/test_results_dpo.txt", "w") as f:
    f.write(all_outputs)
    