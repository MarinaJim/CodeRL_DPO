import os
import json
import matplotlib.pyplot as plt
import sys
from collections import Counter
import matplotlib.style as style

sys.set_int_max_str_digits(0)


def get_number_inputs(folder="data/APPS/train"):
    """
    For each call-based problem in the folder, returns number of inputs in the file input_output.json.
    """
    tasks = os.listdir(folder)
    sys.set_int_max_str_digits(0)
    ns_inputs = {}
    test_cases = 0
    for task in tasks:
        # retrieve the number of outputs
        io_path = folder + "/" + task + "/input_output.json"
        if os.path.exists(io_path):
            with open(io_path, "r") as f:
                inputs = json.load(f)
                inputs = inputs["inputs"]
                test_cases += len(inputs)

                if len(inputs) < 10:
                    ns_inputs[task] = str(len(inputs))

                elif len(inputs) >10 and len(inputs) <=50:
                    ns_inputs[task] = "10-50"
                else:
                    ns_inputs[task] = "50+"
        else:
            ns_inputs[task] = "0"
    avg_amount = test_cases / len(tasks)
    return ns_inputs, avg_amount

labels = [str(i) for i in range(0, 10)]
labels.extend(["10-50", "50+"])

in_train, avg_amount = get_number_inputs(folder="data/APPS/train")
in_train = list(in_train.values())
counter = Counter(in_train)
values_1 = [counter.get(label, 0) for label in labels]

in_test, avg_amount = get_number_inputs(folder="data/APPS/test")
in_test = list(in_test.values())
counter = Counter(in_test)
values_2 = [counter.get(label, 0) for label in labels]

style.use("seaborn-v0_8-darkgrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].bar(labels, values_1, color="darkcyan")
axes[0].set_title("Train set: Test case distribution")
axes[0].set_xlabel("# test cases")
axes[0].set_ylabel("# occurrences")

axes[1].bar(labels, values_2, color="darkolivegreen")
axes[1].set_title("Test set: Test case distribution")
axes[1].set_xlabel("# test cases")
#axes[1].set_ylabel("# occurrences")

plt.tight_layout()

plt.savefig("/storage/athene/work/sakharova/CodeRL_DPO/presentation_scripts_and_pics/pics/traintest_ntests_dist.jpg")