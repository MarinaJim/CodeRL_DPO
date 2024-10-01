import matplotlib.pyplot as plt
import matplotlib.style as style
import json
import os
import sys
sys.set_int_max_str_digits(0)

def is_call_based(root):
    if os.path.exists(os.path.join(root, "input_output.json")):
        with open(os.path.join(root, "input_output.json")) as f:
            in_outs = json.load(f)
            if in_outs.get("fn_name") is None:
                return False
            else:
                return True
    return None


def count_call_based(folder):
    call_based = 0
    for task in os.listdir(folder):
        is_cb = is_call_based(os.path.join(folder, task))
        if is_cb:
            call_based += 1
    return call_based

train_cb = count_call_based("data/APPS/train")
test_cb = count_call_based("data/APPS/test")

print(f"Train cb: {train_cb}, train stdout: {5000-train_cb}\nTest cb: {test_cb}, test stdout: {5000-test_cb}")