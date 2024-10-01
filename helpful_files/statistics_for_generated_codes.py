import json
import os
from collections import Counter
import pickle
import shutil


def stats(folder):

    nl = 0
    c_code = 0
    python_code = 0
    print_3 = 0
    java_code = 0
    total_codes = 0
    for task in os.listdir(folder):
        with open(os.path.join(folder, task), "r") as f:
            data = json.load(f)
            codes = data[list(data.keys())[0]]["code"]
            c_codes = [code for code in codes if "#include" in code]
            java_codes = [code for code in codes if "public static void main" in code and "#include" not in code]
            python_codes = [code for code in codes if (code not in c_codes and code not in java_codes) and ("print " in code or "range" in code or "print(" in code or "input(" in code or "raw_input" in code or "def " in code or "import " in code) and not any(word in code for word in ("ANSWER:", "Output", "Input", "---"))]
            nl_codes = [code for code in codes if all(code not in group for group in (python_codes, c_codes, java_codes))]
            print(python_codes)

            c_code += len(c_codes)
            nl += len(nl_codes)
            python_code += len(python_codes)
            java_code += len(java_codes)
            total_codes += len(codes)


    print("NL percentage", nl / total_codes)
    print("C code percentage", c_code/total_codes)
    print("java", java_code / total_codes)
    print("Python", python_code / total_codes)

    print("Total percentage", (nl + c_code + java_code + python_code) / total_codes)

stats("/storage/athene/work/sakharova/CodeRL_DPO/outputs/results_for_presentation/codet5/sft_1ep/codes")