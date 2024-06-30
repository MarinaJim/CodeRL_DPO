import shutil
import os
import argparse

def bring_data_to_preference(train_root, preference_root, task):
    train_path = os.path.join(train_root, task)
    preference_path = os.path.join(preference_root, task)
    if not os.path.exists(preference_path):
        return
    shutil.copy(os.path.join(train_path, "question.txt"), preference_path)

    if os.path.exists(os.path.join(train_path, "starter_code.py")):
        shutil.copy(os.path.join(train_path, "starter_code.py"), preference_path)

    if os.path.exists(os.path.join(preference_path, "monkeytype.sqlite3")):
        os.remove(os.path.join(preference_path, "monkeytype.sqlite3"))
    
    if os.path.exists(os.path.join(preference_path, "parameters.json")):
        os.remove(os.path.join(preference_path, "parameters.json"))
    
    if os.path.exists(os.path.join(preference_path, "call.py")):
        os.remove(os.path.join(preference_path, "call.py"))

    if os.path.exists(os.path.join(preference_path, "sample_call.py")):
        os.remove(os.path.join(preference_path, "sample_call.py"))
    
    if os.path.exists(os.path.join(preference_path, "solution.py")):
        os.remove(os.path.join(preference_path, "solution.py"))

    if os.path.exists(os.path.join(preference_path, "method_output.json")):
        os.remove(os.path.join(preference_path, "method_output.json"))
    
    if os.path.exists(os.path.join(preference_path, "__pycache__")):
        shutil.rmtree(os.path.join(preference_path, "__pycache__"))


def main(args):
    train_root = args.train_root
    preference_root = "/".join([path for path in args.task.split("/")[:-1]])
    task = args.task.split("/")[-1]
    bring_data_to_preference(train_root, preference_root, task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing generated programs with unit testing")
    parser.add_argument("-tr","--train_root", type=str, help="Path to the train root")
    parser.add_argument("-t","--task", type=str, help="Path to the task crosshair will run for")
    args = parser.parse_args()
    main(args)
