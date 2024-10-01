import os
import shutil

root = "/storage/athene/work/sakharova/CodeRL_DPO/outputs/t5_dpo_models"
for folder in os.listdir(root):
    folder = os.path.join(root, folder)
    max_checkpoint = 0
    for checkpoint in os.listdir(folder):
        checkpoint = int(checkpoint.replace("checkpoint-", ""))
        if max_checkpoint < checkpoint:
            max_checkpoint = checkpoint

    for checkpoint in os.listdir(folder):
        if checkpoint != f"checkpoint-{max_checkpoint}":
            shutil.rmtree(os.path.join(folder, checkpoint))