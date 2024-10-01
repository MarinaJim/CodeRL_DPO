import json
from ast import literal_eval
import matplotlib.pyplot as plt
import math

MIN_EPOCH = 0

def get_losses_from_txt(path):
    with open(path, "r") as f:
        train_output = f.read()
    train_output = train_output.split("\n")
    losses = []
    val_losses = []
    for line in train_output:
        if "\'loss\'" in line:
            line = literal_eval(line)
            line["epoch"] = MIN_EPOCH + math.floor(line["epoch"])
            losses.append(line)
        elif "eval_loss" in line:
            line = literal_eval(line)
            val_losses.append(line["eval_loss"])

    return losses, val_losses


def get_grouped_losses(losses):
    grouped_losses = {}
    for loss in losses:
        if loss["epoch"] in grouped_losses:
            grouped_losses[loss["epoch"]].append(loss["loss"])
        else:
            grouped_losses[loss["epoch"]] = [loss["loss"]]

    return grouped_losses

def get_avg_losses(grouped_losses):
    avg_losses = [sum(losses) / len(losses) for losses in grouped_losses.values()]
    return avg_losses

def plot_and_save(x_data, y_data, val_data, fig_path, title):

    plt.figure(figsize=(6, 5))
    plt.plot(x_data, y_data, label='Train loss')
    plt.plot(x_data, val_data, label='Validation loss')
    plt.xlabel("Epoch")
    plt.xticks(list(x_data))
    plt.ylabel("Cross-entropy loss")
    plt.legend()
    plt.title(title)
    plt.savefig(fig_path)


def main():
    paths = ["/storage/athene/work/sakharova/train_actor_0_2.output",
             "/storage/athene/work/sakharova/train_actor_3_7.output",
             "/storage/athene/work/sakharova/train_actor_7-15.output"]
    json_train_losses_path = "helpful_files/output_with_validation/train_losses.json"
    json_val_losses_path = "helpful_files/output_with_validation/val_losses.json"
    plot_path = "helpful_files/output_with_validation/losses.jpg"

    all_train_losses = []
    all_val_losses = []
    for i, path in enumerate(paths):
        losses, val_losses = get_losses(path)
        if i != len(paths) - 1:
            val_losses = val_losses[:-1]
        all_train_losses.extend(losses)
        all_val_losses.extend(val_losses)
    grouped_losses = get_grouped_losses(all_train_losses)
    del grouped_losses[len(all_val_losses)]
    avg_losses = get_avg_losses(grouped_losses)
    plot_and_save(list(grouped_losses.keys()), avg_losses, all_val_losses, plot_path)

    with open(json_train_losses_path, "w") as f:
        json.dump(all_train_losses, f)
    
    with open(json_val_losses_path, "w") as f:
        json.dump(all_val_losses, f)
    

if __name__ == "__main__":
    main()