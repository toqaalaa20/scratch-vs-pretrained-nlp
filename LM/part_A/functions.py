import csv
import json
import math
import os
from datetime import datetime
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt



def train_loop(data, optimizer, criterion, model):
    """Run one training epoch. Returns (perplexity, average per-token loss) over the epoch."""
    model.train()
    loss_array = []
    number_of_tokens = []
    
    pbar = tqdm(data, desc="Train", unit="batch", total=len(data), leave=False, dynamic_ncols=True)

    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad() # Zeroing the gradient
        output = model(input_ids)
        # need to reshape as (B, vocab, L)
        loss = criterion(output.permute(0,2,1), labels)
        loss_array.append(loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        loss.backward() # Compute the gradient, deleting the computational graph
        optimizer.step() # Update the weights

        if i % 100 == 0:
            pbar.set_postfix({"loss": f"{(sum(loss_array)/sum(number_of_tokens)).item():.4f}"})

    loss_to_return = sum(loss_array)/sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return

def eval_loop(data, eval_criterion, model):
    """Evaluate the model on `data` with gradients disabled. Returns (perplexity, average per-token loss)."""
    model.eval()
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for input_ids, labels, n_tokens in tqdm(data, desc="Eval", unit="batch", total=len(data), leave=False, dynamic_ncols=True):
            output = model(input_ids)
            # need to reshape as (B, vocab, L)
            loss = eval_criterion(output.permute(0,2,1), labels)
            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)
            
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return


def plot_training_curves(exp_name, save_dir, sampled_epochs, losses_train, losses_dev, ppls_train=None, ppls_dev=None):
    """Plot train/dev loss (and PPL, if given) over epochs and save the figure to save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    has_ppl = ppls_train is not None and ppls_dev is not None
    fig, axes = plt.subplots(1, 2 if has_ppl else 1, figsize=(13, 4))
    if not has_ppl:
        axes = [axes]

    axes[0].plot(sampled_epochs, losses_train, label="Train loss", color="#185FA5", linewidth=1.6)
    axes[0].plot(sampled_epochs, losses_dev, label="Dev loss", color="#D85A30", linewidth=1.6, linestyle="--")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    if has_ppl:
        axes[1].plot(sampled_epochs, ppls_train, label="Train PPL", color="#185FA5", linewidth=1.6, marker="o", markersize=3)
        axes[1].plot(sampled_epochs, ppls_dev, label="Dev PPL", color="#D85A30", linewidth=1.6, linestyle="--", marker="s", markersize=3)
        axes[1].set_title("Perplexity"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("PPL")
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    fig.suptitle(exp_name, fontsize=11)
    plt.tight_layout()
    safe_name = exp_name.replace(" ", "_").replace("/", "-")
    path = os.path.join(save_dir, f"curves_{safe_name}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {path}")


def print_results(exp_name, save_dir="LM/part_A/results", **metrics):
    """Print a final summary of the experiment's metrics and append it to
    save_dir/experiments.json and save_dir/experiments.csv."""
    print("\n" + "=" * 50)
    print(f"RESULTS: {exp_name}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print("=" * 50 + "\n")
    log_experiment(exp_name, save_dir, **metrics)


def log_experiment(exp_name, save_dir, **metrics):
    """Append one experiment's metrics to experiments.json and rewrite experiments.csv
    from the full history (column set grows to cover any new metric keys)."""
    os.makedirs(save_dir, exist_ok=True)
    record = {"name": exp_name, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **metrics}

    json_path = os.path.join(save_dir, "experiments.json")
    records = []
    if os.path.exists(json_path):
        with open(json_path) as f:
            records = json.load(f)
    records.append(record)
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2)

    csv_path = os.path.join(save_dir, "experiments.csv")
    fieldnames = list(dict.fromkeys(key for r in records for key in r.keys()))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"Logged experiment to: {json_path} and {csv_path}")


    