import csv
import json
import math
import os
from datetime import datetime

from tqdm import tqdm
import torch
import matplotlib.pyplot as plt


def train_loop(data, optimizer, model, tokenizer, max_grad_norm=1.0):
    """Run one training epoch over the LoRA-adapted GPT2, clipping gradients of the trainable
    (adapter-only) parameters. Returns (perplexity, average per-token loss) over the epoch."""
    model.train()
    loss_array = []
    number_of_tokens = []
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    pbar = tqdm(data, desc="Train", unit="batch", total=len(data), leave=False, dynamic_ncols=True)

    for i, (input_ids, attention_mask, n_tokens) in enumerate(pbar):
        optimizer.zero_grad()  # Zeroing the gradient
        # we don't shift the labels to the left, the model manages it internally
        labels = input_ids.clone().detach()
        # we cannot specify ignore_index, so we replace our pad tokens with -100
        # -100 is ignored by default when the model computes the loss
        labels[labels == tokenizer.pad_token_id] = -100
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss_array.append(output.loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        output.loss.backward()  # Compute the gradient, deleting the computational graph
        # clip gradients to avoid early-training blowups destabilizing the adapters
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()  # Update the weights

        if i % 100 == 0:
            pbar.set_postfix({"loss": f"{(sum(loss_array)/sum(number_of_tokens)).item():.4f}"})

    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return


def eval_loop(data, model, tokenizer):
    """Evaluate the model on `data` with gradients disabled. Returns (perplexity, average per-token loss)."""
    model.eval()
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():  # It is used to avoid the creation of the computational graph
        for input_ids, attention_mask, n_tokens in tqdm(data, desc="Eval", unit="batch", total=len(data), leave=False, dynamic_ncols=True):
            labels = input_ids.clone().detach()
            labels[labels == tokenizer.pad_token_id] = -100
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss_array.append(output.loss.item() * n_tokens)
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


def print_results(exp_name, save_dir="LM/part_B/results", **metrics):
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
