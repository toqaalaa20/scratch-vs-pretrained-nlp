import copy
import math
import os
import time
from functools import partial

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from functions import eval_loop, train_loop, plot_training_curves, print_results
from model import GPT2_LoRA, LoRALinear, reset_lora_parameters
from utils import PennTreeBank, collate_fn, param_stats, read_file


def run_experiment(exp_name, rank, alpha, lr, batch_size=8, n_epochs=100, patience_max=3, max_grad_norm=5.0):
    """Fine-tune a pretrained GPT2 with LoRA adapters on the Q/K/V projections (the backbone
    stays frozen) on Penn Treebank, with early stopping on dev PPL. Evaluates the best checkpoint
    on the test set, saves it to bin/, and reports/plots the results.

    Args:
        exp_name: unique name used for the bin/ checkpoint dir and the results/ plot file.
        rank, alpha: LoRA adapter rank and scaling factor (scaling = alpha / rank).
        lr, batch_size, n_epochs, max_grad_norm: optimizer/training hyperparameters.
        patience_max: epochs without dev PPL improvement before early stopping.
    """
    print("\n" + "=" * 50)
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"Rank: {rank} | Alpha: {alpha} | LR: {lr}")
    print("=" * 50)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size * 2, collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE))
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE))

    model = GPT2_LoRA.from_pretrained("openai-community/gpt2", rank=rank, alpha=alpha)
    # from_pretrained's fast-init path leaves lora_A/lora_B as near-zero garbage
    # instead of running LoRALinear's own init (see model.reset_lora_parameters)
    reset_lora_parameters(model)
    model.to(DEVICE)

    # train only the LoRA adapters: freeze everything, then unfreeze LoRA params
    for param in model.parameters():
        param.requires_grad = False
    for module in model.modules():
        if isinstance(module, LoRALinear):
            for param in module.parameters():
                param.requires_grad = True

    param_stats(model)

    optimizer = optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)

    patience = patience_max
    losses_train, losses_dev = [], []
    ppls_train, ppls_dev = [], []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    best_train_ppl = None

    print("\nStarting Training Loop...")
    for epoch in range(n_epochs):
        start_time = time.time()

        ppl_train, loss_train = train_loop(train_loader, optimizer, model, tokenizer, max_grad_norm)
        ppls_train.append(ppl_train)
        epoch_duration = time.time() - start_time

        sampled_epochs.append(epoch)
        losses_train.append(loss_train.item() if torch.is_tensor(loss_train) else loss_train)

        ppl_dev, loss_dev = eval_loop(dev_loader, model, tokenizer)
        ppls_dev.append(ppl_dev)
        losses_dev.append(loss_dev.item() if torch.is_tensor(loss_dev) else loss_dev)

        if ppl_dev < best_ppl:
            status = "New Best!"
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to("cpu")
            best_train_ppl = ppl_train
            patience = patience_max
        else:
            patience -= 1
            status = f"Patience: {patience}"

        print(f"Epoch {epoch + 1}/{n_epochs}  Train PPL: {ppl_train:.2f}  Dev PPL: {ppl_dev:.2f}  [{status}]  ({epoch_duration:.1f}s)")

        if patience <= 0:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break

    print("\n" + "-" * 30)
    print("Finalizing Experiment...")

    if best_model is None:
        best_model = copy.deepcopy(model)

    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, best_model, tokenizer)

    os.makedirs(f"bin/{exp_name}", exist_ok=True)
    torch.save(best_model.state_dict(), f"bin/{exp_name}/best_model.pt")

    print_results(
        exp_name,
        best_dev_ppl=best_ppl,
        best_train_ppl=best_train_ppl,
        test_ppl=final_ppl,
    )
    plot_training_curves(
        exp_name,
        "LM/part_B/results",
        sampled_epochs,
        losses_train,
        losses_dev,
        ppls_train,
        ppls_dev,
    )


if __name__ == "__main__":
    # LoRA fine-tuning of GPT2 on Q/K/V projections.
    # Mandatory requirement: test PPL < 250, and lower than the from-scratch
    # GPT2 of Part 1.A (best so far ~34.7, see LM/part_A/results/experiments.csv).
    run_experiment(
        exp_name="LoRA rank=8 alpha=16 lr=1e-3",
        rank=8,
        alpha=16,
        lr=1e-3,
        batch_size=8,
        n_epochs=100,
        max_grad_norm=5.0,
    )
