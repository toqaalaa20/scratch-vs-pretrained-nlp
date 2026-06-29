from functions import eval_loop, train_loop, plot_training_curves, print_results
import torch.nn as nn
import torch
import math
import torch.optim as optim
from functools import partial
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from model import GPT2, init_weights
from utils import PennTreeBank, collate_fn, read_file

import copy
import tqdm
import os
import time

def run_experiment(exp_name, phase, lr, d_model, n_heads, num_layers, ff_dim, dropout=0.0, weight_tying=False, n_epochs=100):
    """Train a from-scratch GPT2 language model on Penn Treebank with early stopping on dev PPL,
    then evaluate the best checkpoint on the test set, save it to bin/, and report/plot the results.

    Args:
        exp_name: unique name used for the bin/ checkpoint dir and the results/ plot file.
        phase: hyperparameter-search phase this run belongs to (kept for the printed header only).
        lr, d_model, n_heads, num_layers, ff_dim, dropout, weight_tying: model/optimizer hyperparameters.
        n_epochs: max epochs; training stops early after 3 epochs without dev PPL improvement.
    """
    # --- Reporting: Start Header ---
    print("\n" + "="*50)
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"Phase: {phase} | LR: {lr} | Layers: {num_layers} | Heads: {n_heads}")
    print(f"Model Dim: {d_model} | FF Dim: {ff_dim} | Dropout: {dropout}")
    print("="*50)

    train_raw = read_file("dataset/PennTreeBank/ptb.train.txt")
    dev_raw = read_file("dataset/PennTreeBank/ptb.valid.txt")
    test_raw = read_file("dataset/PennTreeBank/ptb.test.txt")

    train_dataset = PennTreeBank(train_raw)
    dev_dataset = PennTreeBank(dev_raw)
    test_dataset = PennTreeBank(test_raw)

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_len = len(tokenizer)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    train_loader = DataLoader(train_dataset, batch_size=8, collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=16, collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE))
    test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=partial(collate_fn, tokenizer=tokenizer, device=DEVICE))

    model = GPT2(
        vocab_len,
        pos_emb_size=1024,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
        weight_tying=weight_tying,
    ).to(DEVICE)

    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_train = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    criterion_eval = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

    patience_counter = 3
    losses_train = []
    losses_dev = []
    ppls_train = [] 
    ppls_dev = []   
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    best_train_ppl = None
    
    print("\nStarting Training Loop...")
    pbar = tqdm.tqdm(range(n_epochs), desc="Epoch", unit="epoch", dynamic_ncols=True)
    
    for epoch in pbar:
        start_time = time.time()
        
        # Training
        ppl_train, loss_train = train_loop(train_loader, optimizer, criterion_train, model)
        ppls_train.append(ppl_train)
        epoch_duration = time.time() - start_time
        
        # Evaluation
        sampled_epochs.append(epoch)
        losses_train.append(loss_train.item())
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        ppls_dev.append(ppl_dev)
        losses_dev.append(loss_dev.item())

        # --- Reporting: Epoch Summary ---
        if ppl_dev < best_ppl:
            status = "New Best!"
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            best_train_ppl = ppl_train
            patience_counter = 3
        else:
            patience_counter -= 1
            status = f"Patience: {patience_counter}"

        pbar.set_description(f"Epoch {epoch+1}/{n_epochs}")
        pbar.set_postfix({"Train PPL": f"{ppl_train:.2f}", "Dev PPL": f"{ppl_dev:.2f}", "Status": status})

        if patience_counter <= 0:
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break 

    # --- Final Evaluation ---
    print("\n" + "-"*30)
    print("Finalizing Experiment...")
    
    if best_model is None:
        best_model = copy.deepcopy(model)

    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)    

    # Saving
    os.makedirs(f"bin/{exp_name}", exist_ok=True)
    torch.save(best_model.state_dict(), f'bin/{exp_name}/best_model.pt')

    # Results
    print_results(
        exp_name,
        best_dev_ppl=best_ppl,
        best_train_ppl=best_train_ppl,
        test_ppl=final_ppl,
    )
    plot_training_curves(
        exp_name,
        "LM/part_A/results",
        sampled_epochs,
        losses_train,
        losses_dev,
        ppls_train,
        ppls_dev,
    )

if __name__ == "__main__":

    # --- Step 0: Baseline LR search (architecture fixed at d_model=20, n_heads=1, num_layers=1, ff_dim=20) ---
    # lr=1e-3 already run and logged ("Baseline lr=1e-3", dev PPL 51.48). Remaining candidates:
    # for lr in [5e-4]:
        # run_experiment(
        #     exp_name=f"Baseline lr={lr}",
        #     phase=0,
        #     lr=lr,
        #     d_model=20,
        #     n_heads=1,
        #     num_layers=1,
        #     ff_dim=20,
        #     weight_tying=False,
        #     dropout=0.0,
        #     n_epochs=100
        # )

    # --- Step 1: Hyperparameter optimization (num_layers sweep, staged previously) ---
    # run_experiment(
    #     exp_name="Hyperparameter Tuning, d_model=64,ff_dim=256, n_heads=4,num_layers=2, lr=4e-4",
    #     phase=1,
    #     lr=4e-4,
    #     d_model=64,
    #     n_heads=4,
    #     num_layers=2,
    #     ff_dim=256,
    #     weight_tying=False,
    #     dropout=0.0,
    #     n_epochs=100
    # )

    # --- Step 2: Dropout (added at embeddings, post-softmax attention, attn out-proj, FFN output) ---
    # run_experiment(
    #     exp_name="Dropout, d_model=64,ff_dim=256, n_heads=4,num_layers=2, lr=4e-4, dropout=0.1",
    #     phase=2,
    #     lr=4e-4,
    #     d_model=64,
    #     n_heads=4,
    #     num_layers=2,
    #     ff_dim=256,
    #     weight_tying=False,
    #     dropout=0.1,
    #     n_epochs=100
    # )

    # --- Step 3: Weight tying (lm_head.weight = token_embed.weight) ---
    run_experiment(
        exp_name="Weight Tying, d_model=64,ff_dim=256, n_heads=4,num_layers=2, lr=4e-4, dropout=0.1",
        phase=3,
        lr=4e-4,
        d_model=64,
        n_heads=4,
        num_layers=2,
        ff_dim=256,
        weight_tying=True,
        dropout=0.1,
        n_epochs=100
    )
