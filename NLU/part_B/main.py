# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import os
import copy
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import load_atis_splits, build_label_vocabs, JointATISDataset, collate_fn, IGNORE_INDEX
from model import build_model, load_tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Used to report errors on CUDA side
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run_experiment(
    exp_name,
    model_type,
    checkpoint,
    lr,
    dropout=0.1,
    dev_portion=0.10,
    train_batch_size=32,
    eval_batch_size=64,
    n_epochs=10,
    patience=3,
    runs=5,
    max_length=64,
):
    """Fine-tune a pretrained encoder/decoder backbone (BERT or GPT2) with a joint intent/slot
    head on ATIS, repeated `runs` times (the test metrics reported are mean +- std across runs;
    only the last run's model is saved/plotted). Each run trains with early stopping on dev slot
    F1, then evaluates on the test set.

    Args:
        exp_name: unique name used for the bin/ checkpoint dir and the results/ plot file.
        model_type: "bert" or "gpt2" (selects the pooling strategy in model.build_model).
        checkpoint: HuggingFace checkpoint name to load the backbone/tokenizer from.
        lr, dropout, train_batch_size, eval_batch_size, max_length: optimizer/data hyperparameters.
        dev_portion: fraction of the training set held out (stratified by intent) for early stopping.
        n_epochs: max epochs per run; training stops early after `patience` epochs without
            dev slot F1 improvement.
        runs: number of independent training runs (different random init) to average metrics over.
    """
    print("\n" + "=" * 50)
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"Model: {model_type} ({checkpoint}) | LR: {lr} | Dropout: {dropout}")
    print("=" * 50)

    train_raw, dev_raw, test_raw = load_atis_splits(
        os.path.join('dataset', 'ATIS', 'train.json'),
        os.path.join('dataset', 'ATIS', 'test.json'),
        dev_portion=dev_portion,
    )
    slot2id, id2slot, intent2id, id2intent = build_label_vocabs(train_raw, dev_raw, test_raw)
    n_slots = len(slot2id)
    n_intents = len(intent2id)

    tokenizer = load_tokenizer(checkpoint, model_type)

    train_dataset = JointATISDataset(train_raw, tokenizer, slot2id, intent2id, model_type, max_length)
    dev_dataset = JointATISDataset(dev_raw, tokenizer, slot2id, intent2id, model_type, max_length)
    test_dataset = JointATISDataset(test_raw, tokenizer, slot2id, intent2id, model_type, max_length)

    collate = partial(collate_fn, tokenizer=tokenizer, device=DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, collate_fn=collate)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=collate)

    print(f"Using device: {DEVICE}")

    test_f1s, test_accs, dev_f1s, dev_accs = [], [], [], []
    last_run_curves = None
    last_best_model = None

    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")

        model = build_model(model_type, checkpoint, n_slots, n_intents, dropout).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion_slots = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
        criterion_intents = nn.CrossEntropyLoss()  # No pad tokens, all sequences have a single intent label

        patience_counter = patience
        losses_train, losses_dev = [], []
        dev_slot_f1s, dev_intent_accs, sampled_epochs = [], [], []
        best_f1 = 0
        best_intent_acc = 0
        best_model = None

        for i in range(n_epochs):
            loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model)
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                           criterion_intents, model, id2slot, id2intent)

            f1 = results_dev['total']['f']
            intent_acc = intent_res['accuracy']
            sampled_epochs.append(i)
            losses_train.append(np.asarray(loss).mean())
            losses_dev.append(np.asarray(loss_dev).mean())
            dev_slot_f1s.append(f1)
            dev_intent_accs.append(intent_acc)

            print(f"Epoch {i + 1}/{n_epochs}  Slot F1: {f1:.3f}  Intent Acc: {intent_acc:.3f}")

            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                best_intent_acc = intent_acc
                best_model = copy.deepcopy(model).to('cpu')
                patience_counter = patience
            else:
                patience_counter -= 1
            if patience_counter <= 0:  # Early stopping with patience
                print(f"\nEarly stopping triggered at epoch {i}")
                break

        if best_model is None:
            best_model = copy.deepcopy(model)
        best_model.to(DEVICE)

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                                  criterion_intents, best_model, id2slot, id2intent)
        test_f1 = results_test['total']['f']
        test_acc = intent_test['accuracy']
        print(f"Run {run + 1} -- Slot F1: {test_f1:.4f}  Intent Acc: {test_acc:.4f}")

        test_f1s.append(test_f1)
        test_accs.append(test_acc)
        dev_f1s.append(best_f1)
        dev_accs.append(best_intent_acc)

        last_run_curves = dict(
            sampled_epochs=sampled_epochs,
            losses_train=losses_train,
            losses_dev=losses_dev,
            dev_slot_f1s=dev_slot_f1s,
            dev_intent_accs=dev_intent_accs,
        )
        last_best_model = best_model

    print("\n" + "-" * 30)
    print("Finalizing Experiment...")

    test_f1s = np.asarray(test_f1s)
    test_accs = np.asarray(test_accs)
    test_f1_mean, test_f1_std = test_f1s.mean(), test_f1s.std()
    test_acc_mean, test_acc_std = test_accs.mean(), test_accs.std()
    dev_f1_mean = float(np.mean(dev_f1s))
    dev_acc_mean = float(np.mean(dev_accs))

    print(f"Slot F1: {test_f1_mean:.3f} +- {test_f1_std:.3f}")
    print(f"Intent Acc: {test_acc_mean:.3f} +- {test_acc_std:.3f}")

    # Saving (best model from the last run)
    os.makedirs(f"bin/{exp_name}", exist_ok=True)
    torch.save(last_best_model.state_dict(), f"bin/{exp_name}/best_model.pt")

    # Results
    print_results(
        exp_name,
        dev_slot_f1=dev_f1_mean,
        dev_intent_acc=dev_acc_mean,
        test_slot_f1=test_f1_mean,
        test_slot_f1_std=test_f1_std,
        test_intent_acc=test_acc_mean,
        test_intent_acc_std=test_acc_std,
    )
    plot_training_curves(
        exp_name,
        "NLU/part_B/results",
        last_run_curves['sampled_epochs'],
        last_run_curves['losses_train'],
        last_run_curves['losses_dev'],
        last_run_curves['dev_slot_f1s'],
        last_run_curves['dev_intent_accs'],
    )


if __name__ == "__main__":

    # Fine-tune pretrained GPT2 (decoder-only, causal attention).
    # Pooling for intent classification is the last real token (manually
    # appended eos), since causal attention means only that position has
    # attended to the whole sentence.
    # run_experiment(
    #     exp_name="GPT2 fine-tune lr=2e-5",
    #     model_type="gpt2",
    #     checkpoint="openai-community/gpt2",
    #     lr=2e-5,
    # )

    # Fine-tune pretrained BERT (encoder-only, bidirectional attention).
    # Pooling for intent classification is the native [CLS] token at position 0.
    run_experiment(
        exp_name="BERT fine-tune lr=2e-5",
        model_type="bert",
        checkpoint="bert-base-uncased",
        lr=2e-5,
    )
