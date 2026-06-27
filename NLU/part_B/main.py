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
from tracker import ExperimentTracker

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
    last_model = None

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
        last_model = model

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

    # Saving (models from the last run)
    os.makedirs(f"bin/{exp_name}", exist_ok=True)
    torch.save(last_best_model.state_dict(), f"bin/{exp_name}/best_model.pt")
    torch.save(last_model.state_dict(), f"bin/{exp_name}/last_model.pt")

    # Tracker logging
    tracker = ExperimentTracker("NLU/part_B/results")
    tracker.log(
        name=exp_name,
        model_type=model_type,
        checkpoint=checkpoint,
        learning_rate=lr,
        dropout=dropout,
        test_slot_f1=test_f1_mean,
        test_slot_f1_std=test_f1_std,
        test_intent_acc=test_acc_mean,
        test_intent_acc_std=test_acc_std,
        dev_slot_f1=dev_f1_mean,
        dev_intent_acc=dev_acc_mean,
        sampled_epochs=last_run_curves['sampled_epochs'],
        losses_train=last_run_curves['losses_train'],
        losses_dev=last_run_curves['losses_dev'],
        dev_slot_f1s=last_run_curves['dev_slot_f1s'],
        dev_intent_accs=last_run_curves['dev_intent_accs'],
    )

    print("\n" + "=" * 50)
    print(f"EXPERIMENT COMPLETE: {exp_name}")
    print(f"Dev Slot F1 (avg): {dev_f1_mean:.4f}  |  Dev Intent Acc (avg): {dev_acc_mean:.4f}")
    print(f"Test Slot F1: {test_f1_mean:.4f} +- {test_f1_std:.4f}  |  Test Intent Acc: {test_acc_mean:.4f} +- {test_acc_std:.4f}")
    print(f"Models saved in: bin/{exp_name}/")
    print("=" * 50 + "\n")

    tracker.summary()
    tracker.plot_curves(tracker.experiments[-1])
    tracker.plot_metrics()
    tracker.export_csv()


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
