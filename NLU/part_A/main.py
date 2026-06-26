# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
import os
import copy
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from utils import Lang, PAD_TOKEN, IntentsAndSlots, collate_fn, load_data
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from model import GPT2, init_weights
from tqdm import tqdm
import torch.nn as nn
from tracker import ExperimentTracker

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Used to report errors on CUDA side
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def run_experiment(
    exp_name,
    phase,
    lr,
    d_model=20,
    n_heads=1,
    num_layers=1,
    ff_dim=20,
    dropout=0.0,
    pos_emb_size=1024,
    dev_portion=0.10,
    train_batch_size=128,
    eval_batch_size=64,
    n_epochs=200,
    patience=3,
):
    print("\n" + "=" * 50)
    print(f"STARTING EXPERIMENT: {exp_name}")
    print(f"Phase: {phase} | LR: {lr} | Layers: {num_layers} | Heads: {n_heads}")
    print(f"Model Dim: {d_model} | FF Dim: {ff_dim} | Dropout: {dropout}")
    print("=" * 50)

    tmp_train_raw = load_data(os.path.join('dataset', 'ATIS', 'train.json'))
    test_raw = load_data(os.path.join('dataset', 'ATIS', 'test.json'))

    intents = [x['intent'] for x in tmp_train_raw]  # We stratify on intents
    count_y = Counter(intents)

    labels = []
    inputs = []
    mini_train = []

    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # If some intents occurs only once, we put them in training
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])
    # Random Stratify
    X_train, X_dev, _, _ = train_test_split(inputs, labels, test_size=dev_portion,
                                                        random_state=42,
                                                        shuffle=True,
                                                        stratify=labels)
    X_train.extend(mini_train)
    train_raw = X_train
    dev_raw = X_dev

    slot2id = {'pad': PAD_TOKEN}
    intent2id = {}
    # Map the words only from the train set
    # Map slot and intent labels of train, dev and test set. 'unk' is not needed.
    w2id = {'pad': PAD_TOKEN, 'unk': 1}

    for example in train_raw:
        for w in example['utterance'].split():
            if w not in w2id:
                w2id[w] = len(w2id)
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)

    for example in dev_raw:
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)

    for example in test_raw:
        for slot in example['slots'].split():
            if slot not in slot2id:
                slot2id[slot] = len(slot2id)
        if example['intent'] not in intent2id:
            intent2id[example['intent']] = len(intent2id)

    # No set() since we want to compute the cutoff
    words = sum([x['utterance'].split() for x in train_raw], [])  # sum(list[list], []) -> from list of list to list

    # We do not want unk labels (slots),
    # however this depends on the research purpose
    corpus = train_raw + dev_raw + test_raw

    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])

    # words are only from te training set
    # labels from the whole corpus (we do not want unk labels)
    lang = Lang(words, intents, slots, cutoff=0)
    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=collate_fn, shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=eval_batch_size, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=eval_batch_size, collate_fn=collate_fn)

    vocab_len = len(lang.word2id)
    slots_len = len(lang.id2slot)  # pad and cls have the same id
    n_intents = len(lang.intent2id)

    print(f"Using device: {DEVICE}")

    model = GPT2(
        vocab_len,
        slots_len,
        n_intents,
        pos_emb_size=pos_emb_size,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    ).to(DEVICE)
    model.apply(init_weights)

    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()  # No pad tokens, all sequences have a single label for intent

    patience_counter = patience
    losses_train = []
    losses_dev = []
    dev_slot_f1s = []
    dev_intent_accs = []
    sampled_epochs = []
    best_f1 = 0
    best_intent_acc = 0
    best_model = None

    print("\nStarting Training Loop...")
    pbar = tqdm(range(n_epochs))
    for i in pbar:
        loss = train_loop(train_loader, optimizer, criterion_slots,
                           criterion_intents, model)
        sampled_epochs.append(i)
        losses_train.append(np.asarray(loss).mean())
        results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots,
                                                       criterion_intents, model, lang)

        f1 = results_dev['total']['f']
        intent_acc = intent_res['accuracy']
        losses_dev.append(np.asarray(loss_dev).mean())
        dev_slot_f1s.append(f1)
        dev_intent_accs.append(intent_acc)

        pbar.set_description(f"Slot F1: {f1:.2f}; Intent Acc: {intent_acc:.2f}")

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

    print("\n" + "-" * 30)
    print("Finalizing Experiment...")

    if best_model is None:
        best_model = copy.deepcopy(model)
    best_model.to(DEVICE)

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots,
                                              criterion_intents, best_model, lang)
    test_f1 = results_test['total']['f']
    test_acc = intent_test['accuracy']
    print('Slot F1: ', test_f1)
    print('Intent Accuracy:', test_acc)

    # Saving
    os.makedirs(f"bin/{exp_name}", exist_ok=True)
    torch.save(best_model.state_dict(), f"bin/{exp_name}/best_model.pt")
    torch.save(model.state_dict(), f"bin/{exp_name}/last_model.pt")

    # Tracker logging
    tracker = ExperimentTracker("NLU/part_A/results")
    tracker.log(
        name=exp_name,
        phase=phase,
        learning_rate=lr,
        test_slot_f1=test_f1,
        test_intent_acc=test_acc,
        dev_slot_f1=best_f1,
        dev_intent_acc=best_intent_acc,
        sampled_epochs=sampled_epochs,
        losses_train=losses_train,
        losses_dev=losses_dev,
        dev_slot_f1s=dev_slot_f1s,
        dev_intent_accs=dev_intent_accs,
        d_model=d_model,
        n_heads=n_heads,
        num_layers=num_layers,
        ff_dim=ff_dim,
        dropout=dropout,
    )

    print("\n" + "=" * 50)
    print(f"EXPERIMENT COMPLETE: {exp_name}")
    print(f"Best Dev Slot F1: {best_f1:.4f}  |  Best Dev Intent Acc: {best_intent_acc:.4f}")
    print(f"Test Slot F1: {test_f1:.4f}  |  Test Intent Acc: {test_acc:.4f}")
    print(f"Models saved in: bin/{exp_name}/")
    print("=" * 50 + "\n")

    tracker.summary()
    tracker.plot_curves(tracker.experiments[-1])
    tracker.plot_metrics()
    tracker.export_csv()


if __name__ == "__main__":

    # --- Step 0: Baseline LR search (architecture fixed at d_model=20, n_heads=1, num_layers=1, ff_dim=20) ---
    run_experiment(
        exp_name="Baseline lr=1e-3",
        phase=0,
        lr=1e-3,
        d_model=20,
        n_heads=1,
        num_layers=1,
        ff_dim=20,
        dropout=0.0,
        n_epochs=200,
    )

    # --- Step 1: Hyperparameter optimization (change d_model, n_heads, num_layers, ff_dim one at a time) ---
    # run_experiment(
    #     exp_name="Hyperparameter Tuning, d_model=64,ff_dim=256,n_heads=4,num_layers=2, lr=4e-4",
    #     phase=1,
    #     lr=4e-4,
    #     d_model=64,
    #     n_heads=4,
    #     num_layers=2,
    #     ff_dim=256,
    #     dropout=0.0,
    #     n_epochs=200,
    # )

    # --- Step 2: Dropout before the final output layers ---
    # run_experiment(
    #     exp_name="Dropout, d_model=64,ff_dim=256,n_heads=4,num_layers=2, lr=4e-4, dropout=0.1",
    #     phase=2,
    #     lr=4e-4,
    #     d_model=64,
    #     n_heads=4,
    #     num_layers=2,
    #     ff_dim=256,
    #     dropout=0.1,
    #     n_epochs=200,
    # )
