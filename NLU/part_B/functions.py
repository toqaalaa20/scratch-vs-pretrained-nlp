import os
import re

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report

from utils import IGNORE_INDEX


# ── CoNLL slot-F1 evaluation (modified version of https://pypi.org/project/conlleval/) ──

def stats():
    return {'cor': 0, 'hyp': 0, 'ref': 0}


def evaluate(ref, hyp, otag='O'):
    # evaluation for NLTK
    aligned = align_hyp(ref, hyp)
    return conlleval(aligned, otag=otag)


def align_hyp(ref, hyp):
    # align references and hypotheses for evaluation
    # add last element of token tuple in hyp to ref
    if len(ref) != len(hyp):
        raise ValueError("Size Mismatch: ref: {} & hyp: {}".format(len(ref), len(hyp)))

    out = []
    for i in range(len(ref)):
        if len(ref[i]) != len(hyp[i]):
            raise ValueError("Size Mismatch: ref: {} & hyp: {}".format(len(ref), len(hyp)))
        out.append([(*ref[i][j], hyp[i][j][-1]) for j in range(len(ref[i]))])
    return out


def conlleval(data, otag='O'):
    # token, segment & class level counts for TP, TP+FP, TP+FN
    tok = stats()
    seg = stats()
    cls = {}

    for sent in data:

        prev_ref = otag      # previous reference label
        prev_hyp = otag      # previous hypothesis label
        prev_ref_iob = None  # previous reference label IOB
        prev_hyp_iob = None  # previous hypothesis label IOB

        in_correct = False  # currently processed chunks is correct until now

        for token in sent:

            hyp_iob, hyp = parse_iob(token[-1])
            ref_iob, ref = parse_iob(token[-2])

            ref_e = is_eoc(ref, ref_iob, prev_ref, prev_ref_iob, otag)
            hyp_e = is_eoc(hyp, hyp_iob, prev_hyp, prev_hyp_iob, otag)

            ref_b = is_boc(ref, ref_iob, prev_ref, prev_ref_iob, otag)
            hyp_b = is_boc(hyp, hyp_iob, prev_hyp, prev_hyp_iob, otag)

            if not cls.get(ref) and ref:
                cls[ref] = stats()

            if not cls.get(hyp) and hyp:
                cls[hyp] = stats()

            # segment-level counts
            if in_correct:
                if ref_e and hyp_e and prev_hyp == prev_ref:
                    in_correct = False
                    seg['cor'] += 1
                    cls[prev_ref]['cor'] += 1

                elif ref_e != hyp_e or hyp != ref:
                    in_correct = False

            if ref_b and hyp_b and hyp == ref:
                in_correct = True

            if ref_b:
                seg['ref'] += 1
                cls[ref]['ref'] += 1

            if hyp_b:
                seg['hyp'] += 1
                cls[hyp]['hyp'] += 1

            # token-level counts
            if ref == hyp and ref_iob == hyp_iob:
                tok['cor'] += 1

            tok['ref'] += 1

            prev_ref = ref
            prev_hyp = hyp
            prev_ref_iob = ref_iob
            prev_hyp_iob = hyp_iob

        if in_correct:
            seg['cor'] += 1
            cls[prev_ref]['cor'] += 1

    return summarize(seg, cls)


def parse_iob(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, None)


def is_boc(lbl, iob, prev_lbl, prev_iob, otag='O'):
    """
    is beginning of a chunk

    supports: IOB, IOBE, BILOU schemes
        - {E,L} --> last
        - {S,U} --> unit

    :param lbl: current label
    :param iob: current iob
    :param prev_lbl: previous label
    :param prev_iob: previous iob
    :param otag: out-of-chunk label
    :return:
    """
    boc = False

    boc = True if iob in ['B', 'S', 'U'] else boc
    boc = True if iob in ['E', 'L'] and prev_iob in ['E', 'L', 'S', otag] else boc
    boc = True if iob == 'I' and prev_iob in ['S', 'L', 'E', otag] else boc

    boc = True if lbl != prev_lbl and iob != otag and iob != '.' else boc

    # these chunks are assumed to have length 1
    boc = True if iob in ['[', ']'] else boc

    return boc


def is_eoc(lbl, iob, prev_lbl, prev_iob, otag='O'):
    """
    is end of a chunk

    supports: IOB, IOBE, BILOU schemes
        - {E,L} --> last
        - {S,U} --> unit

    :param lbl: current label
    :param iob: current iob
    :param prev_lbl: previous label
    :param prev_iob: previous iob
    :param otag: out-of-chunk label
    :return:
    """
    eoc = False

    eoc = True if iob in ['E', 'L', 'S', 'U'] else eoc
    eoc = True if iob == 'B' and prev_iob in ['B', 'I'] else eoc
    eoc = True if iob in ['S', 'U'] and prev_iob in ['B', 'I'] else eoc

    eoc = True if iob == otag and prev_iob in ['B', 'I'] else eoc

    eoc = True if lbl != prev_lbl and iob != otag and prev_iob != '.' else eoc

    # these chunks are assumed to have length 1
    eoc = True if iob in ['[', ']'] else eoc

    return eoc


def score(cor_cnt, hyp_cnt, ref_cnt):
    # precision
    p = 1 if hyp_cnt == 0 else cor_cnt / hyp_cnt
    # recall
    r = 0 if ref_cnt == 0 else cor_cnt / ref_cnt
    # f-measure (f1)
    f = 0 if p + r == 0 else (2 * p * r) / (p + r)
    return {"p": p, "r": r, "f": f, "s": ref_cnt}


def summarize(seg, cls):
    # class-level
    res = {lbl: score(cls[lbl]['cor'], cls[lbl]['hyp'], cls[lbl]['ref']) for lbl in set(cls.keys())}
    # micro
    res.update({"total": score(seg.get('cor', 0), seg.get('hyp', 0), seg.get('ref', 0))})
    return res


# ── train / eval loops ──────────────────────────────────────────────────────

def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
    """Run one training epoch (joint slot + intent loss, summed). Returns the list of per-batch losses."""
    model.train()
    loss_array = []

    for batch in data:
        optimizer.zero_grad()

        slots, intent = model(batch['input_ids'], batch['attention_mask'])
        slots = slots.permute(0, 2, 1)  # (B, n_slots, L), required by CrossEntropyLoss

        loss_intent = criterion_intents(intent, batch['intent_id'])
        loss_slot = criterion_slots(slots, batch['slot_label_ids'])
        loss = loss_intent + loss_slot  # joint training: unweighted sum
        loss_array.append(loss.item())
        loss.backward()
        optimizer.step()

    return loss_array


def eval_loop(data, criterion_slots, criterion_intents, model, id2slot, id2intent):
    """Evaluate the model on `data` with gradients disabled.

    Returns:
        results: CoNLL slot evaluation dict (results['total']['f'] is the slot F1).
        report_intent: sklearn classification_report dict (report_intent['accuracy'] is the intent accuracy).
        loss_array: list of per-batch losses.
    """
    model.eval()
    loss_array = []

    ref_intents = []
    hyp_intents = []

    ref_slots = []
    hyp_slots = []
    with torch.no_grad():
        for batch in data:
            slots, intents = model(batch['input_ids'], batch['attention_mask'])
            slots_for_loss = slots.permute(0, 2, 1)
            loss_intent = criterion_intents(intents, batch['intent_id'])
            loss_slot = criterion_slots(slots_for_loss, batch['slot_label_ids'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            # Intent inference
            out_intents = [id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [id2intent[x] for x in batch['intent_id'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            # Slot inference: only first-subtoken positions (slot_label_ids != IGNORE_INDEX)
            # carry a real label, so word-level sequences are reconstructed from those only.
            pred_slot_ids = torch.argmax(slots, dim=2).tolist()  # (B, L)
            gt_slot_ids = batch['slot_label_ids'].tolist()
            for seq_idx, words in enumerate(batch['words']):
                gt_seq, hyp_seq = [], []
                word_idx = 0
                for pos, label_id in enumerate(gt_slot_ids[seq_idx]):
                    if label_id == IGNORE_INDEX:
                        continue
                    word = words[word_idx]
                    word_idx += 1
                    gt_seq.append((word, id2slot[label_id]))
                    hyp_seq.append((word, id2slot[pred_slot_ids[seq_idx][pos]]))
                ref_slots.append(gt_seq)
                hyp_slots.append(hyp_seq)
    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        # Sometimes the model predicts a class that is not in REF
        print("Warning:", ex)
        ref_s = set(elem[1] for seq in ref_slots for elem in seq)
        hyp_s = set(elem[1] for seq in hyp_slots for elem in seq)
        print(hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    report_intent = classification_report(ref_intents, hyp_intents,
                                           zero_division=False, output_dict=True)
    return results, report_intent, loss_array


# ── reporting ────────────────────────────────────────────────────────────────

def plot_training_curves(exp_name, save_dir, sampled_epochs, losses_train, losses_dev, dev_slot_f1s=None, dev_intent_accs=None):
    """Plot train/dev loss (and dev slot F1 / intent acc, if given) over epochs and save the figure to save_dir."""
    os.makedirs(save_dir, exist_ok=True)
    has_metrics = dev_slot_f1s is not None and dev_intent_accs is not None
    fig, axes = plt.subplots(1, 2 if has_metrics else 1, figsize=(13, 4))
    if not has_metrics:
        axes = [axes]

    axes[0].plot(sampled_epochs, losses_train, label="Train loss", color="#185FA5", linewidth=1.6)
    axes[0].plot(sampled_epochs, losses_dev, label="Dev loss", color="#D85A30", linewidth=1.6, linestyle="--")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    if has_metrics:
        axes[1].plot(sampled_epochs, dev_slot_f1s, label="Dev slot F1", color="#185FA5", linewidth=1.6, marker="o", markersize=3)
        axes[1].plot(sampled_epochs, dev_intent_accs, label="Dev intent acc", color="#D85A30", linewidth=1.6, linestyle="--", marker="s", markersize=3)
        axes[1].set_title("Dev metrics"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Score")
        axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    fig.suptitle(exp_name, fontsize=11)
    plt.tight_layout()
    safe_name = exp_name.replace(" ", "_").replace("/", "-")
    path = os.path.join(save_dir, f"curves_{safe_name}.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {path}")


def print_results(exp_name, **metrics):
    """Print a final summary of the experiment's metrics."""
    print("\n" + "=" * 50)
    print(f"RESULTS: {exp_name}")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    print("=" * 50 + "\n")
