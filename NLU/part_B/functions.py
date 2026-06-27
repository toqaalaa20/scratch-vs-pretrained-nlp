import torch
from conll import evaluate
from sklearn.metrics import classification_report
from utils import IGNORE_INDEX


def train_loop(data, optimizer, criterion_slots, criterion_intents, model):
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
