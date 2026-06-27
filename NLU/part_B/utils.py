# Add functions or classes used for data loading and preprocessing
import json
from collections import Counter

import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split

IGNORE_INDEX = -100  # nn.CrossEntropyLoss default ignore_index: padding + non-first subtokens


def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    with open(path) as f:
        return json.loads(f.read())


def load_atis_splits(train_path, test_path, dev_portion=0.10, random_state=42):
    '''Stratified (on intent) train/dev split + test set, same logic as Part A.'''
    tmp_train_raw = load_data(train_path)
    test_raw = load_data(test_path)

    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    labels, inputs, mini_train = [], [], []
    for id_y, y in enumerate(intents):
        if count_y[y] > 1:  # intents occurring only once can't be stratified
            inputs.append(tmp_train_raw[id_y])
            labels.append(y)
        else:
            mini_train.append(tmp_train_raw[id_y])

    X_train, X_dev, _, _ = train_test_split(
        inputs, labels, test_size=dev_portion,
        random_state=random_state, shuffle=True, stratify=labels,
    )
    X_train.extend(mini_train)
    return X_train, X_dev, test_raw


def build_label_vocabs(train_raw, dev_raw, test_raw):
    '''slot2id/intent2id built from train+dev+test (no unk labels), no word2id
    needed since the pretrained tokenizer handles words.'''
    corpus = train_raw + dev_raw + test_raw
    slots = sorted(set(sum([x['slots'].split() for x in corpus], [])))
    intents = sorted(set(x['intent'] for x in corpus))

    slot2id = {s: i for i, s in enumerate(slots)}
    intent2id = {x: i for i, x in enumerate(intents)}
    id2slot = {i: s for s, i in slot2id.items()}
    id2intent = {i: x for x, i in intent2id.items()}
    return slot2id, id2slot, intent2id, id2intent


class JointATISDataset(data.Dataset):
    def __init__(self, raw_data, tokenizer, slot2id, intent2id, model_type, max_length=64):
        self.examples = raw_data
        self.tokenizer = tokenizer
        self.slot2id = slot2id
        self.intent2id = intent2id
        self.model_type = model_type
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        words = example['utterance'].split()
        slot_labels = example['slots'].split()
        intent_id = self.intent2id[example['intent']]

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']
        word_ids = encoding.word_ids()

        if self.model_type == 'gpt2':
            # GPT2's fast tokenizer adds no special tokens by default, so there is
            # no token that has attended to the whole (causal) sequence. Append eos
            # manually as the pooling position, mirroring Part A's "cls at the end".
            input_ids = input_ids + [self.tokenizer.eos_token_id]
            attention_mask = attention_mask + [1]
            word_ids = word_ids + [None]

        # First subtoken of each word gets the real slot label; every continuation
        # subtoken and every special-token position (word_id is None) is ignored.
        slot_label_ids = []
        prev_word_id = None
        for w_id in word_ids:
            if w_id is None or w_id == prev_word_id:
                slot_label_ids.append(IGNORE_INDEX)
            else:
                slot_label_ids.append(self.slot2id[slot_labels[w_id]])
            prev_word_id = w_id

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'slot_label_ids': slot_label_ids,
            'intent_id': intent_id,
            'words': words,
            'word_ids': word_ids,
        }


def collate_fn(batch, tokenizer, device):
    max_len = max(len(item['input_ids']) for item in batch)
    pad_id = tokenizer.pad_token_id

    input_ids, attention_mask, slot_label_ids = [], [], []
    for item in batch:
        pad_len = max_len - len(item['input_ids'])
        input_ids.append(item['input_ids'] + [pad_id] * pad_len)
        attention_mask.append(item['attention_mask'] + [0] * pad_len)
        slot_label_ids.append(item['slot_label_ids'] + [IGNORE_INDEX] * pad_len)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long, device=device),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long, device=device),
        'slot_label_ids': torch.tensor(slot_label_ids, dtype=torch.long, device=device),
        'intent_id': torch.tensor([item['intent_id'] for item in batch], dtype=torch.long, device=device),
        # kept as plain Python lists (unpadded) for word-level reconstruction during eval
        'words': [item['words'] for item in batch],
        'word_ids': [item['word_ids'] for item in batch],
    }
