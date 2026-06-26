# Add functions or classes used for data loading and preprocessing
import json
from collections import Counter
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader



PAD_TOKEN = 0
DEVICE = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU


def load_data(path):
    '''
        input: path/to/data
        output: json
    '''
    dataset = []
    with open(path) as f:
        dataset = json.loads(f.read())
    return dataset


class Lang():
    def __init__(self, words, intents, slots, cutoff=0, cls=True):
        self.word2id = self.w2id(words, cutoff=cutoff, unk=True, cls=cls)
        self.slot2id = self.lab2id(slots, cls=cls)
        self.intent2id = self.lab2id(intents, pad=False, cls=False)
        self.id2word = {v:k for k, v in self.word2id.items()}
        # cls will have the same id as the pad token
        self.id2slot = {v:k for k, v in self.slot2id.items() if not cls or k != 'cls'}
        self.id2intent = {v:k for k, v in self.intent2id.items()}
        
    def w2id(self, elements, cutoff=None, unk=True, cls=True):
        vocab = {'pad': PAD_TOKEN}
        if unk:
            vocab['unk'] = len(vocab)
        if cls:
            vocab['cls'] = len(vocab)
        count = Counter(elements)
        for k, v in count.items():
            if v > cutoff:
                vocab[k] = len(vocab)
        return vocab
    
    def lab2id(self, elements, pad=True, cls=True):
        vocab = {}
        if pad:
            vocab['pad'] = PAD_TOKEN
        for elem in elements:
            vocab[elem] = len(vocab)
        if cls:
            # when predicting the slots, we want to ignore the CLS
            # CLS will only be used for intent classification
            vocab['cls'] = PAD_TOKEN
        return vocab



class IntentsAndSlots(data.Dataset):
    # Mandatory methods are __init__, __len__ and __getitem__
    def __init__(self, dataset, lang, unk='unk', cls='cls', add_cls=True):
        self.utterances = []
        self.intents = []
        self.slots = []
        self.unk = unk
        self.cls = cls
        self.add_cls = add_cls
        
        for x in dataset:
            self.utterances.append(x['utterance'])
            self.slots.append(x['slots'])
            self.intents.append(x['intent'])

        self.utt_ids = self.mapping_seq(self.utterances, lang.word2id)
        self.slot_ids = self.mapping_seq(self.slots, lang.slot2id)
        self.intent_ids = self.mapping_lab(self.intents, lang.intent2id)

    def __len__(self):
        return len(self.utterances)

    def __getitem__(self, idx):
        utt = torch.Tensor(self.utt_ids[idx])
        slots = torch.Tensor(self.slot_ids[idx])
        intent = self.intent_ids[idx]
        sample = {'utterance': utt, 'slots': slots, 'intent': intent}
        return sample
    
    # Auxiliary methods
    
    def mapping_lab(self, data, mapper):
        return [mapper[x] if x in mapper else mapper[self.unk] for x in data]
    
    def mapping_seq(self, data, mapper): # Map sequences to number
        res = []
        for seq in data:
            tmp_seq = []
            for x in seq.split():
                if x in mapper:
                    tmp_seq.append(mapper[x])
                else:
                    tmp_seq.append(mapper[self.unk])
            if self.add_cls:
                tmp_seq.append(mapper[self.cls])
            res.append(tmp_seq)
        return res
    


def collate_fn(data):
    def merge(sequences):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        max_len = 1 if max(lengths)==0 else max(lengths)
        # Pad token is zero in our case
        # So we create a matrix full of PAD_TOKEN (i.e. 0) with the shape 
        # batch_size * maximum length of a sequence
        padded_seqs = torch.LongTensor(len(sequences),max_len).fill_(PAD_TOKEN)
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq # We copy each sequence into the matrix
        return padded_seqs, lengths

    data_by_key = {}
    for key in data[0].keys():
        data_by_key[key] = [d[key] for d in data]
        
    # We just need one length for packed pad seq, since len(utt) == len(slots)
    src_utt, _ = merge(data_by_key['utterance'])
    y_slots, y_lengths = merge(data_by_key["slots"])
    intent = torch.LongTensor(data_by_key["intent"])
    
    src_utt = src_utt.to(DEVICE) # We load the Tensor on our selected device
    y_slots = y_slots.to(DEVICE)
    intent = intent.to(DEVICE)
    y_lengths = torch.LongTensor(y_lengths).to(DEVICE)
    
    new_item = {}
    new_item["utterances"] = src_utt
    new_item["intents"] = intent
    new_item["y_slots"] = y_slots
    new_item["slots_len"] = y_lengths
    return new_item

