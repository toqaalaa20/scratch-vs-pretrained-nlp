# Add the class of your model only
# Here is where you define the architecture of your model using pytorch
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


def load_tokenizer(checkpoint, model_type):
    # GPT2's fast BPE tokenizer requires add_prefix_space=True to tokenize
    # pre-split word lists (is_split_into_words=True) consistently with normal text.
    kwargs = {"add_prefix_space": True} if model_type == "gpt2" else {}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, **kwargs)
    if tokenizer.pad_token is None:  # GPT2 has no pad token by default
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class JointSeqModel(nn.Module):
    '''Joint intent classification + slot filling head on top of a pretrained
    encoder/decoder backbone. The only architectural difference between a
    causal model (GPT2) and a bidirectional one (BERT) is *where* the
    sentence-level representation lives:
      - "first": position 0 (BERT's native [CLS], which sees the whole
        sequence thanks to bidirectional attention).
      - "last": the last real (non-padding) token, found via attention_mask
        (GPT2's causal attention means only this position has seen the whole
        sentence; a pooling token -- eos -- is appended there at data-loading time).
    '''

    def __init__(self, backbone, hidden_size, n_slots, n_intents, pooling, dropout=0.1):
        super().__init__()
        assert pooling in ("first", "last")
        self.backbone = backbone
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.slot_out = nn.Linear(hidden_size, n_slots)
        self.intent_out = nn.Linear(hidden_size, n_intents)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = self.dropout(outputs.last_hidden_state)  # (B, L, H)

        slots = self.slot_out(hidden_states)  # (B, L, n_slots)

        if self.pooling == "first":
            pooled = hidden_states[:, 0]
        else:
            last_idx = attention_mask.sum(dim=1) - 1  # index of the last real token
            pooled = hidden_states[torch.arange(hidden_states.size(0)), last_idx]

        intent = self.intent_out(pooled)  # (B, n_intents)
        return slots, intent


def build_model(model_type, checkpoint, n_slots, n_intents, dropout=0.1):
    backbone = AutoModel.from_pretrained(checkpoint)
    if model_type == "gpt2":
        hidden_size = backbone.config.n_embd
        pooling = "last"
    elif model_type == "bert":
        hidden_size = backbone.config.hidden_size
        pooling = "first"
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return JointSeqModel(backbone, hidden_size, n_slots, n_intents, pooling, dropout)
