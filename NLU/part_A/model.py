# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(mat):
    """Initialize every nn.Linear submodule with small uniform weights and a constant bias."""
    for m in mat.modules():
        if type(m) in [nn.Linear]:
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias != None:
                m.bias.data.fill_(0.01)


class MultiHeadAttention(nn.Module):
    """Causal multi-head self-attention (the masking is applied by the caller via `mask`)."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.h_dim = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        # batch size B, sequence length L, d_model
        B, L, d_model = x.size() 

        q = self.w_q(x) # (B, L, d_model)
        k = self.w_k(x) # (B, L, d_model)
        v = self.w_v(x) # (B, L, d_model)

        # reshape to (B, n_heads, L, h_dim)
        q = q.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.h_dim).transpose(1, 2)

        # dot-product between each head's q and k matrices
        # (B, n_heads, L, h_dim) * (B, n_heads, h_dim, L) -> (B, n_heads, L, L), i.e., similarities for each head
        similarity = q @ k.transpose(-2, -1) # transpose(-2,-1) transposes the k matrix for each head

        # normalize
        similarity = similarity * (1 / torch.sqrt(torch.tensor(self.h_dim)))

        # mask
        similarity = similarity.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(similarity, dim=-1)

        y = attn @ v  # (B, n_heads, L, L) * (B, n_heads, L, h_dim) -> (B, n_heads, L, h_dim)
        y = y.transpose(1, 2) # (B, L, n_heads, h_dim)
        # concatenate outputs of each head (contiguous is necessary to use view())
        y = y.contiguous().view(B, L, d_model) # (B, L, d_model)
        y = self.out_proj(y)

        return y

class FeedForward(nn.Module):
    """Position-wise two-layer MLP (Linear -> GELU -> Linear) applied independently to each token."""

    def __init__(self, d_model, hidden_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x):
        # applied over each token independently
        return self.net(x)
    
class TransformerBlock(nn.Module):
    """One decoder block: pre-LayerNorm self-attention + pre-LayerNorm feed-forward, both with residual connections."""

    def __init__(self, d_model, n_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, ff_dim, dropout)

    def forward(self, x, mask):
        # layer norm, attention, residual
        x = x + self.attn(self.ln1(x), mask)
        # layer norm, feedforward, residual
        x = x + self.ff(self.ln2(x))
        return x
    
class GPT2(nn.Module):
    """From-scratch GPT-2-style decoder-only backbone with two task heads on top: a token-level
    slot tagger and a sequence-level intent classifier (joint intent classification + slot filling)."""

    def __init__(
        self,
        vocab_size,
        slots_size,
        n_intents,
        # GPT2 default hyperparameters
        pos_emb_size=1024,
        d_model=768,
        n_heads=12,
        num_layers=12,
        ff_dim=3072,
        dropout=0.1,
    ):
        super().__init__()
        self.pos_emb_size = pos_emb_size

        # learnable token embeddings
        self.token_embed = nn.Embedding(vocab_size, d_model)
        # learnable positional embeddings
        self.pos_embed = nn.Embedding(pos_emb_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)

        #### These are different from the Language Modeling case of Part 1
        # slot outputs (sequence labeling): one slot label for each input token
        self.slot_out = nn.Linear(d_model, slots_size)
        # intent output (sequence classification): classyfy the intent of the input sequence
        self.intent_out = nn.Linear(d_model, n_intents)

        # triangular matrix: (sub)token i will only be able to attend (sub)tokens j, 0 <= j <= i
        mask = torch.tril(torch.ones(pos_emb_size, pos_emb_size)).unsqueeze(0).unsqueeze(0)
        # mask is not a learnable parameter, 
        # but needs to be moved with everything else when doing .to(device) 
        self.register_buffer("mask", mask)
    
    def forward(self, idx, seq_lens):
        """idx: (B, L) token ids, seq_lens: (B,) true length of each sequence (incl. the
        appended CLS token). Returns (slot_logits (B, L, slots_size), intent_logits (B, n_intents))."""
        B, L = idx.shape # batch size, sequence length
        # positional embedding at most for position self.pos_emb_size
        # longer sequences cannot be processed (the model never learned positional embeddings for positions greater than self.pos_emb_size)
        assert L <= self.pos_emb_size

        pos = torch.arange(L, device=idx.device)
        x = self.token_embed(idx) + self.pos_embed(pos)

        mask = self.mask[:, :, :L, :L]

        for block in self.blocks:
            x = block(x, mask)

        # We are also predicting a slot for CLS
        # but it will be ignored, as we have a pad token in the ground truth
        x = self.ln_f(x)
        slots = self.slot_out(x)

        # get intent from last token (CLS)
        tmp = []
        # for sequence i in the batch, 
        # get the vector associated with its CLS token 
        # (last of the sequence)
        for i in range(x.shape[0]):
            tmp.append(x[i, seq_lens[i]-1]) # -1, we count from 0
        cls_tokens = torch.stack(tmp)
        # get logits for intents
        intent = self.intent_out(cls_tokens)

        return slots, intent # ((B, L, slot_size), (B, n_intents)