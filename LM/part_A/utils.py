"""Data loading and preprocessing utilities for Penn Treebank dataset."""

from typing import List

import torch
import torch.utils.data as data


def read_file(path: str, eos_token: str = "<eos>") -> List[str]:
    """Load and preprocess text file.
    
    Args:
        path: Path to text file
        eos_token: End-of-sequence token to append
        
    Returns:
        List of preprocessed sentences
    """
    output = []
    with open(path, "r") as f:
        for line in f.readlines():
            output.append(line.strip() + " " + eos_token)
    return output



class PennTreeBank(data.Dataset):
    """Penn Treebank dataset for language modeling."""
    
    def __init__(self, corpus: List[str]):
        """Initialize dataset.
        
        Args:
            corpus: List of sentences
        """
        self.sents = corpus

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.sents)

    def __getitem__(self, idx: int) -> str:
        """Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Sentence at index
        """
        return self.sents[idx]
    


def collate_fn(
    batch: List[str],
    tokenizer,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate batch of sentences for training.
    
    Args:
        batch: List of sentences
        tokenizer: Tokenizer instance
        device: Device to place tensors on
        
    Returns:
        Tuple of (input_ids, labels, n_tokens)
        - input_ids: Token indices of shape (B, L-1)
        - labels: Target tokens (input shifted right) of shape (B, L-1)
        - n_tokens: Count of non-padding tokens
    """
    # Tokenize and pad batch
    tokenized = tokenizer(batch, padding=True, return_tensors="pt")

    # Input is all tokens except the last, labels are all tokens except the first
    # This creates pairs for next-token prediction
    input_ids = tokenized.input_ids[:, :-1].detach().clone().to(device)
    labels = tokenized.input_ids[:, 1:].detach().clone().to(device)

    # Count actual (non-padding) tokens for normalized loss computation
    n_tokens = torch.sum(input_ids != tokenizer.pad_token_id)

    return input_ids, labels, n_tokens

