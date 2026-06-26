import math

from tqdm import tqdm
import torch


def train_loop(data, optimizer, model, tokenizer, max_grad_norm=1.0):
    model.train()
    loss_array = []
    number_of_tokens = []
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    pbar = tqdm(data, desc="Train", unit="batch", total=len(data), leave=False, dynamic_ncols=True)

    for i, (input_ids, attention_mask, n_tokens) in enumerate(pbar):
        optimizer.zero_grad()  # Zeroing the gradient
        # we don't shift the labels to the left, the model manages it internally
        labels = input_ids.clone().detach()
        # we cannot specify ignore_index, so we replace our pad tokens with -100
        # -100 is ignored by default when the model computes the loss
        labels[labels == tokenizer.pad_token_id] = -100
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss_array.append(output.loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        output.loss.backward()  # Compute the gradient, deleting the computational graph
        # clip gradients to avoid early-training blowups destabilizing the adapters
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()  # Update the weights

        if i % 100 == 0:
            pbar.set_postfix({"loss": f"{(sum(loss_array)/sum(number_of_tokens)).item():.4f}"})

    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return


def eval_loop(data, model, tokenizer):
    model.eval()
    loss_array = []
    number_of_tokens = []
    with torch.no_grad():  # It is used to avoid the creation of the computational graph
        for input_ids, attention_mask, n_tokens in tqdm(data, desc="Eval", unit="batch", total=len(data), leave=False, dynamic_ncols=True):
            labels = input_ids.clone().detach()
            labels[labels == tokenizer.pad_token_id] = -100
            output = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss_array.append(output.loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)

    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return
