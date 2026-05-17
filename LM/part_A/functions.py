import math
from tqdm import tqdm
import torch



def train_loop(data, optimizer, criterion, model):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    pbar = tqdm(data, desc="Train", unit="batch", total=len(data), leave=False, dynamic_ncols=True)

    for i, (input_ids, labels, n_tokens) in enumerate(pbar):
        optimizer.zero_grad() # Zeroing the gradient
        output = model(input_ids)
        # need to reshape as (B, vocab, L)
        loss = criterion(output.permute(0,2,1), labels)
        loss_array.append(loss.item() * n_tokens)
        number_of_tokens.append(n_tokens)
        loss.backward() # Compute the gradient, deleting the computational graph
        optimizer.step() # Update the weights

        if i % 100 == 0:
            pbar.set_postfix({"loss": f"{(sum(loss_array)/sum(number_of_tokens)).item():.4f}"})

    loss_to_return = sum(loss_array)/sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for input_ids, labels, n_tokens in tqdm(data, desc="Eval", unit="batch", total=len(data), leave=False, dynamic_ncols=True):
            output = model(input_ids)
            # need to reshape as (B, vocab, L)
            loss = eval_criterion(output.permute(0,2,1), labels)
            loss_array.append(loss.item() * n_tokens)
            number_of_tokens.append(n_tokens)
            
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    ppl = math.exp(loss_to_return)
    return ppl, loss_to_return





    