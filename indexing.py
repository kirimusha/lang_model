import torch
import random

def encode(s: list, vocab: list) -> torch.tensor:
    """
    Encode a list of tokens into a tensor of integers, given a fixed vocabulary. 
    When a token is not found in the vocabulary, the special unknown token is assigned. 
    When the training set did not use that special token, a random token is assigned.
    """
    rand_token = random.randint(0, len(vocab))

    unknown_token = "<UNK>"

    map = {s:i for i,s in enumerate(vocab)}
    enc = [map.get(c, map.get(unknown_token, rand_token)) for c in s]
    enc = torch.tensor(enc, dtype=torch.long)
    return enc