import random
import math

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def PRF(key, input):
    # Lazy and insecure implementation, replace with a provably secure PRF for real applications
    old_state = random.getstate()
    random.seed(str(key) + "||" + str(input))
    res = random.random()
    random.setstate(old_state)
    return res


def consistent_perm(key, n):
    perm = np.arange(n)
    rng = np.random.default_rng(hash(key) % (2**32))
    rng.shuffle(perm)
    inv_perm = [0 for _ in range(n)]
    for i in range(n):
        inv_perm[perm[i]] = i
    return perm, inv_perm


def apply_perm(vector, inv_perm):
    assert len(vector) == len(inv_perm)
    result = vector[inv_perm].clone().detach()
    return result


def start_model(model_name="gpt2"):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    return model, tokenizer


def tokenize(prompt, tokenizer):
    return tokenizer.encode(
        prompt, return_tensors="pt", truncation=True, max_length=2048
    )


def detokenize(tokenized, tokenizer):
    return tokenizer.decode(tokenized, skip_special_tokens=True)


def binarize_setup(tokenizer):
    blen = math.ceil(math.log2(len(tokenizer)))
    token_to_id = tokenizer.get_vocab()
    id_to_token = {v: k for (k, v) in token_to_id.items()}
    return blen, token_to_id, id_to_token


def binarize_next(probs, ind=0, blen=16, prefix=0):
    relevant_probs = probs[
        prefix << (blen - ind) : min((prefix + 1) << (blen - ind), len(probs))
    ]
    indecies = torch.arange(
        prefix << (blen - ind),
        min((prefix + 1) << (blen - ind), len(probs)),
        dtype=torch.int32,
    )
    is_for_p0 = (indecies >> (blen - ind - 1)) % 2 == 0
    p0 = torch.tensor([relevant_probs[is_for_p0].sum()])
    p1 = torch.tensor([relevant_probs[~is_for_p0].sum()])
    return p0, p1


def normalize_score(score, length):
    return (score - length) / math.sqrt(length)


def compute_score_function(key, prf_input, bit):
    u = PRF(key, prf_input)
    v = u if bit == "1" else (1 - u)
    return -math.log(v)
