import random
from math import log2

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from tqdm import trange

from utils import (
    tokenize,
    binarize_setup,
    binarize_next,
    detokenize,
    start_model,
    PRF,
    compute_score_function,
    normalize_score,
    consistent_perm,
    apply_perm,
)


def keyed_rng(token_array, key):
    """
    Numerical Recipes ranqd1, Chapter 7.1, An Even Quicker Generator, Eq. 7.1.6
    parameters from Knuth and H. W. Lewis
    """
    INC = 1013904223
    MUL = 1664525
    rng = np.random.default_rng(hash(key) % (2**32))
    as_ints = token_array.astype(np.int32)
    as_ints *= rng.integers(1, 2**32)
    as_ints += rng.integers(1, 2**32)
    for _ in range(5):
        as_ints *= MUL
        as_ints += INC
    return abs(as_ints.astype(float)) / (2**31 + 1)


class GumbalWatermark:
    def __init__(
        self,
        security: float,
        num_tokens_per_draw: int,
        max_seed_length: int,
        base_key: str,
        model,
        tokenizer,
    ):
        self.num_tokens_per_draw = num_tokens_per_draw
        self.max_seed_length = max_seed_length
        self.base_key = base_key
        self.security = security
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, length: int, encode_random=False) -> str:
        prompt = tokenize(prompt, self.tokenizer)
        inputs = prompt.to(self.model.device)
        attn = torch.ones_like(inputs)
        blen, token_to_id, id_to_token = binarize_setup(self.tokenizer)
        past = None
        vocab_size = len(self.tokenizer)
        curr_entropy = 0
        curr_biased_token = 0
        curr_key = self.base_key
        # perm, inv_perm = consistent_perm(
        #     self.base_key, len(tokenizer)
        # )  # Not necessary, but makes the token indices spread uniformly.
        key_len = 0
        for i in range(length):
            with torch.no_grad():
                if past:
                    output = model(
                        inputs[:, -1:], past_key_values=past, attention_mask=attn
                    )
                else:
                    output = model(inputs)

            probs = torch.nn.functional.softmax(
                output.logits[:, -1, :vocab_size], dim=-1
            ).cpu()[0, :]
            # probs = apply_perm(probs, inv_perm)
            token_id = 0
            # Switch states only between tokens.
            if encode_random or curr_entropy < self.security:
                # Collect entropy.
                for ind in range(blen):
                    p0, p1 = binarize_next(probs, ind, blen, token_id)
                    p1_hat = p1 / (p0 + p1)
                    token_id = token_id << 1
                    if random.random() < p1_hat:
                        curr_entropy += p1_hat * -log2(p1_hat)
                        token_id += 1
                    else:
                        curr_entropy += (1 - p1_hat) * -log2(1 - p1_hat)
                curr_key += str(token_id)
                key_len += 1
            elif curr_biased_token < self.num_tokens_per_draw:
                token_hashes = keyed_rng(np.arange(len(probs)), PRF(curr_key, [i]))
                token_id = (token_hashes ** (1 / probs.numpy())).argmax()
            if (
                curr_biased_token == self.num_tokens_per_draw
                or key_len > self.max_seed_length
            ):
                curr_key = self.base_key
                curr_entropy = 0
                key_len = 0
            token = torch.tensor([[token_id]])
            # token = torch.tensor([[inv_perm[token_id]]])
            inputs = torch.cat([inputs, token], dim=-1)

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)

        return detokenize(inputs.detach().cpu()[0], tokenizer)

    def calculate_scores(self, text: str) -> list[float]:
        raise NotImplementedError()
        blen, token_to_id, id_to_token = binarize_setup(self.tokenizer)
        tokens = tokenize(text, self.tokenizer)[0]
        scores = []
        for key_len in range(1, self.max_seed_length + 1):
            for i in range(len(tokens) - key_len - self.num_tokens_per_draw):
                split_score = 0
                key = self.base_key + "".join(
                    str(t.item()) for t in tokens[i : i + key_len]
                )
                generated = tokens[i + key_len : i + key_len + self.num_tokens_per_draw]
                for word_ind in range(len(generated)):
                    token_bits = ("0" * blen + bin(generated[word_ind])[2:])[-blen:]
                    for bit_ind in range(blen):
                        split_score += compute_score_function(
                            key, [word_ind, bit_ind], token_bits[bit_ind]
                        )
                s = normalize_score(split_score, blen * self.num_tokens_per_draw)
                scores.append(
                    normalize_score(split_score, blen * self.num_tokens_per_draw)
                )
        return scores

    def calculate_entropyless_score(
        self, text: str, override_key: float = None
    ) -> float:
        """Assumes no entropy"""
        tokens = tokenize(text, self.tokenizer)[0]
        hashed_tokens = []
        for i, token in enumerate(tokens):
            hashed_tokens.append(
                keyed_rng(np.array([token]), PRF(override_key or self.base_key, [i]))[0]
            )
        return np.log(1 / (1 - np.array(hashed_tokens))).sum()


if __name__ == "__main__":
    model, tokenizer = start_model("gpt2")
    # model, tokenizer = start_model("meta-llama/Llama-2-7b-chat-hf")
    prompts = [  # Taken from the GPT-2 official example prompts https://openai.com/research/better-language-models
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
        "A train carriage containing controlled nuclear materials was stolen in Cincinnati today. Its whereabouts are unknown.",
        "Miley Cyrus was caught shoplifting from Abercrombie and Fitch on Hollywood Boulevard today.",
        "Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry.",
        "For today's homework assignment, please describe the reasons for the US Civil War.",
        "John F. Kennedy was just elected President of the United States after rising from the grave decades after his assassination. Due to miraculous developments in nanotechnology, Kennedy's brain was rebuilt from his remains and installed in the control center of a state-of-the art humanoid robot. Below is a transcript of his acceptance speech.",
    ]
    key = random.random()
    watermarker = GumbalWatermark(0, 1000, 4, str(key), model, tokenizer)
    response_sizes = [10, 20, 50, 100]
    samples_per_size = 200  # Set to 10 for a quicker run
    for size in response_sizes:
        watermarked_scores = []
        regular_scores = []
        print("Making samples of size " + str(size) + ":")
        for i in trange(samples_per_size, desc=f"{size=}"):
            watermarker.base_key = random.random()
            prompt = random.choice(prompts)
            res = watermarker.generate(prompt=prompt, length=size)
            watermarked_scores.append(
                watermarker.calculate_entropyless_score(res[len(prompt) :])
            )
            regular_scores.append(
                watermarker.calculate_entropyless_score(
                    res[len(prompt) :], random.random()
                )
            )
        plt.cla()
        plt.hist(watermarked_scores, color="g", alpha=0.5, bins=30)
        plt.hist(regular_scores, color="r", alpha=0.5, bins=30)
        plt.title(f"results for {size=}")
        plt.show()
        plt.cla()
        fpr, tpr, thresholds = roc_curve(
            [0] * samples_per_size + [1] * samples_per_size,
            regular_scores + watermarked_scores,
        )
        plt.plot(fpr, tpr)
        plt.title(f"roc for {size=}")
        plt.show()
        print(f"{regular_scores=}\n{watermarked_scores=}")
