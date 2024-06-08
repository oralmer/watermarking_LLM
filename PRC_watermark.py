import random
from math import ceil, log, log2

import numpy as np
import torch

from PRC import PRC, binary_field
from utils import tokenize, binarize_setup, binarize_next, detokenize, start_model


class PRC_watermark:
    def __init__(self, prc: PRC, model, tokenizer):
        self.prc = prc
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, prompt: str, length: int, encode_random=False) -> str:
        prompt = tokenize(prompt, self.tokenizer)
        inputs = prompt.to(self.model.device)
        attn = torch.ones_like(inputs)
        blen, token_to_id, id_to_token = binarize_setup(self.tokenizer)
        print(token_to_id)
        past = None
        vocab_size = len(self.tokenizer)
        curr_word = []
        num_changed = 0
        total_entropy = np.zeros(blen)
        num_bits = blen * length
        for i in range(length):
            if len(curr_word) == 0:
                curr_word = np.array(self.prc.generate_codeword())
            with torch.no_grad():
                if past:
                    output = self.model(
                        inputs[:, -1:], past_key_values=past, attention_mask=attn
                    )
                else:
                    output = self.model(inputs)

            probs = torch.nn.functional.softmax(
                output.logits[:, -1, :vocab_size], dim=-1
            ).cpu()[0, :]
            token_id = 0
            for ind in range(blen):
                p0, p1 = binarize_next(probs, ind, blen, token_id)
                token_id = token_id << 1
                # For robustness, We don't want to re-draw a
                # codeword in the middle of a token.
                if len(curr_word) > 0 and not encode_random:
                    p1_hat = p1 / (p0 + p1)
                    total_entropy[ind] += (
                        p1_hat * -log2(p1_hat + 1e-7)
                        + (1 - p1_hat) * -log2(1 - p1_hat + 1e-7)
                    )[0]
                    prob = p1_hat - (((-1) ** curr_word[0]) * min(p1_hat, 1 - p1_hat))
                    if random.random() < prob:
                        token_id += 1
                        if curr_word[0] == 0:
                            num_changed += 1
                    elif curr_word[0] == 1:
                        num_changed += 1
                    curr_word = curr_word[1:]
                elif (random.random()) < p1 / (p0 + p1):
                    token_id += 1
            print(token_id)
            token = torch.tensor([[token_id]])
            inputs = torch.cat([inputs, token], dim=-1)

            past = output.past_key_values
            attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
        print(f"{num_changed / num_bits=}")
        print(f"{total_entropy / length=}")
        return detokenize(inputs.detach().cpu()[0], self.tokenizer)

    def detect_watermark(self, text: str) -> bool:
        blen, token_to_id, id_to_token = binarize_setup(self.tokenizer)
        tokens = tokenize(text, self.tokenizer)[0]
        binarized = []
        for token in tokens:
            # Format `tokens[i]` into a binary string with `blen` bits - 0 padded.
            token_bits = f"{token:0{blen}b}"
            binarized.append(
                np.array(
                    [1 if digit == "1" else 0 for digit in token_bits], dtype=np.int8
                )
            )
        binarized = np.array(binarized)
        for binarized_tokens_list in np.lib.stride_tricks.sliding_window_view(
            binarized, (ceil(self.prc.word_length / blen), blen)
        ):
            code_word = np.concatenate(binarized_tokens_list[0])[: self.prc.word_length]
            code_word = binary_field(code_word)
            if self.prc.validate_codeword(code_word):
                return True
        return False


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
    response_sizes = [30]
    samples_per_size = 1  # Set to 10 for a quicker run
    watermarker = PRC_watermark(
        PRC.generate_from_params(16 * 4, 19, 33, 3, 0, 0.3), model, tokenizer
    )
    for size in response_sizes:
        total_successes = 0
        print("Making samples of size " + str(size) + ":")
        for i in range(samples_per_size):
            key = random.random()
            prompt = random.choice(prompts)
            res = watermarker.generate(prompt=prompt, length=size, encode_random=False)
            detected = watermarker.detect_watermark(res[len(prompt) :])
            print(f"Run ended with {detected=}")
            print(res)
            total_successes += int(detected)
        print(f"Succeeded in {total_successes} / {samples_per_size} attempts.")
