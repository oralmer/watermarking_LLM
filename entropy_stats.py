import random
from math import log2
from typing import List

import numpy as np
import scipy as sp
import torch
from matplotlib import pyplot as plt
from tqdm import trange

from utils import start_model, tokenize, binarize_setup, detokenize


# Watermarked response generation, without a payload
def entropy_per_token(model, tokenizer, prompt, length=30) -> List[float]:
    prompt = tokenize(prompt, tokenizer)
    inputs = prompt.to(model.device)
    attn = torch.ones_like(inputs)
    past = None
    vocab_size = len(tokenizer)
    entropies = []
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
        p = probs.numpy().astype("float64")
        p /= p.sum()
        chosen_token = np.random.choice(np.arange(probs.shape[0]), 1, p=p)[0]
        chosen_prob = probs[chosen_token].item()
        entropies.append(-log2(chosen_prob))
        token = torch.tensor([[chosen_token]])
        inputs = torch.cat([inputs, token], dim=-1)

        past = output.past_key_values
        attn = torch.cat([attn, attn.new_ones((attn.shape[0], 1))], dim=-1)
    return entropies


if __name__ == "__main__":
    # --- Generating the example from the paper (Figures 1 and 3) ---
    # model, tokenizer = start_model("meta-llama/Llama-2-7b-chat-hf")  # Requires a LLamma token ID
    # res, ecc = generate_payloaded_response(424242, model, tokenizer, "[INST]Write an email asking my professor Prof. Hannity to not make the final exam in Machine Learning 101 too difficult. Begin directly with the body of the email.[\INST]Sure! Here is the body of such an email:", CompactText.text_to_bits("OZ"), 210, threshold=1.7, bit_limit=4, temperature=1.4)
    # assert(res == '\n\nSubject: Request for Consideration of Final Exam Difficulty in Machine Learning 101\n\nDear Professor Hannity,\n\nI hope this email finds you well. I am writing to respectfully request that you consider the level of difficulty for the final exam in Machine Learning 101. While I am confident in my understanding of the course materials and have put in a significant amount of effort throughout the semester, I do have concerns about the potential difficulty of the final exam.\n\nAs you may recall, several students in my previous sections of Machine Learning have found the final exam to be very challenging, leading to frustration and disappointment. While I understand that the course is intended to push students to their limits and beyond, I believe that some relaxation of the latter may be in order.\n\nI would kindly ask that you consider reducing the difficulty of the final exam or offering some additional supports or resources to help students prepare. I believe that this could enhance the learning experience or')
    # payload = extract_payload(424242, '\n\nSubject: Request for Consideration of Final Exam Difficulty in Machine Learning 101\n\nDear Professor Hannity,\n\nI hope this email finds you well. I am writing to respectfully request that you consider the level of difficulty for the final exam in Machine Learning 101. While I am confident in my understanding of the course materials and have put in a significant amount of effort throughout the semester, I do have concerns about the potential difficulty of the final exam.\n\nAs you may recall, several students in my previous sections of Machine Learning have found the final exam to be very challenging, leading to frustration and disappointment. While I understand that the course is intended to push students to their limits and beyond, I believe that some relaxation of the latter may be in order.\n\nI would kindly ask that you consider reducing the difficulty of the final exam or offering some additional supports or resources to help students prepare. I believe that this could enhance', tokenizer, threshold=1.7, bit_limit=4, skip_prefix=2)
    # assert(CompactText.bits_to_text(payload) == "OZ")

    # --- The plot from the paper (Figure 2) ---
    model, tokenizer = start_model("gpt2")
    prompts = [  # Taken from the GPT-2 official example prompts https://openai.com/research/better-language-models
        "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English.",
        "A train carriage containing controlled nuclear materials was stolen in Cincinnati today. Its whereabouts are unknown.",
        "Miley Cyrus was caught shoplifting from Abercrombie and Fitch on Hollywood Boulevard today.",
        "Legolas and Gimli advanced on the orcs, raising their weapons with a harrowing war cry.",
        "For today's homework assignment, please describe the reasons for the US Civil War.",
        "John F. Kennedy was just elected President of the United States after rising from the grave decades after his assassination. Due to miraculous developments in nanotechnology, Kennedy's brain was rebuilt from his remains and installed in the control center of a state-of-the art humanoid robot. Below is a transcript of his acceptance speech.",
    ]
    response_sizes = [300]
    samples_per_size = 20  # Set to 10 for a quicker run
    for size in response_sizes:
        entropies = []
        print("Making samples of size " + str(size) + ":")
        for i in trange(samples_per_size, desc=f"{size=}"):
            prompt = random.choice(prompts)
            entropies.append(
                entropy_per_token(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=prompt,
                    length=size,
                )
            )
        print(entropies)
    entropies = np.sum(entropies, axis=0) / 20
    plt.cla()
    conv_size = 5
    plt.plot(entropies[:-conv_size], color="r")
    plt.plot(np.convolve(entropies, np.ones(conv_size), "valid") / conv_size, color="b")
    plt.title(f"Average empirical entropy per token for {size=}")
    plt.show()
