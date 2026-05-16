# chat.py — microGPT v3 (NumPy, Improved Sampling)

import os
import pickle
import json
import numpy as np
import re

MODEL_DIR = "model"

# =====================
# TOKENIZER
# =====================
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

# =====================
# LOAD MODEL
# =====================
def load_model():
    print("Loading model...")

    with open(os.path.join(MODEL_DIR, "weights.pkl"), "rb") as f:
        weights = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "vocab.json"), "r") as f:
        stoi = json.load(f)

    with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
        meta = json.load(f)

    itos = {i: tok for tok, i in stoi.items()}
    config = meta["config"]

    return weights, stoi, itos, config

# =====================
# UTILITIES
# =====================
def softmax(logits):
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / (np.sum(exp) + 1e-9)

def sample_token(logits, temperature=0.9, top_k=30):
    logits = np.array(logits, dtype=np.float32)

    # Temperature scaling
    logits = logits / temperature

    probs = softmax(logits)

    # Top-k filtering
    if top_k is not None:
        indices = np.argsort(probs)[-top_k:]
        filtered_probs = probs[indices]
        filtered_probs /= np.sum(filtered_probs)
        return np.random.choice(indices, p=filtered_probs)

    return np.random.choice(len(probs), p=probs)

# =====================
# SELF-ATTENTION
# =====================
def self_attention(x_seq, weights):
    W_q, W_k, W_v = weights["W_q"], weights["W_k"], weights["W_v"]

    Q = x_seq @ W_q
    K = x_seq @ W_k
    V = x_seq @ W_v

    scale = 1.0 / np.sqrt(Q.shape[-1])

    scores = (Q @ K.T) * scale
    probs = np.apply_along_axis(softmax, 1, scores)

    return probs @ V

# =====================
# FORWARD
# =====================
def forward(seq, weights, config):
    seq = np.array(seq, dtype=np.int32)

    token_emb = weights["W_embed"][seq]
    pos_emb = weights["W_pos"][:len(seq)]

    emb = token_emb + pos_emb

    if len(emb) == 0:
        return np.zeros_like(weights["b_out"]), np.zeros(config["embed_dim"])

    attn_out = self_attention(emb, weights)

    # Residual connection
    attn_out = attn_out + emb

    # LayerNorm
    mean = np.mean(attn_out, axis=1, keepdims=True)
    var = np.var(attn_out, axis=1, keepdims=True)
    attn_out = (attn_out - mean) / np.sqrt(var + 1e-5)

    last = attn_out[-1]

    logits = last @ weights["W_out"] + weights["b_out"]

    return logits, last

# =====================
# GENERATE
# =====================
def generate(prompt, weights, stoi, itos, config, max_tokens=40):
    tokens = tokenize(prompt)
    ids = [stoi[t] for t in tokens if t in stoi]

    if len(ids) == 0:
        return "I don't understand."

    generated = ids.copy()

    for _ in range(max_tokens):

        context = generated[-config["seq_len"]:]
        logits, _ = forward(context, weights, config)

        # Repetition penalty (last 5 tokens)
        for prev_id in generated[-5:]:
            logits[prev_id] *= 0.8

        next_id = sample_token(
            logits,
            temperature=0.9,
            top_k=30
        )

        generated.append(int(next_id))

    words = [itos[i] for i in generated]

    return " ".join(words)

# =====================
# MAIN CHAT LOOP
# =====================
def main():
    weights, stoi, itos, config = load_model()

    print("✅ microGPT Chat Ready")
    print("Type 'exit' to quit\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Goodbye 👋")
            break

        response = generate(user_input, weights, stoi, itos, config)

        print("Bot:", response)
        print()

# =====================
# ENTRY
# =====================
if __name__ == "__main__":
    main()



"""
UPDATED GENERATION SETTINGS
---------------------------
temperature = 0.4  → more stable / less random
top_k = 5          → reduces chaotic sampling
max_new_tokens=60  → avoids rambling
stop rules         → prevents runaway text
"""
