# # train.py — microGPT v2.1 (Self-Attention, CPU-only, Pure Python) without NumPy

# import os
# import math
# import pickle
# import json
# import random
# import time
# import re

# # =====================
# # CONFIG
# # =====================
# CONFIG = {
#     "seq_len": 12,          # ✅ reduced → more training sequences
#     "embed_dim": 96,
#     "lr": 0.003,
#     "epochs": 400,          # ✅ tiny models need longer training
#     "lr_decay": 0.985,
#     "lr_min": 5e-4,
#     "grad_clip": 1.0
# }

# MODEL_DIR = "model"
# DATASET_PATH = "data/healthcare_qa_dataset.txt"

# # =====================
# # TOKENIZER
# # =====================
# def tokenize(text):
#     return re.findall(r"\w+|[^\w\s]", text.lower())

# def build_vocab(texts):
#     vocab = sorted(set(tok for txt in texts for tok in tokenize(txt)))
#     stoi = {tok: i for i, tok in enumerate(vocab)}
#     itos = {i: tok for tok, i in stoi.items()}
#     return stoi, itos

# def encode(tokens, stoi):
#     return [stoi[t] for t in tokens if t in stoi]

# # =====================
# # DATASET LOADER
# # =====================


# def load_dataset(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Dataset not found at {path}")

#     with open(path, "r", encoding="utf-8") as f:
#         raw = f.read()

#     # ✅ Split whenever a number + space appears (1 , 2 , 300 ...)
#     chunks = re.split(r"\s(?=\d+\s)", raw)

#     texts = []

#     for chunk in chunks:
#         chunk = chunk.strip()
#         if not chunk:
#             continue

#         parts = chunk.split(" ", 1)
#         if len(parts) < 2:
#             continue

#         serial, content = parts

#         if not serial.isdigit():
#             continue

#         # ---- Q/A Split ----
#         if "?" in content:
#             q, a = content.split("?", 1)
#             q = q.strip() + "?"
#             a = a.strip()
#         else:
#             words = content.split()
#             mid = len(words) // 2
#             q = " ".join(words[:mid])
#             a = " ".join(words[mid:])

#         if len(q) > 3 and len(a) > 3:
#             texts.append(f"Question: {q} Answer: {a}")

#     if len(texts) == 0:
#         raise ValueError(
#             "❌ No valid QA pairs detected.\n"
#             "Expected format example:\n"
#             "1 What is asthma? Asthma is a lung disease."
#         )

#     return texts


# # =====================
# # UTILITIES
# # =====================
# def randn(rows, cols, scale=0.02):
#     return [[random.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]

# def zeros(size):
#     return [0.0 for _ in range(size)]

# def clip_gradients(grads, clip_value):
#     norm = math.sqrt(sum(g*g for g in grads) + 1e-9)
#     if norm > clip_value:
#         scale = clip_value / norm
#         grads = [g * scale for g in grads]
#     return grads

# def softmax(vec):
#     m = max(vec)
#     exps = [math.exp(v - m) for v in vec]
#     s = sum(exps) + 1e-9
#     return [e / s for e in exps]

# def layer_norm(vec, eps=1e-5):
#     mean = sum(vec) / len(vec)
#     var = sum((v - mean) ** 2 for v in vec) / len(vec)
#     return [(v - mean) / math.sqrt(var + eps) for v in vec]

# # =====================
# # MODEL INIT
# # =====================
# def init_weights(vocab_size):
#     D = CONFIG["embed_dim"]

#     return {
#         "W_embed": randn(vocab_size, D),
#         "W_pos": randn(CONFIG["seq_len"], D),

#         "W_q": randn(D, D),
#         "W_k": randn(D, D),
#         "W_v": randn(D, D),

#         "W_out": randn(D, vocab_size),
#         "b_out": zeros(vocab_size)
#     }

# # =====================
# # LINEAR
# # =====================
# def linear(x, W):
#     return [sum(x[i] * W[i][j] for i in range(len(x))) for j in range(len(W[0]))]

# # =====================
# # SELF-ATTENTION
# # =====================
# def self_attention(x_seq, weights):
#     W_q, W_k, W_v = weights["W_q"], weights["W_k"], weights["W_v"]

#     Q = [linear(x, W_q) for x in x_seq]
#     K = [linear(x, W_k) for x in x_seq]
#     V = [linear(x, W_v) for x in x_seq]

#     D = len(Q[0])
#     scale = 1 / math.sqrt(D)

#     out = []

#     for q in Q:
#         scores = [sum(q[i]*k[i] for i in range(D)) * scale for k in K]
#         probs = softmax(scores)

#         attn_vec = [0.0] * D
#         for p, v in zip(probs, V):
#             for i in range(D):
#                 attn_vec[i] += p * v[i]

#         out.append(attn_vec)

#     return out

# # =====================
# # FORWARD
# # =====================
# def forward(seq, weights):
#     emb = []

#     for i, token_id in enumerate(seq):
#         token_vec = weights["W_embed"][token_id]
#         pos_vec = weights["W_pos"][i]

#         emb.append([
#             token_vec[d] + pos_vec[d]
#             for d in range(len(token_vec))
#         ])

#     if len(emb) == 0:
#         return [0.0] * len(weights["b_out"]), [0.0] * CONFIG["embed_dim"]

#     attn_out = self_attention(emb, weights)

#     # ✅ Residual connection (CRITICAL)
#     attn_out = [
#         [attn_out[i][d] + emb[i][d] for d in range(len(emb[i]))]
#         for i in range(len(emb))
#     ]

#     # ✅ LayerNorm (stability)
#     attn_out = [layer_norm(x) for x in attn_out]

#     last = attn_out[-1]

#     logits = linear(last, weights["W_out"])
#     logits = [logits[i] + weights["b_out"][i] for i in range(len(logits))]

#     return logits, last

# # =====================
# # LOSS
# # =====================
# def cross_entropy(probs, target):
#     return -math.log(probs[target] + 1e-9)

# # =====================
# # TRAIN STEP
# # =====================
# def train_step(seq, weights, lr):
#     x = seq[:-1]
#     y = seq[-1]

#     logits, features = forward(x, weights)
#     probs = softmax(logits)

#     loss = cross_entropy(probs, y)

#     grads = probs[:]
#     grads[y] -= 1
#     grads = clip_gradients(grads, CONFIG["grad_clip"])

#     for i in range(len(weights["W_out"])):
#         for j in range(len(weights["W_out"][0])):
#             weights["W_out"][i][j] -= lr * grads[j] * features[i]

#     for j in range(len(weights["b_out"])):
#         weights["b_out"][j] -= lr * grads[j]

#     return loss

# # =====================
# # SAVE / LOAD
# # =====================
# def save_model(weights, stoi, step):
#     os.makedirs(MODEL_DIR, exist_ok=True)

#     with open(os.path.join(MODEL_DIR, "weights.pkl"), "wb") as f:
#         pickle.dump(weights, f)

#     with open(os.path.join(MODEL_DIR, "vocab.json"), "w") as f:
#         json.dump(stoi, f)

#     with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
#         json.dump({"config": CONFIG, "step": step}, f)

# def load_model():
#     with open(os.path.join(MODEL_DIR, "weights.pkl"), "rb") as f:
#         weights = pickle.load(f)

#     with open(os.path.join(MODEL_DIR, "vocab.json"), "r") as f:
#         stoi = json.load(f)

#     with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
#         meta = json.load(f)

#     return weights, stoi, meta["step"]

# # =====================
# # MAIN
# # =====================
# def main():
#     print("Loading dataset...")
#     texts = load_dataset(DATASET_PATH)
#     print(f"✅ Loaded {len(texts)} samples")

#     if os.path.exists(os.path.join(MODEL_DIR, "weights.pkl")):
#         print("Loading existing model...")
#         weights, stoi, global_step = load_model()
#     else:
#         print("Initializing new model...")
#         stoi, _ = build_vocab(texts)
#         weights = init_weights(len(stoi))
#         global_step = 0

#     print(f"Vocab size: {len(stoi)}")

#     lr = CONFIG["lr"]

#     print("\n🚀 Training with Self-Attention...\n")

#     for epoch in range(CONFIG["epochs"]):
#         random.shuffle(texts)

#         total_loss = 0
#         steps = 0
#         start = time.time()

#         for text in texts:
#             tokens = tokenize(text)
#             ids = encode(tokens, stoi)

#             if len(ids) <= CONFIG["seq_len"]:
#                 continue

#             for i in range(len(ids) - CONFIG["seq_len"]):
#                 seq = ids[i:i + CONFIG["seq_len"]]
#                 loss = train_step(seq, weights, lr)

#                 total_loss += loss
#                 steps += 1
#                 global_step += 1

#         elapsed = time.time() - start

#         if steps == 0:
#             print(f"Epoch {epoch+1} | ❌ No sequences")
#         else:
#             print(f"Epoch {epoch+1} | Loss {total_loss/steps:.4f} | LR {lr:.5f} | Steps {steps} | Time {elapsed:.2f}s")

#         lr *= CONFIG["lr_decay"]
#         lr = max(lr, CONFIG["lr_min"])

#         if (epoch + 1) % 20 == 0:
#             save_model(weights, stoi, global_step)
#             print("💾 Checkpoint saved\n")

#     save_model(weights, stoi, global_step)

#     print("\n✅ Training complete")
#     print("✅ Model saved")

# # =====================
# # ENTRY
# # =====================
# if __name__ == "__main__":
#     main()











# train.py — microGPT v3.0 (Self-Attention, NumPy, CPU-fast)  NumPy integration

import os
import math
import pickle
import json
import random
import time
import re
import numpy as np

# =====================
# CONFIG
# =====================
CONFIG = {
    "seq_len": 12,
    "embed_dim": 96,
    "lr": 0.003,
    "epochs": 400,
    "lr_decay": 0.985,
    "lr_min": 5e-4,
    "grad_clip": 1.0
}

MODEL_DIR = "model"
DATASET_PATH = "data/healthcare_qa_dataset.txt"

# =====================
# TOKENIZER
# =====================
def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

def build_vocab(texts):
    vocab = sorted(set(tok for txt in texts for tok in tokenize(txt)))
    stoi = {tok: i for i, tok in enumerate(vocab)}
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos

def encode(tokens, stoi):
    return [stoi[t] for t in tokens if t in stoi]

# =====================
# DATASET LOADER
# =====================
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    chunks = re.split(r"\s(?=\d+\s)", raw)
    texts = []

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue

        parts = chunk.split(" ", 1)
        if len(parts) < 2:
            continue

        serial, content = parts
        if not serial.isdigit():
            continue

        if "?" in content:
            q, a = content.split("?", 1)
            q = q.strip() + "?"
            a = a.strip()
        else:
            words = content.split()
            mid = len(words) // 2
            q = " ".join(words[:mid])
            a = " ".join(words[mid:])

        if len(q) > 3 and len(a) > 3:
            texts.append(f"Question: {q} Answer: {a}")

    if len(texts) == 0:
        raise ValueError("❌ No valid QA pairs detected.")

    return texts

# =====================
# UTILITIES
# =====================
def randn(shape, scale=0.02):
    return np.random.uniform(-scale, scale, size=shape).astype(np.float32)

def clip_gradients(grad, clip_value):
    norm = np.linalg.norm(grad) + 1e-9
    if norm > clip_value:
        grad *= clip_value / norm
    return grad

def softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / (np.sum(exp) + 1e-9)

def layer_norm(x, eps=1e-5):
    mean = np.mean(x)
    var = np.var(x)
    return (x - mean) / np.sqrt(var + eps)

# =====================
# MODEL INIT
# =====================
def init_weights(vocab_size):
    D = CONFIG["embed_dim"]
    S = CONFIG["seq_len"]

    return {
        "W_embed": randn((vocab_size, D)),
        "W_pos": randn((S, D)),

        "W_q": randn((D, D)),
        "W_k": randn((D, D)),
        "W_v": randn((D, D)),

        "W_out": randn((D, vocab_size)),
        "b_out": np.zeros(vocab_size, dtype=np.float32)
    }

# =====================
# SELF-ATTENTION
# =====================
def self_attention(x_seq, weights):
    W_q, W_k, W_v = weights["W_q"], weights["W_k"], weights["W_v"]

    Q = x_seq @ W_q
    K = x_seq @ W_k
    V = x_seq @ W_v

    scale = 1.0 / math.sqrt(Q.shape[-1])

    scores = (Q @ K.T) * scale
    probs = np.apply_along_axis(softmax, 1, scores)

    out = probs @ V
    return out

# =====================
# FORWARD
# =====================
def forward(seq, weights):
    seq = np.array(seq, dtype=np.int32)

    token_emb = weights["W_embed"][seq]
    pos_emb = weights["W_pos"][:len(seq)]

    emb = token_emb + pos_emb

    if len(emb) == 0:
        return np.zeros_like(weights["b_out"]), np.zeros(CONFIG["embed_dim"])

    attn_out = self_attention(emb, weights)

    # Residual
    attn_out = attn_out + emb

    # LayerNorm
    attn_out = np.apply_along_axis(layer_norm, 1, attn_out)

    last = attn_out[-1]

    logits = last @ weights["W_out"] + weights["b_out"]

    return logits, last

# =====================
# LOSS
# =====================
def cross_entropy(probs, target):
    return -np.log(probs[target] + 1e-9)

# =====================
# TRAIN STEP
# =====================
def train_step(seq, weights, lr):
    x = seq[:-1]
    y = seq[-1]

    logits, features = forward(x, weights)
    probs = softmax(logits)

    loss = cross_entropy(probs, y)

    grads = probs.copy()
    grads[y] -= 1.0
    grads = clip_gradients(grads, CONFIG["grad_clip"])

    # Vectorized W_out update
    weights["W_out"] -= lr * np.outer(features, grads)

    # Bias update
    weights["b_out"] -= lr * grads

    return float(loss)

# =====================
# SAVE / LOAD
# =====================
def save_model(weights, stoi, step):
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(os.path.join(MODEL_DIR, "weights.pkl"), "wb") as f:
        pickle.dump(weights, f)

    with open(os.path.join(MODEL_DIR, "vocab.json"), "w") as f:
        json.dump(stoi, f)

    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump({"config": CONFIG, "step": step}, f)

def load_model():
    with open(os.path.join(MODEL_DIR, "weights.pkl"), "rb") as f:
        weights = pickle.load(f)

    with open(os.path.join(MODEL_DIR, "vocab.json"), "r") as f:
        stoi = json.load(f)

    with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
        meta = json.load(f)

    return weights, stoi, meta["step"]

# =====================
# MAIN
# =====================
def main():
    print("Loading dataset...")
    texts = load_dataset(DATASET_PATH)
    print(f"✅ Loaded {len(texts)} samples")

    if os.path.exists(os.path.join(MODEL_DIR, "weights.pkl")):
        print("Loading existing model...")
        weights, stoi, global_step = load_model()
    else:
        print("Initializing new model...")
        stoi, _ = build_vocab(texts)
        weights = init_weights(len(stoi))
        global_step = 0

    print(f"Vocab size: {len(stoi)}")

    lr = CONFIG["lr"]

    print("\n🚀 Training with Self-Attention (NumPy)...\n")

    for epoch in range(CONFIG["epochs"]):
        random.shuffle(texts)

        total_loss = 0
        steps = 0
        start = time.time()

        for text in texts:
            tokens = tokenize(text)
            ids = encode(tokens, stoi)

            if len(ids) <= CONFIG["seq_len"]:
                continue

            for i in range(len(ids) - CONFIG["seq_len"]):
                seq = ids[i:i + CONFIG["seq_len"]]
                loss = train_step(seq, weights, lr)

                total_loss += loss
                steps += 1
                global_step += 1

        elapsed = time.time() - start

        if steps == 0:
            print(f"Epoch {epoch+1} | ❌ No sequences")
        else:
            print(f"Epoch {epoch+1} | Loss {total_loss/steps:.4f} | LR {lr:.5f} | Steps {steps} | Time {elapsed:.2f}s")

        lr *= CONFIG["lr_decay"]
        lr = max(lr, CONFIG["lr_min"])

        if (epoch + 1) % 20 == 0:
            save_model(weights, stoi, global_step)
            print("💾 Checkpoint saved\n")

    save_model(weights, stoi, global_step)

    print("\n✅ Training complete")
    print("✅ Model saved")

# =====================
# ENTRY
# =====================
if __name__ == "__main__":
    main()
