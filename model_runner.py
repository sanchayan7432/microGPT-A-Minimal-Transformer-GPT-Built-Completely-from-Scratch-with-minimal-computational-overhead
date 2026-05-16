"""
Load trained microGPT & generate names
Pure Python • CPU only
"""

import json, math, random
random.seed()

# -----------------------------
# Load dataset metadata
# -----------------------------
docs = [l.strip() for l in open("input.txt") if l.strip()]
uchars = sorted(set("".join(docs)))

stoi = {c:i for i,c in enumerate(uchars)}
itos = {i:c for i,c in enumerate(uchars)}

BOS = len(uchars)
vocab_size = len(uchars) + 1

# -----------------------------
# Autograd Value (inference-safe)
# -----------------------------
class Value:
    def __init__(self, data):
        self.data = float(data)

    def __add__(self, other): return Value(self.data + other.data)
    def __mul__(self, other): return Value(self.data * other.data)
    def __truediv__(self, other): return Value(self.data / other.data)
    def __sub__(self, other): return Value(self.data - other.data)

    def exp(self): return Value(math.exp(self.data))
    def relu(self): return Value(max(0.0, self.data))

# -----------------------------
# Helpers
# -----------------------------
def vsum(vals):
    return Value(sum(v.data for v in vals))

def linear(x, w):
    return [vsum(Value(wi.data * xi.data) for wi, xi in zip(row, x)) for row in w]

def softmax(logits):
    max_val = max(l.data for l in logits)
    exps = [math.exp(l.data - max_val) for l in logits]
    s = sum(exps)
    return [e/s for e in exps]

def rmsnorm(x):
    ms = sum(xi.data**2 for xi in x)/len(x)
    scale = 1.0/math.sqrt(ms + 1e-5)
    return [Value(xi.data * scale) for xi in x]

def sample(probs, temp=0.7):
    temp = max(temp, 0.05)
    adjusted = [p**(1.0/temp) for p in probs]
    s = sum(adjusted)
    adjusted = [p/s for p in adjusted]
    return random.choices(range(len(probs)), adjusted)[0]

# -----------------------------
# Model Hyperparameters
# -----------------------------
n_embd = 16
n_head = 4
n_layer = 2
block_size = 16
head_dim = n_embd // n_head
scale_attn = 1.0 / math.sqrt(head_dim)

# -----------------------------
# Initialize empty weights
# -----------------------------
def matrix(nout, nin):
    return [[Value(0.0) for _ in range(nin)] for _ in range(nout)]

state_dict = {
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "lm_head": matrix(vocab_size, n_embd),
}

for li in range(n_layer):
    state_dict[f"layer{li}.wq"] = matrix(n_embd, n_embd)
    state_dict[f"layer{li}.wk"] = matrix(n_embd, n_embd)
    state_dict[f"layer{li}.wv"] = matrix(n_embd, n_embd)
    state_dict[f"layer{li}.wo"] = matrix(n_embd, n_embd)
    state_dict[f"layer{li}.fc1"] = matrix(4*n_embd, n_embd)
    state_dict[f"layer{li}.fc2"] = matrix(n_embd, 4*n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]

# -----------------------------
# Load trained weights
# -----------------------------
print("Loading weights...")
weights = json.load(open("weights.json"))

for p, w in zip(params, weights):
    p.data = w

# -----------------------------
# GPT Forward (Inference)
# -----------------------------
def gpt(token_id, pos_id, keys, values):

    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = rmsnorm([Value(t.data + p.data) for t,p in zip(tok_emb, pos_emb)])

    for li in range(n_layer):

        x_res = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f"layer{li}.wq"])
        k = linear(x, state_dict[f"layer{li}.wk"])
        v = linear(x, state_dict[f"layer{li}.wv"])

        keys[li].append(k)
        values[li].append(v)

        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            qh = q[hs:hs+head_dim]
            kh = [kt[hs:hs+head_dim] for kt in keys[li]]
            vh = [vt[hs:hs+head_dim] for vt in values[li]]

            logits = [
                sum(qh[j].data * kh[t][j].data for j in range(head_dim)) * scale_attn
                for t in range(len(kh))
            ]

            weights_sm = softmax([Value(l) for l in logits])

            head_out = [
                Value(sum(weights_sm[t] * vh[t][j].data for t in range(len(vh))))
                for j in range(head_dim)
            ]

            x_attn.extend(head_out)

        x = [Value(a.data + b.data) for a,b in zip(
            linear(x_attn, state_dict[f"layer{li}.wo"]),
            x_res
        )]

        x_res = x
        x = linear(rmsnorm(x), state_dict[f"layer{li}.fc1"])
        x = [xi.relu() for xi in x]

        x = [Value(a.data + b.data) for a,b in zip(
            linear(x, state_dict[f"layer{li}.fc2"]),
            x_res
        )]

    return linear(x, state_dict["lm_head"])

# -----------------------------
# Generate Names
# -----------------------------
print("\n--- Generated Names ---")

for _ in range(10):

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    token_id = BOS
    output = []

    for pos in range(block_size):
        logits = gpt(token_id, pos, keys, values)
        probs = softmax(logits)

        token_id = sample(probs, temp=0.7)

        if token_id == BOS:
            break

        output.append(itos[token_id])

    print("".join(output))
