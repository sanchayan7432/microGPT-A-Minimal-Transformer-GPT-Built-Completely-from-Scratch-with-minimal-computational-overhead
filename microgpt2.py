# """
# microgpt2.py — Self-Improving Minimal GPT (Fully Corrected)
# Pure Python • CPU only • No dependencies
# """

# import os, math, random, json, urllib.request
# random.seed(42)

# # ---------------- Dataset ----------------
# DATA_PATH = "input.txt"

# if not os.path.exists(DATA_PATH):
#     print("⬇ Downloading dataset...")
#     url = "https://raw.githubusercontent.com/sanchayan7432/MY-REPOSITORIES/main/names.txt"
#     urllib.request.urlretrieve(url, DATA_PATH)

# docs = [l.strip() for l in open(DATA_PATH, encoding="utf-8") if l.strip()]
# random.shuffle(docs)

# print(f"✅ Dataset loaded: {len(docs)} names")

# uchars = sorted(set("".join(docs)))
# BOS = len(uchars)
# vocab_size = len(uchars) + 1

# print(f"✅ Vocab size: {vocab_size}")

# # ---------------- Autograd ----------------
# class Value:
#     __slots__ = ("data", "grad", "_children", "_local_grads")

#     def __init__(self, data, children=(), local_grads=()):
#         self.data = float(data)
#         self.grad = 0.0
#         self._children = children
#         self._local_grads = local_grads

#     def __add__(self, other):
#         other = other if isinstance(other, Value) else Value(other)
#         return Value(self.data + other.data, (self, other), (1.0, 1.0))

#     def __mul__(self, other):
#         other = other if isinstance(other, Value) else Value(other)
#         return Value(self.data * other.data, (self, other),
#                      (other.data, self.data))

#     def __pow__(self, p):
#         return Value(self.data ** p, (self,),
#                      (p * self.data ** (p - 1),))

#     def log(self):
#         return Value(math.log(self.data), (self,), (1.0 / self.data,))

#     def exp(self):
#         e = math.exp(self.data)
#         return Value(e, (self,), (e,))

#     def relu(self):
#         return Value(max(0.0, self.data), (self,),
#                      (1.0 if self.data > 0 else 0.0,))

#     def __neg__(self): return self * -1
#     def __sub__(self, other): return self + (-other)
#     def __truediv__(self, other): return self * other ** -1

#     def backward(self):
#         topo, visited = [], set()

#         def build(v):
#             if v not in visited:
#                 visited.add(v)
#                 for c in v._children:
#                     build(c)
#                 topo.append(v)

#         build(self)
#         self.grad = 1.0

#         for v in reversed(topo):
#             for child, local_grad in zip(v._children, v._local_grads):
#                 child.grad += local_grad * v.grad


# def vsum(vals):
#     return sum(vals, Value(0.0))

# # ---------------- Utilities ----------------
# def matrix(nout, nin, std=0.08):
#     return [[Value(random.gauss(0, std)) for _ in range(nin)]
#             for _ in range(nout)]

# def linear(x, w):
#     return [vsum(wi * xi for wi, xi in zip(row, x)) for row in w]

# def softmax(logits):
#     max_val = max(l.data for l in logits)
#     exps = [(l - max_val).exp() for l in logits]
#     total = vsum(exps)
#     return [e / total for e in exps]

# def rmsnorm(x):
#     ms = vsum(xi * xi for xi in x) / len(x)
#     scale = (ms + 1e-5) ** -0.5
#     return [xi * scale for xi in x]

# def dropout(x, p=0.1, training=True):
#     if not training:
#         return x
#     mask = 0.0 if random.random() < p else 1.0
#     return x * mask * (1.0 / (1.0 - p))

# # ---------------- Model ----------------
# n_embd = 16
# n_head = 4
# n_layer = 2
# block_size = 16
# head_dim = n_embd // n_head

# state_dict = {
#     "wte": matrix(vocab_size, n_embd),
#     "wpe": matrix(block_size, n_embd),
#     "lm_head": matrix(vocab_size, n_embd),
# }

# for li in range(n_layer):
#     state_dict[f"layer{li}.wq"] = matrix(n_embd, n_embd)
#     state_dict[f"layer{li}.wk"] = matrix(n_embd, n_embd)
#     state_dict[f"layer{li}.wv"] = matrix(n_embd, n_embd)
#     state_dict[f"layer{li}.wo"] = matrix(n_embd, n_embd)
#     state_dict[f"layer{li}.fc1"] = matrix(4*n_embd, n_embd)
#     state_dict[f"layer{li}.fc2"] = matrix(n_embd, 4*n_embd)

# params = [p for mat in state_dict.values() for row in mat for p in row]

# # ---------------- GPT Forward ----------------
# def gpt(token_id, pos_id, keys, values, training=True):
#     tok_emb = state_dict["wte"][token_id]
#     pos_emb = state_dict["wpe"][pos_id]

#     x = rmsnorm([t + p for t, p in zip(tok_emb, pos_emb)])

#     for li in range(n_layer):

#         # ---- Attention ----
#         x_res = x
#         x = rmsnorm(x)

#         q = linear(x, state_dict[f"layer{li}.wq"])
#         k = linear(x, state_dict[f"layer{li}.wk"])
#         v = linear(x, state_dict[f"layer{li}.wv"])

#         keys[li].append(k)
#         values[li].append(v)

#         x_attn = []
#         for h in range(n_head):
#             hs = h * head_dim

#             qh = q[hs:hs+head_dim]
#             kh = [kt[hs:hs+head_dim] for kt in keys[li]]
#             vh = [vt[hs:hs+head_dim] for vt in values[li]]

#             logits = [
#                 vsum(qh[j] * kh[t][j] for j in range(head_dim))
#                 / math.sqrt(head_dim)
#                 for t in range(len(kh))
#             ]

#             weights = softmax(logits)

#             head_out = [
#                 vsum(weights[t] * vh[t][j] for t in range(len(vh)))
#                 for j in range(head_dim)
#             ]

#             x_attn.extend(head_out)

#         x = [a + b for a, b in zip(
#             linear(x_attn, state_dict[f"layer{li}.wo"]),
#             x_res
#         )]

#         # ---- MLP ----
#         x_res = x
#         x = rmsnorm(x)

#         x = linear(x, state_dict[f"layer{li}.fc1"])
#         x = [dropout(xi.relu(), training=training) for xi in x]

#         x = [a + b for a, b in zip(
#             linear(x, state_dict[f"layer{li}.fc2"]),
#             x_res
#         )]

#     return linear(x, state_dict["lm_head"])

# # ---------------- Optimizer (Adam) ----------------
# lr, beta1, beta2, eps = 0.01, 0.9, 0.99, 1e-8
# m = [0.0]*len(params)
# v = [0.0]*len(params)
# global_step = 0

# def clip_gradients(max_norm=1.0):
#     for p in params:
#         if p.grad > max_norm: p.grad = max_norm
#         if p.grad < -max_norm: p.grad = -max_norm

# # ---------------- Checkpoint ----------------
# def save_checkpoint():
#     with open("checkpoint.json", "w") as f:
#         json.dump({
#             "step": global_step,
#             "m": m,
#             "v": v,
#             "weights": [p.data for p in params]
#         }, f)

# def load_checkpoint():
#     global global_step
#     if not os.path.exists("checkpoint.json"):
#         return

#     with open("checkpoint.json") as f:
#         ckpt = json.load(f)

#     global_step = ckpt["step"]
#     m[:] = ckpt["m"]
#     v[:] = ckpt["v"]

#     for p, w_ in zip(params, ckpt["weights"]):
#         p.data = w_

#     print(f"✅ Resumed from step {global_step}")

# load_checkpoint()

# # ---------------- Training ----------------
# batch_size = 4
# steps_per_epoch = len(docs) // batch_size

# print(f"🚀 Steps per epoch: {steps_per_epoch}")

# for step in range(steps_per_epoch):

#     total_loss = Value(0.0)

#     for b in range(batch_size):

#         doc = docs[(global_step*batch_size + b) % len(docs)]
#         tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]

#         n = min(block_size, len(tokens)-1)

#         keys = [[] for _ in range(n_layer)]
#         values = [[] for _ in range(n_layer)]

#         losses = []
#         for pos in range(n):
#             logits = gpt(tokens[pos], pos, keys, values, training=True)
#             probs = softmax(logits)
#             losses.append(-probs[tokens[pos+1]].log())

#         total_loss += vsum(losses) * (1.0 / n)

#     total_loss = total_loss * (1.0 / batch_size)
#     total_loss.backward()
#     clip_gradients()

#     global_step += 1

#     for i, p in enumerate(params):
#         m[i] = beta1*m[i] + (1-beta1)*p.grad
#         v[i] = beta2*v[i] + (1-beta2)*(p.grad**2)

#         m_hat = m[i] / (1 - beta1**global_step)
#         v_hat = v[i] / (1 - beta2**global_step)

#         p.data -= lr * m_hat / (math.sqrt(v_hat) + eps)
#         p.grad = 0.0

#     if global_step % 100 == 0:
#         print(f"step {global_step:5d} | loss {total_loss.data:.4f}")
#         save_checkpoint()

# save_checkpoint()
# print("✅ Training complete & saved.")

# # ---------------- Inference ----------------
# def nucleus_sampling(logits, p=0.9, temperature=0.5):

#     scaled = [l.data / temperature for l in logits]
#     max_val = max(scaled)

#     exps = [math.exp(v - max_val) for v in scaled]
#     total = sum(exps)

#     probs = [e / total for e in exps]

#     sorted_probs = sorted(enumerate(probs),
#                           key=lambda x: x[1],
#                           reverse=True)

#     cum_prob, chosen = 0.0, []
#     for idx, prob in sorted_probs:
#         cum_prob += prob
#         chosen.append((idx, prob))
#         if cum_prob >= p:
#             break

#     r, cum = random.random(), 0.0
#     for idx, prob in chosen:
#         cum += prob
#         if r < cum:
#             return idx

#     return chosen[-1][0]

# print("\n--- inference ---")

# for _ in range(10):

#     keys = [[] for _ in range(n_layer)]
#     values = [[] for _ in range(n_layer)]

#     token_id = BOS
#     output = []

#     for pos in range(block_size):
#         logits = gpt(token_id, pos, keys, values, training=False)
#         token_id = nucleus_sampling(logits)

#         if token_id == BOS:
#             break

#         output.append(uchars[token_id])

#     print("".join(output))


"""
microgpt2.py — Self-Improving Minimal GPT (Improved Version)
Pure Python • CPU only • No dependencies
"""

import os, math, random, json, urllib.request
random.seed(42)

# ---------------- Dataset ----------------
DATA_PATH = "input.txt"

if not os.path.exists(DATA_PATH):
    print("⬇ Downloading dataset...")
    url = "https://raw.githubusercontent.com/sanchayan7432/MY-REPOSITORIES/main/names.txt"
    urllib.request.urlretrieve(url, DATA_PATH)

docs = [l.strip() for l in open(DATA_PATH, encoding="utf-8") if l.strip()]
random.shuffle(docs)

print(f"✅ Dataset loaded: {len(docs)} names")

uchars = sorted(set("".join(docs)))
BOS = len(uchars)
vocab_size = len(uchars) + 1

print(f"✅ Vocab size: {vocab_size}")

# ---------------- Autograd ----------------
class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other),
                     (other.data, self.data))

    def __pow__(self, p):
        return Value(self.data ** p, (self,),
                     (p * self.data ** (p - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1.0 / self.data,))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def relu(self):
        return Value(max(0.0, self.data), (self,),
                     (1.0 if self.data > 0 else 0.0,))

    def __neg__(self): return self * -1
    def __sub__(self, other): return self + (-other)
    def __truediv__(self, other): return self * other ** -1

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._children:
                    build(c)
                topo.append(v)

        build(self)
        self.grad = 1.0

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


def vsum(vals):
    return sum(vals, Value(0.0))

# ---------------- Utilities ----------------
def matrix(nout, nin, std=0.02):  # ✅ Better initialization
    return [[Value(random.gauss(0, std)) for _ in range(nin)]
            for _ in range(nout)]

def linear(x, w):
    return [vsum(wi * xi for wi, xi in zip(row, x)) for row in w]

def softmax(logits):
    max_val = max(l.data for l in logits)
    exps = [(l - max_val).exp() for l in logits]
    total = vsum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = vsum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def dropout(x, p=0.1, training=True):  # ✅ Cleaner dropout
    if not training:
        return x
    if random.random() < p:
        return x * 0.0
    return x * (1.0 / (1.0 - p))

# ---------------- Model ----------------
n_embd = 16
n_head = 4
n_layer = 2
block_size = 16
head_dim = n_embd // n_head

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

# ---------------- GPT Forward ----------------
def gpt(token_id, pos_id, keys, values, training=True):
    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]

    x = rmsnorm([t + p for t, p in zip(tok_emb, pos_emb)])

    for li in range(n_layer):

        # ---- Attention ----
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
                vsum(qh[j] * kh[t][j] for j in range(head_dim))
                / math.sqrt(head_dim)
                for t in range(len(kh))
            ]

            weights = softmax(logits)

            head_out = [
                vsum(weights[t] * vh[t][j] for t in range(len(vh)))
                for j in range(head_dim)
            ]

            x_attn.extend(head_out)

        x = [a + b for a, b in zip(
            linear(x_attn, state_dict[f"layer{li}.wo"]),
            x_res
        )]

        # ---- MLP ----
        x_res = x
        x = rmsnorm(x)

        x = linear(x, state_dict[f"layer{li}.fc1"])
        x = [dropout(xi.relu(), training=training) for xi in x]

        x = [a + b for a, b in zip(
            linear(x, state_dict[f"layer{li}.fc2"]),
            x_res
        )]

    return linear(x, state_dict["lm_head"])

# ---------------- Optimizer (Adam) ----------------
lr, beta1, beta2, eps = 0.005, 0.9, 0.99, 1e-8  # ✅ Lower LR
m = [0.0]*len(params)
v = [0.0]*len(params)
global_step = 0

def clip_gradients(max_norm=1.0):  # ✅ True norm clipping
    total_norm = math.sqrt(sum(p.grad**2 for p in params))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad *= scale

# ---------------- Training ----------------
batch_size = 4
steps_per_epoch = len(docs) // batch_size

ema_loss = 0.0
alpha = 0.95

print(f"🚀 Steps per epoch: {steps_per_epoch}")

for step in range(steps_per_epoch):

    total_loss = Value(0.0)

    for b in range(batch_size):

        doc = docs[(global_step*batch_size + b) % len(docs)]
        tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]

        n = min(block_size, len(tokens)-1)

        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]

        losses = []
        for pos in range(n):
            logits = gpt(tokens[pos], pos, keys, values, training=True)
            probs = softmax(logits)
            losses.append(-probs[tokens[pos+1]].log())

        total_loss += vsum(losses) * (1.0 / n)

    total_loss = total_loss * (1.0 / batch_size)
    total_loss.backward()
    clip_gradients()

    global_step += 1

    for i, p in enumerate(params):
        m[i] = beta1*m[i] + (1-beta1)*p.grad
        v[i] = beta2*v[i] + (1-beta2)*(p.grad**2)

        m_hat = m[i] / (1 - beta1**global_step)
        v_hat = v[i] / (1 - beta2**global_step)

        p.data -= lr * m_hat / (math.sqrt(v_hat) + eps)
        p.grad = 0.0

    ema_loss = alpha * ema_loss + (1-alpha) * total_loss.data

    if global_step % 100 == 0:
        print(f"step {global_step:5d} | loss {ema_loss:.4f}")

print("✅ Training complete.")

# ---------------- Nucleus Sampling ----------------
def nucleus_sampling(logits, p=0.9, temperature=0.6):

    temperature = max(0.05, temperature)  # ✅ Safety clamp

    scaled = [l.data / temperature for l in logits]
    max_val = max(scaled)

    exps = [math.exp(v - max_val) for v in scaled]
    total = sum(exps)
    probs = [e / total for e in exps]

    sorted_probs = sorted(enumerate(probs),
                          key=lambda x: x[1],
                          reverse=True)

    cum_prob, chosen = 0.0, []
    for idx, prob in sorted_probs:
        cum_prob += prob
        chosen.append((idx, prob))
        if cum_prob >= p:
            break

    r, cum = random.random(), 0.0
    for idx, prob in chosen:
        cum += prob
        if r < cum:
            return idx

    return chosen[-1][0]

# ---------------- Inference ----------------
print("\n--- inference ---")

NUM_SAMPLES = 10
MAX_LEN = 20

for _ in range(NUM_SAMPLES):

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    token_id = BOS
    output = []

    for pos in range(MAX_LEN):
        logits = gpt(token_id, pos, keys, values, training=False)
        token_id = nucleus_sampling(logits)

        if token_id == BOS:
            break

        output.append(uchars[token_id])

    print("".join(output))
