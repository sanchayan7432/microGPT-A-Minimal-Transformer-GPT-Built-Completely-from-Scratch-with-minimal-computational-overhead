# # """
# # microgpt1.py — Fully Minimal GPT (Corrected)
# # Pure Python • No dependencies • CPU only
# # improved base version
# # """

# # import os, math, random, json
# # random.seed(42)

# # # -----------------------------
# # # Dataset
# # # -----------------------------
# # if not os.path.exists("input.txt"):
# #     import urllib.request
# #     url = "https://github.com/sanchayan7432/MY-REPOSITORIES/blob/main/names.txt"
# #     urllib.request.urlretrieve(url, "input.txt")

# # docs = [l.strip() for l in open("input.txt") if l.strip()]
# # random.shuffle(docs)

# # uchars = sorted(set("".join(docs)))
# # BOS = len(uchars)
# # vocab_size = len(uchars) + 1

# # # -----------------------------
# # # Autograd Engine
# # # -----------------------------
# # class Value:
# #     __slots__ = ("data", "grad", "_children", "_local_grads")

# #     def __init__(self, data, children=(), local_grads=()):
# #         self.data = float(data)
# #         self.grad = 0.0
# #         self._children = children
# #         self._local_grads = local_grads

# #     def __add__(self, other):
# #         other = other if isinstance(other, Value) else Value(other)
# #         return Value(self.data + other.data, (self, other), (1.0, 1.0))

# #     def __mul__(self, other):
# #         other = other if isinstance(other, Value) else Value(other)
# #         return Value(self.data * other.data, (self, other),
# #                      (other.data, self.data))

# #     def __pow__(self, p):
# #         return Value(self.data ** p, (self,),
# #                      (p * self.data ** (p - 1),))

# #     def log(self):
# #         return Value(math.log(self.data), (self,), (1.0 / self.data,))

# #     def exp(self):
# #         e = math.exp(self.data)
# #         return Value(e, (self,), (e,))

# #     def relu(self):
# #         return Value(max(0.0, self.data), (self,),
# #                      (1.0 if self.data > 0 else 0.0,))

# #     def __neg__(self): return self * -1
# #     def __sub__(self, other): return self + (-other)
# #     def __truediv__(self, other): return self * other ** -1

# #     def backward(self):
# #         topo, visited = [], set()

# #         def build(v):
# #             if v not in visited:
# #                 visited.add(v)
# #                 for c in v._children:
# #                     build(c)
# #                 topo.append(v)

# #         build(self)
# #         self.grad = 1.0

# #         for v in reversed(topo):
# #             for child, local_grad in zip(v._children, v._local_grads):
# #                 child.grad += local_grad * v.grad


# # # -----------------------------
# # # Helpers
# # # -----------------------------
# # def vsum(vals):
# #     return sum(vals, Value(0.0))


# # def matrix(nout, nin, std=0.08):
# #     return [[Value(random.gauss(0, std)) for _ in range(nin)]
# #             for _ in range(nout)]


# # def linear(x, w):
# #     return [vsum(wi * xi for wi, xi in zip(row, x)) for row in w]


# # def softmax(logits):
# #     # ✅ Numerically stable softmax
# #     max_val = max(l.data for l in logits)
# #     exps = [(l - max_val).exp() for l in logits]
# #     total = vsum(exps)
# #     return [e / total for e in exps]


# # def rmsnorm(x):
# #     ms = vsum(xi * xi for xi in x) / len(x)
# #     scale = (ms + 1e-5) ** -0.5
# #     return [xi * scale for xi in x]


# # def dropout(x, p=0.1, training=True):
# #     if not training:
# #         return x
# #     mask = 0.0 if random.random() < p else 1.0
# #     return x * mask * (1.0 / (1.0 - p))


# # # -----------------------------
# # # Model Parameters
# # # -----------------------------
# # n_embd = 16
# # n_head = 4
# # n_layer = 2
# # block_size = 16
# # head_dim = n_embd // n_head

# # state_dict = {
# #     "wte": matrix(vocab_size, n_embd),
# #     "wpe": matrix(block_size, n_embd),
# #     "lm_head": matrix(vocab_size, n_embd),
# # }

# # for li in range(n_layer):
# #     state_dict[f"layer{li}.wq"] = matrix(n_embd, n_embd)
# #     state_dict[f"layer{li}.wk"] = matrix(n_embd, n_embd)
# #     state_dict[f"layer{li}.wv"] = matrix(n_embd, n_embd)
# #     state_dict[f"layer{li}.wo"] = matrix(n_embd, n_embd)
# #     state_dict[f"layer{li}.fc1"] = matrix(4 * n_embd, n_embd)
# #     state_dict[f"layer{li}.fc2"] = matrix(n_embd, 4 * n_embd)

# # params = [p for mat in state_dict.values() for row in mat for p in row]


# # # -----------------------------
# # # GPT Forward
# # # -----------------------------
# # def gpt(token_id, pos_id, keys, values, training=True):
# #     tok_emb = state_dict["wte"][token_id]
# #     pos_emb = state_dict["wpe"][pos_id]

# #     x = rmsnorm([t + p for t, p in zip(tok_emb, pos_emb)])

# #     for li in range(n_layer):

# #         # ---- Attention ----
# #         x_res = x
# #         x = rmsnorm(x)

# #         q = linear(x, state_dict[f"layer{li}.wq"])
# #         k = linear(x, state_dict[f"layer{li}.wk"])
# #         v = linear(x, state_dict[f"layer{li}.wv"])

# #         keys[li].append(k)
# #         values[li].append(v)

# #         x_attn = []
# #         for h in range(n_head):
# #             hs = h * head_dim
# #             qh = q[hs:hs+head_dim]
# #             kh = [kt[hs:hs+head_dim] for kt in keys[li]]
# #             vh = [vt[hs:hs+head_dim] for vt in values[li]]

# #             logits = [
# #                 vsum(qh[j] * kh[t][j] for j in range(head_dim)) /
# #                 math.sqrt(head_dim)
# #                 for t in range(len(kh))
# #             ]

# #             weights = softmax(logits)

# #             head_out = [
# #                 vsum(weights[t] * vh[t][j] for t in range(len(vh)))
# #                 for j in range(head_dim)
# #             ]

# #             x_attn.extend(head_out)

# #         x = [a + b for a, b in zip(
# #             linear(x_attn, state_dict[f"layer{li}.wo"]),
# #             x_res
# #         )]

# #         # ---- MLP ----
# #         x_res = x
# #         x = rmsnorm(x)

# #         x = linear(x, state_dict[f"layer{li}.fc1"])
# #         x = [dropout(xi.relu(), training=training) for xi in x]

# #         x = [a + b for a, b in zip(
# #             linear(x, state_dict[f"layer{li}.fc2"]),
# #             x_res
# #         )]

# #     return linear(x, state_dict["lm_head"])


# # # -----------------------------
# # # Optimizer (Adam)
# # # -----------------------------
# # lr, beta1, beta2, eps = 0.01, 0.9, 0.99, 1e-8
# # m = [0.0] * len(params)
# # v = [0.0] * len(params)

# # def clip_gradients(max_norm=1.0):
# #     for p in params:
# #         if p.grad > max_norm: p.grad = max_norm
# #         if p.grad < -max_norm: p.grad = -max_norm


# # # -----------------------------
# # # ✅ Corrected Nucleus Sampling
# # # -----------------------------
# # def nucleus_sampling(logits, p=0.9):
# #     sorted_logits = sorted(
# #         enumerate(logits),
# #         key=lambda x: x[1].data,
# #         reverse=True
# #     )

# #     max_val = sorted_logits[0][1].data
# #     total = sum(math.exp(v.data - max_val) for _, v in sorted_logits)

# #     cum_prob, chosen = 0.0, []
# #     for idx, val in sorted_logits:
# #         prob = math.exp(val.data - max_val) / total
# #         cum_prob += prob
# #         chosen.append((idx, prob))
# #         if cum_prob >= p:
# #             break

# #     r, cum = random.random(), 0.0
# #     for idx, prob in chosen:
# #         cum += prob
# #         if r < cum:
# #             return idx

# #     # ✅ Safety fallback
# #     return chosen[-1][0]


# # # -----------------------------
# # # Training
# # # -----------------------------
# # batch_size = 4
# # num_steps = 8518

# # for step in range(num_steps):

# #     total_loss = Value(0.0)

# #     for _ in range(batch_size):

# #         doc = random.choice(docs)
# #         tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]
# #         n = min(block_size, len(tokens)-1)

# #         keys = [[] for _ in range(n_layer)]
# #         values = [[] for _ in range(n_layer)]

# #         losses = []
# #         for pos in range(n):
# #             logits = gpt(tokens[pos], pos, keys, values, training=True)
# #             probs = softmax(logits)
# #             losses.append(-probs[tokens[pos+1]].log())

# #         total_loss += vsum(losses) * (1.0 / n)

# #     total_loss = total_loss * (1.0 / batch_size)
# #     total_loss.backward()
# #     clip_gradients()

# #     for i, p in enumerate(params):
# #         m[i] = beta1 * m[i] + (1 - beta1) * p.grad
# #         v[i] = beta2 * v[i] + (1 - beta2) * (p.grad ** 2)

# #         m_hat = m[i] / (1 - beta1 ** (step+1))
# #         v_hat = v[i] / (1 - beta2 ** (step+1))

# #         p.data -= lr * m_hat / (math.sqrt(v_hat) + eps)
# #         p.grad = 0.0

# #     print(f"step {step+1:3d} | loss {total_loss.data:.4f}")


# # # -----------------------------
# # # Inference
# # # -----------------------------
# # print("\n--- inference ---")

# # for _ in range(10):

# #     keys = [[] for _ in range(n_layer)]
# #     values = [[] for _ in range(n_layer)]

# #     token_id = BOS
# #     output = []

# #     for pos in range(block_size):
# #         logits = gpt(token_id, pos, keys, values, training=False)
# #         token_id = nucleus_sampling(logits)

# #         if token_id == BOS:
# #             break

# #         output.append(uchars[token_id])

# #     print("".join(output))









# """
# microgpt1.py — Fully Minimal GPT (Improved)
# Pure Python • CPU only • No dependencies
# """

# import os, math, random, json, urllib.request
# random.seed(42)

# # -----------------------------
# # Dataset (FIXED RAW URL)
# # -----------------------------
# if not os.path.exists("input.txt"):
#     print("Downloading dataset...")
#     url = "https://raw.githubusercontent.com/sanchayan7432/MY-REPOSITORIES/main/names.txt"
#     urllib.request.urlretrieve(url, "input.txt")

# docs = [l.strip() for l in open("input.txt") if l.strip()]
# random.shuffle(docs)

# uchars = sorted(set("".join(docs)))
# BOS = len(uchars)
# vocab_size = len(uchars) + 1

# print("Dataset size:", len(docs))
# print("Vocab size:", vocab_size)

# # -----------------------------
# # Autograd Engine
# # -----------------------------
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


# # -----------------------------
# # Helpers
# # -----------------------------
# def vsum(vals):
#     return sum(vals, Value(0.0))

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

# # -----------------------------
# # Temperature Sampling ✅ NEW
# # -----------------------------
# def sample_with_temp(probs, temp=0.5):
#     adjusted = [p.data ** (1.0 / temp) for p in probs]
#     s = sum(adjusted)
#     adjusted = [p / s for p in adjusted]
#     return random.choices(range(len(probs)), adjusted)[0]

# # -----------------------------
# # Model Parameters
# # -----------------------------
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
#     state_dict[f"layer{li}.fc1"] = matrix(4 * n_embd, n_embd)
#     state_dict[f"layer{li}.fc2"] = matrix(n_embd, 4 * n_embd)

# params = [p for mat in state_dict.values() for row in mat for p in row]

# # -----------------------------
# # Save / Load Weights ✅ NEW
# # -----------------------------
# def save_weights(path="weights.json"):
#     weights = [p.data for p in params]
#     with open(path, "w") as f:
#         json.dump(weights, f)

# def load_weights(path="weights.json"):
#     if not os.path.exists(path):
#         return
#     print("Loading existing weights...")
#     with open(path) as f:
#         weights = json.load(f)
#     for p, w in zip(params, weights):
#         p.data = w

# load_weights()

# # -----------------------------
# # GPT Forward
# # -----------------------------
# def gpt(token_id, pos_id, keys, values, training=True):

#     tok_emb = state_dict["wte"][token_id]
#     pos_emb = state_dict["wpe"][pos_id]
#     x = rmsnorm([t + p for t, p in zip(tok_emb, pos_emb)])

#     for li in range(n_layer):

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
#                 vsum(qh[j] * kh[t][j] for j in range(head_dim)) /
#                 math.sqrt(head_dim)
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

#         x_res = x
#         x = linear(rmsnorm(x), state_dict[f"layer{li}.fc1"])
#         x = [dropout(xi.relu(), training=training) for xi in x]

#         x = [a + b for a, b in zip(
#             linear(x, state_dict[f"layer{li}.fc2"]),
#             x_res
#         )]

#     return linear(x, state_dict["lm_head"])

# # -----------------------------
# # Optimizer (Adam)
# # -----------------------------
# lr, beta1, beta2, eps = 0.01, 0.9, 0.99, 1e-8
# m = [0.0] * len(params)
# v = [0.0] * len(params)

# def clip_gradients(max_norm=1.0):
#     for p in params:
#         if p.grad > max_norm: p.grad = max_norm
#         if p.grad < -max_norm: p.grad = -max_norm

# # -----------------------------
# # Training ✅ LONGER
# # -----------------------------
# batch_size = 4
# num_steps = 20000

# best_loss = 999

# for step in range(num_steps):

#     total_loss = Value(0.0)

#     for _ in range(batch_size):

#         doc = random.choice(docs)
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

#     for i, p in enumerate(params):
#         m[i] = beta1 * m[i] + (1 - beta1) * p.grad
#         v[i] = beta2 * v[i] + (1 - beta2) * (p.grad ** 2)

#         m_hat = m[i] / (1 - beta1 ** (step+1))
#         v_hat = v[i] / (1 - beta2 ** (step+1))

#         p.data -= lr * m_hat / (math.sqrt(v_hat) + eps)
#         p.grad = 0.0

#     if step % 50 == 0:
#         print(f"step {step+1:5d} | loss {total_loss.data:.4f}")

#     if total_loss.data < best_loss:
#         best_loss = total_loss.data
#         save_weights()

# # -----------------------------
# # Inference ✅ Temperature Sampling
# # -----------------------------
# print("\n--- inference ---")

# for _ in range(10):

#     keys = [[] for _ in range(n_layer)]
#     values = [[] for _ in range(n_layer)]

#     token_id = BOS
#     output = []

#     for pos in range(block_size):
#         logits = gpt(token_id, pos, keys, values, training=False)
#         probs = softmax(logits)

#         token_id = sample_with_temp(probs, temp=0.5)

#         if token_id == BOS:
#             break

#         output.append(uchars[token_id])

#     print("".join(output))







"""
microgpt1.py — Fully Minimal GPT (Corrected & Stabilized)
Pure Python • CPU only • No dependencies
"""

import os, math, random, json, urllib.request
random.seed(42)

# -----------------------------
# Dataset
# -----------------------------
if not os.path.exists("input.txt"):
    print("Downloading dataset...")
    url = "https://raw.githubusercontent.com/sanchayan7432/MY-REPOSITORIES/main/names.txt"
    urllib.request.urlretrieve(url, "input.txt")

docs = [l.strip() for l in open("input.txt") if l.strip()]
random.shuffle(docs)

uchars = sorted(set("".join(docs)))
stoi = {c: i for i, c in enumerate(uchars)}   # ✅ Faster lookup
itos = {i: c for i, c in enumerate(uchars)}

BOS = len(uchars)
vocab_size = len(uchars) + 1

print("Dataset size:", len(docs))
print("Vocab size:", vocab_size)

# -----------------------------
# Autograd Engine
# -----------------------------
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
        eps = 1e-12  # ✅ Prevent log(0)
        return Value(math.log(self.data + eps),
                     (self,),
                     (1.0 / (self.data + eps),))

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def relu(self):
        return Value(max(0.0, self.data),
                     (self,),
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


# -----------------------------
# Helpers
# -----------------------------
def vsum(vals):
    return sum(vals, Value(0.0))

def matrix(nout, nin, std=0.08):
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

def dropout(x, p=0.1, training=True):
    if not training:
        return x
    mask = 0.0 if random.random() < p else 1.0
    return x * mask * (1.0 / (1.0 - p))

# -----------------------------
# Temperature Sampling
# -----------------------------
def sample_with_temp(probs, temp=0.7):
    temp = max(temp, 0.05)  # ✅ Prevent instability
    adjusted = [p.data ** (1.0 / temp) for p in probs]
    s = sum(adjusted)
    adjusted = [p / s for p in adjusted]
    return random.choices(range(len(probs)), adjusted)[0]

# -----------------------------
# Model Parameters
# -----------------------------
n_embd = 16
n_head = 4
n_layer = 2
block_size = 16
head_dim = n_embd // n_head
scale = 1.0 / math.sqrt(head_dim)  # ✅ Cached

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
    state_dict[f"layer{li}.fc1"] = matrix(4 * n_embd, n_embd)
    state_dict[f"layer{li}.fc2"] = matrix(n_embd, 4 * n_embd)

params = [p for mat in state_dict.values() for row in mat for p in row]

# -----------------------------
# GPT Forward
# -----------------------------
def gpt(token_id, pos_id, keys, values, training=True):

    tok_emb = state_dict["wte"][token_id]
    pos_emb = state_dict["wpe"][pos_id]
    x = rmsnorm([t + p for t, p in zip(tok_emb, pos_emb)])

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
                vsum(qh[j] * kh[t][j] for j in range(head_dim)) * scale
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

        x_res = x
        x = linear(rmsnorm(x), state_dict[f"layer{li}.fc1"])
        x = [dropout(xi.relu(), training=training) for xi in x]

        x = [a + b for a, b in zip(
            linear(x, state_dict[f"layer{li}.fc2"]),
            x_res
        )]

    return linear(x, state_dict["lm_head"])

# -----------------------------
# Optimizer (Adam)
# -----------------------------
base_lr = 0.003   # ✅ More stable
beta1, beta2, eps = 0.9, 0.99, 1e-8

m = [0.0] * len(params)
v = [0.0] * len(params)

def clip_gradients(max_norm=1.0):
    total_norm = math.sqrt(sum(p.grad**2 for p in params))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-6)
        for p in params:
            p.grad *= scale

# -----------------------------
# Training
# -----------------------------
batch_size = 4
num_steps = 20000

for step in range(num_steps):

    lr = base_lr * (1 - step / num_steps)  # ✅ LR decay

    total_loss = Value(0.0)

    for _ in range(batch_size):

        doc = random.choice(docs)
        tokens = [BOS] + [stoi[c] for c in doc] + [BOS]
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

    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * (p.grad ** 2)

        m_hat = m[i] / (1 - beta1 ** (step+1))
        v_hat = v[i] / (1 - beta2 ** (step+1))

        p.data -= lr * m_hat / (math.sqrt(v_hat) + eps)
        p.grad = 0.0

    if step % 50 == 0:
        print(f"step {step+1:5d} | loss {total_loss.data:.4f}")

# -----------------------------
# Inference
# -----------------------------
print("\n--- inference ---")

for _ in range(10):

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]

    token_id = BOS
    output = []

    for pos in range(block_size):
        logits = gpt(token_id, pos, keys, values, training=False)
        probs = softmax(logits)

        token_id = sample_with_temp(probs, temp=0.7)

        if token_id == BOS:
            break

        output.append(itos[token_id])

    print("".join(output))
