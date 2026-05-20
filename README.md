# μGPT-A-Minimal-Transformer-microGPT-Built-Completely-from-Scratch-with-minimal-computational-overhead
Pure Python + NumPy implementation of a Transformer-based GPT architecture developed for educational research, mathematical interpretability, and low-level understanding of Large Language Models.

## 📌 Overview

microGPT is a research-oriented implementation of a GPT-style Transformer architecture developed entirely from scratch without using deep learning frameworks such as PyTorch or TensorFlow.

The project progressively evolves from:
```
a fully manual scalar-based autograd engine,

to a handcrafted Transformer architecture,

to a vectorized NumPy-powered mini language model capable of conversational inference.
```
The primary objective of this work is to demonstrate the complete internal mechanics of modern Generative Pre-trained Transformers (GPTs) using minimal dependencies and mathematically transparent implementations.

This repository contains:
```
manual automatic differentiation,

self-attention implementation,

RMSNorm/LayerNorm,

Transformer residual blocks,

Adam optimizer,

token and positional embeddings,

nucleus/top-k sampling,

chatbot inference engine,

NumPy-based fast training pipeline,

mathematical derivations and references.
```
---

## 🎯 Research Objectives

The project was designed to achieve the following goals:
```
Understand GPT internals at algorithmic level

Build Transformer architecture without ML frameworks

Implement self-attention mathematically from scratch

Create a lightweight educational LLM pipeline

Explore optimization and sampling techniques

Demonstrate progression from scalar autograd → vectorized Transformers

Provide a reproducible research implementation for students and researchers
```
---

## 🧠 Core Features
Implemented Components

✅ Pure Python Autograd Engine

✅ Transformer Self-Attention

✅ Multi-Head Attention

✅ RMSNorm / LayerNorm

✅ Residual Connections

✅ Feed Forward Network (MLP)

✅ Adam Optimizer

✅ Gradient Clipping

✅ Dropout Regularization

✅ Temperature Sampling

✅ Top-k Sampling

✅ Nucleus (Top-p) Sampling

✅ Character-Level GPT

✅ Token-Level Chatbot

✅ NumPy Vectorized Training

✅ Model Checkpoint Saving/Loading

✅ Interactive Chat Interface
---

## 🏗️ Project Architecture
```
microGPT/
│
├── microgpt_base.py        # Fundamental GPT implementation
├── microgpt1.py            # Stabilized Transformer version
├── microgpt2.py            # Improved optimizer + sampling
├── model_runner.py         # Lightweight inference engine
├── train.py                # NumPy-based Transformer trainer
├── chat.py                 # Interactive chatbot interface
│
├── model/
│   ├── weights.pkl
│   ├── vocab.json
│   └── meta.json
│
├── data/
│   └── healthcare_qa_dataset.txt
│
├── input.txt               # 32k+ names dataset
│
├── microGPT_cheatsheet.pdf
├── microGPT_math_detailed.pdf
├── microGPT_math_reference.pdf
│
└── README.md
```
---

## 🔬 Research Evolution
### Phase 1 — microgpt_base.py

Minimal Transformer implementation with:
```
custom autograd,
scalar computation graph,
self-attention,
character-level generation.
```
This file demonstrates the most atomic implementation of GPT mechanics.

### Phase 2 — microgpt1.py

Introduced:
```
gradient clipping,
dropout,
training stabilization,
safer numerical operations,
improved optimization.
```
### Phase 3 — microgpt2.py

Introduced:
```
improved weight initialization,
nucleus (top-p) sampling,
EMA loss smoothing,
better inference quality.
```
### Phase 4 — model_runner.py

Separated inference from training:
```
lightweight model loading,
optimized generation pipeline,
reduced memory overhead.
```
### Phase 5 — train.py + chat.py

Transitioned to:
```
token-level Transformer,
NumPy vectorization,
healthcare QA chatbot training,
interactive conversational inference.
```
---

## ⚙️ Transformer Mathematics

The implementation follows standard Transformer equations.

---

### Token Embedding

Token embeddings map tokens into dense vector representations.

```math
e_i = E[token_i]
```

---

### Positional Encoding

Positional embeddings inject sequence order information.

```math
x_i = e_i + p_i
```

---

### Scaled Dot-Product Attention

Core Transformer attention mechanism.

```math
\mathrm{Attention}(Q,K,V)
=
\mathrm{softmax}
\left(
\frac{QK^T}{\sqrt{d_k}}
\right)V
```

---

### Softmax Function

Converts logits into probability distributions.

```math
\mathrm{softmax}(z_i)
=
\frac{e^{z_i}}
{\sum_j e^{z_j}}
```

---

### Feed Forward Network (FFN)

Applies non-linear transformation.

```math
\mathrm{FFN}(x)
=
\max(0, xW_1 + b_1)W_2 + b_2
```

---

### Residual Connections

Helps preserve gradients during training.

```math
y = x + F(x)
```

---

### Layer Normalization

Stabilizes activations during training.

```math
\mathrm{LN}(x)
=
\gamma
\left(
\frac{x - \mu}
{\sqrt{\sigma^2 + \epsilon}}
\right)
+ \beta
```

where:

```math
\mu = \mathrm{mean}(x)
```

```math
\sigma^2 = \mathrm{variance}(x)
```

---

### Cross Entropy Loss

Training objective for next-token prediction.

```math
L = -\log(p_{correct})
```

---

### Gradient Descent Update

Parameter optimization step.

```math
\theta \leftarrow \theta - \eta \nabla L(\theta)
```

where:

- $\theta$ = model parameters
- $\eta$ = learning rate
- $\nabla L(\theta)$ = gradient of loss

---

### Multi-Head Attention

The model splits attention into multiple heads.

```math
\mathrm{MultiHead}(Q,K,V)
=
\mathrm{Concat}(head_1, ..., head_h)W^O
```

where:

```math
head_i
=
\mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
```

---

### RMS Normalization

Used in earlier microGPT versions.

```math
\mathrm{RMSNorm}(x)
=
\frac{x}
{\sqrt{\frac{1}{n}\sum_i x_i^2 + \epsilon}}
```

---

## Mathematical Pipeline Summary

```text
Input Tokens
      ↓
Token Embedding
      ↓
Positional Encoding
      ↓
Self-Attention
      ↓
Residual Connection
      ↓
Normalization
      ↓
Feed Forward Network
      ↓
Softmax
      ↓
Cross Entropy Loss
      ↓
Backpropagation
      ↓
Adam Optimization
```
---

## 📚 Mathematical Documentation

The repository contains detailed mathematical references:

File	Description
|-----------------------------------------------------------------------------|
|microGPT_cheatsheet.pdf	    |   One-page Transformer formula summary        |
|microGPT_math_reference.pdf	|   Mathematical functions & derivations        |
|microGPT_math_detailed.pdf	  | Detailed Transformer mathematical explanations|
|-----------------------------------------------------------------------------|

These documents explain:
```
embeddings,
attention derivation,
normalization,
residual learning,
optimization,
loss functions.
```
---

## 🧪 Datasets

### 1. Character-Level Dataset
```
input.txt
```
Contains:

32,000+ names
used for character-level language modeling.

### 2. Healthcare QA Dataset
```
data/healthcare_qa_dataset.txt
```
Used for:

token-level conversational training,
chatbot response generation.
---

## 🚀 Training

Train NumPy Chatbot
```
python train.py
```
Generated model files:
```
model/
├── weights.pkl
├── vocab.json
└── meta.json
```
---

## 💬 Chat Inference
```
python chat.py
```
Example:

You: What is diabetes?
Bot: diabetes is a chronic condition affecting blood sugar levels ...
---

## 🧠 Character-Level Generation
```
python microgpt2.py
python model_runner.py
```
Example generated outputs:

alvion
mariel
davren
kelnor
.....
---

## 📊 Technical Specifications
| Component           | Value                       |
| ------------------- | --------------------------- |
| Embedding Dimension | 16 / 96                     |
| Attention Heads     | 4                           |
| Transformer Layers  | 1–2                         |
| Context Window      | 12–16                       |
| Optimizer           | Adam                        |
| Sampling            | Temperature / Top-k / Top-p |
| Frameworks          | Pure Python + NumPy         |
| Hardware            | CPU Only                    |
---

## 🔍 Key Research Contributions
### 1. Framework-Free GPT

Implements GPT mechanics without:
```
PyTorch
TensorFlow
JAX
```
### 2. Educational Transparency

Every mathematical operation is explicitly visible.

Ideal for:
```
Transformer learning,
academic demonstrations,
research education.
```
### 3. Custom Autograd Engine

Implements reverse-mode automatic differentiation from scratch.

4. Progressive Transformer Evolution

Demonstrates:
```
scalar autograd GPT,
stabilized training,
vectorized Transformer inference.
```
---
### ⚠️ Current Limitations

This repository is educational/research focused and intentionally simplified.

Limitations include:
```
No GPU acceleration
No mixed precision
No distributed training
Limited context window
No causal masking in some versions
Partial backpropagation in NumPy trainer
Small-scale datasets
Single-machine CPU training
```
---

## 🔮 Future Work

Planned improvements:
```
Full Transformer backpropagation
Causal attention masking
Multi-head vectorized attention
Byte Pair Encoding (BPE)
CUDA/GPU acceleration
Quantization support
LoRA fine-tuning
RLHF experimentation
Retrieval-Augmented Generation (RAG)
```
---

## 🎓 Educational Applications

microGPT can be used for:
```
Transformer learning
NLP research education
Deep learning coursework
Attention visualization
GPT internals study
Autograd understanding
AI architecture demonstrations
```
---

## 👨‍💻 Author

Developed by Sanchayan Ghosh. Contact me on sanchayan.ghosh2022@uem.edu.in

Research Focus:
```
Transformer architectures
LLM security
Prompt leakage defense
Lightweight AI systems
Educational AI frameworks
```
---

## 📜 License

This project is released for:

research,
education,
experimentation,
academic learning.

Please cite appropriately if used in academic or research environments. Protected under MIT licensing.

This research work has been submitted as a preprint on 

Zenodo: https://zenodo.org/records/20302616

OpenAIRE Explore: https://explore.openaire.eu/search/result?pid=10.5281%2Fzenodo.20302616

and sumitted on SPRINGER NATURE: https://www.editorialmanager.com/vico/default2.aspx ...........

## ⭐ Acknowledgements

Inspired by:
```
Transformer architecture research
GPT-style autoregressive models
Educational minimal AI implementations
Open-source AI research community
```
Special appreciation to the foundational works on:
```
Transformers,
attention mechanisms,
language modeling,
automatic differentiation.
```
