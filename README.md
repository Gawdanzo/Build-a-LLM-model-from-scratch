# `llm_pipeline.py` — A Unified, Research-Grade Mini-LLM Pipeline

> A single-file, **reproducible** pipeline that goes end-to-end:  
> **data processing → tokenizer → attention & Transformer → GPT training/eval → instruction-style fine-tuning → sampling**.  
> Written for academic readability and engineering rigor; easy to extend for experiments.

---

## Table of Contents
- [0) Quickstart](#0-quickstart)
- [1) Project Rationale](#1-project-rationale)
- [2) Data Processing](#2-data-processing)
- [3) Tokenizer (Byte-Level)](#3-tokenizer-byte-level)
- [4) Dataset Slicing & Dataloaders](#4-dataset-slicing--dataloaders)
- [5) Attention & Transformer Blocks](#5-attention--transformer-blocks)
- [6) GPT Skeleton](#6-gpt-skeleton)
- [7) Optimization & Evaluation](#7-optimization--evaluation)
- [8) Instruction-Style Fine-Tuning](#8-instruction-style-fine-tuning)
- [9) Text Generation](#9-text-generation)
- [10) Reproducibility Policy](#10-reproducibility-policy)
- [11) RTX 4060 Default Presets](#11-rtx-4060-default-presets)
- [12) Training Curve Templates](#12-training-curve-templates)
- [13) Extensions (Roadmap)](#13-extensions-roadmap)
- [14) FAQ](#14-faq)
- [15) License & Citation](#15-license--citation)

---

## 0) Quickstart

### Install (minimal deps)
```bash
pip install torch tqdm numpy
# Optional: transformers peft wandb matplotlib
```

### Data layout (either dataset is fine)
```
data/
  raw/
    sms_spam_collection/SMSSpamCollection     # optional classic SMS dataset
    instruction-data.json                     # optional instruction-style dataset
```

### Train from scratch
```bash
python llm_pipeline.py train --data_dir ./data/raw --epochs 5
```

### Fine-tune on instruction data
```bash
python llm_pipeline.py finetune --data_dir ./data/raw   --ckpt ./ckpts/best.pth --epochs 3
```

### Evaluate perplexity
```bash
python llm_pipeline.py eval --data_dir ./data/raw --ckpt ./ckpts/best.pth
```

### Sample generation
```bash
python llm_pipeline.py sample --ckpt ./ckpts/best.pth   --prompt "Translate to French: Hello world!"
```

---

## 1) Project Rationale

This single file is an **evidence artifact**: it removes boilerplate while keeping the *research-critical* parts explicit.

- **Scientific clarity**: scaled dot-product attention and strict causal masks are implemented explicitly, with shape logic kept readable.
- **Engineering hygiene**: fixed seeds, deterministic splits, `best.pth` + `last.pth`, config snapshots, gradient clipping.
- **Unified objective**: both pretraining-style training and instruction fine-tuning use **next-token prediction (NTP)** to avoid target drift.
- **Extension hooks**: tokenizer swap (BPE), AMP/DDP, PEFT/LoRA, schedulers, long-context methods—added with minimal edits.

---

## 2) Data Processing

`build_corpus_from_sources(data_dir)` merges heterogeneous sources into a **single LM corpus**:

- **SMS Spam** → rewritten as instruction triplets for compatibility with NTP:
  ```
  ### Instruction: classify if the SMS is spam or ham
  ### Input: <sms_text>
  ### Output: <ham/spam>
  <|sep|>
  ```
- **Instruction JSON** (`instruction-data.json`): templated equally and separated with `<|sep|>`.

If neither exists, a tiny toy corpus is injected so all commands remain runnable.

**Why it matters:** Training and fine-tuning share the same LM objective; no head swapping, no mismatch between pretraining and adaptation.

---

## 3) Tokenizer (Byte-Level)

A compact, dependency-free **byte-level** tokenizer keeps the repo self-contained:

- Vocabulary: `0..255` plus 4 specials — `<|pad|>=256`, `<|bos|>=257`, `<|eos|>=258`, `<|sep|>=259` → `vocab_size=260`.
- `ByteTokenizer.encode/decode` keeps the implementation minimal and reproducible.
- **Drop-in upgrade**: later replace with **GPT-2 BPE** (HF `GPT2TokenizerFast`) with minimal changes to training code.

---

## 4) Dataset Slicing & Dataloaders

`LMDataset` slides a `block_size` window across the corpus to construct `(x, y)` pairs:
- `y` is a one-step right-shifted copy of `x` → **next-token prediction** objective.
- `create_dataloaders()` performs a seeded 90/10 split, making validation deterministic.
- Perplexity `ppl = exp(loss)` is reported on the validation split.

> Truncation limits long-range dependencies to the chosen `block_size`. Consider long-context strategies (see Extensions).

---

## 5) Attention & Transformer Blocks

### Scaled dot-product attention
\[
\mathrm{Att}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}\right)V,\quad d_h=\frac{n_{\text{embd}}}{n_{\text{head}}}.
\]
Implementation highlights (`CausalSelfAttention`):
- `qkv = Linear(n_embd, 3*n_embd)` → reshape `(B, n_head, T, head_dim)`.
- Complexity: **O(B · T² · n_embd)** (attention dominates).
- Scaling by `√d_h` improves numerical stability.

### Strict causal masking
- Cached lower-triangular mask sets future logits to `−inf` before softmax → proper auto-regression.

### Residual & MLP
- Each `TransformerBlock` = `x + Att(LN(x))` then `x + MLP(LN(x))`; Dropout regularizes both attention weights and residual path.

---

## 6) GPT Skeleton

`GPTConfig` captures hyperparameters; `GPT` composes:

- **Token embedding** + **learned positional embedding** (`block_size` positions).
- **N × TransformerBlock** with multi-head attention and GELU MLP.
- **Weight tying** (`head.weight = token_emb.weight`) for parameter efficiency/generalization.
- **Initialization**: `normal(0, 0.02)` for Linear and Embeddings.
- **Forward** returns logits and optional cross-entropy loss; **Generate** supports temperature & top-k sampling.

---

## 7) Optimization & Evaluation

### Training (`train_loop`)
- **AdamW** (`lr=3e-4`, `weight_decay=0.01`) + **grad-norm clipping** (`1.0`).
- `eval_interval` steps → compute validation loss on a capped number of batches; save **best** checkpoint when `val_loss` improves; keep **last** snapshot too.

### Validation perplexity
- `evaluate_perplexity(...)` reports `val_loss` and `ppl` (capped to avoid overflow), a standard LM metric.

---

## 8) Instruction-Style Fine-Tuning

`finetune_loop` keeps the **same NTP objective** but trains on the instruction-flavored corpus:
- Stable objective continuity (no head swapping).
- Easy to extend to SFT/DPO/RLHF later by swapping objective and batch builders.

---

## 9) Text Generation

`sample_text` feeds a prompt through `generate`:
- **Temperature** < 1 → sharper/greedier; > 1 → more diverse.
- **Top-k** keeps the k highest logits and sets the rest to `−inf`.
- Inputs are trimmed to the last `block_size` tokens per step to bound memory.

---

## 10) Reproducibility Policy

- `set_seed()` controls Python/NumPy/Torch RNGs.
- `random_split` uses a fixed generator → deterministic splits.
- Checkpoints store **weights + config snapshots** (training/model) for faithful reloads.

---

## 11) RTX 4060 Default Presets

Below are **sane starting points** for 4060-class GPUs. Adjust for 8 GB vs 16 GB and your desired sequence length.

| Preset | Target GPU | `block_size` | `n_layer` | `n_head` | `n_embd` | Global Batch (approx) | Grad Accum | LR | Epochs | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Tiny-8G** (default) | RTX 4060 (8 GB) | 256 | 6 | 8 | 512 | 32 | 1 | 3e-4 | 5 | Good balance; fits comfortably on 8 GB. |
| **Compact-8G** | RTX 4060 (8 GB) | 384 | 6 | 8 | 512 | 24 | 2 | 3e-4 | 6 | Longer sequences; emulate grad-accum=2 by smaller per-step batch. |
| **Small-16G** | RTX 4060 Ti (16 GB) | 512 | 8 | 8 | 640 | 32 | 1 | 2e-4 | 6–8 | Bigger context & width; add AMP for speed. |
| **CPU-Debug** | CPU | 128 | 2 | 4 | 256 | 8 | 1 | 5e-4 | 1–2 | Functional tests only. |

**CLI examples**

8 GB laptop (default tiny):
```bash
python llm_pipeline.py train   --data_dir ./data/raw --epochs 5   --block_size 256 --n_layer 6 --n_head 8 --n_embd 512   --batch_size 32 --lr 3e-4 --eval_interval 200
```

16 GB desktop (more capacity):
```bash
python llm_pipeline.py train   --data_dir ./data/raw --epochs 8   --block_size 512 --n_layer 8 --n_head 8 --n_embd 640   --batch_size 24 --lr 2e-4 --eval_interval 200
```

> **Tip (AMP)**: If you add autocast/GradScaler to the loop, you can often push `batch_size` or `block_size` higher on 8 GB.

---

## 12) Training Curve Templates

### A) Minimal CSV logger (paste into the file)
```python
import csv
from datetime import datetime

class CSVLogger:
    def __init__(self, path: str):
        self.path = path
        with open(self.path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "epoch", "split", "loss", "timestamp"])

    def log(self, step: int, epoch: int, split: str, loss: float):
        with open(self.path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([step, epoch, split, f"{loss:.6f}", datetime.utcnow().isoformat()])
```

Use inside loops:
```python
logger = CSVLogger(str(out_dir / "metrics.csv"))
logger.log(global_step, epoch, "train", float(loss.item()))
logger.log(global_step, epoch, "val", float(val_loss))
```

### B) Plotting script (`plot_curves.py`)
```python
import sys, csv
import matplotlib.pyplot as plt
import math

def load_metrics(path):
    steps, train_loss, val_steps, val_loss = [], [], [], []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["split"] == "train":
                steps.append(int(row["step"]))
                train_loss.append(float(row["loss"]))
            elif row["split"] == "val":
                val_steps.append(int(row["step"]))
                val_loss.append(float(row["loss"]))
    return steps, train_loss, val_steps, val_loss

def main():
    path = sys.argv[1]
    s, tl, vs, vl = load_metrics(path)

    plt.figure(); plt.plot(s, tl)
    plt.xlabel("step"); plt.ylabel("train loss"); plt.title("Training Loss"); plt.grid(True, alpha=0.3)
    plt.savefig("train_loss.png", dpi=150)

    plt.figure(); plt.plot(vs, vl)
    plt.xlabel("step"); plt.ylabel("val loss"); plt.title("Validation Loss"); plt.grid(True, alpha=0.3)
    plt.savefig("val_loss.png", dpi=150)

    ppl = [math.exp(min(20.0, x)) for x in vl]
    plt.figure(); plt.plot(vs, ppl)
    plt.xlabel("step"); plt.ylabel("val perplexity"); plt.title("Validation Perplexity"); plt.grid(True, alpha=0.3)
    plt.savefig("val_ppl.png", dpi=150)

if __name__ == "__main__":
    main()
```

---

## 13) Extensions (Roadmap)

- **Tokenizer**: swap byte-level for **GPT-2 BPE** (`GPT2TokenizerFast`)—update `vocab_size` and encode/decode.
- **Mixed precision**: add `torch.cuda.amp.autocast()` + `GradScaler` to save VRAM on 4060 and speed up training.
- **Schedulers**: cosine or warmup schedules for smoother convergence.
- **Distributed/accumulation**: `torch.distributed` for multi-GPU; gradient accumulation for larger effective batches.
- **PEFT/LoRA**: freeze backbone, add low-rank adapters in `finetune_loop`.
- **Long context**: KV cache reuse, ALiBi/RoPE, or chunk-wise attention for >1k tokens.

---

## 14) FAQ

- **Loss doesn’t go down** → verify files exist; try fewer params or more epochs; add warmup.
- **OOM on 8 GB** → reduce `block_size`/`batch_size`; adopt AMP; keep `n_embd ≤ 512`.
- **Strange tokens in output** → expected with byte-level tokenizer; switch to BPE for cleaner text.
- **Weak instruction following** → fine-tune longer; match the prompt template (Instruction/Input/Output).

---

## 15) License & Citation

- **License**: MIT (or adapt to your institution’s policy).
- **Citation**: If this framework helps your research, please cite the repository in acknowledgements/appendix.
