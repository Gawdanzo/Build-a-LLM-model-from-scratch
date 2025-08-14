llm_pipeline.py — A Unified, Research-Grade Mini-LLM Pipeline

A single-file, reproducible LLM pipeline that goes end-to-end:
data processing → tokenizer → attention & Transformer → GPT training/eval → instruction-style fine-tuning → sampling.
Designed to be academically readable, engineering-sound, and easy to extend.

0) TL;DR Quickstart
Install (minimal deps)
pip install torch tqdm numpy
# (Optional) transformers/peft/wandb if you plan to extend

Data layout (either dataset is fine)
data/
  raw/
    sms_spam_collection/SMSSpamCollection     # optional classic SMS dataset
    instruction-data.json                     # optional instruction-style dataset

Train from scratch
python llm_pipeline.py train --data_dir ./data/raw --epochs 5

Fine-tune on instruction data
python llm_pipeline.py finetune --data_dir ./data/raw \
  --ckpt ./ckpts/best.pth --epochs 3

Sample
python llm_pipeline.py sample --ckpt ./ckpts/best.pth \
  --prompt "Translate to French: Hello world!"

1) Design Principles & Why One File

This file is an evidence artifact: it strips boilerplate while preserving the core research signals reviewers care about:

Scientific clarity: explicit attention math (scaled dot-product) + strict causal masking.

Engineering hygiene: seed control, best/last checkpoints, modular config objects, gradient clipping.

Task unification: both pretraining-style training and instruction fine-tuning use the same NTP objective, avoiding head swaps and target drift.

Extension points: tokenizer, AMP/DDP, PEFT/LoRA, schedulers, long-context strategies—all intentionally easy to plug in.

2) Data Processing (to a Unified LM Objective)

Goal: map heterogeneous sources into a single language modeling corpus.
build_corpus_from_sources() merges:

SMS Spam → rewritten as an instruction triplet:

### Instruction: classify if the SMS is spam or ham
### Input: <sms_text>
### Output: <ham/spam>
<|sep|>


Instruction JSON (instruction-data.json): directly templated as above and appended with <|sep|>.

If neither is present, a tiny toy corpus is injected so the pipeline is always runnable.

Why this matters: pretraining and fine-tuning share the next-token prediction (NTP) objective, which stabilizes optimization on small data.

3) Tokenizer: Minimal Byte-Level

To remain self-contained and reproducible:

Vocabulary: 0..255 plus 4 specials: <|pad|>=256, <|bos|>=257, <|eos|>=258, <|sep|>=259 → vocab_size=260.

ByteTokenizer.encode/decode keeps the implementation compact and dependency-free.

Drop-in upgrade: later you can replace with GPT-2 BPE (HF GPT2TokenizerFast) without changing the training objective.

4) Dataset Slicing

LMDataset constructs (x, y) pairs by sliding a block over the long corpus:

block_size tokens → predict the next token (y is a one-step right-shift of x).

create_dataloaders() performs a seeded 90/10 split, making validation deterministic.

Perplexity ppl = exp(loss) is reported on the validation portion.

Note: truncation limits long-range dependencies to the chosen block_size. Long-context options are discussed in Extensions.

5) Attention & Shapes (what really happens)
Scaled dot-product attention
A
t
t
(
𝑄
,
𝐾
,
𝑉
)
=
s
o
f
t
m
a
x
 ⁣
(
𝑄
𝐾
⊤
𝑑
ℎ
)
𝑉
,
𝑑
ℎ
=
𝑛
embd
𝑛
head
.
Att(Q,K,V)=softmax(
d
h
	​

	​

QK
⊤
	​

)V,d
h
	​

=
n
head
	​

n
embd
	​

	​

.

Implementation details in CausalSelfAttention:

qkv = Linear(n_embd, 3*n_embd) → reshape to (B, n_head, T, head_dim).

Complexity: O(B · T² · n_embd) (attention dominates).

Scaling by √d_h improves numerical stability.

Strict causal masking

A cached lower-triangular mask sets future logits to −inf before softmax.

This is what enforces auto-regression.

Residual path & MLP

Each TransformerBlock = x + Att(LN(x)) then x + MLP(LN(x)).

Dropout regularizes both attention weights and residual.

6) GPT Skeleton (embedding → blocks → head)

GPTConfig captures hyperparameters; GPT composes:

Token embedding + learned positional embedding (block_size positions).

N × TransformerBlock with multi-head attention and GELU MLP.

Weight tying (head.weight = token_emb.weight) for parameter efficiency/generalization.

Initialization: normal(0, 0.02) for Linear and Embeddings.

Forward: returns logits and optional cross-entropy loss if targets supplied.

Generate: temperature scaling, optional top-k filtering, greedy sampling loop.

7) Optimization & Evaluation
Training (train_loop)

AdamW (lr=3e-4, weight_decay=0.01) + grad-norm clipping (1.0).

eval_interval steps → compute validation loss over a capped number of batches;
save best checkpoint when val_loss improves; also save a last snapshot.

Validation perplexity

evaluate_perplexity(...) reports val_loss and ppl (capped to avoid overflow), a standard LM metric.

8) Instruction-Style Fine-Tuning

finetune_loop keeps the same objective (NTP) but trains only on the instruction-flavored corpus.
Rationale: continuous optimization on a consistent target avoids train/inference head mismatch and keeps code lean.
You can later replace the objective for SFT/DPO/RLHF without changing the overall data plumbing.

9) Text Generation

sample_text feeds a prompt through generate:

Temperature < 1 → sharper/greedier; > 1 → more diverse.

Top-k keeps the k highest logits and sets the rest to −inf before softmax.

Input is trimmed to the last block_size tokens at each step to bound memory.

10) Reproducibility

set_seed() controls Python/NumPy/Torch RNGs.

random_split uses a fixed generator for deterministic splits.

Checkpoints store both model weights and the config snapshot (training & model), ensuring a faithful reload months later.

11) Recommended Defaults for NVIDIA RTX 4060

Below are tested-reasonable starting points for 4060-class GPUs. Adjust according to your exact VRAM (8 GB laptop vs 16 GB desktop/Ti) and sequence length needs.

Preset Table
Preset	Target GPU	block_size	n_layer	n_head	n_embd	Global Batch (approx)	Grad Accum	LR	Epochs	Notes
Tiny-8G (default)	RTX 4060 (8 GB)	256	6	8	512	32	1	3e-4	5	Good balance for quick convergence; fits comfortably on 8 GB.
Compact-8G	RTX 4060 (8 GB)	384	6	8	512	24	2	3e-4	6	Slightly longer sequences; use grad-accum=2 to stay in 8 GB.
Small-16G	RTX 4060 Ti (16 GB)	512	8	8	640	32	1	2e-4	6–8	Bigger context & model width; consider AMP for speed.
CPU-Debug	CPU	128	2	4	256	8	1	5e-4	1–2	For functional tests only.

Suggested CLI examples

8 GB laptop (default tiny):

python llm_pipeline.py train \
  --data_dir ./data/raw --epochs 5 \
  --block_size 256 --n_layer 6 --n_head 8 --n_embd 512 \
  --batch_size 32 --lr 3e-4 --eval_interval 200


8 GB with longer sequences (use grad-accum=2):
(Since the file uses per-step batches, emulate accumulation by halving batch_size and doubling eval_interval to keep signal cadence; or extend code to accumulate gradients.)

python llm_pipeline.py train \
  --data_dir ./data/raw --epochs 6 \
  --block_size 384 --n_layer 6 --n_head 8 --n_embd 512 \
  --batch_size 12 --lr 3e-4 --eval_interval 400


16 GB desktop (more capacity):

python llm_pipeline.py train \
  --data_dir ./data/raw --epochs 8 \
  --block_size 512 --n_layer 8 --n_head 8 --n_embd 640 \
  --batch_size 24 --lr 2e-4 --eval_interval 200


Tip (AMP): If you add autocast/GradScaler to the training loop, you can often push batch_size or block_size a notch higher on 8 GB.

12) Training-Curve Templates (Logging + Plotting)

Below are drop-in snippets to log training/validation metrics to a CSV and then visualize curves. They are framework-agnostic and easy to paste into your file.

A) Minimal CSV logger (add to llm_pipeline.py)

Add this near the top:

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


Initialize in train_loop and finetune_loop:

logger = CSVLogger(str(out_dir / "metrics.csv"))


Log inside loops:

# after loss.backward() and optimizer.step()
logger.log(global_step, epoch, "train", float(loss.item()))

# when you compute validation loss:
logger.log(global_step, epoch, "val", float(val_loss))

B) Plotting script (plot_curves.py)

Save this as a separate file and run python plot_curves.py ./ckpts/metrics.csv.

import sys, csv
import matplotlib.pyplot as plt

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

    # 1) Training loss
    plt.figure()
    plt.plot(s, tl)
    plt.xlabel("step"); plt.ylabel("train loss"); plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("train_loss.png", dpi=150)

    # 2) Validation loss
    plt.figure()
    plt.plot(vs, vl)
    plt.xlabel("step"); plt.ylabel("val loss"); plt.title("Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.savefig("val_loss.png", dpi=150)

    # 3) Validation perplexity (derived)
    import math
    ppl = [math.exp(min(20.0, x)) for x in vl]
    plt.figure()
    plt.plot(vs, ppl)
    plt.xlabel("step"); plt.ylabel("val perplexity"); plt.title("Validation Perplexity")
    plt.grid(True, alpha=0.3)
    plt.savefig("val_ppl.png", dpi=150)

if __name__ == "__main__":
    main()


You’ll get train_loss.png, val_loss.png, and val_ppl.png suitable for your application dossier.

13) Extensions (roadmap)

Tokenizer: swap the byte-level tokenizer for GPT-2 BPE to improve text fidelity; update vocab_size accordingly.

Mixed precision: add torch.cuda.amp.autocast() + GradScaler to speed up and save VRAM on 4060.

Schedulers: cosine/warmup schedules for smoother convergence.

Distributed/accumulation: torch.distributed for multi-GPU; gradient accumulation to simulate larger batches on 8 GB.

PEFT/LoRA: freeze backbone and train low-rank adapters in finetune_loop.

Long context: KV-cache reuse, ALiBi/RoPE, or chunk-wise attention to push sequences > 1k.

14) FAQ

Loss doesn’t go down → verify data files exist; try lowering block_size or raising epochs; warmup schedule can help.

Out of memory on 8 GB → reduce block_size and/or batch_size; consider AMP; keep n_embd ≤ 512 for comfort.

Weird tokens in output → expected with byte-level tokenizer (you may see <|bos|> etc.). Use BPE for cleaner text.

Instruction following is weak → fine-tune longer and ensure your prompt matches the template (“Instruction/Input/Output”).

15) Licensing & Citation

License: MIT (or adapt to your institution’s policy).

Citation: please cite this repository in acknowledgements/appendix if it contributes to your work.
