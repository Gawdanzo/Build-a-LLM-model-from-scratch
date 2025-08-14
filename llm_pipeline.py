#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
llm_pipeline.py

A single-file, research-grade pipeline that consolidates:
- Data processing (SMS spam / instruction JSON → plain text corpus)
- Attention module (scaled dot-product, causal mask)
- GPT-like Transformer (from scratch, PyTorch)
- Training / evaluation / generation
- Lightweight fine-tuning on instruction-style data

This file is designed to summarize and professionalize a study-phase codebase
into a PhD-application-ready artifact with clean structure, type hints, and CLI.

Notebooks lineage: 00_data_processing → 01_attention_from_scratch → 
02_gpt_model_from_scratch → 03_finetune_instruction.  (Consolidated here.)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# -----------------------------
# Utils & Reproducibility
# -----------------------------

def set_seed(seed: int = 1337) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Tokenizer: Minimal Byte-level
# -----------------------------
# 为了最小依赖与自包含，这里实现一个 Byte-level tokenizer：
# 0..255 直接对应字节，额外保留若干特殊符号。
# 真实科研可替换为 HuggingFace 的 GPT2Tokenizer/BPE，但本实现足以支撑训练/微调/采样。

SPECIAL_TOKENS = {
    "<|pad|>": 256,
    "<|bos|>": 257,
    "<|eos|>": 258,
    "<|sep|>": 259,     # 用于指令样本拼接
}
ID2SPECIAL = {v: k for k, v in SPECIAL_TOKENS.items()}


class ByteTokenizer:
    """Simple byte-level tokenizer with a handful of special tokens."""

    def __init__(self) -> None:
        self.vocab_size = 260  # 0..255 + 4 specials

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [ord(c) if ord(c) < 256 else ord("?") for c in text]
        if add_bos:
            ids = [SPECIAL_TOKENS["<|bos|>"]] + ids
        if add_eos:
            ids = ids + [SPECIAL_TOKENS["<|eos|>"]]
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        chars = []
        for i in ids:
            if i in ID2SPECIAL:
                # 将特殊符号还原为可见标记
                chars.append(f"{ID2SPECIAL[i]}")
            elif 0 <= i < 256:
                chars.append(chr(i))
            else:
                chars.append("?")
        return "".join(chars)


# -----------------------------
# Data Processing
# -----------------------------

def load_sms_spam(data_dir: Path) -> List[Tuple[str, int]]:
    """Load SMS Spam dataset if exists.

    Expected: data_dir / sms_spam_collection / SMSSpamCollection
    Returns list of (text, label) where label: 0=ham, 1=spam
    """
    p = data_dir / "sms_spam_collection" / "SMSSpamCollection"
    items: List[Tuple[str, int]] = []
    if not p.exists():
        return items
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        head, text = line.split("\t", 1)
        label = 1 if head.strip().lower() == "spam" else 0
        items.append((text, label))
    return items


def load_instruction_json(data_dir: Path) -> List[Dict]:
    """Load instruction dataset if exists.

    Expected: data_dir / instruction-data.json
    Each item: {"instruction": str, "input": str, "output": str}
    """
    p = data_dir / "instruction-data.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8"))
    norm = []
    for ex in data:
        inst = ex.get("instruction", "").strip()
        inp = ex.get("input", "").strip()
        out = ex.get("output", "").strip()
        if not inst and not inp and not out:
            continue
        norm.append({"instruction": inst, "input": inp, "output": out})
    return norm


def build_corpus_from_sources(data_dir: Path) -> str:
    """Merge SMS spam and instruction samples into a single text corpus.

    - SMS: 转化为指令风格：
      ### Instruction: classify if the SMS is spam or ham
      ### Input: <sms_text>
      ### Output: <ham/spam>

    - Instruction JSON: 直接拼接：
      ### Instruction: {instruction}
      ### Input: {input}
      ### Output: {output}
    """
    sms = load_sms_spam(data_dir)
    inst = load_instruction_json(data_dir)
    pieces: List[str] = []

    if sms:
        for text, label in sms[:5000]:  # 若数据很大，控制规模
            label_str = "spam" if label == 1 else "ham"
            s = (
                "### Instruction: classify if the SMS is spam or ham\n"
                f"### Input: {text}\n"
                f"### Output: {label_str}\n"
                "<|sep|>\n"
            )
            pieces.append(s)

    if inst:
        for ex in inst:
            s = (
                f"### Instruction: {ex['instruction']}\n"
                f"### Input: {ex['input']}\n"
                f"### Output: {ex['output']}\n"
                "<|sep|>\n"
            )
            pieces.append(s)

    if not pieces:
        # 兜底：若没有任何数据，给一个小 toy 语料，保证代码可跑通
        pieces = [
            "### Instruction: echo the input\n### Input: hello\n### Output: hello\n<|sep|>\n",
            "### Instruction: translate to French\n### Input: Good morning\n### Output: Bonjour\n<|sep|>\n",
        ]
    corpus = "".join(pieces)
    return corpus


class LMDataset(Dataset):
    """Turn a long corpus into next-token-prediction blocks."""

    def __init__(self, text: str, tokenizer: ByteTokenizer, block_size: int = 256) -> None:
        super().__init__()
        self.tok = tokenizer
        self.block_size = block_size
        ids = self.tok.encode(text, add_bos=False, add_eos=False)
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y


# -----------------------------
# Attention & GPT Blocks
# -----------------------------

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with simple PyTorch ops."""

    def __init__(self, n_embd: int, n_head: int, attn_dropout: float, resid_dropout: float, bias: bool):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)
        self.register_buffer("mask", None, persistent=False)

    def _causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        if self.mask is None or self.mask.size(0) < T:
            mask = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))
            self.mask = mask
        return self.mask[:T, :T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.qkv(x)  # (B, T, 3C)
        q, k, v = qkv.split(C, dim=2)
        # reshape to heads
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hd)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, nh, T, T)
        mask = self._causal_mask(T, x.device)
        att = att.masked_fill(~mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = att @ v  # (B, nh, T, hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float, bias: bool):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd, bias=bias),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd, bias=bias),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, attn_dropout: float, resid_dropout: float, bias: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, attn_dropout, resid_dropout, bias)
        self.ln2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, resid_dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTConfig:
    def __init__(
        self,
        vocab_size: int = 260,
        block_size: int = 256,
        n_layer: int = 6,
        n_head: int = 8,
        n_embd: int = 512,
        attn_dropout: float = 0.1,
        resid_dropout: float = 0.1,
        emb_dropout: float = 0.1,
        bias: bool = True,
        tie_weights: bool = True,
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.emb_dropout = emb_dropout
        self.bias = bias
        self.tie_weights = tie_weights


class GPT(nn.Module):
    """A compact GPT-like language model."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.emb_dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg.n_embd, cfg.n_head, cfg.attn_dropout, cfg.resid_dropout, cfg.bias) for _ in range(cfg.n_layer)]
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        if cfg.tie_weights:
            self.head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # report params
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[GPT] params: {n_params/1e6:.2f}M")

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.cfg.block_size, "sequence too long"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 64, temperature: float = 1.0, top_k: Optional[int] = None) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
        return idx


# -----------------------------
# Training / Evaluation
# -----------------------------

@dataclass
class TrainConfig:
    data_dir: str
    out_dir: str = "./ckpts"
    block_size: int = 256
    batch_size: int = 32
    lr: float = 3e-4
    epochs: int = 5
    seed: int = 1337
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 512
    grad_clip: float = 1.0
    weight_decay: float = 0.01
    eval_interval: int = 200
    device: str = "auto"  # "cpu"/"cuda"/"auto"

    def device_obj(self) -> torch.device:
        if self.device == "cpu":
            return torch.device("cpu")
        if self.device == "cuda":
            return torch.device("cuda")
        return get_device()


def create_dataloaders(corpus: str, tokenizer: ByteTokenizer, block_size: int, batch_size: int, split_ratio: float = 0.9) -> Tuple[DataLoader, DataLoader]:
    ds = LMDataset(corpus, tokenizer, block_size)
    n = len(ds)
    n_train = int(n * split_ratio)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(123))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader


def estimate_loss(model: GPT, loader: DataLoader, device: torch.device, max_batches: int = 50) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(device)
            y = y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def train_loop(cfg: TrainConfig) -> str:
    set_seed(cfg.seed)
    device = cfg.device_obj()
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build corpus & tokenizer
    tokenizer = ByteTokenizer()
    corpus = build_corpus_from_sources(data_dir)

    # 2) Dataloaders
    train_loader, val_loader = create_dataloaders(corpus, tokenizer, cfg.block_size, cfg.batch_size)

    # 3) Model & optim
    gpt_cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
    )
    model = GPT(gpt_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = out_dir / "best.pth"

    # 4) Training
    global_step = 0
    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}")
        for x, y in pbar:
            x = x.to(device); y = y.to(device)
            logits, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            global_step += 1
            if global_step % cfg.eval_interval == 0:
                val_loss = estimate_loss(model, val_loader, device)
                pbar.set_postfix({"train": f"{loss.item():.3f}", "val": f"{val_loss:.3f}"})
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "gpt_cfg": gpt_cfg.__dict__}, best_path)

    # 最终保存
    final_path = out_dir / "last.pth"
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg), "gpt_cfg": gpt_cfg.__dict__}, final_path)
    print(f"[train] best_val={best_val:.4f}, best_ckpt={best_path}")
    return str(best_path)


# -----------------------------
# Fine-tuning
# -----------------------------

@dataclass
class FinetuneConfig(TrainConfig):
    ckpt: str = ""  # path to a pre-trained checkpoint


def load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> GPT:
    obj = torch.load(ckpt_path, map_location=device)
    gpt_cfg = GPTConfig(**obj["gpt_cfg"])
    model = GPT(gpt_cfg).to(device)
    model.load_state_dict(obj["model"])
    return model


def finetune_loop(cfg: FinetuneConfig) -> str:
    assert cfg.ckpt, "Please provide --ckpt for finetuning"
    set_seed(cfg.seed)
    device = cfg.device_obj()
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = ByteTokenizer()
    corpus = build_corpus_from_sources(data_dir)  # 一般这里仅使用指令数据

    train_loader, val_loader = create_dataloaders(corpus, tokenizer, cfg.block_size, cfg.batch_size)

    model = load_model_from_ckpt(Path(cfg.ckpt), device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_path = Path(cfg.out_dir) / "best_finetune.pth"

    global_step = 0
    for epoch in range(cfg.epochs):
        pbar = tqdm(train_loader, desc=f"finetune {epoch+1}/{cfg.epochs}")
        for x, y in pbar:
            x = x.to(device); y = y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            global_step += 1
            if global_step % cfg.eval_interval == 0:
                val_loss = estimate_loss(model, val_loader, device)
                pbar.set_postfix({"train": f"{loss.item():.3f}", "val": f"{val_loss:.3f}"})
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save({"model": model.state_dict(), "gpt_cfg": model.cfg.__dict__}, best_path)

    print(f"[finetune] best_val={best_val:.4f}, best_ckpt={best_path}")
    return str(best_path)


# -----------------------------
# Evaluation & Generation
# -----------------------------

@torch.no_grad()
def evaluate_perplexity(ckpt: str, data_dir: str, block_size: int = 256, batch_size: int = 32) -> float:
    device = get_device()
    tokenizer = ByteTokenizer()
    corpus = build_corpus_from_sources(Path(data_dir))
    _, val_loader = create_dataloaders(corpus, tokenizer, block_size, batch_size)
    model = load_model_from_ckpt(Path(ckpt), device)
    loss = estimate_loss(model, val_loader, device, max_batches=200)
    ppl = float(math.exp(min(20.0, loss)))  # 避免溢出
    print(f"[eval] val_loss={loss:.4f}, ppl≈{ppl:.2f}")
    return ppl


@torch.no_grad()
def sample_text(ckpt: str, prompt: str, max_new_tokens: int = 80, temperature: float = 0.9, top_k: Optional[int] = 30) -> str:
    device = get_device()
    tok = ByteTokenizer()
    model = load_model_from_ckpt(Path(ckpt), device)
    model.eval()
    x = torch.tensor([tok.encode(prompt, add_bos=True)], dtype=torch.long, device=device)
    y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    text = tok.decode(y[0].tolist())
    return text


# -----------------------------
# CLI
# -----------------------------

def cli() -> None:
    p = argparse.ArgumentParser(description="Unified LLM pipeline: data→attention→GPT→finetune")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train GPT from scratch on merged corpus")
    p_train.add_argument("--data_dir", type=str, default="./data/raw")
    p_train.add_argument("--out_dir", type=str, default="./ckpts")
    p_train.add_argument("--epochs", type=int, default=5)
    p_train.add_argument("--batch_size", type=int, default=32)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--block_size", type=int, default=256)
    p_train.add_argument("--n_layer", type=int, default=6)
    p_train.add_argument("--n_head", type=int, default=8)
    p_train.add_argument("--n_embd", type=int, default=512)
    p_train.add_argument("--eval_interval", type=int, default=200)
    p_train.add_argument("--seed", type=int, default=1337)
    p_train.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # finetune
    p_ft = sub.add_parser("finetune", help="Finetune on instruction-style data")
    p_ft.add_argument("--data_dir", type=str, default="./data/raw")
    p_ft.add_argument("--ckpt", type=str, required=True)
    p_ft.add_argument("--out_dir", type=str, default="./ckpts")
    p_ft.add_argument("--epochs", type=int, default=3)
    p_ft.add_argument("--batch_size", type=int, default=32)
    p_ft.add_argument("--lr", type=float, default=1e-4)
    p_ft.add_argument("--block_size", type=int, default=256)
    p_ft.add_argument("--eval_interval", type=int, default=200)
    p_ft.add_argument("--seed", type=int, default=2025)
    p_ft.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate perplexity of a checkpoint")
    p_eval.add_argument("--data_dir", type=str, default="./data/raw")
    p_eval.add_argument("--ckpt", type=str, required=True)
    p_eval.add_argument("--block_size", type=int, default=256)
    p_eval.add_argument("--batch_size", type=int, default=32)

    # sample
    p_gen = sub.add_parser("sample", help="Generate text from a checkpoint")
    p_gen.add_argument("--ckpt", type=str, required=True)
    p_gen.add_argument("--prompt", type=str, default="### Instruction: echo the input\n### Input: hello\n### Output: ")
    p_gen.add_argument("--max_new_tokens", type=int, default=80)
    p_gen.add_argument("--temperature", type=float, default=0.9)
    p_gen.add_argument("--top_k", type=int, default=30)

    args = p.parse_args()
    if args.cmd == "train":
        cfg = TrainConfig(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            eval_interval=args.eval_interval,
            seed=args.seed,
            device=args.device,
        )
        best = train_loop(cfg)
        print(best)

    elif args.cmd == "finetune":
        cfg = FinetuneConfig(
            data_dir=args.data_dir,
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            block_size=args.block_size,
            eval_interval=args.eval_interval,
            seed=args.seed,
            device=args.device,
            ckpt=args.ckpt,
        )
        best = finetune_loop(cfg)
        print(best)

    elif args.cmd == "eval":
        evaluate_perplexity(args.ckpt, args.data_dir, args.block_size, args.batch_size)

    elif args.cmd == "sample":
        text = sample_text(args.ckpt, args.prompt, args.max_new_tokens, args.temperature, args.top_k)
        print(text)


if __name__ == "__main__":
    cli()
