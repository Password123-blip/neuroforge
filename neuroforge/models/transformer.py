"""A decoder-only transformer (GPT-style), built entirely from scratch.

Includes multi-head causal self-attention, a pre-norm residual block, a
position-wise feed-forward network, learned positional embeddings and
autoregressive sampling with temperature and top-k filtering.
"""

from __future__ import annotations

import numpy as np

from ..tensor import Tensor, no_grad
from ..nn.activations import GELU
from ..nn.layers import Dropout, Embedding, Linear
from ..nn.module import Module, ModuleList
from ..nn.norm import LayerNorm
from ..nn import init

__all__ = [
    "MultiHeadAttention", "FeedForward", "TransformerBlock",
    "GPT", "TransformerClassifier",
]


class MultiHeadAttention(Module):
    """Scaled dot-product attention, split across several heads.

    Each head projects the input into its own query/key/value subspace of size
    ``embed_dim / num_heads``; the heads are computed in parallel and their
    outputs concatenated and mixed by an output projection.
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 bias: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / np.sqrt(self.head_dim)

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = Dropout(dropout)
        self.resid_dropout = Dropout(dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        x = x.reshape(b, t, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)          # (B, H, T, head_dim)

    def forward(self, x: Tensor, mask: np.ndarray = None) -> Tensor:
        b, t, c = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = (q @ k.transpose(0, 1, 3, 2)) * self.scale     # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)

        attn = scores.softmax(axis=-1)
        attn = self.attn_dropout(attn)

        out = attn @ v                                          # (B, H, T, head_dim)
        out = out.transpose(0, 2, 1, 3).reshape(b, t, c)        # (B, T, C)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(Module):
    """Position-wise MLP: expand, activate, project back."""

    def __init__(self, embed_dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or 4 * embed_dim
        self.fc1 = Linear(embed_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, embed_dim)
        self.act = GELU()
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class TransformerBlock(Module):
    """Pre-norm transformer block: attention + feed-forward, both residual."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0,
                 hidden_dim: int = None):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.ln2 = LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, hidden_dim, dropout=dropout)

    def forward(self, x: Tensor, mask: np.ndarray = None) -> Tensor:
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class GPT(Module):
    """A small decoder-only transformer language model.

    Args:
        vocab_size: number of distinct tokens.
        block_size: maximum context length the model can attend over.
        embed_dim: width of the residual stream.
        num_heads: attention heads per block.
        num_layers: number of transformer blocks.
        dropout: dropout probability throughout.
        tie_weights: share the token embedding and output projection matrices
            (a standard trick that saves parameters and often helps).
    """

    def __init__(self, vocab_size: int, block_size: int, embed_dim: int = 128,
                 num_heads: int = 4, num_layers: int = 4, dropout: float = 0.1,
                 tie_weights: bool = False):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.embed_dim = embed_dim

        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(block_size, embed_dim)
        self.drop = Dropout(dropout)
        self.blocks = ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = LayerNorm(embed_dim)
        self.lm_head = Linear(embed_dim, vocab_size, bias=False)
        self.tie_weights = tie_weights

        self.apply_init()

    def apply_init(self) -> None:
        """GPT-2 style initialisation: small normal weights on residual paths."""
        for name, param in self.named_parameters():
            if param.ndim >= 2 and "emb" not in name:
                init.normal_(param, 0.0, 0.02)
            elif param.ndim == 1 and "bias" in name:
                init.zeros_(param)
        # residual output projections scaled down by depth
        n_layers = len(self.blocks)
        for block in self.blocks:
            init.normal_(block.attn.out_proj.weight, 0.0, 0.02 / np.sqrt(2 * n_layers))
            init.normal_(block.ff.fc2.weight, 0.0, 0.02 / np.sqrt(2 * n_layers))

    @staticmethod
    def causal_mask(t: int) -> np.ndarray:
        """Upper-triangular boolean mask: True where attention is forbidden."""
        return np.triu(np.ones((t, t), dtype=bool), k=1)

    def forward(self, idx) -> Tensor:
        if isinstance(idx, Tensor):
            idx = idx.data
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim != 2:
            raise ValueError("GPT.forward expects idx of shape (batch, time)")
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(
                f"sequence length {t} exceeds block_size {self.block_size}"
            )

        tok = self.token_emb(idx)
        pos = self.pos_emb(np.arange(t))
        x = self.drop(tok + pos)

        mask = self.causal_mask(t)
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        if self.tie_weights:
            # logits = x @ token_emb.weight.T
            return x @ self.token_emb.weight.T
        return self.lm_head(x)

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------
    def generate(self, idx, max_new_tokens: int = 50, temperature: float = 1.0,
                 top_k: int = None, seed: int = None) -> np.ndarray:
        """Autoregressively sample tokens, appending to ``idx``.

        Args:
            idx: ``(batch, time)`` array of prompt token ids.
            max_new_tokens: how many tokens to produce.
            temperature: >1 flattens the distribution, <1 sharpens it.
            top_k: if set, sample only from the k most likely tokens.
            seed: optional RNG seed for reproducibility.

        Returns the full ``(batch, time + max_new_tokens)`` array.
        """
        rng = np.random.default_rng(seed)
        idx = np.asarray(idx, dtype=np.int64)
        if idx.ndim == 1:
            idx = idx[None, :]

        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size:]
            with no_grad():
                logits = self.forward(context).data[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                k = min(top_k, logits.shape[-1])
                cutoff = np.sort(logits, axis=-1)[:, -k][:, None]
                logits = np.where(logits < cutoff, -np.inf, logits)

            logits -= logits.max(axis=-1, keepdims=True)
            probs = np.exp(logits)
            probs /= probs.sum(axis=-1, keepdims=True)

            next_ids = np.array([
                rng.choice(probs.shape[-1], p=probs[i]) for i in range(probs.shape[0])
            ])
            idx = np.concatenate([idx, next_ids[:, None]], axis=1)

        return idx


class TransformerClassifier(Module):
    """A transformer encoder for sequence classification (mean-pooled)."""

    def __init__(self, vocab_size: int, num_classes: int, block_size: int = 128,
                 embed_dim: int = 64, num_heads: int = 4, num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.token_emb = Embedding(vocab_size, embed_dim)
        self.pos_emb = Embedding(block_size, embed_dim)
        self.drop = Dropout(dropout)
        self.blocks = ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.ln_f = LayerNorm(embed_dim)
        self.head = Linear(embed_dim, num_classes)
        self.block_size = block_size

    def forward(self, idx) -> Tensor:
        if isinstance(idx, Tensor):
            idx = idx.data
        idx = np.asarray(idx, dtype=np.int64)
        b, t = idx.shape
        if t > self.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.block_size}")
        x = self.drop(self.token_emb(idx) + self.pos_emb(np.arange(t)))
        for block in self.blocks:
            x = block(x, mask=None)          # bidirectional (no causal mask)
        x = self.ln_f(x)
        pooled = x.mean(axis=1)              # (B, C)
        return self.head(pooled)
