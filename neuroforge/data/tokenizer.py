"""Tokenisers: character-level and byte-pair encoding, both from scratch."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

__all__ = ["CharTokenizer", "BPETokenizer"]


class CharTokenizer:
    """Map each distinct character to an integer id.

    Simple, lossless and impossible to get wrong — which is exactly why it is
    the right starting point for a from-scratch language model.
    """

    def __init__(self, text: str = None, vocab: list = None):
        if vocab is not None:
            self.itos = list(vocab)
        elif text is not None:
            self.itos = sorted(set(text))
        else:
            self.itos = []
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def fit(self, text: str) -> "CharTokenizer":
        self.itos = sorted(set(text))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        return self

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> np.ndarray:
        unknown = set(text) - set(self.stoi)
        if unknown:
            raise ValueError(f"characters not in vocabulary: {sorted(unknown)}")
        return np.array([self.stoi[ch] for ch in text], dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in np.asarray(ids).reshape(-1))

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"type": "char", "itos": self.itos}, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "CharTokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(vocab=data["itos"])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"CharTokenizer(vocab_size={self.vocab_size})"


class BPETokenizer:
    """Byte-pair encoding trained from scratch.

    Repeatedly merges the most frequent adjacent pair of symbols. Merges are
    applied in order at encode time. This is the algorithm behind the
    tokenisers used by GPT-style models, just without the byte-level details.
    """

    def __init__(self, vocab_size: int = 512):
        self.target_vocab_size = vocab_size
        self.merges: list = []          # list of (int, int) pairs, in order
        self.itos: list = []            # id -> token string
        self.stoi: dict = {}            # token string -> id

    # ------------------------------------------------------------------
    def fit(self, text: str) -> "BPETokenizer":
        chars = sorted(set(text))
        self.itos = list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.merges = []

        # work on a list of symbol ids
        ids = [self.stoi[c] for c in text]

        while len(self.itos) < self.target_vocab_size:
            counts = Counter(zip(ids[:-1], ids[1:]))
            if not counts:
                break
            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break
            new_id = len(self.itos)
            merged_token = self.itos[pair[0]] + self.itos[pair[1]]
            self.merges.append(pair)
            self.itos.append(merged_token)
            self.stoi[merged_token] = new_id
            ids = self._merge(ids, pair, new_id)
        return self

    @staticmethod
    def _merge(ids: list, pair, new_id: int) -> list:
        out, i = [], 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def _encode_str(self, text: str) -> list:
        ids = [self.stoi[c] for c in text]
        for pair, new_id in zip(self.merges, range(len(self.itos) - len(self.merges), len(self.itos))):
            ids = self._merge(ids, pair, new_id)
        return ids

    def encode(self, text: str) -> np.ndarray:
        return np.array(self._encode_str(text), dtype=np.int64)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in np.asarray(ids).reshape(-1))

    def save(self, path: str) -> None:
        payload = {
            "type": "bpe",
            "itos": self.itos,
            "merges": [[int(a), int(b)] for a, b in self.merges],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=len(data["itos"]))
        tok.itos = data["itos"]
        tok.stoi = {t: i for i, t in enumerate(tok.itos)}
        tok.merges = [tuple(m) for m in data["merges"]]
        return tok

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"BPETokenizer(vocab_size={self.vocab_size}, merges={len(self.merges)})"
