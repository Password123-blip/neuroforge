#!/usr/bin/env python3
"""Example 5 — tokenisers: character-level vs byte-pair encoding.

Run:
    python examples/05_tokenizer_demo.py

Shows how the two tokenisers segment the same text, and why BPE produces
shorter sequences (which means more text fits in the model's context window).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neuroforge import data  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vocab-size", type=int, default=300)
    parser.add_argument("--text", type=str, default=None)
    args = parser.parse_args()

    text = args.text or data.BUILTIN_TEXT
    sample = text[:400]

    print(f"corpus length: {len(text)} characters\n")

    # ---- character tokeniser ----------------------------------------
    char_tok = data.CharTokenizer(text)
    char_ids = char_tok.encode(sample)
    print(f"CharTokenizer   vocab={char_tok.vocab_size:4d}  "
          f"sample -> {len(char_ids):4d} tokens")
    print(f"  round trip ok: {char_tok.decode(char_ids) == sample}")

    # ---- BPE tokeniser ----------------------------------------------
    bpe = data.BPETokenizer(vocab_size=args.vocab_size)
    bpe.fit(text)
    bpe_ids = bpe.encode(sample)
    print(f"\nBPETokenizer    vocab={bpe.vocab_size:4d}  "
          f"sample -> {len(bpe_ids):4d} tokens  "
          f"({len(bpe_ids) / len(char_ids):.2f}x shorter)")
    print(f"  round trip ok: {bpe.decode(bpe_ids) == sample}")

    print("\nfirst 12 BPE merges (as text):")
    for a, b in bpe.merges[:12]:
        print(f"  {bpe.itos[a]!r} + {bpe.itos[b]!r} -> {bpe.itos[a] + bpe.itos[b]!r}")

    # ---- save / load round trip -------------------------------------
    char_tok.save("tokenizer_char.json")
    bpe.save("tokenizer_bpe.json")
    reloaded = data.BPETokenizer.load("tokenizer_bpe.json")
    same = reloaded.encode(sample).tolist() == bpe_ids.tolist()
    print(f"\nsaved and reloaded BPE tokeniser; encodes identically: {same}")
    os.remove("tokenizer_char.json")
    os.remove("tokenizer_bpe.json")


if __name__ == "__main__":
    main()
