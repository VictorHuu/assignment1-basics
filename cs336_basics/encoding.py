import re, collections
import os
from functools import partial
from collections import Counter
from multiprocessing import Pool
from typing import Iterable, Iterator
import json

from . import pretokenization
from . import merge

class Tokenizer:
    def __init__(self, vocab= None, merges = None, special_tokens = None):
        # dict INVERSED in order to lookup the encodings of the tokens
        self.vocab = {v: k for k, v in (vocab or {}).items()}
        self.inv_vocab = {k: v for k, v in (vocab or {}).items()}
        self.merge_map = {pair: i for i, pair in enumerate(merges)}
        if special_tokens is not None:
            sorted_special = sorted(special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_special]
            self.special_pattern = "|".join(escaped_tokens)
        else:
            self.special_pattern = ""

    def load_byte_level_vocab(self,vocab_filepath: str) -> dict[int, bytes]:
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_dict = json.load(f)

        byte_level_vocab = {int(k): v.encode('utf-8') for k, v in vocab_dict.items()}
        return byte_level_vocab

    @classmethod
    def from_files(cls,
        vocab_filepath: str,merges_filepath: str,special_tokens: list[str] = None) -> "Tokenizer":
        vocab = self.load_byte_level_vocab(vocab_filepath)

        merges: list[tuple[str, str]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    def encode(self, text: str) -> list[int]:
        bs = pretokenization.process_text_to_bytes_seq(text,self.special_pattern)
        length = len(self.merge_map)
        lists = []
        for symbols in bs:
            new_symbols = symbols
            while True:
                best_pair = None
                min_rank = float('inf')
                for i in range(len(new_symbols)-1):
                    pair = (new_symbols[i], new_symbols[i+1])
                    rank = self.merge_map.get(pair, float('inf'))
                    if rank < min_rank:
                        min_rank = rank
                        best_pair = pair
                if best_pair is None: break
                new_symbols = merge.merge_pair(new_symbols, best_pair)
            for i in range(len(new_symbols)):
                lists.append(self.vocab[new_symbols[i]])
        return lists
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id
    def decode(self, ids: list[int]) -> str:
        res = b""
        for sid in ids:
            res += self.inv_vocab[sid] 
        return res.decode("utf-8",errors="replace")

    