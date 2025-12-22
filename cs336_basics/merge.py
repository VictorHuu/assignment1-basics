import re, collections
import os
from functools import partial
from collections import Counter
from multiprocessing import Pool

from rust_max_heap import RustMaxHeap
from . import pretokenization

w_pairs = collections.defaultdict(lambda: collections.defaultdict(int))
heaps = RustMaxHeap()

def get_stats(vocab):
    global w_pairs
    global heaps
    pairs = collections.defaultdict(int)
    for symbols, freq in vocab.items():
        for i in range(len(symbols)-1):
            pair = (symbols[i], symbols[i+1])
            pairs[pair] += freq
            heaps.add(pair,freq)
            w_pairs[pair][symbols] = freq
    return pairs

def merge_pair(seq: tuple[bytes], pair: tuple[bytes,bytes]) -> tuple[bytes]:
    merged = pair[0] + pair[1]
    out = []
    i = 0

    while i < len(seq):
        if i < len(seq) - 1 and seq[i:i+2] == pair:
            out.append(merged)
            i += 2
        else:
            out.append(seq[i])
            i += 1

    return tuple(out)

def merge_vocab(pairs: dict, pair: tuple[bytes,bytes], v_in_items: list):
    global w_pairs
    global heaps
    for word, freq in v_in_items:
        if word not in w_pairs[pair]:
            continue
        w_out = merge_pair(word, pair)
        if w_out != word:
            for i in range(len(word)-1):
                p_old = (word[i], word[i+1])
                pairs[p_old] -= freq
                heaps.add(p_old,-freq)
                if pairs[p_old] <= 0:
                    del pairs[p_old]
                if word in w_pairs[p_old]:
                    del w_pairs[p_old][word]
                    if not w_pairs[p_old]:
                        del w_pairs[p_old]
            for i in range(len(w_out)-1):
                p_new = (w_out[i], w_out[i+1])
                pairs[p_new] += freq
                heaps.add(p_new,freq)
                w_pairs[p_new][w_out] = freq
    return pairs

def vocab_merge(vocab, num_merges) -> list:
    global w_pairs
    global heaps
    w_pairs.clear()
    heaps.clear()

    merges = []
    pairs = get_stats(vocab)
    for i in range(num_merges):
        if len(pairs) == 0:
            break
        best = heaps.find_max()
        best_pair = best[0]
        v_in_items = list(w_pairs[best_pair].items())
        pairs = merge_vocab(pairs, best_pair, v_in_items)
        merges.append(best)
        if best_pair in pairs:
            heaps.remove(best_pair)
            del pairs[best_pair]
    return merges

def internal_run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # vocabulary initialization
    vocab = collections.defaultdict(bytes)
    merges = collections.defaultdict(int)

    escaped_tokens = [re.escape(token) for token in special_tokens]
    special_pattern = "|".join(escaped_tokens)
    
    vocab = {i: bytes([i]) for i in range(256)}
    cnt = 256
    for token in special_tokens:
        vocab[cnt] = token.encode("utf-8")
        cnt += 1
    
    num_processes = 4
    with open(input_path, "rb") as f:
        boundaries = pretokenization.find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    boundary_pairs = list(zip(boundaries[:-1], boundaries[1:]))
    worker = partial(pretokenization.process_chunk_to_counts,input_path=input_path,special_pattern=special_pattern)
    with Pool(num_processes) as pool:
        wc_list = pool.map(worker,boundary_pairs)

    wc = collections.Counter()
    for c in wc_list:
        wc.update(c)

    # compute bpe merges
    merges = vocab_merge(wc,vocab_size-cnt)

    res = []
    for pair_data in merges:
        pair = pair_data[0]
        vocab[cnt]=pair[0]+pair[1]
        cnt += 1
        res.append(pair)
    return (vocab, res)