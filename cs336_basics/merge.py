import re, collections

from rust_max_heap import RustMaxHeap

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

def merge_pair(seq: tuple[bytes], pair: tuple[bytes]) -> tuple[bytes]:
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

def merge_vocab(pairs: dict, pair: tuple[bytes], v_in_items: list):
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