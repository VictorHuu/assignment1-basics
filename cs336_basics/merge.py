import re, collections


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for symbols, freq in vocab.items():
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq
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

def merge_vocab(pair:tuple[bytes], v_in:dict[tuple[bytes], int]):
    v_out = collections.defaultdict(int)

    for word,freq in v_in.items():
        w_out = merge_pair(word,pair)
        if w_out!=word:
            for i in range(len(word)-1):
                pairs[word[i],word[i+1]]-=freq
            for i in range(len(w_out)-1):
                pairs[w_out[i],w_out[i+1]]+=freq
        v_out[w_out] = v_in[word]
    assert len(v_in)==len(v_out)
    return v_out

def vocab_merge(vocab,num_merges) -> dict[tuple[bytes],int]:
    merges = []
    global pairs

    for i in range(num_merges):
        pairs = get_stats(vocab)
        if len(pairs)==0:
            break
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))

        vocab = merge_vocab(best[0], vocab)
        merges.append(best)

    return merges