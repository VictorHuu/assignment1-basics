import os
from typing import BinaryIO
from multiprocessing import Pool
import re,collections
import regex
from collections import Counter
import heapq
from functools import partial

from . import merge
import os
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def process_chunk_to_counts(boundary_pair,input_path,special_pattern):
    start, end = boundary_pair
    local_wc = collections.Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        parts = re.split(special_pattern,chunk)
        for part in filter(None, parts):
            for match in regex.finditer(PAT, part):
                word = match.group()
                bword = tuple(bytes([b]) for b in word.encode("utf-8"))
                local_wc[bword]+=1
    return local_wc

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
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

    # The following is a serial implementation, but you can parallelize this
    # by sending each start/end pair to a set of processes.
    boundary_pairs = list(zip(boundaries[:-1], boundaries[1:]))
    worker = partial(process_chunk_to_counts,input_path=input_path,special_pattern=special_pattern)
    with Pool(num_processes) as pool:
        wc_list = pool.map(worker,boundary_pairs)

    wc = collections.Counter()
    for c in wc_list:
        wc.update(c)

    # compute bpe merges
    merges = merge.vocab_merge(wc,vocab_size-cnt)

    res = []
    for pair in merges:
        vocab[cnt]=b''.join(pair[0])
        cnt += 1
        res.append(pair[0])
    return (vocab, res)
def main():
    internal_run_train_bpe("/root/projects/cs336/assignment1-basics/tests/fixtures/corpus.en",500,["<|endoftext|>"])
    
if __name__ == "__main__":
    main()

