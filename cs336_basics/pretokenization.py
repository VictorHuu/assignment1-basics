import os
from typing import BinaryIO

import re,collections
import regex

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

PAT = regex.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

def process_chunk_to_counts(boundary_pair,input_path,special_pattern):
    start, end = boundary_pair
    local_wc = collections.Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        parts = re.split(special_pattern,chunk)
        for part in filter(None, parts):
            for match in PAT.finditer(part):
                word = match.group()
                bword = tuple(word.encode("utf-8")[i:i+1] for i in range(len(word.encode("utf-8"))))
                local_wc[bword]+=1
    return local_wc

def process_text_to_bytes_seq(text, special_pattern):
    local_wc = []
    parts = re.split(f"({special_pattern})", text) if special_pattern else [text]

    for part in filter(None, parts):
        if special_pattern and re.fullmatch(special_pattern, part):
            local_wc.append((part.encode("utf-8"),))
        else:
            for match in PAT.finditer(part):
                word = match.group()
                bword = tuple(word.encode("utf-8")[i:i+1] for i in range(len(word.encode("utf-8"))))
                local_wc.append(bword)
    return local_wc



