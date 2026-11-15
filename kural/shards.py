"""Reading the training shards written by kural-gen (format: gen/src/shard.rs).

  for words, out, n_req in examples("data/shard-000000.kds"): ...    one example at a time, as Python lists
  s = Shard(path)                                                    the arrays, memory-mapped (for training)
"""
import glob
import os
import struct

import numpy as np

MAGIC = b"KURALDS1"


class Shard:
    def __init__(self, path):
        self.path = path
        m = np.memmap(path, dtype=np.uint8, mode="r")
        if bytes(m[:8]) != MAGIC:
            raise ValueError(f"{path}: not a kural shard")
        version, self.max_tri, self.n_tri, self.n_word = struct.unpack("<4I", bytes(m[8:24]))
        self.n, n_words, n_refs, n_toks, n_str = struct.unpack("<5Q", bytes(m[24:64]))
        pos = 128

        def take(dtype, count):
            nonlocal pos
            pos += (-pos) % 64
            size = np.dtype(dtype).itemsize * count
            a = np.frombuffer(m, dtype=dtype, count=count, offset=pos)
            pos += size
            return a

        self.tri = take("<u2", n_words * self.max_tri).reshape(n_words, self.max_tri)
        self.wid = take("<u2", n_words)
        self.shape = take("<u2", n_words)
        self.str_off = take("<u4", n_words + 1)
        self.strs = take("u1", n_str)
        self.n_req = take("u1", self.n)
        self.word_off = take("<u4", self.n + 1)
        self.words = take("<u4", n_refs)
        self.tok_off = take("<u4", self.n + 1)
        self.toks = take("<u2", n_toks)

    def word(self, i):
        return bytes(self.strs[self.str_off[i]:self.str_off[i + 1]]).decode("utf-8")

    def example(self, i):
        w = self.words[self.word_off[i]:self.word_off[i + 1]]
        return [self.word(int(j)) for j in w], self.toks[self.tok_off[i]:self.tok_off[i + 1]].tolist(), int(self.n_req[i])


def files(path):
    """A shard file, or every shard in a folder, in order."""
    if os.path.isdir(path):
        return sorted(glob.glob(os.path.join(path, "*.kds")))
    return [path]


def examples(path, limit=None):
    seen = 0
    for f in files(path):
        s = Shard(f)
        for i in range(s.n):
            if limit is not None and seen >= limit:
                return
            yield s.example(i)
            seen += 1
