"""Write the model file the Rust engine loads: kural/model.kural

  python -m kural.export [--weights kural/weights.npz] [--out kural/model.kural] [--int8]

Layout (little-endian):
  8 bytes   magic "KURALMF1"
  8 bytes   length of the header, u64
  header    JSON: model sizes, the grammar as tables, and where each tensor is
  padding   to a 64-byte boundary; then the tensors, each starting on a 64-byte boundary

The grammar travels inside the file, so the engine never needs its own copy of ir.py: which stages can start a
pipeline, which may follow each data type, and every stage's argument slots.

Tensors are float16 by default. --int8 stores every linear layer as int8 with one scale per output row, after
rotating its input dimension by a Hadamard matrix (the idea Cactus uses for low-bit weights: the rotation spreads
outlier values evenly, so one scale per row loses little). Embedding tables stay float16.
"""
import argparse
import json
import os
import struct

import numpy as np

from . import feat, ir, run

MAGIC = b"KURALMF1"
ALIGN = 64


def hadamard(n):
    h = np.array([[1.0]], dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    assert h.shape[0] == n, "Hadamard rotation needs a power-of-two width"
    return h / np.sqrt(n)


def grammar():
    types = {None: "", ir.F: ir.F, ir.L: ir.L, ir.P: ir.P, ir.X: ir.X, ir.NEG: ir.NEG, ir.SUDO: ir.SUDO}
    stages = {}
    for s in ir.STAGES:
        slots = []
        for sl in s.slots:
            if sl.kind == "span":
                slots.append({"span": True, "optional": sl.optional, "name": sl.name})
            else:
                slots.append({"span": False, "optional": False, "name": sl.name, "values": [ir.TOK[v] for v in ir.ENUMS[sl.kind]]})
        stages[str(ir.TOK[s.name])] = {"dst": types[s.dst], "slots": slots}
    nxt = {"": ir._STARTS}
    for t in (ir.F, ir.L, ir.P, ir.NEG, ir.SUDO):
        nxt[t] = ir._NEXT[t]
    return {"vocab": ir.VOCAB, "pad": ir.PAD, "bos": ir.BOS, "end": ir.END, "sep": ir.SEP, "none": ir.NONE, "nv": ir.NV,
            "max_pipelines": ir.MAX_PIPELINES, "max_out": ir.MAX_OUT, "next": nxt, "stages": stages}


LINEAR_SKIP = ("shape",)          # tiny: keep float16


def export(weights, out, int8=False):
    with np.load(weights) as z:
        P = {k: z[k] for k in z.files}
    D = P["pos.weight"].shape[1]
    tensors, blobs = {}, []
    offset = 0

    def add(name, arr, dtype, extra=None):
        nonlocal offset
        data = np.ascontiguousarray(arr.astype({"f16": np.float16, "f32": np.float32, "i8": np.int8}[dtype])).tobytes()
        pad = (-offset) % ALIGN
        blobs.append(b"\0" * pad + data)
        offset += pad
        tensors[name] = {"dtype": dtype, "shape": list(arr.shape), "offset": offset, **(extra or {})}
        offset += len(data)

    for name, arr in P.items():
        base = name.rsplit(".", 1)[0]
        n_in = arr.shape[1] if arr.ndim == 2 else 0
        is_linear = (name.endswith(".weight") and arr.ndim == 2 and f"{base}.bias" in P and n_in >= 64 and n_in & (n_in - 1) == 0
                     and not base.startswith(LINEAR_SKIP))
        if int8 and is_linear:
            w = arr.astype(np.float64) @ hadamard(n_in)         # rotate the input dimension: y = (W H)(H x), H symmetric and orthogonal
            scale = np.abs(w).max(1) / 127.0
            scale[scale == 0] = 1.0
            q = np.clip(np.round(w / scale[:, None]), -127, 127)
            add(name, q, "i8", {"rotated": True})
            add(name + ".scale", scale.astype(np.float32), "f32")
        else:
            add(name, arr, "f16")
    cfg = {"d": D, "heads": run.HEADS, "enc_layers": sum(1 for k in P if k.startswith("enc.") and k.endswith(".n1.weight")),
           "dec_layers": sum(1 for k in P if k.startswith("dec.") and k.endswith(".n1.weight")),
           "ff": P["enc.0.f1.weight"].shape[0], "n_tri": feat.N_TRI, "n_word": feat.N_WORD, "max_tri": feat.MAX_TRI,
           "n_shape": feat.N_SHAPE, "max_request": run.MAX_REQUEST, "max_context": run.MAX_CONTEXT, "max_pos": P["pos.weight"].shape[0],
           "int8": int8}
    header = json.dumps({"config": cfg, "grammar": grammar(), "tensors": tensors}, separators=(",", ":")).encode()
    head = MAGIC + struct.pack("<Q", len(header)) + header
    head += b"\0" * ((-len(head)) % ALIGN)
    with open(out, "wb") as f:
        f.write(head)
        for b in blobs:
            f.write(b)
    return os.path.getsize(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "weights.npz"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "model.kural"))
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()
    print(f"wrote {a.out}: {export(a.weights, a.out, a.int8) / 1e6:.2f} MB")
