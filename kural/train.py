"""Train the parser on synthetic requests and write the weights the run-time loads.

  python -m kural.train [--n 400000] [--epochs 12] [--out kural/weights.npz]          data made here, in memory
  python -m kural.train --data DIR [--held DIR2] [--n N] [--epochs E] [--out ...]     data from kural-gen shards

With --data the shards are streamed: one shard at a time is on the GPU, and each batch is gathered there from the
shard's word table, so memory does not grow with the size of the dataset. --n then caps the examples used per
epoch (default: all of them). The held-out set is --held (shards, e.g. kural-gen data --seed 9999 --n 4000),
DIR/../held if that exists, or else 4,000 requests made here.
"""
import argparse
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import feat, ir, mix, shards
from .model import Parser

L_MAX = mix.MAX_TOTAL


def featurise(pairs):
    n = len(pairs)
    tri = np.full((n, L_MAX, feat.MAX_TRI), feat.N_TRI, dtype=np.int16)
    word = np.zeros((n, L_MAX), dtype=np.int16)
    shape = np.zeros((n, L_MAX, feat.N_SHAPE), dtype=np.uint8)
    nw = np.zeros(n, dtype=np.int16)
    seg = np.zeros((n, L_MAX), dtype=np.uint8)
    out = np.zeros((n, ir.MAX_OUT), dtype=np.int16)
    cache = {}
    for i, (words, toks, n_req) in enumerate(pairs):
        nw[i] = len(words)
        seg[i, n_req:len(words)] = 1
        for j, w in enumerate(words):
            f = cache.get(w)
            if f is None:
                f = cache[w] = feat.word_features(w)
            tri[i, j, :len(f[0])] = f[0]
            word[i, j] = f[1]
            shape[i, j] = f[2]
        out[i, :len(toks)] = toks
    return tri, word, shape, seg, nw, out


def _chunk(args):
    n, seed = args
    pairs = mix.dataset(n, seed)
    return featurise(pairs)


def build(n, seed, procs=12):
    per = math.ceil(n / procs)
    with mp.Pool(procs) as pool:
        parts = pool.map(_chunk, [(per, seed * 1000 + i) for i in range(procs)])
    return [np.concatenate([p[k] for p in parts])[:n] for k in range(6)]


def to_device(arrs, dev):
    return tuple(torch.from_numpy(a).to(dev) for a in arrs)


def batch(data, idx):
    tri, word, shape, seg, nw, out = (d[idx] for d in data)
    L = int(nw.max())
    T = int((out != ir.PAD).sum(1).max())
    return tri[:, :L].long(), word[:, :L].long(), shape[:, :L].float(), seg[:, :L].long(), nw.long(), out[:, :T].long()


@torch.no_grad()
def greedy(model, tri, word, shape, seg, nw):
    mem, keep = model.encode(tri, word, shape, seg, nw)
    B = tri.shape[0]
    prev = torch.full((B, 1), ir.BOS, device=tri.device)
    done = torch.zeros(B, dtype=torch.bool, device=tri.device)
    for _ in range(ir.MAX_OUT):
        nxt = model.decode(mem, keep, prev)[:, -1].argmax(-1)
        nxt = nxt.masked_fill(done, ir.PAD)
        prev = torch.cat([prev, nxt[:, None]], 1)
        done |= nxt == ir.END
        if done.all():
            break
    return prev[:, 1:]


def exact_match(model, data, bs=1000):
    model.eval()
    hit = tot = 0
    for s in range(0, data[0].shape[0], bs):
        tri, word, shape, seg, nw, out = batch(data, slice(s, s + bs))
        pred = greedy(model, tri, word, shape, seg, nw)
        T = max(pred.shape[1], out.shape[1])
        pred = F.pad(pred, (0, T - pred.shape[1]))
        gold = F.pad(out, (0, T - out.shape[1]))
        hit += (pred == gold).all(1).sum().item()
        tot += pred.shape[0]
    return hit / tot


# --------------------------------------------------------------------------- streaming from kural-gen shards
def shard_arrays(path, limit=None):
    """Shards -> the same arrays featurise() makes (for a small set such as the held-out one)."""
    pairs = [(w, t, n) for w, t, n in shards.examples(path, limit) if len(w) <= L_MAX]
    return featurise(pairs)


class ShardStream:
    """Training batches gathered on the GPU from one shard at a time."""

    def __init__(self, path, dev, limit=None):
        self.dev, self.files, self.sizes = dev, [], []
        total = 0
        for f in shards.files(path):
            s = shards.Shard(f)
            if s.n_tri != feat.N_TRI or s.n_word != feat.N_WORD or s.max_tri != feat.MAX_TRI:
                raise SystemExit(f"{f}: made with different feature sizes than kural/feat.py")
            n = s.n if limit is None else min(s.n, limit - total)
            if n <= 0:
                break
            self.files.append(f); self.sizes.append(n)
            total += n
        self.n = total
        if not self.files:
            raise SystemExit(f"no shards in {path}")

    def steps(self, bs):
        return sum(n // bs for n in self.sizes)

    def _load(self, f, n):
        s, d = shards.Shard(f), self.dev
        t = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a).astype(dt)).to(d)
        return {"tri": t(s.tri, np.int16), "wid": t(s.wid, np.int16), "shape": t(s.shape, np.int32), "n_req": t(s.n_req[:n], np.int64),
                "woff": t(s.word_off[:n + 1], np.int64), "words": t(s.words, np.int64), "toff": t(s.tok_off[:n + 1], np.int64),
                "toks": t(s.toks, np.int64), "n": n}

    def epoch(self, bs):
        bits = torch.arange(feat.N_SHAPE, device=self.dev, dtype=torch.int32)
        for fi in torch.randperm(len(self.files)).tolist():
            g = self._load(self.files[fi], self.sizes[fi])
            perm = torch.randperm(g["n"], device=self.dev)
            for b in range(0, g["n"] - bs + 1, bs):
                idx = perm[b:b + bs]
                start, lens = g["woff"][idx], g["woff"][idx + 1] - g["woff"][idx]
                L = int(lens.max())
                pos = torch.arange(L, device=self.dev)
                valid = pos[None] < lens[:, None]
                ref = torch.where(valid, g["words"][(start[:, None] + pos[None]).clamp(max=g["words"].shape[0] - 1)], 0)
                tri = g["tri"][ref].long()
                word = g["wid"][ref].long()
                shape = ((g["shape"][ref][..., None] >> bits) & 1).float()
                seg = (valid & (pos[None] >= g["n_req"][idx][:, None])).long()
                ts, tl = g["toff"][idx], g["toff"][idx + 1] - g["toff"][idx]
                T = int(tl.max())
                tp = torch.arange(T, device=self.dev)
                out = torch.where(tp[None] < tl[:, None], g["toks"][(ts[:, None] + tp[None]).clamp(max=g["toks"].shape[0] - 1)], ir.PAD)
                yield tri, word, shape, seg, lens, out
            del g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--out", default="kural/weights.npz")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--data", help="folder of kural-gen shards to stream instead of making data here")
    ap.add_argument("--held", help="held-out shards (with --data)")
    a, given = ap.parse_args(), {x.split("=")[0] for x in __import__("sys").argv[1:]}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    if a.data:
        stream = ShardStream(a.data, dev, a.n if "--n" in given else None)
        held_path = a.held or os.path.join(os.path.dirname(os.path.abspath(a.data.rstrip("/"))), "held")
        held = to_device(shard_arrays(held_path, 4000) if os.path.exists(held_path) else build(4000, 9999, procs=4), dev)
        n_train, steps_per_epoch = stream.n, stream.steps(a.bs)
        print(f"data: {n_train:,} training requests streamed from {len(stream.files)} shard(s) in {a.data}, "
              f"{held[0].shape[0]:,} held out ({held_path if os.path.exists(held_path) else 'made here'}), on {dev}", flush=True)
    else:
        train = to_device(build(a.n, a.seed), dev)
        held = to_device(build(4000, 9999, procs=4), dev)
        n_train, steps_per_epoch = a.n, a.n // a.bs
        print(f"data: {a.n:,} training requests, 4,000 held out, built in {time.time() - t0:.0f}s on {dev}", flush=True)
    torch.manual_seed(a.seed)
    model = Parser().to(dev)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    steps = a.epochs * steps_per_epoch
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.04, anneal_strategy="cos")
    step = 0
    for ep in range(a.epochs):
        model.train()
        tot_loss = n_b = 0
        if a.data:
            batches = stream.epoch(a.bs)
        else:
            perm = torch.randperm(a.n, device=dev)
            batches = (batch(train, perm[s:s + a.bs]) for s in range(0, a.n - a.bs + 1, a.bs))
        for tri, word, shape, seg, nw, out in batches:
            prev = torch.cat([torch.full((out.shape[0], 1), ir.BOS, device=dev), out[:, :-1]], 1)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                mem, keep = model.encode(tri, word, shape, seg, nw)
                logits = model.decode(mem, keep, prev)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), out.reshape(-1), ignore_index=ir.PAD)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            tot_loss += loss.item(); n_b += 1
        em = exact_match(model, held)
        print(f"epoch {ep + 1:2d}/{a.epochs}  loss {tot_loss / n_b:.4f}  held-out exact match {100 * em:.2f}%  ({time.time() - t0:.0f}s)", flush=True)
    np.savez_compressed(a.out, **{k: v.detach().float().cpu().numpy().astype(np.float16) for k, v in model.state_dict().items()})
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
