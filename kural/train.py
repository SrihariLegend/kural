"""Train the parser on synthetic requests and write the weights the run-time loads.

  python -m kural.train [--n 400000] [--epochs 12] [--out kural/weights.npz]
"""
import argparse
import math
import multiprocessing as mp
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import feat, ir, mix
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400_000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--out", default="kural/weights.npz")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    train = to_device(build(a.n, a.seed), dev)
    held = to_device(build(4000, 9999, procs=4), dev)
    print(f"data: {a.n:,} training requests, 4,000 held out, built in {time.time() - t0:.0f}s on {dev}", flush=True)
    torch.manual_seed(a.seed)
    model = Parser().to(dev)
    print(f"model: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01, betas=(0.9, 0.98))
    steps = a.epochs * (a.n // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps, pct_start=0.04, anneal_strategy="cos")
    step = 0
    for ep in range(a.epochs):
        model.train()
        perm = torch.randperm(a.n, device=dev)
        tot_loss = n_b = 0
        for s in range(0, a.n - a.bs + 1, a.bs):
            tri, word, shape, seg, nw, out = batch(train, perm[s:s + a.bs])
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
