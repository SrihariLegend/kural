"""Run-time parser: NumPy only, no PyTorch. Loads the weights file and turns a request into programs.

Decoding is a small beam search in which the grammar (ir.Grammar) removes every token that would make
the program malformed, so the result is always a valid program whose arguments are words the user typed
(or words of the session context). Each step only computes the newest token: the attention keys and
values of earlier tokens are kept, so the cost grows with the length of the program, not its square.
"""
import math
import os

import numpy as np

from . import feat, ir, tok

HEADS = 4
MAX_REQUEST, MAX_CONTEXT = 44, 36
_W = None


def weights(path=None):
    global _W
    if _W is None:
        path = path or os.environ.get("KURAL_WEIGHTS") or os.path.join(os.path.dirname(__file__), "weights.npz")
        with np.load(path) as z:
            _W = {k: z[k].astype(np.float32) for k in z.files}
        _W["#enc"] = sum(1 for k in _W if k.startswith("enc.") and k.endswith(".n1.weight"))
        _W["#dec"] = sum(1 for k in _W if k.startswith("dec.") and k.endswith(".n1.weight"))
    return _W


def _lin(P, name, x):
    return x @ P[name + ".weight"].T + P[name + ".bias"]


def _norm(P, name, x):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + 1e-5) * P[name + ".weight"] + P[name + ".bias"]


def _split(t):
    """[..., L, D] -> [..., H, L, D/H]"""
    *lead, L, D = t.shape
    return np.moveaxis(t.reshape(*lead, L, HEADS, D // HEADS), -2, -3)


def _softmax(a):
    a = np.exp(a - a.max(-1, keepdims=True))
    return a / a.sum(-1, keepdims=True)


def encode(P, words, n_request):
    L, D = len(words), P["pos.weight"].shape[1]
    x = np.zeros((L, D), dtype=np.float32)
    for i, w in enumerate(words):
        tris, wid, shape = feat.word_features(w)
        x[i] = (P["tri.weight"][tris].mean(0) + P["word.weight"][wid] + P["shape.weight"] @ np.asarray(shape, dtype=np.float32)
                + P["shape.bias"] + P["seg.weight"][1 if i >= n_request else 0])
    x = x + P["pos.weight"][:L]
    scale = math.sqrt(D // HEADS)
    for i in range(P["#enc"]):
        h = _norm(P, f"enc.{i}.n1", x)
        a = _softmax(_split(_lin(P, f"enc.{i}.attn.q", h)) @ _split(_lin(P, f"enc.{i}.attn.k", h)).transpose(0, 2, 1) / scale)
        out = np.moveaxis(a @ _split(_lin(P, f"enc.{i}.attn.v", h)), 0, 1).reshape(L, D)
        x = x + _lin(P, f"enc.{i}.attn.o", out)
        x = x + _lin(P, f"enc.{i}.f2", np.maximum(_lin(P, f"enc.{i}.f1", _norm(P, f"enc.{i}.n2", x)), 0))
    return _norm(P, "enc_norm", x)


class Decoder:
    """One decoding step at a time for a set of beams, reusing everything computed for earlier tokens."""

    def __init__(self, P, mem):
        self.P, self.n = P, P["#dec"]
        self.D = mem.shape[1]
        self.scale = math.sqrt(self.D // HEADS)
        self.ck = [_split(_lin(P, f"dec.{i}.cross.k", mem)).transpose(0, 2, 1) for i in range(self.n)]      # [H, d, L]
        self.cv = [_split(_lin(P, f"dec.{i}.cross.v", mem)) for i in range(self.n)]                         # [H, L, d]
        self.ptr_keys = _lin(P, "ptr_k", mem)
        self.ptr_in = _lin(P, "ptr_in", mem)

    def embed(self, tokens, t):
        P = self.P
        y = np.stack([self.ptr_in[tk - ir.NV] if tk >= ir.NV else P["out_emb.weight"][tk] for tk in tokens])
        return y + P["out_pos.weight"][t]

    def step(self, y, cache):
        """y [B, D]: the newest token of each beam. cache: per layer (keys [B,H,t,d], values [B,H,t,d]) or None.
        -> (logits over vocabulary + input words [B, NV+L], new cache)"""
        P, B, D = self.P, y.shape[0], self.D
        new = []
        for i in range(self.n):
            h = _norm(P, f"dec.{i}.n1", y)[:, None, :]                       # [B,1,D]
            k, v = _split(_lin(P, f"dec.{i}.self_attn.k", h)), _split(_lin(P, f"dec.{i}.self_attn.v", h))   # [B,H,1,d]
            if cache is not None:
                k, v = np.concatenate([cache[i][0], k], 2), np.concatenate([cache[i][1], v], 2)
            new.append((k, v))
            a = _softmax(_split(_lin(P, f"dec.{i}.self_attn.q", h)) @ k.transpose(0, 1, 3, 2) / self.scale)  # [B,H,1,t]
            y = y + _lin(P, f"dec.{i}.self_attn.o", (a @ v).transpose(0, 2, 1, 3).reshape(B, D))
            h = _norm(P, f"dec.{i}.n2", y)[:, None, :]
            a = _softmax(_split(_lin(P, f"dec.{i}.cross.q", h)) @ self.ck[i] / self.scale)                    # [B,H,1,L]
            y = y + _lin(P, f"dec.{i}.cross.o", (a @ self.cv[i]).transpose(0, 2, 1, 3).reshape(B, D))
            y = y + _lin(P, f"dec.{i}.f2", np.maximum(_lin(P, f"dec.{i}.f1", _norm(P, f"dec.{i}.n3", y)), 0))
        last = _norm(P, "dec_norm", y)
        ptr = _lin(P, "ptr_q", last) @ self.ptr_keys.T / math.sqrt(D)
        return np.concatenate([_lin(P, "vocab", last), ptr], -1), new


def parse(text, context=None, beam=4, keep=3, P=None):
    """-> list of (program, probability, request words, positions copied as arguments), best first.
    context: words describing the previous request (ir.context_words), which arguments may also point at."""
    if P is None:
        from . import native
        if native.available():
            return native.parse(text, context, beam, keep)
    request = tok.words(text)[:MAX_REQUEST]
    if not request:
        return []
    words = request + list(context or [])[:MAX_CONTEXT]
    return _search(P or weights(), words, len(request), beam, keep)


def _search(P, words, n_request, beam, keep):
    dec = Decoder(P, encode(P, words, n_request))
    beams = [([ir.BOS], 0.0, ir.Grammar(len(words)))]
    cache, finished = None, []
    for t in range(ir.MAX_OUT):
        logits, cache = dec.step(dec.embed([b[0][-1] for b in beams], t), cache)
        cand = []
        for bi, ((seq, lp, g), row) in enumerate(zip(beams, logits)):
            allowed = g.allowed()
            vals = row[allowed].astype(np.float64)
            vals -= vals.max()
            logp = vals - math.log(np.exp(vals).sum())
            for j in np.argsort(-logp)[:beam]:
                cand.append((lp + float(logp[j]), bi, allowed[int(j)]))
        cand.sort(key=lambda c: -c[0])
        nxt, idx = [], []
        for lp, bi, tk in cand[:beam]:
            seq, _, g = beams[bi]
            g2 = g.copy()
            g2.advance(tk)
            if g2.done:
                finished.append((seq[1:] + [tk], lp))
            else:
                nxt.append((seq + [tk], lp, g2))
                idx.append(bi)
        finished.sort(key=lambda f: -f[1])
        if not nxt or (len(finished) >= keep and finished[keep - 1][1] >= nxt[0][1]):
            break
        beams = nxt
        cache = [(k[idx], v[idx]) for k, v in cache]
    out, seen = [], set()
    for toks, lp in finished:
        key = tuple(toks)
        if key not in seen:
            seen.add(key)
            out.append((ir.delinearise(toks, words), math.exp(lp), words[:n_request], ir.spans_used(toks)))
    return out[:keep]
