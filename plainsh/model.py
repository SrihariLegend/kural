"""The parser network (training side, PyTorch). run.py holds the same maths in NumPy for use at run time.

Encoder reads the request's words once. Decoder writes the program one token at a time; each token is
either a word of the program language or a pointer to one of the request's words.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import feat, ir

D, HEADS, FF, ENC_LAYERS, DEC_LAYERS = 128, 4, 512, 4, 4


class Attn(nn.Module):
    def __init__(self):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(D, D) for _ in range(4))

    def forward(self, x, mem, mask):
        B, Lq, _ = x.shape
        Lk = mem.shape[1]
        split = lambda t, L: t.view(B, L, HEADS, D // HEADS).transpose(1, 2)
        att = split(self.q(x), Lq) @ split(self.k(mem), Lk).transpose(-1, -2) / math.sqrt(D // HEADS) + mask
        out = (F.softmax(att, -1) @ split(self.v(mem), Lk)).transpose(1, 2).reshape(B, Lq, D)
        return self.o(out)


class EncLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.n1, self.n2, self.attn = nn.LayerNorm(D), nn.LayerNorm(D), Attn()
        self.f1, self.f2, self.drop = nn.Linear(D, FF), nn.Linear(FF, D), nn.Dropout(p)

    def forward(self, x, mask):
        h = self.n1(x)
        x = x + self.drop(self.attn(h, h, mask))
        return x + self.drop(self.f2(F.relu(self.f1(self.n2(x)))))


class DecLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.n1, self.n2, self.n3 = nn.LayerNorm(D), nn.LayerNorm(D), nn.LayerNorm(D)
        self.self_attn, self.cross = Attn(), Attn()
        self.f1, self.f2, self.drop = nn.Linear(D, FF), nn.Linear(FF, D), nn.Dropout(p)

    def forward(self, y, mem, causal, mem_mask):
        h = self.n1(y)
        y = y + self.drop(self.self_attn(h, h, causal))
        y = y + self.drop(self.cross(self.n2(y), mem, mem_mask))
        return y + self.drop(self.f2(F.relu(self.f1(self.n3(y)))))


class Parser(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.tri = nn.Embedding(feat.N_TRI + 1, D, padding_idx=feat.N_TRI)
        self.word = nn.Embedding(feat.N_WORD, D)
        self.shape = nn.Linear(feat.N_SHAPE, D)
        self.pos = nn.Embedding(feat.MAX_WORDS, D)
        self.seg = nn.Embedding(2, D)          # 0: a word of the request, 1: a word of the session context
        self.enc = nn.ModuleList(EncLayer(dropout) for _ in range(ENC_LAYERS))
        self.enc_norm = nn.LayerNorm(D)
        self.out_emb = nn.Embedding(ir.NV, D)
        self.ptr_in = nn.Linear(D, D)
        self.out_pos = nn.Embedding(ir.MAX_OUT + 1, D)
        self.dec = nn.ModuleList(DecLayer(dropout) for _ in range(DEC_LAYERS))
        self.dec_norm = nn.LayerNorm(D)
        self.vocab = nn.Linear(D, ir.NV)
        self.ptr_q, self.ptr_k = nn.Linear(D, D), nn.Linear(D, D)
        self.drop = nn.Dropout(dropout)

    def encode(self, tri, word, shape, seg, n_words):
        """tri [B,L,T] ids (N_TRI = pad), word [B,L], shape [B,L,S], seg [B,L], n_words [B] -> memory [B,L,D], key mask [B,L]"""
        B, L, _ = tri.shape
        valid = (tri != feat.N_TRI).float().unsqueeze(-1)
        x = (self.tri(tri) * valid).sum(2) / valid.sum(2).clamp(min=1)
        x = x + self.word(word) + self.shape(shape) + self.seg(seg) + self.pos(torch.arange(L, device=tri.device))[None]
        x = self.drop(x)
        keep = torch.arange(L, device=tri.device)[None] < n_words[:, None]
        mask = torch.zeros(B, 1, 1, L, device=tri.device).masked_fill(~keep[:, None, None, :], -1e9)
        for layer in self.enc:
            x = layer(x, mask)
        return self.enc_norm(x), keep

    def decode(self, mem, keep, prev):
        """prev [B,T] previous output tokens (vocab id, or NV + word position) -> logits [B,T,NV+L]"""
        B, T = prev.shape
        L = mem.shape[1]
        is_ptr = prev >= ir.NV
        y = self.out_emb(prev.clamp(max=ir.NV - 1)) * (~is_ptr).unsqueeze(-1)
        picked = torch.gather(mem, 1, (prev - ir.NV).clamp(min=0, max=L - 1).unsqueeze(-1).expand(-1, -1, D))
        y = y + self.ptr_in(picked) * is_ptr.unsqueeze(-1)
        y = self.drop(y + self.out_pos(torch.arange(T, device=prev.device))[None])
        causal = torch.full((T, T), -1e9, device=prev.device).triu(1)[None, None]
        mem_mask = torch.zeros(B, 1, 1, L, device=prev.device).masked_fill(~keep[:, None, None, :], -1e9)
        for layer in self.dec:
            y = layer(y, mem, causal, mem_mask)
        y = self.dec_norm(y)
        ptr = self.ptr_q(y) @ self.ptr_k(mem).transpose(1, 2) / math.sqrt(D)
        ptr = ptr.masked_fill(~keep[:, None, :], -1e9)
        return torch.cat([self.vocab(y), ptr], -1)
