//! The shard format: a self-contained file of training examples, designed so training can stream billions of
//! them without holding more than one shard in memory.
//!
//! All numbers little-endian. A 128-byte header, then arrays in this order, each starting on a 64-byte boundary:
//!
//!   header   magic "KURALDS1", u32 version (1), u32 max_tri, u32 n_tri, u32 n_word,
//!            u64 examples, u64 table words, u64 word refs, u64 output tokens, u64 string bytes, zero padding
//!   tri      u16 [table words][max_tri]   hashed character trigrams of each distinct word, padded with n_tri
//!   wid      u16 [table words]            hashed whole word
//!   shape    u16 [table words]            the 16 shape flags as bits (bit i = flag i)
//!   str_off  u32 [table words + 1]        where each word's text starts in `str`
//!   str      u8  [string bytes]           the words themselves (UTF-8), for decoding and checking
//!   n_req    u8  [examples]               how many of the example's words are the request (the rest is context)
//!   word_off u32 [examples + 1]           where each example's words start in `words`
//!   words    u32 [word refs]              indices into the word table; entry 0 of the table is padding
//!   tok_off  u32 [examples + 1]           where each example's output tokens start in `toks`
//!   toks     u16 [output tokens]          the program: vocabulary ids, or vocabulary size + word position
//!
//! Features are computed exactly as kural/feat.py does (kural_engine::tok::features), so the model sees identical
//! numbers whichever generator made the data.

use crate::req::Example;
use kural_engine::tok;
use std::collections::HashMap;
use std::io::Write;

pub const N_TRI: usize = 8192;
pub const N_WORD: usize = 4096;
pub const MAX_TRI: usize = 20;
pub const MAGIC: &[u8; 8] = b"KURALDS1";

pub struct ShardWriter {
    table: HashMap<String, u32>,
    tri: Vec<u16>,
    wid: Vec<u16>,
    shape: Vec<u16>,
    str_off: Vec<u32>,
    strs: Vec<u8>,
    n_req: Vec<u8>,
    word_off: Vec<u32>,
    words: Vec<u32>,
    tok_off: Vec<u32>,
    toks: Vec<u16>,
}

impl Default for ShardWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl ShardWriter {
    pub fn new() -> ShardWriter {
        let mut w = ShardWriter {
            table: HashMap::new(),
            tri: Vec::new(),
            wid: Vec::new(),
            shape: Vec::new(),
            str_off: vec![0],
            strs: Vec::new(),
            n_req: Vec::new(),
            word_off: vec![0],
            words: Vec::new(),
            tok_off: vec![0],
            toks: Vec::new(),
        };
        // entry 0: padding (all trigrams are the pad id, contributes nothing)
        w.tri.extend(std::iter::repeat(N_TRI as u16).take(MAX_TRI));
        w.wid.push(0);
        w.shape.push(0);
        w.str_off.push(0);
        w
    }

    fn word(&mut self, s: &str) -> u32 {
        if let Some(&i) = self.table.get(s) {
            return i;
        }
        let i = self.wid.len() as u32;
        let f = tok::features(s, N_TRI, N_WORD, MAX_TRI);
        let mut t = [N_TRI as u16; MAX_TRI];
        for (j, &x) in f.tris.iter().enumerate() {
            t[j] = x as u16;
        }
        self.tri.extend_from_slice(&t);
        self.wid.push(f.word as u16);
        let mut bits = 0u16;
        for (j, &v) in f.shape.iter().enumerate() {
            if v > 0.5 {
                bits |= 1 << j;
            }
        }
        self.shape.push(bits);
        self.strs.extend_from_slice(s.as_bytes());
        self.str_off.push(self.strs.len() as u32);
        self.table.insert(s.to_string(), i);
        i
    }

    pub fn push(&mut self, ex: &Example) {
        for w in &ex.words {
            let i = self.word(w);
            self.words.push(i);
        }
        self.word_off.push(self.words.len() as u32);
        self.n_req.push(ex.n_req as u8);
        self.toks.extend_from_slice(&ex.out);
        self.tok_off.push(self.toks.len() as u32);
    }

    pub fn len(&self) -> usize {
        self.n_req.len()
    }

    pub fn is_empty(&self) -> bool {
        self.n_req.is_empty()
    }

    pub fn write(&self, path: &std::path::Path) -> std::io::Result<()> {
        let tmp = path.with_extension("tmp");
        let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&tmp)?);
        let mut head = Vec::with_capacity(128);
        head.extend_from_slice(MAGIC);
        for v in [1u32, MAX_TRI as u32, N_TRI as u32, N_WORD as u32] {
            head.extend_from_slice(&v.to_le_bytes());
        }
        for v in [self.len() as u64, self.wid.len() as u64, self.words.len() as u64, self.toks.len() as u64, self.strs.len() as u64] {
            head.extend_from_slice(&v.to_le_bytes());
        }
        head.resize(128, 0);
        f.write_all(&head)?;
        let mut pos = 128u64;
        fn put(f: &mut impl Write, pos: &mut u64, bytes: &[u8]) -> std::io::Result<()> {
            let pad = (64 - (*pos % 64)) % 64;
            f.write_all(&vec![0u8; pad as usize])?;
            f.write_all(bytes)?;
            *pos += pad + bytes.len() as u64;
            Ok(())
        }
        fn b16(v: &[u16]) -> Vec<u8> {
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        }
        fn b32(v: &[u32]) -> Vec<u8> {
            v.iter().flat_map(|x| x.to_le_bytes()).collect()
        }
        put(&mut f, &mut pos, &b16(&self.tri))?;
        put(&mut f, &mut pos, &b16(&self.wid))?;
        put(&mut f, &mut pos, &b16(&self.shape))?;
        put(&mut f, &mut pos, &b32(&self.str_off))?;
        put(&mut f, &mut pos, &self.strs)?;
        put(&mut f, &mut pos, &self.n_req)?;
        put(&mut f, &mut pos, &b32(&self.word_off))?;
        put(&mut f, &mut pos, &b32(&self.words))?;
        put(&mut f, &mut pos, &b32(&self.tok_off))?;
        put(&mut f, &mut pos, &b16(&self.toks))?;
        f.flush()?;
        drop(f);
        std::fs::rename(tmp, path)
    }
}
