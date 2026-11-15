//! The model file (memory-mapped, no copying) and the network: encoder, and a decoder that computes one new
//! token at a time for a set of beams while keeping the attention keys and values of earlier tokens.

use crate::json::{self, Json};
use crate::kernels::{self, add_row, layer_norm, linear, softmax, Linear, W};
use crate::tok;
use std::collections::HashMap;

pub struct Config {
    pub d: usize,
    pub heads: usize,
    pub enc_layers: usize,
    pub dec_layers: usize,
    pub ff: usize,
    pub n_tri: usize,
    pub n_word: usize,
    pub max_tri: usize,
    pub max_request: usize,
    pub max_context: usize,
    pub max_pos: usize,
}

/// The bytes of the model file: memory-mapped on Unix, read into memory elsewhere.
struct Bytes {
    ptr: *const u8,
    len: usize,
    mapped: bool,
    _owned: Vec<u64>,
}

unsafe impl Send for Bytes {}
unsafe impl Sync for Bytes {}

#[cfg(unix)]
mod sys {
    extern "C" {
        pub fn mmap(addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, off: i64) -> *mut u8;
        pub fn munmap(addr: *mut u8, len: usize) -> i32;
    }
    pub const PROT_READ: i32 = 1;
    pub const MAP_PRIVATE: i32 = 2;
}

impl Bytes {
    fn open(path: &str) -> Result<Bytes, String> {
        let file = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let len = file.metadata().map_err(|e| e.to_string())?.len() as usize;
        #[cfg(unix)]
        {
            use std::os::unix::io::AsRawFd;
            let p = unsafe { sys::mmap(std::ptr::null_mut(), len, sys::PROT_READ, sys::MAP_PRIVATE, file.as_raw_fd(), 0) };
            if !p.is_null() && p as isize != -1 {
                return Ok(Bytes { ptr: p, len, mapped: true, _owned: Vec::new() });
            }
        }
        use std::io::Read;
        let mut owned = vec![0u64; len.div_ceil(8)];
        let buf = unsafe { std::slice::from_raw_parts_mut(owned.as_mut_ptr() as *mut u8, len) };
        let mut f = file;
        f.read_exact(buf).map_err(|e| e.to_string())?;
        Ok(Bytes { ptr: owned.as_ptr() as *const u8, len, mapped: false, _owned: owned })
    }
    fn slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl Drop for Bytes {
    fn drop(&mut self) {
        #[cfg(unix)]
        if self.mapped {
            unsafe { sys::munmap(self.ptr as *mut u8, self.len) };
        }
    }
}

struct EncLayer {
    n1: (W, W),
    n2: (W, W),
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    f1: Linear,
    f2: Linear,
}

struct DecLayer {
    n1: (W, W),
    n2: (W, W),
    n3: (W, W),
    sq: Linear,
    sk: Linear,
    sv: Linear,
    so: Linear,
    cq: Linear,
    ck: Linear,
    cv: Linear,
    co: Linear,
    f1: Linear,
    f2: Linear,
}

pub struct Model {
    pub cfg: Config,
    pub header: Json,
    _bytes: Bytes,
    _converted: Vec<Vec<f32>>,
    _sums: Vec<Vec<i32>>,
    tri: W,
    word: W,
    shape: Linear,
    pos: W,
    seg: W,
    enc: Vec<EncLayer>,
    enc_norm: (W, W),
    out_emb: W,
    out_pos: W,
    ptr_in: Linear,
    ptr_q: Linear,
    ptr_k: Linear,
    dec: Vec<DecLayer>,
    dec_norm: (W, W),
    vocab: Linear,
}

unsafe impl Send for Model {}
unsafe impl Sync for Model {}

impl Model {
    pub fn load(path: &str) -> Result<Model, String> {
        let bytes = Bytes::open(path)?;
        let all = bytes.slice();
        if all.len() < 16 || &all[..8] != b"KURALMF1" {
            return Err(format!("{path}: not a kural model file"));
        }
        let hlen = u64::from_le_bytes(all[8..16].try_into().unwrap()) as usize;
        let header = json::parse(std::str::from_utf8(&all[16..16 + hlen]).map_err(|e| e.to_string())?)?;
        let data_start = (16 + hlen).div_ceil(64) * 64;
        let c = header.get("config");
        let cfg = Config {
            d: c.get("d").usize(),
            heads: c.get("heads").usize(),
            enc_layers: c.get("enc_layers").usize(),
            dec_layers: c.get("dec_layers").usize(),
            ff: c.get("ff").usize(),
            n_tri: c.get("n_tri").usize(),
            n_word: c.get("n_word").usize(),
            max_tri: c.get("max_tri").usize(),
            max_request: c.get("max_request").usize(),
            max_context: c.get("max_context").usize(),
            max_pos: c.get("max_pos").usize(),
        };
        let native = kernels::native_f16();
        let mut converted: Vec<Vec<f32>> = Vec::new();
        let mut sums: Vec<Vec<i32>> = Vec::new();
        let mut tensors: HashMap<String, (W, Vec<usize>)> = HashMap::new();
        for (name, t) in header.get("tensors").obj().ok_or("model file has no tensors")? {
            let off = data_start + t.get("offset").usize();
            let shape: Vec<usize> = t.get("shape").arr().iter().map(|v| v.usize()).collect();
            let n: usize = shape.iter().product();
            let p = unsafe { bytes.ptr.add(off) };
            let size = match t.get("dtype").str() {
                "f16" => 2,
                "f32" => 4,
                _ => 1,
            };
            if off + n * size > bytes.len {
                return Err(format!("tensor {name} runs past the end of the file"));
            }
            let w = match t.get("dtype").str() {
                "f16" if native => W::F16(p as *const u16),
                "f16" => {
                    // no hardware half-floats: convert once to float32
                    let src = unsafe { std::slice::from_raw_parts(p as *const u16, n) };
                    converted.push(src.iter().map(|&h| kernels::f16_to_f32(h)).collect());
                    W::F32(converted.last().unwrap().as_ptr())
                }
                "f32" => W::F32(p as *const f32),
                "i8" => {
                    // each row's sum, for the integer kernel
                    let (rows, cols) = (shape[0], n / shape[0].max(1));
                    let src = unsafe { std::slice::from_raw_parts(p as *const i8, n) };
                    sums.push((0..rows).map(|r| src[r * cols..(r + 1) * cols].iter().map(|&v| v as i32).sum()).collect());
                    W::I8 { q: p as *const i8, scale: std::ptr::null(), rotated: t.get("rotated").bool(), sums: sums.last().unwrap().as_ptr() }
                }
                other => return Err(format!("unknown dtype {other}")),
            };
            tensors.insert(name.clone(), (w, shape));
        }
        let get = |name: &str| -> Result<W, String> {
            let (w, _) = tensors.get(name).ok_or_else(|| format!("model file is missing {name}"))?;
            Ok(match *w {
                W::I8 { q, rotated, sums, .. } => match tensors.get(&format!("{name}.scale")) {
                    Some((W::F32(s), _)) => W::I8 { q, scale: *s, rotated, sums },
                    _ => return Err(format!("{name} has no scale")),
                },
                w => w,
            })
        };
        let lin = |name: &str| -> Result<Linear, String> {
            let shape = &tensors.get(&format!("{name}.weight")).ok_or_else(|| format!("missing {name}.weight"))?.1;
            Ok(Linear { w: get(&format!("{name}.weight"))?, b: get(&format!("{name}.bias")).ok(), n_out: shape[0], n_in: shape[1] })
        };
        let norm = |name: &str| -> Result<(W, W), String> { Ok((get(&format!("{name}.weight"))?, get(&format!("{name}.bias"))?)) };
        let mut enc = Vec::new();
        for i in 0..cfg.enc_layers {
            let p = format!("enc.{i}");
            enc.push(EncLayer {
                n1: norm(&format!("{p}.n1"))?,
                n2: norm(&format!("{p}.n2"))?,
                q: lin(&format!("{p}.attn.q"))?,
                k: lin(&format!("{p}.attn.k"))?,
                v: lin(&format!("{p}.attn.v"))?,
                o: lin(&format!("{p}.attn.o"))?,
                f1: lin(&format!("{p}.f1"))?,
                f2: lin(&format!("{p}.f2"))?,
            });
        }
        let mut dec = Vec::new();
        for i in 0..cfg.dec_layers {
            let p = format!("dec.{i}");
            dec.push(DecLayer {
                n1: norm(&format!("{p}.n1"))?,
                n2: norm(&format!("{p}.n2"))?,
                n3: norm(&format!("{p}.n3"))?,
                sq: lin(&format!("{p}.self_attn.q"))?,
                sk: lin(&format!("{p}.self_attn.k"))?,
                sv: lin(&format!("{p}.self_attn.v"))?,
                so: lin(&format!("{p}.self_attn.o"))?,
                cq: lin(&format!("{p}.cross.q"))?,
                ck: lin(&format!("{p}.cross.k"))?,
                cv: lin(&format!("{p}.cross.v"))?,
                co: lin(&format!("{p}.cross.o"))?,
                f1: lin(&format!("{p}.f1"))?,
                f2: lin(&format!("{p}.f2"))?,
            });
        }
        Ok(Model {
            tri: get("tri.weight")?,
            word: get("word.weight")?,
            shape: lin("shape")?,
            pos: get("pos.weight")?,
            seg: get("seg.weight")?,
            enc,
            enc_norm: norm("enc_norm")?,
            out_emb: get("out_emb.weight")?,
            out_pos: get("out_pos.weight")?,
            ptr_in: lin("ptr_in")?,
            ptr_q: lin("ptr_q")?,
            ptr_k: lin("ptr_k")?,
            dec,
            dec_norm: norm("dec_norm")?,
            vocab: lin("vocab")?,
            cfg,
            header,
            _bytes: bytes,
            _converted: converted,
            _sums: sums,
        })
    }

    /// Multi-head attention of `q` (lq rows) over keys/values (lk rows), written into `out` (lq x d).
    fn attend(&self, q: &[f32], lq: usize, k: &[f32], v: &[f32], lk: usize, out: &mut [f32]) {
        let (d, h) = (self.cfg.d, self.cfg.heads);
        let dh = d / h;
        let scale = 1.0 / (dh as f32).sqrt();
        let mut a = vec![0f32; lk];
        for i in 0..lq {
            for hd in 0..h {
                let qi = &q[i * d + hd * dh..i * d + (hd + 1) * dh];
                for j in 0..lk {
                    a[j] = kernels::dot(qi, &k[j * d + hd * dh..j * d + (hd + 1) * dh]) * scale;
                }
                softmax(&mut a);
                let o = &mut out[i * d + hd * dh..i * d + (hd + 1) * dh];
                o.fill(0.0);
                for j in 0..lk {
                    let vj = &v[j * d + hd * dh..j * d + (hd + 1) * dh];
                    for (oo, &vv) in o.iter_mut().zip(vj) {
                        *oo += a[j] * vv;
                    }
                }
            }
        }
    }

    fn feed_forward(&self, f1: &Linear, f2: &Linear, n: (W, W), x: &mut [f32], rows: usize) {
        let d = self.cfg.d;
        let mut h = x[..rows * d].to_vec();
        layer_norm(&mut h, d, n.0, n.1);
        let mut mid = vec![0f32; rows * f1.n_out];
        linear(f1, &h, rows, &mut mid);
        for m in mid.iter_mut() {
            *m = m.max(0.0);
        }
        let mut out = vec![0f32; rows * d];
        linear(f2, &mid, rows, &mut out);
        for (a, b) in x.iter_mut().zip(&out) {
            *a += b;
        }
    }

    /// words: request words followed by context words; the first n_request are the request.
    pub fn encode(&self, words: &[String], n_request: usize) -> Vec<f32> {
        let d = self.cfg.d;
        let l = words.len().min(self.cfg.max_pos);
        let mut x = vec![0f32; l * d];
        let mut tmp = vec![0f32; d];
        for (i, w) in words.iter().take(l).enumerate() {
            let f = tok::features(w, self.cfg.n_tri, self.cfg.n_word, self.cfg.max_tri);
            let row = &mut x[i * d..(i + 1) * d];
            tmp.fill(0.0);
            for &t in &f.tris {
                add_row(self.tri, t, d, &mut tmp);
            }
            let inv = 1.0 / f.tris.len() as f32;
            for (r, t) in row.iter_mut().zip(&tmp) {
                *r = t * inv;
            }
            add_row(self.word, f.word, d, row);
            let mut sh = vec![0f32; d];
            linear(&self.shape, &f.shape, 1, &mut sh);
            for (r, s) in row.iter_mut().zip(&sh) {
                *r += s;
            }
            add_row(self.seg, if i >= n_request { 1 } else { 0 }, d, row);
            add_row(self.pos, i, d, row);
        }
        let mut q = vec![0f32; l * d];
        let mut k = vec![0f32; l * d];
        let mut v = vec![0f32; l * d];
        let mut att = vec![0f32; l * d];
        let mut o = vec![0f32; l * d];
        for layer in &self.enc {
            let mut h = x.clone();
            layer_norm(&mut h, d, layer.n1.0, layer.n1.1);
            linear(&layer.q, &h, l, &mut q);
            linear(&layer.k, &h, l, &mut k);
            linear(&layer.v, &h, l, &mut v);
            self.attend(&q, l, &k, &v, l, &mut att);
            linear(&layer.o, &att, l, &mut o);
            for (a, b) in x.iter_mut().zip(&o) {
                *a += b;
            }
            self.feed_forward(&layer.f1, &layer.f2, layer.n2, &mut x, l);
        }
        layer_norm(&mut x, d, self.enc_norm.0, self.enc_norm.1);
        x
    }

    pub fn decoder(&self, mem: Vec<f32>, n_words: usize) -> Decoder<'_> {
        let d = self.cfg.d;
        let mut ck = Vec::new();
        let mut cv = Vec::new();
        for layer in &self.dec {
            let mut a = vec![0f32; n_words * d];
            let mut b = vec![0f32; n_words * d];
            linear(&layer.ck, &mem, n_words, &mut a);
            linear(&layer.cv, &mem, n_words, &mut b);
            ck.push(a);
            cv.push(b);
        }
        let mut ptr_keys = vec![0f32; n_words * d];
        let mut ptr_in = vec![0f32; n_words * d];
        linear(&self.ptr_k, &mem, n_words, &mut ptr_keys);
        linear(&self.ptr_in, &mem, n_words, &mut ptr_in);
        Decoder { m: self, n_words, ck, cv, ptr_keys, ptr_in }
    }
}

/// Keys and values of earlier tokens, for one beam: per decoder layer, t x d each.
#[derive(Clone, Default)]
pub struct Cache {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

pub struct Decoder<'a> {
    m: &'a Model,
    pub n_words: usize,
    ck: Vec<Vec<f32>>,
    cv: Vec<Vec<f32>>,
    ptr_keys: Vec<f32>,
    ptr_in: Vec<f32>,
}

impl Decoder<'_> {
    /// One step for every beam: `tokens[b]` is beam b's newest token, at output position t.
    /// Returns logits [beams][nv + n_words] and appends to each beam's cache.
    pub fn step(&self, tokens: &[usize], t: usize, nv: usize, caches: &mut [Cache]) -> Vec<f32> {
        let m = self.m;
        let d = m.cfg.d;
        let b = tokens.len();
        let mut y = vec![0f32; b * d];
        for (i, &tk) in tokens.iter().enumerate() {
            let row = &mut y[i * d..(i + 1) * d];
            if tk >= nv {
                row.copy_from_slice(&self.ptr_in[(tk - nv) * d..(tk - nv + 1) * d]);
            } else {
                add_row(m.out_emb, tk, d, row);
            }
            add_row(m.out_pos, t, d, row);
        }
        let mut q = vec![0f32; b * d];
        let mut k = vec![0f32; b * d];
        let mut v = vec![0f32; b * d];
        let mut att = vec![0f32; b * d];
        let mut o = vec![0f32; b * d];
        for (li, layer) in m.dec.iter().enumerate() {
            let mut h = y.clone();
            layer_norm(&mut h, d, layer.n1.0, layer.n1.1);
            linear(&layer.sq, &h, b, &mut q);
            linear(&layer.sk, &h, b, &mut k);
            linear(&layer.sv, &h, b, &mut v);
            for i in 0..b {
                let c = &mut caches[i];
                if c.k.len() < m.dec.len() {
                    c.k.resize(m.dec.len(), Vec::new());
                    c.v.resize(m.dec.len(), Vec::new());
                }
                c.k[li].extend_from_slice(&k[i * d..(i + 1) * d]);
                c.v[li].extend_from_slice(&v[i * d..(i + 1) * d]);
                let lk = c.k[li].len() / d;
                m.attend(&q[i * d..(i + 1) * d], 1, &c.k[li], &c.v[li], lk, &mut att[i * d..(i + 1) * d]);
            }
            linear(&layer.so, &att, b, &mut o);
            for (a, bb) in y.iter_mut().zip(&o) {
                *a += bb;
            }
            let mut h = y.clone();
            layer_norm(&mut h, d, layer.n2.0, layer.n2.1);
            linear(&layer.cq, &h, b, &mut q);
            for i in 0..b {
                m.attend(&q[i * d..(i + 1) * d], 1, &self.ck[li], &self.cv[li], self.n_words, &mut att[i * d..(i + 1) * d]);
            }
            linear(&layer.co, &att, b, &mut o);
            for (a, bb) in y.iter_mut().zip(&o) {
                *a += bb;
            }
            m.feed_forward(&layer.f1, &layer.f2, layer.n3, &mut y, b);
        }
        layer_norm(&mut y, d, m.dec_norm.0, m.dec_norm.1);
        let width = nv + self.n_words;
        let mut logits = vec![0f32; b * width];
        let mut voc = vec![0f32; b * nv];
        linear(&m.vocab, &y, b, &mut voc);
        let mut pq = vec![0f32; b * d];
        linear(&m.ptr_q, &y, b, &mut pq);
        let s = 1.0 / (d as f32).sqrt();
        for i in 0..b {
            logits[i * width..i * width + nv].copy_from_slice(&voc[i * nv..(i + 1) * nv]);
            for j in 0..self.n_words {
                logits[i * width + nv + j] = kernels::dot(&pq[i * d..(i + 1) * d], &self.ptr_keys[j * d..(j + 1) * d]) * s;
            }
        }
        logits
    }
}
