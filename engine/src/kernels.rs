//! The arithmetic. Every weight matrix is used as `y = W x + b` with W stored row-major as [out][in], so each
//! output is one dot product over a contiguous row. Rows of x are processed four at a time so a weight row is
//! read from memory once per four inputs.
//!
//! Weights can be float16 (converted on the fly with F16C), float32, or int8 with one scale per output row whose
//! input dimension was rotated by a Hadamard matrix at export time (x is rotated the same way here, in O(n log n)).

#[derive(Clone, Copy)]
pub enum W {
    F16(*const u16),
    F32(*const f32),
    /// int8 rows, one float scale per row, and each row's sum (for the unsigned-activation correction in VNNI)
    I8 { q: *const i8, scale: *const f32, rotated: bool, sums: *const i32 },
}

#[derive(Clone, Copy)]
pub struct Linear {
    pub w: W,
    pub b: Option<W>,
    pub n_out: usize,
    pub n_in: usize,
}

unsafe impl Send for Linear {}
unsafe impl Sync for Linear {}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: normalise
            let mut e = 0i32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e += 1;
            }
            sign | (((127 - 15 - e + 1) as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 31 {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// Value i of a vector stored as W.
#[inline]
pub fn at(w: W, i: usize) -> f32 {
    unsafe {
        match w {
            W::F16(p) => f16_to_f32(*p.add(i)),
            W::F32(p) => *p.add(i),
            W::I8 { .. } => unreachable!("int8 is only used for matrices"),
        }
    }
}

/// out += row r of an embedding table [rows][d]
pub fn add_row(table: W, r: usize, d: usize, out: &mut [f32]) {
    unsafe {
        match table {
            W::F16(p) => {
                let row = std::slice::from_raw_parts(p.add(r * d), d);
                for (o, &h) in out.iter_mut().zip(row) {
                    *o += f16_to_f32(h);
                }
            }
            W::F32(p) => {
                let row = std::slice::from_raw_parts(p.add(r * d), d);
                for (o, &v) in out.iter_mut().zip(row) {
                    *o += v;
                }
            }
            W::I8 { .. } => unreachable!(),
        }
    }
}

/// In-place fast Walsh-Hadamard transform, scaled to be orthonormal (len must be a power of two).
pub fn hadamard(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let (a, b) = (x[j], x[j + h]);
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
        h *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() {
        *v *= s;
    }
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c"))
}

#[cfg(not(target_arch = "x86_64"))]
fn has_avx2() -> bool {
    false
}

#[cfg(target_arch = "x86_64")]
fn has_vnni() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| has_avx2() && is_x86_feature_detected!("avxvnni") && std::env::var_os("KURAL_NO_VNNI").is_none())
}

/// Whether float16 weights can be used as they are (otherwise the model converts them to float32 at load).
pub fn native_f16() -> bool {
    has_avx2()
}

/// y[r][n] = b[n] + sum_k x[r][k] * W[n][k]     x: rows x n_in, y: rows x n_out
pub fn linear(l: &Linear, x: &[f32], rows: usize, y: &mut [f32]) {
    debug_assert!(x.len() >= rows * l.n_in && y.len() >= rows * l.n_out);
    let mut rotated_x;
    let x = match l.w {
        W::I8 { rotated: true, .. } => {
            rotated_x = x[..rows * l.n_in].to_vec();
            for r in 0..rows {
                hadamard(&mut rotated_x[r * l.n_in..(r + 1) * l.n_in]);
            }
            &rotated_x[..]
        }
        _ => x,
    };
    #[cfg(target_arch = "x86_64")]
    if let W::I8 { sums, .. } = l.w {
        if has_vnni() && l.n_in % 32 == 0 && !sums.is_null() {
            unsafe { vnni::linear(l, x, rows, y) };
            add_bias(l, rows, y);
            return;
        }
    }
    #[cfg(target_arch = "x86_64")]
    if has_avx2() && l.n_in % 8 == 0 {
        unsafe { avx2::linear(l, x, rows, y) };
        add_bias(l, rows, y);
        return;
    }
    portable(l, x, rows, y);
    add_bias(l, rows, y);
}

fn add_bias(l: &Linear, rows: usize, y: &mut [f32]) {
    if let Some(b) = l.b {
        for r in 0..rows {
            for n in 0..l.n_out {
                y[r * l.n_out + n] += at(b, n);
            }
        }
    }
}

fn portable(l: &Linear, x: &[f32], rows: usize, y: &mut [f32]) {
    let k = l.n_in;
    let mut wrow = vec![0f32; k];
    for n in 0..l.n_out {
        let mut scale = 1.0;
        unsafe {
            match l.w {
                W::F16(p) => {
                    for (i, w) in wrow.iter_mut().enumerate() {
                        *w = f16_to_f32(*p.add(n * k + i));
                    }
                }
                W::F32(p) => wrow.copy_from_slice(std::slice::from_raw_parts(p.add(n * k), k)),
                W::I8 { q, scale: s, .. } => {
                    for (i, w) in wrow.iter_mut().enumerate() {
                        *w = *q.add(n * k + i) as f32;
                    }
                    scale = *s.add(n);
                }
            }
        }
        for r in 0..rows {
            y[r * l.n_out + n] = dot(&wrow, &x[r * k..(r + 1) * k]) * scale;
        }
    }
}

/// Dot product written so the compiler vectorises it on any target (eight independent sums).
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let chunks = a.len() / 8;
    for c in 0..chunks {
        for j in 0..8 {
            acc[j] += a[c * 8 + j] * b[c * 8 + j];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for i in chunks * 8..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// Integer path (W8A8): each input row is quantised to int8 with its own scale, shifted by +128 to unsigned
/// bytes, and multiplied with the int8 weights by VPDPBUSD (32 multiply-adds per instruction). The shift is
/// undone with the weight row sums: dot(x + 128, w) - 128 * sum(w) = dot(x, w).
#[cfg(target_arch = "x86_64")]
mod vnni {
    use super::{Linear, W};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,avxvnni")]
    unsafe fn hsum_i32(v: __m256i) -> i32 {
        let s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b01_00_11_10));
        let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b10_11_00_01));
        _mm_cvtsi128_si32(s)
    }

    #[target_feature(enable = "avx2,avxvnni")]
    pub unsafe fn linear(l: &Linear, x: &[f32], rows: usize, y: &mut [f32]) {
        let (q, scale, sums) = match l.w {
            W::I8 { q, scale, sums, .. } => (q, scale, sums),
            _ => unreachable!(),
        };
        let k = l.n_in;
        let mut xu = vec![0u8; rows * k];
        let mut sx = vec![0f32; rows];
        for r in 0..rows {
            let row = &x[r * k..(r + 1) * k];
            let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            sx[r] = s;
            let inv = 1.0 / s;
            for (o, v) in xu[r * k..(r + 1) * k].iter_mut().zip(row) {
                *o = ((v * inv).round().clamp(-127.0, 127.0) as i32 + 128) as u8;
            }
        }
        let xp = xu.as_ptr();
        let mut r = 0;
        while r < rows {
            let take = (rows - r).min(4);
            for n in 0..l.n_out {
                let wrow = q.add(n * k);
                let mut acc = [_mm256_setzero_si256(); 4];
                let mut i = 0;
                while i < k {
                    let wv = _mm256_loadu_si256(wrow.add(i) as *const __m256i);
                    for j in 0..take {
                        let xv = _mm256_loadu_si256(xp.add((r + j) * k + i) as *const __m256i);
                        acc[j] = _mm256_dpbusd_avx_epi32(acc[j], xv, wv);
                    }
                    i += 32;
                }
                let corr = 128 * *sums.add(n);
                let ws = *scale.add(n);
                for j in 0..take {
                    *y.get_unchecked_mut((r + j) * l.n_out + n) = (hsum_i32(acc[j]) - corr) as f32 * sx[r + j] * ws;
                }
            }
            r += take;
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{Linear, W};
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    #[inline]
    #[target_feature(enable = "avx2,fma,f16c")]
    unsafe fn load_w(w: W, idx: usize) -> __m256 {
        match w {
            W::F16(p) => _mm256_cvtph_ps(_mm_loadu_si128(p.add(idx) as *const __m128i)),
            W::F32(p) => _mm256_loadu_ps(p.add(idx)),
            W::I8 { q, .. } => _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(q.add(idx) as *const __m128i))),
        }
    }

    #[target_feature(enable = "avx2,fma,f16c")]
    pub unsafe fn linear(l: &Linear, x: &[f32], rows: usize, y: &mut [f32]) {
        let k = l.n_in;
        let xp = x.as_ptr();
        let mut r = 0;
        while r < rows {
            let take = (rows - r).min(4);
            for n in 0..l.n_out {
                let scale = match l.w {
                    W::I8 { scale, .. } => *scale.add(n),
                    _ => 1.0,
                };
                let mut acc = [_mm256_setzero_ps(); 4];
                let mut i = 0;
                while i < k {
                    let wv = load_w(l.w, n * k + i);
                    for j in 0..take {
                        acc[j] = _mm256_fmadd_ps(wv, _mm256_loadu_ps(xp.add((r + j) * k + i)), acc[j]);
                    }
                    i += 8;
                }
                for j in 0..take {
                    *y.get_unchecked_mut((r + j) * l.n_out + n) = hsum(acc[j]) * scale;
                }
            }
            r += take;
        }
    }
}

pub fn layer_norm(x: &mut [f32], d: usize, g: W, b: W) {
    for row in x.chunks_mut(d) {
        let m = row.iter().sum::<f32>() / d as f32;
        let v = row.iter().map(|a| (a - m) * (a - m)).sum::<f32>() / d as f32;
        let inv = 1.0 / (v + 1e-5).sqrt();
        for (i, a) in row.iter_mut().enumerate() {
            *a = (*a - m) * inv * at(g, i) + at(b, i);
        }
    }
}

pub fn softmax(a: &mut [f32]) {
    let m = a.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0.0;
    for v in a.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    for v in a.iter_mut() {
        *v /= s;
    }
}
