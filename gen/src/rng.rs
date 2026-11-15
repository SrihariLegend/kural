//! xoshiro256** seeded through splitmix64: fast, small, and good enough for sampling training data.

pub struct Rng {
    s: [u64; 4],
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut x = seed;
        Rng { s: [splitmix(&mut x), splitmix(&mut x), splitmix(&mut x), splitmix(&mut x)] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }

    /// uniform in [0, 1)
    #[inline]
    pub fn f(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// uniform in 0..n
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// uniform in a..=b
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        a + self.below((b - a + 1) as usize) as i64
    }

    /// index chosen with the given weights (Python's random.choices(..., weights)[0])
    pub fn weighted(&mut self, w: &[f64]) -> usize {
        let total: f64 = w.iter().sum();
        let mut x = self.f() * total;
        for (i, v) in w.iter().enumerate() {
            if x < *v {
                return i;
            }
            x -= v;
        }
        w.len() - 1
    }

    /// k distinct items (Python's random.sample)
    pub fn sample<T: Clone>(&mut self, xs: &[T], k: usize) -> Vec<T> {
        let mut idx: Vec<usize> = (0..xs.len()).collect();
        let k = k.min(xs.len());
        for i in 0..k {
            let j = i + self.below(xs.len() - i);
            idx.swap(i, j);
        }
        idx[..k].iter().map(|&i| xs[i].clone()).collect()
    }

    pub fn shuffle<T>(&mut self, xs: &mut [T]) {
        for i in (1..xs.len()).rev() {
            let j = self.below(i + 1);
            xs.swap(i, j);
        }
    }
}
