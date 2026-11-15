//! Splitting a request into words, and the hashed features of a word. Both must match kural/tok.py and
//! kural/feat.py exactly: the model was trained on what those produce.

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// `delete "old stuff" in ~/tmp, please.` -> ["delete", "old stuff", "in", "~/tmp", "please"]
///
/// Quoted text becomes one word (quotes removed). A quote only opens at the start of a word and closes at the
/// end of one, so "what's" is left alone. Trailing sentence punctuation is dropped, dots inside names are kept.
pub fn words(text: &str) -> Vec<String> {
    let cs: Vec<char> = text.chars().collect();
    let (mut out, mut pos, mut i) = (Vec::new(), 0usize, 0usize);
    while i < cs.len() {
        let q = cs[i];
        let opens = (q == '"' || q == '\'') && (i == 0 || !(is_word_char(cs[i - 1]) || cs[i - 1] == '\''));
        if opens {
            // lazy match: the first closing quote, at least one character later, not followed by a word character
            let mut j = i + 2;
            let mut found = None;
            while j < cs.len() {
                if cs[j - 1] == '\n' {
                    break;
                }
                if cs[j] == q && (j + 1 == cs.len() || !(is_word_char(cs[j + 1]) || cs[j + 1] == '\'')) {
                    found = Some(j);
                    break;
                }
                j += 1;
            }
            if i + 1 < cs.len() && cs[i + 1] == '\n' {
                found = None;
            }
            if let Some(j) = found {
                plain(&cs[pos..i].iter().collect::<String>(), &mut out);
                out.push(cs[i + 1..j].iter().collect());
                pos = j + 1;
                i = j + 1;
                continue;
            }
        }
        i += 1;
    }
    plain(&cs[pos..].iter().collect::<String>(), &mut out);
    out
}

fn plain(chunk: &str, out: &mut Vec<String>) {
    for w in chunk.split_whitespace() {
        let mut w = w.trim_end_matches([',', ';', '!', '?']);
        if w.chars().count() > 1 && w.ends_with('.') && !w.ends_with("..") {
            w = &w[..w.len() - 1];
        }
        let w = w.trim_start_matches('(').trim_end_matches(')');
        if !w.is_empty() {
            out.push(w.to_string());
        }
    }
}

pub fn fnv(s: &str) -> u32 {
    let mut h: u32 = 2166136261;
    for b in s.bytes() {
        h = (h ^ b as u32).wrapping_mul(16777619);
    }
    h
}

pub struct Features {
    pub tris: Vec<usize>,
    pub word: usize,
    pub shape: [f32; 16],
}

/// Hashed character trigrams, a hashed whole word, and 16 shape flags.
pub fn features(w: &str, n_tri: usize, n_word: usize, max_tri: usize) -> Features {
    let low = w.to_lowercase();
    let padded: Vec<char> = format!("<{low}>").chars().collect();
    let n = padded.len().saturating_sub(2).max(1);
    let tris = (0..n)
        .map(|i| {
            let t: String = padded[i..(i + 3).min(padded.len())].iter().collect();
            fnv(&t) as usize % n_tri
        })
        .take(max_tri)
        .collect();
    let cs: Vec<char> = w.chars().collect();
    let len = cs.len();
    let digits = cs.iter().filter(|c| c.is_ascii_digit()).count();
    let isupper = cs.iter().any(|c| c.is_uppercase()) && !cs.iter().any(|c| c.is_lowercase());
    let all_digits = len > 0 && digits == len;
    let flags = [
        digits > 0,
        digits == len,
        w.contains('/'),
        w.starts_with('~'),
        w.contains('.'),
        w.contains('*') || w.contains('?'),
        cs.iter().any(|c| c.is_uppercase()),
        isupper && len > 1,
        len <= 1,
        (2..=3).contains(&len),
        (4..=7).contains(&len),
        len >= 8,
        w.contains('-') || w.contains('_'),
        w.contains(' '),
        w.starts_with('.'),
        len > 0 && cs[0].is_ascii_digit() && !all_digits,
    ];
    let mut shape = [0.0f32; 16];
    for (s, f) in shape.iter_mut().zip(flags) {
        *s = if f { 1.0 } else { 0.0 };
    }
    Features { tris, word: fnv(&format!("w:{low}")) as usize % n_word, shape }
}
