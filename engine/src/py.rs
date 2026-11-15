//! Small pieces of Python behaviour the run time relies on, reproduced exactly: shlex.quote, regular-expression
//! word boundaries, str.split/strip, round-half-even, os.path helpers, and difflib.SequenceMatcher.ratio.

use std::collections::HashMap;

/// Python's \w for a str pattern: letters, digits (any script) and underscore.
pub fn is_word(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Regex \b at char position i of cs.
pub fn boundary(cs: &[char], i: usize) -> bool {
    let before = i > 0 && is_word(cs[i - 1]);
    let after = i < cs.len() && is_word(cs[i]);
    before != after
}

/// shlex.quote
pub fn quote(s: &str) -> String {
    if s.is_empty() {
        return "''".into();
    }
    let safe = s.chars().all(|c| c.is_ascii_alphanumeric() || "_@%+=:,./-".contains(c));
    if safe {
        return s.to_string();
    }
    format!("'{}'", s.replace('\'', "'\"'\"'"))
}

/// Python str.isalnum (non-empty, every char alphanumeric).
pub fn isalnum(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_alphanumeric())
}

pub fn isdigit_str(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

/// Python round(): half to even.
pub fn round_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// Python float formatting "{:.N}" (exact value, ties to even like Python).
pub fn fmt_fixed(x: f64, n: usize) -> String {
    format!("{x:.n$}")
}

/// Every occurrence of a whole-word `word` (ASCII letters) at word boundaries in `t`.
pub fn has_word(t: &str, word: &str) -> bool {
    find_word(&t.chars().collect::<Vec<_>>(), word, 0).is_some()
}

/// First index >= from where `word` occurs in cs with \b before and after.
pub fn find_word(cs: &[char], word: &str, from: usize) -> Option<usize> {
    let w: Vec<char> = word.chars().collect();
    if w.is_empty() || cs.len() < w.len() {
        return None;
    }
    let mut i = from;
    while i + w.len() <= cs.len() {
        if cs[i..i + w.len()] == w[..] && boundary(cs, i) && boundary(cs, i + w.len()) {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// re.sub(r"\b(?:w1|w2|...)\b", rep, t) for literal alternatives tried in order at each position.
/// `pre_b`/`post_b`: whether each alternative needs \b at its start/end.
pub fn sub_alts(t: &str, alts: &[(&str, bool, bool)], rep: &str) -> String {
    let cs: Vec<char> = t.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    'outer: while i <= cs.len() {
        for &(alt, pb, qb) in alts {
            let a: Vec<char> = alt.chars().collect();
            if i + a.len() <= cs.len() && cs[i..i + a.len()] == a[..] && (!pb || boundary(&cs, i)) && (!qb || boundary(&cs, i + a.len())) {
                out.push_str(rep);
                i += a.len();
                continue 'outer;
            }
        }
        if i < cs.len() {
            out.push(cs[i]);
        }
        i += 1;
    }
    out
}

/// Python str.strip(chars)
pub fn strip_chars<'a>(s: &'a str, chars: &str) -> &'a str {
    s.trim_matches(|c| chars.contains(c))
}

// ------------------------------------------------------------------ os.path

pub fn expanduser(p: &str) -> String {
    if p == "~" || p.starts_with("~/") {
        if let Ok(h) = std::env::var("HOME") {
            let h = h.trim_end_matches('/');
            let h = if h.is_empty() { "/" } else { h };
            return if p == "~" { h.to_string() } else { format!("{}{}", h, &p[1..]) };
        }
    }
    p.to_string()
}

pub fn join(a: &str, b: &str) -> String {
    if b.starts_with('/') {
        return b.to_string();
    }
    if a.is_empty() || a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

/// os.path.normpath (POSIX)
pub fn normpath(p: &str) -> String {
    if p.is_empty() {
        return ".".into();
    }
    let initial = if p.starts_with("//") && !p.starts_with("///") { 2 } else if p.starts_with('/') { 1 } else { 0 };
    let mut comps: Vec<&str> = Vec::new();
    for c in p.split('/') {
        if c.is_empty() || c == "." {
            continue;
        }
        if c != ".." || (initial == 0 && comps.is_empty()) || comps.last() == Some(&"..") {
            comps.push(c);
        } else if !comps.is_empty() {
            comps.pop();
        }
    }
    let s = format!("{}{}", "/".repeat(initial), comps.join("/"));
    if s.is_empty() { ".".into() } else { s }
}

/// os.path.split
pub fn split(p: &str) -> (String, String) {
    let i = p.rfind('/').map(|i| i + 1).unwrap_or(0);
    let (head, tail) = (&p[..i], &p[i..]);
    let h = if !head.is_empty() && head.chars().any(|c| c != '/') { head.trim_end_matches('/') } else { head };
    (h.to_string(), tail.to_string())
}

/// os.path.splitext(p)[0]
pub fn splitext_root(p: &str) -> String {
    let sep = p.rfind('/').map(|i| i as isize).unwrap_or(-1);
    if let Some(dot) = p.rfind('.') {
        if dot as isize > sep {
            let start = (sep + 1) as usize;
            if p[start..dot].chars().any(|c| c != '.') {
                return p[..dot].to_string();
            }
        }
    }
    p.to_string()
}

/// shutil.which
pub fn which(cmd: &str) -> bool {
    use std::os::unix::fs::PermissionsExt;
    let ok = |p: &std::path::Path| p.is_file() && std::fs::metadata(p).map(|m| m.permissions().mode() & 0o111 != 0).unwrap_or(false);
    if cmd.contains('/') {
        return ok(std::path::Path::new(cmd));
    }
    let path = std::env::var("PATH").unwrap_or_default();
    path.split(':').filter(|d| !d.is_empty()).any(|d| ok(&std::path::Path::new(d).join(cmd)))
}

// ------------------------------------------------------------------ difflib

/// difflib.SequenceMatcher(None, a, b).ratio(), including its "autojunk" rule for b of 200+ items.
pub fn ratio(a: &str, b: &str) -> f64 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (la, lb) = (a.len(), b.len());
    if la + lb == 0 {
        return 1.0;
    }
    let mut b2j: HashMap<char, Vec<usize>> = HashMap::new();
    for (j, &c) in b.iter().enumerate() {
        b2j.entry(c).or_default().push(j);
    }
    if lb >= 200 {
        let ntest = lb / 100 + 1;
        b2j.retain(|_, v| v.len() <= ntest);
    }
    let longest = |alo: usize, ahi: usize, blo: usize, bhi: usize| -> (usize, usize, usize) {
        let (mut besti, mut bestj, mut bestsize) = (alo, blo, 0usize);
        let mut j2len: HashMap<usize, usize> = HashMap::new();
        for i in alo..ahi {
            let mut newj2len: HashMap<usize, usize> = HashMap::new();
            if let Some(js) = b2j.get(&a[i]) {
                for &j in js {
                    if j < blo {
                        continue;
                    }
                    if j >= bhi {
                        break;
                    }
                    let k = if j > 0 { j2len.get(&(j - 1)).copied().unwrap_or(0) } else { 0 } + 1;
                    newj2len.insert(j, k);
                    if k > bestsize {
                        besti = i + 1 - k;
                        bestj = j + 1 - k;
                        bestsize = k;
                    }
                }
            }
            j2len = newj2len;
        }
        while besti > alo && bestj > blo && a[besti - 1] == b[bestj - 1] {
            besti -= 1;
            bestj -= 1;
            bestsize += 1;
        }
        while besti + bestsize < ahi && bestj + bestsize < bhi && a[besti + bestsize] == b[bestj + bestsize] {
            bestsize += 1;
        }
        (besti, bestj, bestsize)
    };
    let mut queue = vec![(0, la, 0, lb)];
    let mut matches = 0;
    while let Some((alo, ahi, blo, bhi)) = queue.pop() {
        let (i, j, k) = longest(alo, ahi, blo, bhi);
        if k > 0 {
            matches += k;
            if alo < i && blo < j {
                queue.push((alo, i, blo, j));
            }
            if i + k < ahi && j + k < bhi {
                queue.push((i + k, ahi, j + k, bhi));
            }
        }
    }
    2.0 * matches as f64 / (la + lb) as f64
}

/// ratio(a, b) >= cutoff, with difflib's cheap upper bounds tried first (they never change the answer).
pub fn ratio_at_least(a: &str, b: &str, cutoff: f64) -> bool {
    let (la, lb) = (a.chars().count(), b.chars().count());
    if la + lb == 0 {
        return 1.0 >= cutoff;
    }
    if (2.0 * la.min(lb) as f64 / (la + lb) as f64) < cutoff {
        return false;
    }
    let mut avail: HashMap<char, isize> = HashMap::new();
    for c in b.chars() {
        *avail.entry(c).or_default() += 1;
    }
    let mut m = 0;
    for c in a.chars() {
        let e = avail.entry(c).or_default();
        if *e > 0 {
            m += 1;
        }
        *e -= 1;
    }
    if (2.0 * m as f64 / (la + lb) as f64) < cutoff {
        return false;
    }
    ratio(a, b) >= cutoff
}

/// difflib.get_close_matches(word, possibilities, n, cutoff) without the limit n: every qualifying one, best first.
pub fn close_matches<'a>(word: &str, possibilities: impl IntoIterator<Item = &'a str>, cutoff: f64) -> Vec<(f64, &'a str)> {
    let mut out: Vec<(f64, &str)> = possibilities.into_iter().filter(|x| ratio_at_least(x, word, cutoff)).map(|x| (ratio(x, word), x)).collect();
    out.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    out
}
