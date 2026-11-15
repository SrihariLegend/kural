//! The word table behind kural's vocabulary check (kural/lexicon.json), built the same way kural/lexicon.py
//! build() does: which literal (non-argument) words appear in requests for each stage.

use crate::mix;
use crate::req::{lang, spans_used};
use crate::rng::Rng;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

/// Lower-case word stems worth checking: [a-z][a-z0-9+-]*, at least 3 long (lexicon.content).
pub fn content(words: &[&String]) -> Vec<String> {
    let mut out = Vec::new();
    for w in words {
        let low = w.to_lowercase();
        let b: Vec<char> = low.chars().collect();
        let mut i = 0;
        while i < b.len() {
            if b[i].is_ascii_lowercase() {
                let s = i;
                i += 1;
                while i < b.len() && (b[i].is_ascii_lowercase() || b[i].is_ascii_digit() || b[i] == '+' || b[i] == '-') {
                    i += 1;
                }
                if i - s >= 3 {
                    out.push(b[s..i].iter().collect());
                }
            } else {
                i += 1;
            }
        }
    }
    out
}

pub fn build(n: usize, seed: u64) -> (Vec<String>, BTreeMap<String, Vec<String>>, u64) {
    let lg = lang();
    let mut r = Rng::new(seed);
    let mut seen: HashMap<&'static str, HashMap<String, u64>> = HashMap::new();
    let mut total: HashMap<&'static str, u64> = HashMap::new();
    let (mut made, mut errors) = (0, 0u64);
    while made < n {
        let ex = match mix::example(&mut r) {
            Ok(ex) => ex,
            Err(_) => {
                errors += 1;
                continue;
            }
        };
        let toks: Vec<u16> = ex.out.iter().copied().filter(|&t| t != lg.end).collect();
        if toks.contains(&lg.sep) {
            continue;
        }
        made += 1;
        let used: HashSet<usize> = spans_used(&ex.out).into_iter().collect();
        let lit: Vec<&String> = ex.words[..ex.n_req].iter().enumerate().filter(|(i, _)| !used.contains(i)).map(|(_, w)| w).collect();
        let literal: BTreeSet<String> = content(&lit).into_iter().collect();
        let mut names: HashSet<&'static str> = HashSet::new();
        for &t in &toks {
            if (t as usize) < lg.nv {
                let v = lg.l.vocab[t as usize].as_str();
                if lg.stage_ix.contains_key(v) {
                    names.insert(v);
                }
            }
        }
        for name in names {
            *total.entry(name).or_default() += 1;
            let c = seen.entry(name).or_default();
            for w in &literal {
                *c.entry(w.clone()).or_default() += 1;
            }
        }
    }
    let mut stages: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (name, c) in &seen {
        let floor = (4.0f64).max(0.005 * total[name] as f64);
        let mut ws: Vec<String> = c.iter().filter(|(_, &k)| k as f64 >= floor).map(|(w, _)| w.clone()).collect();
        ws.sort();
        stages.insert(name.to_string(), ws);
    }
    let mut df: HashMap<&String, usize> = HashMap::new();
    for ws in stages.values() {
        for w in ws {
            *df.entry(w).or_default() += 1;
        }
    }
    let mut generic: Vec<String> = df.iter().filter(|(_, &k)| k as f64 >= 0.2 * stages.len() as f64).map(|(w, _)| (*w).clone()).collect();
    generic.sort();
    let gs: HashSet<&String> = generic.iter().collect();
    let stages = stages.into_iter().map(|(k, v)| (k, v.into_iter().filter(|w| !gs.contains(w)).collect())).collect();
    (generic, stages, errors)
}

pub fn to_json(generic: &[String], stages: &BTreeMap<String, Vec<String>>) -> String {
    use kural_engine::json::write_str;
    let mut s = String::from("{\"generic\":[");
    for (i, w) in generic.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        write_str(&mut s, w);
    }
    s.push_str("],\"stages\":{");
    for (i, (k, ws)) in stages.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        write_str(&mut s, k);
        s.push_str(":[");
        for (j, w) in ws.iter().enumerate() {
            if j > 0 {
                s.push(',');
            }
            write_str(&mut s, w);
        }
        s.push(']');
    }
    s.push_str("}}");
    s
}
