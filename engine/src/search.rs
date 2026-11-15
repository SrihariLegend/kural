//! The program grammar (read from the model file, so it always matches what the model was trained on) and the
//! beam search that only ever extends a program in ways the grammar allows. A step-for-step port of
//! kural/ir.py (Grammar) and kural/run.py (_search).

use crate::json::{self, Json};
use crate::model::{Cache, Model};
use crate::tok;
use std::collections::HashMap;

#[derive(Clone)]
pub struct Slot {
    pub name: String,
    pub span: bool,
    pub optional: bool,
    pub values: Vec<usize>,
}

pub struct StageInfo {
    pub dst: String,
    pub slots: Vec<Slot>,
}

pub struct Lang {
    pub vocab: Vec<String>,
    pub pad: usize,
    pub bos: usize,
    pub end: usize,
    pub sep: usize,
    pub none: usize,
    pub nv: usize,
    pub max_pipelines: usize,
    pub max_out: usize,
    pub next: HashMap<String, Vec<usize>>,
    pub stages: HashMap<usize, StageInfo>,
}

impl Lang {
    pub fn from_header(h: &Json) -> Lang {
        let g = h.get("grammar");
        let mut next = HashMap::new();
        for (k, v) in g.get("next").obj().into_iter().flatten() {
            next.insert(k.clone(), v.arr().iter().map(|x| x.usize()).collect());
        }
        let mut stages = HashMap::new();
        for (k, v) in g.get("stages").obj().into_iter().flatten() {
            let slots = v
                .get("slots")
                .arr()
                .iter()
                .map(|s| Slot {
                    name: s.get("name").str().to_string(),
                    span: s.get("span").bool(),
                    optional: s.get("optional").bool(),
                    values: s.get("values").arr().iter().map(|x| x.usize()).collect(),
                })
                .collect();
            stages.insert(k.parse().unwrap_or(0), StageInfo { dst: v.get("dst").str().to_string(), slots });
        }
        Lang {
            vocab: g.get("vocab").arr().iter().map(|x| x.str().to_string()).collect(),
            pad: g.get("pad").usize(),
            bos: g.get("bos").usize(),
            end: g.get("end").usize(),
            sep: g.get("sep").usize(),
            none: g.get("none").usize(),
            nv: g.get("nv").usize(),
            max_pipelines: g.get("max_pipelines").usize(),
            max_out: g.get("max_out").usize(),
            next,
            stages,
        }
    }
}

#[derive(Clone)]
struct Grammar {
    n: usize,
    ty: String,
    pending: Vec<Slot>,
    span_start: Option<usize>,
    pipelines: usize,
    done: bool,
}

impl Grammar {
    fn new(n: usize) -> Grammar {
        Grammar { n, ty: String::new(), pending: Vec::new(), span_start: None, pipelines: 1, done: false }
    }

    fn allowed(&self, l: &Lang) -> Vec<usize> {
        if self.done {
            return vec![l.pad];
        }
        if let Some(slot) = self.pending.first() {
            if slot.span {
                return match self.span_start {
                    None => {
                        let mut out: Vec<usize> = (0..self.n).map(|i| l.nv + i).collect();
                        if slot.optional {
                            out.push(l.none);
                        }
                        out
                    }
                    Some(s) => (s..self.n).map(|i| l.nv + i).collect(),
                };
            }
            return slot.values.clone();
        }
        if self.ty == "done" {
            return if self.pipelines < l.max_pipelines { vec![l.end, l.sep] } else { vec![l.end] };
        }
        l.next.get(&self.ty).cloned().unwrap_or_default()
    }

    fn advance(&mut self, l: &Lang, tok: usize) {
        if let Some(slot) = self.pending.first() {
            if slot.span && tok >= l.nv && self.span_start.is_none() {
                self.span_start = Some(tok - l.nv);
                return;
            }
            self.span_start = None;
            self.pending.remove(0);
            return;
        }
        if tok == l.end {
            self.done = true;
        } else if tok == l.sep {
            self.ty.clear();
            self.pipelines += 1;
        } else if let Some(st) = l.stages.get(&tok) {
            self.ty = st.dst.clone();
            self.pending = st.slots.clone();
        }
    }
}

pub struct Parse {
    pub tokens: Vec<usize>,
    pub prob: f64,
}

pub struct Engine {
    pub model: Model,
    pub lang: Lang,
}

impl Engine {
    pub fn load(path: &str) -> Result<Engine, String> {
        let model = Model::load(path)?;
        let lang = Lang::from_header(&model.header);
        Ok(Engine { model, lang })
    }

    /// Request words (from the text) followed by the context words, and how many are request words.
    pub fn words(&self, text: &str, context: &[String]) -> (Vec<String>, usize) {
        let mut w = tok::words(text);
        w.truncate(self.model.cfg.max_request);
        let n = w.len();
        w.extend(context.iter().take(self.model.cfg.max_context).cloned());
        (w, n)
    }

    /// The best `keep` programs for the request, best first.
    pub fn parse(&self, words: &[String], n_request: usize, beam: usize, keep: usize) -> Vec<Parse> {
        if n_request == 0 {
            return Vec::new();
        }
        let l = &self.lang;
        let mem = self.model.encode(words, n_request);
        let dec = self.model.decoder(mem, words.len());
        struct Beam {
            seq: Vec<usize>,
            lp: f64,
            g: Grammar,
            cache: Cache,
        }
        let mut beams = vec![Beam { seq: vec![l.bos], lp: 0.0, g: Grammar::new(words.len()), cache: Cache::default() }];
        let mut finished: Vec<(Vec<usize>, f64)> = Vec::new();
        let width = l.nv + words.len();
        for t in 0..l.max_out {
            let last: Vec<usize> = beams.iter().map(|b| *b.seq.last().unwrap()).collect();
            let mut caches: Vec<Cache> = beams.iter_mut().map(|b| std::mem::take(&mut b.cache)).collect();
            let logits = dec.step(&last, t, l.nv, &mut caches);
            for (b, c) in beams.iter_mut().zip(caches) {
                b.cache = c;
            }
            let mut cand: Vec<(f64, usize, usize)> = Vec::new();
            for (bi, b) in beams.iter().enumerate() {
                let row = &logits[bi * width..(bi + 1) * width];
                let allowed = b.g.allowed(l);
                let vals: Vec<f64> = allowed.iter().map(|&a| row[a] as f64).collect();
                let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let lse = vals.iter().map(|v| (v - mx).exp()).sum::<f64>().ln();
                let logp: Vec<f64> = vals.iter().map(|v| v - mx - lse).collect();
                let mut order: Vec<usize> = (0..logp.len()).collect();
                order.sort_by(|&a, &c| logp[c].partial_cmp(&logp[a]).unwrap_or(std::cmp::Ordering::Equal));
                for &j in order.iter().take(beam) {
                    cand.push((b.lp + logp[j], bi, allowed[j]));
                }
            }
            cand.sort_by(|a, c| c.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let mut nxt: Vec<Beam> = Vec::new();
            for &(lp, bi, tk) in cand.iter().take(beam) {
                let src = &beams[bi];
                let mut g = src.g.clone();
                g.advance(l, tk);
                let mut seq = src.seq.clone();
                seq.push(tk);
                if g.done {
                    finished.push((seq[1..].to_vec(), lp));
                } else {
                    nxt.push(Beam { seq, lp, g, cache: src.cache.clone() });
                }
            }
            finished.sort_by(|a, c| c.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            if nxt.is_empty() || (finished.len() >= keep && finished[keep - 1].1 >= nxt[0].lp) {
                break;
            }
            beams = nxt;
        }
        let mut out: Vec<Parse> = Vec::new();
        for (toks, lp) in finished {
            if !out.iter().any(|p| p.tokens == toks) {
                out.push(Parse { tokens: toks, prob: lp.exp() });
            }
        }
        out.truncate(keep);
        out
    }

    /// The program as JSON in the same shape kural/ir.py's delinearise gives:
    /// [[["stage", {"slot": "copied words" | "enum value" | null}], ...], ...]   and the copied word positions.
    pub fn program_json(&self, toks: &[usize], words: &[String]) -> (String, Vec<usize>) {
        let l = &self.lang;
        let mut s = String::from("[[");
        let (mut i, mut first_stage) = (0, true);
        let mut used = Vec::new();
        while i < toks.len() {
            let t = toks[i];
            i += 1;
            if t == l.end || t == l.pad {
                break;
            }
            if t == l.sep {
                s.push_str("],[");
                first_stage = true;
                continue;
            }
            if !first_stage {
                s.push(',');
            }
            first_stage = false;
            s.push('[');
            json::write_str(&mut s, &l.vocab[t]);
            s.push_str(",{");
            if let Some(st) = l.stages.get(&t) {
                for (si, slot) in st.slots.iter().enumerate() {
                    if si > 0 {
                        s.push(',');
                    }
                    json::write_str(&mut s, &slot.name);
                    s.push(':');
                    let v = toks.get(i).copied().unwrap_or(l.end);
                    i += 1;
                    if slot.span {
                        if v == l.none || v < l.nv {
                            s.push_str("null");
                        } else {
                            let e = toks.get(i).copied().unwrap_or(v);
                            i += 1;
                            let (a, b) = (v - l.nv, (e - l.nv).min(words.len().saturating_sub(1)));
                            used.extend(a..=b);
                            json::write_str(&mut s, &words[a..=b].join(" "));
                        }
                    } else {
                        json::write_str(&mut s, &l.vocab[v]);
                    }
                }
            }
            s.push_str("}]");
        }
        s.push_str("]]");
        used.sort_unstable();
        used.dedup();
        (s, used)
    }
}
