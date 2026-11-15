//! A parsed program: pipelines of stages, each with its arguments in slot order (as kural/ir.py's delinearise
//! gives them). Arguments are the copied words (spans) or an enum value; None when an optional slot is empty.

use crate::json::{self, Json};
use crate::lang::{Language, SlotKind};
use crate::search::Lang;

#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub name: String,
    pub args: Vec<(String, Option<String>)>,
}

impl Stage {
    pub fn new(name: &str) -> Stage {
        Stage { name: name.to_string(), args: Vec::new() }
    }
    /// Argument value, None if missing or empty slot.
    pub fn get(&self, slot: &str) -> Option<&str> {
        self.args.iter().find(|(k, _)| k == slot).and_then(|(_, v)| v.as_deref())
    }
    pub fn set(&mut self, slot: &str, v: Option<String>) {
        if let Some(e) = self.args.iter_mut().find(|(k, _)| k == slot) {
            e.1 = v;
        } else {
            self.args.push((slot.to_string(), v));
        }
    }
}

pub type Pipe = Vec<Stage>;
pub type Program = Vec<Pipe>;

/// Model output tokens -> program, plus the positions of the words copied as arguments.
pub fn from_tokens(l: &Lang, toks: &[usize], words: &[String]) -> (Program, Vec<usize>) {
    let mut program: Program = Vec::new();
    let mut pipe: Pipe = Vec::new();
    let mut used = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        i += 1;
        if t == l.end || t == l.pad {
            break;
        }
        if t == l.sep {
            program.push(std::mem::take(&mut pipe));
            continue;
        }
        let mut st = Stage::new(&l.vocab[t]);
        if let Some(info) = l.stages.get(&t) {
            for slot in &info.slots {
                let v = toks.get(i).copied().unwrap_or(l.end);
                i += 1;
                if slot.span {
                    if v == l.none || v < l.nv {
                        st.args.push((slot.name.clone(), None));
                    } else {
                        let e = toks.get(i).copied().unwrap_or(v);
                        i += 1;
                        let (a, b) = (v - l.nv, (e - l.nv).min(words.len().saturating_sub(1)));
                        used.extend(a..=b);
                        st.args.push((slot.name.clone(), Some(words[a..=b].join(" "))));
                    }
                } else {
                    st.args.push((slot.name.clone(), Some(l.vocab[v].clone())));
                }
            }
        }
        pipe.push(st);
    }
    if !pipe.is_empty() {
        program.push(pipe);
    }
    used.sort_unstable();
    used.dedup();
    (program, used)
}

/// The words a program gives the next request as context (kural/ir.py context_words).
pub fn context_words(lang: &Language, program: &Program) -> Vec<String> {
    let mut words = Vec::new();
    for (i, pipe) in program.iter().enumerate() {
        if i > 0 {
            words.push("@;".to_string());
        }
        for st in pipe {
            words.push(format!("@{}", st.name));
            if let Some(def) = lang.stage(&st.name) {
                for slot in &def.slots {
                    let v = st.get(&slot.name);
                    match (&slot.kind, v) {
                        (SlotKind::Enum(_), Some(v)) => words.push(format!("@{v}")),
                        (SlotKind::Enum(_), None) => words.push("@None".into()),
                        (SlotKind::Span(_), None) => words.push("@none".into()),
                        (SlotKind::Span(_), Some(v)) => words.push(v.to_string()),
                    }
                }
            }
        }
    }
    words
}

/// json.dumps(program, ensure_ascii=False) exactly as Python writes it.
pub fn to_python_json(p: &Program) -> String {
    let mut s = String::from("[");
    for (i, pipe) in p.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        s.push('[');
        for (j, st) in pipe.iter().enumerate() {
            if j > 0 {
                s.push_str(", ");
            }
            s.push('[');
            py_str(&mut s, &st.name);
            s.push_str(", {");
            for (k, (name, v)) in st.args.iter().enumerate() {
                if k > 0 {
                    s.push_str(", ");
                }
                py_str(&mut s, name);
                s.push_str(": ");
                match v {
                    Some(v) => py_str(&mut s, v),
                    None => s.push_str("null"),
                }
            }
            s.push_str("}]");
        }
        s.push(']');
    }
    s.push(']');
    s
}

fn py_str(out: &mut String, v: &str) {
    out.push('"');
    for c in v.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A program from JSON ([[["stage", {...}], ...], ...]); arguments are put in slot order. None if malformed.
pub fn from_json(lang: &Language, j: &Json) -> Option<Program> {
    let mut program = Vec::new();
    for pipe in j.arr() {
        let mut out = Vec::new();
        for st in pipe.arr() {
            let parts = st.arr();
            let name = parts.first()?.str();
            let def = lang.stage(name)?;
            let args = parts.get(1).cloned().unwrap_or(Json::Null);
            let mut s = Stage::new(name);
            for slot in &def.slots {
                let v = args.get(&slot.name);
                s.args.push((slot.name.clone(), match v {
                    Json::Str(x) => Some(x.clone()),
                    Json::Num(n) => Some(format!("{n}")),
                    _ => None,
                }));
            }
            out.push(s);
        }
        program.push(out);
    }
    Some(program)
}

#[allow(dead_code)]
fn _j(s: &str) -> String {
    let mut o = String::new();
    json::write_str(&mut o, s);
    o
}
