//! From a request to the commands offered (kural/cli.py translate/answer and kural/lexicon.py support):
//! the parser's readings, the guard against unasked destructive actions, the vocabulary check, session context,
//! corrections the user made before, and recording what was offered.

use crate::compile::{self, CErr};
use crate::ground::Env;
use crate::json;
use crate::lang::{CommandDef, Language};
use crate::program::{self, Program, Stage};
use crate::py;
use crate::search::Engine;
use crate::session;
use std::collections::{HashMap, HashSet};

// ------------------------------------------------------------------ vocabulary check (lexicon.json)

pub struct Lexicon {
    pub generic: HashSet<String>,
    pub stages: HashMap<String, HashSet<String>>,
}

impl Lexicon {
    pub fn empty() -> Lexicon {
        Lexicon { generic: HashSet::new(), stages: HashMap::new() }
    }
    pub fn load(path: &str) -> Lexicon {
        let Ok(text) = std::fs::read_to_string(path) else { return Lexicon::empty() };
        let Ok(j) = json::parse(&text) else { return Lexicon::empty() };
        let set = |v: &json::Json| v.arr().iter().map(|x| x.str().to_string()).collect::<HashSet<_>>();
        let generic = set(j.get("generic"));
        let mut stages = HashMap::new();
        if let Some(m) = j.get("stages").obj() {
            for (k, v) in m {
                stages.insert(k.clone(), set(v));
            }
        } else {
            return Lexicon::empty();
        }
        Lexicon { generic, stages }
    }

    /// Lower-case stems worth checking: re.findall(r"[a-z][a-z0-9+\-]*", w.lower()), 3+ long
    fn content(words: &[&str]) -> Vec<String> {
        let mut out = Vec::new();
        for w in words {
            let cs: Vec<char> = w.to_lowercase().chars().collect();
            let mut i = 0;
            while i < cs.len() {
                if cs[i].is_ascii_lowercase() {
                    let s = i;
                    i += 1;
                    while i < cs.len() && (cs[i].is_ascii_lowercase() || cs[i].is_ascii_digit() || cs[i] == '+' || cs[i] == '-') {
                        i += 1;
                    }
                    if i - s >= 3 {
                        out.push(cs[s..i].iter().collect());
                    }
                } else {
                    i += 1;
                }
            }
        }
        out
    }

    fn known(word: &str, vocab: &HashSet<&str>) -> bool {
        if vocab.contains(word) {
            return true;
        }
        if word.chars().count() >= 5 {
            let stem: String = word.chars().take(5).collect();
            if vocab.iter().any(|v| v.starts_with(&stem)) {
                return true;
            }
            return vocab.iter().any(|v| py::ratio_at_least(v, word, 0.8));
        }
        false
    }

    /// (meaningful words that fit the reading, meaningful words in the request), or None if nothing can be checked.
    pub fn support(&self, program: &Program, words: &[String], used: &[usize]) -> Option<(usize, usize)> {
        if self.stages.is_empty() {
            return None;
        }
        let literal: Vec<&str> = words.iter().enumerate().filter(|(i, _)| !used.contains(i)).map(|(_, w)| w.as_str()).collect();
        let cand: Vec<String> = Lexicon::content(&literal).into_iter().filter(|w| !self.generic.contains(w)).collect();
        if cand.is_empty() {
            return None;
        }
        let mut vocab: HashSet<&str> = HashSet::new();
        for st in program.iter().flatten() {
            if let Some(s) = self.stages.get(&st.name) {
                vocab.extend(s.iter().map(|x| x.as_str()));
            }
        }
        Some((cand.iter().filter(|w| Lexicon::known(w, &vocab)).count(), cand.len()))
    }
}

// ------------------------------------------------------------------ guards

/// Every word a phrasing can start with (kural/commands.py _first_words).
fn first_words(p: &str) -> HashSet<String> {
    let p = p.trim();
    let cs: Vec<char> = p.chars().collect();
    if cs.is_empty() {
        return HashSet::new();
    }
    if cs[0] == '(' || cs[0] == '[' {
        let close_sq = cs[0] == '[';
        let (mut depth, mut parts, mut cur) = (0i32, Vec::<String>::new(), String::new());
        let mut rest = String::new();
        for (i, &ch) in cs.iter().enumerate() {
            if ch == '(' || ch == '[' {
                depth += 1;
                if depth == 1 {
                    continue;
                }
            } else if ch == ')' || ch == ']' {
                depth -= 1;
                if depth == 0 {
                    parts.push(std::mem::take(&mut cur));
                    rest = cs[i + 1..].iter().collect();
                    break;
                }
            } else if ch == '|' && depth == 1 {
                parts.push(std::mem::take(&mut cur));
                continue;
            }
            cur.push(ch);
        }
        let mut out = HashSet::new();
        for part in &parts {
            if part.trim().is_empty() {
                out.extend(first_words(&rest));
            } else {
                out.extend(first_words(part));
            }
        }
        if close_sq {
            out.extend(first_words(&rest));
        }
        return out;
    }
    let w = p.split_whitespace().next().unwrap_or("");
    if w.starts_with('{') { HashSet::new() } else { [w.to_string()].into_iter().collect() }
}

pub fn verbs(c: &CommandDef) -> Vec<String> {
    let mut out: HashSet<String> = HashSet::new();
    for p in &c.phrases {
        for w in first_words(p) {
            let w: String = w.to_lowercase().chars().filter(|ch| ch.is_alphanumeric()).collect();
            if w.chars().count() >= 2 {
                out.insert(w.chars().take(5).collect());
            }
        }
    }
    let mut v: Vec<String> = out.into_iter().collect();
    v.sort();
    v
}

pub struct Guards {
    asks_for: HashMap<String, Vec<String>>,
}

const RENAME_STEMS: &[&str] = &["renam", "lower", "upper", "caps", "capital", "replac", "chang", "swap", "prefix", "prepend", "suffix",
    "append", "add", "put", "extens", "space", "underscor", "fix", "turn", "convert", "mv", "make", "switch", "give", "remov", "call"];

impl Guards {
    pub fn new(lang: &Language) -> Guards {
        let mut m: HashMap<String, Vec<String>> = HashMap::new();
        let mut put = |k: &str, v: &[&str]| {
            m.insert(k.to_string(), v.iter().map(|s| s.to_string()).collect());
        };
        put("delete", &["delet", "remov", "rm", "eras", "wip", "trash", "purg", "nuk", "clean", "clear", "discard", "destroy", "drop", "rid", "throw"]);
        put("kill", &["kill", "stop", "terminat", "end", "clos", "quit", "pkill", "nuk", "murder", "shut", "exit", "free"]);
        put("move", &["mov", "mv", "transfer", "relocat", "shift", "put", "send", "dump", "stick", "renam"]);
        put("trash", &["trash", "bin", "recycl", "delet", "remov", "rid", "throw", "send", "put", "mov", "dump", "toss"]);
        put("replace", &["replac", "chang", "swap", "substitut", "sed", "turn", "renam", "updat", "fix", "switch", "convert"]);
        put("inplace", &["place", "in-place", "save", "overwrit", "writ", "perman", "updat", "good", "replac", "chang", "swap", "substitut",
            "turn", "renam", "sed", "edit", "modif", "fix"]);
        for n in ["rn_lower", "rn_upper", "rn_spaces", "rn_replace", "rn_prefix", "rn_suffix", "rn_ext"] {
            put(n, RENAME_STEMS);
        }
        for c in &lang.commands {
            if c.risk >= 2 {
                m.insert(c.name.clone(), verbs(c));
            }
        }
        Guards { asks_for: m }
    }

    fn guarded(&self, name: &str) -> bool {
        self.asks_for.contains_key(name)
    }

    /// Destructive stages in the program that no word of the request asks for.
    pub fn unasked(&self, program: &Program, text: &str, context: &[String]) -> Vec<String> {
        let words: Vec<String> = text.split_whitespace().map(|w| py::strip_chars(w, ".,!?'\"").to_lowercase()).collect();
        let before: HashSet<&str> = context.iter().filter(|w| w.starts_with('@')).map(|w| &w[1..]).collect();
        let mut out = Vec::new();
        for st in program.iter().flatten() {
            if let Some(stems) = self.asks_for.get(&st.name) {
                if !before.contains(st.name.as_str()) && !words.iter().any(|w| stems.iter().any(|s| w.starts_with(s.as_str()))) {
                    out.push(st.name.clone());
                }
            }
        }
        out
    }

    /// The same selection with every destructive action replaced by a listing, or None.
    fn safe(&self, lang: &Language, program: &Program) -> Option<Program> {
        let mut out = Vec::new();
        for pipe in program {
            let mut new = Vec::new();
            for st in pipe {
                if !self.guarded(&st.name) {
                    new.push(st.clone());
                } else if st.name == "kill" {
                    new.push(Stage::new("pshow"));
                } else if st.name == "inplace" {
                    new.push(Stage::new("lshow"));
                } else if lang.command(&st.name).is_some() {
                    new.clear();
                    break;
                } else {
                    new.push(Stage::new("fshow"));
                }
            }
            if !new.is_empty() {
                out.push(new);
            }
        }
        if out.is_empty() { None } else { Some(out) }
    }
}

// ------------------------------------------------------------------ translate / answer

#[derive(Clone, Debug)]
pub struct Offer {
    pub command: String,
    pub confidence: f64,
    pub risk: u8,
    pub reading: String,
    pub program: Option<Program>,
}

pub struct Kural {
    pub engine: Engine,
    pub lang: Language,
    pub lexicon: Lexicon,
    pub guards: Guards,
}

impl Kural {
    pub fn new(engine: Engine, lexicon: Lexicon) -> Kural {
        let lang = Language::load();
        let guards = Guards::new(&lang);
        Kural { engine, lang, lexicon, guards }
    }

    /// The parser's readings: (program, probability, request words, copied positions).
    pub fn parse(&self, text: &str, context: &[String]) -> Vec<(Program, f64, Vec<String>, Vec<usize>)> {
        let (words, n) = self.engine.words(text, context);
        self.engine
            .parse(&words, n, 4, 3)
            .into_iter()
            .map(|p| {
                let (prog, used) = program::from_tokens(&self.engine.lang, &p.tokens, &words);
                (prog, p.prob, words[..n].to_vec(), used)
            })
            .collect()
    }

    pub fn translate(&self, text: &str, env: &Env, context: &[String]) -> Vec<Offer> {
        let mut out = Vec::new();
        let mut held: Vec<(Program, f64)> = Vec::new();
        let mut foreign = false;
        for (i, (prog, prob, words, used)) in self.parse(text, context).into_iter().enumerate() {
            let mut e = env.fresh();
            if !self.guards.unasked(&prog, text, context).is_empty() {
                held.push((prog, prob));
                continue;
            }
            let cmd = match compile::compile_program(&self.lang, &prog, &mut e) {
                Ok(c) => c,
                Err(CErr::Unsupported) => {
                    if i == 0 {
                        return Vec::new();
                    }
                    continue;
                }
                Err(CErr::Other(_)) => continue,
            };
            let mut prob = prob;
            if let Some((fit, total)) = self.lexicon.support(&prog, &words, &used) {
                if fit == 0 {
                    if i == 0 && prob < 0.999 {
                        foreign = true;
                    }
                    continue;
                }
                if fit * 2 < total {
                    prob = prob.min(0.6);
                }
            }
            let reading = compile::describe(&self.lang, &prog) + &e.notes.iter().map(|n| format!("  ({n})")).collect::<String>();
            out.push(Offer { command: cmd, confidence: prob, risk: compile::risk(&self.lang, &prog), reading, program: Some(prog) });
        }
        if foreign {
            return Vec::new();
        }
        if out.is_empty() {
            if let Some((prog, prob)) = held.into_iter().next() {
                if let Some(safe) = self.guards.safe(&self.lang, &prog) {
                    let mut e = env.fresh();
                    if let Ok(cmd) = compile::compile_program(&self.lang, &safe, &mut e) {
                        out.push(Offer { command: cmd, confidence: prob * 0.5, risk: 0, reading: compile::describe(&self.lang, &safe) + "  (listing only)", program: Some(safe) });
                    }
                }
            }
        }
        out
    }

    /// One request from a shell: session context in, and on a final request the offer is recorded.
    pub fn answer(&self, text: &str, cwd: Option<&str>, session_path: Option<&str>, final_: bool, live: bool, pm: Option<String>, choose: usize) -> Vec<Offer> {
        let st = session::read(&self.lang, session_path, cwd);
        let env = Env::new(cwd.map(|c| c.to_string()), live, pm, st.prefs.clone(), st.last.clone(), final_);
        let ctx = st.context.clone().unwrap_or_default();
        let mut results = self.translate(text, &env, &ctx);
        if let Some(fixed) = st.fixes.get(&session::norm(text)) {
            if !results.first().map(|r| &r.command == fixed).unwrap_or(false) {
                results.insert(0, Offer { command: fixed.clone(), confidence: 1.0, risk: 1, reading: "what you ran last time you asked this".into(), program: None });
            }
        }
        let mut kept: Vec<Offer> = results.iter().take(1).cloned().collect();
        kept.extend(results.iter().skip(1).filter(|r| r.confidence >= 0.03).cloned());
        let mut results = kept;
        if choose > 0 && choose < results.len() {
            let c = results.remove(choose);
            results.insert(0, c);
        }
        if let Some(best) = results.first_mut() {
            if let Some(prog) = best.program.clone() {
                let first = prog.first().and_then(|p| p.first());
                if let Some(pref) = first.filter(|s| s.name == "pref") {
                    let key = pref.get("key").unwrap_or("").to_string();
                    let state = pref.get("state").unwrap_or("").to_string();
                    let what = match key.as_str() {
                        "trash" => "deletes go to the trash",
                        "confirm" => "ask before deleting or overwriting",
                        _ => "commands print each file",
                    };
                    best.reading = format!("{what}{} (this session)", if state == "on" { ": on" } else { ": off" });
                    if final_ {
                        session::set_pref(session_path, &key, &state);
                    }
                } else if final_ {
                    session::record(session_path, text, cwd, &prog, &best.command);
                }
            }
        }
        results
    }
}
