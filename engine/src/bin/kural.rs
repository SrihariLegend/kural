//! kural: plain English to a bash command. It prints the command; it never runs anything.
//!
//!   kural REQUEST...             the command, on stdout
//!   kural -v REQUEST...          also the reading, confidence and alternatives (stderr)
//!   kural --shell REQUEST...     line 1 command, line 2 risk (0-2), line 3 confidence, line 4 reading
//!   kural --hook REQUEST...      for the bash hook: one line, fields separated by 0x1f (count, then per candidate
//!                                command, risk, confidence, reading; a newline inside a field is sent as 0x1e)
//! Options: --session FILE (this terminal's session file; default $KURAL_SESSION), --cwd DIR, --preview (while typing:
//!          nothing is recorded, nothing slow is looked up), --choose N (the user picked candidate N)
//! Testing: --batch (JSON lines on stdin, see below), --compile-json PROGRAM [--live] [--last CMD] [--prefs a,b]
//!
//! The model file and lexicon.json are found via KURAL_MODEL / KURAL_LEXICON, next to the executable,
//! in ../share/kural/, or in the source tree's kural/ folder.

use kural_engine::answer::{Kural, Lexicon, Offer};
use kural_engine::compile;
use kural_engine::ground::Env;
use kural_engine::json::{self, Json};
use kural_engine::program;
use kural_engine::py;
use kural_engine::Engine;
use std::collections::BTreeMap;
use std::io::{BufRead, Write};

fn find(env_var: &str, name: &str) -> String {
    if let Ok(p) = std::env::var(env_var) {
        if !p.is_empty() {
            return p;
        }
    }
    let mut cands = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        let exe = std::fs::canonicalize(&exe).unwrap_or(exe);
        if let Some(d) = exe.parent() {
            cands.push(d.join(name));
            cands.push(d.join("../share/kural").join(name));
            cands.push(d.join("../../kural").join(name));
        }
    }
    cands.push(std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../kural")).join(name));
    for c in &cands {
        if c.exists() {
            return c.to_string_lossy().into_owned();
        }
    }
    cands.last().unwrap().to_string_lossy().into_owned()
}

fn load() -> Kural {
    let model = find("KURAL_MODEL", "model.kural");
    let engine = match Engine::load(&model) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("kural: {e}");
            std::process::exit(3);
        }
    };
    Kural::new(engine, Lexicon::load(&find("KURAL_LEXICON", "lexicon.json")))
}

fn offers_json(offers: &[Offer]) -> String {
    let mut s = String::from("[");
    for (i, o) in offers.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str("{\"command\":");
        json::write_str(&mut s, &o.command);
        s.push_str(&format!(",\"confidence\":{:.9},\"risk\":{},\"reading\":", o.confidence, o.risk));
        json::write_str(&mut s, &o.reading);
        s.push('}');
    }
    s.push(']');
    s
}

/// --batch: each stdin line is {"text": ..., "context": [...] | null, "last": ... | null}; each output line the offers,
/// computed offline (live=False, pm=pacman), exactly what tests/evaluate.py asks of the Python code.
fn batch(k: &Kural) {
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let j = json::parse(&line).unwrap_or(Json::Null);
        let text = j.get("text").str().to_string();
        let ctx: Vec<String> = j.get("context").arr().iter().map(|w| w.str().to_string()).collect();
        let last = match j.get("last") {
            Json::Str(s) => Some(s.clone()),
            _ => None,
        };
        let env = Env::offline(last);
        let offers = k.translate(&text, &env, &ctx);
        let _ = writeln!(out, "{}", offers_json(&offers));
    }
}

/// --compile-batch: each stdin line {"program": ..., "last": ...}; each output line
/// {"command": ... | null, "error": "unsupported" | "other" | null, "risk": n, "reading": ..., "context": [...]} (offline).
fn compile_batch(k: &Kural) {
    let stdin = std::io::stdin();
    let mut out = std::io::BufWriter::new(std::io::stdout());
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let j = json::parse(&line).unwrap_or(Json::Null);
        let Some(p) = program::from_json(&k.lang, j.get("program")) else {
            let _ = writeln!(out, "{{\"command\":null,\"error\":\"bad program\"}}");
            continue;
        };
        let last = match j.get("last") { Json::Str(s) => Some(s.clone()), _ => None };
        let mut env = Env::offline(last);
        let mut s = String::from("{\"command\":");
        match compile::compile_program(&k.lang, &p, &mut env) {
            Ok(c) => {
                json::write_str(&mut s, &c);
                s.push_str(",\"error\":null");
            }
            Err(e) => s.push_str(&format!("null,\"error\":\"{}\"", if e == compile::CErr::Unsupported { "unsupported" } else { "other" })),
        }
        s.push_str(&format!(",\"risk\":{},\"reading\":", compile::risk(&k.lang, &p)));
        json::write_str(&mut s, &(compile::describe(&k.lang, &p) + &env.notes.iter().map(|n| format!("  ({n})")).collect::<String>()));
        s.push_str(",\"context\":[");
        for (i, w) in program::context_words(&k.lang, &p).iter().enumerate() {
            if i > 0 {
                s.push(',');
            }
            json::write_str(&mut s, w);
        }
        s.push_str("],\"pyjson\":");
        json::write_str(&mut s, &program::to_python_json(&p));
        s.push('}');
        let _ = writeln!(out, "{s}");
    }
}

fn compile_json(k: &Kural, prog: &str, live: bool, cwd: Option<String>, last: Option<String>, prefs: &str) -> i32 {
    let Some(p) = json::parse(prog).ok().and_then(|j| program::from_json(&k.lang, &j)) else {
        eprintln!("kural: not a program");
        return 2;
    };
    let mut pf = BTreeMap::new();
    for x in prefs.split(',').filter(|x| !x.is_empty()) {
        pf.insert(x.to_string(), true);
    }
    let mut env = Env::new(cwd, live, if live { None } else { Some("pacman".into()) }, pf, last, true);
    match compile::compile_program(&k.lang, &p, &mut env) {
        Ok(c) => {
            println!("{c}");
            if !env.notes.is_empty() {
                eprintln!("notes: {}", env.notes.join("; "));
            }
            0
        }
        Err(e) => {
            eprintln!("kural: {e:?}");
            1
        }
    }
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let (mut verbose, mut shell, mut hook, mut preview) = (false, false, false, false);
    let mut sess = std::env::var("KURAL_SESSION").ok();
    let (mut cwd, mut choose): (Option<String>, usize) = (None, 0);
    let (mut live, mut last, mut prefs, mut cjson): (bool, Option<String>, String, Option<String>) = (false, None, String::new(), None);
    while !args.is_empty() && args[0].starts_with('-') && args[0] != "--" {
        let flag = args.remove(0);
        match flag.as_str() {
            "-v" => verbose = true,
            "--shell" => shell = true,
            "--hook" => hook = true,
            "--preview" => preview = true,
            "--batch" => {
                batch(&load());
                return;
            }
            "--compile-batch" => {
                compile_batch(&load());
                return;
            }
            "--live" => live = true,
            "--session" if !args.is_empty() => sess = Some(args.remove(0)),
            "--cwd" if !args.is_empty() => cwd = Some(args.remove(0)),
            "--choose" if !args.is_empty() => choose = args.remove(0).parse().unwrap_or(0),
            "--last" if !args.is_empty() => last = Some(args.remove(0)),
            "--prefs" if !args.is_empty() => prefs = args.remove(0),
            "--compile-json" if !args.is_empty() => cjson = Some(args.remove(0)),
            _ => {
                args.insert(0, flag);
                break;
            }
        }
    }
    if args.first().map(|a| a == "--").unwrap_or(false) {
        args.remove(0);
    }
    if let Some(p) = cjson {
        let k = load();
        std::process::exit(compile_json(&k, &p, live, cwd, last, &prefs));
    }
    let text = args.join(" ").trim().to_string();
    if text.is_empty() {
        eprintln!("usage: kural [-v|--shell|--hook] [--session FILE] [--cwd DIR] [--preview] [--choose N] REQUEST...");
        std::process::exit(2);
    }
    let k = load();
    let cwd = cwd.or_else(|| std::env::current_dir().ok().map(|p| p.to_string_lossy().into_owned()));
    let results = k.answer(&text, cwd.as_deref(), sess.as_deref(), !preview, true, None, choose);
    if hook {
        let clean = |s: &str| s.replace('\n', "\u{1e}").replace('\u{1f}', " ");
        let mut fields = vec![results.len().to_string()];
        for r in &results {
            fields.push(clean(&r.command));
            fields.push(r.risk.to_string());
            fields.push(py::fmt_fixed(r.confidence, 3));
            fields.push(clean(&r.reading));
        }
        println!("{}", fields.join("\u{1f}"));
        return;
    }
    if results.is_empty() {
        eprintln!("kural: that is outside what I can do so far");
        std::process::exit(1);
    }
    let best = &results[0];
    if shell {
        println!("{}\n{}\n{}\n{}", best.command, best.risk, py::fmt_fixed(best.confidence, 3), best.reading);
        return;
    }
    if best.command.is_empty() {
        println!("({})", best.reading);
    } else {
        println!("{}", best.command);
    }
    if verbose {
        eprintln!("  reading:    {}", best.reading);
        eprintln!("  confidence: {}%", py::fmt_fixed(100.0 * best.confidence, 0));
        for alt in &results[1..] {
            eprintln!("  or ({}%): {}", py::fmt_fixed(100.0 * alt.confidence, 0), alt.command);
        }
    }
}
