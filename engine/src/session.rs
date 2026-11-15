//! The per-terminal session file (kural/session.py): same format, read and written the same way.

use crate::json;
use crate::lang::Language;
use crate::program::{self, Program};
use crate::py;
use crate::tok;
use std::collections::{BTreeMap, HashMap};
use std::io::Write;

pub const MAX_RAW_WORDS: usize = 12;

pub fn runtime_dir() -> String {
    let d = match std::env::var("XDG_RUNTIME_DIR") {
        Ok(b) if !b.is_empty() && std::path::Path::new(&b).is_dir() => format!("{b}/kural"),
        _ => format!("/tmp/kural-{}", unsafe { getuid() }),
    };
    let _ = std::fs::create_dir_all(&d);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o700));
    }
    d
}

extern "C" {
    fn getuid() -> u32;
}

pub fn esc(s: &str) -> String {
    s.replace('\\', "\\\\").replace('\t', "\\t").replace('\n', "\\n")
}

pub fn unesc(s: &str) -> String {
    let cs: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < cs.len() {
        if cs[i] == '\\' && i + 1 < cs.len() {
            out.push(match cs[i + 1] {
                't' => '\t',
                'n' => '\n',
                '\\' => '\\',
                c => c,
            });
            i += 2;
        } else {
            out.push(cs[i]);
            i += 1;
        }
    }
    out
}

pub fn norm(request: &str) -> String {
    tok::words(&request.to_lowercase()).join(" ")
}

#[derive(Default)]
pub struct State {
    pub context: Option<Vec<String>>,
    pub last: Option<String>,
    pub prefs: BTreeMap<String, bool>,
    pub fixes: HashMap<String, String>,
    pub asks: Vec<String>,
}

fn config_prefs(lang: &Language) -> BTreeMap<String, bool> {
    let mut prefs = BTreeMap::new();
    let base = match std::env::var("XDG_CONFIG_HOME") {
        Ok(x) if !x.is_empty() => x,
        _ => py::expanduser("~/.config"),
    };
    if let Ok(text) = std::fs::read_to_string(format!("{base}/kural/config")) {
        for line in text.lines() {
            let line = line.trim();
            let (k, v) = line.split_once('=').unwrap_or((line, ""));
            if lang.enum_values("prefkey").iter().any(|p| p == k) {
                prefs.insert(k.to_string(), ["on", "1", "true", "yes"].contains(&v.trim().to_lowercase().as_str()));
            }
        }
    }
    prefs
}

pub fn read(lang: &Language, path: Option<&str>, cwd: Option<&str>) -> State {
    let mut st = State { prefs: config_prefs(lang), ..Default::default() };
    let Some(path) = path.filter(|p| !p.is_empty()) else { return st };
    let Ok(bytes) = std::fs::read(path) else { return st };
    let text = String::from_utf8_lossy(&bytes);
    let mut offer: Option<(Program, String, Option<String>)> = None;
    let mut pending_cwd: Option<String> = None;
    let mut pending_prog: Option<Program> = None;
    let mut have_pending = false;
    let mut newest = "";
    let mut run: Option<(String, String)> = None;
    for line in text.split('\n') {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "run" if f.len() >= 5 => {
                run = Some((unesc(f[4]), unesc(f[3])));
                newest = "run";
            }
            "ask" if f.len() >= 4 => {
                pending_cwd = Some(unesc(f[2]));
                pending_prog = None;
                have_pending = true;
                st.asks.push(unesc(f[3]));
            }
            "got" if f.len() >= 2 => match json::parse(f[1]).ok().and_then(|j| program::from_json(lang, &j)) {
                Some(p) => pending_prog = Some(p),
                None => pending_prog = None,
            },
            "cmd" if f.len() >= 2 && pending_prog.is_some() => {
                offer = Some((pending_prog.clone().unwrap(), unesc(f[1]), if have_pending { pending_cwd.clone() } else { None }));
                newest = "offer";
            }
            "fix" if f.len() >= 4 => {
                let (offered, ran) = (unesc(f[2]), unesc(f[3]));
                let same_tool = offered.split_whitespace().next() == ran.split_whitespace().next();
                if same_tool || py::ratio(&offered, &ran) >= 0.5 {
                    st.fixes.insert(norm(&unesc(f[1])), ran);
                }
            }
            "pref" if f.len() >= 3 && lang.enum_values("prefkey").iter().any(|p| p == f[1]) => {
                st.prefs.insert(f[1].to_string(), f[2] == "on");
            }
            _ => {}
        }
    }
    if let Some((cmd, _)) = &run {
        st.last = Some(cmd.clone());
    }
    if let Some((prog, cmd, ocwd)) = &offer {
        let mut use_program = newest == "offer" || run.as_ref().map(|(r, _)| r == cmd).unwrap_or(false);
        if use_program {
            if let (Some(c), Some(o)) = (cwd.filter(|c| !c.is_empty()), ocwd.as_deref().filter(|o| !o.is_empty())) {
                if py::normpath(c) != py::normpath(o) {
                    use_program = false;
                }
            }
        }
        if use_program && !prog.iter().flatten().any(|s| s.name == "unknown" || s.name == "pref") {
            let words = program::context_words(lang, prog);
            if words.len() <= 36 {
                st.context = Some(words);
            }
        }
    }
    if st.context.is_none() {
        if let Some((cmd, _)) = &run {
            let mut w = vec!["@raw".to_string()];
            w.extend(tok::words(cmd).into_iter().take(MAX_RAW_WORDS));
            st.context = Some(w);
        }
    }
    st
}

fn append(path: &str, lines: &[String]) {
    if let Ok(mut f) = std::fs::OpenOptions::new().append(true).create(true).open(path) {
        let s: String = lines.iter().map(|l| format!("{l}\n")).collect();
        let _ = f.write_all(s.as_bytes());
    }
}

pub fn record(path: Option<&str>, request: &str, cwd: Option<&str>, prog: &Program, command: &str) {
    let Some(path) = path.filter(|p| !p.is_empty()) else { return };
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    append(path, &[
        format!("ask\t{now}\t{}\t{}", esc(cwd.unwrap_or("")), esc(request)),
        format!("got\t{}", program::to_python_json(prog)),
        format!("cmd\t{}", esc(command)),
    ]);
}

pub fn set_pref(path: Option<&str>, key: &str, state: &str) {
    if let Some(path) = path.filter(|p| !p.is_empty()) {
        append(path, &[format!("pref\t{key}\t{state}")]);
    }
}
