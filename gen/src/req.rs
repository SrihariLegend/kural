//! A request under construction (English chunks plus the program that goes with them), argument values, and
//! turning both into a training pair. Ports Req, val, put, emit, tail, finish and filter_stages from kural/gen.py,
//! and delinearise / context_words from kural/ir.py.

use crate::pools::*;
use crate::rng::Rng;
use kural_engine::lang::{Language, SlotKind};
use kural_engine::tok;
use std::collections::HashMap;
use std::sync::LazyLock;

// --------------------------------------------------------------------------- the language
pub struct Lang {
    pub l: Language,
    pub stage_ix: HashMap<String, usize>,
    pub nv: usize,
    pub pad: u16,
    pub bos: u16,
    pub end: u16,
    pub sep: u16,
    pub none: u16,
    pub filter_order: Vec<&'static str>,
    pub negatable: Vec<&'static str>,
}

pub static LANG: LazyLock<Lang> = LazyLock::new(|| {
    let l = Language::load();
    let stage_ix = l.stages.iter().enumerate().map(|(i, s)| (s.name.clone(), i)).collect();
    let t = |s: &str| l.tok[s] as u16;
    let (pad, bos, end, sep, none) = (t("<pad>"), t("<bos>"), t("<end>"), t(";"), t("<none>"));
    let nv = l.vocab.len();
    let leak = |v: &Vec<String>| v.iter().map(|s| &*Box::leak(s.clone().into_boxed_str())).collect::<Vec<&'static str>>();
    let filter_order = leak(&l.lists["filter_order"]);
    let negatable = leak(&l.lists["negatable"]);
    Lang { l, stage_ix, nv, pad, bos, end, sep, none, filter_order, negatable }
});

pub fn lang() -> &'static Lang {
    &LANG
}

/// A name from the language, as a 'static string.
pub fn sname(name: &str) -> &'static str {
    let lg = lang();
    lg.l.stages[lg.stage_ix[name]].name.as_str()
}

// --------------------------------------------------------------------------- programs
/// A slot value while building: a key into the request's chunks, a position in the context, or an enum value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum V {
    K(u32),
    Ctx(usize),
    E(&'static str),
}

pub type Args = Vec<(&'static str, Option<V>)>;
pub type Stage = (&'static str, Args);
pub type Pipe = Vec<Stage>;

pub fn arg(a: &Args, name: &str) -> Option<V> {
    a.iter().find(|(n, _)| *n == name).and_then(|(_, v)| *v)
}

pub fn set_arg(a: &Args, name: &'static str, v: Option<V>) -> Args {
    let mut out: Args = a.iter().filter(|(n, _)| *n != name).cloned().collect();
    out.push((name, v));
    out
}

#[macro_export]
macro_rules! st {
    ($name:expr) => { ($name, Vec::new()) };
    ($name:expr, $($k:literal => $v:expr),+ $(,)?) => { ($name, vec![$(($k, $v)),+]) };
}

/// A delinearised slot value: copied words, an enum value, or nothing.
#[derive(Clone, Debug)]
pub enum DV {
    Str(String),
    Enum(&'static str),
    Null,
}
pub type DStage = (&'static str, Vec<(&'static str, DV)>);
pub type DProg = Vec<Vec<DStage>>;

// --------------------------------------------------------------------------- phrase parts
#[derive(Clone, Debug)]
pub enum P {
    L(String),
    A(String, u32, bool),
}

impl From<&str> for P {
    fn from(s: &str) -> P {
        P::L(s.to_string())
    }
}
impl From<String> for P {
    fn from(s: String) -> P {
        P::L(s)
    }
}
impl From<&String> for P {
    fn from(s: &String) -> P {
        P::L(s.clone())
    }
}
impl From<(String, u32)> for P {
    fn from(a: (String, u32)) -> P {
        P::A(a.0, a.1, false)
    }
}
impl From<&(String, u32)> for P {
    fn from(a: &(String, u32)) -> P {
        P::A(a.0.clone(), a.1, false)
    }
}
impl From<(String, u32, bool)> for P {
    fn from(a: (String, u32, bool)) -> P {
        P::A(a.0, a.1, a.2)
    }
}
impl From<&(String, u32, bool)> for P {
    fn from(a: &(String, u32, bool)) -> P {
        P::A(a.0.clone(), a.1, a.2)
    }
}
impl From<(&str, u32)> for P {
    fn from(a: (&str, u32)) -> P {
        P::A(a.0.to_string(), a.1, false)
    }
}

#[macro_export]
macro_rules! ph {
    ($($e:expr),* $(,)?) => { vec![$($crate::req::P::from($e)),*] };
}

pub type Phrase = Vec<P>;

// --------------------------------------------------------------------------- the request
pub struct Req<'a> {
    pub r: &'a mut Rng,
    pub chunks: Vec<(String, Option<u32>)>,
    pub n: u32,
}

impl<'a> Req<'a> {
    pub fn new(r: &'a mut Rng) -> Req<'a> {
        Req { r, chunks: Vec::new(), n: 0 }
    }
    pub fn key(&mut self) -> u32 {
        self.n += 1;
        self.n
    }
    pub fn f(&mut self) -> f64 {
        self.r.f()
    }
    pub fn chance(&mut self, p: f64) -> bool {
        self.r.f() < p
    }
    pub fn c(&mut self, xs: &[&'static str]) -> &'static str {
        xs[self.r.below(xs.len())]
    }
    /// pick from the concatenation of several lists without building it
    pub fn cat(&mut self, lists: &[&[&'static str]]) -> &'static str {
        let total: usize = lists.iter().map(|l| l.len()).sum();
        let mut i = self.r.below(total);
        for l in lists {
            if i < l.len() {
                return l[i];
            }
            i -= l.len();
        }
        unreachable!()
    }
    pub fn below(&mut self, n: usize) -> usize {
        self.r.below(n)
    }
    pub fn randint(&mut self, a: i64, b: i64) -> i64 {
        self.r.randint(a, b)
    }
    pub fn weighted(&mut self, w: &[f64]) -> usize {
        self.r.weighted(w)
    }
    pub fn pick<T: Clone>(&mut self, xs: &[T]) -> T {
        xs[self.r.below(xs.len())].clone()
    }

    // ---- values
    pub fn word(&mut self) -> &'static str {
        self.c(&WORDS)
    }
    pub fn count(&mut self) -> String {
        if self.chance(0.3) {
            return self.c(NUM_WORDS).to_string();
        }
        let ri = self.randint(2, 99);
        let opts = [1, 2, 3, 3, 5, 5, 10, 10, 10, 15, 20, 20, 25, 30, 50, 100, ri];
        self.pick(&opts).to_string()
    }
    pub fn size(&mut self) -> String {
        let ri = self.randint(1, 999);
        let opts: [String; 14] = ["1", "2", "5", "10", "20", "50", "100", "200", "250", "500", "700", "1.5", "2.5", ""].map(|s| s.to_string());
        let mut n = self.pick(&opts);
        if n.is_empty() {
            n = ri.to_string();
        }
        let style = self.f();
        if style < 0.45 {
            return format!("{n}{}", self.c(&["MB", "GB", "KB", "mb", "gb", "kb", "M", "G", "k", "K", "m", "g", "MiB", "GiB"]));
        }
        if style < 0.9 {
            let unit = self.c(&["mb", "gb", "kb", "MB", "GB", "megabytes", "gigabytes", "megs", "gigs", "kilobytes", "megabyte", "gigabyte", "gig", "meg"]);
            return format!("{n} {unit}");
        }
        self.c(&["a gig", "a gigabyte", "half a gig", "a megabyte", "one gb", "a meg", "one megabyte", "two gigs"]).to_string()
    }
    pub fn duration(&mut self) -> String {
        let style = self.f();
        let ri = self.randint(2, 48);
        let n = self.pick(&[1, 2, 3, 5, 7, 10, 14, 30, 60, 90, ri]);
        if style < 0.55 {
            let mut unit = self.c(&["days", "weeks", "months", "hours", "minutes", "years", "days", "days", "weeks", "hours"]).to_string();
            if n == 1 {
                unit.pop();
            }
            let num = if self.chance(0.2) { self.c(NUM_WORDS).to_string() } else { n.to_string() };
            return format!("{num} {unit}");
        }
        if style < 0.75 {
            return self
                .c(&["a day", "a week", "a month", "a year", "an hour", "one week", "one month", "a couple days", "a few days",
                    "a couple of weeks", "half an hour", "a fortnight"])
                .to_string();
        }
        format!("{n}{}", self.c(&["d", "h", "w", "min", "m", "y", "mo", "days", "hrs", "wks"]))
    }
    pub fn filename(&mut self, exts: Option<&[&'static str]>) -> String {
        if self.chance(0.17) {
            return self.c(SPECIAL_FILES).to_string();
        }
        let mut base = self.word().to_string();
        if self.chance(0.4) {
            base.push_str(self.c(&["_", "-", ""]));
            let w = self.cat(&[&WORDS, &["2023", "2024", "v2", "01", "final", "old"]]);
            base.push_str(w);
        }
        let e = match exts {
            Some(x) => self.c(x),
            None => self.c(&EXTS),
        };
        let mut name = format!("{base}.{e}");
        if self.chance(0.25) {
            name = format!("{}/{name}", self.c(REAL_DIRS));
        }
        name
    }
    /// (surface text for the span, extra words before, extra words after)
    pub fn folder(&mut self) -> (String, &'static str, &'static str) {
        let style = self.f();
        if style < 0.34 {
            return (self.c(REAL_DIRS).to_string(), "", "");
        }
        if style < 0.46 {
            let d = self.c(REAL_DIRS);
            let w = self.cat(&[&WORDS, &DIR_WORDS]);
            return (format!("{d}/{w}"), "", "");
        }
        if style < 0.62 {
            return (self.cat(&[&DIR_WORDS, &WORDS[..60]]).to_string(), "", "");
        }
        if style < 0.85 {
            let t = self.c(FOLDER_NAMES).to_string();
            let pre = self.c(&["the", "my", "the", ""]);
            let post = self.c(&["folder", "directory", "", "dir", "folder"]);
            return (t, pre, post);
        }
        let w = self.word().to_string();
        let pre = self.c(&["the", "my", "a", ""]);
        let post = self.c(&["folder", "directory", "dir"]);
        (w, pre, post)
    }

    // ---- building
    pub fn lit(&mut self, text: &str) {
        if !text.is_empty() {
            self.chunks.push((text.to_string(), None));
        }
    }
    pub fn arg_text(&mut self, text: &str, quote: bool) -> u32 {
        let k = self.key();
        self.chunks.push((if quote { format!("\"{text}\"") } else { text.to_string() }, Some(k)));
        k
    }
    pub fn folder_arg(&mut self, preps: &[&'static str]) -> u32 {
        let (text, pre, post) = self.folder();
        let p = self.c(preps);
        self.lit(p);
        self.lit(pre);
        let k = self.arg_text(&text, false);
        self.lit(post);
        k
    }
    pub fn emit(&mut self, phrase: &[P]) {
        for part in phrase {
            match part {
                P::L(s) => self.lit(s),
                P::A(t, k, q) => self.chunks.push((if *q { format!("\"{t}\"") } else { t.clone() }, Some(*k))),
            }
        }
    }
    /// The clauses that follow the noun, sometimes joined with "and", "but", a comma...
    pub fn tail(&mut self, bits: &[Phrase]) {
        for (i, t) in bits.iter().enumerate() {
            if i > 0 && self.f() < 0.3 {
                let c = self.c(CONNECT);
                self.lit(c);
            }
            self.emit(t);
        }
    }
}

pub fn pkg_list(r: &mut Req, names: &[String]) -> String {
    if names.len() == 1 {
        return names[0].clone();
    }
    let style = r.f();
    if style < 0.4 {
        return names.join(" ");
    }
    let last = names.last().unwrap();
    if style < 0.75 {
        return if names.len() == 2 || r.f() < 0.5 {
            format!("{} and {last}", names[..names.len() - 1].join(" "))
        } else {
            format!("{} and {last}", names[..names.len() - 1].join(", "))
        };
    }
    names.join(", ")
}

fn upper_first(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

/// A random value of the given kind, as the user might type it -> (text, quote it?, words before, words after)
pub fn val(q: &mut Req, kind: &str) -> (String, bool, &'static str, &'static str) {
    let s = |x: &str| (x.to_string(), false, "", "");
    match kind {
        "path" => {
            let (t, pre, post) = q.folder();
            (t, false, pre, post)
        }
        "file" => (q.filename(None), false, "", ""),
        "files" => {
            let n = q.pick(&[1, 1, 1, 2, 2, 3]);
            let names: Vec<String> = (0..n).map(|_| q.filename(None)).collect();
            (if names.len() == 1 { names[0].clone() } else { pkg_list(q, &names) }, false, "", "")
        }
        "word" => {
            let w = q.word();
            let suf = q.c(&["", "", "", "-fix", "2", "_x"]);
            (format!("{w}{suf}"), false, "", "")
        }
        "words" => {
            if q.f() < 0.8 {
                let k = q.pick(&[1, 1, 1, 1, 2, 2, 3, 4]);
                let names: Vec<String> = q.r.sample(&PKGS, k).into_iter().map(|s| s.to_string()).collect();
                (pkg_list(q, &names), false, "", "")
            } else {
                s(q.word())
            }
        }
        "pkg" => {
            if q.f() < 0.75 {
                s(q.c(&PKGS))
            } else {
                s(q.word())
            }
        }
        "text" => {
            let t = if q.f() < 0.6 { q.c(PHRASES) } else { q.word() };
            let quote = t.contains(' ') || q.f() < 0.2;
            (t.to_string(), quote, "", "")
        }
        "sentence" => {
            let t = q.cat(&[SENTENCES, &PHRASES[..10]]);
            let quote = t.contains(' ') || q.f() < 0.6;
            (t.to_string(), quote, "", "")
        }
        "msg" => {
            let t = if q.f() < 0.7 {
                q.c(MSGS).to_string()
            } else {
                let k = q.pick(&[1, 2, 3]);
                q.r.sample(&WORDS, k).join(" ")
            };
            let quote = q.f() < 0.55;
            (t, quote, "", "")
        }
        "count" => (q.count(), false, "", ""),
        "smallcount" => {
            if q.f() < 0.8 {
                (q.pick(&[1, 2, 2, 3, 3, 4, 5]).to_string(), false, "", "")
            } else {
                s(q.c(&["one", "two", "three", "four"]))
            }
        }
        "size" => (q.size(), false, "", ""),
        "dur" => (q.duration(), false, "", ""),
        "ext" => {
            let e = if q.f() < 0.3 { q.c(EXT_NAMES) } else { q.c(&EXTS) };
            if EXTS.contains(&e) {
                let dot = format!(".{e}");
                let opts = [e.to_string(), e.to_string(), dot];
                (q.pick(&opts), false, "", "")
            } else {
                s(e)
            }
        }
        "unit" => {
            let mut u = if q.f() < 0.75 {
                q.c(&UNITS).to_string()
            } else {
                let w = q.word();
                format!("{w}{}", q.c(&["", "d", "-agent", "-server"]))
            };
            if q.f() < 0.06 {
                u.push_str(".service");
            }
            (u, false, "", "")
        }
        "host" => {
            if q.f() < 0.8 {
                s(q.c(HOSTS))
            } else {
                let w = q.word();
                (format!("{w}{}", q.c(&[".com", ".local", "-01", ".lan", ".org", ""])), false, "", "")
            }
        }
        "url" => {
            if q.f() < 0.75 {
                return s(q.c(URLS));
            }
            let a = q.word();
            let tld = q.c(&["com", "org", "io", "net", "dev"]);
            let b = q.word();
            let suf = q.c(&["", ".zip", ".git", ".tar.gz", "/latest", ".json", ".pdf"]);
            (format!("https://{a}.{tld}/{b}{suf}"), false, "", "")
        }
        "port" => {
            if q.f() < 0.75 {
                s(q.c(PORTS))
            } else {
                (q.randint(1024, 65000).to_string(), false, "", "")
            }
        }
        "pct" => {
            if q.f() < 0.7 {
                s(q.c(PCTS))
            } else {
                let n = q.randint(1, 100);
                (format!("{n}{}", q.c(&["", "%", " percent"])), false, "", "")
            }
        }
        "envname" => {
            if q.f() < 0.75 {
                s(q.c(&ENVNAMES))
            } else {
                let w = q.word();
                let extra = if q.f() < 0.5 { String::new() } else { format!("_{}", q.word()) };
                (format!("{w}{extra}").to_uppercase(), false, "", "")
            }
        }
        "cmdname" => {
            if q.f() < 0.7 {
                s(q.c(&CMDNAMES))
            } else {
                s(q.c(&PKGS))
            }
        }
        "container" => {
            if q.f() < 0.75 {
                s(q.c(CONTAINERS))
            } else {
                let w = q.word();
                (format!("{w}{}", q.c(&["", "-1", "_app", "-db"])), false, "", "")
            }
        }
        "image" => {
            if q.f() < 0.75 {
                s(q.c(IMAGES))
            } else {
                let w = q.word();
                (format!("{w}{}", q.c(&["", ":latest", ":1.0", ":dev"])), false, "", "")
            }
        }
        "script" => {
            if q.f() < 0.8 {
                s(q.c(SCRIPTS))
            } else {
                s(q.word())
            }
        }
        "ssid" => {
            let t = if q.f() < 0.7 {
                q.c(SSIDS).to_string()
            } else {
                let w = upper_first(q.word());
                format!("{w}{}", q.c(&["", "-5G", " Guest", "WiFi", "_2.4"]))
            };
            let quote = t.contains(' ');
            (t, quote, "", "")
        }
        "branch" => {
            if q.f() < 0.7 {
                s(q.c(BRANCHES))
            } else {
                let w = q.word();
                (format!("{w}{}", q.c(&["", "-fix", "-wip", "/v2"])), false, "", "")
            }
        }
        "expr" => {
            if q.f() < 0.7 {
                s(q.c(EXPRS))
            } else {
                let a = q.randint(2, 999);
                let op = q.c(&["*", "+", "-", "/", "times", "plus", "minus"]);
                let b = q.randint(2, 99);
                (format!("{a} {op} {b}"), false, "", "")
            }
        }
        "date" => s(q.c(DATES)),
        "owner" => {
            let t = if q.f() < 0.8 { q.c(OWNERS) } else { q.word() };
            let pre = q.c(&["", "", "", "user", "the user"]);
            (t.to_string(), false, pre, "")
        }
        "mode" => s(q.c(MODE_WORDS)),
        "remote" => {
            if q.f() < 0.8 {
                s(q.c(REMOTES))
            } else {
                let a = q.word();
                let b = q.word();
                (format!("{a}@{b}:{}", q.c(&["", "~/", "/srv/", "/tmp/x"])), false, "", "")
            }
        }
        "affix" => {
            let t = q.c(AFFIXES);
            let quote = q.f() < 0.15;
            (t.to_string(), quote, "", "")
        }
        "tag" => s(q.c(TAGS)),
        "alias" => s(q.c(ALIASES)),
        "aliascmd" => {
            let t = q.c(ALIAS_CMDS);
            let quote = t.contains(' ') || q.f() < 0.5;
            (t.to_string(), quote, "", "")
        }
        "varvalue" => {
            let t = q.c(VAR_VALUES);
            let quote = q.f() < 0.15;
            (t.to_string(), quote, "", "")
        }
        "script_file" => {
            let f = q.filename(Some(&["sh", "py", "js", "rb", "pl", "run", "bin"]));
            if q.f() < 0.3 {
                (format!("./{}", f.rsplit('/').next().unwrap_or(&f)), false, "", "")
            } else {
                (f, false, "", "")
            }
        }
        "archive" => (q.filename(Some(&["zip", "tar.gz", "tgz", "tar", "7z", "rar", "tar.xz", "gz", "tar.bz2"])), false, "", ""),
        "textfile" => (q.filename(Some(&["log", "txt", "md", "conf", "json", "py", "csv", "yaml", "ini", "sh", "toml", "cfg", "env"])), false, "", ""),
        "target" => {
            let roll = q.f();
            if roll < 0.55 {
                return (q.filename(Some(&["pdf", "png", "jpg", "html", "mp4", "docx", "txt", "svg", "xlsx", "mkv", "odt"])), false, "", "");
            }
            if roll < 0.75 {
                return s(q.c(&["firefox", "chromium", "gimp", "vlc", "code", "dolphin", "steam", "libreoffice", "thunderbird", "spotify",
                    "discord", "kate", "konsole", "obs"]));
            }
            if roll < 0.87 {
                return s(q.c(URLS));
            }
            let (t, pre, post) = q.folder();
            (t, false, pre, post)
        }
        "bindir" => {
            let w = q.word();
            let opts = ["~/bin".to_string(), "~/.local/bin".into(), "/opt/tool/bin".into(), "~/go/bin".into(), "~/.cargo/bin".into(),
                "/usr/local/go/bin".into(), "./node_modules/.bin".into(), "~/scripts".into(), format!("/opt/{w}/bin")];
            (q.pick(&opts), false, "", "")
        }
        "venv" => {
            let a = q.word();
            let b = q.word();
            let opts = [".venv".to_string(), "venv".into(), "env".into(), format!("~/envs/{a}"), ".env".into(), "myenv".into(),
                format!("~/.virtualenvs/{b}")];
            (q.pick(&opts), false, "", "")
        }
        "binfile" => {
            let c1 = q.c(&CMDNAMES);
            let w1 = q.word();
            let c2 = q.c(&CMDNAMES);
            let w2 = q.word();
            let w3 = q.word();
            let opts = [format!("/usr/bin/{c1}"), format!("/etc/{w1}.conf"), c2.to_string(), format!("/usr/lib/lib{w2}.so"), format!("/usr/bin/{w3}")];
            (q.pick(&opts), false, "", "")
        }
        other => panic!("no value kind {other}"),
    }
}

/// Emit a value of this kind into the request and return its key.
pub fn put(q: &mut Req, kind: &str) -> u32 {
    let (text, quote, pre, post) = val(q, kind);
    q.lit(pre);
    let k = q.arg_text(&text, quote);
    q.lit(post);
    k
}

/// Write a phrase; "{}" marks where a fresh value of `kind` goes. Returns that value's key (or None).
pub fn fill(q: &mut Req, phrase: &str, kind: &str) -> Option<V> {
    match phrase.split_once("{}") {
        None => {
            q.lit(phrase);
            None
        }
        Some((a, b)) => {
            q.lit(a);
            let k = put(q, kind);
            q.lit(b);
            Some(V::K(k))
        }
    }
}

/// Filters in the fixed order the language uses, each negated one preceded by "not".
pub fn filter_stages(stages: &[(&'static str, Args)], negated: &[&'static str]) -> Pipe {
    let mut out = Vec::new();
    for &n in &lang().filter_order {
        if let Some((_, a)) = stages.iter().find(|(m, _)| *m == n) {
            if negated.contains(&n) {
                out.push(("not", Vec::new()));
            }
            out.push((n, a.clone()));
        }
    }
    out
}

// --------------------------------------------------------------------------- into training tokens
pub struct Example {
    pub words: Vec<String>,
    pub out: Vec<u16>,
    pub n_req: usize,
}

/// Resolve argument keys to word positions; -> (words: request words then context words, output tokens).
pub fn finish(chunks: &[(String, Option<u32>)], program: &[Pipe], ctx: &[String]) -> Result<Example, String> {
    let lg = lang();
    let mut words: Vec<String> = Vec::new();
    let mut spans: HashMap<u32, (usize, usize)> = HashMap::new();
    for (text, key) in chunks {
        let ws = tok::words(text);
        if let Some(k) = key {
            spans.insert(*k, (words.len(), (words.len() + ws.len()).wrapping_sub(1)));
        }
        words.extend(ws);
    }
    let n = words.len();
    let mut out: Vec<u16> = Vec::new();
    for (i, pipe) in program.iter().enumerate() {
        if i > 0 {
            out.push(lg.sep);
        }
        for (name, args) in pipe {
            let ix = *lg.stage_ix.get(*name).ok_or_else(|| format!("no stage {name}"))?;
            out.push(lg.l.tok[*name] as u16);
            for slot in &lg.l.stages[ix].slots {
                let v = arg(args, &slot.name);
                match &slot.kind {
                    SlotKind::Span(_) => match v {
                        None => out.push(lg.none),
                        Some(V::K(k)) => {
                            let (a, b) = *spans.get(&k).ok_or_else(|| format!("{name}.{}: key {k} not in the request", slot.name))?;
                            if b == usize::MAX || b < a {
                                return Err(format!("{name}.{}: empty argument", slot.name));
                            }
                            out.push((lg.nv + a) as u16);
                            out.push((lg.nv + b) as u16);
                        }
                        Some(V::Ctx(c)) => {
                            out.push((lg.nv + n + c) as u16);
                            out.push((lg.nv + n + c) as u16);
                        }
                        Some(V::E(e)) => return Err(format!("{name}.{}: enum value {e} in a span slot", slot.name)),
                    },
                    SlotKind::Enum(_) => match v {
                        Some(V::E(e)) => out.push(*lg.l.tok.get(e).ok_or_else(|| format!("no token {e}"))? as u16),
                        _ => return Err(format!("{name}.{}: enum slot without a value", slot.name)),
                    },
                }
            }
        }
    }
    out.push(lg.end);
    words.extend(ctx.iter().cloned());
    Ok(Example { words, out, n_req: n })
}

/// Output tokens back to a program with the copied words as strings (ir.delinearise).
pub fn delinearise(toks: &[u16], words: &[String]) -> DProg {
    let lg = lang();
    let (mut prog, mut pipe): (DProg, Vec<DStage>) = (Vec::new(), Vec::new());
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        i += 1;
        if t == lg.end || t == lg.pad {
            break;
        }
        if t == lg.sep {
            prog.push(std::mem::take(&mut pipe));
            continue;
        }
        let name = lg.l.vocab[t as usize].as_str();
        let sd = &lg.l.stages[lg.stage_ix[name]];
        let mut args = Vec::new();
        for slot in &sd.slots {
            let v = toks[i];
            i += 1;
            let dv = match slot.kind {
                SlotKind::Span(_) => {
                    if v == lg.none {
                        DV::Null
                    } else {
                        let e = toks[i];
                        i += 1;
                        DV::Str(words[v as usize - lg.nv..=e as usize - lg.nv].join(" "))
                    }
                }
                SlotKind::Enum(_) => DV::Enum(lg.l.vocab[v as usize].as_str()),
            };
            args.push((slot.name.as_str(), dv));
        }
        pipe.push((sd.name.as_str(), args));
    }
    if !pipe.is_empty() {
        prog.push(pipe);
    }
    prog
}

/// A program as context for the next request (ir.context_words): words, and where each span value is.
pub fn context_words(prog: &DProg) -> (Vec<String>, HashMap<(usize, usize, &'static str), usize>) {
    let mut words = Vec::new();
    let mut place = HashMap::new();
    for (i, pipe) in prog.iter().enumerate() {
        if i > 0 {
            words.push("@;".to_string());
        }
        for (j, (name, args)) in pipe.iter().enumerate() {
            words.push(format!("@{name}"));
            for (slot, v) in args {
                match v {
                    DV::Enum(e) => words.push(format!("@{e}")),
                    DV::Null => words.push("@none".to_string()),
                    DV::Str(s) => {
                        place.insert((i, j, *slot), words.len());
                        words.push(s.clone());
                    }
                }
            }
        }
    }
    (words, place)
}

/// Positions of the input words that the program copied as arguments (ir.spans_used).
pub fn spans_used(toks: &[u16]) -> Vec<usize> {
    let lg = lang();
    let mut used = Vec::new();
    let mut i = 0;
    while i < toks.len() {
        let t = toks[i];
        i += 1;
        if t == lg.end || t == lg.pad {
            break;
        }
        if t == lg.sep || t as usize >= lg.nv {
            continue;
        }
        let sd = &lg.l.stages[lg.stage_ix[lg.l.vocab[t as usize].as_str()]];
        for slot in &sd.slots {
            let v = toks[i];
            i += 1;
            if matches!(slot.kind, SlotKind::Span(_)) && v != lg.none {
                let e = toks[i];
                i += 1;
                used.extend(v as usize - lg.nv..=e as usize - lg.nv);
            }
        }
    }
    used
}
