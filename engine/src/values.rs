//! Turning copied words into exact values: sizes, ages, counts, folders, file types. A port of kural/values.py;
//! each regular expression there is reproduced here by hand.

use crate::py::{self, boundary, has_word, is_word, quote, round_even};

const NUMBER_WORDS: &[(&str, f64)] = &[
    ("a", 1.0), ("an", 1.0), ("one", 1.0), ("single", 1.0), ("two", 2.0), ("couple", 2.0), ("three", 3.0), ("few", 3.0),
    ("four", 4.0), ("five", 5.0), ("six", 6.0), ("seven", 7.0), ("eight", 8.0), ("nine", 9.0), ("ten", 10.0),
    ("eleven", 11.0), ("twelve", 12.0), ("dozen", 12.0), ("fifteen", 15.0), ("twenty", 20.0), ("thirty", 30.0),
    ("forty", 40.0), ("fifty", 50.0), ("sixty", 60.0), ("hundred", 100.0), ("thousand", 1000.0), ("half", 0.5),
    ("first", 1.0), ("second", 2.0), ("third", 3.0), ("fourth", 4.0), ("fifth", 5.0), ("sixth", 6.0), ("seventh", 7.0),
    ("eighth", 8.0), ("ninth", 9.0), ("tenth", 10.0),
];

pub const BIG: &str = "100M";
pub const SMALL: &str = "100k";
pub const OLD_DAYS: i64 = 90;
pub const NEW_DAYS: i64 = 7;
pub const SHOW_TOP: i64 = 20;

/// Python int(x) for a float: truncation toward zero.
fn pyint(x: f64) -> i64 {
    x.trunc() as i64
}

/// The tokens of re.findall(r"\d+(?:\.\d+)?|[a-z]+", t)
fn num_tokens(t: &str) -> Vec<String> {
    let cs: Vec<char> = t.chars().collect();
    let (mut out, mut i) = (Vec::new(), 0);
    while i < cs.len() {
        if cs[i].is_ascii_digit() {
            let s = i;
            while i < cs.len() && cs[i].is_ascii_digit() {
                i += 1;
            }
            if i + 1 < cs.len() && cs[i] == '.' && cs[i + 1].is_ascii_digit() {
                i += 1;
                while i < cs.len() && cs[i].is_ascii_digit() {
                    i += 1;
                }
            }
            out.push(cs[s..i].iter().collect());
        } else if cs[i].is_ascii_lowercase() {
            let s = i;
            while i < cs.len() && cs[i].is_ascii_lowercase() {
                i += 1;
            }
            out.push(cs[s..i].iter().collect());
        } else {
            i += 1;
        }
    }
    out
}

/// First number in the text, or None.
pub fn number(text: &str) -> Option<f64> {
    for w in num_tokens(&text.to_lowercase()) {
        if w.starts_with(|c: char| c.is_ascii_digit()) {
            return w.parse().ok();
        }
        if let Some(&(_, v)) = NUMBER_WORDS.iter().find(|(k, _)| *k == w) {
            return Some(v);
        }
    }
    None
}

pub fn count(text: Option<&str>, default: i64) -> i64 {
    match number(text.unwrap_or("")) {
        Some(n) if n != 0.0 => pyint(n).max(1),
        _ => default,
    }
}

const SIZE_ALTS: &[&str] = &["tb", "terabytes", "terabyte", "gb", "gigs", "gig", "gigabytes", "gigabyte", "gib", "mb", "megs", "meg",
    "megabytes", "megabyte", "mib", "kb", "kilobytes", "kilobyte", "kib", "bytes", "byte", "t", "g", "m", "k", "b"];

/// '100MB', '2 gigs', '1.5 G', 'half a gig', '500k' -> bytes. Bare numbers are megabytes.
pub fn size_bytes(text: &str) -> i64 {
    let t = text.to_lowercase().replace(',', "");
    let n = number(&t).unwrap_or(1.0);
    let cs: Vec<char> = t.chars().collect();
    let unit_at = |p: usize| -> Option<char> {
        for alt in SIZE_ALTS {
            let a: Vec<char> = alt.chars().collect();
            if p + a.len() <= cs.len() && cs[p..p + a.len()] == a[..] && boundary(&cs, p + a.len()) {
                return Some(a[0]);
            }
        }
        None
    };
    let mut mult = 1024f64.powi(2);
    for p in 0..cs.len() {
        let u = if cs[p].is_ascii_digit() { unit_at(p + 1) } else { None }.or_else(|| if boundary(&cs, p) { unit_at(p) } else { None });
        if let Some(u) = u {
            mult = match u {
                't' => 1024f64.powi(4),
                'g' => 1024f64.powi(3),
                'm' => 1024f64.powi(2),
                'k' => 1024.0,
                _ => 1.0,
            };
            break;
        }
    }
    pyint(n * mult)
}

pub fn find_size(text: &str) -> String {
    let b = size_bytes(text);
    for (unit, val) in [("G", 1i64 << 30), ("M", 1 << 20), ("k", 1 << 10)] {
        if b >= val && b % val == 0 {
            return format!("{}{unit}", b / val);
        }
    }
    for (unit, val) in [("M", 1i64 << 20), ("k", 1 << 10)] {
        if b >= val {
            return format!("{}{unit}", round_even(b as f64 / val as f64) as i64);
        }
    }
    format!("{b}c")
}

fn dur_unit(u: &str) -> Option<f64> {
    Some(match u {
        "s" | "sec" | "secs" | "second" | "seconds" => 1.0 / 60.0,
        "m" | "min" | "mins" | "minute" | "minutes" => 1.0,
        "h" | "hr" | "hrs" | "hour" | "hours" => 60.0,
        "d" | "day" | "days" => 1440.0,
        "w" | "wk" | "wks" | "week" | "weeks" => 10080.0,
        "mo" | "month" | "months" => 43200.0,
        "y" | "yr" | "yrs" | "year" | "years" => 525600.0,
        _ => return None,
    })
}

/// '30 days', 'a week', '2h', 'last month', 'today' -> minutes
pub fn minutes(text: &str) -> i64 {
    let t = text.to_lowercase();
    for (w, v) in [("today", 1440), ("yesterday", 2880), ("tonight", 720), ("fortnight", 20160), ("overnight", 720)] {
        if has_word(&t, w) {
            return v;
        }
    }
    let stripped = py::sub_alts(&t, &[("this", true, true), ("last", true, true), ("past", true, true), ("the", true, true), ("previous", true, true)], "");
    let n = number(&stripped);
    // re.findall(r"\d+(?:\.\d+)?\s*([a-z]+)|\b([a-z]+)\b", t)
    let cs: Vec<char> = t.chars().collect();
    let mut unit = None;
    let mut i = 0;
    while i < cs.len() && unit.is_none() {
        let mut matched: Option<(String, usize)> = None;
        if cs[i].is_ascii_digit() {
            let mut j = i;
            while j < cs.len() && cs[j].is_ascii_digit() {
                j += 1;
            }
            if j + 1 < cs.len() && cs[j] == '.' && cs[j + 1].is_ascii_digit() {
                j += 1;
                while j < cs.len() && cs[j].is_ascii_digit() {
                    j += 1;
                }
            }
            while j < cs.len() && cs[j].is_whitespace() {
                j += 1;
            }
            let s = j;
            while j < cs.len() && cs[j].is_ascii_lowercase() {
                j += 1;
            }
            if j > s {
                matched = Some((cs[s..j].iter().collect(), j));
            }
        }
        if matched.is_none() && boundary(&cs, i) && i < cs.len() && cs[i].is_ascii_lowercase() {
            let mut j = i;
            while j < cs.len() && cs[j].is_ascii_lowercase() {
                j += 1;
            }
            if boundary(&cs, j) {
                matched = Some((cs[i..j].iter().collect(), j));
            }
        }
        match matched {
            Some((u, end)) => {
                unit = dur_unit(&u);
                i = end;
            }
            None => i += 1,
        }
    }
    let unit = unit.unwrap_or(1440.0);
    let n = n.unwrap_or(1.0);
    (round_even(n * unit) as i64).max(1)
}

pub fn find_age(text: &str, newer: bool) -> String {
    let m = minutes(text);
    let sign = if newer { "-" } else { "+" };
    if m % 1440 == 0 {
        let days = m / 1440;
        return format!("-mtime {sign}{}", if newer { days } else { (days - 1).max(0) });
    }
    format!("-mmin {sign}{m}")
}

const FOLDERS: &[(&str, &str)] = &[
    ("downloads", "~/Downloads"), ("download", "~/Downloads"), ("documents", "~/Documents"), ("docs", "~/Documents"),
    ("desktop", "~/Desktop"), ("pictures", "~/Pictures"), ("photos", "~/Pictures"), ("music", "~/Music"),
    ("videos", "~/Videos"), ("movies", "~/Videos"), ("home", "~"), ("here", "."), ("current", "."), ("this", "."), ("cwd", "."),
    ("root", "/"), ("tmp", "/tmp"), ("temp", "/tmp"), ("trash", "~/.local/share/Trash/files"), ("parent", ".."),
    ("config", "~/.config"), ("projects", "~/Projects"),
];
const FILLER: &[&str] = &["the", "my", "our", "a", "folder", "directory", "dir", "folders", "directories", "working", "system", "one"];

/// 'the downloads folder', '~/src', 'here' -> a path (not yet quoted). isdir says whether a folder exists where the command runs.
pub fn path(text: &str, isdir: Option<&dyn Fn(&str) -> bool>) -> String {
    let words: Vec<&str> = text.split_whitespace().collect();
    for w in &words {
        if w.contains('/') || ((w.starts_with('~') || w.starts_with('.')) && w.chars().count() > 1) {
            return w.to_string();
        }
    }
    let core: Vec<&str> = words.iter().copied().filter(|w| !FILLER.contains(&w.to_lowercase().as_str())).collect();
    if core.len() == 1 {
        if let Some(&(_, f)) = FOLDERS.iter().find(|(k, _)| *k == core[0].to_lowercase()) {
            if words.len() == 1 {
                if let Some(isdir) = isdir {
                    if isdir(core[0]) {
                        return core[0].to_string();
                    }
                }
            }
            return f.to_string();
        }
    }
    if core.is_empty() {
        return ".".into();
    }
    core.join(" ")
}

/// Shell-quote a path but keep a leading ~ working.
pub fn quote_path(p: &str) -> String {
    if p == "~" {
        return "~".into();
    }
    if let Some(rest) = p.strip_prefix("~/") {
        return if rest.is_empty() { "~/".into() } else { format!("~/{}", quote(rest)) };
    }
    quote(p)
}

fn ext_group(w: &str) -> Option<&'static [&'static str]> {
    Some(match w {
        "image" | "picture" => &["jpg", "jpeg", "png", "gif", "webp"],
        "photo" => &["jpg", "jpeg", "png", "heic"],
        "video" | "movie" => &["mp4", "mkv", "mov", "avi", "webm"],
        "audio" | "music" | "song" => &["mp3", "flac", "wav", "ogg", "m4a"],
        "archive" => &["zip", "tar", "gz", "xz", "7z", "rar", "tgz"],
        "yaml" => &["yaml", "yml"],
        "jpeg" | "jpg" => &["jpg", "jpeg"],
        _ => return None,
    })
}

fn ext_name(w: &str) -> Option<&'static str> {
    Some(match w {
        "python" => "py", "javascript" => "js", "typescript" => "ts", "rust" => "rs", "golang" => "go", "ruby" => "rb",
        "markdown" => "md", "text" => "txt", "shell" | "bash" | "script" => "sh", "c++" => "cpp", "header" => "h",
        "backup" => "bak", "temp" | "temporary" => "tmp", "word" => "docx", "excel" => "xlsx", "powerpoint" => "pptx",
        "kotlin" => "kt", "perl" => "pl", "config" => "conf", "notebook" | "jupyter" => "ipynb",
        _ => return None,
    })
}

/// 'python', 'pdfs', '.log', 'images', '*.txt' -> extensions without dots
pub fn extensions(text: &str) -> Vec<String> {
    let low = text.to_lowercase();
    let w: Vec<&str> = low.split_whitespace().filter(|x| !["files", "file", "the", "all"].contains(x)).collect();
    let base = if let Some(f) = w.first() { f.to_string() } else { low.clone() };
    let mut w = base.trim_start_matches('*').trim_start_matches('.').to_string();
    let without_s = if w.ends_with('s') { w[..w.len() - 1].to_string() } else { w.clone() };
    for cand in [w.clone(), without_s] {
        if let Some(g) = ext_group(&cand) {
            return g.iter().map(|s| s.to_string()).collect();
        }
        if let Some(e) = ext_name(&cand) {
            return vec![e.to_string()];
        }
    }
    if w.ends_with('s') && w.chars().count() > 3 && py::isalnum(&w[..w.len() - 1]) {
        w.pop();
    }
    vec![w]
}

/// Like extensions(), but also accepts several: 'pdf or docx', 'jpg, png and gif'.
pub fn extensions_any(text: &str) -> Vec<String> {
    let t = text.trim();
    let cs: Vec<char> = t.chars().collect();
    let mut parts: Vec<String> = Vec::new();
    let (mut start, mut i) = (0, 0);
    while i < cs.len() {
        let mut j = i;
        while j < cs.len() && cs[j].is_whitespace() {
            j += 1;
        }
        let mut end = None;
        if j < cs.len() && (cs[j] == ',' || cs[j] == '/') {
            end = Some(j + 1);
        } else {
            for w in ["or", "and"] {
                let n = w.len();
                if j + n <= cs.len() && cs[j..j + n].iter().collect::<String>() == w && boundary(&cs, j) && boundary(&cs, j + n) {
                    end = Some(j + n);
                    break;
                }
            }
        }
        if let Some(mut k) = end {
            while k < cs.len() && cs[k].is_whitespace() {
                k += 1;
            }
            parts.push(cs[start..i].iter().collect());
            start = k;
            i = k.max(i + 1);
            continue;
        }
        i += 1;
    }
    parts.push(cs[start..].iter().collect());
    let parts: Vec<String> = parts.into_iter().filter(|p| !p.trim().is_empty()).collect();
    let mut out: Vec<String> = Vec::new();
    let list = if parts.is_empty() { vec![text.to_string()] } else { parts };
    for p in list {
        for e in extensions(&p) {
            if !out.contains(&e) {
                out.push(e);
            }
        }
    }
    out
}

/// '50%', 'half', 'max' -> 0..100
pub fn percent(text: &str) -> i64 {
    let t = text.to_lowercase();
    for (w, v) in [("half", 50), ("full", 100), ("max", 100), ("maximum", 100), ("quarter", 25), ("zero", 0), ("min", 0), ("minimum", 0)] {
        if has_word(&t, w) {
            return v;
        }
    }
    match number(&t.replace('%', " ")) {
        Some(n) => pyint(n).clamp(0, 100),
        None => 50,
    }
}

pub fn port(text: &str) -> String {
    let d: String = text.chars().skip_while(|c| !c.is_ascii_digit()).take_while(|c| c.is_ascii_digit()).collect();
    if d.is_empty() { "80".into() } else { d }
}

/// '10 minutes', '90s' -> seconds
pub fn seconds(text: &str) -> i64 {
    let t = text.to_lowercase();
    let s = t.trim();
    let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
    if !digits.is_empty() {
        let rest = s[digits.len()..].trim_start();
        if ["s", "sec", "secs", "second", "seconds"].contains(&rest) {
            return digits.parse().unwrap_or(0);
        }
    }
    minutes(&t) * 60
}

/// A date as the user wrote it, tidied for GNU date.
pub fn date(text: &str) -> String {
    let t = py::sub_alts(&text.to_lowercase(), &[("the", true, true), ("on", true, true), ("of", true, true)], " ");
    // re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", t)
    let cs: Vec<char> = t.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < cs.len() {
        out.push(cs[i]);
        if cs[i].is_ascii_digit() && i + 2 < cs.len() + 0 && i + 3 <= cs.len() {
            let suf: String = cs[i + 1..i + 3].iter().collect();
            if ["st", "nd", "rd", "th"].contains(&suf.as_str()) && boundary(&cs, i + 3) {
                i += 3;
                continue;
            }
        }
        i += 1;
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A pattern piece list: literal text where '?' after a '.' means "any one character, or none".
fn pat_search(t: &[char], pat: &str) -> bool {
    // pat uses only literal chars, ".?" (optional any char) and "\+"
    #[derive(Clone)]
    enum P {
        Lit(char),
        Any,
    }
    let mut ps = Vec::new();
    let pc: Vec<char> = pat.chars().collect();
    let mut i = 0;
    while i < pc.len() {
        if pc[i] == '.' && i + 1 < pc.len() && pc[i + 1] == '?' {
            ps.push(P::Any);
            i += 2;
        } else if pc[i] == '\\' && i + 1 < pc.len() {
            ps.push(P::Lit(pc[i + 1]));
            i += 2;
        } else {
            ps.push(P::Lit(pc[i]));
            i += 1;
        }
    }
    fn m(t: &[char], ti: usize, ps: &[P]) -> bool {
        match ps.first() {
            None => true,
            Some(P::Lit(c)) => ti < t.len() && t[ti] == *c && m(t, ti + 1, &ps[1..]),
            Some(P::Any) => (ti < t.len() && t[ti] != '\n' && m(t, ti + 1, &ps[1..])) || m(t, ti, &ps[1..]),
        }
    }
    (0..=t.len()).any(|s| m(t, s, &ps))
}

const MODES: &[(&[&str], &str)] = &[
    (&["public", "everyone", "everybody", "world.?readable", "readable by all", "all users"], "a+rX"),
    (&["private", "only me", "only i", "just me", "mine only", "owner.?only"], "go-rwx"),
    (&["read.?only", "readonly", "unwritable", "write.?protect"], "a-w"),
    (&["writable", "writeable", "editable"], "u+w"),
    (&["not exec", "non.?exec", "-x"], "-x"),
    (&["exec", "runnable", "\\+x"], "+x"),
];

/// '755', 'u+x', 'executable', 'read only', 'private' -> chmod argument
pub fn mode(text: &str) -> String {
    let t = text.trim().to_lowercase();
    let cs: Vec<char> = t.chars().collect();
    let octal = |c: char| ('0'..='7').contains(&c);
    for i in 0..cs.len() {
        if !boundary(&cs, i) {
            continue;
        }
        for len in [4, 3] {
            if i + len <= cs.len() && cs[i..i + len].iter().all(|&c| octal(c)) && boundary(&cs, i + len) {
                return cs[i..i + len].iter().collect();
            }
        }
    }
    // \b[ugoa]*[+=-][rwxXst]+(?:,[ugoa]*[+=-][rwxXst]+)*
    let clause = |mut j: usize| -> Option<usize> {
        while j < cs.len() && "ugoa".contains(cs[j]) {
            j += 1;
        }
        if j < cs.len() && "+=-".contains(cs[j]) {
            j += 1;
        } else {
            return None;
        }
        let s = j;
        while j < cs.len() && "rwxXst".contains(cs[j]) {
            j += 1;
        }
        if j > s { Some(j) } else { None }
    };
    for i in 0..cs.len() {
        if !boundary(&cs, i) {
            continue;
        }
        if let Some(mut end) = clause(i) {
            while end < cs.len() && cs[end] == ',' {
                match clause(end + 1) {
                    Some(e) => end = e,
                    None => break,
                }
            }
            return cs[i..end].iter().collect();
        }
    }
    for (pats, val) in MODES {
        if pats.iter().any(|p| pat_search(&cs, p)) {
            return val.to_string();
        }
    }
    "+x".into()
}

pub fn owner(text: &str) -> String {
    let w: Vec<&str> = text.split_whitespace().filter(|x| !["the", "user", "owner", "account"].contains(&x.to_lowercase().as_str())).collect();
    let name = if let Some(f) = w.first() { f.to_string() } else { text.trim().to_string() };
    if ["me", "myself", "my", "mine", "i"].contains(&name.to_lowercase().as_str()) {
        "\"$USER\"".into()
    } else {
        quote(&name)
    }
}

/// 'a.txt and b.txt, c.txt' -> names
pub fn names(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.trim_matches(','))
        .zip(text.split_whitespace())
        .filter(|(s, w)| !s.is_empty() && !["and", "&", "plus", "also", "the", "file", "files"].contains(&w.to_lowercase().as_str()))
        .map(|(s, _)| s.to_string())
        .collect()
}

/// '12 times 8', '15 percent of 240' -> an arithmetic expression
pub fn expr(text: &str) -> String {
    let mut t = text.to_lowercase().replace(',', "");
    t = py::sub_alts(&t, &[("to the power of", true, true)], "^");
    t = py::sub_alts(&t, &[("percent of", true, true), ("% of", false, true)], "/100*");
    t = py::sub_alts(&t, &[("divided by", true, true), ("over", true, true)], "/");
    // \bmultiplied by\b | \btimes\b | (?<=\d)\s*x\s*(?=\d)
    {
        let cs: Vec<char> = t.chars().collect();
        let mut out = String::new();
        let mut i = 0;
        'o: while i <= cs.len() {
            for w in ["multiplied by", "times"] {
                let a: Vec<char> = w.chars().collect();
                if i + a.len() <= cs.len() && cs[i..i + a.len()] == a[..] && boundary(&cs, i) && boundary(&cs, i + a.len()) {
                    out.push('*');
                    i += a.len();
                    continue 'o;
                }
            }
            if i > 0 && cs[i - 1].is_ascii_digit() {
                let mut j = i;
                while j < cs.len() && cs[j].is_whitespace() {
                    j += 1;
                }
                if j < cs.len() && cs[j] == 'x' {
                    let mut k = j + 1;
                    while k < cs.len() && cs[k].is_whitespace() {
                        k += 1;
                    }
                    if k < cs.len() && cs[k].is_ascii_digit() {
                        out.push('*');
                        i = k;
                        continue 'o;
                    }
                }
            }
            if i < cs.len() {
                out.push(cs[i]);
            }
            i += 1;
        }
        t = out;
    }
    t = py::sub_alts(&t, &[("plus", true, true), ("and", true, true)], "+");
    t = py::sub_alts(&t, &[("minus", true, true)], "-");
    t = py::sub_alts(&t, &[("squared", true, true)], "^2");
    t = py::sub_alts(&t, &[("mod", true, true)], "%");
    let kept: String = t.chars().filter(|c| c.is_ascii_digit() || ".+-*/^%() ".contains(*c)).collect();
    kept.trim().to_string()
}

#[allow(dead_code)]
fn _unused(c: char) -> bool {
    is_word(c)
}
