//! Program -> bash command line (kural/compile.py). Deterministic: same program, same machine, same command.

use crate::ground::Env;
use crate::lang::{CommandDef, Language};
use crate::program::{Pipe, Program, Stage};
use crate::py::quote as q;
use crate::values as V;

#[derive(Debug, PartialEq)]
pub enum CErr {
    /// the request is outside what kural can do (declines)
    Unsupported,
    /// the program cannot be built (Python raised some other exception): this reading is skipped
    Other(String),
}

type R<T> = Result<T, CErr>;

fn other<T>(why: &str) -> R<T> {
    Err(CErr::Other(why.to_string()))
}

fn req<'a>(st: &'a Stage, slot: &str) -> R<&'a str> {
    match st.get(slot) {
        Some(v) => Ok(v),
        None => other(&format!("{}: {slot} missing", st.name)),
    }
}

const BASE_RISK: &[(&str, u8)] = &[("delete", 2), ("kill", 2), ("replace", 2), ("inplace", 2), ("trash", 1), ("move", 1), ("save", 1),
    ("lappend", 1), ("chmod", 1), ("chown", 1), ("rn_lower", 1), ("rn_upper", 1), ("rn_spaces", 1), ("rn_replace", 1), ("rn_prefix", 1),
    ("rn_suffix", 1), ("rn_ext", 1), ("upload", 1), ("sudo", 1)];
const LINE_SOURCES: &[&str] = &["lines", "history", "env", "pkgs", "services", "journal", "dmesg", "clipboard", "last"];

pub fn stage_risk(lang: &Language, name: &str) -> u8 {
    if let Some(c) = lang.command(name) {
        return c.risk;
    }
    BASE_RISK.iter().find(|(n, _)| *n == name).map(|(_, r)| *r).unwrap_or(0)
}

pub fn risk(lang: &Language, program: &Program) -> u8 {
    program.iter().flatten().map(|s| stage_risk(lang, &s.name)).max().unwrap_or(0)
}

fn file(text: &str, env: &mut Env) -> String {
    V::quote_path(&env.file(text.trim()))
}

fn new_file(text: &str) -> String {
    V::quote_path(text.trim())
}

fn dir(text: &str, env: &mut Env) -> String {
    let p = { let e: &Env = env; V::path(text, Some(&|x: &str| e.isdir(x))) };
    V::quote_path(&env.folder(&p))
}

fn vpath(text: &str, env: &Env) -> String {
    V::path(text, Some(&|x: &str| env.isdir(x)))
}

fn sed_pat(t: &str) -> String {
    let mut o = String::new();
    for c in t.chars() {
        if "\\/.*[]^$".contains(c) {
            o.push('\\');
        }
        o.push(c);
    }
    o
}

fn sed_rep(t: &str) -> String {
    let mut o = String::new();
    for c in t.chars() {
        if "\\/&".contains(c) {
            o.push('\\');
        }
        o.push(c);
    }
    o
}

fn words(text: &str) -> String {
    text.replace(',', " ")
        .split_whitespace()
        .filter(|w| !["and", "&", "plus", "also"].contains(&w.to_lowercase().as_str()))
        .map(q)
        .collect::<Vec<_>>()
        .join(" ")
}

fn rsplit_last(s: &str) -> &str {
    s.rsplit('/').next().unwrap_or(s)
}

// --------------------------------------------------------------------------- files
fn rename_loop(sink: &str, sa: &Stage, s: &str, v: &str) -> R<(String, String)> {
    let pre = match sink {
        "rn_replace" => format!("o={} n={}; ", q(req(sa, "old")?), q(req(sa, "new")?)),
        "rn_prefix" | "rn_suffix" => format!("p={}; ", q(req(sa, "text")?)),
        "rn_ext" => format!("p={}; ", q(&V::extensions(req(sa, "ext")?)[0])),
        _ => String::new(),
    };
    let body = match sink {
        "rn_lower" => "t=\"$d/${b,,}\"",
        "rn_upper" => "t=\"$d/${b^^}\"",
        "rn_spaces" => "t=\"$d/${b// /_}\"",
        "rn_replace" => "t=\"$d/${b//\"$o\"/$n}\"",
        "rn_prefix" => "t=\"$d/$p$b\"",
        "rn_suffix" => "case $b in *.*) t=\"$d/${b%.*}$p.${b##*.}\";; *) t=\"$d/$b$p\";; esac",
        _ => "t=\"$d/${b%.*}.$p\"",
    };
    let lp = format!("while IFS= read -r -d '' f; do d=$(dirname -- \"$f\") b=$(basename -- \"$f\"); {body}; [ \"$f\" = \"$t\" ] || {s}mv -n{v} -- \"$f\" \"$t\"; done");
    Ok((pre, lp))
}

fn archive(fmt: &str, to: Option<&str>, default: &str) -> String {
    let mut name = to.unwrap_or(default).trim().to_string();
    if !(name.ends_with(".zip") || name.ends_with(".tar.gz") || name.ends_with(".tgz")) {
        name += if fmt == "zip" { ".zip" } else { ".tar.gz" };
    }
    V::quote_path(&name)
}

fn fix_fmt(sa: &mut Stage) {
    let to = sa.get("to").unwrap_or("").to_lowercase();
    if to.ends_with(".zip") {
        sa.set("fmt", Some("zip".into()));
    } else if to.ends_with(".tar.gz") || to.ends_with(".tgz") {
        sa.set("fmt", Some("tgz".into()));
    }
}

fn direct(raw: &[String], sink: &str, sa: &mut Stage, rest: &[Stage], env: &mut Env, sudo: bool) -> R<Option<String>> {
    let s = if sudo { "sudo " } else { "" };
    let v = if env.pref("verbose") { " -v" } else { "" };
    let i = if env.pref("confirm") { " -i" } else { "" };
    let files = raw.iter().map(|n| V::quote_path(n)).collect::<Vec<_>>().join(" ");
    let anydir = raw.iter().any(|n| env.isdir(n));
    let sink = if sink == "delete" && env.pref("trash") { "trash" } else { sink };
    Ok(Some(match sink {
        "fshow" | "details" => format!("{s}ls -lhd -- {files}"),
        "fcount" => format!("ls -d -- {files} | wc -l"),
        "total" => format!("{s}du -ch -- {files} | tail -n 1"),
        "sizes" => format!("{s}du -sh -- {files}"),
        "delete" => format!("{s}rm{}{i}{v} -- {files}", if anydir { " -r" } else { "" }),
        "trash" => format!("{} -- {files}", if env.has("gio") { "gio trash" } else { "trash-put" }),
        "move" | "copy" => {
            let tool = format!("{}{i}{v}", if sink == "move" { "mv" } else { "cp -r" });
            let to = req(sa, "to")?.to_string();
            let dest = vpath(&to, env);
            if raw.len() == 1 && rsplit_last(&dest).trim_start_matches('.').contains('.') && !env.isdir(&dest) {
                format!("{s}{tool} -- {files} {}", V::quote_path(&dest))
            } else {
                format!("{s}{tool} -t {} -- {files}", dir(&to, env))
            }
        }
        "compress" => {
            fix_fmt(sa);
            let def = rsplit_last(raw[0].trim_end_matches('/')).to_string();
            let fmt = req(sa, "fmt")?.to_string();
            let name = archive(&fmt, sa.get("to"), &def);
            if fmt == "zip" { format!("{s}zip -qr {name} {files}") } else { format!("{s}tar -czf {name} {files}") }
        }
        "fgrep" => format!("{s}grep -rni -- {} {files}", q(req(sa, "text")?)),
        "fopen" => {
            if raw.len() == 1 { format!("xdg-open {files}") } else { format!("for f in {files}; do xdg-open \"$f\"; done") }
        }
        "fedit" => format!("{s}${{EDITOR:-nano}} {files}"),
        "chmod" => format!("{s}chmod {} -- {files}", V::mode(req(sa, "mode")?)),
        "chown" => format!("sudo chown {} -- {files}", V::owner(req(sa, "owner")?)),
        x if x.starts_with("rn_") => {
            let (pre, lp) = rename_loop(x, sa, s, v)?;
            format!("{pre}printf '%s\\0' {files} | {lp}")
        }
        "replace" => format!("{s}sed -i 's/{}/{}/g' -- {files}", sed_pat(req(sa, "old")?), sed_rep(req(sa, "new")?)),
        "fhash" => format!("{s}sha256sum -- {files}"),
        "upload" => format!("rsync -avh --progress {files} {}", q(req(sa, "to")?.trim())),
        "fnames" => line_ops(Some(format!("printf '%s\\n' {files}")), None, rest, env, sudo)?,
        "fcat" => {
            if raw.len() == 1 {
                line_ops(None, Some(files), rest, env, sudo)?
            } else {
                line_ops(Some(format!("{s}cat -- {files}")), None, rest, env, sudo)?
            }
        }
        _ => return Ok(None),
    }))
}

const KILLS_EXTS: &[&str] = &["depth", "name", "larger", "smaller", "newer", "older", "since", "before", "empty", "hidden", "owner", "isexec"];

fn files(pipe: &[Stage], env: &mut Env, mut sudo: bool) -> R<String> {
    let head = &pipe[0];
    let explicit = head.name == "paths";
    let (raw, roots, dirs): (Vec<String>, Vec<String>, bool) = if explicit {
        let mut raw: Vec<String> = V::names(req(head, "names")?).iter().map(|n| env.file(n)).collect();
        if raw.is_empty() {
            raw.push(".".into());
        }
        let roots = raw.iter().map(|n| V::quote_path(n)).collect();
        (raw, roots, false)
    } else {
        let dirs = head.get("what") == Some("dirs");
        let root = match head.get("in") { Some(i) if !i.is_empty() => dir(i, env), _ => ".".into() };
        (Vec::new(), vec![root], dirs)
    };
    let mut tests: Vec<String> = Vec::new();
    let mut exts_only: Option<Vec<String>> = Some(Vec::new());
    let (mut depth, mut contains, mut ncontains): (Option<i64>, Option<String>, Option<String>) = (None, None, None);
    let (mut sort, mut take, mut pick, mut vague): (Option<(String, String)>, Option<i64>, Option<i64>, Option<&str>) = (None, None, None, None);
    let mut sink = "fshow".to_string();
    let mut sa = Stage::new("fshow");
    let mut rest: Vec<Stage> = Vec::new();
    let mut neg = false;
    let stages = &pipe[1..];
    for (idx, st) in stages.iter().enumerate() {
        let name = st.name.as_str();
        if name == "not" {
            neg = true;
            continue;
        }
        let bang = if neg { "! " } else { "" };
        neg = false;
        match name {
            "flat" => depth = Some(1),
            "depth" => depth = Some(V::count(st.get("n"), 1)),
            "ext" => {
                let exts = V::extensions_any(req(st, "ext")?);
                if exts.len() == 1 {
                    tests.push(format!("{bang}-iname {}", q(&format!("*.{}", exts[0]))));
                } else {
                    let alts: Vec<String> = exts.iter().map(|e| format!("-iname {}", q(&format!("*.{e}")))).collect();
                    tests.push(format!("{bang}\\( {} \\)", alts.join(" -o ")));
                }
                exts_only = match (bang.is_empty(), exts_only.take()) {
                    (true, Some(mut e)) => {
                        e.extend(exts);
                        Some(e)
                    }
                    _ => None,
                };
                continue;
            }
            "name" => {
                let t = req(st, "text")?;
                let pat = if t.contains(['*', '?', '[']) {
                    t.to_string()
                } else {
                    match req(st, "mode")? {
                        "is" => t.to_string(),
                        "has" => format!("*{t}*"),
                        "starts" => format!("{t}*"),
                        "ends" => format!("*{t}"),
                        m => return other(&format!("bad mode {m}")),
                    }
                };
                tests.push(format!("{bang}-iname {}", q(&pat)));
            }
            "larger" | "smaller" => match st.get("size") {
                None => vague = Some(if name == "larger" { "larger" } else { "smaller" }),
                Some(sz) => tests.push(format!("-size {}{}", if name == "larger" { "+" } else { "-" }, V::find_size(sz))),
            },
            "newer" | "older" => match st.get("age") {
                None => vague = Some(if name == "newer" { "newer" } else { "older" }),
                Some(age) => tests.push(format!("{bang}{}", V::find_age(age, name == "newer"))),
            },
            "since" => tests.push(format!("-newermt {}", q(&V::date(req(st, "date")?)))),
            "before" => tests.push(format!("! -newermt {}", q(&V::date(req(st, "date")?)))),
            "contains" => {
                let t = req(st, "text")?.to_string();
                if bang.is_empty() { contains = Some(t) } else { ncontains = Some(t) }
            }
            "empty" => tests.push(format!("{bang}-empty")),
            "hidden" => tests.push(format!("{bang}-name {}", q(".*"))),
            "owner" => tests.push(format!("{bang}-user {}", V::owner(req(st, "user")?))),
            "isexec" => tests.push(format!("{bang}-executable")),
            "fsort" => sort = Some((req(st, "key")?.to_string(), req(st, "order")?.to_string())),
            "ftake" => take = Some(V::count(st.get("n"), 1)),
            "fpick" => pick = Some(V::count(st.get("n"), 1)),
            "fnames" | "fcat" => {
                sink = name.to_string();
                rest = stages[idx + 1..].to_vec();
                break;
            }
            _ => {
                sink = name.to_string();
                sa = st.clone();
            }
        }
        if name != "flat" && KILLS_EXTS.contains(&name) {
            exts_only = None;
        }
    }
    if let Some(vg) = vague {
        if ["fshow", "sizes", "details"].contains(&sink.as_str()) && sort.is_none() {
            sort = Some(match vg {
                "larger" => ("size".into(), "desc".into()),
                "smaller" => ("size".into(), "asc".into()),
                "older" => ("time".into(), "asc".into()),
                _ => ("time".into(), "desc".into()),
            });
            take = Some(take.unwrap_or(V::SHOW_TOP));
            env.note(match vg {
                "larger" => "no size given: biggest first",
                "smaller" => "no size given: smallest first",
                "older" => "no age given: oldest first",
                _ => "no age given: newest first",
            }.into());
        } else {
            tests.push(match vg {
                "larger" => format!("-size +{}", V::BIG),
                "smaller" => format!("-size -{}", V::SMALL),
                "older" => format!("-mtime +{}", V::OLD_DAYS - 1),
                _ => format!("-mtime -{}", V::NEW_DAYS),
            });
            env.note(match vg {
                "larger" => format!("no size given: over {}", V::BIG),
                "smaller" => format!("no size given: under {}", V::SMALL),
                "older" => format!("no age given: over {} days", V::OLD_DAYS),
                _ => format!("no age given: last {} days", V::NEW_DAYS),
            });
            exts_only = None;
        }
    }
    let plain = contains.is_none() && ncontains.is_none() && sort.is_none() && take.is_none() && pick.is_none();
    if explicit && plain && tests.is_empty() && depth.is_none() {
        if let Some(out) = direct(&raw, &sink, &mut sa, &rest, env, sudo)? {
            return Ok(out);
        }
    }
    if sink == "chown" {
        sudo = true;
    }
    if sink == "delete" && env.pref("trash") {
        sink = "trash".into();
    }
    let s = if sudo { "sudo " } else { "" };
    let v = if env.pref("verbose") { " -v" } else { "" };
    let ask = if env.pref("confirm") { " -i" } else { "" };
    let rename = sink.starts_with("rn_");
    let mut parts = vec![format!("{s}find")];
    parts.extend(roots.iter().cloned());
    if rename && dirs {
        parts.push("-depth".into());
    }
    if let Some(d) = depth {
        if d != 0 {
            parts.push(format!("-maxdepth {d}"));
        }
    }
    if dirs {
        parts.push("-mindepth 1".into());
    }
    parts.push(format!("-type {}", if dirs { "d" } else { "f" }));
    parts.extend(tests.iter().cloned());
    let base = parts.join(" ");

    let keyed = matches!(&sort, Some((k, _)) if k == "size" || k == "time");
    let mut chain = String::new();
    if !plain {
        let mut grep = String::new();
        if let Some(c) = &contains {
            grep += &format!(" | xargs -0 -r grep -lZ -- {}", q(c));
        }
        if let Some(c) = &ncontains {
            grep += &format!(" | xargs -0 -r grep -LZ -- {}", q(c));
        }
        if keyed {
            let (k, o) = sort.as_ref().unwrap();
            if !grep.is_empty() {
                let stat = if k == "size" { "%s\\t%n\\0" } else { "%Y\\t%n\\0" };
                chain = format!("{base} -print0{grep} | xargs -0 -r stat --printf {}", q(stat));
            } else {
                chain = format!("{base} -printf {}", q(&format!("{}\\t%p\\0", if k == "size" { "%s" } else { "%T@" })));
            }
            chain += &format!(" | sort -z -n{}", if o == "desc" { "r" } else { "" });
            if let Some(t) = take {
                if t != 0 {
                    chain += &format!(" | head -z -n {t}");
                }
            }
            chain += " | cut -z -f2-";
        } else {
            chain = format!("{base} -print0{grep}");
            if let Some((_, o)) = &sort {
                chain += &format!(" | sort -z{}", if o == "desc" { " -r" } else { "" });
            }
            if let Some(t) = take {
                if t != 0 {
                    chain += &format!(" | head -z -n {t}");
                }
            }
        }
        if let Some(p) = pick {
            if p != 0 {
                chain += &format!(" | sed -zn {}", q(&format!("{p}p")));
            }
        }
    }
    let many = |cmd: &str, dd: &str| if plain { format!("{base} -exec {cmd} {{}} +") } else { format!("{chain} | xargs -0 -r {s}{cmd}{dd}") };
    let one = |cmd: &str| if plain { format!("{base} -exec {cmd} {{}} \\;") } else { format!("{chain} | xargs -0 -r -n 1 {s}{cmd}") };
    let nul = || if plain { format!("{base} -print0") } else { chain.clone() };
    let newline = || if plain { base.clone() } else { format!("{chain} | tr '\\0' '\\n'") };
    let sink = sink.as_str();
    Ok(match sink {
        "fshow" => {
            if plain {
                base.clone()
            } else if keyed && sort.as_ref().unwrap().0 == "size" {
                format!("{chain} | xargs -0 -r du -sh --")
            } else {
                newline()
            }
        }
        "fcount" => if plain { format!("{base} | wc -l") } else { format!("{chain} | tr -cd '\\0' | wc -c") },
        "total" => format!("{} | du -ch --files0-from=- | tail -n 1", nul()),
        "sizes" => {
            if plain {
                if dirs && tests.is_empty() && !explicit && depth.is_none() {
                    format!("{s}du -h --max-depth=1 {} | sort -rh", roots[0])
                } else {
                    format!("{base} -exec du -sh {{}} + | sort -rh")
                }
            } else {
                format!("{chain} | xargs -0 -r du -sh --")
            }
        }
        "details" => many("ls -lhd", " --"),
        "delete" => {
            let rm = format!("rm{}{ask}{v}", if dirs { " -r" } else { "" });
            if plain {
                if rm == "rm" { format!("{base} -delete") } else { format!("{base} -exec {rm} {{}} +") }
            } else {
                format!("{chain} | xargs -0 -r {}{s}{rm} --", if ask.is_empty() { "" } else { "-o " })
            }
        }
        "trash" => many(if env.has("gio") { "gio trash" } else { "trash-put" }, " --"),
        "move" | "copy" => {
            let tool = format!("{}{ask}{v}", if sink == "move" { "mv" } else if dirs { "cp -r" } else { "cp" });
            let to = req(&sa, "to")?.to_string();
            let d = dir(&to, env);
            many(&format!("{tool} -t {d}"), " --")
        }
        "compress" => {
            fix_fmt(&mut sa);
            if req(&sa, "fmt")? == "zip" {
                format!("{} | xargs -0 -r zip -q{} {}", nul(), if dirs { "r" } else { "" }, archive("zip", sa.get("to"), "archive"))
            } else {
                format!("{} | tar --null -czf {} -T -", nul(), archive("tgz", sa.get("to"), "archive"))
            }
        }
        "fgrep" => {
            let text = q(req(&sa, "text")?);
            match &exts_only {
                Some(e) if plain && !dirs && depth.is_none() && !explicit && (!e.is_empty() || tests.is_empty()) => {
                    let inc: String = e.iter().map(|x| format!(" --include={}", q(&format!("*.{x}")))).collect();
                    format!("{s}grep -rni{inc} -- {text} {}", roots[0])
                }
                _ => many(&format!("grep -Hni -- {text}"), ""),
            }
        }
        "fopen" => one("xdg-open"),
        "fedit" => if plain { format!("{base} -exec ${{EDITOR:-nano}} {{}} +") } else { format!("{chain} | xargs -0 -r -o ${{EDITOR:-nano}}") },
        "chmod" => many(&format!("chmod {}", V::mode(req(&sa, "mode")?)), " --"),
        "chown" => many(&format!("chown {}", V::owner(req(&sa, "owner")?)), " --"),
        x if x.starts_with("rn_") => {
            let (pre, lp) = rename_loop(x, &sa, s, v)?;
            format!("{pre}{} | {lp}", nul())
        }
        "replace" => many(&format!("sed -i 's/{}/{}/g'", sed_pat(req(&sa, "old")?), sed_rep(req(&sa, "new")?)), " --"),
        "fhash" => many("sha256sum", " --"),
        "by_ext" => format!("{} | sed -n 's/.*[^/]\\.\\([^./]*\\)$/\\1/p' | tr '[:upper:]' '[:lower:]' | sort | uniq -c | sort -rn", newline()),
        "dupes" => format!("{} | xargs -0 -r sha256sum | sort | uniq -w64 --all-repeated=separate", nul()),
        "upload" => {
            let dest = q(req(&sa, "to")?.trim());
            if plain { format!("{base} -exec rsync -avh --progress {{}} {dest} \\;") } else { format!("{chain} | xargs -0 -r -I{{}} rsync -avh --progress {{}} {dest}") }
        }
        "fnames" => line_ops(Some(newline()), None, &rest, env, sudo)?,
        "fcat" => {
            let src = if plain { format!("{base} -exec cat {{}} +") } else { format!("{chain} | xargs -0 -r {s}cat --") };
            line_ops(Some(src), None, &rest, env, sudo)?
        }
        other_sink => return other(other_sink),
    })
}

// --------------------------------------------------------------------------- lines
fn pkg_list(pm: &str) -> R<&'static str> {
    Ok(match pm {
        "pacman" => "pacman -Qq",
        "apt" => "apt list --installed 2>/dev/null",
        "dnf" => "dnf list installed",
        "zypper" => "rpm -qa",
        "brew" => "brew list",
        _ => return other("package manager"),
    })
}

fn line_source(head: &Stage, env: &mut Env) -> R<String> {
    Ok(match head.name.as_str() {
        "history" => "history".into(),
        "env" => "printenv | sort".into(),
        "pkgs" => pkg_list(&env.pm)?.into(),
        "services" => "systemctl list-units --type=service --state=running --no-pager --no-legend".into(),
        "journal" => match head.get("unit") {
            Some(u) if !u.is_empty() => format!("journalctl -b --no-pager -u {}", q(u.trim())),
            _ => "journalctl -b --no-pager".into(),
        },
        "dmesg" => "sudo dmesg".into(),
        "clipboard" => env.clipboard(true),
        "last" => match &env.last {
            Some(l) if !l.is_empty() => l.clone(),
            _ => return Err(CErr::Unsupported),
        },
        x => return other(x),
    })
}

fn lines(pipe: &[Stage], env: &mut Env, sudo: bool) -> R<String> {
    let head = &pipe[0];
    let mut stages = &pipe[1..];
    if head.name == "lines" {
        let f = file(req(head, "of")?, env);
        return line_ops(None, Some(f), stages, env, sudo);
    }
    let mut src = line_source(head, env)?;
    if head.name == "journal" && !stages.is_empty() && stages[0].name == "llast" {
        src += &format!(" -n {}", V::count(stages[0].get("n"), 10));
        stages = &stages[1..];
    }
    line_ops(Some(src), None, stages, env, sudo)
}

fn line_no(text: &str) -> String {
    if ["last", "final", "the last", "bottom"].contains(&text.trim().to_lowercase().as_str()) { "$".into() } else { V::count(Some(text), 1).to_string() }
}

fn digit_runs(s: &str) -> Vec<i64> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in s.chars().chain(std::iter::once(' ')) {
        if c.is_ascii_digit() {
            cur.push(c);
        } else if !cur.is_empty() {
            out.push(cur.parse().unwrap_or(0));
            cur.clear();
        }
    }
    out
}

fn line_ops(src: Option<String>, f: Option<String>, stages: &[Stage], env: &mut Env, sudo: bool) -> R<String> {
    let s = if sudo { "sudo " } else { "" };
    let mut cmds: Vec<(String, bool)> = Vec::new();
    let mut redirect = String::new();
    let mut inplace = false;
    for st in stages {
        match st.name.as_str() {
            "match" => cmds.push((format!("grep -i -- {}", q(req(st, "text")?)), true)),
            "nomatch" => cmds.push((format!("grep -vi -- {}", q(req(st, "text")?)), true)),
            "uniq" => {
                if let Some(last) = cmds.last_mut() {
                    if last.0.starts_with("sort") && !last.0.contains("uniq") {
                        last.0 += " -u";
                        continue;
                    }
                }
                cmds.push(("awk '!seen[$0]++'".into(), true));
            }
            "lsort" => {
                let flags = format!("{}{}", if req(st, "key")? == "num" { "n" } else { "" }, if req(st, "order")? == "desc" { "r" } else { "" });
                cmds.push((if flags.is_empty() { "sort".into() } else { format!("sort -{flags}") }, true));
            }
            "ltake" => cmds.push((format!("head -n {}", V::count(st.get("n"), 10)), true)),
            "llast" => cmds.push((format!("tail -n {}", V::count(st.get("n"), 10)), true)),
            "lrange" => {
                let (a, b) = (req(st, "a")?, req(st, "b")?);
                let nums = digit_runs(&format!("{a} {b}"));
                let (lo, hi) = if a == b && nums.len() >= 4 {
                    (nums[0].min(nums[1]), nums[0].max(nums[1]))
                } else {
                    let (x, y) = (V::count(Some(a), 1), V::count(Some(b), 1));
                    (x.min(y), x.max(y))
                };
                cmds.push((format!("sed -n '{lo},{hi}p'"), true));
            }
            "lline" => cmds.push((format!("sed -n '{}p'", V::count(Some(req(st, "n")?), 1)), true)),
            "ldel" => cmds.push((format!("sed '{}d'", line_no(req(st, "n")?)), true)),
            "col" => {
                let n = V::count(Some(req(st, "n")?), 1);
                let sep = req(st, "sep")?;
                if sep == "space" {
                    cmds.push((format!("awk '{{print ${n}}}'"), true));
                } else {
                    let d = match sep {
                        "comma" => Some(","),
                        "tab" => None,
                        "colon" => Some(":"),
                        "semi" => Some(";"),
                        "bar" => Some("|"),
                        _ => Some(" "),
                    };
                    cmds.push((format!("cut {}-f{n}", d.map(|d| format!("-d{} ", q(d))).unwrap_or_default()), true));
                }
            }
            "lreplace" => cmds.push((format!("sed 's/{}/{}/g'", sed_pat(req(st, "old")?), sed_rep(req(st, "new")?)), true)),
            "lfreq" => cmds.extend([("sort".into(), true), ("uniq -c".into(), false), ("sort -rn".into(), false)]),
            "lnum" => cmds.push(("cat -n".into(), true)),
            "lrev" => cmds.push(("tac".into(), true)),
            "lshuf" => cmds.push(("shuf".into(), true)),
            "lupper" => cmds.push(("tr '[:lower:]' '[:upper:]'".into(), false)),
            "llower" => cmds.push(("tr '[:upper:]' '[:lower:]'".into(), false)),
            "lsum" => cmds.push(("awk '{s += $1} END {print s}'".into(), true)),
            "lblank" => cmds.push(("grep -v '^[[:space:]]*$'".into(), true)),
            "lcount" => cmds.push(("wc -l".into(), false)),
            "wcount" => cmds.push(("wc -w".into(), false)),
            "ccount" => cmds.push(("wc -m".into(), false)),
            "clip" => {
                let c = env.clipboard(false);
                cmds.push((c, false));
            }
            "save" => {
                let t = new_file(req(st, "to")?);
                redirect = if sudo { format!(" | sudo tee {t} > /dev/null") } else { format!(" > {t}") };
            }
            "lappend" => {
                let t = new_file(req(st, "to")?);
                redirect = if sudo { format!(" | sudo tee -a {t} > /dev/null") } else { format!(" >> {t}") };
            }
            "inplace" => inplace = true,
            _ => {}
        }
    }
    let ops: Vec<&str> = stages.iter().map(|s| s.name.as_str()).filter(|n| *n != "inplace" && *n != "lshow").collect();
    if inplace {
        let Some(f) = f else { return Err(CErr::Unsupported) };
        let a = stages.first().cloned().unwrap_or_else(|| Stage::new(""));
        if ops == ["nomatch"] {
            return Ok(format!("{s}sed -i '/{}/Id' {f}", sed_pat(req(&a, "text")?)));
        }
        if ops == ["match"] {
            return Ok(format!("{s}sed -i '/{}/I!d' {f}", sed_pat(req(&a, "text")?)));
        }
        if ops == ["lreplace"] {
            return Ok(format!("{s}sed -i 's/{}/{}/g' {f}", sed_pat(req(&a, "old")?), sed_rep(req(&a, "new")?)));
        }
        if ops == ["ldel"] {
            return Ok(format!("{s}sed -i '{}d' {f}", line_no(req(&a, "n")?)));
        }
        if ops == ["lblank"] {
            return Ok(format!("{s}sed -i '/^[[:space:]]*$/d' {f}"));
        }
        if ops == ["lsort"] || ops == ["lsort", "uniq"] {
            return Ok(format!("{s}{} -o {f} {f}", cmds[0].0));
        }
        let body = join(None, Some(&f), &cmds, s);
        return Ok(format!("tmp=$(mktemp) && {body} > \"$tmp\" && {s}cp \"$tmp\" {f} && rm \"$tmp\""));
    }
    Ok(join(src.as_deref(), f.as_deref(), &cmds, s) + &redirect)
}

fn join(src: Option<&str>, f: Option<&str>, cmds: &[(String, bool)], s: &str) -> String {
    if let Some(src) = src {
        let first = if !s.is_empty() && !src.starts_with("sudo ") { format!("{s}{src}") } else { src.to_string() };
        let mut all = vec![first];
        all.extend(cmds.iter().map(|(c, _)| c.clone()));
        return all.join(" | ");
    }
    let f = f.unwrap_or("");
    if cmds.is_empty() {
        return format!("{s}cat {f}");
    }
    let (first, takes) = &cmds[0];
    let head = if *takes {
        format!("{s}{first} {f}")
    } else if !s.is_empty() {
        format!("sudo cat {f} | {first}")
    } else {
        format!("{first} < {f}")
    };
    let mut all = vec![head];
    all.extend(cmds[1..].iter().map(|(c, _)| c.clone()));
    all.join(" | ")
}

// --------------------------------------------------------------------------- processes
fn procs(pipe: &[Stage], env: &mut Env, sudo: bool) -> R<String> {
    let s = if sudo { "sudo " } else { "" };
    let head = &pipe[0];
    let (mut pname, mut user, mut sort, mut take): (Option<String>, Option<String>, Option<(String, String)>, Option<i64>) = (None, None, None, None);
    let (mut sink, mut force) = ("pshow".to_string(), "soft".to_string());
    for st in &pipe[1..] {
        match st.name.as_str() {
            "pname" => pname = Some(req(st, "name")?.trim().to_string()),
            "puser" => user = Some(V::owner(req(st, "user")?)),
            "psort" => sort = Some((req(st, "key")?.to_string(), req(st, "order")?.to_string())),
            "ptake" => take = Some(V::count(st.get("n"), 1)),
            "kill" => {
                sink = "kill".into();
                force = req(st, "force")?.to_string();
            }
            n => sink = n.to_string(),
        }
    }
    let sig = if force == "hard" { " -9" } else { "" };
    if head.name == "port" {
        let n = V::port(req(head, "n")?);
        return Ok(if sink == "kill" {
            if env.has("lsof") { format!("{s}kill{sig} $(lsof -ti :{n})") } else { format!("{s}fuser -k {n}/tcp") }
        } else if sink == "pcount" {
            format!("{s}lsof -ti :{n} | wc -l")
        } else if env.has("lsof") {
            format!("{s}lsof -i :{n}")
        } else {
            format!("{s}ss -tulpn | grep -w {}", q(&format!(":{n}")))
        });
    }
    let who = user.as_ref().map(|u| format!(" -u {u}")).unwrap_or_default();
    let order = sort.as_ref().map(|(k, o)| format!(" --sort={}%{k}", if o == "asc" { "" } else { "-" })).unwrap_or_default();
    let simple = sort.is_none() && take.is_none();
    let take_n = take.filter(|t| *t != 0);
    let sel = || match (&pname, &user) {
        (Some(p), _) if !p.is_empty() => format!("-C {}", q(p)),
        (_, Some(u)) => format!("-u {u}"),
        _ => "-e".into(),
    };
    let has_name = pname.as_deref().map(|p| !p.is_empty()).unwrap_or(false);
    if sink == "kill" {
        if has_name && simple {
            let p = pname.clone().unwrap();
            if user.is_none() && force == "soft" && env.is_service(&p) {
                env.note(format!("{p} is a running service"));
                return Ok(format!("sudo systemctl stop {}", q(&p)));
            }
            return Ok(format!("{s}pkill{sig}{who} -i -- {}", q(&p)));
        }
        if user.is_some() && !has_name && simple {
            return Ok(format!("{s}pkill{sig}{who}"));
        }
        let ord = if order.is_empty() { " --sort=-%mem".to_string() } else { order.clone() };
        return Ok(format!("ps {} -o pid={ord} | head -n {} | xargs -r {s}kill{sig}", sel(), take_n.unwrap_or(1)));
    }
    if sink == "pcount" {
        if has_name || user.is_some() {
            return Ok(format!("pgrep -c{who}{}", if has_name { format!(" -i -- {}", q(pname.as_ref().unwrap())) } else { String::new() }));
        }
        return Ok("ps -e --no-headers | wc -l".into());
    }
    if has_name && simple {
        return Ok(format!("pgrep -a{who} -i -- {}", q(pname.as_ref().unwrap())));
    }
    let cmd = format!("ps {} -o pid,user,%cpu,%mem,comm{order}", sel());
    Ok(match take_n {
        Some(t) => format!("{cmd} | head -n {}", t + 1),
        None => cmd,
    })
}

// --------------------------------------------------------------------------- stand-alone commands
fn pkg_tpl(pm: &str, key: &str) -> R<Option<&'static str>> {
    let t = match (pm, key) {
        ("pacman", "install") => "sudo pacman -S {}", ("pacman", "remove") => "sudo pacman -Rs {}", ("pacman", "search") => "pacman -Ss {}",
        ("pacman", "update") => "sudo pacman -Syu", ("pacman", "info") => "pacman -Qi {0} 2>/dev/null || pacman -Si {0}",
        ("pacman", "has") => "pacman -Q {}", ("pacman", "owner") => "pacman -Qo {}", ("pacman", "files") => "pacman -Ql {}",
        ("pacman", "clean") => "sudo pacman -Sc", ("pacman", "orphans") => "pacman -Qdtq",
        ("pacman", "autoremove") => "pacman -Qdtq | sudo pacman -Rns -", ("pacman", "upgradable") => "checkupdates",
        ("apt", "install") => "sudo apt install {}", ("apt", "remove") => "sudo apt remove {}", ("apt", "search") => "apt search {}",
        ("apt", "update") => "sudo apt update && sudo apt upgrade", ("apt", "info") => "apt show {}", ("apt", "has") => "dpkg -s {}",
        ("apt", "owner") => "dpkg -S {}", ("apt", "files") => "dpkg -L {}", ("apt", "clean") => "sudo apt clean",
        ("apt", "orphans") => "apt autoremove --dry-run", ("apt", "autoremove") => "sudo apt autoremove", ("apt", "upgradable") => "apt list --upgradable",
        ("dnf", "install") => "sudo dnf install {}", ("dnf", "remove") => "sudo dnf remove {}", ("dnf", "search") => "dnf search {}",
        ("dnf", "update") => "sudo dnf upgrade", ("dnf", "info") => "dnf info {}", ("dnf", "has") => "rpm -q {}", ("dnf", "owner") => "rpm -qf {}",
        ("dnf", "files") => "rpm -ql {}", ("dnf", "clean") => "sudo dnf clean all", ("dnf", "orphans") => "dnf repoquery --unneeded",
        ("dnf", "autoremove") => "sudo dnf autoremove", ("dnf", "upgradable") => "dnf check-update",
        ("zypper", "install") => "sudo zypper install {}", ("zypper", "remove") => "sudo zypper remove {}", ("zypper", "search") => "zypper search {}",
        ("zypper", "update") => "sudo zypper update", ("zypper", "info") => "zypper info {}", ("zypper", "has") => "rpm -q {}",
        ("zypper", "owner") => "rpm -qf {}", ("zypper", "files") => "rpm -ql {}", ("zypper", "clean") => "sudo zypper clean",
        ("zypper", "orphans") => "zypper packages --unneeded", ("zypper", "autoremove") => "sudo zypper remove --clean-deps",
        ("zypper", "upgradable") => "zypper list-updates",
        ("brew", "install") => "brew install {}", ("brew", "remove") => "brew uninstall {}", ("brew", "search") => "brew search {}",
        ("brew", "update") => "brew upgrade", ("brew", "info") => "brew info {}", ("brew", "has") => "brew list {}", ("brew", "owner") => return Ok(None),
        ("brew", "files") => "brew list {}", ("brew", "clean") => "brew cleanup", ("brew", "orphans") => "brew autoremove --dry-run",
        ("brew", "autoremove") => "brew autoremove", ("brew", "upgradable") => "brew outdated",
        _ => return other("package manager"),
    };
    Ok(Some(t))
}

fn pkg_key(name: &str) -> Option<&'static str> {
    Some(match name {
        "pkg_install" => "install", "pkg_remove" => "remove", "pkg_search" => "search", "sys_update" => "update", "pkg_info" => "info",
        "pkg_has" => "has", "pkg_owner" => "owner", "pkg_files" => "files", "pkg_clean" => "clean", "pkg_orphans" => "orphans",
        "pkg_autoremove" => "autoremove", "pkg_upgradable" => "upgradable",
        _ => return None,
    })
}

fn project_cmd(kind: Option<&str>, i: usize) -> &'static str {
    let t: [&str; 3] = match kind {
        Some("cargo") => ["cargo test", "cargo build", "cargo run"],
        Some("npm") => ["npm test", "npm run build", "npm start"],
        Some("go") => ["go test ./...", "go build ./...", "go run ."],
        Some("python") => ["pytest", "python3 -m build", "python3 -m pytest -q"],
        Some("cmake") => ["ctest --test-dir build", "cmake -B build && cmake --build build", "cmake --build build"],
        Some("gradle") => ["./gradlew test", "./gradlew build", "./gradlew run"],
        Some("maven") => ["mvn test", "mvn package", "mvn exec:java"],
        Some("meson") => ["meson test -C build", "meson setup build && meson compile -C build", "meson compile -C build"],
        _ => ["make test", "make", "make run"],
    };
    t[i]
}

const RUNNERS: &[(&str, &str)] = &[(".py", "python3"), (".sh", "bash"), (".bash", "bash"), (".js", "node"), (".mjs", "node"),
    (".ts", "npx tsx"), (".rb", "ruby"), (".pl", "perl"), (".php", "php"), (".lua", "lua"), (".go", "go run"), (".r", "Rscript")];

fn ident(s: &str) -> String {
    let o: String = s.chars().filter(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
    if o.is_empty() { "VAR".into() } else { o }
}

fn value(kind: &str, text: &str, env: &mut Env) -> String {
    let t = text.trim();
    match kind {
        "path" => dir(t, env),
        "file" => file(t, env),
        "words" => words(t),
        "count" => V::count(Some(t), 1).to_string(),
        "port" => V::port(t),
        "pct" => V::percent(t).to_string(),
        "envname" => ident(t),
        _ => q(t),
    }
}

/// Replace {slot} with its quoted value; drop "[...]" when a slot inside it has no value.
fn fill(c: &CommandDef, st: &Stage, env: &mut Env) -> R<String> {
    let tpl: Vec<char> = c.bash.as_deref().unwrap_or("").chars().collect();
    // pass 1: optional groups \[([^\[\]]*)\]
    let mut pass1 = String::new();
    let mut i = 0;
    while i < tpl.len() {
        if tpl[i] == '[' {
            if let Some(off) = tpl[i + 1..].iter().position(|&ch| ch == ']' || ch == '[') {
                let j = i + 1 + off;
                if tpl[j] == ']' {
                    let inner: String = tpl[i + 1..j].iter().collect();
                    let names = slot_names(&inner);
                    if names.iter().all(|n| st.get(n).is_some()) {
                        pass1 += &subst(&inner, c, st, env)?;
                    }
                    i = j + 1;
                    continue;
                }
            }
        }
        pass1.push(tpl[i]);
        i += 1;
    }
    subst(&pass1, c, st, env)
}

fn slot_names(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let cs: Vec<char> = s.chars().collect();
    let mut i = 0;
    while i < cs.len() {
        if cs[i] == '{' {
            let j = cs[i + 1..].iter().position(|c| !crate::py::is_word(*c)).map(|p| i + 1 + p);
            if let Some(j) = j {
                if j > i + 1 && cs[j] == '}' {
                    out.push(cs[i + 1..j].iter().collect());
                    i = j + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    out
}

fn subst(s: &str, c: &CommandDef, st: &Stage, env: &mut Env) -> R<String> {
    let cs: Vec<char> = s.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < cs.len() {
        if cs[i] == '{' {
            if let Some(p) = cs[i + 1..].iter().position(|c| !crate::py::is_word(*c)) {
                let j = i + 1 + p;
                if j > i + 1 && cs[j] == '}' {
                    let name: String = cs[i + 1..j].iter().collect();
                    let kind = c.slots.iter().find(|(n, _, _)| *n == name).map(|(_, k, _)| k.clone());
                    let Some(kind) = kind else { return other(&format!("unknown slot {name}")) };
                    let Some(v) = st.get(&name) else { return other(&format!("{name} has no value")) };
                    out += &value(&kind, v, env);
                    i = j + 1;
                    continue;
                }
            }
        }
        out.push(cs[i]);
        i += 1;
    }
    Ok(out)
}

fn extract(st: &Stage, env: &mut Env) -> R<String> {
    let f = req(st, "archive")?.trim().to_string();
    let low = f.to_lowercase();
    let ff = file(&f, env);
    let into = st.get("into").filter(|s| !s.is_empty()).map(|s| s.to_string());
    let into_d = |env: &mut Env| into.as_ref().map(|i| dir(i, env));
    Ok(if low.ends_with(".zip") {
        format!("unzip {ff}{}", into_d(env).map(|d| format!(" -d {d}")).unwrap_or_default())
    } else if low.ends_with(".7z") {
        format!("7z x {ff}{}", into_d(env).map(|d| format!(" -o{d}")).unwrap_or_default())
    } else if low.ends_with(".rar") {
        format!("unrar x {ff}{}", into_d(env).map(|d| format!(" {d}/")).unwrap_or_default())
    } else if low.ends_with(".gz") && !low.ends_with(".tar.gz") {
        format!("gunzip -k {ff}")
    } else {
        format!("tar -xf {ff}{}", into_d(env).map(|d| format!(" -C {d}")).unwrap_or_default())
    })
}

fn archive_list(st: &Stage, env: &mut Env) -> R<String> {
    let a = req(st, "archive")?;
    let low = a.trim().to_lowercase();
    let ff = file(a, env);
    Ok(if low.ends_with(".zip") {
        format!("unzip -l {ff}")
    } else if low.ends_with(".7z") {
        format!("7z l {ff}")
    } else if low.ends_with(".rar") {
        format!("unrar l {ff}")
    } else {
        format!("tar -tf {ff}")
    })
}

fn download(st: &Stage, env: &mut Env) -> R<String> {
    let url = q(req(st, "url")?.trim());
    let Some(to) = st.get("to").filter(|s| !s.is_empty()) else {
        return Ok(if env.has("curl") { format!("curl -LO {url}") } else { format!("wget {url}") });
    };
    let dest = vpath(to, env);
    let is_dir = env.isdir(&dest) || dest.ends_with('/') || !rsplit_last(&dest).trim_start_matches('.').contains('.');
    if !env.has("curl") {
        return Ok(format!("wget {} {} {url}", if is_dir { "-P" } else { "-O" }, V::quote_path(&dest)));
    }
    Ok(if is_dir {
        format!("curl -L --output-dir {} -O {url}", V::quote_path(&dest))
    } else {
        format!("curl -L -o {} {url}", V::quote_path(&dest))
    })
}

fn assign(name: &str, value: &str, as_ident: bool) -> (String, String) {
    let (mut name, mut value) = (name.trim().to_string(), value.trim().to_string());
    if let Some((head, rest)) = name.clone().split_once('=') {
        if value == name || value.is_empty() {
            value = rest.to_string();
        }
        name = head.to_string();
    }
    if value.contains('=') && value.split('=').next() == Some(name.as_str()) {
        value = value.split_once('=').map(|(_, r)| r.to_string()).unwrap_or_default();
    }
    let n = if as_ident { ident(&name) } else { q(&name) };
    (n, q(&value))
}

fn last(env: &Env) -> R<String> {
    match &env.last {
        Some(l) if !l.is_empty() => Ok(l.clone()),
        _ => Err(CErr::Unsupported),
    }
}

fn built(name: &str, st: &Stage, env: &mut Env) -> R<String> {
    if let Some(key) = pkg_key(name) {
        let Some(tpl) = pkg_tpl(&env.pm.clone(), key)? else { return Err(CErr::Unsupported) };
        let arg = if let Some(slot) = st.args.first().map(|(k, _)| k.as_str()) {
            match slot {
                "names" => words(req(st, "names")?),
                "term" => q(req(st, "term")?.trim()),
                "name" => q(req(st, "name")?.trim()),
                "file" => {
                    let t = req(st, "file")?.trim().to_string();
                    if t.contains('/') || env.exists(&t) { file(&t, env) } else { q(&t) }
                }
                _ => String::new(),
            }
        } else {
            String::new()
        };
        return Ok(tpl.replace("{0}", &arg).replace("{}", &arg));
    }
    Ok(match name {
        "battery" => "for b in /sys/class/power_supply/BAT*; do echo \"$(cat \"$b/capacity\")% $(cat \"$b/status\")\"; done".into(),
        "dns" => {
            let h = q(req(st, "host")?.trim());
            if env.has("dig") { format!("dig +short {h}") } else { format!("getent hosts {h}") }
        }
        "download" => download(st, env)?,
        "serve" => format!("python3 -m http.server {}", st.get("port").filter(|s| !s.is_empty()).map(V::port).unwrap_or_else(|| "8000".into())),
        "poweroff" => match st.get("in").filter(|s| !s.is_empty()) {
            Some(i) => format!("shutdown -h +{}", V::minutes(i)),
            None => "systemctl poweroff".into(),
        },
        "venv_make" => format!("python3 -m venv {}", match st.get("path").filter(|s| !s.is_empty()) { Some(p) => dir(p, env), None => ".venv".into() }),
        "venv_on" => {
            let p = match st.get("path").filter(|s| !s.is_empty()) {
                Some(p) => {
                    let p = vpath(p, env);
                    p.strip_suffix("/bin/activate").map(|x| x.to_string()).unwrap_or(p)
                }
                None => [".venv", "venv", "env"].iter().find(|d| env.isdir(d)).map(|d| d.to_string()).unwrap_or_else(|| ".venv".into()),
            };
            format!("source {}/bin/activate", V::quote_path(&p))
        }
        "proj_test" => project_cmd(env.project(), 0).into(),
        "proj_build" => project_cmd(env.project(), 1).into(),
        "proj_run" => project_cmd(env.project(), 2).into(),
        "run_file" => {
            let n = env.file(req(st, "file")?.trim());
            let low = n.to_lowercase();
            match RUNNERS.iter().find(|(e, _)| low.ends_with(e)) {
                Some((_, tool)) => format!("{tool} {}", V::quote_path(&n)),
                None => V::quote_path(&if n.contains('/') || n.starts_with('~') { n.clone() } else { format!("./{n}") }),
            }
        }
        "dk_build" => format!("docker build {}.", st.get("tag").filter(|s| !s.is_empty()).map(|t| format!("-t {} ", q(t.trim()))).unwrap_or_default()),
        "git_log" => format!("git log --oneline -n {}", V::count(st.get("n"), 10)),
        "git_unstage" => format!("git restore --staged {}", match st.get("file").filter(|s| !s.is_empty()) { Some(f) => file(f, env), None => ".".into() }),
        "ls" => match st.get("path").filter(|s| !s.is_empty()) {
            None => "ls -lah".into(),
            Some(p) => {
                let pp = vpath(p, env);
                let p = env.folder(&pp);
                if env.isfile(&p) { format!("cat {}", V::quote_path(&p)) } else { format!("ls -lah {}", V::quote_path(&p)) }
            }
        },
        "tree" => {
            let p = st.get("path").filter(|s| !s.is_empty()).map(|p| dir(p, env)).unwrap_or_default();
            let d = st.get("depth").filter(|s| !s.is_empty()).map(|d| V::count(Some(d), 1));
            let dflag = |prefix: &str| d.filter(|x| *x != 0).map(|x| format!("{prefix}{x}")).unwrap_or_default();
            let parts: Vec<String> = if env.has("tree") {
                vec!["tree".into(), dflag("-L "), p]
            } else {
                vec!["find".into(), if p.is_empty() { ".".into() } else { p }, dflag("-maxdepth "), "-print | sort".into()]
            };
            parts.into_iter().filter(|x| !x.is_empty()).collect::<Vec<_>>().join(" ")
        }
        "view" => {
            let t = env.file(req(st, "file")?.trim());
            if env.isdir(&t) { format!("ls -lah {}", V::quote_path(&t)) } else { format!("cat {}", V::quote_path(&t)) }
        }
        "edit" => format!("${{EDITOR:-nano}} {}", file(req(st, "file")?, env)),
        "open" => {
            let t = req(st, "target")?.trim().to_string();
            if env.live && !env.exists(&t) && !t.contains(' ') && env.has(&t) {
                format!("setsid -f {} >/dev/null 2>&1", q(&t))
            } else {
                format!("xdg-open {}", file(&t, env))
            }
        }
        "backup" => {
            let f = req(st, "file")?.to_string();
            let a = file(&f, env);
            let b = env.file(f.trim());
            format!("cp -a {a} {}", V::quote_path(&format!("{b}.bak")))
        }
        "extract" => extract(st, env)?,
        "archive_list" => archive_list(st, env)?,
        "var_set" => {
            let (n, v) = assign(req(st, "name")?, req(st, "value")?, true);
            format!("export {n}={v}")
        }
        "path_add" => format!("export PATH={}:\"$PATH\"", dir(req(st, "dir")?, env)),
        "alias_make" => {
            let (n, v) = assign(req(st, "name")?, req(st, "cmd")?, false);
            format!("alias {n}={v}")
        }
        "calc" => {
            let e = V::expr(req(st, "expr")?);
            format!("awk 'BEGIN {{print {}}}'", if e.is_empty() { "0".to_string() } else { e })
        }
        "timer" => format!("sleep {} && {}", V::seconds(req(st, "in")?), if env.has("notify-send") { "notify-send 'Time is up'" } else { "echo 'Time is up'" }),
        "again" => last(env)?,
        "watch_last" => format!("watch -n {} {}", V::count(st.get("n"), 2), q(&last(env)?)),
        "time_last" => format!("time {}", last(env)?),
        x => return other(&format!("no builder for {x}")),
    })
}

const CREATES: &[(&str, &[&str])] = &[("touch", &["file"]), ("write", &["file"]), ("append", &["file"]), ("link", &["name"]),
    ("rename", &["new"]), ("mkdir", &["path"]), ("venv_make", &["path"]), ("git_clone", &["into"]), ("extract", &["into"]),
    ("download", &["to"]), ("edit", &["file"])];

fn single(lang: &Language, st: &Stage, env: &mut Env) -> R<String> {
    let Some(c) = lang.command(&st.name) else { return other(&format!("unknown stage {}", st.name)) };
    if let Some((_, slots)) = CREATES.iter().find(|(n, _)| *n == st.name) {
        for slot in slots.iter() {
            if let Some(v) = st.get(slot).filter(|v| !v.is_empty()) {
                env.exempt.insert(v.trim().to_string());
                env.exempt.insert(V::path(v, None));
            }
        }
    }
    if c.bash.is_none() {
        return built(&st.name, st, env);
    }
    fill(c, st, env)
}

// --------------------------------------------------------------------------- whole programs
pub fn compile_pipeline(lang: &Language, pipe: &Pipe, env: &mut Env) -> R<String> {
    if pipe.is_empty() {
        return other("empty pipeline");
    }
    env.level = if risk(lang, &vec![pipe.clone()]) == 0 { 2 } else { 1 };
    let sudo = pipe[0].name == "sudo";
    let pipe: &[Stage] = if sudo { &pipe[1..] } else { &pipe[..] };
    if pipe.is_empty() {
        return other("sudo alone");
    }
    let head = pipe[0].name.as_str();
    match head {
        "unknown" => Err(CErr::Unsupported),
        "pref" => Ok(String::new()),
        "files" | "paths" => files(pipe, env, sudo),
        h if LINE_SOURCES.contains(&h) => lines(pipe, env, sudo),
        "procs" | "port" => procs(pipe, env, sudo),
        _ => {
            let cmd = single(lang, &pipe[0], env)?;
            Ok(if sudo && !cmd.starts_with("sudo ") { format!("sudo {cmd}") } else { cmd })
        }
    }
}

pub fn compile_program(lang: &Language, program: &Program, env: &mut Env) -> R<String> {
    let mut out = Vec::new();
    for p in program {
        let c = compile_pipeline(lang, p, env)?;
        if !c.is_empty() {
            out.push(c);
        }
    }
    Ok(out.join(" && "))
}

fn word_label(name: &str) -> Option<&'static str> {
    Some(match name {
        "flat" => "top level only", "empty" => "empty", "uniq" => "no duplicates", "fshow" => "list", "lshow" => "show",
        "pshow" => "show", "fcount" => "count", "lcount" => "count lines", "pcount" => "count", "total" => "total size",
        "sizes" => "size of each", "delete" => "DELETE", "kill" => "KILL", "sudo" => "as root", "not" => "not", "procs" => "processes",
        "hidden" => "hidden", "isexec" => "executable", "details" => "list with details", "trash" => "move to trash",
        "fopen" => "open", "fedit" => "edit", "rn_lower" => "RENAME to lowercase", "rn_upper" => "RENAME to uppercase",
        "rn_spaces" => "RENAME spaces to underscores", "fhash" => "checksum", "by_ext" => "count by type", "dupes" => "find duplicates",
        "fnames" => "names", "fcat" => "contents", "wcount" => "count words", "ccount" => "count characters",
        "inplace" => "WRITE BACK to the file", "clip" => "copy to clipboard", "lfreq" => "most common first", "lnum" => "number lines",
        "lrev" => "reverse", "ldel" => "drop line", "lshuf" => "shuffle", "lupper" => "upper case", "llower" => "lower case", "lsum" => "add up",
        "lblank" => "drop blank lines", "history" => "command history", "env" => "environment", "pkgs" => "installed packages",
        "services" => "running services", "dmesg" => "kernel log", "clipboard" => "clipboard", "last" => "output of the last command",
        "again" => "run the last command again", "unknown" => "?",
        _ => return None,
    })
}

/// One-line plain-English reading of the program.
pub fn describe(lang: &Language, program: &Program) -> String {
    let mut parts = Vec::new();
    for pipe in program {
        let mut bits: Vec<String> = Vec::new();
        for st in pipe {
            let name = st.name.as_str();
            let g = |k: &str| st.get(k).unwrap_or("");
            if name == "files" {
                let w = if st.get("what") == Some("dirs") { "folders" } else { "files" };
                bits.push(match st.get("in").filter(|s| !s.is_empty()) { Some(i) => format!("{w} in {i}"), None => format!("{w} here") });
            } else if name == "paths" {
                bits.push(g("names").to_string());
            } else if name == "lines" {
                bits.push(format!("lines of {}", g("of")));
            } else if name == "pref" {
                bits.push(format!("preference: {} {}", g("key"), g("state")));
            } else if let Some(w) = word_label(name) {
                bits.push(format!("{w}{}", if st.get("force") == Some("hard") { " (force)" } else { "" }));
            } else if ["fsort", "lsort", "psort"].contains(&name) {
                bits.push(format!("sort by {} {}", g("key"), if g("order") == "desc" { "high to low" } else { "low to high" }));
            } else if ["ftake", "ltake", "ptake"].contains(&name) {
                bits.push(format!("first {}", V::count(st.get("n"), if name == "ltake" { 10 } else { 1 })));
            } else if name == "llast" {
                bits.push(format!("last {}", V::count(st.get("n"), 10)));
            } else if ["larger", "smaller", "newer", "older"].contains(&name) && !st.args.iter().any(|(_, v)| v.as_deref().map(|x| !x.is_empty()).unwrap_or(false)) {
                bits.push(match name { "larger" => "large", "smaller" => "small", "newer" => "recent", _ => "old" }.into());
            } else {
                let vals: Vec<&str> = st.args.iter().filter(|(k, v)| k != "sep" && v.as_deref().map(|x| !x.is_empty()).unwrap_or(false)).map(|(_, v)| v.as_deref().unwrap()).collect();
                let label = if stage_risk(lang, name) == 2 { name.replace('_', " ").to_uppercase() } else { name.replace('_', " ") };
                bits.push(if vals.is_empty() { label } else { format!("{label} {}", vals.join(" ")) });
            }
        }
        parts.push(bits.join(" > ").replace("not > ", "not ").replace("as root > ", "as root: "));
    }
    parts.join("  then  ")
}
