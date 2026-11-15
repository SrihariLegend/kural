//! Whole training requests: a port of kural/mix.py. Stand-alone commands from the phrasings in
//! kural/lang/commands.txt, requests about named files, the extra line and process requests, follow-ups that build
//! on the previous request, requests kural should decline, surface noise, and the overall mix.

use crate::gen::{self, file_filters, frame_stages};
use crate::pools::*;
use crate::req::*;
use crate::rng::Rng;
use crate::{ph, st};
use kural_engine::lang::{CommandDef, SlotKind};
use kural_engine::tok;
use std::collections::HashMap;
use std::sync::LazyLock;

fn k(v: u32) -> Option<V> {
    Some(V::K(v))
}
fn e(s: &'static str) -> Option<V> {
    Some(V::E(s))
}

// --------------------------------------------------------------------------- phrase templates
#[derive(Clone, Debug)]
enum Node {
    Lit(String),
    Slot(String),
    Alt(Vec<Vec<Node>>, bool),
}

/// "(a|b) [c] {slot}" -> nodes, exactly as mix.parse_template reads it.
fn parse_template(s: &str) -> Vec<Node> {
    let cs: Vec<char> = s.chars().collect();
    let mut pos = 0;
    fn seq(cs: &[char], pos: &mut usize, stop: Option<char>) -> Vec<Vec<Node>> {
        let (mut alts, mut cur, mut lit): (Vec<Vec<Node>>, Vec<Node>, String) = (Vec::new(), Vec::new(), String::new());
        fn flush(lit: &mut String, cur: &mut Vec<Node>) {
            let t = lit.trim();
            if !t.is_empty() {
                cur.push(Node::Lit(t.to_string()));
            }
            lit.clear();
        }
        while *pos < cs.len() {
            let c = cs[*pos];
            if stop.is_some() && Some(c) == stop {
                break;
            }
            if c == '|' && stop.is_some() {
                flush(&mut lit, &mut cur);
                alts.push(std::mem::take(&mut cur));
                *pos += 1;
            } else if c == '(' || c == '[' {
                flush(&mut lit, &mut cur);
                *pos += 1;
                let inner = seq(cs, pos, Some(if c == '(' { ')' } else { ']' }));
                *pos += 1;
                cur.push(Node::Alt(inner, c == '['));
            } else if c == '{' {
                flush(&mut lit, &mut cur);
                let j = (*pos..cs.len()).find(|&j| cs[j] == '}').unwrap_or(cs.len());
                cur.push(Node::Slot(cs[*pos + 1..j].iter().collect()));
                *pos = j + 1;
            } else {
                lit.push(c);
                *pos += 1;
            }
        }
        flush(&mut lit, &mut cur);
        alts.push(cur);
        alts
    }
    seq(&cs, &mut pos, None).into_iter().next().unwrap_or_default()
}

fn flatten(q: &mut Req, nodes: &[Node], out: &mut Vec<Node>) {
    for n in nodes {
        match n {
            Node::Alt(alts, optional) => {
                if *optional && q.f() < 0.5 {
                    continue;
                }
                let a = &alts[q.below(alts.len())];
                flatten(q, a, out);
            }
            other => out.push(other.clone()),
        }
    }
}

struct Cmds {
    list: Vec<&'static CommandDef>,
    by: HashMap<&'static str, &'static CommandDef>,
    templates: HashMap<&'static str, Vec<Vec<Node>>>,
    weights: Vec<f64>,
}

static CMDS: LazyLock<Cmds> = LazyLock::new(|| {
    let list: Vec<&'static CommandDef> = lang().l.commands.iter().collect();
    let by = list.iter().map(|c| (c.name.as_str(), *c)).collect();
    let templates = list.iter().map(|c| (c.name.as_str(), c.phrases.iter().map(|p| parse_template(p)).collect())).collect();
    let weights = list.iter().map(|c| weight(&c.name)).collect();
    Cmds { list, by, templates, weights }
});

fn weight(n: &str) -> f64 {
    match n {
        "pkg_install" => 3.0,
        "pkg_remove" | "git_commit" | "extract" | "download" | "edit" | "open" | "ls" => 2.0,
        "cd" | "mkdir" | "view" | "rename" => 2.5,
        "git_checkout" | "ssh" | "sys_update" | "disk_free" | "mem_free" | "svc_restart" | "svc_status" | "svc_start" | "svc_stop" | "again"
        | "git_status" | "touch" | "append" | "run_file" => 1.5,
        "follow" | "write" | "diff" | "git_log" | "ping" | "man" => 1.3,
        _ => 1.0,
    }
}

/// which kind of value to sample for a slot, where the table's type is too broad
fn kind_override(name: &str, slot: &str) -> Option<&'static str> {
    Some(match (name, slot) {
        ("pkg_info", "name") | ("pkg_has", "name") | ("pkg_files", "name") | ("pkg_search", "term") => "pkg",
        ("pkg_owner", "file") => "binfile",
        ("git_tag", "name") => "tag",
        ("alias_make", "name") => "alias",
        ("alias_make", "cmd") => "aliascmd",
        ("var_set", "value") => "varvalue",
        ("write", "text") | ("append", "text") => "sentence",
        ("write", "file") | ("append", "file") | ("view", "file") | ("page", "file") | ("edit", "file") | ("follow", "file")
        | ("git_blame", "file") | ("git_log_file", "file") | ("git_add", "file") | ("git_discard_file", "file") | ("diff", "a") | ("diff", "b") => "textfile",
        ("run_file", "file") | ("make_exec", "file") => "script_file",
        ("open", "target") => "target",
        ("path_add", "dir") => "bindir",
        ("venv_make", "path") | ("venv_on", "path") => "venv",
        ("tree", "depth") | ("watch_last", "n") => "smallcount",
        ("extract", "archive") | ("archive_list", "archive") => "archive",
        ("dk_build", "tag") => "image",
        ("git_commit", "msg") | ("git_commit_all", "msg") => "msg",
        _ => return None,
    })
}

const FOLDER_NOUNS: [&str; 3] = ["folder", "directory", "dir"];

/// One phrasing of a table command; its slot keys, or None if a required slot was left out.
fn say(q: &mut Req, name: &'static str) -> Option<Args> {
    let c = CMDS.by[name];
    let ts = &CMDS.templates[name];
    let t = &ts[q.below(ts.len())];
    let mut nodes = Vec::new();
    flatten(q, t, &mut nodes);
    let mut keys: HashMap<String, u32> = HashMap::new();
    let mut texts: HashMap<String, String> = HashMap::new();
    for i in 0..nodes.len() {
        let slot = match &nodes[i] {
            Node::Lit(s) => {
                let s = s.clone();
                q.lit(&s);
                continue;
            }
            Node::Slot(s) => s.clone(),
            Node::Alt(..) => unreachable!(),
        };
        if let Some(t) = texts.get(&slot) {
            let t = t.clone();
            q.lit(&t);
            continue;
        }
        let ty = c.slots.iter().find(|(n, _, _)| *n == slot).map(|(_, t, _)| t.as_str()).unwrap_or("text");
        let kind = kind_override(name, &slot).unwrap_or(ty);
        let (text, quote, pre, post) = val(q, kind);
        let before = if i > 0 {
            if let Node::Lit(s) = &nodes[i - 1] { s.split_whitespace().last().unwrap_or("").to_lowercase() } else { String::new() }
        } else {
            String::new()
        };
        let after = if i + 1 < nodes.len() {
            if let Node::Lit(s) = &nodes[i + 1] { s.split_whitespace().next().unwrap_or("").to_lowercase() } else { String::new() }
        } else {
            String::new()
        };
        let b = before.as_str();
        if !pre.is_empty() && !["the", "my", "a", "called", "named", "to", "in", "into", "at"].contains(&b) {
            q.lit(pre);
        } else if !pre.is_empty() && ["to", "in", "into", "at"].contains(&b) && !FOLDER_NOUNS.contains(&after.as_str()) {
            q.lit(pre);
        }
        let kk = q.arg_text(&text, quote);
        if !post.is_empty() && !FOLDER_NOUNS.contains(&after.as_str()) && (!pre.is_empty() || b == "the" || b == "my") {
            q.lit(post);
        }
        keys.insert(slot.clone(), kk);
        texts.insert(slot, text);
    }
    for (n, _, optional) in &c.slots {
        if !keys.contains_key(n) && !optional {
            return None;
        }
    }
    Some(c.slots.iter().map(|(n, _, _)| (slot_name(name, n), keys.get(n).map(|&x| V::K(x)))).collect())
}

fn slot_name(stage: &str, slot: &str) -> &'static str {
    let lg = lang();
    lg.l.stages[lg.stage_ix[stage]].slots.iter().find(|s| s.name == slot).map(|s| s.name.as_str()).unwrap()
}

pub fn command_request(q: &mut Req, name: Option<&'static str>) -> Result<Pipe, String> {
    let name: &'static str = match name {
        Some(n) => sname(n),
        None => CMDS.list[q.weighted(&CMDS.weights)].name.as_str(),
    };
    if q.f() < 0.22 {
        let p = q.c(POLITE);
        q.lit(p);
    }
    let mark = q.chunks.len();
    if name == "var_set" && q.f() < 0.35 {
        let kk = q.key();
        let w = q.c(&["export", "set", "", "export", "define"]);
        q.lit(w);
        let a = val(q, "envname").0;
        let b = val(q, "varvalue").0;
        q.chunks.push((format!("{a}={b}"), Some(kk)));
        return Ok(vec![(name, vec![("name", k(kk)), ("value", k(kk))])]);
    }
    if name == "alias_make" && q.f() < 0.4 {
        let a = val(q, "alias").0;
        let cmd = val(q, "aliascmd").0;
        q.lit("alias");
        if cmd.contains(' ') || q.f() < 0.5 {
            let (k1, k2) = (q.key(), q.key());
            q.chunks.push((format!("{a}="), Some(k1)));
            q.chunks.push((format!("'{cmd}'"), Some(k2)));
            return Ok(vec![(name, vec![("name", k(k1)), ("cmd", k(k2))])]);
        }
        let kk = q.key();
        q.chunks.push((format!("{a}={cmd}"), Some(kk)));
        return Ok(vec![(name, vec![("name", k(kk)), ("cmd", k(kk))])]);
    }
    for _ in 0..8 {
        q.chunks.truncate(mark);
        if let Some(args) = say(q, name) {
            if q.f() < 0.08 {
                let t = q.c(TAILS);
                q.lit(t);
            }
            return Ok(vec![(name, args)]);
        }
    }
    Err(format!("no phrasing of {name} has all its required slots"))
}

// --------------------------------------------------------------------------- files the user names outright
fn paths_request(q: &mut Req) -> Pipe {
    let folder = q.f() < 0.3;
    let n = q.pick(&[1usize, 1, 1, 1, 2, 2, 3]);
    let names: Vec<String> = (0..n).map(|_| if folder { q.cat(&[REAL_DIRS, &DIR_WORDS]).to_string() } else { q.filename(None) }).collect();
    let text = if n == 1 { names[0].clone() } else { pkg_list(q, &names) };
    let lead = if folder {
        q.c(&["", "", "", "the", "the folder", "the directory", "my"])
    } else {
        q.c(&["", "", "", "the file", "file", "my", if n > 1 { "the files" } else { "the file" }])
    };
    let sink = ["delete", "trash", "move", "copy", "compress", "sizes", "chmod", "chown", "upload"][q.weighted(&[30.0, 7.0, 16.0, 14.0, 8.0, 9.0, 6.0, 4.0, 6.0])];
    if q.f() < 0.22 {
        let p = q.c(POLITE);
        q.lit(p);
    }
    let mut key = 0u32;
    let mut target = |q: &mut Req| {
        q.lit(lead);
        key = q.arg_text(&text, false);
    };
    let mut args: Args = Vec::new();
    match sink {
        "delete" => {
            let v = q.c(DELETE);
            q.lit(v);
            target(q);
        }
        "trash" => {
            let form = q.below(4);
            q.lit(["trash", "move", "put", "send"][form]);
            target(q);
            q.lit(["", "to the trash", "in the bin", "to the trash"][form]);
        }
        "move" | "copy" => {
            let v = q.c(if sink == "move" { MOVE } else { COPY });
            q.lit(v);
            target(q);
            let p = q.c(&["to", "into", "to", "over to", "in"]);
            q.lit(p);
            if !folder && n == 1 && q.f() < 0.2 {
                let kk = q.key();
                let f = q.filename(None);
                q.chunks.push((f, Some(kk)));
                args.push(("to", k(kk)));
            } else {
                args.push(("to", k(put(q, "path"))));
            }
        }
        "compress" => {
            let fmt = q.c(&["tgz", "zip"]);
            let v = q.c(if fmt == "zip" { ZIP } else { TGZ });
            q.lit(v);
            target(q);
            args = vec![("fmt", e(fmt)), ("to", None)];
            if q.f() < 0.5 {
                let w = q.word();
                let ext = if fmt == "zip" { ".zip" } else { q.c(&[".tar.gz", ".tgz", ""]) };
                let name = format!("{w}{ext}");
                let p = q.c(&["into", "as", "to", "called", "named"]);
                q.lit(p);
                let kk = q.key();
                q.chunks.push((name, Some(kk)));
                args = vec![("fmt", e(fmt)), ("to", k(kk))];
            }
        }
        "sizes" => {
            let form = q.below(6);
            q.lit(["how big is", "size of", "how much space does", "how large is", "du", "what's the size of"][form]);
            target(q);
            let t = if form == 2 { q.c(&["take", "take up", "use"]) } else { "" };
            q.lit(t);
        }
        "chmod" => {
            let opts: Vec<&'static str> = MODE_WORDS.iter().copied().filter(|w| !["executable", "+x", "u+x"].contains(w)).collect();
            let m = q.c(&opts);
            let mk = q.key();
            match q.below(4) {
                0 => {
                    q.lit("make");
                    target(q);
                    q.chunks.push((m.to_string(), Some(mk)));
                }
                1 => {
                    q.lit("chmod");
                    q.chunks.push((m.to_string(), Some(mk)));
                    target(q);
                }
                2 => {
                    let w = q.c(&["set the permissions of", "change the permissions of", "change the mode of"]);
                    q.lit(w);
                    target(q);
                    q.lit("to");
                    q.chunks.push((m.to_string(), Some(mk)));
                }
                _ => {
                    q.lit("set");
                    target(q);
                    q.lit("to");
                    q.chunks.push((m.to_string(), Some(mk)));
                }
            }
            args = vec![("mode", k(mk))];
        }
        "chown" => {
            let o = q.c(OWNERS);
            let ok = q.key();
            match q.below(3) {
                0 => {
                    let w = q.c(&["change the owner of", "set the owner of", "change ownership of"]);
                    q.lit(w);
                    target(q);
                    q.lit("to");
                    q.chunks.push((o.to_string(), Some(ok)));
                }
                1 => {
                    q.lit("chown");
                    q.chunks.push((o.to_string(), Some(ok)));
                    target(q);
                }
                _ => {
                    q.lit("give");
                    target(q);
                    q.lit("to");
                    q.chunks.push((o.to_string(), Some(ok)));
                }
            }
            args = vec![("owner", k(ok))];
        }
        _ => {
            let v = q.c(&["upload", "send", "rsync", "scp", "copy", "sync", "push", "transfer", "mirror", "back up", "ship"]);
            q.lit(v);
            target(q);
            let p = q.c(&["to", "to", "over to", "to the server"]);
            q.lit(p);
            args = vec![("to", k(put(q, "remote")))];
        }
    }
    if q.f() < 0.08 {
        let t = q.c(TAILS);
        q.lit(t);
    }
    vec![st!("paths", "names" => k(key)), (sinkname(sink), args)]
}

fn sinkname(s: &str) -> &'static str {
    sname(s)
}

// --------------------------------------------------------------------------- more line requests
fn sep_words(s: &str) -> &'static [&'static str] {
    match s {
        "space" => &["", "", "separated by spaces", "whitespace separated", "split on spaces"],
        "comma" => &["separated by commas", "comma separated", "using , as the delimiter", "split on commas", "(comma delimited)"],
        "tab" => &["tab separated", "separated by tabs", "split on tabs", "(tab delimited)"],
        "colon" => &["separated by colons", "using : as the delimiter", "colon delimited", "split on :"],
        "semi" => &["separated by semicolons", "semicolon delimited", "using ; as the separator"],
        _ => &["separated by pipes", "pipe delimited", "using | as the separator"],
    }
}
const INPLACE: &[&str] = &["in place", "in-place", "and save it back", "and overwrite the file", "and write it back to the file", "permanently",
    "and update the file", "and save the file", "for good", "and save the changes"];
const ORD: &[&str] = &["1st", "2nd", "3rd", "4th", "5th", "first", "second", "third", "fourth", "fifth", "sixth"];

type A3 = (String, u32, bool);

fn t3(q: &mut Req) -> A3 {
    let (t, quote, _, _) = val(q, "text");
    let kk = q.key();
    (t, kk, quote)
}

fn lines2_request(q: &mut Req) -> Pipe {
    let src = ["file", "history", "env", "pkgs", "services", "journal", "dmesg", "clipboard"][q.weighted(&[44.0, 13.0, 7.0, 10.0, 6.0, 10.0, 6.0, 4.0])];
    if q.f() < 0.22 {
        let p = q.c(POLITE);
        q.lit(p);
    }
    if src == "file" {
        return lines2_file(q);
    }
    let mut pipe: Pipe;
    match src {
        "history" => {
            pipe = vec![st!("history")];
            let kind = q.c(&["show", "match", "match", "match", "last", "count", "matchlast"]);
            let cmd = q.cat(&[&CMDNAMES, &PKGS[..20], &["docker run", "git push", "ssh", "sudo"]]);
            let tk = q.key();
            let tt = (cmd.to_string(), tk, cmd.contains(' '));
            let n = (q.count(), q.key());
            match kind {
                "show" => {
                    let w = q.c(&["show my command history", "history", "what commands did i run", "list my previous commands", "show shell history",
                        "show my bash history", "what have i run so far", "list the commands i ran"]);
                    q.lit(w);
                }
                "match" => {
                    let p = match q.below(12) {
                        0 => ph!["search my history for", &tt],
                        1 => ph!["history | grep", &tt],
                        2 => ph!["find", &tt, "in my history"],
                        3 => ph!["what", &tt, "commands did i run"],
                        4 => ph!["commands i ran with", &tt, "in them"],
                        5 => ph!["when did i last run", &tt],
                        6 => ph!["show history entries with", &tt],
                        7 => ph!["grep my history for", &tt],
                        8 => ph!["did i run", &tt, "before"],
                        9 => ph!["look through my history for", &tt],
                        10 => ph!["how did i run", &tt, "last time"],
                        _ => ph!["what was that", &tt, "command i ran"],
                    };
                    q.emit(&p);
                    pipe.push(st!("match", "text" => k(tk)));
                }
                "last" => {
                    let p = match q.below(5) {
                        0 => ph!["last", &n, "commands"],
                        1 => ph!["show my last", &n, "commands"],
                        2 => ph!["the", &n, "most recent commands i ran"],
                        3 => ph!["previous", &n, "commands"],
                        _ => ph!["what were my last", &n, "commands"],
                    };
                    q.emit(&p);
                    pipe.push(st!("llast", "n" => k(n.1)));
                }
                "count" => {
                    let p = match q.below(3) {
                        0 => ph!["how many times did i run", &tt],
                        1 => ph!["count", &tt, "commands in my history"],
                        _ => ph!["how often have i run", &tt],
                    };
                    q.emit(&p);
                    pipe.push(st!("match", "text" => k(tk)));
                    pipe.push(st!("lcount"));
                    return pipe;
                }
                _ => {
                    let p = match q.below(3) {
                        0 => ph!["last", &n, &tt, "commands"],
                        1 => ph!["my last", &n, "commands with", &tt],
                        _ => ph!["the", &n, "most recent", &tt, "commands i ran"],
                    };
                    q.emit(&p);
                    pipe.push(st!("match", "text" => k(tk)));
                    pipe.push(st!("llast", "n" => k(n.1)));
                }
            }
        }
        "env" => {
            pipe = vec![st!("env")];
            let kind = q.c(&["show", "show", "match", "match", "count", "save"]);
            let t = q.cat(&[&ENVNAMES, &["proxy", "java", "path", "key", "token", "home", "xdg", "lang", "python"]]);
            let tk = q.key();
            let tt = (t.to_string(), tk);
            match kind {
                "show" => {
                    let w = q.c(&["show environment variables", "list env vars", "printenv", "show all environment variables", "env",
                        "what environment variables are set", "list my environment", "show my env", "dump the environment"]);
                    q.lit(w);
                }
                "match" => {
                    let p = match q.below(8) {
                        0 => ph!["env vars containing", &tt],
                        1 => ph!["environment variables with", &tt, "in them"],
                        2 => ph!["show env vars matching", &tt],
                        3 => ph!["is there an env var with", &tt],
                        4 => ph!["env | grep", &tt],
                        5 => ph!["which environment variables mention", &tt],
                        6 => ph!["find environment variables related to", &tt],
                        _ => ph!["any env vars about", &tt],
                    };
                    q.emit(&p);
                    pipe.push(st!("match", "text" => k(tk)));
                }
                "count" => {
                    let w = q.c(&["how many environment variables are set", "count my env vars", "number of environment variables"]);
                    q.lit(w);
                    pipe.push(st!("lcount"));
                    return pipe;
                }
                _ => {
                    let out = q.filename(Some(&["txt", "env", "log"]));
                    let ok = q.key();
                    let w = q.c(&["save my environment to", "dump env vars into", "write all environment variables to"]);
                    q.emit(&ph![w, (out, ok)]);
                    pipe.push(st!("save", "to" => k(ok)));
                    return pipe;
                }
            }
        }
        "pkgs" => {
            pipe = vec![st!("pkgs")];
            let kind = q.c(&["show", "match", "match", "count", "count", "save"]);
            let t = q.cat(&[&PKGS, &["python", "lib", "font", "qt", "gtk", "kde", "linux", "nvidia", "perl", "rust"]]);
            let tk = q.key();
            let tt = (t.to_string(), tk);
            match kind {
                "show" => {
                    let w = q.c(&["list installed packages", "what packages are installed", "show all installed packages", "installed packages",
                        "what do i have installed", "list everything that is installed", "show my installed packages"]);
                    q.lit(w);
                }
                "match" => {
                    let p = match q.below(8) {
                        0 => ph!["which installed packages have", &tt, "in the name"],
                        1 => ph!["installed packages matching", &tt],
                        2 => ph!["list installed packages containing", &tt],
                        3 => ph!["what", &tt, "packages do i have installed"],
                        4 => ph!["show installed packages with", &tt],
                        5 => ph!["do i have any", &tt, "packages installed"],
                        6 => ph!["installed packages related to", &tt],
                        _ => ph!["which", &tt, "packages are installed"],
                    };
                    q.emit(&p);
                    pipe.push(st!("match", "text" => k(tk)));
                }
                "count" => {
                    let w = q.c(&["how many packages are installed", "count installed packages", "number of installed packages", "how many packages do i have",
                        "how many programs are installed", "total number of installed packages"]);
                    q.lit(w);
                    pipe.push(st!("lcount"));
                    return pipe;
                }
                _ => {
                    let out = q.filename(Some(&["txt", "list", "log"]));
                    let ok = q.key();
                    let w = q.c(&["save the list of installed packages to", "export installed packages to", "write my package list to",
                        "dump the installed packages into"]);
                    q.emit(&ph![w, (out, ok)]);
                    pipe.push(st!("save", "to" => k(ok)));
                    return pipe;
                }
            }
        }
        "services" => {
            pipe = vec![st!("services")];
            let kind = q.c(&["show", "show", "match", "count"]);
            let t = q.cat(&[&UNITS, &["network", "blue", "print", "sound", "docker", "ssh"]]);
            let tk = q.key();
            let tt = (t.to_string(), tk);
            match kind {
                "show" => {
                    let w = q.c(&["list running services", "what services are running", "show running services", "which services are active",
                        "running systemd services", "show all active services", "what daemons are running", "list active services"]);
                    q.lit(w);
                }
                "match" => {
                    let p = match q.below(5) {
                        0 => ph!["running services matching", &tt],
                        1 => ph!["is there a running service with", &tt, "in the name"],
                        2 => ph!["which running services mention", &tt],
                        3 => ph!["list running services containing", &tt],
                        _ => ph!["any running services about", &tt],
                    };
                    q.emit(&p);
                    pipe.push(st!("match", "text" => k(tk)));
                }
                _ => {
                    let w = q.c(&["how many services are running", "count running services", "number of active services"]);
                    q.lit(w);
                    pipe.push(st!("lcount"));
                    return pipe;
                }
            }
        }
        "journal" => {
            let mut unit = None;
            let kind = q.c(&["show", "last", "last", "match"]);
            let n = (q.count(), q.key());
            let tt = t3(q);
            if q.f() < 0.7 {
                let u = val(q, "unit").0;
                let uk = q.key();
                unit = Some(uk);
                let uu = (u, uk);
                let p = match kind {
                    "show" => match q.below(10) {
                        0 => ph!["show the logs of", &uu],
                        1 => ph![&uu, "logs"],
                        2 => ph!["logs for the", &uu, "service"],
                        3 => ph!["what is", &uu, "logging"],
                        4 => ph!["show me the", &uu, "service logs"],
                        5 => ph!["journalctl -u", &uu],
                        6 => ph!["see the journal for", &uu],
                        7 => ph!["show logs from", &uu],
                        8 => ph!["what did", &uu, "log"],
                        _ => ph!["check the", &uu, "logs"],
                    },
                    "last" => match q.below(5) {
                        0 => ph!["last", &n, "lines of the", &uu, "logs"],
                        1 => ph!["show the last", &n, "log lines for", &uu],
                        2 => ph!["the", &n, "most recent", &uu, "log entries"],
                        3 => ph!["tail the", &uu, "logs", &n, "lines"],
                        _ => ph!["show the", &n, "latest log lines from", &uu],
                    },
                    _ => match q.below(5) {
                        0 => ph![&uu, "logs mentioning", &tt],
                        1 => ph!["search the", &uu, "logs for", &tt],
                        2 => ph!["lines with", &tt, "in the", &uu, "logs"],
                        3 => ph!["grep the", &uu, "logs for", &tt],
                        _ => ph!["find", &tt, "in the logs of", &uu],
                    },
                };
                q.emit(&p);
            } else {
                match kind {
                    "show" => {
                        let w = q.c(&["show the system log", "system journal", "show the journal", "journalctl", "what's in the system log",
                            "show system logs", "show me the journal from this boot"]);
                        q.lit(w);
                    }
                    "last" => {
                        let p = match q.below(3) {
                            0 => ph!["last", &n, "lines of the system log"],
                            1 => ph!["show the last", &n, "journal entries"],
                            _ => ph!["the", &n, "latest system log lines"],
                        };
                        q.emit(&p);
                    }
                    _ => {
                        let p = match q.below(5) {
                            0 => ph!["search the journal for", &tt],
                            1 => ph!["journal entries mentioning", &tt],
                            2 => ph!["find", &tt, "in the system log"],
                            3 => ph!["system log lines with", &tt],
                            _ => ph!["grep the journal for", &tt],
                        };
                        q.emit(&p);
                    }
                }
            }
            pipe = vec![st!("journal", "unit" => unit.map(V::K))];
            if kind == "last" {
                pipe.push(st!("llast", "n" => k(n.1)));
            } else if kind == "match" {
                pipe.push(st!("match", "text" => k(tt.1)));
            }
        }
        "dmesg" => {
            pipe = vec![st!("dmesg")];
            let kind = q.c(&["show", "match", "match", "last"]);
            let t = q.c(&["usb", "error", "nvidia", "wifi", "bluetooth", "ata", "thermal", "oom", "firmware", "acpi", "nvme", "i915", "fail"]);
            let tk = q.key();
            let tt = (t.to_string(), tk);
            let n = (q.count(), q.key());
            match kind {
                "show" => {
                    let w = q.c(&["show kernel messages", "dmesg", "kernel log", "show the kernel log", "what is the kernel saying",
                        "show the kernel ring buffer", "print kernel messages"]);
                    q.lit(w);
                }
                "match" => {
                    let p = match q.below(7) {
                        0 => ph!["kernel messages about", &tt],
                        1 => ph!["dmesg | grep", &tt],
                        2 => ph!["search the kernel log for", &tt],
                        3 => ph!["kernel log lines mentioning", &tt],
                        4 => ph!["any", &tt, "messages in dmesg"],
                        5 => ph!["what does the kernel say about", &tt],
                        _ => ph!["grep dmesg for", &tt],
                    };
                    q.emit(&p);
                    pipe.push(st!("match", "text" => k(tk)));
                }
                _ => {
                    let p = match q.below(4) {
                        0 => ph!["last", &n, "kernel messages"],
                        1 => ph!["the", &n, "most recent kernel messages"],
                        2 => ph!["tail of dmesg", &n, "lines"],
                        _ => ph!["show the last", &n, "lines of the kernel log"],
                    };
                    q.emit(&p);
                    pipe.push(st!("llast", "n" => k(n.1)));
                }
            }
        }
        _ => {
            pipe = vec![st!("clipboard")];
            match q.c(&["show", "show", "save", "count", "words"]) {
                "show" => {
                    let w = q.c(&["show what's in the clipboard", "paste the clipboard", "print my clipboard", "what's on my clipboard",
                        "clipboard contents", "what did i copy", "show the clipboard"]);
                    q.lit(w);
                }
                "save" => {
                    let out = q.filename(Some(&["txt", "md", "log", "json"]));
                    let ok = q.key();
                    let w = q.c(&["save the clipboard to", "paste the clipboard into", "write what i copied to", "dump the clipboard into", "put my clipboard in"]);
                    q.emit(&ph![w, (out, ok)]);
                    pipe.push(st!("save", "to" => k(ok)));
                    return pipe;
                }
                "count" => {
                    let w = q.c(&["how many lines are in the clipboard", "count lines in my clipboard", "how many lines did i copy"]);
                    q.lit(w);
                    pipe.push(st!("lcount"));
                    return pipe;
                }
                _ => {
                    let w = q.c(&["how many words are in my clipboard", "word count of the clipboard", "count the words i copied"]);
                    q.lit(w);
                    pipe.push(st!("wcount"));
                    return pipe;
                }
            }
        }
    }
    // a common tail for the listing sources
    let extra = q.f();
    if extra < 0.07 {
        let out = q.filename(Some(&["txt", "log", "out"]));
        let ok = q.key();
        let w = q.c(&["and save it to", "and write that to", "into", "and put it in", "> "]);
        q.emit(&ph![w, (out, ok)]);
        pipe.push(st!("save", "to" => k(ok)));
    } else if extra < 0.13 {
        let w = q.c(&["and count them", "and tell me how many", "- how many are there", "and count"]);
        q.lit(w);
        pipe.push(st!("lcount"));
    } else if extra < 0.17 {
        let w = q.c(&["sorted", "in alphabetical order", "and sort them"]);
        q.lit(w);
        pipe.push(("lsort", vec![("key", e("alpha")), ("order", e("asc"))]));
        pipe.push(st!("lshow"));
    } else if extra < 0.2 {
        let w = q.c(&["and copy it to the clipboard", "and copy that"]);
        q.lit(w);
        pipe.push(st!("clip"));
    } else {
        pipe.push(st!("lshow"));
    }
    pipe
}

fn lines2_file(q: &mut Req) -> Pipe {
    let f = q.filename(Some(&["log", "txt", "csv", "md", "json", "conf", "py", "sh", "out", "tsv", "yaml", "list", "dat"]));
    let fk = q.key();
    let fa = (f.clone(), fk);
    let mut pipe: Pipe = vec![st!("lines", "of" => k(fk))];
    let kind = q.c(&["range", "line", "col", "col", "replace", "replace", "replace_show", "freq", "num", "rev", "shuf", "upper", "lower",
        "sum", "blank", "wcount", "wcount", "ccount", "clip", "inplace", "inplace", "append_out", "random", "countmatch",
        "countmatch", "ldel", "ldel"]);
    match kind {
        "range" => {
            let x = q.randint(1, 399);
            let mut y = q.randint(1, 398);
            if y >= x {
                y += 1;
            }
            let (a, b) = (x.min(y), x.max(y));
            let (ka, mut kb) = (q.key(), q.key());
            if q.f() < 0.18 {
                let glued = (format!("{a}-{b}"), ka);
                let p = match q.below(4) {
                    0 => ph!["show lines", &glued, "of", &fa],
                    1 => ph!["lines", &glued, "of", &fa],
                    2 => ph!["print lines", &glued, "from", &fa],
                    _ => ph![&fa, "lines", &glued],
                };
                q.emit(&p);
                kb = ka;
            } else {
                let (aa, bb) = ((a.to_string(), ka), (b.to_string(), kb));
                let p = match q.below(8) {
                    0 => ph!["show lines", &aa, "to", &bb, "of", &fa],
                    1 => ph!["lines", &aa, "through", &bb, "of", &fa],
                    2 => ph!["print lines", &aa, "to", &bb, "from", &fa],
                    3 => ph!["from line", &aa, "to line", &bb, "of", &fa],
                    4 => ph![&fa, "lines", &aa, "to", &bb],
                    5 => ph!["show", &fa, "from line", &aa, "to", &bb],
                    6 => ph!["give me lines", &aa, "to", &bb, "in", &fa],
                    _ => ph!["what's on lines", &aa, "through", &bb, "of", &fa],
                };
                q.emit(&p);
            }
            pipe.push(st!("lrange", "a" => k(ka), "b" => k(kb)));
            pipe.push(st!("lshow"));
        }
        "line" => {
            let t = if q.f() < 0.7 { q.randint(1, 500).to_string() } else { q.c(ORD).to_string() };
            let n = (t, q.key());
            let p = match q.below(8) {
                0 => ph!["show line", &n, "of", &fa],
                1 => ph!["what's on line", &n, "of", &fa],
                2 => ph!["line", &n, "of", &fa],
                3 => ph!["print line", &n, "from", &fa],
                4 => ph!["print the", &n, "line of", &fa],
                5 => ph!["what is line", &n, "in", &fa],
                6 => ph!["show me line", &n, "in", &fa],
                _ => ph![&fa, "line", &n],
            };
            q.emit(&p);
            pipe.push(st!("lline", "n" => k(n.1)));
            pipe.push(st!("lshow"));
        }
        "col" => {
            let t = if q.f() < 0.6 { q.randint(1, 9).to_string() } else { q.c(&ORD[..10]).to_string() };
            let n = (t, q.key());
            let mut sep = q.c(&["space", "space", "comma", "comma", "tab", "colon", "semi", "bar"]);
            let mut said = q.c(sep_words(sep));
            if f.ends_with(".csv") && q.f() < 0.7 {
                sep = "comma";
                said = q.c(&["", "", "comma separated"]);
            } else if f.ends_with(".tsv") && q.f() < 0.7 {
                sep = "tab";
                said = "";
            } else if f == "/etc/passwd" {
                sep = "colon";
                said = q.c(&["", "separated by colons"]);
            }
            let p = match q.below(9) {
                0 => ph!["print column", &n, "of", &fa, said],
                1 => ph!["show the", &n, "column of", &fa, said],
                2 => ph!["column", &n, "of", &fa, said],
                3 => ph!["get field", &n, "from", &fa, said],
                4 => ph!["cut column", &n, "from", &fa, said],
                5 => ph!["the", &n, "field of each line in", &fa, said],
                6 => ph!["extract column", &n, "from", &fa, said],
                7 => ph!["show only the", &n, "column in", &fa, said],
                _ => ph!["field", &n, "of", &fa, said],
            };
            q.emit(&p);
            pipe.push(st!("col", "n" => k(n.1), "sep" => e(sep)));
            let extra = q.f();
            if extra < 0.12 {
                let w = q.c(&["and add it up", "and sum it", "and total it", "summed", "and add them up"]);
                q.lit(w);
                pipe.push(st!("lsum"));
                pipe.push(st!("lshow"));
            } else if extra < 0.22 {
                let w = q.c(&["sorted", "and sort it", "in sorted order"]);
                q.lit(w);
                pipe.push(("lsort", vec![("key", e("alpha")), ("order", e("asc"))]));
                pipe.push(st!("lshow"));
            } else if extra < 0.3 {
                let w = q.c(&["unique values only", "without duplicates", "deduplicated"]);
                q.lit(w);
                pipe.push(st!("uniq"));
                pipe.push(st!("lshow"));
            } else if extra < 0.38 {
                let w = q.c(&["and count how often each value appears", "with counts", "and count each value"]);
                q.lit(w);
                pipe.push(st!("lfreq"));
                pipe.push(st!("lshow"));
            } else {
                pipe.push(st!("lshow"));
            }
        }
        "replace" | "replace_show" => {
            let o = val(q, "text");
            let nn = val(q, "text");
            let (ok, nk) = (q.key(), q.key());
            let (oo, nw) = ((o.0, ok, o.1), (nn.0, nk, nn.1));
            if kind == "replace" {
                let p = match q.below(10) {
                    0 => ph!["replace", &oo, "with", &nw, "in", &fa],
                    1 => ph!["change", &oo, "to", &nw, "in", &fa],
                    2 => ph!["swap", &oo, "for", &nw, "in", &fa],
                    3 => ph!["substitute", &nw, "for", &oo, "in", &fa],
                    4 => ph!["in", &fa, "replace", &oo, "with", &nw],
                    5 => ph!["find and replace", &oo, "with", &nw, "in", &fa],
                    6 => ph!["replace every", &oo, "with", &nw, "in", &fa],
                    7 => ph!["change all occurrences of", &oo, "to", &nw, "in", &fa],
                    8 => ph!["turn every", &oo, "into", &nw, "in", &fa],
                    _ => ph!["rename", &oo, "to", &nw, "everywhere in", &fa],
                };
                q.emit(&p);
                pipe.push(st!("lreplace", "old" => k(ok), "new" => k(nk)));
                pipe.push(st!("inplace"));
            } else {
                let p = match q.below(6) {
                    0 => ph!["show", &fa, "with", &oo, "replaced by", &nw],
                    1 => ph!["print", &fa, "replacing", &oo, "with", &nw],
                    2 => ph!["cat", &fa, "but with", &oo, "changed to", &nw],
                    3 => ph![&fa, "with", &oo, "swapped for", &nw],
                    4 => ph!["show what", &fa, "looks like with", &oo, "replaced by", &nw],
                    _ => ph!["preview", &fa, "with", &oo, "changed to", &nw],
                };
                q.emit(&p);
                pipe.push(st!("lreplace", "old" => k(ok), "new" => k(nk)));
                pipe.push(st!("lshow"));
            }
        }
        "freq" => {
            if q.f() < 0.35 {
                let n = (q.count(), q.key());
                let p = match q.below(3) {
                    0 => ph!["top", &n, "most frequent lines in", &fa],
                    1 => ph!["the", &n, "most common lines in", &fa],
                    _ => ph!["show the", &n, "most repeated lines of", &fa],
                };
                q.emit(&p);
                pipe.push(st!("lfreq"));
                pipe.push(st!("ltake", "n" => k(n.1)));
                pipe.push(st!("lshow"));
            } else {
                let p = match q.below(8) {
                    0 => ph!["most common lines in", &fa],
                    1 => ph!["count how often each line appears in", &fa],
                    2 => ph!["frequency of each line in", &fa],
                    3 => ph!["which lines repeat the most in", &fa],
                    4 => ph!["count duplicates in", &fa],
                    5 => ph!["histogram of lines in", &fa],
                    6 => ph!["how many times does each line occur in", &fa],
                    _ => ph!["tally the lines of", &fa],
                };
                q.emit(&p);
                pipe.push(st!("lfreq"));
                pipe.push(st!("lshow"));
            }
        }
        "num" => {
            let p = match q.below(7) {
                0 => ph!["show", &fa, "with line numbers"],
                1 => ph!["number the lines of", &fa],
                2 => ph!["cat -n", &fa],
                3 => ph!["print", &fa, "with line numbers"],
                4 => ph![&fa, "with numbered lines"],
                5 => ph!["show line numbers for", &fa],
                _ => ph!["display", &fa, "with each line numbered"],
            };
            q.emit(&p);
            pipe.push(st!("lnum"));
            pipe.push(st!("lshow"));
        }
        "rev" => {
            let p = match q.below(8) {
                0 => ph!["reverse the lines of", &fa],
                1 => ph!["show", &fa, "backwards"],
                2 => ph!["print", &fa, "in reverse line order"],
                3 => ph!["tac", &fa],
                4 => ph![&fa, "from bottom to top"],
                5 => ph!["flip", &fa, "upside down"],
                6 => ph!["show", &fa, "last line first"],
                _ => ph!["reverse", &fa],
            };
            q.emit(&p);
            pipe.push(st!("lrev"));
            pipe.push(st!("lshow"));
        }
        "shuf" => {
            if q.f() < 0.4 {
                let n = (q.count(), q.key());
                let p = match q.below(4) {
                    0 => ph!["pick", &n, "random lines from", &fa],
                    1 => ph!["give me", &n, "random lines of", &fa],
                    2 => ph![&n, "random lines from", &fa],
                    _ => ph!["sample", &n, "lines from", &fa],
                };
                q.emit(&p);
                pipe.push(st!("lshuf"));
                pipe.push(st!("ltake", "n" => k(n.1)));
                pipe.push(st!("lshow"));
            } else {
                let p = match q.below(7) {
                    0 => ph!["shuffle the lines of", &fa],
                    1 => ph!["randomize the order of lines in", &fa],
                    2 => ph![&fa, "in random order"],
                    3 => ph!["shuf", &fa],
                    4 => ph!["scramble the lines of", &fa],
                    5 => ph!["shuffle", &fa],
                    _ => ph!["mix up the lines in", &fa],
                };
                q.emit(&p);
                pipe.push(st!("lshuf"));
                pipe.push(st!("lshow"));
            }
        }
        "upper" | "lower" => {
            let w = if kind == "upper" { q.c(&["uppercase", "upper case", "all caps", "capitals"]) } else { q.c(&["lowercase", "lower case", "all lowercase", "small letters"]) };
            let p = match q.below(7) {
                0 => ph!["convert", &fa, "to", w],
                1 => ph![&fa, "in", w],
                2 => ph!["make", &fa, "all", w],
                3 => ph!["print", &fa, "in", w],
                4 => ph!["show", &fa, "in", w],
                5 => ph!["turn the text of", &fa, "into", w],
                _ => ph![w, "version of", &fa],
            };
            q.emit(&p);
            pipe.push(st!(if kind == "upper" { "lupper" } else { "llower" }));
            pipe.push(st!("lshow"));
        }
        "sum" => {
            let p = match q.below(7) {
                0 => ph!["add up the numbers in", &fa],
                1 => ph!["sum", &fa],
                2 => ph!["total of the numbers in", &fa],
                3 => ph!["sum all the values in", &fa],
                4 => ph!["what do the numbers in", &fa, "add up to"],
                5 => ph!["add up every line of", &fa],
                _ => ph!["sum the numbers in", &fa],
            };
            q.emit(&p);
            pipe.push(st!("lsum"));
            pipe.push(st!("lshow"));
        }
        "blank" => {
            let p = match q.below(8) {
                0 => ph!["show", &fa, "without blank lines"],
                1 => ph![&fa, "with empty lines removed"],
                2 => ph!["print", &fa, "skipping blank lines"],
                3 => ph!["hide empty lines in", &fa],
                4 => ph!["remove blank lines from", &fa],
                5 => ph!["strip empty lines from", &fa],
                6 => ph!["drop the blank lines of", &fa],
                _ => ph!["squeeze out empty lines in", &fa],
            };
            q.emit(&p);
            pipe.push(st!("lblank"));
            if q.f() < 0.3 {
                let w = q.c(INPLACE);
                q.lit(w);
                pipe.push(st!("inplace"));
            } else {
                pipe.push(st!("lshow"));
            }
        }
        "wcount" => {
            let p = match q.below(9) {
                0 => ph!["how many words in", &fa],
                1 => ph!["word count of", &fa],
                2 => ph!["count the words in", &fa],
                3 => ph!["wc -w", &fa],
                4 => ph!["number of words in", &fa],
                5 => ph!["how many words does", &fa, "have"],
                6 => ph!["count words in", &fa],
                7 => ph!["how many words are in", &fa],
                _ => ph!["words in", &fa],
            };
            q.emit(&p);
            pipe.push(st!("wcount"));
        }
        "ccount" => {
            let p = match q.below(8) {
                0 => ph!["how many characters in", &fa],
                1 => ph!["character count of", &fa],
                2 => ph!["count the characters in", &fa],
                3 => ph!["number of chars in", &fa],
                4 => ph!["how long is", &fa, "in characters"],
                5 => ph!["wc -m", &fa],
                6 => ph!["how many letters are in", &fa],
                _ => ph!["count chars in", &fa],
            };
            q.emit(&p);
            pipe.push(st!("ccount"));
        }
        "clip" => {
            let p = match q.below(7) {
                0 => ph!["copy", &fa, "to the clipboard"],
                1 => ph!["copy the contents of", &fa, "to my clipboard"],
                2 => ph!["put", &fa, "on the clipboard"],
                3 => ph!["copy what's in", &fa],
                4 => ph![&fa, "to clipboard"],
                5 => ph!["copy the text of", &fa],
                _ => ph!["send", &fa, "to the clipboard"],
            };
            q.emit(&p);
            pipe.push(st!("clip"));
        }
        "inplace" => {
            let op = q.c(&["nomatch", "nomatch", "match", "uniq", "sort", "sortu", "lower", "upper"]);
            match op {
                "nomatch" | "match" => {
                    let tt = t3(q);
                    let p = if op == "nomatch" {
                        match q.below(6) {
                            0 => ph!["remove lines containing", &tt, "from", &fa],
                            1 => ph!["delete every line with", &tt, "in", &fa],
                            2 => ph!["drop lines mentioning", &tt, "from", &fa],
                            3 => ph!["strip lines with", &tt, "out of", &fa],
                            4 => ph!["delete all lines that contain", &tt, "from", &fa],
                            _ => ph!["get rid of lines with", &tt, "in", &fa],
                        }
                    } else {
                        match q.below(4) {
                            0 => ph!["keep only lines containing", &tt, "in", &fa],
                            1 => ph!["delete every line of", &fa, "that doesn't contain", &tt],
                            2 => ph!["remove all lines without", &tt, "from", &fa],
                            _ => ph!["keep only the lines with", &tt, "in", &fa],
                        }
                    };
                    q.emit(&p);
                    pipe.push((sname(op), vec![("text", k(tt.1))]));
                }
                "uniq" => {
                    let p = match q.below(4) {
                        0 => ph!["remove duplicate lines from", &fa],
                        1 => ph!["dedupe", &fa],
                        2 => ph!["deduplicate", &fa],
                        _ => ph!["delete repeated lines in", &fa],
                    };
                    q.emit(&p);
                    pipe.push(st!("uniq"));
                }
                "sort" | "sortu" => {
                    let p = match q.below(4) {
                        0 => ph!["sort", &fa],
                        1 => ph!["sort the lines of", &fa],
                        2 => ph!["put the lines of", &fa, "in order"],
                        _ => ph!["alphabetize", &fa],
                    };
                    q.emit(&p);
                    pipe.push(("lsort", vec![("key", e("alpha")), ("order", e("asc"))]));
                    if op == "sortu" {
                        let w = q.c(&["and remove duplicates", "and dedupe", "without duplicates", "and drop repeats"]);
                        q.lit(w);
                        pipe.push(st!("uniq"));
                    }
                }
                _ => {
                    let w = if op == "upper" { q.c(&["uppercase", "all caps"]) } else { q.c(&["lowercase", "all lowercase"]) };
                    let p = if q.below(2) == 0 { ph!["convert", &fa, "to", w] } else { ph!["make", &fa, "all", w] };
                    q.emit(&p);
                    pipe.push(st!(if op == "upper" { "lupper" } else { "llower" }));
                }
            }
            let w = q.c(INPLACE);
            q.lit(w);
            pipe.push(st!("inplace"));
        }
        "countmatch" => {
            let tt = t3(q);
            let p = match q.below(8) {
                0 => {
                    let v = q.c(&["contain", "mention", "have", "include"]);
                    ph!["how many lines in", &fa, v, &tt]
                }
                1 => ph!["how many lines of", &fa, "mention", &tt],
                2 => ph!["count the lines with", &tt, "in", &fa],
                3 => {
                    let v = q.c(&["appear in", "show up in", "occur in"]);
                    ph!["how many times does", &tt, v, &fa]
                }
                4 => ph!["number of lines containing", &tt, "in", &fa],
                5 => ph!["count occurrences of", &tt, "in", &fa],
                6 => ph!["how often does", &tt, "appear in", &fa],
                _ => ph!["how many", &tt, "lines are in", &fa],
            };
            q.emit(&p);
            pipe.push(st!("match", "text" => k(tt.1)));
            pipe.push(st!("lcount"));
        }
        "ldel" => {
            let a = q.randint(1, 300).to_string();
            let b = q.randint(1, 40).to_string();
            let opts = [a, b, "first".into(), "last".into(), "1st".into(), "2nd".into(), "3rd".into(), "second".into(), "third".into()];
            let n = (q.pick(&opts), q.key());
            if q.f() < 0.75 {
                let p = match q.below(8) {
                    0 => ph!["delete line", &n, "from", &fa],
                    1 => ph!["remove line", &n, "of", &fa],
                    2 => ph!["delete the", &n, "line of", &fa],
                    3 => ph!["drop line", &n, "from", &fa],
                    4 => ph!["get rid of line", &n, "in", &fa],
                    5 => ph!["remove the", &n, "line from", &fa],
                    6 => ph!["cut line", &n, "out of", &fa],
                    _ => ph!["delete line number", &n, "in", &fa],
                };
                q.emit(&p);
                pipe.push(st!("ldel", "n" => k(n.1)));
                pipe.push(st!("inplace"));
            } else {
                let p = match q.below(5) {
                    0 => ph!["show", &fa, "without line", &n],
                    1 => ph!["print", &fa, "skipping line", &n],
                    2 => ph![&fa, "minus line", &n],
                    3 => ph!["show", &fa, "without the", &n, "line"],
                    _ => ph!["print", &fa, "except line", &n],
                };
                q.emit(&p);
                pipe.push(st!("ldel", "n" => k(n.1)));
                pipe.push(st!("lshow"));
            }
        }
        "append_out" => {
            let tt = t3(q);
            let out = q.filename(Some(&["txt", "log", "out", "md"]));
            let ok = q.key();
            let oo = (out, ok);
            let p = match q.below(5) {
                0 => ph!["append lines containing", &tt, "from", &fa, "to", &oo],
                1 => ph!["add the lines with", &tt, "in", &fa, "to the end of", &oo],
                2 => ph!["find", &tt, "in", &fa, "and append the matches to", &oo],
                3 => ph!["grep", &tt, &fa, ">>", &oo],
                _ => ph!["take lines mentioning", &tt, "from", &fa, "and add them to", &oo],
            };
            q.emit(&p);
            pipe.push(st!("match", "text" => k(tt.1)));
            pipe.push(st!("lappend", "to" => k(ok)));
        }
        _ => {
            let _n = (q.count(), q.key());
            let p = match q.below(4) {
                0 => ph!["a random line from", &fa],
                1 => ph!["pick a random line of", &fa],
                2 => ph!["one random line from", &fa],
                _ => ph!["random line of", &fa],
            };
            q.emit(&p);
            pipe.push(st!("lshuf"));
            pipe.push(st!("ltake", "n" => None));
            pipe.push(st!("lshow"));
        }
    }
    pipe
}

// --------------------------------------------------------------------------- more process requests
fn procs2_request(q: &mut Req) -> Pipe {
    if q.f() < 0.22 {
        let p = q.c(POLITE);
        q.lit(p);
    }
    if q.f() < 0.55 {
        let n = (val(q, "port").0, q.key());
        let kind = q.c(&["show", "show", "show", "kill", "kill"]);
        if kind == "show" {
            let p = match q.below(11) {
                0 => ph!["what's using port", &n],
                1 => ph!["which process is listening on port", &n],
                2 => ph!["who is on port", &n],
                3 => ph!["what is running on port", &n],
                4 => ph!["what's listening on", &n],
                5 => ph!["show the process on port", &n],
                6 => ph!["which program has port", &n],
                7 => ph!["find what's using port", &n],
                8 => ph!["what is bound to port", &n],
                9 => ph!["who has port", &n, "open"],
                _ => ph!["lsof port", &n],
            };
            q.emit(&p);
            return vec![st!("port", "n" => k(n.1)), st!("pshow")];
        }
        let force = q.f() < 0.25;
        let p = if force {
            match q.below(3) {
                0 => ph!["force kill whatever is on port", &n],
                1 => ph!["kill -9 the process on port", &n],
                _ => ph!["forcibly kill the thing using port", &n],
            }
        } else {
            match q.below(8) {
                0 => ph!["kill whatever is on port", &n],
                1 => ph!["free up port", &n],
                2 => ph!["kill the process using port", &n],
                3 => ph!["stop whatever is listening on port", &n],
                4 => ph!["kill port", &n],
                5 => ph!["terminate the thing on port", &n],
                6 => ph!["close whatever has port", &n],
                _ => ph!["kill the server on port", &n],
            }
        };
        q.emit(&p);
        return vec![st!("port", "n" => k(n.1)), st!("kill", "force" => e(if force { "hard" } else { "soft" }))];
    }
    let me = q.f() < 0.35;
    let kind = q.c(&["show", "show", "kill", "count", "top"]);
    let who = if me { q.c(&["my", "i"]) } else { q.cat(&[&OWNERS[..10], &[""]]) };
    let who = if !me && who.is_empty() { q.word() } else { who };
    let uk = q.key();
    let mut u = (who.to_string(), uk);
    let mut pipe: Pipe = vec![st!("procs"), st!("puser", "user" => k(uk))];
    match kind {
        "show" => {
            if me {
                u = (if q.f() < 0.6 { "my" } else { "i" }.to_string(), uk);
                let p = if u.0 == "my" {
                    match q.below(3) {
                        0 => ph!["show", &u, "processes"],
                        1 => ph!["list", &u, "processes"],
                        _ => ph!["what processes are", &u, "running"],
                    }
                } else {
                    match q.below(3) {
                        0 => ph!["what am", &u, "running"],
                        1 => ph!["which processes do", &u, "own"],
                        _ => ph!["what processes am", &u, "running"],
                    }
                };
                q.emit(&p);
            } else {
                let p = match q.below(7) {
                    0 => ph!["show processes owned by", &u],
                    1 => ph!["what is", &u, "running"],
                    2 => ph!["processes of user", &u],
                    3 => ph!["processes belonging to", &u],
                    4 => ph!["ps for user", &u],
                    5 => ph!["list everything", &u, "is running"],
                    _ => ph!["which processes does", &u, "own"],
                };
                q.emit(&p);
            }
            pipe.push(st!("pshow"));
        }
        "kill" => {
            let force = q.f() < 0.25;
            let v = if force { q.c(&["force kill", "kill -9"]) } else { q.c(&["kill", "stop", "terminate", "end"]) };
            let p = if me {
                u = ("my".to_string(), uk);
                if q.below(2) == 0 { ph![v, "all", &u, "processes"] } else { ph![v, "all of", &u, "processes"] }
            } else {
                match q.below(4) {
                    0 => ph![v, "all processes of user", &u],
                    1 => ph![v, "everything", &u, "is running"],
                    2 => ph![v, "all processes owned by", &u],
                    _ => ph![v, "every process belonging to", &u],
                }
            };
            q.emit(&p);
            pipe.push(st!("kill", "force" => e(if force { "hard" } else { "soft" })));
        }
        "count" => {
            let p = if me {
                u = ("i".to_string(), uk);
                if q.below(2) == 0 { ph!["how many processes am", &u, "running"] } else { ph!["how many processes do", &u, "have"] }
            } else {
                match q.below(3) {
                    0 => ph!["how many processes does", &u, "have"],
                    1 => ph!["count processes owned by", &u],
                    _ => ph!["number of processes run by", &u],
                }
            };
            q.emit(&p);
            pipe.push(st!("pcount"));
        }
        _ => {
            let key = q.c(&["mem", "cpu"]);
            let res = if key == "mem" { q.c(&["memory", "ram"]) } else { q.c(&["cpu", "processor"]) };
            let n = (q.count(), q.key());
            let p = if me {
                u = ("my".to_string(), uk);
                if q.below(2) == 0 { ph!["top", &n, "of", &u, "processes by", res] } else { ph![&u, &n, "biggest", res, "users"] }
            } else if q.below(2) == 0 {
                ph!["top", &n, "processes of user", &u, "by", res]
            } else {
                ph!["the", &n, "processes of", &u, "using the most", res]
            };
            q.emit(&p);
            pipe.push(("psort", vec![("key", e(key)), ("order", e("desc"))]));
            pipe.push(st!("ptake", "n" => k(n.1)));
            pipe.push(st!("pshow"));
        }
    }
    pipe
}

// --------------------------------------------------------------------------- session preferences
static PREFS: &[((&str, &str), &[&str])] = &[
    (("trash", "on"), &["use the trash instead of deleting", "always move deleted files to the trash", "send deleted files to the trash from now on",
        "don't delete permanently, use the trash", "prefer the trash over rm", "from now on trash things instead of deleting them",
        "use trash", "turn on trash mode", "deletes should go to the trash", "never rm, always trash"]),
    (("trash", "off"), &["stop using the trash", "delete for real from now on", "delete permanently again", "don't use the trash anymore",
        "turn off trash mode", "go back to real deletes", "no more trash, just delete"]),
    (("confirm", "on"), &["ask me before deleting", "always confirm before overwriting", "ask before you delete or overwrite anything",
        "use interactive mode", "always ask first", "confirm every deletion", "add -i to rm and mv", "make deletes interactive",
        "prompt me before removing files"]),
    (("confirm", "off"), &["stop asking me to confirm", "don't ask before deleting", "no more confirmations", "turn off interactive mode",
        "stop confirming", "don't prompt me anymore"]),
    (("verbose", "on"), &["be verbose", "show what gets moved or deleted", "always use verbose mode", "print every file you touch",
        "add -v to commands", "verbose mode on", "tell me each file as you go", "make commands verbose"]),
    (("verbose", "off"), &["stop being verbose", "quiet mode", "turn off verbose mode", "don't print every file", "be quiet", "verbose mode off"]),
];

fn pref_request(q: &mut Req) -> Pipe {
    let ((key, state), phrases) = PREFS[q.below(PREFS.len())];
    if q.f() < 0.2 {
        let p = q.c(POLITE);
        q.lit(p);
    }
    let w = q.c(phrases);
    q.lit(w);
    vec![("pref", vec![("key", e(key)), ("state", e(state))])]
}

// --------------------------------------------------------------------------- requests kural cannot do
static OUTSIDE: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    OUTSIDE_TXT.lines().flat_map(|l| l.split('|')).collect()
});
const OUTSIDE_TXT: &str = "convert {f} to {e}|convert {f} to a gif|resize {f} to {n}x{n}|trim the first {n} seconds of {f}|merge all pdfs into one
turn {f} into a pdf|rotate {f} by {n} degrees|extract the audio from {f}|make a thumbnail of {f}|compress the video {f}|crop {f}
join {f} and {f2} into one video|speed up {f} by {n}x|make {f} black and white|ocr {f}|print {f} on the printer|scan a document
set the wallpaper to {f}|change the wallpaper|take a screenshot|record the screen|record my voice|play some music|play {f}|pause the music
skip this song|what song is this|open port {n} in the firewall|block port {n}|block {h}|allow ssh through the firewall|set up a vpn
add a user called {w}|delete the user {w}|change my password|change the password for {w}|add {w} to the sudo group|give {w} sudo access
create a group called {w}|lock the account {w}|format the drive|format {d} as ext4|partition the disk|mount the usb drive|unmount {d}
eject the disk|check the disk for errors|repair the filesystem|encrypt {f}|decrypt {f}|sign {f} with gpg|generate an ssh key
generate a password|make a random password|create a qr code for {u}|hello|hi there|who are you|tell me a joke|what's the weather
what's the weather in {w}|thanks|thank you|good morning|how are you|what can you do|help|help me|are you there|you are great|nice
what is the meaning of life|sing a song|write a poem about {w}|make me a sandwich|make it rain|make coffee|make {w} great again
find love|find me a girlfriend|find the meaning of {w}|kill time|kill the vibe|open sesame|show me the money|move on|copy that
delete my ex from my memory|go to hell|run away|stop it|stop|go|run|do it|yes|no|ok|maybe|why|what|test|asdf|hmm|lol|never mind
install confidence|remove all doubt|search the web for {w}|google {w}|look up {w} on wikipedia|translate {w} to french|what does {w} mean
define {w}|how do you spell {w}|explain {f}|summarize {f}|summarise this file|what does this script do|what does this error mean
why did that fail|fix it|fix the bug|fix the error in {f}|debug {f}|refactor {f}|optimize {f}|write a script that {w}s every {n} minutes
write a python script to {w}|write a function that sorts a list|write a bash loop over {e} files|generate a makefile|write a dockerfile
write unit tests for {f}|add a comment to {f}|what is a symlink|what is the difference between {w} and {w2}|how does dns work
what is {n} feet in meters|convert {n} miles to km|how many days until christmas|what time is it in tokyo|what day was {n} days ago
how old is linux|who wrote git|when was {w} released|schedule a job at {n}|run {f} every day at {n}|add a cron job that runs {f} hourly
run this every night at midnight|remind me to {w} tomorrow|set an alarm for {n} am|send an email to {w}|email {f} to {w}|text {w}
call {w}|tweet {w}|post {w} to slack|open a ticket|book a meeting|create a database called {w}|run this sql query|select all users from {w}
drop the table {w}|back up the database|restore the database from {f}|kubectl get pods|deploy to production|deploy the app
scale the deployment to {n}|terraform apply|create an s3 bucket|list my ec2 instances|upload {f} to s3|launch a vm|start the virtual machine
compile {f}|compile {f} with optimizations|link {f} against {w}|disassemble {f}|benchmark {f}|profile {f}|strace {p}|attach gdb to {p}
change the hostname to {w}|set the timezone to {w}|set the keyboard layout to {w}|change the system language|set the date to {n}
enable dark mode|change the theme|make the font bigger|rotate the screen|connect my bluetooth headphones|pair with {w}
switch to the external monitor|turn off the screen|overclock the cpu|update the bios|flash the firmware|install windows|install the nvidia driver
rename it|delete it|move it|do that for all of them|the other one|not that one|undo|undo that|undo the last command|revert that
go back in time|restore what i deleted|recover {f}|bring back the file i deleted|is this safe|are you sure|what would that do
split {f} into {n} parts|merge {f} and {f2}|join the lines of {f}|wrap {f} at {n} columns|indent {f}|format {f}|lint {f}|prettify {f}
convert {f} from csv to json|parse {f} as json|validate {f}|pretty print {f}|base64 encode {f}|base64 decode {w}|url encode {w}
hex dump of {f}|count the vowels in {f}|spell check {f}|find typos in {f}|translate {f} to spanish|read {f} out loud
how's the weather looking|is it going to rain|will it snow tomorrow|what's the forecast|launch the browser|open the browser
open a new terminal|open the file manager|open the settings|who won the {w} cup|who won the game|what's the capital of {w}
who is the president of {w}|why is my computer so slow|why is {p} crashing|why does {p} keep freezing|swap tabs for spaces in {f}
convert tabs to spaces in {f}|indent {f} with tabs|plug in the external drive and mount it|mount {d}|safely remove the usb stick
what is {w}|tell me about {w}|how tall is {w}|shrink {f} to half size|make {f} smaller|make {f} louder|reverse the video {f}
set a reminder|what is my name|where do i live|how much money do i have|order a pizza|buy {w}|what should i eat|i'm bored|i am tired";

fn outside_request(q: &mut Req) -> Pipe {
    let t = q.pick(&OUTSIDE);
    let f = q.filename(None);
    let f2 = q.filename(None);
    let ex = q.c(&EXTS);
    let w = q.word();
    let w2 = q.word();
    let n = q.pick(&[1, 2, 5, 10, 30, 50, 80, 100, 644, 755, 8080]).to_string();
    let h = q.c(HOSTS);
    let u = q.c(URLS);
    let p = q.cat(&[&PROCS, &PKGS]);
    let d = q.c(REAL_DIRS);
    let text = t
        .replace("{f2}", &f2)
        .replace("{f}", &f)
        .replace("{e}", ex)
        .replace("{w2}", w2)
        .replace("{w}", w)
        .replace("{n}", &n)
        .replace("{h}", h)
        .replace("{u}", u)
        .replace("{p}", p)
        .replace("{d}", d);
    if q.f() < 0.2 {
        let pp = q.c(POLITE);
        q.lit(pp);
    }
    q.lit(&text);
    if q.f() < 0.06 {
        let tt = q.c(TAILS);
        q.lit(tt);
    }
    vec![st!("unknown")]
}

// --------------------------------------------------------------------------- one request
const SUDO_PRE: &[&str] = &["sudo", "as root", "with sudo", "as superuser", "with root privileges", "as admin", "using sudo"];
const KINDS: &[&str] = &["files", "lines", "lines2", "procs", "procs2", "single", "fileq", "paths", "pref", "outside"];
const KIND_WT: &[f64] = &[32.0, 6.0, 9.0, 5.0, 2.0, 33.0, 3.0, 5.0, 0.6, 5.0];

fn generate(q: &mut Req, kind: &str) -> Result<Pipe, String> {
    Ok(match kind {
        "files" => gen::files_request(q),
        "lines" => gen::lines_request(q),
        "lines2" => lines2_request(q),
        "procs" => gen::procs_request(q),
        "procs2" => procs2_request(q),
        "single" => command_request(q, None)?,
        "fileq" => gen::file_question(q),
        "paths" => paths_request(q),
        "pref" => pref_request(q),
        _ => outside_request(q),
    })
}

pub fn one_pipeline(q: &mut Req, kinds: &[&str], weights: &[f64]) -> Result<Pipe, String> {
    let kind = kinds[q.weighted(weights)];
    let as_root = ["files", "lines", "lines2", "procs", "paths"].contains(&kind) && q.f() < 0.035;
    let front = as_root && q.f() < 0.5;
    if front {
        let w = q.c(&SUDO_PRE[..3]);
        q.lit(w);
    }
    let pipe = generate(q, kind)?;
    if as_root && !front {
        let w = q.c(SUDO_PRE);
        q.lit(w);
    }
    Ok(if as_root {
        let mut p = vec![st!("sudo")];
        p.extend(pipe);
        p
    } else {
        pipe
    })
}

// --------------------------------------------------------------------------- follow-ups
type Where = HashMap<(usize, usize, &'static str), usize>;

/// The previous pipeline again, with every argument pointing at its word in the context.
fn keep(pipe: &[DStage], place: &Where) -> Pipe {
    let lg = lang();
    let mut out = Vec::new();
    for (j, (name, args)) in pipe.iter().enumerate() {
        let sd = &lg.l.stages[lg.stage_ix[*name]];
        let mut a: Args = Vec::new();
        for slot in &sd.slots {
            let sn = slot.name.as_str();
            match slot.kind {
                SlotKind::Span(_) => a.push((sn, place.get(&(0, j, sn)).map(|&p| V::Ctx(p)))),
                SlotKind::Enum(_) => {
                    let v = args.iter().find(|(n, _)| *n == sn).and_then(|(_, v)| if let DV::Enum(x) = v { Some(V::E(x)) } else { None });
                    a.push((sn, v));
                }
            }
        }
        out.push((*name, a));
    }
    out
}

fn val_kind(v: &str) -> Option<&'static str> {
    Some(match v {
        "size" => "size",
        "dur" => "dur",
        "count" => "count",
        "text" => "text",
        "glob" => "word",
        "ext" => "ext",
        "date" => "date",
        "path" => "path",
        "owner" => "owner",
        "mode" => "mode",
        "file" => "textfile",
        "remote" => "remote",
        "word" => "word",
        "port" => "port",
        "unit" => "unit",
        _ => return None,
    })
}

static FOLLOW_SINK: &[(&str, &[&str])] = &[
    ("delete", &["delete them", "now delete them", "remove those", "delete those files", "get rid of them", "ok delete them all", "wipe them",
        "rm them", "delete it", "nuke them", "delete all of those", "and delete them", "remove them", "erase them", "delete these",
        "go ahead and delete them", "delete them all", "remove all of them", "now remove those", "delete that"]),
    ("trash", &["trash them", "move them to the trash", "put those in the bin", "send them to the trash", "trash it", "bin them"]),
    ("fcount", &["how many are there", "count them", "how many is that", "how many", "count those", "how many of them", "what's the count",
        "give me a count", "how many files is that"]),
    ("total", &["how big are they in total", "how much space do they take", "total size", "what's their combined size", "how much do they add up to",
        "how much space is that", "how much space would that free"]),
    ("sizes", &["how big is each", "show their sizes", "with sizes", "how big are they", "list them with sizes", "and their sizes"]),
    ("details", &["show details", "with permissions", "ls -l them", "show me the details of those", "who owns them", "list them with details"]),
    ("fopen", &["open them", "open it", "open those", "open that one", "open them all"]),
    ("fedit", &["edit it", "edit them", "open them in my editor", "let me edit those"]),
    ("fhash", &["checksum them", "hash those", "sha256 them"]),
    ("rn_lower", &["rename them to lowercase", "lowercase their names", "make their names lowercase"]),
    ("rn_spaces", &["replace the spaces in their names with underscores", "remove spaces from their names"]),
    ("by_ext", &["break them down by type", "count them by extension", "how many of each type"]),
    ("dupes", &["are any of them duplicates", "find duplicates among them", "which of those are identical"]),
    ("fshow", &["show them", "list them", "which ones", "show me them", "list those", "let me see them", "show them again", "which files"]),
    ("cat_count", &["how many lines do they have", "count their lines", "how many lines in total", "count the lines in those"]),
];
static FOLLOW_SINK_ARG: &[(&str, &[&str], &str, &str)] = &[
    ("move", &["move them to {}", "put those in {}", "now move them into {}", "move it to {}", "send them to {}", "move those to {}",
        "shift them over to {}", "stick them in {}", "move them all to {}"], "to", "path"),
    ("copy", &["copy them to {}", "back them up to {}", "copy those into {}", "copy it to {}", "make copies in {}", "copy them all to {}"], "to", "path"),
    ("chmod", &["make them {}", "chmod {} them", "set them to {}", "make those {}"], "mode", "mode"),
    ("chown", &["give them to {}", "change their owner to {}", "chown them to {}"], "owner", "owner"),
    ("upload", &["upload them to {}", "send those to {}", "rsync them to {}", "scp them to {}", "copy them to {}"], "to", "remote"),
    ("fgrep", &["search them for {}", "grep them for {}", "look for {} in those", "which lines mention {}", "find {} in them", "search those for {}"], "text", "text"),
    ("rn_prefix", &["prefix them with {}", "add {} to the start of their names"], "text", "affix"),
    ("rn_ext", &["change their extension to {}", "rename them to {}"], "ext", "ext"),
    ("names_save", &["save that list to {}", "write those names to {}", "save the list to {}", "put that list in {}"], "to", "textfile"),
];
static FOLLOW_ZIP: &[(&str, &str, &str)] = &[("zip them up", "zip", ""), ("compress them", "tgz", ""), ("tar them up", "tgz", ""), ("zip those", "zip", ""),
    ("make a tarball of them", "tgz", ""), ("put them in a zip called {}", "zip", ".zip"), ("archive those as {}", "tgz", ".tar.gz"),
    ("zip them into {}", "zip", ".zip"), ("tar them into {}", "tgz", ".tar.gz")];
static FOLLOW_SORT: &[(&str, &str, &str, u8)] = &[("sort them by size", "size", "desc", 0), ("biggest first", "size", "desc", 0), ("largest first", "size", "desc", 0),
    ("newest first", "time", "desc", 0), ("oldest first", "time", "asc", 0), ("sort by date", "time", "desc", 0),
    ("sorted by name", "name", "asc", 0), ("order them by size", "size", "desc", 0), ("smallest first", "size", "asc", 0),
    ("just the top {}", "", "", 1), ("only the {} largest", "size", "desc", 1), ("show the {} newest of those", "time", "desc", 1),
    ("the biggest one", "size", "desc", 2), ("top {}", "", "", 1), ("first {} only", "", "", 1),
    ("just the {} biggest", "size", "desc", 1), ("only the {} oldest", "time", "asc", 1), ("the newest one", "time", "desc", 2),
    ("only the {} smallest", "size", "asc", 1), ("limit it to {}", "", "", 1), ("the oldest one", "time", "asc", 2)];
static FOLLOW_LOC: &[&str] = &["same but in {}", "now in {}", "do the same for {}", "what about {}", "and in {}", "same thing for {}", "now do {}",
    "repeat for {}", "same for {}", "try {} instead", "how about in {}", "now check {}", "what about in {}", "same in {}"];
static FOLLOW_VAL: &[&str] = &["no, {}", "i meant {}", "actually {}", "make it {}", "make that {}", "sorry, {}", "oops {}", "change it to {}", "{} instead",
    "use {} instead", "no i meant {}", "i said {}", "should be {}", "wait, {}", "not that, {}", "rather {}", "change that to {}"];
static FOLLOW_EXT: &[&str] = &["same for {}", "what about {}", "now {}", "and for {} files", "do the same with {}", "same but {}", "how about {}"];
static FOLLOW_SUDO: &[&str] = &["with sudo", "as root", "try again as root", "do it with sudo", "same but with sudo", "sudo", "use sudo",
    "permission denied, use sudo", "run that as root", "again as root", "needs sudo", "that needs root"];
static FOLLOW_PICK: &[(&str, &str)] = &[("open the {} one", "fopen"), ("delete the {} one", "delete"), ("show me the {}", "fshow"), ("just the {} one", "fshow"),
    ("edit the {} one", "fedit"), ("trash the {} one", "trash"), ("how big is the {} one", "sizes"), ("open the {}", "fopen"),
    ("the {} one", "fshow"), ("delete the {}", "delete")];

fn is_source_or_filter(n: &str) -> bool {
    lang().filter_order.contains(&n) || ["files", "paths", "not", "sudo"].contains(&n)
}
fn is_order(n: &str) -> bool {
    ["fsort", "ftake", "fpick"].contains(&n)
}

fn zip_followup(q: &mut Req) -> Stage {
    let (phrase, fmt, ext) = FOLLOW_ZIP[q.below(FOLLOW_ZIP.len())];
    let mut kk = None;
    if !ext.is_empty() {
        let (a, b) = phrase.split_once("{}").unwrap();
        q.lit(a);
        let key = q.key();
        let w = q.word();
        let x = q.c(&[ext, ext, ""]);
        q.chunks.push((format!("{w}{x}"), Some(key)));
        q.lit(b);
        kk = Some(V::K(key));
    } else {
        q.lit(phrase);
    }
    ("compress", vec![("fmt", e(fmt)), ("to", kk)])
}

fn files_followup(q: &mut Req, prev: &[DStage], place: &Where) -> Option<Vec<Pipe>> {
    let kp = keep(prev, place);
    let body: Pipe = kp.iter().filter(|s| is_source_or_filter(s.0)).cloned().collect();
    let order: Pipe = kp.iter().filter(|s| is_order(s.0)).cloned().collect();
    let sink: Pipe = kp.iter().filter(|s| !is_source_or_filter(s.0) && !is_order(s.0)).cloned().collect();
    let dirs = kp.iter().any(|(n, a)| *n == "files" && arg(a, "what") == Some(V::E("dirs")));
    let edit = ["sink", "filter", "sort", "loc", "value", "pick", "sudo"][q.weighted(&[42.0, 20.0, 10.0, 9.0, 11.0, 5.0, 3.0])];
    let cat = |a: &Pipe, b: &Pipe, c: Pipe| -> Pipe { a.iter().chain(b.iter()).cloned().chain(c).collect() };
    match edit {
        "sink" => {
            let roll = q.f();
            if roll < 0.62 {
                let (name, phrases) = FOLLOW_SINK[q.below(FOLLOW_SINK.len())];
                let w = q.c(phrases);
                q.lit(w);
                return Some(vec![cat(&body, &order, frame_stages(sname_or(name), &[]))]);
            }
            if roll < 0.9 {
                let (name, phrases, slot, kind) = FOLLOW_SINK_ARG[q.below(FOLLOW_SINK_ARG.len())];
                let p = q.c(phrases);
                let kk = fill(q, p, kind);
                return Some(vec![cat(&body, &order, frame_stages(sname_or(name), &[(slot_static(slot), kk)]))]);
            }
            let z = zip_followup(q);
            Some(vec![cat(&body, &order, vec![z])])
        }
        "filter" => {
            let have: Vec<&str> = body.iter().map(|s| s.0).collect();
            let mut found = None;
            for _ in 0..12 {
                let fl = file_filters(q, dirs);
                let names: Vec<&str> = fl.stages.iter().map(|s| s.0).collect();
                if !names.is_empty() && names.len() <= 2 && !names.iter().any(|n| have.contains(n)) {
                    found = Some(fl);
                    break;
                }
            }
            let fl = found?;
            let bare = !fl.pre.is_empty() || fl.noun.is_some();
            let lead = if bare {
                q.c(&["only", "just", "but only", "now only", "filter to", "limit it to", "only show", "narrow it to"])
            } else {
                q.c(&["only the ones", "just the ones", "only those", "keep only the ones", "narrow it to the ones", "of those only the ones",
                    "which of them are", "which of those are", "the ones", "only show the ones", "but only the ones", "just those"])
            };
            q.lit(lead);
            for p in &fl.pre {
                q.emit(p);
            }
            if let Some(n) = &fl.noun {
                q.emit(&[P::from(n)]);
            } else if bare {
                let w = q.c(&["ones", "files", "ones"]);
                q.lit(w);
            }
            q.tail(&fl.post);
            let mut merged: Vec<(&'static str, Args)> = body.iter().filter(|s| lang().filter_order.contains(&s.0)).cloned().collect();
            let mut negs: Vec<&'static str> = Vec::new();
            for (i, (n, _)) in body.iter().enumerate() {
                if *n == "not" && i + 1 < body.len() {
                    negs.push(body[i + 1].0);
                }
            }
            for s in &fl.stages {
                merged.retain(|m| m.0 != s.0);
                merged.push(s.clone());
            }
            negs.extend(fl.negated.iter().copied());
            let head: Pipe = body.iter().filter(|s| ["sudo", "files", "paths"].contains(&s.0)).cloned().collect();
            let mut out = head;
            out.extend(crate::req::filter_stages(&merged, &negs));
            out.extend(order);
            out.extend(sink);
            Some(vec![out])
        }
        "sort" => {
            let (phrase, key, direction, mode) = FOLLOW_SORT[q.below(FOLLOW_SORT.len())];
            let kk = fill(q, phrase, "count");
            let mut new: Pipe = if key.is_empty() {
                order.iter().filter(|s| s.0 == "fsort").cloned().collect()
            } else {
                vec![("fsort", vec![("key", e(key)), ("order", e(direction))])]
            };
            match mode {
                1 => new.push(("ftake", vec![("n", kk)])),
                2 => new.push(("ftake", vec![("n", None)])),
                _ => new.extend(order.iter().filter(|s| s.0 == "ftake").cloned()),
            }
            let mut out = body;
            out.extend(new);
            out.extend(sink);
            Some(vec![out])
        }
        "loc" => {
            if !body.iter().any(|s| s.0 == "files") {
                return None;
            }
            let p = q.c(FOLLOW_LOC);
            let kk = fill(q, p, "path");
            let mut out: Pipe = body.iter().map(|(n, a)| if *n == "files" { (*n, set_arg(a, "in", kk)) } else { (*n, a.clone()) }).collect();
            out.extend(order);
            out.extend(sink);
            Some(vec![out])
        }
        "value" => {
            let lg = lang();
            let mut spots: Vec<(usize, &'static str, &'static str)> = Vec::new();
            for (j, (n, a)) in kp.iter().enumerate() {
                if *n == "files" || *n == "paths" {
                    continue;
                }
                for slot in &lg.l.stages[lg.stage_ix[*n]].slots {
                    if let SlotKind::Span(t) = &slot.kind {
                        if arg(a, &slot.name).is_some() && val_kind(t).is_some() {
                            spots.push((j, slot.name.as_str(), t.as_str()));
                        }
                    }
                }
            }
            if spots.is_empty() {
                return None;
            }
            let (j, slot, vk) = spots[q.below(spots.len())];
            let list = if vk == "ext" && q.f() < 0.6 { FOLLOW_EXT } else { FOLLOW_VAL };
            let p = q.c(list);
            let kk = fill(q, p, val_kind(vk).unwrap());
            Some(vec![kp.iter().enumerate().map(|(i, (n, a))| if i == j { (*n, set_arg(a, slot, kk)) } else { (*n, a.clone()) }).collect()])
        }
        "pick" => {
            let (phrase, name) = FOLLOW_PICK[q.below(FOLLOW_PICK.len())];
            let (a, b) = phrase.split_once("{}").unwrap();
            q.lit(a);
            let key = q.key();
            let o = q.c(ORDINALS);
            q.chunks.push((o.to_string(), Some(key)));
            q.lit(b);
            let mut out = body;
            out.extend(order.into_iter().filter(|s| s.0 != "fpick"));
            out.push(st!("fpick", "n" => k(key)));
            out.push((sname(name), Vec::new()));
            Some(vec![out])
        }
        _ => {
            if kp[0].0 == "sudo" {
                return None;
            }
            let w = q.c(FOLLOW_SUDO);
            q.lit(w);
            let mut out = vec![st!("sudo")];
            out.extend(kp);
            Some(vec![out])
        }
    }
}

fn sname_or(n: &str) -> &'static str {
    match n {
        "names_save" => "names_save",
        "cat_count" => "cat_count",
        "cat_wcount" => "cat_wcount",
        other => sname(other),
    }
}

fn slot_static(s: &str) -> &'static str {
    match s {
        "to" => "to",
        "mode" => "mode",
        "owner" => "owner",
        "text" => "text",
        "ext" => "ext",
        "n" => "n",
        _ => Box::leak(s.to_string().into_boxed_str()),
    }
}

static FOLLOW_LSINK: &[(&str, &[&str])] = &[
    ("lcount", &["count them", "how many", "how many lines", "count those lines", "how many is that", "how many are there", "count that"]),
    ("clip", &["copy that to the clipboard", "copy them", "put that on my clipboard", "copy that"]),
    ("wcount", &["how many words", "count the words", "how many words is that"]),
    ("lshow", &["show them", "show me", "let me see them"]),
];
static FOLLOW_LSINK_ARG: &[(&str, &[&str])] = &[
    ("save", &["save that to {}", "save them to {}", "write that to {}", "put the result in {}", "save it as {}", "save those to {}"]),
    ("lappend", &["append that to {}", "add those to {}", "append them to {}"]),
];
static FOLLOW_LOP: &[(&[&str], &str, &str, &str)] = &[
    (&["only lines with {}", "just the ones containing {}", "filter for {}", "only those mentioning {}", "grep that for {}", "narrow it to {}",
        "only the ones with {}"], "match", "text", "text"),
    (&["without {}", "drop the ones with {}", "exclude {}", "hide lines with {}", "but not the ones mentioning {}"], "nomatch", "text", "text"),
    (&["remove duplicates", "dedupe them", "unique ones only", "without duplicates"], "uniq", "", ""),
    (&["just the first {}", "only the top {}", "first {} only", "limit that to {}"], "ltake", "n", "count"),
    (&["just the last {}", "only the last {}", "the last {} of those"], "llast", "n", "count"),
    (&["which are most common", "count how often each appears", "most frequent first"], "lfreq", "", ""),
    (&["with line numbers", "number them"], "lnum", "", ""),
];
static FOLLOW_LSORT: &[(&str, &str, &str)] = &[("sort them", "alpha", "asc"), ("sorted", "alpha", "asc"), ("sort that", "alpha", "asc"),
    ("in alphabetical order", "alpha", "asc"), ("sort them in reverse", "alpha", "desc"), ("sort numerically", "num", "asc"), ("biggest number first", "num", "desc")];
static FOLLOW_FILE: &[&str] = &["same for {}", "now do {}", "what about {}", "same thing with {}", "and in {}", "try {} instead", "do that for {} too"];

fn lines_followup(q: &mut Req, prev: &[DStage], place: &Where) -> Option<Vec<Pipe>> {
    let kp = keep(prev, place);
    let (mut ops, mut sink): (Pipe, Pipe) = (kp[..kp.len() - 1].to_vec(), kp[kp.len() - 1..].to_vec());
    if !["lshow", "lcount", "wcount", "ccount", "save", "lappend", "clip", "inplace"].contains(&sink[0].0) {
        ops = kp.clone();
        sink = vec![st!("lshow")];
    }
    let edit = ["sink", "op", "sort", "file", "value", "pron", "sudo"][q.weighted(&[30.0, 32.0, 8.0, 9.0, 8.0, 10.0, 3.0])];
    let src = kp.iter().position(|s| s.0 == "lines");
    match edit {
        "sink" => {
            if q.f() < 0.6 {
                let (name, phrases) = FOLLOW_LSINK[q.below(FOLLOW_LSINK.len())];
                let w = q.c(phrases);
                q.lit(w);
                ops.push((sname(name), Vec::new()));
                return Some(vec![ops]);
            }
            let (name, phrases) = FOLLOW_LSINK_ARG[q.below(FOLLOW_LSINK_ARG.len())];
            let p = q.c(phrases);
            let kk = fill(q, p, "textfile");
            ops.push((sname(name), vec![("to", kk)]));
            Some(vec![ops])
        }
        "op" => {
            let (phrases, name, slot, kind) = FOLLOW_LOP[q.below(FOLLOW_LOP.len())];
            let p = q.c(phrases);
            let kk = fill(q, p, if kind.is_empty() { "text" } else { kind });
            ops.push((sname(name), if slot.is_empty() { Vec::new() } else { vec![(slot_static(slot), kk)] }));
            ops.extend(sink);
            Some(vec![ops])
        }
        "sort" => {
            let (phrase, key, dir) = FOLLOW_LSORT[q.below(FOLLOW_LSORT.len())];
            q.lit(phrase);
            ops.push(("lsort", vec![("key", e(key)), ("order", e(dir))]));
            ops.extend(sink);
            Some(vec![ops])
        }
        "file" => {
            let src = src?;
            let p = q.c(FOLLOW_FILE);
            let kk = fill(q, p, "textfile");
            Some(vec![kp.iter().enumerate().map(|(j, (n, a))| if j == src { (*n, vec![("of", kk)]) } else { (*n, a.clone()) }).collect()])
        }
        "value" => {
            let lg = lang();
            let mut spots: Vec<(usize, &'static str, &'static str)> = Vec::new();
            for (j, (n, a)) in kp.iter().enumerate() {
                if *n == "lines" {
                    continue;
                }
                for slot in &lg.l.stages[lg.stage_ix[*n]].slots {
                    if let SlotKind::Span(t) = &slot.kind {
                        if arg(a, &slot.name).is_some() && val_kind(t).is_some() {
                            spots.push((j, slot.name.as_str(), t.as_str()));
                        }
                    }
                }
            }
            if spots.is_empty() {
                return None;
            }
            let (j, slot, vk) = spots[q.below(spots.len())];
            let p = q.c(FOLLOW_VAL);
            let kk = fill(q, p, val_kind(vk).unwrap());
            Some(vec![kp.iter().enumerate().map(|(i, (n, a))| if i == j { (*n, set_arg(a, slot, kk)) } else { (*n, a.clone()) }).collect()])
        }
        "pron" => {
            let src = src?;
            let c = arg(&kp[src].1, "of")?;
            Some(pronoun_followup(q, "file", c))
        }
        _ => {
            if kp[0].0 == "sudo" {
                return None;
            }
            let w = q.c(FOLLOW_SUDO);
            q.lit(w);
            let mut out = vec![st!("sudo")];
            out.extend(kp);
            Some(vec![out])
        }
    }
}

fn procs_followup(q: &mut Req, prev: &[DStage], place: &Where) -> Option<Vec<Pipe>> {
    let kp = keep(prev, place);
    let body: Pipe = kp[..kp.len() - 1].to_vec();
    let edit = ["kill", "count", "show", "top", "name"][q.weighted(&[50.0, 12.0, 8.0, 15.0, 15.0])];
    let with = |mut b: Pipe, s: Stage| -> Pipe {
        b.push(s);
        b
    };
    match edit {
        "kill" => {
            let force = q.f() < 0.3;
            let w = if force {
                q.c(&["force kill it", "kill -9 it", "kill it hard", "force quit it", "nuke it", "force kill them", "kill -9 them"])
            } else {
                q.c(&["kill it", "kill them", "kill that", "stop it", "terminate them", "end it", "kill those", "ok kill it", "now kill it",
                    "kill them all", "stop them", "terminate it"])
            };
            q.lit(w);
            Some(vec![with(body, st!("kill", "force" => e(if force { "hard" } else { "soft" })))])
        }
        "count" => {
            let w = q.c(&["how many", "count them", "how many are there", "how many is that"]);
            q.lit(w);
            Some(vec![with(body, st!("pcount"))])
        }
        "show" => {
            let w = q.c(&["show them", "list them", "show me", "which ones"]);
            q.lit(w);
            Some(vec![with(body, st!("pshow"))])
        }
        "top" => {
            if kp[0].0 == "port" {
                return None;
            }
            let key = q.c(&["mem", "cpu"]);
            let res = if key == "mem" { q.c(&["memory", "ram"]) } else { q.c(&["cpu", "processor"]) };
            let base: Pipe = body.iter().filter(|s| s.0 != "psort" && s.0 != "ptake").cloned().collect();
            let mut out = base;
            if q.f() < 0.5 {
                let opts = [format!("top {{}} by {res}"), format!("just the {{}} using the most {res}"), format!("only the top {{}} by {res}")];
                let p = q.pick(&opts);
                let kk = fill(q, &p, "count");
                out.push(("psort", vec![("key", e(key)), ("order", e("desc"))]));
                out.push(("ptake", vec![("n", kk)]));
            } else {
                let opts = [format!("sort by {res}"), format!("sort them by {res}"), format!("order by {res} usage"), format!("by {res}")];
                let p = q.pick(&opts);
                q.lit(&p);
                out.push(("psort", vec![("key", e(key)), ("order", e("desc"))]));
                out.extend(body.iter().filter(|s| s.0 == "ptake").cloned());
            }
            out.push(kp[kp.len() - 1].clone());
            Some(vec![out])
        }
        _ => {
            let (j, slot) = kp.iter().enumerate().find_map(|(j, (n, a))| {
                if *n == "pname" && a.iter().any(|x| x.0 == "name") {
                    Some((j, "name"))
                } else if *n == "port" && a.iter().any(|x| x.0 == "n") {
                    Some((j, "n"))
                } else {
                    None
                }
            })?;
            let is_port = kp[j].0 == "port";
            let p = q.c(&["same for {}", "what about {}", "now {}", "and {}", "do the same for {}", "i meant {}", "no, {}"]);
            let (a, b) = p.split_once("{}").unwrap();
            q.lit(a);
            let key = q.key();
            let v = if is_port { val(q, "port").0 } else { q.c(&PROCS).to_string() };
            q.chunks.push((v, Some(key)));
            q.lit(b);
            Some(vec![kp.iter().enumerate().map(|(i, (n, a2))| if i == j { (*n, set_arg(a2, slot, k(key))) } else { (*n, a2.clone()) }).collect()])
        }
    }
}

// ---- "it": a new command about the thing the previous request named
struct Pron {
    phrases: &'static [&'static str],
    build: fn(V, Option<V>) -> Pipe,
    fill: &'static str,
}

macro_rules! pr {
    ($phrases:expr, $name:literal, $slot:literal) => {
        Pron { phrases: $phrases, build: |c, _| vec![($name, vec![($slot, Some(c))])], fill: "" }
    };
    ($phrases:expr, $f:expr) => {
        Pron { phrases: $phrases, build: $f, fill: "" }
    };
    ($phrases:expr, $f:expr, $kind:literal) => {
        Pron { phrases: $phrases, build: $f, fill: $kind }
    };
}

fn pron(kind: &str) -> Vec<Pron> {
    match kind {
        "path" => vec![
            pr!(&["go there", "cd there", "cd into it", "take me there", "go into it", "enter it", "now go there", "cd to it", "jump there",
                "go to that folder", "cd into that", "switch to it", "and go there", "now cd into it", "go in there"], "cd", "path"),
            pr!(&["what's in there", "look inside", "list it", "show what's in it", "ls there", "what's inside it", "show me what's in there"], "ls", "path"),
            pr!(&["list the files there", "what files are in it", "show the files in there", "find the files in it"],
                |c, _| vec![("files", vec![("what", Some(V::E("anyfile"))), ("in", Some(c))]), ("fshow", vec![])]),
            pr!(&["how big is it", "how much space does it take", "what's its size"], |c, _| vec![("paths", vec![("names", Some(c))]), ("sizes", vec![])]),
            pr!(&["delete it", "remove it", "get rid of it", "delete that folder"], |c, _| vec![("paths", vec![("names", Some(c))]), ("delete", vec![])]),
            pr!(&["show it as a tree", "tree of it", "show its structure"], |c, _| vec![("tree", vec![("path", Some(c)), ("depth", None)])]),
        ],
        "file" => vec![
            pr!(&["show it", "print it", "cat it", "what's in it", "show me that file", "read it", "let me see it", "display it"], "view", "file"),
            pr!(&["edit it", "open it in my editor", "let me edit it", "now edit that", "open it in vim", "edit that file"], "edit", "file"),
            pr!(&["open it", "open that", "xdg-open it"], "open", "target"),
            pr!(&["delete it", "remove it", "remove that file", "get rid of it", "rm it", "delete that"], |c, _| vec![("paths", vec![("names", Some(c))]), ("delete", vec![])]),
            pr!(&["move it to {}", "put it in {}", "move that to {}"], |c, kk| vec![("paths", vec![("names", Some(c))]), ("move", vec![("to", kk)])], "path"),
            pr!(&["copy it to {}", "copy that into {}"], |c, kk| vec![("paths", vec![("names", Some(c))]), ("copy", vec![("to", kk)])], "path"),
            pr!(&["make it executable", "chmod +x it", "let me run it", "mark it executable"], "make_exec", "file"),
            pr!(&["back it up", "make a backup of it", "keep a copy of it"], "backup", "file"),
            pr!(&["rename it to {}", "call it {} instead", "change its name to {}"], |c, kk| vec![("rename", vec![("old", Some(c)), ("new", kk)])], "textfile"),
            pr!(&["run it", "execute it", "now run it", "run that", "launch it"], "run_file", "file"),
            pr!(&["how big is it", "what's its size", "how large is that"], |c, _| vec![("paths", vec![("names", Some(c))]), ("sizes", vec![])]),
            pr!(&["how many lines does it have", "count its lines", "how long is it"], |c, _| vec![("lines", vec![("of", Some(c))]), ("lcount", vec![])]),
            pr!(&["search it for {}", "grep it for {}", "find {} in it", "does it mention {}"],
                |c, kk| vec![("lines", vec![("of", Some(c))]), ("match", vec![("text", kk)]), ("lshow", vec![])], "text"),
            pr!(&["follow it", "tail it", "watch it for changes"], "follow", "file"),
            pr!(&["who owns it", "when was it last modified", "show its permissions", "stat it"], "finfo", "file"),
            pr!(&["stage it", "git add it", "add it to git"], "git_add", "file"),
            pr!(&["checksum it", "hash it", "sha256 of it"], "checksum", "file"),
            pr!(&["what kind of file is it", "what type is it"], "ftype", "file"),
            pr!(&["show the first {} lines of it", "first {} lines of it"], |c, kk| vec![("lines", vec![("of", Some(c))]), ("ltake", vec![("n", kk)]), ("lshow", vec![])], "count"),
            pr!(&["show the last {} lines of it", "last {} lines of it"], |c, kk| vec![("lines", vec![("of", Some(c))]), ("llast", vec![("n", kk)]), ("lshow", vec![])], "count"),
        ],
        "unit" => vec![
            pr!(&["restart it", "now restart it", "bounce it", "restart that", "restart that service", "give it a restart"], "svc_restart", "unit"),
            pr!(&["stop it", "stop that service", "shut it down", "turn it off"], "svc_stop", "unit"),
            pr!(&["start it", "start it up", "bring it up", "start that service", "turn it on"], "svc_start", "unit"),
            pr!(&["enable it", "enable it at boot", "make it start on boot", "autostart it"], "svc_enable", "unit"),
            pr!(&["disable it", "stop it from starting at boot", "disable that service"], "svc_disable", "unit"),
            pr!(&["check its status", "is it running", "how is it doing", "status of it", "is it up", "did it start"], "svc_status", "unit"),
            pr!(&["show its logs", "what's it logging", "show me the logs", "logs for it", "check its logs"], |c, _| vec![("journal", vec![("unit", Some(c))]), ("lshow", vec![])]),
            pr!(&["follow its logs", "tail its logs", "watch its logs"], "svc_follow", "unit"),
        ],
        "pkg" => vec![
            pr!(&["install it", "ok install it", "install that", "get it installed", "go ahead and install it"], "pkg_install", "names"),
            pr!(&["uninstall it", "remove it", "get rid of it"], "pkg_remove", "names"),
            pr!(&["tell me more about it", "info on it", "show details about it", "what version is it"], "pkg_info", "name"),
            pr!(&["what files does it install", "list its files"], "pkg_files", "name"),
        ],
        "container" => vec![
            pr!(&["stop it", "stop that container", "shut it down"], "dk_stop", "name"),
            pr!(&["restart it"], "dk_restart", "name"),
            pr!(&["show its logs", "logs for it", "what's it printing"], "dk_logs", "name"),
            pr!(&["follow its logs", "tail its logs"], "dk_follow", "name"),
            pr!(&["remove it", "delete it", "delete that container"], "dk_rm", "name"),
            pr!(&["get a shell in it", "exec into it", "open a shell inside it", "jump into it"], "dk_shell", "name"),
            pr!(&["start it", "start it again"], "dk_start", "name"),
        ],
        "image" => vec![
            pr!(&["run it", "start a container from it"], "dk_run", "image"),
            pr!(&["remove it", "delete that image"], "dk_rmi", "image"),
            pr!(&["pull it", "pull it again"], "dk_pull", "image"),
        ],
        "branch" => vec![
            pr!(&["switch to it", "check it out", "go to it", "checkout that branch"], "git_checkout", "branch"),
            pr!(&["merge it", "merge it in", "merge that into this branch"], "git_merge", "branch"),
            pr!(&["delete it", "delete that branch", "remove it", "get rid of that branch"], "git_branch_del", "branch"),
            pr!(&["rebase onto it"], "git_rebase", "branch"),
        ],
        "host" => vec![
            pr!(&["ssh into it", "log in to it", "connect to it", "ssh there", "get a shell on it"], "ssh", "host"),
            pr!(&["ping it", "is it up", "is it reachable", "check if it's online"], "ping", "host"),
            pr!(&["what's its ip", "resolve it", "look up its ip"], "dns", "host"),
        ],
        "url" => vec![
            pr!(&["download it", "fetch it", "grab it", "save it"], |c, _| vec![("download", vec![("url", Some(c)), ("to", None)])]),
            pr!(&["show its headers", "what status does it return"], "http_head", "url"),
            pr!(&["curl it", "show the response"], "http_get", "url"),
            pr!(&["clone it", "git clone it"], |c, _| vec![("git_clone", vec![("url", Some(c)), ("into", None)])]),
            pr!(&["open it", "open it in the browser"], "open", "target"),
        ],
        "cmdname" => vec![
            pr!(&["show its man page", "how do i use it", "help for it", "man it"], "man", "cmd"),
            pr!(&["where is it installed", "which one is it", "where does it live"], "which", "cmd"),
            pr!(&["what version is it", "which version do i have"], "ver", "cmd"),
        ],
        "envname" => vec![
            pr!(&["what is it set to", "show its value", "print it", "echo it"], "var_show", "name"),
            pr!(&["unset it", "clear it", "remove it"], "var_unset", "name"),
            pr!(&["set it to {}", "change it to {}", "make it {}"], |c, kk| vec![("var_set", vec![("name", Some(c)), ("value", kk)])], "varvalue"),
        ],
        _ => vec![
            pr!(&["what's using it", "who is on that port", "what is listening on it"], |c, _| vec![("port", vec![("n", Some(c))]), ("pshow", vec![])]),
            pr!(&["kill whatever is using it", "free it up", "kill the process on it"],
                |c, _| vec![("port", vec![("n", Some(c))]), ("kill", vec![("force", Some(V::E("soft")))])]),
        ],
    }
}

const PRON_KINDS: &[&str] = &["path", "file", "unit", "pkg", "container", "image", "branch", "host", "url", "cmdname", "envname", "port"];
const PKG_CMDS: &[&str] = &["pkg_install", "pkg_remove", "pkg_search", "pkg_info", "pkg_has", "pkg_files"];

/// A new command about "it": c is the context position (or key) of the thing "it" refers to.
fn pronoun_followup(q: &mut Req, kind: &str, c: V) -> Vec<Pipe> {
    let entries = pron(kind);
    let en = &entries[q.below(entries.len())];
    let phrase = q.c(en.phrases);
    if !en.fill.is_empty() {
        let kk = fill(q, phrase, en.fill);
        return vec![(en.build)(c, kk)];
    }
    q.lit(phrase);
    vec![(en.build)(c, None)]
}

struct Step {
    phrases: &'static [&'static str],
    prog: Pipe,
}

fn next_steps(name: &str) -> Vec<Step> {
    let s = |phrases: &'static [&'static str], prog: Pipe| Step { phrases, prog };
    let mut out = Vec::new();
    if ["git_commit", "git_commit_all", "git_merge", "git_amend", "git_add_all", "git_tag"].contains(&name) {
        out.push(s(&["push it", "now push", "push that", "and push", "push it up"], vec![st!("git_push")]));
        out.push(s(&["undo that", "undo it", "take that back"], vec![st!("git_undo")]));
    }
    if name == "git_stash" {
        out.push(s(&["pop it", "bring it back", "unstash", "undo that"], vec![st!("git_stash_pop")]));
    }
    if name == "git_pull" || name == "git_fetch" {
        out.push(s(&["show what changed", "what came in", "what's new"], vec![st!("git_log", "n" => None)]));
    }
    if name == "compose_up" {
        out.push(s(&["show the logs", "logs"], vec![st!("compose_logs")]));
        out.push(s(&["bring it down", "stop it", "take it down"], vec![st!("compose_down")]));
    }
    if name == "poweroff" || name == "reboot" {
        out.push(s(&["cancel that", "never mind, cancel it", "abort", "cancel the shutdown", "wait, cancel"], vec![st!("shutdown_cancel")]));
    }
    if ["vol_set", "vol_up", "vol_down", "mute"].contains(&name) {
        out.push(s(&["louder", "a bit more", "more"], vec![st!("vol_up")]));
        out.push(s(&["quieter", "too loud", "less"], vec![st!("vol_down")]));
    }
    if ["bright_set", "bright_up", "bright_down"].contains(&name) {
        out.push(s(&["brighter", "a bit more"], vec![st!("bright_up")]));
        out.push(s(&["dimmer", "too bright", "less"], vec![st!("bright_down")]));
    }
    if name == "venv_make" {
        out.push(s(&["activate it", "now activate it", "turn it on", "enter it"], vec![st!("venv_on", "path" => None)]));
    }
    if name == "git_add_all" || name == "git_add" {
        out.push(s(&["unstage that", "undo that"], vec![st!("git_unstage", "file" => None)]));
    }
    out
}

/// Each NEXT_STEP group that lists the command, as mix.single_followup walks them: (group, options).
fn next_groups(name: &str) -> Vec<Vec<Step>> {
    let all = next_steps(name);
    // group boundaries as in mix.NEXT_STEP: git_add_all appears in two groups
    if name == "git_add_all" {
        let mut it = all.into_iter();
        let a: Vec<Step> = it.by_ref().take(2).collect();
        let b: Vec<Step> = it.collect();
        return vec![a, b];
    }
    if all.is_empty() {
        vec![]
    } else {
        vec![all]
    }
}

fn single_followup(q: &mut Req, prev: &[DStage], place: &Where) -> Option<Vec<Pipe>> {
    let j = if prev[0].0 == "sudo" { 1 } else { 0 };
    let name = prev.get(j)?.0;
    let c = CMDS.by.get(name)?;
    for group in next_groups(name) {
        if q.f() < 0.5 {
            let st = &group[q.below(group.len())];
            let w = q.c(st.phrases);
            q.lit(w);
            let mut prog = st.prog.clone();
            if prog[0].0 == "venv_on" {
                if let Some(&p) = place.get(&(0, j, "path")) {
                    prog = vec![st!("venv_on", "path" => Some(V::Ctx(p)))];
                }
            }
            return Some(vec![prog]);
        }
    }
    let mut cands: Vec<(&str, V)> = Vec::new();
    for (slot, kind, _) in &c.slots {
        let key = (0usize, j, slot_name(name, slot));
        let Some(&p) = place.get(&key) else { continue };
        let kind = if PKG_CMDS.contains(&name) { "pkg" } else { kind.as_str() };
        if PRON_KINDS.contains(&kind) {
            cands.push((kind, V::Ctx(p)));
        }
    }
    let (kind, cv) = *cands.last()?;
    let kind: &'static str = PRON_KINDS.iter().find(|x| **x == kind).unwrap();
    Some(pronoun_followup(q, kind, cv))
}

/// "stop nginx and disable it", "make a venv and activate it": the second step points at the first's words.
fn chain_request(q: &mut Req) -> Result<Vec<Pipe>, String> {
    for _ in 0..30 {
        let mark = q.chunks.len();
        let pipe = command_request(q, None)?;
        let (name, args) = pipe[0].clone();
        let steps = next_steps(name);
        let c = CMDS.by[name];
        let mut cands: Vec<(&'static str, V)> = Vec::new();
        for (slot, kind, _) in &c.slots {
            let kind = if PKG_CMDS.contains(&name) { "pkg" } else { kind.as_str() };
            if let Some(v) = arg(&args, slot) {
                if let Some(kk) = PRON_KINDS.iter().find(|x| **x == kind) {
                    cands.push((kk, v));
                }
            }
        }
        if steps.is_empty() && cands.is_empty() {
            q.chunks.truncate(mark);
            continue;
        }
        let j = q.c(&["and", "and", "and then", "then", ", then", "and", "and after that", "and also", ","]);
        q.lit(j);
        if !steps.is_empty() && (cands.is_empty() || q.f() < 0.4) {
            let st = &steps[q.below(steps.len())];
            let w = q.c(st.phrases);
            q.lit(w);
            let mut prog = st.prog.clone();
            if prog[0].0 == "venv_on" {
                prog = vec![("venv_on", vec![("path", arg(&args, "path"))])];
            }
            return Ok(vec![pipe, prog]);
        }
        let (kind, cv) = *cands.last().unwrap();
        let second = pronoun_followup(q, kind, cv).remove(0);
        return Ok(vec![pipe, second]);
    }
    Err("no chain".into())
}

/// "find all pdfs in x and delete them": a listing request, then what to do with what was found.
fn find_then(q: &mut Req) -> Pipe {
    let pipe = loop {
        let mark = q.chunks.len();
        let pipe = gen::files_request(q);
        if pipe.last().map(|s| s.0) == Some("fshow") && !pipe.iter().any(|s| s.0 == "fnames" || s.0 == "fcat") {
            break pipe;
        }
        q.chunks.truncate(mark);
    };
    let j = q.c(&["and", "and", "and then", "then", ", then", "and", ", and"]);
    q.lit(j);
    let mut body: Pipe = pipe[..pipe.len() - 1].to_vec();
    let roll = q.f();
    if roll < 0.6 {
        let opts: Vec<&(&str, &[&str])> = FOLLOW_SINK.iter().filter(|(n, _)| *n != "fshow").collect();
        let (name, phrases) = *opts[q.below(opts.len())];
        let w = q.c(phrases);
        q.lit(w);
        body.extend(frame_stages(sname_or(name), &[]));
        return body;
    }
    if roll < 0.9 {
        let (name, phrases, slot, kind) = FOLLOW_SINK_ARG[q.below(FOLLOW_SINK_ARG.len())];
        let p = q.c(phrases);
        let kk = fill(q, p, kind);
        body.extend(frame_stages(sname_or(name), &[(slot_static(slot), kk)]));
        return body;
    }
    let z = zip_followup(q);
    body.push(z);
    body
}

static FOLLOW_AGAIN_SUDO: &[&str] = &["again with sudo", "sudo that", "run that as root", "redo it with sudo", "sudo !!", "same but with sudo", "retry as root",
    "please do that as root", "with sudo", "as root", "do that again as root", "try again with sudo", "use sudo", "sudo",
    "run it with sudo", "needs sudo", "permission denied, try sudo"];
static FOLLOW_OUT: &[(&[&str], &str, &str, &str)] = &[
    (&["search that for {}", "grep that for {}", "only lines with {}", "filter that for {}", "grep {} in the output", "find {} in that output",
        "just the lines containing {}", "search the output for {}", "show only lines mentioning {}", "pipe that to grep {}"], "match", "text", "text"),
    (&["hide lines with {}", "that but without {}", "filter out {} from the output", "exclude lines containing {}"], "nomatch", "text", "text"),
    (&["count the lines", "how many lines is that", "count that", "how many lines", "count the output lines", "pipe that to wc -l",
        "how many lines did that print"], "lcount", "", ""),
    (&["save that to {}", "save the output to {}", "write that to {}", "put the output in {}", "dump that into {}", "redirect that to {}",
        "save the output of that to {}"], "save", "to", "textfile"),
    (&["append that to {}", "add the output to {}", "append the output to {}"], "lappend", "to", "textfile"),
    (&["just the first {} lines", "only the top {}", "first {} lines of that", "show only the first {}", "head that to {} lines"], "ltake", "n", "count"),
    (&["just the last {} lines", "last {} lines of that", "only the last {}", "tail that to {} lines"], "llast", "n", "count"),
    (&["copy that to the clipboard", "copy the output", "put that output on my clipboard", "copy that"], "clip", "", ""),
    (&["with line numbers", "number those lines"], "lnum", "", ""),
    (&["sort that", "sort the output", "sorted", "sort it alphabetically", "pipe it through sort"], "lsort", "", ""),
    (&["unique lines of that", "sort that and remove duplicates", "sort and dedupe that"], "sortu", "", ""),
];

/// Follow-ups about the previous command as a whole: run it again, as root, or do something with its output.
fn output_followup(q: &mut Req) -> Result<Pipe, String> {
    let roll = q.f();
    if roll < 0.24 {
        let n = q.c(&["again", "again", "again", "watch_last", "time_last"]);
        return command_request(q, Some(n));
    }
    if roll < 0.4 {
        let w = q.c(FOLLOW_AGAIN_SUDO);
        q.lit(w);
        return Ok(vec![st!("sudo"), st!("again")]);
    }
    let (phrases, name, slot, kind) = FOLLOW_OUT[q.below(FOLLOW_OUT.len())];
    let p = q.c(phrases);
    let kk = fill(q, p, if kind.is_empty() { "text" } else { kind });
    let ops: Pipe = match name {
        "lsort" => vec![("lsort", vec![("key", e("alpha")), ("order", e("asc"))])],
        "sortu" => vec![("lsort", vec![("key", e("alpha")), ("order", e("asc"))]), st!("uniq")],
        _ => vec![(sname(name), if slot.is_empty() { Vec::new() } else { vec![(slot_static(slot), kk)] })],
    };
    let mut out = vec![st!("last")];
    out.extend(ops);
    if !["lcount", "save", "lappend", "clip"].contains(&name) {
        out.push(st!("lshow"));
    }
    Ok(out)
}

const PREV: &[&str] = &["files", "lines", "lines2", "procs", "procs2", "single", "paths"];
const PREV_WT: &[f64] = &[38.0, 8.0, 6.0, 7.0, 2.0, 30.0, 6.0];

/// A previous request, as the context a follow-up will see: (program with real values, context words, positions).
fn previous(r: &mut Rng) -> Result<(DProg, Vec<String>, Where), String> {
    for _ in 0..50 {
        let mut qa = Req::new(r);
        let pipe = one_pipeline(&mut qa, PREV, PREV_WT)?;
        let ex = finish(&qa.chunks, &[pipe], &[])?;
        if ex.out.len() > 40 {
            continue;
        }
        let prog = delinearise(&ex.out, &ex.words);
        let (ctx, place) = context_words(&prog);
        if ctx.len() <= 30 {
            return Ok((prog, ctx, place));
        }
    }
    Err("no short previous request".into())
}

fn raw_context(r: &mut Rng) -> Vec<String> {
    let c = RAW_COMMANDS[r.below(RAW_COMMANDS.len())];
    let mut w = vec!["@raw".to_string()];
    w.extend(tok::words(c).into_iter().take(12));
    w
}

const HEADS_FILES: &[&str] = &["files", "paths"];
const HEADS_LINES: &[&str] = &["lines", "history", "env", "pkgs", "services", "journal", "dmesg", "clipboard", "last"];

/// A request that builds on the previous one.
pub fn followup(r: &mut Rng) -> Result<Example, String> {
    for _ in 0..40 {
        let (prog, ctx): (Option<Vec<Pipe>>, Vec<String>);
        let mut chunks;
        {
            if r.f() < 0.16 {
                ctx = raw_context(r);
                let mut q = Req::new(r);
                let p = output_followup(&mut q)?;
                noise(&mut q);
                chunks = std::mem::take(&mut q.chunks);
                prog = Some(vec![p]);
            } else {
                let (a, c, place) = previous(r)?;
                ctx = c;
                let mut q = Req::new(r);
                let pipe = &a[0];
                let head = if pipe[0].0 == "sudo" { pipe[1].0 } else { pipe[0].0 };
                if q.f() < 0.3 {
                    let w = q.c(&["ok", "now", "and", "ok now", "great,", "then", "alright", "cool,", "and then", "next", "also", "please", "so"]);
                    q.lit(w);
                }
                let p = if HEADS_FILES.contains(&head) {
                    files_followup(&mut q, pipe, &place)
                } else if HEADS_LINES.contains(&head) {
                    lines_followup(&mut q, pipe, &place)
                } else if head == "procs" || head == "port" {
                    procs_followup(&mut q, pipe, &place)
                } else if q.f() < 0.14 {
                    Some(vec![output_followup(&mut q)?])
                } else {
                    single_followup(&mut q, pipe, &place)
                };
                if p.is_none() {
                    continue;
                }
                noise(&mut q);
                chunks = std::mem::take(&mut q.chunks);
                prog = p;
            }
        }
        let prog = prog.unwrap();
        let ex = finish(&chunks, &prog, &ctx)?;
        chunks.clear();
        if ex.n_req > 0 && ex.out.len() <= MAX_OUT {
            return Ok(ex);
        }
    }
    Err("no follow-up could be made".into())
}

// --------------------------------------------------------------------------- surface noise and whole examples
const FILLERS: &[&str] = &["uh", "like", "just", "really", "actually", "basically", "um", "kinda", "maybe", "quickly"];
const DROPPABLE: &[&str] = &["the", "a", "an", "all", "me", "my", "of", "that", "are", "is"];

/// Small imperfections on the literal (non-argument) words: case, typos, a filler word, a dropped small word.
pub fn noise(q: &mut Req) {
    let style = q.f();
    let mut chunks = std::mem::take(&mut q.chunks);
    for (text, key) in chunks.iter_mut() {
        if key.is_none() {
            if style < 0.72 {
                *text = text.to_lowercase();
            }
            if q.f() < 0.02 {
                let mut ws: Vec<String> = text.split_whitespace().map(str::to_string).collect();
                if !ws.is_empty() {
                    let j = q.below(ws.len());
                    ws[j] = gen::typo(q, &ws[j]);
                    *text = ws.join(" ");
                }
            }
        }
    }
    if (0.72..0.92).contains(&style) && !chunks.is_empty() && chunks[0].1.is_none() && !chunks[0].0.is_empty() {
        let t = &chunks[0].0;
        let mut c = t.chars();
        let first = c.next().unwrap();
        chunks[0].0 = first.to_uppercase().collect::<String>() + c.as_str();
    }
    if q.f() < 0.04 {
        let lits: Vec<usize> = chunks.iter().enumerate().filter(|(_, (t, k))| k.is_none() && t.split_whitespace().count() > 1).map(|(i, _)| i).collect();
        if !lits.is_empty() {
            let i = lits[q.below(lits.len())];
            let mut ws: Vec<&str> = chunks[i].0.split_whitespace().collect();
            let drop: Vec<usize> = ws.iter().enumerate().filter(|(_, w)| DROPPABLE.contains(&w.to_lowercase().as_str())).map(|(j, _)| j).collect();
            if !drop.is_empty() {
                let d = drop[q.below(drop.len())];
                ws.remove(d);
                chunks[i].0 = ws.join(" ");
            }
        }
    }
    if q.f() < 0.035 && !chunks.is_empty() {
        let pos = if q.f() < 0.5 { q.below(chunks.len() + 1) } else { 0 };
        let w = q.c(FILLERS);
        chunks.insert(pos, (w.to_string(), None));
    }
    if q.f() < 0.12 && !chunks.is_empty() && chunks.last().unwrap().1.is_none() {
        let p = q.c(&["?", ".", "!", "?"]);
        chunks.last_mut().unwrap().0.push_str(p);
    }
    chunks.retain(|c| !c.0.is_empty());
    q.chunks = chunks;
}

pub const MAX_REQUEST: usize = 44;
pub const MAX_TOTAL: usize = 80;
pub const MAX_OUT: usize = 72;

/// One training pair. Words after the request are the context.
pub fn example(r: &mut Rng) -> Result<Example, String> {
    if r.f() < 0.3 {
        return followup(r);
    }
    let mut q = Req::new(r);
    let roll = q.f();
    let program: Vec<Pipe> = if roll < 0.035 {
        gen::mkdir_then(&mut q)
    } else if roll < 0.075 {
        chain_request(&mut q)?
    } else if roll < 0.115 {
        vec![find_then(&mut q)]
    } else if roll < 0.2 {
        let mut p = vec![one_pipeline(&mut q, KINDS, KIND_WT)?];
        let j = q.c(JOIN);
        q.lit(j);
        p.push(one_pipeline(&mut q, KINDS, KIND_WT)?);
        if q.f() < 0.15 {
            let j = q.c(JOIN);
            q.lit(j);
            p.push(one_pipeline(&mut q, KINDS, KIND_WT)?);
        }
        p
    } else {
        vec![one_pipeline(&mut q, KINDS, KIND_WT)?]
    };
    let mut ctx: Vec<String> = Vec::new();
    if q.f() < 0.3 {
        ctx = if q.f() < 0.3 { raw_context(q.r) } else { previous(q.r)?.1 };
    }
    noise(&mut q);
    finish(&q.chunks, &program, &ctx)
}

/// An example that passes the length filters (mix.dataset), and how many were rejected or failed on the way.
pub fn accepted(r: &mut Rng, errors: &mut u64, rejected: &mut u64) -> Example {
    loop {
        match example(r) {
            Ok(ex) => {
                if ex.n_req >= 1 && ex.n_req <= MAX_REQUEST && ex.words.len() <= MAX_TOTAL && ex.out.len() <= MAX_OUT {
                    return ex;
                }
                *rejected += 1;
            }
            Err(_) => *errors += 1,
        }
    }
}
