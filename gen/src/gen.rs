//! File, line and process requests: a port of the request builders in kural/gen.py.

use crate::pools::*;
use crate::req::*;
use crate::{ph, st};

pub struct Filters {
    pub stages: Vec<(&'static str, Args)>,
    pub negated: Vec<&'static str>,
    pub pre: Vec<Phrase>,
    pub post: Vec<Phrase>,
    pub noun: Option<(String, u32)>,
}

impl Filters {
    pub fn has(&self, n: &str) -> bool {
        self.stages.iter().any(|(m, _)| *m == n)
    }
}

/// Sample filters: canonical stages, phrases before the noun, phrases after it, and a file type used as the noun.
pub fn file_filters(q: &mut Req, dirs: bool) -> Filters {
    let mut fl = Filters { stages: Vec::new(), negated: Vec::new(), pre: Vec::new(), post: Vec::new(), noun: None };
    let names = ["ext", "name", "larger", "smaller", "newer", "older", "contains", "empty", "since", "before", "hidden", "owner", "isexec", "depth"];
    let probs: [f64; 14] = if !dirs {
        [0.55, 0.14, 0.2, 0.06, 0.2, 0.2, 0.12, 0.05, 0.035, 0.03, 0.03, 0.03, 0.025, 0.02]
    } else {
        [0.0, 0.3, 0.0, 0.0, 0.2, 0.2, 0.0, 0.2, 0.03, 0.02, 0.04, 0.04, 0.0, 0.03]
    };
    let mut chosen: Vec<&str> = names.iter().zip(probs).filter(|_| true).filter_map(|(n, p)| if q.f() < p { Some(*n) } else { None }).collect();
    for (a, b) in [("newer", "older"), ("since", "newer"), ("before", "older"), ("since", "before")] {
        if chosen.contains(&a) && chosen.contains(&b) {
            let drop = q.c(&[a, b]);
            chosen.retain(|x| *x != drop);
        }
    }
    if chosen.contains(&"larger") && chosen.contains(&"smaller") && q.f() < 0.7 {
        chosen.retain(|x| *x != "smaller");
    }
    if chosen.contains(&"empty") {
        chosen.retain(|x| !["larger", "smaller", "contains"].contains(x));
    }
    q.r.shuffle(&mut chosen);
    let keep = q.pick(&[0usize, 1, 1, 1, 2, 2, 2, 3, 3, 4]);
    chosen.truncate(keep);
    for n in chosen {
        let n: &'static str = sname(n);
        let neg = ["ext", "name", "contains", "empty", "hidden", "owner", "isexec"].contains(&n) && q.f() < 0.07;
        if neg {
            fl.negated.push(n);
        }
        match n {
            "ext" => {
                let multi = if q.f() < 0.07 {
                    let k = q.pick(&[2usize, 2, 3]);
                    let es = q.r.sample(&EXTS, k);
                    let conj = q.c(&["or", "and"]);
                    Some(if es.len() == 2 { format!("{} {conj} {}", es[0], es[1]) } else { format!("{}, {} {conj} {}", es[0], es[1], es[2]) })
                } else {
                    None
                };
                let k = q.key();
                if neg {
                    let e = match multi {
                        Some(e) => e,
                        None => (if q.f() < 0.4 { q.c(EXT_NAMES) } else { q.cat(&[&EXTS, EXT_PLURALS]) }).to_string(),
                    };
                    let lead = q.c(&["that are not", "except", "other than", "excluding", "but not", "that aren't", "ignoring",
                        "apart from", "not including", "except for", "but skip", "leaving out", "except the"]);
                    let tail = if EXT_PLURALS.contains(&e.as_str()) { "" } else { q.c(&["files", "", "", "ones"]) };
                    fl.post.push(ph![lead, (e, k), tail]);
                } else if let Some(e) = multi {
                    if q.f() < 0.4 {
                        let lead = q.c(&["that are", "of type", "ending in", "with extension", "in"]);
                        let tail = q.c(&["", "files", "format"]);
                        fl.post.push(ph![lead, (e, k), tail]);
                    } else {
                        fl.pre.push(ph![(e, k)]);
                    }
                } else {
                    let style = q.f();
                    if style < 0.14 {
                        fl.noun = Some((q.c(EXT_PLURALS).to_string(), k));
                    } else if style < 0.6 {
                        let e = if q.f() < 0.45 { q.c(EXT_NAMES) } else { q.c(&EXTS) };
                        let e = if EXTS.contains(&e) {
                            let opts = [e.to_string(), e.to_string(), format!(".{e}"), format!("*.{e}")];
                            q.pick(&opts)
                        } else {
                            e.to_string()
                        };
                        fl.pre.push(ph![(e, k)]);
                    } else {
                        let e = q.c(&EXTS);
                        let lead = q.c(&["ending in", "with the extension", "with extension", "of type", "with the", "that end in",
                            "whose extension is", "that are", "in", "with a", "having the extension"]);
                        let e2 = if lead == "ending in" || lead == "that end in" || q.f() < 0.5 { format!(".{e}") } else { e.to_string() };
                        let tail = match lead {
                            "with the" | "with a" => "extension",
                            "in" => "format",
                            "that are" => q.c(&["files", ""]),
                            _ => "",
                        };
                        fl.post.push(ph![lead, (e2, k), tail]);
                    }
                }
                fl.stages.push(st!("ext", "ext" => Some(V::K(k))));
            }
            "name" => {
                let mode = q.c(&["is", "has", "has", "starts", "starts", "ends"]);
                let (t, lead, nlead): (String, Option<&str>, &str) = match mode {
                    "is" => {
                        let mut t = if !dirs { q.filename(None) } else { q.word().to_string() };
                        if q.f() < 0.25 {
                            let p = q.c(&["*", ""]);
                            t = format!("{p}{}*", q.word());
                        }
                        let lead = q.c(&["named", "called", "with the name", "whose name is", "by the name of", "matching", "named exactly"]);
                        let nlead = q.c(&["not named", "not called", "whose name is not", "that aren't called", "except the one named"]);
                        (t, Some(lead), nlead)
                    }
                    "has" => {
                        let t = q.cat(&[&WORDS, &["2023", "2024", "v2", "old", "copy", "final"]]).to_string();
                        let nlead = q.c(&["whose name doesn't contain", "without", "with names not containing", "that don't have", "not containing"]);
                        (t, None, nlead)
                    }
                    "starts" => {
                        let t = q.cat(&[&WORDS, &["tmp", "IMG", "DSC", "test_", "draft-", "2024", "."]]).to_string();
                        let lead = q.c(&["starting with", "that start with", "whose name begins with", "beginning with",
                            "whose names start with", "with names starting with", "prefixed with", "that begin with"]);
                        let nlead = q.c(&["not starting with", "that don't start with", "whose name doesn't begin with", "that do not begin with"]);
                        (t, Some(lead), nlead)
                    }
                    _ => {
                        let t = q.c(&["_old", "-backup", "copy", "2023", "_final", "~", "-v2", "_test", "bak", ".orig"]).to_string();
                        let lead = q.c(&["ending with", "whose name ends in", "that end with", "whose names end with",
                            "with names ending in", "suffixed with"]);
                        let nlead = q.c(&["not ending with", "that don't end with", "whose name doesn't end in"]);
                        (t, Some(lead), nlead)
                    }
                };
                let k = q.key();
                if neg {
                    let mut p = ph![nlead, (t, k)];
                    if mode == "has" && ["without", "that don't have", "not containing"].contains(&nlead) {
                        p.push(P::from("in the name"));
                    }
                    fl.post.push(p);
                } else if dirs && mode == "is" && q.f() < 0.45 {
                    let w = q.cat(&[&["node_modules", "__pycache__", ".git", "build", "dist", "target", "venv", ".cache", "tmp"], &WORDS[..40]]);
                    fl.pre.push(ph![(w, k)]);
                } else if lead.is_none() {
                    let p = match q.below(6) {
                        0 => ph!["with", (t, k), "in the name"],
                        1 => ph!["whose name contains", (t, k)],
                        2 => ph!["with names containing", (t, k)],
                        3 => ph!["that have", (t, k), "in their name"],
                        4 => ph!["with", (t, k), "in their names"],
                        _ => ph!["containing", (t, k), "in the filename"],
                    };
                    fl.post.push(p);
                } else {
                    fl.post.push(ph![lead.unwrap(), (t, k)]);
                }
                fl.stages.push(st!("name", "mode" => Some(V::E(mode)), "text" => Some(V::K(k))));
            }
            "larger" | "smaller" => {
                if q.f() < 0.14 {
                    let adj = q.c(vague(n));
                    if q.f() < 0.7 {
                        fl.pre.push(ph![adj]);
                    } else {
                        let l = q.c(&["that are", "which are", "that look", "that seem"]);
                        fl.post.push(ph![l, adj]);
                    }
                    fl.stages.push(st!(n, "size" => None));
                    continue;
                }
                let k = q.key();
                let s = q.size();
                let lead = if n == "larger" {
                    q.c(&["larger than", "bigger than", "over", "above", "more than", "greater than", "that are bigger than",
                        "exceeding", "that are larger than", "which are over", "that take more than", "at least",
                        "heavier than", "that exceed", "weighing more than", "bigger then", "larger then"])
                } else {
                    q.c(&["smaller than", "under", "below", "less than", "that are smaller than", "which are under",
                        "at most", "lighter than", "that are less than", "not bigger than", "smaller then"])
                };
                fl.post.push(ph![lead, (s, k)]);
                fl.stages.push(st!(n, "size" => Some(V::K(k))));
            }
            "newer" => {
                if q.f() < 0.12 {
                    let adj = q.c(vague(n));
                    if q.f() < 0.7 {
                        fl.pre.push(ph![adj]);
                    } else {
                        let l = q.c(&["that are", "which are"]);
                        fl.post.push(ph![l, adj]);
                    }
                    fl.stages.push(st!(n, "age" => None));
                    continue;
                }
                let k = q.key();
                if q.f() < 0.3 {
                    let d = q.c(&["today", "yesterday", "this week", "this month", "this year", "last week", "last month"]);
                    let lead = q.c(&["modified", "changed", "from", "edited", "updated", "touched", "created", "that changed",
                        "that were modified", "i changed", "that i edited", "modified since", "changed since", "since"]);
                    fl.post.push(ph![lead, (d, k)]);
                } else {
                    let d = q.duration();
                    let p = match q.below(12) {
                        0 => ph!["modified in the last", (d, k)],
                        1 => ph!["changed in the past", (d, k)],
                        2 => ph!["from the last", (d, k)],
                        3 => ph!["newer than", (d, k)],
                        4 => ph!["edited within the last", (d, k)],
                        5 => ph!["that changed in the last", (d, k)],
                        6 => ph!["less than", (d, k), "old"],
                        7 => ph!["created in the last", (d, k)],
                        8 => ph!["updated in the last", (d, k)],
                        9 => ph!["modified within", (d, k)],
                        10 => ph!["touched in the last", (d, k)],
                        _ => ph!["from the past", (d, k)],
                    };
                    fl.post.push(p);
                }
                fl.stages.push(st!("newer", "age" => Some(V::K(k))));
            }
            "older" => {
                if q.f() < 0.12 {
                    let adj = q.c(vague(n));
                    if q.f() < 0.7 {
                        fl.pre.push(ph![adj]);
                    } else {
                        let l = q.c(&["that are", "which are"]);
                        fl.post.push(ph![l, adj]);
                    }
                    fl.stages.push(st!(n, "age" => None));
                    continue;
                }
                let k = q.key();
                let d = q.duration();
                let p = match q.below(13) {
                    0 => ph!["older than", (d, k)],
                    1 => ph!["not modified in", (d, k)],
                    2 => ph!["that haven't been touched in", (d, k)],
                    3 => ph!["that haven't changed in over", (d, k)],
                    4 => ph!["more than", (d, k), "old"],
                    5 => ph!["last modified more than", (d, k), "ago"],
                    6 => ph!["untouched for", (d, k)],
                    7 => ph!["from more than", (d, k), "ago"],
                    8 => ph!["at least", (d, k), "old"],
                    9 => ph!["that are older than", (d, k)],
                    10 => ph!["not changed for", (d, k)],
                    11 => ph!["unchanged for over", (d, k)],
                    _ => ph!["that nobody has touched in", (d, k)],
                };
                fl.post.push(p);
                fl.stages.push(st!("older", "age" => Some(V::K(k))));
            }
            "since" => {
                let k = q.key();
                let d = q.c(DATES);
                let lead = q.c(&["modified since", "changed after", "from", "modified after", "created after", "since",
                    "edited after", "that changed on or after", "updated since", "newer than", "touched after",
                    "changed since", "modified on or after", "dated after"]);
                let mut p = ph![lead, (d, k)];
                if q.f() < 0.1 {
                    p.push(P::from("onwards"));
                }
                fl.post.push(p);
                fl.stages.push(st!("since", "date" => Some(V::K(k))));
            }
            "before" => {
                let k = q.key();
                let d = q.c(DATES);
                let lead = q.c(&["modified before", "from before", "last changed before", "that haven't changed since",
                    "created before", "edited before", "older than", "untouched since", "dated before",
                    "changed before", "not modified since"]);
                fl.post.push(ph![lead, (d, k)]);
                fl.stages.push(st!("before", "date" => Some(V::K(k))));
            }
            "contains" => {
                let k = q.key();
                let t = if q.f() < 0.6 { q.c(PHRASES) } else { q.word() };
                let quote = t.contains(' ') || q.f() < 0.2;
                let t2 = (t.to_string(), k, quote);
                let p = if neg {
                    match q.below(7) {
                        0 => ph!["that don't contain", &t2],
                        1 => ph!["that do not mention", &t2],
                        2 => ph!["without", &t2, "in them"],
                        3 => ph!["not containing", &t2],
                        4 => ph!["that lack", &t2],
                        5 => ph!["missing the text", &t2],
                        _ => ph!["where", &t2, "does not appear"],
                    }
                } else {
                    match q.below(11) {
                        0 => ph!["containing", &t2],
                        1 => ph!["that contain", &t2],
                        2 => ph!["that mention", &t2],
                        3 => ph!["with", &t2, "in them"],
                        4 => ph!["which include the text", &t2],
                        5 => ph!["that have the string", &t2],
                        6 => ph!["mentioning", &t2],
                        7 => ph!["with the word", &t2],
                        8 => ph!["where", &t2, "appears"],
                        9 => ph!["that include", &t2],
                        _ => ph!["with", &t2, "inside"],
                    }
                };
                fl.post.push(p);
                fl.stages.push(st!("contains", "text" => Some(V::K(k))));
            }
            "empty" => {
                if neg {
                    if q.f() < 0.4 {
                        let w = q.c(&["non-empty", "nonempty", "non empty"]);
                        fl.pre.push(ph![w]);
                    } else {
                        let w = q.c(&["that are not empty", "that aren't empty", "which have something in them", "that are not blank",
                            "with something in them"]);
                        fl.post.push(ph![w]);
                    }
                } else if q.f() < 0.5 {
                    fl.pre.push(ph!["empty"]);
                } else {
                    let last = if !dirs { "of zero size" } else { "with no files inside" };
                    let w = q.c(&["that are empty", "with nothing in them", "which are empty", "that have nothing in them", last]);
                    fl.post.push(ph![w]);
                }
                fl.stages.push(st!("empty"));
            }
            "hidden" => {
                if neg {
                    let w = q.c(&["that are not hidden", "excluding hidden ones", "ignoring dotfiles", "but not hidden ones",
                        "skipping hidden ones", "except hidden ones", "that don't start with a dot"]);
                    fl.post.push(ph![w]);
                } else if q.f() < 0.6 {
                    fl.pre.push(ph!["hidden"]);
                } else {
                    let w = q.c(&["that are hidden", "starting with a dot", "which are hidden", "whose names begin with a dot"]);
                    fl.post.push(ph![w]);
                }
                fl.stages.push(st!("hidden"));
            }
            "owner" => {
                let k = q.key();
                let o = q.c(OWNERS);
                if (o == "me" || o == "myself") && !neg && q.f() < 0.5 {
                    let a = q.c(&["that", "which"]);
                    let b = q.c(&["own", "am the owner of"]);
                    fl.post.push(ph![a, ("i", k), b]);
                } else if neg {
                    let l = q.c(&["not owned by", "that don't belong to", "not belonging to", "that aren't owned by", "owned by someone other than"]);
                    fl.post.push(ph![l, (o, k)]);
                } else {
                    let l = q.c(&["owned by", "belonging to", "that belong to", "whose owner is", "that are owned by", "of user", "created by"]);
                    fl.post.push(ph![l, (o, k)]);
                }
                fl.stages.push(st!("owner", "user" => Some(V::K(k))));
            }
            "isexec" => {
                if neg {
                    let w = q.c(&["that are not executable", "that aren't executable", "without the execute bit", "that can't be run"]);
                    fl.post.push(ph![w]);
                } else if q.f() < 0.6 {
                    fl.pre.push(ph!["executable"]);
                } else {
                    let w = q.c(&["that are executable", "i can run", "with the execute bit", "that can be run", "marked executable"]);
                    fl.post.push(ph![w]);
                }
                fl.stages.push(st!("isexec"));
            }
            _ => {
                // depth
                let k = q.key();
                let d = q.c(&["1", "2", "2", "3", "3", "4", "5", "two", "three"]);
                let p = match q.below(6) {
                    0 => ph!["at most", (d, k), "levels deep"],
                    1 => ph!["up to", (d, k), "levels down"],
                    2 => ph!["no deeper than", (d, k), "levels"],
                    3 => ph!["within", (d, k), "levels"],
                    4 => ph!["max depth", (d, k)],
                    _ => ph!["going", (d, k), "levels deep"],
                };
                fl.post.push(p);
                fl.stages.push(st!("depth", "n" => Some(V::K(k))));
            }
        }
    }
    fl
}

// --------------------------------------------------------------------------- frame actions
#[derive(Clone, Copy)]
pub enum F {
    L(&'static str),
    Np,
    Np1,
    Loc,
    A(&'static str),
}
use F::{Loc, Np, Np1, A as Fa, L as Fl};

pub static FRAME_NAMES: &[&str] = &["trash", "details", "fgrep", "fopen", "fedit", "chmod", "chown", "rn_lower", "rn_upper", "rn_spaces",
    "rn_replace", "rn_prefix", "rn_suffix", "rn_ext", "replace", "fhash", "by_ext", "dupes", "upload", "names_save", "cat_count", "cat_wcount"];

pub fn frames(sink: &str) -> Vec<Vec<F>> {
    macro_rules! fr {
        ($([$($p:expr),*]),* $(,)?) => { vec![$(vec![$($p),*]),*] };
    }
    match sink {
        "trash" => fr![[Fl("trash"), Np], [Fl("move"), Np, Fl("to the trash")], [Fl("send"), Np, Fl("to the trash")], [Fl("recycle"), Np], [Fl("bin"), Np],
            [Fl("put"), Np, Fl("in the trash")], [Fl("move"), Np, Fl("to the recycle bin")], [Fl("throw"), Np, Fl("in the bin")], [Fl("move"), Np, Fl("to trash")]],
        "details" => fr![[Fl("show"), Np, Fl("with details")], [Fl("long listing of"), Np], [Fl("ls -l"), Np], [Fl("list"), Np, Fl("with permissions and sizes")],
            [Fl("show"), Np, Fl("with their permissions")], [Fl("show"), Np, Fl("with owner and date")], [Fl("detailed list of"), Np],
            [Fl("list"), Np, Fl("in long format")], [Fl("show permissions of"), Np], [Fl("who owns"), Np], [Fl("list"), Np, Fl("with full details")]],
        "fgrep" => fr![[Fl("search"), Np, Fl("for"), Fa("text")], [Fl("grep for"), Fa("text"), Fl("in"), Np], [Fl("find"), Fa("text"), Fl("in"), Np],
            [Fl("look for"), Fa("text"), Fl("in"), Np], [Fl("search for"), Fa("text"), Fl("in"), Np], [Fl("grep"), Fa("text"), Fl("in"), Np],
            [Fl("show lines containing"), Fa("text"), Fl("in"), Np], [Fl("where is"), Fa("text"), Fl("used in"), Np],
            [Fl("find every line mentioning"), Fa("text"), Fl("in"), Np], [Fl("grep -r"), Fa("text"), Np],
            [Fl("which lines in"), Np, Fl("mention"), Fa("text")], [Fl("find where"), Fa("text"), Fl("appears in"), Np],
            [Fl("show me every place"), Fa("text"), Fl("shows up in"), Np], [Fl("search through"), Np, Fl("for"), Fa("text")]],
        "fopen" => fr![[Fl("open"), Np], [Fl("open up"), Np], [Fl("xdg-open"), Np], [Fl("open"), Np, Fl("with the default app")]],
        "fedit" => fr![[Fl("edit"), Np], [Fl("open"), Np, Fl("in my editor")], [Fl("open"), Np, Fl("in vim")], [Fl("open"), Np, Fl("for editing")], [Fl("let me edit"), Np]],
        "chmod" => fr![[Fl("make"), Np, Fa("mode")], [Fl("chmod"), Fa("mode"), Np], [Fl("set the permissions of"), Np, Fl("to"), Fa("mode")],
            [Fl("give"), Np, Fa("mode"), Fl("permissions")], [Fl("change the mode of"), Np, Fl("to"), Fa("mode")], [Fl("set"), Np, Fl("to"), Fa("mode")],
            [Fl("mark"), Np, Fl("as"), Fa("mode")], [Fl("change permissions on"), Np, Fl("to"), Fa("mode")], [Fl("make"), Np, Fl("all"), Fa("mode")]],
        "chown" => fr![[Fl("change the owner of"), Np, Fl("to"), Fa("owner")], [Fl("chown"), Fa("owner"), Np], [Fl("give"), Np, Fl("to"), Fa("owner")],
            [Fl("make"), Fa("owner"), Fl("the owner of"), Np], [Fl("set the owner of"), Np, Fl("to"), Fa("owner")],
            [Fl("change ownership of"), Np, Fl("to"), Fa("owner")], [Fl("transfer ownership of"), Np, Fl("to"), Fa("owner")]],
        "rn_lower" => fr![[Fl("rename"), Np, Fl("to lowercase")], [Fl("lowercase the names of"), Np], [Fl("make the names of"), Np, Fl("lowercase")],
            [Fl("convert the filenames of"), Np, Fl("to lower case")], [Fl("make"), Np, Fl("have lowercase names")],
            [Fl("lowercase"), Np, Fl("filenames")], [Fl("rename"), Np, Fl("so the names are all lowercase")], [Fl("make the filenames of"), Np, Fl("all lower case")],
            [Fl("make every filename"), Loc, Fl("lowercase")], [Fl("lowercase all the filenames"), Loc], [Fl("make all file names"), Loc, Fl("lower case")]],
        "rn_upper" => fr![[Fl("rename"), Np, Fl("to uppercase")], [Fl("uppercase the names of"), Np], [Fl("make the names of"), Np, Fl("uppercase")],
            [Fl("convert the filenames of"), Np, Fl("to upper case")], [Fl("rename"), Np, Fl("in all caps")], [Fl("capitalize the names of"), Np],
            [Fl("make the filenames of"), Np, Fl("all caps")], [Fl("make every filename"), Loc, Fl("uppercase")], [Fl("uppercase all the filenames"), Loc]],
        "rn_spaces" => fr![[Fl("replace spaces with underscores in the names of"), Np], [Fl("remove spaces from the names of"), Np],
            [Fl("rename"), Np, Fl("replacing spaces with underscores")], [Fl("get rid of spaces in"), Np, Fl("filenames")],
            [Fl("turn spaces into underscores in the names of"), Np], [Fl("fix the spaces in the names of"), Np],
            [Fl("rename"), Np, Fl("so there are no spaces in the names")], [Fl("get rid of the spaces in every filename"), Loc],
            [Fl("replace spaces with underscores in all filenames"), Loc]],
        "rn_replace" => fr![[Fl("rename"), Np, Fl("replacing"), Fa("old"), Fl("with"), Fa("new")], [Fl("replace"), Fa("old"), Fl("with"), Fa("new"), Fl("in the names of"), Np],
            [Fl("in the filenames of"), Np, Fl("change"), Fa("old"), Fl("to"), Fa("new")], [Fl("rename"), Np, Fl("changing"), Fa("old"), Fl("to"), Fa("new")],
            [Fl("swap"), Fa("old"), Fl("for"), Fa("new"), Fl("in the names of"), Np], [Fl("rename"), Np, Fl("so"), Fa("old"), Fl("becomes"), Fa("new")],
            [Fl("change"), Fa("old"), Fl("to"), Fa("new"), Fl("in the filenames of"), Np]],
        "rn_prefix" => fr![[Fl("add the prefix"), Fa("text"), Fl("to"), Np], [Fl("prefix"), Np, Fl("with"), Fa("text")], [Fl("prepend"), Fa("text"), Fl("to the names of"), Np],
            [Fl("rename"), Np, Fl("adding"), Fa("text"), Fl("at the start")], [Fl("put"), Fa("text"), Fl("in front of the names of"), Np],
            [Fl("add"), Fa("text"), Fl("to the beginning of the names of"), Np]],
        "rn_suffix" => fr![[Fl("add the suffix"), Fa("text"), Fl("to"), Np], [Fl("append"), Fa("text"), Fl("to the names of"), Np],
            [Fl("rename"), Np, Fl("adding"), Fa("text"), Fl("at the end")], [Fl("suffix"), Np, Fl("with"), Fa("text")],
            [Fl("add"), Fa("text"), Fl("to the end of the names of"), Np]],
        "rn_ext" => fr![[Fl("change the extension of"), Np, Fl("to"), Fa("ext")], [Fl("rename"), Np, Fl("to"), Fa("ext")], [Fl("give"), Np, Fl("the extension"), Fa("ext")],
            [Fl("change"), Np, Fl("to"), Fa("ext"), Fl("files")], [Fl("rename"), Np, Fl("to have the"), Fa("ext"), Fl("extension")],
            [Fl("switch the extension of"), Np, Fl("to"), Fa("ext")], [Fl("change the file extension of"), Np, Fl("to"), Fa("ext")]],
        "replace" => fr![[Fl("replace"), Fa("old"), Fl("with"), Fa("new"), Fl("in"), Np], [Fl("change every"), Fa("old"), Fl("to"), Fa("new"), Fl("in"), Np],
            [Fl("swap"), Fa("old"), Fl("for"), Fa("new"), Fl("in"), Np], [Fl("find and replace"), Fa("old"), Fl("with"), Fa("new"), Fl("in"), Np],
            [Fl("in"), Np, Fl("replace"), Fa("old"), Fl("with"), Fa("new")], [Fl("substitute"), Fa("new"), Fl("for"), Fa("old"), Fl("in"), Np],
            [Fl("change"), Fa("old"), Fl("to"), Fa("new"), Fl("in"), Np], [Fl("sed"), Fa("old"), Fl("to"), Fa("new"), Fl("in"), Np],
            [Fl("replace all occurrences of"), Fa("old"), Fl("with"), Fa("new"), Fl("in"), Np], [Fl("replace every"), Fa("old"), Fl("with"), Fa("new"), Fl("across"), Np]],
        "fhash" => fr![[Fl("checksum"), Np], [Fl("sha256 of"), Np], [Fl("hash"), Np], [Fl("compute checksums for"), Np], [Fl("sha256sum"), Np], [Fl("get the hashes of"), Np]],
        "by_ext" => fr![[Fl("count"), Np, Fl("by extension")], [Fl("count"), Np, Fl("by type")], [Fl("how many"), Np1, Fl("of each type")], [Fl("break down"), Np, Fl("by extension")],
            [Fl("what kinds of"), Np1, Fl("are there")], [Fl("file types among"), Np], [Fl("group"), Np, Fl("by extension and count")],
            [Fl("how many of each file type"), Loc], [Fl("which file types are most common"), Loc], [Fl("breakdown of file types"), Loc],
            [Fl("what types of files are"), Loc], [Fl("count files per extension"), Loc]],
        "dupes" => fr![[Fl("find duplicate"), Np1], [Fl("which"), Np1, Fl("are duplicates")], [Fl("find duplicates among"), Np], [Fl("duplicate"), Np1],
            [Fl("find identical"), Np1], [Fl("are there duplicate"), Np1], [Fl("which"), Np1, Fl("are identical")],
            [Fl("find"), Np, Fl("that are copies of each other")], [Fl("look for duplicated"), Np1], [Fl("any duplicate"), Np1]],
        "upload" => fr![[Fl("upload"), Np, Fl("to"), Fa("to")], [Fl("send"), Np, Fl("to"), Fa("to")], [Fl("rsync"), Np, Fl("to"), Fa("to")],
            [Fl("copy"), Np, Fl("to the server"), Fa("to")], [Fl("scp"), Np, Fl("to"), Fa("to")], [Fl("sync"), Np, Fl("to"), Fa("to")],
            [Fl("push"), Np, Fl("to"), Fa("to")], [Fl("transfer"), Np, Fl("to"), Fa("to")], [Fl("copy"), Np, Fl("over to"), Fa("to")]],
        "names_save" => fr![[Fl("save the list of"), Np, Fl("to"), Fa("to")], [Fl("write the names of"), Np, Fl("to"), Fa("to")], [Fl("list"), Np, Fl("into"), Fa("to")],
            [Fl("put the paths of"), Np, Fl("in"), Fa("to")], [Fl("save a list of"), Np, Fl("in"), Fa("to")],
            [Fl("export the names of"), Np, Fl("to"), Fa("to")], [Fl("dump the list of"), Np, Fl("into"), Fa("to")]],
        "cat_count" => fr![[Fl("count the lines in"), Np], [Fl("how many lines of code in"), Np], [Fl("how many lines are in"), Np],
            [Fl("total lines across"), Np], [Fl("count lines of"), Np], [Fl("how many lines do"), Np, Fl("have in total")],
            [Fl("line count of"), Np], [Fl("lines of code in"), Np], [Fl("wc -l on"), Np], [Fl("how many lines of code are in"), Np]],
        _ => fr![[Fl("how many words are in"), Np], [Fl("word count of"), Np], [Fl("count the words in"), Np], [Fl("total words across"), Np]],
    }
}

fn frame_arg_kind(sink: &str, slot: &str) -> &'static str {
    match (sink, slot) {
        ("fgrep", _) => "text",
        ("chmod", _) => "mode",
        ("chown", _) => "owner",
        ("rn_replace", _) => "word",
        ("rn_prefix", _) | ("rn_suffix", _) => "affix",
        ("rn_ext", _) => "ext",
        ("replace", _) => "text",
        ("upload", _) => "remote",
        _ => "textfile",
    }
}

fn frame_wt(f: &str) -> f64 {
    match f {
        "fgrep" => 7.0,
        "trash" | "replace" | "cat_count" => 3.0,
        "details" | "chmod" | "upload" => 2.5,
        "rn_replace" => 2.0,
        "rn_lower" | "rn_ext" | "by_ext" | "dupes" | "fopen" | "names_save" => 1.5,
        _ => 1.0,
    }
}
static FRAME_DIRS: &[&str] = &["trash", "details", "chmod", "chown", "rn_lower", "rn_upper", "rn_spaces", "rn_replace", "rn_prefix", "rn_suffix",
    "upload", "fopen", "names_save"];
static FRAME_SUPER: &[&str] = &["trash", "fopen", "fedit", "upload", "details", "fhash", "chmod"];

/// Write the request for one of the frame actions; returns the action's slot keys.
pub fn frame_emit(q: &mut Req, sink: &str, np: &[P], tailbits: &[Phrase]) -> Vec<(&'static str, u32)> {
    let all = frames(sink);
    let mut seq = all[q.below(all.len())].clone();
    let np_has_arg = np.iter().any(|p| matches!(p, P::A(..)));
    if seq.iter().any(|p| matches!(p, Loc)) && np_has_arg {
        let with: Vec<&Vec<F>> = all.iter().filter(|f| f.iter().any(|p| matches!(p, Np | Np1))).collect();
        seq = with[q.below(with.len())].clone();
    }
    let mut args = Vec::new();
    let mut used_np = false;
    for part in seq {
        match part {
            F::A(slot) => {
                let (text, quote, pre, post) = val(q, frame_arg_kind(sink, slot));
                q.lit(pre);
                let k = q.arg_text(&text, quote);
                args.push((slot, k));
                q.lit(post);
            }
            Np => {
                q.emit(np);
                q.tail(tailbits);
                used_np = true;
            }
            Np1 => {
                q.emit(&np[1..]);
                q.tail(tailbits);
                used_np = true;
            }
            Loc => {
                q.tail(tailbits);
                used_np = true;
            }
            F::L(s) => q.lit(s),
        }
    }
    if !used_np {
        q.tail(tailbits);
    }
    args
}

pub fn frame_stages(sink: &'static str, args: &[(&'static str, Option<V>)]) -> Pipe {
    match sink {
        "names_save" => vec![st!("fnames"), ("save", vec![("to", args.iter().find(|a| a.0 == "to").and_then(|a| a.1))])],
        "cat_count" => vec![st!("fcat"), st!("lcount")],
        "cat_wcount" => vec![st!("fcat"), st!("wcount")],
        _ => vec![(sink, args.to_vec())],
    }
}

fn singular(w: &str) -> &'static str {
    match w {
        "folders" => "folder",
        "directories" => "directory",
        "subfolders" => "subfolder",
        "subdirectories" => "subdirectory",
        "dirs" => "dir",
        "files" => "file",
        "items" => "item",
        "things" => "thing",
        "ones" => "one",
        _ => "file",
    }
}

pub fn files_request(q: &mut Req) -> Pipe {
    let dirs = q.f() < 0.12;
    let mut sinks: Vec<&'static str> = vec!["fshow", "delete", "fcount", "total", "sizes", "move", "copy", "compress"];
    let mut wts: Vec<f64> = if !dirs { vec![30.0, 14.0, 10.0, 6.0, 5.0, 8.0, 8.0, 8.0] } else { vec![30.0, 10.0, 10.0, 2.0, 22.0, 5.0, 5.0, 4.0] };
    for f in FRAME_NAMES {
        if !dirs || FRAME_DIRS.contains(f) {
            sinks.push(f);
            wts.push(frame_wt(f));
        }
    }
    let sink = sinks[q.weighted(&wts)];
    let is_frame = FRAME_NAMES.contains(&sink);
    let mut fl = file_filters(q, dirs);
    let (mut sort, mut take, mut pick): (Option<(&str, &str)>, Option<Option<u32>>, Option<u32>) = (None, None, None);
    let mut sup_phrase: Option<Phrase> = None;
    let mut sort_clause: Option<&str> = None;
    let mut singular_n = false;
    if (["fshow", "delete", "move", "copy", "compress", "sizes"].contains(&sink) || FRAME_SUPER.contains(&sink)) && q.f() < 0.3 {
        let key = q.pick(&[("size", "desc"), ("size", "desc"), ("size", "asc"), ("time", "desc"), ("time", "desc"), ("time", "asc")]);
        sort = Some(key);
        let word = q.c(superlative(key));
        let style = q.f();
        if style < 0.07 && sink != "sizes" {
            let k = q.key();
            pick = Some(k);
            singular_n = true;
            let o = q.c(ORDINALS);
            sup_phrase = Some(ph!["the", (o, k), word]);
        } else if style < 0.6 {
            let n = q.count();
            let k = q.key();
            take = Some(Some(k));
            sup_phrase = Some(match q.below(7) {
                0 | 5 => ph!["the", (n, k), word],
                1 => ph![(n, k), word],
                2 => ph!["the", word, (n, k)],
                3 => ph!["top", (n, k), word],
                4 => ph!["the top", (n, k), word],
                _ => ph![word, (n, k)],
            });
        } else if style < 0.8 {
            singular_n = true;
            take = Some(None);
            sup_phrase = Some(if q.f() < 0.8 { ph!["the", word] } else { ph![word] });
        } else {
            sup_phrase = Some(if q.f() < 0.7 { ph!["the", word] } else { ph![word] });
        }
    } else if sink == "fshow" && q.f() < 0.12 {
        let key = q.pick(SORT_KEYS);
        sort = Some(key);
        sort_clause = Some(q.c(sort_clause_of(key)));
    }
    // location
    let (mut loc_key, mut loc_phrase, mut flat): (Option<u32>, Option<Phrase>, bool) = (None, None, false);
    if q.f() < 0.68 {
        let (text, lpre, lpost) = q.folder();
        let k = q.key();
        loc_key = Some(k);
        if q.f() < 0.07 {
            flat = true;
            let core = ph![lpre, (text, k), lpost];
            let mut p = Vec::new();
            match q.below(5) {
                0 => {
                    p.push(P::from("directly in"));
                    p.extend(core);
                }
                1 => {
                    p.push(P::from("at the top level of"));
                    p.extend(core);
                }
                2 => {
                    p.push(P::from("in"));
                    p.extend(core);
                    p.push(P::from("but not its subfolders"));
                }
                3 => {
                    p.push(P::from("in"));
                    p.extend(core);
                    p.push(P::from("only, not subfolders"));
                }
                _ => {
                    p.push(P::from("just in"));
                    p.extend(core);
                    p.push(P::from("itself"));
                }
            }
            loc_phrase = Some(p);
        } else {
            let mut prep = q.c(&["in", "in", "in", "under", "inside", "within", "from", "in", "inside of", "located in", "that are in", "sitting in"]);
            if (sink == "move" || sink == "copy") && q.f() < 0.5 {
                prep = q.c(&["from", "in", "out of"]);
            }
            loc_phrase = Some(ph![prep, lpre, (text, k), lpost]);
        }
    } else if q.f() < 0.25 {
        let w = q.c(&["here", "in this folder", "in the current directory", "in this directory", "in here", "in the current folder"]);
        loc_phrase = Some(ph![w]);
        if q.f() < 0.12 {
            flat = true;
            let w = q.c(&["in this folder only", "here but not in subfolders", "in this directory, not recursively", "just here, no subfolders",
                "only at the top level here"]);
            loc_phrase = Some(ph![w]);
        }
    }
    // noun
    let ext_noun = fl.noun.clone();
    let mut noun_word: Option<&'static str> = None;
    let noun: Vec<P> = if let Some(n) = &ext_noun {
        vec![P::from(n)]
    } else {
        let mut w = if dirs {
            q.c(&["folders", "directories", "subfolders", "subdirectories", "dirs", "folders"])
        } else {
            q.c(&["files", "files", "files", "files", "files", "files", "files", "items", "stuff", "things", "anything", "everything", "content", "data"])
        };
        if MASS.contains(&w) && (sup_phrase.is_some() || singular_n || !fl.pre.is_empty()) {
            w = "files";
        }
        if !dirs && sup_phrase.is_some() && q.f() < 0.12 {
            w = q.c(&["things", "ones", "items"]);
        }
        if singular_n {
            w = singular(w);
        }
        noun_word = Some(w);
        vec![P::from(w)]
    };
    if singular_n && ext_noun.is_some() {
        singular_n = false;
        if take == Some(None) {
            take = None;
        }
    }
    let _ = singular_n;
    let mut quant: &'static str = if sup_phrase.is_some() {
        ""
    } else {
        q.c(&["all", "all the", "every", "any", "", "", "", "the", "all of the", "each", "all", "all my", "my", "those", "the", "any of the"])
    };
    let mass_noun = noun_word.map(|w| MASS.contains(&w)).unwrap_or(false);
    if mass_noun {
        quant = if ["stuff", "things", "content", "data"].contains(&noun_word.unwrap()) { q.c(&["", "", "all the", "the", "all that"]) } else { "" };
    }
    let adjective: Option<&str> = if sup_phrase.is_none() && q.f() < 0.06 && !mass_noun { Some(q.c(NEUTRAL)) } else { None };
    let mut noun = noun;
    if (quant == "every" || quant == "each") && ext_noun.is_none() && !singular_n {
        let w = noun_word.unwrap();
        let s = match w {
            "files" => "file",
            "folders" => "folder",
            "directories" => "directory",
            "subfolders" => "subfolder",
            "subdirectories" => "subdirectory",
            "dirs" => "dir",
            "items" => "item",
            other => other,
        };
        noun = vec![P::from(s)];
        if q.f() < 0.3 {
            quant = q.c(&["every single", "each and every", "every last"]);
        }
    }
    if (quant == "every" || quant == "each") && ext_noun.is_some() {
        quant = "all";
    }
    let mut np: Phrase = vec![P::from(quant)];
    if let Some(s) = &sup_phrase {
        np.extend(s.iter().cloned());
    }
    if let Some(a) = adjective {
        np.push(P::from(a));
    }
    for p in &fl.pre {
        np.extend(p.iter().cloned());
    }
    np.extend(noun);
    // sink-specific pieces
    let (mut dest_key, mut dest_phrase, mut fmt, mut arch_key): (Option<u32>, Option<Phrase>, &str, Option<u32>) = (None, None, "", None);
    if sink == "move" || sink == "copy" {
        let (text, dpre, dpost) = q.folder();
        let k = q.key();
        dest_key = Some(k);
        let prep = if sink == "move" { q.c(&["to", "into", "to", "over to", "in"]) } else { q.c(&["to", "into", "to", "in", "over to"]) };
        dest_phrase = Some(ph![prep, dpre, (text, k), dpost]);
    }
    if sink == "compress" {
        fmt = q.c(&["tgz", "tgz", "zip"]);
        if q.f() < 0.6 {
            let w = q.word();
            let base = format!("{w}{}", q.c(&["", "", "-backup", "_archive", "2024"]));
            let name = if q.f() < 0.75 {
                let e = if fmt == "zip" { q.c(&[".zip"]) } else { q.c(&[".tar.gz", ".tgz", ".tar.gz", ""]) };
                format!("{base}{e}")
            } else {
                base
            };
            if name.ends_with(".zip") {
                fmt = "zip";
            }
            let k = q.key();
            arch_key = Some(k);
            let lead = q.c(&["into", "as", "to", "called", "named", "into a file called", "and call it", "into an archive called"]);
            dest_phrase = Some(ph![lead, (name, k)]);
        }
    }
    let mut tailbits: Vec<Phrase> = fl.post.clone();
    let mut has_loc = false;
    if let Some(lp) = &loc_phrase {
        let ri = q.randint(0, tailbits.len() as i64) as usize;
        let pos = [tailbits.len(), 0, ri][q.weighted(&[6.0, 2.0, 2.0])];
        tailbits.insert(pos, lp.clone());
        has_loc = true;
    }
    let mut lead_loc: Option<Phrase> = None;
    if has_loc && q.f() < 0.06 {
        let lp = loc_phrase.clone().unwrap();
        if let Some(i) = tailbits.iter().position(|t| phrase_eq(t, &lp)) {
            tailbits.remove(i);
        }
        lead_loc = Some(lp);
    }

    // ---- assemble English
    if q.f() < 0.28 {
        let p = q.c(POLITE);
        q.lit(p);
    }
    if let Some(ll) = &lead_loc {
        q.emit(ll);
        q.lit(",");
    }
    let _style = q.f();
    let mut frame_args: Vec<(&'static str, u32)> = Vec::new();
    match sink {
        "fcount" => match q.below(7) {
            0 => {
                q.lit("how many");
                q.emit(&np[1..]);
                q.tail(&tailbits);
                let w = q.c(&["are there", "", "exist", "do i have", "are there"]);
                q.lit(w);
            }
            1 => {
                q.lit("how many");
                q.emit(&np[1..]);
                let w = q.c(&["are", "are there", "do i have", "exist", "live"]);
                q.lit(w);
                q.tail(&tailbits);
            }
            2 => {
                let w = q.c(&["count", "count up", "count all", "tally", "count how many"]);
                q.lit(w);
                q.emit(&np);
                q.tail(&tailbits);
            }
            3 => {
                let w = q.c(&["tell me how many", "i want to know how many", "find out how many", "show how many"]);
                q.lit(w);
                q.emit(&np[1..]);
                q.tail(&tailbits);
                let w = q.c(&["there are", "", "exist"]);
                q.lit(w);
            }
            4 => {
                let w = q.c(&["number of", "the number of", "give me the number of", "what's the number of", "total number of", "count of"]);
                q.lit(w);
                q.emit(&np[1..]);
                q.tail(&tailbits);
            }
            5 => {
                let w = q.c(&["what is the count of", "get the count of", "how many"]);
                q.lit(w);
                q.emit(&np[1..]);
                q.tail(&tailbits);
            }
            _ => {
                q.emit(&np[1..]);
                q.tail(&tailbits);
                let w = q.c(&["- how many", "how many are there", "count", "how many"]);
                q.lit(w);
            }
        },
        "total" => match q.below(6) {
            0 => {
                let w = q.c(&["how much space do", "how much disk space do", "how much room do", "how much storage do"]);
                q.lit(w);
                q.emit(&np);
                q.tail(&tailbits);
                let w = q.c(&["take", "take up", "use", "occupy", "use up", "eat"]);
                q.lit(w);
            }
            1 => {
                let w = q.c(&["total size of", "what's the total size of", "the combined size of", "what is the combined size of",
                    "add up the size of", "sum the sizes of", "get the total size of", "overall size of"]);
                q.lit(w);
                q.emit(&np);
                q.tail(&tailbits);
            }
            2 => {
                let w = q.c(&["how big are", "how large are", "how heavy are"]);
                q.lit(w);
                q.emit(&np);
                q.tail(&tailbits);
                let w = q.c(&["in total", "altogether", "combined", "all together", "in total"]);
                q.lit(w);
            }
            3 => {
                let w = q.c(&["how much disk space is used by", "how much space is taken by", "how much space is wasted on", "space used by"]);
                q.lit(w);
                q.emit(&np);
                q.tail(&tailbits);
            }
            4 => {
                let w = q.c(&["what do", "how much do"]);
                q.lit(w);
                q.emit(&np);
                q.tail(&tailbits);
                let w = q.c(&["add up to", "weigh in total", "total"]);
                q.lit(w);
            }
            _ => {
                let w = q.c(&["show", "tell me", "give me", "calculate", "work out"]);
                q.lit(w);
                let w = q.c(&["the total size of", "how much space is used by", "the combined size of"]);
                q.lit(w);
                q.emit(&np);
                q.tail(&tailbits);
            }
        },
        "sizes" => {
            let form = q.below(6);
            if dirs && form < 2 && fl.stages.is_empty() && sup_phrase.is_none() {
                let w = q.c(&["what's taking up space", "what is using the most space", "which folders are taking up space",
                    "what's eating my disk space", "where is my disk space going", "what is hogging disk space",
                    "which folders are the biggest", "show what's using space", "disk usage by folder"]);
                q.lit(w);
                q.tail(&tailbits);
            } else if form < 4 {
                let w = q.c(&["show the size of", "how big is", "what's the size of", "show sizes of", "list the sizes of",
                    "give me the size of", "how large is", "print the size of", "show how big", "sizes of"]);
                q.lit(w);
                if quant.is_empty() {
                    let mut e = vec![P::from("each")];
                    e.extend(np[1..].iter().cloned());
                    q.emit(&e);
                } else {
                    q.emit(&np);
                }
                q.tail(&tailbits);
            } else {
                let w = q.c(&SHOW[..12]);
                q.lit(w);
                q.emit(&np);
                q.tail(&tailbits);
                let w = q.c(&["with their sizes", "and their sizes", "along with how big they are", "with sizes", "and how much space each takes", "with size info"]);
                q.lit(w);
            }
        }
        "fshow" => {
            let form = q.f();
            if form < 0.62 {
                let w = q.c(SHOW);
                q.lit(w);
                q.emit(&np);
                q.tail(&tailbits);
            } else if form < 0.78 {
                let w = q.c(&["what", "which"]);
                q.lit(w);
                q.emit(&np[1..]);
                let w = q.c(&["are", "are there", "do i have", "exist", "are sitting", ""]);
                q.lit(w);
                q.tail(&tailbits);
            } else if form < 0.9 {
                q.emit(&np);
                q.tail(&tailbits);
            } else {
                let w = q.c(&["are there any", "do i have any", "is there any", "any", "got any"]);
                q.lit(w);
                q.emit(&np[1..]);
                q.tail(&tailbits);
            }
        }
        _ if is_frame => {
            frame_args = frame_emit(q, sink, &np, &tailbits);
        }
        _ => {
            let verbs: &[&'static str] = match sink {
                "delete" => DELETE,
                "move" => MOVE,
                "copy" => COPY,
                _ => {
                    if fmt == "zip" {
                        ZIP
                    } else {
                        TGZ
                    }
                }
            };
            let verb = q.c(verbs);
            let late = (sink == "move" || sink == "copy" || sink == "compress") && dest_phrase.is_some() && q.f() < 0.12;
            q.lit(verb);
            q.emit(&np);
            if dest_phrase.is_some() && !late && q.f() < 0.25 {
                let d = dest_phrase.take().unwrap();
                q.emit(&d);
            }
            q.tail(&tailbits);
            if let Some(d) = dest_phrase.take() {
                q.emit(&d);
            }
        }
    }
    if let Some(sc) = sort_clause {
        let w = if q.f() < 0.5 { q.c(&[",", "", "and show them", "-"]) } else { "" };
        q.lit(w);
        q.lit(sc);
    }
    if q.f() < 0.1 {
        let w = q.c(TAILS);
        q.lit(w);
    }

    // ---- program in canonical order
    let mut pipe: Pipe = vec![("files", vec![("what", Some(V::E(if dirs { "dirs" } else { "anyfile" }))), ("in", loc_key.map(V::K))])];
    if flat {
        fl.stages.push(st!("flat"));
    }
    pipe.extend(filter_stages(&fl.stages, &fl.negated));
    if let Some((k, o)) = sort {
        pipe.push(("fsort", vec![("key", Some(V::E(k))), ("order", Some(V::E(o)))]));
    }
    if let Some(p) = pick {
        pipe.push(st!("fpick", "n" => Some(V::K(p))));
    } else if let Some(t) = take {
        pipe.push(st!("ftake", "n" => t.map(V::K)));
    }
    if is_frame {
        let a: Vec<(&'static str, Option<V>)> = frame_args.iter().map(|(s, k)| (*s, Some(V::K(*k)))).collect();
        pipe.extend(frame_stages(sink, &a));
        return pipe;
    }
    let args: Args = match sink {
        "move" | "copy" => vec![("to", dest_key.map(V::K))],
        "compress" => vec![("fmt", Some(V::E(fmt))), ("to", arch_key.map(V::K))],
        _ => Vec::new(),
    };
    pipe.push((sink, args));
    pipe
}

fn sort_clause_of(key: (&str, &str)) -> &'static [&'static str] {
    sort_clause(key)
}

fn phrase_eq(a: &[P], b: &[P]) -> bool {
    a.len() == b.len()
        && a.iter().zip(b).all(|(x, y)| match (x, y) {
            (P::L(s), P::L(t)) => s == t,
            (P::A(s, k, q), P::A(t, l, r)) => s == t && k == l && q == r,
            _ => false,
        })
}

// --------------------------------------------------------------------------- line requests
pub fn lines_request(q: &mut Req) -> Pipe {
    let f = q.filename(Some(&["log", "txt", "csv", "md", "json", "conf", "py", "sh", "out", "err", "tsv", "yaml"]));
    let fk = q.key();
    let fa = (f, fk);
    let ofs = ["in", "of", "from", "inside"];
    let kind = ["grep", "head", "tail", "count", "sort", "uniq", "combo", "nomatch"][q.weighted(&[30.0, 12.0, 14.0, 12.0, 8.0, 8.0, 10.0, 6.0])];
    let mut pipe: Pipe = vec![st!("lines", "of" => Some(V::K(fk)))];
    if q.f() < 0.25 {
        let w = q.c(POLITE);
        q.lit(w);
    }
    match kind {
        "grep" | "nomatch" => {
            let t = if q.f() < 0.6 { q.c(PHRASES) } else { q.word() };
            let k = q.key();
            let quote = t.contains(' ') || q.f() < 0.2;
            let ta = (t.to_string(), k, quote);
            if kind == "grep" {
                let p = match q.below(9) {
                    0 => {
                        let a = q.c(&["show", "find", "get", "print", "list", "show me", "display"]);
                        let b = q.c(&["lines", "all lines", "the lines", "every line"]);
                        let c = q.c(&["containing", "with", "that contain", "that mention", "matching", "that include", "that have"]);
                        let o = q.c(&ofs);
                        ph![a, b, c, &ta, o, &fa]
                    }
                    1 => {
                        let a = q.c(&["for", ""]);
                        ph!["grep", a, &ta, "in", &fa]
                    }
                    2 => ph!["search", &fa, "for", &ta],
                    3 => {
                        let a = q.c(&["in", "inside", "within"]);
                        ph!["search for", &ta, a, &fa]
                    }
                    4 => ph!["find", &ta, "in", &fa],
                    5 => {
                        let a = q.c(&["which lines", "what lines"]);
                        let b = q.c(&["in", "of"]);
                        let c = q.c(&["contain", "mention", "have", "include", "match"]);
                        ph![a, b, &fa, c, &ta]
                    }
                    6 => ph!["look for", &ta, "in", &fa],
                    7 => {
                        let a = q.c(&["where does", "where is"]);
                        let b = q.c(&["appear in", "show up in", "occur in", "mentioned in"]);
                        ph![a, &ta, b, &fa]
                    }
                    _ => {
                        let a = q.c(&["lines", "all lines"]);
                        let b = q.c(&["in", "from"]);
                        let c = q.c(&["with", "containing", "that have"]);
                        ph![a, b, &fa, c, &ta]
                    }
                };
                q.emit(&p);
                pipe.push(st!("match", "text" => Some(V::K(k))));
            } else {
                let p = match q.below(5) {
                    0 => {
                        let a = q.c(&["show", "print", "get", "list"]);
                        let b = q.c(&["lines", "the lines", "all lines"]);
                        let o = q.c(&ofs);
                        let c = q.c(&["that don't contain", "without", "that do not mention", "not containing", "that don't have", "excluding", "that lack"]);
                        ph![a, b, o, &fa, c, &ta]
                    }
                    1 => {
                        let a = q.c(&["show", "print"]);
                        let b = q.c(&["without the lines containing", "minus lines with", "but skip lines with", "except lines containing", "hiding lines with"]);
                        ph![a, &fa, b, &ta]
                    }
                    2 => {
                        let a = q.c(&["filter out", "remove", "drop", "hide", "exclude", "strip"]);
                        let b = q.c(&["lines with", "lines containing", "lines that mention", "every line with"]);
                        ph![a, b, &ta, "from", &fa]
                    }
                    3 => {
                        let a = q.c(&["except lines with", "that doesn't mention", "but not lines containing"]);
                        ph!["everything in", &fa, a, &ta]
                    }
                    _ => ph!["grep -v", &ta, &fa],
                };
                q.emit(&p);
                pipe.push(st!("nomatch", "text" => Some(V::K(k))));
            }
            if q.f() < 0.12 {
                let w = q.c(&["and count them", "and tell me how many", "- how many are there", "and count"]);
                q.lit(w);
                pipe.push(st!("lcount"));
            } else if q.f() < 0.1 {
                let out = q.filename(Some(&["txt", "log", "out"]));
                let ok = q.key();
                let w = q.c(&["and save them to", "and write them to", "into", "and put them in", "and save to", "> "]);
                q.emit(&ph![w, (out, ok)]);
                pipe.push(st!("save", "to" => Some(V::K(ok))));
            } else {
                pipe.push(st!("lshow"));
            }
        }
        "head" | "tail" => {
            let n = q.count();
            let k = q.key();
            let has_n = q.f() < 0.8;
            let na = (n, k);
            let head = kind == "head";
            let nforms = if head { 7 } else { 8 };
            let form = q.below(nforms);
            let p = if head {
                match form {
                    0 => {
                        let a = q.c(&["show", "print", "get", "display", "give me", "show me"]);
                        let o = q.c(&ofs);
                        ph![a, "the first", &na, "lines", o, &fa]
                    }
                    1 => {
                        let o = q.c(&ofs);
                        ph!["first", &na, "lines", o, &fa]
                    }
                    2 => {
                        if has_n {
                            ph!["head", &fa, &na]
                        } else {
                            ph!["head", &fa]
                        }
                    }
                    3 => {
                        let a = q.c(&["top", "the top"]);
                        let o = q.c(&ofs);
                        ph![a, &na, "lines", o, &fa]
                    }
                    4 => {
                        let a = q.c(&["the beginning of", "what's at the top of", "the top of", "the start of", "the opening lines of",
                            "peek at the top of", "what does the start of", "the first few lines of", "the first lines of", "how does"]);
                        ph![a, &fa]
                    }
                    5 => {
                        let a = q.c(&["show", "print"]);
                        let b = q.c(&["start", "beginning", "top", "head"]);
                        ph![a, "the", b, "of", &fa]
                    }
                    _ => ph!["what are the first", &na, "lines", "of", &fa],
                }
            } else {
                match form {
                    0 => {
                        let a = q.c(&["show", "print", "get", "display", "give me", "show me"]);
                        let o = q.c(&ofs);
                        ph![a, "the last", &na, "lines", o, &fa]
                    }
                    1 => {
                        let o = q.c(&ofs);
                        ph!["last", &na, "lines", o, &fa]
                    }
                    2 => {
                        if has_n {
                            ph!["tail", &fa, &na]
                        } else {
                            ph!["tail", &fa]
                        }
                    }
                    3 => {
                        let a = q.c(&["bottom", "the bottom", "the final", "final"]);
                        let o = q.c(&ofs);
                        ph![a, &na, "lines", o, &fa]
                    }
                    4 => {
                        let a = q.c(&["the end of", "what's at the end of", "the bottom of", "the tail of", "tail of", "the latest lines in",
                            "what was logged last in", "the last few lines of", "the last lines of", "the most recent lines in"]);
                        ph![a, &fa]
                    }
                    5 => {
                        let a = q.c(&["show", "print"]);
                        let b = q.c(&["end", "tail", "bottom", "last part"]);
                        ph![a, "the", b, "of", &fa]
                    }
                    6 => ph!["what are the last", &na, "lines", "of", &fa],
                    _ => {
                        let a = q.c(&["in", "from"]);
                        ph!["latest", &na, "entries", a, &fa]
                    }
                }
            };
            let used_n = if head { [0, 1, 3, 6].contains(&form) || (form == 2 && has_n) } else { [0, 1, 3, 6, 7].contains(&form) || (form == 2 && has_n) };
            q.emit(&p);
            let nv = if used_n { Some(V::K(k)) } else { None };
            pipe.push((if head { "ltake" } else { "llast" }, vec![("n", nv)]));
            pipe.push(st!("lshow"));
        }
        "count" => {
            let p = match q.below(7) {
                0 => {
                    let a = q.c(&["are in", "in", "does", "are there in"]);
                    ph!["how many lines", a, &fa]
                }
                1 => {
                    let a = q.c(&["the lines", "lines", "the number of lines"]);
                    let o = q.c(&ofs);
                    ph!["count", a, o, &fa]
                }
                2 => {
                    let a = q.c(&["in lines", "", "(lines)"]);
                    ph!["how long is", &fa, a]
                }
                3 => {
                    let a = q.c(&["of", "for"]);
                    ph!["line count", a, &fa]
                }
                4 => ph!["number of lines in", &fa],
                5 => ph!["wc -l", &fa],
                _ => {
                    let a = q.c(&["tell me", "show"]);
                    ph![a, "how many lines", &fa, "has"]
                }
            };
            q.emit(&p);
            pipe.push(st!("lcount"));
        }
        "sort" => {
            let key = q.pick(&[("alpha", "asc"), ("alpha", "asc"), ("alpha", "desc"), ("num", "asc"), ("num", "desc")]);
            let how: &[&'static str] = match key {
                ("alpha", "asc") => &["", "alphabetically", "in alphabetical order", "a to z", ""],
                ("alpha", "desc") => &["in reverse", "in reverse order", "backwards", "z to a", "in reverse alphabetical order", "descending"],
                ("num", "asc") => &["numerically", "by number", "as numbers", "in numeric order"],
                _ => &["numerically in reverse", "by number descending", "from highest to lowest number", "numerically, biggest first"],
            };
            let p = match q.below(4) {
                0 => {
                    let a = q.c(&["the lines of", "the lines in", "", "the contents of"]);
                    let h = q.c(how);
                    ph!["sort", a, &fa, h]
                }
                1 => {
                    let h = q.c(how);
                    ph!["sort", &fa, h]
                }
                2 => {
                    let a = q.c(&["show", "print", "give me"]);
                    let h = q.c(how);
                    ph![a, &fa, "sorted", h]
                }
                _ => {
                    let a = q.c(&["order", "arrange"]);
                    let o = q.c(&ofs);
                    let h = q.c(how);
                    ph![a, "the lines", o, &fa, h]
                }
            };
            q.emit(&p);
            pipe.push(("lsort", vec![("key", Some(V::E(key.0))), ("order", Some(V::E(key.1)))]));
            if q.f() < 0.15 {
                let out = q.filename(Some(&["txt", "csv", "out"]));
                let ok = q.key();
                let w = q.c(&["and save to", "and write the result to", "into", "and save it as"]);
                q.emit(&ph![w, (out, ok)]);
                pipe.push(st!("save", "to" => Some(V::K(ok))));
            } else {
                pipe.push(st!("lshow"));
            }
        }
        "uniq" => {
            let p = match q.below(6) {
                0 => {
                    let a = q.c(&["remove", "drop", "delete", "strip", "get rid of"]);
                    let b = q.c(&["duplicate lines", "duplicates", "repeated lines", "dupes"]);
                    let c = q.c(&["from", "in"]);
                    ph![a, b, c, &fa]
                }
                1 => {
                    let a = q.c(&["show", "list", "print", "get"]);
                    let b = q.c(&["unique lines", "the unique lines", "distinct lines", "each line once"]);
                    let o = q.c(&ofs);
                    ph![a, b, o, &fa]
                }
                2 => ph!["dedupe", &fa],
                3 => {
                    let a = q.c(&["", "the lines in", "lines of"]);
                    ph!["deduplicate", a, &fa]
                }
                4 => {
                    let a = q.c(&["without duplicates", "with duplicates removed", "but no repeated lines", "deduplicated"]);
                    ph![&fa, a]
                }
                _ => {
                    let a = q.c(&["lines", "entries", "values"]);
                    let b = q.c(&["in", "from"]);
                    ph!["unique", a, b, &fa]
                }
            };
            q.emit(&p);
            pipe.push(st!("uniq"));
            if q.f() < 0.2 {
                let out = q.filename(Some(&["txt", "csv", "out"]));
                let ok = q.key();
                let w = q.c(&["and save to", "and write them to", "into", "and store in"]);
                q.emit(&ph![w, (out, ok)]);
                pipe.push(st!("save", "to" => Some(V::K(ok))));
            } else if q.f() < 0.1 {
                let w = q.c(&["and count them", "- how many are there", "and tell me how many"]);
                q.lit(w);
                pipe.push(st!("lcount"));
            } else {
                pipe.push(st!("lshow"));
            }
        }
        _ => {
            // combo: match -> sort -> uniq -> take -> save
            let t = if q.f() < 0.6 { q.c(PHRASES) } else { q.word() };
            let k = q.key();
            let quote = t.contains(' ') || q.f() < 0.2;
            let ta = (t.to_string(), k, quote);
            let a = q.c(&["get", "find", "take", "grab", "pull", "extract"]);
            let b = q.c(&["lines", "the lines", "all lines", "every line"]);
            let c = q.c(&["with", "containing", "that mention", "matching"]);
            let d = q.c(&["from", "in"]);
            q.emit(&ph![a, b, c, &ta, d, &fa]);
            pipe.push(st!("match", "text" => Some(V::K(k))));
            let mut steps: Vec<(Phrase, Stage)> = Vec::new();
            if q.f() < 0.7 {
                let key = q.pick(&[("alpha", "asc"), ("alpha", "asc"), ("alpha", "desc"), ("num", "desc")]);
                let w = match key {
                    ("alpha", "asc") => q.c(&["sort them", "sort", "put them in order", "sort alphabetically"]),
                    ("alpha", "desc") => q.c(&["sort them in reverse", "reverse sort them", "sort descending"]),
                    _ => q.c(&["sort numerically descending", "sort by number biggest first"]),
                };
                steps.push((ph![w], ("lsort", vec![("key", Some(V::E(key.0))), ("order", Some(V::E(key.1)))])));
            }
            if q.f() < 0.6 {
                let w = q.c(&["remove duplicates", "drop duplicates", "dedupe", "keep unique ones", "remove repeats"]);
                steps.push((ph![w], st!("uniq")));
            }
            if q.f() < 0.35 {
                let n = q.count();
                let nk = q.key();
                let w = q.c(&["keep the first", "take the first", "keep only the top", "show the first"]);
                steps.push((ph![w, (n, nk)], st!("ltake", "n" => Some(V::K(nk)))));
            }
            for (phrase, stage) in steps {
                let j = q.c(&[",", "and", "then", ", then", "and then"]);
                q.lit(j);
                q.emit(&phrase);
                pipe.push(stage);
            }
            if q.f() < 0.5 {
                let out = q.filename(Some(&["txt", "log", "out", "csv"]));
                let ok = q.key();
                let w = q.c(&["and save to", "and write to", "then save as", "and put the result in", "and save the result to"]);
                q.emit(&ph![w, (out, ok)]);
                pipe.push(st!("save", "to" => Some(V::K(ok))));
            } else if q.f() < 0.2 {
                let w = q.c(&["and count them", "and count what's left", "then count"]);
                q.lit(w);
                pipe.push(st!("lcount"));
            } else {
                pipe.push(st!("lshow"));
            }
        }
    }
    if q.f() < 0.08 {
        let w = q.c(TAILS);
        q.lit(w);
    }
    pipe
}

// --------------------------------------------------------------------------- process requests
pub fn procs_request(q: &mut Req) -> Pipe {
    let mut pipe: Pipe = vec![st!("procs")];
    let kind = ["top", "kill_name", "kill_top", "show_name", "count_name", "list", "count"][q.weighted(&[26.0, 24.0, 8.0, 16.0, 8.0, 12.0, 6.0])];
    if q.f() < 0.22 {
        let w = q.c(POLITE);
        q.lit(w);
    }
    let procw = q.c(&["processes", "programs", "apps", "tasks", "processes", "things", "procs"]);
    match kind {
        "top" => {
            let key = q.c(&["mem", "mem", "cpu", "cpu"]);
            let res: [&str; 5] = if key == "mem" { ["memory", "ram", "mem", "the most memory", "the most ram"] } else { ["cpu", "the cpu", "processor", "the most cpu", "cpu time"] };
            pipe.push(("psort", vec![("key", Some(V::E(key))), ("order", Some(V::E("desc")))]));
            let form = q.below(9);
            if form < 4 {
                let n = q.count();
                let k = q.key();
                let p = match form {
                    0 => {
                        let a = q.c(&["show", "list", "what are", "give me", "display"]);
                        let r = q.c(&res[..3]);
                        ph![a, "the top", (n, k), procw, "by", r]
                    }
                    1 => {
                        let r = q.c(&res[..3]);
                        let b = q.c(&["hogs", "users", "consumers", "eaters"]);
                        ph!["top", (n, k), r, b]
                    }
                    2 => {
                        let a = q.c(&["which", "what"]);
                        let b = q.c(&["use", "are using", "eat", "take"]);
                        let c = if key == "mem" {
                            let o = [format!("the most {}", res[0]), format!("the most {}", res[1])];
                            q.pick(&o)
                        } else {
                            q.c(&["the most cpu", "the most processor"]).to_string()
                        };
                        ph![a, (n, k), procw, b, c]
                    }
                    _ => {
                        let a = q.c(&["show", "list"]);
                        let b = q.c(&["using the most", "with the highest", "eating the most"]);
                        let r = q.c(&res[..3]);
                        ph![a, "the", (n, k), procw, b, r]
                    }
                };
                q.emit(&p);
                pipe.push(st!("ptake", "n" => Some(V::K(k))));
            } else if form < 7 {
                let p = match form - 4 {
                    0 => {
                        let a = q.c(&["what's using", "what is using", "which process is using", "what's eating", "what is hogging", "who is using"]);
                        let o = [format!("the most {}", res[0]), format!("all my {}", res[0]), format!("all the {}", res[0]), format!("so much {}", res[0])];
                        let b = q.pick(&o);
                        ph![a, b]
                    }
                    1 => {
                        let a = q.c(&["which process is", "what's", "what is"]);
                        let b = q.c(&["hogging", "eating", "maxing out", "burning"]);
                        let o = [format!("the {}", res[0]), format!("my {}", res[0])];
                        let c = q.pick(&o);
                        ph![a, b, c]
                    }
                    _ => {
                        let a = q.c(&["biggest", "top", "worst"]);
                        let b = q.c(&["hog", "user", "consumer"]);
                        ph![a, res[0], b]
                    }
                };
                q.emit(&p);
                pipe.push(st!("ptake", "n" => None));
            } else {
                let p = if form == 7 {
                    let a = q.c(&["show", "list"]);
                    let b = q.c(&["sorted by", "ordered by", "by"]);
                    let r = q.c(&res[..3]);
                    let c = q.c(&["usage", "use", ""]);
                    ph![a, procw, b, r, c]
                } else {
                    let b = q.c(&["by", "sorted by"]);
                    let r = q.c(&res[..3]);
                    let c = q.c(&["usage", ""]);
                    ph![procw, b, r, c]
                };
                q.emit(&p);
            }
            pipe.push(st!("pshow"));
        }
        "kill_name" | "show_name" | "count_name" => {
            let name = if q.f() < 0.7 { q.c(&PROCS) } else { q.word() };
            let k = q.key();
            let na = (name.to_string(), k);
            pipe.push(st!("pname", "name" => Some(V::K(k))));
            if kind == "kill_name" {
                let force = q.f() < 0.3;
                let verb = if force {
                    q.c(&["force kill", "force quit", "kill -9", "forcibly kill", "hard kill", "force close", "nuke", "murder", "force stop"])
                } else {
                    q.c(&["kill", "stop", "terminate", "close", "end", "shut down", "quit", "kill all", "stop all", "exit", "pkill"])
                };
                let p = match q.below(5) {
                    0 => ph![verb, &na],
                    1 => {
                        let a = q.c(&["processes", "process", "", "instances"]);
                        ph![verb, &na, a]
                    }
                    2 => {
                        let a = q.c(&["process", "app", "program"]);
                        ph![verb, "the", &na, a]
                    }
                    3 => {
                        let a = q.c(&["all", "every", "any"]);
                        let b = q.c(&["processes", "processes", "instances"]);
                        ph![verb, a, &na, b]
                    }
                    _ => {
                        let a = q.c(&["anything named", "everything called", "processes named", "the process called"]);
                        ph![verb, a, &na]
                    }
                };
                q.emit(&p);
                pipe.push(st!("kill", "force" => Some(V::E(if force { "hard" } else { "soft" }))));
            } else if kind == "show_name" {
                let p = match q.below(7) {
                    0 => {
                        let a = q.c(&["running", "running?", "still running", "up", "alive", "active"]);
                        ph!["is", &na, a]
                    }
                    1 => {
                        let a = q.c(&["show", "list", "find"]);
                        let b = q.c(&["processes", "process", "instances"]);
                        ph![a, &na, b]
                    }
                    2 => {
                        let a = q.c(&["show", "list", "find"]);
                        let b = q.c(&["processes named", "processes called", "all processes matching", "any process named"]);
                        ph![a, b, &na]
                    }
                    3 => {
                        let a = q.c(&["is running", "is up", "is alive", "is still going"]);
                        ph!["check if", &na, a]
                    }
                    4 => {
                        let a = q.c(&["what", "which"]);
                        let b = q.c(&["running", "active", "alive"]);
                        ph![a, &na, "processes are", b]
                    }
                    5 => ph!["pgrep", &na],
                    _ => {
                        let a = q.c(&["a process called", "processes named", "the process"]);
                        ph!["look for", a, &na]
                    }
                };
                q.emit(&p);
                pipe.push(st!("pshow"));
            } else {
                let p = match q.below(4) {
                    0 => {
                        let a = q.c(&["processes", "instances"]);
                        let b = q.c(&["are running", "are there", "do i have", ""]);
                        ph!["how many", &na, a, b]
                    }
                    1 => {
                        let a = q.c(&["the", ""]);
                        let b = q.c(&["processes", "instances"]);
                        ph!["count", a, &na, b]
                    }
                    2 => {
                        let a = q.c(&["running", ""]);
                        let b = q.c(&["processes", "instances"]);
                        ph!["number of", a, &na, b]
                    }
                    _ => {
                        let a = q.c(&["are running", "are open", "are there"]);
                        ph!["how many copies of", &na, a]
                    }
                };
                q.emit(&p);
                pipe.push(st!("pcount"));
            }
        }
        "kill_top" => {
            let key = q.c(&["mem", "cpu"]);
            let res: [&str; 2] = if key == "mem" { ["memory", "ram"] } else { ["cpu", "processor"] };
            pipe.push(("psort", vec![("key", Some(V::E(key))), ("order", Some(V::E("desc")))]));
            let force = q.f() < 0.25;
            let verb = if force { q.c(&["force kill", "kill -9", "forcibly kill"]) } else { q.c(&["kill", "stop", "terminate", "end"]) };
            if q.f() < 0.5 {
                let n = q.count();
                let k = q.key();
                let a = q.c(&["using the most", "eating the most", "with the highest", "hogging the most"]);
                let r = q.c(&res);
                q.emit(&ph![verb, "the", (n, k), procw, a, r]);
                pipe.push(st!("ptake", "n" => Some(V::K(k))));
            } else {
                let p = if q.below(2) == 0 {
                    let a = q.c(&["the process", "whatever is", "the thing", "the app"]);
                    let b = q.c(&["using the most", "eating the most", "hogging the most", "that uses the most"]);
                    let r = q.c(&res);
                    ph![verb, a, b, r]
                } else {
                    let a = q.c(&["biggest", "top", "worst"]);
                    let r = q.c(&res);
                    let b = q.c(&["hog", "user"]);
                    ph![verb, "the", a, r, b]
                };
                q.emit(&p);
                pipe.push(st!("ptake", "n" => None));
            }
            pipe.push(st!("kill", "force" => Some(V::E(if force { "hard" } else { "soft" }))));
        }
        "list" => {
            let w = q.c(&["show running processes", "list all processes", "what's running", "show me everything that's running",
                "list processes", "what processes are running", "ps", "show all running programs", "which programs are running",
                "running processes", "what is currently running", "show the process list", "display running tasks"]);
            q.lit(w);
            pipe.push(st!("pshow"));
        }
        _ => {
            let w = q.c(&["how many processes are running", "count running processes", "number of processes", "how many programs are running",
                "how many things are running right now", "process count", "total number of running processes"]);
            q.lit(w);
            pipe.push(st!("pcount"));
        }
    }
    if q.f() < 0.08 {
        let w = q.c(TAILS);
        q.lit(w);
    }
    pipe
}

// --------------------------------------------------------------------------- file questions
pub fn file_question(q: &mut Req) -> Pipe {
    if q.f() < 0.2 {
        let w = q.c(POLITE);
        q.lit(w);
    }
    let mut files_args: Args = vec![("what", Some(V::E("anyfile"))), ("in", None)];
    let mut ext: Option<u32> = None;
    let mut np: Phrase = {
        let w = q.c(&["anything", "any file", "any files", "any of the files", "something", "a file", "any document"]);
        ph![w]
    };
    if q.f() < 0.4 {
        let e = if q.f() < 0.5 { q.c(EXT_NAMES) } else { q.c(&EXTS) };
        let k = q.key();
        ext = Some(k);
        let a = q.c(&["any", "any of the", "the", "any of my"]);
        np = ph![a, (e, k), "files"];
    }
    let mut locp: Phrase = Vec::new();
    if q.f() < 0.7 {
        let (text, pre, post) = q.folder();
        let k = q.key();
        files_args = vec![("what", Some(V::E("anyfile"))), ("in", Some(V::K(k)))];
        let p = q.c(&["in", "under", "inside", "in", "within"]);
        locp = ph![p, pre, (text, k), post];
    }
    let files: Stage = ("files", files_args);
    let kind = q.f();
    if kind < 0.6 {
        let t = if q.f() < 0.5 { q.c(PHRASES) } else { q.cat(&[&WORDS, &["api_key", "TODO", "secret", "localhost", "username"]]) };
        let k = q.key();
        let ta = (t.to_string(), k, t.contains(' '));
        let verb = q.c(&["mention", "contain", "include", "have", "reference", "use", "talk about", "refer to"]);
        let form = q.below(5);
        let mut p: Phrase = Vec::new();
        match form {
            0 => {
                let w = q.c(&["does", "do", "did"]);
                p.push(P::from(w));
                p.extend(np.iter().cloned());
                p.extend(locp.iter().cloned());
                p.push(P::from(verb));
                p.push(P::from(&ta));
            }
            1 => {
                let w = q.c(&["which files", "what files", "which of the files", "which file", "what file"]);
                p.push(P::from(w));
                p.extend(locp.iter().cloned());
                let s = q.c(&["", "s"]);
                p.push(P::from(format!("{verb}{s}")));
                p.push(P::from(&ta));
            }
            2 => {
                let w = q.c(&["mentioned", "used", "referenced", "present", "anywhere"]);
                p.extend(ph!["is", &ta, w]);
                if locp.is_empty() {
                    let w = q.c(&["in any file", "anywhere here", "in any of the files"]);
                    p.push(P::from(w));
                } else {
                    p.extend(locp.iter().cloned());
                }
            }
            3 => {
                let a = q.c(&["where is", "where do we use", "where does", "who uses", "where did i write"]);
                let b = q.c(&["used", "mentioned", "appear", "come up", "show up", ""]);
                p.extend(ph![a, &ta, b]);
                p.extend(locp.iter().cloned());
            }
            _ => {
                let a = q.c(&["search", "search all files", "look through", "scan", "grep through", "search everything"]);
                p.push(P::from(a));
                if locp.len() > 1 {
                    p.extend(locp[1..].iter().cloned());
                }
                p.push(P::from("for"));
                p.push(P::from(&ta));
            }
        }
        q.emit(&p);
        if form == 3 || form == 4 {
            return vec![files, st!("fgrep", "text" => Some(V::K(k)))];
        }
        let mut pipe: Pipe = vec![files];
        if form == 0 {
            if let Some(e) = ext {
                pipe.push(st!("ext", "ext" => Some(V::K(e))));
            }
        }
        pipe.push(st!("contains", "text" => Some(V::K(k))));
        pipe.push(st!("fshow"));
        return pipe;
    }
    let name = q.filename(None);
    let k = q.key();
    let mut p = match q.below(3) {
        0 => {
            let a = q.c(&["is there", "do i have", "have i got", "is there still"]);
            let b = q.c(&["a file", "a file", "any file", "something"]);
            let c = q.c(&["called", "named"]);
            ph![a, b, c, (name, k)]
        }
        1 => {
            let a = q.c(&["where is", "where's", "where did i put", "find", "locate", "where did"]);
            ph![a, (name, k)]
        }
        _ => {
            let a = q.c(&["does", "is"]);
            let b = q.c(&["exist", "exist anywhere", "still exist", "around", "there"]);
            ph![a, (name, k), b]
        }
    };
    p.extend(locp);
    q.emit(&p);
    vec![files, st!("name", "mode" => Some(V::E("is")), "text" => Some(V::K(k))), st!("fshow")]
}

/// "make a folder called X and move all pdfs into it": the second step points back at X.
pub fn mkdir_then(q: &mut Req) -> Vec<Pipe> {
    let w = q.word();
    let mut name = format!("{w}{}", q.c(&["", "", "-backup", "_old", "2024"]));
    if q.f() < 0.3 {
        name = format!("{}/{name}", q.c(REAL_DIRS));
    }
    let k = q.key();
    if q.f() < 0.2 {
        let p = q.c(POLITE);
        q.lit(p);
    }
    let lead = q.c(&["make a folder called", "create a directory called", "create a folder named", "mkdir", "make a new folder", "create folder",
        "make a directory named", "create a new directory called"]);
    q.emit(&ph![lead, (name.clone(), k)]);
    let j = q.c(&JOIN[..8]);
    q.lit(j);
    let fl = file_filters(q, false);
    let sink = q.c(&["move", "move", "copy"]);
    let verb = q.c(if sink == "move" { MOVE } else { COPY });
    let mut np: Phrase = vec![P::from(q.c(&["all", "all the", "every", "the", "", "all my"]))];
    for p in &fl.pre {
        np.extend(p.iter().cloned());
    }
    match &fl.noun {
        Some(n) => np.push(P::from(n)),
        None => np.push(P::from("files")),
    }
    q.lit(verb);
    q.emit(&np);
    for t in &fl.post {
        q.emit(t);
    }
    let mut loc = None;
    if q.f() < 0.5 {
        loc = Some(q.folder_arg(&["from", "in", "out of", "under"]));
    }
    let dest = if q.f() < 0.7 {
        let w = q.c(&["into it", "there", "to it", "into that folder", "in there", "into the new folder", "to that folder", "in it", "to the new folder"]);
        q.lit(w);
        k
    } else {
        let d = q.key();
        let w = q.c(&["to", "into"]);
        q.emit(&ph![w, (name, d)]);
        d
    };
    let mut pipe: Pipe = vec![("files", vec![("what", Some(V::E("anyfile"))), ("in", loc.map(V::K))])];
    pipe.extend(filter_stages(&fl.stages, &fl.negated));
    pipe.push((sink, vec![("to", Some(V::K(dest)))]));
    vec![vec![st!("mkdir", "path" => Some(V::K(k)))], pipe]
}

pub fn typo(q: &mut Req, word: &str) -> String {
    let cs: Vec<char> = word.chars().collect();
    if cs.len() < 4 || !cs.iter().all(|c| c.is_alphabetic()) {
        return word.to_string();
    }
    let i = 1 + q.below(cs.len() - 2);
    let kind = q.below(3);
    let mut out: Vec<char> = Vec::with_capacity(cs.len() + 1);
    match kind {
        0 => {
            out.extend(&cs[..i]);
            out.push(cs[i + 1]);
            out.push(cs[i]);
            out.extend(&cs[i + 2..]);
        }
        1 => {
            out.extend(&cs[..i]);
            out.extend(&cs[i + 1..]);
        }
        _ => {
            out.extend(&cs[..i]);
            out.push(cs[i]);
            out.extend(&cs[i..]);
        }
    }
    out.into_iter().collect()
}
