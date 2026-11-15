//! kural's program language, read from kural/lang/stages.txt and kural/lang/commands.txt (the same files
//! kural/ir.py and kural/commands.py read). Both files are compiled into the binary.
//!
//! Token numbers are computed exactly as ir.py does: the 5 special tokens, then every stage name in order,
//! then the enum values that are not also stage names, sorted.

use std::collections::HashMap;

pub const STAGES_TXT: &str = include_str!("../../kural/lang/stages.txt");
pub const COMMANDS_TXT: &str = include_str!("../../kural/lang/commands.txt");
pub const SPECIAL: [&str; 5] = ["<pad>", "<bos>", "<end>", ";", "<none>"];

#[derive(Clone, Debug, PartialEq)]
pub enum SlotKind {
    /// copied words; the string says how they are read (path, file, size, dur, text, ...)
    Span(String),
    /// one value of the named enum
    Enum(String),
}

#[derive(Clone, Debug)]
pub struct SlotDef {
    pub name: String,
    pub kind: SlotKind,
    pub optional: bool,
}

#[derive(Clone, Debug)]
pub struct StageDef {
    pub name: String,
    /// data type taken ("" = starts a pipeline) and given: "", files, lines, procs, done, neg, sudo
    pub src: String,
    pub dst: String,
    pub slots: Vec<SlotDef>,
}

/// One stand-alone command of commands.txt.
#[derive(Clone, Debug)]
pub struct CommandDef {
    pub name: String,
    pub slots: Vec<(String, String, bool)>,
    pub risk: u8,
    /// bash template, or None when the command is built by code
    pub bash: Option<String>,
    pub phrases: Vec<String>,
}

pub struct Language {
    pub enums: Vec<(String, Vec<String>)>,
    pub lists: HashMap<String, Vec<String>>,
    pub stages: Vec<StageDef>,
    pub commands: Vec<CommandDef>,
    pub vocab: Vec<String>,
    pub tok: HashMap<String, usize>,
}

pub fn parse_commands(text: &str) -> Vec<CommandDef> {
    let mut out: Vec<CommandDef> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("@ ") {
            let mut parts = rest.split_whitespace();
            let name = parts.next().unwrap_or("").to_string();
            let mut c = CommandDef { name, slots: Vec::new(), risk: 0, bash: None, phrases: Vec::new() };
            for p in parts {
                if let Some(r) = p.strip_prefix('!') {
                    c.risk = r.parse().unwrap_or(0);
                } else if let Some((n, t)) = p.split_once(':') {
                    c.slots.push((n.to_string(), t.trim_end_matches('?').to_string(), t.ends_with('?')));
                }
            }
            out.push(c);
        } else if line.starts_with('$') {
            if let Some(c) = out.last_mut() {
                c.bash = if line == "$$" { None } else { Some(line[1..].trim().to_string()) };
            }
        } else if let Some(c) = out.last_mut() {
            c.phrases.push(line.to_string());
        }
    }
    out
}

impl Language {
    pub fn load() -> Language {
        Language::from_text(STAGES_TXT, COMMANDS_TXT)
    }

    pub fn from_text(stages_txt: &str, commands_txt: &str) -> Language {
        let commands = parse_commands(commands_txt);
        let (mut enums, mut lists, mut stages): (Vec<(String, Vec<String>)>, HashMap<String, Vec<String>>, Vec<StageDef>) = (Vec::new(), HashMap::new(), Vec::new());
        let ty = |s: &str| if s == "-" { String::new() } else { s.to_string() };
        for raw in stages_txt.lines() {
            let f: Vec<&str> = raw.split_whitespace().collect();
            if f.is_empty() || f[0].starts_with('#') {
                continue;
            }
            match f[0] {
                "%" => enums.push((f[1].to_string(), f[2..].iter().map(|s| s.to_string()).collect())),
                "&" => {
                    lists.insert(f[1].to_string(), f[2..].iter().map(|s| s.to_string()).collect());
                }
                "*commands" => {
                    for c in &commands {
                        stages.push(StageDef {
                            name: c.name.clone(),
                            src: String::new(),
                            dst: "done".into(),
                            slots: c
                                .slots
                                .iter()
                                .map(|(n, t, o)| SlotDef { name: n.clone(), kind: SlotKind::Span(t.clone()), optional: *o })
                                .collect(),
                        });
                    }
                }
                "@" => {
                    let slots = f[4..]
                        .iter()
                        .map(|spec| {
                            if let Some((n, k)) = spec.split_once('=') {
                                SlotDef { name: n.into(), kind: SlotKind::Enum(k.into()), optional: false }
                            } else {
                                let (n, t) = spec.split_once(':').unwrap_or((spec, "text"));
                                SlotDef { name: n.into(), kind: SlotKind::Span(t.trim_end_matches('?').into()), optional: t.ends_with('?') }
                            }
                        })
                        .collect();
                    stages.push(StageDef { name: f[1].into(), src: ty(f[2]), dst: ty(f[3]), slots });
                }
                _ => {}
            }
        }
        let mut vocab: Vec<String> = SPECIAL.iter().map(|s| s.to_string()).collect();
        vocab.extend(stages.iter().map(|s| s.name.clone()));
        let names: std::collections::HashSet<&str> = stages.iter().map(|s| s.name.as_str()).collect();
        let mut vals: Vec<String> = enums.iter().flat_map(|(_, v)| v.iter().cloned()).filter(|v| !names.contains(v.as_str())).collect();
        vals.sort();
        vals.dedup();
        vocab.extend(vals);
        let tok = vocab.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
        Language { enums, lists, stages, commands, vocab, tok }
    }

    pub fn stage(&self, name: &str) -> Option<&StageDef> {
        self.stages.iter().find(|s| s.name == name)
    }

    pub fn command(&self, name: &str) -> Option<&CommandDef> {
        self.commands.iter().find(|c| c.name == name)
    }

    pub fn enum_values(&self, kind: &str) -> &[String] {
        self.enums.iter().find(|(k, _)| k == kind).map(|(_, v)| v.as_slice()).unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The token numbers must be the ones the shipped model was trained with.
    #[test]
    fn vocab_matches_model_file() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../kural/model.kural");
        let m = crate::model::Model::load(path).expect("model file");
        let model_vocab: Vec<String> = m.header.get("grammar").get("vocab").arr().iter().map(|v| v.str().to_string()).collect();
        let lang = Language::load();
        assert_eq!(lang.vocab, model_vocab);
        assert_eq!(lang.stages.len(), 290);
        assert_eq!(lang.commands.len(), 192);
    }
}
