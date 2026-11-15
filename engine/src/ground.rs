//! Facts about the machine the command will run on (kural/ground.py). With live = false nothing is looked at.

use crate::py;
use std::collections::{BTreeMap, HashSet};
use std::path::Path;

const DEFAULT_TOOLS: &[&str] = &["curl", "dig", "lsof", "tree", "wl-copy", "wl-paste", "gio", "notify-send", "xdg-open", "rsync"];

pub fn package_manager() -> String {
    for pm in ["pacman", "apt", "dnf", "zypper", "brew"] {
        if py::which(pm) {
            return pm.into();
        }
    }
    "pacman".into()
}

#[derive(Clone)]
pub struct Env {
    pub live: bool,
    pub cwd: String,
    pub pm: String,
    pub prefs: BTreeMap<String, bool>,
    pub last: Option<String>,
    pub final_: bool,
    pub notes: Vec<String>,
    /// how far a missing name may be bent towards an existing one: 2 read-only, 1 changes things, 0 never
    pub level: u8,
    pub exempt: HashSet<String>,
}

impl Env {
    pub fn new(cwd: Option<String>, live: bool, pm: Option<String>, prefs: BTreeMap<String, bool>, last: Option<String>, final_: bool) -> Env {
        let cwd = cwd.filter(|c| !c.is_empty()).unwrap_or_else(|| {
            if live {
                std::env::current_dir().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|_| "/".into())
            } else {
                "/".into()
            }
        });
        let pm = pm.unwrap_or_else(|| if live { package_manager() } else { "pacman".into() });
        Env { live, cwd, pm, prefs, last, final_, notes: Vec::new(), level: 2, exempt: HashSet::new() }
    }

    /// Env(live=False, pm="pacman"): repeatable, looks at nothing.
    pub fn offline(last: Option<String>) -> Env {
        Env::new(None, false, Some("pacman".into()), BTreeMap::new(), last, true)
    }

    pub fn fresh(&self) -> Env {
        Env::new(Some(self.cwd.clone()), self.live, Some(self.pm.clone()), self.prefs.clone(), self.last.clone(), self.final_)
    }

    pub fn pref(&self, k: &str) -> bool {
        self.prefs.get(k).copied().unwrap_or(false)
    }

    pub fn note(&mut self, text: String) {
        if !self.notes.contains(&text) {
            self.notes.push(text);
        }
    }

    pub fn has(&self, tool: &str) -> bool {
        if self.live { py::which(tool) } else { DEFAULT_TOOLS.contains(&tool) }
    }

    pub fn clipboard(&mut self, paste: bool) -> String {
        let wayland = std::env::var("WAYLAND_DISPLAY").map(|v| !v.is_empty()).unwrap_or(false);
        if self.has("wl-copy") || (!self.has("xclip") && wayland) {
            if !self.has("wl-copy") {
                self.note("needs the wl-clipboard package".into());
            }
            return if paste { "wl-paste" } else { "wl-copy" }.into();
        }
        if paste { "xclip -o -selection clipboard" } else { "xclip -selection clipboard" }.into()
    }

    pub fn resolve(&self, p: &str) -> String {
        py::normpath(&py::join(&self.cwd, &py::expanduser(p)))
    }
    pub fn exists(&self, p: &str) -> bool {
        self.live && std::fs::symlink_metadata(self.resolve(p)).is_ok()
    }
    pub fn isdir(&self, p: &str) -> bool {
        self.live && Path::new(&self.resolve(p)).is_dir()
    }
    pub fn isfile(&self, p: &str) -> bool {
        self.live && Path::new(&self.resolve(p)).is_file()
    }

    fn closest(&self, name: &str, want_dir: Option<bool>) -> Option<String> {
        let (d, base) = py::split(name);
        let folder = if d.is_empty() { self.cwd.clone() } else { self.resolve(&d) };
        let rd = std::fs::read_dir(&folder).ok()?;
        let mut entries: Vec<String> = rd.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        if let Some(w) = want_dir {
            entries.retain(|e| Path::new(&py::join(&folder, e)).is_dir() == w);
        }
        let low = base.to_lowercase();
        let mut rules: Vec<Box<dyn Fn(&str) -> bool>> = vec![Box::new(|e: &str| e.to_lowercase() == low)];
        if self.level >= 2 {
            let l2 = low.clone();
            rules.push(Box::new(move |e: &str| py::splitext_root(e).to_lowercase() == l2));
            let l3 = low.clone();
            rules.push(Box::new(move |e: &str| l3.chars().count() >= 3 && e.to_lowercase().starts_with(&l3)));
        }
        for rule in &rules {
            let hits: Vec<&String> = entries.iter().filter(|e| rule(e)).collect();
            if hits.len() == 1 {
                return Some(py::join(&d, hits[0]));
            }
            if !hits.is_empty() {
                return None;
            }
        }
        if self.level < 2 {
            return None;
        }
        let close = py::close_matches(&base, entries.iter().map(|s| s.as_str()), 0.8);
        if close.len() == 1 {
            return Some(py::join(&d, close[0].1));
        }
        None
    }

    /// The file the user means: unchanged unless it does not exist and exactly one real name is clearly meant.
    pub fn file(&mut self, name: &str) -> String {
        if !self.live || name.is_empty() || self.level == 0 || self.exempt.contains(name) || name.contains(['*', '?', '[']) || self.exists(name) {
            return name.to_string();
        }
        let mut hit = self.closest(name, None);
        if hit.is_none() {
            let words: Vec<&str> = name.split_whitespace().filter(|w| !["the", "my", "a", "file", "folder"].contains(&w.to_lowercase().as_str())).collect();
            if words.len() == 1 && words[0] != name {
                if self.exists(words[0]) {
                    return words[0].to_string();
                }
                hit = self.closest(words[0], None);
            }
        }
        if let Some(h) = hit {
            self.note(format!("{name} taken to mean {h}"));
            return h;
        }
        name.to_string()
    }

    pub fn folder(&mut self, path: &str) -> String {
        if !self.live || path.is_empty() || self.level == 0 || self.exempt.contains(path) || [".", "..", "~", "/"].contains(&path)
            || path.starts_with('~') || path.starts_with('/') || self.exists(path)
        {
            return path.to_string();
        }
        if let Some(h) = self.closest(path, Some(true)) {
            self.note(format!("{path} taken to mean {h}"));
            return h;
        }
        path.to_string()
    }

    pub fn is_service(&self, name: &str) -> bool {
        if !(self.live && self.final_) || name.is_empty() || !py::isalnum(&name.replace(['-', '_', '.'], "")) {
            return false;
        }
        std::process::Command::new("systemctl")
            .args(["is-active", "--quiet", name])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    pub fn project(&self) -> Option<&'static str> {
        if !self.live {
            return None;
        }
        for (marker, kind) in [("Cargo.toml", "cargo"), ("package.json", "npm"), ("go.mod", "go"), ("pyproject.toml", "python"),
            ("setup.py", "python"), ("pytest.ini", "python"), ("Makefile", "make"), ("makefile", "make"),
            ("CMakeLists.txt", "cmake"), ("build.gradle", "gradle"), ("build.gradle.kts", "gradle"), ("pom.xml", "maven"),
            ("meson.build", "meson")]
        {
            if Path::new(&py::join(&self.cwd, marker)).exists() {
                return Some(kind);
            }
        }
        None
    }
}
