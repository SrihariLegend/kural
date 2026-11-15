//! kural-engine: the run-time half of kural. Loads the model file and turns a request into programs.
//!
//! Rust API: `Engine::load`, `Engine::words`, `Engine::parse`, `Engine::program_json`.
//! C API (see include/kural.h): `kural_open`, `kural_parse`, `kural_close`.

pub mod ffi;
pub mod json;
pub mod kernels;
pub mod model;
pub mod search;
pub mod tok;

pub use search::{Engine, Parse};

/// All candidate readings of a request as one JSON array:
/// [{"program": ..., "prob": p, "words": [request words], "used": [copied positions]}, ...]
pub fn parse_json(e: &Engine, text: &str, context: &[String], beam: usize, keep: usize) -> String {
    let (words, n) = e.words(text, context);
    let mut out = String::from("[");
    for (i, p) in e.parse(&words, n, beam, keep).iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let (prog, used) = e.program_json(&p.tokens, &words);
        out.push_str("{\"program\":");
        out.push_str(&prog);
        out.push_str(&format!(",\"prob\":{:.9},\"words\":[", p.prob));
        for (j, w) in words[..n].iter().enumerate() {
            if j > 0 {
                out.push(',');
            }
            json::write_str(&mut out, w);
        }
        out.push_str("],\"used\":[");
        out.push_str(&used.iter().map(|u| u.to_string()).collect::<Vec<_>>().join(","));
        out.push_str("]}");
    }
    out.push(']');
    out
}
