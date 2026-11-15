//! kural-parse MODEL [--bench N] "request" [context words...]
//! Prints the readings as JSON. With --bench, parses the request N times and reports the time per parse.

use kural_engine::{parse_json, Engine};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() < 2 {
        eprintln!("usage: kural-parse MODEL [--bench N] \"request\" [context words...]");
        std::process::exit(2);
    }
    let t0 = Instant::now();
    let e = match Engine::load(&args[0]) {
        Ok(e) => e,
        Err(err) => {
            eprintln!("kural-parse: {err}");
            std::process::exit(1);
        }
    };
    let load = t0.elapsed();
    let mut rest = &args[1..];
    let mut bench = 0usize;
    if rest.len() >= 2 && rest[0] == "--bench" {
        bench = rest[1].parse().unwrap_or(100);
        rest = &rest[2..];
    }
    let text = &rest[0];
    let ctx: Vec<String> = rest[1..].to_vec();
    let t1 = Instant::now();
    let out = parse_json(&e, text, &ctx, 4, 3);
    let first = t1.elapsed();
    println!("{out}");
    if bench > 0 {
        let t = Instant::now();
        for _ in 0..bench {
            std::hint::black_box(parse_json(&e, text, &ctx, 4, 3));
        }
        let per = t.elapsed().as_secs_f64() * 1000.0 / bench as f64;
        eprintln!("load {:.2} ms, first parse {:.2} ms, then {:.3} ms per parse", load.as_secs_f64() * 1e3, first.as_secs_f64() * 1e3, per);
    }
}
