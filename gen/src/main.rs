//! kural-gen data    --n N --seed S --out DIR [--shard-size M] [--threads T]   write training shards
//! kural-gen show    [--n 20] [--seed S] [--follow]                             print examples as text
//! kural-gen lexicon [--n 400000] [--seed 77] [--out kural/lexicon.json]       build the vocabulary-check table
//! kural-gen check   [--n 1000000] [--seed S] [--threads T]                     count generator errors, no output

use kural_gen::mix;
use kural_gen::req::{delinearise, lang, DV, Example};
use kural_gen::rng::Rng;
use kural_gen::shard::ShardWriter;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

fn opt(args: &[String], name: &str, default: &str) -> String {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned().unwrap_or_else(|| default.to_string())
}

fn shard_seed(seed: u64, i: u64) -> u64 {
    let mut z = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i.wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03);
    z = (z ^ (z >> 31)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^ (z >> 29)
}

fn threads(args: &[String]) -> usize {
    let t: usize = opt(args, "--threads", "0").parse().unwrap_or(0);
    if t > 0 {
        t
    } else {
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
    }
}

pub fn program_text(ex: &Example) -> String {
    let prog = delinearise(&ex.out, &ex.words);
    prog.iter()
        .map(|pipe| {
            pipe.iter()
                .map(|(n, args)| {
                    let a: Vec<String> = args
                        .iter()
                        .map(|(s, v)| match v {
                            DV::Str(t) => format!("{s}={t:?}"),
                            DV::Enum(e) => format!("{s}={e}"),
                            DV::Null => format!("{s}=-"),
                        })
                        .collect();
                    if a.is_empty() { n.to_string() } else { format!("{n}({})", a.join(", ")) }
                })
                .collect::<Vec<_>>()
                .join(" | ")
        })
        .collect::<Vec<_>>()
        .join("  ;;  ")
}

fn data(args: &[String], write: bool) {
    let n: u64 = opt(args, "--n", "1000000").parse().expect("--n");
    let seed: u64 = opt(args, "--seed", "1").parse().expect("--seed");
    let shard_size: u64 = opt(args, "--shard-size", "1000000").parse().expect("--shard-size");
    let out = opt(args, "--out", "");
    if write {
        assert!(!out.is_empty(), "--out DIR is required");
        std::fs::create_dir_all(&out).expect("create --out");
    }
    let nshards = n.div_ceil(shard_size);
    let next = Arc::new(AtomicUsize::new(0));
    let (errors, rejected) = (Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)));
    let t0 = Instant::now();
    let nt = threads(args).min(nshards.max(1) as usize).max(1);
    let handles: Vec<_> = (0..nt)
        .map(|_| {
            let (next, errors, rejected, out) = (next.clone(), errors.clone(), rejected.clone(), out.clone());
            std::thread::spawn(move || loop {
                let i = next.fetch_add(1, Ordering::Relaxed) as u64;
                if i >= nshards {
                    break;
                }
                let count = shard_size.min(n - i * shard_size);
                let mut r = Rng::new(shard_seed(seed, i));
                let mut w = ShardWriter::new();
                let (mut e, mut rj) = (0u64, 0u64);
                for _ in 0..count {
                    let ex = mix::accepted(&mut r, &mut e, &mut rj);
                    if write {
                        w.push(&ex);
                    }
                }
                errors.fetch_add(e, Ordering::Relaxed);
                rejected.fetch_add(rj, Ordering::Relaxed);
                if write {
                    w.write(&std::path::Path::new(&out).join(format!("shard-{i:06}.kds"))).expect("write shard");
                }
            })
        })
        .collect();
    for h in handles {
        h.join().expect("worker");
    }
    let dt = t0.elapsed().as_secs_f64();
    eprintln!(
        "{} {n} examples in {nshards} shard(s) with {nt} thread(s) in {dt:.1}s: {:.0} examples/s; generator errors {}, over-length rejected {}",
        if write { "wrote" } else { "generated" },
        n as f64 / dt,
        errors.load(Ordering::Relaxed),
        rejected.load(Ordering::Relaxed)
    );
}

fn show(args: &[String]) {
    let n: usize = opt(args, "--n", "20").parse().unwrap_or(20);
    let mut r = Rng::new(opt(args, "--seed", "0").parse().unwrap_or(0));
    let follow = args.iter().any(|a| a == "--follow");
    let mut shown = 0;
    while shown < n {
        let ex = if follow { mix::followup(&mut r) } else { mix::example(&mut r) };
        match ex {
            Ok(ex) => {
                let req = ex.words[..ex.n_req].join(" ");
                let ctx = if ex.words.len() > ex.n_req { format!("      [after: {}]", ex.words[ex.n_req..].join(" ")) } else { String::new() };
                println!("{req}{ctx}\n   -> {}", program_text(&ex));
                shown += 1;
            }
            Err(e) => println!("(error: {e})"),
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let _ = lang();
    match args.first().map(String::as_str) {
        Some("data") => data(&args, true),
        Some("check") => data(&args, false),
        Some("show") => show(&args),
        Some("lexicon") => {
            let n: usize = opt(&args, "--n", "400000").parse().unwrap_or(400_000);
            let seed: u64 = opt(&args, "--seed", "77").parse().unwrap_or(77);
            let out = opt(&args, "--out", "kural/lexicon.json");
            let t0 = Instant::now();
            let (generic, stages, errors) = kural_gen::lexicon::build(n, seed);
            std::fs::write(&out, kural_gen::lexicon::to_json(&generic, &stages)).expect("write lexicon");
            eprintln!(
                "wrote {out}: {} stages, {} words, {} generic words, from {n} examples in {:.1}s ({errors} generator errors)",
                stages.len(),
                stages.values().map(|v| v.len()).sum::<usize>(),
                generic.len(),
                t0.elapsed().as_secs_f64()
            );
        }
        _ => {
            eprintln!("usage: kural-gen data|check|show|lexicon [options]  (see the top of gen/src/main.rs)");
            std::process::exit(2);
        }
    }
}
