//! Large-scale differential correctness validation for the bidirectional
//! (interleaved two-pointer) `Store::key_at_rank` scan, added to catch the
//! exact class of bug found during its development: a rank-based
//! forward/backward heuristic silently returning a far-off-target bucket
//! for data whose mass is unevenly distributed across the log-spaced
//! bucket array (see `Store::key_at_rank`'s doc comment for the full story).
//!
//! Two complementary checks:
//! - `exact_ground_truth_*`: tens of millions of values per distribution,
//!   compared against a true sorted-array quantile (not another sketch).
//!   Catches any case where ddsketchy's own alpha guarantee is violated.
//! - `billion_scale_vs_sketches_ddsketch`: values are streamed into both
//!   ddsketchy and `sketches-ddsketch` (an independent Datadog-algorithm
//!   port) without ever being stored, so memory stays O(bucket count)
//!   regardless of N — this is what makes genuinely billion-scale manageable
//!   on a machine that can't hold a billion stored f64s (8GB) comfortably.
//!   Two independently-implemented sketches agreeing closely at that scale,
//!   across multiple distributions, is strong evidence against a subtle
//!   divergence that only manifests at extreme bucket counts or extreme
//!   run lengths.
//!
//! These are expensive (the billion-scale ones take real minutes even in
//! release mode) and marked `#[ignore]` so `cargo test` stays fast by
//! default. Run them explicitly with:
//!   cargo test --release --test large_scale_differential -- --ignored --nocapture

use ddsketchy::DDSketch;
use rand::{rngs::StdRng, Rng, SeedableRng};
use rand_distr::{Distribution, Exp, LogNormal, Normal, Pareto};
use sketches_ddsketch::{Config, DDSketch as ReferenceDDSketch};

const ALPHA: f64 = 0.01;
const PROBES: usize = 999;

fn assert_within_alpha(name: &str, q: f64, estimate: f64, exact: f64, alpha: f64, max_seen: &mut f64) {
    if exact == 0.0 {
        return;
    }
    let rel_error = (estimate - exact).abs() / exact.abs();
    *max_seen = max_seen.max(rel_error);
    assert!(
        rel_error <= alpha + 1e-9,
        "{name} q={q}: estimate={estimate}, exact={exact}, rel_error={rel_error} > alpha={alpha}"
    );
}

/// Exact-ground-truth check: build from `n` streamed values, keep only the
/// sorted array (not the generator), compare quantile() against the true
/// rank at `PROBES` points spanning (0, 1).
fn differential_exact(name: &str, n: usize, alpha: f64, mut gen: impl FnMut(&mut StdRng) -> f64, seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sketch = DDSketch::new(alpha).unwrap();
    let mut values = Vec::with_capacity(n);
    for _ in 0..n {
        let v = gen(&mut rng);
        sketch.add(v);
        values.push(v);
    }
    values.sort_unstable_by(f64::total_cmp);

    let mut max_rel_error = 0.0_f64;
    for i in 1..=PROBES {
        let q = i as f64 / (PROBES + 1) as f64;
        let rank = ((q * (n as f64 - 1.0)) as usize).min(n - 1);
        let exact = values[rank];
        let estimate = sketch.quantile(q).unwrap();
        assert_within_alpha(name, q, estimate, exact, alpha, &mut max_rel_error);
    }
    eprintln!("{name}: n={n} alpha={alpha} max_rel_error={max_rel_error:.6}");
}

#[test]
#[ignore]
fn exact_ground_truth_uniform_wide_pathological_case() {
    // The exact distribution shape that originally broke the rank-based
    // bidirectional heuristic: uniform BY VALUE (not by log) across six
    // decades, which concentrates most rank-mass in a short run of
    // log-spaced buckets near the top. This is the permanent regression
    // test for that specific failure mode.
    differential_exact(
        "uniform_wide[1,1e6)",
        20_000_000,
        ALPHA,
        |rng| rng.random_range(1.0..1_000_000.0),
        42,
    );
}

#[test]
#[ignore]
fn exact_ground_truth_uniform_even_wider() {
    // More extreme than the original failure case, spanning twelve decades
    // instead of six, to stress the scan beyond what's already known to
    // have broken once.
    differential_exact(
        "uniform_wide[1e-6,1e6)",
        20_000_000,
        ALPHA,
        |rng| rng.random_range(1e-6..1_000_000.0),
        43,
    );
}

#[test]
#[ignore]
fn exact_ground_truth_log_uniform() {
    // The "easy" case for comparison: uniform in log-space populates the
    // log-spaced bucket array roughly evenly, so this should never have
    // been the problem — included as a control.
    differential_exact(
        "log_uniform[1,1e6)",
        20_000_000,
        ALPHA,
        |rng| {
            let u: f64 = rng.random();
            (1.0_f64 * 1_000_000f64.powf(u)).clamp(1.0, 999_999.0)
        },
        44,
    );
}

#[test]
#[ignore]
fn exact_ground_truth_exponential() {
    let exp = Exp::new(1.0).unwrap();
    differential_exact(
        "exponential(lambda=1)",
        20_000_000,
        ALPHA,
        move |rng| exp.sample(rng),
        45,
    );
}

#[test]
#[ignore]
fn exact_ground_truth_pareto_heavy_tailed() {
    let pareto = Pareto::new(1.0, 1.0).unwrap();
    // Heavy-tailed data gets a looser alpha, matching the convention
    // already used in tests/error_bounds_validation.rs for Pareto.
    differential_exact(
        "pareto(scale=1,shape=1)",
        20_000_000,
        0.05,
        move |rng| pareto.sample(rng),
        46,
    );
}

#[test]
#[ignore]
fn exact_ground_truth_lognormal() {
    let lognormal = LogNormal::new(0.0, 2.0).unwrap();
    differential_exact(
        "lognormal(mu=0,sigma=2)",
        20_000_000,
        ALPHA,
        move |rng| {
            let v: f64 = lognormal.sample(rng);
            v.max(1e-9)
        },
        47,
    );
}

/// Streamed check against an independent Datadog-algorithm port: values
/// are generated and fed into both sketches one at a time and never
/// stored, so this scales to N in the billions without proportional memory
/// growth (both sketches stay O(bucket count) regardless of N).
fn differential_streamed_vs_reference(
    name: &str,
    n: u64,
    mut gen: impl FnMut(&mut StdRng) -> f64,
    seed: u64,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut sketch = DDSketch::new(ALPHA).unwrap();
    let mut reference = ReferenceDDSketch::new(Config::defaults());

    for _ in 0..n {
        let v = gen(&mut rng);
        sketch.add(v);
        reference.add(v);
    }

    assert_eq!(sketch.count(), n, "{name}: ddsketchy lost or gained values");

    // Each sketch independently holds its own alpha-bounded error vs the
    // (unavailable, at this scale) true value, so in the worst case they
    // can disagree with each other by up to ~2*alpha even with both
    // correct; that's the bound checked here, not alpha alone.
    let mut max_rel_diff = 0.0_f64;
    for i in 1..=PROBES {
        let q = i as f64 / (PROBES + 1) as f64;
        let ours = sketch.quantile(q).unwrap();
        let theirs = reference.quantile(q).unwrap().unwrap();
        if theirs != 0.0 {
            let rel_diff = (ours - theirs).abs() / theirs.abs();
            max_rel_diff = max_rel_diff.max(rel_diff);
            assert!(
                rel_diff <= 2.0 * ALPHA + 1e-9,
                "{name} q={q}: ddsketchy={ours}, sketches-ddsketch={theirs}, rel_diff={rel_diff}"
            );
        }
    }
    eprintln!("{name}: n={n} max_rel_diff_vs_reference={max_rel_diff:.6}");
}

#[test]
#[ignore]
fn billion_scale_vs_sketches_ddsketch_uniform_wide() {
    differential_streamed_vs_reference(
        "billion-scale uniform_wide[1,1e6)",
        1_000_000_000,
        |rng| rng.random_range(1.0..1_000_000.0),
        142,
    );
}

#[test]
#[ignore]
fn billion_scale_vs_sketches_ddsketch_log_uniform() {
    differential_streamed_vs_reference(
        "billion-scale log_uniform[1,1e6)",
        1_000_000_000,
        |rng| {
            let u: f64 = rng.random();
            (1.0_f64 * 1_000_000f64.powf(u)).clamp(1.0, 999_999.0)
        },
        144,
    );
}

#[test]
#[ignore]
fn billion_scale_vs_sketches_ddsketch_normal() {
    let normal = Normal::new(1_000_000.0, 200_000.0).unwrap();
    differential_streamed_vs_reference(
        "billion-scale normal(mean=1e6,sd=2e5)",
        1_000_000_000,
        move |rng| {
            let v: f64 = normal.sample(rng);
            v.max(1e-6)
        },
        148,
    );
}
