#!/usr/bin/env python3
"""Render assets/quantile_bench.svg (and the README tables) from the criterion
output of benches/quantile_compare.rs. Standard library only.

Usage:
    RUSTFLAGS="-C target-cpu=native" cargo bench --bench quantile_compare
    python3 scripts/plot_quantile_bench.py \
        --machine "AWS c6a.xlarge (AMD EPYC 7R13), target-cpu=native" \
        --tables

Reads every <criterion-dir>/quantile/*/*/new/{benchmark,estimates}.json and
plots each benchmark's mean time. --tables also prints the README's markdown
tables to stdout.
"""
import argparse
import glob
import json
import math
import os
import sys

IMPLS = ["ddsketchy", "sketches-ddsketch", "quantile-sketch"]
COLORS = {"ddsketchy": "#2563eb", "sketches-ddsketch": "#94a3b8", "quantile-sketch": "#f59e0b"}
DISTS = [
    ("uniform", "Uniform by value, 1..1e6", "Uniform by value, `1..1_000_000`"),
    ("log-uniform", "Log-uniform, 1..1e6", "Log-uniform, `1..1_000_000`"),
    ("lognormal", "Lognormal, σ=2", "Lognormal, μ=0, σ=2"),
    ("pareto", "Pareto, α=1", "Pareto, scale 1, α=1"),
]
QUANTILES = ["0.01", "0.1", "0.5", "0.9", "0.99"]
QLABELS = ["p01", "p10", "p50", "p90", "p99"]


def load(criterion_dir):
    """Return {(impl, dist, q): mean_ns} from criterion's saved estimates."""
    results = {}
    pattern = os.path.join(criterion_dir, "quantile", "*", "*", "new", "benchmark.json")
    for bench_path in glob.glob(pattern):
        with open(bench_path) as f:
            meta = json.load(f)
        with open(os.path.join(os.path.dirname(bench_path), "estimates.json")) as f:
            est = json.load(f)
        dist, _, q = meta["value_str"].rpartition("/")
        results[(meta["function_id"], dist, q)] = est["mean"]["point_estimate"]
    return results


def check_complete(results):
    missing = [
        f"{i} {d}/{q}" for d, _, _ in DISTS for i in IMPLS for q in QUANTILES
        if (i, d, q) not in results
    ]
    if missing:
        sys.exit(f"missing {len(missing)} results, e.g. {missing[:3]}; run the full bench first")


def nice_ceiling(v):
    """Smallest round axis top >= v with at most 6 gridline steps."""
    exp = 10 ** math.floor(math.log10(v))
    for tick in (exp / 10 * m for m in (1, 2, 2.5, 5, 10, 20, 25, 50)):
        steps = math.ceil(v / tick)
        if steps <= 6:
            return steps * tick, tick
    return math.ceil(v / exp) * exp, exp


def svg(results, machine):
    W, H = 960, 680
    pw, ph = 420, 220
    left0, top0, gapx, gapy = 80, 110, 470, 290
    ymax, tick = nice_ceiling(max(results.values()) * 1.05)
    font = "font-family=\"-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif\""
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" {font}>',
         f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
         f'<text x="{W/2}" y="34" text-anchor="middle" font-size="20" font-weight="600" fill="#111827">'
         'quantile() latency, 100,000-value sketch, 1% relative error</text>',
         f'<text x="{W/2}" y="56" text-anchor="middle" font-size="13" fill="#4b5563">'
         f'ns per call, lower is better. {machine}</text>']
    lx = W / 2 - 230
    for i, im in enumerate(IMPLS):
        x = lx + i * 165
        o.append(f'<rect x="{x}" y="70" width="14" height="14" rx="2" fill="{COLORS[im]}"/>')
        o.append(f'<text x="{x+20}" y="82" font-size="13" fill="#111827">{im}</text>')
    for pi, (dist, title, _) in enumerate(DISTS):
        px = left0 + (pi % 2) * gapx
        py = top0 + (pi // 2) * gapy + 20
        o.append(f'<text x="{px}" y="{py-10}" font-size="14" font-weight="600" fill="#111827">{title}</text>')
        t = 0.0
        while t <= ymax + 1e-9:
            y = py + ph - t / ymax * ph
            o.append(f'<line x1="{px}" y1="{y:.1f}" x2="{px+pw}" y2="{y:.1f}" '
                     f'stroke="{"#9ca3af" if t == 0 else "#e5e7eb"}" stroke-width="1"/>')
            o.append(f'<text x="{px-6}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#6b7280">{t:g}</text>')
            t += tick
        gw = pw / len(QUANTILES)
        bw = gw * 0.24
        for qi, (q, ql) in enumerate(zip(QUANTILES, QLABELS)):
            gx = px + qi * gw + (gw - 3 * bw) / 2
            for ii, im in enumerate(IMPLS):
                v = results[(im, dist, q)]
                h = v / ymax * ph
                o.append(f'<rect x="{gx + ii*bw:.1f}" y="{py+ph-h:.1f}" width="{bw-1:.1f}" height="{h:.1f}" '
                         f'fill="{COLORS[im]}"><title>{im} {ql}: {v:.0f} ns</title></rect>')
            o.append(f'<text x="{px+qi*gw+gw/2:.1f}" y="{py+ph+16}" text-anchor="middle" '
                     f'font-size="12" fill="#374151">{ql}</text>')
    o.append("</svg>")
    return "\n".join(o) + "\n"


def tables(results):
    out = []
    for dist, _, heading in DISTS:
        out.append(f"**{heading}**\n")
        out.append("| Implementation | " + " | ".join(QLABELS) + " |")
        out.append("|---" * (len(QLABELS) + 1) + "|")
        for im in IMPLS:
            out.append(f"| {im} | " + " | ".join(f"{results[(im, dist, q)]:.0f} ns" for q in QUANTILES) + " |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--criterion-dir", default="target/criterion")
    ap.add_argument("--out", default="assets/quantile_bench.svg")
    ap.add_argument("--machine", required=True, help="hardware and flags, shown under the title")
    ap.add_argument("--tables", action="store_true", help="also print the README markdown tables")
    args = ap.parse_args()
    results = load(args.criterion_dir)
    check_complete(results)
    with open(args.out, "w") as f:
        f.write(svg(results, args.machine))
    print(f"wrote {args.out}", file=sys.stderr)
    if args.tables:
        print(tables(results))


if __name__ == "__main__":
    main()
