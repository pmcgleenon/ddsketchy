#!/usr/bin/env python3
"""Computes EXACT quantiles of the `trip_time` column across one or more NYC
TLC FHVHV Parquet files, as ground truth to compare against ddsketchy /
sketches-ddsketch / sketches-go / quantile-sketch's estimates.

trip_time is a non-negative integer number of seconds with a modest observed
range (max ~240,473), so rather than sorting ~1 billion values (or even
holding them in memory), this builds an exact count-per-integer-second
histogram in one streaming pass -- a few MB regardless of row count -- then
walks its cumulative sum to find the value at each requested rank. This is
exact, not an approximation: no sketch, no sampling.

Usage: python3 exact_quantiles.py file1.parquet file2.parquet ...
"""
import sys
import pyarrow.parquet as pq
import numpy as np

# Must exceed the true max trip_time across the dataset; a value that's too
# small would silently clip high-end buckets via np.bincount raising instead
# (positional calls use the max seen, so a mismatch here is self-detecting).
HIST_SIZE = 300_001
PROBES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]


def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: exact_quantiles.py <parquet files...>", file=sys.stderr)
        sys.exit(1)

    counts = np.zeros(HIST_SIZE, dtype=np.uint64)
    total_seen = 0
    kept = 0
    max_value_seen = -1

    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["trip_time"], batch_size=1_000_000):
            col = batch.column(0).to_numpy(zero_copy_only=False)
            total_seen += len(col)
            col = col[col > 0]
            kept += len(col)
            if len(col) == 0:
                continue
            batch_max = int(col.max())
            if batch_max > max_value_seen:
                max_value_seen = batch_max
            if batch_max >= HIST_SIZE:
                raise SystemExit(
                    f"HIST_SIZE={HIST_SIZE} too small: saw trip_time={batch_max} in {path}"
                )
            batch_counts = np.bincount(col.astype(np.int64), minlength=HIST_SIZE)
            counts += batch_counts.astype(np.uint64)
        print(f"{path}: done (running total={total_seen}, kept={kept})", file=sys.stderr)

    n = int(counts.sum())
    print(f"TOTAL rows seen={total_seen} kept(positive)={kept} max_value_seen={max_value_seen}", file=sys.stderr)
    assert n == kept, f"histogram total {n} != kept {kept}"

    cumulative = np.cumsum(counts)
    for q in PROBES:
        rank = min(int(q * (n - 1)), n - 1)
        # First index whose cumulative count exceeds `rank` holds the value
        # at that rank (searchsorted on the cumulative sum, 'right' side so
        # rank==cumulative[i]-1 still lands in bucket i, not i+1).
        value = int(np.searchsorted(cumulative, rank, side="right"))
        print(f"q={q:.3f}: exact={value:.3f}")


if __name__ == "__main__":
    main()
