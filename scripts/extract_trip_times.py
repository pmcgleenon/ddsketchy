#!/usr/bin/env python3
"""Streams the `trip_time` column (seconds, int64) out of one or more NYC
TLC FHVHV trip-data Parquet files as raw little-endian f64 bytes on stdout,
one value at a time, filtering out non-positive durations (DDSketch only
accepts positive values). Never materializes the full dataset in memory —
reads and emits in row-group batches.

Usage: python3 extract_trip_times.py file1.parquet file2.parquet ... > /dev/null  # (normally piped, not redirected to a file)
"""
import sys
import pyarrow.parquet as pq
import numpy as np

def main():
    paths = sys.argv[1:]
    if not paths:
        print("usage: extract_trip_times.py <parquet files...>", file=sys.stderr)
        sys.exit(1)

    out = sys.stdout.buffer
    total = 0
    kept = 0
    for path in paths:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(columns=["trip_time"], batch_size=1_000_000):
            col = batch.column(0).to_numpy(zero_copy_only=False)
            total += len(col)
            values = col.astype(np.float64)
            values = values[values > 0.0]
            kept += len(values)
            out.write(values.astype("<f8").tobytes())
        print(f"{path}: done (running total={total}, kept={kept})", file=sys.stderr)
    out.flush()
    print(f"TOTAL rows seen={total} kept(positive)={kept}", file=sys.stderr)

if __name__ == "__main__":
    main()
