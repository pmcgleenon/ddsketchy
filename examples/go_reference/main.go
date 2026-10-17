// Reads raw little-endian f64 values from stdin (the same stream produced
// by scripts/extract_trip_times.py and consumed by
// examples/nyc_taxi_quantile_validation.rs) and builds a DDSketch using
// Datadog's own Go implementation (the original reference the DDSketch
// algorithm and this crate's Logarithmic mapping are both ported from).
// Prints quantiles at the same probe points nyc_taxi_quantile_validation
// prints, so the two outputs can be compared by eye or by a small script.
//
// With no argument, uses NewDefaultDDSketch (an unbounded store, same as
// nyc_taxi_quantile_validation.rs's ddsketchy/sketches-ddsketch default
// bin caps). With a numeric argument, builds a bin-capped store instead
// (store.NewCollapsingLowestDenseStore), for a peak-RSS comparison against
// ddsketchy/sketches-ddsketch's own max_bins-limited stores.
//
// Usage:
//   python3 scripts/extract_trip_times.py file1.parquet file2.parquet... \
//     | go run ./examples/go_reference [max_bins]
package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"

	"github.com/DataDog/sketches-go/ddsketch"
	"github.com/DataDog/sketches-go/ddsketch/mapping"
	"github.com/DataDog/sketches-go/ddsketch/store"
)

const alpha = 0.01

func main() {
	var sketch *ddsketch.DDSketch
	if len(os.Args) > 1 {
		maxBins, err := strconv.Atoi(os.Args[1])
		if err != nil {
			panic(err)
		}
		m, err := mapping.NewLogarithmicMapping(alpha)
		if err != nil {
			panic(err)
		}
		sketch = ddsketch.NewDDSketch(
			m,
			store.NewCollapsingLowestDenseStore(maxBins),
			store.NewCollapsingLowestDenseStore(maxBins),
		)
	} else {
		var err error
		sketch, err = ddsketch.NewDefaultDDSketch(alpha)
		if err != nil {
			panic(err)
		}
	}

	reader := bufio.NewReaderSize(os.Stdin, 1<<20)
	buf := make([]byte, 8)
	var count uint64
	var skipped uint64

	for {
		_, err := io.ReadFull(reader, buf)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			panic(err)
		}
		bits := binary.LittleEndian.Uint64(buf)
		v := math.Float64frombits(bits)
		if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0.0 {
			skipped++
			continue
		}
		if err := sketch.Add(v); err != nil {
			panic(err)
		}
		count++
		if count%100_000_000 == 0 {
			fmt.Fprintf(os.Stderr, "...%d values streamed so far\n", count)
		}
	}

	fmt.Fprintf(os.Stderr, "done: count=%d skipped_nonfinite_or_nonpositive=%d\n", count, skipped)

	probes := []float64{0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999}
	for _, q := range probes {
		v, err := sketch.GetValueAtQuantile(q)
		if err != nil {
			panic(err)
		}
		fmt.Printf("q=%.3f: sketches-go=%.3f\n", q, v)
	}
}
