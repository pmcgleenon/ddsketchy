# dd-sketchy

This is a Rust implementation of the [DDSketch](https://arxiv.org/pdf/1908.10693.pdf) quantile sketch algorithm. 
DDSketch is a fully-mergeable quantile sketch with relative-error guarantees and is extremely fast.
It has no dependencies to other crates.

# dd-sketchy Features

* Implements the DDSketch algorithm with configurable relative error guarantees
* Optimized for high-throughput data collection scenarios
* Memory-efficient with automatic bin collapsing
* Thread-safe operations for concurrent usage
* Support for both positive and negative values
* Designed for distributed systems with efficient sketch merging

## Implementation Details

dd-sketchy is built with performance and reliability in mind:

* Efficient logarithmic binning scheme
* Automatic memory management with configurable thresholds
* Configurable error bounds for different use cases

## Usage

```rust
use dd_sketchy::{DDSketch, DDSketchError};

fn create_sketch(alpha: f64) -> Result<DDSketch, DDSketchError> {
    DDSketch::new(alpha)
}

fn get_percentile(sketch: &DDSketch, percentile: f64) -> Result<f64, DDSketchError> {
    sketch.quantile(percentile / 100.0)
}

// Example usage with proper error handling
fn main() -> Result<(), DDSketchError> {
    // Create a new sketch with 1% relative error
    let mut sketch = create_sketch(0.01)?;

    // Add some values
    sketch.add(1.0);
    sketch.add(1.0);
    sketch.add(1.0);

    // Get the 50th percentile
    let p50 = get_percentile(&sketch, 50.0)?;
    println!("50th percentile: {}", p50);

    Ok(())
}
```

## Performance

dd-sketchy is optimized for high-throughput scenarios:

* Efficient sample insertion with minimal overhead
* Fast sketch merging for distributed systems
* Memory-efficient storage with automatic bin management
* Thread-safe operations for concurrent usage

## References

* [DDSketch: A Fast and Fully-Mergeable Quantile Sketch with Relative-Error Guarantees](https://arxiv.org/pdf/1908.10693.pdf) - The original paper describing the DDSketch algorithm
