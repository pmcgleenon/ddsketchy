use ddsketchy::{DDSketch, DDSketchError};

fn main() -> Result<(), DDSketchError> {
    // Create a new sketch with 1% relative error
    let mut sketch = DDSketch::new(0.01)?;

    // Add some values
    sketch.add(1.0);
    sketch.add(2.0);
    sketch.add(3.0);
    sketch.add(4.0);
    sketch.add(5.0);

    // Get the 50th percentile (median)
    let median = sketch.quantile(0.5)?;
    println!("Median: {}", median); // Will print approximately 3.0

    // Get the 90th percentile
    let p90 = sketch.quantile(0.9)?;
    println!("90th percentile: {}", p90); // Will print approximately 4.6

    Ok(())
}
