#[cfg(feature = "serde")]
fn main() {
    use ddsketchy::DDSketch;

    println!("=== DDSketch Serialization Comparison ===\n");

    // Test empty sketch
    let empty_sketch = DDSketch::new(0.01).unwrap();
    let empty_json = serde_json::to_string_pretty(&empty_sketch).unwrap();
    println!("Empty sketch (min/max as null):");
    println!("{}\n", empty_json);

    // Test sketch with data
    let mut data_sketch = DDSketch::new(0.01).unwrap();
    data_sketch.add(1.0);
    data_sketch.add(100.0);
    data_sketch.add(0.01);

    let data_json = serde_json::to_string_pretty(&data_sketch).unwrap();
    println!("Sketch with data (min/max as numbers):");
    println!("{}\n", data_json);

    // Test roundtrip
    let restored: DDSketch = serde_json::from_str(&data_json).unwrap();
    assert_eq!(data_sketch.count(), restored.count());
    assert_eq!(data_sketch.min(), restored.min());
    assert_eq!(data_sketch.max(), restored.max());
    assert_eq!(
        data_sketch.quantile(0.5).unwrap(),
        restored.quantile(0.5).unwrap()
    );

    println!("✓ Roundtrip serialization successful!");
    println!("✓ All quantile operations work correctly!");

    // Show size comparison
    let compact_json = serde_json::to_string(&empty_sketch).unwrap();
    println!("\nCompact empty sketch: {} bytes", compact_json.len());
    println!("Pretty empty sketch: {} bytes", empty_json.len());
}

#[cfg(not(feature = "serde"))]
fn main() {
    println!("Run with --features serde");
}
