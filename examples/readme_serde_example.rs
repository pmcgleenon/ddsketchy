#[cfg(feature = "serde")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use dd_sketchy::DDSketch;

    let mut sketch = DDSketch::new(0.01)?;
    sketch.add(1.0);
    sketch.add(2.0);

    // Serialize to JSON
    let json = serde_json::to_string(&sketch)?;
    println!("Serialized: {}", json);

    // Deserialize from JSON
    let restored: DDSketch = serde_json::from_str(&json)?;

    // Verify the sketch works correctly
    assert_eq!(sketch.count(), restored.count());
    assert_eq!(sketch.quantile(0.5)?, restored.quantile(0.5)?);

    println!("✓ Serialization example works correctly!");
    Ok(())
}

#[cfg(not(feature = "serde"))]
fn main() {
    println!("Run with --features serde to test serialization");
}
