#[cfg(feature = "serde")]
fn main() {
    use dd_sketchy::DDSketch;

    let sketch = DDSketch::new(0.01).unwrap();
    let json = serde_json::to_string_pretty(&sketch).unwrap();
    println!("JSON output:\n{}", json);
}

#[cfg(not(feature = "serde"))]
fn main() {
    println!("Run with --features serde");
}
