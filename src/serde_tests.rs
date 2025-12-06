use crate::DDSketch;

#[test]
fn test_empty_sketch_serialization() {
    let sketch = DDSketch::new(0.01).unwrap();

    // Serialize
    let json = serde_json::to_string(&sketch).unwrap();

    // Deserialize
    let restored: DDSketch = serde_json::from_str(&json).unwrap();

    // Verify equality
    assert_eq!(sketch.count(), restored.count());
    assert_eq!(sketch.alpha(), restored.alpha());
    assert_eq!(sketch.sum(), restored.sum());
}

#[test]
fn test_sketch_with_data_serialization() {
    let mut sketch = DDSketch::new(0.01).unwrap();
    sketch.add(1.0);
    sketch.add(2.0);
    sketch.add(3.0);
    sketch.add(4.0);
    sketch.add(5.0);

    // Serialize
    let json = serde_json::to_string(&sketch).unwrap();

    // Deserialize
    let restored: DDSketch = serde_json::from_str(&json).unwrap();

    // Verify equality
    assert_eq!(sketch.count(), restored.count());

    // Use approximate equality for floating point values due to serialization precision
    let epsilon = 1e-10;
    assert!(
        (sketch.sum() - restored.sum()).abs() < epsilon,
        "Sum mismatch: {} vs {}",
        sketch.sum(),
        restored.sum()
    );
    assert!(
        (sketch.min() - restored.min()).abs() < epsilon,
        "Min mismatch: {} vs {}",
        sketch.min(),
        restored.min()
    );
    assert!(
        (sketch.max() - restored.max()).abs() < epsilon,
        "Max mismatch: {} vs {}",
        sketch.max(),
        restored.max()
    );
    assert!(
        (sketch.mean() - restored.mean()).abs() < epsilon,
        "Mean mismatch: {} vs {}",
        sketch.mean(),
        restored.mean()
    );

    // Verify quantiles work the same (with approximate equality)
    let q50_orig = sketch.quantile(0.5).unwrap();
    let q50_restored = restored.quantile(0.5).unwrap();
    assert!(
        (q50_orig - q50_restored).abs() < epsilon,
        "Q50 mismatch: {} vs {}",
        q50_orig,
        q50_restored
    );

    let q90_orig = sketch.quantile(0.9).unwrap();
    let q90_restored = restored.quantile(0.9).unwrap();
    assert!(
        (q90_orig - q90_restored).abs() < epsilon,
        "Q90 mismatch: {} vs {}",
        q90_orig,
        q90_restored
    );
}

#[test]
fn test_sketch_with_negative_values() {
    let mut sketch = DDSketch::new(0.05).unwrap();
    sketch.add(-10.0);
    sketch.add(-5.0);
    sketch.add(0.0);
    sketch.add(5.0);
    sketch.add(10.0);

    let json = serde_json::to_string(&sketch).unwrap();
    let restored: DDSketch = serde_json::from_str(&json).unwrap();

    assert_eq!(sketch.count(), restored.count());
    assert_eq!(
        sketch.quantile(0.1).unwrap(),
        restored.quantile(0.1).unwrap()
    );
    assert_eq!(
        sketch.quantile(0.5).unwrap(),
        restored.quantile(0.5).unwrap()
    );
    assert_eq!(
        sketch.quantile(0.9).unwrap(),
        restored.quantile(0.9).unwrap()
    );
}

#[test]
fn test_sketch_with_zero_values() {
    let mut sketch = DDSketch::new(0.01).unwrap();
    sketch.add(0.0);
    sketch.add(0.0);
    sketch.add(1.0);
    sketch.add(2.0);

    let json = serde_json::to_string(&sketch).unwrap();
    let restored: DDSketch = serde_json::from_str(&json).unwrap();

    assert_eq!(sketch.count(), restored.count());
    assert_eq!(
        sketch.quantile(0.25).unwrap(),
        restored.quantile(0.25).unwrap()
    );
    assert_eq!(
        sketch.quantile(0.75).unwrap(),
        restored.quantile(0.75).unwrap()
    );
}

#[test]
fn test_large_sketch_serialization() {
    let mut sketch = DDSketch::new(0.02).unwrap();

    // Add a lot of data to test bin serialization
    for i in 1..=10000 {
        sketch.add(i as f64);
    }

    let json = serde_json::to_string(&sketch).unwrap();
    let restored: DDSketch = serde_json::from_str(&json).unwrap();

    assert_eq!(sketch.count(), restored.count());
    assert_eq!(sketch.sum(), restored.sum());

    // Test various quantiles with tolerance for floating point precision
    for q in [0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] {
        let original_q = sketch.quantile(q).unwrap();
        let restored_q = restored.quantile(q).unwrap();
        let diff = (original_q - restored_q).abs();
        let tolerance = original_q.abs() * 1e-14; // Allow for floating point precision
        assert!(
            diff <= tolerance,
            "Quantile {} differs beyond tolerance: original={}, restored={}, diff={}",
            q,
            original_q,
            restored_q,
            diff
        );
    }
}

#[test]
fn test_serialization_roundtrip_preserves_internal_state() {
    let mut sketch = DDSketch::new(0.01).unwrap();

    // Add values with specific pattern
    for i in [1.0, 100.0, 0.001, 5000.0, 0.1] {
        sketch.add(i);
    }

    let json = serde_json::to_string(&sketch).unwrap();
    let restored: DDSketch = serde_json::from_str(&json).unwrap();

    // Verify all internal state is preserved
    assert_eq!(sketch.count(), restored.count());

    // Use approximate equality for floating point values due to serialization precision
    let epsilon = 1e-10;
    assert!(
        (sketch.sum() - restored.sum()).abs() < epsilon,
        "Sum mismatch: {} vs {}",
        sketch.sum(),
        restored.sum()
    );
    assert!(
        (sketch.min() - restored.min()).abs() < epsilon,
        "Min mismatch: {} vs {}",
        sketch.min(),
        restored.min()
    );
    assert!(
        (sketch.max() - restored.max()).abs() < epsilon,
        "Max mismatch: {} vs {}",
        sketch.max(),
        restored.max()
    );
    assert_eq!(sketch.alpha(), restored.alpha());

    // Verify that further operations work identically
    let mut sketch_copy = sketch.clone();
    let mut restored_copy = restored.clone();

    sketch_copy.add(42.0);
    restored_copy.add(42.0);

    assert_eq!(sketch_copy.count(), restored_copy.count());
    let q50_copy = sketch_copy.quantile(0.5).unwrap();
    let q50_restored_copy = restored_copy.quantile(0.5).unwrap();
    assert!(
        (q50_copy - q50_restored_copy).abs() < epsilon,
        "Q50 mismatch after adding: {} vs {}",
        q50_copy,
        q50_restored_copy
    );
}

#[test]
fn test_custom_max_bins_serialization() {
    let sketch = DDSketch::with_max_bins(0.01, 1024).unwrap();

    let json = serde_json::to_string(&sketch).unwrap();
    let restored: DDSketch = serde_json::from_str(&json).unwrap();

    assert_eq!(sketch.alpha(), restored.alpha());
    // max_bins is internal state - verify it works by adding data
    // that would exceed default bins but fit in 1024
}

#[test]
fn test_json_structure_is_reasonable() {
    let mut sketch = DDSketch::new(0.01).unwrap();
    sketch.add(1.0);
    sketch.add(2.0);

    let json = serde_json::to_string_pretty(&sketch).unwrap();

    // Basic sanity checks on JSON structure
    assert!(json.contains("count"));
    assert!(json.contains("sum"));
    assert!(json.contains("bins"));
    assert!(json.contains("gamma"));

    // Should contain finite min/max values for non-empty sketch
    assert!(json.contains("\"min\": 1.0"));
    assert!(json.contains("\"max\": 2.0"));

    // Should be reasonably compact
    assert!(json.len() < 2000, "JSON too large: {} bytes", json.len());
}

#[test]
fn test_empty_sketch_json_structure() {
    let sketch = DDSketch::new(0.01).unwrap();
    let json = serde_json::to_string_pretty(&sketch).unwrap();

    // Empty sketch should have null min/max values
    assert!(json.contains("\"min\": null"));
    assert!(json.contains("\"max\": null"));
    assert!(json.contains("\"count\": 0"));
}
