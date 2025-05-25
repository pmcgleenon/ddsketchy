use crate::dd_sketchy::{DDSketch, DDSketchError};

fn create_sketch(alpha: f64) -> Result<DDSketch, DDSketchError> {
    DDSketch::new(alpha)
}

fn get_percentile(sketch: &DDSketch, percentile: f64) -> Result<f64, DDSketchError> {
    sketch.quantile(percentile / 100.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_usage() -> Result<(), DDSketchError> {
        // Create a new sketch with 1% relative error
        let mut sketch = create_sketch(0.01)?;

        // Add some values
        sketch.add(1.0);
        sketch.add(1.0);
        sketch.add(1.0);

        // Get the 50th percentile
        let p50 = get_percentile(&sketch, 50.0)?;
        assert_eq!(p50, 1.0);

        Ok(())
    }

    #[test]
    fn test_invalid_alpha() {
        // Test alpha values outside valid range
        assert!(matches!(create_sketch(0.0), Err(DDSketchError::InvalidAlpha)));
        assert!(matches!(create_sketch(1.0), Err(DDSketchError::InvalidAlpha)));
        assert!(matches!(create_sketch(-1.0), Err(DDSketchError::InvalidAlpha)));
        assert!(matches!(create_sketch(2.0), Err(DDSketchError::InvalidAlpha)));
    }

    #[test]
    fn test_invalid_percentile() -> Result<(), DDSketchError> {
        let sketch = create_sketch(0.01)?;
        
        // Test percentile values outside valid range
        assert!(matches!(get_percentile(&sketch, -1.0), Err(DDSketchError::InvalidQuantile)));
        assert!(matches!(get_percentile(&sketch, 101.0), Err(DDSketchError::InvalidQuantile)));
        
        Ok(())
    }
} 