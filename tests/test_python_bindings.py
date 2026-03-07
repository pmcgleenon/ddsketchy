import unittest
import math
from ddsketchy import DDSketch


class TestDDSketchBasic(unittest.TestCase):
    def test_create_default_alpha(self):
        sketch = DDSketch()
        self.assertEqual(sketch.count, 0)
        self.assertTrue(sketch.is_empty())
        self.assertAlmostEqual(sketch.alpha, 0.01, places=10)

    def test_create_custom_alpha(self):
        sketch = DDSketch(alpha=0.05)
        self.assertAlmostEqual(sketch.alpha, 0.05, places=10)

    def test_invalid_alpha(self):
        with self.assertRaises(ValueError):
            DDSketch(alpha=0.0)
        with self.assertRaises(ValueError):
            DDSketch(alpha=1.0)
        with self.assertRaises(ValueError):
            DDSketch(alpha=-0.1)

    def test_add_single_value(self):
        sketch = DDSketch()
        sketch.add(100.0)
        self.assertEqual(sketch.count, 1)
        self.assertFalse(sketch.is_empty())
        self.assertAlmostEqual(sketch.sum, 100.0, places=10)

    def test_add_multiple_values(self):
        sketch = DDSketch()
        for i in range(1, 101):
            sketch.add(float(i))
        self.assertEqual(sketch.count, 100)
        self.assertAlmostEqual(sketch.sum, 5050.0, places=10)
        self.assertAlmostEqual(sketch.mean, 50.5, places=10)

    def test_add_batch(self):
        sketch = DDSketch()
        values = [float(i) for i in range(1, 101)]
        sketch.add_batch(values)
        self.assertEqual(sketch.count, 100)
        self.assertAlmostEqual(sketch.sum, 5050.0, places=10)


class TestDDSketchQuantiles(unittest.TestCase):
    def setUp(self):
        self.sketch = DDSketch(alpha=0.01)
        for i in range(1, 1001):
            self.sketch.add(float(i))

    def test_median(self):
        median = self.sketch.quantile(0.5)
        self.assertAlmostEqual(median, 500.0, delta=500.0 * 0.01)

    def test_p90(self):
        p90 = self.sketch.quantile(0.9)
        self.assertAlmostEqual(p90, 900.0, delta=900.0 * 0.01)

    def test_p99(self):
        p99 = self.sketch.quantile(0.99)
        self.assertAlmostEqual(p99, 990.0, delta=990.0 * 0.01)

    def test_min_quantile(self):
        p0 = self.sketch.quantile(0.0)
        self.assertAlmostEqual(p0, 1.0, delta=1.0 * 0.01 + 0.1)

    def test_max_quantile(self):
        p100 = self.sketch.quantile(1.0)
        self.assertAlmostEqual(p100, 1000.0, delta=1000.0 * 0.01)

    def test_invalid_quantile(self):
        with self.assertRaises(ValueError):
            self.sketch.quantile(-0.1)
        with self.assertRaises(ValueError):
            self.sketch.quantile(1.1)

    def test_percentiles(self):
        p50, p90, p95, p99 = self.sketch.percentiles()
        self.assertAlmostEqual(p50, 500.0, delta=500.0 * 0.01)
        self.assertAlmostEqual(p90, 900.0, delta=900.0 * 0.01)
        self.assertAlmostEqual(p95, 950.0, delta=950.0 * 0.01)
        self.assertAlmostEqual(p99, 990.0, delta=990.0 * 0.01)


class TestDDSketchMinMax(unittest.TestCase):
    def test_min_max_positive(self):
        sketch = DDSketch()
        sketch.add_batch([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertAlmostEqual(sketch.min, 10.0, delta=10.0 * 0.01)
        self.assertAlmostEqual(sketch.max, 50.0, delta=50.0 * 0.01)

    def test_min_max_with_negatives(self):
        sketch = DDSketch()
        sketch.add_batch([-50.0, -10.0, 0.0, 10.0, 50.0])
        self.assertAlmostEqual(sketch.min, -50.0, delta=50.0 * 0.01)
        self.assertAlmostEqual(sketch.max, 50.0, delta=50.0 * 0.01)


class TestDDSketchMerge(unittest.TestCase):
    def test_merge_two_sketches(self):
        sketch1 = DDSketch()
        sketch2 = DDSketch()

        for i in range(1, 51):
            sketch1.add(float(i))
        for i in range(51, 101):
            sketch2.add(float(i))

        sketch1.merge(sketch2)

        self.assertEqual(sketch1.count, 100)
        self.assertAlmostEqual(sketch1.sum, 5050.0, places=10)
        self.assertAlmostEqual(sketch1.mean, 50.5, places=10)

    def test_merge_incompatible_alpha(self):
        sketch1 = DDSketch(alpha=0.01)
        sketch2 = DDSketch(alpha=0.05)

        sketch1.add(1.0)
        sketch2.add(2.0)

        with self.assertRaises(ValueError):
            sketch1.merge(sketch2)


class TestDDSketchClear(unittest.TestCase):
    def test_clear(self):
        sketch = DDSketch()
        sketch.add_batch([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(sketch.count, 5)

        sketch.clear()
        self.assertEqual(sketch.count, 0)
        self.assertTrue(sketch.is_empty())
        self.assertAlmostEqual(sketch.sum, 0.0, places=10)


class TestDDSketchLen(unittest.TestCase):
    def test_len_protocol(self):
        sketch = DDSketch()
        self.assertEqual(len(sketch), 0)

        sketch.add_batch([1.0, 2.0, 3.0])
        self.assertEqual(len(sketch), 3)


class TestDDSketchRepr(unittest.TestCase):
    def test_repr(self):
        sketch = DDSketch()
        sketch.add(100.0)
        repr_str = repr(sketch)
        self.assertIn("DDSketch", repr_str)
        self.assertIn("count=1", repr_str)

    def test_str(self):
        sketch = DDSketch()
        sketch.add(100.0)
        str_repr = str(sketch)
        self.assertIn("DDSketch", str_repr)


class TestDDSketchEmptyBehavior(unittest.TestCase):
    def test_empty_quantile(self):
        sketch = DDSketch()
        result = sketch.quantile(0.5)
        self.assertEqual(result, 0.0)

    def test_empty_percentiles(self):
        sketch = DDSketch()
        result = sketch.percentiles()
        self.assertIsNone(result)

    def test_empty_mean(self):
        sketch = DDSketch()
        self.assertEqual(sketch.mean, 0.0)


class TestDDSketchAccuracy(unittest.TestCase):
    def test_relative_accuracy_guarantee(self):
        alpha = 0.01
        sketch = DDSketch(alpha=alpha)

        values = [float(i) for i in range(1, 10001)]
        sketch.add_batch(values)

        for q in [0.5, 0.9, 0.95, 0.99]:
            estimated = sketch.quantile(q)
            actual = values[int(q * (len(values) - 1))]
            relative_error = abs(estimated - actual) / actual
            self.assertLessEqual(relative_error, alpha,
                f"Relative error {relative_error} exceeds alpha {alpha} at quantile {q}")


if __name__ == "__main__":
    unittest.main()
