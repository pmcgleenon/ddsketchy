const CHUNK_SIZE: i32 = 128;

fn div_ceil(dividend: i32, divisor: i32) -> i32 {
    (dividend + divisor - 1) / divisor
}

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Store {
    bins: Vec<u64>,
    count: u64,
    min_key: i32,
    max_key: i32,
    offset: i32,
    bin_limit: usize,
    is_collapsed: bool,
}

impl Store {
    pub fn new(bin_limit: usize) -> Self {
        Store {
            bins: Vec::new(),
            count: 0,
            min_key: i32::MAX,
            max_key: i32::MIN,
            offset: 0,
            bin_limit,
            is_collapsed: false,
        }
    }

    /// Return the number of bins.
    pub fn length(&self) -> i32 {
        self.bins.len() as i32
    }

    pub fn is_empty(&self) -> bool {
        self.bins.is_empty()
    }

    #[inline]
    pub fn add(&mut self, key: i32) {
        let idx = self.get_index(key);
        self.bins[idx] += 1;
        self.count += 1;
    }

    #[inline]
    fn get_index(&mut self, key: i32) -> usize {
        if key < self.min_key {
            if self.is_collapsed {
                return 0;
            }

            self.extend_range(key, None);
            if self.is_collapsed {
                return 0;
            }
        } else if key > self.max_key {
            self.extend_range(key, None);
        }

        (key - self.offset) as usize
    }

    #[cold]
    fn extend_range(&mut self, key: i32, second_key: Option<i32>) {
        let second_key = second_key.unwrap_or(key);
        let new_min_key = i32::min(key, i32::min(second_key, self.min_key));
        let new_max_key = i32::max(key, i32::max(second_key, self.max_key));

        if self.is_empty() {
            let new_len = self.get_new_length(new_min_key, new_max_key);
            self.bins.resize(new_len, 0);
            self.offset = new_min_key;
            self.adjust(new_min_key, new_max_key);
        } else if new_min_key >= self.min_key && new_max_key < self.offset + self.length() {
            self.min_key = new_min_key;
            self.max_key = new_max_key;
        } else {
            // Grow bins
            let new_length = self.get_new_length(new_min_key, new_max_key);
            if new_length > self.length() as usize {
                self.bins.resize(new_length, 0);
            }
            self.adjust(new_min_key, new_max_key);
        }
    }

    fn get_new_length(&self, new_min_key: i32, new_max_key: i32) -> usize {
        let desired_length = new_max_key - new_min_key + 1;
        usize::min(
            (CHUNK_SIZE * div_ceil(desired_length, CHUNK_SIZE)) as usize,
            self.bin_limit,
        )
    }

    fn adjust(&mut self, new_min_key: i32, new_max_key: i32) {
        if new_max_key - new_min_key + 1 > self.length() {
            let new_min_key = new_max_key - self.length() + 1;

            if new_min_key >= self.max_key {
                // Put everything in the first bin.
                self.offset = new_min_key;
                self.min_key = new_min_key;
                self.bins.fill(0);
                self.bins[0] = self.count;
            } else {
                let shift = self.offset - new_min_key;
                if shift < 0 {
                    let collapse_start_index = (self.min_key - self.offset) as usize;
                    let collapse_end_index = (new_min_key - self.offset) as usize;
                    let collapsed_count: u64 = self.bins[collapse_start_index..collapse_end_index]
                        .iter()
                        .sum();
                    // Same-length replacement: just fill with zeros (avoids splice iterator overhead)
                    self.bins[collapse_start_index..collapse_end_index].fill(0);
                    self.bins[collapse_end_index] += collapsed_count;
                }
                self.min_key = new_min_key;
                self.shift_bins(shift);
            }

            self.max_key = new_max_key;
            self.is_collapsed = true;
        } else {
            self.center_bins(new_min_key, new_max_key);
            self.min_key = new_min_key;
            self.max_key = new_max_key;
        }
    }

    fn shift_bins(&mut self, shift: i32) {
        let len = self.bins.len();
        if shift > 0 {
            let s = shift as usize;
            if s < len {
                // copy_within + fill: 2 memops instead of rotate's 3
                self.bins.copy_within(0..len - s, s);
                self.bins[..s].fill(0);
            } else {
                self.bins.fill(0);
            }
        } else {
            let s = shift.unsigned_abs() as usize;
            if s < len {
                self.bins.copy_within(s.., 0);
                self.bins[len - s..].fill(0);
            } else {
                self.bins.fill(0);
            }
        }

        self.offset -= shift;
    }

    fn center_bins(&mut self, new_min_key: i32, new_max_key: i32) {
        let middle_key = new_min_key + (new_max_key - new_min_key + 1) / 2;
        let shift = self.offset + self.length() / 2 - middle_key;
        self.shift_bins(shift)
    }

    /// Finds the key of the bucket containing the `rank`-th smallest
    /// element (0-indexed). Returns `self.max_key` if `rank` is out of
    /// range (including on an empty store, where `count == 0` makes every
    /// `rank` out of range).
    ///
    /// Scans from *both* ends of the bin array, returning as soon as either
    /// side's cumulative count crosses its target (`rank` from the front,
    /// `reversed_rank = count - 1 - rank` from the back). Both directions
    /// land on the same bucket for a given rank, so this is a pure cost
    /// optimization, not a behavior change.
    ///
    /// If `bins[0]` alone covers `rank` it returns at once; that is where a
    /// collapsed store keeps all its folded low values.
    ///
    /// The scan first advances each end a whole block of `BLOCK` bins at a
    /// time while the block's sum shows the target is not inside it. Those
    /// block sums have no early exit, so they vectorise. The scalar
    /// two-pointer loop then finishes inside the remaining range.
    ///
    /// Racing both ends, rather than picking a direction by comparing `rank`
    /// to `count`, matters because rank-space and bin-index-space are not
    /// proportional: values uniform by value concentrate most of their rank
    /// mass in a few bins near the top of the log-spaced array, so a
    /// rank-based guess can pick the much longer direction.
    ///
    /// See `tests::key_at_rank_bidirectional_matches_forward_scan` and
    /// `tests::key_at_rank_matches_forward_scan_dense_spiky_and_ramp`, which
    /// check this against a forward-only reference scan.
    #[inline]
    pub fn key_at_rank(&self, rank: u64) -> i32 {
        if rank >= self.count {
            return self.max_key;
        }
        let bins = self.bins.as_slice();
        // A store that hit its bin limit folds every lower key into bins[0],
        // so all low quantiles land there. Answer those without the block
        // sums, which would otherwise cost more than main's one-step scan.
        if bins[0] > rank {
            return self.offset;
        }
        let reversed_rank = self.count - 1 - rank;
        const BLOCK: usize = 16;
        let mut forward_sum = 0u64;
        let mut backward_sum = 0u64;
        let mut lo = 0usize;
        let mut hi = bins.len(); // exclusive

        // Blocked phase: from each end in turn, skip a whole block when its
        // sum shows the target cannot be inside it. The block sums have no
        // early exit, so they vectorise.
        while hi - lo >= 2 * BLOCK {
            let f: u64 = bins[lo..lo + BLOCK].iter().sum();
            if forward_sum + f > rank {
                break;
            }
            forward_sum += f;
            lo += BLOCK;
            let b: u64 = bins[hi - BLOCK..hi].iter().sum();
            if backward_sum + b > reversed_rank {
                break;
            }
            backward_sum += b;
            hi -= BLOCK;
        }

        // Scalar phase over bins[lo..=hi]; the target is inside this range.
        let mut hi = hi - 1;
        loop {
            forward_sum += bins[lo];
            if forward_sum > rank {
                return lo as i32 + self.offset;
            }
            if lo == hi {
                break;
            }
            backward_sum += bins[hi];
            if backward_sum > reversed_rank {
                return hi as i32 + self.offset;
            }
            lo += 1;
            hi -= 1;
        }

        self.max_key
    }

    pub fn count(&self) -> u64 {
        self.count
    }

    pub fn merge(&mut self, other: &Store) {
        if other.count == 0 {
            return;
        }

        if self.count == 0 {
            self.copy(other);
            return;
        }

        if other.min_key < self.min_key || other.max_key > self.max_key {
            self.extend_range(other.min_key, Some(other.max_key));
        }

        let collapse_start_index = other.min_key - other.offset;
        let mut collapse_end_index = i32::min(self.min_key, other.max_key + 1) - other.offset;
        if collapse_end_index > collapse_start_index {
            let collapsed_count: u64 = other.bins
                [collapse_start_index as usize..collapse_end_index as usize]
                .iter()
                .sum();
            self.bins[0] += collapsed_count;
        } else {
            collapse_end_index = collapse_start_index;
        }

        // Use slice zip instead of per-element indexing — eliminates bounds checks
        // and lets LLVM auto-vectorize the addition loop
        let start_key = collapse_end_index + other.offset;
        let len = (other.max_key + 1 - start_key) as usize;
        if len > 0 {
            let self_start = (start_key - self.offset) as usize;
            let other_start = (start_key - other.offset) as usize;
            let dst = &mut self.bins[self_start..self_start + len];
            let src = &other.bins[other_start..other_start + len];
            for (d, s) in dst.iter_mut().zip(src.iter()) {
                *d += *s;
            }
        }

        self.count += other.count;
    }

    fn copy(&mut self, o: &Store) {
        self.bins = o.bins.clone();
        self.count = o.count;
        self.min_key = o.min_key;
        self.max_key = o.max_key;
        self.offset = o.offset;
        self.bin_limit = o.bin_limit;
        self.is_collapsed = o.is_collapsed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_store() {
        let store = Store::new(2048);
        assert_eq!(store.count(), 0);
        assert!(store.is_empty());
        assert_eq!(store.length(), 0);
    }

    #[test]
    fn test_simple_store() {
        let mut s = Store::new(2048);

        for i in 0..100 {
            s.add(i);
        }

        assert_eq!(s.count(), 100);
        assert!(!s.is_empty());
        assert_eq!(s.key_at_rank(0), 0);
        assert_eq!(s.key_at_rank(99), 99);
    }

    #[test]
    fn test_negative_keys() {
        let mut s = Store::new(2048);

        // Add negative keys (these represent negative values in the negative store)
        for i in -50..50 {
            s.add(i);
        }

        assert_eq!(s.count(), 100);
        assert_eq!(s.key_at_rank(0), -50);
        assert_eq!(s.key_at_rank(99), 49);
    }

    #[test]
    fn test_merge_stores() {
        let mut s1 = Store::new(2048);
        let mut s2 = Store::new(2048);

        // Add different ranges to each store
        for i in 0..50 {
            s1.add(i);
        }
        for i in 50..100 {
            s2.add(i);
        }

        s1.merge(&s2);
        assert_eq!(s1.count(), 100);
        assert_eq!(s2.count(), 50); // s2 unchanged

        // Verify merged range
        assert_eq!(s1.key_at_rank(0), 0);
        assert_eq!(s1.key_at_rank(99), 99);
    }

    /// Reference implementation: always scans forward from the start,
    /// never choosing a direction. This is exactly what `key_at_rank` did
    /// before the bidirectional-scan optimization.
    fn key_at_rank_forward_reference(store: &Store, rank: u64) -> i32 {
        let mut n = 0u64;
        for (i, bin) in store.bins.iter().enumerate() {
            n += *bin;
            if n > rank {
                return i as i32 + store.offset;
            }
        }
        store.max_key
    }

    #[test]
    fn key_at_rank_bidirectional_matches_forward_scan() {
        // Property: scanning from the nearer end must always land on the
        // same bucket as always scanning forward, for every rank —
        // including out-of-range ranks and an empty store. A simple xorshift
        // RNG avoids pulling `rand` into this unit-test module.
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for empty in [true, false] {
            if empty {
                let store = Store::new(2048);
                for rank in [0u64, 1, 100] {
                    assert_eq!(
                        store.key_at_rank(rank),
                        key_at_rank_forward_reference(&store, rank)
                    );
                }
                continue;
            }

            for _ in 0..200 {
                let mut store = Store::new(4096);
                let num_keys = 1 + (next() % 500) as i32;
                let start_key = -1000 + (next() % 2000) as i32;
                for k in 0..num_keys {
                    let key = start_key + k;
                    // Insert a variable, sometimes-zero count per key so
                    // sparse bins (count 0) are exercised too.
                    let times = next() % 4;
                    for _ in 0..times {
                        store.add(key);
                    }
                }
                if store.count() == 0 {
                    continue;
                }

                // In-range ranks, out-of-range ranks, and the exact
                // boundary (count - 1, count, count + 1).
                let probe_ranks = [
                    0u64,
                    store.count() / 4,
                    store.count() / 2,
                    (3 * store.count()) / 4,
                    store.count().saturating_sub(1),
                    store.count(),
                    store.count() + 1,
                    next() % (store.count() + 10),
                ];
                for &rank in &probe_ranks {
                    let expected = key_at_rank_forward_reference(&store, rank);
                    let actual = store.key_at_rank(rank);
                    assert_eq!(
                        actual,
                        expected,
                        "rank={rank} count={} start_key={start_key} num_keys={num_keys}",
                        store.count()
                    );
                }
            }
        }
    }

    #[test]
    fn key_at_rank_matches_forward_scan_dense_spiky_and_ramp() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..300 {
            let mut store = Store::new(4096);
            let num_keys = 1 + (next() % 3000) as i32;
            let start_key = -1500 + (next() % 3000) as i32;
            let mode = next() % 3;
            for k in 0..num_keys {
                let key = start_key + k;
                let times = match mode {
                    0 => next() % 3,
                    1 => {
                        if k % 97 == 0 {
                            1000
                        } else {
                            0
                        }
                    }
                    _ => (k as u64 * 7) % 50,
                };
                for _ in 0..times {
                    store.add(key);
                }
            }
            if store.count() == 0 {
                continue;
            }
            let count = store.count();
            let step = (count / 200).max(1);
            let mut rank = 0;
            while rank < count + 2 {
                assert_eq!(
                    store.key_at_rank(rank),
                    key_at_rank_forward_reference(&store, rank),
                    "rank={rank} count={count} mode={mode}"
                );
                rank += step;
            }
        }
    }

    #[test]
    fn key_at_rank_matches_forward_scan_on_collapsed_store() {
        // A small bin limit forces collapsing, so bins[0] holds folded low
        // keys and the first-bin fast path is exercised.
        for limit in [128usize, 256] {
            let mut store = Store::new(limit);
            for k in -2000..2000 {
                store.add(k);
                if k % 3 == 0 {
                    store.add(k);
                }
            }
            assert!(store.is_collapsed);
            let count = store.count();
            for rank in (0..count + 2).step_by(7) {
                assert_eq!(
                    store.key_at_rank(rank),
                    key_at_rank_forward_reference(&store, rank),
                    "limit={limit} rank={rank} count={count}"
                );
            }
        }
    }
}
