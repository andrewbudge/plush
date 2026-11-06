//! Alignment-free protein sketch: binned MinHash over 6-mer windows of the
//! sequence, partitioned by the top 7 bits of an avalanche-mixed FNV-1a hash.
//! Each of the 128 bins keeps the per-bin minimum hash; empty bins are filled
//! by deterministic probe-copy so that two sketches of the same input are
//! byte-identical and absent bins never match spuriously.
//!
//! On disk: 64 bytes (4 bits per bin). LSH: 16 bands x 8 bins -> 16 u32 band
//! keys per sketch. Verification: nibble-wise compare of two 64-byte sketches.
//! Similarity: fraction of matching bins with a 1/16 empty-bin correction.

pub const K: usize = 6;
pub const BINS: usize = 128;
pub const BYTES: usize = BINS / 2;
pub const BANDS: usize = 16;
const BAND_BYTES: usize = BYTES / BANDS;

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

pub fn sketch(seq: &[u8]) -> [u8; BYTES] {
    let mut mins = [u64::MAX; BINS];
    if seq.len() >= K {
        for w in seq.windows(K) {
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            for &b in w {
                h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
            }
            let h = mix(h);
            let bin = (h >> 57) as usize;
            if h < mins[bin] {
                mins[bin] = h;
            }
        }
    }
    let snapshot = mins;
    for i in 0..BINS {
        if snapshot[i] != u64::MAX {
            continue;
        }
        for attempt in 0..1024u64 {
            let j = (mix(((i as u64) << 32) | attempt) % BINS as u64) as usize;
            if snapshot[j] != u64::MAX {
                mins[i] = snapshot[j];
                break;
            }
        }
    }
    let mut out = [0u8; BYTES];
    for (i, &m) in mins.iter().enumerate() {
        out[i / 2] |= ((m & 0xF) as u8) << (4 * (i % 2));
    }
    out
}

pub fn band_keys(sk: &[u8]) -> [u32; BANDS] {
    let mut out = [0u32; BANDS];
    for (j, o) in out.iter_mut().enumerate() {
        let k = u32::from_le_bytes(sk[j * BAND_BYTES..(j + 1) * BAND_BYTES].try_into().unwrap());
        *o = k ^ (j as u32).wrapping_mul(0x9E37_79B1);
    }
    out
}

pub fn matching_bins(a: &[u8], b: &[u8]) -> u32 {
    let mut m = 0u32;
    for (ca, cb) in a.chunks_exact(8).zip(b.chunks_exact(8)) {
        let x = u64::from_le_bytes(ca.try_into().unwrap()) ^ u64::from_le_bytes(cb.try_into().unwrap());
        let t = (x | (x >> 1) | (x >> 2) | (x >> 3)) & 0x1111_1111_1111_1111;
        m += 16 - t.count_ones();
    }
    m
}

pub fn estimate_pct(matching: u32) -> u32 {
    let frac = matching as f64 / BINS as f64;
    let corrected = ((frac - 1.0 / 16.0) / (15.0 / 16.0)).max(0.0);
    (corrected * 100.0).round() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    const REF: &[u8] = b"MAENQYYGTGRRKSSAARVFIKPGNGKIVINQRSLEQYFGRETARMVVRQPLELVDMVEKLDLYITVKGGGISGQAGAIRHGITRALMEYDESLRSELRKAGFVTRDARQVERKKVGLRKARRRPQFSKR";

    #[test]
    fn identical_sequences_match_completely() {
        let a = sketch(REF);
        let b = sketch(REF);
        assert_eq!(a, b, "sketch must be deterministic");
        assert_eq!(matching_bins(&a, &b), BINS as u32);
        assert_eq!(estimate_pct(BINS as u32), 100);
    }

    #[test]
    fn single_substitution_stays_high() {
        let mut variant = REF.to_vec();
        let mid = variant.len() / 2;
        variant[mid] = if variant[mid] == b'A' { b'G' } else { b'A' };
        let est = estimate_pct(matching_bins(&sketch(REF), &sketch(&variant)));
        assert!(est >= 90, "one substitution should score >= 90, got {est}");
    }

    #[test]
    fn shuffled_sequence_scores_low() {
        let mut shuffled = REF.to_vec();
        shuffled.reverse();
        let est = estimate_pct(matching_bins(&sketch(REF), &sketch(&shuffled)));
        assert!(est <= 20, "reversed sequence should score <= 20, got {est}");
    }

    #[test]
    fn band_keys_differ_per_band() {
        let keys = band_keys(&sketch(REF));
        for i in 0..BANDS {
            for j in i + 1..BANDS {
                assert_ne!(keys[i], keys[j], "band keys should differ (i={i}, j={j})");
            }
        }
    }

    #[test]
    fn estimate_pct_bounds() {
        assert_eq!(estimate_pct(0), 0);
        assert_eq!(estimate_pct(BINS as u32), 100);
    }

    #[test]
    fn short_input_is_all_ones() {
        // Fewer than K residues yields no windows and leaves every bin empty;
        // the packing degenerates to 0xFF. The CLI enforces a floor of K.
        assert_eq!(sketch(b"MAE"), [0xFF; BYTES]);
    }
}
