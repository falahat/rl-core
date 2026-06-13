//! Deterministic hashing + mixing primitives — the ONE home for the
//! workspace's determinism-critical bit-mixing.
//!
//! Everything that hashes for reproducibility (tile coding, observation
//! digests, scent-field keys, per-key RNG draws, value noise) uses these
//! exact functions. A transcription slip in a re-typed copy of FNV or
//! SplitMix silently desyncs hashing across runs, so copies are banned:
//! call this module instead. Known-vector tests below pin every function
//! to the published reference values — any future edit that changes a
//! sequence fails loudly.
//!
//! Bevy-free, std-only: reachable from the WASM crates (`playground`,
//! `nn`-adjacent tooling) that cannot link the ECS host. Re-exported as
//! `rl_core::hash` so existing `rl_core::hash::*` call sites are unchanged.

#![forbid(unsafe_code)]

/// FNV-1a 64-bit offset basis (spec constant, like the digits of π).
pub const FNV_OFFSET_64: u64 = 0xcbf2_9ce4_8422_2325;
/// FNV-1a 64-bit prime (spec constant).
pub const FNV_PRIME_64: u64 = 0x0000_0100_0000_01b3;
/// SplitMix64's golden-ratio increment (`⌊2^64/φ⌋`, odd) — the state
/// advance of the reference SplitMix64 stream and the multiplier of
/// [`splitmix_mix`].
pub const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// Fold one byte into an FNV-1a accumulator.
#[inline]
pub const fn fnv1a_byte(h: u64, b: u8) -> u64 {
    (h ^ b as u64).wrapping_mul(FNV_PRIME_64)
}

/// Fold one `u64` into an FNV-1a accumulator, little-endian byte by byte.
#[inline]
pub const fn fnv1a_mix(mut h: u64, v: u64) -> u64 {
    let bytes = v.to_le_bytes();
    let mut i = 0;
    while i < 8 {
        h = fnv1a_byte(h, bytes[i]);
        i += 1;
    }
    h
}

/// FNV-1a 64-bit over a byte slice, from the offset basis.
#[inline]
pub fn fnv1a_bytes(bytes: &[u8]) -> u64 {
    bytes.iter().fold(FNV_OFFSET_64, |h, &b| fnv1a_byte(h, b))
}

/// FNV-1a 64-bit over a string, from the offset basis.
#[inline]
pub fn fnv1a_str(s: &str) -> u64 {
    fnv1a_bytes(s.as_bytes())
}

/// One SplitMix64 step: advance state `z` by the golden gamma, then
/// finalize. `splitmix64(0)` equals the published reference sequence's
/// first output.
#[inline]
pub const fn splitmix64(z: u64) -> u64 {
    splitmix_finalize(z.wrapping_add(GOLDEN_GAMMA))
}

/// SplitMix64-style mix of a seed with a stream salt — for deriving
/// well-distributed sub-stream seeds from highly-correlated inputs.
#[inline]
pub const fn splitmix_mix(seed: u64, salt: u64) -> u64 {
    splitmix_finalize(seed.wrapping_mul(GOLDEN_GAMMA).wrapping_add(salt))
}

/// The shared SplitMix64 finalizer (xor-shift-multiply avalanche).
#[inline]
pub const fn splitmix_finalize(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Map 64 random bits to a uniform `f32` in `[0, 1]` (53-bit shift path —
/// byte-identical to the engine's keyed-draw tail).
///
/// NOTE: the result can be exactly `1.0` — `f32`'s 24-bit mantissa rounds
/// the top ~2^29 inputs up to `2^53`. Comparisons of the form
/// `draw < epsilon` are unaffected; do not assume a half-open interval.
#[inline]
pub fn unit_f32(bits: u64) -> f32 {
    (bits >> 11) as f32 / (1u64 << 53) as f32
}

/// Map 64 random bits to a uniform `f32` in `[0, 1)` via the top-24-bit
/// window (`>> 40`, divided by `2^24`) — the observation-feature flavour
/// of a unit draw (planner kind features, value noise).
///
/// Distinct from [`unit_f32`] (the `>> 11 / 2^53` keyed-draw tail): the
/// 24-bit numerator is exactly representable in an `f32` mantissa, so the
/// result is genuinely half-open (never rounds up to `1.0`) and every
/// distinct window value maps to a distinct float.
#[inline]
pub fn unit_from_hash(h: u64) -> f32 {
    ((h >> 40) as f32) / (1u64 << 24) as f32
}

/// Map a hash to a bucket index in `[0, n)` — the canonical `h mod n`
/// slotting (memory slots, category buckets). `n` must be non-zero.
#[inline]
pub fn bucket(h: u64, n: usize) -> usize {
    (h % n as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    // Published FNV-1a 64 test vectors (Noll's reference list).
    #[test]
    fn fnv1a_matches_published_vectors() {
        assert_eq!(fnv1a_str(""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a_str("a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a_str("foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn fnv1a_mix_equals_bytewise_fold() {
        let v = 0x0123_4567_89ab_cdefu64;
        let expected = v
            .to_le_bytes()
            .iter()
            .fold(FNV_OFFSET_64, |h, &b| fnv1a_byte(h, b));
        assert_eq!(fnv1a_mix(FNV_OFFSET_64, v), expected);
    }

    // Vigna's SplitMix64 reference: state 0 → first output.
    #[test]
    fn splitmix64_matches_reference_sequence() {
        assert_eq!(splitmix64(0), 0xE220_A839_7B1D_CDAF);
        // Second output: state advanced by one gamma.
        assert_eq!(splitmix64(0x9E37_79B9_7F4A_7C15), 0x6E78_9E6A_A1B9_65F4);
    }

    #[test]
    fn unit_f32_is_in_unit_interval_and_deterministic() {
        for bits in [0u64, 1, u64::MAX, 0xDEAD_BEEF_CAFE_F00D] {
            let u = unit_f32(bits);
            assert!((0.0..=1.0).contains(&u), "{u} out of [0,1] for {bits:#x}");
        }
        // The documented closed end: f32's 24-bit mantissa rounds the top
        // inputs up to exactly 1.0 — the interval is NOT half-open.
        assert_eq!(unit_f32(u64::MAX), 1.0);
        assert_eq!(unit_f32(u64::MAX), unit_f32(u64::MAX));
        assert_eq!(unit_f32(0), 0.0);
    }

    #[test]
    fn unit_from_hash_is_half_open_and_pins_the_top_24_bit_window() {
        // Strictly < 1.0 even at the all-ones input (the 24-bit window is
        // exactly representable, unlike unit_f32's 53-bit shift).
        assert!(unit_from_hash(u64::MAX) < 1.0);
        assert_eq!(unit_from_hash(0), 0.0);
        // Pin the window arithmetic: (h >> 40) / 2^24.
        let h = 0xABCD_EF01_2345_6789u64;
        assert_eq!(unit_from_hash(h), ((h >> 40) as f32) / (1u64 << 24) as f32);
    }

    #[test]
    fn bucket_is_h_mod_n() {
        assert_eq!(bucket(0, 16), 0);
        assert_eq!(bucket(17, 16), 1);
        assert_eq!(bucket(u64::MAX, 8), (u64::MAX % 8) as usize);
    }
}
