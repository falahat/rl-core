//! `Observation` — the learner's raw real-valued input vector.
//!
//! An `Observation` is a **fixed-width real-valued vector**: the raw
//! signal the [`ValueFunction`](crate::ValueFunction) learns over.
//! Quantisation happens *inside* the tile coder
//! ([`TileCoder`](crate::TileCoder)), never here — so the learner
//! generalises over the raw signal instead of colliding on a
//! brutally-compressed key.
//!
//! This container is **generic**: it knows nothing about any particular
//! observation layout. The game-specific block schema (drives, body,
//! emotions, …) is defined entirely by the game's streaming
//! [`Featurize`](crate::Featurize) / [`Encoder`](crate::Encoder) — in the
//! sim, `planner`'s `featurize` — which also fixes the width. Nothing reads
//! this vector by semantic offset; it is written by the encoder and consumed
//! (hashed whole, by block) by the tile coder.

use serde::{Deserialize, Serialize};

/// A fixed-width real-valued snapshot — the input the
/// [`ValueFunction`](crate::ValueFunction) learns over.
///
/// The width invariant (an `Observation` is always exactly the game's
/// schema width) is enforced game-side; this generic container only
/// stores and serves the raw vector. The `Component` derive is gated
/// behind the default-on `bevy` feature so a `--no-default-features`
/// build (WASM widgets, isolated harnesses) drops the `bevy_ecs` dep.
#[cfg_attr(feature = "bevy", derive(bevy_ecs::prelude::Component))]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Observation {
    values: Vec<f32>,
}

impl Default for Observation {
    fn default() -> Self {
        Self::zeros(0)
    }
}

impl Observation {
    /// An all-zero observation of the given width.
    pub fn zeros(width: usize) -> Self {
        Self {
            values: vec![0.0; width],
        }
    }

    /// Build from a full-width vector. The width invariant is enforced
    /// by the game-side checked constructor, not here.
    pub fn from_values(values: Vec<f32>) -> Self {
        Self { values }
    }

    /// The raw vector — what the tile coder consumes.
    pub fn as_slice(&self) -> &[f32] {
        &self.values
    }

    /// Mutable access for `build_observation` to fill blocks in place.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.values
    }

    /// Width — the number of real-valued channels.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Whether the observation has no channels — present for the
    /// `clippy::len_without_is_empty` lint.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// A deterministic `u64` digest of the observation — a coarse
    /// fingerprint of the situation. Each channel is quantised to a
    /// 0.05-wide bucket before an FNV-1a fold, so observations that
    /// differ only by sub-bucket noise digest identically. Used purely
    /// as an opaque grouping key by the attribution ledger
    /// (`UpdatedCell`); it feeds no decision and the learner never
    /// consults it. Pure float/integer arithmetic — run-stable.
    pub fn digest(&self) -> u64 {
        let mut h = crate::hash::FNV_OFFSET_64;
        for &x in &self.values {
            // Quantise to a 0.05 bucket; the resulting integer is the
            // hashed token, so near-equal observations collide.
            let bucket = (x * 20.0).floor() as i64;
            h = crate::hash::fnv1a_mix(h, bucket as u64);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_has_requested_width() {
        assert_eq!(Observation::zeros(199).len(), 199);
        assert!(!Observation::zeros(199).is_empty());
    }

    #[test]
    fn round_trips_through_serde() {
        let mut o = Observation::zeros(199);
        o.as_mut_slice()[3] = 0.7;
        let json = serde_json::to_string(&o).unwrap();
        let back: Observation = serde_json::from_str(&json).unwrap();
        assert_eq!(o, back);
    }

    #[test]
    fn from_values_keeps_the_vector_verbatim() {
        let o = Observation::from_values(vec![0.0; 3]);
        assert_eq!(o.len(), 3);
    }
}
