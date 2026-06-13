//! Shared ε-greedy policy arithmetic — the ONE home for the argmax scan
//! and the explore/exploit split every ε-greedy action picker uses.
//!
//! Every ε-greedy picker calls these — the playground's `TdMind`,
//! `q_learning`'s `TileCodedQMind`, and the planner's
//! `epsilon_greedy_pick`/`exploit_index`. A transcription slip in a
//! hand-rolled copy silently skews action selection, so copies are banned —
//! call these instead. Each site keeps its OWN salt constants (intentionally
//! distinct draw streams) and its own layering (the planner stacks a
//! tie-pool + recipe priority on top of [`greedy_argmax`]).
//!
//! Bevy-free, std-only — reachable from the WASM crates.

/// Index of the highest-valued action — a linear scan in which the
/// **first** maximum wins ties (the shared tie semantics of every
/// ε-greedy exploit scan in the workspace).
///
/// Values are compared with strict `>` against a `NEG_INFINITY` seed, so
/// a NaN value never wins: NaNs are skipped, matching the planner's
/// `f32::max` fold over candidate scores. If *every* value is NaN, index
/// 0 is returned (the caller's non-finite-score guard handles that
/// degenerate read). On all-finite inputs this is byte-identical to the
/// classic "seed from element 0, take strict improvements" scan.
///
/// `actions` must be non-empty.
pub fn greedy_argmax<A>(actions: &[A], value: impl Fn(&A) -> f32) -> usize {
    debug_assert!(
        !actions.is_empty(),
        "greedy_argmax needs at least one action"
    );
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, a) in actions.iter().enumerate() {
        let v = value(a);
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

/// The ε-greedy explore/exploit split: `Some(index)` of a uniform-random
/// pick over `n` candidates when `draw < epsilon`, `None` to exploit.
///
/// `draw` and `pick_draw` are two unit-interval values from **distinct**
/// salted streams (the explore coin must not correlate with the index
/// pick). Both are plain values, not closures, because every caller's
/// draws are *stateless* keyed hashes — evaluating the pick draw on an
/// exploit step consumes nothing and changes nothing.
///
/// The index map `(pick_draw * n) clamped to n − 1` is the shared
/// uniform-pick formula (the clamp covers a draw that rounds to exactly
/// `n`). `n` must be non-zero.
pub fn epsilon_explore(draw: f32, epsilon: f32, n: usize, pick_draw: f32) -> Option<usize> {
    debug_assert!(n > 0, "epsilon_explore needs at least one candidate");
    if draw < epsilon {
        Some(((pick_draw * n as f32) as usize).min(n.saturating_sub(1)))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_argmax_first_max_wins_ties() {
        let vals = [1.0f32, 3.0, 3.0, 2.0];
        assert_eq!(greedy_argmax(&vals, |&v| v), 1);
    }

    #[test]
    fn greedy_argmax_skips_nan_values() {
        let vals = [f32::NAN, 2.0, 5.0, f32::NAN];
        assert_eq!(greedy_argmax(&vals, |&v| v), 2);
        // All-NaN degenerates to index 0 (caller's non-finite guard owns it).
        let all_nan = [f32::NAN, f32::NAN];
        assert_eq!(greedy_argmax(&all_nan, |&v| v), 0);
    }

    #[test]
    fn greedy_argmax_handles_all_negative_values() {
        let vals = [-3.0f32, -1.0, -2.0];
        assert_eq!(greedy_argmax(&vals, |&v| v), 1);
    }

    #[test]
    fn epsilon_explore_splits_on_the_coin() {
        // draw < ε ⇒ explore with the pick formula; draw ≥ ε ⇒ exploit.
        assert_eq!(epsilon_explore(0.05, 0.1, 4, 0.6), Some(2));
        assert_eq!(epsilon_explore(0.5, 0.1, 4, 0.6), None);
        // ε = 0 never explores; ε = 1 with a sub-1 draw always does.
        assert_eq!(epsilon_explore(0.0, 0.0, 4, 0.0), None);
        assert_eq!(epsilon_explore(0.99, 1.0, 4, 0.0), Some(0));
    }

    #[test]
    fn epsilon_explore_clamps_a_full_scale_pick_draw() {
        // A pick draw of exactly 1.0 (unit_f32 can produce it) maps to n,
        // which the clamp pulls back to the last valid index.
        assert_eq!(epsilon_explore(0.0, 0.5, 3, 1.0), Some(2));
    }
}
