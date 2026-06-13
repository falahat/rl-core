//! `ValueFunction` — the learner abstraction, and `TileCodedValue`, its
//! tile-coding implementation.
//!
//! The trait is the **stable seam** the design calls for: tile coding
//! ships now; a neural-net implementation (small MLP / attention-over-
//! memory) drops in later behind the same trait without touching any
//! caller (`emergent_cognition_rebuild.md` §3.2).
//!
//! `TileCodedValue` is a learned scalar function `f(observation, action)`
//! — a weight per tile-coder feature. `value` sums the active weights;
//! `update` distributes the TD error across them. Because the active
//! tiles of *similar* observations overlap, an update to one observation
//! generalises to its neighbours — the property a lookup table lacks.

use crate::action_id::ActionTemplateId;
use crate::observation::Observation;
use crate::tile_coding::{PreparedObservation, TileCoder};
use serde::{Deserialize, Serialize};

/// A learned scalar function of `(observation, action)`. Seam between
/// the TD algorithm and the value-function representation: a future
/// neural-net Q-function would impl this without changes to callers.
/// [`TileCodedValue`] is the shipped implementation.
///
/// Implementations must be deterministic — identical inputs and identical
/// update history yield identical values across runs.
pub trait ValueFunction: Send + Sync {
    /// Estimated value of taking `action` in situation `obs`.
    fn value(&self, obs: &Observation, action: ActionTemplateId) -> f32;

    /// Move the estimate for `(obs, action)` toward `target` at learning
    /// rate `alpha`. This is one TD / gradient step.
    fn update(&mut self, obs: &Observation, action: ActionTemplateId, target: f32, alpha: f32);
}

/// Tile-coded linear value function: a weight vector indexed by the
/// shared [`TileCoder`]'s feature hash.
///
/// `PartialEq` is **hand-written** to compare only the semantically meaningful
/// state (`coder` + `weights`) and ignore the diagnostic `counts` — so a
/// save/load round trip (which drops `counts`, see below) still compares equal,
/// and "identical update history ⇒ identical value function" holds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileCodedValue {
    coder: TileCoder,
    /// One weight per IHT slot. `len == coder.iht_size()`.
    weights: Vec<f32>,
    /// **Per-slot update counter** — how many TD updates have touched each tile
    /// slot. The instrument behind the per-probe "is THIS state trained?"
    /// question ([`visit_count`](Self::visit_count)): a probe state whose active
    /// tiles were updated only a handful of times has an unreliable learned `Q`,
    /// regardless of the global sample budget. Diagnostic only — it feeds no
    /// decision and is **not** part of the value-function identity.
    ///
    /// `#[serde(skip)]`: excluded from the save payload (it would add
    /// `iht_size · 4 B` ≈ 0.25 MiB/agent for a pure diagnostic, and force a
    /// schema bump). After a load it deserialises empty and is lazily re-sized to
    /// `weights.len()` on the next update — so counts reflect *this run's*
    /// training, which is exactly what the probe wants.
    #[serde(skip)]
    counts: Vec<u32>,
}

impl PartialEq for TileCodedValue {
    /// Identity is `coder` + `weights` only — the diagnostic `counts` is
    /// deliberately excluded (see the type docs).
    fn eq(&self, other: &Self) -> bool {
        self.coder == other.coder && self.weights == other.weights
    }
}

/// A learner's **training-coverage report** — derived purely from the per-slot
/// update counter, so it is self-contained (needs no experiment budget) and
/// **replay-inclusive** (every weight write is counted). The instrument behind
/// the "is this learner under-sampled?" question for ANY learning test, not just
/// nav: a model with thousands of `unique_tiles` but a `mean_updates_per_state`
/// of ~1–2 trained each state only once or twice and cannot have converged,
/// regardless of algorithm or hyper-parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Coverage {
    /// Distinct tile slots that received ≥1 update — the realised state-space size.
    pub unique_tiles: usize,
    /// Total weight-update events across all slots (online + replay).
    pub total_updates: u64,
    /// `total_updates / unique_tiles` — average TD-updates per touched tile. The
    /// headline under-sampling scalar. `0` if nothing was trained.
    ///
    /// **Caveat for replay-heavy credit:** every backward-replay pass over a stored
    /// episode counts here, so under `EpisodeReplay` this reflects replay *volume /
    /// importance-sampling concentration* (millions), not distinct-state coverage —
    /// a huge value means "heavily replayed", not "many distinct experiences". For
    /// the precise "is THIS specific state trained?" question use
    /// [`visit_count`](TileCodedValue::visit_count) (the MIN over one state's tiles).
    pub mean_updates_per_state: f64,
    /// The least-trained touched tile's update count — the worst-case "how
    /// stale is the thinnest part of the learned function?" `0` if nothing trained.
    /// NOTE: the min ALONE is degenerate — under ε-exploration there is always a
    /// just-discovered count-1 tile, so it pins at 1 regardless of difficulty /
    /// episodes. It is still a real signal (the agent is *still meeting fresh
    /// tiles*, so the space isn't saturated), but read it alongside the percentiles
    /// below, which reveal the DISTRIBUTION shape rather than one worst-case number.
    pub min_updates_per_state: u32,
    /// **Per-tile update-count distribution** over the touched tiles (`counts[i]`
    /// for every `i` with `counts[i] > 0`). Separates "a thin fresh tail over a
    /// well-trained bulk" (high p25/median) from "most of the space barely trained"
    /// (low p25/median) — the thing the lone `min` cannot tell you. `0` if nothing trained.
    pub p5_updates_per_state: u32,
    pub p25_updates_per_state: u32,
    pub median_updates_per_state: u32,
    pub p95_updates_per_state: u32,
    pub p99_updates_per_state: u32,
    /// The most-trained touched tile's update count — the top of the distribution.
    pub max_updates_per_state: u32,
    /// Fraction of TOUCHED tiles whose update-count is below the under-training
    /// threshold ([`UNDERTRAINED_THRESHOLD`] = 10) — the share of the realised
    /// state-space that is barely trained, in `[0,1]`. `1.0` when every touched
    /// tile is under-trained (a freshly-started learner), `0.0` once every touched
    /// tile clears the threshold (or nothing is trained). The headline scalar for
    /// "is the learner under-sampled for the task's complexity?" — read alongside
    /// the percentile distribution above (a high p25/median means a well-trained
    /// bulk under a thin fresh tail; a high `undertrained_frac` means most of the
    /// space is barely trained).
    pub undertrained_frac: f64,
    /// **Replay-independent** online TD-decision count (online updates only; replay
    /// passes excluded). `0` at the [`TileCodedValue`] layer, which can't tell
    /// online from replay — filled by `Learner::coverage` from its own counter.
    pub online_updates: u64,
    /// `online_updates / unique_tiles` — the **robust under-sampling gate**: how
    /// many *distinct* training decisions landed per trained state, which a
    /// replay-heavy credit mode cannot inflate (unlike `mean_updates_per_state`).
    /// `0` until filled by `Learner::coverage`.
    pub online_updates_per_state: f64,
}

/// Per-tile update count below which a touched tile is "under-trained" — the
/// threshold behind [`Coverage::undertrained_frac`]. A tile updated fewer than
/// this many times has too thin a sample to have converged its local value.
pub const UNDERTRAINED_THRESHOLD: u32 = 10;

/// Nearest-rank percentile of a SORTED-ascending slice of per-tile update counts
/// (0-indexed). `p` in `[0,100]`; `pctl(s,0)` = min, `pctl(s,100)` = max; empty ⇒ 0.
fn pctl(sorted: &[u32], p: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((p / 100.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

impl TileCodedValue {
    /// A fresh zero-initialised value function over `coder`.
    pub fn new(coder: TileCoder) -> Self {
        Self::with_init(coder, 0.0)
    }

    /// A fresh value function whose every untrained `(state, action)` reads
    /// `init_q`. Implemented as a uniform per-weight offset `init_q / active_count`
    /// so the sum over a query's active tiles is `≈ init_q` regardless of how many
    /// tiles are active. `init_q = 0.0` is the standard zero-init (byte-identical to
    /// [`new`](Self::new)); a **negative** `init_q` makes untried actions pessimistic
    /// (the optimistic-max intervention — see `LearningConfig::init_value`).
    pub fn with_init(coder: TileCoder, init_q: f32) -> Self {
        let n = coder.iht_size() as usize;
        let active = coder.active_count().max(1) as f32;
        let w = init_q / active;
        Self {
            coder,
            weights: vec![w; n],
            counts: vec![0; n],
        }
    }

    /// The feature map this value function is built on.
    pub fn coder(&self) -> &TileCoder {
        &self.coder
    }

    /// Lazily ensure the per-slot `counts` table matches `weights` — it starts
    /// sized in [`new`](Self::new) but deserialises empty (`#[serde(skip)]`), so
    /// the first post-load update restores it. Cheap length check on the hot path.
    fn ensure_counts(&mut self) {
        if self.counts.len() != self.weights.len() {
            self.counts = vec![0; self.weights.len()];
        }
    }

    /// **Per-probe training visits**: the *minimum* update count over the tiles
    /// active for `(obs, action)` — the least-trained tile of that state, the
    /// conservative "how well-sampled is this exact state's learned value?".
    /// `0` if the state's tiles were never updated (or counts not yet sized).
    /// Used by the nav diagnostic to annotate an eat_value/approach probe read
    /// off an under-sampled state as unreliable rather than a credit bug.
    pub fn visit_count(&self, obs: &Observation, action: ActionTemplateId) -> u32 {
        self.visit_count_prepared(&self.coder.prepare(obs), action)
    }

    /// [`visit_count`](Self::visit_count) for an already-[`prepare`](Self::prepare)d
    /// observation.
    pub fn visit_count_prepared(
        &self,
        prepared: &PreparedObservation,
        action: ActionTemplateId,
    ) -> u32 {
        if self.counts.len() != self.weights.len() {
            return 0; // not yet sized (fresh-loaded, no updates since)
        }
        self.coder
            .active_indices(prepared, action)
            .map(|i| self.counts[i as usize])
            .min()
            .unwrap_or(0)
    }

    /// Count of non-zero weights — a "how much has been learned" signal.
    pub fn learned_weight_count(&self) -> usize {
        self.weights.iter().filter(|w| **w != 0.0).count()
    }

    /// This learner's [`Coverage`] report (unique tiles, total updates, mean/min
    /// updates-per-state) from the per-slot update counter. Cheap single scan;
    /// `0`-everything if no updates have run (or counts not yet sized post-load).
    pub fn coverage(&self) -> Coverage {
        // Collect the touched tiles' update counts, then read the DISTRIBUTION off the
        // sorted vector. A once-per-run diagnostic (not a hot path), so the transient
        // Vec + sort over ≤ iht_size slots is fine.
        let mut touched: Vec<u32> = self.counts.iter().copied().filter(|&c| c > 0).collect();
        let unique = touched.len();
        let total: u64 = touched.iter().map(|&c| c as u64).sum();
        // Fraction of touched tiles below the under-training threshold (counted before
        // the sort, order-independent). `0.0` when nothing is trained (no `0/0`).
        let undertrained = touched
            .iter()
            .filter(|&&c| c < UNDERTRAINED_THRESHOLD)
            .count();
        touched.sort_unstable();
        Coverage {
            unique_tiles: unique,
            total_updates: total,
            mean_updates_per_state: if unique > 0 {
                total as f64 / unique as f64
            } else {
                0.0
            },
            // Nearest-rank percentiles of the per-tile update counts (pctl(0)=min,
            // pctl(100)=max) — the DISTRIBUTION, not just the degenerate min.
            min_updates_per_state: pctl(&touched, 0.0),
            p5_updates_per_state: pctl(&touched, 5.0),
            p25_updates_per_state: pctl(&touched, 25.0),
            median_updates_per_state: pctl(&touched, 50.0),
            p95_updates_per_state: pctl(&touched, 95.0),
            p99_updates_per_state: pctl(&touched, 99.0),
            max_updates_per_state: pctl(&touched, 100.0),
            undertrained_frac: if unique > 0 {
                undertrained as f64 / unique as f64
            } else {
                0.0
            },
            // Online-vs-replay is invisible at this layer; Learner::coverage fills these.
            online_updates: 0,
            online_updates_per_state: 0.0,
        }
    }

    /// **IHT occupancy** of the trained value table: `(used_slots, iht_size)` where
    /// `used_slots` is the number of distinct hash slots that received ≥1 update and
    /// `iht_size` is the bounded hash-space size. `used_slots / iht_size` is the
    /// fraction of the table actually populated by training — high occupancy means
    /// heavy cross-state aliasing (distinct states forced to share weights through
    /// the naive `hash % iht_size`), which corrupts the learned value. This reads the
    /// REAL trained counts (the on-distribution occupancy), the callable form of the
    /// print-only occupancy diagnostic. `used_slots == unique_tiles`; exposed as a
    /// pair so a caller gets the denominator without a separate `coder()` call.
    pub fn iht_occupancy(&self) -> (usize, u32) {
        let used = self.counts.iter().filter(|&&c| c > 0).count();
        (used, self.coder.iht_size())
    }

    /// Hash an observation once for reuse across many action queries —
    /// see [`TileCoder::prepare`].
    pub fn prepare(&self, obs: &Observation) -> PreparedObservation {
        self.coder.prepare(obs)
    }

    /// [`value`](ValueFunction::value) for an observation already hashed
    /// by [`prepare`](Self::prepare). The per-observation hashing is
    /// skipped — only the action's single hash mix is done.
    pub fn value_prepared(&self, prepared: &PreparedObservation, action: ActionTemplateId) -> f32 {
        self.coder
            .active_indices(prepared, action)
            .map(|i| self.weights[i as usize])
            .sum()
    }

    /// [`update`](ValueFunction::update) for an observation already
    /// hashed by [`prepare`](Self::prepare).
    pub fn update_prepared(
        &mut self,
        prepared: &PreparedObservation,
        action: ActionTemplateId,
        target: f32,
        alpha: f32,
    ) {
        let current: f32 = self
            .coder
            .active_indices(prepared, action)
            .map(|i| self.weights[i as usize])
            .sum();
        // The TD error is spread across the active features, so the
        // step size is independent of the active-feature count — the
        // standard tile-coding correction.
        let n = self.coder.active_count().max(1);
        let step = alpha / n as f32 * (target - current);
        self.ensure_counts();
        for i in self.coder.active_indices(prepared, action) {
            self.weights[i as usize] += step;
            self.counts[i as usize] = self.counts[i as usize].saturating_add(1);
        }
    }
}

/// Per-agent **eligibility trace** over the tile-coded feature space —
/// the companion state TD(λ) maintains alongside the value function.
///
/// Each `(state, action)` the agent visits marks its active tile features
/// "eligible"; eligibility decays by `λ·γ` every *decision step* (see the
/// decision-level note on [`TileCodedValue::td_lambda_update`] for why the
/// per-step factor is `λ·γ^n`, not `(λγ)^n`, when a decision spans `n`
/// ticks). When a reward
/// arrives its TD error is scattered across *all* still-eligible features,
/// so the credit reaches the whole recent trajectory in one update — not
/// just the last transition. This is what lets a multi-step goal
/// (navigate N cells → reach food → relief) propagate back to the early
/// navigation steps within a bounded horizon, where one-step TD cannot.
///
/// Representation: a **sparse, sorted** `Vec<(feature, eligibility)>`.
/// Sorted-by-feature so iteration (hence the weight scatter) is
/// deterministic — no `HashMap`. Entries that decay below a floor are
/// pruned, so the trace stays bounded by the recent active set. A
/// *replacing* trace (re-marking a feature sets it to 1.0) — stable for
/// tile coding, where features are binary.
#[cfg_attr(feature = "bevy", derive(bevy_ecs::prelude::Component))]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct EligibilityTrace {
    /// `(feature index, eligibility)`, kept sorted ascending by feature
    /// index for deterministic iteration.
    entries: Vec<(u32, f32)>,
}

/// The running average-reward estimate `R̄` for a *differential*
/// (average-reward) TD rule — the discount-free alternative to `γ` (see the
/// textbook, Ch. 4 §2.3). Per-agent, because each agent experiences its own
/// reward stream, and it **persists across episodes** within a run (it
/// estimates the *long-run* rate, not an episodic quantity — so unlike the
/// eligibility trace it is not reset at an episode boundary). Threaded into
/// the update rule exactly like [`EligibilityTrace`]; one-step and
/// discounted rules ignore it.
#[cfg_attr(feature = "bevy", derive(bevy_ecs::prelude::Component))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
pub struct AverageReward {
    /// `R̄` — the estimated long-run reward rate (tracked per decision).
    pub rate: f32,
}

impl EligibilityTrace {
    /// A fresh, empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Decay every entry by `factor` (the `λ·γ` step discount), then drop
    /// entries whose eligibility has fallen below `floor` so the trace
    /// stays sparse and bounded.
    pub fn decay_and_prune(&mut self, factor: f32, floor: f32) {
        for e in &mut self.entries {
            e.1 *= factor;
        }
        self.entries.retain(|(_, e)| e.abs() >= floor);
    }

    /// Mark `feature` fully eligible (replacing trace: set to 1.0),
    /// preserving sorted order.
    pub fn mark(&mut self, feature: u32) {
        match self.entries.binary_search_by_key(&feature, |(f, _)| *f) {
            Ok(i) => self.entries[i].1 = 1.0,
            Err(i) => self.entries.insert(i, (feature, 1.0)),
        }
    }

    /// Eligible `(feature, eligibility)` pairs, ascending by feature.
    pub fn iter(&self) -> impl Iterator<Item = (u32, f32)> + '_ {
        self.entries.iter().copied()
    }

    /// Number of eligible features.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True when nothing is eligible.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Reset to empty (e.g. at an episode boundary).
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl TileCodedValue {
    /// One TD(λ) weight update. Decays `trace` by `decay`, marks the active
    /// features of `(obs, action)` eligible, then scatters `α·δ` across
    /// **every** eligible feature — normalised by the active-tile count
    /// exactly as [`update_prepared`](Self::update_prepared), so at `λ = 0`
    /// (trace = only the current active features) this reduces precisely to
    /// the one-step update. `floor` prunes spent eligibility.
    ///
    /// **Decision-level decay (`λ·γ^n`, by design — not `(λγ)^n`).** The
    /// "step" of this learner is a *decision* (commit→commit), not a
    /// primitive tick: the trace is touched once per decision, and a decision
    /// may span `n` ticks (cadence-gating + the planner's decision-record
    /// lifecycle). The caller therefore passes `decay = λ·γ^n`, where `λ` is the
    /// per-*decision* trace decay and `γ^n` is the realised elapsed-time
    /// discount for that decision (the same `γ^n` the bootstrap carries). At
    /// `n = 1` this is exactly the textbook `γλ`; for longer decisions it
    /// keeps prior decisions eligible decayed by *one* `λ` plus physical
    /// time-discount — which is the intended SMDP semantics. `(λγ)^n` would
    /// instead model `λ` as a *per-tick* decay, which this learner never
    /// applies (it does not update the trace per tick).
    #[allow(clippy::too_many_arguments)]
    pub fn td_lambda_update(
        &mut self,
        prepared: &PreparedObservation,
        action: ActionTemplateId,
        delta: f32,
        alpha: f32,
        decay: f32,
        trace: &mut EligibilityTrace,
        floor: f32,
    ) {
        trace.decay_and_prune(decay, floor);
        self.ensure_counts();
        for f in self.coder.active_indices(prepared, action) {
            trace.mark(f);
            // Count a "visit" against the CURRENT state's active tiles (not the
            // whole decayed trace) — `visit_count` answers "how often was THIS
            // state the one being updated?".
            self.counts[f as usize] = self.counts[f as usize].saturating_add(1);
        }
        // Same `α/n` normalisation as the one-step path: the per-feature
        // step is independent of the active-feature count.
        let n = self.coder.active_count().max(1) as f32;
        let step = alpha / n * delta;
        for (f, e) in trace.iter() {
            self.weights[f as usize] += step * e;
        }
    }
}

impl ValueFunction for TileCodedValue {
    fn value(&self, obs: &Observation, action: ActionTemplateId) -> f32 {
        self.value_prepared(&self.coder.prepare(obs), action)
    }

    fn update(&mut self, obs: &Observation, action: ActionTemplateId, target: f32, alpha: f32) {
        let prepared = self.coder.prepare(obs);
        self.update_prepared(&prepared, action, target, alpha);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test observation width — a representative game width (the sim's
    /// `agency::planner::featurize` output), fixed here so the generic crate stays
    /// game-agnostic.
    const WIDTH: usize = 199;

    fn value_fn() -> TileCodedValue {
        TileCodedValue::new(TileCoder::uniform(16, 0.25, WIDTH, 1 << 16))
    }

    fn zeros() -> Observation {
        Observation::zeros(WIDTH)
    }

    fn obs_with(dim: usize, v: f32) -> Observation {
        let mut o = zeros();
        o.as_mut_slice()[dim] = v;
        o
    }

    #[test]
    fn fresh_value_is_zero() {
        let vf = value_fn();
        assert_eq!(vf.value(&zeros(), ActionTemplateId(0)), 0.0);
    }

    #[test]
    fn update_moves_the_value_toward_target() {
        let mut vf = value_fn();
        let o = obs_with(3, 0.7);
        let a = ActionTemplateId(1);
        for _ in 0..50 {
            vf.update(&o, a, 1.0, 0.2);
        }
        let v = vf.value(&o, a);
        assert!(
            (v - 1.0).abs() < 0.05,
            "value should converge to target, got {v}"
        );
    }

    #[test]
    fn learning_generalises_to_a_nearby_observation() {
        // Train one observation; a sub-tile-width-nearby observation
        // (never trained) inherits a non-trivial fraction of the value.
        // This is the whole point of tile coding vs a lookup table.
        let mut vf = value_fn();
        let a = ActionTemplateId(0);
        let trained = obs_with(0, 0.50);
        for _ in 0..50 {
            vf.update(&trained, a, 1.0, 0.2);
        }
        let neighbour = obs_with(0, 0.52); // 0.02 << 0.25 tile width
        let v = vf.value(&neighbour, a);
        assert!(
            v > 0.3,
            "a nearby untrained observation must inherit value, got {v}"
        );
    }

    #[test]
    fn learning_does_not_leak_to_a_distant_observation() {
        let mut vf = value_fn();
        let a = ActionTemplateId(0);
        let trained = obs_with(0, 0.50);
        for _ in 0..50 {
            vf.update(&trained, a, 1.0, 0.2);
        }
        let distant = obs_with(0, 90.0);
        let v = vf.value(&distant, a);
        assert!(
            v.abs() < 0.05,
            "a distant observation must not inherit value, got {v}"
        );
    }

    #[test]
    fn lambda_zero_matches_one_step_update() {
        // At λ=0 the trace holds only the current active features, so a
        // TD(λ) update must move the value identically to update_prepared.
        let o = obs_with(4, 0.6);
        let a = ActionTemplateId(1);
        let mut one_step = value_fn();
        let mut lambda = value_fn();
        let mut trace = EligibilityTrace::new();
        for _ in 0..20 {
            let prep = one_step.prepare(&o);
            one_step.update_prepared(&prep, a, 1.0, 0.2); // target=1.0
                                                          // Match: δ = target − Q_now for the λ-path.
            let prep_l = lambda.prepare(&o);
            let q = lambda.value_prepared(&prep_l, a);
            lambda.td_lambda_update(&prep_l, a, 1.0 - q, 0.2, 0.0, &mut trace, 1e-4);
        }
        let v1 = one_step.value(&o, a);
        let vl = lambda.value(&o, a);
        assert!(
            (v1 - vl).abs() < 1e-5,
            "λ=0 must equal one-step: {v1} vs {vl}"
        );
    }

    #[test]
    fn trace_propagates_credit_to_an_earlier_state() {
        // THE multi-step-credit property navigation needs. Visit (o_early,
        // a) with no reward, then (o_late, a) with a positive TD error. A
        // pure one-step update at o_late would leave o_early untouched; the
        // eligibility trace must carry credit back so o_early's value rises.
        let mut vf = value_fn();
        let a = ActionTemplateId(0);
        let o_early = obs_with(0, 0.0);
        let o_late = obs_with(0, 100.0); // far apart ⇒ disjoint tiles
        let mut trace = EligibilityTrace::new();
        let decay = 0.9 * 0.95; // λ·γ

        // Step 1: at o_early, nothing learned yet (δ = 0) — only marks it.
        let p_early = vf.prepare(&o_early);
        vf.td_lambda_update(&p_early, a, 0.0, 0.2, decay, &mut trace, 1e-4);
        assert_eq!(vf.value(&o_early, a), 0.0, "δ=0 must not move any weight");

        // Step 2: at o_late, a positive TD error arrives.
        let p_late = vf.prepare(&o_late);
        vf.td_lambda_update(&p_late, a, 1.0, 0.2, decay, &mut trace, 1e-4);

        let early = vf.value(&o_early, a);
        let late = vf.value(&o_late, a);
        assert!(
            late > 0.0,
            "the rewarded state must gain value (got {late})"
        );
        assert!(
            early > 0.0,
            "the EARLIER state must inherit credit through the trace (got \
             {early}) — this is the multi-step propagation one-step TD lacks",
        );
        // The earlier state gets the discounted share (decay·late-ish), so
        // strictly less than the directly-rewarded late state.
        assert!(
            early < late,
            "earlier credit must be discounted below late ({early} !< {late})"
        );
    }

    #[test]
    fn determinism_identical_update_history() {
        fn run() -> f32 {
            let mut vf = value_fn();
            let o = obs_with(7, 0.4);
            for i in 0..20 {
                vf.update(&o, ActionTemplateId(i % 3), 0.8, 0.1);
            }
            vf.value(&o, ActionTemplateId(0))
        }
        assert_eq!(run(), run());
    }

    #[test]
    fn visit_count_tracks_per_state_updates() {
        // The per-probe "is THIS state trained?" instrument: visit_count is the
        // MIN update count over a state's active tiles, 0 for an untrained state.
        let mut vf = value_fn();
        let a = ActionTemplateId(1);
        let trained = obs_with(2, 0.6);
        let untrained = obs_with(2, 90.0); // disjoint tiles

        assert_eq!(vf.visit_count(&trained, a), 0, "no updates yet ⇒ 0 visits");
        for _ in 0..7 {
            vf.update(&trained, a, 1.0, 0.2);
        }
        assert_eq!(
            vf.visit_count(&trained, a),
            7,
            "7 updates ⇒ every active tile of the state was touched 7×"
        );
        assert_eq!(
            vf.visit_count(&untrained, a),
            0,
            "a never-updated (disjoint-tile) state still reads 0 visits"
        );
        // A different action at the same obs shares no action-mixed tiles ⇒ 0.
        assert_eq!(vf.visit_count(&trained, ActionTemplateId(2)), 0);
    }

    #[test]
    fn coverage_reports_unique_and_per_state_updates() {
        let mut vf = value_fn();
        assert_eq!(
            vf.coverage().unique_tiles,
            0,
            "fresh learner trained nothing"
        );
        assert_eq!(vf.coverage().mean_updates_per_state, 0.0);

        // Train two disjoint states: A 10× (1 action), B 4× — each update touches
        // that state's `active_count` tiles, so unique_tiles = 2·active_count and
        // mean_updates_per_state is between the two per-state counts.
        let a = ActionTemplateId(1);
        let sa = obs_with(2, 0.6);
        let sb = obs_with(2, 90.0); // disjoint tiles
        for _ in 0..10 {
            vf.update(&sa, a, 1.0, 0.1);
        }
        for _ in 0..4 {
            vf.update(&sb, a, 1.0, 0.1);
        }
        let cov = vf.coverage();
        let active = vf.coder().active_count();
        assert_eq!(
            cov.unique_tiles,
            2 * active,
            "two disjoint states ⇒ 2× the active-tile count touched"
        );
        assert_eq!(
            cov.total_updates,
            (10 + 4) as u64 * active as u64,
            "total = (updates) × (tiles per update)"
        );
        // Per-state mean = total/unique = 14·active / (2·active) = 7.
        assert!((cov.mean_updates_per_state - 7.0).abs() < 1e-9);
        // The least-trained tiles belong to state B (4 updates).
        assert_eq!(cov.min_updates_per_state, 4);
        // DISTRIBUTION: half the touched tiles are at count 10 (state A), half at 4 (B).
        assert_eq!(
            cov.max_updates_per_state, 10,
            "top of the distribution = the 10×-trained state"
        );
        assert_eq!(
            cov.p25_updates_per_state, 4,
            "lower quartile is the 4×-trained half"
        );
        assert!(
            cov.min_updates_per_state <= cov.p5_updates_per_state
                && cov.p5_updates_per_state <= cov.p25_updates_per_state
                && cov.p25_updates_per_state <= cov.median_updates_per_state
                && cov.median_updates_per_state <= cov.p95_updates_per_state
                && cov.p95_updates_per_state <= cov.p99_updates_per_state
                && cov.p99_updates_per_state <= cov.max_updates_per_state,
            "percentiles must be monotone non-decreasing: {cov:?}"
        );
    }

    #[test]
    fn pctl_nearest_rank_distribution() {
        // 1..=100 ⇒ percentile p maps to ~value p (nearest-rank over indices 0..99).
        let v: Vec<u32> = (1..=100).collect();
        assert_eq!(pctl(&v, 0.0), 1, "p0 = min");
        assert_eq!(pctl(&v, 100.0), 100, "p100 = max");
        assert_eq!(pctl(&v, 5.0), 6); // idx round(.05*99)=5 → v[5]=6
        assert_eq!(pctl(&v, 25.0), 26); // idx round(.25*99)=25 → v[25]=26
        assert_eq!(pctl(&v, 50.0), 51); // idx round(.5*99)=50 → v[50]=51
        assert_eq!(pctl(&v, 95.0), 95); // idx round(.95*99)=94 → v[94]=95
        assert_eq!(pctl(&v, 99.0), 99); // idx round(.99*99)=98 → v[98]=99
                                        // Degenerate inputs.
        assert_eq!(pctl(&[], 50.0), 0, "empty ⇒ 0");
        assert_eq!(pctl(&[7], 0.0), 7);
        assert_eq!(
            pctl(&[7], 100.0),
            7,
            "singleton ⇒ that value at every percentile"
        );
    }

    #[test]
    fn counts_are_excluded_from_value_identity() {
        // PartialEq compares coder+weights only — two value functions with the
        // same weights but different visit histories must compare EQUAL, so a
        // save/load round trip (which drops counts) stays equal.
        let mut a = value_fn();
        let mut b = value_fn();
        let o = obs_with(1, 0.3);
        // Same NET weight change, different number of updates (different counts):
        // 4 small steps on `a`, 1 equivalent-target step on `b` won't match
        // weights — so instead drive both to the same converged weights with a
        // different update COUNT by over-training `a`.
        for _ in 0..200 {
            a.update(&o, ActionTemplateId(0), 1.0, 0.5);
        }
        for _ in 0..200 {
            b.update(&o, ActionTemplateId(0), 1.0, 0.5);
        }
        // Both fully converged to target ⇒ equal weights; counts equal here too.
        assert_eq!(a, b, "equal weights ⇒ equal value functions");
        // Now bump a's counts further WITHOUT moving weights (target == current
        // ⇒ zero step), proving counts don't enter equality.
        let before = a.value(&o, ActionTemplateId(0));
        for _ in 0..50 {
            a.update(&o, ActionTemplateId(0), before, 0.5); // step ≈ 0
        }
        assert!(
            a.visit_count(&o, ActionTemplateId(0)) > b.visit_count(&o, ActionTemplateId(0)),
            "a now has strictly more visits than b"
        );
        assert_eq!(
            a, b,
            "extra visits (counts) must NOT change value-function identity"
        );
    }
}
