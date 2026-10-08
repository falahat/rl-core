//! `rl_core` — generic, game-agnostic reinforcement-learning machinery.
//!
//! This crate holds the pieces of the value learner that have nothing to
//! do with any particular game: the generic [`Observation`] input
//! vector, the [`TileCoder`] feature map, the [`ValueFunction`] trait
//! and its tile-coded implementation [`TileCodedValue`], and the
//! [`ActionTemplateId`] action handle. It depends only on `serde` and
//! `bevy_ecs` (ECS infrastructure, not game content).
//!
//! Game-specific content — the `Observation` *schema* (block layout +
//! width invariant), the named action roster, the per-agent `Learner` —
//! lives in `q_learning`, which builds on top of this crate.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod agent;
pub mod environment;
pub mod observation;
pub mod policy;
pub mod tile_coding;
pub mod value_function;

// `hash` (the determinism kit) and `ActionTemplateId` (the action key) now live
// in their own pure leaf crates so the foundation (`kernel`, `ecs_core`) no
// longer reaches up into `rl_core` for them. Re-exported under their historical
// paths so every call site — including rl_core's own `crate::hash::*` /
// `crate::action_id::*` — is unchanged.
pub use action_id;
pub use action_id::ActionTemplateId;
pub use agent::Mind;
pub use determinism as hash;
pub use environment::{Environment, StepResult};
pub use observation::Observation;
pub use policy::{epsilon_explore, greedy_argmax};
pub use tile_coding::{
    Encoder, FeatureBlock, FeatureLayout, Featurize, PreparedObservation, TileCoder,
};
pub use value_function::{
    AverageReward, Coverage, EligibilityTrace, TileCodedValue, ValueFunction,
    UNDERTRAINED_THRESHOLD,
};
