//! `Environment` — the agent-environment loop the RL textbook calls
//! its most fundamental abstraction.
//!
//! Sutton & Barto, Chapter 3, §3.1: at every time step the agent picks
//! an action; the environment returns a next-observation and a reward.
//! That's the whole protocol. This module is exactly that protocol —
//! the trait and its `StepResult` return type, nothing else.
//!
//! Concrete environments are deliberately *not* in this crate. Toy
//! envs (`MultiArmedBandit`, `GridWorld`) live next to the teaching
//! harness in [`playground::envs`](https://docs.rs/playground/latest/playground/envs/);
//! the Bevy simulation is its own special case the engine wires up.
//! Keeping `rl_core` to the trait alone is the same separation the
//! crate enforces elsewhere — generic machinery here, concrete content
//! one tier up.

use crate::observation::Observation;

/// What an [`Environment::step`] returns: the next observation, the
/// scalar reward, and whether the episode terminated.
#[derive(Debug, Clone)]
pub struct StepResult {
    /// The next observation — what the agent sees after the action.
    pub observation: Observation,
    /// The scalar reward the agent collects on this transition.
    pub reward: f32,
    /// `true` if the episode terminated on this step. The caller stops
    /// bootstrapping past a terminal transition.
    pub done: bool,
}

/// The agent-environment loop. An environment owns its own state,
/// produces an [`Observation`] the learner can ingest, and accepts a
/// game-specific `Action` chosen by a policy. Implementations must be
/// deterministic given their seed — a re-run with the same seed and the
/// same action sequence produces the same step results.
pub trait Environment {
    /// The concrete action type this environment accepts. Toy envs use
    /// a small enum; richer envs may use a struct or an opaque id.
    type Action: Copy;

    /// Reset to the start state and return the initial observation.
    /// `seed` controls any internal randomness — the same seed must
    /// produce the same trajectory.
    fn reset(&mut self, seed: u64) -> Observation;

    /// Step forward with `action`. The returned [`StepResult`] carries
    /// the next observation, the reward, and the terminal flag.
    fn step(&mut self, action: Self::Action) -> StepResult;

    /// The full discrete action set — what the policy enumerates over.
    /// Static for each environment: a bandit's K arms, a gridworld's
    /// four cardinal directions. Returned as a borrowed slice so the
    /// caller can pass it straight to `Learner::best_for_observation`.
    fn action_space(&self) -> &[Self::Action];
}
