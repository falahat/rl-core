//! `Mind` — the agent-side port of the RL loop.
//!
//! [`Environment`](crate::Environment) is the *world* side of Sutton & Barto's
//! agent-environment loop; this module is the *agent* side. A [`Mind`] observes,
//! chooses an action, and learns from the outcome — owning its **own**
//! representation, value function, and policy internally. The loop only feeds it
//! observations, the candidate action set, and rewards, so swapping the AI
//! driving an agent is swapping one `Mind` impl; the world is untouched.
//!
//! Both traits are **game-agnostic** (no schema, no ECS): the same `Mind` plugs
//! into this voxel sim, a gridworld, or a hypothetical Pac-Man, as long as the
//! action type matches. Concrete impls live one tier up (`playground::TdMind`,
//! the sim's `TileCodedQMind`), mirroring the `Environment` (trait here) /
//! `GridWorld` (impl in `playground`) split.

use crate::Observation;

/// A pluggable agent: **observe → act**, then **learn** from the transition.
///
/// The `Mind` owns whatever it needs to decide and learn (a value function,
/// eligibility trace, replay buffer, policy hyper-parameters …) — none of that
/// leaks into the loop. This is the seam that makes "swap the AI" a one-type
/// change, and the unit that a different game reuses unchanged.
pub trait Mind {
    /// The action type this mind emits — the environment's action enum/id.
    type Action: Copy;

    /// Choose an action for `obs` out of `actions`.
    ///
    /// `draw(salt) -> [0, 1)` is a **deterministic** uniform source for
    /// stochastic policies (e.g. ε-greedy draws its explore/exploit coin and
    /// its random pick from it). A greedy mind ignores it. Passing the draw in
    /// — rather than owning an RNG — keeps the mind deterministic and agnostic
    /// to the caller's RNG discipline (the sim's keyed `RngResource`, a test's
    /// `splitmix64`, …).
    fn act(
        &mut self,
        obs: &Observation,
        actions: &[Self::Action],
        draw: &mut dyn FnMut(u64) -> f32,
    ) -> Self::Action;

    /// Learn from one committed transition. `next_obs = None` marks a terminal
    /// (the loop stops bootstrapping past it). `actions` is the candidate set at
    /// the successor, for rules that bootstrap over it (`max_a Q(s', a)`).
    ///
    /// Returns the **learning signal** for this step — the TD error `δ` for a
    /// bootstrapping mind, `0.0` for one that doesn't learn (random / scripted).
    /// A diagnostic the loop can log; `|δ| → 0` is the textbook convergence
    /// signal.
    fn learn(
        &mut self,
        obs: &Observation,
        action: Self::Action,
        reward: f32,
        next_obs: Option<&Observation>,
        actions: &[Self::Action],
    ) -> f32;
}
