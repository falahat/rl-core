//! `playground` — the textbook's agent-environment loop, in 30 lines.
//!
//! [`run_episode`] runs one episode against any [`Environment`]
//! implementation: reset, then alternate ε-greedy action selection with
//! TD updates until the episode terminates or `max_steps` is reached.
//! No Bevy, no ECS, no game-specific content — the loop is generic over
//! `(Env, ValueFunction)` so any pair plugs in.
//!
//! This is the "Simulator-in-a-page" artefact the textbook proposal
//! asks for. Examples (`bandit.rs`, `gridworld.rs`) live in
//! `examples/` and train against the reference environments in
//! `rl_core::environment`.

#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod envs;
pub mod senses;

pub use envs::{ForageAction, ForageEnv, GridAction, GridWorld, MultiArmedBandit};
pub use senses::Senses;

use rl_core::policy::{epsilon_explore, greedy_argmax};
use rl_core::{ActionTemplateId, Environment, Mind, Observation, StepResult, ValueFunction};

/// Tunable knobs of one [`run_episode`] call.
#[derive(Debug, Clone)]
pub struct EpisodeConfig {
    /// Maximum steps before the episode is cut off. Hard cap on a
    /// non-terminating env.
    pub max_steps: usize,
    /// ε for ε-greedy action selection. `0.0` = pure greedy, `1.0` =
    /// pure exploration.
    pub epsilon: f32,
    /// TD learning rate `α`.
    pub alpha: f32,
    /// Discount factor `γ` on the bootstrap.
    pub gamma: f32,
    /// Seed for the deterministic explore/exploit hash and the env's
    /// `reset(seed)`. Same seed + same `ValueFunction` snapshot = same
    /// trajectory.
    pub seed: u64,
}

impl Default for EpisodeConfig {
    fn default() -> Self {
        Self {
            max_steps: 1_000,
            epsilon: 0.1,
            alpha: 0.1,
            gamma: 0.9,
            seed: 0,
        }
    }
}

/// What [`run_episode`] returns — enough to plot a learning curve.
#[derive(Debug, Clone)]
pub struct EpisodeStats {
    /// Sum of per-step rewards over the episode.
    pub total_reward: f32,
    /// One TD error per step — `|δ|` shrinks as `Q` converges.
    pub td_errors: Vec<f32>,
    /// Index of the action chosen at each step, into `env.action_space()`.
    pub action_history: Vec<usize>,
    /// `true` if the episode terminated naturally (env returned
    /// `done`), `false` if it was cut off at `max_steps`.
    pub terminated: bool,
}

/// One episode of the textbook agent-environment loop. The body is
/// exactly the loop Chapters 5–8 of Sutton & Barto describe:
///
/// 1. Reset the env, get the initial observation.
/// 2. For each step: pick an action via ε-greedy over the learner's
///    `Q(obs, a)`; step the env; apply a TD update.
/// 3. Stop on `done` or after `max_steps`.
///
/// `action_key` maps the env's `Action` to the learner's
/// [`ActionTemplateId`] key — environments use a natural enum/index,
/// but the value function indexes weights by a flat id. For the
/// reference envs in `rl_core`, `|a| ActionTemplateId(a as u32)` (or
/// the bandit's `|a| ActionTemplateId(a as u32)`) does the job.
pub fn run_episode<E: Environment, V: ValueFunction>(
    env: &mut E,
    value: &mut V,
    action_key: fn(E::Action) -> ActionTemplateId,
    cfg: &EpisodeConfig,
) -> EpisodeStats
where
    E::Action: PartialEq,
{
    // The reference ε-greedy TD agent, expressed as a `Mind` borrowing `value`
    // so the caller keeps it across episodes. Byte-identical to the old inline
    // loop (same draws, same `td_step`).
    let mut mind = TdMind {
        value,
        action_key,
        epsilon: cfg.epsilon,
        alpha: cfg.alpha,
        gamma: cfg.gamma,
    };
    run_episode_with_mind(env, &mut mind, cfg)
}

/// The same loop, driving **any** [`Mind`] — the swappable-agent seam. The
/// fixed ε-greedy/TD policy lives in [`TdMind`]; drop in a different `Mind`
/// (a learned net, a scripted baseline) and the loop is unchanged. Determinism:
/// the mind's stochastic draws come from `splitmix64(seed, step, salt)`, never
/// a process RNG — same seed + same mind ⇒ same trajectory.
pub fn run_episode_with_mind<E, M>(env: &mut E, mind: &mut M, cfg: &EpisodeConfig) -> EpisodeStats
where
    E: Environment,
    E::Action: PartialEq,
    M: Mind<Action = E::Action>,
{
    let mut obs = env.reset(cfg.seed);
    let actions: Vec<E::Action> = env.action_space().to_vec();
    let mut total_reward = 0.0;
    let mut td_errors = Vec::with_capacity(cfg.max_steps);
    let mut action_history = Vec::with_capacity(cfg.max_steps);
    let mut terminated = false;
    for step in 0..cfg.max_steps {
        let seed = cfg.seed;
        let mut draw = move |salt: u64| unit_draw(seed, step as u64, salt);
        let action = mind.act(&obs, &actions, &mut draw);
        // Record the chosen action's index for the stats / determinism check.
        let chosen_idx = actions.iter().position(|a| *a == action).unwrap_or(0);
        let StepResult {
            observation: next_obs,
            reward,
            done,
        } = env.step(action);
        let td_error = mind.learn(
            &obs,
            action,
            reward,
            if done { None } else { Some(&next_obs) },
            &actions,
        );
        total_reward += reward;
        td_errors.push(td_error);
        action_history.push(chosen_idx);
        if done {
            terminated = true;
            break;
        }
        obs = next_obs;
    }
    EpisodeStats {
        total_reward,
        td_errors,
        action_history,
        terminated,
    }
}

/// Aggregate of an [`evaluate`] run — enough to rank agents on a leaderboard.
#[derive(Debug, Clone)]
pub struct Eval {
    /// Episodes run.
    pub episodes: usize,
    /// Mean total reward across all episodes (higher = better).
    pub mean_reward: f32,
    /// Total reward of the final episode — end-of-training performance for a
    /// learner (≈ `mean_reward` for a fixed/reactive agent).
    pub last_reward: f32,
}

/// Run a [`Mind`] for `episodes` **sequential** episodes on fresh envs (the mind
/// persists and keeps learning across them; each episode reseeds off `cfg.seed`),
/// aggregating per-episode reward. This is the one call a "can my agent beat the
/// Q-learner?" comparison needs — point it at any `Mind` (a hand-written
/// heuristic, a `TileCodedQMind`) over any [`Environment`]. Deterministic.
pub fn evaluate<E, M>(
    mut make_env: impl FnMut() -> E,
    mind: &mut M,
    episodes: usize,
    cfg: &EpisodeConfig,
) -> Eval
where
    E: Environment,
    E::Action: PartialEq,
    M: Mind<Action = E::Action>,
{
    let mut total = 0.0;
    let mut last = 0.0;
    for ep in 0..episodes {
        let mut env = make_env();
        let c = EpisodeConfig {
            seed: cfg.seed.wrapping_add(ep as u64),
            ..cfg.clone()
        };
        let stats = run_episode_with_mind(&mut env, mind, &c);
        total += stats.total_reward;
        last = stats.total_reward;
    }
    Eval {
        episodes,
        mean_reward: total / episodes.max(1) as f32,
        last_reward: last,
    }
}

/// The reference [`Mind`]: ε-greedy over a [`ValueFunction`] with one-step
/// `max_a` bootstrap TD. Borrows the value function so it persists across
/// episodes (the gridworld trains one value over many resets).
pub struct TdMind<'a, V, A> {
    value: &'a mut V,
    action_key: fn(A) -> ActionTemplateId,
    epsilon: f32,
    alpha: f32,
    gamma: f32,
}

impl<V, A> std::fmt::Debug for TdMind<'_, V, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The borrowed value function isn't `Debug`-bound; show the policy
        // hyper-parameters, which is what a reader wants anyway.
        f.debug_struct("TdMind")
            .field("epsilon", &self.epsilon)
            .field("alpha", &self.alpha)
            .field("gamma", &self.gamma)
            .finish_non_exhaustive()
    }
}

impl<V: ValueFunction, A: Copy> Mind for TdMind<'_, V, A> {
    type Action = A;

    fn act(&mut self, obs: &Observation, actions: &[A], draw: &mut dyn FnMut(u64) -> f32) -> A {
        debug_assert!(!actions.is_empty(), "act needs at least one candidate");
        let idx = epsilon_explore(
            draw(EXPLORE_SALT),
            self.epsilon,
            actions.len(),
            draw(PICK_SALT),
        )
        .unwrap_or_else(|| {
            greedy_argmax(actions, |&a| self.value.value(obs, (self.action_key)(a)))
        });
        actions[idx]
    }

    fn learn(
        &mut self,
        obs: &Observation,
        action: A,
        reward: f32,
        next_obs: Option<&Observation>,
        actions: &[A],
    ) -> f32 {
        td_step(
            self.value,
            obs,
            (self.action_key)(action),
            reward,
            next_obs,
            actions,
            &self.action_key,
            self.alpha,
            self.gamma,
        )
    }
}

/// One TD step on a generic [`ValueFunction`] — the in-place equivalent
/// of `Learner::td_update`. Computed: target = reward + γ · max_a
/// Q(next, a); update Q(obs, action) toward target with rate α.
/// Returns the TD error δ = target − Q(obs, action).
///
/// The bootstrap max is **unfloored** — the invariant
/// `Learner::best_for_observation` keeps: under a cost-shaped reward
/// every learned `Q` is ≤ 0, and flooring the max at 0 would collapse
/// the target to `reward + γ·0`, discarding the successor's value.
/// Empty roster (a terminal, no successor) ⇒ 0.
#[allow(clippy::too_many_arguments)]
fn td_step<EAction: Copy, V: ValueFunction>(
    value: &mut V,
    obs: &Observation,
    action: ActionTemplateId,
    reward: f32,
    next_obs: Option<&Observation>,
    actions: &[EAction],
    action_key: &impl Fn(EAction) -> ActionTemplateId,
    alpha: f32,
    gamma: f32,
) -> f32 {
    let q_now = value.value(obs, action);
    let next_max = match next_obs {
        Some(o) if !actions.is_empty() => actions
            .iter()
            .map(|&a| value.value(o, action_key(a)))
            .fold(f32::NEG_INFINITY, f32::max),
        _ => 0.0,
    };
    let target = reward + gamma * next_max;
    value.update(obs, action, target, alpha);
    target - q_now
}

/// Salts for the ε-greedy explore coin + random pick — two intentionally
/// distinct streams of [`unit_draw`], and intentionally distinct from
/// every other site's salts (q_learning's mind, the planner's keyed
/// draws): each picker keeps its own streams.
const EXPLORE_SALT: u64 = 0xC0FFEE_C0FFEE;
const PICK_SALT: u64 = 0xDEAD_BEEF;

/// Stateless unit draw keyed on `(seed, step, salt)` — the playground's
/// per-step decision draw, shared with the toy envs' noise
/// (`envs::unit_draw` callers). The first stage is a deliberate partial
/// SplitMix mix of `(seed, step)` (two multiply rounds, no trailing
/// xor-shift — pinned by the determinism tests); the salt is then folded
/// in and finished with the shared [`rl_core::hash`] finalizer + unit
/// tail.
pub(crate) fn unit_draw(seed: u64, step: u64, salt: u64) -> f32 {
    let z = seed.wrapping_add(rl_core::hash::GOLDEN_GAMMA);
    let mut t = z ^ step;
    t = (t ^ (t >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    t = (t ^ (t >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    t ^= salt;
    rl_core::hash::unit_f32(rl_core::hash::splitmix_finalize(t))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rl_core::{TileCodedValue, TileCoder};

    fn bandit_value_fn(width: usize) -> TileCodedValue {
        TileCodedValue::new(TileCoder::uniform(8, 0.25, width, 1 << 14))
    }

    #[test]
    fn bandit_learner_prefers_the_best_arm() {
        // Three arms with very different means; after 2000 ε-greedy
        // steps the learner should pick the best arm overwhelmingly.
        let mut env = MultiArmedBandit::new(vec![0.05, 0.4, 0.9], 0.05);
        let mut value = bandit_value_fn(1);
        let cfg = EpisodeConfig {
            max_steps: 2_000,
            epsilon: 0.1,
            alpha: 0.1,
            gamma: 0.0, // bandit has no successor state
            seed: 7,
        };
        let stats = run_episode(&mut env, &mut value, |a| ActionTemplateId(a as u32), &cfg);
        let best_arm = env.best_arm();
        let best_count = stats
            .action_history
            .iter()
            .filter(|&&a| a == best_arm)
            .count();
        let frac = best_count as f32 / stats.action_history.len() as f32;
        assert!(
            frac > 0.7,
            "best arm picked {best_count} / {} times ({frac:.2})",
            stats.action_history.len()
        );
    }

    #[test]
    fn gridworld_learner_eventually_reaches_the_goal() {
        // 4×4 gridworld, start (0, 0), goal (3, 3). Train across
        // multiple episodes; later episodes should reach the goal
        // in many fewer steps than the first random-flailing ones.
        let mut value = TileCodedValue::new(TileCoder::uniform(8, 0.25, 2, 1 << 14));
        let mut steps_per_ep = Vec::new();
        for ep in 0..30 {
            let mut env = GridWorld::new(4, 4, (0, 0), (3, 3));
            let cfg = EpisodeConfig {
                max_steps: 500,
                epsilon: 0.2,
                alpha: 0.2,
                gamma: 0.95,
                seed: ep as u64,
            };
            let stats = run_episode(
                &mut env,
                &mut value,
                |a| match a {
                    GridAction::North => ActionTemplateId(0),
                    GridAction::South => ActionTemplateId(1),
                    GridAction::East => ActionTemplateId(2),
                    GridAction::West => ActionTemplateId(3),
                },
                &cfg,
            );
            steps_per_ep.push(stats.action_history.len());
        }
        let first = steps_per_ep[0..5].iter().sum::<usize>() as f32 / 5.0;
        let last = steps_per_ep[25..30].iter().sum::<usize>() as f32 / 5.0;
        assert!(
            last < first,
            "later episodes should take fewer steps to reach the goal \
             (first 5 avg {first:.1}, last 5 avg {last:.1})"
        );
    }

    #[test]
    fn determinism_same_seed_same_trajectory() {
        // Two independent runs with the same seed and the same fresh
        // value function must produce the exact same action history.
        let cfg = EpisodeConfig {
            max_steps: 500,
            epsilon: 0.3,
            alpha: 0.1,
            gamma: 0.9,
            seed: 42,
        };
        let mk = || {
            let mut env = MultiArmedBandit::new(vec![0.1, 0.5, 0.9], 0.05);
            let mut value = bandit_value_fn(1);
            run_episode(&mut env, &mut value, |a| ActionTemplateId(a as u32), &cfg)
        };
        let a = mk();
        let b = mk();
        assert_eq!(a.action_history, b.action_history);
        assert_eq!(a.td_errors, b.td_errors);
    }

    /// The swappable-agent seam: a *different* `Mind` (not the ε-greedy TD one)
    /// drives the identical loop. A scripted mind that always takes action 0 and
    /// never learns must yield an all-zero action history and all-zero learning
    /// signal — proof the loop is agnostic to the agent algorithm.
    #[test]
    fn run_episode_with_mind_accepts_an_arbitrary_mind() {
        use rl_core::Mind;

        #[derive(Debug)]
        struct FixedMind;
        impl Mind for FixedMind {
            type Action = GridAction;
            fn act(
                &mut self,
                _obs: &Observation,
                _actions: &[GridAction],
                _draw: &mut dyn FnMut(u64) -> f32,
            ) -> GridAction {
                GridAction::North // action index 0
            }
            fn learn(
                &mut self,
                _obs: &Observation,
                _action: GridAction,
                _reward: f32,
                _next: Option<&Observation>,
                _actions: &[GridAction],
            ) -> f32 {
                0.0 // a scripted mind doesn't learn
            }
        }

        let mut env = GridWorld::new(4, 4, (0, 0), (3, 3));
        let cfg = EpisodeConfig {
            max_steps: 20,
            seed: 1,
            ..EpisodeConfig::default()
        };
        let stats = run_episode_with_mind(&mut env, &mut FixedMind, &cfg);
        assert!(!stats.action_history.is_empty());
        let first = stats.action_history[0];
        assert!(
            stats.action_history.iter().all(|&i| i == first),
            "a fixed mind takes the same action every step",
        );
        assert!(
            stats.td_errors.iter().all(|&d| d == 0.0),
            "scripted mind never learns"
        );
    }
}
