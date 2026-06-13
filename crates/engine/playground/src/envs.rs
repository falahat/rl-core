//! Toy environments for the teaching harness — a [`MultiArmedBandit`]
//! and a [`GridWorld`], both implementing
//! [`Environment`](rl_core::Environment).
//!
//! These exist purely so the playground harness has something concrete
//! to learn against in tests, examples, and (eventually) browser
//! widgets. They are deliberately small and deliberately *not* in
//! `rl_core` — the foundation crate keeps just the trait; concrete
//! content lives next to the harness that consumes it.

use crate::unit_draw;
use rl_core::{Environment, Observation, StepResult};

/// The K-armed bandit — RL's "hello world." `K` arms; pulling arm `i`
/// returns a noisy reward centred on `means[i]`. Deterministic given
/// the seed and action sequence. No state to observe, so every step
/// returns a one-cell zero [`Observation`] (the policy still maps
/// action → value purely through the learner's action key).
#[derive(Debug, Clone)]
pub struct MultiArmedBandit {
    /// The true mean reward of each arm. Hidden from the agent — the
    /// whole point is for the learner to estimate these from samples.
    means: Vec<f32>,
    /// Reward noise standard deviation. A fixed-shape Gaussian-ish draw
    /// (deterministic — see [`gaussian_unit`]).
    noise: f32,
    /// The set of valid actions — `[0, 1, …, K-1]` materialised so
    /// `action_space()` can hand back a borrowed slice.
    actions: Vec<usize>,
    /// Step counter; folded into the per-draw hash so successive pulls
    /// of the same arm see different noise.
    step: u64,
    /// Seed mixed into every noise draw; set in [`Self::reset`].
    seed: u64,
}

impl MultiArmedBandit {
    /// Build a bandit over `means`. `noise` is the reward noise stdev.
    pub fn new(means: Vec<f32>, noise: f32) -> Self {
        let actions = (0..means.len()).collect();
        Self {
            means,
            noise,
            actions,
            step: 0,
            seed: 0,
        }
    }

    /// The arm with the highest true mean — the regret oracle.
    pub fn best_arm(&self) -> usize {
        let mut best = 0;
        for i in 1..self.means.len() {
            if self.means[i] > self.means[best] {
                best = i;
            }
        }
        best
    }
}

impl Environment for MultiArmedBandit {
    type Action = usize;

    fn reset(&mut self, seed: u64) -> Observation {
        self.seed = seed;
        self.step = 0;
        // The bandit has no internal state — a one-cell zero observation
        // is enough for tile coding to range over a single bucket.
        Observation::zeros(1)
    }

    fn step(&mut self, action: Self::Action) -> StepResult {
        let mean = self.means[action];
        let noise = self.noise * gaussian_unit(self.seed, self.step, action as u64);
        self.step += 1;
        StepResult {
            observation: Observation::zeros(1),
            reward: mean + noise,
            done: false,
        }
    }

    fn action_space(&self) -> &[Self::Action] {
        &self.actions
    }
}

/// A 4×4 (or general `W×H`) gridworld with one goal cell. Movement is
/// deterministic: the agent steps in the chosen direction unless that
/// would leave the grid, in which case it stays put. Reaching the goal
/// gives reward `+1` and ends the episode; every other step gives `0`.
/// The observation is the agent's `(x, y)` scaled to `[0, 1]` — two
/// real-valued features the tile coder can range over.
#[derive(Debug, Clone)]
pub struct GridWorld {
    width: u32,
    height: u32,
    goal: (u32, u32),
    agent: (u32, u32),
    start: (u32, u32),
    actions: Vec<GridAction>,
}

/// The four cardinal moves a [`GridWorld`] accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridAction {
    North,
    South,
    East,
    West,
}

impl GridWorld {
    /// A `width × height` grid with the agent starting at `start` and
    /// the goal at `goal`.
    pub fn new(width: u32, height: u32, start: (u32, u32), goal: (u32, u32)) -> Self {
        Self {
            width,
            height,
            goal,
            agent: start,
            start,
            actions: vec![
                GridAction::North,
                GridAction::South,
                GridAction::East,
                GridAction::West,
            ],
        }
    }

    /// The agent's current cell — exposed for diagnostics, not the
    /// learner (the learner reads the same data via [`Environment::reset`]
    /// / [`Environment::step`] observations).
    pub fn agent(&self) -> (u32, u32) {
        self.agent
    }

    fn observation(&self) -> Observation {
        // Scale (x, y) into [0, 1] — a width-2 observation the
        // tile coder generalises across.
        let mut o = Observation::zeros(2);
        let v = o.as_mut_slice();
        v[0] = self.agent.0 as f32 / (self.width.saturating_sub(1).max(1)) as f32;
        v[1] = self.agent.1 as f32 / (self.height.saturating_sub(1).max(1)) as f32;
        o
    }
}

impl Environment for GridWorld {
    type Action = GridAction;

    fn reset(&mut self, _seed: u64) -> Observation {
        // The gridworld is fully deterministic — the seed is unused
        // beyond signalling the start of an episode.
        self.agent = self.start;
        self.observation()
    }

    fn step(&mut self, action: Self::Action) -> StepResult {
        let (x, y) = self.agent;
        let (nx, ny) = match action {
            GridAction::North => (x, y.saturating_add(1).min(self.height - 1)),
            GridAction::South => (x, y.saturating_sub(1)),
            GridAction::East => (x.saturating_add(1).min(self.width - 1), y),
            GridAction::West => (x.saturating_sub(1), y),
        };
        self.agent = (nx, ny);
        let done = self.agent == self.goal;
        let reward = if done { 1.0 } else { 0.0 };
        StepResult {
            observation: self.observation(),
            reward,
            done,
        }
    }

    fn action_space(&self) -> &[Self::Action] {
        &self.actions
    }
}

/// The five moves a [`ForageEnv`] accepts — the four cardinals plus `Eat`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForageAction {
    North,
    South,
    East,
    West,
    Eat,
}

/// A minimal **forage** task: an agent on a `W×H` grid whose `hunger` rises every
/// step and must reach food and `Eat` to bring it back down. The reward is the
/// homeostatic `-hunger` (CostOnly, ≤ 0 — the same shape the voxel sim uses), so
/// minimising cumulative hunger by foraging promptly is optimal. `Eat` when
/// adjacent to food resets hunger and respawns the food at a fresh deterministic
/// cell; a mistimed `Eat` is a wasted step. The episode runs to `max_steps`
/// (no terminal — so there is no "starve early to stop the cost" shortcut).
///
/// The observation is five **legible** features — `hunger`, the relative food
/// bearing `(dx, dy)`, normalised distance, and an adjacency flag — decode them
/// ergonomically with [`crate::Senses`] instead of indexing the raw vector.
#[derive(Debug, Clone)]
pub struct ForageEnv {
    width: i32,
    height: i32,
    agent: (i32, i32),
    food: (i32, i32),
    hunger: f32,
    /// Hunger gained per step (rises toward 1.0).
    drain: f32,
    step: u64,
    seed: u64,
    actions: Vec<ForageAction>,
}

impl ForageEnv {
    /// A `width × height` forage grid with the default hunger drain (0.04/step,
    /// i.e. ~25 steps from sated to starving).
    pub fn new(width: i32, height: i32) -> Self {
        Self {
            width: width.max(2),
            height: height.max(2),
            agent: (width / 2, height / 2),
            food: (0, 0),
            hunger: 0.5,
            drain: 0.04,
            step: 0,
            seed: 0,
            actions: vec![
                ForageAction::North,
                ForageAction::South,
                ForageAction::East,
                ForageAction::West,
                ForageAction::Eat,
            ],
        }
    }

    /// Agent / food cells + current hunger — for diagnostics + viz, not the
    /// learner (which reads the same facts through the observation).
    pub fn agent(&self) -> (i32, i32) {
        self.agent
    }
    pub fn food(&self) -> (i32, i32) {
        self.food
    }
    pub fn hunger(&self) -> f32 {
        self.hunger
    }

    /// Deterministic interior food cell, never the agent's own cell.
    fn respawn_food(&mut self) {
        let fx = (unit_draw(self.seed, self.step, 0x_F0) * self.width as f32) as i32;
        let fy = (unit_draw(self.seed, self.step, 0x_F1) * self.height as f32) as i32;
        let mut f = (fx.clamp(0, self.width - 1), fy.clamp(0, self.height - 1));
        if f == self.agent {
            f.0 = if f.0 < self.width - 1 {
                f.0 + 1
            } else {
                f.0 - 1
            };
        }
        self.food = f;
    }

    fn manhattan_to_food(&self) -> i32 {
        (self.food.0 - self.agent.0).abs() + (self.food.1 - self.agent.1).abs()
    }

    /// Five legible features — see [`crate::Senses`] for the read side.
    fn observation(&self) -> Observation {
        let mut o = Observation::zeros(5);
        let v = o.as_mut_slice();
        let (dx, dy) = (self.food.0 - self.agent.0, self.food.1 - self.agent.1);
        v[0] = self.hunger;
        v[1] = dx as f32 / self.width as f32;
        v[2] = dy as f32 / self.height as f32;
        v[3] = self.manhattan_to_food() as f32 / (self.width + self.height) as f32;
        v[4] = if self.manhattan_to_food() <= 1 {
            1.0
        } else {
            0.0
        };
        o
    }
}

impl Environment for ForageEnv {
    type Action = ForageAction;

    fn reset(&mut self, seed: u64) -> Observation {
        self.seed = seed;
        self.step = 0;
        self.agent = (self.width / 2, self.height / 2);
        self.hunger = 0.5;
        self.respawn_food();
        self.observation()
    }

    fn step(&mut self, action: Self::Action) -> StepResult {
        let (x, y) = self.agent;
        match action {
            ForageAction::North => self.agent = (x, (y + 1).min(self.height - 1)),
            ForageAction::South => self.agent = (x, (y - 1).max(0)),
            ForageAction::East => self.agent = ((x + 1).min(self.width - 1), y),
            ForageAction::West => self.agent = ((x - 1).max(0), y),
            ForageAction::Eat => {
                if self.manhattan_to_food() <= 1 {
                    self.hunger = 0.0;
                    self.respawn_food();
                }
            }
        }
        self.step += 1;
        self.hunger = (self.hunger + self.drain).min(1.0);
        StepResult {
            observation: self.observation(),
            reward: -self.hunger, // CostOnly homeostatic: minimise cumulative hunger
            done: false,          // no terminal ⇒ no "starve early" shortcut
        }
    }

    fn action_space(&self) -> &[Self::Action] {
        &self.actions
    }
}

/// A deterministic, roughly-Gaussian-shaped unit draw — the Box-Muller
/// transform fed by two [`unit_draw`] hashes of `(seed, step, salt)`.
/// Range is approximately `[-3, 3]`. This is the same determinism rule
/// the simulation uses: noise is a pure function of inputs, never a
/// process RNG, so save/load round-trips stay byte-identical.
fn gaussian_unit(seed: u64, step: u64, salt: u64) -> f32 {
    let u1 = unit_draw(seed, step, salt ^ 0xA1);
    let u2 = unit_draw(seed, step, salt ^ 0xB2);
    // Guard against u1 = 0 so the log is finite.
    let u1 = u1.max(1.0e-9);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * core::f32::consts::PI * u2;
    r * theta.cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bandit_reset_returns_a_well_formed_observation() {
        let mut b = MultiArmedBandit::new(vec![0.1, 0.5, 0.9], 0.05);
        let o = b.reset(7);
        assert_eq!(o.as_slice().len(), 1);
    }

    #[test]
    fn bandit_reward_concentrates_around_the_arms_mean() {
        let mut b = MultiArmedBandit::new(vec![0.1, 0.5, 0.9], 0.05);
        b.reset(7);
        let mut sum = 0.0;
        let n = 500;
        for _ in 0..n {
            sum += b.step(2).reward;
        }
        let mean = sum / n as f32;
        assert!(
            (mean - 0.9).abs() < 0.1,
            "arm-2 mean should be near 0.9, got {mean}"
        );
    }

    #[test]
    fn gridworld_reset_returns_to_start() {
        let mut g = GridWorld::new(4, 4, (0, 0), (3, 3));
        g.step(GridAction::East);
        g.step(GridAction::North);
        assert_ne!(g.agent(), (0, 0));
        g.reset(0);
        assert_eq!(g.agent(), (0, 0));
    }

    #[test]
    fn gridworld_goal_terminates_with_reward_one() {
        let mut g = GridWorld::new(4, 4, (3, 2), (3, 3));
        g.reset(0);
        let r = g.step(GridAction::North);
        assert!(r.done, "stepping onto the goal must terminate the episode");
        assert_eq!(r.reward, 1.0);
    }

    #[test]
    fn gridworld_clamps_at_edges() {
        let mut g = GridWorld::new(4, 4, (0, 0), (3, 3));
        g.reset(0);
        // Walking west / south from the corner can't leave the grid.
        g.step(GridAction::West);
        g.step(GridAction::South);
        assert_eq!(g.agent(), (0, 0));
    }

    #[test]
    fn bandit_step_count_is_deterministic_per_seed() {
        let mut a = MultiArmedBandit::new(vec![0.1, 0.5, 0.9], 0.1);
        let mut b = MultiArmedBandit::new(vec![0.1, 0.5, 0.9], 0.1);
        a.reset(42);
        b.reset(42);
        for arm in [0, 1, 2, 0, 1, 2] {
            assert_eq!(a.step(arm).reward, b.step(arm).reward);
        }
    }
}
