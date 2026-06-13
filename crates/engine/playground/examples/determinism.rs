//! `cargo run --example determinism -p playground` — prove the
//! playground loop is byte-deterministic given seed + value-fn
//! snapshot. The simulation has a determinism canary; this example is
//! the analogue for the playground harness.
//!
//! Runs two independent bandit episodes with the same seed and a
//! fresh value function each time, then prints whether the action
//! histories and TD-error sequences match.

use playground::{run_episode, EpisodeConfig, MultiArmedBandit};
use rl_core::{ActionTemplateId, TileCodedValue, TileCoder};

fn run(seed: u64) -> (Vec<usize>, Vec<f32>, f32) {
    let mut env = MultiArmedBandit::new(vec![0.1, 0.3, 0.5, 0.7, 0.9], 0.05);
    let mut value = TileCodedValue::new(TileCoder::uniform(8, 0.25, 1, 1 << 14));
    let cfg = EpisodeConfig {
        max_steps: 500,
        epsilon: 0.2,
        alpha: 0.1,
        gamma: 0.0,
        seed,
    };
    let stats = run_episode(&mut env, &mut value, |a| ActionTemplateId(a as u32), &cfg);
    (stats.action_history, stats.td_errors, stats.total_reward)
}

fn main() {
    let (h_a, e_a, r_a) = run(42);
    let (h_b, e_b, r_b) = run(42);
    let histories_match = h_a == h_b;
    let errors_match = e_a == e_b;
    let rewards_match = r_a == r_b;
    println!("Seed: 42, max_steps: 500");
    println!("Action history match : {histories_match}");
    println!("TD-error match       : {errors_match}");
    println!("Total-reward match   : {rewards_match} ({r_a} vs {r_b})");
    if histories_match && errors_match && rewards_match {
        println!("Deterministic — same seed produces the same trajectory.");
    } else {
        std::process::exit(1);
    }
}
