//! `cargo run --example bandit -p playground` — solve a 10-arm bandit
//! with a tile-coded value function and ε-greedy action selection.
//! Prints, per 200-step block, the fraction of pulls that landed on the
//! true best arm. With learning, that fraction should climb from
//! ~1/K = 0.1 toward (1 − ε).

use playground::{run_episode, EpisodeConfig, MultiArmedBandit};
use rl_core::{ActionTemplateId, TileCodedValue, TileCoder};

fn main() {
    let means = vec![0.10, 0.25, 0.30, 0.45, 0.55, 0.60, 0.65, 0.70, 0.80, 0.95];
    let mut env = MultiArmedBandit::new(means.clone(), 0.1);
    let mut value = TileCodedValue::new(TileCoder::uniform(8, 0.25, 1, 1 << 14));
    let cfg = EpisodeConfig {
        max_steps: 2_000,
        epsilon: 0.1,
        alpha: 0.1,
        gamma: 0.0,
        seed: 0,
    };
    let best_arm = env.best_arm();
    let stats = run_episode(&mut env, &mut value, |a| ActionTemplateId(a as u32), &cfg);

    println!(
        "10-armed bandit, best arm = {best_arm} (mean {:.2})",
        means[best_arm]
    );
    println!("Block | best-arm pulls / 200");
    let block = 200;
    for (i, chunk) in stats.action_history.chunks(block).enumerate() {
        let best = chunk.iter().filter(|&&a| a == best_arm).count();
        let frac = best as f32 / chunk.len() as f32;
        let bar = "#".repeat((frac * 40.0) as usize);
        println!(" {:>4} | {:>3}/{:<3} {bar}", i * block, best, chunk.len());
    }
    println!("Total reward over episode: {:.1}", stats.total_reward);
}
