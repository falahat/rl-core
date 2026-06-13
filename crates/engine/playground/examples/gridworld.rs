//! `cargo run --example gridworld -p playground` — train ε-greedy
//! Q-learning on a 5×5 gridworld. Goal at (4, 4); start at (0, 0).
//! Prints steps-to-goal per episode; later episodes should be much
//! shorter than the first random-flailing ones.

use playground::{run_episode, EpisodeConfig, GridAction, GridWorld};
use rl_core::{ActionTemplateId, TileCodedValue, TileCoder};

fn main() {
    let mut value = TileCodedValue::new(TileCoder::uniform(8, 0.25, 2, 1 << 14));
    let action_key = |a: GridAction| match a {
        GridAction::North => ActionTemplateId(0),
        GridAction::South => ActionTemplateId(1),
        GridAction::East => ActionTemplateId(2),
        GridAction::West => ActionTemplateId(3),
    };

    println!("5x5 gridworld, start (0, 0), goal (4, 4)");
    println!("Episode | steps-to-goal");
    let mut first_block = 0.0;
    let mut last_block = 0.0;
    let total_episodes = 50;
    for ep in 0..total_episodes {
        let mut env = GridWorld::new(5, 5, (0, 0), (4, 4));
        let cfg = EpisodeConfig {
            max_steps: 1_000,
            epsilon: if ep < 30 { 0.2 } else { 0.05 },
            alpha: 0.2,
            gamma: 0.95,
            seed: ep as u64,
        };
        let stats = run_episode(&mut env, &mut value, action_key, &cfg);
        let steps = stats.action_history.len();
        if ep < 5 {
            first_block += steps as f32;
        }
        if ep >= total_episodes - 5 {
            last_block += steps as f32;
        }
        if ep % 5 == 0 || ep == total_episodes - 1 {
            let bar = "#".repeat((steps as f32 / 20.0).min(40.0) as usize);
            println!(" {:>4}   | {:>4}  {bar}", ep, steps);
        }
    }
    println!(
        "First-5 mean steps-to-goal: {:.1}, last-5: {:.1}",
        first_block / 5.0,
        last_block / 5.0
    );
}
