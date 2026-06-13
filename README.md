# rl-core

Generic, game-agnostic reinforcement-learning machinery in Rust — the value
learner, tile coding, the environment/agent traits, and a toy agent-environment
loop. Extracted from the simulator monorepo so it can be reused standalone (the
interactive RL textbook's WASM widgets build directly on these crates).

## Crates

- [`determinism`](crates/engine/determinism) — the FNV-1a / SplitMix64 /
  unit-window determinism hash kit (pure std, zero deps; the one canonical home
  for reproducibility-critical bit-mixing).
- [`action_id`](crates/engine/action_id) — the `ActionTemplateId` action-key
  newtype (serde only).
- [`rl_core`](crates/engine/rl_core) — `Observation`, `TileCoder`,
  `ValueFunction` + `TileCodedValue` (one-step TD and TD(λ) with eligibility
  traces + average-reward), `Environment`/`StepResult`, the `Mind` trait, and
  ε-greedy / argmax policy helpers. The default-on `bevy` feature adds `Component`
  derives; build `--no-default-features` for a Bevy-free / WASM target.
- [`playground`](crates/engine/playground) — `run_episode` over any
  `Environment` plus reference toy envs (multi-armed bandit, gridworld) — the
  "textbook in a page".

## Build

```sh
cargo test
cargo run --example gridworld -p playground
```

Extracted from [`falahat/simulator`](https://github.com/falahat/simulator),
which consumes these crates as a git dependency; git history remains there.
