//! `Senses` — a typed, autocomplete-friendly view over a [`ForageEnv`](crate::envs::ForageEnv) observation.
//!
//! A [`Mind`](rl_core::Mind) receives a raw [`Observation`](rl_core::Observation)
//! — a fixed-width `f32` vector that's right for the tile coder but hostile to a
//! human writing an agent (`obs.as_slice()[3]` tells you nothing). `Senses`
//! decodes that vector into named, intent-revealing accessors so a hand-written
//! agent reads like prose:
//!
//! ```ignore
//! let s = Senses::read(obs);
//! if s.adjacent_food() { ForageAction::Eat }
//! else { s.food_bearing().unwrap_or(ForageAction::Eat) }
//! ```
//!
//! This is the per-environment "observation space" view (à la a Gym `Env`'s
//! typed spaces): it knows *this* env's layout, so the engine keeps its flat
//! vector and the author never indexes raw offsets.

use crate::envs::ForageAction;
use rl_core::Observation;

/// A decoded snapshot of a [`ForageEnv`](crate::ForageEnv) observation. Cheap to
/// build (`read`) and `Copy`, so an agent can pull it at the top of `act`.
#[derive(Debug, Clone, Copy)]
pub struct Senses {
    hunger: f32,
    food_dx: f32,
    food_dy: f32,
    adjacent: bool,
}

impl Senses {
    /// Decode a `ForageEnv` observation. (Layout: `[hunger, dx, dy, dist, adj]`;
    /// the `dist` slot at index 3 is not surfaced as an accessor.)
    pub fn read(obs: &Observation) -> Self {
        let v = obs.as_slice();
        Self {
            hunger: v[0],
            food_dx: v[1],
            food_dy: v[2],
            adjacent: v[4] > 0.5,
        }
    }

    /// How hungry the agent is, `0.0` (sated) .. `1.0` (starving).
    pub fn hunger(&self) -> f32 {
        self.hunger
    }

    /// `true` when the agent is on or next to food — i.e. `Eat` would land.
    pub fn adjacent_food(&self) -> bool {
        self.adjacent
    }

    /// The single cardinal step that reduces the distance to food the most
    /// (dominant axis; ties favour x). `None` only when already on the food.
    pub fn food_bearing(&self) -> Option<ForageAction> {
        if self.food_dx == 0.0 && self.food_dy == 0.0 {
            return None;
        }
        Some(if self.food_dx.abs() >= self.food_dy.abs() {
            if self.food_dx >= 0.0 {
                ForageAction::East
            } else {
                ForageAction::West
            }
        } else if self.food_dy >= 0.0 {
            ForageAction::North
        } else {
            ForageAction::South
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_named_fields_and_bearing() {
        // Food to the east-ish and a bit north, not adjacent, hunger 0.7.
        let obs = Observation::from_values(vec![0.7, 0.5, 0.2, 0.4, 0.0]);
        let s = Senses::read(&obs);
        assert!((s.hunger() - 0.7).abs() < 1e-6);
        assert!(!s.adjacent_food());
        assert_eq!(s.food_bearing(), Some(ForageAction::East)); // |dx| > |dy|, dx>0
    }

    #[test]
    fn adjacency_flag_and_on_food_bearing() {
        let on_food = Observation::from_values(vec![0.3, 0.0, 0.0, 0.0, 1.0]);
        let s = Senses::read(&on_food);
        assert!(s.adjacent_food());
        assert_eq!(s.food_bearing(), None);
    }
}
