//! `ActionTemplateId` — small integer handle for an action.
//!
//! A generic, game-agnostic action handle: the tile coder folds it into
//! the feature hash so a learned value function can be keyed by
//! `(observation, action)`, and the ECS act-identity table
//! (`ecs_core::act`) numbers its templates with the same type. Game-specific
//! action rosters (the named template enums that hand out fixed numeric
//! values) live in the crates that consume this one.
//!
//! Storing a `u32` instead of an `Arc<str>` keeps the per-(template, ctx)
//! hash-map keys 12 bytes wide and the table dense.

#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ActionTemplateId(pub u32);
