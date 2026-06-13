//! `TileCoder` — the generalising feature map (Sutton & Barto ch. 9,
//! hashing / IHT variant).
//!
//! A lookup table only learns from *exact-key collisions*; rich raw
//! input destroys collisions, so a tabular learner over a real-valued
//! observation would learn nothing (`emergent_cognition_rebuild.md` §2).
//! Tile coding fixes that: instead of one ultra-fine grid it lays down
//! `num_tilings` **coarse** grids, each offset asymmetrically. A
//! real-valued observation falls into one coarse tile of *each* grid;
//! those coarse tiles are big, so many similar observations share them
//! and get revisited — they converge, and a brand-new observation
//! inherits value from the neighbours that shared its tiles. That is
//! generalisation, deterministic and with no neural net.
//!
//! ## Per-block tiling
//!
//! The observation is a concatenation of semantically distinct blocks
//! (drives, body, perception, episodic memory, …). Hashing all of them
//! into **one** tile index per tiling is brittle: a noisy memory or
//! perception block then shifts the joint tile, so two observations
//! with identical drives but different memories share *no* tiles and
//! the clean drive signal never generalises.
//!
//! [`with_blocks`](TileCoder::with_blocks) fixes that — each block is
//! tiled **independently**, into its own FNV chain. Two observations
//! that agree on drives now share their drive-block tiles regardless of
//! what their perception/memory blocks do, so each block's signal
//! generalises on its own terms. The active-feature count becomes
//! `num_tilings × n_blocks`. With no `with_blocks` call the coder has a
//! single block spanning the whole observation — the classic behaviour.
//!
//! ## Name-seeded block identity
//!
//! On the production path a block's hash chain seeds on its **name**
//! ([`FeatureBlock::name`] via [`TileCoder::from_layout`], or
//! [`Encoder::block`]'s `name` argument) and its quantisation offsets are
//! **block-local** — so a block's tiles depend only on its own name,
//! values, and tilings, never on its position. Inserting, removing, or
//! reordering blocks re-keys nothing else; only renaming/resizing a block
//! re-keys that block. [`FeatureLayout::schema_fingerprint`] folds the
//! ordered `(name, width)` rows into the schema identity a persisted
//! learner carries.
//!
//! ## Determinism
//!
//! Tile indices are produced by pure integer/float arithmetic + a
//! fixed-seed FNV-1a hash — no `HashMap` iteration, no process-random
//! seed. Identical `(observation, action)` ⇒ identical indices across
//! runs and builds.

use crate::action_id::ActionTemplateId;
use crate::hash::{fnv1a_mix, fnv1a_str, FNV_OFFSET_64};
use crate::observation::Observation;
use serde::{Deserialize, Serialize};

/// One contiguous observation block — `[start, start + len)` — tiled as
/// its own independent feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
struct Block {
    start: u32,
    len: u32,
    /// How many of the coder's tilings this block uses — its *capacity*.
    /// A block uses tilings `0..tilings`, so a decision-critical block
    /// (drives) can take the full set for fine resolution while a noisy
    /// block (memory) takes fewer, capping how many tile-weights its
    /// churn can perturb (proposal §7.1). `#[serde(default)]` so a coder
    /// saved before per-block capacity reads `0` — normalised to the
    /// coder's full `num_tilings` by [`block_tilings`](TileCoder::block_tilings).
    #[serde(default)]
    tilings: u32,
    /// The block's **identity seed** — `fnv1a_str(name)` for a named block
    /// ([`TileCoder::from_layout`] / [`Encoder::block`]), or a positional
    /// fold for the name-free convenience constructors. Folded into every
    /// tile hash of this block instead of the block's *position*, so
    /// inserting or reordering blocks never re-keys an unrelated named
    /// block's tiles (and two blocks with identical content stay in
    /// disjoint tiles). Persisted with the coder: a saved learner keeps
    /// hashing exactly as it trained.
    name_seed: u64,
}

/// One contiguous feature block in a [`FeatureLayout`] — a span of the
/// observation vector tiled independently, at its own capacity, under its
/// own **name**.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FeatureBlock {
    /// The block's stable identity. The tile hash seeds on
    /// `fnv1a_str(name)` (not the block's position), so renaming a block
    /// re-keys *its own* tiles only, and inserting/reordering blocks never
    /// re-keys the others. Names must be unique within a layout — two
    /// blocks sharing a name would share tiles.
    pub name: &'static str,
    /// Start index into the observation vector.
    pub start: usize,
    /// Number of dimensions in this block.
    pub len: usize,
    /// How many of the coder's tilings this block uses (`1..=num_tilings`) —
    /// finer on decision-critical blocks, coarser on noisy ones.
    pub tilings: u32,
}

/// A **game-agnostic** description of how to build a block-structured tile
/// coder: the overall grid resolution plus the contiguous feature blocks
/// (each tiled independently). A game supplies one of these to
/// [`TileCoder::from_layout`]; the coder math never needs to know what the
/// blocks *mean* (drives, ghosts, pellets, …) — only their spans and
/// capacities. This is the seam that lets the same learner serve any
/// environment: each game owns its `FeatureLayout`, nothing here is specific
/// to one observation schema.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureLayout {
    /// Coder's full tiling count. A block may use fewer via its capacity.
    pub num_tilings: u32,
    /// Uniform tile width applied to every observation dimension.
    pub tile_width: f32,
    /// Total observation width — the sum of every block's `len`, and the
    /// coder's dimensionality.
    pub total_width: usize,
    /// Bounded index-hash-table size (weights vector length).
    pub iht_size: u32,
    /// Contiguous feature blocks. Must partition `[0, total_width)`:
    /// gap-free, ascending, summing to `total_width`.
    pub blocks: Vec<FeatureBlock>,
}

impl FeatureLayout {
    /// A deterministic identity of the **observation schema** this layout
    /// describes: an FNV-1a fold over the ordered `(name, width)` rows.
    ///
    /// This is the value a persisted learner carries so a load can refuse
    /// a save trained over a different schema (see `q_learning::Learner`).
    /// Deliberately *only* names + widths, in order:
    /// - renaming, resizing, inserting, removing, or reordering a block
    ///   changes the fingerprint — every change that alters what the
    ///   observation vector *means* positionally;
    /// - tiling capacities / tile width / IHT size do **not** — they are
    ///   baked into the saved coder itself and stay self-consistent with
    ///   its weights, and they don't change the observation vector;
    /// - a width is the block's **reserved** width, so zero-padded
    ///   headroom (e.g. the drives block's spare slots) absorbs small
    ///   growth without a fingerprint change.
    pub fn schema_fingerprint(&self) -> u64 {
        let mut h = FNV_OFFSET_64;
        for b in &self.blocks {
            h = fnv1a_mix(h, fnv1a_str(b.name));
            h = fnv1a_mix(h, b.len as u64);
        }
        h
    }
}

/// Maps a real-valued observation + action to a small set of active
/// tile indices into a bounded hash space.
///
/// ```
/// use rl_core::{TileCoder, Observation, ActionTemplateId};
///
/// // 8 tilings, tile width 0.25, a 2-dim observation, 1024-slot hash space.
/// let coder = TileCoder::uniform(8, 0.25, 2, 1024);
/// let step = ActionTemplateId(0);
///
/// let a = Observation::from_values(vec![0.10, 0.20]);
/// let near = Observation::from_values(vec![0.11, 0.21]); // same coarse tiles
/// let far = Observation::from_values(vec![0.90, 0.90]); // different region
///
/// // Deterministic: identical input round-trips to identical active tiles.
/// assert_eq!(coder.active(&a, step), coder.active(&a, step));
/// // One active tile per tiling.
/// assert_eq!(coder.active(&a, step).len(), 8);
///
/// // Generalisation: a nearby observation shares most tiles; a far one shares few.
/// let shared = |x: &Observation| {
///     let ta = coder.active(&a, step);
///     coder.active(x, step).iter().filter(|t| ta.contains(t)).count()
/// };
/// assert!(shared(&near) > shared(&far));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TileCoder {
    /// Number of overlapping tilings — more ⇒ finer effective
    /// resolution + smoother generalisation.
    num_tilings: u32,
    /// Uniform tile width, applied to every observation dimension. The
    /// asymmetric per-`(tiling, dimension)` offset is derived from this on
    /// the fly (see [`offset`](Self::offset)) rather than stored in a table,
    /// so the coder never needs to know the total observation width:
    /// [`encode`](Self::encode) discovers it per call.
    tile_width: f32,
    /// Bounded index space — tile indices are taken mod this.
    iht_size: u32,
    /// The observation blocks for the flat-[`Observation`] path
    /// ([`prepare`](Self::prepare)), each tiled independently. Set by
    /// [`uniform`](Self::uniform) / [`with_blocks`](Self::with_blocks) /
    /// [`from_layout`](Self::from_layout). Empty on a coder built with
    /// [`new`](Self::new) for the streaming [`encode`](Self::encode) path,
    /// which supplies its blocks per call. `#[serde(default)]` for
    /// forward-compatible deserialisation.
    #[serde(default)]
    blocks: Vec<Block>,
}

impl TileCoder {
    /// Build a **width-agnostic** coder: a uniform `tile_width`, an FNV
    /// index space of `iht_size`, and *no* blocks. Feed it observations via
    /// [`encode`](Self::encode), which streams the blocks per call — so the
    /// coder never needs to know the total observation width. For the flat
    /// [`Observation`] path use [`uniform`](Self::uniform) /
    /// [`from_layout`](Self::from_layout), which set the blocks up front.
    pub fn new(num_tilings: u32, tile_width: f32, iht_size: u32) -> Self {
        assert!(num_tilings >= 1, "a tile coder needs at least one tiling");
        assert!(iht_size >= 1);
        assert!(tile_width > 0.0, "tile width must be positive");
        Self {
            num_tilings,
            tile_width,
            iht_size,
            blocks: Vec::new(),
        }
    }

    /// A flat-[`Observation`] coder of `n_dims` dimensions tiled as a single
    /// block — convenience for tests / starter configs / toy environments
    /// that hold a whole `Observation` and call [`prepare`](Self::prepare).
    /// (For per-block tiling build via [`with_blocks`](Self::with_blocks) or
    /// [`from_layout`](Self::from_layout) instead.)
    pub fn uniform(num_tilings: u32, width: f32, n_dims: usize, iht_size: u32) -> Self {
        Self::new(num_tilings, width, iht_size).with_block_capacities(vec![(
            0,
            n_dims,
            num_tilings,
        )])
    }

    /// Tile each `(start, len)` block independently — the per-block
    /// tiling that keeps a noisy block from corrupting a clean one. Every
    /// block uses the coder's full `num_tilings`. The blocks must
    /// **partition** `[0, observation width)`: contiguous, gap-free, in
    /// ascending order. Blocks are identified by **position** (see
    /// [`with_block_capacities`](Self::with_block_capacities)).
    pub fn with_blocks(self, blocks: Vec<(usize, usize)>) -> Self {
        // A block with no explicit capacity gets the coder's full tiling
        // count — the classic "every block tiled equally" behaviour.
        let full = self.num_tilings;
        self.with_block_capacities(blocks.into_iter().map(|(s, l)| (s, l, full)).collect())
    }

    /// Tile each `(start, len, tilings)` block independently, with a
    /// **per-block capacity** — `tilings` is how many of the coder's
    /// tilings that block uses (the proposal's §7.1 noise containment:
    /// finer on decision-critical blocks like drives, coarser on noisy
    /// ones like memory). Each `tilings` must be in `1..=num_tilings`.
    /// The blocks must **partition** `[0, observation width)`: contiguous,
    /// gap-free, in ascending order.
    ///
    /// Blocks here are identified by **position** (the seed is a fold of
    /// the block index) — fine for the fixed test/toy schemas this
    /// convenience serves. The production path is name-stable: build via
    /// [`from_layout`](Self::from_layout) (or hash via a named
    /// [`Encoder`]), where each block seeds on its *name* so layout
    /// evolution never re-keys unrelated blocks.
    pub fn with_block_capacities(self, blocks: Vec<(usize, usize, u32)>) -> Self {
        let seeded = blocks
            .into_iter()
            .enumerate()
            .map(|(i, (s, l, t))| (positional_seed(i), s, l, t))
            .collect();
        self.with_seeded_blocks(seeded)
    }

    /// Build a block-structured coder from a game-agnostic [`FeatureLayout`].
    /// This is the reuse seam: any environment (this voxel sim, a gridworld, a
    /// Pac-Man clone) describes its features as a `FeatureLayout` and gets the
    /// same coder machinery — no game-specific schema is baked in here. Each
    /// block's tile hashes seed on `fnv1a_str(block.name)`, so a layout edit
    /// re-keys only the blocks it actually renames/moves-within (see
    /// [`FeatureBlock::name`]). Inherits the partition/capacity invariants of
    /// the block constructors (panics identically on a malformed layout).
    pub fn from_layout(layout: &FeatureLayout) -> Self {
        Self::new(layout.num_tilings, layout.tile_width, layout.iht_size).with_seeded_blocks(
            layout
                .blocks
                .iter()
                .map(|b| (fnv1a_str(b.name), b.start, b.len, b.tilings))
                .collect(),
        )
    }

    /// The shared block-assembly core: each entry is `(name_seed, start,
    /// len, tilings)`. Enforces the partition + capacity invariants stated
    /// on the public constructors.
    fn with_seeded_blocks(mut self, blocks: Vec<(u64, usize, usize, u32)>) -> Self {
        assert!(!blocks.is_empty(), "need at least one block");
        let mut cursor = 0usize;
        let mut out = Vec::with_capacity(blocks.len());
        for (name_seed, start, len, tilings) in blocks {
            assert_eq!(start, cursor, "blocks must be contiguous and gap-free");
            assert!(len > 0, "a block must be non-empty");
            assert!(
                tilings >= 1 && tilings <= self.num_tilings,
                "a block's tiling capacity must be in 1..=num_tilings",
            );
            out.push(Block {
                start: start as u32,
                len: len as u32,
                tilings,
                name_seed,
            });
            cursor += len;
        }
        // The blocks now *define* the observation width (`cursor`); there is
        // no separately-stored width to cross-check against.
        self.blocks = out;
        self
    }

    /// The observation blocks for the flat [`prepare`](Self::prepare) path.
    /// Empty for a streaming [`encode`](Self::encode) coder, which supplies
    /// its blocks per call.
    fn blocks(&self) -> &[Block] {
        &self.blocks
    }

    /// A block's tiling capacity, normalising the `0` a pre-capacity
    /// coder deserialises to up to the coder's full `num_tilings`.
    #[inline]
    fn block_tilings(&self, block: &Block) -> u32 {
        if block.tilings == 0 {
            self.num_tilings
        } else {
            block.tilings
        }
    }

    /// Number of tilings.
    pub fn num_tilings(&self) -> u32 {
        self.num_tilings
    }

    /// Bounded hash-space size.
    pub fn iht_size(&self) -> u32 {
        self.iht_size
    }

    /// The flat observation width this coder expects on the [`prepare`](Self::prepare)
    /// path — the sum of its block lengths. Zero for a streaming-only
    /// ([`new`](Self::new)) coder that has no blocks.
    pub fn observation_width(&self) -> usize {
        self.blocks.iter().map(|b| b.len as usize).sum()
    }

    /// The active tile indices for `(obs, action)` — one per
    /// `(tiling, block)` pair. Deterministic. The observation's width
    /// must match the coder's.
    ///
    /// Equivalent to `active_prepared(&self.prepare(obs), action)` — use
    /// the two-step form directly when the same observation is queried
    /// against many actions (the planner / `best_for_observation` hot
    /// path): [`prepare`](Self::prepare) does the per-observation
    /// hashing **once**, then each action costs a single extra hash mix.
    pub fn active(&self, obs: &Observation, action: ActionTemplateId) -> Vec<u32> {
        self.active_prepared(&self.prepare(obs), action)
    }

    /// Hash the observation once, deferring the action.
    ///
    /// Per `(tiling, block)` the FNV-1a chain mixes
    /// `tiling → block id → block coords → action`, so everything up to
    /// (and excluding) the action is observation-only and reused across
    /// every action. [`PreparedObservation`] caches that partial hash;
    /// [`active_prepared`](Self::active_prepared) finishes it.
    pub fn prepare(&self, obs: &Observation) -> PreparedObservation {
        self.debug_check_width(obs);
        self.hash_blocks(obs.as_slice(), &self.blocks)
    }

    /// Hash a flat value buffer partitioned into `blocks` — the shared core
    /// of [`prepare`](Self::prepare) (blocks from `self.blocks`) and
    /// [`encode`](Self::encode) (blocks streamed into an [`Encoder`]). Both
    /// land in the SAME per-`(block, tiling)` FNV chain, so a streamed
    /// observation and the equivalent flat one prepare byte-identically.
    fn hash_blocks(&self, values: &[f32], blocks: &[Block]) -> PreparedObservation {
        let mut base =
            Vec::with_capacity(blocks.iter().map(|b| self.block_tilings(b) as usize).sum());
        // Block-outer / tiling-inner: each block contributes only its own
        // capacity (`block_tilings`) of tilings, so a coarser block adds
        // fewer features to the active set.
        for block in blocks {
            debug_assert!(
                self.block_tilings(block) <= self.num_tilings,
                "a block's tiling capacity must not exceed the coder's num_tilings",
            );
            for t in 0..self.block_tilings(block) as usize {
                base.push(self.fold_block(self.block_seed(t, block.name_seed), t, *block, values));
            }
        }
        PreparedObservation { base }
    }

    /// Hash an observation supplied by a closure that streams feature blocks
    /// into an [`Encoder`] — the schema-free, width-free path. Each
    /// `enc.block(name, tilings, &values)` call is one independently-tiled
    /// block, in push order. The result is a [`PreparedObservation`]
    /// byte-identical to [`prepare`](Self::prepare) over the equivalent
    /// [`FeatureLayout`] (same names, same blocks, same order), so the two
    /// paths are interchangeable — see the `encode_matches_prepare` test.
    pub fn encode(&self, f: impl FnOnce(&mut Encoder)) -> PreparedObservation {
        let mut enc = Encoder::default();
        f(&mut enc);
        self.hash_blocks(&enc.values, &enc.blocks)
    }

    #[inline]
    fn debug_check_width(&self, obs: &Observation) {
        let width: usize = self.blocks.iter().map(|b| b.len as usize).sum();
        debug_assert_eq!(
            obs.len(),
            width,
            "Observation width must match the TileCoder's blocks",
        );
    }

    /// The per-`(tiling, block)` FNV seed — `tiling` then the block's
    /// **name seed** folded in, before any observation coords. Seeding on
    /// the name (not the block's position) keeps two blocks with identical
    /// content in disjoint tiles AND keeps a named block's tiles stable
    /// when other blocks are inserted/reordered around it.
    #[inline]
    fn block_seed(&self, t: usize, name_seed: u64) -> u64 {
        fnv1a_mix(fnv1a_mix(FNV_OFFSET_64, t as u64), name_seed)
    }

    /// Fold every dimension of `block` into accumulator `h` for tiling `t`.
    /// The asymmetric grid offset is indexed by the **block-local**
    /// dimension, so a block's quantisation — like its hash seed — is
    /// independent of where the block sits in the observation: moving a
    /// named block (because another was inserted before it) keeps its
    /// tiles bit-identical.
    #[inline]
    fn fold_block(&self, mut h: u64, t: usize, block: Block, values: &[f32]) -> u64 {
        for (local_d, d) in (block.start as usize..(block.start + block.len) as usize).enumerate() {
            // Quantise: which tile of grid `t` does dimension `d` fall in?
            // `floor` after the asymmetric (on-the-fly) offset.
            let coord = ((values[d] + self.offset(t, local_d)) / self.tile_width).floor() as i64;
            h = fnv1a_mix(h, coord as u64);
        }
        h
    }

    /// The asymmetric grid displacement for tiling `t`, **block-local**
    /// dimension `d`, computed on the fly. Tiling `t` shifts dimension `d`
    /// by an odd-multiple fraction of the tile width, so no two tilings'
    /// grid lines coincide; `fract` keeps it inside one tile width. Indexed
    /// block-locally (see [`fold_block`](Self::fold_block)) so a block's
    /// quantisation never depends on its position in the observation.
    #[inline]
    fn offset(&self, t: usize, d: usize) -> f32 {
        let frac = ((t as f64) * ((2 * d + 1) as f64) / (self.num_tilings as f64)).fract();
        (frac as f32) * self.tile_width
    }

    /// The active tile indices for `action`, given an observation
    /// already hashed by [`prepare`](Self::prepare) — one FNV mix + one
    /// modulo per `(tiling, block)` slot.
    pub fn active_prepared(
        &self,
        prepared: &PreparedObservation,
        action: ActionTemplateId,
    ) -> Vec<u32> {
        self.active_indices(prepared, action).collect()
    }

    /// The active tile indices for `action` as a lazy iterator — the
    /// allocation-free form of [`active_prepared`](Self::active_prepared)
    /// for the value-function hot path (sum / scatter-add over the
    /// active weights without materialising a `Vec`).
    pub fn active_indices<'a>(
        &'a self,
        prepared: &'a PreparedObservation,
        action: ActionTemplateId,
    ) -> impl Iterator<Item = u32> + 'a {
        let iht = self.iht_size as u64;
        prepared
            .base
            .iter()
            .map(move |&h| (fnv1a_mix(h, action.0 as u64) % iht) as u32)
    }

    /// Number of active tile indices per `(observation, action)` query —
    /// the sum of every block's tiling capacity (`num_tilings × n_blocks`
    /// when every block uses the full capacity).
    pub fn active_count(&self) -> usize {
        self.blocks()
            .iter()
            .map(|b| self.block_tilings(b) as usize)
            .sum()
    }

    /// The observation-schema identity of this coder's blocks — the FNV-1a
    /// fold over the ordered `(name seed, width)` rows. **Identical** to
    /// [`FeatureLayout::schema_fingerprint`] for a coder built via
    /// [`from_layout`](Self::from_layout) (the seed *is* `fnv1a_str(name)`),
    /// which is what lets a `Learner` stamp its schema at construction from
    /// the coder alone. The empty fold (`FNV_OFFSET_64`) for a blockless
    /// streaming coder ([`new`](Self::new)).
    pub fn schema_fingerprint(&self) -> u64 {
        let mut h = FNV_OFFSET_64;
        for b in &self.blocks {
            h = fnv1a_mix(h, b.name_seed);
            h = fnv1a_mix(h, b.len as u64);
        }
        h
    }
}

/// The identity seed for a block declared **without** a name (the
/// `uniform` / `with_blocks` / `with_block_capacities` conveniences): a
/// fold of the block's position. Such coders keep the classic
/// position-identified behaviour; name-stable hashing requires
/// [`TileCoder::from_layout`] or a named [`Encoder`].
#[inline]
fn positional_seed(index: usize) -> u64 {
    fnv1a_mix(FNV_OFFSET_64, index as u64)
}

/// The streaming feature sink an environment writes into — the *`Hasher`* in
/// the `Hash`/`Hasher` analogy. A type describes its observation by pushing
/// independently-tiled [`block`](Encoder::block)s of real values; the coder
/// folds them into tiles. The flat vector never materialises as anything a
/// caller can see or maintain — exactly as you never build a byte buffer to
/// `Hash` a struct. Drive one via [`TileCoder::encode`].
#[derive(Debug, Default)]
pub struct Encoder {
    /// Streamed values, concatenated in push order.
    values: Vec<f32>,
    /// One entry per pushed block: its span in `values`, tiling capacity,
    /// and name seed.
    blocks: Vec<Block>,
    /// The pushed blocks' names, parallel to `blocks` — carried so
    /// [`feature_layout`](Self::feature_layout) can hand them on.
    names: Vec<&'static str>,
}

impl Encoder {
    /// Push one independently-tiled **named** block of `values`, at
    /// `tilings` resolution (`1..=num_tilings`: finer on decision-critical
    /// blocks, coarser on noisy ones — the per-block tiling described at
    /// the module level). The block's *offset* falls out of push order,
    /// but its tile-hash identity is the `name` — so inserting or
    /// reordering blocks re-keys nothing else, and only a rename re-keys
    /// this block. Names must be unique within one observation.
    pub fn block(&mut self, name: &'static str, tilings: u32, values: &[f32]) {
        assert!(tilings >= 1, "a block needs at least one tiling");
        let start = self.values.len() as u32;
        self.values.extend_from_slice(values);
        self.blocks.push(Block {
            start,
            len: values.len() as u32,
            tilings,
            name_seed: fnv1a_str(name),
        });
        self.names.push(name);
    }

    /// A one-value block — `self.block(name, tilings, &[v])`.
    pub fn scalar(&mut self, name: &'static str, tilings: u32, v: f32) {
        self.block(name, tilings, &[v]);
    }

    /// Open one named block of `width` **zeroed** slots and hand the slice
    /// to the caller to fill — the registry-friendly twin of
    /// [`block`](Self::block): a declarative `(name, width)` row plus a
    /// pure filler function, with zero-fill as the default for reserved
    /// (spare) slots. Equivalent to `block(name, tilings, &buf)` for a
    /// `buf` the filler wrote.
    pub fn begin_block(&mut self, name: &'static str, tilings: u32, width: usize) -> &mut [f32] {
        assert!(tilings >= 1, "a block needs at least one tiling");
        assert!(width > 0, "a block must be non-empty");
        let start = self.values.len();
        self.values.resize(start + width, 0.0);
        self.blocks.push(Block {
            start: start as u32,
            len: width as u32,
            tilings,
            name_seed: fnv1a_str(name),
        });
        self.names.push(name);
        &mut self.values[start..]
    }

    /// Stream a [`Featurize`] value's blocks into this encoder.
    pub fn observe<T: Featurize + ?Sized>(&mut self, x: &T) {
        x.featurize(self);
    }

    /// The [`FeatureLayout`] described by the blocks streamed so far — the
    /// flat-path coder recipe for the observation this encoder built. Lets the
    /// *same* `featurize` code that fills an observation also define the
    /// coder's block structure: run featurize once on any state, read the
    /// layout off it. No separately-maintained layout, no desync.
    pub fn feature_layout(
        &self,
        num_tilings: u32,
        tile_width: f32,
        iht_size: u32,
    ) -> FeatureLayout {
        FeatureLayout {
            num_tilings,
            tile_width,
            iht_size,
            total_width: self.values.len(),
            blocks: self
                .blocks
                .iter()
                .zip(&self.names)
                .map(|(b, name)| FeatureBlock {
                    name,
                    start: b.start as usize,
                    len: b.len as usize,
                    tilings: b.tilings,
                })
                .collect(),
        }
    }

    /// Consume the encoder into the flat [`Observation`] it accumulated — the
    /// values streamed via [`block`](Self::block), in push order.
    pub fn into_observation(self) -> Observation {
        Observation::from_values(self.values)
    }

    /// The streamed values so far (push order) — e.g. to refill an existing
    /// `Observation`'s buffer in place without reallocating.
    pub fn values(&self) -> &[f32] {
        &self.values
    }
}

/// A type that describes itself to a tile coder by streaming feature blocks
/// into an [`Encoder`] — the `Hash` to the coder's `Hasher`. The flat
/// encoding is the coder's private business; an implementor pushes named,
/// typed values and never authors (or even sees) an offset or a width.
pub trait Featurize {
    /// Push this value's feature blocks into `enc`.
    fn featurize(&self, enc: &mut Encoder);
}

/// An observation hashed by [`TileCoder::prepare`] — the per-`(block,
/// tiling)` partial FNV state with the observation coords folded in but
/// the action not yet mixed. Reusing one of these across many actions is
/// the difference between O(actions × observation-width) and
/// O(observation-width + actions) hashing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedObservation {
    /// Per-`(block, tiling)` partial FNV-1a hash, action not yet mixed.
    /// `len == active_count` (`Σ block tiling-capacities`).
    base: Vec<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test observation width — fixed here so the generic crate has no
    /// game dependency.
    const WIDTH: usize = 199;
    /// Test drive-block width.
    const DRIVES_W: usize = 14;

    fn coder() -> TileCoder {
        TileCoder::uniform(16, 0.25, WIDTH, 1 << 16)
    }

    /// A coder with the drive block (dims `[0, DRIVES_W)`) tiled
    /// separately from the rest — the shape `coder_from_config` builds.
    fn blocked_coder() -> TileCoder {
        TileCoder::uniform(16, 0.25, WIDTH, 1 << 16)
            .with_blocks(vec![(0, DRIVES_W), (DRIVES_W, WIDTH - DRIVES_W)])
    }

    fn zeros() -> Observation {
        Observation::zeros(WIDTH)
    }

    fn obs_with(dim: usize, value: f32) -> Observation {
        let mut o = zeros();
        o.as_mut_slice()[dim] = value;
        o
    }

    #[test]
    fn active_returns_one_index_per_tiling_for_a_single_block() {
        let c = coder();
        let idx = c.active(&zeros(), ActionTemplateId(0));
        assert_eq!(idx.len(), 16);
        assert!(idx.iter().all(|i| *i < c.iht_size()));
    }

    #[test]
    fn active_returns_one_index_per_tiling_block_pair() {
        // Two blocks × 16 tilings ⇒ 32 active features.
        let c = blocked_coder();
        let idx = c.active(&zeros(), ActionTemplateId(0));
        assert_eq!(idx.len(), 32);
        assert!(idx.iter().all(|i| *i < c.iht_size()));
    }

    #[test]
    fn identical_inputs_are_deterministic() {
        let c = coder();
        let o = obs_with(5, 0.6);
        assert_eq!(
            c.active(&o, ActionTemplateId(2)),
            c.active(&o, ActionTemplateId(2))
        );
    }

    #[test]
    fn different_action_changes_the_tiles() {
        let c = coder();
        let o = zeros();
        assert_ne!(
            c.active(&o, ActionTemplateId(0)),
            c.active(&o, ActionTemplateId(1))
        );
    }

    #[test]
    fn a_far_apart_observation_shares_no_tiles() {
        // Two observations many tile-widths apart land in disjoint tiles.
        let c = coder();
        let near = obs_with(0, 0.0);
        let far = obs_with(0, 100.0);
        let a = c.active(&near, ActionTemplateId(0));
        let b = c.active(&far, ActionTemplateId(0));
        assert!(
            a.iter().all(|i| !b.contains(i)),
            "distant observations must not collide"
        );
    }

    #[test]
    fn per_block_tiling_isolates_a_noisy_block() {
        // Two observations agree on the drive block but differ wildly in
        // the (non-deferred) rest. With per-block tiling their drive-block
        // tiles must still coincide — the whole point of the feature.
        let c = blocked_coder();
        let mut a = obs_with(2, 0.5); // a drive-block dim
        a.as_mut_slice()[100] = 0.0; // a non-drive dim
        let mut b = a.clone();
        b.as_mut_slice()[100] = 7.0; // far apart in the non-drive block only
        let ia = c.active(&a, ActionTemplateId(0));
        let ib = c.active(&b, ActionTemplateId(0));
        // The first 16 features are the drive block (block 0, all tilings);
        // they must be identical.
        let shared = ia.iter().filter(|i| ib.contains(i)).count();
        assert!(
            shared >= 16,
            "the shared drive block must keep its tiles despite a noisy \
             other block — got {shared} shared",
        );
    }

    #[test]
    fn per_block_capacity_sets_the_per_block_tile_count() {
        // Drives take all 16 tilings (fine), the rest takes 4 (coarse).
        // The active set is the sum of the per-block capacities.
        let c = TileCoder::uniform(16, 0.25, WIDTH, 1 << 16)
            .with_block_capacities(vec![(0, DRIVES_W, 16), (DRIVES_W, WIDTH - DRIVES_W, 4)]);
        assert_eq!(c.active_count(), 16 + 4);
        let idx = c.active(&zeros(), ActionTemplateId(0));
        assert_eq!(idx.len(), 20, "16 fine drive tiles + 4 coarse memory tiles");
    }

    #[test]
    fn full_capacity_blocks_match_with_blocks() {
        // `with_blocks` is `with_block_capacities` at full capacity — the
        // two must produce byte-identical active sets.
        let blocks = vec![(0, DRIVES_W), (DRIVES_W, WIDTH - DRIVES_W)];
        let via_blocks = TileCoder::uniform(16, 0.25, WIDTH, 1 << 16).with_blocks(blocks.clone());
        let via_caps = TileCoder::uniform(16, 0.25, WIDTH, 1 << 16)
            .with_block_capacities(blocks.into_iter().map(|(s, l)| (s, l, 16)).collect());
        let o = obs_with(3, 0.7);
        assert_eq!(
            via_blocks.active(&o, ActionTemplateId(1)),
            via_caps.active(&o, ActionTemplateId(1)),
        );
    }

    /// An arbitrary (non-voxel) two-block layout — the game-agnostic reuse
    /// seam fixture shared by the `from_layout` tests below.
    fn arbitrary_layout() -> FeatureLayout {
        FeatureLayout {
            num_tilings: 8,
            tile_width: 0.25,
            total_width: 6,
            iht_size: 1 << 12,
            blocks: vec![
                FeatureBlock {
                    name: "pose",
                    start: 0,
                    len: 2,
                    tilings: 8,
                },
                FeatureBlock {
                    name: "goal",
                    start: 2,
                    len: 4,
                    tilings: 3,
                },
            ],
        }
    }

    #[test]
    fn from_layout_builds_a_deterministic_named_coder() {
        // A `FeatureLayout` is game-agnostic *data* — here an ARBITRARY
        // (non-voxel) 6-dim split no observation schema in this repo uses,
        // proving the reuse seam: any game can describe its own features and
        // get the same coder machinery. Two builds from the same layout are
        // identical, and the active set is the sum of per-block capacities.
        let layout = arbitrary_layout();
        let a = TileCoder::from_layout(&layout);
        let b = TileCoder::from_layout(&layout);
        assert_eq!(a, b, "same layout ⇒ identical coder");
        assert_eq!(a.active_count(), 8 + 3);
        assert_eq!(a.observation_width(), 6);
        // The coder's fingerprint IS the layout's (the name seeds carry over).
        assert_eq!(a.schema_fingerprint(), layout.schema_fingerprint());
    }

    #[test]
    fn renaming_a_block_rekeys_only_that_block() {
        // Name-seeded hashing: a rename re-keys the renamed block's tiles and
        // ONLY those — the other block's tiles are bit-identical.
        let base = arbitrary_layout();
        let mut renamed = base.clone();
        renamed.blocks[1].name = "target";
        let ca = TileCoder::from_layout(&base);
        let cb = TileCoder::from_layout(&renamed);
        let obs = Observation::from_values(vec![0.7, -0.3, 0.1, 0.9, 0.4, -1.2]);
        let ta = ca.active(&obs, ActionTemplateId(1));
        let tb = cb.active(&obs, ActionTemplateId(1));
        // Block 0 ("pose", 8 tilings) leads the active set in both coders.
        assert_eq!(&ta[..8], &tb[..8], "the untouched block keeps its tiles");
        assert_ne!(&ta[8..], &tb[8..], "the renamed block is re-keyed");
    }

    #[test]
    fn inserting_a_block_keeps_every_other_named_block_s_tiles() {
        // THE name-seeding payoff: insert a new leading block and every
        // existing block's tiles — hash seed AND quantisation (block-local
        // offsets) — stay bit-identical, so a saved learner's weights for the
        // surviving blocks would still be addressed correctly.
        let base = arbitrary_layout();
        let mut grown = base.clone();
        grown.blocks.insert(
            0,
            FeatureBlock {
                name: "carrying",
                start: 0,
                len: 1,
                tilings: 8,
            },
        );
        grown.blocks[1].start = 1;
        grown.blocks[2].start = 3;
        grown.total_width = 7;

        let ca = TileCoder::from_layout(&base);
        let cb = TileCoder::from_layout(&grown);
        let vals = [0.7f32, -0.3, 0.1, 0.9, 0.4, -1.2];
        let obs_base = Observation::from_values(vals.to_vec());
        let mut grown_vals = vec![0.5]; // the new block's value — content irrelevant
        grown_vals.extend_from_slice(&vals);
        let obs_grown = Observation::from_values(grown_vals);

        let a = ActionTemplateId(3);
        let ta = ca.active(&obs_base, a);
        let tb = cb.active(&obs_grown, a);
        // Old active set: pose(8) + goal(3). New: carrying(8) + pose(8) + goal(3),
        // with pose/goal occupying the tail in the same order.
        assert_eq!(
            &ta[..],
            &tb[8..],
            "pre-existing blocks' tiles must survive the insertion bit-identically",
        );
        // And the fingerprint DID change — the schema is different.
        assert_ne!(base.schema_fingerprint(), grown.schema_fingerprint());
    }

    #[test]
    fn schema_fingerprint_reads_names_and_widths_in_order() {
        let base = arbitrary_layout();
        // Stable: same rows ⇒ same fingerprint.
        assert_eq!(
            base.schema_fingerprint(),
            arbitrary_layout().schema_fingerprint()
        );
        // A rename changes it.
        let mut renamed = base.clone();
        renamed.blocks[0].name = "stance";
        assert_ne!(base.schema_fingerprint(), renamed.schema_fingerprint());
        // A resize changes it.
        let mut resized = base.clone();
        resized.blocks[1].len = 5;
        resized.total_width = 7;
        assert_ne!(base.schema_fingerprint(), resized.schema_fingerprint());
        // A reorder changes it (the fold is ordered).
        let mut reordered = base.clone();
        reordered.blocks.swap(0, 1);
        reordered.blocks[0].start = 0;
        reordered.blocks[1].start = 4;
        assert_ne!(base.schema_fingerprint(), reordered.schema_fingerprint());
        // Tiling capacity does NOT change it — capacities are baked into the
        // saved coder itself and don't alter what the observation vector means.
        let mut retiled = base.clone();
        retiled.blocks[1].tilings = 7;
        assert_eq!(base.schema_fingerprint(), retiled.schema_fingerprint());
    }

    #[test]
    #[should_panic(expected = "1..=num_tilings")]
    fn block_capacity_above_num_tilings_panics() {
        let _ =
            TileCoder::uniform(8, 0.25, WIDTH, 1 << 16).with_block_capacities(vec![(0, WIDTH, 9)]);
    }

    #[test]
    fn a_nearby_observation_shares_some_tiles() {
        // Generalisation: a small nudge (< one tile width) keeps the
        // observation inside most of the same coarse tiles.
        let c = coder();
        let base = obs_with(0, 0.30);
        let nudged = obs_with(0, 0.32); // 0.02 << 0.25 tile width
        let a = c.active(&base, ActionTemplateId(0));
        let b = c.active(&nudged, ActionTemplateId(0));
        let shared = a.iter().filter(|i| b.contains(i)).count();
        assert!(
            shared > 0,
            "a sub-tile nudge must keep shared tiles (generalisation)"
        );
    }

    #[test]
    fn encode_matches_prepare() {
        // The hashing-identity guarantee: streaming blocks into `encode`
        // produces the byte-identical `PreparedObservation` as `prepare` over
        // the equivalent `FeatureLayout` (same blocks, same order, same
        // per-block tilings). This is what lets the sim move off the flat
        // schema onto the streaming encoder with NO change to learned weights.
        let num_tilings = 16;
        let tile_width = 0.25;
        let iht = 1 << 16;
        // An arbitrary multi-block split with mixed per-block capacities,
        // including a 1-D block (like a perception bearing axis).
        let layout = FeatureLayout {
            num_tilings,
            tile_width,
            total_width: 24,
            iht_size: iht,
            blocks: vec![
                FeatureBlock {
                    name: "drives",
                    start: 0,
                    len: 14,
                    tilings: 16,
                },
                FeatureBlock {
                    name: "bearing",
                    start: 14,
                    len: 1,
                    tilings: 4,
                },
                FeatureBlock {
                    name: "memory",
                    start: 15,
                    len: 9,
                    tilings: 7,
                },
            ],
        };
        let flat_coder = TileCoder::from_layout(&layout);
        let stream_coder = TileCoder::new(num_tilings, tile_width, iht);

        // A non-trivial value buffer (negatives, >1, sub-tile fractions).
        let vals: Vec<f32> = (0..24).map(|i| (i as f32) * 0.137 - 1.3).collect();

        let via_prepare = flat_coder.prepare(&Observation::from_values(vals.clone()));
        let via_encode = stream_coder.encode(|e| {
            e.block("drives", 16, &vals[0..14]);
            e.block("bearing", 4, &vals[14..15]);
            e.block("memory", 7, &vals[15..24]);
        });
        assert_eq!(
            via_prepare, via_encode,
            "encode() must produce byte-identical tiles to prepare() over the same blocks",
        );

        // And the action-mixed active sets match too (the full pipeline).
        let a = ActionTemplateId(3);
        assert_eq!(
            flat_coder.active_prepared(&via_prepare, a),
            stream_coder.active_prepared(&via_encode, a),
        );
    }

    #[test]
    fn begin_block_equals_block_and_zero_fills_the_spares() {
        // `begin_block` (zeroed slice handed to a filler) must be
        // byte-identical to `block` over a buffer the filler wrote — including
        // reserved slots the filler leaves at the zero-fill default.
        let mut via_begin = Encoder::default();
        {
            let v = via_begin.begin_block("drives", 4, 5);
            v[0] = 0.9;
            v[2] = -0.3;
            // v[1], v[3], v[4] stay reserved (zero).
        }
        via_begin.scalar("prox", 2, 0.7);

        let mut via_block = Encoder::default();
        via_block.block("drives", 4, &[0.9, 0.0, -0.3, 0.0, 0.0]);
        via_block.block("prox", 2, &[0.7]);

        assert_eq!(via_begin.values(), via_block.values());
        let lay = |e: &Encoder| e.feature_layout(8, 0.25, 1 << 12);
        assert_eq!(lay(&via_begin), lay(&via_block));
        assert_eq!(
            lay(&via_begin).schema_fingerprint(),
            lay(&via_block).schema_fingerprint()
        );
    }
}
