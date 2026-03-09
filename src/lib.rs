//! Deterministic binary split tiling for editor, window, and TUI layouts.
//!
//! `libtiler` models a layout as a single-root binary split tree with exact half-open integer
//! rectangles. The crate is split into two layers: [`Tree`] exposes validated topology and leaf
//! metadata, while [`Session`] adds focus, selection, revision tracking, and geometry-driven
//! editing commands.
//!
//! Solving is deterministic and certifying. [`solve`] and [`Session::solve`] always produce a
//! [`Snapshot`] when the tree is representable, even if some leaves exceed hard limits; callers
//! inspect [`Snapshot::strict_feasible`] and [`Snapshot::violations`] to decide whether that
//! best-effort result is acceptable. [`solve_strict`] rejects any solve with hard-limit
//! violations.
//!
//! Geometry-dependent session commands require a fresh snapshot from the current session revision.
//! Reusing an older snapshot after a structural edit yields a stale-snapshot error instead of
//! silently acting on outdated rectangles.
//!
//! Public surface map:
//!
//! - Structure: [`Tree`], [`LeafNode`], [`SplitNode`], [`NodeId`]
//! - Solving: [`solve`], [`solve_strict`], [`SolverPolicy`], [`Snapshot`]
//! - Editing: [`Session`], [`PresetKind`], [`ResizeStrategy`], [`RebalanceMode`]
//! - Geometry and limits: [`Rect`], [`Axis`], [`Slot`], [`Direction`], [`LeafMeta`],
//!   [`SizeLimits`], [`WeightPair`]
//!
//! ```text
//! Tree<T>
//!   - validated topology
//!   - stable node ids
//!   - leaf metadata
//!       |
//!       v
//! solve / solve_strict
//!       |
//!       v
//! Snapshot
//!   - rects
//!   - traces
//!   - violations
//!       ^
//!       |
//! Session<T>
//!   - tree + focus + selection + revision
//!   - editing ops
//!   - geometry-driven ops require fresh Snapshot
//! ```
//!
//! ```text
//! Rect { x: 10, y: 5, w: 4, h: 3 }
//! covers x in [10, 14), y in [5, 8)
//! right/bottom edges are exclusive
//! ```
//!
//! Error model:
//!
//! - [`ValidationError`] reports invalid tree or session structure
//! - [`SolveError`] separates invalid input from strict infeasibility
//! - [`NavError`] and [`OpError`] report stale snapshots and invalid session operations
//!
//! Determinism:
//!
//! - leaf ids remain stable while those leaves survive; split ids remain stable until removed or
//!   rebuilt
//! - solving uses deterministic scoring and tie-breaking
//! - navigation and resize operations consume a specific snapshot revision
//!
//! ```
//! use libtiler::{
//!     Axis, Direction, LeafMeta, Rect, ResizeStrategy, Session, Slot, SolverPolicy,
//! };
//!
//! let mut session = Session::new();
//! let _main = session.insert_root("main", LeafMeta::default())?;
//! let _side = session.split_focus(Axis::X, Slot::B, "side", LeafMeta::default(), None)?;
//! let _log = session.wrap_selection(Axis::Y, Slot::B, "log", LeafMeta::default(), None)?;
//!
//! let root = Rect { x: 0, y: 0, w: 120, h: 40 };
//! let snapshot = session.solve(root, &SolverPolicy::default());
//! session.focus_dir(Direction::Right, &snapshot)?;
//! session.grow_focus(Direction::Down, 4, ResizeStrategy::Local, &snapshot)?;
//!
//! let solved = session.solve(root, &SolverPolicy::default());
//! assert!(solved.strict_feasible);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::bare_urls)]
#![forbid(unsafe_code)]

pub mod error;
pub mod geom;
pub mod ids;
pub mod limits;
pub mod nav;
pub mod preset;
pub mod resize;
pub mod session;
pub mod snapshot;
pub mod solver;
pub mod tree;

/// Error types returned by solving, validation, navigation, and session operations.
pub use error::{NavError, OpError, SolveError, ValidationError};
/// Geometry primitives and directional vocabulary used throughout the crate.
pub use geom::{Axis, Direction, Rect, Slot};
/// Opaque identifier for a node inside a single [`Tree`] or [`Session`].
pub type NodeId = ids::NodeId;
/// Monotonic session revision used to reject stale geometry snapshots.
pub use ids::Revision;
/// Leaf sizing metadata, subtree summaries, and split-weight helpers.
pub use limits::{LeafMeta, Priority, SizeLimits, Summary, WeightPair, canonicalize_weights};
/// Preset subtree shapes used by session-level rebuild operations.
pub use preset::{BalancedPreset, DwindlePreset, PresetKind, TallPreset, WidePreset};
/// Strategies for distributing geometry changes across eligible ancestor splits.
pub use resize::ResizeStrategy;
/// Focus-aware state machine built on top of a validated [`Tree`].
pub use session::{RebalanceMode, Session};
/// Solved layout output, scoring traces, and hard-limit violations.
pub use snapshot::{ScoreTuple, Snapshot, SplitTrace, Violation, ViolationKind};
/// Deterministic solver policy, helpers, and solving entry points.
pub use solver::{
    OverflowMode, PairSpec, ShortageMode, SolverPolicy, TieBreakMode, choose_extent,
    choose_extent_with_score, score, solve, solve_strict, solve_strict_with_revision,
    solve_with_revision, summarize,
};
/// Leaf node view returned by [`Tree::leaf`].
pub type LeafNode<T> = tree::LeafNode<T>;
/// Split node view returned by [`Tree::split`].
pub type SplitNode = tree::SplitNode;
/// Validated binary split topology with stable node ids and leaf metadata.
pub use tree::Tree;
