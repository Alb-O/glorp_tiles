//! Immutable solved layout output and diagnostics.
//!
//! A [`Snapshot`] records the solved rectangles for one tree or session revision. It stores
//! rectangles for every node, deterministic split-allocation traces, and any hard-limit
//! violations observed in the solved leaf rectangles.
//!
//! Snapshots are inspectable data objects. When produced by [`crate::Session::solve`], they also
//! carry an internal live-session binding used to reject foreign geometry commands. That binding
//! is intentionally not part of the public serialized form or equality semantics.

use {
	crate::{
		geom::{Axis, Rect},
		ids::{NodeId, Revision, SessionOwner},
		limits::WeightPair,
	},
	serde::{Deserialize, Serialize},
	std::collections::HashMap,
};

/// Lexicographic score used when comparing candidate split allocations.
///
/// This is solver output, not a user-facing configuration type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ScoreTuple {
	/// Penalty for falling below child minimum extents.
	pub shortage_penalty: u128,
	/// Penalty for exceeding child maximum extents.
	pub overflow_penalty: u128,
	/// Penalty for deviating from the requested weight ratio.
	pub preference_penalty: u128,
	/// Final deterministic tie-break component.
	pub tie_break: u128,
}

/// Deterministic diagnostic record for one solved split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitTrace {
	/// Split node whose extent was allocated.
	pub split: NodeId,
	/// Axis along which the split was solved.
	pub axis: Axis,
	/// Total extent available to the split on `axis`.
	pub total: u32,
	/// Extent assigned to child `A`.
	pub chosen_a: u32,
	/// Lexicographic score of the chosen allocation.
	pub score: ScoreTuple,
	/// Relative split weights used while solving this split.
	pub weights: WeightPair,
}

/// Kind of hard-limit violation recorded for a solved leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationKind {
	/// The solved width fell below `min_w`.
	MinWidth,
	/// The solved height fell below `min_h`.
	MinHeight,
	/// The solved width exceeded `max_w`.
	MaxWidth,
	/// The solved height exceeded `max_h`.
	MaxHeight,
}

/// Hard-limit violation observed for one solved leaf.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Violation {
	/// Leaf node that violated a hard limit.
	pub node: NodeId,
	/// Violated limit dimension and direction.
	pub kind: ViolationKind,
	/// Required minimum or maximum bound on the violated dimension.
	pub required: u32,
	/// Actual solved extent on the violated dimension.
	pub actual: u32,
}

/// Solved layout snapshot for one tree or session revision.
///
/// `node_rects` includes every solved node, not only leaves. `split_traces` records the
/// deterministic allocation decision for each split. `violations` records leaf hard-limit
/// violations discovered after solving, and `strict_feasible` is `true` exactly when
/// `violations.is_empty()`. Free solver entry points produce ownerless snapshots; session entry
/// points additionally bind the snapshot to one live session instance for geometry commands.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
	revision: Revision,
	root: Rect,
	node_rects: HashMap<NodeId, Rect>,
	split_traces: Vec<SplitTrace>,
	violations: Vec<Violation>,
	strict_feasible: bool,
	#[serde(skip, default)]
	owner: Option<SessionOwner>,
}

impl PartialEq for Snapshot {
	fn eq(&self, other: &Self) -> bool {
		// Ownership is intentionally excluded so equality stays about solved geometry/diagnostics,
		// not which live session instance produced the snapshot.
		self.revision == other.revision
			&& self.root == other.root
			&& self.node_rects == other.node_rects
			&& self.split_traces == other.split_traces
			&& self.violations == other.violations
			&& self.strict_feasible == other.strict_feasible
	}
}

impl Eq for Snapshot {}

impl Snapshot {
	#[must_use]
	pub(crate) fn new_unowned(revision: Revision, root: Rect, node_capacity: usize) -> Self {
		Self {
			revision,
			root,
			node_rects: HashMap::with_capacity(node_capacity),
			split_traces: Vec::with_capacity(node_capacity.saturating_sub(1)),
			violations: Vec::new(),
			strict_feasible: true,
			owner: None,
		}
	}

	pub(crate) fn set_owner(&mut self, owner: SessionOwner) {
		self.owner = Some(owner);
	}

	#[must_use]
	pub(crate) fn owner(&self) -> Option<SessionOwner> {
		self.owner
	}

	pub(crate) fn node_rects_mut(&mut self) -> &mut HashMap<NodeId, Rect> {
		&mut self.node_rects
	}

	pub(crate) fn split_traces_mut(&mut self) -> &mut Vec<SplitTrace> {
		&mut self.split_traces
	}

	pub(crate) fn violations_mut(&mut self) -> &mut Vec<Violation> {
		&mut self.violations
	}

	pub(crate) fn set_strict_feasible(&mut self, strict_feasible: bool) {
		self.strict_feasible = strict_feasible
	}

	/// Returns the revision this snapshot was solved for.
	#[must_use]
	pub fn revision(&self) -> Revision {
		self.revision
	}

	/// Returns the root rectangle passed to the solver.
	#[must_use]
	pub fn root(&self) -> Rect {
		self.root
	}

	/// Returns the solved rectangles for every node reached during the solve.
	#[must_use]
	pub fn node_rects(&self) -> &HashMap<NodeId, Rect> {
		&self.node_rects
	}

	/// Returns the deterministic split-allocation diagnostics in solve order.
	#[must_use]
	pub fn split_traces(&self) -> &[SplitTrace] {
		&self.split_traces
	}

	/// Returns the leaf hard-limit violations observed in the solved layout.
	#[must_use]
	pub fn violations(&self) -> &[Violation] {
		&self.violations
	}

	/// Returns whether the solve satisfied all hard limits.
	#[must_use]
	pub fn strict_feasible(&self) -> bool {
		self.strict_feasible
	}

	/// Returns the solved rectangle for `node`, if the snapshot contains one.
	///
	/// This works for any node represented in [`Self::node_rects`], including splits and leaves.
	#[must_use]
	pub fn rect(&self, node: NodeId) -> Option<Rect> {
		self.node_rects.get(&node).copied()
	}
}
