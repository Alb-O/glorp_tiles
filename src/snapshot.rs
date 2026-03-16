//! Immutable solved layout output and diagnostics.
//!
//! A [`Snapshot`] records the solved rectangles for one tree or session revision. It stores
//! rectangles for every node, deterministic split-allocation traces, a geometry fingerprint for
//! tree pairing, and any hard-limit violations observed in the solved leaf rectangles.
//!
//! Snapshots are inspectable data objects. When produced by [`crate::Session::solve`], they also
//! carry an internal live-session binding used to reject foreign geometry commands. That binding
//! is intentionally not part of the public serialized form or equality semantics, so deserialized
//! snapshots remain inspectable and tree-checkable but cannot directly drive session geometry
//! commands.

use {
	crate::{
		geom::{Axis, Rect},
		ids::{NodeId, Revision, SessionOwner},
		limits::WeightPair,
		tree::Tree,
	},
	serde::{
		Deserialize, Serialize,
		de::{self, Deserializer},
	},
	std::collections::BTreeMap,
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
/// points additionally bind the snapshot to one live session instance for geometry commands. Free
/// snapshots also carry a deterministic geometry fingerprint so low-level consumers can verify
/// they are being paired with the same tree state they were solved from.
///
/// Serialization preserves the revision, diagnostics, solved rectangles, and geometry
/// fingerprint. Deserialization intentionally restores an ownerless snapshot.
///
/// `node_rects` iterates in stable ascending [`NodeId`] order.
#[derive(Debug, Clone, Serialize)]
pub struct Snapshot {
	revision: Revision,
	root: Rect,
	tree_fingerprint: (u64, u64),
	node_rects: BTreeMap<NodeId, Rect>,
	split_traces: Vec<SplitTrace>,
	violations: Vec<Violation>,
	strict_feasible: bool,
	#[serde(skip, default)]
	owner: Option<SessionOwner>,
}

#[derive(Deserialize)]
struct SnapshotWire {
	revision: Revision,
	root: Rect,
	tree_fingerprint: (u64, u64),
	node_rects: BTreeMap<NodeId, Rect>,
	split_traces: Vec<SplitTrace>,
	violations: Vec<Violation>,
	strict_feasible: bool,
}

impl<'de> Deserialize<'de> for Snapshot {
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>, {
		let wire = SnapshotWire::deserialize(deserializer)?;
		if wire.strict_feasible != wire.violations.is_empty() {
			return Err(de::Error::custom(
				"snapshot strict_feasible must match whether violations is empty",
			));
		}
		Ok(Self {
			revision: wire.revision,
			root: wire.root,
			tree_fingerprint: wire.tree_fingerprint,
			node_rects: wire.node_rects,
			split_traces: wire.split_traces,
			violations: wire.violations,
			strict_feasible: wire.strict_feasible,
			// Persisted snapshots are data objects, not live session capabilities.
			owner: None,
		})
	}
}

impl PartialEq for Snapshot {
	fn eq(&self, other: &Self) -> bool {
		// Ownership is intentionally excluded so equality stays about solved geometry/diagnostics,
		// not which live session instance produced the snapshot.
		(
			self.revision,
			self.root,
			self.tree_fingerprint,
			&self.node_rects,
			&self.split_traces,
			&self.violations,
			self.strict_feasible,
		) == (
			other.revision,
			other.root,
			other.tree_fingerprint,
			&other.node_rects,
			&other.split_traces,
			&other.violations,
			other.strict_feasible,
		)
	}
}

impl Eq for Snapshot {}

impl Snapshot {
	#[must_use]
	pub(crate) fn new_unowned(
		revision: Revision, root: Rect, tree_fingerprint: (u64, u64), node_capacity: usize,
	) -> Self {
		Self {
			revision,
			root,
			tree_fingerprint,
			// We keep sorted storage for deterministic external iteration; only the vec fields
			// benefit from the node-count hint.
			node_rects: BTreeMap::new(),
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

	pub(crate) fn node_rects_mut(&mut self) -> &mut BTreeMap<NodeId, Rect> {
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

	/// Returns whether this snapshot was solved from the same geometry-affecting tree state.
	///
	/// This validates `tree` before comparing. Payload-only changes do not affect the match result,
	/// which makes this suitable for IDE-side geometry caches keyed independently from leaf
	/// payloads.
	pub fn matches_tree<T>(&self, tree: &Tree<T>) -> Result<bool, crate::ValidationError> {
		tree.validate()?;
		Ok(self.matches_tree_validated(tree))
	}

	pub(crate) fn matches_tree_validated<T>(&self, tree: &Tree<T>) -> bool {
		self.tree_fingerprint == tree.geometry_fingerprint_validated()
	}

	/// Returns the solved rectangles for every node reached during the solve.
	///
	/// This includes split nodes as well as leaves. Iteration order is stable ascending [`NodeId`]
	/// order, which makes it suitable for deterministic IDE-side caches and serialization.
	#[must_use]
	pub fn node_rects(&self) -> &BTreeMap<NodeId, Rect> {
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
