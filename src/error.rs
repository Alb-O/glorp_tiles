//! Public error taxonomy for validation, solving, navigation, and mutation.

use {
	crate::ids::NodeId,
	std::fmt::{Display, Formatter},
};

/// Structural or metadata validation failure for a tree or session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
	/// The declared root id is missing from the node map.
	MissingRoot(NodeId),
	/// The root node incorrectly points at a parent.
	RootHasParent(NodeId),
	/// A referenced node id is absent from storage.
	MissingNode(NodeId),
	/// Arithmetic overflow occurred while deriving or validating metadata.
	ArithmeticOverflow {
		/// Node whose derived value overflowed.
		node: NodeId,
		/// Summary or metadata field that overflowed.
		field: &'static str,
	},
	/// A node's stored parent disagrees with the expected structural parent.
	ParentMismatch {
		/// Node with the mismatched parent pointer.
		node: NodeId,
		/// Expected parent id, using `0` when no parent should exist.
		expected: NodeId,
		/// Actual stored parent pointer.
		actual: Option<NodeId>,
	},
	/// A split references the same child in both slots.
	DuplicateChild {
		/// Split node containing the duplicate child reference.
		split: NodeId,
		/// Child id reused in both slots.
		child: NodeId,
	},
	/// A cycle was detected during tree traversal.
	Cycle(NodeId),
	/// A node exists in storage but is unreachable from the root.
	Unreachable(NodeId),
	/// A split stores an invalid weight pair such as `(0, 0)`.
	InvalidWeights(NodeId),
	/// A leaf stores inconsistent hard limits or priorities.
	InvalidLeafLimits(NodeId),
	/// Session focus points to a split instead of a leaf.
	NonLeafFocus(NodeId),
	/// Session selection does not point to a valid node or focused subtree.
	InvalidSelection(NodeId),
	/// An empty tree/session has non-empty focus or selection state.
	EmptyStateInconsistent,
}

impl Display for ValidationError {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(f, "{self:?}")
	}
}

impl std::error::Error for ValidationError {}

/// Failure returned by direct solve entry points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveError {
	/// The input tree failed structural validation before solving began.
	Validation(ValidationError),
	/// Strict solving was requested but no hard-limit-feasible layout exists.
	Infeasible,
}

impl Display for SolveError {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(f, "{self:?}")
	}
}

impl std::error::Error for SolveError {}

/// Failure returned by geometry-based focus navigation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NavError {
	/// Navigation was requested on an empty session.
	Empty,
	/// The supplied snapshot revision does not match the current session revision.
	StaleSnapshot,
	/// The snapshot is missing a solved rectangle for the referenced node.
	MissingSnapshotRect(NodeId),
	/// No candidate leaf exists in the requested direction.
	NoCandidate,
	/// The underlying session state failed validation.
	Validation(ValidationError),
}

impl Display for NavError {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(f, "{self:?}")
	}
}

impl std::error::Error for NavError {}

/// Failure returned by structural or geometry-driven session mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpError {
	/// The operation requires a non-empty session.
	Empty,
	/// The operation requires an empty session.
	NonEmpty,
	/// The referenced node id does not exist.
	MissingNode(NodeId),
	/// The operation requires a leaf node but received a split.
	NotLeaf(NodeId),
	/// The operation requires a split node but received a leaf.
	NotSplit(NodeId),
	/// The operation requires a parent for the referenced node.
	NoParent(NodeId),
	/// The supplied snapshot revision does not match the current session revision.
	StaleSnapshot,
	/// The requested split weights were invalid, typically `(0, 0)`.
	InvalidWeights,
	/// The operation would create an ancestor/descendant structural conflict.
	AncestorConflict,
	/// The operation requires two distinct node ids.
	SameNode,
	/// The requested target lies inside the subtree currently being moved.
	TargetInsideSelection,
	/// The resulting session failed validation.
	Validation(ValidationError),
}

impl Display for OpError {
	fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
		write!(f, "{self:?}")
	}
}

impl std::error::Error for OpError {}
