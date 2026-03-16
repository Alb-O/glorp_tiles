//! Geometry-based directional navigation.
//!
//! Navigation operates on solved leaf rectangles rather than tree topology. Candidates must lie in
//! the directional half-plane, then are ranked lexicographically by gap and alignment metrics.
//! Depth-first leaf order provides the final stable tie-break source.

use {
	crate::{
		error::NeighborError,
		geom::{Axis, Direction, Rect, orth_gap},
		ids::NodeId,
		snapshot::Snapshot,
		tree::Tree,
	},
	std::ops::ControlFlow,
};

/// Lexicographic navigation score for one candidate leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NavScore {
	/// Primary distance in the requested direction.
	pub primary_gap: u32,
	/// Gap between the candidates on the orthogonal axis.
	pub orth_gap: u32,
	/// Distance between orthogonal centers, using doubled coordinates for exact ordering.
	pub orth_center_delta: u64,
	/// Depth-first leaf order rank used as the final deterministic tie-break.
	pub tree_order_rank: usize,
}

/// Returns the best solved leaf neighbor of `current` in `dir`.
///
/// The tree is validated first. Empty trees return `Ok(None)`. Valid non-empty queries require
/// `current` to name a leaf in `tree`, and `snap` must contain solved rectangles for `current`
/// and every leaf in that same tree.
///
/// This is the checked low-level navigation helper for callers working directly with [`Tree`] and
/// free-solver snapshots. [`crate::Session::focus_dir`] layers live-session ownership and revision
/// checks on top of the same geometric ranking.
///
/// ```
/// use glorp_tiles::{Axis, Direction, LeafMeta, Rect, Slot, SolverPolicy, Tree, nav::best_neighbor, solve};
///
/// let mut tree = Tree::new();
/// let left = tree.insert_root("left", LeafMeta::default())?;
/// let right = tree.split_leaf(left, Axis::X, Slot::B, "right", LeafMeta::default(), None)?;
/// let snap = solve(&tree, Rect { x: 0, y: 0, w: 10, h: 4 }, &SolverPolicy::default())?;
///
/// assert_eq!(best_neighbor(&tree, &snap, left, Direction::Right)?, Some(right));
/// assert_eq!(best_neighbor(&tree, &snap, left, Direction::Left)?, None);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn best_neighbor<T>(
	tree: &Tree<T>, snap: &Snapshot, current: NodeId, dir: Direction,
) -> Result<Option<NodeId>, NeighborError> {
	tree.validate().map_err(NeighborError::Validation)?;
	let Some(root) = tree.root_id() else {
		return Ok(None);
	};
	if !tree.contains(current) {
		return Err(NeighborError::MissingNode(current));
	}
	if !tree.is_leaf(current) {
		return Err(NeighborError::NotLeaf(current));
	}
	let current_rect = snap.rect(current).ok_or(NeighborError::MissingSnapshotRect(current))?;
	// Checked low-level navigation treats a partial snapshot as invalid input rather than silently
	// changing "no candidate" behavior based on whichever leaves happen to be present.
	if let Some(id) = first_missing_leaf_rect(tree, snap, root) {
		return Err(NeighborError::MissingSnapshotRect(id));
	}
	Ok(best_neighbor_unchecked(tree, snap, root, current, current_rect, dir))
}

fn best_neighbor_unchecked<T>(
	tree: &Tree<T>, snap: &Snapshot, root: NodeId, current: NodeId, current_rect: Rect, dir: Direction,
) -> Option<NodeId> {
	let mut best = None;
	let mut rank = 0;
	let _ = tree.visit_leaves_dfs(root, &mut |id| {
		if id != current
			&& let Some(candidate_rect) = snap.rect(id)
			&& let Some(score) = nav_score(current_rect, candidate_rect, dir, rank)
			&& best.is_none_or(|(_, best_score)| score < best_score)
		{
			best = Some((id, score));
		}
		rank += 1;
		ControlFlow::<()>::Continue(())
	});
	best.map(|(id, _)| id)
}

fn first_missing_leaf_rect<T>(tree: &Tree<T>, snap: &Snapshot, root: NodeId) -> Option<NodeId> {
	match tree.visit_leaves_dfs(root, &mut |id| {
		if snap.rect(id).is_some() {
			ControlFlow::Continue(())
		} else {
			ControlFlow::Break(id)
		}
	}) {
		ControlFlow::Continue(()) => None,
		ControlFlow::Break(id) => Some(id),
	}
}

/// Scores `candidate` as a directional neighbor of `current`.
///
/// Eligibility and orthogonal measurements depend on direction:
///
/// - left/right use the horizontal half-plane and compare vertical overlap/alignment
/// - up/down use the vertical half-plane and compare horizontal overlap/alignment
///
/// Returned scores are ordered lexicographically by [`NavScore`].
#[must_use]
pub fn nav_score(current: Rect, candidate: Rect, dir: Direction, rank: usize) -> Option<NavScore> {
	let (eligible, primary_gap, axis, current_orth, candidate_orth) = match dir {
		Direction::Left => (
			candidate.right() <= current.left(),
			u32::try_from(current.left() - candidate.right()).ok()?,
			Axis::X,
			(current.top(), current.bottom()),
			(candidate.top(), candidate.bottom()),
		),
		Direction::Right => (
			candidate.left() >= current.right(),
			u32::try_from(candidate.left() - current.right()).ok()?,
			Axis::X,
			(current.top(), current.bottom()),
			(candidate.top(), candidate.bottom()),
		),
		Direction::Up => (
			candidate.bottom() <= current.top(),
			u32::try_from(current.top() - candidate.bottom()).ok()?,
			Axis::Y,
			(current.left(), current.right()),
			(candidate.left(), candidate.right()),
		),
		Direction::Down => (
			candidate.top() >= current.bottom(),
			u32::try_from(candidate.top() - current.bottom()).ok()?,
			Axis::Y,
			(current.left(), current.right()),
			(candidate.left(), candidate.right()),
		),
	};
	eligible.then_some(NavScore {
		primary_gap,
		orth_gap: orth_gap(current_orth.0, current_orth.1, candidate_orth.0, candidate_orth.1),
		orth_center_delta: current
			.center_twice_orth(axis)
			.abs_diff(candidate.center_twice_orth(axis)),
		tree_order_rank: rank,
	})
}
