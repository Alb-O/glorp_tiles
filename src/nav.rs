//! Geometry-based directional navigation.
//!
//! Navigation operates on solved leaf rectangles rather than tree topology. Candidates must lie in
//! the directional half-plane, then are ranked lexicographically by gap and alignment metrics.
//! Depth-first leaf order provides the final stable tie-break source.

use {
	crate::{
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
/// `snap` is expected to contain solved rectangles for leaves of the same tree. If `current` has
/// no rectangle, the tree is empty, or no candidate lies in the directional half-plane, this
/// returns `None`.
#[must_use]
pub fn best_neighbor<T>(tree: &Tree<T>, snap: &Snapshot, current: NodeId, dir: Direction) -> Option<NodeId> {
	let current_rect = snap.rect(current)?;
	let root = tree.root_id()?;
	let mut best = None;
	let mut rank = 0;
	let _ = tree.visit_leaves_dfs(root, &mut |id| {
		if id != current
			&& let Some((candidate, score)) = snap
				.rect(id)
				.and_then(|rect| nav_score(current_rect, rect, dir, rank).map(|score| (id, score)))
			&& best.is_none_or(|(_, best_score)| score < best_score)
		{
			best = Some((candidate, score));
		}
		rank += 1;
		ControlFlow::<()>::Continue(())
	});
	best.map(|(id, _)| id)
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
