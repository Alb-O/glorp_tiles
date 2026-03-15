//! Geometry-based directional navigation.
//!
//! Navigation operates on solved leaf rectangles rather than tree topology. Candidates must lie in
//! the directional half-plane, then are ranked lexicographically by gap and alignment metrics.
//! Depth-first leaf order provides the final stable tie-break source.

use {
	crate::{
		geom::{Direction, Rect, orth_gap},
		ids::NodeId,
		tree::Tree,
	},
	std::collections::HashMap,
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
/// `leaf_rects` is expected to contain solved rectangles for leaves of the same tree or snapshot.
/// If `current` has no rectangle, the tree is empty, or no candidate lies in the directional
/// half-plane, this returns `None`. Missing tree-order ranks fall back to `usize::MAX`.
#[must_use]
pub fn best_neighbor<T>(
	tree: &Tree<T>, leaf_rects: &HashMap<NodeId, Rect>, current: NodeId, dir: Direction,
) -> Option<NodeId> {
	let current_rect = leaf_rects.get(&current).copied()?;
	let order = tree.root_id().map(|root| tree.leaf_ids_dfs(root)).unwrap_or_default();
	let order_rank = order
		.into_iter()
		.enumerate()
		.map(|(idx, id)| (id, idx))
		.collect::<HashMap<_, _>>();

	leaf_rects
		.iter()
		.filter(|(id, _)| **id != current)
		.filter_map(|(id, rect)| {
			nav_score(current_rect, *rect, dir, *order_rank.get(id).unwrap_or(&usize::MAX)).map(|score| (*id, score))
		})
		.min_by_key(|(_, score)| *score)
		.map(|(id, _)| id)
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
	let (eligible, primary_gap, orth_gap_value, orth_center_delta) = match dir {
		Direction::Left => {
			let eligible = candidate.right() <= current.left();
			let primary_gap = u32::try_from(current.left() - candidate.right()).ok()?;
			let orth_gap_value = orth_gap(current.top(), current.bottom(), candidate.top(), candidate.bottom());
			let orth_center_delta = current
				.center_twice_orth(crate::geom::Axis::X)
				.abs_diff(candidate.center_twice_orth(crate::geom::Axis::X));
			(eligible, primary_gap, orth_gap_value, orth_center_delta)
		}
		Direction::Right => {
			let eligible = candidate.left() >= current.right();
			let primary_gap = u32::try_from(candidate.left() - current.right()).ok()?;
			let orth_gap_value = orth_gap(current.top(), current.bottom(), candidate.top(), candidate.bottom());
			let orth_center_delta = current
				.center_twice_orth(crate::geom::Axis::X)
				.abs_diff(candidate.center_twice_orth(crate::geom::Axis::X));
			(eligible, primary_gap, orth_gap_value, orth_center_delta)
		}
		Direction::Up => {
			let eligible = candidate.bottom() <= current.top();
			let primary_gap = u32::try_from(current.top() - candidate.bottom()).ok()?;
			let orth_gap_value = orth_gap(current.left(), current.right(), candidate.left(), candidate.right());
			let orth_center_delta = current
				.center_twice_orth(crate::geom::Axis::Y)
				.abs_diff(candidate.center_twice_orth(crate::geom::Axis::Y));
			(eligible, primary_gap, orth_gap_value, orth_center_delta)
		}
		Direction::Down => {
			let eligible = candidate.top() >= current.bottom();
			let primary_gap = u32::try_from(candidate.top() - current.bottom()).ok()?;
			let orth_gap_value = orth_gap(current.left(), current.right(), candidate.left(), candidate.right());
			let orth_center_delta = current
				.center_twice_orth(crate::geom::Axis::Y)
				.abs_diff(candidate.center_twice_orth(crate::geom::Axis::Y));
			(eligible, primary_gap, orth_gap_value, orth_center_delta)
		}
	};
	eligible.then_some(NavScore {
		primary_gap,
		orth_gap: orth_gap_value,
		orth_center_delta,
		tree_order_rank: rank,
	})
}
