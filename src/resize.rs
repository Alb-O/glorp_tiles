//! Geometry-driven resize helpers.
//!
//! Resize eligibility is based on the exact solved edge touched by the focused leaf, not merely on
//! ancestor axis. Resizing rewrites split weights rather than storing absolute pixel or cell
//! extents, and strategies differ only in how they distribute a requested movement across eligible
//! ancestor splits.

use {
	crate::{
		error::OpError,
		geom::{Axis, Direction},
		ids::NodeId,
		limits::Summary,
		snapshot::Snapshot,
		tree::Tree,
	},
	serde::{Deserialize, Serialize},
	std::collections::HashMap,
};

/// Strategy for distributing a resize request across eligible ancestor splits.
///
/// For example, a rightward grow request can either consume only the nearest matching divider
/// ([`Self::Local`]), walk outward through all matching ancestors ([`Self::AncestorChain`]), or
/// spread the movement across all matching ancestors by available slack
/// ([`Self::DistributedBySlack`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResizeStrategy {
	/// Apply the entire request to the nearest eligible split only.
	Local,
	/// Consume slack greedily from nearest eligible split to farthest.
	AncestorChain,
	/// Distribute by strict slack proportion, with deterministic remainder assignment.
	DistributedBySlack,
}

/// Ancestor split that can participate in a geometry-driven resize.
#[derive(Debug, Clone, Copy)]
pub(crate) struct EligibleSplit {
	/// Split node id.
	pub split: NodeId,
	/// Current total extent of the split on its axis.
	pub total: u32,
	/// Current solved extent of child `A`.
	pub current_a: u32,
	/// Lowest strictly feasible extent for child `A` under the current cross-size.
	pub lo: u32,
	/// Highest strictly feasible extent for child `A` under the current cross-size.
	pub hi: u32,
}

impl EligibleSplit {
	/// Returns remaining strict slack in the direction indicated by `sign`.
	pub(crate) fn slack(self, sign: i8) -> u32 {
		if self.lo > self.hi {
			return 0;
		}
		if sign > 0 {
			self.hi.saturating_sub(self.current_a)
		} else {
			self.current_a.saturating_sub(self.lo)
		}
	}
}

/// Returns the sign to apply to child `A` extents for a resize request.
///
/// Positive means "increase child `A`"; negative means "decrease child `A`".
pub(crate) fn resize_sign(dir: Direction, outward: bool) -> i8 {
	match (dir, outward) {
		(Direction::Right | Direction::Down, true) | (Direction::Left | Direction::Up, false) => 1,
		(Direction::Left | Direction::Up, true) | (Direction::Right | Direction::Down, false) => -1,
	}
}

/// Distributes a requested resize amount across `eligible` splits.
///
/// The returned vector is in nearest-first eligible order.
///
/// - [`ResizeStrategy::Local`] may return a zero-delta entry if the nearest split has zero slack.
/// - [`ResizeStrategy::AncestorChain`] consumes slack greedily from nearest to farthest.
/// - [`ResizeStrategy::DistributedBySlack`] floors proportional shares, assigns leftover units by
///   largest remainder with earlier eligible splits winning ties, then restores original eligible
///   order in the output.
pub(crate) fn distribute_resize(
	amount: u32, strategy: ResizeStrategy, sign: i8, eligible: &[EligibleSplit],
) -> Vec<(usize, u32)> {
	match strategy {
		ResizeStrategy::Local => eligible
			.first()
			.map(|entry| vec![(0, amount.min(entry.slack(sign)))])
			.unwrap_or_default(),
		ResizeStrategy::AncestorChain => {
			let mut remaining = amount;
			let mut out = Vec::with_capacity(eligible.len());
			for (idx, entry) in eligible.iter().enumerate() {
				if remaining == 0 {
					break;
				}
				let delta = remaining.min(entry.slack(sign));
				if delta != 0 {
					out.push((idx, delta));
					remaining -= delta;
				}
			}
			out
		}
		ResizeStrategy::DistributedBySlack => {
			let total_slack = eligible.iter().map(|entry| entry.slack(sign)).sum::<u32>();
			if total_slack == 0 {
				return Vec::new();
			}
			let request = amount.min(total_slack);
			let mut assigned = 0_u32;
			let mut allocations = eligible
				.iter()
				.enumerate()
				.map(|(idx, entry)| {
					let slack = entry.slack(sign);
					let product = u128::from(request) * u128::from(slack);
					let base = u32::try_from(product / u128::from(total_slack)).expect("base resize share exceeds u32");
					let remainder = u32::try_from(product % u128::from(total_slack)).expect("remainder exceeds u32");
					assigned += base;
					(idx, slack, base.min(slack), remainder)
				})
				.collect::<Vec<_>>();
			let mut leftover = request - assigned;
			// Use largest-remainder apportionment after flooring proportional shares; tie-break by
			// original eligible order to keep the result deterministic.
			allocations.sort_by_key(|(idx, _, _, remainder)| (std::cmp::Reverse(*remainder), *idx));
			for (_, slack, base, _) in &mut allocations {
				if leftover == 0 {
					break;
				}
				if *base < *slack {
					*base += 1;
					leftover -= 1;
				}
			}
			allocations.sort_by_key(|(idx, ..)| *idx);
			allocations
				.into_iter()
				.filter_map(|(idx, _, base, _)| (base != 0).then_some((idx, base)))
				.collect()
		}
	}
}

/// Returns resize-eligible ancestor splits for `focus`, nearest first.
///
/// Eligibility requires exact solved divider alignment on the requested edge of the focused leaf.
/// The returned `lo` and `hi` bounds are the strictly feasible bounds for child `A` under the
/// current cross-size at that split.
pub(crate) fn eligible_splits<T>(
	tree: &Tree<T>, focus: NodeId, dir: Direction, snap: &Snapshot, summaries: &HashMap<NodeId, Summary>,
) -> Result<Vec<EligibleSplit>, OpError> {
	let focus_rect = snap.rect(focus).ok_or(OpError::MissingNode(focus))?;
	let mut out = Vec::new();
	let mut child_on_path = focus;
	let mut cursor = tree.parent_of(focus);
	while let Some(split_id) = cursor {
		let split = tree.split(split_id).ok_or(OpError::NotSplit(split_id))?;
		let a_rect = snap.rect(split.a()).ok_or(OpError::MissingNode(split.a()))?;
		let b_rect = snap.rect(split.b()).ok_or(OpError::MissingNode(split.b()))?;
		let focus_in_a = split.a() == child_on_path;
		let eligible = match dir {
			Direction::Right => split.axis() == Axis::X && focus_in_a && focus_rect.right() == a_rect.right(),
			Direction::Left => split.axis() == Axis::X && !focus_in_a && focus_rect.left() == b_rect.left(),
			Direction::Down => split.axis() == Axis::Y && focus_in_a && focus_rect.bottom() == a_rect.bottom(),
			Direction::Up => split.axis() == Axis::Y && !focus_in_a && focus_rect.top() == b_rect.top(),
		};
		if eligible {
			let total = a_rect.extent(split.axis()) + b_rect.extent(split.axis());
			let sum_a = summaries
				.get(&split.a())
				.copied()
				.ok_or(OpError::MissingNode(split.a()))?;
			let sum_b = summaries
				.get(&split.b())
				.copied()
				.ok_or(OpError::MissingNode(split.b()))?;
			let (min_a, max_a) = sum_a.axis_limits(split.axis());
			let (min_b, max_b) = sum_b.axis_limits(split.axis());
			out.push(EligibleSplit {
				split: split_id,
				total,
				current_a: a_rect.extent(split.axis()),
				lo: min_a.max(max_b.map_or(0, |max_b| total.saturating_sub(max_b))),
				hi: total.saturating_sub(min_b).min(max_a.unwrap_or(total)),
			});
		}
		child_on_path = split_id;
		cursor = split.parent();
	}
	Ok(out)
}
