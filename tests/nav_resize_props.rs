mod common;

use {
	common::{best_neighbor_oracle, eligible_splits_oracle, exercise_trace, leaf_ids, root_rect},
	glorp_tiles::{
		Axis, Direction, LeafMeta, NavError, NodeId, ResizeStrategy, Session, Slot, SolverPolicy, WeightPair,
		canonicalize_weights,
	},
	proptest::prelude::*,
};

fn zero_meta() -> LeafMeta {
	LeafMeta {
		limits: glorp_tiles::SizeLimits {
			min_w: 0,
			min_h: 0,
			max_w: None,
			max_h: None,
		},
		..LeafMeta::default()
	}
}

fn split_weights(session: &Session<u8>, split_id: NodeId) -> WeightPair {
	session.tree().split(split_id).expect("split should exist").weights()
}

fn left_edge(rect: glorp_tiles::Rect) -> i32 {
	rect.left()
}

fn edge_mismatch_session() -> (Session<u8>, NodeId, NodeId) {
	let mut session = Session::new();
	let focus = session.insert_root(1, zero_meta()).expect("insert root");
	let _ = session
		.split_focus(Axis::X, Slot::B, 2, zero_meta(), None)
		.expect("split root");
	session.set_selection(focus).expect("select focus leaf for inner split");
	let _ = session
		.wrap_selection(Axis::X, Slot::B, 3, zero_meta(), None)
		.expect("wrap focus in inner x split");
	let root = session.tree().root_id().expect("root should exist");
	(session, focus, root)
}

fn chain_session() -> (Session<u8>, NodeId, Vec<NodeId>) {
	let mut session = Session::new();
	let focus = session.insert_root(1, zero_meta()).expect("insert root");
	let _ = session
		.split_focus(Axis::X, Slot::A, 2, zero_meta(), Some(WeightPair { a: 0, b: 1 }))
		.expect("split root with zero-width left sibling");
	session
		.set_selection(focus)
		.expect("select focus leaf for nested split");
	let _ = session
		.wrap_selection(Axis::X, Slot::A, 3, zero_meta(), Some(WeightPair { a: 0, b: 1 }))
		.expect("wrap focus with inner zero-width sibling");
	let root = session.tree().root_id().expect("root should exist");
	let inner = session.tree().parent_of(focus).expect("inner split should exist");
	(session, focus, vec![inner, root])
}

fn distributed_allocations(eligible: &[common::RefEligibleSplit], request: u32, sign: i8) -> Vec<(NodeId, u32)> {
	let total_slack = eligible.iter().map(|entry| entry.slack(sign)).sum::<u32>();
	if total_slack == 0 {
		return Vec::new();
	}
	let request = request.min(total_slack);
	let mut assigned = 0_u32;
	let mut allocations = eligible
		.iter()
		.enumerate()
		.map(|(idx, entry)| {
			let slack = entry.slack(sign);
			let product = u128::from(request) * u128::from(slack);
			let base = u32::try_from(product / u128::from(total_slack)).expect("base share should fit u32");
			let remainder = u32::try_from(product % u128::from(total_slack)).expect("remainder should fit u32");
			assigned += base;
			(idx, slack, base.min(slack), remainder)
		})
		.collect::<Vec<_>>();
	let mut leftover = request - assigned;
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
		.filter_map(|(idx, _, delta, _)| (delta != 0).then_some((eligible[idx].split, delta)))
		.collect()
}

fn ancestor_chain_allocations(eligible: &[common::RefEligibleSplit], request: u32, sign: i8) -> Vec<(NodeId, u32)> {
	let mut remaining = request;
	let mut out = Vec::new();
	for entry in eligible {
		if remaining == 0 {
			break;
		}
		let delta = remaining.min(entry.slack(sign));
		if delta != 0 {
			out.push((entry.split, delta));
			remaining -= delta;
		}
	}
	out
}

proptest! {
	#[test]
	fn navigation_matches_oracle_on_random_sessions(
		bytes in prop::collection::vec(any::<u8>(), 1..48),
		w in 1_u32..12,
		h in 1_u32..12,
	) {
		let mut session = exercise_trace(&bytes);
		let root = root_rect(w, h);
		let snap = session.solve(root, &SolverPolicy::default()).expect("solve");

		for leaf in leaf_ids(&session) {
			for dir in [Direction::Left, Direction::Right, Direction::Up, Direction::Down] {
				session.set_selection(leaf).expect("select leaf");
				if let Some(expected) = best_neighbor_oracle(&session, &snap, leaf, dir) {
					session.focus_dir(dir, &snap).expect("focus move should succeed");
					prop_assert_eq!(session.focus(), Some(expected));
				} else {
					prop_assert_eq!(session.focus_dir(dir, &snap), Err(NavError::NoCandidate));
				}
			}
		}
	}
}

#[test]
fn navigation_tie_breaks_by_dfs_order() {
	let mut session = Session::new();
	let current = session
		.insert_root(3, LeafMeta::default())
		.expect("insert current leaf");
	let top_left = session
		.split_focus(Axis::X, Slot::A, 1, LeafMeta::default(), None)
		.expect("create left branch");
	session.set_selection(top_left).expect("select left branch");
	let bottom_left = session
		.wrap_selection(Axis::Y, Slot::B, 2, LeafMeta::default(), None)
		.expect("split left branch vertically");

	session.set_selection(current).expect("restore current focus");
	let snap = session.solve(root_rect(8, 8), &SolverPolicy::default()).expect("solve");

	assert_eq!(
		best_neighbor_oracle(&session, &snap, current, Direction::Left),
		Some(top_left)
	);
	session
		.focus_dir(Direction::Left, &snap)
		.expect("left navigation should succeed");
	assert_eq!(session.focus(), Some(top_left));
	assert_ne!(top_left, bottom_left);
}

#[test]
fn resize_excludes_same_axis_ancestor_when_edge_does_not_match() {
	let (mut session, focus, root_split) = edge_mismatch_session();
	let root = root_rect(12, 6);
	let snap = session.solve(root, &SolverPolicy::default()).expect("solve");
	let eligible = eligible_splits_oracle(&session, &snap, focus, Direction::Right);
	let root_weights_before = split_weights(&session, root_split);
	let before = snap.rect(focus).expect("focus rect");

	assert_eq!(eligible.len(), 1);
	session
		.grow_focus(Direction::Right, 99, ResizeStrategy::Local, &snap)
		.expect("local grow should succeed");
	let after = session.solve(root, &SolverPolicy::default()).expect("solve");
	let after_rect = after.rect(focus).expect("focus rect after resize");

	assert_eq!(split_weights(&session, root_split), root_weights_before);
	assert_eq!(after_rect.w - before.w, eligible[0].slack(1));
}

#[test]
fn local_resize_matches_nearest_eligible_slack() {
	let (mut session, focus, _) = edge_mismatch_session();
	let root = root_rect(12, 6);
	let snap = session.solve(root, &SolverPolicy::default()).expect("solve");
	let eligible = eligible_splits_oracle(&session, &snap, focus, Direction::Right);
	let before = snap.rect(focus).expect("focus rect");
	let expected = eligible[0].slack(1);

	session
		.grow_focus(Direction::Right, 99, ResizeStrategy::Local, &snap)
		.expect("local grow should succeed");
	let after = session.solve(root, &SolverPolicy::default()).expect("solve");

	assert_eq!(
		after.rect(focus).expect("focus rect after resize").right() - before.right(),
		i32::try_from(expected).expect("expected motion should fit i32")
	);
}

#[test]
fn ancestor_chain_rewrites_split_preferences_greedily_nearest_first() {
	let (mut session, focus, _) = chain_session();
	let root = root_rect(18, 6);
	let snap = session.solve(root, &SolverPolicy::default()).expect("solve");
	let eligible = eligible_splits_oracle(&session, &snap, focus, Direction::Left);
	let request = 24;
	let expected = ancestor_chain_allocations(&eligible, request, 1);
	let before = eligible
		.iter()
		.map(|entry| (entry.split, split_weights(&session, entry.split)))
		.collect::<Vec<_>>();

	session
		.shrink_focus(Direction::Left, request, ResizeStrategy::AncestorChain, &snap)
		.expect("ancestor-chain shrink should succeed");

	for (split, old_weight) in before {
		let entry = eligible
			.iter()
			.find(|entry| entry.split == split)
			.expect("eligible split should exist");
		let expected_delta = expected
			.iter()
			.find_map(|(expected_split, delta)| (*expected_split == split).then_some(*delta))
			.unwrap_or(0);
		let expected_a = entry.current_a + expected_delta;
		let expected_weight = canonicalize_weights(expected_a, entry.total - expected_a);
		let actual = split_weights(&session, split);
		if expected_delta == 0 {
			assert_eq!(actual, old_weight, "untouched split {split} should keep its weight");
		} else {
			assert_eq!(actual, expected_weight, "split {split} should use greedy chain weights");
		}
	}
}

#[test]
fn distributed_by_slack_uses_proportional_allocation_with_stable_remainder() {
	let (mut session, focus, splits) = chain_session();
	let root = root_rect(18, 6);
	let snap = session.solve(root, &SolverPolicy::default()).expect("solve");
	let eligible = eligible_splits_oracle(&session, &snap, focus, Direction::Left);
	let request = 5;
	let expected = distributed_allocations(&eligible, request, 1);
	let before = splits
		.iter()
		.copied()
		.map(|split| (split, split_weights(&session, split)))
		.collect::<Vec<_>>();

	session
		.shrink_focus(Direction::Left, request, ResizeStrategy::DistributedBySlack, &snap)
		.expect("distributed shrink should succeed");

	for (split, old_weight) in before {
		let entry = eligible
			.iter()
			.find(|entry| entry.split == split)
			.expect("eligible split should exist");
		let expected_delta = expected
			.iter()
			.find_map(|(expected_split, delta)| (*expected_split == split).then_some(*delta))
			.unwrap_or(0);
		let expected_a = entry.current_a + expected_delta;
		let expected_weight = canonicalize_weights(expected_a, entry.total - expected_a);
		let actual = split_weights(&session, split);
		if expected_delta == 0 {
			assert_eq!(actual, old_weight, "untouched split {split} should keep its weight");
		} else {
			assert_eq!(actual, expected_weight, "split {split} should use distributed weights");
		}
	}
}

#[test]
fn resize_oracle_and_session_agree_on_eligible_split_order() {
	let (mut session, focus, splits) = chain_session();
	let root = root_rect(18, 6);
	let snap = session.solve(root, &SolverPolicy::default()).expect("solve");
	let eligible = eligible_splits_oracle(&session, &snap, focus, Direction::Left);
	let before = splits
		.iter()
		.copied()
		.map(|split| (split, split_weights(&session, split)))
		.collect::<Vec<_>>();

	session
		.shrink_focus(Direction::Left, 1, ResizeStrategy::Local, &snap)
		.expect("local shrink should succeed");
	let moved = before
		.into_iter()
		.find_map(|(split, weight)| (split_weights(&session, split) != weight).then_some(split))
		.expect("one split should move");

	assert_eq!(moved, eligible[0].split);
}

#[test]
fn same_axis_chain_slack_budget_can_exceed_observed_leaf_motion() {
	let (mut session, focus, _) = chain_session();
	let root = root_rect(18, 6);
	let snap = session.solve(root, &SolverPolicy::default()).expect("solve");
	let eligible = eligible_splits_oracle(&session, &snap, focus, Direction::Left);
	let before = snap.rect(focus).expect("focus rect");
	let raw_sum = eligible.iter().map(|entry| entry.slack(1)).sum::<u32>();

	session
		.shrink_focus(Direction::Left, 24, ResizeStrategy::AncestorChain, &snap)
		.expect("ancestor-chain shrink should succeed");
	let after = session.solve(root, &SolverPolicy::default()).expect("solve");
	let motion = left_edge(after.rect(focus).expect("focus rect after resize")) - left_edge(before);

	assert!(raw_sum > before.w);
	assert!(motion <= i32::try_from(before.w).expect("width should fit i32"));
	assert_eq!(motion, i32::try_from(before.w).expect("width should fit i32"));
}
