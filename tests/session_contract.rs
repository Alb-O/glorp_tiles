mod common;

use {
	common::root_rect,
	glorp_tiles::{
		Axis, BalancedPreset, Direction, LeafMeta, NavError, NodeId, OpError, PresetKind, ResizeStrategy, Session,
		SizeLimits, Slot, SolverPolicy, TallPreset, ValidationError, WeightPair, solve_with_revision,
	},
};

fn two_leaf_session() -> Session<u8> {
	two_leaf_session_with_ids().0
}

fn two_leaf_session_with_ids() -> (Session<u8>, NodeId, NodeId) {
	let mut session = Session::new();
	let left = session.insert_root(1, LeafMeta::default()).expect("insert root");
	let right = session
		.split_focus(Axis::X, Slot::B, 2, LeafMeta::default(), None)
		.expect("split root");
	(session, left, right)
}

fn three_leaf_session() -> (Session<u8>, NodeId) {
	let (mut session, _, right) = two_leaf_session_with_ids();
	session.set_focus_leaf(right).expect("focus right leaf");
	let leaf = session
		.split_focus(Axis::Y, Slot::B, 3, LeafMeta::default(), None)
		.expect("split right leaf");
	(session, leaf)
}

fn invalid_meta() -> LeafMeta {
	LeafMeta {
		limits: SizeLimits {
			min_w: 4,
			min_h: 1,
			max_w: Some(2),
			max_h: None,
		},
		..LeafMeta::default()
	}
}

#[test]
fn empty_navigation_reports_empty_without_synthesized_validation() {
	let mut session = Session::<u8>::new();
	let snap = session
		.solve(root_rect(12, 8), &SolverPolicy::default())
		.expect("solve");

	assert_eq!(session.focus_dir(Direction::Right, &snap), Err(NavError::Empty));
}

#[test]
fn split_ids_postorder_returns_descendants_before_ancestors() {
	let mut session = Session::new();
	let a = session.insert_root(1_u8, LeafMeta::default()).expect("insert root");
	let b = session
		.split_focus(Axis::X, Slot::B, 2_u8, LeafMeta::default(), None)
		.expect("split root");

	session.set_selection(a).expect("select left leaf for nested split");
	let _ = session
		.wrap_selection(Axis::Y, Slot::B, 3_u8, LeafMeta::default(), None)
		.expect("wrap left leaf");
	session
		.set_selection(a)
		.expect("reselect original leaf for deeper split");
	let _ = session
		.wrap_selection(Axis::X, Slot::B, 4_u8, LeafMeta::default(), None)
		.expect("wrap left leaf again");

	let tree = session.tree();
	let deepest = tree.parent_of(a).expect("deepest split should exist");
	let middle = tree.parent_of(deepest).expect("middle split should exist");
	let root = tree.parent_of(middle).expect("root split should exist");
	let ids = tree.split_ids_postorder(root).expect("root should exist");

	assert_eq!(ids.len(), 3);
	assert_eq!(ids, vec![deepest, middle, root]);
	assert_eq!(ids.iter().copied().collect::<std::collections::BTreeSet<_>>().len(), 3);
	assert_eq!(tree.parent_of(b), Some(root));
}

#[test]
fn focus_and_selection_ops_do_not_stale_snapshots() {
	let mut session = two_leaf_session();
	let root = root_rect(12, 8);
	let snap = session.solve(root, &SolverPolicy::default()).expect("solve");

	session
		.focus_dir(Direction::Right, &snap)
		.expect("focus move should accept current snapshot");
	session.select_parent().expect("parent selection should work");
	session.select_focus();
	session
		.focus_dir(Direction::Left, &snap)
		.expect("selection-only ops should keep the original snapshot fresh");

	assert_eq!(session.revision(), snap.revision());
	assert_eq!(
		session
			.solve(root, &SolverPolicy::default())
			.expect("solve should remain fresh")
			.revision(),
		session.revision()
	);
}

#[test]
fn structural_mutation_stales_old_snapshot() {
	let mut session = two_leaf_session();
	let root = root_rect(12, 8);
	let snap = session.solve(root, &SolverPolicy::default()).expect("solve");

	let _ = session
		.split_focus(Axis::Y, Slot::B, 3, LeafMeta::default(), None)
		.expect("split focus");

	assert_eq!(session.focus_dir(Direction::Left, &snap), Err(NavError::StaleSnapshot));
	assert_eq!(
		session.grow_focus(Direction::Right, 1, ResizeStrategy::Local, &snap),
		Err(OpError::StaleSnapshot)
	);
}

#[test]
fn resize_mutation_stales_old_snapshot() {
	let mut session = two_leaf_session();
	let root = root_rect(12, 8);
	let snap = session.solve(root, &SolverPolicy::default()).expect("solve");

	session
		.grow_focus(Direction::Right, 1, ResizeStrategy::Local, &snap)
		.expect("resize should work with fresh snapshot");

	assert_eq!(session.focus_dir(Direction::Left, &snap), Err(NavError::StaleSnapshot));
	assert_eq!(
		session.shrink_focus(Direction::Right, 1, ResizeStrategy::Local, &snap),
		Err(OpError::StaleSnapshot)
	);
}

#[test]
fn foreign_and_ownerless_snapshots_are_rejected() {
	let root = root_rect(12, 8);
	let mut first = two_leaf_session();
	let mut second = two_leaf_session();
	let first_snap = first.solve(root, &SolverPolicy::default()).expect("solve");
	let ownerless = solve_with_revision(first.tree(), root, first.revision(), &SolverPolicy::default()).expect("solve");

	assert_eq!(
		second.focus_dir(Direction::Left, &first_snap),
		Err(NavError::ForeignSnapshot)
	);
	assert_eq!(
		second.grow_focus(Direction::Right, 1, ResizeStrategy::Local, &first_snap),
		Err(OpError::ForeignSnapshot)
	);
	assert_eq!(
		first.focus_dir(Direction::Left, &ownerless),
		Err(NavError::ForeignSnapshot)
	);
	assert_eq!(
		first.grow_focus(Direction::Right, 1, ResizeStrategy::Local, &ownerless),
		Err(OpError::ForeignSnapshot)
	);
}

#[test]
fn invalid_leaf_meta_insertions_are_atomic() {
	let bad = invalid_meta();

	let mut empty = Session::<u8>::new();
	assert_eq!(empty.insert_root(1, bad.clone()), Err(OpError::InvalidLeafMeta));
	assert_eq!(empty, Session::new());

	let mut split = two_leaf_session();
	let split_before = split.clone();
	assert_eq!(
		split.split_focus(Axis::Y, Slot::B, 3, bad.clone(), None),
		Err(OpError::InvalidLeafMeta)
	);
	assert_eq!(split, split_before);

	let mut wrap = two_leaf_session();
	let wrap_before = wrap.clone();
	assert_eq!(
		wrap.wrap_selection(Axis::Y, Slot::B, 3, bad, None),
		Err(OpError::InvalidLeafMeta)
	);
	assert_eq!(wrap, wrap_before);
}

#[test]
fn tree_helpers_return_none_for_missing_ids() {
	let session = two_leaf_session();
	let tree = session.tree();
	let missing = NodeId::from_raw(999);

	assert_eq!(tree.path_to_root(missing), None);
	assert_eq!(tree.ancestors_nearest_first(missing), None);
	assert_eq!(tree.leaf_ids_dfs(missing), None);
	assert_eq!(tree.split_ids_postorder(missing), None);
}

#[test]
fn targeting_helpers_preserve_invariants_without_bumping_revision() {
	let (mut session, left_leaf, right_leaf) = two_leaf_session_with_ids();
	session
		.set_selection(right_leaf)
		.expect("select right leaf for nested split");
	let lower_right = session
		.split_focus(Axis::Y, Slot::B, 3, LeafMeta::default(), None)
		.expect("split right leaf");
	let snap = session
		.solve(root_rect(12, 8), &SolverPolicy::default())
		.expect("solve");
	let root = session.tree().root_id().expect("root should exist");
	let right_split = session.tree().parent_of(right_leaf).expect("nested split should exist");
	let revision = session.revision();

	session.set_selection(root).expect("select root split");
	session.set_focus_leaf(lower_right).expect("set focus to nested leaf");
	assert_eq!(session.selection(), Some(root));
	assert_eq!(session.revision(), revision);
	session
		.focus_dir(Direction::Up, &snap)
		.expect("targeting helpers should not stale a fresh snapshot");
	session.validate().expect("state should remain valid");

	session.set_focus_leaf(left_leaf).expect("set focus to left leaf");
	assert_eq!(
		session.set_selection(right_split),
		Err(OpError::Validation(ValidationError::InvalidSelection(right_split,)))
	);
	assert_eq!(session.selection(), Some(root));
	assert_eq!(session.revision(), revision);

	session.set_focus_leaf(lower_right).expect("restore nested focus");
	session
		.set_selection(right_split)
		.expect("select split containing focus");
	assert_eq!(session.focus(), Some(lower_right));
	assert_eq!(session.selection(), Some(right_split));
	assert_eq!(session.revision(), revision);
	session.validate().expect("helpers should preserve invariants");
}

#[test]
fn apply_preset_invalid_weights_is_atomic() {
	let mut session = two_leaf_session();
	let root = session.tree().root_id().expect("root should exist");
	session.set_selection(root).expect("select root split");
	let before = session.clone();

	let result = session.apply_preset(PresetKind::Tall(TallPreset {
		master_slot: Slot::A,
		root_weights: WeightPair { a: 0, b: 0 },
	}));

	assert_eq!(result, Err(OpError::InvalidWeights));
	assert_eq!(session, before);
	session.validate().expect("state should remain valid");
}

#[test]
fn apply_preset_noops_do_not_bump_revision() {
	let mut leaf_selected = two_leaf_session();
	let leaf_selected_before = leaf_selected.clone();
	let leaf_revision = leaf_selected.revision();

	leaf_selected
		.apply_preset(PresetKind::Balanced(BalancedPreset {
			start_axis: Axis::X,
			alternate: false,
		}))
		.expect("leaf selection should no-op");

	assert_eq!(leaf_selected.revision(), leaf_revision);
	assert_eq!(leaf_selected, leaf_selected_before);
	leaf_selected.validate().expect("leaf no-op should stay valid");

	let mut matching = two_leaf_session();
	let root = matching.tree().root_id().expect("root should exist");
	matching.set_selection(root).expect("select root split");
	let matching_before = matching.clone();
	let matching_revision = matching.revision();

	matching
		.apply_preset(PresetKind::Balanced(BalancedPreset {
			start_axis: Axis::X,
			alternate: false,
		}))
		.expect("matching preset should no-op");

	assert_eq!(matching.revision(), matching_revision);
	assert_eq!(matching, matching_before);
	matching.validate().expect("matching no-op should stay valid");
}

#[test]
fn mirror_rebalance_and_zero_effect_resize_noops_do_not_bump_revision() {
	let (mut mirror, left_leaf, _) = two_leaf_session_with_ids();
	let mirror_revision = mirror.revision();
	mirror.set_selection(left_leaf).expect("select leaf");
	mirror.mirror_selection(Axis::X).expect("leaf mirror should no-op");
	assert_eq!(mirror.revision(), mirror_revision);

	let (mut rebalance, left_leaf, _) = two_leaf_session_with_ids();
	let leaf_revision = rebalance.revision();
	rebalance.set_selection(left_leaf).expect("select leaf");
	rebalance
		.rebalance_selection(glorp_tiles::RebalanceMode::LeafCount)
		.expect("leaf rebalance should no-op");
	assert_eq!(rebalance.revision(), leaf_revision);

	let root = rebalance.tree().root_id().expect("root should exist");
	rebalance.set_selection(root).expect("select root split");
	rebalance
		.rebalance_selection(glorp_tiles::RebalanceMode::LeafCount)
		.expect("first rebalance");
	let canonical_revision = rebalance.revision();
	rebalance
		.rebalance_selection(glorp_tiles::RebalanceMode::LeafCount)
		.expect("second rebalance should no-op");
	assert_eq!(rebalance.revision(), canonical_revision);

	let mut resize = Session::new();
	let _left = resize
		.insert_root(
			1_u8,
			LeafMeta {
				limits: SizeLimits {
					min_w: 4,
					min_h: 1,
					max_w: None,
					max_h: None,
				},
				..LeafMeta::default()
			},
		)
		.expect("insert root");
	let _right = resize
		.split_focus(
			Axis::X,
			Slot::B,
			2_u8,
			LeafMeta {
				limits: SizeLimits {
					min_w: 3,
					min_h: 1,
					max_w: None,
					max_h: None,
				},
				..LeafMeta::default()
			},
			None,
		)
		.expect("split root");
	let root_rect = root_rect(10, 4);
	let first_snap = resize.solve(root_rect, &SolverPolicy::default()).expect("solve");
	resize
		.grow_focus(Direction::Right, 10, ResizeStrategy::Local, &first_snap)
		.expect("first grow");
	let maxed_revision = resize.revision();
	let second_snap = resize.solve(root_rect, &SolverPolicy::default()).expect("solve");
	resize
		.grow_focus(Direction::Right, 1, ResizeStrategy::Local, &second_snap)
		.expect("second grow should clamp to no-op");
	assert_eq!(resize.revision(), maxed_revision);
}

#[test]
fn apply_preset_preserves_focus_and_retargets_selection_to_rebuilt_subtree() {
	let (mut session, preserved_leaf) = three_leaf_session();
	let old_root = session.tree().root_id().expect("root should exist");
	session.set_selection(old_root).expect("select root split");
	session.set_focus_leaf(preserved_leaf).expect("focus preserved leaf");
	let revision = session.revision();

	session
		.apply_preset(PresetKind::Balanced(BalancedPreset {
			start_axis: Axis::X,
			alternate: true,
		}))
		.expect("preset rewrite should succeed");

	let rebuilt = session.tree().root_id().expect("rebuilt root should exist");
	assert_eq!(session.focus(), Some(preserved_leaf));
	assert_eq!(session.selection(), Some(rebuilt));
	assert_ne!(session.selection(), session.focus());
	assert_eq!(session.revision(), revision + 1);
	session.validate().expect("rewrite should stay valid");
}

#[test]
fn apply_preset_nested_selection_retargets_rebuilt_subtree_and_root_id_stays() {
	let (mut session, preserved_leaf) = three_leaf_session();
	let root = session.tree().root_id().expect("root should exist");
	let nested = session
		.tree()
		.parent_of(preserved_leaf)
		.expect("nested split should exist");
	session.set_selection(nested).expect("select nested subtree for preset");
	session.set_focus_leaf(preserved_leaf).expect("focus preserved leaf");
	let revision = session.revision();

	session
		.apply_preset(PresetKind::Balanced(BalancedPreset {
			start_axis: Axis::X,
			alternate: false,
		}))
		.expect("nested preset rewrite should succeed");

	let rebuilt = session.selection().expect("selection should retarget to rebuilt");
	assert_eq!(session.focus(), Some(preserved_leaf));
	assert_eq!(session.selection(), Some(rebuilt));
	assert_eq!(
		session.tree().parent_of(rebuilt),
		Some(root),
		"parent_of(rebuilt) should stay attached to the outer split"
	);
	assert_eq!(session.tree().root_id(), Some(root), "root id stays the same");
	assert_eq!(session.revision(), revision + 1);
	session.validate().expect("nested rewrite should stay valid");
}

#[test]
fn move_selection_as_sibling_of_preserves_focus_and_selected_subtree() {
	let (mut session, preserved_leaf) = three_leaf_session();
	let root = session.tree().root_id().expect("root should exist");
	let selected = session
		.tree()
		.parent_of(preserved_leaf)
		.expect("nested split should exist");
	session.set_selection(selected).expect("select nested subtree to move");
	session.set_focus_leaf(preserved_leaf).expect("focus preserved leaf");
	let revision = session.revision();

	session
		.move_selection_as_sibling_of(root, Axis::X, Slot::B)
		.expect("move next to ancestor target");

	assert_eq!(session.focus(), Some(preserved_leaf));
	assert_eq!(session.selection(), Some(selected));
	assert_eq!(
		session.tree().parent_of(selected),
		session.tree().root_id(),
		"selected subtree should stay selected under the new wrapper root"
	);
	assert_eq!(session.revision(), revision + 1);
	session.validate().expect("move rewrite should stay valid");
}
