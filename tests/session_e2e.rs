mod common;

use {
	common::{assert_partition, exercise_trace, leaf_ids, meta, root_rect, split_ids},
	glorp_tiles::{
		Axis, BalancedPreset, Direction, LeafMeta, NodeId, PresetKind, RebalanceMode, ResizeStrategy, Session, Slot,
		SolverPolicy, TallPreset, WeightPair, WidePreset,
	},
	std::collections::BTreeSet,
};

#[derive(Debug, Clone, PartialEq, Eq)]
enum NodeSnapshot {
	Leaf {
		parent: Option<NodeId>,
		payload: u16,
		meta: LeafMeta,
	},
	Split {
		parent: Option<NodeId>,
		axis: Axis,
		a: NodeId,
		b: NodeId,
		weights: WeightPair,
	},
}

fn snapshot_node(tree: &glorp_tiles::Tree<u16>, id: NodeId) -> NodeSnapshot {
	if let Some(leaf) = tree.leaf(id) {
		NodeSnapshot::Leaf {
			parent: leaf.parent(),
			payload: *leaf.payload(),
			meta: leaf.meta().clone(),
		}
	} else {
		let split = tree.split(id).expect("node should exist");
		NodeSnapshot::Split {
			parent: split.parent(),
			axis: split.axis(),
			a: split.a(),
			b: split.b(),
			weights: split.weights(),
		}
	}
}

#[test]
fn end_to_end_structural_edit_navigation_resize_flow() {
	let mut session = Session::new();
	let a = session
		.insert_root(1_u32, meta(2, 2, None, None, 4))
		.expect("insert root");
	let b = session
		.split_focus(Axis::X, Slot::B, 2_u32, meta(1, 1, Some(8), None, 1), None)
		.expect("split root");
	let c = session
		.wrap_selection(
			Axis::Y,
			Slot::B,
			3_u32,
			meta(1, 1, None, Some(6), 2),
			Some(WeightPair { a: 2, b: 1 }),
		)
		.expect("wrap selection");
	session.select_focus();
	let d = session
		.split_focus(Axis::Y, Slot::B, 4_u32, LeafMeta::default(), None)
		.expect("split focus vertically");

	let root = root_rect(18, 12);
	let snap = session.solve(root, &SolverPolicy::default());
	assert_partition(&session, root, &snap);

	session.focus_dir(Direction::Right, &snap).expect("move focus right");
	let before = session.solve(root, &SolverPolicy::default());
	let focused = session.focus().expect("focus should exist");
	let before_rect = before.rect(focused).expect("focus rect should exist");
	session
		.grow_focus(Direction::Down, 3, ResizeStrategy::Local, &before)
		.expect("grow focus downward");
	let after = session.solve(root, &SolverPolicy::default());
	let after_rect = after.rect(focused).expect("focus rect should exist after resize");

	assert!(after_rect.h >= before_rect.h);
	assert!(after.strict_feasible);
	assert_eq!(leaf_ids(&session).len(), 4);
	assert!(split_ids(&session).len() >= 3);
	assert!(a != b && b != c && c != d);
}

#[test]
fn mirror_swap_and_move_round_trip_behaviors_hold() {
	let mut session = exercise_trace(&[0, 1, 0, 6, 11, 5, 7, 8]);
	let original = session.clone();
	let root_id = session.tree().root_id().expect("non-empty tree");
	session.set_selection(root_id).expect("select root");
	session.mirror_selection(Axis::X).expect("mirror x");
	session.mirror_selection(Axis::X).expect("mirror x again");
	assert_eq!(session.tree(), original.tree());

	let nodes = session.tree().node_ids();
	let (a, b) = nodes
		.iter()
		.enumerate()
		.find_map(|(idx, a)| {
			nodes.iter().skip(idx + 1).find_map(|b| {
				(!session.tree().contains_in_subtree(*a, *b) && !session.tree().contains_in_subtree(*b, *a))
					.then_some((*a, *b))
			})
		})
		.expect("need a disjoint swap pair");
	session.swap_nodes(a, b).expect("first swap");
	session.swap_nodes(a, b).expect("second swap");
	assert_eq!(session.tree(), original.tree());
}

#[test]
fn split_remove_roundtrip_returns_original_layout() {
	let mut session = exercise_trace(&[0, 1, 0, 3]);
	let original_root = session.tree().root_id();
	let mut original_nodes = session
		.tree()
		.node_ids()
		.into_iter()
		.map(|id| (id, snapshot_node(session.tree(), id)))
		.collect::<Vec<_>>();
	original_nodes.sort_by_key(|(id, _)| *id);
	let original_snap = session.solve(root_rect(20, 10), &SolverPolicy::default());
	let new_leaf = session
		.split_focus(Axis::X, Slot::B, 99_u16, LeafMeta::default(), None)
		.expect("split focus");
	session.set_selection(new_leaf).expect("select inserted leaf");
	session.remove_focus().expect("remove inserted leaf");
	let roundtrip = session.solve(root_rect(20, 10), &SolverPolicy::default());

	let mut roundtrip_nodes = session
		.tree()
		.node_ids()
		.into_iter()
		.map(|id| (id, snapshot_node(session.tree(), id)))
		.collect::<Vec<_>>();
	roundtrip_nodes.sort_by_key(|(id, _)| *id);
	assert_eq!(session.tree().root_id(), original_root);
	assert_eq!(roundtrip_nodes, original_nodes);
	assert_eq!(roundtrip.node_rects, original_snap.node_rects);
}

#[test]
fn rebalance_and_preset_are_idempotent_after_first_application() {
	let mut session = exercise_trace(&[0, 1, 0, 1, 6, 6]);
	let root = session.tree().root_id().expect("non-empty tree");
	session.set_selection(root).expect("select root");
	session
		.rebalance_selection(RebalanceMode::LeafCount)
		.expect("first rebalance");
	let rebalanced = session.tree().clone();
	session
		.rebalance_selection(RebalanceMode::LeafCount)
		.expect("second rebalance");
	assert_eq!(session.tree(), &rebalanced);

	session.set_selection(root).expect("reselect root");
	let preset = PresetKind::Balanced(BalancedPreset {
		start_axis: Axis::X,
		alternate: true,
	});
	session.apply_preset(preset).expect("first preset");
	let preset_tree = session.tree().clone();
	session
		.set_selection(session.tree().root_id().expect("root should exist"))
		.expect("select rebuilt root");
	session.apply_preset(preset).expect("second preset");
	assert_eq!(session.tree(), &preset_tree);
}

#[test]
fn tall_and_wide_presets_keep_leaf_identity_and_validate() {
	let mut session = exercise_trace(&[0, 1, 0, 1, 0]);
	let before_leaves = leaf_ids(&session);
	let root = session.tree().root_id().expect("non-empty tree");
	session.set_selection(root).expect("select root");
	session
		.apply_preset(PresetKind::Tall(TallPreset {
			master_slot: Slot::A,
			root_weights: WeightPair { a: 3, b: 2 },
		}))
		.expect("apply tall");
	let after_tall = leaf_ids(&session);
	assert_eq!(before_leaves, after_tall);

	session
		.set_selection(session.tree().root_id().expect("root should exist"))
		.expect("select rebuilt root");
	session
		.apply_preset(PresetKind::Wide(WidePreset {
			master_slot: Slot::B,
			root_weights: WeightPair { a: 1, b: 1 },
		}))
		.expect("apply wide");
	assert_eq!(
		leaf_ids(&session).into_iter().collect::<BTreeSet<_>>(),
		before_leaves.into_iter().collect::<BTreeSet<_>>()
	);
	session.validate().expect("preset rebuild should preserve invariants");
}

#[test]
fn mirror_x_matches_geometric_reflection() {
	let mut session = Session::new();
	let _a = session.insert_root(1_u16, LeafMeta::default()).expect("insert root");
	let _b = session
		.split_focus(
			Axis::X,
			Slot::B,
			2_u16,
			LeafMeta::default(),
			Some(WeightPair { a: 2, b: 1 }),
		)
		.expect("split x");
	let _c = session
		.wrap_selection(Axis::Y, Slot::B, 3_u16, LeafMeta::default(), None)
		.expect("wrap y");
	let root = root_rect(18, 12);
	let original = session.solve(root, &SolverPolicy::default());

	let root_id = session.tree().root_id().expect("non-empty tree");
	session.set_selection(root_id).expect("select root");
	session.mirror_selection(Axis::X).expect("mirror selection");
	let mirrored = session.solve(root, &SolverPolicy::default());

	for leaf in leaf_ids(&session) {
		let original_rect = original.rect(leaf).expect("original rect missing");
		let mirrored_rect = mirrored.rect(leaf).expect("mirrored rect missing");
		assert_eq!(mirrored_rect, original_rect.mirrored(Axis::X, root));
	}
}

#[test]
fn resize_clamps_to_strict_slack() {
	let mut session = Session::new();
	let left = session
		.insert_root(1_u16, meta(4, 1, None, None, 3))
		.expect("insert root");
	let _right = session
		.split_focus(Axis::X, Slot::B, 2_u16, meta(3, 1, None, None, 1), None)
		.expect("split x");
	let root = root_rect(10, 4);
	let snap = session.solve(root, &SolverPolicy::default());
	let before = snap.rect(left).expect("left rect missing");
	session
		.grow_focus(Direction::Right, 10, ResizeStrategy::Local, &snap)
		.expect("grow right");
	let after = session.solve(root, &SolverPolicy::default());
	let after_rect = after.rect(left).expect("left rect missing after resize");

	assert_eq!(before.w, 5);
	assert_eq!(after_rect.w, 7);
	assert_eq!(after_rect.w - before.w, 2);
}

#[test]
fn move_selection_supports_ancestor_targets() {
	let mut session = Session::new();
	let _a = session.insert_root(1_u16, LeafMeta::default()).expect("insert root");
	let b = session
		.split_focus(Axis::X, Slot::B, 2_u16, LeafMeta::default(), None)
		.expect("split x");
	session.set_selection(b).expect("select leaf b");
	let _c = session
		.split_focus(Axis::Y, Slot::B, 3_u16, LeafMeta::default(), None)
		.expect("split y");

	let ancestor_target = session.tree().root_id().expect("root should exist");
	let selected = session.selection().expect("selection should exist");
	session
		.move_selection_as_sibling_of(ancestor_target, Axis::X, Slot::B)
		.expect("move next to ancestor target");

	assert!(session.tree().contains(selected));
	assert!(session.validate().is_ok());
}
