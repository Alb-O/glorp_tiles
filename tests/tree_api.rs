use glorp_tiles::{
	Axis, Direction, LeafMeta, NavError, NeighborError, NodeId, Rect, Session, SizeLimits, Slot, SolverPolicy, Tree,
	nav::best_neighbor, solve, solve_with_revision,
};

#[test]
fn direct_tree_edit_flow_matches_equivalent_session_flow() {
	let mut tree = Tree::new();
	let main = tree.insert_root("main", LeafMeta::default()).expect("insert root");
	let side = tree
		.split_leaf(main, Axis::X, Slot::B, "side", LeafMeta::default(), None)
		.expect("split leaf");
	let log = tree
		.wrap_node(main, Axis::Y, Slot::B, "log", LeafMeta::default(), None)
		.expect("wrap node");

	let mut session = Session::new();
	let main_session = session.insert_root("main", LeafMeta::default()).expect("insert root");
	let side_session = session
		.split_focus(Axis::X, Slot::B, "side", LeafMeta::default(), None)
		.expect("split focus");
	let log_session = session
		.wrap_selection(Axis::Y, Slot::B, "log", LeafMeta::default(), None)
		.expect("wrap selection");

	assert_eq!((main, side, log), (main_session, side_session, log_session));
	assert_eq!(tree, session.tree().clone());

	let root = Rect {
		x: 0,
		y: 0,
		w: 120,
		h: 40,
	};
	let tree_snap = solve(&tree, root, &SolverPolicy::default()).expect("tree solve");
	let session_snap =
		solve_with_revision(session.tree(), root, 0, &SolverPolicy::default()).expect("ownerless session solve");
	assert_eq!(tree_snap, session_snap);
}

#[test]
fn session_from_tree_defaults_to_first_leaf_and_roundtrips_back() {
	let mut tree = Tree::new();
	let main = tree.insert_root("main", LeafMeta::default()).expect("insert root");
	let _side = tree
		.split_leaf(main, Axis::X, Slot::B, "side", LeafMeta::default(), None)
		.expect("split leaf");
	let _log = tree
		.wrap_node(main, Axis::Y, Slot::B, "log", LeafMeta::default(), None)
		.expect("wrap node");
	let original = tree.clone();

	let session = Session::from_tree(tree).expect("session from tree");

	assert_eq!(session.focus(), Some(main));
	assert_eq!(session.selection(), Some(main));
	assert_eq!(session.into_tree(), original);
}

#[test]
fn session_from_tree_with_state_rejects_invalid_selection() {
	let mut tree = Tree::new();
	let left = tree.insert_root("left", LeafMeta::default()).expect("insert root");
	let right = tree
		.split_leaf(left, Axis::X, Slot::B, "right", LeafMeta::default(), None)
		.expect("split leaf");
	let root = tree.root_id().expect("root should exist");

	assert_eq!(
		Session::from_tree_with_state(tree, Some(right), Some(left)),
		Err(glorp_tiles::ValidationError::InvalidSelection(left))
	);
	assert_ne!(left, right);
	assert!(root != left && root != right);
}

#[test]
fn session_from_tree_with_state_handles_empty_and_rejects_partial_state() {
	let empty = Session::<&'static str>::from_tree_with_state(Tree::new(), None, None).expect("empty session");
	assert_eq!(empty.focus(), None);
	assert_eq!(empty.selection(), None);

	let mut tree = Tree::new();
	let leaf = tree.insert_root("main", LeafMeta::default()).expect("insert root");
	let partial_focus = Session::from_tree_with_state(tree.clone(), Some(leaf), None);
	let partial_selection = Session::from_tree_with_state(tree, None, Some(leaf));

	assert_eq!(partial_focus, Err(glorp_tiles::ValidationError::EmptyStateInconsistent));
	assert_eq!(
		partial_selection,
		Err(glorp_tiles::ValidationError::EmptyStateInconsistent)
	);
}

#[test]
fn best_neighbor_reports_checked_query_errors() {
	let mut tree = Tree::new();
	let left = tree.insert_root("left", LeafMeta::default()).expect("insert root");
	let right = tree
		.split_leaf(left, Axis::X, Slot::B, "right", LeafMeta::default(), None)
		.expect("split leaf");
	let root = tree.root_id().expect("root should exist");
	let snapshot = solve(
		&tree,
		Rect {
			x: 0,
			y: 0,
			w: 10,
			h: 4,
		},
		&SolverPolicy::default(),
	)
	.expect("solve");

	assert_eq!(
		best_neighbor(&Tree::<&'static str>::new(), &snapshot, left, Direction::Left),
		Ok(None)
	);
	assert_eq!(
		best_neighbor(&tree, &snapshot, NodeId::from_raw(999), Direction::Left),
		Err(NeighborError::MissingNode(NodeId::from_raw(999)))
	);
	assert_eq!(
		best_neighbor(&tree, &snapshot, root, Direction::Left),
		Err(NeighborError::NotLeaf(root))
	);
	assert_eq!(best_neighbor(&tree, &snapshot, left, Direction::Left), Ok(None));
	assert_eq!(best_neighbor(&tree, &snapshot, left, Direction::Right), Ok(Some(right)));

	let mut partial = serde_json::to_value(&snapshot).expect("snapshot should serialize");
	let node_rects = partial["node_rects"]
		.as_object_mut()
		.expect("snapshot node_rects should serialize as an object");
	node_rects.remove(&right.to_string());
	let partial_snapshot = serde_json::from_value(partial).expect("snapshot should deserialize");
	assert_eq!(
		best_neighbor(&tree, &partial_snapshot, left, Direction::Right),
		Err(NeighborError::MissingSnapshotRect(right))
	);

	let mut invalid = serde_json::to_value(&tree).expect("tree should serialize");
	invalid["root"] = serde_json::json!(999);
	let invalid_tree: Tree<String> = serde_json::from_value(invalid).expect("tree should deserialize");
	assert_eq!(
		best_neighbor(&invalid_tree, &snapshot, left, Direction::Right),
		Err(NeighborError::Validation(glorp_tiles::ValidationError::MissingRoot(
			NodeId::from_raw(999)
		)))
	);
}

#[test]
fn session_leaf_setters_preserve_identity_and_revision_contract() {
	let mut session = Session::new();
	let left = session.insert_root(1_u8, LeafMeta::default()).expect("insert root");
	let _right = session
		.split_focus(Axis::X, Slot::B, 2_u8, LeafMeta::default(), None)
		.expect("split focus");
	let root = Rect {
		x: 0,
		y: 0,
		w: 12,
		h: 8,
	};
	let snap = session.solve(root, &SolverPolicy::default()).expect("solve");
	let revision = session.revision();

	session
		.set_leaf_payload(left, 9_u8)
		.expect("payload update should succeed");
	assert_eq!(session.revision(), revision);
	assert_eq!(*session.tree().leaf(left).expect("leaf should exist").payload(), 9);
	session
		.focus_dir(Direction::Right, &snap)
		.expect("payload updates should not stale geometry snapshots");

	let changed = session
		.set_leaf_meta(
			left,
			LeafMeta {
				limits: SizeLimits {
					min_w: 4,
					..SizeLimits::default()
				},
				..LeafMeta::default()
			},
		)
		.expect("meta update should succeed");
	assert!(changed);
	assert_eq!(session.revision(), revision + 1);
	assert_eq!(session.focus_dir(Direction::Left, &snap), Err(NavError::StaleSnapshot));
	assert_eq!(
		session
			.tree()
			.leaf(left)
			.expect("leaf should exist")
			.meta()
			.limits
			.min_w,
		4
	);
}

#[test]
fn node_id_display_and_serde_roundtrip_raw_value() {
	let id = NodeId::from_raw(42);

	assert_eq!(id.to_string(), "42");
	assert_eq!(serde_json::to_string(&id).expect("serialize"), "42");
	assert_eq!(serde_json::from_str::<NodeId>("42").expect("deserialize"), id);
	assert_eq!(id.into_raw(), 42);
}
