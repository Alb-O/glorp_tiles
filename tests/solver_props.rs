mod common;

use {
	common::{assert_partition, choose_extent_oracle, exercise_trace, root_rect, solve_reference, stressed_policy},
	glorp_tiles::{
		Axis, LeafMeta, PairSpec, Session, Slot, SolveError, SolverPolicy, ValidationError, WeightPair,
		choose_extent_with_score, solve, summarize,
	},
	proptest::prelude::*,
	std::collections::HashMap,
};

fn two_leaf_tree(left: LeafMeta, right: LeafMeta) -> glorp_tiles::Tree<u8> {
	let mut session = Session::new();
	let _ = session.insert_root(1, left).expect("insert root");
	let _ = session
		.split_focus(Axis::X, Slot::B, 2, right, None)
		.expect("split root");
	session.tree().clone()
}

#[test]
fn tree_enumeration_is_sorted() {
	let mut session = Session::new();
	let _ = session.insert_root(1_u8, LeafMeta::default()).expect("insert root");
	let _ = session
		.split_focus(Axis::X, Slot::B, 2_u8, LeafMeta::default(), None)
		.expect("split root");
	let _ = session
		.split_focus(Axis::Y, Slot::B, 3_u8, LeafMeta::default(), None)
		.expect("split focus");
	let tree = session.tree();

	let ids = tree.node_ids();
	assert!(ids.windows(2).all(|w| w[0] < w[1]));

	let split_ids = tree.split_ids();
	assert!(split_ids.windows(2).all(|w| w[0] < w[1]));
}

proptest! {
	#[test]
	fn allocator_matches_oracle(
		total in 0_u32..16,
		min_a in 0_u32..8,
		min_b in 0_u32..8,
		wa in 0_u32..6,
		wb in 0_u32..6,
		sa in 1_u64..8,
		sb in 1_u64..8,
		max_a_extra in 0_u32..8,
		max_b_extra in 0_u32..8,
		use_max_a in any::<bool>(),
		use_max_b in any::<bool>(),
		seed in any::<u8>(),
	) {
		let wa = if wa == 0 && wb == 0 { 1 } else { wa };
		let max_a = use_max_a.then_some(min_a + max_a_extra);
		let max_b = use_max_b.then_some(min_b + max_b_extra);
		let spec = PairSpec {
			total,
			min_a,
			min_b,
			max_a,
			max_b,
			wa,
			wb,
			sa,
			sb,
		};
		let policy = stressed_policy(seed);
		let expected = choose_extent_oracle(spec, &policy);
		let actual = choose_extent_with_score(spec, &policy);
		prop_assert_eq!(actual, expected);
	}

	#[test]
	fn production_solver_matches_reference_and_partitions(bytes in prop::collection::vec(any::<u8>(), 1..48), w in 0_u32..9, h in 0_u32..9, seed in any::<u8>()) {
		let session = exercise_trace(&bytes);
		let root = root_rect(w, h);
		let policy = stressed_policy(seed);
		let production = session.solve(root, &policy);
		let reference = solve_reference(session.tree(), session.revision(), root, &policy);
		prop_assert_eq!(&production.node_rects, &reference.node_rects);
		prop_assert_eq!(&production.violations, &reference.violations);
		prop_assert_eq!(production.strict_feasible, reference.strict_feasible);
		assert_partition(&session, root, &production);

		let encoded = serde_json::to_string(&session).expect("session should serialize");
		let decoded: Session<u16> = serde_json::from_str(&encoded).expect("session should deserialize");
		let replay = decoded.solve(root, &policy);
		prop_assert_eq!(production, replay);
	}

	#[test]
	fn reference_solver_is_deterministic(bytes in prop::collection::vec(any::<u8>(), 1..48), w in 0_u32..9, h in 0_u32..9, seed in any::<u8>()) {
		let session = exercise_trace(&bytes);
		let root = root_rect(w, h);
		let policy = stressed_policy(seed);

		prop_assert_eq!(
			solve_reference(session.tree(), session.revision(), root, &policy),
			solve_reference(session.tree(), session.revision(), root, &policy),
		);
	}
}

#[test]
fn summary_matches_bruteforce_feasibility_envelope() {
	let session = exercise_trace(&[0, 1, 7, 0, 3, 6, 5, 2, 8, 11, 4, 10]);
	let root = session.tree().root_id().expect("non-empty session");
	let mut summaries = HashMap::new();
	let summary = summarize(session.tree(), root, &mut summaries).expect("summary should build");
	let policy = SolverPolicy::default();

	let strict_pairs = (0_u32..=20)
		.flat_map(|w| (0_u32..=20).map(move |h| (w, h)))
		.filter(|(w, h)| {
			solve_reference(session.tree(), session.revision(), root_rect(*w, *h), &policy).strict_feasible
		})
		.collect::<Vec<_>>();

	let min_w = strict_pairs
		.iter()
		.map(|(w, _)| *w)
		.min()
		.expect("some strict solution should exist");
	let min_h = strict_pairs
		.iter()
		.map(|(_, h)| *h)
		.min()
		.expect("some strict solution should exist");
	let max_w = strict_pairs
		.iter()
		.map(|(w, _)| *w)
		.max()
		.expect("strict widths should exist");
	let max_h = strict_pairs
		.iter()
		.map(|(_, h)| *h)
		.max()
		.expect("strict heights should exist");

	assert_eq!(summary.min_w, min_w);
	assert_eq!(summary.min_h, min_h);
	match summary.max_w {
		Some(value) => assert_eq!(value, max_w),
		None => assert_eq!(max_w, 20),
	}
	match summary.max_h {
		Some(value) => assert_eq!(value, max_h),
		None => assert_eq!(max_h, 20),
	}
}

#[test]
fn scale_symmetry_holds_with_exact_integer_arithmetic() {
	let mut session = Session::new();
	let _a = session.insert_root(1_u16, LeafMeta::default()).expect("insert root");
	let _b = session
		.split_focus(Axis::X, Slot::B, 2_u16, LeafMeta::default(), None)
		.expect("split x");
	let _c = session
		.split_focus(Axis::Y, Slot::B, 3_u16, LeafMeta::default(), None)
		.expect("split y");
	let scaled_meta = LeafMeta {
		limits: glorp_tiles::SizeLimits {
			min_w: 3,
			min_h: 3,
			..LeafMeta::default().limits
		},
		..LeafMeta::default()
	};
	let mut scaled = Session::new();
	let _a = scaled
		.insert_root(1_u16, scaled_meta.clone())
		.expect("insert scaled root");
	let _b = scaled
		.split_focus(Axis::X, Slot::B, 2_u16, scaled_meta.clone(), None)
		.expect("split scaled x");
	let _c = scaled
		.split_focus(Axis::Y, Slot::B, 3_u16, scaled_meta, None)
		.expect("split scaled y");
	let policy = SolverPolicy::default();
	let base = session.solve(root_rect(12, 12), &policy);
	let scaled_snap = scaled.solve(root_rect(36, 36), &policy);

	for (id, rect) in &base.node_rects {
		let scaled_rect = scaled_snap.rect(*id).expect("scaled rect missing");
		assert_eq!(scaled_rect.x, rect.x * 3);
		assert_eq!(scaled_rect.y, rect.y * 3);
		assert_eq!(scaled_rect.w, rect.w * 3);
		assert_eq!(scaled_rect.h, rect.h * 3);
	}
}

#[test]
fn solver_handles_maximal_weights_without_panicking() {
	let mut session = Session::new();
	let _ = session.insert_root(1_u8, LeafMeta::default()).expect("insert root");
	let _ = session
		.split_focus(
			Axis::X,
			Slot::B,
			2_u8,
			LeafMeta::default(),
			Some(WeightPair {
				a: u32::MAX,
				b: u32::MAX,
			}),
		)
		.expect("split root");

	let snapshot = solve(session.tree(), root_rect(12, 5), &SolverPolicy::default())
		.expect("maximal weights should remain solvable");

	assert!(snapshot.strict_feasible);
	assert_eq!(snapshot.rect(1).expect("left rect").w, 6);
	assert_eq!(snapshot.rect(2).expect("right rect").w, 6);
}

#[test]
fn summarize_rejects_min_width_sum_overflow() {
	let huge = LeafMeta {
		limits: glorp_tiles::SizeLimits {
			min_w: u32::MAX,
			..LeafMeta::default().limits
		},
		..LeafMeta::default()
	};
	let tree = two_leaf_tree(huge.clone(), huge);
	let root = tree.root_id().expect("split root");
	assert_eq!(
		summarize(&tree, root, &mut HashMap::new()),
		Err(ValidationError::ArithmeticOverflow {
			node: root,
			field: "min_w",
		})
	);
	assert!(matches!(
		solve(&tree, root_rect(8, 4), &SolverPolicy::default()),
		Err(SolveError::Validation(ValidationError::ArithmeticOverflow {
			node,
			field: "min_w",
		})) if node == root
	));
}

#[test]
fn summarize_rejects_finite_max_width_sum_overflow() {
	let bounded = LeafMeta {
		limits: glorp_tiles::SizeLimits {
			max_w: Some(u32::MAX),
			..LeafMeta::default().limits
		},
		..LeafMeta::default()
	};
	let tree = two_leaf_tree(bounded.clone(), bounded);
	let root = tree.root_id().expect("split root");

	assert_eq!(
		summarize(&tree, root, &mut HashMap::new()),
		Err(ValidationError::ArithmeticOverflow {
			node: root,
			field: "max_w",
		})
	);
}

#[test]
fn session_try_solve_reports_unrepresentable_summary_overflow() {
	let huge = LeafMeta {
		limits: glorp_tiles::SizeLimits {
			min_w: u32::MAX,
			..LeafMeta::default().limits
		},
		..LeafMeta::default()
	};
	let mut session = Session::new();
	let _ = session.insert_root(1_u8, huge.clone()).expect("insert root");
	let _ = session
		.split_focus(Axis::X, Slot::B, 2_u8, huge, None)
		.expect("split root");

	assert!(matches!(
		session.try_solve(root_rect(8, 4), &SolverPolicy::default()),
		Err(SolveError::Validation(ValidationError::ArithmeticOverflow {
			field: "min_w",
			..
		}))
	));
}
