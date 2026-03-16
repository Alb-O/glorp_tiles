#![allow(dead_code)]

use {
	glorp_tiles::{
		Axis, BalancedPreset, Direction, LeafMeta, NodeId, PairSpec, PresetKind, RebalanceMode, Rect, ResizeStrategy,
		ScoreTuple, Session, ShortageMode, SizeLimits, Slot, SolverPolicy, TallPreset, TieBreakMode, Tree, WidePreset,
		geom::orth_gap,
	},
	std::collections::{BTreeMap, HashMap},
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefSnapshot {
	pub revision: u64,
	pub root: Rect,
	pub node_rects: BTreeMap<NodeId, Rect>,
	pub split_traces: Vec<glorp_tiles::SplitTrace>,
	pub violations: Vec<glorp_tiles::Violation>,
	pub strict_feasible: bool,
}

impl RefSnapshot {
	#[must_use]
	pub fn rect(&self, node: NodeId) -> Option<Rect> {
		self.node_rects.get(&node).copied()
	}
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RefSummary {
	min_w: u64,
	min_h: u64,
	max_w: Option<u64>,
	max_h: Option<u64>,
	leaf_count: u64,
	shrink_cost: u64,
	grow_cost: u64,
}

impl RefSummary {
	fn axis_limits(self, axis: Axis) -> (u64, Option<u64>) {
		match axis {
			Axis::X => (self.min_w, self.max_w),
			Axis::Y => (self.min_h, self.max_h),
		}
	}
}

pub fn root_rect(w: u32, h: u32) -> Rect {
	Rect { x: 0, y: 0, w, h }
}

pub fn meta(min_w: u32, min_h: u32, max_w: Option<u32>, max_h: Option<u32>, shrink: u16) -> LeafMeta {
	LeafMeta {
		limits: SizeLimits {
			min_w,
			min_h,
			max_w,
			max_h,
		},
		priority: glorp_tiles::Priority { shrink, grow: 1 },
	}
}

pub fn choose_extent_oracle(spec: PairSpec, policy: SolverPolicy) -> (u32, ScoreTuple) {
	let mut best_a = 0;
	let mut best_score = oracle_score(spec, policy, best_a);

	for a in 1..=spec.total {
		let candidate = oracle_score(spec, policy, a);
		if candidate < best_score {
			best_a = a;
			best_score = candidate;
		}
	}

	(best_a, best_score)
}

fn oracle_score(spec: PairSpec, policy: SolverPolicy, a: u32) -> ScoreTuple {
	let size_b = spec.total - a;
	let short_a = u128::from(spec.min_a.saturating_sub(a));
	let short_b = u128::from(spec.min_b.saturating_sub(size_b));
	let over_a = u128::from(spec.max_a.map_or(0, |max| a.saturating_sub(max)));
	let over_b = u128::from(spec.max_b.map_or(0, |max| size_b.saturating_sub(max)));
	let total_weight = u128::from(spec.wa) + u128::from(spec.wb);

	ScoreTuple {
		shortage_penalty: match policy.shortage_mode {
			ShortageMode::Equal => short_a + short_b,
			ShortageMode::ByShrinkPriority => short_a * u128::from(spec.sa) + short_b * u128::from(spec.sb),
		},
		overflow_penalty: over_a + over_b,
		preference_penalty: (u128::from(a) * total_weight).abs_diff(u128::from(spec.total) * u128::from(spec.wa)),
		tie_break: match policy.tie_break {
			TieBreakMode::PreferA => u128::from(spec.total - a),
			TieBreakMode::PreferB => u128::from(a),
		},
	}
}

pub fn leaf_ids<T>(session: &Session<T>) -> Vec<NodeId> {
	session
		.tree()
		.root_id()
		.map(|root| {
			session
				.tree()
				.leaf_ids_dfs(root)
				.expect("root_id should point at a valid node")
		})
		.unwrap_or_default()
}

pub fn split_ids<T>(session: &Session<T>) -> Vec<NodeId> {
	session.tree().split_ids()
}

pub fn assert_partition<T>(session: &Session<T>, root: Rect, snap: &glorp_tiles::Snapshot) {
	let Some(tree_root) = session.tree().root_id() else {
		assert!(snap.node_rects().is_empty());
		return;
	};
	let mut cells = vec![None; usize::try_from(root.w.saturating_mul(root.h)).expect("root too large")];
	for leaf in session
		.tree()
		.leaf_ids_dfs(tree_root)
		.expect("root_id should point at a valid node")
	{
		let rect = snap.rect(leaf).expect("missing leaf rect");
		for y in rect.top()..rect.bottom() {
			for x in rect.left()..rect.right() {
				let x_off = u32::try_from(x - root.left()).expect("x below root");
				let y_off = u32::try_from(y - root.top()).expect("y below root");
				let idx = usize::try_from(y_off * root.w + x_off).expect("cell index overflow");
				assert!(cells[idx].is_none(), "cell overlapped at ({x}, {y})");
				cells[idx] = Some(leaf);
			}
		}
	}
	assert!(cells.iter().all(Option::is_some), "partition left holes inside root");
}

pub fn best_neighbor_oracle<T>(
	session: &Session<T>, snap: &glorp_tiles::Snapshot, current: NodeId, dir: Direction,
) -> Option<NodeId> {
	let current_rect = snap.rect(current)?;
	let order = leaf_ids(session);
	let mut best = None;

	for (rank, id) in order.into_iter().enumerate() {
		if id != current
			&& let Some(candidate) = snap.rect(id)
			&& let Some(score) = nav_score_oracle(current_rect, candidate, dir, rank)
			&& best.is_none_or(|(_, best_score)| score < best_score)
		{
			best = Some((id, score));
		}
	}

	best.map(|(id, _)| id)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RefEligibleSplit {
	pub split: NodeId,
	pub total: u32,
	pub current_a: u32,
	pub lo: u32,
	pub hi: u32,
}

impl RefEligibleSplit {
	pub fn slack(self, sign: i8) -> u32 {
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

pub fn eligible_splits_oracle<T>(
	session: &Session<T>, snap: &glorp_tiles::Snapshot, focus: NodeId, dir: Direction,
) -> Vec<RefEligibleSplit> {
	let focus_rect = snap.rect(focus).expect("focus rect should exist");
	let mut summaries = HashMap::new();
	if let Some(root) = session.tree().root_id() {
		let _ = summarize_reference(session.tree(), root, &mut summaries);
	}

	session
		.tree()
		.ancestors_nearest_first(focus)
		.expect("focus should exist")
		.into_iter()
		.filter_map(|split_id| {
			let split = session.tree().split(split_id)?;
			let a_rect = snap.rect(split.a())?;
			let b_rect = snap.rect(split.b())?;
			let focus_in_a = session.tree().contains_in_subtree(split.a(), focus);
			let eligible = match dir {
				Direction::Right => split.axis() == Axis::X && focus_in_a && focus_rect.right() == a_rect.right(),
				Direction::Left => split.axis() == Axis::X && !focus_in_a && focus_rect.left() == b_rect.left(),
				Direction::Down => split.axis() == Axis::Y && focus_in_a && focus_rect.bottom() == a_rect.bottom(),
				Direction::Up => split.axis() == Axis::Y && !focus_in_a && focus_rect.top() == b_rect.top(),
			};
			eligible.then(|| {
				let total = a_rect.extent(split.axis()) + b_rect.extent(split.axis());
				let sum_a = summaries[&split.a()];
				let sum_b = summaries[&split.b()];
				let (min_a, max_a) = sum_a.axis_limits(split.axis());
				let (min_b, max_b) = sum_b.axis_limits(split.axis());
				let min_a = u32::try_from(min_a).expect("test bounds should fit u32");
				let min_b = u32::try_from(min_b).expect("test bounds should fit u32");
				let max_a = max_a.map(|value| u32::try_from(value).expect("test bounds should fit u32"));
				let max_b = max_b.map(|value| u32::try_from(value).expect("test bounds should fit u32"));
				RefEligibleSplit {
					split: split_id,
					total,
					current_a: a_rect.extent(split.axis()),
					lo: min_a.max(max_b.map_or(0, |value| total.saturating_sub(value))),
					hi: total.saturating_sub(min_b).min(max_a.unwrap_or(total)),
				}
			})
		})
		.collect()
}

pub fn solve_reference<T>(tree: &Tree<T>, revision: u64, root: Rect, policy: SolverPolicy) -> RefSnapshot {
	let mut summaries = HashMap::new();
	let mut snapshot = RefSnapshot {
		revision,
		root,
		node_rects: BTreeMap::new(),
		split_traces: Vec::new(),
		violations: Vec::new(),
		strict_feasible: true,
	};
	let Some(root_id) = tree.root_id() else {
		return snapshot;
	};
	summarize_reference(tree, root_id, &mut summaries);
	solve_reference_node(tree, root_id, root, &summaries, policy, &mut snapshot);
	snapshot.strict_feasible = snapshot.violations.is_empty();
	snapshot
}

fn solve_reference_node<T>(
	tree: &Tree<T>, id: NodeId, rect: Rect, summaries: &HashMap<NodeId, RefSummary>, policy: SolverPolicy,
	out: &mut RefSnapshot,
) {
	out.node_rects.insert(id, rect);
	if let Some(leaf) = tree.leaf(id) {
		record_leaf_violations(id, rect, &leaf.meta().limits, out);
	} else {
		let split = tree.split(id).expect("missing node");
		let sum_a = summaries[&split.a()];
		let sum_b = summaries[&split.b()];
		let spec = PairSpec {
			total: rect.extent(split.axis()),
			min_a: u32::try_from(sum_a.axis_limits(split.axis()).0).expect("test bounds should fit u32"),
			min_b: u32::try_from(sum_b.axis_limits(split.axis()).0).expect("test bounds should fit u32"),
			max_a: sum_a
				.axis_limits(split.axis())
				.1
				.map(|value| u32::try_from(value).expect("test bounds should fit u32")),
			max_b: sum_b
				.axis_limits(split.axis())
				.1
				.map(|value| u32::try_from(value).expect("test bounds should fit u32")),
			wa: split.weights().a,
			wb: split.weights().b,
			sa: sum_a.shrink_cost,
			sb: sum_b.shrink_cost,
		};
		let (chosen_a, score) = choose_extent_oracle(spec, policy);
		let (rect_a, rect_b) = rect.split(split.axis(), chosen_a);
		out.split_traces.push(glorp_tiles::SplitTrace {
			split: id,
			axis: split.axis(),
			total: spec.total,
			chosen_a,
			score,
			weights: split.weights(),
		});
		solve_reference_node(tree, split.a(), rect_a, summaries, policy, out);
		solve_reference_node(tree, split.b(), rect_b, summaries, policy, out);
	}
}

fn summarize_reference<T>(tree: &Tree<T>, id: NodeId, out: &mut HashMap<NodeId, RefSummary>) -> RefSummary {
	if let Some(summary) = out.get(&id).copied() {
		return summary;
	}
	let summary = if let Some(leaf) = tree.leaf(id) {
		RefSummary {
			min_w: u64::from(leaf.meta().limits.min_w),
			min_h: u64::from(leaf.meta().limits.min_h),
			max_w: leaf.meta().limits.max_w.map(u64::from),
			max_h: leaf.meta().limits.max_h.map(u64::from),
			leaf_count: 1,
			shrink_cost: u64::from(leaf.meta().priority.shrink),
			grow_cost: u64::from(leaf.meta().priority.grow),
		}
	} else {
		let split = tree.split(id).expect("missing node in reference summary");
		let a = summarize_reference(tree, split.a(), out);
		let b = summarize_reference(tree, split.b(), out);
		match split.axis() {
			Axis::X => RefSummary {
				min_w: checked_add_u64(a.min_w, b.min_w, "min_w"),
				min_h: a.min_h.max(b.min_h),
				max_w: checked_add_option_u64(a.max_w, b.max_w, "max_w"),
				max_h: min_option_u64(a.max_h, b.max_h),
				leaf_count: checked_add_u64(a.leaf_count, b.leaf_count, "leaf_count"),
				shrink_cost: checked_add_u64(a.shrink_cost, b.shrink_cost, "shrink_cost"),
				grow_cost: checked_add_u64(a.grow_cost, b.grow_cost, "grow_cost"),
			},
			Axis::Y => RefSummary {
				min_w: a.min_w.max(b.min_w),
				min_h: checked_add_u64(a.min_h, b.min_h, "min_h"),
				max_w: min_option_u64(a.max_w, b.max_w),
				max_h: checked_add_option_u64(a.max_h, b.max_h, "max_h"),
				leaf_count: checked_add_u64(a.leaf_count, b.leaf_count, "leaf_count"),
				shrink_cost: checked_add_u64(a.shrink_cost, b.shrink_cost, "shrink_cost"),
				grow_cost: checked_add_u64(a.grow_cost, b.grow_cost, "grow_cost"),
			},
		}
	};
	out.insert(id, summary);
	summary
}

fn checked_add_u64(a: u64, b: u64, field: &str) -> u64 {
	a.checked_add(b)
		.unwrap_or_else(|| panic!("reference summary overflow in {field}"))
}

fn checked_add_option_u64(a: Option<u64>, b: Option<u64>, field: &str) -> Option<u64> {
	match (a, b) {
		(Some(a), Some(b)) => Some(checked_add_u64(a, b, field)),
		_ => None,
	}
}

fn min_option_u64(a: Option<u64>, b: Option<u64>) -> Option<u64> {
	match (a, b) {
		(Some(a), Some(b)) => Some(a.min(b)),
		(Some(a), None) => Some(a),
		(None, Some(b)) => Some(b),
		(None, None) => None,
	}
}

fn record_leaf_violations(node: NodeId, rect: Rect, limits: &SizeLimits, out: &mut RefSnapshot) {
	if rect.w < limits.min_w {
		out.violations.push(glorp_tiles::Violation {
			node,
			kind: glorp_tiles::ViolationKind::MinWidth,
			required: limits.min_w,
			actual: rect.w,
		});
	}
	if rect.h < limits.min_h {
		out.violations.push(glorp_tiles::Violation {
			node,
			kind: glorp_tiles::ViolationKind::MinHeight,
			required: limits.min_h,
			actual: rect.h,
		});
	}
	if let Some(max_w) = limits.max_w.filter(|max_w| rect.w > *max_w) {
		out.violations.push(glorp_tiles::Violation {
			node,
			kind: glorp_tiles::ViolationKind::MaxWidth,
			required: max_w,
			actual: rect.w,
		});
	}
	if let Some(max_h) = limits.max_h.filter(|max_h| rect.h > *max_h) {
		out.violations.push(glorp_tiles::Violation {
			node,
			kind: glorp_tiles::ViolationKind::MaxHeight,
			required: max_h,
			actual: rect.h,
		});
	}
}

pub fn exercise_trace(bytes: &[u8]) -> Session<u16> {
	let mut session = Session::new();
	let _ = session
		.insert_root(0, LeafMeta::default())
		.expect("root insert should work");
	let mut next_payload = 1_u16;
	for &byte in bytes {
		let leaves = leaf_ids(&session);
		let splits = split_ids(&session);
		let nodes = session.tree().node_ids();
		match byte % 12 {
			0 => {
				let _ = session.split_focus(axis(byte), slot(byte), next_payload, LeafMeta::default(), None);
				next_payload += 1;
			}
			1 => {
				let _ = session.wrap_selection(axis(byte), slot(byte), next_payload, LeafMeta::default(), None);
				next_payload += 1;
			}
			2 if leaves.len() > 1 => {
				let leaf = leaves[usize::from(byte) % leaves.len()];
				let _ = session.set_selection(leaf);
				let _ = session.remove_focus();
			}
			3 if !splits.is_empty() => {
				let split = splits[usize::from(byte) % splits.len()];
				if prepare_selected_split(&mut session, split) {
					let _ = session.toggle_axis();
				}
			}
			4 if !splits.is_empty() => {
				let split = splits[usize::from(byte) % splits.len()];
				if prepare_selected_split(&mut session, split) {
					let _ = session.mirror_selection(axis(byte));
				}
			}
			5 if !splits.is_empty() => {
				let split = splits[usize::from(byte) % splits.len()];
				if prepare_selected_split(&mut session, split) {
					let mode = if byte & 1 == 0 {
						RebalanceMode::BinaryEqual
					} else {
						RebalanceMode::LeafCount
					};
					let _ = session.rebalance_selection(mode);
				}
			}
			6 if !splits.is_empty() => {
				let split = splits[usize::from(byte) % splits.len()];
				if prepare_selected_split(&mut session, split) {
					let preset = match byte & 0b11 {
						0 => PresetKind::Balanced(BalancedPreset {
							start_axis: axis(byte),
							alternate: byte & 0b100 != 0,
						}),
						1 => PresetKind::Tall(TallPreset {
							master_slot: slot(byte),
							root_weights: glorp_tiles::WeightPair { a: 2, b: 1 },
						}),
						2 => PresetKind::Wide(WidePreset {
							master_slot: slot(byte),
							root_weights: glorp_tiles::WeightPair { a: 1, b: 3 },
						}),
						_ => PresetKind::Dwindle(glorp_tiles::DwindlePreset {
							start_axis: axis(byte),
							new_leaf_slot: slot(byte),
						}),
					};
					let _ = session.apply_preset(preset);
				}
			}
			7 if leaves.len() > 1 => {
				let focus = leaves[usize::from(byte) % leaves.len()];
				let _ = session.set_selection(focus);
				let snap = session.solve(root_rect(9, 7), &SolverPolicy::default()).expect("solve");
				let _ = session.focus_dir(direction(byte), &snap);
			}
			8 if leaves.len() > 1 => {
				let focus = leaves[usize::from(byte) % leaves.len()];
				let _ = session.set_selection(focus);
				let snap = session
					.solve(root_rect(11, 9), &SolverPolicy::default())
					.expect("solve");
				let _ = session.grow_focus(direction(byte), 2 + u32::from(byte & 0b11), strategy(byte), &snap);
			}
			9 if leaves.len() > 1 => {
				let focus = leaves[usize::from(byte) % leaves.len()];
				let _ = session.set_selection(focus);
				let snap = session
					.solve(root_rect(11, 9), &SolverPolicy::default())
					.expect("solve");
				let _ = session.shrink_focus(direction(byte), 1 + u32::from(byte & 0b11), strategy(byte), &snap);
			}
			10 if nodes.len() > 1 => {
				if let Some((a, b)) = first_disjoint_pair(&session, &nodes) {
					let _ = session.swap_nodes(a, b);
				}
			}
			11 if nodes.len() > 1 && !splits.is_empty() => {
				let split = splits[usize::from(byte) % splits.len()];
				// Pick from the implicit `nodes - {split}` sequence without allocating that filtered
				// vector on every trace step.
				let target_idx = usize::from(byte) % (nodes.len() - 1);
				if let Some(target) = nodes.iter().copied().filter(|target| *target != split).nth(target_idx) {
					let _ = prepare_selected_split(&mut session, split);
					let _ = session.move_selection_as_sibling_of(target, axis(byte), slot(byte));
				}
			}
			_ => {}
		}
		session.validate().expect("trace should preserve session invariants");
	}
	session
}

pub fn first_disjoint_pair<T>(session: &Session<T>, nodes: &[NodeId]) -> Option<(NodeId, NodeId)> {
	for (idx, a) in nodes.iter().enumerate() {
		for b in nodes.iter().skip(idx + 1) {
			if !session.tree().contains_in_subtree(*a, *b) && !session.tree().contains_in_subtree(*b, *a) {
				return Some((*a, *b));
			}
		}
	}
	None
}

fn prepare_selected_split<T>(session: &mut Session<T>, split: NodeId) -> bool {
	session.tree().first_leaf(split).is_some_and(|focus| {
		let _ = session.set_focus_leaf(focus);
		let _ = session.set_selection(split);
		true
	})
}

fn nav_score_oracle(current: Rect, candidate: Rect, dir: Direction, rank: usize) -> Option<(u32, u32, u64, usize)> {
	let (eligible, primary_gap, orth_gap, orth_center_delta) = match dir {
		Direction::Left => (
			candidate.right() <= current.left(),
			u32::try_from(current.left() - candidate.right()).ok()?,
			orth_gap(current.top(), current.bottom(), candidate.top(), candidate.bottom()),
			current
				.center_twice_orth(Axis::X)
				.abs_diff(candidate.center_twice_orth(Axis::X)),
		),
		Direction::Right => (
			candidate.left() >= current.right(),
			u32::try_from(candidate.left() - current.right()).ok()?,
			orth_gap(current.top(), current.bottom(), candidate.top(), candidate.bottom()),
			current
				.center_twice_orth(Axis::X)
				.abs_diff(candidate.center_twice_orth(Axis::X)),
		),
		Direction::Up => (
			candidate.bottom() <= current.top(),
			u32::try_from(current.top() - candidate.bottom()).ok()?,
			orth_gap(current.left(), current.right(), candidate.left(), candidate.right()),
			current
				.center_twice_orth(Axis::Y)
				.abs_diff(candidate.center_twice_orth(Axis::Y)),
		),
		Direction::Down => (
			candidate.top() >= current.bottom(),
			u32::try_from(candidate.top() - current.bottom()).ok()?,
			orth_gap(current.left(), current.right(), candidate.left(), candidate.right()),
			current
				.center_twice_orth(Axis::Y)
				.abs_diff(candidate.center_twice_orth(Axis::Y)),
		),
	};

	eligible.then_some((primary_gap, orth_gap, orth_center_delta, rank))
}

pub fn axis(byte: u8) -> Axis {
	if byte & 1 == 0 { Axis::X } else { Axis::Y }
}

pub fn slot(byte: u8) -> Slot {
	if byte & 2 == 0 { Slot::A } else { Slot::B }
}

pub fn direction(byte: u8) -> Direction {
	match byte & 0b11 {
		0 => Direction::Left,
		1 => Direction::Right,
		2 => Direction::Up,
		_ => Direction::Down,
	}
}

pub fn strategy(byte: u8) -> ResizeStrategy {
	match byte % 3 {
		0 => ResizeStrategy::Local,
		1 => ResizeStrategy::AncestorChain,
		_ => ResizeStrategy::DistributedBySlack,
	}
}

pub fn stressed_policy(seed: u8) -> SolverPolicy {
	SolverPolicy {
		shortage_mode: if seed & 1 == 0 {
			ShortageMode::Equal
		} else {
			ShortageMode::ByShrinkPriority
		},
		overflow_mode: glorp_tiles::OverflowMode::Uniform,
		tie_break: if seed & 2 == 0 {
			TieBreakMode::PreferA
		} else {
			TieBreakMode::PreferB
		},
	}
}
