//! Deterministic best-effort solver for binary split trees.
//!
//! Solving proceeds in two phases: [`summarize`] computes the exact feasible envelope of each
//! subtree, then the solver chooses an extent for every split by lexicographically minimizing a
//! scored allocation problem. Non-strict entry points always return a full [`Snapshot`] for valid
//! trees, while strict entry points reject any solve that records hard-limit violations.

use {
	crate::{
		error::{SolveError, ValidationError},
		geom::{Axis, Rect},
		ids::NodeId,
		limits::{Summary, WeightPair},
		snapshot::{ScoreTuple, Snapshot, SplitTrace, Violation, ViolationKind},
		tree::Tree,
	},
	serde::{Deserialize, Serialize},
	std::collections::HashMap,
};

/// Strategy for pricing shortage below child minimum extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ShortageMode {
	/// Count each missing unit equally, regardless of per-leaf priority.
	Equal,
	/// Weight each missing unit by the subtree's aggregate shrink priority.
	ByShrinkPriority,
}

/// Strategy for pricing overflow beyond child maximum extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OverflowMode {
	/// Count overflow units uniformly.
	Uniform,
}

/// Final deterministic tie-break preference when earlier score components tie.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TieBreakMode {
	/// Prefer giving more extent to child `A`.
	PreferA,
	/// Prefer giving more extent to child `B`.
	PreferB,
}

/// Solver configuration.
///
/// The default policy favors respecting shrink priorities, counts overflow uniformly, and breaks
/// perfect ties toward child `A` for deterministic repeatability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SolverPolicy {
	/// How shortage below minimum extents is priced.
	pub shortage_mode: ShortageMode,
	/// How overflow above maximum extents is priced.
	pub overflow_mode: OverflowMode,
	/// Deterministic preference when earlier score components tie.
	pub tie_break: TieBreakMode,
}

impl Default for SolverPolicy {
	fn default() -> Self {
		Self {
			shortage_mode: ShortageMode::ByShrinkPriority,
			overflow_mode: OverflowMode::Uniform,
			tie_break: TieBreakMode::PreferA,
		}
	}
}

/// Fully materialized single-split allocation problem.
///
/// This captures the total extent available on one axis, the child minimum and optional maximum
/// extents on that axis, preferred weights, and subtree shortage costs. Callers are expected to
/// supply a valid non-zero weight pair; `(0, 0)` is outside the contract even though it is not
/// revalidated here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairSpec {
	/// Total extent available to the split on the chosen axis.
	pub total: u32,
	/// Minimum feasible extent for child `A`.
	pub min_a: u32,
	/// Minimum feasible extent for child `B`.
	pub min_b: u32,
	/// Maximum feasible extent for child `A`, if bounded.
	pub max_a: Option<u32>,
	/// Maximum feasible extent for child `B`, if bounded.
	pub max_b: Option<u32>,
	/// Relative weight preference for child `A`.
	pub wa: u32,
	/// Relative weight preference for child `B`.
	pub wb: u32,
	/// Aggregate shortage cost for child `A`.
	pub sa: u64,
	/// Aggregate shortage cost for child `B`.
	pub sb: u64,
}

/// Chooses an extent for child `A` from raw split inputs.
///
/// This is a convenience wrapper around [`choose_extent_with_score`] that returns only the chosen
/// extent.
#[must_use]
pub fn choose_extent(
	total: u32, a_limits: (u32, Option<u32>), b_limits: (u32, Option<u32>), weights: WeightPair, a_shrink_cost: u64,
	b_shrink_cost: u64, policy: &SolverPolicy,
) -> u32 {
	let spec = PairSpec {
		total,
		min_a: a_limits.0,
		min_b: b_limits.0,
		max_a: a_limits.1,
		max_b: b_limits.1,
		wa: weights.a,
		wb: weights.b,
		sa: a_shrink_cost,
		sb: b_shrink_cost,
	};
	choose_extent_with_score(spec, policy).0
}

/// Chooses an extent for child `A` and returns the corresponding lexicographic score.
///
/// The current implementation exhaustively searches `0..=total`, computes [`score`] for each
/// candidate, and returns the deterministic minimum over `(score, tie-break key)`.
#[must_use]
pub fn choose_extent_with_score(spec: PairSpec, policy: &SolverPolicy) -> (u32, ScoreTuple) {
	(0..=spec.total)
		.map(|a| (a, score(spec, a, policy)))
		.min_by_key(|(a, score)| (*score, tie_break_key(*a, spec.total, policy.tie_break, score.tie_break)))
		.expect("total extent search space is never empty")
}

fn tie_break_key(a: u32, total: u32, mode: TieBreakMode, fallback: u128) -> u128 {
	if fallback != 0 {
		return fallback;
	}
	match mode {
		TieBreakMode::PreferA => u128::from(total - a),
		TieBreakMode::PreferB => u128::from(a),
	}
}

/// Scores a candidate allocation for child `A`.
///
/// The returned [`ScoreTuple`] is compared lexicographically in this order:
///
/// 1. shortage penalty
/// 2. overflow penalty
/// 3. preference penalty
/// 4. tie-break
#[must_use]
pub fn score(spec: PairSpec, a: u32, policy: &SolverPolicy) -> ScoreTuple {
	let size_a = a;
	let size_b = spec.total - a;
	let short_a = spec.min_a.saturating_sub(size_a);
	let short_b = spec.min_b.saturating_sub(size_b);
	let over_a = spec.max_a.map_or(0, |max| size_a.saturating_sub(max));
	let over_b = spec.max_b.map_or(0, |max| size_b.saturating_sub(max));
	let pref = pref_penalty(spec.total, a, spec.wa, spec.wb);
	let tie_break = match policy.tie_break {
		TieBreakMode::PreferA => u128::from(spec.total - a),
		TieBreakMode::PreferB => u128::from(a),
	};

	ScoreTuple {
		shortage_penalty: match policy.shortage_mode {
			ShortageMode::Equal => u128::from(short_a) + u128::from(short_b),
			ShortageMode::ByShrinkPriority => {
				u128::from(short_a) * u128::from(spec.sa) + u128::from(short_b) * u128::from(spec.sb)
			}
		},
		overflow_penalty: match policy.overflow_mode {
			OverflowMode::Uniform => u128::from(over_a) + u128::from(over_b),
		},
		preference_penalty: pref,
		tie_break,
	}
}

fn pref_penalty(total: u32, a: u32, wa: u32, wb: u32) -> u128 {
	let total_weight = u128::from(wa) + u128::from(wb);
	let left = u128::from(a) * total_weight;
	let right = u128::from(total) * u128::from(wa);
	left.abs_diff(right)
}

/// Solves `tree` into a best-effort snapshot with revision `0`.
///
/// The tree is validated before solving. For valid trees this returns a full [`Snapshot`] even
/// when some solved leaves violate hard limits; inspect [`Snapshot::strict_feasible`] and
/// [`Snapshot::violations`] to distinguish strict feasibility from best effort.
pub fn solve<T>(tree: &Tree<T>, root: Rect, policy: &SolverPolicy) -> Result<Snapshot, SolveError> {
	solve_with_revision(tree, root, 0, policy)
}

/// Solves `tree` into a best-effort snapshot tagged with `revision`.
///
/// This differs from [`solve`] only in the stored [`Snapshot::revision`]. The tree is validated
/// before solving. For valid trees this returns a full [`Snapshot`] even when some solved leaves
/// violate hard limits; strict feasibility is reported through [`Snapshot::strict_feasible`].
pub fn solve_with_revision<T>(
	tree: &Tree<T>, root: Rect, revision: u64, policy: &SolverPolicy,
) -> Result<Snapshot, SolveError> {
	tree.validate().map_err(SolveError::Validation)?;
	let mut snapshot = Snapshot {
		revision,
		root,
		node_rects: HashMap::new(),
		split_traces: Vec::new(),
		violations: Vec::new(),
		strict_feasible: true,
	};
	let Some(root_id) = tree.root_id() else {
		return Ok(snapshot);
	};
	let mut summaries = HashMap::new();
	summarize(tree, root_id, &mut summaries).map_err(SolveError::Validation)?;
	solve_node(tree, root_id, root, &summaries, policy, &mut snapshot)?;
	snapshot.strict_feasible = snapshot.violations.is_empty();
	Ok(snapshot)
}

/// Solves `tree` and rejects any hard-limit violation.
///
/// This validates first, then delegates to [`solve`]. A valid tree whose best-effort solve
/// produces any violation returns [`SolveError::Infeasible`].
///
/// ```
/// use libtiler::{
///     solve, solve_strict, Axis, LeafMeta, Rect, Session, SizeLimits, Slot, SolveError,
///     SolverPolicy,
/// };
///
/// let tight = LeafMeta {
///     limits: SizeLimits {
///         min_w: 5,
///         ..SizeLimits::default()
///     },
///     ..LeafMeta::default()
/// };
/// let mut session = Session::new();
/// session.insert_root("a", tight.clone())?;
/// session.split_focus(Axis::X, Slot::B, "b", tight, None)?;
///
/// let root = Rect { x: 0, y: 0, w: 8, h: 4 };
/// let snapshot = solve(session.tree(), root, &SolverPolicy::default())?;
/// assert!(!snapshot.strict_feasible);
/// assert_eq!(
///     solve_strict(session.tree(), root, &SolverPolicy::default()),
///     Err(SolveError::Infeasible)
/// );
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn solve_strict<T>(tree: &Tree<T>, root: Rect, policy: &SolverPolicy) -> Result<Snapshot, SolveError> {
	let snapshot = solve(tree, root, policy)?;
	if snapshot.strict_feasible {
		Ok(snapshot)
	} else {
		Err(SolveError::Infeasible)
	}
}

/// Strict solve variant that tags the returned snapshot with `revision`.
///
/// This differs from [`solve_strict`] only in the stored [`Snapshot::revision`].
pub fn solve_strict_with_revision<T>(
	tree: &Tree<T>, root: Rect, revision: u64, policy: &SolverPolicy,
) -> Result<Snapshot, SolveError> {
	let snapshot = solve_with_revision(tree, root, revision, policy)?;
	if snapshot.strict_feasible {
		Ok(snapshot)
	} else {
		Err(SolveError::Infeasible)
	}
}

/// Computes the exact bottom-up summary envelope for the subtree rooted at `id`.
///
/// The resulting [`Summary`] captures minimum and optional maximum extents, leaf count, and
/// aggregate cost weights for the entire subtree. Results are memoized into `out`. Arithmetic
/// overflow while combining subtree values is reported as [`ValidationError::ArithmeticOverflow`].
pub fn summarize<T>(
	tree: &Tree<T>, id: NodeId, out: &mut HashMap<NodeId, Summary>,
) -> Result<Summary, ValidationError> {
	if let Some(summary) = out.get(&id).copied() {
		return Ok(summary);
	}
	let summary = if let Some(leaf) = tree.leaf(id) {
		Summary {
			min_w: leaf.meta().limits.min_w,
			min_h: leaf.meta().limits.min_h,
			max_w: leaf.meta().limits.max_w,
			max_h: leaf.meta().limits.max_h,
			leaf_count: 1,
			shrink_cost: u64::from(leaf.meta().priority.shrink),
			grow_cost: u64::from(leaf.meta().priority.grow),
		}
	} else {
		let split = tree.split(id).ok_or(ValidationError::MissingNode(id))?;
		let a = summarize(tree, split.a(), out)?;
		let b = summarize(tree, split.b(), out)?;
		match split.axis() {
			Axis::X => Summary {
				min_w: checked_add_u32(a.min_w, b.min_w, id, "min_w")?,
				min_h: a.min_h.max(b.min_h),
				max_w: checked_add_option_u32(a.max_w, b.max_w, id, "max_w")?,
				max_h: min_option(a.max_h, b.max_h),
				leaf_count: checked_add_u32(a.leaf_count, b.leaf_count, id, "leaf_count")?,
				shrink_cost: checked_add_u64(a.shrink_cost, b.shrink_cost, id, "shrink_cost")?,
				grow_cost: checked_add_u64(a.grow_cost, b.grow_cost, id, "grow_cost")?,
			},
			Axis::Y => Summary {
				min_w: a.min_w.max(b.min_w),
				min_h: checked_add_u32(a.min_h, b.min_h, id, "min_h")?,
				max_w: min_option(a.max_w, b.max_w),
				max_h: checked_add_option_u32(a.max_h, b.max_h, id, "max_h")?,
				leaf_count: checked_add_u32(a.leaf_count, b.leaf_count, id, "leaf_count")?,
				shrink_cost: checked_add_u64(a.shrink_cost, b.shrink_cost, id, "shrink_cost")?,
				grow_cost: checked_add_u64(a.grow_cost, b.grow_cost, id, "grow_cost")?,
			},
		}
	};
	out.insert(id, summary);
	Ok(summary)
}

fn checked_add_u32(a: u32, b: u32, node: NodeId, field: &'static str) -> Result<u32, ValidationError> {
	a.checked_add(b)
		.ok_or(ValidationError::ArithmeticOverflow { node, field })
}

fn checked_add_u64(a: u64, b: u64, node: NodeId, field: &'static str) -> Result<u64, ValidationError> {
	a.checked_add(b)
		.ok_or(ValidationError::ArithmeticOverflow { node, field })
}

fn checked_add_option_u32(
	a: Option<u32>, b: Option<u32>, node: NodeId, field: &'static str,
) -> Result<Option<u32>, ValidationError> {
	match (a, b) {
		(Some(a), Some(b)) => checked_add_u32(a, b, node, field).map(Some),
		_ => Ok(None),
	}
}

fn min_option(a: Option<u32>, b: Option<u32>) -> Option<u32> {
	match (a, b) {
		(Some(a), Some(b)) => Some(a.min(b)),
		(Some(a), None) => Some(a),
		(None, Some(b)) => Some(b),
		(None, None) => None,
	}
}

fn solve_node<T>(
	tree: &Tree<T>, id: NodeId, rect: Rect, summaries: &HashMap<NodeId, Summary>, policy: &SolverPolicy,
	out: &mut Snapshot,
) -> Result<(), SolveError> {
	out.node_rects.insert(id, rect);
	if let Some(leaf) = tree.leaf(id) {
		record_leaf_violations(id, rect, &leaf.meta().limits, out);
	} else {
		let split = tree
			.split(id)
			.ok_or(SolveError::Validation(ValidationError::MissingNode(id)))?;
		let sum_a = summaries
			.get(&split.a())
			.copied()
			.ok_or(SolveError::Validation(ValidationError::MissingNode(split.a())))?;
		let sum_b = summaries
			.get(&split.b())
			.copied()
			.ok_or(SolveError::Validation(ValidationError::MissingNode(split.b())))?;
		let total = rect.extent(split.axis());
		let spec = PairSpec {
			total,
			min_a: sum_a.axis_limits(split.axis()).0,
			min_b: sum_b.axis_limits(split.axis()).0,
			max_a: sum_a.axis_limits(split.axis()).1,
			max_b: sum_b.axis_limits(split.axis()).1,
			wa: split.weights().a,
			wb: split.weights().b,
			sa: sum_a.shrink_cost,
			sb: sum_b.shrink_cost,
		};
		let (chosen_a, chosen_score) = choose_extent_with_score(spec, policy);
		let (rect_a, rect_b) = rect.split(split.axis(), chosen_a);
		out.split_traces.push(SplitTrace {
			split: id,
			axis: split.axis(),
			total,
			chosen_a,
			score: chosen_score,
			weights: split.weights(),
		});
		solve_node(tree, split.a(), rect_a, summaries, policy, out)?;
		solve_node(tree, split.b(), rect_b, summaries, policy, out)?;
	}
	Ok(())
}

fn record_leaf_violations(node: NodeId, rect: Rect, limits: &crate::limits::SizeLimits, out: &mut Snapshot) {
	if rect.w < limits.min_w {
		out.violations.push(Violation {
			node,
			kind: ViolationKind::MinWidth,
			required: limits.min_w,
			actual: rect.w,
		});
	}
	if rect.h < limits.min_h {
		out.violations.push(Violation {
			node,
			kind: ViolationKind::MinHeight,
			required: limits.min_h,
			actual: rect.h,
		});
	}
	if let Some(max_w) = limits.max_w.filter(|max_w| rect.w > *max_w) {
		out.violations.push(Violation {
			node,
			kind: ViolationKind::MaxWidth,
			required: max_w,
			actual: rect.w,
		});
	}
	if let Some(max_h) = limits.max_h.filter(|max_h| rect.h > *max_h) {
		out.violations.push(Violation {
			node,
			kind: ViolationKind::MaxHeight,
			required: max_h,
			actual: rect.h,
		});
	}
}
