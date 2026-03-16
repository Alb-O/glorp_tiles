//! Deterministic subtree rebuild presets.
//!
//! Presets rebuild a selected subtree from its leaves in stable depth-first `A`-before-`B` order.
//! Leaf ids, payloads, and metadata are preserved across the rebuild, while split ids may change.
//! Presets operate purely on topology and leaf order; they do not inspect solved geometry.
//!
//! ```text
//! leaf order consumed by preset rebuild: [a, b, c, d]
//!
//! Balanced(start_axis = X, alternate = true)
//!         [X]
//!        /   \
//!      [Y]   [Y]
//!     /   \ /   \
//!   [a] [b][c] [d]
//!
//! Dwindle(start_axis = X, new_leaf_slot = B)
//!           [X]
//!          /   \
//!        [a]   [Y]
//!             /   \
//!           [b]   [X]
//!                /   \
//!              [c]   [d]
//!
//! Tall(master_slot = A)
//!           [X]
//!          /   \
//!        [a]   [Y]
//!             /   \
//!           [b]   [Y]
//!                /   \
//!              [c]   [d]
//!
//! Wide(master_slot = A)
//!           [Y]
//!          /   \
//!        [a]   [X]
//!             /   \
//!           [b]   [X]
//!                /   \
//!              [c]   [d]
//! ```
//!
//! For [`TallPreset`] and [`WidePreset`], only the root split uses caller-provided
//! `root_weights`; the stack side is rebuilt as equal-share linear splits.

use {
	crate::{
		error::OpError,
		geom::{Axis, Slot},
		ids::NodeId,
		limits::{WeightPair, canonicalize_weights},
		tree::Tree,
	},
	serde::{Deserialize, Serialize},
};

/// Balanced midpoint-splitting preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BalancedPreset {
	/// Axis used at the root of the rebuilt subtree.
	pub start_axis: Axis,
	/// Whether each recursive level toggles axis instead of repeating `start_axis`.
	pub alternate: bool,
}

/// Alternating chain preset that inserts one new leaf per recursive step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DwindlePreset {
	/// Axis used at the root of the rebuilt subtree.
	pub start_axis: Axis,
	/// Side that receives the next leaf at each recursive step.
	pub new_leaf_slot: Slot,
}

/// Master-and-stack preset with a horizontal root split.
///
/// The non-master side is rebuilt as an equal-share vertical stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TallPreset {
	/// Side of the master leaf in the root split.
	pub master_slot: Slot,
	/// Weight pair applied only to the root split between master and stack.
	pub root_weights: WeightPair,
}

/// Master-and-stack preset with a vertical root split.
///
/// The non-master side is rebuilt as an equal-share horizontal stack.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WidePreset {
	/// Side of the master leaf in the root split.
	pub master_slot: Slot,
	/// Weight pair applied only to the root split between master and stack.
	pub root_weights: WeightPair,
}

/// Public preset selector for subtree rebuild operations.
///
/// ```
/// use glorp_tiles::{Axis, BalancedPreset, LeafMeta, PresetKind, Session, Slot};
///
/// let mut session = Session::new();
/// let focus = session.insert_root("a", LeafMeta::default())?;
/// let _b = session.split_focus(Axis::X, Slot::B, "b", LeafMeta::default(), None)?;
/// let root = session.tree().root_id().expect("root should exist");
///
/// session.set_selection(root)?;
/// session.apply_preset(PresetKind::Balanced(BalancedPreset {
///     start_axis: Axis::Y,
///     alternate: true,
/// }))?;
///
/// assert_eq!(session.focus(), Some(focus));
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresetKind {
	/// Midpoint-balanced recursive shape using a left-biased midpoint split.
	Balanced(BalancedPreset),
	/// Alternating chain shape that repeatedly inserts the next leaf on a chosen side.
	Dwindle(DwindlePreset),
	/// One master leaf plus an orthogonal stack of remaining leaves.
	Tall(TallPreset),
	/// Rotated tall layout with one master leaf plus an orthogonal stack.
	Wide(WidePreset),
}

/// Validates public preset parameters before a rebuild.
///
/// Only [`PresetKind::Tall`] and [`PresetKind::Wide`] can fail validation, and only when their
/// `root_weights` are invalid.
pub(crate) fn validate_preset(preset: PresetKind) -> Result<(), OpError> {
	match preset {
		PresetKind::Balanced(_) | PresetKind::Dwindle(_) => Ok(()),
		PresetKind::Tall(TallPreset { root_weights, .. }) | PresetKind::Wide(WidePreset { root_weights, .. }) => {
			validate_root_weights(root_weights)
		}
	}
}

pub(crate) fn build_preset_subtree<T>(
	tree: &mut Tree<T>, leaves: &[NodeId], preset: PresetKind,
) -> Result<NodeId, OpError> {
	match preset {
		PresetKind::Balanced(preset) => build_balanced(tree, leaves, preset),
		PresetKind::Dwindle(preset) => build_dwindle(tree, leaves, preset.start_axis, preset.new_leaf_slot),
		PresetKind::Tall(preset) => build_tall(tree, leaves, preset),
		PresetKind::Wide(preset) => build_wide(tree, leaves, preset),
	}
}

fn subtree_matches_preset_with_leaves<T>(tree: &Tree<T>, root: NodeId, leaves: &[NodeId], preset: PresetKind) -> bool {
	match preset {
		PresetKind::Balanced(preset) => matches_balanced(tree, root, leaves, preset),
		PresetKind::Dwindle(preset) => matches_dwindle(tree, root, leaves, preset.start_axis, preset.new_leaf_slot),
		PresetKind::Tall(preset) => matches_tall(tree, root, leaves, preset),
		PresetKind::Wide(preset) => matches_wide(tree, root, leaves, preset),
	}
}

/// Rebuilds `selection` to match `preset`.
///
/// Returns `Ok(None)` when `selection` is a leaf or the subtree already matches `preset`.
/// Otherwise the rebuild preserves existing leaves and reconnects a new split structure.
pub(crate) fn apply_preset_subtree<T>(
	tree: &mut Tree<T>, selection: NodeId, preset: PresetKind,
) -> Result<Option<NodeId>, OpError> {
	validate_preset(preset)?;
	if tree.is_leaf(selection) {
		return Ok(None);
	}
	let (leaves, split_ids) = collect_rebuild_parts(tree, selection);
	if subtree_matches_preset_with_leaves(tree, selection, &leaves, preset) {
		return Ok(None);
	}

	let parent = tree.parent_of(selection);

	for leaf in &leaves {
		tree.set_parent(*leaf, None);
	}
	for split in split_ids {
		tree.remove_node(split);
	}

	let rebuilt = build_preset_subtree(tree, &leaves, preset).expect("validated preset rebuild should succeed");
	if let Some(parent) = parent {
		tree.replace_child(parent, selection, rebuilt);
	} else {
		tree.set_root(Some(rebuilt));
		tree.set_parent(rebuilt, None);
	}

	Ok(Some(rebuilt))
}

fn collect_rebuild_parts<T>(tree: &Tree<T>, root: NodeId) -> (Vec<NodeId>, Vec<NodeId>) {
	let mut leaves = Vec::new();
	let mut split_ids = Vec::new();
	collect_rebuild_parts_inner(tree, root, &mut leaves, &mut split_ids);
	(leaves, split_ids)
}

fn collect_rebuild_parts_inner<T>(tree: &Tree<T>, id: NodeId, leaves: &mut Vec<NodeId>, split_ids: &mut Vec<NodeId>) {
	if let Some(split) = tree.split(id) {
		// One DFS yields both artifacts the rebuild needs: stable leaf order for reconstruction and
		// descendant-first split order for teardown.
		collect_rebuild_parts_inner(tree, split.a(), leaves, split_ids);
		collect_rebuild_parts_inner(tree, split.b(), leaves, split_ids);
		split_ids.push(id);
	} else {
		debug_assert!(
			tree.is_leaf(id),
			"validated preset rebuild should only visit leaves or splits"
		);
		leaves.push(id);
	}
}

fn build_balanced<T>(tree: &mut Tree<T>, leaves: &[NodeId], preset: BalancedPreset) -> Result<NodeId, OpError> {
	if leaves.is_empty() {
		return Err(OpError::Empty);
	}
	if leaves.len() == 1 {
		return Ok(leaves[0]);
	}
	let mid = leaves.len().div_ceil(2);
	let next_preset = next_balanced_preset(preset);
	let a = build_balanced(tree, &leaves[..mid], next_preset)?;
	let b = build_balanced(tree, &leaves[mid..], next_preset)?;
	Ok(new_internal_split(
		tree,
		preset.start_axis,
		a,
		b,
		leaf_count_weights(mid, leaves.len() - mid),
	))
}

fn build_dwindle<T>(tree: &mut Tree<T>, leaves: &[NodeId], axis: Axis, slot: Slot) -> Result<NodeId, OpError> {
	let Some((&last, rest)) = leaves.split_last() else {
		return Err(OpError::Empty);
	};
	// Build from the tail inward so each step wraps the subtree that would appear "after" the next
	// leaf in stable DFS order.
	let mut subtree = last;
	// Reverse iteration flips the alternation parity relative to the public root axis, so seed the
	// first rebuilt split with the axis that would have appeared deepest in forward construction.
	let mut split_axis = if rest.len() % 2 == 0 { axis.toggled() } else { axis };
	for leaf in rest.iter().copied().rev() {
		let (a, b) = match slot {
			Slot::A => (subtree, leaf),
			Slot::B => (leaf, subtree),
		};
		subtree = new_internal_split(tree, split_axis, a, b, WeightPair::default());
		split_axis = split_axis.toggled();
	}
	Ok(subtree)
}

fn build_tall<T>(tree: &mut Tree<T>, leaves: &[NodeId], preset: TallPreset) -> Result<NodeId, OpError> {
	build_master_stack(tree, leaves, Axis::X, Axis::Y, preset.master_slot, preset.root_weights)
}

fn build_wide<T>(tree: &mut Tree<T>, leaves: &[NodeId], preset: WidePreset) -> Result<NodeId, OpError> {
	build_master_stack(tree, leaves, Axis::Y, Axis::X, preset.master_slot, preset.root_weights)
}

fn build_master_stack<T>(
	tree: &mut Tree<T>, leaves: &[NodeId], root_axis: Axis, stack_axis: Axis, master_slot: Slot,
	root_weights: WeightPair,
) -> Result<NodeId, OpError> {
	if leaves.is_empty() {
		return Err(OpError::Empty);
	}
	if leaves.len() == 1 {
		return Ok(leaves[0]);
	}
	let master = leaves[0];
	let stack = build_equal_linear(tree, &leaves[1..], stack_axis)?;
	let (a, b) = match master_slot {
		Slot::A => (master, stack),
		Slot::B => (stack, master),
	};
	Ok(new_internal_split(
		tree,
		root_axis,
		a,
		b,
		root_weights.checked().ok_or(OpError::InvalidWeights)?,
	))
}

fn build_equal_linear<T>(tree: &mut Tree<T>, leaves: &[NodeId], axis: Axis) -> Result<NodeId, OpError> {
	let Some((&last, rest)) = leaves.split_last() else {
		return Err(OpError::Empty);
	};
	let mut subtree = last;
	for (subtree_leaf_count, leaf) in (1..).zip(rest.iter().copied().rev()) {
		// The zipped counter tracks the leaves already packed into `subtree`, so the next split can
		// reuse it directly instead of recovering the same number from reverse indices.
		subtree = new_internal_split(tree, axis, leaf, subtree, leaf_count_weights(1, subtree_leaf_count));
	}
	Ok(subtree)
}

fn matches_balanced<T>(tree: &Tree<T>, id: NodeId, leaves: &[NodeId], preset: BalancedPreset) -> bool {
	if leaves.is_empty() {
		return false;
	}
	if leaves.len() == 1 {
		return tree.is_leaf(id) && id == leaves[0];
	}
	let Some(split) = tree.split(id) else {
		return false;
	};
	if split.axis() != preset.start_axis {
		return false;
	}
	let mid = leaves.len().div_ceil(2);
	if split.weights() != leaf_count_weights(mid, leaves.len() - mid) {
		return false;
	}
	let next = next_balanced_preset(preset);
	matches_balanced(tree, split.a(), &leaves[..mid], next) && matches_balanced(tree, split.b(), &leaves[mid..], next)
}

fn matches_dwindle<T>(tree: &Tree<T>, id: NodeId, leaves: &[NodeId], axis: Axis, slot: Slot) -> bool {
	if leaves.is_empty() {
		return false;
	}
	if leaves.len() == 1 {
		return tree.is_leaf(id) && id == leaves[0];
	}
	let Some(split) = tree.split(id) else {
		return false;
	};
	if split.axis() != axis || split.weights() != WeightPair::default() {
		return false;
	}
	match slot {
		Slot::A => {
			split.b() == leaves[0]
				&& tree.is_leaf(split.b())
				&& matches_dwindle(tree, split.a(), &leaves[1..], axis.toggled(), slot)
		}
		Slot::B => {
			split.a() == leaves[0]
				&& tree.is_leaf(split.a())
				&& matches_dwindle(tree, split.b(), &leaves[1..], axis.toggled(), slot)
		}
	}
}

fn matches_tall<T>(tree: &Tree<T>, id: NodeId, leaves: &[NodeId], preset: TallPreset) -> bool {
	matches_master_stack(
		tree,
		id,
		leaves,
		Axis::X,
		Axis::Y,
		preset.master_slot,
		preset.root_weights,
	)
}

fn matches_wide<T>(tree: &Tree<T>, id: NodeId, leaves: &[NodeId], preset: WidePreset) -> bool {
	matches_master_stack(
		tree,
		id,
		leaves,
		Axis::Y,
		Axis::X,
		preset.master_slot,
		preset.root_weights,
	)
}

fn matches_master_stack<T>(
	tree: &Tree<T>, id: NodeId, leaves: &[NodeId], root_axis: Axis, stack_axis: Axis, master_slot: Slot,
	root_weights: WeightPair,
) -> bool {
	if leaves.is_empty() {
		return false;
	}
	if leaves.len() == 1 {
		return tree.is_leaf(id) && id == leaves[0];
	}
	let Some(split) = tree.split(id) else {
		return false;
	};
	if split.axis() != root_axis || split.weights() != root_weights {
		return false;
	}
	match master_slot {
		Slot::A => {
			split.a() == leaves[0]
				&& tree.is_leaf(split.a())
				&& matches_equal_linear(tree, split.b(), &leaves[1..], stack_axis)
		}
		Slot::B => {
			split.b() == leaves[0]
				&& tree.is_leaf(split.b())
				&& matches_equal_linear(tree, split.a(), &leaves[1..], stack_axis)
		}
	}
}

fn matches_equal_linear<T>(tree: &Tree<T>, id: NodeId, leaves: &[NodeId], axis: Axis) -> bool {
	if leaves.is_empty() {
		return false;
	}
	if leaves.len() == 1 {
		return tree.is_leaf(id) && id == leaves[0];
	}
	let Some(split) = tree.split(id) else {
		return false;
	};
	split.axis() == axis
		&& split.weights() == leaf_count_weights(1, leaves.len() - 1)
		&& split.a() == leaves[0]
		&& tree.is_leaf(split.a())
		&& matches_equal_linear(tree, split.b(), &leaves[1..], axis)
}

fn leaf_count_weights(a: usize, b: usize) -> WeightPair {
	canonicalize_weights(
		u32::try_from(a).expect("preset leaf count exceeds u32"),
		u32::try_from(b).expect("preset leaf count exceeds u32"),
	)
}

fn next_balanced_preset(preset: BalancedPreset) -> BalancedPreset {
	BalancedPreset {
		start_axis: if preset.alternate {
			preset.start_axis.toggled()
		} else {
			preset.start_axis
		},
		alternate: preset.alternate,
	}
}

fn validate_root_weights(weights: WeightPair) -> Result<(), OpError> {
	weights.checked().map(|_| ()).ok_or(OpError::InvalidWeights)
}

fn new_internal_split<T>(tree: &mut Tree<T>, axis: Axis, a: NodeId, b: NodeId, weights: WeightPair) -> NodeId {
	let split_id = tree.new_split(axis, a, b, weights);
	tree.set_parent(a, Some(split_id));
	tree.set_parent(b, Some(split_id));
	split_id
}
