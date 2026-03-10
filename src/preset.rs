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

use serde::{Deserialize, Serialize};

use crate::{
	error::OpError,
	geom::{Axis, Slot},
	ids::NodeId,
	limits::{WeightPair, canonicalize_weights},
	tree::Tree,
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
/// use libtiler::{Axis, BalancedPreset, LeafMeta, PresetKind, Session, Slot};
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
		PresetKind::Tall(preset) => preset.root_weights.checked().map(|_| ()).ok_or(OpError::InvalidWeights),
		PresetKind::Wide(preset) => preset.root_weights.checked().map(|_| ()).ok_or(OpError::InvalidWeights),
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

/// Returns whether `root` already matches `preset`.
///
/// Matching is checked against the subtree's current leaf order in stable depth-first
/// `A`-before-`B` order.
pub(crate) fn subtree_matches_preset<T>(tree: &Tree<T>, root: NodeId, preset: PresetKind) -> Result<bool, OpError> {
	validate_preset(preset)?;
	let leaves = tree.leaf_ids_dfs(root);
	match preset {
		PresetKind::Balanced(preset) => Ok(matches_balanced(tree, root, &leaves, preset)),
		PresetKind::Dwindle(preset) => Ok(matches_dwindle(
			tree,
			root,
			&leaves,
			preset.start_axis,
			preset.new_leaf_slot,
		)),
		PresetKind::Tall(preset) => Ok(matches_tall(tree, root, &leaves, preset)),
		PresetKind::Wide(preset) => Ok(matches_wide(tree, root, &leaves, preset)),
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
	if subtree_matches_preset(tree, selection, preset)? {
		return Ok(None);
	}

	let parent = tree.parent_of(selection);
	let leaves = tree.leaf_ids_dfs(selection);
	let split_ids = tree.split_ids_postorder(selection);

	for leaf in &leaves {
		tree.set_parent(*leaf, None);
	}
	for split in split_ids {
		tree.remove_node(split);
	}

	let rebuilt = build_preset_subtree_validated(tree, &leaves, preset);
	match parent {
		Some(parent) => {
			tree.replace_child(parent, selection, rebuilt);
		}
		None => {
			tree.set_root(Some(rebuilt));
			tree.set_parent(rebuilt, None);
		}
	}

	Ok(Some(rebuilt))
}

fn build_preset_subtree_validated<T>(tree: &mut Tree<T>, leaves: &[NodeId], preset: PresetKind) -> NodeId {
	build_preset_subtree(tree, leaves, preset).expect("validated preset rebuild should succeed")
}

fn build_balanced<T>(tree: &mut Tree<T>, leaves: &[NodeId], preset: BalancedPreset) -> Result<NodeId, OpError> {
	if leaves.is_empty() {
		return Err(OpError::Empty);
	}
	if leaves.len() == 1 {
		return Ok(leaves[0]);
	}
	let mid = leaves.len().div_ceil(2);
	let next_axis = if preset.alternate {
		preset.start_axis.toggled()
	} else {
		preset.start_axis
	};
	let a = build_balanced(
		tree,
		&leaves[..mid],
		BalancedPreset {
			start_axis: next_axis,
			alternate: preset.alternate,
		},
	)?;
	let b = build_balanced(
		tree,
		&leaves[mid..],
		BalancedPreset {
			start_axis: next_axis,
			alternate: preset.alternate,
		},
	)?;
	Ok(new_internal_split(
		tree,
		preset.start_axis,
		a,
		b,
		canonicalize_weights(mid as u32, (leaves.len() - mid) as u32),
	))
}

fn build_dwindle<T>(tree: &mut Tree<T>, leaves: &[NodeId], axis: Axis, slot: Slot) -> Result<NodeId, OpError> {
	if leaves.is_empty() {
		return Err(OpError::Empty);
	}
	if leaves.len() == 1 {
		return Ok(leaves[0]);
	}
	let first = leaves[0];
	let rest = build_dwindle(tree, &leaves[1..], axis.toggled(), slot)?;
	let (a, b) = match slot {
		Slot::A => (rest, first),
		Slot::B => (first, rest),
	};
	Ok(new_internal_split(tree, axis, a, b, WeightPair::default()))
}

fn build_tall<T>(tree: &mut Tree<T>, leaves: &[NodeId], preset: TallPreset) -> Result<NodeId, OpError> {
	if leaves.is_empty() {
		return Err(OpError::Empty);
	}
	if leaves.len() == 1 {
		return Ok(leaves[0]);
	}
	let master = leaves[0];
	let stack = build_equal_linear(tree, &leaves[1..], Axis::Y)?;
	let (a, b) = match preset.master_slot {
		Slot::A => (master, stack),
		Slot::B => (stack, master),
	};
	Ok(new_internal_split(
		tree,
		Axis::X,
		a,
		b,
		preset.root_weights.checked().ok_or(OpError::InvalidWeights)?,
	))
}

fn build_wide<T>(tree: &mut Tree<T>, leaves: &[NodeId], preset: WidePreset) -> Result<NodeId, OpError> {
	if leaves.is_empty() {
		return Err(OpError::Empty);
	}
	if leaves.len() == 1 {
		return Ok(leaves[0]);
	}
	let master = leaves[0];
	let stack = build_equal_linear(tree, &leaves[1..], Axis::X)?;
	let (a, b) = match preset.master_slot {
		Slot::A => (master, stack),
		Slot::B => (stack, master),
	};
	Ok(new_internal_split(
		tree,
		Axis::Y,
		a,
		b,
		preset.root_weights.checked().ok_or(OpError::InvalidWeights)?,
	))
}

fn build_equal_linear<T>(tree: &mut Tree<T>, leaves: &[NodeId], axis: Axis) -> Result<NodeId, OpError> {
	if leaves.is_empty() {
		return Err(OpError::Empty);
	}
	if leaves.len() == 1 {
		return Ok(leaves[0]);
	}
	let head = leaves[0];
	let rest = build_equal_linear(tree, &leaves[1..], axis)?;
	Ok(new_internal_split(
		tree,
		axis,
		head,
		rest,
		canonicalize_weights(1, (leaves.len() - 1) as u32),
	))
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
	if split.weights() != canonicalize_weights(mid as u32, (leaves.len() - mid) as u32) {
		return false;
	}
	let next = BalancedPreset {
		start_axis: if preset.alternate {
			preset.start_axis.toggled()
		} else {
			preset.start_axis
		},
		alternate: preset.alternate,
	};
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
	if leaves.is_empty() {
		return false;
	}
	if leaves.len() == 1 {
		return tree.is_leaf(id) && id == leaves[0];
	}
	let Some(split) = tree.split(id) else {
		return false;
	};
	if split.axis() != Axis::X || split.weights() != preset.root_weights {
		return false;
	}
	match preset.master_slot {
		Slot::A => {
			split.a() == leaves[0]
				&& tree.is_leaf(split.a())
				&& matches_equal_linear(tree, split.b(), &leaves[1..], Axis::Y)
		}
		Slot::B => {
			split.b() == leaves[0]
				&& tree.is_leaf(split.b())
				&& matches_equal_linear(tree, split.a(), &leaves[1..], Axis::Y)
		}
	}
}

fn matches_wide<T>(tree: &Tree<T>, id: NodeId, leaves: &[NodeId], preset: WidePreset) -> bool {
	if leaves.is_empty() {
		return false;
	}
	if leaves.len() == 1 {
		return tree.is_leaf(id) && id == leaves[0];
	}
	let Some(split) = tree.split(id) else {
		return false;
	};
	if split.axis() != Axis::Y || split.weights() != preset.root_weights {
		return false;
	}
	match preset.master_slot {
		Slot::A => {
			split.a() == leaves[0]
				&& tree.is_leaf(split.a())
				&& matches_equal_linear(tree, split.b(), &leaves[1..], Axis::X)
		}
		Slot::B => {
			split.b() == leaves[0]
				&& tree.is_leaf(split.b())
				&& matches_equal_linear(tree, split.a(), &leaves[1..], Axis::X)
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
		&& split.weights() == canonicalize_weights(1, (leaves.len() - 1) as u32)
		&& split.a() == leaves[0]
		&& tree.is_leaf(split.a())
		&& matches_equal_linear(tree, split.b(), &leaves[1..], axis)
}

fn new_internal_split<T>(tree: &mut Tree<T>, axis: Axis, a: NodeId, b: NodeId, weights: WeightPair) -> NodeId {
	let split_id = tree.new_split(axis, a, b, weights);
	tree.set_parent(a, Some(split_id));
	tree.set_parent(b, Some(split_id));
	split_id
}
