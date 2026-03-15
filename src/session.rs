//! Focus-aware state machine layered on top of a validated [`Tree`].
//!
//! [`Session`] owns a tree plus focus, selection, and a monotonic revision counter. Operations
//! that rewrite structure or weights bump the revision, while targeting-only changes do not.
//! Geometry-driven commands such as navigation and resize require a fresh [`Snapshot`] whose
//! revision matches the current session revision.
//!
//! ```text
//!             [root]
//!             /    \
//!         [left]  [right]
//!          /  \
//!       [a]  [b]
//!
//! focus = b
//! valid selection = b | left | root
//! invalid selection = a | any node not containing b
//! ```

use {
	crate::{
		error::{NavError, OpError, SolveError, ValidationError},
		geom::{Axis, Direction, Rect, Slot},
		ids::{NodeId, Revision},
		limits::{LeafMeta, WeightPair, canonicalize_weights},
		nav::best_neighbor,
		preset::{PresetKind, apply_preset_subtree},
		resize::{ResizeStrategy, distribute_resize, eligible_splits, resize_sign},
		snapshot::Snapshot,
		solver::{SolverPolicy, summarize},
		tree::Tree,
	},
	serde::{Deserialize, Serialize},
	std::collections::HashMap,
};

/// Rebalancing policy for the currently selected subtree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebalanceMode {
	/// Reset every split in the selected subtree to equal `1:1` weights.
	BinaryEqual,
	/// Reset every split in the selected subtree using descendant leaf counts.
	LeafCount,
}

/// Mutable editing session over a validated tiling tree.
///
/// Session invariants:
///
/// - an empty tree has `focus == None` and `selection == None`
/// - a non-empty session always has a focused leaf
/// - selection is either that focused leaf or a split that contains it
///
/// ```
/// use libtiler::{Axis, LeafMeta, Rect, Session, Slot, SolverPolicy};
///
/// let mut session = Session::new();
/// session.insert_root("main", LeafMeta::default())?;
/// session.split_focus(Axis::X, Slot::B, "side", LeafMeta::default(), None)?;
///
/// let snapshot = session.solve(Rect { x: 0, y: 0, w: 120, h: 40 }, &SolverPolicy::default());
/// assert_eq!(snapshot.node_rects.len(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session<T> {
	tree: Tree<T>,
	focus: Option<NodeId>,
	selection: Option<NodeId>,
	revision: Revision,
}

impl<T> Default for Session<T> {
	fn default() -> Self {
		Self {
			tree: Tree::default(),
			focus: None,
			selection: None,
			revision: 0,
		}
	}
}

impl<T> Session<T> {
	/// Creates an empty session.
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Validates both the underlying tree and the session targeting invariants.
	pub fn validate(&self) -> Result<(), ValidationError> {
		self.tree.validate()?;
		match self.tree.root_id() {
			None => {
				if self.focus.is_none() && self.selection.is_none() {
					Ok(())
				} else {
					Err(ValidationError::EmptyStateInconsistent)
				}
			}
			Some(_) => {
				let focus = self.focus.ok_or(ValidationError::EmptyStateInconsistent)?;
				if !self.tree.is_leaf(focus) {
					return Err(ValidationError::NonLeafFocus(focus));
				}
				let selection = self.selection.ok_or(ValidationError::EmptyStateInconsistent)?;
				if !self.tree.contains(selection) {
					return Err(ValidationError::InvalidSelection(selection));
				}
				if self.tree.is_leaf(selection) {
					if selection == focus {
						Ok(())
					} else {
						Err(ValidationError::InvalidSelection(selection))
					}
				} else if self.tree.contains_in_subtree(selection, focus) {
					Ok(())
				} else {
					Err(ValidationError::InvalidSelection(selection))
				}
			}
		}
	}

	/// Solves the current tree into a snapshot tagged with the current revision.
	///
	/// This is the infallible convenience wrapper over [`Self::try_solve`]. It does not mutate the
	/// session.
	///
	/// ```text
	/// edit tree/weights
	///     |
	///     v
	/// revision += 1
	///     |
	///     v
	/// solve(root, policy) -> Snapshot { revision = N }
	///   -> focus_dir(..., snapshot N)   OK
	///   -> grow_focus(..., snapshot N)  OK
	/// ```
	///
	/// # Panics
	///
	/// Panics if session invariants have been broken badly enough that solving unexpectedly fails.
	#[must_use]
	pub fn solve(&self, root: Rect, policy: &SolverPolicy) -> Snapshot {
		self.try_solve(root, policy)
			.expect("session should maintain a valid and representable tree")
	}

	/// Solves the current tree into a snapshot tagged with the current revision.
	///
	/// Unlike [`Self::solve`], this returns [`SolveError`] instead of panicking when validation or
	/// solving fails. Solving never mutates the session.
	pub fn try_solve(&self, root: Rect, policy: &SolverPolicy) -> Result<Snapshot, SolveError> {
		crate::solver::solve_with_revision(&self.tree, root, self.revision, policy)
	}

	/// Returns the underlying tree.
	#[must_use]
	pub fn tree(&self) -> &Tree<T> {
		&self.tree
	}

	/// Returns the currently focused leaf id, if any.
	///
	/// In a valid non-empty session this is always `Some(leaf_id)`.
	#[must_use]
	pub fn focus(&self) -> Option<NodeId> {
		self.focus
	}

	/// Returns the current selection.
	///
	/// In a valid non-empty session this is either the focused leaf or a split containing it.
	#[must_use]
	pub fn selection(&self) -> Option<NodeId> {
		self.selection
	}

	/// Returns the current session revision.
	#[must_use]
	pub fn revision(&self) -> Revision {
		self.revision
	}

	/// Moves focus to an existing leaf and repairs selection to keep containing it.
	///
	/// This does not bump the revision.
	pub fn set_focus_leaf(&mut self, id: NodeId) -> Result<(), OpError> {
		let old_focus = self.focus;
		let old_selection = self.selection;
		self.require_leaf(id)?;
		self.focus = Some(id);
		self.repair_selection_for_current_focus();
		self.validate().map_err(OpError::Validation).inspect_err(|_| {
			self.focus = old_focus;
			self.selection = old_selection;
		})
	}

	/// Sets the current selection.
	///
	/// Selecting a leaf also moves focus to that leaf. Selecting a split requires the current
	/// focused leaf to already lie inside that split. This does not bump the revision.
	pub fn set_selection(&mut self, id: NodeId) -> Result<(), OpError> {
		let old_focus = self.focus;
		let old_selection = self.selection;
		self.require_node(id)?;
		if self.tree.is_leaf(id) {
			self.focus = Some(id);
			self.selection = Some(id);
		} else {
			let focus = self.require_focus_leaf()?;
			if !self.tree.contains_in_subtree(id, focus) {
				return Err(OpError::Validation(ValidationError::InvalidSelection(id)));
			}
			self.selection = Some(id);
		}
		self.validate().map_err(OpError::Validation).inspect_err(|_| {
			self.focus = old_focus;
			self.selection = old_selection;
		})
	}

	/// Inserts the first root leaf into an empty session.
	///
	/// The inserted leaf becomes both focus and selection.
	pub fn insert_root(&mut self, payload: T, meta: LeafMeta) -> Result<NodeId, OpError> {
		if self.tree.root_id().is_some() {
			return Err(OpError::NonEmpty);
		}
		let id = self.tree.new_leaf(payload, meta);
		self.tree.set_root(Some(id));
		self.focus = Some(id);
		self.selection = Some(id);
		self.bump_revision();
		self.validate().map_err(OpError::Validation)?;
		Ok(id)
	}

	/// Splits the focused leaf by inserting a new sibling.
	///
	/// Returns the new leaf id. The existing focused leaf remains focused when it survives, and
	/// selection is repaired to continue containing the focused leaf. Invalid weight pairs such as
	/// `(0, 0)` are rejected.
	pub fn split_focus(
		&mut self, axis: Axis, slot: Slot, payload: T, meta: LeafMeta, weights: Option<WeightPair>,
	) -> Result<NodeId, OpError> {
		let focus = self.require_focus_leaf()?;
		let weights = weights.unwrap_or_default().checked().ok_or(OpError::InvalidWeights)?;
		let old_selection = self.selection;
		let new_leaf = self.tree.new_leaf(payload, meta);
		let split_id = self.tree.attach_as_sibling(focus, new_leaf, axis, slot, weights);
		self.repair_after_mutation(focus, old_selection, Some(split_id));
		self.bump_revision();
		self.validate().map_err(OpError::Validation)?;
		Ok(new_leaf)
	}

	/// Wraps the current selection together with a newly inserted sibling leaf.
	///
	/// The focused leaf is preserved when it survives, and selection is repaired to continue
	/// containing that focus. Invalid weight pairs such as `(0, 0)` are rejected.
	pub fn wrap_selection(
		&mut self, axis: Axis, slot: Slot, payload: T, meta: LeafMeta, weights: Option<WeightPair>,
	) -> Result<NodeId, OpError> {
		let selection = self.selection.ok_or(OpError::Empty)?;
		let focus = self.require_focus_leaf()?;
		let old_selection = self.selection;
		let new_leaf = self.tree.new_leaf(payload, meta);
		let split_id = self.tree.attach_as_sibling(
			selection,
			new_leaf,
			axis,
			slot,
			weights.unwrap_or_default().checked().ok_or(OpError::InvalidWeights)?,
		);
		self.repair_after_mutation(focus, old_selection, Some(split_id));
		self.bump_revision();
		self.validate().map_err(OpError::Validation)?;
		Ok(new_leaf)
	}

	/// Removes the focused leaf and collapses any unary parent created by that removal.
	///
	/// Removing the last remaining root leaf empties the session. Otherwise focus falls back
	/// deterministically to the first leaf at the replacement site, and selection is repaired to
	/// continue containing that focus when possible.
	pub fn remove_focus(&mut self) -> Result<(), OpError> {
		let focus = self.require_focus_leaf()?;
		let old_selection = self.selection;
		let fallback = self
			.tree
			.remove_leaf_and_collapse(focus)
			.ok_or(OpError::NotLeaf(focus))?;
		self.repair_after_mutation(focus, old_selection, fallback);
		self.bump_revision();
		self.validate().map_err(OpError::Validation)?;
		Ok(())
	}

	/// Swaps two distinct, structurally disjoint nodes.
	///
	/// Same-node swaps and ancestor/descendant swaps are rejected. Focus is preserved when its leaf
	/// survives, and selection is repaired afterward.
	pub fn swap_nodes(&mut self, a: NodeId, b: NodeId) -> Result<(), OpError> {
		if a == b {
			return Err(OpError::SameNode);
		}
		self.require_node(a)?;
		self.require_node(b)?;
		if self.tree.contains_in_subtree(a, b) || self.tree.contains_in_subtree(b, a) {
			return Err(OpError::AncestorConflict);
		}
		let focus = self.require_focus_leaf()?;
		let old_selection = self.selection;
		self.tree
			.swap_disjoint_nodes(a, b)
			.expect("validated disjoint swap should succeed");
		self.repair_after_mutation(focus, old_selection, self.tree.root_id());
		self.bump_revision();
		self.validate().map_err(OpError::Validation)?;
		Ok(())
	}

	/// Moves the selected subtree so it becomes a sibling of `target`.
	///
	/// The selected subtree may be moved next to an ancestor target; the session first computes the
	/// target that remains after detaching the selection. Moving onto the same node or inside the
	/// selected subtree is rejected. Focus is preserved when its leaf survives, and selection is
	/// repaired to continue referring to the moved subtree when possible.
	pub fn move_selection_as_sibling_of(&mut self, target: NodeId, axis: Axis, slot: Slot) -> Result<(), OpError> {
		let selection = self.selection.ok_or(OpError::Empty)?;
		let focus = self.require_focus_leaf()?;
		let old_selection = self.selection;
		let split_id = self
			.tree
			.move_subtree_as_sibling_of(selection, target, axis, slot, WeightPair::default())?;
		self.repair_after_mutation(focus, old_selection, Some(split_id));
		self.bump_revision();
		self.validate().map_err(OpError::Validation)?;
		Ok(())
	}

	/// Moves focus to the best solved leaf neighbor in `dir`.
	///
	/// The supplied snapshot must be fresh for the current revision. On success this updates focus,
	/// repairs selection to keep containing the new focus, and does not bump the revision.
	pub fn focus_dir(&mut self, dir: Direction, snap: &Snapshot) -> Result<(), NavError> {
		self.ensure_fresh_snapshot(snap)
			.map_err(|error| map_op_to_nav(error).expect("focus_dir should only map nav-compatible op errors"))?;
		let focus = self.focus.ok_or(NavError::Empty)?;
		let leaf_rects = self.leaf_rects_from_snapshot(snap)?;
		let next = best_neighbor(&self.tree, &leaf_rects, focus, dir).ok_or(NavError::NoCandidate)?;
		self.focus = Some(next);
		self.repair_selection_for_current_focus();
		self.validate().map_err(NavError::Validation)
	}

	/// Promotes the current selection to its parent split.
	///
	/// If a selection already exists it is used as the base; otherwise the focused leaf is used.
	/// Selecting the root has no parent and returns [`OpError::NoParent`]. This does not bump the
	/// revision.
	pub fn select_parent(&mut self) -> Result<(), OpError> {
		let base = self.selection.or(self.focus).ok_or(OpError::Empty)?;
		let parent = self.tree.parent_of(base).ok_or(OpError::NoParent(base))?;
		self.selection = Some(parent);
		self.validate().map_err(OpError::Validation)
	}

	/// Collapses the selection back to the current focus.
	///
	/// This is a targeting-only change and does not bump the revision.
	pub fn select_focus(&mut self) {
		self.selection = self.focus;
	}

	/// Attempts to grow the focused leaf outward in `dir`.
	///
	/// The supplied snapshot must be fresh for the current revision. `amount == 0` is a no-op, and
	/// requests with no eligible split are also no-ops. Successful calls with at least one eligible
	/// split bump the revision after attempting the resize, even if slack clamped every resulting
	/// delta to zero.
	pub fn grow_focus(
		&mut self, dir: Direction, amount: u32, strategy: ResizeStrategy, snap: &Snapshot,
	) -> Result<(), OpError> {
		self.resize_focus(dir, amount, strategy, snap, true)
	}

	/// Attempts to shrink the focused leaf inward from `dir`.
	///
	/// The supplied snapshot must be fresh for the current revision. `amount == 0` is a no-op, and
	/// requests with no eligible split are also no-ops. Successful calls with at least one eligible
	/// split bump the revision after attempting the resize, even if slack clamped every resulting
	/// delta to zero.
	pub fn shrink_focus(
		&mut self, dir: Direction, amount: u32, strategy: ResizeStrategy, snap: &Snapshot,
	) -> Result<(), OpError> {
		self.resize_focus(dir, amount, strategy, snap, false)
	}

	/// Toggles the axis of the currently selected split.
	///
	/// Selecting a leaf returns [`OpError::NotSplit`].
	pub fn toggle_axis(&mut self) -> Result<(), OpError> {
		let selection = self.selection.ok_or(OpError::Empty)?;
		self.tree
			.toggle_split_axis(selection)
			.ok_or(OpError::NotSplit(selection))?;
		self.bump_revision();
		self.validate().map_err(OpError::Validation)
	}

	/// Mirrors the selected subtree across `axis`.
	///
	/// Leaf selections are a structural no-op, but the operation still succeeds and bumps the
	/// revision.
	pub fn mirror_selection(&mut self, axis: Axis) -> Result<(), OpError> {
		let selection = self.selection.ok_or(OpError::Empty)?;
		self.tree.mirror_subtree_axis(selection, axis);
		self.bump_revision();
		self.validate().map_err(OpError::Validation)
	}

	/// Rebalances split weights within the selected subtree according to `mode`.
	///
	/// Selecting a leaf currently behaves as a no-op in the underlying tree implementation, but
	/// the operation still succeeds and bumps the revision.
	pub fn rebalance_selection(&mut self, mode: RebalanceMode) -> Result<(), OpError> {
		let selection = self.selection.ok_or(OpError::Empty)?;
		match mode {
			RebalanceMode::BinaryEqual => self
				.tree
				.rebalance_subtree_binary_equal(selection)
				.ok_or(OpError::NotSplit(selection))?,
			RebalanceMode::LeafCount => {
				self.tree
					.rebalance_subtree_leaf_count(selection)
					.ok_or(OpError::NotSplit(selection))?;
			}
		}
		self.bump_revision();
		self.validate().map_err(OpError::Validation)
	}

	/// Rebuilds the selected subtree to match `preset`.
	///
	/// Leaf selections and selections that already match `preset` are no-ops and do not bump the
	/// revision. Successful preset application preserves existing leaf ids, payloads, and metadata,
	/// rebuilds split structure as needed, preserves the focused leaf when it survives, and
	/// retargets selection to the rebuilt subtree root.
	pub fn apply_preset(&mut self, preset: PresetKind) -> Result<(), OpError> {
		let selection = self.selection.ok_or(OpError::Empty)?;
		let focus = self.require_focus_leaf()?;
		let Some(rebuilt) = apply_preset_subtree(&mut self.tree, selection, preset)? else {
			return Ok(());
		};

		self.repair_after_mutation(focus, Some(rebuilt), Some(rebuilt));
		self.bump_revision();
		self.validate().map_err(OpError::Validation)
	}

	fn resize_focus(
		&mut self, dir: Direction, amount: u32, strategy: ResizeStrategy, snap: &Snapshot, outward: bool,
	) -> Result<(), OpError> {
		if amount == 0 {
			return Ok(());
		}
		self.ensure_fresh_snapshot(snap)?;
		let focus = self.require_focus_leaf()?;
		let mut summaries = HashMap::new();
		if let Some(root) = self.tree.root_id() {
			summarize(&self.tree, root, &mut summaries).map_err(OpError::Validation)?;
		}
		let eligible = eligible_splits(&self.tree, focus, dir, snap, &summaries)?;
		if eligible.is_empty() {
			return Ok(());
		}
		let sign = resize_sign(dir, outward);
		let allocations = distribute_resize(amount, strategy, sign, &eligible);
		for (split_id, delta) in allocations {
			if delta == 0 {
				continue;
			}
			let info = eligible
				.iter()
				.find(|entry| entry.split == split_id)
				.expect("eligible split missing during resize");
			let new_a = if sign > 0 {
				info.current_a + delta
			} else {
				info.current_a - delta
			};
			let total = info.total;
			let weights = canonicalize_weights(new_a, total - new_a);
			self.tree
				.set_split_weights(split_id, weights)
				.ok_or(OpError::NotSplit(split_id))?;
		}
		self.bump_revision();
		self.validate().map_err(OpError::Validation)
	}

	fn leaf_rects_from_snapshot(&self, snap: &Snapshot) -> Result<HashMap<NodeId, Rect>, NavError> {
		self.tree
			.root_id()
			.map(|root| self.tree.leaf_ids_dfs(root))
			.unwrap_or_default()
			.into_iter()
			.map(|id| {
				snap.rect(id)
					.map(|rect| (id, rect))
					.ok_or(NavError::MissingSnapshotRect(id))
			})
			.collect()
	}

	fn repair_after_mutation(
		&mut self, old_focus: NodeId, old_selection: Option<NodeId>, replacement_site: Option<NodeId>,
	) {
		self.focus = if self.tree.root_id().is_none() {
			None
		} else if self.tree.is_leaf(old_focus) {
			Some(old_focus)
		} else {
			replacement_site.and_then(|id| self.tree.first_leaf(id))
		};

		self.selection = match (self.tree.root_id(), self.focus) {
			(None, _) | (_, None) => None,
			(Some(_), Some(focus)) => old_selection
				.filter(|selection| self.tree.contains(*selection))
				.filter(|selection| {
					if self.tree.is_leaf(*selection) {
						*selection == focus
					} else {
						self.tree.contains_in_subtree(*selection, focus)
					}
				})
				.or(Some(focus)),
		};
	}

	fn repair_selection_for_current_focus(&mut self) {
		self.selection = match (self.selection, self.focus) {
			(_, None) => None,
			(Some(selection), Some(focus)) if self.tree.contains(selection) => {
				if self.tree.is_leaf(selection) {
					Some(focus)
				} else if self.tree.contains_in_subtree(selection, focus) {
					Some(selection)
				} else {
					Some(focus)
				}
			}
			(None, Some(focus)) => Some(focus),
			(Some(_), Some(focus)) => Some(focus),
		};
	}

	fn ensure_fresh_snapshot(&self, snap: &Snapshot) -> Result<(), OpError> {
		if snap.revision == self.revision {
			Ok(())
		} else {
			Err(OpError::StaleSnapshot)
		}
	}

	fn require_focus_leaf(&self) -> Result<NodeId, OpError> {
		let focus = self.focus.ok_or(OpError::Empty)?;
		if self.tree.is_leaf(focus) {
			Ok(focus)
		} else {
			Err(OpError::NotLeaf(focus))
		}
	}

	fn require_node(&self, id: NodeId) -> Result<(), OpError> {
		if self.tree.contains(id) {
			Ok(())
		} else {
			Err(OpError::MissingNode(id))
		}
	}

	fn require_leaf(&self, id: NodeId) -> Result<(), OpError> {
		self.require_node(id)?;
		if self.tree.is_leaf(id) {
			Ok(())
		} else {
			Err(OpError::NotLeaf(id))
		}
	}

	fn bump_revision(&mut self) {
		self.revision += 1;
	}
}

fn map_op_to_nav(error: OpError) -> Option<NavError> {
	match error {
		OpError::Empty => Some(NavError::Empty),
		OpError::StaleSnapshot => Some(NavError::StaleSnapshot),
		OpError::Validation(err) => Some(NavError::Validation(err)),
		OpError::MissingNode(id) | OpError::NotLeaf(id) | OpError::NotSplit(id) => {
			Some(NavError::MissingSnapshotRect(id))
		}
		OpError::NonEmpty
		| OpError::InvalidWeights
		| OpError::NoParent(_)
		| OpError::SameNode
		| OpError::AncestorConflict
		| OpError::TargetInsideSelection => None,
	}
}

#[cfg(test)]
mod tests {
	use {
		super::map_op_to_nav,
		crate::{NavError, OpError, ValidationError},
	};

	#[test]
	fn nav_compatible_op_errors_map_exactly() {
		assert_eq!(map_op_to_nav(OpError::Empty), Some(NavError::Empty));
		assert_eq!(map_op_to_nav(OpError::StaleSnapshot), Some(NavError::StaleSnapshot));
		assert_eq!(
			map_op_to_nav(OpError::Validation(ValidationError::Cycle(7))),
			Some(NavError::Validation(ValidationError::Cycle(7)))
		);
		assert_eq!(
			map_op_to_nav(OpError::MissingNode(11)),
			Some(NavError::MissingSnapshotRect(11))
		);
		assert_eq!(
			map_op_to_nav(OpError::NotLeaf(13)),
			Some(NavError::MissingSnapshotRect(13))
		);
		assert_eq!(
			map_op_to_nav(OpError::NotSplit(17)),
			Some(NavError::MissingSnapshotRect(17))
		);
	}

	#[test]
	fn non_nav_op_errors_do_not_synthesize_nav_behavior() {
		assert_eq!(map_op_to_nav(OpError::NonEmpty), None);
		assert_eq!(map_op_to_nav(OpError::InvalidWeights), None);
		assert_eq!(map_op_to_nav(OpError::NoParent(19)), None);
		assert_eq!(map_op_to_nav(OpError::SameNode), None);
		assert_eq!(map_op_to_nav(OpError::AncestorConflict), None);
		assert_eq!(map_op_to_nav(OpError::TargetInsideSelection), None);
	}
}
