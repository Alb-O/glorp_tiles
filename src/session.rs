//! Focus-aware state machine layered on top of a validated [`Tree`].
//!
//! [`Session`] owns a tree plus focus, selection, and a monotonic revision counter. Operations
//! that rewrite structure or weights bump the revision, while targeting-only changes do not.
//! Geometry-driven commands such as navigation and resize require a fresh [`Snapshot`] whose
//! owner and revision match the current live session instance. Snapshots produced by the free
//! solver functions are ownerless and therefore usable for inspection but not for session-driven
//! geometry commands.
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
		error::{NavError, NeighborError, OpError, SolveError, ValidationError},
		geom::{Axis, Direction, Rect, Slot},
		ids::{NodeId, Revision, SessionOwner},
		limits::{LeafMeta, WeightPair, canonicalize_weights},
		nav::best_neighbor,
		preset::PresetKind,
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
/// use glorp_tiles::{Axis, LeafMeta, Rect, Session, Slot, SolverPolicy};
///
/// let mut session = Session::new();
/// session.insert_root("main", LeafMeta::default())?;
/// session.split_focus(Axis::X, Slot::B, "side", LeafMeta::default(), None)?;
///
/// let snapshot = session.solve(Rect { x: 0, y: 0, w: 120, h: 40 }, &SolverPolicy::default())?;
/// assert_eq!(snapshot.node_rects().len(), 3);
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Serialize, Deserialize)]
pub struct Session<T> {
	tree: Tree<T>,
	focus: Option<NodeId>,
	selection: Option<NodeId>,
	revision: Revision,
	#[serde(skip, default = "SessionOwner::fresh")]
	owner: SessionOwner,
}

impl<T> Clone for Session<T>
where
	T: Clone,
{
	fn clone(&self) -> Self {
		Self {
			tree: self.tree.clone(),
			focus: self.focus,
			selection: self.selection,
			revision: self.revision,
			owner: SessionOwner::fresh(),
		}
	}
}

impl<T> PartialEq for Session<T>
where
	T: PartialEq,
{
	fn eq(&self, other: &Self) -> bool {
		self.tree == other.tree
			&& self.focus == other.focus
			&& self.selection == other.selection
			&& self.revision == other.revision
	}
}

impl<T> Eq for Session<T> where T: Eq {}

impl<T> Default for Session<T> {
	fn default() -> Self {
		Self {
			tree: Tree::default(),
			focus: None,
			selection: None,
			revision: 0,
			owner: SessionOwner::fresh(),
		}
	}
}

impl<T> Session<T> {
	/// Creates an empty session.
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Creates a session from `tree` using a deterministic default targeting state.
	///
	/// Empty trees produce empty sessions. Non-empty trees focus and select the first leaf in
	/// depth-first `A`-before-`B` order. The resulting session starts at revision `0`, so any
	/// session-driven geometry work should solve fresh from the returned session rather than trying
	/// to reuse snapshots produced elsewhere.
	///
	/// ```
	/// use glorp_tiles::{Axis, LeafMeta, Session, Slot, Tree};
	///
	/// let mut tree = Tree::new();
	/// let main = tree.insert_root("main", LeafMeta::default())?;
	/// let _side = tree.split_leaf(main, Axis::X, Slot::B, "side", LeafMeta::default(), None)?;
	///
	/// let session = Session::from_tree(tree)?;
	/// assert_eq!(session.focus(), Some(main));
	/// assert_eq!(session.selection(), Some(main));
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	pub fn from_tree(tree: Tree<T>) -> Result<Self, ValidationError> {
		tree.validate()?;
		let (focus, selection) = match tree.root_id() {
			Some(root) => {
				// Reuse the crate's canonical leaf order so Tree -> Session bridging stays stable.
				let focus = tree
					.first_leaf(root)
					.expect("validated non-empty tree should contain a leaf");
				(Some(focus), Some(focus))
			}
			None => (None, None),
		};
		Ok(Self {
			tree,
			focus,
			selection,
			revision: 0,
			owner: SessionOwner::fresh(),
		})
	}

	/// Creates a session from `tree` with explicit focus and selection state.
	///
	/// Empty trees require `focus == None` and `selection == None`. Non-empty trees require
	/// `focus` to point at a leaf, and `selection` must be either that same leaf or a split that
	/// contains it. This is the lossless bridge for editors that persist both topology and targeting
	/// state across process boundaries, including the empty-session case. Partial state such as
	/// `Some(focus)` with `None` selection is rejected.
	///
	/// ```
	/// use glorp_tiles::{LeafMeta, Session, Tree};
	///
	/// let empty = Session::<&'static str>::from_tree_with_state(Tree::new(), None, None)?;
	/// assert_eq!(empty.focus(), None);
	/// assert_eq!(empty.selection(), None);
	///
	/// let mut tree = Tree::new();
	/// let leaf = tree.insert_root("main", LeafMeta::default())?;
	/// let session = Session::from_tree_with_state(tree, Some(leaf), Some(leaf))?;
	/// assert_eq!(session.focus(), Some(leaf));
	/// assert_eq!(session.selection(), Some(leaf));
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	pub fn from_tree_with_state(
		tree: Tree<T>, focus: Option<NodeId>, selection: Option<NodeId>,
	) -> Result<Self, ValidationError> {
		let session = Self {
			tree,
			focus,
			selection,
			revision: 0,
			owner: SessionOwner::fresh(),
		};
		session.validate()?;
		Ok(session)
	}

	/// Removes session targeting state and returns the underlying tree.
	///
	/// The returned tree preserves all topology, ids, payloads, metadata, and split weights, but
	/// drops focus, selection, revision, and live-session snapshot ownership.
	#[must_use]
	pub fn into_tree(self) -> Tree<T> {
		self.tree
	}

	/// Validates both the underlying tree and the session targeting invariants.
	pub fn validate(&self) -> Result<(), ValidationError> {
		self.tree.validate()?;
		self.validate_targeting()
	}

	fn validate_targeting(&self) -> Result<(), ValidationError> {
		if self.tree.root_id().is_none() {
			return if self.focus.is_none() && self.selection.is_none() {
				Ok(())
			} else {
				Err(ValidationError::EmptyStateInconsistent)
			};
		}

		let focus = self.focus.ok_or(ValidationError::EmptyStateInconsistent)?;
		if !self.tree.is_leaf(focus) {
			return Err(ValidationError::NonLeafFocus(focus));
		}
		let selection = self.selection.ok_or(ValidationError::EmptyStateInconsistent)?;
		if self.selection_contains_focus(selection, focus) {
			Ok(())
		} else {
			Err(ValidationError::InvalidSelection(selection))
		}
	}

	/// Solves the current tree into a snapshot tagged with the current revision and session owner.
	///
	/// Solving does not mutate the session. The returned snapshot can be reused for geometry-driven
	/// session commands until the session structure or split weights change.
	///
	/// ```text
	/// edit tree/weights
	///     |
	///     v
	/// revision += 1
	///     |
	///     v
	/// solve(root, policy) -> Snapshot { revision = N, owner = this session }
	///   -> focus_dir(..., snapshot N)   OK
	///   -> grow_focus(..., snapshot N)  OK
	/// ```
	pub fn solve(&self, root: Rect, policy: &SolverPolicy) -> Result<Snapshot, SolveError> {
		let mut snapshot = crate::solver::solve_with_revision(&self.tree, root, self.revision, policy)?;
		// Only live session solves stamp ownership; free solver entry points stay ownerless and
		// therefore cannot drive session-relative navigation or resize commands later.
		snapshot.set_owner(self.owner);
		Ok(snapshot)
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

	/// Replaces the payload stored in `leaf`.
	///
	/// Payload changes do not affect geometry and therefore do not bump the revision or stale
	/// existing snapshots.
	pub fn set_leaf_payload(&mut self, leaf: NodeId, payload: T) -> Result<(), OpError> {
		self.tree.set_leaf_payload(leaf, payload)
	}

	/// Replaces the sizing metadata stored in `leaf`.
	///
	/// Returns `true` when the metadata changed. Geometry-affecting metadata changes bump the
	/// revision and stale previously solved snapshots; unchanged metadata is a no-op that keeps the
	/// current revision.
	pub fn set_leaf_meta(&mut self, leaf: NodeId, meta: LeafMeta) -> Result<bool, OpError> {
		let changed = self.tree.set_leaf_meta(leaf, meta)?;
		if changed {
			self.bump_revision();
		}
		self.validate_targeting().map_err(OpError::Validation)?;
		Ok(changed)
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
		self.validate_targeting().map_err(OpError::Validation).inspect_err(|_| {
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
			if !self.selection_contains_focus(id, focus) {
				return Err(OpError::Validation(ValidationError::InvalidSelection(id)));
			}
			self.selection = Some(id);
		}
		self.validate_targeting().map_err(OpError::Validation).inspect_err(|_| {
			self.focus = old_focus;
			self.selection = old_selection;
		})
	}

	/// Inserts the first root leaf into an empty session.
	///
	/// The inserted leaf becomes both focus and selection.
	pub fn insert_root(&mut self, payload: T, meta: LeafMeta) -> Result<NodeId, OpError> {
		let id = self.tree.insert_root(payload, meta)?;
		self.focus = Some(id);
		self.selection = Some(id);
		self.bump_revision();
		self.validate_targeting().map_err(OpError::Validation)?;
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
		let old_selection = self.selection;
		let new_leaf = self.tree.split_leaf(focus, axis, slot, payload, meta, weights)?;
		let split_id = self
			.tree
			.parent_of(new_leaf)
			.expect("newly inserted split sibling should have a parent split");
		self.repair_after_mutation(focus, old_selection, Some(split_id));
		self.bump_revision();
		self.validate_targeting().map_err(OpError::Validation)?;
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
		let new_leaf = self.tree.wrap_node(selection, axis, slot, payload, meta, weights)?;
		let split_id = self
			.tree
			.parent_of(new_leaf)
			.expect("wrapped node should be attached under a new split");
		self.repair_after_mutation(focus, old_selection, Some(split_id));
		self.bump_revision();
		self.validate_targeting().map_err(OpError::Validation)?;
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
		let fallback = self.tree.remove_leaf(focus)?;
		self.repair_after_mutation(focus, old_selection, fallback);
		self.bump_revision();
		self.validate_targeting().map_err(OpError::Validation)?;
		Ok(())
	}

	/// Swaps two distinct, structurally disjoint nodes.
	///
	/// Same-node swaps and ancestor/descendant swaps are rejected. Focus is preserved when its leaf
	/// survives, and selection is repaired afterward.
	pub fn swap_nodes(&mut self, a: NodeId, b: NodeId) -> Result<(), OpError> {
		let focus = self.require_focus_leaf()?;
		let old_selection = self.selection;
		self.tree.swap_nodes(a, b)?;
		self.repair_after_mutation(focus, old_selection, self.tree.root_id());
		self.bump_revision();
		self.validate_targeting().map_err(OpError::Validation)?;
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
			.move_subtree_as_sibling_of(selection, target, axis, slot, None)?;
		self.repair_after_mutation(focus, old_selection, Some(split_id));
		self.bump_revision();
		self.validate_targeting().map_err(OpError::Validation)?;
		Ok(())
	}

	/// Moves focus to the best solved leaf neighbor in `dir`.
	///
	/// The supplied snapshot must belong to this live session and match the current revision. On
	/// success this updates focus, repairs selection to keep containing the new focus, and does not
	/// bump the revision.
	pub fn focus_dir(&mut self, dir: Direction, snap: &Snapshot) -> Result<(), NavError> {
		self.ensure_fresh_snapshot(snap)
			.map_err(|error| map_op_to_nav(error).expect("focus_dir should only map nav-compatible op errors"))?;
		// Revalidate the session-side invariant before delegating so the low-level helper can treat
		// missing/non-leaf focus cases as unreachable here.
		self.validate_targeting().map_err(NavError::Validation)?;
		let focus = self.focus.ok_or(NavError::Empty)?;
		let next = best_neighbor(&self.tree, snap, focus, dir)
			.map_err(map_neighbor_to_nav)?
			.ok_or(NavError::NoCandidate)?;
		self.focus = Some(next);
		self.repair_selection_for_current_focus();
		self.validate_targeting().map_err(NavError::Validation)
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
		self.validate_targeting().map_err(OpError::Validation)
	}

	/// Collapses the selection back to the current focus.
	///
	/// This is a targeting-only change and does not bump the revision.
	pub fn select_focus(&mut self) {
		self.selection = self.focus;
	}

	/// Attempts to grow the focused leaf outward in `dir`.
	///
	/// The supplied snapshot must belong to this live session and match the current revision.
	/// `amount == 0` is a no-op, and requests with no effective weight change are also no-ops.
	pub fn grow_focus(
		&mut self, dir: Direction, amount: u32, strategy: ResizeStrategy, snap: &Snapshot,
	) -> Result<(), OpError> {
		self.resize_focus(dir, amount, strategy, snap, true)
	}

	/// Attempts to shrink the focused leaf inward from `dir`.
	///
	/// The supplied snapshot must belong to this live session and match the current revision.
	/// `amount == 0` is a no-op, and requests with no effective weight change are also no-ops.
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
		self.tree.toggle_split_axis(selection)?;
		self.bump_revision();
		self.validate_targeting().map_err(OpError::Validation)
	}

	/// Mirrors the selected subtree across `axis`.
	///
	/// Leaf selections are a structural no-op and do not bump the revision.
	pub fn mirror_selection(&mut self, axis: Axis) -> Result<(), OpError> {
		let selection = self.selection.ok_or(OpError::Empty)?;
		if !self.tree.mirror_subtree(selection, axis)? {
			return Ok(());
		}
		self.bump_revision();
		self.validate_targeting().map_err(OpError::Validation)
	}

	/// Rebalances split weights within the selected subtree according to `mode`.
	///
	/// Selecting a leaf or an already-canonical subtree is a no-op and does not bump the revision.
	pub fn rebalance_selection(&mut self, mode: RebalanceMode) -> Result<(), OpError> {
		let selection = self.selection.ok_or(OpError::Empty)?;
		let changed = match mode {
			RebalanceMode::BinaryEqual => self.tree.rebalance_subtree_binary_equal(selection)?,
			RebalanceMode::LeafCount => self.tree.rebalance_subtree_leaf_count(selection)?,
		};
		if !changed {
			return Ok(());
		}
		self.bump_revision();
		self.validate_targeting().map_err(OpError::Validation)
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
		let Some(rebuilt) = self.tree.apply_preset(selection, preset)? else {
			return Ok(());
		};

		self.repair_after_mutation(focus, Some(rebuilt), Some(rebuilt));
		self.bump_revision();
		self.validate_targeting().map_err(OpError::Validation)
	}

	fn resize_focus(
		&mut self, dir: Direction, amount: u32, strategy: ResizeStrategy, snap: &Snapshot, outward: bool,
	) -> Result<(), OpError> {
		if amount == 0 {
			return Ok(());
		}
		self.ensure_fresh_snapshot(snap)?;
		let focus = self.require_focus_leaf()?;
		let mut summaries = HashMap::with_capacity(self.tree.node_count());
		if let Some(root) = self.tree.root_id() {
			summarize(&self.tree, root, &mut summaries).map_err(OpError::Validation)?;
		}
		let eligible = eligible_splits(&self.tree, focus, dir, snap, &summaries)?;
		if eligible.is_empty() {
			return Ok(());
		}
		let sign = resize_sign(dir, outward);
		let allocations = distribute_resize(amount, strategy, sign, &eligible);
		if allocations.is_empty() {
			return Ok(());
		}
		let mut changed = false;
		for (eligible_idx, delta) in allocations {
			if delta == 0 {
				continue;
			}
			let info = eligible
				.get(eligible_idx)
				.expect("eligible split index should come from distribute_resize");
			let new_a = if sign > 0 {
				info.current_a + delta
			} else {
				info.current_a - delta
			};
			let total = info.total;
			let weights = canonicalize_weights(new_a, total - new_a);
			changed |= self.tree.set_split_weights(info.split, weights)?;
		}
		if !changed {
			return Ok(());
		}
		self.bump_revision();
		self.validate_targeting().map_err(OpError::Validation)
	}

	fn repair_after_mutation(
		&mut self, old_focus: NodeId, old_selection: Option<NodeId>, replacement_site: Option<NodeId>,
	) {
		let root = self.tree.root_id();
		self.focus = if root.is_none() {
			None
		} else if self.tree.is_leaf(old_focus) {
			// Most mutations preserve the focused leaf id, so keep it when it still exists instead of
			// retargeting through tree order.
			Some(old_focus)
		} else {
			replacement_site.and_then(|id| self.tree.first_leaf(id))
		};

		// Preserve the caller's subtree selection only when it still contains the surviving focus;
		// otherwise collapse targeting back to that focus.
		self.selection = match (root, self.focus) {
			(None, _) | (_, None) => None,
			(Some(_), Some(focus)) => old_selection
				.filter(|selection| self.selection_contains_focus(*selection, focus))
				.or(Some(focus)),
		};
	}

	fn repair_selection_for_current_focus(&mut self) {
		let Some(focus) = self.focus else {
			self.selection = None;
			return;
		};
		self.selection = self
			.selection
			.filter(|selection| self.selection_contains_focus(*selection, focus))
			.filter(|selection| self.tree.is_split(*selection))
			.or(Some(focus));
	}

	fn selection_contains_focus(&self, selection: NodeId, focus: NodeId) -> bool {
		if self.tree.is_leaf(selection) {
			selection == focus
		} else {
			self.tree.contains_in_subtree(selection, focus)
		}
	}

	fn ensure_fresh_snapshot(&self, snap: &Snapshot) -> Result<(), OpError> {
		// Check ownership before revision so callers can distinguish "wrong live session" from
		// "right session, but stale".
		if snap.owner() != Some(self.owner) {
			return Err(OpError::ForeignSnapshot);
		}
		(snap.revision() == self.revision)
			.then_some(())
			.ok_or(OpError::StaleSnapshot)
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
		OpError::ForeignSnapshot => Some(NavError::ForeignSnapshot),
		OpError::StaleSnapshot => Some(NavError::StaleSnapshot),
		OpError::Validation(err) => Some(NavError::Validation(err)),
		OpError::MissingNode(id) | OpError::NotLeaf(id) | OpError::NotSplit(id) => {
			Some(NavError::MissingSnapshotRect(id))
		}
		OpError::NonEmpty
		| OpError::InvalidLeafMeta
		| OpError::InvalidWeights
		| OpError::NoParent(_)
		| OpError::SameNode
		| OpError::AncestorConflict
		| OpError::TargetInsideSelection => None,
	}
}

fn map_neighbor_to_nav(error: NeighborError) -> NavError {
	match error {
		NeighborError::Validation(err) => NavError::Validation(err),
		NeighborError::MissingSnapshotRect(id) => NavError::MissingSnapshotRect(id),
		NeighborError::MissingNode(id) => {
			unreachable!("focus_dir validated session targeting, but focused node {id} was missing")
		}
		NeighborError::NotLeaf(id) => {
			unreachable!("focus_dir validated session targeting, but focused node {id} was not a leaf")
		}
	}
}

#[cfg(test)]
mod tests {
	use {
		super::map_op_to_nav,
		crate::{NavError, NodeId, OpError, ValidationError},
	};

	#[test]
	fn nav_compatible_op_errors_map_exactly() {
		assert_eq!(map_op_to_nav(OpError::Empty), Some(NavError::Empty));
		assert_eq!(map_op_to_nav(OpError::ForeignSnapshot), Some(NavError::ForeignSnapshot));
		assert_eq!(map_op_to_nav(OpError::StaleSnapshot), Some(NavError::StaleSnapshot));
		assert_eq!(
			map_op_to_nav(OpError::Validation(ValidationError::Cycle(NodeId::from_raw(7)))),
			Some(NavError::Validation(ValidationError::Cycle(NodeId::from_raw(7))))
		);
		assert_eq!(
			map_op_to_nav(OpError::MissingNode(NodeId::from_raw(11))),
			Some(NavError::MissingSnapshotRect(NodeId::from_raw(11)))
		);
		assert_eq!(
			map_op_to_nav(OpError::NotLeaf(NodeId::from_raw(13))),
			Some(NavError::MissingSnapshotRect(NodeId::from_raw(13)))
		);
		assert_eq!(
			map_op_to_nav(OpError::NotSplit(NodeId::from_raw(17))),
			Some(NavError::MissingSnapshotRect(NodeId::from_raw(17)))
		);
	}

	#[test]
	fn non_nav_op_errors_do_not_synthesize_nav_behavior() {
		assert_eq!(map_op_to_nav(OpError::NonEmpty), None);
		assert_eq!(map_op_to_nav(OpError::InvalidWeights), None);
		assert_eq!(map_op_to_nav(OpError::NoParent(NodeId::from_raw(19))), None);
		assert_eq!(map_op_to_nav(OpError::SameNode), None);
		assert_eq!(map_op_to_nav(OpError::AncestorConflict), None);
		assert_eq!(map_op_to_nav(OpError::TargetInsideSelection), None);
	}
}
