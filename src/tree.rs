//! Validated binary split topology with semantic node accessors.
//!
//! [`Tree`] stores a single-root binary split tree whose public API exposes read-only semantic
//! views such as [`LeafNode`] and [`SplitNode`] rather than the internal node enum or storage map.
//! A valid tree has either no nodes and no root, or one root whose descendants form an acyclic
//! connected structure with consistent parent pointers.
//!
//! ```text
//! root = 7
//!
//!           [7 split X]
//!            /       \
//!       A-> [3]     [6 leaf]
//!           / \
//!      A-> [1] [2] <-B
//!         leaf leaf
//!
//! - every non-root node has exactly one parent
//! - every split has distinct A/B children
//! - leaf ids are stable while the leaf survives
//! - split ids are stable until that split is removed or rebuilt
//! ```

use {
	crate::{
		error::{OpError, ValidationError},
		geom::{Axis, Slot},
		ids::NodeId,
		limits::{LeafMeta, WeightPair, canonicalize_weights, leaf_meta_is_valid},
		preset::{PresetKind, apply_preset_subtree},
	},
	serde::{
		Deserialize, Serialize,
		de::{self, Deserializer},
	},
	std::{
		collections::{HashMap, HashSet},
		ops::ControlFlow,
	},
};

/// Read-only view of a leaf node inside a [`Tree`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeafNode<T> {
	parent: Option<NodeId>,
	payload: T,
	meta: LeafMeta,
}

impl<T> LeafNode<T> {
	/// Returns the parent split id, or `None` when this leaf is the root.
	#[must_use]
	pub fn parent(&self) -> Option<NodeId> {
		self.parent
	}

	/// Returns the leaf payload stored in the tree.
	#[must_use]
	pub fn payload(&self) -> &T {
		&self.payload
	}

	/// Returns the sizing metadata used by solving and validation.
	#[must_use]
	pub fn meta(&self) -> &LeafMeta {
		&self.meta
	}
}

/// Read-only view of an internal split node inside a [`Tree`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitNode {
	parent: Option<NodeId>,
	axis: Axis,
	a: NodeId,
	b: NodeId,
	weights: WeightPair,
}

impl SplitNode {
	/// Returns the parent split id, or `None` when this split is the root.
	#[must_use]
	pub fn parent(&self) -> Option<NodeId> {
		self.parent
	}

	/// Returns the axis along which this split divides its extent.
	#[must_use]
	pub fn axis(&self) -> Axis {
		self.axis
	}

	/// Returns the child id stored in slot `A`.
	#[must_use]
	pub fn a(&self) -> NodeId {
		self.a
	}

	/// Returns the child id stored in slot `B`.
	#[must_use]
	pub fn b(&self) -> NodeId {
		self.b
	}

	/// Returns the relative weight preference between child `A` and child `B`.
	#[must_use]
	pub fn weights(&self) -> WeightPair {
		self.weights
	}

	pub(crate) fn set_axis(&mut self, axis: Axis) {
		self.axis = axis;
	}

	pub(crate) fn set_weights(&mut self, weights: WeightPair) {
		self.weights = weights;
	}

	pub(crate) fn swap_children(&mut self) {
		std::mem::swap(&mut self.a, &mut self.b);
	}

	pub(crate) fn swap_weights(&mut self) {
		std::mem::swap(&mut self.weights.a, &mut self.weights.b);
	}
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Node<T> {
	Leaf(LeafNode<T>),
	Split(SplitNode),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RemoveLeafResult {
	Emptied,
	Replaced(NodeId),
}

impl RemoveLeafResult {
	pub(crate) fn replacement_site(self) -> Option<NodeId> {
		match self {
			Self::Emptied => None,
			Self::Replaced(node) => Some(node),
		}
	}
}

impl<T> Node<T> {
	#[must_use]
	fn parent(&self) -> Option<NodeId> {
		match self {
			Self::Leaf(leaf) => leaf.parent,
			Self::Split(split) => split.parent,
		}
	}

	fn parent_mut(&mut self) -> &mut Option<NodeId> {
		match self {
			Self::Leaf(leaf) => &mut leaf.parent,
			Self::Split(split) => &mut split.parent,
		}
	}

	#[must_use]
	fn as_split(&self) -> Option<&SplitNode> {
		match self {
			Self::Split(split) => Some(split),
			Self::Leaf(_) => None,
		}
	}

	#[must_use]
	fn as_split_mut(&mut self) -> Option<&mut SplitNode> {
		match self {
			Self::Split(split) => Some(split),
			Self::Leaf(_) => None,
		}
	}

	#[must_use]
	fn as_leaf(&self) -> Option<&LeafNode<T>> {
		match self {
			Self::Leaf(leaf) => Some(leaf),
			Self::Split(_) => None,
		}
	}
}

/// Validated single-root binary split topology.
///
/// `Tree` is the crate's direct-editing layer: it exposes structural mutation, leaf payload and
/// metadata updates, stable node ids, and free solving without focus or selection state. Use
/// [`crate::Session`] when you also need editor-style targeting state or geometry-driven commands.
///
/// Node ids are allocated monotonically within a tree. Leaves keep their ids while they survive;
/// split ids keep their ids until that split is removed or a subtree rebuild replaces it.
/// Deserialization validates the entire structure before returning.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Tree<T> {
	root: Option<NodeId>,
	nodes: HashMap<NodeId, Node<T>>,
	next_id_raw: u64,
}

#[derive(Deserialize)]
struct TreeWire<T> {
	root: Option<NodeId>,
	nodes: HashMap<NodeId, Node<T>>,
	next_id_raw: u64,
}

impl<'de, T> Deserialize<'de> for Tree<T>
where
	T: Deserialize<'de>,
{
	fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
	where
		D: Deserializer<'de>, {
		let wire = TreeWire::<T>::deserialize(deserializer)?;
		let tree = Self {
			root: wire.root,
			nodes: wire.nodes,
			next_id_raw: wire.next_id_raw,
		};
		tree.validate().map_err(de::Error::custom)?;
		Ok(tree)
	}
}

impl<T> Default for Tree<T> {
	fn default() -> Self {
		Self {
			root: None,
			nodes: HashMap::new(),
			next_id_raw: 1,
		}
	}
}

impl<T> Tree<T> {
	/// Creates an empty tree with no root node.
	#[must_use]
	pub fn new() -> Self {
		Self::default()
	}

	/// Returns the current root id, or `None` when the tree is empty.
	#[must_use]
	pub fn root_id(&self) -> Option<NodeId> {
		self.root
	}

	/// Returns a read-only split view for `id`.
	#[must_use]
	pub fn split(&self, id: NodeId) -> Option<&SplitNode> {
		self.nodes.get(&id).and_then(Node::as_split)
	}

	/// Returns a read-only leaf view for `id`.
	#[must_use]
	pub fn leaf(&self, id: NodeId) -> Option<&LeafNode<T>> {
		self.nodes.get(&id).and_then(Node::as_leaf)
	}

	/// Returns all node ids sorted by numeric id.
	///
	/// This is allocation order, not traversal order.
	#[must_use]
	pub fn node_ids(&self) -> Vec<NodeId> {
		let mut ids = self.nodes.keys().copied().collect::<Vec<_>>();
		ids.sort_unstable();
		ids
	}

	/// Returns the total number of nodes stored in the tree.
	#[must_use]
	pub fn node_count(&self) -> usize {
		self.nodes.len()
	}

	/// Returns all split ids sorted by numeric id.
	///
	/// This is allocation order, not topology order.
	#[must_use]
	pub fn split_ids(&self) -> Vec<NodeId> {
		let mut ids = self
			.nodes
			.iter()
			.filter_map(|(id, node)| matches!(node, Node::Split(_)).then_some(*id))
			.collect::<Vec<_>>();
		ids.sort_unstable();
		ids
	}

	pub(crate) fn geometry_fingerprint_validated(&self) -> (u64, u64) {
		let mut fingerprinter = Fingerprinter::new();
		fingerprinter.write_tag(b"glorp_tiles.tree.geometry.v1");
		fingerprinter.write_option_node_id(self.root);
		for id in self.node_ids() {
			fingerprinter.write_node_id(id);
			match self
				.nodes
				.get(&id)
				.expect("validated tree missing node during fingerprint")
			{
				Node::Leaf(leaf) => {
					// Payload is intentionally excluded so editor-side content changes do not invalidate
					// geometry caches or low-level navigation snapshots.
					fingerprinter.write_u8(0);
					fingerprinter.write_option_node_id(leaf.parent);
					fingerprinter.write_leaf_meta(&leaf.meta);
				}
				Node::Split(split) => {
					fingerprinter.write_u8(1);
					fingerprinter.write_option_node_id(split.parent);
					fingerprinter.write_axis(split.axis);
					fingerprinter.write_node_id(split.a);
					fingerprinter.write_node_id(split.b);
					fingerprinter.write_weight_pair(split.weights);
				}
			}
		}
		fingerprinter.finish()
	}

	pub(crate) fn set_root(&mut self, root: Option<NodeId>) {
		self.root = root;
	}

	fn split_mut(&mut self, id: NodeId) -> Option<&mut SplitNode> {
		self.nodes.get_mut(&id).and_then(Node::as_split_mut)
	}

	fn split_mut_or_error(&mut self, id: NodeId) -> Result<&mut SplitNode, OpError> {
		match self.nodes.get_mut(&id) {
			Some(Node::Split(split)) => Ok(split),
			Some(Node::Leaf(_)) => Err(OpError::NotSplit(id)),
			None => Err(OpError::MissingNode(id)),
		}
	}

	fn leaf_mut_or_error(&mut self, id: NodeId) -> Result<&mut LeafNode<T>, OpError> {
		match self.nodes.get_mut(&id) {
			Some(Node::Leaf(leaf)) => Ok(leaf),
			Some(Node::Split(_)) => Err(OpError::NotLeaf(id)),
			None => Err(OpError::MissingNode(id)),
		}
	}

	pub(crate) fn remove_node(&mut self, id: NodeId) -> Option<Node<T>> {
		self.nodes.remove(&id)
	}

	/// Validates structural and metadata invariants.
	///
	/// A valid tree satisfies:
	///
	/// - an empty tree has no root
	/// - a non-empty tree root exists and has no parent
	/// - every non-root node has exactly one parent
	/// - every split has two distinct children
	/// - split weights are not `(0, 0)`
	/// - leaf limits and priorities are internally consistent
	/// - the graph is acyclic and has no unreachable stored nodes
	pub fn validate(&self) -> Result<(), ValidationError> {
		match self.root {
			None => {
				if self.nodes.is_empty() {
					Ok(())
				} else {
					let extra = self
						.nodes
						.keys()
						.copied()
						.min()
						.expect("non-empty node map should have a minimum id");
					Err(ValidationError::Unreachable(extra))
				}
			}
			Some(root) => {
				let root_node = self.nodes.get(&root).ok_or(ValidationError::MissingRoot(root))?;
				if root_node.parent().is_some() {
					return Err(ValidationError::RootHasParent(root));
				}
				let mut visited = HashSet::with_capacity(self.nodes.len());
				self.validate_node(root, None, &mut visited)?;
				if let Some(unreachable) = self.nodes.keys().copied().find(|id| !visited.contains(id)) {
					return Err(ValidationError::Unreachable(unreachable));
				}
				Ok(())
			}
		}
	}

	fn validate_node(
		&self, id: NodeId, expected_parent: Option<NodeId>, visited: &mut HashSet<NodeId>,
	) -> Result<(), ValidationError> {
		if !visited.insert(id) {
			return Err(ValidationError::Cycle(id));
		}
		let node = self.nodes.get(&id).ok_or(ValidationError::MissingNode(id))?;
		if node.parent() != expected_parent {
			return Err(ValidationError::ParentMismatch {
				node: id,
				expected: expected_parent,
				actual: node.parent(),
			});
		}
		match node {
			Node::Leaf(leaf) => {
				if !leaf_meta_is_valid(&leaf.meta) {
					return Err(ValidationError::InvalidLeafLimits(id));
				}
			}
			Node::Split(split) => {
				if split.a == split.b {
					return Err(ValidationError::DuplicateChild {
						split: id,
						child: split.a,
					});
				}
				if split.weights.a == 0 && split.weights.b == 0 {
					return Err(ValidationError::InvalidWeights(id));
				}
				self.validate_node(split.a, Some(id), visited)?;
				self.validate_node(split.b, Some(id), visited)?;
			}
		}
		Ok(())
	}

	/// Returns `true` if `id` exists in the tree.
	#[must_use]
	pub fn contains(&self, id: NodeId) -> bool {
		self.nodes.contains_key(&id)
	}

	/// Returns `true` if `id` exists and refers to a leaf.
	#[must_use]
	pub fn is_leaf(&self, id: NodeId) -> bool {
		matches!(self.nodes.get(&id), Some(Node::Leaf(_)))
	}

	/// Returns `true` if `id` exists and refers to a split.
	#[must_use]
	pub fn is_split(&self, id: NodeId) -> bool {
		matches!(self.nodes.get(&id), Some(Node::Split(_)))
	}

	/// Returns the parent of `id`, if the node exists and is not the root.
	#[must_use]
	pub fn parent_of(&self, id: NodeId) -> Option<NodeId> {
		self.nodes.get(&id).and_then(Node::parent)
	}

	/// Inserts the first root leaf into an empty tree.
	///
	/// The inserted leaf becomes the tree root.
	pub fn insert_root(&mut self, payload: T, meta: LeafMeta) -> Result<NodeId, OpError> {
		if self.root_id().is_some() {
			return Err(OpError::NonEmpty);
		}
		validate_leaf_meta(&meta)?;
		let id = self.new_leaf(payload, meta);
		self.set_root(Some(id));
		self.validate().map_err(OpError::Validation)?;
		Ok(id)
	}

	/// Splits an existing leaf by inserting a new sibling leaf.
	///
	/// Returns the newly inserted leaf id. The existing leaf keeps its id, and the returned leaf is
	/// attached next to it under a new split parent.
	pub fn split_leaf(
		&mut self, leaf: NodeId, axis: Axis, slot: Slot, payload: T, meta: LeafMeta, weights: Option<WeightPair>,
	) -> Result<NodeId, OpError> {
		if !self.contains(leaf) {
			return Err(OpError::MissingNode(leaf));
		}
		if !self.is_leaf(leaf) {
			return Err(OpError::NotLeaf(leaf));
		}
		validate_leaf_meta(&meta)?;
		let weights = weights.unwrap_or_default().checked().ok_or(OpError::InvalidWeights)?;
		let new_leaf = self.new_leaf(payload, meta);
		self.attach_as_sibling(leaf, new_leaf, axis, slot, weights);
		self.validate().map_err(OpError::Validation)?;
		Ok(new_leaf)
	}

	/// Wraps `target` together with a newly inserted sibling leaf.
	///
	/// Returns the newly inserted leaf id. Unlike [`Self::split_leaf`], `target` may be either a
	/// leaf or an existing split subtree.
	pub fn wrap_node(
		&mut self, target: NodeId, axis: Axis, slot: Slot, payload: T, meta: LeafMeta, weights: Option<WeightPair>,
	) -> Result<NodeId, OpError> {
		if !self.contains(target) {
			return Err(OpError::MissingNode(target));
		}
		validate_leaf_meta(&meta)?;
		let weights = weights.unwrap_or_default().checked().ok_or(OpError::InvalidWeights)?;
		let new_leaf = self.new_leaf(payload, meta);
		self.attach_as_sibling(target, new_leaf, axis, slot, weights);
		self.validate().map_err(OpError::Validation)?;
		Ok(new_leaf)
	}

	/// Removes `leaf` and collapses any unary parent introduced by the removal.
	///
	/// Returns the surviving replacement site, or `None` when the tree becomes empty. Removing the
	/// only root leaf empties the tree.
	pub fn remove_leaf(&mut self, leaf: NodeId) -> Result<Option<NodeId>, OpError> {
		if !self.contains(leaf) {
			return Err(OpError::MissingNode(leaf));
		}
		if !self.is_leaf(leaf) {
			return Err(OpError::NotLeaf(leaf));
		}
		// The internal helper only fails for missing/non-leaf ids, which we've ruled out above.
		let removed = self
			.remove_leaf_and_collapse(leaf)
			.map(RemoveLeafResult::replacement_site)
			.expect("validated leaf removal should succeed");
		self.validate().map_err(OpError::Validation)?;
		Ok(removed)
	}

	/// Swaps two distinct, structurally disjoint nodes.
	pub fn swap_nodes(&mut self, a: NodeId, b: NodeId) -> Result<(), OpError> {
		if a == b {
			return Err(OpError::SameNode);
		}
		if !self.contains(a) {
			return Err(OpError::MissingNode(a));
		}
		if !self.contains(b) {
			return Err(OpError::MissingNode(b));
		}
		if self.contains_in_subtree(a, b) || self.contains_in_subtree(b, a) {
			return Err(OpError::AncestorConflict);
		}
		self.swap_disjoint_nodes(a, b)
			.expect("validated disjoint tree swap should succeed");
		self.validate().map_err(OpError::Validation)
	}

	/// Moves `selection` so it becomes a sibling of `target`.
	///
	/// Returns the newly introduced split id. `target` may be an ancestor of `selection`; the move
	/// is interpreted relative to the subtree that remains after detaching `selection`.
	pub fn move_subtree_as_sibling_of(
		&mut self, selection: NodeId, target: NodeId, axis: Axis, slot: Slot, weights: Option<WeightPair>,
	) -> Result<NodeId, OpError> {
		let split = self.move_subtree_as_sibling_of_inner(
			selection,
			target,
			axis,
			slot,
			weights.unwrap_or_default().checked().ok_or(OpError::InvalidWeights)?,
		)?;
		self.validate().map_err(OpError::Validation)?;
		Ok(split)
	}

	/// Replaces the payload stored in `leaf`.
	///
	/// Payload changes do not affect validation or solved geometry.
	pub fn set_leaf_payload(&mut self, leaf: NodeId, payload: T) -> Result<(), OpError> {
		self.leaf_mut_or_error(leaf)?.payload = payload;
		Ok(())
	}

	/// Replaces the sizing metadata stored in `leaf`.
	///
	/// Returns `true` when the metadata changed. Metadata is validated atomically; invalid metadata
	/// leaves the tree unchanged.
	pub fn set_leaf_meta(&mut self, leaf: NodeId, meta: LeafMeta) -> Result<bool, OpError> {
		validate_leaf_meta(&meta)?;
		let changed = {
			// Keep the mutable leaf borrow inside this block so the whole tree can be revalidated
			// immediately afterward without splitting the update into a separate helper.
			let node = self.leaf_mut_or_error(leaf)?;
			if node.meta == meta {
				return Ok(false);
			}
			node.meta = meta;
			true
		};
		self.validate().map_err(OpError::Validation)?;
		Ok(changed)
	}

	pub(crate) fn new_leaf(&mut self, payload: T, meta: LeafMeta) -> NodeId {
		let id = self.alloc_id();
		self.nodes.insert(
			id,
			Node::Leaf(LeafNode {
				parent: None,
				payload,
				meta,
			}),
		);
		id
	}

	pub(crate) fn new_split(&mut self, axis: Axis, a: NodeId, b: NodeId, weights: WeightPair) -> NodeId {
		let id = self.alloc_id();
		self.nodes.insert(
			id,
			Node::Split(SplitNode {
				parent: None,
				axis,
				a,
				b,
				weights,
			}),
		);
		id
	}

	pub(crate) fn set_parent(&mut self, id: NodeId, parent: Option<NodeId>) {
		*self
			.nodes
			.get_mut(&id)
			.expect("node missing when setting parent")
			.parent_mut() = parent;
	}

	pub(crate) fn replace_child(&mut self, parent: NodeId, old: NodeId, new: NodeId) {
		let split = self
			.nodes
			.get_mut(&parent)
			.and_then(Node::as_split_mut)
			.expect("parent missing or not split");
		if split.a == old {
			split.a = new;
		} else if split.b == old {
			split.b = new;
		} else {
			panic!("old child not found under parent");
		}
		self.set_parent(new, Some(parent));
	}

	/// Returns the child ids of a split as `(a, b)`.
	pub fn children_of(&self, id: NodeId) -> Option<(NodeId, NodeId)> {
		self.nodes
			.get(&id)
			.and_then(Node::as_split)
			.map(|split| (split.a, split.b))
	}

	/// Returns the sibling of `id`, if `id` exists and has a parent split.
	#[must_use]
	pub fn sibling_of(&self, id: NodeId) -> Option<NodeId> {
		let parent = self.parent_of(id)?;
		let split = self.nodes.get(&parent)?.as_split()?;
		if split.a == id {
			Some(split.b)
		} else if split.b == id {
			Some(split.a)
		} else {
			None
		}
	}

	/// Returns the path from `id` to the root, inclusive.
	///
	/// The returned vector starts with `id` and ends with the root. Returns `None` when `id` is
	/// not present in the tree.
	#[must_use]
	pub fn path_to_root(&self, id: NodeId) -> Option<Vec<NodeId>> {
		self.collect_parent_chain(id, true)
	}

	/// Returns the ancestors of `id` from nearest parent to root.
	///
	/// `id` itself is not included. Returns `None` when `id` is not present in the tree.
	#[must_use]
	pub fn ancestors_nearest_first(&self, id: NodeId) -> Option<Vec<NodeId>> {
		self.collect_parent_chain(id, false)
	}

	/// Returns `true` if `needle` occurs anywhere inside the subtree rooted at `root`.
	///
	/// A node is considered to be inside its own subtree.
	#[must_use]
	pub fn contains_in_subtree(&self, root: NodeId, needle: NodeId) -> bool {
		if !self.contains(needle) {
			return false;
		}
		let mut cursor = Some(needle);
		while let Some(id) = cursor {
			if id == root {
				return true;
			}
			cursor = self.parent_of(id);
		}
		false
	}

	/// Returns the first leaf reachable from `id` in depth-first `A`-before-`B` order.
	#[must_use]
	pub fn first_leaf(&self, mut id: NodeId) -> Option<NodeId> {
		loop {
			match self.nodes.get(&id)? {
				Node::Leaf(_) => return Some(id),
				Node::Split(split) => id = split.a,
			}
		}
	}

	/// Returns leaf ids in depth-first `A`-before-`B` order.
	///
	/// This ordering is the crate's canonical leaf traversal and is reused by preset rebuilds and
	/// geometry tie-breaking.
	///
	/// ```text
	///           [X]
	///          /   \
	///      A [Y]   [d]
	///        / \
	///     A[a] [b]B
	///
	/// DFS A-before-B leaf order: [a, b, d]
	/// ```
	///
	/// ```
	/// use glorp_tiles::{Axis, LeafMeta, Session, Slot};
	///
	/// let mut session = Session::new();
	/// let left = session.insert_root("left", LeafMeta::default())?;
	/// let right = session.split_focus(Axis::X, Slot::B, "right", LeafMeta::default(), None)?;
	///
	/// let root = session.tree().root_id().expect("root should exist");
	/// assert_eq!(session.tree().leaf_ids_dfs(root), Some(vec![left, right]));
	/// # Ok::<(), Box<dyn std::error::Error>>(())
	/// ```
	///
	/// Returns `None` when `root` is not present in the tree.
	#[must_use]
	pub fn leaf_ids_dfs(&self, root: NodeId) -> Option<Vec<NodeId>> {
		if !self.contains(root) {
			return None;
		}
		let mut out = Vec::new();
		let _ = self.visit_leaves_dfs(root, &mut |id| {
			out.push(id);
			ControlFlow::<()>::Continue(())
		});
		Some(out)
	}

	/// Returns split ids in subtree postorder, with descendants before ancestors.
	///
	/// The subtree root split, if present, is returned last. This order is intended for bottom-up
	/// teardown and rewrite paths. Returns `None` when `root` is not present in the tree.
	#[must_use]
	pub fn split_ids_postorder(&self, root: NodeId) -> Option<Vec<NodeId>> {
		if !self.contains(root) {
			return None;
		}
		let mut out = Vec::new();
		self.collect_split_ids(root, &mut out);
		Some(out)
	}

	#[must_use]
	fn remaining_subtree_after_removal(&self, current: NodeId, removed: NodeId) -> Option<NodeId> {
		if current == removed {
			return None;
		}
		// If `removed` is somewhere below `current`, detaching it may collapse one level of structure;
		// callers need the surviving subtree root that will still exist afterward.
		if !self.contains_in_subtree(current, removed) {
			Some(current)
		} else if self.parent_of(removed) == Some(current) {
			self.sibling_of(removed)
		} else {
			Some(current)
		}
	}

	pub(crate) fn visit_leaves_dfs<B>(
		&self, id: NodeId, f: &mut impl FnMut(NodeId) -> ControlFlow<B>,
	) -> ControlFlow<B> {
		match self.nodes.get(&id).expect("missing node in visit_leaves_dfs") {
			Node::Leaf(_) => f(id),
			Node::Split(split) => match self.visit_leaves_dfs(split.a, f) {
				ControlFlow::Continue(()) => self.visit_leaves_dfs(split.b, f),
				ControlFlow::Break(value) => ControlFlow::Break(value),
			},
		}
	}

	fn collect_split_ids(&self, id: NodeId, out: &mut Vec<NodeId>) {
		if let Some((a, b)) = self.children_of(id) {
			self.collect_split_ids(a, out);
			self.collect_split_ids(b, out);
			out.push(id);
		}
	}

	fn split_root_error(&self, id: NodeId) -> OpError {
		if self.contains(id) {
			OpError::NotSplit(id)
		} else {
			OpError::MissingNode(id)
		}
	}

	fn collect_parent_chain(&self, id: NodeId, include_self: bool) -> Option<Vec<NodeId>> {
		if !self.contains(id) {
			return None;
		}
		// `include_self = false` starts at the parent so the same helper can serve both
		// `path_to_root` and `ancestors_nearest_first` without post-processing.
		let mut cursor = include_self.then_some(id).or_else(|| self.parent_of(id));
		let mut out = Vec::new();
		while let Some(id) = cursor {
			out.push(id);
			cursor = self.parent_of(id);
		}
		Some(out)
	}

	fn swap_parent_slots(&mut self, parent: NodeId) {
		let split = self
			.nodes
			.get_mut(&parent)
			.and_then(Node::as_split_mut)
			.expect("split missing");
		std::mem::swap(&mut split.a, &mut split.b);
	}

	fn set_split_weights_inner(&mut self, id: NodeId, weights: WeightPair) -> Option<bool> {
		let split = self.split_mut(id)?;
		let changed = split.weights() != weights;
		if changed {
			split.set_weights(weights);
		}
		Some(changed)
	}

	/// Toggles the axis of `split`.
	///
	/// This preserves child ids and weights while flipping `X <-> Y`.
	pub fn toggle_split_axis(&mut self, split: NodeId) -> Result<(), OpError> {
		let split = self.split_mut_or_error(split)?;
		split.set_axis(split.axis().toggled());
		self.validate().map_err(OpError::Validation)
	}

	/// Replaces the relative weight preference stored on `split`.
	///
	/// Returns `true` when the weights changed. Invalid all-zero weights are rejected.
	pub fn set_split_weights(&mut self, split: NodeId, weights: WeightPair) -> Result<bool, OpError> {
		let weights = weights.checked().ok_or(OpError::InvalidWeights)?;
		let changed = {
			let split = self.split_mut_or_error(split)?;
			if split.weights() == weights {
				return Ok(false);
			}
			split.set_weights(weights);
			true
		};
		if changed {
			self.validate().map_err(OpError::Validation)?;
		}
		Ok(changed)
	}

	fn rebalance_subtree_binary_equal_inner(&mut self, id: NodeId) -> Option<bool> {
		match self.children_of(id) {
			Some((a, b)) => {
				let changed_a = self.rebalance_subtree_binary_equal_inner(a)?;
				let changed_b = self.rebalance_subtree_binary_equal_inner(b)?;
				let changed_here = self.set_split_weights_inner(id, WeightPair::default())?;
				Some(changed_a || changed_b || changed_here)
			}
			None => Some(false),
		}
	}

	fn rebalance_subtree_leaf_count_inner(&mut self, id: NodeId) -> Option<(u32, bool)> {
		match self.children_of(id) {
			Some((a, b)) => {
				let (count_a, changed_a) = self.rebalance_subtree_leaf_count_inner(a)?;
				let (count_b, changed_b) = self.rebalance_subtree_leaf_count_inner(b)?;
				let changed_here = self.set_split_weights_inner(id, canonicalize_weights(count_a, count_b))?;
				Some((count_a + count_b, changed_a || changed_b || changed_here))
			}
			None => Some((1, false)),
		}
	}

	/// Rebalances every split in the subtree rooted at `id` to equal `1:1` weights.
	///
	/// Returns `true` when any split weights changed. Leaf roots return [`OpError::NotSplit`].
	pub fn rebalance_subtree_binary_equal(&mut self, id: NodeId) -> Result<bool, OpError> {
		let changed = self
			.rebalance_subtree_binary_equal_inner(id)
			.ok_or_else(|| self.split_root_error(id))?;
		if changed {
			self.validate().map_err(OpError::Validation)?;
		}
		Ok(changed)
	}

	/// Rebalances every split in the subtree rooted at `id` using descendant leaf counts.
	///
	/// Returns `true` when any split weights changed. Leaf roots return [`OpError::NotSplit`].
	pub fn rebalance_subtree_leaf_count(&mut self, id: NodeId) -> Result<bool, OpError> {
		let changed = self
			.rebalance_subtree_leaf_count_inner(id)
			.ok_or_else(|| self.split_root_error(id))?
			.1;
		if changed {
			self.validate().map_err(OpError::Validation)?;
		}
		Ok(changed)
	}

	fn mirror_subtree_axis_inner(&mut self, id: NodeId, axis: Axis) -> bool {
		if let Some((a, b)) = self.children_of(id) {
			let changed_a = self.mirror_subtree_axis_inner(a, axis);
			let changed_b = self.mirror_subtree_axis_inner(b, axis);
			let split = self.split_mut(id).expect("split missing during mirror");
			if split.axis() == axis {
				split.swap_children();
				split.swap_weights();
				true
			} else {
				changed_a || changed_b
			}
		} else {
			false
		}
	}

	/// Mirrors the subtree rooted at `id` across `axis`.
	///
	/// Returns `true` when the topology changed. Leaf roots are a no-op and return `Ok(false)`.
	pub fn mirror_subtree(&mut self, id: NodeId, axis: Axis) -> Result<bool, OpError> {
		if !self.contains(id) {
			return Err(OpError::MissingNode(id));
		}
		let changed = self.mirror_subtree_axis_inner(id, axis);
		if changed {
			self.validate().map_err(OpError::Validation)?;
		}
		Ok(changed)
	}

	/// Rebuilds the subtree rooted at `id` to match `preset`.
	///
	/// Returns `Ok(None)` when `id` is a leaf or the subtree already matches `preset`.
	/// Otherwise returns the rebuilt subtree root id. Existing leaf ids, payloads, and metadata are
	/// preserved across the rebuild.
	pub fn apply_preset(&mut self, id: NodeId, preset: PresetKind) -> Result<Option<NodeId>, OpError> {
		if !self.contains(id) {
			return Err(OpError::MissingNode(id));
		}
		let rebuilt = apply_preset_subtree(self, id, preset)?;
		if rebuilt.is_some() {
			self.validate().map_err(OpError::Validation)?;
		}
		Ok(rebuilt)
	}

	fn reattach_child(&mut self, parent: Option<NodeId>, replaced: Option<NodeId>, child: NodeId) {
		if let Some(parent) = parent {
			self.replace_child(
				parent,
				replaced.expect("replacement target missing for non-root reattach"),
				child,
			);
		} else {
			self.root = Some(child);
			self.set_parent(child, None);
		}
	}

	fn collapse_unary_parent(&mut self, removed_child: NodeId) -> Option<NodeId> {
		let parent = self.parent_of(removed_child)?;
		let sibling = self.sibling_of(removed_child)?;
		self.reattach_child(self.parent_of(parent), Some(parent), sibling);
		self.nodes.remove(&parent);
		Some(sibling)
	}

	pub(crate) fn remove_leaf_and_collapse(&mut self, leaf: NodeId) -> Option<RemoveLeafResult> {
		if !self.is_leaf(leaf) {
			return None;
		}
		if self.root_id() == Some(leaf) {
			self.set_root(None);
			self.remove_node(leaf)?;
			return Some(RemoveLeafResult::Emptied);
		}
		// Collapsing the unary parent already identifies the surviving attachment site, so there is
		// no need to probe the sibling separately before the rewrite.
		let replacement = self.collapse_unary_parent(leaf)?;
		self.remove_node(leaf)?;
		Some(RemoveLeafResult::Replaced(replacement))
	}

	pub(crate) fn swap_disjoint_nodes(&mut self, a: NodeId, b: NodeId) -> Option<()> {
		let parent_a = self.parent_of(a);
		let parent_b = self.parent_of(b);
		if parent_a == parent_b {
			let parent = parent_a?;
			self.swap_parent_slots(parent);
			return Some(());
		}
		self.reattach_child(parent_a, Some(a), b);
		self.reattach_child(parent_b, Some(b), a);
		Some(())
	}

	fn detach_subtree(&mut self, id: NodeId) {
		if self.root == Some(id) {
			self.root = None;
			self.set_parent(id, None);
			return;
		}
		// Detaching a non-root subtree also removes its former parent, replacing that split with the
		// detached node's sibling so the remaining tree stays connected and strictly binary.
		let parent = self.parent_of(id).expect("detached subtree missing parent");
		let sibling = self.sibling_of(id).expect("detached subtree missing sibling");
		self.reattach_child(self.parent_of(parent), Some(parent), sibling);
		self.nodes.remove(&parent);
		self.set_parent(id, None);
	}

	pub(crate) fn attach_as_sibling(
		&mut self, target: NodeId, incoming: NodeId, axis: Axis, slot: Slot, weights: WeightPair,
	) -> NodeId {
		let (a, b) = match slot {
			Slot::A => (incoming, target),
			Slot::B => (target, incoming),
		};
		let parent_of_target = self.parent_of(target);
		let split_id = self.new_split(axis, a, b, weights);
		self.set_parent(a, Some(split_id));
		self.set_parent(b, Some(split_id));
		self.reattach_child(parent_of_target, Some(target), split_id);
		split_id
	}

	fn move_subtree_as_sibling_of_inner(
		&mut self, selection: NodeId, target: NodeId, axis: Axis, slot: Slot, weights: WeightPair,
	) -> Result<NodeId, OpError> {
		if selection == target {
			return Err(OpError::SameNode);
		}
		if !self.contains(selection) {
			return Err(OpError::MissingNode(selection));
		}
		if !self.contains(target) {
			return Err(OpError::MissingNode(target));
		}
		if self.contains_in_subtree(selection, target) {
			return Err(OpError::TargetInsideSelection);
		}

		let effective_target = self
			.remaining_subtree_after_removal(target, selection)
			.ok_or(OpError::AncestorConflict)?;
		self.detach_subtree(selection);
		Ok(self.attach_as_sibling(effective_target, selection, axis, slot, weights))
	}

	fn alloc_id(&mut self) -> NodeId {
		let id = NodeId::from_raw(self.next_id_raw);
		self.next_id_raw += 1;
		id
	}
}

fn validate_leaf_meta(meta: &LeafMeta) -> Result<(), OpError> {
	leaf_meta_is_valid(meta).then_some(()).ok_or(OpError::InvalidLeafMeta)
}

struct Fingerprinter(u128);

impl Fingerprinter {
	const OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
	const PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;

	fn new() -> Self {
		Self(Self::OFFSET)
	}

	fn finish(self) -> (u64, u64) {
		(
			u64::try_from(self.0 >> 64).expect("fingerprint high half should fit u64"),
			u64::try_from(self.0 & u128::from(u64::MAX)).expect("fingerprint low half should fit u64"),
		)
	}

	fn write_tag(&mut self, tag: &[u8]) {
		self.write_u64(u64::try_from(tag.len()).expect("tag length should fit u64"));
		for byte in tag {
			self.write_u8(*byte);
		}
	}

	fn write_u8(&mut self, value: u8) {
		self.0 ^= u128::from(value);
		self.0 = self.0.wrapping_mul(Self::PRIME);
	}

	fn write_u16(&mut self, value: u16) {
		for byte in value.to_le_bytes() {
			self.write_u8(byte);
		}
	}

	fn write_u32(&mut self, value: u32) {
		for byte in value.to_le_bytes() {
			self.write_u8(byte);
		}
	}

	fn write_u64(&mut self, value: u64) {
		for byte in value.to_le_bytes() {
			self.write_u8(byte);
		}
	}

	fn write_node_id(&mut self, id: NodeId) {
		self.write_u64(id.into_raw());
	}

	fn write_option_node_id(&mut self, id: Option<NodeId>) {
		match id {
			Some(id) => {
				self.write_u8(1);
				self.write_node_id(id);
			}
			None => self.write_u8(0),
		}
	}

	fn write_axis(&mut self, axis: Axis) {
		self.write_u8(match axis {
			Axis::X => 0,
			Axis::Y => 1,
		});
	}

	fn write_option_u32(&mut self, value: Option<u32>) {
		match value {
			Some(value) => {
				self.write_u8(1);
				self.write_u32(value);
			}
			None => self.write_u8(0),
		}
	}

	fn write_weight_pair(&mut self, weights: WeightPair) {
		self.write_u32(weights.a);
		self.write_u32(weights.b);
	}

	fn write_leaf_meta(&mut self, meta: &LeafMeta) {
		self.write_u32(meta.limits.min_w);
		self.write_u32(meta.limits.min_h);
		self.write_option_u32(meta.limits.max_w);
		self.write_option_u32(meta.limits.max_h);
		self.write_u16(meta.priority.shrink);
		self.write_u16(meta.priority.grow);
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn replace_child_sets_new_child_parent_in_non_root_path() {
		let mut tree = Tree::default();
		let a = tree.new_leaf(1_u8, LeafMeta::default());
		let b = tree.new_leaf(2_u8, LeafMeta::default());
		let c = tree.new_leaf(3_u8, LeafMeta::default());
		let d = tree.new_leaf(4_u8, LeafMeta::default());
		let inner = tree.new_split(Axis::X, a, b, WeightPair::default());
		tree.set_parent(a, Some(inner));
		tree.set_parent(b, Some(inner));
		let root = tree.new_split(Axis::Y, inner, d, WeightPair::default());
		tree.set_parent(inner, Some(root));
		tree.set_parent(d, Some(root));
		tree.set_root(Some(root));

		let replacement = tree.attach_as_sibling(a, c, Axis::Y, Slot::A, WeightPair::default());

		assert_eq!(tree.parent_of(replacement), Some(inner));
		assert_eq!(tree.parent_of(a), Some(replacement));
		assert_eq!(tree.parent_of(c), Some(replacement));
		tree.validate().expect("replace_child should set parent");
	}

	#[test]
	fn swap_disjoint_nodes_non_root_paths_preserve_parent_links() {
		let mut tree = Tree::default();
		let a = tree.new_leaf(1_u8, LeafMeta::default());
		let b = tree.new_leaf(2_u8, LeafMeta::default());
		let c = tree.new_leaf(3_u8, LeafMeta::default());
		let d = tree.new_leaf(4_u8, LeafMeta::default());
		let left = tree.new_split(Axis::X, a, b, WeightPair::default());
		tree.set_parent(a, Some(left));
		tree.set_parent(b, Some(left));
		let right = tree.new_split(Axis::X, c, d, WeightPair::default());
		tree.set_parent(c, Some(right));
		tree.set_parent(d, Some(right));
		let root = tree.new_split(Axis::Y, left, right, WeightPair::default());
		tree.set_parent(left, Some(root));
		tree.set_parent(right, Some(root));
		tree.set_root(Some(root));

		tree.swap_disjoint_nodes(a, c)
			.expect("swap between non-root leaves should succeed");

		assert_eq!(tree.parent_of(a), Some(right));
		assert_eq!(tree.parent_of(c), Some(left));
		tree.validate().expect("swap should preserve parent links");
	}
}
