use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::{
    error::{OpError, ValidationError},
    geom::{Axis, Slot},
    ids::NodeId,
    limits::{LeafMeta, WeightPair, canonicalize_weights},
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LeafNode<T> {
    parent: Option<NodeId>,
    payload: T,
    meta: LeafMeta,
}

impl<T> LeafNode<T> {
    #[must_use]
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    #[must_use]
    pub fn payload(&self) -> &T {
        &self.payload
    }

    #[must_use]
    pub fn meta(&self) -> &LeafMeta {
        &self.meta
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitNode {
    parent: Option<NodeId>,
    axis: Axis,
    a: NodeId,
    b: NodeId,
    weights: WeightPair,
}

impl SplitNode {
    #[must_use]
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }

    #[must_use]
    pub fn axis(&self) -> Axis {
        self.axis
    }

    #[must_use]
    pub fn a(&self) -> NodeId {
        self.a
    }

    #[must_use]
    pub fn b(&self) -> NodeId {
        self.b
    }

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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tree<T> {
    root: Option<NodeId>,
    nodes: HashMap<NodeId, Node<T>>,
    next_id: NodeId,
}

impl<T> Default for Tree<T> {
    fn default() -> Self {
        Self {
            root: None,
            nodes: HashMap::new(),
            next_id: 1,
        }
    }
}

impl<T> Tree<T> {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn root_id(&self) -> Option<NodeId> {
        self.root
    }

    #[must_use]
    pub fn split(&self, id: NodeId) -> Option<&SplitNode> {
        self.nodes.get(&id).and_then(Node::as_split)
    }

    #[must_use]
    pub fn leaf(&self, id: NodeId) -> Option<&LeafNode<T>> {
        self.nodes.get(&id).and_then(Node::as_leaf)
    }

    #[must_use]
    pub fn node_ids(&self) -> Vec<NodeId> {
        let mut ids = self.nodes.keys().copied().collect::<Vec<_>>();
        ids.sort_unstable();
        ids
    }

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

    pub(crate) fn set_root(&mut self, root: Option<NodeId>) {
        self.root = root;
    }

    fn split_mut(&mut self, id: NodeId) -> Option<&mut SplitNode> {
        self.nodes.get_mut(&id).and_then(Node::as_split_mut)
    }

    pub(crate) fn remove_node(&mut self, id: NodeId) -> Option<Node<T>> {
        self.nodes.remove(&id)
    }

    pub fn validate(&self) -> Result<(), ValidationError> {
        match self.root {
            None => {
                if self.nodes.is_empty() {
                    Ok(())
                } else {
                    let extra = self.nodes.keys().copied().min().unwrap_or_default();
                    Err(ValidationError::Unreachable(extra))
                }
            }
            Some(root) => {
                let root_node = self
                    .nodes
                    .get(&root)
                    .ok_or(ValidationError::MissingRoot(root))?;
                if root_node.parent().is_some() {
                    return Err(ValidationError::RootHasParent(root));
                }
                let mut visited = HashSet::new();
                self.validate_node(root, None, &mut visited)?;
                if let Some(unreachable) =
                    self.nodes.keys().copied().find(|id| !visited.contains(id))
                {
                    return Err(ValidationError::Unreachable(unreachable));
                }
                Ok(())
            }
        }
    }

    fn validate_node(
        &self,
        id: NodeId,
        expected_parent: Option<NodeId>,
        visited: &mut HashSet<NodeId>,
    ) -> Result<(), ValidationError> {
        if !visited.insert(id) {
            return Err(ValidationError::Cycle(id));
        }
        let node = self
            .nodes
            .get(&id)
            .ok_or(ValidationError::MissingNode(id))?;
        if node.parent() != expected_parent {
            return Err(ValidationError::ParentMismatch {
                node: id,
                expected: expected_parent.unwrap_or_default(),
                actual: node.parent(),
            });
        }
        match node {
            Node::Leaf(leaf) => {
                let limits = leaf.meta.limits;
                if limits.max_w.is_some_and(|max_w| limits.min_w > max_w)
                    || limits.max_h.is_some_and(|max_h| limits.min_h > max_h)
                    || leaf.meta.priority.shrink == 0
                    || leaf.meta.priority.grow == 0
                {
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
                self.validate_child(id, split.a, visited)?;
                self.validate_child(id, split.b, visited)?;
            }
        }
        Ok(())
    }

    fn validate_child(
        &self,
        parent: NodeId,
        child: NodeId,
        visited: &mut HashSet<NodeId>,
    ) -> Result<(), ValidationError> {
        if !self.nodes.contains_key(&child) {
            return Err(ValidationError::MissingNode(child));
        }
        self.validate_node(child, Some(parent), visited)
    }

    #[must_use]
    pub fn contains(&self, id: NodeId) -> bool {
        self.nodes.contains_key(&id)
    }

    #[must_use]
    pub fn is_leaf(&self, id: NodeId) -> bool {
        matches!(self.nodes.get(&id), Some(Node::Leaf(_)))
    }

    #[must_use]
    pub fn is_split(&self, id: NodeId) -> bool {
        matches!(self.nodes.get(&id), Some(Node::Split(_)))
    }

    #[must_use]
    pub fn parent_of(&self, id: NodeId) -> Option<NodeId> {
        self.nodes.get(&id).and_then(Node::parent)
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

    pub(crate) fn new_split(
        &mut self,
        axis: Axis,
        a: NodeId,
        b: NodeId,
        weights: WeightPair,
    ) -> NodeId {
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

    pub fn children_of(&self, id: NodeId) -> Option<(NodeId, NodeId)> {
        self.nodes
            .get(&id)
            .and_then(Node::as_split)
            .map(|split| (split.a, split.b))
    }

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

    #[must_use]
    pub fn path_to_root(&self, mut id: NodeId) -> Vec<NodeId> {
        let mut out = vec![id];
        while let Some(parent) = self.parent_of(id) {
            out.push(parent);
            id = parent;
        }
        out
    }

    #[must_use]
    pub fn ancestors_nearest_first(&self, id: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        let mut cursor = self.parent_of(id);
        while let Some(parent) = cursor {
            out.push(parent);
            cursor = self.parent_of(parent);
        }
        out
    }

    #[must_use]
    pub fn contains_in_subtree(&self, root: NodeId, needle: NodeId) -> bool {
        if root == needle {
            return true;
        }
        match self.nodes.get(&root) {
            Some(Node::Leaf(_)) | None => false,
            Some(Node::Split(split)) => {
                self.contains_in_subtree(split.a, needle)
                    || self.contains_in_subtree(split.b, needle)
            }
        }
    }

    #[must_use]
    pub fn first_leaf(&self, id: NodeId) -> Option<NodeId> {
        match self.nodes.get(&id)? {
            Node::Leaf(_) => Some(id),
            Node::Split(split) => self
                .first_leaf(split.a)
                .or_else(|| self.first_leaf(split.b)),
        }
    }

    #[must_use]
    pub fn leaf_ids_dfs(&self, root: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.collect_leaf_ids(root, &mut out);
        out
    }

    #[must_use]
    /// Returns split ids in subtree postorder, with descendants before ancestors.
    ///
    /// The subtree root split, if present, is returned last. This order is
    /// intended for bottom-up teardown and rewrite paths.
    pub fn split_ids_postorder(&self, root: NodeId) -> Vec<NodeId> {
        let mut out = Vec::new();
        self.collect_split_ids(root, &mut out);
        out
    }

    #[must_use]
    fn remaining_subtree_after_removal(&self, current: NodeId, removed: NodeId) -> Option<NodeId> {
        if current == removed {
            return None;
        }
        if self.leaf(current).is_some() {
            Some(current)
        } else {
            let split = self.split(current)?;
            match (
                self.remaining_subtree_after_removal(split.a, removed),
                self.remaining_subtree_after_removal(split.b, removed),
            ) {
                (Some(_), Some(_)) => Some(current),
                (Some(id), None) | (None, Some(id)) => Some(id),
                (None, None) => None,
            }
        }
    }

    fn collect_leaf_ids(&self, id: NodeId, out: &mut Vec<NodeId>) {
        match self
            .nodes
            .get(&id)
            .expect("missing node in collect_leaf_ids")
        {
            Node::Leaf(_) => out.push(id),
            Node::Split(split) => {
                self.collect_leaf_ids(split.a, out);
                self.collect_leaf_ids(split.b, out);
            }
        }
    }

    fn collect_split_ids(&self, id: NodeId, out: &mut Vec<NodeId>) {
        if let Some((a, b)) = self.children_of(id) {
            self.collect_split_ids(a, out);
            self.collect_split_ids(b, out);
            out.push(id);
        }
    }

    fn swap_parent_slots(&mut self, parent: NodeId) {
        let split = self
            .nodes
            .get_mut(&parent)
            .and_then(Node::as_split_mut)
            .expect("split missing");
        std::mem::swap(&mut split.a, &mut split.b);
    }

    pub(crate) fn toggle_split_axis(&mut self, id: NodeId) -> Option<()> {
        let split = self.split_mut(id)?;
        split.set_axis(split.axis().toggled());
        Some(())
    }

    pub(crate) fn set_split_weights(&mut self, id: NodeId, weights: WeightPair) -> Option<()> {
        let split = self.split_mut(id)?;
        split.set_weights(weights);
        Some(())
    }

    pub(crate) fn rebalance_subtree_binary_equal(&mut self, id: NodeId) -> Option<()> {
        if self.leaf(id).is_some() {
            return Some(());
        }
        let (a, b) = self.children_of(id)?;
        self.rebalance_subtree_binary_equal(a)?;
        self.rebalance_subtree_binary_equal(b)?;
        self.set_split_weights(id, WeightPair::default())
    }

    pub(crate) fn rebalance_subtree_leaf_count(&mut self, id: NodeId) -> Option<u32> {
        if self.leaf(id).is_some() {
            return Some(1);
        }
        let (a, b) = self.children_of(id)?;
        let count_a = self.rebalance_subtree_leaf_count(a)?;
        let count_b = self.rebalance_subtree_leaf_count(b)?;
        self.set_split_weights(id, canonicalize_weights(count_a, count_b))?;
        Some(count_a + count_b)
    }

    pub(crate) fn mirror_subtree_axis(&mut self, id: NodeId, axis: Axis) {
        let children = match self.children_of(id) {
            Some(children) => children,
            None => return,
        };
        self.mirror_subtree_axis(children.0, axis);
        self.mirror_subtree_axis(children.1, axis);
        let split = self.split_mut(id).expect("split missing during mirror");
        if split.axis() == axis {
            split.swap_children();
            split.swap_weights();
        }
    }

    fn collapse_unary_parent(&mut self, removed_child: NodeId) -> Option<NodeId> {
        let parent = self.parent_of(removed_child)?;
        let sibling = self.sibling_of(removed_child)?;
        let grand = self.parent_of(parent);
        if let Some(grand) = grand {
            self.replace_child(grand, parent, sibling);
        } else {
            self.root = Some(sibling);
            self.set_parent(sibling, None);
        }
        self.nodes.remove(&parent);
        Some(sibling)
    }

    pub(crate) fn remove_leaf_and_collapse(&mut self, leaf: NodeId) -> Option<Option<NodeId>> {
        if !self.is_leaf(leaf) {
            return None;
        }
        if self.root_id() == Some(leaf) {
            self.set_root(None);
            self.remove_node(leaf)?;
            return Some(None);
        }
        let sibling = self.sibling_of(leaf)?;
        let replacement = self.collapse_unary_parent(leaf).unwrap_or(sibling);
        self.remove_node(leaf)?;
        Some(Some(replacement))
    }

    pub(crate) fn swap_disjoint_nodes(&mut self, a: NodeId, b: NodeId) -> Option<()> {
        let parent_a = self.parent_of(a);
        let parent_b = self.parent_of(b);
        if parent_a == parent_b {
            let parent = parent_a?;
            self.swap_parent_slots(parent);
            return Some(());
        }
        match parent_a {
            Some(parent) => self.replace_child(parent, a, b),
            None => {
                self.set_root(Some(b));
                self.set_parent(b, None);
            }
        }
        match parent_b {
            Some(parent) => self.replace_child(parent, b, a),
            None => {
                self.set_root(Some(a));
                self.set_parent(a, None);
            }
        }
        Some(())
    }

    fn detach_subtree(&mut self, id: NodeId) {
        if self.root == Some(id) {
            self.root = None;
            self.set_parent(id, None);
            return;
        }
        let parent = self.parent_of(id).expect("detached subtree missing parent");
        let sibling = self
            .sibling_of(id)
            .expect("detached subtree missing sibling");
        let grand = self.parent_of(parent);
        if let Some(grand) = grand {
            self.replace_child(grand, parent, sibling);
        } else {
            self.root = Some(sibling);
            self.set_parent(sibling, None);
        }
        self.nodes.remove(&parent);
        self.set_parent(id, None);
    }

    pub(crate) fn attach_as_sibling(
        &mut self,
        target: NodeId,
        incoming: NodeId,
        axis: Axis,
        slot: Slot,
        weights: WeightPair,
    ) -> NodeId {
        let (a, b) = match slot {
            Slot::A => (incoming, target),
            Slot::B => (target, incoming),
        };
        let parent_of_target = self.parent_of(target);
        let split_id = self.new_split(axis, a, b, weights);
        self.set_parent(a, Some(split_id));
        self.set_parent(b, Some(split_id));
        match parent_of_target {
            Some(parent) => {
                self.replace_child(parent, target, split_id);
            }
            None => {
                self.root = Some(split_id);
                self.set_parent(split_id, None);
            }
        }
        split_id
    }

    pub(crate) fn move_subtree_as_sibling_of(
        &mut self,
        selection: NodeId,
        target: NodeId,
        axis: Axis,
        slot: Slot,
        weights: WeightPair,
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
        let id = self.next_id;
        self.next_id += 1;
        id
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
