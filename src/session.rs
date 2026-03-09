use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
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
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RebalanceMode {
    BinaryEqual,
    LeafCount,
}

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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

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
                let selection = self
                    .selection
                    .ok_or(ValidationError::EmptyStateInconsistent)?;
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

    #[must_use]
    pub fn solve(&self, root: Rect, policy: &SolverPolicy) -> Snapshot {
        self.try_solve(root, policy)
            .expect("session should maintain a valid and representable tree")
    }

    pub fn try_solve(&self, root: Rect, policy: &SolverPolicy) -> Result<Snapshot, SolveError> {
        crate::solver::solve_with_revision(&self.tree, root, self.revision, policy)
    }

    #[must_use]
    pub fn tree(&self) -> &Tree<T> {
        &self.tree
    }

    #[must_use]
    pub fn focus(&self) -> Option<NodeId> {
        self.focus
    }

    #[must_use]
    pub fn selection(&self) -> Option<NodeId> {
        self.selection
    }

    #[must_use]
    pub fn revision(&self) -> Revision {
        self.revision
    }

    pub fn set_focus_leaf(&mut self, id: NodeId) -> Result<(), OpError> {
        let old_focus = self.focus;
        let old_selection = self.selection;
        self.require_leaf(id)?;
        self.focus = Some(id);
        self.repair_selection_for_current_focus();
        self.validate()
            .map_err(OpError::Validation)
            .inspect_err(|_| {
                self.focus = old_focus;
                self.selection = old_selection;
            })
    }

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
        self.validate()
            .map_err(OpError::Validation)
            .inspect_err(|_| {
                self.focus = old_focus;
                self.selection = old_selection;
            })
    }

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

    pub fn split_focus(
        &mut self,
        axis: Axis,
        slot: Slot,
        payload: T,
        meta: LeafMeta,
        weights: Option<WeightPair>,
    ) -> Result<NodeId, OpError> {
        let focus = self.require_focus_leaf()?;
        let weights = weights
            .unwrap_or_default()
            .checked()
            .ok_or(OpError::InvalidWeights)?;
        let old_selection = self.selection;
        let new_leaf = self.tree.new_leaf(payload, meta);
        let split_id = self
            .tree
            .attach_as_sibling(focus, new_leaf, axis, slot, weights);
        self.repair_after_mutation(focus, old_selection, Some(split_id));
        self.bump_revision();
        self.validate().map_err(OpError::Validation)?;
        Ok(new_leaf)
    }

    pub fn wrap_selection(
        &mut self,
        axis: Axis,
        slot: Slot,
        payload: T,
        meta: LeafMeta,
        weights: Option<WeightPair>,
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
            weights
                .unwrap_or_default()
                .checked()
                .ok_or(OpError::InvalidWeights)?,
        );
        self.repair_after_mutation(focus, old_selection, Some(split_id));
        self.bump_revision();
        self.validate().map_err(OpError::Validation)?;
        Ok(new_leaf)
    }

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

    pub fn move_selection_as_sibling_of(
        &mut self,
        target: NodeId,
        axis: Axis,
        slot: Slot,
    ) -> Result<(), OpError> {
        let selection = self.selection.ok_or(OpError::Empty)?;
        let focus = self.require_focus_leaf()?;
        let old_selection = self.selection;
        let split_id = self.tree.move_subtree_as_sibling_of(
            selection,
            target,
            axis,
            slot,
            WeightPair::default(),
        )?;
        self.repair_after_mutation(focus, old_selection, Some(split_id));
        self.bump_revision();
        self.validate().map_err(OpError::Validation)?;
        Ok(())
    }

    pub fn focus_dir(&mut self, dir: Direction, snap: &Snapshot) -> Result<(), NavError> {
        self.ensure_fresh_snapshot(snap).map_err(|error| {
            map_op_to_nav(error).expect("focus_dir should only map nav-compatible op errors")
        })?;
        let focus = self.focus.ok_or(NavError::Empty)?;
        let leaf_rects = self.leaf_rects_from_snapshot(snap)?;
        let next =
            best_neighbor(&self.tree, &leaf_rects, focus, dir).ok_or(NavError::NoCandidate)?;
        self.focus = Some(next);
        self.repair_selection_for_current_focus();
        self.validate().map_err(NavError::Validation)
    }

    pub fn select_parent(&mut self) -> Result<(), OpError> {
        let base = self.selection.or(self.focus).ok_or(OpError::Empty)?;
        let parent = self.tree.parent_of(base).ok_or(OpError::NoParent(base))?;
        self.selection = Some(parent);
        self.validate().map_err(OpError::Validation)
    }

    pub fn select_focus(&mut self) {
        self.selection = self.focus;
    }

    pub fn grow_focus(
        &mut self,
        dir: Direction,
        amount: u32,
        strategy: ResizeStrategy,
        snap: &Snapshot,
    ) -> Result<(), OpError> {
        self.resize_focus(dir, amount, strategy, snap, true)
    }

    pub fn shrink_focus(
        &mut self,
        dir: Direction,
        amount: u32,
        strategy: ResizeStrategy,
        snap: &Snapshot,
    ) -> Result<(), OpError> {
        self.resize_focus(dir, amount, strategy, snap, false)
    }

    pub fn toggle_axis(&mut self) -> Result<(), OpError> {
        let selection = self.selection.ok_or(OpError::Empty)?;
        self.tree
            .toggle_split_axis(selection)
            .ok_or(OpError::NotSplit(selection))?;
        self.bump_revision();
        self.validate().map_err(OpError::Validation)
    }

    pub fn mirror_selection(&mut self, axis: Axis) -> Result<(), OpError> {
        let selection = self.selection.ok_or(OpError::Empty)?;
        self.tree.mirror_subtree_axis(selection, axis);
        self.bump_revision();
        self.validate().map_err(OpError::Validation)
    }

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
        &mut self,
        dir: Direction,
        amount: u32,
        strategy: ResizeStrategy,
        snap: &Snapshot,
        outward: bool,
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
        &mut self,
        old_focus: NodeId,
        old_selection: Option<NodeId>,
        replacement_site: Option<NodeId>,
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
    use super::map_op_to_nav;
    use crate::{NavError, OpError, ValidationError};

    #[test]
    fn nav_compatible_op_errors_map_exactly() {
        assert_eq!(map_op_to_nav(OpError::Empty), Some(NavError::Empty));
        assert_eq!(
            map_op_to_nav(OpError::StaleSnapshot),
            Some(NavError::StaleSnapshot)
        );
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
