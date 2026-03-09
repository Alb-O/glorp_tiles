//! Immutable solved layout output and diagnostics.
//!
//! A [`Snapshot`] records the solved rectangles for one tree or session revision. It stores
//! rectangles for every node, deterministic split-allocation traces, and any hard-limit
//! violations observed in the solved leaf rectangles.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    geom::{Axis, Rect},
    ids::{NodeId, Revision},
    limits::WeightPair,
};

/// Lexicographic score used when comparing candidate split allocations.
///
/// This is solver output, not a user-facing configuration type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ScoreTuple {
    /// Penalty for falling below child minimum extents.
    pub shortage_penalty: u128,
    /// Penalty for exceeding child maximum extents.
    pub overflow_penalty: u128,
    /// Penalty for deviating from the requested weight ratio.
    pub preference_penalty: u128,
    /// Final deterministic tie-break component.
    pub tie_break: u128,
}

/// Deterministic diagnostic record for one solved split.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SplitTrace {
    /// Split node whose extent was allocated.
    pub split: NodeId,
    /// Axis along which the split was solved.
    pub axis: Axis,
    /// Total extent available to the split on `axis`.
    pub total: u32,
    /// Extent assigned to child `A`.
    pub chosen_a: u32,
    /// Lexicographic score of the chosen allocation.
    pub score: ScoreTuple,
    /// Relative split weights used while solving this split.
    pub weights: WeightPair,
}

/// Kind of hard-limit violation recorded for a solved leaf.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViolationKind {
    /// The solved width fell below `min_w`.
    MinWidth,
    /// The solved height fell below `min_h`.
    MinHeight,
    /// The solved width exceeded `max_w`.
    MaxWidth,
    /// The solved height exceeded `max_h`.
    MaxHeight,
}

/// Hard-limit violation observed for one solved leaf.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Violation {
    /// Leaf node that violated a hard limit.
    pub node: NodeId,
    /// Violated limit dimension and direction.
    pub kind: ViolationKind,
    /// Required minimum or maximum bound on the violated dimension.
    pub required: u32,
    /// Actual solved extent on the violated dimension.
    pub actual: u32,
}

/// Solved layout snapshot for one tree or session revision.
///
/// `node_rects` includes every solved node, not only leaves. `split_traces` records the
/// deterministic allocation decision for each split. `violations` records leaf hard-limit
/// violations discovered after solving, and `strict_feasible` is `true` exactly when
/// `violations.is_empty()`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Snapshot {
    /// Tree or session revision this snapshot was solved for.
    pub revision: Revision,
    /// Root rectangle passed to the solver.
    pub root: Rect,
    /// Solved rectangle for every node reached during the solve.
    pub node_rects: HashMap<NodeId, Rect>,
    /// Deterministic split-allocation diagnostics in solve order.
    pub split_traces: Vec<SplitTrace>,
    /// Leaf hard-limit violations observed in the solved layout.
    pub violations: Vec<Violation>,
    /// Whether the solve satisfied all hard limits.
    pub strict_feasible: bool,
}

impl Snapshot {
    /// Returns the solved rectangle for `node`, if the snapshot contains one.
    ///
    /// This works for any node represented in [`Self::node_rects`], including splits and leaves.
    #[must_use]
    pub fn rect(&self, node: NodeId) -> Option<Rect> {
        self.node_rects.get(&node).copied()
    }
}
