//! Leaf sizing metadata and derived subtree feasibility summaries.
//!
//! Leaves contribute three distinct kinds of sizing information:
//!
//! - [`WeightPair`] expresses relative split preference, not absolute pixels or cells
//! - [`SizeLimits`] defines hard per-leaf minimum and optional maximum extents
//! - [`Priority`] defines shortage and future growth cost weighting
//!
//! The solver combines leaf metadata into a [`Summary`] that describes the feasible envelope of a
//! subtree.

use serde::{Deserialize, Serialize};

use crate::geom::Axis;

/// Relative preference between the `A` and `B` children of a split.
///
/// Weights influence how an extent is divided before hard minimum and maximum constraints are
/// applied. They are ratios, not concrete sizes. The pair `(0, 0)` is invalid, while a one-sided
/// zero is valid and means "give all preferred slack to the other child".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct WeightPair {
    /// Relative preference for the `A` child.
    pub a: u32,
    /// Relative preference for the `B` child.
    pub b: u32,
}

impl WeightPair {
    #[must_use]
    pub(crate) fn checked(self) -> Option<Self> {
        (self.a != 0 || self.b != 0).then_some(self)
    }
}

impl Default for WeightPair {
    fn default() -> Self {
        Self { a: 1, b: 1 }
    }
}

/// Per-leaf shortage and growth priority.
///
/// In the v1 solver, `shrink` directly affects shortage allocation cost. `grow` is carried through
/// the public model for forward compatibility and summary reporting, but does not materially alter
/// v1 solve behavior. Zero priorities are rejected by validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Priority {
    /// Relative cost of shrinking this leaf below its preferred allocation.
    pub shrink: u16,
    /// Relative cost of growing this leaf beyond baseline allocation.
    pub grow: u16,
}

impl Default for Priority {
    fn default() -> Self {
        Self { shrink: 1, grow: 1 }
    }
}

/// Hard per-leaf size envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SizeLimits {
    /// Minimum allowed width.
    pub min_w: u32,
    /// Minimum allowed height.
    pub min_h: u32,
    /// Maximum allowed width, if bounded.
    pub max_w: Option<u32>,
    /// Maximum allowed height, if bounded.
    pub max_h: Option<u32>,
}

impl Default for SizeLimits {
    fn default() -> Self {
        Self {
            min_w: 1,
            min_h: 1,
            max_w: None,
            max_h: None,
        }
    }
}

/// Metadata attached to a leaf node.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct LeafMeta {
    /// Hard minimum and optional maximum extents for the leaf.
    pub limits: SizeLimits,
    /// Relative solver cost parameters for shortage and growth.
    pub priority: Priority,
}

/// Derived feasibility envelope for an entire subtree.
///
/// This summary is computed bottom-up from leaf metadata and lets the solver reason about a
/// subtree without expanding all leaves at each decision point.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Summary {
    /// Minimum width required for strict feasibility.
    pub min_w: u32,
    /// Minimum height required for strict feasibility.
    pub min_h: u32,
    /// Maximum width allowed for strict feasibility, if bounded.
    pub max_w: Option<u32>,
    /// Maximum height allowed for strict feasibility, if bounded.
    pub max_h: Option<u32>,
    /// Number of leaves contained in the summarized subtree.
    pub leaf_count: u32,
    /// Aggregate shortage cost used by the solver on the width and height axes.
    pub shrink_cost: u64,
    /// Aggregate growth cost carried alongside the summary.
    pub grow_cost: u64,
}

impl Summary {
    /// Returns the minimum and optional maximum extent for `axis`.
    #[must_use]
    pub fn axis_limits(self, axis: Axis) -> (u32, Option<u32>) {
        match axis {
            Axis::X => (self.min_w, self.max_w),
            Axis::Y => (self.min_h, self.max_h),
        }
    }
}

/// Reduces a weight pair to canonical form.
///
/// Non-zero pairs are divided by their greatest common divisor. One-sided zero pairs are preserved
/// as `(0, 1)` or `(1, 0)` instead of collapsing to all-zero.
///
/// ```
/// use libtiler::{WeightPair, canonicalize_weights};
///
/// assert_eq!(canonicalize_weights(4, 2), WeightPair { a: 2, b: 1 });
/// assert_eq!(canonicalize_weights(0, 9), WeightPair { a: 0, b: 1 });
/// assert_eq!(canonicalize_weights(7, 0), WeightPair { a: 1, b: 0 });
/// ```
///
/// # Panics
///
/// Panics if both inputs are zero.
#[must_use]
pub fn canonicalize_weights(a: u32, b: u32) -> WeightPair {
    match (a, b) {
        (0, 0) => panic!("invalid zero weight pair"),
        (0, _) => WeightPair { a: 0, b: 1 },
        (_, 0) => WeightPair { a: 1, b: 0 },
        _ => {
            let gcd = gcd(a, b);
            WeightPair {
                a: a / gcd,
                b: b / gcd,
            }
        }
    }
}

const fn gcd(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let next = a % b;
        a = b;
        b = next;
    }
    a
}
