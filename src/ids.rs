//! Semantic identifiers used by the public API.

use std::sync::atomic::{AtomicU64, Ordering};

/// Opaque identifier for a node inside a single tree or session.
///
/// Node ids are monotonically allocated within a structure, but callers should treat them as
/// stable opaque identities rather than inferring topology from numeric ordering.
pub type NodeId = u64;

/// Monotonic session revision used to detect stale snapshots.
///
/// Geometry-based commands require a snapshot produced from the current revision.
pub type Revision = u64;

/// Opaque identity used internally to bind live snapshots to a single session instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct SessionOwner(u64);

impl SessionOwner {
	#[must_use]
	pub(crate) fn fresh() -> Self {
		static NEXT_OWNER: AtomicU64 = AtomicU64::new(1);

		Self(NEXT_OWNER.fetch_add(1, Ordering::Relaxed))
	}
}
