//! Semantic identifiers used by the public API.

use {
	serde::{Deserialize, Serialize},
	std::{
		fmt::Display,
		sync::atomic::{AtomicU64, Ordering},
	},
};

/// Strongly typed identifier for a node inside a single tree or session.
///
/// Node ids are monotonically allocated within a structure, but callers should treat them as
/// stable opaque identities rather than inferring topology from numeric ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[repr(transparent)]
#[serde(transparent)]
pub struct NodeId(u64);

impl NodeId {
	/// Creates a node id from its serialized/raw numeric form.
	#[must_use]
	pub const fn from_raw(raw: u64) -> Self {
		Self(raw)
	}

	/// Returns the serialized/raw numeric form of this id.
	#[must_use]
	pub const fn into_raw(self) -> u64 {
		self.0
	}
}

impl Display for NodeId {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		self.0.fmt(f)
	}
}

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
