//! Semantic identifiers used by the public API.

/// Opaque identifier for a node inside a single tree or session.
///
/// Node ids are monotonically allocated within a structure, but callers should treat them as
/// stable opaque identities rather than inferring topology from numeric ordering.
pub type NodeId = u64;

/// Monotonic session revision used to detect stale snapshots.
///
/// Geometry-based commands require a snapshot produced from the current revision.
pub type Revision = u64;
