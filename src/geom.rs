//! Half-open integer geometry primitives used by the tiling model.
//!
//! Rectangles use arbitrary integer units and follow half-open semantics: `x..right()` on the X
//! axis and `y..bottom()` on the Y axis. Splits and directional navigation are expressed in terms
//! of [`Axis`], [`Slot`], and [`Direction`].

use serde::{Deserialize, Serialize};

/// Primary split axis for binary layout nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Axis {
	/// Horizontal extent, so a split produces left/right children.
	X,
	/// Vertical extent, so a split produces top/bottom children.
	Y,
}

impl Axis {
	/// Returns the orthogonal axis.
	#[must_use]
	pub fn toggled(self) -> Self {
		match self {
			Self::X => Self::Y,
			Self::Y => Self::X,
		}
	}
}

/// Child position inside a split node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Slot {
	/// The first child of a split.
	A,
	/// The second child of a split.
	B,
}

/// Cardinal direction used for geometry-based navigation and resize operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
	/// Negative X direction.
	Left,
	/// Positive X direction.
	Right,
	/// Negative Y direction.
	Up,
	/// Positive Y direction.
	Down,
}

/// Half-open integer rectangle.
///
/// A rectangle covers `x..x + w` horizontally and `y..y + h` vertically. The crate does not
/// assume any particular unit system; terminal cells, pixels, and abstract layout units all work
/// so long as they fit within `i32` origin arithmetic and `u32` extents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rect {
	/// Left edge of the rectangle.
	pub x: i32,
	/// Top edge of the rectangle.
	pub y: i32,
	/// Width of the rectangle in half-open integer units.
	pub w: u32,
	/// Height of the rectangle in half-open integer units.
	pub h: u32,
}

impl Rect {
	/// Returns the inclusive left boundary.
	#[must_use]
	pub fn left(self) -> i32 {
		self.x
	}

	/// Returns the exclusive right boundary.
	///
	/// # Panics
	///
	/// Panics if `self.w` does not fit in `i32` for boundary arithmetic.
	#[must_use]
	pub fn right(self) -> i32 {
		self.x + i32::try_from(self.w).expect("rect width exceeds i32")
	}

	/// Returns the inclusive top boundary.
	#[must_use]
	pub fn top(self) -> i32 {
		self.y
	}

	/// Returns the exclusive bottom boundary.
	///
	/// # Panics
	///
	/// Panics if `self.h` does not fit in `i32` for boundary arithmetic.
	#[must_use]
	pub fn bottom(self) -> i32 {
		self.y + i32::try_from(self.h).expect("rect height exceeds i32")
	}

	/// Returns the width or height of the rectangle along `axis`.
	#[must_use]
	pub fn extent(self, axis: Axis) -> u32 {
		match axis {
			Axis::X => self.w,
			Axis::Y => self.h,
		}
	}

	/// Splits the rectangle into a leading and trailing half-open rectangle along `axis`.
	///
	/// `lead_extent` becomes the width or height of the first returned rectangle. If
	/// `lead_extent` exceeds the source extent, the trailing rectangle saturates to zero size on
	/// that axis and the leading rectangle no longer represents an exact partition of the source.
	/// Callers that require an exact partition should pass `lead_extent <= self.extent(axis)`.
	///
	/// # Panics
	///
	/// Panics if `lead_extent` does not fit in `i32` for origin arithmetic.
	#[must_use]
	pub fn split(self, axis: Axis, lead_extent: u32) -> (Self, Self) {
		match axis {
			Axis::X => {
				let right_x = self.x + i32::try_from(lead_extent).expect("lead extent exceeds i32");
				(
					Self {
						x: self.x,
						y: self.y,
						w: lead_extent,
						h: self.h,
					},
					Self {
						x: right_x,
						y: self.y,
						w: self.w.saturating_sub(lead_extent),
						h: self.h,
					},
				)
			}
			Axis::Y => {
				let bottom_y = self.y + i32::try_from(lead_extent).expect("lead extent exceeds i32");
				(
					Self {
						x: self.x,
						y: self.y,
						w: self.w,
						h: lead_extent,
					},
					Self {
						x: self.x,
						y: bottom_y,
						w: self.w,
						h: self.h.saturating_sub(lead_extent),
					},
				)
			}
		}
	}

	/// Mirrors the rectangle across the corresponding axis of `root`.
	///
	/// This preserves the rectangle extent and reflects its offset inside `root`.
	///
	/// # Panics
	///
	/// Panics if computing `root.right()` or `root.bottom()` requires extents that do not fit in
	/// `i32`.
	#[must_use]
	pub fn mirrored(self, axis: Axis, root: Self) -> Self {
		match axis {
			Axis::X => Self {
				x: root.right() - i32::try_from(self.w).expect("rect width exceeds i32") - (self.x - root.left()),
				y: self.y,
				w: self.w,
				h: self.h,
			},
			Axis::Y => Self {
				x: self.x,
				y: root.bottom() - i32::try_from(self.h).expect("rect height exceeds i32") - (self.y - root.top()),
				w: self.w,
				h: self.h,
			},
		}
	}

	/// Returns twice the orthogonal center coordinate.
	///
	/// The doubled form preserves exact integer ordering without introducing fractions for odd
	/// extents.
	#[must_use]
	pub fn center_twice_orth(self, axis: Axis) -> i64 {
		match axis {
			Axis::X => i64::from(self.top()) + i64::from(self.bottom()),
			Axis::Y => i64::from(self.left()) + i64::from(self.right()),
		}
	}
}

/// Returns the orthogonal gap between two half-open intervals.
///
/// Overlapping or touching intervals have gap `0`.
#[must_use]
pub fn orth_gap(a_start: i32, a_end: i32, b_start: i32, b_end: i32) -> u32 {
	if a_end <= b_start {
		u32::try_from(b_start - a_end).expect("gap negative")
	} else if b_end <= a_start {
		u32::try_from(a_start - b_end).expect("gap negative")
	} else {
		0
	}
}
