# glorp_tiles

`glorp_tiles` is a deterministic binary split tiling library for editor and TUI layouts.

- single-root binary split trees
- exact half-open integer rectangles
- pure best-effort solver with strict feasibility certification
- hard leaf min/max constraints and shrink priorities
- snapshot-gated geometry operations
- structural edits, presets, navigation, and edge-eligible resize

## Crate model

The library is intentionally split into two layers.

- Core: topology, metadata, validation, summaries, and solving
- Session: focus, selection, geometry-driven commands, and revision tracking

Public modules are re-exported from `glorp_tiles` for ergonomic use.

## Example

```rust
use glorp_tiles::{
    Axis, Direction, LeafMeta, Rect, ResizeStrategy, Session, Slot, SolverPolicy,
};

let mut session = Session::new();
let _a = session.insert_root("main", LeafMeta::default())?;
let _b = session.split_focus(Axis::X, Slot::B, "side", LeafMeta::default(), None)?;
let _c = session.wrap_selection(Axis::Y, Slot::B, "log", LeafMeta::default(), None)?;

let root = Rect { x: 0, y: 0, w: 120, h: 40 };
let snap = session.solve(root, &SolverPolicy::default())?;
session.focus_dir(Direction::Right, &snap)?;
session.grow_focus(Direction::Down, 4, ResizeStrategy::Local, &snap)?;

let solved = session.solve(root, &SolverPolicy::default())?;
assert!(solved.strict_feasible());
# Ok::<(), Box<dyn std::error::Error>>(())
```

## Interactive demo

Run the line-oriented interactive demo with:

```bash
cargo run --example interactive
```

It demonstrates focus and selection targeting, structural edits, geometry-driven
navigation and resize, subtree presets and transforms, and a live re-solve and
re-render after each successful command.

Command groups:

- `general`: `help`, `print`, `quit|exit`, `reset`, `size <w> <h>`
- `targeting`: `focus left|right|up|down`, `select parent`, `select focus`
- `structure`: `split x|y a|b [label]`, `wrap x|y a|b [label]`, `remove`
- `resize`: `grow ...`, `shrink ...`
- `subtree`: `preset ...`, `rebalance ...`, `toggle-axis`, `mirror x|y`

Short demo transcript:

```text
help
split x b aux
focus right
select parent
preset tall a 3 2
grow left 8 chain
print
quit
```

The demo re-solves after each successful command and prints the current tree
plus solved leaf rectangles.

Behavior notes:

- subtree operations target the current `selection`, not necessarily the
  focused leaf
- geometry-driven commands solve fresh from the current root size before acting
- geometry-driven commands reject snapshots from other live `Session` instances
- free solver snapshots are inspectable, but session geometry commands require a snapshot created
  by that same live `Session`
- some leaf-targeted subtree operations are no-ops or errors according to the
  library API

## Presets and rebalancing

Included subtree rebuild presets:

- `Balanced`
- `Dwindle`
- `Tall`
- `Wide`

Included rebalance modes:

- `BinaryEqual`
- `LeafCount`

## Validation and testing

The crate ships with:

- exact allocator oracle checks
- reference-solver comparisons
- raster partition proofs
- brute-force summary envelope checks
- symmetry and roundtrip regression tests
- end-to-end session mutation, navigation, preset, and resize coverage

Run the full suite with:

```bash
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
