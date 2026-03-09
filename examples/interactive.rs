// Line-oriented interactive demo over `Session<String>`.

use std::error::Error;
use std::io::{self, Write};

use libtiler::{
    Axis, BalancedPreset, Direction, DwindlePreset, LeafMeta, OpError, PresetKind, RebalanceMode,
    Rect, ResizeStrategy, Session, Slot, Snapshot, SolverPolicy, TallPreset, WeightPair,
    WidePreset,
};

fn main() -> Result<(), Box<dyn Error>> {
    let mut demo = reset_demo()?;

    print_help();
    demo.rerender()?;
    repl_loop(&mut demo)?;
    Ok(())
}

struct Demo {
    session: Session<String>,
    root: Rect,
    next_label: u32,
}

impl Demo {
    fn snapshot(&self) -> Result<Snapshot, String> {
        self.session
            .try_solve(self.root, &SolverPolicy::default())
            .map_err(|error| error.to_string())
    }

    fn fresh_label(&mut self) -> String {
        let label = format!("leaf-{}", self.next_label);
        self.next_label += 1;
        label
    }

    fn rerender(&self) -> Result<(), Box<dyn Error>> {
        let snapshot = self
            .session
            .try_solve(self.root, &SolverPolicy::default())?;
        let focus = self.session.focus();
        let root_id = self.session.tree().root_id();

        println!();
        println!(
            "status size={}x{} revision={} strict={} violations={} root={} nodes={} focus={} selection={}",
            self.root.w,
            self.root.h,
            self.session.revision(),
            if snapshot.strict_feasible {
                "yes"
            } else {
                "no"
            },
            snapshot.violations.len(),
            fmt_id(root_id),
            self.session.tree().node_ids().len(),
            fmt_id(focus),
            fmt_selection(&self.session),
        );
        println!();
        println!("tree:");
        if let Some(root_id) = root_id {
            print_tree(&self.session, root_id, "", None, true);
        } else {
            println!("<empty>");
        }
        println!();
        println!("leaf rects:");
        println!(
            "{:<4} {:<16} {:>4} {:>4} {:>4} {:>4}",
            "id", "label", "x", "y", "w", "h"
        );
        if let Some(root_id) = root_id {
            for id in self.session.tree().leaf_ids_dfs(root_id) {
                let leaf = self
                    .session
                    .tree()
                    .leaf(id)
                    .expect("leaf_ids_dfs should only return leaves");
                let rect = snapshot
                    .rect(id)
                    .expect("snapshot should contain all solved leaf rects");
                println!(
                    "{:<4} {:<16} {:>4} {:>4} {:>4} {:>4}",
                    id,
                    leaf.payload(),
                    rect.x,
                    rect.y,
                    rect.w,
                    rect.h,
                );
            }
        }

        if !snapshot.violations.is_empty() {
            println!();
            println!("violations:");
            for violation in &snapshot.violations {
                println!(
                    "- node={} kind={:?} required={} actual={}",
                    violation.node, violation.kind, violation.required, violation.actual
                );
            }
        }

        println!();
        Ok(())
    }
}

fn reset_demo() -> Result<Demo, OpError> {
    let mut session = Session::new();
    let _main = session.insert_root(String::from("main"), LeafMeta::default())?;
    let _side = session.split_focus(
        Axis::X,
        Slot::B,
        String::from("side"),
        LeafMeta::default(),
        None,
    )?;
    let _log = session.wrap_selection(
        Axis::Y,
        Slot::B,
        String::from("log"),
        LeafMeta::default(),
        None,
    )?;

    Ok(Demo {
        session,
        root: Rect {
            x: 0,
            y: 0,
            w: 120,
            h: 40,
        },
        next_label: 1,
    })
}

fn repl_loop(demo: &mut Demo) -> Result<(), Box<dyn Error>> {
    let stdin = io::stdin();
    let mut line = String::new();

    loop {
        print!("libtiler> ");
        io::stdout().flush()?;

        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            println!();
            break;
        }

        let input = line.trim();
        if input.is_empty() {
            continue;
        }

        match dispatch_command(input, demo) {
            Ok(CommandOutcome::Quit) => break,
            Ok(CommandOutcome::Continue { rerender }) => {
                if rerender {
                    demo.rerender()?;
                }
            }
            Err(error) => eprintln!("error: {error}"),
        }
    }

    Ok(())
}

fn print_tree(
    session: &Session<String>,
    id: libtiler::NodeId,
    prefix: &str,
    branch: Option<char>,
    last: bool,
) {
    let marker = match branch {
        Some(slot) if last => format!("\\-{slot} "),
        Some(slot) => format!("|-{slot} "),
        None => String::new(),
    };

    if let Some(split) = session.tree().split(id) {
        println!(
            "{prefix}{marker}split#{id} axis={:?} weights={}:{}{}",
            split.axis(),
            split.weights().a,
            split.weights().b,
            annotations(session, id),
        );
        let child_prefix = match branch {
            Some(_) if last => format!("{prefix}   "),
            Some(_) => format!("{prefix}|  "),
            None => prefix.to_owned(),
        };
        print_tree(session, split.a(), &child_prefix, Some('A'), false);
        print_tree(session, split.b(), &child_prefix, Some('B'), true);
    } else if let Some(leaf) = session.tree().leaf(id) {
        println!(
            "{prefix}{marker}leaf#{id} {:?}{}",
            leaf.payload(),
            annotations(session, id),
        );
    }
}

fn annotations(session: &Session<String>, id: libtiler::NodeId) -> String {
    let mut tags = Vec::new();
    if session.focus() == Some(id) {
        tags.push("focus");
    }
    if session.selection() == Some(id) {
        tags.push("sel");
    }
    if tags.is_empty() {
        String::new()
    } else {
        format!(" [{}]", tags.join("]["))
    }
}

fn fmt_id(id: Option<libtiler::NodeId>) -> String {
    id.map_or_else(|| String::from("-"), |value| value.to_string())
}

fn fmt_selection(session: &Session<String>) -> String {
    match session.selection() {
        Some(id) if session.tree().is_leaf(id) => format!("{id}(leaf)"),
        Some(id) if session.tree().is_split(id) => format!("{id}(split)"),
        Some(id) => format!("{id}(?)"),
        None => String::from("-"),
    }
}

fn print_help() {
    println!("commands:");
    println!("  general:");
    println!("    help");
    println!("    print");
    println!("    quit|exit");
    println!("    reset");
    println!("    size <w> <h>");
    println!("  targeting:");
    println!("    focus left|right|up|down");
    println!("    select parent");
    println!("    select focus");
    println!("  structure:");
    println!("    split x|y a|b [label]");
    println!("    wrap x|y a|b [label]");
    println!("    remove");
    println!("  resize:");
    println!("    grow left|right|up|down <amount> local|chain|slack");
    println!("    shrink left|right|up|down <amount> local|chain|slack");
    println!("  subtree:");
    println!("    preset balanced x|y alt|same");
    println!("    preset dwindle x|y a|b");
    println!("    preset tall a|b [wa wb]");
    println!("    preset wide a|b [wa wb]");
    println!("    rebalance equal|count");
    println!("    toggle-axis");
    println!("    mirror x|y");
    println!();
}

fn dispatch_command(input: &str, demo: &mut Demo) -> Result<CommandOutcome, String> {
    let parts = input.split_whitespace().collect::<Vec<_>>();
    match parts.as_slice() {
        ["help"] => {
            print_help();
            Ok(CommandOutcome::Continue { rerender: false })
        }
        ["print"] => Ok(CommandOutcome::Continue { rerender: true }),
        ["quit"] | ["exit"] => Ok(CommandOutcome::Quit),
        ["reset"] => {
            *demo = reset_demo().map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["size", width, height] => {
            demo.root.w = parse_u32(width, "width")?;
            demo.root.h = parse_u32(height, "height")?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["focus", direction] => {
            let direction = parse_direction(direction)?;
            let snapshot = demo.snapshot()?;
            demo.session
                .focus_dir(direction, &snapshot)
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["select", "parent"] => {
            demo.session
                .select_parent()
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["select", "focus"] => {
            demo.session.select_focus();
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["split", axis, slot] => {
            let axis = parse_axis(axis)?;
            let slot = parse_slot(slot)?;
            let label = demo.fresh_label();
            demo.session
                .split_focus(axis, slot, label, LeafMeta::default(), None)
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["split", axis, slot, label] => {
            let axis = parse_axis(axis)?;
            let slot = parse_slot(slot)?;
            demo.session
                .split_focus(axis, slot, (*label).to_owned(), LeafMeta::default(), None)
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["wrap", axis, slot] => {
            let axis = parse_axis(axis)?;
            let slot = parse_slot(slot)?;
            let label = demo.fresh_label();
            demo.session
                .wrap_selection(axis, slot, label, LeafMeta::default(), None)
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["wrap", axis, slot, label] => {
            let axis = parse_axis(axis)?;
            let slot = parse_slot(slot)?;
            demo.session
                .wrap_selection(axis, slot, (*label).to_owned(), LeafMeta::default(), None)
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["remove"] => {
            demo.session
                .remove_focus()
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["grow", direction, amount, strategy] => {
            let direction = parse_direction(direction)?;
            let amount = parse_u32(amount, "amount")?;
            let strategy = parse_resize_strategy(strategy)?;
            let snapshot = demo.snapshot()?;
            demo.session
                .grow_focus(direction, amount, strategy, &snapshot)
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["shrink", direction, amount, strategy] => {
            let direction = parse_direction(direction)?;
            let amount = parse_u32(amount, "amount")?;
            let strategy = parse_resize_strategy(strategy)?;
            let snapshot = demo.snapshot()?;
            demo.session
                .shrink_focus(direction, amount, strategy, &snapshot)
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["rebalance", mode] => {
            let mode = parse_rebalance_mode(mode)?;
            demo.session
                .rebalance_selection(mode)
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["toggle-axis"] => {
            demo.session
                .toggle_axis()
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["mirror", axis] => {
            let axis = parse_axis(axis)?;
            demo.session
                .mirror_selection(axis)
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["preset", "balanced", axis, alternation] => {
            let axis = parse_axis(axis)?;
            let alternate = parse_alternation(alternation)?;
            demo.session
                .apply_preset(PresetKind::Balanced(BalancedPreset {
                    start_axis: axis,
                    alternate,
                }))
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["preset", "dwindle", axis, slot] => {
            let axis = parse_axis(axis)?;
            let slot = parse_slot(slot)?;
            demo.session
                .apply_preset(PresetKind::Dwindle(DwindlePreset {
                    start_axis: axis,
                    new_leaf_slot: slot,
                }))
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["preset", "tall", slot] => {
            let slot = parse_slot(slot)?;
            demo.session
                .apply_preset(PresetKind::Tall(TallPreset {
                    master_slot: slot,
                    root_weights: WeightPair::default(),
                }))
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["preset", "tall", slot, weight_a, weight_b] => {
            let slot = parse_slot(slot)?;
            let weights = parse_weight_pair(weight_a, weight_b)?;
            demo.session
                .apply_preset(PresetKind::Tall(TallPreset {
                    master_slot: slot,
                    root_weights: weights,
                }))
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["preset", "wide", slot] => {
            let slot = parse_slot(slot)?;
            demo.session
                .apply_preset(PresetKind::Wide(WidePreset {
                    master_slot: slot,
                    root_weights: WeightPair::default(),
                }))
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        ["preset", "wide", slot, weight_a, weight_b] => {
            let slot = parse_slot(slot)?;
            let weights = parse_weight_pair(weight_a, weight_b)?;
            demo.session
                .apply_preset(PresetKind::Wide(WidePreset {
                    master_slot: slot,
                    root_weights: weights,
                }))
                .map_err(|error| error.to_string())?;
            Ok(CommandOutcome::Continue { rerender: true })
        }
        _ => Err(format!("unknown command: {input}")),
    }
}

fn parse_u32(value: &str, name: &str) -> Result<u32, String> {
    value
        .parse::<u32>()
        .map_err(|error| format!("invalid {name} {value:?}: {error}"))
}

fn parse_direction(value: &str) -> Result<Direction, String> {
    match value {
        "left" => Ok(Direction::Left),
        "right" => Ok(Direction::Right),
        "up" => Ok(Direction::Up),
        "down" => Ok(Direction::Down),
        _ => Err(format!("unknown direction: {value}")),
    }
}

fn parse_axis(value: &str) -> Result<Axis, String> {
    match value {
        "x" => Ok(Axis::X),
        "y" => Ok(Axis::Y),
        _ => Err(format!("unknown axis: {value}")),
    }
}

fn parse_slot(value: &str) -> Result<Slot, String> {
    match value {
        "a" => Ok(Slot::A),
        "b" => Ok(Slot::B),
        _ => Err(format!("unknown slot: {value}")),
    }
}

fn parse_resize_strategy(value: &str) -> Result<ResizeStrategy, String> {
    match value {
        "local" => Ok(ResizeStrategy::Local),
        "chain" => Ok(ResizeStrategy::AncestorChain),
        "slack" => Ok(ResizeStrategy::DistributedBySlack),
        _ => Err(format!("unknown resize strategy: {value}")),
    }
}

fn parse_alternation(value: &str) -> Result<bool, String> {
    match value {
        "alt" => Ok(true),
        "same" => Ok(false),
        _ => Err(format!("unknown alternation mode: {value}")),
    }
}

fn parse_rebalance_mode(value: &str) -> Result<RebalanceMode, String> {
    match value {
        "equal" => Ok(RebalanceMode::BinaryEqual),
        "count" => Ok(RebalanceMode::LeafCount),
        _ => Err(format!("unknown rebalance mode: {value}")),
    }
}

fn parse_weight_pair(weight_a: &str, weight_b: &str) -> Result<WeightPair, String> {
    Ok(WeightPair {
        a: parse_u32(weight_a, "weight a")?,
        b: parse_u32(weight_b, "weight b")?,
    })
}

enum CommandOutcome {
    Continue { rerender: bool },
    Quit,
}
