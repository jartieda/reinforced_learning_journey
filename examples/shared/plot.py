"""Plot training curves from a skrl TensorBoard run directory.

skrl writes TensorBoard logs automatically when *experiment_directory* is set
in the PPO config.  This script reads those logs and saves a PNG summary that
you can share or inspect without running TensorBoard.

Usage
-----
After any training run:

    python -m examples.shared.plot runs/02_ppo_classic_control/02_ppo_classic_control

Or to scan a whole runs/ tree and plot every sub-directory:

    python -m examples.shared.plot runs/ --all

For a live interactive view during or after training:

    tensorboard --logdir runs/
    # then open http://localhost:6006 in a browser

The PNG is saved as  <run_dir>/training_curves.png  next to the event files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt


# ── TensorBoard log reader ────────────────────────────────────────────────────

def _load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} from TensorBoard event files."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        sys.exit(
            "tensorboard package not found.\n"
            "Install it with:  pip install tensorboard"
        )

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return {}
    return {tag: [(e.step, e.value) for e in ea.Scalars(tag)] for tag in tags}


# ── Core plotter ─────────────────────────────────────────────────────────────

def plot_run(run_dir: str | Path, show: bool = False) -> Path | None:
    """Generate training-curve plots for one run directory.

    Parameters
    ----------
    run_dir:
        Directory that contains TensorBoard event files.
    show:
        If True, try to open an interactive matplotlib window after saving.

    Returns
    -------
    Path of the saved PNG, or None if no data was found.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        print(f"[plot] Directory not found: {run_dir}", file=sys.stderr)
        return None

    scalars = _load_scalars(run_dir)
    if not scalars:
        print(f"[plot] No TensorBoard scalars in {run_dir}  (training still running?)")
        return None

    # ── Group tags into panels ───────────────────────────────────────────────
    reward_tags = sorted(t for t in scalars if "reward" in t.lower())
    loss_tags   = sorted(t for t in scalars if "loss"   in t.lower())
    other_tags  = sorted(t for t in scalars if t not in reward_tags + loss_tags)

    groups = [(name, tags) for name, tags in [
        ("Reward", reward_tags), ("Loss", loss_tags), ("Other", other_tags)
    ] if tags]

    if not groups:
        print("[plot] No plottable scalars found.")
        return None

    fig, axes = plt.subplots(1, len(groups), figsize=(6 * len(groups), 4), squeeze=False)
    fig.suptitle(run_dir.name, fontsize=11, fontweight="bold")

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for col, (group_name, tags) in enumerate(groups):
        ax = axes[0][col]
        for i, tag in enumerate(tags):
            data   = scalars[tag]
            steps  = [d[0] for d in data]
            values = [d[1] for d in data]
            label  = tag.split("/")[-1]           # strip parent prefix
            ax.plot(steps, values, label=label, linewidth=1.4,
                    color=colours[i % len(colours)])

        ax.set_xlabel("Timestep")
        ax.set_title(group_name)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out = run_dir / "training_curves.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved: {out}")

    if show:
        try:
            plt.show()
        except Exception:
            pass   # headless — silently skip

    plt.close(fig)
    return out


# ── CLI entry-point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot skrl training curves from a TensorBoard run directory."
    )
    parser.add_argument("run_dir", help="Run directory (or parent dir with --all).")
    parser.add_argument("--all", action="store_true",
                        help="Recursively plot every sub-directory that contains "
                             "TensorBoard event files.")
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive matplotlib window after saving.")
    args = parser.parse_args()

    # Use non-interactive backend unless the user explicitly wants a window.
    matplotlib.use("TkAgg" if args.show else "Agg")

    root = Path(args.run_dir)

    if args.all:
        # Walk the directory tree and plot any folder that has event files.
        plotted = 0
        for event_file in sorted(root.rglob("events.out.tfevents.*")):
            plot_run(event_file.parent, show=args.show)
            plotted += 1
        if plotted == 0:
            print(f"No TensorBoard event files found under {root}")
    else:
        plot_run(root, show=args.show)


if __name__ == "__main__":
    main()
