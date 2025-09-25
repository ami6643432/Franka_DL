#!/usr/bin/env python3
"""Plot RMS end-effector position tracking error from a saved trajectory log."""

import argparse
from pathlib import Path

import numpy as np


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load a tracking log (generated via simulate_franka_trajectory.py --save-track) and "
            "plot the root-mean-square (RMS) error between commanded and measured end-effector "
            "positions."
        )
    )
    parser.add_argument("log", type=Path, help="Path to the .npz tracking log.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If provided, save the plot to this path instead of (or as well as) showing it.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive window; useful when combined with --save in headless runs.",
    )
    return parser


def _load_positions(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path)

    if "ee_pos" not in data or "ee_pos_des" not in data:
        raise KeyError(
            "Tracking log is missing 'ee_pos' or 'ee_pos_des'. Re-run simulate_franka_trajectory.py "
            "with --save-track to capture the required fields."
        )

    actual = np.asarray(data["ee_pos"], dtype=np.float64)
    desired = np.asarray(data["ee_pos_des"], dtype=np.float64)

    if actual.ndim != 2 or desired.ndim != 2 or actual.shape[1] != 3 or desired.shape[1] != 3:
        raise ValueError("Expected end-effector positions to have shape (N, 3).")

    count = min(len(actual), len(desired))
    actual = actual[:count]
    desired = desired[:count]

    timestamps = np.asarray(data.get("time", np.arange(count)), dtype=np.float64)
    if timestamps.size < count:
        timestamps = np.arange(count)
    else:
        timestamps = timestamps[:count]

    return timestamps, actual, desired


def _compute_running_rms(actual: np.ndarray, desired: np.ndarray) -> np.ndarray:
    delta = actual - desired
    sq_error = np.sum(delta * delta, axis=1)
    cumulative = np.cumsum(sq_error)
    steps = np.arange(1, len(cumulative) + 1, dtype=np.float64)
    return np.sqrt(cumulative / steps)


def _plot_rms(timestamps: np.ndarray, rms: np.ndarray, save_path: Path | None, show: bool) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - plotting requires matplotlib
        raise RuntimeError("matplotlib is required to generate the plot.") from exc

    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(timestamps, rms, color="tab:blue", linewidth=2.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS position error (m)")
    ax.set_title("End-effector RMS position tracking error")
    ax.grid(True, linestyle="--", linewidth=0.5)

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] Saved RMS error plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    timestamps, actual, desired = _load_positions(args.log)
    rms = _compute_running_rms(actual, desired)

    # Align timestamps length with RMS array
    if timestamps.shape[0] != rms.shape[0]:
        timestamps = timestamps[: rms.shape[0]]

    show_plot = not args.no_show
    _plot_rms(timestamps, rms, args.save, show_plot)


if __name__ == "__main__":
    main()
