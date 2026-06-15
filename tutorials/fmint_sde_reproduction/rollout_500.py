#!/usr/bin/env python3
"""Evaluate a 500-step FMint-SDE rollout from coarse and correction arrays."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def chunk_count(steps: int, window: int) -> int:
    if steps % window != 0:
        raise ValueError("--steps must be divisible by --window")
    return steps // window


def metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    diff = pred - target
    return {
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "amd": float(np.mean(np.max(np.abs(diff), axis=2))),
        "mad": float(np.mean(np.max(np.abs(np.mean(pred, axis=1) - np.mean(target, axis=1)), axis=-1))),
    }


def rollout_from_correction(coarse: np.ndarray, correction: np.ndarray, steps: int, window: int) -> Tuple[np.ndarray, np.ndarray]:
    chunks = chunk_count(steps, window)
    rolled = np.zeros_like(coarse[:, :steps, :])
    aligned_coarse = np.zeros_like(rolled)
    for chunk_idx in range(chunks):
        start = chunk_idx * window
        end = start + window
        if start == 0:
            offset = np.zeros((coarse.shape[0], 1, coarse.shape[2]), dtype=coarse.dtype)
        else:
            offset = rolled[:, start - 1 : start, :] - coarse[:, start - 1 : start, :]
        aligned_window = coarse[:, start:end, :] + offset
        aligned_coarse[:, start:end, :] = aligned_window
        rolled[:, start:end, :] = aligned_window + correction[:, start:end, :]
    return rolled, aligned_coarse


def load_arrays(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.check_only:
        rng = np.random.default_rng(args.seed)
        fine = rng.normal(size=(8, args.steps, 3)).astype(np.float32)
        coarse = fine + 0.2 + 0.05 * rng.normal(size=fine.shape).astype(np.float32)
        correction = fine - coarse + 0.01 * rng.normal(size=fine.shape).astype(np.float32)
        return coarse, fine, correction

    if args.input_npz is None:
        raise ValueError("Provide --input-npz or use --check-only")
    data = np.load(args.input_npz)
    coarse = np.asarray(data[args.coarse_key], dtype=np.float32)
    fine = np.asarray(data[args.fine_key], dtype=np.float32)
    if args.prediction_key and args.prediction_key in data:
        prediction = np.asarray(data[args.prediction_key], dtype=np.float32)
        correction = prediction - coarse
    elif args.correction_key and args.correction_key in data:
        correction = np.asarray(data[args.correction_key], dtype=np.float32)
    else:
        correction = np.zeros_like(coarse)
    return coarse, fine, correction


def ensure_shape(name: str, arr: np.ndarray, steps: int) -> None:
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (samples, time, dim); got {arr.shape}")
    if arr.shape[1] < steps:
        raise ValueError(f"{name} has only {arr.shape[1]} time steps, needs {steps}")


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "mae", "rmse", "amd", "mad", "chunks", "steps", "window"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path)
    parser.add_argument("--coarse-key", default="coarse")
    parser.add_argument("--fine-key", default="fine")
    parser.add_argument("--correction-key", default="correction")
    parser.add_argument("--prediction-key", default="")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--window", type=int, default=50)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    chunks = chunk_count(args.steps, args.window)
    coarse, fine, correction = load_arrays(args)
    ensure_shape("coarse", coarse, args.steps)
    ensure_shape("fine", fine, args.steps)
    ensure_shape("correction", correction, args.steps)

    coarse = coarse[:, : args.steps, :]
    fine = fine[:, : args.steps, :]
    correction = correction[:, : args.steps, :]

    rolled, aligned_coarse = rollout_from_correction(coarse, correction, args.steps, args.window)
    coarse_metrics = metrics(coarse, fine)
    aligned_coarse_metrics = metrics(aligned_coarse, fine)
    rollout_metrics = metrics(rolled, fine)

    rows = []
    for method, values in [
        ("coarse", coarse_metrics),
        ("aligned_coarse", aligned_coarse_metrics),
        ("fmint_rollout", rollout_metrics),
    ]:
        row = {
            "method": method,
            "chunks": str(chunks),
            "steps": str(args.steps),
            "window": str(args.window),
        }
        row.update({key: f"{value:.10g}" for key, value in values.items()})
        rows.append(row)

    write_csv(args.output_csv, rows)
    print(f"ROLL_OUT_WINDOWS: {chunks}")
    print(f"Wrote rollout metrics to {args.output_csv}")
    for row in rows:
        print(
            f"{row['method']}: MAE={row['mae']}, RMSE={row['rmse']}, "
            f"AMD={row['amd']}, MAD={row['mad']}"
        )


if __name__ == "__main__":
    main()

