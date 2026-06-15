#!/usr/bin/env python3
"""Measure fine, coarse, and FMint-SDE normalized runtime for stochastic Lorenz."""

from __future__ import annotations

import argparse
import csv
import re
import time
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


def lorenz_drift(state: np.ndarray, sigma: float = 10.0, rho: float = 28.0, beta: float = 8.0 / 3.0) -> np.ndarray:
    x = state[..., 0]
    y = state[..., 1]
    z = state[..., 2]
    return np.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], axis=-1)


def make_initial_conditions(rng: np.random.Generator, n_ic: int, n_noise: int) -> np.ndarray:
    init = np.empty((n_ic, n_noise, 3), dtype=np.float32)
    init[..., 0] = rng.uniform(-10.0, 10.0, size=(n_ic, n_noise))
    init[..., 1] = rng.uniform(-10.0, 10.0, size=(n_ic, n_noise))
    init[..., 2] = rng.uniform(5.0, 30.0, size=(n_ic, n_noise))
    return init


def simulate_fine(initial: np.ndarray, noise: np.ndarray, dt: float, eta: float) -> np.ndarray:
    state = initial.copy()
    for step in range(noise.shape[0]):
        state = state + dt * lorenz_drift(state) + eta * noise[step]
    return state


def simulate_coarse(initial: np.ndarray, fine_noise: np.ndarray, fine_dt: float, coarse_dt: float, eta: float) -> np.ndarray:
    ratio = int(round(coarse_dt / fine_dt))
    if not np.isclose(ratio * fine_dt, coarse_dt):
        raise ValueError("coarse_dt must be an integer multiple of fine_dt")
    usable = (fine_noise.shape[0] // ratio) * ratio
    coarse_noise = fine_noise[:usable].reshape(-1, ratio, *fine_noise.shape[1:]).sum(axis=1)
    state = initial.copy()
    for step in range(coarse_noise.shape[0]):
        state = state + coarse_dt * lorenz_drift(state) + eta * coarse_noise[step]
    return state


def time_call(repeats: int, fn) -> Tuple[float, np.ndarray]:
    times: List[float] = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    assert result is not None
    return float(np.median(times)), result


def analysis_logs(root: Path) -> Iterable[Path]:
    if root.is_file():
        yield root
    elif root.exists():
        yield from sorted(root.rglob("*.log"))


def parse_inference_seconds(root: Path | None) -> float:
    if root is None:
        return 0.0
    total = 0.0
    pattern = re.compile(r"(?:ANALYSIS|ROLLOUT)\s+TOTAL\s+INFERENCE\s+TIME:\s*([-+0-9.eE]+)\s+seconds")
    for path in analysis_logs(root):
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in pattern.finditer(text):
            total += float(match.group(1))
    return total


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "runtime_seconds", "normalized_to_fine", "details"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-conditions", type=int, default=25)
    parser.add_argument("--noise-realizations", type=int, default=40)
    parser.add_argument("--fine-dt", type=float, default=1e-5)
    parser.add_argument("--coarse-dt", type=float, default=1e-3)
    parser.add_argument("--coarse-steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--analysis-log-root", type=Path)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.initial_conditions = 2
        args.noise_realizations = 3
        args.coarse_steps = 5
        args.repeats = 1

    ratio = int(round(args.coarse_dt / args.fine_dt))
    if not np.isclose(ratio * args.fine_dt, args.coarse_dt):
        raise ValueError("--coarse-dt must be an integer multiple of --fine-dt")
    fine_steps = args.coarse_steps * ratio

    rng = np.random.default_rng(args.seed)
    initial = make_initial_conditions(rng, args.initial_conditions, args.noise_realizations)
    fine_noise = (
        rng.normal(size=(fine_steps, args.initial_conditions, args.noise_realizations, 3)).astype(np.float32)
        * np.sqrt(args.fine_dt)
    )
    fine_noise[0, ...] = 0.0

    fine_seconds, fine_final = time_call(
        args.repeats,
        lambda: simulate_fine(initial, fine_noise, args.fine_dt, args.eta),
    )
    coarse_seconds, coarse_final = time_call(
        args.repeats,
        lambda: simulate_coarse(initial, fine_noise, args.fine_dt, args.coarse_dt, args.eta),
    )
    correction_seconds = parse_inference_seconds(args.analysis_log_root)
    fmint_seconds = coarse_seconds + correction_seconds

    rows = [
        {
            "method": "fine",
            "runtime_seconds": f"{fine_seconds:.10g}",
            "normalized_to_fine": "1",
            "details": f"dt={args.fine_dt}, steps={fine_steps}",
        },
        {
            "method": "coarse",
            "runtime_seconds": f"{coarse_seconds:.10g}",
            "normalized_to_fine": f"{coarse_seconds / fine_seconds:.10g}",
            "details": f"dt={args.coarse_dt}, steps={args.coarse_steps}",
        },
        {
            "method": "fmint_sde",
            "runtime_seconds": f"{fmint_seconds:.10g}",
            "normalized_to_fine": f"{fmint_seconds / fine_seconds:.10g}",
            "details": f"coarse_runtime + parsed_correction_runtime={correction_seconds:.10g}",
        },
    ]
    write_csv(args.output_csv, rows)
    print(f"Fine final state mean: {float(np.mean(fine_final)):.6g}")
    print(f"Coarse final state mean: {float(np.mean(coarse_final)):.6g}")
    print(f"Wrote timing table to {args.output_csv}")
    for row in rows:
        print(f"{row['method']}: {row['runtime_seconds']} sec, normalized={row['normalized_to_fine']}")


if __name__ == "__main__":
    main()

