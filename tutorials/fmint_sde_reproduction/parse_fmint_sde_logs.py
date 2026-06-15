#!/usr/bin/env python3
"""Parse FMint-SDE analysis logs into CSV and Markdown summary tables."""

from __future__ import annotations

import argparse
import ast
import csv
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple


RowKey = Tuple[str, str, str, str, str]


FIELDNAMES = [
    "log_file",
    "equation",
    "source",
    "demo_num",
    "scope",
    "mae",
    "rmse",
    "amd",
    "mad",
    "rel_err",
    "seconds",
]


def clean_literal(text: str) -> str:
    text = re.sub(r"np\.float(?:16|32|64)\(([^()]*)\)", r"\1", text)
    text = re.sub(r"jnp\.float(?:16|32|64)\(([^()]*)\)", r"\1", text)
    text = re.sub(r"array\((\[[^\)]*\])(?:,\s*dtype=[^\)]*)?\)", r"\1", text)
    return text


def parse_dict(text: str) -> Dict[str, Any]:
    try:
        parsed = ast.literal_eval(clean_literal(text))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse metric dictionary: {text}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected dictionary, got {type(parsed).__name__}: {text}")
    return parsed


def first_float(value: Any, index: int = 0) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if index >= len(value):
            return None
        return first_float(value[index])
    return None


def update_row(
    rows: MutableMapping[RowKey, Dict[str, str]],
    key: RowKey,
    **metrics: Optional[float],
) -> None:
    row = rows.setdefault(
        key,
        {
            "log_file": key[0],
            "equation": key[1],
            "source": key[2],
            "demo_num": key[3],
            "scope": key[4],
            "mae": "",
            "rmse": "",
            "amd": "",
            "mad": "",
            "rel_err": "",
            "seconds": "",
        },
    )
    for name, value in metrics.items():
        if value is not None:
            row[name] = f"{float(value):.10g}"


def parse_metric_dict_pair(
    rows: MutableMapping[RowKey, Dict[str, str]],
    rel_log: str,
    source: str,
    scope: str,
    left_name: str,
    left: Dict[str, Any],
    right_name: str,
    right: Dict[str, Any],
) -> None:
    equations = sorted(set(left) | set(right))
    for equation in equations:
        update_row(
            rows,
            (rel_log, equation, source, "", scope),
            **{
                left_name: first_float(left.get(equation)),
                right_name: first_float(right.get(equation)),
            },
        )


def parse_log_file(path: Path, root: Path) -> List[Dict[str, str]]:
    rel_log = str(path.relative_to(root))
    rows: Dict[RowKey, Dict[str, str]] = {}

    mae_re = re.compile(r"\bMAE:\s*(\{.*\}),\s*RMSE:\s*(\{.*\})(?:\s*\(|$)")
    coarse_mae_re = re.compile(r"MAE,\s*RMSE\s*for\s*coarse\s*solu(?:tion|ion):\s*(\{.*\})")
    strong_re = re.compile(r"^strong:\s*(\{.*\}),\s*weak:\s*(\{.*\})")
    coarse_strong_re = re.compile(r"^strong\s+for\s+coarse\s+u:\s*(\{.*\}),\s*weak\s+for\s+coarse\s+u:\s*(\{.*\})")
    demo_re = re.compile(
        r"equation=([^,]+),\s*num_demos=(\d+):\s*"
        r"MAE=([-+0-9.eE]+),\s*RMSE=([-+0-9.eE]+)"
        r"(?:,\s*strong=([-+0-9.eE]+),\s*weak=([-+0-9.eE]+)|,\s*rel_err=([-+0-9.eE]+))?"
    )
    total_time_re = re.compile(r"(ANALYSIS|ROLLOUT|DATA GENERATION)\s+TOTAL.*?:\s*([-+0-9.eE]+)\s+seconds")
    batch_time_re = re.compile(r"(Analysis|Rollout)\s+inference\s+time.*?:\s*([-+0-9.eE]+)\s+seconds")

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = mae_re.search(line)
        if match and not line.startswith("Errors without correction"):
            try:
                mae = parse_dict(match.group(1))
                rmse = parse_dict(match.group(2))
            except ValueError:
                pass
            else:
                parse_metric_dict_pair(rows, rel_log, "fmint", "summary", "mae", mae, "rmse", rmse)
            continue

        match = coarse_mae_re.search(line)
        if match:
            try:
                coarse = parse_dict(match.group(1))
            except ValueError:
                pass
            else:
                for equation, values in coarse.items():
                    update_row(
                        rows,
                        (rel_log, equation, "coarse", "", "summary"),
                        mae=first_float(values, 0),
                        rmse=first_float(values, 1),
                    )
            continue

        match = strong_re.search(line)
        if match:
            try:
                strong = parse_dict(match.group(1))
                weak = parse_dict(match.group(2))
            except ValueError:
                pass
            else:
                parse_metric_dict_pair(rows, rel_log, "fmint", "summary", "amd", strong, "mad", weak)
            continue

        match = coarse_strong_re.search(line)
        if match:
            try:
                strong = parse_dict(match.group(1))
                weak = parse_dict(match.group(2))
            except ValueError:
                pass
            else:
                parse_metric_dict_pair(rows, rel_log, "coarse", "summary", "amd", strong, "mad", weak)
            continue

        match = demo_re.search(line)
        if match:
            equation = match.group(1).strip()
            demo_num = match.group(2)
            update_row(
                rows,
                (rel_log, equation, "fmint", demo_num, "demo_sweep"),
                mae=float(match.group(3)),
                rmse=float(match.group(4)),
                amd=float(match.group(5)) if match.group(5) else None,
                mad=float(match.group(6)) if match.group(6) else None,
                rel_err=float(match.group(7)) if match.group(7) else None,
            )
            continue

        match = total_time_re.search(line)
        if match:
            update_row(
                rows,
                (rel_log, "all", "timing", "", match.group(1).lower().replace(" ", "_")),
                seconds=float(match.group(2)),
            )
            continue

        match = batch_time_re.search(line)
        if match:
            update_row(
                rows,
                (rel_log, "batch", "timing", "", match.group(1).lower()),
                seconds=float(match.group(2)),
            )

    return [rows[key] for key in sorted(rows)]


def discover_logs(log_root: Path) -> Iterable[Path]:
    if log_root.is_file():
        yield log_root
        return
    for path in sorted(log_root.rglob("*.log")):
        if path.is_file():
            yield path


def read_baseline_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        out = []
        for row in reader:
            normalized = {name: row.get(name, "") for name in FIELDNAMES}
            normalized["source"] = normalized["source"] or "baseline"
            normalized["scope"] = normalized["scope"] or "external"
            normalized["log_file"] = normalized["log_file"] or str(path)
            out.append(normalized)
        return out


def write_csv(rows: List[Dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows: List[Dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = ["equation", "source", "demo_num", "scope", "mae", "rmse", "amd", "mad", "seconds", "log_file"]
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in columns) + " |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", required=True, type=Path, help="Log file or directory tree containing .log files.")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--baseline-csv", type=Path, help="Optional external baseline rows to append.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_root = args.log_root.resolve()
    rows: List[Dict[str, str]] = []
    for log_path in discover_logs(log_root):
        root = log_root.parent if log_root.is_file() else log_root
        rows.extend(parse_log_file(log_path.resolve(), root.resolve()))
    if args.baseline_csv:
        rows.extend(read_baseline_csv(args.baseline_csv))
    rows.sort(key=lambda row: (row["equation"], row["source"], row["demo_num"], row["scope"], row["log_file"]))
    write_csv(rows, args.output_csv)
    if args.output_md:
        write_markdown(rows, args.output_md)
    print(f"Parsed {len(rows)} row(s) into {args.output_csv}")


if __name__ == "__main__":
    main()
