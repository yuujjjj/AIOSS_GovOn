#!/usr/bin/env python3
"""Generate an M4 Markdown report from the local GovOn experiment database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.ab_testing import ExperimentStore, render_markdown_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "docs" / "outputs" / "M4_Testing" / "ab-test-report.md",
    )
    args = parser.parse_args()

    summary = ExperimentStore(db_path=args.db_path).summarize(days=args.days)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown_report(summary), encoding="utf-8")
    print(f"report={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
