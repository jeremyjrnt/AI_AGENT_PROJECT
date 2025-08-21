#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
⚠️ IMPORTANT NOTES ⚠️

- This report of tokens is only an ESTIMATION because some usage in the
  very first days was not logged.

- The total includes both embedding tokens and 4o tokens.

- Token logging inside the main project code was later removed entirely,
  since the API key will not be used anymore after project submission.

---------------------------------------------------------------
This script reads key_usage.json and writes two summary files into
a folder called "tokens_count" (next to the JSON file):
  - usage_report.txt  (human-readable recap)
  - total_tokens.txt  (raw integer total)
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Union
from datetime import datetime

def load_sessions(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []
    if isinstance(data, list):
        return [s for s in data if isinstance(s, dict)]
    if isinstance(data, dict):
        sessions = data.get("sessions", [])
        return [s for s in sessions if isinstance(s, dict)]
    return []

def safe_int(x: Any) -> int:
    try:
        return int(x)
    except Exception:
        return 0

def summarize_total_tokens(sessions: List[Dict[str, Any]]) -> int:
    return sum(safe_int(s.get("estimated_tokens", 0)) for s in sessions)

def detect_span(sessions: List[Dict[str, Any]]) -> Union[str, None]:
    def parse_iso(x: Any):
        try:
            return datetime.fromisoformat(str(x))
        except Exception:
            return None
    stamps: List[datetime] = []
    for s in sessions:
        ts = s.get("session_timestamp") or s.get("timestamp")
        dt = parse_iso(ts)
        if dt:
            stamps.append(dt)
    if not stamps:
        return None
    start = min(stamps).date().isoformat()
    end = max(stamps).date().isoformat()
    return f"{start} → {end}"

def main():
    in_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("key_usage.json")
    sessions = load_sessions(in_path)
    total_tokens = summarize_total_tokens(sessions)

    # Create tokens_count folder beside the input file
    out_dir = in_path.parent / "tokens_count"
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "usage_report.txt"
    total_path = out_dir / "total_tokens.txt"

    span = detect_span(sessions)
    lines = [
        "Usage Report — Estimated Tokens (All Time)",
        "=" * 48,
        f"Source file: {in_path.resolve()}",
        f"Sessions found: {len(sessions)}",
    ]
    if span:
        lines.append(f"Date span: {span}")
    lines.append("")
    lines.append(f"Total estimated tokens: {total_tokens:,}")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    total_path.write_text(str(total_tokens), encoding="utf-8")

    print(f"✓ Wrote {report_path}")
    print(f"✓ Wrote {total_path}")

if __name__ == "__main__":
    main()
