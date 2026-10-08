"""experimentguard CLI.

  experimentguard demo --scenario exposure_loss --out demo/
  experimentguard diagnose --assignments demo/assignments.csv --events demo/events.csv --out report.html

Exit code: 0 trustworthy, 1 suspect, 2 unreliable (so it can gate a CI job).
"""
from __future__ import annotations

import argparse
import os
import sys

from .diagnose import diagnose
from .report import to_html, to_json, to_markdown
from .sample_data import SCENARIOS, generate

EXIT = {"TRUSTWORTHY": 0, "SUSPECT": 1, "UNRELIABLE": 2}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="experimentguard", description="Diagnose broken A/B tests.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("diagnose", help="Diagnose an experiment from assignment and event CSVs")
    d.add_argument("--assignments", required=True, help="CSV: user_id, variant, assigned_at [+ device, app_version, channel]")
    d.add_argument("--events", required=True, help="CSV: user_id, event_type, timestamp [+ event_id, variant]")
    d.add_argument("--out", help="Output file; .html, .md or .json decides the format. Omit to print Markdown.")
    d.add_argument("--split", help="Expected traffic split in sorted-variant order, e.g. 50/50 or 90/10")
    d.add_argument("--control", help="Control variant name (default: 'control' or first alphabetically)")
    d.add_argument("--exposure-event", default="exposure", help="event_type that marks exposure (default: exposure)")

    g = sub.add_parser("demo", help="Write a synthetic experiment with a known fault")
    g.add_argument("--scenario", default="exposure_loss", choices=SCENARIOS)
    g.add_argument("--out", default="demo_data")
    g.add_argument("--users", type=int, default=40_000)
    g.add_argument("--seed", type=int, default=7)

    args = ap.parse_args(argv)

    if args.cmd == "demo":
        a, e = generate(args.scenario, n_users=args.users, seed=args.seed)
        os.makedirs(args.out, exist_ok=True)
        a.to_csv(os.path.join(args.out, "assignments.csv"), index=False)
        e.to_csv(os.path.join(args.out, "events.csv"), index=False)
        print(f"Wrote {len(a):,} assignments and {len(e):,} events to {args.out}/ (scenario: {args.scenario})")
        return 0

    try:
        result = diagnose(args.assignments, args.events, split=args.split, control=args.control,
                          exposure_event=args.exposure_event)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    ext = os.path.splitext(args.out)[1].lower() if args.out else ""
    text = to_html(result) if ext in (".html", ".htm") else to_json(result) if ext == ".json" else to_markdown(result)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"{result['verdict']}: {result['headline']}\nReport written to {args.out}")
    else:
        print(text)
    return EXIT[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
