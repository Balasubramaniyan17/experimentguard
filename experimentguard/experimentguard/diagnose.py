"""Run all checks and turn the findings into a verdict and a recommended action."""
from __future__ import annotations

from dataclasses import asdict

import pandas as pd

from .checks import ALL_CHECKS, prepare

SEVERITY_ORDER = {"critical": 0, "warning": 1, "pass": 2}


def _load(x):
    return x if isinstance(x, pd.DataFrame) else pd.read_csv(x)


def diagnose(assignments, events, *, split: str | None = None, control: str | None = None,
             exposure_event: str = "exposure") -> dict:
    """Diagnose one experiment.

    ``assignments`` / ``events`` may be DataFrames or CSV paths. Returns a JSON-serializable dict.
    """
    ds = prepare(_load(assignments), _load(events), split=split, control=control, exposure_event=exposure_event)

    findings = []
    for check in ALL_CHECKS:
        findings.extend(check(ds))
    findings.sort(key=lambda f: SEVERITY_ORDER[f.severity])

    critical = [f for f in findings if f.severity == "critical"]
    warnings = [f for f in findings if f.severity == "warning"]
    if critical:
        verdict = "UNRELIABLE"
        headline = "Do not act on this experiment's results. " + "; ".join(f.title for f in critical) + "."
    elif warnings:
        verdict = "SUSPECT"
        headline = "Results may be usable, but investigate first: " + "; ".join(f.title for f in warnings) + "."
    else:
        verdict = "TRUSTWORTHY"
        headline = "No data-quality problems detected by these checks."

    action, detail = _recommend(critical, warnings)
    counts = ds.assign["variant"].value_counts()
    return {
        "verdict": verdict,
        "headline": headline,
        "recommended_action": action,
        "action_detail": detail,
        "experiment": {
            "users_assigned": int(len(ds.assign)),
            "variants": {v: int(counts.get(v, 0)) for v in ds.variants},
            "control": ds.control,
            "first_assignment": f"{ds.assign['assigned_at'].min():%Y-%m-%d}",
            "last_assignment": f"{ds.assign['assigned_at'].max():%Y-%m-%d}",
            "events": int(len(ds.events)),
            "exposure_event": ds.exposure_event,
            "segment_dimensions": ds.dims,
        },
        "findings": [asdict(f) for f in findings],
        "needs_human_approval": action in ("repair_instrumentation", "restart_experiment"),
    }


def _recommend(critical, warnings):
    if not critical and not warnings:
        return "proceed", ("Proceed with normal analysis. Passing these checks does not prove the experiment is valid; "
                           "it means these specific failure modes were not found.")
    if critical:
        onsets = [f.evidence.get("onset") for f in critical if f.evidence.get("onset")]
        if onsets:
            return "repair_instrumentation", (
                "Repair the instrumentation behind the critical findings, then restart the experiment, or restrict "
                f"analysis to cohorts assigned before {min(onsets)}. A person should approve either change.")
        return "restart_experiment", ("The problem is present throughout the run, so no clean sub-period exists. "
                                       "Fix the root cause and restart. A person should approve the restart.")
    return "investigate", "Review the warnings below before trusting the metrics."
