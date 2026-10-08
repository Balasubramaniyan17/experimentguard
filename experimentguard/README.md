# ExperimentGuard

Keep your existing experimentation tools. ExperimentGuard tells you whether an A/B test's results are
trustworthy, what broke, and how to reproduce the evidence.

This is the MVP: one workflow. Give it assignment and event data, get a diagnostic report.

## Quick start

```bash
pip install -e .                      # needs pandas, numpy, scipy

experimentguard demo --scenario exposure_loss --out demo_data
experimentguard diagnose --assignments demo_data/assignments.csv \
                         --events demo_data/events.csv --out report.html
```

Exit codes: `0` trustworthy, `1` suspect, `2` unreliable, `3` bad input. That makes it usable as a CI gate.

Without installing: `python -m experimentguard diagnose ...`

## Input

| File | Required columns | Optional |
|---|---|---|
| assignments.csv | `user_id`, `variant`, `assigned_at` | `device`, `app_version`, `channel` (enable segment localization) |
| events.csv | `user_id`, `event_type`, `timestamp` | `event_id` (exact duplicate detection), `variant` (exposure/assignment reconciliation) |

Exposure events are rows where `event_type == "exposure"` (change with `--exposure-event`).
Use `--split 90/10` for uneven allocations (sorted-variant order) and `--control NAME` to pick the baseline.

## What it checks

| Check | Finds | Localizes |
|---|---|---|
| Allocation balance | Sample ratio mismatch (chi-square, p < 0.001) | Onset day, segment |
| Assignment-to-exposure reconciliation | Assigned users with no exposure event, differing by variant | Onset day, device / app version / channel |
| Identity consistency | Users in several variants, exposure under another variant, orphan or pre-assignment exposures | Assigned vs. exposed matrix |
| Duplicate events | Duplicate rate differing by variant | Onset day |
| Event volume | Days or variants with a logging collapse | Day, variant |

Every finding includes the numbers, a recommendation, and a SQL query (tables `assignments`, `events`)
that reproduces it. All verdicts come from statistical tests; no LLM is involved. An LLM layer for
explaining findings can be added on top, but it should only narrate the evidence, never decide validity.

## How localization works

- **When:** a single change-point scan over per-cohort-day statistics, flagged only when the before/after shift is |z| >= 5.
- **Where:** per-segment exposure gaps (and SRM) across device, app version, channel and device x app version, with Bonferroni correction and a requirement that the segment explains at least 10% of the total gap.

## Known limits

- It reports data-quality problems. It does not check the statistical validity of your metric analysis (peeking, multiple metrics, novelty effects).
- One change-point per check. Several simultaneous faults are reported, but the onset reflects the dominant one.
- Segment localization needs the segment columns in the assignment table.
- Tested on synthetic faults only (`tests/`); it has not been validated on real customer data yet.
- Exposure is measured per assignment cohort, which assumes exposure normally follows assignment closely.

## Tests

```bash
python -m unittest discover -s tests -v
```
