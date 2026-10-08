"""Statistical and data-quality checks for A/B test diagnostics.

Every check returns a list of ``Finding`` objects. A finding is either a
``critical`` problem (results should not be interpreted), a ``warning``
(investigate), or a ``pass`` (the check ran and found nothing). Every finding
carries the numbers behind it and a SQL query that reproduces them.

Expected input tables
---------------------
assignments: user_id, variant, assigned_at  [+ optional device, app_version, channel]
events:      user_id, event_type, timestamp [+ optional event_id, variant]
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

SRM_ALPHA = 0.001        # threshold commonly used for sample ratio mismatch
FAMILY_ALPHA = 0.01      # family-wise alpha for segment scans (Bonferroni)
MIN_SEGMENT_USERS = 100  # per variant, to be eligible in a segment scan
CP_Z = 5.0               # |z| needed to call a change-point (scan-adjusted, conservative)
MIN_EXPOSURE_GAP = 0.005 # 0.5 percentage points
CANDIDATE_DIMS = ("device", "app_version", "channel")


@dataclass
class Finding:
    id: str
    severity: str  # critical | warning | pass
    title: str
    summary: str
    evidence: dict = field(default_factory=dict)
    sql: str = ""
    recommendation: str = ""


@dataclass
class Dataset:
    assign: pd.DataFrame      # one row per user (first assignment) + day + exposed + dims
    raw_assign: pd.DataFrame  # as supplied (may contain several rows per user)
    events: pd.DataFrame
    exposures: pd.DataFrame
    variants: list
    control: str
    dims: list
    expected: dict            # variant -> expected share
    exposure_event: str


# --------------------------------------------------------------------------- prep

def _ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.tz_localize(None)


def parse_split(text: str | None, variants: list) -> dict:
    """'50/50' or '90/10' (in sorted-variant order) -> {variant: share}."""
    if not text:
        return {v: 1 / len(variants) for v in variants}
    parts = [float(x) for x in text.replace(":", "/").split("/")]
    if len(parts) != len(variants):
        raise ValueError(f"--split has {len(parts)} parts but the data has {len(variants)} variants: {variants}")
    total = sum(parts)
    return {v: p / total for v, p in zip(variants, parts)}


def prepare(assignments: pd.DataFrame, events: pd.DataFrame, split: str | None = None,
            control: str | None = None, exposure_event: str = "exposure") -> Dataset:
    for name, df, cols in (("assignments", assignments, ("user_id", "variant", "assigned_at")),
                           ("events", events, ("user_id", "event_type", "timestamp"))):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"{name} is missing required column(s): {', '.join(missing)}")

    raw = assignments.copy()
    raw["assigned_at"] = _ts(raw["assigned_at"])
    ev = events.copy()
    ev["timestamp"] = _ts(ev["timestamp"])

    first = raw.sort_values("assigned_at").drop_duplicates("user_id", keep="first").copy()
    first["day"] = first["assigned_at"].dt.floor("D")
    exposures = ev[ev["event_type"] == exposure_event]
    first["exposed"] = first["user_id"].isin(set(exposures["user_id"]))

    variants = sorted(first["variant"].astype(str).unique())
    if len(variants) < 2:
        raise ValueError("Need at least two variants in assignments.")
    first["variant"] = first["variant"].astype(str)
    if control is None:
        control = "control" if "control" in variants else variants[0]
    if control not in variants:
        raise ValueError(f"control '{control}' not found in variants {variants}")

    dims = [d for d in CANDIDATE_DIMS if d in first.columns]
    if "device" in dims and "app_version" in dims:
        first["device+app_version"] = first["device"].astype(str) + " / " + first["app_version"].astype(str)
        dims.append("device+app_version")

    return Dataset(first, raw, ev, exposures, variants, control, dims,
                   parse_split(split, variants), exposure_event)


# ---------------------------------------------------------------- stats helpers

def _two_prop(s1, n1, s0, n0):
    p1, p0 = s1 / n1, s0 / n0
    pp = (s1 + s0) / (n1 + n0)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    z = (p1 - p0) / se if se > 0 else 0.0
    return p1, p0, z, float(2 * stats.norm.sf(abs(z)))


def _chi2_homogeneity(successes, totals):
    table = np.array([[s, t - s] for s, t in zip(successes, totals)], dtype=float)
    if (table.sum(axis=0) == 0).any() or (table.sum(axis=1) == 0).any():
        return 0.0, 1.0
    chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
    return float(chi2), float(p)


def scan_changepoint(n, period_stat, min_seg=2):
    """Find the single split point with the largest standardized before/after shift.

    ``period_stat(lo, hi) -> (estimate, variance)`` over day indices [lo, hi).
    """
    best = None
    for k in range(min_seg, n - min_seg + 1):
        e1, v1 = period_stat(0, k)
        e2, v2 = period_stat(k, n)
        if not (np.isfinite(v1) and np.isfinite(v2)):
            continue
        z = (e2 - e1) / math.sqrt(v1 + v2 + 1e-12)
        if best is None or abs(z) > abs(best["z"]):
            best = {"k": k, "before": e1, "after": e2, "z": z}
    return best


def _day_range(ds):
    return pd.date_range(ds.assign["day"].min(), ds.assign["day"].max(), freq="D")


def _q(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _segment_sql_filter(dim: str, seg: str, alias="a") -> str:
    cols, vals = dim.split("+"), seg.split(" / ")
    return " AND ".join(f"{alias}.{c} = {_q(v)}" for c, v in zip(cols, vals))


# ------------------------------------------------------------------ 1. allocation

def srm_segments(ds: Dataset, since=None, top=5) -> list[dict]:
    a = ds.assign if since is None else ds.assign[ds.assign["day"] >= since]
    exp_share = np.array([ds.expected[v] for v in ds.variants])
    rows = []
    for dim in ds.dims:
        counts = a.groupby([dim, "variant"]).size().unstack("variant").reindex(columns=ds.variants).fillna(0)
        for seg, r in counts.iterrows():
            n = r.sum()
            if n < MIN_SEGMENT_USERS * len(ds.variants):
                continue
            chi2, p = stats.chisquare(r.values, exp_share * n)
            rows.append({"dim": dim, "segment": str(seg), "users": int(n), "chi2": float(chi2), "p": float(p),
                         "counts": {v: int(r[v]) for v in ds.variants}})
    alpha = FAMILY_ALPHA / max(1, len(rows))
    hits = sorted([r for r in rows if r["p"] < alpha], key=lambda r: -r["chi2"])
    return hits[:top]


def check_allocation(ds: Dataset) -> list[Finding]:
    counts = ds.assign["variant"].value_counts().reindex(ds.variants).fillna(0)
    n = counts.sum()
    expected = np.array([ds.expected[v] for v in ds.variants]) * n
    chi2, p = stats.chisquare(counts.values, expected)
    base = {"counts": {v: int(counts[v]) for v in ds.variants},
            "expected": {v: round(float(e), 1) for v, e in zip(ds.variants, expected)},
            "chi2": round(float(chi2), 2), "p_value": float(p)}
    sql = ("SELECT variant, COUNT(DISTINCT user_id) AS users\nFROM assignments\nGROUP BY variant;\n\n"
           "-- daily view, to find when the imbalance starts\n"
           "SELECT DATE(assigned_at) AS day, variant, COUNT(DISTINCT user_id) AS users\n"
           "FROM assignments\nGROUP BY 1, 2\nORDER BY 1, 2;")
    if p >= SRM_ALPHA:
        return [Finding("allocation", "pass", "Allocation balance",
                        f"Assigned users match the expected split (chi-square p={p:.3f}).", base, sql)]

    # Localize in time: pairwise treatment-vs-control share of users per cohort day.
    days = _day_range(ds)
    daily = ds.assign.groupby(["day", "variant"]).size().unstack("variant").reindex(days).fillna(0)
    worst, onset, cp_info = None, None, None
    for t in [v for v in ds.variants if v != ds.control]:
        T, C = daily[t].values, daily[ds.control].values
        cT, cN = np.concatenate([[0], np.cumsum(T)]), np.concatenate([[0], np.cumsum(T + C)])

        def stat(lo, hi, cT=cT, cN=cN):
            N, X = cN[hi] - cN[lo], cT[hi] - cT[lo]
            if N <= 0:
                return float("nan"), float("nan")
            q = X / N
            return q, q * (1 - q) / N
        cp = scan_changepoint(len(days), stat)
        if cp and (worst is None or abs(cp["z"]) > abs(worst["z"])):
            worst, cp_info = cp, t
    if worst and abs(worst["z"]) >= CP_Z:
        onset = days[worst["k"]]
    segs = srm_segments(ds, since=onset)

    lines = [f"Observed split differs from expected (chi-square p={p:.1e}): "
             + ", ".join(f"{v}={int(counts[v]):,}" for v in ds.variants) + "."]
    if onset is not None:
        lines.append(f"The imbalance starts on {onset:%Y-%m-%d} (share of '{cp_info}' vs control moves from "
                     f"{worst['before']:.1%} to {worst['after']:.1%}).")
    else:
        lines.append("No single onset date stands out; the imbalance looks present throughout the run.")
    if segs:
        s = segs[0]
        lines.append(f"The imbalance is concentrated in {s['dim']} = {s['segment']} "
                     f"({', '.join(f'{k}={v:,}' for k, v in s['counts'].items())}).")
    base.update({"onset": None if onset is None else f"{onset:%Y-%m-%d}",
                 "share_before": None if worst is None else round(float(worst["before"]), 4),
                 "share_after": None if worst is None else round(float(worst["after"]), 4),
                 "top_segments": [{"dimension": s["dim"], "segment": s["segment"], "users": s["users"],
                                   "chi2": round(s["chi2"], 1), "p_value": s["p"],
                                   "counts": ", ".join(f"{k}={v:,}" for k, v in s["counts"].items())} for s in segs]})
    rec = ("Do not interpret conversion lift. Find what changed in assignment, redirects, bot/internal-traffic "
           "filtering or caching" + (f" around {onset:%Y-%m-%d}" if onset is not None else "") +
           (f", starting with {segs[0]['dim']} = {segs[0]['segment']}" if segs else "") + ". "
           "Fix it, then restart the experiment or restrict analysis to cohorts before the onset.")
    return [Finding("allocation", "critical", "Sample ratio mismatch (SRM)", " ".join(lines), base, sql, rec)]


# ------------------------------------------------------------------- 2. exposure

def _daily_exposure(ds, treat):
    a = ds.assign[ds.assign["variant"].isin([treat, ds.control])]
    g = a.groupby(["day", "variant"])["exposed"].agg(["sum", "count"]).unstack("variant")
    g = g.reindex(_day_range(ds)).fillna(0)
    return (g.index, g["sum"][treat].values, g["count"][treat].values,
            g["sum"][ds.control].values, g["count"][ds.control].values)


def segment_gaps(ds: Dataset, treat: str, since=None) -> pd.DataFrame:
    """Exposure-rate gap (treatment - control) per segment, ranked by concentration."""
    a = ds.assign[ds.assign["variant"].isin([treat, ds.control])]
    if since is not None:
        a = a[a["day"] >= since]
    nt_all = (a["variant"] == treat).sum()
    nc_all = (a["variant"] == ds.control).sum()
    if nt_all == 0 or nc_all == 0:
        return pd.DataFrame()
    total_excess = ((a.loc[a.variant == treat, "exposed"].mean()) - (a.loc[a.variant == ds.control, "exposed"].mean())) * nt_all
    rows = []
    for dim in ds.dims:
        g = a.groupby([dim, "variant"])["exposed"].agg(["sum", "count"]).unstack("variant")
        for seg in g.index:
            try:
                st, nt = g.loc[seg, ("sum", treat)], g.loc[seg, ("count", treat)]
                sc, nc = g.loc[seg, ("sum", ds.control)], g.loc[seg, ("count", ds.control)]
            except KeyError:
                continue
            if not (nt >= MIN_SEGMENT_USERS and nc >= MIN_SEGMENT_USERS):
                continue
            pt, pc, z, p = _two_prop(st, nt, sc, nc)
            excess = (pt - pc) * nt
            rows.append({"dimension": dim, "segment": str(seg), "users_treatment": int(nt), "users_control": int(nc),
                         "rate_treatment": pt, "rate_control": pc, "gap": pt - pc, "z": z, "p_value": p,
                         "share_of_total_gap": float(excess / total_excess) if total_excess else float("nan")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["significant"] = df["p_value"] < FAMILY_ALPHA / len(df)
    return df


def _top_segment(seg_df: pd.DataFrame, direction: float):
    if seg_df.empty:
        return None
    cand = seg_df[seg_df["significant"] & (np.sign(seg_df["gap"]) == direction) & (seg_df["share_of_total_gap"] >= 0.10)]
    if cand.empty:
        return None
    return cand.sort_values("gap", key=lambda s: -abs(s)).iloc[0]


def check_exposure(ds: Dataset) -> list[Finding]:
    a = ds.assign
    tab = a.groupby("variant")["exposed"].agg(["sum", "count"]).reindex(ds.variants)
    rates = tab["sum"] / tab["count"]
    chi2, p_all = _chi2_homogeneity(tab["sum"].values, tab["count"].values)
    exposed_counts = {v: int(tab.loc[v, "sum"]) for v in ds.variants}
    exp_share = np.array([ds.expected[v] for v in ds.variants])
    srm_exposed_p = float(stats.chisquare(list(exposed_counts.values()),
                                          exp_share * sum(exposed_counts.values()))[1]) if sum(exposed_counts.values()) else 1.0
    base = {"exposure_rate": {v: round(float(rates[v]), 4) for v in ds.variants},
            "assigned": {v: int(tab.loc[v, "count"]) for v in ds.variants},
            "exposed": exposed_counts, "p_value": p_all,
            "srm_p_among_exposed_users": srm_exposed_p}
    sql = ("SELECT DATE(a.assigned_at) AS day, a.variant,\n"
           "       COUNT(DISTINCT a.user_id) AS assigned,\n"
           "       COUNT(DISTINCT e.user_id) AS exposed,\n"
           "       1.0 * COUNT(DISTINCT e.user_id) / COUNT(DISTINCT a.user_id) AS exposure_rate\n"
           "FROM assignments a\n"
           "LEFT JOIN events e ON e.user_id = a.user_id AND e.event_type = " + _q(ds.exposure_event) + "\n"
           "GROUP BY 1, 2\nORDER BY 1, 2;")

    findings = []
    for t in [v for v in ds.variants if v != ds.control]:
        pt, pc, z, p = _two_prop(tab.loc[t, "sum"], tab.loc[t, "count"], tab.loc[ds.control, "sum"], tab.loc[ds.control, "count"])
        gap = pt - pc
        if p >= SRM_ALPHA or abs(gap) < MIN_EXPOSURE_GAP:
            continue
        days, st, nt, sc, nc = _daily_exposure(ds, t)
        cst, cnt, csc, cnc = (np.concatenate([[0], np.cumsum(x)]) for x in (st, nt, sc, nc))

        def stat(lo, hi):
            n1, n0 = cnt[hi] - cnt[lo], cnc[hi] - cnc[lo]
            if n1 <= 0 or n0 <= 0:
                return float("nan"), float("nan")
            a1, a0 = (cst[hi] - cst[lo]) / n1, (csc[hi] - csc[lo]) / n0
            return a1 - a0, a1 * (1 - a1) / n1 + a0 * (1 - a0) / n0
        cp = scan_changepoint(len(days), stat)
        onset = None
        if cp and abs(cp["z"]) >= CP_Z and abs(cp["after"] - cp["before"]) >= MIN_EXPOSURE_GAP:
            onset = days[cp["k"]]
        seg_df = segment_gaps(ds, t, since=onset)
        top = _top_segment(seg_df, np.sign(gap))

        msg = [f"'{t}' exposure rate is {pt:.1%} vs {pc:.1%} for '{ds.control}' "
               f"(gap {gap * 100:+.1f} pts, p={p:.1e}). Assigned users are missing exposure events, so any analysis "
               f"restricted to exposed users is biased."]
        if onset is not None:
            msg.append(f"The gap begins on {onset:%A %Y-%m-%d}: {cp['before'] * 100:+.1f} pts before, {cp['after'] * 100:+.1f} pts after.")
        else:
            msg.append("No single onset date stands out; the gap looks present throughout the run.")
        if top is not None:
            msg.append(f"It is concentrated in {top['dimension']} = {top['segment']} "
                       f"(exposure {top['rate_treatment']:.1%} vs {top['rate_control']:.1%}; "
                       f"{top['share_of_total_gap']:.0%} of the missing exposures).")
        ev = dict(base)
        ev.update({"treatment": t, "gap_pts": round(gap * 100, 2), "z": round(z, 1),
                   "onset": None if onset is None else f"{onset:%Y-%m-%d}",
                   "gap_before_pts": None if cp is None else round(cp["before"] * 100, 2),
                   "gap_after_pts": None if cp is None else round(cp["after"] * 100, 2),
                   "changepoint_z": None if cp is None else round(cp["z"], 1)})
        if not seg_df.empty:
            shown = seg_df[seg_df["significant"]].copy()
            shown["abs_gap"] = shown["gap"].abs()
            shown = shown.sort_values("abs_gap", ascending=False).head(6)
            ev["top_segments"] = [{"dimension": r.dimension, "segment": r.segment,
                                   "exposure_treatment": f"{r.rate_treatment:.1%}", "exposure_control": f"{r.rate_control:.1%}",
                                   "gap_pts": round(r.gap * 100, 1), "share_of_total_gap": f"{r.share_of_total_gap:.0%}",
                                   "users_treatment": r.users_treatment, "p_value": r.p_value}
                                  for r in shown.itertuples()]
        t_sql = sql
        if top is not None:
            t_sql += ("\n\n-- narrow to the suspect segment"
                      + (f" and the period from {onset:%Y-%m-%d}" if onset is not None else "") + "\n"
                      "SELECT a.variant,\n       COUNT(DISTINCT a.user_id) AS assigned,\n       COUNT(DISTINCT e.user_id) AS exposed\n"
                      "FROM assignments a\nLEFT JOIN events e ON e.user_id = a.user_id AND e.event_type = " + _q(ds.exposure_event) + "\n"
                      "WHERE " + _segment_sql_filter(top["dimension"], top["segment"])
                      + (f"\n  AND a.assigned_at >= {_q(f'{onset:%Y-%m-%d}')}" if onset is not None else "") + "\nGROUP BY a.variant;")
        where = (f"{top['dimension']} = {top['segment']}" if top is not None else "the affected users")
        rec = (f"Do not interpret conversion lift for '{t}' yet. Check the exposure-tracking code path for {where}"
               + (f" in the release deployed on or just before {onset:%Y-%m-%d}" if onset is not None else "") + ". "
               "Repair instrumentation, then restart the experiment"
               + (f" or analyze only cohorts assigned before {onset:%Y-%m-%d}" if onset is not None else "") + ".")
        findings.append(Finding(f"exposure_gap:{t}", "critical", f"Exposure events missing for '{t}'",
                                " ".join(msg), ev, t_sql, rec))
    if findings:
        return findings
    if p_all < FAMILY_ALPHA:
        return [Finding("exposure_gap", "warning", "Small exposure-rate difference between variants",
                        f"Exposure rates differ slightly across variants (p={p_all:.1e}) but the gap is under "
                        f"{MIN_EXPOSURE_GAP * 100:.1f} pts.", base, sql,
                        "Monitor; confirm the exposure event fires identically in every variant.")]
    return [Finding("exposure_gap", "pass", "Assignment-to-exposure reconciliation",
                    "Exposure rates are consistent across variants (" +
                    ", ".join(f"{v}: {rates[v]:.1%}" for v in ds.variants) + ").", base, sql)]


# ----------------------------------------------------------------- 3. identities

def check_identity(ds: Dataset) -> list[Finding]:
    raw, exp = ds.raw_assign, ds.exposures
    users = raw["user_id"].nunique()
    findings = []

    multi = raw.groupby("user_id")["variant"].nunique()
    n_multi = int((multi > 1).sum())
    pct = n_multi / users if users else 0
    if pct >= 0.001:
        sev = "critical" if pct >= 0.01 else "warning"
        findings.append(Finding("identity_multi_assignment", sev, "Users assigned to more than one variant",
                                f"{n_multi:,} of {users:,} users ({pct:.2%}) appear in more than one variant, so "
                                f"groups are contaminated and treatment effects are diluted.",
                                {"users_in_multiple_variants": n_multi, "share": round(pct, 4)},
                                "SELECT user_id, COUNT(DISTINCT variant) AS variants\nFROM assignments\n"
                                "GROUP BY user_id\nHAVING COUNT(DISTINCT variant) > 1;",
                                "Check bucketing consistency across anonymous and logged-in identities and "
                                "re-bucketing on login; exclude affected users after repair."))

    if len(exp) and "variant" in exp.columns:
        m = exp.merge(ds.assign[["user_id", "variant", "assigned_at"]].rename(
            columns={"variant": "assigned_variant", "assigned_at": "assigned_at_"}), on="user_id", how="left")
        exposed_users = m["user_id"].nunique()
        orphan = m["assigned_variant"].isna()
        mism = ~orphan & m["variant"].notna() & (m["variant"].astype(str) != m["assigned_variant"])
        early = ~orphan & (m["timestamp"] < m["assigned_at_"] - pd.Timedelta(minutes=1))
        for key, mask, title, text, sql, rec in (
            ("identity_exposure_mismatch", mism, "Exposure logged under a different variant than assigned",
             "{n:,} exposed users ({pct:.2%}) were exposed to a variant other than the one they were assigned.",
             "SELECT a.variant AS assigned, e.variant AS exposed, COUNT(DISTINCT e.user_id) AS users\n"
             "FROM events e JOIN assignments a ON a.user_id = e.user_id\n"
             "WHERE e.event_type = " + _q(ds.exposure_event) + "\nGROUP BY 1, 2;",
             "Fix variant propagation between the assignment service and the exposure logger."),
            ("identity_orphan_exposures", orphan, "Exposure events from users with no assignment",
             "{n:,} exposed users ({pct:.2%}) have no assignment record, which usually means identity mismatch "
             "between the assignment and analytics systems.",
             "SELECT COUNT(DISTINCT e.user_id) AS orphan_users\nFROM events e\n"
             "LEFT JOIN assignments a ON a.user_id = e.user_id\nWHERE e.event_type = " + _q(ds.exposure_event)
             + " AND a.user_id IS NULL;",
             "Reconcile user identifiers across the assignment service and the analytics pipeline."),
            ("identity_early_exposure", early, "Exposure recorded before assignment",
             "{n:,} exposed users ({pct:.2%}) have exposure events timestamped before their assignment.",
             "SELECT COUNT(DISTINCT e.user_id) AS users\nFROM events e JOIN assignments a ON a.user_id = e.user_id\n"
             "WHERE e.event_type = " + _q(ds.exposure_event) + " AND e.timestamp < a.assigned_at;",
             "Check clock skew and event-time vs. ingestion-time fields."),
        ):
            n = int(m.loc[mask, "user_id"].nunique())
            pct = n / exposed_users if exposed_users else 0
            if pct >= 0.001:
                ev = {"users": n, "share_of_exposed_users": round(pct, 4)}
                if key == "identity_exposure_mismatch":
                    ev["assigned_vs_exposed"] = (m[mask].groupby(["assigned_variant", "variant"])["user_id"].nunique()
                                                 .reset_index().rename(columns={"variant": "exposed_variant", "user_id": "users"})
                                                 .to_dict("records"))
                findings.append(Finding(key, "critical" if pct >= 0.01 else "warning", title,
                                        text.format(n=n, pct=pct), ev, sql, rec))
    if not findings:
        findings.append(Finding("identity", "pass", "Identity consistency",
                                "Users stay in one variant and exposure events match their assignment.",
                                {"users": users},
                                "SELECT user_id, COUNT(DISTINCT variant) AS variants\nFROM assignments\n"
                                "GROUP BY user_id\nHAVING COUNT(DISTINCT variant) > 1;"))
    return findings


# ---------------------------------------------------------------- 4. duplicates

def check_duplicates(ds: Dataset) -> list[Finding]:
    ev = ds.events
    if "event_id" in ev.columns and ev["event_id"].notna().any():
        dup = ev["event_id"].notna() & ev.duplicated("event_id", keep="first")
        basis = "event_id"
        sql = ("SELECT a.variant, COUNT(*) AS events,\n       COUNT(*) - COUNT(DISTINCT e.event_id) AS duplicate_rows\n"
               "FROM events e JOIN assignments a ON a.user_id = e.user_id\nGROUP BY a.variant;")
    else:
        dup = ev.duplicated(["user_id", "event_type", "timestamp"], keep="first")
        basis = "(user_id, event_type, timestamp)"
        sql = ("SELECT a.variant, COUNT(*) AS events,\n"
               "       COUNT(*) - COUNT(DISTINCT e.user_id || e.event_type || e.timestamp) AS duplicate_rows\n"
               "FROM events e JOIN assignments a ON a.user_id = e.user_id\nGROUP BY a.variant;")
    variant = ev["user_id"].map(ds.assign.set_index("user_id")["variant"])
    df = pd.DataFrame({"variant": variant, "dup": dup}).dropna(subset=["variant"])
    tab = df.groupby("variant")["dup"].agg(["sum", "count"]).reindex(ds.variants).fillna(0)
    rates = (tab["sum"] / tab["count"].replace(0, np.nan)).fillna(0)
    chi2, p = _chi2_homogeneity(tab["sum"].values, tab["count"].values)
    overall = float(dup.mean()) if len(ev) else 0.0
    evd = {"duplicate_basis": basis, "overall_duplicate_rate": round(overall, 4),
           "duplicate_rate_by_variant": {v: round(float(rates[v]), 4) for v in ds.variants},
           "duplicate_rows_by_variant": {v: int(tab.loc[v, "sum"]) for v in ds.variants}, "p_value": p}
    spread = float(rates.max() - rates.min())
    if p < SRM_ALPHA and spread >= 0.002:
        worst = rates.idxmax()
        # when do they start?
        d = ev.assign(dup=dup, variant=variant).dropna(subset=["variant"])
        d = d[d["variant"] == worst]
        by_day = d.groupby(d["timestamp"].dt.floor("D"))["dup"].mean()
        onset_txt = ""
        if len(by_day) >= 4:
            vals = by_day.values
            cp = scan_changepoint(len(vals), lambda lo, hi: (float(np.mean(vals[lo:hi])), float(np.var(vals[lo:hi]) / max(1, hi - lo))))
            if cp and abs(cp["z"]) >= 4 and cp["after"] - cp["before"] > 0.002:
                onset_txt = f" Duplicates in '{worst}' step up from {cp['before']:.1%} to {cp['after']:.1%} on {by_day.index[cp['k']]:%Y-%m-%d}."
                evd["onset"] = f"{by_day.index[cp['k']]:%Y-%m-%d}"
        return [Finding("duplicates", "critical", "Duplicate events differ by variant",
                        f"Duplicate-event rate is {rates[worst]:.1%} in '{worst}' vs {rates.min():.1%} in the lowest variant "
                        f"(p={p:.1e}), so event counts and per-user metrics are inflated unevenly.{onset_txt}",
                        evd, sql, "Deduplicate on event_id upstream and check client retry or double-fire logic in the "
                        f"'{worst}' code path before comparing metrics.")]
    if overall >= 0.01:
        return [Finding("duplicates", "warning", "Duplicate events present",
                        f"{overall:.1%} of events are duplicates (spread evenly across variants).", evd, sql,
                        "Deduplicate on event_id before computing metrics.")]
    return [Finding("duplicates", "pass", "Duplicate events", f"Duplicate rate is {overall:.2%} and similar across variants.", evd, sql)]


# ------------------------------------------------------------ 5. event volume

def check_volume(ds: Dataset) -> list[Finding]:
    ev = ds.events
    variant = ev["user_id"].map(ds.assign.set_index("user_id")["variant"])
    df = pd.DataFrame({"day": ev["timestamp"].dt.floor("D"), "variant": variant}).dropna()
    sql = ("SELECT DATE(e.timestamp) AS day, a.variant, COUNT(*) AS events\nFROM events e\n"
           "JOIN assignments a ON a.user_id = e.user_id\nGROUP BY 1, 2\nORDER BY 1, 2;")
    if df.empty:
        return []
    days = pd.date_range(df["day"].min(), df["day"].max(), freq="D")
    counts = pd.crosstab(df["day"], df["variant"]).reindex(days).fillna(0)
    if len(days) < 5:
        return [Finding("volume", "pass", "Event volume continuity", "Too few days to assess volume gaps.", {}, sql)]
    interior = counts.iloc[1:-1]
    total_med = interior.sum(axis=1).median()
    issues = []
    for d, row in interior.iterrows():
        tot = row.sum()
        if total_med > 0 and tot < 0.4 * total_med:
            issues.append({"day": f"{d:%Y-%m-%d}", "scope": "all variants", "events": int(tot), "typical": int(total_med)})
            continue
        for v in ds.variants:
            share = interior[v].sum() / interior.values.sum()
            if share > 0 and tot > 0 and row[v] < 0.5 * share * tot:
                issues.append({"day": f"{d:%Y-%m-%d}", "scope": v, "events": int(row[v]), "typical": int(round(share * tot))})
    if issues:
        return [Finding("volume", "warning", "Event-volume gaps detected",
                        f"{len(issues)} day/variant combinations have less than half the expected event volume "
                        f"(first: {issues[0]['day']}, {issues[0]['scope']}). This often indicates a logging outage.",
                        {"gaps": issues[:10]}, sql,
                        "Check pipeline health and backfill missing events before analysis.")]
    return [Finding("volume", "pass", "Event volume continuity", "No day or variant shows a volume collapse.", {}, sql)]


ALL_CHECKS = (check_allocation, check_exposure, check_identity, check_duplicates, check_volume)
