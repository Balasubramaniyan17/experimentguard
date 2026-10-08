"""Synthetic experiments with known faults, for demos and tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

SCENARIOS = ("healthy", "exposure_loss", "srm_dropout", "duplicates", "crossover")


def generate(scenario: str = "exposure_loss", n_users: int = 40_000, days: int = 14,
             seed: int = 7, start: str = "2026-09-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (assignments, events). Fault onset is always on day index 7 (a Tuesday for the default start)."""
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(start)
    onset = 7

    day = rng.integers(0, days, n_users)
    assigned_at = t0 + pd.to_timedelta(day, unit="D") + pd.to_timedelta(rng.integers(0, 86_400, n_users), unit="s")
    variant = np.where(rng.random(n_users) < 0.5, "control", "treatment")
    device = rng.choice(["ios", "android", "web"], n_users, p=[0.30, 0.25, 0.45])
    app_version = np.where(device == "web", "latest", np.where(rng.random(n_users) < 0.55, "3.2", "3.1"))
    channel = rng.choice(["organic", "paid", "email"], n_users, p=[0.50, 0.35, 0.15])
    users = pd.DataFrame({"user_id": [f"u{i:07d}" for i in range(n_users)], "variant": variant,
                          "assigned_at": assigned_at, "device": device, "app_version": app_version,
                          "channel": channel, "day": day})

    if scenario == "srm_dropout":  # e.g. a redirect fails for treatment on paid traffic after the onset
        drop = (users.day >= onset) & (users.variant == "treatment") & (users.channel == "paid") & (rng.random(n_users) < 0.35)
        users = users[~drop].reset_index(drop=True)
    n = len(users)

    really_exposed = rng.random(n) < 0.97
    logged = really_exposed.copy()
    if scenario == "exposure_loss":  # new tracking code on iOS 3.2 drops treatment exposure events
        broken = (users.day >= onset) & (users.variant == "treatment") & (users.device == "ios") & (users.app_version == "3.2")
        logged &= ~(broken.values & (rng.random(n) < 0.75))

    exp_variant = users.variant.values.copy()
    if scenario == "crossover":  # bucketing inconsistency between services
        flip = (rng.random(n) < 0.03)
        exp_variant = np.where(flip, np.where(exp_variant == "control", "treatment", "control"), exp_variant)

    exposure = users.loc[logged, ["user_id", "assigned_at", "device", "app_version", "channel"]].copy()
    exposure["event_type"] = "exposure"
    exposure["timestamp"] = exposure.pop("assigned_at") + pd.to_timedelta(rng.integers(1, 120, len(exposure)), unit="s")
    exposure["variant"] = exp_variant[logged]

    converts = really_exposed & (rng.random(n) < np.where(users.variant == "treatment", 0.105, 0.10))
    conv = users.loc[converts, ["user_id", "assigned_at", "device", "app_version", "channel"]].copy()
    conv["event_type"] = "conversion"
    conv["timestamp"] = conv.pop("assigned_at") + pd.to_timedelta(rng.integers(120, 3_600, len(conv)), unit="s")
    conv["variant"] = np.nan

    events = pd.concat([exposure, conv], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    events.insert(0, "event_id", [f"e{i:08d}" for i in range(len(events))])

    if scenario == "duplicates":  # retry bug double-fires events for treatment on web
        v = events.user_id.map(users.set_index("user_id").variant)
        d = events.user_id.map(users.set_index("user_id").device)
        late = events.timestamp >= t0 + pd.Timedelta(days=onset)
        pick = (v == "treatment") & (d == "web") & late & (rng.random(len(events)) < 0.25)
        events = pd.concat([events, events[pick]], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    assignments = users.drop(columns="day")
    if scenario == "crossover":  # some users are re-bucketed on login and get a second assignment row
        re = assignments[rng.random(len(assignments)) < 0.03].copy()
        re["variant"] = np.where(re.variant == "control", "treatment", "control")
        re["assigned_at"] = re.assigned_at + pd.Timedelta(hours=2)
        assignments = pd.concat([assignments, re], ignore_index=True)
    return assignments, events[["event_id", "user_id", "event_type", "timestamp", "variant", "device", "app_version", "channel"]]
