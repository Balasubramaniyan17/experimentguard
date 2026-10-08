# ExperimentGuard report: UNRELIABLE

Do not act on this experiment's results. Exposure events missing for 'treatment'.

**Recommended action:** Repair instrumentation, then restart or restrict the analysis window. Repair the instrumentation behind the critical findings, then restart the experiment, or restrict analysis to cohorts assigned before 2026-09-08. A person should approve either change.

Experiment: 40,000 users, control=20,065, treatment=19,935, assigned 2026-09-01 to 2026-09-14, 41,566 events.

## [CRITICAL] Exposure events missing for 'treatment'

'treatment' exposure rate is 90.9% vs 97.0% for 'control' (gap -6.1 pts, p=2.0e-144). Assigned users are missing exposure events, so any analysis restricted to exposed users is biased. The gap begins on Tuesday 2026-09-08: +0.1 pts before, -12.1 pts after. It is concentrated in device+app_version = ios / 3.2 (exposure 24.5% vs 96.9%; 102% of the missing exposures).

**What to do:** Do not interpret conversion lift for 'treatment' yet. Check the exposure-tracking code path for device+app_version = ios / 3.2 in the release deployed on or just before 2026-09-08. Repair instrumentation, then restart the experiment or analyze only cohorts assigned before 2026-09-08.

- exposure_rate: control: 0.9698, treatment: 0.9088
- assigned: control: 20065, treatment: 19935
- exposed: control: 19460, treatment: 18117
- p_value: 1.99e-144
- srm_p_among_exposed_users: 4.26e-12
- treatment: treatment
- gap_pts: -6.1
- z: -25.6
- onset: 2026-09-08
- gap_before_pts: 0.06
- gap_after_pts: -12.09
- changepoint_z: -26.2
*top_segments*

| dimension | segment | exposure_treatment | exposure_control | gap_pts | share_of_total_gap | users_treatment | p_value |
|---|---|---|---|---|---|---|---|
| device+app_version | ios / 3.2 | 24.5% | 96.9% | -72.3 | 102% | 1716 | 0 |
| device | ios | 56.9% | 97.2% | -40.3 | 102% | 3080 | 3.13e-304 |
| app_version | 3.2 | 56.9% | 97.0% | -40.1 | 102% | 3105 | 2.86e-301 |
| channel | organic | 85.2% | 97.4% | -12.1 | 50% | 5037 | 1.56e-102 |
| channel | email | 84.4% | 96.5% | -12.1 | 15% | 1538 | 1.54e-29 |
| channel | paid | 84.7% | 96.7% | -12 | 35% | 3538 | 2.45e-67 |


Reproduce:

```sql
SELECT DATE(a.assigned_at) AS day, a.variant,
       COUNT(DISTINCT a.user_id) AS assigned,
       COUNT(DISTINCT e.user_id) AS exposed,
       1.0 * COUNT(DISTINCT e.user_id) / COUNT(DISTINCT a.user_id) AS exposure_rate
FROM assignments a
LEFT JOIN events e ON e.user_id = a.user_id AND e.event_type = 'exposure'
GROUP BY 1, 2
ORDER BY 1, 2;

-- narrow to the suspect segment and the period from 2026-09-08
SELECT a.variant,
       COUNT(DISTINCT a.user_id) AS assigned,
       COUNT(DISTINCT e.user_id) AS exposed
FROM assignments a
LEFT JOIN events e ON e.user_id = a.user_id AND e.event_type = 'exposure'
WHERE a.device = 'ios' AND a.app_version = '3.2'
  AND a.assigned_at >= '2026-09-08'
GROUP BY a.variant;
```

## [PASS] Allocation balance

Assigned users match the expected split (chi-square p=0.516).

Reproduce:

```sql
SELECT variant, COUNT(DISTINCT user_id) AS users
FROM assignments
GROUP BY variant;

-- daily view, to find when the imbalance starts
SELECT DATE(assigned_at) AS day, variant, COUNT(DISTINCT user_id) AS users
FROM assignments
GROUP BY 1, 2
ORDER BY 1, 2;
```

## [PASS] Identity consistency

Users stay in one variant and exposure events match their assignment.

Reproduce:

```sql
SELECT user_id, COUNT(DISTINCT variant) AS variants
FROM assignments
GROUP BY user_id
HAVING COUNT(DISTINCT variant) > 1;
```

## [PASS] Duplicate events

Duplicate rate is 0.00% and similar across variants.

Reproduce:

```sql
SELECT a.variant, COUNT(*) AS events,
       COUNT(*) - COUNT(DISTINCT e.event_id) AS duplicate_rows
FROM events e JOIN assignments a ON a.user_id = e.user_id
GROUP BY a.variant;
```

## [PASS] Event volume continuity

No day or variant shows a volume collapse.

Reproduce:

```sql
SELECT DATE(e.timestamp) AS day, a.variant, COUNT(*) AS events
FROM events e
JOIN assignments a ON a.user_id = e.user_id
GROUP BY 1, 2
ORDER BY 1, 2;
```
