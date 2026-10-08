"""Render a diagnosis as Markdown, a self-contained HTML page, or JSON."""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd

ACTION_LABEL = {
    "proceed": "Proceed with analysis",
    "investigate": "Investigate before trusting results",
    "repair_instrumentation": "Repair instrumentation, then restart or restrict the analysis window",
    "restart_experiment": "Fix the root cause and restart the experiment",
}


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    return str(o)


def to_json(result: dict) -> str:
    return json.dumps(result, indent=2, default=_default)


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.2e}" if (v != 0 and abs(v) < 0.001) else f"{v:,.4g}"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt(x)}" for k, x in v.items())
    if v is None:
        return "-"
    return str(v)


# ---------------------------------------------------------------------- markdown

def to_markdown(result: dict) -> str:
    e = result["experiment"]
    out = [f"# ExperimentGuard report: {result['verdict']}", "", result["headline"], "",
           f"**Recommended action:** {ACTION_LABEL[result['recommended_action']]}. {result['action_detail']}", "",
           f"Experiment: {e['users_assigned']:,} users, "
           + ", ".join(f"{k}={v:,}" for k, v in e["variants"].items())
           + f", assigned {e['first_assignment']} to {e['last_assignment']}, {e['events']:,} events.", ""]
    for f in result["findings"]:
        tag = {"critical": "CRITICAL", "warning": "WARNING", "pass": "PASS"}[f["severity"]]
        out += [f"## [{tag}] {f['title']}", "", f["summary"], ""]
        if f["recommendation"]:
            out += [f"**What to do:** {f['recommendation']}", ""]
        if f["severity"] != "pass":
            for k, v in f["evidence"].items():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    cols = list(v[0].keys())
                    out += [f"*{k}*", "", "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
                    out += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in v] + [""]
                elif not isinstance(v, list):
                    out.append(f"- {k}: {_fmt(v)}")
            out.append("")
        if f["sql"]:
            out += ["Reproduce:", "", "```sql", f["sql"], "```", ""]
    return "\n".join(out)


# -------------------------------------------------------------------------- html

CSS = """
:root{--bg:#f7f6f3;--fg:#1d2127;--muted:#5d6672;--line:#d9dbd6;--card:#fff;--crit:#b3261e;--crit-bg:#fbe9e7;
--warn:#8a5a00;--warn-bg:#fff3d6;--ok:#1b6b3a;--ok-bg:#e3f4e8;--code:#eef0ec}
:root:not([data-theme="light"]){color-scheme:light dark}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#15181c;--fg:#e8eaed;--muted:#9aa3ad;--line:#2b3138;
--card:#1c2025;--crit:#ff8a80;--crit-bg:#3a1f1d;--warn:#f2c36b;--warn-bg:#352a12;--ok:#7bd89b;--ok-bg:#16301f;--code:#23282e}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.55 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:32px 20px 64px}
h1{font-size:1.5rem;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:1.1rem;margin:0 0 6px}
.meta{color:var(--muted);font-size:.9rem;margin:0 0 20px}
.verdict{border-left:6px solid var(--c);background:var(--bg-c);padding:16px 18px;border-radius:4px;margin-bottom:24px}
.verdict.UNRELIABLE{--c:var(--crit);--bg-c:var(--crit-bg)}.verdict.SUSPECT{--c:var(--warn);--bg-c:var(--warn-bg)}
.verdict.TRUSTWORTHY{--c:var(--ok);--bg-c:var(--ok-bg)}
.verdict strong.v{font-size:1.25rem;display:block}.verdict p{margin:.4rem 0 0}
.finding{background:var(--card);border:1px solid var(--line);border-left:5px solid var(--c);border-radius:4px;padding:16px 18px;margin:0 0 16px}
.finding.critical{--c:var(--crit)}.finding.warning{--c:var(--warn)}.finding.pass{--c:var(--ok)}
.sev{font-size:.8rem;font-weight:600;color:var(--c)}
.todo{background:var(--code);padding:10px 12px;border-radius:4px;margin:10px 0}
table{border-collapse:collapse;font-size:.85rem;font-variant-numeric:tabular-nums}
.scroll{overflow-x:auto;margin:10px 0}th,td{text-align:left;padding:4px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th{color:var(--muted);font-weight:600}
dl{display:grid;grid-template-columns:max-content 1fr;gap:2px 14px;font-size:.88rem;margin:10px 0}dt{color:var(--muted)}dd{margin:0}
details{margin-top:10px}summary{cursor:pointer;color:var(--muted);font-size:.9rem}
pre{background:var(--code);padding:12px;border-radius:4px;overflow-x:auto;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace}
.passes{color:var(--muted);font-size:.95rem}.passes li{margin:2px 0}
"""


def _e(x):
    return html.escape(str(x))


def _evidence_html(ev: dict) -> str:
    scalars, tables = [], []
    for k, v in ev.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            cols = list(v[0].keys())
            head = "".join(f"<th>{_e(c)}</th>" for c in cols)
            body = "".join("<tr>" + "".join(f"<td>{_e(_fmt(r.get(c)))}</td>" for c in cols) + "</tr>" for r in v)
            tables.append(f"<div class='scroll'><table><caption style='text-align:left;color:var(--muted)'>{_e(k)}</caption>"
                          f"<tr>{head}</tr>{body}</table></div>")
        elif not isinstance(v, list):
            scalars.append(f"<dt>{_e(k)}</dt><dd>{_e(_fmt(v))}</dd>")
    return (f"<dl>{''.join(scalars)}</dl>" if scalars else "") + "".join(tables)


def to_html(result: dict) -> str:
    e = result["experiment"]
    parts = [f"<main><h1>ExperimentGuard report</h1><p class='meta'>{e['users_assigned']:,} users ("
             + _e(", ".join(f"{k}: {v:,}" for k, v in e["variants"].items()))
             + f") · assigned {e['first_assignment']} to {e['last_assignment']} · {e['events']:,} events</p>",
             f"<section class='verdict {result['verdict']}'><strong class='v'>{result['verdict']}</strong>"
             f"<p>{_e(result['headline'])}</p>"
             f"<p><b>Recommended action:</b> {_e(ACTION_LABEL[result['recommended_action']])}. {_e(result['action_detail'])}</p></section>"]
    passes = []
    for f in result["findings"]:
        if f["severity"] == "pass":
            passes.append(f"<li><b>{_e(f['title'])}</b>: {_e(f['summary'])}</li>")
            continue
        parts.append(f"<article class='finding {f['severity']}'><span class='sev'>{f['severity']}</span>"
                     f"<h2>{_e(f['title'])}</h2><p>{_e(f['summary'])}</p>"
                     + (f"<div class='todo'><b>What to do:</b> {_e(f['recommendation'])}</div>" if f["recommendation"] else "")
                     + _evidence_html(f["evidence"])
                     + (f"<details><summary>Reproduce in SQL</summary><pre>{_e(f['sql'])}</pre></details>" if f["sql"] else "")
                     + "</article>")
    if passes:
        parts.append(f"<h2>Checks passed</h2><ul class='passes'>{''.join(passes)}</ul>")
    parts.append("</main>")
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>ExperimentGuard: {result['verdict']}</title><style>{CSS}</style></head><body>"
            + "".join(parts) + "</body></html>")
