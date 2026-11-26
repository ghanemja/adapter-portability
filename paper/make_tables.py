#!/usr/bin/env python
"""Turn a run directory's results into LaTeX the paper \\input{}s directly.

    python paper/make_tables.py --run runs/qwen2_to_qwen25_1.5b \
                                --config configs/qwen2_to_qwen25_1.5b.yaml

Writes (into paper/tables/ unless --out is given):
  main_results.tex   Table: naive | Cross-LoRA | Procrustes | ridge | oracle per
                     task, plus a gap-recovered row block. Best transfer method
                     per task in bold.
  sweeps.tex         Table: gap recovered vs ridge lambda and vs Cross-LoRA rank k
                     (from results_aligned_ridge_lam*.json / results_crosslora_r*.json).
  results_macros.tex \\newcommand macros for every number quoted in prose,
                     e.g. \\gapRidgeGsmeightk, \\gapCrossloraMean, \\marginMean.
  verdict.json       the kill-condition numbers (HANDOFF.md section 4).

It reads only results_*.json and the config (for task metrics and the
higher-is-better direction), so it runs anywhere, no GPU or model needed.
Missing conditions render as '--'; nothing is ever filled in or estimated.
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adaptport import load_config  # noqa: E402
from adaptport.eval import HIGHER_BETTER, gap_recovered  # noqa: E402

CORE = [
    ("naive", "results_naive.json", "Naive"),
    ("crosslora", "results_crosslora.json", "Cross-LoRA"),
    ("procrustes", "results_aligned_procrustes.json", "Procrustes"),
    ("ridge", "results_aligned_ridge.json", "Ridge"),
    ("oracle", "results_oracle.json", "Oracle"),
]
TRANSFER = ["crosslora", "procrustes", "ridge"]
METRIC_LABEL = {
    "accuracy": r"acc.\,$\uparrow$",
    "exact_match": r"EM\,$\uparrow$",
    "perplexity": r"ppl\,$\downarrow$",
    "completion_perplexity": r"ppl\,$\downarrow$",
}
NAN = float("nan")

# Kill condition: activation ridge must beat Cross-LoRA by a clear margin in
# mean gap recovered. 10 points is the pre-registered threshold written into
# HANDOFF.md before any GPU result exists; do not tune it after the fact.
KILL_MARGIN_PTS = 10.0


def _load(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def _val(d: dict, task: str) -> float:
    return d.get(task, {}).get("value", NAN)


def _fmt(x: float, metric: str) -> str:
    if x is None or math.isnan(x):
        return "--"
    if metric in ("accuracy", "exact_match"):
        return f"{100 * x:.1f}"
    return f"{x:.2f}"


def _pct(x: float) -> str:
    return "--" if x is None or math.isnan(x) else f"{x:.1f}"


def _macro_name(*parts: str) -> str:
    """LaTeX macro names allow letters only: spell digits, drop punctuation."""
    digits = dict(zip("0123456789", ["zero", "one", "two", "three", "four", "five",
                                     "six", "seven", "eight", "nine"]))
    out = ""
    for p in parts:
        p = "".join(digits.get(c, c) for c in p)
        p = re.sub(r"[^A-Za-z]", "", p)
        out += p[:1].upper() + p[1:]
    return out[:1].lower() + out[1:]


def gaps_for(tasks, data):
    """{task: {method: gap%}} using the config's metric direction."""
    out = {}
    for t in tasks:
        n, o = _val(data["naive"], t.name), _val(data["oracle"], t.name)
        higher = t.metric in HIGHER_BETTER
        out[t.name] = {}
        for m in data:
            if m in ("naive", "oracle"):
                continue
            a = _val(data[m], t.name)
            if math.isnan(n) or math.isnan(o) or math.isnan(a):
                out[t.name][m] = NAN
            else:
                out[t.name][m] = 100 * gap_recovered(n, a, o, higher_better=higher)
    return out


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else NAN


def main_table(tasks, data, gaps) -> str:
    cols = [c for c in CORE if data.get(c[0])]
    head = " & ".join(["Task", "Metric"] + [c[2] for c in cols])
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{@{}ll" + "r" * len(cols) + r"@{}}", r"\toprule",
        head + r" \\", r"\midrule",
    ]
    for t in tasks:
        g = gaps[t.name]
        best = max((m for m in TRANSFER if not math.isnan(g.get(m, NAN))),
                   key=lambda m: g[m], default=None)
        cells = []
        for key, _, _ in cols:
            s = _fmt(_val(data[key], t.name), t.metric)
            cells.append(rf"\textbf{{{s}}}" if key == best else s)
        lines.append(" & ".join([t.name, METRIC_LABEL.get(t.metric, t.metric)] + cells) + r" \\")
    if data.get("oracle"):
        lines.append(r"\midrule")
        lines.append(r"\multicolumn{" + str(len(cols) + 2) + r"}{@{}l}{\emph{oracle$-$naive gap recovered (\%)}} \\")
        for t in tasks:
            cells = [("" if k in ("naive", "oracle") else _pct(gaps[t.name].get(k, NAN)))
                     for k, _, _ in cols]
            lines.append(" & ".join([t.name, ""] + cells) + r" \\")
        cells = [("" if k in ("naive", "oracle") else _pct(mean([gaps[t.name].get(k, NAN) for t in tasks])))
                 for k, _, _ in cols]
        lines.append(" & ".join([r"\emph{mean}", ""] + cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Transferring LoRA adapters from Qwen2-1.5B to Qwen2.5-1.5B with no task "
        r"data. Naive loads the source adapter unchanged; Cross-LoRA projects it onto the "
        r"target weights' top-$k$ singular subspaces ($k=320$); Procrustes and ridge use "
        r"per-module maps fit on unlabeled anchor activations; Oracle is a LoRA retrained on "
        r"the new base with task data. Accuracy and EM in \%. Bold: best transfer method per "
        r"task by gap recovered.}",
        r"\label{tab:main}", r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def sweep_table(tasks, run: Path, data, oracle_present: bool) -> str | None:
    ridge = sorted(run.glob("results_aligned_ridge_lam*.json"))
    cross = sorted(run.glob("results_crosslora_r*.json"))
    if not (ridge or cross) or not oracle_present:
        return None
    entries = []   # (family, setting, data)
    if data.get("ridge"):
        entries.append(("ridge", r"$\lambda=1$", data["ridge"]))
    for p in ridge:
        lam = p.stem.split("lam", 1)[1]
        entries.append(("ridge", rf"$\lambda={lam}$", _load(p)))
    if data.get("crosslora"):
        entries.append(("crosslora", r"$k=320$", data["crosslora"]))
    for p in cross:
        k = p.stem.split("_r", 1)[1]
        entries.append(("crosslora", rf"$k={k}$", _load(p)))

    def sort_key(e):
        num = re.findall(r"[\d.]+", e[1])
        return (e[0] != "ridge", float(num[0]) if num else 0.0)

    entries.sort(key=sort_key)
    lines = [
        r"\begin{table}[t]", r"\centering", r"\small",
        r"\begin{tabular}{@{}ll" + "r" * (len(tasks) + 1) + r"@{}}", r"\toprule",
        " & ".join(["Method", "Setting"] + [t.name for t in tasks] + ["mean"]) + r" \\",
        r"\midrule",
    ]
    prev = None
    for fam, setting, d in entries:
        if prev and fam != prev:
            lines.append(r"\midrule")
        prev = fam
        g = gaps_for(tasks, {"naive": data["naive"], "oracle": data["oracle"], "x": d})
        vals = [g[t.name]["x"] for t in tasks]
        name = "Ridge (acts.)" if fam == "ridge" else "Cross-LoRA (weights)"
        lines.append(" & ".join([name, setting] + [_pct(v) for v in vals] + [_pct(mean(vals))]) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Sensitivity of gap recovered (\%) to the ridge penalty $\lambda$ and to "
        r"Cross-LoRA's truncation rank $k$ on Qwen2-1.5B$\to$Qwen2.5-1.5B. Each method is "
        r"also reported at its best setting so the comparison does not hinge on defaults.}",
        r"\label{tab:sweeps}", r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def macros(tasks, data, gaps) -> str:
    out = ["% Auto-generated by paper/make_tables.py. Do not edit by hand."]

    def add(name, value):
        out.append(rf"\newcommand{{\{name}}}{{{value}}}")

    for t in tasks:
        for key, _, _ in CORE:
            if data.get(key):
                add(_macro_name("val", key, t.name), _fmt(_val(data[key], t.name), t.metric))
        for m in TRANSFER:
            if m in gaps[t.name]:
                add(_macro_name("gap", m, t.name), _pct(gaps[t.name][m]))
    for m in TRANSFER:
        add(_macro_name("gap", m, "mean"), _pct(mean([gaps[t.name].get(m, NAN) for t in tasks])))
    margin = mean([gaps[t.name].get("ridge", NAN) - gaps[t.name].get("crosslora", NAN) for t in tasks])
    add("marginMean", _pct(margin))
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=str(ROOT / "paper" / "tables"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    data = {k: _load(run / fn) for k, fn, _ in CORE}
    if not data["naive"]:
        sys.exit(f"[tables] no results_naive.json in {run}; nothing to tabulate")
    gaps = gaps_for(cfg.tasks, data)

    (out / "main_results.tex").write_text(main_table(cfg.tasks, data, gaps))
    (out / "results_macros.tex").write_text(macros(cfg.tasks, data, gaps))
    print(f"[tables] {out / 'main_results.tex'}")
    print(f"[tables] {out / 'results_macros.tex'}")
    sw = sweep_table(cfg.tasks, run, data, bool(data["oracle"]))
    if sw:
        (out / "sweeps.tex").write_text(sw)
        print(f"[tables] {out / 'sweeps.tex'}")

    # ---- kill condition ----------------------------------------------------
    per_task = {t.name: {m: gaps[t.name].get(m, NAN) for m in TRANSFER} for t in cfg.tasks}
    m_ridge = mean([v["ridge"] for v in per_task.values()])
    m_cross = mean([v["crosslora"] for v in per_task.values()])
    margin = m_ridge - m_cross
    wins = sum(1 for v in per_task.values()
               if not math.isnan(v["ridge"]) and not math.isnan(v["crosslora"])
               and v["ridge"] > v["crosslora"])
    if math.isnan(margin):
        verdict = "INCOMPLETE: need naive, oracle, ridge and crosslora results"
    elif margin >= KILL_MARGIN_PTS and wins == len(cfg.tasks):
        verdict = f"CONTINUE: ridge beats Cross-LoRA on every task, mean margin {margin:.1f} pts"
    elif margin >= KILL_MARGIN_PTS:
        verdict = (f"CONTINUE WITH CARE: mean margin {margin:.1f} pts but ridge wins only "
                   f"{wins}/{len(cfg.tasks)} tasks")
    else:
        verdict = (f"KILL CONDITION MET: mean margin {margin:.1f} pts < {KILL_MARGIN_PTS:.0f}; "
                   "activation alignment is not clearly better than weight-only Cross-LoRA")
    (out / "verdict.json").write_text(json.dumps({
        "per_task_gap_recovered_pct": per_task,
        "mean_ridge": m_ridge, "mean_crosslora": m_cross, "mean_margin_pts": margin,
        "ridge_wins": wins, "n_tasks": len(cfg.tasks),
        "threshold_pts": KILL_MARGIN_PTS, "verdict": verdict,
    }, indent=2, allow_nan=True))
    print(f"[verdict] {verdict}")


if __name__ == "__main__":
    main()
