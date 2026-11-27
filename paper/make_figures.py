#!/usr/bin/env python
"""Build paper figures from the small JSON result files in runs/.

    python paper/make_figures.py                       # 0.5B drift + synthetic ablations
    python paper/make_figures.py --run runs/qwen2_to_qwen25_1.5b --tag qwen15b
        # also: drift_<tag>.pdf and gap_<tag>.pdf from that run dir, if present

Figures:
  drift_qwen05b.pdf    per-module 1-CKA drift by layer and by module type
  ablations_synth.pdf  synthetic sweep: recovery vs anchor tokens, per solver/scope
  drift_<tag>.pdf      same drift figure for another run
  gap_<tag>.pdf        gap recovered per task, Cross-LoRA vs Procrustes vs ridge
"""
import argparse
import collections
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "paper" / "figures"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
MODS = ["q_proj", "k_proj", "v_proj", "o_proj"]
COLORS = {"q_proj": "#4C72B0", "k_proj": "#55A868", "v_proj": "#C44E52", "o_proj": "#8172B2"}
METHOD_COLORS = {"crosslora": "#999999", "procrustes": "#DD8452", "ridge": "#4C72B0"}
METHOD_LABELS = {"crosslora": "Cross-LoRA (weights)", "procrustes": "Procrustes (acts.)",
                 "ridge": "ridge (acts.)"}


def drift_figure(path: Path, out: Path, title: str):
    d = json.loads(path.read_text())
    by_type = collections.defaultdict(dict)
    for k, v in d.items():
        parts = k.split(".")
        layer = int(parts[parts.index("layers") + 1])
        by_type[parts[-1]][layer] = v

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [2.2, 1]})
    for t in MODS:
        xs = sorted(by_type[t])
        ax1.plot(xs, [by_type[t][x] for x in xs], marker="o", ms=3, lw=1.2, color=COLORS[t],
                 label=t, alpha=0.6 if t in ("k_proj", "v_proj") else 1.0)
    ax1.set_xlabel("layer")
    ax1.set_ylabel("drift  (1 $-$ CKA)")
    ax1.set_ylim(0, None)
    ax1.legend(frameon=False, ncol=2)
    ax1.set_title(title, loc="left")

    data = [list(by_type[t].values()) for t in MODS]
    bp = ax2.boxplot(data, tick_labels=["q", "k", "v", "o"], patch_artist=True, widths=0.6)
    for patch, t in zip(bp["boxes"], MODS):
        patch.set_facecolor(COLORS[t])
        patch.set_alpha(0.5)
    ax2.set_ylabel("drift")
    ax2.set_ylim(0, None)
    ax2.set_title("by module", loc="left")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] {out}  mean drift {statistics.mean(d.values()):.3f}  n={len(d)}")


def ablation_figure(path: Path, out: Path):
    rows = json.loads(path.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for ax, scope in zip(axes, ["per_layer", "shared"]):
        series = collections.defaultdict(list)
        for r in rows:
            if r["map_scope"] != scope:
                continue
            key = "Procrustes" if r["method"] == "procrustes" else f"ridge $\\lambda$={r['lam']:g}"
            series[key].append((r["anchor_tokens"], r["score"]["recovery"]))
        for key, pts in sorted(series.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [100 * p[1] for p in pts], marker="o", ms=3, label=key,
                    ls="--" if key == "Procrustes" else "-")
        ax.set_xscale("log")
        ax.set_xlabel("anchor tokens")
        ax.set_title("per-module maps" if scope == "per_layer" else "one shared map", loc="left")
        ax.axhline(0, color="k", lw=0.6)
    axes[0].set_ylabel("error reduction vs naive (%)")
    axes[0].set_ylim(0, 100)
    axes[1].set_ylim(-100, 100)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] {out}")


def gap_figure(summary_path: Path, out: Path):
    """Grouped bars: gap recovered (%) per task for the three transfer methods."""
    rows = json.loads(summary_path.read_text())
    rows = [r for r in rows if r.get("gap_recovered_pct")]
    if not rows:
        print(f"[fig] skip {out}: no gap_recovered_pct in {summary_path} (oracle missing?)")
        return
    methods = [m for m in ("crosslora", "procrustes", "ridge")
               if any(m in r["gap_recovered_pct"] for r in rows)]
    fig, ax = plt.subplots(figsize=(3.4 + 0.6 * len(rows), 2.5))
    width = 0.8 / len(methods)
    for i, m in enumerate(methods):
        xs = [j + (i - (len(methods) - 1) / 2) * width for j in range(len(rows))]
        ys = [r["gap_recovered_pct"].get(m, float("nan")) for r in rows]
        ax.bar(xs, ys, width, color=METHOD_COLORS[m], label=METHOD_LABELS[m])
        for x, y in zip(xs, ys):
            if not math.isnan(y):
                ax.text(x, y + (2 if y >= 0 else -6), f"{y:.0f}", ha="center", fontsize=7)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([r["task"] for r in rows])
    ax.axhline(0, color="k", lw=0.6)
    ax.axhline(100, color="k", lw=0.6, ls=":")
    ax.set_ylabel("oracle$-$naive gap recovered (%)")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] {out}")


def heads_figure(path: Path, out: Path):
    """Held-out R^2 per layer: how well linear maps of different structure
    explain version drift at the residual stream and the attention output."""
    d = json.loads(path.read_text())
    L = [r["layer"] for r in d["layers"]]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for key, lab, c, ls in [("full_ridge", "ridge (unconstrained)", "#4C72B0", "-"),
                            ("orthogonal", "orthogonal", "#DD8452", "--")]:
        ax1.plot(L, [r["residual"][key] for r in d["layers"]], marker="o", ms=3, color=c, ls=ls, label=lab)
    ax1.set_title("residual stream (q/k/v input)", loc="left")
    for key, lab, c, ls in [("full_ridge", "full map", "#4C72B0", "-"),
                            ("block_matched_head", "per-head, heads matched", "#55A868", "-."),
                            ("block_same_head", "per-head, same head", "#C44E52", ":")]:
        ax2.plot(L, [r["o_proj"][key] for r in d["layers"]], marker="o", ms=3, color=c, ls=ls, label=lab)
    ax2.set_title("attention output (o_proj input)", loc="left")
    for ax in (ax1, ax2):
        ax.set_xlabel("layer")
        ax.set_ylim(0, 1)
        ax.legend(frameon=False, fontsize=7)
    ax1.set_ylabel("held-out $R^2$")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"[fig] {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="run dir with drift.json / summary.json")
    ap.add_argument("--tag", default=None, help="figure-name suffix for --run outputs")
    ap.add_argument("--title", default=None, help="drift figure title for --run")
    args = ap.parse_args()
    FIG.mkdir(exist_ok=True)

    drift_figure(ROOT / "runs/qwen05b_cpu/drift.json", FIG / "drift_qwen05b.pdf",
                 "Qwen2-0.5B $\\to$ Qwen2.5-0.5B, input activations")
    ablation_figure(ROOT / "runs/ablations_synth.json", FIG / "ablations_synth.pdf")
    if (ROOT / "runs/qwen05b_heads/heads.json").exists():
        heads_figure(ROOT / "runs/qwen05b_heads/heads.json", FIG / "heads_qwen05b.pdf")

    if args.run:
        run = Path(args.run)
        tag = args.tag or run.name
        if (run / "drift.json").exists():
            drift_figure(run / "drift.json", FIG / f"drift_{tag}.pdf",
                         args.title or f"{tag}, input activations")
        if (run / "summary.json").exists():
            gap_figure(run / "summary.json", FIG / f"gap_{tag}.pdf")


if __name__ == "__main__":
    main()
