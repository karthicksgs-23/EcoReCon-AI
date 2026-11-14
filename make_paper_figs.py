"""Figures 4 and 5 of the paper from results/*.json (reads results only; runs no experiments).
Fig 4 = adaptation-compute breakdown, Fig 5 = sensitivity on matched streams 0-2.
Figures 1-3 come from fig1_diagram.py and aggregate.py."""
import os, json, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
FIG = os.path.join(RES, "figs"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "mathtext.fontset": "stix"})
NAME = {"bn_adapt": "BN-adapt", "periodic_full": "Periodic-full", "drift_ft": "Drift-FT",
        "drift_ft_defer": "Drift-FT-defer", "bank": "Bank", "ecorecon": "EcoReCon"}


def L(s, d, c, p):
    return json.load(open(os.path.join(RES, f"run_s{s}_pre0_D{d}_C{c}_{p}.json")))



# ---- Fig 4: adaptation compute split into drift monitoring and the rest (mean of 5 streams)
pre = json.load(open(os.path.join(RES, "pretrain_seed0.json"))); m = pre["macs"]
mon = 241 * 400 * 2 * (m["all"] - m["module"]) / 1e9            # 240 hourly signatures + 1 reference, GFLOP
pols = ["bn_adapt", "periodic_full", "drift_ft", "drift_ft_defer", "bank", "ecorecon"]
tot = np.array([np.mean([L(s, 6, 8, p)["adapt_gflop"] for s in range(5)]) for p in pols])
monv = np.array([mon if p in ("drift_ft", "drift_ft_defer", "bank", "ecorecon") else 0 for p in pols])
fig, ax = plt.subplots(figsize=(3.4, 2.3))
y = np.arange(len(pols))
ax.barh(y, monv / 1000, color="#56B4E9", label="Drift monitoring (signature)")
ax.barh(y, (tot - monv) / 1000, left=monv / 1000, color="#D55E00", label="Training and BN recalibration")
for i, (t, mv) in enumerate(zip(tot, monv)):
    ax.text(t / 1000 + 0.4, i, f"{t/1000:.1f}", va="center", fontsize=7)
ax.set_yticks(y); ax.set_yticklabels([NAME[p] for p in pols]); ax.invert_yaxis()
ax.set_xlabel("Adaptation compute per 240-h stream (TFLOP)"); ax.set_xlim(0, 38)
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=6.5)
fig.savefig(os.path.join(FIG, "fig4_breakdown.png")); plt.close(fig)

# ---- Fig 5: sensitivity on matched streams 0-2
ds = [0, 3, 6, 12, 24]
R = {d: [L(s, 6, 8, "bank") if d == 0 else L(s, d, 8, "ecorecon") for s in range(3)] for d in ds}
co2 = [np.mean([r["co2_adapt_g"] for r in R[d]]) * 1000 for d in ds]
ci = [np.mean([r["mean_ci_of_adapt"] for r in R[d]]) for d in ds]
acc = [np.mean([r["mean_acc"] for r in R[d]]) * 100 for d in ds]
fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.1))
ax[0].plot(ds, co2, "-o", color="#D55E00", ms=3); ax[0].set_ylabel("Adapt. CO$_2$ (mg)")
a2 = ax[0].twinx(); a2.plot(ds, ci, "--s", color="#0072B2", ms=3); a2.set_ylabel("Mean CI of adapt. (kg/kWh)", color="#0072B2")
ax[0].set_xlabel("Deferral window D (h); D = 0 is Bank"); ax[0].set_xticks(ds); ax[0].set_title("(a) Emissions and carbon intensity")
ax[1].plot(ds, acc, "-o", color="#009E73", ms=3); ax[1].set_xticks(ds); ax[1].set_xlabel("Deferral window D (h)")
ax[1].set_ylabel("Mean accuracy (%)"); ax[1].set_title("(b) Accuracy")
labels, ntr, cc = [], [], []
for c in (2, 8):
    for p in ("bank", "ecorecon"):
        RR = [L(s, 6, c, p) for s in range(3)]
        labels.append(f"{NAME[p]}\nC = {c}"); ntr.append(np.mean([r["n_train"] for r in RR])); cc.append(np.mean([r["co2_adapt_g"] for r in RR]) * 1000)
b = ax[2].bar(range(4), cc, color=["#E69F00", "#E69F00", "#0072B2", "#0072B2"])
for i, (v, n) in enumerate(zip(cc, ntr)):
    ax[2].text(i, v + 80, f"{n:.0f} trains", ha="center", fontsize=6.5)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(labels, fontsize=6.5); ax[2].set_ylabel("Adapt. CO$_2$ (mg)")
ax[2].set_ylim(0, max(cc) * 1.18); ax[2].set_title("(c) Bank capacity")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig5_sensitivity.png")); plt.close(fig)
print("figures:", sorted(os.listdir(FIG)))
