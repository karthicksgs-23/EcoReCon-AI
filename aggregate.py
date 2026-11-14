"""Aggregate stream results -> tables (json/csv/md) and figures for the EcoReCon paper."""
import os, glob, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
FIG = os.path.join(RES, "figs"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 8,
                     "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "savefig.dpi": 300,
                     "savefig.bbox": "tight"})
OI = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
POL = ["static", "bn_adapt", "periodic_full", "drift_ft", "drift_ft_defer", "bank", "ecorecon"]
NAME = {"static": "Static", "bn_adapt": "BN-adapt", "periodic_full": "Periodic-full", "drift_ft": "Drift-FT",
        "drift_ft_defer": "Drift-FT-defer", "bank": "Bank", "ecorecon": "EcoReCon"}
CTX = ["clean", "low_light", "noise", "blur", "haze"]


def load(d=6, c=8):
    out = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(RES, f"run_s*_pre0_D{d}_C{c}_*.json")):
        r = json.load(open(f)); out[r["policy"]][r["seed"]] = r
    return out


def retrain_on_recurrence(r):
    """Training events whose context had already been trained on earlier in the stream."""
    seen, rep, tot = set(), 0, 0
    for x in r["records"]:
        if x["action"] in ("train", "train_full"):
            tot += 1
            if x["ctx"] in seen: rep += 1
            seen.add(x["ctx"])
    return rep, tot


def ms(v):
    v = np.asarray(v, float); return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


main = load()
seeds = sorted(set.intersection(*[set(main[p]) for p in POL if p in main])) if all(p in main for p in POL) else []
print("policies:", {p: sorted(main[p]) for p in main}, "common seeds:", seeds)
summ = {}
for p in POL:
    if p not in main: continue
    S = [main[p][s] for s in seeds] if seeds else list(main[p].values())
    reps = [retrain_on_recurrence(r) for r in S]
    acts = collections.Counter(x["action"] for r in S for x in r["records"])
    summ[p] = dict(n=len(S), acc=ms([r["mean_acc"] for r in S]), gflop=ms([r["adapt_gflop"] for r in S]),
                   co2=ms([r["co2_adapt_g"] for r in S]), co2_total=ms([r["co2_total_g"] for r in S]),
                   ci=ms([r["mean_ci_of_adapt"] for r in S if r["adapt_gflop"] > 0] or [0]),
                   ntrain=ms([r["n_train"] for r in S]), retrain_rep=ms([a for a, b in reps]),
                   bank=ms([r["bank_size"] or 0 for r in S]),
                   acc_ctx={c: ms([r["acc_by_ctx"].get(c, np.nan) for r in S]) for c in CTX},
                   actions={k: v / len(S) for k, v in acts.items()},
                   per_seed=dict(acc=[r["mean_acc"] for r in S], co2=[r["co2_adapt_g"] for r in S],
                                 gflop=[r["adapt_gflop"] for r in S], ntrain=[r["n_train"] for r in S]))
# paired differences vs EcoReCon
pairs = {}
if "ecorecon" in summ:
    E = summ["ecorecon"]["per_seed"]
    for p in ("drift_ft", "drift_ft_defer", "bank", "periodic_full", "bn_adapt"):
        if p not in summ: continue
        O = summ[p]["per_seed"]
        pairs[p] = dict(d_acc_pp=[100 * (a - b) for a, b in zip(E["acc"], O["acc"])],
                        co2_ratio=[a / b if b else None for a, b in zip(E["co2"], O["co2"])],
                        gflop_ratio=[a / b if b else None for a, b in zip(E["gflop"], O["gflop"])])
# context occupancy
ctx_share = collections.Counter(x["ctx"] for s in seeds for x in main["static"][s]["records"]) if "static" in main else {}
tot = sum(ctx_share.values()) or 1
# sensitivity
sens = {}
for d in (3, 6, 12, 24):
    R = load(d, 8).get("ecorecon", {})
    if R: sens[f"D{d}"] = dict(n=len(R), acc=ms([r["mean_acc"] for r in R.values()]), co2=ms([r["co2_adapt_g"] for r in R.values()]),
                              ci=ms([r["mean_ci_of_adapt"] for r in R.values()]))
for c in (2, 3, 8):
    for p in ("bank", "ecorecon"):
        R = load(6, c).get(p, {})
        if R: sens[f"C{c}_{p}"] = dict(n=len(R), acc=ms([r["mean_acc"] for r in R.values()]), co2=ms([r["co2_adapt_g"] for r in R.values()]),
                                      ntrain=ms([r["n_train"] for r in R.values()]))
tau = json.load(open(os.path.join(RES, "tau_pre0.json")))
pre = json.load(open(os.path.join(RES, "pretrain_seed0.json")))
diag = json.load(open(os.path.join(RES, "diag_tau.json")))
out = dict(seeds=seeds, summary=summ, pairs=pairs, ctx_share={k: v / tot for k, v in ctx_share.items()}, sensitivity=sens,
           tau=tau, pretrain=dict(final_acc=pre["log"][-1]["test_acc"], params=pre["params"], macs=pre["macs"]), diag=diag)
json.dump(out, open(os.path.join(RES, "aggregate.json"), "w"), indent=1)

# ---- markdown table preview
lines = ["| Policy | Acc. (%) | Adapt. GFLOP | Adapt. CO2 (g) | Mean CI of adapt. | Train events | Re-trains on recurrence |"]
for p in POL:
    if p not in summ: continue
    s = summ[p]
    lines.append(f"| {NAME[p]} | {100*s['acc'][0]:.1f} ± {100*s['acc'][1]:.1f} | {s['gflop'][0]:.0f} ± {s['gflop'][1]:.0f} | "
                 f"{s['co2'][0]:.3f} ± {s['co2'][1]:.3f} | {s['ci'][0]:.3f} | {s['ntrain'][0]:.1f} | {s['retrain_rep'][0]:.1f} |")
open(os.path.join(RES, "table_main.md"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
for p, v in pairs.items():
    print("EcoReCon vs", p, "dAcc pp", np.round(v["d_acc_pp"], 2), "CO2 ratio", np.round([x for x in v["co2_ratio"] if x], 3))
print("sens", json.dumps(sens, indent=0)[:1500])

# ---- Fig 2: accuracy vs adaptation CO2
if summ:
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for i, p in enumerate(POL):
        if p not in summ: continue
        s = summ[p]
        ax.errorbar(s["co2"][0] * 1000, 100 * s["acc"][0], xerr=s["co2"][1] * 1000, yerr=100 * s["acc"][1],
                    fmt="o", ms=4, color=OI[i], capsize=2, label=NAME[p])
    ax.set_xlabel("Adaptation emissions (mg CO$_2$, 240 h)"); ax.set_ylabel("Mean accuracy (%)")
    ax.legend(frameon=False, fontsize=6.5, loc="lower right"); ax.grid(alpha=0.25, lw=0.5)
    fig.savefig(os.path.join(FIG, "fig2_acc_vs_co2.png")); plt.close(fig)

# ---- Fig 3: timeline, seed 0, first 72 h
if seeds and "ecorecon" in main and "drift_ft" in main:
    s0 = seeds[0]; T = 72
    fig, ax = plt.subplots(4, 1, figsize=(7.0, 4.2), sharex=True, gridspec_kw=dict(height_ratios=[0.5, 1, 0.8, 1.4]))
    rec = main["ecorecon"][s0]["records"][:T]
    cmap = {c: col for c, col in zip(CTX, ["#FFFFFF", "#333333", "#999999", "#56B4E9", "#E69F00"])}
    for x in rec:
        ax[0].axvspan(x["t"], x["t"] + 1, color=cmap[x["ctx"]], lw=0)
    ax[0].set_yticks([]); ax[0].set_ylabel("Context", rotation=0, ha="right", va="center")
    from matplotlib.patches import Patch
    ax[0].legend(handles=[Patch(facecolor=cmap[c], edgecolor="0.5", label=c.replace("_", " ")) for c in CTX], ncol=5,
                 frameon=False, fontsize=6.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax[1].plot([x["t"] + 0.5 for x in rec], [x["ci"] for x in rec], color="k", lw=1)
    ax[1].set_ylabel("CI (kg/kWh)")
    for j, p in enumerate(["drift_ft", "bank", "ecorecon"]):
        rr = main[p][s0]["records"][:T]
        for x in rr:
            if x["action"] in ("train",):
                ax[2].plot(x["t"] + 0.5, j, "v", color=OI[[1, 5, 6][j]], ms=5)
            elif x["action"] == "reuse":
                ax[2].plot(x["t"] + 0.5, j, "|", color="0.6", ms=5)
    ax[2].set_yticks([0, 1, 2]); ax[2].set_yticklabels(["Drift-FT", "Bank", "EcoReCon"]); ax[2].set_ylim(-0.6, 2.6)
    for j, p in enumerate(["static", "drift_ft", "ecorecon"]):
        rr = main[p][s0]["records"][:T]
        ax[3].step([x["t"] for x in rr], [100 * x["acc"] for x in rr], where="post", color=OI[[0, 1, 6][j]], lw=0.9, label=NAME[p])
    ax[3].set_ylabel("Accuracy (%)"); ax[3].set_xlabel("Hour of stream"); ax[3].legend(frameon=False, ncol=3, fontsize=6.5)
    for a in ax: a.set_xlim(0, T)
    fig.savefig(os.path.join(FIG, "fig3_timeline.png")); plt.close(fig)

# ---- Fig 4: sensitivity
if sens:
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    ds = [d for d in (3, 6, 12, 24) if f"D{d}" in sens]
    if ds:
        a2 = ax[0].twinx()
        ax[0].errorbar(ds, [100 * sens[f"D{d}"]["acc"][0] for d in ds], yerr=[100 * sens[f"D{d}"]["acc"][1] for d in ds], fmt="-o", color=OI[5], ms=3, capsize=2)
        a2.errorbar(ds, [1000 * sens[f"D{d}"]["co2"][0] for d in ds], yerr=[1000 * sens[f"D{d}"]["co2"][1] for d in ds], fmt="--s", color=OI[6], ms=3, capsize=2)
        if "bank" in summ:
            ax[0].axhline(100 * summ["bank"]["acc"][0], color=OI[5], lw=0.6, ls=":")
            a2.axhline(1000 * summ["bank"]["co2"][0], color=OI[6], lw=0.6, ls=":")
        ax[0].set_xlabel("Deferral window D (h)"); ax[0].set_ylabel("Accuracy (%)", color=OI[5]); a2.set_ylabel("Adapt. CO$_2$ (mg)", color=OI[6])
        ax[0].set_xticks(ds); ax[0].set_title("(a) Deferral window (dotted: Bank, D = 0)")
    cs = [c for c in (2, 3, 8) if f"C{c}_ecorecon" in sens]
    if cs:
        w = 0.35; x = np.arange(len(cs))
        for k, p in enumerate(("bank", "ecorecon")):
            v = [100 * sens[f"C{c}_{p}"]["acc"][0] if f"C{c}_{p}" in sens else np.nan for c in cs]
            e = [100 * sens[f"C{c}_{p}"]["acc"][1] if f"C{c}_{p}" in sens else 0 for c in cs]
            ax[1].bar(x + (k - 0.5) * w, v, w, yerr=e, color=[OI[5], OI[6]][k], capsize=2, label=NAME[p])
        ax[1].set_xticks(x); ax[1].set_xticklabels([f"C = {c}" for c in cs]); ax[1].set_ylabel("Accuracy (%)")
        lo = min(100 * sens[f"C{c}_{p}"]["acc"][0] for c in cs for p in ("bank", "ecorecon") if f"C{c}_{p}" in sens)
        ax[1].set_ylim(max(0, lo - 8), None); ax[1].legend(frameon=False); ax[1].set_title("(b) Module-bank capacity")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig4_sensitivity.png")); plt.close(fig)
print("figures written")
