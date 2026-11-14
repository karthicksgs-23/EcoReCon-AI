"""Fig. 1: EcoReCon-AI controller diagram."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7,
                     "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 100); ax.set_ylim(0, 46); ax.axis("off")


def box(x, y, w, h, txt, fc="#FFFFFF", bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.2", fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontweight="bold" if bold else "normal", linespacing=1.3)


def arr(x0, y0, x1, y1, txt=None, ox=0, oy=1.0, ha="center"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.8, color="#333333"))
    if txt: ax.text((x0 + x1) / 2 + ox, (y0 + y1) / 2 + oy, txt, ha=ha, va="bottom", fontsize=6.5, style="italic")


Y = 33; H = 10
box(1, Y, 16, H, "Hourly unlabelled\nbatch $X_t$ and\ndelayed labels $L_t$", fc="#F2F2F2")
box(21, Y, 15, H, "Frozen trunk $f_T$\n(stem, stages 1–2)", fc="#DCEBF7")
box(40, Y, 15, H, "Signature $s(X_t)$\nchannel mean and\nlog-std (128-d)", fc="#DCEBF7")
box(59, Y, 18, H, "Nearest bank entry\n$k^*=\\arg\\min_k\\, d(s, s_k)$", fc="#FFF2CC")
box(81, Y, 18, H, "REUSE $\\theta_{k^*}$\nno training", fc="#D9EAD3", bold=True)
box(59, 16, 18, 10, "Serve $\\theta_{k^*}$ + BN recal.;\nexpansion job collects\n$L_t$ until $N_{min}=128$", fc="#FCE5CD")
box(81, 21.5, 18, 8, "ADAPT NOW\n(Bank, $D=0$)", fc="#F4CCCC", bold=True)
box(81, 10, 18, 9, "DEFER to cleanest\nforecast hour in $[t, t+D]$\n(EcoReCon)", fc="#F4CCCC", bold=True)
box(59, 1, 18, 8, "Carbon forecast\n$CI_{u-24}$ (India trace)", fc="#F2F2F2")
box(5, 12, 40, 12, "Module bank $B$ (capacity $C$, LRU eviction)\n$\\theta_k$ = stage 3 + head + trunk BN statistics\nentry 0 = pre-trained model (never evicted)", fc="#EAD1DC")
arr(17, Y + H / 2, 21, Y + H / 2); arr(36, Y + H / 2, 40, Y + H / 2); arr(55, Y + H / 2, 59, Y + H / 2)
arr(77, Y + H / 2, 81, Y + H / 2, "$d\\leq\\tau$", oy=0.6)
arr(68, Y, 68, 26, "$d>\\tau$", ox=1.2, oy=-0.5, ha="left")
arr(77, 23, 81, 25.5); arr(77, 19, 81, 15.5)
arr(77, 5, 90, 10)
arr(59, 21, 45, 19, "fine-tune module,\nEXPAND bank", oy=0.3)
arr(25, 24, 60, 33, "stored signatures $s_k$", ox=-6, oy=0.5)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig1_ecorecon.png"), dpi=300, bbox_inches="tight")
print("ok")
