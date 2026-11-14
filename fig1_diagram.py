"""Fig. 1: EcoReCon-AI controller diagram.

Every box label is measured after drawing; the script stops if any text
extends past its box, so overlaps cannot slip into the figure unnoticed.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7,
                     "mathtext.fontset": "stix"})
fig, ax = plt.subplots(figsize=(7.2, 3.6)); ax.set_xlim(0, 100); ax.set_ylim(0, 50); ax.axis("off")
fig.canvas.draw()

EDGE = "#333333"
boxes = []  # (patch extent in data units, text artist)


def box(x, y, w, h, txt, fc="#FFFFFF", bold=False, fs=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=EDGE, lw=0.8))
    t = ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", linespacing=1.3)
    boxes.append(((x, y, w, h), t, txt))
    return dict(l=x, r=x + w, b=y, t=y + h, cx=x + w / 2, cy=y + h / 2)


def arr(p0, p1, conn="arc3"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=8, lw=0.8, color=EDGE,
                                 connectionstyle=conn, shrinkA=0, shrinkB=0))


def label(x, y, txt, ha="center", va="center"):
    ax.text(x, y, txt, ha=ha, va=va, fontsize=6.5, style="italic", linespacing=1.2)


# Row 1: hourly pipeline
Y, H = 37, 11
inp = box(0.5, Y, 16.5, H, "Hourly unlabelled\nbatch $X_t$ and\ndelayed labels $L_t$", fc="#F2F2F2")
trk = box(20, Y, 16, H, "Frozen trunk $f_T$\n(stem,\nstages 1–2)", fc="#DCEBF7")
sig = box(39, Y, 16, H, "Signature $s(X_t)$:\nchannel mean and\nlog-std (128-d)", fc="#DCEBF7")
nb = box(58, Y, 19, H, "Nearest bank entry\n$k^*=\\arg\\min_k\\, d(s, s_k)$", fc="#FFF2CC")
reu = box(82, Y, 17.5, H, "REUSE $\\theta_{k^*}$\n(no training)", fc="#D9EAD3", bold=True)

# Row 2: new-context path
srv = box(58, 16, 19, 12, "Serve $\\theta_{k^*}$ with BN\nrecalibration; collect\n$L_t$ until $N_{min}=128$", fc="#FCE5CD")
ada = box(82, 24.5, 17.5, 7.5, "ADAPT NOW\n(Bank, $D=0$)", fc="#F4CCCC", bold=True)
dfr = box(82, 9.5, 17.5, 12, "DEFER to cleanest\nforecast hour\nin $[t,\\, t+D]$\n(EcoReCon)", fc="#F4CCCC", bold=True, fs=6.6)
car = box(58, 0.5, 19, 9, "Carbon forecast\n$CI_{u-24}$ (India trace)", fc="#F2F2F2")
bank = box(2, 13, 39, 13, "Module bank $B$ (capacity $C$, LRU eviction)\n"
                         "$\\theta_k$ = stage 3 + head + trunk BN statistics\n"
                         "entry 0 = pre-trained model (never evicted)", fc="#EAD1DC")

# Arrows along row 1
arr((inp["r"], inp["cy"]), (trk["l"], trk["cy"]))
arr((trk["r"], trk["cy"]), (sig["l"], sig["cy"]))
arr((sig["r"], sig["cy"]), (nb["l"], nb["cy"]))
arr((nb["r"], nb["cy"]), (reu["l"], reu["cy"]))
label((nb["r"] + reu["l"]) / 2, nb["cy"] + 2.2, "$d\\leq\\tau$")

# New context: nearest entry -> serve
arr((nb["cx"], nb["b"]), (srv["cx"], srv["t"]))
label(nb["cx"] + 1.2, (nb["b"] + srv["t"]) / 2, "$d>\\tau$", ha="left")

# Serve -> ADAPT NOW / DEFER
arr((srv["r"], srv["cy"] + 2), (ada["l"], ada["cy"]))
arr((srv["r"], srv["cy"] - 2), (dfr["l"], dfr["cy"]))

# Carbon forecast feeds the DEFER decision
arr((car["r"], car["cy"]), (dfr["cx"], dfr["b"]), conn="angle,angleA=0,angleB=90,rad=0")

# Fine-tuned module goes into the bank
arr((srv["l"], srv["cy"]), (bank["r"], srv["cy"]))
label((bank["r"] + srv["l"]) / 2, srv["cy"] + 3.2, "fine-tune module,\nEXPAND bank")

# Stored signatures feed the nearest-entry search (routed under row 1)
gy = (Y + bank["t"]) / 2 + 0.5
sx, ex = bank["cx"], nb["l"] + 4
ax.plot([sx, sx, ex], [bank["t"], gy, gy], color=EDGE, lw=0.8, solid_joinstyle="miter")
arr((ex, gy), (ex, nb["b"]))
label(sx + 1.2, gy + 1.6, "stored signatures $s_k$", ha="left")

# Overlap check: every label must sit inside its box with a small margin
fig.canvas.draw()
r = fig.canvas.get_renderer()
inv = ax.transData.inverted()
bad = []
for (x, y, w, h), t, txt in boxes:
    bb = t.get_window_extent(r)
    (x0, y0), (x1, y1) = inv.transform([(bb.x0, bb.y0), (bb.x1, bb.y1)])
    if x0 < x + 0.6 or x1 > x + w - 0.6 or y0 < y + 0.4 or y1 > y + h - 0.4:
        bad.append(txt.splitlines()[0])
if bad:
    raise SystemExit(f"text does not fit in box: {bad}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs"); os.makedirs(out, exist_ok=True)
fig.savefig(os.path.join(out, "fig1_ecorecon.png"), dpi=300, bbox_inches="tight", pad_inches=0.03)
print("ok", matplotlib.font_manager.findfont("Times New Roman"))
