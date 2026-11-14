"""Write jobs.txt: one line per (env, seed, policy) stream run. Main grid first, then sensitivity."""
import os
from stream import ALL_POLICIES
lines = []
for s in range(5):
    for p in ALL_POLICIES:
        lines.append(f"D_DEFER=6 CAP=8 {s} {p}")
for d in (3, 12, 24):                 # D=0 is identical to 'bank' by construction (best slot of [t,t] is t)
    for s in range(3):                # shorter plan (user, 2026-11-11): sensitivity on seeds 0-2
        lines.append(f"D_DEFER={d} CAP=8 {s} ecorecon")
for c in (2,):                        # shorter plan: CAP=2 only
    for s in range(3):
        for p in ("bank", "ecorecon"):
            lines.append(f"D_DEFER=6 CAP={c} {s} {p}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs.txt"), "w").write("\n".join(lines) + "\n")
print(len(lines), "jobs")
