"""Run every job in jobs.txt inside ONE process (Colab-friendly). Each finished run is saved at once as
results/run_s{seed}_pre0_D{D}_C{C}_{policy}.json; finished runs are skipped when restarted."""
import os, json, time, sys
import numpy as np, torch
import stream
from stream import make_stream, run, calibrate_tau
from common import Net, count_macs, RES, save_json, DEVICE

jobs = [l.split() for l in open(os.environ.get("JOBS_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs.txt"))) if l.strip()]
base_state = torch.load(os.path.join(RES, "base_seed0.pt"), map_location=DEVICE)
tmp = Net().to(DEVICE); tmp.load_state_dict(base_state); macs = count_macs(tmp)
trunk_flops = 2 * (macs["all"] - macs["module"])
tau_file = os.path.join(RES, "tau_pre0.json")
if os.path.exists(tau_file):
    tau = json.load(open(tau_file))["tau"]
else:
    tau, ds = calibrate_tau(base_state, trunk_flops, macs)
    save_json(dict(tau=tau, clean_p99=float(np.percentile(ds, 99)), clean_max=float(max(ds))), "tau_pre0.json")
print("device", DEVICE, "tau", round(tau, 4), "jobs", len(jobs), flush=True)
t_all = time.time(); done = 0
for k, (dd, cc, s, p) in enumerate(jobs):
    d = int(dd.split("=")[1]); c = int(cc.split("=")[1]); s = int(s)
    tag = f"s{s}_pre0_D{d}_C{c}"
    out = os.path.join(RES, f"run_{tag}_{p}.json")
    if os.path.exists(out):
        continue
    t0 = time.time()
    r = run(p, make_stream(s), s, base_state, tau, macs, trunk_flops, d_defer=d, cap=c)
    r.update(seed=s, pre_seed=0, d_defer=d, cap=c, tau=tau, start_day=make_stream(s)["start_day"], sec=time.time() - t0,
             device=str(DEVICE))
    save_json(r, os.path.basename(out)); done += 1
    el = time.time() - t_all
    print(f"[{k+1}/{len(jobs)}] {tag} {p:15s} acc={r['mean_acc']:.4f} adaptGF={r['adapt_gflop']:.1f} "
          f"co2a={r['co2_adapt_g']:.4f}g ntrain={r['n_train']} bank={r['bank_size']} {r['sec']:.0f}s | elapsed {el/60:.1f} min",
          flush=True)
print("ALL_JOBS_FINISHED", flush=True)
