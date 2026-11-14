"""Diagnostic: are contexts separable by the label-free signature at the clean-calibrated tau?
Reports within-context and between-context signature distances, and base accuracy per context."""
import os, json, numpy as np, torch
from common import load_cifar, corrupt, to_tensor, Net, CONTEXTS, count_macs, Meter, RES, save_json, DEVICE
from stream import Probe, calibrate_tau, EVAL_N

base_state = torch.load(os.path.join(RES, "base_seed0.pt"), map_location=DEVICE)
net = Net().to(DEVICE); net.load_state_dict(base_state); net.eval()
macs = count_macs(net); tf = 2 * (macs["all"] - macs["module"])
tau, ds = calibrate_tau(base_state, tf, macs)
save_json(dict(tau=tau, clean_p99=float(np.percentile(ds, 99)), clean_max=float(max(ds))), "tau_pre0.json")
probe = Probe(net, tf); meter = Meter(macs)
xtr, ytr, xte, yte = load_cifar()
xt = xte.astype(np.float32) / 255
rng = np.random.default_rng(3)
sigs = {c: [] for c in CONTEXTS}; acc = {}
for c in CONTEXTS:
    for r in range(8):
        i = rng.choice(len(xt), EVAL_N, replace=False)
        x = to_tensor(corrupt(xt[i], c, rng))
        sigs[c].append(probe.sig(x, meter))
        if r == 0:
            with torch.no_grad():
                acc[c] = (net(x).argmax(1).cpu().numpy() == yte[i]).mean()
within = {c: float(np.max([Probe.dist(a, b) for a in sigs[c] for b in sigs[c] if a is not b])) for c in CONTEXTS}
between = {f"{a}-{b}": float(np.min([Probe.dist(x, y) for x in sigs[a] for y in sigs[b]]))
           for i, a in enumerate(CONTEXTS) for b in CONTEXTS[i + 1:]}
out = dict(tau=tau, within_max=within, between_min=between, base_acc={k: float(v) for k, v in acc.items()})
save_json(out, "diag_tau.json"); print(json.dumps(out, indent=1))
