"""Recurring-drift deployment simulator for EcoReCon-AI.

A camera node runs for H hours. Each hour has one operating context (night low-light, haze, blur,
noise, clean) that recurs daily. Per hour the node sees an unlabelled batch (used for inference,
the context signature and BN recalibration) and receives LAB_PER_H labelled field samples of the
current context (delayed annotation). Policies decide whether to reuse, adapt now, defer adaptation
to a low-carbon hour, or expand the module bank. Every policy sees identical data.

Usage: python stream.py <stream_seed> [policy ...]     (env D_DEFER, CAP, PRETRAIN_SEED)
"""
import sys, os, copy, time, json
import numpy as np, torch, torch.nn.functional as F
from common import (load_cifar, corrupt, to_tensor, Net, CONTEXTS, module_keys, count_macs, Meter, RES, DATA,
                    save_json, DEVICE)

H = int(os.environ.get("HOURS", 240))
EVAL_N, LAB_PER_H, N_MIN, BUF = 400, 64, 128, 1024
EPOCHS_MOD, EPOCHS_FULL = 4, 2
D_DEFER = int(os.environ.get("D_DEFER", 6))
CAP = int(os.environ.get("CAP", 8))
EPS_J_PER_FLOP = 1e-9          # assumed edge-CPU efficiency: 1 GFLOP per joule (only scales absolute numbers)
PRE_SEED = int(os.environ.get("PRETRAIN_SEED", 0))

ALL_POLICIES = ["static", "bn_adapt", "periodic_full", "drift_ft", "drift_ft_defer", "bank", "ecorecon"]


# ------------------------------------------------------------------ stream construction
def make_stream(seed):
    rng = np.random.default_rng(1000 + seed)
    start_day = int(rng.integers(0, 365 - H // 24 - 1))
    ci_year = np.load(os.path.join(DATA, "ci_india_2023.npy"))
    ci = ci_year[start_day * 24: start_day * 24 + H]
    ci_prev = ci_year[start_day * 24 - 24: start_day * 24 + H - 24] if start_day > 0 else np.roll(ci, 24)
    ctx = []
    day_ctx = "clean"
    for h in range(H):
        hod = h % 24
        if hod >= 19 or hod < 6:                       # night
            c = "low_light" if rng.random() < 0.9 else "noise"
        else:
            if hod == 6 or rng.random() < 0.25:        # day episodes: change ~every 4 h
                day_ctx = rng.choice(["clean", "haze", "blur", "noise"], p=[0.45, 0.25, 0.15, 0.15])
                if hod in (6, 7) and rng.random() < 0.5:
                    day_ctx = "haze"                   # morning haze/fog
            c = day_ctx
        ctx.append(str(c))
    return dict(start_day=start_day, ci=ci, ci_forecast=ci_prev, ctx=ctx, rng_seed=seed)


class Data:
    def __init__(self, seed):
        xtr, ytr, xte, yte = load_cifar()
        field = np.load(os.path.join(RES, "field_idx.npy"))
        self.xf, self.yf = xtr[field].astype(np.float32) / 255, ytr[field]
        self.xt, self.yt = xte.astype(np.float32) / 255, yte
        self.rng = np.random.default_rng(2000 + seed)

    def hour(self, ctx):
        i = self.rng.choice(len(self.xt), EVAL_N, replace=False)
        j = self.rng.choice(len(self.xf), LAB_PER_H, replace=False)
        xe = to_tensor(corrupt(self.xt[i], ctx, self.rng)); ye = torch.from_numpy(self.yt[i]).long().to(DEVICE)
        xl = to_tensor(corrupt(self.xf[j], ctx, self.rng)); yl = torch.from_numpy(self.yf[j]).long().to(DEVICE)
        return xe, ye, xl, yl


# ------------------------------------------------------------------ model helpers
def set_trunk_bn_train(net, on):
    for m in [net.stem, net.s1, net.s2, net.s3]:
        for mod in m.modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                mod.train(on)
                if on:
                    mod.momentum = None            # cumulative average over the given data
                    mod.reset_running_stats()


def bn_recal(net, x, meter):
    set_trunk_bn_train(net, True)
    with torch.no_grad():
        for i in range(0, len(x), 200):
            net(x[i:i + 200])
    net.eval(); meter.bn_recal(len(x))


def accuracy(net, x, y, meter):
    net.eval()
    with torch.no_grad():
        p = net(x).argmax(1)
    meter.infer(len(x))
    return (p == y).float().mean().item()


def train_module(net, x, y, meter, epochs=EPOCHS_MOD, seed=0):
    """Fine-tune stage3 + head on labelled context data; trunk BN statistics recalibrated first."""
    g = torch.Generator().manual_seed(seed)
    bn_recal(net, x, meter)
    for p in net.parameters(): p.requires_grad_(False)
    params = list(net.s3.parameters()) + list(net.head.parameters())
    for p in params: p.requires_grad_(True)
    opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=5e-4)
    net.eval(); net.s3.train()                     # trunk frozen incl. its BN stats; module BN adapts
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for b in range(0, len(x), 64):
            idx = perm[b:b + 64]
            loss = F.cross_entropy(net(x[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    meter.train_module(len(x) * epochs)
    for p in net.parameters(): p.requires_grad_(True)
    net.eval()


def train_full(net, x, y, meter, epochs=EPOCHS_FULL, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(x), generator=g)
        for b in range(0, len(x), 64):
            idx = perm[b:b + 64]
            loss = F.cross_entropy(net(x[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    meter.train_full(len(x) * epochs)
    net.eval()


# ------------------------------------------------------------------ label-free context signature
class Probe:
    """Per-channel mean/std of the frozen base trunk output (clean BN statistics)."""
    def __init__(self, base, meter_trunk_flops):
        self.net = copy.deepcopy(base).eval(); self.f = meter_trunk_flops

    def sig(self, x, meter):
        with torch.no_grad():
            h = self.net.trunk(x)                     # N,64,8,8
        meter.flops += len(x) * self.f
        m = h.mean((0, 2, 3)); s = h.std((0, 2, 3)) + 1e-4
        return torch.cat([m, s.log()])

    @staticmethod
    def dist(a, b):
        return (a - b).abs().mean().item()


# ------------------------------------------------------------------ policies
def best_slot(stream, t, deadline):
    """Hour in [t, deadline] with the lowest forecast carbon intensity (forecast = same hour yesterday)."""
    hi = min(deadline, H - 1)
    cand = range(t, hi + 1)
    return min(cand, key=lambda k: stream["ci_forecast"][k])


def run(policy, stream, data_seed, base_state, tau, macs, trunk_flops, d_defer=D_DEFER, cap=CAP):
    torch.manual_seed(0)
    data = Data(data_seed)
    meter = Meter(macs)
    adapt_flops_t = np.zeros(H)                      # adaptation compute (excl. plain inference) per hour
    infer_flops_t = np.zeros(H)
    base = Net().to(DEVICE); base.load_state_dict(base_state); base.eval()
    probe = Probe(base, trunk_flops)
    net = copy.deepcopy(base)
    bank = []                                        # list of dict(sig, state, ctx_seen, last_used)
    buf = {}                                         # labelled buffer per detected cluster id
    pending = None                                   # dict(cluster, due, deadline)
    cur_cluster = None
    rec = []
    trained_events = 0
    sig_last = None
    hist_x, hist_y = [], []

    for t in range(H):
        ctx = stream["ctx"][t]
        xe, ye, xl, yl = data.hour(ctx)
        f0 = meter.flops
        action = "serve"

        if policy == "static":
            pass

        elif policy == "bn_adapt":                      # test-time BN recalibration every hour, label-free
            bn_recal(net, xe, meter); action = "bn"

        elif policy == "periodic_full":                  # full fine-tune at midnight on the last 24 h of labels
            hist_x.append(xl); hist_y.append(yl)
            hist_x, hist_y = hist_x[-24:], hist_y[-24:]
            if t % 24 == 0 and t > 0:
                train_full(net, torch.cat(hist_x), torch.cat(hist_y), meter, seed=t); trained_events += 1
                action = "train_full"

        elif policy in ("drift_ft", "drift_ft_defer"):   # single model, threshold-triggered in-place retrain
            s = probe.sig(xe, meter)
            if sig_last is None:
                sig_last = probe.sig(to_tensor(corrupt(data.xf[:400], "clean", np.random.default_rng(0))), meter)
            drift = Probe.dist(s, sig_last) > tau
            if drift and pending is None:
                pending = dict(sig=s, start=t, xs=[], ys=[], due=None)
                bn_recal(net, xe, meter); action = "bn"
            if pending is not None:
                if Probe.dist(s, pending["sig"]) <= tau:
                    pending["xs"].append(xl); pending["ys"].append(yl)
                elif pending["due"] is None:              # context moved on before a job was booked: restart
                    pending = dict(sig=s, start=t, xs=[xl], ys=[yl], due=None)
                # a booked (deferred) job is kept and runs at its due hour on the labels it already holds
                n = sum(len(v) for v in pending["xs"])
                if n >= N_MIN and pending["due"] is None:
                    pending["due"] = t if policy == "drift_ft" else best_slot(stream, t, t + d_defer)
                if pending["due"] is not None and t >= pending["due"]:
                    x = torch.cat(pending["xs"])[-BUF:]; y = torch.cat(pending["ys"])[-BUF:]
                    train_module(net, x, y, meter, seed=t); trained_events += 1
                    sig_last = pending["sig"]; pending = None; action = "train"
                elif action == "serve":
                    bn_recal(net, xe, meter); action = "bn"

        elif policy in ("bank", "ecorecon"):
            s = probe.sig(xe, meter)
            if not bank:                                  # base model = first module (clean context)
                s0 = probe.sig(to_tensor(corrupt(data.xf[:400], "clean", np.random.default_rng(0))), meter)
                bank.append(dict(sig=s0, state=copy.deepcopy(base.state_dict()), last=0, n_train=0))
            if pending is None:
                pending = []                              # list of expansion jobs (one per unknown context)
            d = [Probe.dist(s, m["sig"]) for m in bank]
            k = int(np.argmin(d))
            if d[k] <= tau:                               # REUSE a stored module (no training)
                net.load_state_dict(bank[k]["state"]); bank[k]["last"] = t; action = "reuse"
            else:                                         # unknown context -> serve nearest + BN recal, feed EXPAND job
                net.load_state_dict(bank[k]["state"]); bn_recal(net, xe, meter); action = "bn"
                dj = [Probe.dist(s, j["sig"]) for j in pending]
                if dj and min(dj) <= tau:
                    job = pending[int(np.argmin(dj))]
                else:
                    job = dict(sig=s, start=t, xs=[], ys=[], due=None, warm=k); pending.append(job)
                job["xs"].append(xl); job["ys"].append(yl)
            for job in list(pending):
                n = sum(len(v) for v in job["xs"])
                if n >= N_MIN and job["due"] is None:
                    job["due"] = t if policy == "bank" else best_slot(stream, t, t + d_defer)
                if job["due"] is not None and t >= job["due"]:
                    x = torch.cat(job["xs"])[-BUF:]; y = torch.cat(job["ys"])[-BUF:]
                    new = copy.deepcopy(base); new.load_state_dict(bank[job["warm"]]["state"])
                    train_module(new, x, y, meter, seed=t); trained_events += 1
                    entry = dict(sig=job["sig"], state=copy.deepcopy(new.state_dict()), last=t, n_train=1)
                    if len(bank) >= cap:                  # evict least recently used (never the base module 0)
                        j = 1 + int(np.argmin([m["last"] for m in bank[1:]]))
                        bank.pop(j)
                    bank.append(entry); pending[:] = [j_ for j_ in pending if j_ is not job]; action = "train"   # remove by identity (dicts hold tensors)
                    if Probe.dist(s, entry["sig"]) <= tau:   # current hour belongs to the new module: serve it
                        net.load_state_dict(entry["state"])
        else:
            raise ValueError(policy)

        adapt_flops_t[t] = meter.flops - f0
        acc = accuracy(net, xe, ye, meter)
        infer_flops_t[t] = 2 * macs["all"] * len(xe)
        rec.append(dict(t=t, ctx=ctx, acc=acc, action=action, ci=float(stream["ci"][t])))

    accs = np.array([r["acc"] for r in rec])
    co2_adapt = float((adapt_flops_t * EPS_J_PER_FLOP / 3.6e6 * stream["ci"]).sum() * 1000)   # grams
    co2_total = float(((adapt_flops_t + infer_flops_t) * EPS_J_PER_FLOP / 3.6e6 * stream["ci"]).sum() * 1000)
    out = dict(policy=policy, mean_acc=float(accs.mean()),
               acc_by_ctx={c: float(np.mean([r["acc"] for r in rec if r["ctx"] == c])) for c in CONTEXTS
                           if any(r["ctx"] == c for r in rec)},
               adapt_gflop=float(adapt_flops_t.sum() / 1e9), co2_adapt_g=co2_adapt, co2_total_g=co2_total,
               mean_ci_of_adapt=float((adapt_flops_t * stream["ci"]).sum() / max(adapt_flops_t.sum(), 1)),
               n_train=trained_events, bank_size=len(bank) if bank else None, records=rec)
    return out


def calibrate_tau(base_state, trunk_flops, macs):
    """tau from clean data only: 3x the 99th percentile of clean batch-to-batch signature distances."""
    base = Net().to(DEVICE); base.load_state_dict(base_state)
    probe = Probe(base, trunk_flops); meter = Meter(macs)
    xtr, ytr, xte, yte = load_cifar()
    field = np.load(os.path.join(RES, "field_idx.npy"))
    xf = xtr[field].astype(np.float32) / 255
    rng = np.random.default_rng(7)
    ref = probe.sig(to_tensor(xf[:400]), meter)
    ds = [Probe.dist(probe.sig(to_tensor(xf[rng.choice(len(xf), EVAL_N, replace=False)]), meter), ref) for _ in range(200)]
    return 3 * float(np.percentile(ds, 99)), ds


if __name__ == "__main__":
    seed = int(sys.argv[1])
    pols = sys.argv[2:] or ALL_POLICIES
    base_state = torch.load(os.path.join(RES, f"base_seed{PRE_SEED}.pt"), map_location=DEVICE)
    tmp = Net(); tmp.load_state_dict(base_state)
    macs = count_macs(tmp)
    trunk_flops = 2 * (macs["all"] - macs["module"])
    tau_file = os.path.join(RES, f"tau_pre{PRE_SEED}.json")
    if os.path.exists(tau_file):
        tau = json.load(open(tau_file))["tau"]
    else:
        tau, ds = calibrate_tau(base_state, trunk_flops, macs)
        save_json(dict(tau=tau, clean_p99=float(np.percentile(ds, 99)), clean_max=float(max(ds))), os.path.basename(tau_file))
    stream = make_stream(seed)
    tag = f"s{seed}_pre{PRE_SEED}_D{D_DEFER}_C{CAP}"
    for p in pols:
        t0 = time.time()
        r = run(p, stream, seed, base_state, tau, macs, trunk_flops)
        r.update(seed=seed, pre_seed=PRE_SEED, d_defer=D_DEFER, cap=CAP, tau=tau, start_day=stream["start_day"],
                 sec=time.time() - t0)
        save_json(r, f"run_{tag}_{p}.json")
        print(f"{tag} {p:15s} acc={r['mean_acc']:.4f} adaptGF={r['adapt_gflop']:.1f} co2a={r['co2_adapt_g']:.4f}g "
              f"ciAdapt={r['mean_ci_of_adapt']:.3f} ntrain={r['n_train']} bank={r['bank_size']} {r['sec']:.0f}s", flush=True)
