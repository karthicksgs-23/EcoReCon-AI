"""Pre-train the base network on clean CIFAR-10 (40k images); keep 10k training images as the
'field pool' that supplies labelled samples during deployment. Saves a checkpoint after every epoch."""
import sys, time, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from common import load_cifar, to_tensor, Net, RES, save_json, count_macs, DEVICE

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
EPOCHS = int(os.environ.get("EPOCHS", 15))
torch.manual_seed(seed); rng = np.random.default_rng(seed)
xtr, ytr, xte, yte = load_cifar()
perm = np.random.default_rng(123).permutation(len(xtr))     # split fixed across seeds
pre_idx, field_idx = perm[:40000], perm[40000:]
np.save(os.path.join(RES, "field_idx.npy"), field_idx)
_sub = int(os.environ.get("PRETRAIN_SUBSET", 0))             # testing only: train on a subset
if _sub: pre_idx = pre_idx[:_sub]
X = xtr[pre_idx].astype(np.float32) / 255; Y = torch.from_numpy(ytr[pre_idx]).long().to(DEVICE)
Xte = to_tensor(xte.astype(np.float32) / 255); Yte = torch.from_numpy(yte).long().to(DEVICE)

net = Net().to(DEVICE)
opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
steps = EPOCHS * (len(X) // 128)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=steps, pct_start=0.25)
log = []
# ---- checkpoint: resume after a disconnect from the last finished epoch (identical result)
CKPT = os.path.join(RES, f"ckpt_pretrain_seed{seed}.pt")
start = 0
if os.path.exists(CKPT):
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    if ck.get("epochs") == EPOCHS and ck.get("n_train") == len(X):
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(ck["cuda_rng"])
        log, start = ck["log"], ck["epoch"]
        print(f"resuming pre-training after epoch {start}/{EPOCHS}", flush=True)
_stop = int(os.environ.get("TEST_INTERRUPT_EPOCH", 0))      # testing only: simulate a disconnect
for ep in range(start, EPOCHS):
    net.train(); t0 = time.time(); p = rng.permutation(len(X)); tot = 0
    for b in range(len(X) // 128):
        idx = p[b * 128:(b + 1) * 128]
        xb = X[idx]
        if True:  # augmentation: random crop (pad 4) + horizontal flip
            xb = np.pad(xb, ((0, 0), (4, 4), (4, 4), (0, 0)), mode="reflect")
            ox, oy = rng.integers(0, 9, 2)
            xb = xb[:, oy:oy + 32, ox:ox + 32]
            flip = rng.random(len(idx)) < 0.5
            xb[flip] = xb[flip, :, ::-1]
        out = net(to_tensor(np.ascontiguousarray(xb)))
        loss = F.cross_entropy(out, Y[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
    net.eval()
    with torch.no_grad():
        acc = sum((net(Xte[i:i + 1000]).argmax(1) == Yte[i:i + 1000]).sum().item() for i in range(0, 10000, 1000)) / 100
    log.append(dict(epoch=ep + 1, loss=tot / (len(X) // 128), test_acc=acc, sec=time.time() - t0))
    print(log[-1], flush=True)
    ck = dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), np_rng=rng.bit_generator.state,
              torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
              log=log, epoch=ep + 1, epochs=EPOCHS, n_train=len(X))
    torch.save(ck, CKPT + ".tmp"); os.replace(CKPT + ".tmp", CKPT)    # atomic
    if _stop and ep + 1 == _stop:
        print("TEST: simulated disconnect", flush=True); os._exit(3)
_bp = os.path.join(RES, f"base_seed{seed}.pt")
torch.save({k: v.cpu() for k, v in net.state_dict().items()}, _bp + ".tmp"); os.replace(_bp + ".tmp", _bp)
save_json(dict(seed=seed, epochs=EPOCHS, log=log, params=sum(p.numel() for p in net.parameters()),
               macs=count_macs(net)), f"pretrain_seed{seed}.json")
