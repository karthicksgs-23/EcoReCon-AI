"""EcoReCon-AI experiments: shared data, corruption, model and cost utilities (CPU only)."""
import os, pickle, tarfile, json, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get("DATA_DIR", os.path.join(HERE, "data"))
RES = os.environ.get("RESULTS_DIR", os.path.join(HERE, "results"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1" else "cpu")
os.makedirs(RES, exist_ok=True)
torch.set_num_threads(int(os.environ.get("THREADS", 8)))

MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)


# ---------------------------------------------------------------- data
def load_cifar():
    """Returns (xtr uint8 N,32,32,3, ytr, xte, yte) from the official python tarball."""
    cache = os.path.join(DATA, "cifar10.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["xtr"], z["ytr"], z["xte"], z["yte"]
    tf = tarfile.open(os.path.join(DATA, "cifar-10-python.tar.gz"))
    def batch(name):
        d = pickle.load(tf.extractfile(f"cifar-10-batches-py/{name}"), encoding="bytes")
        return d[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(d[b"labels"])
    xs, ys = zip(*[batch(f"data_batch_{i}") for i in range(1, 6)])
    xtr, ytr = np.concatenate(xs), np.concatenate(ys)
    xte, yte = batch("test_batch")
    np.savez_compressed(cache + ".tmp.npz", xtr=xtr, ytr=ytr, xte=xte, yte=yte)
    os.replace(cache + ".tmp.npz", cache)
    return xtr, ytr, xte, yte


# ---------------------------------------------------------------- recurring contexts (edge camera conditions)
CONTEXTS = ["clean", "low_light", "noise", "blur", "haze"]


def _gauss_kernel(sigma, size=7):
    ax = np.arange(size) - size // 2
    k = np.exp(-ax ** 2 / (2 * sigma ** 2)); k /= k.sum()
    return k


def corrupt(x, ctx, rng):
    """x: float32 N,32,32,3 in [0,1]. Fixed mid-level severity per context."""
    if ctx == "clean":
        y = x
    elif ctx == "low_light":          # night-time: dark, low contrast, sensor noise
        y = 0.30 * x ** 1.2 + rng.normal(0, 0.03, x.shape)
    elif ctx == "noise":              # high-gain / rain-on-lens style sensor noise
        y = x + rng.normal(0, 0.10, x.shape)
    elif ctx == "blur":               # motion / defocus blur
        k = _gauss_kernel(1.3)
        t = torch.from_numpy(x).permute(0, 3, 1, 2)
        kk = torch.from_numpy(k).float()
        t = F.conv2d(F.pad(t.reshape(-1, 1, 32, 32), (3, 3, 0, 0), mode="reflect"), kk.view(1, 1, 1, 7))
        t = F.conv2d(F.pad(t, (0, 0, 3, 3), mode="reflect"), kk.view(1, 1, 7, 1))
        y = t.reshape(-1, 3, 32, 32).permute(0, 2, 3, 1).numpy()
    elif ctx == "haze":               # fog / dust haze: blend towards grey airlight
        y = 0.45 * x + 0.55 * 0.75
    else:
        raise ValueError(ctx)
    return np.clip(y, 0, 1).astype(np.float32)


def to_tensor(x01):
    return torch.from_numpy(((x01 - MEAN) / STD).transpose(0, 3, 1, 2).copy()).to(DEVICE)


# ---------------------------------------------------------------- model: ResNet-10-ish, split into shared trunk + reconfigurable module
class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = None
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        o = F.relu(self.b1(self.c1(x)))
        o = self.b2(self.c2(o))
        return F.relu(o + (x if self.sc is None else self.sc(x)))


class Net(nn.Module):
    """trunk = stem + stage1 + stage2 (shared, frozen after pre-training);
    module = stage3 + head (swappable per context). BN layers everywhere."""
    def __init__(self, w=32, ncls=10):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.s1 = nn.Sequential(Block(w, w, 1), Block(w, w, 1))
        self.s2 = nn.Sequential(Block(w, 2 * w, 2), Block(2 * w, 2 * w, 1))
        self.s3 = nn.Sequential(Block(2 * w, 4 * w, 2), Block(4 * w, 4 * w, 1))
        self.head = nn.Linear(4 * w, ncls)

    def trunk(self, x):
        return self.s2(self.s1(self.stem(x)))

    def module(self, h):
        h = self.s3(h)
        return self.head(F.adaptive_avg_pool2d(h, 1).flatten(1))

    def forward(self, x):
        return self.module(self.trunk(x))


def module_keys(net):
    """state-dict keys that belong to the reconfigurable module (stage3 + head)."""
    return [k for k in net.state_dict() if k.startswith(("s3.", "head."))]


def bn_keys(net):
    return [k for k in net.state_dict() if ("running_" in k or "num_batches" in k)]


# ---------------------------------------------------------------- compute cost (MAC counting by forward hooks)
def count_macs(net, part="all"):
    """MACs for one 32x32 image: total forward, and the module (stage3+head) part."""
    macs = {"all": 0, "module": 0}
    hooks = []
    def mk(name):
        def h(m, i, o):
            if isinstance(m, nn.Conv2d):
                n = o.shape[1] * o.shape[2] * o.shape[3] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
            elif isinstance(m, nn.Linear):
                n = m.in_features * m.out_features
            else:
                return
            macs["all"] += n
            if name.startswith(("s3", "head")):
                macs["module"] += n
        return h
    for name, m in net.named_modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(mk(name)))
    net.eval()
    with torch.no_grad():
        net(torch.zeros(1, 3, 32, 32, device=next(net.parameters()).device))
    for h in hooks: h.remove()
    return macs


class Meter:
    """Counts FLOPs (2 x MACs). Backward ~ 2 x forward for the layers that are trained."""
    def __init__(self, macs):
        self.f_all = 2 * macs["all"]; self.f_mod = 2 * macs["module"]
        self.flops = 0.0
    def infer(self, n):               # forward of whole net
        self.flops += n * self.f_all
    def bn_recal(self, n):            # forward only (statistics), whole net
        self.flops += n * self.f_all
    def train_module(self, n):        # forward whole + backward through module only
        self.flops += n * (self.f_all + 2 * self.f_mod)
    def train_full(self, n):          # forward + backward whole net
        self.flops += n * 3 * self.f_all
    def signature(self, n):           # trunk forward is shared with inference; extra cost ~0
        pass


def save_json(obj, name):
    path = os.path.join(RES, name)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)           # atomic: a disconnect never leaves a half-written file
