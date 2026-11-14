"""Build EcoReCon_colab.ipynb: embeds the experiment sources so the notebook is self-contained."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ["common.py", "carbon.py", "pretrain.py", "stream.py", "diag_tau.py", "make_jobs.py", "run_all.py",
       "aggregate.py", "fig1_diagram.py"]
files = {f: open(os.path.join(HERE, f), encoding="utf-8").read() for f in SRC}


def md(t): return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}
def code(t): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


cells = [md("""# EcoReCon-AI experiments (ICNGMR 2026 paper)
Recurring-drift module reuse + carbon-aware deferral on a 240-hour simulated edge-camera stream (CIFAR-10, five camera contexts, India grid carbon trace).

**How to run**
1. Runtime → Change runtime type → **T4 GPU** (PyTorch does not use a TPU here; CPU-only would take many hours).
2. Runtime → **Run all**. Allow Google Drive access when asked.
3. Expected time on a T4: about 1–1.5 h (pretraining ~10 min, 50 stream runs, aggregation).
4. **Checkpoints (safe against internet drops / Colab disconnects):** everything is saved on **Google Drive → MyDrive/EcoReCon_results** (results) and **MyDrive/EcoReCon_data** (downloaded data).
   - Pre-training saves a checkpoint after **every epoch** (model, optimiser, learning-rate schedule and random-number states) and resumes from the last finished epoch.
   - Each of the 50 stream runs is saved the moment it finishes; finished runs are skipped.
   - Data downloads are kept on Drive, so they are not repeated.
   - All files are written atomically, so a disconnect cannot leave a broken file.

   **After a disconnect: reconnect, choose T4 GPU again if asked, and press Run all.** It continues where it stopped. (A stream run that was half-way through restarts from its beginning; on a T4 that costs about a minute.)
5. At the end `EcoReCon_results.zip` downloads to your computer automatically.
"""),
code("""# 1. Setup: folders, Google Drive, GPU check
import os, sys, subprocess, hashlib, json, shutil, time
WORK = "/content/ecorecon"; os.makedirs(WORK, exist_ok=True)
try:
    from google.colab import drive
    drive.mount("/content/drive")
    RESULTS = "/content/drive/MyDrive/EcoReCon_results"
except Exception as e:
    print("Drive not mounted (", e, ") - results stay in /content only")
    RESULTS = os.path.join(WORK, "results")
DATA = os.path.join(os.path.dirname(RESULTS), "EcoReCon_data") if RESULTS.startswith("/content/drive") else os.path.join(WORK, "data")
for d in (RESULTS, DATA): os.makedirs(d, exist_ok=True)
os.environ.update(RESULTS_DIR=RESULTS, DATA_DIR=DATA, THREADS="2", EPOCHS="15", PYTHONUNBUFFERED="1")
import torch
print("torch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE (CPU only - will be very slow)")
print("results ->", RESULTS, "| data ->", DATA)

def sh(cmd):
    \"\"\"Run a shell command in the code folder and stream its output live.\"\"\"
    p = subprocess.Popen(cmd, shell=True, cwd=WORK, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=os.environ.copy())
    for line in p.stdout: print(line, end="", flush=True)
    if p.wait() != 0: raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
"""),
code("# 2. Write the experiment source files\nSOURCES = " + json.dumps(files, indent=0) + """
for name, text in SOURCES.items():
    open(os.path.join(WORK, name), "w", encoding="utf-8").write(text)
os.chdir(WORK); print("written:", sorted(SOURCES))
"""),
code("""# 3. Data: CIFAR-10 (official tarball, md5-checked) and the India hourly carbon-intensity trace (NASA POWER)
import urllib.request
tgz = os.path.join(DATA, "cifar-10-python.tar.gz")
if not os.path.exists(tgz):
    urllib.request.urlretrieve("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", tgz)
assert hashlib.md5(open(tgz, "rb").read()).hexdigest() == "c58f30108f718f92721af3b95e74349a", "CIFAR md5 mismatch"
pj = os.path.join(DATA, "nasa_power_hourly_Nagpur_2023.json")
if not os.path.exists(pj):
    url = ("https://power.larc.nasa.gov/api/temporal/hourly/point?parameters=ALLSKY_SFC_SW_DWN&community=RE"
           "&longitude=79.09&latitude=21.15&start=20230101&end=20231231&format=JSON&time-standard=LST")
    for a in range(4):
        try:
            urllib.request.urlretrieve(url, pj); break
        except Exception as e:
            print("retry", e); time.sleep(15)
sh("python carbon.py")
import numpy as np
ci = np.load(os.path.join(DATA, "ci_india_2023.npy")); print("carbon trace: hours", ci.shape[0], "mean", round(ci.mean(), 4), "(expected 0.727)")
from common import load_cifar; x = load_cifar(); print("CIFAR train", x[0].shape, "test", x[2].shape)
"""),
code("""# 4. Pre-train the base network (skipped if finished; resumes from the last saved epoch otherwise)
if os.path.exists(os.path.join(RESULTS, "base_seed0.pt")) and os.path.exists(os.path.join(RESULTS, "pretrain_seed0.json")):
    print("base model found - skipping pre-training")
else:   # resumes automatically from MyDrive/EcoReCon_results/ckpt_pretrain_seed0.pt if a checkpoint exists
    sh("python pretrain.py 0")
print(json.load(open(os.path.join(RESULTS, "pretrain_seed0.json")))["log"][-1])
"""),
code("""# 5. Clean-data threshold tau and context-separability check (no test labels used for tau)
sh("python diag_tau.py")
d = json.load(open(os.path.join(RESULTS, "diag_tau.json")))
print("tau =", round(d["tau"], 4), "| max within-context =", round(max(d["within_max"].values()), 4),
      "| min between-context =", round(min(d["between_min"].values()), 4))
print("SEPARABLE" if max(d["within_max"].values()) < d["tau"] < min(d["between_min"].values()) else "WARNING: contexts not separable at tau")
"""),
code("""# 6. All stream runs (35 main + 15 sensitivity). Each run is saved to Drive as soon as it finishes; re-running skips finished runs.
sh("python make_jobs.py && python run_all.py")
"""),
code("""# 7. Tables + figures, then zip and download to your computer
sh("python aggregate.py && python fig1_diagram.py")
shutil.copy(os.path.join(WORK, "figs", "fig1_ecorecon.png"), os.path.join(RESULTS, "figs", "fig1_ecorecon.png"))
zp = shutil.make_archive("/content/EcoReCon_results", "zip", RESULTS)
shutil.copy(zp, os.path.join(os.path.dirname(RESULTS), "EcoReCon_results.zip")) if RESULTS.startswith("/content/drive") else None
print(open(os.path.join(RESULTS, "table_main.md")).read())
from google.colab import files
files.download(zp)
"""),
]
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(os.path.dirname(HERE), "EcoReCon_colab.ipynb")
json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
print("wrote", out)
