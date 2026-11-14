# EcoReCon-AI

Code and results for the paper *"EcoReCon-AI: Module Reuse, Not Carbon-Aware Deferral, Cuts the Cost of Adapting Edge Vision Models to Recurring Drift"* (ICNGMR 2026).

An edge camera sees conditions that come back every day (night-time low light, morning haze, blur, sensor noise). EcoReCon-AI keeps a bank of small context modules on a frozen feature trunk. Every hour it computes a label-free signature of the incoming batch and chooses to **reuse** a stored module, **adapt now**, **defer** training to the cleanest forecast hour of an Indian grid carbon-intensity trace, or **expand** the bank. Seven policies are compared on identical simulated 240-hour streams.

## Main results (5 streams, mean ± s.d.)

| Policy | Accuracy (%) | Adapt. compute (TFLOP) | Adapt. CO₂ (g)* | Training events |
|---|---|---|---|---|
| Static | 48.8 ± 1.3 | 0 | 0 | 0 |
| BN-adapt | 76.2 ± 0.8 | 20.30 | 4.13 ± 0.08 | 0 |
| Periodic-full | 74.1 ± 0.4 | 17.54 | 3.86 | 9 |
| Drift-FT | 75.3 ± 0.6 | 26.53 ± 0.65 | 5.41 ± 0.11 | 39.0 |
| Drift-FT-defer | 73.5 ± 0.7 | 33.70 ± 1.51 | 6.68 ± 0.22 | 35.4 |
| Bank | 74.8 ± 1.0 | 15.41 | 3.15 ± 0.06 | 4 |
| EcoReCon | 75.5 ± 0.9 | 16.46 ± 0.28 | 3.35 ± 0.12 | 4 |

\*Energy is a FLOP count at an assumed 1 nJ per FLOP, so absolute grams are illustrative; compare ratios between policies.

- Reuse cut training from 39 to 4 events per stream and adaptation emissions by 34–40% at equal accuracy.
- Carbon-aware deferral lowered the carbon intensity of adaptation but raised emissions by 2–12%, because the node recalibrated batch-normalisation statistics every hour while jobs waited.
- With reuse in place, the hourly drift check was about 90% of adaptation compute.

Full tables: `results/table_main.md`, `results/aggregate.json`. Figures: `results/figs/`.

## Run it

**Google Colab (recommended).** Open `EcoReCon_colab.ipynb`, choose a **T4 GPU** runtime and *Run all*; allow Google Drive access when asked. On a T4 the run used for the paper took about 16 s per pre-training epoch and 18–35 s per stream run, roughly 30 minutes in total. Results go to `MyDrive/EcoReCon_results` and downloaded data to `MyDrive/EcoReCon_data`. Pre-training saves a checkpoint after every epoch and every stream run is saved when it finishes, so after a disconnect you run the notebook again and it continues.

**Locally** (a GPU is strongly recommended; on CPU one stream run takes several minutes):

```bash
pip install -r requirements.txt
# put cifar-10-python.tar.gz (https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) in data/
python carbon.py            # India hourly carbon-intensity trace from data/nasa_power_hourly_Nagpur_2023.json
python pretrain.py 0        # base network (15 epochs)
python diag_tau.py          # clean-data threshold tau and context-separability check
python make_jobs.py         # 50 jobs: 35 main + 15 sensitivity
python run_all.py           # all stream runs (finished runs are skipped)
python aggregate.py         # tables + Figs 2-3
python make_paper_figs.py   # Figs 4-5
python fig1_diagram.py      # Fig 1 (method diagram)
```

The `results/` folder already holds the outputs of the Colab run used in the paper, including the trained base model (`base_seed0.pt`). Because `run_all.py` skips runs whose result file exists, delete the `run_*.json` files (and `base_seed0.pt`, `pretrain_seed0.json`, `tau_pre0.json`) to recompute everything. GPU runs are not bit-for-bit deterministic, so a rerun gives close but not identical numbers.

## Files

| File | Purpose |
|---|---|
| `common.py` | CIFAR-10 loading, five camera contexts, ResNet-style trunk + module network, FLOP meter |
| `carbon.py` | Modelled hourly carbon intensity for India (NASA POWER irradiance, CEA mean 0.727 kg/kWh, Ember solar share) |
| `pretrain.py` | Base network on 40,000 clean images; checkpoint after every epoch |
| `stream.py` | 240-hour stream simulator and the seven policies |
| `diag_tau.py` | Threshold τ from clean data and separability check |
| `make_jobs.py`, `jobs.txt`, `run_all.py` | Job list and resumable runner |
| `aggregate.py`, `make_paper_figs.py`, `fig1_diagram.py` | Tables and figures (`aggregate.py` also writes a preliminary `fig4_sensitivity.png` that the paper does not use; the paper's Figs 4–5 come from `make_paper_figs.py`) |
| `EcoReCon_colab.ipynb`, `build_colab_notebook.py` | Self-contained Colab notebook and the script that builds it |
| `data/nasa_power_hourly_Nagpur_2023.json` | NASA POWER hourly irradiance used for the carbon trace |
| `results/` | All 50 run files, aggregate tables, base model, threshold, figures |

## Data sources

- CIFAR-10: A. Krizhevsky, *Learning Multiple Layers of Features from Tiny Images*, Tech. Rep., University of Toronto (2009).
- NASA Langley Research Center POWER Project, Hourly API (ALLSKY_SFC_SW_DWN), https://power.larc.nasa.gov.
- Central Electricity Authority, *CO₂ Baseline Database for the Indian Power Sector*, Version 20.0 (2024); Ember, *Global Electricity Review 2025*.

## Limitations

Streams are simulated from 32 × 32 CIFAR-10 images with five fixed-severity corruptions that change abruptly. Energy is a FLOP proxy and the carbon trace is a modelled diurnal shape, not metered data. One base model, five streams for the main comparison and three for the sensitivity analysis.

## Citation

[Author names], "EcoReCon-AI: Module Reuse, Not Carbon-Aware Deferral, Cuts the Cost of Adapting Edge Vision Models to Recurring Drift", in *International Conference on Next Generation Machine Learning and Reconfigurable Intelligence (ICNGMR 2026)*, ITM Web of Conferences (2026).

## Licence

MIT, see `LICENSE`.
