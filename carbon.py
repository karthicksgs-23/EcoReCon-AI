"""Modelled hourly carbon intensity of the Indian grid for 2023 (kg CO2 / kWh).

CI(h) = EF_fossil * (1 - c_other - s(h))
 - s(h): solar share of generation in hour h, proportional to NASA POWER all-sky surface irradiance at
   Nagpur (21.15 N, 79.09 E, central India, local solar time), scaled so its annual mean equals 7 %
   (Ember, Global Electricity Review 2025: solar ~7 % of India's 2024 generation).
 - c_other: non-solar clean share (hydro 8 % + wind/nuclear) = 22 % clean - 7 % solar = 15 %, held constant.
 - EF_fossil chosen so the annual mean of CI equals 0.727 kg/kWh (CEA CO2 Baseline Database v20,
   weighted average for FY 2023-24).
Demand is assumed flat over the day. This is an estimate of the diurnal shape, not measured data.
"""
import json, os
import numpy as np
from common import DATA, save_json

SOLAR_SHARE, OTHER_CLEAN, MEAN_CI = 0.07, 0.15, 0.727


def build():
    j = json.load(open(os.path.join(DATA, "nasa_power_hourly_Nagpur_2023.json")))
    v = j["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    g = np.array([max(v[k], 0.0) for k in sorted(v)], dtype=float)       # 8760 h, Wh/m2
    s = SOLAR_SHARE * g / g.mean()
    fossil = 1 - OTHER_CLEAN - s
    ef = MEAN_CI / fossil.mean()
    ci = ef * fossil
    return ci, ef, s


if __name__ == "__main__":
    ci, ef, s = build()
    np.save(os.path.join(DATA, "ci_india_2023.npy"), ci)
    hod = ci.reshape(365, 24).mean(0)
    out = dict(ef_fossil=ef, mean=ci.mean(), min=ci.min(), max=ci.max(), p5=np.percentile(ci, 5),
               p95=np.percentile(ci, 95), peak_solar_share=s.max(),
               hour_of_day_mean=[round(x, 4) for x in hod])
    save_json(out, "carbon_trace_summary.json")
    print(json.dumps(out, indent=1))
