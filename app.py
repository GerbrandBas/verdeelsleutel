
import io
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from openpyxl import load_workbook

st.set_page_config(page_title="Thuiskopie Verdeelsleutel Simulator", layout="wide")

st.title("Thuiskopie Verdeelsleutel — Scenario A/B/C • met stabilisatie & 25% beeld-target")

st.markdown("""
Gebruik dit hulpmiddel om de **disciplineverdeling** te simuleren o.b.v. je Excel-model
(**Parameters**, **Baseline**, **Invoer_2023**). De app leest standaard het meegeleverde modelbestand,
maar je kunt ook een eigen Excel uploaden met dezelfde sheet- en veldnamen.
""")

# Load model (default or user upload)
default_model_path = "Thuiskopie_model.xlsx"
uploaded = st.file_uploader("Upload je Excel-model (optioneel)", type=["xlsx"])

def load_workbook_from_bytes(b):
    # Save to buffer file because openpyxl expects a file path or file-like object
    bio = io.BytesIO(b.read()) if hasattr(b, "read") else io.BytesIO(b)
    return load_workbook(bio, data_only=True)

if uploaded is not None:
    wb = load_workbook_from_bytes(uploaded)
else:
    wb = load_workbook(default_model_path, data_only=True)

# Read parameters
ws_params = wb["Parameters"]
def get_param(name, default=None):
    for r in range(2, ws_params.max_row+1):
        if ws_params.cell(row=r, column=1).value == name:
            val = ws_params.cell(row=r, column=3).value
            return float(val) if isinstance(val, (int, float)) or (isinstance(val,str) and val.replace('.','',1).isdigit()) else val
    return default

# Discipline and device setup
disciplines = ["Audio","AV","Geschriften","Beeld"]
device_cols = ["Desktop/Laptop","Smartphone","Tablet","E-reader","Externe HDD/NAS"]

# Baseline (trend shares + device matrix)
ws_base = wb["Baseline"]
T = {}
for i, d in enumerate(disciplines):
    v = ws_base.cell(row=5+i, column=2).value  # B5..B8
    T[d] = float(v)/100.0

device_pct = {}
for i, d in enumerate(disciplines):
    r = 14+i  # rows 14..17
    device_pct[d] = {
        "Desktop/Laptop": float(ws_base.cell(row=r, column=2).value)/100.0,
        "Smartphone":     float(ws_base.cell(row=r, column=3).value)/100.0,
        "Tablet":         float(ws_base.cell(row=r, column=4).value)/100.0,
        "E-reader":       float(ws_base.cell(row=r, column=5).value)/100.0,
        "Externe HDD/NAS":float(ws_base.cell(row=r, column=6).value)/100.0,
    }

# Invoer_2023 shares
ws23 = wb["Invoer_2023"]
share_2023 = {}
total_2023 = 0.0
vals_2023 = {}
for i, d in enumerate(disciplines):
    v = ws23.cell(row=4+i, column=2).value  # B4..B7 numeric
    vals_2023[d] = float(v) if v is not None else None
total_2023 = sum([v for v in vals_2023.values() if v is not None])
for d in disciplines:
    share_2023[d] = (vals_2023[d] / total_2023) if (vals_2023[d] is not None and total_2023>0) else np.nan

# Parameters — weights and factors
w_trend_A   = get_param("w_trend_A", 0.20)
w_val_A     = get_param("w_valuation_A", 0.20)
w_drag_A    = get_param("w_drager_A", 0.20)
w_fore_A    = get_param("w_foreign_A", 0.20)
w_eq_A      = get_param("w_equal_A", 0.20)

w_trend_B   = get_param("w_trend_B", 0.35)
w_val_B     = get_param("w_valuation_B", 0.35)
w_drag_B    = get_param("w_drager_B", 0.20)
w_fore_B    = get_param("w_foreign_B", 0.10)
bodem_beeld = get_param("bodem_beeld_percent", 10.0) / 100.0

# Scenario C weights
w_trend_C   = get_param("w_trend_C", 0.70)
w_drager_C  = get_param("w_drager_C", 0.30)

# Drager weights
dev_w_desktop   = get_param("dev_w_desktop", 0.67)
dev_w_smartphone= get_param("dev_w_smartphone", 0.65)
dev_w_tablet    = get_param("dev_w_tablet", 0.35)
dev_w_ereader   = get_param("dev_w_ereader", 0.48)
dev_w_ext       = get_param("dev_w_ext", 1.00)
w_sp_beeld      = get_param("w_smartphone_beeld", 1.10)

# Foreign factors
f_audio = get_param("foreign_factor_audio", 1.0)
f_av    = get_param("foreign_factor_av", 1.0)
f_ges   = get_param("foreign_factor_geschriften", 1.0)
f_bld   = get_param("foreign_factor_beeld", 1.0)

# Valuation factors
ppk   = get_param("ppk_beeld_factor", 1.20)
longv = get_param("longevity_factor_beeld", 1.10)
emb   = get_param("embed_factor_beeld", 1.15)
cld   = get_param("cloud_correction_beeld", 1.05)
msg   = get_param("messenger_cache_factor_beeld", 1.05)
master= get_param("embed_cloud_factor_beeld", 1.00)

# Target
target = get_param("beeld_target_percent", 25.0)/100.0

# Stability cap
cap = get_param("stability_cap_pp", 0.10)

# ---- Helpers ----
def device_factor_of(d):
    s = device_pct[d]
    sp_mult = w_sp_beeld if d=="Beeld" else 1.0
    return (
        s["Desktop/Laptop"]*dev_w_desktop +
        s["Smartphone"]*dev_w_smartphone*sp_mult +
        s["Tablet"]*dev_w_tablet +
        s["E-reader"]*dev_w_ereader +
        s["Externe HDD/NAS"]*dev_w_ext
    )

def normalize_dict(dct):
    total = sum(dct.values())
    return {k: (v/total if total>0 else 0.0) for k,v in dct.items()}

# ---- Component shares ----
# Valuation
val_factor = {d: 1.0 for d in disciplines}
val_factor["Beeld"] = ppk * longv * emb * cld * msg * master
val_raw = {d: T[d]*val_factor[d] for d in disciplines}
val_norm = normalize_dict(val_raw)

# Drager
devF = {d: device_factor_of(d) for d in disciplines}
drager_raw = {d: T[d]*devF[d] for d in disciplines}
drager_norm = normalize_dict(drager_raw)

# Foreign
Ffac = {"Audio": f_audio, "AV": f_av, "Geschriften": f_ges, "Beeld": f_bld}
foreign_raw = {d: T[d]*Ffac[d] for d in disciplines}
foreign_norm = normalize_dict(foreign_raw)

# Equal
equal_share = {d: 0.25 for d in disciplines}

# ---- Scenarios ----
# A
share_A = {d: w_trend_A*T[d] + w_val_A*val_norm[d] + w_drag_A*drager_norm[d] + w_fore_A*foreign_norm[d] + w_eq_A*equal_share[d] for d in disciplines}
# normalize to be safe
share_A = normalize_dict(share_A)

# B (evidence + bodem beeld)
evidence = {d: w_trend_B*T[d] + w_val_B*val_norm[d] + w_drag_B*drager_norm[d] + w_fore_B*foreign_norm[d] for d in disciplines}
evidence = normalize_dict(evidence)
share_B = {}
for d in disciplines:
    if d == "Beeld":
        share_B[d] = bodem_beeld + (1 - bodem_beeld) * evidence[d]
    else:
        share_B[d] = (1 - bodem_beeld) * evidence[d]
# normalize to be sure
share_B = normalize_dict(share_B)

# C (status-quo 2023 + dragercorrectie 2024)
dragerC_raw = {}
for d in disciplines:
    base = share_2023[d] if not math.isnan(share_2023[d]) else T[d]
    dragerC_raw[d] = base * devF[d]
dragerC_norm = normalize_dict(dragerC_raw)
share_C = {d: w_trend_C*(share_2023[d] if not math.isnan(share_2023[d]) else T[d]) + w_drager_C*dragerC_norm[d] for d in disciplines}
share_C = normalize_dict(share_C)

# Stabilisatie (Scenario B t.o.v. 2023)
stab_pre = {}
for d in disciplines:
    base = share_2023[d] if not math.isnan(share_2023[d]) else T[d]
    delta = share_B[d] - base
    adj = np.sign(delta) * min(abs(delta), cap)
    stab_pre[d] = base + adj
stab = normalize_dict(stab_pre)

# ---- Output tables ----
def to_df(dct, name):
    return pd.DataFrame({"Discipline": list(dct.keys()), name: list(dct.values())}).set_index("Discipline")

dfA = to_df(share_A, "Scenario A")
dfB = to_df(share_B, "Scenario B")
dfC = to_df(share_C, "Scenario C")
dfS = to_df(stab, "B gestabiliseerd")

df_all = dfA.join(dfB).join(dfS).join(dfC)

st.subheader("Eindverdeling per scenario")
st.dataframe((df_all*100).round(2))

# Charts
def plot_series(series, title):
    fig, ax = plt.subplots()
    ax.bar(series.index, series.values)
    ax.set_title(title)
    ax.set_ylabel("Aandeel")
    ax.set_ylim(0, 1.0)
    return fig

col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_series(dfB["Scenario B"], "Scenario B"))
with col2:
    st.pyplot(plot_series(dfS["B gestabiliseerd"], "Scenario B (gestabiliseerd)"))

st.markdown("---")
st.subheader("Target helper — Beeld")
st.write(f"Doel (Parameters): **{target*100:.1f}%**")
st.write(f"Huidig Scenario B — Beeld: **{dfB.loc['Beeld','Scenario B']*100:.2f}%**")
st.write(f"Gap: **{(target - dfB.loc['Beeld','Scenario B'])*100:.2f} pp**")

# Download CSV
st.markdown("### Download resultaten")
csv = (df_all*100).round(3).to_csv().encode("utf-8")
st.download_button("Download CSV (percentages)", data=csv, file_name="thuiskopie_scenario_resultaten.csv", mime="text/csv")

st.markdown("---")
st.caption("Let op: Dit is een parametrische simulator. Gebruik voor besluitvorming gevalideerde exports en documenteer parameterkeuzes.")
