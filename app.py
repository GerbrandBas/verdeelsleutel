
import streamlit as st
import pandas as pd
import numpy as np
import io
import matplotlib.pyplot as plt

st.set_page_config(page_title="Thuiskopie – Verdeelsleutel Scenario-tool", layout="wide")

st.title("Thuiskopie – Verdeelsleutel Scenario-tool")
st.caption("Interactieve rekenhulp voor scenario's A (Hybride 5×20), B (10% bodem) en C (Trend-plus met dragercorrectie)")

# --- Setup ---
disciplines = ["Audio", "Audiovisueel", "Geschriften", "Beeld"]
dragers = ["Desktop/Laptop", "Smartphone", "Tablet", "E-reader", "Externe HDD/NAS"]

def normalize_rowwise(df):
    df = df.copy()
    df = df.apply(lambda r: r / (r.sum() if r.sum() != 0 else 1), axis=1)
    return df

def normalize_vec(s):
    s = s.copy()
    total = s.sum()
    if total == 0:
        return s
    return s / total

def vector_from_editor(df, row_name):
    return df.loc[row_name, disciplines]

# --- Sidebar parameters ---
st.sidebar.header("Parameters")

# Evidence vectors editor
default_vectors = pd.DataFrame(
    {
        "Audio":        [0.25, 0.25, 0.25, 0.25],
        "Audiovisueel": [0.25, 0.25, 0.25, 0.25],
        "Geschriften":  [0.25, 0.25, 0.25, 0.25],
        "Beeld":        [0.25, 0.25, 0.25, 0.25],
    },
    index=["Trend (2023)", "Waardering", "Dragerprofiel", "Buitenland"]
)
st.sidebar.subheader("Evidence-vectoren per discipline")
evidence_vectors = st.sidebar.data_editor(
    default_vectors, use_container_width=True, num_rows="fixed", key="evidence_vectors"
)
evidence_vectors = normalize_rowwise(evidence_vectors)

# Forfaitair vector
default_forfaitair = pd.Series([0.25, 0.25, 0.25, 0.25], index=disciplines, name="Forfaitair")
st.sidebar.subheader("Forfaitair vector per discipline")
forfaitair_series = st.sidebar.data_editor(
    default_forfaitair.to_frame().T, use_container_width=True, num_rows="fixed", key="forfaitair_vector"
)
forfaitair_series = normalize_rowwise(forfaitair_series).iloc[0]

# Scenario A weights
st.sidebar.subheader("Scenario A – componentgewichten (som = 100%)")
A_Trend   = st.sidebar.number_input("Trend",     min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
A_Waard   = st.sidebar.number_input("Waardering",min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
A_Drager  = st.sidebar.number_input("Drager",    min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
A_Buiten  = st.sidebar.number_input("Buitenland",min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
A_Forfait = st.sidebar.number_input("Forfaitair",min_value=0.0, max_value=1.0, value=0.20, step=0.01, format="%.2f")
sumA = A_Trend + A_Waard + A_Drager + A_Buiten + A_Forfait
if abs(sumA - 1.0) > 1e-9:
    st.sidebar.warning(f"Som componentgewichten Scenario A = {sumA:.2f} (wordt automatisch genormaliseerd).")
A_weights = np.array([A_Trend, A_Waard, A_Drager, A_Buiten, A_Forfait])
A_weights = A_weights / (A_weights.sum() if A_weights.sum() != 0 else 1)

# Scenario B weights
st.sidebar.subheader("Scenario B – bodem & evidence")
B_bodem = st.sidebar.number_input("Bodem – totaal (%)", min_value=0.0, max_value=1.0, value=0.10, step=0.01, format="%.2f")
B_evidence_total = max(0.0, 1.0 - B_bodem)
st.sidebar.caption(f"Evidence – totaal (%) = {B_evidence_total:.2f}")
B_w_trend  = st.sidebar.number_input("Evidence: Trend",     min_value=0.0, max_value=1.0, value=B_evidence_total/4, step=0.01, format="%.2f")
B_w_waard  = st.sidebar.number_input("Evidence: Waardering",min_value=0.0, max_value=1.0, value=B_evidence_total/4, step=0.01, format="%.2f")
B_w_drager = st.sidebar.number_input("Evidence: Drager",    min_value=0.0, max_value=1.0, value=B_evidence_total/4, step=0.01, format="%.2f")
B_w_buiten = st.sidebar.number_input("Evidence: Buitenland",min_value=0.0, max_value=1.0, value=B_evidence_total/4, step=0.01, format="%.2f")
sumB_evidence = B_w_trend + B_w_waard + B_w_drager + B_w_buiten
if abs(sumB_evidence - B_evidence_total) > 1e-9:
    st.sidebar.warning(f"Som evidence-componenten = {sumB_evidence:.2f} (wordt automatisch geschaald naar {B_evidence_total:.2f}).")
B_weights_raw = np.array([B_w_trend, B_w_waard, B_w_drager, B_w_buiten])
scale = (B_evidence_total / sumB_evidence) if sumB_evidence > 0 else 0
B_weights = B_weights_raw * scale

# Beeld interne shares (genormaliseerd binnen Beeld = 100%)
st.sidebar.subheader("Beeld – interne shares (binnen Beeld = 100%)")
default_beeld_internal = pd.Series(
    [0.0882352941, 0.0882352941, 0.4117647059, 0.4117647059],
    index=["Audio-cover", "Beeld in AV", "Beeld in geschriften", "Losstaand beeld"]
)
beeld_internal = st.sidebar.data_editor(default_beeld_internal.to_frame().T, num_rows="fixed", use_container_width=True, key="beeld_internal")
beeld_internal = normalize_rowwise(beeld_internal).iloc[0]

# Drager factors
st.sidebar.subheader("Drager-correctiefactoren (2024/2023)")
default_drager_factors = pd.Series([0.67, 0.65, 0.35, 0.48, 0.46], index=dragers)
drager_factors = st.sidebar.data_editor(default_drager_factors.to_frame().T, num_rows="fixed", use_container_width=True, key="drager_factors").iloc[0]

# Baseline 2023 dragerverdeling per discipline (rijsom = 100%)
st.sidebar.subheader("Baseline 2023 – dragerverdeling per discipline")
default_base = pd.DataFrame(
    np.full((len(disciplines), len(dragers)), 1/len(dragers)),
    index=disciplines, columns=dragers
)
baseline_matrix = st.sidebar.data_editor(default_base, use_container_width=True, key="baseline_matrix")
baseline_matrix = normalize_rowwise(baseline_matrix)

# Trend 2023 discipline vector
st.sidebar.subheader("Trend 2023 – disciplinevector")
default_trend2023 = pd.Series([0.25, 0.25, 0.25, 0.25], index=disciplines)
trend2023 = st.sidebar.data_editor(default_trend2023.to_frame().T, num_rows="fixed", use_container_width=True, key="trend2023").iloc[0]
trend2023 = normalize_vec(trend2023)

# Convenience: expose vectors
trend_vec      = evidence_vectors.loc["Trend (2023)"]
waardering_vec = evidence_vectors.loc["Waardering"]
drager_vec     = evidence_vectors.loc["Dragerprofiel"]
buitenland_vec = evidence_vectors.loc["Buitenland"]

# --- Calculations ---
# Scenario A: sum_i A_weight[i] * vector[i]
A_components = [
    trend_vec, waardering_vec, drager_vec, buitenland_vec, forfaitair_series
]
A_result = (
    A_components[0] * A_weights[0] +
    A_components[1] * A_weights[1] +
    A_components[2] * A_weights[2] +
    A_components[3] * A_weights[3] +
    A_components[4] * A_weights[4]
)
A_result = normalize_vec(A_result)

# Scenario B: bodem * forfaitair + evidence_total * (sum evidence weights * vectors)
B_result = (
    forfaitair_series * B_bodem +
    trend_vec      * B_weights[0] +
    waardering_vec * B_weights[1] +
    drager_vec     * B_weights[2] +
    buitenland_vec * B_weights[3]
)
B_result = normalize_vec(B_result)

# Scenario C: multiplier per discipline = sumproduct(row, drager_factors); raw = trend2023 * multiplier; normalize
multipliers = (baseline_matrix * drager_factors).sum(axis=1)
C_raw = trend2023 * multipliers
C_result = C_raw / C_raw.sum() if C_raw.sum() != 0 else C_raw

# Beeld sub-splits (apply to each scenario's Beeld share)
def beeld_breakdown(beeld_share):
    return pd.Series({
        "Audio-cover":        beeld_share * beeld_internal["Audio-cover"],
        "Beeld in AV":        beeld_share * beeld_internal["Beeld in AV"],
        "Beeld in geschriften": beeld_share * beeld_internal["Beeld in geschriften"],
        "Losstaand beeld":    beeld_share * beeld_internal["Losstaand beeld"],
    })

A_beeld = beeld_breakdown(A_result["Beeld"])
B_beeld = beeld_breakdown(B_result["Beeld"])
C_beeld = beeld_breakdown(C_result["Beeld"])

# --- Display ---
colA, colB, colC = st.columns(3)
with colA:
    st.subheader("Scenario A – Hybride 5×20")
    st.dataframe(A_result.to_frame("Share"), use_container_width=True)
    # Chart
    fig, ax = plt.subplots()
    ax.bar(A_result.index, A_result.values)
    ax.set_title("Scenario A – Shares per discipline")
    st.pyplot(fig)
    st.caption("Beeld – interne uitsplitsing")
    st.dataframe(A_beeld.to_frame("Share"), use_container_width=True)

with colB:
    st.subheader("Scenario B – 10% bodem + evidence")
    st.dataframe(B_result.to_frame("Share"), use_container_width=True)
    fig2, ax2 = plt.subplots()
    ax2.bar(B_result.index, B_result.values)
    ax2.set_title("Scenario B – Shares per discipline")
    st.pyplot(fig2)
    st.caption("Beeld – interne uitsplitsing")
    st.dataframe(B_beeld.to_frame("Share"), use_container_width=True)

with colC:
    st.subheader("Scenario C – Trend-plus (dragercorrectie)")
    st.dataframe(C_result.to_frame("Share"), use_container_width=True)
    fig3, ax3 = plt.subplots()
    ax3.bar(C_result.index, C_result.values)
    ax3.set_title("Scenario C – Shares per discipline")
    st.pyplot(fig3)
    st.caption("Beeld – interne uitsplitsing")
    st.dataframe(C_beeld.to_frame("Share"), use_container_width=True)

# Downloads
def to_csv_download(df, name):
    csv = df.to_csv().encode("utf-8")
    st.download_button(f"Download {name} (CSV)", data=csv, file_name=f"{name}.csv", mime="text/csv")

st.markdown("---")
st.subheader("Downloads")
results_pack = {
    "A_result": A_result.to_dict(),
    "B_result": B_result.to_dict(),
    "C_result": C_result.to_dict(),
    "A_beeld": A_beeld.to_dict(),
    "B_beeld": B_beeld.to_dict(),
    "C_beeld": C_beeld.to_dict(),
    "config": {
        "evidence_vectors": evidence_vectors.to_dict(),
        "forfaitair": forfaitair_series.to_dict(),
        "A_weights": A_weights.tolist(),
        "B_bodem": B_bodem,
        "B_weights": B_weights.tolist(),
        "beeld_internal": beeld_internal.to_dict(),
        "drager_factors": drager_factors.to_dict(),
        "baseline_matrix": baseline_matrix.to_dict(),
        "trend2023": trend2023.to_dict(),
    }
}
json_bytes = io.BytesIO()
json_bytes.write(json.dumps(results_pack, indent=2).encode("utf-8"))
json_bytes.seek(0)
st.download_button("Download resultaten + config (JSON)", data=json_bytes, file_name="thuiskopie_scenario_results.json", mime="application/json")

# Individual CSVs
to_csv_download(A_result.to_frame("Share"), "Scenario_A_shares")
to_csv_download(B_result.to_frame("Share"), "Scenario_B_shares")
to_csv_download(C_result.to_frame("Share"), "Scenario_C_shares")
to_csv_download(A_beeld.to_frame("Share"), "Scenario_A_beeld_sub")
to_csv_download(B_beeld.to_frame("Share"), "Scenario_B_beeld_sub")
to_csv_download(C_beeld.to_frame("Share"), "Scenario_C_beeld_sub")

st.markdown("---")
st.caption("Tip: sla je configuratie als JSON op, zodat je exact dezelfde parameters later kunt herladen (vervolgversie van de app kan import ondersteunen).")
