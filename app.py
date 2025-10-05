
import streamlit as st
import pandas as pd
import numpy as np
import io
import json
import altair as alt

st.set_page_config(page_title="Thuiskopie – Scenario-tool (RED)", layout="wide")

st.title("Thuiskopie – Verdeelsleutel Scenario-tool")
st.caption("Variant met rode kolommen/balken (Altair + pandas Styler) inclusief **Reset naar standaard**.")

# --- Constants / Defaults ---
DISCIPLINES = ["Audio", "Audiovisueel", "Geschriften", "Beeld"]
DRAGERS = ["Desktop/Laptop", "Smartphone", "Tablet", "E-reader", "Externe HDD/NAS"]

def df_evidence_default():
    return pd.DataFrame(
        {
            "Audio":        [0.25, 0.25, 0.25, 0.25],
            "Audiovisueel": [0.25, 0.25, 0.25, 0.25],
            "Geschriften":  [0.25, 0.25, 0.25, 0.25],
            "Beeld":        [0.25, 0.25, 0.25, 0.25],
        },
        index=["Trend (2023)", "Waardering", "Dragerprofiel", "Buitenland"]
    )

def s_forfait_default():
    return pd.Series([0.25, 0.25, 0.25, 0.25], index=DISCIPLINES, name="Forfaitair")

def s_beeld_internal_default():
    # genormaliseerd binnen Beeld = 100%
    return pd.Series(
        [0.0882352941, 0.0882352941, 0.4117647059, 0.4117647059],
        index=["Audio-cover", "Beeld in AV", "Beeld in geschriften", "Losstaand beeld"]
    )

def s_drager_factors_default():
    return pd.Series([0.67, 0.65, 0.35, 0.48, 0.46], index=DRAGERS)

def df_baseline_default():
    return pd.DataFrame(np.full((len(DISCIPLINES), len(DRAGERS)), 1/len(DRAGERS)), index=DISCIPLINES, columns=DRAGERS)

def s_trend2023_default():
    return pd.Series([0.25, 0.25, 0.25, 0.25], index=DISCIPLINES)

A_DEFAULT = dict(Trend=0.20, Waardering=0.20, Drager=0.20, Buitenland=0.20, Forfaitair=0.20)
B_DEFAULT = dict(Bodem=0.10)

# --- Helpers ---
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

def style_red_columns(s):
    return ["color: red; font-weight: 600;"] * len(s)

def red_bar_chart(series, title):
    df = series.reset_index()
    df.columns = ["Discipline", "Share"]
    chart = (
        alt.Chart(df)
        .mark_bar(color="red")
        .encode(
            x=alt.X("Discipline:N", sort=list(series.index)),
            y=alt.Y("Share:Q"),
            tooltip=["Discipline", alt.Tooltip("Share:Q", format=".2%")]
        )
        .properties(title=title, width="container", height=220)
    )
    return chart

def reset_all():
    # Clear widget keys
    for key in [
        "evidence_vectors", "forfaitair_vector", "beeld_internal",
        "drager_factors", "baseline_matrix", "trend2023",
        "A_Trend", "A_Waard", "A_Drager", "A_Buiten", "A_Forfait",
        "B_Bodem", "B_w_trend", "B_w_waard", "B_w_drager", "B_w_buiten"
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# --- Sidebar: Reset & Parameters ---
st.sidebar.header("Parameters")

# Reset button
if st.sidebar.button("🔄 Reset naar standaard", type="primary"):
    reset_all()

# Evidence vectors editor
default_vectors = df_evidence_default()
st.sidebar.subheader("Evidence-vectoren per discipline")
evidence_vectors = st.sidebar.data_editor(
    st.session_state.get("evidence_vectors_default", default_vectors),
    use_container_width=True, num_rows="fixed", key="evidence_vectors"
)
evidence_vectors = normalize_rowwise(evidence_vectors)

# Forfaitair vector
default_forfaitair = s_forfait_default()
st.sidebar.subheader("Forfaitair vector per discipline")
forfaitair_series = st.sidebar.data_editor(
    default_forfaitair.to_frame().T,
    use_container_width=True, num_rows="fixed", key="forfaitair_vector"
)
forfaitair_series = normalize_rowwise(forfaitair_series).iloc[0]

# Scenario A weights (with explicit keys so reset can clear)
st.sidebar.subheader("Scenario A – componentgewichten (som = 100%)")
A_Trend   = st.sidebar.number_input("Trend",      min_value=0.0, max_value=1.0, value=A_DEFAULT["Trend"],      step=0.01, format="%.2f", key="A_Trend")
A_Waard   = st.sidebar.number_input("Waardering", min_value=0.0, max_value=1.0, value=A_DEFAULT["Waardering"], step=0.01, format="%.2f", key="A_Waard")
A_Drager  = st.sidebar.number_input("Drager",     min_value=0.0, max_value=1.0, value=A_DEFAULT["Drager"],     step=0.01, format="%.2f", key="A_Drager")
A_Buiten  = st.sidebar.number_input("Buitenland", min_value=0.0, max_value=1.0, value=A_DEFAULT["Buitenland"], step=0.01, format="%.2f", key="A_Buiten")
A_Forfait = st.sidebar.number_input("Forfaitair", min_value=0.0, max_value=1.0, value=A_DEFAULT["Forfaitair"], step=0.01, format="%.2f", key="A_Forfait")
A_weights = np.array([A_Trend, A_Waard, A_Drager, A_Buiten, A_Forfait])
A_weights = A_weights / (A_weights.sum() if A_weights.sum() != 0 else 1)

# Scenario B weights (with keys)
st.sidebar.subheader("Scenario B – bodem & evidence")
B_bodem = st.sidebar.number_input("Bodem – totaal (%)", min_value=0.0, max_value=1.0, value=B_DEFAULT["Bodem"], step=0.01, format="%.2f", key="B_Bodem")
B_evidence_total = max(0.0, 1.0 - B_bodem)
st.sidebar.caption(f"Evidence – totaal (%) = {B_evidence_total:.2f}")
B_w_trend  = st.sidebar.number_input("Evidence: Trend",      min_value=0.0, max_value=1.0, value=B_evidence_total/4, step=0.01, format="%.2f", key="B_w_trend")
B_w_waard  = st.sidebar.number_input("Evidence: Waardering", min_value=0.0, max_value=1.0, value=B_evidence_total/4, step=0.01, format="%.2f", key="B_w_waard")
B_w_drager = st.sidebar.number_input("Evidence: Drager",     min_value=0.0, max_value=1.0, value=B_evidence_total/4, step=0.01, format="%.2f", key="B_w_drager")
B_w_buiten = st.sidebar.number_input("Evidence: Buitenland", min_value=0.0, max_value=1.0, value=B_evidence_total/4, step=0.01, format="%.2f", key="B_w_buiten")
sumB = B_w_trend + B_w_waard + B_w_drager + B_w_buiten
scale = (B_evidence_total / sumB) if sumB > 0 else 0
B_weights = np.array([B_w_trend, B_w_waard, B_w_drager, B_w_buiten]) * scale

# Beeld interne shares
st.sidebar.subheader("Beeld – interne shares (binnen Beeld = 100%)")
default_beeld_internal = s_beeld_internal_default()
beeld_internal = st.sidebar.data_editor(default_beeld_internal.to_frame().T, num_rows="fixed", use_container_width=True, key="beeld_internal")
beeld_internal = normalize_rowwise(beeld_internal).iloc[0]

# Drager factors
st.sidebar.subheader("Drager-correctiefactoren (2024/2023)")
default_drager_factors = s_drager_factors_default()
drager_factors = st.sidebar.data_editor(default_drager_factors.to_frame().T, num_rows="fixed", use_container_width=True, key="drager_factors").iloc[0]

# Baseline 2023 dragerverdeling
st.sidebar.subheader("Baseline 2023 – dragerverdeling per discipline")
default_base = df_baseline_default()
baseline_matrix = st.sidebar.data_editor(default_base, use_container_width=True, key="baseline_matrix")
baseline_matrix = normalize_rowwise(baseline_matrix)

# Trend 2023 discipline vector
st.sidebar.subheader("Trend 2023 – disciplinevector")
default_trend2023 = s_trend2023_default()
trend2023 = st.sidebar.data_editor(default_trend2023.to_frame().T, num_rows="fixed", use_container_width=True, key="trend2023").iloc[0]
trend2023 = normalize_vec(trend2023)

# Convenience vectors
trend_vec      = evidence_vectors.loc["Trend (2023)"]
waardering_vec = evidence_vectors.loc["Waardering"]
drager_vec     = evidence_vectors.loc["Dragerprofiel"]
buitenland_vec = evidence_vectors.loc["Buitenland"]

# --- Calculations ---
# Scenario A
A_components = [trend_vec, waardering_vec, drager_vec, buitenland_vec, forfaitair_series]
A_result = sum(comp * w for comp, w in zip(A_components, A_weights))
A_result = normalize_vec(A_result)

# Scenario B
B_result = (
    forfaitair_series * B_bodem +
    trend_vec      * B_weights[0] +
    waardering_vec * B_weights[1] +
    drager_vec     * B_weights[2] +
    buitenland_vec * B_weights[3]
)
B_result = normalize_vec(B_result)

# Scenario C
multipliers = (baseline_matrix * drager_factors).sum(axis=1)
C_raw = trend2023 * multipliers
C_result = C_raw / C_raw.sum() if C_raw.sum() != 0 else C_raw

# Beeld sub-splits
def beeld_breakdown(beeld_share):
    return pd.Series({
        "Audio-cover":          beeld_share * beeld_internal["Audio-cover"],
        "Beeld in AV":          beeld_share * beeld_internal["Beeld in AV"],
        "Beeld in geschriften": beeld_share * beeld_internal["Beeld in geschriften"],
        "Losstaand beeld":      beeld_share * beeld_internal["Losstaand beeld"],
    })

A_beeld = beeld_breakdown(A_result["Beeld"])
B_beeld = beeld_breakdown(B_result["Beeld"])
C_beeld = beeld_breakdown(C_result["Beeld"])

# --- Display ---
colA, colB, colC = st.columns(3)

with colA:
    st.subheader("Scenario A – Hybride 5×20")
    st.dataframe(A_result.to_frame("Share").style.apply(style_red_columns, axis=1).format({"Share": "{:.2%}"}), use_container_width=True)
    st.altair_chart(red_bar_chart(A_result, "Scenario A – Shares per discipline"), use_container_width=True)
    st.caption("Beeld – interne uitsplitsing")
    st.dataframe(A_beeld.to_frame("Share").style.apply(style_red_columns, axis=1).format({"Share": "{:.2%}"}), use_container_width=True)

with colB:
    st.subheader("Scenario B – 10% bodem + evidence")
    st.dataframe(B_result.to_frame("Share").style.apply(style_red_columns, axis=1).format({"Share": "{:.2%}"}), use_container_width=True)
    st.altair_chart(red_bar_chart(B_result, "Scenario B – Shares per discipline"), use_container_width=True)
    st.caption("Beeld – interne uitsplitsing")
    st.dataframe(B_beeld.to_frame("Share").style.apply(style_red_columns, axis=1).format({"Share": "{:.2%}"}), use_container_width=True)

with colC:
    st.subheader("Scenario C – Trend-plus (dragercorrectie)")
    st.dataframe(C_result.to_frame("Share").style.apply(style_red_columns, axis=1).format({"Share": "{:.2%}"}), use_container_width=True)
    st.altair_chart(red_bar_chart(C_result, "Scenario C – Shares per discipline"), use_container_width=True)
    st.caption("Beeld – interne uitsplitsing")
    st.dataframe(C_beeld.to_frame("Share").style.apply(style_red_columns, axis=1).format({"Share": "{:.2%}"}), use_container_width=True)

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
}
json_bytes = io.BytesIO()
json_bytes.write(json.dumps(results_pack, indent=2).encode("utf-8"))
json_bytes.seek(0)
st.download_button("Download resultaten (JSON)", data=json_bytes, file_name="thuiskopie_scenario_results.json", mime="application/json")

to_csv_download(A_result.to_frame("Share"), "Scenario_A_shares")
to_csv_download(B_result.to_frame("Share"), "Scenario_B_shares")
to_csv_download(C_result.to_frame("Share"), "Scenario_C_shares")
to_csv_download(A_beeld.to_frame("Share"), "Scenario_A_beeld_sub")
to_csv_download(B_beeld.to_frame("Share"), "Scenario_B_beeld_sub")
to_csv_download(C_beeld.to_frame("Share"), "Scenario_C_beeld_sub")

st.markdown("---")
st.caption("Gebruik de knop 'Reset naar standaard' in de zijbalk om alle parameters terug te zetten.")
