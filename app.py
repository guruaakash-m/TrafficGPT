import os

import pandas as pd
import plotly.express as px
import streamlit as st
 

from traffic_engine import (
    analyze_traffic,
    load_traffic_data,
)
from llm_explainer import explain_traffic


st.set_page_config(
    page_title="TrafficGPT",
    page_icon="🚦",
    layout="wide",
)


st.title("🚦 TrafficGPT")
st.subheader(
    "AI-Assisted Traffic Congestion Analysis Using Finite Automata"
)

st.markdown(
    """
    TrafficGPT analyzes historical vehicle counts, assigns relative
    traffic categories, and uses a Deterministic Finite Automaton (DFA)
    to identify whether a sequence contains a congested observation.
    """
)


# ---------------- SIDEBAR ----------------

st.sidebar.header("Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload your traffic CSV file",
    type=["csv"],
)

default_path = os.path.join("data", "traffic.csv")


# ---------------- LOAD DATA ----------------

try:
    if uploaded_file is not None:
        raw_df = load_traffic_data(uploaded_file)
        dataset_source = "Uploaded CSV"

    elif os.path.exists(default_path):
        raw_df = load_traffic_data(default_path)
        dataset_source = default_path

    else:
        st.info(
            "Upload your traffic CSV using the sidebar, or place "
            "the dataset at data/traffic.csv."
        )

        st.code(
            "TrafficGPT/\n"
            "├── app.py\n"
            "├── traffic_engine.py\n"
            "├── llm_explainer.py\n"
            "├── requirements.txt\n"
            "└── data/\n"
            "    └── traffic.csv"
        )

        st.stop()

    classified_df, dfa_result = analyze_traffic(raw_df)

except Exception as error:
    st.error(f"Unable to process the dataset: {error}")
    st.stop()


# ---------------- DATASET SUMMARY ----------------

st.caption(f"Dataset source: {dataset_source}")

total_records = len(classified_df)
junction_count = classified_df["Junction"].nunique()
total_vehicles = classified_df["Vehicles"].sum()
average_vehicles = classified_df["Vehicles"].mean()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Observations", f"{total_records:,}")
col2.metric("Junctions", f"{junction_count:,}")
col3.metric("Total Vehicle Counts", f"{total_vehicles:,.0f}")
col4.metric("Average Vehicles", f"{average_vehicles:.2f}")


# ---------------- FILTERS ----------------

st.sidebar.header("Filters")

junction_options = sorted(
    classified_df["Junction"].unique().tolist()
)

selected_junctions = st.sidebar.multiselect(
    "Select junctions",
    options=junction_options,
    default=junction_options,
)

filtered_df = classified_df[
    classified_df["Junction"].isin(selected_junctions)
].copy()

if filtered_df.empty:
    st.warning("Select at least one junction to display results.")
    st.stop()


# ---------------- TRAFFIC DISTRIBUTION ----------------

st.header("1. Traffic State Distribution")

state_order = [
    "FREE",
    "LIGHT",
    "MODERATE",
    "HEAVY",
    "CONGESTED",
]

state_counts = (
    filtered_df["Traffic_State"]
    .value_counts()
    .reindex(state_order, fill_value=0)
    .rename_axis("Traffic State")
    .reset_index(name="Observations")
)

fig_distribution = px.bar(
    state_counts,
    x="Traffic State",
    y="Observations",
    title="Observations by Derived Traffic Category",
    category_orders={"Traffic State": state_order},
    text="Observations",
)

fig_distribution.update_layout(
    xaxis_title="Derived Traffic Category",
    yaxis_title="Number of Observations",
)

st.plotly_chart(
    fig_distribution,
    use_container_width=True,
)


# ---------------- VEHICLE COUNTS OVER TIME ----------------

st.header("2. Vehicle Counts Over Time")

fig_vehicles = px.line(
    filtered_df,
    x="DateTime",
    y="Vehicles",
    color="Junction",
    title="Historical Vehicle Counts",
    labels={
        "DateTime": "Date and Time",
        "Vehicles": "Vehicle Count",
        "Junction": "Junction",
    },
)

st.plotly_chart(
    fig_vehicles,
    use_container_width=True,
)


# ---------------- TRAFFIC STATE TIMELINE ----------------

st.header("3. Traffic State Timeline")

timeline_df = filtered_df.sort_values("DateTime").copy()

fig_timeline = px.scatter(
    timeline_df,
    x="DateTime",
    y="Junction",
    color="Traffic_State",
    category_orders={"Traffic_State": state_order},
    hover_data=["Vehicles", "State_Symbol"],
    title="Derived Traffic Categories Across Time",
    labels={
        "DateTime": "Date and Time",
        "Junction": "Junction",
        "Traffic_State": "Traffic State",
    },
)

st.plotly_chart(
    fig_timeline,
    use_container_width=True,
)


# ---------------- DFA ANALYSIS ----------------

st.header("4. Finite Automata Analysis")

st.markdown(
    """
    **DFA states**

    - `q0`: No congested observation has been encountered.
    - `q1`: At least one congested observation has been encountered.

    **Transition rules**

    - From `q0`, symbol `C` moves to `q1`.
    - From `q0`, symbols `F`, `L`, `M`, and `H` remain in `q0`.
    - From `q1`, every valid symbol remains in `q1`.

    The DFA accepts a sequence if it contains at least one `C`.
    """
)

junction_for_dfa = st.selectbox(
    "Choose a junction for DFA analysis",
    options=junction_options,
)

dfa_df = (
    classified_df[
        classified_df["Junction"] == junction_for_dfa
    ]
    .sort_values("DateTime")
    .copy()
)

dfa_symbols = dfa_df["State_Symbol"].tolist()

# Limit the displayed trace to keep the dashboard responsive.
max_symbols = min(len(dfa_symbols), 2000)

dfa_result = None

from traffic_engine import run_dfa

dfa_result = run_dfa(dfa_symbols[:max_symbols])

dfa_col1, dfa_col2, dfa_col3 = st.columns(3)

dfa_col1.metric("Symbols Processed", f"{max_symbols:,}")
dfa_col2.metric("Final DFA State", dfa_result["final_state"])
dfa_col3.metric(
    "Congestion Detected",
    "Yes" if dfa_result["accepted"] else "No",
)

if dfa_result["accepted"]:
    st.error(dfa_result["message"])
else:
    st.success(dfa_result["message"])

if len(dfa_symbols) > max_symbols:
    st.info(
        f"The DFA display analyzed the first {max_symbols:,} "
        f"chronological observations for Junction "
        f"{junction_for_dfa}."
    )

with st.expander("View DFA transition trace"):
    trace_df = pd.DataFrame(dfa_result["trace"])
    st.dataframe(trace_df, use_container_width=True)


# ---------------- AI EXPLANATION ----------------

st.header("5. TrafficGPT Explanation")

st.caption(
    "Uses a local explanation by default. Configure an OpenAI API "
    "key to enable AI-generated explanations."
)

explanation = explain_traffic(filtered_df, dfa_result)

st.info(explanation)


# ---------------- DATA TABLE ----------------

st.header("6. Processed Traffic Dataset")

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=400,
)

csv_data = filtered_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Processed Traffic CSV",
    data=csv_data,
    file_name="trafficgpt_processed.csv",
    mime="text/csv",
)


# ---------------- FOOTNOTE ----------------

st.divider()

st.caption(
    "TrafficGPT is an educational project. Traffic categories are "
    "derived from historical vehicle-count distributions. The "
    "dataset does not directly provide speed, road occupancy, or "
    "real-time traffic conditions. The DFA detects a congested "
    "category in the analyzed sequence; it does not forecast traffic."
)