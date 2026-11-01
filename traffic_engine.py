import pandas as pd


TRAFFIC_STATES = [
    "FREE",
    "LIGHT",
    "MODERATE",
    "HEAVY",
    "CONGESTED",
]

STATE_SYMBOLS = {
    "FREE": "F",
    "LIGHT": "L",
    "MODERATE": "M",
    "HEAVY": "H",
    "CONGESTED": "C",
}


def load_traffic_data(file_path):
    """Load and validate the traffic CSV dataset."""
    df = pd.read_csv(file_path)

    required_columns = {"DateTime", "Junction", "Vehicles"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: "
            f"{', '.join(sorted(missing_columns))}"
        )

    df = df.copy()

    df["DateTime"] = pd.to_datetime(
        df["DateTime"], errors="coerce"
    )
    df["Junction"] = pd.to_numeric(
        df["Junction"], errors="coerce"
    )
    df["Vehicles"] = pd.to_numeric(
        df["Vehicles"], errors="coerce"
    )

    df = df.dropna(
        subset=["DateTime", "Junction", "Vehicles"]
    )

    df = df[df["Vehicles"] >= 0]

    df = df.sort_values(
        ["Junction", "DateTime"]
    ).reset_index(drop=True)

    if df.empty:
        raise ValueError(
            "No valid traffic records were found in the CSV file."
        )

    return df



def classify_traffic(df):
    """Classify traffic observations while preserving all columns."""

    df = df.copy().reset_index(drop=True)

    traffic_states = []

    for junction, group in df.groupby("Junction", sort=False):
        ranks = group["Vehicles"].rank(
            method="average",
            pct=True
        )

        states = pd.cut(
            ranks,
            bins=[0, 0.20, 0.40, 0.60, 0.80, 1.0],
            labels=TRAFFIC_STATES,
            include_lowest=True
        )

        traffic_states.append(
            pd.Series(
                states.astype(str).values,
                index=group.index
            )
        )

    # Assign classifications without removing original columns
    df["Traffic_State"] = pd.concat(
        traffic_states
    ).sort_index()

    df["State_Symbol"] = df["Traffic_State"].map(
        STATE_SYMBOLS
    )

    return df


def run_dfa(symbols):
    """
    DFA that accepts a sequence if it contains C
    (at least one CONGESTED observation).

    q0: No congested observation encountered.
    q1: At least one congested observation encountered.

    q0 -- C --> q1
    q0 -- F/L/M/H --> q0
    q1 -- any valid symbol --> q1
    """
    current_state = "q0"

    transitions = {
        "q0": {
            "F": "q0",
            "L": "q0",
            "M": "q0",
            "H": "q0",
            "C": "q1",
        },
        "q1": {
            "F": "q1",
            "L": "q1",
            "M": "q1",
            "H": "q1",
            "C": "q1",
        },
    }

    trace = []

    for symbol in symbols:
        if symbol not in STATE_SYMBOLS.values():
            raise ValueError(
                f"Invalid traffic symbol: {symbol}"
            )

        previous_state = current_state
        current_state = transitions[current_state][symbol]

        trace.append({
            "Symbol": symbol,
            "Previous_State": previous_state,
            "Next_State": current_state,
        })

    return {
        "accepted": current_state == "q1",
        "final_state": current_state,
        "trace": trace,
        "message": (
            "Congestion detected: the sequence contains "
            "at least one CONGESTED observation."
            if current_state == "q1"
            else "No CONGESTED observation was found "
            "in this sequence."
        ),
    }


def analyze_traffic(df):
    """Run traffic classification and DFA analysis."""
    classified_df = classify_traffic(df)

    symbols = classified_df["State_Symbol"].tolist()
    dfa_result = run_dfa(symbols)

    return classified_df, dfa_result
