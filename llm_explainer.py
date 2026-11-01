import os


def explain_traffic(df, dfa_result):
    """Generate an explanation of the analyzed traffic data."""

    total_records = len(df)

    state_counts = (
        df["Traffic_State"]
        .value_counts()
        .reindex(
            ["FREE", "LIGHT", "MODERATE", "HEAVY", "CONGESTED"],
            fill_value=0,
        )
    )

    most_common_state = state_counts.idxmax()

    congested_count = int(state_counts["CONGESTED"])
    congested_percentage = (
        congested_count / total_records * 100
        if total_records
        else 0
    )

    summary = (
        f"Analyzed {total_records:,} traffic observations. "
        f"The most common derived traffic category is "
        f"{most_common_state}. "
        f"{congested_count:,} observations "
        f"({congested_percentage:.1f}%) were classified as "
        f"CONGESTED using relative vehicle-count thresholds. "
        f"DFA result: {dfa_result['message']}"
    )

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "Local Traffic Analysis\n\n"
            + summary
            + "\n\nNote: These categories are derived from "
            "vehicle-count percentiles within each junction. "
            "They are not official congestion measurements. "
            "The dataset does not directly provide vehicle speed "
            "or road occupancy."
        )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

        response = client.responses.create(
            model=model,
            instructions=(
                "You are TrafficGPT, an assistant that explains "
                "traffic dataset analysis. Explain the results "
                "clearly in simple language. Do not claim that "
                "the data provides vehicle speed, road occupancy, "
                "live traffic, or future predictions. The traffic "
                "categories are derived from vehicle-count "
                "percentiles, not official ground-truth labels. "
                "Do not invent facts or causes."
            ),
            input=summary,
        )

        return response.output_text

    except Exception as error:
        return (
            "AI explanation could not be generated. "
            "Showing the local analysis instead.\n\n"
            f"{summary}\n\n"
            f"Technical detail: {error}"
        )