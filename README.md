# TrafficGPT: AI Traffic Congestion Detection Using Finite Automata

## 1. Project Overview

TrafficGPT is an educational traffic-analysis application that
processes historical vehicle-count data, classifies observations
into relative traffic categories, and applies a Deterministic
Finite Automaton (DFA) to identify sequences containing a
congested observation.

The project combines data analysis, visualization, formal
language concepts, and optional AI-generated explanations.

## 2. Objectives

- Analyze historical vehicle counts.
- Categorize observations into five traffic states.
- Implement a DFA for congestion-state detection.
- Visualize traffic patterns across junctions and time.
- Generate explanations of analysis results.
- Demonstrate an application of Finite Automata in traffic analysis.

## 3. Technologies Used

- Python
- Pandas
- Streamlit
- Plotly
- Deterministic Finite Automaton (DFA)
- OpenAI API (optional)
- Jupyter Notebook / Google Colab
- Git and GitHub

## 4. Traffic Categories

Traffic observations are assigned one of five relative categories:

| Category | Symbol |
|---|---|
| FREE | F |
| LIGHT | L |
| MODERATE | M |
| HEAVY | H |
| CONGESTED | C |

Categories are derived from vehicle-count percentile ranks
within each junction. They are not official traffic labels.

## 5. DFA Design

The DFA recognizes sequences containing at least one `C` symbol.

- Start state: q0
- Accepting state: q1
- Alphabet: {F, L, M, H, C}

Transition rules:

- q0 transitions to q1 when it reads C.
- q0 remains q0 when it reads F, L, M, or H.
- q1 remains q1 for every valid symbol.

The language recognized by this DFA consists of all valid
traffic-state sequences containing at least one C.

## 6. Dataset

Dataset: Traffic Prediction Dataset by fedesoriano.

Source:
https://www.kaggle.com/datasets/fedesoriano/traffic-prediction-dataset

Download the dataset from Kaggle and place the relevant CSV
file in the `data` directory. Rename it to `traffic.csv`.

The expected columns are:

- DateTime
- Junction
- Vehicles
- ID (optional)

The dataset contains historical vehicle counts. It does not
directly provide vehicle speed, road occupancy, or live traffic
conditions.

Please follow the dataset's license and attribution requirements.

## 7. Project Structure

TrafficGPT/
- app.py
- traffic_engine.py
- llm_explainer.py
- requirements.txt
- README.md
- .gitignore
- data/
  - traffic.csv
- notebooks/
  - TrafficGPT_Colab.ipynb

## 8. Installation

Create and activate a virtual environment:

    py -m venv .venv

In Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Install the dependencies:

    python -m pip install -r requirements.txt

## 9. Run the Application

    python -m streamlit run app.py

If PowerShell blocks virtual-environment activation, run:

    .\.venv\Scripts\python.exe -m pip install -r requirements.txt

    .\.venv\Scripts\python.exe -m streamlit run app.py

Alternatively, upload the CSV using the dashboard's sidebar.

## 10. Optional AI Explanation

The dashboard works without an API key.

To enable OpenAI-generated explanations in Windows PowerShell:

    $env:OPENAI_API_KEY="your_api_key_here"

Then run:

    python -m streamlit run app.py

Keep API keys private. Never commit them to GitHub.

API usage may incur charges.

## 11. Limitations

- Traffic categories are derived from the dataset's vehicle counts.
- The categories are not verified against official congestion labels.
- The application does not provide live traffic monitoring.
- The DFA detects an existing congested category; it does not
  predict future congestion.
- Vehicle speed and road occupancy are not available in the
  selected dataset.

## 12. Future Enhancements

- Integrate real-time traffic data from an authorized source.
- Compare DFA-based detection with other sequence models.
- Add traffic forecasting using machine learning.
- Introduce more advanced traffic-state transition rules.
- Evaluate results against independently verified congestion labels.

## 13. Social Impact

The project is related to UN Sustainable Development Goal 11:
Sustainable Cities and Communities. It demonstrates how
historical traffic data and computational models can support
traffic analysis and urban-mobility research.