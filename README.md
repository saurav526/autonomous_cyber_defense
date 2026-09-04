# Autonomous Cyber Defense System

A defensive ML simulation demonstrating:

1. Synthetic network telemetry generation
2. Graph construction
3. Neighborhood-based graph representation learning
4. Self-supervised graph autoencoder
5. Reconstruction-error anomaly detection
6. Reinforcement-learning response policy
7. Streamlit dashboard and CSV export

## Run

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Architecture

Network telemetry
    ↓
Graph construction
    ↓
Normalized neighborhood aggregation
    ↓
Graph representation
    ↓
Autoencoder
    ↓
Reconstruction error
    ↓
Anomaly threshold
    ↓
RL response policy
    ↓
Monitor / Rate-limit / Isolate / Block source

## Important note

This is an educational defensive simulation. The network traffic and attacks are synthetic. The response actions do not touch real systems.
