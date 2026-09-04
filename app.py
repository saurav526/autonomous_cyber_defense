import streamlit as st
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from cyber_model import (
    generate_network,
    train_autoencoder,
    score_anomalies,
    build_graph_features,
    train_dqn,
    choose_action,
)

st.set_page_config(
    page_title="Autonomous Cyber Defense",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 30px;
    }

    .alert-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #3b1f1f;
        border: 1px solid #ef4444;
        margin-bottom: 15px;
    }

    .safe-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #183322;
        border: 1px solid #22c55e;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛡️ Autonomous Cyber Defense System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Graph-based anomaly detection + self-supervised learning +
    reinforcement-learning based defensive response
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Simulation Controls")

st.sidebar.markdown("---")

network_nodes = st.sidebar.slider(
    "🌐 Network Nodes",
    min_value=20,
    max_value=100,
    value=40,
    step=5,
)

attack_intensity = st.sidebar.slider(
    "🚨 Attack Intensity",
    min_value=0.02,
    max_value=0.30,
    value=0.10,
    step=0.01,
)

epochs = st.sidebar.slider(
    "🧠 Autoencoder Epochs",
    min_value=10,
    max_value=100,
    value=40,
    step=10,
)

random_seed = st.sidebar.number_input(
    "🎲 Random Seed",
    min_value=1,
    max_value=9999,
    value=42,
    step=1,
)

st.sidebar.markdown("---")

run_simulation = st.sidebar.button(
    "🚀 Run Defense Simulation",
    type="primary",
    use_container_width=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "simulation" not in st.session_state:
    st.session_state.simulation = None


# ============================================================
# RUN SIMULATION
# ============================================================

if run_simulation:

    with st.spinner(
        "Generating network → learning representations → "
        "detecting anomalies → selecting defensive responses..."
    ):

        # ----------------------------------------------------
        # STEP 1: Generate synthetic network
        # ----------------------------------------------------

        data = generate_network(
            n_nodes=network_nodes,
            attack_rate=attack_intensity,
            seed=int(random_seed),
        )

        graph = data["graph"]
        node_features = data["node_features"]
        df = data["table"].copy()

        # ----------------------------------------------------
        # STEP 2: Graph feature construction
        # ----------------------------------------------------

        graph_features = build_graph_features(
            graph,
            node_features,
        )

        # ----------------------------------------------------
        # STEP 3: Train self-supervised autoencoder
        # ----------------------------------------------------

        losses = train_autoencoder(
            graph_features,
            epochs=epochs,
        )

        # ----------------------------------------------------
        # STEP 4: Calculate anomaly scores
        # ----------------------------------------------------

        anomaly_scores = score_anomalies(
            graph_features
        )

        # ----------------------------------------------------
        # STEP 5: Determine anomaly threshold
        # ----------------------------------------------------

        threshold = float(
            np.quantile(
                anomaly_scores,
                0.90,
            )
        )

        detected = anomaly_scores >= threshold

        # ----------------------------------------------------
        # STEP 6: Train RL agent
        # ----------------------------------------------------

        rl_agent = train_dqn(
            episodes=800,
            seed=int(random_seed),
        )

        # ----------------------------------------------------
        # STEP 7: Select defensive action
        # ----------------------------------------------------

        actions = []

        for i, score in enumerate(anomaly_scores):

            if detected[i]:

                action = choose_action(
                    float(score),
                    rl_agent,
                )

            else:

                action = "Monitor"

            actions.append(action)

        # ====================================================
        # IMPORTANT FIX
        # Add ALL ML results directly into dataframe
        # ====================================================

        df["anomaly_score"] = anomaly_scores
        df["detected"] = detected
        df["actions"] = actions

        # ----------------------------------------------------
        # Save simulation
        # ----------------------------------------------------

        st.session_state.simulation = {
            "graph": graph,
            "dataframe": df,
            "losses": losses,
            "threshold": threshold,
            "detected": detected,
            "anomaly_scores": anomaly_scores,
            "actions": actions,
        }


# ============================================================
# CHECK IF SIMULATION EXISTS
# ============================================================

simulation = st.session_state.simulation


if simulation is None:

    st.info(
        "👈 Configure the simulation from the sidebar and "
        "click **Run Defense Simulation**."
    )

    st.markdown(
        """
        ### 🔬 System Architecture

        ```text
        Network Traffic
                ↓
        Network Graph
                ↓
        Graph Representation
                ↓
        Self-Supervised Autoencoder
                ↓
        Reconstruction Error
                ↓
        Anomaly Score
                ↓
        Reinforcement Learning Agent
                ↓
        Defensive Response
        ```

        ### Technologies

        - Python
        - PyTorch
        - NetworkX
        - Self-Supervised Learning
        - Graph-based Machine Learning
        - Reinforcement Learning
        - Streamlit
        """
    )

    st.stop()


# ============================================================
# LOAD RESULTS
# ============================================================

graph = simulation["graph"]
df = simulation["dataframe"]
losses = simulation["losses"]
threshold = simulation["threshold"]
detected = simulation["detected"]
anomaly_scores = simulation["anomaly_scores"]
actions = simulation["actions"]


# ============================================================
# EVALUATION METRICS
# ============================================================

actual_attack = df["true_attack"].astype(bool).values

true_positive = int(
    np.sum(
        detected & actual_attack
    )
)

false_positive = int(
    np.sum(
        detected & ~actual_attack
    )
)

false_negative = int(
    np.sum(
        ~detected & actual_attack
    )
)

true_negative = int(
    np.sum(
        ~detected & ~actual_attack
    )
)


precision = (
    true_positive /
    (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0
)

recall = (
    true_positive /
    (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0
)

f1_score = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# TOP METRICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Security Overview</div>',
    unsafe_allow_html=True,
)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        "Network Nodes",
        len(df),
    )

with col2:
    st.metric(
        "Actual Attacks",
        int(actual_attack.sum()),
    )

with col3:
    st.metric(
        "Detected Threats",
        int(detected.sum()),
    )

with col4:
    st.metric(
        "Precision",
        f"{precision:.2%}",
    )

with col5:
    st.metric(
        "F1 Score",
        f"{f1_score:.2%}",
    )


# ============================================================
# SECURITY STATUS
# ============================================================

if detected.sum() > 0:

    st.markdown(
        f"""
        <div class="alert-box">
        🚨 <b>Security Alert</b><br>
        The system detected <b>{int(detected.sum())}</b>
        anomalous network nodes.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:

    st.markdown(
        """
        <div class="safe-box">
        🟢 <b>Network Status: Normal</b><br>
        No significant anomalies detected.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# NETWORK TOPOLOGY
# ============================================================

st.markdown(
    '<div class="section-title">🌐 1. Network Topology & Threat Detection</div>',
    unsafe_allow_html=True,
)

st.write(
    "Red nodes represent detected anomalies. "
    "Blue nodes represent normal network nodes."
)

fig, ax = plt.subplots(
    figsize=(13, 7)
)

positions = nx.spring_layout(
    graph,
    seed=int(random_seed),
    k=0.6,
)

# Edges
nx.draw_networkx_edges(
    graph,
    positions,
    ax=ax,
    alpha=0.25,
    width=1,
)

# Normal nodes
normal_nodes = [
    node
    for node in graph.nodes
    if not detected[node]
]

# Threat nodes
threat_nodes = [
    node
    for node in graph.nodes
    if detected[node]
]

nx.draw_networkx_nodes(
    graph,
    positions,
    nodelist=normal_nodes,
    node_size=250,
    ax=ax,
)

nx.draw_networkx_nodes(
    graph,
    positions,
    nodelist=threat_nodes,
    node_size=400,
    node_color="red",
    ax=ax,
)

nx.draw_networkx_labels(
    graph,
    positions,
    ax=ax,
    font_size=8,
)

ax.set_title(
    "Network Threat Map"
)

ax.axis("off")

st.pyplot(
    fig,
    clear_figure=True,
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.markdown(
    '<div class="section-title">🔍 2. Anomaly Detection</div>',
    unsafe_allow_html=True,
)

st.write(
    f"Anomaly threshold: **{threshold:.6f}**"
)


# Create display dataframe
anomaly_table = df.copy()

anomaly_table["status"] = np.where(
    anomaly_table["detected"],
    "🚨 ALERT",
    "🟢 NORMAL",
)


# Sort by anomaly score
anomaly_table = anomaly_table.sort_values(
    by="anomaly_score",
    ascending=False,
)


display_columns = [
    "node",
    "ip",
    "traffic_rate",
    "failed_connections",
    "packet_entropy",
    "anomaly_score",
    "status",
    "actions",
    "true_attack",
]


st.dataframe(
    anomaly_table[
        display_columns
    ],
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# TOP THREATS
# ============================================================

st.markdown(
    '<div class="section-title">🚨 3. Highest-Risk Network Nodes</div>',
    unsafe_allow_html=True,
)

top_threats = anomaly_table[
    anomaly_table["detected"] == True
].head(10)

if len(top_threats) > 0:

    st.dataframe(
        top_threats[
            [
                "node",
                "ip",
                "anomaly_score",
                "actions",
                "true_attack",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

else:

    st.success(
        "No high-risk nodes detected."
    )


# ============================================================
# RL RESPONSE
# ============================================================

st.markdown(
    '<div class="section-title">🤖 4. Autonomous RL Response</div>',
    unsafe_allow_html=True,
)

st.write(
    """
    The reinforcement-learning agent chooses a defensive
    action according to the detected anomaly severity.
    """
)


action_counts = (
    pd.Series(actions)
    .value_counts()
)


col1, col2 = st.columns(
    [2, 1]
)


with col1:

    st.bar_chart(
        action_counts
    )


with col2:

    st.markdown("### Defensive Actions")

    st.markdown(
        """
        **Monitor**  
        Continue observing the node.

        **Rate-limit**  
        Reduce suspicious traffic.

        **Isolate**  
        Simulate removing the node from the network.

        **Block source**  
        Simulate blocking malicious traffic.
        """
    )


# ============================================================
# RL RESPONSE TABLE
# ============================================================

response_table = anomaly_table[
    anomaly_table["detected"] == True
][
    [
        "node",
        "ip",
        "anomaly_score",
        "actions",
        "true_attack",
    ]
]


if len(response_table) > 0:

    st.dataframe(
        response_table,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# AUTOENCODER TRAINING
# ============================================================

st.markdown(
    '<div class="section-title">🧠 5. Self-Supervised Learning</div>',
    unsafe_allow_html=True,
)

st.write(
    """
    The graph autoencoder learns a compact representation
    and attempts to reconstruct the input features.
    Large reconstruction errors indicate unusual behavior.
    """
)


loss_dataframe = pd.DataFrame(
    {
        "Epoch": np.arange(
            1,
            len(losses) + 1,
        ),
        "Reconstruction Loss": losses,
    }
)

loss_dataframe = loss_dataframe.set_index(
    "Epoch"
)


st.line_chart(
    loss_dataframe
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.markdown(
    '<div class="section-title">📈 6. Detection Evaluation</div>',
    unsafe_allow_html=True,
)

confusion_matrix = pd.DataFrame(
    [
        [
            true_negative,
            false_positive,
        ],
        [
            false_negative,
            true_positive,
        ],
    ],
    index=[
        "Actual Normal",
        "Actual Attack",
    ],
    columns=[
        "Predicted Normal",
        "Predicted Attack",
    ],
)


st.dataframe(
    confusion_matrix,
    use_container_width=True,
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "True Positives",
        true_positive,
    )

with col2:
    st.metric(
        "False Positives",
        false_positive,
    )

with col3:
    st.metric(
        "False Negatives",
        false_negative,
    )

with col4:
    st.metric(
        "True Negatives",
        true_negative,
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.markdown(
    '<div class="section-title">📊 7. Model Performance</div>',
    unsafe_allow_html=True,
)

accuracy = (
    (true_positive + true_negative)
    /
    (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    )
    if (
        true_positive
        + true_negative
        + false_positive
        + false_negative
    ) > 0
    else 0
)

metric_table = pd.DataFrame(
    {
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
        ],
        "Value": [
            f"{accuracy:.2%}",
            f"{precision:.2%}",
            f"{recall:.2%}",
            f"{f1_score:.2%}",
        ],
    }
)

st.dataframe(
    metric_table,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# DOWNLOAD RESULTS
# ============================================================

st.markdown(
    '<div class="section-title">💾 8. Export Results</div>',
    unsafe_allow_html=True,
)

csv_data = df.to_csv(
    index=False
).encode("utf-8")


st.download_button(
    label="⬇️ Download Detection Results CSV",
    data=csv_data,
    file_name="cyber_defense_results.csv",
    mime="text/csv",
    use_container_width=True,
)


# ============================================================
# PIPELINE SUMMARY
# ============================================================

st.markdown(
    '<div class="section-title">🏗️ ML Pipeline</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    ```text
                    NETWORK
                       │
                       ▼
              ┌─────────────────┐
              │ Network Graph   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Graph Features  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Graph Autoencoder│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Anomaly Score   │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Threat Detection│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  RL Agent       │
              └────────┬────────┘
                       │
             ┌─────────┼─────────┐
             ▼         ▼         ▼
          Monitor  Rate-limit  Isolate
                       │
                       ▼
                  Block Source
    ```
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Autonomous Cyber Defense System | "
    "Educational defensive simulation | "
    "Synthetic network data"
)