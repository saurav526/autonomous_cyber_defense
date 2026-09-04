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
    ACTIONS,
)

st.set_page_config(page_title="Autonomous Cyber Defense", page_icon="🛡️", layout="wide")

st.title("🛡️ Autonomous Cyber Defense System")
st.caption("Defensive simulation: graph anomaly detection + reinforcement-learning response")

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Simulation Controls")
n_nodes = st.sidebar.slider("Network nodes", 20, 100, 40)
attack_rate = st.sidebar.slider("Attack intensity", 0.02, 0.30, 0.10, 0.01)
epochs = st.sidebar.slider("Autoencoder epochs", 10, 100, 40, 10)
seed = st.sidebar.number_input("Random seed", min_value=1, max_value=9999, value=42)

if "result" not in st.session_state:
    st.session_state.result = None

run = st.sidebar.button("🚀 Run Defense Simulation", type="primary")

if run:
    with st.spinner("Generating traffic, learning graph representations, detecting anomalies, and selecting responses..."):
        data = generate_network(n_nodes=n_nodes, attack_rate=attack_rate, seed=int(seed))
        X = build_graph_features(data["graph"], data["node_features"])

        losses = train_autoencoder(X, epochs=epochs)
        scores = score_anomalies(X)

        threshold = np.quantile(scores, 0.90)
        detected = scores >= threshold

        # Train a small DQN on the simulated defensive environment.
        qtable = train_dqn(episodes=800, seed=int(seed))
        actions = [choose_action(float(s), qtable) if detected[i] else "Monitor"
                   for i, s in enumerate(scores)]

        data["scores"] = scores
        data["detected"] = detected
        data["actions"] = actions
        data["threshold"] = threshold
        data["losses"] = losses
        st.session_state.result = data

result = st.session_state.result

if result is None:
    st.info("Set the controls and click **Run Defense Simulation** to generate a live defensive simulation.")
    st.markdown("""
### Pipeline

`Network Traffic → Graph Construction → Graph Representation → Autoencoder → Anomaly Score → RL Response → Action`

The demo intentionally uses **synthetic network traffic** so it can be run locally without collecting real user traffic.
""")
    st.stop()

# -----------------------------
# Metrics
# -----------------------------
df = result["table"]
detected = result["detected"]
true_attack = df["true_attack"].to_numpy(dtype=bool)

tp = int(np.sum(detected & true_attack))
fp = int(np.sum(detected & ~true_attack))
fn = int(np.sum(~detected & true_attack))
tn = int(np.sum(~detected & ~true_attack))

precision = tp / (tp + fp) if tp + fp else 0
recall = tp / (tp + fn) if tp + fn else 0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Nodes", len(df))
c2.metric("Detected anomalies", int(detected.sum()))
c3.metric("Precision", f"{precision:.2%}")
c4.metric("Recall", f"{recall:.2%}")
c5.metric("F1", f"{f1:.2%}")

# -----------------------------
# Network graph
# -----------------------------
st.subheader("1. Network topology and detected threats")

fig, ax = plt.subplots(figsize=(10, 6))
G = result["graph"]
pos = nx.spring_layout(G, seed=int(seed))

normal_nodes = [i for i in G.nodes if not detected[i]]
alert_nodes = [i for i in G.nodes if detected[i]]

nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25)
nx.draw_networkx_nodes(G, pos, nodelist=normal_nodes, node_size=180, ax=ax)
nx.draw_networkx_nodes(G, pos, nodelist=alert_nodes, node_size=300, ax=ax, node_color="red")
nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
ax.set_axis_off()
st.pyplot(fig, clear_figure=True)

# -----------------------------
# Anomaly ranking
# -----------------------------
st.subheader("2. Anomaly detection")

ranked = df.sort_values("anomaly_score", ascending=False).copy()
ranked["status"] = np.where(ranked["detected"], "🚨 ALERT", "Normal")
st.dataframe(
    ranked[["node", "ip", "traffic_rate", "failed_connections",
            "packet_entropy", "anomaly_score", "status", "true_attack", "actions"]]
    .head(15),
    use_container_width=True,
    hide_index=True,
)

# -----------------------------
# Response actions
# -----------------------------
st.subheader("3. Autonomous response selected by RL agent")

action_counts = pd.Series(result["actions"]).value_counts()
st.bar_chart(action_counts)

st.dataframe(
    ranked[ranked["detected"]][
        ["node", "ip", "anomaly_score", "actions", "true_attack"]
    ].head(20),
    use_container_width=True,
    hide_index=True,
)

st.markdown("""
**Defensive actions in the simulation:**
- **Monitor** — continue observing the host.
- **Rate-limit** — reduce suspicious traffic.
- **Isolate** — remove a suspicious host from the simulated network.
- **Block source** — block the simulated source of malicious traffic.
""")

# -----------------------------
# Training curve
# -----------------------------
st.subheader("4. Self-supervised representation learning")

loss_df = pd.DataFrame({
    "epoch": np.arange(1, len(result["losses"]) + 1),
    "reconstruction_loss": result["losses"]
}).set_index("epoch")
st.line_chart(loss_df)

# -----------------------------
# Confusion matrix
# -----------------------------
st.subheader("5. Detection evaluation")

cm = pd.DataFrame(
    [[tn, fp], [fn, tp]],
    index=["Actual Normal", "Actual Attack"],
    columns=["Predicted Normal", "Predicted Attack"],
)
st.dataframe(cm, use_container_width=True)

st.success(
    f"Simulation complete. The anomaly threshold was {result['threshold']:.4f}. "
    "The response layer then selected defensive actions for detected nodes."
)

st.download_button(
    "⬇️ Download simulation results",
    df.to_csv(index=False).encode("utf-8"),
    file_name="cyber_defense_results.csv",
    mime="text/csv",
)
