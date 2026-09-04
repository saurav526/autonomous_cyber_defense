import random
import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim

ACTIONS = ["Monitor", "Rate-limit", "Isolate", "Block source"]

# 1. Synthetic network generation
def generate_network(n_nodes=40, attack_rate=0.10, seed=42):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    G = nx.barabasi_albert_graph(n_nodes, max(1, min(3, n_nodes - 1)), seed=seed)

    # Normal node telemetry.
    traffic = rng.lognormal(mean=2.2, sigma=0.45, size=n_nodes)
    failed = rng.poisson(lam=2.0, size=n_nodes)
    entropy = np.clip(rng.normal(0.72, 0.08, n_nodes), 0.2, 1.0)
    connections = rng.poisson(lam=8, size=n_nodes)

    n_attacks = max(1, int(n_nodes * attack_rate))
    attack_nodes = rng.choice(n_nodes, size=n_attacks, replace=False)

    # Inject synthetic attack behavior.
    traffic[attack_nodes] *= rng.uniform(2.5, 6.0, len(attack_nodes))
    failed[attack_nodes] += rng.poisson(15, len(attack_nodes))
    entropy[attack_nodes] = np.clip(
        entropy[attack_nodes] - rng.uniform(0.20, 0.45, len(attack_nodes)),
        0.02, 1.0
    )
    connections[attack_nodes] += rng.poisson(20, len(attack_nodes))

    node_features = np.column_stack([
        np.log1p(traffic),
        np.log1p(failed),
        entropy,
        np.log1p(connections),
        np.array([G.degree(i) for i in range(n_nodes)])
    ]).astype(np.float32)

    # Standardize feature columns.
    node_features = (node_features - node_features.mean(axis=0)) / (
        node_features.std(axis=0) + 1e-6
    )

    table = pd.DataFrame({
        "node": np.arange(n_nodes),
        "ip": [f"10.0.0.{i+1}" for i in range(n_nodes)],
        "traffic_rate": np.round(traffic, 2),
        "failed_connections": failed,
        "packet_entropy": np.round(entropy, 3),
        "connections": connections,
        "true_attack": False,
    })

    table.loc[attack_nodes, "true_attack"] = True

    return {
        "graph": G,
        "node_features": node_features,
        "attack_nodes": attack_nodes,
        "table": table,
    }


# ============================================================
# 2. Graph representation
# ============================================================
def build_graph_features(G, X):
    """
    Simple graph convolution:
        H = D^-1 A X W
    Here we first perform normalized neighborhood aggregation
    and concatenate it with the original node telemetry.

    This keeps the project easy to install while demonstrating
    the central idea of a GNN.
    """
    n = len(G.nodes)
    A = nx.to_numpy_array(G, nodelist=range(n), dtype=np.float32)
    A = A + np.eye(n, dtype=np.float32)

    degree = A.sum(axis=1)
    D_inv = np.diag(1.0 / np.sqrt(degree + 1e-8))
    A_norm = D_inv @ A @ D_inv

    neighborhood = A_norm @ X
    return np.concatenate([X, neighborhood], axis=1).astype(np.float32)


# ============================================================
# 3. Self-supervised graph anomaly detection
# ============================================================
class GraphAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden=16, latent=8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z


def train_autoencoder(X, epochs=40, lr=1e-3):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model = GraphAutoencoder(X.shape[1])

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    losses = []
    model.train()

    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed, _ = model(X_tensor)
        loss = criterion(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    # Store model globally for scoring in this small educational app.
    global _MODEL
    _MODEL = model
    return losses


_MODEL = None


def score_anomalies(X):
    X_tensor = torch.tensor(X, dtype=torch.float32)
    _MODEL.eval()

    with torch.no_grad():
        reconstructed, _ = _MODEL(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)

    return errors.numpy()


# ============================================================
# 4. Reinforcement-learning response policy
# ============================================================
# State = discretized anomaly severity.
# Action = defensive response.
#
# This is a small tabular DQN-style environment. It is deliberately
# self-contained so the Streamlit app can run on a normal laptop.
class QNetwork(nn.Module):
    def __init__(self, state_dim=1, n_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def environment_step(severity, action):
    """
    Simulated defensive environment.

    Higher severity means higher expected damage.
    Effective actions reduce future severity.
    """
    mitigation = {
        0: 0.00,  # Monitor
        1: 0.25,  # Rate-limit
        2: 0.65,  # Isolate
        3: 0.85,  # Block source
    }[action]

    new_severity = max(
        0.0,
        severity + np.random.normal(0, 0.08) - mitigation
    )

    # Reward balances security with intervention cost.
    damage = 4.0 * new_severity
    action_cost = [0.05, 0.20, 0.55, 0.75][action]
    reward = -(damage + action_cost)

    if new_severity < 0.15:
        reward += 1.0

    return float(new_severity), float(reward)


def train_dqn(episodes=800, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net = QNetwork()
    target = QNetwork()
    target.load_state_dict(net.state_dict())

    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    gamma = 0.90
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.995

    replay = []
    batch_size = 64

    for step in range(episodes):
        severity = random.random()

        for _ in range(12):
            state = torch.tensor([[severity]], dtype=torch.float32)

            if random.random() < epsilon:
                action = random.randrange(4)
            else:
                with torch.no_grad():
                    action = int(torch.argmax(net(state)).item())

            new_severity, reward = environment_step(severity, action)
            next_state = torch.tensor([[new_severity]], dtype=torch.float32)

            replay.append((state, action, reward, next_state, new_severity))
            if len(replay) > 2000:
                replay.pop(0)

            if len(replay) >= batch_size:
                batch = random.sample(replay, batch_size)

                states = torch.cat([b[0] for b in batch])
                actions = torch.tensor([b[1] for b in batch], dtype=torch.long)
                rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states = torch.cat([b[3] for b in batch])

                q = net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_q = target(next_states).max(dim=1).values
                    target_q = rewards + gamma * next_q

                loss = nn.functional.smooth_l1_loss(q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            severity = new_severity

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if step % 50 == 0:
            target.load_state_dict(net.state_dict())

    return net


def choose_action(score, qnetwork):
    # Convert anomaly score into 0..1 severity using a smooth transform.
    severity = float(1.0 - np.exp(-max(score, 0.0)))

    with torch.no_grad():
        q = qnetwork(torch.tensor([[severity]], dtype=torch.float32))
        action_id = int(torch.argmax(q).item())

    return ACTIONS[action_id]
