# Classic Encoder (MLP 7→4)
#         ↓
# Quantum Bottleneck (PQC, learnable)
#         ↓
# Light MLPAttentionRes Tail (1–2 blocks)
#         ↓
# Linear → risk


import torch
import torch.nn as nn
import pennylane as qml


# ---------------------------------------------------------
# 1. Classic Encoder: 7 → 4 (tanh for angle embedding)
# ---------------------------------------------------------
class ClassicEncoder(nn.Module):
    def __init__(self, in_dim=7, latent_dim=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim),
            nn.Tanh()   # output ∈ [-1,1], multiply π inside PQC
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# 2. Quantum Bottleneck (4 qubits, learnable PQC)
# ---------------------------------------------------------
class QuantumBottleneck(nn.Module):
    def __init__(self, n_qubits=4, n_layers=2):
        super().__init__()
        self.n_qubits = n_qubits

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # inputs: (4,) in [-1,1]
            qml.AngleEmbedding(inputs * torch.pi, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.q_layer(x)   # (B, 4)


# ---------------------------------------------------------
# 3. Light MLPAttentionRes Tail (1–2 blocks)
# ---------------------------------------------------------
class FeatureAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        gate = self.fc(x)
        return x * gate


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        out = out + identity
        out = self.act(out)
        return out


class AttentionResTail(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=32, num_blocks=1):
        super().__init__()

        self.att = FeatureAttention(input_dim, hidden_dim=16)
        self.fc_in = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.att(x)
        x = self.act(self.fc_in(x))
        for blk in self.blocks:
            x = blk(x)
        return self.fc_out(x)


# ---------------------------------------------------------
# 4. Full Hybrid Model
# ---------------------------------------------------------
class HybridQNN_Bottleneck(nn.Module):
    """
    Classic Encoder (7→4)
        → Quantum Bottleneck (4 qubits)
        → Light MLPAttentionRes Tail (1–2 blocks)
        → Linear → risk
    """
    def __init__(self, encoder_dim=7, latent_dim=4, tail_blocks=1):
        super().__init__()
        self.encoder = ClassicEncoder(encoder_dim, latent_dim)
        self.qbottleneck = QuantumBottleneck(n_qubits=latent_dim, n_layers=2)
        self.tail = AttentionResTail(input_dim=latent_dim,
                                     hidden_dim=32,
                                     num_blocks=tail_blocks)

    def forward(self, x):
        u = self.encoder(x)          # (B, 4)
        hq = self.qbottleneck(u)     # (B, 4)
        out = self.tail(hq)          # (B, 1)
        return out
