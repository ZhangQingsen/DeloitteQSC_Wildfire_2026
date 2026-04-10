# Classic Encoder (MLP 7→4)
#         ↓
# Quantum Bottleneck (PQC, learnable)
#         ↓
# Quantum Tail Block（量子 residual / quantum linear）
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
            qml.AngleEmbedding(inputs * torch.pi, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(circuit, weight_shapes)

    def forward(self, x):
        return self.q_layer(x)   # (B, 4)


# ---------------------------------------------------------
# 3. Quantum Tail Block (Quantum Residual Block)
# ---------------------------------------------------------
class QuantumResidualBlock(nn.Module):
    """
    Quantum residual block:
        h → PQC → Linear → + h
    PQC acts as a nonlinear transformation on the 4-dim quantum features.
    """
    def __init__(self, n_qubits=4, n_layers=1):
        super().__init__()
        self.n_qubits = n_qubits

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch")
        def qblock(inputs, weights):
            qml.AngleEmbedding(inputs * torch.pi, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        self.q_layer = qml.qnn.TorchLayer(qblock, weight_shapes)

        # classical linear refinement after PQC
        self.fc = nn.Linear(n_qubits, n_qubits)
        self.act = nn.ReLU()

    def forward(self, x):
        identity = x
        q_out = self.q_layer(x)      # (B, 4)
        out = self.fc(q_out)
        out = out + identity         # residual
        return self.act(out)


# ---------------------------------------------------------
# 4. Final Linear Head
# ---------------------------------------------------------
class FinalHead(nn.Module):
    def __init__(self, in_dim=4):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x)


# ---------------------------------------------------------
# 5. Full Hybrid Model (Quantum Tail)
# ---------------------------------------------------------
class HybridQNN_QuantumTail(nn.Module):
    """
    Classic Encoder (7→4)
        → Quantum Bottleneck (4 qubits)
        → Quantum Tail Block (1–2 blocks)
        → Linear → risk
    """
    def __init__(self, encoder_dim=7, latent_dim=4, tail_blocks=1):
        super().__init__()

        self.encoder = ClassicEncoder(encoder_dim, latent_dim)
        self.qbottleneck = QuantumBottleneck(n_qubits=latent_dim, n_layers=2)

        self.tail_blocks = nn.ModuleList([
            QuantumResidualBlock(n_qubits=latent_dim, n_layers=1)
            for _ in range(tail_blocks)
        ])

        self.head = FinalHead(in_dim=latent_dim)

    def forward(self, x):
        u = self.encoder(x)          # (B, 4)
        hq = self.qbottleneck(u)     # (B, 4)

        for blk in self.tail_blocks:
            hq = blk(hq)

        out = self.head(hq)          # (B, 1)
        return out
