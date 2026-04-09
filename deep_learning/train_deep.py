import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from utils.dataset import WildfireDataset

# -----------------------------
# 导入 8 个模型
# -----------------------------
from models.mlp_net import MLPPlain, MLPResNet
from models.mlp_attention import MLPAttentionPlain, MLPAttentionRes
from models.cnn1d import CNN1DPlain, CNN1DResNet
from models.naf1d import NAFNet1DPlain, NAFNet1DRes

# -----------------------------
# 全局参数
# -----------------------------
BATCH = 256
EPOCHS = 500
LR = 1e-3

HIDDEN_DIM = 64
CHANNELS = 32
NUM_LAYERS = 3
NUM_BLOCKS = 3


# ============================================================
# 训练函数（保存 best model）
# ============================================================
def train_one_model(model, train_loader, val_loader, device, lr=1e-3, epochs=200, name="model"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):

        # -----------------------------
        # Train
        # -----------------------------
        model.train()
        train_losses = []

        pbar = tqdm(train_loader, desc=f"[{name}] Train Epoch {epoch}/{epochs}", ncols=120)
        for X, y in pbar:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            pbar.set_postfix({"train_loss": f"{loss.item():.6f}"})

        train_loss = float(np.mean(train_losses))

        # -----------------------------
        # Validation
        # -----------------------------
        model.eval()
        val_losses = []

        pbar = tqdm(val_loader, desc=f"[{name}] Val   Epoch {epoch}/{epochs}", ncols=120)
        with torch.no_grad():
            for X, y in pbar:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = criterion(pred, y)
                val_losses.append(loss.item())
                pbar.set_postfix({"val_loss": f"{loss.item():.6f}"})

        val_loss = float(np.mean(val_losses))

        # -----------------------------
        # Save best
        # -----------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            print(f"\n[{name}] New BEST at epoch {epoch}: val_loss={val_loss:.6f}\n")

    # return best model
    model.load_state_dict(best_state)
    return model


# ============================================================
# 主程序
# ============================================================
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 加载数据
    # -----------------------------
    df = pd.read_csv("Task1_wildfire_weather_risk.csv")

    df["year"] = pd.to_datetime(df["year_month"]).dt.year

    # 只用 2018–2020 做 train/val
    panel_df = df[(df["year"] >= 2018) & (df["year"] <= 2020)].copy()

    feature_cols = [
        "lat", "lon",
        "month_sin", "month_cos",
        "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm"
    ]

    X = panel_df[feature_cols].values
    y = panel_df["kernel_risk_target"].values

    # -----------------------------
    # 2018–2020 内部做 80/20 split
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_ds = WildfireDataset(X_train, y_train)
    val_ds = WildfireDataset(X_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    # -----------------------------
    # 定义 8 个模型
    # -----------------------------
    models = {
        "MLPPlain": MLPPlain(input_dim=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS),
        "MLPResNet": MLPResNet(input_dim=7, hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS),

        "MLPAttPlain": MLPAttentionPlain(input_dim=7, hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS),
        "MLPAttRes": MLPAttentionRes(input_dim=7, hidden_dim=HIDDEN_DIM, num_blocks=NUM_BLOCKS),

        "CNN1DPlain": CNN1DPlain(input_dim=7, hidden_channels=CHANNELS, num_layers=NUM_LAYERS),
        "CNN1DResNet": CNN1DResNet(input_dim=7, hidden_channels=CHANNELS, num_blocks=NUM_BLOCKS),

        "NAF1DPlain": NAFNet1DPlain(input_dim=7, channels=CHANNELS, num_layers=NUM_LAYERS),
        "NAF1DRes": NAFNet1DRes(input_dim=7, channels=CHANNELS, num_blocks=NUM_BLOCKS),
    }

    # -----------------------------
    # 创建保存目录
    # -----------------------------
    os.makedirs("saved_models", exist_ok=True)

    # -----------------------------
    # 依次训练所有模型
    # -----------------------------
    for name, model in models.items():
        print(f"\n==============================")
        print(f"Training model: {name}")
        print("==============================")

        best_model = train_one_model(
            model, train_loader, val_loader,
            device, lr=LR, epochs=EPOCHS, name=name
        )

        # ---- save best model ----
        torch.save(best_model.state_dict(), f"saved_models/{name}.pth")
        print(f"[{name}] Saved best model → saved_models/{name}.pth")


if __name__ == "__main__":
    main()
