import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from utils.dataset import WildfireDataset

# -----------------------------
# 导入 8 个模型
# -----------------------------
from models.mlp_net import MLPPlain, MLPResNet
from models.mlp_attention import MLPAttentionPlain, MLPAttentionRes
from models.cnn1d import CNN1DPlain, CNN1DResNet
from models.naf1d import NAFNet1DPlain, NAFNet1DRes

# -----------------------------
# 评估指标
# -----------------------------
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# -----------------------------
# risk@k%
# -----------------------------
def risk_at_k_percent(risk, y_true, k):
    N = len(risk)
    top_k = int(N * k)
    idx = np.argsort(risk)[-top_k:]
    return y_true[idx].mean()


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 加载数据
    # -----------------------------
    df = pd.read_csv("Task1_wildfire_weather_risk.csv")
    df["year"] = pd.to_datetime(df["year_month"]).dt.year

    # 2021 = test set
    test_df = df[df["year"] == 2021].copy()

    feature_cols = [
        "lat", "lon",
        "month_sin", "month_cos",
        "avg_tmax_c", "avg_tmin_c", "tot_prcp_mm"
    ]

    X_val = test_df[feature_cols].values
    y_val = test_df["kernel_risk_target"].values
    y_val_fire = test_df["has_fire"].values
    p = y_val_fire.mean()

    val_ds = WildfireDataset(X_val, y_val)
    val_loader = DataLoader(val_ds, batch_size=512, shuffle=False)

    # -----------------------------
    # 定义 8 个模型（结构必须和训练时一致）
    # -----------------------------
    HIDDEN_DIM = 64
    CHANNELS = 32
    NUM_LAYERS = 3
    NUM_BLOCKS = 3

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
    # 结果表
    # -----------------------------
    table_reg = []
    table_cls = []
    table_risk = []

    os.makedirs("results", exist_ok=True)

    # -----------------------------
    # 依次评估所有模型
    # -----------------------------
    for name, model in models.items():
        print(f"\n==============================")
        print(f"Evaluating model: {name}")
        print("==============================")

        # ---- load best model ----
        model_path = f"saved_models/{name}.pth"
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        # ---- predict ----
        preds = []
        with torch.no_grad():
            for Xb, _ in val_loader:
                Xb = Xb.to(device)
                pred = model(Xb).cpu().numpy().reshape(-1)
                preds.append(pred)

        pred_val = np.concatenate(preds)
        risk_val = pred_val

        # -----------------------------
        # 回归指标
        # -----------------------------
        r2 = r2_score(y_val, pred_val)
        mae = mean_absolute_error(y_val, pred_val)
        mse = mean_squared_error(y_val, pred_val)
        table_reg.append([name, r2, mae, mse])

        # -----------------------------
        # 分类指标（prevalence-matched）
        # -----------------------------
        threshold = np.quantile(risk_val, 1 - p)
        pred_bin = (risk_val >= threshold).astype(int)

        acc = accuracy_score(y_val_fire, pred_bin)
        prec = precision_score(y_val_fire, pred_bin, zero_division=0)
        rec = recall_score(y_val_fire, pred_bin, zero_division=0)
        f1 = f1_score(y_val_fire, pred_bin, zero_division=0)
        auc = roc_auc_score(y_val_fire, risk_val)

        table_cls.append([name, acc, prec, rec, f1, auc])

        # -----------------------------
        # risk@k%
        # -----------------------------
        r1 = risk_at_k_percent(risk_val, y_val_fire, 0.01)
        r3 = risk_at_k_percent(risk_val, y_val_fire, 0.03)
        r5 = risk_at_k_percent(risk_val, y_val_fire, 0.05)
        r10 = risk_at_k_percent(risk_val, y_val_fire, 0.10)
        r20 = risk_at_k_percent(risk_val, y_val_fire, 0.20)

        table_risk.append([name, r1, r3, r5, r10, r20])

        print(f"{name} risk@20% = {r20:.4f}")

    # -----------------------------
    # 保存 3 张表
    # -----------------------------
    df_reg = pd.DataFrame(table_reg, columns=["Model", "R2", "MAE", "MSE"])
    df_cls = pd.DataFrame(table_cls, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "AUC"])
    df_risk = pd.DataFrame(table_risk, columns=["Model", "risk@1%", "risk@3%", "risk@5%", "risk@10%", "risk@20%"])

    df_reg.to_csv("results/table_reg.csv", index=False)
    df_cls.to_csv("results/table_cls.csv", index=False)
    df_risk.to_csv("results/table_risk.csv", index=False)

    print("\n===== Evaluation Complete =====")
    print("Saved tables → results/")


if __name__ == "__main__":
    main()
