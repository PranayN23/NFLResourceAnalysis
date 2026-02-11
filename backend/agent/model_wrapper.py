
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


# --- SHARED MODEL ARCHITECTURE (matches training scripts) ---
class Time2Vec(nn.Module):
    def __init__(self, input_dim: int, kernel_size: int = 1):
        super().__init__()
        self.k = kernel_size
        self.input_dim = input_dim
        self.w0 = nn.Parameter(torch.randn(input_dim, 1))
        self.b0 = nn.Parameter(torch.randn(input_dim, 1))
        self.wk = nn.Parameter(torch.randn(input_dim, kernel_size))
        self.bk = nn.Parameter(torch.randn(input_dim, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_uns = x.unsqueeze(-1)
        linear = x_uns * self.w0 + self.b0
        periodic = torch.sin(x_uns * self.wk + self.bk)
        out = torch.cat([linear, periodic], dim=-1)
        out = out.reshape(x.size(0), x.size(1), -1)
        return out


class PlayerTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        num_classes: int = 3,
        kernel_size: int = 1,
        num_heads: int = 4,
        ff_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.time2vec = Time2Vec(input_dim, kernel_size)
        self.embed_dim = input_dim * (kernel_size + 1)

        if self.embed_dim % num_heads != 0:
            new_dim = (self.embed_dim // num_heads + 1) * num_heads
            self.pad_proj = nn.Linear(self.embed_dim, new_dim)
            self.embed_dim = new_dim
        else:
            self.pad_proj = nn.Identity()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.time2vec(x)
        x = self.pad_proj(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x


class BasePlayerModelInference:
    """
    Shared inference utilities for sequence-based Time2Vec+Transformer models.
    Specialized subclasses set features and tier naming.
    """

    def __init__(self, model_path: str, features: List[str], seq_len: int = 3):
        self.device = torch.device("cpu")
        self.features = features
        self.seq_len = seq_len
        self.num_classes = 3
        self.tier_names = ["Reserve/Poor", "Starter/Average", "Elite/High Quality"]

        self.model = PlayerTransformerClassifier(
            input_dim=len(self.features),
            seq_len=self.seq_len,
            num_classes=self.num_classes,
        ).to(self.device)

        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Model loaded from {model_path}")
        else:
            print(f"WARNING: Model path {model_path} not found.")

        # REVIEW NOTE:
        # We still re-fit the StandardScaler from CSV on startup (same behavior as QB).
        # For production, persist the scaler during training and load it here.
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _fit_scaler_from_df(self, df: pd.DataFrame):
        available_feat = [f for f in self.features if f in df.columns]
        if len(available_feat) == len(self.features):
            self.scaler.fit(df[self.features])
            self.is_fitted = True
            print("Scaler fitted on historical data.")
        else:
            print(
                f"Missing features in CSV, cannot fit scaler. "
                f"Expected: {self.features}, got: {available_feat}"
            )

    def predict(self, player_history_df: pd.DataFrame):
        """
        Expects a DataFrame with at least `seq_len` rows for the player.
        """
        if not self.is_fitted:
            print("Warning: Scaler not fitted. Predictions may be inaccurate.")

        player_history_df = player_history_df.sort_values("Year")

        if len(player_history_df) < self.seq_len:
            return "Insufficient Data", {}

        history = player_history_df.iloc[-self.seq_len :][self.features].copy()

        if self.is_fitted:
            history[self.features] = self.scaler.transform(history[self.features])

        x = torch.tensor(history.values, dtype=torch.float32).unsqueeze(0).to(
            self.device
        )

        with torch.no_grad():
            outputs = self.model(x)
            probs = torch.softmax(outputs, dim=1).squeeze().numpy()
            pred_idx = int(np.argmax(probs))

        prediction = self.tier_names[pred_idx]
        confidence = {
            self.tier_names[0]: float(probs[0]),
            self.tier_names[1]: float(probs[1]),
            self.tier_names[2]: float(probs[2]),
        }

        return prediction, confidence


class PlayerModelInference(BasePlayerModelInference):
    """
    QB-specific inference (matches Player_Model_QB.py).
    """

    def __init__(self, model_path: str):
        features = [
            "grades_pass",
            "grades_offense",
            "qb_rating",
            "adjusted_value",
            "Cap_Space",
            "ypa",
            "twp_rate",
            "btt_rate",
            "completion_percent",
        ]
        super().__init__(model_path=model_path, features=features, seq_len=3)

    def fit_scaler(self, csv_path: str):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "dropbacks" in df.columns:
                df = df[df["dropbacks"] >= 100]
            self._fit_scaler_from_df(df)
        else:
            print(f"CSV path {csv_path} not found. Scaler not fitted.")


class EdgePlayerModelInference(BasePlayerModelInference):
    """
    EDGE-specific inference (matches Player_Model_EDGE.py).
    """

    def __init__(self, model_path: str):
        features = [
            "grades_defense",
            "grades_pass_rush_defense",
            "grades_run_defense",
            "grades_tackle",
            "Cap_Space",
            "Net EPA",
            "snap_counts_pass_rush",
            "snap_counts_run_defense",
            "total_pressures",
        ]
        super().__init__(model_path=model_path, features=features, seq_len=3)

    def fit_scaler(self, csv_path: str):
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            if "position" in df.columns:
                df = df[df["position"] == "ED"]
            if "snap_counts_defense" in df.columns:
                df = df[df["snap_counts_defense"] >= 300]
            self._fit_scaler_from_df(df)
        else:
            print(f"CSV path {csv_path} not found. Scaler not fitted.")

