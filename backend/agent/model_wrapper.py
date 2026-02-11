import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- MODEL ARCHITECTURE (Hybrid Regressor) ---
class Time2Vec(nn.Module):
    def __init__(self, input_dim, kernel_size=1):
        super(Time2Vec, self).__init__()
        self.k = kernel_size
        self.input_dim = input_dim
        self.w0 = nn.Parameter(torch.randn(input_dim, 1)) 
        self.b0 = nn.Parameter(torch.randn(input_dim, 1))
        self.wk = nn.Parameter(torch.randn(input_dim, kernel_size))
        self.bk = nn.Parameter(torch.randn(input_dim, kernel_size))
        
    def forward(self, x):
        x_uns = x.unsqueeze(-1)
        linear = x_uns * self.w0 + self.b0
        periodic = torch.sin(x_uns * self.wk + self.bk)
        out = torch.cat([linear, periodic], dim=-1)
        out = out.reshape(x.size(0), x.size(1), -1)
        return out

class PlayerTransformerRegressor(nn.Module):
    def __init__(self, input_dim, seq_len, kernel_size=1, num_heads=4, ff_dim=64, num_layers=2, dropout=0.1):
        super(PlayerTransformerRegressor, self).__init__()
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
            batch_first=True
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.embed_dim))
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.regressor = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, mask=None):
        x = self.time2vec(x)
        x = self.pad_proj(x)
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        if mask is not None:
            cls_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
        else:
            full_mask = None
            
        x = x + self.pos_embedding
        x = self.transformer_encoder(x, src_key_padding_mask=full_mask)
        x = x[:, 0, :]
        x = self.regressor(x)
        return x

# --- INFERENCE WRAPPER ---
class PlayerModelInference:
    def __init__(self, model_path, scaler_path=None):
        self.device = torch.device('cpu')
        self.features = [
            'grades_pass', 'grades_offense', 'qb_rating', 'adjusted_value',
            'Cap_Space', 'ypa', 'twp_rate', 'btt_rate', 'completion_percent',
            'years_in_league', 'delta_grade', 'delta_epa', 'delta_btt',
            'team_performance_proxy'
        ]
        self.max_seq_len = 5
        self.model = PlayerTransformerRegressor(input_dim=len(self.features), seq_len=5).to(self.device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Model loaded from {model_path}")
        
        self.scaler = None
        self.is_fitted = False
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
            self.is_fitted = True
            print(f"Scaler loaded from {scaler_path}")

    def predict(self, player_history_df):
        if not self.is_fitted:
            print("Warning: Scaler not loaded.")
        
        player_history_df = player_history_df.copy()
        player_history_df['adjusted_value'] = pd.to_numeric(player_history_df['adjusted_value'], errors='coerce')
        player_history_df = player_history_df.sort_values('Year')
        
        # Engineering
        player_history_df["years_in_league"] = player_history_df.groupby("player").cumcount()
        player_history_df["delta_grade"] = player_history_df["grades_offense"].diff().fillna(0)
        player_history_df["delta_epa"]   = player_history_df["Net EPA"].diff().fillna(0)
        player_history_df["delta_btt"]   = player_history_df["btt_rate"].diff().fillna(0)
        
        # Team Proxy handling
        if 'team_performance_proxy' not in player_history_df.columns:
            player_history_df['team_performance_proxy'] = player_history_df['Net EPA'].fillna(0)
        
        # Final NaN clean
        player_history_df = player_history_df.fillna(0)

        history = player_history_df.iloc[-self.max_seq_len:][self.features].copy()
        actual_len = len(history)
        
        if self.is_fitted:
            history[self.features] = self.scaler.transform(history[self.features])
            
        vals = history.values
        padding_len = self.max_seq_len - actual_len
        if padding_len > 0:
            padded_x = np.vstack([np.zeros((padding_len, len(self.features))), vals])
            mask = [True] * padding_len + [False] * actual_len
        else:
            padded_x = vals
            mask = [False] * actual_len

        x = torch.tensor(padded_x, dtype=torch.float32).unsqueeze(0)
        m = torch.tensor(mask, dtype=torch.bool).unsqueeze(0)
        
        with torch.no_grad():
            output = self.model(x, mask=m).squeeze().item()
            
        if output >= 80.0: tier = "Elite"
        elif output >= 60.0: tier = "Starter"
        else: tier = "Reserve/Poor"
        
        return tier, {"predicted_grade": output}
