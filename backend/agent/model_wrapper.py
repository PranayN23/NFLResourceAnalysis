
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- MODEL ARCHITECTURE (Must match training script) ---
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

class PlayerTransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes=3, kernel_size=1, num_heads=4, ff_dim=64, num_layers=2, dropout=0.1):
        super(PlayerTransformerClassifier, self).__init__()
        
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
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.time2vec(x)
        x = self.pad_proj(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1) 
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

# --- INFERENCE WRAPPER ---
class PlayerModelInference:
    def __init__(self, model_path, scaler=None):
        self.device = torch.device('cpu')
        self.features = [
            'grades_pass', 'grades_offense', 'qb_rating', 'adjusted_value',
            'Cap_Space', 'ypa', 'twp_rate', 'btt_rate', 'completion_percent'
        ]
        self.seq_len = 3
        self.num_classes = 3
        self.tier_names = ["Reserve/Poor", "Starter/Average", "Elite/High Quality"]

        # Load Model
        # We need to instantiate the model structure first with the same params as training
        # Features length = 9
        self.model = PlayerTransformerClassifier(input_dim=9, seq_len=3, num_classes=3).to(self.device)
        
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            print(f"Model loaded from {model_path}")
        else:
            print(f"WARNING: Model path {model_path} not found.")

        # TEAM REVIEW NOTE (STAGE 3 - INFERENCE):
        # Currently, we re-fit the Standard Scaler on the CSV every time the Agent starts.
        # PRODUCTION FIX: We must save `scaler.save` during training and load it here.
        # Otherwise, if the CSV changes, the model input distribution shifts.
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit_scaler(self, csv_path):
        """Fit scaler on the training data to ensure inference input standardization matches training."""
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Filter backups just like training
            if 'dropbacks' in df.columns:
                df = df[df['dropbacks'] >= 100]
            
            # Ensure columns exist
            available_feat = [f for f in self.features if f in df.columns]
            if len(available_feat) == len(self.features):
                self.scaler.fit(df[self.features])
                self.is_fitted = True
                print("Scaler fitted on historical data.")
            else:
                print("Missing features in CSV, cannot fit scaler.")
        else:
            print(f"CSV path {csv_path} not found. Scaler not fitted.")

    def predict(self, player_history_df):
        """
        Expects a DataFrame with at least 3 rows (years) for the player.
        Columns must include: grades_pass, grades_offense, qb_rating... etc.
        """
        if not self.is_fitted:
            print("Warning: Scaler not fitted. Predictions may be inaccurate.")
        
        # Ensure latest years are used
        player_history_df = player_history_df.sort_values('Year')
        
        if len(player_history_df) < self.seq_len:
            return "Insufficient Data", {}

        # Get last 3 years
        history = player_history_df.iloc[-self.seq_len:][self.features].copy()
        
        # Normalize
        if self.is_fitted:
            history[self.features] = self.scaler.transform(history[self.features])
            
        # Create tensor [1, seq_len, features]
        x = torch.tensor(history.values, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(x) # [1, 3] -> logits
            probs = torch.softmax(outputs, dim=1).squeeze().numpy()
            pred_idx = np.argmax(probs)
            
        prediction = self.tier_names[pred_idx]
        confidence = {
            self.tier_names[0]: float(probs[0]),
            self.tier_names[1]: float(probs[1]),
            self.tier_names[2]: float(probs[2])
        }
        
        return prediction, confidence
