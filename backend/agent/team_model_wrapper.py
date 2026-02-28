"""
team_model_wrapper.py
Loads trained team-level XGBoost models and provides roster evaluation inference.
Input: team context dict with lagged features (or direct overrides)
Output: predicted Net EPA, Win %, position impact, tier
"""

import os
import json
import numpy as np
import joblib

TEAM_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "ML", "team_model")
)

# Team tier boundaries (based on Net EPA distribution 2010-2024)
def _epa_to_tier(net_epa: float) -> str:
    if net_epa >= 0.20:
        return "Super Bowl Contender"
    elif net_epa >= 0.08:
        return "Playoff Team"
    elif net_epa >= -0.05:
        return "Fringe / Bubble"
    else:
        return "Rebuilding"


class TeamModelInference:
    """Wraps trained XGBoost team models for prediction and what-if analysis."""

    def __init__(self):
        self._models = {}
        self._scalers = {}
        self._feature_meta = {}
        self._feat_importance_epa = {}
        self._feat_importance_win = {}
        self._loaded = False

    def _load(self):
        if self._loaded:
            return
        try:
            self._models["epa"] = joblib.load(os.path.join(TEAM_MODEL_DIR, "epa_model.joblib"))
            self._models["win"] = joblib.load(os.path.join(TEAM_MODEL_DIR, "win_model.joblib"))
            self._scalers["epa"] = joblib.load(os.path.join(TEAM_MODEL_DIR, "epa_scaler.joblib"))
            self._scalers["win"] = joblib.load(os.path.join(TEAM_MODEL_DIR, "win_scaler.joblib"))

            with open(os.path.join(TEAM_MODEL_DIR, "feature_meta.json")) as f:
                self._feature_meta = json.load(f)

            self._loaded = True
        except FileNotFoundError as e:
            raise RuntimeError(
                f"Team model files not found in {TEAM_MODEL_DIR}. "
                f"Run train_team_model.py first. Error: {e}"
            )

    def _build_feature_vector(self, context: dict, feature_cols: list) -> np.ndarray:
        """
        Build the feature array from a context dict.
        Missing values default to column medians stored during training (approximated as 0 here;
        the scaler handles centering, and XGBoost handles missing via fill_value).
        """
        vec = []
        for col in feature_cols:
            val = context.get(col, np.nan)
            vec.append(float(val) if val is not None else np.nan)
        return np.array(vec, dtype=float)

    def predict(self, context: dict, position_overrides: dict | None = None) -> dict:
        """
        Make team predictions.

        Args:
            context: dict of feature values for the team-season (lagged features).
                     Can be loaded from team_dataset.csv for a specific team-year.
            position_overrides: optional dict of position grade overrides,
                                e.g. {"lag_qb_grade": 85.0, "lag_ol_grade": 70.0}

        Returns:
            dict with keys:
                predicted_net_epa, predicted_win_pct, tier,
                position_impact (dict of feature → importance contribution)
        """
        self._load()

        # Apply overrides
        ctx = dict(context)
        if position_overrides:
            ctx.update(position_overrides)

        # EPA model
        epa_feats = self._feature_meta.get("epa_features", [])
        vec_epa = self._build_feature_vector(ctx, epa_feats)
        # Fill NaN with 0 (scaler will normalise)
        vec_epa = np.where(np.isnan(vec_epa), 0.0, vec_epa)
        X_epa = self._scalers["epa"].transform(vec_epa.reshape(1, -1))
        pred_epa = float(self._models["epa"].predict(X_epa)[0])

        # Win % model
        win_feats = self._feature_meta.get("win_features", [])
        vec_win = self._build_feature_vector(ctx, win_feats)
        vec_win = np.where(np.isnan(vec_win), 0.0, vec_win)
        X_win = self._scalers["win"].transform(vec_win.reshape(1, -1))
        pred_win = float(self._models["win"].predict(X_win)[0])
        pred_win = max(0.0, min(1.0, pred_win))  # clip to [0, 1]

        # Position impact: map feature importance → position group labels
        position_impact = self._build_position_impact(epa_feats)

        return {
            "predicted_net_epa": round(pred_epa, 4),
            "predicted_win_pct": round(pred_win, 4),
            "predicted_wins": round(pred_win * 17, 1),
            "tier": _epa_to_tier(pred_epa),
            "position_impact": position_impact,
        }

    def _build_position_impact(self, epa_feats: list) -> dict:
        """
        Map Ridge feature coefficients to position group labels using scaled weights.
        """
        POSITION_MAP = {
            "lag_qb_grade":   "QB",
            "lag_ol_grade":   "OL",
            "lag_rb_grade":   "RB",
            "lag_wr_grade":   "WR",
            "lag_te_grade":   "TE",
            "lag_edge_grade": "EDGE",
            "lag_idl_grade":  "IDL",
            "lag_lb_grade":   "LB",
            "lag_cb_grade":   "CB",
            "lag_s_grade":    "S",
        }

        impact = {}
        # Get raw coefficients array from the EPA model
        if "epa" in self._models:
            coeffs = self._models["epa"].coef_
            for feat_name, pos_label in POSITION_MAP.items():
                if feat_name in epa_feats:
                    idx = epa_feats.index(feat_name)
                    # Use absolute value of coefficient so impact is purely magnitude
                    impact[pos_label] = round(float(abs(coeffs[idx])), 4)

        # Sort by impact descending
        return dict(sorted(impact.items(), key=lambda x: x[1], reverse=True))

    def project_roster_performance(self, team: str, year: int, players: list, dataset_path: str | None = None) -> dict:
        """
        Evaluate full roster based on projected grades.

        Args:
            team: team abbreviation (e.g. "KC", "BAL")
            year: base year (will predict year+1 performance)
            players: list of dicts: {"name": str, "position": str, "projected_grade": float}
            dataset_path: path to team_dataset.csv
        """
        position_grades = {}
        counts = {}
        for p in players:
            pos = p["position"].upper()
            grade = p["projected_grade"]
            position_grades[pos] = position_grades.get(pos, 0.0) + grade
            counts[pos] = counts.get(pos, 0) + 1
            
        overrides = {}
        for pos, total in position_grades.items():
            avg = total / counts[pos]
            feat_name = f"lag_{pos.lower()}_grade"
            # Special case for IDL vs DI edge cases, though the feature is `lag_idl_grade`
            if pos == "DI": feat_name = "lag_idl_grade"
            if pos == "HB": feat_name = "lag_rb_grade"
            if pos == "ED": feat_name = "lag_edge_grade"
            
            overrides[feat_name] = float(avg)
            
        return self.evaluate_roster(team, year, position_overrides=overrides, dataset_path=dataset_path)

    def evaluate_roster(
        self,
        team: str,
        year: int,
        position_overrides: dict | None = None,
        dataset_path: str | None = None,
    ) -> dict:
        """
        Load a specific team-year from the dataset and predict next season.

        Args:
            team: team abbreviation (e.g. "KC", "BAL")
            year: base year (will predict year+1 performance)
            position_overrides: optional grade overrides {"lag_qb_grade": 85.0}
            dataset_path: path to team_dataset.csv; defaults to TEAM_MODEL_DIR location
        """
        import pandas as pd

        path = dataset_path or os.path.join(TEAM_MODEL_DIR, "team_dataset.csv")
        df = pd.read_csv(path)

        # Match team+year
        row = df[(df["abbr"] == team.upper()) & (df["year"] == int(year))]
        if row.empty:
            raise ValueError(
                f"No data found for team={team}, year={year}. "
                f"Available teams: {sorted(df['abbr'].unique())}, "
                f"years: {df['year'].min()}–{df['year'].max()}"
            )

        context = row.iloc[0].to_dict()
        result = self.predict(context, position_overrides=position_overrides)
        result["team"] = team.upper()
        result["base_year"] = year
        result["predicted_year"] = year + 1
        return result


# Module-level singleton
_inference = None

def get_team_model() -> TeamModelInference:
    global _inference
    if _inference is None:
        _inference = TeamModelInference()
    return _inference


if __name__ == "__main__":
    # Quick smoke test
    model = get_team_model()

    print("=== Kansas City Chiefs 2023 (predict 2024) ===")
    result = model.evaluate_roster("KC", 2023)
    for k, v in result.items():
        print(f"  {k}: {v}")

    print("\n=== KC 2023 with backup QB (grade 55) ===")
    result2 = model.evaluate_roster("KC", 2023, position_overrides={"lag_qb_grade": 55.0})
    print(f"  Baseline EPA : {result['predicted_net_epa']:.4f}  → Backup QB: {result2['predicted_net_epa']:.4f}")
    print(f"  Baseline Win%: {result['predicted_win_pct']:.3f}   → Backup QB: {result2['predicted_win_pct']:.3f}")

    print("\n=== Baltimore Ravens 2023 (predict 2024) ===")
    bal = model.evaluate_roster("BAL", 2023)
    for k, v in bal.items():
        print(f"  {k}: {v}")
