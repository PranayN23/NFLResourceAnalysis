import os
import pandas as pd
from typing import Optional, Dict, Any

from backend.ML.team_model.correlation_tool import get_team_context as _get_team_context
from backend.ML.team_model.correlation_tool import compute_correlation

# Minimal per-position model wrappers as available in /backend/agent
from backend.agent.model_wrapper import PlayerModelInference
from backend.agent.rb_model_wrapper import RBModelInference
from backend.agent.wr_model_wrapper import WRModelInference
from backend.agent.te_model_wrapper import TEModelInference
from backend.agent.ol_model_wrapper import OLModelInference
from backend.agent.di_model_wrapper import DIModelInference
from backend.agent.ed_model_wrapper import EDModelInference
from backend.agent.lb_model_wrapper import LBModelInference
from backend.agent.cb_model_wrapper import CBModelInference
from backend.agent.s_model_wrapper import SModelInference


DEFAULT_CORRELATION_CSV = os.path.abspath(os.path.join(os.path.dirname(__file__), "correlation_audit.csv"))

POSITION_FILES = {
    "qb": os.path.abspath("backend/ML/QB.csv"),
    "rb": os.path.abspath("backend/ML/HB.csv"),
    "wr": os.path.abspath("backend/ML/WR.csv"),
    "te": os.path.abspath("backend/ML/TPFF.csv"),
    "ol": os.path.abspath("backend/ML/OLPFF.csv"),
    "di": os.path.abspath("backend/ML/DI.csv"),
    "ed": os.path.abspath("backend/ML/ED.csv"),
    "lb": os.path.abspath("backend/ML/LB.csv"),
    "cb": os.path.abspath("backend/ML/CB.csv"),
    "s": os.path.abspath("backend/ML/S.csv"),
}

MODEL_REGISTRY = {
    "qb": {
        "class": PlayerModelInference,
        "transformer": os.path.abspath("backend/ML/QB_Pranay_Transformers/best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/QB_Pranay_Transformers/player_scaler.joblib"),
        "xgb": None,
    },
    "rb": {
        "class": RBModelInference,
        "transformer": os.path.abspath("backend/ML/RB_Pranay_Transformers/rb_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/RB_Pranay_Transformers/rb_player_scaler.joblib"),
        "xgb": None,
    },
    "wr": {
        "class": WRModelInference,
        "transformer": os.path.abspath("backend/ML/WR_Pranay_Transformers/wr_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/WR_Pranay_Transformers/wr_player_scaler.joblib"),
        "xgb": None,
    },
    "te": {
        "class": TEModelInference,
        "transformer": os.path.abspath("backend/ML/TightEnds/models/transformer_te_ensemble.pth"),
        "scaler": os.path.abspath("backend/ML/TightEnds/models/transformer_scaler.pkl"),
        "xgb": os.path.abspath("backend/ML/TightEnds/models/xgb_te_ensemble.json"),
    },
    "ol": {
        "class": OLModelInference,
        "transformer": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_player_scaler.joblib"),
        "xgb": None,
    },
    "di": {
        "class": DIModelInference,
        "transformer": os.path.abspath("backend/ML/DI_Pranay_Transformers/di_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/DI_Pranay_Transformers/di_player_scaler.joblib"),
        "xgb": None,
    },
    "ed": {
        "class": EDModelInference,
        "transformer": os.path.abspath("backend/ML/ED_Transformers/ed_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/ED_Transformers/ed_player_scaler.joblib"),
        "xgb": None,
    },
    # CB/S/LB not wired to dedicated agent wrappers here, will fallback by data average.
}


def get_team_context(team: str, season: Optional[int] = None) -> Dict[str, Any]:
    # use correlation tool provided context. If season unspecified, pick latest year available.
    if season is None:
        # find from team dataset quickly: maybe 2023 or max found
        df = _get_team_context(team, 9999) if False else None
    # re-use existing get_team_context (currently requires season)
    if season is None:
        # fallback: choose 2023 with try/catch
        for year in [2024, 2023, 2022, 2021, 2020]:
            try:
                ctx = _get_team_context(team, year)
                if "error" not in ctx:
                    season = year
                    break
            except Exception:
                continue
        if season is None:
            return {"error": "Unable to resolve season for team context"}

    return _get_team_context(team, season)


def positional_value_lookup(position: str) -> Dict[str, Any]:
    if not os.path.exists(DEFAULT_CORRELATION_CSV):
        return {"error": "correlation_audit.csv not found"}

    df = pd.read_csv(DEFAULT_CORRELATION_CSV)
    # dynamic: both specific and generic position labels
    position = position.lower()
    candidates = [f"{position}_grade_win_pct", f"{position}_grade_mean", position]

    # determine weight by abs(correlation) in audit data
    matches = df[df["feature"].isin(candidates)]
    if matches.empty:
        # fallback to best match by positional keyword
        matches = df[df["feature"].str.contains(position, case=False, na=False)]

    if matches.empty:
        return {"position": position, "importance_score": 0.0, "message": "no correlation row found"}

    row = matches.loc[matches["correlation"].abs().idxmax()]
    importance = float(abs(row["correlation"]))

    return {
        "position": position,
        "importance_score": importance,
        "source_feature": row["feature"],
        "correlation": float(row["correlation"]),
        "p_value": float(row.get("p_value", 1.0)),
        "sample_size": int(row.get("sample_size", 0)),
    }


def evaluate_player(player_name: str) -> Dict[str, Any]:
    # Search across supported position files and infer position key
    for pos, path in POSITION_FILES.items():
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue

        if 'player' in df.columns:
            candidate = df[df['player'].astype(str).str.lower() == player_name.lower()]
        elif 'Player' in df.columns:
            candidate = df[df['Player'].astype(str).str.lower() == player_name.lower()]
        else:
            continue

        if candidate.empty:
            continue

        candidate = candidate.sort_values(by=['Year'] if 'Year' in df.columns else candidate.columns[0])
        history = candidate.copy()

        wrapper_config = MODEL_REGISTRY.get(pos)

        if wrapper_config is not None:
            transformer = wrapper_config["transformer"]
            scaler = wrapper_config["scaler"]
            xgb = wrapper_config.get("xgb")
            if os.path.exists(transformer) and os.path.exists(scaler):
                model_cls = wrapper_config["class"]
                try:
                    model = model_cls(transformer, scaler, xgb)
                    tier, details = model.predict(history, mode="ensemble")
                    return {
                        "player_name": player_name,
                        "position": pos,
                        "tier": tier,
                        "predicted_grade": details.get("predicted_grade"),
                        "confidence": details,
                        "history_years": list(history['Year'].astype(int).tolist()) if 'Year' in history.columns else [],
                        "last_year": int(history['Year'].max()) if 'Year' in history.columns else None,
                    }
                except Exception as e:
                    # fallback to non-model heuristic
                    pass

        # fallback for CB/S/LB or missing model files
        if pos in {"cb", "s", "lb"}:
            if 'grades_defense' in history.columns or 'grades_offense' in history.columns:
                grade_col = 'grades_defense' if 'grades_defense' in history.columns else 'grades_offense'
                predicted = float(history[grade_col].iloc[-1])
                if pd.isna(predicted):
                    predicted = float(history[grade_col].mean()) if not history[grade_col].dropna().empty else 55.0
            else:
                predicted = 55.0

            if predicted >= 80:
                tier = 'Elite'
            elif predicted >= 65:
                tier = 'Starter'
            elif predicted >= 55:
                tier = 'Rotation'
            else:
                tier = 'Reserve/Poor'

            return {
                "player_name": player_name,
                "position": pos,
                "tier": tier,
                "predicted_grade": round(predicted, 2),
                "confidence": {
                    "predicted_grade": round(predicted, 2),
                    "source": "fallback historical grades"
                },
                "history_years": list(history['Year'].astype(int).tolist()) if 'Year' in history.columns else [],
                "last_year": int(history['Year'].max()) if 'Year' in history.columns else None,
            }

        # If we get this far and no prediction path is present, continue looping

    return {"player_name": player_name, "error": "Player not found in dataset"}

    return {"player_name": player_name, "error": "Player not found in dataset"}


def simulate_team_impact(move: Dict[str, Any]) -> Dict[str, Any]:
    # move: {team, position, player_grade, candidate_name, season}
    required = ["team", "position", "player_grade"]
    for field in required:
        if field not in move:
            return {"error": f"Missing required field: {field}"}

    position = move["position"].lower()
    team = move["team"]
    player_grade = float(move["player_grade"])
    season = move.get("season")

    team_context = get_team_context(team, season)
    if "error" in team_context:
        return {"error": "Cannot simulate team impact - team context unavailable"}

    team_pos_grade = team_context.get("positional_grades", {}).get(position)
    if team_pos_grade is None:
        team_pos_grade = 60.0

    # positional value lever from correlation audit
    pos_val = positional_value_lookup(position).get("importance_score", 0.0)

    delta_grade = player_grade - team_pos_grade
    # convert grade delta into expected win% delta (heuristic)
    # 1 point grade equals ~=0.010 win_pct units when high-correlation
    baseline = (delta_grade / 100.0) * 0.12
    impact = baseline * max(0.05, pos_val)

    return {
        "team": team,
        "position": position,
        "team_pos_grade": team_pos_grade,
        "player_grade": player_grade,
        "positional_value": pos_val,
        "grade_delta": delta_grade,
        "estimated_win_pct_delta": round(impact, 4),
        "estimated_season_wins_delta": round(impact * 17, 3),
    }
