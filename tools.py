import os
import logging
import pandas as pd
import numpy as np
import joblib
from typing import Optional, Dict, Any, List

from backend.agent.exceptions import UngradablePlayerError

from backend.ML.team_model.correlation_tool import get_team_context as _get_team_context
from backend.ML.team_model.correlation_tool import compute_correlation

# Setup logging for debugging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

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
    "te": os.path.abspath("backend/ML/TightEnds/TE.csv"),
    "t": os.path.abspath("backend/ML/T.csv"),          # Tackles
    "g": os.path.abspath("backend/ML/G.csv"),          # Guards
    "c": os.path.abspath("backend/ML/C.csv"),          # Centers
    "lb": os.path.abspath("backend/ML/LB.csv"),
    "cb": os.path.abspath("backend/ML/CB.csv"),
    "s": os.path.abspath("backend/ML/S.csv"),
    "di": os.path.abspath("backend/ML/DI.csv"),
    "ed": os.path.abspath("backend/ML/ED.csv"),
    # OL is an alias that expands to T, G, C (searched in order)
}

MODEL_REGISTRY = {
    "qb": {
        "class": PlayerModelInference,
        "transformer": os.path.abspath("backend/ML/QB_Pranay_Transformers/best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/QB_Pranay_Transformers/player_scaler.joblib"),
        "xgb": os.path.abspath("backend/ML/QB_Pranay_Transformers/best_xgb.joblib"),
    },
    "rb": {
        "class": RBModelInference,
        "transformer": os.path.abspath("backend/ML/RB_Pranay_Transformers/rb_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/RB_Pranay_Transformers/rb_player_scaler.joblib"),
        "xgb": os.path.abspath("backend/ML/RB_Pranay_Transformers/rb_best_xgb.joblib"),
    },
    "wr": {
        "class": WRModelInference,
        "transformer": os.path.abspath("backend/ML/WR_Pranay_Transformers/wr_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/WR_Pranay_Transformers/wr_player_scaler.joblib"),
        "xgb": None, # WR uses Transformer only (classification)
    },
    "te": {
    "class": TEModelInference,
    "transformer": os.path.abspath("backend/ML/TE_Pranay_Transformers/te_best_transformer.pth"),
    "scaler": os.path.abspath("backend/ML/TE_Pranay_Transformers/te_player_scaler.joblib"),
    "xgb": os.path.abspath("backend/ML/TE_Pranay_Transformers/te_best_xgb.joblib"),
},
    "ol": {
        "class": OLModelInference,
        "transformer": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_player_scaler.joblib"),
        "xgb": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_best_xgb.joblib"),
    },
    # T, G, C all use the same OL model (unified offensive line model)
    "t": {
        "class": OLModelInference,
        "transformer": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_player_scaler.joblib"),
        "xgb": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_best_xgb.joblib"),
    },
    "g": {
        "class": OLModelInference,
        "transformer": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_player_scaler.joblib"),
        "xgb": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_best_xgb.joblib"),
    },
    "c": {
        "class": OLModelInference,
        "transformer": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_player_scaler.joblib"),
        "xgb": os.path.abspath("backend/ML/OL_Pranay_Transformers/ol_best_xgb.joblib"),
    },
    "di": {
        "class": DIModelInference,
        "transformer": os.path.abspath("backend/ML/DI_Pranay_Transformers/di_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/DI_Pranay_Transformers/di_player_scaler.joblib"),
        "xgb": os.path.abspath("backend/ML/DI_Pranay_Transformers/di_best_xgb.joblib"),
    },
    "ed": {
        "class": EDModelInference,
        "transformer": os.path.abspath("backend/ML/ED_Pranay_Transformers/ed_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/ED_Pranay_Transformers/ed_player_scaler.joblib"),
        "xgb": os.path.abspath("backend/ML/ED_Pranay_Transformers/ed_best_xgb.joblib"),
    },
    "lb": {
        "class": LBModelInference,
        "transformer": os.path.abspath("backend/ML/LB_Pranay_Transformers/lb_best_classifier.pth"),
        "scaler": os.path.abspath("backend/ML/LB_Pranay_Transformers/lb_player_scaler.joblib"),
        "xgb": os.path.abspath("backend/ML/LB_Pranay_Transformers/lb_best_xgb.joblib"),
    },
    "cb": {
        "class": CBModelInference,
        "transformer": os.path.abspath("backend/ML/CB_Transformers/cb_best_transformer.pth"),
        "scaler": os.path.abspath("backend/ML/CB_Transformers/cb_player_scaler.joblib"),
        "xgb": None,
    },
    "s": {
        "class": SModelInference,
        "transformer": os.path.abspath("backend/ML/S_Transformers/s_best_transformer.pth"),
        "scaler": os.path.abspath("backend/ML/S_Transformers/s_player_scaler.joblib"),
        "xgb": None,
    },
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
    # Normalize position: t/g/c all map to ol (offensive line)
    position = position.lower()
    if position in ["t", "g", "c"]:
        position = "ol"
    
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


def evaluate_player(player_name: str, position: Optional[str] = None) -> Dict[str, Any]:
    """
    Evaluate a player's predicted performance grade.
    
    Args:
        player_name: Player to look up
        position: Optional position (qb, rb, wr, te, ol/t/g/c, lb, cb, s, di, ed).
                 "ol" expands to search T.csv, G.csv, C.csv in order.
                 Other positions search specific file directly.
                 If not provided, searches all position files (slower, ambiguity risk).
    
    Returns:
        Dict with player_name, position, tier, predicted_grade, confidence, history_years, last_year
        or {\"player_name\": ..., \"error\": ...} if not found
    """
    
    # Determine which position files to search
    if position:
        position = position.strip().lower()
        
        # OL special case: expand to T, G, C
        if position == "ol":
            positions_to_search = [("t", POSITION_FILES["t"]), 
                                   ("g", POSITION_FILES["g"]), 
                                   ("c", POSITION_FILES["c"])]
        elif position in POSITION_FILES:
            positions_to_search = [(position, POSITION_FILES[position])]
        else:
            return {
                "player_name": player_name,
                "error": f"Unknown position '{position}'. Valid: {', '.join(set(POSITION_FILES.keys()) - {'t', 'g', 'c'})} (or 'ol' for tackles/guards/centers)"
            }
    else:
        # Fallback: search all positions in order
        # Put OL components (t, g, c) early to avoid misclassification
        # And LB/CB/S before ED/DI to disambiguate defensive players
        positions_to_search = [
            ("qb", POSITION_FILES["qb"]),
            ("rb", POSITION_FILES["rb"]),
            ("wr", POSITION_FILES["wr"]),
            ("te", POSITION_FILES["te"]),
            ("t", POSITION_FILES["t"]),
            ("g", POSITION_FILES["g"]),
            ("c", POSITION_FILES["c"]),
            ("lb", POSITION_FILES["lb"]),
            ("cb", POSITION_FILES["cb"]),
            ("s", POSITION_FILES["s"]),
            ("di", POSITION_FILES["di"]),
            ("ed", POSITION_FILES["ed"]),
        ]
    
    for pos, path in positions_to_search:
        if not os.path.exists(path):
            logger.warning(f"CSV file missing for position '{pos}': {path}")
            continue
        
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            logger.error(f"FileNotFoundError: Cannot load position data from {path}")
            continue
        except Exception as e:
            logger.error(f"Error reading {path}: {e}")
            continue
        
        # Find player (case-insensitive)
        if 'player' in df.columns:
            candidate = df[df['player'].astype(str).str.lower() == player_name.lower()]
        elif 'Player' in df.columns:
            candidate = df[df['Player'].astype(str).str.lower() == player_name.lower()]
        else:
            continue
        
        if candidate.empty:
            if position:  # Only if we were searching a specific position
                return {
                    "player_name": player_name,
                    "error": f"Player '{player_name}' not found in {position.upper()} dataset"
                }
            continue
        
        candidate = candidate.sort_values(by=['Year'] if 'Year' in df.columns else candidate.columns[0])
        history = candidate.copy()
        
        # Try model-based prediction
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
                    predicted_grade = details.get("predicted_grade")
                    
                    # Filter out NaN grades - user wants to exclude these players
                    if predicted_grade is None or (isinstance(predicted_grade, float) and pd.isna(predicted_grade)):
                        logger.warning(f"WARNING: Player {player_name} ({pos}) could not be graded (sparse data) — excluded from candidate pool")
                        continue # Skip this player entirely
                    
                    return {
                        "player_name": player_name,
                        "position": pos,
                        "tier": tier,
                        "predicted_grade": predicted_grade,
                        "confidence": details,
                        "history_years": list(history['Year'].astype(int).tolist()) if 'Year' in history.columns else [],
                        "last_year": int(history['Year'].max()) if 'Year' in history.columns else None,
                    }
                except UngradablePlayerError:
                    logger.warning(f"WARNING: Player {player_name} ({pos}) could not be graded (sparse data) — excluded from candidate pool")
                    continue # Skip this player entirely
                except Exception as e:
                    logger.warning(f"Model prediction failed for {player_name} ({pos}): {e}. Using fallback.")
        
        # Fallback to historical grades
        if 'grades_defense' in history.columns or 'grades_offense' in history.columns:
            grade_col = 'grades_defense' if 'grades_defense' in history.columns else 'grades_offense'
            predicted = float(history[grade_col].iloc[-1])
            if pd.isna(predicted):
                predicted = float(history[grade_col].mean()) if not history[grade_col].dropna().empty else 55.0
        else:
            predicted = 55.0
        
        # Filter out NaN fallback grades
        if pd.isna(predicted):
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
    
    return {"player_name": player_name, "error": "Player not found in dataset"}


def simulate_team_impact(move: Dict[str, Any]) -> Dict[str, Any]:
    # move: {team, position, player_grade, candidate_name, season}
    required = ["team", "position", "player_grade"]
    for field in required:
        if field not in move:
            return {"error": f"Missing required field: {field}"}

    position = move["position"].lower()
    team = move["team"]
    player_grade = move["player_grade"]
    
    # Filter out NaN grades BEFORE converting to float
    if player_grade is None or (isinstance(player_grade, float) and pd.isna(player_grade)):
        logger.warning(f"simulate_team_impact: NaN player_grade for team {team}/{position}, skipping")
        return {"error": "Cannot simulate impact with NaN player grade"}
    
    try:
        player_grade = float(player_grade)
    except (ValueError, TypeError):
        logger.error(f"simulate_team_impact: Invalid player_grade {player_grade}")
        return {"error": "Invalid player_grade value"}
    
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
    # --- SIGNING IMPACT LOGIC ---
    # Goal: Convert a player's grade advantage into estimated team wins improvement
    # 
    # Step 1: Normalize grade delta to ratio (0-1 scale)
    #   A 10-point advantage on 100-point scale = 0.10 (10% better)
    # 
    # Step 2: Apply baseline conversion (0.12)
    #   This assumes ~12% of grade delta converts to win probability improvement
    #   This is a conservative estimate; empirically 1 point ≈ 0.01 win pct roughly
    # 
    # Step 3: Weight by position importance
    #   pos_val (from correlation audit) = how much this position drives team wins
    #   Ranges ~0.05 (bench positions) to 1.0 (QB/premium defense)
    # 
    # Example: RB, 10-point advantage vs team average, pos_val=0.3
    #   baseline = (10 / 100) * 0.12 = 0.012 (1.2% win pct) 
    #   impact = 0.012 * 0.3 = 0.0036 (0.36% win pct ≈ 0.06 wins/season)
    
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
