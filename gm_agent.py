import math
import pandas as pd
from typing import List, Dict, Any, Optional

from tools import get_team_context, evaluate_player, positional_value_lookup, simulate_team_impact
from agent_memory import AgentMemory


class GMReActAgent:
    def __init__(self):
        self.memory = AgentMemory()

    def _find_weakest_positions(self, team_context: Dict[str, Any], n: int = 3) -> List[str]:
        grade_map = team_context.get("positional_grades", {})
        if not grade_map:
            return []

        sorted_positions = sorted(grade_map.items(), key=lambda x: float(x[1]) if x[1] is not None else 100.0)
        return [pos for pos, _ in sorted_positions[:n]]

    def _normalize_position(self, pos: str) -> str:
        """Normalize position strings. Maps t/g/c to 'ol' (offensive line)."""
        pos = pos.strip().lower()
        if pos in ["t", "g", "c"]:
            pos = "ol"
        return pos

    def run(self, team: str, candidates: List[str], season: Optional[int] = None) -> Dict[str, Any]:
        # 1) Thought: inspect team context
        thought_1 = f"Need to assess {team} positional grade strengths and weaknesses before roster moves."

        action_1 = {"tool": "get_team_context", "team": team, "season": season}
        observation_1 = get_team_context(team, season)

        if "error" in observation_1:
            final = {
                "Thought": thought_1,
                "Action": action_1,
                "Observation": observation_1,
                "Final Decision": "PASS",
                "Expected Win Impact": 0.0,
                "Football Explanation": "Team context unavailable; cannot make data-driven roster move."
            }
            self.memory.record_decision(final)
            return final

        top_weak = self._find_weakest_positions(observation_1, 3)
        self.memory.store_insight("weakest_positions", top_weak)

        # 2) Thought: figure out correlation-weighted needs
        thought_2 = (
            "Weakness identified in positions: " + ", ".join(top_weak) + ". "
            "Now get positional value from correlation audit and candidate impact."
        )

        action_2 = {"tool": "positional_value_lookup", "positions": top_weak}
        pos_values = {p: positional_value_lookup(p) for p in top_weak}
        observation_2 = pos_values

        # Evaluate candidates
        candidate_evaluations = []
        for candidate in candidates:
            # Support both plain names and (name, position) tuples
            if isinstance(candidate, tuple):
                player_name, position_hint = candidate
            else:
                player_name = candidate
                position_hint = None
            
            action_e = {"tool": "evaluate_player", "player_name": player_name, "position": position_hint}
            observation_e = evaluate_player(player_name, position_hint)
            self.memory.store_player_evaluation(player_name, observation_e)

            # Filter: Skip if error in evaluation
            if "error" in observation_e:
                candidate_evaluations.append({
                    "player": player_name,
                    "error": observation_e.get("error"),
                })
                continue

            # Filter: Skip if predicted_grade is NaN
            predicted_grade = observation_e.get("predicted_grade")
            if predicted_grade is None or (isinstance(predicted_grade, float) and pd.isna(predicted_grade)):
                candidate_evaluations.append({
                    "player": player_name,
                    "error": "Predicted grade is NaN",
                })
                continue

            position = self._normalize_position(observation_e.get("position", "unknown"))
            pos_score = positional_value_lookup(position).get("importance_score", 0.0)

            move_out = simulate_team_impact({
                "team": team,
                "position": position,
                "player_grade": predicted_grade,
                "season": season,
            })
            
            # Filter: Skip if simulate_team_impact returned error (e.g., NaN grade slipped through)
            if "error" in move_out:
                candidate_evaluations.append({
                    "player": player_name,
                    "error": f"Simulation error: {move_out['error']}",
                })
                continue

            candidate_evaluations.append({
                "player": player_name,
                "position": position,
                "prediction": observation_e,
                "positional_value": pos_score,
                "simulated_impact": move_out,
            })

        # Choose best candidate: sort by estimated_season_wins_delta descending, pick first positive
        best = None
        best_impact = -math.inf
        valid_candidates = [c for c in candidate_evaluations if "simulated_impact" in c]
        
        if valid_candidates:
            # Sort by impact descending
            sorted_candidates = sorted(
                valid_candidates,
                key=lambda c: c["simulated_impact"].get("estimated_season_wins_delta", -math.inf),
                reverse=True
            )
            
            # Pick first one with positive impact
            for candidate in sorted_candidates:
                impact = candidate["simulated_impact"].get("estimated_season_wins_delta", -math.inf)
                if impact > 0:
                    best = candidate
                    best_impact = impact
                    break

        if best is None or best_impact <= 0.0:
            final_decision = "PASS"
            estimated = best_impact if best_impact != -math.inf else 0.0
            football_explanation = (
                "No available candidate improves weak positions with positive expected win impact."
            )
        else:
            final_decision = f"SIGN {best['player']} ({best.get('position', 'unknown')})"
            estimated = round(best_impact, 3)
            football_explanation = (
                f"Selected {best['player']} to address {best.get('position')} deficiency. "
                + f"Estimated net season win impact: {estimated:.2f} games."
            )

        final = {
            "Thought": thought_2,
            "Action": {
                "tool": "evaluate_and_simulate",
                "candidates": candidates,
                "weak_positions": top_weak,
            },
            "Observation": candidate_evaluations,
            "Final Decision": final_decision,
            "Expected Win Impact": estimated,
            "Football Explanation": football_explanation,
        }

        self.memory.record_decision(final)
        return final
