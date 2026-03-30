import math
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
        return pos.strip().lower()

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
            action_e = {"tool": "evaluate_player", "player_name": candidate}
            observation_e = evaluate_player(candidate)
            self.memory.store_player_evaluation(candidate, observation_e)

            if "error" in observation_e:
                candidate_evaluations.append({
                    "player": candidate,
                    "error": observation_e.get("error"),
                })
                continue

            position = self._normalize_position(observation_e.get("position", "unknown"))
            pos_score = positional_value_lookup(position).get("importance_score", 0.0)

            move_out = simulate_team_impact({
                "team": team,
                "position": position,
                "player_grade": observation_e.get("predicted_grade", 0.0),
                "season": season,
            })

            candidate_evaluations.append({
                "player": candidate,
                "position": position,
                "prediction": observation_e,
                "positional_value": pos_score,
                "simulated_impact": move_out,
            })

        # Choose best candidate by estimated_win_pct_delta (positive and high)
        best = None
        best_impact = -math.inf
        for c in candidate_evaluations:
            sim = c.get("simulated_impact")
            if not sim or "estimated_win_pct_delta" not in sim:
                continue
            impact = sim["estimated_win_pct_delta"]
            if impact is not None and impact > best_impact:
                best_impact = impact
                best = c

        if best is None or best_impact <= 0.0:
            final_decision = "PASS"
            estimated = best_impact if best_impact != -math.inf else 0.0
            football_explanation = (
                "No available candidate improves weak positions with positive expected win impact."
            )
        else:
            final_decision = f"SIGN {best['player']} ({best.get('position', 'unknown')})"
            estimated = round(best_impact * 17, 3)
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
