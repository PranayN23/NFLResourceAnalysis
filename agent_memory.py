from typing import Any, Dict


class AgentMemory:
    """Simple in-memory state storage for GM decisions."""
    def __init__(self):
        self.memory = {
            "history": [],
            "team_insights": {},
            "player_evaluations": {},
            "decisions": []
        }

    def store_insight(self, key: str, value: Any):
        self.memory["team_insights"][key] = value

    def store_player_evaluation(self, player_name: str, evaluation: Dict[str, Any]):
        self.memory["player_evaluations"][player_name] = evaluation

    def record_decision(self, decision: Dict[str, Any]):
        self.memory["decisions"].append(decision)

    def get_state(self) -> Dict[str, Any]:
        return self.memory

    def clear(self):
        self.memory = {
            "history": [],
            "team_insights": {},
            "player_evaluations": {},
            "decisions": []
        }
