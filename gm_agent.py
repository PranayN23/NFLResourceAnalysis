import json
from typing import Dict, Any, List, Optional

from tools import (
    get_team_context,
    evaluate_player,
    positional_value_lookup,
)

from agent_memory import AgentMemory
from backend.agent.llm_client import LLMClient


class GMReActAgent:
    """
    LLM-Driven NFL GM using TRUE ReAct reasoning.

    Thought -> Action -> Observation loop
    """

    def __init__(self, max_steps: int = 8):
        self.memory = AgentMemory()
        self.llm = LLMClient()
        self.max_steps = max_steps

        self.tools = {
            "get_team_context": get_team_context,
            "evaluate_player": evaluate_player,
            "positional_value_lookup": positional_value_lookup,
        }

    # --------------------------------------------------
    # Prompt Builder
    # --------------------------------------------------

    def _build_prompt(
        self,
        team: str,
        candidates: List[str],
        scratchpad: str,
        season: Optional[int],
    ) -> str:

        return f"""
You are an NFL General Manager AI.

Your job:
Make roster decisions using analytics + football reasoning.

AVAILABLE TOOLS:

1. get_team_context(team, season)
   -> team positional grades

2. positional_value_lookup(position)
   -> correlation importance with winning

3. evaluate_player(player_name, position?)
   -> predicted player performance grade

RULES:
- Think step-by-step.
- Call tools before deciding.
- Improve WEAK and IMPORTANT positions.
- Respond ONLY valid JSON.

FORMAT:

{{
  "Thought": "...",
  "Action": {{
      "tool": "<tool name>" OR "finish",
      "args": {{}}
  }}
}}

FINAL FORMAT:

{{
  "Thought": "...",
  "Action": {{ "tool": "finish" }},
  "Decision": "SIGN <player>" or "PASS",
  "Football Explanation": "Explain reasoning"
}}

TEAM: {team}
SEASON: {season}
CANDIDATES: {candidates}

Previous reasoning:
{scratchpad}
"""

    # --------------------------------------------------
    # Tool Executor
    # --------------------------------------------------

    def _execute(self, action: Dict[str, Any]):

        tool = action.get("tool")
        args = action.get("args", {})

        if tool == "finish":
            return None

        if tool not in self.tools:
            return {"error": f"Unknown tool {tool}"}

        try:
            return self.tools[tool](**args)
        except Exception as e:
            return {"error": str(e)}

    # --------------------------------------------------
    # MAIN REACT LOOP
    # --------------------------------------------------

    def run(
        self,
        team: str,
        candidates: List[str],
        season: Optional[int] = None,
    ) -> Dict[str, Any]:

        scratchpad = ""
        final_answer = None

        for step in range(self.max_steps):

            prompt = self._build_prompt(
                team,
                candidates,
                scratchpad,
                season,
            )

            llm_output = self.llm.generate(prompt)
            print("LLM Output:", llm_output)
            try:
                response = json.loads(llm_output)
            except Exception:
                scratchpad += f"\nInvalid JSON from LLM:\n{llm_output}\n"
                continue

            thought = response.get("Thought", "")
            action = response.get("Action", {})

            scratchpad += f"\nThought {step+1}: {thought}\n"

            # ---------- FINISH ----------
            if action.get("tool") == "finish":

                final_answer = {
                    "Decision": response.get("Decision", "PASS"),
                    "Football Explanation": response.get(
                        "Football Explanation",
                        "No explanation provided."
                    ),
                    "Reasoning Trace": scratchpad,
                }

                self.memory.record_decision(final_answer)
                return final_answer

            # ---------- TOOL CALL ----------
            observation = self._execute(action)

            scratchpad += (
                f"Action: {action}\n"
                f"Observation: {observation}\n"
            )

        # fallback
        return {
            "Decision": "PASS",
            "Football Explanation": "Agent exceeded reasoning limit.",
            "Reasoning Trace": scratchpad,
        }