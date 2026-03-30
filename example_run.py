from gm_agent import GMReActAgent
import pandas as pd

# Example using the GMReActAgent in this repository
if __name__ == "__main__":
    agent = GMReActAgent()

    # Select some candidates from existing dataset for demonstration.
    # This is not a full market list. Picking known QB / RB names from CSV check.
    g_df = pd.read_csv("backend/ML/ED.csv")
    g_df = g_df[(g_df["Year"] < 2023) & (g_df["Year"] > 2020)]    
    sample_qbs = g_df["player"].dropna().unique()[:4].tolist()

    candidates = sample_qbs

    print("Candidate pool:", candidates)

    decision = agent.run("SEA", candidates, season=2023)

    print("--- ReAct GM Decision ---")
    print("Thought:", decision.get("Thought"))
    print("Action:", decision.get("Action"))
    print("Observation: (sample)", decision.get("Observation")[:2] if isinstance(decision.get("Observation"), list) else decision.get("Observation"))
    print("Final Decision:", decision.get("Final Decision"))
    print("Expected Win Impact:", decision.get("Expected Win Impact"))
    print("Football Explanation:", decision.get("Football Explanation"))
