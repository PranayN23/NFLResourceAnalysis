from gm_agent import GMReActAgent
import pandas as pd

# Example using the GMReActAgent in this repository
if __name__ == "__main__":
    agent = GMReActAgent()
    team = "SEA"
    season = 2023
    
    # Test each position group separately to understand agent behavior
    position_groups = {
        "QB": ("backend/ML/QB.csv", 4),
        "RB": ("backend/ML/HB.csv", 4),
        "WR": ("backend/ML/WR.csv", 4),
        "TE": ("backend/ML/TightEnds/TE.csv", 3),
        "T": ("backend/ML/T.csv", 3),          # Tackles
        "G": ("backend/ML/G.csv", 3),          # Guards  
        "C": ("backend/ML/C.csv", 3),          # Centers
        "LB": ("backend/ML/LB.csv", 4),
        "CB": ("backend/ML/CB.csv", 3),
        "S": ("backend/ML/S.csv", 3),
        "DI": ("backend/ML/DI.csv", 3),
        "ED": ("backend/ML/ED.csv", 3),
    }
    
    for position_name, (csv_file, sample_size) in position_groups.items():
        print(f"\n{'='*70}")
        print(f"TESTING {position_name} POSITION")
        print(f"{'='*70}")
        
        try:
            df = pd.read_csv(csv_file)
            df = df[(df["Year"] < 2023) & (df["Year"] > 2020)]
            sample_players = df["player"].dropna().unique()[:sample_size].tolist()
            
            if not sample_players:
                print(f"⚠️  No players found in {csv_file}")
                continue
                
            print(f"Candidates: {sample_players}")
            
            # For OL positions, use position hint; others can auto-detect
            if position_name in ["T", "G", "C"]:
                candidates = [(p, position_name.lower()) for p in sample_players]
            else:
                candidates = [(p, position_name.lower()) for p in sample_players]
            
            decision = agent.run(team, candidates, season=season)
            
            print(f"\nThought: {decision.get('Thought')}")
            print(f"Observation (first 2 candidates):")
            obs = decision.get("Observation", [])
            for o in obs[:2]:
                if "error" in o:
                    print(f"  - {o['player']}: ERROR - {o['error']}")
                else:
                    print(f"  - {o['player']}: Pos={o.get('position')}, Grade={o.get('prediction', {}).get('predicted_grade')}")
            
            print(f"\nFinal Decision: {decision.get('Final Decision')}")
            print(f"Expected Win Impact: {decision.get('Expected Win Impact')} games")
            print(f"Football Explanation: {decision.get('Football Explanation')}")
            
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()


