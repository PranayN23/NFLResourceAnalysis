
import pandas as pd
import numpy as np

df = pd.read_csv("backend/ML/WideReceivers/WR_2024_Detailed_Report.csv")

actual = df["weighted_grade"]
pred = df["Ensemble_Pred"]

std_act = actual.std()
std_pred = pred.std()
ratio = std_pred / std_act

print(f"Actual Std: {std_act:.4f}")
print(f"Pred Std:   {std_pred:.4f}")
print(f"Ratio:      {ratio:.4f}")
print(f"Recipr:     {1/ratio:.4f}")

# Check specific players
players = ["Ja'Marr Chase", "Zay Flowers", "Justin Jefferson", "A.J. Brown"]
print("\nTarget Players:")
print(df[df["player"].isin(players)][["player", "weighted_grade", "Ensemble_Pred", "Error"]])
