import matplotlib.pyplot as plt

r_square_val_dict = {
    "QB": 0.25,

    "RB": 0.83,

    "WR": 0.02,

    "TE": 0.83,

    "OL": 0.79,

    "Defense": 0.4625
}

positions = list(r_square_val_dict.keys())
r_square_values = list(r_square_val_dict.values())
# Define colors for each position group
colors = {
    "QB": "#5DADE2",       # Light Blue
    "RB": "#76D7C4",       # Mint Green
    "WR": "#DDFF03",       # Neon Green
    "TE": "#E74C3C",       # Neon Red
    "OL": "#AF7AC5",       # Purple
    "Defense": "#58D68D"   # Neon
}
bar_colors = [colors[pos] for pos in positions]  # Map colors to positions

# Create the bar plot
plt.figure(figsize=(16, 10))
bars = plt.bar(positions, r_square_values, color=bar_colors, edgecolor='black')

# Add labels and title
plt.xlabel('Position Groups', fontsize=12)
plt.ylabel('R² Value', fontsize=12)
plt.title('Player Performance R² Model Values by Position Group Using Multilayer Perceptron (2022)', fontsize=14)
plt.ylim(0, 1.1)  # Set y-axis limit to show the full range of R² values

plt.grid(axis='y', linestyle='--', alpha=0.7)
for bar, pos in zip(bars, positions):
    bar.set_label(f"{pos}")
plt.legend(title="Position Colors", fontsize=10)
plt.savefig("player_r_square_by_position_mlp.png")
# Display the plot
plt.show()

