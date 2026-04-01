from gm_agent import GMReActAgent

agent = GMReActAgent()

result = agent.run(
    team="SEA",
    candidates=[
        ("Josh Jacobs", "rb"),
        ("Trent Williams", "t"),
        ("Mike Evans", "wr"),
        ("Maxx Crosby", "edge"),
        ("Derwin James", "s"),
    ],
    season=2022,
)

print("\n===== GM DECISION =====")
for k, v in result.items():
    print(f"{k}: {v}")