"""Quick script to check cluster distribution."""
import pandas as pd

df = pd.read_csv('backend/ML/scheme/data/2025_schemes.csv')
print('Cluster distribution in 2025:')
print(df['scheme_cluster'].value_counts().sort_index())
print('\nTeams in each cluster:')
for cluster in sorted(df['scheme_cluster'].unique()):
    teams = df[df['scheme_cluster'] == cluster]['team_abbr'].tolist()
    print(f'Cluster {cluster}: {len(teams)} teams - {", ".join(teams)}')

print('\nMissing clusters:')
all_clusters = set(range(4))
assigned_clusters = set(df['scheme_cluster'].unique())
missing = all_clusters - assigned_clusters
if missing:
    print(f'Cluster(s) {missing} have no teams assigned (empty clusters)')
else:
    print('All clusters have teams assigned')
