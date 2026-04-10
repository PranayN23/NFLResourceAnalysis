# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:27:14Z
- **Requested analysis_year:** 2010 (clamped to 2010)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Myers | 95.09 | 90.00 | 94.32 | 1067 | Texans |
| 2 | 2 | Nick Mangold | 93.95 | 88.10 | 93.68 | 1227 | Jets |
| 3 | 3 | Scott Wells | 91.24 | 84.60 | 91.50 | 1282 | Packers |
| 4 | 4 | Jeff Saturday | 87.31 | 83.90 | 85.42 | 1176 | Colts |
| 5 | 5 | David Baas | 86.43 | 78.80 | 87.35 | 939 | 49ers |
| 6 | 6 | Kyle Cook | 86.07 | 78.50 | 86.95 | 1094 | Bengals |
| 7 | 7 | Ryan Kalil | 86.05 | 78.50 | 86.91 | 989 | Panthers |
| 8 | 8 | Matt Birk | 86.05 | 81.00 | 85.25 | 1186 | Ravens |
| 9 | 9 | Alex Mack | 84.83 | 76.20 | 86.41 | 958 | Browns |
| 10 | 10 | Brad Meester | 84.53 | 77.00 | 85.38 | 1064 | Jaguars |
| 11 | 11 | Andre Gurode | 83.81 | 74.00 | 86.19 | 1073 | Cowboys |
| 12 | 12 | Dan Koppen | 82.98 | 74.40 | 84.54 | 1080 | Patriots |
| 13 | 13 | Todd McClure | 82.85 | 74.90 | 83.99 | 1193 | Falcons |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Samson Satele | 79.23 | 69.50 | 81.55 | 979 | Raiders |
| 15 | 2 | Jonathan Goodwin | 78.26 | 68.60 | 80.53 | 1178 | Saints |
| 16 | 3 | John Sullivan | 77.62 | 68.30 | 79.67 | 792 | Vikings |
| 17 | 4 | Nick Hardwick | 77.51 | 67.80 | 79.81 | 1050 | Chargers |
| 18 | 5 | Maurkice Pouncey | 77.36 | 67.70 | 79.64 | 1073 | Steelers |
| 19 | 6 | Dominic Raiola | 76.28 | 64.70 | 79.83 | 1107 | Lions |
| 20 | 7 | J.D. Walton | 75.63 | 65.10 | 78.48 | 1061 | Broncos |
| 21 | 8 | Jeff Faine | 74.07 | 62.50 | 77.61 | 432 | Buccaneers |

### Starter (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Jeremy Zuttah | 73.69 | 65.50 | 74.99 | 721 | Buccaneers |
| 23 | 2 | Lyle Sendlein | 72.27 | 64.00 | 73.61 | 959 | Cardinals |
| 24 | 3 | Eugene Amano | 66.46 | 53.60 | 70.86 | 706 | Titans |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Champ Bailey | 91.39 | 85.80 | 91.98 | 966 | Broncos |
| 2 | 2 | Josh Wilson | 91.05 | 87.20 | 90.49 | 652 | Ravens |
| 3 | 3 | Joe Haden | 90.91 | 84.20 | 91.22 | 791 | Browns |
| 4 | 4 | Antoine Winfield | 88.64 | 88.90 | 84.30 | 981 | Vikings |
| 5 | 5 | Darrelle Revis | 87.71 | 81.30 | 87.82 | 1013 | Jets |
| 6 | 6 | Tramon Williams | 87.39 | 83.00 | 86.15 | 1232 | Packers |
| 7 | 7 | Brandon Flowers | 86.35 | 80.60 | 86.01 | 1036 | Chiefs |
| 8 | 8 | Asante Samuel | 85.69 | 81.40 | 88.55 | 729 | Eagles |
| 9 | 9 | Brent Grimes | 85.59 | 81.20 | 84.35 | 1035 | Falcons |
| 10 | 10 | Charles Tillman | 83.69 | 77.60 | 83.59 | 1165 | Bears |
| 11 | 11 | Nnamdi Asomugha | 83.37 | 78.50 | 84.53 | 767 | Raiders |
| 12 | 12 | Jason McCourty | 82.89 | 77.40 | 86.55 | 483 | Titans |
| 13 | 13 | Alterraun Verner | 82.32 | 78.30 | 80.84 | 997 | Titans |
| 14 | 14 | Joselio Hanson | 81.05 | 73.20 | 82.11 | 694 | Eagles |
| 15 | 15 | Brandon Carr | 80.92 | 73.00 | 82.03 | 1133 | Chiefs |
| 16 | 16 | Antoine Cason | 80.91 | 72.50 | 82.35 | 918 | Chargers |
| 17 | 17 | Aqib Talib | 80.63 | 74.70 | 85.62 | 635 | Buccaneers |
| 18 | 18 | Roy Lewis | 80.47 | 75.70 | 82.61 | 263 | Seahawks |
| 19 | 19 | Vontae Davis | 80.44 | 75.40 | 79.63 | 1001 | Dolphins |
| 20 | 20 | Chris Carr | 80.34 | 78.90 | 77.14 | 1131 | Ravens |
| 21 | 21 | Leon Hall | 80.20 | 75.00 | 79.50 | 966 | Bengals |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Captain Munnerlyn | 79.88 | 72.30 | 80.77 | 695 | Panthers |
| 23 | 2 | DeAngelo Hall | 79.83 | 72.10 | 80.82 | 1079 | Commanders |
| 24 | 3 | Jabari Greer | 79.56 | 74.70 | 79.66 | 812 | Saints |
| 25 | 4 | Sean Smith | 78.82 | 71.40 | 80.64 | 746 | Dolphins |
| 26 | 5 | Dunta Robinson | 78.34 | 75.10 | 76.34 | 899 | Falcons |
| 27 | 6 | Ronde Barber | 77.62 | 69.20 | 79.06 | 987 | Buccaneers |
| 28 | 7 | Nate Clements | 77.58 | 70.50 | 78.13 | 1056 | 49ers |
| 29 | 8 | Sam Shields | 77.15 | 71.20 | 76.95 | 803 | Packers |
| 30 | 9 | Kelvin Hayden | 77.14 | 76.20 | 78.80 | 666 | Colts |
| 31 | 10 | Syd'Quan Thompson | 77.13 | 72.20 | 81.45 | 211 | Broncos |
| 32 | 11 | Kelly Jennings | 76.81 | 66.50 | 79.52 | 1005 | Seahawks |
| 33 | 12 | Javier Arenas | 75.67 | 68.80 | 76.09 | 502 | Chiefs |
| 34 | 13 | Sheldon Brown | 74.99 | 65.90 | 76.88 | 890 | Browns |
| 35 | 14 | Terrence McGee | 74.47 | 72.70 | 78.78 | 313 | Bills |
| 36 | 15 | Johnathan Joseph | 74.46 | 70.20 | 77.30 | 595 | Bengals |
| 37 | 16 | Lardarius Webb | 74.25 | 67.00 | 74.91 | 559 | Ravens |
| 38 | 17 | Perrish Cox | 74.03 | 64.70 | 77.11 | 771 | Broncos |

### Starter (52 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Phillip Buchanon | 73.66 | 61.90 | 77.34 | 720 | Commanders |
| 40 | 2 | Kevin Barnes | 73.30 | 65.60 | 81.56 | 269 | Commanders |
| 41 | 3 | Leodis McKelvin | 73.14 | 64.00 | 75.06 | 851 | Bills |
| 42 | 4 | Reggie Corner | 73.05 | 65.30 | 74.05 | 462 | Bills |
| 43 | 5 | Corey Webster | 72.93 | 64.40 | 75.48 | 915 | Giants |
| 44 | 6 | Ronald Bartell | 72.91 | 68.40 | 72.78 | 852 | Rams |
| 45 | 7 | Tarell Brown | 72.85 | 68.60 | 74.65 | 310 | 49ers |
| 46 | 8 | Greg Toler | 72.68 | 66.00 | 75.05 | 887 | Cardinals |
| 47 | 9 | William Middleton | 72.48 | 68.40 | 71.04 | 447 | Jaguars |
| 48 | 10 | Shawntae Spencer | 72.27 | 65.20 | 72.82 | 1021 | 49ers |
| 49 | 11 | Tracy Porter | 72.25 | 65.30 | 75.85 | 727 | Saints |
| 50 | 12 | Bradley Fletcher | 72.09 | 60.80 | 75.45 | 978 | Rams |
| 51 | 13 | Drayton Florence | 71.91 | 60.10 | 75.61 | 1063 | Bills |
| 52 | 14 | Kyle Wilson | 71.73 | 62.30 | 74.89 | 335 | Jets |
| 53 | 15 | Tim Jennings | 71.66 | 64.70 | 72.14 | 948 | Bears |
| 54 | 16 | E.J. Biggers | 71.14 | 61.70 | 73.26 | 669 | Buccaneers |
| 55 | 17 | D.J. Moore | 70.89 | 63.00 | 71.98 | 607 | Bears |
| 56 | 18 | Adam Jones | 70.86 | 72.60 | 81.50 | 165 | Bengals |
| 57 | 19 | Orlando Scandrick | 70.24 | 61.00 | 72.24 | 597 | Cowboys |
| 58 | 20 | Kyle Arrington | 70.12 | 60.70 | 72.23 | 899 | Patriots |
| 59 | 21 | Chris Johnson | 69.86 | 60.60 | 76.04 | 376 | Raiders |
| 60 | 22 | Terence Newman | 69.83 | 60.60 | 71.82 | 877 | Cowboys |
| 61 | 23 | Nathan Vasher | 69.83 | 67.00 | 70.68 | 338 | Lions |
| 62 | 24 | Morgan Trent | 69.58 | 69.50 | 76.34 | 202 | Bengals |
| 63 | 25 | Patrick Robinson | 69.44 | 67.20 | 74.06 | 283 | Saints |
| 64 | 26 | Brian Williams | 69.17 | 64.00 | 69.48 | 386 | Falcons |
| 65 | 27 | Antonio Cromartie | 69.06 | 57.20 | 72.80 | 1087 | Jets |
| 66 | 28 | Stanford Routt | 68.96 | 57.00 | 72.76 | 980 | Raiders |
| 67 | 29 | Andre' Goodman | 68.74 | 62.70 | 76.93 | 367 | Broncos |
| 68 | 30 | Chris Gamble | 68.71 | 64.20 | 72.75 | 710 | Panthers |
| 69 | 31 | Terrell Thomas | 68.70 | 56.70 | 72.54 | 986 | Giants |
| 70 | 32 | Jason Allen | 68.64 | 58.50 | 73.31 | 608 | Texans |
| 71 | 33 | Carlos Rogers | 68.63 | 62.50 | 72.72 | 760 | Commanders |
| 72 | 34 | Jonathan Wilhite | 68.20 | 63.60 | 74.40 | 202 | Patriots |
| 73 | 35 | Jeremy Ware | 67.91 | 65.40 | 78.84 | 114 | Raiders |
| 74 | 36 | Cortland Finnegan | 67.89 | 58.60 | 69.91 | 1173 | Titans |
| 75 | 37 | Marcus Trufant | 67.64 | 56.50 | 70.90 | 1049 | Seahawks |
| 76 | 38 | Jarrett Bush | 67.52 | 56.80 | 74.66 | 176 | Packers |
| 77 | 39 | Cedric Griffin | 66.96 | 75.60 | 78.54 | 132 | Vikings |
| 78 | 40 | Bryan McCann | 66.70 | 64.60 | 71.23 | 146 | Cowboys |
| 79 | 41 | Trevard Lindley | 66.59 | 67.60 | 70.09 | 184 | Eagles |
| 80 | 42 | Rashean Mathis | 66.31 | 54.40 | 70.09 | 964 | Jaguars |
| 81 | 43 | Dante Hughes | 65.88 | 67.70 | 68.84 | 200 | Chargers |
| 82 | 44 | William Gay | 65.34 | 54.80 | 68.20 | 726 | Steelers |
| 83 | 45 | Bryant McFadden | 64.71 | 54.00 | 67.69 | 961 | Steelers |
| 84 | 46 | Dominique Rodgers-Cromartie | 64.55 | 51.50 | 69.08 | 1103 | Cardinals |
| 85 | 47 | Aaron Ross | 64.39 | 55.00 | 67.52 | 398 | Giants |
| 86 | 48 | Alphonso Smith | 63.55 | 54.40 | 69.65 | 573 | Lions |
| 87 | 49 | Richard Marshall | 62.97 | 53.10 | 65.39 | 1086 | Panthers |
| 88 | 50 | Drew Coleman | 62.96 | 55.50 | 63.76 | 576 | Jets |
| 89 | 51 | Justin King | 62.95 | 64.60 | 66.02 | 186 | Rams |
| 90 | 52 | Kareem Jackson | 62.58 | 48.90 | 67.53 | 874 | Texans |

### Rotation/backup (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 91 | 1 | Derek Cox | 61.82 | 48.40 | 69.73 | 672 | Jaguars |
| 92 | 2 | Mike Jenkins | 61.53 | 53.70 | 62.58 | 914 | Cowboys |
| 93 | 3 | Jerome Murphy | 61.15 | 52.30 | 70.19 | 177 | Rams |
| 94 | 4 | Asher Allen | 60.24 | 51.50 | 65.04 | 690 | Vikings |
| 95 | 5 | Jerraud Powers | 59.48 | 51.50 | 66.88 | 586 | Colts |
| 96 | 6 | Jacob Lacey | 59.42 | 52.20 | 63.20 | 622 | Colts |
| 97 | 7 | David Jones | 59.34 | 49.80 | 69.86 | 288 | Jaguars |
| 98 | 8 | Fabian Washington | 58.48 | 56.00 | 60.13 | 511 | Ravens |
| 99 | 9 | Dimitri Patterson | 57.27 | 38.50 | 66.65 | 636 | Eagles |
| 100 | 10 | Myron Lewis | 57.09 | 45.90 | 67.68 | 196 | Buccaneers |
| 101 | 11 | Ellis Hobbs | 55.14 | 48.20 | 66.46 | 452 | Eagles |
| 102 | 12 | Zackary Bowman | 53.93 | 44.90 | 60.98 | 231 | Bears |
| 103 | 13 | Brice McCain | 51.02 | 48.20 | 53.93 | 390 | Texans |
| 104 | 14 | Eric Wright | 50.50 | 38.80 | 57.27 | 708 | Browns |
| 105 | 15 | Chris Owens | 49.41 | 31.50 | 63.44 | 307 | Falcons |
| 106 | 16 | Chris Cook | 45.00 | 35.40 | 54.22 | 229 | Vikings |

## DI — Defensive Interior

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Antonio Garay | 91.74 | 84.83 | 92.18 | 457 | Chargers |
| 2 | 2 | Justin Smith | 90.06 | 85.62 | 88.86 | 799 | 49ers |
| 3 | 3 | Kyle Williams | 89.46 | 86.97 | 86.96 | 897 | Bills |
| 4 | 4 | Albert Haynesworth | 86.32 | 87.12 | 89.96 | 204 | Commanders |
| 5 | 5 | Jason Jones | 86.00 | 80.85 | 86.30 | 653 | Titans |
| 6 | 6 | Haloti Ngata | 85.31 | 84.95 | 81.38 | 891 | Ravens |
| 7 | 7 | Sammie Lee Hill | 84.28 | 75.94 | 86.71 | 349 | Lions |
| 8 | 8 | Ricky Jean Francois | 83.41 | 84.22 | 78.71 | 144 | 49ers |
| 9 | 9 | Richard Seymour | 82.94 | 72.48 | 88.88 | 622 | Raiders |
| 10 | 10 | Geno Atkins | 82.93 | 81.64 | 79.62 | 349 | Bengals |
| 11 | 11 | Ndamukong Suh | 82.48 | 71.45 | 85.67 | 959 | Lions |
| 12 | 12 | Paul Soliai | 82.15 | 75.81 | 82.21 | 546 | Dolphins |
| 13 | 13 | Desmond Bryant | 80.84 | 72.83 | 83.04 | 325 | Raiders |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Antonio Dixon | 79.35 | 73.48 | 79.09 | 439 | Eagles |
| 15 | 2 | Letroy Guion | 78.45 | 74.24 | 78.12 | 262 | Vikings |
| 16 | 3 | Calais Campbell | 78.15 | 63.54 | 84.75 | 771 | Cardinals |
| 17 | 4 | Red Bryant | 77.11 | 83.85 | 79.31 | 283 | Seahawks |
| 18 | 5 | Dan Williams | 76.73 | 82.76 | 69.57 | 382 | Cardinals |
| 19 | 6 | B.J. Raji | 76.72 | 77.51 | 72.03 | 1069 | Packers |
| 20 | 7 | Barry Cofield | 76.63 | 76.55 | 72.52 | 768 | Giants |
| 21 | 8 | Kevin Williams | 76.03 | 75.72 | 72.07 | 895 | Vikings |
| 22 | 9 | Wallace Gilberry | 75.83 | 60.59 | 81.82 | 519 | Chiefs |
| 23 | 10 | Vince Wilfork | 75.67 | 67.07 | 77.24 | 818 | Patriots |
| 24 | 11 | John Henderson | 75.25 | 76.87 | 77.30 | 264 | Raiders |
| 25 | 12 | Aubrayo Franklin | 75.22 | 73.72 | 72.06 | 566 | 49ers |

### Starter (61 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Gerald McCoy | 73.80 | 82.22 | 67.15 | 669 | Buccaneers |
| 27 | 2 | Tony McDaniel | 73.77 | 63.27 | 77.63 | 492 | Dolphins |
| 28 | 3 | Glenn Dorsey | 73.54 | 66.31 | 74.19 | 989 | Chiefs |
| 29 | 4 | Sione Pouha | 73.45 | 68.87 | 72.33 | 668 | Jets |
| 30 | 5 | Peria Jerry | 73.38 | 58.81 | 78.92 | 211 | Falcons |
| 31 | 6 | Mike Devito | 73.28 | 64.83 | 74.74 | 684 | Jets |
| 32 | 7 | Shaun Ellis | 72.70 | 61.67 | 75.88 | 810 | Jets |
| 33 | 8 | Kendall Langford | 72.57 | 68.99 | 70.79 | 729 | Dolphins |
| 34 | 9 | Stephen Bowen | 72.50 | 65.24 | 73.17 | 542 | Cowboys |
| 35 | 10 | Cullen Jenkins | 72.49 | 59.68 | 77.90 | 566 | Packers |
| 36 | 11 | Shaun Rogers | 72.13 | 59.91 | 77.15 | 395 | Browns |
| 37 | 12 | Alan Branch | 72.06 | 61.89 | 74.67 | 564 | Cardinals |
| 38 | 13 | Vonnie Holliday | 71.66 | 57.56 | 77.93 | 410 | Commanders |
| 39 | 14 | Ryan Pickett | 71.45 | 56.82 | 77.03 | 470 | Packers |
| 40 | 15 | Tommy Kelly | 71.39 | 60.36 | 74.58 | 835 | Raiders |
| 41 | 16 | Adam Carriker | 71.36 | 58.57 | 75.72 | 591 | Commanders |
| 42 | 17 | Chris Canty | 71.03 | 71.34 | 66.66 | 611 | Giants |
| 43 | 18 | Shaun Smith | 70.99 | 57.69 | 75.69 | 480 | Chiefs |
| 44 | 19 | Brodrick Bunkley | 70.94 | 58.58 | 76.04 | 344 | Eagles |
| 45 | 20 | C.J. Wilson | 70.86 | 54.91 | 77.33 | 290 | Packers |
| 46 | 21 | Pat Sims | 70.78 | 57.25 | 77.71 | 434 | Bengals |
| 47 | 22 | Colin Cole | 70.78 | 60.58 | 76.55 | 525 | Seahawks |
| 48 | 23 | Andre Fluellen | 70.67 | 63.68 | 71.16 | 151 | Lions |
| 49 | 24 | Rocky Bernard | 69.68 | 58.57 | 75.01 | 321 | Giants |
| 50 | 25 | Brandon Mebane | 69.67 | 64.13 | 71.28 | 717 | Seahawks |
| 51 | 26 | Mike Patterson | 68.97 | 58.91 | 71.51 | 589 | Eagles |
| 52 | 27 | Marcus Thomas | 68.48 | 56.46 | 72.32 | 523 | Broncos |
| 53 | 28 | Cory Redding | 68.11 | 49.48 | 76.36 | 683 | Ravens |
| 54 | 29 | Earl Mitchell | 67.48 | 55.30 | 72.46 | 310 | Texans |
| 55 | 30 | Sedrick Ellis | 67.39 | 60.45 | 67.85 | 817 | Saints |
| 56 | 31 | Amobi Okoye | 67.08 | 55.73 | 70.48 | 785 | Texans |
| 57 | 32 | Darnell Dockett | 66.92 | 49.22 | 75.59 | 903 | Cardinals |
| 58 | 33 | Tyson Alualu | 66.72 | 57.64 | 68.60 | 789 | Jaguars |
| 59 | 34 | Corey Williams | 66.70 | 54.17 | 70.89 | 773 | Lions |
| 60 | 35 | Tyson Jackson | 66.60 | 60.75 | 69.47 | 330 | Chiefs |
| 61 | 36 | Shaun Cody | 66.52 | 52.70 | 71.57 | 485 | Texans |
| 62 | 37 | Randy Starks | 66.12 | 57.92 | 67.42 | 734 | Dolphins |
| 63 | 38 | Brandon Deaderick | 66.01 | 60.25 | 71.93 | 253 | Patriots |
| 64 | 39 | Terrance Knighton | 65.85 | 55.32 | 68.70 | 801 | Jaguars |
| 65 | 40 | Jonathan Babineaux | 65.61 | 54.70 | 68.72 | 799 | Falcons |
| 66 | 41 | Antonio Johnson | 65.60 | 53.50 | 71.58 | 308 | Colts |
| 67 | 42 | Tony Hargrove | 65.44 | 58.04 | 68.29 | 438 | Saints |
| 68 | 43 | Brett Keisel | 65.32 | 57.07 | 68.74 | 609 | Steelers |
| 69 | 44 | Vance Walker | 65.23 | 50.00 | 71.22 | 294 | Falcons |
| 70 | 45 | Andre Neblett | 65.00 | 69.96 | 68.40 | 151 | Panthers |
| 71 | 46 | Kyle Love | 64.89 | 62.46 | 68.59 | 166 | Patriots |
| 72 | 47 | Jason Hatcher | 64.79 | 55.35 | 70.05 | 257 | Cowboys |
| 73 | 48 | Derek Landri | 64.77 | 48.28 | 71.59 | 785 | Panthers |
| 74 | 49 | Antonio Smith | 64.01 | 52.21 | 67.71 | 902 | Texans |
| 75 | 50 | Ray McDonald | 63.92 | 57.44 | 64.08 | 569 | 49ers |
| 76 | 51 | Josh Price-Brent | 63.90 | 58.14 | 63.58 | 254 | Cowboys |
| 77 | 52 | Ron Brace | 63.76 | 54.91 | 68.63 | 283 | Patriots |
| 78 | 53 | Jacques Cesaire | 63.55 | 41.86 | 73.84 | 607 | Chargers |
| 79 | 54 | Domata Peko Sr. | 63.53 | 49.42 | 68.77 | 647 | Bengals |
| 80 | 55 | Ahtyba Rubin | 63.22 | 48.70 | 68.73 | 722 | Browns |
| 81 | 56 | Roy Miller | 63.19 | 45.88 | 70.57 | 600 | Buccaneers |
| 82 | 57 | Ron Edwards | 63.03 | 59.25 | 61.39 | 512 | Chiefs |
| 83 | 58 | Trevor Laws | 62.82 | 55.29 | 63.67 | 470 | Eagles |
| 84 | 59 | Tommie Harris | 62.24 | 57.75 | 61.07 | 617 | Bears |
| 85 | 60 | Jimmy Kennedy | 62.20 | 50.09 | 73.41 | 147 | Vikings |
| 86 | 61 | Corey Peters | 62.01 | 46.76 | 68.01 | 597 | Falcons |

### Rotation/backup (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 87 | 1 | Anthony Adams | 61.89 | 44.40 | 69.39 | 654 | Bears |
| 88 | 2 | Luis Castillo | 61.63 | 50.67 | 64.77 | 614 | Chargers |
| 89 | 3 | Alex Carrington | 61.60 | 64.48 | 62.82 | 214 | Bills |
| 90 | 4 | Jay Ratliff | 61.53 | 53.19 | 62.92 | 720 | Cowboys |
| 91 | 5 | Isaac Sopoaga | 61.51 | 45.46 | 68.04 | 569 | 49ers |
| 92 | 6 | Henry Melton | 61.42 | 44.92 | 68.25 | 390 | Bears |
| 93 | 7 | Al Woods | 61.32 | 48.11 | 73.26 | 130 | Buccaneers |
| 94 | 8 | Fili Moala | 61.23 | 44.41 | 68.28 | 527 | Colts |
| 95 | 9 | Marcus Stroud | 60.98 | 41.92 | 70.55 | 673 | Bills |
| 96 | 10 | Marcus Spears | 60.94 | 57.79 | 67.21 | 259 | Cowboys |
| 97 | 11 | Igor Olshansky | 59.53 | 52.51 | 60.05 | 569 | Cowboys |
| 98 | 12 | Jermelle Cudjo | 58.96 | 49.13 | 67.60 | 231 | Rams |
| 99 | 13 | Nick Hayden | 58.84 | 48.48 | 63.67 | 473 | Panthers |
| 100 | 14 | Terrence Cody | 58.79 | 51.07 | 60.81 | 141 | Ravens |
| 101 | 15 | Remi Ayodele | 58.67 | 47.06 | 62.24 | 627 | Saints |
| 102 | 16 | Daniel Muir | 58.43 | 42.20 | 66.11 | 526 | Colts |
| 103 | 17 | Sen'Derrick Marks | 58.25 | 48.91 | 64.47 | 460 | Titans |
| 104 | 18 | Brandon McKinney | 58.12 | 60.79 | 55.31 | 225 | Ravens |
| 105 | 19 | Vaughn Martin | 57.55 | 54.50 | 61.67 | 181 | Chargers |
| 106 | 20 | Pat Williams | 57.47 | 42.53 | 63.26 | 541 | Vikings |
| 107 | 21 | Jovan Haye | 57.16 | 44.28 | 63.66 | 470 | Titans |
| 108 | 22 | Matt Toeaina | 56.93 | 49.96 | 57.41 | 676 | Bears |
| 109 | 23 | Ronald Fields | 55.80 | 49.36 | 55.92 | 298 | Broncos |
| 110 | 24 | Gary Gibson | 55.73 | 47.48 | 57.07 | 589 | Rams |
| 111 | 25 | Leger Douzable | 55.58 | 50.32 | 55.95 | 217 | Jaguars |
| 112 | 26 | Ma'ake Kemoeatu | 55.55 | 40.95 | 63.20 | 362 | Commanders |
| 113 | 27 | Torell Troup | 55.15 | 49.77 | 55.61 | 298 | Bills |
| 114 | 28 | Casey Hampton | 55.07 | 49.01 | 54.94 | 503 | Steelers |
| 115 | 29 | Kedric Golston | 54.34 | 43.98 | 60.21 | 445 | Commanders |
| 116 | 30 | Tank Johnson | 54.22 | 55.30 | 60.20 | 223 | Bengals |
| 117 | 31 | Justin Bannan | 54.17 | 43.59 | 57.06 | 774 | Broncos |
| 118 | 32 | Bryan Robinson | 53.56 | 34.01 | 62.43 | 338 | Cardinals |
| 119 | 33 | Ogemdi Nwagbuo | 52.19 | 51.44 | 49.55 | 258 | Chargers |
| 120 | 34 | Darell Scott | 51.47 | 49.08 | 56.20 | 253 | Rams |
| 121 | 35 | Nick Eason | 51.16 | 38.54 | 55.40 | 529 | Steelers |
| 122 | 36 | Ryan McBean | 50.85 | 42.21 | 52.45 | 445 | Broncos |
| 123 | 37 | Myron Pryor | 50.41 | 44.18 | 57.70 | 233 | Patriots |
| 124 | 38 | Aaron Smith | 49.81 | 47.93 | 60.31 | 279 | Steelers |
| 125 | 39 | Gabe Watson | 49.11 | 45.16 | 58.44 | 142 | Cardinals |
| 126 | 40 | Craig Terrill | 47.07 | 42.21 | 48.23 | 380 | Seahawks |
| 127 | 41 | Brian Price | 45.24 | 49.18 | 54.41 | 110 | Buccaneers |
| 128 | 42 | Phillip Merling | 45.00 | 50.65 | 48.83 | 112 | Dolphins |
| 129 | 43 | Anthony Bryant | 45.00 | 50.78 | 49.86 | 165 | Commanders |

## ED — Edge

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | DeMarcus Ware | 92.00 | 90.14 | 89.07 | 926 | Cowboys |
| 2 | 2 | Charles Johnson | 91.45 | 87.67 | 89.80 | 884 | Panthers |
| 3 | 3 | Terrell Suggs | 87.97 | 87.43 | 84.16 | 1136 | Ravens |
| 4 | 4 | Cameron Wake | 86.23 | 76.83 | 88.33 | 907 | Dolphins |
| 5 | 5 | Jason Babin | 85.51 | 75.92 | 87.74 | 713 | Titans |
| 6 | 6 | Ray Edwards | 84.38 | 79.56 | 85.51 | 745 | Vikings |
| 7 | 7 | Mario Williams | 84.00 | 86.04 | 81.60 | 764 | Texans |
| 8 | 8 | Justin Tuck | 80.95 | 75.78 | 80.23 | 830 | Giants |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Lawrence Jackson | 79.45 | 77.39 | 81.85 | 327 | Lions |
| 10 | 2 | Cliff Avril | 79.31 | 72.20 | 83.01 | 625 | Lions |
| 11 | 3 | Julius Peppers | 78.42 | 85.88 | 69.28 | 1036 | Bears |
| 12 | 4 | Juqua Parker | 77.16 | 73.79 | 77.32 | 513 | Eagles |
| 13 | 5 | Jarvis Moss | 77.01 | 62.02 | 87.01 | 105 | Raiders |
| 14 | 6 | Matt Shaughnessy | 76.66 | 70.46 | 76.62 | 628 | Raiders |
| 15 | 7 | Manny Lawson | 76.04 | 68.14 | 77.14 | 620 | 49ers |
| 16 | 8 | Carlos Dunlap | 75.55 | 66.42 | 81.63 | 281 | Bengals |
| 17 | 9 | Chris Long | 75.41 | 71.87 | 73.61 | 962 | Rams |
| 18 | 10 | Jared Allen | 75.02 | 63.75 | 78.37 | 934 | Vikings |

### Starter (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Parys Haralson | 71.78 | 62.15 | 75.07 | 677 | 49ers |
| 20 | 2 | Shaun Phillips | 71.66 | 60.00 | 75.26 | 909 | Chargers |
| 21 | 3 | Quentin Moses | 71.60 | 63.71 | 74.77 | 182 | Dolphins |
| 22 | 4 | Greg Hardy | 69.72 | 67.49 | 68.08 | 391 | Panthers |
| 23 | 5 | Matt Roth | 69.36 | 59.32 | 71.89 | 1014 | Browns |
| 24 | 6 | Turk McBride | 69.27 | 61.35 | 71.42 | 448 | Lions |
| 25 | 7 | Jermaine Cunningham | 69.17 | 66.70 | 66.65 | 575 | Patriots |
| 26 | 8 | Jeff Charleston | 68.25 | 61.22 | 68.77 | 267 | Saints |
| 27 | 9 | James Hall | 67.94 | 53.88 | 73.14 | 802 | Rams |
| 28 | 10 | Anthony Spencer | 67.66 | 65.26 | 65.10 | 938 | Cowboys |
| 29 | 11 | Darryl Tapp | 67.49 | 63.12 | 67.27 | 460 | Eagles |
| 30 | 12 | Jason Pierre-Paul | 67.37 | 73.75 | 58.95 | 397 | Giants |
| 31 | 13 | Brandon Graham | 67.21 | 64.38 | 68.07 | 472 | Eagles |
| 32 | 14 | Antwan Applewhite | 67.14 | 55.35 | 71.87 | 606 | Chargers |
| 33 | 15 | Dave Ball | 67.04 | 55.87 | 75.52 | 432 | Titans |
| 34 | 16 | Kroy Biermann | 66.92 | 63.14 | 65.27 | 654 | Falcons |
| 35 | 17 | Mathias Kiwanuka | 66.78 | 60.09 | 87.08 | 155 | Giants |
| 36 | 18 | Mark Anderson | 66.74 | 60.22 | 67.96 | 456 | Texans |
| 37 | 19 | O'Brien Schofield | 66.24 | 57.58 | 75.15 | 133 | Cardinals |
| 38 | 20 | Aaron Kampman | 65.53 | 57.55 | 75.02 | 498 | Jaguars |
| 39 | 21 | Israel Idonije | 64.86 | 56.82 | 66.06 | 957 | Bears |
| 40 | 22 | Jeremy Mincey | 64.77 | 60.92 | 64.20 | 586 | Jaguars |
| 41 | 23 | Jason Hunter | 64.49 | 57.32 | 65.11 | 754 | Broncos |
| 42 | 24 | Jamaal Anderson | 63.75 | 67.89 | 56.83 | 431 | Falcons |
| 43 | 25 | Derrick Harvey | 62.61 | 57.41 | 62.95 | 350 | Jaguars |
| 44 | 26 | William Hayes | 62.34 | 63.27 | 59.64 | 545 | Titans |

### Rotation/backup (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 45 | 1 | Frostee Rucker | 61.29 | 59.47 | 65.64 | 283 | Bengals |
| 46 | 2 | Brian Robison | 61.03 | 58.12 | 58.80 | 323 | Vikings |
| 47 | 3 | C.J. Ah You | 60.46 | 54.02 | 60.59 | 344 | Rams |
| 48 | 4 | Chris Kelsay | 60.25 | 51.86 | 61.68 | 1063 | Bills |
| 49 | 5 | Jacob Ford | 60.20 | 56.60 | 60.52 | 568 | Titans |
| 50 | 6 | Michael Johnson | 60.06 | 59.43 | 56.31 | 666 | Bengals |
| 51 | 7 | Robert Ayers | 59.23 | 62.23 | 58.26 | 651 | Broncos |
| 52 | 8 | Kyle Vanden Bosch | 58.96 | 55.10 | 62.57 | 652 | Lions |
| 53 | 9 | Tim Jamison | 58.45 | 60.28 | 59.32 | 211 | Texans |
| 54 | 10 | George Selvie | 58.25 | 57.48 | 54.60 | 310 | Rams |
| 55 | 11 | Chauncey Davis | 57.36 | 55.69 | 54.30 | 423 | Falcons |
| 56 | 12 | Tim Crowder | 57.29 | 55.54 | 54.29 | 630 | Buccaneers |
| 57 | 13 | Trevor Scott | 57.20 | 57.78 | 58.89 | 476 | Raiders |
| 58 | 14 | Austen Lane | 56.67 | 58.03 | 56.79 | 307 | Jaguars |
| 59 | 15 | Andre Carter | 56.10 | 49.87 | 56.09 | 731 | Commanders |
| 60 | 16 | Jason Taylor | 55.88 | 45.36 | 58.72 | 715 | Jets |
| 61 | 17 | Robert Geathers | 55.81 | 55.26 | 52.01 | 796 | Bengals |
| 62 | 18 | Dave Tollefson | 54.62 | 55.06 | 55.36 | 143 | Giants |
| 63 | 19 | Kentwan Balmer | 54.59 | 54.53 | 50.47 | 564 | Seahawks |
| 64 | 20 | Michael Bennett | 54.49 | 56.32 | 52.23 | 423 | Buccaneers |
| 65 | 21 | Daniel Te'o-Nesheim | 54.09 | 55.90 | 59.58 | 110 | Eagles |
| 66 | 22 | Keyunta Dawson | 53.65 | 54.45 | 48.95 | 597 | Colts |
| 67 | 23 | Jesse Nading | 49.48 | 54.53 | 57.91 | 106 | Texans |
| 68 | 24 | Kyle Moore | 49.08 | 53.99 | 52.50 | 309 | Buccaneers |
| 69 | 25 | Jay Richardson | 48.30 | 52.83 | 48.42 | 119 | Seahawks |
| 70 | 26 | Antwan Odom | 45.00 | 54.45 | 50.17 | 158 | Bengals |

## G — Guard

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Carl Nicks | 94.10 | 87.40 | 94.40 | 1192 | Saints |
| 2 | 2 | Logan Mankins | 94.00 | 91.40 | 91.57 | 610 | Patriots |
| 3 | 3 | Mike Brisiel | 93.29 | 86.90 | 93.39 | 502 | Texans |
| 4 | 4 | Josh Sitton | 92.39 | 90.10 | 89.75 | 1294 | Packers |
| 5 | 5 | Wade Smith | 92.00 | 86.70 | 91.37 | 1067 | Texans |
| 6 | 6 | Evan Mathis | 91.62 | 92.80 | 86.67 | 110 | Bengals |
| 7 | 7 | John Greco | 91.49 | 91.10 | 87.58 | 146 | Rams |
| 8 | 8 | Chilo Rachal | 90.70 | 87.90 | 88.40 | 761 | 49ers |
| 9 | 9 | Mike Iupati | 89.47 | 82.10 | 90.22 | 951 | 49ers |
| 10 | 10 | Ben Grubbs | 88.76 | 82.70 | 88.64 | 1191 | Ravens |
| 11 | 11 | Todd Herremans | 88.33 | 80.30 | 89.52 | 1060 | Eagles |
| 12 | 12 | Brian Waters | 88.16 | 82.80 | 87.56 | 1083 | Chiefs |
| 13 | 13 | Harvey Dahl | 87.01 | 79.70 | 87.71 | 1211 | Falcons |
| 14 | 14 | Kris Dielman | 86.31 | 79.30 | 86.81 | 1001 | Chargers |
| 15 | 15 | Chris Snee | 86.11 | 77.40 | 87.75 | 1074 | Giants |
| 16 | 16 | Geoff Schwartz | 85.30 | 77.30 | 86.46 | 989 | Panthers |
| 17 | 17 | Richie Incognito | 85.16 | 77.50 | 86.10 | 1072 | Dolphins |
| 18 | 18 | Brandon Moore | 84.71 | 77.60 | 85.28 | 1249 | Jets |
| 19 | 19 | Rich Seubert | 84.55 | 80.80 | 82.88 | 1021 | Giants |
| 20 | 20 | Justin Blalock | 84.47 | 77.70 | 84.81 | 1211 | Falcons |
| 21 | 21 | Bobbie Williams | 84.14 | 75.60 | 85.66 | 1094 | Bengals |
| 22 | 22 | Chris Chester | 83.13 | 75.70 | 83.92 | 1006 | Ravens |
| 23 | 23 | Antoine Caldwell | 83.09 | 75.80 | 83.78 | 555 | Texans |
| 24 | 24 | Nate Livings | 82.94 | 73.70 | 84.93 | 986 | Bengals |
| 25 | 25 | Steve Hutchinson | 82.73 | 75.30 | 83.52 | 715 | Vikings |
| 26 | 26 | Louis Vasquez | 82.45 | 75.20 | 83.12 | 534 | Chargers |
| 27 | 27 | Vince Manuwai | 82.08 | 75.70 | 82.16 | 795 | Jaguars |
| 28 | 28 | Jahri Evans | 81.90 | 74.10 | 82.94 | 1192 | Saints |
| 29 | 29 | Lance Louis | 81.79 | 73.00 | 83.49 | 263 | Bears |
| 30 | 30 | Jacob Bell | 81.73 | 73.70 | 82.91 | 1079 | Rams |
| 31 | 31 | Daryn Colledge | 81.68 | 71.70 | 84.16 | 1228 | Packers |
| 32 | 32 | Kyle Kosier | 81.25 | 73.40 | 82.31 | 825 | Cowboys |
| 33 | 33 | Mike Pollak | 80.71 | 76.60 | 79.28 | 980 | Colts |
| 34 | 34 | Rob Sims | 80.35 | 72.40 | 81.48 | 1107 | Lions |
| 35 | 35 | Jon Asamoah | 80.17 | 74.80 | 79.58 | 158 | Chiefs |
| 36 | 36 | Matt Slauson | 80.15 | 71.50 | 81.75 | 1268 | Jets |
| 37 | 37 | Montrae Holland | 80.03 | 75.50 | 78.88 | 170 | Cowboys |
| 38 | 38 | Andy Levitre | 80.00 | 70.10 | 82.44 | 927 | Bills |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Dan Connolly | 79.75 | 71.60 | 81.02 | 872 | Patriots |
| 40 | 2 | Zane Beadles | 79.14 | 70.20 | 80.94 | 929 | Broncos |
| 41 | 3 | Eric Steinbach | 78.45 | 70.70 | 79.45 | 958 | Browns |
| 42 | 4 | Kraig Urbik | 78.12 | 75.90 | 75.44 | 159 | Bills |
| 43 | 5 | Chris Kuper | 78.09 | 68.40 | 80.38 | 987 | Broncos |
| 44 | 6 | Chad Rinehart | 77.80 | 78.80 | 72.97 | 202 | Bills |
| 45 | 7 | Alan Faneca | 77.63 | 68.60 | 79.49 | 960 | Cardinals |
| 46 | 8 | Justin Smiley | 77.54 | 66.10 | 81.00 | 269 | Jaguars |
| 47 | 9 | Stephen Peterman | 76.45 | 64.30 | 80.38 | 1076 | Lions |
| 48 | 10 | Ramon Foster | 76.41 | 65.90 | 79.25 | 832 | Steelers |
| 49 | 11 | John Jerry | 76.22 | 67.30 | 78.00 | 624 | Dolphins |
| 50 | 12 | Uche Nwaneri | 75.98 | 66.20 | 78.34 | 1061 | Jaguars |
| 51 | 13 | Adam Snyder | 75.86 | 70.80 | 75.07 | 262 | 49ers |
| 52 | 14 | Russ Hochstein | 75.73 | 65.50 | 78.39 | 347 | Broncos |
| 53 | 15 | Cooper Carlisle | 75.52 | 66.00 | 77.70 | 1084 | Raiders |
| 54 | 16 | Travelle Wharton | 75.42 | 64.80 | 78.33 | 538 | Panthers |
| 55 | 17 | Chris Kemoeatu | 75.10 | 63.50 | 78.66 | 1080 | Steelers |
| 56 | 18 | Ted Larsen | 74.46 | 62.40 | 78.33 | 634 | Buccaneers |

### Starter (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 57 | 1 | Max Jean-Gilles | 73.29 | 67.50 | 72.98 | 582 | Eagles |
| 58 | 2 | Jake Scott | 73.08 | 60.70 | 77.17 | 953 | Titans |
| 59 | 3 | Chris Williams | 72.68 | 61.50 | 75.97 | 903 | Bears |
| 60 | 4 | Artis Hicks | 72.63 | 58.80 | 77.68 | 547 | Commanders |
| 61 | 5 | Floyd Womack | 72.56 | 62.70 | 74.96 | 888 | Browns |
| 62 | 6 | Adam Goldberg | 71.14 | 59.90 | 74.46 | 963 | Rams |
| 63 | 7 | Tyronne Green | 69.87 | 57.80 | 73.75 | 598 | Chargers |
| 64 | 8 | Stacy Andrews | 68.48 | 59.40 | 70.36 | 763 | Seahawks |
| 65 | 9 | Kevin Boothe | 66.54 | 57.30 | 68.54 | 347 | Giants |
| 66 | 10 | Jamey Richard | 65.14 | 55.70 | 67.26 | 322 | Colts |
| 67 | 11 | Davin Joseph | 64.38 | 50.80 | 69.26 | 632 | Buccaneers |
| 68 | 12 | Cord Howard | 62.79 | 45.80 | 69.95 | 343 | Bills |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 69 | 1 | Shawn Lauvao | 57.85 | 48.20 | 60.12 | 113 | Browns |

## HB — Running Back

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jamaal Charles | 83.19 | 85.20 | 77.69 | 267 | Chiefs |
| 2 | 2 | Adrian Peterson | 80.70 | 86.50 | 72.66 | 264 | Vikings |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 3 | 1 | Darren McFadden | 77.17 | 69.20 | 78.32 | 202 | Raiders |
| 4 | 2 | Ahmad Bradshaw | 76.10 | 76.60 | 71.60 | 262 | Giants |
| 5 | 3 | Darren Sproles | 75.60 | 69.60 | 75.44 | 275 | Chargers |
| 6 | 4 | Arian Foster | 74.57 | 73.70 | 70.98 | 363 | Texans |
| 7 | 5 | Peyton Hillis | 74.30 | 74.20 | 70.20 | 316 | Browns |
| 8 | 6 | LeSean McCoy | 74.26 | 67.10 | 74.86 | 452 | Eagles |

### Starter (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Chris Johnson | 73.50 | 74.80 | 68.46 | 349 | Titans |
| 10 | 2 | Rashad Jennings | 73.34 | 62.10 | 76.66 | 164 | Jaguars |
| 11 | 3 | Matt Forte | 72.76 | 74.40 | 67.50 | 320 | Bears |
| 12 | 4 | Maurice Jones-Drew | 71.67 | 72.10 | 67.22 | 235 | Jaguars |
| 13 | 5 | Marshawn Lynch | 71.28 | 68.50 | 68.96 | 182 | Seahawks |
| 14 | 6 | Justin Forsett | 71.01 | 71.10 | 66.79 | 255 | Seahawks |
| 15 | 7 | Rashard Mendenhall | 70.48 | 74.90 | 63.37 | 228 | Steelers |
| 16 | 8 | Mewelde Moore | 70.43 | 74.10 | 63.82 | 140 | Steelers |
| 17 | 9 | Brandon Jackson | 70.19 | 71.70 | 65.01 | 259 | Packers |
| 18 | 10 | Felix Jones | 70.03 | 67.10 | 67.82 | 227 | Cowboys |
| 19 | 11 | Fred Jackson | 69.38 | 61.70 | 70.34 | 294 | Bills |
| 20 | 12 | Mike Goodson | 69.27 | 62.10 | 69.88 | 199 | Panthers |
| 21 | 13 | C.J. Spiller | 68.57 | 70.80 | 62.91 | 114 | Bills |
| 22 | 14 | Knowshon Moreno | 68.00 | 65.50 | 65.50 | 207 | Broncos |
| 23 | 15 | Ray Rice | 67.82 | 68.60 | 63.13 | 364 | Ravens |
| 24 | 16 | Shonn Greene | 67.26 | 67.90 | 62.67 | 133 | Jets |
| 25 | 17 | Michael Turner | 67.09 | 64.00 | 64.99 | 183 | Falcons |
| 26 | 18 | Pierre Thomas | 66.93 | 70.80 | 60.18 | 100 | Saints |
| 27 | 19 | Tim Hightower | 66.85 | 56.90 | 69.32 | 280 | Cardinals |
| 28 | 20 | BenJarvus Green-Ellis | 66.56 | 63.40 | 64.50 | 137 | Patriots |
| 29 | 21 | Joseph Addai | 66.43 | 64.70 | 63.42 | 188 | Colts |
| 30 | 22 | Keiland Williams | 66.38 | 63.20 | 64.33 | 264 | Commanders |
| 31 | 23 | Michael Bush | 66.29 | 67.60 | 61.25 | 108 | Raiders |
| 32 | 24 | Carnell Williams | 66.02 | 63.50 | 63.54 | 276 | Buccaneers |
| 33 | 25 | Jonathan Stewart | 65.78 | 58.70 | 66.34 | 108 | Panthers |
| 34 | 26 | Frank Gore | 65.75 | 61.30 | 64.55 | 271 | 49ers |
| 35 | 27 | Reggie Bush | 65.23 | 64.90 | 61.29 | 179 | Saints |
| 36 | 28 | Steven Jackson | 65.17 | 62.20 | 62.99 | 372 | Rams |
| 37 | 29 | Toby Gerhart | 64.40 | 57.40 | 64.90 | 133 | Vikings |
| 38 | 30 | Ronnie Brown | 62.62 | 62.30 | 58.66 | 225 | Dolphins |
| 39 | 31 | Ricky Williams | 62.49 | 55.50 | 62.98 | 154 | Dolphins |
| 40 | 32 | Jason Snelling | 62.34 | 55.60 | 62.67 | 245 | Falcons |
| 41 | 33 | Donald Brown | 62.19 | 54.70 | 63.02 | 211 | Colts |
| 42 | 34 | LaDainian Tomlinson | 62.09 | 54.00 | 63.32 | 377 | Jets |

### Rotation/backup (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 43 | 1 | Maurice Morris | 61.34 | 59.20 | 58.60 | 133 | Lions |
| 44 | 2 | Chester Taylor | 61.15 | 64.30 | 54.89 | 133 | Bears |
| 45 | 3 | Jahvid Best | 61.12 | 60.30 | 57.50 | 302 | Lions |
| 46 | 4 | Cedric Benson | 61.02 | 58.20 | 58.73 | 238 | Bengals |
| 47 | 5 | Julius Jones | 58.91 | 53.40 | 58.42 | 124 | Saints |
| 48 | 6 | Kenneth Darby | 58.02 | 56.80 | 54.66 | 108 | Rams |
| 49 | 7 | Correll Buckhalter | 56.99 | 53.40 | 55.22 | 182 | Broncos |
| 50 | 8 | Thomas Jones | 56.11 | 50.00 | 56.02 | 140 | Chiefs |

## LB — Linebacker

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Sean Lee | 87.75 | 91.70 | 84.08 | 167 | Cowboys |
| 2 | 2 | Derrick Johnson | 85.78 | 90.20 | 78.66 | 1110 | Chiefs |
| 3 | 3 | Lawrence Timmons | 84.62 | 87.50 | 78.53 | 1116 | Steelers |
| 4 | 4 | Brandon Spikes | 84.57 | 90.00 | 79.92 | 360 | Patriots |
| 5 | 5 | Patrick Willis | 84.32 | 90.40 | 77.14 | 966 | 49ers |
| 6 | 6 | Takeo Spikes | 84.26 | 85.30 | 79.40 | 787 | 49ers |
| 7 | 7 | Daryl Washington | 84.18 | 86.90 | 78.20 | 518 | Cardinals |
| 8 | 8 | Desmond Bishop | 83.20 | 89.10 | 75.10 | 941 | Packers |
| 9 | 9 | Bart Scott | 82.31 | 84.60 | 76.61 | 1020 | Jets |
| 10 | 10 | Brian Urlacher | 81.92 | 83.30 | 76.84 | 1156 | Bears |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Ray Lewis | 79.96 | 80.10 | 75.70 | 1145 | Ravens |
| 12 | 2 | Stephen Tulloch | 79.87 | 77.80 | 77.09 | 1186 | Titans |
| 13 | 3 | Dan Connor | 79.61 | 91.60 | 75.78 | 258 | Panthers |
| 14 | 4 | James Farrior | 79.01 | 81.50 | 73.19 | 1074 | Steelers |
| 15 | 5 | Bradie James | 78.98 | 78.70 | 75.00 | 907 | Cowboys |
| 16 | 6 | E.J. Henderson | 78.07 | 77.20 | 74.48 | 945 | Vikings |
| 17 | 7 | Larry Foote | 76.23 | 80.40 | 69.28 | 176 | Steelers |
| 18 | 8 | Marvin Mitchell | 76.10 | 73.60 | 73.60 | 378 | Saints |
| 19 | 9 | Karlos Dansby | 75.83 | 77.90 | 72.36 | 846 | Dolphins |
| 20 | 10 | Chris Gocong | 75.40 | 76.50 | 70.50 | 962 | Browns |
| 21 | 11 | James Laurinaitis | 75.39 | 73.10 | 72.75 | 1058 | Rams |
| 22 | 12 | Rolando McClain | 75.30 | 74.00 | 73.03 | 928 | Raiders |
| 23 | 13 | D.J. Williams | 75.10 | 73.40 | 72.07 | 1063 | Broncos |
| 24 | 14 | Jerod Mayo | 74.51 | 72.30 | 71.81 | 1119 | Patriots |

### Starter (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 25 | 1 | Stephen Nicholas | 73.40 | 70.80 | 70.96 | 704 | Falcons |
| 26 | 2 | London Fletcher | 73.10 | 70.40 | 70.74 | 1094 | Commanders |
| 27 | 3 | Channing Crowder | 72.80 | 74.30 | 72.83 | 503 | Dolphins |
| 28 | 4 | Kevin Burnett | 72.57 | 71.80 | 68.92 | 965 | Chargers |
| 29 | 5 | Michael Boley | 72.28 | 70.00 | 69.64 | 940 | Giants |
| 30 | 6 | Jon Beason | 71.91 | 68.50 | 70.02 | 1106 | Panthers |
| 31 | 7 | Gary Brackett | 71.58 | 71.90 | 70.34 | 835 | Colts |
| 32 | 8 | Quincy Black | 70.85 | 71.70 | 71.31 | 514 | Buccaneers |
| 33 | 9 | Keith Bulluck | 70.80 | 71.90 | 69.04 | 313 | Giants |
| 34 | 10 | NaVorro Bowman | 70.70 | 71.90 | 70.94 | 212 | 49ers |
| 35 | 11 | Adam Hayward | 70.40 | 71.50 | 71.75 | 140 | Buccaneers |
| 36 | 12 | Chad Greenway | 70.32 | 66.40 | 68.76 | 969 | Vikings |
| 37 | 13 | Rey Maualuga | 70.24 | 67.90 | 67.63 | 611 | Bengals |
| 38 | 14 | Brandon Johnson | 70.14 | 68.30 | 67.20 | 500 | Bengals |
| 39 | 15 | Daryl Smith | 70.13 | 69.40 | 66.45 | 964 | Jaguars |
| 40 | 16 | James Anderson | 70.11 | 67.90 | 67.42 | 1071 | Panthers |
| 41 | 17 | Kirk Morrison | 69.83 | 65.40 | 68.62 | 737 | Jaguars |
| 42 | 18 | Dane Fletcher | 69.35 | 74.10 | 67.22 | 166 | Patriots |
| 43 | 19 | Jameel McClain | 68.74 | 64.60 | 67.34 | 574 | Ravens |
| 44 | 20 | A.J. Hawk | 68.71 | 65.30 | 66.82 | 1166 | Packers |
| 45 | 21 | David Harris | 68.52 | 64.60 | 66.96 | 1183 | Jets |
| 46 | 22 | Tim Dobbins | 68.29 | 65.40 | 69.19 | 305 | Dolphins |
| 47 | 23 | JoLonn Dunbar | 68.25 | 64.80 | 70.55 | 357 | Saints |
| 48 | 24 | Keith Brooking | 68.09 | 63.30 | 67.12 | 891 | Cowboys |
| 49 | 25 | Bryan Kehl | 67.91 | 69.30 | 68.02 | 206 | Rams |
| 50 | 26 | Lance Briggs | 67.67 | 65.10 | 65.22 | 1014 | Bears |
| 51 | 27 | Dekoda Watson | 67.17 | 67.40 | 71.19 | 130 | Buccaneers |
| 52 | 28 | Jovan Belcher | 66.87 | 64.10 | 64.55 | 670 | Chiefs |
| 53 | 29 | Will Herring | 66.83 | 64.70 | 64.08 | 289 | Seahawks |
| 54 | 30 | Keith Rivers | 66.71 | 67.00 | 64.44 | 500 | Bengals |
| 55 | 31 | Aaron Curry | 66.69 | 62.30 | 65.45 | 979 | Seahawks |
| 56 | 32 | Akeem Jordan | 66.69 | 70.10 | 68.58 | 195 | Eagles |
| 57 | 33 | Paris Lenon | 66.55 | 61.90 | 65.48 | 1092 | Cardinals |
| 58 | 34 | Barrett Ruud | 66.45 | 60.50 | 66.25 | 1023 | Buccaneers |
| 59 | 35 | Gerald McRath | 66.39 | 65.20 | 67.18 | 454 | Titans |
| 60 | 36 | DeMeco Ryans | 66.20 | 73.90 | 70.32 | 375 | Texans |
| 61 | 37 | Nick Roach | 66.18 | 70.60 | 66.36 | 126 | Bears |
| 62 | 38 | Pisa Tinoisamoa | 65.99 | 64.20 | 65.10 | 489 | Bears |
| 63 | 39 | DeAndre Levy | 65.92 | 68.70 | 65.10 | 716 | Lions |
| 64 | 40 | Lofa Tatupu | 65.56 | 61.30 | 64.23 | 1213 | Seahawks |
| 65 | 41 | Reggie Torbor | 65.55 | 64.00 | 68.67 | 411 | Bills |
| 66 | 42 | Mike Peterson | 65.53 | 59.60 | 65.31 | 511 | Falcons |
| 67 | 43 | Philip Wheeler | 64.71 | 59.70 | 63.88 | 376 | Colts |
| 68 | 44 | Curtis Lofton | 64.65 | 58.50 | 64.58 | 1005 | Falcons |
| 69 | 45 | Mario Haggan | 64.19 | 57.40 | 64.55 | 1090 | Broncos |
| 70 | 46 | Brandon Siler | 63.68 | 63.30 | 66.02 | 314 | Chargers |
| 71 | 47 | Moise Fokou | 63.61 | 59.20 | 63.41 | 416 | Eagles |
| 72 | 48 | Paul Posluszny | 63.57 | 58.80 | 64.66 | 932 | Bills |
| 73 | 49 | Gary Guyton | 62.90 | 57.90 | 62.06 | 661 | Patriots |
| 74 | 50 | Keenan Clayton | 62.88 | 68.80 | 68.18 | 107 | Eagles |
| 75 | 51 | Dannell Ellerbe | 62.38 | 57.40 | 64.67 | 301 | Ravens |

### Rotation/backup (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 76 | 1 | Kavell Conner | 61.87 | 60.20 | 64.02 | 316 | Colts |
| 77 | 2 | Na'il Diggs | 61.29 | 59.50 | 62.49 | 389 | Rams |
| 78 | 3 | Will Witherspoon | 60.33 | 55.90 | 59.11 | 1157 | Titans |
| 79 | 4 | Brian Cushing | 60.20 | 58.00 | 61.67 | 788 | Texans |
| 80 | 5 | Chris Chamberlain | 60.05 | 58.80 | 64.02 | 172 | Rams |
| 81 | 6 | Quentin Groves | 59.96 | 55.60 | 60.79 | 469 | Raiders |
| 82 | 7 | Stewart Bradley | 59.59 | 56.40 | 61.71 | 681 | Eagles |
| 83 | 8 | Julian Peterson | 58.96 | 53.10 | 59.73 | 919 | Lions |
| 84 | 9 | Justin Durant | 58.58 | 52.40 | 64.78 | 480 | Jaguars |
| 85 | 10 | Jonathan Vilma | 57.48 | 47.50 | 59.97 | 1018 | Saints |
| 86 | 11 | Russell Allen | 56.85 | 47.50 | 63.09 | 289 | Jaguars |
| 87 | 12 | Rocky McIntosh | 56.44 | 49.50 | 57.93 | 933 | Commanders |
| 88 | 13 | Pat Angerer | 55.78 | 49.30 | 59.06 | 574 | Colts |
| 89 | 14 | Darryl Sharpton | 55.37 | 53.00 | 60.08 | 204 | Texans |
| 90 | 15 | Omar Gaither | 55.11 | 59.80 | 66.31 | 179 | Eagles |
| 91 | 16 | Andra Davis | 54.57 | 58.00 | 61.53 | 220 | Bills |
| 92 | 17 | Clint Session | 54.31 | 57.80 | 63.78 | 291 | Colts |
| 93 | 18 | Tavares Gooden | 54.22 | 50.30 | 58.92 | 264 | Ravens |
| 94 | 19 | Nick Barnett | 53.17 | 55.40 | 66.02 | 252 | Packers |
| 95 | 20 | Sean Weatherspoon | 52.83 | 44.60 | 58.32 | 455 | Falcons |
| 96 | 21 | Zach Diles | 52.49 | 37.90 | 59.08 | 648 | Texans |
| 97 | 22 | Wesley Woodyard | 51.31 | 47.40 | 60.62 | 145 | Broncos |
| 98 | 23 | Brandon Chillar | 50.15 | 46.30 | 56.89 | 130 | Packers |
| 99 | 24 | Ernie Sims | 49.78 | 34.50 | 55.80 | 902 | Eagles |
| 100 | 25 | Scott Shanle | 49.71 | 37.50 | 54.71 | 895 | Saints |
| 101 | 26 | Keith Ellison | 49.20 | 53.40 | 62.24 | 121 | Bills |
| 102 | 27 | Larry Grant | 47.49 | 35.10 | 54.72 | 305 | Rams |
| 103 | 28 | Ricky Brown | 47.27 | 43.90 | 58.77 | 138 | Raiders |
| 104 | 29 | Zack Follett | 46.81 | 46.80 | 58.62 | 146 | Lions |
| 105 | 30 | Jamar Chaney | 45.00 | 30.20 | 60.34 | 245 | Eagles |

## QB — Quarterback

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Rodgers | 81.95 | 91.60 | 81.27 | 728 | Packers |

### Good (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 2 | 1 | Philip Rivers | 78.50 | 92.80 | 80.35 | 612 | Chargers |
| 3 | 2 | Drew Brees | 77.88 | 87.00 | 72.52 | 776 | Saints |
| 4 | 3 | Peyton Manning | 77.75 | 89.50 | 71.50 | 746 | Colts |
| 5 | 4 | Tom Brady | 75.54 | 88.50 | 77.67 | 602 | Patriots |

### Starter (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Ben Roethlisberger | 73.72 | 87.90 | 75.59 | 571 | Steelers |
| 7 | 2 | Matt Schaub | 73.60 | 84.10 | 72.15 | 645 | Texans |
| 8 | 3 | Matt Ryan | 73.47 | 85.70 | 68.33 | 700 | Falcons |
| 9 | 4 | Joe Flacco | 72.17 | 78.10 | 72.89 | 655 | Ravens |
| 10 | 5 | Eli Manning | 71.27 | 82.40 | 71.06 | 597 | Giants |
| 11 | 6 | Josh Freeman | 69.98 | 77.00 | 73.48 | 568 | Buccaneers |
| 12 | 7 | Michael Vick | 69.13 | 74.30 | 76.15 | 531 | Eagles |
| 13 | 8 | Carson Palmer | 68.71 | 75.50 | 65.74 | 655 | Bengals |
| 14 | 9 | Tony Romo | 68.60 | 90.90 | 76.44 | 235 | Cowboys |
| 15 | 10 | Kyle Orton | 67.04 | 70.70 | 69.72 | 574 | Broncos |
| 16 | 11 | David Garrard | 66.86 | 77.60 | 73.31 | 442 | Jaguars |
| 17 | 12 | Mark Sanchez | 66.75 | 72.10 | 62.83 | 664 | Jets |
| 18 | 13 | Jay Cutler | 66.51 | 65.80 | 70.86 | 601 | Bears |
| 19 | 14 | Matt Hasselbeck | 66.11 | 73.40 | 63.40 | 592 | Seahawks |
| 20 | 15 | Shaun Hill | 64.64 | 79.00 | 61.63 | 464 | Lions |
| 21 | 16 | Matt Cassel | 64.63 | 67.20 | 66.82 | 544 | Chiefs |
| 22 | 17 | Vince Young | 64.45 | 70.30 | 77.76 | 192 | Titans |
| 23 | 18 | Troy Smith | 64.25 | 76.10 | 72.30 | 186 | 49ers |
| 24 | 19 | Ryan Fitzpatrick | 64.19 | 68.20 | 66.04 | 512 | Bills |
| 25 | 20 | Chad Henne | 64.00 | 69.70 | 61.66 | 561 | Dolphins |
| 26 | 21 | Donovan McNabb | 63.69 | 67.00 | 63.77 | 550 | Commanders |
| 27 | 22 | Sam Bradford | 63.29 | 65.70 | 59.21 | 672 | Rams |
| 28 | 23 | Jason Campbell | 63.24 | 68.20 | 68.46 | 421 | Raiders |
| 29 | 24 | Jon Kitna | 63.05 | 64.70 | 71.47 | 374 | Cowboys |
| 30 | 25 | Matthew Stafford | 62.55 | 78.80 | 64.00 | 114 | Lions |
| 31 | 26 | Kerry Collins | 62.34 | 70.00 | 66.20 | 309 | Titans |

### Rotation/backup (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 32 | 1 | Colt McCoy | 61.90 | 72.60 | 63.22 | 272 | Browns |
| 33 | 2 | Seneca Wallace | 61.27 | 67.20 | 68.33 | 116 | Browns |
| 34 | 3 | Rex Grossman | 60.95 | 67.70 | 66.64 | 152 | Commanders |
| 35 | 4 | Alex Smith | 60.82 | 62.40 | 63.79 | 392 | 49ers |
| 36 | 5 | Brett Favre | 60.62 | 64.10 | 61.33 | 409 | Vikings |
| 37 | 6 | Jimmy Clausen | 58.65 | 61.70 | 56.50 | 359 | Panthers |
| 38 | 7 | Bruce Gradkowski | 58.01 | 59.40 | 60.93 | 182 | Raiders |
| 39 | 8 | Drew Stanton | 57.77 | 56.70 | 62.78 | 137 | Lions |
| 40 | 9 | Jake Delhomme | 56.85 | 59.50 | 56.64 | 164 | Browns |
| 41 | 10 | Derek Anderson | 56.68 | 49.60 | 58.11 | 369 | Cardinals |
| 42 | 11 | Kevin Kolb | 56.63 | 52.10 | 60.17 | 225 | Eagles |
| 43 | 12 | Charlie Whitehurst | 56.38 | 59.20 | 55.10 | 121 | Seahawks |
| 44 | 13 | John Skelton | 56.36 | 56.80 | 57.00 | 147 | Cardinals |
| 45 | 14 | Trent Edwards | 55.33 | 55.30 | 54.06 | 129 | Jaguars |
| 46 | 15 | Matt Moore | 54.69 | 49.60 | 55.85 | 165 | Panthers |
| 47 | 16 | Joe Webb III | 54.07 | 46.40 | 56.27 | 116 | Vikings |

## S — Safety

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Michael Huff | 88.05 | 81.90 | 87.99 | 972 | Raiders |
| 2 | 2 | Quintin Mikell | 84.58 | 82.20 | 82.00 | 1032 | Eagles |
| 3 | 3 | Dawan Landry | 80.33 | 74.50 | 80.05 | 1121 | Ravens |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Eric Weddle | 79.62 | 73.50 | 79.53 | 957 | Chargers |
| 5 | 2 | Dwight Lowery | 79.59 | 79.80 | 75.28 | 420 | Jets |
| 6 | 3 | Danieal Manning | 78.79 | 71.40 | 79.55 | 991 | Bears |
| 7 | 4 | Sherrod Martin | 78.63 | 80.10 | 74.52 | 975 | Panthers |
| 8 | 5 | Jordan Pugh | 77.30 | 78.90 | 78.32 | 134 | Panthers |
| 9 | 6 | Malcolm Jenkins | 76.39 | 76.00 | 73.52 | 821 | Saints |
| 10 | 7 | Troy Polamalu | 75.94 | 70.80 | 75.20 | 1019 | Steelers |
| 11 | 8 | Rashad Johnson | 75.84 | 70.60 | 75.16 | 376 | Cardinals |
| 12 | 9 | Antoine Bethea | 75.74 | 66.00 | 78.06 | 1102 | Colts |
| 13 | 10 | Kerry Rhodes | 75.34 | 73.00 | 72.74 | 1130 | Cardinals |
| 14 | 11 | Brodney Pool | 75.31 | 68.90 | 75.41 | 929 | Jets |
| 15 | 12 | Kenny Phillips | 75.13 | 67.00 | 76.38 | 943 | Giants |

### Starter (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Ryan Mundy | 73.14 | 70.70 | 77.90 | 182 | Steelers |
| 17 | 2 | Thomas DeCoud | 72.24 | 70.20 | 69.44 | 972 | Falcons |
| 18 | 3 | Nick Collins | 71.76 | 65.70 | 71.63 | 1208 | Packers |
| 19 | 4 | Yeremiah Bell | 71.70 | 63.40 | 73.06 | 965 | Dolphins |
| 20 | 5 | George Wilson | 71.68 | 69.60 | 69.94 | 256 | Bills |
| 21 | 6 | Abram Elam | 70.80 | 62.40 | 72.24 | 1047 | Browns |
| 22 | 7 | Gerald Sensabaugh | 69.22 | 61.90 | 69.93 | 918 | Cowboys |
| 23 | 8 | Patrick Chung | 69.02 | 60.70 | 71.43 | 835 | Patriots |
| 24 | 9 | Chris Crocker | 68.11 | 66.80 | 72.12 | 519 | Bengals |
| 25 | 10 | Jairus Byrd | 67.89 | 56.20 | 71.52 | 897 | Bills |
| 26 | 11 | Louis Delmas | 67.64 | 62.30 | 68.07 | 900 | Lions |
| 27 | 12 | Adrian Wilson | 67.62 | 59.30 | 69.00 | 1129 | Cardinals |
| 28 | 13 | Jordan Babineaux | 66.79 | 62.00 | 65.82 | 528 | Seahawks |
| 29 | 14 | Mike Mitchell | 66.72 | 59.90 | 68.14 | 497 | Raiders |
| 30 | 15 | Eric Berry | 66.63 | 56.50 | 69.22 | 1145 | Chiefs |
| 31 | 16 | Reshad Jones | 65.98 | 63.30 | 77.02 | 151 | Dolphins |
| 32 | 17 | Cody Grimm | 65.71 | 62.30 | 71.11 | 526 | Buccaneers |
| 33 | 18 | Ryan Clark | 65.19 | 54.40 | 68.22 | 1143 | Steelers |
| 34 | 19 | Eric Smith | 65.03 | 57.60 | 65.81 | 693 | Jets |
| 35 | 20 | Darian Stewart | 64.78 | 63.30 | 66.80 | 188 | Rams |
| 36 | 21 | Kam Chancellor | 64.59 | 52.80 | 68.29 | 159 | Seahawks |
| 37 | 22 | Kendrick Lewis | 64.32 | 61.20 | 65.36 | 848 | Chiefs |
| 38 | 23 | Chris Hope | 64.29 | 52.80 | 67.78 | 1184 | Titans |
| 39 | 24 | James Sanders | 64.26 | 56.50 | 65.27 | 810 | Patriots |
| 40 | 25 | Mike Adams | 64.23 | 56.30 | 66.38 | 306 | Browns |
| 41 | 26 | LaRon Landry | 64.21 | 56.70 | 72.35 | 645 | Commanders |
| 42 | 27 | Jarrad Page | 63.88 | 63.50 | 66.21 | 186 | Patriots |
| 43 | 28 | Usama Young | 63.21 | 62.80 | 65.57 | 201 | Saints |
| 44 | 29 | Charlie Peprah | 62.72 | 56.30 | 62.84 | 906 | Packers |
| 45 | 30 | Brian Dawkins | 62.64 | 54.70 | 68.96 | 678 | Broncos |
| 46 | 31 | Roy Williams | 62.15 | 54.90 | 66.99 | 466 | Bengals |

### Rotation/backup (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 47 | 1 | Sean Jones | 61.27 | 53.40 | 62.35 | 938 | Buccaneers |
| 48 | 2 | Roman Harper | 61.17 | 48.10 | 65.72 | 913 | Saints |
| 49 | 3 | Husain Abdullah | 61.05 | 48.20 | 66.48 | 874 | Vikings |
| 50 | 4 | Reggie Smith | 60.98 | 56.70 | 62.80 | 548 | 49ers |
| 51 | 5 | Tom Zbikowski | 60.93 | 63.20 | 68.66 | 323 | Ravens |
| 52 | 6 | Nedu Ndukwe | 60.91 | 54.40 | 65.25 | 530 | Bengals |
| 53 | 7 | Jim Leonhard | 60.81 | 54.60 | 65.99 | 719 | Jets |
| 54 | 8 | Amari Spievey | 60.63 | 54.70 | 64.58 | 553 | Lions |
| 55 | 9 | James Ihedigbo | 60.49 | 60.60 | 58.34 | 119 | Jets |
| 56 | 10 | William Moore | 60.43 | 51.10 | 62.49 | 961 | Falcons |
| 57 | 11 | Earl Thomas III | 59.84 | 47.90 | 63.64 | 1241 | Seahawks |
| 58 | 12 | Haruki Nakamura | 59.56 | 53.70 | 59.30 | 275 | Ravens |
| 59 | 13 | Reed Doughty | 59.48 | 49.90 | 63.78 | 615 | Commanders |
| 60 | 14 | Reggie Nelson | 59.36 | 50.30 | 66.44 | 489 | Bengals |
| 61 | 15 | Kurt Coleman | 59.22 | 45.00 | 66.61 | 333 | Eagles |
| 62 | 16 | Darren Sharper | 58.84 | 58.30 | 62.33 | 320 | Saints |
| 63 | 17 | Corey Lynch | 58.63 | 59.20 | 70.05 | 312 | Buccaneers |
| 64 | 18 | Michael Griffin | 58.02 | 48.30 | 60.34 | 1179 | Titans |
| 65 | 19 | Anthony Smith | 57.94 | 65.60 | 64.63 | 163 | Packers |
| 66 | 20 | Taylor Mays | 57.73 | 48.40 | 64.99 | 427 | 49ers |
| 67 | 21 | Dashon Goldson | 57.49 | 47.00 | 60.32 | 1011 | 49ers |
| 68 | 22 | Charles Godfrey | 57.43 | 46.60 | 60.48 | 1103 | Panthers |
| 69 | 23 | Donte Whitner | 57.16 | 35.90 | 67.16 | 1093 | Bills |
| 70 | 24 | Sabby Piscitelli | 57.12 | 54.00 | 62.33 | 143 | Browns |
| 71 | 25 | Chris Clemons | 56.64 | 46.00 | 60.60 | 865 | Dolphins |
| 72 | 26 | Steve Gregory | 56.10 | 52.10 | 61.90 | 354 | Chargers |
| 73 | 27 | Courtney Greene | 56.03 | 45.40 | 62.08 | 584 | Jaguars |
| 74 | 28 | Bernard Pollard | 55.83 | 39.30 | 63.72 | 987 | Texans |
| 75 | 29 | Michael Lewis | 55.43 | 60.00 | 57.01 | 194 | Rams |
| 76 | 30 | Chris Harris | 55.21 | 39.10 | 61.79 | 918 | Bears |
| 77 | 31 | T.J. Ward | 55.03 | 37.90 | 62.29 | 1053 | Browns |
| 78 | 32 | Tyvon Branch | 54.93 | 35.20 | 63.91 | 992 | Raiders |
| 79 | 33 | David Bruton | 53.96 | 54.40 | 55.75 | 169 | Broncos |
| 80 | 34 | Gerald Alexander | 53.13 | 58.00 | 65.72 | 187 | Panthers |
| 81 | 35 | Craig Dahl | 53.08 | 41.10 | 57.93 | 907 | Rams |
| 82 | 36 | Brandon Meriweather | 52.77 | 33.80 | 61.25 | 930 | Patriots |
| 83 | 37 | Antrel Rolle | 52.31 | 35.60 | 59.29 | 991 | Giants |
| 84 | 38 | Major Wright | 52.07 | 38.40 | 61.19 | 368 | Bears |
| 85 | 39 | Nate Allen | 51.92 | 36.40 | 61.24 | 781 | Eagles |
| 86 | 40 | Barry Church | 50.66 | 47.20 | 51.94 | 117 | Cowboys |
| 87 | 41 | Troy Nolan | 50.52 | 36.90 | 59.60 | 400 | Texans |
| 88 | 42 | Sean Considine | 49.24 | 42.30 | 54.90 | 411 | Jaguars |
| 89 | 43 | Jamarca Sanford | 49.23 | 53.40 | 55.70 | 209 | Vikings |
| 90 | 44 | Don Carey | 48.92 | 39.10 | 54.44 | 656 | Jaguars |
| 91 | 45 | Tyrone Culver | 48.47 | 41.70 | 57.15 | 117 | Dolphins |
| 92 | 46 | Darcel McBath | 48.44 | 44.20 | 60.52 | 160 | Broncos |
| 93 | 47 | Madieu Williams | 48.33 | 32.40 | 56.86 | 801 | Vikings |
| 94 | 48 | Ray Ventrone | 48.07 | 42.60 | 51.71 | 105 | Browns |
| 95 | 49 | Kareem Moore | 45.23 | 31.80 | 54.18 | 774 | Commanders |
| 96 | 50 | Erik Coleman | 45.00 | 41.40 | 52.58 | 147 | Falcons |
| 97 | 51 | Morgan Burnett | 45.00 | 40.10 | 56.68 | 197 | Packers |

## T — Tackle

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Andrew Whitworth | 95.27 | 90.90 | 94.02 | 1093 | Bengals |
| 2 | 2 | Doug Free | 91.36 | 83.80 | 92.23 | 1073 | Cowboys |
| 3 | 3 | Jake Long | 89.85 | 84.10 | 89.51 | 1054 | Dolphins |
| 4 | 4 | Jordan Gross | 89.58 | 84.30 | 88.93 | 988 | Panthers |
| 5 | 5 | D'Brickashaw Ferguson | 89.01 | 83.30 | 88.65 | 1307 | Jets |
| 6 | 6 | Jason Peters | 88.24 | 82.60 | 87.83 | 857 | Eagles |
| 7 | 7 | Ryan Clady | 88.16 | 82.20 | 87.97 | 1049 | Broncos |
| 8 | 8 | Joe Thomas | 87.74 | 82.20 | 87.26 | 958 | Browns |
| 9 | 9 | Eric Winston | 87.42 | 79.10 | 88.80 | 1067 | Texans |
| 10 | 10 | Marcus McNeill | 87.14 | 80.70 | 87.26 | 693 | Chargers |
| 11 | 11 | Tyson Clabo | 87.01 | 80.80 | 86.98 | 1193 | Falcons |
| 12 | 12 | Bryant McKinnie | 85.35 | 78.60 | 85.69 | 1023 | Vikings |
| 13 | 13 | Duane Brown | 85.07 | 76.90 | 86.35 | 825 | Texans |
| 14 | 14 | Matt Light | 84.77 | 77.00 | 85.78 | 1057 | Patriots |
| 15 | 15 | Jeff Backus | 83.92 | 77.60 | 83.97 | 1102 | Lions |
| 16 | 16 | Branden Albert | 82.81 | 73.70 | 84.72 | 1047 | Chiefs |
| 17 | 17 | Sebastian Vollmer | 82.61 | 74.00 | 84.18 | 1073 | Patriots |
| 18 | 18 | David Stewart | 82.49 | 72.70 | 84.85 | 953 | Titans |
| 19 | 19 | Barry Richardson | 81.74 | 71.20 | 84.60 | 1080 | Chiefs |
| 20 | 20 | Jon Stinchcomb | 81.56 | 70.50 | 84.76 | 1179 | Saints |
| 21 | 21 | Michael Roos | 81.34 | 73.10 | 82.67 | 953 | Titans |
| 22 | 22 | Ryan O'Callaghan | 81.27 | 77.50 | 79.62 | 160 | Chiefs |
| 23 | 23 | Gosder Cherilus | 81.20 | 72.60 | 82.76 | 820 | Lions |
| 24 | 24 | Flozell Adams | 81.07 | 70.10 | 84.21 | 1063 | Steelers |
| 25 | 25 | Eben Britton | 80.97 | 71.70 | 82.99 | 424 | Jaguars |
| 26 | 26 | Sean Locklear | 80.93 | 72.60 | 82.31 | 1103 | Seahawks |
| 27 | 27 | Donald Penn | 80.81 | 73.30 | 81.65 | 983 | Buccaneers |
| 28 | 28 | Ryan Harris | 80.71 | 69.50 | 84.01 | 655 | Broncos |
| 29 | 29 | Michael Oher | 80.70 | 69.50 | 84.00 | 1146 | Ravens |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Mario Henderson | 79.98 | 69.90 | 82.53 | 450 | Raiders |
| 31 | 2 | Chad Clifton | 79.97 | 71.70 | 81.32 | 1219 | Packers |
| 32 | 3 | Brandon Keith | 79.78 | 70.30 | 81.94 | 526 | Cardinals |
| 33 | 4 | King Dunlap | 79.77 | 70.40 | 81.85 | 433 | Eagles |
| 34 | 5 | Vernon Carey | 79.61 | 69.50 | 82.19 | 764 | Dolphins |
| 35 | 6 | Russell Okung | 79.50 | 69.90 | 81.74 | 671 | Seahawks |
| 36 | 7 | Demetress Bell | 78.69 | 69.00 | 80.98 | 950 | Bills |
| 37 | 8 | Winston Justice | 78.55 | 68.30 | 81.22 | 921 | Eagles |
| 38 | 9 | Joe Staley | 78.27 | 68.60 | 80.55 | 558 | 49ers |
| 39 | 10 | Jason Smith | 77.53 | 65.60 | 81.31 | 1022 | Rams |
| 40 | 11 | Jammal Brown | 77.29 | 66.10 | 80.59 | 833 | Commanders |
| 41 | 12 | Sam Baker | 77.01 | 67.20 | 79.39 | 1211 | Falcons |
| 42 | 13 | Corey Hilliard | 76.97 | 68.30 | 78.58 | 266 | Lions |
| 43 | 14 | Jermon Bushrod | 76.87 | 67.40 | 79.02 | 1185 | Saints |
| 44 | 15 | Dennis Roland | 76.59 | 64.30 | 80.61 | 645 | Bengals |
| 45 | 16 | Phil Loadholt | 75.90 | 63.30 | 80.13 | 1030 | Vikings |
| 46 | 17 | Bryan Bulaga | 75.77 | 63.70 | 79.65 | 1088 | Packers |
| 47 | 18 | Stephon Heyer | 74.90 | 61.20 | 79.87 | 445 | Commanders |
| 48 | 19 | Frank Omiyale | 74.67 | 61.40 | 79.35 | 1123 | Bears |
| 49 | 20 | Jared Veldheer | 74.35 | 60.50 | 79.41 | 882 | Raiders |
| 50 | 21 | Trent Williams | 74.34 | 63.40 | 77.46 | 870 | Commanders |
| 51 | 22 | Eugene Monroe | 74.19 | 61.90 | 78.22 | 988 | Jaguars |

### Starter (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 52 | 1 | Jeremy Trueblood | 73.47 | 58.60 | 79.21 | 496 | Buccaneers |
| 53 | 2 | Max Starks | 73.45 | 62.20 | 76.79 | 341 | Steelers |
| 54 | 3 | Jonathan Scott | 73.42 | 62.20 | 76.74 | 938 | Steelers |
| 55 | 4 | Anthony Davis | 73.27 | 59.70 | 78.15 | 979 | 49ers |
| 56 | 5 | Lydon Murtha | 72.76 | 60.50 | 76.77 | 230 | Dolphins |
| 57 | 6 | Tony Pashos | 71.95 | 58.90 | 76.48 | 242 | Browns |
| 58 | 7 | Anthony Collins | 71.51 | 67.90 | 69.75 | 255 | Bengals |
| 59 | 8 | Marc Colombo | 71.45 | 56.50 | 77.25 | 998 | Cowboys |
| 60 | 9 | Rashad Butler | 70.37 | 56.00 | 75.79 | 246 | Texans |
| 61 | 10 | J'Marcus Webb | 70.05 | 53.10 | 77.19 | 921 | Bears |
| 62 | 11 | Wayne Hunter | 69.88 | 55.00 | 75.63 | 511 | Jets |
| 63 | 12 | David Diehl | 69.44 | 56.60 | 73.84 | 780 | Giants |
| 64 | 13 | Levi Brown | 69.18 | 55.30 | 74.26 | 960 | Cardinals |
| 65 | 14 | Andre Smith | 67.99 | 50.60 | 75.42 | 281 | Bengals |
| 66 | 15 | Jeff Linkenbach | 67.53 | 59.50 | 68.71 | 367 | Colts |
| 67 | 16 | Will Beatty | 65.22 | 56.60 | 66.80 | 163 | Giants |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Antonio Gates | 86.09 | 84.90 | 82.72 | 354 | Chargers |
| 2 | 2 | Jermichael Finley | 86.01 | 81.30 | 84.98 | 129 | Packers |
| 3 | 3 | Marcedes Lewis | 85.26 | 89.90 | 78.00 | 450 | Jaguars |
| 4 | 4 | Jason Witten | 84.74 | 90.30 | 76.86 | 601 | Cowboys |
| 5 | 5 | Rob Gronkowski | 82.63 | 86.20 | 76.08 | 422 | Patriots |
| 6 | 6 | Vernon Davis | 81.64 | 79.90 | 78.63 | 551 | 49ers |
| 7 | 7 | Anthony Fasano | 80.43 | 81.30 | 75.68 | 500 | Dolphins |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Tony Moeaki | 79.86 | 83.00 | 73.60 | 471 | Chiefs |
| 9 | 2 | Todd Heap | 79.69 | 78.70 | 76.18 | 432 | Ravens |
| 10 | 3 | Jimmy Graham | 79.68 | 76.30 | 77.76 | 167 | Saints |
| 11 | 4 | Heath Miller | 79.65 | 80.80 | 74.72 | 465 | Steelers |
| 12 | 5 | Joel Dreessen | 78.58 | 74.50 | 77.14 | 406 | Texans |
| 13 | 6 | Zach Miller | 78.23 | 82.80 | 71.02 | 641 | Raiders |
| 14 | 7 | Jim Kleinsasser | 78.17 | 79.90 | 72.85 | 226 | Vikings |
| 15 | 8 | Benjamin Watson | 77.39 | 76.50 | 73.82 | 523 | Browns |
| 16 | 9 | Jacob Tamme | 76.20 | 79.50 | 69.84 | 416 | Colts |
| 17 | 10 | Owen Daniels | 75.58 | 69.70 | 75.34 | 352 | Texans |
| 18 | 11 | Kevin Boss | 75.52 | 70.10 | 74.96 | 495 | Giants |
| 19 | 12 | Kellen Winslow | 75.41 | 74.70 | 71.71 | 501 | Buccaneers |
| 20 | 13 | Aaron Hernandez | 75.28 | 68.90 | 75.37 | 315 | Patriots |
| 21 | 14 | Martellus Bennett | 74.85 | 78.70 | 68.11 | 232 | Cowboys |
| 22 | 15 | Dallas Clark | 74.72 | 71.70 | 72.56 | 260 | Colts |
| 23 | 16 | Fred Davis | 74.71 | 65.10 | 76.95 | 206 | Commanders |
| 24 | 17 | Jared Cook | 74.17 | 71.50 | 71.78 | 164 | Titans |
| 25 | 18 | Randy McMichael | 74.02 | 77.00 | 67.87 | 297 | Chargers |

### Starter (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Dustin Keller | 73.88 | 66.90 | 74.36 | 556 | Jets |
| 27 | 2 | Greg Olsen | 73.25 | 69.50 | 71.59 | 469 | Bears |
| 28 | 3 | Cameron Morrah | 72.46 | 61.60 | 75.54 | 149 | Seahawks |
| 29 | 4 | Justin Peelle | 72.30 | 81.90 | 61.74 | 154 | Falcons |
| 30 | 5 | Delanie Walker | 71.62 | 69.40 | 68.94 | 196 | 49ers |
| 31 | 6 | Craig Stevens | 71.46 | 72.20 | 66.80 | 210 | Titans |
| 32 | 7 | Brent Celek | 71.31 | 64.50 | 71.68 | 567 | Eagles |
| 33 | 8 | Tony Gonzalez | 71.16 | 68.70 | 68.63 | 638 | Falcons |
| 34 | 9 | Brandon Pettigrew | 70.82 | 65.00 | 70.54 | 563 | Lions |
| 35 | 10 | Jeremy Shockey | 70.76 | 67.20 | 68.97 | 372 | Saints |
| 36 | 11 | Chris Baker | 69.62 | 57.90 | 73.27 | 209 | Seahawks |
| 37 | 12 | Chris Cooley | 69.62 | 67.60 | 66.80 | 629 | Commanders |
| 38 | 13 | Dan Gronkowski | 69.53 | 67.60 | 66.65 | 141 | Broncos |
| 39 | 14 | Tony Scheffler | 69.28 | 67.70 | 66.17 | 302 | Lions |
| 40 | 15 | Andrew Quarless | 68.50 | 57.70 | 71.53 | 271 | Packers |
| 41 | 16 | Visanthe Shiancoe | 68.44 | 57.80 | 71.36 | 427 | Vikings |
| 42 | 17 | Ben Patrick | 67.77 | 66.00 | 64.78 | 120 | Cardinals |
| 43 | 18 | Jermaine Gresham | 67.76 | 64.60 | 65.70 | 401 | Bengals |
| 44 | 19 | Tom Crabtree | 66.63 | 58.00 | 68.22 | 122 | Packers |
| 45 | 20 | Billy Bajema | 66.11 | 56.20 | 68.55 | 170 | Rams |
| 46 | 21 | Brandon Myers | 66.10 | 59.00 | 66.66 | 115 | Raiders |
| 47 | 22 | Daniel Graham | 65.86 | 58.40 | 66.66 | 511 | Broncos |
| 48 | 23 | David Martin | 65.82 | 64.60 | 62.46 | 140 | Bills |
| 49 | 24 | Jeff King | 65.75 | 65.60 | 61.69 | 279 | Panthers |
| 50 | 25 | Jonathan Stupar | 65.64 | 60.40 | 64.97 | 131 | Bills |
| 51 | 26 | Dante Rosario | 65.29 | 57.60 | 66.25 | 392 | Panthers |
| 52 | 27 | John Carlson | 65.26 | 58.50 | 65.60 | 495 | Seahawks |
| 53 | 28 | Ed Dickson | 65.21 | 54.00 | 68.52 | 187 | Ravens |
| 54 | 29 | Bo Scaife | 64.99 | 56.70 | 66.35 | 258 | Titans |
| 55 | 30 | Jim Dray | 64.58 | 51.80 | 68.94 | 106 | Cardinals |
| 56 | 31 | Reggie Kelly | 64.50 | 63.70 | 60.86 | 159 | Bengals |
| 57 | 32 | Ben Hartsock | 64.43 | 63.00 | 61.21 | 145 | Jets |
| 58 | 33 | Matt Spaeth | 64.24 | 62.00 | 61.57 | 206 | Steelers |
| 59 | 34 | Leonard Pope | 64.00 | 60.30 | 62.30 | 232 | Chiefs |
| 60 | 35 | Travis Beckum | 62.76 | 58.80 | 61.23 | 145 | Giants |
| 61 | 36 | Donald Lee | 62.11 | 56.60 | 61.62 | 124 | Packers |

### Rotation/backup (0 players)

_None._

## WR — Wide Receiver

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Brandon Lloyd | 89.22 | 90.70 | 84.06 | 610 | Broncos |
| 2 | 2 | Kenny Britt | 87.40 | 86.00 | 84.16 | 277 | Titans |
| 3 | 3 | Patrick Crayton | 86.38 | 79.30 | 86.93 | 209 | Chargers |
| 4 | 4 | Arrelious Benn | 85.31 | 81.00 | 84.01 | 177 | Buccaneers |
| 5 | 5 | Percy Harvin | 85.02 | 87.60 | 79.14 | 379 | Vikings |
| 6 | 6 | Andre Johnson | 84.68 | 84.90 | 80.37 | 524 | Texans |
| 7 | 7 | Malcom Floyd | 84.13 | 80.30 | 82.51 | 353 | Chargers |
| 8 | 8 | Vincent Jackson | 83.91 | 81.40 | 81.41 | 117 | Chargers |
| 9 | 9 | Demaryius Thomas | 83.10 | 79.20 | 81.54 | 104 | Broncos |
| 10 | 10 | Brandon Stokley | 82.21 | 82.30 | 77.99 | 239 | Seahawks |
| 11 | 11 | Mike Wallace | 82.13 | 76.90 | 81.45 | 666 | Steelers |
| 12 | 12 | Calvin Johnson | 82.06 | 83.20 | 77.13 | 652 | Lions |
| 13 | 13 | Dwayne Bowe | 82.00 | 78.70 | 80.03 | 547 | Chiefs |
| 14 | 14 | Hakeem Nicks | 81.75 | 80.60 | 78.35 | 478 | Giants |
| 15 | 15 | Mario Manningham | 81.41 | 73.40 | 82.58 | 444 | Giants |
| 16 | 16 | Roddy White | 81.29 | 82.80 | 76.11 | 651 | Falcons |
| 17 | 17 | Greg Jennings | 81.08 | 75.00 | 80.97 | 758 | Packers |

### Good (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Seyi Ajirotutu | 79.66 | 67.30 | 83.74 | 157 | Chargers |
| 19 | 2 | Earl Bennett | 79.12 | 75.10 | 77.64 | 375 | Bears |
| 20 | 3 | Braylon Edwards | 79.05 | 70.00 | 80.91 | 604 | Jets |
| 21 | 4 | Santonio Holmes | 79.04 | 77.30 | 76.03 | 462 | Jets |
| 22 | 5 | Larry Fitzgerald | 79.00 | 78.40 | 75.23 | 636 | Cardinals |
| 23 | 6 | Austin Collie | 78.93 | 77.40 | 75.78 | 293 | Colts |
| 24 | 7 | Johnny Knox | 78.80 | 71.40 | 79.57 | 640 | Bears |
| 25 | 8 | Danario Alexander | 78.66 | 69.80 | 80.40 | 145 | Rams |
| 26 | 9 | Deion Branch | 78.35 | 75.50 | 76.08 | 521 | Patriots |
| 27 | 10 | Santana Moss | 77.37 | 76.10 | 74.05 | 660 | Commanders |
| 28 | 11 | Robert Meachem | 77.29 | 73.20 | 75.85 | 344 | Saints |
| 29 | 12 | Steve Johnson | 77.26 | 76.10 | 73.86 | 583 | Bills |
| 30 | 13 | Lance Moore | 77.20 | 75.80 | 73.96 | 513 | Saints |
| 31 | 14 | Reggie Wayne | 77.14 | 75.70 | 73.93 | 738 | Colts |
| 32 | 15 | Mike Thomas | 76.96 | 73.90 | 74.83 | 490 | Jaguars |
| 33 | 16 | Mike Williams | 76.77 | 74.70 | 73.99 | 1042 | Buccaneers |
| 34 | 17 | Davone Bess | 76.64 | 77.90 | 71.63 | 461 | Dolphins |
| 35 | 18 | Damian Williams | 76.63 | 69.70 | 77.09 | 143 | Titans |
| 36 | 19 | Jordy Nelson | 76.47 | 74.70 | 73.48 | 432 | Packers |
| 37 | 20 | Dez Bryant | 76.10 | 72.90 | 74.07 | 331 | Cowboys |
| 38 | 21 | Derrick Mason | 75.74 | 74.80 | 72.20 | 571 | Ravens |
| 39 | 22 | Terrell Owens | 75.72 | 69.90 | 75.43 | 554 | Bengals |
| 40 | 23 | Legedu Naanee | 75.68 | 67.60 | 76.90 | 313 | Chargers |
| 41 | 24 | Emmanuel Sanders | 75.59 | 74.70 | 72.02 | 350 | Steelers |
| 42 | 25 | Brian Hartline | 75.56 | 69.80 | 75.24 | 404 | Dolphins |
| 43 | 26 | Michael Jenkins | 75.47 | 73.70 | 72.48 | 425 | Falcons |
| 44 | 27 | Marques Colston | 75.45 | 74.40 | 71.98 | 641 | Saints |
| 45 | 28 | Anquan Boldin | 75.45 | 71.70 | 73.79 | 629 | Ravens |
| 46 | 29 | Anthony Armstrong | 75.27 | 65.40 | 77.68 | 547 | Commanders |
| 47 | 30 | Hines Ward | 75.22 | 74.70 | 71.40 | 599 | Steelers |
| 48 | 31 | Sidney Rice | 75.17 | 68.40 | 75.51 | 181 | Vikings |
| 49 | 32 | Miles Austin | 75.11 | 68.10 | 75.61 | 628 | Cowboys |
| 50 | 33 | Eddie Royal | 74.86 | 70.60 | 73.54 | 450 | Broncos |
| 51 | 34 | Nate Washington | 74.83 | 68.20 | 75.09 | 515 | Titans |
| 52 | 35 | Jeremy Maclin | 74.79 | 70.20 | 73.68 | 700 | Eagles |
| 53 | 36 | Jacoby Ford | 74.68 | 66.50 | 75.96 | 326 | Raiders |
| 54 | 37 | Mike Sims-Walker | 74.41 | 73.10 | 71.11 | 408 | Jaguars |
| 55 | 38 | DeSean Jackson | 74.38 | 61.50 | 78.80 | 593 | Eagles |
| 56 | 39 | Jabar Gaffney | 74.37 | 68.70 | 73.98 | 573 | Broncos |
| 57 | 40 | Wes Welker | 74.30 | 76.40 | 68.73 | 494 | Patriots |
| 58 | 41 | Andre Caldwell | 74.10 | 66.00 | 75.33 | 180 | Bengals |

### Starter (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | Josh Morgan | 73.74 | 64.30 | 75.86 | 527 | 49ers |
| 60 | 2 | Jordan Shipley | 73.61 | 66.90 | 73.92 | 448 | Bengals |
| 61 | 3 | Joshua Cribbs | 73.57 | 62.00 | 77.12 | 175 | Browns |
| 62 | 4 | Mark Clayton | 72.93 | 68.80 | 71.51 | 170 | Rams |
| 63 | 5 | Lee Evans | 72.78 | 64.40 | 74.20 | 487 | Bills |
| 64 | 6 | Kevin Walter | 72.78 | 67.80 | 71.94 | 525 | Texans |
| 65 | 7 | Chad Johnson | 72.76 | 70.20 | 70.30 | 573 | Bengals |
| 66 | 8 | James Jones | 72.73 | 64.40 | 74.12 | 531 | Packers |
| 67 | 9 | Louis Murphy Jr. | 72.60 | 60.80 | 76.30 | 459 | Raiders |
| 68 | 10 | David Gettis | 72.56 | 64.40 | 73.83 | 446 | Panthers |
| 69 | 11 | Brandon Tate | 72.31 | 59.20 | 76.89 | 341 | Patriots |
| 70 | 12 | Steve Breaston | 72.19 | 65.50 | 72.49 | 493 | Cardinals |
| 71 | 13 | Golden Tate | 72.00 | 64.50 | 72.83 | 160 | Seahawks |
| 72 | 14 | Michael Crabtree | 71.95 | 66.20 | 71.61 | 565 | 49ers |
| 73 | 15 | Jason Avant | 71.95 | 70.30 | 68.88 | 538 | Eagles |
| 74 | 16 | Danny Amendola | 71.89 | 72.10 | 67.59 | 454 | Rams |
| 75 | 17 | Steve Smith | 71.64 | 71.40 | 67.64 | 786 | Panthers |
| 76 | 18 | Brandon Gibson | 71.26 | 66.80 | 70.07 | 440 | Rams |
| 77 | 19 | David Nelson | 71.07 | 68.50 | 68.62 | 291 | Bills |
| 78 | 20 | Blair White | 70.95 | 69.10 | 68.01 | 365 | Colts |
| 79 | 21 | Pierre Garcon | 70.88 | 67.10 | 69.24 | 640 | Colts |
| 80 | 22 | Roy Williams | 70.82 | 61.80 | 72.67 | 479 | Cowboys |
| 81 | 23 | Nate Burleson | 70.62 | 65.80 | 69.67 | 550 | Lions |
| 82 | 24 | Devin Aromashodu | 70.18 | 65.70 | 69.00 | 134 | Bears |
| 83 | 25 | Mohamed Massaquoi | 69.85 | 62.30 | 70.72 | 456 | Browns |
| 84 | 26 | Roscoe Parrish | 69.79 | 64.90 | 68.89 | 304 | Bills |
| 85 | 27 | Antwaan Randle El | 69.49 | 62.80 | 69.79 | 277 | Steelers |
| 86 | 28 | Jacoby Jones | 69.47 | 64.10 | 68.88 | 391 | Texans |
| 87 | 29 | Brandon LaFell | 69.41 | 64.60 | 68.45 | 353 | Panthers |
| 88 | 30 | Jerricho Cotchery | 68.30 | 64.80 | 66.46 | 468 | Jets |
| 89 | 31 | Donald Jones | 68.10 | 61.30 | 68.46 | 184 | Bills |
| 90 | 32 | Donald Driver | 67.98 | 61.70 | 68.00 | 606 | Packers |
| 91 | 33 | Derek Hagan | 67.87 | 68.10 | 63.55 | 172 | Giants |
| 92 | 34 | Kassim Osgood | 67.81 | 63.10 | 66.78 | 131 | Jaguars |
| 93 | 35 | Devery Henderson | 67.63 | 60.40 | 68.29 | 473 | Saints |
| 94 | 36 | Sammie Stroughter | 67.33 | 60.40 | 67.78 | 306 | Buccaneers |
| 95 | 37 | Roydell Williams | 67.19 | 49.50 | 74.81 | 222 | Commanders |
| 96 | 38 | Brian Finneran | 67.19 | 66.50 | 63.48 | 186 | Falcons |
| 97 | 39 | Chansi Stuckey | 67.17 | 61.80 | 66.58 | 297 | Browns |
| 98 | 40 | Andre Roberts | 66.97 | 55.90 | 70.18 | 308 | Cardinals |
| 99 | 41 | Marlon Moore | 66.86 | 53.50 | 71.60 | 140 | Dolphins |
| 100 | 42 | Mardy Gilyard | 66.81 | 59.50 | 67.51 | 110 | Rams |
| 101 | 43 | Riley Cooper | 66.71 | 56.20 | 69.55 | 180 | Eagles |
| 102 | 44 | Devin Hester | 66.48 | 58.30 | 67.76 | 534 | Bears |
| 103 | 45 | Deon Butler | 65.70 | 60.40 | 65.07 | 355 | Seahawks |
| 104 | 46 | David Anderson | 65.37 | 57.30 | 66.59 | 157 | Texans |
| 105 | 47 | Joey Galloway | 65.21 | 54.60 | 68.11 | 264 | Commanders |
| 106 | 48 | Laurent Robinson | 65.17 | 58.00 | 65.78 | 467 | Rams |
| 107 | 49 | Harry Douglas | 65.06 | 54.20 | 68.13 | 455 | Falcons |
| 108 | 50 | Terrance Copper | 64.85 | 61.40 | 62.98 | 172 | Chiefs |
| 109 | 51 | Brian Robiskie | 64.82 | 59.60 | 64.14 | 362 | Browns |
| 110 | 52 | Sam Hurd | 64.19 | 61.20 | 62.02 | 158 | Cowboys |
| 111 | 53 | Max Komar | 63.99 | 51.40 | 68.21 | 146 | Cardinals |
| 112 | 54 | Darrius Heyward-Bey | 63.69 | 52.80 | 66.78 | 421 | Raiders |
| 113 | 55 | Early Doucet | 63.18 | 53.40 | 65.53 | 261 | Cardinals |
| 114 | 56 | Brad Smith | 62.62 | 57.80 | 61.66 | 123 | Jets |

### Rotation/backup (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 115 | 1 | Bernard Berrian | 61.93 | 54.40 | 62.79 | 369 | Vikings |
| 116 | 2 | Johnnie Lee Higgins | 61.19 | 51.00 | 63.81 | 253 | Raiders |
| 117 | 3 | Tiquan Underwood | 60.27 | 48.90 | 63.68 | 208 | Jaguars |
| 118 | 4 | Ted Ginn Jr. | 59.38 | 46.00 | 64.13 | 191 | 49ers |
| 119 | 5 | Stephen Williams | 58.63 | 47.60 | 61.81 | 149 | Cardinals |
| 120 | 6 | Bryant Johnson | 58.46 | 48.80 | 60.73 | 493 | Lions |
| 121 | 7 | Derrick Williams | 55.06 | 45.50 | 57.27 | 125 | Lions |
