# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:38:35Z
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
| 5 | 5 | David Baas | 86.25 | 78.36 | 87.35 | 939 | 49ers |
| 6 | 6 | Kyle Cook | 86.07 | 78.50 | 86.95 | 1094 | Bengals |
| 7 | 7 | Ryan Kalil | 86.05 | 78.50 | 86.91 | 989 | Panthers |
| 8 | 8 | Matt Birk | 86.05 | 81.00 | 85.25 | 1186 | Ravens |
| 9 | 9 | Alex Mack | 84.74 | 75.98 | 86.41 | 958 | Browns |
| 10 | 10 | Brad Meester | 84.53 | 77.00 | 85.38 | 1064 | Jaguars |
| 11 | 11 | Andre Gurode | 83.81 | 74.00 | 86.19 | 1073 | Cowboys |
| 12 | 12 | Dan Koppen | 82.98 | 74.40 | 84.54 | 1080 | Patriots |
| 13 | 13 | Todd McClure | 82.85 | 74.90 | 83.99 | 1193 | Falcons |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Samson Satele | 79.22 | 69.47 | 81.55 | 979 | Raiders |
| 15 | 2 | Jonathan Goodwin | 78.26 | 68.60 | 80.53 | 1178 | Saints |
| 16 | 3 | Nick Hardwick | 77.51 | 67.80 | 79.81 | 1050 | Chargers |
| 17 | 4 | Maurkice Pouncey | 77.36 | 67.70 | 79.64 | 1073 | Steelers |
| 18 | 5 | John Sullivan | 77.28 | 67.44 | 79.67 | 792 | Vikings |
| 19 | 6 | Dominic Raiola | 76.28 | 64.70 | 79.83 | 1107 | Lions |
| 20 | 7 | J.D. Walton | 75.63 | 65.10 | 78.48 | 1061 | Broncos |

### Starter (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Jeff Faine | 73.73 | 61.66 | 77.61 | 432 | Buccaneers |
| 22 | 2 | Jeremy Zuttah | 73.38 | 64.71 | 74.99 | 721 | Buccaneers |
| 23 | 3 | Lyle Sendlein | 72.25 | 63.95 | 73.61 | 959 | Cardinals |
| 24 | 4 | Eugene Amano | 66.85 | 54.58 | 70.86 | 706 | Titans |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Champ Bailey | 91.39 | 85.80 | 91.98 | 966 | Broncos |
| 2 | 2 | Joe Haden | 90.71 | 83.70 | 91.22 | 791 | Browns |
| 3 | 3 | Josh Wilson | 89.85 | 84.18 | 90.49 | 652 | Ravens |
| 4 | 4 | Antoine Winfield | 88.64 | 88.90 | 84.30 | 981 | Vikings |
| 5 | 5 | Darrelle Revis | 87.71 | 81.30 | 87.82 | 1013 | Jets |
| 6 | 6 | Tramon Williams | 87.39 | 83.00 | 86.15 | 1232 | Packers |
| 7 | 7 | Brandon Flowers | 86.35 | 80.60 | 86.01 | 1036 | Chiefs |
| 8 | 8 | Brent Grimes | 85.59 | 81.20 | 84.35 | 1035 | Falcons |
| 9 | 9 | Asante Samuel | 85.18 | 80.12 | 88.55 | 729 | Eagles |
| 10 | 10 | Charles Tillman | 83.69 | 77.60 | 83.59 | 1165 | Bears |
| 11 | 11 | Nnamdi Asomugha | 83.10 | 77.84 | 84.53 | 767 | Raiders |
| 12 | 12 | Alterraun Verner | 82.32 | 78.30 | 80.84 | 997 | Titans |
| 13 | 13 | Jason McCourty | 81.25 | 73.31 | 86.55 | 483 | Titans |
| 14 | 14 | Brandon Carr | 80.92 | 73.00 | 82.03 | 1133 | Chiefs |
| 15 | 15 | Antoine Cason | 80.91 | 72.50 | 82.35 | 918 | Chargers |
| 16 | 16 | Joselio Hanson | 80.61 | 72.11 | 82.11 | 694 | Eagles |
| 17 | 17 | Vontae Davis | 80.44 | 75.40 | 79.63 | 1001 | Dolphins |
| 18 | 18 | Chris Carr | 80.34 | 78.90 | 77.14 | 1131 | Ravens |
| 19 | 19 | Leon Hall | 80.20 | 75.00 | 79.50 | 966 | Bengals |

### Good (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Aqib Talib | 79.91 | 72.90 | 85.62 | 635 | Buccaneers |
| 21 | 2 | DeAngelo Hall | 79.83 | 72.10 | 80.82 | 1079 | Commanders |
| 22 | 3 | Jabari Greer | 79.51 | 74.58 | 79.66 | 812 | Saints |
| 23 | 4 | Captain Munnerlyn | 79.48 | 71.29 | 80.77 | 695 | Panthers |
| 24 | 5 | Sean Smith | 78.60 | 70.84 | 80.64 | 746 | Dolphins |
| 25 | 6 | Dunta Robinson | 78.34 | 75.10 | 76.34 | 899 | Falcons |
| 26 | 7 | Roy Lewis | 77.73 | 68.86 | 82.61 | 263 | Seahawks |
| 27 | 8 | Ronde Barber | 77.62 | 69.20 | 79.06 | 987 | Buccaneers |
| 28 | 9 | Nate Clements | 77.58 | 70.50 | 78.13 | 1056 | 49ers |
| 29 | 10 | Sam Shields | 77.09 | 71.05 | 76.95 | 803 | Packers |
| 30 | 11 | Kelly Jennings | 76.81 | 66.50 | 79.52 | 1005 | Seahawks |
| 31 | 12 | Kelvin Hayden | 76.48 | 74.56 | 78.80 | 666 | Colts |
| 32 | 13 | Sheldon Brown | 74.99 | 65.90 | 76.88 | 890 | Browns |
| 33 | 14 | Javier Arenas | 74.90 | 66.86 | 76.09 | 502 | Chiefs |
| 34 | 15 | Syd'Quan Thompson | 74.72 | 66.17 | 81.45 | 211 | Broncos |

### Starter (57 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Perrish Cox | 73.96 | 64.54 | 77.11 | 771 | Broncos |
| 36 | 2 | Johnathan Joseph | 73.84 | 68.66 | 77.30 | 595 | Bengals |
| 37 | 3 | Lardarius Webb | 73.75 | 65.76 | 74.91 | 559 | Ravens |
| 38 | 4 | Phillip Buchanon | 73.61 | 61.77 | 77.34 | 720 | Commanders |
| 39 | 5 | Leodis McKelvin | 73.14 | 64.00 | 75.06 | 851 | Bills |
| 40 | 6 | Corey Webster | 72.93 | 64.40 | 75.48 | 915 | Giants |
| 41 | 7 | Ronald Bartell | 72.91 | 68.40 | 72.78 | 852 | Rams |
| 42 | 8 | Greg Toler | 72.68 | 66.00 | 75.05 | 887 | Cardinals |
| 43 | 9 | Reggie Corner | 72.52 | 63.97 | 74.05 | 462 | Bills |
| 44 | 10 | Terrence McGee | 72.52 | 67.82 | 78.78 | 313 | Bills |
| 45 | 11 | Kevin Barnes | 72.34 | 63.20 | 81.56 | 269 | Commanders |
| 46 | 12 | Shawntae Spencer | 72.27 | 65.20 | 72.82 | 1021 | 49ers |
| 47 | 13 | Tracy Porter | 72.12 | 64.98 | 75.85 | 727 | Saints |
| 48 | 14 | Bradley Fletcher | 72.09 | 60.80 | 75.45 | 978 | Rams |
| 49 | 15 | Drayton Florence | 71.91 | 60.10 | 75.61 | 1063 | Bills |
| 50 | 16 | Tim Jennings | 71.66 | 64.70 | 72.14 | 948 | Bears |
| 51 | 17 | William Middleton | 71.60 | 66.18 | 71.04 | 447 | Jaguars |
| 52 | 18 | Tarell Brown | 71.52 | 65.27 | 74.65 | 310 | 49ers |
| 53 | 19 | Kyle Wilson | 71.40 | 61.47 | 74.89 | 335 | Jets |
| 54 | 20 | E.J. Biggers | 71.07 | 61.53 | 73.26 | 669 | Buccaneers |
| 55 | 21 | D.J. Moore | 70.72 | 62.57 | 71.98 | 607 | Bears |
| 56 | 22 | Orlando Scandrick | 70.18 | 60.85 | 72.24 | 597 | Cowboys |
| 57 | 23 | Kyle Arrington | 70.12 | 60.70 | 72.23 | 899 | Patriots |
| 58 | 24 | Terence Newman | 69.83 | 60.60 | 71.82 | 877 | Cowboys |
| 59 | 25 | Chris Johnson | 69.79 | 60.41 | 76.04 | 376 | Raiders |
| 60 | 26 | Antonio Cromartie | 69.06 | 57.20 | 72.80 | 1087 | Jets |
| 61 | 27 | Stanford Routt | 68.96 | 57.00 | 72.76 | 980 | Raiders |
| 62 | 28 | Nathan Vasher | 68.82 | 64.48 | 70.68 | 338 | Lions |
| 63 | 29 | Jason Allen | 68.72 | 58.71 | 73.31 | 608 | Texans |
| 64 | 30 | Terrell Thomas | 68.70 | 56.70 | 72.54 | 986 | Giants |
| 65 | 31 | Brian Williams | 68.66 | 62.74 | 69.48 | 386 | Falcons |
| 66 | 32 | Chris Gamble | 68.59 | 63.90 | 72.75 | 710 | Panthers |
| 67 | 33 | Carlos Rogers | 68.59 | 62.40 | 72.72 | 760 | Commanders |
| 68 | 34 | Andre' Goodman | 68.38 | 61.80 | 76.93 | 367 | Broncos |
| 69 | 35 | Patrick Robinson | 68.24 | 64.22 | 74.06 | 283 | Saints |
| 70 | 36 | Jarrett Bush | 68.20 | 58.52 | 74.66 | 176 | Packers |
| 71 | 37 | Adam Jones | 68.07 | 65.63 | 81.50 | 165 | Bengals |
| 72 | 38 | Cortland Finnegan | 67.89 | 58.60 | 69.91 | 1173 | Titans |
| 73 | 39 | Morgan Trent | 67.66 | 64.70 | 76.34 | 202 | Bengals |
| 74 | 40 | Marcus Trufant | 67.64 | 56.50 | 70.90 | 1049 | Seahawks |
| 75 | 41 | Jonathan Wilhite | 67.47 | 61.78 | 74.40 | 202 | Patriots |
| 76 | 42 | Jeremy Ware | 66.62 | 62.16 | 78.84 | 114 | Raiders |
| 77 | 43 | Rashean Mathis | 66.31 | 54.40 | 70.09 | 964 | Jaguars |
| 78 | 44 | Bryan McCann | 65.63 | 61.94 | 71.23 | 146 | Cowboys |
| 79 | 45 | William Gay | 65.47 | 55.12 | 68.20 | 726 | Steelers |
| 80 | 46 | Aaron Ross | 65.00 | 56.53 | 67.52 | 398 | Giants |
| 81 | 47 | Trevard Lindley | 64.99 | 63.59 | 70.09 | 184 | Eagles |
| 82 | 48 | Bryant McFadden | 64.71 | 54.00 | 67.69 | 961 | Steelers |
| 83 | 49 | Dominique Rodgers-Cromartie | 64.55 | 51.50 | 69.08 | 1103 | Cardinals |
| 84 | 50 | Dante Hughes | 64.32 | 63.79 | 68.84 | 200 | Chargers |
| 85 | 51 | Alphonso Smith | 63.92 | 55.33 | 69.65 | 573 | Lions |
| 86 | 52 | Drew Coleman | 63.25 | 56.24 | 63.76 | 576 | Jets |
| 87 | 53 | Cedric Griffin | 63.22 | 66.24 | 78.54 | 132 | Vikings |
| 88 | 54 | Richard Marshall | 62.97 | 53.10 | 65.39 | 1086 | Panthers |
| 89 | 55 | Jerome Murphy | 62.81 | 56.43 | 70.19 | 177 | Rams |
| 90 | 56 | Kareem Jackson | 62.58 | 48.90 | 67.53 | 874 | Texans |
| 91 | 57 | Derek Cox | 62.27 | 49.53 | 69.73 | 672 | Jaguars |

### Rotation/backup (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 92 | 1 | Justin King | 61.98 | 62.18 | 66.02 | 186 | Rams |
| 93 | 2 | Mike Jenkins | 61.53 | 53.70 | 62.58 | 914 | Cowboys |
| 94 | 3 | David Jones | 61.00 | 53.97 | 69.86 | 288 | Jaguars |
| 95 | 4 | Asher Allen | 60.54 | 52.23 | 65.04 | 690 | Vikings |
| 96 | 5 | Jerraud Powers | 60.01 | 52.84 | 66.88 | 586 | Colts |
| 97 | 6 | Myron Lewis | 59.98 | 53.13 | 67.68 | 196 | Buccaneers |
| 98 | 7 | Jacob Lacey | 59.83 | 53.23 | 63.20 | 622 | Colts |
| 99 | 8 | Fabian Washington | 58.82 | 56.85 | 60.13 | 511 | Ravens |
| 100 | 9 | Dimitri Patterson | 58.32 | 41.12 | 66.65 | 636 | Eagles |
| 101 | 10 | Zackary Bowman | 56.77 | 52.01 | 60.98 | 231 | Bears |
| 102 | 11 | Ellis Hobbs | 56.36 | 51.27 | 66.46 | 452 | Eagles |
| 103 | 12 | Chris Owens | 53.86 | 42.61 | 63.44 | 307 | Falcons |
| 104 | 13 | Brice McCain | 52.49 | 51.89 | 53.93 | 390 | Texans |
| 105 | 14 | Eric Wright | 51.13 | 40.36 | 57.27 | 708 | Browns |
| 106 | 15 | Chris Cook | 45.80 | 47.04 | 54.22 | 229 | Vikings |

## DI — Defensive Interior

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Justin Smith | 89.93 | 85.29 | 88.86 | 799 | 49ers |
| 2 | 2 | Kyle Williams | 89.46 | 86.97 | 86.96 | 897 | Bills |
| 3 | 3 | Antonio Garay | 89.22 | 78.54 | 92.18 | 457 | Chargers |
| 4 | 4 | Haloti Ngata | 85.31 | 84.95 | 81.38 | 891 | Ravens |
| 5 | 5 | Jason Jones | 85.10 | 78.61 | 86.30 | 653 | Titans |
| 6 | 6 | Ndamukong Suh | 82.48 | 71.45 | 85.67 | 959 | Lions |
| 7 | 7 | Richard Seymour | 82.30 | 70.87 | 88.88 | 622 | Raiders |
| 8 | 8 | Sammie Lee Hill | 82.07 | 70.40 | 86.71 | 349 | Lions |
| 9 | 9 | Paul Soliai | 80.99 | 72.90 | 82.21 | 546 | Dolphins |
| 10 | 10 | Albert Haynesworth | 80.89 | 73.53 | 89.96 | 204 | Commanders |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Geno Atkins | 79.92 | 74.12 | 79.62 | 349 | Bengals |
| 12 | 2 | Desmond Bryant | 78.94 | 68.08 | 83.04 | 325 | Raiders |
| 13 | 3 | Calais Campbell | 78.10 | 63.43 | 84.75 | 771 | Cardinals |
| 14 | 4 | Antonio Dixon | 77.90 | 69.86 | 79.09 | 439 | Eagles |
| 15 | 5 | Ricky Jean Francois | 77.79 | 70.15 | 78.71 | 144 | 49ers |
| 16 | 6 | B.J. Raji | 76.72 | 77.51 | 72.03 | 1069 | Packers |
| 17 | 7 | Barry Cofield | 76.42 | 76.02 | 72.52 | 768 | Giants |
| 18 | 8 | Kevin Williams | 76.03 | 75.72 | 72.07 | 895 | Vikings |
| 19 | 9 | Letroy Guion | 75.97 | 68.05 | 78.12 | 262 | Vikings |
| 20 | 10 | Wallace Gilberry | 75.78 | 60.47 | 81.82 | 519 | Chiefs |
| 21 | 11 | Vince Wilfork | 75.67 | 67.06 | 77.24 | 818 | Patriots |
| 22 | 12 | Aubrayo Franklin | 74.30 | 71.40 | 72.06 | 566 | 49ers |

### Starter (70 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Dan Williams | 73.83 | 75.53 | 69.57 | 382 | Cardinals |
| 24 | 2 | Peria Jerry | 73.61 | 59.40 | 78.92 | 211 | Falcons |
| 25 | 3 | Glenn Dorsey | 73.54 | 66.31 | 74.19 | 989 | Chiefs |
| 26 | 4 | Tony McDaniel | 73.47 | 62.53 | 77.63 | 492 | Dolphins |
| 27 | 5 | Red Bryant | 73.17 | 74.01 | 79.31 | 283 | Seahawks |
| 28 | 6 | Mike Devito | 73.11 | 64.41 | 74.74 | 684 | Jets |
| 29 | 7 | Sione Pouha | 73.10 | 68.01 | 72.33 | 668 | Jets |
| 30 | 8 | Gerald McCoy | 72.94 | 80.07 | 67.15 | 669 | Buccaneers |
| 31 | 9 | Shaun Ellis | 72.69 | 61.66 | 75.88 | 810 | Jets |
| 32 | 10 | Cullen Jenkins | 72.51 | 59.73 | 77.90 | 566 | Packers |
| 33 | 11 | Kendall Langford | 72.37 | 68.48 | 70.79 | 729 | Dolphins |
| 34 | 12 | John Henderson | 72.33 | 69.57 | 77.30 | 264 | Raiders |
| 35 | 13 | Shaun Rogers | 72.15 | 59.94 | 77.15 | 395 | Browns |
| 36 | 14 | Stephen Bowen | 72.11 | 64.26 | 73.17 | 542 | Cowboys |
| 37 | 15 | Vonnie Holliday | 71.95 | 58.27 | 77.93 | 410 | Commanders |
| 38 | 16 | Alan Branch | 71.93 | 61.57 | 74.67 | 564 | Cardinals |
| 39 | 17 | Ryan Pickett | 71.75 | 57.59 | 77.03 | 470 | Packers |
| 40 | 18 | C.J. Wilson | 71.69 | 56.97 | 77.33 | 290 | Packers |
| 41 | 19 | Adam Carriker | 71.45 | 58.79 | 75.72 | 591 | Commanders |
| 42 | 20 | Tommy Kelly | 71.39 | 60.36 | 74.58 | 835 | Raiders |
| 43 | 21 | Shaun Smith | 71.21 | 58.23 | 75.69 | 480 | Chiefs |
| 44 | 22 | Brodrick Bunkley | 71.14 | 59.08 | 76.04 | 344 | Eagles |
| 45 | 23 | Pat Sims | 71.08 | 58.00 | 77.71 | 434 | Bengals |
| 46 | 24 | Colin Cole | 70.73 | 60.46 | 76.55 | 525 | Seahawks |
| 47 | 25 | Chris Canty | 70.41 | 69.79 | 66.66 | 611 | Giants |
| 48 | 26 | Rocky Bernard | 69.90 | 59.11 | 75.01 | 321 | Giants |
| 49 | 27 | Andre Fluellen | 69.83 | 61.58 | 71.16 | 151 | Lions |
| 50 | 28 | Brandon Mebane | 69.56 | 63.86 | 71.28 | 717 | Seahawks |
| 51 | 29 | Mike Patterson | 69.04 | 59.08 | 71.51 | 589 | Eagles |
| 52 | 30 | Marcus Thomas | 68.76 | 57.17 | 72.32 | 523 | Broncos |
| 53 | 31 | Cory Redding | 68.48 | 50.40 | 76.36 | 683 | Ravens |
| 54 | 32 | Earl Mitchell | 68.20 | 57.11 | 72.46 | 310 | Texans |
| 55 | 33 | Sedrick Ellis | 67.39 | 60.45 | 67.85 | 817 | Saints |
| 56 | 34 | Shaun Cody | 67.20 | 54.39 | 71.57 | 485 | Texans |
| 57 | 35 | Amobi Okoye | 67.12 | 55.82 | 70.48 | 785 | Texans |
| 58 | 36 | Darnell Dockett | 66.92 | 49.22 | 75.59 | 903 | Cardinals |
| 59 | 37 | Vance Walker | 66.84 | 54.01 | 71.22 | 294 | Falcons |
| 60 | 38 | Corey Williams | 66.77 | 54.34 | 70.89 | 773 | Lions |
| 61 | 39 | Tyson Alualu | 66.74 | 57.69 | 68.60 | 789 | Jaguars |
| 62 | 40 | Antonio Johnson | 66.61 | 56.02 | 71.58 | 308 | Colts |
| 63 | 41 | Tyson Jackson | 66.49 | 60.48 | 69.47 | 330 | Chiefs |
| 64 | 42 | Randy Starks | 66.16 | 58.03 | 67.42 | 734 | Dolphins |
| 65 | 43 | Brandon Deaderick | 65.96 | 60.14 | 71.93 | 253 | Patriots |
| 66 | 44 | Terrance Knighton | 65.87 | 55.37 | 68.70 | 801 | Jaguars |
| 67 | 45 | Tony Hargrove | 65.65 | 58.57 | 68.29 | 438 | Saints |
| 68 | 46 | Jonathan Babineaux | 65.64 | 54.77 | 68.72 | 799 | Falcons |
| 69 | 47 | Jason Hatcher | 65.61 | 57.40 | 70.05 | 257 | Cowboys |
| 70 | 48 | Brett Keisel | 65.48 | 57.47 | 68.74 | 609 | Steelers |
| 71 | 49 | Derek Landri | 64.87 | 48.53 | 71.59 | 785 | Panthers |
| 72 | 50 | Ron Brace | 64.60 | 57.01 | 68.63 | 283 | Patriots |
| 73 | 51 | Jacques Cesaire | 64.56 | 44.39 | 73.84 | 607 | Chargers |
| 74 | 52 | Jimmy Kennedy | 64.49 | 55.80 | 73.41 | 147 | Vikings |
| 75 | 53 | Kyle Love | 64.35 | 61.11 | 68.59 | 166 | Patriots |
| 76 | 54 | Josh Price-Brent | 64.23 | 58.96 | 63.58 | 254 | Cowboys |
| 77 | 55 | Al Woods | 64.17 | 55.24 | 73.26 | 130 | Buccaneers |
| 78 | 56 | Ray McDonald | 64.10 | 57.87 | 64.08 | 569 | 49ers |
| 79 | 57 | Roy Miller | 64.01 | 47.92 | 70.57 | 600 | Buccaneers |
| 80 | 58 | Antonio Smith | 64.01 | 52.21 | 67.71 | 902 | Texans |
| 81 | 59 | Domata Peko Sr. | 64.00 | 50.60 | 68.77 | 647 | Bengals |
| 82 | 60 | Ahtyba Rubin | 63.50 | 49.40 | 68.73 | 722 | Browns |
| 83 | 61 | Henry Melton | 63.29 | 49.60 | 68.25 | 390 | Bears |
| 84 | 62 | Trevor Laws | 63.27 | 56.43 | 63.67 | 470 | Eagles |
| 85 | 63 | Ron Edwards | 63.10 | 59.41 | 61.39 | 512 | Chiefs |
| 86 | 64 | Corey Peters | 62.79 | 48.70 | 68.01 | 597 | Falcons |
| 87 | 65 | Andre Neblett | 62.73 | 64.27 | 68.40 | 151 | Panthers |
| 88 | 66 | Anthony Adams | 62.56 | 46.07 | 69.39 | 654 | Bears |
| 89 | 67 | Isaac Sopoaga | 62.48 | 47.89 | 68.04 | 569 | 49ers |
| 90 | 68 | Fili Moala | 62.47 | 47.50 | 68.28 | 527 | Colts |
| 91 | 69 | Tommie Harris | 62.36 | 58.05 | 61.07 | 617 | Bears |
| 92 | 70 | Luis Castillo | 62.13 | 51.93 | 64.77 | 614 | Chargers |

### Rotation/backup (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 93 | 1 | Jay Ratliff | 61.70 | 53.62 | 62.92 | 720 | Cowboys |
| 94 | 2 | Marcus Stroud | 61.66 | 43.62 | 70.55 | 673 | Bills |
| 95 | 3 | Marcus Spears | 61.33 | 58.76 | 67.21 | 259 | Cowboys |
| 96 | 4 | Jermelle Cudjo | 61.00 | 54.23 | 67.60 | 231 | Rams |
| 97 | 5 | Terrence Cody | 60.89 | 56.30 | 60.81 | 141 | Ravens |
| 98 | 6 | Alex Carrington | 60.73 | 62.29 | 62.82 | 214 | Bills |
| 99 | 7 | Igor Olshansky | 60.03 | 53.76 | 60.05 | 569 | Cowboys |
| 100 | 8 | Nick Hayden | 59.95 | 51.25 | 63.67 | 473 | Panthers |
| 101 | 9 | Daniel Muir | 59.84 | 45.74 | 66.11 | 526 | Colts |
| 102 | 10 | Sen'Derrick Marks | 59.36 | 51.69 | 64.47 | 460 | Titans |
| 103 | 11 | Remi Ayodele | 59.32 | 48.68 | 62.24 | 627 | Saints |
| 104 | 12 | Pat Williams | 58.78 | 45.81 | 63.26 | 541 | Vikings |
| 105 | 13 | Vaughn Martin | 58.72 | 57.42 | 61.67 | 181 | Chargers |
| 106 | 14 | Jovan Haye | 58.69 | 48.10 | 63.66 | 470 | Titans |
| 107 | 15 | Ma'ake Kemoeatu | 58.11 | 47.34 | 63.20 | 362 | Commanders |
| 108 | 16 | Brandon McKinney | 57.97 | 60.41 | 55.31 | 225 | Ravens |
| 109 | 17 | Ronald Fields | 57.49 | 53.59 | 55.92 | 298 | Broncos |
| 110 | 18 | Leger Douzable | 57.46 | 55.02 | 55.95 | 217 | Jaguars |
| 111 | 19 | Matt Toeaina | 57.30 | 50.88 | 57.41 | 676 | Bears |
| 112 | 20 | Bryan Robinson | 57.28 | 43.31 | 62.43 | 338 | Cardinals |
| 113 | 21 | Torell Troup | 56.78 | 53.83 | 55.61 | 298 | Bills |
| 114 | 22 | Gary Gibson | 56.50 | 49.39 | 57.07 | 589 | Rams |
| 115 | 23 | Kedric Golston | 56.03 | 48.20 | 60.21 | 445 | Commanders |
| 116 | 24 | Casey Hampton | 56.02 | 51.39 | 54.94 | 503 | Steelers |
| 117 | 25 | Tank Johnson | 55.12 | 57.55 | 60.20 | 223 | Bengals |
| 118 | 26 | Justin Bannan | 54.36 | 44.06 | 57.06 | 774 | Broncos |
| 119 | 27 | Ogemdi Nwagbuo | 53.69 | 55.20 | 49.55 | 258 | Chargers |
| 120 | 28 | Darell Scott | 53.41 | 53.93 | 56.20 | 253 | Rams |
| 121 | 29 | Myron Pryor | 53.37 | 51.57 | 57.70 | 233 | Patriots |
| 122 | 30 | Nick Eason | 52.84 | 42.76 | 55.40 | 529 | Steelers |
| 123 | 31 | Ryan McBean | 52.73 | 46.89 | 52.45 | 445 | Broncos |
| 124 | 32 | Gabe Watson | 52.57 | 53.82 | 58.44 | 142 | Cardinals |
| 125 | 33 | Aaron Smith | 51.82 | 52.96 | 60.31 | 279 | Steelers |
| 126 | 34 | Craig Terrill | 49.34 | 47.89 | 48.23 | 380 | Seahawks |
| 127 | 35 | Brian Price | 47.83 | 55.67 | 54.41 | 110 | Buccaneers |
| 128 | 36 | Anthony Bryant | 46.71 | 55.86 | 49.86 | 165 | Commanders |
| 129 | 37 | Phillip Merling | 45.00 | 56.26 | 48.83 | 112 | Dolphins |

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
| 5 | 5 | Jason Babin | 85.08 | 74.85 | 87.74 | 713 | Titans |
| 6 | 6 | Ray Edwards | 84.01 | 78.64 | 85.51 | 745 | Vikings |
| 7 | 7 | Mario Williams | 83.64 | 85.14 | 81.60 | 764 | Texans |
| 8 | 8 | Justin Tuck | 80.95 | 75.78 | 80.23 | 830 | Giants |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Cliff Avril | 78.69 | 70.65 | 83.01 | 625 | Lions |
| 10 | 2 | Julius Peppers | 78.42 | 85.88 | 69.28 | 1036 | Bears |
| 11 | 3 | Lawrence Jackson | 76.88 | 70.98 | 81.85 | 327 | Lions |
| 12 | 4 | Jarvis Moss | 76.53 | 60.81 | 87.01 | 105 | Raiders |
| 13 | 5 | Matt Shaughnessy | 76.13 | 69.15 | 76.62 | 628 | Raiders |
| 14 | 6 | Juqua Parker | 76.01 | 70.91 | 77.32 | 513 | Eagles |
| 15 | 7 | Manny Lawson | 75.62 | 67.08 | 77.14 | 620 | 49ers |
| 16 | 8 | Chris Long | 75.41 | 71.87 | 73.61 | 962 | Rams |
| 17 | 9 | Jared Allen | 75.02 | 63.75 | 78.37 | 934 | Vikings |
| 18 | 10 | Carlos Dunlap | 74.48 | 63.76 | 81.63 | 281 | Bengals |

### Starter (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Parys Haralson | 71.70 | 61.95 | 75.07 | 677 | 49ers |
| 20 | 2 | Shaun Phillips | 71.66 | 60.00 | 75.26 | 909 | Chargers |
| 21 | 3 | Quentin Moses | 70.81 | 61.75 | 74.77 | 182 | Dolphins |
| 22 | 4 | Matt Roth | 69.36 | 59.32 | 71.89 | 1014 | Browns |
| 23 | 5 | Turk McBride | 69.13 | 61.00 | 71.42 | 448 | Lions |
| 24 | 6 | Greg Hardy | 68.80 | 65.17 | 68.08 | 391 | Panthers |
| 25 | 7 | Jermaine Cunningham | 68.73 | 65.61 | 66.65 | 575 | Patriots |
| 26 | 8 | Jeff Charleston | 68.04 | 60.70 | 68.77 | 267 | Saints |
| 27 | 9 | James Hall | 67.96 | 53.95 | 73.14 | 802 | Rams |
| 28 | 10 | Anthony Spencer | 67.66 | 65.26 | 65.10 | 938 | Cowboys |
| 29 | 11 | Dave Ball | 67.49 | 57.00 | 75.52 | 432 | Titans |
| 30 | 12 | Antwan Applewhite | 67.40 | 56.00 | 71.87 | 606 | Chargers |
| 31 | 13 | Darryl Tapp | 67.18 | 62.34 | 67.27 | 460 | Eagles |
| 32 | 14 | O'Brien Schofield | 66.82 | 59.03 | 75.15 | 133 | Cardinals |
| 33 | 15 | Brandon Graham | 66.79 | 63.32 | 68.07 | 472 | Eagles |
| 34 | 16 | Kroy Biermann | 66.78 | 62.80 | 65.27 | 654 | Falcons |
| 35 | 17 | Mathias Kiwanuka | 66.76 | 60.04 | 87.08 | 155 | Giants |
| 36 | 18 | Mark Anderson | 66.72 | 60.16 | 67.96 | 456 | Texans |
| 37 | 19 | Aaron Kampman | 65.75 | 58.09 | 75.02 | 498 | Jaguars |
| 38 | 20 | Jason Pierre-Paul | 65.70 | 69.57 | 58.95 | 397 | Giants |
| 39 | 21 | Israel Idonije | 64.86 | 56.82 | 66.06 | 957 | Bears |
| 40 | 22 | Jeremy Mincey | 64.71 | 60.78 | 64.20 | 586 | Jaguars |
| 41 | 23 | Jason Hunter | 64.54 | 57.43 | 65.11 | 754 | Broncos |
| 42 | 24 | Derrick Harvey | 62.97 | 58.31 | 62.95 | 350 | Jaguars |
| 43 | 25 | Jamaal Anderson | 62.89 | 65.72 | 56.83 | 431 | Falcons |
| 44 | 26 | William Hayes | 62.10 | 62.67 | 59.64 | 545 | Titans |

### Rotation/backup (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 45 | 1 | Frostee Rucker | 61.38 | 59.69 | 65.64 | 283 | Bengals |
| 46 | 2 | C.J. Ah You | 61.31 | 56.13 | 60.59 | 344 | Rams |
| 47 | 3 | Brian Robison | 61.31 | 58.82 | 58.80 | 323 | Vikings |
| 48 | 4 | Jacob Ford | 60.43 | 57.17 | 60.52 | 568 | Titans |
| 49 | 5 | Chris Kelsay | 60.25 | 51.86 | 61.68 | 1063 | Bills |
| 50 | 6 | Michael Johnson | 60.08 | 59.49 | 56.31 | 666 | Bengals |
| 51 | 7 | Kyle Vanden Bosch | 59.17 | 55.63 | 62.57 | 652 | Lions |
| 52 | 8 | Robert Ayers | 59.13 | 61.99 | 58.26 | 651 | Broncos |
| 53 | 9 | George Selvie | 58.64 | 58.45 | 54.60 | 310 | Rams |
| 54 | 10 | Tim Jamison | 58.40 | 60.14 | 59.32 | 211 | Texans |
| 55 | 11 | Chauncey Davis | 57.84 | 56.90 | 54.30 | 423 | Falcons |
| 56 | 12 | Tim Crowder | 57.51 | 56.09 | 54.29 | 630 | Buccaneers |
| 57 | 13 | Trevor Scott | 57.41 | 58.31 | 58.89 | 476 | Raiders |
| 58 | 14 | Austen Lane | 56.97 | 58.79 | 56.79 | 307 | Jaguars |
| 59 | 15 | Andre Carter | 56.33 | 50.44 | 56.09 | 731 | Commanders |
| 60 | 16 | Jason Taylor | 56.26 | 46.33 | 58.72 | 715 | Jets |
| 61 | 17 | Robert Geathers | 55.84 | 55.33 | 52.01 | 796 | Bengals |
| 62 | 18 | Dave Tollefson | 55.77 | 57.94 | 55.36 | 143 | Giants |
| 63 | 19 | Daniel Te'o-Nesheim | 55.07 | 58.36 | 59.58 | 110 | Eagles |
| 64 | 20 | Kentwan Balmer | 54.97 | 55.46 | 50.47 | 564 | Seahawks |
| 65 | 21 | Michael Bennett | 54.90 | 57.36 | 52.23 | 423 | Buccaneers |
| 66 | 22 | Keyunta Dawson | 53.97 | 55.26 | 48.95 | 597 | Colts |
| 67 | 23 | Jesse Nading | 50.79 | 57.81 | 57.91 | 106 | Texans |
| 68 | 24 | Jay Richardson | 50.02 | 57.13 | 48.42 | 119 | Seahawks |
| 69 | 25 | Kyle Moore | 50.00 | 56.31 | 52.50 | 309 | Buccaneers |
| 70 | 26 | Antwan Odom | 45.00 | 57.56 | 50.17 | 158 | Bengals |

## G — Guard

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Carl Nicks | 94.10 | 87.40 | 94.40 | 1192 | Saints |
| 2 | 2 | Josh Sitton | 92.39 | 90.10 | 89.75 | 1294 | Packers |
| 3 | 3 | Wade Smith | 92.00 | 86.70 | 91.37 | 1067 | Texans |
| 4 | 4 | Logan Mankins | 91.33 | 84.71 | 91.57 | 610 | Patriots |
| 5 | 5 | Mike Brisiel | 90.21 | 79.20 | 93.39 | 502 | Texans |
| 6 | 6 | Chilo Rachal | 89.35 | 84.52 | 88.40 | 761 | 49ers |
| 7 | 7 | Mike Iupati | 89.32 | 81.72 | 90.22 | 951 | 49ers |
| 8 | 8 | Ben Grubbs | 88.76 | 82.70 | 88.64 | 1191 | Ravens |
| 9 | 9 | Todd Herremans | 88.33 | 80.30 | 89.52 | 1060 | Eagles |
| 10 | 10 | Brian Waters | 88.16 | 82.80 | 87.56 | 1083 | Chiefs |
| 11 | 11 | Harvey Dahl | 87.01 | 79.70 | 87.71 | 1211 | Falcons |
| 12 | 12 | Kris Dielman | 86.31 | 79.30 | 86.81 | 1001 | Chargers |
| 13 | 13 | Chris Snee | 86.11 | 77.40 | 87.75 | 1074 | Giants |
| 14 | 14 | Geoff Schwartz | 85.30 | 77.30 | 86.46 | 989 | Panthers |
| 15 | 15 | Richie Incognito | 85.16 | 77.50 | 86.10 | 1072 | Dolphins |
| 16 | 16 | Brandon Moore | 84.71 | 77.60 | 85.28 | 1249 | Jets |
| 17 | 17 | Rich Seubert | 84.55 | 80.80 | 82.88 | 1021 | Giants |
| 18 | 18 | Justin Blalock | 84.47 | 77.70 | 84.81 | 1211 | Falcons |
| 19 | 19 | Bobbie Williams | 84.14 | 75.60 | 85.66 | 1094 | Bengals |
| 20 | 20 | John Greco | 84.02 | 72.44 | 87.58 | 146 | Rams |
| 21 | 21 | Evan Mathis | 83.75 | 73.12 | 86.67 | 110 | Bengals |
| 22 | 22 | Chris Chester | 83.13 | 75.70 | 83.92 | 1006 | Ravens |
| 23 | 23 | Nate Livings | 82.94 | 73.70 | 84.93 | 986 | Bengals |
| 24 | 24 | Jahri Evans | 81.90 | 74.10 | 82.94 | 1192 | Saints |
| 25 | 25 | Steve Hutchinson | 81.83 | 73.04 | 83.52 | 715 | Vikings |
| 26 | 26 | Jacob Bell | 81.73 | 73.70 | 82.91 | 1079 | Rams |
| 27 | 27 | Daryn Colledge | 81.68 | 71.70 | 84.16 | 1228 | Packers |
| 28 | 28 | Antoine Caldwell | 81.51 | 71.86 | 83.78 | 555 | Texans |
| 29 | 29 | Vince Manuwai | 81.44 | 74.10 | 82.16 | 795 | Jaguars |
| 30 | 30 | Louis Vasquez | 80.85 | 71.19 | 83.12 | 534 | Chargers |
| 31 | 31 | Kyle Kosier | 80.79 | 72.26 | 82.31 | 825 | Cowboys |
| 32 | 32 | Mike Pollak | 80.69 | 76.56 | 79.28 | 980 | Colts |
| 33 | 33 | Rob Sims | 80.35 | 72.40 | 81.48 | 1107 | Lions |
| 34 | 34 | Matt Slauson | 80.15 | 71.50 | 81.75 | 1268 | Jets |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Andy Levitre | 79.88 | 69.80 | 82.44 | 927 | Bills |
| 36 | 2 | Dan Connolly | 79.48 | 70.91 | 81.02 | 872 | Patriots |
| 37 | 3 | Lance Louis | 79.28 | 66.72 | 83.49 | 263 | Bears |
| 38 | 4 | Zane Beadles | 79.03 | 69.91 | 80.94 | 929 | Broncos |
| 39 | 5 | Eric Steinbach | 78.39 | 70.55 | 79.45 | 958 | Browns |
| 40 | 6 | Chris Kuper | 78.09 | 68.40 | 80.38 | 987 | Broncos |
| 41 | 7 | Alan Faneca | 77.59 | 68.49 | 79.49 | 960 | Cardinals |
| 42 | 8 | Jon Asamoah | 76.62 | 65.93 | 79.58 | 158 | Chiefs |
| 43 | 9 | Stephen Peterman | 76.45 | 64.30 | 80.38 | 1076 | Lions |
| 44 | 10 | Montrae Holland | 76.40 | 66.44 | 78.88 | 170 | Cowboys |
| 45 | 11 | Justin Smiley | 76.38 | 63.19 | 81.00 | 269 | Jaguars |
| 46 | 12 | Ramon Foster | 76.22 | 65.42 | 79.25 | 832 | Steelers |
| 47 | 13 | Uche Nwaneri | 75.98 | 66.20 | 78.34 | 1061 | Jaguars |
| 48 | 14 | John Jerry | 75.62 | 65.81 | 78.00 | 624 | Dolphins |
| 49 | 15 | Cooper Carlisle | 75.52 | 66.00 | 77.70 | 1084 | Raiders |
| 50 | 16 | Chris Kemoeatu | 75.10 | 63.50 | 78.66 | 1080 | Steelers |
| 51 | 17 | Travelle Wharton | 74.92 | 63.55 | 78.33 | 538 | Panthers |
| 52 | 18 | Russ Hochstein | 74.84 | 63.26 | 78.39 | 347 | Broncos |
| 53 | 19 | Kraig Urbik | 74.32 | 66.39 | 75.44 | 159 | Bills |
| 54 | 20 | Ted Larsen | 74.27 | 61.93 | 78.33 | 634 | Buccaneers |

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 55 | 1 | Adam Snyder | 73.77 | 65.57 | 75.07 | 262 | 49ers |
| 56 | 2 | Chad Rinehart | 73.69 | 68.51 | 72.97 | 202 | Bills |
| 57 | 3 | Jake Scott | 73.08 | 60.69 | 77.17 | 953 | Titans |
| 58 | 4 | Artis Hicks | 72.75 | 59.11 | 77.68 | 547 | Commanders |
| 59 | 5 | Chris Williams | 72.66 | 61.44 | 75.97 | 903 | Bears |
| 60 | 6 | Max Jean-Gilles | 72.60 | 65.77 | 72.98 | 582 | Eagles |
| 61 | 7 | Floyd Womack | 72.50 | 62.56 | 74.96 | 888 | Browns |
| 62 | 8 | Adam Goldberg | 71.14 | 59.90 | 74.46 | 963 | Rams |
| 63 | 9 | Tyronne Green | 70.07 | 58.29 | 73.75 | 598 | Chargers |
| 64 | 10 | Stacy Andrews | 68.50 | 59.47 | 70.36 | 763 | Seahawks |
| 65 | 11 | Kevin Boothe | 66.98 | 58.40 | 68.54 | 347 | Giants |
| 66 | 12 | Jamey Richard | 65.87 | 57.54 | 67.26 | 322 | Colts |
| 67 | 13 | Cord Howard | 65.12 | 51.62 | 69.95 | 343 | Bills |
| 68 | 14 | Davin Joseph | 65.11 | 52.63 | 69.26 | 632 | Buccaneers |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 69 | 1 | Shawn Lauvao | 60.68 | 55.28 | 60.12 | 113 | Browns |

## HB — Running Back

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jamaal Charles | 81.56 | 81.12 | 77.69 | 267 | Chiefs |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 2 | 1 | Adrian Peterson | 78.93 | 82.09 | 72.66 | 264 | Vikings |
| 3 | 2 | Darren McFadden | 76.18 | 66.71 | 78.32 | 202 | Raiders |
| 4 | 3 | Darren Sproles | 75.03 | 68.17 | 75.44 | 275 | Chargers |
| 5 | 4 | Ahmad Bradshaw | 74.97 | 73.78 | 71.60 | 262 | Giants |
| 6 | 5 | Arian Foster | 74.44 | 73.39 | 70.98 | 363 | Texans |
| 7 | 6 | LeSean McCoy | 74.26 | 67.10 | 74.86 | 452 | Eagles |

### Starter (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Peyton Hillis | 73.80 | 72.95 | 70.20 | 316 | Browns |
| 9 | 2 | Chris Johnson | 73.25 | 74.18 | 68.46 | 349 | Titans |
| 10 | 3 | Rashad Jennings | 73.05 | 61.38 | 76.66 | 164 | Jaguars |
| 11 | 4 | Matt Forte | 72.28 | 73.21 | 67.50 | 320 | Bears |
| 12 | 5 | Maurice Jones-Drew | 70.64 | 69.52 | 67.22 | 235 | Jaguars |
| 13 | 6 | Marshawn Lynch | 70.23 | 65.88 | 68.96 | 182 | Seahawks |
| 14 | 7 | Justin Forsett | 70.21 | 69.09 | 66.79 | 255 | Seahawks |
| 15 | 8 | Felix Jones | 69.39 | 65.49 | 67.82 | 227 | Cowboys |
| 16 | 9 | Brandon Jackson | 69.37 | 69.66 | 65.01 | 259 | Packers |
| 17 | 10 | Fred Jackson | 69.30 | 61.50 | 70.34 | 294 | Bills |
| 18 | 11 | Rashard Mendenhall | 69.14 | 71.54 | 63.37 | 228 | Steelers |
| 19 | 12 | Mike Goodson | 69.04 | 61.52 | 69.88 | 199 | Panthers |
| 20 | 13 | Mewelde Moore | 68.22 | 68.56 | 63.82 | 140 | Steelers |
| 21 | 14 | Ray Rice | 67.75 | 68.42 | 63.13 | 364 | Ravens |
| 22 | 15 | Knowshon Moreno | 67.42 | 64.06 | 65.50 | 207 | Broncos |
| 23 | 16 | Tim Hightower | 67.03 | 57.34 | 69.32 | 280 | Cardinals |
| 24 | 17 | C.J. Spiller | 66.61 | 65.92 | 62.91 | 114 | Bills |
| 25 | 18 | Michael Turner | 66.61 | 62.78 | 64.99 | 183 | Falcons |
| 26 | 19 | Keiland Williams | 66.17 | 62.67 | 64.33 | 264 | Commanders |
| 27 | 20 | Jonathan Stewart | 66.03 | 59.31 | 66.34 | 108 | Panthers |
| 28 | 21 | BenJarvus Green-Ellis | 66.02 | 62.04 | 64.50 | 137 | Patriots |
| 29 | 22 | Shonn Greene | 65.97 | 64.67 | 62.67 | 133 | Jets |
| 30 | 23 | Joseph Addai | 65.88 | 63.31 | 63.42 | 188 | Colts |
| 31 | 24 | Carnell Williams | 65.82 | 62.98 | 63.54 | 276 | Buccaneers |
| 32 | 25 | Frank Gore | 65.67 | 61.10 | 64.55 | 271 | 49ers |
| 33 | 26 | Steven Jackson | 65.17 | 62.18 | 62.99 | 372 | Rams |
| 34 | 27 | Michael Bush | 64.87 | 64.05 | 61.25 | 108 | Raiders |
| 35 | 28 | Pierre Thomas | 64.82 | 65.54 | 60.18 | 100 | Saints |
| 36 | 29 | Toby Gerhart | 64.82 | 58.46 | 64.90 | 133 | Vikings |
| 37 | 30 | Reggie Bush | 64.62 | 63.36 | 61.29 | 179 | Saints |
| 38 | 31 | Ricky Williams | 63.14 | 57.14 | 62.98 | 154 | Dolphins |
| 39 | 32 | Donald Brown | 62.73 | 56.05 | 63.02 | 211 | Colts |
| 40 | 33 | Jason Snelling | 62.69 | 56.47 | 62.67 | 245 | Falcons |
| 41 | 34 | Ronnie Brown | 62.40 | 61.77 | 58.66 | 225 | Dolphins |
| 42 | 35 | LaDainian Tomlinson | 62.10 | 54.02 | 63.32 | 377 | Jets |

### Rotation/backup (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 43 | 1 | Maurice Morris | 61.47 | 59.53 | 58.60 | 133 | Lions |
| 44 | 2 | Cedric Benson | 61.17 | 58.58 | 58.73 | 238 | Bengals |
| 45 | 3 | Jahvid Best | 61.11 | 60.27 | 57.50 | 302 | Lions |
| 46 | 4 | Chester Taylor | 60.45 | 62.54 | 54.89 | 133 | Bears |
| 47 | 5 | Julius Jones | 60.04 | 56.23 | 58.42 | 124 | Saints |
| 48 | 6 | Kenneth Darby | 58.61 | 58.29 | 54.66 | 108 | Rams |
| 49 | 7 | Correll Buckhalter | 57.80 | 55.43 | 55.22 | 182 | Broncos |
| 50 | 8 | Thomas Jones | 57.68 | 53.93 | 56.02 | 140 | Chiefs |

## LB — Linebacker

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Derrick Johnson | 85.78 | 90.20 | 78.66 | 1110 | Chiefs |
| 2 | 2 | Lawrence Timmons | 84.62 | 87.50 | 78.53 | 1116 | Steelers |
| 3 | 3 | Patrick Willis | 84.32 | 90.40 | 77.14 | 966 | 49ers |
| 4 | 4 | Takeo Spikes | 84.06 | 84.79 | 79.40 | 787 | 49ers |
| 5 | 5 | Desmond Bishop | 83.20 | 89.10 | 75.10 | 941 | Packers |
| 6 | 6 | Bart Scott | 82.31 | 84.60 | 76.61 | 1020 | Jets |
| 7 | 7 | Daryl Washington | 81.97 | 81.38 | 78.20 | 518 | Cardinals |
| 8 | 8 | Brian Urlacher | 81.92 | 83.30 | 76.84 | 1156 | Bears |
| 9 | 9 | Sean Lee | 80.79 | 74.31 | 84.08 | 167 | Cowboys |
| 10 | 10 | Brandon Spikes | 80.52 | 79.88 | 79.92 | 360 | Patriots |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Ray Lewis | 79.96 | 80.10 | 75.70 | 1145 | Ravens |
| 12 | 2 | Stephen Tulloch | 79.87 | 77.80 | 77.09 | 1186 | Titans |
| 13 | 3 | James Farrior | 79.01 | 81.50 | 73.19 | 1074 | Steelers |
| 14 | 4 | Bradie James | 78.98 | 78.70 | 75.00 | 907 | Cowboys |
| 15 | 5 | E.J. Henderson | 78.07 | 77.20 | 74.48 | 945 | Vikings |
| 16 | 6 | Karlos Dansby | 75.83 | 77.90 | 72.36 | 846 | Dolphins |
| 17 | 7 | Chris Gocong | 75.40 | 76.50 | 70.50 | 962 | Browns |
| 18 | 8 | James Laurinaitis | 75.39 | 73.10 | 72.75 | 1058 | Rams |
| 19 | 9 | Rolando McClain | 75.30 | 74.00 | 73.03 | 928 | Raiders |
| 20 | 10 | D.J. Williams | 75.10 | 73.40 | 72.07 | 1063 | Broncos |
| 21 | 11 | Jerod Mayo | 74.51 | 72.30 | 71.81 | 1119 | Patriots |
| 22 | 12 | Marvin Mitchell | 74.35 | 69.23 | 73.60 | 378 | Saints |
| 23 | 13 | Dan Connor | 74.06 | 77.73 | 75.78 | 258 | Panthers |

### Starter (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | London Fletcher | 73.10 | 70.40 | 70.74 | 1094 | Commanders |
| 25 | 2 | Stephen Nicholas | 73.08 | 70.01 | 70.96 | 704 | Falcons |
| 26 | 3 | Kevin Burnett | 72.57 | 71.80 | 68.92 | 965 | Chargers |
| 27 | 4 | Michael Boley | 72.28 | 70.00 | 69.64 | 940 | Giants |
| 28 | 5 | Jon Beason | 71.91 | 68.50 | 70.02 | 1106 | Panthers |
| 29 | 6 | Larry Foote | 71.85 | 69.45 | 69.28 | 176 | Steelers |
| 30 | 7 | Gary Brackett | 71.58 | 71.90 | 70.34 | 835 | Colts |
| 31 | 8 | Channing Crowder | 71.56 | 71.20 | 72.83 | 503 | Dolphins |
| 32 | 9 | Chad Greenway | 70.32 | 66.40 | 68.76 | 969 | Vikings |
| 33 | 10 | Daryl Smith | 70.13 | 69.40 | 66.45 | 964 | Jaguars |
| 34 | 11 | James Anderson | 70.11 | 67.90 | 67.42 | 1071 | Panthers |
| 35 | 12 | Quincy Black | 69.87 | 69.26 | 71.31 | 514 | Buccaneers |
| 36 | 13 | Rey Maualuga | 69.81 | 66.82 | 67.63 | 611 | Bengals |
| 37 | 14 | Kirk Morrison | 69.72 | 65.12 | 68.62 | 737 | Jaguars |
| 38 | 15 | Brandon Johnson | 69.41 | 66.48 | 67.20 | 500 | Bengals |
| 39 | 16 | Keith Bulluck | 68.98 | 67.35 | 69.04 | 313 | Giants |
| 40 | 17 | A.J. Hawk | 68.71 | 65.30 | 66.82 | 1166 | Packers |
| 41 | 18 | David Harris | 68.52 | 64.60 | 66.96 | 1183 | Jets |
| 42 | 19 | Jameel McClain | 68.44 | 63.85 | 67.34 | 574 | Ravens |
| 43 | 20 | NaVorro Bowman | 68.36 | 66.05 | 70.94 | 212 | 49ers |
| 44 | 21 | Keith Brooking | 68.09 | 63.30 | 67.12 | 891 | Cowboys |
| 45 | 22 | Adam Hayward | 67.70 | 64.75 | 71.75 | 140 | Buccaneers |
| 46 | 23 | Lance Briggs | 67.67 | 65.10 | 65.22 | 1014 | Bears |
| 47 | 24 | JoLonn Dunbar | 67.60 | 63.17 | 70.55 | 357 | Saints |
| 48 | 25 | Tim Dobbins | 67.45 | 63.29 | 69.19 | 305 | Dolphins |
| 49 | 26 | Jovan Belcher | 66.71 | 63.71 | 64.55 | 670 | Chiefs |
| 50 | 27 | Aaron Curry | 66.69 | 62.30 | 65.45 | 979 | Seahawks |
| 51 | 28 | Paris Lenon | 66.55 | 61.90 | 65.48 | 1092 | Cardinals |
| 52 | 29 | Barrett Ruud | 66.45 | 60.50 | 66.25 | 1023 | Buccaneers |
| 53 | 30 | Dane Fletcher | 66.25 | 66.34 | 67.22 | 166 | Patriots |
| 54 | 31 | Keith Rivers | 66.10 | 65.47 | 64.44 | 500 | Bengals |
| 55 | 32 | Will Herring | 66.06 | 62.79 | 64.08 | 289 | Seahawks |
| 56 | 33 | Bryan Kehl | 66.06 | 64.66 | 68.02 | 206 | Rams |
| 57 | 34 | Gerald McRath | 65.86 | 63.87 | 67.18 | 454 | Titans |
| 58 | 35 | DeAndre Levy | 65.69 | 68.13 | 65.10 | 716 | Lions |
| 59 | 36 | Pisa Tinoisamoa | 65.61 | 63.24 | 65.10 | 489 | Bears |
| 60 | 37 | Mike Peterson | 65.56 | 59.68 | 65.31 | 511 | Falcons |
| 61 | 38 | Lofa Tatupu | 65.56 | 61.30 | 64.23 | 1213 | Seahawks |
| 62 | 39 | Dekoda Watson | 65.40 | 62.96 | 71.19 | 130 | Buccaneers |
| 63 | 40 | Reggie Torbor | 65.08 | 62.83 | 68.67 | 411 | Bills |
| 64 | 41 | Philip Wheeler | 64.75 | 59.80 | 63.88 | 376 | Colts |
| 65 | 42 | Curtis Lofton | 64.65 | 58.50 | 64.58 | 1005 | Falcons |
| 66 | 43 | Akeem Jordan | 64.62 | 64.93 | 68.58 | 195 | Eagles |
| 67 | 44 | DeMeco Ryans | 64.40 | 69.40 | 70.32 | 375 | Texans |
| 68 | 45 | Mario Haggan | 64.19 | 57.40 | 64.55 | 1090 | Broncos |
| 69 | 46 | Moise Fokou | 63.70 | 59.43 | 63.41 | 416 | Eagles |
| 70 | 47 | Nick Roach | 63.63 | 64.24 | 66.36 | 126 | Bears |
| 71 | 48 | Paul Posluszny | 63.57 | 58.80 | 64.66 | 932 | Bills |
| 72 | 49 | Brandon Siler | 63.18 | 62.04 | 66.02 | 314 | Chargers |
| 73 | 50 | Gary Guyton | 62.98 | 58.11 | 62.06 | 661 | Patriots |
| 74 | 51 | Dannell Ellerbe | 62.79 | 58.42 | 64.67 | 301 | Ravens |

### Rotation/backup (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 75 | 1 | Kavell Conner | 61.84 | 60.12 | 64.02 | 316 | Colts |
| 76 | 2 | Na'il Diggs | 61.36 | 59.66 | 62.49 | 389 | Rams |
| 77 | 3 | Keenan Clayton | 60.77 | 63.52 | 68.18 | 107 | Eagles |
| 78 | 4 | Quentin Groves | 60.39 | 56.67 | 60.79 | 469 | Raiders |
| 79 | 5 | Will Witherspoon | 60.33 | 55.90 | 59.11 | 1157 | Titans |
| 80 | 6 | Chris Chamberlain | 60.31 | 59.45 | 64.02 | 172 | Rams |
| 81 | 7 | Brian Cushing | 60.22 | 58.04 | 61.67 | 788 | Texans |
| 82 | 8 | Stewart Bradley | 59.71 | 56.72 | 61.71 | 681 | Eagles |
| 83 | 9 | Justin Durant | 59.29 | 54.19 | 64.78 | 480 | Jaguars |
| 84 | 10 | Julian Peterson | 58.96 | 53.10 | 59.73 | 919 | Lions |
| 85 | 11 | Russell Allen | 58.89 | 52.58 | 63.09 | 289 | Jaguars |
| 86 | 12 | Jonathan Vilma | 57.48 | 47.50 | 59.97 | 1018 | Saints |
| 87 | 13 | Darryl Sharpton | 56.77 | 56.51 | 60.08 | 204 | Texans |
| 88 | 14 | Pat Angerer | 56.48 | 51.05 | 59.06 | 574 | Colts |
| 89 | 15 | Rocky McIntosh | 56.44 | 49.50 | 57.93 | 933 | Commanders |
| 90 | 16 | Tavares Gooden | 55.90 | 54.50 | 58.92 | 264 | Ravens |
| 91 | 17 | Omar Gaither | 55.15 | 59.91 | 66.31 | 179 | Eagles |
| 92 | 18 | Andra Davis | 54.95 | 58.96 | 61.53 | 220 | Bills |
| 93 | 19 | Clint Session | 54.66 | 58.69 | 63.78 | 291 | Colts |
| 94 | 20 | Sean Weatherspoon | 54.40 | 48.53 | 58.32 | 455 | Falcons |
| 95 | 21 | Wesley Woodyard | 54.23 | 54.70 | 60.62 | 145 | Broncos |
| 96 | 22 | Nick Barnett | 53.99 | 57.45 | 66.02 | 252 | Packers |
| 97 | 23 | Zach Diles | 53.47 | 40.35 | 59.08 | 648 | Texans |
| 98 | 24 | Brandon Chillar | 53.44 | 54.52 | 56.89 | 130 | Packers |
| 99 | 25 | Ernie Sims | 51.98 | 40.00 | 55.80 | 902 | Eagles |
| 100 | 26 | Larry Grant | 51.38 | 44.81 | 54.72 | 305 | Rams |
| 101 | 27 | Ricky Brown | 51.07 | 53.40 | 58.77 | 138 | Raiders |
| 102 | 28 | Keith Ellison | 50.79 | 57.36 | 62.24 | 121 | Bills |
| 103 | 29 | Scott Shanle | 50.71 | 40.00 | 54.71 | 895 | Saints |
| 104 | 30 | Zack Follett | 49.86 | 54.43 | 58.62 | 146 | Lions |
| 105 | 31 | Jamar Chaney | 48.14 | 43.71 | 60.34 | 245 | Eagles |

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

### Starter (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Ben Roethlisberger | 73.62 | 87.55 | 75.59 | 571 | Steelers |
| 7 | 2 | Matt Schaub | 73.60 | 84.10 | 72.15 | 645 | Texans |
| 8 | 3 | Matt Ryan | 73.47 | 85.70 | 68.33 | 700 | Falcons |
| 9 | 4 | Joe Flacco | 72.17 | 78.10 | 72.89 | 655 | Ravens |
| 10 | 5 | Eli Manning | 71.27 | 82.40 | 71.06 | 597 | Giants |
| 11 | 6 | Josh Freeman | 69.92 | 76.76 | 73.48 | 568 | Buccaneers |
| 12 | 7 | Michael Vick | 68.96 | 73.62 | 76.15 | 531 | Eagles |
| 13 | 8 | Carson Palmer | 68.71 | 75.50 | 65.74 | 655 | Bengals |
| 14 | 9 | Kyle Orton | 67.00 | 70.56 | 69.72 | 574 | Broncos |
| 15 | 10 | Mark Sanchez | 66.75 | 72.10 | 62.83 | 664 | Jets |
| 16 | 11 | Jay Cutler | 66.51 | 65.80 | 70.86 | 601 | Bears |
| 17 | 12 | David Garrard | 66.37 | 75.26 | 73.31 | 442 | Jaguars |
| 18 | 13 | Tony Romo | 66.30 | 79.50 | 76.44 | 235 | Cowboys |
| 19 | 14 | Matt Hasselbeck | 66.11 | 73.40 | 63.40 | 592 | Seahawks |
| 20 | 15 | Matt Cassel | 64.56 | 66.93 | 66.82 | 544 | Chiefs |
| 21 | 16 | Shaun Hill | 64.19 | 76.99 | 61.63 | 464 | Lions |
| 22 | 17 | Ryan Fitzpatrick | 64.07 | 67.70 | 66.04 | 512 | Bills |
| 23 | 18 | Chad Henne | 63.95 | 69.53 | 61.66 | 561 | Dolphins |
| 24 | 19 | Donovan McNabb | 63.63 | 66.77 | 63.77 | 550 | Commanders |
| 25 | 20 | Vince Young | 63.55 | 65.87 | 77.76 | 192 | Titans |
| 26 | 21 | Sam Bradford | 63.29 | 65.70 | 59.21 | 672 | Rams |
| 27 | 22 | Jason Campbell | 62.97 | 66.82 | 68.46 | 421 | Raiders |
| 28 | 23 | Jon Kitna | 62.86 | 63.75 | 71.47 | 374 | Cowboys |
| 29 | 24 | Troy Smith | 62.83 | 69.08 | 72.30 | 186 | 49ers |

### Rotation/backup (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Kerry Collins | 61.78 | 67.22 | 66.20 | 309 | Titans |
| 31 | 2 | Colt McCoy | 61.09 | 68.60 | 63.22 | 272 | Browns |
| 32 | 3 | Alex Smith | 60.74 | 61.96 | 63.79 | 392 | 49ers |
| 33 | 4 | Brett Favre | 60.48 | 63.39 | 61.33 | 409 | Vikings |
| 34 | 5 | Seneca Wallace | 60.45 | 63.18 | 68.33 | 116 | Browns |
| 35 | 6 | Matthew Stafford | 60.36 | 68.02 | 64.00 | 114 | Lions |
| 36 | 7 | Rex Grossman | 60.18 | 63.90 | 66.64 | 152 | Commanders |
| 37 | 8 | Jimmy Clausen | 58.57 | 61.33 | 56.50 | 359 | Panthers |
| 38 | 9 | Drew Stanton | 58.11 | 58.40 | 62.78 | 137 | Lions |
| 39 | 10 | Bruce Gradkowski | 58.06 | 59.67 | 60.93 | 182 | Raiders |
| 40 | 11 | Kevin Kolb | 57.25 | 55.15 | 60.17 | 225 | Eagles |
| 41 | 12 | Derek Anderson | 57.11 | 51.72 | 58.11 | 369 | Cardinals |
| 42 | 13 | Jake Delhomme | 56.90 | 59.74 | 56.64 | 164 | Browns |
| 43 | 14 | John Skelton | 56.68 | 58.39 | 57.00 | 147 | Cardinals |
| 44 | 15 | Charlie Whitehurst | 56.46 | 59.64 | 55.10 | 121 | Seahawks |
| 45 | 16 | Trent Edwards | 55.84 | 57.81 | 54.06 | 129 | Jaguars |
| 46 | 17 | Matt Moore | 55.66 | 54.42 | 55.85 | 165 | Panthers |
| 47 | 18 | Joe Webb III | 55.59 | 53.92 | 56.27 | 116 | Vikings |

## S — Safety

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Michael Huff | 88.05 | 81.90 | 87.99 | 972 | Raiders |
| 2 | 2 | Quintin Mikell | 84.58 | 82.20 | 82.00 | 1032 | Eagles |
| 3 | 3 | Dawan Landry | 80.33 | 74.50 | 80.05 | 1121 | Ravens |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Eric Weddle | 79.62 | 73.50 | 79.53 | 957 | Chargers |
| 5 | 2 | Danieal Manning | 78.79 | 71.40 | 79.55 | 991 | Bears |
| 6 | 3 | Sherrod Martin | 78.63 | 80.10 | 74.52 | 975 | Panthers |
| 7 | 4 | Dwight Lowery | 77.32 | 74.13 | 75.28 | 420 | Jets |
| 8 | 5 | Malcolm Jenkins | 76.38 | 75.96 | 73.52 | 821 | Saints |
| 9 | 6 | Troy Polamalu | 75.94 | 70.80 | 75.20 | 1019 | Steelers |
| 10 | 7 | Antoine Bethea | 75.74 | 66.00 | 78.06 | 1102 | Colts |
| 11 | 8 | Kerry Rhodes | 75.34 | 73.00 | 72.74 | 1130 | Cardinals |
| 12 | 9 | Brodney Pool | 75.31 | 68.90 | 75.41 | 929 | Jets |
| 13 | 10 | Kenny Phillips | 75.13 | 67.00 | 76.38 | 943 | Giants |
| 14 | 11 | Rashad Johnson | 74.46 | 67.16 | 75.16 | 376 | Cardinals |

### Starter (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Jordan Pugh | 72.79 | 67.62 | 78.32 | 134 | Panthers |
| 16 | 2 | Thomas DeCoud | 72.24 | 70.20 | 69.44 | 972 | Falcons |
| 17 | 3 | Nick Collins | 71.76 | 65.70 | 71.63 | 1208 | Packers |
| 18 | 4 | Yeremiah Bell | 71.70 | 63.40 | 73.06 | 965 | Dolphins |
| 19 | 5 | Ryan Mundy | 70.87 | 65.03 | 77.90 | 182 | Steelers |
| 20 | 6 | Abram Elam | 70.80 | 62.40 | 72.24 | 1047 | Browns |
| 21 | 7 | George Wilson | 69.98 | 65.35 | 69.94 | 256 | Bills |
| 22 | 8 | Gerald Sensabaugh | 69.22 | 61.90 | 69.93 | 918 | Cowboys |
| 23 | 9 | Patrick Chung | 69.02 | 60.70 | 71.43 | 835 | Patriots |
| 24 | 10 | Jairus Byrd | 67.89 | 56.20 | 71.52 | 897 | Bills |
| 25 | 11 | Louis Delmas | 67.64 | 62.30 | 68.07 | 900 | Lions |
| 26 | 12 | Adrian Wilson | 67.62 | 59.30 | 69.00 | 1129 | Cardinals |
| 27 | 13 | Chris Crocker | 67.55 | 65.39 | 72.12 | 519 | Bengals |
| 28 | 14 | Mike Mitchell | 66.73 | 59.92 | 68.14 | 497 | Raiders |
| 29 | 15 | Eric Berry | 66.63 | 56.50 | 69.22 | 1145 | Chiefs |
| 30 | 16 | Jordan Babineaux | 66.63 | 61.60 | 65.82 | 528 | Seahawks |
| 31 | 17 | Kam Chancellor | 66.21 | 56.84 | 68.29 | 159 | Seahawks |
| 32 | 18 | Cody Grimm | 65.52 | 61.84 | 71.11 | 526 | Buccaneers |
| 33 | 19 | Reshad Jones | 65.23 | 61.41 | 77.02 | 151 | Dolphins |
| 34 | 20 | Ryan Clark | 65.19 | 54.40 | 68.22 | 1143 | Steelers |
| 35 | 21 | Eric Smith | 65.11 | 57.80 | 65.81 | 693 | Jets |
| 36 | 22 | Mike Adams | 64.81 | 57.75 | 66.38 | 306 | Browns |
| 37 | 23 | LaRon Landry | 64.36 | 57.08 | 72.35 | 645 | Commanders |
| 38 | 24 | Kendrick Lewis | 64.32 | 61.20 | 65.36 | 848 | Chiefs |
| 39 | 25 | Chris Hope | 64.29 | 52.80 | 67.78 | 1184 | Titans |
| 40 | 26 | James Sanders | 64.27 | 56.53 | 65.27 | 810 | Patriots |
| 41 | 27 | Darian Stewart | 64.09 | 61.58 | 66.80 | 188 | Rams |
| 42 | 28 | Jarrad Page | 63.14 | 61.66 | 66.21 | 186 | Patriots |
| 43 | 29 | Brian Dawkins | 62.84 | 55.20 | 68.96 | 678 | Broncos |
| 44 | 30 | Charlie Peprah | 62.72 | 56.30 | 62.84 | 906 | Packers |
| 45 | 31 | Roy Williams | 62.66 | 56.17 | 66.99 | 466 | Bengals |
| 46 | 32 | Usama Young | 62.64 | 61.38 | 65.57 | 201 | Saints |

### Rotation/backup (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 47 | 1 | Kurt Coleman | 61.40 | 50.47 | 66.61 | 333 | Eagles |
| 48 | 2 | Nedu Ndukwe | 61.35 | 55.51 | 65.25 | 530 | Bengals |
| 49 | 3 | Sean Jones | 61.27 | 53.40 | 62.35 | 938 | Buccaneers |
| 50 | 4 | Reggie Smith | 61.22 | 57.31 | 62.80 | 548 | 49ers |
| 51 | 5 | Roman Harper | 61.17 | 48.10 | 65.72 | 913 | Saints |
| 52 | 6 | Husain Abdullah | 61.05 | 48.20 | 66.48 | 874 | Vikings |
| 53 | 7 | Amari Spievey | 61.01 | 55.66 | 64.58 | 553 | Lions |
| 54 | 8 | Jim Leonhard | 60.96 | 54.96 | 65.99 | 719 | Jets |
| 55 | 9 | Haruki Nakamura | 60.62 | 56.36 | 59.30 | 275 | Ravens |
| 56 | 10 | Tom Zbikowski | 60.45 | 62.00 | 68.66 | 323 | Ravens |
| 57 | 11 | William Moore | 60.43 | 51.10 | 62.49 | 961 | Falcons |
| 58 | 12 | James Ihedigbo | 60.35 | 60.24 | 58.34 | 119 | Jets |
| 59 | 13 | Reggie Nelson | 60.26 | 52.53 | 66.44 | 489 | Bengals |
| 60 | 14 | Reed Doughty | 60.03 | 51.28 | 63.78 | 615 | Commanders |
| 61 | 15 | Earl Thomas III | 59.84 | 47.90 | 63.64 | 1241 | Seahawks |
| 62 | 16 | Darren Sharper | 59.09 | 58.94 | 62.33 | 320 | Saints |
| 63 | 17 | Taylor Mays | 59.03 | 51.65 | 64.99 | 427 | 49ers |
| 64 | 18 | Donte Whitner | 58.80 | 40.00 | 67.16 | 1093 | Bills |
| 65 | 19 | Corey Lynch | 58.75 | 59.51 | 70.05 | 312 | Buccaneers |
| 66 | 20 | Sabby Piscitelli | 58.52 | 57.50 | 62.33 | 143 | Browns |
| 67 | 21 | Michael Griffin | 58.02 | 48.30 | 60.34 | 1179 | Titans |
| 68 | 22 | Dashon Goldson | 57.49 | 47.00 | 60.32 | 1011 | 49ers |
| 69 | 23 | Charles Godfrey | 57.43 | 46.60 | 60.48 | 1103 | Panthers |
| 70 | 24 | Steve Gregory | 57.19 | 54.83 | 61.90 | 354 | Chargers |
| 71 | 25 | Courtney Greene | 56.96 | 47.72 | 62.08 | 584 | Jaguars |
| 72 | 26 | Tyvon Branch | 56.85 | 40.00 | 63.91 | 992 | Raiders |
| 73 | 27 | Anthony Smith | 56.69 | 62.49 | 64.63 | 163 | Packers |
| 74 | 28 | Chris Clemons | 56.64 | 46.00 | 60.60 | 865 | Dolphins |
| 75 | 29 | Bernard Pollard | 56.11 | 40.00 | 63.72 | 987 | Texans |
| 76 | 30 | T.J. Ward | 55.87 | 40.00 | 62.29 | 1053 | Browns |
| 77 | 31 | Chris Harris | 55.57 | 40.00 | 61.79 | 918 | Bears |
| 78 | 32 | Michael Lewis | 55.43 | 60.00 | 57.01 | 194 | Rams |
| 79 | 33 | Brandon Meriweather | 55.25 | 40.00 | 61.25 | 930 | Patriots |
| 80 | 34 | David Bruton | 55.19 | 57.47 | 55.75 | 169 | Broncos |
| 81 | 35 | Major Wright | 54.94 | 45.57 | 61.19 | 368 | Bears |
| 82 | 36 | Antrel Rolle | 54.07 | 40.00 | 59.29 | 991 | Giants |
| 83 | 37 | Barry Church | 53.74 | 54.88 | 51.94 | 117 | Cowboys |
| 84 | 38 | Gerald Alexander | 53.55 | 59.05 | 65.72 | 187 | Panthers |
| 85 | 39 | Nate Allen | 53.36 | 40.00 | 61.24 | 781 | Eagles |
| 86 | 40 | Troy Nolan | 53.33 | 43.92 | 59.60 | 400 | Texans |
| 87 | 41 | Craig Dahl | 53.08 | 41.10 | 57.93 | 907 | Rams |
| 88 | 42 | Tyrone Culver | 52.86 | 52.68 | 57.15 | 117 | Dolphins |
| 89 | 43 | Ray Ventrone | 52.24 | 53.04 | 51.71 | 105 | Browns |
| 90 | 44 | Darcel McBath | 51.98 | 53.04 | 60.52 | 160 | Broncos |
| 91 | 45 | Madieu Williams | 51.37 | 40.00 | 56.86 | 801 | Vikings |
| 92 | 46 | Sean Considine | 51.32 | 47.51 | 54.90 | 411 | Jaguars |
| 93 | 47 | Jamarca Sanford | 50.54 | 56.68 | 55.70 | 209 | Vikings |
| 94 | 48 | Don Carey | 49.83 | 41.36 | 54.44 | 656 | Jaguars |
| 95 | 49 | Kareem Moore | 48.51 | 40.00 | 54.18 | 774 | Commanders |
| 96 | 50 | Morgan Burnett | 45.52 | 50.28 | 56.68 | 197 | Packers |
| 97 | 51 | Erik Coleman | 45.33 | 52.15 | 52.58 | 147 | Falcons |

## T — Tackle

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Andrew Whitworth | 95.27 | 90.90 | 94.02 | 1093 | Bengals |
| 2 | 2 | Doug Free | 91.36 | 83.80 | 92.23 | 1073 | Cowboys |
| 3 | 3 | Jake Long | 89.85 | 84.10 | 89.51 | 1054 | Dolphins |
| 4 | 4 | Jordan Gross | 89.58 | 84.30 | 88.93 | 988 | Panthers |
| 5 | 5 | D'Brickashaw Ferguson | 89.01 | 83.30 | 88.65 | 1307 | Jets |
| 6 | 6 | Ryan Clady | 88.16 | 82.20 | 87.97 | 1049 | Broncos |
| 7 | 7 | Jason Peters | 87.63 | 81.08 | 87.83 | 857 | Eagles |
| 8 | 8 | Joe Thomas | 87.61 | 81.89 | 87.26 | 958 | Browns |
| 9 | 9 | Eric Winston | 87.42 | 79.10 | 88.80 | 1067 | Texans |
| 10 | 10 | Tyson Clabo | 87.01 | 80.80 | 86.98 | 1193 | Falcons |
| 11 | 11 | Marcus McNeill | 85.80 | 77.36 | 87.26 | 693 | Chargers |
| 12 | 12 | Bryant McKinnie | 85.35 | 78.60 | 85.69 | 1023 | Vikings |
| 13 | 13 | Matt Light | 84.77 | 77.00 | 85.78 | 1057 | Patriots |
| 14 | 14 | Duane Brown | 84.50 | 75.47 | 86.35 | 825 | Texans |
| 15 | 15 | Jeff Backus | 83.92 | 77.60 | 83.97 | 1102 | Lions |
| 16 | 16 | Branden Albert | 82.81 | 73.70 | 84.72 | 1047 | Chiefs |
| 17 | 17 | Sebastian Vollmer | 82.61 | 74.00 | 84.18 | 1073 | Patriots |
| 18 | 18 | David Stewart | 82.41 | 72.49 | 84.85 | 953 | Titans |
| 19 | 19 | Barry Richardson | 81.74 | 71.20 | 84.60 | 1080 | Chiefs |
| 20 | 20 | Jon Stinchcomb | 81.56 | 70.50 | 84.76 | 1179 | Saints |
| 21 | 21 | Michael Roos | 81.26 | 72.89 | 82.67 | 953 | Titans |
| 22 | 22 | Flozell Adams | 81.07 | 70.10 | 84.21 | 1063 | Steelers |
| 23 | 23 | Sean Locklear | 80.93 | 72.60 | 82.31 | 1103 | Seahawks |
| 24 | 24 | Donald Penn | 80.81 | 73.29 | 81.65 | 983 | Buccaneers |
| 25 | 25 | Gosder Cherilus | 80.76 | 71.50 | 82.76 | 820 | Lions |
| 26 | 26 | Michael Oher | 80.70 | 69.50 | 84.00 | 1146 | Ravens |
| 27 | 27 | Ryan Harris | 80.01 | 67.75 | 84.01 | 655 | Broncos |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Chad Clifton | 79.97 | 71.70 | 81.32 | 1219 | Packers |
| 29 | 2 | Eben Britton | 79.37 | 67.68 | 82.99 | 424 | Jaguars |
| 30 | 3 | Vernon Carey | 79.16 | 68.37 | 82.19 | 764 | Dolphins |
| 31 | 4 | Russell Okung | 78.81 | 68.17 | 81.74 | 671 | Seahawks |
| 32 | 5 | Mario Henderson | 78.69 | 66.69 | 82.53 | 450 | Raiders |
| 33 | 6 | Brandon Keith | 78.68 | 67.53 | 81.94 | 526 | Cardinals |
| 34 | 7 | Demetress Bell | 78.62 | 68.84 | 80.98 | 950 | Bills |
| 35 | 8 | Winston Justice | 78.44 | 68.03 | 81.22 | 921 | Eagles |
| 36 | 9 | King Dunlap | 78.37 | 66.90 | 81.85 | 433 | Eagles |
| 37 | 10 | Jason Smith | 77.53 | 65.60 | 81.31 | 1022 | Rams |
| 38 | 11 | Joe Staley | 77.42 | 66.47 | 80.55 | 558 | 49ers |
| 39 | 12 | Jammal Brown | 77.10 | 65.61 | 80.59 | 833 | Commanders |
| 40 | 13 | Ryan O'Callaghan | 77.09 | 67.05 | 79.62 | 160 | Chiefs |
| 41 | 14 | Sam Baker | 77.01 | 67.20 | 79.39 | 1211 | Falcons |
| 42 | 15 | Jermon Bushrod | 76.87 | 67.40 | 79.02 | 1185 | Saints |
| 43 | 16 | Dennis Roland | 76.26 | 63.48 | 80.61 | 645 | Bengals |
| 44 | 17 | Phil Loadholt | 75.90 | 63.30 | 80.13 | 1030 | Vikings |
| 45 | 18 | Bryan Bulaga | 75.77 | 63.70 | 79.65 | 1088 | Packers |
| 46 | 19 | Corey Hilliard | 75.37 | 64.31 | 78.58 | 266 | Lions |
| 47 | 20 | Stephon Heyer | 74.75 | 60.81 | 79.87 | 445 | Commanders |
| 48 | 21 | Frank Omiyale | 74.67 | 61.40 | 79.35 | 1123 | Bears |
| 49 | 22 | Jared Veldheer | 74.33 | 60.47 | 79.41 | 882 | Raiders |
| 50 | 23 | Trent Williams | 74.26 | 63.20 | 77.46 | 870 | Commanders |
| 51 | 24 | Eugene Monroe | 74.19 | 61.90 | 78.22 | 988 | Jaguars |

### Starter (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 52 | 1 | Jeremy Trueblood | 73.63 | 59.01 | 79.21 | 496 | Buccaneers |
| 53 | 2 | Jonathan Scott | 73.40 | 62.15 | 76.74 | 938 | Steelers |
| 54 | 3 | Anthony Davis | 73.27 | 59.70 | 78.15 | 979 | 49ers |
| 55 | 4 | Max Starks | 73.09 | 61.29 | 76.79 | 341 | Steelers |
| 56 | 5 | Lydon Murtha | 72.66 | 60.24 | 76.77 | 230 | Dolphins |
| 57 | 6 | Tony Pashos | 72.17 | 59.45 | 76.48 | 242 | Browns |
| 58 | 7 | Marc Colombo | 71.45 | 56.50 | 77.25 | 998 | Cowboys |
| 59 | 8 | Rashad Butler | 71.17 | 58.00 | 75.79 | 246 | Texans |
| 60 | 9 | Wayne Hunter | 70.44 | 56.40 | 75.63 | 511 | Jets |
| 61 | 10 | J'Marcus Webb | 70.15 | 53.33 | 77.19 | 921 | Bears |
| 62 | 11 | Anthony Collins | 69.96 | 64.02 | 69.75 | 255 | Bengals |
| 63 | 12 | Andre Smith | 69.74 | 54.98 | 75.42 | 281 | Bengals |
| 64 | 13 | David Diehl | 69.59 | 56.97 | 73.84 | 780 | Giants |
| 65 | 14 | Levi Brown | 69.20 | 55.36 | 74.26 | 960 | Cardinals |
| 66 | 15 | Jeff Linkenbach | 67.60 | 59.69 | 68.71 | 367 | Colts |
| 67 | 16 | Will Beatty | 66.03 | 58.62 | 66.80 | 163 | Giants |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jason Witten | 84.18 | 88.91 | 76.86 | 601 | Cowboys |
| 2 | 2 | Antonio Gates | 83.43 | 78.24 | 82.72 | 354 | Chargers |
| 3 | 3 | Marcedes Lewis | 83.18 | 84.69 | 78.00 | 450 | Jaguars |
| 4 | 4 | Jermichael Finley | 81.26 | 69.42 | 84.98 | 129 | Packers |
| 5 | 5 | Vernon Davis | 80.95 | 78.18 | 78.63 | 551 | 49ers |
| 6 | 6 | Rob Gronkowski | 80.53 | 80.95 | 76.08 | 422 | Patriots |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Anthony Fasano | 79.32 | 78.54 | 75.68 | 500 | Dolphins |
| 8 | 2 | Tony Moeaki | 78.43 | 79.43 | 73.60 | 471 | Chiefs |
| 9 | 3 | Heath Miller | 78.32 | 77.46 | 74.72 | 465 | Steelers |
| 10 | 4 | Todd Heap | 78.26 | 75.13 | 76.18 | 432 | Ravens |
| 11 | 5 | Zach Miller | 78.10 | 82.47 | 71.02 | 641 | Raiders |
| 12 | 6 | Joel Dreessen | 77.33 | 71.37 | 77.14 | 406 | Texans |
| 13 | 7 | Benjamin Watson | 76.67 | 74.69 | 73.82 | 523 | Browns |
| 14 | 8 | Jimmy Graham | 76.44 | 68.20 | 77.76 | 167 | Saints |
| 15 | 9 | Kevin Boss | 74.98 | 68.75 | 74.96 | 495 | Giants |
| 16 | 10 | Jim Kleinsasser | 74.87 | 71.64 | 72.85 | 226 | Vikings |
| 17 | 11 | Kellen Winslow | 74.65 | 72.81 | 71.71 | 501 | Buccaneers |
| 18 | 12 | Jacob Tamme | 74.60 | 75.48 | 69.84 | 416 | Colts |
| 19 | 13 | Owen Daniels | 74.54 | 67.08 | 75.34 | 352 | Texans |
| 20 | 14 | Aaron Hernandez | 74.18 | 66.15 | 75.37 | 315 | Patriots |

### Starter (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Fred Davis | 73.81 | 62.85 | 76.95 | 206 | Commanders |
| 22 | 2 | Dustin Keller | 73.65 | 66.33 | 74.36 | 556 | Jets |
| 23 | 3 | Dallas Clark | 72.97 | 67.34 | 72.56 | 260 | Colts |
| 24 | 4 | Greg Olsen | 72.66 | 68.01 | 71.59 | 469 | Bears |
| 25 | 5 | Cameron Morrah | 72.13 | 60.76 | 75.54 | 149 | Seahawks |
| 26 | 6 | Jared Cook | 71.86 | 65.73 | 71.78 | 164 | Titans |
| 27 | 7 | Martellus Bennett | 71.80 | 71.09 | 68.11 | 232 | Cowboys |
| 28 | 8 | Randy McMichael | 71.78 | 71.40 | 67.87 | 297 | Chargers |
| 29 | 9 | Brent Celek | 71.18 | 64.17 | 71.68 | 567 | Eagles |
| 30 | 10 | Tony Gonzalez | 71.10 | 68.55 | 68.63 | 638 | Falcons |
| 31 | 11 | Brandon Pettigrew | 70.67 | 64.62 | 70.54 | 563 | Lions |
| 32 | 12 | Jeremy Shockey | 70.05 | 65.41 | 68.97 | 372 | Saints |
| 33 | 13 | Chris Baker | 69.99 | 58.82 | 73.27 | 209 | Seahawks |
| 34 | 14 | Delanie Walker | 69.91 | 65.12 | 68.94 | 196 | 49ers |
| 35 | 15 | Chris Cooley | 69.55 | 67.42 | 66.80 | 629 | Commanders |
| 36 | 16 | Craig Stevens | 69.33 | 66.88 | 66.80 | 210 | Titans |
| 37 | 17 | Andrew Quarless | 68.83 | 58.53 | 71.53 | 271 | Packers |
| 38 | 18 | Visanthe Shiancoe | 68.61 | 58.23 | 71.36 | 427 | Vikings |
| 39 | 19 | Tony Scheffler | 68.29 | 65.21 | 66.17 | 302 | Lions |
| 40 | 20 | Dan Gronkowski | 67.89 | 63.51 | 66.65 | 141 | Broncos |
| 41 | 21 | Justin Peelle | 67.78 | 70.58 | 61.74 | 154 | Falcons |
| 42 | 22 | Jermaine Gresham | 67.36 | 63.59 | 65.70 | 401 | Bengals |
| 43 | 23 | Tom Crabtree | 67.09 | 59.14 | 68.22 | 122 | Packers |
| 44 | 24 | Billy Bajema | 66.86 | 58.07 | 68.55 | 170 | Rams |
| 45 | 25 | Jim Dray | 66.55 | 56.71 | 68.94 | 106 | Cardinals |
| 46 | 26 | Ben Patrick | 66.39 | 62.56 | 64.78 | 120 | Cardinals |
| 47 | 27 | Ed Dickson | 66.34 | 56.81 | 68.52 | 187 | Ravens |
| 48 | 28 | Brandon Myers | 66.33 | 59.58 | 66.66 | 115 | Raiders |
| 49 | 29 | Daniel Graham | 65.93 | 58.59 | 66.66 | 511 | Broncos |
| 50 | 30 | Jonathan Stupar | 65.55 | 60.18 | 64.97 | 131 | Bills |
| 51 | 31 | Dante Rosario | 65.51 | 58.15 | 66.25 | 392 | Panthers |
| 52 | 32 | Bo Scaife | 65.49 | 57.94 | 66.35 | 258 | Titans |
| 53 | 33 | John Carlson | 65.34 | 58.70 | 65.60 | 495 | Seahawks |
| 54 | 34 | Jeff King | 64.97 | 63.64 | 61.69 | 279 | Panthers |
| 55 | 35 | David Martin | 64.82 | 62.12 | 62.46 | 140 | Bills |
| 56 | 36 | Leonard Pope | 63.95 | 60.18 | 62.30 | 232 | Chiefs |
| 57 | 37 | Matt Spaeth | 63.89 | 61.12 | 61.57 | 206 | Steelers |
| 58 | 38 | Ben Hartsock | 63.79 | 61.41 | 61.21 | 145 | Jets |
| 59 | 39 | Reggie Kelly | 63.74 | 61.82 | 60.86 | 159 | Bengals |
| 60 | 40 | Travis Beckum | 63.01 | 59.44 | 61.23 | 145 | Giants |
| 61 | 41 | Donald Lee | 62.88 | 58.53 | 61.62 | 124 | Packers |

### Rotation/backup (0 players)

_None._

## WR — Wide Receiver

- **Season used:** `2010`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Brandon Lloyd | 88.74 | 89.51 | 84.06 | 610 | Broncos |
| 2 | 2 | Kenny Britt | 83.73 | 76.84 | 84.16 | 277 | Titans |
| 3 | 3 | Andre Johnson | 83.60 | 82.19 | 80.37 | 524 | Texans |
| 4 | 4 | Patrick Crayton | 83.00 | 70.86 | 86.93 | 209 | Chargers |
| 5 | 5 | Percy Harvin | 82.35 | 80.91 | 79.14 | 379 | Vikings |
| 6 | 6 | Mike Wallace | 82.13 | 76.90 | 81.45 | 666 | Steelers |
| 7 | 7 | Calvin Johnson | 82.00 | 83.06 | 77.13 | 652 | Lions |
| 8 | 8 | Malcom Floyd | 81.95 | 74.85 | 82.51 | 353 | Chargers |
| 9 | 9 | Dwayne Bowe | 81.33 | 77.02 | 80.03 | 547 | Chiefs |
| 10 | 10 | Arrelious Benn | 81.26 | 70.88 | 84.01 | 177 | Buccaneers |
| 11 | 11 | Roddy White | 81.22 | 82.64 | 76.11 | 651 | Falcons |
| 12 | 12 | Greg Jennings | 81.08 | 75.00 | 80.97 | 758 | Packers |
| 13 | 13 | Hakeem Nicks | 80.52 | 77.53 | 78.35 | 478 | Giants |
| 14 | 14 | Mario Manningham | 80.44 | 70.99 | 82.58 | 444 | Giants |

### Good (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Vincent Jackson | 78.95 | 69.01 | 81.41 | 117 | Chargers |
| 16 | 2 | Braylon Edwards | 78.87 | 69.57 | 80.91 | 604 | Jets |
| 17 | 3 | Larry Fitzgerald | 78.86 | 78.06 | 75.23 | 636 | Cardinals |
| 18 | 4 | Johnny Knox | 78.73 | 71.23 | 79.57 | 640 | Bears |
| 19 | 5 | Brandon Stokley | 78.66 | 73.42 | 77.99 | 239 | Seahawks |
| 20 | 6 | Demaryius Thomas | 78.50 | 67.68 | 81.54 | 104 | Broncos |
| 21 | 7 | Seyi Ajirotutu | 78.17 | 63.56 | 83.74 | 157 | Chargers |
| 22 | 8 | Santonio Holmes | 77.91 | 74.47 | 76.03 | 462 | Jets |
| 23 | 9 | Deion Branch | 77.66 | 73.77 | 76.08 | 521 | Patriots |
| 24 | 10 | Earl Bennett | 77.64 | 71.38 | 77.64 | 375 | Bears |
| 25 | 11 | Santana Moss | 77.37 | 76.10 | 74.05 | 660 | Commanders |
| 26 | 12 | Reggie Wayne | 77.14 | 75.70 | 73.93 | 738 | Colts |
| 27 | 13 | Steve Johnson | 76.87 | 75.13 | 73.86 | 583 | Bills |
| 28 | 14 | Mike Williams | 76.77 | 74.70 | 73.99 | 1042 | Buccaneers |
| 29 | 15 | Austin Collie | 76.60 | 71.59 | 75.78 | 293 | Colts |
| 30 | 16 | Danario Alexander | 76.58 | 64.59 | 80.40 | 145 | Rams |
| 31 | 17 | Lance Moore | 76.45 | 73.93 | 73.96 | 513 | Saints |
| 32 | 18 | Mike Thomas | 76.19 | 71.98 | 74.83 | 490 | Jaguars |
| 33 | 19 | Robert Meachem | 75.82 | 69.53 | 75.85 | 344 | Saints |
| 34 | 20 | Davone Bess | 75.46 | 74.96 | 71.63 | 461 | Dolphins |
| 35 | 21 | Terrell Owens | 75.39 | 69.07 | 75.43 | 554 | Bengals |
| 36 | 22 | Marques Colston | 75.36 | 74.19 | 71.98 | 641 | Saints |
| 37 | 23 | Jordy Nelson | 75.34 | 71.89 | 73.48 | 432 | Packers |
| 38 | 24 | Anquan Boldin | 75.34 | 71.42 | 73.79 | 629 | Ravens |
| 39 | 25 | Derrick Mason | 75.33 | 73.77 | 72.20 | 571 | Ravens |
| 40 | 26 | Anthony Armstrong | 75.08 | 64.92 | 77.68 | 547 | Commanders |
| 41 | 27 | Miles Austin | 75.03 | 67.90 | 75.61 | 628 | Cowboys |
| 42 | 28 | Hines Ward | 74.94 | 74.00 | 71.40 | 599 | Steelers |
| 43 | 29 | Jeremy Maclin | 74.79 | 70.20 | 73.68 | 700 | Eagles |
| 44 | 30 | Legedu Naanee | 74.73 | 65.23 | 76.90 | 313 | Chargers |
| 45 | 31 | Brian Hartline | 74.71 | 67.67 | 75.24 | 404 | Dolphins |
| 46 | 32 | Dez Bryant | 74.60 | 69.14 | 74.07 | 331 | Cowboys |
| 47 | 33 | Damian Williams | 74.56 | 64.52 | 77.09 | 143 | Titans |
| 48 | 34 | Nate Washington | 74.45 | 67.24 | 75.09 | 515 | Titans |
| 49 | 35 | Michael Jenkins | 74.38 | 70.99 | 72.48 | 425 | Falcons |
| 50 | 36 | DeSean Jackson | 74.35 | 61.42 | 78.80 | 593 | Eagles |
| 51 | 37 | Jabar Gaffney | 74.13 | 68.11 | 73.98 | 573 | Broncos |
| 52 | 38 | Eddie Royal | 74.12 | 68.75 | 73.54 | 450 | Broncos |

### Starter (65 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 53 | 1 | Emmanuel Sanders | 73.99 | 70.70 | 72.02 | 350 | Steelers |
| 54 | 2 | Jacoby Ford | 73.90 | 64.57 | 75.96 | 326 | Raiders |
| 55 | 3 | Sidney Rice | 73.57 | 64.40 | 75.51 | 181 | Vikings |
| 56 | 4 | Josh Morgan | 73.55 | 63.84 | 75.86 | 527 | 49ers |
| 57 | 5 | Wes Welker | 73.41 | 74.19 | 68.73 | 494 | Patriots |
| 58 | 6 | Mike Sims-Walker | 73.29 | 70.30 | 71.11 | 408 | Jaguars |
| 59 | 7 | Joshua Cribbs | 73.18 | 61.03 | 77.12 | 175 | Browns |
| 60 | 8 | Jordan Shipley | 73.12 | 65.68 | 73.92 | 448 | Bengals |
| 61 | 9 | Andre Caldwell | 72.95 | 63.13 | 75.33 | 180 | Bengals |
| 62 | 10 | James Jones | 72.55 | 63.95 | 74.12 | 531 | Packers |
| 63 | 11 | Louis Murphy Jr. | 72.55 | 60.67 | 76.30 | 459 | Raiders |
| 64 | 12 | Lee Evans | 72.53 | 63.78 | 74.20 | 487 | Bills |
| 65 | 13 | Chad Johnson | 72.48 | 69.50 | 70.30 | 573 | Bengals |
| 66 | 14 | Kevin Walter | 72.45 | 66.96 | 71.94 | 525 | Texans |
| 67 | 15 | Brandon Tate | 72.40 | 59.42 | 76.89 | 341 | Patriots |
| 68 | 16 | David Gettis | 72.25 | 63.62 | 73.83 | 446 | Panthers |
| 69 | 17 | Steve Breaston | 71.89 | 64.75 | 72.49 | 493 | Cardinals |
| 70 | 18 | Michael Crabtree | 71.76 | 65.74 | 71.61 | 565 | 49ers |
| 71 | 19 | Steve Smith | 71.64 | 71.40 | 67.64 | 786 | Panthers |
| 72 | 20 | Jason Avant | 71.55 | 69.30 | 68.88 | 538 | Eagles |
| 73 | 21 | Mark Clayton | 71.19 | 64.47 | 71.51 | 170 | Rams |
| 74 | 22 | Golden Tate | 71.09 | 62.22 | 72.83 | 160 | Seahawks |
| 75 | 23 | Danny Amendola | 71.07 | 70.04 | 67.59 | 454 | Rams |
| 76 | 24 | Pierre Garcon | 70.84 | 66.99 | 69.24 | 640 | Colts |
| 77 | 25 | Brandon Gibson | 70.76 | 65.55 | 70.07 | 440 | Rams |
| 78 | 26 | Roy Williams | 70.71 | 61.53 | 72.67 | 479 | Cowboys |
| 79 | 27 | Nate Burleson | 70.42 | 65.29 | 69.67 | 550 | Lions |
| 80 | 28 | Blair White | 70.01 | 66.77 | 68.01 | 365 | Colts |
| 81 | 29 | David Nelson | 69.93 | 65.64 | 68.62 | 291 | Bills |
| 82 | 30 | Mohamed Massaquoi | 69.70 | 61.91 | 70.72 | 456 | Browns |
| 83 | 31 | Roscoe Parrish | 69.17 | 63.33 | 68.89 | 304 | Bills |
| 84 | 32 | Antwaan Randle El | 69.10 | 61.81 | 69.79 | 277 | Steelers |
| 85 | 33 | Jacoby Jones | 69.09 | 63.16 | 68.88 | 391 | Texans |
| 86 | 34 | Roydell Williams | 68.95 | 53.91 | 74.81 | 222 | Commanders |
| 87 | 35 | Devin Aromashodu | 68.93 | 62.57 | 69.00 | 134 | Bears |
| 88 | 36 | Brandon LaFell | 68.91 | 63.36 | 68.45 | 353 | Panthers |
| 89 | 37 | Marlon Moore | 68.26 | 57.01 | 71.60 | 140 | Dolphins |
| 90 | 38 | Jerricho Cotchery | 67.99 | 64.04 | 66.46 | 468 | Jets |
| 91 | 39 | Donald Driver | 67.95 | 61.63 | 68.00 | 606 | Packers |
| 92 | 40 | Donald Jones | 67.85 | 60.69 | 68.46 | 184 | Bills |
| 93 | 41 | Devery Henderson | 67.61 | 60.34 | 68.29 | 473 | Saints |
| 94 | 42 | Andre Roberts | 67.49 | 57.20 | 70.18 | 308 | Cardinals |
| 95 | 43 | Riley Cooper | 67.44 | 58.02 | 69.55 | 180 | Eagles |
| 96 | 44 | Sammie Stroughter | 67.28 | 60.27 | 67.78 | 306 | Buccaneers |
| 97 | 45 | Kassim Osgood | 67.12 | 61.38 | 66.78 | 131 | Jaguars |
| 98 | 46 | Mardy Gilyard | 66.93 | 59.80 | 67.51 | 110 | Rams |
| 99 | 47 | Chansi Stuckey | 66.93 | 61.21 | 66.58 | 297 | Browns |
| 100 | 48 | Devin Hester | 66.54 | 58.47 | 67.76 | 534 | Bears |
| 101 | 49 | Derek Hagan | 66.29 | 64.14 | 63.55 | 172 | Giants |
| 102 | 50 | Joey Galloway | 66.00 | 56.58 | 68.11 | 264 | Commanders |
| 103 | 51 | Brian Finneran | 65.97 | 63.45 | 63.48 | 186 | Falcons |
| 104 | 52 | David Anderson | 65.93 | 58.68 | 66.59 | 157 | Texans |
| 105 | 53 | Max Komar | 65.81 | 55.96 | 68.21 | 146 | Cardinals |
| 106 | 54 | Deon Butler | 65.66 | 60.29 | 65.07 | 355 | Seahawks |
| 107 | 55 | Harry Douglas | 65.45 | 55.18 | 68.13 | 455 | Falcons |
| 108 | 56 | Laurent Robinson | 65.30 | 58.32 | 65.78 | 467 | Rams |
| 109 | 57 | Brian Robiskie | 64.86 | 59.70 | 64.14 | 362 | Browns |
| 110 | 58 | Terrance Copper | 64.57 | 60.71 | 62.98 | 172 | Chiefs |
| 111 | 59 | Darrius Heyward-Bey | 64.27 | 54.25 | 66.78 | 421 | Raiders |
| 112 | 60 | Early Doucet | 64.16 | 55.85 | 65.53 | 261 | Cardinals |
| 113 | 61 | Sam Hurd | 63.95 | 60.59 | 62.02 | 158 | Cowboys |
| 114 | 62 | Brad Smith | 63.12 | 59.05 | 61.66 | 123 | Jets |
| 115 | 63 | Johnnie Lee Higgins | 62.56 | 54.43 | 63.81 | 253 | Raiders |
| 116 | 64 | Bernard Berrian | 62.50 | 55.81 | 62.79 | 369 | Vikings |
| 117 | 65 | Tiquan Underwood | 62.22 | 53.77 | 63.68 | 208 | Jaguars |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 118 | 1 | Ted Ginn Jr. | 61.97 | 52.47 | 64.13 | 191 | 49ers |
| 119 | 2 | Stephen Williams | 61.23 | 54.11 | 61.81 | 149 | Cardinals |
| 120 | 3 | Bryant Johnson | 59.07 | 50.32 | 60.73 | 493 | Lions |
| 121 | 4 | Derrick Williams | 58.34 | 53.69 | 57.27 | 125 | Lions |
