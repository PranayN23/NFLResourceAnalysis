# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:38:32Z
- **Requested analysis_year:** 2011 (clamped to 2011)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Myers | 97.54 | 92.60 | 96.66 | 1165 | Texans |
| 2 | 2 | Nick Mangold | 94.42 | 87.98 | 94.54 | 894 | Jets |
| 3 | 3 | John Sullivan | 88.70 | 81.93 | 89.05 | 936 | Vikings |
| 4 | 4 | Jeff Saturday | 88.35 | 81.33 | 88.87 | 975 | Colts |
| 5 | 5 | Scott Wells | 88.20 | 80.60 | 89.10 | 1085 | Packers |
| 6 | 6 | Matt Birk | 84.46 | 76.00 | 85.93 | 1192 | Ravens |
| 7 | 7 | Brad Meester | 82.93 | 75.20 | 83.91 | 1041 | Jaguars |
| 8 | 8 | Alex Mack | 82.87 | 74.60 | 84.21 | 1063 | Browns |
| 9 | 9 | Ryan Kalil | 82.68 | 73.60 | 84.56 | 1039 | Panthers |
| 10 | 10 | Todd McClure | 82.05 | 73.66 | 83.47 | 928 | Falcons |
| 11 | 11 | Mike Pouncey | 81.52 | 73.00 | 83.03 | 1005 | Dolphins |
| 12 | 12 | Eric Wood | 81.18 | 73.81 | 81.93 | 543 | Bills |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Kyle Cook | 79.50 | 69.00 | 82.34 | 1134 | Bengals |
| 14 | 2 | Lyle Sendlein | 79.15 | 70.30 | 80.88 | 1029 | Cardinals |
| 15 | 3 | Nick Hardwick | 79.11 | 69.70 | 81.22 | 1050 | Chargers |
| 16 | 4 | Max Unger | 78.89 | 75.60 | 76.91 | 992 | Seahawks |
| 17 | 5 | Will Montgomery | 78.83 | 69.20 | 81.08 | 1081 | Commanders |
| 18 | 6 | Ryan Wendell | 78.54 | 69.59 | 80.34 | 354 | Patriots |
| 19 | 7 | Jonathan Goodwin | 78.03 | 68.80 | 80.02 | 1144 | 49ers |
| 20 | 8 | Dominic Raiola | 77.76 | 67.70 | 80.30 | 1148 | Lions |
| 21 | 9 | David Baas | 75.92 | 65.48 | 78.71 | 971 | Giants |
| 22 | 10 | Eugene Amano | 74.62 | 64.80 | 77.00 | 1021 | Titans |
| 23 | 11 | Jason Kelce | 74.10 | 62.50 | 77.66 | 1063 | Eagles |

### Starter (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Maurkice Pouncey | 73.63 | 62.69 | 76.75 | 845 | Steelers |
| 25 | 2 | Jeff Faine | 71.63 | 60.87 | 74.64 | 925 | Buccaneers |
| 26 | 3 | Roberto Garza | 71.20 | 58.00 | 75.83 | 1007 | Bears |
| 27 | 4 | Phil Costa | 70.11 | 56.90 | 74.75 | 992 | Cowboys |
| 28 | 5 | J.D. Walton | 68.91 | 57.10 | 72.61 | 1195 | Broncos |
| 29 | 6 | Nick McDonald | 68.30 | 61.08 | 68.94 | 106 | Patriots |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Darrelle Revis | 92.66 | 87.60 | 91.86 | 1014 | Jets |
| 2 | 2 | Lardarius Webb | 89.99 | 87.30 | 87.62 | 1100 | Ravens |
| 3 | 3 | Champ Bailey | 88.12 | 84.50 | 87.40 | 1013 | Broncos |
| 4 | 4 | Asante Samuel | 86.46 | 83.76 | 86.96 | 809 | Eagles |
| 5 | 5 | Brent Grimes | 86.29 | 82.41 | 87.31 | 692 | Falcons |
| 6 | 6 | Cortland Finnegan | 85.72 | 84.20 | 82.56 | 1122 | Titans |
| 7 | 7 | Richard Sherman | 83.58 | 78.56 | 84.85 | 763 | Seahawks |
| 8 | 8 | Johnathan Joseph | 83.28 | 80.20 | 82.74 | 1022 | Texans |
| 9 | 9 | Brandon Flowers | 83.06 | 77.50 | 82.60 | 974 | Chiefs |
| 10 | 10 | Alterraun Verner | 81.79 | 76.64 | 81.06 | 655 | Titans |
| 11 | 11 | Brice McCain | 80.82 | 74.19 | 83.02 | 465 | Texans |
| 12 | 12 | Carlos Rogers | 80.63 | 75.10 | 81.71 | 1173 | 49ers |
| 13 | 13 | Charles Tillman | 80.61 | 75.20 | 80.05 | 1070 | Bears |
| 14 | 14 | Corey Webster | 80.44 | 73.10 | 81.55 | 1344 | Giants |
| 15 | 15 | Chris Gamble | 80.08 | 77.50 | 80.24 | 913 | Panthers |

### Good (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Aqib Talib | 79.56 | 75.32 | 82.12 | 682 | Buccaneers |
| 17 | 2 | Tim Jennings | 79.39 | 75.30 | 77.95 | 995 | Bears |
| 18 | 3 | Tramon Williams | 78.66 | 71.70 | 79.13 | 1056 | Packers |
| 19 | 4 | Joe Haden | 78.42 | 67.20 | 82.39 | 973 | Browns |
| 20 | 5 | Jabari Greer | 78.41 | 72.20 | 78.76 | 1097 | Saints |
| 21 | 6 | Antoine Winfield | 78.18 | 78.83 | 80.74 | 322 | Vikings |
| 22 | 7 | Derek Cox | 78.12 | 73.29 | 84.85 | 324 | Jaguars |
| 23 | 8 | Brandon Carr | 78.03 | 69.60 | 79.48 | 1006 | Chiefs |
| 24 | 9 | Tarell Brown | 77.83 | 70.20 | 79.92 | 1145 | 49ers |
| 25 | 10 | Jimmy Smith | 77.08 | 69.37 | 83.26 | 331 | Ravens |
| 26 | 11 | Chris Houston | 76.46 | 71.40 | 76.70 | 826 | Lions |
| 27 | 12 | Chris Harris Jr. | 76.23 | 70.21 | 78.16 | 550 | Broncos |
| 28 | 13 | Josh Wilson | 75.96 | 66.80 | 78.28 | 944 | Commanders |
| 29 | 14 | Leodis McKelvin | 75.86 | 69.53 | 78.51 | 503 | Bills |
| 30 | 15 | Jason McCourty | 75.83 | 67.50 | 79.44 | 964 | Titans |
| 31 | 16 | Brandon Browner | 75.58 | 61.90 | 80.53 | 1071 | Seahawks |
| 32 | 17 | Vontae Davis | 75.52 | 69.53 | 77.94 | 677 | Dolphins |
| 33 | 18 | Drew Coleman | 75.40 | 69.52 | 75.16 | 494 | Jaguars |
| 34 | 19 | Sheldon Brown | 75.39 | 67.30 | 76.62 | 959 | Browns |
| 35 | 20 | Patrick Robinson | 74.93 | 69.53 | 77.09 | 813 | Saints |
| 36 | 21 | Antoine Cason | 74.66 | 63.00 | 78.26 | 870 | Chargers |
| 37 | 22 | Antonio Cromartie | 74.65 | 65.80 | 76.38 | 944 | Jets |
| 38 | 23 | Aaron Berry | 74.61 | 70.69 | 81.53 | 455 | Lions |
| 39 | 24 | William Middleton | 74.33 | 70.84 | 77.69 | 388 | Jaguars |
| 40 | 25 | Jason Allen | 74.10 | 64.22 | 77.31 | 544 | Texans |
| 41 | 26 | Nnamdi Asomugha | 74.07 | 65.80 | 76.20 | 932 | Eagles |

### Starter (55 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 42 | 1 | Leon Hall | 73.46 | 67.36 | 77.91 | 542 | Bengals |
| 43 | 2 | Dimitri Patterson | 73.44 | 63.33 | 77.72 | 545 | Browns |
| 44 | 3 | Chris Culliver | 73.26 | 64.72 | 75.82 | 527 | 49ers |
| 45 | 4 | Richard Marshall | 73.05 | 64.18 | 74.80 | 819 | Cardinals |
| 46 | 5 | Kyle Arrington | 72.99 | 62.70 | 75.68 | 1155 | Patriots |
| 47 | 6 | Javier Arenas | 72.83 | 63.84 | 75.31 | 374 | Chiefs |
| 48 | 7 | William Gay | 72.57 | 62.70 | 74.98 | 1009 | Steelers |
| 49 | 8 | Sam Shields | 72.44 | 63.25 | 74.40 | 752 | Packers |
| 50 | 9 | DeAngelo Hall | 71.70 | 60.10 | 75.26 | 1006 | Commanders |
| 51 | 10 | Jerraud Powers | 71.61 | 68.34 | 74.58 | 794 | Colts |
| 52 | 11 | Cary Williams | 71.49 | 65.40 | 76.46 | 1089 | Ravens |
| 53 | 12 | Mike Jenkins | 71.46 | 63.88 | 74.94 | 586 | Cowboys |
| 54 | 13 | Aaron Ross | 70.97 | 60.70 | 74.04 | 1161 | Giants |
| 55 | 14 | Tramaine Brock Sr. | 70.94 | 64.24 | 85.10 | 116 | 49ers |
| 56 | 15 | Stanford Routt | 70.88 | 59.60 | 74.23 | 1100 | Raiders |
| 57 | 16 | Cedric Griffin | 70.85 | 67.10 | 75.30 | 920 | Vikings |
| 58 | 17 | Chris Cook | 70.44 | 66.37 | 79.40 | 253 | Vikings |
| 59 | 18 | Roy Lewis | 70.35 | 65.34 | 74.60 | 261 | Seahawks |
| 60 | 19 | Kyle Wilson | 70.24 | 61.91 | 72.01 | 569 | Jets |
| 61 | 20 | D.J. Moore | 70.19 | 60.46 | 74.46 | 485 | Bears |
| 62 | 21 | Morgan Trent | 69.87 | 64.28 | 78.14 | 225 | Jaguars |
| 63 | 22 | Jarrett Bush | 69.85 | 57.66 | 75.37 | 313 | Packers |
| 64 | 23 | Drayton Florence | 69.80 | 58.70 | 73.03 | 1003 | Bills |
| 65 | 24 | Joselio Hanson | 69.63 | 59.35 | 72.31 | 395 | Eagles |
| 66 | 25 | Dominique Rodgers-Cromartie | 69.61 | 60.83 | 73.25 | 473 | Eagles |
| 67 | 26 | Keenan Lewis | 69.39 | 62.55 | 74.47 | 393 | Steelers |
| 68 | 27 | Adam Jones | 69.19 | 65.81 | 76.78 | 495 | Bengals |
| 69 | 28 | Justin Rogers | 68.93 | 61.68 | 76.89 | 214 | Bills |
| 70 | 29 | Andre' Goodman | 68.76 | 62.80 | 71.70 | 1167 | Broncos |
| 71 | 30 | Rashean Mathis | 68.25 | 62.80 | 72.27 | 528 | Jaguars |
| 72 | 31 | Chris Carr | 67.46 | 61.44 | 71.85 | 184 | Ravens |
| 73 | 32 | Jacob Lacey | 67.29 | 62.30 | 68.92 | 701 | Colts |
| 74 | 33 | Tracy Porter | 67.10 | 57.63 | 70.41 | 807 | Saints |
| 75 | 34 | Kelly Jennings | 66.78 | 55.03 | 71.74 | 405 | Bengals |
| 76 | 35 | Chris Rucker | 66.58 | 63.29 | 70.86 | 317 | Colts |
| 77 | 36 | Nate Clements | 66.50 | 57.00 | 68.66 | 983 | Bengals |
| 78 | 37 | Buster Skrine | 66.44 | 58.28 | 72.91 | 123 | Browns |
| 79 | 38 | Captain Munnerlyn | 66.35 | 56.21 | 70.25 | 816 | Panthers |
| 80 | 39 | Chris Owens | 66.26 | 59.55 | 70.86 | 338 | Falcons |
| 81 | 40 | Alan Ball | 65.59 | 51.78 | 70.63 | 478 | Cowboys |
| 82 | 41 | Josh Gordy | 65.38 | 56.84 | 69.51 | 674 | Rams |
| 83 | 42 | Alphonso Smith | 65.37 | 59.44 | 71.92 | 257 | Lions |
| 84 | 43 | Orlando Scandrick | 65.26 | 56.78 | 68.70 | 658 | Cowboys |
| 85 | 44 | Dunta Robinson | 65.21 | 56.60 | 66.78 | 1034 | Falcons |
| 86 | 45 | DeMarcus Van Dyke | 64.91 | 58.51 | 71.26 | 319 | Raiders |
| 87 | 46 | Bradley Fletcher | 64.81 | 58.03 | 72.98 | 294 | Rams |
| 88 | 47 | Eric Wright | 64.27 | 53.60 | 68.39 | 1089 | Lions |
| 89 | 48 | Dominique Franks | 63.64 | 58.04 | 74.27 | 403 | Falcons |
| 90 | 49 | R.J. Stanford | 63.64 | 60.61 | 67.75 | 255 | Panthers |
| 91 | 50 | E.J. Biggers | 63.58 | 51.68 | 67.35 | 660 | Buccaneers |
| 92 | 51 | Sean Smith | 62.96 | 51.30 | 66.95 | 1064 | Dolphins |
| 93 | 52 | Nolan Carroll | 62.68 | 54.64 | 71.42 | 321 | Dolphins |
| 94 | 53 | Kevin Barnes | 62.58 | 54.23 | 67.37 | 418 | Commanders |
| 95 | 54 | Kareem Jackson | 62.45 | 46.73 | 68.77 | 621 | Texans |
| 96 | 55 | Antwaun Molden | 62.28 | 54.96 | 69.38 | 345 | Patriots |

### Rotation/backup (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 97 | 1 | Terence Newman | 61.74 | 49.47 | 67.05 | 799 | Cowboys |
| 98 | 2 | A.J. Jefferson | 61.62 | 51.49 | 70.08 | 789 | Cardinals |
| 99 | 3 | Brandon McDonald | 61.01 | 58.71 | 61.62 | 153 | Lions |
| 100 | 4 | Terrence McGee | 60.51 | 55.98 | 68.61 | 280 | Bills |
| 101 | 5 | Ronde Barber | 60.15 | 43.00 | 67.42 | 968 | Buccaneers |
| 102 | 6 | Patrick Peterson | 60.05 | 47.50 | 64.25 | 1098 | Cardinals |
| 103 | 7 | Reggie Corner | 59.56 | 53.72 | 66.45 | 128 | Bills |
| 104 | 8 | Prince Amukamara | 59.30 | 59.75 | 62.13 | 199 | Giants |
| 105 | 9 | Dante Hughes | 58.87 | 51.56 | 62.71 | 429 | Chargers |
| 106 | 10 | Marcus Trufant | 58.52 | 52.02 | 66.51 | 268 | Seahawks |
| 107 | 11 | Marcus Sherels | 58.08 | 48.98 | 63.23 | 296 | Vikings |
| 108 | 12 | Chris Johnson | 57.34 | 47.92 | 68.83 | 157 | Raiders |
| 109 | 13 | Kevin Thomas | 57.29 | 58.27 | 59.77 | 428 | Colts |
| 110 | 14 | Zackary Bowman | 57.25 | 54.48 | 61.44 | 106 | Bears |
| 111 | 15 | Asher Allen | 56.45 | 45.79 | 63.18 | 520 | Vikings |
| 112 | 16 | Kelvin Hayden | 56.06 | 44.77 | 66.59 | 218 | Falcons |
| 113 | 17 | Cassius Vaughn | 55.35 | 52.68 | 65.20 | 232 | Broncos |
| 114 | 18 | Kevin Rutland | 52.35 | 50.01 | 60.61 | 266 | Jaguars |
| 115 | 19 | Ashton Youboty | 48.49 | 47.92 | 55.50 | 301 | Jaguars |
| 116 | 20 | Terrence Johnson | 48.43 | 51.31 | 49.64 | 277 | Colts |
| 117 | 21 | Justin King | 46.60 | 40.00 | 52.57 | 706 | Rams |
| 118 | 22 | Chimdi Chekwa | 45.00 | 52.20 | 49.52 | 123 | Raiders |

## DI — Defensive Interior

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | J.J. Watt | 91.35 | 86.43 | 90.46 | 878 | Texans |
| 2 | 2 | Geno Atkins | 87.07 | 84.95 | 84.32 | 772 | Bengals |
| 3 | 3 | Justin Smith | 85.82 | 84.51 | 82.53 | 1096 | 49ers |
| 4 | 4 | Haloti Ngata | 83.04 | 85.72 | 77.08 | 918 | Ravens |
| 5 | 5 | Marcell Dareus | 82.87 | 79.20 | 81.15 | 734 | Bills |
| 6 | 6 | Calais Campbell | 81.77 | 72.25 | 84.34 | 998 | Cardinals |
| 7 | 7 | Antonio Garay | 80.76 | 65.82 | 86.55 | 530 | Chargers |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Sammie Lee Hill | 78.64 | 69.61 | 80.87 | 452 | Lions |
| 9 | 2 | Jurrell Casey | 78.01 | 71.50 | 78.19 | 654 | Titans |
| 10 | 3 | Ndamukong Suh | 78.01 | 69.56 | 80.13 | 809 | Lions |
| 11 | 4 | Sione Pouha | 78.01 | 74.74 | 76.02 | 619 | Jets |
| 12 | 5 | Cullen Jenkins | 77.46 | 65.09 | 81.93 | 631 | Eagles |
| 13 | 6 | Brodrick Bunkley | 77.15 | 69.59 | 78.40 | 528 | Broncos |
| 14 | 7 | Richard Seymour | 76.94 | 71.22 | 77.75 | 845 | Raiders |
| 15 | 8 | David Carter | 76.93 | 67.88 | 78.80 | 240 | Cardinals |
| 16 | 9 | Vonnie Holliday | 76.87 | 60.87 | 83.75 | 152 | Cardinals |
| 17 | 10 | Tyson Jackson | 76.52 | 73.33 | 75.64 | 595 | Chiefs |
| 18 | 11 | Kevin Williams | 76.37 | 78.21 | 72.27 | 822 | Vikings |
| 19 | 12 | Paul Soliai | 76.23 | 68.92 | 76.93 | 436 | Dolphins |
| 20 | 13 | Muhammad Wilkerson | 76.17 | 59.97 | 82.80 | 599 | Jets |
| 21 | 14 | Sean Lissemore | 76.13 | 60.56 | 82.35 | 275 | Cowboys |
| 22 | 15 | Cory Redding | 75.71 | 60.54 | 81.65 | 608 | Ravens |
| 23 | 16 | Derek Landri | 74.98 | 68.83 | 77.51 | 348 | Eagles |
| 24 | 17 | Ray McDonald | 74.67 | 73.93 | 70.99 | 983 | 49ers |
| 25 | 18 | John Henderson | 74.63 | 71.18 | 77.44 | 343 | Raiders |
| 26 | 19 | Steve McLendon | 74.51 | 66.86 | 80.78 | 217 | Steelers |
| 27 | 20 | Alan Branch | 74.44 | 71.06 | 73.18 | 673 | Seahawks |

### Starter (82 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Kenyon Coleman | 73.93 | 57.71 | 80.57 | 413 | Cowboys |
| 29 | 2 | Cam Thomas | 73.86 | 65.96 | 78.86 | 380 | Chargers |
| 30 | 3 | Kyle Williams | 73.68 | 63.97 | 83.15 | 218 | Bills |
| 31 | 4 | Chris Canty | 73.66 | 69.82 | 72.06 | 822 | Giants |
| 32 | 5 | Vince Wilfork | 73.47 | 62.82 | 76.41 | 1149 | Patriots |
| 33 | 6 | Randy Starks | 73.01 | 66.21 | 73.37 | 657 | Dolphins |
| 34 | 7 | Nick Fairley | 72.99 | 68.53 | 77.00 | 263 | Lions |
| 35 | 8 | Glenn Dorsey | 72.98 | 68.85 | 72.21 | 623 | Chiefs |
| 36 | 9 | Karl Klug | 72.87 | 61.17 | 76.51 | 507 | Titans |
| 37 | 10 | Mike Devito | 72.83 | 65.28 | 76.29 | 452 | Jets |
| 38 | 11 | Henry Melton | 72.14 | 59.74 | 76.89 | 621 | Bears |
| 39 | 12 | Cameron Heyward | 71.81 | 58.60 | 76.45 | 242 | Steelers |
| 40 | 13 | Brandon Mebane | 71.52 | 63.78 | 73.30 | 734 | Seahawks |
| 41 | 14 | Leger Douzable | 71.20 | 63.87 | 72.31 | 420 | Jaguars |
| 42 | 15 | Linval Joseph | 71.08 | 63.23 | 76.44 | 746 | Giants |
| 43 | 16 | Tommy Kelly | 70.92 | 64.59 | 70.97 | 840 | Raiders |
| 44 | 17 | Kellen Heard | 70.76 | 55.94 | 77.51 | 254 | Bills |
| 45 | 18 | Wallace Gilberry | 70.65 | 58.34 | 74.69 | 366 | Chiefs |
| 46 | 19 | Ropati Pitoitua | 70.57 | 55.11 | 78.80 | 338 | Jets |
| 47 | 20 | Earl Mitchell | 70.46 | 57.64 | 75.22 | 320 | Texans |
| 48 | 21 | Jay Ratliff | 70.39 | 65.83 | 69.27 | 722 | Cowboys |
| 49 | 22 | Brett Keisel | 70.33 | 65.40 | 70.88 | 769 | Steelers |
| 50 | 23 | Ricky Jean Francois | 70.27 | 63.36 | 70.71 | 306 | 49ers |
| 51 | 24 | Tony Hargrove | 70.18 | 61.27 | 73.39 | 298 | Seahawks |
| 52 | 25 | Jason Hatcher | 70.13 | 58.13 | 77.10 | 416 | Cowboys |
| 53 | 26 | Ahtyba Rubin | 69.91 | 61.97 | 71.03 | 929 | Browns |
| 54 | 27 | C.J. Wilson | 69.77 | 55.59 | 75.05 | 405 | Packers |
| 55 | 28 | Shaun Smith | 69.67 | 54.87 | 76.02 | 283 | Titans |
| 56 | 29 | Fred Evans | 69.54 | 57.30 | 76.67 | 293 | Vikings |
| 57 | 30 | Stephen Bowen | 69.08 | 56.22 | 73.49 | 793 | Commanders |
| 58 | 31 | Mike Patterson | 68.79 | 57.79 | 72.61 | 638 | Eagles |
| 59 | 32 | Vance Walker | 68.79 | 55.97 | 73.17 | 384 | Falcons |
| 60 | 33 | Christian Ballard | 68.70 | 58.16 | 71.56 | 237 | Vikings |
| 61 | 34 | Amobi Okoye | 68.66 | 60.58 | 69.88 | 593 | Bears |
| 62 | 35 | Phil Taylor Sr. | 68.63 | 58.20 | 71.42 | 732 | Browns |
| 63 | 36 | Ryan Pickett | 68.62 | 56.26 | 73.34 | 499 | Packers |
| 64 | 37 | Domata Peko Sr. | 68.58 | 55.23 | 73.31 | 687 | Bengals |
| 65 | 38 | Shaun Cody | 68.55 | 56.17 | 72.63 | 437 | Texans |
| 66 | 39 | C.J. Mosley | 68.49 | 57.39 | 75.50 | 318 | Jaguars |
| 67 | 40 | Darnell Dockett | 68.46 | 54.47 | 74.00 | 1001 | Cardinals |
| 68 | 41 | Dan Williams | 68.39 | 67.23 | 69.29 | 237 | Cardinals |
| 69 | 42 | Letroy Guion | 68.34 | 61.77 | 68.94 | 483 | Vikings |
| 70 | 43 | Antonio Smith | 68.04 | 53.27 | 73.72 | 808 | Texans |
| 71 | 44 | Gerald McCoy | 67.98 | 72.59 | 68.43 | 218 | Buccaneers |
| 72 | 45 | Stephen Paea | 67.77 | 62.23 | 72.50 | 309 | Bears |
| 73 | 46 | Tony McDaniel | 67.44 | 55.62 | 74.16 | 325 | Dolphins |
| 74 | 47 | Marcus Thomas | 67.05 | 55.32 | 72.00 | 604 | Broncos |
| 75 | 48 | Chris Neild | 67.04 | 57.62 | 70.19 | 160 | Commanders |
| 76 | 49 | Red Bryant | 66.87 | 61.16 | 70.02 | 707 | Seahawks |
| 77 | 50 | Shaun Ellis | 66.86 | 52.04 | 72.57 | 415 | Patriots |
| 78 | 51 | Tim Jamison | 66.80 | 54.95 | 70.54 | 366 | Texans |
| 79 | 52 | Gary Gibson | 66.70 | 62.87 | 65.08 | 384 | Rams |
| 80 | 53 | Pat Sims | 66.41 | 54.84 | 73.99 | 291 | Bengals |
| 81 | 54 | Adam Carriker | 66.20 | 52.51 | 71.16 | 604 | Commanders |
| 82 | 55 | Jonathan Babineaux | 66.19 | 61.09 | 66.72 | 593 | Falcons |
| 83 | 56 | Sedrick Ellis | 65.94 | 63.19 | 63.61 | 791 | Saints |
| 84 | 57 | Terrance Knighton | 65.82 | 59.84 | 67.59 | 528 | Jaguars |
| 85 | 58 | Andre Neblett | 65.77 | 56.37 | 73.98 | 369 | Panthers |
| 86 | 59 | Rocky Bernard | 65.58 | 55.19 | 69.13 | 457 | Giants |
| 87 | 60 | Corey Peters | 65.48 | 52.71 | 69.82 | 657 | Falcons |
| 88 | 61 | Kyle Love | 65.37 | 57.60 | 68.73 | 681 | Patriots |
| 89 | 62 | Barry Cofield | 65.27 | 58.16 | 65.85 | 762 | Commanders |
| 90 | 63 | Isaac Sopoaga | 65.10 | 54.80 | 67.80 | 482 | 49ers |
| 91 | 64 | Kendall Langford | 65.06 | 56.04 | 66.91 | 538 | Dolphins |
| 92 | 65 | Tommie Harris | 64.90 | 58.39 | 67.03 | 274 | Chargers |
| 93 | 66 | Jared Odrick | 64.77 | 61.71 | 68.51 | 580 | Dolphins |
| 94 | 67 | Tyson Alualu | 64.73 | 53.31 | 68.18 | 851 | Jaguars |
| 95 | 68 | Anthony Adams | 64.32 | 52.53 | 71.26 | 280 | Bears |
| 96 | 69 | Allen Bailey | 64.31 | 58.69 | 64.92 | 289 | Chiefs |
| 97 | 70 | Peria Jerry | 64.24 | 52.93 | 67.62 | 362 | Falcons |
| 98 | 71 | Corey Williams | 64.17 | 51.92 | 68.17 | 745 | Lions |
| 99 | 72 | Brandon Deaderick | 63.74 | 54.24 | 70.20 | 376 | Patriots |
| 100 | 73 | Trevor Laws | 63.56 | 54.79 | 66.54 | 342 | Eagles |
| 101 | 74 | Jacques Cesaire | 63.24 | 49.36 | 70.92 | 242 | Chargers |
| 102 | 75 | Marcus Spears | 63.09 | 52.50 | 69.11 | 385 | Cowboys |
| 103 | 76 | Brian Price | 63.03 | 53.37 | 70.25 | 494 | Buccaneers |
| 104 | 77 | B.J. Raji | 62.89 | 54.71 | 64.17 | 924 | Packers |
| 105 | 78 | Josh Price-Brent | 62.84 | 63.42 | 61.53 | 133 | Cowboys |
| 106 | 79 | Tom Johnson | 62.70 | 54.18 | 65.25 | 326 | Saints |
| 107 | 80 | Andre Fluellen | 62.57 | 55.25 | 65.24 | 205 | Lions |
| 108 | 81 | Ricardo Mathews | 62.20 | 56.66 | 68.25 | 334 | Colts |
| 109 | 82 | Marcus Dixon | 62.08 | 52.38 | 69.47 | 422 | Jets |

### Rotation/backup (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 110 | 1 | Remi Ayodele | 61.86 | 54.83 | 62.38 | 252 | Vikings |
| 111 | 2 | Sen'Derrick Marks | 61.78 | 52.62 | 65.28 | 411 | Titans |
| 112 | 3 | Darell Scott | 61.65 | 58.78 | 64.08 | 242 | Rams |
| 113 | 4 | Fili Moala | 61.24 | 48.56 | 66.82 | 480 | Colts |
| 114 | 5 | Casey Hampton | 61.11 | 55.96 | 61.68 | 444 | Steelers |
| 115 | 6 | Clinton McDonald | 60.94 | 56.47 | 63.53 | 425 | Seahawks |
| 116 | 7 | Kedric Golston | 60.92 | 56.76 | 65.26 | 170 | Commanders |
| 117 | 8 | Roy Miller | 60.81 | 49.26 | 64.35 | 494 | Buccaneers |
| 118 | 9 | Antonio Johnson | 60.79 | 51.42 | 63.65 | 502 | Colts |
| 119 | 10 | Vaughn Martin | 60.62 | 50.81 | 65.35 | 611 | Chargers |
| 120 | 11 | Ryan McBean | 60.49 | 46.76 | 65.48 | 684 | Broncos |
| 121 | 12 | Dwan Edwards | 59.82 | 40.00 | 68.87 | 736 | Bills |
| 122 | 13 | Terrence Cody | 59.17 | 51.39 | 60.57 | 532 | Ravens |
| 123 | 14 | Corey Liuget | 59.06 | 51.95 | 60.67 | 452 | Chargers |
| 124 | 15 | Darrion Scott | 58.87 | 57.06 | 67.24 | 111 | Commanders |
| 125 | 16 | Drake Nevis | 58.52 | 59.42 | 69.72 | 163 | Colts |
| 126 | 17 | Arthur Jones | 58.48 | 53.95 | 62.80 | 276 | Ravens |
| 127 | 18 | Igor Olshansky | 58.23 | 54.71 | 61.61 | 100 | Dolphins |
| 128 | 19 | Matt Toeaina | 56.61 | 52.17 | 58.00 | 390 | Bears |
| 129 | 20 | Brandon McKinney | 56.59 | 56.95 | 55.31 | 203 | Ravens |
| 130 | 21 | Tim Bulman | 56.18 | 53.39 | 58.04 | 136 | Texans |
| 131 | 22 | Nick Eason | 55.13 | 47.45 | 56.09 | 249 | Cardinals |
| 132 | 23 | Ogemdi Nwagbuo | 55.05 | 57.75 | 54.69 | 159 | Panthers |
| 133 | 24 | Sione Fua | 54.81 | 51.63 | 57.96 | 406 | Panthers |
| 134 | 25 | Terrell McClain | 54.17 | 47.22 | 58.80 | 471 | Panthers |
| 135 | 26 | Frank Okam | 52.70 | 49.46 | 57.99 | 296 | Buccaneers |
| 136 | 27 | Mitch Unrein | 52.60 | 53.07 | 49.16 | 105 | Broncos |
| 137 | 28 | John McCargo | 52.40 | 55.61 | 64.60 | 109 | Buccaneers |
| 138 | 29 | Aaron Smith | 51.43 | 49.89 | 60.00 | 172 | Steelers |
| 139 | 30 | Phillip Merling | 49.96 | 55.47 | 50.32 | 199 | Dolphins |
| 140 | 31 | Frank Kearse | 48.14 | 56.86 | 51.57 | 170 | Panthers |
| 141 | 32 | Kevin Vickerson | 45.00 | 51.34 | 49.58 | 198 | Broncos |

## ED — Edge

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 93.45 | 95.83 | 87.70 | 985 | Broncos |
| 2 | 2 | DeMarcus Ware | 91.96 | 88.78 | 89.92 | 880 | Cowboys |
| 3 | 3 | Cameron Wake | 87.55 | 82.59 | 86.69 | 875 | Dolphins |
| 4 | 4 | Charles Johnson | 87.39 | 84.18 | 86.01 | 794 | Panthers |
| 5 | 5 | Jason Babin | 87.04 | 78.06 | 88.86 | 697 | Eagles |
| 6 | 6 | Terrell Suggs | 86.94 | 87.73 | 82.24 | 1117 | Ravens |
| 7 | 7 | Jared Allen | 82.75 | 77.03 | 82.39 | 1005 | Vikings |
| 8 | 8 | Chris Long | 81.50 | 76.37 | 80.75 | 920 | Rams |
| 9 | 9 | Mario Williams | 81.23 | 82.75 | 84.38 | 217 | Texans |
| 10 | 10 | Jason Pierre-Paul | 80.23 | 87.05 | 71.52 | 1173 | Giants |
| 11 | 11 | Carlos Dunlap | 80.03 | 73.66 | 83.63 | 434 | Bengals |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Cliff Avril | 79.21 | 72.40 | 80.75 | 856 | Lions |
| 13 | 2 | Mark Anderson | 78.60 | 69.20 | 81.08 | 625 | Patriots |
| 14 | 3 | Julius Peppers | 78.41 | 77.76 | 74.68 | 887 | Bears |
| 15 | 4 | Chris Clemons | 78.35 | 67.05 | 81.72 | 929 | Seahawks |
| 16 | 5 | Justin Tuck | 76.13 | 67.56 | 77.68 | 797 | Giants |
| 17 | 6 | Justin Houston | 75.88 | 65.11 | 78.89 | 754 | Chiefs |
| 18 | 7 | Ryan Kerrigan | 75.07 | 62.80 | 79.09 | 1026 | Commanders |

### Starter (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Lawrence Jackson | 73.94 | 70.35 | 76.72 | 363 | Lions |
| 20 | 2 | Ray Edwards | 73.75 | 69.03 | 73.51 | 718 | Falcons |
| 21 | 3 | Jabaal Sheard | 73.62 | 66.34 | 74.31 | 949 | Browns |
| 22 | 4 | Juqua Parker | 72.72 | 66.24 | 76.26 | 254 | Eagles |
| 23 | 5 | Phillip Hunt | 72.71 | 65.25 | 80.82 | 177 | Eagles |
| 24 | 6 | Adrian Clayborn | 72.19 | 63.12 | 74.07 | 849 | Buccaneers |
| 25 | 7 | Brooks Reed | 71.97 | 66.02 | 71.77 | 902 | Texans |
| 26 | 8 | Anthony Spencer | 71.39 | 68.30 | 69.28 | 931 | Cowboys |
| 27 | 9 | Darryl Tapp | 70.90 | 68.81 | 71.13 | 304 | Eagles |
| 28 | 10 | Everson Griffen | 70.61 | 65.03 | 72.90 | 271 | Vikings |
| 29 | 11 | Mathias Kiwanuka | 70.58 | 60.85 | 77.98 | 948 | Giants |
| 30 | 12 | Shaun Phillips | 69.13 | 58.19 | 74.85 | 626 | Chargers |
| 31 | 13 | Parys Haralson | 68.85 | 58.86 | 71.72 | 570 | 49ers |
| 32 | 14 | Jeremy Mincey | 68.82 | 64.88 | 67.67 | 950 | Jaguars |
| 33 | 15 | Robert Quinn | 67.51 | 60.96 | 68.75 | 561 | Rams |
| 34 | 16 | Jarvis Moss | 67.26 | 59.71 | 71.00 | 298 | Raiders |
| 35 | 17 | Matt Shaughnessy | 66.85 | 64.38 | 72.79 | 143 | Raiders |
| 36 | 18 | James Hall | 66.63 | 52.99 | 72.20 | 718 | Rams |
| 37 | 19 | Matt Roth | 66.00 | 60.24 | 70.22 | 401 | Jaguars |
| 38 | 20 | O'Brien Schofield | 65.94 | 57.24 | 70.31 | 433 | Cardinals |
| 39 | 21 | Brian Robison | 65.94 | 60.12 | 65.66 | 900 | Vikings |
| 40 | 22 | Antwan Applewhite | 65.64 | 59.83 | 68.35 | 335 | Panthers |
| 41 | 23 | Greg Hardy | 65.09 | 63.37 | 62.46 | 890 | Panthers |
| 42 | 24 | Michael Bennett | 64.71 | 68.96 | 60.17 | 610 | Buccaneers |
| 43 | 25 | Jonathan Fanene | 64.37 | 58.88 | 69.33 | 472 | Bengals |
| 44 | 26 | Turk McBride | 63.00 | 60.55 | 66.06 | 205 | Saints |
| 45 | 27 | Robert Ayers | 62.25 | 62.91 | 59.60 | 732 | Broncos |
| 46 | 28 | Kroy Biermann | 62.23 | 59.32 | 60.00 | 556 | Falcons |

### Rotation/backup (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 47 | 1 | Jamaal Anderson | 61.83 | 61.15 | 58.76 | 412 | Colts |
| 48 | 2 | Israel Idonije | 61.74 | 55.00 | 62.06 | 915 | Bears |
| 49 | 3 | Frostee Rucker | 61.58 | 58.33 | 62.31 | 475 | Bengals |
| 50 | 4 | Michael Johnson | 60.81 | 58.89 | 57.93 | 708 | Bengals |
| 51 | 5 | Chris Kelsay | 60.64 | 52.30 | 64.64 | 674 | Bills |
| 52 | 6 | Kyle Vanden Bosch | 60.26 | 51.45 | 63.92 | 799 | Lions |
| 53 | 7 | Cameron Jordan | 60.09 | 62.20 | 54.51 | 648 | Saints |
| 54 | 8 | Jason Hunter | 59.66 | 56.52 | 57.59 | 406 | Broncos |
| 55 | 9 | Dave Ball | 59.05 | 52.79 | 62.30 | 647 | Titans |
| 56 | 10 | William Hayes | 57.91 | 58.89 | 57.78 | 330 | Titans |
| 57 | 11 | Robert Geathers | 56.80 | 55.84 | 53.92 | 534 | Bengals |
| 58 | 12 | C.J. Ah You | 56.54 | 54.38 | 58.37 | 246 | Rams |
| 59 | 13 | Danny Batten | 55.77 | 58.30 | 55.12 | 224 | Bills |
| 60 | 14 | Jerry Hughes | 55.55 | 58.82 | 55.19 | 150 | Colts |
| 61 | 15 | Dave Tollefson | 55.52 | 51.30 | 56.11 | 551 | Giants |
| 62 | 16 | Austen Lane | 55.06 | 59.06 | 56.69 | 126 | Jaguars |
| 63 | 17 | Jason Jones | 54.30 | 51.09 | 53.57 | 675 | Titans |
| 64 | 18 | Trevor Scott | 54.11 | 55.61 | 51.95 | 243 | Raiders |
| 65 | 19 | Eugene Sims | 52.63 | 56.40 | 50.63 | 276 | Rams |
| 66 | 20 | Emmanuel Stephens | 51.93 | 58.60 | 49.57 | 148 | Browns |
| 67 | 21 | Mario Addison | 46.58 | 58.08 | 45.61 | 131 | Colts |

## G — Guard

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Evan Mathis | 94.33 | 92.80 | 91.18 | 996 | Eagles |
| 2 | 2 | Marshal Yanda | 90.42 | 84.70 | 90.06 | 1172 | Ravens |
| 3 | 3 | Mike Brisiel | 89.46 | 81.68 | 90.48 | 982 | Texans |
| 4 | 4 | Carl Nicks | 88.81 | 83.00 | 88.52 | 1320 | Saints |
| 5 | 5 | Josh Sitton | 88.59 | 81.39 | 89.22 | 924 | Packers |
| 6 | 6 | Brian Waters | 87.75 | 82.00 | 87.41 | 1310 | Patriots |
| 7 | 7 | Steve Hutchinson | 87.22 | 80.56 | 87.49 | 889 | Vikings |
| 8 | 8 | Joe Berger | 87.04 | 76.27 | 90.05 | 493 | Vikings |
| 9 | 9 | Andy Levitre | 86.88 | 81.40 | 86.36 | 1043 | Bills |
| 10 | 10 | Mike Iupati | 86.31 | 79.60 | 86.61 | 1147 | 49ers |
| 11 | 11 | Kris Dielman | 86.02 | 77.17 | 87.76 | 419 | Chargers |
| 12 | 12 | Travelle Wharton | 85.58 | 77.60 | 86.74 | 1000 | Panthers |
| 13 | 13 | Logan Mankins | 85.23 | 75.90 | 87.29 | 1162 | Patriots |
| 14 | 14 | Jahri Evans | 84.36 | 76.50 | 85.44 | 1283 | Saints |
| 15 | 15 | Brandon Moore | 83.92 | 75.70 | 85.23 | 1078 | Jets |
| 16 | 16 | Jake Scott | 83.28 | 76.30 | 83.77 | 1022 | Titans |
| 17 | 17 | T.J. Lang | 83.16 | 73.60 | 85.37 | 1109 | Packers |
| 18 | 18 | Richie Incognito | 83.02 | 73.95 | 84.90 | 928 | Dolphins |
| 19 | 19 | Harvey Dahl | 82.85 | 74.00 | 84.59 | 1055 | Rams |
| 20 | 20 | Jon Asamoah | 81.86 | 74.00 | 82.94 | 1051 | Chiefs |
| 21 | 21 | Uche Nwaneri | 81.83 | 73.90 | 82.95 | 1041 | Jaguars |
| 22 | 22 | Ben Grubbs | 81.82 | 73.99 | 82.87 | 794 | Ravens |
| 23 | 23 | Daryn Colledge | 81.50 | 71.80 | 83.80 | 990 | Cardinals |
| 24 | 24 | Ramon Foster | 81.13 | 73.44 | 82.09 | 976 | Steelers |
| 25 | 25 | Rob Sims | 80.79 | 73.00 | 81.81 | 1157 | Lions |
| 26 | 26 | Bobbie Williams | 80.64 | 70.23 | 83.41 | 560 | Bengals |
| 27 | 27 | Antoine Caldwell | 80.39 | 67.01 | 85.15 | 221 | Texans |
| 28 | 28 | Stefen Wisniewski | 80.35 | 71.90 | 81.81 | 1073 | Raiders |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 29 | 1 | Stephen Peterman | 79.88 | 71.30 | 81.44 | 1157 | Lions |
| 30 | 2 | Chad Rinehart | 79.64 | 73.31 | 79.69 | 853 | Bills |
| 31 | 3 | Geoff Hangartner | 79.20 | 70.30 | 80.96 | 1046 | Panthers |
| 32 | 4 | Justin Blalock | 79.12 | 71.50 | 80.03 | 1191 | Falcons |
| 33 | 5 | Kyle Kosier | 78.94 | 70.50 | 80.40 | 1018 | Cowboys |
| 34 | 6 | Shawn Lauvao | 78.73 | 67.70 | 81.92 | 1013 | Browns |
| 35 | 7 | Wade Smith | 78.66 | 70.20 | 80.13 | 1203 | Texans |
| 36 | 8 | Matt Slauson | 78.45 | 68.90 | 80.65 | 1088 | Jets |
| 37 | 9 | Nate Livings | 77.43 | 67.70 | 79.75 | 1135 | Bengals |
| 38 | 10 | Louis Vasquez | 76.34 | 67.43 | 78.11 | 894 | Chargers |
| 39 | 11 | Chris Kemoeatu | 76.03 | 64.39 | 79.62 | 758 | Steelers |
| 40 | 12 | Kraig Urbik | 75.40 | 71.17 | 74.06 | 750 | Bills |
| 41 | 13 | Cooper Carlisle | 75.32 | 65.90 | 77.44 | 1072 | Raiders |
| 42 | 14 | Maurice Hurt | 75.24 | 63.88 | 78.65 | 548 | Commanders |
| 43 | 15 | Davin Joseph | 75.15 | 66.50 | 76.75 | 1022 | Buccaneers |
| 44 | 16 | Jacob Bell | 75.15 | 63.46 | 78.78 | 713 | Rams |
| 45 | 17 | Rex Hadnot | 75.06 | 65.40 | 77.34 | 985 | Cardinals |
| 46 | 18 | Chris Snee | 74.97 | 64.60 | 77.72 | 1277 | Giants |
| 47 | 19 | Adam Snyder | 74.96 | 63.96 | 78.12 | 964 | 49ers |
| 48 | 20 | Joe Reitz | 74.82 | 63.37 | 78.28 | 530 | Colts |
| 49 | 21 | Zane Beadles | 74.81 | 64.10 | 77.78 | 1195 | Broncos |
| 50 | 22 | Artis Hicks | 74.07 | 58.81 | 80.07 | 191 | Browns |

### Starter (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 51 | 1 | Chris Kuper | 73.77 | 63.60 | 76.38 | 997 | Broncos |
| 52 | 2 | Mike McGlynn | 73.64 | 60.77 | 78.05 | 408 | Bengals |
| 53 | 3 | John Jerry | 73.10 | 62.35 | 76.10 | 357 | Dolphins |
| 54 | 4 | Chris Spencer | 72.94 | 64.97 | 74.09 | 901 | Bears |
| 55 | 5 | Chris Chester | 72.58 | 63.30 | 74.60 | 1081 | Commanders |
| 56 | 6 | Ted Larsen | 72.27 | 57.44 | 77.99 | 229 | Buccaneers |
| 57 | 7 | Kevin Boothe | 72.10 | 61.00 | 75.33 | 980 | Giants |
| 58 | 8 | Jason Pinkston | 72.02 | 61.10 | 75.13 | 1063 | Browns |
| 59 | 9 | Danny Watkins | 71.53 | 63.22 | 72.90 | 787 | Eagles |
| 60 | 10 | Chris Williams | 70.56 | 59.19 | 73.98 | 538 | Bears |
| 61 | 11 | Chilo Rachal | 69.57 | 56.94 | 73.82 | 240 | 49ers |
| 62 | 12 | Colin Brown | 69.51 | 62.56 | 69.97 | 134 | Bills |
| 63 | 13 | Donald Thomas | 69.42 | 60.76 | 71.03 | 103 | Patriots |
| 64 | 14 | Tyronne Green | 69.29 | 58.80 | 72.11 | 556 | Chargers |
| 65 | 15 | Steve Schilling | 68.96 | 62.36 | 69.19 | 137 | Chargers |
| 66 | 16 | Bill Nagy | 68.00 | 60.05 | 69.14 | 274 | Cowboys |
| 67 | 17 | Mitch Petrus | 67.87 | 58.24 | 70.12 | 224 | Giants |
| 68 | 18 | Russ Hochstein | 67.84 | 57.71 | 70.42 | 199 | Broncos |
| 69 | 19 | Bryan Mattison | 67.75 | 56.66 | 70.97 | 269 | Rams |
| 70 | 20 | Will Rackley | 67.37 | 53.55 | 72.41 | 940 | Jaguars |
| 71 | 21 | Garrett Reynolds | 66.94 | 52.94 | 72.11 | 511 | Falcons |
| 72 | 22 | John Moffitt | 66.06 | 54.72 | 69.45 | 502 | Seahawks |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 73 | 1 | Clint Boling | 60.33 | 55.09 | 59.66 | 168 | Bengals |

## HB — Running Back

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Darren Sproles | 82.51 | 79.60 | 80.29 | 433 | Saints |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 2 | 1 | Fred Jackson | 79.71 | 77.89 | 76.76 | 238 | Bills |
| 3 | 2 | Pierre Thomas | 78.01 | 81.83 | 71.29 | 231 | Saints |
| 4 | 3 | Isaac Redman | 77.97 | 75.27 | 75.61 | 131 | Steelers |
| 5 | 4 | Jonathan Stewart | 77.37 | 77.17 | 73.34 | 254 | Panthers |
| 6 | 5 | Matt Forte | 77.32 | 82.40 | 69.77 | 256 | Bears |
| 7 | 6 | Adrian Peterson | 77.22 | 78.85 | 71.96 | 165 | Vikings |
| 8 | 7 | Arian Foster | 76.65 | 81.09 | 69.53 | 305 | Texans |
| 9 | 8 | LeSean McCoy | 75.47 | 73.80 | 72.42 | 388 | Eagles |
| 10 | 9 | Darren McFadden | 75.28 | 69.78 | 74.78 | 104 | Raiders |
| 11 | 10 | Ben Tate | 74.61 | 68.05 | 74.81 | 108 | Texans |

### Starter (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | LeGarrette Blount | 73.88 | 66.08 | 74.91 | 116 | Buccaneers |
| 13 | 2 | DeMarco Murray | 73.48 | 67.18 | 73.51 | 148 | Cowboys |
| 14 | 3 | C.J. Spiller | 73.38 | 72.79 | 69.61 | 246 | Bills |
| 15 | 4 | Ahmad Bradshaw | 73.29 | 72.48 | 69.67 | 242 | Giants |
| 16 | 5 | Ryan Mathews | 73.27 | 70.09 | 71.22 | 229 | Chargers |
| 17 | 6 | Maurice Jones-Drew | 73.18 | 73.56 | 68.76 | 302 | Jaguars |
| 18 | 7 | Toby Gerhart | 72.61 | 70.93 | 69.57 | 157 | Vikings |
| 19 | 8 | Ray Rice | 72.04 | 75.70 | 65.43 | 412 | Ravens |
| 20 | 9 | Marshawn Lynch | 71.50 | 71.44 | 67.37 | 185 | Seahawks |
| 21 | 10 | Felix Jones | 71.23 | 67.24 | 69.72 | 134 | Cowboys |
| 22 | 11 | Rashard Mendenhall | 71.18 | 77.02 | 63.12 | 162 | Steelers |
| 23 | 12 | James Starks | 70.86 | 72.61 | 65.52 | 209 | Packers |
| 24 | 13 | DeAngelo Williams | 70.44 | 66.38 | 68.98 | 179 | Panthers |
| 25 | 14 | Michael Turner | 70.40 | 67.74 | 68.01 | 188 | Falcons |
| 26 | 15 | Brandon Jacobs | 70.01 | 67.55 | 67.49 | 117 | Giants |
| 27 | 16 | Willis McGahee | 68.98 | 69.82 | 64.26 | 143 | Broncos |
| 28 | 17 | Donald Brown | 68.74 | 65.90 | 66.46 | 171 | Colts |
| 29 | 18 | Kendall Hunter | 68.64 | 70.28 | 63.38 | 131 | 49ers |
| 30 | 19 | Steven Jackson | 68.34 | 68.50 | 64.07 | 292 | Rams |
| 31 | 20 | Reggie Bush | 68.21 | 62.12 | 68.11 | 252 | Dolphins |
| 32 | 21 | Dexter McCluster | 67.62 | 66.98 | 63.88 | 234 | Chiefs |
| 33 | 22 | Peyton Hillis | 67.56 | 64.84 | 65.20 | 200 | Browns |
| 34 | 23 | Kahlil Bell | 67.46 | 64.10 | 65.54 | 100 | Bears |
| 35 | 24 | Chris Johnson | 67.25 | 63.93 | 65.30 | 328 | Titans |
| 36 | 25 | Kevin Smith | 66.95 | 64.13 | 64.67 | 180 | Lions |
| 37 | 26 | Roy Helu | 66.92 | 64.26 | 64.52 | 276 | Commanders |
| 38 | 27 | Frank Gore | 66.82 | 62.90 | 65.26 | 260 | 49ers |
| 39 | 28 | Michael Bush | 66.31 | 67.96 | 61.05 | 241 | Raiders |
| 40 | 29 | BenJarvus Green-Ellis | 66.23 | 68.68 | 60.43 | 146 | Patriots |
| 41 | 30 | Justin Forsett | 65.12 | 62.87 | 62.45 | 155 | Seahawks |
| 42 | 31 | Carnell Williams | 65.05 | 62.89 | 62.32 | 127 | Rams |
| 43 | 32 | LaDainian Tomlinson | 64.83 | 63.08 | 61.83 | 237 | Jets |
| 44 | 33 | Shonn Greene | 64.80 | 62.64 | 62.08 | 183 | Jets |
| 45 | 34 | Ryan Grant | 64.73 | 63.59 | 61.32 | 141 | Packers |
| 46 | 35 | Jacquizz Rodgers | 64.69 | 61.27 | 62.81 | 190 | Falcons |
| 47 | 36 | Jason Snelling | 64.47 | 61.66 | 62.17 | 144 | Falcons |
| 48 | 37 | Danny Woodhead | 64.20 | 61.63 | 61.74 | 253 | Patriots |
| 49 | 38 | Jahvid Best | 64.17 | 66.01 | 58.78 | 162 | Lions |
| 50 | 39 | Joseph Addai | 63.97 | 63.30 | 60.25 | 142 | Colts |
| 51 | 40 | Javon Ringer | 63.56 | 63.24 | 59.60 | 137 | Titans |
| 52 | 41 | Cedric Benson | 63.03 | 64.43 | 57.93 | 200 | Bengals |
| 53 | 42 | Lance Ball | 62.40 | 62.10 | 58.43 | 200 | Broncos |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 54 | 1 | Chris Wells | 61.43 | 57.45 | 59.91 | 190 | Cardinals |
| 55 | 2 | Maurice Morris | 61.28 | 57.97 | 59.32 | 187 | Lions |
| 56 | 3 | Chester Taylor | 59.30 | 58.24 | 55.84 | 123 | Cardinals |
| 57 | 4 | Montario Hardesty | 58.48 | 55.14 | 56.54 | 116 | Browns |
| 58 | 5 | Tashard Choice | 56.47 | 52.98 | 54.63 | 116 | Bills |

## LB — Linebacker

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Patrick Willis | 87.33 | 89.90 | 82.49 | 937 | 49ers |
| 2 | 2 | Paul Posluszny | 85.96 | 90.10 | 79.81 | 971 | Jaguars |
| 3 | 3 | NaVorro Bowman | 81.69 | 84.80 | 77.40 | 1159 | 49ers |
| 4 | 4 | Stephen Tulloch | 81.59 | 82.30 | 76.95 | 1165 | Lions |
| 5 | 5 | Ray Lewis | 80.80 | 82.80 | 76.60 | 916 | Ravens |
| 6 | 6 | D'Qwell Jackson | 80.21 | 80.70 | 75.71 | 1073 | Browns |
| 7 | 7 | Brian Urlacher | 80.18 | 79.80 | 76.26 | 1067 | Bears |

### Good (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Derrick Johnson | 79.69 | 81.80 | 74.12 | 1021 | Chiefs |
| 9 | 2 | London Fletcher | 79.01 | 80.40 | 73.91 | 1003 | Commanders |
| 10 | 3 | Daryl Smith | 78.80 | 79.30 | 74.30 | 897 | Jaguars |
| 11 | 4 | Brian Cushing | 78.52 | 81.80 | 73.73 | 1064 | Texans |
| 12 | 5 | Karlos Dansby | 77.56 | 78.70 | 73.41 | 905 | Dolphins |
| 13 | 6 | James Laurinaitis | 77.28 | 75.40 | 74.36 | 1071 | Rams |
| 14 | 7 | Desmond Bishop | 76.49 | 77.60 | 72.89 | 899 | Packers |
| 15 | 8 | Erin Henderson | 76.08 | 78.22 | 75.43 | 573 | Vikings |
| 16 | 9 | David Hawthorne | 75.51 | 75.00 | 72.72 | 988 | Seahawks |
| 17 | 10 | Daryl Washington | 75.35 | 75.70 | 71.60 | 986 | Cardinals |
| 18 | 11 | Mario Haggan | 74.77 | 68.71 | 75.30 | 188 | Broncos |
| 19 | 12 | Donald Butler | 74.73 | 73.28 | 71.53 | 634 | Chargers |
| 20 | 13 | Bart Scott | 74.70 | 75.15 | 70.23 | 667 | Jets |
| 21 | 14 | Curtis Lofton | 74.16 | 70.50 | 72.44 | 1053 | Falcons |
| 22 | 15 | David Harris | 74.06 | 72.20 | 71.14 | 961 | Jets |

### Starter (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Pat Angerer | 73.16 | 73.00 | 70.27 | 1027 | Colts |
| 24 | 2 | Brandon Spikes | 72.88 | 72.36 | 73.49 | 535 | Patriots |
| 25 | 3 | Jerod Mayo | 71.88 | 70.20 | 68.83 | 1103 | Patriots |
| 26 | 4 | Nick Barnett | 71.86 | 73.50 | 71.29 | 956 | Bills |
| 27 | 5 | E.J. Henderson | 71.67 | 67.70 | 70.15 | 862 | Vikings |
| 28 | 6 | Philip Wheeler | 70.51 | 67.73 | 70.14 | 532 | Colts |
| 29 | 7 | Lance Briggs | 70.38 | 65.70 | 69.33 | 1081 | Bears |
| 30 | 8 | Leroy Hill | 70.23 | 67.00 | 68.22 | 961 | Seahawks |
| 31 | 9 | Sean Lee | 69.86 | 68.70 | 68.28 | 838 | Cowboys |
| 32 | 10 | Dan Connor | 69.73 | 68.56 | 70.12 | 471 | Panthers |
| 33 | 11 | Sean Weatherspoon | 69.19 | 68.20 | 67.25 | 1043 | Falcons |
| 34 | 12 | Lawrence Timmons | 68.50 | 63.10 | 67.93 | 1029 | Steelers |
| 35 | 13 | Kelvin Sheppard | 68.22 | 65.90 | 68.73 | 435 | Bills |
| 36 | 14 | Bobby Carpenter | 68.01 | 63.24 | 67.03 | 248 | Lions |
| 37 | 15 | Jameel McClain | 67.91 | 65.78 | 65.16 | 814 | Ravens |
| 38 | 16 | Takeo Spikes | 67.78 | 62.60 | 67.06 | 918 | Chargers |
| 39 | 17 | Chase Blackburn | 67.64 | 69.26 | 71.25 | 375 | Giants |
| 40 | 18 | Brendon Ayanbadejo | 67.43 | 67.17 | 68.91 | 308 | Ravens |
| 41 | 19 | K.J. Wright | 67.13 | 64.64 | 65.66 | 562 | Seahawks |
| 42 | 20 | Chris Chamberlain | 67.08 | 64.01 | 67.69 | 597 | Rams |
| 43 | 21 | Colin McCarthy | 66.71 | 67.18 | 67.43 | 534 | Titans |
| 44 | 22 | James Anderson | 66.45 | 60.30 | 66.39 | 996 | Panthers |
| 45 | 23 | Brandon Johnson | 66.37 | 58.69 | 67.98 | 319 | Bengals |
| 46 | 24 | Russell Allen | 66.22 | 63.65 | 67.94 | 293 | Jaguars |
| 47 | 25 | DeMeco Ryans | 66.20 | 64.95 | 66.76 | 663 | Texans |
| 48 | 26 | Brian Rolle | 66.03 | 62.39 | 64.29 | 644 | Eagles |
| 49 | 27 | Rolando McClain | 65.94 | 61.30 | 65.90 | 1007 | Raiders |
| 50 | 28 | A.J. Hawk | 65.79 | 62.80 | 64.26 | 943 | Packers |
| 51 | 29 | Will Witherspoon | 65.67 | 60.72 | 64.80 | 665 | Titans |
| 52 | 30 | Keenan Clayton | 65.39 | 64.60 | 69.57 | 149 | Eagles |
| 53 | 31 | Aaron Curry | 64.99 | 57.15 | 66.05 | 695 | Raiders |
| 54 | 32 | Rey Maualuga | 64.96 | 60.47 | 65.08 | 728 | Bengals |
| 55 | 33 | Michael Boley | 64.91 | 60.90 | 63.42 | 1146 | Giants |
| 56 | 34 | Jovan Belcher | 64.87 | 59.47 | 64.30 | 649 | Chiefs |
| 57 | 35 | Greg Jones | 64.65 | 60.89 | 64.03 | 200 | Giants |
| 58 | 36 | Akeem Ayers | 64.60 | 59.20 | 64.03 | 818 | Titans |
| 59 | 37 | Akeem Jordan | 64.45 | 62.63 | 67.88 | 236 | Eagles |
| 60 | 38 | Stephen Nicholas | 64.23 | 62.34 | 65.22 | 281 | Falcons |
| 61 | 39 | Josh Mauga | 64.12 | 62.46 | 66.66 | 142 | Jets |
| 62 | 40 | Kevin Burnett | 64.05 | 57.10 | 64.51 | 1020 | Dolphins |
| 63 | 41 | Thomas Howard | 64.00 | 61.70 | 65.66 | 1032 | Bengals |
| 64 | 42 | Justin Durant | 63.62 | 62.10 | 64.12 | 626 | Lions |
| 65 | 43 | Perry Riley | 63.47 | 65.21 | 67.40 | 495 | Commanders |
| 66 | 44 | D.J. Williams | 63.26 | 57.40 | 63.65 | 1040 | Broncos |
| 67 | 45 | Jacquian Williams | 63.23 | 56.99 | 63.23 | 573 | Giants |
| 68 | 46 | D.J. Smith | 63.14 | 68.82 | 71.16 | 262 | Packers |
| 69 | 47 | Mike Peterson | 62.91 | 61.12 | 64.49 | 214 | Falcons |
| 70 | 48 | Kaluka Maiava | 62.86 | 62.00 | 61.21 | 268 | Browns |
| 71 | 49 | Dekoda Watson | 62.82 | 60.56 | 64.59 | 212 | Buccaneers |
| 72 | 50 | Andra Davis | 62.70 | 61.68 | 65.73 | 125 | Bills |
| 73 | 51 | Paris Lenon | 62.58 | 52.80 | 64.94 | 1086 | Cardinals |
| 74 | 52 | Bradie James | 62.36 | 55.74 | 62.61 | 402 | Cowboys |
| 75 | 53 | Stewart Bradley | 62.31 | 57.94 | 63.28 | 228 | Cardinals |
| 76 | 54 | Larry Grant | 62.29 | 62.94 | 65.38 | 226 | 49ers |

### Rotation/backup (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 77 | 1 | Dane Fletcher | 61.69 | 60.97 | 62.55 | 299 | Patriots |
| 78 | 2 | Wesley Woodyard | 61.63 | 55.85 | 64.84 | 639 | Broncos |
| 79 | 3 | James Farrior | 61.41 | 54.61 | 62.43 | 767 | Steelers |
| 80 | 4 | Nick Roach | 61.17 | 58.41 | 61.58 | 520 | Bears |
| 81 | 5 | Martez Wilson | 60.84 | 62.98 | 62.54 | 133 | Saints |
| 82 | 6 | Omar Gaither | 60.78 | 62.37 | 66.10 | 237 | Panthers |
| 83 | 7 | Darryl Blackstock | 60.48 | 64.24 | 72.31 | 112 | Raiders |
| 84 | 8 | Tracy White | 60.19 | 60.33 | 65.70 | 246 | Patriots |
| 85 | 9 | DeAndre Levy | 60.15 | 54.00 | 62.03 | 997 | Lions |
| 86 | 10 | Thomas Williams | 60.03 | 64.52 | 68.84 | 104 | Panthers |
| 87 | 11 | Quentin Groves | 60.03 | 55.11 | 63.17 | 204 | Raiders |
| 88 | 12 | Chad Greenway | 59.93 | 50.00 | 62.39 | 1059 | Vikings |
| 89 | 13 | Chris Gocong | 59.83 | 53.70 | 59.75 | 863 | Browns |
| 90 | 14 | Marvin Mitchell | 59.72 | 57.58 | 59.58 | 178 | Dolphins |
| 91 | 15 | Adam Hayward | 59.67 | 56.67 | 61.16 | 180 | Buccaneers |
| 92 | 16 | Moise Fokou | 59.46 | 56.74 | 61.41 | 227 | Eagles |
| 93 | 17 | Scott Shanle | 59.34 | 50.20 | 61.65 | 914 | Saints |
| 94 | 18 | Jason Williams | 59.20 | 61.32 | 64.42 | 104 | Panthers |
| 95 | 19 | Scott Fujita | 58.51 | 53.85 | 61.35 | 633 | Browns |
| 96 | 20 | Jonathan Vilma | 58.50 | 48.24 | 63.13 | 762 | Saints |
| 97 | 21 | Keith Brooking | 58.47 | 52.23 | 58.46 | 395 | Cowboys |
| 98 | 22 | Rocky McIntosh | 58.23 | 54.00 | 59.88 | 498 | Commanders |
| 99 | 23 | Brady Poppinga | 57.85 | 51.95 | 58.27 | 553 | Rams |
| 100 | 24 | Jamar Chaney | 57.71 | 53.00 | 60.59 | 848 | Eagles |
| 101 | 25 | Gary Guyton | 57.39 | 49.91 | 61.46 | 397 | Patriots |
| 102 | 26 | Larry Foote | 57.24 | 50.62 | 57.49 | 427 | Steelers |
| 103 | 27 | Na'il Diggs | 56.97 | 51.13 | 60.22 | 354 | Chargers |
| 104 | 28 | Robert Francois | 56.68 | 64.26 | 65.96 | 165 | Packers |
| 105 | 29 | Will Herring | 56.27 | 54.44 | 57.87 | 111 | Saints |
| 106 | 30 | Gerald McRath | 55.69 | 54.38 | 59.17 | 138 | Titans |
| 107 | 31 | Clint Session | 55.65 | 54.21 | 61.30 | 254 | Jaguars |
| 108 | 32 | Mason Foster | 55.63 | 43.40 | 59.62 | 865 | Buccaneers |
| 109 | 33 | Jordan Senn | 55.39 | 50.66 | 63.22 | 398 | Panthers |
| 110 | 34 | Kavell Conner | 55.27 | 46.28 | 59.04 | 788 | Colts |
| 111 | 35 | Jonathan Casillas | 55.13 | 50.46 | 57.21 | 545 | Saints |
| 112 | 36 | Dannell Ellerbe | 54.97 | 48.84 | 59.32 | 253 | Ravens |
| 113 | 37 | Barrett Ruud | 54.36 | 48.52 | 58.63 | 584 | Titans |
| 114 | 38 | Thomas Davis Sr. | 52.30 | 59.09 | 62.11 | 208 | Panthers |
| 115 | 39 | Casey Matthews | 52.27 | 43.67 | 58.01 | 326 | Eagles |
| 116 | 40 | Quincy Black | 51.65 | 44.03 | 55.82 | 660 | Buccaneers |

## QB — Quarterback

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Rodgers | 91.32 | 92.30 | 85.97 | 681 | Packers |
| 2 | 2 | Drew Brees | 88.15 | 91.12 | 81.18 | 840 | Saints |
| 3 | 3 | Tom Brady | 86.30 | 89.90 | 79.12 | 807 | Patriots |
| 4 | 4 | Eli Manning | 83.99 | 88.45 | 76.15 | 841 | Giants |
| 5 | 5 | Philip Rivers | 81.17 | 83.17 | 75.14 | 656 | Chargers |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Ben Roethlisberger | 78.49 | 78.91 | 74.13 | 644 | Steelers |
| 7 | 2 | Matt Ryan | 78.08 | 81.74 | 70.54 | 696 | Falcons |
| 8 | 3 | Tony Romo | 76.34 | 78.07 | 78.75 | 585 | Cowboys |
| 9 | 4 | Matt Schaub | 75.65 | 78.33 | 75.26 | 325 | Texans |
| 10 | 5 | Matthew Stafford | 75.46 | 76.14 | 76.01 | 793 | Lions |
| 11 | 6 | Michael Vick | 75.27 | 77.48 | 72.12 | 522 | Eagles |
| 12 | 7 | Alex Smith | 74.78 | 76.61 | 70.99 | 626 | 49ers |

### Starter (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Joe Flacco | 71.70 | 70.92 | 67.79 | 692 | Ravens |
| 14 | 2 | Matt Hasselbeck | 69.52 | 70.25 | 66.11 | 568 | Titans |
| 15 | 3 | Carson Palmer | 69.04 | 69.47 | 69.70 | 374 | Raiders |
| 16 | 4 | Ryan Fitzpatrick | 68.81 | 69.12 | 64.95 | 655 | Bills |
| 17 | 5 | Jay Cutler | 68.44 | 67.24 | 70.89 | 364 | Bears |
| 18 | 6 | Cam Newton | 67.60 | 65.10 | 72.24 | 634 | Panthers |
| 19 | 7 | Josh Freeman | 65.84 | 65.27 | 62.28 | 655 | Buccaneers |
| 20 | 8 | Matt Moore | 65.39 | 70.46 | 71.14 | 411 | Dolphins |
| 21 | 9 | Mark Sanchez | 64.36 | 60.86 | 62.85 | 642 | Jets |
| 22 | 10 | Andy Dalton | 63.79 | 62.50 | 63.58 | 651 | Bengals |
| 23 | 11 | Tarvaris Jackson | 63.30 | 70.14 | 63.04 | 539 | Seahawks |
| 24 | 12 | Kyle Orton | 62.47 | 64.80 | 64.42 | 287 | Chiefs |
| 25 | 13 | Jason Campbell | 62.10 | 66.79 | 67.10 | 190 | Raiders |

### Rotation/backup (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Chad Henne | 61.66 | 65.39 | 65.86 | 139 | Dolphins |
| 27 | 2 | Rex Grossman | 61.54 | 62.83 | 63.61 | 515 | Commanders |
| 28 | 3 | Donovan McNabb | 61.45 | 64.58 | 65.00 | 187 | Vikings |
| 29 | 4 | Matt Cassel | 61.39 | 63.82 | 62.25 | 326 | Chiefs |
| 30 | 5 | Kevin Kolb | 60.59 | 59.14 | 68.22 | 311 | Cardinals |
| 31 | 6 | Colt McCoy | 60.38 | 62.13 | 58.96 | 566 | Browns |
| 32 | 7 | Sam Bradford | 60.25 | 63.40 | 57.66 | 423 | Rams |
| 33 | 8 | Vince Young | 59.21 | 58.87 | 66.83 | 140 | Eagles |
| 34 | 9 | Dan Orlovsky | 58.76 | 58.71 | 64.88 | 215 | Colts |
| 35 | 10 | T.J. Yates | 58.55 | 59.07 | 62.16 | 220 | Texans |
| 36 | 11 | Christian Ponder | 57.48 | 52.19 | 59.60 | 356 | Vikings |
| 37 | 12 | Seneca Wallace | 57.15 | 60.03 | 57.53 | 122 | Browns |
| 38 | 13 | Kerry Collins | 57.06 | 59.92 | 57.15 | 105 | Colts |
| 39 | 14 | John Beck | 56.22 | 55.18 | 57.74 | 164 | Commanders |
| 40 | 15 | A.J. Feeley | 56.00 | 55.57 | 56.55 | 113 | Rams |
| 41 | 16 | Curtis Painter | 55.76 | 48.96 | 59.46 | 277 | Colts |
| 42 | 17 | John Skelton | 55.03 | 45.48 | 59.75 | 326 | Cardinals |
| 43 | 18 | Blaine Gabbert | 54.01 | 40.00 | 55.17 | 499 | Jaguars |

## S — Safety

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Eric Weddle | 88.35 | 85.60 | 86.02 | 974 | Chargers |
| 2 | 2 | Adrian Wilson | 84.26 | 82.50 | 81.27 | 1135 | Cardinals |
| 3 | 3 | Kenny Phillips | 83.03 | 78.30 | 82.02 | 1248 | Giants |
| 4 | 4 | Troy Polamalu | 82.95 | 75.90 | 83.48 | 960 | Steelers |
| 5 | 5 | Kam Chancellor | 82.49 | 79.20 | 81.16 | 1013 | Seahawks |
| 6 | 6 | Jairus Byrd | 81.36 | 75.10 | 81.37 | 1021 | Bills |

### Good (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Ryan Clark | 76.95 | 71.10 | 76.68 | 989 | Steelers |
| 8 | 2 | Mike Adams | 75.85 | 71.40 | 75.04 | 827 | Browns |
| 9 | 3 | Tyvon Branch | 75.49 | 68.30 | 76.11 | 1133 | Raiders |
| 10 | 4 | Bernard Pollard | 75.07 | 70.50 | 74.34 | 1014 | Ravens |
| 11 | 5 | Jim Leonhard | 74.99 | 78.00 | 72.71 | 781 | Jets |

### Starter (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | William Moore | 73.98 | 67.51 | 76.07 | 601 | Falcons |
| 13 | 2 | Michael Griffin | 72.97 | 67.90 | 72.18 | 1121 | Titans |
| 14 | 3 | Isa Abdul-Quddus | 72.32 | 65.41 | 74.84 | 186 | Saints |
| 15 | 4 | Earl Thomas III | 71.73 | 64.20 | 72.58 | 1097 | Seahawks |
| 16 | 5 | Danieal Manning | 71.43 | 62.60 | 73.80 | 846 | Texans |
| 17 | 6 | Michael Huff | 70.54 | 67.10 | 71.27 | 624 | Raiders |
| 18 | 7 | Rashad Johnson | 70.31 | 62.67 | 73.83 | 480 | Cardinals |
| 19 | 8 | Nate Allen | 70.20 | 62.96 | 72.67 | 752 | Eagles |
| 20 | 9 | Quintin Mikell | 69.81 | 62.20 | 70.72 | 1068 | Rams |
| 21 | 10 | Donte Whitner | 69.75 | 61.00 | 71.42 | 1048 | 49ers |
| 22 | 11 | Atari Bigby | 69.15 | 63.73 | 73.93 | 155 | Seahawks |
| 23 | 12 | Kendrick Lewis | 69.13 | 66.00 | 68.21 | 963 | Chiefs |
| 24 | 13 | Barry Church | 68.78 | 62.20 | 72.79 | 166 | Cowboys |
| 25 | 14 | T.J. Ward | 67.97 | 60.53 | 73.97 | 468 | Browns |
| 26 | 15 | Reggie Nelson | 67.78 | 62.30 | 69.21 | 1080 | Bengals |
| 27 | 16 | Troy Nolan | 67.71 | 60.57 | 71.17 | 424 | Texans |
| 28 | 17 | Morgan Burnett | 67.18 | 60.40 | 72.21 | 1144 | Packers |
| 29 | 18 | Brodney Pool | 67.11 | 62.52 | 67.30 | 545 | Jets |
| 30 | 19 | Dawan Landry | 66.27 | 58.20 | 67.48 | 1012 | Jaguars |
| 31 | 20 | Antoine Bethea | 66.15 | 52.50 | 71.09 | 1085 | Colts |
| 32 | 21 | Craig Steltz | 65.71 | 61.25 | 69.72 | 400 | Bears |
| 33 | 22 | Gerald Sensabaugh | 65.60 | 56.20 | 67.70 | 971 | Cowboys |
| 34 | 23 | Gibril Wilson | 65.52 | 62.68 | 64.28 | 211 | Bengals |
| 35 | 24 | LaRon Landry | 65.31 | 62.33 | 71.08 | 496 | Commanders |
| 36 | 25 | Brian Dawkins | 65.26 | 58.06 | 69.15 | 773 | Broncos |
| 37 | 26 | Usama Young | 64.92 | 55.82 | 69.17 | 652 | Browns |
| 38 | 27 | Dwight Lowery | 64.69 | 58.43 | 66.65 | 634 | Jaguars |
| 39 | 28 | Glover Quin | 64.12 | 58.80 | 63.50 | 1119 | Texans |
| 40 | 29 | Travis Daniels | 63.69 | 61.37 | 65.24 | 292 | Chiefs |
| 41 | 30 | Louis Delmas | 63.21 | 63.50 | 61.85 | 737 | Lions |
| 42 | 31 | Kerry Rhodes | 63.10 | 54.94 | 70.24 | 397 | Cardinals |
| 43 | 32 | Ryan Mundy | 62.41 | 55.23 | 67.07 | 273 | Steelers |
| 44 | 33 | Husain Abdullah | 62.21 | 54.71 | 67.99 | 564 | Vikings |
| 45 | 34 | Thomas DeCoud | 62.20 | 51.40 | 65.23 | 1013 | Falcons |

### Rotation/backup (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 46 | 1 | Matt Giordano | 61.23 | 58.10 | 65.27 | 850 | Raiders |
| 47 | 2 | David Bruton | 61.18 | 56.67 | 64.32 | 272 | Broncos |
| 48 | 3 | Corey Lynch | 60.97 | 59.83 | 65.12 | 273 | Buccaneers |
| 49 | 4 | Mistral Raymond | 60.93 | 59.80 | 65.85 | 378 | Vikings |
| 50 | 5 | Da'Norris Searcy | 60.74 | 57.27 | 66.19 | 227 | Bills |
| 51 | 6 | Chris Crocker | 60.73 | 55.80 | 62.59 | 1047 | Bengals |
| 52 | 7 | Reggie Smith | 60.65 | 55.45 | 61.11 | 260 | 49ers |
| 53 | 8 | Chris Hope | 60.50 | 54.11 | 64.50 | 286 | Titans |
| 54 | 9 | Dashon Goldson | 60.28 | 49.40 | 63.36 | 1016 | 49ers |
| 55 | 10 | Chris Prosinski | 60.21 | 59.66 | 60.58 | 195 | Jaguars |
| 56 | 11 | Kurt Coleman | 60.21 | 51.14 | 64.17 | 748 | Eagles |
| 57 | 12 | Darian Stewart | 59.93 | 57.40 | 60.05 | 891 | Rams |
| 58 | 13 | David Caldwell | 59.75 | 49.59 | 62.35 | 601 | Colts |
| 59 | 14 | Tanard Jackson | 59.33 | 61.59 | 59.90 | 523 | Buccaneers |
| 60 | 15 | Eric Hagg | 59.25 | 60.23 | 61.73 | 180 | Browns |
| 61 | 16 | George Wilson | 59.22 | 45.73 | 66.39 | 821 | Bills |
| 62 | 17 | Patrick Chung | 58.86 | 49.74 | 64.42 | 758 | Patriots |
| 63 | 18 | Yeremiah Bell | 58.82 | 46.30 | 63.00 | 1092 | Dolphins |
| 64 | 19 | Antrel Rolle | 58.66 | 51.40 | 59.33 | 1343 | Giants |
| 65 | 20 | Tom Zbikowski | 58.54 | 59.69 | 62.73 | 223 | Ravens |
| 66 | 21 | Abram Elam | 58.08 | 43.00 | 63.96 | 1015 | Cowboys |
| 67 | 22 | Josh Barrett | 58.04 | 64.27 | 65.68 | 218 | Patriots |
| 68 | 23 | Sergio Brown | 58.04 | 56.39 | 61.36 | 342 | Patriots |
| 69 | 24 | Malcolm Jenkins | 57.92 | 48.00 | 60.75 | 1092 | Saints |
| 70 | 25 | Quinton Carter | 57.88 | 50.41 | 59.73 | 824 | Broncos |
| 71 | 26 | Brandon Meriweather | 57.85 | 53.69 | 59.70 | 406 | Bears |
| 72 | 27 | Sherrod Martin | 57.54 | 50.60 | 58.38 | 990 | Panthers |
| 73 | 28 | Tyrone Culver | 57.37 | 54.91 | 59.92 | 437 | Dolphins |
| 74 | 29 | Chris Conte | 57.18 | 55.17 | 60.60 | 593 | Bears |
| 75 | 30 | Eric Smith | 56.96 | 45.80 | 60.24 | 943 | Jets |
| 76 | 31 | Madieu Williams | 56.87 | 57.78 | 60.04 | 139 | 49ers |
| 77 | 32 | Rahim Moore | 56.32 | 53.19 | 57.38 | 542 | Broncos |
| 78 | 33 | Charlie Peprah | 56.07 | 43.30 | 60.42 | 993 | Packers |
| 79 | 34 | Quintin Demps | 56.01 | 52.32 | 60.56 | 275 | Texans |
| 80 | 35 | Major Wright | 55.99 | 49.85 | 60.09 | 581 | Bears |
| 81 | 36 | Craig Dahl | 55.93 | 44.85 | 59.53 | 469 | Rams |
| 82 | 37 | Mike Mitchell | 55.71 | 51.42 | 56.75 | 493 | Raiders |
| 83 | 38 | Courtney Greene | 54.44 | 55.48 | 58.57 | 114 | Jaguars |
| 84 | 39 | Nick Collins | 54.03 | 55.84 | 57.77 | 132 | Packers |
| 85 | 40 | Amari Spievey | 53.85 | 45.30 | 56.95 | 988 | Lions |
| 86 | 41 | James Sanders | 53.69 | 43.11 | 59.83 | 509 | Falcons |
| 87 | 42 | Jordan Pugh | 53.39 | 46.59 | 59.35 | 243 | Panthers |
| 88 | 43 | James Ihedigbo | 53.33 | 43.00 | 56.84 | 897 | Patriots |
| 89 | 44 | Steve Gregory | 53.27 | 40.91 | 60.72 | 759 | Chargers |
| 90 | 45 | Melvin Bullitt | 52.67 | 60.80 | 64.59 | 129 | Colts |
| 91 | 46 | Sean Jones | 52.56 | 40.70 | 56.30 | 984 | Buccaneers |
| 92 | 47 | Cody Grimm | 52.16 | 55.81 | 56.76 | 175 | Buccaneers |
| 93 | 48 | Roman Harper | 51.61 | 40.00 | 55.18 | 1112 | Saints |
| 94 | 49 | Charles Godfrey | 51.40 | 40.60 | 55.73 | 856 | Panthers |
| 95 | 50 | Reed Doughty | 51.39 | 40.00 | 56.25 | 663 | Commanders |
| 96 | 51 | Reshad Jones | 51.36 | 41.23 | 59.15 | 653 | Dolphins |
| 97 | 52 | DeJon Gomes | 51.07 | 49.61 | 58.74 | 204 | Commanders |
| 98 | 53 | Tyrell Johnson | 49.72 | 43.58 | 56.81 | 324 | Vikings |
| 99 | 54 | Jamarca Sanford | 48.91 | 40.00 | 55.23 | 845 | Vikings |
| 100 | 55 | Jaiquawn Jarrett | 47.87 | 52.34 | 54.14 | 247 | Eagles |
| 101 | 56 | John Wendling | 45.00 | 48.41 | 49.91 | 159 | Lions |

## T — Tackle

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jason Peters | 95.24 | 89.44 | 94.94 | 925 | Eagles |
| 2 | 2 | Jordan Gross | 90.59 | 84.86 | 90.25 | 967 | Panthers |
| 3 | 3 | Duane Brown | 88.88 | 83.10 | 88.57 | 1176 | Texans |
| 4 | 4 | Zach Strief | 88.63 | 81.28 | 89.37 | 897 | Saints |
| 5 | 5 | Bryan Bulaga | 88.11 | 78.55 | 90.32 | 754 | Packers |
| 6 | 6 | Tyron Smith | 87.89 | 80.20 | 88.85 | 1040 | Cowboys |
| 7 | 7 | Andrew Whitworth | 87.30 | 81.50 | 87.00 | 1132 | Bengals |
| 8 | 8 | Jared Veldheer | 87.06 | 80.50 | 87.26 | 1073 | Raiders |
| 9 | 9 | Eugene Monroe | 86.39 | 79.40 | 86.89 | 891 | Jaguars |
| 10 | 10 | Phil Loadholt | 86.30 | 76.70 | 88.54 | 1003 | Vikings |
| 11 | 11 | David Stewart | 86.29 | 79.82 | 86.43 | 909 | Titans |
| 12 | 12 | Michael Roos | 86.16 | 79.50 | 86.43 | 1022 | Titans |
| 13 | 13 | Eric Winston | 86.15 | 78.90 | 86.81 | 1185 | Texans |
| 14 | 14 | Joe Staley | 85.91 | 77.90 | 87.09 | 1076 | 49ers |
| 15 | 15 | Tyson Clabo | 85.46 | 77.30 | 86.74 | 1191 | Falcons |
| 16 | 16 | Matt Light | 85.45 | 79.40 | 85.32 | 1232 | Patriots |
| 17 | 17 | Donald Penn | 85.28 | 77.70 | 86.16 | 1022 | Buccaneers |
| 18 | 18 | Trent Williams | 84.90 | 75.16 | 87.23 | 627 | Commanders |
| 19 | 19 | Joe Thomas | 84.16 | 77.30 | 84.57 | 1063 | Browns |
| 20 | 20 | Branden Albert | 84.13 | 75.60 | 85.65 | 1038 | Chiefs |
| 21 | 21 | Jermon Bushrod | 83.87 | 77.00 | 84.29 | 1320 | Saints |
| 22 | 22 | Jeff Backus | 83.38 | 75.60 | 84.40 | 1130 | Lions |
| 23 | 23 | D'Brickashaw Ferguson | 83.01 | 77.00 | 82.85 | 1088 | Jets |
| 24 | 24 | Demetress Bell | 82.90 | 71.89 | 86.08 | 394 | Bills |
| 25 | 25 | Jake Long | 82.13 | 73.94 | 83.43 | 784 | Dolphins |
| 26 | 26 | Tony Pashos | 81.97 | 72.20 | 84.31 | 781 | Browns |
| 27 | 27 | Will Beatty | 81.36 | 72.09 | 83.38 | 666 | Giants |
| 28 | 28 | Bryant McKinnie | 81.09 | 73.30 | 82.12 | 1213 | Ravens |
| 29 | 29 | Marcus Gilbert | 80.77 | 70.79 | 83.26 | 883 | Steelers |
| 30 | 30 | Andre Smith | 80.54 | 69.61 | 83.66 | 967 | Bengals |
| 31 | 31 | Cameron Bradfield | 80.42 | 66.56 | 85.50 | 181 | Jaguars |
| 32 | 32 | Sebastian Vollmer | 80.36 | 68.64 | 84.00 | 377 | Patriots |
| 33 | 33 | Nate Solder | 80.11 | 68.10 | 83.95 | 1023 | Patriots |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Michael Oher | 79.87 | 70.20 | 82.15 | 1203 | Ravens |
| 35 | 2 | Russell Okung | 79.87 | 68.56 | 83.25 | 767 | Seahawks |
| 36 | 3 | Anthony Davis | 79.69 | 68.50 | 82.98 | 1123 | 49ers |
| 37 | 4 | Doug Free | 79.43 | 70.00 | 81.55 | 1046 | Cowboys |
| 38 | 5 | King Dunlap | 79.02 | 64.88 | 84.28 | 144 | Eagles |
| 39 | 6 | Levi Brown | 78.45 | 66.60 | 82.18 | 1016 | Cardinals |
| 40 | 7 | Gosder Cherilus | 78.30 | 68.50 | 80.66 | 1031 | Lions |
| 41 | 8 | Khalif Barnes | 77.81 | 66.20 | 81.38 | 1000 | Raiders |
| 42 | 9 | Anthony Castonzo | 77.51 | 67.47 | 80.04 | 693 | Colts |
| 43 | 10 | Max Starks | 77.35 | 67.40 | 79.81 | 823 | Steelers |
| 44 | 11 | Breno Giacomini | 77.14 | 63.13 | 82.31 | 548 | Seahawks |
| 45 | 12 | Will Svitek | 76.96 | 66.02 | 80.08 | 796 | Falcons |
| 46 | 13 | Brandon Keith | 76.60 | 65.23 | 80.01 | 534 | Cardinals |
| 47 | 14 | Jammal Brown | 76.31 | 64.86 | 79.78 | 768 | Commanders |
| 48 | 15 | Marcus McNeill | 76.18 | 63.71 | 80.32 | 535 | Chargers |
| 49 | 16 | Ryan Clady | 76.09 | 66.70 | 78.18 | 1193 | Broncos |
| 50 | 17 | Jared Gaither | 75.99 | 69.59 | 76.09 | 337 | Chargers |
| 51 | 18 | Erik Pears | 75.67 | 64.50 | 78.95 | 1041 | Bills |
| 52 | 19 | Stephon Heyer | 75.20 | 61.94 | 79.88 | 244 | Raiders |
| 53 | 20 | Sean Locklear | 74.57 | 63.61 | 77.71 | 284 | Commanders |
| 54 | 21 | Anthony Collins | 74.52 | 66.54 | 75.68 | 169 | Bengals |
| 55 | 22 | Jeff Linkenbach | 74.44 | 63.19 | 77.77 | 981 | Colts |
| 56 | 23 | Jason Smith | 74.26 | 60.17 | 79.49 | 321 | Rams |
| 57 | 24 | Chris Hairston | 74.20 | 62.54 | 77.80 | 466 | Bills |
| 58 | 25 | Wayne Hunter | 74.13 | 59.80 | 79.51 | 1082 | Jets |

### Starter (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | Jeff Otah | 73.14 | 66.70 | 73.26 | 229 | Panthers |
| 60 | 2 | Sam Baker | 72.73 | 60.93 | 76.43 | 436 | Falcons |
| 61 | 3 | Chad Clifton | 72.69 | 60.35 | 76.75 | 329 | Packers |
| 62 | 4 | J'Marcus Webb | 72.41 | 58.10 | 77.79 | 1007 | Bears |
| 63 | 5 | Demar Dotson | 71.99 | 62.16 | 74.38 | 132 | Buccaneers |
| 64 | 6 | Jeremy Trueblood | 71.97 | 58.44 | 76.83 | 941 | Buccaneers |
| 65 | 7 | Frank Omiyale | 71.81 | 55.27 | 78.67 | 220 | Bears |
| 66 | 8 | Marcus Cannon | 71.15 | 62.24 | 72.93 | 164 | Patriots |
| 67 | 9 | Charles Brown | 70.85 | 59.30 | 74.39 | 400 | Saints |
| 68 | 10 | Corey Hilliard | 69.81 | 56.80 | 74.32 | 157 | Lions |
| 69 | 11 | Byron Bell | 69.16 | 55.54 | 74.08 | 815 | Panthers |
| 70 | 12 | Barry Richardson | 68.77 | 54.40 | 74.19 | 1053 | Chiefs |
| 71 | 13 | Chris Clark | 68.08 | 58.84 | 70.08 | 147 | Broncos |
| 72 | 14 | Derek Sherrod | 67.92 | 57.84 | 70.47 | 110 | Packers |
| 73 | 15 | Jonathan Scott | 67.20 | 54.86 | 71.26 | 406 | Steelers |
| 74 | 16 | Marshall Newhouse | 66.71 | 51.95 | 72.39 | 927 | Packers |
| 75 | 17 | Joe Barksdale | 66.05 | 58.04 | 67.23 | 152 | Raiders |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 86.37 | 92.90 | 77.85 | 745 | Patriots |
| 2 | 2 | Jimmy Graham | 83.71 | 90.10 | 75.29 | 699 | Saints |
| 3 | 3 | Joel Dreessen | 82.87 | 82.90 | 78.69 | 353 | Texans |
| 4 | 4 | Tony Gonzalez | 81.67 | 87.33 | 73.73 | 647 | Falcons |
| 5 | 5 | Heath Miller | 81.67 | 86.98 | 73.96 | 568 | Steelers |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Vernon Davis | 79.66 | 80.62 | 74.86 | 611 | 49ers |
| 7 | 2 | Fred Davis | 79.56 | 75.06 | 78.40 | 478 | Commanders |
| 8 | 3 | Anthony Fasano | 79.46 | 78.69 | 75.80 | 497 | Dolphins |
| 9 | 4 | Jason Witten | 79.38 | 78.26 | 75.96 | 609 | Cowboys |
| 10 | 5 | Brent Celek | 78.12 | 78.19 | 73.91 | 574 | Eagles |
| 11 | 6 | Marcedes Lewis | 77.88 | 74.86 | 75.73 | 421 | Jaguars |
| 12 | 7 | Jake Ballard | 77.87 | 76.46 | 74.65 | 584 | Giants |
| 13 | 8 | Aaron Hernandez | 77.79 | 80.27 | 71.97 | 576 | Patriots |
| 14 | 9 | Antonio Gates | 77.75 | 78.20 | 73.29 | 512 | Chargers |
| 15 | 10 | Todd Heap | 76.96 | 68.05 | 78.74 | 202 | Cardinals |
| 16 | 11 | Jared Cook | 76.12 | 63.90 | 80.10 | 495 | Titans |
| 17 | 12 | Jeremy Shockey | 75.69 | 75.87 | 71.40 | 266 | Panthers |
| 18 | 13 | Owen Daniels | 75.03 | 77.23 | 69.40 | 480 | Texans |
| 19 | 14 | Jermichael Finley | 75.01 | 66.67 | 76.41 | 599 | Packers |
| 20 | 15 | Dennis Pitta | 74.69 | 71.99 | 72.33 | 336 | Ravens |
| 21 | 16 | Kellen Davis | 74.58 | 73.01 | 71.46 | 316 | Bears |
| 22 | 17 | Kellen Winslow | 74.41 | 67.21 | 75.04 | 606 | Buccaneers |
| 23 | 18 | Randy McMichael | 74.17 | 73.39 | 70.53 | 287 | Chargers |

### Starter (39 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Kyle Rudolph | 73.94 | 71.15 | 71.64 | 245 | Vikings |
| 25 | 2 | Tony Scheffler | 73.89 | 69.62 | 72.57 | 230 | Lions |
| 26 | 3 | Dustin Keller | 73.54 | 67.61 | 73.32 | 505 | Jets |
| 27 | 4 | Visanthe Shiancoe | 73.49 | 67.52 | 73.31 | 422 | Vikings |
| 28 | 5 | Brandon Pettigrew | 73.28 | 69.80 | 71.44 | 719 | Lions |
| 29 | 6 | Benjamin Watson | 72.62 | 63.42 | 74.58 | 418 | Browns |
| 30 | 7 | Jermaine Gresham | 72.51 | 72.73 | 68.19 | 554 | Bengals |
| 31 | 8 | Greg Olsen | 72.34 | 68.52 | 70.72 | 470 | Panthers |
| 32 | 9 | Craig Stevens | 72.24 | 63.80 | 73.70 | 142 | Titans |
| 33 | 10 | Scott Chandler | 72.05 | 69.67 | 69.47 | 282 | Bills |
| 34 | 11 | Martellus Bennett | 71.53 | 69.85 | 68.48 | 202 | Cowboys |
| 35 | 12 | Jeff King | 71.25 | 77.38 | 62.99 | 229 | Cardinals |
| 36 | 13 | Evan Moore | 71.24 | 67.86 | 69.33 | 241 | Browns |
| 37 | 14 | Michael Hoomanawanui | 70.97 | 63.66 | 71.67 | 209 | Rams |
| 38 | 15 | Kevin Boss | 70.62 | 66.76 | 69.02 | 296 | Raiders |
| 39 | 16 | Delanie Walker | 69.80 | 63.43 | 69.88 | 257 | 49ers |
| 40 | 17 | Donald Lee | 69.66 | 64.60 | 68.87 | 122 | Bengals |
| 41 | 18 | Bear Pascoe | 69.43 | 62.95 | 69.58 | 177 | Giants |
| 42 | 19 | Zach Miller | 69.25 | 59.91 | 71.31 | 508 | Seahawks |
| 43 | 20 | Clay Harbor | 69.05 | 64.70 | 67.79 | 152 | Eagles |
| 44 | 21 | Ed Dickson | 68.61 | 65.65 | 66.41 | 549 | Ravens |
| 45 | 22 | Jacob Tamme | 68.41 | 63.55 | 67.48 | 192 | Colts |
| 46 | 23 | Matt Spaeth | 68.28 | 68.58 | 63.92 | 150 | Bears |
| 47 | 24 | Dante Rosario | 68.10 | 61.55 | 68.30 | 110 | Broncos |
| 48 | 25 | Logan Paulsen | 68.00 | 61.32 | 68.29 | 170 | Commanders |
| 49 | 26 | Anthony Becht | 67.56 | 62.27 | 66.92 | 136 | Chiefs |
| 50 | 27 | Lance Kendricks | 66.63 | 58.86 | 67.64 | 378 | Rams |
| 51 | 28 | Luke Stocker | 66.63 | 63.90 | 64.28 | 139 | Buccaneers |
| 52 | 29 | Brandon Myers | 66.58 | 64.45 | 63.83 | 177 | Raiders |
| 53 | 30 | Michael Palmer | 66.57 | 62.64 | 65.02 | 112 | Falcons |
| 54 | 31 | Dallas Clark | 66.49 | 61.64 | 65.55 | 334 | Colts |
| 55 | 32 | Alex Smith | 66.35 | 63.93 | 63.80 | 141 | Browns |
| 56 | 33 | Rob Housler | 65.86 | 56.60 | 67.87 | 136 | Cardinals |
| 57 | 34 | Zach Potter | 65.61 | 60.80 | 64.65 | 116 | Jaguars |
| 58 | 35 | Matthew Mulligan | 65.41 | 63.02 | 62.84 | 173 | Jets |
| 59 | 36 | Travis Beckum | 64.76 | 61.55 | 62.74 | 154 | Giants |
| 60 | 37 | Jake O'Connell | 64.17 | 59.30 | 63.25 | 144 | Chiefs |
| 61 | 38 | Billy Bajema | 63.16 | 56.41 | 63.49 | 229 | Rams |
| 62 | 39 | Tom Crabtree | 62.43 | 56.23 | 62.40 | 116 | Packers |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 63 | 1 | Brody Eldridge | 60.80 | 58.56 | 58.13 | 100 | Colts |
| 64 | 2 | Anthony McCoy | 60.79 | 53.44 | 61.53 | 215 | Seahawks |

## WR — Wide Receiver

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Calvin Johnson | 88.84 | 91.20 | 83.10 | 766 | Lions |
| 2 | 2 | Larry Fitzgerald | 87.58 | 89.79 | 81.94 | 659 | Cardinals |
| 3 | 3 | Malcom Floyd | 87.51 | 84.58 | 85.29 | 358 | Chargers |
| 4 | 4 | Hakeem Nicks | 86.49 | 89.20 | 80.51 | 775 | Giants |
| 5 | 5 | Antonio Brown | 86.36 | 85.04 | 83.07 | 499 | Steelers |
| 6 | 6 | Jordy Nelson | 86.25 | 84.69 | 83.13 | 505 | Packers |
| 7 | 7 | Victor Cruz | 85.49 | 82.10 | 83.58 | 701 | Giants |
| 8 | 8 | Marques Colston | 84.69 | 88.66 | 77.87 | 599 | Saints |
| 9 | 9 | Andre Johnson | 83.32 | 80.54 | 81.01 | 231 | Texans |
| 10 | 10 | Wes Welker | 83.25 | 89.20 | 75.11 | 748 | Patriots |
| 11 | 11 | Mike Wallace | 82.92 | 78.15 | 81.93 | 630 | Steelers |
| 12 | 12 | Dwayne Bowe | 82.79 | 82.75 | 78.65 | 527 | Chiefs |
| 13 | 13 | Percy Harvin | 82.41 | 82.39 | 78.26 | 363 | Vikings |
| 14 | 14 | Brandon Lloyd | 82.41 | 80.80 | 79.31 | 512 | Rams |
| 15 | 15 | Demaryius Thomas | 81.94 | 71.26 | 84.89 | 331 | Broncos |
| 16 | 16 | Dez Bryant | 81.11 | 78.42 | 78.74 | 538 | Cowboys |
| 17 | 17 | Greg Jennings | 80.84 | 76.71 | 79.43 | 522 | Packers |
| 18 | 18 | Denarius Moore | 80.33 | 72.28 | 81.53 | 357 | Raiders |
| 19 | 19 | Doug Baldwin | 80.26 | 75.10 | 79.53 | 365 | Seahawks |

### Good (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Vincent Jackson | 79.81 | 75.25 | 78.68 | 607 | Chargers |
| 21 | 2 | Johnny Knox | 79.55 | 71.31 | 80.88 | 334 | Bears |
| 22 | 3 | Roddy White | 79.43 | 80.90 | 74.28 | 696 | Falcons |
| 23 | 4 | Brandon Marshall | 79.07 | 75.38 | 77.36 | 534 | Dolphins |
| 24 | 5 | A.J. Green | 78.91 | 71.26 | 79.84 | 581 | Bengals |
| 25 | 6 | Nate Washington | 78.70 | 75.67 | 76.56 | 581 | Titans |
| 26 | 7 | Julio Jones | 78.60 | 69.81 | 80.29 | 525 | Falcons |
| 27 | 8 | Darrius Heyward-Bey | 78.00 | 74.98 | 75.84 | 462 | Raiders |
| 28 | 9 | Jabar Gaffney | 77.96 | 75.24 | 75.61 | 599 | Commanders |
| 29 | 10 | Steve Johnson | 77.80 | 75.44 | 75.21 | 585 | Bills |
| 30 | 11 | Reggie Wayne | 77.75 | 75.80 | 74.88 | 594 | Colts |
| 31 | 12 | Lance Moore | 77.29 | 75.72 | 74.17 | 335 | Saints |
| 32 | 13 | DeSean Jackson | 77.16 | 69.53 | 78.08 | 565 | Eagles |
| 33 | 14 | Anquan Boldin | 77.10 | 73.21 | 75.52 | 623 | Ravens |
| 34 | 15 | Jeremy Maclin | 76.51 | 73.16 | 74.57 | 482 | Eagles |
| 35 | 16 | Jacoby Ford | 75.96 | 65.44 | 78.81 | 119 | Raiders |
| 36 | 17 | Torrey Smith | 75.78 | 68.45 | 76.50 | 623 | Ravens |
| 37 | 18 | Deion Branch | 75.67 | 68.18 | 76.50 | 655 | Patriots |
| 38 | 19 | Earl Bennett | 75.26 | 64.74 | 78.10 | 271 | Bears |
| 39 | 20 | Vincent Brown | 75.20 | 65.67 | 77.38 | 235 | Chargers |
| 40 | 21 | Miles Austin | 75.13 | 68.75 | 75.22 | 363 | Cowboys |
| 41 | 22 | Robert Meachem | 75.04 | 70.15 | 74.13 | 533 | Saints |
| 42 | 23 | Michael Crabtree | 75.02 | 73.81 | 71.66 | 511 | 49ers |
| 43 | 24 | Mario Manningham | 74.94 | 68.05 | 75.37 | 527 | Giants |
| 44 | 25 | Steve Breaston | 74.93 | 69.11 | 74.65 | 506 | Chiefs |
| 45 | 26 | Josh Morgan | 74.82 | 65.75 | 76.70 | 144 | 49ers |
| 46 | 27 | Danario Alexander | 74.70 | 66.81 | 75.80 | 283 | Rams |
| 47 | 28 | Randall Cobb | 74.67 | 63.56 | 77.91 | 204 | Packers |
| 48 | 29 | Donte' Stallworth | 74.65 | 64.85 | 77.01 | 231 | Commanders |
| 49 | 30 | Sidney Rice | 74.48 | 65.36 | 76.39 | 289 | Seahawks |
| 50 | 31 | Golden Tate | 74.23 | 69.83 | 73.00 | 299 | Seahawks |
| 51 | 32 | Emmanuel Sanders | 74.18 | 68.35 | 73.90 | 272 | Steelers |
| 52 | 33 | Plaxico Burress | 74.17 | 70.29 | 72.59 | 547 | Jets |
| 53 | 34 | Brandon LaFell | 74.13 | 66.81 | 74.85 | 466 | Panthers |
| 54 | 35 | Joshua Cribbs | 74.02 | 69.92 | 72.58 | 373 | Browns |

### Starter (75 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 55 | 1 | James Jones | 73.82 | 64.15 | 76.10 | 375 | Packers |
| 56 | 2 | Braylon Edwards | 73.77 | 59.85 | 78.89 | 156 | 49ers |
| 57 | 3 | Santana Moss | 73.69 | 68.92 | 72.70 | 413 | Commanders |
| 58 | 4 | Steve Smith | 73.05 | 56.80 | 79.72 | 718 | Panthers |
| 59 | 5 | Pierre Garcon | 72.91 | 66.22 | 73.20 | 587 | Colts |
| 60 | 6 | Preston Parker | 72.62 | 63.06 | 74.82 | 407 | Buccaneers |
| 61 | 7 | Chastin West | 72.44 | 63.24 | 74.41 | 134 | Jaguars |
| 62 | 8 | Devin Aromashodu | 72.30 | 62.61 | 74.60 | 413 | Vikings |
| 63 | 9 | Roy Williams | 72.20 | 67.81 | 70.96 | 339 | Bears |
| 64 | 10 | Brian Hartline | 72.02 | 65.88 | 71.95 | 452 | Dolphins |
| 65 | 11 | Jason Avant | 71.99 | 66.44 | 71.52 | 487 | Eagles |
| 66 | 12 | Damian Williams | 71.86 | 64.52 | 72.59 | 481 | Titans |
| 67 | 13 | Jerome Simpson | 71.69 | 64.02 | 72.63 | 605 | Bengals |
| 68 | 14 | Kevin Walter | 71.64 | 67.27 | 70.39 | 421 | Texans |
| 69 | 15 | Michael Jenkins | 71.45 | 65.19 | 71.46 | 330 | Vikings |
| 70 | 16 | Arrelious Benn | 71.36 | 60.95 | 74.13 | 301 | Buccaneers |
| 71 | 17 | Jordan Norwood | 71.34 | 63.54 | 72.37 | 184 | Browns |
| 72 | 18 | Louis Murphy Jr. | 71.30 | 57.39 | 76.41 | 166 | Raiders |
| 73 | 19 | Jacoby Jones | 71.18 | 62.99 | 72.47 | 432 | Texans |
| 74 | 20 | Austin Collie | 71.02 | 66.32 | 69.98 | 457 | Colts |
| 75 | 21 | Naaman Roosevelt | 70.87 | 58.58 | 74.90 | 231 | Bills |
| 76 | 22 | Jeremy Kerley | 70.82 | 65.97 | 69.88 | 240 | Jets |
| 77 | 23 | Chad Johnson | 70.81 | 63.04 | 71.83 | 265 | Patriots |
| 78 | 24 | Ruvell Martin | 70.68 | 58.50 | 74.64 | 114 | Bills |
| 79 | 25 | Devery Henderson | 70.65 | 61.02 | 72.90 | 565 | Saints |
| 80 | 26 | Santonio Holmes | 70.56 | 61.47 | 72.46 | 616 | Jets |
| 81 | 27 | Jerricho Cotchery | 70.48 | 67.37 | 68.38 | 226 | Steelers |
| 82 | 28 | Kyle Williams | 70.36 | 66.05 | 69.07 | 232 | 49ers |
| 83 | 29 | Matt Willis | 70.08 | 61.10 | 71.90 | 277 | Broncos |
| 84 | 30 | Hines Ward | 70.06 | 65.21 | 69.12 | 314 | Steelers |
| 85 | 31 | Kevin Ogletree | 69.95 | 61.15 | 71.65 | 199 | Cowboys |
| 86 | 32 | Riley Cooper | 69.67 | 59.43 | 72.33 | 217 | Eagles |
| 87 | 33 | Anthony Armstrong | 69.65 | 55.62 | 74.83 | 193 | Commanders |
| 88 | 34 | Lee Evans | 69.65 | 58.80 | 72.72 | 217 | Ravens |
| 89 | 35 | Eric Decker | 69.63 | 62.80 | 70.02 | 504 | Broncos |
| 90 | 36 | Mike Williams | 69.47 | 62.50 | 69.95 | 948 | Buccaneers |
| 91 | 37 | David Nelson | 69.47 | 64.44 | 68.66 | 565 | Bills |
| 92 | 38 | Greg Salas | 69.29 | 61.42 | 70.37 | 139 | Rams |
| 93 | 39 | Patrick Crayton | 69.14 | 57.71 | 72.59 | 266 | Chargers |
| 94 | 40 | Early Doucet | 69.14 | 59.84 | 71.18 | 447 | Cardinals |
| 95 | 41 | Mike Thomas | 68.89 | 61.42 | 69.70 | 434 | Jaguars |
| 96 | 42 | Davone Bess | 68.84 | 60.96 | 69.92 | 405 | Dolphins |
| 97 | 43 | Brandon Gibson | 68.70 | 64.63 | 67.25 | 393 | Rams |
| 98 | 44 | Chaz Schilens | 68.69 | 64.65 | 67.22 | 241 | Raiders |
| 99 | 45 | Nate Burleson | 68.30 | 62.50 | 68.00 | 700 | Lions |
| 100 | 46 | Titus Young | 68.00 | 63.09 | 67.10 | 544 | Lions |
| 101 | 47 | Dezmon Briscoe | 67.76 | 61.01 | 68.10 | 396 | Buccaneers |
| 102 | 48 | Legedu Naanee | 67.71 | 58.35 | 69.79 | 497 | Panthers |
| 103 | 49 | Patrick Turner | 67.69 | 62.27 | 67.13 | 108 | Jets |
| 104 | 50 | Lavelle Hawkins | 67.63 | 61.42 | 67.60 | 463 | Titans |
| 105 | 51 | Donald Driver | 67.34 | 61.83 | 66.84 | 419 | Packers |
| 106 | 52 | Harry Douglas | 67.33 | 58.61 | 68.98 | 498 | Falcons |
| 107 | 53 | Greg Camarillo | 67.10 | 59.23 | 68.18 | 154 | Vikings |
| 108 | 54 | Mohamed Massaquoi | 66.84 | 58.30 | 68.37 | 431 | Browns |
| 109 | 55 | LaQuan Williams | 66.78 | 58.16 | 68.36 | 101 | Ravens |
| 110 | 56 | Andre Roberts | 66.77 | 58.91 | 67.85 | 647 | Cardinals |
| 111 | 57 | Devin Hester | 66.66 | 55.77 | 69.75 | 340 | Bears |
| 112 | 58 | Donald Jones | 66.51 | 59.82 | 66.81 | 239 | Bills |
| 113 | 59 | Eddie Royal | 66.39 | 57.22 | 68.34 | 303 | Broncos |
| 114 | 60 | Derrick Mason | 66.34 | 57.63 | 67.98 | 213 | Texans |
| 115 | 61 | Ted Ginn Jr. | 65.76 | 62.22 | 63.96 | 252 | 49ers |
| 116 | 62 | Ramses Barden | 65.75 | 59.48 | 65.76 | 106 | Giants |
| 117 | 63 | Eric Weems | 65.75 | 59.24 | 65.92 | 170 | Falcons |
| 118 | 64 | Derek Hagan | 65.40 | 61.43 | 63.88 | 235 | Bills |
| 119 | 65 | Austin Pettis | 65.34 | 60.18 | 64.61 | 246 | Rams |
| 120 | 66 | Greg Little | 65.28 | 58.73 | 65.48 | 627 | Browns |
| 121 | 67 | Ben Obomanu | 65.13 | 59.34 | 64.82 | 359 | Seahawks |
| 122 | 68 | Sam Hurd | 64.83 | 57.88 | 65.29 | 134 | Bears |
| 123 | 69 | Julian Edelman | 64.77 | 58.40 | 64.85 | 105 | Patriots |
| 124 | 70 | Terrence Austin | 64.36 | 57.55 | 64.74 | 152 | Commanders |
| 125 | 71 | David Anderson | 64.28 | 58.88 | 63.71 | 100 | Commanders |
| 126 | 72 | Brad Smith | 64.21 | 56.80 | 64.98 | 271 | Bills |
| 127 | 73 | Jonathan Baldwin | 64.07 | 59.28 | 63.09 | 281 | Chiefs |
| 128 | 74 | Andre Caldwell | 63.89 | 55.76 | 65.14 | 352 | Bengals |
| 129 | 75 | Jarett Dillard | 63.28 | 59.36 | 61.73 | 337 | Jaguars |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 130 | 1 | Cecil Shorts | 61.53 | 54.22 | 62.23 | 123 | Jaguars |
