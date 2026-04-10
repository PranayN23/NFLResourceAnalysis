# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:27:12Z
- **Requested analysis_year:** 2011 (clamped to 2011)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Myers | 97.54 | 92.60 | 96.66 | 1165 | Texans |
| 2 | 2 | Nick Mangold | 94.74 | 88.80 | 94.54 | 894 | Jets |
| 3 | 3 | John Sullivan | 88.93 | 82.50 | 89.05 | 936 | Vikings |
| 4 | 4 | Jeff Saturday | 88.38 | 81.40 | 88.87 | 975 | Colts |
| 5 | 5 | Scott Wells | 88.20 | 80.60 | 89.10 | 1085 | Packers |
| 6 | 6 | Matt Birk | 84.46 | 76.00 | 85.93 | 1192 | Ravens |
| 7 | 7 | Eric Wood | 83.10 | 78.60 | 81.93 | 543 | Bills |
| 8 | 8 | Brad Meester | 82.93 | 75.20 | 83.91 | 1041 | Jaguars |
| 9 | 9 | Alex Mack | 82.87 | 74.60 | 84.21 | 1063 | Browns |
| 10 | 10 | Ryan Kalil | 82.68 | 73.60 | 84.56 | 1039 | Panthers |
| 11 | 11 | Todd McClure | 82.14 | 73.90 | 83.47 | 928 | Falcons |
| 12 | 12 | Mike Pouncey | 81.52 | 73.00 | 83.03 | 1005 | Dolphins |
| 13 | 13 | Ryan Wendell | 81.10 | 76.00 | 80.34 | 354 | Patriots |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Kyle Cook | 79.50 | 69.00 | 82.34 | 1134 | Bengals |
| 15 | 2 | Lyle Sendlein | 79.15 | 70.30 | 80.88 | 1029 | Cardinals |
| 16 | 3 | Nick Hardwick | 79.11 | 69.70 | 81.22 | 1050 | Chargers |
| 17 | 4 | Max Unger | 78.89 | 75.60 | 76.91 | 992 | Seahawks |
| 18 | 5 | Will Montgomery | 78.83 | 69.20 | 81.08 | 1081 | Commanders |
| 19 | 6 | Jonathan Goodwin | 78.03 | 68.80 | 80.02 | 1144 | 49ers |
| 20 | 7 | Dominic Raiola | 77.76 | 67.70 | 80.30 | 1148 | Lions |
| 21 | 8 | David Baas | 75.93 | 65.50 | 78.71 | 971 | Giants |
| 22 | 9 | Eugene Amano | 74.62 | 64.80 | 77.00 | 1021 | Titans |
| 23 | 10 | Jason Kelce | 74.10 | 62.50 | 77.66 | 1063 | Eagles |

### Starter (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Maurkice Pouncey | 73.71 | 62.90 | 76.75 | 845 | Steelers |
| 25 | 2 | Jeff Faine | 71.64 | 60.90 | 74.64 | 925 | Buccaneers |
| 26 | 3 | Roberto Garza | 71.20 | 58.00 | 75.83 | 1007 | Bears |
| 27 | 4 | Phil Costa | 70.11 | 56.90 | 74.75 | 992 | Cowboys |
| 28 | 5 | Nick McDonald | 68.94 | 62.70 | 68.94 | 106 | Patriots |
| 29 | 6 | J.D. Walton | 68.91 | 57.10 | 72.61 | 1195 | Broncos |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Darrelle Revis | 92.66 | 87.60 | 91.86 | 1014 | Jets |
| 2 | 2 | Lardarius Webb | 89.99 | 87.30 | 87.62 | 1100 | Ravens |
| 3 | 3 | Champ Bailey | 88.12 | 84.50 | 87.40 | 1013 | Broncos |
| 4 | 4 | Brent Grimes | 86.77 | 83.60 | 87.31 | 692 | Falcons |
| 5 | 5 | Asante Samuel | 86.52 | 83.90 | 86.96 | 809 | Eagles |
| 6 | 6 | Cortland Finnegan | 85.72 | 84.20 | 82.56 | 1122 | Titans |
| 7 | 7 | Richard Sherman | 83.88 | 79.30 | 84.85 | 763 | Seahawks |
| 8 | 8 | Johnathan Joseph | 83.28 | 80.20 | 82.74 | 1022 | Texans |
| 9 | 9 | Brandon Flowers | 83.06 | 77.50 | 82.60 | 974 | Chiefs |
| 10 | 10 | Brice McCain | 82.70 | 78.90 | 83.02 | 465 | Texans |
| 11 | 11 | Alterraun Verner | 82.26 | 77.80 | 81.06 | 655 | Titans |
| 12 | 12 | Derek Cox | 81.28 | 81.20 | 84.85 | 324 | Jaguars |
| 13 | 13 | Carlos Rogers | 80.63 | 75.10 | 81.71 | 1173 | 49ers |
| 14 | 14 | Charles Tillman | 80.61 | 75.20 | 80.05 | 1070 | Bears |
| 15 | 15 | Corey Webster | 80.44 | 73.10 | 81.55 | 1344 | Giants |
| 16 | 16 | Antoine Winfield | 80.36 | 84.30 | 80.74 | 322 | Vikings |
| 17 | 17 | Chris Gamble | 80.08 | 77.50 | 80.24 | 913 | Panthers |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Aqib Talib | 79.91 | 76.20 | 82.12 | 682 | Buccaneers |
| 19 | 2 | Tim Jennings | 79.39 | 75.30 | 77.95 | 995 | Bears |
| 20 | 3 | Jimmy Smith | 79.26 | 74.80 | 83.26 | 331 | Ravens |
| 21 | 4 | Tramon Williams | 78.66 | 71.70 | 79.13 | 1056 | Packers |
| 22 | 5 | Joe Haden | 78.42 | 67.20 | 82.39 | 973 | Browns |
| 23 | 6 | Jabari Greer | 78.41 | 72.20 | 78.76 | 1097 | Saints |
| 24 | 7 | Brandon Carr | 78.03 | 69.60 | 79.48 | 1006 | Chiefs |
| 25 | 8 | Tarell Brown | 77.83 | 70.20 | 79.92 | 1145 | 49ers |
| 26 | 9 | Chris Harris Jr. | 77.15 | 72.50 | 78.16 | 550 | Broncos |
| 27 | 10 | Leodis McKelvin | 76.93 | 72.20 | 78.51 | 503 | Bills |
| 28 | 11 | Drew Coleman | 76.52 | 72.30 | 75.16 | 494 | Jaguars |
| 29 | 12 | Chris Houston | 76.46 | 71.40 | 76.70 | 826 | Lions |
| 30 | 13 | William Middleton | 76.31 | 75.80 | 77.69 | 388 | Jaguars |
| 31 | 14 | Aaron Berry | 76.10 | 74.40 | 81.53 | 455 | Lions |
| 32 | 15 | Josh Wilson | 75.96 | 66.80 | 78.28 | 944 | Commanders |
| 33 | 16 | Jason McCourty | 75.83 | 67.50 | 79.44 | 964 | Titans |
| 34 | 17 | Vontae Davis | 75.74 | 70.10 | 77.94 | 677 | Dolphins |
| 35 | 18 | Brandon Browner | 75.58 | 61.90 | 80.53 | 1071 | Seahawks |
| 36 | 19 | Sheldon Brown | 75.39 | 67.30 | 76.62 | 959 | Browns |
| 37 | 20 | Patrick Robinson | 74.95 | 69.60 | 77.09 | 813 | Saints |
| 38 | 21 | Antoine Cason | 74.66 | 63.00 | 78.26 | 870 | Chargers |
| 39 | 22 | Antonio Cromartie | 74.65 | 65.80 | 76.38 | 944 | Jets |
| 40 | 23 | Jason Allen | 74.50 | 65.20 | 77.31 | 544 | Texans |
| 41 | 24 | Nnamdi Asomugha | 74.07 | 65.80 | 76.20 | 932 | Eagles |

### Starter (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 42 | 1 | Leon Hall | 73.84 | 68.30 | 77.91 | 542 | Bengals |
| 43 | 2 | Dimitri Patterson | 73.75 | 64.10 | 77.72 | 545 | Browns |
| 44 | 3 | Chris Culliver | 73.73 | 65.90 | 75.82 | 527 | 49ers |
| 45 | 4 | Javier Arenas | 73.58 | 65.70 | 75.31 | 374 | Chiefs |
| 46 | 5 | Tramaine Brock Sr. | 73.48 | 70.60 | 85.10 | 116 | 49ers |
| 47 | 6 | Richard Marshall | 73.06 | 64.20 | 74.80 | 819 | Cardinals |
| 48 | 7 | Kyle Arrington | 72.99 | 62.70 | 75.68 | 1155 | Patriots |
| 49 | 8 | William Gay | 72.57 | 62.70 | 74.98 | 1009 | Steelers |
| 50 | 9 | Sam Shields | 72.50 | 63.40 | 74.40 | 752 | Packers |
| 51 | 10 | Chris Cook | 72.49 | 71.50 | 79.40 | 253 | Vikings |
| 52 | 11 | Roy Lewis | 72.01 | 69.50 | 74.60 | 261 | Seahawks |
| 53 | 12 | Mike Jenkins | 71.74 | 64.60 | 74.94 | 586 | Cowboys |
| 54 | 13 | DeAngelo Hall | 71.70 | 60.10 | 75.26 | 1006 | Commanders |
| 55 | 14 | Jerraud Powers | 71.68 | 68.50 | 74.58 | 794 | Colts |
| 56 | 15 | Cary Williams | 71.49 | 65.40 | 76.46 | 1089 | Ravens |
| 57 | 16 | Morgan Trent | 71.43 | 68.20 | 78.14 | 225 | Jaguars |
| 58 | 17 | Aaron Ross | 70.97 | 60.70 | 74.04 | 1161 | Giants |
| 59 | 18 | Stanford Routt | 70.88 | 59.60 | 74.23 | 1100 | Raiders |
| 60 | 19 | Cedric Griffin | 70.85 | 67.10 | 75.30 | 920 | Vikings |
| 61 | 20 | Kyle Wilson | 70.40 | 62.30 | 72.01 | 569 | Jets |
| 62 | 21 | D.J. Moore | 70.25 | 60.60 | 74.46 | 485 | Bears |
| 63 | 22 | Adam Jones | 69.87 | 67.50 | 76.78 | 495 | Bengals |
| 64 | 23 | Keenan Lewis | 69.85 | 63.70 | 74.47 | 393 | Steelers |
| 65 | 24 | Drayton Florence | 69.80 | 58.70 | 73.03 | 1003 | Bills |
| 66 | 25 | Dominique Rodgers-Cromartie | 69.72 | 61.10 | 73.25 | 473 | Eagles |
| 67 | 26 | Joselio Hanson | 69.57 | 59.20 | 72.31 | 395 | Eagles |
| 68 | 27 | Justin Rogers | 69.57 | 63.30 | 76.89 | 214 | Bills |
| 69 | 28 | Jarrett Bush | 69.26 | 56.20 | 75.37 | 313 | Packers |
| 70 | 29 | Andre' Goodman | 68.76 | 62.80 | 71.70 | 1167 | Broncos |
| 71 | 30 | Rashean Mathis | 68.53 | 63.50 | 72.27 | 528 | Jaguars |
| 72 | 31 | Chris Carr | 67.72 | 62.10 | 71.85 | 184 | Ravens |
| 73 | 32 | Chris Rucker | 67.39 | 65.30 | 70.86 | 317 | Colts |
| 74 | 33 | Jacob Lacey | 67.37 | 62.50 | 68.92 | 701 | Colts |
| 75 | 34 | Tracy Porter | 67.09 | 57.60 | 70.41 | 807 | Saints |
| 76 | 35 | Nate Clements | 66.50 | 57.00 | 68.66 | 983 | Bengals |
| 77 | 36 | Captain Munnerlyn | 66.35 | 56.20 | 70.25 | 816 | Panthers |
| 78 | 37 | Chris Owens | 66.16 | 59.30 | 70.86 | 338 | Falcons |
| 79 | 38 | Kelly Jennings | 65.92 | 52.90 | 71.74 | 405 | Bengals |
| 80 | 39 | Buster Skrine | 65.41 | 55.70 | 72.91 | 123 | Browns |
| 81 | 40 | Josh Gordy | 65.25 | 56.50 | 69.51 | 674 | Rams |
| 82 | 41 | Dunta Robinson | 65.21 | 56.60 | 66.78 | 1034 | Falcons |
| 83 | 42 | Alphonso Smith | 65.19 | 59.00 | 71.92 | 257 | Lions |
| 84 | 43 | Orlando Scandrick | 65.11 | 56.40 | 68.70 | 658 | Cowboys |
| 85 | 44 | Alan Ball | 64.56 | 49.20 | 70.63 | 478 | Cowboys |
| 86 | 45 | DeMarcus Van Dyke | 64.55 | 57.60 | 71.26 | 319 | Raiders |
| 87 | 46 | Bradley Fletcher | 64.28 | 56.70 | 72.98 | 294 | Rams |
| 88 | 47 | Eric Wright | 64.27 | 53.60 | 68.39 | 1089 | Lions |
| 89 | 48 | R.J. Stanford | 63.84 | 61.10 | 67.75 | 255 | Panthers |
| 90 | 49 | Dominique Franks | 63.30 | 57.20 | 74.27 | 403 | Falcons |
| 91 | 50 | E.J. Biggers | 63.19 | 50.70 | 67.35 | 660 | Buccaneers |
| 92 | 51 | Sean Smith | 62.96 | 51.30 | 66.95 | 1064 | Dolphins |

### Rotation/backup (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 93 | 1 | Terence Newman | 61.67 | 49.30 | 67.05 | 799 | Cowboys |
| 94 | 2 | Kevin Barnes | 61.65 | 51.90 | 67.37 | 418 | Commanders |
| 95 | 3 | Kareem Jackson | 61.64 | 44.70 | 68.77 | 621 | Texans |
| 96 | 4 | A.J. Jefferson | 61.55 | 51.30 | 70.08 | 789 | Cardinals |
| 97 | 5 | Nolan Carroll | 61.38 | 51.40 | 71.42 | 321 | Dolphins |
| 98 | 6 | Antwaun Molden | 61.18 | 52.20 | 69.38 | 345 | Patriots |
| 99 | 7 | Brandon McDonald | 60.32 | 57.00 | 61.62 | 153 | Lions |
| 100 | 8 | Ronde Barber | 60.15 | 43.00 | 67.42 | 968 | Buccaneers |
| 101 | 9 | Patrick Peterson | 60.05 | 47.50 | 64.25 | 1098 | Cardinals |
| 102 | 10 | Terrence McGee | 59.36 | 53.10 | 68.61 | 280 | Bills |
| 103 | 11 | Prince Amukamara | 59.20 | 59.50 | 62.13 | 199 | Giants |
| 104 | 12 | Dante Hughes | 57.57 | 48.30 | 62.71 | 429 | Chargers |
| 105 | 13 | Kevin Thomas | 57.02 | 57.60 | 59.77 | 428 | Colts |
| 106 | 14 | Marcus Trufant | 56.12 | 46.00 | 66.51 | 268 | Seahawks |
| 107 | 15 | Reggie Corner | 55.79 | 44.30 | 66.45 | 128 | Bills |
| 108 | 16 | Marcus Sherels | 55.13 | 41.60 | 63.23 | 296 | Vikings |
| 109 | 17 | Asher Allen | 54.98 | 42.10 | 63.18 | 520 | Vikings |
| 110 | 18 | Zackary Bowman | 53.93 | 46.20 | 61.44 | 106 | Bears |
| 111 | 19 | Kelvin Hayden | 53.55 | 38.50 | 66.59 | 218 | Falcons |
| 112 | 20 | Cassius Vaughn | 52.76 | 46.20 | 65.20 | 232 | Broncos |
| 113 | 21 | Chris Johnson | 51.09 | 32.30 | 68.83 | 157 | Raiders |
| 114 | 22 | Kevin Rutland | 49.31 | 42.40 | 60.61 | 266 | Jaguars |
| 115 | 23 | Terrence Johnson | 45.90 | 45.00 | 49.64 | 277 | Colts |
| 116 | 24 | Ashton Youboty | 45.32 | 40.00 | 55.50 | 301 | Jaguars |
| 117 | 25 | Justin King | 45.00 | 28.50 | 52.57 | 706 | Rams |
| 118 | 26 | Chimdi Chekwa | 45.00 | 40.50 | 49.52 | 123 | Raiders |

## DI — Defensive Interior

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | J.J. Watt | 91.35 | 86.43 | 90.46 | 878 | Texans |
| 2 | 2 | Geno Atkins | 87.38 | 85.71 | 84.32 | 772 | Bengals |
| 3 | 3 | Justin Smith | 85.82 | 84.51 | 82.53 | 1096 | 49ers |
| 4 | 4 | Marcell Dareus | 83.31 | 80.29 | 81.15 | 734 | Bills |
| 5 | 5 | Haloti Ngata | 83.04 | 85.72 | 77.08 | 918 | Ravens |
| 6 | 6 | Calais Campbell | 81.77 | 72.25 | 84.34 | 998 | Cardinals |
| 7 | 7 | Antonio Garay | 81.07 | 66.60 | 86.55 | 530 | Chargers |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Sammie Lee Hill | 79.97 | 72.95 | 80.87 | 452 | Lions |
| 9 | 2 | David Carter | 79.61 | 74.57 | 78.80 | 240 | Cardinals |
| 10 | 3 | Jurrell Casey | 78.57 | 72.88 | 78.19 | 654 | Titans |
| 11 | 4 | Sione Pouha | 78.51 | 76.00 | 76.02 | 619 | Jets |
| 12 | 5 | Brodrick Bunkley | 78.09 | 71.95 | 78.40 | 528 | Broncos |
| 13 | 6 | Ndamukong Suh | 78.04 | 69.62 | 80.13 | 809 | Lions |
| 14 | 7 | Cullen Jenkins | 77.75 | 65.80 | 81.93 | 631 | Eagles |
| 15 | 8 | Paul Soliai | 77.55 | 72.23 | 76.93 | 436 | Dolphins |
| 16 | 9 | Tyson Jackson | 77.44 | 75.65 | 75.64 | 595 | Chiefs |
| 17 | 10 | Vonnie Holliday | 77.33 | 62.02 | 83.75 | 152 | Cardinals |
| 18 | 11 | Steve McLendon | 77.10 | 73.33 | 80.78 | 217 | Steelers |
| 19 | 12 | John Henderson | 77.07 | 77.29 | 77.44 | 343 | Raiders |
| 20 | 13 | Richard Seymour | 76.94 | 71.22 | 77.75 | 845 | Raiders |
| 21 | 14 | Derek Landri | 76.87 | 73.56 | 77.51 | 348 | Eagles |
| 22 | 15 | Kevin Williams | 76.37 | 78.21 | 72.27 | 822 | Vikings |
| 23 | 16 | Sean Lissemore | 76.29 | 60.96 | 82.35 | 275 | Cowboys |
| 24 | 17 | Muhammad Wilkerson | 76.17 | 59.97 | 82.80 | 599 | Jets |
| 25 | 18 | Cory Redding | 75.74 | 60.63 | 81.65 | 608 | Ravens |
| 26 | 19 | Nick Fairley | 75.61 | 75.07 | 77.00 | 263 | Lions |
| 27 | 20 | Cam Thomas | 74.98 | 68.76 | 78.86 | 380 | Chargers |
| 28 | 21 | Alan Branch | 74.90 | 72.21 | 73.18 | 673 | Seahawks |
| 29 | 22 | Ray McDonald | 74.67 | 73.93 | 70.99 | 983 | 49ers |
| 30 | 23 | Kyle Williams | 74.33 | 65.60 | 83.15 | 218 | Bills |

### Starter (73 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Chris Canty | 73.66 | 69.82 | 72.06 | 822 | Giants |
| 32 | 2 | Kenyon Coleman | 73.55 | 56.78 | 80.57 | 413 | Cowboys |
| 33 | 3 | Glenn Dorsey | 73.50 | 70.15 | 72.21 | 623 | Chiefs |
| 34 | 4 | Vince Wilfork | 73.47 | 62.82 | 76.41 | 1149 | Patriots |
| 35 | 5 | Randy Starks | 73.30 | 66.94 | 73.37 | 657 | Dolphins |
| 36 | 6 | Mike Devito | 73.21 | 66.24 | 76.29 | 452 | Jets |
| 37 | 7 | Karl Klug | 73.00 | 61.49 | 76.51 | 507 | Titans |
| 38 | 8 | Gerald McCoy | 72.72 | 84.42 | 68.43 | 218 | Buccaneers |
| 39 | 9 | Henry Melton | 72.12 | 59.70 | 76.89 | 621 | Bears |
| 40 | 10 | Leger Douzable | 71.82 | 65.41 | 72.31 | 420 | Jaguars |
| 41 | 11 | Brandon Mebane | 71.61 | 64.00 | 73.30 | 734 | Seahawks |
| 42 | 12 | Cameron Heyward | 71.34 | 57.43 | 76.45 | 242 | Steelers |
| 43 | 13 | Linval Joseph | 71.14 | 63.39 | 76.44 | 746 | Giants |
| 44 | 14 | Ricky Jean Francois | 71.13 | 65.50 | 70.71 | 306 | 49ers |
| 45 | 15 | Tommy Kelly | 70.92 | 64.59 | 70.97 | 840 | Raiders |
| 46 | 16 | Dan Williams | 70.87 | 73.44 | 69.29 | 237 | Cardinals |
| 47 | 17 | Jay Ratliff | 70.55 | 66.21 | 69.27 | 722 | Cowboys |
| 48 | 18 | Tony Hargrove | 70.51 | 62.10 | 73.39 | 298 | Seahawks |
| 49 | 19 | Brett Keisel | 70.40 | 65.58 | 70.88 | 769 | Steelers |
| 50 | 20 | Wallace Gilberry | 70.32 | 57.52 | 74.69 | 366 | Chiefs |
| 51 | 21 | Ahtyba Rubin | 69.91 | 61.97 | 71.03 | 929 | Browns |
| 52 | 22 | Earl Mitchell | 69.89 | 56.23 | 75.22 | 320 | Texans |
| 53 | 23 | Jason Hatcher | 69.83 | 57.37 | 77.10 | 416 | Cowboys |
| 54 | 24 | Ropati Pitoitua | 69.48 | 52.38 | 78.80 | 338 | Jets |
| 55 | 25 | Kellen Heard | 69.47 | 52.71 | 77.51 | 254 | Bills |
| 56 | 26 | Stephen Bowen | 69.06 | 56.16 | 73.49 | 793 | Commanders |
| 57 | 27 | C.J. Wilson | 69.02 | 53.73 | 75.05 | 405 | Packers |
| 58 | 28 | Fred Evans | 68.81 | 55.48 | 76.67 | 293 | Vikings |
| 59 | 29 | Amobi Okoye | 68.70 | 60.68 | 69.88 | 593 | Bears |
| 60 | 30 | Mike Patterson | 68.68 | 57.50 | 72.61 | 638 | Eagles |
| 61 | 31 | Phil Taylor Sr. | 68.59 | 58.09 | 71.42 | 732 | Browns |
| 62 | 32 | Letroy Guion | 68.56 | 62.31 | 68.94 | 483 | Vikings |
| 63 | 33 | Darnell Dockett | 68.46 | 54.47 | 74.00 | 1001 | Cardinals |
| 64 | 34 | Domata Peko Sr. | 68.40 | 54.79 | 73.31 | 687 | Bengals |
| 65 | 35 | Stephen Paea | 68.34 | 63.64 | 72.50 | 309 | Bears |
| 66 | 36 | Shaun Smith | 68.23 | 51.26 | 76.02 | 283 | Titans |
| 67 | 37 | Ryan Pickett | 68.19 | 55.20 | 73.34 | 499 | Packers |
| 68 | 38 | Christian Ballard | 68.06 | 56.57 | 71.56 | 237 | Vikings |
| 69 | 39 | Vance Walker | 68.05 | 54.11 | 73.17 | 384 | Falcons |
| 70 | 40 | Antonio Smith | 68.02 | 53.22 | 73.72 | 808 | Texans |
| 71 | 41 | Shaun Cody | 67.98 | 54.75 | 72.63 | 437 | Texans |
| 72 | 42 | C.J. Mosley | 67.85 | 55.81 | 75.50 | 318 | Jaguars |
| 73 | 43 | Gary Gibson | 67.22 | 64.19 | 65.08 | 384 | Rams |
| 74 | 44 | Red Bryant | 66.90 | 61.25 | 70.02 | 707 | Seahawks |
| 75 | 45 | Marcus Thomas | 66.74 | 54.55 | 72.00 | 604 | Broncos |
| 76 | 46 | Tony McDaniel | 66.42 | 53.05 | 74.16 | 325 | Dolphins |
| 77 | 47 | Jonathan Babineaux | 66.26 | 61.28 | 66.72 | 593 | Falcons |
| 78 | 48 | Sedrick Ellis | 65.97 | 63.25 | 63.61 | 791 | Saints |
| 79 | 49 | Chris Neild | 65.84 | 54.62 | 70.19 | 160 | Commanders |
| 80 | 50 | Tim Jamison | 65.80 | 52.44 | 70.54 | 366 | Texans |
| 81 | 51 | Terrance Knighton | 65.80 | 59.80 | 67.59 | 528 | Jaguars |
| 82 | 52 | Adam Carriker | 65.70 | 51.27 | 71.16 | 604 | Commanders |
| 83 | 53 | Shaun Ellis | 65.57 | 48.81 | 72.57 | 415 | Patriots |
| 84 | 54 | Kyle Love | 65.28 | 57.37 | 68.73 | 681 | Patriots |
| 85 | 55 | Barry Cofield | 65.26 | 58.12 | 65.85 | 762 | Commanders |
| 86 | 56 | Corey Peters | 65.14 | 51.86 | 69.82 | 657 | Falcons |
| 87 | 57 | Andre Neblett | 65.05 | 54.59 | 73.98 | 369 | Panthers |
| 88 | 58 | Pat Sims | 65.01 | 51.34 | 73.99 | 291 | Bengals |
| 89 | 59 | Rocky Bernard | 64.93 | 53.56 | 69.13 | 457 | Giants |
| 90 | 60 | Jared Odrick | 64.90 | 62.03 | 68.51 | 580 | Dolphins |
| 91 | 61 | Josh Price-Brent | 64.86 | 68.49 | 61.53 | 133 | Cowboys |
| 92 | 62 | Tyson Alualu | 64.73 | 53.31 | 68.18 | 851 | Jaguars |
| 93 | 63 | Kendall Langford | 64.69 | 55.11 | 66.91 | 538 | Dolphins |
| 94 | 64 | Isaac Sopoaga | 64.47 | 53.22 | 67.80 | 482 | 49ers |
| 95 | 65 | Tommie Harris | 64.43 | 57.21 | 67.03 | 274 | Chargers |
| 96 | 66 | Corey Williams | 64.01 | 51.52 | 68.17 | 745 | Lions |
| 97 | 67 | Allen Bailey | 63.95 | 57.79 | 64.92 | 289 | Chiefs |
| 98 | 68 | B.J. Raji | 62.89 | 54.71 | 64.17 | 924 | Packers |
| 99 | 69 | Peria Jerry | 62.82 | 49.36 | 67.62 | 362 | Falcons |
| 100 | 70 | Brandon Deaderick | 62.64 | 51.49 | 70.20 | 376 | Patriots |
| 101 | 71 | Trevor Laws | 62.42 | 51.93 | 66.54 | 342 | Eagles |
| 102 | 72 | Brian Price | 62.26 | 51.46 | 70.25 | 494 | Buccaneers |
| 103 | 73 | Anthony Adams | 62.19 | 47.22 | 71.26 | 280 | Bears |

### Rotation/backup (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 104 | 1 | Marcus Spears | 61.71 | 49.05 | 69.11 | 385 | Cowboys |
| 105 | 2 | Ricardo Mathews | 61.45 | 54.77 | 68.25 | 334 | Colts |
| 106 | 3 | Tom Johnson | 61.34 | 50.77 | 65.25 | 326 | Saints |
| 107 | 4 | Darell Scott | 61.24 | 57.75 | 64.08 | 242 | Rams |
| 108 | 5 | Marcus Dixon | 60.88 | 49.38 | 69.47 | 422 | Jets |
| 109 | 6 | Andre Fluellen | 60.68 | 50.51 | 65.24 | 205 | Lions |
| 110 | 7 | Sen'Derrick Marks | 60.56 | 49.58 | 65.28 | 411 | Titans |
| 111 | 8 | Casey Hampton | 60.53 | 54.51 | 61.68 | 444 | Steelers |
| 112 | 9 | Clinton McDonald | 60.38 | 55.09 | 63.53 | 425 | Seahawks |
| 113 | 10 | Remi Ayodele | 60.20 | 50.68 | 62.38 | 252 | Vikings |
| 114 | 11 | Vaughn Martin | 60.04 | 49.35 | 65.35 | 611 | Chargers |
| 115 | 12 | Ryan McBean | 59.99 | 45.50 | 65.48 | 684 | Broncos |
| 116 | 13 | Fili Moala | 59.83 | 45.05 | 66.82 | 480 | Colts |
| 117 | 14 | Antonio Johnson | 59.83 | 49.03 | 63.65 | 502 | Colts |
| 118 | 15 | Jacques Cesaire | 59.66 | 40.41 | 70.92 | 242 | Chargers |
| 119 | 16 | Roy Miller | 59.57 | 46.16 | 64.35 | 494 | Buccaneers |
| 120 | 17 | Kedric Golston | 59.37 | 52.89 | 65.26 | 170 | Commanders |
| 121 | 18 | Dwan Edwards | 58.96 | 37.85 | 68.87 | 736 | Bills |
| 122 | 19 | Terrence Cody | 58.34 | 49.31 | 60.57 | 532 | Ravens |
| 123 | 20 | Drake Nevis | 58.23 | 58.69 | 69.72 | 163 | Colts |
| 124 | 21 | Corey Liuget | 57.95 | 49.16 | 60.67 | 452 | Chargers |
| 125 | 22 | Darrion Scott | 57.10 | 52.65 | 67.24 | 111 | Commanders |
| 126 | 23 | Arthur Jones | 56.73 | 49.57 | 62.80 | 276 | Ravens |
| 127 | 24 | Brandon McKinney | 55.36 | 53.88 | 55.31 | 203 | Ravens |
| 128 | 25 | Matt Toeaina | 55.20 | 48.64 | 58.00 | 390 | Bears |
| 129 | 26 | Igor Olshansky | 55.05 | 46.77 | 61.61 | 100 | Dolphins |
| 130 | 27 | Ogemdi Nwagbuo | 53.91 | 54.89 | 54.69 | 159 | Panthers |
| 131 | 28 | Sione Fua | 53.40 | 48.10 | 57.96 | 406 | Panthers |
| 132 | 29 | Terrell McClain | 52.54 | 43.14 | 58.80 | 471 | Panthers |
| 133 | 30 | Tim Bulman | 52.34 | 43.78 | 58.04 | 136 | Texans |
| 134 | 31 | Nick Eason | 51.04 | 37.22 | 56.09 | 249 | Cardinals |
| 135 | 32 | Frank Okam | 49.89 | 42.45 | 57.99 | 296 | Buccaneers |
| 136 | 33 | John McCargo | 49.77 | 49.03 | 64.60 | 109 | Buccaneers |
| 137 | 34 | Mitch Unrein | 48.44 | 42.67 | 49.16 | 105 | Broncos |
| 138 | 35 | Phillip Merling | 48.10 | 50.81 | 50.32 | 199 | Dolphins |
| 139 | 36 | Frank Kearse | 46.64 | 53.11 | 51.57 | 170 | Panthers |
| 140 | 37 | Aaron Smith | 46.64 | 37.93 | 60.00 | 172 | Steelers |
| 141 | 38 | Kevin Vickerson | 45.00 | 42.37 | 49.58 | 198 | Broncos |

## ED — Edge

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 93.45 | 95.83 | 87.70 | 985 | Broncos |
| 2 | 2 | DeMarcus Ware | 91.96 | 88.78 | 89.92 | 880 | Cowboys |
| 3 | 3 | Cameron Wake | 87.55 | 82.59 | 86.69 | 875 | Dolphins |
| 4 | 4 | Charles Johnson | 87.48 | 84.41 | 86.01 | 794 | Panthers |
| 5 | 5 | Jason Babin | 87.40 | 78.95 | 88.86 | 697 | Eagles |
| 6 | 6 | Terrell Suggs | 86.94 | 87.73 | 82.24 | 1117 | Ravens |
| 7 | 7 | Mario Williams | 84.97 | 92.11 | 84.38 | 217 | Texans |
| 8 | 8 | Jared Allen | 82.75 | 77.03 | 82.39 | 1005 | Vikings |
| 9 | 9 | Carlos Dunlap | 82.08 | 78.78 | 83.63 | 434 | Bengals |
| 10 | 10 | Chris Long | 81.50 | 76.37 | 80.75 | 920 | Rams |
| 11 | 11 | Jason Pierre-Paul | 80.23 | 87.05 | 71.52 | 1173 | Giants |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Cliff Avril | 79.21 | 72.40 | 80.75 | 856 | Lions |
| 13 | 2 | Mark Anderson | 79.13 | 70.54 | 81.08 | 625 | Patriots |
| 14 | 3 | Julius Peppers | 78.41 | 77.76 | 74.68 | 887 | Bears |
| 15 | 4 | Chris Clemons | 78.35 | 67.05 | 81.72 | 929 | Seahawks |
| 16 | 5 | Justin Tuck | 76.16 | 67.62 | 77.68 | 797 | Giants |
| 17 | 6 | Lawrence Jackson | 76.02 | 75.55 | 76.72 | 363 | Lions |
| 18 | 7 | Justin Houston | 75.97 | 65.33 | 78.89 | 754 | Chiefs |
| 19 | 8 | Phillip Hunt | 75.14 | 71.31 | 80.82 | 177 | Eagles |
| 20 | 9 | Ryan Kerrigan | 75.07 | 62.80 | 79.09 | 1026 | Commanders |

### Starter (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Ray Edwards | 73.89 | 69.39 | 73.51 | 718 | Falcons |
| 22 | 2 | Juqua Parker | 73.63 | 68.50 | 76.26 | 254 | Eagles |
| 23 | 3 | Jabaal Sheard | 73.62 | 66.34 | 74.31 | 949 | Browns |
| 24 | 4 | Darryl Tapp | 73.17 | 74.47 | 71.13 | 304 | Eagles |
| 25 | 5 | Adrian Clayborn | 72.19 | 63.12 | 74.07 | 849 | Buccaneers |
| 26 | 6 | Everson Griffen | 72.10 | 68.75 | 72.90 | 271 | Vikings |
| 27 | 7 | Brooks Reed | 71.97 | 66.02 | 71.77 | 902 | Texans |
| 28 | 8 | Anthony Spencer | 71.39 | 68.30 | 69.28 | 931 | Cowboys |
| 29 | 9 | Mathias Kiwanuka | 70.58 | 60.85 | 77.98 | 948 | Giants |
| 30 | 10 | Shaun Phillips | 69.02 | 57.93 | 74.85 | 626 | Chargers |
| 31 | 11 | Jeremy Mincey | 68.82 | 64.88 | 67.67 | 950 | Jaguars |
| 32 | 12 | Parys Haralson | 68.75 | 58.63 | 71.72 | 570 | 49ers |
| 33 | 13 | Matt Shaughnessy | 67.79 | 66.74 | 72.79 | 143 | Raiders |
| 34 | 14 | Robert Quinn | 67.59 | 61.16 | 68.75 | 561 | Rams |
| 35 | 15 | Jarvis Moss | 67.19 | 59.52 | 71.00 | 298 | Raiders |
| 36 | 16 | James Hall | 66.43 | 52.51 | 72.20 | 718 | Rams |
| 37 | 17 | Matt Roth | 66.04 | 60.35 | 70.22 | 401 | Jaguars |
| 38 | 18 | Brian Robison | 65.94 | 60.12 | 65.66 | 900 | Vikings |
| 39 | 19 | Antwan Applewhite | 65.61 | 59.74 | 68.35 | 335 | Panthers |
| 40 | 20 | O'Brien Schofield | 65.53 | 56.20 | 70.31 | 433 | Cardinals |
| 41 | 21 | Michael Bennett | 65.28 | 70.39 | 60.17 | 610 | Buccaneers |
| 42 | 22 | Greg Hardy | 65.09 | 63.37 | 62.46 | 890 | Panthers |
| 43 | 23 | Jonathan Fanene | 64.23 | 58.53 | 69.33 | 472 | Bengals |
| 44 | 24 | Turk McBride | 63.22 | 61.10 | 66.06 | 205 | Saints |
| 45 | 25 | Robert Ayers | 62.32 | 63.08 | 59.60 | 732 | Broncos |
| 46 | 26 | Kroy Biermann | 62.17 | 59.17 | 60.00 | 556 | Falcons |
| 47 | 27 | Jamaal Anderson | 62.01 | 61.62 | 58.76 | 412 | Colts |

### Rotation/backup (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 48 | 1 | Israel Idonije | 61.74 | 55.00 | 62.06 | 915 | Bears |
| 49 | 2 | Frostee Rucker | 61.37 | 57.80 | 62.31 | 475 | Bengals |
| 50 | 3 | Michael Johnson | 60.78 | 58.81 | 57.93 | 708 | Bengals |
| 51 | 4 | Chris Kelsay | 60.33 | 51.51 | 64.64 | 674 | Bills |
| 52 | 5 | Kyle Vanden Bosch | 60.22 | 51.34 | 63.92 | 799 | Lions |
| 53 | 6 | Cameron Jordan | 60.20 | 62.48 | 54.51 | 648 | Saints |
| 54 | 7 | Jason Hunter | 59.36 | 55.77 | 57.59 | 406 | Broncos |
| 55 | 8 | Dave Ball | 58.68 | 51.88 | 62.30 | 647 | Titans |
| 56 | 9 | William Hayes | 57.66 | 58.25 | 57.78 | 330 | Titans |
| 57 | 10 | Robert Geathers | 56.40 | 54.85 | 53.92 | 534 | Bengals |
| 58 | 11 | Danny Batten | 55.15 | 56.75 | 55.12 | 224 | Bills |
| 59 | 12 | Jerry Hughes | 54.92 | 57.25 | 55.19 | 150 | Colts |
| 60 | 13 | Dave Tollefson | 54.75 | 49.39 | 56.11 | 551 | Giants |
| 61 | 14 | C.J. Ah You | 54.69 | 49.74 | 58.37 | 246 | Rams |
| 62 | 15 | Austen Lane | 54.49 | 57.65 | 56.69 | 126 | Jaguars |
| 63 | 16 | Jason Jones | 53.93 | 50.18 | 53.57 | 675 | Titans |
| 64 | 17 | Trevor Scott | 52.64 | 51.93 | 51.95 | 243 | Raiders |
| 65 | 18 | Eugene Sims | 51.58 | 53.79 | 50.63 | 276 | Rams |
| 66 | 19 | Emmanuel Stephens | 51.17 | 56.70 | 49.57 | 148 | Browns |
| 67 | 20 | Mario Addison | 45.43 | 55.21 | 45.61 | 131 | Colts |

## G — Guard

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Evan Mathis | 94.33 | 92.80 | 91.18 | 996 | Eagles |
| 2 | 2 | Marshal Yanda | 90.42 | 84.70 | 90.06 | 1172 | Ravens |
| 3 | 3 | Joe Berger | 89.73 | 83.00 | 90.05 | 493 | Vikings |
| 4 | 4 | Mike Brisiel | 89.47 | 81.70 | 90.48 | 982 | Texans |
| 5 | 5 | Carl Nicks | 88.81 | 83.00 | 88.52 | 1320 | Saints |
| 6 | 6 | Josh Sitton | 88.75 | 81.80 | 89.22 | 924 | Packers |
| 7 | 7 | Kris Dielman | 87.84 | 81.70 | 87.76 | 419 | Chargers |
| 8 | 8 | Brian Waters | 87.75 | 82.00 | 87.41 | 1310 | Patriots |
| 9 | 9 | Steve Hutchinson | 87.47 | 81.20 | 87.49 | 889 | Vikings |
| 10 | 10 | Andy Levitre | 86.88 | 81.40 | 86.36 | 1043 | Bills |
| 11 | 11 | Mike Iupati | 86.31 | 79.60 | 86.61 | 1147 | 49ers |
| 12 | 12 | Travelle Wharton | 85.58 | 77.60 | 86.74 | 1000 | Panthers |
| 13 | 13 | Logan Mankins | 85.23 | 75.90 | 87.29 | 1162 | Patriots |
| 14 | 14 | Jahri Evans | 84.36 | 76.50 | 85.44 | 1283 | Saints |
| 15 | 15 | Brandon Moore | 83.92 | 75.70 | 85.23 | 1078 | Jets |
| 16 | 16 | Antoine Caldwell | 83.51 | 74.80 | 85.15 | 221 | Texans |
| 17 | 17 | Jake Scott | 83.28 | 76.30 | 83.77 | 1022 | Titans |
| 18 | 18 | T.J. Lang | 83.16 | 73.60 | 85.37 | 1109 | Packers |
| 19 | 19 | Richie Incognito | 83.12 | 74.20 | 84.90 | 928 | Dolphins |
| 20 | 20 | Harvey Dahl | 82.85 | 74.00 | 84.59 | 1055 | Rams |
| 21 | 21 | Ben Grubbs | 82.18 | 74.90 | 82.87 | 794 | Ravens |
| 22 | 22 | Jon Asamoah | 81.86 | 74.00 | 82.94 | 1051 | Chiefs |
| 23 | 23 | Uche Nwaneri | 81.83 | 73.90 | 82.95 | 1041 | Jaguars |
| 24 | 24 | Daryn Colledge | 81.50 | 71.80 | 83.80 | 990 | Cardinals |
| 25 | 25 | Bobbie Williams | 81.35 | 72.00 | 83.41 | 560 | Bengals |
| 26 | 26 | Ramon Foster | 81.15 | 73.50 | 82.09 | 976 | Steelers |
| 27 | 27 | Rob Sims | 80.79 | 73.00 | 81.81 | 1157 | Lions |
| 28 | 28 | Stefen Wisniewski | 80.35 | 71.90 | 81.81 | 1073 | Raiders |
| 29 | 29 | Chad Rinehart | 80.03 | 74.30 | 79.69 | 853 | Bills |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Stephen Peterman | 79.88 | 71.30 | 81.44 | 1157 | Lions |
| 31 | 2 | Geoff Hangartner | 79.20 | 70.30 | 80.96 | 1046 | Panthers |
| 32 | 3 | Justin Blalock | 79.12 | 71.50 | 80.03 | 1191 | Falcons |
| 33 | 4 | Kyle Kosier | 78.94 | 70.50 | 80.40 | 1018 | Cowboys |
| 34 | 5 | Shawn Lauvao | 78.73 | 67.70 | 81.92 | 1013 | Browns |
| 35 | 6 | Wade Smith | 78.66 | 70.20 | 80.13 | 1203 | Texans |
| 36 | 7 | Matt Slauson | 78.45 | 68.90 | 80.65 | 1088 | Jets |
| 37 | 8 | Nate Livings | 77.43 | 67.70 | 79.75 | 1135 | Bengals |
| 38 | 9 | Louis Vasquez | 76.49 | 67.80 | 78.11 | 894 | Chargers |
| 39 | 10 | Chris Kemoeatu | 76.27 | 65.00 | 79.62 | 758 | Steelers |
| 40 | 11 | Kraig Urbik | 76.06 | 72.80 | 74.06 | 750 | Bills |
| 41 | 12 | Maurice Hurt | 75.77 | 65.20 | 78.65 | 548 | Commanders |
| 42 | 13 | Cooper Carlisle | 75.32 | 65.90 | 77.44 | 1072 | Raiders |
| 43 | 14 | Joe Reitz | 75.31 | 64.60 | 78.28 | 530 | Colts |
| 44 | 15 | Jacob Bell | 75.29 | 63.80 | 78.78 | 713 | Rams |
| 45 | 16 | Davin Joseph | 75.15 | 66.50 | 76.75 | 1022 | Buccaneers |
| 46 | 17 | Rex Hadnot | 75.06 | 65.40 | 77.34 | 985 | Cardinals |
| 47 | 18 | Chris Snee | 74.97 | 64.60 | 77.72 | 1277 | Giants |
| 48 | 19 | Adam Snyder | 74.97 | 64.00 | 78.12 | 964 | 49ers |
| 49 | 20 | Zane Beadles | 74.81 | 64.10 | 77.78 | 1195 | Broncos |

### Starter (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 50 | 1 | Mike McGlynn | 73.81 | 61.20 | 78.05 | 408 | Bengals |
| 51 | 2 | Chris Kuper | 73.77 | 63.60 | 76.38 | 997 | Broncos |
| 52 | 3 | John Jerry | 73.72 | 63.90 | 76.10 | 357 | Dolphins |
| 53 | 4 | Artis Hicks | 73.46 | 57.30 | 80.07 | 191 | Browns |
| 54 | 5 | Chris Spencer | 73.03 | 65.20 | 74.09 | 901 | Bears |
| 55 | 6 | Chris Chester | 72.58 | 63.30 | 74.60 | 1081 | Commanders |
| 56 | 7 | Kevin Boothe | 72.10 | 61.00 | 75.33 | 980 | Giants |
| 57 | 8 | Jason Pinkston | 72.02 | 61.10 | 75.13 | 1063 | Browns |
| 58 | 9 | Danny Watkins | 71.68 | 63.60 | 72.90 | 787 | Eagles |
| 59 | 10 | Ted Larsen | 71.17 | 54.70 | 77.99 | 229 | Buccaneers |
| 60 | 11 | Colin Brown | 71.04 | 66.40 | 69.97 | 134 | Bills |
| 61 | 12 | Chris Williams | 70.45 | 58.90 | 73.98 | 538 | Bears |
| 62 | 13 | Steve Schilling | 70.37 | 65.90 | 69.19 | 137 | Chargers |
| 63 | 14 | Donald Thomas | 69.88 | 61.90 | 71.03 | 103 | Patriots |
| 64 | 15 | Tyronne Green | 69.13 | 58.40 | 72.11 | 556 | Chargers |
| 65 | 16 | Chilo Rachal | 69.03 | 55.60 | 73.82 | 240 | 49ers |
| 66 | 17 | Bill Nagy | 68.02 | 60.10 | 69.14 | 274 | Cowboys |
| 67 | 18 | Will Rackley | 67.31 | 53.40 | 72.41 | 940 | Jaguars |
| 68 | 19 | Mitch Petrus | 67.09 | 56.30 | 70.12 | 224 | Giants |
| 69 | 20 | Russ Hochstein | 66.71 | 54.90 | 70.42 | 199 | Broncos |
| 70 | 21 | Bryan Mattison | 66.52 | 53.60 | 70.97 | 269 | Rams |
| 71 | 22 | Garrett Reynolds | 65.85 | 50.20 | 72.11 | 511 | Falcons |
| 72 | 23 | John Moffitt | 65.21 | 52.60 | 69.45 | 502 | Seahawks |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 73 | 1 | Clint Boling | 57.54 | 48.10 | 59.66 | 168 | Bengals |

## HB — Running Back

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Darren Sproles | 82.51 | 79.60 | 80.29 | 433 | Saints |
| 2 | 2 | Isaac Redman | 82.27 | 86.00 | 75.61 | 131 | Steelers |
| 3 | 3 | Fred Jackson | 81.60 | 82.60 | 76.76 | 238 | Bills |
| 4 | 4 | Pierre Thomas | 80.47 | 88.00 | 71.29 | 231 | Saints |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Adrian Peterson | 79.16 | 83.70 | 71.96 | 165 | Vikings |
| 6 | 2 | Jonathan Stewart | 78.90 | 81.00 | 73.34 | 254 | Panthers |
| 7 | 3 | Darren McFadden | 78.85 | 78.70 | 74.78 | 104 | Raiders |
| 8 | 4 | Matt Forte | 78.40 | 85.10 | 69.77 | 256 | Bears |
| 9 | 5 | Ben Tate | 77.43 | 75.10 | 74.81 | 108 | Texans |
| 10 | 6 | Arian Foster | 77.22 | 82.50 | 69.53 | 305 | Texans |
| 11 | 7 | LeGarrette Blount | 75.85 | 71.00 | 74.91 | 116 | Buccaneers |
| 12 | 8 | LeSean McCoy | 75.47 | 73.80 | 72.42 | 388 | Eagles |
| 13 | 9 | DeMarco Murray | 75.21 | 71.50 | 73.51 | 148 | Cowboys |
| 14 | 10 | Toby Gerhart | 75.04 | 77.00 | 69.57 | 157 | Vikings |
| 15 | 11 | C.J. Spiller | 74.63 | 75.90 | 69.61 | 246 | Bills |
| 16 | 12 | Ryan Mathews | 74.43 | 73.00 | 71.22 | 229 | Chargers |

### Starter (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Ahmad Bradshaw | 73.98 | 74.20 | 69.67 | 242 | Giants |
| 18 | 2 | Maurice Jones-Drew | 73.56 | 74.50 | 68.76 | 302 | Jaguars |
| 19 | 3 | Marshawn Lynch | 73.48 | 76.40 | 67.37 | 185 | Seahawks |
| 20 | 4 | Felix Jones | 73.21 | 72.20 | 69.72 | 134 | Cowboys |
| 21 | 5 | Rashard Mendenhall | 72.97 | 81.50 | 63.12 | 162 | Steelers |
| 22 | 6 | James Starks | 72.61 | 77.00 | 65.52 | 209 | Packers |
| 23 | 7 | Brandon Jacobs | 72.43 | 73.60 | 67.49 | 117 | Giants |
| 24 | 8 | Ray Rice | 72.04 | 75.70 | 65.43 | 412 | Ravens |
| 25 | 9 | Michael Turner | 71.71 | 71.00 | 68.01 | 188 | Falcons |
| 26 | 10 | DeAngelo Williams | 71.61 | 69.30 | 68.98 | 179 | Panthers |
| 27 | 11 | Kendall Hunter | 71.53 | 77.50 | 63.38 | 131 | 49ers |
| 28 | 12 | Willis McGahee | 71.46 | 76.00 | 64.26 | 143 | Broncos |
| 29 | 13 | Donald Brown | 69.90 | 68.80 | 66.46 | 171 | Colts |
| 30 | 14 | Kahlil Bell | 69.02 | 68.00 | 65.54 | 100 | Bears |
| 31 | 15 | Steven Jackson | 68.82 | 69.70 | 64.07 | 292 | Rams |
| 32 | 16 | Reggie Bush | 68.41 | 62.60 | 68.11 | 252 | Dolphins |
| 33 | 17 | Dexter McCluster | 68.39 | 68.90 | 63.88 | 234 | Chiefs |
| 34 | 18 | BenJarvus Green-Ellis | 68.36 | 74.00 | 60.43 | 146 | Patriots |
| 35 | 19 | Peyton Hillis | 67.94 | 65.80 | 65.20 | 200 | Browns |
| 36 | 20 | Kevin Smith | 67.70 | 66.00 | 64.67 | 180 | Lions |
| 37 | 21 | Chris Johnson | 67.32 | 64.10 | 65.30 | 328 | Titans |
| 38 | 22 | Roy Helu | 67.21 | 65.00 | 64.52 | 276 | Commanders |
| 39 | 23 | Michael Bush | 67.13 | 70.00 | 61.05 | 241 | Raiders |
| 40 | 24 | Frank Gore | 67.06 | 63.50 | 65.26 | 260 | 49ers |
| 41 | 25 | Carnell Williams | 65.89 | 65.00 | 62.32 | 127 | Rams |
| 42 | 26 | Justin Forsett | 65.77 | 64.50 | 62.45 | 155 | Seahawks |
| 43 | 27 | Ryan Grant | 65.65 | 65.90 | 61.32 | 141 | Packers |
| 44 | 28 | Jahvid Best | 65.45 | 69.20 | 58.78 | 162 | Lions |
| 45 | 29 | Shonn Greene | 65.27 | 63.80 | 62.08 | 183 | Jets |
| 46 | 30 | LaDainian Tomlinson | 65.16 | 63.90 | 61.83 | 237 | Jets |
| 47 | 31 | Jacquizz Rodgers | 64.91 | 61.80 | 62.81 | 190 | Falcons |
| 48 | 32 | Jason Snelling | 64.88 | 62.70 | 62.17 | 144 | Falcons |
| 49 | 33 | Joseph Addai | 64.81 | 65.40 | 60.25 | 142 | Colts |
| 50 | 34 | Javon Ringer | 64.42 | 65.40 | 59.60 | 137 | Titans |
| 51 | 35 | Danny Woodhead | 64.34 | 62.00 | 61.74 | 253 | Patriots |
| 52 | 36 | Cedric Benson | 63.70 | 66.10 | 57.93 | 200 | Bengals |
| 53 | 37 | Lance Ball | 62.72 | 62.90 | 58.43 | 200 | Broncos |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 54 | 1 | Chris Wells | 61.01 | 56.40 | 59.91 | 190 | Cardinals |
| 55 | 2 | Maurice Morris | 60.93 | 57.10 | 59.32 | 187 | Lions |
| 56 | 3 | Chester Taylor | 58.76 | 56.90 | 55.84 | 123 | Cardinals |
| 57 | 4 | Montario Hardesty | 56.90 | 51.20 | 56.54 | 116 | Browns |
| 58 | 5 | Tashard Choice | 54.20 | 47.30 | 54.63 | 116 | Bills |

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

### Good (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Derrick Johnson | 79.69 | 81.80 | 74.12 | 1021 | Chiefs |
| 9 | 2 | London Fletcher | 79.01 | 80.40 | 73.91 | 1003 | Commanders |
| 10 | 3 | Daryl Smith | 78.80 | 79.30 | 74.30 | 897 | Jaguars |
| 11 | 4 | Mario Haggan | 78.57 | 78.20 | 75.30 | 188 | Broncos |
| 12 | 5 | Brian Cushing | 78.52 | 81.80 | 73.73 | 1064 | Texans |
| 13 | 6 | Karlos Dansby | 77.56 | 78.70 | 73.41 | 905 | Dolphins |
| 14 | 7 | Erin Henderson | 77.51 | 81.80 | 75.43 | 573 | Vikings |
| 15 | 8 | James Laurinaitis | 77.28 | 75.40 | 74.36 | 1071 | Rams |
| 16 | 9 | Desmond Bishop | 76.49 | 77.60 | 72.89 | 899 | Packers |
| 17 | 10 | David Hawthorne | 75.51 | 75.00 | 72.72 | 988 | Seahawks |
| 18 | 11 | Donald Butler | 75.46 | 75.10 | 71.53 | 634 | Chargers |
| 19 | 12 | Daryl Washington | 75.35 | 75.70 | 71.60 | 986 | Cardinals |
| 20 | 13 | Bart Scott | 75.08 | 76.10 | 70.23 | 667 | Jets |
| 21 | 14 | Curtis Lofton | 74.16 | 70.50 | 72.44 | 1053 | Falcons |
| 22 | 15 | David Harris | 74.06 | 72.20 | 71.14 | 961 | Jets |
| 23 | 16 | Brandon Spikes | 74.05 | 75.30 | 73.49 | 535 | Patriots |

### Starter (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Pat Angerer | 73.16 | 73.00 | 70.27 | 1027 | Colts |
| 25 | 2 | Jerod Mayo | 71.88 | 70.20 | 68.83 | 1103 | Patriots |
| 26 | 3 | Nick Barnett | 71.86 | 73.50 | 71.29 | 956 | Bills |
| 27 | 4 | E.J. Henderson | 71.67 | 67.70 | 70.15 | 862 | Vikings |
| 28 | 5 | Philip Wheeler | 71.25 | 69.60 | 70.14 | 532 | Colts |
| 29 | 6 | Dan Connor | 70.82 | 71.30 | 70.12 | 471 | Panthers |
| 30 | 7 | Lance Briggs | 70.38 | 65.70 | 69.33 | 1081 | Bears |
| 31 | 8 | Leroy Hill | 70.23 | 67.00 | 68.22 | 961 | Seahawks |
| 32 | 9 | Sean Lee | 69.86 | 68.70 | 68.28 | 838 | Cowboys |
| 33 | 10 | Chase Blackburn | 69.42 | 73.70 | 71.25 | 375 | Giants |
| 34 | 11 | Brendon Ayanbadejo | 69.25 | 71.70 | 68.91 | 308 | Ravens |
| 35 | 12 | Sean Weatherspoon | 69.19 | 68.20 | 67.25 | 1043 | Falcons |
| 36 | 13 | Kelvin Sheppard | 69.10 | 68.10 | 68.73 | 435 | Bills |
| 37 | 14 | Bobby Carpenter | 69.08 | 65.90 | 67.03 | 248 | Lions |
| 38 | 15 | Lawrence Timmons | 68.50 | 63.10 | 67.93 | 1029 | Steelers |
| 39 | 16 | Jameel McClain | 67.92 | 65.80 | 65.16 | 814 | Ravens |
| 40 | 17 | Keenan Clayton | 67.87 | 70.80 | 69.57 | 149 | Eagles |
| 41 | 18 | Takeo Spikes | 67.78 | 62.60 | 67.06 | 918 | Chargers |
| 42 | 19 | K.J. Wright | 67.52 | 65.60 | 65.66 | 562 | Seahawks |
| 43 | 20 | Colin McCarthy | 67.40 | 68.90 | 67.43 | 534 | Titans |
| 44 | 21 | Chris Chamberlain | 67.35 | 64.70 | 67.69 | 597 | Rams |
| 45 | 22 | Russell Allen | 67.20 | 66.10 | 67.94 | 293 | Jaguars |
| 46 | 23 | James Anderson | 66.45 | 60.30 | 66.39 | 996 | Panthers |
| 47 | 24 | DeMeco Ryans | 66.42 | 65.50 | 66.76 | 663 | Texans |
| 48 | 25 | Brian Rolle | 66.15 | 62.70 | 64.29 | 644 | Eagles |
| 49 | 26 | Brandon Johnson | 66.06 | 57.90 | 67.98 | 319 | Bengals |
| 50 | 27 | Rolando McClain | 65.94 | 61.30 | 65.90 | 1007 | Raiders |
| 51 | 28 | D.J. Smith | 65.86 | 75.60 | 71.16 | 262 | Packers |
| 52 | 29 | A.J. Hawk | 65.79 | 62.80 | 64.26 | 943 | Packers |
| 53 | 30 | Will Witherspoon | 65.70 | 60.80 | 64.80 | 665 | Titans |
| 54 | 31 | Josh Mauga | 65.50 | 65.90 | 66.66 | 142 | Jets |
| 55 | 32 | Akeem Jordan | 65.36 | 64.90 | 67.88 | 236 | Eagles |
| 56 | 33 | Greg Jones | 65.02 | 61.80 | 64.03 | 200 | Giants |
| 57 | 34 | Rey Maualuga | 64.97 | 60.50 | 65.08 | 728 | Bengals |
| 58 | 35 | Michael Boley | 64.91 | 60.90 | 63.42 | 1146 | Giants |
| 59 | 36 | Stephen Nicholas | 64.89 | 64.00 | 65.22 | 281 | Falcons |
| 60 | 37 | Aaron Curry | 64.89 | 56.90 | 66.05 | 695 | Raiders |
| 61 | 38 | Jovan Belcher | 64.84 | 59.40 | 64.30 | 649 | Chiefs |
| 62 | 39 | Akeem Ayers | 64.60 | 59.20 | 64.03 | 818 | Titans |
| 63 | 40 | Perry Riley | 64.07 | 66.70 | 67.40 | 495 | Commanders |
| 64 | 41 | Kevin Burnett | 64.05 | 57.10 | 64.51 | 1020 | Dolphins |
| 65 | 42 | Thomas Howard | 64.00 | 61.70 | 65.66 | 1032 | Bengals |
| 66 | 43 | Justin Durant | 63.74 | 62.40 | 64.12 | 626 | Lions |
| 67 | 44 | Andra Davis | 63.71 | 64.20 | 65.73 | 125 | Bills |
| 68 | 45 | Kaluka Maiava | 63.46 | 63.50 | 61.21 | 268 | Browns |
| 69 | 46 | Larry Grant | 63.36 | 65.60 | 65.38 | 226 | 49ers |
| 70 | 47 | Mike Peterson | 63.34 | 62.20 | 64.49 | 214 | Falcons |
| 71 | 48 | D.J. Williams | 63.26 | 57.40 | 63.65 | 1040 | Broncos |
| 72 | 49 | Dekoda Watson | 63.03 | 61.10 | 64.59 | 212 | Buccaneers |
| 73 | 50 | Darryl Blackstock | 63.03 | 70.60 | 72.31 | 112 | Raiders |
| 74 | 51 | Jacquian Williams | 63.00 | 56.40 | 63.23 | 573 | Giants |
| 75 | 52 | Thomas Williams | 62.74 | 71.30 | 68.84 | 104 | Panthers |
| 76 | 53 | Martez Wilson | 62.60 | 67.40 | 62.54 | 133 | Saints |
| 77 | 54 | Paris Lenon | 62.58 | 52.80 | 64.94 | 1086 | Cardinals |

### Rotation/backup (39 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 78 | 1 | Bradie James | 61.99 | 54.80 | 62.61 | 402 | Cowboys |
| 79 | 2 | Dane Fletcher | 61.94 | 61.60 | 62.55 | 299 | Patriots |
| 80 | 3 | Omar Gaither | 61.59 | 64.40 | 66.10 | 237 | Panthers |
| 81 | 4 | Stewart Bradley | 61.58 | 56.10 | 63.28 | 228 | Cardinals |
| 82 | 5 | Wesley Woodyard | 61.41 | 55.30 | 64.84 | 639 | Broncos |
| 83 | 6 | James Farrior | 61.37 | 54.50 | 62.43 | 767 | Steelers |
| 84 | 7 | Nick Roach | 61.01 | 58.00 | 61.58 | 520 | Bears |
| 85 | 8 | Tracy White | 60.30 | 60.60 | 65.70 | 246 | Patriots |
| 86 | 9 | DeAndre Levy | 60.15 | 54.00 | 62.03 | 997 | Lions |
| 87 | 10 | Jason Williams | 59.99 | 63.30 | 64.42 | 104 | Panthers |
| 88 | 11 | Chad Greenway | 59.93 | 50.00 | 62.39 | 1059 | Vikings |
| 89 | 12 | Chris Gocong | 59.83 | 53.70 | 59.75 | 863 | Browns |
| 90 | 13 | Scott Shanle | 59.34 | 50.20 | 61.65 | 914 | Saints |
| 91 | 14 | Robert Francois | 58.78 | 69.50 | 65.96 | 165 | Packers |
| 92 | 15 | Marvin Mitchell | 58.61 | 54.80 | 59.58 | 178 | Dolphins |
| 93 | 16 | Jonathan Vilma | 58.33 | 47.80 | 63.13 | 762 | Saints |
| 94 | 17 | Moise Fokou | 58.29 | 53.80 | 61.41 | 227 | Eagles |
| 95 | 18 | Adam Hayward | 58.17 | 52.90 | 61.16 | 180 | Buccaneers |
| 96 | 19 | Scott Fujita | 58.17 | 53.00 | 61.35 | 633 | Browns |
| 97 | 20 | Quentin Groves | 58.06 | 50.20 | 63.17 | 204 | Raiders |
| 98 | 21 | Jamar Chaney | 57.71 | 53.00 | 60.59 | 848 | Eagles |
| 99 | 22 | Rocky McIntosh | 57.55 | 52.30 | 59.88 | 498 | Commanders |
| 100 | 23 | Brady Poppinga | 57.15 | 50.20 | 58.27 | 553 | Rams |
| 101 | 24 | Keith Brooking | 57.10 | 48.80 | 58.46 | 395 | Cowboys |
| 102 | 25 | Larry Foote | 55.79 | 47.00 | 57.49 | 427 | Steelers |
| 103 | 26 | Mason Foster | 55.63 | 43.40 | 59.62 | 865 | Buccaneers |
| 104 | 27 | Gary Guyton | 55.63 | 45.50 | 61.46 | 397 | Patriots |
| 105 | 28 | Kavell Conner | 55.15 | 46.00 | 59.04 | 788 | Colts |
| 106 | 29 | Na'il Diggs | 55.12 | 46.50 | 60.22 | 354 | Chargers |
| 107 | 30 | Jonathan Casillas | 54.27 | 48.30 | 57.21 | 545 | Saints |
| 108 | 31 | Clint Session | 53.81 | 49.60 | 61.30 | 254 | Jaguars |
| 109 | 32 | Jordan Senn | 53.76 | 46.60 | 63.22 | 398 | Panthers |
| 110 | 33 | Barrett Ruud | 53.51 | 46.40 | 58.63 | 584 | Titans |
| 111 | 34 | Will Herring | 52.93 | 46.10 | 57.87 | 111 | Saints |
| 112 | 35 | Gerald McRath | 52.46 | 46.30 | 59.17 | 138 | Titans |
| 113 | 36 | Thomas Davis Sr. | 51.95 | 58.20 | 62.11 | 208 | Panthers |
| 114 | 37 | Dannell Ellerbe | 51.39 | 39.90 | 59.32 | 253 | Ravens |
| 115 | 38 | Quincy Black | 50.92 | 42.20 | 55.82 | 660 | Buccaneers |
| 116 | 39 | Casey Matthews | 48.45 | 34.10 | 58.01 | 326 | Eagles |

## QB — Quarterback

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Rodgers | 91.32 | 92.30 | 85.97 | 681 | Packers |
| 2 | 2 | Drew Brees | 88.15 | 91.12 | 81.18 | 840 | Saints |
| 3 | 3 | Tom Brady | 86.30 | 89.90 | 79.12 | 807 | Patriots |
| 4 | 4 | Eli Manning | 83.98 | 88.45 | 76.15 | 841 | Giants |
| 5 | 5 | Philip Rivers | 81.17 | 83.17 | 75.14 | 656 | Chargers |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Ben Roethlisberger | 78.49 | 78.91 | 74.13 | 644 | Steelers |
| 7 | 2 | Matt Ryan | 78.08 | 81.74 | 70.54 | 696 | Falcons |
| 8 | 3 | Matt Schaub | 77.12 | 81.60 | 75.26 | 325 | Texans |
| 9 | 4 | Tony Romo | 76.34 | 78.07 | 78.75 | 585 | Cowboys |
| 10 | 5 | Michael Vick | 75.55 | 78.09 | 72.12 | 522 | Eagles |
| 11 | 6 | Matthew Stafford | 75.46 | 76.14 | 76.01 | 793 | Lions |
| 12 | 7 | Alex Smith | 74.78 | 76.61 | 70.99 | 626 | 49ers |

### Starter (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Joe Flacco | 71.70 | 70.92 | 67.79 | 692 | Ravens |
| 14 | 2 | Carson Palmer | 69.65 | 70.83 | 69.70 | 374 | Raiders |
| 15 | 3 | Matt Hasselbeck | 69.56 | 70.34 | 66.11 | 568 | Titans |
| 16 | 4 | Jay Cutler | 69.33 | 69.22 | 70.89 | 364 | Bears |
| 17 | 5 | Ryan Fitzpatrick | 68.80 | 69.12 | 64.95 | 655 | Bills |
| 18 | 6 | Cam Newton | 67.60 | 65.10 | 72.24 | 634 | Panthers |
| 19 | 7 | Matt Moore | 65.95 | 72.47 | 71.14 | 411 | Dolphins |
| 20 | 8 | Josh Freeman | 65.84 | 65.27 | 62.28 | 655 | Buccaneers |
| 21 | 9 | Mark Sanchez | 64.36 | 60.86 | 62.85 | 642 | Jets |
| 22 | 10 | Andy Dalton | 63.79 | 62.50 | 63.58 | 651 | Bengals |
| 23 | 11 | Jason Campbell | 63.59 | 72.05 | 67.10 | 190 | Raiders |
| 24 | 12 | Chad Henne | 63.56 | 71.02 | 65.86 | 139 | Dolphins |
| 25 | 13 | Tarvaris Jackson | 63.43 | 70.61 | 63.04 | 539 | Seahawks |
| 26 | 14 | Kyle Orton | 63.35 | 66.98 | 64.42 | 287 | Chiefs |
| 27 | 15 | Donovan McNabb | 62.68 | 68.08 | 65.00 | 187 | Vikings |

### Rotation/backup (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Matt Cassel | 61.93 | 65.14 | 62.25 | 326 | Chiefs |
| 29 | 2 | Rex Grossman | 61.60 | 63.04 | 63.61 | 515 | Commanders |
| 30 | 3 | Sam Bradford | 60.54 | 64.04 | 57.66 | 423 | Rams |
| 31 | 4 | Kevin Kolb | 60.50 | 58.81 | 68.22 | 311 | Cardinals |
| 32 | 5 | Colt McCoy | 60.40 | 62.18 | 58.96 | 566 | Browns |
| 33 | 6 | Vince Young | 58.97 | 57.70 | 66.83 | 140 | Eagles |
| 34 | 7 | Dan Orlovsky | 58.60 | 57.89 | 64.88 | 215 | Colts |
| 35 | 8 | T.J. Yates | 58.43 | 58.50 | 62.16 | 220 | Texans |
| 36 | 9 | Seneca Wallace | 57.15 | 60.06 | 57.53 | 122 | Browns |
| 37 | 10 | Kerry Collins | 57.03 | 59.82 | 57.15 | 105 | Colts |
| 38 | 11 | Christian Ponder | 57.03 | 50.00 | 59.60 | 356 | Vikings |
| 39 | 12 | John Beck | 55.34 | 50.87 | 57.74 | 164 | Commanders |
| 40 | 13 | A.J. Feeley | 54.86 | 49.92 | 56.55 | 113 | Rams |
| 41 | 14 | Curtis Painter | 54.77 | 44.04 | 59.46 | 277 | Colts |
| 42 | 15 | John Skelton | 53.92 | 40.63 | 59.75 | 326 | Cardinals |
| 43 | 16 | Blaine Gabbert | 53.39 | 37.40 | 55.17 | 499 | Jaguars |

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

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Ryan Clark | 76.95 | 71.10 | 76.68 | 989 | Steelers |
| 8 | 2 | Mike Adams | 75.85 | 71.40 | 75.04 | 827 | Browns |
| 9 | 3 | Tyvon Branch | 75.49 | 68.30 | 76.11 | 1133 | Raiders |
| 10 | 4 | Jim Leonhard | 75.19 | 78.50 | 72.71 | 781 | Jets |
| 11 | 5 | Bernard Pollard | 75.07 | 70.50 | 74.34 | 1014 | Ravens |
| 12 | 6 | Isa Abdul-Quddus | 74.71 | 71.40 | 74.84 | 186 | Saints |
| 13 | 7 | William Moore | 74.49 | 68.80 | 76.07 | 601 | Falcons |

### Starter (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Michael Griffin | 72.97 | 67.90 | 72.18 | 1121 | Titans |
| 15 | 2 | Earl Thomas III | 71.73 | 64.20 | 72.58 | 1097 | Seahawks |
| 16 | 3 | Danieal Manning | 71.43 | 62.60 | 73.80 | 846 | Texans |
| 17 | 4 | Atari Bigby | 71.10 | 68.60 | 73.93 | 155 | Seahawks |
| 18 | 5 | Michael Huff | 70.78 | 67.70 | 71.27 | 624 | Raiders |
| 19 | 6 | Rashad Johnson | 70.64 | 63.50 | 73.83 | 480 | Cardinals |
| 20 | 7 | Nate Allen | 70.25 | 63.10 | 72.67 | 752 | Eagles |
| 21 | 8 | Barry Church | 69.86 | 64.90 | 72.79 | 166 | Cowboys |
| 22 | 9 | Quintin Mikell | 69.81 | 62.20 | 70.72 | 1068 | Rams |
| 23 | 10 | Donte Whitner | 69.75 | 61.00 | 71.42 | 1048 | 49ers |
| 24 | 11 | Kendrick Lewis | 69.13 | 66.00 | 68.21 | 963 | Chiefs |
| 25 | 12 | T.J. Ward | 68.04 | 60.70 | 73.97 | 468 | Browns |
| 26 | 13 | Troy Nolan | 67.80 | 60.80 | 71.17 | 424 | Texans |
| 27 | 14 | Reggie Nelson | 67.78 | 62.30 | 69.21 | 1080 | Bengals |
| 28 | 15 | Brodney Pool | 67.34 | 63.10 | 67.30 | 545 | Jets |
| 29 | 16 | Morgan Burnett | 67.18 | 60.40 | 72.21 | 1144 | Packers |
| 30 | 17 | Gibril Wilson | 66.57 | 65.30 | 64.28 | 211 | Bengals |
| 31 | 18 | Dawan Landry | 66.27 | 58.20 | 67.48 | 1012 | Jaguars |
| 32 | 19 | Antoine Bethea | 66.15 | 52.50 | 71.09 | 1085 | Colts |
| 33 | 20 | Craig Steltz | 65.93 | 61.80 | 69.72 | 400 | Bears |
| 34 | 21 | Gerald Sensabaugh | 65.60 | 56.20 | 67.70 | 971 | Cowboys |
| 35 | 22 | LaRon Landry | 65.58 | 63.00 | 71.08 | 496 | Commanders |
| 36 | 23 | Brian Dawkins | 65.24 | 58.00 | 69.15 | 773 | Broncos |
| 37 | 24 | Usama Young | 64.71 | 55.30 | 69.17 | 652 | Browns |
| 38 | 25 | Dwight Lowery | 64.64 | 58.30 | 66.65 | 634 | Jaguars |
| 39 | 26 | Glover Quin | 64.12 | 58.80 | 63.50 | 1119 | Texans |
| 40 | 27 | Travis Daniels | 64.06 | 62.30 | 65.24 | 292 | Chiefs |
| 41 | 28 | Louis Delmas | 63.29 | 63.70 | 61.85 | 737 | Lions |
| 42 | 29 | Kerry Rhodes | 62.64 | 53.80 | 70.24 | 397 | Cardinals |
| 43 | 30 | Thomas DeCoud | 62.20 | 51.40 | 65.23 | 1013 | Falcons |

### Rotation/backup (58 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 44 | 1 | Husain Abdullah | 61.76 | 53.60 | 67.99 | 564 | Vikings |
| 45 | 2 | Matt Giordano | 61.23 | 58.10 | 65.27 | 850 | Raiders |
| 46 | 3 | Ryan Mundy | 61.00 | 51.70 | 67.07 | 273 | Steelers |
| 47 | 4 | Corey Lynch | 60.92 | 59.70 | 65.12 | 273 | Buccaneers |
| 48 | 5 | Mistral Raymond | 60.89 | 59.70 | 65.85 | 378 | Vikings |
| 49 | 6 | Chris Crocker | 60.73 | 55.80 | 62.59 | 1047 | Bengals |
| 50 | 7 | Dashon Goldson | 60.28 | 49.40 | 63.36 | 1016 | 49ers |
| 51 | 8 | David Bruton | 60.19 | 54.20 | 64.32 | 272 | Broncos |
| 52 | 9 | Chris Prosinski | 60.07 | 59.30 | 60.58 | 195 | Jaguars |
| 53 | 10 | Kurt Coleman | 60.03 | 50.70 | 64.17 | 748 | Eagles |
| 54 | 11 | Darian Stewart | 59.93 | 57.40 | 60.05 | 891 | Rams |
| 55 | 12 | Da'Norris Searcy | 59.75 | 54.80 | 66.19 | 227 | Bills |
| 56 | 13 | Josh Barrett | 59.65 | 68.30 | 65.68 | 218 | Patriots |
| 57 | 14 | Tanard Jackson | 59.49 | 62.00 | 59.90 | 523 | Buccaneers |
| 58 | 15 | Eric Hagg | 59.36 | 60.50 | 61.73 | 180 | Browns |
| 59 | 16 | Reggie Smith | 59.23 | 51.90 | 61.11 | 260 | 49ers |
| 60 | 17 | George Wilson | 59.20 | 45.70 | 66.39 | 821 | Bills |
| 61 | 18 | David Caldwell | 59.03 | 47.80 | 62.35 | 601 | Colts |
| 62 | 19 | Chris Hope | 58.86 | 50.00 | 64.50 | 286 | Titans |
| 63 | 20 | Yeremiah Bell | 58.82 | 46.30 | 63.00 | 1092 | Dolphins |
| 64 | 21 | Patrick Chung | 58.68 | 49.30 | 64.42 | 758 | Patriots |
| 65 | 22 | Antrel Rolle | 58.66 | 51.40 | 59.33 | 1343 | Giants |
| 66 | 23 | Tom Zbikowski | 58.43 | 59.40 | 62.73 | 223 | Ravens |
| 67 | 24 | Abram Elam | 58.08 | 43.00 | 63.96 | 1015 | Cowboys |
| 68 | 25 | Malcolm Jenkins | 57.92 | 48.00 | 60.75 | 1092 | Saints |
| 69 | 26 | Quinton Carter | 57.88 | 50.40 | 59.73 | 824 | Broncos |
| 70 | 27 | Sherrod Martin | 57.54 | 50.60 | 58.38 | 990 | Panthers |
| 71 | 28 | Sergio Brown | 57.25 | 54.40 | 61.36 | 342 | Patriots |
| 72 | 29 | Eric Smith | 56.96 | 45.80 | 60.24 | 943 | Jets |
| 73 | 30 | Chris Conte | 56.83 | 54.30 | 60.60 | 593 | Bears |
| 74 | 31 | Brandon Meriweather | 56.77 | 51.00 | 59.70 | 406 | Bears |
| 75 | 32 | Tyrone Culver | 56.60 | 53.00 | 59.92 | 437 | Dolphins |
| 76 | 33 | Charlie Peprah | 56.07 | 43.30 | 60.42 | 993 | Packers |
| 77 | 34 | Rahim Moore | 55.69 | 51.60 | 57.38 | 542 | Broncos |
| 78 | 35 | Madieu Williams | 55.59 | 54.60 | 60.04 | 139 | 49ers |
| 79 | 36 | Major Wright | 55.21 | 47.90 | 60.09 | 581 | Bears |
| 80 | 37 | Mike Mitchell | 54.70 | 48.90 | 56.75 | 493 | Raiders |
| 81 | 38 | Craig Dahl | 53.95 | 39.90 | 59.53 | 469 | Rams |
| 82 | 39 | Amari Spievey | 53.85 | 45.30 | 56.95 | 988 | Lions |
| 83 | 40 | Quintin Demps | 53.77 | 46.70 | 60.56 | 275 | Texans |
| 84 | 41 | James Ihedigbo | 53.33 | 43.00 | 56.84 | 897 | Patriots |
| 85 | 42 | Melvin Bullitt | 53.15 | 62.00 | 64.59 | 129 | Colts |
| 86 | 43 | Steve Gregory | 52.94 | 40.10 | 60.72 | 759 | Chargers |
| 87 | 44 | Sean Jones | 52.56 | 40.70 | 56.30 | 984 | Buccaneers |
| 88 | 45 | James Sanders | 51.85 | 38.50 | 59.83 | 509 | Falcons |
| 89 | 46 | Courtney Greene | 51.73 | 48.70 | 58.57 | 114 | Jaguars |
| 90 | 47 | Nick Collins | 51.53 | 49.60 | 57.77 | 132 | Packers |
| 91 | 48 | Charles Godfrey | 51.40 | 40.60 | 55.73 | 856 | Panthers |
| 92 | 49 | Roman Harper | 51.37 | 39.40 | 55.18 | 1112 | Saints |
| 93 | 50 | Reshad Jones | 50.43 | 38.90 | 59.15 | 653 | Dolphins |
| 94 | 51 | Cody Grimm | 50.20 | 50.90 | 56.76 | 175 | Buccaneers |
| 95 | 52 | Jordan Pugh | 48.87 | 35.30 | 59.35 | 243 | Panthers |
| 96 | 53 | Reed Doughty | 47.59 | 30.50 | 56.25 | 663 | Commanders |
| 97 | 54 | DeJon Gomes | 46.86 | 39.10 | 58.74 | 204 | Commanders |
| 98 | 55 | Tyrell Johnson | 45.81 | 33.80 | 56.81 | 324 | Vikings |
| 99 | 56 | Jaiquawn Jarrett | 45.33 | 46.00 | 54.14 | 247 | Eagles |
| 100 | 57 | Jamarca Sanford | 45.00 | 28.00 | 55.23 | 845 | Vikings |
| 101 | 58 | John Wendling | 45.00 | 33.60 | 49.91 | 159 | Lions |

## T — Tackle

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jason Peters | 95.46 | 90.00 | 94.94 | 925 | Eagles |
| 2 | 2 | Jordan Gross | 90.65 | 85.00 | 90.25 | 967 | Panthers |
| 3 | 3 | Bryan Bulaga | 89.17 | 81.20 | 90.32 | 754 | Packers |
| 4 | 4 | Zach Strief | 89.04 | 82.30 | 89.37 | 897 | Saints |
| 5 | 5 | Duane Brown | 88.88 | 83.10 | 88.57 | 1176 | Texans |
| 6 | 6 | Tyron Smith | 87.89 | 80.20 | 88.85 | 1040 | Cowboys |
| 7 | 7 | Andrew Whitworth | 87.30 | 81.50 | 87.00 | 1132 | Bengals |
| 8 | 8 | Jared Veldheer | 87.06 | 80.50 | 87.26 | 1073 | Raiders |
| 9 | 9 | Eugene Monroe | 86.79 | 80.40 | 86.89 | 891 | Jaguars |
| 10 | 10 | David Stewart | 86.48 | 80.30 | 86.43 | 909 | Titans |
| 11 | 11 | Trent Williams | 86.44 | 79.00 | 87.23 | 627 | Commanders |
| 12 | 12 | Phil Loadholt | 86.30 | 76.70 | 88.54 | 1003 | Vikings |
| 13 | 13 | Michael Roos | 86.16 | 79.50 | 86.43 | 1022 | Titans |
| 14 | 14 | Eric Winston | 86.15 | 78.90 | 86.81 | 1185 | Texans |
| 15 | 15 | Joe Staley | 85.91 | 77.90 | 87.09 | 1076 | 49ers |
| 16 | 16 | Demetress Bell | 85.67 | 78.80 | 86.08 | 394 | Bills |
| 17 | 17 | Tyson Clabo | 85.46 | 77.30 | 86.74 | 1191 | Falcons |
| 18 | 18 | Matt Light | 85.45 | 79.40 | 85.32 | 1232 | Patriots |
| 19 | 19 | Donald Penn | 85.28 | 77.70 | 86.16 | 1022 | Buccaneers |
| 20 | 20 | Joe Thomas | 84.16 | 77.30 | 84.57 | 1063 | Browns |
| 21 | 21 | Branden Albert | 84.13 | 75.60 | 85.65 | 1038 | Chiefs |
| 22 | 22 | Cameron Bradfield | 83.92 | 75.30 | 85.50 | 181 | Jaguars |
| 23 | 23 | Jermon Bushrod | 83.87 | 77.00 | 84.29 | 1320 | Saints |
| 24 | 24 | Jeff Backus | 83.38 | 75.60 | 84.40 | 1130 | Lions |
| 25 | 25 | D'Brickashaw Ferguson | 83.01 | 77.00 | 82.85 | 1088 | Jets |
| 26 | 26 | Tony Pashos | 82.57 | 73.70 | 84.31 | 781 | Browns |
| 27 | 27 | Jake Long | 82.52 | 74.90 | 83.43 | 784 | Dolphins |
| 28 | 28 | Will Beatty | 82.41 | 74.70 | 83.38 | 666 | Giants |
| 29 | 29 | King Dunlap | 81.95 | 72.20 | 84.28 | 144 | Eagles |
| 30 | 30 | Sebastian Vollmer | 81.38 | 71.20 | 84.00 | 377 | Patriots |
| 31 | 31 | Bryant McKinnie | 81.09 | 73.30 | 82.12 | 1213 | Ravens |
| 32 | 32 | Marcus Gilbert | 81.02 | 71.40 | 83.26 | 883 | Steelers |
| 33 | 33 | Andre Smith | 80.58 | 69.70 | 83.66 | 967 | Bengals |
| 34 | 34 | Russell Okung | 80.33 | 69.70 | 83.25 | 767 | Seahawks |
| 35 | 35 | Nate Solder | 80.11 | 68.10 | 83.95 | 1023 | Patriots |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 36 | 1 | Michael Oher | 79.87 | 70.20 | 82.15 | 1203 | Ravens |
| 37 | 2 | Anthony Davis | 79.69 | 68.50 | 82.98 | 1123 | 49ers |
| 38 | 3 | Doug Free | 79.43 | 70.00 | 81.55 | 1046 | Cowboys |
| 39 | 4 | Jared Gaither | 78.71 | 76.40 | 76.09 | 337 | Chargers |
| 40 | 5 | Levi Brown | 78.45 | 66.60 | 82.18 | 1016 | Cardinals |
| 41 | 6 | Gosder Cherilus | 78.30 | 68.50 | 80.66 | 1031 | Lions |
| 42 | 7 | Anthony Collins | 78.23 | 75.80 | 75.68 | 169 | Bengals |
| 43 | 8 | Anthony Castonzo | 78.08 | 68.90 | 80.04 | 693 | Colts |
| 44 | 9 | Khalif Barnes | 77.81 | 66.20 | 81.38 | 1000 | Raiders |
| 45 | 10 | Max Starks | 77.63 | 68.10 | 79.81 | 823 | Steelers |
| 46 | 11 | Breno Giacomini | 77.57 | 64.20 | 82.31 | 548 | Seahawks |
| 47 | 12 | Brandon Keith | 77.35 | 67.10 | 80.01 | 534 | Cardinals |
| 48 | 13 | Will Svitek | 77.23 | 66.70 | 80.08 | 796 | Falcons |
| 49 | 14 | Jammal Brown | 76.57 | 65.50 | 79.78 | 768 | Commanders |
| 50 | 15 | Marcus McNeill | 76.45 | 64.40 | 80.32 | 535 | Chargers |
| 51 | 16 | Ryan Clady | 76.09 | 66.70 | 78.18 | 1193 | Broncos |
| 52 | 17 | Jeff Otah | 76.02 | 73.90 | 73.26 | 229 | Panthers |
| 53 | 18 | Stephon Heyer | 75.99 | 63.90 | 79.88 | 244 | Raiders |
| 54 | 19 | Erik Pears | 75.67 | 64.50 | 78.95 | 1041 | Bills |
| 55 | 20 | Sean Locklear | 75.13 | 65.00 | 77.71 | 284 | Commanders |
| 56 | 21 | Chris Hairston | 74.66 | 63.70 | 77.80 | 466 | Bills |
| 57 | 22 | Jeff Linkenbach | 74.44 | 63.20 | 77.77 | 981 | Colts |
| 58 | 23 | Jason Smith | 74.31 | 60.30 | 79.49 | 321 | Rams |
| 59 | 24 | Wayne Hunter | 74.13 | 59.80 | 79.51 | 1082 | Jets |

### Starter (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 60 | 1 | Demar Dotson | 73.29 | 65.40 | 74.38 | 132 | Buccaneers |
| 61 | 2 | Sam Baker | 72.92 | 61.40 | 76.43 | 436 | Falcons |
| 62 | 3 | Chad Clifton | 72.79 | 60.60 | 76.75 | 329 | Packers |
| 63 | 4 | Marcus Cannon | 72.46 | 65.50 | 72.93 | 164 | Patriots |
| 64 | 5 | J'Marcus Webb | 72.41 | 58.10 | 77.79 | 1007 | Bears |
| 65 | 6 | Jeremy Trueblood | 71.96 | 58.40 | 76.83 | 941 | Buccaneers |
| 66 | 7 | Charles Brown | 70.69 | 58.90 | 74.39 | 400 | Saints |
| 67 | 8 | Frank Omiyale | 69.70 | 50.00 | 78.67 | 220 | Bears |
| 68 | 9 | Byron Bell | 68.99 | 55.10 | 74.08 | 815 | Panthers |
| 69 | 10 | Barry Richardson | 68.77 | 54.40 | 74.19 | 1053 | Chiefs |
| 70 | 11 | Corey Hilliard | 67.89 | 52.00 | 74.32 | 157 | Lions |
| 71 | 12 | Chris Clark | 67.39 | 57.10 | 70.08 | 147 | Broncos |
| 72 | 13 | Derek Sherrod | 66.62 | 54.60 | 70.47 | 110 | Packers |
| 73 | 14 | Marshall Newhouse | 66.61 | 51.70 | 72.39 | 927 | Packers |
| 74 | 15 | Jonathan Scott | 66.06 | 52.00 | 71.26 | 406 | Steelers |
| 75 | 16 | Joe Barksdale | 64.88 | 55.10 | 67.23 | 152 | Raiders |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 86.37 | 92.90 | 77.85 | 745 | Patriots |
| 2 | 2 | Joel Dreessen | 84.63 | 87.30 | 78.69 | 353 | Texans |
| 3 | 3 | Jimmy Graham | 83.71 | 90.10 | 75.29 | 699 | Saints |
| 4 | 4 | Heath Miller | 82.16 | 88.20 | 73.96 | 568 | Steelers |
| 5 | 5 | Tony Gonzalez | 81.78 | 87.60 | 73.73 | 647 | Falcons |
| 6 | 6 | Fred Davis | 80.62 | 77.70 | 78.40 | 478 | Commanders |
| 7 | 7 | Anthony Fasano | 80.10 | 80.30 | 75.80 | 497 | Dolphins |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Vernon Davis | 79.86 | 81.10 | 74.86 | 611 | 49ers |
| 9 | 2 | Jason Witten | 79.56 | 78.70 | 75.96 | 609 | Cowboys |
| 10 | 3 | Jeremy Shockey | 79.34 | 85.00 | 71.40 | 266 | Panthers |
| 11 | 4 | Marcedes Lewis | 78.70 | 76.90 | 75.73 | 421 | Jaguars |
| 12 | 5 | Brent Celek | 78.65 | 79.50 | 73.91 | 574 | Eagles |
| 13 | 6 | Aaron Hernandez | 78.36 | 81.70 | 71.97 | 576 | Patriots |
| 14 | 7 | Antonio Gates | 78.31 | 79.60 | 73.29 | 512 | Chargers |
| 15 | 8 | Jake Ballard | 78.29 | 77.50 | 74.65 | 584 | Giants |
| 16 | 9 | Todd Heap | 78.14 | 71.00 | 78.74 | 202 | Cardinals |
| 17 | 10 | Randy McMichael | 76.94 | 80.30 | 70.53 | 287 | Chargers |
| 18 | 11 | Kellen Davis | 76.90 | 78.80 | 71.46 | 316 | Bears |
| 19 | 12 | Kyle Rudolph | 76.80 | 78.30 | 71.64 | 245 | Vikings |
| 20 | 13 | Dennis Pitta | 76.62 | 76.80 | 72.33 | 336 | Ravens |
| 21 | 14 | Tony Scheffler | 76.56 | 76.30 | 72.57 | 230 | Lions |
| 22 | 15 | Jared Cook | 76.36 | 64.50 | 80.10 | 495 | Titans |
| 23 | 16 | Owen Daniels | 76.22 | 80.20 | 69.40 | 480 | Texans |
| 24 | 17 | Jeff King | 76.09 | 89.50 | 62.99 | 229 | Cardinals |
| 25 | 18 | Jermichael Finley | 75.15 | 67.00 | 76.41 | 599 | Packers |
| 26 | 19 | Martellus Bennett | 74.71 | 77.80 | 68.48 | 202 | Cowboys |
| 27 | 20 | Kellen Winslow | 74.48 | 67.40 | 75.04 | 606 | Buccaneers |
| 28 | 21 | Visanthe Shiancoe | 74.25 | 69.40 | 73.31 | 422 | Vikings |
| 29 | 22 | Scott Chandler | 74.10 | 74.80 | 69.47 | 282 | Bills |
| 30 | 23 | Craig Stevens | 74.00 | 68.20 | 73.70 | 142 | Titans |

### Starter (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Dustin Keller | 73.97 | 68.70 | 73.32 | 505 | Jets |
| 32 | 2 | Evan Moore | 73.30 | 73.00 | 69.33 | 241 | Browns |
| 33 | 3 | Brandon Pettigrew | 73.28 | 69.80 | 71.44 | 719 | Lions |
| 34 | 4 | Greg Olsen | 72.97 | 70.10 | 70.72 | 470 | Panthers |
| 35 | 5 | Jermaine Gresham | 72.97 | 73.90 | 68.19 | 554 | Bengals |
| 36 | 6 | Benjamin Watson | 72.81 | 63.90 | 74.58 | 418 | Browns |
| 37 | 7 | Michael Hoomanawanui | 72.10 | 66.50 | 71.67 | 209 | Rams |
| 38 | 8 | Donald Lee | 72.10 | 70.70 | 68.87 | 122 | Bengals |
| 39 | 9 | Matt Spaeth | 72.05 | 78.00 | 63.92 | 150 | Bears |
| 40 | 10 | Kevin Boss | 71.95 | 70.10 | 69.02 | 296 | Raiders |
| 41 | 11 | Clay Harbor | 71.09 | 69.80 | 67.79 | 152 | Eagles |
| 42 | 12 | Delanie Walker | 70.63 | 65.50 | 69.88 | 257 | 49ers |
| 43 | 13 | Bear Pascoe | 70.53 | 65.70 | 69.58 | 177 | Giants |
| 44 | 14 | Zach Miller | 69.25 | 59.90 | 71.31 | 508 | Seahawks |
| 45 | 15 | Dante Rosario | 69.00 | 63.80 | 68.30 | 110 | Broncos |
| 46 | 16 | Jacob Tamme | 68.95 | 64.90 | 67.48 | 192 | Colts |
| 47 | 17 | Ed Dickson | 68.83 | 66.20 | 66.41 | 549 | Ravens |
| 48 | 18 | Anthony Becht | 68.65 | 65.00 | 66.92 | 136 | Chiefs |
| 49 | 19 | Logan Paulsen | 68.51 | 62.60 | 68.29 | 170 | Commanders |
| 50 | 20 | Luke Stocker | 68.47 | 68.50 | 64.28 | 139 | Buccaneers |
| 51 | 21 | Brandon Myers | 68.24 | 68.60 | 63.83 | 177 | Raiders |
| 52 | 22 | Alex Smith | 68.18 | 68.50 | 63.80 | 141 | Browns |
| 53 | 23 | Michael Palmer | 68.07 | 66.40 | 65.02 | 112 | Falcons |
| 54 | 24 | Dallas Clark | 66.75 | 62.30 | 65.55 | 334 | Colts |
| 55 | 25 | Matthew Mulligan | 66.56 | 65.90 | 62.84 | 173 | Jets |
| 56 | 26 | Lance Kendricks | 66.48 | 58.50 | 67.64 | 378 | Rams |
| 57 | 27 | Zach Potter | 66.05 | 61.90 | 64.65 | 116 | Jaguars |
| 58 | 28 | Travis Beckum | 65.42 | 63.20 | 62.74 | 154 | Giants |
| 59 | 29 | Rob Housler | 64.22 | 52.50 | 67.87 | 136 | Cardinals |
| 60 | 30 | Jake O'Connell | 63.85 | 58.50 | 63.25 | 144 | Chiefs |
| 61 | 31 | Billy Bajema | 62.15 | 53.90 | 63.49 | 229 | Rams |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 62 | 1 | Tom Crabtree | 60.34 | 51.00 | 62.40 | 116 | Packers |
| 63 | 2 | Brody Eldridge | 59.94 | 56.40 | 58.13 | 100 | Colts |
| 64 | 3 | Anthony McCoy | 58.82 | 48.50 | 61.53 | 215 | Seahawks |

## WR — Wide Receiver

- **Season used:** `2011`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Malcom Floyd | 89.35 | 89.20 | 85.29 | 358 | Chargers |
| 2 | 2 | Calvin Johnson | 88.84 | 91.20 | 83.10 | 766 | Lions |
| 3 | 3 | Antonio Brown | 87.86 | 88.80 | 83.07 | 499 | Steelers |
| 4 | 4 | Larry Fitzgerald | 87.58 | 89.80 | 81.94 | 659 | Cardinals |
| 5 | 5 | Jordy Nelson | 87.06 | 86.70 | 83.13 | 505 | Packers |
| 6 | 6 | Hakeem Nicks | 86.49 | 89.20 | 80.51 | 775 | Giants |
| 7 | 7 | Andre Johnson | 85.99 | 87.20 | 81.01 | 231 | Texans |
| 8 | 8 | Victor Cruz | 85.49 | 82.10 | 83.58 | 701 | Giants |
| 9 | 9 | Marques Colston | 85.02 | 89.50 | 77.87 | 599 | Saints |
| 10 | 10 | Percy Harvin | 84.06 | 86.50 | 78.26 | 363 | Vikings |
| 11 | 11 | Demaryius Thomas | 83.79 | 75.90 | 84.89 | 331 | Broncos |
| 12 | 12 | Dwayne Bowe | 83.41 | 84.30 | 78.65 | 527 | Chiefs |
| 13 | 13 | Wes Welker | 83.25 | 89.20 | 75.11 | 748 | Patriots |
| 14 | 14 | Brandon Lloyd | 83.05 | 82.40 | 79.31 | 512 | Rams |
| 15 | 15 | Mike Wallace | 83.02 | 78.40 | 81.93 | 630 | Steelers |
| 16 | 16 | Doug Baldwin | 82.34 | 80.30 | 79.53 | 365 | Seahawks |
| 17 | 17 | Denarius Moore | 82.10 | 76.70 | 81.53 | 357 | Raiders |
| 18 | 18 | Dez Bryant | 81.90 | 80.40 | 78.74 | 538 | Cowboys |
| 19 | 19 | Johnny Knox | 81.39 | 75.90 | 80.88 | 334 | Bears |
| 20 | 20 | Greg Jennings | 81.32 | 77.90 | 79.43 | 522 | Packers |
| 21 | 21 | Vincent Jackson | 80.07 | 75.90 | 78.68 | 607 | Chargers |

### Good (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Brandon Marshall | 79.76 | 77.10 | 77.36 | 534 | Dolphins |
| 23 | 2 | Roddy White | 79.43 | 80.90 | 74.28 | 696 | Falcons |
| 24 | 3 | A.J. Green | 79.20 | 72.00 | 79.84 | 581 | Bengals |
| 25 | 4 | Darrius Heyward-Bey | 79.16 | 77.90 | 75.84 | 462 | Raiders |
| 26 | 5 | Nate Washington | 79.12 | 76.70 | 76.56 | 581 | Titans |
| 27 | 6 | Julio Jones | 79.07 | 71.00 | 80.29 | 525 | Falcons |
| 28 | 7 | Jacoby Ford | 78.91 | 72.80 | 78.81 | 119 | Raiders |
| 29 | 8 | Lance Moore | 78.60 | 79.00 | 74.17 | 335 | Saints |
| 30 | 9 | Jabar Gaffney | 78.27 | 76.00 | 75.61 | 599 | Commanders |
| 31 | 10 | Steve Johnson | 78.03 | 76.00 | 75.21 | 585 | Bills |
| 32 | 11 | Reggie Wayne | 77.95 | 76.30 | 74.88 | 594 | Colts |
| 33 | 12 | DeSean Jackson | 77.47 | 70.30 | 78.08 | 565 | Eagles |
| 34 | 13 | Josh Morgan | 77.44 | 72.30 | 76.70 | 144 | 49ers |
| 35 | 14 | Jeremy Maclin | 77.40 | 75.40 | 74.57 | 482 | Eagles |
| 36 | 15 | Anquan Boldin | 77.25 | 73.60 | 75.52 | 623 | Ravens |
| 37 | 16 | Vincent Brown | 76.73 | 69.50 | 77.38 | 235 | Chargers |
| 38 | 17 | Miles Austin | 76.35 | 71.80 | 75.22 | 363 | Cowboys |
| 39 | 18 | Earl Bennett | 76.32 | 67.40 | 78.10 | 271 | Bears |
| 40 | 19 | Danario Alexander | 76.14 | 70.40 | 75.80 | 283 | Rams |
| 41 | 20 | Golden Tate | 76.14 | 74.60 | 73.00 | 299 | Seahawks |
| 42 | 21 | Emmanuel Sanders | 76.04 | 73.00 | 73.90 | 272 | Steelers |
| 43 | 22 | Donte' Stallworth | 75.99 | 68.20 | 77.01 | 231 | Commanders |
| 44 | 23 | Torrey Smith | 75.88 | 68.70 | 76.50 | 623 | Ravens |
| 45 | 24 | Randall Cobb | 75.81 | 66.40 | 77.91 | 204 | Packers |
| 46 | 25 | Michael Crabtree | 75.78 | 75.70 | 71.66 | 511 | 49ers |
| 47 | 26 | Deion Branch | 75.68 | 68.20 | 76.50 | 655 | Patriots |
| 48 | 27 | Sidney Rice | 75.57 | 68.10 | 76.39 | 289 | Seahawks |
| 49 | 28 | Robert Meachem | 75.50 | 71.30 | 74.13 | 533 | Saints |
| 50 | 29 | Steve Breaston | 75.45 | 70.40 | 74.65 | 506 | Chiefs |
| 51 | 30 | Joshua Cribbs | 75.33 | 73.20 | 72.58 | 373 | Browns |
| 52 | 31 | Mario Manningham | 75.16 | 68.60 | 75.37 | 527 | Giants |
| 53 | 32 | Brandon LaFell | 74.65 | 68.10 | 74.85 | 466 | Panthers |
| 54 | 33 | Plaxico Burress | 74.57 | 71.30 | 72.59 | 547 | Jets |
| 55 | 34 | James Jones | 74.36 | 65.50 | 76.10 | 375 | Packers |
| 56 | 35 | Santana Moss | 74.20 | 70.20 | 72.70 | 413 | Commanders |
| 57 | 36 | Chastin West | 74.03 | 67.20 | 74.41 | 134 | Jaguars |

### Starter (72 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 58 | 1 | Braylon Edwards | 73.71 | 59.70 | 78.89 | 156 | 49ers |
| 59 | 2 | Roy Williams | 73.44 | 70.90 | 70.96 | 339 | Bears |
| 60 | 3 | Pierre Garcon | 73.06 | 66.60 | 73.20 | 587 | Colts |
| 61 | 4 | Steve Smith | 73.05 | 56.80 | 79.72 | 718 | Panthers |
| 62 | 5 | Preston Parker | 72.95 | 63.90 | 74.82 | 407 | Buccaneers |
| 63 | 6 | Jordan Norwood | 72.60 | 66.70 | 72.37 | 184 | Browns |
| 64 | 7 | Devin Aromashodu | 72.58 | 63.30 | 74.60 | 413 | Vikings |
| 65 | 8 | Jerricho Cotchery | 72.57 | 72.60 | 68.38 | 226 | Steelers |
| 66 | 9 | Brian Hartline | 72.51 | 67.10 | 71.95 | 452 | Dolphins |
| 67 | 10 | Jason Avant | 72.41 | 67.50 | 71.52 | 487 | Eagles |
| 68 | 11 | Jeremy Kerley | 72.39 | 69.90 | 69.88 | 240 | Jets |
| 69 | 12 | Kevin Walter | 72.37 | 69.10 | 70.39 | 421 | Texans |
| 70 | 13 | Damian Williams | 72.17 | 65.30 | 72.59 | 481 | Titans |
| 71 | 14 | Kyle Williams | 72.02 | 70.20 | 69.07 | 232 | 49ers |
| 72 | 15 | Michael Jenkins | 71.90 | 66.30 | 71.46 | 330 | Vikings |
| 73 | 16 | Jerome Simpson | 71.76 | 64.20 | 72.63 | 605 | Bengals |
| 74 | 17 | Arrelious Benn | 71.54 | 61.40 | 74.13 | 301 | Buccaneers |
| 75 | 18 | Austin Collie | 71.53 | 67.60 | 69.98 | 457 | Colts |
| 76 | 19 | Chad Johnson | 71.52 | 64.80 | 71.83 | 265 | Patriots |
| 77 | 20 | Jacoby Jones | 71.46 | 63.70 | 72.47 | 432 | Texans |
| 78 | 21 | Devery Henderson | 70.68 | 61.10 | 72.90 | 565 | Saints |
| 79 | 22 | Santonio Holmes | 70.58 | 61.50 | 72.46 | 616 | Jets |
| 80 | 23 | Hines Ward | 70.53 | 66.40 | 69.12 | 314 | Steelers |
| 81 | 24 | Naaman Roosevelt | 70.48 | 57.60 | 74.90 | 231 | Bills |
| 82 | 25 | Kevin Ogletree | 70.33 | 62.10 | 71.65 | 199 | Cowboys |
| 83 | 26 | Matt Willis | 70.32 | 61.70 | 71.90 | 277 | Broncos |
| 84 | 27 | Louis Murphy Jr. | 70.27 | 54.80 | 76.41 | 166 | Raiders |
| 85 | 28 | Greg Salas | 69.96 | 63.10 | 70.37 | 139 | Rams |
| 86 | 29 | Chaz Schilens | 69.91 | 67.70 | 67.22 | 241 | Raiders |
| 87 | 30 | Ruvell Martin | 69.84 | 56.40 | 74.64 | 114 | Bills |
| 88 | 31 | Eric Decker | 69.79 | 63.20 | 70.02 | 504 | Broncos |
| 89 | 32 | David Nelson | 69.62 | 64.80 | 68.66 | 565 | Bills |
| 90 | 33 | Riley Cooper | 69.50 | 59.00 | 72.33 | 217 | Eagles |
| 91 | 34 | Mike Williams | 69.47 | 62.50 | 69.95 | 948 | Buccaneers |
| 92 | 35 | Lee Evans | 69.29 | 57.90 | 72.72 | 217 | Ravens |
| 93 | 36 | Brandon Gibson | 69.25 | 66.00 | 67.25 | 393 | Rams |
| 94 | 37 | Early Doucet | 69.13 | 59.80 | 71.18 | 447 | Cardinals |
| 95 | 38 | Patrick Turner | 69.02 | 65.60 | 67.13 | 108 | Jets |
| 96 | 39 | Mike Thomas | 68.96 | 61.60 | 69.70 | 434 | Jaguars |
| 97 | 40 | Davone Bess | 68.89 | 61.10 | 69.92 | 405 | Dolphins |
| 98 | 41 | Patrick Crayton | 68.61 | 56.40 | 72.59 | 266 | Chargers |
| 99 | 42 | Nate Burleson | 68.30 | 62.50 | 68.00 | 700 | Lions |
| 100 | 43 | Anthony Armstrong | 68.16 | 51.90 | 74.83 | 193 | Commanders |
| 101 | 44 | Titus Young | 68.12 | 63.40 | 67.10 | 544 | Lions |
| 102 | 45 | Dezmon Briscoe | 67.88 | 61.30 | 68.10 | 396 | Buccaneers |
| 103 | 46 | Lavelle Hawkins | 67.74 | 61.70 | 67.60 | 463 | Titans |
| 104 | 47 | Legedu Naanee | 67.61 | 58.10 | 69.79 | 497 | Panthers |
| 105 | 48 | Donald Driver | 67.52 | 62.30 | 66.84 | 419 | Packers |
| 106 | 49 | Harry Douglas | 67.25 | 58.40 | 68.98 | 498 | Falcons |
| 107 | 50 | Andre Roberts | 66.77 | 58.90 | 67.85 | 647 | Cardinals |
| 108 | 51 | Greg Camarillo | 66.77 | 58.40 | 68.18 | 154 | Vikings |
| 109 | 52 | Mohamed Massaquoi | 66.68 | 57.90 | 68.37 | 431 | Browns |
| 110 | 53 | Donald Jones | 66.47 | 59.70 | 66.81 | 239 | Bills |
| 111 | 54 | Ted Ginn Jr. | 66.32 | 63.60 | 63.96 | 252 | 49ers |
| 112 | 55 | Derrick Mason | 66.01 | 56.80 | 67.98 | 213 | Texans |
| 113 | 56 | Devin Hester | 65.99 | 54.10 | 69.75 | 340 | Bears |
| 114 | 57 | Eddie Royal | 65.86 | 55.90 | 68.34 | 303 | Broncos |
| 115 | 58 | Derek Hagan | 65.79 | 62.40 | 63.88 | 235 | Bills |
| 116 | 59 | LaQuan Williams | 65.68 | 55.40 | 68.36 | 101 | Ravens |
| 117 | 60 | Eric Weems | 65.45 | 58.50 | 65.92 | 170 | Falcons |
| 118 | 61 | Ramses Barden | 65.44 | 58.70 | 65.76 | 106 | Giants |
| 119 | 62 | Austin Pettis | 65.39 | 60.30 | 64.61 | 246 | Rams |
| 120 | 63 | Greg Little | 65.27 | 58.70 | 65.48 | 627 | Browns |
| 121 | 64 | Ben Obomanu | 65.03 | 59.10 | 64.82 | 359 | Seahawks |
| 122 | 65 | Jonathan Baldwin | 63.91 | 58.90 | 63.09 | 281 | Chiefs |
| 123 | 66 | Julian Edelman | 63.81 | 56.00 | 64.85 | 105 | Patriots |
| 124 | 67 | Sam Hurd | 63.79 | 55.30 | 65.29 | 134 | Bears |
| 125 | 68 | David Anderson | 63.61 | 57.20 | 63.71 | 100 | Commanders |
| 126 | 69 | Brad Smith | 63.49 | 55.00 | 64.98 | 271 | Bills |
| 127 | 70 | Terrence Austin | 63.30 | 54.90 | 64.74 | 152 | Commanders |
| 128 | 71 | Andre Caldwell | 63.26 | 54.20 | 65.14 | 352 | Bengals |
| 129 | 72 | Jarett Dillard | 63.18 | 59.10 | 61.73 | 337 | Jaguars |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 130 | 1 | Cecil Shorts | 58.48 | 46.60 | 62.23 | 123 | Jaguars |
