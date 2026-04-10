# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:38:25Z
- **Requested analysis_year:** 2013 (clamped to 2013)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Myers | 95.63 | 90.80 | 94.69 | 1121 | Texans |
| 2 | 2 | Jason Kelce | 93.85 | 87.50 | 93.91 | 1154 | Eagles |
| 3 | 3 | Dominic Raiola | 93.67 | 88.10 | 93.21 | 1135 | Lions |
| 4 | 4 | Travis Frederick | 93.04 | 85.60 | 93.84 | 999 | Cowboys |
| 5 | 5 | John Sullivan | 92.02 | 86.50 | 91.53 | 1020 | Vikings |
| 6 | 6 | Alex Mack | 91.38 | 85.50 | 91.13 | 1110 | Browns |
| 7 | 7 | Evan Smith | 89.58 | 83.50 | 89.46 | 1118 | Packers |
| 8 | 8 | Will Montgomery | 88.33 | 80.50 | 89.39 | 1145 | Commanders |
| 9 | 9 | Jeremy Zuttah | 87.36 | 80.10 | 88.03 | 1016 | Buccaneers |
| 10 | 10 | Mike Pouncey | 85.95 | 78.31 | 86.87 | 901 | Dolphins |
| 11 | 11 | Ryan Kalil | 85.57 | 77.60 | 86.71 | 1071 | Panthers |
| 12 | 12 | Brian De La Puente | 85.13 | 78.00 | 85.72 | 1273 | Saints |
| 13 | 13 | Nick Hardwick | 84.30 | 76.60 | 85.26 | 1135 | Chargers |
| 14 | 14 | Roberto Garza | 83.69 | 75.80 | 84.79 | 1059 | Bears |
| 15 | 15 | Rodney Hudson | 83.24 | 75.10 | 84.50 | 1089 | Chiefs |
| 16 | 16 | Kyle Cook | 83.02 | 74.80 | 84.33 | 1131 | Bengals |
| 17 | 17 | Nick Mangold | 82.16 | 73.20 | 83.96 | 1050 | Jets |
| 18 | 18 | Eric Wood | 81.70 | 72.20 | 83.86 | 1161 | Bills |
| 19 | 19 | Scott Wells | 81.69 | 71.13 | 84.56 | 739 | Rams |
| 20 | 20 | Max Unger | 81.28 | 71.80 | 83.43 | 930 | Seahawks |
| 21 | 21 | Lyle Sendlein | 81.00 | 71.20 | 83.36 | 1084 | Cardinals |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Ryan Wendell | 79.88 | 71.10 | 81.56 | 1330 | Patriots |
| 23 | 2 | Jonathan Goodwin | 79.81 | 70.60 | 81.78 | 1156 | 49ers |
| 24 | 3 | Gino Gradkowski | 79.21 | 69.20 | 81.72 | 1136 | Ravens |
| 25 | 4 | Brian Schwenke | 77.57 | 66.68 | 80.66 | 567 | Titans |
| 26 | 5 | Tim Barnes | 76.56 | 64.83 | 80.21 | 266 | Rams |
| 27 | 6 | Brad Meester | 76.43 | 67.10 | 78.48 | 1058 | Jaguars |
| 28 | 7 | David Baas | 76.25 | 65.31 | 79.38 | 143 | Giants |
| 29 | 8 | Rich Ohrnberger | 75.92 | 64.52 | 79.35 | 201 | Chargers |
| 30 | 9 | Samson Satele | 75.50 | 66.49 | 77.34 | 953 | Colts |
| 31 | 10 | Jim Cordle | 75.41 | 64.20 | 78.71 | 482 | Giants |
| 32 | 11 | Joe Hawley | 75.28 | 67.99 | 75.98 | 539 | Falcons |

### Starter (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Robert Turner | 70.75 | 59.81 | 73.87 | 394 | Titans |
| 34 | 2 | Andre Gurode | 70.32 | 55.12 | 76.29 | 275 | Raiders |
| 35 | 3 | Peter Konz | 70.12 | 59.62 | 72.95 | 889 | Falcons |
| 36 | 4 | Cody Wallace | 68.04 | 58.65 | 70.13 | 288 | Steelers |
| 37 | 5 | Lemuel Jeanpierre | 67.34 | 60.48 | 67.75 | 283 | Seahawks |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Richard Sherman | 93.89 | 90.10 | 92.67 | 1166 | Seahawks |
| 2 | 2 | Brandon Boykin | 92.93 | 87.75 | 92.22 | 657 | Eagles |
| 3 | 3 | Brent Grimes | 88.93 | 89.40 | 89.97 | 1110 | Dolphins |
| 4 | 4 | Desmond Trufant | 88.91 | 82.50 | 89.01 | 1001 | Falcons |
| 5 | 5 | Dominique Rodgers-Cromartie | 86.51 | 82.60 | 85.59 | 935 | Broncos |
| 6 | 6 | Darrelle Revis | 86.09 | 83.20 | 88.22 | 949 | Buccaneers |
| 7 | 7 | Chris Harris Jr. | 85.71 | 82.70 | 83.97 | 1066 | Broncos |
| 8 | 8 | Tramaine Brock Sr. | 85.43 | 85.30 | 86.86 | 840 | 49ers |
| 9 | 9 | Byron Maxwell | 85.37 | 82.56 | 89.95 | 677 | Seahawks |
| 10 | 10 | Alterraun Verner | 84.36 | 79.60 | 83.36 | 1002 | Titans |
| 11 | 11 | Tramon Williams | 83.07 | 77.40 | 82.69 | 1101 | Packers |
| 12 | 12 | Joe Haden | 82.69 | 77.60 | 84.20 | 1052 | Browns |
| 13 | 13 | Keenan Lewis | 82.66 | 76.40 | 82.66 | 989 | Saints |
| 14 | 14 | Dimitri Patterson | 82.66 | 76.03 | 88.54 | 237 | Dolphins |
| 15 | 15 | Patrick Peterson | 82.54 | 80.00 | 80.07 | 1075 | Cardinals |
| 16 | 16 | Leon Hall | 82.05 | 79.54 | 87.05 | 274 | Bengals |
| 17 | 17 | Vontae Davis | 81.88 | 77.60 | 82.97 | 1081 | Colts |
| 18 | 18 | Captain Munnerlyn | 81.55 | 75.90 | 81.56 | 1035 | Panthers |
| 19 | 19 | Jason McCourty | 81.15 | 75.00 | 81.28 | 1056 | Titans |
| 20 | 20 | Orlando Scandrick | 80.89 | 80.50 | 79.16 | 1089 | Cowboys |
| 21 | 21 | Adam Jones | 80.65 | 78.30 | 79.72 | 1035 | Bengals |
| 22 | 22 | Leodis McKelvin | 80.39 | 76.40 | 82.44 | 927 | Bills |
| 23 | 23 | Alan Ball | 80.34 | 75.60 | 82.35 | 1005 | Jaguars |
| 24 | 24 | Nickell Robey-Coleman | 80.02 | 74.59 | 79.48 | 608 | Bills |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 25 | 1 | Johnathan Joseph | 79.28 | 73.80 | 79.28 | 833 | Texans |
| 26 | 2 | Sam Shields | 78.76 | 71.80 | 81.00 | 881 | Packers |
| 27 | 3 | Drayton Florence | 78.56 | 73.27 | 80.94 | 621 | Panthers |
| 28 | 4 | William Gay | 78.11 | 73.70 | 76.89 | 912 | Steelers |
| 29 | 5 | Trumaine McBride | 77.98 | 74.93 | 81.45 | 607 | Giants |
| 30 | 6 | Lardarius Webb | 77.83 | 74.20 | 79.22 | 972 | Ravens |
| 31 | 7 | Jarrett Bush | 77.77 | 68.28 | 84.20 | 125 | Packers |
| 32 | 8 | DeAngelo Hall | 77.39 | 71.00 | 77.49 | 998 | Commanders |
| 33 | 9 | Logan Ryan | 77.18 | 66.63 | 80.04 | 700 | Patriots |
| 34 | 10 | Brandon Browner | 76.79 | 67.61 | 83.53 | 455 | Seahawks |
| 35 | 11 | Tarell Brown | 76.25 | 68.10 | 77.52 | 931 | 49ers |
| 36 | 12 | Rashean Mathis | 76.09 | 72.28 | 77.69 | 771 | Lions |
| 37 | 13 | Tim Jennings | 75.86 | 70.00 | 76.23 | 1032 | Bears |
| 38 | 14 | Cortez Allen | 75.85 | 69.92 | 78.46 | 704 | Steelers |
| 39 | 15 | Terence Newman | 75.69 | 71.57 | 76.25 | 819 | Bengals |
| 40 | 16 | Mike Harris | 75.27 | 67.28 | 77.59 | 404 | Jaguars |
| 41 | 17 | Terrell Thomas | 75.00 | 67.89 | 75.57 | 569 | Giants |
| 42 | 18 | Trumaine Johnson | 74.79 | 66.70 | 76.40 | 873 | Rams |
| 43 | 19 | Kyle Arrington | 74.69 | 66.40 | 76.05 | 916 | Patriots |
| 44 | 20 | Robert McClain | 74.68 | 66.58 | 75.91 | 577 | Falcons |
| 45 | 21 | Brandon Carr | 74.55 | 66.70 | 75.62 | 1117 | Cowboys |
| 46 | 22 | Jeremy Lane | 74.06 | 68.84 | 79.36 | 201 | Seahawks |

### Starter (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 47 | 1 | Bradley Fletcher | 73.88 | 67.70 | 77.39 | 997 | Eagles |
| 48 | 2 | Coty Sensabaugh | 73.76 | 68.98 | 74.86 | 494 | Titans |
| 49 | 3 | Jimmy Smith | 73.59 | 66.80 | 75.94 | 1041 | Ravens |
| 50 | 4 | Sean Smith | 73.57 | 65.80 | 74.58 | 1074 | Chiefs |
| 51 | 5 | Janoris Jenkins | 73.57 | 65.20 | 75.37 | 1045 | Rams |
| 52 | 6 | Robert Alford | 73.51 | 65.49 | 75.72 | 570 | Falcons |
| 53 | 7 | Xavier Rhodes | 72.96 | 64.88 | 77.31 | 675 | Vikings |
| 54 | 8 | Cary Williams | 72.72 | 64.20 | 74.24 | 1211 | Eagles |
| 55 | 9 | Prince Amukamara | 72.58 | 65.80 | 75.34 | 1080 | Giants |
| 56 | 10 | Josh Wilson | 72.50 | 63.80 | 74.13 | 968 | Commanders |
| 57 | 11 | Zackary Bowman | 72.26 | 67.38 | 76.45 | 593 | Bears |
| 58 | 12 | Corey White | 72.23 | 68.02 | 73.60 | 670 | Saints |
| 59 | 13 | Asante Samuel | 72.21 | 62.77 | 77.36 | 499 | Falcons |
| 60 | 14 | Chris Owens | 72.03 | 66.79 | 75.62 | 539 | Dolphins |
| 61 | 15 | Alfonzo Dennard | 71.86 | 64.60 | 74.75 | 838 | Patriots |
| 62 | 16 | Champ Bailey | 71.63 | 64.82 | 76.37 | 326 | Broncos |
| 63 | 17 | Aqib Talib | 71.54 | 62.50 | 75.80 | 927 | Patriots |
| 64 | 18 | Eric Wright | 71.40 | 63.76 | 79.41 | 118 | 49ers |
| 65 | 19 | Antoine Cason | 70.84 | 61.60 | 75.97 | 165 | Cardinals |
| 66 | 20 | Melvin White | 70.41 | 64.41 | 71.28 | 726 | Panthers |
| 67 | 21 | Josh Gordy | 70.32 | 62.13 | 74.94 | 247 | Colts |
| 68 | 22 | Aaron Ross | 70.31 | 63.81 | 77.88 | 155 | Giants |
| 69 | 23 | Mike Jenkins | 69.90 | 60.60 | 74.21 | 903 | Raiders |
| 70 | 24 | Brandon Flowers | 69.78 | 58.90 | 74.22 | 907 | Chiefs |
| 71 | 25 | Dwayne Gratz | 69.75 | 64.83 | 75.11 | 485 | Jaguars |
| 72 | 26 | Jabari Greer | 69.57 | 59.11 | 76.13 | 541 | Saints |
| 73 | 27 | Carlos Rogers | 69.32 | 60.50 | 71.03 | 1068 | 49ers |
| 74 | 28 | Brandon Harris | 68.99 | 60.55 | 77.43 | 206 | Texans |
| 75 | 29 | Kyle Wilson | 68.78 | 60.53 | 70.11 | 465 | Jets |
| 76 | 30 | Jerraud Powers | 68.73 | 60.80 | 73.18 | 1031 | Cardinals |
| 77 | 31 | David Amerson | 68.45 | 57.90 | 71.31 | 685 | Commanders |
| 78 | 32 | Kareem Jackson | 68.45 | 57.66 | 72.51 | 760 | Texans |
| 79 | 33 | Chris Carr | 68.31 | 66.20 | 74.82 | 146 | Saints |
| 80 | 34 | Marcus Cooper | 68.30 | 53.79 | 73.81 | 708 | Chiefs |
| 81 | 35 | Corey Webster | 67.77 | 61.74 | 73.88 | 165 | Giants |
| 82 | 36 | Javier Arenas | 67.55 | 59.20 | 72.80 | 102 | Cardinals |
| 83 | 37 | Nolan Carroll | 67.54 | 56.66 | 72.81 | 794 | Dolphins |
| 84 | 38 | Kayvon Webster | 67.40 | 56.99 | 70.18 | 492 | Broncos |
| 85 | 39 | Davon House | 66.37 | 55.37 | 72.04 | 526 | Packers |
| 86 | 40 | Stephon Gilmore | 66.02 | 56.72 | 71.31 | 647 | Bills |
| 87 | 41 | Darrin Walls | 65.97 | 61.95 | 74.58 | 289 | Jets |
| 88 | 42 | Dre Kirkpatrick | 65.56 | 61.25 | 70.65 | 356 | Bengals |
| 89 | 43 | Josh Thomas | 65.38 | 59.37 | 72.92 | 273 | Panthers |
| 90 | 44 | Dee Milliner | 64.79 | 54.29 | 70.76 | 722 | Jets |
| 91 | 45 | Buster Skrine | 64.60 | 53.40 | 68.93 | 1052 | Browns |
| 92 | 46 | Morris Claiborne | 64.50 | 57.18 | 69.51 | 508 | Cowboys |
| 93 | 47 | Will Blackmon | 64.38 | 64.51 | 67.95 | 670 | Jaguars |
| 94 | 48 | Cassius Vaughn | 64.34 | 57.96 | 68.80 | 410 | Colts |
| 95 | 49 | Leonard Johnson | 64.22 | 54.13 | 68.35 | 693 | Buccaneers |
| 96 | 50 | Charles Tillman | 63.74 | 54.33 | 70.02 | 432 | Bears |
| 97 | 51 | Chris Cook | 63.66 | 58.96 | 68.36 | 736 | Vikings |
| 98 | 52 | Greg Toler | 62.76 | 55.16 | 69.07 | 458 | Colts |
| 99 | 53 | Johnthan Banks | 62.40 | 53.20 | 64.37 | 941 | Buccaneers |
| 100 | 54 | Antonio Cromartie | 62.32 | 46.30 | 68.83 | 1067 | Jets |
| 101 | 55 | Roc Carmichael | 62.17 | 56.90 | 70.89 | 220 | Eagles |
| 102 | 56 | Darius Slay | 62.12 | 55.90 | 67.30 | 338 | Lions |

### Rotation/backup (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 103 | 1 | Tracy Porter | 61.09 | 50.80 | 66.92 | 987 | Raiders |
| 104 | 2 | Richard Marshall | 60.96 | 50.50 | 67.52 | 776 | Chargers |
| 105 | 3 | Jayron Hosley | 60.79 | 58.32 | 66.33 | 112 | Giants |
| 106 | 4 | Jonte Green | 60.78 | 57.24 | 69.00 | 149 | Lions |
| 107 | 5 | Chris Houston | 59.89 | 49.10 | 65.83 | 729 | Lions |
| 108 | 6 | B.W. Webb | 59.85 | 58.84 | 61.55 | 179 | Cowboys |
| 109 | 7 | Marcus Sherels | 59.73 | 54.07 | 64.76 | 530 | Vikings |
| 110 | 8 | Dunta Robinson | 58.65 | 50.01 | 63.89 | 282 | Chiefs |
| 111 | 9 | Phillip Adams | 58.61 | 47.93 | 63.65 | 340 | Raiders |
| 112 | 10 | Cortland Finnegan | 58.28 | 49.23 | 64.83 | 362 | Rams |
| 113 | 11 | Tony Carter | 58.21 | 43.71 | 66.71 | 330 | Broncos |
| 114 | 12 | Johnny Patrick | 57.94 | 51.58 | 62.18 | 475 | Chargers |
| 115 | 13 | Derek Cox | 57.68 | 46.13 | 66.12 | 556 | Chargers |
| 116 | 14 | Shaun Prater | 57.48 | 61.84 | 68.91 | 159 | Vikings |
| 117 | 15 | D.J. Hayden | 57.09 | 55.97 | 62.01 | 338 | Raiders |
| 118 | 16 | Shareece Wright | 56.52 | 45.40 | 65.60 | 935 | Chargers |
| 119 | 17 | Leon McFadden | 56.47 | 50.59 | 63.52 | 244 | Browns |
| 120 | 18 | Brice McCain | 56.05 | 40.00 | 63.84 | 605 | Texans |
| 121 | 19 | Josh Norman | 53.00 | 55.72 | 56.66 | 102 | Panthers |
| 122 | 20 | Chimdi Chekwa | 51.25 | 48.85 | 61.18 | 164 | Raiders |

## DI — Defensive Interior

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | J.J. Watt | 96.98 | 90.32 | 97.26 | 960 | Texans |
| 2 | 2 | Calais Campbell | 87.95 | 86.91 | 85.41 | 963 | Cardinals |
| 3 | 3 | Geno Atkins | 85.69 | 83.12 | 86.89 | 455 | Bengals |
| 4 | 4 | Marcell Dareus | 85.55 | 88.07 | 79.71 | 823 | Bills |
| 5 | 5 | Ndamukong Suh | 85.33 | 88.30 | 79.38 | 878 | Lions |
| 6 | 6 | Muhammad Wilkerson | 84.75 | 82.83 | 81.86 | 1041 | Jets |
| 7 | 7 | Kyle Williams | 84.35 | 82.94 | 83.41 | 940 | Bills |
| 8 | 8 | Sheldon Richardson | 83.94 | 81.13 | 81.65 | 882 | Jets |
| 9 | 9 | Star Lotulelei | 83.46 | 76.00 | 84.27 | 653 | Panthers |
| 10 | 10 | Jurrell Casey | 83.15 | 83.02 | 79.59 | 874 | Titans |
| 11 | 11 | Gerald McCoy | 81.55 | 89.83 | 73.95 | 963 | Buccaneers |
| 12 | 12 | Nick Fairley | 80.73 | 74.52 | 83.20 | 670 | Lions |

### Good (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Kawann Short | 79.35 | 72.91 | 79.48 | 551 | Panthers |
| 14 | 2 | Jason Hatcher | 78.20 | 70.43 | 80.37 | 748 | Cowboys |
| 15 | 3 | Steve McLendon | 77.89 | 71.53 | 79.21 | 350 | Steelers |
| 16 | 4 | Damon Harrison Sr. | 77.86 | 70.96 | 82.59 | 500 | Jets |
| 17 | 5 | Mike Daniels | 77.69 | 65.69 | 81.52 | 542 | Packers |
| 18 | 6 | Brandon Mebane | 77.42 | 73.74 | 75.71 | 623 | Seahawks |
| 19 | 7 | Mike Martin | 77.41 | 63.97 | 84.15 | 233 | Titans |
| 20 | 8 | Haloti Ngata | 77.28 | 76.00 | 74.49 | 700 | Ravens |
| 21 | 9 | Linval Joseph | 77.17 | 71.78 | 77.11 | 578 | Giants |
| 22 | 10 | Akiem Hicks | 76.50 | 66.03 | 80.10 | 741 | Saints |
| 23 | 11 | Malik Jackson | 75.91 | 64.66 | 80.03 | 713 | Broncos |
| 24 | 12 | Randy Starks | 75.77 | 73.02 | 73.44 | 729 | Dolphins |
| 25 | 13 | Terrance Knighton | 75.73 | 75.96 | 72.04 | 698 | Broncos |
| 26 | 14 | Dontari Poe | 75.19 | 74.67 | 71.37 | 1036 | Chiefs |
| 27 | 15 | Arthur Jones | 74.93 | 64.91 | 78.48 | 521 | Ravens |
| 28 | 16 | Fletcher Cox | 74.47 | 71.11 | 72.93 | 948 | Eagles |
| 29 | 17 | Justin Smith | 74.45 | 58.62 | 80.83 | 913 | 49ers |
| 30 | 18 | Paul Soliai | 74.39 | 65.55 | 76.64 | 522 | Dolphins |
| 31 | 19 | Dan Williams | 74.34 | 73.35 | 73.44 | 286 | Cardinals |
| 32 | 20 | Leger Douzable | 74.23 | 63.21 | 77.61 | 237 | Jets |
| 33 | 21 | Karl Klug | 74.05 | 69.22 | 73.11 | 319 | Titans |

### Starter (90 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Tyson Jackson | 73.97 | 69.81 | 72.89 | 510 | Chiefs |
| 35 | 2 | Kenrick Ellis | 73.84 | 66.49 | 78.12 | 208 | Jets |
| 36 | 3 | Fred Evans | 73.47 | 60.84 | 78.76 | 349 | Vikings |
| 37 | 4 | Michael Brockers | 73.36 | 66.10 | 75.20 | 792 | Rams |
| 38 | 5 | Glenn Dorsey | 73.18 | 73.82 | 72.56 | 521 | 49ers |
| 39 | 6 | Brodrick Bunkley | 73.12 | 63.47 | 76.73 | 311 | Saints |
| 40 | 7 | Desmond Bryant | 72.57 | 62.60 | 77.33 | 581 | Browns |
| 41 | 8 | Jared Odrick | 72.20 | 65.23 | 72.68 | 856 | Dolphins |
| 42 | 9 | Sean Lissemore | 72.16 | 61.57 | 77.45 | 213 | Chargers |
| 43 | 10 | Mike Devito | 71.90 | 64.53 | 74.00 | 447 | Chiefs |
| 44 | 11 | Johnathan Hankins | 71.70 | 67.13 | 75.78 | 192 | Giants |
| 45 | 12 | Sammie Lee Hill | 71.69 | 62.68 | 75.42 | 384 | Titans |
| 46 | 13 | Pat Sims | 71.54 | 63.34 | 76.08 | 677 | Raiders |
| 47 | 14 | Antonio Smith | 71.47 | 62.36 | 73.90 | 739 | Texans |
| 48 | 15 | Ropati Pitoitua | 71.43 | 57.11 | 77.54 | 576 | Titans |
| 49 | 16 | Sylvester Williams | 71.41 | 55.61 | 77.77 | 390 | Broncos |
| 50 | 17 | Phil Taylor Sr. | 71.30 | 66.27 | 73.50 | 547 | Browns |
| 51 | 18 | Cameron Heyward | 70.99 | 61.13 | 73.39 | 830 | Steelers |
| 52 | 19 | Earl Mitchell | 70.94 | 59.63 | 74.31 | 535 | Texans |
| 53 | 20 | Ahtyba Rubin | 70.55 | 64.82 | 72.19 | 624 | Browns |
| 54 | 21 | Cory Redding | 70.43 | 54.17 | 77.42 | 731 | Colts |
| 55 | 22 | C.J. Mosley | 70.25 | 63.96 | 70.48 | 320 | Lions |
| 56 | 23 | Lawrence Guy Sr. | 70.14 | 62.22 | 74.90 | 206 | Chargers |
| 57 | 24 | Cedric Thornton | 70.08 | 54.29 | 76.44 | 767 | Eagles |
| 58 | 25 | Clinton McDonald | 69.93 | 58.71 | 73.44 | 655 | Seahawks |
| 59 | 26 | Bennie Logan | 69.80 | 57.29 | 73.98 | 524 | Eagles |
| 60 | 27 | Ricky Jean Francois | 69.72 | 59.25 | 74.62 | 500 | Colts |
| 61 | 28 | Sealver Siliga | 69.65 | 68.03 | 78.28 | 323 | Patriots |
| 62 | 29 | Alan Branch | 69.40 | 60.77 | 71.18 | 596 | Bills |
| 63 | 30 | Vance Walker | 69.39 | 57.04 | 73.97 | 767 | Raiders |
| 64 | 31 | John Hughes | 69.32 | 62.67 | 70.24 | 398 | Browns |
| 65 | 32 | Billy Winn | 69.25 | 60.88 | 73.92 | 312 | Browns |
| 66 | 33 | Henry Melton | 69.25 | 58.49 | 79.85 | 123 | Bears |
| 67 | 34 | Tony McDaniel | 69.17 | 56.19 | 76.05 | 610 | Seahawks |
| 68 | 35 | Chris Canty | 68.96 | 60.23 | 73.32 | 565 | Ravens |
| 69 | 36 | Sharrif Floyd | 68.74 | 60.77 | 69.88 | 461 | Vikings |
| 70 | 37 | Datone Jones | 68.63 | 55.67 | 73.10 | 269 | Packers |
| 71 | 38 | Cullen Jenkins | 68.52 | 49.89 | 76.78 | 698 | Giants |
| 72 | 39 | Corey Liuget | 68.45 | 56.61 | 72.37 | 822 | Chargers |
| 73 | 40 | DeAngelo Tyson | 68.40 | 59.30 | 73.44 | 149 | Ravens |
| 74 | 41 | Armonty Bryant | 68.36 | 53.85 | 78.03 | 188 | Browns |
| 75 | 42 | Cam Thomas | 68.27 | 56.31 | 72.07 | 532 | Chargers |
| 76 | 43 | Red Bryant | 67.66 | 59.23 | 69.12 | 553 | Seahawks |
| 77 | 44 | Ray McDonald | 67.61 | 57.65 | 70.09 | 787 | 49ers |
| 78 | 45 | John Jenkins | 67.47 | 58.67 | 69.17 | 478 | Saints |
| 79 | 46 | C.J. Wilson | 67.44 | 55.97 | 75.50 | 127 | Packers |
| 80 | 47 | Joe Vellano | 67.33 | 50.68 | 74.26 | 699 | Patriots |
| 81 | 48 | Kevin Williams | 67.16 | 60.99 | 68.04 | 722 | Vikings |
| 82 | 49 | Chris Baker | 67.09 | 53.39 | 76.02 | 411 | Commanders |
| 83 | 50 | Al Woods | 66.84 | 57.73 | 72.60 | 217 | Steelers |
| 84 | 51 | Domata Peko Sr. | 66.84 | 51.10 | 73.16 | 691 | Bengals |
| 85 | 52 | Kevin Vickerson | 66.69 | 53.01 | 76.55 | 392 | Broncos |
| 86 | 53 | Ishmaa'ily Kitchen | 66.65 | 58.46 | 70.30 | 190 | Browns |
| 87 | 54 | Kendall Langford | 66.56 | 56.73 | 68.95 | 743 | Rams |
| 88 | 55 | Tyson Alualu | 66.56 | 58.72 | 67.62 | 743 | Jaguars |
| 89 | 56 | Tyrunn Walker | 66.40 | 58.98 | 78.05 | 119 | Saints |
| 90 | 57 | Derek Landri | 66.31 | 55.13 | 74.08 | 123 | Buccaneers |
| 91 | 58 | Stephen Paea | 66.11 | 55.36 | 72.03 | 474 | Bears |
| 92 | 59 | Vince Wilfork | 66.06 | 56.70 | 74.38 | 172 | Patriots |
| 93 | 60 | Colin Cole | 66.05 | 49.23 | 74.27 | 329 | Panthers |
| 94 | 61 | Jonathan Babineaux | 65.88 | 53.75 | 70.21 | 903 | Falcons |
| 95 | 62 | Glenn Foster | 65.77 | 54.28 | 71.34 | 221 | Saints |
| 96 | 63 | Ryan Pickett | 65.65 | 53.17 | 70.00 | 534 | Packers |
| 97 | 64 | Sen'Derrick Marks | 65.63 | 56.63 | 68.09 | 930 | Jaguars |
| 98 | 65 | Dwan Edwards | 65.53 | 52.03 | 73.06 | 368 | Panthers |
| 99 | 66 | Darnell Dockett | 65.32 | 47.08 | 73.63 | 866 | Cardinals |
| 100 | 67 | Letroy Guion | 65.19 | 54.83 | 69.50 | 391 | Vikings |
| 101 | 68 | Mike Patterson | 65.16 | 55.46 | 71.11 | 402 | Giants |
| 102 | 69 | Alex Carrington | 65.15 | 60.35 | 72.42 | 163 | Bills |
| 103 | 70 | Corey Peters | 65.09 | 54.06 | 70.04 | 656 | Falcons |
| 104 | 71 | Tommy Kelly | 65.06 | 57.32 | 71.78 | 218 | Patriots |
| 105 | 72 | Derek Wolfe | 65.00 | 57.81 | 68.87 | 552 | Broncos |
| 106 | 73 | Barry Cofield | 64.99 | 56.26 | 66.64 | 729 | Commanders |
| 107 | 74 | Tom Johnson | 64.93 | 53.26 | 70.11 | 243 | Saints |
| 108 | 75 | Aubrayo Franklin | 64.87 | 50.98 | 71.21 | 382 | Colts |
| 109 | 76 | Shaun Rogers | 64.28 | 55.28 | 73.87 | 223 | Giants |
| 110 | 77 | Allen Bailey | 63.38 | 57.46 | 65.86 | 492 | Chiefs |
| 111 | 78 | Antonio Johnson | 63.23 | 54.06 | 65.50 | 379 | Titans |
| 112 | 79 | Alameda Ta'amu | 63.17 | 55.89 | 65.94 | 225 | Cardinals |
| 113 | 80 | Abry Jones | 63.16 | 59.86 | 69.52 | 129 | Jaguars |
| 114 | 81 | Tony Jerod-Eddie | 63.14 | 53.06 | 71.56 | 437 | 49ers |
| 115 | 82 | Brian Sanford | 63.04 | 53.85 | 78.42 | 105 | Browns |
| 116 | 83 | Josh Chapman | 62.90 | 52.27 | 66.86 | 300 | Colts |
| 117 | 84 | Jared Crick | 62.67 | 55.99 | 62.95 | 268 | Texans |
| 118 | 85 | Johnny Jolly | 62.65 | 49.68 | 70.26 | 286 | Packers |
| 119 | 86 | Brett Keisel | 62.64 | 48.12 | 70.44 | 565 | Steelers |
| 120 | 87 | Gary Gibson | 62.51 | 51.61 | 67.18 | 164 | Buccaneers |
| 121 | 88 | Clifton Geathers | 62.47 | 56.20 | 66.78 | 257 | Eagles |
| 122 | 89 | Stephen Bowen | 62.44 | 50.10 | 69.64 | 415 | Commanders |
| 123 | 90 | B.J. Raji | 62.40 | 53.94 | 63.87 | 653 | Packers |

### Rotation/backup (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 124 | 1 | Corbin Bryant | 61.96 | 49.69 | 67.00 | 330 | Bills |
| 125 | 2 | Isaac Sopoaga | 61.91 | 49.12 | 67.30 | 356 | Patriots |
| 126 | 3 | Kendall Reyes | 61.69 | 50.08 | 65.26 | 817 | Chargers |
| 127 | 4 | Stacy McGee | 61.67 | 52.40 | 64.71 | 350 | Raiders |
| 128 | 5 | Chris Jones | 61.66 | 44.02 | 70.29 | 902 | Patriots |
| 129 | 6 | Peria Jerry | 61.43 | 47.37 | 66.63 | 663 | Falcons |
| 130 | 7 | Nate Collins | 60.89 | 59.57 | 67.51 | 191 | Bears |
| 131 | 8 | Brandon Thompson | 60.86 | 54.78 | 65.83 | 428 | Bengals |
| 132 | 9 | Jermelle Cudjo | 60.84 | 52.59 | 65.31 | 212 | Rams |
| 133 | 10 | Josh Boyd | 60.69 | 56.01 | 66.94 | 116 | Packers |
| 134 | 11 | Jarvis Jenkins | 60.47 | 55.03 | 62.53 | 331 | Commanders |
| 135 | 12 | Corey Wootton | 60.21 | 42.67 | 67.73 | 846 | Bears |
| 136 | 13 | Frostee Rucker | 60.09 | 45.98 | 65.33 | 356 | Cardinals |
| 137 | 14 | Terrence Cody | 60.08 | 54.26 | 61.87 | 234 | Ravens |
| 138 | 15 | Fili Moala | 59.87 | 50.32 | 64.98 | 506 | Colts |
| 139 | 16 | Mitch Unrein | 59.79 | 52.43 | 60.73 | 407 | Broncos |
| 140 | 17 | Daniel Muir | 59.60 | 51.24 | 68.40 | 212 | Raiders |
| 141 | 18 | Jay Ratliff | 59.47 | 51.11 | 69.73 | 207 | Bears |
| 142 | 19 | Roy Miller | 59.45 | 49.83 | 63.04 | 575 | Jaguars |
| 143 | 20 | Brandon Deaderick | 59.29 | 52.09 | 62.11 | 302 | Jaguars |
| 144 | 21 | Devon Still | 59.11 | 56.82 | 63.50 | 130 | Bengals |
| 145 | 22 | Kedric Golston | 59.08 | 48.21 | 63.63 | 467 | Commanders |
| 146 | 23 | Drake Nevis | 58.44 | 47.18 | 68.35 | 270 | Jaguars |
| 147 | 24 | Ricardo Mathews | 58.36 | 49.80 | 60.74 | 477 | Colts |
| 148 | 25 | Andre Fluellen | 58.22 | 55.07 | 61.79 | 165 | Lions |
| 149 | 26 | Matt Conrath | 57.69 | 54.82 | 66.51 | 135 | Rams |
| 150 | 27 | Terrell McClain | 57.50 | 57.16 | 58.14 | 176 | Texans |
| 151 | 28 | Akeem Spence | 56.86 | 47.75 | 58.77 | 695 | Buccaneers |
| 152 | 29 | Nick Hayden | 56.78 | 44.01 | 61.91 | 822 | Cowboys |
| 153 | 30 | Landon Cohen | 54.50 | 48.64 | 60.75 | 381 | Bears |
| 154 | 31 | Corvey Irvin | 52.39 | 54.83 | 55.87 | 117 | Cowboys |
| 155 | 32 | Damion Square | 51.05 | 52.85 | 50.89 | 154 | Eagles |

## ED — Edge

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 92.08 | 93.28 | 90.77 | 539 | Broncos |
| 2 | 2 | Robert Quinn | 89.57 | 96.11 | 81.24 | 831 | Rams |
| 3 | 3 | Justin Houston | 87.90 | 86.56 | 86.71 | 756 | Chiefs |
| 4 | 4 | Cameron Wake | 87.79 | 83.12 | 87.25 | 682 | Dolphins |
| 5 | 5 | Charles Johnson | 87.19 | 84.62 | 85.47 | 778 | Panthers |
| 6 | 6 | DeMarcus Ware | 85.13 | 77.85 | 87.39 | 630 | Cowboys |
| 7 | 7 | Carlos Dunlap | 84.72 | 90.57 | 77.59 | 985 | Bengals |
| 8 | 8 | Jerry Hughes | 84.45 | 82.91 | 82.35 | 602 | Bills |
| 9 | 9 | Greg Hardy | 84.12 | 88.42 | 77.41 | 937 | Panthers |
| 10 | 10 | Brandon Graham | 83.38 | 81.88 | 82.91 | 346 | Eagles |
| 11 | 11 | Mario Williams | 83.26 | 84.11 | 80.81 | 1000 | Bills |
| 12 | 12 | Michael Bennett | 82.83 | 92.41 | 72.70 | 736 | Seahawks |
| 13 | 13 | Cliff Avril | 81.26 | 77.69 | 79.48 | 670 | Seahawks |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Chris Long | 79.93 | 73.38 | 80.13 | 839 | Rams |
| 15 | 2 | Terrell Suggs | 79.51 | 77.22 | 78.12 | 903 | Ravens |
| 16 | 3 | Cameron Jordan | 79.12 | 86.83 | 69.82 | 1017 | Saints |
| 17 | 4 | Jason Babin | 78.56 | 63.86 | 84.20 | 764 | Jaguars |
| 18 | 5 | Jared Allen | 78.39 | 71.24 | 78.99 | 1062 | Vikings |
| 19 | 6 | Justin Tuck | 75.96 | 71.69 | 74.95 | 873 | Giants |
| 20 | 7 | Ezekiel Ansah | 74.95 | 74.17 | 73.38 | 554 | Lions |
| 21 | 8 | Michael Johnson | 74.57 | 83.91 | 64.18 | 960 | Bengals |
| 22 | 9 | Chandler Jones | 74.49 | 76.08 | 69.27 | 1264 | Patriots |
| 23 | 10 | Julius Peppers | 74.11 | 68.57 | 73.64 | 851 | Bears |

### Starter (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Ryan Kerrigan | 73.31 | 62.49 | 76.36 | 974 | Commanders |
| 25 | 2 | William Hayes | 73.12 | 74.52 | 70.31 | 345 | Rams |
| 26 | 3 | Everson Griffen | 72.94 | 71.15 | 69.96 | 698 | Vikings |
| 27 | 4 | Jason Pierre-Paul | 72.92 | 77.52 | 68.29 | 568 | Giants |
| 28 | 5 | Vinny Curry | 72.86 | 65.91 | 77.88 | 332 | Eagles |
| 29 | 6 | Chris Clemons | 72.34 | 62.10 | 75.00 | 707 | Seahawks |
| 30 | 7 | Manny Lawson | 71.88 | 49.70 | 83.01 | 705 | Bills |
| 31 | 8 | Parys Haralson | 70.80 | 57.04 | 76.01 | 367 | Saints |
| 32 | 9 | Robert Ayers | 70.45 | 74.98 | 63.26 | 605 | Broncos |
| 33 | 10 | Dion Jordan | 69.73 | 68.08 | 66.66 | 330 | Dolphins |
| 34 | 11 | Jabaal Sheard | 69.20 | 67.92 | 67.46 | 651 | Browns |
| 35 | 12 | Brian Robison | 69.18 | 63.19 | 69.00 | 973 | Vikings |
| 36 | 13 | Wallace Gilberry | 68.36 | 54.10 | 74.01 | 537 | Bengals |
| 37 | 14 | Olivier Vernon | 68.16 | 66.47 | 65.12 | 912 | Dolphins |
| 38 | 15 | Devin Taylor | 67.78 | 64.35 | 67.98 | 301 | Lions |
| 39 | 16 | Shaun Phillips | 67.56 | 53.47 | 73.62 | 912 | Broncos |
| 40 | 17 | Ryan Davis Sr. | 67.22 | 61.48 | 78.59 | 105 | Jaguars |
| 41 | 18 | O'Brien Schofield | 66.67 | 59.71 | 70.89 | 169 | Seahawks |
| 42 | 19 | Jonathan Massaquoi | 64.60 | 61.89 | 66.92 | 527 | Falcons |
| 43 | 20 | Brooks Reed | 64.02 | 59.98 | 63.18 | 999 | Texans |
| 44 | 21 | Israel Idonije | 63.52 | 56.31 | 64.67 | 333 | Lions |
| 45 | 22 | Mathias Kiwanuka | 63.20 | 51.88 | 66.58 | 871 | Giants |
| 46 | 23 | Adrian Clayborn | 63.19 | 60.56 | 64.84 | 934 | Buccaneers |
| 47 | 24 | Melvin Ingram III | 62.99 | 62.10 | 65.94 | 235 | Chargers |
| 48 | 25 | Michael Buchanan | 62.34 | 60.27 | 65.80 | 119 | Patriots |
| 49 | 26 | Derrick Shelby | 62.29 | 61.52 | 58.63 | 438 | Dolphins |
| 50 | 27 | William Gholston | 62.15 | 60.10 | 63.52 | 312 | Buccaneers |

### Rotation/backup (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 51 | 1 | Matt Shaughnessy | 61.95 | 59.86 | 61.88 | 717 | Cardinals |
| 52 | 2 | Andre Branch | 61.74 | 63.77 | 57.38 | 592 | Jaguars |
| 53 | 3 | George Selvie | 61.58 | 54.43 | 64.36 | 746 | Cowboys |
| 54 | 4 | Frank Alexander | 61.51 | 60.38 | 60.05 | 243 | Panthers |
| 55 | 5 | Mario Addison | 61.43 | 59.24 | 60.61 | 266 | Panthers |
| 56 | 6 | David Bass | 60.64 | 59.76 | 61.22 | 311 | Bears |
| 57 | 7 | Damontre Moore | 58.19 | 61.29 | 55.09 | 133 | Giants |
| 58 | 8 | Cliff Matthews | 57.37 | 56.46 | 57.04 | 170 | Falcons |
| 59 | 9 | Wes Horton | 57.18 | 57.25 | 59.22 | 169 | Panthers |
| 60 | 10 | Andre Carter | 56.99 | 54.45 | 58.37 | 195 | Patriots |
| 61 | 11 | Eugene Sims | 56.81 | 55.68 | 54.96 | 385 | Rams |
| 62 | 12 | Jason Hunter | 55.74 | 50.20 | 56.83 | 615 | Raiders |
| 63 | 13 | Malliciah Goodman | 55.35 | 58.86 | 50.92 | 302 | Falcons |
| 64 | 14 | Margus Hunt | 55.05 | 57.40 | 54.51 | 168 | Bengals |
| 65 | 15 | Daniel Te'o-Nesheim | 55.04 | 54.21 | 53.61 | 602 | Buccaneers |
| 66 | 16 | Keyunta Dawson | 52.29 | 59.03 | 51.33 | 105 | Saints |
| 67 | 17 | Stansly Maponga | 51.82 | 57.25 | 48.20 | 126 | Falcons |
| 68 | 18 | Lavar Edwards | 49.83 | 58.97 | 50.43 | 150 | Titans |
| 69 | 19 | Josh Martin | 49.23 | 60.12 | 56.30 | 102 | Chiefs |

## G — Guard

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Evan Mathis | 97.93 | 95.00 | 95.71 | 1163 | Eagles |
| 2 | 2 | Josh Sitton | 95.55 | 90.70 | 94.62 | 1185 | Packers |
| 3 | 3 | Brandon Brooks | 94.02 | 87.70 | 94.06 | 1043 | Texans |
| 4 | 4 | Louis Vasquez | 93.02 | 88.20 | 92.07 | 1409 | Broncos |
| 5 | 5 | Geoff Schwartz | 91.49 | 83.70 | 92.51 | 625 | Chiefs |
| 6 | 6 | Brandon Fusco | 90.14 | 83.09 | 90.67 | 904 | Vikings |
| 7 | 7 | Marshal Yanda | 89.98 | 83.80 | 89.93 | 1136 | Ravens |
| 8 | 8 | Matt Slauson | 89.05 | 84.80 | 87.72 | 1059 | Bears |
| 9 | 9 | Andy Levitre | 88.51 | 82.10 | 88.61 | 1069 | Titans |
| 10 | 10 | Ben Grubbs | 87.80 | 82.70 | 87.04 | 1274 | Saints |
| 11 | 11 | David DeCastro | 87.66 | 81.19 | 87.80 | 948 | Steelers |
| 12 | 12 | Larry Warford | 87.61 | 85.10 | 85.11 | 1138 | Lions |
| 13 | 13 | T.J. Lang | 87.60 | 80.40 | 88.23 | 1157 | Packers |
| 14 | 14 | Richie Incognito | 86.35 | 78.47 | 87.44 | 468 | Dolphins |
| 15 | 15 | Todd Herremans | 86.32 | 79.80 | 86.50 | 1163 | Eagles |
| 16 | 16 | Kevin Zeitler | 85.76 | 77.68 | 86.98 | 860 | Bengals |
| 17 | 17 | Shelley Smith | 85.67 | 74.45 | 88.99 | 363 | Rams |
| 18 | 18 | Zane Beadles | 85.50 | 78.00 | 86.34 | 1415 | Broncos |
| 19 | 19 | Logan Mankins | 85.33 | 78.90 | 85.45 | 1317 | Patriots |
| 20 | 20 | Jahri Evans | 85.30 | 79.30 | 85.13 | 1122 | Saints |
| 21 | 21 | Amini Silatolu | 84.84 | 71.34 | 89.68 | 170 | Panthers |
| 22 | 22 | Clint Boling | 84.51 | 76.78 | 85.50 | 776 | Bengals |
| 23 | 23 | Kraig Urbik | 83.91 | 77.20 | 84.22 | 1143 | Bills |
| 24 | 24 | Travelle Wharton | 83.84 | 77.01 | 84.22 | 896 | Panthers |
| 25 | 25 | Garrett Reynolds | 83.34 | 75.81 | 84.19 | 682 | Falcons |
| 26 | 26 | Carl Nicks | 83.22 | 71.07 | 87.15 | 145 | Buccaneers |
| 27 | 27 | Joe Reitz | 82.79 | 69.64 | 87.39 | 145 | Colts |
| 28 | 28 | Chad Rinehart | 81.49 | 73.98 | 82.33 | 775 | Chargers |
| 29 | 29 | Ramon Foster | 81.42 | 73.65 | 82.43 | 842 | Steelers |
| 30 | 30 | Chris Chester | 81.15 | 73.50 | 82.09 | 1145 | Commanders |
| 31 | 31 | Mike Pollak | 81.04 | 71.09 | 83.50 | 397 | Bengals |
| 32 | 32 | John Greco | 80.86 | 73.93 | 81.32 | 923 | Browns |
| 33 | 33 | Chance Warmack | 80.79 | 73.40 | 81.55 | 1075 | Titans |
| 34 | 34 | Uche Nwaneri | 80.69 | 73.70 | 81.19 | 1058 | Jaguars |
| 35 | 35 | Brian Waters | 80.65 | 72.11 | 82.18 | 330 | Cowboys |
| 36 | 36 | Rob Sims | 80.47 | 73.00 | 81.28 | 1138 | Lions |
| 37 | 37 | Justin Blalock | 80.29 | 73.60 | 80.58 | 1077 | Falcons |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 38 | 1 | Jon Asamoah | 79.90 | 71.32 | 81.45 | 661 | Chiefs |
| 39 | 2 | J.R. Sweezy | 79.77 | 71.30 | 81.25 | 1136 | Seahawks |
| 40 | 3 | Willie Colon | 79.62 | 69.10 | 82.47 | 1039 | Jets |
| 41 | 4 | Wade Smith | 79.54 | 72.55 | 80.03 | 972 | Texans |
| 42 | 5 | Joe Berger | 79.48 | 67.88 | 83.05 | 214 | Vikings |
| 43 | 6 | Kyle Long | 79.36 | 70.40 | 81.16 | 1059 | Bears |
| 44 | 7 | Mike Iupati | 79.09 | 70.31 | 80.78 | 842 | 49ers |
| 45 | 8 | Mike Brisiel | 78.92 | 69.35 | 81.13 | 870 | Raiders |
| 46 | 9 | Ron Leary | 78.87 | 70.00 | 80.61 | 992 | Cowboys |
| 47 | 10 | Jason Pinkston | 78.05 | 65.08 | 82.53 | 151 | Browns |
| 48 | 11 | Johnnie Troutman | 77.82 | 66.90 | 80.93 | 680 | Chargers |
| 49 | 12 | John Jerry | 77.55 | 69.30 | 78.89 | 1015 | Dolphins |
| 50 | 13 | Harvey Dahl | 77.27 | 66.21 | 80.47 | 529 | Rams |
| 51 | 14 | Jeff Allen | 77.04 | 68.10 | 78.83 | 961 | Chiefs |
| 52 | 15 | Alex Boone | 76.91 | 66.90 | 79.41 | 1176 | 49ers |
| 53 | 16 | Dan Connolly | 76.91 | 67.70 | 78.88 | 1249 | Patriots |
| 54 | 17 | Daryn Colledge | 76.73 | 67.40 | 78.79 | 1027 | Cardinals |
| 55 | 18 | Kevin Boothe | 76.46 | 67.80 | 78.06 | 1022 | Giants |
| 56 | 19 | Hugh Thornton | 76.02 | 66.50 | 78.20 | 987 | Colts |
| 57 | 20 | Charlie Johnson | 75.61 | 67.30 | 76.99 | 986 | Vikings |
| 58 | 21 | Chris Kuper | 75.59 | 63.44 | 79.53 | 106 | Broncos |
| 59 | 22 | James Carpenter | 74.59 | 63.66 | 77.71 | 823 | Seahawks |
| 60 | 23 | Kelechi Osemele | 74.46 | 65.77 | 76.09 | 424 | Ravens |

### Starter (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 61 | 1 | Jeromey Clary | 73.92 | 65.10 | 75.64 | 1094 | Chargers |
| 62 | 2 | Adam Snyder | 73.89 | 63.04 | 76.95 | 412 | 49ers |
| 63 | 3 | James Brewer | 72.94 | 62.39 | 75.80 | 434 | Giants |
| 64 | 4 | Oniel Cousins | 72.51 | 59.49 | 77.02 | 314 | Browns |
| 65 | 5 | Shawn Lauvao | 72.51 | 63.21 | 74.54 | 742 | Browns |
| 66 | 6 | Gabe Carimi | 72.50 | 63.11 | 74.59 | 212 | Buccaneers |
| 67 | 7 | Vladimir Ducasse | 71.80 | 57.56 | 77.12 | 318 | Jets |
| 68 | 8 | Ted Larsen | 71.40 | 59.58 | 75.12 | 363 | Buccaneers |
| 69 | 9 | Paul Fanaika | 70.68 | 61.00 | 72.96 | 1084 | Cardinals |
| 70 | 10 | Josh Kline | 70.55 | 61.92 | 72.14 | 112 | Patriots |
| 71 | 11 | Brian Winters | 70.44 | 59.65 | 73.46 | 763 | Jets |
| 72 | 12 | Chris Williams | 69.43 | 56.65 | 73.78 | 902 | Rams |
| 73 | 13 | Davin Joseph | 68.79 | 58.90 | 71.22 | 1013 | Buccaneers |
| 74 | 14 | Chris Scott | 68.43 | 56.31 | 72.34 | 537 | Panthers |
| 75 | 15 | Chris Snee | 68.10 | 56.10 | 71.93 | 187 | Giants |
| 76 | 16 | Mike McGlynn | 67.74 | 55.80 | 71.53 | 1019 | Colts |
| 77 | 17 | Tim Lelito | 63.89 | 57.11 | 64.24 | 159 | Saints |
| 78 | 18 | Will Rackley | 62.17 | 47.26 | 67.94 | 649 | Jaguars |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 79 | 1 | Lucas Nix | 59.32 | 42.80 | 66.17 | 642 | Raiders |

## HB — Running Back

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Andre Ellington | 80.59 | 78.41 | 77.88 | 220 | Cardinals |
| 2 | 2 | Marshawn Lynch | 80.01 | 89.05 | 69.82 | 315 | Seahawks |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 3 | 1 | Darren Sproles | 79.94 | 81.78 | 74.55 | 311 | Saints |
| 4 | 2 | LeSean McCoy | 79.86 | 87.00 | 70.93 | 428 | Eagles |
| 5 | 3 | Adrian Peterson | 79.67 | 77.66 | 76.84 | 260 | Vikings |
| 6 | 4 | Jamaal Charles | 79.01 | 84.90 | 70.91 | 396 | Chiefs |
| 7 | 5 | DeMarco Murray | 76.45 | 77.74 | 71.43 | 305 | Cowboys |
| 8 | 6 | Fred Jackson | 75.70 | 75.05 | 71.96 | 271 | Bills |
| 9 | 7 | Pierre Thomas | 75.05 | 77.84 | 69.03 | 301 | Saints |
| 10 | 8 | Eddie Lacy | 74.62 | 83.53 | 64.51 | 252 | Packers |
| 11 | 9 | Alfred Morris | 74.56 | 76.42 | 69.15 | 173 | Commanders |
| 12 | 10 | Joique Bell | 74.08 | 74.76 | 69.46 | 288 | Lions |
| 13 | 11 | C.J. Spiller | 74.02 | 68.55 | 73.50 | 145 | Bills |

### Starter (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Giovani Bernard | 72.38 | 75.75 | 65.96 | 342 | Bengals |
| 15 | 2 | Matt Forte | 72.33 | 72.40 | 68.12 | 433 | Bears |
| 16 | 3 | Toby Gerhart | 71.82 | 66.05 | 71.50 | 107 | Vikings |
| 17 | 4 | Rashad Jennings | 71.33 | 73.89 | 65.46 | 212 | Raiders |
| 18 | 5 | Jacquizz Rodgers | 71.07 | 72.85 | 65.72 | 237 | Falcons |
| 19 | 6 | Maurice Jones-Drew | 70.64 | 71.73 | 65.75 | 266 | Jaguars |
| 20 | 7 | Danny Woodhead | 70.58 | 79.18 | 60.68 | 323 | Chargers |
| 21 | 8 | Donald Brown | 70.49 | 68.67 | 67.53 | 192 | Colts |
| 22 | 9 | Ben Tate | 70.24 | 65.45 | 69.26 | 212 | Texans |
| 23 | 10 | Arian Foster | 70.22 | 70.70 | 65.74 | 158 | Texans |
| 24 | 11 | Reggie Bush | 70.14 | 71.05 | 65.36 | 302 | Lions |
| 25 | 12 | Ryan Mathews | 70.09 | 68.68 | 66.86 | 133 | Chargers |
| 26 | 13 | Knowshon Moreno | 69.59 | 73.60 | 62.75 | 396 | Broncos |
| 27 | 14 | Shane Vereen | 69.44 | 72.39 | 63.31 | 249 | Patriots |
| 28 | 15 | Montee Ball | 69.43 | 68.39 | 65.95 | 166 | Broncos |
| 29 | 16 | Bobby Rainey Jr. | 68.73 | 70.83 | 63.16 | 106 | Buccaneers |
| 30 | 17 | DeAngelo Williams | 68.65 | 65.84 | 66.36 | 171 | Panthers |
| 31 | 18 | Frank Gore | 68.49 | 69.09 | 63.92 | 294 | 49ers |
| 32 | 19 | Steven Jackson | 67.56 | 67.96 | 63.13 | 172 | Falcons |
| 33 | 20 | Doug Martin | 67.13 | 60.00 | 67.71 | 105 | Buccaneers |
| 34 | 21 | Bernard Pierce | 66.94 | 64.96 | 64.10 | 150 | Ravens |
| 35 | 22 | Jason Snelling | 66.62 | 65.60 | 63.14 | 117 | Falcons |
| 36 | 23 | Le'Veon Bell | 66.55 | 70.78 | 59.56 | 328 | Steelers |
| 37 | 24 | Lamar Miller | 66.28 | 68.40 | 60.70 | 279 | Dolphins |
| 38 | 25 | Zac Stacy | 66.23 | 67.22 | 61.40 | 215 | Rams |
| 39 | 26 | Chris Johnson | 65.04 | 62.40 | 62.64 | 379 | Titans |
| 40 | 27 | Darren McFadden | 65.00 | 58.74 | 65.00 | 125 | Raiders |
| 41 | 28 | Brandon Bolden | 64.79 | 58.23 | 65.00 | 152 | Patriots |
| 42 | 29 | Chris Ogbonnaya | 64.69 | 59.58 | 63.93 | 267 | Browns |
| 43 | 30 | Roy Helu | 64.22 | 60.45 | 62.56 | 301 | Commanders |
| 44 | 31 | Rashard Mendenhall | 64.04 | 64.01 | 59.90 | 158 | Cardinals |
| 45 | 32 | Daniel Thomas | 63.77 | 68.42 | 56.51 | 128 | Dolphins |
| 46 | 33 | Trent Richardson | 63.76 | 62.32 | 60.55 | 256 | Colts |
| 47 | 34 | Andre Brown | 63.49 | 61.62 | 60.57 | 137 | Giants |
| 48 | 35 | Robert Turbin | 63.26 | 64.01 | 58.60 | 106 | Seahawks |
| 49 | 36 | Bilal Powell | 62.37 | 63.79 | 57.26 | 270 | Jets |
| 50 | 37 | Ray Rice | 62.26 | 54.82 | 63.06 | 331 | Ravens |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 51 | 1 | BenJarvus Green-Ellis | 61.09 | 58.62 | 58.57 | 150 | Bengals |
| 52 | 2 | Fozzy Whittaker | 59.08 | 59.08 | 54.92 | 112 | Browns |
| 53 | 3 | Jordan Todman | 57.70 | 58.74 | 52.84 | 136 | Jaguars |

## LB — Linebacker

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | NaVorro Bowman | 85.83 | 90.60 | 78.48 | 1208 | 49ers |
| 2 | 2 | Patrick Willis | 85.01 | 88.60 | 78.65 | 1044 | 49ers |
| 3 | 3 | Lavonte David | 84.52 | 85.90 | 79.44 | 1023 | Buccaneers |
| 4 | 4 | Stephen Tulloch | 81.85 | 81.60 | 77.85 | 1028 | Lions |
| 5 | 5 | Derrick Johnson | 81.73 | 85.80 | 74.85 | 1076 | Chiefs |
| 6 | 6 | Thomas Davis Sr. | 81.68 | 83.00 | 77.15 | 1056 | Panthers |
| 7 | 7 | Vontaze Burfict | 80.44 | 77.90 | 77.97 | 1085 | Bengals |
| 8 | 8 | Kiko Alonso | 80.29 | 80.30 | 76.12 | 1144 | Bills |
| 9 | 9 | Karlos Dansby | 80.00 | 81.00 | 75.16 | 1078 | Cardinals |

### Good (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Luke Kuechly | 78.50 | 77.00 | 75.34 | 1057 | Panthers |
| 11 | 2 | James Harrison | 77.33 | 72.95 | 76.08 | 402 | Bengals |
| 12 | 3 | Nigel Bradham | 75.55 | 72.68 | 74.73 | 285 | Bills |
| 13 | 4 | Vincent Rey | 74.84 | 76.84 | 73.50 | 344 | Bengals |
| 14 | 5 | Danny Trevathan | 74.76 | 74.00 | 71.89 | 1116 | Broncos |

### Starter (65 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Dont'a Hightower | 73.79 | 71.60 | 71.09 | 998 | Patriots |
| 16 | 2 | Akeem Jordan | 73.78 | 71.64 | 72.71 | 475 | Chiefs |
| 17 | 3 | K.J. Wright | 73.34 | 71.90 | 70.86 | 806 | Seahawks |
| 18 | 4 | Daryl Washington | 72.78 | 72.70 | 70.95 | 798 | Cardinals |
| 19 | 5 | Malcolm Smith | 72.53 | 73.02 | 71.89 | 618 | Seahawks |
| 20 | 6 | DeAndre Levy | 72.11 | 69.60 | 70.25 | 1018 | Lions |
| 21 | 7 | Jamie Collins Sr. | 71.39 | 68.81 | 68.94 | 435 | Patriots |
| 22 | 8 | Sean Lee | 71.32 | 72.13 | 72.55 | 703 | Cowboys |
| 23 | 9 | Daryl Smith | 71.16 | 72.20 | 70.66 | 1071 | Ravens |
| 24 | 10 | James Laurinaitis | 70.80 | 67.20 | 69.04 | 1055 | Rams |
| 25 | 11 | Jerrell Freeman | 70.66 | 71.10 | 66.20 | 1107 | Colts |
| 26 | 12 | Brandon Spikes | 70.59 | 68.73 | 68.70 | 686 | Patriots |
| 27 | 13 | Bobby Wagner | 70.41 | 66.60 | 68.78 | 1058 | Seahawks |
| 28 | 14 | David Harris | 70.25 | 67.20 | 68.11 | 1100 | Jets |
| 29 | 15 | Arthur Moats | 70.17 | 67.16 | 71.24 | 292 | Bills |
| 30 | 16 | Sio Moore | 70.04 | 66.97 | 68.95 | 578 | Raiders |
| 31 | 17 | Dekoda Watson | 70.02 | 66.51 | 69.45 | 258 | Buccaneers |
| 32 | 18 | Curtis Lofton | 69.47 | 65.80 | 67.75 | 1053 | Saints |
| 33 | 19 | Kavell Conner | 69.38 | 64.27 | 72.89 | 144 | Colts |
| 34 | 20 | Erin Henderson | 68.71 | 65.40 | 68.31 | 850 | Vikings |
| 35 | 21 | Ramon Humber | 68.69 | 63.48 | 69.56 | 172 | Saints |
| 36 | 22 | Paul Posluszny | 68.63 | 63.30 | 68.54 | 1038 | Jaguars |
| 37 | 23 | Nick Roach | 68.04 | 66.20 | 65.10 | 1075 | Raiders |
| 38 | 24 | Brian Cushing | 67.78 | 67.65 | 71.84 | 330 | Texans |
| 39 | 25 | Lawrence Timmons | 67.72 | 64.00 | 66.04 | 1072 | Steelers |
| 40 | 26 | Nate Irving | 67.71 | 65.19 | 70.12 | 354 | Broncos |
| 41 | 27 | Stephen Nicholas | 67.58 | 66.68 | 69.95 | 130 | Falcons |
| 42 | 28 | A.J. Hawk | 67.11 | 61.20 | 67.09 | 1057 | Packers |
| 43 | 29 | Alec Ogletree | 67.06 | 63.00 | 65.60 | 1033 | Rams |
| 44 | 30 | Koa Misi | 67.01 | 64.19 | 65.87 | 473 | Dolphins |
| 45 | 31 | Josh Bynes | 66.98 | 65.21 | 67.32 | 455 | Ravens |
| 46 | 32 | Akeem Ayers | 66.59 | 64.13 | 64.06 | 724 | Titans |
| 47 | 33 | Wesley Woodyard | 66.42 | 61.90 | 65.26 | 844 | Broncos |
| 48 | 34 | Jason Trusnik | 65.91 | 64.71 | 67.54 | 182 | Dolphins |
| 49 | 35 | Russell Allen | 65.45 | 63.24 | 64.64 | 596 | Jaguars |
| 50 | 36 | Zach Brown | 65.39 | 60.48 | 64.49 | 759 | Titans |
| 51 | 37 | Manti Te'o | 65.38 | 58.99 | 66.51 | 585 | Chargers |
| 52 | 38 | D'Qwell Jackson | 65.32 | 60.10 | 64.64 | 1147 | Browns |
| 53 | 39 | John Lotulelei | 65.31 | 60.00 | 67.03 | 108 | Jaguars |
| 54 | 40 | Marvin Mitchell | 65.13 | 62.87 | 67.37 | 306 | Vikings |
| 55 | 41 | A.J. Klein | 64.84 | 63.48 | 66.78 | 130 | Panthers |
| 56 | 42 | Josh McNary | 64.80 | 65.46 | 71.06 | 132 | Colts |
| 57 | 43 | James Anderson | 64.75 | 56.20 | 67.53 | 999 | Bears |
| 58 | 44 | Justin Durant | 64.49 | 62.32 | 65.32 | 200 | Cowboys |
| 59 | 45 | Jonathan Casillas | 64.32 | 62.26 | 65.37 | 198 | Buccaneers |
| 60 | 46 | Geno Hayes | 64.26 | 63.00 | 65.37 | 920 | Jaguars |
| 61 | 47 | Spencer Paysinger | 63.90 | 60.83 | 67.51 | 691 | Giants |
| 62 | 48 | Jacquian Williams | 63.89 | 60.17 | 64.08 | 606 | Giants |
| 63 | 49 | Keith Rivers | 63.87 | 62.29 | 64.41 | 419 | Giants |
| 64 | 50 | Joplo Bartu | 63.80 | 59.42 | 63.59 | 772 | Falcons |
| 65 | 51 | Vince Williams | 63.41 | 57.76 | 64.05 | 401 | Steelers |
| 66 | 52 | Will Witherspoon | 63.39 | 61.55 | 66.70 | 136 | Rams |
| 67 | 53 | DeMeco Ryans | 63.10 | 55.80 | 63.80 | 1234 | Eagles |
| 68 | 54 | Brad Jones | 63.10 | 60.37 | 63.25 | 639 | Packers |
| 69 | 55 | Rey Maualuga | 63.09 | 59.47 | 62.80 | 649 | Bengals |
| 70 | 56 | Donald Butler | 63.02 | 57.70 | 64.16 | 851 | Chargers |
| 71 | 57 | Dannell Ellerbe | 62.97 | 59.10 | 62.95 | 1011 | Dolphins |
| 72 | 58 | Audie Cole | 62.86 | 66.61 | 67.91 | 325 | Vikings |
| 73 | 59 | Mason Foster | 62.85 | 59.13 | 61.68 | 773 | Buccaneers |
| 74 | 60 | Bruce Carter | 62.82 | 59.00 | 65.37 | 875 | Cowboys |
| 75 | 61 | Colin McCarthy | 62.78 | 62.80 | 65.58 | 332 | Titans |
| 76 | 62 | Arthur Brown | 62.59 | 60.75 | 61.73 | 205 | Ravens |
| 77 | 63 | Kevin Burnett | 62.28 | 54.70 | 63.17 | 978 | Raiders |
| 78 | 64 | Jasper Brinkley | 62.24 | 59.70 | 66.63 | 203 | Cardinals |
| 79 | 65 | Paul Worrilow | 62.23 | 57.09 | 63.57 | 773 | Falcons |

### Rotation/backup (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 80 | 1 | Jerod Mayo | 61.83 | 58.12 | 65.34 | 399 | Patriots |
| 81 | 2 | Kelvin Sheppard | 61.17 | 53.54 | 62.72 | 422 | Colts |
| 82 | 3 | Reggie Walker | 60.88 | 54.81 | 63.26 | 540 | Chargers |
| 83 | 4 | Perry Riley | 60.85 | 56.10 | 61.31 | 989 | Commanders |
| 84 | 5 | Jameel McClain | 60.56 | 55.18 | 64.04 | 368 | Ravens |
| 85 | 6 | Bront Bird | 60.53 | 60.25 | 65.09 | 142 | Chargers |
| 86 | 7 | Darryl Sharpton | 60.41 | 54.67 | 64.55 | 718 | Texans |
| 87 | 8 | Joe Mays | 60.39 | 60.57 | 62.08 | 544 | Texans |
| 88 | 9 | Dane Fletcher | 60.21 | 58.03 | 60.83 | 232 | Patriots |
| 89 | 10 | Lance Briggs | 60.14 | 58.35 | 60.81 | 557 | Bears |
| 90 | 11 | Ashlee Palmer | 59.85 | 56.50 | 60.83 | 357 | Lions |
| 91 | 12 | Mychal Kendricks | 59.85 | 53.10 | 60.56 | 1069 | Eagles |
| 92 | 13 | Michael Wilhoite | 59.70 | 61.39 | 66.39 | 175 | 49ers |
| 93 | 14 | Adam Hayward | 59.52 | 56.50 | 61.95 | 188 | Buccaneers |
| 94 | 15 | D.J. Williams | 59.50 | 59.93 | 62.97 | 213 | Bears |
| 95 | 16 | Andrew Gachkar | 59.40 | 60.59 | 65.27 | 167 | Chargers |
| 96 | 17 | Jelani Jenkins | 59.21 | 59.04 | 59.32 | 125 | Dolphins |
| 97 | 18 | Demario Davis | 58.84 | 52.60 | 59.22 | 1049 | Jets |
| 98 | 19 | Pat Angerer | 58.73 | 53.86 | 61.66 | 493 | Colts |
| 99 | 20 | Chase Blackburn | 58.24 | 56.58 | 59.55 | 242 | Panthers |
| 100 | 21 | Sean Weatherspoon | 58.04 | 55.85 | 60.33 | 392 | Falcons |
| 101 | 22 | Jamari Lattimore | 57.86 | 61.76 | 61.09 | 265 | Packers |
| 102 | 23 | JoLonn Dunbar | 57.23 | 52.92 | 58.85 | 419 | Rams |
| 103 | 24 | Jon Bostic | 57.13 | 48.83 | 63.70 | 605 | Bears |
| 104 | 25 | Kaluka Maiava | 56.99 | 54.37 | 60.41 | 118 | Raiders |
| 105 | 26 | London Fletcher | 56.73 | 45.30 | 60.18 | 917 | Commanders |
| 106 | 27 | David Hawthorne | 56.59 | 50.24 | 58.43 | 783 | Saints |
| 107 | 28 | Craig Robertson | 56.46 | 46.90 | 59.97 | 845 | Browns |
| 108 | 29 | Jeff Tarpinian | 56.30 | 59.06 | 62.79 | 181 | Texans |
| 109 | 30 | Philip Wheeler | 55.76 | 48.00 | 57.40 | 1034 | Dolphins |
| 110 | 31 | Moise Fokou | 55.68 | 48.57 | 59.58 | 720 | Titans |
| 111 | 32 | Akeem Dent | 55.61 | 51.08 | 59.25 | 363 | Falcons |
| 112 | 33 | Chad Greenway | 55.58 | 47.70 | 56.66 | 1156 | Vikings |
| 113 | 34 | Paris Lenon | 55.48 | 46.37 | 57.90 | 301 | Broncos |
| 114 | 35 | Najee Goode | 54.95 | 60.39 | 61.02 | 191 | Eagles |
| 115 | 36 | Mark Herzlich | 54.36 | 50.73 | 59.70 | 191 | Giants |
| 116 | 37 | Jon Beason | 53.81 | 48.18 | 61.31 | 783 | Giants |
| 117 | 38 | J.T. Thomas | 52.57 | 52.15 | 59.36 | 190 | Jaguars |
| 118 | 39 | Khaseem Greene | 52.26 | 48.54 | 57.88 | 231 | Bears |
| 119 | 40 | Ernie Sims | 50.98 | 40.00 | 58.10 | 382 | Cowboys |
| 120 | 41 | DeVonte Holloman | 45.00 | 44.94 | 56.00 | 208 | Cowboys |

## QB — Quarterback

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Peyton Manning | 87.65 | 92.62 | 80.74 | 858 | Broncos |
| 2 | 2 | Drew Brees | 84.87 | 86.75 | 78.78 | 822 | Saints |
| 3 | 3 | Philip Rivers | 83.70 | 83.67 | 79.35 | 684 | Chargers |
| 4 | 4 | Russell Wilson | 82.96 | 84.19 | 78.41 | 623 | Seahawks |
| 5 | 5 | Aaron Rodgers | 82.01 | 84.15 | 80.54 | 381 | Packers |

### Good (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Ben Roethlisberger | 79.30 | 83.37 | 72.30 | 663 | Steelers |
| 7 | 2 | Tom Brady | 78.42 | 83.83 | 69.45 | 785 | Patriots |
| 8 | 3 | Tony Romo | 77.97 | 78.07 | 74.04 | 614 | Cowboys |
| 9 | 4 | Matt Ryan | 77.60 | 79.95 | 71.14 | 744 | Falcons |
| 10 | 5 | Matthew Stafford | 75.88 | 79.34 | 68.51 | 694 | Lions |

### Starter (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Nick Foles | 73.20 | 75.07 | 81.66 | 423 | Eagles |
| 12 | 2 | Cam Newton | 72.61 | 70.28 | 70.81 | 612 | Panthers |
| 13 | 3 | Ryan Tannehill | 72.56 | 78.20 | 64.04 | 679 | Dolphins |
| 14 | 4 | Andy Dalton | 72.43 | 70.61 | 69.37 | 726 | Bengals |
| 15 | 5 | Colin Kaepernick | 72.02 | 70.80 | 72.26 | 629 | 49ers |
| 16 | 6 | Andrew Luck | 71.79 | 71.05 | 67.85 | 784 | Colts |
| 17 | 7 | Jay Cutler | 71.77 | 73.66 | 71.60 | 406 | Bears |
| 18 | 8 | Eli Manning | 71.31 | 75.19 | 64.16 | 625 | Giants |
| 19 | 9 | Carson Palmer | 71.30 | 68.80 | 69.67 | 655 | Cardinals |
| 20 | 10 | Alex Smith | 70.59 | 71.77 | 67.76 | 684 | Chiefs |
| 21 | 11 | Robert Griffin III | 68.66 | 71.32 | 64.62 | 548 | Commanders |
| 22 | 12 | Joe Flacco | 68.53 | 69.63 | 63.09 | 716 | Ravens |
| 23 | 13 | Sam Bradford | 66.45 | 70.27 | 65.73 | 297 | Rams |
| 24 | 14 | Josh McCown | 66.03 | 75.92 | 79.42 | 258 | Bears |
| 25 | 15 | Ryan Fitzpatrick | 64.38 | 63.20 | 65.18 | 422 | Titans |
| 26 | 16 | Michael Vick | 63.66 | 62.17 | 70.57 | 178 | Eagles |
| 27 | 17 | Matt Schaub | 62.97 | 65.19 | 61.86 | 400 | Texans |
| 28 | 18 | Jake Locker | 62.32 | 66.37 | 67.08 | 221 | Titans |
| 29 | 19 | Mike Glennon | 62.20 | 65.49 | 62.84 | 502 | Buccaneers |

### Rotation/backup (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Geno Smith | 61.05 | 60.58 | 61.04 | 538 | Jets |
| 31 | 2 | Matt Cassel | 60.50 | 62.44 | 63.83 | 287 | Vikings |
| 32 | 3 | Matt McGloin | 60.36 | 61.82 | 67.00 | 233 | Raiders |
| 33 | 4 | Thaddeus Lewis | 60.19 | 64.21 | 66.42 | 190 | Bills |
| 34 | 5 | Scott Tolzien | 60.18 | 63.27 | 67.16 | 102 | Packers |
| 35 | 6 | Case Keenum | 59.98 | 60.56 | 64.79 | 283 | Texans |
| 36 | 7 | Chad Henne | 59.88 | 60.26 | 59.50 | 577 | Jaguars |
| 37 | 8 | Kellen Clemens | 59.70 | 64.64 | 62.93 | 291 | Rams |
| 38 | 9 | Matt Flynn | 59.35 | 60.91 | 65.24 | 250 | Packers |
| 39 | 10 | Brian Hoyer | 58.88 | 63.98 | 61.32 | 105 | Browns |
| 40 | 11 | E.J. Manuel | 58.75 | 58.00 | 59.51 | 387 | Bills |
| 41 | 12 | Christian Ponder | 58.47 | 56.64 | 62.76 | 301 | Vikings |
| 42 | 13 | Brandon Weeden | 58.05 | 59.19 | 59.43 | 316 | Browns |
| 43 | 14 | Josh Freeman | 57.67 | 59.42 | 60.49 | 167 | Vikings |
| 44 | 15 | Jason Campbell | 57.50 | 60.32 | 58.69 | 357 | Browns |
| 45 | 16 | Kirk Cousins | 56.93 | 58.73 | 57.73 | 165 | Commanders |
| 46 | 17 | Blaine Gabbert | 50.62 | 54.93 | 54.75 | 109 | Jaguars |

## S — Safety

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Kam Chancellor | 91.31 | 89.30 | 88.68 | 1199 | Seahawks |
| 2 | 2 | Eric Weddle | 90.38 | 89.20 | 87.00 | 1145 | Chargers |
| 3 | 3 | Devin McCourty | 89.56 | 86.00 | 87.77 | 1164 | Patriots |
| 4 | 4 | Jairus Byrd | 89.50 | 87.59 | 89.20 | 634 | Bills |
| 5 | 5 | Earl Thomas III | 89.04 | 89.10 | 84.83 | 1208 | Seahawks |
| 6 | 6 | Troy Polamalu | 88.35 | 90.10 | 85.83 | 1072 | Steelers |
| 7 | 7 | Eric Berry | 87.99 | 87.90 | 87.02 | 1077 | Chiefs |
| 8 | 8 | Donte Whitner | 87.57 | 86.40 | 84.18 | 1194 | 49ers |
| 9 | 9 | Will Hill III | 86.65 | 87.94 | 86.95 | 766 | Giants |
| 10 | 10 | Glover Quin | 81.32 | 78.60 | 78.96 | 991 | Lions |
| 11 | 11 | Antrel Rolle | 80.92 | 81.50 | 76.36 | 1126 | Giants |
| 12 | 12 | T.J. Ward | 80.82 | 77.90 | 80.89 | 1112 | Browns |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Aaron Williams | 79.14 | 72.90 | 80.17 | 926 | Bills |
| 14 | 2 | Chris Clemons | 77.76 | 77.70 | 76.33 | 1138 | Dolphins |
| 15 | 3 | Husain Abdullah | 77.48 | 71.86 | 79.46 | 339 | Chiefs |
| 16 | 4 | Keith Tandy | 76.48 | 71.50 | 76.28 | 443 | Buccaneers |
| 17 | 5 | George Wilson | 75.01 | 68.95 | 75.51 | 406 | Titans |
| 18 | 6 | Rashad Johnson | 74.57 | 72.32 | 76.49 | 621 | Cardinals |
| 19 | 7 | Charles Woodson | 74.20 | 71.30 | 71.96 | 1068 | Raiders |

### Starter (45 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Eric Reid | 73.65 | 69.60 | 72.19 | 1159 | 49ers |
| 21 | 2 | Usama Young | 73.62 | 64.53 | 79.36 | 196 | Raiders |
| 22 | 3 | Ryan Clark | 72.98 | 67.90 | 72.52 | 1063 | Steelers |
| 23 | 4 | Mike Mitchell | 72.84 | 73.10 | 69.44 | 964 | Panthers |
| 24 | 5 | Michael Griffin | 72.41 | 65.70 | 73.75 | 900 | Titans |
| 25 | 6 | David Bruton | 72.29 | 63.51 | 76.79 | 171 | Broncos |
| 26 | 7 | Bernard Pollard | 71.90 | 63.50 | 73.34 | 1056 | Titans |
| 27 | 8 | Jamarca Sanford | 71.51 | 69.61 | 70.37 | 793 | Vikings |
| 28 | 9 | Kenny Vaccaro | 71.38 | 63.92 | 74.27 | 793 | Saints |
| 29 | 10 | Robert Lester | 71.37 | 67.16 | 75.21 | 331 | Panthers |
| 30 | 11 | Yeremiah Bell | 71.03 | 65.30 | 70.69 | 1070 | Cardinals |
| 31 | 12 | Jahleel Addae | 70.91 | 66.05 | 69.99 | 496 | Chargers |
| 32 | 13 | Reggie Nelson | 70.86 | 63.80 | 71.72 | 1029 | Bengals |
| 33 | 14 | Nate Allen | 70.70 | 64.00 | 71.84 | 1174 | Eagles |
| 34 | 15 | Rafael Bush | 70.64 | 66.78 | 75.62 | 592 | Saints |
| 35 | 16 | Darrell Stuckey | 69.48 | 62.22 | 75.36 | 162 | Chargers |
| 36 | 17 | Darian Stewart | 69.34 | 70.21 | 69.80 | 568 | Rams |
| 37 | 18 | Rahim Moore | 69.27 | 64.38 | 72.11 | 659 | Broncos |
| 38 | 19 | Shiloh Keo | 69.03 | 70.71 | 68.85 | 768 | Texans |
| 39 | 20 | Duron Harmon | 68.92 | 64.33 | 68.84 | 430 | Patriots |
| 40 | 21 | Taylor Mays | 68.79 | 66.88 | 72.98 | 202 | Bengals |
| 41 | 22 | Antoine Bethea | 68.64 | 57.40 | 71.97 | 1185 | Colts |
| 42 | 23 | Barry Church | 68.14 | 64.70 | 71.16 | 1018 | Cowboys |
| 43 | 24 | Malcolm Jenkins | 68.03 | 65.40 | 66.55 | 938 | Saints |
| 44 | 25 | Quintin Mikell | 67.69 | 60.48 | 68.85 | 706 | Panthers |
| 45 | 26 | Da'Norris Searcy | 67.54 | 62.16 | 69.36 | 728 | Bills |
| 46 | 27 | William Moore | 67.32 | 62.70 | 67.48 | 1043 | Falcons |
| 47 | 28 | Chris Banjo | 67.30 | 64.69 | 68.01 | 193 | Packers |
| 48 | 29 | LaRon Landry | 67.25 | 66.40 | 66.35 | 927 | Colts |
| 49 | 30 | Jim Leonhard | 66.94 | 67.06 | 63.84 | 612 | Bills |
| 50 | 31 | George Iloka | 66.94 | 61.60 | 66.33 | 1088 | Bengals |
| 51 | 32 | Louis Delmas | 66.93 | 67.50 | 65.72 | 1019 | Lions |
| 52 | 33 | Kelcie McCray | 66.89 | 64.28 | 72.79 | 102 | Buccaneers |
| 53 | 34 | Marcus Gilchrist | 66.78 | 56.90 | 69.20 | 1126 | Chargers |
| 54 | 35 | Roman Harper | 66.31 | 60.76 | 68.44 | 482 | Saints |
| 55 | 36 | Eddie Pleasant | 65.50 | 62.52 | 73.74 | 151 | Texans |
| 56 | 37 | Tashaun Gipson Sr. | 65.36 | 59.40 | 67.51 | 1099 | Browns |
| 57 | 38 | Danieal Manning | 65.27 | 62.31 | 68.50 | 321 | Texans |
| 58 | 39 | D.J. Swearinger Sr. | 64.97 | 59.21 | 64.64 | 797 | Texans |
| 59 | 40 | Mike Adams | 64.63 | 52.60 | 68.48 | 861 | Broncos |
| 60 | 41 | Ryan Mundy | 63.73 | 55.99 | 67.96 | 654 | Giants |
| 61 | 42 | Jaiquawn Jarrett | 63.56 | 61.74 | 62.69 | 276 | Jets |
| 62 | 43 | Dawan Landry | 62.99 | 53.80 | 64.95 | 1083 | Jets |
| 63 | 44 | Robert Blanton | 62.88 | 61.11 | 68.22 | 395 | Vikings |
| 64 | 45 | Steve Gregory | 62.86 | 54.10 | 65.37 | 971 | Patriots |

### Rotation/backup (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 65 | 1 | Reshad Jones | 61.58 | 49.90 | 65.61 | 1146 | Dolphins |
| 66 | 2 | T.J. McDonald | 60.12 | 59.47 | 62.64 | 650 | Rams |
| 67 | 3 | Quintin Demps | 60.12 | 54.84 | 61.98 | 653 | Chiefs |
| 68 | 4 | Morgan Burnett | 59.63 | 48.10 | 64.19 | 919 | Packers |
| 69 | 5 | Will Allen | 58.93 | 54.53 | 59.88 | 534 | Steelers |
| 70 | 6 | Sean Richardson | 58.84 | 60.41 | 65.35 | 173 | Packers |
| 71 | 7 | Andrew Sendejo | 58.09 | 53.14 | 62.64 | 728 | Vikings |
| 72 | 8 | James Ihedigbo | 57.96 | 48.60 | 60.35 | 1072 | Ravens |
| 73 | 9 | Dashon Goldson | 57.57 | 51.69 | 58.89 | 807 | Buccaneers |
| 74 | 10 | Earl Wolff | 57.30 | 50.83 | 62.64 | 525 | Eagles |
| 75 | 11 | Harrison Smith | 57.14 | 51.02 | 62.25 | 529 | Vikings |
| 76 | 12 | Jeff Heath | 57.06 | 53.37 | 59.52 | 596 | Cowboys |
| 77 | 13 | Tony Jefferson | 56.57 | 57.21 | 62.84 | 198 | Cardinals |
| 78 | 14 | Rodney McLeod | 56.40 | 54.70 | 58.84 | 1061 | Rams |
| 79 | 15 | Chris Conte | 56.38 | 50.30 | 57.84 | 1029 | Bears |
| 80 | 16 | Matt Giordano | 56.18 | 54.72 | 57.88 | 117 | Rams |
| 81 | 17 | Reed Doughty | 55.99 | 51.27 | 56.22 | 409 | Commanders |
| 82 | 18 | Duke Ihenacho | 55.94 | 46.30 | 58.20 | 923 | Broncos |
| 83 | 19 | Zeke Motta | 55.79 | 60.09 | 62.18 | 153 | Falcons |
| 84 | 20 | Charles Godfrey | 55.30 | 56.72 | 58.21 | 112 | Panthers |
| 85 | 21 | Craig Steltz | 55.19 | 58.88 | 56.70 | 121 | Bears |
| 86 | 22 | Delano Howell | 55.05 | 58.35 | 60.94 | 206 | Colts |
| 87 | 23 | M.D. Jennings | 54.96 | 46.20 | 59.77 | 835 | Packers |
| 88 | 24 | Omar Bolden | 54.55 | 54.89 | 52.10 | 207 | Broncos |
| 89 | 25 | Kendrick Lewis | 54.37 | 43.00 | 59.97 | 1109 | Chiefs |
| 90 | 26 | Patrick Chung | 54.29 | 48.01 | 57.55 | 797 | Eagles |
| 91 | 27 | Mistral Raymond | 53.56 | 53.70 | 57.21 | 203 | Vikings |
| 92 | 28 | Brandon Meriweather | 53.29 | 51.58 | 57.56 | 738 | Commanders |
| 93 | 29 | Matt Elam | 53.02 | 40.00 | 57.53 | 1011 | Ravens |
| 94 | 30 | Johnathan Cyprien | 52.91 | 40.00 | 58.38 | 1044 | Jaguars |
| 95 | 31 | J.J. Wilcox | 52.88 | 43.49 | 59.14 | 515 | Cowboys |
| 96 | 32 | Major Wright | 52.74 | 40.00 | 58.42 | 935 | Bears |
| 97 | 33 | Michael Huff | 52.70 | 47.13 | 57.25 | 163 | Broncos |
| 98 | 34 | Thomas DeCoud | 51.30 | 40.00 | 55.19 | 891 | Falcons |
| 99 | 35 | Josh Evans | 50.74 | 41.46 | 53.80 | 675 | Jaguars |
| 100 | 36 | Bacarri Rambo | 49.21 | 49.83 | 51.93 | 333 | Commanders |
| 101 | 37 | Winston Guy | 48.58 | 50.12 | 52.10 | 353 | Jaguars |

## T — Tackle

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Andrew Whitworth | 95.79 | 91.40 | 94.55 | 996 | Bengals |
| 2 | 2 | Trent Williams | 95.41 | 91.80 | 93.65 | 1143 | Commanders |
| 3 | 3 | Jason Peters | 94.02 | 90.70 | 92.06 | 1077 | Eagles |
| 4 | 4 | Tyron Smith | 93.70 | 89.00 | 92.67 | 997 | Cowboys |
| 5 | 5 | Jake Long | 93.50 | 89.22 | 92.19 | 852 | Rams |
| 6 | 6 | Joe Thomas | 92.34 | 90.30 | 89.53 | 1109 | Browns |
| 7 | 7 | King Dunlap | 91.27 | 85.86 | 90.71 | 779 | Chargers |
| 8 | 8 | Jordan Gross | 90.03 | 87.40 | 87.61 | 1058 | Panthers |
| 9 | 9 | Duane Brown | 89.80 | 84.46 | 89.19 | 953 | Texans |
| 10 | 10 | Michael Roos | 89.51 | 84.80 | 88.48 | 1075 | Titans |
| 11 | 11 | Joe Staley | 89.47 | 84.70 | 88.48 | 1114 | 49ers |
| 12 | 12 | Nate Solder | 89.15 | 84.90 | 87.81 | 1219 | Patriots |
| 13 | 13 | Phil Loadholt | 88.78 | 81.79 | 89.28 | 954 | Vikings |
| 14 | 14 | Eugene Monroe | 88.47 | 84.10 | 87.21 | 1031 | Ravens |
| 15 | 15 | Cordy Glenn | 88.21 | 82.40 | 87.91 | 1161 | Bills |
| 16 | 16 | Chris Clark | 87.99 | 82.30 | 87.61 | 1274 | Broncos |
| 17 | 17 | Andre Smith | 86.84 | 80.00 | 87.24 | 1179 | Bengals |
| 18 | 18 | Doug Free | 86.79 | 79.30 | 87.61 | 999 | Cowboys |
| 19 | 19 | Demar Dotson | 86.71 | 80.80 | 86.48 | 1037 | Buccaneers |
| 20 | 20 | Sebastian Vollmer | 85.91 | 77.97 | 87.04 | 502 | Patriots |
| 21 | 21 | Branden Albert | 84.76 | 76.51 | 86.09 | 858 | Chiefs |
| 22 | 22 | Zach Strief | 84.72 | 79.00 | 84.36 | 1183 | Saints |
| 23 | 23 | Donald Penn | 84.53 | 78.70 | 84.25 | 1036 | Buccaneers |
| 24 | 24 | Tyler Polumbus | 84.40 | 78.10 | 84.44 | 1145 | Commanders |
| 25 | 25 | Lane Johnson | 83.82 | 74.90 | 85.60 | 1162 | Eagles |
| 26 | 26 | Anthony Castonzo | 83.82 | 77.10 | 84.13 | 1197 | Colts |
| 27 | 27 | Michael Bowie | 83.32 | 72.17 | 86.58 | 584 | Seahawks |
| 28 | 28 | Mitchell Schwartz | 82.93 | 76.10 | 83.32 | 1110 | Browns |
| 29 | 29 | Anthony Davis | 82.29 | 73.60 | 83.91 | 1161 | 49ers |
| 30 | 30 | Jermon Bushrod | 82.03 | 74.30 | 83.01 | 1059 | Bears |
| 31 | 31 | Gosder Cherilus | 81.81 | 74.10 | 82.79 | 1200 | Colts |
| 32 | 32 | Terron Armstead | 81.25 | 68.78 | 85.40 | 286 | Saints |
| 33 | 33 | D'Brickashaw Ferguson | 80.69 | 73.70 | 81.18 | 1052 | Jets |
| 34 | 34 | Matt Kalil | 80.65 | 72.90 | 81.65 | 1041 | Vikings |

### Good (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Riley Reiff | 79.91 | 70.50 | 82.02 | 1107 | Lions |
| 36 | 2 | Austin Howard | 79.57 | 70.80 | 81.25 | 1050 | Jets |
| 37 | 3 | Will Beatty | 79.46 | 70.30 | 81.40 | 1004 | Giants |
| 38 | 4 | J'Marcus Webb | 79.10 | 66.24 | 83.51 | 106 | Vikings |
| 39 | 5 | Anthony Collins | 79.09 | 74.00 | 78.32 | 660 | Bengals |
| 40 | 6 | Joe Barksdale | 78.96 | 70.33 | 80.54 | 823 | Rams |
| 41 | 7 | Michael Oher | 78.36 | 70.10 | 79.70 | 1080 | Ravens |
| 42 | 8 | David Stewart | 78.35 | 69.69 | 79.96 | 800 | Titans |
| 43 | 9 | Marcus Cannon | 78.34 | 67.60 | 81.34 | 703 | Patriots |
| 44 | 10 | Jeremy Trueblood | 78.30 | 67.43 | 81.38 | 628 | Falcons |
| 45 | 11 | Russell Okung | 78.18 | 66.62 | 81.72 | 608 | Seahawks |
| 46 | 12 | Marcus Gilbert | 78.17 | 68.15 | 80.68 | 950 | Steelers |
| 47 | 13 | David Bakhtiari | 78.12 | 69.00 | 80.04 | 1171 | Packers |
| 48 | 14 | Charles Brown | 78.08 | 67.37 | 81.06 | 950 | Saints |
| 49 | 15 | Breno Giacomini | 77.99 | 67.34 | 80.93 | 717 | Seahawks |
| 50 | 16 | Tyson Clabo | 77.98 | 69.10 | 79.73 | 949 | Dolphins |
| 51 | 17 | LaAdrian Waddle | 77.50 | 71.62 | 77.26 | 540 | Lions |
| 52 | 18 | Ryan Clady | 77.27 | 65.63 | 80.86 | 141 | Broncos |
| 53 | 19 | Kelvin Beachum | 77.20 | 67.98 | 79.18 | 828 | Steelers |
| 54 | 20 | Erik Pears | 76.91 | 66.80 | 79.48 | 1161 | Bills |
| 55 | 21 | Eric Winston | 76.39 | 64.60 | 80.08 | 1063 | Cardinals |
| 56 | 22 | Donald Stephenson | 76.16 | 64.50 | 79.77 | 614 | Chiefs |
| 57 | 23 | Austin Pasztor | 75.33 | 64.57 | 78.33 | 791 | Jaguars |
| 58 | 24 | Jared Veldheer | 75.28 | 63.64 | 78.88 | 322 | Raiders |
| 59 | 25 | Corey Hilliard | 75.16 | 63.74 | 78.61 | 455 | Lions |
| 60 | 26 | Byron Bell | 75.06 | 65.10 | 77.54 | 1069 | Panthers |
| 61 | 27 | Ryan Harris | 75.04 | 63.56 | 78.53 | 479 | Texans |
| 62 | 28 | Jonathan Martin | 74.89 | 63.94 | 78.03 | 454 | Dolphins |
| 63 | 29 | Tony Pashos | 74.74 | 64.19 | 77.60 | 712 | Raiders |
| 64 | 30 | Bryce Harris | 74.18 | 62.13 | 78.04 | 241 | Saints |
| 65 | 31 | Jason Fox | 74.04 | 61.54 | 78.21 | 203 | Lions |
| 66 | 32 | Mike Adams | 74.04 | 62.79 | 77.38 | 480 | Steelers |
| 67 | 33 | Khalif Barnes | 74.02 | 63.10 | 77.13 | 1033 | Raiders |

### Starter (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 68 | 1 | Jeff Linkenbach | 73.96 | 61.72 | 77.96 | 398 | Colts |
| 69 | 2 | Don Barclay | 73.94 | 62.00 | 77.73 | 1027 | Packers |
| 70 | 3 | Bryant McKinnie | 73.40 | 63.90 | 75.57 | 1021 | Dolphins |
| 71 | 4 | Derek Newton | 73.40 | 59.72 | 78.36 | 830 | Texans |
| 72 | 5 | Matt McCants | 73.20 | 66.63 | 73.42 | 256 | Raiders |
| 73 | 6 | Lamar Holmes | 72.93 | 60.50 | 77.05 | 1052 | Falcons |
| 74 | 7 | Xavier Nixon | 72.66 | 60.64 | 76.50 | 153 | Colts |
| 75 | 8 | Levi Brown | 72.61 | 58.66 | 77.74 | 262 | Steelers |
| 76 | 9 | Menelik Watson | 71.93 | 58.99 | 76.39 | 173 | Raiders |
| 77 | 10 | Eric Fisher | 71.92 | 58.03 | 77.02 | 789 | Chiefs |
| 78 | 11 | Will Svitek | 71.55 | 59.31 | 75.54 | 239 | Patriots |
| 79 | 12 | Marshall Newhouse | 71.52 | 57.86 | 76.46 | 256 | Packers |
| 80 | 13 | Byron Stingily | 70.07 | 59.04 | 73.25 | 140 | Titans |
| 81 | 14 | Ryan Schraeder | 69.97 | 61.73 | 71.29 | 306 | Falcons |
| 82 | 15 | Jordan Mills | 69.94 | 55.80 | 75.20 | 1012 | Bears |
| 83 | 16 | Cameron Bradfield | 68.07 | 55.24 | 72.45 | 794 | Jaguars |
| 84 | 17 | Sam Baker | 67.14 | 53.37 | 72.15 | 189 | Falcons |
| 85 | 18 | Bradley Sowell | 65.02 | 50.94 | 70.24 | 825 | Cardinals |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 84.27 | 83.98 | 80.30 | 256 | Patriots |
| 2 | 2 | Jimmy Graham | 81.55 | 81.30 | 77.55 | 667 | Saints |
| 3 | 3 | Ladarius Green | 81.15 | 73.36 | 82.17 | 190 | Chargers |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Julius Thomas | 79.68 | 79.60 | 75.56 | 695 | Broncos |
| 5 | 2 | Jordan Reed | 79.67 | 76.97 | 77.30 | 226 | Commanders |
| 6 | 3 | Jason Witten | 79.18 | 78.09 | 75.74 | 647 | Cowboys |
| 7 | 4 | Zach Ertz | 78.92 | 73.41 | 78.43 | 291 | Eagles |
| 8 | 5 | Benjamin Watson | 78.18 | 81.57 | 71.76 | 263 | Saints |
| 9 | 6 | Brent Celek | 78.04 | 76.58 | 74.84 | 426 | Eagles |
| 10 | 7 | Vernon Davis | 77.92 | 71.75 | 77.87 | 534 | 49ers |
| 11 | 8 | Greg Olsen | 77.65 | 78.01 | 73.25 | 550 | Panthers |
| 12 | 9 | Charles Clay | 75.55 | 74.41 | 72.15 | 516 | Dolphins |
| 13 | 10 | Jacob Tamme | 75.04 | 69.00 | 74.90 | 175 | Broncos |
| 14 | 11 | Tony Gonzalez | 74.99 | 75.50 | 70.48 | 707 | Falcons |
| 15 | 12 | Coby Fleener | 74.96 | 67.87 | 75.52 | 639 | Colts |
| 16 | 13 | Jared Cook | 74.58 | 64.15 | 77.36 | 421 | Rams |
| 17 | 14 | Gary Barnidge | 74.51 | 67.92 | 74.74 | 197 | Browns |
| 18 | 15 | Antonio Gates | 74.40 | 68.70 | 74.03 | 664 | Chargers |
| 19 | 16 | Zach Miller | 74.22 | 72.31 | 71.33 | 468 | Seahawks |
| 20 | 17 | Kyle Rudolph | 74.20 | 67.78 | 74.32 | 231 | Vikings |

### Starter (45 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Marcedes Lewis | 73.94 | 65.56 | 75.36 | 360 | Jaguars |
| 22 | 2 | Kellen Winslow | 73.87 | 72.17 | 70.84 | 226 | Jets |
| 23 | 3 | Lance Kendricks | 73.80 | 70.28 | 71.98 | 225 | Rams |
| 24 | 4 | Heath Miller | 73.47 | 68.29 | 72.76 | 561 | Steelers |
| 25 | 5 | Dennis Pitta | 72.77 | 61.54 | 76.09 | 132 | Ravens |
| 26 | 6 | Owen Daniels | 72.61 | 67.87 | 71.60 | 224 | Texans |
| 27 | 7 | Tyler Eifert | 72.40 | 64.69 | 73.38 | 344 | Bengals |
| 28 | 8 | Jordan Cameron | 72.22 | 70.60 | 69.13 | 668 | Browns |
| 29 | 9 | Gavin Escobar | 72.11 | 61.65 | 74.92 | 132 | Cowboys |
| 30 | 10 | Martellus Bennett | 72.09 | 74.65 | 66.21 | 596 | Bears |
| 31 | 11 | Luke Willson | 71.98 | 64.66 | 72.69 | 230 | Seahawks |
| 32 | 12 | John Carlson | 71.63 | 70.78 | 68.03 | 224 | Vikings |
| 33 | 13 | Mychal Rivera | 71.33 | 66.59 | 70.33 | 370 | Raiders |
| 34 | 14 | Jeff Cumberland | 71.32 | 67.19 | 69.90 | 328 | Jets |
| 35 | 15 | Joseph Fauria | 71.21 | 71.44 | 66.89 | 196 | Lions |
| 36 | 16 | Tim Wright | 70.99 | 65.44 | 70.53 | 435 | Buccaneers |
| 37 | 17 | Craig Stevens | 70.79 | 62.60 | 72.08 | 133 | Titans |
| 38 | 18 | Brandon Myers | 70.49 | 66.74 | 68.83 | 552 | Giants |
| 39 | 19 | Fred Davis | 70.34 | 54.07 | 77.02 | 123 | Commanders |
| 40 | 20 | Clay Harbor | 69.99 | 61.86 | 71.25 | 224 | Jaguars |
| 41 | 21 | Brandon Pettigrew | 69.90 | 63.58 | 69.95 | 528 | Lions |
| 42 | 22 | Ed Dickson | 69.81 | 62.07 | 70.81 | 295 | Ravens |
| 43 | 23 | Jeron Mastrud | 69.80 | 59.94 | 72.21 | 228 | Raiders |
| 44 | 24 | Jermichael Finley | 69.65 | 66.57 | 67.54 | 166 | Packers |
| 45 | 25 | Anthony Fasano | 69.56 | 62.38 | 70.18 | 372 | Chiefs |
| 46 | 26 | Ryan Griffin | 69.03 | 64.16 | 68.11 | 233 | Texans |
| 47 | 27 | Andrew Quarless | 68.92 | 64.64 | 67.60 | 369 | Packers |
| 48 | 28 | Lee Smith | 68.89 | 64.17 | 67.87 | 142 | Bills |
| 49 | 29 | Garrett Graham | 68.86 | 63.52 | 68.25 | 487 | Texans |
| 50 | 30 | Dante Rosario | 68.79 | 59.64 | 70.73 | 107 | Bears |
| 51 | 31 | Scott Chandler | 68.79 | 64.54 | 67.46 | 545 | Bills |
| 52 | 32 | Jermaine Gresham | 68.64 | 61.61 | 69.16 | 547 | Bengals |
| 53 | 33 | Jim Dray | 68.51 | 64.11 | 67.28 | 290 | Cardinals |
| 54 | 34 | Josh Hill | 68.51 | 59.40 | 70.42 | 105 | Saints |
| 55 | 35 | Delanie Walker | 68.46 | 67.02 | 65.26 | 508 | Titans |
| 56 | 36 | Dallas Clark | 68.29 | 60.71 | 69.17 | 274 | Ravens |
| 57 | 37 | Michael Hoomanawanui | 67.78 | 58.83 | 69.58 | 402 | Patriots |
| 58 | 38 | Virgil Green | 67.56 | 61.72 | 67.28 | 116 | Broncos |
| 59 | 39 | David Paulson | 66.92 | 54.16 | 71.26 | 118 | Steelers |
| 60 | 40 | Matthew Mulligan | 66.87 | 62.82 | 65.40 | 117 | Patriots |
| 61 | 41 | James Hanna | 66.85 | 57.31 | 69.04 | 128 | Cowboys |
| 62 | 42 | Logan Paulsen | 66.37 | 58.70 | 67.31 | 387 | Commanders |
| 63 | 43 | Vance McDonald | 65.77 | 57.17 | 67.34 | 249 | 49ers |
| 64 | 44 | Levine Toilolo | 65.46 | 61.14 | 64.18 | 110 | Falcons |
| 65 | 45 | Rob Housler | 64.13 | 58.61 | 63.65 | 318 | Cardinals |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 66 | 1 | Dion Sims | 60.64 | 57.76 | 58.39 | 127 | Dolphins |
| 67 | 2 | Allen Reisner | 59.81 | 53.67 | 59.73 | 141 | Jaguars |

## WR — Wide Receiver

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Calvin Johnson | 87.98 | 88.56 | 83.42 | 565 | Lions |
| 2 | 2 | Antonio Brown | 86.07 | 88.66 | 80.17 | 627 | Steelers |
| 3 | 3 | Alshon Jeffery | 85.99 | 86.41 | 81.55 | 636 | Bears |
| 4 | 4 | Anquan Boldin | 85.99 | 89.38 | 79.57 | 582 | 49ers |
| 5 | 5 | Andre Johnson | 85.87 | 89.06 | 79.57 | 635 | Texans |
| 6 | 6 | Josh Gordon | 85.77 | 82.21 | 83.98 | 632 | Browns |
| 7 | 7 | Keenan Allen | 85.32 | 85.55 | 81.00 | 600 | Chargers |
| 8 | 8 | Jordy Nelson | 84.77 | 86.00 | 79.79 | 715 | Packers |
| 9 | 9 | DeSean Jackson | 84.69 | 83.63 | 81.23 | 592 | Eagles |
| 10 | 10 | Demaryius Thomas | 84.60 | 84.20 | 80.70 | 820 | Broncos |
| 11 | 11 | Brandon Marshall | 84.21 | 91.28 | 75.33 | 638 | Bears |
| 12 | 12 | T.Y. Hilton | 82.46 | 81.98 | 78.62 | 659 | Colts |
| 13 | 13 | Marques Colston | 82.31 | 85.00 | 76.35 | 634 | Saints |
| 14 | 14 | Vincent Jackson | 82.29 | 78.48 | 80.67 | 591 | Buccaneers |
| 15 | 15 | A.J. Green | 82.17 | 81.40 | 78.51 | 703 | Bengals |
| 16 | 16 | Doug Baldwin | 82.12 | 81.22 | 78.56 | 524 | Seahawks |
| 17 | 17 | Dez Bryant | 81.84 | 82.07 | 77.52 | 647 | Cowboys |
| 18 | 18 | Pierre Garcon | 81.30 | 81.72 | 76.85 | 632 | Commanders |
| 19 | 19 | Marvin Jones Jr. | 81.17 | 82.31 | 76.24 | 444 | Bengals |
| 20 | 20 | Golden Tate | 81.16 | 78.87 | 78.52 | 551 | Seahawks |
| 21 | 21 | Julio Jones | 80.87 | 75.56 | 80.24 | 222 | Falcons |
| 22 | 22 | Michael Floyd | 80.83 | 81.11 | 76.47 | 602 | Cardinals |
| 23 | 23 | Larry Fitzgerald | 80.01 | 80.04 | 75.83 | 632 | Cardinals |

### Good (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Victor Cruz | 79.63 | 75.80 | 78.02 | 525 | Giants |
| 25 | 2 | Nate Washington | 79.35 | 77.26 | 76.58 | 584 | Titans |
| 26 | 3 | Steve Smith | 79.34 | 79.43 | 75.11 | 501 | Panthers |
| 27 | 4 | Brian Hartline | 78.86 | 77.70 | 75.47 | 636 | Dolphins |
| 28 | 5 | Kendall Wright | 78.45 | 79.84 | 73.36 | 562 | Titans |
| 29 | 6 | Reggie Wayne | 78.28 | 76.52 | 75.28 | 274 | Colts |
| 30 | 7 | Torrey Smith | 78.24 | 70.50 | 79.24 | 707 | Ravens |
| 31 | 8 | Eric Decker | 78.03 | 77.20 | 74.41 | 810 | Broncos |
| 32 | 9 | Ted Ginn Jr. | 77.90 | 71.58 | 77.94 | 388 | Panthers |
| 33 | 10 | Andre Holmes | 77.81 | 69.12 | 79.43 | 261 | Raiders |
| 34 | 11 | Jerrel Jernigan | 77.78 | 72.68 | 77.01 | 151 | Giants |
| 35 | 12 | Rod Streater | 77.54 | 72.74 | 76.57 | 517 | Raiders |
| 36 | 13 | Michael Crabtree | 77.48 | 69.69 | 78.51 | 242 | 49ers |
| 37 | 14 | Randall Cobb | 77.40 | 73.87 | 75.58 | 235 | Packers |
| 38 | 15 | Hakeem Nicks | 77.22 | 69.57 | 78.16 | 550 | Giants |
| 39 | 16 | Julian Edelman | 77.01 | 81.20 | 70.05 | 702 | Patriots |
| 40 | 17 | Denarius Moore | 76.68 | 70.50 | 76.63 | 388 | Raiders |
| 41 | 18 | Dwayne Bowe | 76.39 | 74.58 | 73.43 | 628 | Chiefs |
| 42 | 19 | Riley Cooper | 76.38 | 71.19 | 75.67 | 594 | Eagles |
| 43 | 20 | Jeremy Kerley | 76.04 | 69.53 | 76.21 | 334 | Jets |
| 44 | 21 | Greg Jennings | 75.91 | 72.78 | 73.83 | 508 | Vikings |
| 45 | 22 | DeAndre Hopkins | 75.73 | 68.65 | 76.28 | 652 | Texans |
| 46 | 23 | Wes Welker | 75.70 | 74.12 | 72.59 | 632 | Broncos |
| 47 | 24 | Jarius Wright | 75.55 | 67.61 | 76.67 | 322 | Vikings |
| 48 | 25 | Cecil Shorts | 75.54 | 71.05 | 74.36 | 497 | Jaguars |
| 49 | 26 | Eddie Royal | 75.51 | 72.73 | 73.19 | 509 | Chargers |
| 50 | 27 | Robert Meachem | 75.42 | 67.67 | 76.42 | 204 | Saints |
| 51 | 28 | Jarrett Boykin | 75.33 | 70.86 | 74.15 | 454 | Packers |
| 52 | 29 | Jermaine Kearse | 75.22 | 69.65 | 74.77 | 305 | Seahawks |
| 53 | 30 | Jerome Simpson | 75.08 | 71.31 | 73.42 | 463 | Vikings |
| 54 | 31 | Steve Johnson | 74.97 | 71.62 | 73.03 | 411 | Bills |
| 55 | 32 | Kenny Stills | 74.95 | 61.98 | 79.43 | 588 | Saints |
| 56 | 33 | Terrance Williams | 74.87 | 65.51 | 76.95 | 521 | Cowboys |
| 57 | 34 | Jerricho Cotchery | 74.86 | 72.36 | 72.36 | 454 | Steelers |
| 58 | 35 | Mike Wallace | 74.66 | 69.89 | 73.67 | 658 | Dolphins |
| 59 | 36 | James Jones | 74.64 | 67.63 | 75.14 | 588 | Packers |
| 60 | 37 | Rueben Randle | 74.59 | 67.50 | 75.15 | 439 | Giants |
| 61 | 38 | Cordarrelle Patterson | 74.36 | 73.55 | 70.73 | 291 | Vikings |
| 62 | 39 | Danny Amendola | 74.36 | 71.43 | 72.14 | 458 | Patriots |
| 63 | 40 | Aldrick Robinson | 74.09 | 64.44 | 76.36 | 273 | Commanders |
| 64 | 41 | Sidney Rice | 74.03 | 65.93 | 75.27 | 213 | Seahawks |
| 65 | 42 | Jacoby Jones | 74.01 | 67.03 | 74.50 | 377 | Ravens |

### Starter (72 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 66 | 1 | Emmanuel Sanders | 73.88 | 70.37 | 72.06 | 556 | Steelers |
| 67 | 2 | Justin Hunter | 73.86 | 64.90 | 75.67 | 219 | Titans |
| 68 | 3 | Justin Blackmon | 73.73 | 65.91 | 74.77 | 163 | Jaguars |
| 69 | 4 | Lance Moore | 73.63 | 68.17 | 73.10 | 403 | Saints |
| 70 | 5 | Brandon Gibson | 73.52 | 74.04 | 69.00 | 201 | Dolphins |
| 71 | 6 | Cole Beasley | 73.45 | 69.94 | 71.63 | 208 | Cowboys |
| 72 | 7 | Stedman Bailey | 73.44 | 64.72 | 75.09 | 124 | Rams |
| 73 | 8 | Harry Douglas | 73.28 | 69.80 | 71.43 | 674 | Falcons |
| 74 | 9 | Santonio Holmes | 73.23 | 66.13 | 73.79 | 304 | Jets |
| 75 | 10 | Chris Givens | 73.12 | 60.84 | 77.14 | 465 | Rams |
| 76 | 11 | Tiquan Underwood | 72.70 | 64.39 | 74.08 | 378 | Buccaneers |
| 77 | 12 | Leonard Hankerson | 72.58 | 65.29 | 73.28 | 275 | Commanders |
| 78 | 13 | Griff Whalen | 72.54 | 68.49 | 71.08 | 277 | Colts |
| 79 | 14 | Brandon LaFell | 72.50 | 65.82 | 72.79 | 582 | Panthers |
| 80 | 15 | Roddy White | 72.43 | 65.33 | 73.00 | 559 | Falcons |
| 81 | 16 | Jaron Brown | 72.21 | 63.31 | 73.98 | 119 | Cardinals |
| 82 | 17 | LaVon Brazill | 72.21 | 62.95 | 74.22 | 260 | Colts |
| 83 | 18 | Brandon Stokley | 72.00 | 65.76 | 71.99 | 106 | Ravens |
| 84 | 19 | Travis Benjamin | 72.00 | 59.64 | 76.07 | 107 | Browns |
| 85 | 20 | Rishard Matthews | 71.98 | 66.36 | 71.56 | 378 | Dolphins |
| 86 | 21 | Andrew Hawkins | 71.67 | 63.49 | 72.96 | 129 | Bengals |
| 87 | 22 | David Nelson | 71.63 | 67.74 | 70.06 | 304 | Jets |
| 88 | 23 | Aaron Dobson | 71.20 | 63.01 | 72.49 | 356 | Patriots |
| 89 | 24 | Kenbrell Thompkins | 71.14 | 63.37 | 72.15 | 387 | Patriots |
| 90 | 25 | Robert Woods | 71.12 | 63.45 | 72.06 | 516 | Bills |
| 91 | 26 | Brian Quick | 71.05 | 61.43 | 73.30 | 234 | Rams |
| 92 | 27 | Da'Rick Rogers | 71.04 | 59.19 | 74.77 | 171 | Colts |
| 93 | 28 | Tandon Doss | 70.67 | 61.62 | 72.54 | 222 | Ravens |
| 94 | 29 | Mike Williams | 70.65 | 61.68 | 72.47 | 240 | Buccaneers |
| 95 | 30 | Tavon Austin | 70.44 | 66.94 | 68.61 | 318 | Rams |
| 96 | 31 | Damian Williams | 70.11 | 64.54 | 69.66 | 136 | Titans |
| 97 | 32 | Jacoby Ford | 69.75 | 57.89 | 73.49 | 159 | Raiders |
| 98 | 33 | A.J. Jenkins | 69.71 | 59.58 | 72.30 | 143 | Chiefs |
| 99 | 34 | Vincent Brown | 69.67 | 63.37 | 69.71 | 580 | Chargers |
| 100 | 35 | Kevin Ogletree | 69.36 | 62.02 | 70.09 | 232 | Lions |
| 101 | 36 | Miles Austin | 69.31 | 58.75 | 72.18 | 345 | Cowboys |
| 102 | 37 | Mario Manningham | 69.27 | 60.84 | 70.72 | 108 | 49ers |
| 103 | 38 | Andre Roberts | 69.15 | 65.00 | 67.75 | 430 | Cardinals |
| 104 | 39 | Jason Avant | 68.98 | 61.12 | 70.05 | 521 | Eagles |
| 105 | 40 | Nate Burleson | 68.92 | 63.40 | 68.44 | 305 | Lions |
| 106 | 41 | Austin Pettis | 68.51 | 65.21 | 66.54 | 399 | Rams |
| 107 | 42 | Marlon Brown | 68.43 | 65.17 | 66.44 | 542 | Ravens |
| 108 | 43 | Donnie Avery | 68.39 | 58.75 | 70.65 | 525 | Chiefs |
| 109 | 44 | Mike Brown | 68.27 | 57.65 | 71.18 | 378 | Jaguars |
| 110 | 45 | Earl Bennett | 67.90 | 62.14 | 67.58 | 385 | Bears |
| 111 | 46 | Ace Sanders | 67.58 | 61.54 | 67.44 | 436 | Jaguars |
| 112 | 47 | Stephen Hill | 67.55 | 60.44 | 68.13 | 349 | Jets |
| 113 | 48 | DeVier Posey | 67.43 | 59.59 | 68.49 | 172 | Texans |
| 114 | 49 | Deonte Thompson | 67.42 | 60.52 | 67.85 | 104 | Ravens |
| 115 | 50 | Josh Morgan | 67.35 | 61.99 | 66.75 | 190 | Commanders |
| 116 | 51 | Kerry Taylor | 67.29 | 62.77 | 66.14 | 202 | Jaguars |
| 117 | 52 | Andre Caldwell | 67.17 | 64.65 | 64.68 | 176 | Broncos |
| 118 | 53 | Josh Boyce | 67.17 | 60.35 | 67.55 | 128 | Patriots |
| 119 | 54 | Junior Hemingway | 66.92 | 58.04 | 68.67 | 167 | Chiefs |
| 120 | 55 | Mohamed Sanu | 66.85 | 62.62 | 65.51 | 505 | Bengals |
| 121 | 56 | Santana Moss | 66.72 | 58.28 | 68.18 | 468 | Commanders |
| 122 | 57 | Darrius Heyward-Bey | 66.43 | 54.76 | 70.05 | 410 | Colts |
| 123 | 58 | Kyle Williams | 66.38 | 56.12 | 69.06 | 197 | Chiefs |
| 124 | 59 | Ryan Broyles | 66.33 | 55.83 | 69.17 | 115 | Lions |
| 125 | 60 | T.J. Graham | 66.31 | 57.49 | 68.02 | 496 | Bills |
| 126 | 61 | Chris Hogan | 65.46 | 59.08 | 65.55 | 126 | Bills |
| 127 | 62 | Chris Owusu | 64.79 | 56.39 | 66.22 | 204 | Buccaneers |
| 128 | 63 | Kris Durham | 64.03 | 55.67 | 65.43 | 586 | Lions |
| 129 | 64 | Nick Toon | 63.98 | 56.23 | 64.98 | 136 | Saints |
| 130 | 65 | Keshawn Martin | 63.75 | 57.52 | 63.74 | 296 | Texans |
| 131 | 66 | Jeremy Ross | 63.64 | 60.68 | 61.45 | 101 | Lions |
| 132 | 67 | Greg Little | 63.14 | 53.60 | 65.34 | 691 | Browns |
| 133 | 68 | Davone Bess | 63.03 | 54.64 | 64.45 | 447 | Browns |
| 134 | 69 | Markus Wheaton | 62.93 | 58.01 | 62.05 | 109 | Steelers |
| 135 | 70 | Kenny Britt | 62.62 | 50.93 | 66.24 | 215 | Titans |
| 136 | 71 | Darius Johnson | 62.44 | 56.72 | 62.09 | 308 | Falcons |
| 137 | 72 | Brice Butler | 62.31 | 58.45 | 60.72 | 145 | Raiders |

### Rotation/backup (0 players)

_None._
