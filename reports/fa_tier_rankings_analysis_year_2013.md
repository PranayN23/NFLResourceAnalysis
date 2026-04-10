# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:27:07Z
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
| 10 | 10 | Mike Pouncey | 86.14 | 78.80 | 86.87 | 901 | Dolphins |
| 11 | 11 | Ryan Kalil | 85.57 | 77.60 | 86.71 | 1071 | Panthers |
| 12 | 12 | Brian De La Puente | 85.13 | 78.00 | 85.72 | 1273 | Saints |
| 13 | 13 | Nick Hardwick | 84.30 | 76.60 | 85.26 | 1135 | Chargers |
| 14 | 14 | Roberto Garza | 83.69 | 75.80 | 84.79 | 1059 | Bears |
| 15 | 15 | Rodney Hudson | 83.24 | 75.10 | 84.50 | 1089 | Chiefs |
| 16 | 16 | Kyle Cook | 83.02 | 74.80 | 84.33 | 1131 | Bengals |
| 17 | 17 | Nick Mangold | 82.16 | 73.20 | 83.96 | 1050 | Jets |
| 18 | 18 | Scott Wells | 82.08 | 72.10 | 84.56 | 739 | Rams |
| 19 | 19 | Eric Wood | 81.70 | 72.20 | 83.86 | 1161 | Bills |
| 20 | 20 | Max Unger | 81.36 | 72.00 | 83.43 | 930 | Seahawks |
| 21 | 21 | Lyle Sendlein | 81.00 | 71.20 | 83.36 | 1084 | Cardinals |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Ryan Wendell | 79.88 | 71.10 | 81.56 | 1330 | Patriots |
| 23 | 2 | Jonathan Goodwin | 79.81 | 70.60 | 81.78 | 1156 | 49ers |
| 24 | 3 | Gino Gradkowski | 79.21 | 69.20 | 81.72 | 1136 | Ravens |
| 25 | 4 | Brian Schwenke | 78.42 | 68.80 | 80.66 | 567 | Titans |
| 26 | 5 | Tim Barnes | 78.35 | 69.30 | 80.21 | 266 | Rams |
| 27 | 6 | Rich Ohrnberger | 78.11 | 70.00 | 79.35 | 201 | Chargers |
| 28 | 7 | David Baas | 77.45 | 68.30 | 79.38 | 143 | Giants |
| 29 | 8 | Brad Meester | 76.43 | 67.10 | 78.48 | 1058 | Jaguars |
| 30 | 9 | Joe Hawley | 76.41 | 70.80 | 75.98 | 539 | Falcons |
| 31 | 10 | Jim Cordle | 76.13 | 66.00 | 78.71 | 482 | Giants |
| 32 | 11 | Samson Satele | 75.54 | 66.60 | 77.34 | 953 | Colts |

### Starter (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Robert Turner | 70.70 | 59.70 | 73.87 | 394 | Titans |
| 34 | 2 | Peter Konz | 70.11 | 59.60 | 72.95 | 889 | Falcons |
| 35 | 3 | Andre Gurode | 69.55 | 53.20 | 76.29 | 275 | Raiders |
| 36 | 4 | Cody Wallace | 67.58 | 57.50 | 70.13 | 288 | Steelers |
| 37 | 5 | Lemuel Jeanpierre | 67.51 | 60.90 | 67.75 | 283 | Seahawks |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Brandon Boykin | 94.27 | 91.10 | 92.22 | 657 | Eagles |
| 2 | 2 | Richard Sherman | 93.89 | 90.10 | 92.67 | 1166 | Seahawks |
| 3 | 3 | Brent Grimes | 88.93 | 89.40 | 89.97 | 1110 | Dolphins |
| 4 | 4 | Desmond Trufant | 88.91 | 82.50 | 89.01 | 1001 | Falcons |
| 5 | 5 | Dimitri Patterson | 88.20 | 89.90 | 88.54 | 237 | Dolphins |
| 6 | 6 | Dominique Rodgers-Cromartie | 86.51 | 82.60 | 85.59 | 935 | Broncos |
| 7 | 7 | Byron Maxwell | 86.31 | 84.90 | 89.95 | 677 | Seahawks |
| 8 | 8 | Darrelle Revis | 86.09 | 83.20 | 88.22 | 949 | Buccaneers |
| 9 | 9 | Chris Harris Jr. | 85.71 | 82.70 | 83.97 | 1066 | Broncos |
| 10 | 10 | Tramaine Brock Sr. | 85.43 | 85.30 | 86.86 | 840 | 49ers |
| 11 | 11 | Leon Hall | 84.71 | 86.20 | 87.05 | 274 | Bengals |
| 12 | 12 | Alterraun Verner | 84.36 | 79.60 | 83.36 | 1002 | Titans |
| 13 | 13 | Tramon Williams | 83.07 | 77.40 | 82.69 | 1101 | Packers |
| 14 | 14 | Jarrett Bush | 82.74 | 80.70 | 84.20 | 125 | Packers |
| 15 | 15 | Joe Haden | 82.69 | 77.60 | 84.20 | 1052 | Browns |
| 16 | 16 | Keenan Lewis | 82.66 | 76.40 | 82.66 | 989 | Saints |
| 17 | 17 | Patrick Peterson | 82.54 | 80.00 | 80.07 | 1075 | Cardinals |
| 18 | 18 | Vontae Davis | 81.88 | 77.60 | 82.97 | 1081 | Colts |
| 19 | 19 | Captain Munnerlyn | 81.55 | 75.90 | 81.56 | 1035 | Panthers |
| 20 | 20 | Jason McCourty | 81.15 | 75.00 | 81.28 | 1056 | Titans |
| 21 | 21 | Nickell Robey-Coleman | 80.99 | 77.00 | 79.48 | 608 | Bills |
| 22 | 22 | Orlando Scandrick | 80.89 | 80.50 | 79.16 | 1089 | Cowboys |
| 23 | 23 | Adam Jones | 80.65 | 78.30 | 79.72 | 1035 | Bengals |
| 24 | 24 | Leodis McKelvin | 80.39 | 76.40 | 82.44 | 927 | Bills |
| 25 | 25 | Alan Ball | 80.34 | 75.60 | 82.35 | 1005 | Jaguars |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Drayton Florence | 79.37 | 75.30 | 80.94 | 621 | Panthers |
| 27 | 2 | Johnathan Joseph | 79.28 | 73.80 | 79.28 | 833 | Texans |
| 28 | 3 | Trumaine McBride | 78.97 | 77.40 | 81.45 | 607 | Giants |
| 29 | 4 | Sam Shields | 78.76 | 71.80 | 81.00 | 881 | Packers |
| 30 | 5 | William Gay | 78.11 | 73.70 | 76.89 | 912 | Steelers |
| 31 | 6 | Lardarius Webb | 77.83 | 74.20 | 79.22 | 972 | Ravens |
| 32 | 7 | Jeremy Lane | 77.69 | 77.90 | 79.36 | 201 | Seahawks |
| 33 | 8 | Logan Ryan | 77.40 | 67.20 | 80.04 | 700 | Patriots |
| 34 | 9 | DeAngelo Hall | 77.39 | 71.00 | 77.49 | 998 | Commanders |
| 35 | 10 | Brandon Browner | 77.35 | 69.00 | 83.53 | 455 | Seahawks |
| 36 | 11 | Mike Harris | 76.51 | 70.40 | 77.59 | 404 | Jaguars |
| 37 | 12 | Rashean Mathis | 76.25 | 72.70 | 77.69 | 771 | Lions |
| 38 | 13 | Tarell Brown | 76.25 | 68.10 | 77.52 | 931 | 49ers |
| 39 | 14 | Cortez Allen | 76.05 | 70.40 | 78.46 | 704 | Steelers |
| 40 | 15 | Tim Jennings | 75.86 | 70.00 | 76.23 | 1032 | Bears |
| 41 | 16 | Terence Newman | 75.70 | 71.60 | 76.25 | 819 | Bengals |
| 42 | 17 | Terrell Thomas | 75.64 | 69.50 | 75.57 | 569 | Giants |
| 43 | 18 | Robert McClain | 74.97 | 67.30 | 75.91 | 577 | Falcons |
| 44 | 19 | Coty Sensabaugh | 74.81 | 71.60 | 74.86 | 494 | Titans |
| 45 | 20 | Trumaine Johnson | 74.79 | 66.70 | 76.40 | 873 | Rams |
| 46 | 21 | Kyle Arrington | 74.69 | 66.40 | 76.05 | 916 | Patriots |
| 47 | 22 | Brandon Carr | 74.55 | 66.70 | 75.62 | 1117 | Cowboys |

### Starter (53 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 48 | 1 | Robert Alford | 73.95 | 66.60 | 75.72 | 570 | Falcons |
| 49 | 2 | Bradley Fletcher | 73.88 | 67.70 | 77.39 | 997 | Eagles |
| 50 | 3 | Eric Wright | 73.66 | 69.40 | 79.41 | 118 | 49ers |
| 51 | 4 | Jimmy Smith | 73.59 | 66.80 | 75.94 | 1041 | Ravens |
| 52 | 5 | Sean Smith | 73.57 | 65.80 | 74.58 | 1074 | Chiefs |
| 53 | 6 | Janoris Jenkins | 73.57 | 65.20 | 75.37 | 1045 | Rams |
| 54 | 7 | Xavier Rhodes | 73.17 | 65.40 | 77.31 | 675 | Vikings |
| 55 | 8 | Zackary Bowman | 72.79 | 68.70 | 76.45 | 593 | Bears |
| 56 | 9 | Cary Williams | 72.72 | 64.20 | 74.24 | 1211 | Eagles |
| 57 | 10 | Chris Owens | 72.67 | 68.40 | 75.62 | 539 | Dolphins |
| 58 | 11 | Prince Amukamara | 72.58 | 65.80 | 75.34 | 1080 | Giants |
| 59 | 12 | Corey White | 72.58 | 68.90 | 73.60 | 670 | Saints |
| 60 | 13 | Josh Wilson | 72.50 | 63.80 | 74.13 | 968 | Commanders |
| 61 | 14 | Asante Samuel | 72.39 | 63.20 | 77.36 | 499 | Falcons |
| 62 | 15 | Aaron Ross | 72.31 | 68.80 | 77.88 | 155 | Giants |
| 63 | 16 | Champ Bailey | 72.18 | 66.20 | 76.37 | 326 | Broncos |
| 64 | 17 | Alfonzo Dennard | 71.86 | 64.60 | 74.75 | 838 | Patriots |
| 65 | 18 | Aqib Talib | 71.54 | 62.50 | 75.80 | 927 | Patriots |
| 66 | 19 | Antoine Cason | 71.16 | 62.40 | 75.97 | 165 | Cardinals |
| 67 | 20 | Josh Gordy | 71.02 | 63.90 | 74.94 | 247 | Colts |
| 68 | 21 | Melvin White | 70.53 | 64.70 | 71.28 | 726 | Panthers |
| 69 | 22 | Dwayne Gratz | 70.34 | 66.30 | 75.11 | 485 | Jaguars |
| 70 | 23 | Mike Jenkins | 69.90 | 60.60 | 74.21 | 903 | Raiders |
| 71 | 24 | Brandon Flowers | 69.78 | 58.90 | 74.22 | 907 | Chiefs |
| 72 | 25 | Chris Carr | 69.63 | 69.50 | 74.82 | 146 | Saints |
| 73 | 26 | Jabari Greer | 69.53 | 59.00 | 76.13 | 541 | Saints |
| 74 | 27 | Carlos Rogers | 69.32 | 60.50 | 71.03 | 1068 | 49ers |
| 75 | 28 | Brandon Harris | 69.21 | 61.10 | 77.43 | 206 | Texans |
| 76 | 29 | Kyle Wilson | 68.85 | 60.70 | 70.11 | 465 | Jets |
| 77 | 30 | Jerraud Powers | 68.73 | 60.80 | 73.18 | 1031 | Cardinals |
| 78 | 31 | Kareem Jackson | 68.43 | 57.60 | 72.51 | 760 | Texans |
| 79 | 32 | David Amerson | 68.37 | 57.70 | 71.31 | 685 | Commanders |
| 80 | 33 | Corey Webster | 68.12 | 62.60 | 73.88 | 165 | Giants |
| 81 | 34 | Marcus Cooper | 68.11 | 53.30 | 73.81 | 708 | Chiefs |
| 82 | 35 | Nolan Carroll | 67.52 | 56.60 | 72.81 | 794 | Dolphins |
| 83 | 36 | Javier Arenas | 67.07 | 58.00 | 72.80 | 102 | Cardinals |
| 84 | 37 | Kayvon Webster | 67.05 | 56.10 | 70.18 | 492 | Broncos |
| 85 | 38 | Darrin Walls | 66.51 | 63.30 | 74.58 | 289 | Jets |
| 86 | 39 | Davon House | 65.90 | 54.20 | 72.04 | 526 | Packers |
| 87 | 40 | Stephon Gilmore | 65.86 | 56.30 | 71.31 | 647 | Bills |
| 88 | 41 | Dre Kirkpatrick | 65.82 | 61.90 | 70.65 | 356 | Bengals |
| 89 | 42 | Josh Thomas | 65.19 | 58.90 | 72.92 | 273 | Panthers |
| 90 | 43 | Dee Milliner | 64.64 | 53.90 | 70.76 | 722 | Jets |
| 91 | 44 | Buster Skrine | 64.60 | 53.40 | 68.93 | 1052 | Browns |
| 92 | 45 | Will Blackmon | 64.58 | 65.00 | 67.95 | 670 | Jaguars |
| 93 | 46 | Morris Claiborne | 64.19 | 56.40 | 69.51 | 508 | Cowboys |
| 94 | 47 | Leonard Johnson | 64.01 | 53.60 | 68.35 | 693 | Buccaneers |
| 95 | 48 | Cassius Vaughn | 64.00 | 57.10 | 68.80 | 410 | Colts |
| 96 | 49 | Chris Cook | 63.64 | 58.90 | 68.36 | 736 | Vikings |
| 97 | 50 | Charles Tillman | 63.29 | 53.20 | 70.02 | 432 | Bears |
| 98 | 51 | Johnthan Banks | 62.40 | 53.20 | 64.37 | 941 | Buccaneers |
| 99 | 52 | Antonio Cromartie | 62.32 | 46.30 | 68.83 | 1067 | Jets |
| 100 | 53 | Greg Toler | 62.09 | 53.50 | 69.07 | 458 | Colts |

### Rotation/backup (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 101 | 1 | Darius Slay | 61.20 | 53.60 | 67.30 | 338 | Lions |
| 102 | 2 | Tracy Porter | 61.09 | 50.80 | 66.92 | 987 | Raiders |
| 103 | 3 | Roc Carmichael | 61.01 | 54.00 | 70.89 | 220 | Eagles |
| 104 | 4 | Richard Marshall | 60.84 | 50.20 | 67.52 | 776 | Chargers |
| 105 | 5 | Jayron Hosley | 59.78 | 55.80 | 66.33 | 112 | Giants |
| 106 | 6 | Chris Houston | 59.61 | 48.40 | 65.83 | 729 | Lions |
| 107 | 7 | B.W. Webb | 59.31 | 57.50 | 61.55 | 179 | Cowboys |
| 108 | 8 | Jonte Green | 59.28 | 53.50 | 69.00 | 149 | Lions |
| 109 | 9 | Marcus Sherels | 59.15 | 52.60 | 64.76 | 530 | Vikings |
| 110 | 10 | Shaun Prater | 58.43 | 64.20 | 68.91 | 159 | Vikings |
| 111 | 11 | Dunta Robinson | 57.32 | 46.70 | 63.89 | 282 | Chiefs |
| 112 | 12 | Cortland Finnegan | 57.19 | 46.50 | 64.83 | 362 | Rams |
| 113 | 13 | Johnny Patrick | 56.87 | 48.90 | 62.18 | 475 | Chargers |
| 114 | 14 | Shareece Wright | 56.52 | 45.40 | 65.60 | 935 | Chargers |
| 115 | 15 | Derek Cox | 56.47 | 43.10 | 66.12 | 556 | Chargers |
| 116 | 16 | Tony Carter | 56.37 | 39.10 | 66.71 | 330 | Broncos |
| 117 | 17 | D.J. Hayden | 56.19 | 53.70 | 62.01 | 338 | Raiders |
| 118 | 18 | Phillip Adams | 55.92 | 41.20 | 63.65 | 340 | Raiders |
| 119 | 19 | Leon McFadden | 53.31 | 42.70 | 63.52 | 244 | Browns |
| 120 | 20 | Brice McCain | 51.81 | 29.40 | 63.84 | 605 | Texans |
| 121 | 21 | Josh Norman | 50.44 | 49.30 | 56.66 | 102 | Panthers |
| 122 | 22 | Chimdi Chekwa | 45.71 | 35.00 | 61.18 | 164 | Raiders |

## DI — Defensive Interior

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | J.J. Watt | 96.98 | 90.32 | 97.26 | 960 | Texans |
| 2 | 2 | Calais Campbell | 87.95 | 86.91 | 85.41 | 963 | Cardinals |
| 3 | 3 | Geno Atkins | 87.36 | 87.30 | 86.89 | 455 | Bengals |
| 4 | 4 | Marcell Dareus | 85.55 | 88.07 | 79.71 | 823 | Bills |
| 5 | 5 | Ndamukong Suh | 85.33 | 88.30 | 79.38 | 878 | Lions |
| 6 | 6 | Muhammad Wilkerson | 84.75 | 82.83 | 81.86 | 1041 | Jets |
| 7 | 7 | Kyle Williams | 84.35 | 82.94 | 83.41 | 940 | Bills |
| 8 | 8 | Star Lotulelei | 84.23 | 77.93 | 84.27 | 653 | Panthers |
| 9 | 9 | Sheldon Richardson | 83.94 | 81.13 | 81.65 | 882 | Jets |
| 10 | 10 | Jurrell Casey | 83.15 | 83.02 | 79.59 | 874 | Titans |
| 11 | 11 | Gerald McCoy | 81.55 | 89.83 | 73.95 | 963 | Buccaneers |
| 12 | 12 | Nick Fairley | 81.08 | 75.41 | 83.20 | 670 | Lions |
| 13 | 13 | Kawann Short | 80.49 | 75.75 | 79.48 | 551 | Panthers |
| 14 | 14 | Steve McLendon | 80.34 | 77.65 | 79.21 | 350 | Steelers |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Damon Harrison Sr. | 79.09 | 74.03 | 82.59 | 500 | Jets |
| 16 | 2 | Mike Martin | 78.80 | 67.45 | 84.15 | 233 | Titans |
| 17 | 3 | Jason Hatcher | 78.40 | 70.92 | 80.37 | 748 | Cowboys |
| 18 | 4 | Brandon Mebane | 78.23 | 75.76 | 75.71 | 623 | Seahawks |
| 19 | 5 | Mike Daniels | 78.21 | 67.00 | 81.52 | 542 | Packers |
| 20 | 6 | Linval Joseph | 78.07 | 74.03 | 77.11 | 578 | Giants |
| 21 | 7 | Dan Williams | 78.04 | 82.60 | 73.44 | 286 | Cardinals |
| 22 | 8 | Haloti Ngata | 77.59 | 76.76 | 74.49 | 700 | Ravens |
| 23 | 9 | Akiem Hicks | 76.63 | 66.34 | 80.10 | 741 | Saints |
| 24 | 10 | Kenrick Ellis | 76.40 | 72.89 | 78.12 | 208 | Jets |
| 25 | 11 | Karl Klug | 76.28 | 74.78 | 73.11 | 319 | Titans |
| 26 | 12 | Terrance Knighton | 76.26 | 77.30 | 72.04 | 698 | Broncos |
| 27 | 13 | Randy Starks | 76.09 | 73.81 | 73.44 | 729 | Dolphins |
| 28 | 14 | Malik Jackson | 76.05 | 65.00 | 80.03 | 713 | Broncos |
| 29 | 15 | Arthur Jones | 75.43 | 66.16 | 78.48 | 521 | Ravens |
| 30 | 16 | Leger Douzable | 75.33 | 65.97 | 77.61 | 237 | Jets |
| 31 | 17 | Dontari Poe | 75.19 | 74.67 | 71.37 | 1036 | Chiefs |
| 32 | 18 | Tyson Jackson | 75.02 | 72.44 | 72.89 | 510 | Chiefs |
| 33 | 19 | Paul Soliai | 74.95 | 66.95 | 76.64 | 522 | Dolphins |
| 34 | 20 | Johnathan Hankins | 74.74 | 74.73 | 75.78 | 192 | Giants |
| 35 | 21 | Glenn Dorsey | 74.59 | 77.34 | 72.56 | 521 | 49ers |
| 36 | 22 | Fletcher Cox | 74.47 | 71.11 | 72.93 | 948 | Eagles |
| 37 | 23 | Justin Smith | 74.45 | 58.62 | 80.83 | 913 | 49ers |

### Starter (78 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 38 | 1 | Fred Evans | 73.65 | 61.28 | 78.76 | 349 | Vikings |
| 39 | 2 | Brodrick Bunkley | 73.53 | 64.51 | 76.73 | 311 | Saints |
| 40 | 3 | Michael Brockers | 73.40 | 66.21 | 75.20 | 792 | Rams |
| 41 | 4 | Sean Lissemore | 72.77 | 63.09 | 77.45 | 213 | Chargers |
| 42 | 5 | Desmond Bryant | 72.68 | 62.87 | 77.33 | 581 | Browns |
| 43 | 6 | Mike Devito | 72.24 | 65.37 | 74.00 | 447 | Chiefs |
| 44 | 7 | Jared Odrick | 72.20 | 65.23 | 72.68 | 856 | Dolphins |
| 45 | 8 | Sammie Lee Hill | 72.19 | 63.91 | 75.42 | 384 | Titans |
| 46 | 9 | Phil Taylor Sr. | 71.86 | 67.68 | 73.50 | 547 | Browns |
| 47 | 10 | Pat Sims | 71.68 | 63.68 | 76.08 | 677 | Raiders |
| 48 | 11 | Sealver Siliga | 71.56 | 72.80 | 78.28 | 323 | Patriots |
| 49 | 12 | Antonio Smith | 71.50 | 62.43 | 73.90 | 739 | Texans |
| 50 | 13 | Ropati Pitoitua | 71.20 | 56.55 | 77.54 | 576 | Titans |
| 51 | 14 | Lawrence Guy Sr. | 71.02 | 64.43 | 74.90 | 206 | Chargers |
| 52 | 15 | Cameron Heyward | 70.99 | 61.13 | 73.39 | 830 | Steelers |
| 53 | 16 | Earl Mitchell | 70.90 | 59.54 | 74.31 | 535 | Texans |
| 54 | 17 | Ahtyba Rubin | 70.84 | 65.53 | 72.19 | 624 | Browns |
| 55 | 18 | C.J. Mosley | 70.71 | 65.11 | 70.48 | 320 | Lions |
| 56 | 19 | Sylvester Williams | 70.62 | 53.64 | 77.77 | 390 | Broncos |
| 57 | 20 | Cory Redding | 70.29 | 53.82 | 77.42 | 731 | Colts |
| 58 | 21 | Cedric Thornton | 70.00 | 54.10 | 76.44 | 767 | Eagles |
| 59 | 22 | Clinton McDonald | 69.87 | 58.56 | 73.44 | 655 | Seahawks |
| 60 | 23 | John Hughes | 69.79 | 63.83 | 70.24 | 398 | Browns |
| 61 | 24 | Ricky Jean Francois | 69.64 | 59.04 | 74.62 | 500 | Colts |
| 62 | 25 | Bennie Logan | 69.53 | 56.61 | 73.98 | 524 | Eagles |
| 63 | 26 | Billy Winn | 69.47 | 61.43 | 73.92 | 312 | Browns |
| 64 | 27 | Alan Branch | 69.43 | 60.85 | 71.18 | 596 | Bills |
| 65 | 28 | Vance Walker | 69.35 | 56.94 | 73.97 | 767 | Raiders |
| 66 | 29 | Chris Canty | 68.98 | 60.26 | 73.32 | 565 | Ravens |
| 67 | 30 | Tony McDaniel | 68.92 | 55.58 | 76.05 | 610 | Seahawks |
| 68 | 31 | Sharrif Floyd | 68.84 | 61.03 | 69.88 | 461 | Vikings |
| 69 | 32 | Corey Liuget | 68.45 | 56.61 | 72.37 | 822 | Chargers |
| 70 | 33 | Henry Melton | 68.34 | 56.23 | 79.85 | 123 | Bears |
| 71 | 34 | Cullen Jenkins | 68.18 | 49.04 | 76.78 | 698 | Giants |
| 72 | 35 | DeAngelo Tyson | 68.03 | 58.36 | 73.44 | 149 | Ravens |
| 73 | 36 | Cam Thomas | 67.91 | 55.42 | 72.07 | 532 | Chargers |
| 74 | 37 | Ray McDonald | 67.60 | 57.62 | 70.09 | 787 | 49ers |
| 75 | 38 | Red Bryant | 67.60 | 59.06 | 69.12 | 553 | Seahawks |
| 76 | 39 | Datone Jones | 67.34 | 52.44 | 73.10 | 269 | Packers |
| 77 | 40 | John Jenkins | 67.31 | 58.26 | 69.17 | 478 | Saints |
| 78 | 41 | Kevin Williams | 67.18 | 61.03 | 68.04 | 722 | Vikings |
| 79 | 42 | Joe Vellano | 67.02 | 49.91 | 74.26 | 699 | Patriots |
| 80 | 43 | Tyson Alualu | 66.54 | 58.66 | 67.62 | 743 | Jaguars |
| 81 | 44 | Domata Peko Sr. | 66.52 | 50.31 | 73.16 | 691 | Bengals |
| 82 | 45 | Kendall Langford | 66.49 | 56.56 | 68.95 | 743 | Rams |
| 83 | 46 | Chris Baker | 66.00 | 50.67 | 76.02 | 411 | Commanders |
| 84 | 47 | Ishmaa'ily Kitchen | 65.99 | 56.81 | 70.30 | 190 | Browns |
| 85 | 48 | Al Woods | 65.98 | 55.58 | 72.60 | 217 | Steelers |
| 86 | 49 | Jonathan Babineaux | 65.88 | 53.75 | 70.21 | 903 | Falcons |
| 87 | 50 | Tyrunn Walker | 65.79 | 57.44 | 78.05 | 119 | Saints |
| 88 | 51 | Armonty Bryant | 65.68 | 47.16 | 78.03 | 188 | Browns |
| 89 | 52 | Sen'Derrick Marks | 65.63 | 56.63 | 68.09 | 930 | Jaguars |
| 90 | 53 | Stephen Paea | 65.53 | 53.90 | 72.03 | 474 | Bears |
| 91 | 54 | Kevin Vickerson | 65.45 | 49.89 | 76.55 | 392 | Broncos |
| 92 | 55 | Alex Carrington | 65.33 | 60.79 | 72.42 | 163 | Bills |
| 93 | 56 | Darnell Dockett | 65.32 | 47.08 | 73.63 | 866 | Cardinals |
| 94 | 57 | C.J. Wilson | 65.02 | 49.93 | 75.50 | 127 | Packers |
| 95 | 58 | Ryan Pickett | 65.00 | 51.54 | 70.00 | 534 | Packers |
| 96 | 59 | Barry Cofield | 64.94 | 56.13 | 66.64 | 729 | Commanders |
| 97 | 60 | Corey Peters | 64.81 | 53.36 | 70.04 | 656 | Falcons |
| 98 | 61 | Derek Wolfe | 64.80 | 57.33 | 68.87 | 552 | Broncos |
| 99 | 62 | Vince Wilfork | 64.50 | 52.80 | 74.38 | 172 | Patriots |
| 100 | 63 | Mike Patterson | 64.38 | 53.52 | 71.11 | 402 | Giants |
| 101 | 64 | Letroy Guion | 64.26 | 52.51 | 69.50 | 391 | Vikings |
| 102 | 65 | Tommy Kelly | 64.05 | 54.81 | 71.78 | 218 | Patriots |
| 103 | 66 | Aubrayo Franklin | 64.02 | 48.86 | 71.21 | 382 | Colts |
| 104 | 67 | Dwan Edwards | 63.96 | 48.10 | 73.06 | 368 | Panthers |
| 105 | 68 | Glenn Foster | 63.65 | 48.99 | 71.34 | 221 | Saints |
| 106 | 69 | Colin Cole | 63.56 | 42.99 | 74.27 | 329 | Panthers |
| 107 | 70 | Derek Landri | 63.39 | 47.82 | 74.08 | 123 | Buccaneers |
| 108 | 71 | Allen Bailey | 63.08 | 56.72 | 65.86 | 492 | Chiefs |
| 109 | 72 | Abry Jones | 63.07 | 59.65 | 69.52 | 129 | Jaguars |
| 110 | 73 | Tom Johnson | 62.67 | 47.61 | 70.11 | 243 | Saints |
| 111 | 74 | Shaun Rogers | 62.55 | 50.94 | 73.87 | 223 | Giants |
| 112 | 75 | Antonio Johnson | 62.12 | 51.27 | 65.50 | 379 | Titans |
| 113 | 76 | Tony Jerod-Eddie | 62.11 | 50.49 | 71.56 | 437 | 49ers |
| 114 | 77 | B.J. Raji | 62.11 | 53.21 | 63.87 | 653 | Packers |
| 115 | 78 | Brett Keisel | 62.10 | 46.77 | 70.44 | 565 | Steelers |

### Rotation/backup (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 116 | 1 | Kendall Reyes | 61.68 | 50.06 | 65.26 | 817 | Chargers |
| 117 | 2 | Alameda Ta'amu | 61.68 | 52.16 | 65.94 | 225 | Cardinals |
| 118 | 3 | Chris Jones | 61.66 | 44.02 | 70.29 | 902 | Patriots |
| 119 | 4 | Jared Crick | 61.47 | 52.99 | 62.95 | 268 | Texans |
| 120 | 5 | Clifton Geathers | 61.28 | 53.22 | 66.78 | 257 | Eagles |
| 121 | 6 | Josh Chapman | 60.88 | 47.22 | 66.86 | 300 | Colts |
| 122 | 7 | Peria Jerry | 60.86 | 45.95 | 66.63 | 663 | Falcons |
| 123 | 8 | Stephen Bowen | 60.84 | 46.09 | 69.64 | 415 | Commanders |
| 124 | 9 | Nate Collins | 60.71 | 59.10 | 67.51 | 191 | Bears |
| 125 | 10 | Corey Wootton | 60.21 | 42.67 | 67.73 | 846 | Bears |
| 126 | 11 | Brandon Thompson | 60.06 | 52.78 | 65.83 | 428 | Bengals |
| 127 | 12 | Stacy McGee | 60.05 | 48.37 | 64.71 | 350 | Raiders |
| 128 | 13 | Johnny Jolly | 59.79 | 42.53 | 70.26 | 286 | Packers |
| 129 | 14 | Isaac Sopoaga | 59.65 | 43.48 | 67.30 | 356 | Patriots |
| 130 | 15 | Corbin Bryant | 59.58 | 43.75 | 67.00 | 330 | Bills |
| 131 | 16 | Brian Sanford | 59.35 | 44.62 | 78.42 | 105 | Browns |
| 132 | 17 | Jarvis Jenkins | 59.33 | 52.17 | 62.53 | 331 | Commanders |
| 133 | 18 | Fili Moala | 58.81 | 47.68 | 64.98 | 506 | Colts |
| 134 | 19 | Roy Miller | 58.65 | 47.85 | 63.04 | 575 | Jaguars |
| 135 | 20 | Mitch Unrein | 58.52 | 49.26 | 60.73 | 407 | Broncos |
| 136 | 21 | Gary Gibson | 58.37 | 41.25 | 67.18 | 164 | Buccaneers |
| 137 | 22 | Josh Boyd | 58.29 | 50.02 | 66.94 | 116 | Packers |
| 138 | 23 | Terrence Cody | 58.08 | 49.26 | 61.87 | 234 | Ravens |
| 139 | 24 | Jermelle Cudjo | 57.98 | 45.43 | 65.31 | 212 | Rams |
| 140 | 25 | Jay Ratliff | 57.96 | 47.32 | 69.73 | 207 | Bears |
| 141 | 26 | Kedric Golston | 57.55 | 44.38 | 63.63 | 467 | Commanders |
| 142 | 27 | Brandon Deaderick | 57.24 | 46.97 | 62.11 | 302 | Jaguars |
| 143 | 28 | Devon Still | 57.20 | 52.05 | 63.50 | 130 | Bengals |
| 144 | 29 | Frostee Rucker | 57.19 | 38.72 | 65.33 | 356 | Cardinals |
| 145 | 30 | Ricardo Mathews | 57.09 | 46.62 | 60.74 | 477 | Colts |
| 146 | 31 | Nick Hayden | 56.78 | 44.01 | 61.91 | 822 | Cowboys |
| 147 | 32 | Akeem Spence | 56.44 | 46.69 | 58.77 | 695 | Buccaneers |
| 148 | 33 | Daniel Muir | 56.21 | 42.77 | 68.40 | 212 | Raiders |
| 149 | 34 | Terrell McClain | 56.19 | 53.88 | 58.14 | 176 | Texans |
| 150 | 35 | Andre Fluellen | 55.80 | 49.02 | 61.79 | 165 | Lions |
| 151 | 36 | Matt Conrath | 54.66 | 47.24 | 66.51 | 135 | Rams |
| 152 | 37 | Drake Nevis | 54.63 | 37.65 | 68.35 | 270 | Jaguars |
| 153 | 38 | Landon Cohen | 52.38 | 43.34 | 60.75 | 381 | Bears |
| 154 | 39 | Corvey Irvin | 49.29 | 47.08 | 55.87 | 117 | Cowboys |
| 155 | 40 | Damion Square | 47.31 | 43.49 | 50.89 | 154 | Eagles |

## ED — Edge

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 93.79 | 97.54 | 90.77 | 539 | Broncos |
| 2 | 2 | Robert Quinn | 89.57 | 96.11 | 81.24 | 831 | Rams |
| 3 | 3 | Cameron Wake | 88.30 | 84.41 | 87.25 | 682 | Dolphins |
| 4 | 4 | Justin Houston | 88.16 | 87.21 | 86.71 | 756 | Chiefs |
| 5 | 5 | Charles Johnson | 87.35 | 85.01 | 85.47 | 778 | Panthers |
| 6 | 6 | Jerry Hughes | 85.99 | 86.74 | 82.35 | 602 | Bills |
| 7 | 7 | DeMarcus Ware | 85.71 | 79.28 | 87.39 | 630 | Cowboys |
| 8 | 8 | Brandon Graham | 85.71 | 87.70 | 82.91 | 346 | Eagles |
| 9 | 9 | Carlos Dunlap | 84.72 | 90.57 | 77.59 | 985 | Bengals |
| 10 | 10 | Greg Hardy | 84.12 | 88.42 | 77.41 | 937 | Panthers |
| 11 | 11 | Mario Williams | 83.26 | 84.11 | 80.81 | 1000 | Bills |
| 12 | 12 | Michael Bennett | 83.26 | 93.47 | 72.70 | 736 | Seahawks |
| 13 | 13 | Cliff Avril | 81.70 | 78.77 | 79.48 | 670 | Seahawks |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Chris Long | 79.93 | 73.38 | 80.13 | 839 | Rams |
| 15 | 2 | Terrell Suggs | 79.51 | 77.22 | 78.12 | 903 | Ravens |
| 16 | 3 | Cameron Jordan | 79.12 | 86.83 | 69.82 | 1017 | Saints |
| 17 | 4 | Jason Babin | 78.60 | 63.94 | 84.20 | 764 | Jaguars |
| 18 | 5 | Jared Allen | 78.39 | 71.24 | 78.99 | 1062 | Vikings |
| 19 | 6 | William Hayes | 76.27 | 82.38 | 70.31 | 345 | Rams |
| 20 | 7 | Ezekiel Ansah | 76.17 | 77.24 | 73.38 | 554 | Lions |
| 21 | 8 | Justin Tuck | 75.96 | 71.69 | 74.95 | 873 | Giants |
| 22 | 9 | Michael Johnson | 74.57 | 83.91 | 64.18 | 960 | Bengals |
| 23 | 10 | Chandler Jones | 74.49 | 76.08 | 69.27 | 1264 | Patriots |
| 24 | 11 | Vinny Curry | 74.21 | 69.29 | 77.88 | 332 | Eagles |
| 25 | 12 | Julius Peppers | 74.11 | 68.57 | 73.64 | 851 | Bears |

### Starter (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Jason Pierre-Paul | 73.71 | 79.48 | 68.29 | 568 | Giants |
| 27 | 2 | Ryan Kerrigan | 73.31 | 62.49 | 76.36 | 974 | Commanders |
| 28 | 3 | Everson Griffen | 73.31 | 72.09 | 69.96 | 698 | Vikings |
| 29 | 4 | Chris Clemons | 72.38 | 62.19 | 75.00 | 707 | Seahawks |
| 30 | 5 | Manny Lawson | 71.69 | 49.23 | 83.01 | 705 | Bills |
| 31 | 6 | Dion Jordan | 71.59 | 72.73 | 66.66 | 330 | Dolphins |
| 32 | 7 | Robert Ayers | 71.43 | 77.44 | 63.26 | 605 | Broncos |
| 33 | 8 | Parys Haralson | 70.21 | 55.57 | 76.01 | 367 | Saints |
| 34 | 9 | Jabaal Sheard | 69.59 | 68.89 | 67.46 | 651 | Browns |
| 35 | 10 | Brian Robison | 69.18 | 63.19 | 69.00 | 973 | Vikings |
| 36 | 11 | Devin Taylor | 68.91 | 67.18 | 67.98 | 301 | Lions |
| 37 | 12 | Olivier Vernon | 68.16 | 66.47 | 65.12 | 912 | Dolphins |
| 38 | 13 | Ryan Davis Sr. | 68.10 | 63.69 | 78.59 | 105 | Jaguars |
| 39 | 14 | Wallace Gilberry | 67.80 | 52.71 | 74.01 | 537 | Bengals |
| 40 | 15 | Shaun Phillips | 67.56 | 53.47 | 73.62 | 912 | Broncos |
| 41 | 16 | O'Brien Schofield | 66.53 | 59.37 | 70.89 | 169 | Seahawks |
| 42 | 17 | Jonathan Massaquoi | 64.79 | 62.36 | 66.92 | 527 | Falcons |
| 43 | 18 | Brooks Reed | 64.02 | 59.98 | 63.18 | 999 | Texans |
| 44 | 19 | Melvin Ingram III | 63.73 | 63.93 | 65.94 | 235 | Chargers |
| 45 | 20 | Mathias Kiwanuka | 63.20 | 51.88 | 66.58 | 871 | Giants |
| 46 | 21 | Adrian Clayborn | 63.19 | 60.56 | 64.84 | 934 | Buccaneers |
| 47 | 22 | Israel Idonije | 63.10 | 55.28 | 64.67 | 333 | Lions |
| 48 | 23 | Derrick Shelby | 62.51 | 62.08 | 58.63 | 438 | Dolphins |
| 49 | 24 | Michael Buchanan | 62.50 | 60.68 | 65.80 | 119 | Patriots |
| 50 | 25 | William Gholston | 62.18 | 60.17 | 63.52 | 312 | Buccaneers |
| 51 | 26 | Andre Branch | 62.00 | 64.44 | 57.38 | 592 | Jaguars |

### Rotation/backup (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 52 | 1 | Matt Shaughnessy | 61.95 | 59.85 | 61.88 | 717 | Cardinals |
| 53 | 2 | Frank Alexander | 61.64 | 60.70 | 60.05 | 243 | Panthers |
| 54 | 3 | George Selvie | 61.47 | 54.16 | 64.36 | 746 | Cowboys |
| 55 | 4 | Mario Addison | 61.20 | 58.66 | 60.61 | 266 | Panthers |
| 56 | 5 | David Bass | 60.58 | 59.61 | 61.22 | 311 | Bears |
| 57 | 6 | Damontre Moore | 58.95 | 63.20 | 55.09 | 133 | Giants |
| 58 | 7 | Eugene Sims | 56.01 | 53.69 | 54.96 | 385 | Rams |
| 59 | 8 | Wes Horton | 55.86 | 53.94 | 59.22 | 169 | Panthers |
| 60 | 9 | Cliff Matthews | 55.67 | 52.22 | 57.04 | 170 | Falcons |
| 61 | 10 | Jason Hunter | 55.39 | 49.34 | 56.83 | 615 | Raiders |
| 62 | 11 | Malliciah Goodman | 55.05 | 58.12 | 50.92 | 302 | Falcons |
| 63 | 12 | Andre Carter | 54.66 | 48.61 | 58.37 | 195 | Patriots |
| 64 | 13 | Daniel Te'o-Nesheim | 54.65 | 53.24 | 53.61 | 602 | Buccaneers |
| 65 | 14 | Margus Hunt | 53.79 | 54.25 | 54.51 | 168 | Bengals |
| 66 | 15 | Keyunta Dawson | 51.71 | 57.58 | 51.33 | 105 | Saints |
| 67 | 16 | Stansly Maponga | 50.17 | 53.12 | 48.20 | 126 | Falcons |
| 68 | 17 | Josh Martin | 49.30 | 60.30 | 56.30 | 102 | Chiefs |
| 69 | 18 | Lavar Edwards | 49.28 | 57.60 | 50.43 | 150 | Titans |

## G — Guard

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Evan Mathis | 97.93 | 95.00 | 95.71 | 1163 | Eagles |
| 2 | 2 | Josh Sitton | 95.55 | 90.70 | 94.62 | 1185 | Packers |
| 3 | 3 | Brandon Brooks | 94.02 | 87.70 | 94.06 | 1043 | Texans |
| 4 | 4 | Louis Vasquez | 93.02 | 88.20 | 92.07 | 1409 | Broncos |
| 5 | 5 | Geoff Schwartz | 92.81 | 87.00 | 92.51 | 625 | Chiefs |
| 6 | 6 | Amini Silatolu | 91.23 | 87.30 | 89.68 | 170 | Panthers |
| 7 | 7 | Brandon Fusco | 90.54 | 84.10 | 90.67 | 904 | Vikings |
| 8 | 8 | Marshal Yanda | 89.98 | 83.80 | 89.93 | 1136 | Ravens |
| 9 | 9 | Shelley Smith | 89.41 | 83.80 | 88.99 | 363 | Rams |
| 10 | 10 | Matt Slauson | 89.05 | 84.80 | 87.72 | 1059 | Bears |
| 11 | 11 | Joe Reitz | 88.57 | 84.10 | 87.39 | 145 | Colts |
| 12 | 12 | Andy Levitre | 88.51 | 82.10 | 88.61 | 1069 | Titans |
| 13 | 13 | Richie Incognito | 88.04 | 82.70 | 87.44 | 468 | Dolphins |
| 14 | 14 | David DeCastro | 87.82 | 81.60 | 87.80 | 948 | Steelers |
| 15 | 15 | Ben Grubbs | 87.80 | 82.70 | 87.04 | 1274 | Saints |
| 16 | 16 | Larry Warford | 87.61 | 85.10 | 85.11 | 1138 | Lions |
| 17 | 17 | T.J. Lang | 87.60 | 80.40 | 88.23 | 1157 | Packers |
| 18 | 18 | Todd Herremans | 86.32 | 79.80 | 86.50 | 1163 | Eagles |
| 19 | 19 | Kevin Zeitler | 86.05 | 78.40 | 86.98 | 860 | Bengals |
| 20 | 20 | Carl Nicks | 85.71 | 77.30 | 87.15 | 145 | Buccaneers |
| 21 | 21 | Zane Beadles | 85.50 | 78.00 | 86.34 | 1415 | Broncos |
| 22 | 22 | Clint Boling | 85.36 | 78.90 | 85.50 | 776 | Bengals |
| 23 | 23 | Logan Mankins | 85.33 | 78.90 | 85.45 | 1317 | Patriots |
| 24 | 24 | Jahri Evans | 85.30 | 79.30 | 85.13 | 1122 | Saints |
| 25 | 25 | Garrett Reynolds | 84.61 | 79.00 | 84.19 | 682 | Falcons |
| 26 | 26 | Travelle Wharton | 84.03 | 77.50 | 84.22 | 896 | Panthers |
| 27 | 27 | Kraig Urbik | 83.91 | 77.20 | 84.22 | 1143 | Bills |
| 28 | 28 | Brian Waters | 82.29 | 76.20 | 82.18 | 330 | Cowboys |
| 29 | 29 | Mike Pollak | 82.28 | 74.20 | 83.50 | 397 | Bengals |
| 30 | 30 | Chad Rinehart | 81.90 | 75.00 | 82.33 | 775 | Chargers |
| 31 | 31 | Ramon Foster | 81.68 | 74.30 | 82.43 | 842 | Steelers |
| 32 | 32 | Chris Chester | 81.15 | 73.50 | 82.09 | 1145 | Commanders |
| 33 | 33 | Jason Pinkston | 81.10 | 72.70 | 82.53 | 151 | Browns |
| 34 | 34 | John Greco | 80.97 | 74.20 | 81.32 | 923 | Browns |
| 35 | 35 | Joe Berger | 80.97 | 71.60 | 83.05 | 214 | Vikings |
| 36 | 36 | Chance Warmack | 80.79 | 73.40 | 81.55 | 1075 | Titans |
| 37 | 37 | Uche Nwaneri | 80.69 | 73.70 | 81.19 | 1058 | Jaguars |
| 38 | 38 | Rob Sims | 80.47 | 73.00 | 81.28 | 1138 | Lions |
| 39 | 39 | Jon Asamoah | 80.45 | 72.70 | 81.45 | 661 | Chiefs |
| 40 | 40 | Justin Blalock | 80.29 | 73.60 | 80.58 | 1077 | Falcons |

### Good (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 41 | 1 | J.R. Sweezy | 79.77 | 71.30 | 81.25 | 1136 | Seahawks |
| 42 | 2 | Willie Colon | 79.62 | 69.10 | 82.47 | 1039 | Jets |
| 43 | 3 | Wade Smith | 79.56 | 72.60 | 80.03 | 972 | Texans |
| 44 | 4 | Kyle Long | 79.36 | 70.40 | 81.16 | 1059 | Bears |
| 45 | 5 | Mike Iupati | 79.29 | 70.80 | 80.78 | 842 | 49ers |
| 46 | 6 | Mike Brisiel | 79.06 | 69.70 | 81.13 | 870 | Raiders |
| 47 | 7 | Ron Leary | 78.87 | 70.00 | 80.61 | 992 | Cowboys |
| 48 | 8 | Johnnie Troutman | 78.38 | 68.30 | 80.93 | 680 | Chargers |
| 49 | 9 | Harvey Dahl | 77.74 | 67.40 | 80.47 | 529 | Rams |
| 50 | 10 | Chris Kuper | 77.66 | 68.60 | 79.53 | 106 | Broncos |
| 51 | 11 | John Jerry | 77.55 | 69.30 | 78.89 | 1015 | Dolphins |
| 52 | 12 | Jeff Allen | 77.08 | 68.20 | 78.83 | 961 | Chiefs |
| 53 | 13 | Dan Connolly | 76.91 | 67.70 | 78.88 | 1249 | Patriots |
| 54 | 14 | Alex Boone | 76.91 | 66.90 | 79.41 | 1176 | 49ers |
| 55 | 15 | Daryn Colledge | 76.73 | 67.40 | 78.79 | 1027 | Cardinals |
| 56 | 16 | Kevin Boothe | 76.46 | 67.80 | 78.06 | 1022 | Giants |
| 57 | 17 | Hugh Thornton | 76.02 | 66.50 | 78.20 | 987 | Colts |
| 58 | 18 | Kelechi Osemele | 75.67 | 68.80 | 76.09 | 424 | Ravens |
| 59 | 19 | Charlie Johnson | 75.61 | 67.30 | 76.99 | 986 | Vikings |
| 60 | 20 | James Carpenter | 74.73 | 64.00 | 77.71 | 823 | Seahawks |
| 61 | 21 | Adam Snyder | 74.55 | 64.70 | 76.95 | 412 | 49ers |

### Starter (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 62 | 1 | Gabe Carimi | 73.93 | 66.70 | 74.59 | 212 | Buccaneers |
| 63 | 2 | Jeromey Clary | 73.92 | 65.10 | 75.64 | 1094 | Chargers |
| 64 | 3 | James Brewer | 73.42 | 63.60 | 75.80 | 434 | Giants |
| 65 | 4 | Shawn Lauvao | 72.70 | 63.70 | 74.54 | 742 | Browns |
| 66 | 5 | Oniel Cousins | 72.35 | 59.10 | 77.02 | 314 | Browns |
| 67 | 6 | Josh Kline | 71.70 | 64.80 | 72.14 | 112 | Patriots |
| 68 | 7 | Ted Larsen | 71.29 | 59.30 | 75.12 | 363 | Buccaneers |
| 69 | 8 | Vladimir Ducasse | 71.05 | 55.70 | 77.12 | 318 | Jets |
| 70 | 9 | Paul Fanaika | 70.68 | 61.00 | 72.96 | 1084 | Cardinals |
| 71 | 10 | Brian Winters | 70.42 | 59.60 | 73.46 | 763 | Jets |
| 72 | 11 | Chris Williams | 69.37 | 56.50 | 73.78 | 902 | Rams |
| 73 | 12 | Davin Joseph | 68.79 | 58.90 | 71.22 | 1013 | Buccaneers |
| 74 | 13 | Chris Scott | 67.90 | 55.00 | 72.34 | 537 | Panthers |
| 75 | 14 | Mike McGlynn | 67.74 | 55.80 | 71.53 | 1019 | Colts |
| 76 | 15 | Chris Snee | 67.30 | 54.10 | 71.93 | 187 | Giants |
| 77 | 16 | Tim Lelito | 62.16 | 52.80 | 64.24 | 159 | Saints |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 78 | 1 | Will Rackley | 60.98 | 44.30 | 67.94 | 649 | Jaguars |
| 79 | 2 | Lucas Nix | 57.68 | 38.70 | 66.17 | 642 | Raiders |

## HB — Running Back

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Andre Ellington | 82.91 | 84.20 | 77.88 | 220 | Cardinals |
| 2 | 2 | Marshawn Lynch | 80.67 | 90.70 | 69.82 | 315 | Seahawks |
| 3 | 3 | Adrian Peterson | 80.48 | 79.70 | 76.84 | 260 | Vikings |
| 4 | 4 | Darren Sproles | 80.47 | 83.10 | 74.55 | 311 | Saints |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | LeSean McCoy | 79.86 | 87.00 | 70.93 | 428 | Eagles |
| 6 | 2 | Jamaal Charles | 79.01 | 84.90 | 70.91 | 396 | Chiefs |
| 7 | 3 | DeMarco Murray | 77.28 | 79.80 | 71.43 | 305 | Cowboys |
| 8 | 4 | Eddie Lacy | 76.77 | 88.90 | 64.51 | 252 | Packers |
| 9 | 5 | Fred Jackson | 76.32 | 76.60 | 71.96 | 271 | Bills |
| 10 | 6 | Alfred Morris | 76.15 | 80.40 | 69.15 | 173 | Commanders |
| 11 | 7 | Pierre Thomas | 75.56 | 79.10 | 69.03 | 301 | Saints |
| 12 | 8 | C.J. Spiller | 75.04 | 71.10 | 73.50 | 145 | Bills |
| 13 | 9 | Joique Bell | 74.58 | 76.00 | 69.46 | 288 | Lions |

### Starter (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Toby Gerhart | 73.96 | 71.40 | 71.50 | 107 | Vikings |
| 15 | 2 | Rashad Jennings | 73.22 | 78.60 | 65.46 | 212 | Raiders |
| 16 | 3 | Giovani Bernard | 72.72 | 76.60 | 65.96 | 342 | Bengals |
| 17 | 4 | Bobby Rainey Jr. | 72.60 | 80.50 | 63.16 | 106 | Buccaneers |
| 18 | 5 | Matt Forte | 72.33 | 72.40 | 68.12 | 433 | Bears |
| 19 | 6 | Donald Brown | 71.90 | 72.20 | 67.53 | 192 | Colts |
| 20 | 7 | Jacquizz Rodgers | 71.81 | 74.70 | 65.72 | 237 | Falcons |
| 21 | 8 | Arian Foster | 71.38 | 73.60 | 65.74 | 158 | Texans |
| 22 | 9 | Danny Woodhead | 71.23 | 80.80 | 60.68 | 323 | Chargers |
| 23 | 10 | Ryan Mathews | 71.22 | 71.50 | 66.86 | 133 | Chargers |
| 24 | 11 | Montee Ball | 71.15 | 72.70 | 65.95 | 166 | Broncos |
| 25 | 12 | Maurice Jones-Drew | 71.15 | 73.00 | 65.75 | 266 | Jaguars |
| 26 | 13 | Ben Tate | 70.98 | 67.30 | 69.26 | 212 | Texans |
| 27 | 14 | Reggie Bush | 70.68 | 72.40 | 65.36 | 302 | Lions |
| 28 | 15 | Shane Vereen | 70.61 | 75.30 | 63.31 | 249 | Patriots |
| 29 | 16 | DeAngelo Williams | 69.80 | 68.70 | 66.36 | 171 | Panthers |
| 30 | 17 | Knowshon Moreno | 69.59 | 73.60 | 62.75 | 396 | Broncos |
| 31 | 18 | Frank Gore | 68.77 | 69.80 | 63.92 | 294 | 49ers |
| 32 | 19 | Jason Snelling | 68.42 | 70.10 | 63.14 | 117 | Falcons |
| 33 | 20 | Steven Jackson | 68.34 | 69.90 | 63.13 | 172 | Falcons |
| 34 | 21 | Bernard Pierce | 68.12 | 67.90 | 64.10 | 150 | Ravens |
| 35 | 22 | Zac Stacy | 67.18 | 69.60 | 61.40 | 215 | Rams |
| 36 | 23 | Doug Martin | 67.13 | 60.00 | 67.71 | 105 | Buccaneers |
| 37 | 24 | Le'Veon Bell | 66.88 | 71.60 | 59.56 | 328 | Steelers |
| 38 | 25 | Lamar Miller | 66.84 | 69.80 | 60.70 | 279 | Dolphins |
| 39 | 26 | Daniel Thomas | 66.21 | 74.50 | 56.51 | 128 | Dolphins |
| 40 | 27 | Chris Johnson | 65.04 | 62.40 | 62.64 | 379 | Titans |
| 41 | 28 | Robert Turbin | 64.70 | 67.60 | 58.60 | 106 | Seahawks |
| 42 | 29 | Chris Ogbonnaya | 64.66 | 59.50 | 63.93 | 267 | Browns |
| 43 | 30 | Darren McFadden | 64.62 | 57.80 | 65.00 | 125 | Raiders |
| 44 | 31 | Rashard Mendenhall | 64.48 | 65.10 | 59.90 | 158 | Cardinals |
| 45 | 32 | Brandon Bolden | 64.38 | 57.20 | 65.00 | 152 | Patriots |
| 46 | 33 | Roy Helu | 64.24 | 60.50 | 62.56 | 301 | Commanders |
| 47 | 34 | Andre Brown | 63.92 | 62.70 | 60.57 | 137 | Giants |
| 48 | 35 | Trent Richardson | 63.87 | 62.60 | 60.55 | 256 | Colts |
| 49 | 36 | Bilal Powell | 62.66 | 64.50 | 57.26 | 270 | Jets |
| 50 | 37 | Ray Rice | 62.18 | 54.60 | 63.06 | 331 | Ravens |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 51 | 1 | BenJarvus Green-Ellis | 60.76 | 57.80 | 58.57 | 150 | Bengals |
| 52 | 2 | Fozzy Whittaker | 58.77 | 58.30 | 54.92 | 112 | Browns |
| 53 | 3 | Jordan Todman | 57.36 | 57.90 | 52.84 | 136 | Jaguars |

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

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | James Harrison | 79.55 | 78.50 | 76.08 | 402 | Bengals |
| 11 | 2 | Nigel Bradham | 79.08 | 81.50 | 74.73 | 285 | Bills |
| 12 | 3 | Luke Kuechly | 78.50 | 77.00 | 75.34 | 1057 | Panthers |
| 13 | 4 | Vincent Rey | 78.50 | 86.00 | 73.50 | 344 | Bengals |
| 14 | 5 | Akeem Jordan | 75.25 | 75.30 | 72.71 | 475 | Chiefs |
| 15 | 6 | Danny Trevathan | 74.76 | 74.00 | 71.89 | 1116 | Broncos |

### Starter (64 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Dont'a Hightower | 73.79 | 71.60 | 71.09 | 998 | Patriots |
| 17 | 2 | K.J. Wright | 73.38 | 72.00 | 70.86 | 806 | Seahawks |
| 18 | 3 | Malcolm Smith | 73.32 | 75.00 | 71.89 | 618 | Seahawks |
| 19 | 4 | Daryl Washington | 72.82 | 72.80 | 70.95 | 798 | Cardinals |
| 20 | 5 | Jamie Collins Sr. | 72.70 | 72.10 | 68.94 | 435 | Patriots |
| 21 | 6 | DeAndre Levy | 72.11 | 69.60 | 70.25 | 1018 | Lions |
| 22 | 7 | Arthur Moats | 72.10 | 72.00 | 71.24 | 292 | Bills |
| 23 | 8 | Dekoda Watson | 72.06 | 71.60 | 69.45 | 258 | Buccaneers |
| 24 | 9 | Kavell Conner | 71.75 | 70.20 | 72.89 | 144 | Colts |
| 25 | 10 | Sean Lee | 71.71 | 73.10 | 72.55 | 703 | Cowboys |
| 26 | 11 | Stephen Nicholas | 71.59 | 76.70 | 69.95 | 130 | Falcons |
| 27 | 12 | Daryl Smith | 71.16 | 72.20 | 70.66 | 1071 | Ravens |
| 28 | 13 | James Laurinaitis | 70.80 | 67.20 | 69.04 | 1055 | Rams |
| 29 | 14 | Brandon Spikes | 70.78 | 69.20 | 68.70 | 686 | Patriots |
| 30 | 15 | Jerrell Freeman | 70.66 | 71.10 | 66.20 | 1107 | Colts |
| 31 | 16 | Sio Moore | 70.57 | 68.30 | 68.95 | 578 | Raiders |
| 32 | 17 | Bobby Wagner | 70.41 | 66.60 | 68.78 | 1058 | Seahawks |
| 33 | 18 | Ramon Humber | 70.34 | 67.60 | 69.56 | 172 | Saints |
| 34 | 19 | David Harris | 70.25 | 67.20 | 68.11 | 1100 | Jets |
| 35 | 20 | Curtis Lofton | 69.47 | 65.80 | 67.75 | 1053 | Saints |
| 36 | 21 | Nate Irving | 68.79 | 67.90 | 70.12 | 354 | Broncos |
| 37 | 22 | Erin Henderson | 68.71 | 65.40 | 68.31 | 850 | Vikings |
| 38 | 23 | Brian Cushing | 68.64 | 69.80 | 71.84 | 330 | Texans |
| 39 | 24 | Paul Posluszny | 68.63 | 63.30 | 68.54 | 1038 | Jaguars |
| 40 | 25 | Josh McNary | 68.06 | 73.60 | 71.06 | 132 | Colts |
| 41 | 26 | Nick Roach | 68.04 | 66.20 | 65.10 | 1075 | Raiders |
| 42 | 27 | Jason Trusnik | 68.02 | 70.00 | 67.54 | 182 | Dolphins |
| 43 | 28 | Lawrence Timmons | 67.72 | 64.00 | 66.04 | 1072 | Steelers |
| 44 | 29 | Josh Bynes | 67.69 | 67.00 | 67.32 | 455 | Ravens |
| 45 | 30 | Koa Misi | 67.29 | 64.90 | 65.87 | 473 | Dolphins |
| 46 | 31 | A.J. Hawk | 67.11 | 61.20 | 67.09 | 1057 | Packers |
| 47 | 32 | Alec Ogletree | 67.06 | 63.00 | 65.60 | 1033 | Rams |
| 48 | 33 | A.J. Klein | 66.93 | 68.70 | 66.78 | 130 | Panthers |
| 49 | 34 | Akeem Ayers | 66.70 | 64.40 | 64.06 | 724 | Titans |
| 50 | 35 | Wesley Woodyard | 66.42 | 61.90 | 65.26 | 844 | Broncos |
| 51 | 36 | Marvin Mitchell | 65.86 | 64.70 | 67.37 | 306 | Vikings |
| 52 | 37 | Russell Allen | 65.67 | 63.80 | 64.64 | 596 | Jaguars |
| 53 | 38 | Justin Durant | 65.44 | 64.70 | 65.32 | 200 | Cowboys |
| 54 | 39 | Zach Brown | 65.39 | 60.50 | 64.49 | 759 | Titans |
| 55 | 40 | D'Qwell Jackson | 65.32 | 60.10 | 64.64 | 1147 | Browns |
| 56 | 41 | John Lotulelei | 65.31 | 60.00 | 67.03 | 108 | Jaguars |
| 57 | 42 | Manti Te'o | 65.31 | 58.80 | 66.51 | 585 | Chargers |
| 58 | 43 | Jonathan Casillas | 65.25 | 64.60 | 65.37 | 198 | Buccaneers |
| 59 | 44 | James Anderson | 64.75 | 56.20 | 67.53 | 999 | Bears |
| 60 | 45 | Audie Cole | 64.42 | 70.50 | 67.91 | 325 | Vikings |
| 61 | 46 | Will Witherspoon | 64.29 | 63.80 | 66.70 | 136 | Rams |
| 62 | 47 | Geno Hayes | 64.26 | 63.00 | 65.37 | 920 | Jaguars |
| 63 | 48 | Keith Rivers | 64.24 | 63.20 | 64.41 | 419 | Giants |
| 64 | 49 | Spencer Paysinger | 63.93 | 60.90 | 67.51 | 691 | Giants |
| 65 | 50 | Jacquian Williams | 63.90 | 60.20 | 64.08 | 606 | Giants |
| 66 | 51 | Joplo Bartu | 63.79 | 59.40 | 63.59 | 772 | Falcons |
| 67 | 52 | Colin McCarthy | 63.42 | 64.40 | 65.58 | 332 | Titans |
| 68 | 53 | Brad Jones | 63.11 | 60.40 | 63.25 | 639 | Packers |
| 69 | 54 | DeMeco Ryans | 63.10 | 55.80 | 63.80 | 1234 | Eagles |
| 70 | 55 | Rey Maualuga | 63.06 | 59.40 | 62.80 | 649 | Bengals |
| 71 | 56 | Vince Williams | 63.03 | 56.80 | 64.05 | 401 | Steelers |
| 72 | 57 | Donald Butler | 63.02 | 57.70 | 64.16 | 851 | Chargers |
| 73 | 58 | Dannell Ellerbe | 62.97 | 59.10 | 62.95 | 1011 | Dolphins |
| 74 | 59 | Arthur Brown | 62.89 | 61.50 | 61.73 | 205 | Ravens |
| 75 | 60 | Mason Foster | 62.84 | 59.10 | 61.68 | 773 | Buccaneers |
| 76 | 61 | Bruce Carter | 62.82 | 59.00 | 65.37 | 875 | Cowboys |
| 77 | 62 | Kevin Burnett | 62.28 | 54.70 | 63.17 | 978 | Raiders |
| 78 | 63 | Paul Worrilow | 62.19 | 57.00 | 63.57 | 773 | Falcons |
| 79 | 64 | Jasper Brinkley | 62.12 | 59.40 | 66.63 | 203 | Cardinals |

### Rotation/backup (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 80 | 1 | Jerod Mayo | 61.66 | 57.70 | 65.34 | 399 | Patriots |
| 81 | 2 | Perry Riley | 60.85 | 56.10 | 61.31 | 989 | Commanders |
| 82 | 3 | Bront Bird | 60.67 | 60.60 | 65.09 | 142 | Chargers |
| 83 | 4 | Joe Mays | 60.44 | 60.70 | 62.08 | 544 | Texans |
| 84 | 5 | Reggie Walker | 60.40 | 53.60 | 63.26 | 540 | Chargers |
| 85 | 6 | Michael Wilhoite | 60.34 | 63.00 | 66.39 | 175 | 49ers |
| 86 | 7 | Darryl Sharpton | 60.26 | 54.30 | 64.55 | 718 | Texans |
| 87 | 8 | Kelvin Sheppard | 60.15 | 51.00 | 62.72 | 422 | Colts |
| 88 | 9 | Lance Briggs | 60.00 | 58.00 | 60.81 | 557 | Bears |
| 89 | 10 | Mychal Kendricks | 59.85 | 53.10 | 60.56 | 1069 | Eagles |
| 90 | 11 | Andrew Gachkar | 59.68 | 61.30 | 65.27 | 167 | Chargers |
| 91 | 12 | Jameel McClain | 59.60 | 52.80 | 64.04 | 368 | Ravens |
| 92 | 13 | Dane Fletcher | 59.52 | 56.30 | 60.83 | 232 | Patriots |
| 93 | 14 | D.J. Williams | 59.49 | 59.90 | 62.97 | 213 | Bears |
| 94 | 15 | Ashlee Palmer | 59.13 | 54.70 | 60.83 | 357 | Lions |
| 95 | 16 | Demario Davis | 58.84 | 52.60 | 59.22 | 1049 | Jets |
| 96 | 17 | Jelani Jenkins | 58.63 | 57.60 | 59.32 | 125 | Dolphins |
| 97 | 18 | Jamari Lattimore | 58.39 | 63.10 | 61.09 | 265 | Packers |
| 98 | 19 | Pat Angerer | 58.35 | 52.90 | 61.66 | 493 | Colts |
| 99 | 20 | Adam Hayward | 58.00 | 52.70 | 61.95 | 188 | Buccaneers |
| 100 | 21 | Sean Weatherspoon | 57.30 | 54.00 | 60.33 | 392 | Falcons |
| 101 | 22 | Chase Blackburn | 57.09 | 53.70 | 59.55 | 242 | Panthers |
| 102 | 23 | London Fletcher | 56.73 | 45.30 | 60.18 | 917 | Commanders |
| 103 | 24 | David Hawthorne | 56.54 | 50.10 | 58.43 | 783 | Saints |
| 104 | 25 | Craig Robertson | 56.46 | 46.90 | 59.97 | 845 | Browns |
| 105 | 26 | Jon Bostic | 56.40 | 47.00 | 63.70 | 605 | Bears |
| 106 | 27 | JoLonn Dunbar | 56.10 | 50.10 | 58.85 | 419 | Rams |
| 107 | 28 | Jeff Tarpinian | 55.87 | 58.00 | 62.79 | 181 | Texans |
| 108 | 29 | Philip Wheeler | 55.76 | 48.00 | 57.40 | 1034 | Dolphins |
| 109 | 30 | Kaluka Maiava | 55.73 | 51.20 | 60.41 | 118 | Raiders |
| 110 | 31 | Chad Greenway | 55.58 | 47.70 | 56.66 | 1156 | Vikings |
| 111 | 32 | Moise Fokou | 55.37 | 47.80 | 59.58 | 720 | Titans |
| 112 | 33 | Najee Goode | 55.11 | 60.80 | 61.02 | 191 | Eagles |
| 113 | 34 | Akeem Dent | 53.82 | 46.60 | 59.25 | 363 | Falcons |
| 114 | 35 | Jon Beason | 53.70 | 47.90 | 61.31 | 783 | Giants |
| 115 | 36 | Paris Lenon | 51.93 | 37.50 | 57.90 | 301 | Broncos |
| 116 | 37 | Mark Herzlich | 50.39 | 40.80 | 59.70 | 191 | Giants |
| 117 | 38 | J.T. Thomas | 49.19 | 43.70 | 59.36 | 190 | Jaguars |
| 118 | 39 | Khaseem Greene | 48.21 | 38.40 | 57.88 | 231 | Bears |
| 119 | 40 | Ernie Sims | 47.22 | 30.60 | 58.10 | 382 | Cowboys |
| 120 | 41 | DeVonte Holloman | 45.00 | 30.10 | 56.00 | 208 | Cowboys |

## QB — Quarterback

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Peyton Manning | 87.64 | 92.62 | 80.74 | 858 | Broncos |
| 2 | 2 | Drew Brees | 84.87 | 86.75 | 78.78 | 822 | Saints |
| 3 | 3 | Philip Rivers | 83.71 | 83.67 | 79.35 | 684 | Chargers |
| 4 | 4 | Aaron Rodgers | 83.48 | 87.41 | 80.54 | 381 | Packers |
| 5 | 5 | Russell Wilson | 82.96 | 84.19 | 78.41 | 623 | Seahawks |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Ben Roethlisberger | 79.30 | 83.37 | 72.30 | 663 | Steelers |
| 7 | 2 | Tom Brady | 78.42 | 83.83 | 69.45 | 785 | Patriots |
| 8 | 3 | Tony Romo | 77.98 | 78.07 | 74.04 | 614 | Cowboys |
| 9 | 4 | Matt Ryan | 77.61 | 79.95 | 71.14 | 744 | Falcons |
| 10 | 5 | Matthew Stafford | 75.88 | 79.34 | 68.51 | 694 | Lions |
| 11 | 6 | Nick Foles | 74.18 | 77.89 | 81.66 | 423 | Eagles |

### Starter (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Jay Cutler | 73.02 | 76.43 | 71.60 | 406 | Bears |
| 13 | 2 | Cam Newton | 72.61 | 70.28 | 70.81 | 612 | Panthers |
| 14 | 3 | Ryan Tannehill | 72.56 | 78.20 | 64.04 | 679 | Dolphins |
| 15 | 4 | Andy Dalton | 72.43 | 70.61 | 69.37 | 726 | Bengals |
| 16 | 5 | Colin Kaepernick | 72.02 | 70.80 | 72.26 | 629 | 49ers |
| 17 | 6 | Andrew Luck | 71.79 | 71.05 | 67.85 | 784 | Colts |
| 18 | 7 | Eli Manning | 71.31 | 75.19 | 64.16 | 625 | Giants |
| 19 | 8 | Carson Palmer | 71.30 | 68.80 | 69.67 | 655 | Cardinals |
| 20 | 9 | Alex Smith | 70.59 | 71.77 | 67.76 | 684 | Chiefs |
| 21 | 10 | Robert Griffin III | 68.74 | 71.51 | 64.62 | 548 | Commanders |
| 22 | 11 | Joe Flacco | 68.53 | 69.63 | 63.09 | 716 | Ravens |
| 23 | 12 | Josh McCown | 67.70 | 84.17 | 79.42 | 258 | Bears |
| 24 | 13 | Sam Bradford | 67.42 | 72.41 | 65.73 | 297 | Rams |
| 25 | 14 | Ryan Fitzpatrick | 64.64 | 63.79 | 65.18 | 422 | Titans |
| 26 | 15 | Michael Vick | 64.02 | 62.97 | 70.57 | 178 | Eagles |
| 27 | 16 | Jake Locker | 63.63 | 70.33 | 67.08 | 221 | Titans |
| 28 | 17 | Matt Schaub | 63.24 | 65.79 | 61.86 | 400 | Texans |
| 29 | 18 | Mike Glennon | 62.32 | 66.00 | 62.84 | 502 | Buccaneers |

### Rotation/backup (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Scott Tolzien | 61.11 | 67.83 | 67.16 | 102 | Packers |
| 31 | 2 | Geno Smith | 61.06 | 60.60 | 61.04 | 538 | Jets |
| 32 | 3 | Matt Cassel | 60.96 | 63.45 | 63.83 | 287 | Vikings |
| 33 | 4 | Thaddeus Lewis | 60.83 | 67.35 | 66.42 | 190 | Bills |
| 34 | 5 | Matt McGloin | 60.58 | 62.90 | 67.00 | 233 | Raiders |
| 35 | 6 | Kellen Clemens | 60.11 | 66.66 | 62.93 | 291 | Rams |
| 36 | 7 | Case Keenum | 60.02 | 60.80 | 64.79 | 283 | Texans |
| 37 | 8 | Brian Hoyer | 59.95 | 69.23 | 61.32 | 105 | Browns |
| 38 | 9 | Chad Henne | 59.88 | 60.26 | 59.50 | 577 | Jaguars |
| 39 | 10 | Matt Flynn | 59.45 | 61.40 | 65.24 | 250 | Packers |
| 40 | 11 | E.J. Manuel | 58.65 | 57.50 | 59.51 | 387 | Bills |
| 41 | 12 | Brandon Weeden | 57.93 | 58.89 | 59.43 | 316 | Browns |
| 42 | 13 | Christian Ponder | 57.88 | 55.34 | 62.76 | 301 | Vikings |
| 43 | 14 | Josh Freeman | 57.56 | 59.19 | 60.49 | 167 | Vikings |
| 44 | 15 | Jason Campbell | 57.53 | 60.41 | 58.69 | 357 | Browns |
| 45 | 16 | Kirk Cousins | 56.71 | 57.64 | 57.73 | 165 | Commanders |
| 46 | 17 | Blaine Gabbert | 47.66 | 48.28 | 54.75 | 109 | Jaguars |

## S — Safety

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Kam Chancellor | 91.31 | 89.30 | 88.68 | 1199 | Seahawks |
| 2 | 2 | Jairus Byrd | 90.38 | 89.80 | 89.20 | 634 | Bills |
| 3 | 3 | Eric Weddle | 90.38 | 89.20 | 87.00 | 1145 | Chargers |
| 4 | 4 | Devin McCourty | 89.56 | 86.00 | 87.77 | 1164 | Patriots |
| 5 | 5 | Earl Thomas III | 89.04 | 89.10 | 84.83 | 1208 | Seahawks |
| 6 | 6 | Troy Polamalu | 88.35 | 90.10 | 85.83 | 1072 | Steelers |
| 7 | 7 | Eric Berry | 87.99 | 87.90 | 87.02 | 1077 | Chiefs |
| 8 | 8 | Donte Whitner | 87.57 | 86.40 | 84.18 | 1194 | 49ers |
| 9 | 9 | Will Hill III | 87.07 | 89.00 | 86.95 | 766 | Giants |
| 10 | 10 | Glover Quin | 81.32 | 78.60 | 78.96 | 991 | Lions |
| 11 | 11 | Antrel Rolle | 80.92 | 81.50 | 76.36 | 1126 | Giants |
| 12 | 12 | T.J. Ward | 80.82 | 77.90 | 80.89 | 1112 | Browns |
| 13 | 13 | Husain Abdullah | 80.14 | 78.50 | 79.46 | 339 | Chiefs |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Aaron Williams | 79.14 | 72.90 | 80.17 | 926 | Bills |
| 15 | 2 | Keith Tandy | 78.16 | 75.70 | 76.28 | 443 | Buccaneers |
| 16 | 3 | Chris Clemons | 77.76 | 77.70 | 76.33 | 1138 | Dolphins |
| 17 | 4 | George Wilson | 75.79 | 70.90 | 75.51 | 406 | Titans |
| 18 | 5 | Usama Young | 75.53 | 69.30 | 79.36 | 196 | Raiders |
| 19 | 6 | Rashad Johnson | 75.32 | 74.20 | 76.49 | 621 | Cardinals |
| 20 | 7 | Charles Woodson | 74.20 | 71.30 | 71.96 | 1068 | Raiders |

### Starter (44 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | David Bruton | 73.96 | 67.70 | 76.79 | 171 | Broncos |
| 22 | 2 | Eric Reid | 73.65 | 69.60 | 72.19 | 1159 | 49ers |
| 23 | 3 | Robert Lester | 73.03 | 71.30 | 75.21 | 331 | Panthers |
| 24 | 4 | Ryan Clark | 72.98 | 67.90 | 72.52 | 1063 | Steelers |
| 25 | 5 | Mike Mitchell | 72.84 | 73.10 | 69.44 | 964 | Panthers |
| 26 | 6 | Michael Griffin | 72.41 | 65.70 | 73.75 | 900 | Titans |
| 27 | 7 | Bernard Pollard | 71.90 | 63.50 | 73.34 | 1056 | Titans |
| 28 | 8 | Jahleel Addae | 71.61 | 67.80 | 69.99 | 496 | Chargers |
| 29 | 9 | Taylor Mays | 71.60 | 73.90 | 72.98 | 202 | Bengals |
| 30 | 10 | Jamarca Sanford | 71.58 | 69.80 | 70.37 | 793 | Vikings |
| 31 | 11 | Kenny Vaccaro | 71.41 | 64.00 | 74.27 | 793 | Saints |
| 32 | 12 | Rafael Bush | 71.13 | 68.00 | 75.62 | 592 | Saints |
| 33 | 13 | Yeremiah Bell | 71.03 | 65.30 | 70.69 | 1070 | Cardinals |
| 34 | 14 | Reggie Nelson | 70.86 | 63.80 | 71.72 | 1029 | Bengals |
| 35 | 15 | Nate Allen | 70.70 | 64.00 | 71.84 | 1174 | Eagles |
| 36 | 16 | Darrell Stuckey | 70.60 | 65.00 | 75.36 | 162 | Chargers |
| 37 | 17 | Darian Stewart | 70.18 | 72.30 | 69.80 | 568 | Rams |
| 38 | 18 | Duron Harmon | 69.58 | 66.00 | 68.84 | 430 | Patriots |
| 39 | 19 | Rahim Moore | 69.48 | 64.90 | 72.11 | 659 | Broncos |
| 40 | 20 | Kelcie McCray | 69.45 | 70.70 | 72.79 | 102 | Buccaneers |
| 41 | 21 | Chris Banjo | 69.31 | 69.70 | 68.01 | 193 | Packers |
| 42 | 22 | Shiloh Keo | 69.19 | 71.10 | 68.85 | 768 | Texans |
| 43 | 23 | Antoine Bethea | 68.64 | 57.40 | 71.97 | 1185 | Colts |
| 44 | 24 | Barry Church | 68.14 | 64.70 | 71.16 | 1018 | Cowboys |
| 45 | 25 | Malcolm Jenkins | 68.03 | 65.40 | 66.55 | 938 | Saints |
| 46 | 26 | Quintin Mikell | 67.70 | 60.50 | 68.85 | 706 | Panthers |
| 47 | 27 | Da'Norris Searcy | 67.60 | 62.30 | 69.36 | 728 | Bills |
| 48 | 28 | William Moore | 67.32 | 62.70 | 67.48 | 1043 | Falcons |
| 49 | 29 | LaRon Landry | 67.25 | 66.40 | 66.35 | 927 | Colts |
| 50 | 30 | Jim Leonhard | 67.19 | 67.70 | 63.84 | 612 | Bills |
| 51 | 31 | George Iloka | 66.94 | 61.60 | 66.33 | 1088 | Bengals |
| 52 | 32 | Louis Delmas | 66.93 | 67.50 | 65.72 | 1019 | Lions |
| 53 | 33 | Eddie Pleasant | 66.85 | 65.90 | 73.74 | 151 | Texans |
| 54 | 34 | Marcus Gilchrist | 66.78 | 56.90 | 69.20 | 1126 | Chargers |
| 55 | 35 | Roman Harper | 66.40 | 61.00 | 68.44 | 482 | Saints |
| 56 | 36 | Danieal Manning | 65.83 | 63.70 | 68.50 | 321 | Texans |
| 57 | 37 | Tashaun Gipson Sr. | 65.36 | 59.40 | 67.51 | 1099 | Browns |
| 58 | 38 | D.J. Swearinger Sr. | 64.96 | 59.20 | 64.64 | 797 | Texans |
| 59 | 39 | Mike Adams | 64.63 | 52.60 | 68.48 | 861 | Broncos |
| 60 | 40 | Jaiquawn Jarrett | 64.06 | 63.00 | 62.69 | 276 | Jets |
| 61 | 41 | Ryan Mundy | 63.54 | 55.50 | 67.96 | 654 | Giants |
| 62 | 42 | Robert Blanton | 63.07 | 61.60 | 68.22 | 395 | Vikings |
| 63 | 43 | Dawan Landry | 62.99 | 53.80 | 64.95 | 1083 | Jets |
| 64 | 44 | Steve Gregory | 62.86 | 54.10 | 65.37 | 971 | Patriots |

### Rotation/backup (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 65 | 1 | Reshad Jones | 61.58 | 49.90 | 65.61 | 1146 | Dolphins |
| 66 | 2 | T.J. McDonald | 60.09 | 59.40 | 62.64 | 650 | Rams |
| 67 | 3 | Quintin Demps | 59.87 | 54.20 | 61.98 | 653 | Chiefs |
| 68 | 4 | Morgan Burnett | 59.63 | 48.10 | 64.19 | 919 | Packers |
| 69 | 5 | Sean Richardson | 59.04 | 60.90 | 65.35 | 173 | Packers |
| 70 | 6 | Will Allen | 58.40 | 53.20 | 59.88 | 534 | Steelers |
| 71 | 7 | James Ihedigbo | 57.96 | 48.60 | 60.35 | 1072 | Ravens |
| 72 | 8 | Andrew Sendejo | 57.91 | 52.70 | 62.64 | 728 | Vikings |
| 73 | 9 | Dashon Goldson | 57.53 | 51.60 | 58.89 | 807 | Buccaneers |
| 74 | 10 | Harrison Smith | 56.65 | 49.80 | 62.25 | 529 | Vikings |
| 75 | 11 | Jeff Heath | 56.59 | 52.20 | 59.52 | 596 | Cowboys |
| 76 | 12 | Rodney McLeod | 56.40 | 54.70 | 58.84 | 1061 | Rams |
| 77 | 13 | Chris Conte | 56.38 | 50.30 | 57.84 | 1029 | Bears |
| 78 | 14 | Earl Wolff | 56.36 | 48.50 | 62.64 | 525 | Eagles |
| 79 | 15 | Duke Ihenacho | 55.94 | 46.30 | 58.20 | 923 | Broncos |
| 80 | 16 | Zeke Motta | 55.84 | 60.20 | 62.18 | 153 | Falcons |
| 81 | 17 | Tony Jefferson | 55.40 | 54.30 | 62.84 | 198 | Cardinals |
| 82 | 18 | M.D. Jennings | 54.96 | 46.20 | 59.77 | 835 | Packers |
| 83 | 19 | Craig Steltz | 54.52 | 57.20 | 56.70 | 121 | Bears |
| 84 | 20 | Reed Doughty | 54.52 | 47.60 | 56.22 | 409 | Commanders |
| 85 | 21 | Delano Howell | 54.39 | 56.70 | 60.94 | 206 | Colts |
| 86 | 22 | Kendrick Lewis | 54.37 | 43.00 | 59.97 | 1109 | Chiefs |
| 87 | 23 | Patrick Chung | 54.21 | 47.80 | 57.55 | 797 | Eagles |
| 88 | 24 | Charles Godfrey | 53.34 | 51.80 | 58.21 | 112 | Panthers |
| 89 | 25 | Brandon Meriweather | 53.10 | 51.10 | 57.56 | 738 | Commanders |
| 90 | 26 | Matt Giordano | 53.01 | 46.80 | 57.88 | 117 | Rams |
| 91 | 27 | Matt Elam | 52.90 | 39.70 | 57.53 | 1011 | Ravens |
| 92 | 28 | Omar Bolden | 52.51 | 49.80 | 52.10 | 207 | Broncos |
| 93 | 29 | J.J. Wilcox | 51.12 | 39.10 | 59.14 | 515 | Cowboys |
| 94 | 30 | Mistral Raymond | 51.00 | 47.30 | 57.21 | 203 | Vikings |
| 95 | 31 | Michael Huff | 50.13 | 40.70 | 57.25 | 163 | Broncos |
| 96 | 32 | Josh Evans | 49.96 | 39.50 | 53.80 | 675 | Jaguars |
| 97 | 33 | Thomas DeCoud | 49.86 | 36.40 | 55.19 | 891 | Falcons |
| 98 | 34 | Major Wright | 48.74 | 30.00 | 58.42 | 935 | Bears |
| 99 | 35 | Johnathan Cyprien | 48.43 | 28.80 | 58.38 | 1044 | Jaguars |
| 100 | 36 | Bacarri Rambo | 46.88 | 44.00 | 51.93 | 333 | Commanders |
| 101 | 37 | Winston Guy | 46.49 | 44.90 | 52.10 | 353 | Jaguars |

## T — Tackle

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Andrew Whitworth | 95.79 | 91.40 | 94.55 | 996 | Bengals |
| 2 | 2 | Trent Williams | 95.41 | 91.80 | 93.65 | 1143 | Commanders |
| 3 | 3 | Jason Peters | 94.02 | 90.70 | 92.06 | 1077 | Eagles |
| 4 | 4 | Jake Long | 94.01 | 90.50 | 92.19 | 852 | Rams |
| 5 | 5 | Tyron Smith | 93.70 | 89.00 | 92.67 | 997 | Cowboys |
| 6 | 6 | Joe Thomas | 92.34 | 90.30 | 89.53 | 1109 | Browns |
| 7 | 7 | King Dunlap | 92.01 | 87.70 | 90.71 | 779 | Chargers |
| 8 | 8 | Jordan Gross | 90.03 | 87.40 | 87.61 | 1058 | Panthers |
| 9 | 9 | Duane Brown | 89.89 | 84.70 | 89.19 | 953 | Texans |
| 10 | 10 | Michael Roos | 89.51 | 84.80 | 88.48 | 1075 | Titans |
| 11 | 11 | Joe Staley | 89.47 | 84.70 | 88.48 | 1114 | 49ers |
| 12 | 12 | Nate Solder | 89.15 | 84.90 | 87.81 | 1219 | Patriots |
| 13 | 13 | Phil Loadholt | 88.87 | 82.00 | 89.28 | 954 | Vikings |
| 14 | 14 | Eugene Monroe | 88.47 | 84.10 | 87.21 | 1031 | Ravens |
| 15 | 15 | Cordy Glenn | 88.21 | 82.40 | 87.91 | 1161 | Bills |
| 16 | 16 | Chris Clark | 87.99 | 82.30 | 87.61 | 1274 | Broncos |
| 17 | 17 | Sebastian Vollmer | 87.40 | 81.70 | 87.04 | 502 | Patriots |
| 18 | 18 | Andre Smith | 86.84 | 80.00 | 87.24 | 1179 | Bengals |
| 19 | 19 | Doug Free | 86.79 | 79.30 | 87.61 | 999 | Cowboys |
| 20 | 20 | Demar Dotson | 86.71 | 80.80 | 86.48 | 1037 | Buccaneers |
| 21 | 21 | Branden Albert | 85.03 | 77.20 | 86.09 | 858 | Chiefs |
| 22 | 22 | Michael Bowie | 84.77 | 75.80 | 86.58 | 584 | Seahawks |
| 23 | 23 | Zach Strief | 84.72 | 79.00 | 84.36 | 1183 | Saints |
| 24 | 24 | Donald Penn | 84.53 | 78.70 | 84.25 | 1036 | Buccaneers |
| 25 | 25 | Tyler Polumbus | 84.40 | 78.10 | 84.44 | 1145 | Commanders |
| 26 | 26 | Terron Armstead | 84.26 | 76.30 | 85.40 | 286 | Saints |
| 27 | 27 | Anthony Castonzo | 83.82 | 77.10 | 84.13 | 1197 | Colts |
| 28 | 28 | Lane Johnson | 83.82 | 74.90 | 85.60 | 1162 | Eagles |
| 29 | 29 | Mitchell Schwartz | 82.93 | 76.10 | 83.32 | 1110 | Browns |
| 30 | 30 | J'Marcus Webb | 82.85 | 75.60 | 83.51 | 106 | Vikings |
| 31 | 31 | Anthony Davis | 82.29 | 73.60 | 83.91 | 1161 | 49ers |
| 32 | 32 | Jermon Bushrod | 82.03 | 74.30 | 83.01 | 1059 | Bears |
| 33 | 33 | Gosder Cherilus | 81.81 | 74.10 | 82.79 | 1200 | Colts |
| 34 | 34 | D'Brickashaw Ferguson | 80.69 | 73.70 | 81.18 | 1052 | Jets |
| 35 | 35 | Matt Kalil | 80.65 | 72.90 | 81.65 | 1041 | Vikings |
| 36 | 36 | Anthony Collins | 80.33 | 77.10 | 78.32 | 660 | Bengals |

### Good (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 37 | 1 | Riley Reiff | 79.91 | 70.50 | 82.02 | 1107 | Lions |
| 38 | 2 | Austin Howard | 79.57 | 70.80 | 81.25 | 1050 | Jets |
| 39 | 3 | Will Beatty | 79.46 | 70.30 | 81.40 | 1004 | Giants |
| 40 | 4 | Joe Barksdale | 79.34 | 71.30 | 80.54 | 823 | Rams |
| 41 | 5 | LaAdrian Waddle | 79.14 | 75.70 | 77.26 | 540 | Lions |
| 42 | 6 | Jeremy Trueblood | 79.05 | 69.30 | 81.38 | 628 | Falcons |
| 43 | 7 | Marcus Cannon | 78.90 | 69.00 | 81.34 | 703 | Patriots |
| 44 | 8 | David Stewart | 78.60 | 70.30 | 79.96 | 800 | Titans |
| 45 | 9 | Russell Okung | 78.57 | 67.60 | 81.72 | 608 | Seahawks |
| 46 | 10 | Ryan Clady | 78.54 | 68.80 | 80.86 | 141 | Broncos |
| 47 | 11 | Breno Giacomini | 78.50 | 68.60 | 80.93 | 717 | Seahawks |
| 48 | 12 | Michael Oher | 78.36 | 70.10 | 79.70 | 1080 | Ravens |
| 49 | 13 | Marcus Gilbert | 78.23 | 68.30 | 80.68 | 950 | Steelers |
| 50 | 14 | Charles Brown | 78.14 | 67.50 | 81.06 | 950 | Saints |
| 51 | 15 | David Bakhtiari | 78.12 | 69.00 | 80.04 | 1171 | Packers |
| 52 | 16 | Tyson Clabo | 78.02 | 69.20 | 79.73 | 949 | Dolphins |
| 53 | 17 | Kelvin Beachum | 77.49 | 68.70 | 79.18 | 828 | Steelers |
| 54 | 18 | Erik Pears | 76.91 | 66.80 | 79.48 | 1161 | Bills |
| 55 | 19 | Donald Stephenson | 76.64 | 65.70 | 79.77 | 614 | Chiefs |
| 56 | 20 | Eric Winston | 76.39 | 64.60 | 80.08 | 1063 | Cardinals |
| 57 | 21 | Corey Hilliard | 75.87 | 65.50 | 78.61 | 455 | Lions |
| 58 | 22 | Jared Veldheer | 75.79 | 64.90 | 78.88 | 322 | Raiders |
| 59 | 23 | Matt McCants | 75.75 | 73.00 | 73.42 | 256 | Raiders |
| 60 | 24 | Ryan Harris | 75.66 | 65.10 | 78.53 | 479 | Texans |
| 61 | 25 | Jonathan Martin | 75.64 | 65.80 | 78.03 | 454 | Dolphins |
| 62 | 26 | Austin Pasztor | 75.54 | 65.10 | 78.33 | 791 | Jaguars |
| 63 | 27 | Byron Bell | 75.06 | 65.10 | 77.54 | 1069 | Panthers |
| 64 | 28 | Bryce Harris | 75.04 | 64.30 | 78.04 | 241 | Saints |
| 65 | 29 | Tony Pashos | 74.90 | 64.60 | 77.60 | 712 | Raiders |
| 66 | 30 | Jason Fox | 74.79 | 63.40 | 78.21 | 203 | Lions |
| 67 | 31 | Mike Adams | 74.53 | 64.00 | 77.38 | 480 | Steelers |
| 68 | 32 | Jeff Linkenbach | 74.36 | 62.70 | 77.96 | 398 | Colts |
| 69 | 33 | Khalif Barnes | 74.02 | 63.10 | 77.13 | 1033 | Raiders |

### Starter (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 70 | 1 | Don Barclay | 73.94 | 62.00 | 77.73 | 1027 | Packers |
| 71 | 2 | Derek Newton | 73.40 | 59.70 | 78.36 | 830 | Texans |
| 72 | 3 | Bryant McKinnie | 73.40 | 63.90 | 75.57 | 1021 | Dolphins |
| 73 | 4 | Xavier Nixon | 73.04 | 61.60 | 76.50 | 153 | Colts |
| 74 | 5 | Lamar Holmes | 72.93 | 60.50 | 77.05 | 1052 | Falcons |
| 75 | 6 | Levi Brown | 72.10 | 57.40 | 77.74 | 262 | Steelers |
| 76 | 7 | Eric Fisher | 71.83 | 57.80 | 77.02 | 789 | Chiefs |
| 77 | 8 | Menelik Watson | 71.37 | 57.60 | 76.39 | 173 | Raiders |
| 78 | 9 | Will Svitek | 71.26 | 58.60 | 75.54 | 239 | Patriots |
| 79 | 10 | Marshall Newhouse | 70.70 | 55.80 | 76.46 | 256 | Packers |
| 80 | 11 | Ryan Schraeder | 70.51 | 63.10 | 71.29 | 306 | Falcons |
| 81 | 12 | Jordan Mills | 69.94 | 55.80 | 75.20 | 1012 | Bears |
| 82 | 13 | Byron Stingily | 69.49 | 57.60 | 73.25 | 140 | Titans |
| 83 | 14 | Cameron Bradfield | 67.85 | 54.70 | 72.45 | 794 | Jaguars |
| 84 | 15 | Sam Baker | 65.79 | 50.00 | 72.15 | 189 | Falcons |
| 85 | 16 | Bradley Sowell | 64.68 | 50.10 | 70.24 | 825 | Cardinals |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 87.08 | 91.00 | 80.30 | 256 | Patriots |
| 2 | 2 | Ladarius Green | 85.76 | 84.90 | 82.17 | 190 | Chargers |
| 3 | 3 | Jordan Reed | 84.48 | 89.00 | 77.30 | 226 | Commanders |
| 4 | 4 | Zach Ertz | 81.64 | 80.20 | 78.43 | 291 | Eagles |
| 5 | 5 | Jimmy Graham | 81.55 | 81.30 | 77.55 | 667 | Saints |
| 6 | 6 | Benjamin Watson | 80.64 | 87.70 | 71.76 | 263 | Saints |

### Good (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Julius Thomas | 79.68 | 79.60 | 75.56 | 695 | Broncos |
| 8 | 2 | Jason Witten | 79.22 | 78.20 | 75.74 | 647 | Cowboys |
| 9 | 3 | Brent Celek | 78.92 | 78.80 | 74.84 | 426 | Eagles |
| 10 | 4 | Vernon Davis | 78.22 | 72.50 | 77.87 | 534 | 49ers |
| 11 | 5 | Greg Olsen | 78.05 | 79.00 | 73.25 | 550 | Panthers |
| 12 | 6 | Gary Barnidge | 77.14 | 74.50 | 74.74 | 197 | Browns |
| 13 | 7 | Lance Kendricks | 76.73 | 77.60 | 71.98 | 225 | Rams |
| 14 | 8 | Jacob Tamme | 76.52 | 72.70 | 74.90 | 175 | Broncos |
| 15 | 9 | Charles Clay | 76.31 | 76.30 | 72.15 | 516 | Dolphins |
| 16 | 10 | Kellen Winslow | 75.48 | 76.20 | 70.84 | 226 | Jets |
| 17 | 11 | Kyle Rudolph | 75.21 | 70.30 | 74.32 | 231 | Vikings |
| 18 | 12 | Joseph Fauria | 75.03 | 81.00 | 66.89 | 196 | Lions |
| 19 | 13 | Coby Fleener | 75.01 | 68.00 | 75.52 | 639 | Colts |
| 20 | 14 | Jared Cook | 75.00 | 65.20 | 77.36 | 421 | Rams |
| 21 | 15 | Tony Gonzalez | 74.99 | 75.50 | 70.48 | 707 | Falcons |
| 22 | 16 | Zach Miller | 74.74 | 73.60 | 71.33 | 468 | Seahawks |
| 23 | 17 | John Carlson | 74.72 | 78.50 | 68.03 | 224 | Vikings |
| 24 | 18 | Antonio Gates | 74.40 | 68.70 | 74.03 | 664 | Chargers |
| 25 | 19 | Marcedes Lewis | 74.36 | 66.60 | 75.36 | 360 | Jaguars |

### Starter (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Owen Daniels | 73.66 | 70.50 | 71.60 | 224 | Texans |
| 27 | 2 | Heath Miller | 73.64 | 68.70 | 72.76 | 561 | Steelers |
| 28 | 3 | Luke Willson | 73.27 | 67.90 | 72.69 | 230 | Seahawks |
| 29 | 4 | Tyler Eifert | 73.13 | 66.50 | 73.38 | 344 | Bengals |
| 30 | 5 | Dennis Pitta | 73.07 | 62.30 | 76.09 | 132 | Ravens |
| 31 | 6 | Gavin Escobar | 72.93 | 63.70 | 74.92 | 132 | Cowboys |
| 32 | 7 | Jeff Cumberland | 72.52 | 70.20 | 69.90 | 328 | Jets |
| 33 | 8 | Martellus Bennett | 72.27 | 75.10 | 66.21 | 596 | Bears |
| 34 | 9 | Jermichael Finley | 72.26 | 73.10 | 67.54 | 166 | Packers |
| 35 | 10 | Jordan Cameron | 72.22 | 70.60 | 69.13 | 668 | Browns |
| 36 | 11 | Mychal Rivera | 72.22 | 68.80 | 70.33 | 370 | Raiders |
| 37 | 12 | Craig Stevens | 72.07 | 65.80 | 72.08 | 133 | Titans |
| 38 | 13 | Tim Wright | 71.50 | 66.70 | 70.53 | 435 | Buccaneers |
| 39 | 14 | Lee Smith | 70.82 | 69.00 | 67.87 | 142 | Bills |
| 40 | 15 | Brandon Myers | 70.64 | 67.10 | 68.83 | 552 | Giants |
| 41 | 16 | Clay Harbor | 70.53 | 63.20 | 71.25 | 224 | Jaguars |
| 42 | 17 | Ed Dickson | 70.23 | 63.10 | 70.81 | 295 | Ravens |
| 43 | 18 | Ryan Griffin | 70.17 | 67.00 | 68.11 | 233 | Texans |
| 44 | 19 | Brandon Pettigrew | 70.07 | 64.00 | 69.95 | 528 | Lions |
| 45 | 20 | Jeron Mastrud | 69.79 | 59.90 | 72.21 | 228 | Raiders |
| 46 | 21 | Anthony Fasano | 69.73 | 62.80 | 70.18 | 372 | Chiefs |
| 47 | 22 | Andrew Quarless | 69.54 | 66.20 | 67.60 | 369 | Packers |
| 48 | 23 | Jim Dray | 69.35 | 66.20 | 67.28 | 290 | Cardinals |
| 49 | 24 | Fred Davis | 69.11 | 51.00 | 77.02 | 123 | Commanders |
| 50 | 25 | Garrett Graham | 69.09 | 64.10 | 68.25 | 487 | Texans |
| 51 | 26 | Scott Chandler | 68.90 | 64.80 | 67.46 | 545 | Bills |
| 52 | 27 | Delanie Walker | 68.86 | 68.00 | 65.26 | 508 | Titans |
| 53 | 28 | Jermaine Gresham | 68.68 | 61.70 | 69.16 | 547 | Bengals |
| 54 | 29 | Dante Rosario | 68.58 | 59.10 | 70.73 | 107 | Bears |
| 55 | 30 | Virgil Green | 68.51 | 64.10 | 67.28 | 116 | Broncos |
| 56 | 31 | Dallas Clark | 68.44 | 61.10 | 69.17 | 274 | Ravens |
| 57 | 32 | Matthew Mulligan | 68.42 | 66.70 | 65.40 | 117 | Patriots |
| 58 | 33 | Josh Hill | 68.15 | 58.50 | 70.42 | 105 | Saints |
| 59 | 34 | Michael Hoomanawanui | 67.65 | 58.50 | 69.58 | 402 | Patriots |
| 60 | 35 | Logan Paulsen | 66.21 | 58.30 | 67.31 | 387 | Commanders |
| 61 | 36 | Levine Toilolo | 66.13 | 62.80 | 64.18 | 110 | Falcons |
| 62 | 37 | James Hanna | 65.48 | 53.90 | 69.04 | 128 | Cowboys |
| 63 | 38 | Vance McDonald | 65.06 | 55.40 | 67.34 | 249 | 49ers |
| 64 | 39 | Rob Housler | 63.89 | 58.00 | 63.65 | 318 | Cardinals |
| 65 | 40 | David Paulson | 63.74 | 46.20 | 71.26 | 118 | Steelers |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 66 | 1 | Dion Sims | 59.49 | 54.90 | 58.39 | 127 | Dolphins |
| 67 | 2 | Allen Reisner | 56.86 | 46.30 | 59.73 | 141 | Jaguars |

## WR — Wide Receiver

- **Season used:** `2013`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Calvin Johnson | 88.51 | 89.90 | 83.42 | 565 | Lions |
| 2 | 2 | Anquan Boldin | 86.44 | 90.50 | 79.57 | 582 | 49ers |
| 3 | 3 | Antonio Brown | 86.24 | 89.10 | 80.17 | 627 | Steelers |
| 4 | 4 | Alshon Jeffery | 86.19 | 86.90 | 81.55 | 636 | Bears |
| 5 | 5 | Andre Johnson | 86.00 | 89.40 | 79.57 | 635 | Texans |
| 6 | 6 | Josh Gordon | 85.97 | 82.70 | 83.98 | 632 | Browns |
| 7 | 7 | Keenan Allen | 85.82 | 86.80 | 81.00 | 600 | Chargers |
| 8 | 8 | DeSean Jackson | 85.00 | 84.40 | 81.23 | 592 | Eagles |
| 9 | 9 | Jordy Nelson | 84.77 | 86.00 | 79.79 | 715 | Packers |
| 10 | 10 | Demaryius Thomas | 84.60 | 84.20 | 80.70 | 820 | Broncos |
| 11 | 11 | Brandon Marshall | 84.34 | 91.60 | 75.33 | 638 | Bears |
| 12 | 12 | Jerrel Jernigan | 83.31 | 86.50 | 77.01 | 151 | Giants |
| 13 | 13 | Marvin Jones Jr. | 83.12 | 87.20 | 76.24 | 444 | Bengals |
| 14 | 14 | Julio Jones | 82.96 | 80.80 | 80.24 | 222 | Falcons |
| 15 | 15 | Doug Baldwin | 82.72 | 82.70 | 78.56 | 524 | Seahawks |
| 16 | 16 | Vincent Jackson | 82.54 | 79.10 | 80.67 | 591 | Buccaneers |
| 17 | 17 | T.Y. Hilton | 82.47 | 82.00 | 78.62 | 659 | Colts |
| 18 | 18 | Marques Colston | 82.43 | 85.30 | 76.35 | 634 | Saints |
| 19 | 19 | A.J. Green | 82.17 | 81.40 | 78.51 | 703 | Bengals |
| 20 | 20 | Dez Bryant | 81.89 | 82.20 | 77.52 | 647 | Cowboys |
| 21 | 21 | Golden Tate | 81.57 | 79.90 | 78.52 | 551 | Seahawks |
| 22 | 22 | Pierre Garcon | 81.49 | 82.20 | 76.85 | 632 | Commanders |
| 23 | 23 | Michael Floyd | 81.22 | 82.10 | 76.47 | 602 | Cardinals |
| 24 | 24 | Steve Smith | 80.49 | 82.30 | 75.11 | 501 | Panthers |
| 25 | 25 | Larry Fitzgerald | 80.12 | 80.30 | 75.83 | 632 | Cardinals |
| 26 | 26 | Victor Cruz | 80.07 | 76.90 | 78.02 | 525 | Giants |
| 27 | 27 | Reggie Wayne | 80.07 | 81.00 | 75.28 | 274 | Colts |

### Good (47 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Andre Holmes | 79.96 | 74.50 | 79.43 | 261 | Raiders |
| 29 | 2 | Nate Washington | 79.61 | 77.90 | 76.58 | 584 | Titans |
| 30 | 3 | Ted Ginn Jr. | 79.30 | 75.10 | 77.94 | 388 | Panthers |
| 31 | 4 | Randall Cobb | 79.17 | 78.30 | 75.58 | 235 | Packers |
| 32 | 5 | Kendall Wright | 79.12 | 81.50 | 73.36 | 562 | Titans |
| 33 | 6 | Brian Hartline | 78.94 | 77.90 | 75.47 | 636 | Dolphins |
| 34 | 7 | Michael Crabtree | 78.69 | 72.70 | 78.51 | 242 | 49ers |
| 35 | 8 | Torrey Smith | 78.24 | 70.50 | 79.24 | 707 | Ravens |
| 36 | 9 | Rod Streater | 78.20 | 74.40 | 76.57 | 517 | Raiders |
| 37 | 10 | Eric Decker | 78.03 | 77.20 | 74.41 | 810 | Broncos |
| 38 | 11 | Denarius Moore | 77.96 | 73.70 | 76.63 | 388 | Raiders |
| 39 | 12 | Robert Meachem | 77.87 | 73.80 | 76.42 | 204 | Saints |
| 40 | 13 | Jeremy Kerley | 77.59 | 73.40 | 76.21 | 334 | Jets |
| 41 | 14 | Hakeem Nicks | 77.44 | 70.10 | 78.16 | 550 | Giants |
| 42 | 15 | Cordarrelle Patterson | 77.10 | 80.40 | 70.73 | 291 | Vikings |
| 43 | 16 | Jermaine Kearse | 77.04 | 74.20 | 74.77 | 305 | Seahawks |
| 44 | 17 | Julian Edelman | 77.01 | 81.20 | 70.05 | 702 | Patriots |
| 45 | 18 | Jarius Wright | 76.86 | 70.90 | 76.67 | 322 | Vikings |
| 46 | 19 | Riley Cooper | 76.62 | 71.80 | 75.67 | 594 | Eagles |
| 47 | 20 | Cole Beasley | 76.56 | 77.70 | 71.63 | 208 | Cowboys |
| 48 | 21 | Dwayne Bowe | 76.48 | 74.80 | 73.43 | 628 | Chiefs |
| 49 | 22 | Greg Jennings | 76.32 | 73.80 | 73.83 | 508 | Vikings |
| 50 | 23 | Jarrett Boykin | 76.23 | 73.10 | 74.15 | 454 | Packers |
| 51 | 24 | Eddie Royal | 76.21 | 74.50 | 73.19 | 509 | Chargers |
| 52 | 25 | Justin Blackmon | 76.12 | 71.90 | 74.77 | 163 | Jaguars |
| 53 | 26 | Jerome Simpson | 75.95 | 73.50 | 73.42 | 463 | Vikings |
| 54 | 27 | Cecil Shorts | 75.92 | 72.00 | 74.36 | 497 | Jaguars |
| 55 | 28 | Stedman Bailey | 75.91 | 70.90 | 75.09 | 124 | Rams |
| 56 | 29 | Jerricho Cotchery | 75.88 | 74.90 | 72.36 | 454 | Steelers |
| 57 | 30 | Wes Welker | 75.77 | 74.30 | 72.59 | 632 | Broncos |
| 58 | 31 | DeAndre Hopkins | 75.75 | 68.70 | 76.28 | 652 | Texans |
| 59 | 32 | Steve Johnson | 75.64 | 73.30 | 73.03 | 411 | Bills |
| 60 | 33 | Brandon Gibson | 75.58 | 79.20 | 69.00 | 201 | Dolphins |
| 61 | 34 | Justin Hunter | 75.30 | 68.50 | 75.67 | 219 | Titans |
| 62 | 35 | Rueben Randle | 75.27 | 69.20 | 75.15 | 439 | Giants |
| 63 | 36 | Terrance Williams | 75.15 | 66.20 | 76.95 | 521 | Cowboys |
| 64 | 37 | Aldrick Robinson | 75.08 | 66.90 | 76.36 | 273 | Commanders |
| 65 | 38 | Kenny Stills | 75.00 | 62.10 | 79.43 | 588 | Saints |
| 66 | 39 | Jacoby Jones | 74.92 | 69.30 | 74.50 | 377 | Ravens |
| 67 | 40 | Danny Amendola | 74.86 | 72.70 | 72.14 | 458 | Patriots |
| 68 | 41 | Sidney Rice | 74.86 | 68.00 | 75.27 | 213 | Seahawks |
| 69 | 42 | James Jones | 74.74 | 67.90 | 75.14 | 588 | Packers |
| 70 | 43 | Mike Wallace | 74.66 | 69.90 | 73.67 | 658 | Dolphins |
| 71 | 44 | Griff Whalen | 74.39 | 73.10 | 71.08 | 277 | Colts |
| 72 | 45 | Emmanuel Sanders | 74.26 | 71.30 | 72.06 | 556 | Steelers |
| 73 | 46 | Lance Moore | 74.12 | 69.40 | 73.10 | 403 | Saints |
| 74 | 47 | Jaron Brown | 74.01 | 67.80 | 73.98 | 119 | Cardinals |

### Starter (59 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 75 | 1 | Santonio Holmes | 73.81 | 67.60 | 73.79 | 304 | Jets |
| 76 | 2 | Leonard Hankerson | 73.75 | 68.20 | 73.28 | 275 | Commanders |
| 77 | 3 | Andrew Hawkins | 73.44 | 67.90 | 72.96 | 129 | Bengals |
| 78 | 4 | Brandon Stokley | 73.29 | 69.00 | 71.99 | 106 | Ravens |
| 79 | 5 | Harry Douglas | 73.28 | 69.80 | 71.43 | 674 | Falcons |
| 80 | 6 | Tiquan Underwood | 73.27 | 65.80 | 74.08 | 378 | Buccaneers |
| 81 | 7 | Chris Givens | 73.18 | 61.00 | 77.14 | 465 | Rams |
| 82 | 8 | David Nelson | 73.10 | 71.40 | 70.06 | 304 | Jets |
| 83 | 9 | LaVon Brazill | 72.91 | 64.70 | 74.22 | 260 | Colts |
| 84 | 10 | Rishard Matthews | 72.80 | 68.40 | 71.56 | 378 | Dolphins |
| 85 | 11 | Brandon LaFell | 72.65 | 66.20 | 72.79 | 582 | Panthers |
| 86 | 12 | Roddy White | 72.54 | 65.60 | 73.00 | 559 | Falcons |
| 87 | 13 | Damian Williams | 72.30 | 70.00 | 69.66 | 136 | Titans |
| 88 | 14 | Travis Benjamin | 71.78 | 59.10 | 76.07 | 107 | Browns |
| 89 | 15 | Tavon Austin | 71.67 | 70.00 | 68.61 | 318 | Rams |
| 90 | 16 | Aaron Dobson | 71.63 | 64.10 | 72.49 | 356 | Patriots |
| 91 | 17 | Kenbrell Thompkins | 71.55 | 64.40 | 72.15 | 387 | Patriots |
| 92 | 18 | Brian Quick | 71.44 | 62.40 | 73.30 | 234 | Rams |
| 93 | 19 | Robert Woods | 71.30 | 63.90 | 72.06 | 516 | Bills |
| 94 | 20 | Tandon Doss | 71.14 | 62.80 | 72.54 | 222 | Ravens |
| 95 | 21 | Mike Williams | 70.86 | 62.20 | 72.47 | 240 | Buccaneers |
| 96 | 22 | Da'Rick Rogers | 70.72 | 58.40 | 74.77 | 171 | Colts |
| 97 | 23 | Kevin Ogletree | 69.91 | 63.40 | 70.09 | 232 | Lions |
| 98 | 24 | Vincent Brown | 69.77 | 63.60 | 69.71 | 580 | Chargers |
| 99 | 25 | Andre Roberts | 69.63 | 66.20 | 67.75 | 430 | Cardinals |
| 100 | 26 | Nate Burleson | 69.56 | 65.00 | 68.44 | 305 | Lions |
| 101 | 27 | A.J. Jenkins | 69.52 | 59.10 | 72.30 | 143 | Chiefs |
| 102 | 28 | Mario Manningham | 69.45 | 61.30 | 70.72 | 108 | 49ers |
| 103 | 29 | Miles Austin | 69.21 | 58.50 | 72.18 | 345 | Cowboys |
| 104 | 30 | Austin Pettis | 69.10 | 66.70 | 66.54 | 399 | Rams |
| 105 | 31 | Jason Avant | 69.01 | 61.20 | 70.05 | 521 | Eagles |
| 106 | 32 | Andre Caldwell | 68.91 | 69.00 | 64.68 | 176 | Broncos |
| 107 | 33 | Jacoby Ford | 68.87 | 55.70 | 73.49 | 159 | Raiders |
| 108 | 34 | Marlon Brown | 68.64 | 65.70 | 66.44 | 542 | Ravens |
| 109 | 35 | Donnie Avery | 68.33 | 58.60 | 70.65 | 525 | Chiefs |
| 110 | 36 | Kerry Taylor | 68.18 | 65.00 | 66.14 | 202 | Jaguars |
| 111 | 37 | Earl Bennett | 68.17 | 62.80 | 67.58 | 385 | Bears |
| 112 | 38 | Josh Morgan | 68.03 | 63.70 | 66.75 | 190 | Commanders |
| 113 | 39 | Mike Brown | 67.97 | 56.90 | 71.18 | 378 | Jaguars |
| 114 | 40 | Deonte Thompson | 67.73 | 61.30 | 67.85 | 104 | Ravens |
| 115 | 41 | Ace Sanders | 67.72 | 61.90 | 67.44 | 436 | Jaguars |
| 116 | 42 | Stephen Hill | 67.62 | 60.60 | 68.13 | 349 | Jets |
| 117 | 43 | Josh Boyce | 67.35 | 60.80 | 67.55 | 128 | Patriots |
| 118 | 44 | DeVier Posey | 67.27 | 59.20 | 68.49 | 172 | Texans |
| 119 | 45 | Mohamed Sanu | 67.01 | 63.00 | 65.51 | 505 | Bengals |
| 120 | 46 | Santana Moss | 66.65 | 58.10 | 68.18 | 468 | Commanders |
| 121 | 47 | T.J. Graham | 66.15 | 57.10 | 68.02 | 496 | Bills |
| 122 | 48 | Junior Hemingway | 66.14 | 56.10 | 68.67 | 167 | Chiefs |
| 123 | 49 | Darrius Heyward-Bey | 66.13 | 54.00 | 70.05 | 410 | Colts |
| 124 | 50 | Kyle Williams | 65.10 | 52.90 | 69.06 | 197 | Chiefs |
| 125 | 51 | Chris Hogan | 64.99 | 57.90 | 65.55 | 126 | Bills |
| 126 | 52 | Jeremy Ross | 64.05 | 61.70 | 61.45 | 101 | Lions |
| 127 | 53 | Ryan Broyles | 64.00 | 50.00 | 69.17 | 115 | Lions |
| 128 | 54 | Kris Durham | 63.92 | 55.40 | 65.43 | 586 | Lions |
| 129 | 55 | Chris Owusu | 63.63 | 53.50 | 66.22 | 204 | Buccaneers |
| 130 | 56 | Keshawn Martin | 63.26 | 56.30 | 63.74 | 296 | Texans |
| 131 | 57 | Greg Little | 63.14 | 53.60 | 65.34 | 691 | Browns |
| 132 | 58 | Davone Bess | 62.77 | 54.00 | 64.45 | 447 | Browns |
| 133 | 59 | Nick Toon | 62.17 | 51.70 | 64.98 | 136 | Saints |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 134 | 1 | Darius Johnson | 61.83 | 55.20 | 62.09 | 308 | Falcons |
| 135 | 2 | Markus Wheaton | 61.77 | 55.10 | 62.05 | 109 | Steelers |
| 136 | 3 | Brice Butler | 61.61 | 56.70 | 60.72 | 145 | Raiders |
| 137 | 4 | Kenny Britt | 59.88 | 44.10 | 66.24 | 215 | Titans |
