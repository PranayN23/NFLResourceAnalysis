# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:38:13Z
- **Requested analysis_year:** 2016 (clamped to 2016)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Alex Mack | 96.01 | 90.80 | 95.31 | 1204 | Falcons |
| 2 | 2 | Matt Paradis | 95.69 | 90.20 | 95.19 | 1073 | Broncos |
| 3 | 3 | Travis Frederick | 93.56 | 86.30 | 94.23 | 1126 | Cowboys |
| 4 | 4 | Rodney Hudson | 91.62 | 83.70 | 92.73 | 1180 | Raiders |
| 5 | 5 | Brandon Linder | 88.27 | 80.94 | 88.99 | 909 | Jaguars |
| 6 | 6 | Maurkice Pouncey | 85.42 | 78.10 | 86.13 | 1151 | Steelers |
| 7 | 7 | Weston Richburg | 85.31 | 77.60 | 86.28 | 1110 | Giants |
| 8 | 8 | J.C. Tretter | 84.26 | 72.81 | 87.73 | 488 | Packers |
| 9 | 9 | Corey Linsley | 83.65 | 75.44 | 84.96 | 802 | Packers |
| 10 | 10 | Greg Mancz | 83.55 | 75.70 | 84.61 | 1261 | Texans |
| 11 | 11 | Max Unger | 83.26 | 75.10 | 84.53 | 1091 | Saints |
| 12 | 12 | Mitch Morse | 82.93 | 74.60 | 84.31 | 1075 | Chiefs |
| 13 | 13 | Justin Britt | 82.80 | 81.60 | 79.43 | 1121 | Seahawks |
| 14 | 14 | Ben Jones | 81.87 | 72.90 | 83.69 | 1061 | Titans |
| 15 | 15 | Jason Kelce | 81.74 | 71.40 | 84.46 | 1132 | Eagles |
| 16 | 16 | Travis Swanson | 81.72 | 72.88 | 83.44 | 766 | Lions |
| 17 | 17 | A.Q. Shipley | 81.24 | 71.90 | 83.30 | 1148 | Cardinals |
| 18 | 18 | Jeremy Zuttah | 81.03 | 73.20 | 82.08 | 1109 | Ravens |
| 19 | 19 | Joe Hawley | 80.42 | 71.30 | 82.34 | 1956 | Buccaneers |
| 20 | 20 | Ryan Kalil | 80.37 | 69.87 | 83.21 | 504 | Panthers |
| 21 | 21 | David Andrews | 80.26 | 71.00 | 82.26 | 1355 | Patriots |
| 22 | 22 | Kory Lichtensteiger | 80.03 | 64.10 | 86.48 | 159 | Commanders |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Spencer Long | 79.30 | 69.94 | 81.37 | 804 | Commanders |
| 24 | 2 | Russell Bodine | 78.79 | 69.50 | 80.82 | 1060 | Bengals |
| 25 | 3 | Ryan Kelly | 78.16 | 72.40 | 77.84 | 1018 | Colts |
| 26 | 4 | Evan Smith | 78.09 | 66.81 | 81.44 | 177 | Buccaneers |
| 27 | 5 | Daniel Kilgore | 76.60 | 65.75 | 79.66 | 794 | 49ers |
| 28 | 6 | Nick Mangold | 76.29 | 65.35 | 79.42 | 433 | Jets |
| 29 | 7 | Tim Barnes | 75.24 | 64.80 | 78.04 | 1004 | Rams |
| 30 | 8 | Mike Pouncey | 74.92 | 64.17 | 77.92 | 301 | Dolphins |
| 31 | 9 | Wesley Johnson | 74.28 | 62.70 | 77.83 | 657 | Jets |

### Starter (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 32 | 1 | Gino Gradkowski | 72.54 | 60.34 | 76.50 | 235 | Panthers |
| 33 | 2 | Eric Wood | 72.17 | 61.28 | 75.27 | 570 | Bills |
| 34 | 3 | Nick Easton | 65.31 | 57.67 | 66.24 | 414 | Vikings |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Harris Jr. | 92.18 | 90.60 | 89.06 | 1092 | Broncos |
| 2 | 2 | Dominique Rodgers-Cromartie | 91.94 | 89.58 | 89.66 | 736 | Giants |
| 3 | 3 | A.J. Bouye | 91.71 | 89.90 | 90.10 | 859 | Texans |
| 4 | 4 | Brent Grimes | 91.52 | 87.40 | 90.42 | 998 | Buccaneers |
| 5 | 5 | Aqib Talib | 91.31 | 89.80 | 89.72 | 870 | Broncos |
| 6 | 6 | Malcolm Butler | 89.62 | 86.90 | 88.73 | 1192 | Patriots |
| 7 | 7 | Casey Hayward Jr. | 88.15 | 83.60 | 87.02 | 988 | Chargers |
| 8 | 8 | Janoris Jenkins | 85.92 | 83.20 | 84.30 | 1028 | Giants |
| 9 | 9 | Marcus Peters | 85.58 | 79.20 | 85.67 | 1073 | Chiefs |
| 10 | 10 | Terence Newman | 85.18 | 82.39 | 83.80 | 754 | Vikings |
| 11 | 11 | Richard Sherman | 84.56 | 76.90 | 85.50 | 1175 | Seahawks |
| 12 | 12 | Darius Slay | 84.08 | 80.02 | 83.66 | 801 | Lions |
| 13 | 13 | Logan Ryan | 83.42 | 76.20 | 84.07 | 1081 | Patriots |
| 14 | 14 | Patrick Peterson | 83.28 | 80.10 | 81.24 | 1035 | Cardinals |
| 15 | 15 | Josh Norman | 82.84 | 76.60 | 83.03 | 1059 | Commanders |
| 16 | 16 | William Gay | 81.68 | 79.20 | 79.16 | 971 | Steelers |
| 17 | 17 | Byron Maxwell | 80.41 | 74.50 | 82.58 | 846 | Dolphins |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Jalen Ramsey | 79.82 | 73.10 | 80.13 | 1061 | Jaguars |
| 19 | 2 | Sean Smith | 79.62 | 74.30 | 79.32 | 958 | Raiders |
| 20 | 3 | Xavier Rhodes | 79.39 | 72.70 | 80.72 | 787 | Vikings |
| 21 | 4 | Tavon Young | 79.13 | 75.60 | 77.32 | 833 | Ravens |
| 22 | 5 | Nickell Robey-Coleman | 77.91 | 72.60 | 77.29 | 573 | Bills |
| 23 | 6 | Trumaine Johnson | 77.70 | 73.90 | 79.20 | 954 | Rams |
| 24 | 7 | Kevin Johnson | 76.93 | 72.33 | 82.34 | 287 | Texans |
| 25 | 8 | Kareem Jackson | 76.65 | 71.69 | 77.36 | 822 | Texans |
| 26 | 9 | Adam Jones | 76.64 | 72.00 | 75.88 | 1058 | Bengals |
| 27 | 10 | Ross Cockrell | 76.25 | 70.10 | 79.10 | 1217 | Steelers |
| 28 | 11 | Terrance Mitchell | 76.05 | 68.79 | 85.37 | 295 | Chiefs |
| 29 | 12 | Artie Burns | 75.80 | 69.20 | 76.03 | 991 | Steelers |
| 30 | 13 | Dre Kirkpatrick | 75.63 | 71.00 | 75.48 | 978 | Bengals |
| 31 | 14 | James Bradberry | 75.30 | 70.04 | 77.78 | 799 | Panthers |
| 32 | 15 | Robert Alford | 75.25 | 67.20 | 78.02 | 1297 | Falcons |
| 33 | 16 | Jalen Collins | 75.01 | 72.03 | 79.08 | 636 | Falcons |
| 34 | 17 | Morris Claiborne | 74.85 | 72.81 | 80.28 | 432 | Cowboys |
| 35 | 18 | Orlando Scandrick | 74.57 | 69.45 | 75.39 | 704 | Cowboys |
| 36 | 19 | Jamar Taylor | 74.42 | 77.70 | 70.66 | 921 | Browns |
| 37 | 20 | Tramaine Brock Sr. | 74.41 | 67.70 | 77.73 | 1099 | 49ers |

### Starter (78 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 38 | 1 | Desmond Trufant | 73.84 | 64.54 | 79.53 | 591 | Falcons |
| 39 | 2 | Cre'Von LeBlanc | 73.76 | 65.42 | 78.28 | 696 | Bears |
| 40 | 3 | Johnathan Joseph | 73.62 | 64.06 | 76.35 | 737 | Texans |
| 41 | 4 | Captain Munnerlyn | 73.30 | 65.84 | 74.62 | 636 | Vikings |
| 42 | 5 | David Amerson | 72.96 | 65.10 | 74.55 | 1031 | Raiders |
| 43 | 6 | Brice McCain | 72.46 | 65.80 | 73.77 | 844 | Titans |
| 44 | 7 | Deji Olatoye | 72.39 | 66.28 | 83.23 | 125 | Falcons |
| 45 | 8 | Anthony Brown | 72.37 | 64.46 | 73.48 | 742 | Cowboys |
| 46 | 9 | Daryl Worley | 72.32 | 63.50 | 75.06 | 865 | Panthers |
| 47 | 10 | Brandon Carr | 71.97 | 63.70 | 73.31 | 1077 | Cowboys |
| 48 | 11 | Leon Hall | 71.95 | 64.57 | 74.58 | 455 | Giants |
| 49 | 12 | Ronald Darby | 71.65 | 61.30 | 76.09 | 822 | Bills |
| 50 | 13 | Darrelle Revis | 71.60 | 62.30 | 74.79 | 922 | Jets |
| 51 | 14 | Jimmy Smith | 71.59 | 65.06 | 76.05 | 583 | Ravens |
| 52 | 15 | Alterraun Verner | 71.58 | 64.35 | 77.14 | 241 | Buccaneers |
| 53 | 16 | T.J. Carrie | 71.50 | 63.99 | 74.00 | 378 | Raiders |
| 54 | 17 | Briean Boddy-Calhoun | 71.41 | 60.58 | 76.54 | 571 | Browns |
| 55 | 18 | Tramon Williams | 71.20 | 63.41 | 74.63 | 625 | Browns |
| 56 | 19 | Jason McCourty | 71.19 | 63.49 | 76.94 | 814 | Titans |
| 57 | 20 | Harlan Miller | 71.16 | 67.54 | 90.91 | 140 | Cardinals |
| 58 | 21 | Bradley Roby | 71.04 | 62.93 | 72.28 | 684 | Broncos |
| 59 | 22 | Sterling Moore | 71.01 | 63.65 | 73.31 | 805 | Saints |
| 60 | 23 | Kevon Seymour | 70.89 | 59.94 | 75.05 | 286 | Bills |
| 61 | 24 | Bobby McCain | 70.89 | 64.53 | 73.32 | 650 | Dolphins |
| 62 | 25 | Rashard Robinson | 70.73 | 60.65 | 75.37 | 543 | 49ers |
| 63 | 26 | Nevin Lawson | 70.71 | 67.80 | 72.96 | 994 | Lions |
| 64 | 27 | Prince Amukamara | 70.44 | 63.80 | 74.97 | 873 | Jaguars |
| 65 | 28 | Trae Waynes | 70.29 | 59.66 | 75.81 | 581 | Vikings |
| 66 | 29 | Aaron Colvin | 70.22 | 66.66 | 73.62 | 292 | Jaguars |
| 67 | 30 | Bashaud Breeland | 70.16 | 60.48 | 73.48 | 766 | Commanders |
| 68 | 31 | Eric Rowe | 69.95 | 62.36 | 75.80 | 587 | Patriots |
| 69 | 32 | Stephon Gilmore | 69.75 | 61.50 | 73.26 | 982 | Bills |
| 70 | 33 | Nolan Carroll | 69.72 | 61.50 | 72.60 | 912 | Eagles |
| 71 | 34 | Juston Burris | 69.58 | 61.90 | 78.86 | 187 | Jets |
| 72 | 35 | Brandon Flowers | 69.51 | 64.43 | 75.91 | 352 | Chargers |
| 73 | 36 | Justin Bethel | 69.51 | 61.26 | 76.26 | 270 | Cardinals |
| 74 | 37 | Tony Lippett | 69.41 | 63.00 | 75.12 | 917 | Dolphins |
| 75 | 38 | Quinton Dunbar | 69.09 | 59.46 | 76.67 | 301 | Commanders |
| 76 | 39 | Dontae Johnson | 68.90 | 60.04 | 74.61 | 101 | 49ers |
| 77 | 40 | Marcus Williams | 68.88 | 59.85 | 75.42 | 455 | Jets |
| 78 | 41 | Josh Shaw | 68.78 | 64.93 | 69.91 | 618 | Bengals |
| 79 | 42 | Cyrus Jones | 68.76 | 62.70 | 75.94 | 147 | Patriots |
| 80 | 43 | Jerraud Powers | 68.69 | 60.52 | 71.85 | 510 | Ravens |
| 81 | 44 | Steven Nelson | 68.56 | 61.90 | 71.96 | 1077 | Chiefs |
| 82 | 45 | Leodis McKelvin | 67.93 | 61.45 | 73.09 | 586 | Eagles |
| 83 | 46 | Joe Haden | 67.86 | 60.00 | 74.13 | 854 | Browns |
| 84 | 47 | Josh Johnson | 67.54 | 64.23 | 81.54 | 134 | Jaguars |
| 85 | 48 | Darryl Morris | 67.51 | 62.44 | 74.42 | 359 | Colts |
| 86 | 49 | Rashaan Melvin | 67.42 | 63.21 | 70.54 | 655 | Colts |
| 87 | 50 | Quinten Rollins | 67.41 | 56.73 | 72.45 | 722 | Packers |
| 88 | 51 | Bryce Callahan | 66.86 | 61.31 | 72.77 | 489 | Bears |
| 89 | 52 | Eli Apple | 66.81 | 59.81 | 68.34 | 772 | Giants |
| 90 | 53 | Xavien Howard | 66.34 | 60.67 | 74.29 | 582 | Dolphins |
| 91 | 54 | Johnson Bademosi | 66.16 | 63.57 | 72.68 | 283 | Lions |
| 92 | 55 | Trovon Reed | 65.77 | 65.20 | 80.48 | 123 | Chargers |
| 93 | 56 | Buster Skrine | 65.57 | 56.02 | 68.81 | 816 | Jets |
| 94 | 57 | Keith Reaser | 65.51 | 59.67 | 68.62 | 351 | 49ers |
| 95 | 58 | Darqueze Dennard | 65.39 | 58.92 | 69.60 | 330 | Bengals |
| 96 | 59 | Vernon Hargreaves III | 65.37 | 56.40 | 67.18 | 1038 | Buccaneers |
| 97 | 60 | Jason Verrett | 65.30 | 60.81 | 73.10 | 260 | Chargers |
| 98 | 61 | Vontae Davis | 65.29 | 52.81 | 70.47 | 822 | Colts |
| 99 | 62 | Shareece Wright | 65.24 | 59.82 | 68.76 | 673 | Ravens |
| 100 | 63 | Corey White | 65.13 | 61.48 | 69.65 | 412 | Bills |
| 101 | 64 | Valentino Blake | 64.48 | 55.86 | 70.22 | 367 | Titans |
| 102 | 65 | Patrick Robinson | 64.46 | 56.23 | 70.88 | 402 | Colts |
| 103 | 66 | Justin Coleman | 64.44 | 60.21 | 70.91 | 225 | Patriots |
| 104 | 67 | Marcus Cooper | 63.95 | 55.60 | 71.09 | 827 | Cardinals |
| 105 | 68 | Kenneth Acker | 63.87 | 59.49 | 70.57 | 147 | Chiefs |
| 106 | 69 | Darryl Roberts | 63.66 | 57.41 | 71.99 | 286 | Jets |
| 107 | 70 | D.J. Hayden | 63.20 | 54.30 | 68.81 | 476 | Raiders |
| 108 | 71 | Davon House | 63.17 | 50.35 | 69.95 | 272 | Jaguars |
| 109 | 72 | Johnthan Banks | 62.84 | 60.12 | 68.51 | 133 | Bears |
| 110 | 73 | Ken Crawley | 62.81 | 53.75 | 67.81 | 503 | Saints |
| 111 | 74 | Neiko Thorpe | 62.71 | 58.60 | 68.59 | 105 | Seahawks |
| 112 | 75 | Robert Nelson | 62.37 | 57.86 | 70.98 | 196 | Texans |
| 113 | 76 | Coty Sensabaugh | 62.26 | 54.47 | 67.03 | 257 | Giants |
| 114 | 77 | Greg Toler | 62.25 | 55.43 | 67.11 | 256 | Commanders |
| 115 | 78 | Jeremy Lane | 62.18 | 54.60 | 67.24 | 871 | Seahawks |

### Rotation/backup (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 116 | 1 | Trevin Wade | 61.90 | 53.44 | 66.50 | 393 | Giants |
| 117 | 2 | Delvin Breaux Sr. | 61.32 | 50.20 | 71.08 | 296 | Saints |
| 118 | 3 | LaDarius Gunter | 61.12 | 50.30 | 68.47 | 1055 | Packers |
| 119 | 4 | De'Vante Harris | 59.36 | 60.68 | 70.28 | 108 | Saints |
| 120 | 5 | E.J. Gaines | 59.14 | 45.59 | 67.65 | 614 | Rams |
| 121 | 6 | Trevor Williams | 58.91 | 53.00 | 65.99 | 389 | Chargers |
| 122 | 7 | B.W. Webb | 58.67 | 55.36 | 62.96 | 588 | Saints |
| 123 | 8 | LeShaun Sims | 58.62 | 57.49 | 66.08 | 236 | Titans |
| 124 | 9 | Tracy Porter | 58.50 | 46.00 | 66.00 | 944 | Bears |
| 125 | 10 | Robert McClain | 57.17 | 50.28 | 63.84 | 327 | Chargers |
| 126 | 11 | Leonard Johnson | 57.12 | 50.69 | 64.10 | 436 | Panthers |
| 127 | 12 | Kendall Fuller | 56.82 | 56.13 | 57.28 | 476 | Commanders |
| 128 | 13 | Charles James II | 56.43 | 48.94 | 65.08 | 164 | Colts |
| 129 | 14 | Ron Brooks | 56.32 | 54.93 | 63.18 | 235 | Eagles |
| 130 | 15 | Jude Adjei-Barimah | 56.27 | 58.28 | 56.50 | 290 | Buccaneers |
| 131 | 16 | Phillip Gaines | 55.78 | 46.50 | 66.14 | 449 | Chiefs |
| 132 | 17 | Demetri Goodson | 55.34 | 54.65 | 59.86 | 182 | Packers |
| 133 | 18 | Chris Milton | 55.29 | 59.16 | 67.04 | 106 | Colts |
| 134 | 19 | D.J. White | 54.38 | 55.50 | 60.34 | 138 | Chiefs |
| 135 | 20 | Javien Elliott | 54.12 | 60.19 | 61.88 | 184 | Buccaneers |
| 136 | 21 | Brandon Williams | 53.53 | 50.92 | 64.52 | 241 | Cardinals |
| 137 | 22 | Craig Mager | 53.39 | 48.37 | 60.38 | 410 | Chargers |
| 138 | 23 | Mike Jordan | 50.30 | 58.10 | 59.44 | 177 | Rams |
| 139 | 24 | Troy Hill | 48.85 | 50.53 | 53.60 | 338 | Rams |
| 140 | 25 | Dashaun Phillips | 46.93 | 50.85 | 57.15 | 148 | Commanders |

## DI — Defensive Interior

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 94.29 | 89.96 | 93.01 | 830 | Rams |
| 2 | 2 | Leonard Williams | 89.40 | 89.83 | 84.95 | 898 | Jets |
| 3 | 3 | Geno Atkins | 88.75 | 86.69 | 85.96 | 781 | Bengals |
| 4 | 4 | J.J. Watt | 88.36 | 77.95 | 97.90 | 157 | Texans |
| 5 | 5 | Calais Campbell | 88.32 | 86.78 | 85.38 | 831 | Cardinals |
| 6 | 6 | Kawann Short | 87.99 | 85.99 | 85.15 | 779 | Panthers |
| 7 | 7 | Ndamukong Suh | 86.02 | 86.50 | 81.53 | 1028 | Dolphins |
| 8 | 8 | Jurrell Casey | 85.26 | 85.73 | 81.29 | 728 | Titans |
| 9 | 9 | Fletcher Cox | 85.22 | 89.16 | 78.42 | 772 | Eagles |
| 10 | 10 | Malik Jackson | 85.19 | 83.19 | 82.36 | 718 | Jaguars |
| 11 | 11 | Mike Daniels | 85.14 | 83.53 | 82.04 | 797 | Packers |
| 12 | 12 | Damon Harrison Sr. | 84.71 | 83.50 | 81.35 | 720 | Giants |
| 13 | 13 | Michael Pierce | 84.59 | 76.28 | 85.97 | 375 | Ravens |
| 14 | 14 | Linval Joseph | 82.62 | 84.98 | 77.82 | 718 | Vikings |
| 15 | 15 | Marcell Dareus | 80.71 | 82.11 | 80.30 | 417 | Bills |
| 16 | 16 | Gerald McCoy | 80.64 | 85.56 | 74.66 | 796 | Buccaneers |
| 17 | 17 | Akiem Hicks | 80.59 | 77.29 | 78.82 | 931 | Bears |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | DeForest Buckner | 79.51 | 82.80 | 74.19 | 1006 | 49ers |
| 19 | 2 | Chris Baker | 79.27 | 71.45 | 80.52 | 782 | Commanders |
| 20 | 3 | Deon Simon | 79.09 | 68.80 | 82.81 | 204 | Jets |
| 21 | 4 | Derek Wolfe | 78.72 | 78.15 | 76.28 | 663 | Broncos |
| 22 | 5 | Danny Shelton | 78.63 | 80.97 | 72.90 | 745 | Browns |
| 23 | 6 | Brandon Williams | 78.08 | 71.65 | 78.20 | 635 | Ravens |
| 24 | 7 | David Irving | 77.75 | 73.93 | 77.70 | 530 | Cowboys |
| 25 | 8 | Dan Williams | 77.29 | 75.31 | 74.45 | 397 | Raiders |
| 26 | 9 | Malcom Brown | 77.03 | 69.71 | 77.74 | 709 | Patriots |
| 27 | 10 | Timmy Jernigan | 76.77 | 66.80 | 80.18 | 631 | Ravens |
| 28 | 11 | Abry Jones | 76.39 | 69.84 | 77.42 | 462 | Jaguars |
| 29 | 12 | Dominique Easley | 76.37 | 74.21 | 76.25 | 470 | Rams |
| 30 | 13 | Grady Jarrett | 76.36 | 64.03 | 80.79 | 763 | Falcons |
| 31 | 14 | Lawrence Guy Sr. | 76.31 | 66.74 | 78.53 | 487 | Ravens |
| 32 | 15 | Dean Lowry | 75.77 | 67.24 | 77.29 | 211 | Packers |
| 33 | 16 | Ra'Shede Hageman | 75.56 | 63.67 | 79.83 | 353 | Falcons |
| 34 | 17 | Michael Brockers | 75.26 | 75.71 | 71.83 | 419 | Rams |
| 35 | 18 | Brent Urban | 75.26 | 65.61 | 81.43 | 150 | Ravens |
| 36 | 19 | Nick Fairley | 75.23 | 63.89 | 80.60 | 722 | Saints |
| 37 | 20 | Eddie Goldman | 75.11 | 72.48 | 79.60 | 197 | Bears |
| 38 | 21 | Karl Klug | 74.97 | 69.00 | 75.81 | 398 | Titans |
| 39 | 22 | Cameron Heyward | 74.70 | 71.65 | 77.25 | 363 | Steelers |
| 40 | 23 | Sheldon Day | 74.57 | 58.13 | 81.36 | 203 | Jaguars |
| 41 | 24 | Kyle Williams | 74.50 | 63.94 | 81.23 | 794 | Bills |
| 42 | 25 | Bennie Logan | 74.14 | 60.54 | 81.22 | 467 | Eagles |

### Starter (87 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 43 | 1 | Johnathan Hankins | 73.95 | 66.57 | 76.88 | 816 | Giants |
| 44 | 2 | Kenny Clark | 73.71 | 70.73 | 71.53 | 410 | Packers |
| 45 | 3 | Stephon Tuitt | 72.89 | 65.98 | 73.33 | 945 | Steelers |
| 46 | 4 | Jaye Howard Jr. | 72.49 | 59.09 | 81.43 | 360 | Chiefs |
| 47 | 5 | Alan Branch | 72.20 | 64.85 | 73.96 | 760 | Patriots |
| 48 | 6 | Dan McCullers | 72.11 | 65.08 | 74.81 | 206 | Steelers |
| 49 | 7 | Stephen Paea | 71.85 | 67.47 | 73.73 | 320 | Browns |
| 50 | 8 | Henry Anderson | 71.84 | 65.91 | 77.61 | 308 | Colts |
| 51 | 9 | DJ Reader | 71.82 | 65.41 | 71.93 | 477 | Texans |
| 52 | 10 | Leger Douzable | 71.39 | 55.04 | 78.12 | 481 | Bills |
| 53 | 11 | Brandon Mebane | 71.38 | 67.20 | 74.58 | 340 | Chargers |
| 54 | 12 | Corey Liuget | 71.30 | 62.16 | 74.80 | 812 | Chargers |
| 55 | 13 | Adolphus Washington | 70.86 | 58.19 | 76.18 | 331 | Bills |
| 56 | 14 | Steve McLendon | 70.29 | 58.32 | 77.34 | 382 | Jets |
| 57 | 15 | Javon Hargrave | 70.14 | 57.93 | 74.11 | 645 | Steelers |
| 58 | 16 | Star Lotulelei | 70.11 | 57.42 | 74.61 | 701 | Panthers |
| 59 | 17 | A'Shawn Robinson | 70.03 | 59.00 | 73.21 | 446 | Lions |
| 60 | 18 | Arik Armstead | 69.83 | 63.23 | 75.26 | 332 | 49ers |
| 61 | 19 | Vincent Valentine | 69.41 | 55.46 | 74.55 | 319 | Patriots |
| 62 | 20 | C.J. Wilson | 69.35 | 62.52 | 76.18 | 128 | Bears |
| 63 | 21 | Ricky Jean Francois | 69.23 | 57.41 | 72.94 | 442 | Commanders |
| 64 | 22 | Stacy McGee | 68.91 | 68.64 | 69.09 | 242 | Raiders |
| 65 | 23 | Zach Kerr | 68.85 | 55.87 | 77.51 | 317 | Colts |
| 66 | 24 | Christian Covington | 68.67 | 60.30 | 70.08 | 475 | Texans |
| 67 | 25 | Dontari Poe | 68.48 | 64.10 | 67.24 | 876 | Chiefs |
| 68 | 26 | Cedric Thornton | 68.21 | 54.43 | 75.21 | 291 | Cowboys |
| 69 | 27 | Beau Allen | 68.07 | 56.91 | 71.34 | 412 | Eagles |
| 70 | 28 | Paul Soliai | 68.06 | 55.15 | 76.46 | 152 | Panthers |
| 71 | 29 | Rodney Gunter | 67.99 | 58.36 | 70.24 | 244 | Cardinals |
| 72 | 30 | Pat Sims | 67.85 | 53.81 | 75.23 | 409 | Bengals |
| 73 | 31 | Haloti Ngata | 67.72 | 56.96 | 72.81 | 577 | Lions |
| 74 | 32 | Tyson Jackson | 67.68 | 58.51 | 69.63 | 389 | Falcons |
| 75 | 33 | Rakeem Nunez-Roches | 67.52 | 53.49 | 79.60 | 313 | Chiefs |
| 76 | 34 | DaQuan Jones | 67.51 | 65.39 | 66.64 | 673 | Titans |
| 77 | 35 | Quinton Dial | 67.34 | 57.64 | 71.41 | 478 | 49ers |
| 78 | 36 | Sealver Siliga | 67.32 | 56.73 | 74.89 | 160 | Buccaneers |
| 79 | 37 | David Parry | 67.03 | 56.13 | 70.13 | 644 | Colts |
| 80 | 38 | Tony McDaniel | 67.02 | 50.89 | 74.24 | 501 | Seahawks |
| 81 | 39 | Tom Johnson | 66.91 | 51.57 | 74.00 | 476 | Vikings |
| 82 | 40 | Frostee Rucker | 66.69 | 48.73 | 76.38 | 304 | Cardinals |
| 83 | 41 | Al Woods | 66.57 | 54.28 | 73.29 | 245 | Titans |
| 84 | 42 | Corey Peters | 66.53 | 57.16 | 69.64 | 498 | Cardinals |
| 85 | 43 | Sylvester Williams | 66.49 | 57.78 | 68.13 | 644 | Broncos |
| 86 | 44 | Jamie Meder | 66.25 | 60.99 | 68.73 | 722 | Browns |
| 87 | 45 | Clinton McDonald | 66.10 | 53.68 | 76.05 | 485 | Buccaneers |
| 88 | 46 | Tyrunn Walker | 66.10 | 54.76 | 73.24 | 377 | Lions |
| 89 | 47 | Cullen Jenkins | 66.06 | 50.32 | 73.74 | 308 | Commanders |
| 90 | 48 | Ahtyba Rubin | 65.84 | 52.22 | 71.39 | 675 | Seahawks |
| 91 | 49 | Letroy Guion | 65.84 | 53.80 | 70.02 | 524 | Packers |
| 92 | 50 | Sen'Derrick Marks | 65.79 | 54.06 | 73.19 | 543 | Jaguars |
| 93 | 51 | Chris Jones | 65.77 | 53.72 | 71.20 | 916 | Chiefs |
| 94 | 52 | Stefan Charles | 65.70 | 53.84 | 72.45 | 235 | Lions |
| 95 | 53 | Jarran Reed | 65.69 | 52.98 | 69.99 | 545 | Seahawks |
| 96 | 54 | Tyson Alualu | 65.61 | 58.14 | 67.46 | 507 | Jaguars |
| 97 | 55 | T.Y. McGill | 65.46 | 60.96 | 67.81 | 302 | Colts |
| 98 | 56 | Justin Ellis | 65.40 | 62.29 | 64.56 | 352 | Raiders |
| 99 | 57 | Jonathan Bullard | 65.33 | 53.24 | 71.31 | 296 | Bears |
| 100 | 58 | Jared Odrick | 65.29 | 59.75 | 70.02 | 261 | Jaguars |
| 101 | 59 | Earl Mitchell | 64.98 | 53.31 | 72.96 | 334 | Dolphins |
| 102 | 60 | Cam Thomas | 64.95 | 53.37 | 68.50 | 391 | Rams |
| 103 | 61 | Billy Winn | 64.92 | 53.18 | 70.47 | 341 | Broncos |
| 104 | 62 | Hassan Ridgeway | 64.83 | 54.64 | 67.45 | 442 | Colts |
| 105 | 63 | Denico Autry | 64.75 | 47.87 | 72.62 | 742 | Raiders |
| 106 | 64 | Arthur Jones | 64.62 | 51.92 | 74.76 | 322 | Colts |
| 107 | 65 | Vince Wilfork | 64.42 | 48.35 | 70.97 | 588 | Texans |
| 108 | 66 | John Jenkins | 64.38 | 55.50 | 70.71 | 217 | Seahawks |
| 109 | 67 | Jonathan Babineaux | 64.20 | 47.62 | 71.28 | 533 | Falcons |
| 110 | 68 | Austin Johnson | 64.17 | 58.21 | 70.22 | 190 | Titans |
| 111 | 69 | Destiny Vaeao | 64.10 | 51.64 | 68.24 | 268 | Eagles |
| 112 | 70 | Allen Bailey | 64.09 | 57.89 | 70.83 | 181 | Chiefs |
| 113 | 71 | Jared Crick | 64.04 | 50.41 | 68.96 | 938 | Broncos |
| 114 | 72 | Jarvis Jenkins | 64.01 | 55.14 | 66.08 | 266 | Chiefs |
| 115 | 73 | Derrick Shelby | 63.82 | 49.47 | 76.12 | 244 | Falcons |
| 116 | 74 | Vernon Butler | 63.78 | 57.03 | 70.36 | 226 | Panthers |
| 117 | 75 | Roy Miller | 63.65 | 58.09 | 68.83 | 156 | Jaguars |
| 118 | 76 | Domata Peko Sr. | 63.65 | 47.67 | 70.13 | 593 | Bengals |
| 119 | 77 | Antonio Smith | 63.00 | 43.89 | 72.09 | 277 | Texans |
| 120 | 78 | Tenny Palepoi | 62.86 | 53.06 | 67.18 | 377 | Chargers |
| 121 | 79 | Kendall Reyes | 62.81 | 53.83 | 66.19 | 249 | Chiefs |
| 122 | 80 | Angelo Blackson | 62.74 | 53.52 | 66.67 | 248 | Titans |
| 123 | 81 | Maliek Collins | 62.53 | 47.51 | 68.37 | 697 | Cowboys |
| 124 | 82 | Terrell McClain | 62.46 | 54.46 | 68.41 | 500 | Cowboys |
| 125 | 83 | Mike Purcell | 62.29 | 55.58 | 69.05 | 280 | 49ers |
| 126 | 84 | Jay Bromley | 62.17 | 54.87 | 64.54 | 264 | Giants |
| 127 | 85 | Jordan Phillips | 62.08 | 52.50 | 64.69 | 653 | Dolphins |
| 128 | 86 | Caraun Reid | 62.05 | 56.20 | 67.93 | 112 | Chargers |
| 129 | 87 | Kendall Langford | 62.02 | 51.82 | 69.33 | 300 | Colts |

### Rotation/backup (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 130 | 1 | Jerel Worthy | 61.94 | 55.82 | 68.20 | 150 | Bills |
| 131 | 2 | Will Sutton III | 61.84 | 60.27 | 64.04 | 174 | Bears |
| 132 | 3 | Akeem Spence | 61.50 | 48.08 | 67.94 | 724 | Buccaneers |
| 133 | 4 | Darius Philon | 61.50 | 54.44 | 66.47 | 267 | Chargers |
| 134 | 5 | Joel Heath | 61.46 | 52.54 | 65.32 | 277 | Texans |
| 135 | 6 | Mitch Unrein | 61.34 | 53.74 | 66.21 | 436 | Bears |
| 136 | 7 | Sheldon Rankins | 61.33 | 54.44 | 69.06 | 335 | Saints |
| 137 | 8 | Glenn Dorsey | 61.29 | 46.45 | 70.99 | 402 | 49ers |
| 138 | 9 | Xavier Cooper | 61.08 | 53.41 | 64.76 | 448 | Browns |
| 139 | 10 | Kyle Love | 60.39 | 50.59 | 69.00 | 224 | Panthers |
| 140 | 11 | Ricardo Mathews | 60.37 | 51.01 | 63.28 | 330 | Steelers |
| 141 | 12 | Corbin Bryant | 59.84 | 53.44 | 64.10 | 233 | Bills |
| 142 | 13 | Adam Gotsis | 59.75 | 54.25 | 59.25 | 221 | Broncos |
| 143 | 14 | Ed Stinson | 59.64 | 54.87 | 66.16 | 116 | Cardinals |
| 144 | 15 | Matt Ioannidis | 59.38 | 57.80 | 63.57 | 103 | Commanders |
| 145 | 16 | Darius Latham | 58.81 | 51.75 | 60.38 | 319 | Raiders |
| 146 | 17 | Cornelius Washington | 58.16 | 52.02 | 64.61 | 364 | Bears |
| 147 | 18 | Khyri Thornton | 57.74 | 49.64 | 64.18 | 328 | Lions |
| 148 | 19 | David Onyemata | 57.30 | 52.13 | 56.58 | 393 | Saints |
| 149 | 20 | Damion Square | 57.17 | 54.35 | 61.65 | 362 | Chargers |
| 150 | 21 | Shamar Stephen | 56.97 | 49.25 | 61.38 | 551 | Vikings |
| 151 | 22 | Jihad Ward | 56.53 | 45.58 | 59.66 | 637 | Raiders |
| 152 | 23 | Tyeler Davison | 56.45 | 51.55 | 56.20 | 438 | Saints |
| 153 | 24 | Xavier Williams | 54.04 | 56.96 | 57.82 | 118 | Cardinals |
| 154 | 25 | Anthony Johnson | 53.30 | 55.40 | 54.90 | 127 | Jets |

## ED — Edge

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 94.99 | 96.41 | 89.87 | 930 | Broncos |
| 2 | 2 | Khalil Mack | 91.33 | 97.42 | 83.10 | 1023 | Raiders |
| 3 | 3 | Joey Bosa | 89.82 | 90.92 | 89.08 | 563 | Chargers |
| 4 | 4 | Whitney Mercilus | 87.54 | 88.05 | 83.24 | 1008 | Texans |
| 5 | 5 | Brandon Graham | 87.40 | 90.40 | 81.24 | 764 | Eagles |
| 6 | 6 | Danielle Hunter | 85.66 | 82.66 | 83.88 | 602 | Vikings |
| 7 | 7 | Justin Houston | 84.26 | 79.63 | 89.33 | 349 | Chiefs |
| 8 | 8 | Ezekiel Ansah | 82.91 | 78.25 | 82.88 | 540 | Lions |
| 9 | 9 | Pernell McPhee | 82.16 | 77.01 | 85.69 | 273 | Bears |
| 10 | 10 | Cameron Wake | 82.01 | 72.87 | 86.75 | 626 | Dolphins |
| 11 | 11 | Frank Clark | 81.97 | 80.53 | 78.76 | 751 | Seahawks |
| 12 | 12 | Michael Bennett | 81.72 | 87.02 | 75.59 | 675 | Seahawks |
| 13 | 13 | James Harrison | 81.70 | 69.75 | 86.33 | 758 | Steelers |
| 14 | 14 | Olivier Vernon | 81.68 | 87.58 | 73.58 | 1112 | Giants |
| 15 | 15 | Carlos Dunlap | 81.60 | 81.63 | 77.41 | 840 | Bengals |
| 16 | 16 | Markus Golden | 81.34 | 73.79 | 82.21 | 762 | Cardinals |
| 17 | 17 | Cameron Jordan | 80.45 | 88.99 | 70.59 | 963 | Saints |
| 18 | 18 | Chandler Jones | 80.01 | 82.53 | 74.79 | 939 | Cardinals |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Cliff Avril | 78.98 | 68.70 | 81.67 | 926 | Seahawks |
| 20 | 2 | Melvin Ingram III | 78.66 | 76.63 | 77.31 | 960 | Chargers |
| 21 | 3 | Nick Perry | 78.34 | 74.62 | 76.65 | 704 | Packers |
| 22 | 4 | Jabaal Sheard | 77.79 | 83.50 | 70.14 | 670 | Patriots |
| 23 | 5 | Ryan Kerrigan | 77.73 | 67.56 | 80.34 | 787 | Commanders |
| 24 | 6 | DeMarcus Ware | 77.66 | 64.81 | 85.81 | 315 | Broncos |
| 25 | 7 | Charles Johnson | 77.49 | 69.53 | 81.45 | 542 | Panthers |
| 26 | 8 | William Hayes | 77.30 | 77.26 | 74.19 | 514 | Rams |
| 27 | 9 | Shaquil Barrett | 76.56 | 70.98 | 76.11 | 415 | Broncos |
| 28 | 10 | Robert Quinn | 75.72 | 75.13 | 78.09 | 370 | Rams |
| 29 | 11 | Everson Griffen | 75.27 | 69.65 | 74.85 | 888 | Vikings |
| 30 | 12 | Shane Ray | 75.09 | 63.28 | 78.79 | 664 | Broncos |
| 31 | 13 | Vic Beasley Jr. | 75.06 | 69.64 | 74.50 | 822 | Falcons |
| 32 | 14 | Leonard Floyd | 74.80 | 61.29 | 83.80 | 537 | Bears |
| 33 | 15 | Elvis Dumervil | 74.65 | 61.88 | 83.16 | 272 | Ravens |
| 34 | 16 | Derrick Morgan | 74.27 | 65.48 | 78.37 | 768 | Titans |
| 35 | 17 | Robert Ayers | 74.17 | 75.34 | 73.39 | 577 | Buccaneers |

### Starter (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 36 | 1 | Jadeveon Clowney | 73.96 | 88.83 | 63.32 | 871 | Texans |
| 37 | 2 | Jason Pierre-Paul | 73.49 | 78.18 | 70.78 | 793 | Giants |
| 38 | 3 | Lorenzo Alexander | 73.14 | 55.10 | 86.84 | 788 | Bills |
| 39 | 4 | Mario Addison | 72.99 | 59.97 | 78.53 | 433 | Panthers |
| 40 | 5 | Trent Murphy | 72.83 | 68.16 | 71.97 | 676 | Commanders |
| 41 | 6 | Jerry Hughes | 72.81 | 64.33 | 74.29 | 857 | Bills |
| 42 | 7 | Vinny Curry | 72.53 | 63.22 | 74.57 | 436 | Eagles |
| 43 | 8 | Bruce Irvin | 72.35 | 59.22 | 76.93 | 997 | Raiders |
| 44 | 9 | Cam Johnson | 71.76 | 60.01 | 85.33 | 350 | Browns |
| 45 | 10 | Willie Young | 71.66 | 59.11 | 76.37 | 715 | Bears |
| 46 | 11 | Mario Williams | 71.12 | 60.03 | 75.70 | 453 | Dolphins |
| 47 | 12 | Jordan Jenkins | 71.05 | 60.12 | 76.25 | 514 | Jets |
| 48 | 13 | Julius Peppers | 70.90 | 59.17 | 74.55 | 715 | Packers |
| 49 | 14 | Aaron Lynch | 70.86 | 60.41 | 78.98 | 222 | 49ers |
| 50 | 15 | Clay Matthews | 70.70 | 52.54 | 79.15 | 630 | Packers |
| 51 | 16 | Dee Ford | 70.50 | 59.62 | 74.62 | 855 | Chiefs |
| 52 | 17 | Brian Orakpo | 70.42 | 58.26 | 76.24 | 868 | Titans |
| 53 | 18 | Kerry Hyder Jr. | 69.94 | 63.21 | 76.13 | 709 | Lions |
| 54 | 19 | Alex Okafor | 69.88 | 62.91 | 72.24 | 231 | Cardinals |
| 55 | 20 | Matthew Judon | 69.26 | 59.67 | 73.57 | 308 | Ravens |
| 56 | 21 | Connor Barwin | 69.21 | 54.21 | 75.04 | 705 | Eagles |
| 57 | 22 | John Simon | 69.17 | 61.83 | 73.75 | 516 | Texans |
| 58 | 23 | Lorenzo Mauldin IV | 68.91 | 61.08 | 74.00 | 354 | Jets |
| 59 | 24 | Arthur Moats | 68.54 | 55.03 | 73.90 | 401 | Steelers |
| 60 | 25 | Jeremiah Attaochu | 68.45 | 62.03 | 74.08 | 178 | Chargers |
| 61 | 26 | Tamba Hali | 68.38 | 60.91 | 69.20 | 599 | Chiefs |
| 62 | 27 | Kyler Fackrell | 68.05 | 58.84 | 72.10 | 173 | Packers |
| 63 | 28 | Terrell Suggs | 67.81 | 55.21 | 77.25 | 697 | Ravens |
| 64 | 29 | Chris Long | 67.61 | 55.50 | 74.85 | 741 | Patriots |
| 65 | 30 | Dan Skuta | 67.29 | 53.04 | 74.80 | 267 | Jaguars |
| 66 | 31 | Trey Flowers | 67.05 | 62.15 | 72.01 | 726 | Patriots |
| 67 | 32 | Preston Smith | 66.78 | 60.16 | 67.02 | 770 | Commanders |
| 68 | 33 | Marcus Smith | 66.74 | 57.82 | 70.41 | 220 | Eagles |
| 69 | 34 | Paul Kruger | 66.59 | 55.02 | 70.66 | 572 | Saints |
| 70 | 35 | Erik Walden | 66.36 | 49.77 | 73.57 | 759 | Colts |
| 71 | 36 | Dante Fowler Jr. | 66.08 | 64.71 | 62.82 | 569 | Jaguars |
| 72 | 37 | Akeem Ayers | 65.72 | 56.51 | 68.32 | 360 | Colts |
| 73 | 38 | Dwight Freeney | 65.54 | 49.28 | 73.14 | 538 | Falcons |
| 74 | 39 | Trent Cole | 65.49 | 56.90 | 72.56 | 237 | Colts |
| 75 | 40 | Yannick Ngakoue | 65.26 | 56.46 | 66.96 | 706 | Jaguars |
| 76 | 41 | Ryan Delaire | 65.21 | 59.37 | 73.79 | 139 | Panthers |
| 77 | 42 | DeMarcus Lawrence | 64.98 | 67.54 | 63.69 | 369 | Cowboys |
| 78 | 43 | Armonty Bryant | 64.97 | 57.11 | 72.40 | 104 | Lions |
| 79 | 44 | Noah Spence | 64.87 | 59.83 | 64.06 | 569 | Buccaneers |
| 80 | 45 | Devon Kennard | 64.63 | 55.96 | 66.25 | 549 | Giants |
| 81 | 46 | Kony Ealy | 64.33 | 58.09 | 64.32 | 624 | Panthers |
| 82 | 47 | Emmanuel Ogbah | 64.18 | 58.76 | 63.63 | 849 | Browns |
| 83 | 48 | Joe Schobert | 63.89 | 57.55 | 67.09 | 246 | Browns |
| 84 | 49 | Rob Ninkovich | 63.88 | 49.57 | 69.77 | 566 | Patriots |
| 85 | 50 | Darryl Tapp | 63.48 | 56.66 | 64.38 | 291 | Saints |
| 86 | 51 | Ahmad Brooks | 63.38 | 48.19 | 70.59 | 918 | 49ers |
| 87 | 52 | Damontre Moore | 63.23 | 60.42 | 67.80 | 104 | Seahawks |
| 88 | 53 | Michael Johnson | 63.08 | 59.34 | 61.82 | 831 | Bengals |
| 89 | 54 | Tyrone Holmes | 62.96 | 59.66 | 66.20 | 124 | Browns |
| 90 | 55 | Devin Taylor | 62.94 | 56.68 | 63.26 | 715 | Lions |
| 91 | 56 | Shea McClellin | 62.61 | 52.66 | 65.07 | 441 | Patriots |

### Rotation/backup (49 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 92 | 1 | Jayrone Elliott | 61.75 | 59.59 | 66.00 | 136 | Packers |
| 93 | 2 | Brian Robison | 61.74 | 49.00 | 66.07 | 838 | Vikings |
| 94 | 3 | Ryan Davis Sr. | 61.73 | 58.32 | 65.99 | 155 | Cowboys |
| 95 | 4 | Brooks Reed | 61.15 | 55.35 | 61.79 | 529 | Falcons |
| 96 | 5 | Ethan Westbrooks | 61.01 | 56.23 | 60.96 | 531 | Rams |
| 97 | 6 | Datone Jones | 61.00 | 54.28 | 61.32 | 635 | Packers |
| 98 | 7 | Ronald Blair III | 60.98 | 58.78 | 58.28 | 307 | 49ers |
| 99 | 8 | William Gholston | 60.84 | 59.50 | 58.82 | 586 | Buccaneers |
| 100 | 9 | Adrian Clayborn | 60.73 | 57.80 | 62.68 | 589 | Falcons |
| 101 | 10 | Shaq Lawson | 60.67 | 58.37 | 64.28 | 237 | Bills |
| 102 | 11 | Lerentee McCray | 60.23 | 59.21 | 61.64 | 163 | Bills |
| 103 | 12 | Bud Dupree | 60.13 | 57.11 | 61.88 | 509 | Steelers |
| 104 | 13 | Kasim Edebali | 60.09 | 55.40 | 59.46 | 254 | Saints |
| 105 | 14 | Sam Acho | 59.91 | 54.31 | 59.79 | 499 | Bears |
| 106 | 15 | Tyrone Crawford | 59.75 | 53.76 | 60.09 | 650 | Cowboys |
| 107 | 16 | Wallace Gilberry | 59.61 | 53.46 | 63.20 | 266 | Bengals |
| 108 | 17 | Eli Harold | 59.40 | 57.82 | 56.29 | 691 | 49ers |
| 109 | 18 | Andre Branch | 59.36 | 55.93 | 59.88 | 814 | Dolphins |
| 110 | 19 | Benson Mayowa | 59.26 | 59.96 | 57.12 | 406 | Cowboys |
| 111 | 20 | Kyle Emanuel | 59.19 | 55.37 | 57.95 | 545 | Chargers |
| 112 | 21 | Muhammad Wilkerson | 59.07 | 53.16 | 59.36 | 848 | Jets |
| 113 | 22 | Jarvis Jones | 58.80 | 56.81 | 57.63 | 499 | Steelers |
| 114 | 23 | Za'Darius Smith | 58.67 | 57.44 | 57.67 | 492 | Ravens |
| 115 | 24 | Robert Mathis | 58.67 | 45.16 | 65.21 | 536 | Colts |
| 116 | 25 | David Bass | 58.66 | 58.30 | 58.29 | 224 | Titans |
| 117 | 26 | Aaron Wallace | 58.21 | 59.61 | 61.44 | 117 | Titans |
| 118 | 27 | Cassius Marsh | 58.20 | 59.61 | 55.38 | 438 | Seahawks |
| 119 | 28 | Romeo Okwara | 58.09 | 58.43 | 53.69 | 427 | Giants |
| 120 | 29 | Anthony Zettel | 57.33 | 57.23 | 55.32 | 224 | Lions |
| 121 | 30 | Wes Horton | 56.93 | 56.37 | 56.98 | 332 | Panthers |
| 122 | 31 | Kerry Wynn | 56.88 | 59.11 | 54.87 | 131 | Giants |
| 123 | 32 | Will Clarke | 56.30 | 56.32 | 54.94 | 374 | Bengals |
| 124 | 33 | Jack Crawford | 56.28 | 50.63 | 57.97 | 548 | Cowboys |
| 125 | 34 | Tourek Williams | 56.03 | 55.82 | 54.19 | 142 | Chargers |
| 126 | 35 | Anthony Chickillo | 55.59 | 57.86 | 57.72 | 318 | Steelers |
| 127 | 36 | Carl Nassib | 55.52 | 54.89 | 53.86 | 541 | Browns |
| 128 | 37 | Shilique Calhoun | 55.46 | 56.93 | 56.56 | 172 | Raiders |
| 129 | 38 | Frank Zombo | 55.34 | 51.96 | 57.18 | 501 | Chiefs |
| 130 | 39 | Eugene Sims | 55.01 | 50.41 | 54.84 | 537 | Rams |
| 131 | 40 | Terrence Fede | 54.33 | 57.65 | 53.59 | 191 | Dolphins |
| 132 | 41 | Mike Catapano | 53.74 | 54.71 | 51.52 | 210 | Jets |
| 133 | 42 | Kevin Dodd | 53.24 | 57.40 | 53.60 | 179 | Titans |
| 134 | 43 | Corey Lemonier | 53.11 | 59.50 | 52.39 | 135 | Jets |
| 135 | 44 | Albert McClellan | 52.81 | 45.79 | 54.25 | 604 | Ravens |
| 136 | 45 | Matt Longacre | 51.69 | 59.13 | 53.36 | 157 | Rams |
| 137 | 46 | Ryan Russell | 50.72 | 57.43 | 53.15 | 174 | Buccaneers |
| 138 | 47 | DaVonte Lambert | 49.02 | 54.13 | 46.65 | 374 | Buccaneers |
| 139 | 48 | Brennan Scarlett | 48.89 | 58.26 | 51.89 | 119 | Texans |
| 140 | 49 | Freddie Bishop | 45.00 | 56.49 | 51.28 | 152 | Jets |

## G — Guard

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Zack Martin | 93.65 | 89.40 | 92.31 | 1126 | Cowboys |
| 2 | 2 | Marshal Yanda | 92.33 | 86.96 | 91.75 | 899 | Ravens |
| 3 | 3 | David DeCastro | 91.02 | 84.70 | 91.06 | 1274 | Steelers |
| 4 | 4 | Kelechi Osemele | 90.47 | 84.20 | 90.48 | 1110 | Raiders |
| 5 | 5 | Andy Levitre | 90.35 | 84.60 | 90.02 | 1212 | Falcons |
| 6 | 6 | Shaq Mason | 89.62 | 84.60 | 88.80 | 1258 | Patriots |
| 7 | 7 | Kevin Zeitler | 89.18 | 83.30 | 88.94 | 1087 | Bengals |
| 8 | 8 | John Greco | 88.19 | 80.84 | 88.93 | 746 | Browns |
| 9 | 9 | Quinton Spain | 88.18 | 82.09 | 88.08 | 821 | Titans |
| 10 | 10 | Brandon Brooks | 88.16 | 82.40 | 87.84 | 989 | Eagles |
| 11 | 11 | Ramon Foster | 87.89 | 83.60 | 86.59 | 1099 | Steelers |
| 12 | 12 | Josh Sitton | 87.39 | 80.29 | 87.95 | 734 | Bears |
| 13 | 13 | Larry Warford | 87.36 | 80.30 | 87.90 | 1024 | Lions |
| 14 | 14 | T.J. Lang | 86.80 | 80.96 | 86.52 | 964 | Packers |
| 15 | 15 | Ron Leary | 84.75 | 77.65 | 85.31 | 871 | Cowboys |
| 16 | 16 | Brandon Scherff | 84.69 | 77.20 | 85.52 | 1045 | Commanders |
| 17 | 17 | Justin Pugh | 84.48 | 76.90 | 85.37 | 750 | Giants |
| 18 | 18 | Allen Barbre | 84.39 | 75.69 | 86.03 | 672 | Eagles |
| 19 | 19 | Richie Incognito | 83.86 | 76.50 | 84.60 | 1060 | Bills |
| 20 | 20 | Andrew Norwell | 83.80 | 77.30 | 83.96 | 1108 | Panthers |
| 21 | 21 | Evan Mathis | 83.55 | 72.05 | 87.05 | 199 | Cardinals |
| 22 | 22 | James Carpenter | 82.73 | 76.20 | 82.92 | 994 | Jets |
| 23 | 23 | Joel Bitonio | 82.54 | 73.68 | 84.28 | 331 | Browns |
| 24 | 24 | Gabe Jackson | 82.29 | 74.70 | 83.19 | 1188 | Raiders |
| 25 | 25 | Kyle Long | 81.52 | 72.99 | 83.04 | 431 | Bears |
| 26 | 26 | Tim Lelito | 81.41 | 71.78 | 83.67 | 406 | Saints |
| 27 | 27 | Ted Larsen | 81.02 | 69.98 | 84.22 | 581 | Bears |
| 28 | 28 | Rodger Saffold | 80.68 | 72.15 | 82.20 | 916 | Rams |
| 29 | 29 | John Jerry | 80.34 | 72.60 | 81.33 | 1123 | Giants |
| 30 | 30 | A.J. Cann | 80.29 | 72.60 | 81.25 | 1113 | Jaguars |
| 31 | 31 | Max Garcia | 80.15 | 72.20 | 81.28 | 1074 | Broncos |

### Good (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 32 | 1 | Chris Chester | 79.40 | 71.00 | 80.83 | 1233 | Falcons |
| 33 | 2 | Brian Winters | 78.93 | 69.50 | 81.05 | 806 | Jets |
| 34 | 3 | Mike Iupati | 78.71 | 69.80 | 80.49 | 1035 | Cardinals |
| 35 | 4 | Laurent Duvernay-Tardif | 78.71 | 71.76 | 79.18 | 946 | Chiefs |
| 36 | 5 | Josh Kline | 78.65 | 70.10 | 80.19 | 929 | Titans |
| 37 | 6 | Spencer Drango | 78.55 | 69.59 | 80.36 | 599 | Browns |
| 38 | 7 | Patrick Omameh | 78.29 | 68.89 | 80.39 | 454 | Jaguars |
| 39 | 8 | Isaac Seumalo | 78.25 | 66.19 | 82.12 | 336 | Eagles |
| 40 | 9 | John Miller | 77.85 | 69.60 | 79.18 | 1047 | Bills |
| 41 | 10 | Zach Fulton | 77.84 | 70.45 | 78.60 | 857 | Chiefs |
| 42 | 11 | Trai Turner | 77.73 | 67.00 | 80.71 | 1098 | Panthers |
| 43 | 12 | Joe Haeg | 77.73 | 67.77 | 80.21 | 952 | Colts |
| 44 | 13 | Luke Joeckel | 77.41 | 64.22 | 82.04 | 221 | Jaguars |
| 45 | 14 | Jahri Evans | 77.27 | 69.00 | 78.62 | 1138 | Saints |
| 46 | 15 | Alex Boone | 76.71 | 66.85 | 79.11 | 873 | Vikings |
| 47 | 16 | Xavier Su'a-Filo | 76.62 | 67.60 | 78.46 | 1168 | Texans |
| 48 | 17 | Cody Wichmann | 76.43 | 66.06 | 79.17 | 594 | Rams |
| 49 | 18 | Parker Ehinger | 76.23 | 64.00 | 80.21 | 229 | Chiefs |
| 50 | 19 | Dakota Dozier | 76.18 | 67.96 | 77.50 | 144 | Jets |
| 51 | 20 | Jack Mewhort | 75.82 | 66.07 | 78.15 | 666 | Colts |
| 52 | 21 | Lane Taylor | 75.60 | 66.50 | 77.50 | 1239 | Packers |
| 53 | 22 | Andrew Tiller | 75.57 | 65.01 | 78.44 | 485 | 49ers |
| 54 | 23 | D.J. Fluker | 75.50 | 66.00 | 77.66 | 993 | Chargers |
| 55 | 24 | Joe Thuney | 75.49 | 65.60 | 77.91 | 1354 | Patriots |
| 56 | 25 | Kenny Wiggins | 75.11 | 62.28 | 79.49 | 134 | Chargers |
| 57 | 26 | Mark Glowinski | 75.09 | 64.60 | 77.91 | 1186 | Seahawks |
| 58 | 27 | Clint Boling | 74.97 | 66.12 | 76.70 | 942 | Bengals |

### Starter (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | Senio Kelemete | 73.61 | 62.46 | 76.88 | 664 | Saints |
| 60 | 2 | Brandon Fusco | 73.49 | 63.33 | 76.09 | 834 | Vikings |
| 61 | 3 | Vladimir Ducasse | 72.75 | 62.33 | 75.53 | 555 | Ravens |
| 62 | 4 | John Urschel | 72.41 | 61.61 | 75.45 | 267 | Ravens |
| 63 | 5 | Chris Scott | 72.30 | 62.08 | 74.95 | 295 | Panthers |
| 64 | 6 | Joshua Garnett | 72.15 | 61.28 | 75.23 | 716 | 49ers |
| 65 | 7 | Orlando Franklin | 71.92 | 61.47 | 74.72 | 919 | Chargers |
| 66 | 8 | Oday Aboushi | 71.25 | 60.18 | 74.47 | 358 | Texans |
| 67 | 9 | Shawn Lauvao | 71.11 | 59.81 | 74.48 | 913 | Commanders |
| 68 | 10 | Alex Lewis | 71.02 | 60.15 | 74.10 | 539 | Ravens |
| 69 | 11 | Alvin Bailey | 70.23 | 59.82 | 73.01 | 373 | Browns |
| 70 | 12 | Denver Kirkland | 69.44 | 61.52 | 70.56 | 130 | Raiders |
| 71 | 13 | Laken Tomlinson | 68.54 | 57.73 | 71.58 | 699 | Lions |
| 72 | 14 | Zane Beadles | 67.84 | 55.30 | 72.04 | 1035 | 49ers |
| 73 | 15 | Jeff Allen | 67.81 | 56.32 | 71.30 | 968 | Texans |
| 74 | 16 | Jermon Bushrod | 67.36 | 55.70 | 70.97 | 1011 | Dolphins |
| 75 | 17 | Kevin Pamphile | 67.35 | 56.16 | 70.64 | 957 | Buccaneers |
| 76 | 18 | Chance Warmack | 67.33 | 56.35 | 70.48 | 134 | Titans |
| 77 | 19 | Ted Karras | 66.12 | 60.24 | 65.88 | 109 | Patriots |
| 78 | 20 | Tyler Shatley | 63.94 | 51.67 | 67.96 | 316 | Jaguars |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 79 | 1 | Billy Turner | 60.61 | 48.92 | 64.23 | 138 | Broncos |
| 80 | 2 | Arie Kouandjio | 59.93 | 52.44 | 60.75 | 128 | Commanders |

## HB — Running Back

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Ty Montgomery | 83.82 | 74.38 | 85.95 | 307 | Packers |
| 2 | 2 | Jalen Richard | 81.04 | 71.68 | 83.12 | 135 | Raiders |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 3 | 1 | Jay Ajayi | 77.73 | 73.75 | 76.21 | 240 | Dolphins |
| 4 | 2 | Le'Veon Bell | 76.54 | 78.20 | 71.26 | 439 | Steelers |
| 5 | 3 | David Johnson | 76.37 | 79.90 | 69.85 | 509 | Cardinals |
| 6 | 4 | Jordan Howard | 76.09 | 72.32 | 74.44 | 303 | Bears |
| 7 | 5 | Ezekiel Elliott | 75.68 | 74.97 | 71.98 | 288 | Cowboys |
| 8 | 6 | LeSean McCoy | 74.26 | 76.56 | 68.56 | 254 | Bills |
| 9 | 7 | Spencer Ware | 74.23 | 70.48 | 72.56 | 276 | Chiefs |
| 10 | 8 | Kenneth Dixon | 74.12 | 67.23 | 74.54 | 127 | Ravens |

### Starter (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Darren Sproles | 72.43 | 67.40 | 71.62 | 318 | Eagles |
| 12 | 2 | Chris Thompson | 72.41 | 62.89 | 74.59 | 309 | Commanders |
| 13 | 3 | Mike Gillislee | 72.10 | 70.04 | 69.30 | 104 | Bills |
| 14 | 4 | Devonta Freeman | 71.61 | 75.63 | 64.77 | 345 | Falcons |
| 15 | 5 | Bilal Powell | 71.56 | 76.26 | 64.26 | 317 | Jets |
| 16 | 6 | DeMarco Murray | 71.26 | 69.31 | 68.40 | 355 | Titans |
| 17 | 7 | Mark Ingram II | 70.98 | 68.88 | 68.22 | 239 | Saints |
| 18 | 8 | Thomas Rawls | 70.94 | 69.29 | 67.88 | 142 | Seahawks |
| 19 | 9 | DeAndre Washington | 70.84 | 63.01 | 71.90 | 127 | Raiders |
| 20 | 10 | Dion Lewis | 70.29 | 62.51 | 71.31 | 104 | Patriots |
| 21 | 11 | Duke Johnson Jr. | 70.16 | 63.93 | 70.15 | 277 | Browns |
| 22 | 12 | Carlos Hyde | 70.09 | 62.79 | 70.79 | 229 | 49ers |
| 23 | 13 | Rex Burkhead | 69.96 | 68.48 | 66.78 | 108 | Bengals |
| 24 | 14 | Theo Riddick | 69.05 | 70.69 | 63.79 | 254 | Lions |
| 25 | 15 | Jacquizz Rodgers | 68.99 | 69.81 | 64.28 | 142 | Buccaneers |
| 26 | 16 | Jonathan Stewart | 68.47 | 62.44 | 68.32 | 179 | Panthers |
| 27 | 17 | Tevin Coleman | 68.43 | 70.62 | 62.81 | 201 | Falcons |
| 28 | 18 | Damien Williams | 68.18 | 67.87 | 64.22 | 106 | Dolphins |
| 29 | 19 | C.J. Anderson | 68.12 | 61.47 | 68.38 | 146 | Broncos |
| 30 | 20 | Giovani Bernard | 68.11 | 65.47 | 65.71 | 218 | Bengals |
| 31 | 21 | DeAngelo Williams | 67.71 | 66.30 | 64.49 | 175 | Steelers |
| 32 | 22 | Rob Kelley | 67.43 | 59.84 | 68.32 | 110 | Commanders |
| 33 | 23 | Melvin Gordon III | 67.37 | 68.04 | 62.76 | 310 | Chargers |
| 34 | 24 | Matt Forte | 67.22 | 63.15 | 65.77 | 194 | Jets |
| 35 | 25 | T.J. Yeldon | 67.08 | 65.46 | 63.99 | 305 | Jaguars |
| 36 | 26 | Isaiah Crowell | 66.83 | 65.60 | 63.49 | 212 | Browns |
| 37 | 27 | Chris Ivory | 66.78 | 55.93 | 69.84 | 140 | Jaguars |
| 38 | 28 | Lance Dunbar | 66.73 | 56.86 | 69.15 | 101 | Cowboys |
| 39 | 29 | Doug Martin | 66.71 | 61.22 | 66.21 | 106 | Buccaneers |
| 40 | 30 | Terrance West | 66.55 | 68.88 | 60.83 | 155 | Ravens |
| 41 | 31 | Latavius Murray | 66.43 | 65.21 | 63.07 | 230 | Raiders |
| 42 | 32 | James White | 66.36 | 72.70 | 57.96 | 384 | Patriots |
| 43 | 33 | LeGarrette Blount | 66.24 | 59.04 | 66.87 | 155 | Patriots |
| 44 | 34 | Christine Michael | 66.16 | 58.84 | 66.88 | 178 | Packers |
| 45 | 35 | Jerick McKinnon | 64.80 | 59.43 | 64.22 | 254 | Vikings |
| 46 | 36 | Jeremy Hill | 64.66 | 62.21 | 62.13 | 160 | Bengals |
| 47 | 37 | Robert Turbin | 64.63 | 64.03 | 60.87 | 166 | Colts |
| 48 | 38 | Lamar Miller | 64.60 | 60.92 | 62.88 | 283 | Texans |
| 49 | 39 | Jonathan Grimes | 64.59 | 65.35 | 59.92 | 123 | Texans |
| 50 | 40 | Justin Forsett | 64.25 | 54.89 | 66.32 | 130 | Broncos |
| 51 | 41 | Benny Cunningham | 63.97 | 59.06 | 63.08 | 116 | Rams |
| 52 | 42 | Todd Gurley II | 63.96 | 57.47 | 64.12 | 333 | Rams |
| 53 | 43 | Frank Gore | 63.91 | 59.64 | 62.59 | 263 | Colts |
| 54 | 44 | Tim Hightower | 63.22 | 57.41 | 62.92 | 102 | Saints |
| 55 | 45 | Charles Sims | 63.08 | 59.04 | 61.60 | 123 | Buccaneers |
| 56 | 46 | Paul Perkins | 62.84 | 62.02 | 59.22 | 152 | Giants |
| 57 | 47 | Zach Zenner | 62.82 | 61.06 | 59.83 | 166 | Lions |
| 58 | 48 | Rashad Jennings | 62.76 | 56.17 | 62.99 | 198 | Giants |
| 59 | 49 | James Starks | 62.50 | 54.23 | 63.84 | 140 | Packers |
| 60 | 50 | Fozzy Whittaker | 62.31 | 60.39 | 59.42 | 160 | Panthers |

### Rotation/backup (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 61 | 1 | Charcandrick West | 61.35 | 59.07 | 58.71 | 228 | Chiefs |
| 62 | 2 | Travaris Cadet | 61.14 | 64.04 | 55.04 | 213 | Saints |
| 63 | 3 | Devontae Booker | 60.89 | 56.37 | 59.74 | 227 | Broncos |
| 64 | 4 | Matt Asiata | 60.58 | 59.52 | 57.12 | 179 | Vikings |
| 65 | 5 | Shaun Draughn | 59.92 | 60.76 | 55.19 | 182 | 49ers |
| 66 | 6 | Jeremy Langford | 58.16 | 56.66 | 55.00 | 135 | Bears |
| 67 | 7 | Dwayne Washington | 57.66 | 56.29 | 54.40 | 101 | Lions |

## LB — Linebacker

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jerrell Freeman | 86.37 | 90.74 | 82.52 | 806 | Bears |
| 2 | 2 | Luke Kuechly | 85.71 | 89.04 | 82.46 | 656 | Panthers |
| 3 | 3 | Bobby Wagner | 83.06 | 86.40 | 77.09 | 1195 | Seahawks |
| 4 | 4 | Jordan Hicks | 80.22 | 88.40 | 74.11 | 971 | Eagles |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Paul Posluszny | 78.42 | 82.00 | 74.36 | 1057 | Jaguars |
| 6 | 2 | Sean Lee | 78.35 | 78.50 | 75.75 | 1041 | Cowboys |
| 7 | 3 | Mason Foster | 77.70 | 79.01 | 75.80 | 771 | Commanders |
| 8 | 4 | K.J. Wright | 77.44 | 75.60 | 74.50 | 1174 | Seahawks |
| 9 | 5 | C.J. Mosley | 77.40 | 76.30 | 75.00 | 875 | Ravens |
| 10 | 6 | Vincent Rey | 77.09 | 75.69 | 73.86 | 590 | Bengals |
| 11 | 7 | Shaq Thompson | 76.45 | 75.00 | 74.55 | 533 | Panthers |
| 12 | 8 | Vontaze Burfict | 75.80 | 81.99 | 73.95 | 674 | Bengals |
| 13 | 9 | Brandon Marshall | 75.24 | 75.77 | 73.54 | 596 | Broncos |
| 14 | 10 | Nigel Bradham | 74.91 | 75.90 | 72.07 | 989 | Eagles |
| 15 | 11 | Dont'a Hightower | 74.57 | 78.20 | 68.82 | 864 | Patriots |
| 16 | 12 | Benardrick McKinney | 74.16 | 72.00 | 72.22 | 1044 | Texans |
| 17 | 13 | Gerald Hodges | 74.15 | 75.53 | 71.34 | 584 | 49ers |

### Starter (58 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Zach Brown | 73.93 | 75.70 | 71.71 | 978 | Bills |
| 19 | 2 | Wesley Woodyard | 73.23 | 73.05 | 69.19 | 614 | Titans |
| 20 | 3 | Telvin Smith Sr. | 72.89 | 73.00 | 69.29 | 1048 | Jaguars |
| 21 | 4 | Perry Riley | 72.88 | 74.39 | 72.39 | 698 | Raiders |
| 22 | 5 | Ramik Wilson | 72.47 | 75.62 | 73.88 | 591 | Chiefs |
| 23 | 6 | Deion Jones | 71.20 | 70.10 | 67.76 | 1114 | Falcons |
| 24 | 7 | Jatavis Brown | 70.86 | 75.06 | 68.06 | 600 | Chargers |
| 25 | 8 | Eric Kendricks | 70.61 | 68.90 | 68.62 | 869 | Vikings |
| 26 | 9 | Christian Kirksey | 70.26 | 67.50 | 67.94 | 1111 | Browns |
| 27 | 10 | Brian Cushing | 70.16 | 65.17 | 70.26 | 754 | Texans |
| 28 | 11 | David Harris | 70.03 | 65.00 | 69.73 | 900 | Jets |
| 29 | 12 | Kwon Alexander | 69.63 | 68.90 | 67.52 | 1023 | Buccaneers |
| 30 | 13 | Mark Barron | 69.57 | 65.00 | 68.45 | 1087 | Rams |
| 31 | 14 | Avery Williamson | 69.56 | 64.10 | 69.55 | 908 | Titans |
| 32 | 15 | Kevin Minter | 69.07 | 67.60 | 65.89 | 1002 | Cardinals |
| 33 | 16 | Korey Toomer | 68.94 | 69.94 | 70.87 | 479 | Chargers |
| 34 | 17 | Jamie Collins Sr. | 68.84 | 68.80 | 65.85 | 980 | Browns |
| 35 | 18 | Todd Davis | 68.79 | 65.62 | 68.41 | 697 | Broncos |
| 36 | 19 | Bruce Carter | 68.72 | 68.26 | 70.38 | 121 | Jets |
| 37 | 20 | NaVorro Bowman | 68.49 | 68.64 | 70.47 | 251 | 49ers |
| 38 | 21 | Derrick Johnson | 68.46 | 70.50 | 67.62 | 841 | Chiefs |
| 39 | 22 | Max Bullough | 68.37 | 64.65 | 69.39 | 240 | Texans |
| 40 | 23 | Joshua Perry | 68.27 | 67.56 | 72.91 | 114 | Chargers |
| 41 | 24 | Karlos Dansby | 68.27 | 64.93 | 67.16 | 782 | Bengals |
| 42 | 25 | Thomas Davis Sr. | 68.16 | 66.40 | 65.16 | 1008 | Panthers |
| 43 | 26 | Anthony Hitchens | 67.77 | 64.04 | 66.09 | 632 | Cowboys |
| 44 | 27 | Ryan Shazier | 67.38 | 67.20 | 65.42 | 968 | Steelers |
| 45 | 28 | Lavonte David | 67.28 | 63.70 | 65.91 | 1042 | Buccaneers |
| 46 | 29 | Sean Spence | 67.04 | 63.29 | 66.84 | 503 | Titans |
| 47 | 30 | Myles Jack | 66.73 | 62.43 | 67.51 | 239 | Jaguars |
| 48 | 31 | Jake Ryan | 65.71 | 61.10 | 66.57 | 686 | Packers |
| 49 | 32 | Damien Wilson | 65.68 | 65.70 | 68.92 | 289 | Cowboys |
| 50 | 33 | Demario Davis | 65.42 | 61.47 | 63.88 | 786 | Browns |
| 51 | 34 | Elandon Roberts | 65.37 | 61.49 | 64.82 | 344 | Patriots |
| 52 | 35 | Danny Trevathan | 65.35 | 64.49 | 68.10 | 565 | Bears |
| 53 | 36 | Alec Ogletree | 65.26 | 63.30 | 66.15 | 1090 | Rams |
| 54 | 37 | Preston Brown | 64.81 | 60.20 | 63.72 | 1066 | Bills |
| 55 | 38 | Corey Nelson | 64.77 | 61.14 | 65.73 | 543 | Broncos |
| 56 | 39 | Kiko Alonso | 64.76 | 61.30 | 64.46 | 1106 | Dolphins |
| 57 | 40 | Lawrence Timmons | 64.70 | 59.20 | 64.20 | 1145 | Steelers |
| 58 | 41 | Vince Williams | 64.46 | 61.54 | 65.26 | 268 | Steelers |
| 59 | 42 | Deone Bucannon | 64.38 | 61.40 | 65.33 | 819 | Cardinals |
| 60 | 43 | DeAndre Levy | 64.16 | 63.57 | 70.29 | 248 | Lions |
| 61 | 44 | Jonathan Casillas | 64.12 | 59.70 | 64.25 | 850 | Giants |
| 62 | 45 | Spencer Paysinger | 64.08 | 63.25 | 64.63 | 334 | Dolphins |
| 63 | 46 | Blake Martinez | 63.93 | 58.32 | 63.51 | 480 | Packers |
| 64 | 47 | Mychal Kendricks | 63.88 | 62.98 | 62.60 | 273 | Eagles |
| 65 | 48 | Craig Robertson | 63.87 | 59.90 | 64.12 | 970 | Saints |
| 66 | 49 | Paul Worrilow | 63.64 | 59.82 | 65.45 | 167 | Falcons |
| 67 | 50 | Daryl Smith | 63.61 | 57.86 | 63.28 | 476 | Buccaneers |
| 68 | 51 | Philip Wheeler | 63.27 | 59.53 | 64.00 | 367 | Falcons |
| 69 | 52 | Michael Morgan | 63.22 | 62.13 | 66.44 | 175 | Seahawks |
| 70 | 53 | Joe Thomas | 62.95 | 57.02 | 62.74 | 809 | Packers |
| 71 | 54 | Kyle Van Noy | 62.60 | 61.22 | 62.37 | 624 | Patriots |
| 72 | 55 | Antonio Morrison | 62.43 | 57.51 | 63.63 | 333 | Colts |
| 73 | 56 | Sio Moore | 62.39 | 63.04 | 65.19 | 411 | Cardinals |
| 74 | 57 | Brock Coyle | 62.33 | 62.32 | 68.81 | 128 | Seahawks |
| 75 | 58 | Malcolm Smith | 62.03 | 55.50 | 63.47 | 971 | Raiders |

### Rotation/backup (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 76 | 1 | Denzel Perryman | 61.86 | 59.46 | 63.08 | 481 | Chargers |
| 77 | 2 | Justin Durant | 61.37 | 56.74 | 64.36 | 298 | Cowboys |
| 78 | 3 | Akeem Dent | 61.36 | 59.92 | 65.34 | 115 | Texans |
| 79 | 4 | Edwin Jackson | 61.32 | 60.16 | 64.18 | 495 | Colts |
| 80 | 5 | Nick Kwiatkoski | 61.31 | 59.33 | 61.60 | 458 | Bears |
| 81 | 6 | Chad Greenway | 60.94 | 56.37 | 60.66 | 399 | Vikings |
| 82 | 7 | Josh Bynes | 60.40 | 57.92 | 61.85 | 421 | Lions |
| 83 | 8 | Donald Butler | 60.25 | 51.92 | 63.52 | 379 | Dolphins |
| 84 | 9 | Koa Misi | 59.86 | 60.19 | 64.23 | 128 | Dolphins |
| 85 | 10 | De'Vondre Campbell | 59.63 | 55.98 | 59.98 | 715 | Falcons |
| 86 | 11 | Neville Hewitt | 59.50 | 52.84 | 61.72 | 374 | Dolphins |
| 87 | 12 | Will Compton | 59.46 | 53.70 | 60.49 | 937 | Commanders |
| 88 | 13 | Zachary Orr | 59.23 | 54.80 | 62.38 | 961 | Ravens |
| 89 | 14 | Keenan Robinson | 58.54 | 49.80 | 62.09 | 844 | Giants |
| 90 | 15 | Tahir Whitehead | 58.21 | 48.80 | 60.31 | 998 | Lions |
| 91 | 16 | Nick Bellore | 58.02 | 59.36 | 62.12 | 692 | 49ers |
| 92 | 17 | Cory Littleton | 57.98 | 58.16 | 59.94 | 123 | Rams |
| 93 | 18 | Nate Stupar | 57.90 | 54.71 | 60.44 | 377 | Saints |
| 94 | 19 | Michael Wilhoite | 57.64 | 48.01 | 62.18 | 510 | 49ers |
| 95 | 20 | Sean Weatherspoon | 57.60 | 58.41 | 61.02 | 190 | Falcons |
| 96 | 21 | LaRoy Reynolds | 57.55 | 54.91 | 62.32 | 143 | Falcons |
| 97 | 22 | Dannell Ellerbe | 57.47 | 58.97 | 62.20 | 443 | Saints |
| 98 | 23 | A.J. Klein | 56.86 | 51.44 | 58.80 | 350 | Panthers |
| 99 | 24 | Anthony Barr | 56.78 | 50.60 | 57.89 | 1025 | Vikings |
| 100 | 25 | Ben Heeney | 56.74 | 57.73 | 61.30 | 135 | Raiders |
| 101 | 26 | D'Qwell Jackson | 56.60 | 49.85 | 59.02 | 708 | Colts |
| 102 | 27 | Nick Vigil | 56.42 | 54.56 | 58.69 | 111 | Bengals |
| 103 | 28 | Justin March-Lillard | 56.32 | 60.53 | 65.31 | 159 | Chiefs |
| 104 | 29 | Mike Hull | 56.21 | 59.20 | 64.05 | 111 | Dolphins |
| 105 | 30 | Cory James | 55.96 | 50.17 | 58.78 | 410 | Raiders |
| 106 | 31 | Darron Lee | 55.22 | 47.18 | 59.54 | 641 | Jets |
| 107 | 32 | Antwione Williams | 55.02 | 53.27 | 56.18 | 204 | Lions |
| 108 | 33 | Stephone Anthony | 54.40 | 50.01 | 58.36 | 133 | Saints |
| 109 | 34 | Kelvin Sheppard | 54.36 | 44.36 | 58.94 | 459 | Giants |
| 110 | 35 | Manti Te'o | 54.05 | 55.38 | 58.27 | 142 | Chargers |
| 111 | 36 | Rey Maualuga | 53.94 | 45.69 | 56.94 | 326 | Bengals |
| 112 | 37 | John Timu | 53.51 | 55.88 | 59.34 | 184 | Bears |
| 113 | 38 | Martrell Spaight | 53.49 | 53.24 | 56.85 | 150 | Commanders |
| 114 | 39 | Erin Henderson | 52.57 | 48.11 | 58.77 | 168 | Jets |
| 115 | 40 | Josh McNary | 51.54 | 47.65 | 56.63 | 178 | Colts |
| 116 | 41 | Jelani Jenkins | 49.04 | 40.00 | 55.17 | 406 | Dolphins |
| 117 | 42 | Julian Stanford | 46.90 | 43.36 | 54.16 | 244 | Jets |

## QB — Quarterback

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Matt Ryan | 87.67 | 87.96 | 82.88 | 738 | Falcons |
| 2 | 2 | Tom Brady | 87.57 | 91.97 | 79.43 | 654 | Patriots |
| 3 | 3 | Aaron Rodgers | 82.89 | 86.42 | 75.46 | 882 | Packers |
| 4 | 4 | Drew Brees | 82.68 | 83.65 | 77.34 | 739 | Saints |
| 5 | 5 | Ben Roethlisberger | 81.03 | 82.47 | 75.61 | 674 | Steelers |
| 6 | 6 | Russell Wilson | 80.43 | 81.58 | 74.95 | 741 | Seahawks |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Andrew Luck | 78.76 | 82.27 | 73.40 | 677 | Colts |
| 8 | 2 | Kirk Cousins | 78.65 | 76.93 | 76.69 | 675 | Commanders |
| 9 | 3 | Derek Carr | 78.07 | 81.21 | 71.24 | 643 | Raiders |
| 10 | 4 | Sam Bradford | 76.68 | 75.74 | 74.39 | 626 | Vikings |
| 11 | 5 | Carson Palmer | 74.98 | 77.26 | 69.67 | 694 | Cardinals |
| 12 | 6 | Matthew Stafford | 74.48 | 74.44 | 69.96 | 742 | Lions |

### Starter (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Andy Dalton | 73.61 | 71.70 | 72.08 | 660 | Bengals |
| 14 | 2 | Alex Smith | 73.27 | 74.94 | 68.12 | 607 | Chiefs |
| 15 | 3 | Ryan Tannehill | 72.81 | 70.81 | 73.27 | 456 | Dolphins |
| 16 | 4 | Philip Rivers | 72.37 | 70.52 | 69.33 | 659 | Chargers |
| 17 | 5 | Dak Prescott | 72.33 | 81.38 | 76.00 | 586 | Cowboys |
| 18 | 6 | Jameis Winston | 70.99 | 70.93 | 66.58 | 689 | Buccaneers |
| 19 | 7 | Cam Newton | 70.96 | 74.12 | 64.86 | 596 | Panthers |
| 20 | 8 | Tyrod Taylor | 70.90 | 74.29 | 67.88 | 567 | Bills |
| 21 | 9 | Marcus Mariota | 69.72 | 65.44 | 72.56 | 541 | Titans |
| 22 | 10 | Eli Manning | 67.19 | 64.31 | 65.00 | 705 | Giants |
| 23 | 11 | Joe Flacco | 66.44 | 66.35 | 63.29 | 759 | Ravens |
| 24 | 12 | Brian Hoyer | 64.95 | 65.14 | 69.75 | 216 | Bears |
| 25 | 13 | Carson Wentz | 64.69 | 68.30 | 60.24 | 700 | Eagles |

### Rotation/backup (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Cody Kessler | 61.86 | 67.48 | 68.45 | 235 | Browns |
| 27 | 2 | Matt Moore | 61.76 | 59.16 | 76.87 | 136 | Dolphins |
| 28 | 3 | Jay Cutler | 61.44 | 62.48 | 65.40 | 167 | Bears |
| 29 | 4 | Blake Bortles | 61.00 | 56.45 | 60.35 | 745 | Jaguars |
| 30 | 5 | Matt Barkley | 60.67 | 68.24 | 64.88 | 241 | Bears |
| 31 | 6 | Trevor Siemian | 59.99 | 56.36 | 64.85 | 566 | Broncos |
| 32 | 7 | Colin Kaepernick | 59.69 | 55.34 | 65.12 | 421 | 49ers |
| 33 | 8 | Ryan Fitzpatrick | 59.18 | 57.82 | 59.72 | 466 | Jets |
| 34 | 9 | Case Keenum | 58.35 | 59.48 | 61.21 | 373 | Rams |
| 35 | 10 | Paxton Lynch | 56.58 | 55.92 | 58.59 | 107 | Broncos |
| 36 | 11 | Josh McCown | 55.78 | 58.65 | 59.81 | 199 | Browns |
| 37 | 12 | Bryce Petty | 55.61 | 53.52 | 56.67 | 157 | Jets |
| 38 | 13 | Blaine Gabbert | 55.46 | 55.51 | 56.94 | 198 | 49ers |
| 39 | 14 | Robert Griffin III | 55.33 | 58.54 | 59.22 | 194 | Browns |
| 40 | 15 | Brock Osweiler | 54.96 | 52.35 | 56.47 | 659 | Texans |
| 41 | 16 | Jared Goff | 54.73 | 48.95 | 54.38 | 253 | Rams |

## S — Safety

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Devin McCourty | 94.97 | 92.10 | 92.72 | 1209 | Patriots |
| 2 | 2 | Eric Weddle | 92.27 | 88.70 | 91.42 | 1031 | Ravens |
| 3 | 3 | Eric Berry | 90.67 | 90.30 | 88.83 | 1151 | Chiefs |
| 4 | 4 | Ricardo Allen | 87.20 | 85.80 | 83.97 | 1322 | Falcons |
| 5 | 5 | Keith Tandy | 86.18 | 81.29 | 88.40 | 402 | Buccaneers |
| 6 | 6 | Quintin Demps | 84.66 | 81.41 | 84.23 | 709 | Texans |
| 7 | 7 | Barry Church | 82.62 | 80.05 | 82.05 | 724 | Cowboys |
| 8 | 8 | Glover Quin | 82.59 | 79.30 | 80.61 | 1099 | Lions |
| 9 | 9 | Kam Chancellor | 82.49 | 78.50 | 82.97 | 854 | Seahawks |
| 10 | 10 | D.J. Swearinger Sr. | 82.43 | 82.80 | 79.90 | 839 | Cardinals |
| 11 | 11 | Reshad Jones | 82.11 | 82.09 | 84.01 | 437 | Dolphins |
| 12 | 12 | J.J. Wilcox | 81.60 | 78.75 | 80.36 | 573 | Cowboys |
| 13 | 13 | Keanu Neal | 80.83 | 80.30 | 77.02 | 1136 | Falcons |
| 14 | 14 | Landon Collins | 80.15 | 73.40 | 80.49 | 1176 | Giants |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Earl Thomas III | 78.56 | 79.28 | 76.51 | 693 | Seahawks |
| 16 | 2 | Tavon Wilson | 78.33 | 72.99 | 77.72 | 775 | Lions |
| 17 | 3 | Ron Parker | 77.48 | 74.50 | 75.30 | 1168 | Chiefs |
| 18 | 4 | Mike Mitchell | 77.46 | 71.20 | 77.46 | 1197 | Steelers |
| 19 | 5 | Kurt Coleman | 77.44 | 72.50 | 77.30 | 996 | Panthers |
| 20 | 6 | Daniel Sorensen | 77.26 | 77.08 | 77.38 | 573 | Chiefs |
| 21 | 7 | Morgan Burnett | 77.22 | 71.30 | 77.93 | 1096 | Packers |
| 22 | 8 | Maurice Alexander | 77.13 | 79.80 | 73.79 | 920 | Rams |
| 23 | 9 | Jeff Heath | 76.82 | 70.45 | 79.81 | 263 | Cowboys |
| 24 | 10 | Jahleel Addae | 76.71 | 75.96 | 79.20 | 510 | Chargers |
| 25 | 11 | Justin Simmons | 76.32 | 69.19 | 81.07 | 294 | Broncos |
| 26 | 12 | Malcolm Jenkins | 76.02 | 74.70 | 72.74 | 1018 | Eagles |
| 27 | 13 | Darian Stewart | 75.59 | 72.80 | 73.29 | 1080 | Broncos |
| 28 | 14 | Dwight Lowery | 75.34 | 73.10 | 72.66 | 1003 | Chargers |
| 29 | 15 | Eddie Pleasant | 75.29 | 73.81 | 74.19 | 285 | Texans |
| 30 | 16 | Mike Adams | 75.21 | 70.10 | 75.91 | 997 | Colts |
| 31 | 17 | Reggie Nelson | 75.04 | 69.30 | 74.70 | 1121 | Raiders |
| 32 | 18 | Jairus Byrd | 75.03 | 70.50 | 77.31 | 900 | Saints |
| 33 | 19 | George Iloka | 74.72 | 69.40 | 75.03 | 1050 | Bengals |
| 34 | 20 | Rodney McLeod | 74.57 | 70.10 | 73.39 | 1014 | Eagles |

### Starter (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Duron Harmon | 73.99 | 70.38 | 72.23 | 617 | Patriots |
| 36 | 2 | Harrison Smith | 73.97 | 68.50 | 75.11 | 894 | Vikings |
| 37 | 3 | Ha Ha Clinton-Dix | 73.77 | 63.10 | 76.72 | 1231 | Packers |
| 38 | 4 | Adrian Amos | 73.52 | 70.30 | 72.54 | 938 | Bears |
| 39 | 5 | Isa Abdul-Quddus | 73.37 | 70.20 | 72.47 | 952 | Dolphins |
| 40 | 6 | Shawn Williams | 72.14 | 67.00 | 73.49 | 912 | Bengals |
| 41 | 7 | Rontez Miles | 72.01 | 64.42 | 75.41 | 393 | Jets |
| 42 | 8 | Karl Joseph | 71.86 | 66.17 | 76.69 | 623 | Raiders |
| 43 | 9 | Corey Graham | 71.08 | 67.90 | 69.04 | 1053 | Bills |
| 44 | 10 | Kevin Byard | 70.96 | 65.44 | 70.47 | 656 | Titans |
| 45 | 11 | Tre Boston | 70.14 | 65.40 | 71.01 | 840 | Panthers |
| 46 | 12 | Da'Norris Searcy | 70.05 | 65.65 | 70.39 | 553 | Titans |
| 47 | 13 | T.J. McDonald | 69.66 | 68.80 | 67.63 | 1072 | Rams |
| 48 | 14 | Dexter McCoil | 69.18 | 66.69 | 71.87 | 248 | Chargers |
| 49 | 15 | Tony Jefferson | 69.04 | 59.50 | 72.27 | 931 | Cardinals |
| 50 | 16 | Clayton Geathers | 68.35 | 64.53 | 72.07 | 559 | Colts |
| 51 | 17 | T.J. Ward | 68.16 | 65.20 | 67.31 | 984 | Broncos |
| 52 | 18 | Bradley McDougald | 67.99 | 62.90 | 67.63 | 1012 | Buccaneers |
| 53 | 19 | Corey Moore | 67.85 | 60.70 | 70.53 | 501 | Texans |
| 54 | 20 | Miles Killebrew | 67.69 | 62.10 | 68.29 | 158 | Lions |
| 55 | 21 | Kentrell Brice | 67.22 | 60.06 | 67.82 | 334 | Packers |
| 56 | 22 | Marcus Gilchrist | 67.15 | 60.20 | 69.18 | 819 | Jets |
| 57 | 23 | Chris Prosinski | 67.11 | 60.92 | 73.42 | 173 | Bears |
| 58 | 24 | Johnathan Cyprien | 67.02 | 59.10 | 68.97 | 1070 | Jaguars |
| 59 | 25 | Kenny Vaccaro | 66.56 | 61.68 | 68.47 | 720 | Saints |
| 60 | 26 | Harold Jones-Quartey | 65.21 | 57.27 | 67.37 | 731 | Bears |
| 61 | 27 | Tyrann Mathieu | 64.96 | 60.33 | 67.43 | 561 | Cardinals |
| 62 | 28 | Nate Allen | 64.91 | 62.59 | 70.62 | 232 | Raiders |
| 63 | 29 | Rashad Johnson | 64.85 | 60.54 | 64.59 | 555 | Titans |
| 64 | 30 | Andre Hal | 64.71 | 59.50 | 64.33 | 949 | Texans |
| 65 | 31 | Rafael Bush | 64.17 | 60.24 | 68.55 | 512 | Lions |
| 66 | 32 | Daimion Stafford | 64.06 | 59.05 | 65.12 | 613 | Titans |
| 67 | 33 | Jaylen Watkins | 63.75 | 55.69 | 65.48 | 387 | Eagles |
| 68 | 34 | Jaquiski Tartt | 63.40 | 59.22 | 63.06 | 612 | 49ers |
| 69 | 35 | Andrew Sendejo | 63.37 | 60.40 | 64.10 | 855 | Vikings |
| 70 | 36 | Sean Davis | 63.31 | 60.50 | 61.01 | 930 | Steelers |
| 71 | 37 | Cody Davis | 63.29 | 62.79 | 66.03 | 278 | Rams |
| 72 | 38 | Antoine Bethea | 62.82 | 54.10 | 67.28 | 1125 | 49ers |
| 73 | 39 | Eric Reid | 62.24 | 56.02 | 65.55 | 742 | 49ers |
| 74 | 40 | Kemal Ishmael | 62.22 | 57.43 | 64.48 | 310 | Falcons |
| 75 | 41 | Duke Ihenacho | 62.00 | 66.51 | 63.67 | 638 | Commanders |

### Rotation/backup (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 76 | 1 | Donte Whitner | 61.99 | 53.55 | 66.69 | 564 | Commanders |
| 77 | 2 | Jordan Dangerfield | 61.94 | 64.02 | 63.69 | 145 | Steelers |
| 78 | 3 | Jordan Poyer | 61.91 | 56.00 | 68.98 | 354 | Browns |
| 79 | 4 | Will Parks | 61.73 | 56.07 | 62.37 | 268 | Broncos |
| 80 | 5 | Derron Smith | 61.33 | 60.80 | 65.98 | 103 | Bengals |
| 81 | 6 | Tashaun Gipson Sr. | 60.19 | 47.40 | 66.53 | 1040 | Jaguars |
| 82 | 7 | Deon Bush | 60.13 | 61.33 | 63.50 | 333 | Bears |
| 83 | 8 | Michael Thomas | 59.61 | 53.85 | 61.16 | 620 | Dolphins |
| 84 | 9 | Vonn Bell | 59.19 | 44.60 | 65.79 | 889 | Saints |
| 85 | 10 | Robert Blanton | 59.16 | 55.82 | 61.80 | 270 | Bills |
| 86 | 11 | Roman Harper | 59.09 | 57.23 | 58.24 | 300 | Saints |
| 87 | 12 | Patrick Chung | 58.88 | 48.30 | 61.77 | 1176 | Patriots |
| 88 | 13 | Adrian Phillips | 58.47 | 48.81 | 62.83 | 542 | Chargers |
| 89 | 14 | Derrick Kindred | 58.27 | 55.40 | 60.18 | 538 | Browns |
| 90 | 15 | James Ihedigbo | 58.22 | 59.15 | 60.42 | 146 | Bills |
| 91 | 16 | Michael Griffin | 57.61 | 55.31 | 57.90 | 284 | Panthers |
| 92 | 17 | Aaron Williams | 57.29 | 56.62 | 62.54 | 340 | Bills |
| 93 | 18 | Robert Golden | 57.23 | 53.47 | 59.93 | 382 | Steelers |
| 94 | 19 | Chris Conte | 55.98 | 49.09 | 58.90 | 718 | Buccaneers |
| 95 | 20 | Ed Reynolds Jr. | 55.79 | 52.65 | 62.05 | 505 | Browns |
| 96 | 21 | Bacarri Rambo | 55.20 | 50.91 | 59.41 | 525 | Dolphins |
| 97 | 22 | Calvin Pryor | 55.11 | 46.55 | 58.32 | 814 | Jets |
| 98 | 23 | Anthony Harris | 55.01 | 51.09 | 64.39 | 235 | Vikings |
| 99 | 24 | Ibraheim Campbell | 53.65 | 53.74 | 57.62 | 418 | Browns |
| 100 | 25 | Keith McGill | 53.26 | 51.73 | 53.76 | 150 | Raiders |
| 101 | 26 | Kelcie McCray | 52.30 | 50.44 | 55.94 | 344 | Seahawks |
| 102 | 27 | Nat Berhe | 52.20 | 54.14 | 56.50 | 165 | Giants |
| 103 | 28 | T.J. Green | 48.67 | 40.00 | 51.32 | 478 | Colts |
| 104 | 29 | Blake Countess | 45.00 | 47.10 | 55.63 | 142 | Rams |

## T — Tackle

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Trent Williams | 95.96 | 90.71 | 95.29 | 795 | Commanders |
| 2 | 2 | Donald Penn | 94.82 | 90.00 | 93.86 | 1108 | Raiders |
| 3 | 3 | Taylor Lewan | 93.12 | 87.00 | 93.03 | 991 | Titans |
| 4 | 4 | Marcus Cannon | 92.74 | 86.60 | 92.67 | 1272 | Patriots |
| 5 | 5 | Jason Peters | 92.06 | 86.90 | 91.33 | 1098 | Eagles |
| 6 | 6 | Ryan Schraeder | 90.53 | 83.40 | 91.12 | 1223 | Falcons |
| 7 | 7 | Tyron Smith | 90.34 | 85.23 | 89.58 | 902 | Cowboys |
| 8 | 8 | Joe Thomas | 89.94 | 87.00 | 87.73 | 1030 | Browns |
| 9 | 9 | Duane Brown | 89.71 | 85.66 | 88.25 | 918 | Texans |
| 10 | 10 | Andrew Whitworth | 89.29 | 84.00 | 88.65 | 1064 | Bengals |
| 11 | 11 | David Bakhtiari | 89.02 | 86.60 | 86.47 | 1257 | Packers |
| 12 | 12 | Lane Johnson | 88.88 | 79.17 | 91.18 | 407 | Eagles |
| 13 | 13 | Anthony Castonzo | 87.77 | 83.40 | 86.51 | 1074 | Colts |
| 14 | 14 | Nate Solder | 87.57 | 82.20 | 86.99 | 1270 | Patriots |
| 15 | 15 | Taylor Decker | 87.00 | 81.90 | 86.23 | 1089 | Lions |
| 16 | 16 | Jack Conklin | 86.66 | 80.60 | 86.54 | 1062 | Titans |
| 17 | 17 | Marcus Gilbert | 86.01 | 78.50 | 86.85 | 1038 | Steelers |
| 18 | 18 | Joe Staley | 85.97 | 80.17 | 85.67 | 845 | 49ers |
| 19 | 19 | Terron Armstead | 85.82 | 77.65 | 87.10 | 397 | Saints |
| 20 | 20 | Eric Fisher | 85.59 | 77.70 | 86.68 | 1078 | Chiefs |
| 21 | 21 | Alejandro Villanueva | 85.47 | 79.90 | 85.02 | 1276 | Steelers |
| 22 | 22 | Zach Strief | 84.69 | 77.40 | 85.38 | 1125 | Saints |
| 23 | 23 | Morgan Moses | 84.50 | 76.80 | 85.46 | 1017 | Commanders |
| 24 | 24 | Cordy Glenn | 84.35 | 76.11 | 85.67 | 657 | Bills |
| 25 | 25 | Bryan Bulaga | 83.74 | 78.10 | 83.33 | 1256 | Packers |
| 26 | 26 | Demar Dotson | 83.61 | 74.01 | 85.85 | 942 | Buccaneers |
| 27 | 27 | Ja'Wuan James | 83.59 | 75.20 | 85.02 | 1000 | Dolphins |
| 28 | 28 | Jared Veldheer | 83.14 | 74.18 | 84.94 | 578 | Cardinals |
| 29 | 29 | Jermey Parnell | 82.91 | 73.00 | 85.35 | 1113 | Jaguars |
| 30 | 30 | Mitchell Schwartz | 82.29 | 73.40 | 84.05 | 1078 | Chiefs |
| 31 | 31 | Ty Nsekhe | 82.10 | 71.44 | 85.04 | 385 | Commanders |
| 32 | 32 | Russell Okung | 82.03 | 74.00 | 83.21 | 1061 | Broncos |
| 33 | 33 | Ronnie Stanley | 81.74 | 73.62 | 82.98 | 834 | Ravens |
| 34 | 34 | Rick Wagner | 81.37 | 73.74 | 82.29 | 926 | Ravens |
| 35 | 35 | Cam Fleming | 81.34 | 67.52 | 86.39 | 301 | Patriots |
| 36 | 36 | Austin Pasztor | 80.42 | 71.80 | 82.00 | 1020 | Browns |
| 37 | 37 | Mike Remmers | 80.14 | 71.20 | 81.94 | 1106 | Panthers |
| 38 | 38 | Jake Matthews | 80.08 | 71.90 | 81.37 | 1172 | Falcons |

### Good (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Cyrus Kouandjio | 79.82 | 69.50 | 82.53 | 406 | Bills |
| 40 | 2 | Ereck Flowers | 79.51 | 69.40 | 82.08 | 1125 | Giants |
| 41 | 3 | D.J. Humphries | 79.18 | 68.03 | 82.45 | 922 | Cardinals |
| 42 | 4 | Charles Leno Jr. | 79.18 | 70.10 | 81.06 | 1011 | Bears |
| 43 | 5 | Riley Reiff | 78.08 | 68.82 | 80.08 | 888 | Lions |
| 44 | 6 | Doug Free | 77.93 | 67.50 | 80.72 | 1126 | Cowboys |
| 45 | 7 | Austin Howard | 77.30 | 66.75 | 80.16 | 792 | Raiders |
| 46 | 8 | Halapoulivaati Vaitai | 77.30 | 65.75 | 80.84 | 421 | Eagles |
| 47 | 9 | Menelik Watson | 77.29 | 62.54 | 82.95 | 328 | Raiders |
| 48 | 10 | Marshall Newhouse | 76.88 | 64.08 | 81.24 | 524 | Giants |
| 49 | 11 | Daryl Williams | 76.64 | 66.08 | 79.52 | 647 | Panthers |
| 50 | 12 | King Dunlap | 76.57 | 67.55 | 78.42 | 775 | Chargers |
| 51 | 13 | Bobby Massie | 76.45 | 66.56 | 78.88 | 916 | Bears |
| 52 | 14 | Eric Winston | 76.39 | 63.82 | 80.61 | 282 | Bengals |
| 53 | 15 | Donovan Smith | 76.01 | 63.90 | 79.91 | 1135 | Buccaneers |
| 54 | 16 | Kendall Lamm | 75.30 | 63.50 | 79.00 | 167 | Texans |
| 55 | 17 | Ryan Clady | 75.30 | 64.89 | 78.08 | 537 | Jets |
| 56 | 18 | Le'Raven Clark | 75.11 | 61.94 | 79.72 | 201 | Colts |
| 57 | 19 | Brent Qvale | 74.60 | 61.42 | 79.22 | 347 | Jets |
| 58 | 20 | Trent Brown | 74.31 | 61.90 | 78.42 | 1035 | 49ers |
| 59 | 21 | Chris Hubbard | 74.16 | 64.78 | 76.25 | 351 | Steelers |

### Starter (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 60 | 1 | Ben Ijalana | 73.96 | 63.57 | 76.72 | 867 | Jets |
| 61 | 2 | Jake Long | 73.39 | 61.83 | 76.93 | 209 | Vikings |
| 62 | 3 | Derek Newton | 73.32 | 60.99 | 77.38 | 358 | Texans |
| 63 | 4 | Jordan Mills | 72.97 | 61.80 | 76.25 | 1033 | Bills |
| 64 | 5 | Vadal Alexander | 72.91 | 57.27 | 79.17 | 306 | Raiders |
| 65 | 6 | Cedric Ogbuehi | 72.87 | 60.50 | 76.95 | 677 | Bengals |
| 66 | 7 | Dennis Kelly | 72.73 | 60.28 | 76.86 | 146 | Titans |
| 67 | 8 | Kelvin Beachum | 72.56 | 61.40 | 75.83 | 1024 | Jaguars |
| 68 | 9 | Michael Oher | 72.38 | 61.55 | 75.43 | 232 | Panthers |
| 69 | 10 | Branden Albert | 72.29 | 61.40 | 75.38 | 786 | Dolphins |
| 70 | 11 | John Wetzel | 72.12 | 59.11 | 76.63 | 647 | Cardinals |
| 71 | 12 | James Hurst | 72.00 | 60.50 | 75.50 | 305 | Ravens |
| 72 | 13 | Jah Reid | 71.85 | 58.52 | 76.57 | 101 | Chiefs |
| 73 | 14 | Andrew Donnal | 71.79 | 61.92 | 74.21 | 297 | Rams |
| 74 | 15 | Chris Hairston | 71.66 | 60.00 | 75.27 | 327 | Chargers |
| 75 | 16 | Greg Robinson | 71.29 | 59.90 | 74.72 | 892 | Rams |
| 76 | 17 | Sam Young | 71.21 | 59.64 | 74.76 | 146 | Dolphins |
| 77 | 18 | Bobby Hart | 70.99 | 56.53 | 76.46 | 868 | Giants |
| 78 | 19 | Ulrick John | 70.82 | 56.29 | 76.34 | 212 | Cardinals |
| 79 | 20 | Chris Clark | 70.56 | 56.90 | 75.50 | 1219 | Texans |
| 80 | 21 | Brandon Shell | 70.54 | 66.10 | 69.34 | 204 | Jets |
| 81 | 22 | Garry Gilliam | 70.09 | 56.78 | 74.80 | 939 | Seahawks |
| 82 | 23 | Joe Barksdale | 70.08 | 56.32 | 75.08 | 967 | Chargers |
| 83 | 24 | Corey Robinson | 69.45 | 56.64 | 73.82 | 165 | Lions |
| 84 | 25 | Jake Fisher | 69.31 | 53.26 | 75.85 | 296 | Bengals |
| 85 | 26 | Matt Kalil | 68.88 | 55.14 | 73.87 | 121 | Vikings |
| 86 | 27 | Bradley Sowell | 68.63 | 54.65 | 73.78 | 629 | Seahawks |
| 87 | 28 | Breno Giacomini | 68.27 | 55.22 | 72.81 | 266 | Jets |
| 88 | 29 | Gosder Cherilus | 65.81 | 49.35 | 72.61 | 219 | Buccaneers |
| 89 | 30 | T.J. Clemmings | 65.50 | 47.60 | 73.26 | 882 | Vikings |
| 90 | 31 | Jason Spriggs | 65.37 | 54.23 | 68.63 | 276 | Packers |
| 91 | 32 | Donald Stephenson | 65.31 | 46.18 | 73.90 | 744 | Broncos |
| 92 | 33 | Ty Sambrailo | 65.25 | 50.91 | 70.64 | 243 | Broncos |
| 93 | 34 | George Fant | 62.22 | 44.49 | 69.88 | 792 | Seahawks |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 85.68 | 82.40 | 83.70 | 214 | Patriots |
| 2 | 2 | Travis Kelce | 85.10 | 88.19 | 78.87 | 607 | Chiefs |
| 3 | 3 | Jimmy Graham | 83.95 | 84.73 | 79.27 | 603 | Seahawks |
| 4 | 4 | Greg Olsen | 81.83 | 82.72 | 77.07 | 598 | Panthers |
| 5 | 5 | Hunter Henry | 80.89 | 75.25 | 80.49 | 290 | Chargers |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Erik Swoope | 79.57 | 65.78 | 84.59 | 108 | Colts |
| 7 | 2 | Jordan Reed | 79.14 | 80.32 | 74.18 | 390 | Commanders |
| 8 | 3 | Cameron Brate | 79.13 | 77.11 | 76.31 | 460 | Buccaneers |
| 9 | 4 | Jared Cook | 77.88 | 73.37 | 76.72 | 340 | Packers |
| 10 | 5 | Delanie Walker | 77.83 | 74.58 | 75.83 | 479 | Titans |
| 11 | 6 | Jeff Heuerman | 76.93 | 66.75 | 79.55 | 130 | Broncos |
| 12 | 7 | Ladarius Green | 76.89 | 67.65 | 78.88 | 114 | Steelers |
| 13 | 8 | Vernon Davis | 76.66 | 73.46 | 74.63 | 357 | Commanders |
| 14 | 9 | Martellus Bennett | 76.06 | 75.13 | 72.51 | 608 | Patriots |
| 15 | 10 | Tyler Eifert | 75.96 | 76.21 | 71.62 | 269 | Bengals |
| 16 | 11 | C.J. Fiedorowicz | 75.34 | 74.31 | 71.86 | 395 | Texans |
| 17 | 12 | Gary Barnidge | 75.02 | 68.50 | 75.20 | 635 | Browns |
| 18 | 13 | Kyle Rudolph | 74.98 | 72.96 | 72.16 | 621 | Vikings |
| 19 | 14 | Anthony Fasano | 74.89 | 73.38 | 71.73 | 187 | Titans |
| 20 | 15 | Zach Ertz | 74.76 | 70.38 | 73.51 | 560 | Eagles |
| 21 | 16 | Zach Miller | 74.70 | 70.76 | 73.16 | 335 | Bears |
| 22 | 17 | Austin Hooper | 74.35 | 65.81 | 75.87 | 308 | Falcons |

### Starter (46 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Antonio Gates | 73.55 | 71.68 | 70.63 | 431 | Chargers |
| 24 | 2 | Charles Clay | 72.42 | 67.56 | 71.49 | 500 | Bills |
| 25 | 3 | Jacob Tamme | 72.38 | 67.61 | 71.40 | 202 | Falcons |
| 26 | 4 | Coby Fleener | 72.36 | 61.11 | 75.70 | 513 | Saints |
| 27 | 5 | Eric Ebron | 72.33 | 64.95 | 73.09 | 555 | Lions |
| 28 | 6 | Austin Seferian-Jenkins | 72.17 | 60.92 | 75.51 | 126 | Jets |
| 29 | 7 | Jason Witten | 72.10 | 68.47 | 70.36 | 580 | Cowboys |
| 30 | 8 | Darren Fells | 71.81 | 61.64 | 74.43 | 198 | Cardinals |
| 31 | 9 | Brent Celek | 71.71 | 63.89 | 72.75 | 188 | Eagles |
| 32 | 10 | Marcedes Lewis | 71.67 | 64.35 | 72.39 | 167 | Jaguars |
| 33 | 11 | Vance McDonald | 71.49 | 65.48 | 71.33 | 262 | 49ers |
| 34 | 12 | Jack Doyle | 71.01 | 72.06 | 66.14 | 421 | Colts |
| 35 | 13 | Rhett Ellison | 70.90 | 63.73 | 71.51 | 106 | Vikings |
| 36 | 14 | Mychal Rivera | 70.33 | 62.70 | 71.25 | 201 | Raiders |
| 37 | 15 | Virgil Green | 70.03 | 64.88 | 69.29 | 287 | Broncos |
| 38 | 16 | Clive Walford | 70.01 | 58.58 | 73.46 | 409 | Raiders |
| 39 | 17 | Luke Willson | 69.67 | 60.34 | 71.73 | 210 | Seahawks |
| 40 | 18 | Lance Kendricks | 69.44 | 63.21 | 69.43 | 498 | Rams |
| 41 | 19 | A.J. Derby | 69.35 | 59.31 | 71.87 | 139 | Broncos |
| 42 | 20 | Nick O'Leary | 69.14 | 63.41 | 68.80 | 206 | Bills |
| 43 | 21 | Trey Burton | 69.06 | 63.37 | 68.68 | 231 | Eagles |
| 44 | 22 | Ben Koyack | 68.57 | 61.01 | 69.44 | 188 | Jaguars |
| 45 | 23 | Neal Sterling | 68.37 | 59.58 | 70.07 | 118 | Jaguars |
| 46 | 24 | Jesse James | 68.18 | 64.50 | 66.46 | 580 | Steelers |
| 47 | 25 | Ed Dickson | 67.78 | 61.43 | 67.85 | 216 | Panthers |
| 48 | 26 | Dwayne Allen | 67.70 | 63.90 | 66.06 | 368 | Colts |
| 49 | 27 | Jerell Adams | 67.69 | 57.88 | 70.07 | 123 | Giants |
| 50 | 28 | Levine Toilolo | 67.24 | 64.28 | 65.05 | 325 | Falcons |
| 51 | 29 | Richard Rodgers | 67.04 | 57.81 | 69.02 | 405 | Packers |
| 52 | 30 | Will Tye | 67.02 | 58.30 | 68.67 | 434 | Giants |
| 53 | 31 | Josh Hill | 66.91 | 58.91 | 68.08 | 179 | Saints |
| 54 | 32 | Julius Thomas | 66.89 | 61.49 | 66.33 | 339 | Jaguars |
| 55 | 33 | Garrett Celek | 66.87 | 58.83 | 68.07 | 315 | 49ers |
| 56 | 34 | Jermaine Gresham | 66.74 | 63.46 | 64.76 | 477 | Cardinals |
| 57 | 35 | Ryan Griffin | 66.58 | 57.88 | 68.22 | 354 | Texans |
| 58 | 36 | Dennis Pitta | 66.50 | 61.75 | 65.50 | 605 | Ravens |
| 59 | 37 | Brandon Myers | 66.38 | 54.33 | 70.24 | 196 | Buccaneers |
| 60 | 38 | John Phillips | 65.74 | 59.95 | 65.43 | 179 | Saints |
| 61 | 39 | Tyler Kroft | 65.59 | 58.32 | 66.27 | 162 | Bengals |
| 62 | 40 | Dion Sims | 65.42 | 60.66 | 64.42 | 447 | Dolphins |
| 63 | 41 | Larry Donnell | 65.13 | 51.74 | 69.89 | 133 | Giants |
| 64 | 42 | C.J. Uzomah | 65.12 | 61.88 | 63.11 | 260 | Bengals |
| 65 | 43 | Xavier Grimble | 65.12 | 60.10 | 64.30 | 154 | Steelers |
| 66 | 44 | Logan Paulsen | 63.63 | 54.37 | 65.63 | 161 | Bears |
| 67 | 45 | Kellen Davis | 63.58 | 54.40 | 65.54 | 123 | Jets |
| 68 | 46 | Darren Waller | 62.20 | 56.76 | 61.66 | 134 | Ravens |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 69 | 1 | Daniel Brown | 61.86 | 59.79 | 59.08 | 181 | Bears |
| 70 | 2 | Randall Telfer | 61.61 | 57.40 | 60.25 | 112 | Browns |
| 71 | 3 | Tyler Higbee | 61.02 | 51.30 | 63.33 | 231 | Rams |
| 72 | 4 | Demetrius Harris | 58.83 | 52.14 | 59.13 | 253 | Chiefs |

## WR — Wide Receiver

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Julio Jones | 90.09 | 92.10 | 84.58 | 571 | Falcons |
| 2 | 2 | Antonio Brown | 86.90 | 88.40 | 81.73 | 711 | Steelers |
| 3 | 3 | A.J. Green | 86.28 | 85.31 | 82.76 | 361 | Bengals |
| 4 | 4 | T.Y. Hilton | 85.51 | 86.69 | 80.56 | 659 | Colts |
| 5 | 5 | Mike Evans | 85.44 | 91.26 | 77.39 | 630 | Buccaneers |
| 6 | 6 | Odell Beckham Jr. | 84.62 | 84.60 | 80.47 | 680 | Giants |
| 7 | 7 | Doug Baldwin | 83.06 | 86.10 | 76.87 | 692 | Seahawks |
| 8 | 8 | Michael Thomas | 82.94 | 84.90 | 77.47 | 587 | Saints |
| 9 | 9 | Dez Bryant | 82.72 | 81.42 | 79.42 | 480 | Cowboys |
| 10 | 10 | Emmanuel Sanders | 82.39 | 80.64 | 79.39 | 575 | Broncos |
| 11 | 11 | Jordy Nelson | 82.23 | 82.20 | 78.09 | 773 | Packers |
| 12 | 12 | Jarvis Landry | 81.71 | 83.58 | 76.29 | 572 | Dolphins |
| 13 | 13 | Rishard Matthews | 81.53 | 78.15 | 79.61 | 475 | Titans |
| 14 | 14 | Taylor Gabriel | 81.41 | 78.06 | 79.47 | 321 | Falcons |
| 15 | 15 | Pierre Garcon | 81.37 | 84.31 | 75.24 | 561 | Commanders |
| 16 | 16 | DeSean Jackson | 80.88 | 72.12 | 82.55 | 520 | Commanders |
| 17 | 17 | Adam Thielen | 80.44 | 77.75 | 78.07 | 525 | Vikings |
| 18 | 18 | Demaryius Thomas | 80.03 | 79.73 | 76.07 | 590 | Broncos |

### Good (44 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Larry Fitzgerald | 79.95 | 83.30 | 73.55 | 701 | Cardinals |
| 20 | 2 | Cole Beasley | 79.81 | 81.62 | 74.44 | 460 | Cowboys |
| 21 | 3 | Willie Snead IV | 79.59 | 78.18 | 76.36 | 510 | Saints |
| 22 | 4 | DeAndre Hopkins | 79.38 | 77.80 | 76.27 | 731 | Texans |
| 23 | 5 | Tyrell Williams | 79.11 | 74.86 | 77.78 | 599 | Chargers |
| 24 | 6 | Alshon Jeffery | 79.05 | 73.08 | 78.86 | 462 | Bears |
| 25 | 7 | Kenny Britt | 78.85 | 73.12 | 78.50 | 529 | Rams |
| 26 | 8 | Amari Cooper | 78.64 | 76.70 | 75.77 | 678 | Raiders |
| 27 | 9 | Brandin Cooks | 78.47 | 72.89 | 78.03 | 641 | Saints |
| 28 | 10 | Julian Edelman | 78.46 | 80.30 | 73.06 | 757 | Patriots |
| 29 | 11 | Kelvin Benjamin | 78.46 | 75.29 | 76.40 | 508 | Panthers |
| 30 | 12 | Stefon Diggs | 78.30 | 77.59 | 74.60 | 498 | Vikings |
| 31 | 13 | DeVante Parker | 78.23 | 73.13 | 77.46 | 499 | Dolphins |
| 32 | 14 | Sammy Watkins | 78.12 | 70.38 | 79.12 | 235 | Bills |
| 33 | 15 | Steve Smith | 77.75 | 73.31 | 76.54 | 492 | Ravens |
| 34 | 16 | Marvin Jones Jr. | 77.25 | 72.48 | 76.26 | 657 | Lions |
| 35 | 17 | Terrance Williams | 76.82 | 71.42 | 76.26 | 509 | Cowboys |
| 36 | 18 | Mohamed Sanu | 76.53 | 78.73 | 70.89 | 579 | Falcons |
| 37 | 19 | John Brown | 76.29 | 70.06 | 76.27 | 426 | Cardinals |
| 38 | 20 | Terrelle Pryor Sr. | 76.25 | 74.35 | 73.35 | 629 | Browns |
| 39 | 21 | J.J. Nelson | 76.23 | 67.06 | 78.17 | 364 | Cardinals |
| 40 | 22 | Randall Cobb | 76.17 | 74.00 | 73.45 | 601 | Packers |
| 41 | 23 | Cameron Meredith | 76.06 | 71.38 | 75.01 | 455 | Bears |
| 42 | 24 | Aldrick Robinson | 76.06 | 67.14 | 77.84 | 174 | Falcons |
| 43 | 25 | Kenny Stills | 75.77 | 68.28 | 76.59 | 536 | Dolphins |
| 44 | 26 | Tyreek Hill | 75.68 | 72.81 | 73.43 | 285 | Chiefs |
| 45 | 27 | Kendall Wright | 75.51 | 69.22 | 75.53 | 240 | Titans |
| 46 | 28 | Michael Crabtree | 75.36 | 78.16 | 69.33 | 609 | Raiders |
| 47 | 29 | Dontrelle Inman | 75.17 | 71.84 | 73.23 | 612 | Chargers |
| 48 | 30 | Russell Shepard | 75.07 | 71.10 | 73.55 | 240 | Buccaneers |
| 49 | 31 | Mike Wallace | 75.02 | 70.00 | 74.20 | 617 | Ravens |
| 50 | 32 | Tyler Lockett | 75.01 | 67.82 | 75.64 | 420 | Seahawks |
| 51 | 33 | Chris Hogan | 74.99 | 65.70 | 77.02 | 637 | Patriots |
| 52 | 34 | Geronimo Allison | 74.85 | 63.80 | 78.05 | 225 | Packers |
| 53 | 35 | Golden Tate | 74.84 | 70.60 | 73.50 | 676 | Lions |
| 54 | 36 | Paul Richardson Jr. | 74.81 | 70.44 | 73.56 | 274 | Seahawks |
| 55 | 37 | Allen Robinson II | 74.80 | 69.80 | 73.97 | 711 | Jaguars |
| 56 | 38 | Ted Ginn Jr. | 74.80 | 68.29 | 74.97 | 461 | Panthers |
| 57 | 39 | Marqise Lee | 74.72 | 71.23 | 72.88 | 588 | Jaguars |
| 58 | 40 | Brandon Marshall | 74.48 | 68.85 | 74.06 | 580 | Jets |
| 59 | 41 | Davante Adams | 74.31 | 72.60 | 71.29 | 768 | Packers |
| 60 | 42 | Brandon LaFell | 74.19 | 69.39 | 73.22 | 635 | Bengals |
| 61 | 43 | Eric Decker | 74.05 | 67.67 | 74.14 | 131 | Jets |
| 62 | 44 | Travis Benjamin | 74.04 | 65.68 | 75.45 | 411 | Chargers |

### Starter (77 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 63 | 1 | Donte Moncrief | 73.30 | 70.85 | 70.77 | 308 | Colts |
| 64 | 2 | Quincy Enunwa | 73.24 | 69.25 | 71.74 | 565 | Jets |
| 65 | 3 | Jamison Crowder | 72.98 | 66.38 | 73.22 | 549 | Commanders |
| 66 | 4 | Jeremy Maclin | 72.97 | 61.60 | 76.38 | 440 | Chiefs |
| 67 | 5 | Breshad Perriman | 72.48 | 63.05 | 74.60 | 348 | Ravens |
| 68 | 6 | Jordan Matthews | 72.45 | 67.19 | 71.79 | 522 | Eagles |
| 69 | 7 | Michael Floyd | 72.35 | 65.25 | 72.91 | 530 | Patriots |
| 70 | 8 | Sammie Coates | 72.07 | 58.85 | 76.72 | 241 | Steelers |
| 71 | 9 | Eli Rogers | 71.78 | 64.87 | 72.22 | 499 | Steelers |
| 72 | 10 | Robert Woods | 71.56 | 68.66 | 69.32 | 381 | Bills |
| 73 | 11 | Devin Funchess | 71.47 | 62.92 | 73.00 | 291 | Panthers |
| 74 | 12 | Jaron Brown | 71.46 | 62.59 | 73.20 | 119 | Cardinals |
| 75 | 13 | Adam Humphries | 71.46 | 65.11 | 71.53 | 478 | Buccaneers |
| 76 | 14 | Andre Holmes | 71.42 | 67.19 | 70.08 | 160 | Raiders |
| 77 | 15 | Deonte Thompson | 71.21 | 65.69 | 70.72 | 194 | Bears |
| 78 | 16 | Victor Cruz | 71.13 | 60.93 | 73.76 | 523 | Giants |
| 79 | 17 | Cody Core | 71.04 | 60.32 | 74.02 | 138 | Bengals |
| 80 | 18 | Jeremy Kerley | 70.95 | 67.90 | 68.81 | 520 | 49ers |
| 81 | 19 | Brandon Coleman | 70.79 | 65.16 | 70.38 | 199 | Saints |
| 82 | 20 | Vincent Jackson | 70.76 | 60.60 | 73.37 | 221 | Buccaneers |
| 83 | 21 | Rashard Higgins | 70.60 | 60.62 | 73.08 | 112 | Browns |
| 84 | 22 | Marquise Goodwin | 70.60 | 61.82 | 72.29 | 414 | Bills |
| 85 | 23 | Jordan Taylor | 70.59 | 61.09 | 72.75 | 178 | Broncos |
| 86 | 24 | Harry Douglas | 70.55 | 63.17 | 71.31 | 167 | Titans |
| 87 | 25 | Charles Johnson | 70.54 | 60.26 | 73.22 | 271 | Vikings |
| 88 | 26 | Justin Hunter | 70.31 | 61.28 | 72.16 | 205 | Bills |
| 89 | 27 | Tyler Boyd | 70.27 | 64.48 | 69.97 | 510 | Bengals |
| 90 | 28 | Eddie Royal | 70.21 | 63.88 | 70.27 | 239 | Bears |
| 91 | 29 | Josh Bellamy | 70.07 | 65.70 | 68.81 | 184 | Bears |
| 92 | 30 | Danny Amendola | 70.03 | 66.91 | 67.94 | 285 | Patriots |
| 93 | 31 | Jalin Marshall | 69.90 | 62.38 | 70.75 | 115 | Jets |
| 94 | 32 | Allen Hurns | 69.85 | 57.40 | 73.98 | 454 | Jaguars |
| 95 | 33 | Justin Hardy | 69.84 | 69.12 | 66.16 | 217 | Falcons |
| 96 | 34 | Cordarrelle Patterson | 69.83 | 63.44 | 69.92 | 385 | Vikings |
| 97 | 35 | Paul Turner | 69.77 | 60.49 | 71.79 | 112 | Eagles |
| 98 | 36 | Anquan Boldin | 69.68 | 64.82 | 68.76 | 625 | Lions |
| 99 | 37 | Malcolm Mitchell | 69.61 | 64.56 | 68.81 | 408 | Patriots |
| 100 | 38 | Cobi Hamilton | 69.39 | 62.02 | 70.14 | 300 | Steelers |
| 101 | 39 | Brice Butler | 69.36 | 59.75 | 71.60 | 248 | Cowboys |
| 102 | 40 | Brian Quick | 69.33 | 60.42 | 71.10 | 470 | Rams |
| 103 | 41 | Will Fuller V | 69.00 | 61.56 | 69.80 | 627 | Texans |
| 104 | 42 | Kevin White | 68.98 | 62.74 | 68.98 | 138 | Bears |
| 105 | 43 | Sterling Shepard | 68.94 | 66.20 | 66.60 | 692 | Giants |
| 106 | 44 | Chester Rogers | 68.88 | 59.45 | 71.00 | 315 | Colts |
| 107 | 45 | Kamar Aiken | 68.87 | 60.44 | 70.32 | 425 | Ravens |
| 108 | 46 | Bryan Walters | 68.84 | 61.71 | 69.43 | 230 | Jaguars |
| 109 | 47 | Corey Brown | 68.82 | 58.12 | 71.79 | 372 | Panthers |
| 110 | 48 | Tajae Sharpe | 68.80 | 60.71 | 70.03 | 522 | Titans |
| 111 | 49 | Darrius Heyward-Bey | 68.79 | 56.64 | 72.72 | 182 | Steelers |
| 112 | 50 | Phillip Dorsett | 68.54 | 56.76 | 72.23 | 567 | Colts |
| 113 | 51 | Quinton Patton | 68.34 | 61.75 | 68.57 | 417 | 49ers |
| 114 | 52 | Chris Conley | 68.06 | 61.39 | 68.34 | 568 | Chiefs |
| 115 | 53 | Corey Coleman | 68.04 | 61.29 | 68.38 | 379 | Browns |
| 116 | 54 | Rod Streater | 68.04 | 63.71 | 66.76 | 166 | 49ers |
| 117 | 55 | Dorial Green-Beckham | 68.01 | 59.92 | 69.23 | 446 | Eagles |
| 118 | 56 | Andre Roberts | 67.72 | 60.16 | 68.59 | 185 | Lions |
| 119 | 57 | Jermaine Kearse | 67.67 | 55.59 | 71.55 | 635 | Seahawks |
| 120 | 58 | Cecil Shorts | 67.57 | 60.50 | 68.11 | 181 | Buccaneers |
| 121 | 59 | Andrew Hawkins | 67.48 | 58.28 | 69.45 | 472 | Browns |
| 122 | 60 | James Wright | 67.37 | 62.22 | 66.63 | 120 | Bengals |
| 123 | 61 | Bennie Fowler | 67.33 | 56.66 | 70.28 | 159 | Broncos |
| 124 | 62 | Torrey Smith | 67.31 | 54.78 | 71.49 | 380 | 49ers |
| 125 | 63 | Brittan Golden | 67.27 | 59.19 | 68.49 | 109 | Cardinals |
| 126 | 64 | Aaron Burbridge | 66.66 | 57.93 | 68.31 | 139 | 49ers |
| 127 | 65 | Jeff Janis | 66.21 | 55.89 | 68.92 | 179 | Packers |
| 128 | 66 | Christopher Harper | 66.12 | 60.18 | 65.92 | 133 | 49ers |
| 129 | 67 | Ricardo Louis | 65.71 | 58.78 | 66.16 | 222 | Browns |
| 130 | 68 | Tavon Austin | 65.56 | 61.12 | 64.36 | 492 | Rams |
| 131 | 69 | Cody Latimer | 65.46 | 60.22 | 64.78 | 132 | Broncos |
| 132 | 70 | Albert Wilson | 65.07 | 55.03 | 67.59 | 353 | Chiefs |
| 133 | 71 | Charone Peake | 65.03 | 59.40 | 64.61 | 239 | Jets |
| 134 | 72 | Jaelen Strong | 64.76 | 57.29 | 65.57 | 202 | Texans |
| 135 | 73 | Jordan Norwood | 64.26 | 53.88 | 67.02 | 342 | Broncos |
| 136 | 74 | Ryan Grant | 64.09 | 57.49 | 64.32 | 128 | Commanders |
| 137 | 75 | Seth Roberts | 63.89 | 56.31 | 64.78 | 590 | Raiders |
| 138 | 76 | Roger Lewis Jr. | 63.73 | 56.30 | 64.51 | 131 | Giants |
| 139 | 77 | Walter Powell | 62.29 | 58.66 | 60.55 | 191 | Bills |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 140 | 1 | Pharoh Cooper | 61.56 | 56.71 | 60.62 | 138 | Rams |
| 141 | 2 | Nelson Agholor | 61.16 | 53.15 | 62.33 | 565 | Eagles |
| 142 | 3 | Braxton Miller | 60.04 | 54.44 | 59.60 | 241 | Texans |
| 143 | 4 | Chris Moore | 58.66 | 56.12 | 56.19 | 101 | Ravens |
