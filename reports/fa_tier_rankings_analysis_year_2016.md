# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:26:58Z
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
| 5 | 5 | Brandon Linder | 88.61 | 81.80 | 88.99 | 909 | Jaguars |
| 6 | 6 | J.C. Tretter | 86.42 | 78.20 | 87.73 | 488 | Packers |
| 7 | 7 | Maurkice Pouncey | 85.42 | 78.10 | 86.13 | 1151 | Steelers |
| 8 | 8 | Weston Richburg | 85.31 | 77.60 | 86.28 | 1110 | Giants |
| 9 | 9 | Corey Linsley | 84.04 | 76.40 | 84.96 | 802 | Packers |
| 10 | 10 | Greg Mancz | 83.55 | 75.70 | 84.61 | 1261 | Texans |
| 11 | 11 | Max Unger | 83.26 | 75.10 | 84.53 | 1091 | Saints |
| 12 | 12 | Mitch Morse | 82.93 | 74.60 | 84.31 | 1075 | Chiefs |
| 13 | 13 | Justin Britt | 82.80 | 81.60 | 79.43 | 1121 | Seahawks |
| 14 | 14 | Travis Swanson | 82.40 | 74.60 | 83.44 | 766 | Lions |
| 15 | 15 | Ben Jones | 81.87 | 72.90 | 83.69 | 1061 | Titans |
| 16 | 16 | Jason Kelce | 81.74 | 71.40 | 84.46 | 1132 | Eagles |
| 17 | 17 | A.Q. Shipley | 81.24 | 71.90 | 83.30 | 1148 | Cardinals |
| 18 | 18 | Ryan Kalil | 81.19 | 71.90 | 83.21 | 504 | Panthers |
| 19 | 19 | Jeremy Zuttah | 81.03 | 73.20 | 82.08 | 1109 | Ravens |
| 20 | 20 | Kory Lichtensteiger | 80.95 | 66.40 | 86.48 | 159 | Commanders |
| 21 | 21 | Joe Hawley | 80.42 | 71.30 | 82.34 | 1956 | Buccaneers |
| 22 | 22 | David Andrews | 80.26 | 71.00 | 82.26 | 1355 | Patriots |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Spencer Long | 79.72 | 71.00 | 81.37 | 804 | Commanders |
| 24 | 2 | Evan Smith | 79.52 | 70.40 | 81.44 | 177 | Buccaneers |
| 25 | 3 | Russell Bodine | 78.79 | 69.50 | 80.82 | 1060 | Bengals |
| 26 | 4 | Ryan Kelly | 78.16 | 72.40 | 77.84 | 1018 | Colts |
| 27 | 5 | Daniel Kilgore | 76.86 | 66.40 | 79.66 | 794 | 49ers |
| 28 | 6 | Nick Mangold | 76.83 | 66.70 | 79.42 | 433 | Jets |
| 29 | 7 | Mike Pouncey | 75.53 | 65.70 | 77.92 | 301 | Dolphins |
| 30 | 8 | Tim Barnes | 75.24 | 64.80 | 78.04 | 1004 | Rams |
| 31 | 9 | Wesley Johnson | 74.52 | 63.30 | 77.83 | 657 | Jets |

### Starter (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 32 | 1 | Gino Gradkowski | 72.68 | 60.70 | 76.50 | 235 | Panthers |
| 33 | 2 | Eric Wood | 72.26 | 61.50 | 75.27 | 570 | Bills |
| 34 | 3 | Nick Easton | 64.80 | 56.40 | 66.24 | 414 | Vikings |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Dominique Rodgers-Cromartie | 92.35 | 90.60 | 89.66 | 736 | Giants |
| 2 | 2 | Chris Harris Jr. | 92.18 | 90.60 | 89.06 | 1092 | Broncos |
| 3 | 3 | A.J. Bouye | 91.71 | 89.90 | 90.10 | 859 | Texans |
| 4 | 4 | Brent Grimes | 91.52 | 87.40 | 90.42 | 998 | Buccaneers |
| 5 | 5 | Aqib Talib | 91.31 | 89.80 | 89.72 | 870 | Broncos |
| 6 | 6 | Malcolm Butler | 89.62 | 86.90 | 88.73 | 1192 | Patriots |
| 7 | 7 | Casey Hayward Jr. | 88.15 | 83.60 | 87.02 | 988 | Chargers |
| 8 | 8 | Janoris Jenkins | 85.92 | 83.20 | 84.30 | 1028 | Giants |
| 9 | 9 | Marcus Peters | 85.58 | 79.20 | 85.67 | 1073 | Chiefs |
| 10 | 10 | Terence Newman | 85.42 | 83.00 | 83.80 | 754 | Vikings |
| 11 | 11 | Richard Sherman | 84.56 | 76.90 | 85.50 | 1175 | Seahawks |
| 12 | 12 | Darius Slay | 84.16 | 80.20 | 83.66 | 801 | Lions |
| 13 | 13 | Logan Ryan | 83.42 | 76.20 | 84.07 | 1081 | Patriots |
| 14 | 14 | Patrick Peterson | 83.28 | 80.10 | 81.24 | 1035 | Cardinals |
| 15 | 15 | Josh Norman | 82.84 | 76.60 | 83.03 | 1059 | Commanders |
| 16 | 16 | William Gay | 81.68 | 79.20 | 79.16 | 971 | Steelers |
| 17 | 17 | Byron Maxwell | 80.41 | 74.50 | 82.58 | 846 | Dolphins |
| 18 | 18 | Kevin Johnson | 80.35 | 80.90 | 82.34 | 287 | Texans |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Jalen Ramsey | 79.82 | 73.10 | 80.13 | 1061 | Jaguars |
| 20 | 2 | Sean Smith | 79.62 | 74.30 | 79.32 | 958 | Raiders |
| 21 | 3 | Xavier Rhodes | 79.51 | 73.00 | 80.72 | 787 | Vikings |
| 22 | 4 | Tavon Young | 79.13 | 75.60 | 77.32 | 833 | Ravens |
| 23 | 5 | Nickell Robey-Coleman | 78.47 | 74.00 | 77.29 | 573 | Bills |
| 24 | 6 | Terrance Mitchell | 78.41 | 74.70 | 85.37 | 295 | Chiefs |
| 25 | 7 | Trumaine Johnson | 77.70 | 73.90 | 79.20 | 954 | Rams |
| 26 | 8 | Morris Claiborne | 76.81 | 77.70 | 80.28 | 432 | Cowboys |
| 27 | 9 | Kareem Jackson | 76.66 | 71.70 | 77.36 | 822 | Texans |
| 28 | 10 | Adam Jones | 76.64 | 72.00 | 75.88 | 1058 | Bengals |
| 29 | 11 | Ross Cockrell | 76.25 | 70.10 | 79.10 | 1217 | Steelers |
| 30 | 12 | Deji Olatoye | 76.16 | 75.70 | 83.23 | 125 | Falcons |
| 31 | 13 | Artie Burns | 75.80 | 69.20 | 76.03 | 991 | Steelers |
| 32 | 14 | Jalen Collins | 75.68 | 73.70 | 79.08 | 636 | Falcons |
| 33 | 15 | Dre Kirkpatrick | 75.63 | 71.00 | 75.48 | 978 | Bengals |
| 34 | 16 | Harlan Miller | 75.47 | 78.30 | 90.91 | 140 | Cardinals |
| 35 | 17 | James Bradberry | 75.37 | 70.20 | 77.78 | 799 | Panthers |
| 36 | 18 | Robert Alford | 75.25 | 67.20 | 78.02 | 1297 | Falcons |
| 37 | 19 | Orlando Scandrick | 74.75 | 69.90 | 75.39 | 704 | Cowboys |
| 38 | 20 | Jamar Taylor | 74.42 | 77.70 | 70.66 | 921 | Browns |
| 39 | 21 | Tramaine Brock Sr. | 74.41 | 67.70 | 77.73 | 1099 | 49ers |
| 40 | 22 | Desmond Trufant | 74.03 | 65.00 | 79.53 | 591 | Falcons |

### Starter (70 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 41 | 1 | Cre'Von LeBlanc | 73.95 | 65.90 | 78.28 | 696 | Bears |
| 42 | 2 | Johnathan Joseph | 73.68 | 64.20 | 76.35 | 737 | Texans |
| 43 | 3 | Captain Munnerlyn | 73.48 | 66.30 | 74.62 | 636 | Vikings |
| 44 | 4 | David Amerson | 72.96 | 65.10 | 74.55 | 1031 | Raiders |
| 45 | 5 | Anthony Brown | 72.47 | 64.70 | 73.48 | 742 | Cowboys |
| 46 | 6 | Brice McCain | 72.46 | 65.80 | 73.77 | 844 | Titans |
| 47 | 7 | Daryl Worley | 72.32 | 63.50 | 75.06 | 865 | Panthers |
| 48 | 8 | Leon Hall | 72.28 | 65.40 | 74.58 | 455 | Giants |
| 49 | 9 | T.J. Carrie | 72.26 | 65.90 | 74.00 | 378 | Raiders |
| 50 | 10 | Alterraun Verner | 72.24 | 66.00 | 77.14 | 241 | Buccaneers |
| 51 | 11 | Aaron Colvin | 72.03 | 71.20 | 73.62 | 292 | Jaguars |
| 52 | 12 | Brandon Carr | 71.97 | 63.70 | 73.31 | 1077 | Cowboys |
| 53 | 13 | Jimmy Smith | 71.81 | 65.60 | 76.05 | 583 | Ravens |
| 54 | 14 | Ronald Darby | 71.65 | 61.30 | 76.09 | 822 | Bills |
| 55 | 15 | Darrelle Revis | 71.60 | 62.30 | 74.79 | 922 | Jets |
| 56 | 16 | Briean Boddy-Calhoun | 71.45 | 60.70 | 76.54 | 571 | Browns |
| 57 | 17 | Tramon Williams | 71.32 | 63.70 | 74.63 | 625 | Browns |
| 58 | 18 | Jason McCourty | 71.19 | 63.50 | 76.94 | 814 | Titans |
| 59 | 19 | Bobby McCain | 71.12 | 65.10 | 73.32 | 650 | Dolphins |
| 60 | 20 | Bradley Roby | 71.11 | 63.10 | 72.28 | 684 | Broncos |
| 61 | 21 | Sterling Moore | 71.03 | 63.70 | 73.31 | 805 | Saints |
| 62 | 22 | Kevon Seymour | 70.87 | 59.90 | 75.05 | 286 | Bills |
| 63 | 23 | Rashard Robinson | 70.79 | 60.80 | 75.37 | 543 | 49ers |
| 64 | 24 | Nevin Lawson | 70.71 | 67.80 | 72.96 | 994 | Lions |
| 65 | 25 | Prince Amukamara | 70.44 | 63.80 | 74.97 | 873 | Jaguars |
| 66 | 26 | Juston Burris | 70.42 | 64.00 | 78.86 | 187 | Jets |
| 67 | 27 | Trae Waynes | 70.27 | 59.60 | 75.81 | 581 | Vikings |
| 68 | 28 | Cyrus Jones | 70.24 | 66.40 | 75.94 | 147 | Patriots |
| 69 | 29 | Bashaud Breeland | 70.17 | 60.50 | 73.48 | 766 | Commanders |
| 70 | 30 | Eric Rowe | 70.13 | 62.80 | 75.80 | 587 | Patriots |
| 71 | 31 | Josh Johnson | 70.04 | 70.50 | 81.54 | 134 | Jaguars |
| 72 | 32 | Brandon Flowers | 69.98 | 65.60 | 75.91 | 352 | Chargers |
| 73 | 33 | Justin Bethel | 69.89 | 62.20 | 76.26 | 270 | Cardinals |
| 74 | 34 | Stephon Gilmore | 69.75 | 61.50 | 73.26 | 982 | Bills |
| 75 | 35 | Nolan Carroll | 69.72 | 61.50 | 72.60 | 912 | Eagles |
| 76 | 36 | Tony Lippett | 69.41 | 63.00 | 75.12 | 917 | Dolphins |
| 77 | 37 | Josh Shaw | 69.09 | 65.70 | 69.91 | 618 | Bengals |
| 78 | 38 | Quinton Dunbar | 68.94 | 59.10 | 76.67 | 301 | Commanders |
| 79 | 39 | Dontae Johnson | 68.93 | 60.10 | 74.61 | 101 | 49ers |
| 80 | 40 | Trovon Reed | 68.89 | 73.00 | 80.48 | 123 | Chargers |
| 81 | 41 | Marcus Williams | 68.86 | 59.80 | 75.42 | 455 | Jets |
| 82 | 42 | Jerraud Powers | 68.72 | 60.60 | 71.85 | 510 | Ravens |
| 83 | 43 | Steven Nelson | 68.56 | 61.90 | 71.96 | 1077 | Chiefs |
| 84 | 44 | Darryl Morris | 68.01 | 63.70 | 74.42 | 359 | Colts |
| 85 | 45 | Leodis McKelvin | 67.99 | 61.60 | 73.09 | 586 | Eagles |
| 86 | 46 | Joe Haden | 67.86 | 60.00 | 74.13 | 854 | Browns |
| 87 | 47 | Rashaan Melvin | 67.57 | 63.60 | 70.54 | 655 | Colts |
| 88 | 48 | Quinten Rollins | 67.32 | 56.50 | 72.45 | 722 | Packers |
| 89 | 49 | Johnson Bademosi | 67.17 | 66.10 | 72.68 | 283 | Lions |
| 90 | 50 | Bryce Callahan | 67.01 | 61.70 | 72.77 | 489 | Bears |
| 91 | 51 | Eli Apple | 66.80 | 59.80 | 68.34 | 772 | Giants |
| 92 | 52 | Xavien Howard | 66.39 | 60.80 | 74.29 | 582 | Dolphins |
| 93 | 53 | Buster Skrine | 65.57 | 56.00 | 68.81 | 816 | Jets |
| 94 | 54 | Keith Reaser | 65.44 | 59.50 | 68.62 | 351 | 49ers |
| 95 | 55 | Jason Verrett | 65.42 | 61.10 | 73.10 | 260 | Chargers |
| 96 | 56 | Corey White | 65.38 | 62.10 | 69.65 | 412 | Bills |
| 97 | 57 | Vernon Hargreaves III | 65.37 | 56.40 | 67.18 | 1038 | Buccaneers |
| 98 | 58 | Vontae Davis | 65.28 | 52.80 | 70.47 | 822 | Colts |
| 99 | 59 | Shareece Wright | 65.24 | 59.80 | 68.76 | 673 | Ravens |
| 100 | 60 | Darqueze Dennard | 65.14 | 58.30 | 69.60 | 330 | Bengals |
| 101 | 61 | Justin Coleman | 64.52 | 60.40 | 70.91 | 225 | Patriots |
| 102 | 62 | Marcus Cooper | 63.95 | 55.60 | 71.09 | 827 | Cardinals |
| 103 | 63 | Patrick Robinson | 63.81 | 54.60 | 70.88 | 402 | Colts |
| 104 | 64 | Valentino Blake | 63.65 | 53.80 | 70.22 | 367 | Titans |
| 105 | 65 | Kenneth Acker | 63.59 | 58.80 | 70.57 | 147 | Chiefs |
| 106 | 66 | Darryl Roberts | 62.93 | 55.60 | 71.99 | 286 | Jets |
| 107 | 67 | Johnthan Banks | 62.92 | 60.30 | 68.51 | 133 | Bears |
| 108 | 68 | D.J. Hayden | 62.48 | 52.50 | 68.81 | 476 | Raiders |
| 109 | 69 | Jeremy Lane | 62.18 | 54.60 | 67.24 | 871 | Seahawks |
| 110 | 70 | Ken Crawley | 62.11 | 52.00 | 67.81 | 503 | Saints |

### Rotation/backup (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 111 | 1 | Neiko Thorpe | 61.87 | 56.50 | 68.59 | 105 | Seahawks |
| 112 | 2 | Robert Nelson | 61.47 | 55.60 | 70.98 | 196 | Texans |
| 113 | 3 | LaDarius Gunter | 61.12 | 50.30 | 68.47 | 1055 | Packers |
| 114 | 4 | Greg Toler | 60.80 | 51.80 | 67.11 | 256 | Commanders |
| 115 | 5 | Trevin Wade | 60.72 | 50.50 | 66.50 | 393 | Giants |
| 116 | 6 | Coty Sensabaugh | 60.51 | 50.10 | 67.03 | 257 | Giants |
| 117 | 7 | Davon House | 60.31 | 43.20 | 69.95 | 272 | Jaguars |
| 118 | 8 | Delvin Breaux Sr. | 60.08 | 47.10 | 71.08 | 296 | Saints |
| 119 | 9 | De'Vante Harris | 59.77 | 61.70 | 70.28 | 108 | Saints |
| 120 | 10 | Tracy Porter | 58.50 | 46.00 | 66.00 | 944 | Bears |
| 121 | 11 | B.W. Webb | 58.33 | 54.50 | 62.96 | 588 | Saints |
| 122 | 12 | E.J. Gaines | 58.22 | 43.30 | 67.65 | 614 | Rams |
| 123 | 13 | LeShaun Sims | 57.75 | 55.30 | 66.08 | 236 | Titans |
| 124 | 14 | Trevor Williams | 57.63 | 49.80 | 65.99 | 389 | Chargers |
| 125 | 15 | Kendall Fuller | 56.33 | 54.90 | 57.28 | 476 | Commanders |
| 126 | 16 | Robert McClain | 56.05 | 47.50 | 63.84 | 327 | Chargers |
| 127 | 17 | Jude Adjei-Barimah | 55.80 | 57.10 | 56.50 | 290 | Buccaneers |
| 128 | 18 | Leonard Johnson | 55.72 | 47.20 | 64.10 | 436 | Panthers |
| 129 | 19 | Chris Milton | 54.78 | 57.90 | 67.04 | 106 | Colts |
| 130 | 20 | Ron Brooks | 54.55 | 50.50 | 63.18 | 235 | Eagles |
| 131 | 21 | Javien Elliott | 54.21 | 60.40 | 61.88 | 184 | Buccaneers |
| 132 | 22 | Phillip Gaines | 53.86 | 41.70 | 66.14 | 449 | Chiefs |
| 133 | 23 | Demetri Goodson | 52.92 | 48.60 | 59.86 | 182 | Packers |
| 134 | 24 | D.J. White | 51.78 | 49.00 | 60.34 | 138 | Chiefs |
| 135 | 25 | Craig Mager | 51.44 | 43.50 | 60.38 | 410 | Chargers |
| 136 | 26 | Charles James II | 50.94 | 35.20 | 65.08 | 164 | Colts |
| 137 | 27 | Brandon Williams | 50.44 | 43.20 | 64.52 | 241 | Cardinals |
| 138 | 28 | Mike Jordan | 49.42 | 55.90 | 59.44 | 177 | Rams |
| 139 | 29 | Troy Hill | 46.72 | 45.20 | 53.60 | 338 | Rams |
| 140 | 30 | Dashaun Phillips | 45.00 | 38.40 | 57.15 | 148 | Commanders |

## DI — Defensive Interior

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 94.29 | 89.96 | 93.01 | 830 | Rams |
| 2 | 2 | J.J. Watt | 92.02 | 87.09 | 97.90 | 157 | Texans |
| 3 | 3 | Leonard Williams | 89.40 | 89.83 | 84.95 | 898 | Jets |
| 4 | 4 | Geno Atkins | 88.91 | 87.08 | 85.96 | 781 | Bengals |
| 5 | 5 | Calais Campbell | 88.32 | 86.78 | 85.38 | 831 | Cardinals |
| 6 | 6 | Kawann Short | 88.15 | 86.39 | 85.15 | 779 | Panthers |
| 7 | 7 | Michael Pierce | 87.71 | 84.07 | 85.97 | 375 | Ravens |
| 8 | 8 | Ndamukong Suh | 86.02 | 86.50 | 81.53 | 1028 | Dolphins |
| 9 | 9 | Jurrell Casey | 85.62 | 86.65 | 81.29 | 728 | Titans |
| 10 | 10 | Malik Jackson | 85.56 | 84.12 | 82.36 | 718 | Jaguars |
| 11 | 11 | Fletcher Cox | 85.43 | 89.69 | 78.42 | 772 | Eagles |
| 12 | 12 | Mike Daniels | 85.22 | 83.73 | 82.04 | 797 | Packers |
| 13 | 13 | Damon Harrison Sr. | 85.08 | 84.42 | 81.35 | 720 | Giants |
| 14 | 14 | Linval Joseph | 83.02 | 85.98 | 77.82 | 718 | Vikings |
| 15 | 15 | Deon Simon | 82.62 | 77.64 | 82.81 | 204 | Jets |
| 16 | 16 | Marcell Dareus | 82.55 | 86.71 | 80.30 | 417 | Bills |
| 17 | 17 | Gerald McCoy | 80.73 | 85.79 | 74.66 | 796 | Buccaneers |
| 18 | 18 | Akiem Hicks | 80.59 | 77.29 | 78.82 | 931 | Bears |
| 19 | 19 | Eddie Goldman | 80.31 | 85.47 | 79.60 | 197 | Bears |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | DeForest Buckner | 79.51 | 82.80 | 74.19 | 1006 | 49ers |
| 21 | 2 | Chris Baker | 79.34 | 71.61 | 80.52 | 782 | Commanders |
| 22 | 3 | Derek Wolfe | 79.19 | 79.32 | 76.28 | 663 | Broncos |
| 23 | 4 | David Irving | 79.11 | 77.33 | 77.70 | 530 | Cowboys |
| 24 | 5 | Danny Shelton | 79.04 | 82.00 | 72.90 | 745 | Browns |
| 25 | 6 | Dan Williams | 78.66 | 78.73 | 74.45 | 397 | Raiders |
| 26 | 7 | Dean Lowry | 78.58 | 74.27 | 77.29 | 211 | Packers |
| 27 | 8 | Brandon Williams | 78.44 | 72.55 | 78.20 | 635 | Ravens |
| 28 | 9 | Brent Urban | 78.27 | 73.12 | 81.43 | 150 | Ravens |
| 29 | 10 | Dominique Easley | 78.20 | 78.77 | 76.25 | 470 | Rams |
| 30 | 11 | Michael Brockers | 77.77 | 81.98 | 71.83 | 419 | Rams |
| 31 | 12 | Abry Jones | 77.70 | 73.11 | 77.42 | 462 | Jaguars |
| 32 | 13 | Malcom Brown | 77.32 | 70.44 | 77.74 | 709 | Patriots |
| 33 | 14 | Timmy Jernigan | 77.15 | 67.75 | 80.18 | 631 | Ravens |
| 34 | 15 | Lawrence Guy Sr. | 77.12 | 68.75 | 78.53 | 487 | Ravens |
| 35 | 16 | Karl Klug | 76.53 | 72.92 | 75.81 | 398 | Titans |
| 36 | 17 | Grady Jarrett | 76.42 | 64.18 | 80.79 | 763 | Falcons |
| 37 | 18 | Ra'Shede Hageman | 76.33 | 65.60 | 79.83 | 353 | Falcons |
| 38 | 19 | Cameron Heyward | 75.87 | 74.58 | 77.25 | 363 | Steelers |
| 39 | 20 | Kenny Clark | 75.49 | 75.18 | 71.53 | 410 | Packers |
| 40 | 21 | Nick Fairley | 75.29 | 64.04 | 80.60 | 722 | Saints |
| 41 | 22 | Kyle Williams | 74.52 | 63.98 | 81.23 | 794 | Bills |
| 42 | 23 | Bennie Logan | 74.17 | 60.63 | 81.22 | 467 | Eagles |
| 43 | 24 | Dan McCullers | 74.13 | 70.13 | 74.81 | 206 | Steelers |

### Starter (74 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 44 | 1 | Johnathan Hankins | 73.95 | 66.58 | 76.88 | 816 | Giants |
| 45 | 2 | Sheldon Day | 73.82 | 56.25 | 81.36 | 203 | Jaguars |
| 46 | 3 | Henry Anderson | 73.34 | 69.65 | 77.61 | 308 | Colts |
| 47 | 4 | Stephon Tuitt | 72.89 | 65.98 | 73.33 | 945 | Steelers |
| 48 | 5 | Stephen Paea | 72.71 | 69.64 | 73.73 | 320 | Browns |
| 49 | 6 | DJ Reader | 72.49 | 67.09 | 71.93 | 477 | Texans |
| 50 | 7 | Jaye Howard Jr. | 72.31 | 58.62 | 81.43 | 360 | Chiefs |
| 51 | 8 | Alan Branch | 72.24 | 64.96 | 73.96 | 760 | Patriots |
| 52 | 9 | Brandon Mebane | 72.16 | 69.15 | 74.58 | 340 | Chargers |
| 53 | 10 | Stacy McGee | 71.82 | 75.91 | 69.09 | 242 | Raiders |
| 54 | 11 | Corey Liuget | 71.31 | 62.17 | 74.80 | 812 | Chargers |
| 55 | 12 | C.J. Wilson | 70.86 | 66.31 | 76.18 | 128 | Bears |
| 56 | 13 | Leger Douzable | 70.78 | 53.53 | 78.12 | 481 | Bills |
| 57 | 14 | Arik Armstead | 70.56 | 65.07 | 75.26 | 332 | 49ers |
| 58 | 15 | Adolphus Washington | 70.45 | 57.15 | 76.18 | 331 | Bills |
| 59 | 16 | Star Lotulelei | 70.07 | 57.30 | 74.61 | 701 | Panthers |
| 60 | 17 | Javon Hargrave | 70.03 | 57.67 | 74.11 | 645 | Steelers |
| 61 | 18 | Steve McLendon | 69.98 | 57.54 | 77.34 | 382 | Jets |
| 62 | 19 | A'Shawn Robinson | 69.88 | 58.64 | 73.21 | 446 | Lions |
| 63 | 20 | Ricky Jean Francois | 68.85 | 56.47 | 72.94 | 442 | Commanders |
| 64 | 21 | Christian Covington | 68.70 | 60.39 | 70.08 | 475 | Texans |
| 65 | 22 | Dontari Poe | 68.48 | 64.10 | 67.24 | 876 | Chiefs |
| 66 | 23 | Vincent Valentine | 68.32 | 52.72 | 74.55 | 319 | Patriots |
| 67 | 24 | Zach Kerr | 67.85 | 53.36 | 77.51 | 317 | Colts |
| 68 | 25 | DaQuan Jones | 67.73 | 65.95 | 66.64 | 673 | Titans |
| 69 | 26 | Haloti Ngata | 67.59 | 56.63 | 72.81 | 577 | Lions |
| 70 | 27 | Beau Allen | 67.56 | 55.64 | 71.34 | 412 | Eagles |
| 71 | 28 | Rodney Gunter | 67.44 | 56.99 | 70.24 | 244 | Cardinals |
| 72 | 29 | Tyson Jackson | 67.41 | 57.84 | 69.63 | 389 | Falcons |
| 73 | 30 | Quinton Dial | 67.05 | 56.91 | 71.41 | 478 | 49ers |
| 74 | 31 | David Parry | 66.83 | 55.63 | 70.13 | 644 | Colts |
| 75 | 32 | Pat Sims | 66.82 | 51.23 | 75.23 | 409 | Bengals |
| 76 | 33 | Cedric Thornton | 66.70 | 50.65 | 75.21 | 291 | Cowboys |
| 77 | 34 | Tony McDaniel | 66.47 | 49.52 | 74.24 | 501 | Seahawks |
| 78 | 35 | Sylvester Williams | 66.37 | 57.49 | 68.13 | 644 | Broncos |
| 79 | 36 | Jamie Meder | 66.28 | 61.05 | 68.73 | 722 | Browns |
| 80 | 37 | Corey Peters | 66.20 | 56.35 | 69.64 | 498 | Cardinals |
| 81 | 38 | Rakeem Nunez-Roches | 65.90 | 49.46 | 79.60 | 313 | Chiefs |
| 82 | 39 | Justin Ellis | 65.88 | 63.49 | 64.56 | 352 | Raiders |
| 83 | 40 | Tom Johnson | 65.85 | 48.93 | 74.00 | 476 | Vikings |
| 84 | 41 | Chris Jones | 65.77 | 53.72 | 71.20 | 916 | Chiefs |
| 85 | 42 | T.Y. McGill | 65.71 | 61.58 | 67.81 | 302 | Colts |
| 86 | 43 | Sealver Siliga | 65.66 | 52.59 | 74.89 | 160 | Buccaneers |
| 87 | 44 | Ahtyba Rubin | 65.52 | 51.42 | 71.39 | 675 | Seahawks |
| 88 | 45 | Paul Soliai | 65.49 | 48.73 | 76.46 | 152 | Panthers |
| 89 | 46 | Sen'Derrick Marks | 65.49 | 53.31 | 73.19 | 543 | Jaguars |
| 90 | 47 | Tyson Alualu | 65.41 | 57.63 | 67.46 | 507 | Jaguars |
| 91 | 48 | Clinton McDonald | 65.34 | 51.78 | 76.05 | 485 | Buccaneers |
| 92 | 49 | Letroy Guion | 65.22 | 52.25 | 70.02 | 524 | Packers |
| 93 | 50 | Jared Odrick | 65.21 | 59.55 | 70.02 | 261 | Jaguars |
| 94 | 51 | Tyrunn Walker | 65.10 | 52.27 | 73.24 | 377 | Lions |
| 95 | 52 | Jarran Reed | 65.05 | 51.39 | 69.99 | 545 | Seahawks |
| 96 | 53 | Al Woods | 64.67 | 49.53 | 73.29 | 245 | Titans |
| 97 | 54 | Denico Autry | 64.50 | 47.25 | 72.62 | 742 | Raiders |
| 98 | 55 | Hassan Ridgeway | 64.05 | 52.70 | 67.45 | 442 | Colts |
| 99 | 56 | Jared Crick | 64.04 | 50.41 | 68.96 | 938 | Broncos |
| 100 | 57 | Vince Wilfork | 63.95 | 47.17 | 70.97 | 588 | Texans |
| 101 | 58 | Frostee Rucker | 63.79 | 41.49 | 76.38 | 304 | Cardinals |
| 102 | 59 | Cam Thomas | 63.76 | 50.40 | 68.50 | 391 | Rams |
| 103 | 60 | Cullen Jenkins | 63.62 | 44.21 | 73.74 | 308 | Commanders |
| 104 | 61 | Stefan Charles | 63.56 | 48.49 | 72.45 | 235 | Lions |
| 105 | 62 | Jonathan Bullard | 63.54 | 48.75 | 71.31 | 296 | Bears |
| 106 | 63 | Earl Mitchell | 63.46 | 49.52 | 72.96 | 334 | Dolphins |
| 107 | 64 | Billy Winn | 63.42 | 49.42 | 70.47 | 341 | Broncos |
| 108 | 65 | Austin Johnson | 63.39 | 56.28 | 70.22 | 190 | Titans |
| 109 | 66 | Roy Miller | 63.26 | 57.11 | 68.83 | 156 | Jaguars |
| 110 | 67 | Allen Bailey | 63.14 | 55.51 | 70.83 | 181 | Chiefs |
| 111 | 68 | Jonathan Babineaux | 63.01 | 44.65 | 71.28 | 533 | Falcons |
| 112 | 69 | Domata Peko Sr. | 62.78 | 45.50 | 70.13 | 593 | Bengals |
| 113 | 70 | Arthur Jones | 62.70 | 47.10 | 74.76 | 322 | Colts |
| 114 | 71 | Vernon Butler | 62.70 | 54.34 | 70.36 | 226 | Panthers |
| 115 | 72 | John Jenkins | 62.68 | 51.25 | 70.71 | 217 | Seahawks |
| 116 | 73 | Jarvis Jenkins | 62.54 | 51.46 | 66.08 | 266 | Chiefs |
| 117 | 74 | Maliek Collins | 62.10 | 46.45 | 68.37 | 697 | Cowboys |

### Rotation/backup (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 118 | 1 | Will Sutton III | 61.97 | 60.58 | 64.04 | 174 | Bears |
| 119 | 2 | Terrell McClain | 61.84 | 52.91 | 68.41 | 500 | Cowboys |
| 120 | 3 | Jordan Phillips | 61.72 | 51.59 | 64.69 | 653 | Dolphins |
| 121 | 4 | Destiny Vaeao | 61.59 | 45.37 | 68.24 | 268 | Eagles |
| 122 | 5 | Tenny Palepoi | 61.55 | 49.77 | 67.18 | 377 | Chargers |
| 123 | 6 | Akeem Spence | 61.19 | 47.31 | 67.94 | 724 | Buccaneers |
| 124 | 7 | Mike Purcell | 61.04 | 52.44 | 69.05 | 280 | 49ers |
| 125 | 8 | Antonio Smith | 60.84 | 38.49 | 72.09 | 277 | Texans |
| 126 | 9 | Kendall Reyes | 60.79 | 48.80 | 66.19 | 249 | Chiefs |
| 127 | 10 | Angelo Blackson | 60.62 | 48.22 | 66.67 | 248 | Titans |
| 128 | 11 | Jay Bromley | 60.61 | 50.96 | 64.54 | 264 | Giants |
| 129 | 12 | Mitch Unrein | 60.41 | 51.42 | 66.21 | 436 | Bears |
| 130 | 13 | Derrick Shelby | 60.31 | 40.69 | 76.12 | 244 | Falcons |
| 131 | 14 | Xavier Cooper | 60.15 | 51.08 | 64.76 | 448 | Browns |
| 132 | 15 | Glenn Dorsey | 60.11 | 43.48 | 70.99 | 402 | 49ers |
| 133 | 16 | Sheldon Rankins | 60.08 | 51.30 | 69.06 | 335 | Saints |
| 134 | 17 | Kendall Langford | 59.88 | 46.48 | 69.33 | 300 | Colts |
| 135 | 18 | Darius Philon | 59.83 | 50.26 | 66.47 | 267 | Chargers |
| 136 | 19 | Caraun Reid | 59.76 | 50.49 | 67.93 | 112 | Chargers |
| 137 | 20 | Jerel Worthy | 59.70 | 50.23 | 68.20 | 150 | Bills |
| 138 | 21 | Joel Heath | 59.31 | 47.17 | 65.32 | 277 | Texans |
| 139 | 22 | Ricardo Mathews | 58.30 | 45.83 | 63.28 | 330 | Steelers |
| 140 | 23 | Matt Ioannidis | 58.06 | 54.49 | 63.57 | 103 | Commanders |
| 141 | 24 | Adam Gotsis | 57.62 | 48.93 | 59.25 | 221 | Broncos |
| 142 | 25 | Corbin Bryant | 57.54 | 47.69 | 64.10 | 233 | Bills |
| 143 | 26 | Kyle Love | 56.95 | 42.00 | 69.00 | 224 | Panthers |
| 144 | 27 | Darius Latham | 56.82 | 46.78 | 60.38 | 319 | Raiders |
| 145 | 28 | Cornelius Washington | 56.57 | 48.03 | 64.61 | 364 | Bears |
| 146 | 29 | Ed Stinson | 56.57 | 47.18 | 66.16 | 116 | Cardinals |
| 147 | 30 | Damion Square | 56.03 | 51.50 | 61.65 | 362 | Chargers |
| 148 | 31 | Shamar Stephen | 56.02 | 46.89 | 61.38 | 551 | Vikings |
| 149 | 32 | David Onyemata | 55.90 | 48.63 | 56.58 | 393 | Saints |
| 150 | 33 | Jihad Ward | 55.75 | 43.64 | 59.66 | 637 | Raiders |
| 151 | 34 | Khyri Thornton | 55.34 | 43.62 | 64.18 | 328 | Lions |
| 152 | 35 | Tyeler Davison | 55.21 | 48.44 | 56.20 | 438 | Saints |
| 153 | 36 | Xavier Williams | 52.21 | 52.40 | 57.82 | 118 | Cardinals |
| 154 | 37 | Anthony Johnson | 50.54 | 48.51 | 54.90 | 127 | Jets |

## ED — Edge

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 94.99 | 96.41 | 89.87 | 930 | Broncos |
| 2 | 2 | Joey Bosa | 92.37 | 97.31 | 89.08 | 563 | Chargers |
| 3 | 3 | Khalil Mack | 91.33 | 97.42 | 83.10 | 1023 | Raiders |
| 4 | 4 | Brandon Graham | 87.66 | 91.05 | 81.24 | 764 | Eagles |
| 5 | 5 | Whitney Mercilus | 87.54 | 88.05 | 83.24 | 1008 | Texans |
| 6 | 6 | Danielle Hunter | 87.18 | 86.45 | 83.88 | 602 | Vikings |
| 7 | 7 | Justin Houston | 86.33 | 84.80 | 89.33 | 349 | Chiefs |
| 8 | 8 | Pernell McPhee | 84.47 | 82.79 | 85.69 | 273 | Bears |
| 9 | 9 | Ezekiel Ansah | 83.84 | 80.58 | 82.88 | 540 | Lions |
| 10 | 10 | Cameron Wake | 82.43 | 73.93 | 86.75 | 626 | Dolphins |
| 11 | 11 | Michael Bennett | 82.36 | 88.61 | 75.59 | 675 | Seahawks |
| 12 | 12 | Frank Clark | 82.34 | 81.45 | 78.76 | 751 | Seahawks |
| 13 | 13 | James Harrison | 81.79 | 69.98 | 86.33 | 758 | Steelers |
| 14 | 14 | Olivier Vernon | 81.68 | 87.58 | 73.58 | 1112 | Giants |
| 15 | 15 | Carlos Dunlap | 81.60 | 81.63 | 77.41 | 840 | Bengals |
| 16 | 16 | Markus Golden | 81.55 | 74.31 | 82.21 | 762 | Cardinals |
| 17 | 17 | Cameron Jordan | 80.45 | 88.99 | 70.59 | 963 | Saints |
| 18 | 18 | Chandler Jones | 80.01 | 82.53 | 74.79 | 939 | Cardinals |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Cliff Avril | 78.98 | 68.70 | 81.67 | 926 | Seahawks |
| 20 | 2 | Nick Perry | 78.80 | 75.78 | 76.65 | 704 | Packers |
| 21 | 3 | Melvin Ingram III | 78.66 | 76.63 | 77.31 | 960 | Chargers |
| 22 | 4 | Jabaal Sheard | 78.37 | 84.94 | 70.14 | 670 | Patriots |
| 23 | 5 | William Hayes | 78.29 | 79.73 | 74.19 | 514 | Rams |
| 24 | 6 | DeMarcus Ware | 78.23 | 66.23 | 85.81 | 315 | Broncos |
| 25 | 7 | Charles Johnson | 77.98 | 70.74 | 81.45 | 542 | Panthers |
| 26 | 8 | Ryan Kerrigan | 77.76 | 67.65 | 80.34 | 787 | Commanders |
| 27 | 9 | Shaquil Barrett | 77.48 | 73.28 | 76.11 | 415 | Broncos |
| 28 | 10 | Robert Quinn | 77.20 | 78.84 | 78.09 | 370 | Rams |
| 29 | 11 | Everson Griffen | 75.27 | 69.65 | 74.85 | 888 | Vikings |
| 30 | 12 | Shane Ray | 75.23 | 63.65 | 78.79 | 664 | Broncos |
| 31 | 13 | Vic Beasley Jr. | 75.06 | 69.64 | 74.50 | 822 | Falcons |
| 32 | 14 | Leonard Floyd | 74.92 | 61.60 | 83.80 | 537 | Bears |
| 33 | 15 | Elvis Dumervil | 74.90 | 62.52 | 83.16 | 272 | Ravens |
| 34 | 16 | Robert Ayers | 74.83 | 76.98 | 73.39 | 577 | Buccaneers |
| 35 | 17 | Derrick Morgan | 74.32 | 65.59 | 78.37 | 768 | Titans |

### Starter (55 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 36 | 1 | Jadeveon Clowney | 73.96 | 88.83 | 63.32 | 871 | Texans |
| 37 | 2 | Jason Pierre-Paul | 73.56 | 78.36 | 70.78 | 793 | Giants |
| 38 | 3 | Trent Murphy | 73.16 | 68.99 | 71.97 | 676 | Commanders |
| 39 | 4 | Lorenzo Alexander | 73.10 | 55.00 | 86.84 | 788 | Bills |
| 40 | 5 | Vinny Curry | 73.01 | 64.41 | 74.57 | 436 | Eagles |
| 41 | 6 | Mario Addison | 72.98 | 59.96 | 78.53 | 433 | Panthers |
| 42 | 7 | Jerry Hughes | 72.81 | 64.33 | 74.29 | 857 | Bills |
| 43 | 8 | Bruce Irvin | 72.35 | 59.22 | 76.93 | 997 | Raiders |
| 44 | 9 | Cam Johnson | 71.77 | 60.02 | 85.33 | 350 | Browns |
| 45 | 10 | Willie Young | 71.64 | 59.07 | 76.37 | 715 | Bears |
| 46 | 11 | Mario Williams | 71.12 | 60.03 | 75.70 | 453 | Dolphins |
| 47 | 12 | Jordan Jenkins | 71.06 | 60.15 | 76.25 | 514 | Jets |
| 48 | 13 | Aaron Lynch | 71.01 | 60.78 | 78.98 | 222 | 49ers |
| 49 | 14 | Alex Okafor | 70.91 | 65.49 | 72.24 | 231 | Cardinals |
| 50 | 15 | Julius Peppers | 70.89 | 59.14 | 74.55 | 715 | Packers |
| 51 | 16 | Dee Ford | 70.50 | 59.62 | 74.62 | 855 | Chiefs |
| 52 | 17 | Brian Orakpo | 70.42 | 58.26 | 76.24 | 868 | Titans |
| 53 | 18 | Clay Matthews | 70.28 | 51.49 | 79.15 | 630 | Packers |
| 54 | 19 | Kerry Hyder Jr. | 70.04 | 63.45 | 76.13 | 709 | Lions |
| 55 | 20 | John Simon | 69.36 | 62.31 | 73.75 | 516 | Texans |
| 56 | 21 | Matthew Judon | 69.18 | 59.46 | 73.57 | 308 | Ravens |
| 57 | 22 | Lorenzo Mauldin IV | 69.14 | 61.65 | 74.00 | 354 | Jets |
| 58 | 23 | Connor Barwin | 69.03 | 53.76 | 75.04 | 705 | Eagles |
| 59 | 24 | Jeremiah Attaochu | 68.83 | 62.99 | 74.08 | 178 | Chargers |
| 60 | 25 | Tamba Hali | 68.42 | 61.00 | 69.20 | 599 | Chiefs |
| 61 | 26 | Terrell Suggs | 67.72 | 54.97 | 77.25 | 697 | Ravens |
| 62 | 27 | Arthur Moats | 67.69 | 52.89 | 73.90 | 401 | Steelers |
| 63 | 28 | Chris Long | 67.52 | 55.27 | 74.85 | 741 | Patriots |
| 64 | 29 | Kyler Fackrell | 67.50 | 57.48 | 72.10 | 173 | Packers |
| 65 | 30 | Trey Flowers | 67.10 | 62.28 | 72.01 | 726 | Patriots |
| 66 | 31 | Preston Smith | 66.78 | 60.16 | 67.02 | 770 | Commanders |
| 67 | 32 | Dante Fowler Jr. | 66.45 | 65.65 | 62.82 | 569 | Jaguars |
| 68 | 33 | Paul Kruger | 66.20 | 54.04 | 70.66 | 572 | Saints |
| 69 | 34 | Erik Walden | 66.20 | 49.37 | 73.57 | 759 | Colts |
| 70 | 35 | Marcus Smith | 65.93 | 55.79 | 70.41 | 220 | Eagles |
| 71 | 36 | DeMarcus Lawrence | 65.72 | 69.39 | 63.69 | 369 | Cowboys |
| 72 | 37 | Dan Skuta | 65.19 | 47.80 | 74.80 | 267 | Jaguars |
| 73 | 38 | Yannick Ngakoue | 65.15 | 56.19 | 66.96 | 706 | Jaguars |
| 74 | 39 | Akeem Ayers | 65.01 | 54.74 | 68.32 | 360 | Colts |
| 75 | 40 | Noah Spence | 64.86 | 59.80 | 64.06 | 569 | Buccaneers |
| 76 | 41 | Ryan Delaire | 64.85 | 58.47 | 73.79 | 139 | Panthers |
| 77 | 42 | Dwight Freeney | 64.53 | 46.77 | 73.14 | 538 | Falcons |
| 78 | 43 | Trent Cole | 64.42 | 54.24 | 72.56 | 237 | Colts |
| 79 | 44 | Devon Kennard | 64.27 | 55.06 | 66.25 | 549 | Giants |
| 80 | 45 | Kony Ealy | 64.22 | 57.81 | 64.32 | 624 | Panthers |
| 81 | 46 | Emmanuel Ogbah | 64.18 | 58.76 | 63.63 | 849 | Browns |
| 82 | 47 | Damontre Moore | 63.48 | 61.06 | 67.80 | 104 | Seahawks |
| 83 | 48 | Ahmad Brooks | 63.38 | 48.19 | 70.59 | 918 | 49ers |
| 84 | 49 | Armonty Bryant | 63.24 | 52.77 | 72.40 | 104 | Lions |
| 85 | 50 | Joe Schobert | 63.08 | 55.52 | 67.09 | 246 | Browns |
| 86 | 51 | Michael Johnson | 63.08 | 59.34 | 61.82 | 831 | Bengals |
| 87 | 52 | Rob Ninkovich | 63.03 | 47.45 | 69.77 | 566 | Patriots |
| 88 | 53 | Devin Taylor | 62.84 | 56.44 | 63.26 | 715 | Lions |
| 89 | 54 | Tyrone Holmes | 62.76 | 59.16 | 66.20 | 124 | Browns |
| 90 | 55 | Darryl Tapp | 62.57 | 54.39 | 64.38 | 291 | Saints |

### Rotation/backup (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 91 | 1 | Brian Robison | 61.74 | 49.00 | 66.07 | 838 | Vikings |
| 92 | 2 | Shea McClellin | 61.54 | 49.99 | 65.07 | 441 | Patriots |
| 93 | 3 | Jayrone Elliott | 61.51 | 59.00 | 66.00 | 136 | Packers |
| 94 | 4 | Brooks Reed | 60.91 | 54.73 | 61.79 | 529 | Falcons |
| 95 | 5 | Ryan Davis Sr. | 60.86 | 56.13 | 65.99 | 155 | Cowboys |
| 96 | 6 | William Gholston | 60.81 | 59.41 | 58.82 | 586 | Buccaneers |
| 97 | 7 | Datone Jones | 60.69 | 53.50 | 61.32 | 635 | Packers |
| 98 | 8 | Ronald Blair III | 60.67 | 58.00 | 58.28 | 307 | 49ers |
| 99 | 9 | Ethan Westbrooks | 60.64 | 55.32 | 60.96 | 531 | Rams |
| 100 | 10 | Adrian Clayborn | 60.57 | 57.40 | 62.68 | 589 | Falcons |
| 101 | 11 | Shaq Lawson | 60.11 | 56.97 | 64.28 | 237 | Bills |
| 102 | 12 | Lerentee McCray | 59.83 | 58.22 | 61.64 | 163 | Bills |
| 103 | 13 | Bud Dupree | 59.82 | 56.33 | 61.88 | 509 | Steelers |
| 104 | 14 | Tyrone Crawford | 59.44 | 52.99 | 60.09 | 650 | Cowboys |
| 105 | 15 | Andre Branch | 59.36 | 55.92 | 59.88 | 814 | Dolphins |
| 106 | 16 | Eli Harold | 59.33 | 57.63 | 56.29 | 691 | 49ers |
| 107 | 17 | Sam Acho | 59.26 | 52.70 | 59.79 | 499 | Bears |
| 108 | 18 | Benson Mayowa | 59.25 | 59.95 | 57.12 | 406 | Cowboys |
| 109 | 19 | Muhammad Wilkerson | 59.07 | 53.16 | 59.36 | 848 | Jets |
| 110 | 20 | Kyle Emanuel | 58.77 | 54.32 | 57.95 | 545 | Chargers |
| 111 | 21 | Kasim Edebali | 58.62 | 51.74 | 59.46 | 254 | Saints |
| 112 | 22 | Jarvis Jones | 58.44 | 55.91 | 57.63 | 499 | Steelers |
| 113 | 23 | Za'Darius Smith | 58.37 | 56.70 | 57.67 | 492 | Ravens |
| 114 | 24 | Cassius Marsh | 58.15 | 59.47 | 55.38 | 438 | Seahawks |
| 115 | 25 | David Bass | 58.04 | 56.74 | 58.29 | 224 | Titans |
| 116 | 26 | Aaron Wallace | 57.97 | 59.02 | 61.44 | 117 | Titans |
| 117 | 27 | Romeo Okwara | 57.84 | 57.82 | 53.69 | 427 | Giants |
| 118 | 28 | Wallace Gilberry | 57.64 | 48.52 | 63.20 | 266 | Bengals |
| 119 | 29 | Robert Mathis | 57.27 | 41.65 | 65.21 | 536 | Colts |
| 120 | 30 | Kerry Wynn | 56.34 | 57.78 | 54.87 | 131 | Giants |
| 121 | 31 | Anthony Zettel | 56.32 | 54.70 | 55.32 | 224 | Lions |
| 122 | 32 | Wes Horton | 56.10 | 54.30 | 56.98 | 332 | Panthers |
| 123 | 33 | Will Clarke | 55.59 | 54.55 | 54.94 | 374 | Bengals |
| 124 | 34 | Jack Crawford | 55.45 | 48.54 | 57.97 | 548 | Cowboys |
| 125 | 35 | Anthony Chickillo | 55.07 | 56.57 | 57.72 | 318 | Steelers |
| 126 | 36 | Carl Nassib | 55.05 | 53.71 | 53.86 | 541 | Browns |
| 127 | 37 | Frank Zombo | 54.45 | 49.72 | 57.18 | 501 | Chiefs |
| 128 | 38 | Eugene Sims | 54.10 | 48.15 | 54.84 | 537 | Rams |
| 129 | 39 | Shilique Calhoun | 54.00 | 53.29 | 56.56 | 172 | Raiders |
| 130 | 40 | Tourek Williams | 53.69 | 49.96 | 54.19 | 142 | Chargers |
| 131 | 41 | Terrence Fede | 53.33 | 55.14 | 53.59 | 191 | Dolphins |
| 132 | 42 | Corey Lemonier | 52.82 | 58.77 | 52.39 | 135 | Jets |
| 133 | 43 | Kevin Dodd | 52.06 | 54.44 | 53.60 | 179 | Titans |
| 134 | 44 | Albert McClellan | 51.87 | 43.44 | 54.25 | 604 | Ravens |
| 135 | 45 | Mike Catapano | 51.67 | 49.55 | 51.52 | 210 | Jets |
| 136 | 46 | Matt Longacre | 51.24 | 58.02 | 53.36 | 157 | Rams |
| 137 | 47 | Ryan Russell | 49.52 | 54.43 | 53.15 | 174 | Buccaneers |
| 138 | 48 | DaVonte Lambert | 47.89 | 51.31 | 46.65 | 374 | Buccaneers |
| 139 | 49 | Brennan Scarlett | 47.84 | 55.65 | 51.89 | 119 | Texans |
| 140 | 50 | Freddie Bishop | 45.00 | 51.85 | 51.28 | 152 | Jets |

## G — Guard

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Zack Martin | 93.65 | 89.40 | 92.31 | 1126 | Cowboys |
| 2 | 2 | Marshal Yanda | 92.63 | 87.70 | 91.75 | 899 | Ravens |
| 3 | 3 | David DeCastro | 91.02 | 84.70 | 91.06 | 1274 | Steelers |
| 4 | 4 | Kelechi Osemele | 90.47 | 84.20 | 90.48 | 1110 | Raiders |
| 5 | 5 | Andy Levitre | 90.35 | 84.60 | 90.02 | 1212 | Falcons |
| 6 | 6 | Shaq Mason | 89.62 | 84.60 | 88.80 | 1258 | Patriots |
| 7 | 7 | Kevin Zeitler | 89.18 | 83.30 | 88.94 | 1087 | Bengals |
| 8 | 8 | Quinton Spain | 89.03 | 84.20 | 88.08 | 821 | Titans |
| 9 | 9 | John Greco | 88.90 | 82.60 | 88.93 | 746 | Browns |
| 10 | 10 | Brandon Brooks | 88.16 | 82.40 | 87.84 | 989 | Eagles |
| 11 | 11 | Josh Sitton | 88.11 | 82.10 | 87.95 | 734 | Bears |
| 12 | 12 | Ramon Foster | 87.89 | 83.60 | 86.59 | 1099 | Steelers |
| 13 | 13 | Larry Warford | 87.36 | 80.30 | 87.90 | 1024 | Lions |
| 14 | 14 | T.J. Lang | 86.85 | 81.10 | 86.52 | 964 | Packers |
| 15 | 15 | Evan Mathis | 85.93 | 78.00 | 87.05 | 199 | Cardinals |
| 16 | 16 | Allen Barbre | 85.72 | 79.00 | 86.03 | 672 | Eagles |
| 17 | 17 | Justin Pugh | 85.04 | 78.30 | 85.37 | 750 | Giants |
| 18 | 18 | Ron Leary | 85.01 | 78.30 | 85.31 | 871 | Cowboys |
| 19 | 19 | Brandon Scherff | 84.69 | 77.20 | 85.52 | 1045 | Commanders |
| 20 | 20 | Joel Bitonio | 84.39 | 78.30 | 84.28 | 331 | Browns |
| 21 | 21 | Richie Incognito | 83.86 | 76.50 | 84.60 | 1060 | Bills |
| 22 | 22 | Andrew Norwell | 83.80 | 77.30 | 83.96 | 1108 | Panthers |
| 23 | 23 | Kyle Long | 82.84 | 76.30 | 83.04 | 431 | Bears |
| 24 | 24 | James Carpenter | 82.73 | 76.20 | 82.92 | 994 | Jets |
| 25 | 25 | Tim Lelito | 82.70 | 75.00 | 83.67 | 406 | Saints |
| 26 | 26 | Gabe Jackson | 82.29 | 74.70 | 83.19 | 1188 | Raiders |
| 27 | 27 | Ted Larsen | 82.23 | 73.00 | 84.22 | 581 | Bears |
| 28 | 28 | Dakota Dozier | 80.96 | 79.90 | 77.50 | 144 | Jets |
| 29 | 29 | Rodger Saffold | 80.86 | 72.60 | 82.20 | 916 | Rams |
| 30 | 30 | John Jerry | 80.34 | 72.60 | 81.33 | 1123 | Giants |
| 31 | 31 | A.J. Cann | 80.29 | 72.60 | 81.25 | 1113 | Jaguars |
| 32 | 32 | Max Garcia | 80.15 | 72.20 | 81.28 | 1074 | Broncos |
| 33 | 33 | Isaac Seumalo | 80.01 | 70.60 | 82.12 | 336 | Eagles |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Patrick Omameh | 79.97 | 73.10 | 80.39 | 454 | Jaguars |
| 35 | 2 | Spencer Drango | 79.64 | 72.30 | 80.36 | 599 | Browns |
| 36 | 3 | Chris Chester | 79.40 | 71.00 | 80.83 | 1233 | Falcons |
| 37 | 4 | Brian Winters | 79.33 | 70.50 | 81.05 | 806 | Jets |
| 38 | 5 | Luke Joeckel | 79.28 | 68.90 | 82.04 | 221 | Jaguars |
| 39 | 6 | Laurent Duvernay-Tardif | 78.81 | 72.00 | 79.18 | 946 | Chiefs |
| 40 | 7 | Josh Kline | 78.77 | 70.40 | 80.19 | 929 | Titans |
| 41 | 8 | Mike Iupati | 78.71 | 69.80 | 80.49 | 1035 | Cardinals |
| 42 | 9 | Zach Fulton | 78.14 | 71.20 | 78.60 | 857 | Chiefs |
| 43 | 10 | Parker Ehinger | 77.95 | 68.30 | 80.21 | 229 | Chiefs |
| 44 | 11 | John Miller | 77.85 | 69.60 | 79.18 | 1047 | Bills |
| 45 | 12 | Joe Haeg | 77.79 | 67.90 | 80.21 | 952 | Colts |
| 46 | 13 | Trai Turner | 77.73 | 67.00 | 80.71 | 1098 | Panthers |
| 47 | 14 | Jahri Evans | 77.27 | 69.00 | 78.62 | 1138 | Saints |
| 48 | 15 | Cody Wichmann | 77.12 | 67.80 | 79.17 | 594 | Rams |
| 49 | 16 | Alex Boone | 76.81 | 67.10 | 79.11 | 873 | Vikings |
| 50 | 17 | Xavier Su'a-Filo | 76.62 | 67.60 | 78.46 | 1168 | Texans |
| 51 | 18 | Kenny Wiggins | 76.47 | 65.70 | 79.49 | 134 | Chargers |
| 52 | 19 | Jack Mewhort | 76.11 | 66.80 | 78.15 | 666 | Colts |
| 53 | 20 | Andrew Tiller | 76.00 | 66.10 | 78.44 | 485 | 49ers |
| 54 | 21 | Lane Taylor | 75.60 | 66.50 | 77.50 | 1239 | Packers |
| 55 | 22 | D.J. Fluker | 75.50 | 66.00 | 77.66 | 993 | Chargers |
| 56 | 23 | Joe Thuney | 75.49 | 65.60 | 77.91 | 1354 | Patriots |
| 57 | 24 | Mark Glowinski | 75.09 | 64.60 | 77.91 | 1186 | Seahawks |
| 58 | 25 | Clint Boling | 75.00 | 66.20 | 76.70 | 942 | Bengals |

### Starter (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | Senio Kelemete | 73.83 | 63.00 | 76.88 | 664 | Saints |
| 60 | 2 | Brandon Fusco | 73.55 | 63.50 | 76.09 | 834 | Vikings |
| 61 | 3 | Vladimir Ducasse | 73.06 | 63.10 | 75.53 | 555 | Ravens |
| 62 | 4 | John Urschel | 73.01 | 63.10 | 75.45 | 267 | Ravens |
| 63 | 5 | Chris Scott | 72.99 | 63.80 | 74.95 | 295 | Panthers |
| 64 | 6 | Joshua Garnett | 72.24 | 61.50 | 75.23 | 716 | 49ers |
| 65 | 7 | Orlando Franklin | 71.93 | 61.50 | 74.72 | 919 | Chargers |
| 66 | 8 | Oday Aboushi | 71.30 | 60.30 | 74.47 | 358 | Texans |
| 67 | 9 | Shawn Lauvao | 71.11 | 59.80 | 74.48 | 913 | Commanders |
| 68 | 10 | Alex Lewis | 71.04 | 60.20 | 74.10 | 539 | Ravens |
| 69 | 11 | Denver Kirkland | 70.36 | 63.80 | 70.56 | 130 | Raiders |
| 70 | 12 | Alvin Bailey | 70.19 | 59.70 | 73.01 | 373 | Browns |
| 71 | 13 | Laken Tomlinson | 68.37 | 57.30 | 71.58 | 699 | Lions |
| 72 | 14 | Zane Beadles | 67.84 | 55.30 | 72.04 | 1035 | 49ers |
| 73 | 15 | Jeff Allen | 67.80 | 56.30 | 71.30 | 968 | Texans |
| 74 | 16 | Jermon Bushrod | 67.36 | 55.70 | 70.97 | 1011 | Dolphins |
| 75 | 17 | Kevin Pamphile | 67.32 | 56.10 | 70.64 | 957 | Buccaneers |
| 76 | 18 | Chance Warmack | 66.51 | 54.30 | 70.48 | 134 | Titans |
| 77 | 19 | Ted Karras | 66.27 | 60.60 | 65.88 | 109 | Patriots |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 78 | 1 | Tyler Shatley | 61.40 | 45.30 | 67.96 | 316 | Jaguars |
| 79 | 2 | Arie Kouandjio | 55.39 | 41.10 | 60.75 | 128 | Commanders |
| 80 | 3 | Billy Turner | 53.96 | 32.30 | 64.23 | 138 | Broncos |

## HB — Running Back

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Ty Montgomery | 84.47 | 76.00 | 85.95 | 307 | Packers |
| 2 | 2 | Jalen Richard | 84.21 | 79.60 | 83.12 | 135 | Raiders |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 3 | 1 | Jay Ajayi | 79.15 | 77.30 | 76.21 | 240 | Dolphins |
| 4 | 2 | Jordan Howard | 76.68 | 73.80 | 74.44 | 303 | Bears |
| 5 | 3 | Ezekiel Elliott | 76.57 | 77.20 | 71.98 | 288 | Cowboys |
| 6 | 4 | Le'Veon Bell | 76.54 | 78.20 | 71.26 | 439 | Steelers |
| 7 | 5 | David Johnson | 76.37 | 79.90 | 69.85 | 509 | Cardinals |
| 8 | 6 | Kenneth Dixon | 76.22 | 72.50 | 74.54 | 127 | Ravens |
| 9 | 7 | Mike Gillislee | 75.76 | 79.20 | 69.30 | 104 | Bills |
| 10 | 8 | LeSean McCoy | 75.08 | 78.60 | 68.56 | 254 | Bills |
| 11 | 9 | Spencer Ware | 74.96 | 72.30 | 72.56 | 276 | Chiefs |

### Starter (48 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Thomas Rawls | 73.31 | 75.20 | 67.88 | 142 | Seahawks |
| 13 | 2 | Rex Burkhead | 72.93 | 75.90 | 66.78 | 108 | Bengals |
| 14 | 3 | Darren Sproles | 72.59 | 67.80 | 71.62 | 318 | Eagles |
| 15 | 4 | Chris Thompson | 72.53 | 63.20 | 74.59 | 309 | Commanders |
| 16 | 5 | Bilal Powell | 72.18 | 77.80 | 64.26 | 317 | Jets |
| 17 | 6 | Devonta Freeman | 71.92 | 76.40 | 64.77 | 345 | Falcons |
| 18 | 7 | Mark Ingram II | 71.91 | 71.20 | 68.22 | 239 | Saints |
| 19 | 8 | DeAndre Washington | 71.72 | 65.20 | 71.90 | 127 | Raiders |
| 20 | 9 | DeMarco Murray | 71.34 | 69.50 | 68.40 | 355 | Titans |
| 21 | 10 | Dion Lewis | 71.21 | 64.80 | 71.31 | 104 | Patriots |
| 22 | 11 | Damien Williams | 70.99 | 74.90 | 64.22 | 106 | Dolphins |
| 23 | 12 | Duke Johnson Jr. | 70.43 | 64.60 | 70.15 | 277 | Browns |
| 24 | 13 | Carlos Hyde | 70.41 | 63.60 | 70.79 | 229 | 49ers |
| 25 | 14 | Jacquizz Rodgers | 70.19 | 72.80 | 64.28 | 142 | Buccaneers |
| 26 | 15 | Tevin Coleman | 70.03 | 74.60 | 62.81 | 201 | Falcons |
| 27 | 16 | Theo Riddick | 69.57 | 72.00 | 63.79 | 254 | Lions |
| 28 | 17 | Jonathan Stewart | 68.69 | 63.00 | 68.32 | 179 | Panthers |
| 29 | 18 | Terrance West | 68.56 | 73.90 | 60.83 | 155 | Ravens |
| 30 | 19 | Giovani Bernard | 68.49 | 66.40 | 65.71 | 218 | Bengals |
| 31 | 20 | DeAngelo Williams | 68.31 | 67.80 | 64.49 | 175 | Steelers |
| 32 | 21 | C.J. Anderson | 68.29 | 61.90 | 68.38 | 146 | Broncos |
| 33 | 22 | Melvin Gordon III | 67.72 | 68.90 | 62.76 | 310 | Chargers |
| 34 | 23 | Isaiah Crowell | 67.59 | 67.50 | 63.49 | 212 | Browns |
| 35 | 24 | Matt Forte | 67.48 | 63.80 | 65.77 | 194 | Jets |
| 36 | 25 | Rob Kelley | 67.37 | 59.70 | 68.32 | 110 | Commanders |
| 37 | 26 | T.J. Yeldon | 67.33 | 66.10 | 63.99 | 305 | Jaguars |
| 38 | 27 | Latavius Murray | 67.02 | 66.70 | 63.07 | 230 | Raiders |
| 39 | 28 | Doug Martin | 66.91 | 61.70 | 66.21 | 106 | Buccaneers |
| 40 | 29 | James White | 66.36 | 72.70 | 57.96 | 384 | Patriots |
| 41 | 30 | Jonathan Grimes | 66.21 | 69.40 | 59.92 | 123 | Texans |
| 42 | 31 | LeGarrette Blount | 66.02 | 58.50 | 66.87 | 155 | Patriots |
| 43 | 32 | Christine Michael | 65.95 | 58.30 | 66.88 | 178 | Packers |
| 44 | 33 | Chris Ivory | 65.72 | 53.30 | 69.84 | 140 | Jaguars |
| 45 | 34 | Lance Dunbar | 65.55 | 53.90 | 69.15 | 101 | Cowboys |
| 46 | 35 | Robert Turbin | 65.46 | 66.10 | 60.87 | 166 | Colts |
| 47 | 36 | Jeremy Hill | 65.14 | 63.40 | 62.13 | 160 | Bengals |
| 48 | 37 | Jerick McKinnon | 64.75 | 59.30 | 64.22 | 254 | Vikings |
| 49 | 38 | Lamar Miller | 64.63 | 61.00 | 62.88 | 283 | Texans |
| 50 | 39 | Frank Gore | 63.89 | 59.60 | 62.59 | 263 | Colts |
| 51 | 40 | Todd Gurley II | 63.89 | 57.30 | 64.12 | 333 | Rams |
| 52 | 41 | Benny Cunningham | 63.67 | 58.30 | 63.08 | 116 | Rams |
| 53 | 42 | Justin Forsett | 63.57 | 53.20 | 66.32 | 130 | Broncos |
| 54 | 43 | Paul Perkins | 63.31 | 63.20 | 59.22 | 152 | Giants |
| 55 | 44 | Zach Zenner | 63.04 | 61.60 | 59.83 | 166 | Lions |
| 56 | 45 | Charles Sims | 62.94 | 58.70 | 61.60 | 123 | Buccaneers |
| 57 | 46 | Rashad Jennings | 62.45 | 55.40 | 62.99 | 198 | Giants |
| 58 | 47 | Fozzy Whittaker | 62.39 | 60.60 | 59.42 | 160 | Panthers |
| 59 | 48 | Tim Hightower | 62.25 | 55.00 | 62.92 | 102 | Saints |

### Rotation/backup (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 60 | 1 | Travaris Cadet | 61.68 | 65.40 | 55.04 | 213 | Saints |
| 61 | 2 | Charcandrick West | 61.25 | 58.80 | 58.71 | 228 | Chiefs |
| 62 | 3 | James Starks | 61.00 | 50.50 | 63.84 | 140 | Packers |
| 63 | 4 | Matt Asiata | 60.49 | 59.30 | 57.12 | 179 | Vikings |
| 64 | 5 | Devontae Booker | 60.46 | 55.30 | 59.74 | 227 | Broncos |
| 65 | 6 | Shaun Draughn | 60.05 | 61.10 | 55.19 | 182 | 49ers |
| 66 | 7 | Jeremy Langford | 57.26 | 54.40 | 55.00 | 135 | Bears |
| 67 | 8 | Dwayne Washington | 56.26 | 52.80 | 54.40 | 101 | Lions |

## LB — Linebacker

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Luke Kuechly | 86.50 | 91.00 | 82.46 | 656 | Panthers |
| 2 | 2 | Jerrell Freeman | 86.43 | 90.90 | 82.52 | 806 | Bears |
| 3 | 3 | Bobby Wagner | 83.06 | 86.40 | 77.09 | 1195 | Seahawks |
| 4 | 4 | Jordan Hicks | 80.22 | 88.40 | 74.11 | 971 | Eagles |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Paul Posluszny | 78.42 | 82.00 | 74.36 | 1057 | Jaguars |
| 6 | 2 | Sean Lee | 78.35 | 78.50 | 75.75 | 1041 | Cowboys |
| 7 | 3 | Vincent Rey | 78.22 | 78.50 | 73.86 | 590 | Bengals |
| 8 | 4 | Mason Foster | 77.94 | 79.60 | 75.80 | 771 | Commanders |
| 9 | 5 | Shaq Thompson | 77.89 | 78.60 | 74.55 | 533 | Panthers |
| 10 | 6 | K.J. Wright | 77.44 | 75.60 | 74.50 | 1174 | Seahawks |
| 11 | 7 | C.J. Mosley | 77.40 | 76.30 | 75.00 | 875 | Ravens |
| 12 | 8 | Vontaze Burfict | 76.32 | 83.30 | 73.95 | 674 | Bengals |
| 13 | 9 | Brandon Marshall | 75.85 | 77.30 | 73.54 | 596 | Broncos |
| 14 | 10 | Gerald Hodges | 75.29 | 78.40 | 71.34 | 584 | 49ers |
| 15 | 11 | Nigel Bradham | 74.91 | 75.90 | 72.07 | 989 | Eagles |
| 16 | 12 | Dont'a Hightower | 74.57 | 78.20 | 68.82 | 864 | Patriots |
| 17 | 13 | Benardrick McKinney | 74.16 | 72.00 | 72.22 | 1044 | Texans |

### Starter (57 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Zach Brown | 73.93 | 75.70 | 71.71 | 978 | Bills |
| 19 | 2 | Wesley Woodyard | 73.69 | 74.20 | 69.19 | 614 | Titans |
| 20 | 3 | Ramik Wilson | 73.58 | 78.40 | 73.88 | 591 | Chiefs |
| 21 | 4 | Perry Riley | 73.36 | 75.60 | 72.39 | 698 | Raiders |
| 22 | 5 | Telvin Smith Sr. | 72.89 | 73.00 | 69.29 | 1048 | Jaguars |
| 23 | 6 | Joshua Perry | 72.81 | 78.90 | 72.91 | 114 | Chargers |
| 24 | 7 | Jatavis Brown | 71.88 | 77.60 | 68.06 | 600 | Chargers |
| 25 | 8 | Deion Jones | 71.20 | 70.10 | 67.76 | 1114 | Falcons |
| 26 | 9 | Eric Kendricks | 70.61 | 68.90 | 68.62 | 869 | Vikings |
| 27 | 10 | Bruce Carter | 70.58 | 72.90 | 70.38 | 121 | Jets |
| 28 | 11 | Christian Kirksey | 70.26 | 67.50 | 67.94 | 1111 | Browns |
| 29 | 12 | Brian Cushing | 70.22 | 65.30 | 70.26 | 754 | Texans |
| 30 | 13 | Korey Toomer | 70.16 | 73.00 | 70.87 | 479 | Chargers |
| 31 | 14 | David Harris | 70.03 | 65.00 | 69.73 | 900 | Jets |
| 32 | 15 | Max Bullough | 69.95 | 68.60 | 69.39 | 240 | Texans |
| 33 | 16 | NaVorro Bowman | 69.75 | 71.80 | 70.47 | 251 | 49ers |
| 34 | 17 | Kwon Alexander | 69.63 | 68.90 | 67.52 | 1023 | Buccaneers |
| 35 | 18 | Mark Barron | 69.57 | 65.00 | 68.45 | 1087 | Rams |
| 36 | 19 | Avery Williamson | 69.56 | 64.10 | 69.55 | 908 | Titans |
| 37 | 20 | Kevin Minter | 69.07 | 67.60 | 65.89 | 1002 | Cardinals |
| 38 | 21 | Todd Davis | 68.99 | 66.10 | 68.41 | 697 | Broncos |
| 39 | 22 | Jamie Collins Sr. | 68.84 | 68.80 | 65.85 | 980 | Browns |
| 40 | 23 | Derrick Johnson | 68.46 | 70.50 | 67.62 | 841 | Chiefs |
| 41 | 24 | Karlos Dansby | 68.30 | 65.00 | 67.16 | 782 | Bengals |
| 42 | 25 | Thomas Davis Sr. | 68.16 | 66.40 | 65.16 | 1008 | Panthers |
| 43 | 26 | Anthony Hitchens | 67.99 | 64.60 | 66.09 | 632 | Cowboys |
| 44 | 27 | Myles Jack | 67.56 | 64.50 | 67.51 | 239 | Jaguars |
| 45 | 28 | Sean Spence | 67.40 | 64.20 | 66.84 | 503 | Titans |
| 46 | 29 | Ryan Shazier | 67.38 | 67.20 | 65.42 | 968 | Steelers |
| 47 | 30 | Lavonte David | 67.28 | 63.70 | 65.91 | 1042 | Buccaneers |
| 48 | 31 | Damien Wilson | 67.24 | 69.60 | 68.92 | 289 | Cowboys |
| 49 | 32 | Jake Ryan | 65.75 | 61.20 | 66.57 | 686 | Packers |
| 50 | 33 | Elandon Roberts | 65.69 | 62.30 | 64.82 | 344 | Patriots |
| 51 | 34 | Danny Trevathan | 65.55 | 65.00 | 68.10 | 565 | Bears |
| 52 | 35 | Demario Davis | 65.43 | 61.50 | 63.88 | 786 | Browns |
| 53 | 36 | DeAndre Levy | 65.33 | 66.50 | 70.29 | 248 | Lions |
| 54 | 37 | Alec Ogletree | 65.26 | 63.30 | 66.15 | 1090 | Rams |
| 55 | 38 | Vince Williams | 64.93 | 62.70 | 65.26 | 268 | Steelers |
| 56 | 39 | Corey Nelson | 64.88 | 61.40 | 65.73 | 543 | Broncos |
| 57 | 40 | Spencer Paysinger | 64.82 | 65.10 | 64.63 | 334 | Dolphins |
| 58 | 41 | Preston Brown | 64.81 | 60.20 | 63.72 | 1066 | Bills |
| 59 | 42 | Kiko Alonso | 64.76 | 61.30 | 64.46 | 1106 | Dolphins |
| 60 | 43 | Lawrence Timmons | 64.70 | 59.20 | 64.20 | 1145 | Steelers |
| 61 | 44 | Deone Bucannon | 64.38 | 61.40 | 65.33 | 819 | Cardinals |
| 62 | 45 | Mychal Kendricks | 64.29 | 64.00 | 62.60 | 273 | Eagles |
| 63 | 46 | Michael Morgan | 64.20 | 64.60 | 66.44 | 175 | Seahawks |
| 64 | 47 | Jonathan Casillas | 64.12 | 59.70 | 64.25 | 850 | Giants |
| 65 | 48 | Craig Robertson | 63.87 | 59.90 | 64.12 | 970 | Saints |
| 66 | 49 | Brock Coyle | 63.73 | 65.80 | 68.81 | 128 | Seahawks |
| 67 | 50 | Blake Martinez | 63.73 | 57.80 | 63.51 | 480 | Packers |
| 68 | 51 | Paul Worrilow | 63.55 | 59.60 | 65.45 | 167 | Falcons |
| 69 | 52 | Daryl Smith | 63.47 | 57.50 | 63.28 | 476 | Buccaneers |
| 70 | 53 | Philip Wheeler | 63.18 | 59.30 | 64.00 | 367 | Falcons |
| 71 | 54 | Joe Thomas | 62.94 | 57.00 | 62.74 | 809 | Packers |
| 72 | 55 | Sio Moore | 62.89 | 64.30 | 65.19 | 411 | Cardinals |
| 73 | 56 | Kyle Van Noy | 62.67 | 61.40 | 62.37 | 624 | Patriots |
| 74 | 57 | Malcolm Smith | 62.03 | 55.50 | 63.47 | 971 | Raiders |

### Rotation/backup (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 75 | 1 | Antonio Morrison | 61.87 | 56.10 | 63.63 | 333 | Colts |
| 76 | 2 | Denzel Perryman | 61.80 | 59.30 | 63.08 | 481 | Chargers |
| 77 | 3 | Edwin Jackson | 61.34 | 60.20 | 64.18 | 495 | Colts |
| 78 | 4 | Akeem Dent | 61.31 | 59.80 | 65.34 | 115 | Texans |
| 79 | 5 | Nick Kwiatkoski | 61.22 | 59.10 | 61.60 | 458 | Bears |
| 80 | 6 | Justin Durant | 60.52 | 54.60 | 64.36 | 298 | Cowboys |
| 81 | 7 | Chad Greenway | 60.32 | 54.80 | 60.66 | 399 | Vikings |
| 82 | 8 | Josh Bynes | 60.07 | 57.10 | 61.85 | 421 | Lions |
| 83 | 9 | Koa Misi | 59.91 | 60.30 | 64.23 | 128 | Dolphins |
| 84 | 10 | De'Vondre Campbell | 59.52 | 55.70 | 59.98 | 715 | Falcons |
| 85 | 11 | Donald Butler | 59.48 | 50.00 | 63.52 | 379 | Dolphins |
| 86 | 12 | Will Compton | 59.46 | 53.70 | 60.49 | 937 | Commanders |
| 87 | 13 | Zachary Orr | 59.23 | 54.80 | 62.38 | 961 | Ravens |
| 88 | 14 | Keenan Robinson | 58.54 | 49.80 | 62.09 | 844 | Giants |
| 89 | 15 | Tahir Whitehead | 58.21 | 48.80 | 60.31 | 998 | Lions |
| 90 | 16 | Neville Hewitt | 58.12 | 49.40 | 61.72 | 374 | Dolphins |
| 91 | 17 | Nick Bellore | 57.99 | 59.30 | 62.12 | 692 | 49ers |
| 92 | 18 | Dannell Ellerbe | 57.32 | 58.60 | 62.20 | 443 | Saints |
| 93 | 19 | Sean Weatherspoon | 56.91 | 56.70 | 61.02 | 190 | Falcons |
| 94 | 20 | Nate Stupar | 56.89 | 52.20 | 60.44 | 377 | Saints |
| 95 | 21 | Cory Littleton | 56.87 | 55.40 | 59.94 | 123 | Rams |
| 96 | 22 | Anthony Barr | 56.78 | 50.60 | 57.89 | 1025 | Vikings |
| 97 | 23 | Justin March-Lillard | 56.59 | 61.20 | 65.31 | 159 | Chiefs |
| 98 | 24 | D'Qwell Jackson | 56.42 | 49.40 | 59.02 | 708 | Colts |
| 99 | 25 | Michael Wilhoite | 56.36 | 44.80 | 62.18 | 510 | 49ers |
| 100 | 26 | Mike Hull | 55.73 | 58.00 | 64.05 | 111 | Dolphins |
| 101 | 27 | Ben Heeney | 55.41 | 54.40 | 61.30 | 135 | Raiders |
| 102 | 28 | A.J. Klein | 55.04 | 46.90 | 58.80 | 350 | Panthers |
| 103 | 29 | LaRoy Reynolds | 54.70 | 47.80 | 62.32 | 143 | Falcons |
| 104 | 30 | Darron Lee | 54.54 | 45.50 | 59.54 | 641 | Jets |
| 105 | 31 | Cory James | 54.33 | 46.10 | 58.78 | 410 | Raiders |
| 106 | 32 | Nick Vigil | 53.15 | 46.40 | 58.69 | 111 | Bengals |
| 107 | 33 | Antwione Williams | 52.31 | 46.50 | 56.18 | 204 | Lions |
| 108 | 34 | Kelvin Sheppard | 52.25 | 39.10 | 58.94 | 459 | Giants |
| 109 | 35 | John Timu | 51.67 | 51.30 | 59.34 | 184 | Bears |
| 110 | 36 | Manti Te'o | 51.46 | 48.90 | 58.27 | 142 | Chargers |
| 111 | 37 | Rey Maualuga | 50.58 | 37.30 | 56.94 | 326 | Bengals |
| 112 | 38 | Erin Henderson | 50.24 | 42.30 | 58.77 | 168 | Jets |
| 113 | 39 | Martrell Spaight | 49.87 | 44.20 | 56.85 | 150 | Commanders |
| 114 | 40 | Stephone Anthony | 48.48 | 35.20 | 58.36 | 133 | Saints |
| 115 | 41 | Josh McNary | 45.88 | 33.50 | 56.63 | 178 | Colts |
| 116 | 42 | Jelani Jenkins | 45.00 | 29.90 | 55.17 | 406 | Dolphins |
| 117 | 43 | Julian Stanford | 45.00 | 29.50 | 54.16 | 244 | Jets |

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

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Andy Dalton | 73.61 | 71.70 | 72.08 | 660 | Bengals |
| 14 | 2 | Alex Smith | 73.27 | 74.94 | 68.12 | 607 | Chiefs |
| 15 | 3 | Ryan Tannehill | 73.18 | 71.64 | 73.27 | 456 | Dolphins |
| 16 | 4 | Dak Prescott | 72.37 | 81.50 | 76.00 | 586 | Cowboys |
| 17 | 5 | Philip Rivers | 72.36 | 70.52 | 69.33 | 659 | Chargers |
| 18 | 6 | Tyrod Taylor | 71.01 | 74.53 | 67.88 | 567 | Bills |
| 19 | 7 | Jameis Winston | 70.99 | 70.93 | 66.58 | 689 | Buccaneers |
| 20 | 8 | Cam Newton | 70.97 | 74.14 | 64.86 | 596 | Panthers |
| 21 | 9 | Marcus Mariota | 69.86 | 65.76 | 72.56 | 541 | Titans |
| 22 | 10 | Eli Manning | 67.19 | 64.31 | 65.00 | 705 | Giants |
| 23 | 11 | Brian Hoyer | 66.45 | 68.48 | 69.75 | 216 | Bears |
| 24 | 12 | Joe Flacco | 66.44 | 66.35 | 63.29 | 759 | Ravens |
| 25 | 13 | Carson Wentz | 64.69 | 68.30 | 60.24 | 700 | Eagles |
| 26 | 14 | Cody Kessler | 62.76 | 71.90 | 68.45 | 235 | Browns |

### Rotation/backup (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | Jay Cutler | 61.88 | 63.46 | 65.40 | 167 | Bears |
| 28 | 2 | Matt Moore | 61.68 | 58.78 | 76.87 | 136 | Dolphins |
| 29 | 3 | Matt Barkley | 61.65 | 73.06 | 64.88 | 241 | Bears |
| 30 | 4 | Blake Bortles | 61.00 | 56.45 | 60.35 | 745 | Jaguars |
| 31 | 5 | Trevor Siemian | 59.96 | 56.26 | 64.85 | 566 | Broncos |
| 32 | 6 | Colin Kaepernick | 59.48 | 54.87 | 65.12 | 421 | 49ers |
| 33 | 7 | Ryan Fitzpatrick | 59.11 | 57.67 | 59.72 | 466 | Jets |
| 34 | 8 | Case Keenum | 58.31 | 59.35 | 61.21 | 373 | Rams |
| 35 | 9 | Paxton Lynch | 55.44 | 50.30 | 58.59 | 107 | Broncos |
| 36 | 10 | Josh McCown | 55.35 | 57.67 | 59.81 | 199 | Browns |
| 37 | 11 | Robert Griffin III | 55.11 | 58.04 | 59.22 | 194 | Browns |
| 38 | 12 | Brock Osweiler | 54.96 | 52.35 | 56.47 | 659 | Texans |
| 39 | 13 | Blaine Gabbert | 54.60 | 52.27 | 56.94 | 198 | 49ers |
| 40 | 14 | Bryce Petty | 54.39 | 47.48 | 56.67 | 157 | Jets |
| 41 | 15 | Jared Goff | 53.50 | 42.90 | 54.38 | 253 | Rams |

## S — Safety

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Devin McCourty | 94.97 | 92.10 | 92.72 | 1209 | Patriots |
| 2 | 2 | Eric Weddle | 92.27 | 88.70 | 91.42 | 1031 | Ravens |
| 3 | 3 | Eric Berry | 90.67 | 90.30 | 88.83 | 1151 | Chiefs |
| 4 | 4 | Keith Tandy | 89.86 | 90.50 | 88.40 | 402 | Buccaneers |
| 5 | 5 | Ricardo Allen | 87.20 | 85.80 | 83.97 | 1322 | Falcons |
| 6 | 6 | Quintin Demps | 85.34 | 83.10 | 84.23 | 709 | Texans |
| 7 | 7 | Reshad Jones | 83.84 | 86.40 | 84.01 | 437 | Dolphins |
| 8 | 8 | Barry Church | 83.16 | 81.40 | 82.05 | 724 | Cowboys |
| 9 | 9 | J.J. Wilcox | 83.10 | 82.50 | 80.36 | 573 | Cowboys |
| 10 | 10 | Glover Quin | 82.59 | 79.30 | 80.61 | 1099 | Lions |
| 11 | 11 | Kam Chancellor | 82.49 | 78.50 | 82.97 | 854 | Seahawks |
| 12 | 12 | D.J. Swearinger Sr. | 82.43 | 82.80 | 79.90 | 839 | Cardinals |
| 13 | 13 | Keanu Neal | 80.83 | 80.30 | 77.02 | 1136 | Falcons |
| 14 | 14 | Landon Collins | 80.15 | 73.40 | 80.49 | 1176 | Giants |
| 15 | 15 | Jeff Heath | 80.04 | 78.50 | 79.81 | 263 | Cowboys |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Eddie Pleasant | 79.16 | 83.50 | 74.19 | 285 | Texans |
| 17 | 2 | Earl Thomas III | 78.97 | 80.30 | 76.51 | 693 | Seahawks |
| 18 | 3 | Justin Simmons | 78.80 | 75.40 | 81.07 | 294 | Broncos |
| 19 | 4 | Daniel Sorensen | 78.63 | 80.50 | 77.38 | 573 | Chiefs |
| 20 | 5 | Tavon Wilson | 78.49 | 73.40 | 77.72 | 775 | Lions |
| 21 | 6 | Jahleel Addae | 78.45 | 80.30 | 79.20 | 510 | Chargers |
| 22 | 7 | Ron Parker | 77.48 | 74.50 | 75.30 | 1168 | Chiefs |
| 23 | 8 | Mike Mitchell | 77.46 | 71.20 | 77.46 | 1197 | Steelers |
| 24 | 9 | Kurt Coleman | 77.44 | 72.50 | 77.30 | 996 | Panthers |
| 25 | 10 | Morgan Burnett | 77.22 | 71.30 | 77.93 | 1096 | Packers |
| 26 | 11 | Maurice Alexander | 77.13 | 79.80 | 73.79 | 920 | Rams |
| 27 | 12 | Malcolm Jenkins | 76.02 | 74.70 | 72.74 | 1018 | Eagles |
| 28 | 13 | Darian Stewart | 75.59 | 72.80 | 73.29 | 1080 | Broncos |
| 29 | 14 | Dwight Lowery | 75.34 | 73.10 | 72.66 | 1003 | Chargers |
| 30 | 15 | Mike Adams | 75.21 | 70.10 | 75.91 | 997 | Colts |
| 31 | 16 | Reggie Nelson | 75.04 | 69.30 | 74.70 | 1121 | Raiders |
| 32 | 17 | Jairus Byrd | 75.03 | 70.50 | 77.31 | 900 | Saints |
| 33 | 18 | George Iloka | 74.72 | 69.40 | 75.03 | 1050 | Bengals |
| 34 | 19 | Rodney McLeod | 74.57 | 70.10 | 73.39 | 1014 | Eagles |
| 35 | 20 | Duron Harmon | 74.36 | 71.30 | 72.23 | 617 | Patriots |

### Starter (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 36 | 1 | Harrison Smith | 73.97 | 68.50 | 75.11 | 894 | Vikings |
| 37 | 2 | Ha Ha Clinton-Dix | 73.77 | 63.10 | 76.72 | 1231 | Packers |
| 38 | 3 | Adrian Amos | 73.52 | 70.30 | 72.54 | 938 | Bears |
| 39 | 4 | Isa Abdul-Quddus | 73.37 | 70.20 | 72.47 | 952 | Dolphins |
| 40 | 5 | Rontez Miles | 72.81 | 66.40 | 75.41 | 393 | Jets |
| 41 | 6 | Karl Joseph | 72.23 | 67.10 | 76.69 | 623 | Raiders |
| 42 | 7 | Shawn Williams | 72.14 | 67.00 | 73.49 | 912 | Bengals |
| 43 | 8 | Dexter McCoil | 71.38 | 72.20 | 71.87 | 248 | Chargers |
| 44 | 9 | Kevin Byard | 71.22 | 66.10 | 70.47 | 656 | Titans |
| 45 | 10 | Corey Graham | 71.08 | 67.90 | 69.04 | 1053 | Bills |
| 46 | 11 | Da'Norris Searcy | 70.55 | 66.90 | 70.39 | 553 | Titans |
| 47 | 12 | Tre Boston | 70.14 | 65.40 | 71.01 | 840 | Panthers |
| 48 | 13 | T.J. McDonald | 69.66 | 68.80 | 67.63 | 1072 | Rams |
| 49 | 14 | Tony Jefferson | 69.04 | 59.50 | 72.27 | 931 | Cardinals |
| 50 | 15 | Miles Killebrew | 68.77 | 64.80 | 68.29 | 158 | Lions |
| 51 | 16 | Clayton Geathers | 68.74 | 65.50 | 72.07 | 559 | Colts |
| 52 | 17 | T.J. Ward | 68.16 | 65.20 | 67.31 | 984 | Broncos |
| 53 | 18 | Bradley McDougald | 67.99 | 62.90 | 67.63 | 1012 | Buccaneers |
| 54 | 19 | Corey Moore | 67.93 | 60.90 | 70.53 | 501 | Texans |
| 55 | 20 | Chris Prosinski | 67.54 | 62.00 | 73.42 | 173 | Bears |
| 56 | 21 | Kentrell Brice | 67.23 | 60.10 | 67.82 | 334 | Packers |
| 57 | 22 | Marcus Gilchrist | 67.15 | 60.20 | 69.18 | 819 | Jets |
| 58 | 23 | Johnathan Cyprien | 67.02 | 59.10 | 68.97 | 1070 | Jaguars |
| 59 | 24 | Kenny Vaccaro | 66.61 | 61.80 | 68.47 | 720 | Saints |
| 60 | 25 | Nate Allen | 65.31 | 63.60 | 70.62 | 232 | Raiders |
| 61 | 26 | Harold Jones-Quartey | 65.14 | 57.10 | 67.37 | 731 | Bears |
| 62 | 27 | Tyrann Mathieu | 64.99 | 60.40 | 67.43 | 561 | Cardinals |
| 63 | 28 | Rashad Johnson | 64.87 | 60.60 | 64.59 | 555 | Titans |
| 64 | 29 | Andre Hal | 64.71 | 59.50 | 64.33 | 949 | Texans |
| 65 | 30 | Rafael Bush | 64.19 | 60.30 | 68.55 | 512 | Lions |
| 66 | 31 | Jordan Dangerfield | 64.17 | 69.60 | 63.69 | 145 | Steelers |
| 67 | 32 | Cody Davis | 64.10 | 64.80 | 66.03 | 278 | Rams |
| 68 | 33 | Daimion Stafford | 64.00 | 58.90 | 65.12 | 613 | Titans |
| 69 | 34 | Andrew Sendejo | 63.37 | 60.40 | 64.10 | 855 | Vikings |
| 70 | 35 | Jaquiski Tartt | 63.36 | 59.10 | 63.06 | 612 | 49ers |
| 71 | 36 | Sean Davis | 63.31 | 60.50 | 61.01 | 930 | Steelers |
| 72 | 37 | Jaylen Watkins | 62.96 | 53.70 | 65.48 | 387 | Eagles |
| 73 | 38 | Antoine Bethea | 62.82 | 54.10 | 67.28 | 1125 | 49ers |
| 74 | 39 | Duke Ihenacho | 62.35 | 67.40 | 63.67 | 638 | Commanders |
| 75 | 40 | Eric Reid | 62.15 | 55.80 | 65.55 | 742 | 49ers |

### Rotation/backup (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 76 | 1 | Derron Smith | 61.81 | 62.00 | 65.98 | 103 | Bengals |
| 77 | 2 | Donte Whitner | 61.69 | 52.80 | 66.69 | 564 | Commanders |
| 78 | 3 | Kemal Ishmael | 61.57 | 55.80 | 64.48 | 310 | Falcons |
| 79 | 4 | Jordan Poyer | 61.07 | 53.90 | 68.98 | 354 | Browns |
| 80 | 5 | Will Parks | 60.54 | 53.10 | 62.37 | 268 | Broncos |
| 81 | 6 | Deon Bush | 60.44 | 62.10 | 63.50 | 333 | Bears |
| 82 | 7 | Tashaun Gipson Sr. | 60.19 | 47.40 | 66.53 | 1040 | Jaguars |
| 83 | 8 | Michael Thomas | 59.23 | 52.90 | 61.16 | 620 | Dolphins |
| 84 | 9 | Vonn Bell | 59.19 | 44.60 | 65.79 | 889 | Saints |
| 85 | 10 | Patrick Chung | 58.88 | 48.30 | 61.77 | 1176 | Patriots |
| 86 | 11 | Roman Harper | 58.35 | 55.40 | 58.24 | 300 | Saints |
| 87 | 12 | James Ihedigbo | 58.04 | 58.70 | 60.42 | 146 | Bills |
| 88 | 13 | Robert Blanton | 57.91 | 52.70 | 61.80 | 270 | Bills |
| 89 | 14 | Derrick Kindred | 57.83 | 54.30 | 60.18 | 538 | Browns |
| 90 | 15 | Adrian Phillips | 57.43 | 46.20 | 62.83 | 542 | Chargers |
| 91 | 16 | Aaron Williams | 56.92 | 55.70 | 62.54 | 340 | Bills |
| 92 | 17 | Michael Griffin | 56.29 | 52.00 | 57.90 | 284 | Panthers |
| 93 | 18 | Robert Golden | 56.00 | 50.40 | 59.93 | 382 | Steelers |
| 94 | 19 | Chris Conte | 55.66 | 48.30 | 58.90 | 718 | Buccaneers |
| 95 | 20 | Calvin Pryor | 55.09 | 46.50 | 58.32 | 814 | Jets |
| 96 | 21 | Ed Reynolds Jr. | 54.97 | 50.60 | 62.05 | 505 | Browns |
| 97 | 22 | Bacarri Rambo | 54.28 | 48.60 | 59.41 | 525 | Dolphins |
| 98 | 23 | Ibraheim Campbell | 52.63 | 51.20 | 57.62 | 418 | Browns |
| 99 | 24 | Anthony Harris | 51.89 | 43.30 | 64.39 | 235 | Vikings |
| 100 | 25 | Kelcie McCray | 50.20 | 45.20 | 55.94 | 344 | Seahawks |
| 101 | 26 | Nat Berhe | 49.30 | 46.90 | 56.50 | 165 | Giants |
| 102 | 27 | Keith McGill | 48.81 | 40.60 | 53.76 | 150 | Raiders |
| 103 | 28 | T.J. Green | 45.00 | 30.20 | 51.32 | 478 | Colts |
| 104 | 29 | Blake Countess | 45.00 | 28.90 | 55.63 | 142 | Rams |

## T — Tackle

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (39 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Trent Williams | 96.75 | 92.70 | 95.29 | 795 | Commanders |
| 2 | 2 | Donald Penn | 94.82 | 90.00 | 93.86 | 1108 | Raiders |
| 3 | 3 | Taylor Lewan | 93.12 | 87.00 | 93.03 | 991 | Titans |
| 4 | 4 | Marcus Cannon | 92.74 | 86.60 | 92.67 | 1272 | Patriots |
| 5 | 5 | Jason Peters | 92.06 | 86.90 | 91.33 | 1098 | Eagles |
| 6 | 6 | Lane Johnson | 90.97 | 84.40 | 91.18 | 407 | Eagles |
| 7 | 7 | Tyron Smith | 90.61 | 85.90 | 89.58 | 902 | Cowboys |
| 8 | 8 | Ryan Schraeder | 90.53 | 83.40 | 91.12 | 1223 | Falcons |
| 9 | 9 | Joe Thomas | 89.94 | 87.00 | 87.73 | 1030 | Browns |
| 10 | 10 | Duane Brown | 89.93 | 86.20 | 88.25 | 918 | Texans |
| 11 | 11 | Andrew Whitworth | 89.29 | 84.00 | 88.65 | 1064 | Bengals |
| 12 | 12 | David Bakhtiari | 89.02 | 86.60 | 86.47 | 1257 | Packers |
| 13 | 13 | Terron Armstead | 87.80 | 82.60 | 87.10 | 397 | Saints |
| 14 | 14 | Anthony Castonzo | 87.77 | 83.40 | 86.51 | 1074 | Colts |
| 15 | 15 | Nate Solder | 87.57 | 82.20 | 86.99 | 1270 | Patriots |
| 16 | 16 | Taylor Decker | 87.00 | 81.90 | 86.23 | 1089 | Lions |
| 17 | 17 | Jack Conklin | 86.66 | 80.60 | 86.54 | 1062 | Titans |
| 18 | 18 | Joe Staley | 86.34 | 81.10 | 85.67 | 845 | 49ers |
| 19 | 19 | Marcus Gilbert | 86.01 | 78.50 | 86.85 | 1038 | Steelers |
| 20 | 20 | Eric Fisher | 85.59 | 77.70 | 86.68 | 1078 | Chiefs |
| 21 | 21 | Alejandro Villanueva | 85.47 | 79.90 | 85.02 | 1276 | Steelers |
| 22 | 22 | Cordy Glenn | 85.14 | 78.10 | 85.67 | 657 | Bills |
| 23 | 23 | Ty Nsekhe | 84.84 | 78.30 | 85.04 | 385 | Commanders |
| 24 | 24 | Zach Strief | 84.69 | 77.40 | 85.38 | 1125 | Saints |
| 25 | 25 | Morgan Moses | 84.50 | 76.80 | 85.46 | 1017 | Commanders |
| 26 | 26 | Jared Veldheer | 84.06 | 76.50 | 84.94 | 578 | Cardinals |
| 27 | 27 | Cam Fleming | 83.77 | 73.60 | 86.39 | 301 | Patriots |
| 28 | 28 | Bryan Bulaga | 83.74 | 78.10 | 83.33 | 1256 | Packers |
| 29 | 29 | Demar Dotson | 83.69 | 74.20 | 85.85 | 942 | Buccaneers |
| 30 | 30 | Ja'Wuan James | 83.59 | 75.20 | 85.02 | 1000 | Dolphins |
| 31 | 31 | Jermey Parnell | 82.91 | 73.00 | 85.35 | 1113 | Jaguars |
| 32 | 32 | Mitchell Schwartz | 82.29 | 73.40 | 84.05 | 1078 | Chiefs |
| 33 | 33 | Ronnie Stanley | 82.21 | 74.80 | 82.98 | 834 | Ravens |
| 34 | 34 | Russell Okung | 82.03 | 74.00 | 83.21 | 1061 | Broncos |
| 35 | 35 | Cyrus Kouandjio | 81.94 | 74.80 | 82.53 | 406 | Bills |
| 36 | 36 | Rick Wagner | 81.47 | 74.00 | 82.29 | 926 | Ravens |
| 37 | 37 | Austin Pasztor | 80.42 | 71.80 | 82.00 | 1020 | Browns |
| 38 | 38 | Mike Remmers | 80.14 | 71.20 | 81.94 | 1106 | Panthers |
| 39 | 39 | Jake Matthews | 80.08 | 71.90 | 81.37 | 1172 | Falcons |

### Good (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 40 | 1 | Ereck Flowers | 79.51 | 69.40 | 82.08 | 1125 | Giants |
| 41 | 2 | D.J. Humphries | 79.29 | 68.30 | 82.45 | 922 | Cardinals |
| 42 | 3 | Charles Leno Jr. | 79.18 | 70.10 | 81.06 | 1011 | Bears |
| 43 | 4 | Halapoulivaati Vaitai | 78.52 | 68.80 | 80.84 | 421 | Eagles |
| 44 | 5 | Riley Reiff | 78.19 | 69.10 | 80.08 | 888 | Lions |
| 45 | 6 | Menelik Watson | 78.03 | 64.40 | 82.95 | 328 | Raiders |
| 46 | 7 | Doug Free | 77.93 | 67.50 | 80.72 | 1126 | Cowboys |
| 47 | 8 | Austin Howard | 77.48 | 67.20 | 80.16 | 792 | Raiders |
| 48 | 9 | Marshall Newhouse | 77.48 | 65.60 | 81.24 | 524 | Giants |
| 49 | 10 | Kendall Lamm | 77.30 | 68.50 | 79.00 | 167 | Texans |
| 50 | 11 | Daryl Williams | 77.21 | 67.50 | 79.52 | 647 | Panthers |
| 51 | 12 | Eric Winston | 76.99 | 65.30 | 80.61 | 282 | Bengals |
| 52 | 13 | King Dunlap | 76.79 | 68.10 | 78.42 | 775 | Chargers |
| 53 | 14 | Bobby Massie | 76.55 | 66.80 | 78.88 | 916 | Bears |
| 54 | 15 | Le'Raven Clark | 76.05 | 64.30 | 79.72 | 201 | Colts |
| 55 | 16 | Donovan Smith | 76.01 | 63.90 | 79.91 | 1135 | Buccaneers |
| 56 | 17 | Ryan Clady | 75.67 | 65.80 | 78.08 | 537 | Jets |
| 57 | 18 | Chris Hubbard | 75.45 | 68.00 | 76.25 | 351 | Steelers |
| 58 | 19 | Brent Qvale | 74.99 | 62.40 | 79.22 | 347 | Jets |
| 59 | 20 | Trent Brown | 74.31 | 61.90 | 78.42 | 1035 | 49ers |
| 60 | 21 | Ben Ijalana | 74.05 | 63.80 | 76.72 | 867 | Jets |

### Starter (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 61 | 1 | Jake Long | 73.74 | 62.70 | 76.93 | 209 | Vikings |
| 62 | 2 | Brandon Shell | 73.46 | 73.40 | 69.34 | 204 | Jets |
| 63 | 3 | Derek Newton | 73.45 | 61.30 | 77.38 | 358 | Texans |
| 64 | 4 | Michael Oher | 73.04 | 63.20 | 75.43 | 232 | Panthers |
| 65 | 5 | Jordan Mills | 72.97 | 61.80 | 76.25 | 1033 | Bills |
| 66 | 6 | Cedric Ogbuehi | 72.91 | 60.60 | 76.95 | 677 | Bengals |
| 67 | 7 | Dennis Kelly | 72.90 | 60.70 | 76.86 | 146 | Titans |
| 68 | 8 | Kelvin Beachum | 72.56 | 61.40 | 75.83 | 1024 | Jaguars |
| 69 | 9 | Andrew Donnal | 72.43 | 63.50 | 74.21 | 297 | Rams |
| 70 | 10 | Branden Albert | 72.33 | 61.50 | 75.38 | 786 | Dolphins |
| 71 | 11 | James Hurst | 72.16 | 60.90 | 75.50 | 305 | Ravens |
| 72 | 12 | John Wetzel | 72.04 | 58.90 | 76.63 | 647 | Cardinals |
| 73 | 13 | Vadal Alexander | 72.04 | 55.10 | 79.17 | 306 | Raiders |
| 74 | 14 | Chris Hairston | 71.66 | 60.00 | 75.27 | 327 | Chargers |
| 75 | 15 | Greg Robinson | 71.29 | 59.90 | 74.72 | 892 | Rams |
| 76 | 16 | Sam Young | 71.00 | 59.10 | 74.76 | 146 | Dolphins |
| 77 | 17 | Jah Reid | 70.96 | 56.30 | 76.57 | 101 | Chiefs |
| 78 | 18 | Bobby Hart | 70.90 | 56.30 | 76.46 | 868 | Giants |
| 79 | 19 | Chris Clark | 70.56 | 56.90 | 75.50 | 1219 | Texans |
| 80 | 20 | Joe Barksdale | 70.07 | 56.30 | 75.08 | 967 | Chargers |
| 81 | 21 | Garry Gilliam | 70.06 | 56.70 | 74.80 | 939 | Seahawks |
| 82 | 22 | Ulrick John | 69.10 | 52.00 | 76.34 | 212 | Cardinals |
| 83 | 23 | Bradley Sowell | 68.09 | 53.30 | 73.78 | 629 | Seahawks |
| 84 | 24 | Matt Kalil | 67.78 | 52.40 | 73.87 | 121 | Vikings |
| 85 | 25 | Corey Robinson | 67.51 | 51.80 | 73.82 | 165 | Lions |
| 86 | 26 | Jake Fisher | 67.09 | 47.70 | 75.85 | 296 | Bengals |
| 87 | 27 | Breno Giacomini | 66.51 | 50.80 | 72.81 | 266 | Jets |
| 88 | 28 | T.J. Clemmings | 65.22 | 46.90 | 73.26 | 882 | Vikings |
| 89 | 29 | Donald Stephenson | 64.48 | 44.10 | 73.90 | 744 | Broncos |
| 90 | 30 | Gosder Cherilus | 63.83 | 44.40 | 72.61 | 219 | Buccaneers |
| 91 | 31 | Jason Spriggs | 63.32 | 49.10 | 68.63 | 276 | Packers |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 92 | 1 | Ty Sambrailo | 61.56 | 41.70 | 70.64 | 243 | Broncos |
| 93 | 2 | George Fant | 61.51 | 42.70 | 69.88 | 792 | Seahawks |

## TE — Tight End

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 88.80 | 90.20 | 83.70 | 214 | Patriots |
| 2 | 2 | Travis Kelce | 85.38 | 88.90 | 78.87 | 607 | Chiefs |
| 3 | 3 | Jimmy Graham | 84.22 | 85.40 | 79.27 | 603 | Seahawks |
| 4 | 4 | Hunter Henry | 83.99 | 83.00 | 80.49 | 290 | Chargers |
| 5 | 5 | Erik Swoope | 82.97 | 74.30 | 84.59 | 108 | Colts |
| 6 | 6 | Greg Olsen | 82.10 | 83.40 | 77.07 | 598 | Panthers |
| 7 | 7 | Ladarius Green | 81.19 | 78.40 | 78.88 | 114 | Steelers |
| 8 | 8 | Cameron Brate | 80.49 | 80.50 | 76.31 | 460 | Buccaneers |
| 9 | 9 | Jordan Reed | 80.45 | 83.60 | 74.18 | 390 | Commanders |
| 10 | 10 | Jeff Heuerman | 80.31 | 75.20 | 79.55 | 130 | Broncos |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Jared Cook | 78.97 | 76.10 | 76.72 | 340 | Packers |
| 12 | 2 | Delanie Walker | 78.40 | 76.00 | 75.83 | 479 | Titans |
| 13 | 3 | Tyler Eifert | 77.75 | 80.70 | 71.62 | 269 | Bengals |
| 14 | 4 | Vernon Davis | 77.68 | 76.00 | 74.63 | 357 | Commanders |
| 15 | 5 | C.J. Fiedorowicz | 77.02 | 78.50 | 71.86 | 395 | Texans |
| 16 | 6 | Anthony Fasano | 76.98 | 78.60 | 71.73 | 187 | Titans |
| 17 | 7 | Martellus Bennett | 76.21 | 75.50 | 72.51 | 608 | Patriots |
| 18 | 8 | Zach Miller | 75.60 | 73.00 | 73.16 | 335 | Bears |
| 19 | 9 | Austin Hooper | 75.42 | 68.50 | 75.87 | 308 | Falcons |
| 20 | 10 | Kyle Rudolph | 75.08 | 73.20 | 72.16 | 621 | Vikings |
| 21 | 11 | Gary Barnidge | 75.06 | 68.60 | 75.20 | 635 | Browns |
| 22 | 12 | Zach Ertz | 74.97 | 70.90 | 73.51 | 560 | Eagles |
| 23 | 13 | Antonio Gates | 74.16 | 73.20 | 70.63 | 431 | Chargers |

### Starter (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Jacob Tamme | 73.50 | 70.40 | 71.40 | 202 | Falcons |
| 25 | 2 | Rhett Ellison | 73.13 | 69.30 | 71.51 | 106 | Vikings |
| 26 | 3 | Vance McDonald | 72.78 | 68.70 | 71.33 | 262 | 49ers |
| 27 | 4 | Charles Clay | 72.67 | 68.20 | 71.49 | 500 | Bills |
| 28 | 5 | Austin Seferian-Jenkins | 72.65 | 62.10 | 75.51 | 126 | Jets |
| 29 | 6 | Eric Ebron | 72.51 | 65.40 | 73.09 | 555 | Lions |
| 30 | 7 | Marcedes Lewis | 72.41 | 66.20 | 72.39 | 167 | Jaguars |
| 31 | 8 | Coby Fleener | 72.40 | 61.20 | 75.70 | 513 | Saints |
| 32 | 9 | Darren Fells | 72.36 | 63.00 | 74.43 | 198 | Cardinals |
| 33 | 10 | Brent Celek | 72.31 | 65.40 | 72.75 | 188 | Eagles |
| 34 | 11 | Jason Witten | 72.24 | 68.80 | 70.36 | 580 | Cowboys |
| 35 | 12 | Jack Doyle | 72.22 | 75.10 | 66.14 | 421 | Colts |
| 36 | 13 | Mychal Rivera | 71.21 | 64.90 | 71.25 | 201 | Raiders |
| 37 | 14 | Virgil Green | 71.03 | 67.40 | 69.29 | 287 | Broncos |
| 38 | 15 | Nick O'Leary | 70.22 | 66.10 | 68.80 | 206 | Bills |
| 39 | 16 | Trey Burton | 69.99 | 65.70 | 68.68 | 231 | Eagles |
| 40 | 17 | Clive Walford | 69.86 | 58.20 | 73.46 | 409 | Raiders |
| 41 | 18 | Luke Willson | 69.78 | 60.60 | 71.73 | 210 | Seahawks |
| 42 | 19 | Lance Kendricks | 69.64 | 63.70 | 69.43 | 498 | Rams |
| 43 | 20 | A.J. Derby | 69.02 | 58.50 | 71.87 | 139 | Broncos |
| 44 | 21 | Ben Koyack | 68.92 | 61.90 | 69.44 | 188 | Jaguars |
| 45 | 22 | Jesse James | 68.30 | 64.80 | 66.46 | 580 | Steelers |
| 46 | 23 | Ed Dickson | 68.21 | 62.50 | 67.85 | 216 | Panthers |
| 47 | 24 | Neal Sterling | 68.14 | 59.00 | 70.07 | 118 | Jaguars |
| 48 | 25 | Dwayne Allen | 67.98 | 64.60 | 66.06 | 368 | Colts |
| 49 | 26 | Levine Toilolo | 67.97 | 66.10 | 65.05 | 325 | Falcons |
| 50 | 27 | Julius Thomas | 67.02 | 61.80 | 66.33 | 339 | Jaguars |
| 51 | 28 | Jermaine Gresham | 66.88 | 63.80 | 64.76 | 477 | Cardinals |
| 52 | 29 | Will Tye | 66.86 | 57.90 | 68.67 | 434 | Giants |
| 53 | 30 | Richard Rodgers | 66.79 | 57.20 | 69.02 | 405 | Packers |
| 54 | 31 | Garrett Celek | 66.66 | 58.30 | 68.07 | 315 | 49ers |
| 55 | 32 | Jerell Adams | 66.58 | 55.10 | 70.07 | 123 | Giants |
| 56 | 33 | Dennis Pitta | 66.52 | 61.80 | 65.50 | 605 | Ravens |
| 57 | 34 | Josh Hill | 66.51 | 57.90 | 68.08 | 179 | Saints |
| 58 | 35 | Ryan Griffin | 66.27 | 57.10 | 68.22 | 354 | Texans |
| 59 | 36 | John Phillips | 65.72 | 59.90 | 65.43 | 179 | Saints |
| 60 | 37 | C.J. Uzomah | 65.57 | 63.00 | 63.11 | 260 | Bengals |
| 61 | 38 | Brandon Myers | 65.52 | 52.20 | 70.24 | 196 | Buccaneers |
| 62 | 39 | Dion Sims | 65.47 | 60.80 | 64.42 | 447 | Dolphins |
| 63 | 40 | Xavier Grimble | 65.16 | 60.20 | 64.30 | 154 | Steelers |
| 64 | 41 | Tyler Kroft | 64.90 | 56.60 | 66.27 | 162 | Bengals |
| 65 | 42 | Kellen Davis | 62.42 | 51.50 | 65.54 | 123 | Jets |

### Rotation/backup (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 66 | 1 | Daniel Brown | 61.79 | 59.60 | 59.08 | 181 | Bears |
| 67 | 2 | Logan Paulsen | 61.32 | 48.60 | 65.63 | 161 | Bears |
| 68 | 3 | Larry Donnell | 61.07 | 41.60 | 69.89 | 133 | Giants |
| 69 | 4 | Darren Waller | 60.62 | 52.80 | 61.66 | 134 | Ravens |
| 70 | 5 | Randall Telfer | 60.13 | 53.70 | 60.25 | 112 | Browns |
| 71 | 6 | Tyler Higbee | 58.62 | 45.30 | 63.33 | 231 | Rams |
| 72 | 7 | Demetrius Harris | 56.90 | 47.30 | 59.13 | 253 | Chiefs |

## WR — Wide Receiver

- **Season used:** `2016`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Julio Jones | 90.65 | 93.50 | 84.58 | 571 | Falcons |
| 2 | 2 | A.J. Green | 88.16 | 90.00 | 82.76 | 361 | Bengals |
| 3 | 3 | Antonio Brown | 86.90 | 88.40 | 81.73 | 711 | Steelers |
| 4 | 4 | Mike Evans | 85.61 | 91.70 | 77.39 | 630 | Buccaneers |
| 5 | 5 | T.Y. Hilton | 85.52 | 86.70 | 80.56 | 659 | Colts |
| 6 | 6 | Odell Beckham Jr. | 84.62 | 84.60 | 80.47 | 680 | Giants |
| 7 | 7 | Taylor Gabriel | 84.54 | 85.90 | 79.47 | 321 | Falcons |
| 8 | 8 | Dez Bryant | 83.55 | 83.50 | 79.42 | 480 | Cowboys |
| 9 | 9 | Michael Thomas | 83.54 | 86.40 | 77.47 | 587 | Saints |
| 10 | 10 | Doug Baldwin | 83.06 | 86.10 | 76.87 | 692 | Seahawks |
| 11 | 11 | Rishard Matthews | 82.83 | 81.40 | 79.61 | 475 | Titans |
| 12 | 12 | Emmanuel Sanders | 82.73 | 81.50 | 79.39 | 575 | Broncos |
| 13 | 13 | Jordy Nelson | 82.23 | 82.20 | 78.09 | 773 | Packers |
| 14 | 14 | Jarvis Landry | 82.11 | 84.60 | 76.29 | 572 | Dolphins |
| 15 | 15 | Pierre Garcon | 81.84 | 85.50 | 75.24 | 561 | Commanders |
| 16 | 16 | Cole Beasley | 81.52 | 85.90 | 74.44 | 460 | Cowboys |
| 17 | 17 | Adam Thielen | 81.30 | 79.90 | 78.07 | 525 | Vikings |
| 18 | 18 | DeSean Jackson | 81.23 | 73.00 | 82.55 | 520 | Commanders |
| 19 | 19 | Demaryius Thomas | 80.30 | 80.40 | 76.07 | 590 | Broncos |
| 20 | 20 | Willie Snead IV | 80.16 | 79.60 | 76.36 | 510 | Saints |

### Good (44 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Larry Fitzgerald | 79.95 | 83.30 | 73.55 | 701 | Cardinals |
| 22 | 2 | Alshon Jeffery | 79.62 | 74.50 | 78.86 | 462 | Bears |
| 23 | 3 | Sammy Watkins | 79.45 | 73.70 | 79.12 | 235 | Bills |
| 24 | 4 | Tyrell Williams | 79.41 | 75.60 | 77.78 | 599 | Chargers |
| 25 | 5 | DeAndre Hopkins | 79.38 | 77.80 | 76.27 | 731 | Texans |
| 26 | 6 | Kenny Britt | 79.20 | 74.00 | 78.50 | 529 | Rams |
| 27 | 7 | DeVante Parker | 79.02 | 75.10 | 77.46 | 499 | Dolphins |
| 28 | 8 | Kelvin Benjamin | 78.94 | 76.50 | 76.40 | 508 | Panthers |
| 29 | 9 | Stefon Diggs | 78.90 | 79.10 | 74.60 | 498 | Vikings |
| 30 | 10 | Aldrick Robinson | 78.76 | 73.90 | 77.84 | 174 | Falcons |
| 31 | 11 | Amari Cooper | 78.64 | 76.70 | 75.77 | 678 | Raiders |
| 32 | 12 | Brandin Cooks | 78.52 | 73.00 | 78.03 | 641 | Saints |
| 33 | 13 | Julian Edelman | 78.46 | 80.30 | 73.06 | 757 | Patriots |
| 34 | 14 | Tyreek Hill | 78.36 | 79.50 | 73.43 | 285 | Chiefs |
| 35 | 15 | Steve Smith | 78.22 | 74.50 | 76.54 | 492 | Ravens |
| 36 | 16 | Russell Shepard | 77.99 | 78.40 | 73.55 | 240 | Buccaneers |
| 37 | 17 | Terrance Williams | 77.46 | 73.00 | 76.26 | 509 | Cowboys |
| 38 | 18 | Marvin Jones Jr. | 77.26 | 72.50 | 76.26 | 657 | Lions |
| 39 | 19 | J.J. Nelson | 77.20 | 69.50 | 78.17 | 364 | Cardinals |
| 40 | 20 | Paul Richardson Jr. | 77.12 | 76.20 | 73.56 | 274 | Seahawks |
| 41 | 21 | Mohamed Sanu | 77.03 | 80.00 | 70.89 | 579 | Falcons |
| 42 | 22 | Cameron Meredith | 76.99 | 73.70 | 75.01 | 455 | Bears |
| 43 | 23 | John Brown | 76.82 | 71.40 | 76.27 | 426 | Cardinals |
| 44 | 24 | Kendall Wright | 76.66 | 72.10 | 75.53 | 240 | Titans |
| 45 | 25 | Terrelle Pryor Sr. | 76.39 | 74.70 | 73.35 | 629 | Browns |
| 46 | 26 | Randall Cobb | 76.33 | 74.40 | 73.45 | 601 | Packers |
| 47 | 27 | Kenny Stills | 75.97 | 68.80 | 76.59 | 536 | Dolphins |
| 48 | 28 | Geronimo Allison | 75.93 | 66.50 | 78.05 | 225 | Packers |
| 49 | 29 | Eric Decker | 75.58 | 71.50 | 74.14 | 131 | Jets |
| 50 | 30 | Michael Crabtree | 75.54 | 78.60 | 69.33 | 609 | Raiders |
| 51 | 31 | Tyler Lockett | 75.44 | 68.90 | 75.64 | 420 | Seahawks |
| 52 | 32 | Dontrelle Inman | 75.36 | 72.30 | 73.23 | 612 | Chargers |
| 53 | 33 | Ted Ginn Jr. | 75.16 | 69.20 | 74.97 | 461 | Panthers |
| 54 | 34 | Mike Wallace | 75.10 | 70.20 | 74.20 | 617 | Ravens |
| 55 | 35 | Chris Hogan | 75.03 | 65.80 | 77.02 | 637 | Patriots |
| 56 | 36 | Marqise Lee | 74.99 | 71.90 | 72.88 | 588 | Jaguars |
| 57 | 37 | Golden Tate | 74.84 | 70.60 | 73.50 | 676 | Lions |
| 58 | 38 | Allen Robinson II | 74.80 | 69.80 | 73.97 | 711 | Jaguars |
| 59 | 39 | Travis Benjamin | 74.65 | 67.20 | 75.45 | 411 | Chargers |
| 60 | 40 | Brandon Marshall | 74.62 | 69.20 | 74.06 | 580 | Jets |
| 61 | 41 | Andre Holmes | 74.39 | 74.60 | 70.08 | 160 | Raiders |
| 62 | 42 | Donte Moncrief | 74.32 | 73.40 | 70.77 | 308 | Colts |
| 63 | 43 | Davante Adams | 74.31 | 72.60 | 71.29 | 768 | Packers |
| 64 | 44 | Brandon LaFell | 74.23 | 69.50 | 73.22 | 635 | Bengals |

### Starter (73 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 65 | 1 | Quincy Enunwa | 73.54 | 70.00 | 71.74 | 565 | Jets |
| 66 | 2 | Jamison Crowder | 73.23 | 67.00 | 73.22 | 549 | Commanders |
| 67 | 3 | Deonte Thompson | 73.13 | 70.50 | 70.72 | 194 | Bears |
| 68 | 4 | Jeremy Maclin | 73.05 | 61.80 | 76.38 | 440 | Chiefs |
| 69 | 5 | Breshad Perriman | 72.94 | 64.20 | 74.60 | 348 | Ravens |
| 70 | 6 | Jaron Brown | 72.86 | 66.10 | 73.20 | 119 | Cardinals |
| 71 | 7 | Robert Woods | 72.65 | 71.40 | 69.32 | 381 | Bills |
| 72 | 8 | Jordan Matthews | 72.65 | 67.70 | 71.79 | 522 | Eagles |
| 73 | 9 | Justin Hardy | 72.56 | 75.90 | 66.16 | 217 | Falcons |
| 74 | 10 | Michael Floyd | 72.49 | 65.60 | 72.91 | 530 | Patriots |
| 75 | 11 | Brandon Coleman | 72.49 | 69.40 | 70.38 | 199 | Saints |
| 76 | 12 | Josh Bellamy | 72.11 | 70.80 | 68.81 | 184 | Bears |
| 77 | 13 | Eli Rogers | 72.07 | 65.60 | 72.22 | 499 | Steelers |
| 78 | 14 | Devin Funchess | 72.06 | 64.40 | 73.00 | 291 | Panthers |
| 79 | 15 | Adam Humphries | 71.82 | 66.00 | 71.53 | 478 | Buccaneers |
| 80 | 16 | Harry Douglas | 71.81 | 66.30 | 71.31 | 167 | Titans |
| 81 | 17 | Sammie Coates | 71.77 | 58.10 | 76.72 | 241 | Steelers |
| 82 | 18 | Jeremy Kerley | 71.35 | 68.90 | 68.81 | 520 | 49ers |
| 83 | 19 | Jalin Marshall | 71.23 | 65.70 | 70.75 | 115 | Jets |
| 84 | 20 | Cody Core | 71.19 | 60.70 | 74.02 | 138 | Bengals |
| 85 | 21 | Victor Cruz | 71.16 | 61.00 | 73.76 | 523 | Giants |
| 86 | 22 | Jordan Taylor | 70.99 | 62.10 | 72.75 | 178 | Broncos |
| 87 | 23 | Rashard Higgins | 70.95 | 61.50 | 73.08 | 112 | Browns |
| 88 | 24 | Vincent Jackson | 70.84 | 60.80 | 73.37 | 221 | Buccaneers |
| 89 | 25 | Marquise Goodwin | 70.79 | 62.30 | 72.29 | 414 | Bills |
| 90 | 26 | Danny Amendola | 70.74 | 68.70 | 67.94 | 285 | Patriots |
| 91 | 27 | Justin Hunter | 70.72 | 62.30 | 72.16 | 205 | Bills |
| 92 | 28 | Eddie Royal | 70.70 | 65.10 | 70.27 | 239 | Bears |
| 93 | 29 | Charles Johnson | 70.59 | 60.40 | 73.22 | 271 | Vikings |
| 94 | 30 | Tyler Boyd | 70.52 | 65.10 | 69.97 | 510 | Bengals |
| 95 | 31 | Kevin White | 70.29 | 66.00 | 68.98 | 138 | Bears |
| 96 | 32 | Cordarrelle Patterson | 70.25 | 64.50 | 69.92 | 385 | Vikings |
| 97 | 33 | Malcolm Mitchell | 70.11 | 65.80 | 68.81 | 408 | Patriots |
| 98 | 34 | Paul Turner | 70.05 | 61.20 | 71.79 | 112 | Eagles |
| 99 | 35 | Cobi Hamilton | 69.78 | 63.00 | 70.14 | 300 | Steelers |
| 100 | 36 | Allen Hurns | 69.73 | 57.10 | 73.98 | 454 | Jaguars |
| 101 | 37 | Anquan Boldin | 69.72 | 64.90 | 68.76 | 625 | Lions |
| 102 | 38 | Brian Quick | 69.36 | 60.50 | 71.10 | 470 | Rams |
| 103 | 39 | Bryan Walters | 69.32 | 62.90 | 69.43 | 230 | Jaguars |
| 104 | 40 | Brice Butler | 69.30 | 59.60 | 71.60 | 248 | Cowboys |
| 105 | 41 | Will Fuller V | 69.02 | 61.60 | 69.80 | 627 | Texans |
| 106 | 42 | Sterling Shepard | 68.94 | 66.20 | 66.60 | 692 | Giants |
| 107 | 43 | Kamar Aiken | 68.89 | 60.50 | 70.32 | 425 | Ravens |
| 108 | 44 | Tajae Sharpe | 68.84 | 60.80 | 70.03 | 522 | Titans |
| 109 | 45 | Chester Rogers | 68.78 | 59.20 | 71.00 | 315 | Colts |
| 110 | 46 | Rod Streater | 68.68 | 65.30 | 66.76 | 166 | 49ers |
| 111 | 47 | Corey Brown | 68.57 | 57.50 | 71.79 | 372 | Panthers |
| 112 | 48 | James Wright | 68.56 | 65.20 | 66.63 | 120 | Bengals |
| 113 | 49 | Quinton Patton | 68.52 | 62.20 | 68.57 | 417 | 49ers |
| 114 | 50 | Phillip Dorsett | 68.44 | 56.50 | 72.23 | 567 | Colts |
| 115 | 51 | Darrius Heyward-Bey | 68.25 | 55.30 | 72.72 | 182 | Steelers |
| 116 | 52 | Corey Coleman | 68.21 | 61.70 | 68.38 | 379 | Browns |
| 117 | 53 | Chris Conley | 68.10 | 61.50 | 68.34 | 568 | Chiefs |
| 118 | 54 | Dorial Green-Beckham | 68.00 | 59.90 | 69.23 | 446 | Eagles |
| 119 | 55 | Andre Roberts | 67.77 | 60.30 | 68.59 | 185 | Lions |
| 120 | 56 | Cecil Shorts | 67.65 | 60.70 | 68.11 | 181 | Buccaneers |
| 121 | 57 | Jermaine Kearse | 67.63 | 55.50 | 71.55 | 635 | Seahawks |
| 122 | 58 | Andrew Hawkins | 67.41 | 58.10 | 69.45 | 472 | Browns |
| 123 | 59 | Torrey Smith | 66.95 | 53.90 | 71.49 | 380 | 49ers |
| 124 | 60 | Brittan Golden | 66.79 | 58.00 | 68.49 | 109 | Cardinals |
| 125 | 61 | Christopher Harper | 66.21 | 60.40 | 65.92 | 133 | 49ers |
| 126 | 62 | Bennie Fowler | 65.95 | 53.20 | 70.28 | 159 | Broncos |
| 127 | 63 | Aaron Burbridge | 65.69 | 55.50 | 68.31 | 139 | 49ers |
| 128 | 64 | Tavon Austin | 65.64 | 61.30 | 64.36 | 492 | Rams |
| 129 | 65 | Cody Latimer | 65.57 | 60.50 | 64.78 | 132 | Broncos |
| 130 | 66 | Ricardo Louis | 65.36 | 57.90 | 66.16 | 222 | Browns |
| 131 | 67 | Charone Peake | 64.87 | 59.00 | 64.61 | 239 | Jets |
| 132 | 68 | Jeff Janis | 64.69 | 52.10 | 68.92 | 179 | Packers |
| 133 | 69 | Albert Wilson | 64.33 | 53.20 | 67.59 | 353 | Chiefs |
| 134 | 70 | Jaelen Strong | 63.88 | 55.10 | 65.57 | 202 | Texans |
| 135 | 71 | Seth Roberts | 63.81 | 56.10 | 64.78 | 590 | Raiders |
| 136 | 72 | Jordan Norwood | 63.31 | 51.50 | 67.02 | 342 | Broncos |
| 137 | 73 | Ryan Grant | 62.81 | 54.30 | 64.32 | 128 | Commanders |

### Rotation/backup (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 138 | 1 | Roger Lewis Jr. | 61.89 | 51.70 | 64.51 | 131 | Giants |
| 139 | 2 | Walter Powell | 61.83 | 57.50 | 60.55 | 191 | Bills |
| 140 | 3 | Nelson Agholor | 60.94 | 52.60 | 62.33 | 565 | Eagles |
| 141 | 4 | Pharoh Cooper | 59.99 | 52.80 | 60.62 | 138 | Rams |
| 142 | 5 | Braxton Miller | 58.58 | 50.80 | 59.60 | 241 | Texans |
| 143 | 6 | Chris Moore | 56.33 | 50.30 | 56.19 | 101 | Ravens |
