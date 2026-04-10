# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:38:08Z
- **Requested analysis_year:** 2017 (clamped to 2017)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jason Kelce | 97.30 | 94.10 | 95.26 | 1281 | Eagles |
| 2 | 2 | Alex Mack | 96.63 | 91.50 | 95.88 | 1158 | Falcons |
| 3 | 3 | Travis Frederick | 92.58 | 86.20 | 92.67 | 1065 | Cowboys |
| 4 | 4 | Brandon Linder | 89.93 | 82.70 | 90.58 | 1119 | Jaguars |
| 5 | 5 | David Andrews | 89.59 | 83.50 | 89.49 | 1207 | Patriots |
| 6 | 6 | Ali Marpet | 88.56 | 78.42 | 91.16 | 723 | Buccaneers |
| 7 | 7 | Rodney Hudson | 84.62 | 76.20 | 86.07 | 1007 | Raiders |
| 8 | 8 | Matt Paradis | 82.80 | 74.90 | 83.90 | 1128 | Broncos |
| 9 | 9 | Ryan Jensen | 81.74 | 72.30 | 83.86 | 1085 | Ravens |
| 10 | 10 | Weston Richburg | 81.65 | 68.85 | 86.01 | 241 | Giants |
| 11 | 11 | Maurkice Pouncey | 81.64 | 72.80 | 83.37 | 1114 | Steelers |
| 12 | 12 | Ben Jones | 81.24 | 71.60 | 83.50 | 1153 | Titans |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Eric Wood | 79.83 | 70.90 | 81.62 | 1130 | Bills |
| 14 | 2 | John Sullivan | 79.57 | 70.31 | 81.58 | 926 | Rams |
| 15 | 3 | Brett Jones | 79.14 | 70.10 | 81.00 | 966 | Giants |
| 16 | 4 | J.C. Tretter | 77.56 | 69.10 | 79.03 | 1068 | Browns |
| 17 | 5 | Corey Linsley | 77.21 | 67.70 | 79.38 | 1047 | Packers |
| 18 | 6 | Daniel Kilgore | 77.00 | 66.80 | 79.64 | 1098 | 49ers |
| 19 | 7 | Russell Bodine | 76.87 | 66.72 | 79.47 | 962 | Bengals |
| 20 | 8 | Justin Britt | 76.84 | 65.50 | 80.23 | 1062 | Seahawks |
| 21 | 9 | Pat Elflein | 76.25 | 66.60 | 78.52 | 1081 | Vikings |
| 22 | 10 | Tyler Larsen | 75.82 | 66.16 | 78.10 | 720 | Panthers |
| 23 | 11 | Nick Martin | 74.61 | 64.77 | 77.01 | 971 | Texans |
| 24 | 12 | Mike Pouncey | 74.45 | 62.09 | 78.53 | 971 | Dolphins |

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 25 | 1 | Max Unger | 73.70 | 63.10 | 76.60 | 1164 | Saints |
| 26 | 2 | Mitch Morse | 73.26 | 61.47 | 76.95 | 383 | Chiefs |
| 27 | 3 | A.Q. Shipley | 72.46 | 61.30 | 75.73 | 1120 | Cardinals |
| 28 | 4 | Spencer Long | 72.31 | 60.76 | 75.84 | 397 | Commanders |
| 29 | 5 | Chase Roullier | 71.28 | 63.00 | 72.64 | 457 | Commanders |
| 30 | 6 | B.J. Finney | 70.73 | 63.13 | 71.63 | 235 | Steelers |
| 31 | 7 | Travis Swanson | 70.14 | 58.45 | 73.77 | 710 | Lions |
| 32 | 8 | Joe Hawley | 69.04 | 57.64 | 72.48 | 203 | Buccaneers |
| 33 | 9 | Ryan Kelly | 68.86 | 60.31 | 70.39 | 394 | Colts |
| 34 | 10 | Ryan Kalil | 68.35 | 57.40 | 71.49 | 412 | Panthers |
| 35 | 11 | Spencer Pulley | 66.25 | 53.60 | 70.52 | 1054 | Chargers |
| 36 | 12 | Wesley Johnson | 65.97 | 51.90 | 71.18 | 938 | Jets |
| 37 | 13 | Austin Blythe | 65.70 | 56.38 | 67.75 | 197 | Rams |
| 38 | 14 | Jonotthan Harrison | 62.50 | 50.88 | 66.08 | 102 | Jets |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Hroniss Grasu | 61.87 | 48.05 | 66.91 | 259 | Bears |

## CB — Cornerback

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Casey Hayward Jr. | 93.79 | 92.10 | 90.75 | 1003 | Chargers |
| 2 | 2 | Tre'Davious White | 93.72 | 90.10 | 91.97 | 1149 | Bills |
| 3 | 3 | Jalen Ramsey | 93.36 | 91.60 | 90.37 | 1212 | Jaguars |
| 4 | 4 | Marshon Lattimore | 93.18 | 87.90 | 93.57 | 900 | Saints |
| 5 | 5 | William Jackson III | 92.27 | 87.96 | 93.06 | 698 | Bengals |
| 6 | 6 | Kendall Fuller | 89.58 | 89.15 | 87.27 | 720 | Commanders |
| 7 | 7 | Stephon Gilmore | 89.58 | 87.20 | 88.15 | 1027 | Patriots |
| 8 | 8 | Patrick Robinson | 89.42 | 89.90 | 87.75 | 853 | Eagles |
| 9 | 9 | A.J. Bouye | 88.28 | 84.70 | 87.14 | 1229 | Jaguars |
| 10 | 10 | Marcus Peters | 87.69 | 81.70 | 88.03 | 1034 | Chiefs |
| 11 | 11 | Aqib Talib | 86.16 | 81.02 | 86.88 | 753 | Broncos |
| 12 | 12 | Tramon Williams | 86.15 | 82.35 | 87.54 | 666 | Cardinals |
| 13 | 13 | Nickell Robey-Coleman | 85.20 | 81.97 | 83.19 | 676 | Rams |
| 14 | 14 | Darius Slay | 85.01 | 80.00 | 84.81 | 1064 | Lions |
| 15 | 15 | Jimmy Smith | 84.89 | 81.98 | 86.32 | 601 | Ravens |
| 16 | 16 | Trevor Williams | 83.79 | 80.20 | 84.75 | 1004 | Chargers |
| 17 | 17 | Bradley Roby | 82.96 | 76.49 | 83.10 | 674 | Broncos |
| 18 | 18 | Mike Hilton | 82.39 | 78.55 | 80.78 | 592 | Steelers |
| 19 | 19 | Kyle Fuller | 82.14 | 79.00 | 80.07 | 1017 | Bears |
| 20 | 20 | Chris Harris Jr. | 82.09 | 75.80 | 82.11 | 869 | Broncos |
| 21 | 21 | E.J. Gaines | 81.86 | 79.12 | 83.37 | 711 | Bills |
| 22 | 22 | Robert Alford | 81.53 | 74.50 | 82.25 | 1171 | Falcons |
| 23 | 23 | Adoree' Jackson | 81.47 | 75.50 | 81.29 | 1154 | Titans |
| 24 | 24 | Briean Boddy-Calhoun | 81.45 | 74.09 | 84.93 | 535 | Browns |
| 25 | 25 | Ronald Darby | 81.07 | 76.35 | 83.48 | 580 | Eagles |
| 26 | 26 | Richard Sherman | 81.01 | 72.68 | 86.05 | 572 | Seahawks |
| 27 | 27 | T.J. Carrie | 80.65 | 75.00 | 80.76 | 1023 | Raiders |
| 28 | 28 | Marlon Humphrey | 80.57 | 73.02 | 81.44 | 597 | Ravens |
| 29 | 29 | Brent Grimes | 80.53 | 75.20 | 81.68 | 844 | Buccaneers |
| 30 | 30 | Bobby McCain | 80.20 | 76.96 | 79.45 | 664 | Dolphins |
| 31 | 31 | Quinton Dunbar | 80.18 | 72.98 | 84.57 | 373 | Commanders |

### Good (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 32 | 1 | Desmond Trufant | 79.98 | 72.80 | 82.78 | 1030 | Falcons |
| 33 | 2 | Artie Burns | 79.92 | 73.60 | 79.96 | 1028 | Steelers |
| 34 | 3 | Ken Crawley | 79.54 | 74.10 | 80.81 | 967 | Saints |
| 35 | 4 | Chidobe Awuzie | 79.20 | 71.57 | 87.42 | 309 | Cowboys |
| 36 | 5 | Malcolm Butler | 79.03 | 69.30 | 81.35 | 1175 | Patriots |
| 37 | 6 | Ross Cockrell | 78.90 | 73.31 | 79.49 | 679 | Giants |
| 38 | 7 | Xavier Rhodes | 78.89 | 71.00 | 80.62 | 1031 | Vikings |
| 39 | 8 | Jason McCourty | 78.50 | 73.50 | 81.83 | 899 | Browns |
| 40 | 9 | Rashaan Melvin | 78.31 | 73.01 | 83.73 | 552 | Colts |
| 41 | 10 | Josh Norman | 78.22 | 69.30 | 81.03 | 902 | Commanders |
| 42 | 11 | Justin Coleman | 77.95 | 74.33 | 80.57 | 654 | Seahawks |
| 43 | 12 | Bashaud Breeland | 77.73 | 72.00 | 78.53 | 856 | Commanders |
| 44 | 13 | Jourdan Lewis | 77.68 | 69.70 | 79.87 | 746 | Cowboys |
| 45 | 14 | Johnathan Joseph | 77.44 | 70.68 | 78.10 | 746 | Texans |
| 46 | 15 | Ahkello Witherspoon | 77.42 | 72.25 | 80.86 | 660 | 49ers |
| 47 | 16 | K'Waun Williams | 76.90 | 72.08 | 78.54 | 632 | 49ers |
| 48 | 17 | Darqueze Dennard | 76.69 | 72.40 | 76.95 | 900 | Bengals |
| 49 | 18 | Prince Amukamara | 76.66 | 71.50 | 78.63 | 849 | Bears |
| 50 | 19 | Trumaine Johnson | 76.61 | 68.40 | 78.95 | 1005 | Rams |
| 51 | 20 | Bryce Callahan | 76.49 | 73.63 | 79.54 | 512 | Bears |
| 52 | 21 | Dominique Rodgers-Cromartie | 76.03 | 68.22 | 77.81 | 604 | Giants |
| 53 | 22 | Patrick Peterson | 75.83 | 68.40 | 76.61 | 1013 | Cardinals |
| 54 | 23 | Cre'Von LeBlanc | 75.46 | 66.59 | 83.59 | 212 | Bears |
| 55 | 24 | Aaron Colvin | 75.03 | 74.40 | 73.16 | 833 | Jaguars |
| 56 | 25 | Byron Maxwell | 75.00 | 68.85 | 79.94 | 580 | Seahawks |
| 57 | 26 | Joe Haden | 74.57 | 69.89 | 78.84 | 673 | Steelers |
| 58 | 27 | Brandon Carr | 74.55 | 66.20 | 75.95 | 1023 | Ravens |
| 59 | 28 | Shaquill Griffin | 74.43 | 63.00 | 78.92 | 876 | Seahawks |

### Starter (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 60 | 1 | Janoris Jenkins | 73.70 | 68.83 | 76.63 | 619 | Giants |
| 61 | 2 | Sean Smith | 72.68 | 63.72 | 75.73 | 701 | Raiders |
| 62 | 3 | Logan Ryan | 72.67 | 62.00 | 75.61 | 1048 | Titans |
| 63 | 4 | Darryl Roberts | 72.44 | 65.05 | 78.28 | 468 | Jets |
| 64 | 5 | Terence Newman | 72.32 | 63.57 | 74.30 | 611 | Vikings |
| 65 | 6 | Kenny Moore II | 72.19 | 66.34 | 80.26 | 384 | Colts |
| 66 | 7 | Jonathan Jones | 72.17 | 66.03 | 76.64 | 446 | Patriots |
| 67 | 8 | Rasul Douglas | 72.08 | 65.16 | 76.69 | 424 | Eagles |
| 68 | 9 | Xavien Howard | 71.43 | 63.60 | 75.61 | 1016 | Dolphins |
| 69 | 10 | Troy Hill | 70.80 | 68.38 | 75.95 | 338 | Rams |
| 70 | 11 | Trae Waynes | 70.40 | 61.80 | 73.32 | 1044 | Vikings |
| 71 | 12 | Steven Nelson | 70.33 | 66.45 | 73.53 | 579 | Chiefs |
| 72 | 13 | Davon House | 70.18 | 60.00 | 75.81 | 658 | Packers |
| 73 | 14 | Robert McClain | 69.95 | 65.31 | 73.66 | 690 | Buccaneers |
| 74 | 15 | William Gay | 69.89 | 62.01 | 70.98 | 271 | Steelers |
| 75 | 16 | Anthony Brown | 69.88 | 58.70 | 73.16 | 846 | Cowboys |
| 76 | 17 | Dre Kirkpatrick | 69.87 | 61.80 | 72.44 | 868 | Bengals |
| 77 | 18 | Terrance Mitchell | 69.60 | 61.02 | 77.61 | 705 | Chiefs |
| 78 | 19 | Buster Skrine | 69.56 | 61.40 | 71.98 | 1010 | Jets |
| 79 | 20 | Quincy Wilson | 69.43 | 65.79 | 78.56 | 402 | Colts |
| 80 | 21 | Kayvon Webster | 68.87 | 61.88 | 74.05 | 550 | Rams |
| 81 | 22 | Daryl Worley | 68.84 | 59.22 | 71.47 | 777 | Panthers |
| 82 | 23 | Shareece Wright | 68.80 | 64.61 | 71.79 | 456 | Bills |
| 83 | 24 | P.J. Williams | 68.59 | 61.89 | 74.36 | 740 | Saints |
| 84 | 25 | Captain Munnerlyn | 68.16 | 60.33 | 70.05 | 404 | Panthers |
| 85 | 26 | Marcus Williams | 67.71 | 58.76 | 73.47 | 188 | Texans |
| 86 | 27 | Kareem Jackson | 67.62 | 57.90 | 70.57 | 868 | Texans |
| 87 | 28 | Darrelle Revis | 67.49 | 59.57 | 74.53 | 238 | Chiefs |
| 88 | 29 | Eli Apple | 67.24 | 59.11 | 72.79 | 649 | Giants |
| 89 | 30 | Vernon Hargreaves III | 67.22 | 63.59 | 70.02 | 502 | Buccaneers |
| 90 | 31 | Pierre Desir | 67.14 | 59.39 | 76.27 | 375 | Colts |
| 91 | 32 | LeShaun Sims | 66.75 | 63.46 | 70.24 | 428 | Titans |
| 92 | 33 | D.J. Hayden | 66.51 | 58.77 | 69.07 | 489 | Lions |
| 93 | 34 | Leonard Johnson | 66.19 | 60.74 | 70.02 | 705 | Bills |
| 94 | 35 | Nate Hairston | 66.13 | 57.90 | 69.54 | 537 | Colts |
| 95 | 36 | Tye Smith | 66.02 | 62.16 | 69.62 | 228 | Titans |
| 96 | 37 | Morris Claiborne | 65.94 | 58.30 | 70.93 | 919 | Jets |
| 97 | 38 | Dexter McDonald | 65.85 | 62.41 | 71.80 | 534 | Raiders |
| 98 | 39 | Eric Rowe | 65.76 | 57.70 | 72.07 | 427 | Patriots |
| 99 | 40 | Adam Jones | 65.50 | 57.64 | 70.42 | 299 | Bengals |
| 100 | 41 | Kenneth Acker | 65.35 | 58.54 | 71.14 | 210 | Chiefs |
| 101 | 42 | James Bradberry | 65.34 | 55.30 | 69.03 | 1044 | Panthers |
| 102 | 43 | Mike Jordan | 65.14 | 61.31 | 72.77 | 210 | Browns |
| 103 | 44 | Jamar Taylor | 64.98 | 62.70 | 64.00 | 966 | Browns |
| 104 | 45 | Lardarius Webb | 64.36 | 52.76 | 68.12 | 374 | Ravens |
| 105 | 46 | Justin Bethel | 64.19 | 53.23 | 70.88 | 456 | Cardinals |
| 106 | 47 | Maurice Canady | 64.19 | 61.43 | 67.06 | 320 | Ravens |
| 107 | 48 | Darryl Morris | 63.80 | 59.96 | 72.00 | 114 | Giants |
| 108 | 49 | Mackensie Alexander | 63.74 | 55.42 | 67.85 | 385 | Vikings |
| 109 | 50 | Kevon Seymour | 63.64 | 51.82 | 69.06 | 317 | Panthers |
| 110 | 51 | Johnson Bademosi | 63.00 | 58.33 | 70.91 | 224 | Patriots |
| 111 | 52 | Alterraun Verner | 62.66 | 56.49 | 70.00 | 157 | Dolphins |
| 112 | 53 | Cordrea Tankersley | 62.57 | 59.47 | 65.67 | 638 | Dolphins |
| 113 | 54 | Nevin Lawson | 62.14 | 53.36 | 65.39 | 555 | Lions |

### Rotation/backup (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 114 | 1 | Sterling Moore | 61.68 | 56.28 | 67.27 | 108 | Saints |
| 115 | 2 | Rashard Robinson | 61.55 | 49.81 | 69.90 | 499 | Jets |
| 116 | 3 | Quinten Rollins | 61.53 | 56.18 | 67.18 | 139 | Packers |
| 117 | 4 | Leon Hall | 61.49 | 53.99 | 67.10 | 205 | 49ers |
| 118 | 5 | Brice McCain | 61.37 | 49.43 | 66.09 | 416 | Titans |
| 119 | 6 | Coty Sensabaugh | 60.87 | 55.94 | 67.08 | 242 | Steelers |
| 120 | 7 | Dontae Johnson | 60.63 | 48.90 | 66.37 | 1026 | 49ers |
| 121 | 8 | Lenzy Pipkins | 60.55 | 60.32 | 72.51 | 122 | Packers |
| 122 | 9 | Marcus Cooper | 60.50 | 49.90 | 67.47 | 246 | Bears |
| 123 | 10 | Jeremy Lane | 59.89 | 47.63 | 67.65 | 346 | Seahawks |
| 124 | 11 | Ryan Smith | 59.73 | 58.98 | 58.66 | 598 | Buccaneers |
| 125 | 12 | Keith Reaser | 59.00 | 58.28 | 67.81 | 100 | Chiefs |
| 126 | 13 | Orlando Scandrick | 58.96 | 49.81 | 64.44 | 614 | Cowboys |
| 127 | 14 | Kevin King | 58.91 | 51.45 | 67.02 | 380 | Packers |
| 128 | 15 | David Amerson | 58.83 | 45.22 | 69.16 | 287 | Raiders |
| 129 | 16 | Javien Elliott | 58.48 | 61.52 | 60.49 | 130 | Buccaneers |
| 130 | 17 | Juston Burris | 58.11 | 52.24 | 64.89 | 334 | Jets |
| 131 | 18 | Josh Hawkins | 57.78 | 53.51 | 62.20 | 402 | Packers |
| 132 | 19 | Phillip Gaines | 55.89 | 46.30 | 65.51 | 420 | Chiefs |
| 133 | 20 | Brandon Dixon | 55.06 | 53.13 | 66.58 | 257 | Giants |
| 134 | 21 | Kevin Johnson | 53.26 | 40.00 | 63.14 | 579 | Texans |
| 135 | 22 | Cameron Sutton | 52.15 | 59.20 | 61.79 | 113 | Steelers |
| 136 | 23 | Brendan Langley | 45.00 | 46.80 | 47.85 | 107 | Broncos |

## DI — Defensive Interior

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 93.61 | 89.92 | 92.42 | 856 | Rams |
| 2 | 2 | Kawann Short | 88.82 | 86.32 | 86.32 | 743 | Panthers |
| 3 | 3 | Geno Atkins | 88.27 | 85.84 | 85.73 | 754 | Bengals |
| 4 | 4 | J.J. Watt | 86.97 | 79.87 | 97.33 | 217 | Texans |
| 5 | 5 | Jurrell Casey | 86.82 | 88.40 | 81.91 | 988 | Titans |
| 6 | 6 | Leonard Williams | 86.32 | 88.58 | 80.65 | 877 | Jets |
| 7 | 7 | Ndamukong Suh | 85.58 | 86.61 | 80.72 | 877 | Dolphins |
| 8 | 8 | Malik Jackson | 85.47 | 84.27 | 82.11 | 922 | Jaguars |
| 9 | 9 | Fletcher Cox | 84.67 | 88.37 | 78.04 | 783 | Eagles |
| 10 | 10 | Damon Harrison Sr. | 84.42 | 82.59 | 81.47 | 644 | Giants |
| 11 | 11 | Mike Daniels | 83.89 | 80.70 | 82.89 | 629 | Packers |
| 12 | 12 | Akiem Hicks | 83.22 | 81.12 | 80.46 | 900 | Bears |
| 13 | 13 | Michael Pierce | 83.22 | 80.28 | 81.01 | 594 | Ravens |
| 14 | 14 | Linval Joseph | 83.21 | 85.25 | 78.31 | 759 | Vikings |
| 15 | 15 | Kenny Clark | 83.13 | 84.63 | 78.61 | 684 | Packers |
| 16 | 16 | DeForest Buckner | 82.66 | 88.08 | 75.26 | 867 | 49ers |
| 17 | 17 | Olsen Pierre | 82.56 | 61.71 | 94.38 | 351 | Cardinals |
| 18 | 18 | Gerald McCoy | 81.74 | 86.95 | 75.14 | 807 | Buccaneers |
| 19 | 19 | Marcell Dareus | 81.62 | 81.73 | 80.08 | 540 | Jaguars |
| 20 | 20 | Cameron Heyward | 81.58 | 83.10 | 79.21 | 835 | Steelers |
| 21 | 21 | Larry Ogunjobi | 81.27 | 73.79 | 84.17 | 300 | Browns |
| 22 | 22 | Grady Jarrett | 80.23 | 73.75 | 80.58 | 896 | Falcons |
| 23 | 23 | Malcom Brown | 80.15 | 76.90 | 78.15 | 682 | Patriots |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | David Irving | 78.77 | 73.97 | 82.80 | 338 | Cowboys |
| 25 | 2 | Vincent Taylor | 78.19 | 71.59 | 82.59 | 185 | Dolphins |
| 26 | 3 | Johnathan Hankins | 78.19 | 76.59 | 77.07 | 687 | Colts |
| 27 | 4 | Michael Brockers | 78.14 | 79.51 | 73.70 | 751 | Rams |
| 28 | 5 | Muhammad Wilkerson | 77.96 | 67.97 | 82.64 | 698 | Jets |
| 29 | 6 | Stephon Tuitt | 77.56 | 78.48 | 74.34 | 627 | Steelers |
| 30 | 7 | Sheldon Richardson | 77.25 | 68.43 | 81.04 | 654 | Seahawks |
| 31 | 8 | Dean Lowry | 77.15 | 69.80 | 77.88 | 493 | Packers |
| 32 | 9 | DJ Reader | 77.13 | 76.36 | 74.77 | 526 | Texans |
| 33 | 10 | Brandon Williams | 77.00 | 72.18 | 78.13 | 475 | Ravens |
| 34 | 11 | Timmy Jernigan | 76.57 | 68.24 | 78.15 | 586 | Eagles |
| 35 | 12 | Danny Shelton | 76.33 | 75.31 | 73.88 | 469 | Browns |
| 36 | 13 | Steve McLendon | 75.53 | 65.79 | 79.43 | 488 | Jets |
| 37 | 14 | Bennie Logan | 74.95 | 62.00 | 80.77 | 617 | Chiefs |
| 38 | 15 | Javon Hargrave | 74.85 | 69.25 | 74.42 | 478 | Steelers |
| 39 | 16 | Abry Jones | 74.54 | 68.11 | 75.18 | 579 | Jaguars |
| 40 | 17 | Eddie Goldman | 74.50 | 71.60 | 76.11 | 608 | Bears |
| 41 | 18 | Lawrence Guy Sr. | 74.20 | 62.06 | 78.12 | 715 | Patriots |

### Starter (90 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 42 | 1 | Shelby Harris | 73.79 | 70.72 | 77.61 | 516 | Broncos |
| 43 | 2 | Derek Wolfe | 73.56 | 67.81 | 76.66 | 458 | Broncos |
| 44 | 3 | Henry Anderson | 73.33 | 71.82 | 76.83 | 380 | Colts |
| 45 | 4 | Corey Liuget | 73.33 | 69.97 | 74.53 | 415 | Chargers |
| 46 | 5 | Chris Baker | 73.16 | 59.89 | 78.35 | 455 | Buccaneers |
| 47 | 6 | Dalvin Tomlinson | 73.09 | 75.09 | 67.59 | 588 | Giants |
| 48 | 7 | Kyle Williams | 72.82 | 60.64 | 79.17 | 801 | Bills |
| 49 | 8 | Sheldon Day | 72.66 | 63.22 | 77.38 | 320 | 49ers |
| 50 | 9 | Jonathan Allen | 72.02 | 71.07 | 84.46 | 159 | Commanders |
| 51 | 10 | Mike Pennel | 71.47 | 59.34 | 75.59 | 301 | Jets |
| 52 | 11 | Karl Klug | 70.94 | 59.23 | 75.22 | 363 | Titans |
| 53 | 12 | A'Shawn Robinson | 70.91 | 63.83 | 71.47 | 735 | Lions |
| 54 | 13 | Caleb Brantley | 70.86 | 59.02 | 78.76 | 217 | Browns |
| 55 | 14 | Al Woods | 70.84 | 62.16 | 74.13 | 564 | Colts |
| 56 | 15 | Zach Kerr | 70.82 | 62.35 | 76.99 | 249 | Broncos |
| 57 | 16 | Stacy McGee | 70.77 | 67.26 | 70.83 | 432 | Commanders |
| 58 | 17 | Datone Jones | 70.48 | 59.04 | 78.84 | 204 | Cowboys |
| 59 | 18 | Denico Autry | 70.46 | 59.08 | 74.30 | 594 | Raiders |
| 60 | 19 | Dontari Poe | 70.33 | 68.02 | 67.71 | 880 | Falcons |
| 61 | 20 | Eli Ankou | 70.27 | 59.86 | 80.35 | 172 | Jaguars |
| 62 | 21 | DaQuan Jones | 70.16 | 67.89 | 69.59 | 436 | Titans |
| 63 | 22 | Mario Edwards Jr. | 70.06 | 63.30 | 75.92 | 475 | Raiders |
| 64 | 23 | Hassan Ridgeway | 69.84 | 61.57 | 73.13 | 177 | Colts |
| 65 | 24 | Justin Ellis | 69.80 | 67.93 | 67.71 | 462 | Raiders |
| 66 | 25 | Jarran Reed | 69.56 | 61.83 | 71.20 | 616 | Seahawks |
| 67 | 26 | Sylvester Williams | 69.54 | 64.18 | 68.95 | 410 | Titans |
| 68 | 27 | Christian Covington | 69.44 | 66.96 | 71.61 | 166 | Texans |
| 69 | 28 | Beau Allen | 69.36 | 57.58 | 73.05 | 496 | Eagles |
| 70 | 29 | Ricky Jean Francois | 69.30 | 58.15 | 73.09 | 206 | Patriots |
| 71 | 30 | Star Lotulelei | 69.20 | 57.22 | 73.02 | 621 | Panthers |
| 72 | 31 | Jonathan Bullard | 68.49 | 58.62 | 71.69 | 437 | Bears |
| 73 | 32 | Rodney Gunter | 68.25 | 59.82 | 69.71 | 291 | Cardinals |
| 74 | 33 | Adolphus Washington | 67.98 | 55.03 | 72.83 | 535 | Bills |
| 75 | 34 | Roy Robertson-Harris | 67.96 | 57.92 | 74.65 | 212 | Bears |
| 76 | 35 | Vernon Butler | 67.92 | 63.09 | 69.98 | 312 | Panthers |
| 77 | 36 | Brent Urban | 67.89 | 61.14 | 77.08 | 123 | Ravens |
| 78 | 37 | Carl Davis Jr. | 67.81 | 58.00 | 72.00 | 300 | Ravens |
| 79 | 38 | Adam Gotsis | 67.59 | 56.91 | 70.54 | 555 | Broncos |
| 80 | 39 | Xavier Williams | 67.58 | 62.71 | 74.26 | 249 | Cardinals |
| 81 | 40 | Corey Peters | 67.53 | 60.98 | 70.33 | 442 | Cardinals |
| 82 | 41 | Sheldon Rankins | 67.44 | 65.27 | 67.46 | 928 | Saints |
| 83 | 42 | David Onyemata | 67.23 | 62.08 | 66.50 | 696 | Saints |
| 84 | 43 | Ahtyba Rubin | 67.11 | 57.13 | 71.16 | 174 | Falcons |
| 85 | 44 | Carlos Watkins | 66.85 | 57.64 | 72.99 | 328 | Texans |
| 86 | 45 | Alan Branch | 66.82 | 53.68 | 73.50 | 274 | Patriots |
| 87 | 46 | Tyson Alualu | 66.76 | 59.01 | 68.40 | 451 | Steelers |
| 88 | 47 | Ethan Westbrooks | 66.69 | 52.79 | 75.69 | 359 | Rams |
| 89 | 48 | Willie Henry | 66.64 | 53.98 | 73.00 | 596 | Ravens |
| 90 | 49 | Domata Peko Sr. | 66.54 | 55.47 | 70.78 | 460 | Broncos |
| 91 | 50 | Tyrunn Walker | 66.50 | 53.79 | 73.31 | 333 | Rams |
| 92 | 51 | Tony McDaniel | 66.48 | 56.88 | 74.35 | 100 | Saints |
| 93 | 52 | Austin Johnson | 66.47 | 61.79 | 67.77 | 391 | Titans |
| 94 | 53 | Davon Godchaux | 66.45 | 53.00 | 72.29 | 500 | Dolphins |
| 95 | 54 | Mitch Unrein | 66.39 | 60.75 | 69.22 | 389 | Bears |
| 96 | 55 | Pat Sims | 66.36 | 51.96 | 74.29 | 304 | Bengals |
| 97 | 56 | Cedric Thornton | 66.33 | 49.18 | 74.85 | 404 | Bills |
| 98 | 57 | Jamie Meder | 66.19 | 59.46 | 69.64 | 178 | Browns |
| 99 | 58 | Haloti Ngata | 66.10 | 59.65 | 73.00 | 145 | Lions |
| 100 | 59 | Margus Hunt | 66.04 | 50.76 | 72.06 | 578 | Colts |
| 101 | 60 | Akeem Spence | 65.98 | 52.99 | 70.47 | 662 | Lions |
| 102 | 61 | Rakeem Nunez-Roches | 65.93 | 53.52 | 73.58 | 390 | Chiefs |
| 103 | 62 | Frostee Rucker | 65.83 | 45.75 | 76.20 | 607 | Cardinals |
| 104 | 63 | Tom Johnson | 65.82 | 49.34 | 73.28 | 786 | Vikings |
| 105 | 64 | Darius Philon | 65.77 | 56.03 | 70.38 | 509 | Chargers |
| 106 | 65 | Clinton McDonald | 65.51 | 50.23 | 75.90 | 460 | Buccaneers |
| 107 | 66 | Sealver Siliga | 65.50 | 55.46 | 74.28 | 118 | Buccaneers |
| 108 | 67 | Brandon Mebane | 65.26 | 50.92 | 72.54 | 535 | Chargers |
| 109 | 68 | John Jenkins | 65.15 | 60.42 | 70.58 | 109 | Bears |
| 110 | 69 | Ryan Glasgow | 65.14 | 57.85 | 65.83 | 412 | Bengals |
| 111 | 70 | Quinton Dial | 64.99 | 54.51 | 70.21 | 309 | Packers |
| 112 | 71 | Adam Butler | 64.95 | 52.90 | 68.81 | 524 | Patriots |
| 113 | 72 | Earl Mitchell | 64.33 | 50.74 | 71.93 | 622 | 49ers |
| 114 | 73 | Xavier Cooper | 64.25 | 57.21 | 67.70 | 305 | Jets |
| 115 | 74 | Jordan Phillips | 64.20 | 57.01 | 66.60 | 401 | Dolphins |
| 116 | 75 | Allen Bailey | 64.14 | 53.08 | 71.72 | 683 | Chiefs |
| 117 | 76 | D.J. Jones | 64.12 | 56.51 | 72.32 | 147 | 49ers |
| 118 | 77 | Grover Stewart | 63.92 | 58.73 | 64.24 | 258 | Colts |
| 119 | 78 | Chris Jones | 63.40 | 45.49 | 72.11 | 698 | Chiefs |
| 120 | 79 | Nazair Jones | 63.32 | 59.44 | 66.94 | 284 | Seahawks |
| 121 | 80 | Jay Bromley | 63.27 | 56.05 | 63.92 | 424 | Giants |
| 122 | 81 | Treyvon Hester | 63.24 | 60.38 | 63.07 | 346 | Raiders |
| 123 | 82 | Brandon Dunn | 63.08 | 59.89 | 64.69 | 416 | Texans |
| 124 | 83 | Matt Ioannidis | 62.94 | 60.03 | 64.74 | 584 | Commanders |
| 125 | 84 | Tyeler Davison | 62.86 | 58.58 | 61.86 | 666 | Saints |
| 126 | 85 | Jarvis Jenkins | 62.72 | 54.43 | 66.37 | 249 | Chiefs |
| 127 | 86 | Angelo Blackson | 62.72 | 57.49 | 66.62 | 193 | Texans |
| 128 | 87 | John Hughes | 62.45 | 56.84 | 69.32 | 158 | Saints |
| 129 | 88 | Robert Thomas | 62.15 | 53.33 | 67.24 | 236 | Giants |
| 130 | 89 | Stephen Paea | 62.03 | 51.44 | 73.15 | 145 | Cowboys |
| 131 | 90 | Trevon Coley | 62.02 | 52.35 | 65.34 | 656 | Browns |

### Rotation/backup (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 132 | 1 | Eddie Vanderdoes | 61.74 | 51.69 | 64.28 | 464 | Raiders |
| 133 | 2 | Richard Ash | 61.51 | 53.34 | 73.43 | 233 | Cowboys |
| 134 | 3 | Shamar Stephen | 61.47 | 55.76 | 63.39 | 388 | Vikings |
| 135 | 4 | Kyle Love | 61.26 | 47.14 | 68.39 | 398 | Panthers |
| 136 | 5 | Terrell McClain | 61.03 | 51.03 | 68.53 | 328 | Commanders |
| 137 | 6 | Destiny Vaeao | 60.91 | 51.49 | 65.63 | 242 | Eagles |
| 138 | 7 | Maliek Collins | 60.76 | 48.69 | 64.64 | 684 | Cowboys |
| 139 | 8 | Jack Crawford | 60.63 | 51.34 | 70.47 | 101 | Falcons |
| 140 | 9 | Brian Price | 60.44 | 55.18 | 66.54 | 150 | Cowboys |
| 141 | 10 | A.J. Francis | 60.42 | 57.20 | 69.99 | 164 | Commanders |
| 142 | 11 | Joel Heath | 60.30 | 52.42 | 64.12 | 323 | Texans |
| 143 | 12 | Josh Mauro | 60.16 | 50.48 | 68.32 | 334 | Cardinals |
| 144 | 13 | Anthony Lanier II | 59.90 | 53.69 | 67.82 | 339 | Commanders |
| 145 | 14 | David King | 59.00 | 55.40 | 67.65 | 121 | Titans |
| 146 | 15 | Damion Square | 58.88 | 54.93 | 61.00 | 362 | Chargers |
| 147 | 16 | Jeremiah Ledbetter | 58.73 | 53.97 | 57.73 | 349 | Lions |
| 148 | 17 | Tanzel Smart | 58.34 | 49.59 | 60.00 | 319 | Rams |
| 149 | 18 | Andrew Billings | 57.58 | 48.34 | 60.60 | 334 | Bengals |
| 150 | 19 | Garrison Smith | 56.94 | 55.86 | 63.77 | 115 | Seahawks |
| 151 | 20 | Robert Nkemdiche | 56.74 | 56.45 | 59.66 | 252 | Cardinals |
| 152 | 21 | Jihad Ward | 56.60 | 55.36 | 60.42 | 125 | Raiders |
| 153 | 22 | Quinton Jefferson | 56.52 | 55.86 | 64.38 | 129 | Seahawks |
| 154 | 23 | Christian Ringo | 55.68 | 57.30 | 59.69 | 130 | Lions |
| 155 | 24 | Justin Hamilton | 52.59 | 54.26 | 63.27 | 100 | Chiefs |
| 156 | 25 | Chris Wormley | 51.36 | 56.24 | 54.80 | 120 | Ravens |
| 157 | 26 | Elijah Qualls | 49.13 | 55.82 | 53.92 | 103 | Eagles |

## ED — Edge

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 94.43 | 95.25 | 89.72 | 846 | Broncos |
| 2 | 2 | Joey Bosa | 92.73 | 96.81 | 87.41 | 850 | Chargers |
| 3 | 3 | Khalil Mack | 91.58 | 97.27 | 83.62 | 930 | Raiders |
| 4 | 4 | Brandon Graham | 88.03 | 92.18 | 81.09 | 823 | Eagles |
| 5 | 5 | Justin Houston | 87.56 | 85.65 | 88.42 | 1015 | Chiefs |
| 6 | 6 | DeMarcus Lawrence | 85.07 | 93.34 | 77.27 | 701 | Cowboys |
| 7 | 7 | Danielle Hunter | 84.85 | 85.37 | 80.53 | 873 | Vikings |
| 8 | 8 | Melvin Ingram III | 83.50 | 84.26 | 78.82 | 890 | Chargers |
| 9 | 9 | Cameron Jordan | 83.38 | 93.13 | 72.71 | 1136 | Saints |
| 10 | 10 | Chandler Jones | 83.30 | 85.94 | 77.38 | 1043 | Cardinals |
| 11 | 11 | Ezekiel Ansah | 82.71 | 78.82 | 82.81 | 516 | Lions |
| 12 | 12 | Ryan Kerrigan | 82.14 | 74.60 | 83.00 | 820 | Commanders |
| 13 | 13 | Calais Campbell | 81.69 | 67.41 | 87.05 | 984 | Jaguars |
| 14 | 14 | Cameron Wake | 81.46 | 70.48 | 86.50 | 610 | Dolphins |
| 15 | 15 | Myles Garrett | 81.06 | 85.27 | 79.29 | 518 | Browns |
| 16 | 16 | Michael Bennett | 80.76 | 83.54 | 75.68 | 931 | Seahawks |
| 17 | 17 | Carlos Dunlap | 80.20 | 78.59 | 77.10 | 876 | Bengals |
| 18 | 18 | Jadeveon Clowney | 80.09 | 93.15 | 67.85 | 895 | Texans |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | T.J. Watt | 79.42 | 66.80 | 83.66 | 809 | Steelers |
| 20 | 2 | Frank Clark | 79.38 | 75.52 | 77.79 | 740 | Seahawks |
| 21 | 3 | Shaquil Barrett | 78.92 | 78.90 | 74.77 | 664 | Broncos |
| 22 | 4 | Jabaal Sheard | 78.64 | 83.63 | 71.34 | 900 | Colts |
| 23 | 5 | Everson Griffen | 78.58 | 77.08 | 75.42 | 910 | Vikings |
| 24 | 6 | Takk McKinley | 78.18 | 69.13 | 80.04 | 464 | Falcons |
| 25 | 7 | DeMarcus Walker | 78.04 | 72.19 | 84.03 | 100 | Broncos |
| 26 | 8 | Olivier Vernon | 78.01 | 81.16 | 73.82 | 698 | Giants |
| 27 | 9 | Pernell McPhee | 77.70 | 68.20 | 84.04 | 385 | Bears |
| 28 | 10 | Robert Quinn | 77.17 | 76.88 | 77.05 | 696 | Rams |
| 29 | 11 | Carl Lawson | 76.97 | 62.12 | 82.70 | 477 | Bengals |
| 30 | 12 | Nick Perry | 76.65 | 74.35 | 76.10 | 542 | Packers |
| 31 | 13 | Derek Barnett | 75.91 | 74.32 | 72.80 | 506 | Eagles |
| 32 | 14 | Whitney Mercilus | 75.60 | 68.58 | 81.85 | 203 | Texans |
| 33 | 15 | Mario Addison | 75.57 | 64.55 | 79.39 | 692 | Panthers |
| 34 | 16 | Junior Galette | 75.32 | 65.37 | 77.78 | 407 | Commanders |
| 35 | 17 | Yannick Ngakoue | 75.28 | 67.19 | 76.51 | 920 | Jaguars |
| 36 | 18 | James Harrison | 75.05 | 56.89 | 86.64 | 193 | Patriots |
| 37 | 19 | Jerry Hughes | 74.98 | 72.08 | 72.75 | 781 | Bills |
| 38 | 20 | Vinny Curry | 74.93 | 71.15 | 73.29 | 702 | Eagles |
| 39 | 21 | Leonard Floyd | 74.86 | 63.88 | 83.48 | 582 | Bears |
| 40 | 22 | William Hayes | 74.85 | 75.17 | 74.22 | 271 | Dolphins |
| 41 | 23 | Trey Flowers | 74.66 | 72.52 | 75.05 | 994 | Patriots |

### Starter (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 42 | 1 | Terrell Suggs | 73.99 | 65.53 | 78.89 | 845 | Ravens |
| 43 | 2 | Chris Long | 73.66 | 66.73 | 74.94 | 588 | Eagles |
| 44 | 3 | Robert Ayers | 73.60 | 75.23 | 72.51 | 588 | Buccaneers |
| 45 | 4 | Clay Matthews | 73.49 | 61.34 | 78.78 | 656 | Packers |
| 46 | 5 | Elvis Dumervil | 73.36 | 55.55 | 83.57 | 341 | 49ers |
| 47 | 6 | Derrick Morgan | 73.04 | 62.05 | 77.76 | 827 | Titans |
| 48 | 7 | Vic Beasley Jr. | 72.78 | 66.24 | 72.97 | 556 | Falcons |
| 49 | 8 | Lorenzo Alexander | 72.76 | 49.47 | 86.62 | 719 | Bills |
| 50 | 9 | Brian Orakpo | 72.57 | 59.53 | 77.09 | 938 | Titans |
| 51 | 10 | Markus Golden | 71.77 | 62.03 | 80.34 | 231 | Cardinals |
| 52 | 11 | Matthew Judon | 71.37 | 58.11 | 76.82 | 787 | Ravens |
| 53 | 12 | Jason Pierre-Paul | 71.14 | 70.53 | 70.29 | 1010 | Giants |
| 54 | 13 | Charles Johnson | 71.08 | 58.51 | 79.15 | 389 | Panthers |
| 55 | 14 | Julius Peppers | 71.07 | 58.06 | 75.57 | 531 | Panthers |
| 56 | 15 | Dante Fowler Jr. | 70.96 | 67.30 | 69.24 | 581 | Jaguars |
| 57 | 16 | Cliff Avril | 70.69 | 58.13 | 81.14 | 151 | Seahawks |
| 58 | 17 | Dont'a Hightower | 70.03 | 54.72 | 86.31 | 237 | Patriots |
| 59 | 18 | Aaron Lynch | 69.77 | 63.21 | 77.89 | 157 | 49ers |
| 60 | 19 | Tyus Bowser | 69.45 | 60.37 | 71.34 | 162 | Ravens |
| 61 | 20 | Alex Okafor | 69.43 | 64.40 | 72.69 | 486 | Saints |
| 62 | 21 | Bruce Irvin | 68.90 | 51.15 | 76.57 | 880 | Raiders |
| 63 | 22 | Dion Jordan | 68.64 | 66.28 | 77.53 | 135 | Seahawks |
| 64 | 23 | Preston Smith | 68.56 | 60.91 | 69.50 | 754 | Commanders |
| 65 | 24 | Chris McCain | 68.53 | 58.19 | 71.78 | 242 | Chargers |
| 66 | 25 | Shane Ray | 67.95 | 59.38 | 73.66 | 354 | Broncos |
| 67 | 26 | Jordan Jenkins | 67.47 | 60.74 | 68.57 | 715 | Jets |
| 68 | 27 | Adrian Clayborn | 67.40 | 64.81 | 65.60 | 632 | Falcons |
| 69 | 28 | Willie Young | 66.74 | 56.00 | 76.19 | 119 | Bears |
| 70 | 29 | Kyler Fackrell | 66.72 | 57.54 | 69.46 | 446 | Packers |
| 71 | 30 | Barkevious Mingo | 66.37 | 60.81 | 68.72 | 503 | Colts |
| 72 | 31 | Tarell Basham | 66.11 | 58.82 | 67.84 | 222 | Colts |
| 73 | 32 | John Simon | 66.05 | 59.15 | 71.68 | 472 | Colts |
| 74 | 33 | Deatrich Wise Jr. | 65.90 | 60.78 | 65.15 | 593 | Patriots |
| 75 | 34 | Charles Harris | 65.75 | 63.41 | 63.14 | 496 | Dolphins |
| 76 | 35 | Dee Ford | 65.70 | 58.03 | 71.85 | 316 | Chiefs |
| 77 | 36 | Samson Ebukam | 65.46 | 60.13 | 64.85 | 359 | Rams |
| 78 | 37 | Connor Barwin | 65.25 | 50.20 | 71.64 | 722 | Rams |
| 79 | 38 | Hau'oli Kikaha | 64.66 | 57.04 | 67.66 | 209 | Saints |
| 80 | 39 | Anthony Zettel | 64.42 | 59.77 | 64.13 | 753 | Lions |
| 81 | 40 | Lamarr Houston | 64.36 | 55.74 | 73.44 | 377 | Bears |
| 82 | 41 | Chris Smith | 64.17 | 60.17 | 67.87 | 401 | Bengals |
| 83 | 42 | Erik Walden | 63.82 | 47.81 | 70.53 | 644 | Titans |
| 84 | 43 | Devon Kennard | 63.72 | 54.47 | 66.23 | 543 | Giants |
| 85 | 44 | Bud Dupree | 63.46 | 57.36 | 65.24 | 850 | Steelers |
| 86 | 45 | Dwight Freeney | 63.39 | 50.78 | 71.89 | 227 | Lions |
| 87 | 46 | Marcus Smith | 63.22 | 56.81 | 66.15 | 252 | Seahawks |
| 88 | 47 | Taco Charlton | 63.18 | 59.55 | 61.44 | 399 | Cowboys |
| 89 | 48 | Ryan Davis Sr. | 62.96 | 56.35 | 65.80 | 489 | Bills |
| 90 | 49 | David Bass | 62.94 | 60.55 | 62.55 | 352 | Jets |
| 91 | 50 | Brooks Reed | 62.93 | 57.00 | 63.35 | 460 | Falcons |
| 92 | 51 | Tanoh Kpassagnon | 62.89 | 57.90 | 68.30 | 158 | Chiefs |
| 93 | 52 | Emmanuel Ogbah | 62.83 | 61.14 | 63.69 | 462 | Browns |
| 94 | 53 | Matt Longacre | 62.81 | 61.20 | 66.17 | 377 | Rams |
| 95 | 54 | Kony Ealy | 62.49 | 58.01 | 61.82 | 451 | Jets |
| 96 | 55 | Ahmad Brooks | 62.44 | 51.75 | 67.90 | 346 | Packers |
| 97 | 56 | Za'Darius Smith | 62.38 | 59.91 | 62.05 | 533 | Ravens |

### Rotation/backup (46 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 98 | 1 | Jordan Willis | 61.78 | 61.69 | 57.67 | 361 | Bengals |
| 99 | 2 | Cornelius Washington | 61.70 | 54.45 | 62.88 | 488 | Lions |
| 100 | 3 | Michael Johnson | 61.69 | 56.60 | 61.44 | 685 | Bengals |
| 101 | 4 | Sam Acho | 61.62 | 54.24 | 62.57 | 639 | Bears |
| 102 | 5 | Tamba Hali | 61.38 | 53.92 | 67.38 | 116 | Chiefs |
| 103 | 6 | Cassius Marsh | 61.23 | 60.03 | 58.38 | 455 | 49ers |
| 104 | 7 | Trey Hendrickson | 61.21 | 60.96 | 59.30 | 344 | Saints |
| 105 | 8 | Shaq Lawson | 60.80 | 60.15 | 62.67 | 436 | Bills |
| 106 | 9 | Tyrone Crawford | 60.78 | 55.61 | 60.37 | 626 | Cowboys |
| 107 | 10 | Solomon Thomas | 60.77 | 62.51 | 57.52 | 696 | 49ers |
| 108 | 11 | Ufomba Kamalu | 60.76 | 57.44 | 63.36 | 185 | Texans |
| 109 | 12 | Brian Robison | 60.65 | 47.19 | 65.45 | 642 | Vikings |
| 110 | 13 | James Cowser | 60.65 | 57.74 | 62.32 | 155 | Raiders |
| 111 | 14 | Wes Horton | 60.57 | 56.29 | 60.83 | 385 | Panthers |
| 112 | 15 | Anthony Chickillo | 60.47 | 57.17 | 62.57 | 272 | Steelers |
| 113 | 16 | Ronald Blair III | 60.13 | 59.14 | 63.14 | 140 | 49ers |
| 114 | 17 | Tim Williams | 59.66 | 59.76 | 63.76 | 125 | Ravens |
| 115 | 18 | Eddie Yarbrough | 59.60 | 58.87 | 55.92 | 494 | Bills |
| 116 | 19 | Dawuane Smoot | 59.57 | 59.58 | 55.39 | 286 | Jaguars |
| 117 | 20 | Andre Branch | 59.48 | 55.32 | 59.75 | 561 | Dolphins |
| 118 | 21 | Eric Lee | 59.27 | 55.85 | 64.68 | 355 | Patriots |
| 119 | 22 | Will Clarke | 59.15 | 57.11 | 57.49 | 315 | Buccaneers |
| 120 | 23 | Kasim Edebali | 59.03 | 55.95 | 59.00 | 103 | Saints |
| 121 | 24 | William Gholston | 58.79 | 57.18 | 57.37 | 448 | Buccaneers |
| 122 | 25 | Carl Nassib | 58.78 | 58.03 | 55.89 | 643 | Browns |
| 123 | 26 | Benson Mayowa | 58.69 | 59.66 | 56.38 | 381 | Cowboys |
| 124 | 27 | Josh Martin | 58.31 | 55.81 | 59.97 | 489 | Jets |
| 125 | 28 | Arik Armstead | 58.03 | 59.11 | 61.34 | 304 | 49ers |
| 126 | 29 | Nate Orchard | 57.85 | 57.33 | 58.29 | 431 | Browns |
| 127 | 30 | Kareem Martin | 57.43 | 58.03 | 56.52 | 458 | Cardinals |
| 128 | 31 | Noah Spence | 57.31 | 58.89 | 58.61 | 246 | Buccaneers |
| 129 | 32 | Kerry Wynn | 57.27 | 57.28 | 54.45 | 252 | Giants |
| 130 | 33 | Shilique Calhoun | 57.17 | 59.34 | 59.11 | 103 | Raiders |
| 131 | 34 | Brennan Scarlett | 56.83 | 59.19 | 58.25 | 302 | Texans |
| 132 | 35 | Branden Jackson | 56.72 | 58.22 | 59.24 | 263 | Seahawks |
| 133 | 36 | George Johnson | 56.40 | 53.10 | 59.12 | 269 | Saints |
| 134 | 37 | Vince Biegel | 55.82 | 59.51 | 57.53 | 121 | Packers |
| 135 | 38 | Lewis Neal | 55.66 | 59.07 | 60.08 | 140 | Cowboys |
| 136 | 39 | Terrence Fede | 55.42 | 57.31 | 53.32 | 173 | Dolphins |
| 137 | 40 | Ryan Anderson | 55.18 | 57.84 | 51.33 | 194 | Commanders |
| 138 | 41 | Kevin Dodd | 55.05 | 59.51 | 53.90 | 106 | Titans |
| 139 | 42 | Avery Moss | 54.74 | 60.42 | 53.03 | 248 | Giants |
| 140 | 43 | Frank Zombo | 54.40 | 47.44 | 55.71 | 639 | Chiefs |
| 141 | 44 | Bryan Cox Jr. | 52.71 | 57.57 | 53.64 | 145 | Panthers |
| 142 | 45 | Ryan Russell | 52.69 | 55.77 | 53.13 | 456 | Buccaneers |
| 143 | 46 | Cameron Malveaux | 45.08 | 58.70 | 50.34 | 107 | Dolphins |

## G — Guard

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | David DeCastro | 93.97 | 89.00 | 93.12 | 1125 | Steelers |
| 2 | 2 | Zack Martin | 93.23 | 89.00 | 91.88 | 1018 | Cowboys |
| 3 | 3 | Brandon Brooks | 90.72 | 86.20 | 89.56 | 1279 | Eagles |
| 4 | 4 | Shaq Mason | 87.90 | 81.60 | 87.94 | 1356 | Patriots |
| 5 | 5 | Rodger Saffold | 86.83 | 80.80 | 86.69 | 1010 | Rams |
| 6 | 6 | Brandon Scherff | 86.13 | 79.07 | 86.67 | 867 | Commanders |
| 7 | 7 | Marshal Yanda | 85.16 | 74.46 | 88.12 | 102 | Ravens |
| 8 | 8 | Josh Sitton | 84.13 | 75.20 | 85.91 | 712 | Bears |
| 9 | 9 | John Greco | 84.09 | 71.01 | 88.65 | 105 | Giants |
| 10 | 10 | Andrew Norwell | 83.26 | 76.30 | 83.73 | 1140 | Panthers |
| 11 | 11 | Kelechi Osemele | 83.17 | 76.50 | 83.45 | 1006 | Raiders |
| 12 | 12 | Andy Levitre | 82.78 | 73.22 | 84.99 | 699 | Falcons |
| 13 | 13 | Larry Warford | 81.60 | 73.85 | 82.60 | 951 | Saints |
| 14 | 14 | Trai Turner | 81.60 | 73.51 | 82.83 | 918 | Panthers |
| 15 | 15 | Joe Berger | 81.50 | 74.00 | 82.33 | 1259 | Vikings |
| 16 | 16 | Richie Incognito | 81.29 | 74.40 | 81.72 | 1109 | Bills |
| 17 | 17 | Joe Thuney | 80.97 | 74.40 | 81.18 | 1354 | Patriots |
| 18 | 18 | Joel Bitonio | 80.94 | 75.40 | 80.47 | 1068 | Browns |
| 19 | 19 | Laurent Duvernay-Tardif | 80.78 | 70.37 | 83.55 | 688 | Chiefs |
| 20 | 20 | Ron Leary | 80.56 | 71.47 | 82.46 | 712 | Broncos |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Kevin Zeitler | 79.76 | 73.20 | 79.96 | 1068 | Browns |
| 22 | 2 | Kyle Long | 79.42 | 69.17 | 82.08 | 447 | Bears |
| 23 | 3 | Graham Glasgow | 78.97 | 70.60 | 80.38 | 1042 | Lions |
| 24 | 4 | Brandon Fusco | 78.69 | 69.40 | 80.71 | 1083 | 49ers |
| 25 | 5 | Jahri Evans | 78.41 | 70.16 | 79.74 | 912 | Packers |
| 26 | 6 | Clint Boling | 77.80 | 69.14 | 79.40 | 962 | Bengals |
| 27 | 7 | Jon Feliciano | 77.70 | 65.40 | 81.73 | 124 | Raiders |
| 28 | 8 | Josh Kline | 77.60 | 69.50 | 78.84 | 1152 | Titans |
| 29 | 9 | Jonathan Cooper | 77.16 | 67.92 | 79.15 | 835 | Cowboys |
| 30 | 10 | Quinton Spain | 76.46 | 66.90 | 78.67 | 1007 | Titans |
| 31 | 11 | T.J. Lang | 76.40 | 67.46 | 78.19 | 809 | Lions |
| 32 | 12 | Gabe Jackson | 76.34 | 67.08 | 78.35 | 887 | Raiders |
| 33 | 13 | Andrus Peat | 76.32 | 68.07 | 77.65 | 932 | Saints |
| 34 | 14 | Ben Garland | 76.20 | 63.55 | 80.47 | 476 | Falcons |
| 35 | 15 | Joe Dahl | 75.93 | 64.56 | 79.34 | 182 | Lions |
| 36 | 16 | Laken Tomlinson | 75.90 | 66.40 | 78.07 | 1042 | 49ers |
| 37 | 17 | John Jerry | 75.89 | 66.05 | 78.29 | 958 | Giants |
| 38 | 18 | Lane Taylor | 75.71 | 68.01 | 76.67 | 939 | Packers |
| 39 | 19 | Vladimir Ducasse | 75.43 | 66.31 | 77.34 | 874 | Bills |
| 40 | 20 | Wes Schweitzer | 74.10 | 63.60 | 76.93 | 1149 | Falcons |

### Starter (44 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 41 | 1 | Evan Boehm | 73.81 | 61.39 | 77.92 | 588 | Cardinals |
| 42 | 2 | Patrick Omameh | 73.51 | 62.90 | 76.41 | 1052 | Jaguars |
| 43 | 3 | Dakota Dozier | 73.02 | 61.61 | 76.46 | 248 | Jets |
| 44 | 4 | D.J. Fluker | 72.91 | 60.20 | 77.22 | 446 | Giants |
| 45 | 5 | John Miller | 72.83 | 59.29 | 77.69 | 256 | Bills |
| 46 | 6 | A.J. Cann | 72.26 | 61.70 | 75.13 | 1228 | Jaguars |
| 47 | 7 | Dan Feeney | 72.18 | 60.66 | 75.69 | 665 | Chargers |
| 48 | 8 | Chad Slade | 72.08 | 61.46 | 75.00 | 163 | Texans |
| 49 | 9 | Senio Kelemete | 71.99 | 61.74 | 74.66 | 748 | Saints |
| 50 | 10 | Chance Warmack | 71.82 | 61.34 | 74.64 | 321 | Eagles |
| 51 | 11 | Allen Barbre | 71.49 | 58.56 | 75.95 | 553 | Broncos |
| 52 | 12 | Kevin Pamphile | 71.09 | 59.73 | 74.49 | 782 | Buccaneers |
| 53 | 13 | James Carpenter | 70.97 | 59.30 | 74.59 | 1035 | Jets |
| 54 | 14 | Kenny Wiggins | 70.74 | 60.10 | 73.67 | 1040 | Chargers |
| 55 | 15 | Luke Joeckel | 69.97 | 58.90 | 73.19 | 702 | Seahawks |
| 56 | 16 | J.R. Sweezy | 69.88 | 59.32 | 72.75 | 903 | Buccaneers |
| 57 | 17 | Ramon Foster | 69.88 | 60.20 | 72.17 | 1006 | Steelers |
| 58 | 18 | Mark Glowinski | 69.69 | 56.76 | 74.15 | 199 | Colts |
| 59 | 19 | Jack Mewhort | 69.40 | 57.79 | 72.98 | 313 | Colts |
| 60 | 20 | Anthony Steen | 69.38 | 60.75 | 70.97 | 327 | Dolphins |
| 61 | 21 | Brian Winters | 69.34 | 56.92 | 73.46 | 807 | Jets |
| 62 | 22 | Max Garcia | 69.20 | 58.46 | 72.19 | 869 | Broncos |
| 63 | 23 | Tyler Shatley | 69.15 | 57.50 | 72.75 | 386 | Jaguars |
| 64 | 24 | Jordan Devey | 68.72 | 58.48 | 71.38 | 147 | Chiefs |
| 65 | 25 | Arie Kouandjio | 68.58 | 57.24 | 71.98 | 424 | Commanders |
| 66 | 26 | Lucas Patrick | 68.43 | 57.07 | 71.83 | 227 | Packers |
| 67 | 27 | Trey Hopkins | 68.21 | 60.08 | 69.47 | 707 | Bengals |
| 68 | 28 | Matt Slauson | 67.95 | 55.87 | 71.84 | 424 | Chargers |
| 69 | 29 | Bryan Witzmann | 67.61 | 54.06 | 72.48 | 933 | Chiefs |
| 70 | 30 | Shawn Lauvao | 67.44 | 55.08 | 71.52 | 531 | Commanders |
| 71 | 31 | Alex Boone | 67.41 | 56.04 | 70.82 | 874 | Cardinals |
| 72 | 32 | Jermaine Eluemunor | 66.89 | 55.38 | 70.40 | 198 | Ravens |
| 73 | 33 | Jeremiah Sirles | 66.74 | 55.12 | 70.32 | 366 | Vikings |
| 74 | 34 | Oday Aboushi | 66.55 | 54.43 | 70.46 | 558 | Seahawks |
| 75 | 35 | Jermon Bushrod | 66.39 | 53.81 | 70.61 | 604 | Dolphins |
| 76 | 36 | Ted Larsen | 65.82 | 52.05 | 70.84 | 521 | Dolphins |
| 77 | 37 | Amini Silatolu | 65.71 | 55.09 | 68.62 | 247 | Panthers |
| 78 | 38 | Xavier Su'a-Filo | 65.67 | 53.70 | 69.48 | 1075 | Texans |
| 79 | 39 | Jeff Allen | 65.23 | 51.02 | 70.53 | 728 | Texans |
| 80 | 40 | Connor McGovern | 65.14 | 52.77 | 69.22 | 418 | Broncos |
| 81 | 41 | Cameron Erving | 65.02 | 53.17 | 68.75 | 276 | Chiefs |
| 82 | 42 | Alex Redmond | 64.74 | 57.36 | 65.50 | 104 | Bengals |
| 83 | 43 | Isaac Seumalo | 63.79 | 52.34 | 67.25 | 308 | Eagles |
| 84 | 44 | Jon Halapio | 62.92 | 57.95 | 62.06 | 403 | Giants |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 85 | 1 | Zane Beadles | 61.62 | 48.14 | 66.44 | 395 | 49ers |
| 86 | 2 | Danny Isidora | 61.09 | 53.36 | 62.07 | 147 | Vikings |

## HB — Running Back

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Alvin Kamara | 90.73 | 88.88 | 87.79 | 343 | Saints |
| 2 | 2 | Kareem Hunt | 81.01 | 80.53 | 77.16 | 328 | Chiefs |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 3 | 1 | Kenyan Drake | 78.85 | 72.00 | 79.25 | 264 | Dolphins |
| 4 | 2 | Dion Lewis | 78.25 | 78.85 | 73.68 | 206 | Patriots |
| 5 | 3 | Austin Ekeler | 77.21 | 68.27 | 79.01 | 122 | Chargers |
| 6 | 4 | Ty Montgomery | 76.10 | 69.68 | 76.21 | 148 | Packers |
| 7 | 5 | Duke Johnson Jr. | 76.09 | 77.04 | 71.29 | 333 | Browns |
| 8 | 6 | Todd Gurley II | 76.04 | 82.60 | 67.50 | 408 | Rams |
| 9 | 7 | Chris Thompson | 74.59 | 70.76 | 72.97 | 218 | Commanders |
| 10 | 8 | Marshawn Lynch | 74.36 | 71.90 | 71.84 | 182 | Raiders |

### Starter (58 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Le'Veon Bell | 73.87 | 72.20 | 70.82 | 557 | Steelers |
| 12 | 2 | Jay Ajayi | 73.80 | 66.26 | 74.66 | 271 | Eagles |
| 13 | 3 | Jalen Richard | 73.75 | 65.21 | 75.27 | 130 | Raiders |
| 14 | 4 | C.J. Anderson | 73.61 | 75.98 | 67.87 | 241 | Broncos |
| 15 | 5 | Jerick McKinnon | 73.12 | 78.68 | 65.24 | 345 | Vikings |
| 16 | 6 | Aaron Jones | 73.07 | 70.07 | 70.90 | 109 | Packers |
| 17 | 7 | Alex Collins | 73.02 | 72.47 | 69.22 | 108 | Ravens |
| 18 | 8 | LeSean McCoy | 72.51 | 73.45 | 67.71 | 357 | Bills |
| 19 | 9 | Tarik Cohen | 72.33 | 72.63 | 67.97 | 198 | Bears |
| 20 | 10 | Derrick Henry | 72.25 | 69.26 | 70.08 | 193 | Titans |
| 21 | 11 | Ezekiel Elliott | 71.54 | 68.58 | 69.34 | 267 | Cowboys |
| 22 | 12 | Mark Ingram II | 71.41 | 68.74 | 69.03 | 263 | Saints |
| 23 | 13 | Corey Clement | 71.06 | 69.23 | 68.11 | 163 | Eagles |
| 24 | 14 | Wayne Gallman | 70.52 | 66.59 | 68.97 | 165 | Giants |
| 25 | 15 | Christian McCaffrey | 70.51 | 74.60 | 63.62 | 481 | Panthers |
| 26 | 16 | Melvin Gordon III | 69.52 | 71.20 | 64.24 | 320 | Chargers |
| 27 | 17 | Marlon Mack | 69.41 | 65.21 | 68.04 | 161 | Colts |
| 28 | 18 | Theo Riddick | 69.40 | 69.97 | 64.86 | 271 | Lions |
| 29 | 19 | Devonta Freeman | 69.14 | 68.33 | 65.51 | 303 | Falcons |
| 30 | 20 | Jordan Howard | 69.02 | 63.21 | 68.73 | 190 | Bears |
| 31 | 21 | Giovani Bernard | 68.67 | 66.52 | 65.94 | 256 | Bengals |
| 32 | 22 | Jamaal Charles | 68.44 | 59.57 | 70.19 | 104 | Broncos |
| 33 | 23 | Tevin Coleman | 68.24 | 67.72 | 64.42 | 222 | Falcons |
| 34 | 24 | DeMarco Murray | 68.07 | 63.06 | 67.25 | 292 | Titans |
| 35 | 25 | LeGarrette Blount | 68.01 | 62.00 | 67.85 | 124 | Eagles |
| 36 | 26 | Matt Forte | 67.90 | 65.03 | 65.64 | 191 | Jets |
| 37 | 27 | Damien Williams | 67.71 | 62.31 | 67.14 | 110 | Dolphins |
| 38 | 28 | Chris Ivory | 67.49 | 59.16 | 68.88 | 119 | Jaguars |
| 39 | 29 | Lamar Miller | 67.47 | 68.58 | 62.57 | 336 | Texans |
| 40 | 30 | Frank Gore | 67.44 | 68.66 | 62.46 | 198 | Colts |
| 41 | 31 | Carlos Hyde | 67.36 | 58.01 | 69.42 | 376 | 49ers |
| 42 | 32 | Bilal Powell | 67.36 | 65.74 | 64.27 | 158 | Jets |
| 43 | 33 | Rex Burkhead | 67.28 | 65.73 | 64.14 | 111 | Patriots |
| 44 | 34 | Rod Smith | 67.18 | 64.49 | 64.81 | 129 | Cowboys |
| 45 | 35 | Orleans Darkwa | 66.59 | 60.16 | 66.71 | 109 | Giants |
| 46 | 36 | Latavius Murray | 66.11 | 64.76 | 62.84 | 145 | Vikings |
| 47 | 37 | T.J. Yeldon | 65.81 | 62.50 | 63.85 | 149 | Jaguars |
| 48 | 38 | DeAndre Washington | 65.77 | 58.98 | 66.13 | 123 | Raiders |
| 49 | 39 | James White | 65.72 | 68.85 | 59.47 | 346 | Patriots |
| 50 | 40 | J.D. McKissic | 65.61 | 65.62 | 61.44 | 219 | Seahawks |
| 51 | 41 | Jonathan Stewart | 65.55 | 57.40 | 66.82 | 110 | Panthers |
| 52 | 42 | Joe Mixon | 65.44 | 67.78 | 59.71 | 157 | Bengals |
| 53 | 43 | Danny Woodhead | 65.43 | 68.51 | 59.21 | 113 | Ravens |
| 54 | 44 | Benny Cunningham | 65.39 | 61.58 | 63.77 | 121 | Bears |
| 55 | 45 | Isaiah Crowell | 65.16 | 60.90 | 63.84 | 214 | Browns |
| 56 | 46 | Javorius Allen | 64.66 | 66.59 | 59.21 | 218 | Ravens |
| 57 | 47 | Elijah McGuire | 64.59 | 63.89 | 60.89 | 128 | Jets |
| 58 | 48 | Matt Breida | 64.33 | 62.52 | 61.37 | 159 | 49ers |
| 59 | 49 | Leonard Fournette | 64.25 | 62.01 | 61.58 | 267 | Jaguars |
| 60 | 50 | Andre Ellington | 64.11 | 61.59 | 61.62 | 249 | Texans |
| 61 | 51 | Devontae Booker | 63.97 | 62.93 | 60.49 | 168 | Broncos |
| 62 | 52 | Ameer Abdullah | 63.92 | 55.82 | 65.15 | 157 | Lions |
| 63 | 53 | Thomas Rawls | 63.44 | 55.55 | 64.54 | 104 | Seahawks |
| 64 | 54 | Shane Vereen | 63.13 | 58.18 | 62.27 | 228 | Giants |
| 65 | 55 | Jamaal Williams | 62.42 | 61.38 | 58.94 | 200 | Packers |
| 66 | 56 | Charles Sims | 62.39 | 57.63 | 61.39 | 240 | Buccaneers |
| 67 | 57 | Charcandrick West | 62.09 | 60.07 | 59.27 | 173 | Chiefs |
| 68 | 58 | Samaje Perine | 62.06 | 59.28 | 59.75 | 136 | Commanders |

### Rotation/backup (0 players)

_None._

## LB — Linebacker

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Bobby Wagner | 89.95 | 91.40 | 84.82 | 1022 | Seahawks |
| 2 | 2 | Luke Kuechly | 86.80 | 90.60 | 81.98 | 973 | Panthers |
| 3 | 3 | Lavonte David | 86.41 | 90.43 | 81.13 | 814 | Buccaneers |
| 4 | 4 | Paul Posluszny | 83.14 | 84.14 | 78.73 | 520 | Jaguars |
| 5 | 5 | Deion Jones | 82.97 | 88.00 | 75.45 | 1148 | Falcons |
| 6 | 6 | Telvin Smith Sr. | 80.32 | 81.90 | 75.51 | 1065 | Jaguars |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Sean Lee | 79.82 | 80.39 | 78.29 | 622 | Cowboys |
| 8 | 2 | Myles Jack | 78.35 | 78.80 | 74.66 | 1223 | Jaguars |
| 9 | 3 | Mychal Kendricks | 77.93 | 79.19 | 73.86 | 726 | Eagles |
| 10 | 4 | Avery Williamson | 77.68 | 79.22 | 72.68 | 742 | Titans |
| 11 | 5 | Danny Trevathan | 77.17 | 79.58 | 75.67 | 714 | Bears |
| 12 | 6 | Ryan Shazier | 76.78 | 80.08 | 72.91 | 671 | Steelers |
| 13 | 7 | Korey Toomer | 75.41 | 79.66 | 74.88 | 266 | Chargers |
| 14 | 8 | Demario Davis | 75.14 | 73.70 | 71.94 | 1115 | Jets |
| 15 | 9 | Reuben Foster | 74.43 | 77.41 | 74.52 | 553 | 49ers |
| 16 | 10 | C.J. Mosley | 74.04 | 72.20 | 71.74 | 1077 | Ravens |

### Starter (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Nick Kwiatkoski | 73.99 | 73.99 | 74.25 | 382 | Bears |
| 18 | 2 | Dylan Cole | 73.15 | 72.68 | 73.46 | 206 | Texans |
| 19 | 3 | Kyle Emanuel | 73.01 | 68.48 | 71.87 | 301 | Chargers |
| 20 | 4 | Jake Ryan | 72.77 | 72.33 | 70.98 | 506 | Packers |
| 21 | 5 | Benardrick McKinney | 72.58 | 69.00 | 71.21 | 959 | Texans |
| 22 | 6 | Anthony Hitchens | 72.55 | 70.59 | 71.78 | 544 | Cowboys |
| 23 | 7 | K.J. Wright | 71.68 | 68.70 | 70.01 | 956 | Seahawks |
| 24 | 8 | Reggie Ragland | 71.63 | 70.73 | 70.01 | 369 | Chiefs |
| 25 | 9 | Ben Gedeon | 71.52 | 69.45 | 68.74 | 272 | Vikings |
| 26 | 10 | De'Vondre Campbell | 71.04 | 69.20 | 68.88 | 1056 | Falcons |
| 27 | 11 | Joe Schobert | 70.94 | 68.70 | 68.26 | 1068 | Browns |
| 28 | 12 | Zach Cunningham | 70.64 | 66.57 | 69.19 | 812 | Texans |
| 29 | 13 | Blake Martinez | 70.63 | 66.20 | 69.41 | 978 | Packers |
| 30 | 14 | Vontaze Burfict | 70.43 | 71.72 | 71.14 | 589 | Bengals |
| 31 | 15 | Jaylon Smith | 69.97 | 67.87 | 67.20 | 575 | Cowboys |
| 32 | 16 | Jatavis Brown | 69.88 | 67.22 | 69.05 | 504 | Chargers |
| 33 | 17 | Wesley Woodyard | 69.40 | 65.10 | 68.10 | 1153 | Titans |
| 34 | 18 | Todd Davis | 69.03 | 64.22 | 69.10 | 520 | Broncos |
| 35 | 19 | Eric Kendricks | 68.84 | 67.00 | 66.41 | 1099 | Vikings |
| 36 | 20 | Michael Wilhoite | 68.81 | 64.52 | 70.00 | 306 | Seahawks |
| 37 | 21 | Thomas Davis Sr. | 68.27 | 64.10 | 66.89 | 849 | Panthers |
| 38 | 22 | Manti Te'o | 68.07 | 68.09 | 68.79 | 621 | Saints |
| 39 | 23 | NaVorro Bowman | 67.90 | 67.60 | 68.20 | 997 | Raiders |
| 40 | 24 | Nigel Bradham | 67.75 | 65.60 | 66.05 | 1125 | Eagles |
| 41 | 25 | Derrick Johnson | 67.57 | 65.10 | 65.99 | 893 | Chiefs |
| 42 | 26 | Shaq Thompson | 67.19 | 63.41 | 66.69 | 684 | Panthers |
| 43 | 27 | John Timu | 67.19 | 67.59 | 72.24 | 135 | Bears |
| 44 | 28 | Christian Kirksey | 67.11 | 62.00 | 66.35 | 1068 | Browns |
| 45 | 29 | Josh Bynes | 67.11 | 65.31 | 67.58 | 236 | Cardinals |
| 46 | 30 | Craig Robertson | 67.00 | 64.30 | 65.78 | 921 | Saints |
| 47 | 31 | Anthony Barr | 66.70 | 63.80 | 64.67 | 1050 | Vikings |
| 48 | 32 | Patrick Onwuasor | 66.64 | 65.77 | 66.18 | 647 | Ravens |
| 49 | 33 | Jon Bostic | 66.50 | 63.40 | 67.53 | 914 | Colts |
| 50 | 34 | Karlos Dansby | 66.44 | 63.00 | 64.56 | 921 | Cardinals |
| 51 | 35 | Preston Brown | 66.31 | 60.10 | 66.28 | 1157 | Bills |
| 52 | 36 | Tahir Whitehead | 66.02 | 61.40 | 64.93 | 950 | Lions |
| 53 | 37 | Chase Allen | 65.40 | 60.88 | 66.33 | 220 | Dolphins |
| 54 | 38 | Mark Barron | 65.30 | 61.40 | 64.38 | 896 | Rams |
| 55 | 39 | Kwon Alexander | 65.20 | 65.14 | 63.99 | 717 | Buccaneers |
| 56 | 40 | Zach Brown | 64.97 | 58.70 | 66.55 | 834 | Commanders |
| 57 | 41 | Zaire Anderson | 64.80 | 60.45 | 68.35 | 136 | Broncos |
| 58 | 42 | Jalen Reeves-Maybin | 64.54 | 64.43 | 64.61 | 239 | Lions |
| 59 | 43 | Vince Williams | 64.24 | 61.86 | 63.23 | 785 | Steelers |
| 60 | 44 | Brandon Marshall | 64.04 | 61.80 | 62.94 | 910 | Broncos |
| 61 | 45 | Bryce Hager | 63.91 | 62.55 | 67.42 | 153 | Rams |
| 62 | 46 | Matt Milano | 63.39 | 60.22 | 63.42 | 450 | Bills |
| 63 | 47 | Zach Vigil | 63.22 | 65.34 | 68.38 | 394 | Commanders |
| 64 | 48 | Marquel Lee | 63.06 | 60.60 | 64.70 | 172 | Raiders |
| 65 | 49 | Christian Jones | 63.06 | 56.34 | 64.10 | 623 | Bears |
| 66 | 50 | Jeremiah George | 62.93 | 61.55 | 68.54 | 180 | Colts |
| 67 | 51 | Kyle Van Noy | 62.55 | 58.50 | 62.12 | 894 | Patriots |
| 68 | 52 | Vincent Rey | 62.52 | 56.61 | 63.33 | 607 | Bengals |
| 69 | 53 | Mason Foster | 62.33 | 63.40 | 64.43 | 288 | Commanders |
| 70 | 54 | Eli Harold | 62.32 | 57.62 | 61.28 | 452 | 49ers |

### Rotation/backup (57 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 71 | 1 | David Harris | 61.96 | 60.68 | 62.08 | 181 | Patriots |
| 72 | 2 | Nick Bellore | 61.70 | 60.84 | 66.96 | 106 | Lions |
| 73 | 3 | Kelvin Sheppard | 61.61 | 57.94 | 64.57 | 386 | Giants |
| 74 | 4 | Cory Littleton | 61.53 | 60.12 | 60.66 | 278 | Rams |
| 75 | 5 | Paul Worrilow | 61.52 | 59.48 | 62.88 | 272 | Lions |
| 76 | 6 | Jordan Hicks | 61.35 | 63.19 | 62.53 | 268 | Eagles |
| 77 | 7 | Kendell Beckwith | 61.35 | 53.00 | 62.75 | 847 | Buccaneers |
| 78 | 8 | Damien Wilson | 60.83 | 54.06 | 65.02 | 321 | Cowboys |
| 79 | 9 | Brian Cushing | 60.81 | 59.53 | 63.54 | 163 | Texans |
| 80 | 10 | Jayon Brown | 60.65 | 55.35 | 60.01 | 527 | Titans |
| 81 | 11 | Nicholas Morrow | 60.48 | 54.66 | 61.23 | 553 | Raiders |
| 82 | 12 | Will Compton | 60.28 | 57.94 | 63.19 | 151 | Commanders |
| 83 | 13 | Denzel Perryman | 60.13 | 59.54 | 62.92 | 273 | Chargers |
| 84 | 14 | Martrell Spaight | 60.12 | 58.65 | 61.84 | 414 | Commanders |
| 85 | 15 | Joe Thomas | 60.10 | 59.20 | 61.73 | 104 | Packers |
| 86 | 16 | Marquis Flowers | 59.89 | 57.15 | 64.21 | 361 | Patriots |
| 87 | 17 | Stephone Anthony | 59.43 | 58.48 | 63.08 | 130 | Dolphins |
| 88 | 18 | Lawrence Timmons | 59.22 | 53.76 | 59.73 | 794 | Dolphins |
| 89 | 19 | Duke Riley | 59.02 | 56.99 | 59.34 | 245 | Falcons |
| 90 | 20 | Adarius Taylor | 58.84 | 59.29 | 61.56 | 284 | Buccaneers |
| 91 | 21 | Jarrad Davis | 58.75 | 52.80 | 60.63 | 828 | Lions |
| 92 | 22 | Najee Goode | 58.68 | 57.38 | 62.46 | 208 | Eagles |
| 93 | 23 | Dannell Ellerbe | 58.59 | 59.20 | 64.01 | 102 | Eagles |
| 94 | 24 | Alec Ogletree | 58.57 | 55.00 | 59.29 | 994 | Rams |
| 95 | 25 | Devante Bond | 58.49 | 58.00 | 58.55 | 137 | Buccaneers |
| 96 | 26 | Kiko Alonso | 58.44 | 49.10 | 61.54 | 1007 | Dolphins |
| 97 | 27 | Keenan Robinson | 58.41 | 56.06 | 61.86 | 292 | Giants |
| 98 | 28 | Kevin Minter | 58.02 | 51.84 | 61.62 | 196 | Bengals |
| 99 | 29 | Elandon Roberts | 57.88 | 49.33 | 59.79 | 670 | Patriots |
| 100 | 30 | Ramik Wilson | 57.88 | 55.33 | 65.63 | 125 | Chiefs |
| 101 | 31 | Kevin Pierre-Louis | 57.42 | 54.63 | 60.74 | 273 | Chiefs |
| 102 | 32 | Terence Garvin | 57.26 | 59.22 | 60.75 | 195 | Seahawks |
| 103 | 33 | Kamalei Correa | 57.21 | 53.37 | 56.26 | 148 | Ravens |
| 104 | 34 | James Burgess | 57.04 | 50.15 | 59.55 | 646 | Browns |
| 105 | 35 | Cory James | 56.92 | 53.67 | 60.00 | 455 | Raiders |
| 106 | 36 | David Mayo | 56.90 | 56.32 | 59.78 | 134 | Panthers |
| 107 | 37 | Ramon Humber | 56.88 | 51.33 | 59.75 | 629 | Bills |
| 108 | 38 | Ray-Ray Armstrong | 56.69 | 54.87 | 61.14 | 481 | Giants |
| 109 | 39 | B.J. Goodson | 56.44 | 53.45 | 64.81 | 374 | Giants |
| 110 | 40 | Hardy Nickerson | 56.38 | 53.17 | 58.52 | 159 | Bengals |
| 111 | 41 | Mike Hull | 56.06 | 57.34 | 62.07 | 185 | Dolphins |
| 112 | 42 | Jonathan Freeny | 56.00 | 57.80 | 60.54 | 101 | Saints |
| 113 | 43 | Antonio Morrison | 55.71 | 45.36 | 59.87 | 813 | Colts |
| 114 | 44 | Deone Bucannon | 55.33 | 48.70 | 59.36 | 704 | Cardinals |
| 115 | 45 | A.J. Klein | 55.32 | 49.38 | 58.35 | 664 | Saints |
| 116 | 46 | Brock Coyle | 55.08 | 52.72 | 58.53 | 646 | 49ers |
| 117 | 47 | Alex Anzalone | 55.05 | 60.61 | 65.67 | 158 | Saints |
| 118 | 48 | Anthony Walker Jr. | 54.99 | 62.08 | 66.10 | 115 | Colts |
| 119 | 49 | Nick Vigil | 54.62 | 50.76 | 58.23 | 759 | Bengals |
| 120 | 50 | Calvin Munson | 54.61 | 48.24 | 58.86 | 388 | Giants |
| 121 | 51 | Darron Lee | 54.58 | 44.20 | 59.15 | 1025 | Jets |
| 122 | 52 | Jonathan Casillas | 54.38 | 49.85 | 57.60 | 457 | Giants |
| 123 | 53 | Hayes Pullard | 53.39 | 48.60 | 58.67 | 474 | Chargers |
| 124 | 54 | Jordan Evans | 51.96 | 46.68 | 56.52 | 312 | Bengals |
| 125 | 55 | Jamie Collins Sr. | 51.09 | 46.03 | 56.23 | 330 | Browns |
| 126 | 56 | Curtis Grant | 50.93 | 53.76 | 58.29 | 109 | Giants |
| 127 | 57 | Sean Spence | 48.35 | 43.09 | 54.35 | 218 | Steelers |

## QB — Quarterback

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Tom Brady | 87.96 | 93.25 | 79.08 | 827 | Patriots |
| 2 | 2 | Drew Brees | 85.26 | 86.14 | 80.00 | 667 | Saints |
| 3 | 3 | Matt Ryan | 82.00 | 87.33 | 73.09 | 674 | Falcons |
| 4 | 4 | Ben Roethlisberger | 81.33 | 83.67 | 75.00 | 697 | Steelers |
| 5 | 5 | Alex Smith | 80.56 | 78.90 | 77.59 | 647 | Chiefs |

### Good (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Russell Wilson | 78.08 | 77.90 | 73.68 | 713 | Seahawks |
| 7 | 2 | Matthew Stafford | 77.88 | 75.90 | 74.95 | 674 | Lions |
| 8 | 3 | Philip Rivers | 76.63 | 76.16 | 72.70 | 634 | Chargers |
| 9 | 4 | Carson Wentz | 76.28 | 78.12 | 72.52 | 537 | Eagles |
| 10 | 5 | Kirk Cousins | 75.19 | 71.87 | 73.39 | 626 | Commanders |

### Starter (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Aaron Rodgers | 73.73 | 77.71 | 72.38 | 302 | Packers |
| 12 | 2 | Jameis Winston | 73.46 | 72.57 | 71.77 | 529 | Buccaneers |
| 13 | 3 | Case Keenum | 72.95 | 73.69 | 71.13 | 666 | Vikings |
| 14 | 4 | Derek Carr | 72.18 | 75.13 | 66.59 | 586 | Raiders |
| 15 | 5 | Andy Dalton | 71.58 | 72.24 | 68.19 | 575 | Bengals |
| 16 | 6 | Dak Prescott | 70.65 | 71.60 | 66.95 | 606 | Cowboys |
| 17 | 7 | Jared Goff | 70.59 | 70.43 | 72.90 | 589 | Rams |
| 18 | 8 | Marcus Mariota | 70.23 | 70.16 | 67.48 | 626 | Titans |
| 19 | 9 | Tyrod Taylor | 69.94 | 74.45 | 63.80 | 584 | Bills |
| 20 | 10 | Cam Newton | 69.28 | 69.93 | 64.67 | 641 | Panthers |
| 21 | 11 | Carson Palmer | 68.68 | 70.16 | 69.10 | 312 | Cardinals |
| 22 | 12 | Blake Bortles | 67.06 | 63.40 | 65.50 | 706 | Jaguars |
| 23 | 13 | Eli Manning | 66.11 | 66.30 | 61.43 | 645 | Giants |
| 24 | 14 | Josh McCown | 66.08 | 63.58 | 70.52 | 482 | Jets |
| 25 | 15 | Joe Flacco | 65.53 | 67.37 | 60.50 | 605 | Ravens |
| 26 | 16 | Jimmy Garoppolo | 64.53 | 74.79 | 75.29 | 196 | 49ers |
| 27 | 17 | Nick Foles | 63.79 | 67.87 | 71.35 | 225 | Eagles |
| 28 | 18 | Deshaun Watson | 63.76 | 62.69 | 79.42 | 267 | Texans |

### Rotation/backup (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 29 | 1 | Mitch Trubisky | 61.71 | 69.30 | 61.41 | 415 | Bears |
| 30 | 2 | Jacoby Brissett | 60.95 | 62.17 | 61.68 | 591 | Colts |
| 31 | 3 | Ryan Fitzpatrick | 60.08 | 60.66 | 64.62 | 194 | Buccaneers |
| 32 | 4 | Jay Cutler | 59.51 | 59.76 | 61.33 | 476 | Dolphins |
| 33 | 5 | Brian Hoyer | 58.44 | 64.62 | 59.64 | 241 | Patriots |
| 34 | 6 | Matt Moore | 58.07 | 56.62 | 64.05 | 148 | Dolphins |
| 35 | 7 | Mike Glennon | 58.02 | 61.62 | 59.78 | 159 | Bears |
| 36 | 8 | Tom Savage | 57.96 | 59.24 | 60.53 | 268 | Texans |
| 37 | 9 | C.J. Beathard | 57.50 | 57.76 | 57.68 | 265 | 49ers |
| 38 | 10 | DeShone Kizer | 57.22 | 49.90 | 57.00 | 605 | Browns |
| 39 | 11 | Brett Hundley | 57.11 | 58.61 | 55.55 | 389 | Packers |
| 40 | 12 | Trevor Siemian | 56.58 | 54.40 | 60.03 | 432 | Broncos |
| 41 | 13 | Drew Stanton | 56.01 | 56.29 | 56.01 | 181 | Cardinals |
| 42 | 14 | T.J. Yates | 55.90 | 53.84 | 57.57 | 123 | Texans |
| 43 | 15 | Bryce Petty | 55.20 | 52.47 | 55.86 | 135 | Jets |
| 44 | 16 | Brock Osweiler | 53.47 | 54.82 | 56.81 | 206 | Broncos |
| 45 | 17 | Blaine Gabbert | 53.06 | 52.71 | 57.55 | 219 | Cardinals |

## S — Safety

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Harrison Smith | 93.50 | 91.60 | 91.64 | 1103 | Vikings |
| 2 | 2 | Lamarcus Joyner | 92.94 | 89.87 | 92.39 | 756 | Rams |
| 3 | 3 | Micah Hyde | 91.43 | 90.40 | 87.95 | 1101 | Bills |
| 4 | 4 | Marcus Williams | 91.41 | 89.60 | 88.45 | 1103 | Saints |
| 5 | 5 | Glover Quin | 91.13 | 91.30 | 86.85 | 1053 | Lions |
| 6 | 6 | Earl Thomas III | 90.91 | 90.50 | 89.61 | 921 | Seahawks |
| 7 | 7 | Tre Boston | 89.82 | 89.00 | 86.72 | 1039 | Chargers |
| 8 | 8 | Jordan Poyer | 88.58 | 90.90 | 86.41 | 1097 | Bills |
| 9 | 9 | Antoine Bethea | 88.46 | 86.73 | 87.84 | 741 | Cardinals |
| 10 | 10 | Adrian Amos | 87.82 | 86.58 | 87.48 | 670 | Bears |
| 11 | 11 | Kevin Byard | 87.58 | 85.40 | 84.86 | 1223 | Titans |
| 12 | 12 | John Johnson III | 85.14 | 82.28 | 82.88 | 781 | Rams |
| 13 | 13 | Landon Collins | 84.37 | 81.60 | 82.57 | 908 | Giants |
| 14 | 14 | Andrew Sendejo | 83.44 | 81.30 | 82.27 | 853 | Vikings |
| 15 | 15 | Bradley McDougald | 81.79 | 79.90 | 78.89 | 675 | Seahawks |
| 16 | 16 | Rodney McLeod | 81.72 | 81.60 | 77.64 | 1048 | Eagles |
| 17 | 17 | Mike Adams | 81.39 | 80.30 | 78.88 | 1022 | Panthers |
| 18 | 18 | Jeff Heath | 80.90 | 80.80 | 78.88 | 880 | Cowboys |
| 19 | 19 | Duron Harmon | 80.62 | 79.50 | 77.20 | 829 | Patriots |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Tyvon Branch | 79.80 | 76.50 | 81.48 | 579 | Cardinals |
| 21 | 2 | Malcolm Jenkins | 78.07 | 74.30 | 76.42 | 1151 | Eagles |
| 22 | 3 | Jaquiski Tartt | 77.39 | 78.00 | 76.99 | 595 | 49ers |
| 23 | 4 | Corey Graham | 77.18 | 74.84 | 74.57 | 488 | Eagles |
| 24 | 5 | Patrick Chung | 76.81 | 73.60 | 74.79 | 1127 | Patriots |
| 25 | 6 | Eric Weddle | 76.63 | 76.10 | 73.45 | 1084 | Ravens |
| 26 | 7 | Eddie Pleasant | 76.08 | 69.09 | 76.58 | 307 | Texans |
| 27 | 8 | Ricardo Allen | 74.69 | 71.10 | 72.91 | 1083 | Falcons |
| 28 | 9 | Corey Moore | 74.59 | 70.65 | 77.09 | 241 | Texans |
| 29 | 10 | Ha Ha Clinton-Dix | 74.51 | 69.20 | 73.89 | 1043 | Packers |
| 30 | 11 | George Iloka | 74.48 | 70.10 | 73.87 | 988 | Bengals |
| 31 | 12 | Devin McCourty | 74.40 | 66.70 | 75.36 | 1224 | Patriots |
| 32 | 13 | Eddie Jackson | 74.13 | 69.00 | 73.39 | 1055 | Bears |

### Starter (46 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Cody Davis | 73.73 | 74.33 | 76.66 | 280 | Rams |
| 34 | 2 | Barry Church | 73.51 | 72.20 | 71.37 | 1203 | Jaguars |
| 35 | 3 | D.J. Swearinger Sr. | 73.25 | 70.40 | 72.24 | 1092 | Commanders |
| 36 | 4 | Kam Chancellor | 73.04 | 71.57 | 74.76 | 598 | Seahawks |
| 37 | 5 | Eric Reid | 72.74 | 71.90 | 72.56 | 736 | 49ers |
| 38 | 6 | Chris Conte | 72.56 | 71.62 | 70.06 | 774 | Buccaneers |
| 39 | 7 | Michael Thomas | 72.26 | 63.96 | 78.31 | 153 | Dolphins |
| 40 | 8 | Da'Norris Searcy | 71.94 | 68.31 | 71.03 | 383 | Titans |
| 41 | 9 | Morgan Burnett | 71.82 | 66.09 | 74.17 | 724 | Packers |
| 42 | 10 | Shawn Williams | 71.80 | 69.75 | 72.14 | 579 | Bengals |
| 43 | 11 | Tashaun Gipson Sr. | 71.78 | 65.10 | 72.70 | 1199 | Jaguars |
| 44 | 12 | Justin Simmons | 71.32 | 68.03 | 72.86 | 736 | Broncos |
| 45 | 13 | Keanu Neal | 70.97 | 64.50 | 71.12 | 1174 | Falcons |
| 46 | 14 | Tony Jefferson | 70.94 | 65.50 | 71.04 | 1084 | Ravens |
| 47 | 15 | Adrian Colbert | 70.84 | 70.34 | 74.30 | 530 | 49ers |
| 48 | 16 | Jamal Adams | 70.78 | 65.50 | 70.14 | 1100 | Jets |
| 49 | 17 | Andre Hal | 70.64 | 64.40 | 70.84 | 939 | Texans |
| 50 | 18 | Montae Nicholson | 70.61 | 69.45 | 75.55 | 319 | Commanders |
| 51 | 19 | Daniel Sorensen | 70.53 | 71.80 | 66.35 | 1055 | Chiefs |
| 52 | 20 | Xavier Woods | 70.32 | 65.78 | 70.21 | 547 | Cowboys |
| 53 | 21 | Jahleel Addae | 70.21 | 70.30 | 69.12 | 1030 | Chargers |
| 54 | 22 | Anthony Harris | 69.44 | 65.83 | 77.88 | 254 | Vikings |
| 55 | 23 | Adrian Phillips | 69.30 | 66.36 | 68.52 | 521 | Chargers |
| 56 | 24 | Vonn Bell | 69.00 | 65.40 | 67.61 | 938 | Saints |
| 57 | 25 | Matthias Farley | 68.65 | 64.50 | 70.76 | 928 | Colts |
| 58 | 26 | Robert Golden | 68.47 | 64.38 | 69.22 | 214 | Steelers |
| 59 | 27 | Darius Butler | 67.98 | 64.28 | 66.79 | 500 | Colts |
| 60 | 28 | Clayton Fejedelem | 67.60 | 61.83 | 73.53 | 377 | Bengals |
| 61 | 29 | Marcus Gilchrist | 67.26 | 63.18 | 66.75 | 813 | Texans |
| 62 | 30 | Derrick Kindred | 67.15 | 63.93 | 67.99 | 688 | Browns |
| 63 | 31 | Mike Mitchell | 67.11 | 61.26 | 67.87 | 738 | Steelers |
| 64 | 32 | Jabrill Peppers | 66.71 | 61.68 | 69.03 | 806 | Browns |
| 65 | 33 | T.J. Ward | 66.00 | 59.10 | 69.35 | 405 | Buccaneers |
| 66 | 34 | Budda Baker | 65.74 | 57.63 | 66.98 | 515 | Cardinals |
| 67 | 35 | Reshad Jones | 65.25 | 56.10 | 70.31 | 1015 | Dolphins |
| 68 | 36 | Malik Hooker | 65.05 | 64.58 | 72.06 | 410 | Colts |
| 69 | 37 | Reggie Nelson | 64.41 | 52.90 | 67.92 | 1026 | Raiders |
| 70 | 38 | Shalom Luani | 64.33 | 58.43 | 68.26 | 187 | Raiders |
| 71 | 39 | Rafael Bush | 63.95 | 59.57 | 66.36 | 236 | Saints |
| 72 | 40 | Kemal Ishmael | 63.84 | 57.08 | 67.73 | 126 | Falcons |
| 73 | 41 | Keith Tandy | 63.60 | 58.21 | 68.13 | 226 | Buccaneers |
| 74 | 42 | Kai Nacua | 63.46 | 65.91 | 71.07 | 214 | Browns |
| 75 | 43 | Darian Stewart | 63.14 | 56.00 | 63.73 | 888 | Broncos |
| 76 | 44 | Miles Killebrew | 63.09 | 56.27 | 64.51 | 353 | Lions |
| 77 | 45 | T.J. Green | 62.27 | 53.06 | 64.62 | 382 | Colts |
| 78 | 46 | Marcus Maye | 62.19 | 53.70 | 63.69 | 1063 | Jets |

### Rotation/backup (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 79 | 1 | Ibraheim Campbell | 61.99 | 58.92 | 69.23 | 198 | Texans |
| 80 | 2 | Colin Jones | 61.89 | 62.90 | 67.25 | 213 | Panthers |
| 81 | 3 | Karl Joseph | 61.79 | 56.50 | 63.75 | 867 | Raiders |
| 82 | 4 | Jermaine Whitehead | 61.59 | 58.72 | 63.88 | 116 | Packers |
| 83 | 5 | Ron Parker | 60.98 | 56.00 | 60.14 | 1099 | Chiefs |
| 84 | 6 | T.J. McDonald | 60.86 | 57.33 | 64.24 | 539 | Dolphins |
| 85 | 7 | Deshazor Everett | 60.82 | 57.30 | 64.31 | 588 | Commanders |
| 86 | 8 | Darian Thompson | 60.71 | 57.30 | 64.29 | 1066 | Giants |
| 87 | 9 | Josh Jones | 60.18 | 52.66 | 62.06 | 730 | Packers |
| 88 | 10 | Blake Countess | 60.07 | 62.20 | 61.11 | 166 | Rams |
| 89 | 11 | Kavon Frazier | 60.07 | 62.15 | 60.90 | 226 | Cowboys |
| 90 | 12 | Marwin Evans | 59.76 | 59.32 | 63.43 | 151 | Packers |
| 91 | 13 | Kentrell Brice | 59.72 | 59.53 | 62.20 | 289 | Packers |
| 92 | 14 | Justin Evans | 59.60 | 54.88 | 61.71 | 715 | Buccaneers |
| 93 | 15 | Tavon Wilson | 59.39 | 56.62 | 60.21 | 547 | Lions |
| 94 | 16 | Jordan Richards | 59.07 | 59.20 | 59.18 | 316 | Patriots |
| 95 | 17 | Jairus Byrd | 57.72 | 55.82 | 59.09 | 135 | Panthers |
| 96 | 18 | Jimmie Ward | 57.50 | 53.87 | 61.93 | 429 | 49ers |
| 97 | 19 | J.J. Wilcox | 57.42 | 57.82 | 58.30 | 132 | Steelers |
| 98 | 20 | Clayton Geathers | 57.28 | 58.20 | 60.84 | 112 | Colts |
| 99 | 21 | Kurt Coleman | 56.93 | 45.98 | 61.95 | 772 | Panthers |
| 100 | 22 | Johnathan Cyprien | 56.89 | 49.09 | 60.42 | 743 | Titans |
| 101 | 23 | Will Parks | 55.69 | 49.20 | 56.24 | 597 | Broncos |
| 102 | 24 | Sean Davis | 54.78 | 46.70 | 56.00 | 1010 | Steelers |
| 103 | 25 | Kenny Vaccaro | 53.36 | 43.34 | 59.52 | 691 | Saints |
| 104 | 26 | Nate Allen | 52.55 | 54.26 | 57.04 | 362 | Dolphins |
| 105 | 27 | Quintin Demps | 52.38 | 50.24 | 57.24 | 177 | Bears |
| 106 | 28 | Rontez Miles | 51.57 | 49.32 | 52.13 | 125 | Jets |

## T — Tackle

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Joe Staley | 93.67 | 89.58 | 92.23 | 983 | 49ers |
| 2 | 2 | David Bakhtiari | 90.00 | 85.62 | 88.75 | 754 | Packers |
| 3 | 3 | Lane Johnson | 88.61 | 80.60 | 89.79 | 1166 | Eagles |
| 4 | 4 | Ryan Ramczyk | 87.84 | 81.40 | 87.96 | 1164 | Saints |
| 5 | 5 | Jason Peters | 87.75 | 79.20 | 89.28 | 423 | Eagles |
| 6 | 6 | Trent Williams | 87.64 | 79.75 | 88.74 | 615 | Commanders |
| 7 | 7 | Joe Thomas | 86.84 | 80.06 | 87.20 | 465 | Browns |
| 8 | 8 | Charles Leno Jr. | 86.68 | 78.70 | 87.83 | 988 | Bears |
| 9 | 9 | Andrew Whitworth | 86.56 | 80.49 | 86.44 | 983 | Rams |
| 10 | 10 | Donald Penn | 86.47 | 78.37 | 87.71 | 819 | Raiders |
| 11 | 11 | Dion Dawkins | 86.21 | 79.05 | 86.82 | 859 | Bills |
| 12 | 12 | Anthony Castonzo | 86.15 | 80.30 | 85.89 | 1030 | Colts |
| 13 | 13 | Jake Matthews | 85.72 | 80.30 | 85.17 | 1159 | Falcons |
| 14 | 14 | Tyron Smith | 84.45 | 75.66 | 86.14 | 758 | Cowboys |
| 15 | 15 | Russell Okung | 84.30 | 78.06 | 84.30 | 926 | Chargers |
| 16 | 16 | Daryl Williams | 84.02 | 78.00 | 83.86 | 1140 | Panthers |
| 17 | 17 | D.J. Humphries | 83.92 | 69.88 | 89.12 | 204 | Cardinals |
| 18 | 18 | Marcus Gilbert | 83.68 | 73.15 | 86.53 | 411 | Steelers |
| 19 | 19 | Taylor Lewan | 83.57 | 76.40 | 84.19 | 1066 | Titans |
| 20 | 20 | Nate Solder | 83.08 | 75.50 | 83.96 | 1334 | Patriots |
| 21 | 21 | Ronnie Stanley | 82.91 | 75.10 | 83.95 | 1009 | Ravens |
| 22 | 22 | Terron Armstead | 82.81 | 74.75 | 84.01 | 667 | Saints |
| 23 | 23 | Trent Brown | 82.52 | 70.71 | 86.23 | 669 | 49ers |
| 24 | 24 | Alejandro Villanueva | 82.39 | 76.10 | 82.41 | 1155 | Steelers |
| 25 | 25 | Demar Dotson | 82.14 | 72.39 | 84.47 | 715 | Buccaneers |
| 26 | 26 | Marcus Cannon | 82.05 | 71.86 | 84.68 | 478 | Patriots |
| 27 | 27 | Garett Bolles | 81.88 | 72.90 | 83.70 | 1107 | Broncos |
| 28 | 28 | Jack Conklin | 81.76 | 72.40 | 83.83 | 1099 | Titans |
| 29 | 29 | Rick Wagner | 81.52 | 74.26 | 82.19 | 792 | Lions |
| 30 | 30 | Mitchell Schwartz | 81.36 | 72.90 | 82.83 | 1083 | Chiefs |
| 31 | 31 | Jermey Parnell | 81.01 | 72.50 | 82.51 | 1078 | Jaguars |
| 32 | 32 | Duane Brown | 80.96 | 72.35 | 82.54 | 620 | Seahawks |
| 33 | 33 | Cam Fleming | 80.43 | 70.32 | 83.00 | 543 | Patriots |
| 34 | 34 | Ryan Schraeder | 80.33 | 70.54 | 82.69 | 967 | Falcons |

### Good (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Mike Remmers | 79.72 | 69.58 | 82.32 | 820 | Vikings |
| 36 | 2 | Ja'Wuan James | 79.15 | 67.42 | 82.81 | 494 | Dolphins |
| 37 | 3 | Eric Fisher | 78.21 | 69.70 | 79.72 | 1017 | Chiefs |
| 38 | 4 | Morgan Moses | 77.78 | 67.14 | 80.71 | 958 | Commanders |
| 39 | 5 | Kelvin Beachum | 77.52 | 69.30 | 78.84 | 1034 | Jets |
| 40 | 6 | Chris Hubbard | 77.51 | 67.60 | 79.95 | 847 | Steelers |
| 41 | 7 | Austin Howard | 76.82 | 66.30 | 79.66 | 1081 | Ravens |
| 42 | 8 | Cordy Glenn | 76.09 | 65.31 | 79.11 | 275 | Bills |
| 43 | 9 | Josh Wells | 75.69 | 63.80 | 79.45 | 469 | Jaguars |
| 44 | 10 | Taylor Decker | 75.65 | 64.32 | 79.04 | 471 | Lions |
| 45 | 11 | Donovan Smith | 75.49 | 64.90 | 78.38 | 1059 | Buccaneers |
| 46 | 12 | Brandon Shell | 75.43 | 63.95 | 78.92 | 696 | Jets |
| 47 | 13 | Bobby Massie | 75.30 | 64.81 | 78.13 | 912 | Bears |
| 48 | 14 | Ereck Flowers | 75.21 | 66.80 | 76.65 | 1001 | Giants |
| 49 | 15 | Riley Reiff | 74.84 | 63.60 | 78.17 | 1153 | Vikings |
| 50 | 16 | Brent Qvale | 74.35 | 61.33 | 78.86 | 394 | Jets |
| 51 | 17 | La'el Collins | 74.25 | 63.30 | 77.39 | 1065 | Cowboys |
| 52 | 18 | Jared Veldheer | 74.18 | 62.24 | 77.97 | 895 | Cardinals |
| 53 | 19 | Kyle Murphy | 74.18 | 61.44 | 78.50 | 228 | Packers |

### Starter (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 54 | 1 | Matt Kalil | 73.74 | 63.10 | 76.66 | 1138 | Panthers |
| 55 | 2 | Jason Spriggs | 73.72 | 62.13 | 77.28 | 278 | Packers |
| 56 | 3 | Jordan Mills | 73.64 | 62.10 | 77.17 | 1101 | Bills |
| 57 | 4 | John Wetzel | 73.03 | 59.71 | 77.75 | 916 | Cardinals |
| 58 | 5 | Denzelle Good | 72.90 | 60.22 | 77.18 | 293 | Colts |
| 59 | 6 | Shon Coleman | 72.87 | 61.40 | 76.35 | 1044 | Browns |
| 60 | 7 | Laremy Tunsil | 72.82 | 60.10 | 77.13 | 934 | Dolphins |
| 61 | 8 | Chad Wheeler | 72.55 | 60.26 | 76.58 | 261 | Giants |
| 62 | 9 | Sam Young | 72.31 | 60.88 | 75.77 | 451 | Dolphins |
| 63 | 10 | Ty Nsekhe | 72.26 | 60.17 | 76.15 | 305 | Commanders |
| 64 | 11 | Bryan Bulaga | 72.13 | 60.97 | 75.41 | 232 | Packers |
| 65 | 12 | Halapoulivaati Vaitai | 71.79 | 59.40 | 75.88 | 1031 | Eagles |
| 66 | 13 | LaAdrian Waddle | 71.46 | 58.63 | 75.84 | 380 | Patriots |
| 67 | 14 | Dennis Kelly | 71.24 | 59.12 | 75.15 | 234 | Titans |
| 68 | 15 | Ty Sambrailo | 71.02 | 58.51 | 75.19 | 227 | Falcons |
| 69 | 16 | Michael Schofield III | 70.87 | 58.07 | 75.23 | 407 | Chargers |
| 70 | 17 | Rashod Hill | 70.79 | 58.53 | 74.79 | 737 | Vikings |
| 71 | 18 | Kendall Lamm | 70.66 | 59.36 | 74.03 | 159 | Texans |
| 72 | 19 | Will Holden | 70.25 | 57.46 | 74.61 | 327 | Cardinals |
| 73 | 20 | Brian Mihalik | 70.22 | 56.51 | 75.19 | 192 | Lions |
| 74 | 21 | Marshall Newhouse | 70.09 | 58.06 | 73.95 | 841 | Raiders |
| 75 | 22 | Joe Barksdale | 69.78 | 54.66 | 75.70 | 657 | Chargers |
| 76 | 23 | Jake Fisher | 69.77 | 56.79 | 74.25 | 361 | Bengals |
| 77 | 24 | Vadal Alexander | 68.96 | 57.50 | 72.44 | 256 | Raiders |
| 78 | 25 | Cedric Ogbuehi | 68.87 | 57.20 | 72.49 | 667 | Bengals |
| 79 | 26 | Greg Robinson | 68.85 | 56.58 | 72.86 | 395 | Lions |
| 80 | 27 | Eric Winston | 68.49 | 55.37 | 73.07 | 201 | Bengals |
| 81 | 28 | Caleb Benenoch | 68.15 | 54.45 | 73.11 | 359 | Buccaneers |
| 82 | 29 | Donald Stephenson | 68.12 | 56.45 | 71.74 | 303 | Broncos |
| 83 | 30 | Darrell Williams | 68.00 | 60.00 | 69.17 | 124 | Rams |
| 84 | 31 | Germain Ifedi | 67.98 | 51.70 | 74.67 | 1068 | Seahawks |
| 85 | 32 | Cam Robinson | 67.58 | 52.40 | 73.54 | 1080 | Jaguars |
| 86 | 33 | Corey Robinson | 67.54 | 53.75 | 72.56 | 324 | Lions |
| 87 | 34 | Chris Clark | 67.20 | 52.37 | 72.92 | 548 | Texans |
| 88 | 35 | David Sharpe | 66.81 | 55.52 | 70.17 | 124 | Raiders |
| 89 | 36 | Julie'n Davenport | 65.95 | 55.38 | 68.83 | 238 | Texans |
| 90 | 37 | Menelik Watson | 65.57 | 51.50 | 70.79 | 448 | Broncos |
| 91 | 38 | T.J. Clemmings | 64.38 | 50.56 | 69.43 | 142 | Commanders |
| 92 | 39 | Sam Tevi | 63.81 | 58.60 | 63.12 | 135 | Chargers |
| 93 | 40 | Rees Odhiambo | 63.07 | 46.82 | 69.74 | 484 | Seahawks |
| 94 | 41 | Breno Giacomini | 62.74 | 45.60 | 70.00 | 1095 | Texans |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 87.44 | 90.50 | 81.23 | 667 | Patriots |
| 2 | 2 | Hunter Henry | 80.81 | 79.06 | 77.81 | 324 | Chargers |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 3 | 1 | Travis Kelce | 79.25 | 82.46 | 72.95 | 583 | Chiefs |
| 4 | 2 | Zach Ertz | 79.14 | 79.75 | 74.56 | 572 | Eagles |
| 5 | 3 | Delanie Walker | 77.71 | 77.25 | 73.85 | 571 | Titans |
| 6 | 4 | Greg Olsen | 75.53 | 66.51 | 77.38 | 234 | Panthers |
| 7 | 5 | Vance McDonald | 74.62 | 68.99 | 74.21 | 189 | Steelers |
| 8 | 6 | O.J. Howard | 74.20 | 60.99 | 78.84 | 329 | Buccaneers |

### Starter (64 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Jared Cook | 73.97 | 62.99 | 77.12 | 526 | Raiders |
| 10 | 2 | Jimmy Graham | 73.56 | 65.65 | 74.67 | 538 | Seahawks |
| 11 | 3 | Trey Burton | 73.10 | 67.68 | 72.55 | 217 | Eagles |
| 12 | 4 | Cameron Brate | 72.73 | 72.06 | 69.01 | 434 | Buccaneers |
| 13 | 5 | Coby Fleener | 72.62 | 62.34 | 75.31 | 201 | Saints |
| 14 | 6 | Jeff Heuerman | 72.57 | 63.19 | 74.65 | 164 | Broncos |
| 15 | 7 | Zach Miller | 72.52 | 64.80 | 73.50 | 196 | Bears |
| 16 | 8 | Darren Fells | 72.32 | 65.87 | 72.45 | 281 | Lions |
| 17 | 9 | Eric Ebron | 72.09 | 67.83 | 70.76 | 430 | Lions |
| 18 | 10 | Marcedes Lewis | 71.92 | 67.04 | 71.00 | 531 | Jaguars |
| 19 | 11 | Rhett Ellison | 71.74 | 69.22 | 69.25 | 263 | Giants |
| 20 | 12 | Garrett Celek | 71.72 | 68.59 | 69.64 | 288 | 49ers |
| 21 | 13 | David Morgan | 71.53 | 71.53 | 67.37 | 135 | Vikings |
| 22 | 14 | Jack Doyle | 71.21 | 69.98 | 67.86 | 504 | Colts |
| 23 | 15 | Niles Paul | 71.18 | 56.27 | 76.95 | 121 | Commanders |
| 24 | 16 | Evan Engram | 71.14 | 61.90 | 73.13 | 542 | Giants |
| 25 | 17 | Vernon Davis | 71.12 | 61.72 | 73.22 | 465 | Commanders |
| 26 | 18 | Eric Tomlinson | 71.09 | 60.19 | 74.19 | 154 | Jets |
| 27 | 19 | Charles Clay | 71.02 | 69.22 | 68.06 | 377 | Bills |
| 28 | 20 | Julius Thomas | 70.84 | 62.20 | 72.43 | 423 | Dolphins |
| 29 | 21 | Kyle Rudolph | 70.69 | 68.00 | 68.31 | 607 | Vikings |
| 30 | 22 | James O'Shaughnessy | 70.49 | 63.03 | 71.30 | 143 | Jaguars |
| 31 | 23 | Antonio Gates | 70.18 | 65.14 | 69.38 | 358 | Chargers |
| 32 | 24 | Jordan Reed | 70.11 | 63.42 | 70.40 | 162 | Commanders |
| 33 | 25 | Stephen Anderson | 69.90 | 59.37 | 72.75 | 325 | Texans |
| 34 | 26 | Austin Traylor | 69.86 | 59.31 | 72.72 | 159 | Broncos |
| 35 | 27 | Benjamin Watson | 69.83 | 60.63 | 71.79 | 452 | Ravens |
| 36 | 28 | Josh Hill | 69.76 | 61.13 | 71.34 | 329 | Saints |
| 37 | 29 | Jason Witten | 69.72 | 60.58 | 71.65 | 596 | Cowboys |
| 38 | 30 | Martellus Bennett | 69.53 | 59.36 | 72.14 | 288 | Patriots |
| 39 | 31 | A.J. Derby | 69.36 | 57.22 | 73.28 | 263 | Dolphins |
| 40 | 32 | George Kittle | 69.03 | 64.53 | 67.87 | 402 | 49ers |
| 41 | 33 | Nick Boyle | 68.90 | 63.59 | 68.28 | 292 | Ravens |
| 42 | 34 | Austin Seferian-Jenkins | 68.85 | 60.65 | 70.15 | 432 | Jets |
| 43 | 35 | David Njoku | 68.47 | 64.56 | 66.91 | 335 | Browns |
| 44 | 36 | Anthony Fasano | 68.35 | 59.14 | 70.33 | 263 | Dolphins |
| 45 | 37 | Ed Dickson | 68.33 | 59.35 | 70.15 | 439 | Panthers |
| 46 | 38 | Michael Hoomanawanui | 68.27 | 56.49 | 71.95 | 176 | Saints |
| 47 | 39 | Brent Celek | 68.17 | 54.71 | 72.97 | 247 | Eagles |
| 48 | 40 | Austin Hooper | 68.17 | 61.51 | 68.44 | 521 | Falcons |
| 49 | 41 | Brandon Williams | 67.88 | 59.06 | 69.60 | 121 | Colts |
| 50 | 42 | Nick O'Leary | 67.62 | 59.53 | 68.84 | 297 | Bills |
| 51 | 43 | Virgil Green | 67.38 | 56.25 | 70.63 | 227 | Broncos |
| 52 | 44 | C.J. Fiedorowicz | 67.38 | 59.87 | 68.22 | 129 | Texans |
| 53 | 45 | Dion Sims | 67.36 | 58.07 | 69.38 | 239 | Bears |
| 54 | 46 | Levine Toilolo | 67.34 | 64.10 | 65.33 | 208 | Falcons |
| 55 | 47 | Ryan Griffin | 67.34 | 59.46 | 68.43 | 158 | Texans |
| 56 | 48 | Nick Vannett | 67.32 | 59.44 | 68.40 | 144 | Seahawks |
| 57 | 49 | Luke Willson | 67.27 | 60.05 | 67.92 | 197 | Seahawks |
| 58 | 50 | Gerald Everett | 67.16 | 55.31 | 70.90 | 239 | Rams |
| 59 | 51 | Jermaine Gresham | 67.04 | 62.48 | 65.91 | 433 | Cardinals |
| 60 | 52 | Tyler Kroft | 66.95 | 60.34 | 67.19 | 464 | Bengals |
| 61 | 53 | Richard Rodgers | 66.81 | 58.15 | 68.41 | 185 | Packers |
| 62 | 54 | Seth DeValve | 66.11 | 59.39 | 66.42 | 385 | Browns |
| 63 | 55 | Lee Smith | 65.72 | 59.39 | 65.78 | 126 | Raiders |
| 64 | 56 | Lance Kendricks | 65.64 | 53.12 | 69.82 | 268 | Packers |
| 65 | 57 | Dwayne Allen | 65.62 | 56.61 | 67.46 | 253 | Patriots |
| 66 | 58 | Tyler Higbee | 65.46 | 61.91 | 63.66 | 385 | Rams |
| 67 | 59 | Troy Niklas | 65.38 | 59.51 | 65.13 | 192 | Cardinals |
| 68 | 60 | Daniel Brown | 65.18 | 57.20 | 66.34 | 184 | Bears |
| 69 | 61 | Jesse James | 64.61 | 57.47 | 65.20 | 540 | Steelers |
| 70 | 62 | Jonnu Smith | 64.43 | 56.19 | 65.75 | 234 | Titans |
| 71 | 63 | Demetrius Harris | 62.96 | 56.79 | 62.91 | 295 | Chiefs |
| 72 | 64 | Ben Koyack | 62.68 | 58.04 | 61.61 | 158 | Jaguars |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 73 | 1 | Randall Telfer | 60.13 | 54.30 | 59.85 | 137 | Browns |

## WR — Wide Receiver

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Julio Jones | 89.29 | 90.31 | 84.45 | 561 | Falcons |
| 2 | 2 | Antonio Brown | 88.78 | 90.82 | 83.26 | 627 | Steelers |
| 3 | 3 | Michael Thomas | 86.44 | 90.83 | 79.35 | 621 | Saints |
| 4 | 4 | DeAndre Hopkins | 85.06 | 89.77 | 77.75 | 629 | Texans |
| 5 | 5 | Keenan Allen | 84.82 | 88.19 | 78.41 | 579 | Chargers |
| 6 | 6 | A.J. Green | 83.19 | 80.95 | 80.51 | 542 | Bengals |
| 7 | 7 | Adam Thielen | 82.81 | 82.40 | 78.91 | 693 | Vikings |
| 8 | 8 | Doug Baldwin | 82.47 | 83.71 | 77.48 | 599 | Seahawks |
| 9 | 9 | Josh Gordon | 82.24 | 71.54 | 85.20 | 179 | Browns |
| 10 | 10 | Tyreek Hill | 81.69 | 78.20 | 79.85 | 552 | Chiefs |
| 11 | 11 | Stefon Diggs | 80.95 | 81.64 | 76.33 | 570 | Vikings |
| 12 | 12 | Chris Godwin | 80.35 | 74.11 | 80.35 | 274 | Buccaneers |
| 13 | 13 | Keelan Cole Sr. | 80.15 | 72.59 | 81.03 | 534 | Jaguars |

### Good (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Alshon Jeffery | 79.93 | 77.30 | 77.52 | 691 | Eagles |
| 15 | 2 | DeSean Jackson | 79.84 | 75.17 | 78.79 | 464 | Buccaneers |
| 16 | 3 | Marvin Jones Jr. | 79.66 | 76.30 | 77.73 | 674 | Lions |
| 17 | 4 | Mike Evans | 79.53 | 79.02 | 75.70 | 628 | Buccaneers |
| 18 | 5 | Golden Tate | 79.30 | 81.98 | 73.35 | 575 | Lions |
| 19 | 6 | JuJu Smith-Schuster | 79.27 | 71.54 | 80.26 | 512 | Steelers |
| 20 | 7 | Robert Woods | 79.16 | 76.91 | 76.49 | 428 | Rams |
| 21 | 8 | Cooper Kupp | 78.62 | 75.79 | 76.34 | 502 | Rams |
| 22 | 9 | Marquise Goodwin | 78.54 | 74.97 | 76.75 | 506 | 49ers |
| 23 | 10 | Larry Fitzgerald | 78.50 | 80.00 | 73.34 | 700 | Cardinals |
| 24 | 11 | T.Y. Hilton | 77.99 | 68.59 | 80.09 | 585 | Colts |
| 25 | 12 | Jarvis Landry | 77.97 | 79.19 | 72.99 | 625 | Dolphins |
| 26 | 13 | Pierre Garcon | 77.86 | 73.98 | 76.28 | 292 | 49ers |
| 27 | 14 | Rishard Matthews | 77.70 | 71.38 | 77.75 | 568 | Titans |
| 28 | 15 | Davante Adams | 77.65 | 79.31 | 72.37 | 538 | Packers |
| 29 | 16 | Ted Ginn Jr. | 77.49 | 70.97 | 77.67 | 499 | Saints |
| 30 | 17 | Brandin Cooks | 77.33 | 70.70 | 77.59 | 771 | Patriots |
| 31 | 18 | Danny Amendola | 76.83 | 76.49 | 72.89 | 550 | Patriots |
| 32 | 19 | Sammy Watkins | 76.67 | 68.61 | 77.87 | 526 | Rams |
| 33 | 20 | Devin Funchess | 76.48 | 71.56 | 75.59 | 593 | Panthers |
| 34 | 21 | Allen Hurns | 76.22 | 71.99 | 74.87 | 392 | Jaguars |
| 35 | 22 | Odell Beckham Jr. | 76.20 | 66.34 | 78.60 | 153 | Giants |
| 36 | 23 | Mohamed Sanu | 76.08 | 76.17 | 71.86 | 535 | Falcons |
| 37 | 24 | Kenny Golladay | 75.86 | 65.55 | 78.56 | 318 | Lions |
| 38 | 25 | Demaryius Thomas | 75.46 | 73.58 | 72.54 | 582 | Broncos |
| 39 | 26 | Marqise Lee | 75.45 | 74.20 | 72.11 | 520 | Jaguars |
| 40 | 27 | Dez Bryant | 75.22 | 72.15 | 73.10 | 568 | Cowboys |
| 41 | 28 | Martavis Bryant | 74.98 | 68.27 | 75.29 | 514 | Steelers |
| 42 | 29 | Emmanuel Sanders | 74.80 | 67.19 | 75.70 | 435 | Broncos |
| 43 | 30 | Mike Wallace | 74.70 | 68.19 | 74.87 | 476 | Ravens |
| 44 | 31 | Brenton Bersin | 74.52 | 63.35 | 77.80 | 113 | Panthers |
| 45 | 32 | Jamison Crowder | 74.44 | 70.67 | 72.78 | 466 | Commanders |
| 46 | 33 | DeVante Parker | 74.42 | 68.64 | 74.10 | 475 | Dolphins |
| 47 | 34 | Travis Benjamin | 74.32 | 65.29 | 76.17 | 400 | Chargers |
| 48 | 35 | Alex Erickson | 74.02 | 62.78 | 77.35 | 117 | Bengals |

### Starter (88 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 49 | 1 | Kendall Wright | 73.98 | 71.71 | 71.33 | 435 | Bears |
| 50 | 2 | Jordy Nelson | 73.93 | 68.10 | 73.65 | 539 | Packers |
| 51 | 3 | Kendrick Bourne | 73.76 | 61.47 | 77.79 | 212 | 49ers |
| 52 | 4 | Tyler Lockett | 73.72 | 66.80 | 74.17 | 517 | Seahawks |
| 53 | 5 | Albert Wilson | 73.67 | 70.70 | 71.49 | 397 | Chiefs |
| 54 | 6 | Nelson Agholor | 73.53 | 74.14 | 68.95 | 594 | Eagles |
| 55 | 7 | Jarius Wright | 73.43 | 67.32 | 73.33 | 234 | Vikings |
| 56 | 8 | Jermaine Kearse | 73.39 | 68.99 | 72.15 | 567 | Jets |
| 57 | 9 | Tyrell Williams | 73.36 | 61.90 | 76.83 | 553 | Chargers |
| 58 | 10 | Jeremy Maclin | 72.95 | 67.46 | 72.44 | 323 | Ravens |
| 59 | 11 | T.J. Jones | 72.87 | 69.50 | 70.95 | 287 | Lions |
| 60 | 12 | Taylor Gabriel | 72.59 | 63.17 | 74.70 | 381 | Falcons |
| 61 | 13 | Amari Cooper | 72.54 | 63.26 | 74.56 | 471 | Raiders |
| 62 | 14 | Michael Crabtree | 72.48 | 70.48 | 69.65 | 411 | Raiders |
| 63 | 15 | Sterling Shepard | 72.35 | 68.97 | 70.43 | 464 | Giants |
| 64 | 16 | Ryan Grant | 72.31 | 67.18 | 71.56 | 402 | Commanders |
| 65 | 17 | Paul Richardson Jr. | 72.25 | 66.55 | 71.89 | 595 | Seahawks |
| 66 | 18 | Chris Hogan | 72.15 | 61.17 | 75.31 | 531 | Patriots |
| 67 | 19 | Brice Butler | 72.02 | 62.89 | 73.94 | 158 | Cowboys |
| 68 | 20 | Terrance Williams | 71.88 | 64.78 | 72.44 | 461 | Cowboys |
| 69 | 21 | Cody Latimer | 71.82 | 66.11 | 71.46 | 246 | Broncos |
| 70 | 22 | Deonte Thompson | 71.82 | 63.01 | 73.52 | 462 | Bills |
| 71 | 23 | Eric Decker | 71.65 | 68.75 | 69.41 | 516 | Titans |
| 72 | 24 | Willie Snead IV | 71.63 | 60.14 | 75.12 | 176 | Saints |
| 73 | 25 | Kenny Stills | 71.52 | 59.90 | 75.10 | 649 | Dolphins |
| 74 | 26 | Will Fuller V | 71.39 | 65.03 | 71.47 | 322 | Texans |
| 75 | 27 | Josh Doctson | 71.16 | 62.11 | 73.03 | 470 | Commanders |
| 76 | 28 | Dontrelle Inman | 71.12 | 64.77 | 71.18 | 286 | Bears |
| 77 | 29 | Brandon Marshall | 71.08 | 61.58 | 73.24 | 183 | Giants |
| 78 | 30 | Randall Cobb | 70.99 | 64.04 | 71.46 | 491 | Packers |
| 79 | 31 | Tavarres King | 70.93 | 61.45 | 73.09 | 262 | Giants |
| 80 | 32 | J.J. Nelson | 70.73 | 59.24 | 74.23 | 386 | Cardinals |
| 81 | 33 | Kenny Britt | 70.56 | 57.00 | 75.44 | 277 | Patriots |
| 82 | 34 | Jaron Brown | 70.52 | 59.44 | 73.74 | 566 | Cardinals |
| 83 | 35 | Josh Bellamy | 70.49 | 62.48 | 71.67 | 282 | Bears |
| 84 | 36 | John Brown | 70.44 | 58.34 | 74.34 | 340 | Cardinals |
| 85 | 37 | Cordarrelle Patterson | 70.07 | 63.63 | 70.20 | 288 | Raiders |
| 86 | 38 | Louis Murphy Jr. | 70.00 | 61.49 | 71.50 | 107 | 49ers |
| 87 | 39 | Phillip Dorsett | 69.85 | 58.76 | 73.07 | 255 | Patriots |
| 88 | 40 | Adam Humphries | 69.58 | 62.69 | 70.00 | 529 | Buccaneers |
| 89 | 41 | ArDarius Stewart | 69.58 | 58.74 | 72.64 | 116 | Jets |
| 90 | 42 | Brandon Tate | 69.54 | 61.80 | 70.53 | 102 | Bills |
| 91 | 43 | Brandon Coleman | 69.54 | 56.52 | 74.05 | 394 | Saints |
| 92 | 44 | Jordan Taylor | 69.52 | 61.43 | 70.74 | 173 | Broncos |
| 93 | 45 | Markus Wheaton | 69.44 | 55.56 | 74.52 | 125 | Bears |
| 94 | 46 | Corey Davis | 69.34 | 62.60 | 69.67 | 436 | Titans |
| 95 | 47 | Taywan Taylor | 69.32 | 57.81 | 72.83 | 189 | Titans |
| 96 | 48 | Dede Westbrook | 69.32 | 61.47 | 70.39 | 322 | Jaguars |
| 97 | 49 | Mack Hollins | 69.11 | 61.92 | 69.74 | 198 | Eagles |
| 98 | 50 | Chris Conley | 68.97 | 61.61 | 69.71 | 179 | Chiefs |
| 99 | 51 | Trent Taylor | 68.97 | 66.37 | 66.54 | 379 | 49ers |
| 100 | 52 | Cole Beasley | 68.61 | 59.91 | 70.24 | 409 | Cowboys |
| 101 | 53 | De'Anthony Thomas | 68.59 | 60.88 | 69.56 | 117 | Chiefs |
| 102 | 54 | Terrelle Pryor Sr. | 68.44 | 58.39 | 70.97 | 247 | Commanders |
| 103 | 55 | Aldrick Robinson | 68.37 | 57.34 | 71.56 | 278 | 49ers |
| 104 | 56 | Donte Moncrief | 68.25 | 60.96 | 68.95 | 403 | Colts |
| 105 | 57 | Johnny Holton | 68.15 | 54.71 | 72.95 | 142 | Raiders |
| 106 | 58 | Brandon LaFell | 68.06 | 57.91 | 70.66 | 553 | Bengals |
| 107 | 59 | Mike Williams | 68.03 | 54.96 | 72.58 | 145 | Chargers |
| 108 | 60 | Tyler Boyd | 67.86 | 61.15 | 68.17 | 219 | Bengals |
| 109 | 61 | Damiere Byrd | 67.45 | 59.92 | 68.31 | 105 | Panthers |
| 110 | 62 | Chester Rogers | 67.28 | 58.75 | 68.80 | 284 | Colts |
| 111 | 63 | Corey Coleman | 66.84 | 60.27 | 67.05 | 311 | Browns |
| 112 | 64 | Bruce Ellington | 66.75 | 57.79 | 68.56 | 359 | Texans |
| 113 | 65 | Torrey Smith | 66.72 | 56.89 | 69.11 | 542 | Eagles |
| 114 | 66 | Bennie Fowler | 65.94 | 56.97 | 67.76 | 379 | Broncos |
| 115 | 67 | Jordan Matthews | 65.94 | 54.00 | 69.74 | 327 | Bills |
| 116 | 68 | Justin Hardy | 65.92 | 59.75 | 65.86 | 251 | Falcons |
| 117 | 69 | Roger Lewis Jr. | 65.89 | 58.82 | 66.43 | 468 | Giants |
| 118 | 70 | Chris Moore | 65.52 | 59.39 | 65.44 | 246 | Ravens |
| 119 | 71 | Geronimo Allison | 65.36 | 54.06 | 68.72 | 281 | Packers |
| 120 | 72 | Eli Rogers | 65.26 | 56.08 | 67.21 | 302 | Steelers |
| 121 | 73 | Andre Holmes | 64.90 | 61.84 | 62.77 | 205 | Bills |
| 122 | 74 | Chad Hansen | 64.85 | 56.19 | 66.45 | 242 | Jets |
| 123 | 75 | Josh Malone | 64.75 | 55.46 | 66.77 | 161 | Bengals |
| 124 | 76 | Josh Reynolds | 64.74 | 59.20 | 64.27 | 164 | Rams |
| 125 | 77 | Rashard Higgins | 64.52 | 51.61 | 68.96 | 504 | Browns |
| 126 | 78 | Travis Rudolph | 64.46 | 54.09 | 67.21 | 163 | Giants |
| 127 | 79 | Michael Campanaro | 64.40 | 60.36 | 62.93 | 179 | Ravens |
| 128 | 80 | Kaelin Clay | 64.37 | 57.38 | 64.87 | 188 | Panthers |
| 129 | 81 | Laquon Treadwell | 63.99 | 55.16 | 65.71 | 344 | Vikings |
| 130 | 82 | Ricardo Louis | 63.56 | 54.59 | 65.38 | 406 | Browns |
| 131 | 83 | Braxton Miller | 63.54 | 56.94 | 63.77 | 268 | Texans |
| 132 | 84 | Breshad Perriman | 63.35 | 51.32 | 67.20 | 224 | Ravens |
| 133 | 85 | Zay Jones | 63.33 | 55.56 | 64.34 | 520 | Bills |
| 134 | 86 | Seth Roberts | 63.19 | 54.71 | 64.67 | 514 | Raiders |
| 135 | 87 | Demarcus Robinson | 62.89 | 55.66 | 63.55 | 411 | Chiefs |
| 136 | 88 | Leonte Carroo | 62.84 | 57.96 | 61.93 | 101 | Dolphins |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 137 | 1 | Russell Shepard | 61.78 | 50.37 | 65.22 | 312 | Panthers |
| 138 | 2 | Kamar Aiken | 61.15 | 48.94 | 65.12 | 372 | Colts |
| 139 | 3 | Curtis Samuel | 60.46 | 58.57 | 57.56 | 140 | Panthers |
