# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:26:55Z
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
| 5 | 5 | Ali Marpet | 89.80 | 81.50 | 91.16 | 723 | Buccaneers |
| 6 | 6 | David Andrews | 89.59 | 83.50 | 89.49 | 1207 | Patriots |
| 7 | 7 | Rodney Hudson | 84.62 | 76.20 | 86.07 | 1007 | Raiders |
| 8 | 8 | Weston Richburg | 83.19 | 72.70 | 86.01 | 241 | Giants |
| 9 | 9 | Matt Paradis | 82.80 | 74.90 | 83.90 | 1128 | Broncos |
| 10 | 10 | Ryan Jensen | 81.74 | 72.30 | 83.86 | 1085 | Ravens |
| 11 | 11 | Maurkice Pouncey | 81.64 | 72.80 | 83.37 | 1114 | Steelers |
| 12 | 12 | Ben Jones | 81.24 | 71.60 | 83.50 | 1153 | Titans |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Eric Wood | 79.83 | 70.90 | 81.62 | 1130 | Bills |
| 14 | 2 | John Sullivan | 79.65 | 70.50 | 81.58 | 926 | Rams |
| 15 | 3 | Brett Jones | 79.18 | 70.20 | 81.00 | 966 | Giants |
| 16 | 4 | J.C. Tretter | 77.56 | 69.10 | 79.03 | 1068 | Browns |
| 17 | 5 | Corey Linsley | 77.21 | 67.70 | 79.38 | 1047 | Packers |
| 18 | 6 | Daniel Kilgore | 77.00 | 66.80 | 79.64 | 1098 | 49ers |
| 19 | 7 | Russell Bodine | 76.90 | 66.80 | 79.47 | 962 | Bengals |
| 20 | 8 | Justin Britt | 76.84 | 65.50 | 80.23 | 1062 | Seahawks |
| 21 | 9 | Pat Elflein | 76.25 | 66.60 | 78.52 | 1081 | Vikings |
| 22 | 10 | Tyler Larsen | 76.24 | 67.20 | 78.10 | 720 | Panthers |
| 23 | 11 | Nick Martin | 74.63 | 64.80 | 77.01 | 971 | Texans |
| 24 | 12 | Mike Pouncey | 74.46 | 62.10 | 78.53 | 971 | Dolphins |

### Starter (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 25 | 1 | Max Unger | 73.70 | 63.10 | 76.60 | 1164 | Saints |
| 26 | 2 | Mitch Morse | 73.43 | 61.90 | 76.95 | 383 | Chiefs |
| 27 | 3 | Spencer Long | 72.48 | 61.20 | 75.84 | 397 | Commanders |
| 28 | 4 | A.Q. Shipley | 72.46 | 61.30 | 75.73 | 1120 | Cardinals |
| 29 | 5 | B.J. Finney | 72.04 | 66.40 | 71.63 | 235 | Steelers |
| 30 | 6 | Chase Roullier | 71.84 | 64.40 | 72.64 | 457 | Commanders |
| 31 | 7 | Travis Swanson | 70.08 | 58.30 | 73.77 | 710 | Lions |
| 32 | 8 | Ryan Kelly | 68.89 | 60.40 | 70.39 | 394 | Colts |
| 33 | 9 | Ryan Kalil | 68.07 | 56.70 | 71.49 | 412 | Panthers |
| 34 | 10 | Joe Hawley | 67.91 | 54.80 | 72.48 | 203 | Buccaneers |
| 35 | 11 | Spencer Pulley | 66.25 | 53.60 | 70.52 | 1054 | Chargers |
| 36 | 12 | Wesley Johnson | 65.89 | 51.70 | 71.18 | 938 | Jets |
| 37 | 13 | Austin Blythe | 63.91 | 51.90 | 67.75 | 197 | Rams |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 38 | 1 | Hroniss Grasu | 57.33 | 36.70 | 66.91 | 259 | Bears |
| 39 | 2 | Jonotthan Harrison | 57.03 | 37.20 | 66.08 | 102 | Jets |

## CB — Cornerback

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Casey Hayward Jr. | 93.79 | 92.10 | 90.75 | 1003 | Chargers |
| 2 | 2 | Tre'Davious White | 93.72 | 90.10 | 91.97 | 1149 | Bills |
| 3 | 3 | Jalen Ramsey | 93.36 | 91.60 | 90.37 | 1212 | Jaguars |
| 4 | 4 | William Jackson III | 93.25 | 90.40 | 93.06 | 698 | Bengals |
| 5 | 5 | Marshon Lattimore | 93.18 | 87.90 | 93.57 | 900 | Saints |
| 6 | 6 | Kendall Fuller | 90.40 | 91.20 | 87.27 | 720 | Commanders |
| 7 | 7 | Stephon Gilmore | 89.58 | 87.20 | 88.15 | 1027 | Patriots |
| 8 | 8 | Patrick Robinson | 89.42 | 89.90 | 87.75 | 853 | Eagles |
| 9 | 9 | A.J. Bouye | 88.28 | 84.70 | 87.14 | 1229 | Jaguars |
| 10 | 10 | Marcus Peters | 87.69 | 81.70 | 88.03 | 1034 | Chiefs |
| 11 | 11 | Tramon Williams | 86.73 | 83.80 | 87.54 | 666 | Cardinals |
| 12 | 12 | Aqib Talib | 86.39 | 81.60 | 86.88 | 753 | Broncos |
| 13 | 13 | Jimmy Smith | 85.74 | 84.10 | 86.32 | 601 | Ravens |
| 14 | 14 | Nickell Robey-Coleman | 85.73 | 83.30 | 83.19 | 676 | Rams |
| 15 | 15 | Darius Slay | 85.01 | 80.00 | 84.81 | 1064 | Lions |
| 16 | 16 | Trevor Williams | 83.79 | 80.20 | 84.75 | 1004 | Chargers |
| 17 | 17 | Mike Hilton | 83.73 | 81.90 | 80.78 | 592 | Steelers |
| 18 | 18 | Bradley Roby | 83.36 | 77.50 | 83.10 | 674 | Broncos |
| 19 | 19 | Briean Boddy-Calhoun | 82.82 | 77.50 | 84.93 | 535 | Browns |
| 20 | 20 | Quinton Dunbar | 82.71 | 79.30 | 84.57 | 373 | Commanders |
| 21 | 21 | E.J. Gaines | 82.45 | 80.60 | 83.37 | 711 | Bills |
| 22 | 22 | Kyle Fuller | 82.14 | 79.00 | 80.07 | 1017 | Bears |
| 23 | 23 | Chidobe Awuzie | 82.13 | 78.90 | 87.42 | 309 | Cowboys |
| 24 | 24 | Chris Harris Jr. | 82.09 | 75.80 | 82.11 | 869 | Broncos |
| 25 | 25 | Ronald Darby | 81.77 | 78.10 | 83.48 | 580 | Eagles |
| 26 | 26 | Richard Sherman | 81.58 | 74.10 | 86.05 | 572 | Seahawks |
| 27 | 27 | Robert Alford | 81.53 | 74.50 | 82.25 | 1171 | Falcons |
| 28 | 28 | Marlon Humphrey | 81.48 | 75.30 | 81.44 | 597 | Ravens |
| 29 | 29 | Adoree' Jackson | 81.47 | 75.50 | 81.29 | 1154 | Titans |
| 30 | 30 | Bobby McCain | 80.98 | 78.90 | 79.45 | 664 | Dolphins |
| 31 | 31 | T.J. Carrie | 80.65 | 75.00 | 80.76 | 1023 | Raiders |
| 32 | 32 | Brent Grimes | 80.53 | 75.20 | 81.68 | 844 | Buccaneers |

### Good (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Desmond Trufant | 79.98 | 72.80 | 82.78 | 1030 | Falcons |
| 34 | 2 | Artie Burns | 79.92 | 73.60 | 79.96 | 1028 | Steelers |
| 35 | 3 | Ken Crawley | 79.54 | 74.10 | 80.81 | 967 | Saints |
| 36 | 4 | Rashaan Melvin | 79.47 | 75.90 | 83.73 | 552 | Colts |
| 37 | 5 | Ross Cockrell | 79.21 | 74.10 | 79.49 | 679 | Giants |
| 38 | 6 | Malcolm Butler | 79.03 | 69.30 | 81.35 | 1175 | Patriots |
| 39 | 7 | Xavier Rhodes | 78.89 | 71.00 | 80.62 | 1031 | Vikings |
| 40 | 8 | Justin Coleman | 78.66 | 76.10 | 80.57 | 654 | Seahawks |
| 41 | 9 | Jason McCourty | 78.50 | 73.50 | 81.83 | 899 | Browns |
| 42 | 10 | Josh Norman | 78.22 | 69.30 | 81.03 | 902 | Commanders |
| 43 | 11 | Cre'Von LeBlanc | 78.02 | 73.00 | 83.59 | 212 | Bears |
| 44 | 12 | Ahkello Witherspoon | 78.00 | 73.70 | 80.86 | 660 | 49ers |
| 45 | 13 | Bryce Callahan | 77.95 | 77.30 | 79.54 | 512 | Bears |
| 46 | 14 | Jourdan Lewis | 77.88 | 70.20 | 79.87 | 746 | Cowboys |
| 47 | 15 | Bashaud Breeland | 77.73 | 72.00 | 78.53 | 856 | Commanders |
| 48 | 16 | K'Waun Williams | 77.58 | 73.80 | 78.54 | 632 | 49ers |
| 49 | 17 | Johnathan Joseph | 77.57 | 71.00 | 78.10 | 746 | Texans |
| 50 | 18 | Darqueze Dennard | 76.69 | 72.40 | 76.95 | 900 | Bengals |
| 51 | 19 | Prince Amukamara | 76.66 | 71.50 | 78.63 | 849 | Bears |
| 52 | 20 | Trumaine Johnson | 76.61 | 68.40 | 78.95 | 1005 | Rams |
| 53 | 21 | Dominique Rodgers-Cromartie | 76.35 | 69.00 | 77.81 | 604 | Giants |
| 54 | 22 | Patrick Peterson | 75.83 | 68.40 | 76.61 | 1013 | Cardinals |
| 55 | 23 | Byron Maxwell | 75.38 | 69.80 | 79.94 | 580 | Seahawks |
| 56 | 24 | Aaron Colvin | 75.03 | 74.40 | 73.16 | 833 | Jaguars |
| 57 | 25 | Joe Haden | 74.81 | 70.50 | 78.84 | 673 | Steelers |
| 58 | 26 | Brandon Carr | 74.55 | 66.20 | 75.95 | 1023 | Ravens |
| 59 | 27 | Shaquill Griffin | 74.43 | 63.00 | 78.92 | 876 | Seahawks |
| 60 | 28 | Janoris Jenkins | 74.01 | 69.60 | 76.63 | 619 | Giants |

### Starter (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 61 | 1 | Kenny Moore II | 73.38 | 69.30 | 80.26 | 384 | Colts |
| 62 | 2 | Darryl Roberts | 73.10 | 66.70 | 78.28 | 468 | Jets |
| 63 | 3 | Jonathan Jones | 73.03 | 68.20 | 76.64 | 446 | Patriots |
| 64 | 4 | Rasul Douglas | 72.89 | 67.20 | 76.69 | 424 | Eagles |
| 65 | 5 | Sean Smith | 72.75 | 63.90 | 75.73 | 701 | Raiders |
| 66 | 6 | Troy Hill | 72.69 | 73.10 | 75.95 | 338 | Rams |
| 67 | 7 | Logan Ryan | 72.67 | 62.00 | 75.61 | 1048 | Titans |
| 68 | 8 | Terence Newman | 72.45 | 63.90 | 74.30 | 611 | Vikings |
| 69 | 9 | Xavien Howard | 71.43 | 63.60 | 75.61 | 1016 | Dolphins |
| 70 | 10 | Steven Nelson | 70.83 | 67.70 | 73.53 | 579 | Chiefs |
| 71 | 11 | Quincy Wilson | 70.44 | 68.30 | 78.56 | 402 | Colts |
| 72 | 12 | Trae Waynes | 70.40 | 61.80 | 73.32 | 1044 | Vikings |
| 73 | 13 | Davon House | 70.18 | 60.00 | 75.81 | 658 | Packers |
| 74 | 14 | William Gay | 70.17 | 62.70 | 70.98 | 271 | Steelers |
| 75 | 15 | Robert McClain | 70.07 | 65.60 | 73.66 | 690 | Buccaneers |
| 76 | 16 | Anthony Brown | 69.88 | 58.70 | 73.16 | 846 | Cowboys |
| 77 | 17 | Dre Kirkpatrick | 69.87 | 61.80 | 72.44 | 868 | Bengals |
| 78 | 18 | Terrance Mitchell | 69.64 | 61.10 | 77.61 | 705 | Chiefs |
| 79 | 19 | Buster Skrine | 69.56 | 61.40 | 71.98 | 1010 | Jets |
| 80 | 20 | Shareece Wright | 69.43 | 66.20 | 71.79 | 456 | Bills |
| 81 | 21 | Kayvon Webster | 69.04 | 62.30 | 74.05 | 550 | Rams |
| 82 | 22 | Daryl Worley | 68.83 | 59.20 | 71.47 | 777 | Panthers |
| 83 | 23 | P.J. Williams | 68.64 | 62.00 | 74.36 | 740 | Saints |
| 84 | 24 | Captain Munnerlyn | 68.19 | 60.40 | 70.05 | 404 | Panthers |
| 85 | 25 | Vernon Hargreaves III | 67.62 | 64.60 | 70.02 | 502 | Buccaneers |
| 86 | 26 | Kareem Jackson | 67.62 | 57.90 | 70.57 | 868 | Texans |
| 87 | 27 | Darrelle Revis | 67.42 | 59.40 | 74.53 | 238 | Chiefs |
| 88 | 28 | LeShaun Sims | 67.28 | 64.80 | 70.24 | 428 | Titans |
| 89 | 29 | Eli Apple | 67.19 | 59.00 | 72.79 | 649 | Giants |
| 90 | 30 | Marcus Williams | 67.16 | 57.40 | 73.47 | 188 | Texans |
| 91 | 31 | Pierre Desir | 67.02 | 59.10 | 76.27 | 375 | Colts |
| 92 | 32 | Tye Smith | 66.79 | 64.10 | 69.62 | 228 | Titans |
| 93 | 33 | D.J. Hayden | 66.36 | 58.40 | 69.07 | 489 | Lions |
| 94 | 34 | Leonard Johnson | 66.21 | 60.80 | 70.02 | 705 | Bills |
| 95 | 35 | Dexter McDonald | 66.09 | 63.00 | 71.80 | 534 | Raiders |
| 96 | 36 | Morris Claiborne | 65.94 | 58.30 | 70.93 | 919 | Jets |
| 97 | 37 | Nate Hairston | 65.93 | 57.40 | 69.54 | 537 | Colts |
| 98 | 38 | Mike Jordan | 65.65 | 62.60 | 72.77 | 210 | Browns |
| 99 | 39 | Eric Rowe | 65.40 | 56.80 | 72.07 | 427 | Patriots |
| 100 | 40 | James Bradberry | 65.34 | 55.30 | 69.03 | 1044 | Panthers |
| 101 | 41 | Adam Jones | 65.20 | 56.90 | 70.42 | 299 | Bengals |
| 102 | 42 | Jamar Taylor | 64.98 | 62.70 | 64.00 | 966 | Browns |
| 103 | 43 | Kenneth Acker | 64.77 | 57.10 | 71.14 | 210 | Chiefs |
| 104 | 44 | Maurice Canady | 64.54 | 62.30 | 67.06 | 320 | Ravens |
| 105 | 45 | Darryl Morris | 63.78 | 59.90 | 72.00 | 114 | Giants |
| 106 | 46 | Lardarius Webb | 63.65 | 51.00 | 68.12 | 374 | Ravens |
| 107 | 47 | Justin Bethel | 63.26 | 50.90 | 70.88 | 456 | Cardinals |
| 108 | 48 | Mackensie Alexander | 62.89 | 53.30 | 67.85 | 385 | Vikings |
| 109 | 49 | Cordrea Tankersley | 62.54 | 59.40 | 65.67 | 638 | Dolphins |
| 110 | 50 | Johnson Bademosi | 62.39 | 56.80 | 70.91 | 224 | Patriots |

### Rotation/backup (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 111 | 1 | Alterraun Verner | 61.94 | 54.70 | 70.00 | 157 | Dolphins |
| 112 | 2 | Kevon Seymour | 61.64 | 46.80 | 69.06 | 317 | Panthers |
| 113 | 3 | Nevin Lawson | 61.55 | 51.90 | 65.39 | 555 | Lions |
| 114 | 4 | Lenzy Pipkins | 60.75 | 60.80 | 72.51 | 122 | Packers |
| 115 | 5 | Dontae Johnson | 60.63 | 48.90 | 66.37 | 1026 | 49ers |
| 116 | 6 | Brice McCain | 60.47 | 47.20 | 66.09 | 416 | Titans |
| 117 | 7 | Leon Hall | 60.45 | 51.40 | 67.10 | 205 | 49ers |
| 118 | 8 | Rashard Robinson | 60.39 | 46.90 | 69.90 | 499 | Jets |
| 119 | 9 | Ryan Smith | 59.66 | 58.80 | 58.66 | 598 | Buccaneers |
| 120 | 10 | Coty Sensabaugh | 59.50 | 52.50 | 67.08 | 242 | Steelers |
| 121 | 11 | Sterling Moore | 59.45 | 50.70 | 67.27 | 108 | Saints |
| 122 | 12 | Javien Elliott | 59.39 | 63.80 | 60.49 | 130 | Buccaneers |
| 123 | 13 | Quinten Rollins | 59.34 | 50.70 | 67.18 | 139 | Packers |
| 124 | 14 | Orlando Scandrick | 58.59 | 48.90 | 64.44 | 614 | Cowboys |
| 125 | 15 | Keith Reaser | 57.97 | 55.70 | 67.81 | 100 | Chiefs |
| 126 | 16 | Kevin King | 57.29 | 47.40 | 67.02 | 380 | Packers |
| 127 | 17 | Jeremy Lane | 57.20 | 40.90 | 67.65 | 346 | Seahawks |
| 128 | 18 | Marcus Cooper | 57.14 | 41.50 | 67.47 | 246 | Bears |
| 129 | 19 | David Amerson | 56.91 | 40.40 | 69.16 | 287 | Raiders |
| 130 | 20 | Josh Hawkins | 56.66 | 50.70 | 62.20 | 402 | Packers |
| 131 | 21 | Juston Burris | 56.33 | 47.80 | 64.89 | 334 | Jets |
| 132 | 22 | Phillip Gaines | 53.69 | 40.80 | 65.51 | 420 | Chiefs |
| 133 | 23 | Brandon Dixon | 52.89 | 47.70 | 66.58 | 257 | Giants |
| 134 | 24 | Cameron Sutton | 51.67 | 58.00 | 61.79 | 113 | Steelers |
| 135 | 25 | Kevin Johnson | 50.86 | 34.00 | 63.14 | 579 | Texans |
| 136 | 26 | Brendan Langley | 45.00 | 27.00 | 47.85 | 107 | Broncos |

## DI — Defensive Interior

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 93.61 | 89.92 | 92.42 | 856 | Rams |
| 2 | 2 | J.J. Watt | 90.23 | 88.04 | 97.33 | 217 | Texans |
| 3 | 3 | Kawann Short | 89.13 | 87.10 | 86.32 | 743 | Panthers |
| 4 | 4 | Geno Atkins | 88.53 | 86.49 | 85.73 | 754 | Bengals |
| 5 | 5 | Jurrell Casey | 86.82 | 88.40 | 81.91 | 988 | Titans |
| 6 | 6 | Leonard Williams | 86.32 | 88.58 | 80.65 | 877 | Jets |
| 7 | 7 | Ndamukong Suh | 85.58 | 86.61 | 80.72 | 877 | Dolphins |
| 8 | 8 | Malik Jackson | 85.47 | 84.27 | 82.11 | 922 | Jaguars |
| 9 | 9 | Damon Harrison Sr. | 85.08 | 84.24 | 81.47 | 644 | Giants |
| 10 | 10 | Larry Ogunjobi | 84.87 | 82.80 | 84.17 | 300 | Browns |
| 11 | 11 | Fletcher Cox | 84.83 | 88.76 | 78.04 | 783 | Eagles |
| 12 | 12 | Michael Pierce | 84.64 | 83.83 | 81.01 | 594 | Ravens |
| 13 | 13 | Mike Daniels | 84.56 | 82.37 | 82.89 | 629 | Packers |
| 14 | 14 | Kenny Clark | 84.06 | 86.97 | 78.61 | 684 | Packers |
| 15 | 15 | Linval Joseph | 83.44 | 85.84 | 78.31 | 759 | Vikings |
| 16 | 16 | Vincent Taylor | 83.32 | 84.41 | 82.59 | 185 | Dolphins |
| 17 | 17 | Akiem Hicks | 83.22 | 81.12 | 80.46 | 900 | Bears |
| 18 | 18 | Olsen Pierre | 82.92 | 62.61 | 94.38 | 351 | Cardinals |
| 19 | 19 | Marcell Dareus | 82.73 | 84.50 | 80.08 | 540 | Jaguars |
| 20 | 20 | DeForest Buckner | 82.66 | 88.08 | 75.26 | 867 | 49ers |
| 21 | 21 | David Irving | 81.88 | 81.76 | 82.80 | 338 | Cowboys |
| 22 | 22 | Gerald McCoy | 81.80 | 87.08 | 75.14 | 807 | Buccaneers |
| 23 | 23 | Cameron Heyward | 81.58 | 83.10 | 79.21 | 835 | Steelers |
| 24 | 24 | Malcom Brown | 80.53 | 77.84 | 78.15 | 682 | Patriots |
| 25 | 25 | Grady Jarrett | 80.23 | 73.75 | 80.58 | 896 | Falcons |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | DJ Reader | 78.75 | 80.43 | 74.77 | 526 | Texans |
| 27 | 2 | Stephon Tuitt | 78.62 | 81.13 | 74.34 | 627 | Steelers |
| 28 | 3 | Johnathan Hankins | 78.54 | 77.48 | 77.07 | 687 | Colts |
| 29 | 4 | Michael Brockers | 78.35 | 80.03 | 73.70 | 751 | Rams |
| 30 | 5 | Dean Lowry | 78.28 | 72.64 | 77.88 | 493 | Packers |
| 31 | 6 | Muhammad Wilkerson | 78.12 | 68.36 | 82.64 | 698 | Jets |
| 32 | 7 | Brandon Williams | 77.82 | 74.22 | 78.13 | 475 | Ravens |
| 33 | 8 | Jonathan Allen | 77.66 | 85.15 | 84.46 | 159 | Commanders |
| 34 | 9 | Sheldon Richardson | 77.48 | 69.01 | 81.04 | 654 | Seahawks |
| 35 | 10 | Danny Shelton | 77.38 | 77.93 | 73.88 | 469 | Browns |
| 36 | 11 | Timmy Jernigan | 77.17 | 69.75 | 78.15 | 586 | Eagles |
| 37 | 12 | Steve McLendon | 76.22 | 67.50 | 79.43 | 488 | Jets |
| 38 | 13 | Javon Hargrave | 76.00 | 72.12 | 74.42 | 478 | Steelers |
| 39 | 14 | Henry Anderson | 75.55 | 77.37 | 76.83 | 380 | Colts |
| 40 | 15 | Eddie Goldman | 75.24 | 73.47 | 76.11 | 608 | Bears |
| 41 | 16 | Bennie Logan | 75.02 | 62.17 | 80.77 | 617 | Chiefs |
| 42 | 17 | Corey Liuget | 74.95 | 74.02 | 74.53 | 415 | Chargers |
| 43 | 18 | Shelby Harris | 74.91 | 73.52 | 77.61 | 516 | Broncos |
| 44 | 19 | Abry Jones | 74.89 | 68.97 | 75.18 | 579 | Jaguars |
| 45 | 20 | Lawrence Guy Sr. | 74.23 | 62.15 | 78.12 | 715 | Patriots |
| 46 | 21 | Dalvin Tomlinson | 74.18 | 77.82 | 67.59 | 588 | Giants |
| 47 | 22 | Derek Wolfe | 74.12 | 69.21 | 76.66 | 458 | Broncos |

### Starter (78 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 48 | 1 | Sheldon Day | 73.43 | 65.16 | 77.38 | 320 | 49ers |
| 49 | 2 | Chris Baker | 73.15 | 59.87 | 78.35 | 455 | Buccaneers |
| 50 | 3 | Christian Covington | 72.84 | 75.46 | 71.61 | 166 | Texans |
| 51 | 4 | Kyle Williams | 72.82 | 60.64 | 79.17 | 801 | Bills |
| 52 | 5 | Stacy McGee | 71.87 | 70.00 | 70.83 | 432 | Commanders |
| 53 | 6 | Zach Kerr | 71.59 | 64.26 | 76.99 | 249 | Broncos |
| 54 | 7 | Mike Pennel | 71.30 | 58.91 | 75.59 | 301 | Jets |
| 55 | 8 | Al Woods | 71.02 | 62.61 | 74.13 | 564 | Colts |
| 56 | 9 | A'Shawn Robinson | 71.00 | 64.05 | 71.47 | 735 | Lions |
| 57 | 10 | Justin Ellis | 70.85 | 70.57 | 67.71 | 462 | Raiders |
| 58 | 11 | Karl Klug | 70.79 | 58.85 | 75.22 | 363 | Titans |
| 59 | 12 | DaQuan Jones | 70.77 | 69.42 | 69.59 | 436 | Titans |
| 60 | 13 | Hassan Ridgeway | 70.56 | 63.39 | 73.13 | 177 | Colts |
| 61 | 14 | Caleb Brantley | 70.49 | 58.09 | 78.76 | 217 | Browns |
| 62 | 15 | Denico Autry | 70.40 | 58.92 | 74.30 | 594 | Raiders |
| 63 | 16 | Dontari Poe | 70.33 | 68.02 | 67.71 | 880 | Falcons |
| 64 | 17 | Mario Edwards Jr. | 70.28 | 63.85 | 75.92 | 475 | Raiders |
| 65 | 18 | Sylvester Williams | 70.23 | 65.91 | 68.95 | 410 | Titans |
| 66 | 19 | Eli Ankou | 70.21 | 59.69 | 80.35 | 172 | Jaguars |
| 67 | 20 | Datone Jones | 70.09 | 58.07 | 78.84 | 204 | Cowboys |
| 68 | 21 | Jarran Reed | 69.67 | 62.11 | 71.20 | 616 | Seahawks |
| 69 | 22 | Star Lotulelei | 69.10 | 56.98 | 73.02 | 621 | Panthers |
| 70 | 23 | Beau Allen | 69.09 | 56.89 | 73.05 | 496 | Eagles |
| 71 | 24 | Vernon Butler | 68.69 | 65.01 | 69.98 | 312 | Panthers |
| 72 | 25 | Brent Urban | 68.58 | 62.86 | 77.08 | 123 | Ravens |
| 73 | 26 | Ricky Jean Francois | 68.56 | 56.30 | 73.09 | 206 | Patriots |
| 74 | 27 | Xavier Williams | 68.46 | 64.91 | 74.26 | 249 | Cardinals |
| 75 | 28 | Jonathan Bullard | 68.29 | 58.11 | 71.69 | 437 | Bears |
| 76 | 29 | Rodney Gunter | 68.20 | 59.69 | 69.71 | 291 | Cardinals |
| 77 | 30 | Corey Peters | 67.67 | 61.34 | 70.33 | 442 | Cardinals |
| 78 | 31 | Adolphus Washington | 67.51 | 53.85 | 72.83 | 535 | Bills |
| 79 | 32 | Sheldon Rankins | 67.44 | 65.27 | 67.46 | 928 | Saints |
| 80 | 33 | Adam Gotsis | 67.32 | 56.25 | 70.54 | 555 | Broncos |
| 81 | 34 | David Onyemata | 67.30 | 62.26 | 66.50 | 696 | Saints |
| 82 | 35 | Carl Davis Jr. | 67.29 | 56.70 | 72.00 | 300 | Ravens |
| 83 | 36 | Roy Robertson-Harris | 67.15 | 55.91 | 74.65 | 212 | Bears |
| 84 | 37 | Austin Johnson | 66.79 | 62.59 | 67.77 | 391 | Titans |
| 85 | 38 | Tyson Alualu | 66.63 | 58.67 | 68.40 | 451 | Steelers |
| 86 | 39 | Mitch Unrein | 66.53 | 61.09 | 69.22 | 389 | Bears |
| 87 | 40 | Carlos Watkins | 66.30 | 56.27 | 72.99 | 328 | Texans |
| 88 | 41 | Willie Henry | 66.23 | 52.94 | 73.00 | 596 | Ravens |
| 89 | 42 | Haloti Ngata | 66.03 | 59.47 | 73.00 | 145 | Lions |
| 90 | 43 | Alan Branch | 65.97 | 51.54 | 73.50 | 274 | Patriots |
| 91 | 44 | Jamie Meder | 65.94 | 58.84 | 69.64 | 178 | Browns |
| 92 | 45 | Domata Peko Sr. | 65.93 | 53.95 | 70.78 | 460 | Broncos |
| 93 | 46 | Tony McDaniel | 65.78 | 55.13 | 74.35 | 100 | Saints |
| 94 | 47 | Ahtyba Rubin | 65.77 | 53.78 | 71.16 | 174 | Falcons |
| 95 | 48 | Tom Johnson | 65.73 | 49.11 | 73.28 | 786 | Vikings |
| 96 | 49 | Davon Godchaux | 65.67 | 51.03 | 72.29 | 500 | Dolphins |
| 97 | 50 | Akeem Spence | 65.66 | 52.20 | 70.47 | 662 | Lions |
| 98 | 51 | John Jenkins | 65.40 | 61.06 | 70.58 | 109 | Bears |
| 99 | 52 | Darius Philon | 65.34 | 54.96 | 70.38 | 509 | Chargers |
| 100 | 53 | Margus Hunt | 65.33 | 48.99 | 72.06 | 578 | Colts |
| 101 | 54 | Ethan Westbrooks | 65.22 | 49.11 | 75.69 | 359 | Rams |
| 102 | 55 | Tyrunn Walker | 65.09 | 50.25 | 73.31 | 333 | Rams |
| 103 | 56 | Frostee Rucker | 64.91 | 43.44 | 76.20 | 607 | Cardinals |
| 104 | 57 | Ryan Glasgow | 64.79 | 56.97 | 65.83 | 412 | Bengals |
| 105 | 58 | Brandon Mebane | 64.79 | 49.74 | 72.54 | 535 | Chargers |
| 106 | 59 | Rakeem Nunez-Roches | 64.76 | 50.61 | 73.58 | 390 | Chiefs |
| 107 | 60 | Cedric Thornton | 64.50 | 44.59 | 74.85 | 404 | Bills |
| 108 | 61 | Pat Sims | 64.29 | 46.80 | 74.29 | 304 | Bengals |
| 109 | 62 | Adam Butler | 64.23 | 51.12 | 68.81 | 524 | Patriots |
| 110 | 63 | Clinton McDonald | 64.20 | 46.95 | 75.90 | 460 | Buccaneers |
| 111 | 64 | Allen Bailey | 63.88 | 52.42 | 71.72 | 683 | Chiefs |
| 112 | 65 | Earl Mitchell | 63.79 | 49.37 | 71.93 | 622 | 49ers |
| 113 | 66 | Jordan Phillips | 63.69 | 55.73 | 66.60 | 401 | Dolphins |
| 114 | 67 | Quinton Dial | 63.61 | 51.05 | 70.21 | 309 | Packers |
| 115 | 68 | Xavier Cooper | 63.54 | 55.42 | 67.70 | 305 | Jets |
| 116 | 69 | Grover Stewart | 63.52 | 57.73 | 64.24 | 258 | Colts |
| 117 | 70 | Treyvon Hester | 63.33 | 60.59 | 63.07 | 346 | Raiders |
| 118 | 71 | Nazair Jones | 63.16 | 59.05 | 66.94 | 284 | Seahawks |
| 119 | 72 | Brandon Dunn | 63.06 | 59.84 | 64.69 | 416 | Texans |
| 120 | 73 | Matt Ioannidis | 62.94 | 60.03 | 64.74 | 584 | Commanders |
| 121 | 74 | Chris Jones | 62.91 | 44.27 | 72.11 | 698 | Chiefs |
| 122 | 75 | Tyeler Davison | 62.79 | 58.42 | 61.86 | 666 | Saints |
| 123 | 76 | Sealver Siliga | 62.78 | 48.65 | 74.28 | 118 | Buccaneers |
| 124 | 77 | Jay Bromley | 62.65 | 54.50 | 63.92 | 424 | Giants |
| 125 | 78 | D.J. Jones | 62.22 | 51.76 | 72.32 | 147 | 49ers |

### Rotation/backup (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 126 | 1 | Trevon Coley | 61.66 | 51.45 | 65.34 | 656 | Browns |
| 127 | 2 | Angelo Blackson | 61.65 | 54.83 | 66.62 | 193 | Texans |
| 128 | 3 | Jarvis Jenkins | 60.91 | 49.89 | 66.37 | 249 | Chiefs |
| 129 | 4 | John Hughes | 60.83 | 52.80 | 69.32 | 158 | Saints |
| 130 | 5 | Shamar Stephen | 60.70 | 53.83 | 63.39 | 388 | Vikings |
| 131 | 6 | Eddie Vanderdoes | 60.65 | 48.95 | 64.28 | 464 | Raiders |
| 132 | 7 | Maliek Collins | 60.33 | 47.62 | 64.64 | 684 | Cowboys |
| 133 | 8 | Stephen Paea | 60.20 | 46.87 | 73.15 | 145 | Cowboys |
| 134 | 9 | Robert Thomas | 59.84 | 47.57 | 67.24 | 236 | Giants |
| 135 | 10 | Richard Ash | 59.18 | 47.51 | 73.43 | 233 | Cowboys |
| 136 | 11 | A.J. Francis | 59.04 | 53.74 | 69.99 | 164 | Commanders |
| 137 | 12 | Kyle Love | 59.02 | 41.54 | 68.39 | 398 | Panthers |
| 138 | 13 | Terrell McClain | 58.95 | 45.82 | 68.53 | 328 | Commanders |
| 139 | 14 | Joel Heath | 58.50 | 47.93 | 64.12 | 323 | Texans |
| 140 | 15 | Anthony Lanier II | 58.49 | 50.18 | 67.82 | 339 | Commanders |
| 141 | 16 | Destiny Vaeao | 58.05 | 44.34 | 65.63 | 242 | Eagles |
| 142 | 17 | Josh Mauro | 58.01 | 45.09 | 68.32 | 334 | Cardinals |
| 143 | 18 | Damion Square | 57.86 | 52.37 | 61.00 | 362 | Chargers |
| 144 | 19 | Brian Price | 57.86 | 48.74 | 66.54 | 150 | Cowboys |
| 145 | 20 | Jeremiah Ledbetter | 57.44 | 50.76 | 57.73 | 349 | Lions |
| 146 | 21 | David King | 56.24 | 48.50 | 67.65 | 121 | Titans |
| 147 | 22 | Tanzel Smart | 55.82 | 43.31 | 60.00 | 319 | Rams |
| 148 | 23 | Robert Nkemdiche | 55.60 | 53.60 | 59.66 | 252 | Cardinals |
| 149 | 24 | Jack Crawford | 55.43 | 38.34 | 70.47 | 101 | Falcons |
| 150 | 25 | Andrew Billings | 54.93 | 41.73 | 60.60 | 334 | Bengals |
| 151 | 26 | Garrison Smith | 54.45 | 49.65 | 63.77 | 115 | Seahawks |
| 152 | 27 | Christian Ringo | 54.06 | 53.24 | 59.69 | 130 | Lions |
| 153 | 28 | Quinton Jefferson | 54.03 | 49.64 | 64.38 | 129 | Seahawks |
| 154 | 29 | Jihad Ward | 53.81 | 48.40 | 60.42 | 125 | Raiders |
| 155 | 30 | Justin Hamilton | 49.14 | 45.65 | 63.27 | 100 | Chiefs |
| 156 | 31 | Chris Wormley | 49.10 | 50.60 | 54.80 | 120 | Ravens |
| 157 | 32 | Elijah Qualls | 46.63 | 49.56 | 53.92 | 103 | Eagles |

## ED — Edge

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 94.43 | 95.25 | 89.72 | 846 | Broncos |
| 2 | 2 | Joey Bosa | 92.73 | 96.81 | 87.41 | 850 | Chargers |
| 3 | 3 | Khalil Mack | 91.58 | 97.27 | 83.62 | 930 | Raiders |
| 4 | 4 | Brandon Graham | 88.03 | 92.18 | 81.09 | 823 | Eagles |
| 5 | 5 | Justin Houston | 87.56 | 85.65 | 88.42 | 1015 | Chiefs |
| 6 | 6 | DeMarcus Lawrence | 85.70 | 94.92 | 77.27 | 701 | Cowboys |
| 7 | 7 | DeMarcus Walker | 85.36 | 90.48 | 84.03 | 100 | Broncos |
| 8 | 8 | Danielle Hunter | 84.85 | 85.37 | 80.53 | 873 | Vikings |
| 9 | 9 | Ezekiel Ansah | 83.78 | 81.49 | 82.81 | 516 | Lions |
| 10 | 10 | Myles Garrett | 83.67 | 91.79 | 79.29 | 518 | Browns |
| 11 | 11 | Melvin Ingram III | 83.50 | 84.26 | 78.82 | 890 | Chargers |
| 12 | 12 | Cameron Jordan | 83.38 | 93.13 | 72.71 | 1136 | Saints |
| 13 | 13 | Chandler Jones | 83.30 | 85.94 | 77.38 | 1043 | Cardinals |
| 14 | 14 | Ryan Kerrigan | 82.14 | 74.60 | 83.00 | 820 | Commanders |
| 15 | 15 | Cameron Wake | 81.84 | 71.42 | 86.50 | 610 | Dolphins |
| 16 | 16 | Calais Campbell | 81.69 | 67.41 | 87.05 | 984 | Jaguars |
| 17 | 17 | Michael Bennett | 80.76 | 83.54 | 75.68 | 931 | Seahawks |
| 18 | 18 | Carlos Dunlap | 80.20 | 78.59 | 77.10 | 876 | Bengals |
| 19 | 19 | Jadeveon Clowney | 80.09 | 93.15 | 67.85 | 895 | Texans |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Frank Clark | 79.57 | 76.00 | 77.79 | 740 | Seahawks |
| 21 | 2 | T.J. Watt | 79.44 | 66.85 | 83.66 | 809 | Steelers |
| 22 | 3 | Shaquil Barrett | 79.41 | 80.11 | 74.77 | 664 | Broncos |
| 23 | 4 | Takk McKinley | 79.38 | 72.14 | 80.04 | 464 | Falcons |
| 24 | 5 | Jabaal Sheard | 78.64 | 83.63 | 71.34 | 900 | Colts |
| 25 | 6 | Everson Griffen | 78.58 | 77.08 | 75.42 | 910 | Vikings |
| 26 | 7 | Pernell McPhee | 78.47 | 70.11 | 84.04 | 385 | Bears |
| 27 | 8 | Olivier Vernon | 78.42 | 82.19 | 73.82 | 698 | Giants |
| 28 | 9 | Robert Quinn | 77.51 | 77.72 | 77.05 | 696 | Rams |
| 29 | 10 | Derek Barnett | 77.47 | 78.23 | 72.80 | 506 | Eagles |
| 30 | 11 | Nick Perry | 77.37 | 76.16 | 76.10 | 542 | Packers |
| 31 | 12 | Carl Lawson | 77.23 | 62.78 | 82.70 | 477 | Bengals |
| 32 | 13 | Whitney Mercilus | 77.09 | 72.29 | 81.85 | 203 | Texans |
| 33 | 14 | William Hayes | 76.93 | 80.37 | 74.22 | 271 | Dolphins |
| 34 | 15 | Junior Galette | 75.78 | 66.53 | 77.78 | 407 | Commanders |
| 35 | 16 | Mario Addison | 75.67 | 64.78 | 79.39 | 692 | Panthers |
| 36 | 17 | Vinny Curry | 75.29 | 72.05 | 73.29 | 702 | Eagles |
| 37 | 18 | Yannick Ngakoue | 75.28 | 67.19 | 76.51 | 920 | Jaguars |
| 38 | 19 | Leonard Floyd | 75.15 | 64.61 | 83.48 | 582 | Bears |
| 39 | 20 | Jerry Hughes | 75.05 | 72.26 | 72.75 | 781 | Bills |
| 40 | 21 | Trey Flowers | 74.66 | 72.52 | 75.05 | 994 | Patriots |
| 41 | 22 | James Harrison | 74.49 | 55.50 | 86.64 | 193 | Patriots |
| 42 | 23 | Robert Ayers | 74.21 | 76.77 | 72.51 | 588 | Buccaneers |
| 43 | 24 | Chris Long | 74.14 | 67.95 | 74.94 | 588 | Eagles |

### Starter (52 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 44 | 1 | Terrell Suggs | 73.99 | 65.53 | 78.89 | 845 | Ravens |
| 45 | 2 | Clay Matthews | 73.56 | 61.50 | 78.78 | 656 | Packers |
| 46 | 3 | Vic Beasley Jr. | 73.07 | 66.98 | 72.97 | 556 | Falcons |
| 47 | 4 | Derrick Morgan | 73.04 | 62.05 | 77.76 | 827 | Titans |
| 48 | 5 | Elvis Dumervil | 72.88 | 54.35 | 83.57 | 341 | 49ers |
| 49 | 6 | Lorenzo Alexander | 72.59 | 49.05 | 86.62 | 719 | Bills |
| 50 | 7 | Brian Orakpo | 72.57 | 59.53 | 77.09 | 938 | Titans |
| 51 | 8 | Dion Jordan | 72.32 | 75.48 | 77.53 | 135 | Seahawks |
| 52 | 9 | Markus Golden | 72.08 | 62.82 | 80.34 | 231 | Cardinals |
| 53 | 10 | Dante Fowler Jr. | 71.51 | 68.67 | 69.24 | 581 | Jaguars |
| 54 | 11 | Aaron Lynch | 71.42 | 67.33 | 77.89 | 157 | 49ers |
| 55 | 12 | Matthew Judon | 71.35 | 58.07 | 76.82 | 787 | Ravens |
| 56 | 13 | Jason Pierre-Paul | 71.14 | 70.53 | 70.29 | 1010 | Giants |
| 57 | 14 | Julius Peppers | 70.96 | 57.80 | 75.57 | 531 | Panthers |
| 58 | 15 | Charles Johnson | 70.95 | 58.17 | 79.15 | 389 | Panthers |
| 59 | 16 | Cliff Avril | 70.30 | 57.16 | 81.14 | 151 | Seahawks |
| 60 | 17 | Alex Okafor | 69.96 | 65.72 | 72.69 | 486 | Saints |
| 61 | 18 | Tyus Bowser | 69.64 | 60.84 | 71.34 | 162 | Ravens |
| 62 | 19 | Bruce Irvin | 68.90 | 51.15 | 76.57 | 880 | Raiders |
| 63 | 20 | Preston Smith | 68.58 | 60.95 | 69.50 | 754 | Commanders |
| 64 | 21 | Dont'a Hightower | 68.22 | 50.18 | 86.31 | 237 | Patriots |
| 65 | 22 | Chris McCain | 67.93 | 56.67 | 71.78 | 242 | Chargers |
| 66 | 23 | Shane Ray | 67.82 | 59.05 | 73.66 | 354 | Broncos |
| 67 | 24 | Adrian Clayborn | 67.67 | 65.48 | 65.60 | 632 | Falcons |
| 68 | 25 | Jordan Jenkins | 67.49 | 60.79 | 68.57 | 715 | Jets |
| 69 | 26 | Barkevious Mingo | 66.41 | 60.93 | 68.72 | 503 | Colts |
| 70 | 27 | Kyler Fackrell | 66.37 | 56.66 | 69.46 | 446 | Packers |
| 71 | 28 | Charles Harris | 66.14 | 64.38 | 63.14 | 496 | Dolphins |
| 72 | 29 | John Simon | 65.99 | 59.01 | 71.68 | 472 | Colts |
| 73 | 30 | Deatrich Wise Jr. | 65.96 | 60.92 | 65.15 | 593 | Patriots |
| 74 | 31 | Willie Young | 65.84 | 53.75 | 76.19 | 119 | Bears |
| 75 | 32 | Tarell Basham | 65.68 | 57.73 | 67.84 | 222 | Colts |
| 76 | 33 | Samson Ebukam | 65.49 | 60.20 | 64.85 | 359 | Rams |
| 77 | 34 | Dee Ford | 65.22 | 56.82 | 71.85 | 316 | Chiefs |
| 78 | 35 | Connor Barwin | 65.00 | 49.56 | 71.64 | 722 | Rams |
| 79 | 36 | Anthony Zettel | 64.41 | 59.76 | 64.13 | 753 | Lions |
| 80 | 37 | Chris Smith | 64.20 | 60.25 | 67.87 | 401 | Bengals |
| 81 | 38 | Lamarr Houston | 63.55 | 53.71 | 73.44 | 377 | Bears |
| 82 | 39 | Hau'oli Kikaha | 63.50 | 54.13 | 67.66 | 209 | Saints |
| 83 | 40 | Bud Dupree | 63.46 | 57.36 | 65.24 | 850 | Steelers |
| 84 | 41 | Devon Kennard | 63.44 | 53.77 | 66.23 | 543 | Giants |
| 85 | 42 | Erik Walden | 63.19 | 46.24 | 70.53 | 644 | Titans |
| 86 | 43 | Taco Charlton | 63.10 | 59.35 | 61.44 | 399 | Cowboys |
| 87 | 44 | David Bass | 63.06 | 60.84 | 62.55 | 352 | Jets |
| 88 | 45 | Matt Longacre | 63.04 | 61.77 | 66.17 | 377 | Rams |
| 89 | 46 | Emmanuel Ogbah | 62.98 | 61.52 | 63.69 | 462 | Browns |
| 90 | 47 | Brooks Reed | 62.72 | 56.47 | 63.35 | 460 | Falcons |
| 91 | 48 | Ryan Davis Sr. | 62.53 | 55.27 | 65.80 | 489 | Bills |
| 92 | 49 | Za'Darius Smith | 62.38 | 59.89 | 62.05 | 533 | Ravens |
| 93 | 50 | Kony Ealy | 62.21 | 57.32 | 61.82 | 451 | Jets |
| 94 | 51 | Marcus Smith | 62.20 | 54.24 | 66.15 | 252 | Seahawks |
| 95 | 52 | Jordan Willis | 62.12 | 62.54 | 57.67 | 361 | Bengals |

### Rotation/backup (48 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 96 | 1 | Tanoh Kpassagnon | 61.81 | 55.21 | 68.30 | 158 | Chiefs |
| 97 | 2 | Michael Johnson | 61.62 | 56.41 | 61.44 | 685 | Bengals |
| 98 | 3 | Trey Hendrickson | 61.42 | 61.48 | 59.30 | 344 | Saints |
| 99 | 4 | Sam Acho | 61.31 | 53.48 | 62.57 | 639 | Bears |
| 100 | 5 | Cassius Marsh | 61.23 | 60.04 | 58.38 | 455 | 49ers |
| 101 | 6 | Cornelius Washington | 61.04 | 52.81 | 62.88 | 488 | Lions |
| 102 | 7 | Solomon Thomas | 60.85 | 62.72 | 57.52 | 696 | 49ers |
| 103 | 8 | Shaq Lawson | 60.83 | 60.21 | 62.67 | 436 | Bills |
| 104 | 9 | Ahmad Brooks | 60.66 | 47.30 | 67.90 | 346 | Packers |
| 105 | 10 | Tyrone Crawford | 60.52 | 54.97 | 60.37 | 626 | Cowboys |
| 106 | 11 | Brian Robison | 60.27 | 46.24 | 65.45 | 642 | Vikings |
| 107 | 12 | Dwight Freeney | 60.07 | 42.48 | 71.89 | 227 | Lions |
| 108 | 13 | Tamba Hali | 60.01 | 50.50 | 67.38 | 116 | Chiefs |
| 109 | 14 | Wes Horton | 59.89 | 54.58 | 60.83 | 385 | Panthers |
| 110 | 15 | Anthony Chickillo | 59.64 | 55.09 | 62.57 | 272 | Steelers |
| 111 | 16 | Ronald Blair III | 59.64 | 57.91 | 63.14 | 140 | 49ers |
| 112 | 17 | Ufomba Kamalu | 59.63 | 54.61 | 63.36 | 185 | Texans |
| 113 | 18 | Tim Williams | 59.52 | 59.40 | 63.76 | 125 | Ravens |
| 114 | 19 | James Cowser | 59.48 | 54.81 | 62.32 | 155 | Raiders |
| 115 | 20 | Eddie Yarbrough | 59.47 | 58.54 | 55.92 | 494 | Bills |
| 116 | 21 | Dawuane Smoot | 59.45 | 59.29 | 55.39 | 286 | Jaguars |
| 117 | 22 | Andre Branch | 59.09 | 54.34 | 59.75 | 561 | Dolphins |
| 118 | 23 | Carl Nassib | 58.67 | 57.77 | 55.89 | 643 | Browns |
| 119 | 24 | Benson Mayowa | 58.63 | 59.50 | 56.38 | 381 | Cowboys |
| 120 | 25 | Will Clarke | 58.44 | 55.34 | 57.49 | 315 | Buccaneers |
| 121 | 26 | Eric Lee | 58.40 | 53.69 | 64.68 | 355 | Patriots |
| 122 | 27 | William Gholston | 58.40 | 56.19 | 57.37 | 448 | Buccaneers |
| 123 | 28 | Josh Martin | 57.81 | 54.58 | 59.97 | 489 | Jets |
| 124 | 29 | Arik Armstead | 57.80 | 58.54 | 61.34 | 304 | 49ers |
| 125 | 30 | Nate Orchard | 57.44 | 56.32 | 58.29 | 431 | Browns |
| 126 | 31 | Kareem Martin | 57.17 | 57.37 | 56.52 | 458 | Cardinals |
| 127 | 32 | Noah Spence | 56.94 | 57.97 | 58.61 | 246 | Buccaneers |
| 128 | 33 | Shilique Calhoun | 56.77 | 58.34 | 59.11 | 103 | Raiders |
| 129 | 34 | Brennan Scarlett | 56.62 | 58.67 | 58.25 | 302 | Texans |
| 130 | 35 | Kasim Edebali | 56.60 | 49.88 | 59.00 | 103 | Saints |
| 131 | 36 | Kerry Wynn | 56.40 | 55.10 | 54.45 | 252 | Giants |
| 132 | 37 | Branden Jackson | 56.18 | 56.86 | 59.24 | 263 | Seahawks |
| 133 | 38 | Vince Biegel | 55.53 | 58.78 | 57.53 | 121 | Packers |
| 134 | 39 | Lewis Neal | 55.13 | 57.76 | 60.08 | 140 | Cowboys |
| 135 | 40 | Avery Moss | 54.88 | 60.77 | 53.03 | 248 | Giants |
| 136 | 41 | Kevin Dodd | 54.76 | 58.78 | 53.90 | 106 | Titans |
| 137 | 42 | George Johnson | 54.35 | 47.96 | 59.12 | 269 | Saints |
| 138 | 43 | Ryan Anderson | 54.27 | 55.55 | 51.33 | 194 | Commanders |
| 139 | 44 | Terrence Fede | 54.15 | 54.14 | 53.32 | 173 | Dolphins |
| 140 | 45 | Frank Zombo | 53.73 | 45.77 | 55.71 | 639 | Chiefs |
| 141 | 46 | Ryan Russell | 52.11 | 54.33 | 53.13 | 456 | Buccaneers |
| 142 | 47 | Bryan Cox Jr. | 51.37 | 54.22 | 53.64 | 145 | Panthers |
| 143 | 48 | Cameron Malveaux | 45.00 | 56.75 | 50.34 | 107 | Dolphins |

## G — Guard

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | David DeCastro | 93.97 | 89.00 | 93.12 | 1125 | Steelers |
| 2 | 2 | Zack Martin | 93.23 | 89.00 | 91.88 | 1018 | Cowboys |
| 3 | 3 | Brandon Brooks | 90.72 | 86.20 | 89.56 | 1279 | Eagles |
| 4 | 4 | Marshal Yanda | 88.41 | 82.60 | 88.12 | 102 | Ravens |
| 5 | 5 | Shaq Mason | 87.90 | 81.60 | 87.94 | 1356 | Patriots |
| 6 | 6 | Rodger Saffold | 86.83 | 80.80 | 86.69 | 1010 | Rams |
| 7 | 7 | John Greco | 86.57 | 77.20 | 88.65 | 105 | Giants |
| 8 | 8 | Brandon Scherff | 86.42 | 79.80 | 86.67 | 867 | Commanders |
| 9 | 9 | Josh Sitton | 84.73 | 76.70 | 85.91 | 712 | Bears |
| 10 | 10 | Andy Levitre | 83.33 | 74.60 | 84.99 | 699 | Falcons |
| 11 | 11 | Andrew Norwell | 83.26 | 76.30 | 83.73 | 1140 | Panthers |
| 12 | 12 | Kelechi Osemele | 83.17 | 76.50 | 83.45 | 1006 | Raiders |
| 13 | 13 | Trai Turner | 81.72 | 73.80 | 82.83 | 918 | Panthers |
| 14 | 14 | Larry Warford | 81.66 | 74.00 | 82.60 | 951 | Saints |
| 15 | 15 | Joe Berger | 81.50 | 74.00 | 82.33 | 1259 | Vikings |
| 16 | 16 | Richie Incognito | 81.29 | 74.40 | 81.72 | 1109 | Bills |
| 17 | 17 | Laurent Duvernay-Tardif | 81.23 | 71.50 | 83.55 | 688 | Chiefs |
| 18 | 18 | Ron Leary | 81.02 | 72.60 | 82.46 | 712 | Broncos |
| 19 | 19 | Joe Thuney | 80.97 | 74.40 | 81.18 | 1354 | Patriots |
| 20 | 20 | Jon Feliciano | 80.94 | 73.50 | 81.73 | 124 | Raiders |
| 21 | 21 | Joel Bitonio | 80.94 | 75.40 | 80.47 | 1068 | Browns |
| 22 | 22 | Kyle Long | 80.31 | 71.40 | 82.08 | 447 | Bears |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Kevin Zeitler | 79.76 | 73.20 | 79.96 | 1068 | Browns |
| 24 | 2 | Graham Glasgow | 78.97 | 70.60 | 80.38 | 1042 | Lions |
| 25 | 3 | Brandon Fusco | 78.69 | 69.40 | 80.71 | 1083 | 49ers |
| 26 | 4 | Jahri Evans | 78.50 | 70.40 | 79.74 | 912 | Packers |
| 27 | 5 | Joe Dahl | 78.34 | 70.60 | 79.34 | 182 | Lions |
| 28 | 6 | Clint Boling | 77.82 | 69.20 | 79.40 | 962 | Bengals |
| 29 | 7 | Josh Kline | 77.60 | 69.50 | 78.84 | 1152 | Titans |
| 30 | 8 | Jonathan Cooper | 77.43 | 68.60 | 79.15 | 835 | Cowboys |
| 31 | 9 | Ben Garland | 76.82 | 65.10 | 80.47 | 476 | Falcons |
| 32 | 10 | T.J. Lang | 76.57 | 67.90 | 78.19 | 809 | Lions |
| 33 | 11 | Quinton Spain | 76.46 | 66.90 | 78.67 | 1007 | Titans |
| 34 | 12 | Gabe Jackson | 76.43 | 67.30 | 78.35 | 887 | Raiders |
| 35 | 13 | Andrus Peat | 76.41 | 68.30 | 77.65 | 932 | Saints |
| 36 | 14 | John Jerry | 75.91 | 66.10 | 78.29 | 958 | Giants |
| 37 | 15 | Laken Tomlinson | 75.90 | 66.40 | 78.07 | 1042 | 49ers |
| 38 | 16 | Lane Taylor | 75.78 | 68.20 | 76.67 | 939 | Packers |
| 39 | 17 | Vladimir Ducasse | 75.58 | 66.70 | 77.34 | 874 | Bills |
| 40 | 18 | Wes Schweitzer | 74.10 | 63.60 | 76.93 | 1149 | Falcons |

### Starter (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 41 | 1 | Evan Boehm | 73.97 | 61.80 | 77.92 | 588 | Cardinals |
| 42 | 2 | Dakota Dozier | 73.66 | 63.20 | 76.46 | 248 | Jets |
| 43 | 3 | Patrick Omameh | 73.51 | 62.90 | 76.41 | 1052 | Jaguars |
| 44 | 4 | D.J. Fluker | 72.95 | 60.30 | 77.22 | 446 | Giants |
| 45 | 5 | Chad Slade | 72.94 | 63.60 | 75.00 | 163 | Texans |
| 46 | 6 | John Miller | 72.55 | 58.60 | 77.69 | 256 | Bills |
| 47 | 7 | A.J. Cann | 72.26 | 61.70 | 75.13 | 1228 | Jaguars |
| 48 | 8 | Dan Feeney | 72.23 | 60.80 | 75.69 | 665 | Chargers |
| 49 | 9 | Senio Kelemete | 72.10 | 62.00 | 74.66 | 748 | Saints |
| 50 | 10 | Chance Warmack | 72.00 | 61.80 | 74.64 | 321 | Eagles |
| 51 | 11 | Allen Barbre | 71.39 | 58.30 | 75.95 | 553 | Broncos |
| 52 | 12 | Kevin Pamphile | 71.07 | 59.70 | 74.49 | 782 | Buccaneers |
| 53 | 13 | James Carpenter | 70.97 | 59.30 | 74.59 | 1035 | Jets |
| 54 | 14 | Kenny Wiggins | 70.74 | 60.10 | 73.67 | 1040 | Chargers |
| 55 | 15 | Luke Joeckel | 69.89 | 58.70 | 73.19 | 702 | Seahawks |
| 56 | 16 | Ramon Foster | 69.88 | 60.20 | 72.17 | 1006 | Steelers |
| 57 | 17 | J.R. Sweezy | 69.87 | 59.30 | 72.75 | 903 | Buccaneers |
| 58 | 18 | Anthony Steen | 69.60 | 61.30 | 70.97 | 327 | Dolphins |
| 59 | 19 | Brian Winters | 69.22 | 56.60 | 73.46 | 807 | Jets |
| 60 | 20 | Max Garcia | 69.17 | 58.40 | 72.19 | 869 | Broncos |
| 61 | 21 | Jack Mewhort | 69.09 | 57.00 | 72.98 | 313 | Colts |
| 62 | 22 | Tyler Shatley | 68.55 | 56.00 | 72.75 | 386 | Jaguars |
| 63 | 23 | Trey Hopkins | 68.22 | 60.10 | 69.47 | 707 | Bengals |
| 64 | 24 | Mark Glowinski | 68.11 | 52.80 | 74.15 | 199 | Colts |
| 65 | 25 | Arie Kouandjio | 68.01 | 55.80 | 71.98 | 424 | Commanders |
| 66 | 26 | Jordan Devey | 67.81 | 56.20 | 71.38 | 147 | Chiefs |
| 67 | 27 | Bryan Witzmann | 67.55 | 53.90 | 72.48 | 933 | Chiefs |
| 68 | 28 | Matt Slauson | 67.52 | 54.80 | 71.84 | 424 | Chargers |
| 69 | 29 | Alex Boone | 67.35 | 55.90 | 70.82 | 874 | Cardinals |
| 70 | 30 | Lucas Patrick | 67.16 | 53.90 | 71.83 | 227 | Packers |
| 71 | 31 | Shawn Lauvao | 66.73 | 53.30 | 71.52 | 531 | Commanders |
| 72 | 32 | Oday Aboushi | 65.82 | 52.60 | 70.46 | 558 | Seahawks |
| 73 | 33 | Jermon Bushrod | 65.71 | 52.10 | 70.61 | 604 | Dolphins |
| 74 | 34 | Xavier Su'a-Filo | 65.67 | 53.70 | 69.48 | 1075 | Texans |
| 75 | 35 | Jeremiah Sirles | 65.49 | 52.00 | 70.32 | 366 | Vikings |
| 76 | 36 | Ted Larsen | 65.20 | 50.50 | 70.84 | 521 | Dolphins |
| 77 | 37 | Jeff Allen | 64.90 | 50.20 | 70.53 | 728 | Texans |
| 78 | 38 | Jermaine Eluemunor | 64.62 | 49.70 | 70.40 | 198 | Ravens |
| 79 | 39 | Amini Silatolu | 63.75 | 50.20 | 68.62 | 247 | Panthers |
| 80 | 40 | Connor McGovern | 63.59 | 48.90 | 69.22 | 418 | Broncos |
| 81 | 41 | Alex Redmond | 63.16 | 53.40 | 65.50 | 104 | Bengals |
| 82 | 42 | Cameron Erving | 62.59 | 47.10 | 68.75 | 276 | Chiefs |
| 83 | 43 | Jon Halapio | 62.46 | 56.80 | 62.06 | 403 | Giants |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 84 | 1 | Isaac Seumalo | 61.37 | 46.30 | 67.25 | 308 | Eagles |
| 85 | 2 | Zane Beadles | 60.28 | 44.80 | 66.44 | 395 | 49ers |
| 86 | 3 | Danny Isidora | 57.10 | 43.40 | 62.07 | 147 | Vikings |

## HB — Running Back

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Alvin Kamara | 91.33 | 90.40 | 87.79 | 343 | Saints |
| 2 | 2 | Kareem Hunt | 81.64 | 82.10 | 77.16 | 328 | Chiefs |
| 3 | 3 | Dion Lewis | 80.95 | 85.60 | 73.68 | 206 | Patriots |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Kenyan Drake | 79.81 | 74.40 | 79.25 | 264 | Dolphins |
| 5 | 2 | Austin Ekeler | 79.75 | 74.60 | 79.01 | 122 | Chargers |
| 6 | 3 | Alex Collins | 77.39 | 83.40 | 69.22 | 108 | Ravens |
| 7 | 4 | Ty Montgomery | 77.23 | 72.50 | 76.21 | 148 | Packers |
| 8 | 5 | Aaron Jones | 76.56 | 78.80 | 70.90 | 109 | Packers |
| 9 | 6 | Duke Johnson Jr. | 76.55 | 78.20 | 71.29 | 333 | Browns |
| 10 | 7 | Todd Gurley II | 76.04 | 82.60 | 67.50 | 408 | Rams |
| 11 | 8 | Chris Thompson | 75.96 | 74.20 | 72.97 | 218 | Commanders |
| 12 | 9 | Marshawn Lynch | 75.44 | 74.60 | 71.84 | 182 | Raiders |
| 13 | 10 | Jalen Richard | 75.22 | 68.90 | 75.27 | 130 | Raiders |
| 14 | 11 | C.J. Anderson | 74.50 | 78.20 | 67.87 | 241 | Broncos |
| 15 | 12 | Tarik Cohen | 74.28 | 77.50 | 67.97 | 198 | Bears |
| 16 | 13 | Jay Ajayi | 74.06 | 66.90 | 74.66 | 271 | Eagles |

### Starter (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Le'Veon Bell | 73.87 | 72.20 | 70.82 | 557 | Steelers |
| 18 | 2 | Derrick Henry | 73.75 | 73.00 | 70.08 | 193 | Titans |
| 19 | 3 | Jerick McKinnon | 73.48 | 79.60 | 65.24 | 345 | Vikings |
| 20 | 4 | Corey Clement | 73.01 | 74.10 | 68.11 | 163 | Eagles |
| 21 | 5 | LeSean McCoy | 72.61 | 73.70 | 67.71 | 357 | Bills |
| 22 | 6 | Mark Ingram II | 72.12 | 70.50 | 69.03 | 263 | Saints |
| 23 | 7 | Ezekiel Elliott | 71.90 | 69.50 | 69.34 | 267 | Cowboys |
| 24 | 8 | Wayne Gallman | 71.88 | 70.00 | 68.97 | 165 | Giants |
| 25 | 9 | Marlon Mack | 70.52 | 68.00 | 68.04 | 161 | Colts |
| 26 | 10 | Christian McCaffrey | 70.51 | 74.60 | 63.62 | 481 | Panthers |
| 27 | 11 | Melvin Gordon III | 69.92 | 72.20 | 64.24 | 320 | Chargers |
| 28 | 12 | Theo Riddick | 69.82 | 71.00 | 64.86 | 271 | Lions |
| 29 | 13 | Devonta Freeman | 69.37 | 68.90 | 65.51 | 303 | Falcons |
| 30 | 14 | Jordan Howard | 69.30 | 63.90 | 68.73 | 190 | Bears |
| 31 | 15 | Rex Burkhead | 69.22 | 70.60 | 64.14 | 111 | Patriots |
| 32 | 16 | Tevin Coleman | 69.19 | 70.10 | 64.42 | 222 | Falcons |
| 33 | 17 | Giovani Bernard | 68.98 | 67.30 | 65.94 | 256 | Bengals |
| 34 | 18 | LeGarrette Blount | 68.61 | 63.50 | 67.85 | 124 | Eagles |
| 35 | 19 | Damien Williams | 68.50 | 64.30 | 67.14 | 110 | Dolphins |
| 36 | 20 | Rod Smith | 68.47 | 67.70 | 64.81 | 129 | Cowboys |
| 37 | 21 | Jamaal Charles | 68.37 | 59.40 | 70.19 | 104 | Broncos |
| 38 | 22 | Matt Forte | 68.32 | 66.10 | 65.64 | 191 | Jets |
| 39 | 23 | DeMarco Murray | 68.17 | 63.30 | 67.25 | 292 | Titans |
| 40 | 24 | Frank Gore | 68.14 | 70.40 | 62.46 | 198 | Colts |
| 41 | 25 | Bilal Powell | 67.98 | 67.30 | 64.27 | 158 | Jets |
| 42 | 26 | Lamar Miller | 67.60 | 68.90 | 62.57 | 336 | Texans |
| 43 | 27 | Carlos Hyde | 67.35 | 58.00 | 69.42 | 376 | 49ers |
| 44 | 28 | Latavius Murray | 67.28 | 67.70 | 62.84 | 145 | Vikings |
| 45 | 29 | Chris Ivory | 67.23 | 58.50 | 68.88 | 119 | Jaguars |
| 46 | 30 | Joe Mixon | 67.17 | 72.10 | 59.71 | 157 | Bengals |
| 47 | 31 | Danny Woodhead | 66.71 | 71.70 | 59.21 | 113 | Ravens |
| 48 | 32 | Orleans Darkwa | 66.65 | 60.30 | 66.71 | 109 | Giants |
| 49 | 33 | T.J. Yeldon | 66.41 | 64.00 | 63.85 | 149 | Jaguars |
| 50 | 34 | J.D. McKissic | 66.32 | 67.40 | 61.44 | 219 | Seahawks |
| 51 | 35 | Benny Cunningham | 65.88 | 62.80 | 63.77 | 121 | Bears |
| 52 | 36 | James White | 65.82 | 69.10 | 59.47 | 346 | Patriots |
| 53 | 37 | Elijah McGuire | 65.71 | 66.70 | 60.89 | 128 | Jets |
| 54 | 38 | Javorius Allen | 65.51 | 68.70 | 59.21 | 218 | Ravens |
| 55 | 39 | DeAndre Washington | 65.46 | 58.20 | 66.13 | 123 | Raiders |
| 56 | 40 | Isaiah Crowell | 65.28 | 61.20 | 63.84 | 214 | Browns |
| 57 | 41 | Jonathan Stewart | 65.15 | 56.40 | 66.82 | 110 | Panthers |
| 58 | 42 | Matt Breida | 64.88 | 63.90 | 61.37 | 159 | 49ers |
| 59 | 43 | Devontae Booker | 64.55 | 64.40 | 60.49 | 168 | Broncos |
| 60 | 44 | Leonard Fournette | 64.41 | 62.40 | 61.58 | 267 | Jaguars |
| 61 | 45 | Andre Ellington | 64.19 | 61.80 | 61.62 | 249 | Texans |
| 62 | 46 | Shane Vereen | 63.02 | 57.90 | 62.27 | 228 | Giants |
| 63 | 47 | Ameer Abdullah | 62.99 | 53.50 | 65.15 | 157 | Lions |
| 64 | 48 | Jamaal Williams | 62.62 | 61.90 | 58.94 | 200 | Packers |
| 65 | 49 | Charles Sims | 62.25 | 57.30 | 61.39 | 240 | Buccaneers |
| 66 | 50 | Charcandrick West | 62.10 | 60.10 | 59.27 | 173 | Chiefs |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 67 | 1 | Samaje Perine | 61.87 | 58.80 | 59.75 | 136 | Commanders |
| 68 | 2 | Thomas Rawls | 61.82 | 51.50 | 64.54 | 104 | Seahawks |

## LB — Linebacker

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Bobby Wagner | 89.95 | 91.40 | 84.82 | 1022 | Seahawks |
| 2 | 2 | Luke Kuechly | 86.80 | 90.60 | 81.98 | 973 | Panthers |
| 3 | 3 | Lavonte David | 86.44 | 90.50 | 81.13 | 814 | Buccaneers |
| 4 | 4 | Paul Posluszny | 84.49 | 87.50 | 78.73 | 520 | Jaguars |
| 5 | 5 | Deion Jones | 82.97 | 88.00 | 75.45 | 1148 | Falcons |
| 6 | 6 | Sean Lee | 80.50 | 82.10 | 78.29 | 622 | Cowboys |
| 7 | 7 | Telvin Smith Sr. | 80.32 | 81.90 | 75.51 | 1065 | Jaguars |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Myles Jack | 78.35 | 78.80 | 74.66 | 1223 | Jaguars |
| 9 | 2 | Mychal Kendricks | 78.22 | 79.90 | 73.86 | 726 | Eagles |
| 10 | 3 | Dylan Cole | 78.20 | 85.30 | 73.46 | 206 | Texans |
| 11 | 4 | Korey Toomer | 78.15 | 86.50 | 74.88 | 266 | Chargers |
| 12 | 5 | Avery Williamson | 78.07 | 80.20 | 72.68 | 742 | Titans |
| 13 | 6 | Ryan Shazier | 77.63 | 82.20 | 72.91 | 671 | Steelers |
| 14 | 7 | Danny Trevathan | 77.50 | 80.40 | 75.67 | 714 | Bears |
| 15 | 8 | Nick Kwiatkoski | 76.59 | 80.50 | 74.25 | 382 | Bears |
| 16 | 9 | Reuben Foster | 75.94 | 81.20 | 74.52 | 553 | 49ers |
| 17 | 10 | Kyle Emanuel | 75.22 | 74.00 | 71.87 | 301 | Chargers |
| 18 | 11 | Demario Davis | 75.14 | 73.70 | 71.94 | 1115 | Jets |
| 19 | 12 | Ben Gedeon | 74.30 | 76.40 | 68.74 | 272 | Vikings |
| 20 | 13 | Jake Ryan | 74.12 | 75.70 | 70.98 | 506 | Packers |
| 21 | 14 | C.J. Mosley | 74.04 | 72.20 | 71.74 | 1077 | Ravens |

### Starter (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Reggie Ragland | 73.74 | 76.00 | 70.01 | 369 | Chiefs |
| 23 | 2 | Anthony Hitchens | 73.52 | 73.00 | 71.78 | 544 | Cowboys |
| 24 | 3 | Benardrick McKinney | 72.58 | 69.00 | 71.21 | 959 | Texans |
| 25 | 4 | K.J. Wright | 71.68 | 68.70 | 70.01 | 956 | Seahawks |
| 26 | 5 | John Timu | 71.63 | 78.70 | 72.24 | 135 | Bears |
| 27 | 6 | De'Vondre Campbell | 71.04 | 69.20 | 68.88 | 1056 | Falcons |
| 28 | 7 | Joe Schobert | 70.94 | 68.70 | 68.26 | 1068 | Browns |
| 29 | 8 | Vontaze Burfict | 70.90 | 72.90 | 71.14 | 589 | Bengals |
| 30 | 9 | Zach Cunningham | 70.65 | 66.60 | 69.19 | 812 | Texans |
| 31 | 10 | Blake Martinez | 70.63 | 66.20 | 69.41 | 978 | Packers |
| 32 | 11 | Jaylon Smith | 70.58 | 69.40 | 67.20 | 575 | Cowboys |
| 33 | 12 | Jatavis Brown | 70.31 | 68.30 | 69.05 | 504 | Chargers |
| 34 | 13 | Michael Wilhoite | 69.96 | 67.40 | 70.00 | 306 | Seahawks |
| 35 | 14 | Todd Davis | 69.46 | 65.30 | 69.10 | 520 | Broncos |
| 36 | 15 | Wesley Woodyard | 69.40 | 65.10 | 68.10 | 1153 | Titans |
| 37 | 16 | Josh Bynes | 68.95 | 69.90 | 67.58 | 236 | Cardinals |
| 38 | 17 | Eric Kendricks | 68.84 | 67.00 | 66.41 | 1099 | Vikings |
| 39 | 18 | Manti Te'o | 68.55 | 69.30 | 68.79 | 621 | Saints |
| 40 | 19 | Thomas Davis Sr. | 68.27 | 64.10 | 66.89 | 849 | Panthers |
| 41 | 20 | NaVorro Bowman | 67.90 | 67.60 | 68.20 | 997 | Raiders |
| 42 | 21 | Nigel Bradham | 67.75 | 65.60 | 66.05 | 1125 | Eagles |
| 43 | 22 | Derrick Johnson | 67.57 | 65.10 | 65.99 | 893 | Chiefs |
| 44 | 23 | Shaq Thompson | 67.26 | 63.60 | 66.69 | 684 | Panthers |
| 45 | 24 | Christian Kirksey | 67.11 | 62.00 | 66.35 | 1068 | Browns |
| 46 | 25 | Craig Robertson | 67.00 | 64.30 | 65.78 | 921 | Saints |
| 47 | 26 | Patrick Onwuasor | 66.93 | 66.50 | 66.18 | 647 | Ravens |
| 48 | 27 | Anthony Barr | 66.70 | 63.80 | 64.67 | 1050 | Vikings |
| 49 | 28 | Jon Bostic | 66.50 | 63.40 | 67.53 | 914 | Colts |
| 50 | 29 | Karlos Dansby | 66.44 | 63.00 | 64.56 | 921 | Cardinals |
| 51 | 30 | Preston Brown | 66.31 | 60.10 | 66.28 | 1157 | Bills |
| 52 | 31 | Jalen Reeves-Maybin | 66.05 | 68.20 | 64.61 | 239 | Lions |
| 53 | 32 | Tahir Whitehead | 66.02 | 61.40 | 64.93 | 950 | Lions |
| 54 | 33 | Chase Allen | 65.73 | 61.70 | 66.33 | 220 | Dolphins |
| 55 | 34 | Kwon Alexander | 65.34 | 65.50 | 63.99 | 717 | Buccaneers |
| 56 | 35 | Mark Barron | 65.30 | 61.40 | 64.38 | 896 | Rams |
| 57 | 36 | Bryce Hager | 65.25 | 65.90 | 67.42 | 153 | Rams |
| 58 | 37 | Zaire Anderson | 65.06 | 61.10 | 68.35 | 136 | Broncos |
| 59 | 38 | Zach Brown | 64.97 | 58.70 | 66.55 | 834 | Commanders |
| 60 | 39 | Vince Williams | 64.26 | 61.90 | 63.23 | 785 | Steelers |
| 61 | 40 | Zach Vigil | 64.17 | 67.70 | 68.38 | 394 | Commanders |
| 62 | 41 | Brandon Marshall | 64.04 | 61.80 | 62.94 | 910 | Broncos |
| 63 | 42 | Jeremiah George | 63.63 | 63.30 | 68.54 | 180 | Colts |
| 64 | 43 | Matt Milano | 63.42 | 60.30 | 63.42 | 450 | Bills |
| 65 | 44 | Marquel Lee | 63.34 | 61.30 | 64.70 | 172 | Raiders |
| 66 | 45 | Christian Jones | 62.84 | 55.80 | 64.10 | 623 | Bears |
| 67 | 46 | Mason Foster | 62.77 | 64.50 | 64.43 | 288 | Commanders |
| 68 | 47 | Kyle Van Noy | 62.55 | 58.50 | 62.12 | 894 | Patriots |
| 69 | 48 | Vincent Rey | 62.40 | 56.30 | 63.33 | 607 | Bengals |
| 70 | 49 | Nick Bellore | 62.21 | 62.10 | 66.96 | 106 | Lions |
| 71 | 50 | David Harris | 62.09 | 61.00 | 62.08 | 181 | Patriots |

### Rotation/backup (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 72 | 1 | Eli Harold | 61.99 | 56.80 | 61.28 | 452 | 49ers |
| 73 | 2 | Jordan Hicks | 61.80 | 64.30 | 62.53 | 268 | Eagles |
| 74 | 3 | Cory Littleton | 61.57 | 60.20 | 60.66 | 278 | Rams |
| 75 | 4 | Paul Worrilow | 61.37 | 59.10 | 62.88 | 272 | Lions |
| 76 | 5 | Kendell Beckwith | 61.35 | 53.00 | 62.75 | 847 | Buccaneers |
| 77 | 6 | Kelvin Sheppard | 61.23 | 57.00 | 64.57 | 386 | Giants |
| 78 | 7 | Brian Cushing | 60.71 | 59.30 | 63.54 | 163 | Texans |
| 79 | 8 | Jayon Brown | 60.19 | 54.20 | 60.01 | 527 | Titans |
| 80 | 9 | Nicholas Morrow | 60.02 | 53.50 | 61.23 | 553 | Raiders |
| 81 | 10 | Denzel Perryman | 59.99 | 59.20 | 62.92 | 273 | Chargers |
| 82 | 11 | Martrell Spaight | 59.90 | 58.10 | 61.84 | 414 | Commanders |
| 83 | 12 | Joe Thomas | 59.62 | 58.00 | 61.73 | 104 | Packers |
| 84 | 13 | Damien Wilson | 59.40 | 50.50 | 65.02 | 321 | Cowboys |
| 85 | 14 | Marquis Flowers | 59.31 | 55.70 | 64.21 | 361 | Patriots |
| 86 | 15 | Lawrence Timmons | 59.20 | 53.70 | 59.73 | 794 | Dolphins |
| 87 | 16 | Will Compton | 59.18 | 55.20 | 63.19 | 151 | Commanders |
| 88 | 17 | Jarrad Davis | 58.75 | 52.80 | 60.63 | 828 | Lions |
| 89 | 18 | Adarius Taylor | 58.65 | 58.80 | 61.56 | 284 | Buccaneers |
| 90 | 19 | Alec Ogletree | 58.57 | 55.00 | 59.29 | 994 | Rams |
| 91 | 20 | Stephone Anthony | 58.52 | 56.20 | 63.08 | 130 | Dolphins |
| 92 | 21 | Kiko Alonso | 58.44 | 49.10 | 61.54 | 1007 | Dolphins |
| 93 | 22 | Dannell Ellerbe | 58.11 | 58.00 | 64.01 | 102 | Eagles |
| 94 | 23 | Duke Riley | 58.02 | 54.50 | 59.34 | 245 | Falcons |
| 95 | 24 | Najee Goode | 57.65 | 54.80 | 62.46 | 208 | Eagles |
| 96 | 25 | Elandon Roberts | 57.42 | 48.20 | 59.79 | 670 | Patriots |
| 97 | 26 | Keenan Robinson | 57.35 | 53.40 | 61.86 | 292 | Giants |
| 98 | 27 | Devante Bond | 57.33 | 55.10 | 58.55 | 137 | Buccaneers |
| 99 | 28 | Terence Garvin | 56.93 | 58.40 | 60.75 | 195 | Seahawks |
| 100 | 29 | Ramik Wilson | 56.83 | 52.70 | 65.63 | 125 | Chiefs |
| 101 | 30 | James Burgess | 56.54 | 48.90 | 59.55 | 646 | Browns |
| 102 | 31 | Ramon Humber | 56.39 | 50.10 | 59.75 | 629 | Bills |
| 103 | 32 | Anthony Walker Jr. | 56.24 | 65.20 | 66.10 | 115 | Colts |
| 104 | 33 | Ray-Ray Armstrong | 56.06 | 53.30 | 61.14 | 481 | Giants |
| 105 | 34 | Cory James | 56.05 | 51.50 | 60.00 | 455 | Raiders |
| 106 | 35 | Kevin Pierre-Louis | 55.84 | 50.70 | 60.74 | 273 | Chiefs |
| 107 | 36 | Antonio Morrison | 55.68 | 45.30 | 59.87 | 813 | Colts |
| 108 | 37 | Alex Anzalone | 55.36 | 61.40 | 65.67 | 158 | Saints |
| 109 | 38 | B.J. Goodson | 55.18 | 50.30 | 64.81 | 374 | Giants |
| 110 | 39 | Deone Bucannon | 54.97 | 47.80 | 59.36 | 704 | Cardinals |
| 111 | 40 | Mike Hull | 54.88 | 54.40 | 62.07 | 185 | Dolphins |
| 112 | 41 | A.J. Klein | 54.85 | 48.20 | 58.35 | 664 | Saints |
| 113 | 42 | David Mayo | 54.73 | 50.90 | 59.78 | 134 | Panthers |
| 114 | 43 | Brock Coyle | 54.71 | 51.80 | 58.53 | 646 | 49ers |
| 115 | 44 | Jonathan Freeny | 54.68 | 54.50 | 60.54 | 101 | Saints |
| 116 | 45 | Kevin Minter | 54.60 | 43.30 | 61.62 | 196 | Bengals |
| 117 | 46 | Darron Lee | 54.58 | 44.20 | 59.15 | 1025 | Jets |
| 118 | 47 | Nick Vigil | 54.48 | 50.40 | 58.23 | 759 | Bengals |
| 119 | 48 | Kamalei Correa | 53.63 | 44.40 | 56.26 | 148 | Ravens |
| 120 | 49 | Jonathan Casillas | 53.00 | 46.40 | 57.60 | 457 | Giants |
| 121 | 50 | Hardy Nickerson | 52.91 | 44.50 | 58.52 | 159 | Bengals |
| 122 | 51 | Calvin Munson | 52.48 | 42.90 | 58.86 | 388 | Giants |
| 123 | 52 | Hayes Pullard | 51.95 | 45.00 | 58.67 | 474 | Chargers |
| 124 | 53 | Jamie Collins Sr. | 49.52 | 42.10 | 56.23 | 330 | Browns |
| 125 | 54 | Jordan Evans | 48.65 | 38.40 | 56.52 | 312 | Bengals |
| 126 | 55 | Curtis Grant | 47.18 | 44.40 | 58.29 | 109 | Giants |
| 127 | 56 | Sean Spence | 45.00 | 27.20 | 54.35 | 218 | Steelers |

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

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Russell Wilson | 78.08 | 77.90 | 73.68 | 713 | Seahawks |
| 7 | 2 | Matthew Stafford | 77.88 | 75.90 | 74.95 | 674 | Lions |
| 8 | 3 | Carson Wentz | 76.74 | 79.14 | 72.52 | 537 | Eagles |
| 9 | 4 | Philip Rivers | 76.63 | 76.16 | 72.70 | 634 | Chargers |
| 10 | 5 | Aaron Rodgers | 75.46 | 81.57 | 72.38 | 302 | Packers |
| 11 | 6 | Kirk Cousins | 75.19 | 71.87 | 73.39 | 626 | Commanders |

### Starter (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Jameis Winston | 73.65 | 73.00 | 71.77 | 529 | Buccaneers |
| 13 | 2 | Case Keenum | 72.95 | 73.69 | 71.13 | 666 | Vikings |
| 14 | 3 | Derek Carr | 72.24 | 75.26 | 66.59 | 586 | Raiders |
| 15 | 4 | Andy Dalton | 71.62 | 72.32 | 68.19 | 575 | Bengals |
| 16 | 5 | Dak Prescott | 70.67 | 71.64 | 66.95 | 606 | Cowboys |
| 17 | 6 | Jared Goff | 70.59 | 70.43 | 72.90 | 589 | Rams |
| 18 | 7 | Marcus Mariota | 70.23 | 70.16 | 67.48 | 626 | Titans |
| 19 | 8 | Tyrod Taylor | 69.97 | 74.51 | 63.80 | 584 | Bills |
| 20 | 9 | Carson Palmer | 69.60 | 72.22 | 69.10 | 312 | Cardinals |
| 21 | 10 | Cam Newton | 69.28 | 69.93 | 64.67 | 641 | Panthers |
| 22 | 11 | Blake Bortles | 67.06 | 63.40 | 65.50 | 706 | Jaguars |
| 23 | 12 | Jimmy Garoppolo | 66.68 | 85.40 | 75.29 | 196 | 49ers |
| 24 | 13 | Josh McCown | 66.25 | 63.97 | 70.52 | 482 | Jets |
| 25 | 14 | Eli Manning | 66.11 | 66.30 | 61.43 | 645 | Giants |
| 26 | 15 | Joe Flacco | 65.53 | 67.37 | 60.50 | 605 | Ravens |
| 27 | 16 | Nick Foles | 64.53 | 70.20 | 71.35 | 225 | Eagles |
| 28 | 17 | Deshaun Watson | 64.05 | 64.10 | 79.42 | 267 | Texans |
| 29 | 18 | Mitch Trubisky | 62.10 | 71.20 | 61.41 | 415 | Bears |

### Rotation/backup (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Jacoby Brissett | 60.95 | 62.17 | 61.68 | 591 | Colts |
| 31 | 2 | Ryan Fitzpatrick | 60.18 | 60.88 | 64.62 | 194 | Buccaneers |
| 32 | 3 | Brian Hoyer | 59.57 | 67.23 | 59.64 | 241 | Patriots |
| 33 | 4 | Jay Cutler | 59.51 | 59.74 | 61.33 | 476 | Dolphins |
| 34 | 5 | Mike Glennon | 58.33 | 63.13 | 59.78 | 159 | Bears |
| 35 | 6 | Tom Savage | 57.88 | 58.85 | 60.53 | 268 | Texans |
| 36 | 7 | Matt Moore | 57.78 | 55.18 | 64.05 | 148 | Dolphins |
| 37 | 8 | C.J. Beathard | 57.29 | 56.70 | 57.68 | 265 | 49ers |
| 38 | 9 | DeShone Kizer | 57.22 | 49.90 | 57.00 | 605 | Browns |
| 39 | 10 | Brett Hundley | 57.04 | 58.28 | 55.55 | 389 | Packers |
| 40 | 11 | Trevor Siemian | 56.14 | 53.42 | 60.03 | 432 | Broncos |
| 41 | 12 | Drew Stanton | 55.39 | 53.21 | 56.01 | 181 | Cardinals |
| 42 | 13 | T.J. Yates | 54.40 | 46.41 | 57.57 | 123 | Texans |
| 43 | 14 | Bryce Petty | 53.52 | 44.19 | 55.86 | 135 | Jets |
| 44 | 15 | Brock Osweiler | 51.83 | 51.16 | 56.81 | 206 | Broncos |
| 45 | 16 | Blaine Gabbert | 51.36 | 48.01 | 57.55 | 219 | Cardinals |

## S — Safety

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Harrison Smith | 93.50 | 91.60 | 91.64 | 1103 | Vikings |
| 2 | 2 | Lamarcus Joyner | 93.47 | 91.20 | 92.39 | 756 | Rams |
| 3 | 3 | Micah Hyde | 91.43 | 90.40 | 87.95 | 1101 | Bills |
| 4 | 4 | Marcus Williams | 91.41 | 89.60 | 88.45 | 1103 | Saints |
| 5 | 5 | Glover Quin | 91.13 | 91.30 | 86.85 | 1053 | Lions |
| 6 | 6 | Earl Thomas III | 90.91 | 90.50 | 89.61 | 921 | Seahawks |
| 7 | 7 | Tre Boston | 89.82 | 89.00 | 86.72 | 1039 | Chargers |
| 8 | 8 | Adrian Amos | 88.99 | 89.50 | 87.48 | 670 | Bears |
| 9 | 9 | Antoine Bethea | 88.80 | 87.60 | 87.84 | 741 | Cardinals |
| 10 | 10 | Jordan Poyer | 88.58 | 90.90 | 86.41 | 1097 | Bills |
| 11 | 11 | Kevin Byard | 87.58 | 85.40 | 84.86 | 1223 | Titans |
| 12 | 12 | John Johnson III | 85.39 | 82.90 | 82.88 | 781 | Rams |
| 13 | 13 | Landon Collins | 84.37 | 81.60 | 82.57 | 908 | Giants |
| 14 | 14 | Andrew Sendejo | 83.44 | 81.30 | 82.27 | 853 | Vikings |
| 15 | 15 | Bradley McDougald | 82.63 | 82.00 | 78.89 | 675 | Seahawks |
| 16 | 16 | Rodney McLeod | 81.72 | 81.60 | 77.64 | 1048 | Eagles |
| 17 | 17 | Mike Adams | 81.39 | 80.30 | 78.88 | 1022 | Panthers |
| 18 | 18 | Tyvon Branch | 81.08 | 79.70 | 81.48 | 579 | Cardinals |
| 19 | 19 | Jeff Heath | 80.90 | 80.80 | 78.88 | 880 | Cowboys |
| 20 | 20 | Duron Harmon | 80.62 | 79.50 | 77.20 | 829 | Patriots |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Corey Graham | 78.96 | 79.30 | 74.57 | 488 | Eagles |
| 22 | 2 | Jaquiski Tartt | 78.67 | 81.20 | 76.99 | 595 | 49ers |
| 23 | 3 | Eddie Pleasant | 78.41 | 74.90 | 76.58 | 307 | Texans |
| 24 | 4 | Corey Moore | 78.21 | 79.70 | 77.09 | 241 | Texans |
| 25 | 5 | Malcolm Jenkins | 78.07 | 74.30 | 76.42 | 1151 | Eagles |
| 26 | 6 | Cody Davis | 77.84 | 84.60 | 76.66 | 280 | Rams |
| 27 | 7 | Patrick Chung | 76.81 | 73.60 | 74.79 | 1127 | Patriots |
| 28 | 8 | Eric Weddle | 76.63 | 76.10 | 73.45 | 1084 | Ravens |
| 29 | 9 | Ricardo Allen | 74.69 | 71.10 | 72.91 | 1083 | Falcons |
| 30 | 10 | Ha Ha Clinton-Dix | 74.51 | 69.20 | 73.89 | 1043 | Packers |
| 31 | 11 | George Iloka | 74.48 | 70.10 | 73.87 | 988 | Bengals |
| 32 | 12 | Devin McCourty | 74.40 | 66.70 | 75.36 | 1224 | Patriots |
| 33 | 13 | Michael Thomas | 74.36 | 69.20 | 78.31 | 153 | Dolphins |
| 34 | 14 | Eddie Jackson | 74.13 | 69.00 | 73.39 | 1055 | Bears |

### Starter (44 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Barry Church | 73.51 | 72.20 | 71.37 | 1203 | Jaguars |
| 36 | 2 | Kam Chancellor | 73.50 | 72.70 | 74.76 | 598 | Seahawks |
| 37 | 3 | Da'Norris Searcy | 73.50 | 72.20 | 71.03 | 383 | Titans |
| 38 | 4 | D.J. Swearinger Sr. | 73.25 | 70.40 | 72.24 | 1092 | Commanders |
| 39 | 5 | Eric Reid | 73.02 | 72.60 | 72.56 | 736 | 49ers |
| 40 | 6 | Montae Nicholson | 72.91 | 75.20 | 75.55 | 319 | Commanders |
| 41 | 7 | Chris Conte | 72.72 | 72.00 | 70.06 | 774 | Buccaneers |
| 42 | 8 | Shawn Williams | 72.22 | 70.80 | 72.14 | 579 | Bengals |
| 43 | 9 | Morgan Burnett | 71.98 | 66.50 | 74.17 | 724 | Packers |
| 44 | 10 | Adrian Colbert | 71.86 | 72.90 | 74.30 | 530 | 49ers |
| 45 | 11 | Tashaun Gipson Sr. | 71.78 | 65.10 | 72.70 | 1199 | Jaguars |
| 46 | 12 | Justin Simmons | 71.51 | 68.50 | 72.86 | 736 | Broncos |
| 47 | 13 | Anthony Harris | 71.31 | 70.50 | 77.88 | 254 | Vikings |
| 48 | 14 | Keanu Neal | 70.97 | 64.50 | 71.12 | 1174 | Falcons |
| 49 | 15 | Tony Jefferson | 70.94 | 65.50 | 71.04 | 1084 | Ravens |
| 50 | 16 | Xavier Woods | 70.85 | 67.10 | 70.21 | 547 | Cowboys |
| 51 | 17 | Jamal Adams | 70.78 | 65.50 | 70.14 | 1100 | Jets |
| 52 | 18 | Andre Hal | 70.64 | 64.40 | 70.84 | 939 | Texans |
| 53 | 19 | Daniel Sorensen | 70.53 | 71.80 | 66.35 | 1055 | Chiefs |
| 54 | 20 | Jahleel Addae | 70.21 | 70.30 | 69.12 | 1030 | Chargers |
| 55 | 21 | Robert Golden | 70.16 | 68.60 | 69.22 | 214 | Steelers |
| 56 | 22 | Adrian Phillips | 69.95 | 68.00 | 68.52 | 521 | Chargers |
| 57 | 23 | Vonn Bell | 69.00 | 65.40 | 67.61 | 938 | Saints |
| 58 | 24 | Matthias Farley | 68.65 | 64.50 | 70.76 | 928 | Colts |
| 59 | 25 | Darius Butler | 68.46 | 65.50 | 66.79 | 500 | Colts |
| 60 | 26 | Clayton Fejedelem | 67.95 | 62.70 | 73.53 | 377 | Bengals |
| 61 | 27 | Derrick Kindred | 67.29 | 64.30 | 67.99 | 688 | Browns |
| 62 | 28 | Marcus Gilchrist | 67.27 | 63.20 | 66.75 | 813 | Texans |
| 63 | 29 | Mike Mitchell | 67.12 | 61.30 | 67.87 | 738 | Steelers |
| 64 | 30 | Jabrill Peppers | 66.72 | 61.70 | 69.03 | 806 | Browns |
| 65 | 31 | T.J. Ward | 65.92 | 58.90 | 69.35 | 405 | Buccaneers |
| 66 | 32 | Malik Hooker | 65.82 | 66.50 | 72.06 | 410 | Colts |
| 67 | 33 | Kai Nacua | 65.73 | 71.60 | 71.07 | 214 | Browns |
| 68 | 34 | Budda Baker | 65.49 | 57.00 | 66.98 | 515 | Cardinals |
| 69 | 35 | Reshad Jones | 65.25 | 56.10 | 70.31 | 1015 | Dolphins |
| 70 | 36 | Reggie Nelson | 64.41 | 52.90 | 67.92 | 1026 | Raiders |
| 71 | 37 | Rafael Bush | 63.81 | 59.20 | 66.36 | 236 | Saints |
| 72 | 38 | Shalom Luani | 63.64 | 56.70 | 68.26 | 187 | Raiders |
| 73 | 39 | Keith Tandy | 63.32 | 57.50 | 68.13 | 226 | Buccaneers |
| 74 | 40 | Darian Stewart | 63.14 | 56.00 | 63.73 | 888 | Broncos |
| 75 | 41 | Colin Jones | 63.01 | 65.70 | 67.25 | 213 | Panthers |
| 76 | 42 | Miles Killebrew | 62.31 | 54.30 | 64.51 | 353 | Lions |
| 77 | 43 | Marcus Maye | 62.19 | 53.70 | 63.69 | 1063 | Jets |
| 78 | 44 | Kemal Ishmael | 62.09 | 52.70 | 67.73 | 126 | Falcons |

### Rotation/backup (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 79 | 1 | Karl Joseph | 61.79 | 56.50 | 63.75 | 867 | Raiders |
| 80 | 2 | Ibraheim Campbell | 61.54 | 57.80 | 69.23 | 198 | Texans |
| 81 | 3 | Blake Countess | 61.15 | 64.90 | 61.11 | 166 | Rams |
| 82 | 4 | Ron Parker | 60.98 | 56.00 | 60.14 | 1099 | Chiefs |
| 83 | 5 | T.J. Green | 60.96 | 49.80 | 64.62 | 382 | Colts |
| 84 | 6 | Kavon Frazier | 60.85 | 64.10 | 60.90 | 226 | Cowboys |
| 85 | 7 | Jermaine Whitehead | 60.82 | 56.80 | 63.88 | 116 | Packers |
| 86 | 8 | Darian Thompson | 60.71 | 57.30 | 64.29 | 1066 | Giants |
| 87 | 9 | Deshazor Everett | 60.62 | 56.80 | 64.31 | 588 | Commanders |
| 88 | 10 | T.J. McDonald | 60.60 | 56.70 | 64.24 | 539 | Dolphins |
| 89 | 11 | Josh Jones | 60.00 | 52.20 | 62.06 | 730 | Packers |
| 90 | 12 | Kentrell Brice | 59.59 | 59.20 | 62.20 | 289 | Packers |
| 91 | 13 | Justin Evans | 59.45 | 54.50 | 61.71 | 715 | Buccaneers |
| 92 | 14 | Marwin Evans | 59.39 | 58.40 | 63.43 | 151 | Packers |
| 93 | 15 | Tavon Wilson | 59.23 | 56.20 | 60.21 | 547 | Lions |
| 94 | 16 | Jordan Richards | 58.87 | 58.70 | 59.18 | 316 | Patriots |
| 95 | 17 | J.J. Wilcox | 56.93 | 56.60 | 58.30 | 132 | Steelers |
| 96 | 18 | Kurt Coleman | 56.82 | 45.70 | 61.95 | 772 | Panthers |
| 97 | 19 | Jairus Byrd | 56.79 | 53.50 | 59.09 | 135 | Panthers |
| 98 | 20 | Johnathan Cyprien | 56.65 | 48.50 | 60.42 | 743 | Titans |
| 99 | 21 | Jimmie Ward | 56.55 | 51.50 | 61.93 | 429 | 49ers |
| 100 | 22 | Clayton Geathers | 56.20 | 55.50 | 60.84 | 112 | Colts |
| 101 | 23 | Will Parks | 54.93 | 47.30 | 56.24 | 597 | Broncos |
| 102 | 24 | Sean Davis | 54.78 | 46.70 | 56.00 | 1010 | Steelers |
| 103 | 25 | Kenny Vaccaro | 52.74 | 41.80 | 59.52 | 691 | Saints |
| 104 | 26 | Nate Allen | 51.96 | 52.80 | 57.04 | 362 | Dolphins |
| 105 | 27 | Quintin Demps | 50.52 | 45.60 | 57.24 | 177 | Bears |
| 106 | 28 | Rontez Miles | 45.16 | 33.30 | 52.13 | 125 | Jets |

## T — Tackle

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Joe Staley | 93.68 | 89.60 | 92.23 | 983 | 49ers |
| 2 | 2 | David Bakhtiari | 90.83 | 87.70 | 88.75 | 754 | Packers |
| 3 | 3 | Jason Peters | 89.75 | 84.20 | 89.28 | 423 | Eagles |
| 4 | 4 | Trent Williams | 88.78 | 82.60 | 88.74 | 615 | Commanders |
| 5 | 5 | Joe Thomas | 88.70 | 84.70 | 87.20 | 465 | Browns |
| 6 | 6 | D.J. Humphries | 88.65 | 81.70 | 89.12 | 204 | Cardinals |
| 7 | 7 | Lane Johnson | 88.61 | 80.60 | 89.79 | 1166 | Eagles |
| 8 | 8 | Ryan Ramczyk | 87.84 | 81.40 | 87.96 | 1164 | Saints |
| 9 | 9 | Donald Penn | 86.89 | 79.40 | 87.71 | 819 | Raiders |
| 10 | 10 | Dion Dawkins | 86.75 | 80.40 | 86.82 | 859 | Bills |
| 11 | 11 | Charles Leno Jr. | 86.68 | 78.70 | 87.83 | 988 | Bears |
| 12 | 12 | Andrew Whitworth | 86.56 | 80.50 | 86.44 | 983 | Rams |
| 13 | 13 | Anthony Castonzo | 86.15 | 80.30 | 85.89 | 1030 | Colts |
| 14 | 14 | Jake Matthews | 85.72 | 80.30 | 85.17 | 1159 | Falcons |
| 15 | 15 | Marcus Gilbert | 85.10 | 76.70 | 86.53 | 411 | Steelers |
| 16 | 16 | Tyron Smith | 84.94 | 76.90 | 86.14 | 758 | Cowboys |
| 17 | 17 | Russell Okung | 84.44 | 78.40 | 84.30 | 926 | Chargers |
| 18 | 18 | Daryl Williams | 84.02 | 78.00 | 83.86 | 1140 | Panthers |
| 19 | 19 | Taylor Lewan | 83.57 | 76.40 | 84.19 | 1066 | Titans |
| 20 | 20 | Terron Armstead | 83.51 | 76.50 | 84.01 | 667 | Saints |
| 21 | 21 | Trent Brown | 83.44 | 73.00 | 86.23 | 669 | 49ers |
| 22 | 22 | Marcus Cannon | 83.11 | 74.50 | 84.68 | 478 | Patriots |
| 23 | 23 | Nate Solder | 83.08 | 75.50 | 83.96 | 1334 | Patriots |
| 24 | 24 | Ronnie Stanley | 82.91 | 75.10 | 83.95 | 1009 | Ravens |
| 25 | 25 | Demar Dotson | 82.62 | 73.60 | 84.47 | 715 | Buccaneers |
| 26 | 26 | Alejandro Villanueva | 82.39 | 76.10 | 82.41 | 1155 | Steelers |
| 27 | 27 | Rick Wagner | 81.89 | 75.20 | 82.19 | 792 | Lions |
| 28 | 28 | Garett Bolles | 81.88 | 72.90 | 83.70 | 1107 | Broncos |
| 29 | 29 | Cam Fleming | 81.86 | 73.90 | 83.00 | 543 | Patriots |
| 30 | 30 | Jack Conklin | 81.76 | 72.40 | 83.83 | 1099 | Titans |
| 31 | 31 | Duane Brown | 81.66 | 74.10 | 82.54 | 620 | Seahawks |
| 32 | 32 | Mitchell Schwartz | 81.36 | 72.90 | 82.83 | 1083 | Chiefs |
| 33 | 33 | Jermey Parnell | 81.01 | 72.50 | 82.51 | 1078 | Jaguars |
| 34 | 34 | Ryan Schraeder | 80.35 | 70.60 | 82.69 | 967 | Falcons |
| 35 | 35 | Mike Remmers | 80.09 | 70.50 | 82.32 | 820 | Vikings |

### Good (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 36 | 1 | Ja'Wuan James | 79.79 | 69.00 | 82.81 | 494 | Dolphins |
| 37 | 2 | Eric Fisher | 78.21 | 69.70 | 79.72 | 1017 | Chiefs |
| 38 | 3 | Morgan Moses | 77.81 | 67.20 | 80.71 | 958 | Commanders |
| 39 | 4 | Chris Hubbard | 77.75 | 68.20 | 79.95 | 847 | Steelers |
| 40 | 5 | Kelvin Beachum | 77.52 | 69.30 | 78.84 | 1034 | Jets |
| 41 | 6 | Cordy Glenn | 76.93 | 67.40 | 79.11 | 275 | Bills |
| 42 | 7 | Austin Howard | 76.82 | 66.30 | 79.66 | 1081 | Ravens |
| 43 | 8 | Josh Wells | 76.37 | 65.50 | 79.45 | 469 | Jaguars |
| 44 | 9 | Taylor Decker | 76.04 | 65.30 | 79.04 | 471 | Lions |
| 45 | 10 | Brandon Shell | 75.73 | 64.70 | 78.92 | 696 | Jets |
| 46 | 11 | Donovan Smith | 75.49 | 64.90 | 78.38 | 1059 | Buccaneers |
| 47 | 12 | Bobby Massie | 75.38 | 65.00 | 78.13 | 912 | Bears |
| 48 | 13 | Ereck Flowers | 75.21 | 66.80 | 76.65 | 1001 | Giants |
| 49 | 14 | Riley Reiff | 74.84 | 63.60 | 78.17 | 1153 | Vikings |
| 50 | 15 | Kyle Murphy | 74.80 | 63.00 | 78.50 | 228 | Packers |
| 51 | 16 | Brent Qvale | 74.66 | 62.10 | 78.86 | 394 | Jets |
| 52 | 17 | Jason Spriggs | 74.47 | 64.00 | 77.28 | 278 | Packers |
| 53 | 18 | La'el Collins | 74.25 | 63.30 | 77.39 | 1065 | Cowboys |
| 54 | 19 | Jared Veldheer | 74.20 | 62.30 | 77.97 | 895 | Cardinals |

### Starter (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 55 | 1 | Matt Kalil | 73.74 | 63.10 | 76.66 | 1138 | Panthers |
| 56 | 2 | Jordan Mills | 73.64 | 62.10 | 77.17 | 1101 | Bills |
| 57 | 3 | John Wetzel | 73.03 | 59.70 | 77.75 | 916 | Cardinals |
| 58 | 4 | Denzelle Good | 72.97 | 60.40 | 77.18 | 293 | Colts |
| 59 | 5 | Shon Coleman | 72.87 | 61.40 | 76.35 | 1044 | Browns |
| 60 | 6 | Laremy Tunsil | 72.82 | 60.10 | 77.13 | 934 | Dolphins |
| 61 | 7 | Chad Wheeler | 72.65 | 60.50 | 76.58 | 261 | Giants |
| 62 | 8 | Sam Young | 72.48 | 61.30 | 75.77 | 451 | Dolphins |
| 63 | 9 | Bryan Bulaga | 72.31 | 61.40 | 75.41 | 232 | Packers |
| 64 | 10 | Ty Nsekhe | 72.31 | 60.30 | 76.15 | 305 | Commanders |
| 65 | 11 | Halapoulivaati Vaitai | 71.79 | 59.40 | 75.88 | 1031 | Eagles |
| 66 | 12 | LaAdrian Waddle | 71.12 | 57.80 | 75.84 | 380 | Patriots |
| 67 | 13 | Dennis Kelly | 70.87 | 58.20 | 75.15 | 234 | Titans |
| 68 | 14 | Rashod Hill | 70.69 | 58.30 | 74.79 | 737 | Vikings |
| 69 | 15 | Michael Schofield III | 70.44 | 57.00 | 75.23 | 407 | Chargers |
| 70 | 16 | Ty Sambrailo | 70.37 | 56.90 | 75.19 | 227 | Falcons |
| 71 | 17 | Kendall Lamm | 70.28 | 58.40 | 74.03 | 159 | Texans |
| 72 | 18 | Marshall Newhouse | 70.03 | 57.90 | 73.95 | 841 | Raiders |
| 73 | 19 | Joe Barksdale | 69.52 | 54.00 | 75.70 | 657 | Chargers |
| 74 | 20 | Will Holden | 69.51 | 55.60 | 74.61 | 327 | Cardinals |
| 75 | 21 | Jake Fisher | 68.93 | 54.70 | 74.25 | 361 | Bengals |
| 76 | 22 | Cedric Ogbuehi | 68.63 | 56.60 | 72.49 | 667 | Bengals |
| 77 | 23 | Brian Mihalik | 68.45 | 52.10 | 75.19 | 192 | Lions |
| 78 | 24 | Greg Robinson | 68.06 | 54.60 | 72.86 | 395 | Lions |
| 79 | 25 | Darrell Williams | 68.00 | 60.00 | 69.17 | 124 | Rams |
| 80 | 26 | Vadal Alexander | 68.00 | 55.10 | 72.44 | 256 | Raiders |
| 81 | 27 | Germain Ifedi | 67.98 | 51.70 | 74.67 | 1068 | Seahawks |
| 82 | 28 | Eric Winston | 67.58 | 53.10 | 73.07 | 201 | Bengals |
| 83 | 29 | Cam Robinson | 67.58 | 52.40 | 73.54 | 1080 | Jaguars |
| 84 | 30 | Donald Stephenson | 66.98 | 53.60 | 71.74 | 303 | Broncos |
| 85 | 31 | Caleb Benenoch | 66.69 | 50.80 | 73.11 | 359 | Buccaneers |
| 86 | 32 | Chris Clark | 66.65 | 51.00 | 72.92 | 548 | Texans |
| 87 | 33 | Corey Robinson | 65.68 | 49.10 | 72.56 | 324 | Lions |
| 88 | 34 | David Sharpe | 64.12 | 48.80 | 70.17 | 124 | Raiders |
| 89 | 35 | Julie'n Davenport | 64.04 | 50.60 | 68.83 | 238 | Texans |
| 90 | 36 | Menelik Watson | 63.93 | 47.40 | 70.79 | 448 | Broncos |
| 91 | 37 | Sam Tevi | 62.97 | 56.50 | 63.12 | 135 | Chargers |
| 92 | 38 | Breno Giacomini | 62.74 | 45.60 | 70.00 | 1095 | Texans |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 93 | 1 | Rees Odhiambo | 60.82 | 41.20 | 69.74 | 484 | Seahawks |
| 94 | 2 | T.J. Clemmings | 58.72 | 36.40 | 69.43 | 142 | Commanders |

## TE — Tight End

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 87.44 | 90.50 | 81.23 | 667 | Patriots |
| 2 | 2 | Hunter Henry | 84.07 | 87.20 | 77.81 | 324 | Chargers |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 3 | 1 | Travis Kelce | 79.59 | 83.30 | 72.95 | 583 | Chiefs |
| 4 | 2 | Zach Ertz | 79.48 | 80.60 | 74.56 | 572 | Eagles |
| 5 | 3 | Delanie Walker | 78.01 | 78.00 | 73.85 | 571 | Titans |
| 6 | 4 | Vance McDonald | 77.75 | 76.80 | 74.21 | 189 | Steelers |
| 7 | 5 | David Morgan | 77.12 | 85.50 | 67.37 | 135 | Vikings |
| 8 | 6 | Greg Olsen | 76.37 | 68.60 | 77.38 | 234 | Panthers |
| 9 | 7 | Trey Burton | 75.39 | 73.40 | 72.55 | 217 | Eagles |
| 10 | 8 | O.J. Howard | 74.36 | 61.40 | 78.84 | 329 | Buccaneers |
| 11 | 9 | Jared Cook | 74.05 | 63.20 | 77.12 | 526 | Raiders |

### Starter (60 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Rhett Ellison | 73.89 | 74.60 | 69.25 | 263 | Giants |
| 13 | 2 | Jeff Heuerman | 73.85 | 66.40 | 74.65 | 164 | Broncos |
| 14 | 3 | Jimmy Graham | 73.70 | 66.00 | 74.67 | 538 | Seahawks |
| 15 | 4 | Darren Fells | 73.57 | 69.00 | 72.45 | 281 | Lions |
| 16 | 5 | Garrett Celek | 73.48 | 73.00 | 69.64 | 288 | 49ers |
| 17 | 6 | Cameron Brate | 73.35 | 73.60 | 69.01 | 434 | Buccaneers |
| 18 | 7 | Zach Miller | 73.24 | 66.60 | 73.50 | 196 | Bears |
| 19 | 8 | Coby Fleener | 72.97 | 63.20 | 75.31 | 201 | Saints |
| 20 | 9 | Eric Ebron | 72.84 | 69.70 | 70.76 | 430 | Lions |
| 21 | 10 | Marcedes Lewis | 72.10 | 67.50 | 71.00 | 531 | Jaguars |
| 22 | 11 | James O'Shaughnessy | 71.88 | 66.50 | 71.30 | 143 | Jaguars |
| 23 | 12 | Charles Clay | 71.66 | 70.80 | 68.06 | 377 | Bills |
| 24 | 13 | Jack Doyle | 71.54 | 70.80 | 67.86 | 504 | Colts |
| 25 | 14 | Evan Engram | 71.22 | 62.10 | 73.13 | 542 | Giants |
| 26 | 15 | Vernon Davis | 71.19 | 61.90 | 73.22 | 465 | Commanders |
| 27 | 16 | Eric Tomlinson | 71.17 | 60.40 | 74.19 | 154 | Jets |
| 28 | 17 | Julius Thomas | 70.96 | 62.50 | 72.43 | 423 | Dolphins |
| 29 | 18 | Kyle Rudolph | 70.77 | 68.20 | 68.31 | 607 | Vikings |
| 30 | 19 | Jordan Reed | 70.70 | 64.90 | 70.40 | 162 | Commanders |
| 31 | 20 | Antonio Gates | 70.57 | 66.10 | 69.38 | 358 | Chargers |
| 32 | 21 | Josh Hill | 69.94 | 61.60 | 71.34 | 329 | Saints |
| 33 | 22 | Benjamin Watson | 69.85 | 60.70 | 71.79 | 452 | Ravens |
| 34 | 23 | Stephen Anderson | 69.79 | 59.10 | 72.75 | 325 | Texans |
| 35 | 24 | Jason Witten | 69.73 | 60.60 | 71.65 | 596 | Cowboys |
| 36 | 25 | Nick Boyle | 69.63 | 65.40 | 68.28 | 292 | Ravens |
| 37 | 26 | Austin Traylor | 69.57 | 58.60 | 72.72 | 159 | Broncos |
| 38 | 27 | George Kittle | 69.54 | 65.80 | 67.87 | 402 | 49ers |
| 39 | 28 | Martellus Bennett | 69.46 | 59.20 | 72.14 | 288 | Patriots |
| 40 | 29 | David Njoku | 69.21 | 66.40 | 66.91 | 335 | Browns |
| 41 | 30 | Niles Paul | 69.19 | 51.30 | 76.95 | 121 | Commanders |
| 42 | 31 | Austin Seferian-Jenkins | 68.91 | 60.80 | 70.15 | 432 | Jets |
| 43 | 32 | A.J. Derby | 68.71 | 55.60 | 73.28 | 263 | Dolphins |
| 44 | 33 | Levine Toilolo | 68.62 | 67.30 | 65.33 | 208 | Falcons |
| 45 | 34 | Ed Dickson | 68.27 | 59.20 | 70.15 | 439 | Panthers |
| 46 | 35 | Anthony Fasano | 68.26 | 58.90 | 70.33 | 263 | Dolphins |
| 47 | 36 | Austin Hooper | 68.24 | 61.70 | 68.44 | 521 | Falcons |
| 48 | 37 | Nick O'Leary | 67.52 | 59.30 | 68.84 | 297 | Bills |
| 49 | 38 | Brent Celek | 67.52 | 53.10 | 72.97 | 247 | Eagles |
| 50 | 39 | Brandon Williams | 67.38 | 57.80 | 69.60 | 121 | Colts |
| 51 | 40 | C.J. Fiedorowicz | 67.35 | 59.80 | 68.22 | 129 | Texans |
| 52 | 41 | Luke Willson | 67.29 | 60.10 | 67.92 | 197 | Seahawks |
| 53 | 42 | Jermaine Gresham | 67.17 | 62.80 | 65.91 | 433 | Cardinals |
| 54 | 43 | Ryan Griffin | 67.12 | 58.90 | 68.43 | 158 | Texans |
| 55 | 44 | Nick Vannett | 67.06 | 58.80 | 68.40 | 144 | Seahawks |
| 56 | 45 | Tyler Kroft | 66.97 | 60.40 | 67.19 | 464 | Bengals |
| 57 | 46 | Michael Hoomanawanui | 66.95 | 53.20 | 71.95 | 176 | Saints |
| 58 | 47 | Dion Sims | 66.85 | 56.80 | 69.38 | 239 | Bears |
| 59 | 48 | Virgil Green | 66.32 | 53.60 | 70.63 | 227 | Broncos |
| 60 | 49 | Richard Rodgers | 66.15 | 56.50 | 68.41 | 185 | Packers |
| 61 | 50 | Seth DeValve | 66.03 | 59.20 | 66.42 | 385 | Browns |
| 62 | 51 | Gerald Everett | 65.92 | 52.20 | 70.90 | 239 | Rams |
| 63 | 52 | Tyler Higbee | 65.70 | 62.50 | 63.66 | 385 | Rams |
| 64 | 53 | Lee Smith | 65.41 | 58.60 | 65.78 | 126 | Raiders |
| 65 | 54 | Troy Niklas | 65.22 | 59.10 | 65.13 | 192 | Cardinals |
| 66 | 55 | Dwayne Allen | 65.22 | 55.60 | 67.46 | 253 | Patriots |
| 67 | 56 | Jesse James | 64.50 | 57.20 | 65.20 | 540 | Steelers |
| 68 | 57 | Daniel Brown | 64.18 | 54.70 | 66.34 | 184 | Bears |
| 69 | 58 | Lance Kendricks | 64.07 | 49.20 | 69.82 | 268 | Packers |
| 70 | 59 | Jonnu Smith | 63.39 | 53.60 | 65.75 | 234 | Titans |
| 71 | 60 | Demetrius Harris | 62.33 | 55.20 | 62.91 | 295 | Chiefs |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 72 | 1 | Ben Koyack | 61.87 | 56.00 | 61.61 | 158 | Jaguars |
| 73 | 2 | Randall Telfer | 57.41 | 47.50 | 59.85 | 137 | Browns |

## WR — Wide Receiver

- **Season used:** `2017`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Julio Jones | 89.89 | 91.80 | 84.45 | 561 | Falcons |
| 2 | 2 | Antonio Brown | 88.98 | 91.30 | 83.26 | 627 | Steelers |
| 3 | 3 | Michael Thomas | 86.67 | 91.40 | 79.35 | 621 | Saints |
| 4 | 4 | Keenan Allen | 85.27 | 89.30 | 78.41 | 579 | Chargers |
| 5 | 5 | DeAndre Hopkins | 85.23 | 90.20 | 77.75 | 629 | Texans |
| 6 | 6 | Josh Gordon | 84.10 | 76.20 | 85.20 | 179 | Browns |
| 7 | 7 | A.J. Green | 83.69 | 82.20 | 80.51 | 542 | Bengals |
| 8 | 8 | Chris Godwin | 83.47 | 81.90 | 80.35 | 274 | Buccaneers |
| 9 | 9 | Adam Thielen | 82.81 | 82.40 | 78.91 | 693 | Vikings |
| 10 | 10 | Doug Baldwin | 82.75 | 84.40 | 77.48 | 599 | Seahawks |
| 11 | 11 | Tyreek Hill | 82.37 | 79.90 | 79.85 | 552 | Chiefs |
| 12 | 12 | Stefon Diggs | 81.34 | 82.60 | 76.33 | 570 | Vikings |
| 13 | 13 | Robert Woods | 80.79 | 81.00 | 76.49 | 428 | Rams |
| 14 | 14 | Keelan Cole Sr. | 80.72 | 74.00 | 81.03 | 534 | Jaguars |
| 15 | 15 | DeSean Jackson | 80.49 | 76.80 | 78.79 | 464 | Buccaneers |

### Good (39 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Alshon Jeffery | 79.93 | 77.30 | 77.52 | 691 | Eagles |
| 17 | 2 | JuJu Smith-Schuster | 79.90 | 73.10 | 80.26 | 512 | Steelers |
| 18 | 3 | Golden Tate | 79.67 | 82.90 | 73.35 | 575 | Lions |
| 19 | 4 | Marvin Jones Jr. | 79.66 | 76.30 | 77.73 | 674 | Lions |
| 20 | 5 | Mike Evans | 79.64 | 79.30 | 75.70 | 628 | Buccaneers |
| 21 | 6 | Cooper Kupp | 79.54 | 78.10 | 76.34 | 502 | Rams |
| 22 | 7 | Marquise Goodwin | 79.39 | 77.10 | 76.75 | 506 | 49ers |
| 23 | 8 | Pierre Garcon | 79.27 | 77.50 | 76.28 | 292 | 49ers |
| 24 | 9 | Larry Fitzgerald | 78.50 | 80.00 | 73.34 | 700 | Cardinals |
| 25 | 10 | Davante Adams | 78.12 | 80.50 | 72.37 | 538 | Packers |
| 26 | 11 | T.Y. Hilton | 78.11 | 68.90 | 80.09 | 585 | Colts |
| 27 | 12 | Jarvis Landry | 78.09 | 79.50 | 72.99 | 625 | Dolphins |
| 28 | 13 | Rishard Matthews | 77.91 | 71.90 | 77.75 | 568 | Titans |
| 29 | 14 | Ted Ginn Jr. | 77.86 | 71.90 | 77.67 | 499 | Saints |
| 30 | 15 | Odell Beckham Jr. | 77.34 | 69.20 | 78.60 | 153 | Giants |
| 31 | 16 | Brandin Cooks | 77.33 | 70.70 | 77.59 | 771 | Patriots |
| 32 | 17 | Danny Amendola | 77.19 | 77.40 | 72.89 | 550 | Patriots |
| 33 | 18 | Allen Hurns | 76.98 | 73.90 | 74.87 | 392 | Jaguars |
| 34 | 19 | Sammy Watkins | 76.90 | 69.20 | 77.87 | 526 | Rams |
| 35 | 20 | Kenny Golladay | 76.84 | 68.00 | 78.56 | 318 | Lions |
| 36 | 21 | Devin Funchess | 76.73 | 72.20 | 75.59 | 593 | Panthers |
| 37 | 22 | Mohamed Sanu | 76.50 | 77.20 | 71.86 | 535 | Falcons |
| 38 | 23 | Brenton Bersin | 76.42 | 68.10 | 77.80 | 113 | Panthers |
| 39 | 24 | Marqise Lee | 76.17 | 76.00 | 72.11 | 520 | Jaguars |
| 40 | 25 | Demaryius Thomas | 75.66 | 74.10 | 72.54 | 582 | Broncos |
| 41 | 26 | Alex Erickson | 75.55 | 66.60 | 77.35 | 117 | Bengals |
| 42 | 27 | Dez Bryant | 75.44 | 72.70 | 73.10 | 568 | Cowboys |
| 43 | 28 | Jarius Wright | 75.42 | 72.30 | 73.33 | 234 | Vikings |
| 44 | 29 | Jamison Crowder | 75.25 | 72.70 | 72.78 | 466 | Commanders |
| 45 | 30 | Martavis Bryant | 75.23 | 68.90 | 75.29 | 514 | Steelers |
| 46 | 31 | Emmanuel Sanders | 75.16 | 68.10 | 75.70 | 435 | Broncos |
| 47 | 32 | Mike Wallace | 75.02 | 69.00 | 74.87 | 476 | Ravens |
| 48 | 33 | Travis Benjamin | 74.92 | 66.80 | 76.17 | 400 | Chargers |
| 49 | 34 | Albert Wilson | 74.91 | 73.80 | 71.49 | 397 | Chiefs |
| 50 | 35 | T.J. Jones | 74.83 | 74.40 | 70.95 | 287 | Lions |
| 51 | 36 | DeVante Parker | 74.76 | 69.50 | 74.10 | 475 | Dolphins |
| 52 | 37 | Kendall Wright | 74.58 | 73.20 | 71.33 | 435 | Bears |
| 53 | 38 | Kendrick Bourne | 74.21 | 62.60 | 77.79 | 212 | 49ers |
| 54 | 39 | Jordy Nelson | 74.13 | 68.60 | 73.65 | 539 | Packers |

### Starter (80 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 55 | 1 | Tyler Lockett | 73.92 | 67.30 | 74.17 | 517 | Seahawks |
| 56 | 2 | Nelson Agholor | 73.83 | 74.90 | 68.95 | 594 | Eagles |
| 57 | 3 | Jermaine Kearse | 73.67 | 69.70 | 72.15 | 567 | Jets |
| 58 | 4 | Jeremy Maclin | 73.60 | 69.10 | 72.44 | 323 | Ravens |
| 59 | 5 | Tyrell Williams | 73.40 | 62.00 | 76.83 | 553 | Chargers |
| 60 | 6 | Cody Latimer | 73.38 | 70.00 | 71.46 | 246 | Broncos |
| 61 | 7 | Brice Butler | 73.22 | 65.90 | 73.94 | 158 | Cowboys |
| 62 | 8 | Ryan Grant | 73.12 | 69.20 | 71.56 | 402 | Commanders |
| 63 | 9 | Michael Crabtree | 73.09 | 72.00 | 69.65 | 411 | Raiders |
| 64 | 10 | Sterling Shepard | 73.04 | 70.70 | 70.43 | 464 | Giants |
| 65 | 11 | Taylor Gabriel | 72.80 | 63.70 | 74.70 | 381 | Falcons |
| 66 | 12 | Amari Cooper | 72.68 | 63.60 | 74.56 | 471 | Raiders |
| 67 | 13 | Paul Richardson Jr. | 72.39 | 66.90 | 71.89 | 595 | Seahawks |
| 68 | 14 | Will Fuller V | 72.26 | 67.20 | 71.47 | 322 | Texans |
| 69 | 15 | Chris Hogan | 72.21 | 61.30 | 75.31 | 531 | Patriots |
| 70 | 16 | Terrance Williams | 72.08 | 65.30 | 72.44 | 461 | Cowboys |
| 71 | 17 | Deonte Thompson | 72.05 | 63.60 | 73.52 | 462 | Bills |
| 72 | 18 | Eric Decker | 71.91 | 69.40 | 69.41 | 516 | Titans |
| 73 | 19 | Willie Snead IV | 71.65 | 60.20 | 75.12 | 176 | Saints |
| 74 | 20 | Dontrelle Inman | 71.61 | 66.00 | 71.18 | 286 | Bears |
| 75 | 21 | Kenny Stills | 71.52 | 59.90 | 75.10 | 649 | Dolphins |
| 76 | 22 | Brandon Marshall | 71.32 | 62.20 | 73.24 | 183 | Giants |
| 77 | 23 | Josh Doctson | 71.32 | 62.50 | 73.03 | 470 | Commanders |
| 78 | 24 | Tavarres King | 71.27 | 62.30 | 73.09 | 262 | Giants |
| 79 | 25 | Randall Cobb | 71.14 | 64.40 | 71.46 | 491 | Packers |
| 80 | 26 | Josh Bellamy | 71.02 | 63.80 | 71.67 | 282 | Bears |
| 81 | 27 | Louis Murphy Jr. | 70.88 | 63.70 | 71.50 | 107 | 49ers |
| 82 | 28 | Cordarrelle Patterson | 70.82 | 65.50 | 70.20 | 288 | Raiders |
| 83 | 29 | J.J. Nelson | 70.64 | 59.00 | 74.23 | 386 | Cardinals |
| 84 | 30 | Brandon Tate | 70.62 | 64.50 | 70.53 | 102 | Bills |
| 85 | 31 | Jaron Brown | 70.50 | 59.40 | 73.74 | 566 | Cardinals |
| 86 | 32 | John Brown | 70.30 | 58.00 | 74.34 | 340 | Cardinals |
| 87 | 33 | Kenny Britt | 70.24 | 56.20 | 75.44 | 277 | Patriots |
| 88 | 34 | Jordan Taylor | 70.06 | 62.80 | 70.74 | 173 | Broncos |
| 89 | 35 | Trent Taylor | 69.78 | 68.40 | 66.54 | 379 | 49ers |
| 90 | 36 | Mack Hollins | 69.74 | 63.50 | 69.74 | 198 | Eagles |
| 91 | 37 | Adam Humphries | 69.70 | 63.00 | 70.00 | 529 | Buccaneers |
| 92 | 38 | Corey Davis | 69.58 | 63.20 | 69.67 | 436 | Titans |
| 93 | 39 | Chris Conley | 69.57 | 63.10 | 69.71 | 179 | Chiefs |
| 94 | 40 | Dede Westbrook | 69.57 | 62.10 | 70.39 | 322 | Jaguars |
| 95 | 41 | Phillip Dorsett | 69.54 | 58.00 | 73.07 | 255 | Patriots |
| 96 | 42 | Brandon Coleman | 69.13 | 55.50 | 74.05 | 394 | Saints |
| 97 | 43 | De'Anthony Thomas | 69.08 | 62.10 | 69.56 | 117 | Chiefs |
| 98 | 44 | ArDarius Stewart | 68.88 | 57.00 | 72.64 | 116 | Jets |
| 99 | 45 | Cole Beasley | 68.60 | 59.90 | 70.24 | 409 | Cowboys |
| 100 | 46 | Taywan Taylor | 68.56 | 55.90 | 72.83 | 189 | Titans |
| 101 | 47 | Donte Moncrief | 68.31 | 61.10 | 68.95 | 403 | Colts |
| 102 | 48 | Terrelle Pryor Sr. | 68.24 | 57.90 | 70.97 | 247 | Commanders |
| 103 | 49 | Tyler Boyd | 68.20 | 62.00 | 68.17 | 219 | Bengals |
| 104 | 50 | Brandon LaFell | 68.02 | 57.80 | 70.66 | 553 | Bengals |
| 105 | 51 | Aldrick Robinson | 67.80 | 55.90 | 71.56 | 278 | 49ers |
| 106 | 52 | Damiere Byrd | 67.41 | 59.80 | 68.31 | 105 | Panthers |
| 107 | 53 | Markus Wheaton | 67.13 | 49.80 | 74.52 | 125 | Bears |
| 108 | 54 | Mike Williams | 67.09 | 52.60 | 72.58 | 145 | Chargers |
| 109 | 55 | Chester Rogers | 67.02 | 58.10 | 68.80 | 284 | Colts |
| 110 | 56 | Corey Coleman | 66.89 | 60.40 | 67.05 | 311 | Browns |
| 111 | 57 | Torrey Smith | 66.65 | 56.70 | 69.11 | 542 | Eagles |
| 112 | 58 | Bruce Ellington | 66.44 | 57.00 | 68.56 | 359 | Texans |
| 113 | 59 | Justin Hardy | 65.86 | 59.60 | 65.86 | 251 | Falcons |
| 114 | 60 | Roger Lewis Jr. | 65.80 | 58.60 | 66.43 | 468 | Giants |
| 115 | 61 | Johnny Holton | 65.71 | 48.60 | 72.95 | 142 | Raiders |
| 116 | 62 | Bennie Fowler | 65.56 | 56.00 | 67.76 | 379 | Broncos |
| 117 | 63 | Andre Holmes | 65.48 | 63.30 | 62.77 | 205 | Bills |
| 118 | 64 | Jordan Matthews | 65.42 | 52.70 | 69.74 | 327 | Bills |
| 119 | 65 | Chris Moore | 65.36 | 59.00 | 65.44 | 246 | Ravens |
| 120 | 66 | Michael Campanaro | 64.54 | 60.70 | 62.93 | 179 | Ravens |
| 121 | 67 | Eli Rogers | 64.51 | 54.20 | 67.21 | 302 | Steelers |
| 122 | 68 | Josh Reynolds | 64.42 | 58.40 | 64.27 | 164 | Rams |
| 123 | 69 | Geronimo Allison | 64.09 | 50.90 | 68.72 | 281 | Packers |
| 124 | 70 | Rashard Higgins | 64.04 | 50.40 | 68.96 | 504 | Browns |
| 125 | 71 | Chad Hansen | 63.85 | 53.70 | 66.45 | 242 | Jets |
| 126 | 72 | Kaelin Clay | 63.46 | 55.10 | 64.87 | 188 | Panthers |
| 127 | 73 | Laquon Treadwell | 63.25 | 53.30 | 65.71 | 344 | Vikings |
| 128 | 74 | Zay Jones | 63.10 | 55.00 | 64.34 | 520 | Bills |
| 129 | 75 | Ricardo Louis | 62.97 | 53.10 | 65.38 | 406 | Browns |
| 130 | 76 | Seth Roberts | 62.90 | 54.00 | 64.67 | 514 | Raiders |
| 131 | 77 | Josh Malone | 62.88 | 50.80 | 66.77 | 161 | Bengals |
| 132 | 78 | Braxton Miller | 62.84 | 55.20 | 63.77 | 268 | Texans |
| 133 | 79 | Demarcus Robinson | 62.43 | 54.50 | 63.55 | 411 | Chiefs |
| 134 | 80 | Travis Rudolph | 62.07 | 48.10 | 67.21 | 163 | Giants |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 135 | 1 | Leonte Carroo | 61.62 | 54.90 | 61.93 | 101 | Dolphins |
| 136 | 2 | Breshad Perriman | 60.86 | 45.10 | 67.20 | 224 | Ravens |
| 137 | 3 | Kamar Aiken | 60.37 | 47.00 | 65.12 | 372 | Colts |
| 138 | 4 | Russell Shepard | 60.03 | 46.00 | 65.22 | 312 | Panthers |
| 139 | 5 | Curtis Samuel | 59.80 | 56.90 | 57.56 | 140 | Panthers |
