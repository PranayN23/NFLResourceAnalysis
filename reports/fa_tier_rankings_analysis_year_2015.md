# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:38:17Z
- **Requested analysis_year:** 2015 (clamped to 2015)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Travis Frederick | 89.55 | 82.80 | 89.88 | 1020 | Cowboys |
| 2 | 2 | Weston Richburg | 86.50 | 83.30 | 84.46 | 1013 | Giants |
| 3 | 3 | Corey Linsley | 86.20 | 79.04 | 86.81 | 942 | Packers |
| 4 | 4 | Ryan Kalil | 83.83 | 76.00 | 84.89 | 1193 | Panthers |
| 5 | 5 | Ben Jones | 83.78 | 74.90 | 85.53 | 1245 | Texans |
| 6 | 6 | Mike Pouncey | 83.34 | 75.16 | 84.63 | 786 | Dolphins |
| 7 | 7 | Matt Paradis | 83.28 | 75.70 | 84.16 | 1299 | Broncos |
| 8 | 8 | Jason Kelce | 83.23 | 74.10 | 85.15 | 1156 | Eagles |
| 9 | 9 | Rodney Hudson | 82.95 | 72.99 | 85.43 | 801 | Raiders |
| 10 | 10 | Nick Mangold | 82.49 | 73.87 | 84.07 | 933 | Jets |
| 11 | 11 | Jeremy Zuttah | 81.84 | 72.38 | 83.98 | 610 | Ravens |
| 12 | 12 | Stefen Wisniewski | 80.96 | 72.50 | 82.43 | 1058 | Jaguars |
| 13 | 13 | Alex Mack | 80.92 | 71.80 | 82.84 | 1103 | Browns |
| 14 | 14 | Eric Wood | 80.32 | 71.70 | 81.90 | 1075 | Bills |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Max Unger | 79.47 | 70.50 | 81.28 | 1154 | Saints |
| 16 | 2 | A.Q. Shipley | 79.37 | 65.88 | 84.20 | 155 | Cardinals |
| 17 | 3 | J.C. Tretter | 79.36 | 70.35 | 81.20 | 445 | Packers |
| 18 | 4 | Evan Smith | 78.97 | 68.45 | 81.81 | 370 | Buccaneers |
| 19 | 5 | Josh LeRibeus | 78.51 | 68.10 | 81.29 | 731 | Commanders |
| 20 | 6 | Mitch Morse | 74.81 | 68.00 | 75.18 | 916 | Chiefs |
| 21 | 7 | Wesley Johnson | 74.79 | 64.45 | 77.51 | 170 | Jets |
| 22 | 8 | Joe Hawley | 74.22 | 63.79 | 77.00 | 981 | Buccaneers |
| 23 | 9 | Brian Schwenke | 74.10 | 61.66 | 78.22 | 303 | Titans |

### Starter (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Patrick Lewis | 73.88 | 62.31 | 77.42 | 723 | Seahawks |
| 25 | 2 | Russell Bodine | 73.02 | 62.20 | 76.06 | 1126 | Bengals |
| 26 | 3 | David Andrews | 73.00 | 61.33 | 76.62 | 776 | Patriots |
| 27 | 4 | Will Montgomery | 72.95 | 60.33 | 77.19 | 188 | Bears |
| 28 | 5 | James Stone | 72.91 | 62.72 | 75.54 | 133 | Falcons |
| 29 | 6 | Lyle Sendlein | 72.69 | 61.90 | 75.72 | 1112 | Cardinals |
| 30 | 7 | Jonotthan Harrison | 72.49 | 61.81 | 75.45 | 669 | Colts |
| 31 | 8 | Bryan Stork | 72.47 | 59.21 | 77.14 | 612 | Patriots |
| 32 | 9 | Travis Swanson | 72.02 | 60.29 | 75.67 | 947 | Lions |
| 33 | 10 | Khaled Holmes | 71.45 | 59.72 | 75.10 | 485 | Colts |
| 34 | 11 | Tony Bergstrom | 71.41 | 64.18 | 72.07 | 250 | Raiders |
| 35 | 12 | Tim Barnes | 71.22 | 58.42 | 75.59 | 955 | Rams |
| 36 | 13 | Drew Nowak | 70.97 | 61.23 | 73.29 | 461 | Seahawks |
| 37 | 14 | Cody Wallace | 70.15 | 56.80 | 74.88 | 1207 | Steelers |
| 38 | 15 | J.D. Walton | 70.07 | 58.16 | 73.84 | 100 | Chargers |
| 39 | 16 | Daniel Kilgore | 68.42 | 58.28 | 71.02 | 267 | 49ers |
| 40 | 17 | Kory Lichtensteiger | 67.87 | 55.76 | 71.77 | 437 | Commanders |
| 41 | 18 | Demetrius Rhaney | 67.02 | 58.36 | 68.63 | 124 | Rams |
| 42 | 19 | Hroniss Grasu | 65.03 | 55.13 | 67.47 | 552 | Bears |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 43 | 1 | Andy Gallik | 61.48 | 46.25 | 67.46 | 505 | Titans |
| 44 | 2 | Trevor Robinson | 61.41 | 46.14 | 67.42 | 980 | Chargers |
| 45 | 3 | Chris Watt | 60.20 | 51.69 | 61.70 | 177 | Chargers |

## CB — Cornerback

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Josh Norman | 88.60 | 89.90 | 86.59 | 1240 | Panthers |
| 2 | 2 | Patrick Peterson | 87.23 | 84.50 | 84.89 | 1116 | Cardinals |
| 3 | 3 | Jason Verrett | 86.80 | 88.83 | 86.48 | 718 | Chargers |
| 4 | 4 | Richard Sherman | 86.59 | 79.40 | 87.22 | 1098 | Seahawks |
| 5 | 5 | Johnathan Joseph | 85.13 | 83.00 | 82.59 | 925 | Texans |
| 6 | 6 | Delvin Breaux Sr. | 83.79 | 75.90 | 84.88 | 941 | Saints |
| 7 | 7 | Darius Slay | 82.23 | 77.00 | 82.58 | 995 | Lions |
| 8 | 8 | Chris Harris Jr. | 82.05 | 77.50 | 80.91 | 1261 | Broncos |
| 9 | 9 | Ronald Darby | 81.96 | 74.90 | 83.54 | 911 | Bills |
| 10 | 10 | Casey Hayward Jr. | 81.88 | 77.50 | 83.34 | 1038 | Packers |
| 11 | 11 | Dominique Rodgers-Cromartie | 81.78 | 76.90 | 81.39 | 890 | Giants |
| 12 | 12 | Trumaine Johnson | 81.13 | 78.80 | 81.75 | 901 | Rams |
| 13 | 13 | Adam Jones | 80.62 | 77.80 | 78.85 | 923 | Bengals |
| 14 | 14 | Aqib Talib | 80.39 | 73.40 | 81.08 | 1186 | Broncos |
| 15 | 15 | David Amerson | 80.11 | 76.00 | 79.51 | 882 | Raiders |
| 16 | 16 | Captain Munnerlyn | 80.08 | 75.91 | 78.69 | 745 | Vikings |
| 17 | 17 | Malcolm Butler | 80.02 | 73.30 | 83.06 | 1234 | Patriots |

### Good (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Logan Ryan | 79.91 | 70.20 | 82.22 | 1129 | Patriots |
| 19 | 2 | Sam Shields | 79.85 | 74.62 | 80.94 | 756 | Packers |
| 20 | 3 | Quinten Rollins | 79.65 | 70.30 | 83.80 | 355 | Packers |
| 21 | 4 | Byron Jones | 79.45 | 72.30 | 80.05 | 872 | Cowboys |
| 22 | 5 | Brandon Boykin | 79.22 | 71.75 | 82.63 | 397 | Steelers |
| 23 | 6 | Sean Smith | 79.07 | 71.40 | 80.54 | 948 | Chiefs |
| 24 | 7 | Bradley Roby | 78.89 | 74.95 | 77.35 | 778 | Broncos |
| 25 | 8 | Vontae Davis | 78.61 | 69.70 | 80.38 | 1046 | Colts |
| 26 | 9 | Bashaud Breeland | 78.44 | 70.40 | 79.63 | 983 | Commanders |
| 27 | 10 | Desmond Trufant | 78.34 | 67.60 | 81.34 | 963 | Falcons |
| 28 | 11 | Marcus Williams | 78.02 | 71.48 | 83.30 | 283 | Jets |
| 29 | 12 | Cody Riggs | 77.63 | 69.08 | 87.49 | 170 | Titans |
| 30 | 13 | Janoris Jenkins | 77.52 | 72.30 | 77.98 | 1035 | Rams |
| 31 | 14 | Ross Cockrell | 77.09 | 72.60 | 81.38 | 719 | Steelers |
| 32 | 15 | Leon Hall | 76.42 | 71.15 | 78.59 | 721 | Bengals |
| 33 | 16 | Darrelle Revis | 76.20 | 66.30 | 79.66 | 884 | Jets |
| 34 | 17 | Davon House | 75.10 | 66.60 | 77.85 | 1035 | Jaguars |
| 35 | 18 | Kyle Fuller | 74.72 | 70.10 | 73.64 | 1021 | Bears |
| 36 | 19 | Stephon Gilmore | 74.47 | 68.48 | 78.05 | 788 | Bills |

### Starter (68 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 37 | 1 | Tramaine Brock Sr. | 73.96 | 68.20 | 78.22 | 1065 | 49ers |
| 38 | 2 | Terrance Mitchell | 73.93 | 67.56 | 85.68 | 131 | Cowboys |
| 39 | 3 | Patrick Robinson | 73.79 | 66.77 | 77.85 | 691 | Chargers |
| 40 | 4 | William Gay | 73.45 | 66.50 | 73.91 | 1207 | Steelers |
| 41 | 5 | Prince Amukamara | 73.38 | 69.64 | 76.80 | 766 | Giants |
| 42 | 6 | Tramon Williams | 73.14 | 62.70 | 76.45 | 964 | Browns |
| 43 | 7 | Brent Grimes | 72.95 | 63.90 | 75.34 | 957 | Dolphins |
| 44 | 8 | K'Waun Williams | 72.82 | 64.81 | 77.12 | 514 | Browns |
| 45 | 9 | Marcus Burley | 72.73 | 66.85 | 77.68 | 203 | Seahawks |
| 46 | 10 | Terence Newman | 72.66 | 64.90 | 74.91 | 994 | Vikings |
| 47 | 11 | Lardarius Webb | 72.58 | 63.40 | 75.37 | 879 | Ravens |
| 48 | 12 | Trae Waynes | 72.53 | 64.02 | 79.24 | 214 | Vikings |
| 49 | 13 | Robert Alford | 72.20 | 63.70 | 76.30 | 903 | Falcons |
| 50 | 14 | Kyle Wilson | 71.86 | 65.61 | 72.70 | 501 | Saints |
| 51 | 15 | Kareem Jackson | 71.73 | 64.54 | 75.27 | 682 | Texans |
| 52 | 16 | Perrish Cox | 71.64 | 64.70 | 73.99 | 700 | Titans |
| 53 | 17 | Quinton Dunbar | 71.20 | 62.57 | 80.08 | 310 | Commanders |
| 54 | 18 | Marcus Peters | 70.93 | 58.10 | 75.31 | 1152 | Chiefs |
| 55 | 19 | Dontae Johnson | 70.58 | 63.46 | 74.95 | 366 | 49ers |
| 56 | 20 | Xavier Rhodes | 70.53 | 59.10 | 74.61 | 1070 | Vikings |
| 57 | 21 | Justin Bethel | 69.81 | 61.86 | 75.94 | 540 | Cardinals |
| 58 | 22 | Darqueze Dennard | 69.46 | 64.30 | 75.77 | 188 | Bengals |
| 59 | 23 | DeAngelo Hall | 69.18 | 61.30 | 76.42 | 499 | Commanders |
| 60 | 24 | Nolan Carroll | 69.16 | 61.14 | 73.16 | 745 | Eagles |
| 61 | 25 | Jeremy Lane | 69.13 | 64.96 | 74.82 | 351 | Seahawks |
| 62 | 26 | Shareece Wright | 69.12 | 61.84 | 73.24 | 487 | Ravens |
| 63 | 27 | Jimmy Smith | 68.58 | 60.60 | 72.23 | 991 | Ravens |
| 64 | 28 | Rashean Mathis | 68.33 | 62.09 | 73.23 | 434 | Lions |
| 65 | 29 | Sterling Moore | 68.09 | 61.76 | 70.84 | 704 | Buccaneers |
| 66 | 30 | Kevin Johnson | 67.99 | 59.00 | 69.82 | 823 | Texans |
| 67 | 31 | Brandon Carr | 67.78 | 56.00 | 71.47 | 1053 | Cowboys |
| 68 | 32 | Cortland Finnegan | 67.78 | 64.28 | 73.24 | 354 | Panthers |
| 69 | 33 | Trevin Wade | 67.65 | 63.84 | 71.66 | 529 | Giants |
| 70 | 34 | Byron Maxwell | 67.51 | 54.40 | 73.44 | 898 | Eagles |
| 71 | 35 | Charles James II | 67.51 | 63.64 | 77.26 | 151 | Texans |
| 72 | 36 | Marcus Roberson | 67.48 | 62.90 | 75.61 | 328 | Rams |
| 73 | 37 | Keenan Lewis | 67.23 | 59.62 | 73.33 | 106 | Saints |
| 74 | 38 | Darrin Walls | 67.23 | 59.88 | 75.05 | 121 | Jets |
| 75 | 39 | Brice McCain | 67.16 | 58.47 | 70.45 | 706 | Dolphins |
| 76 | 40 | Tarell Brown | 66.90 | 60.53 | 74.38 | 162 | Patriots |
| 77 | 41 | A.J. Bouye | 66.90 | 59.55 | 72.95 | 208 | Texans |
| 78 | 42 | Alan Ball | 66.78 | 57.01 | 72.67 | 248 | Bears |
| 79 | 43 | Justin Coleman | 66.41 | 58.24 | 72.89 | 408 | Patriots |
| 80 | 44 | Alterraun Verner | 66.20 | 54.05 | 71.28 | 575 | Buccaneers |
| 81 | 45 | Kenneth Acker | 66.11 | 59.11 | 68.69 | 806 | 49ers |
| 82 | 46 | Phillip Adams | 66.08 | 59.35 | 70.88 | 426 | Falcons |
| 83 | 47 | Josh Shaw | 66.03 | 65.20 | 69.72 | 117 | Bengals |
| 84 | 48 | Mario Butler | 65.74 | 59.96 | 78.85 | 126 | Bills |
| 85 | 49 | Leodis McKelvin | 65.67 | 59.19 | 71.56 | 388 | Bills |
| 86 | 50 | Jerraud Powers | 65.30 | 55.00 | 68.52 | 906 | Cardinals |
| 87 | 51 | Kyle Arrington | 65.13 | 53.96 | 68.93 | 333 | Ravens |
| 88 | 52 | Corey White | 65.05 | 61.08 | 69.58 | 113 | Cardinals |
| 89 | 53 | Bryce Callahan | 65.04 | 61.44 | 71.60 | 322 | Bears |
| 90 | 54 | Aaron Colvin | 65.01 | 61.10 | 67.35 | 1069 | Jaguars |
| 91 | 55 | Tony Lippett | 64.90 | 63.96 | 77.33 | 132 | Dolphins |
| 92 | 56 | Bobby McCain | 64.85 | 59.86 | 70.26 | 390 | Dolphins |
| 93 | 57 | Charles Tillman | 64.78 | 58.18 | 73.15 | 711 | Panthers |
| 94 | 58 | Bene Benwikere | 64.56 | 56.45 | 69.32 | 785 | Panthers |
| 95 | 59 | Antonio Cromartie | 64.22 | 52.20 | 68.58 | 898 | Jets |
| 96 | 60 | Eric Rowe | 64.21 | 57.58 | 70.72 | 504 | Eagles |
| 97 | 61 | Demetrius McCray | 64.20 | 58.65 | 67.70 | 224 | Jaguars |
| 98 | 62 | T.J. Carrie | 64.17 | 54.60 | 68.60 | 928 | Raiders |
| 99 | 63 | Marcus Cromartie | 63.90 | 59.40 | 72.53 | 122 | 49ers |
| 100 | 64 | Buster Skrine | 63.76 | 51.22 | 67.96 | 720 | Jets |
| 101 | 65 | Will Blackmon | 63.26 | 54.00 | 67.97 | 861 | Commanders |
| 102 | 66 | Nickell Robey-Coleman | 62.55 | 49.38 | 67.16 | 668 | Bills |
| 103 | 67 | Tracy Porter | 62.54 | 54.20 | 69.03 | 845 | Bears |
| 104 | 68 | Coty Sensabaugh | 62.52 | 53.40 | 65.78 | 1003 | Titans |

### Rotation/backup (39 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 105 | 1 | Phillip Gaines | 61.97 | 60.76 | 70.20 | 167 | Chiefs |
| 106 | 2 | Dwayne Gratz | 61.94 | 54.36 | 66.48 | 454 | Jaguars |
| 107 | 3 | E.J. Biggers | 61.81 | 49.87 | 66.23 | 564 | Eagles |
| 108 | 4 | Damian Swann | 61.42 | 61.31 | 68.20 | 228 | Saints |
| 109 | 5 | Dre Kirkpatrick | 61.30 | 49.50 | 66.04 | 1089 | Bengals |
| 110 | 6 | Robert McClain | 61.06 | 56.38 | 66.88 | 220 | Panthers |
| 111 | 7 | Steve Williams | 60.71 | 53.04 | 67.64 | 287 | Chargers |
| 112 | 8 | Neiko Thorpe | 60.68 | 54.51 | 66.56 | 454 | Raiders |
| 113 | 9 | Leonard Johnson | 60.43 | 54.88 | 66.21 | 158 | Patriots |
| 114 | 10 | Josh Wilson | 60.20 | 49.32 | 68.39 | 276 | Lions |
| 115 | 11 | Trumaine McBride | 59.81 | 50.60 | 66.78 | 337 | Giants |
| 116 | 12 | Blidi Wreh-Wilson | 59.47 | 54.40 | 65.77 | 293 | Titans |
| 117 | 13 | Jason McCourty | 59.26 | 49.51 | 67.85 | 212 | Titans |
| 118 | 14 | Johnthan Banks | 59.07 | 50.31 | 63.66 | 431 | Buccaneers |
| 119 | 15 | Mike Jenkins | 59.07 | 53.36 | 65.69 | 330 | Buccaneers |
| 120 | 16 | Pierre Desir | 59.04 | 47.78 | 70.07 | 389 | Browns |
| 121 | 17 | Craig Mager | 58.71 | 61.99 | 65.78 | 226 | Chargers |
| 122 | 18 | Greg Toler | 58.68 | 41.96 | 70.02 | 685 | Colts |
| 123 | 19 | Morris Claiborne | 58.48 | 50.51 | 67.22 | 661 | Cowboys |
| 124 | 20 | Valentino Blake | 58.47 | 47.70 | 65.55 | 1027 | Steelers |
| 125 | 21 | Tyler Patmon | 58.40 | 50.88 | 65.50 | 305 | Dolphins |
| 126 | 22 | Jayron Hosley | 57.20 | 56.73 | 59.60 | 525 | Giants |
| 127 | 23 | Brandon Browner | 57.19 | 40.00 | 67.40 | 1023 | Saints |
| 128 | 24 | Jalil Brown | 56.62 | 55.11 | 65.12 | 308 | Colts |
| 129 | 25 | Nevin Lawson | 56.54 | 51.84 | 64.22 | 561 | Lions |
| 130 | 26 | Jalen Collins | 56.30 | 52.70 | 60.78 | 300 | Falcons |
| 131 | 27 | Joe Haden | 55.23 | 40.00 | 67.46 | 284 | Browns |
| 132 | 28 | Brandon Flowers | 55.01 | 40.00 | 64.50 | 602 | Chargers |
| 133 | 29 | Johnson Bademosi | 54.96 | 46.46 | 63.96 | 167 | Browns |
| 134 | 30 | Deji Olatoye | 54.72 | 55.42 | 70.09 | 138 | Cowboys |
| 135 | 31 | Jude Adjei-Barimah | 54.67 | 55.10 | 53.35 | 468 | Buccaneers |
| 136 | 32 | Chris Culliver | 54.59 | 40.00 | 65.98 | 350 | Commanders |
| 137 | 33 | D.J. Hayden | 53.97 | 40.00 | 62.66 | 897 | Raiders |
| 138 | 34 | Nick Marshall | 51.69 | 54.32 | 59.19 | 142 | Jaguars |
| 139 | 35 | Sherrick McManis | 51.54 | 48.04 | 59.51 | 295 | Bears |
| 140 | 36 | B.W. Webb | 50.65 | 48.56 | 57.15 | 257 | Titans |
| 141 | 37 | Brian Dixon | 50.29 | 55.16 | 49.91 | 111 | Saints |
| 142 | 38 | Charles Gaines | 48.87 | 53.99 | 54.71 | 265 | Browns |
| 143 | 39 | Jamar Taylor | 46.83 | 41.79 | 52.08 | 712 | Dolphins |

## DI — Defensive Interior

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | J.J. Watt | 96.72 | 88.10 | 98.30 | 1033 | Texans |
| 2 | 2 | Aaron Donald | 94.49 | 90.00 | 93.31 | 911 | Rams |
| 3 | 3 | Geno Atkins | 88.55 | 87.36 | 86.64 | 849 | Bengals |
| 4 | 4 | Muhammad Wilkerson | 88.34 | 89.79 | 84.14 | 939 | Jets |
| 5 | 5 | Calais Campbell | 88.19 | 86.49 | 85.48 | 931 | Cardinals |
| 6 | 6 | Kawann Short | 87.41 | 86.23 | 84.03 | 893 | Panthers |
| 7 | 7 | Jurrell Casey | 86.46 | 87.89 | 81.54 | 825 | Titans |
| 8 | 8 | Ndamukong Suh | 86.25 | 87.45 | 81.28 | 983 | Dolphins |
| 9 | 9 | Leonard Williams | 86.10 | 86.64 | 81.58 | 809 | Jets |
| 10 | 10 | Fletcher Cox | 85.09 | 89.08 | 78.26 | 978 | Eagles |
| 11 | 11 | Malik Jackson | 85.06 | 83.93 | 81.64 | 988 | Broncos |
| 12 | 12 | Marcell Dareus | 85.03 | 87.57 | 80.01 | 755 | Bills |
| 13 | 13 | Mike Daniels | 84.93 | 81.45 | 83.09 | 799 | Packers |
| 14 | 14 | Sheldon Richardson | 84.70 | 85.44 | 82.64 | 619 | Jets |
| 15 | 15 | Damon Harrison Sr. | 82.61 | 79.10 | 80.78 | 566 | Jets |
| 16 | 16 | Cameron Heyward | 82.31 | 83.39 | 77.42 | 1103 | Steelers |
| 17 | 17 | Nick Fairley | 81.23 | 80.10 | 81.05 | 420 | Rams |
| 18 | 18 | Chris Baker | 80.91 | 73.19 | 82.40 | 671 | Commanders |
| 19 | 19 | Dan Williams | 80.71 | 83.14 | 75.34 | 568 | Raiders |
| 20 | 20 | Brandon Williams | 80.69 | 78.29 | 80.00 | 723 | Ravens |
| 21 | 21 | Gerald McCoy | 80.61 | 86.02 | 74.30 | 795 | Buccaneers |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Linval Joseph | 79.75 | 79.56 | 77.48 | 568 | Vikings |
| 23 | 2 | Grady Jarrett | 79.44 | 69.81 | 82.72 | 266 | Falcons |
| 24 | 3 | Mario Edwards Jr. | 79.23 | 79.86 | 76.73 | 597 | Raiders |
| 25 | 4 | Derek Wolfe | 79.16 | 80.88 | 75.41 | 808 | Broncos |
| 26 | 5 | Eddie Goldman | 78.46 | 73.71 | 78.49 | 517 | Bears |
| 27 | 6 | Bennie Logan | 77.84 | 68.33 | 81.04 | 578 | Eagles |
| 28 | 7 | Timmy Jernigan | 77.73 | 71.36 | 79.62 | 529 | Ravens |
| 29 | 8 | Christian Covington | 77.01 | 62.50 | 82.52 | 193 | Texans |
| 30 | 9 | Arik Armstead | 76.99 | 79.01 | 71.47 | 376 | 49ers |
| 31 | 10 | Henry Anderson | 76.92 | 73.27 | 82.48 | 447 | Colts |
| 32 | 11 | Johnathan Hankins | 76.86 | 76.25 | 77.78 | 409 | Giants |
| 33 | 12 | Ian Williams | 76.84 | 73.71 | 79.86 | 660 | 49ers |
| 34 | 13 | Malcom Brown | 76.70 | 68.68 | 77.88 | 596 | Patriots |
| 35 | 14 | Sharrif Floyd | 76.33 | 72.50 | 76.38 | 572 | Vikings |
| 36 | 15 | Lawrence Guy Sr. | 76.31 | 64.98 | 80.12 | 478 | Ravens |
| 37 | 16 | Stephen Paea | 75.64 | 74.86 | 75.23 | 214 | Commanders |
| 38 | 17 | Haloti Ngata | 75.62 | 74.56 | 74.05 | 595 | Lions |
| 39 | 18 | Akiem Hicks | 75.52 | 65.78 | 78.16 | 471 | Patriots |
| 40 | 19 | Vance Walker | 74.95 | 66.10 | 76.88 | 438 | Broncos |
| 41 | 20 | Desmond Bryant | 74.93 | 67.73 | 77.75 | 531 | Browns |
| 42 | 21 | Star Lotulelei | 74.85 | 62.53 | 79.22 | 630 | Panthers |
| 43 | 22 | Jaye Howard Jr. | 74.71 | 59.30 | 83.32 | 827 | Chiefs |
| 44 | 23 | Karl Klug | 74.61 | 65.71 | 76.38 | 329 | Titans |
| 45 | 24 | Michael Brockers | 74.27 | 70.35 | 72.72 | 690 | Rams |
| 46 | 25 | Jason Hatcher | 74.22 | 63.26 | 78.51 | 576 | Commanders |

### Starter (86 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 47 | 1 | Datone Jones | 73.33 | 62.43 | 76.74 | 421 | Packers |
| 48 | 2 | Kyle Williams | 72.69 | 61.86 | 81.26 | 341 | Bills |
| 49 | 3 | Terrance Knighton | 72.21 | 64.30 | 73.32 | 403 | Commanders |
| 50 | 4 | Mike Pennel | 72.01 | 59.61 | 76.49 | 314 | Packers |
| 51 | 5 | Leger Douzable | 71.99 | 57.75 | 77.32 | 289 | Jets |
| 52 | 6 | Ra'Shede Hageman | 71.95 | 58.64 | 76.66 | 419 | Falcons |
| 53 | 7 | Paul Soliai | 71.80 | 60.81 | 76.53 | 357 | Falcons |
| 54 | 8 | Jordan Hill | 71.75 | 57.44 | 82.64 | 344 | Seahawks |
| 55 | 9 | Corey Liuget | 71.55 | 63.27 | 75.50 | 443 | Chargers |
| 56 | 10 | Dontari Poe | 71.50 | 68.79 | 69.14 | 852 | Chiefs |
| 57 | 11 | Tyrone Crawford | 71.38 | 63.11 | 72.73 | 705 | Cowboys |
| 58 | 12 | Dominique Easley | 71.30 | 72.16 | 71.76 | 272 | Patriots |
| 59 | 13 | Zach Kerr | 71.28 | 67.10 | 74.07 | 320 | Colts |
| 60 | 14 | Tyson Jackson | 71.23 | 65.59 | 70.82 | 462 | Falcons |
| 61 | 15 | Cedric Thornton | 70.65 | 60.32 | 74.94 | 488 | Eagles |
| 62 | 16 | David Irving | 70.61 | 62.22 | 76.21 | 199 | Cowboys |
| 63 | 17 | Steve McLendon | 70.42 | 59.75 | 74.72 | 408 | Steelers |
| 64 | 18 | Jared Odrick | 70.28 | 64.47 | 69.98 | 874 | Jaguars |
| 65 | 19 | John Jenkins | 69.39 | 61.41 | 72.83 | 530 | Saints |
| 66 | 20 | Sealver Siliga | 69.39 | 57.65 | 77.31 | 280 | Patriots |
| 67 | 21 | Allen Bailey | 69.31 | 62.30 | 71.49 | 655 | Chiefs |
| 68 | 22 | Randy Starks | 69.26 | 57.03 | 74.08 | 466 | Browns |
| 69 | 23 | Danny Shelton | 69.13 | 64.21 | 68.24 | 505 | Browns |
| 70 | 24 | Mike Devito | 69.09 | 61.85 | 75.16 | 317 | Chiefs |
| 71 | 25 | Roy Miller | 69.03 | 65.37 | 68.33 | 556 | Jaguars |
| 72 | 26 | Sammie Lee Hill | 68.86 | 59.87 | 74.76 | 188 | Titans |
| 73 | 27 | Clinton McDonald | 68.84 | 61.00 | 76.05 | 245 | Buccaneers |
| 74 | 28 | Ricky Jean Francois | 68.80 | 57.18 | 73.22 | 419 | Commanders |
| 75 | 29 | Stephon Tuitt | 68.77 | 59.48 | 70.79 | 1005 | Steelers |
| 76 | 30 | Frostee Rucker | 68.66 | 48.17 | 78.67 | 587 | Cardinals |
| 77 | 31 | Pat Sims | 68.39 | 57.73 | 74.98 | 210 | Bengals |
| 78 | 32 | Kendall Langford | 68.24 | 59.75 | 69.73 | 847 | Colts |
| 79 | 33 | Brandon Mebane | 68.24 | 56.18 | 74.29 | 573 | Seahawks |
| 80 | 34 | Chris Canty | 68.11 | 60.83 | 73.58 | 281 | Ravens |
| 81 | 35 | Alan Branch | 68.09 | 58.37 | 71.97 | 505 | Patriots |
| 82 | 36 | Henry Melton | 68.06 | 52.67 | 76.85 | 508 | Buccaneers |
| 83 | 37 | Quinton Dial | 67.99 | 58.97 | 73.69 | 641 | 49ers |
| 84 | 38 | C.J. Wilson | 67.98 | 56.78 | 74.83 | 225 | Lions |
| 85 | 39 | Abry Jones | 67.98 | 55.26 | 74.48 | 363 | Jaguars |
| 86 | 40 | Sylvester Williams | 67.92 | 57.88 | 70.45 | 620 | Broncos |
| 87 | 41 | Tony McDaniel | 67.77 | 53.22 | 74.33 | 293 | Buccaneers |
| 88 | 42 | Jamie Meder | 67.77 | 64.14 | 71.89 | 387 | Browns |
| 89 | 43 | Cory Redding | 67.76 | 53.87 | 74.94 | 180 | Cardinals |
| 90 | 44 | Al Woods | 67.61 | 55.00 | 73.10 | 356 | Titans |
| 91 | 45 | Stefan Charles | 67.51 | 55.27 | 74.74 | 231 | Bills |
| 92 | 46 | Angelo Blackson | 67.49 | 57.44 | 70.02 | 244 | Titans |
| 93 | 47 | Jonathan Babineaux | 67.42 | 55.73 | 71.37 | 545 | Falcons |
| 94 | 48 | Sean Lissemore | 67.28 | 55.27 | 74.25 | 235 | Chargers |
| 95 | 49 | Tom Johnson | 67.16 | 50.14 | 74.76 | 779 | Vikings |
| 96 | 50 | Rodney Gunter | 66.96 | 56.42 | 69.82 | 471 | Cardinals |
| 97 | 51 | John Hughes | 66.92 | 61.55 | 69.99 | 428 | Browns |
| 98 | 52 | Earl Mitchell | 66.64 | 53.67 | 73.21 | 501 | Dolphins |
| 99 | 53 | Letroy Guion | 66.63 | 56.74 | 70.21 | 371 | Packers |
| 100 | 54 | Ahtyba Rubin | 66.61 | 55.09 | 71.48 | 550 | Seahawks |
| 101 | 55 | Demarcus Dobbs | 66.32 | 55.36 | 73.52 | 142 | Seahawks |
| 102 | 56 | DaQuan Jones | 66.20 | 63.11 | 67.61 | 670 | Titans |
| 103 | 57 | Cam Thomas | 66.16 | 54.54 | 69.74 | 198 | Steelers |
| 104 | 58 | Vince Wilfork | 66.04 | 55.03 | 71.72 | 591 | Texans |
| 105 | 59 | Sen'Derrick Marks | 66.04 | 58.93 | 72.86 | 144 | Jaguars |
| 106 | 60 | Xavier Cooper | 65.81 | 52.76 | 72.42 | 361 | Browns |
| 107 | 61 | Shelby Harris | 65.65 | 63.44 | 74.68 | 144 | Raiders |
| 108 | 62 | Justin Ellis | 65.56 | 67.90 | 62.44 | 361 | Raiders |
| 109 | 63 | Domata Peko Sr. | 65.28 | 50.39 | 71.04 | 562 | Bengals |
| 110 | 64 | Tyrunn Walker | 65.27 | 59.47 | 73.10 | 176 | Lions |
| 111 | 65 | Kevin Williams | 65.20 | 52.55 | 69.66 | 553 | Saints |
| 112 | 66 | Antonio Smith | 65.18 | 47.96 | 72.69 | 423 | Broncos |
| 113 | 67 | David Parry | 65.11 | 53.58 | 68.63 | 657 | Colts |
| 114 | 68 | T.Y. McGill | 64.98 | 59.33 | 68.75 | 223 | Colts |
| 115 | 69 | Brandon Thompson | 64.96 | 54.85 | 72.44 | 181 | Bengals |
| 116 | 70 | Ryan Carrethers | 64.96 | 57.33 | 71.08 | 219 | Chargers |
| 117 | 71 | Cullen Jenkins | 64.84 | 48.00 | 73.15 | 731 | Giants |
| 118 | 72 | Beau Allen | 64.71 | 56.14 | 66.25 | 341 | Eagles |
| 119 | 73 | Billy Winn | 64.67 | 54.09 | 71.62 | 325 | Colts |
| 120 | 74 | Jack Crawford | 64.54 | 45.74 | 72.90 | 487 | Cowboys |
| 121 | 75 | Jared Crick | 64.37 | 51.82 | 68.57 | 836 | Texans |
| 122 | 76 | Tyson Alualu | 64.17 | 53.30 | 67.25 | 688 | Jaguars |
| 123 | 77 | Stephen Bowen | 64.16 | 57.09 | 68.97 | 139 | Jets |
| 124 | 78 | Tony Jerod-Eddie | 63.71 | 52.60 | 66.95 | 290 | 49ers |
| 125 | 79 | Red Bryant | 63.57 | 56.88 | 68.54 | 132 | Cardinals |
| 126 | 80 | Bruce Gaston | 63.02 | 55.35 | 71.27 | 179 | Bears |
| 127 | 81 | Akeem Spence | 62.93 | 57.29 | 66.69 | 289 | Buccaneers |
| 128 | 82 | Denico Autry | 62.92 | 49.77 | 69.61 | 681 | Raiders |
| 129 | 83 | Ego Ferguson | 62.76 | 57.47 | 69.94 | 105 | Bears |
| 130 | 84 | Darius Philon | 62.44 | 53.81 | 72.36 | 145 | Chargers |
| 131 | 85 | Corbin Bryant | 62.38 | 51.55 | 65.63 | 633 | Bills |
| 132 | 86 | Brandon Dunn | 62.27 | 61.17 | 65.87 | 139 | Texans |

### Rotation/backup (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 133 | 1 | Dwan Edwards | 61.63 | 42.18 | 71.78 | 447 | Panthers |
| 134 | 2 | Montori Hughes | 61.55 | 55.06 | 69.21 | 118 | Giants |
| 135 | 3 | B.J. Raji | 61.45 | 50.88 | 64.33 | 492 | Packers |
| 136 | 4 | Ed Stinson | 61.05 | 50.85 | 66.03 | 444 | Cardinals |
| 137 | 5 | Jarvis Jenkins | 61.02 | 49.27 | 66.03 | 635 | Bears |
| 138 | 6 | Caraun Reid | 60.83 | 52.23 | 65.27 | 530 | Lions |
| 139 | 7 | Kendall Reyes | 60.55 | 47.57 | 65.03 | 656 | Chargers |
| 140 | 8 | Glenn Dorsey | 60.43 | 46.59 | 71.13 | 351 | 49ers |
| 141 | 9 | Alex Carrington | 60.31 | 56.38 | 68.67 | 172 | Bills |
| 142 | 10 | Will Sutton III | 60.29 | 53.27 | 63.15 | 417 | Bears |
| 143 | 11 | Stacy McGee | 60.18 | 52.52 | 62.88 | 408 | Raiders |
| 144 | 12 | Jay Bromley | 60.16 | 56.93 | 61.28 | 477 | Giants |
| 145 | 13 | Mitch Unrein | 59.81 | 53.38 | 63.58 | 380 | Bears |
| 146 | 14 | Justin Tuck | 59.57 | 50.39 | 77.49 | 244 | Raiders |
| 147 | 15 | Kyle Love | 59.42 | 49.74 | 69.31 | 387 | Panthers |
| 148 | 16 | Jermelle Cudjo | 59.38 | 53.56 | 64.93 | 127 | Lions |
| 149 | 17 | Ricardo Mathews | 59.30 | 49.37 | 63.00 | 510 | Chargers |
| 150 | 18 | Gabe Wright | 59.12 | 56.66 | 67.46 | 135 | Lions |
| 151 | 19 | Jordan Phillips | 58.93 | 50.13 | 61.67 | 429 | Dolphins |
| 152 | 20 | Nick Hayden | 58.88 | 44.81 | 64.09 | 579 | Cowboys |
| 153 | 21 | Tyeler Davison | 58.19 | 52.95 | 57.52 | 528 | Saints |
| 154 | 22 | Geneo Grissom | 58.09 | 55.24 | 63.12 | 131 | Patriots |
| 155 | 23 | Kedric Golston | 57.89 | 48.98 | 61.23 | 205 | Commanders |
| 156 | 24 | Markus Kuhn | 57.60 | 50.14 | 65.27 | 310 | Giants |
| 157 | 25 | Dan McCullers | 57.39 | 62.76 | 53.95 | 110 | Steelers |
| 158 | 26 | Darius Kilgo | 56.95 | 56.70 | 60.25 | 113 | Broncos |
| 159 | 27 | Mike Purcell | 56.93 | 55.66 | 64.29 | 289 | 49ers |
| 160 | 28 | Michael Bennett | 56.04 | 43.83 | 63.14 | 294 | Jaguars |
| 161 | 29 | Carl Davis Jr. | 54.35 | 52.41 | 54.61 | 235 | Ravens |
| 162 | 30 | Khyri Thornton | 51.92 | 53.28 | 60.27 | 102 | Lions |
| 163 | 31 | Damion Square | 49.81 | 52.09 | 52.59 | 155 | Chargers |

## ED — Edge

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 94.41 | 97.44 | 89.69 | 994 | Broncos |
| 2 | 2 | Justin Houston | 90.96 | 89.99 | 89.84 | 669 | Chiefs |
| 3 | 3 | Khalil Mack | 89.59 | 97.24 | 80.32 | 999 | Raiders |
| 4 | 4 | Pernell McPhee | 89.33 | 91.14 | 85.26 | 593 | Bears |
| 5 | 5 | Ezekiel Ansah | 88.98 | 90.10 | 84.49 | 656 | Lions |
| 6 | 6 | Whitney Mercilus | 87.02 | 88.64 | 82.15 | 796 | Texans |
| 7 | 7 | Brandon Graham | 86.03 | 85.63 | 82.13 | 849 | Eagles |
| 8 | 8 | Michael Bennett | 84.29 | 91.51 | 75.31 | 922 | Seahawks |
| 9 | 9 | DeMarcus Ware | 83.49 | 75.47 | 86.34 | 553 | Broncos |
| 10 | 10 | Carlos Dunlap | 81.86 | 81.51 | 77.93 | 945 | Bengals |
| 11 | 11 | Cliff Avril | 81.76 | 77.23 | 80.61 | 861 | Seahawks |
| 12 | 12 | Robert Quinn | 81.05 | 83.20 | 79.61 | 334 | Rams |
| 13 | 13 | Charles Johnson | 80.61 | 74.63 | 82.71 | 528 | Panthers |
| 14 | 14 | Olivier Vernon | 80.35 | 85.93 | 72.46 | 943 | Dolphins |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Jabaal Sheard | 79.97 | 90.49 | 69.94 | 665 | Patriots |
| 16 | 2 | James Harrison | 79.62 | 66.05 | 86.06 | 698 | Steelers |
| 17 | 3 | Cameron Wake | 79.17 | 68.77 | 86.83 | 246 | Dolphins |
| 18 | 4 | Everson Griffen | 78.99 | 78.52 | 75.13 | 913 | Vikings |
| 19 | 5 | Elvis Dumervil | 78.47 | 62.54 | 84.92 | 786 | Ravens |
| 20 | 6 | Mario Williams | 77.81 | 72.22 | 77.89 | 880 | Bills |
| 21 | 7 | Chandler Jones | 77.60 | 78.50 | 73.76 | 943 | Patriots |
| 22 | 8 | Jerry Hughes | 77.56 | 74.48 | 75.45 | 1003 | Bills |
| 23 | 9 | Shaquil Barrett | 77.54 | 69.29 | 78.88 | 552 | Broncos |
| 24 | 10 | William Hayes | 77.38 | 76.75 | 74.05 | 579 | Rams |
| 25 | 11 | Danielle Hunter | 77.34 | 72.27 | 77.58 | 420 | Vikings |
| 26 | 12 | Markus Golden | 77.03 | 68.46 | 78.57 | 633 | Cardinals |
| 27 | 13 | Cameron Jordan | 76.24 | 80.36 | 69.33 | 979 | Saints |
| 28 | 14 | Willie Young | 76.04 | 67.94 | 78.30 | 525 | Bears |
| 29 | 15 | Aaron Lynch | 75.44 | 66.01 | 78.86 | 794 | 49ers |
| 30 | 16 | Connor Barwin | 74.84 | 58.33 | 81.68 | 1043 | Eagles |
| 31 | 17 | Ryan Kerrigan | 74.19 | 61.44 | 78.52 | 959 | Commanders |

### Starter (60 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 32 | 1 | Jeremiah Attaochu | 73.83 | 68.90 | 75.55 | 666 | Chargers |
| 33 | 2 | Greg Hardy | 73.64 | 71.49 | 77.67 | 596 | Cowboys |
| 34 | 3 | Robert Ayers | 73.60 | 74.80 | 71.97 | 570 | Giants |
| 35 | 4 | Marcus Smith | 73.54 | 58.60 | 83.24 | 127 | Eagles |
| 36 | 5 | DeMarcus Lawrence | 73.44 | 75.69 | 70.51 | 698 | Cowboys |
| 37 | 6 | Vinny Curry | 73.09 | 62.44 | 76.23 | 424 | Eagles |
| 38 | 7 | Jason Pierre-Paul | 72.56 | 77.69 | 70.18 | 503 | Giants |
| 39 | 8 | Julius Peppers | 72.54 | 61.56 | 75.70 | 779 | Packers |
| 40 | 9 | Lorenzo Mauldin IV | 72.42 | 59.93 | 78.67 | 253 | Jets |
| 41 | 10 | Paul Kruger | 72.33 | 60.63 | 75.96 | 679 | Browns |
| 42 | 11 | Melvin Ingram III | 72.26 | 67.23 | 75.72 | 961 | Chargers |
| 43 | 12 | Preston Smith | 72.13 | 61.87 | 74.80 | 563 | Commanders |
| 44 | 13 | Nick Perry | 72.02 | 63.63 | 73.44 | 402 | Packers |
| 45 | 14 | Arthur Moats | 71.18 | 59.53 | 74.78 | 594 | Steelers |
| 46 | 15 | Tamba Hali | 70.42 | 63.81 | 70.66 | 905 | Chiefs |
| 47 | 16 | Lamarr Houston | 70.16 | 64.22 | 73.08 | 416 | Bears |
| 48 | 17 | Mario Addison | 69.27 | 59.36 | 71.71 | 447 | Panthers |
| 49 | 18 | Trent Cole | 69.19 | 60.19 | 72.72 | 527 | Colts |
| 50 | 19 | Jared Allen | 68.97 | 54.14 | 75.00 | 730 | Panthers |
| 51 | 20 | Derrick Morgan | 68.91 | 60.09 | 74.53 | 524 | Titans |
| 52 | 21 | Shane Ray | 68.56 | 59.36 | 70.53 | 403 | Broncos |
| 53 | 22 | Vic Beasley Jr. | 68.15 | 67.90 | 64.15 | 534 | Falcons |
| 54 | 23 | Damontre Moore | 68.04 | 60.84 | 70.34 | 277 | Dolphins |
| 55 | 24 | Dwight Freeney | 67.54 | 52.58 | 75.29 | 308 | Cardinals |
| 56 | 25 | Armonty Bryant | 67.54 | 55.50 | 72.43 | 478 | Browns |
| 57 | 26 | Ryan Davis Sr. | 67.42 | 61.44 | 74.23 | 244 | Jaguars |
| 58 | 27 | Frank Clark | 67.40 | 61.38 | 67.24 | 355 | Seahawks |
| 59 | 28 | Quinton Coples | 67.04 | 60.20 | 68.08 | 286 | Dolphins |
| 60 | 29 | Erik Walden | 66.97 | 54.59 | 71.70 | 788 | Colts |
| 61 | 30 | Chris Long | 66.87 | 55.21 | 75.67 | 481 | Rams |
| 62 | 31 | Dee Ford | 66.82 | 58.70 | 70.02 | 552 | Chiefs |
| 63 | 32 | Jayrone Elliott | 66.80 | 59.68 | 73.89 | 174 | Packers |
| 64 | 33 | Devin Taylor | 66.70 | 60.43 | 67.65 | 550 | Lions |
| 65 | 34 | Ahmad Brooks | 66.36 | 50.39 | 75.30 | 740 | 49ers |
| 66 | 35 | John Simon | 66.20 | 61.62 | 70.08 | 683 | Texans |
| 67 | 36 | Aldon Smith | 66.01 | 69.70 | 67.08 | 1034 | Raiders |
| 68 | 37 | Kony Ealy | 66.01 | 59.34 | 66.29 | 740 | Panthers |
| 69 | 38 | Za'Darius Smith | 65.84 | 59.15 | 67.16 | 403 | Ravens |
| 70 | 39 | Barkevious Mingo | 65.65 | 60.98 | 64.98 | 255 | Browns |
| 71 | 40 | Jonathan Newsome | 65.64 | 56.69 | 69.39 | 344 | Colts |
| 72 | 41 | Robert Mathis | 65.54 | 49.58 | 73.05 | 546 | Colts |
| 73 | 42 | Brian Orakpo | 65.41 | 57.48 | 70.04 | 960 | Titans |
| 74 | 43 | Jadeveon Clowney | 64.85 | 77.08 | 59.17 | 559 | Texans |
| 75 | 44 | Ryan Delaire | 64.82 | 57.45 | 71.82 | 245 | Panthers |
| 76 | 45 | Nate Orchard | 64.64 | 58.57 | 65.55 | 471 | Browns |
| 77 | 46 | Alex Okafor | 64.59 | 58.12 | 70.06 | 605 | Cardinals |
| 78 | 47 | O'Brien Schofield | 64.57 | 55.09 | 67.36 | 494 | Falcons |
| 79 | 48 | Rob Ninkovich | 64.28 | 50.31 | 69.43 | 1018 | Patriots |
| 80 | 49 | Chris Smith | 64.19 | 60.55 | 72.48 | 155 | Jaguars |
| 81 | 50 | Trent Murphy | 64.00 | 60.34 | 62.66 | 688 | Commanders |
| 82 | 51 | William Gholston | 63.38 | 62.49 | 60.95 | 670 | Buccaneers |
| 83 | 52 | Sam Acho | 63.16 | 57.98 | 63.09 | 447 | Bears |
| 84 | 53 | Mike Neal | 63.14 | 56.08 | 63.68 | 1642 | Packers |
| 85 | 54 | Randy Gregory | 63.12 | 59.89 | 65.27 | 245 | Cowboys |
| 86 | 55 | Brian Robison | 63.10 | 52.52 | 65.99 | 946 | Vikings |
| 87 | 56 | Eli Harold | 62.92 | 59.42 | 61.08 | 336 | 49ers |
| 88 | 57 | Michael Johnson | 62.86 | 58.65 | 62.13 | 902 | Bengals |
| 89 | 58 | Howard Jones | 62.80 | 57.54 | 66.31 | 387 | Buccaneers |
| 90 | 59 | Lerentee McCray | 62.55 | 60.22 | 64.62 | 118 | Broncos |
| 91 | 60 | Kyle Emanuel | 62.17 | 56.10 | 63.09 | 300 | Chargers |

### Rotation/backup (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 92 | 1 | Adrian Clayborn | 61.60 | 59.87 | 63.27 | 519 | Falcons |
| 93 | 2 | Darryl Tapp | 61.57 | 54.00 | 63.48 | 409 | Lions |
| 94 | 3 | Chris Clemons | 61.36 | 46.20 | 67.30 | 659 | Jaguars |
| 95 | 4 | Kasim Edebali | 61.26 | 55.23 | 61.89 | 358 | Saints |
| 96 | 5 | Jeremy Mincey | 61.23 | 54.15 | 62.82 | 380 | Cowboys |
| 97 | 6 | Frank Zombo | 61.22 | 55.55 | 68.13 | 249 | Chiefs |
| 98 | 7 | Jacquies Smith | 60.77 | 54.71 | 63.65 | 542 | Buccaneers |
| 99 | 8 | Brooks Reed | 60.35 | 55.15 | 61.21 | 345 | Falcons |
| 100 | 9 | David Bass | 60.23 | 57.48 | 61.74 | 530 | Titans |
| 101 | 10 | Kroy Biermann | 60.18 | 55.80 | 61.85 | 516 | Falcons |
| 102 | 11 | Jarvis Jones | 60.16 | 57.14 | 61.14 | 522 | Steelers |
| 103 | 12 | Ethan Westbrooks | 60.06 | 57.03 | 59.87 | 274 | Rams |
| 104 | 13 | Albert McClellan | 59.85 | 53.89 | 61.60 | 158 | Ravens |
| 105 | 14 | Calvin Pace | 59.48 | 47.60 | 63.24 | 519 | Jets |
| 106 | 15 | Wallace Gilberry | 59.46 | 47.63 | 63.18 | 667 | Bengals |
| 107 | 16 | George Selvie | 59.05 | 51.81 | 61.79 | 370 | Giants |
| 108 | 17 | Bud Dupree | 58.63 | 54.68 | 57.09 | 653 | Steelers |
| 109 | 18 | Bjoern Werner | 58.23 | 58.75 | 57.62 | 150 | Colts |
| 110 | 19 | Andre Branch | 57.58 | 56.64 | 57.79 | 596 | Jaguars |
| 111 | 20 | Jason Jones | 57.01 | 54.12 | 58.00 | 543 | Lions |
| 112 | 21 | Kareem Martin | 56.81 | 57.26 | 56.89 | 199 | Cardinals |
| 113 | 22 | Terrence Fede | 55.69 | 58.14 | 54.70 | 210 | Dolphins |
| 114 | 23 | Cassius Marsh | 55.54 | 58.82 | 53.48 | 210 | Seahawks |
| 115 | 24 | Bobby Richardson | 55.19 | 55.22 | 52.04 | 583 | Saints |
| 116 | 25 | George Johnson | 55.18 | 51.26 | 58.31 | 429 | Buccaneers |
| 117 | 26 | Kerry Wynn | 55.10 | 58.22 | 53.81 | 577 | Giants |
| 118 | 27 | Eugene Sims | 54.98 | 51.48 | 54.71 | 581 | Rams |
| 119 | 28 | Tavaris Barnes | 54.92 | 58.32 | 52.66 | 130 | Saints |
| 120 | 29 | Chris Carter | 54.90 | 55.54 | 54.48 | 116 | Ravens |
| 121 | 30 | Benson Mayowa | 54.83 | 59.25 | 53.03 | 376 | Raiders |
| 122 | 31 | Matt Longacre | 54.46 | 63.55 | 60.20 | 138 | Rams |
| 123 | 32 | IK Enemkpali | 53.70 | 56.35 | 51.02 | 146 | Bills |
| 124 | 33 | Will Clarke | 52.88 | 58.05 | 50.74 | 140 | Bengals |
| 125 | 34 | Malliciah Goodman | 51.68 | 58.11 | 49.90 | 106 | Falcons |
| 126 | 35 | Scott Crichton | 51.52 | 58.72 | 51.27 | 128 | Vikings |
| 127 | 36 | Lamarr Woodley | 50.33 | 51.89 | 52.94 | 277 | Cardinals |
| 128 | 37 | Corey Lemonier | 49.86 | 55.79 | 48.77 | 271 | 49ers |

## G — Guard

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Marshal Yanda | 96.91 | 91.40 | 96.42 | 1130 | Ravens |
| 2 | 2 | Evan Mathis | 94.54 | 90.10 | 93.34 | 988 | Broncos |
| 3 | 3 | T.J. Lang | 93.83 | 89.60 | 92.48 | 1173 | Packers |
| 4 | 4 | Zack Martin | 90.86 | 86.10 | 89.86 | 1020 | Cowboys |
| 5 | 5 | Kelechi Osemele | 90.46 | 81.13 | 92.51 | 974 | Ravens |
| 6 | 6 | David DeCastro | 88.67 | 83.00 | 88.28 | 1211 | Steelers |
| 7 | 7 | Trai Turner | 87.13 | 80.40 | 87.45 | 1277 | Panthers |
| 8 | 8 | Mike Iupati | 86.44 | 79.91 | 86.62 | 938 | Cardinals |
| 9 | 9 | Andrew Norwell | 86.31 | 81.10 | 85.61 | 999 | Panthers |
| 10 | 10 | Kevin Zeitler | 85.92 | 78.90 | 86.44 | 1112 | Bengals |
| 11 | 11 | Gabe Jackson | 85.86 | 79.90 | 85.67 | 1050 | Raiders |
| 12 | 12 | Justin Pugh | 85.71 | 78.59 | 86.29 | 963 | Giants |
| 13 | 13 | Logan Mankins | 85.30 | 77.30 | 86.47 | 1018 | Buccaneers |
| 14 | 14 | Josh Sitton | 85.01 | 78.10 | 85.45 | 1280 | Packers |
| 15 | 15 | Jeff Allen | 84.67 | 75.10 | 86.88 | 579 | Chiefs |
| 16 | 16 | Richie Incognito | 84.41 | 77.50 | 84.85 | 1075 | Bills |
| 17 | 17 | Ramon Foster | 84.37 | 76.70 | 85.32 | 1211 | Steelers |
| 18 | 18 | Brandon Brooks | 84.24 | 78.10 | 84.16 | 1042 | Texans |
| 19 | 19 | Tim Lelito | 84.14 | 75.21 | 85.93 | 948 | Saints |
| 20 | 20 | Jack Mewhort | 83.72 | 76.60 | 84.30 | 1098 | Colts |
| 21 | 21 | Brandon Scherff | 83.49 | 76.40 | 84.05 | 1135 | Commanders |
| 22 | 22 | Chris Chester | 83.47 | 76.50 | 83.95 | 1139 | Falcons |
| 23 | 23 | Garrett Reynolds | 82.98 | 75.32 | 83.92 | 732 | Rams |
| 24 | 24 | Matt Slauson | 82.82 | 74.10 | 84.47 | 1075 | Bears |
| 25 | 25 | James Carpenter | 82.46 | 75.40 | 83.00 | 1104 | Jets |
| 26 | 26 | Andy Levitre | 82.36 | 74.00 | 83.77 | 1126 | Falcons |
| 27 | 27 | Andrew Tiller | 82.15 | 71.92 | 84.81 | 614 | 49ers |
| 28 | 28 | Chance Warmack | 82.05 | 74.07 | 83.20 | 829 | Titans |
| 29 | 29 | John Greco | 81.98 | 73.88 | 83.22 | 890 | Browns |
| 30 | 30 | Manuel Ramirez | 81.94 | 71.12 | 84.98 | 488 | Lions |
| 31 | 31 | Geoff Schwartz | 81.11 | 73.05 | 82.31 | 668 | Giants |
| 32 | 32 | D.J. Fluker | 80.76 | 70.76 | 83.26 | 863 | Chargers |
| 33 | 33 | Andrew Gardner | 80.75 | 69.71 | 83.95 | 174 | Eagles |
| 34 | 34 | Clint Boling | 80.55 | 73.00 | 81.41 | 1121 | Bengals |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Ron Leary | 79.53 | 68.73 | 82.56 | 218 | Cowboys |
| 36 | 2 | Larry Warford | 79.32 | 71.05 | 80.66 | 812 | Lions |
| 37 | 3 | Jahri Evans | 78.98 | 69.41 | 81.19 | 800 | Saints |
| 38 | 4 | Lane Taylor | 78.74 | 67.56 | 82.02 | 153 | Packers |
| 39 | 5 | Joel Bitonio | 78.63 | 68.22 | 81.41 | 615 | Browns |
| 40 | 6 | Patrick Omameh | 78.37 | 68.39 | 80.85 | 667 | Bears |
| 41 | 7 | Hugh Thornton | 77.80 | 67.75 | 80.34 | 799 | Colts |
| 42 | 8 | Brian Winters | 77.74 | 67.76 | 80.22 | 766 | Jets |
| 43 | 9 | A.J. Cann | 77.72 | 68.88 | 79.45 | 861 | Jaguars |
| 44 | 10 | J.R. Sweezy | 77.57 | 67.50 | 80.12 | 1127 | Seahawks |
| 45 | 11 | Zach Fulton | 77.44 | 67.11 | 80.16 | 551 | Chiefs |
| 46 | 12 | Vladimir Ducasse | 77.38 | 66.34 | 80.58 | 743 | Bears |
| 47 | 13 | Laken Tomlinson | 77.27 | 69.20 | 78.48 | 986 | Lions |
| 48 | 14 | Ben Grubbs | 76.44 | 66.07 | 79.18 | 459 | Chiefs |
| 49 | 15 | Amini Silatolu | 76.31 | 64.92 | 79.73 | 238 | Panthers |
| 50 | 16 | Shaq Mason | 76.26 | 64.32 | 80.06 | 868 | Patriots |
| 51 | 17 | Josh Kline | 76.07 | 65.90 | 78.69 | 996 | Patriots |
| 52 | 18 | Max Garcia | 76.02 | 64.51 | 79.53 | 575 | Broncos |
| 53 | 19 | Brandon Fusco | 75.87 | 66.10 | 78.22 | 1078 | Vikings |
| 54 | 20 | Senio Kelemete | 75.87 | 64.40 | 79.35 | 424 | Saints |
| 55 | 21 | Louis Vasquez | 75.48 | 67.20 | 76.84 | 1048 | Broncos |
| 56 | 22 | Matt Tobin | 75.45 | 65.00 | 78.25 | 984 | Eagles |
| 57 | 23 | Allen Barbre | 75.24 | 65.10 | 77.84 | 1156 | Eagles |
| 58 | 24 | John Jerry | 74.97 | 65.50 | 77.11 | 644 | Giants |
| 59 | 25 | Laurent Duvernay-Tardif | 74.15 | 66.66 | 74.97 | 820 | Chiefs |

### Starter (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 60 | 1 | Xavier Su'a-Filo | 73.43 | 63.18 | 76.09 | 691 | Texans |
| 61 | 2 | Shawn Lauvao | 73.35 | 60.36 | 77.85 | 150 | Commanders |
| 62 | 3 | Tre Jackson | 73.27 | 64.16 | 75.17 | 607 | Patriots |
| 63 | 4 | Kraig Urbik | 73.22 | 62.39 | 76.27 | 431 | Bills |
| 64 | 5 | Jonathan Cooper | 73.20 | 62.97 | 75.86 | 635 | Cardinals |
| 65 | 6 | Alex Boone | 72.75 | 63.43 | 74.80 | 761 | 49ers |
| 66 | 7 | Jon Feliciano | 72.70 | 60.17 | 76.89 | 187 | Raiders |
| 67 | 8 | Cody Wichmann | 71.46 | 63.36 | 72.69 | 427 | Rams |
| 68 | 9 | Kenny Wiggins | 71.35 | 59.55 | 75.05 | 791 | Chargers |
| 69 | 10 | Willie Colon | 71.29 | 60.53 | 74.30 | 341 | Jets |
| 70 | 11 | Chris Scott | 70.97 | 59.64 | 74.35 | 107 | Panthers |
| 71 | 12 | Quinton Spain | 70.08 | 61.18 | 71.85 | 383 | Titans |
| 72 | 13 | Zane Beadles | 69.38 | 58.50 | 72.46 | 1058 | Jaguars |
| 73 | 14 | Rodger Saffold | 69.33 | 55.11 | 74.64 | 231 | Rams |
| 74 | 15 | Jordan Devey | 68.85 | 58.13 | 71.83 | 384 | 49ers |
| 75 | 16 | Oday Aboushi | 67.79 | 55.93 | 71.53 | 398 | Texans |
| 76 | 17 | Jeff Adams | 67.25 | 59.12 | 68.50 | 116 | Texans |
| 77 | 18 | Billy Turner | 67.23 | 55.77 | 70.70 | 765 | Dolphins |
| 78 | 19 | Lance Louis | 66.95 | 55.68 | 70.30 | 237 | Colts |
| 79 | 20 | Ted Larsen | 66.65 | 53.95 | 70.95 | 802 | Cardinals |
| 80 | 21 | Orlando Franklin | 64.22 | 52.30 | 68.00 | 616 | Chargers |
| 81 | 22 | John Miller | 64.16 | 50.27 | 69.25 | 648 | Bills |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 82 | 1 | Dallas Thomas | 61.26 | 45.50 | 67.60 | 1030 | Dolphins |
| 83 | 2 | Cameron Erving | 60.81 | 47.01 | 65.84 | 424 | Browns |

## HB — Running Back

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (0 players)

_None._

### Good (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Marshawn Lynch | 77.49 | 79.80 | 71.78 | 157 | Seahawks |
| 2 | 2 | Dion Lewis | 76.60 | 68.01 | 78.16 | 191 | Patriots |
| 3 | 3 | Le'Veon Bell | 76.51 | 82.24 | 68.53 | 138 | Steelers |
| 4 | 4 | Doug Martin | 75.38 | 80.28 | 67.95 | 209 | Buccaneers |

### Starter (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Charles Sims | 73.01 | 80.52 | 63.84 | 246 | Buccaneers |
| 6 | 2 | Todd Gurley II | 72.71 | 68.00 | 71.69 | 146 | Rams |
| 7 | 3 | David Johnson | 72.37 | 70.01 | 69.77 | 288 | Cardinals |
| 8 | 4 | Chris Thompson | 72.29 | 63.15 | 74.22 | 204 | Commanders |
| 9 | 5 | Chris Ivory | 72.25 | 67.32 | 71.37 | 188 | Jets |
| 10 | 6 | Adrian Peterson | 72.20 | 64.95 | 72.87 | 231 | Vikings |
| 11 | 7 | Jamaal Charles | 72.18 | 67.74 | 70.98 | 142 | Chiefs |
| 12 | 8 | Jonathan Stewart | 71.95 | 70.17 | 68.97 | 225 | Panthers |
| 13 | 9 | DeAngelo Williams | 71.41 | 74.30 | 65.32 | 375 | Steelers |
| 14 | 10 | Darren Sproles | 71.38 | 64.82 | 71.59 | 256 | Eagles |
| 15 | 11 | LeSean McCoy | 71.15 | 69.89 | 67.82 | 254 | Bills |
| 16 | 12 | Theo Riddick | 71.08 | 80.26 | 60.80 | 300 | Lions |
| 17 | 13 | Lamar Miller | 70.77 | 72.90 | 65.19 | 310 | Dolphins |
| 18 | 14 | Carlos Hyde | 70.67 | 66.35 | 69.39 | 103 | 49ers |
| 19 | 15 | Duke Johnson Jr. | 70.65 | 70.89 | 66.32 | 341 | Browns |
| 20 | 16 | C.J. Spiller | 70.54 | 65.12 | 69.98 | 140 | Saints |
| 21 | 17 | Matt Forte | 70.52 | 69.87 | 66.78 | 288 | Bears |
| 22 | 18 | Eddie Lacy | 70.33 | 67.47 | 68.07 | 179 | Packers |
| 23 | 19 | C.J. Anderson | 70.09 | 63.30 | 70.45 | 282 | Broncos |
| 24 | 20 | Giovani Bernard | 70.06 | 68.33 | 67.04 | 280 | Bengals |
| 25 | 21 | Rashad Jennings | 68.84 | 68.76 | 64.72 | 166 | Giants |
| 26 | 22 | T.J. Yeldon | 68.79 | 68.17 | 65.04 | 287 | Jaguars |
| 27 | 23 | Justin Forsett | 68.63 | 63.68 | 67.77 | 219 | Ravens |
| 28 | 24 | Mark Ingram II | 68.48 | 65.70 | 66.16 | 267 | Saints |
| 29 | 25 | Arian Foster | 67.96 | 64.24 | 66.28 | 107 | Texans |
| 30 | 26 | Fred Jackson | 67.87 | 58.49 | 69.96 | 201 | Seahawks |
| 31 | 27 | Ryan Mathews | 67.71 | 62.16 | 67.25 | 108 | Eagles |
| 32 | 28 | Danny Woodhead | 67.13 | 72.60 | 59.31 | 396 | Chargers |
| 33 | 29 | Devonta Freeman | 67.05 | 67.89 | 62.33 | 361 | Falcons |
| 34 | 30 | Andre Ellington | 66.65 | 64.62 | 63.83 | 126 | Cardinals |
| 35 | 31 | DeMarco Murray | 66.53 | 56.66 | 68.94 | 242 | Eagles |
| 36 | 32 | James Starks | 66.01 | 60.35 | 65.61 | 284 | Packers |
| 37 | 33 | Shane Vereen | 65.88 | 64.59 | 62.58 | 303 | Giants |
| 38 | 34 | Chris Polk | 65.71 | 56.18 | 67.89 | 127 | Texans |
| 39 | 35 | Latavius Murray | 65.54 | 63.82 | 62.52 | 274 | Raiders |
| 40 | 36 | Antonio Andrews | 65.37 | 65.62 | 61.04 | 174 | Titans |
| 41 | 37 | Alfred Morris | 64.96 | 58.51 | 65.10 | 125 | Commanders |
| 42 | 38 | Frank Gore | 64.93 | 61.48 | 63.06 | 291 | Colts |
| 43 | 39 | Darren McFadden | 64.69 | 60.08 | 63.60 | 263 | Cowboys |
| 44 | 40 | Bilal Powell | 64.57 | 65.75 | 59.62 | 236 | Jets |
| 45 | 41 | Benny Cunningham | 64.42 | 60.64 | 62.78 | 156 | Rams |
| 46 | 42 | Dexter McCluster | 64.03 | 61.13 | 61.80 | 189 | Titans |
| 47 | 43 | Jeremy Hill | 63.83 | 58.82 | 63.01 | 164 | Bengals |
| 48 | 44 | Damien Williams | 63.59 | 62.66 | 60.05 | 117 | Dolphins |
| 49 | 45 | Ameer Abdullah | 63.40 | 62.81 | 59.63 | 170 | Lions |
| 50 | 46 | Brandon Bolden | 63.17 | 59.63 | 61.37 | 109 | Patriots |
| 51 | 47 | Isaiah Crowell | 62.85 | 63.79 | 58.05 | 194 | Browns |
| 52 | 48 | Fozzy Whittaker | 62.51 | 63.87 | 57.44 | 101 | Panthers |
| 53 | 49 | Jonathan Grimes | 62.31 | 62.03 | 58.33 | 174 | Texans |
| 54 | 50 | Matt Asiata | 62.05 | 62.41 | 57.65 | 100 | Vikings |

### Rotation/backup (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 55 | 1 | Javorius Allen | 61.76 | 58.55 | 59.74 | 199 | Ravens |
| 56 | 2 | Melvin Gordon III | 61.72 | 56.53 | 61.01 | 157 | Chargers |
| 57 | 3 | Denard Robinson | 61.47 | 57.70 | 59.81 | 132 | Jaguars |
| 58 | 4 | James White | 61.29 | 65.20 | 54.51 | 276 | Patriots |
| 59 | 5 | Charcandrick West | 61.14 | 58.37 | 58.82 | 278 | Chiefs |
| 60 | 6 | Alfred Blue | 60.83 | 60.53 | 56.86 | 133 | Texans |
| 61 | 7 | Chris Johnson | 60.50 | 50.80 | 62.80 | 115 | Cardinals |
| 62 | 8 | Ronnie Hillman | 60.26 | 57.89 | 57.68 | 251 | Broncos |
| 63 | 9 | Jeremy Langford | 60.12 | 60.73 | 55.54 | 166 | Bears |
| 64 | 10 | Shaun Draughn | 59.93 | 61.05 | 55.02 | 146 | 49ers |
| 65 | 11 | Matt Jones | 58.28 | 52.40 | 58.04 | 128 | Commanders |

## LB — Linebacker

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Luke Kuechly | 90.96 | 93.60 | 85.03 | 946 | Panthers |
| 2 | 2 | K.J. Wright | 85.74 | 87.90 | 80.34 | 1089 | Seahawks |
| 3 | 3 | Wesley Woodyard | 84.42 | 86.26 | 79.02 | 502 | Titans |
| 4 | 4 | Anthony Barr | 83.75 | 90.00 | 77.63 | 882 | Vikings |
| 5 | 5 | Jerrell Freeman | 83.64 | 88.08 | 78.40 | 748 | Colts |
| 6 | 6 | Sean Lee | 81.94 | 88.34 | 78.19 | 814 | Cowboys |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Dont'a Hightower | 79.68 | 85.42 | 73.03 | 722 | Patriots |
| 8 | 2 | Derrick Johnson | 79.64 | 85.10 | 76.51 | 1178 | Chiefs |
| 9 | 3 | Thomas Davis Sr. | 78.75 | 80.00 | 73.75 | 1154 | Panthers |
| 10 | 4 | Jamie Collins Sr. | 77.80 | 84.00 | 70.54 | 869 | Patriots |
| 11 | 5 | Danny Trevathan | 77.57 | 80.40 | 75.59 | 883 | Broncos |
| 12 | 6 | Karlos Dansby | 76.86 | 76.70 | 74.05 | 1028 | Browns |
| 13 | 7 | Brandon Marshall | 75.95 | 77.00 | 74.51 | 1082 | Broncos |
| 14 | 8 | Benardrick McKinney | 74.88 | 73.18 | 73.93 | 465 | Texans |
| 15 | 9 | Shaq Thompson | 74.67 | 72.32 | 72.07 | 441 | Panthers |
| 16 | 10 | Vontaze Burfict | 74.24 | 74.72 | 75.80 | 540 | Bengals |

### Starter (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Zach Brown | 73.74 | 73.90 | 74.15 | 489 | Titans |
| 18 | 2 | Denzel Perryman | 73.69 | 70.99 | 74.46 | 387 | Chargers |
| 19 | 3 | A.J. Klein | 73.09 | 71.45 | 72.51 | 328 | Panthers |
| 20 | 4 | Rolando McClain | 73.03 | 73.85 | 72.28 | 639 | Cowboys |
| 21 | 5 | Vince Williams | 72.39 | 69.46 | 71.01 | 195 | Steelers |
| 22 | 6 | Tahir Whitehead | 71.20 | 69.04 | 68.48 | 585 | Lions |
| 23 | 7 | David Harris | 71.01 | 69.70 | 67.72 | 970 | Jets |
| 24 | 8 | Bobby Wagner | 70.89 | 69.20 | 68.49 | 1027 | Seahawks |
| 25 | 9 | Bruce Irvin | 70.77 | 67.00 | 69.11 | 798 | Seahawks |
| 26 | 10 | Jasper Brinkley | 70.68 | 73.45 | 69.45 | 420 | Giants |
| 27 | 11 | Erin Henderson | 70.12 | 66.79 | 71.09 | 226 | Jets |
| 28 | 12 | Christian Kirksey | 70.07 | 68.09 | 67.23 | 570 | Browns |
| 29 | 13 | C.J. Mosley | 69.46 | 66.80 | 67.06 | 1043 | Ravens |
| 30 | 14 | Josh Mauga | 69.32 | 69.55 | 67.28 | 519 | Chiefs |
| 31 | 15 | Telvin Smith Sr. | 69.28 | 68.30 | 67.06 | 994 | Jaguars |
| 32 | 16 | Josh Bynes | 69.11 | 68.37 | 66.91 | 815 | Lions |
| 33 | 17 | Rey Maualuga | 68.98 | 68.25 | 66.65 | 660 | Bengals |
| 34 | 18 | Jordan Hicks | 68.88 | 75.41 | 71.22 | 446 | Eagles |
| 35 | 19 | Manny Lawson | 68.24 | 64.62 | 66.48 | 701 | Bills |
| 36 | 20 | NaVorro Bowman | 67.74 | 63.70 | 66.27 | 1097 | 49ers |
| 37 | 21 | Koa Misi | 67.71 | 68.40 | 66.41 | 728 | Dolphins |
| 38 | 22 | Christian Jones | 67.64 | 65.42 | 67.56 | 741 | Bears |
| 39 | 23 | Nate Stupar | 67.37 | 68.96 | 68.39 | 257 | Falcons |
| 40 | 24 | Lavonte David | 66.99 | 63.80 | 65.59 | 1083 | Buccaneers |
| 41 | 25 | Max Bullough | 66.47 | 63.40 | 68.90 | 119 | Texans |
| 42 | 26 | Craig Robertson | 66.36 | 61.78 | 67.75 | 383 | Browns |
| 43 | 27 | Nate Irving | 66.14 | 66.92 | 69.15 | 106 | Colts |
| 44 | 28 | Daryl Smith | 65.94 | 63.20 | 63.60 | 971 | Ravens |
| 45 | 29 | Avery Williamson | 65.91 | 62.20 | 65.25 | 922 | Titans |
| 46 | 30 | Sean Spence | 65.90 | 63.28 | 65.43 | 271 | Steelers |
| 47 | 31 | Zachary Orr | 65.58 | 65.20 | 69.08 | 142 | Ravens |
| 48 | 32 | Todd Davis | 64.95 | 62.06 | 65.84 | 134 | Broncos |
| 49 | 33 | Joe Thomas | 64.49 | 61.68 | 62.20 | 316 | Packers |
| 50 | 34 | A.J. Hawk | 64.49 | 60.37 | 63.07 | 304 | Bengals |
| 51 | 35 | Eric Kendricks | 64.47 | 59.20 | 64.85 | 816 | Vikings |
| 52 | 36 | Clay Matthews | 64.15 | 62.60 | 61.01 | 1144 | Packers |
| 53 | 37 | Jelani Jenkins | 64.09 | 60.37 | 65.11 | 696 | Dolphins |
| 54 | 38 | J.T. Thomas | 63.93 | 60.28 | 66.26 | 400 | Giants |
| 55 | 39 | Stephen Tulloch | 63.86 | 60.67 | 65.89 | 722 | Lions |
| 56 | 40 | Vincent Rey | 63.81 | 56.26 | 64.67 | 753 | Bengals |
| 57 | 41 | D'Qwell Jackson | 63.74 | 57.30 | 63.86 | 1087 | Colts |
| 58 | 42 | Paul Posluszny | 63.47 | 60.90 | 65.08 | 977 | Jaguars |
| 59 | 43 | Ryan Shazier | 63.45 | 61.79 | 64.43 | 808 | Steelers |
| 60 | 44 | Michael Mauti | 63.39 | 64.33 | 67.87 | 174 | Saints |
| 61 | 45 | Danny Lansanah | 63.10 | 62.55 | 62.74 | 369 | Buccaneers |
| 62 | 46 | Akeem Dent | 63.06 | 61.84 | 68.77 | 118 | Texans |
| 63 | 47 | Zach Vigil | 62.54 | 60.70 | 63.76 | 141 | Dolphins |
| 64 | 48 | Stephone Anthony | 62.48 | 55.30 | 63.10 | 987 | Saints |
| 65 | 49 | Neville Hewitt | 62.46 | 61.23 | 64.31 | 342 | Dolphins |
| 66 | 50 | Philip Wheeler | 62.39 | 59.87 | 63.87 | 143 | Falcons |
| 67 | 51 | Chad Greenway | 62.33 | 56.16 | 63.52 | 655 | Vikings |

### Rotation/backup (61 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 68 | 1 | Ben Heeney | 61.99 | 61.65 | 62.22 | 307 | Raiders |
| 69 | 2 | Jonathan Freeny | 61.94 | 60.07 | 65.07 | 444 | Patriots |
| 70 | 3 | Dan Skuta | 61.93 | 59.57 | 60.90 | 415 | Jaguars |
| 71 | 4 | Devon Kennard | 61.90 | 61.84 | 64.68 | 483 | Giants |
| 72 | 5 | Mason Foster | 61.66 | 59.68 | 64.01 | 334 | Commanders |
| 73 | 6 | Jerod Mayo | 61.40 | 59.67 | 63.58 | 401 | Patriots |
| 74 | 7 | Mark Herzlich | 61.34 | 59.32 | 61.53 | 130 | Giants |
| 75 | 8 | Brian Cushing | 61.28 | 55.20 | 63.67 | 1039 | Texans |
| 76 | 9 | Joplo Bartu | 60.82 | 58.76 | 63.96 | 140 | Jaguars |
| 77 | 10 | Sean Weatherspoon | 60.63 | 57.02 | 61.89 | 153 | Cardinals |
| 78 | 11 | Kyle Wilber | 60.41 | 58.11 | 61.84 | 225 | Cowboys |
| 79 | 12 | Shea McClellin | 60.39 | 58.01 | 61.98 | 673 | Bears |
| 80 | 13 | Travis Lewis | 60.21 | 57.83 | 61.79 | 138 | Lions |
| 81 | 14 | Andrew Gachkar | 60.13 | 59.88 | 63.12 | 115 | Cowboys |
| 82 | 15 | Josh McNary | 59.98 | 61.12 | 63.48 | 117 | Colts |
| 83 | 16 | Bruce Carter | 59.92 | 53.79 | 61.40 | 307 | Buccaneers |
| 84 | 17 | Mychal Kendricks | 59.55 | 52.15 | 63.14 | 623 | Eagles |
| 85 | 18 | Demario Davis | 59.48 | 52.50 | 59.96 | 847 | Jets |
| 86 | 19 | Justin Durant | 59.41 | 58.22 | 61.97 | 652 | Falcons |
| 87 | 20 | Alec Ogletree | 59.08 | 56.57 | 62.83 | 260 | Rams |
| 88 | 21 | Malcolm Smith | 58.94 | 53.20 | 60.48 | 1139 | Raiders |
| 89 | 22 | Hau'oli Kikaha | 58.92 | 52.97 | 59.75 | 617 | Saints |
| 90 | 23 | Akeem Ayers | 58.55 | 51.66 | 58.97 | 527 | Rams |
| 91 | 24 | Gerald Hodges | 58.54 | 53.57 | 62.78 | 517 | 49ers |
| 92 | 25 | Ramon Humber | 58.47 | 49.46 | 62.60 | 272 | Saints |
| 93 | 26 | Anthony Hitchens | 58.45 | 51.41 | 58.98 | 538 | Cowboys |
| 94 | 27 | Jake Ryan | 58.39 | 53.68 | 62.56 | 328 | Packers |
| 95 | 28 | Hayes Pullard | 58.22 | 64.55 | 68.34 | 151 | Jaguars |
| 96 | 29 | DeMeco Ryans | 58.12 | 54.16 | 60.15 | 599 | Eagles |
| 97 | 30 | Michael Wilhoite | 57.98 | 52.90 | 61.36 | 615 | 49ers |
| 98 | 31 | Kelvin Sheppard | 57.85 | 52.67 | 60.27 | 706 | Dolphins |
| 99 | 32 | Jonathan Anderson | 57.49 | 55.73 | 59.69 | 314 | Bears |
| 100 | 33 | Justin Tuggle | 57.17 | 56.68 | 61.25 | 105 | Texans |
| 101 | 34 | Kevin Minter | 56.54 | 50.30 | 59.66 | 1033 | Cardinals |
| 102 | 35 | Curtis Lofton | 56.01 | 45.19 | 59.06 | 574 | Raiders |
| 103 | 36 | Kevin Pierre-Louis | 55.99 | 58.40 | 59.72 | 104 | Seahawks |
| 104 | 37 | Ray-Ray Armstrong | 55.78 | 56.98 | 59.15 | 208 | 49ers |
| 105 | 38 | Nigel Bradham | 55.60 | 50.32 | 58.38 | 724 | Bills |
| 106 | 39 | Kavell Conner | 55.35 | 56.07 | 59.04 | 184 | Chargers |
| 107 | 40 | James Laurinaitis | 55.31 | 43.70 | 58.88 | 1152 | Rams |
| 108 | 41 | Jonathan Casillas | 55.15 | 49.41 | 57.94 | 672 | Giants |
| 109 | 42 | Ramik Wilson | 55.09 | 61.64 | 66.56 | 129 | Chiefs |
| 110 | 43 | Thurston Armbrister | 55.09 | 52.63 | 58.81 | 206 | Jaguars |
| 111 | 44 | Paul Worrilow | 54.97 | 46.40 | 57.45 | 869 | Falcons |
| 112 | 45 | Emmanuel Lamur | 54.89 | 51.15 | 56.04 | 337 | Bengals |
| 113 | 46 | Dannell Ellerbe | 54.07 | 53.26 | 60.54 | 250 | Saints |
| 114 | 47 | Will Compton | 53.71 | 43.85 | 57.36 | 786 | Commanders |
| 115 | 48 | Donald Butler | 52.87 | 42.20 | 58.21 | 500 | Chargers |
| 116 | 49 | Preston Brown | 52.77 | 40.00 | 57.12 | 1064 | Bills |
| 117 | 50 | Perry Riley | 52.76 | 45.23 | 57.88 | 461 | Commanders |
| 118 | 51 | David Hawthorne | 52.71 | 43.62 | 59.51 | 222 | Saints |
| 119 | 52 | Nate Palmer | 52.64 | 42.41 | 57.25 | 534 | Packers |
| 120 | 53 | Keenan Robinson | 52.63 | 43.00 | 57.90 | 548 | Commanders |
| 121 | 54 | Manti Te'o | 52.52 | 45.03 | 57.52 | 709 | Chargers |
| 122 | 55 | Uani' Unga | 52.49 | 46.44 | 56.52 | 431 | Giants |
| 123 | 56 | James Anderson | 52.33 | 48.20 | 57.68 | 108 | Saints |
| 124 | 57 | Lawrence Timmons | 52.15 | 40.30 | 55.88 | 1179 | Steelers |
| 125 | 58 | Kiko Alonso | 51.85 | 43.00 | 56.84 | 470 | Eagles |
| 126 | 59 | Jon Beason | 50.69 | 49.70 | 57.09 | 159 | Giants |
| 127 | 60 | Kwon Alexander | 49.82 | 41.53 | 55.34 | 809 | Buccaneers |
| 128 | 61 | John Timu | 48.71 | 53.70 | 61.21 | 159 | Bears |

## QB — Quarterback

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Ben Roethlisberger | 84.57 | 88.15 | 77.89 | 623 | Steelers |
| 2 | 2 | Drew Brees | 83.65 | 85.95 | 77.22 | 706 | Saints |
| 3 | 3 | Tom Brady | 82.48 | 87.96 | 73.45 | 824 | Patriots |
| 4 | 4 | Russell Wilson | 81.14 | 78.64 | 78.71 | 699 | Seahawks |
| 5 | 5 | Carson Palmer | 80.87 | 81.74 | 78.37 | 706 | Cardinals |

### Good (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Cam Newton | 79.59 | 80.96 | 74.11 | 692 | Panthers |
| 7 | 2 | Andy Dalton | 77.46 | 75.42 | 78.33 | 463 | Bengals |
| 8 | 3 | Matt Ryan | 76.97 | 80.55 | 69.50 | 703 | Falcons |
| 9 | 4 | Aaron Rodgers | 75.76 | 77.09 | 70.88 | 832 | Packers |
| 10 | 5 | Philip Rivers | 75.25 | 74.04 | 71.70 | 766 | Chargers |

### Starter (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Kirk Cousins | 72.73 | 69.31 | 75.28 | 668 | Commanders |
| 12 | 2 | Jay Cutler | 72.17 | 71.45 | 70.45 | 556 | Bears |
| 13 | 3 | Matthew Stafford | 72.03 | 67.43 | 71.24 | 694 | Lions |
| 14 | 4 | Derek Carr | 71.57 | 70.78 | 67.67 | 652 | Raiders |
| 15 | 5 | Alex Smith | 71.54 | 69.80 | 68.88 | 697 | Chiefs |
| 16 | 6 | Sam Bradford | 71.53 | 75.20 | 67.14 | 611 | Eagles |
| 17 | 7 | Ryan Tannehill | 71.43 | 70.50 | 67.65 | 673 | Dolphins |
| 18 | 8 | Eli Manning | 70.58 | 66.79 | 69.18 | 699 | Giants |
| 19 | 9 | Teddy Bridgewater | 69.51 | 68.75 | 68.02 | 579 | Vikings |
| 20 | 10 | Joe Flacco | 67.63 | 67.73 | 66.45 | 458 | Ravens |
| 21 | 11 | Peyton Manning | 66.88 | 69.39 | 63.37 | 473 | Broncos |
| 22 | 12 | Jameis Winston | 66.83 | 67.20 | 68.61 | 650 | Buccaneers |
| 23 | 13 | Ryan Fitzpatrick | 66.77 | 63.92 | 66.99 | 638 | Jets |
| 24 | 14 | Tyrod Taylor | 66.54 | 72.50 | 76.39 | 507 | Bills |
| 25 | 15 | Tony Romo | 66.18 | 69.46 | 68.90 | 137 | Cowboys |
| 26 | 16 | Blake Bortles | 65.15 | 58.66 | 66.41 | 724 | Jaguars |
| 27 | 17 | Josh McCown | 62.95 | 59.96 | 69.09 | 356 | Browns |
| 28 | 18 | Andrew Luck | 62.77 | 63.22 | 63.21 | 353 | Colts |
| 29 | 19 | Marcus Mariota | 62.36 | 60.96 | 69.28 | 461 | Titans |

### Rotation/backup (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Brian Hoyer | 61.42 | 59.82 | 64.08 | 457 | Texans |
| 31 | 2 | Brandon Weeden | 61.09 | 62.61 | 70.48 | 173 | Texans |
| 32 | 3 | Brock Osweiler | 60.37 | 64.03 | 65.41 | 326 | Broncos |
| 33 | 4 | Colin Kaepernick | 60.28 | 58.78 | 63.31 | 306 | 49ers |
| 34 | 5 | A.J. McCarron | 60.16 | 63.86 | 65.56 | 197 | Bengals |
| 35 | 6 | Blaine Gabbert | 60.06 | 60.59 | 66.36 | 345 | 49ers |
| 36 | 7 | Case Keenum | 59.98 | 62.96 | 66.62 | 142 | Rams |
| 37 | 8 | Matt Hasselbeck | 59.77 | 65.19 | 62.63 | 297 | Colts |
| 38 | 9 | Mark Sanchez | 59.76 | 62.67 | 65.91 | 104 | Eagles |
| 39 | 10 | Nick Foles | 58.64 | 62.13 | 58.82 | 379 | Rams |
| 40 | 11 | Kellen Moore | 58.49 | 62.06 | 61.29 | 113 | Cowboys |
| 41 | 12 | Ryan Mallett | 56.71 | 58.35 | 56.12 | 265 | Ravens |
| 42 | 13 | Johnny Manziel | 56.52 | 51.59 | 60.30 | 281 | Browns |
| 43 | 14 | Austin Davis | 56.37 | 57.20 | 57.46 | 119 | Browns |
| 44 | 15 | Zach Mettenberger | 55.45 | 51.71 | 56.47 | 188 | Titans |
| 45 | 16 | Jimmy Clausen | 55.41 | 55.59 | 54.16 | 145 | Ravens |
| 46 | 17 | Matt Cassel | 54.79 | 55.29 | 57.18 | 248 | Cowboys |

## S — Safety

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Earl Thomas III | 90.56 | 90.90 | 86.17 | 1098 | Seahawks |
| 2 | 2 | Charles Woodson | 90.21 | 92.10 | 84.78 | 1109 | Raiders |
| 3 | 3 | Kurt Coleman | 87.03 | 88.80 | 84.08 | 1193 | Panthers |
| 4 | 4 | Eric Berry | 82.40 | 82.30 | 81.44 | 1148 | Chiefs |
| 5 | 5 | Duron Harmon | 81.98 | 82.60 | 77.60 | 669 | Patriots |
| 6 | 6 | Devin McCourty | 81.77 | 78.00 | 80.12 | 1079 | Patriots |
| 7 | 7 | Patrick Chung | 81.52 | 79.30 | 79.46 | 1040 | Patriots |
| 8 | 8 | Harrison Smith | 81.26 | 78.20 | 81.84 | 848 | Vikings |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Malcolm Jenkins | 79.38 | 75.50 | 77.80 | 1203 | Eagles |
| 10 | 2 | Eric Weddle | 79.17 | 77.89 | 77.43 | 750 | Chargers |
| 11 | 3 | Ha Ha Clinton-Dix | 79.08 | 71.80 | 79.76 | 1176 | Packers |
| 12 | 4 | Mike Mitchell | 78.75 | 78.60 | 74.68 | 1188 | Steelers |
| 13 | 5 | Calvin Pryor | 78.34 | 80.30 | 75.22 | 709 | Jets |
| 14 | 6 | Darian Stewart | 77.42 | 76.20 | 74.70 | 1011 | Broncos |
| 15 | 7 | Shawn Williams | 76.90 | 78.33 | 77.10 | 524 | Bengals |
| 16 | 8 | Reggie Nelson | 76.81 | 74.20 | 74.38 | 1068 | Bengals |
| 17 | 9 | Rodney McLeod | 76.77 | 71.30 | 76.25 | 1149 | Rams |
| 18 | 10 | Andre Hal | 75.80 | 76.02 | 72.13 | 787 | Texans |
| 19 | 11 | George Iloka | 75.24 | 71.53 | 75.12 | 713 | Bengals |
| 20 | 12 | Reshad Jones | 74.62 | 67.30 | 76.59 | 1117 | Dolphins |
| 21 | 13 | Donte Whitner | 74.48 | 70.40 | 74.07 | 847 | Browns |

### Starter (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Husain Abdullah | 73.73 | 67.74 | 75.13 | 486 | Chiefs |
| 23 | 2 | Da'Norris Searcy | 73.62 | 66.20 | 75.23 | 883 | Titans |
| 24 | 3 | Morgan Burnett | 73.31 | 68.35 | 74.43 | 816 | Packers |
| 25 | 4 | Tre Boston | 73.11 | 67.39 | 75.36 | 293 | Panthers |
| 26 | 5 | Chris Banjo | 73.04 | 65.08 | 78.96 | 102 | Packers |
| 27 | 6 | Marcus Gilchrist | 72.87 | 64.70 | 74.15 | 1045 | Jets |
| 28 | 7 | Quintin Demps | 72.30 | 67.76 | 72.00 | 817 | Texans |
| 29 | 8 | Kelcie McCray | 72.29 | 67.83 | 78.40 | 225 | Seahawks |
| 30 | 9 | T.J. Ward | 72.11 | 70.20 | 69.74 | 886 | Broncos |
| 31 | 10 | Ron Parker | 71.59 | 69.70 | 68.69 | 1177 | Chiefs |
| 32 | 11 | Jordan Richards | 71.53 | 67.89 | 72.93 | 244 | Patriots |
| 33 | 12 | Keith Tandy | 71.45 | 65.13 | 75.76 | 280 | Buccaneers |
| 34 | 13 | Micah Hyde | 71.41 | 68.99 | 68.86 | 709 | Packers |
| 35 | 14 | Rontez Miles | 71.27 | 62.24 | 78.33 | 127 | Jets |
| 36 | 15 | Kam Chancellor | 70.44 | 67.79 | 69.60 | 756 | Seahawks |
| 37 | 16 | Eric Reid | 70.11 | 65.80 | 69.13 | 1113 | 49ers |
| 38 | 17 | David Bruton | 69.85 | 66.71 | 71.53 | 480 | Broncos |
| 39 | 18 | Chris Conte | 69.44 | 65.74 | 70.02 | 731 | Buccaneers |
| 40 | 19 | Roman Harper | 68.95 | 63.40 | 69.52 | 1067 | Panthers |
| 41 | 20 | Kemal Ishmael | 68.80 | 66.20 | 70.54 | 375 | Falcons |
| 42 | 21 | Isa Abdul-Quddus | 68.57 | 62.25 | 70.39 | 569 | Lions |
| 43 | 22 | Bradley McDougald | 68.07 | 61.80 | 68.72 | 888 | Buccaneers |
| 44 | 23 | Kendrick Lewis | 67.94 | 64.00 | 66.91 | 925 | Ravens |
| 45 | 24 | Jamarca Sanford | 67.70 | 62.56 | 72.89 | 103 | Saints |
| 46 | 25 | Tony Jefferson | 67.57 | 60.40 | 70.06 | 887 | Cardinals |
| 47 | 26 | Kenny Vaccaro | 67.20 | 62.10 | 67.16 | 1059 | Saints |
| 48 | 27 | Michael Thomas | 67.19 | 67.90 | 68.48 | 697 | Dolphins |
| 49 | 28 | Kyshoen Jarrett | 67.16 | 65.88 | 63.84 | 600 | Commanders |
| 50 | 29 | Glover Quin | 66.52 | 57.40 | 68.43 | 951 | Lions |
| 51 | 30 | Will Hill III | 66.44 | 59.80 | 68.79 | 943 | Ravens |
| 52 | 31 | Taylor Mays | 65.82 | 63.64 | 69.26 | 314 | Raiders |
| 53 | 32 | Eddie Pleasant | 65.56 | 64.94 | 66.39 | 388 | Texans |
| 54 | 33 | Tyvon Branch | 65.17 | 60.88 | 70.84 | 526 | Chiefs |
| 55 | 34 | Dashon Goldson | 64.41 | 61.60 | 63.37 | 1041 | Commanders |
| 56 | 35 | Jordan Poyer | 64.34 | 62.51 | 64.62 | 424 | Browns |
| 57 | 36 | Dwight Lowery | 64.18 | 60.90 | 63.67 | 1084 | Colts |
| 58 | 37 | Anthony Harris | 64.11 | 65.83 | 78.80 | 147 | Vikings |
| 59 | 38 | Will Allen | 64.05 | 60.80 | 64.14 | 941 | Steelers |
| 60 | 39 | Robert Blanton | 63.99 | 58.36 | 67.01 | 231 | Vikings |
| 61 | 40 | Jairus Byrd | 63.43 | 60.50 | 67.56 | 810 | Saints |
| 62 | 41 | Bacarri Rambo | 63.27 | 57.63 | 67.96 | 686 | Bills |
| 63 | 42 | Ibraheim Campbell | 63.21 | 64.60 | 74.08 | 102 | Browns |
| 64 | 43 | Shiloh Keo | 63.19 | 58.68 | 69.22 | 123 | Broncos |
| 65 | 44 | Rashad Johnson | 63.15 | 57.10 | 63.65 | 978 | Cardinals |
| 66 | 45 | Jeff Heath | 63.14 | 59.40 | 64.49 | 205 | Cowboys |
| 67 | 46 | Corey Graham | 62.94 | 50.40 | 67.14 | 976 | Bills |
| 68 | 47 | Robert Golden | 62.85 | 60.00 | 66.94 | 426 | Steelers |
| 69 | 48 | Chris Maragos | 62.65 | 65.67 | 64.71 | 300 | Eagles |
| 70 | 49 | Michael Griffin | 62.53 | 54.50 | 64.65 | 939 | Titans |
| 71 | 50 | Colt Anderson | 62.25 | 61.46 | 66.52 | 161 | Colts |
| 72 | 51 | Mike Adams | 62.05 | 53.00 | 65.49 | 825 | Colts |

### Rotation/backup (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 73 | 1 | Jahleel Addae | 61.95 | 58.05 | 63.52 | 711 | Chargers |
| 74 | 2 | Aaron Williams | 61.42 | 61.16 | 64.92 | 186 | Bills |
| 75 | 3 | James Ihedigbo | 60.99 | 54.37 | 62.38 | 591 | Lions |
| 76 | 4 | Duke Williams | 60.96 | 59.18 | 61.63 | 282 | Bills |
| 77 | 5 | Dion Bailey | 60.53 | 61.85 | 71.45 | 153 | Jets |
| 78 | 6 | J.J. Wilcox | 60.27 | 57.50 | 58.78 | 822 | Cowboys |
| 79 | 7 | Daimion Stafford | 60.05 | 58.36 | 61.79 | 330 | Titans |
| 80 | 8 | Josh Evans | 60.05 | 57.31 | 58.44 | 619 | Jaguars |
| 81 | 9 | Antoine Bethea | 59.70 | 51.04 | 65.99 | 439 | 49ers |
| 82 | 10 | William Moore | 59.54 | 54.71 | 64.01 | 500 | Falcons |
| 83 | 11 | Rahim Moore | 59.37 | 57.15 | 62.61 | 441 | Texans |
| 84 | 12 | Colin Jones | 58.49 | 59.69 | 59.98 | 161 | Panthers |
| 85 | 13 | Barry Church | 58.22 | 49.40 | 60.45 | 865 | Cowboys |
| 86 | 14 | Josh Bush | 57.94 | 56.79 | 63.50 | 370 | Broncos |
| 87 | 15 | Larry Asante | 57.88 | 56.34 | 61.73 | 366 | Raiders |
| 88 | 16 | T.J. McDonald | 57.67 | 53.93 | 59.84 | 766 | Rams |
| 89 | 17 | D.J. Swearinger Sr. | 57.47 | 55.89 | 57.49 | 261 | Cardinals |
| 90 | 18 | Jaquiski Tartt | 56.95 | 49.92 | 58.50 | 718 | 49ers |
| 91 | 19 | Craig Dahl | 56.94 | 55.17 | 58.12 | 429 | Giants |
| 92 | 20 | Clayton Geathers | 56.73 | 51.99 | 57.80 | 270 | Colts |
| 93 | 21 | Brandon Meriweather | 56.09 | 51.30 | 59.19 | 831 | Giants |
| 94 | 22 | Johnathan Cyprien | 55.86 | 50.80 | 56.64 | 1013 | Jaguars |
| 95 | 23 | Chris Prosinski | 55.39 | 55.33 | 61.37 | 337 | Bears |
| 96 | 24 | Walt Aikens | 54.24 | 55.23 | 55.80 | 444 | Dolphins |
| 97 | 25 | Andrew Sendejo | 53.84 | 48.50 | 56.57 | 830 | Vikings |
| 98 | 26 | Landon Collins | 53.63 | 40.00 | 58.55 | 1091 | Giants |
| 99 | 27 | Major Wright | 53.37 | 47.46 | 58.56 | 218 | Buccaneers |
| 100 | 28 | Shamiel Gary | 53.03 | 57.24 | 59.48 | 108 | Dolphins |
| 101 | 29 | Nate Allen | 52.85 | 51.09 | 55.90 | 224 | Raiders |
| 102 | 30 | Tashaun Gipson Sr. | 52.59 | 43.07 | 57.90 | 798 | Browns |
| 103 | 31 | Daniel Sorensen | 51.99 | 54.61 | 54.15 | 235 | Chiefs |
| 104 | 32 | Maurice Alexander | 51.36 | 45.94 | 54.05 | 420 | Rams |
| 105 | 33 | Sergio Brown | 51.18 | 44.35 | 54.49 | 554 | Jaguars |
| 106 | 34 | Jeron Johnson | 49.45 | 47.52 | 56.47 | 196 | Commanders |
| 107 | 35 | James Sample | 49.29 | 53.84 | 60.59 | 129 | Jaguars |
| 108 | 36 | Antone Exum Jr. | 45.00 | 47.48 | 50.50 | 140 | Vikings |

## T — Tackle

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Terron Armstead | 96.86 | 91.91 | 95.99 | 926 | Saints |
| 2 | 2 | Tyron Smith | 96.50 | 91.70 | 95.53 | 1020 | Cowboys |
| 3 | 3 | Joe Thomas | 94.27 | 91.60 | 91.89 | 1103 | Browns |
| 4 | 4 | Joe Staley | 91.04 | 87.30 | 89.37 | 1007 | 49ers |
| 5 | 5 | Jason Peters | 89.64 | 80.57 | 91.52 | 759 | Eagles |
| 6 | 6 | Trent Williams | 89.16 | 82.90 | 89.17 | 993 | Commanders |
| 7 | 7 | Donald Penn | 88.92 | 83.80 | 88.17 | 1034 | Raiders |
| 8 | 8 | Andrew Whitworth | 88.80 | 83.40 | 88.23 | 1094 | Bengals |
| 9 | 9 | Duane Brown | 86.16 | 79.22 | 86.62 | 906 | Texans |
| 10 | 10 | Jared Veldheer | 85.76 | 80.10 | 85.37 | 1193 | Cardinals |
| 11 | 11 | Ryan Schraeder | 85.34 | 78.20 | 85.93 | 1128 | Falcons |
| 12 | 12 | Lane Johnson | 85.08 | 74.80 | 87.76 | 1156 | Eagles |
| 13 | 13 | Jake Matthews | 84.97 | 79.30 | 84.58 | 1126 | Falcons |
| 14 | 14 | Mitchell Schwartz | 84.90 | 79.00 | 84.67 | 1103 | Browns |
| 15 | 15 | Riley Reiff | 84.83 | 78.00 | 85.21 | 1073 | Lions |
| 16 | 16 | Morgan Moses | 84.58 | 76.20 | 86.00 | 1098 | Commanders |
| 17 | 17 | Russell Okung | 84.13 | 76.72 | 84.90 | 914 | Seahawks |
| 18 | 18 | Derek Newton | 83.77 | 75.20 | 85.31 | 1225 | Texans |
| 19 | 19 | Taylor Lewan | 83.45 | 75.63 | 84.49 | 906 | Titans |
| 20 | 20 | David Bakhtiari | 82.53 | 76.00 | 82.71 | 1024 | Packers |
| 21 | 21 | Cordy Glenn | 82.40 | 76.50 | 82.17 | 1059 | Bills |
| 22 | 22 | Austin Howard | 82.26 | 72.36 | 84.69 | 809 | Raiders |
| 23 | 23 | Anthony Castonzo | 81.76 | 74.27 | 82.59 | 891 | Colts |
| 24 | 24 | Marcus Gilbert | 81.55 | 73.10 | 83.01 | 1206 | Steelers |
| 25 | 25 | Joe Barksdale | 81.52 | 72.80 | 83.17 | 1114 | Chargers |
| 26 | 26 | Zach Strief | 81.52 | 72.50 | 83.36 | 1071 | Saints |
| 27 | 27 | Eric Fisher | 81.50 | 72.54 | 83.31 | 976 | Chiefs |
| 28 | 28 | Demar Dotson | 81.03 | 69.46 | 84.58 | 201 | Buccaneers |
| 29 | 29 | Bobby Massie | 80.63 | 70.10 | 83.48 | 1103 | Cardinals |
| 30 | 30 | Jermey Parnell | 80.35 | 70.49 | 82.76 | 983 | Jaguars |

### Good (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Matt Kalil | 79.14 | 68.20 | 82.26 | 1072 | Vikings |
| 32 | 2 | Doug Free | 79.14 | 67.20 | 82.94 | 1020 | Cowboys |
| 33 | 3 | Eugene Monroe | 78.89 | 69.54 | 80.96 | 316 | Ravens |
| 34 | 4 | Tyler Polumbus | 78.79 | 65.38 | 83.57 | 156 | Broncos |
| 35 | 5 | Kelvin Beachum | 78.70 | 65.95 | 83.03 | 324 | Steelers |
| 36 | 6 | Charles Leno Jr. | 78.36 | 68.20 | 80.97 | 917 | Bears |
| 37 | 7 | Sebastian Vollmer | 78.22 | 68.71 | 80.40 | 951 | Patriots |
| 38 | 8 | Branden Albert | 78.21 | 69.36 | 79.94 | 787 | Dolphins |
| 39 | 9 | Denzelle Good | 78.00 | 64.12 | 83.09 | 275 | Colts |
| 40 | 10 | Nate Solder | 77.98 | 66.11 | 81.73 | 225 | Patriots |
| 41 | 11 | Chris Clark | 77.93 | 65.85 | 81.81 | 492 | Texans |
| 42 | 12 | Mike Remmers | 76.69 | 66.20 | 79.52 | 1305 | Panthers |
| 43 | 13 | Sam Young | 76.63 | 63.50 | 81.21 | 233 | Jaguars |
| 44 | 14 | Michael Oher | 75.92 | 65.60 | 78.63 | 1287 | Panthers |
| 45 | 15 | Cyrus Kouandjio | 75.92 | 60.48 | 82.04 | 226 | Bills |
| 46 | 16 | Jamon Meredith | 75.81 | 63.41 | 79.91 | 393 | Titans |
| 47 | 17 | Michael Ola | 75.79 | 64.39 | 79.22 | 450 | Lions |
| 48 | 18 | Tom Compton | 75.77 | 64.31 | 79.24 | 231 | Commanders |
| 49 | 19 | King Dunlap | 75.40 | 65.38 | 77.91 | 310 | Chargers |
| 50 | 20 | Khalif Barnes | 74.89 | 61.04 | 79.96 | 110 | Raiders |
| 51 | 21 | Ty Nsekhe | 74.86 | 65.56 | 76.90 | 192 | Commanders |
| 52 | 22 | Bryan Bulaga | 74.82 | 65.25 | 77.03 | 952 | Packers |
| 53 | 23 | Rick Wagner | 74.65 | 64.30 | 77.38 | 1126 | Ravens |
| 54 | 24 | Dennis Kelly | 74.50 | 63.29 | 77.80 | 395 | Eagles |
| 55 | 25 | Ja'Wuan James | 74.29 | 62.14 | 78.23 | 391 | Dolphins |
| 56 | 26 | Donovan Smith | 74.14 | 62.60 | 77.66 | 1087 | Buccaneers |

### Starter (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 57 | 1 | Greg Robinson | 73.69 | 60.89 | 78.05 | 959 | Rams |
| 58 | 2 | Donald Stephenson | 73.67 | 60.59 | 78.23 | 700 | Chiefs |
| 59 | 3 | D'Brickashaw Ferguson | 73.43 | 62.40 | 76.61 | 1107 | Jets |
| 60 | 4 | Michael Schofield III | 73.35 | 62.40 | 76.48 | 1068 | Broncos |
| 61 | 5 | Ryan Harris | 73.26 | 62.00 | 76.60 | 1196 | Broncos |
| 62 | 6 | Garry Gilliam | 73.04 | 60.90 | 76.97 | 1180 | Seahawks |
| 63 | 7 | Cornelius Lucas | 72.95 | 59.43 | 77.79 | 323 | Lions |
| 64 | 8 | Eric Winston | 72.73 | 61.24 | 76.23 | 172 | Bengals |
| 65 | 9 | Jeremiah Poutasi | 72.27 | 59.94 | 76.32 | 394 | Titans |
| 66 | 10 | Alejandro Villanueva | 71.92 | 61.33 | 74.82 | 890 | Steelers |
| 67 | 11 | Breno Giacomini | 71.32 | 59.00 | 75.36 | 1104 | Jets |
| 68 | 12 | Bobby Hart | 71.27 | 58.48 | 75.63 | 153 | Giants |
| 69 | 13 | Erik Pears | 70.69 | 58.60 | 74.59 | 1007 | 49ers |
| 70 | 14 | Marcus Cannon | 70.57 | 56.84 | 75.55 | 758 | Patriots |
| 71 | 15 | Jah Reid | 70.42 | 56.62 | 75.45 | 738 | Chiefs |
| 72 | 16 | Chris Hairston | 70.22 | 57.68 | 74.41 | 783 | Chargers |
| 73 | 17 | Ty Sambrailo | 70.19 | 57.57 | 74.43 | 207 | Broncos |
| 74 | 18 | T.J. Clemmings | 70.13 | 58.50 | 73.72 | 1073 | Vikings |
| 75 | 19 | Jordan Mills | 69.69 | 56.58 | 74.26 | 355 | Bills |
| 76 | 20 | Trent Brown | 69.66 | 61.78 | 70.74 | 186 | 49ers |
| 77 | 21 | Cam Fleming | 69.40 | 56.62 | 73.76 | 468 | Patriots |
| 78 | 22 | Gosder Cherilus | 69.01 | 54.77 | 74.34 | 886 | Buccaneers |
| 79 | 23 | Ereck Flowers | 68.87 | 55.07 | 73.90 | 958 | Giants |
| 80 | 24 | Seantrel Henderson | 68.29 | 53.26 | 74.15 | 592 | Bills |
| 81 | 25 | Marshall Newhouse | 67.95 | 54.26 | 72.91 | 933 | Giants |
| 82 | 26 | Jake Fisher | 67.12 | 58.48 | 68.71 | 132 | Bengals |
| 83 | 27 | Kendall Lamm | 66.14 | 54.74 | 69.58 | 262 | Texans |
| 84 | 28 | James Hurst | 64.19 | 49.28 | 69.97 | 569 | Ravens |
| 85 | 29 | Don Barclay | 63.28 | 49.15 | 68.53 | 421 | Packers |
| 86 | 30 | LaAdrian Waddle | 62.56 | 46.40 | 69.17 | 409 | Patriots |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 87.42 | 91.30 | 80.66 | 720 | Patriots |
| 2 | 2 | Greg Olsen | 85.89 | 89.88 | 79.06 | 623 | Panthers |
| 3 | 3 | Jordan Reed | 83.57 | 86.17 | 77.67 | 499 | Commanders |
| 4 | 4 | Delanie Walker | 82.29 | 86.11 | 75.57 | 520 | Titans |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Zach Miller | 79.98 | 80.35 | 75.56 | 256 | Bears |
| 6 | 2 | Tyler Eifert | 79.17 | 82.37 | 72.87 | 485 | Bengals |
| 7 | 3 | Zach Ertz | 78.39 | 80.74 | 72.65 | 532 | Eagles |
| 8 | 4 | Brent Celek | 76.60 | 78.03 | 71.48 | 261 | Eagles |
| 9 | 5 | Travis Kelce | 76.21 | 73.61 | 73.77 | 646 | Chiefs |
| 10 | 6 | Jimmy Graham | 76.04 | 76.11 | 71.82 | 368 | Seahawks |
| 11 | 7 | Benjamin Watson | 75.65 | 74.84 | 72.03 | 622 | Saints |
| 12 | 8 | Austin Seferian-Jenkins | 74.98 | 68.17 | 75.36 | 162 | Buccaneers |
| 13 | 9 | Garrett Celek | 74.59 | 63.88 | 77.56 | 209 | 49ers |
| 14 | 10 | Charles Clay | 74.38 | 71.71 | 71.99 | 435 | Bills |
| 15 | 11 | Craig Stevens | 74.37 | 74.05 | 70.42 | 156 | Titans |
| 16 | 12 | Jason Witten | 74.34 | 66.32 | 75.52 | 602 | Cowboys |
| 17 | 13 | Antonio Gates | 74.12 | 75.41 | 69.10 | 389 | Chargers |

### Starter (60 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Marcedes Lewis | 73.38 | 66.08 | 74.08 | 402 | Jaguars |
| 19 | 2 | Gary Barnidge | 73.28 | 74.60 | 68.24 | 664 | Browns |
| 20 | 3 | Nick Boyle | 73.09 | 67.43 | 72.70 | 164 | Ravens |
| 21 | 4 | Will Tye | 72.97 | 65.18 | 74.00 | 324 | Giants |
| 22 | 5 | Heath Miller | 72.77 | 67.20 | 72.32 | 681 | Steelers |
| 23 | 6 | Vernon Davis | 72.63 | 61.38 | 75.97 | 389 | Broncos |
| 24 | 7 | Cameron Brate | 72.56 | 64.00 | 74.10 | 228 | Buccaneers |
| 25 | 8 | Rhett Ellison | 72.55 | 68.12 | 71.33 | 154 | Vikings |
| 26 | 9 | Martellus Bennett | 72.54 | 71.31 | 69.19 | 405 | Bears |
| 27 | 10 | Coby Fleener | 72.33 | 64.20 | 73.59 | 520 | Colts |
| 28 | 11 | Ladarius Green | 72.30 | 65.51 | 72.66 | 447 | Chargers |
| 29 | 12 | Virgil Green | 72.15 | 67.19 | 71.29 | 208 | Broncos |
| 30 | 13 | Scott Chandler | 72.11 | 64.79 | 72.83 | 255 | Patriots |
| 31 | 14 | Jermaine Gresham | 71.81 | 70.70 | 68.39 | 291 | Cardinals |
| 32 | 15 | Tyler Kroft | 71.39 | 62.43 | 73.19 | 162 | Bengals |
| 33 | 16 | Darren Fells | 71.36 | 65.99 | 70.77 | 379 | Cardinals |
| 34 | 17 | Julius Thomas | 71.22 | 62.80 | 72.66 | 414 | Jaguars |
| 35 | 18 | Owen Daniels | 71.06 | 65.04 | 70.90 | 596 | Broncos |
| 36 | 19 | Richard Rodgers | 70.99 | 69.68 | 67.70 | 540 | Packers |
| 37 | 20 | Luke Willson | 70.69 | 66.38 | 69.40 | 263 | Seahawks |
| 38 | 21 | Vance McDonald | 70.44 | 63.76 | 70.73 | 268 | 49ers |
| 39 | 22 | Derek Carrier | 70.33 | 62.03 | 71.70 | 189 | Commanders |
| 40 | 23 | Anthony Fasano | 70.30 | 67.53 | 67.98 | 308 | Titans |
| 41 | 24 | Jordan Cameron | 70.24 | 61.53 | 71.88 | 476 | Dolphins |
| 42 | 25 | Gavin Escobar | 70.07 | 59.52 | 72.93 | 107 | Cowboys |
| 43 | 26 | Clive Walford | 70.00 | 67.10 | 67.77 | 243 | Raiders |
| 44 | 27 | Kyle Rudolph | 69.94 | 70.42 | 65.45 | 485 | Vikings |
| 45 | 28 | Lance Kendricks | 69.42 | 61.53 | 70.52 | 211 | Rams |
| 46 | 29 | Clay Harbor | 69.36 | 59.81 | 71.56 | 154 | Jaguars |
| 47 | 30 | Jacob Tamme | 69.36 | 66.33 | 67.21 | 490 | Falcons |
| 48 | 31 | Mychal Rivera | 68.92 | 56.71 | 72.90 | 220 | Raiders |
| 49 | 32 | Josh Hill | 68.91 | 59.88 | 70.77 | 223 | Saints |
| 50 | 33 | Jeff Cumberland | 68.90 | 58.09 | 71.94 | 125 | Jets |
| 51 | 34 | Jesse James | 68.77 | 65.15 | 67.02 | 130 | Steelers |
| 52 | 35 | Larry Donnell | 68.75 | 59.08 | 71.03 | 219 | Giants |
| 53 | 36 | Jared Cook | 68.69 | 57.76 | 71.81 | 393 | Rams |
| 54 | 37 | C.J. Fiedorowicz | 68.31 | 61.78 | 68.50 | 308 | Texans |
| 55 | 38 | Cooper Helfet | 68.16 | 58.72 | 70.29 | 161 | Seahawks |
| 56 | 39 | John Phillips | 67.99 | 65.56 | 65.44 | 101 | Chargers |
| 57 | 40 | Brandon Myers | 67.94 | 60.14 | 68.97 | 172 | Buccaneers |
| 58 | 41 | Lee Smith | 67.73 | 65.36 | 65.15 | 219 | Raiders |
| 59 | 42 | Dwayne Allen | 67.49 | 57.74 | 69.82 | 266 | Colts |
| 60 | 43 | Matt Spaeth | 67.40 | 60.68 | 67.71 | 101 | Steelers |
| 61 | 44 | Tim Wright | 67.32 | 54.40 | 71.76 | 137 | Lions |
| 62 | 45 | Luke Stocker | 67.03 | 63.07 | 65.51 | 173 | Buccaneers |
| 63 | 46 | Michael Hoomanawanui | 67.03 | 59.74 | 67.72 | 120 | Saints |
| 64 | 47 | Ed Dickson | 67.00 | 58.34 | 68.61 | 270 | Panthers |
| 65 | 48 | Eric Ebron | 66.70 | 64.15 | 64.23 | 437 | Lions |
| 66 | 49 | Michael Williams | 66.40 | 61.73 | 65.35 | 206 | Patriots |
| 67 | 50 | Brandon Pettigrew | 66.37 | 59.15 | 67.01 | 146 | Lions |
| 68 | 51 | Chris Gragg | 66.30 | 61.36 | 65.43 | 211 | Bills |
| 69 | 52 | Maxx Williams | 65.79 | 61.94 | 64.19 | 316 | Ravens |
| 70 | 53 | Demetrius Harris | 64.80 | 59.84 | 63.94 | 189 | Chiefs |
| 71 | 54 | Jim Dray | 64.29 | 53.71 | 67.17 | 212 | Browns |
| 72 | 55 | Ryan Griffin | 64.24 | 56.84 | 65.00 | 264 | Texans |
| 73 | 56 | Dion Sims | 64.06 | 60.12 | 62.52 | 222 | Dolphins |
| 74 | 57 | Blake Bell | 63.57 | 54.31 | 65.57 | 232 | 49ers |
| 75 | 58 | Levine Toilolo | 63.38 | 58.19 | 62.68 | 241 | Falcons |
| 76 | 59 | Kellen Davis | 62.66 | 52.23 | 65.44 | 187 | Jets |
| 77 | 60 | Garrett Graham | 62.64 | 48.88 | 67.64 | 208 | Texans |

### Rotation/backup (0 players)

_None._

## WR — Wide Receiver

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Julio Jones | 89.16 | 92.46 | 82.79 | 657 | Falcons |
| 2 | 2 | Antonio Brown | 88.96 | 92.50 | 82.43 | 725 | Steelers |
| 3 | 3 | Odell Beckham Jr. | 88.15 | 87.76 | 84.24 | 634 | Giants |
| 4 | 4 | A.J. Green | 87.22 | 89.92 | 81.26 | 626 | Bengals |
| 5 | 5 | DeAndre Hopkins | 85.70 | 90.60 | 78.26 | 738 | Texans |
| 6 | 6 | Alshon Jeffery | 85.34 | 85.64 | 80.97 | 295 | Bears |
| 7 | 7 | Calvin Johnson | 85.30 | 88.60 | 78.94 | 688 | Lions |
| 8 | 8 | Sammy Watkins | 84.30 | 83.74 | 80.50 | 419 | Bills |
| 9 | 9 | Larry Fitzgerald | 84.24 | 89.40 | 76.63 | 694 | Cardinals |
| 10 | 10 | Doug Baldwin | 84.23 | 87.71 | 77.74 | 622 | Seahawks |
| 11 | 11 | Allen Robinson II | 84.07 | 84.00 | 79.95 | 682 | Jaguars |
| 12 | 12 | Emmanuel Sanders | 83.59 | 84.40 | 78.88 | 661 | Broncos |
| 13 | 13 | Steve Smith | 82.77 | 83.29 | 78.26 | 249 | Ravens |
| 14 | 14 | DeSean Jackson | 82.47 | 71.93 | 85.33 | 302 | Commanders |
| 15 | 15 | T.Y. Hilton | 82.39 | 78.85 | 80.58 | 654 | Colts |
| 16 | 16 | J.J. Nelson | 82.35 | 69.82 | 86.53 | 128 | Cardinals |
| 17 | 17 | Mike Evans | 82.11 | 82.39 | 77.75 | 552 | Buccaneers |
| 18 | 18 | Brandon Marshall | 81.92 | 85.40 | 75.43 | 665 | Jets |
| 19 | 19 | Jarvis Landry | 81.57 | 86.98 | 73.80 | 582 | Dolphins |
| 20 | 20 | Jeremy Maclin | 80.76 | 77.61 | 78.69 | 580 | Chiefs |
| 21 | 21 | John Brown | 80.37 | 78.30 | 77.58 | 632 | Cardinals |
| 22 | 22 | Michael Floyd | 80.36 | 78.70 | 77.30 | 525 | Cardinals |

### Good (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Allen Hurns | 79.95 | 78.09 | 77.02 | 611 | Jaguars |
| 24 | 2 | Demaryius Thomas | 79.79 | 76.40 | 77.89 | 715 | Broncos |
| 25 | 3 | Stefon Diggs | 79.27 | 75.18 | 77.83 | 449 | Vikings |
| 26 | 4 | Vincent Jackson | 79.13 | 76.10 | 76.98 | 313 | Buccaneers |
| 27 | 5 | Willie Snead IV | 78.56 | 72.82 | 78.22 | 546 | Saints |
| 28 | 6 | Eric Decker | 78.21 | 77.39 | 74.59 | 588 | Jets |
| 29 | 7 | Martavis Bryant | 78.03 | 70.72 | 78.73 | 478 | Steelers |
| 30 | 8 | Jeff Janis | 77.80 | 64.27 | 82.65 | 111 | Packers |
| 31 | 9 | Keenan Allen | 77.76 | 76.18 | 74.65 | 359 | Chargers |
| 32 | 10 | Julian Edelman | 77.74 | 81.76 | 70.90 | 478 | Patriots |
| 33 | 11 | Golden Tate | 77.66 | 77.40 | 73.67 | 676 | Lions |
| 34 | 12 | Ted Ginn Jr. | 77.56 | 71.12 | 77.69 | 494 | Panthers |
| 35 | 13 | Brandin Cooks | 77.50 | 73.10 | 76.27 | 673 | Saints |
| 36 | 14 | Kamar Aiken | 77.44 | 77.95 | 72.94 | 621 | Ravens |
| 37 | 15 | Tyler Lockett | 77.25 | 74.05 | 75.21 | 515 | Seahawks |
| 38 | 16 | Terrance Williams | 76.77 | 69.02 | 77.77 | 548 | Cowboys |
| 39 | 17 | Kenny Britt | 76.31 | 70.83 | 75.80 | 387 | Rams |
| 40 | 18 | Marvin Jones Jr. | 76.31 | 72.53 | 74.67 | 599 | Bengals |
| 41 | 19 | Dorial Green-Beckham | 76.13 | 68.98 | 76.73 | 432 | Titans |
| 42 | 20 | Anquan Boldin | 75.97 | 71.18 | 74.99 | 519 | 49ers |
| 43 | 21 | DeVante Parker | 75.90 | 68.87 | 76.42 | 312 | Dolphins |
| 44 | 22 | Dez Bryant | 75.83 | 67.06 | 77.51 | 311 | Cowboys |
| 45 | 23 | Pierre Garcon | 75.76 | 74.56 | 72.40 | 584 | Commanders |
| 46 | 24 | Rishard Matthews | 75.48 | 70.02 | 74.96 | 338 | Dolphins |
| 47 | 25 | Jermaine Kearse | 75.23 | 70.51 | 74.21 | 592 | Seahawks |
| 48 | 26 | Michael Crabtree | 75.06 | 76.68 | 69.81 | 595 | Raiders |
| 49 | 27 | James Jones | 74.85 | 69.70 | 74.11 | 800 | Packers |
| 50 | 28 | Devin Funchess | 74.75 | 69.69 | 73.95 | 287 | Panthers |
| 51 | 29 | Corey Brown | 74.75 | 68.97 | 74.44 | 531 | Panthers |
| 52 | 30 | Donte Moncrief | 74.52 | 71.17 | 72.58 | 562 | Colts |
| 53 | 31 | Amari Cooper | 74.52 | 70.74 | 72.87 | 607 | Raiders |
| 54 | 32 | Travis Benjamin | 74.49 | 66.83 | 75.43 | 628 | Browns |
| 55 | 33 | Jerricho Cotchery | 74.41 | 73.03 | 71.17 | 307 | Panthers |
| 56 | 34 | Markus Wheaton | 74.34 | 67.81 | 74.52 | 571 | Steelers |
| 57 | 35 | Kendall Wright | 74.31 | 67.66 | 74.58 | 290 | Titans |

### Starter (89 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 58 | 1 | Torrey Smith | 73.56 | 62.63 | 76.68 | 533 | 49ers |
| 59 | 2 | Jarius Wright | 73.53 | 65.86 | 74.47 | 329 | Vikings |
| 60 | 3 | Randall Cobb | 73.42 | 68.50 | 72.53 | 723 | Packers |
| 61 | 4 | Brandon Coleman | 73.26 | 68.18 | 72.48 | 317 | Saints |
| 62 | 5 | Bryan Walters | 73.18 | 69.50 | 71.46 | 258 | Jaguars |
| 63 | 6 | Brice Butler | 73.09 | 64.11 | 74.91 | 174 | Cowboys |
| 64 | 7 | Danny Amendola | 73.06 | 74.15 | 68.17 | 517 | Patriots |
| 65 | 8 | Marc Mariani | 72.95 | 65.35 | 73.85 | 336 | Bears |
| 66 | 9 | Malcom Floyd | 72.83 | 61.62 | 76.13 | 559 | Chargers |
| 67 | 10 | Jordan Matthews | 72.62 | 65.36 | 73.30 | 606 | Eagles |
| 68 | 11 | Rueben Randle | 72.43 | 63.88 | 73.97 | 651 | Giants |
| 69 | 12 | Andre Johnson | 71.84 | 65.69 | 71.78 | 490 | Colts |
| 70 | 13 | Kenny Stills | 71.66 | 59.91 | 75.33 | 430 | Dolphins |
| 71 | 14 | Adam Thielen | 71.65 | 62.11 | 73.84 | 109 | Vikings |
| 72 | 15 | Nate Washington | 71.34 | 64.21 | 71.92 | 524 | Texans |
| 73 | 16 | Percy Harvin | 71.26 | 65.16 | 71.16 | 152 | Bills |
| 74 | 17 | Andrew Hawkins | 71.26 | 59.76 | 74.76 | 286 | Browns |
| 75 | 18 | Bennie Fowler | 71.25 | 63.75 | 72.08 | 201 | Broncos |
| 76 | 19 | Marques Colston | 71.12 | 62.45 | 72.73 | 416 | Saints |
| 77 | 20 | Jamison Crowder | 71.11 | 68.48 | 68.69 | 526 | Commanders |
| 78 | 21 | Jeremy Butler | 71.04 | 63.15 | 72.13 | 273 | Ravens |
| 79 | 22 | Chris Givens | 70.95 | 59.65 | 74.32 | 318 | Ravens |
| 80 | 23 | Brian Hartline | 70.94 | 67.04 | 69.38 | 386 | Browns |
| 81 | 24 | Stedman Bailey | 70.91 | 59.71 | 74.21 | 155 | Rams |
| 82 | 25 | Cole Beasley | 70.86 | 63.50 | 71.60 | 438 | Cowboys |
| 83 | 26 | Albert Wilson | 70.72 | 62.17 | 72.25 | 495 | Chiefs |
| 84 | 27 | Phillip Dorsett | 70.66 | 60.24 | 73.44 | 150 | Colts |
| 85 | 28 | Quinton Patton | 70.64 | 63.97 | 70.92 | 320 | 49ers |
| 86 | 29 | Justin Hunter | 70.43 | 61.92 | 71.93 | 252 | Titans |
| 87 | 30 | Marquess Wilson | 70.36 | 62.70 | 71.30 | 392 | Bears |
| 88 | 31 | Steve Johnson | 69.94 | 62.79 | 70.54 | 406 | Chargers |
| 89 | 32 | T.J. Jones | 69.92 | 60.86 | 71.79 | 111 | Lions |
| 90 | 33 | Andre Holmes | 69.76 | 58.76 | 72.92 | 208 | Raiders |
| 91 | 34 | Chris Matthews | 69.58 | 61.44 | 70.84 | 152 | Ravens |
| 92 | 35 | Tavon Austin | 69.48 | 67.69 | 66.50 | 432 | Rams |
| 93 | 36 | Jaelen Strong | 69.38 | 62.58 | 69.75 | 190 | Texans |
| 94 | 37 | Harry Douglas | 69.28 | 60.08 | 71.24 | 465 | Titans |
| 95 | 38 | Louis Murphy Jr. | 69.04 | 62.05 | 69.54 | 150 | Buccaneers |
| 96 | 39 | Riley Cooper | 68.79 | 58.68 | 71.36 | 357 | Eagles |
| 97 | 40 | Darrius Heyward-Bey | 68.79 | 60.57 | 70.11 | 308 | Steelers |
| 98 | 41 | Dontrelle Inman | 68.54 | 57.80 | 71.54 | 474 | Chargers |
| 99 | 42 | Mike Wallace | 68.54 | 58.79 | 70.87 | 517 | Vikings |
| 100 | 43 | Seth Roberts | 68.38 | 59.92 | 69.86 | 446 | Raiders |
| 101 | 44 | Kenbrell Thompkins | 68.34 | 60.29 | 69.54 | 221 | Jets |
| 102 | 45 | Jeremy Kerley | 68.25 | 61.74 | 68.42 | 172 | Jets |
| 103 | 46 | Eddie Royal | 68.15 | 59.52 | 69.74 | 284 | Bears |
| 104 | 47 | Josh Huff | 68.13 | 63.02 | 67.37 | 341 | Eagles |
| 105 | 48 | Robert Woods | 68.11 | 61.53 | 68.33 | 478 | Bills |
| 106 | 49 | Aaron Dobson | 68.08 | 60.19 | 69.18 | 153 | Patriots |
| 107 | 50 | Taylor Gabriel | 68.05 | 57.74 | 70.76 | 292 | Browns |
| 108 | 51 | Corey Fuller | 67.87 | 57.88 | 70.37 | 114 | Lions |
| 109 | 52 | Cecil Shorts | 67.78 | 60.35 | 68.57 | 411 | Texans |
| 110 | 53 | Leonard Hankerson | 67.73 | 58.46 | 69.74 | 252 | Bills |
| 111 | 54 | Keshawn Martin | 67.63 | 59.55 | 68.85 | 369 | Patriots |
| 112 | 55 | Roddy White | 67.60 | 62.56 | 66.80 | 629 | Falcons |
| 113 | 56 | Marqise Lee | 67.57 | 60.84 | 67.89 | 183 | Jaguars |
| 114 | 57 | Devin Street | 67.57 | 59.00 | 69.11 | 137 | Cowboys |
| 115 | 58 | Jared Abbrederis | 67.50 | 60.58 | 67.94 | 133 | Packers |
| 116 | 59 | Jaron Brown | 67.42 | 61.39 | 67.28 | 174 | Cardinals |
| 117 | 60 | Chris Hogan | 67.41 | 60.23 | 68.03 | 394 | Bills |
| 118 | 61 | Brian Quick | 67.36 | 56.91 | 70.16 | 169 | Rams |
| 119 | 62 | Nick Williams | 67.31 | 62.18 | 66.56 | 125 | Falcons |
| 120 | 63 | Hakeem Nicks | 67.16 | 54.67 | 71.32 | 115 | Giants |
| 121 | 64 | Josh Bellamy | 67.13 | 60.41 | 67.45 | 226 | Bears |
| 122 | 65 | Brandon LaFell | 67.11 | 56.67 | 69.91 | 503 | Patriots |
| 123 | 66 | Adam Humphries | 67.08 | 62.43 | 66.02 | 317 | Buccaneers |
| 124 | 67 | Chris Conley | 66.75 | 60.73 | 66.59 | 291 | Chiefs |
| 125 | 68 | Greg Jennings | 66.73 | 55.09 | 70.33 | 217 | Dolphins |
| 126 | 69 | Mohamed Sanu | 66.71 | 57.44 | 68.72 | 480 | Bengals |
| 127 | 70 | Lance Moore | 66.11 | 57.11 | 67.95 | 463 | Lions |
| 128 | 71 | Dwayne Harris | 66.08 | 60.81 | 65.43 | 435 | Giants |
| 129 | 72 | Jason Avant | 65.99 | 59.68 | 66.03 | 291 | Chiefs |
| 130 | 73 | Ryan Grant | 65.98 | 60.57 | 65.42 | 267 | Commanders |
| 131 | 74 | Bradley Marquez | 65.46 | 61.41 | 63.99 | 120 | Rams |
| 132 | 75 | Justin Hardy | 65.46 | 62.10 | 63.53 | 238 | Falcons |
| 133 | 76 | Davante Adams | 65.39 | 59.18 | 65.36 | 550 | Packers |
| 134 | 77 | Andre Roberts | 65.30 | 55.64 | 67.57 | 166 | Commanders |
| 135 | 78 | Jordan Norwood | 65.28 | 56.88 | 66.71 | 332 | Broncos |
| 136 | 79 | Quincy Enunwa | 65.16 | 55.12 | 67.69 | 350 | Jets |
| 137 | 80 | Cody Latimer | 63.95 | 60.53 | 62.07 | 111 | Broncos |
| 138 | 81 | Devin Smith | 63.86 | 55.36 | 65.36 | 240 | Jets |
| 139 | 82 | Javontee Herndon | 63.58 | 59.83 | 61.91 | 206 | Chargers |
| 140 | 83 | Myles White | 63.54 | 58.84 | 62.51 | 113 | Giants |
| 141 | 84 | Donteea Dye Jr. | 63.08 | 55.96 | 63.66 | 272 | Buccaneers |
| 142 | 85 | Marlon Moore | 62.75 | 57.32 | 62.21 | 141 | Browns |
| 143 | 86 | Marlon Brown | 62.68 | 53.86 | 64.39 | 254 | Ravens |
| 144 | 87 | Russell Shepard | 62.56 | 59.18 | 60.65 | 110 | Buccaneers |
| 145 | 88 | Nelson Agholor | 62.37 | 50.88 | 65.86 | 430 | Eagles |
| 146 | 89 | Andre Caldwell | 62.03 | 57.63 | 60.79 | 183 | Broncos |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 147 | 1 | Darius Jennings | 61.17 | 60.45 | 57.49 | 133 | Browns |
| 148 | 2 | Rashad Greene Sr. | 60.08 | 58.33 | 57.08 | 134 | Jaguars |
