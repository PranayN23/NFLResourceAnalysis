# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:27:01Z
- **Requested analysis_year:** 2015 (clamped to 2015)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Travis Frederick | 89.55 | 82.80 | 89.88 | 1020 | Cowboys |
| 2 | 2 | Weston Richburg | 86.50 | 83.30 | 84.46 | 1013 | Giants |
| 3 | 3 | Corey Linsley | 86.31 | 79.30 | 86.81 | 942 | Packers |
| 4 | 4 | Ryan Kalil | 83.83 | 76.00 | 84.89 | 1193 | Panthers |
| 5 | 5 | Ben Jones | 83.78 | 74.90 | 85.53 | 1245 | Texans |
| 6 | 6 | Mike Pouncey | 83.76 | 76.20 | 84.63 | 786 | Dolphins |
| 7 | 7 | Matt Paradis | 83.28 | 75.70 | 84.16 | 1299 | Broncos |
| 8 | 8 | Rodney Hudson | 83.28 | 73.80 | 85.43 | 801 | Raiders |
| 9 | 9 | Jason Kelce | 83.23 | 74.10 | 85.15 | 1156 | Eagles |
| 10 | 10 | A.Q. Shipley | 82.90 | 74.70 | 84.20 | 155 | Cardinals |
| 11 | 11 | Nick Mangold | 82.58 | 74.10 | 84.07 | 933 | Jets |
| 12 | 12 | Jeremy Zuttah | 82.57 | 74.20 | 83.98 | 610 | Ravens |
| 13 | 13 | J.C. Tretter | 81.38 | 75.40 | 81.20 | 445 | Packers |
| 14 | 14 | Stefen Wisniewski | 80.96 | 72.50 | 82.43 | 1058 | Jaguars |
| 15 | 15 | Alex Mack | 80.92 | 71.80 | 82.84 | 1103 | Browns |
| 16 | 16 | Eric Wood | 80.32 | 71.70 | 81.90 | 1075 | Bills |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Evan Smith | 79.99 | 71.00 | 81.81 | 370 | Buccaneers |
| 18 | 2 | Max Unger | 79.47 | 70.50 | 81.28 | 1154 | Saints |
| 19 | 3 | Josh LeRibeus | 79.03 | 69.40 | 81.29 | 731 | Commanders |
| 20 | 4 | Wesley Johnson | 77.29 | 70.70 | 77.51 | 170 | Jets |
| 21 | 5 | Mitch Morse | 74.93 | 68.30 | 75.18 | 916 | Chiefs |
| 22 | 6 | Brian Schwenke | 74.63 | 63.00 | 78.22 | 303 | Titans |
| 23 | 7 | James Stone | 74.54 | 66.80 | 75.54 | 133 | Falcons |
| 24 | 8 | Joe Hawley | 74.22 | 63.80 | 77.00 | 981 | Buccaneers |
| 25 | 9 | Patrick Lewis | 74.03 | 62.70 | 77.42 | 723 | Seahawks |

### Starter (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | David Andrews | 73.07 | 61.50 | 76.62 | 776 | Patriots |
| 27 | 2 | Tony Bergstrom | 73.06 | 68.30 | 72.07 | 250 | Raiders |
| 28 | 3 | Russell Bodine | 73.02 | 62.20 | 76.06 | 1126 | Bengals |
| 29 | 4 | Will Montgomery | 73.01 | 60.50 | 77.19 | 188 | Bears |
| 30 | 5 | Lyle Sendlein | 72.69 | 61.90 | 75.72 | 1112 | Cardinals |
| 31 | 6 | Jonotthan Harrison | 72.65 | 62.20 | 75.45 | 669 | Colts |
| 32 | 7 | Bryan Stork | 72.38 | 59.00 | 77.14 | 612 | Patriots |
| 33 | 8 | Travis Swanson | 72.02 | 60.30 | 75.67 | 947 | Lions |
| 34 | 9 | Khaled Holmes | 71.40 | 59.60 | 75.10 | 485 | Colts |
| 35 | 10 | Tim Barnes | 71.21 | 58.40 | 75.59 | 955 | Rams |
| 36 | 11 | Drew Nowak | 71.19 | 61.80 | 73.29 | 461 | Seahawks |
| 37 | 12 | Cody Wallace | 70.15 | 56.80 | 74.88 | 1207 | Steelers |
| 38 | 13 | J.D. Walton | 68.96 | 55.40 | 73.84 | 100 | Chargers |
| 39 | 14 | Daniel Kilgore | 67.79 | 56.70 | 71.02 | 267 | 49ers |
| 40 | 15 | Kory Lichtensteiger | 67.44 | 54.70 | 71.77 | 437 | Commanders |
| 41 | 16 | Demetrius Rhaney | 66.04 | 55.90 | 68.63 | 124 | Rams |
| 42 | 17 | Hroniss Grasu | 64.38 | 53.50 | 67.47 | 552 | Bears |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 43 | 1 | Trevor Robinson | 61.39 | 46.10 | 67.42 | 980 | Chargers |
| 44 | 2 | Andy Gallik | 59.30 | 40.80 | 67.46 | 505 | Titans |
| 45 | 3 | Chris Watt | 55.68 | 40.40 | 61.70 | 177 | Chargers |

## CB — Cornerback

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Josh Norman | 88.60 | 89.90 | 86.59 | 1240 | Panthers |
| 2 | 2 | Jason Verrett | 87.63 | 90.90 | 86.48 | 718 | Chargers |
| 3 | 3 | Patrick Peterson | 87.23 | 84.50 | 84.89 | 1116 | Cardinals |
| 4 | 4 | Richard Sherman | 86.59 | 79.40 | 87.22 | 1098 | Seahawks |
| 5 | 5 | Johnathan Joseph | 85.13 | 83.00 | 82.59 | 925 | Texans |
| 6 | 6 | Delvin Breaux Sr. | 83.79 | 75.90 | 84.88 | 941 | Saints |
| 7 | 7 | Darius Slay | 82.23 | 77.00 | 82.58 | 995 | Lions |
| 8 | 8 | Chris Harris Jr. | 82.05 | 77.50 | 80.91 | 1261 | Broncos |
| 9 | 9 | Cody Riggs | 81.99 | 80.00 | 87.49 | 170 | Titans |
| 10 | 10 | Ronald Darby | 81.96 | 74.90 | 83.54 | 911 | Bills |
| 11 | 11 | Casey Hayward Jr. | 81.88 | 77.50 | 83.34 | 1038 | Packers |
| 12 | 12 | Quinten Rollins | 81.81 | 75.70 | 83.80 | 355 | Packers |
| 13 | 13 | Dominique Rodgers-Cromartie | 81.78 | 76.90 | 81.39 | 890 | Giants |
| 14 | 14 | Marcus Williams | 81.27 | 79.60 | 83.30 | 283 | Jets |
| 15 | 15 | Trumaine Johnson | 81.13 | 78.80 | 81.75 | 901 | Rams |
| 16 | 16 | Adam Jones | 80.62 | 77.80 | 78.85 | 923 | Bengals |
| 17 | 17 | Aqib Talib | 80.39 | 73.40 | 81.08 | 1186 | Broncos |
| 18 | 18 | Brandon Boykin | 80.28 | 74.40 | 82.63 | 397 | Steelers |
| 19 | 19 | Captain Munnerlyn | 80.27 | 76.40 | 78.69 | 745 | Vikings |
| 20 | 20 | David Amerson | 80.11 | 76.00 | 79.51 | 882 | Raiders |
| 21 | 21 | Malcolm Butler | 80.02 | 73.30 | 83.06 | 1234 | Patriots |
| 22 | 22 | Sam Shields | 80.00 | 75.00 | 80.94 | 756 | Packers |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Logan Ryan | 79.91 | 70.20 | 82.22 | 1129 | Patriots |
| 24 | 2 | Byron Jones | 79.45 | 72.30 | 80.05 | 872 | Cowboys |
| 25 | 3 | Sean Smith | 79.07 | 71.40 | 80.54 | 948 | Chiefs |
| 26 | 4 | Bradley Roby | 79.07 | 75.40 | 77.35 | 778 | Broncos |
| 27 | 5 | Vontae Davis | 78.61 | 69.70 | 80.38 | 1046 | Colts |
| 28 | 6 | Terrance Mitchell | 78.47 | 78.90 | 85.68 | 131 | Cowboys |
| 29 | 7 | Bashaud Breeland | 78.44 | 70.40 | 79.63 | 983 | Commanders |
| 30 | 8 | Desmond Trufant | 78.34 | 67.60 | 81.34 | 963 | Falcons |
| 31 | 9 | Janoris Jenkins | 77.52 | 72.30 | 77.98 | 1035 | Rams |
| 32 | 10 | Ross Cockrell | 77.45 | 73.50 | 81.38 | 719 | Steelers |
| 33 | 11 | Leon Hall | 76.60 | 71.60 | 78.59 | 721 | Bengals |
| 34 | 12 | Darrelle Revis | 76.20 | 66.30 | 79.66 | 884 | Jets |
| 35 | 13 | Marcus Burley | 75.51 | 73.80 | 77.68 | 203 | Seahawks |
| 36 | 14 | Davon House | 75.10 | 66.60 | 77.85 | 1035 | Jaguars |
| 37 | 15 | Kyle Fuller | 74.72 | 70.10 | 73.64 | 1021 | Bears |
| 38 | 16 | Stephon Gilmore | 74.52 | 68.60 | 78.05 | 788 | Bills |
| 39 | 17 | Trae Waynes | 74.08 | 67.90 | 79.24 | 214 | Vikings |
| 40 | 18 | Patrick Robinson | 74.04 | 67.40 | 77.85 | 691 | Chargers |

### Starter (65 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 41 | 1 | Tramaine Brock Sr. | 73.96 | 68.20 | 78.22 | 1065 | 49ers |
| 42 | 2 | Prince Amukamara | 73.52 | 70.00 | 76.80 | 766 | Giants |
| 43 | 3 | William Gay | 73.45 | 66.50 | 73.91 | 1207 | Steelers |
| 44 | 4 | K'Waun Williams | 73.33 | 66.10 | 77.12 | 514 | Browns |
| 45 | 5 | Tramon Williams | 73.14 | 62.70 | 76.45 | 964 | Browns |
| 46 | 6 | Brent Grimes | 72.95 | 63.90 | 75.34 | 957 | Dolphins |
| 47 | 7 | Terence Newman | 72.66 | 64.90 | 74.91 | 994 | Vikings |
| 48 | 8 | Lardarius Webb | 72.58 | 63.40 | 75.37 | 879 | Ravens |
| 49 | 9 | Kyle Wilson | 72.50 | 67.20 | 72.70 | 501 | Saints |
| 50 | 10 | Robert Alford | 72.20 | 63.70 | 76.30 | 903 | Falcons |
| 51 | 11 | Quinton Dunbar | 71.85 | 64.20 | 80.08 | 310 | Commanders |
| 52 | 12 | Kareem Jackson | 71.83 | 64.80 | 75.27 | 682 | Texans |
| 53 | 13 | Perrish Cox | 71.80 | 65.10 | 73.99 | 700 | Titans |
| 54 | 14 | Darqueze Dennard | 71.34 | 69.00 | 75.77 | 188 | Bengals |
| 55 | 15 | Dontae Johnson | 71.28 | 65.20 | 74.95 | 366 | 49ers |
| 56 | 16 | Marcus Peters | 70.93 | 58.10 | 75.31 | 1152 | Chiefs |
| 57 | 17 | Xavier Rhodes | 70.53 | 59.10 | 74.61 | 1070 | Vikings |
| 58 | 18 | Jeremy Lane | 70.18 | 67.60 | 74.82 | 351 | Seahawks |
| 59 | 19 | Justin Bethel | 69.98 | 62.30 | 75.94 | 540 | Cardinals |
| 60 | 20 | Charles James II | 69.46 | 68.50 | 77.26 | 151 | Texans |
| 61 | 21 | Shareece Wright | 69.34 | 62.40 | 73.24 | 487 | Ravens |
| 62 | 22 | DeAngelo Hall | 69.26 | 61.50 | 76.42 | 499 | Commanders |
| 63 | 23 | Nolan Carroll | 69.19 | 61.20 | 73.16 | 745 | Eagles |
| 64 | 24 | Josh Shaw | 69.15 | 73.00 | 69.72 | 117 | Bengals |
| 65 | 25 | Jimmy Smith | 68.58 | 60.60 | 72.23 | 991 | Ravens |
| 66 | 26 | Rashean Mathis | 68.50 | 62.50 | 73.23 | 434 | Lions |
| 67 | 27 | Cortland Finnegan | 68.22 | 65.40 | 73.24 | 354 | Panthers |
| 68 | 28 | Marcus Roberson | 68.16 | 64.60 | 75.61 | 328 | Rams |
| 69 | 29 | Sterling Moore | 68.14 | 61.90 | 70.84 | 704 | Buccaneers |
| 70 | 30 | Trevin Wade | 68.04 | 64.80 | 71.66 | 529 | Giants |
| 71 | 31 | Kevin Johnson | 67.99 | 59.00 | 69.82 | 823 | Texans |
| 72 | 32 | Brandon Carr | 67.78 | 56.00 | 71.47 | 1053 | Cowboys |
| 73 | 33 | Byron Maxwell | 67.51 | 54.40 | 73.44 | 898 | Eagles |
| 74 | 34 | Tony Lippett | 67.28 | 69.90 | 77.33 | 132 | Dolphins |
| 75 | 35 | Darrin Walls | 67.16 | 59.70 | 75.05 | 121 | Jets |
| 76 | 36 | Keenan Lewis | 67.14 | 59.40 | 73.33 | 106 | Saints |
| 77 | 37 | Brice McCain | 67.13 | 58.40 | 70.45 | 706 | Dolphins |
| 78 | 38 | Tarell Brown | 67.01 | 60.80 | 74.38 | 162 | Patriots |
| 79 | 39 | A.J. Bouye | 66.72 | 59.10 | 72.95 | 208 | Texans |
| 80 | 40 | Alan Ball | 66.33 | 55.90 | 72.67 | 248 | Bears |
| 81 | 41 | Justin Coleman | 66.11 | 57.50 | 72.89 | 408 | Patriots |
| 82 | 42 | Kenneth Acker | 66.10 | 59.10 | 68.69 | 806 | 49ers |
| 83 | 43 | Phillip Adams | 65.98 | 59.10 | 70.88 | 426 | Falcons |
| 84 | 44 | Alterraun Verner | 65.94 | 53.40 | 71.28 | 575 | Buccaneers |
| 85 | 45 | Mario Butler | 65.72 | 59.90 | 78.85 | 126 | Bills |
| 86 | 46 | Corey White | 65.70 | 62.70 | 69.58 | 113 | Cardinals |
| 87 | 47 | Leodis McKelvin | 65.60 | 59.00 | 71.56 | 388 | Bills |
| 88 | 48 | Bryce Callahan | 65.38 | 62.30 | 71.60 | 322 | Bears |
| 89 | 49 | Jerraud Powers | 65.30 | 55.00 | 68.52 | 906 | Cardinals |
| 90 | 50 | Aaron Colvin | 65.01 | 61.10 | 67.35 | 1069 | Jaguars |
| 91 | 51 | Bobby McCain | 64.83 | 59.80 | 70.26 | 390 | Dolphins |
| 92 | 52 | Charles Tillman | 64.75 | 58.10 | 73.15 | 711 | Panthers |
| 93 | 53 | Bene Benwikere | 64.54 | 56.40 | 69.32 | 785 | Panthers |
| 94 | 54 | Antonio Cromartie | 64.22 | 52.20 | 68.58 | 898 | Jets |
| 95 | 55 | T.J. Carrie | 64.17 | 54.60 | 68.60 | 928 | Raiders |
| 96 | 56 | Eric Rowe | 63.94 | 56.90 | 70.72 | 504 | Eagles |
| 97 | 57 | Kyle Arrington | 63.75 | 50.50 | 68.93 | 333 | Ravens |
| 98 | 58 | Demetrius McCray | 63.70 | 57.40 | 67.70 | 224 | Jaguars |
| 99 | 59 | Marcus Cromartie | 63.54 | 58.50 | 72.53 | 122 | 49ers |
| 100 | 60 | Buster Skrine | 63.52 | 50.60 | 67.96 | 720 | Jets |
| 101 | 61 | Will Blackmon | 63.26 | 54.00 | 67.97 | 861 | Commanders |
| 102 | 62 | Tracy Porter | 62.54 | 54.20 | 69.03 | 845 | Bears |
| 103 | 63 | Coty Sensabaugh | 62.52 | 53.40 | 65.78 | 1003 | Titans |
| 104 | 64 | Phillip Gaines | 62.35 | 61.70 | 70.20 | 167 | Chiefs |
| 105 | 65 | Nickell Robey-Coleman | 62.28 | 48.70 | 67.16 | 668 | Bills |

### Rotation/backup (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 106 | 1 | Damian Swann | 61.90 | 62.50 | 68.20 | 228 | Saints |
| 107 | 2 | E.J. Biggers | 61.34 | 48.70 | 66.23 | 564 | Eagles |
| 108 | 3 | Dre Kirkpatrick | 61.30 | 49.50 | 66.04 | 1089 | Bengals |
| 109 | 4 | Dwayne Gratz | 61.16 | 52.40 | 66.48 | 454 | Jaguars |
| 110 | 5 | Robert McClain | 60.47 | 54.90 | 66.88 | 220 | Panthers |
| 111 | 6 | Neiko Thorpe | 59.92 | 52.60 | 66.56 | 454 | Raiders |
| 112 | 7 | Craig Mager | 59.44 | 63.80 | 65.78 | 226 | Chargers |
| 113 | 8 | Steve Williams | 58.77 | 48.20 | 67.64 | 287 | Chargers |
| 114 | 9 | Trumaine McBride | 58.77 | 48.00 | 66.78 | 337 | Giants |
| 115 | 10 | Josh Wilson | 58.75 | 45.70 | 68.39 | 276 | Lions |
| 116 | 11 | Valentino Blake | 58.47 | 47.70 | 65.55 | 1027 | Steelers |
| 117 | 12 | Morris Claiborne | 58.03 | 49.40 | 67.22 | 661 | Cowboys |
| 118 | 13 | Greg Toler | 57.97 | 40.20 | 70.02 | 685 | Colts |
| 119 | 14 | Blidi Wreh-Wilson | 57.95 | 50.60 | 65.77 | 293 | Titans |
| 120 | 15 | Leonard Johnson | 57.80 | 48.30 | 66.21 | 158 | Patriots |
| 121 | 16 | Johnthan Banks | 57.59 | 46.60 | 63.66 | 431 | Buccaneers |
| 122 | 17 | Mike Jenkins | 57.52 | 49.50 | 65.69 | 330 | Buccaneers |
| 123 | 18 | Jason McCourty | 57.50 | 45.10 | 67.85 | 212 | Titans |
| 124 | 19 | Jayron Hosley | 56.87 | 55.90 | 59.60 | 525 | Giants |
| 125 | 20 | Pierre Desir | 56.81 | 42.20 | 70.07 | 389 | Browns |
| 126 | 21 | Tyler Patmon | 56.05 | 45.00 | 65.50 | 305 | Dolphins |
| 127 | 22 | Nevin Lawson | 55.84 | 50.10 | 64.22 | 561 | Lions |
| 128 | 23 | Jalil Brown | 55.37 | 52.00 | 65.12 | 308 | Colts |
| 129 | 24 | Jalen Collins | 54.38 | 47.90 | 60.78 | 300 | Falcons |
| 130 | 25 | Jude Adjei-Barimah | 54.03 | 53.50 | 53.35 | 468 | Buccaneers |
| 131 | 26 | Brandon Flowers | 53.61 | 36.50 | 64.50 | 602 | Chargers |
| 132 | 27 | D.J. Hayden | 53.13 | 37.90 | 62.66 | 897 | Raiders |
| 133 | 28 | Brandon Browner | 52.19 | 27.50 | 67.40 | 1023 | Saints |
| 134 | 29 | Deji Olatoye | 52.07 | 48.80 | 70.09 | 138 | Cowboys |
| 135 | 30 | Chris Culliver | 51.67 | 32.70 | 65.98 | 350 | Commanders |
| 136 | 31 | Joe Haden | 51.23 | 30.00 | 67.46 | 284 | Browns |
| 137 | 32 | Nick Marshall | 48.48 | 46.30 | 59.19 | 142 | Jaguars |
| 138 | 33 | Johnson Bademosi | 48.34 | 29.90 | 63.96 | 167 | Browns |
| 139 | 34 | Sherrick McManis | 48.33 | 40.00 | 59.51 | 295 | Bears |
| 140 | 35 | Brian Dixon | 47.39 | 47.90 | 49.91 | 111 | Saints |
| 141 | 36 | Charles Gaines | 47.04 | 49.40 | 54.71 | 265 | Browns |
| 142 | 37 | B.W. Webb | 47.03 | 39.50 | 57.15 | 257 | Titans |
| 143 | 38 | Jamar Taylor | 46.28 | 40.40 | 52.08 | 712 | Dolphins |

## DI — Defensive Interior

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (25 players)

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
| 9 | 9 | Leonard Williams | 86.18 | 86.82 | 81.58 | 809 | Jets |
| 10 | 10 | Sheldon Richardson | 85.57 | 87.61 | 82.64 | 619 | Jets |
| 11 | 11 | Marcell Dareus | 85.31 | 88.26 | 80.01 | 755 | Bills |
| 12 | 12 | Fletcher Cox | 85.09 | 89.08 | 78.26 | 978 | Eagles |
| 13 | 13 | Malik Jackson | 85.06 | 83.93 | 81.64 | 988 | Broncos |
| 14 | 14 | Mike Daniels | 85.05 | 81.73 | 83.09 | 799 | Packers |
| 15 | 15 | Damon Harrison Sr. | 83.47 | 81.26 | 80.78 | 566 | Jets |
| 16 | 16 | Nick Fairley | 82.88 | 84.23 | 81.05 | 420 | Rams |
| 17 | 17 | Grady Jarrett | 82.40 | 77.23 | 82.72 | 266 | Falcons |
| 18 | 18 | Cameron Heyward | 82.31 | 83.39 | 77.42 | 1103 | Steelers |
| 19 | 19 | Dan Williams | 81.75 | 85.73 | 75.34 | 568 | Raiders |
| 20 | 20 | Chris Baker | 81.46 | 74.58 | 82.40 | 671 | Commanders |
| 21 | 21 | Brandon Williams | 80.97 | 78.99 | 80.00 | 723 | Ravens |
| 22 | 22 | Gerald McCoy | 80.70 | 86.26 | 74.30 | 795 | Buccaneers |
| 23 | 23 | Linval Joseph | 80.63 | 81.75 | 77.48 | 568 | Vikings |
| 24 | 24 | Arik Armstead | 80.61 | 88.08 | 71.47 | 376 | 49ers |
| 25 | 25 | Mario Edwards Jr. | 80.60 | 83.27 | 76.73 | 597 | Raiders |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Eddie Goldman | 79.88 | 77.27 | 78.49 | 517 | Bears |
| 27 | 2 | Derek Wolfe | 79.22 | 81.03 | 75.41 | 808 | Broncos |
| 28 | 3 | Timmy Jernigan | 78.84 | 74.14 | 79.62 | 529 | Ravens |
| 29 | 4 | Henry Anderson | 78.80 | 77.97 | 82.48 | 447 | Colts |
| 30 | 5 | Bennie Logan | 78.47 | 69.92 | 81.04 | 578 | Eagles |
| 31 | 6 | Johnathan Hankins | 78.25 | 79.73 | 77.78 | 409 | Giants |
| 32 | 7 | Stephen Paea | 78.11 | 81.03 | 75.23 | 214 | Commanders |
| 33 | 8 | Christian Covington | 78.07 | 65.15 | 82.52 | 193 | Texans |
| 34 | 9 | Ian Williams | 77.47 | 75.28 | 79.86 | 660 | 49ers |
| 35 | 10 | Malcom Brown | 77.30 | 70.18 | 77.88 | 596 | Patriots |
| 36 | 11 | Lawrence Guy Sr. | 76.93 | 66.52 | 80.12 | 478 | Ravens |
| 37 | 12 | Sharrif Floyd | 76.88 | 73.87 | 76.38 | 572 | Vikings |
| 38 | 13 | Akiem Hicks | 76.25 | 67.62 | 78.16 | 471 | Patriots |
| 39 | 14 | Haloti Ngata | 76.19 | 75.98 | 74.05 | 595 | Lions |
| 40 | 15 | Karl Klug | 75.93 | 69.01 | 76.38 | 329 | Titans |
| 41 | 16 | Vance Walker | 75.85 | 68.35 | 76.88 | 438 | Broncos |
| 42 | 17 | Desmond Bryant | 75.34 | 68.76 | 77.75 | 531 | Browns |
| 43 | 18 | Star Lotulelei | 74.93 | 62.73 | 79.22 | 630 | Panthers |
| 44 | 19 | Dominique Easley | 74.88 | 81.12 | 71.76 | 272 | Patriots |
| 45 | 20 | Jaye Howard Jr. | 74.71 | 59.30 | 83.32 | 827 | Chiefs |
| 46 | 21 | Michael Brockers | 74.64 | 71.28 | 72.72 | 690 | Rams |
| 47 | 22 | Jason Hatcher | 74.36 | 63.61 | 78.51 | 576 | Commanders |

### Starter (79 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 48 | 1 | Datone Jones | 73.71 | 63.39 | 76.74 | 421 | Packers |
| 49 | 2 | Zach Kerr | 72.99 | 71.36 | 74.07 | 320 | Colts |
| 50 | 3 | Kyle Williams | 72.89 | 62.36 | 81.26 | 341 | Bills |
| 51 | 4 | Terrance Knighton | 72.59 | 65.24 | 73.32 | 403 | Commanders |
| 52 | 5 | Corey Liuget | 72.02 | 64.45 | 75.50 | 443 | Chargers |
| 53 | 6 | Tyson Jackson | 71.97 | 67.45 | 70.82 | 462 | Falcons |
| 54 | 7 | Paul Soliai | 71.97 | 61.23 | 76.53 | 357 | Falcons |
| 55 | 8 | Mike Pennel | 71.91 | 59.37 | 76.49 | 314 | Packers |
| 56 | 9 | Ra'Shede Hageman | 71.74 | 58.10 | 76.66 | 419 | Falcons |
| 57 | 10 | David Irving | 71.53 | 64.51 | 76.21 | 199 | Cowboys |
| 58 | 11 | Dontari Poe | 71.50 | 68.79 | 69.14 | 852 | Chiefs |
| 59 | 12 | Tyrone Crawford | 71.44 | 63.25 | 72.73 | 705 | Cowboys |
| 60 | 13 | Leger Douzable | 71.38 | 56.21 | 77.32 | 289 | Jets |
| 61 | 14 | Jordan Hill | 71.19 | 56.05 | 82.64 | 344 | Seahawks |
| 62 | 15 | Cedric Thornton | 70.69 | 60.42 | 74.94 | 488 | Eagles |
| 63 | 16 | Steve McLendon | 70.38 | 59.64 | 74.72 | 408 | Steelers |
| 64 | 17 | Jared Odrick | 70.28 | 64.47 | 69.98 | 874 | Jaguars |
| 65 | 18 | Danny Shelton | 69.59 | 65.36 | 68.24 | 505 | Browns |
| 66 | 19 | John Jenkins | 69.53 | 61.75 | 72.83 | 530 | Saints |
| 67 | 20 | Roy Miller | 69.49 | 66.52 | 68.33 | 556 | Jaguars |
| 68 | 21 | Allen Bailey | 69.42 | 62.57 | 71.49 | 655 | Chiefs |
| 69 | 22 | Mike Devito | 69.30 | 62.39 | 75.16 | 317 | Chiefs |
| 70 | 23 | Clinton McDonald | 69.17 | 61.83 | 76.05 | 245 | Buccaneers |
| 71 | 24 | Randy Starks | 69.05 | 56.51 | 74.08 | 466 | Browns |
| 72 | 25 | Sammie Lee Hill | 68.80 | 59.72 | 74.76 | 188 | Titans |
| 73 | 26 | Stephon Tuitt | 68.77 | 59.48 | 70.79 | 1005 | Steelers |
| 74 | 27 | Sealver Siliga | 68.71 | 55.97 | 77.31 | 280 | Patriots |
| 75 | 28 | Jamie Meder | 68.53 | 66.03 | 71.89 | 387 | Browns |
| 76 | 29 | Ricky Jean Francois | 68.35 | 56.05 | 73.22 | 419 | Commanders |
| 77 | 30 | Kendall Langford | 68.24 | 59.75 | 69.73 | 847 | Colts |
| 78 | 31 | Chris Canty | 68.22 | 61.10 | 73.58 | 281 | Ravens |
| 79 | 32 | Brandon Mebane | 68.07 | 55.76 | 74.29 | 573 | Seahawks |
| 80 | 33 | Alan Branch | 67.99 | 58.13 | 71.97 | 505 | Patriots |
| 81 | 34 | Quinton Dial | 67.94 | 58.83 | 73.69 | 641 | 49ers |
| 82 | 35 | Frostee Rucker | 67.80 | 46.02 | 78.67 | 587 | Cardinals |
| 83 | 36 | Sylvester Williams | 67.79 | 57.56 | 70.45 | 620 | Broncos |
| 84 | 37 | Shelby Harris | 67.56 | 68.20 | 74.68 | 144 | Raiders |
| 85 | 38 | Pat Sims | 67.51 | 55.52 | 74.98 | 210 | Bengals |
| 86 | 39 | Henry Melton | 67.27 | 50.69 | 76.85 | 508 | Buccaneers |
| 87 | 40 | Justin Ellis | 67.17 | 71.91 | 62.44 | 361 | Raiders |
| 88 | 41 | John Hughes | 67.16 | 62.14 | 69.99 | 428 | Browns |
| 89 | 42 | Tom Johnson | 67.06 | 49.88 | 74.76 | 779 | Vikings |
| 90 | 43 | Jonathan Babineaux | 67.04 | 54.76 | 71.37 | 545 | Falcons |
| 91 | 44 | Abry Jones | 67.03 | 52.87 | 74.48 | 363 | Jaguars |
| 92 | 45 | Tony McDaniel | 66.90 | 51.06 | 74.33 | 293 | Buccaneers |
| 93 | 46 | C.J. Wilson | 66.81 | 53.86 | 74.83 | 225 | Lions |
| 94 | 47 | Angelo Blackson | 66.64 | 55.31 | 70.02 | 244 | Titans |
| 95 | 48 | Al Woods | 66.57 | 52.41 | 73.10 | 356 | Titans |
| 96 | 49 | Rodney Gunter | 66.50 | 55.28 | 69.82 | 471 | Cardinals |
| 97 | 50 | DaQuan Jones | 66.33 | 63.44 | 67.61 | 670 | Titans |
| 98 | 51 | Ahtyba Rubin | 66.18 | 54.00 | 71.48 | 550 | Seahawks |
| 99 | 52 | Letroy Guion | 66.00 | 55.16 | 70.21 | 371 | Packers |
| 100 | 53 | Earl Mitchell | 65.94 | 51.90 | 73.21 | 501 | Dolphins |
| 101 | 54 | Stefan Charles | 65.84 | 51.08 | 74.74 | 231 | Bills |
| 102 | 55 | Sen'Derrick Marks | 65.81 | 58.36 | 72.86 | 144 | Jaguars |
| 103 | 56 | Vince Wilfork | 65.69 | 54.14 | 71.72 | 591 | Texans |
| 104 | 57 | Sean Lissemore | 65.63 | 51.16 | 74.25 | 235 | Chargers |
| 105 | 58 | Tyrunn Walker | 65.02 | 58.86 | 73.10 | 176 | Lions |
| 106 | 59 | Cory Redding | 64.98 | 46.91 | 74.94 | 180 | Cardinals |
| 107 | 60 | Kevin Williams | 64.84 | 51.65 | 69.66 | 553 | Saints |
| 108 | 61 | David Parry | 64.81 | 52.83 | 68.63 | 657 | Colts |
| 109 | 62 | T.Y. McGill | 64.73 | 58.71 | 68.75 | 223 | Colts |
| 110 | 63 | Cullen Jenkins | 64.56 | 47.29 | 73.15 | 731 | Giants |
| 111 | 64 | Domata Peko Sr. | 64.48 | 48.39 | 71.04 | 562 | Bengals |
| 112 | 65 | Jared Crick | 64.37 | 51.82 | 68.57 | 836 | Texans |
| 113 | 66 | Xavier Cooper | 64.34 | 49.09 | 72.42 | 361 | Browns |
| 114 | 67 | Antonio Smith | 64.20 | 45.51 | 72.69 | 423 | Broncos |
| 115 | 68 | Ryan Carrethers | 63.96 | 54.84 | 71.08 | 219 | Chargers |
| 116 | 69 | Tyson Alualu | 63.92 | 52.68 | 67.25 | 688 | Jaguars |
| 117 | 70 | Cam Thomas | 63.90 | 48.89 | 69.74 | 198 | Steelers |
| 118 | 71 | Beau Allen | 63.85 | 54.01 | 66.25 | 341 | Eagles |
| 119 | 72 | Demarcus Dobbs | 63.71 | 48.84 | 73.52 | 142 | Seahawks |
| 120 | 73 | Billy Winn | 63.28 | 50.61 | 71.62 | 325 | Colts |
| 121 | 74 | Brandon Dunn | 62.93 | 62.83 | 65.87 | 139 | Texans |
| 122 | 75 | Jack Crawford | 62.84 | 41.50 | 72.90 | 487 | Cowboys |
| 123 | 76 | Brandon Thompson | 62.64 | 49.04 | 72.44 | 181 | Bengals |
| 124 | 77 | Denico Autry | 62.52 | 48.77 | 69.61 | 681 | Raiders |
| 125 | 78 | Stephen Bowen | 62.50 | 52.94 | 68.97 | 139 | Jets |
| 126 | 79 | Akeem Spence | 62.19 | 55.44 | 66.69 | 289 | Buccaneers |

### Rotation/backup (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 127 | 1 | Corbin Bryant | 61.91 | 50.38 | 65.63 | 633 | Bills |
| 128 | 2 | Red Bryant | 61.71 | 52.23 | 68.54 | 132 | Cardinals |
| 129 | 3 | Tony Jerod-Eddie | 61.69 | 47.56 | 66.95 | 290 | 49ers |
| 130 | 4 | Ego Ferguson | 61.24 | 53.67 | 69.94 | 105 | Bears |
| 131 | 5 | Bruce Gaston | 60.90 | 50.05 | 71.27 | 179 | Bears |
| 132 | 6 | Jarvis Jenkins | 60.43 | 47.81 | 66.03 | 635 | Bears |
| 133 | 7 | B.J. Raji | 60.39 | 48.22 | 64.33 | 492 | Packers |
| 134 | 8 | Caraun Reid | 60.08 | 50.34 | 65.27 | 530 | Lions |
| 135 | 9 | Kendall Reyes | 59.96 | 46.10 | 65.03 | 656 | Chargers |
| 136 | 10 | Jay Bromley | 59.78 | 55.97 | 61.28 | 477 | Giants |
| 137 | 11 | Ed Stinson | 59.74 | 47.57 | 66.03 | 444 | Cardinals |
| 138 | 12 | Will Sutton III | 59.20 | 50.56 | 63.15 | 417 | Bears |
| 139 | 13 | Dwan Edwards | 59.10 | 35.86 | 71.78 | 447 | Panthers |
| 140 | 14 | Dan McCullers | 59.05 | 66.90 | 53.95 | 110 | Steelers |
| 141 | 15 | Glenn Dorsey | 59.03 | 43.08 | 71.13 | 351 | 49ers |
| 142 | 16 | Darius Philon | 59.03 | 45.29 | 72.36 | 145 | Chargers |
| 143 | 17 | Stacy McGee | 58.93 | 49.40 | 62.88 | 408 | Raiders |
| 144 | 18 | Alex Carrington | 58.60 | 52.09 | 68.67 | 172 | Bills |
| 145 | 19 | Montori Hughes | 58.59 | 47.66 | 69.21 | 118 | Giants |
| 146 | 20 | Mitch Unrein | 58.57 | 50.27 | 63.58 | 380 | Bears |
| 147 | 21 | Ricardo Mathews | 58.16 | 46.52 | 63.00 | 510 | Chargers |
| 148 | 22 | Nick Hayden | 57.72 | 41.92 | 64.09 | 579 | Cowboys |
| 149 | 23 | Kyle Love | 57.55 | 45.07 | 69.31 | 387 | Panthers |
| 150 | 24 | Tyeler Davison | 57.50 | 51.21 | 57.52 | 528 | Saints |
| 151 | 25 | Jordan Phillips | 57.42 | 46.35 | 61.67 | 429 | Dolphins |
| 152 | 26 | Gabe Wright | 57.17 | 51.78 | 67.46 | 135 | Lions |
| 153 | 27 | Justin Tuck | 56.37 | 42.38 | 77.49 | 244 | Raiders |
| 154 | 28 | Mike Purcell | 55.74 | 52.69 | 64.29 | 289 | 49ers |
| 155 | 29 | Jermelle Cudjo | 55.52 | 43.90 | 64.93 | 127 | Lions |
| 156 | 30 | Geneo Grissom | 55.23 | 48.09 | 63.12 | 131 | Patriots |
| 157 | 31 | Markus Kuhn | 55.13 | 43.97 | 65.27 | 310 | Giants |
| 158 | 32 | Darius Kilgo | 54.97 | 51.74 | 60.25 | 113 | Broncos |
| 159 | 33 | Kedric Golston | 53.48 | 37.95 | 61.23 | 205 | Commanders |
| 160 | 34 | Carl Davis Jr. | 51.71 | 45.82 | 54.61 | 235 | Ravens |
| 161 | 35 | Michael Bennett | 51.70 | 33.00 | 63.14 | 294 | Jaguars |
| 162 | 36 | Khyri Thornton | 47.90 | 43.21 | 60.27 | 102 | Lions |
| 163 | 37 | Damion Square | 45.69 | 41.80 | 52.59 | 155 | Chargers |

## ED — Edge

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 94.41 | 97.44 | 89.69 | 994 | Broncos |
| 2 | 2 | Justin Houston | 91.70 | 91.84 | 89.84 | 669 | Chiefs |
| 3 | 3 | Pernell McPhee | 90.56 | 94.21 | 85.26 | 593 | Bears |
| 4 | 4 | Ezekiel Ansah | 89.80 | 92.14 | 84.49 | 656 | Lions |
| 5 | 5 | Khalil Mack | 89.59 | 97.24 | 80.32 | 999 | Raiders |
| 6 | 6 | Whitney Mercilus | 87.19 | 89.07 | 82.15 | 796 | Texans |
| 7 | 7 | Brandon Graham | 86.03 | 85.63 | 82.13 | 849 | Eagles |
| 8 | 8 | Michael Bennett | 84.29 | 91.51 | 75.31 | 922 | Seahawks |
| 9 | 9 | DeMarcus Ware | 84.24 | 77.33 | 86.34 | 553 | Broncos |
| 10 | 10 | Robert Quinn | 83.62 | 89.63 | 79.61 | 334 | Rams |
| 11 | 11 | Carlos Dunlap | 81.86 | 81.51 | 77.93 | 945 | Bengals |
| 12 | 12 | Cliff Avril | 81.76 | 77.23 | 80.61 | 861 | Seahawks |
| 13 | 13 | Charles Johnson | 81.40 | 76.60 | 82.71 | 528 | Panthers |
| 14 | 14 | Jabaal Sheard | 80.75 | 92.43 | 69.94 | 665 | Patriots |
| 15 | 15 | Cameron Wake | 80.47 | 72.03 | 86.83 | 246 | Dolphins |
| 16 | 16 | Olivier Vernon | 80.35 | 85.93 | 72.46 | 943 | Dolphins |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | James Harrison | 79.73 | 66.34 | 86.06 | 698 | Steelers |
| 18 | 2 | Danielle Hunter | 79.28 | 77.14 | 77.58 | 420 | Vikings |
| 19 | 3 | Everson Griffen | 78.99 | 78.52 | 75.13 | 913 | Vikings |
| 20 | 4 | William Hayes | 78.65 | 79.93 | 74.05 | 579 | Rams |
| 21 | 5 | Elvis Dumervil | 78.48 | 62.57 | 84.92 | 786 | Ravens |
| 22 | 6 | Shaquil Barrett | 78.36 | 71.32 | 78.88 | 552 | Broncos |
| 23 | 7 | Mario Williams | 77.81 | 72.22 | 77.89 | 880 | Bills |
| 24 | 8 | Chandler Jones | 77.60 | 78.50 | 73.76 | 943 | Patriots |
| 25 | 9 | Jerry Hughes | 77.56 | 74.48 | 75.45 | 1003 | Bills |
| 26 | 10 | Markus Golden | 77.49 | 69.63 | 78.57 | 633 | Cardinals |
| 27 | 11 | Willie Young | 76.83 | 69.92 | 78.30 | 525 | Bears |
| 28 | 12 | Cameron Jordan | 76.24 | 80.36 | 69.33 | 979 | Saints |
| 29 | 13 | Aaron Lynch | 75.48 | 66.11 | 78.86 | 794 | 49ers |
| 30 | 14 | Connor Barwin | 74.84 | 58.33 | 81.68 | 1043 | Eagles |
| 31 | 15 | Robert Ayers | 74.26 | 76.44 | 71.97 | 570 | Giants |
| 32 | 16 | Jeremiah Attaochu | 74.22 | 69.87 | 75.55 | 666 | Chargers |
| 33 | 17 | Ryan Kerrigan | 74.19 | 61.44 | 78.52 | 959 | Commanders |
| 34 | 18 | Greg Hardy | 74.08 | 72.60 | 77.67 | 596 | Cowboys |

### Starter (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | DeMarcus Lawrence | 73.97 | 77.01 | 70.51 | 698 | Cowboys |
| 36 | 2 | Jason Pierre-Paul | 73.62 | 80.34 | 70.18 | 503 | Giants |
| 37 | 3 | Vinny Curry | 73.48 | 63.40 | 76.23 | 424 | Eagles |
| 38 | 4 | Marcus Smith | 72.70 | 56.50 | 83.24 | 127 | Eagles |
| 39 | 5 | Nick Perry | 72.64 | 65.19 | 73.44 | 402 | Packers |
| 40 | 6 | Julius Peppers | 72.55 | 61.58 | 75.70 | 779 | Packers |
| 41 | 7 | Lorenzo Mauldin IV | 72.40 | 59.88 | 78.67 | 253 | Jets |
| 42 | 8 | Paul Kruger | 72.35 | 60.69 | 75.96 | 679 | Browns |
| 43 | 9 | Preston Smith | 72.28 | 62.26 | 74.80 | 563 | Commanders |
| 44 | 10 | Melvin Ingram III | 72.26 | 67.23 | 75.72 | 961 | Chargers |
| 45 | 11 | Arthur Moats | 71.15 | 59.45 | 74.78 | 594 | Steelers |
| 46 | 12 | Lamarr Houston | 70.84 | 65.93 | 73.08 | 416 | Bears |
| 47 | 13 | Tamba Hali | 70.42 | 63.81 | 70.66 | 905 | Chiefs |
| 48 | 14 | Trent Cole | 69.21 | 60.24 | 72.72 | 527 | Colts |
| 49 | 15 | Mario Addison | 69.18 | 59.13 | 71.71 | 447 | Panthers |
| 50 | 16 | Derrick Morgan | 68.92 | 60.10 | 74.53 | 524 | Titans |
| 51 | 17 | Vic Beasley Jr. | 68.91 | 69.79 | 64.15 | 534 | Falcons |
| 52 | 18 | Jared Allen | 68.88 | 53.93 | 75.00 | 730 | Panthers |
| 53 | 19 | Shane Ray | 68.45 | 59.08 | 70.53 | 403 | Broncos |
| 54 | 20 | Damontre Moore | 68.28 | 61.45 | 70.34 | 277 | Dolphins |
| 55 | 21 | Ryan Davis Sr. | 67.90 | 62.64 | 74.23 | 244 | Jaguars |
| 56 | 22 | Frank Clark | 67.68 | 62.09 | 67.24 | 355 | Seahawks |
| 57 | 23 | Quinton Coples | 67.09 | 60.34 | 68.08 | 286 | Dolphins |
| 58 | 24 | Armonty Bryant | 66.98 | 54.11 | 72.43 | 478 | Browns |
| 59 | 25 | Erik Walden | 66.92 | 54.48 | 71.70 | 788 | Colts |
| 60 | 26 | Devin Taylor | 66.74 | 60.52 | 67.65 | 550 | Lions |
| 61 | 27 | Dee Ford | 66.71 | 58.41 | 70.02 | 552 | Chiefs |
| 62 | 28 | Jayrone Elliott | 66.65 | 59.31 | 73.89 | 174 | Packers |
| 63 | 29 | Jadeveon Clowney | 66.30 | 80.69 | 59.17 | 559 | Texans |
| 64 | 30 | Chris Long | 66.28 | 53.74 | 75.67 | 481 | Rams |
| 65 | 31 | John Simon | 66.26 | 61.78 | 70.08 | 683 | Texans |
| 66 | 32 | Ahmad Brooks | 66.15 | 49.88 | 75.30 | 740 | 49ers |
| 67 | 33 | Aldon Smith | 66.01 | 69.70 | 67.08 | 1034 | Raiders |
| 68 | 34 | Kony Ealy | 66.00 | 59.31 | 66.29 | 740 | Panthers |
| 69 | 35 | Barkevious Mingo | 65.79 | 61.33 | 64.98 | 255 | Browns |
| 70 | 36 | Za'Darius Smith | 65.69 | 58.79 | 67.16 | 403 | Ravens |
| 71 | 37 | Dwight Freeney | 65.66 | 47.89 | 75.29 | 308 | Cardinals |
| 72 | 38 | Brian Orakpo | 65.41 | 57.48 | 70.04 | 960 | Titans |
| 73 | 39 | Jonathan Newsome | 64.92 | 54.89 | 69.39 | 344 | Colts |
| 74 | 40 | Robert Mathis | 64.60 | 47.23 | 73.05 | 546 | Colts |
| 75 | 41 | Alex Okafor | 64.47 | 57.81 | 70.06 | 605 | Cardinals |
| 76 | 42 | Chris Smith | 64.47 | 61.26 | 72.48 | 155 | Jaguars |
| 77 | 43 | Nate Orchard | 64.45 | 58.11 | 65.55 | 471 | Browns |
| 78 | 44 | Rob Ninkovich | 64.28 | 50.31 | 69.43 | 1018 | Patriots |
| 79 | 45 | Trent Murphy | 64.01 | 60.37 | 62.66 | 688 | Commanders |
| 80 | 46 | O'Brien Schofield | 64.00 | 53.67 | 67.36 | 494 | Falcons |
| 81 | 47 | Ryan Delaire | 63.97 | 55.33 | 71.82 | 245 | Panthers |
| 82 | 48 | William Gholston | 63.48 | 62.76 | 60.95 | 670 | Buccaneers |
| 83 | 49 | Mike Neal | 63.14 | 56.08 | 63.68 | 1642 | Packers |
| 84 | 50 | Brian Robison | 63.10 | 52.52 | 65.99 | 946 | Vikings |
| 85 | 51 | Randy Gregory | 63.08 | 59.79 | 65.27 | 245 | Cowboys |
| 86 | 52 | Sam Acho | 62.87 | 57.27 | 63.09 | 447 | Bears |
| 87 | 53 | Michael Johnson | 62.86 | 58.65 | 62.13 | 902 | Bengals |
| 88 | 54 | Eli Harold | 62.79 | 59.10 | 61.08 | 336 | 49ers |
| 89 | 55 | Lerentee McCray | 62.68 | 60.55 | 64.62 | 118 | Broncos |
| 90 | 56 | Howard Jones | 62.35 | 56.42 | 66.31 | 387 | Buccaneers |

### Rotation/backup (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 91 | 1 | Adrian Clayborn | 61.59 | 59.84 | 63.27 | 519 | Falcons |
| 92 | 2 | Kyle Emanuel | 61.15 | 53.55 | 63.09 | 300 | Chargers |
| 93 | 3 | Chris Clemons | 61.00 | 45.29 | 67.30 | 659 | Jaguars |
| 94 | 4 | Jeremy Mincey | 60.68 | 52.76 | 62.82 | 380 | Cowboys |
| 95 | 5 | Darryl Tapp | 60.57 | 51.51 | 63.48 | 409 | Lions |
| 96 | 6 | Jacquies Smith | 60.29 | 53.49 | 63.65 | 542 | Buccaneers |
| 97 | 7 | Kasim Edebali | 60.28 | 52.78 | 61.89 | 358 | Saints |
| 98 | 8 | David Bass | 59.98 | 56.87 | 61.74 | 530 | Titans |
| 99 | 9 | Jarvis Jones | 59.87 | 56.41 | 61.14 | 522 | Steelers |
| 100 | 10 | Brooks Reed | 59.83 | 53.85 | 61.21 | 345 | Falcons |
| 101 | 11 | Frank Zombo | 59.77 | 51.93 | 68.13 | 249 | Chiefs |
| 102 | 12 | Kroy Biermann | 59.74 | 54.71 | 61.85 | 516 | Falcons |
| 103 | 13 | Ethan Westbrooks | 59.20 | 54.87 | 59.87 | 274 | Rams |
| 104 | 14 | Wallace Gilberry | 58.92 | 46.28 | 63.18 | 667 | Bengals |
| 105 | 15 | Bud Dupree | 58.37 | 54.04 | 57.09 | 653 | Steelers |
| 106 | 16 | Calvin Pace | 58.21 | 44.41 | 63.24 | 519 | Jets |
| 107 | 17 | Bjoern Werner | 57.56 | 57.08 | 57.62 | 150 | Colts |
| 108 | 18 | George Selvie | 57.45 | 47.81 | 61.79 | 370 | Giants |
| 109 | 19 | Andre Branch | 57.35 | 56.06 | 57.79 | 596 | Jaguars |
| 110 | 20 | Albert McClellan | 56.72 | 46.08 | 61.60 | 158 | Ravens |
| 111 | 21 | Matt Longacre | 56.50 | 68.66 | 60.20 | 138 | Rams |
| 112 | 22 | Jason Jones | 56.47 | 52.78 | 58.00 | 543 | Lions |
| 113 | 23 | Kareem Martin | 55.68 | 54.44 | 56.89 | 199 | Cardinals |
| 114 | 24 | Cassius Marsh | 55.08 | 57.67 | 53.48 | 210 | Seahawks |
| 115 | 25 | Kerry Wynn | 54.97 | 57.88 | 53.81 | 577 | Giants |
| 116 | 26 | Terrence Fede | 54.96 | 56.33 | 54.70 | 210 | Dolphins |
| 117 | 27 | Bobby Richardson | 54.84 | 54.33 | 52.04 | 583 | Saints |
| 118 | 28 | Benson Mayowa | 54.68 | 58.89 | 53.03 | 376 | Raiders |
| 119 | 29 | Eugene Sims | 54.34 | 49.88 | 54.71 | 581 | Rams |
| 120 | 30 | Tavaris Barnes | 53.92 | 55.81 | 52.66 | 130 | Saints |
| 121 | 31 | George Johnson | 53.84 | 47.92 | 58.31 | 429 | Buccaneers |
| 122 | 32 | Chris Carter | 52.23 | 48.86 | 54.48 | 116 | Ravens |
| 123 | 33 | Will Clarke | 51.78 | 55.29 | 50.74 | 140 | Bengals |
| 124 | 34 | IK Enemkpali | 51.71 | 51.36 | 51.02 | 146 | Bills |
| 125 | 35 | Scott Crichton | 50.75 | 56.79 | 51.27 | 128 | Vikings |
| 126 | 36 | Malliciah Goodman | 50.55 | 55.28 | 49.90 | 106 | Falcons |
| 127 | 37 | Corey Lemonier | 48.61 | 52.68 | 48.77 | 271 | 49ers |
| 128 | 38 | Lamarr Woodley | 47.99 | 46.04 | 52.94 | 277 | Cardinals |

## G — Guard

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Marshal Yanda | 96.91 | 91.40 | 96.42 | 1130 | Ravens |
| 2 | 2 | Evan Mathis | 94.54 | 90.10 | 93.34 | 988 | Broncos |
| 3 | 3 | T.J. Lang | 93.83 | 89.60 | 92.48 | 1173 | Packers |
| 4 | 4 | Zack Martin | 90.86 | 86.10 | 89.86 | 1020 | Cowboys |
| 5 | 5 | Kelechi Osemele | 90.49 | 81.20 | 92.51 | 974 | Ravens |
| 6 | 6 | David DeCastro | 88.67 | 83.00 | 88.28 | 1211 | Steelers |
| 7 | 7 | Trai Turner | 87.13 | 80.40 | 87.45 | 1277 | Panthers |
| 8 | 8 | Mike Iupati | 86.55 | 80.20 | 86.62 | 938 | Cardinals |
| 9 | 9 | Jeff Allen | 86.51 | 79.70 | 86.88 | 579 | Chiefs |
| 10 | 10 | Andrew Norwell | 86.31 | 81.10 | 85.61 | 999 | Panthers |
| 11 | 11 | Andrew Gardner | 86.11 | 83.10 | 83.95 | 174 | Eagles |
| 12 | 12 | Kevin Zeitler | 85.92 | 78.90 | 86.44 | 1112 | Bengals |
| 13 | 13 | Gabe Jackson | 85.86 | 79.90 | 85.67 | 1050 | Raiders |
| 14 | 14 | Justin Pugh | 85.79 | 78.80 | 86.29 | 963 | Giants |
| 15 | 15 | Logan Mankins | 85.30 | 77.30 | 86.47 | 1018 | Buccaneers |
| 16 | 16 | Josh Sitton | 85.01 | 78.10 | 85.45 | 1280 | Packers |
| 17 | 17 | Richie Incognito | 84.41 | 77.50 | 84.85 | 1075 | Bills |
| 18 | 18 | Ramon Foster | 84.37 | 76.70 | 85.32 | 1211 | Steelers |
| 19 | 19 | Tim Lelito | 84.26 | 75.50 | 85.93 | 948 | Saints |
| 20 | 20 | Brandon Brooks | 84.24 | 78.10 | 84.16 | 1042 | Texans |
| 21 | 21 | Manuel Ramirez | 83.81 | 75.80 | 84.98 | 488 | Lions |
| 22 | 22 | Jack Mewhort | 83.72 | 76.60 | 84.30 | 1098 | Colts |
| 23 | 23 | Garrett Reynolds | 83.53 | 76.70 | 83.92 | 732 | Rams |
| 24 | 24 | Brandon Scherff | 83.49 | 76.40 | 84.05 | 1135 | Commanders |
| 25 | 25 | Chris Chester | 83.47 | 76.50 | 83.95 | 1139 | Falcons |
| 26 | 26 | Andrew Tiller | 83.43 | 75.10 | 84.81 | 614 | 49ers |
| 27 | 27 | Lane Taylor | 83.27 | 78.90 | 82.02 | 153 | Packers |
| 28 | 28 | Matt Slauson | 82.82 | 74.10 | 84.47 | 1075 | Bears |
| 29 | 29 | James Carpenter | 82.46 | 75.40 | 83.00 | 1104 | Jets |
| 30 | 30 | Andy Levitre | 82.36 | 74.00 | 83.77 | 1126 | Falcons |
| 31 | 31 | Chance Warmack | 82.34 | 74.80 | 83.20 | 829 | Titans |
| 32 | 32 | John Greco | 82.15 | 74.30 | 83.22 | 890 | Browns |
| 33 | 33 | Geoff Schwartz | 81.73 | 74.60 | 82.31 | 668 | Giants |
| 34 | 34 | Ron Leary | 81.16 | 72.80 | 82.56 | 218 | Cowboys |
| 35 | 35 | D.J. Fluker | 81.06 | 71.50 | 83.26 | 863 | Chargers |
| 36 | 36 | Clint Boling | 80.55 | 73.00 | 81.41 | 1121 | Bengals |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 37 | 1 | Larry Warford | 79.58 | 71.70 | 80.66 | 812 | Lions |
| 38 | 2 | Jahri Evans | 79.21 | 70.00 | 81.19 | 800 | Saints |
| 39 | 3 | Joel Bitonio | 79.11 | 69.40 | 81.41 | 615 | Browns |
| 40 | 4 | Patrick Omameh | 79.09 | 70.20 | 80.85 | 667 | Bears |
| 41 | 5 | Zach Fulton | 78.40 | 69.50 | 80.16 | 551 | Chiefs |
| 42 | 6 | Amini Silatolu | 78.34 | 70.00 | 79.73 | 238 | Panthers |
| 43 | 7 | Brian Winters | 78.15 | 68.80 | 80.22 | 766 | Jets |
| 44 | 8 | Hugh Thornton | 78.14 | 68.60 | 80.34 | 799 | Colts |
| 45 | 9 | A.J. Cann | 77.97 | 69.50 | 79.45 | 861 | Jaguars |
| 46 | 10 | Vladimir Ducasse | 77.77 | 67.30 | 80.58 | 743 | Bears |
| 47 | 11 | J.R. Sweezy | 77.57 | 67.50 | 80.12 | 1127 | Seahawks |
| 48 | 12 | Laken Tomlinson | 77.27 | 69.20 | 78.48 | 986 | Lions |
| 49 | 13 | Ben Grubbs | 77.01 | 67.50 | 79.18 | 459 | Chiefs |
| 50 | 14 | Senio Kelemete | 76.79 | 66.70 | 79.35 | 424 | Saints |
| 51 | 15 | Max Garcia | 76.58 | 65.90 | 79.53 | 575 | Broncos |
| 52 | 16 | Shaq Mason | 76.38 | 64.60 | 80.06 | 868 | Patriots |
| 53 | 17 | Josh Kline | 76.07 | 65.90 | 78.69 | 996 | Patriots |
| 54 | 18 | Brandon Fusco | 75.87 | 66.10 | 78.22 | 1078 | Vikings |
| 55 | 19 | John Jerry | 75.49 | 66.80 | 77.11 | 644 | Giants |
| 56 | 20 | Louis Vasquez | 75.48 | 67.20 | 76.84 | 1048 | Broncos |
| 57 | 21 | Matt Tobin | 75.45 | 65.00 | 78.25 | 984 | Eagles |
| 58 | 22 | Allen Barbre | 75.24 | 65.10 | 77.84 | 1156 | Eagles |
| 59 | 23 | Laurent Duvernay-Tardif | 74.40 | 67.30 | 74.97 | 820 | Chiefs |

### Starter (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 60 | 1 | Tre Jackson | 73.72 | 65.30 | 75.17 | 607 | Patriots |
| 61 | 2 | Xavier Su'a-Filo | 73.67 | 63.80 | 76.09 | 691 | Texans |
| 62 | 3 | Shawn Lauvao | 73.57 | 60.90 | 77.85 | 150 | Commanders |
| 63 | 4 | Jonathan Cooper | 73.50 | 63.70 | 75.86 | 635 | Cardinals |
| 64 | 5 | Kraig Urbik | 73.46 | 63.00 | 76.27 | 431 | Bills |
| 65 | 6 | Alex Boone | 72.86 | 63.70 | 74.80 | 761 | 49ers |
| 66 | 7 | Jon Feliciano | 72.79 | 60.40 | 76.89 | 187 | Raiders |
| 67 | 8 | Cody Wichmann | 72.15 | 65.10 | 72.69 | 427 | Rams |
| 68 | 9 | Willie Colon | 71.44 | 60.90 | 74.30 | 341 | Jets |
| 69 | 10 | Kenny Wiggins | 71.33 | 59.50 | 75.05 | 791 | Chargers |
| 70 | 11 | Chris Scott | 70.75 | 59.10 | 74.35 | 107 | Panthers |
| 71 | 12 | Quinton Spain | 70.37 | 61.90 | 71.85 | 383 | Titans |
| 72 | 13 | Zane Beadles | 69.38 | 58.50 | 72.46 | 1058 | Jaguars |
| 73 | 14 | Jordan Devey | 68.40 | 57.00 | 71.83 | 384 | 49ers |
| 74 | 15 | Rodger Saffold | 67.24 | 49.90 | 74.64 | 231 | Rams |
| 75 | 16 | Billy Turner | 67.00 | 55.20 | 70.70 | 765 | Dolphins |
| 76 | 17 | Oday Aboushi | 66.86 | 53.60 | 71.53 | 398 | Texans |
| 77 | 18 | Jeff Adams | 66.72 | 57.80 | 68.50 | 116 | Texans |
| 78 | 19 | Ted Larsen | 66.39 | 53.30 | 70.95 | 802 | Cardinals |
| 79 | 20 | Lance Louis | 65.16 | 51.20 | 70.30 | 237 | Colts |
| 80 | 21 | Orlando Franklin | 63.78 | 51.20 | 68.00 | 616 | Chargers |
| 81 | 22 | John Miller | 63.25 | 48.00 | 69.25 | 648 | Bills |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 82 | 1 | Dallas Thomas | 61.26 | 45.50 | 67.60 | 1030 | Dolphins |
| 83 | 2 | Cameron Erving | 58.08 | 40.20 | 65.84 | 424 | Browns |

## HB — Running Back

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (0 players)

_None._

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Marshawn Lynch | 79.65 | 85.20 | 71.78 | 157 | Seahawks |
| 2 | 2 | Le'Veon Bell | 79.30 | 89.20 | 68.53 | 138 | Steelers |
| 3 | 3 | Dion Lewis | 77.92 | 71.30 | 78.16 | 191 | Patriots |
| 4 | 4 | Doug Martin | 76.87 | 84.00 | 67.95 | 209 | Buccaneers |
| 5 | 5 | Charles Sims | 75.00 | 85.50 | 63.84 | 246 | Buccaneers |
| 6 | 6 | Todd Gurley II | 74.67 | 72.90 | 71.69 | 146 | Rams |

### Starter (48 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Chris Ivory | 73.48 | 70.40 | 71.37 | 188 | Jets |
| 8 | 2 | Jamaal Charles | 73.13 | 70.10 | 70.98 | 142 | Chiefs |
| 9 | 3 | Carlos Hyde | 73.01 | 72.20 | 69.39 | 103 | 49ers |
| 10 | 4 | David Johnson | 72.96 | 71.50 | 69.77 | 288 | Cardinals |
| 11 | 5 | Chris Thompson | 72.75 | 64.30 | 74.22 | 204 | Commanders |
| 12 | 6 | Jonathan Stewart | 72.60 | 71.80 | 68.97 | 225 | Panthers |
| 13 | 7 | Adrian Peterson | 72.50 | 65.70 | 72.87 | 231 | Vikings |
| 14 | 8 | Theo Riddick | 72.10 | 82.80 | 60.80 | 300 | Lions |
| 15 | 9 | LeSean McCoy | 71.63 | 71.10 | 67.82 | 254 | Bills |
| 16 | 10 | Darren Sproles | 71.61 | 65.40 | 71.59 | 256 | Eagles |
| 17 | 11 | DeAngelo Williams | 71.45 | 74.40 | 65.32 | 375 | Steelers |
| 18 | 12 | C.J. Spiller | 71.17 | 66.70 | 69.98 | 140 | Saints |
| 19 | 13 | Lamar Miller | 71.09 | 73.70 | 65.19 | 310 | Dolphins |
| 20 | 14 | Eddie Lacy | 71.02 | 69.20 | 68.07 | 179 | Packers |
| 21 | 15 | Duke Johnson Jr. | 70.89 | 71.50 | 66.32 | 341 | Browns |
| 22 | 16 | Matt Forte | 70.85 | 70.70 | 66.78 | 288 | Bears |
| 23 | 17 | Giovani Bernard | 70.36 | 69.10 | 67.04 | 280 | Bengals |
| 24 | 18 | C.J. Anderson | 70.21 | 63.60 | 70.45 | 282 | Broncos |
| 25 | 19 | Rashad Jennings | 69.73 | 71.00 | 64.72 | 166 | Giants |
| 26 | 20 | T.J. Yeldon | 69.28 | 69.40 | 65.04 | 287 | Jaguars |
| 27 | 21 | Mark Ingram II | 68.92 | 66.80 | 66.16 | 267 | Saints |
| 28 | 22 | Justin Forsett | 68.88 | 64.30 | 67.77 | 219 | Ravens |
| 29 | 23 | Arian Foster | 68.63 | 65.90 | 66.28 | 107 | Texans |
| 30 | 24 | Ryan Mathews | 68.05 | 63.00 | 67.25 | 108 | Eagles |
| 31 | 25 | Fred Jackson | 67.76 | 58.20 | 69.96 | 201 | Seahawks |
| 32 | 26 | Andre Ellington | 67.28 | 66.20 | 63.83 | 126 | Cardinals |
| 33 | 27 | Devonta Freeman | 67.14 | 68.10 | 62.33 | 361 | Falcons |
| 34 | 28 | Danny Woodhead | 67.13 | 72.60 | 59.31 | 396 | Chargers |
| 35 | 29 | Antonio Andrews | 66.44 | 68.30 | 61.04 | 174 | Titans |
| 36 | 30 | DeMarco Murray | 66.34 | 56.20 | 68.94 | 242 | Eagles |
| 37 | 31 | James Starks | 66.03 | 60.40 | 65.61 | 284 | Packers |
| 38 | 32 | Shane Vereen | 66.01 | 64.90 | 62.58 | 303 | Giants |
| 39 | 33 | Latavius Murray | 65.81 | 64.50 | 62.52 | 274 | Raiders |
| 40 | 34 | Bilal Powell | 65.19 | 67.30 | 59.62 | 236 | Jets |
| 41 | 35 | Frank Gore | 64.98 | 61.60 | 63.06 | 291 | Colts |
| 42 | 36 | Alfred Morris | 64.76 | 58.00 | 65.10 | 125 | Commanders |
| 43 | 37 | Darren McFadden | 64.70 | 60.10 | 63.60 | 263 | Cowboys |
| 44 | 38 | Chris Polk | 64.59 | 53.40 | 67.89 | 127 | Texans |
| 45 | 39 | Benny Cunningham | 64.57 | 61.00 | 62.78 | 156 | Rams |
| 46 | 40 | Damien Williams | 64.45 | 64.80 | 60.05 | 117 | Dolphins |
| 47 | 41 | Dexter McCluster | 64.22 | 61.60 | 61.80 | 189 | Titans |
| 48 | 42 | Fozzy Whittaker | 63.96 | 67.50 | 57.44 | 101 | Panthers |
| 49 | 43 | Ameer Abdullah | 63.96 | 64.20 | 59.63 | 170 | Lions |
| 50 | 44 | Jeremy Hill | 63.59 | 58.20 | 63.01 | 164 | Bengals |
| 51 | 45 | Isaiah Crowell | 63.45 | 65.30 | 58.05 | 194 | Browns |
| 52 | 46 | Brandon Bolden | 63.04 | 59.30 | 61.37 | 109 | Patriots |
| 53 | 47 | Matt Asiata | 62.97 | 64.70 | 57.65 | 100 | Vikings |
| 54 | 48 | Jonathan Grimes | 62.70 | 63.00 | 58.33 | 174 | Texans |

### Rotation/backup (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 55 | 1 | James White | 61.65 | 66.10 | 54.51 | 276 | Patriots |
| 56 | 2 | Javorius Allen | 61.54 | 58.00 | 59.74 | 199 | Ravens |
| 57 | 3 | Charcandrick West | 61.03 | 58.10 | 58.82 | 278 | Chiefs |
| 58 | 4 | Alfred Blue | 60.98 | 60.90 | 56.86 | 133 | Texans |
| 59 | 5 | Melvin Gordon III | 60.95 | 54.60 | 61.01 | 157 | Chargers |
| 60 | 6 | Denard Robinson | 60.83 | 56.10 | 59.81 | 132 | Jaguars |
| 61 | 7 | Jeremy Langford | 60.26 | 61.10 | 55.54 | 166 | Bears |
| 62 | 8 | Shaun Draughn | 60.19 | 61.70 | 55.02 | 146 | 49ers |
| 63 | 9 | Ronnie Hillman | 60.07 | 57.40 | 57.68 | 251 | Broncos |
| 64 | 10 | Chris Johnson | 59.14 | 47.40 | 62.80 | 115 | Cardinals |
| 65 | 11 | Matt Jones | 56.08 | 46.90 | 58.04 | 128 | Commanders |

## LB — Linebacker

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Luke Kuechly | 90.96 | 93.60 | 85.03 | 946 | Panthers |
| 2 | 2 | Wesley Woodyard | 85.99 | 90.20 | 79.02 | 502 | Titans |
| 3 | 3 | K.J. Wright | 85.74 | 87.90 | 80.34 | 1089 | Seahawks |
| 4 | 4 | Jerrell Freeman | 84.17 | 89.40 | 78.40 | 748 | Colts |
| 5 | 5 | Anthony Barr | 83.75 | 90.00 | 77.63 | 882 | Vikings |
| 6 | 6 | Sean Lee | 81.96 | 88.40 | 78.19 | 814 | Cowboys |
| 7 | 7 | Dont'a Hightower | 80.07 | 86.40 | 73.03 | 722 | Patriots |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Derrick Johnson | 79.64 | 85.10 | 76.51 | 1178 | Chiefs |
| 9 | 2 | Thomas Davis Sr. | 78.75 | 80.00 | 73.75 | 1154 | Panthers |
| 10 | 3 | Jamie Collins Sr. | 77.80 | 84.00 | 70.54 | 869 | Patriots |
| 11 | 4 | Danny Trevathan | 77.57 | 80.40 | 75.59 | 883 | Broncos |
| 12 | 5 | Karlos Dansby | 76.86 | 76.70 | 74.05 | 1028 | Browns |
| 13 | 6 | Benardrick McKinney | 76.61 | 77.50 | 73.93 | 465 | Texans |
| 14 | 7 | Shaq Thompson | 76.46 | 76.80 | 72.07 | 441 | Panthers |
| 15 | 8 | Vince Williams | 76.37 | 79.40 | 71.01 | 195 | Steelers |
| 16 | 9 | Brandon Marshall | 75.95 | 77.00 | 74.51 | 1082 | Broncos |
| 17 | 10 | A.J. Klein | 75.75 | 78.10 | 72.51 | 328 | Panthers |
| 18 | 11 | Denzel Perryman | 75.70 | 76.00 | 74.46 | 387 | Chargers |
| 19 | 12 | Zach Brown | 75.38 | 78.00 | 74.15 | 489 | Titans |
| 20 | 13 | Vontaze Burfict | 74.99 | 76.60 | 75.80 | 540 | Bengals |

### Starter (49 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Rolando McClain | 73.45 | 74.90 | 72.28 | 639 | Cowboys |
| 22 | 2 | Jasper Brinkley | 72.82 | 78.80 | 69.45 | 420 | Giants |
| 23 | 3 | Tahir Whitehead | 71.87 | 70.70 | 68.48 | 585 | Lions |
| 24 | 4 | Erin Henderson | 71.20 | 69.50 | 71.09 | 226 | Jets |
| 25 | 5 | Jordan Hicks | 71.07 | 80.90 | 71.22 | 446 | Eagles |
| 26 | 6 | David Harris | 71.01 | 69.70 | 67.72 | 970 | Jets |
| 27 | 7 | Bobby Wagner | 70.89 | 69.20 | 68.49 | 1027 | Seahawks |
| 28 | 8 | Bruce Irvin | 70.81 | 67.10 | 69.11 | 798 | Seahawks |
| 29 | 9 | Christian Kirksey | 70.72 | 69.70 | 67.23 | 570 | Browns |
| 30 | 10 | Josh Mauga | 70.30 | 72.00 | 67.28 | 519 | Chiefs |
| 31 | 11 | Nate Irving | 70.29 | 77.30 | 69.15 | 106 | Colts |
| 32 | 12 | Nate Stupar | 70.18 | 76.00 | 68.39 | 257 | Falcons |
| 33 | 13 | C.J. Mosley | 69.46 | 66.80 | 67.06 | 1043 | Ravens |
| 34 | 14 | Rey Maualuga | 69.36 | 69.20 | 66.65 | 660 | Bengals |
| 35 | 15 | Telvin Smith Sr. | 69.28 | 68.30 | 67.06 | 994 | Jaguars |
| 36 | 16 | Josh Bynes | 69.13 | 68.40 | 66.91 | 815 | Lions |
| 37 | 17 | Max Bullough | 68.51 | 68.50 | 68.90 | 119 | Texans |
| 38 | 18 | Zachary Orr | 68.50 | 72.50 | 69.08 | 142 | Ravens |
| 39 | 19 | Manny Lawson | 68.39 | 65.00 | 66.48 | 701 | Bills |
| 40 | 20 | Koa Misi | 67.83 | 68.70 | 66.41 | 728 | Dolphins |
| 41 | 21 | Christian Jones | 67.76 | 65.70 | 67.56 | 741 | Bears |
| 42 | 22 | NaVorro Bowman | 67.74 | 63.70 | 66.27 | 1097 | 49ers |
| 43 | 23 | Lavonte David | 66.99 | 63.80 | 65.59 | 1083 | Buccaneers |
| 44 | 24 | Sean Spence | 66.87 | 65.70 | 65.43 | 271 | Steelers |
| 45 | 25 | Craig Robertson | 66.69 | 62.60 | 67.75 | 383 | Browns |
| 46 | 26 | Todd Davis | 66.16 | 65.10 | 65.84 | 134 | Broncos |
| 47 | 27 | Daryl Smith | 65.94 | 63.20 | 63.60 | 971 | Ravens |
| 48 | 28 | Avery Williamson | 65.91 | 62.20 | 65.25 | 922 | Titans |
| 49 | 29 | Michael Mauti | 65.42 | 69.40 | 67.87 | 174 | Saints |
| 50 | 30 | Joe Thomas | 64.90 | 62.70 | 62.20 | 316 | Packers |
| 51 | 31 | A.J. Hawk | 64.58 | 60.60 | 63.07 | 304 | Bengals |
| 52 | 32 | Eric Kendricks | 64.47 | 59.20 | 64.85 | 816 | Vikings |
| 53 | 33 | Akeem Dent | 64.16 | 64.60 | 68.77 | 118 | Texans |
| 54 | 34 | Clay Matthews | 64.15 | 62.60 | 61.01 | 1144 | Packers |
| 55 | 35 | Jelani Jenkins | 64.11 | 60.40 | 65.11 | 696 | Dolphins |
| 56 | 36 | J.T. Thomas | 63.98 | 60.40 | 66.26 | 400 | Giants |
| 57 | 37 | Stephen Tulloch | 63.87 | 60.70 | 65.89 | 722 | Lions |
| 58 | 38 | Vincent Rey | 63.74 | 56.10 | 64.67 | 753 | Bengals |
| 59 | 39 | D'Qwell Jackson | 63.74 | 57.30 | 63.86 | 1087 | Colts |
| 60 | 40 | Danny Lansanah | 63.60 | 63.80 | 62.74 | 369 | Buccaneers |
| 61 | 41 | Paul Posluszny | 63.47 | 60.90 | 65.08 | 977 | Jaguars |
| 62 | 42 | Ryan Shazier | 63.46 | 61.80 | 64.43 | 808 | Steelers |
| 63 | 43 | Zach Vigil | 62.94 | 61.70 | 63.76 | 141 | Dolphins |
| 64 | 44 | Neville Hewitt | 62.73 | 61.90 | 64.31 | 342 | Dolphins |
| 65 | 45 | Stephone Anthony | 62.48 | 55.30 | 63.10 | 987 | Saints |
| 66 | 46 | Ben Heeney | 62.41 | 62.70 | 62.22 | 307 | Raiders |
| 67 | 47 | Philip Wheeler | 62.32 | 59.70 | 63.87 | 143 | Falcons |
| 68 | 48 | Chad Greenway | 62.14 | 55.70 | 63.52 | 655 | Vikings |
| 69 | 49 | Devon Kennard | 62.13 | 62.40 | 64.68 | 483 | Giants |

### Rotation/backup (59 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 70 | 1 | Jonathan Freeny | 61.95 | 60.10 | 65.07 | 444 | Patriots |
| 71 | 2 | Dan Skuta | 61.86 | 59.40 | 60.90 | 415 | Jaguars |
| 72 | 3 | Mason Foster | 61.59 | 59.50 | 64.01 | 334 | Commanders |
| 73 | 4 | Jerod Mayo | 61.37 | 59.60 | 63.58 | 401 | Patriots |
| 74 | 5 | Brian Cushing | 61.28 | 55.20 | 63.67 | 1039 | Texans |
| 75 | 6 | Mark Herzlich | 60.93 | 58.30 | 61.53 | 130 | Giants |
| 76 | 7 | Josh McNary | 60.65 | 62.80 | 63.48 | 117 | Colts |
| 77 | 8 | Hayes Pullard | 60.64 | 70.60 | 68.34 | 151 | Jaguars |
| 78 | 9 | Shea McClellin | 60.31 | 57.80 | 61.98 | 673 | Bears |
| 79 | 10 | Joplo Bartu | 60.12 | 57.00 | 63.96 | 140 | Jaguars |
| 80 | 11 | Andrew Gachkar | 60.06 | 59.70 | 63.12 | 115 | Cowboys |
| 81 | 12 | Kyle Wilber | 59.72 | 56.40 | 61.84 | 225 | Cowboys |
| 82 | 13 | Demario Davis | 59.48 | 52.50 | 59.96 | 847 | Jets |
| 83 | 14 | Justin Durant | 59.32 | 58.00 | 61.97 | 652 | Falcons |
| 84 | 15 | Mychal Kendricks | 59.29 | 51.50 | 63.14 | 623 | Eagles |
| 85 | 16 | Bruce Carter | 59.16 | 51.90 | 61.40 | 307 | Buccaneers |
| 86 | 17 | Sean Weatherspoon | 59.06 | 53.10 | 61.89 | 153 | Cardinals |
| 87 | 18 | Travis Lewis | 58.95 | 54.70 | 61.79 | 138 | Lions |
| 88 | 19 | Malcolm Smith | 58.94 | 53.20 | 60.48 | 1139 | Raiders |
| 89 | 20 | Hau'oli Kikaha | 58.49 | 51.90 | 59.75 | 617 | Saints |
| 90 | 21 | Alec Ogletree | 58.01 | 53.90 | 62.83 | 260 | Rams |
| 91 | 22 | DeMeco Ryans | 57.90 | 53.60 | 60.15 | 599 | Eagles |
| 92 | 23 | Gerald Hodges | 57.87 | 51.90 | 62.78 | 517 | 49ers |
| 93 | 24 | Akeem Ayers | 57.72 | 49.60 | 58.97 | 527 | Rams |
| 94 | 25 | Anthony Hitchens | 57.65 | 49.40 | 58.98 | 538 | Cowboys |
| 95 | 26 | Kelvin Sheppard | 57.62 | 52.10 | 60.27 | 706 | Dolphins |
| 96 | 27 | Michael Wilhoite | 57.54 | 51.80 | 61.36 | 615 | 49ers |
| 97 | 28 | Jake Ryan | 56.92 | 50.00 | 62.56 | 328 | Packers |
| 98 | 29 | Kevin Minter | 56.54 | 50.30 | 59.66 | 1033 | Cardinals |
| 99 | 30 | Jonathan Anderson | 56.43 | 53.10 | 59.69 | 314 | Bears |
| 100 | 31 | Ramik Wilson | 56.08 | 64.10 | 66.56 | 129 | Chiefs |
| 101 | 32 | Ramon Humber | 55.37 | 41.70 | 62.60 | 272 | Saints |
| 102 | 33 | Nigel Bradham | 55.35 | 49.70 | 58.38 | 724 | Bills |
| 103 | 34 | James Laurinaitis | 55.31 | 43.70 | 58.88 | 1152 | Rams |
| 104 | 35 | Justin Tuggle | 55.18 | 51.70 | 61.25 | 105 | Texans |
| 105 | 36 | Kevin Pierre-Louis | 55.03 | 56.00 | 59.72 | 104 | Seahawks |
| 106 | 37 | Paul Worrilow | 54.97 | 46.40 | 57.45 | 869 | Falcons |
| 107 | 38 | Curtis Lofton | 54.86 | 42.30 | 59.06 | 574 | Raiders |
| 108 | 39 | Jonathan Casillas | 54.70 | 48.30 | 57.94 | 672 | Giants |
| 109 | 40 | Ray-Ray Armstrong | 54.59 | 54.00 | 59.15 | 208 | 49ers |
| 110 | 41 | Kavell Conner | 53.60 | 51.70 | 59.04 | 184 | Chargers |
| 111 | 42 | Will Compton | 53.57 | 43.50 | 57.36 | 786 | Commanders |
| 112 | 43 | Emmanuel Lamur | 52.91 | 46.20 | 56.04 | 337 | Bengals |
| 113 | 44 | Thurston Armbrister | 52.16 | 45.30 | 58.81 | 206 | Jaguars |
| 114 | 45 | Lawrence Timmons | 52.15 | 40.30 | 55.88 | 1179 | Steelers |
| 115 | 46 | Manti Te'o | 52.07 | 43.90 | 57.52 | 709 | Chargers |
| 116 | 47 | Dannell Ellerbe | 51.88 | 47.80 | 60.54 | 250 | Saints |
| 117 | 48 | Donald Butler | 51.79 | 39.50 | 58.21 | 500 | Chargers |
| 118 | 49 | Preston Brown | 51.69 | 37.30 | 57.12 | 1064 | Bills |
| 119 | 50 | Keenan Robinson | 51.11 | 39.20 | 57.90 | 548 | Commanders |
| 120 | 51 | Nate Palmer | 50.96 | 38.20 | 57.25 | 534 | Packers |
| 121 | 52 | Perry Riley | 50.79 | 40.30 | 57.88 | 461 | Commanders |
| 122 | 53 | Kiko Alonso | 50.69 | 40.10 | 56.84 | 470 | Eagles |
| 123 | 54 | Uani' Unga | 50.43 | 41.30 | 56.52 | 431 | Giants |
| 124 | 55 | David Hawthorne | 50.07 | 37.00 | 59.51 | 222 | Saints |
| 125 | 56 | Kwon Alexander | 49.76 | 41.40 | 55.34 | 809 | Buccaneers |
| 126 | 57 | John Timu | 45.51 | 45.70 | 61.21 | 159 | Bears |
| 127 | 58 | Jon Beason | 45.45 | 36.60 | 57.09 | 159 | Giants |
| 128 | 59 | James Anderson | 45.25 | 30.50 | 57.68 | 108 | Saints |

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
| 7 | 2 | Andy Dalton | 78.00 | 76.63 | 78.33 | 463 | Bengals |
| 8 | 3 | Matt Ryan | 76.97 | 80.55 | 69.50 | 703 | Falcons |
| 9 | 4 | Aaron Rodgers | 75.76 | 77.09 | 70.88 | 832 | Packers |
| 10 | 5 | Philip Rivers | 75.25 | 74.04 | 71.70 | 766 | Chargers |

### Starter (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Kirk Cousins | 72.73 | 69.31 | 75.28 | 668 | Commanders |
| 12 | 2 | Jay Cutler | 72.27 | 71.66 | 70.45 | 556 | Bears |
| 13 | 3 | Matthew Stafford | 72.02 | 67.43 | 71.24 | 694 | Lions |
| 14 | 4 | Derek Carr | 71.57 | 70.78 | 67.67 | 652 | Raiders |
| 15 | 5 | Alex Smith | 71.54 | 69.80 | 68.88 | 697 | Chiefs |
| 16 | 6 | Sam Bradford | 71.53 | 75.20 | 67.14 | 611 | Eagles |
| 17 | 7 | Ryan Tannehill | 71.43 | 70.50 | 67.65 | 673 | Dolphins |
| 18 | 8 | Eli Manning | 70.58 | 66.79 | 69.18 | 699 | Giants |
| 19 | 9 | Teddy Bridgewater | 69.54 | 68.82 | 68.02 | 579 | Vikings |
| 20 | 10 | Tony Romo | 68.11 | 73.74 | 68.90 | 137 | Cowboys |
| 21 | 11 | Joe Flacco | 67.91 | 68.35 | 66.45 | 458 | Ravens |
| 22 | 12 | Peyton Manning | 67.16 | 70.02 | 63.37 | 473 | Broncos |
| 23 | 13 | Jameis Winston | 66.83 | 67.20 | 68.61 | 650 | Buccaneers |
| 24 | 14 | Tyrod Taylor | 66.82 | 73.67 | 76.39 | 507 | Bills |
| 25 | 15 | Ryan Fitzpatrick | 66.77 | 63.92 | 66.99 | 638 | Jets |
| 26 | 16 | Blake Bortles | 65.15 | 58.66 | 66.41 | 724 | Jaguars |
| 27 | 17 | Andrew Luck | 63.00 | 63.73 | 63.21 | 353 | Colts |
| 28 | 18 | Josh McCown | 62.94 | 59.94 | 69.09 | 356 | Browns |
| 29 | 19 | Marcus Mariota | 62.39 | 61.10 | 69.28 | 461 | Titans |

### Rotation/backup (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Brandon Weeden | 61.66 | 64.90 | 70.48 | 173 | Texans |
| 31 | 2 | Brian Hoyer | 61.41 | 59.80 | 64.08 | 457 | Texans |
| 32 | 3 | Case Keenum | 60.76 | 66.08 | 66.62 | 142 | Rams |
| 33 | 4 | A.J. McCarron | 60.73 | 66.70 | 65.56 | 197 | Bengals |
| 34 | 5 | Brock Osweiler | 60.65 | 65.42 | 65.41 | 326 | Broncos |
| 35 | 6 | Colin Kaepernick | 60.18 | 58.54 | 63.31 | 306 | 49ers |
| 36 | 7 | Blaine Gabbert | 60.10 | 60.77 | 66.36 | 345 | 49ers |
| 37 | 8 | Mark Sanchez | 60.07 | 64.08 | 65.91 | 104 | Eagles |
| 38 | 9 | Matt Hasselbeck | 60.00 | 66.31 | 62.63 | 297 | Colts |
| 39 | 10 | Kellen Moore | 59.02 | 64.68 | 61.29 | 113 | Cowboys |
| 40 | 11 | Nick Foles | 58.77 | 62.43 | 58.82 | 379 | Rams |
| 41 | 12 | Ryan Mallett | 56.55 | 57.55 | 56.12 | 265 | Ravens |
| 42 | 13 | Johnny Manziel | 55.76 | 47.87 | 60.30 | 281 | Browns |
| 43 | 14 | Austin Davis | 55.61 | 53.66 | 57.46 | 119 | Browns |
| 44 | 15 | Jimmy Clausen | 54.52 | 51.18 | 54.16 | 145 | Ravens |
| 45 | 16 | Zach Mettenberger | 54.18 | 45.42 | 56.47 | 188 | Titans |
| 46 | 17 | Matt Cassel | 54.01 | 52.65 | 57.18 | 248 | Cowboys |

## S — Safety

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Earl Thomas III | 90.56 | 90.90 | 86.17 | 1098 | Seahawks |
| 2 | 2 | Charles Woodson | 90.21 | 92.10 | 84.78 | 1109 | Raiders |
| 3 | 3 | Kurt Coleman | 87.03 | 88.80 | 84.08 | 1193 | Panthers |
| 4 | 4 | Duron Harmon | 82.98 | 85.10 | 77.60 | 669 | Patriots |
| 5 | 5 | Eric Berry | 82.40 | 82.30 | 81.44 | 1148 | Chiefs |
| 6 | 6 | Devin McCourty | 81.77 | 78.00 | 80.12 | 1079 | Patriots |
| 7 | 7 | Patrick Chung | 81.52 | 79.30 | 79.46 | 1040 | Patriots |
| 8 | 8 | Harrison Smith | 81.26 | 78.20 | 81.84 | 848 | Vikings |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Malcolm Jenkins | 79.38 | 75.50 | 77.80 | 1203 | Eagles |
| 10 | 2 | Eric Weddle | 79.38 | 78.40 | 77.43 | 750 | Chargers |
| 11 | 3 | Ha Ha Clinton-Dix | 79.08 | 71.80 | 79.76 | 1176 | Packers |
| 12 | 4 | Calvin Pryor | 78.98 | 81.90 | 75.22 | 709 | Jets |
| 13 | 5 | Shawn Williams | 78.77 | 83.00 | 77.10 | 524 | Bengals |
| 14 | 6 | Mike Mitchell | 78.75 | 78.60 | 74.68 | 1188 | Steelers |
| 15 | 7 | Darian Stewart | 77.42 | 76.20 | 74.70 | 1011 | Broncos |
| 16 | 8 | Reggie Nelson | 76.81 | 74.20 | 74.38 | 1068 | Bengals |
| 17 | 9 | Rodney McLeod | 76.77 | 71.30 | 76.25 | 1149 | Rams |
| 18 | 10 | Chris Banjo | 76.09 | 72.70 | 78.96 | 102 | Packers |
| 19 | 11 | Andre Hal | 75.95 | 76.40 | 72.13 | 787 | Texans |
| 20 | 12 | George Iloka | 75.59 | 72.40 | 75.12 | 713 | Bengals |
| 21 | 13 | Kelcie McCray | 75.16 | 75.00 | 78.40 | 225 | Seahawks |
| 22 | 14 | Tre Boston | 75.12 | 72.40 | 75.36 | 293 | Panthers |
| 23 | 15 | Reshad Jones | 74.62 | 67.30 | 76.59 | 1117 | Dolphins |
| 24 | 16 | Donte Whitner | 74.48 | 70.40 | 74.07 | 847 | Browns |
| 25 | 17 | Husain Abdullah | 74.24 | 69.00 | 75.13 | 486 | Chiefs |
| 26 | 18 | Jordan Richards | 74.18 | 74.50 | 72.93 | 244 | Patriots |

### Starter (46 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | Da'Norris Searcy | 73.62 | 66.20 | 75.23 | 883 | Titans |
| 28 | 2 | Morgan Burnett | 73.33 | 68.40 | 74.43 | 816 | Packers |
| 29 | 3 | Keith Tandy | 72.92 | 68.80 | 75.76 | 280 | Buccaneers |
| 30 | 4 | Marcus Gilchrist | 72.87 | 64.70 | 74.15 | 1045 | Jets |
| 31 | 5 | Rontez Miles | 72.62 | 65.60 | 78.33 | 127 | Jets |
| 32 | 6 | Quintin Demps | 72.32 | 67.80 | 72.00 | 817 | Texans |
| 33 | 7 | T.J. Ward | 72.11 | 70.20 | 69.74 | 886 | Broncos |
| 34 | 8 | Micah Hyde | 71.70 | 69.70 | 68.86 | 709 | Packers |
| 35 | 9 | Ron Parker | 71.59 | 69.70 | 68.69 | 1177 | Chiefs |
| 36 | 10 | David Bruton | 70.69 | 68.80 | 71.53 | 480 | Broncos |
| 37 | 11 | Kam Chancellor | 70.52 | 68.00 | 69.60 | 756 | Seahawks |
| 38 | 12 | Eric Reid | 70.11 | 65.80 | 69.13 | 1113 | 49ers |
| 39 | 13 | Kemal Ishmael | 70.00 | 69.20 | 70.54 | 375 | Falcons |
| 40 | 14 | Chris Conte | 69.58 | 66.10 | 70.02 | 731 | Buccaneers |
| 41 | 15 | Jamarca Sanford | 69.23 | 66.40 | 72.89 | 103 | Saints |
| 42 | 16 | Roman Harper | 68.95 | 63.40 | 69.52 | 1067 | Panthers |
| 43 | 17 | Isa Abdul-Quddus | 68.67 | 62.50 | 70.39 | 569 | Lions |
| 44 | 18 | Bradley McDougald | 68.07 | 61.80 | 68.72 | 888 | Buccaneers |
| 45 | 19 | Kendrick Lewis | 67.94 | 64.00 | 66.91 | 925 | Ravens |
| 46 | 20 | Tony Jefferson | 67.57 | 60.40 | 70.06 | 887 | Cardinals |
| 47 | 21 | Kyshoen Jarrett | 67.56 | 66.90 | 63.84 | 600 | Commanders |
| 48 | 22 | Michael Thomas | 67.47 | 68.60 | 68.48 | 697 | Dolphins |
| 49 | 23 | Anthony Harris | 67.30 | 73.80 | 78.80 | 147 | Vikings |
| 50 | 24 | Kenny Vaccaro | 67.20 | 62.10 | 67.16 | 1059 | Saints |
| 51 | 25 | Taylor Mays | 66.73 | 65.90 | 69.26 | 314 | Raiders |
| 52 | 26 | Glover Quin | 66.52 | 57.40 | 68.43 | 951 | Lions |
| 53 | 27 | Eddie Pleasant | 66.46 | 67.20 | 66.39 | 388 | Texans |
| 54 | 28 | Will Hill III | 66.44 | 59.80 | 68.79 | 943 | Ravens |
| 55 | 29 | Ibraheim Campbell | 65.97 | 71.50 | 74.08 | 102 | Browns |
| 56 | 30 | Tyvon Branch | 65.25 | 61.10 | 70.84 | 526 | Chiefs |
| 57 | 31 | Jordan Poyer | 64.73 | 63.50 | 64.62 | 424 | Browns |
| 58 | 32 | Dashon Goldson | 64.41 | 61.60 | 63.37 | 1041 | Commanders |
| 59 | 33 | Dwight Lowery | 64.18 | 60.90 | 63.67 | 1084 | Colts |
| 60 | 34 | Chris Maragos | 64.15 | 69.40 | 64.71 | 300 | Eagles |
| 61 | 35 | Will Allen | 64.05 | 60.80 | 64.14 | 941 | Steelers |
| 62 | 36 | Jairus Byrd | 63.43 | 60.50 | 67.56 | 810 | Saints |
| 63 | 37 | Robert Blanton | 63.41 | 56.90 | 67.01 | 231 | Vikings |
| 64 | 38 | Bacarri Rambo | 63.18 | 57.40 | 67.96 | 686 | Bills |
| 65 | 39 | Rashad Johnson | 63.15 | 57.10 | 63.65 | 978 | Cardinals |
| 66 | 40 | Colt Anderson | 62.98 | 63.30 | 66.52 | 161 | Colts |
| 67 | 41 | Corey Graham | 62.94 | 50.40 | 67.14 | 976 | Bills |
| 68 | 42 | Jeff Heath | 62.90 | 58.80 | 64.49 | 205 | Cowboys |
| 69 | 43 | Robert Golden | 62.85 | 60.00 | 66.94 | 426 | Steelers |
| 70 | 44 | Michael Griffin | 62.53 | 54.50 | 64.65 | 939 | Titans |
| 71 | 45 | Shiloh Keo | 62.40 | 56.70 | 69.22 | 123 | Broncos |
| 72 | 46 | Mike Adams | 62.05 | 53.00 | 65.49 | 825 | Colts |

### Rotation/backup (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 73 | 1 | Jahleel Addae | 61.89 | 57.90 | 63.52 | 711 | Chargers |
| 74 | 2 | Aaron Williams | 61.63 | 61.70 | 64.92 | 186 | Bills |
| 75 | 3 | Dion Bailey | 61.51 | 64.30 | 71.45 | 153 | Jets |
| 76 | 4 | James Ihedigbo | 60.76 | 53.80 | 62.38 | 591 | Lions |
| 77 | 5 | Duke Williams | 60.73 | 58.60 | 61.63 | 282 | Bills |
| 78 | 6 | J.J. Wilcox | 60.27 | 57.50 | 58.78 | 822 | Cowboys |
| 79 | 7 | Josh Evans | 59.88 | 56.90 | 58.44 | 619 | Jaguars |
| 80 | 8 | Daimion Stafford | 59.66 | 57.40 | 61.79 | 330 | Titans |
| 81 | 9 | Rahim Moore | 59.15 | 56.60 | 62.61 | 441 | Texans |
| 82 | 10 | Antoine Bethea | 59.00 | 49.30 | 65.99 | 439 | 49ers |
| 83 | 11 | William Moore | 58.94 | 53.20 | 64.01 | 500 | Falcons |
| 84 | 12 | Colin Jones | 58.34 | 59.30 | 59.98 | 161 | Panthers |
| 85 | 13 | Barry Church | 58.22 | 49.40 | 60.45 | 865 | Cowboys |
| 86 | 14 | T.J. McDonald | 57.57 | 53.70 | 59.84 | 766 | Rams |
| 87 | 15 | Josh Bush | 57.30 | 55.20 | 63.50 | 370 | Broncos |
| 88 | 16 | Larry Asante | 57.15 | 54.50 | 61.73 | 366 | Raiders |
| 89 | 17 | Jaquiski Tartt | 56.66 | 49.20 | 58.50 | 718 | 49ers |
| 90 | 18 | D.J. Swearinger Sr. | 56.19 | 52.70 | 57.49 | 261 | Cardinals |
| 91 | 19 | Craig Dahl | 56.19 | 53.30 | 58.12 | 429 | Giants |
| 92 | 20 | Brandon Meriweather | 56.09 | 51.30 | 59.19 | 831 | Giants |
| 93 | 21 | Johnathan Cyprien | 55.86 | 50.80 | 56.64 | 1013 | Jaguars |
| 94 | 22 | Chris Prosinski | 54.34 | 52.70 | 61.37 | 337 | Bears |
| 95 | 23 | Clayton Geathers | 54.33 | 46.00 | 57.80 | 270 | Colts |
| 96 | 24 | Andrew Sendejo | 53.84 | 48.50 | 56.57 | 830 | Vikings |
| 97 | 25 | Walt Aikens | 53.55 | 53.50 | 55.80 | 444 | Dolphins |
| 98 | 26 | Landon Collins | 52.79 | 37.90 | 58.55 | 1091 | Giants |
| 99 | 27 | Tashaun Gipson Sr. | 52.52 | 42.90 | 57.90 | 798 | Browns |
| 100 | 28 | Nate Allen | 51.41 | 47.50 | 55.90 | 224 | Raiders |
| 101 | 29 | Shamiel Gary | 51.38 | 53.10 | 59.48 | 108 | Dolphins |
| 102 | 30 | Daniel Sorensen | 50.11 | 49.90 | 54.15 | 235 | Chiefs |
| 103 | 31 | Sergio Brown | 49.80 | 40.90 | 54.49 | 554 | Jaguars |
| 104 | 32 | Maurice Alexander | 49.10 | 40.30 | 54.05 | 420 | Rams |
| 105 | 33 | Major Wright | 48.63 | 35.60 | 58.56 | 218 | Buccaneers |
| 106 | 34 | James Sample | 45.59 | 44.60 | 60.59 | 129 | Jaguars |
| 107 | 35 | Antone Exum Jr. | 45.00 | 29.60 | 50.50 | 140 | Vikings |
| 108 | 36 | Jeron Johnson | 45.00 | 34.40 | 56.47 | 196 | Commanders |

## T — Tackle

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Terron Armstead | 97.09 | 92.50 | 95.99 | 926 | Saints |
| 2 | 2 | Tyron Smith | 96.50 | 91.70 | 95.53 | 1020 | Cowboys |
| 3 | 3 | Joe Thomas | 94.27 | 91.60 | 91.89 | 1103 | Browns |
| 4 | 4 | Joe Staley | 91.04 | 87.30 | 89.37 | 1007 | 49ers |
| 5 | 5 | Jason Peters | 90.29 | 82.20 | 91.52 | 759 | Eagles |
| 6 | 6 | Trent Williams | 89.16 | 82.90 | 89.17 | 993 | Commanders |
| 7 | 7 | Donald Penn | 88.92 | 83.80 | 88.17 | 1034 | Raiders |
| 8 | 8 | Andrew Whitworth | 88.80 | 83.40 | 88.23 | 1094 | Bengals |
| 9 | 9 | Duane Brown | 86.35 | 79.70 | 86.62 | 906 | Texans |
| 10 | 10 | Jared Veldheer | 85.76 | 80.10 | 85.37 | 1193 | Cardinals |
| 11 | 11 | Ryan Schraeder | 85.34 | 78.20 | 85.93 | 1128 | Falcons |
| 12 | 12 | Lane Johnson | 85.08 | 74.80 | 87.76 | 1156 | Eagles |
| 13 | 13 | Jake Matthews | 84.97 | 79.30 | 84.58 | 1126 | Falcons |
| 14 | 14 | Mitchell Schwartz | 84.90 | 79.00 | 84.67 | 1103 | Browns |
| 15 | 15 | Riley Reiff | 84.83 | 78.00 | 85.21 | 1073 | Lions |
| 16 | 16 | Morgan Moses | 84.58 | 76.20 | 86.00 | 1098 | Commanders |
| 17 | 17 | Russell Okung | 84.28 | 77.10 | 84.90 | 914 | Seahawks |
| 18 | 18 | Derek Newton | 83.77 | 75.20 | 85.31 | 1225 | Texans |
| 19 | 19 | Taylor Lewan | 83.71 | 76.30 | 84.49 | 906 | Titans |
| 20 | 20 | Demar Dotson | 82.89 | 74.10 | 84.58 | 201 | Buccaneers |
| 21 | 21 | Austin Howard | 82.55 | 73.10 | 84.69 | 809 | Raiders |
| 22 | 22 | David Bakhtiari | 82.53 | 76.00 | 82.71 | 1024 | Packers |
| 23 | 23 | Cordy Glenn | 82.40 | 76.50 | 82.17 | 1059 | Bills |
| 24 | 24 | Anthony Castonzo | 81.93 | 74.70 | 82.59 | 891 | Colts |
| 25 | 25 | Marcus Gilbert | 81.55 | 73.10 | 83.01 | 1206 | Steelers |
| 26 | 26 | Eric Fisher | 81.53 | 72.60 | 83.31 | 976 | Chiefs |
| 27 | 27 | Joe Barksdale | 81.52 | 72.80 | 83.17 | 1114 | Chargers |
| 28 | 28 | Zach Strief | 81.52 | 72.50 | 83.36 | 1071 | Saints |
| 29 | 29 | Bobby Massie | 80.63 | 70.10 | 83.48 | 1103 | Cardinals |
| 30 | 30 | Jermey Parnell | 80.36 | 70.50 | 82.76 | 983 | Jaguars |
| 31 | 31 | Eugene Monroe | 80.24 | 72.90 | 80.96 | 316 | Ravens |
| 32 | 32 | Tyler Polumbus | 80.00 | 68.40 | 83.57 | 156 | Broncos |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Kelvin Beachum | 79.52 | 68.00 | 83.03 | 324 | Steelers |
| 34 | 2 | Denzelle Good | 79.47 | 67.80 | 83.09 | 275 | Colts |
| 35 | 3 | Matt Kalil | 79.14 | 68.20 | 82.26 | 1072 | Vikings |
| 36 | 4 | Doug Free | 79.14 | 67.20 | 82.94 | 1020 | Cowboys |
| 37 | 5 | Nate Solder | 79.10 | 68.90 | 81.73 | 225 | Patriots |
| 38 | 6 | Charles Leno Jr. | 78.48 | 68.50 | 80.97 | 917 | Bears |
| 39 | 7 | Branden Albert | 78.46 | 70.00 | 79.94 | 787 | Dolphins |
| 40 | 8 | Chris Clark | 78.43 | 67.10 | 81.81 | 492 | Texans |
| 41 | 9 | Sebastian Vollmer | 78.26 | 68.80 | 80.40 | 951 | Patriots |
| 42 | 10 | Sam Young | 78.11 | 67.20 | 81.21 | 233 | Jaguars |
| 43 | 11 | Ty Nsekhe | 77.68 | 72.60 | 76.90 | 192 | Commanders |
| 44 | 12 | Tom Compton | 77.60 | 68.90 | 79.24 | 231 | Commanders |
| 45 | 13 | Mike Remmers | 76.69 | 66.20 | 79.52 | 1305 | Panthers |
| 46 | 14 | Michael Ola | 76.63 | 66.50 | 79.22 | 450 | Lions |
| 47 | 15 | Jamon Meredith | 76.61 | 65.40 | 79.91 | 393 | Titans |
| 48 | 16 | King Dunlap | 76.17 | 67.30 | 77.91 | 310 | Chargers |
| 49 | 17 | Cyrus Kouandjio | 76.12 | 61.00 | 82.04 | 226 | Bills |
| 50 | 18 | Michael Oher | 75.92 | 65.60 | 78.63 | 1287 | Panthers |
| 51 | 19 | Khalif Barnes | 75.52 | 62.60 | 79.96 | 110 | Raiders |
| 52 | 20 | Dennis Kelly | 75.26 | 65.20 | 77.80 | 395 | Eagles |
| 53 | 21 | Bryan Bulaga | 74.84 | 65.30 | 77.03 | 952 | Packers |
| 54 | 22 | Ja'Wuan James | 74.80 | 63.40 | 78.23 | 391 | Dolphins |
| 55 | 23 | Rick Wagner | 74.65 | 64.30 | 77.38 | 1126 | Ravens |
| 56 | 24 | Donovan Smith | 74.14 | 62.60 | 77.66 | 1087 | Buccaneers |

### Starter (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 57 | 1 | Donald Stephenson | 73.72 | 60.70 | 78.23 | 700 | Chiefs |
| 58 | 2 | Greg Robinson | 73.69 | 60.90 | 78.05 | 959 | Rams |
| 59 | 3 | D'Brickashaw Ferguson | 73.43 | 62.40 | 76.61 | 1107 | Jets |
| 60 | 4 | Michael Schofield III | 73.35 | 62.40 | 76.48 | 1068 | Broncos |
| 61 | 5 | Ryan Harris | 73.26 | 62.00 | 76.60 | 1196 | Broncos |
| 62 | 6 | Garry Gilliam | 73.04 | 60.90 | 76.97 | 1180 | Seahawks |
| 63 | 7 | Eric Winston | 73.00 | 61.90 | 76.23 | 172 | Bengals |
| 64 | 8 | Cornelius Lucas | 72.77 | 59.00 | 77.79 | 323 | Lions |
| 65 | 9 | Jeremiah Poutasi | 72.25 | 59.90 | 76.32 | 394 | Titans |
| 66 | 10 | Alejandro Villanueva | 71.95 | 61.40 | 74.82 | 890 | Steelers |
| 67 | 11 | Breno Giacomini | 71.32 | 59.00 | 75.36 | 1104 | Jets |
| 68 | 12 | Erik Pears | 70.69 | 58.60 | 74.59 | 1007 | 49ers |
| 69 | 13 | Trent Brown | 70.58 | 64.10 | 70.74 | 186 | 49ers |
| 70 | 14 | Marcus Cannon | 70.39 | 56.40 | 75.55 | 758 | Patriots |
| 71 | 15 | Bobby Hart | 70.36 | 56.20 | 75.63 | 153 | Giants |
| 72 | 16 | Jah Reid | 70.21 | 56.10 | 75.45 | 738 | Chiefs |
| 73 | 17 | T.J. Clemmings | 70.13 | 58.50 | 73.72 | 1073 | Vikings |
| 74 | 18 | Chris Hairston | 70.11 | 57.40 | 74.41 | 783 | Chargers |
| 75 | 19 | Ty Sambrailo | 69.04 | 54.70 | 74.43 | 207 | Broncos |
| 76 | 20 | Gosder Cherilus | 68.94 | 54.60 | 74.34 | 886 | Buccaneers |
| 77 | 21 | Ereck Flowers | 68.84 | 55.00 | 73.90 | 958 | Giants |
| 78 | 22 | Cam Fleming | 68.80 | 55.10 | 73.76 | 468 | Patriots |
| 79 | 23 | Jordan Mills | 68.78 | 54.30 | 74.26 | 355 | Bills |
| 80 | 24 | Marshall Newhouse | 67.89 | 54.10 | 72.91 | 933 | Giants |
| 81 | 25 | Seantrel Henderson | 67.51 | 51.30 | 74.15 | 592 | Bills |
| 82 | 26 | Jake Fisher | 66.21 | 56.20 | 68.71 | 132 | Bengals |
| 83 | 27 | Kendall Lamm | 64.17 | 49.80 | 69.58 | 262 | Texans |
| 84 | 28 | James Hurst | 62.84 | 45.90 | 69.97 | 569 | Ravens |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 85 | 1 | Don Barclay | 60.98 | 43.40 | 68.53 | 421 | Packers |
| 86 | 2 | LaAdrian Waddle | 59.56 | 38.90 | 69.17 | 409 | Patriots |

## TE — Tight End

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 87.42 | 91.30 | 80.66 | 720 | Patriots |
| 2 | 2 | Greg Olsen | 86.10 | 90.40 | 79.06 | 623 | Panthers |
| 3 | 3 | Jordan Reed | 85.14 | 90.10 | 77.67 | 499 | Commanders |
| 4 | 4 | Delanie Walker | 83.04 | 88.00 | 75.57 | 520 | Titans |
| 5 | 5 | Zach Miller | 82.36 | 86.30 | 75.56 | 256 | Bears |
| 6 | 6 | Tyler Eifert | 80.66 | 86.10 | 72.87 | 485 | Bengals |
| 7 | 7 | Craig Stevens | 80.31 | 88.90 | 70.42 | 156 | Titans |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Zach Ertz | 78.93 | 82.10 | 72.65 | 532 | Eagles |
| 9 | 2 | Brent Celek | 78.67 | 83.20 | 71.48 | 261 | Eagles |
| 10 | 3 | Austin Seferian-Jenkins | 78.32 | 76.50 | 75.36 | 162 | Buccaneers |
| 11 | 4 | Jimmy Graham | 77.19 | 79.00 | 71.82 | 368 | Seahawks |
| 12 | 5 | Travis Kelce | 76.24 | 73.70 | 73.77 | 646 | Chiefs |
| 13 | 6 | Nick Boyle | 76.08 | 74.90 | 72.70 | 164 | Ravens |
| 14 | 7 | Rhett Ellison | 76.02 | 76.80 | 71.33 | 154 | Vikings |
| 15 | 8 | Garrett Celek | 75.80 | 66.90 | 77.56 | 209 | 49ers |
| 16 | 9 | Benjamin Watson | 75.76 | 75.10 | 72.03 | 622 | Saints |
| 17 | 10 | Antonio Gates | 75.12 | 77.90 | 69.10 | 389 | Chargers |
| 18 | 11 | Charles Clay | 74.97 | 73.20 | 71.99 | 435 | Bills |
| 19 | 12 | Jason Witten | 74.41 | 66.50 | 75.52 | 602 | Cowboys |
| 20 | 13 | Virgil Green | 74.39 | 72.80 | 71.29 | 208 | Broncos |

### Starter (55 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Will Tye | 73.86 | 67.40 | 74.00 | 324 | Giants |
| 22 | 2 | Marcedes Lewis | 73.75 | 67.00 | 74.08 | 402 | Jaguars |
| 23 | 3 | Cameron Brate | 73.68 | 66.80 | 74.10 | 228 | Buccaneers |
| 24 | 4 | Gary Barnidge | 73.28 | 74.60 | 68.24 | 664 | Browns |
| 25 | 5 | Martellus Bennett | 73.21 | 73.00 | 69.19 | 405 | Bears |
| 26 | 6 | Jermaine Gresham | 72.89 | 73.40 | 68.39 | 291 | Cardinals |
| 27 | 7 | Ladarius Green | 72.78 | 66.70 | 72.66 | 447 | Chargers |
| 28 | 8 | Heath Miller | 72.77 | 67.20 | 72.32 | 681 | Steelers |
| 29 | 9 | Vernon Davis | 72.72 | 61.60 | 75.97 | 389 | Broncos |
| 30 | 10 | Scott Chandler | 72.68 | 66.20 | 72.83 | 255 | Patriots |
| 31 | 11 | Coby Fleener | 72.45 | 64.50 | 73.59 | 520 | Colts |
| 32 | 12 | Tyler Kroft | 72.37 | 64.90 | 73.19 | 162 | Bengals |
| 33 | 13 | Luke Willson | 72.18 | 70.10 | 69.40 | 263 | Seahawks |
| 34 | 14 | Darren Fells | 72.12 | 67.90 | 70.77 | 379 | Cardinals |
| 35 | 15 | Clive Walford | 71.84 | 71.70 | 67.77 | 243 | Raiders |
| 36 | 16 | Richard Rodgers | 71.40 | 70.70 | 67.70 | 540 | Packers |
| 37 | 17 | Julius Thomas | 71.38 | 63.20 | 72.66 | 414 | Jaguars |
| 38 | 18 | Jesse James | 71.35 | 71.60 | 67.02 | 130 | Steelers |
| 39 | 19 | John Phillips | 71.32 | 73.90 | 65.44 | 101 | Chargers |
| 40 | 20 | Vance McDonald | 71.30 | 65.90 | 70.73 | 268 | 49ers |
| 41 | 21 | Owen Daniels | 71.12 | 65.20 | 70.90 | 596 | Broncos |
| 42 | 22 | Derek Carrier | 71.04 | 63.80 | 71.70 | 189 | Commanders |
| 43 | 23 | Anthony Fasano | 71.01 | 69.30 | 67.98 | 308 | Titans |
| 44 | 24 | Jordan Cameron | 70.35 | 61.80 | 71.88 | 476 | Dolphins |
| 45 | 25 | Kyle Rudolph | 70.33 | 71.40 | 65.45 | 485 | Vikings |
| 46 | 26 | Lance Kendricks | 69.89 | 62.70 | 70.52 | 211 | Rams |
| 47 | 27 | Gavin Escobar | 69.78 | 58.80 | 72.93 | 107 | Cowboys |
| 48 | 28 | Jacob Tamme | 69.59 | 66.90 | 67.21 | 490 | Falcons |
| 49 | 29 | Lee Smith | 69.31 | 69.30 | 65.15 | 219 | Raiders |
| 50 | 30 | Clay Harbor | 69.28 | 59.60 | 71.56 | 154 | Jaguars |
| 51 | 31 | Josh Hill | 68.88 | 59.80 | 70.77 | 223 | Saints |
| 52 | 32 | C.J. Fiedorowicz | 68.64 | 62.60 | 68.50 | 308 | Texans |
| 53 | 33 | Jared Cook | 68.55 | 57.40 | 71.81 | 393 | Rams |
| 54 | 34 | Larry Donnell | 68.48 | 58.40 | 71.03 | 219 | Giants |
| 55 | 35 | Luke Stocker | 68.21 | 66.00 | 65.51 | 173 | Buccaneers |
| 56 | 36 | Brandon Myers | 67.96 | 60.20 | 68.97 | 172 | Buccaneers |
| 57 | 37 | Mychal Rivera | 67.96 | 54.30 | 72.90 | 220 | Raiders |
| 58 | 38 | Jeff Cumberland | 67.90 | 55.60 | 71.94 | 125 | Jets |
| 59 | 39 | Matt Spaeth | 67.81 | 61.70 | 67.71 | 101 | Steelers |
| 60 | 40 | Cooper Helfet | 67.63 | 57.40 | 70.29 | 161 | Seahawks |
| 61 | 41 | Dwayne Allen | 67.23 | 57.10 | 69.82 | 266 | Colts |
| 62 | 42 | Eric Ebron | 67.08 | 65.10 | 64.23 | 437 | Lions |
| 63 | 43 | Michael Williams | 66.95 | 63.10 | 65.35 | 206 | Patriots |
| 64 | 44 | Michael Hoomanawanui | 66.89 | 59.40 | 67.72 | 120 | Saints |
| 65 | 45 | Chris Gragg | 66.72 | 62.40 | 65.43 | 211 | Bills |
| 66 | 46 | Ed Dickson | 66.63 | 57.40 | 68.61 | 270 | Panthers |
| 67 | 47 | Maxx Williams | 66.13 | 62.80 | 64.19 | 316 | Ravens |
| 68 | 48 | Brandon Pettigrew | 65.99 | 58.20 | 67.01 | 146 | Lions |
| 69 | 49 | Demetrius Harris | 64.74 | 59.70 | 63.94 | 189 | Chiefs |
| 70 | 50 | Tim Wright | 64.64 | 47.70 | 71.76 | 137 | Lions |
| 71 | 51 | Dion Sims | 64.09 | 60.20 | 62.52 | 222 | Dolphins |
| 72 | 52 | Ryan Griffin | 63.50 | 55.00 | 65.00 | 264 | Texans |
| 73 | 53 | Levine Toilolo | 62.91 | 57.00 | 62.68 | 241 | Falcons |
| 74 | 54 | Jim Dray | 62.36 | 48.90 | 67.17 | 212 | Browns |
| 75 | 55 | Blake Bell | 62.00 | 50.40 | 65.57 | 232 | 49ers |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 76 | 1 | Kellen Davis | 61.44 | 49.20 | 65.44 | 187 | Jets |
| 77 | 2 | Garrett Graham | 59.16 | 40.20 | 67.64 | 208 | Texans |

## WR — Wide Receiver

- **Season used:** `2015`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Julio Jones | 89.17 | 92.50 | 82.79 | 657 | Falcons |
| 2 | 2 | Antonio Brown | 88.96 | 92.50 | 82.43 | 725 | Steelers |
| 3 | 3 | Odell Beckham Jr. | 88.28 | 88.10 | 84.24 | 634 | Giants |
| 4 | 4 | Alshon Jeffery | 87.88 | 92.00 | 80.97 | 295 | Bears |
| 5 | 5 | A.J. Green | 87.42 | 90.40 | 81.26 | 626 | Bengals |
| 6 | 6 | J.J. Nelson | 87.34 | 82.30 | 86.53 | 128 | Cardinals |
| 7 | 7 | Sammy Watkins | 86.72 | 89.80 | 80.50 | 419 | Bills |
| 8 | 8 | DeAndre Hopkins | 85.70 | 90.60 | 78.26 | 738 | Texans |
| 9 | 9 | Steve Smith | 85.58 | 90.30 | 78.26 | 249 | Ravens |
| 10 | 10 | Calvin Johnson | 85.30 | 88.60 | 78.94 | 688 | Lions |
| 11 | 11 | Doug Baldwin | 84.42 | 88.20 | 77.74 | 622 | Seahawks |
| 12 | 12 | Larry Fitzgerald | 84.24 | 89.40 | 76.63 | 694 | Cardinals |
| 13 | 13 | Allen Robinson II | 84.07 | 84.00 | 79.95 | 682 | Jaguars |
| 14 | 14 | DeSean Jackson | 83.62 | 74.80 | 85.33 | 302 | Commanders |
| 15 | 15 | Emmanuel Sanders | 83.59 | 84.40 | 78.88 | 661 | Broncos |
| 16 | 16 | Mike Evans | 82.59 | 83.60 | 77.75 | 552 | Buccaneers |
| 17 | 17 | T.Y. Hilton | 82.41 | 78.90 | 80.58 | 654 | Colts |
| 18 | 18 | Jarvis Landry | 81.98 | 88.00 | 73.80 | 582 | Dolphins |
| 19 | 19 | Brandon Marshall | 81.92 | 85.40 | 75.43 | 665 | Jets |
| 20 | 20 | Jeremy Maclin | 81.03 | 78.30 | 78.69 | 580 | Chiefs |
| 21 | 21 | Michael Floyd | 80.88 | 80.00 | 77.30 | 525 | Cardinals |
| 22 | 22 | Vincent Jackson | 80.61 | 79.80 | 76.98 | 313 | Buccaneers |
| 23 | 23 | Stefon Diggs | 80.56 | 78.40 | 77.83 | 449 | Vikings |
| 24 | 24 | John Brown | 80.53 | 78.70 | 77.58 | 632 | Cardinals |
| 25 | 25 | Jeff Janis | 80.25 | 70.40 | 82.65 | 111 | Packers |
| 26 | 26 | Allen Hurns | 80.23 | 78.80 | 77.02 | 611 | Jaguars |

### Good (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | Demaryius Thomas | 79.79 | 76.40 | 77.89 | 715 | Broncos |
| 28 | 2 | Willie Snead IV | 79.07 | 74.10 | 78.22 | 546 | Saints |
| 29 | 3 | Keenan Allen | 78.97 | 79.20 | 74.65 | 359 | Chargers |
| 30 | 4 | Martavis Bryant | 78.78 | 72.60 | 78.73 | 478 | Steelers |
| 31 | 5 | Julian Edelman | 78.60 | 83.90 | 70.90 | 478 | Patriots |
| 32 | 6 | Eric Decker | 78.45 | 78.00 | 74.59 | 588 | Jets |
| 33 | 7 | Tyler Lockett | 77.99 | 75.90 | 75.21 | 515 | Seahawks |
| 34 | 8 | Ted Ginn Jr. | 77.95 | 72.10 | 77.69 | 494 | Panthers |
| 35 | 9 | Golden Tate | 77.66 | 77.40 | 73.67 | 676 | Lions |
| 36 | 10 | Kamar Aiken | 77.66 | 78.50 | 72.94 | 621 | Ravens |
| 37 | 11 | DeVante Parker | 77.51 | 72.90 | 76.42 | 312 | Dolphins |
| 38 | 12 | Brandin Cooks | 77.50 | 73.10 | 76.27 | 673 | Saints |
| 39 | 13 | Terrance Williams | 77.12 | 69.90 | 77.77 | 548 | Cowboys |
| 40 | 14 | Rishard Matthews | 77.08 | 74.00 | 74.96 | 338 | Dolphins |
| 41 | 15 | Kenny Britt | 77.02 | 72.60 | 75.80 | 387 | Rams |
| 42 | 16 | Dorial Green-Beckham | 76.98 | 71.10 | 76.73 | 432 | Titans |
| 43 | 17 | Devin Funchess | 76.75 | 74.70 | 73.95 | 287 | Panthers |
| 44 | 18 | Dez Bryant | 76.49 | 68.70 | 77.51 | 311 | Cowboys |
| 45 | 19 | Marvin Jones Jr. | 76.46 | 72.90 | 74.67 | 599 | Bengals |
| 46 | 20 | Anquan Boldin | 76.29 | 72.00 | 74.99 | 519 | 49ers |
| 47 | 21 | Pierre Garcon | 75.98 | 75.10 | 72.40 | 584 | Commanders |
| 48 | 22 | Jerricho Cotchery | 75.64 | 76.10 | 71.17 | 307 | Panthers |
| 49 | 23 | Jermaine Kearse | 75.47 | 71.10 | 74.21 | 592 | Seahawks |
| 50 | 24 | Bryan Walters | 75.46 | 75.20 | 71.46 | 258 | Jaguars |
| 51 | 25 | Michael Crabtree | 75.27 | 77.20 | 69.81 | 595 | Raiders |
| 52 | 26 | Corey Brown | 75.16 | 70.00 | 74.44 | 531 | Panthers |
| 53 | 27 | Kendall Wright | 75.09 | 69.60 | 74.58 | 290 | Titans |
| 54 | 28 | Donte Moncrief | 74.89 | 72.10 | 72.58 | 562 | Colts |
| 55 | 29 | James Jones | 74.85 | 69.70 | 74.11 | 800 | Packers |
| 56 | 30 | Brandon Coleman | 74.71 | 71.80 | 72.48 | 317 | Saints |
| 57 | 31 | Amari Cooper | 74.70 | 71.20 | 72.87 | 607 | Raiders |
| 58 | 32 | Brice Butler | 74.65 | 68.00 | 74.91 | 174 | Cowboys |
| 59 | 33 | Markus Wheaton | 74.57 | 68.40 | 74.52 | 571 | Steelers |
| 60 | 34 | Travis Benjamin | 74.56 | 67.00 | 75.43 | 628 | Browns |
| 61 | 35 | Jarius Wright | 74.50 | 68.30 | 74.47 | 329 | Vikings |

### Starter (81 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 62 | 1 | Marc Mariani | 73.81 | 67.50 | 73.85 | 336 | Bears |
| 63 | 2 | Torrey Smith | 73.63 | 62.80 | 76.68 | 533 | 49ers |
| 64 | 3 | Danny Amendola | 73.48 | 75.20 | 68.17 | 517 | Patriots |
| 65 | 4 | Randall Cobb | 73.42 | 68.50 | 72.53 | 723 | Packers |
| 66 | 5 | Adam Thielen | 72.88 | 65.20 | 73.84 | 109 | Vikings |
| 67 | 6 | Malcom Floyd | 72.86 | 61.70 | 76.13 | 559 | Chargers |
| 68 | 7 | Jordan Matthews | 72.68 | 65.50 | 73.30 | 606 | Eagles |
| 69 | 8 | Bennie Fowler | 72.47 | 66.80 | 72.08 | 201 | Broncos |
| 70 | 9 | Rueben Randle | 72.44 | 63.90 | 73.97 | 651 | Giants |
| 71 | 10 | Percy Harvin | 72.20 | 67.50 | 71.16 | 152 | Bills |
| 72 | 11 | Andre Johnson | 72.05 | 66.20 | 71.78 | 490 | Colts |
| 73 | 12 | Jeremy Butler | 71.74 | 64.90 | 72.13 | 273 | Ravens |
| 74 | 13 | Kenny Stills | 71.66 | 59.90 | 75.33 | 430 | Dolphins |
| 75 | 14 | Jamison Crowder | 71.51 | 69.50 | 68.69 | 526 | Commanders |
| 76 | 15 | Nate Washington | 71.45 | 64.50 | 71.92 | 524 | Texans |
| 77 | 16 | Brian Hartline | 71.41 | 68.20 | 69.38 | 386 | Browns |
| 78 | 17 | Quinton Patton | 71.33 | 65.70 | 70.92 | 320 | 49ers |
| 79 | 18 | Marques Colston | 71.26 | 62.80 | 72.73 | 416 | Saints |
| 80 | 19 | Andrew Hawkins | 71.24 | 59.70 | 74.76 | 286 | Browns |
| 81 | 20 | Cole Beasley | 71.18 | 64.30 | 71.60 | 438 | Cowboys |
| 82 | 21 | Justin Hunter | 70.90 | 63.10 | 71.93 | 252 | Titans |
| 83 | 22 | Chris Givens | 70.89 | 59.50 | 74.32 | 318 | Ravens |
| 84 | 23 | Albert Wilson | 70.85 | 62.50 | 72.25 | 495 | Chiefs |
| 85 | 24 | Stedman Bailey | 70.79 | 59.40 | 74.21 | 155 | Rams |
| 86 | 25 | Phillip Dorsett | 70.76 | 60.50 | 73.44 | 150 | Colts |
| 87 | 26 | Marquess Wilson | 70.68 | 63.50 | 71.30 | 392 | Bears |
| 88 | 27 | T.J. Jones | 70.41 | 62.10 | 71.79 | 111 | Lions |
| 89 | 28 | Jaelen Strong | 70.27 | 64.80 | 69.75 | 190 | Texans |
| 90 | 29 | Chris Matthews | 70.20 | 63.00 | 70.84 | 152 | Ravens |
| 91 | 30 | Tavon Austin | 70.20 | 69.50 | 66.50 | 432 | Rams |
| 92 | 31 | Steve Johnson | 70.10 | 63.20 | 70.54 | 406 | Chargers |
| 93 | 32 | Louis Murphy Jr. | 69.94 | 64.30 | 69.54 | 150 | Buccaneers |
| 94 | 33 | Andre Holmes | 69.37 | 57.80 | 72.92 | 208 | Raiders |
| 95 | 34 | Harry Douglas | 69.28 | 60.10 | 71.24 | 465 | Titans |
| 96 | 35 | Jeremy Kerley | 68.91 | 63.40 | 68.42 | 172 | Jets |
| 97 | 36 | Darrius Heyward-Bey | 68.85 | 60.70 | 70.11 | 308 | Steelers |
| 98 | 37 | Josh Huff | 68.60 | 64.20 | 67.37 | 341 | Eagles |
| 99 | 38 | Riley Cooper | 68.60 | 58.20 | 71.36 | 357 | Eagles |
| 100 | 39 | Mike Wallace | 68.50 | 58.70 | 70.87 | 517 | Vikings |
| 101 | 40 | Nick Williams | 68.44 | 65.00 | 66.56 | 125 | Falcons |
| 102 | 41 | Kenbrell Thompkins | 68.42 | 60.50 | 69.54 | 221 | Jets |
| 103 | 42 | Dontrelle Inman | 68.38 | 57.40 | 71.54 | 474 | Chargers |
| 104 | 43 | Seth Roberts | 68.38 | 59.90 | 69.86 | 446 | Raiders |
| 105 | 44 | Robert Woods | 68.22 | 61.80 | 68.33 | 478 | Bills |
| 106 | 45 | Aaron Dobson | 68.17 | 60.40 | 69.18 | 153 | Patriots |
| 107 | 46 | Eddie Royal | 68.10 | 59.40 | 69.74 | 284 | Bears |
| 108 | 47 | Jaron Brown | 67.95 | 62.70 | 67.28 | 174 | Cardinals |
| 109 | 48 | Marqise Lee | 67.87 | 61.60 | 67.89 | 183 | Jaguars |
| 110 | 49 | Cecil Shorts | 67.80 | 60.40 | 68.57 | 411 | Texans |
| 111 | 50 | Jared Abbrederis | 67.78 | 61.30 | 67.94 | 133 | Packers |
| 112 | 51 | Roddy White | 67.62 | 62.60 | 66.80 | 629 | Falcons |
| 113 | 52 | Taylor Gabriel | 67.60 | 56.60 | 70.76 | 292 | Browns |
| 114 | 53 | Keshawn Martin | 67.57 | 59.40 | 68.85 | 369 | Patriots |
| 115 | 54 | Adam Humphries | 67.51 | 63.50 | 66.02 | 317 | Buccaneers |
| 116 | 55 | Chris Hogan | 67.44 | 60.30 | 68.03 | 394 | Bills |
| 117 | 56 | Leonard Hankerson | 67.34 | 57.50 | 69.74 | 252 | Bills |
| 118 | 57 | Josh Bellamy | 67.25 | 60.70 | 67.45 | 226 | Bears |
| 119 | 58 | Devin Street | 67.09 | 57.80 | 69.11 | 137 | Cowboys |
| 120 | 59 | Brandon LaFell | 67.01 | 56.40 | 69.91 | 503 | Patriots |
| 121 | 60 | Chris Conley | 66.89 | 61.10 | 66.59 | 291 | Chiefs |
| 122 | 61 | Corey Fuller | 66.68 | 54.90 | 70.37 | 114 | Lions |
| 123 | 62 | Mohamed Sanu | 66.53 | 57.00 | 68.72 | 480 | Bengals |
| 124 | 63 | Bradley Marquez | 66.21 | 63.30 | 63.99 | 120 | Rams |
| 125 | 64 | Brian Quick | 66.16 | 53.90 | 70.16 | 169 | Rams |
| 126 | 65 | Dwayne Harris | 66.16 | 61.00 | 65.43 | 435 | Giants |
| 127 | 66 | Ryan Grant | 66.11 | 60.90 | 65.42 | 267 | Commanders |
| 128 | 67 | Greg Jennings | 66.06 | 53.40 | 70.33 | 217 | Dolphins |
| 129 | 68 | Justin Hardy | 66.02 | 63.50 | 63.53 | 238 | Falcons |
| 130 | 69 | Hakeem Nicks | 66.01 | 51.80 | 71.32 | 115 | Giants |
| 131 | 70 | Lance Moore | 65.99 | 56.80 | 67.95 | 463 | Lions |
| 132 | 71 | Jason Avant | 65.96 | 59.60 | 66.03 | 291 | Chiefs |
| 133 | 72 | Davante Adams | 65.36 | 59.10 | 65.36 | 550 | Packers |
| 134 | 73 | Jordan Norwood | 64.77 | 55.60 | 66.71 | 332 | Broncos |
| 135 | 74 | Quincy Enunwa | 64.43 | 53.30 | 67.69 | 350 | Jets |
| 136 | 75 | Cody Latimer | 64.26 | 61.30 | 62.07 | 111 | Broncos |
| 137 | 76 | Andre Roberts | 63.56 | 51.30 | 67.57 | 166 | Commanders |
| 138 | 77 | Javontee Herndon | 63.53 | 59.70 | 61.91 | 206 | Chargers |
| 139 | 78 | Myles White | 62.89 | 57.20 | 62.51 | 113 | Giants |
| 140 | 79 | Devin Smith | 62.64 | 52.30 | 65.36 | 240 | Jets |
| 141 | 80 | Donteea Dye Jr. | 62.18 | 53.70 | 63.66 | 272 | Buccaneers |
| 142 | 81 | Russell Shepard | 62.09 | 58.00 | 60.65 | 110 | Buccaneers |

### Rotation/backup (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 143 | 1 | Marlon Moore | 61.51 | 54.20 | 62.21 | 141 | Browns |
| 144 | 2 | Nelson Agholor | 61.50 | 48.70 | 65.86 | 430 | Eagles |
| 145 | 3 | Darius Jennings | 61.39 | 61.00 | 57.49 | 133 | Browns |
| 146 | 4 | Andre Caldwell | 61.17 | 55.50 | 60.79 | 183 | Broncos |
| 147 | 5 | Marlon Brown | 61.17 | 50.10 | 64.39 | 254 | Ravens |
| 148 | 6 | Rashad Greene Sr. | 59.27 | 56.30 | 57.08 | 134 | Jaguars |
