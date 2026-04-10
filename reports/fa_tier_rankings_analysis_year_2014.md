# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:38:21Z
- **Requested analysis_year:** 2014 (clamped to 2014)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Travis Frederick | 94.52 | 89.30 | 93.83 | 1178 | Cowboys |
| 2 | 2 | Nick Mangold | 91.38 | 84.55 | 91.77 | 979 | Jets |
| 3 | 3 | Jason Kelce | 86.53 | 77.29 | 88.53 | 828 | Eagles |
| 4 | 4 | Maurkice Pouncey | 84.42 | 76.30 | 85.66 | 1178 | Steelers |
| 5 | 5 | Jeremy Zuttah | 84.36 | 76.20 | 85.63 | 1201 | Ravens |
| 6 | 6 | Corey Linsley | 84.29 | 76.80 | 85.11 | 1188 | Packers |
| 7 | 7 | Stefen Wisniewski | 84.14 | 74.40 | 86.47 | 1014 | Raiders |
| 8 | 8 | Max Unger | 84.08 | 74.66 | 86.19 | 548 | Seahawks |
| 9 | 9 | Rodney Hudson | 83.97 | 76.30 | 84.92 | 1006 | Chiefs |
| 10 | 10 | Chris Myers | 83.32 | 74.80 | 84.84 | 1102 | Texans |
| 11 | 11 | Will Montgomery | 83.11 | 73.14 | 85.59 | 651 | Broncos |
| 12 | 12 | Eric Wood | 82.25 | 73.70 | 83.79 | 1058 | Bills |
| 13 | 13 | Alex Mack | 81.78 | 70.50 | 85.13 | 297 | Browns |
| 14 | 14 | Kory Lichtensteiger | 81.48 | 73.20 | 82.84 | 1052 | Commanders |
| 15 | 15 | Ryan Kalil | 81.41 | 73.10 | 82.78 | 1225 | Panthers |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Evan Smith | 79.82 | 67.47 | 83.89 | 930 | Buccaneers |
| 17 | 2 | Roberto Garza | 79.36 | 69.50 | 81.77 | 747 | Bears |
| 18 | 3 | John Sullivan | 79.23 | 70.17 | 81.11 | 975 | Vikings |
| 19 | 4 | J.D. Walton | 78.07 | 67.50 | 80.95 | 1118 | Giants |
| 20 | 5 | Daniel Kilgore | 77.71 | 72.17 | 77.23 | 450 | 49ers |
| 21 | 6 | Brian De La Puente | 77.58 | 67.57 | 80.09 | 488 | Bears |
| 22 | 7 | Jonathan Goodwin | 77.55 | 67.28 | 80.23 | 853 | Saints |
| 23 | 8 | A.Q. Shipley | 76.66 | 67.99 | 78.28 | 423 | Colts |
| 24 | 9 | Russell Bodine | 76.21 | 65.60 | 79.11 | 1124 | Bengals |
| 25 | 10 | Bryan Stork | 75.39 | 64.28 | 78.63 | 891 | Patriots |
| 26 | 11 | Luke Bowanko | 75.26 | 64.16 | 78.49 | 922 | Jaguars |
| 27 | 12 | Brian Schwenke | 74.83 | 62.92 | 78.61 | 647 | Titans |
| 28 | 13 | Samson Satele | 74.77 | 63.40 | 78.19 | 1075 | Dolphins |
| 29 | 14 | Dominic Raiola | 74.44 | 63.60 | 77.50 | 1091 | Lions |

### Starter (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Jonotthan Harrison | 73.82 | 60.92 | 78.25 | 693 | Colts |
| 31 | 2 | Trevor Robinson | 72.39 | 60.25 | 76.32 | 169 | Chargers |
| 32 | 3 | David Molk | 71.77 | 62.17 | 74.00 | 403 | Eagles |
| 33 | 4 | Peter Konz | 70.16 | 58.49 | 73.77 | 183 | Falcons |
| 34 | 5 | Joe Hawley | 70.13 | 61.39 | 71.79 | 243 | Falcons |
| 35 | 6 | Cody Wallace | 69.69 | 58.56 | 72.94 | 148 | Steelers |
| 36 | 7 | Scott Wells | 69.37 | 55.11 | 74.71 | 979 | Rams |
| 37 | 8 | Dalton Freeman | 68.29 | 57.76 | 71.15 | 110 | Jets |
| 38 | 9 | Khaled Holmes | 67.74 | 56.17 | 71.28 | 376 | Colts |
| 39 | 10 | James Stone | 67.67 | 60.00 | 68.61 | 672 | Falcons |
| 40 | 11 | Travis Swanson | 67.14 | 56.74 | 69.91 | 373 | Lions |
| 41 | 12 | Lyle Sendlein | 66.32 | 53.90 | 70.44 | 1104 | Cardinals |
| 42 | 13 | Rich Ohrnberger | 65.56 | 57.44 | 66.80 | 447 | Chargers |
| 43 | 14 | Doug Legursky | 65.21 | 52.28 | 69.67 | 133 | Chargers |
| 44 | 15 | Patrick Lewis | 64.83 | 55.83 | 66.67 | 274 | Seahawks |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 45 | 1 | Nick McDonald | 61.32 | 46.06 | 67.32 | 469 | Browns |

## CB — Cornerback

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Vontae Davis | 94.12 | 92.50 | 92.06 | 1022 | Colts |
| 2 | 2 | Richard Sherman | 93.37 | 89.50 | 91.78 | 1177 | Seahawks |
| 3 | 3 | Chris Harris Jr. | 92.63 | 92.20 | 88.75 | 1056 | Broncos |
| 4 | 4 | Sean Smith | 88.53 | 87.50 | 85.05 | 1038 | Chiefs |
| 5 | 5 | K'Waun Williams | 87.98 | 79.74 | 92.44 | 350 | Browns |
| 6 | 6 | Darrelle Revis | 87.23 | 83.20 | 88.66 | 1188 | Patriots |
| 7 | 7 | Casey Hayward Jr. | 86.71 | 82.06 | 89.71 | 461 | Packers |
| 8 | 8 | Kareem Jackson | 85.80 | 85.41 | 84.08 | 774 | Texans |
| 9 | 9 | Desmond Trufant | 85.05 | 79.00 | 84.91 | 1076 | Falcons |
| 10 | 10 | Orlando Scandrick | 82.69 | 81.90 | 80.08 | 1000 | Cowboys |
| 11 | 11 | Rashean Mathis | 82.41 | 80.30 | 80.80 | 1045 | Lions |
| 12 | 12 | Dominique Rodgers-Cromartie | 82.16 | 75.76 | 82.26 | 752 | Giants |
| 13 | 13 | Brandon Flowers | 81.66 | 78.45 | 81.52 | 818 | Chargers |
| 14 | 14 | Jimmy Smith | 81.20 | 76.59 | 84.89 | 461 | Ravens |
| 15 | 15 | Chris Culliver | 80.89 | 75.47 | 81.58 | 822 | 49ers |
| 16 | 16 | Brandon Boykin | 80.87 | 71.87 | 82.71 | 499 | Eagles |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Johnathan Joseph | 79.97 | 74.20 | 79.97 | 859 | Texans |
| 18 | 2 | Xavier Rhodes | 79.97 | 71.20 | 82.81 | 1027 | Vikings |
| 19 | 3 | Jerraud Powers | 79.68 | 75.21 | 80.16 | 774 | Cardinals |
| 20 | 4 | Bene Benwikere | 78.82 | 76.29 | 80.51 | 542 | Panthers |
| 21 | 5 | Joe Haden | 78.58 | 69.70 | 82.21 | 1023 | Browns |
| 22 | 6 | William Gay | 77.88 | 73.90 | 76.36 | 919 | Steelers |
| 23 | 7 | Stephon Gilmore | 76.97 | 73.60 | 77.65 | 838 | Bills |
| 24 | 8 | E.J. Gaines | 76.67 | 70.30 | 77.78 | 937 | Rams |
| 25 | 9 | Trumaine McBride | 76.58 | 74.44 | 82.38 | 212 | Giants |
| 26 | 10 | Tramon Williams | 76.43 | 67.70 | 78.08 | 1138 | Packers |
| 27 | 11 | Malcolm Butler | 76.14 | 67.03 | 85.35 | 217 | Patriots |
| 28 | 12 | Bradley Roby | 75.69 | 69.90 | 75.39 | 844 | Broncos |
| 29 | 13 | Janoris Jenkins | 75.21 | 70.00 | 75.77 | 838 | Rams |
| 30 | 14 | Dre Kirkpatrick | 75.16 | 68.79 | 79.61 | 276 | Bengals |
| 31 | 15 | Darius Slay | 75.07 | 70.00 | 76.23 | 1073 | Lions |
| 32 | 16 | Aqib Talib | 74.95 | 66.10 | 77.83 | 988 | Broncos |
| 33 | 17 | Josh Norman | 74.75 | 71.76 | 77.77 | 730 | Panthers |
| 34 | 18 | Tim Jennings | 74.62 | 67.80 | 75.42 | 1006 | Bears |
| 35 | 19 | Nolan Carroll | 74.48 | 65.99 | 76.70 | 374 | Eagles |
| 36 | 20 | Cary Williams | 74.46 | 65.60 | 76.20 | 1154 | Eagles |
| 37 | 21 | Sterling Moore | 74.17 | 70.70 | 77.00 | 839 | Cowboys |
| 38 | 22 | Jason Verrett | 74.13 | 71.59 | 85.08 | 219 | Chargers |

### Starter (65 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Captain Munnerlyn | 73.30 | 64.70 | 74.86 | 1062 | Vikings |
| 40 | 2 | Alterraun Verner | 73.23 | 63.80 | 76.39 | 859 | Buccaneers |
| 41 | 3 | Perrish Cox | 73.20 | 64.00 | 75.90 | 941 | 49ers |
| 42 | 4 | Brandon Harris | 72.96 | 64.68 | 81.62 | 106 | Titans |
| 43 | 5 | Josh Robinson | 72.58 | 64.60 | 73.73 | 671 | Vikings |
| 44 | 6 | Justin Gilbert | 72.50 | 62.85 | 77.90 | 362 | Browns |
| 45 | 7 | Adam Jones | 72.39 | 65.50 | 72.82 | 838 | Bengals |
| 46 | 8 | Davon House | 72.35 | 65.11 | 76.76 | 405 | Packers |
| 47 | 9 | Prince Amukamara | 72.34 | 64.77 | 78.00 | 458 | Giants |
| 48 | 10 | Leon Hall | 72.30 | 66.90 | 75.38 | 951 | Bengals |
| 49 | 11 | Josh Gordy | 72.25 | 63.88 | 75.74 | 285 | Colts |
| 50 | 12 | Sam Shields | 72.23 | 63.20 | 75.23 | 929 | Packers |
| 51 | 13 | Antonio Cromartie | 72.16 | 62.70 | 74.30 | 1064 | Cardinals |
| 52 | 14 | Jumal Rolle | 72.04 | 65.00 | 79.94 | 202 | Texans |
| 53 | 15 | Brent Grimes | 71.97 | 64.80 | 75.71 | 1014 | Dolphins |
| 54 | 16 | Jason McCourty | 71.77 | 59.40 | 75.85 | 1077 | Titans |
| 55 | 17 | Patrick Robinson | 71.77 | 65.25 | 77.36 | 612 | Saints |
| 56 | 18 | Logan Ryan | 71.37 | 58.75 | 75.61 | 571 | Patriots |
| 57 | 19 | Kyle Arrington | 70.68 | 60.66 | 73.19 | 565 | Patriots |
| 58 | 20 | Byron Maxwell | 70.68 | 60.60 | 76.25 | 842 | Seahawks |
| 59 | 21 | Kayvon Webster | 70.63 | 62.64 | 76.34 | 130 | Broncos |
| 60 | 22 | Trumaine Johnson | 70.42 | 62.39 | 75.46 | 433 | Rams |
| 61 | 23 | Patrick Peterson | 70.40 | 62.00 | 71.83 | 1042 | Cardinals |
| 62 | 24 | Steve Williams | 70.38 | 64.72 | 77.29 | 130 | Chargers |
| 63 | 25 | Mike Harris | 70.28 | 65.37 | 75.74 | 216 | Giants |
| 64 | 26 | T.J. Carrie | 70.22 | 63.89 | 74.44 | 543 | Raiders |
| 65 | 27 | Brice McCain | 70.08 | 63.09 | 72.46 | 659 | Steelers |
| 66 | 28 | Phillip Gaines | 69.91 | 67.03 | 75.99 | 370 | Chiefs |
| 67 | 29 | Tarell Brown | 69.31 | 57.70 | 73.92 | 959 | Raiders |
| 68 | 30 | Nickell Robey-Coleman | 69.10 | 60.19 | 70.87 | 642 | Bills |
| 69 | 31 | Tyler Patmon | 68.96 | 65.28 | 73.50 | 113 | Cowboys |
| 70 | 32 | Johnthan Banks | 68.84 | 62.30 | 69.68 | 910 | Buccaneers |
| 71 | 33 | A.J. Bouye | 68.74 | 64.56 | 74.53 | 634 | Texans |
| 72 | 34 | Jeremy Lane | 68.65 | 64.38 | 74.63 | 274 | Seahawks |
| 73 | 35 | Josh Wilson | 68.61 | 59.75 | 71.92 | 448 | Falcons |
| 74 | 36 | Lardarius Webb | 68.58 | 60.40 | 72.46 | 916 | Ravens |
| 75 | 37 | Valentino Blake | 68.57 | 62.65 | 76.30 | 287 | Steelers |
| 76 | 38 | Demetrius McCray | 68.54 | 63.98 | 70.93 | 815 | Jaguars |
| 77 | 39 | Leodis McKelvin | 68.49 | 62.38 | 73.30 | 535 | Bills |
| 78 | 40 | Marcus Burley | 68.44 | 63.74 | 72.61 | 321 | Seahawks |
| 79 | 41 | Buster Skrine | 68.40 | 56.50 | 72.17 | 1130 | Browns |
| 80 | 42 | Dontae Johnson | 68.26 | 61.54 | 71.71 | 490 | 49ers |
| 81 | 43 | Terence Newman | 67.88 | 58.70 | 71.82 | 907 | Bengals |
| 82 | 44 | Alan Ball | 67.82 | 61.47 | 74.55 | 499 | Jaguars |
| 83 | 45 | Danny Gorrer | 67.49 | 64.05 | 74.26 | 321 | Ravens |
| 84 | 46 | Cortland Finnegan | 66.84 | 62.86 | 70.22 | 705 | Dolphins |
| 85 | 47 | Darryl Morris | 65.99 | 62.75 | 72.59 | 259 | Texans |
| 86 | 48 | Pierre Desir | 65.91 | 63.56 | 83.31 | 118 | Browns |
| 87 | 49 | Kyle Wilson | 65.51 | 57.37 | 67.29 | 309 | Jets |
| 88 | 50 | Marcus Williams | 65.33 | 60.59 | 72.66 | 446 | Jets |
| 89 | 51 | D.J. Hayden | 64.83 | 60.00 | 70.91 | 581 | Raiders |
| 90 | 52 | Darrin Walls | 64.53 | 57.13 | 70.93 | 753 | Jets |
| 91 | 53 | Keenan Lewis | 64.46 | 51.80 | 68.74 | 899 | Saints |
| 92 | 54 | Brandon Carr | 64.35 | 52.20 | 68.29 | 1138 | Cowboys |
| 93 | 55 | Brandon Browner | 64.28 | 52.32 | 73.09 | 719 | Patriots |
| 94 | 56 | Brandon Dixon | 63.98 | 58.44 | 70.80 | 163 | Buccaneers |
| 95 | 57 | Carlos Rogers | 63.95 | 57.46 | 68.79 | 460 | Raiders |
| 96 | 58 | Dee Milliner | 63.91 | 59.32 | 72.44 | 116 | Jets |
| 97 | 59 | Zackary Bowman | 63.87 | 53.87 | 69.80 | 450 | Giants |
| 98 | 60 | Coty Sensabaugh | 63.58 | 56.54 | 66.71 | 721 | Titans |
| 99 | 61 | Bashaud Breeland | 63.49 | 49.90 | 68.39 | 864 | Commanders |
| 100 | 62 | Robert Alford | 63.05 | 51.87 | 70.63 | 617 | Falcons |
| 101 | 63 | DeAngelo Hall | 62.59 | 53.88 | 70.99 | 145 | Commanders |
| 102 | 64 | Bradley Fletcher | 62.19 | 46.90 | 69.37 | 1053 | Eagles |
| 103 | 65 | Marcus Roberson | 62.05 | 61.04 | 77.05 | 130 | Rams |

### Rotation/backup (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 104 | 1 | Aaron Colvin | 61.81 | 64.81 | 69.06 | 277 | Jaguars |
| 105 | 2 | Leonard Johnson | 61.71 | 50.62 | 65.77 | 387 | Buccaneers |
| 106 | 3 | Melvin White | 61.09 | 52.91 | 64.08 | 512 | Panthers |
| 107 | 4 | Greg Toler | 60.84 | 50.10 | 66.97 | 1166 | Colts |
| 108 | 5 | Cassius Vaughn | 60.74 | 52.82 | 64.67 | 425 | Lions |
| 109 | 6 | Robert McClain | 60.72 | 50.58 | 64.34 | 628 | Falcons |
| 110 | 7 | Ron Brooks | 60.45 | 60.04 | 64.57 | 145 | Bills |
| 111 | 8 | Brian Dixon | 60.09 | 62.13 | 57.69 | 163 | Saints |
| 112 | 9 | E.J. Biggers | 59.83 | 46.92 | 65.93 | 452 | Commanders |
| 113 | 10 | Chris Owens | 59.82 | 52.29 | 65.58 | 490 | Chiefs |
| 114 | 11 | Tharold Simon | 59.69 | 49.71 | 66.35 | 410 | Seahawks |
| 115 | 12 | Chykie Brown | 59.61 | 52.19 | 64.56 | 503 | Giants |
| 116 | 13 | Kyle Fuller | 59.06 | 46.30 | 63.40 | 858 | Bears |
| 117 | 14 | Rashaan Melvin | 58.70 | 59.58 | 65.61 | 303 | Ravens |
| 118 | 15 | Dwayne Gratz | 58.69 | 47.10 | 65.25 | 855 | Jaguars |
| 119 | 16 | Will Davis | 58.66 | 58.49 | 65.54 | 137 | Dolphins |
| 120 | 17 | Antoine Cason | 58.55 | 40.36 | 69.95 | 679 | Ravens |
| 121 | 18 | David Amerson | 58.30 | 43.30 | 64.79 | 905 | Commanders |
| 122 | 19 | Alfonzo Dennard | 58.18 | 48.11 | 67.08 | 236 | Patriots |
| 123 | 20 | R.J. Stanford | 57.61 | 58.74 | 64.56 | 136 | Dolphins |
| 124 | 21 | Justin Bethel | 57.34 | 58.80 | 62.72 | 101 | Cardinals |
| 125 | 22 | Phillip Adams | 57.22 | 49.81 | 62.99 | 307 | Jets |
| 126 | 23 | Morris Claiborne | 57.06 | 51.39 | 65.00 | 150 | Cowboys |
| 127 | 24 | Will Blackmon | 56.69 | 49.11 | 64.97 | 355 | Jaguars |
| 128 | 25 | Cortez Allen | 56.53 | 41.42 | 67.44 | 453 | Steelers |
| 129 | 26 | Antonio Allen | 56.13 | 52.42 | 55.47 | 515 | Jets |
| 130 | 27 | Marcus Cooper | 56.07 | 43.43 | 66.19 | 287 | Chiefs |
| 131 | 28 | Isaiah Frey | 55.35 | 56.07 | 55.90 | 227 | Buccaneers |
| 132 | 29 | Corey White | 54.97 | 40.00 | 62.77 | 757 | Saints |
| 133 | 30 | Shareece Wright | 54.96 | 40.00 | 63.58 | 833 | Chargers |
| 134 | 31 | Demontre Hurst | 54.74 | 55.14 | 55.50 | 366 | Bears |
| 135 | 32 | Jayron Hosley | 53.28 | 57.02 | 55.78 | 150 | Giants |
| 136 | 33 | Blidi Wreh-Wilson | 53.09 | 40.00 | 64.42 | 669 | Titans |
| 137 | 34 | Jamar Taylor | 51.44 | 55.64 | 52.15 | 294 | Dolphins |
| 138 | 35 | Chris Davis | 51.09 | 58.00 | 53.18 | 112 | Chargers |
| 139 | 36 | Asa Jackson | 51.05 | 44.92 | 57.15 | 323 | Ravens |
| 140 | 37 | Al Louis-Jean | 45.46 | 47.68 | 50.68 | 119 | Bears |
| 141 | 38 | Terrence Frederick | 45.14 | 52.99 | 50.00 | 190 | Saints |

## DI — Defensive Interior

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | J.J. Watt | 97.46 | 89.58 | 98.54 | 1050 | Texans |
| 2 | 2 | Aaron Donald | 91.81 | 87.04 | 90.83 | 707 | Rams |
| 3 | 3 | Calais Campbell | 88.35 | 87.83 | 85.68 | 858 | Cardinals |
| 4 | 4 | Sheldon Richardson | 87.93 | 89.36 | 82.81 | 810 | Jets |
| 5 | 5 | Ndamukong Suh | 87.30 | 90.11 | 81.26 | 903 | Lions |
| 6 | 6 | Ryan Davis Sr. | 86.36 | 76.98 | 88.45 | 303 | Jaguars |
| 7 | 7 | Marcell Dareus | 85.93 | 87.51 | 81.22 | 678 | Bills |
| 8 | 8 | Muhammad Wilkerson | 85.83 | 85.45 | 83.48 | 718 | Jets |
| 9 | 9 | Geno Atkins | 85.82 | 84.87 | 84.47 | 790 | Bengals |
| 10 | 10 | Jurrell Casey | 84.48 | 83.92 | 81.00 | 909 | Titans |
| 11 | 11 | Malik Jackson | 84.20 | 78.69 | 84.12 | 609 | Broncos |
| 12 | 12 | Johnathan Hankins | 84.01 | 83.36 | 82.23 | 682 | Giants |
| 13 | 13 | Kyle Williams | 83.08 | 77.15 | 83.39 | 717 | Bills |
| 14 | 14 | Kawann Short | 82.22 | 80.48 | 79.21 | 656 | Panthers |
| 15 | 15 | Gerald McCoy | 81.69 | 87.34 | 75.32 | 665 | Buccaneers |
| 16 | 16 | Star Lotulelei | 81.20 | 74.35 | 82.25 | 508 | Panthers |
| 17 | 17 | Sharrif Floyd | 80.87 | 79.51 | 78.91 | 568 | Vikings |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Fletcher Cox | 79.98 | 80.77 | 75.48 | 926 | Eagles |
| 19 | 2 | Mike Daniels | 79.92 | 70.55 | 82.00 | 781 | Packers |
| 20 | 3 | Brandon Williams | 79.77 | 76.72 | 81.15 | 623 | Ravens |
| 21 | 4 | Cameron Heyward | 79.02 | 76.14 | 76.77 | 913 | Steelers |
| 22 | 5 | Nick Fairley | 78.85 | 74.45 | 82.72 | 286 | Lions |
| 23 | 6 | Haloti Ngata | 78.32 | 81.01 | 73.71 | 627 | Ravens |
| 24 | 7 | Linval Joseph | 78.26 | 75.08 | 76.53 | 729 | Vikings |
| 25 | 8 | Timmy Jernigan | 78.21 | 71.16 | 81.87 | 322 | Ravens |
| 26 | 9 | Damon Harrison Sr. | 77.88 | 72.84 | 79.36 | 485 | Jets |
| 27 | 10 | Dan Williams | 77.71 | 78.01 | 74.18 | 462 | Cardinals |
| 28 | 11 | Jason Hatcher | 77.36 | 69.13 | 80.56 | 503 | Commanders |
| 29 | 12 | Justin Smith | 75.67 | 63.60 | 79.55 | 694 | 49ers |
| 30 | 13 | Ra'Shede Hageman | 75.57 | 57.03 | 83.76 | 220 | Falcons |
| 31 | 14 | Kenrick Ellis | 75.25 | 64.93 | 79.85 | 154 | Jets |
| 32 | 15 | Brandon Bair | 74.93 | 50.58 | 86.99 | 196 | Eagles |
| 33 | 16 | Steve McLendon | 74.84 | 67.47 | 77.77 | 330 | Steelers |
| 34 | 17 | Tyrone Crawford | 74.73 | 70.03 | 73.70 | 718 | Cowboys |
| 35 | 18 | Sen'Derrick Marks | 74.57 | 71.39 | 72.94 | 718 | Jaguars |
| 36 | 19 | Terrance Knighton | 74.43 | 71.46 | 72.25 | 566 | Broncos |
| 37 | 20 | Bennie Logan | 74.20 | 62.24 | 78.00 | 639 | Eagles |
| 38 | 21 | Leger Douzable | 74.07 | 62.17 | 77.83 | 318 | Jets |
| 39 | 22 | Akiem Hicks | 74.07 | 62.96 | 78.25 | 717 | Saints |

### Starter (93 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 40 | 1 | Stefan Charles | 73.50 | 57.33 | 83.25 | 335 | Bills |
| 41 | 2 | Karl Klug | 73.33 | 62.09 | 76.65 | 330 | Titans |
| 42 | 3 | Tyrunn Walker | 73.14 | 68.38 | 75.66 | 303 | Saints |
| 43 | 4 | Vance Walker | 73.01 | 63.30 | 75.63 | 229 | Chiefs |
| 44 | 5 | Ian Williams | 72.96 | 62.24 | 86.67 | 212 | 49ers |
| 45 | 6 | Chris Baker | 72.55 | 59.93 | 77.83 | 503 | Commanders |
| 46 | 7 | Paul Soliai | 72.44 | 61.91 | 76.12 | 505 | Falcons |
| 47 | 8 | Earl Mitchell | 72.41 | 61.62 | 75.43 | 536 | Dolphins |
| 48 | 9 | Clinton McDonald | 72.16 | 63.17 | 75.55 | 619 | Buccaneers |
| 49 | 10 | Ego Ferguson | 71.95 | 60.07 | 75.71 | 312 | Bears |
| 50 | 11 | Michael Brockers | 71.92 | 66.76 | 71.82 | 622 | Rams |
| 51 | 12 | Ropati Pitoitua | 71.87 | 59.61 | 78.16 | 390 | Titans |
| 52 | 13 | Jordan Hill | 71.84 | 57.26 | 84.03 | 360 | Seahawks |
| 53 | 14 | Tyson Jackson | 71.84 | 66.56 | 71.39 | 512 | Falcons |
| 54 | 15 | Corey Liuget | 71.64 | 59.85 | 75.33 | 775 | Chargers |
| 55 | 16 | Randy Starks | 71.62 | 63.41 | 73.44 | 540 | Dolphins |
| 56 | 17 | Henry Melton | 71.59 | 60.27 | 79.45 | 424 | Cowboys |
| 57 | 18 | Jared Odrick | 71.29 | 66.36 | 70.41 | 807 | Dolphins |
| 58 | 19 | Derrick Shelby | 71.28 | 52.13 | 80.91 | 417 | Dolphins |
| 59 | 20 | Stephen Paea | 70.99 | 63.32 | 73.08 | 700 | Bears |
| 60 | 21 | Jaye Howard Jr. | 70.83 | 57.27 | 82.37 | 437 | Chiefs |
| 61 | 22 | Cedric Thornton | 70.67 | 56.61 | 75.87 | 641 | Eagles |
| 62 | 23 | Brandon Mebane | 70.67 | 62.03 | 75.92 | 280 | Seahawks |
| 63 | 24 | Frostee Rucker | 70.55 | 52.52 | 78.40 | 539 | Cardinals |
| 64 | 25 | Sean Lissemore | 70.41 | 58.89 | 76.01 | 332 | Chargers |
| 65 | 26 | Sealver Siliga | 70.40 | 62.46 | 80.60 | 357 | Patriots |
| 66 | 27 | Al Woods | 70.36 | 58.64 | 75.58 | 292 | Titans |
| 67 | 28 | Sammie Lee Hill | 70.36 | 58.92 | 75.49 | 588 | Titans |
| 68 | 29 | Mike Martin | 70.21 | 60.20 | 75.74 | 347 | Titans |
| 69 | 30 | C.J. Mosley | 70.14 | 62.61 | 71.00 | 528 | Lions |
| 70 | 31 | John Jenkins | 70.07 | 60.30 | 75.01 | 390 | Saints |
| 71 | 32 | C.J. Wilson | 69.97 | 59.79 | 75.40 | 362 | Raiders |
| 72 | 33 | Dontari Poe | 69.93 | 63.24 | 70.23 | 946 | Chiefs |
| 73 | 34 | Pat Sims | 69.68 | 57.96 | 74.80 | 419 | Raiders |
| 74 | 35 | Desmond Bryant | 69.63 | 56.58 | 75.93 | 733 | Browns |
| 75 | 36 | Lawrence Guy Sr. | 69.55 | 58.77 | 74.45 | 272 | Ravens |
| 76 | 37 | Arthur Jones | 69.38 | 57.00 | 76.17 | 517 | Colts |
| 77 | 38 | Ricky Jean Francois | 69.22 | 60.31 | 72.24 | 728 | Colts |
| 78 | 39 | Abry Jones | 68.71 | 54.70 | 77.02 | 376 | Jaguars |
| 79 | 40 | Brodrick Bunkley | 68.70 | 57.57 | 75.39 | 274 | Saints |
| 80 | 41 | Kevin Vickerson | 68.64 | 54.98 | 76.18 | 168 | Chiefs |
| 81 | 42 | Cory Redding | 68.55 | 51.66 | 75.84 | 904 | Colts |
| 82 | 43 | Tom Johnson | 68.30 | 56.10 | 73.10 | 435 | Vikings |
| 83 | 44 | Tenny Palepoi | 68.13 | 54.60 | 72.98 | 280 | Chargers |
| 84 | 45 | Cullen Jenkins | 67.93 | 53.16 | 75.69 | 357 | Giants |
| 85 | 46 | Red Bryant | 67.85 | 60.48 | 68.59 | 522 | Jaguars |
| 86 | 47 | Kendall Langford | 67.55 | 59.95 | 68.45 | 481 | Rams |
| 87 | 48 | Quinton Dial | 67.42 | 61.20 | 73.78 | 326 | 49ers |
| 88 | 49 | Derek Wolfe | 67.22 | 61.20 | 68.63 | 764 | Broncos |
| 89 | 50 | Armonty Bryant | 67.15 | 56.31 | 78.92 | 132 | Browns |
| 90 | 51 | DeAngelo Tyson | 67.11 | 55.49 | 74.34 | 288 | Ravens |
| 91 | 52 | Montori Hughes | 67.08 | 59.97 | 72.98 | 225 | Colts |
| 92 | 53 | Datone Jones | 67.03 | 53.51 | 72.53 | 376 | Packers |
| 93 | 54 | Tony McDaniel | 67.01 | 53.33 | 72.99 | 492 | Seahawks |
| 94 | 55 | Chris Canty | 66.95 | 56.53 | 73.07 | 408 | Ravens |
| 95 | 56 | John Hughes | 66.95 | 63.36 | 71.22 | 207 | Browns |
| 96 | 57 | Sylvester Williams | 66.91 | 52.30 | 72.49 | 457 | Broncos |
| 97 | 58 | Billy Winn | 66.82 | 56.34 | 72.78 | 499 | Browns |
| 98 | 59 | Jared Crick | 66.76 | 54.26 | 70.92 | 714 | Texans |
| 99 | 60 | Corey Peters | 66.69 | 57.59 | 70.25 | 526 | Falcons |
| 100 | 61 | Tyson Alualu | 66.67 | 59.05 | 67.58 | 464 | Jaguars |
| 101 | 62 | Jay Ratliff | 66.42 | 63.10 | 72.60 | 461 | Bears |
| 102 | 63 | Josh Chapman | 66.25 | 54.96 | 70.00 | 446 | Colts |
| 103 | 64 | Ahtyba Rubin | 66.08 | 56.24 | 71.29 | 449 | Browns |
| 104 | 65 | Demarcus Dobbs | 65.90 | 56.59 | 72.89 | 203 | Seahawks |
| 105 | 66 | Zach Kerr | 65.70 | 56.83 | 71.61 | 287 | Colts |
| 106 | 67 | Antonio Smith | 65.68 | 49.65 | 72.52 | 756 | Raiders |
| 107 | 68 | Allen Bailey | 65.29 | 55.79 | 70.15 | 749 | Chiefs |
| 108 | 69 | Jonathan Babineaux | 65.23 | 53.16 | 69.63 | 695 | Falcons |
| 109 | 70 | Alan Branch | 65.19 | 54.48 | 70.76 | 231 | Patriots |
| 110 | 71 | Phil Taylor Sr. | 65.12 | 57.70 | 73.60 | 130 | Browns |
| 111 | 72 | Letroy Guion | 64.88 | 51.81 | 70.36 | 622 | Packers |
| 112 | 73 | Domata Peko Sr. | 64.81 | 48.79 | 71.32 | 730 | Bengals |
| 113 | 74 | Kevin Williams | 64.80 | 52.18 | 69.37 | 545 | Seahawks |
| 114 | 75 | Vince Wilfork | 64.60 | 51.77 | 72.73 | 935 | Patriots |
| 115 | 76 | Cam Thomas | 64.55 | 50.72 | 69.60 | 448 | Steelers |
| 116 | 77 | Dwan Edwards | 64.44 | 46.84 | 73.68 | 627 | Panthers |
| 117 | 78 | Casey Walker | 64.28 | 54.05 | 80.35 | 168 | Ravens |
| 118 | 79 | Chris Jones | 64.21 | 49.47 | 70.26 | 527 | Patriots |
| 119 | 80 | Colin Cole | 64.08 | 46.44 | 72.31 | 407 | Panthers |
| 120 | 81 | Tommy Kelly | 64.02 | 52.01 | 71.30 | 765 | Cardinals |
| 121 | 82 | Josh Boyd | 63.99 | 52.86 | 69.97 | 415 | Packers |
| 122 | 83 | Ishmaa'ily Kitchen | 63.92 | 58.59 | 66.54 | 301 | Browns |
| 123 | 84 | Brandon Thompson | 63.79 | 55.93 | 69.65 | 254 | Bengals |
| 124 | 85 | Roy Miller | 63.47 | 56.27 | 65.99 | 480 | Jaguars |
| 125 | 86 | Mike Patterson | 62.96 | 49.73 | 69.90 | 422 | Giants |
| 126 | 87 | Ryan Pickett | 62.96 | 49.81 | 69.12 | 287 | Texans |
| 127 | 88 | Akeem Spence | 62.85 | 52.23 | 65.77 | 492 | Buccaneers |
| 128 | 89 | Terrell McClain | 62.82 | 60.85 | 63.52 | 342 | Cowboys |
| 129 | 90 | Beau Allen | 62.80 | 54.60 | 64.10 | 195 | Eagles |
| 130 | 91 | Will Sutton III | 62.36 | 53.90 | 64.87 | 459 | Bears |
| 131 | 92 | Shamar Stephen | 62.18 | 55.91 | 62.19 | 404 | Vikings |
| 132 | 93 | Ricardo Mathews | 62.12 | 57.74 | 62.96 | 297 | Chargers |

### Rotation/backup (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 133 | 1 | Tony Jerod-Eddie | 61.78 | 50.55 | 68.23 | 420 | 49ers |
| 134 | 2 | Stephen Bowen | 61.46 | 52.79 | 69.12 | 241 | Commanders |
| 135 | 3 | Alex Carrington | 61.30 | 55.89 | 68.98 | 147 | Rams |
| 136 | 4 | Marvin Austin | 61.06 | 58.96 | 61.42 | 294 | Broncos |
| 137 | 5 | Kendall Reyes | 60.93 | 50.05 | 64.02 | 643 | Chargers |
| 138 | 6 | Jarvis Jenkins | 60.88 | 51.26 | 64.37 | 541 | Commanders |
| 139 | 7 | Brett Keisel | 60.75 | 46.41 | 69.48 | 439 | Steelers |
| 140 | 8 | Frank Kearse | 60.64 | 54.55 | 67.10 | 259 | Commanders |
| 141 | 9 | Andre Fluellen | 60.60 | 55.06 | 66.99 | 180 | Lions |
| 142 | 10 | Mike Pennel | 60.57 | 54.09 | 61.76 | 179 | Packers |
| 143 | 11 | Da'Quan Bowers | 60.15 | 49.62 | 68.21 | 344 | Buccaneers |
| 144 | 12 | Barry Cofield | 60.03 | 51.49 | 65.73 | 249 | Commanders |
| 145 | 13 | Stacy McGee | 59.78 | 54.89 | 62.53 | 118 | Raiders |
| 146 | 14 | Josh Mauro | 59.73 | 60.54 | 68.44 | 105 | Cardinals |
| 147 | 15 | Stephon Tuitt | 59.32 | 50.45 | 61.06 | 449 | Steelers |
| 148 | 16 | Corbin Bryant | 59.30 | 49.54 | 62.02 | 353 | Bills |
| 149 | 17 | DaQuan Jones | 59.24 | 57.30 | 67.23 | 136 | Titans |
| 150 | 18 | Kedric Golston | 58.86 | 50.62 | 62.78 | 178 | Commanders |
| 151 | 19 | Justin Ellis | 58.59 | 58.23 | 54.66 | 622 | Raiders |
| 152 | 20 | Devon Still | 58.23 | 54.49 | 62.19 | 230 | Bengals |
| 153 | 21 | Dominique Easley | 58.05 | 55.71 | 60.64 | 263 | Patriots |
| 154 | 22 | Jerrell Powe | 57.95 | 55.82 | 61.04 | 273 | Texans |
| 155 | 23 | Nick Hayden | 57.64 | 43.04 | 63.62 | 662 | Cowboys |
| 156 | 24 | Tim Jamison | 57.49 | 45.24 | 66.18 | 411 | Texans |
| 157 | 25 | Brandon Deaderick | 56.91 | 49.22 | 60.37 | 337 | Saints |
| 158 | 26 | Markus Kuhn | 56.81 | 50.19 | 64.04 | 249 | Giants |
| 159 | 27 | Jeoffrey Pagan | 55.84 | 53.70 | 53.10 | 189 | Texans |
| 160 | 28 | Caraun Reid | 52.58 | 54.66 | 51.19 | 111 | Lions |
| 161 | 29 | Ed Stinson | 52.18 | 53.49 | 53.39 | 204 | Cardinals |
| 162 | 30 | Sione Fua | 51.85 | 50.46 | 53.20 | 252 | Browns |
| 163 | 31 | Jay Bromley | 49.88 | 56.30 | 49.76 | 112 | Giants |
| 164 | 32 | Ethan Westbrooks | 49.33 | 56.07 | 54.09 | 114 | Rams |

## ED — Edge

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 93.84 | 97.43 | 89.47 | 973 | Broncos |
| 2 | 2 | Justin Houston | 92.29 | 91.20 | 90.10 | 1033 | Chiefs |
| 3 | 3 | Brandon Graham | 89.36 | 88.56 | 85.72 | 503 | Eagles |
| 4 | 4 | Ezekiel Ansah | 87.46 | 91.01 | 81.71 | 704 | Lions |
| 5 | 5 | Robert Quinn | 87.46 | 91.65 | 80.50 | 786 | Rams |
| 6 | 6 | Charles Johnson | 87.09 | 84.50 | 84.97 | 876 | Panthers |
| 7 | 7 | Pernell McPhee | 86.78 | 84.20 | 84.34 | 591 | Ravens |
| 8 | 8 | Cameron Wake | 86.38 | 79.74 | 86.96 | 759 | Dolphins |
| 9 | 9 | Mario Williams | 86.22 | 86.86 | 81.63 | 787 | Bills |
| 10 | 10 | Jerry Hughes | 85.36 | 84.74 | 81.60 | 782 | Bills |
| 11 | 11 | Junior Galette | 83.52 | 77.47 | 83.39 | 794 | Saints |
| 12 | 12 | Connor Barwin | 83.11 | 66.14 | 90.26 | 1008 | Eagles |
| 13 | 13 | Carlos Dunlap | 82.39 | 84.94 | 76.72 | 1001 | Bengals |
| 14 | 14 | DeMarcus Ware | 82.16 | 72.96 | 85.06 | 789 | Broncos |
| 15 | 15 | Michael Bennett | 81.28 | 88.21 | 72.49 | 1022 | Seahawks |
| 16 | 16 | Cliff Avril | 80.53 | 75.79 | 79.53 | 847 | Seahawks |
| 17 | 17 | Terrell Suggs | 80.37 | 77.36 | 79.04 | 957 | Ravens |
| 18 | 18 | Khalil Mack | 80.16 | 90.18 | 69.32 | 992 | Raiders |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Elvis Dumervil | 79.70 | 62.09 | 87.27 | 693 | Ravens |
| 20 | 2 | Clay Matthews | 77.95 | 61.33 | 84.86 | 1010 | Packers |
| 21 | 3 | Arthur Moats | 77.85 | 62.23 | 84.10 | 344 | Steelers |
| 22 | 4 | James Harrison | 77.74 | 62.45 | 87.93 | 485 | Steelers |
| 23 | 5 | Jason Babin | 77.66 | 63.07 | 83.22 | 458 | Jets |
| 24 | 6 | Ryan Kerrigan | 77.19 | 66.24 | 80.32 | 978 | Commanders |
| 25 | 7 | Everson Griffen | 76.92 | 77.75 | 72.20 | 967 | Vikings |
| 26 | 8 | Paul Kruger | 76.74 | 66.53 | 79.38 | 899 | Browns |
| 27 | 9 | Jason Pierre-Paul | 76.36 | 81.53 | 70.31 | 960 | Giants |
| 28 | 10 | Chandler Jones | 75.49 | 77.91 | 71.27 | 747 | Patriots |
| 29 | 11 | Quinton Coples | 75.11 | 65.66 | 77.24 | 691 | Jets |

### Starter (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Cameron Jordan | 73.98 | 76.95 | 67.83 | 999 | Saints |
| 31 | 2 | Vinny Curry | 73.90 | 64.01 | 78.73 | 377 | Eagles |
| 32 | 3 | William Hayes | 73.58 | 71.47 | 71.45 | 537 | Rams |
| 33 | 4 | Jason Worilds | 73.15 | 64.44 | 74.79 | 1031 | Steelers |
| 34 | 5 | Trent Cole | 73.04 | 62.75 | 76.77 | 806 | Eagles |
| 35 | 6 | Derrick Morgan | 72.90 | 61.62 | 76.25 | 1005 | Titans |
| 36 | 7 | Damontre Moore | 72.88 | 66.48 | 74.15 | 320 | Giants |
| 37 | 8 | Jonathan Newsome | 72.66 | 60.85 | 76.36 | 522 | Colts |
| 38 | 9 | Jared Allen | 72.47 | 59.39 | 77.54 | 888 | Bears |
| 39 | 10 | Aaron Lynch | 72.41 | 63.23 | 74.36 | 514 | 49ers |
| 40 | 11 | Dan Skuta | 72.35 | 58.90 | 80.29 | 385 | 49ers |
| 41 | 12 | Erik Walden | 72.31 | 57.68 | 77.90 | 845 | Colts |
| 42 | 13 | Justin Tuck | 72.17 | 65.60 | 73.11 | 640 | Raiders |
| 43 | 14 | Olivier Vernon | 71.59 | 73.37 | 66.24 | 839 | Dolphins |
| 44 | 15 | Julius Peppers | 71.34 | 61.14 | 73.97 | 904 | Packers |
| 45 | 16 | Barkevious Mingo | 70.42 | 68.13 | 68.81 | 668 | Browns |
| 46 | 17 | Chris Long | 70.36 | 59.89 | 78.38 | 232 | Rams |
| 47 | 18 | Willie Young | 70.00 | 61.59 | 72.47 | 664 | Bears |
| 48 | 19 | Robert Ayers | 69.97 | 69.25 | 68.37 | 379 | Giants |
| 49 | 20 | Sam Acho | 69.46 | 61.43 | 70.64 | 517 | Cardinals |
| 50 | 21 | Akeem Ayers | 68.79 | 63.56 | 71.25 | 416 | Patriots |
| 51 | 22 | Dee Ford | 68.33 | 59.16 | 75.48 | 122 | Chiefs |
| 52 | 23 | Parys Haralson | 68.31 | 53.13 | 74.27 | 488 | Saints |
| 53 | 24 | Nick Perry | 68.20 | 60.91 | 68.90 | 420 | Packers |
| 54 | 25 | Jabaal Sheard | 67.72 | 66.67 | 65.19 | 675 | Browns |
| 55 | 26 | Whitney Mercilus | 67.28 | 63.28 | 66.81 | 804 | Texans |
| 56 | 27 | Rob Ninkovich | 67.14 | 53.77 | 71.88 | 1203 | Patriots |
| 57 | 28 | Chris Clemons | 67.11 | 54.72 | 71.21 | 790 | Jaguars |
| 58 | 29 | Lerentee McCray | 66.73 | 60.90 | 68.53 | 149 | Broncos |
| 59 | 30 | Jonathan Massaquoi | 66.66 | 61.64 | 68.85 | 327 | Falcons |
| 60 | 31 | Michael Johnson | 66.55 | 67.23 | 62.97 | 628 | Buccaneers |
| 61 | 32 | Jacquies Smith | 66.39 | 58.88 | 68.26 | 455 | Buccaneers |
| 62 | 33 | David Bass | 66.33 | 62.29 | 71.63 | 143 | Bears |
| 63 | 34 | Jeremy Mincey | 66.24 | 61.89 | 64.98 | 790 | Cowboys |
| 64 | 35 | Alex Okafor | 66.17 | 57.09 | 75.22 | 770 | Cardinals |
| 65 | 36 | Anthony Spencer | 66.12 | 59.08 | 72.28 | 444 | Cowboys |
| 66 | 37 | Mario Addison | 66.04 | 57.80 | 67.37 | 471 | Panthers |
| 67 | 38 | Dion Jordan | 65.84 | 64.70 | 66.34 | 224 | Dolphins |
| 68 | 39 | Kasim Edebali | 65.57 | 57.31 | 68.99 | 179 | Saints |
| 69 | 40 | Melvin Ingram III | 65.51 | 64.51 | 68.77 | 497 | Chargers |
| 70 | 41 | O'Brien Schofield | 65.50 | 57.53 | 69.04 | 426 | Seahawks |
| 71 | 42 | Osi Umenyiora | 65.44 | 58.91 | 65.63 | 338 | Falcons |
| 72 | 43 | Kony Ealy | 64.95 | 58.69 | 64.96 | 400 | Panthers |
| 73 | 44 | Shaun Phillips | 64.45 | 48.25 | 71.08 | 486 | Colts |
| 74 | 45 | Aldon Smith | 64.44 | 65.88 | 70.18 | 418 | 49ers |
| 75 | 46 | Jeremiah Attaochu | 64.26 | 61.81 | 66.93 | 178 | Chargers |
| 76 | 47 | Brian Robison | 64.20 | 54.19 | 66.71 | 908 | Vikings |
| 77 | 48 | Tamba Hali | 64.20 | 57.19 | 64.71 | 975 | Chiefs |
| 78 | 49 | Ahmad Brooks | 64.03 | 53.01 | 70.34 | 603 | 49ers |
| 79 | 50 | Darryl Tapp | 63.86 | 57.14 | 66.36 | 297 | Lions |
| 80 | 51 | Mike Neal | 63.72 | 59.59 | 62.31 | 1442 | Packers |
| 81 | 52 | Manny Lawson | 63.60 | 49.82 | 68.93 | 341 | Bills |
| 82 | 53 | Devin Taylor | 63.30 | 60.97 | 61.47 | 241 | Lions |
| 83 | 54 | Calvin Pace | 63.28 | 48.92 | 68.68 | 823 | Jets |
| 84 | 55 | Dwight Freeney | 62.69 | 50.11 | 66.91 | 572 | Chargers |
| 85 | 56 | Brooks Reed | 62.37 | 57.21 | 62.06 | 787 | Texans |

### Rotation/backup (39 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 86 | 1 | Bjoern Werner | 61.71 | 58.06 | 59.97 | 807 | Colts |
| 87 | 2 | John Simon | 61.41 | 61.58 | 66.90 | 233 | Texans |
| 88 | 3 | Trent Murphy | 61.37 | 62.08 | 57.77 | 579 | Commanders |
| 89 | 4 | George Johnson | 61.21 | 52.74 | 65.83 | 520 | Lions |
| 90 | 5 | Wallace Gilberry | 61.00 | 46.27 | 66.85 | 883 | Bengals |
| 91 | 6 | Kroy Biermann | 60.92 | 58.75 | 62.56 | 848 | Falcons |
| 92 | 7 | William Gholston | 60.66 | 61.21 | 58.35 | 570 | Buccaneers |
| 93 | 8 | Jack Crawford | 60.45 | 58.43 | 71.04 | 143 | Cowboys |
| 94 | 9 | George Selvie | 60.38 | 52.92 | 62.65 | 564 | Cowboys |
| 95 | 10 | Jarius Wynn | 59.98 | 58.61 | 61.93 | 311 | Bills |
| 96 | 11 | DeMarcus Lawrence | 59.67 | 68.64 | 56.83 | 270 | Cowboys |
| 97 | 12 | Wes Horton | 59.49 | 58.29 | 58.47 | 498 | Panthers |
| 98 | 13 | Mathias Kiwanuka | 59.42 | 50.30 | 63.94 | 552 | Giants |
| 99 | 14 | Andre Branch | 59.25 | 61.85 | 57.62 | 330 | Jaguars |
| 100 | 15 | Matt Shaughnessy | 59.10 | 56.78 | 60.13 | 357 | Cardinals |
| 101 | 16 | Lamarr Houston | 58.63 | 61.26 | 61.05 | 396 | Bears |
| 102 | 17 | Denico Autry | 58.41 | 60.30 | 59.24 | 130 | Raiders |
| 103 | 18 | Brian Orakpo | 58.39 | 61.37 | 63.11 | 388 | Commanders |
| 104 | 19 | Eugene Sims | 57.58 | 55.11 | 55.70 | 490 | Rams |
| 105 | 20 | Quentin Groves | 56.87 | 52.63 | 57.62 | 242 | Titans |
| 106 | 21 | Jarvis Jones | 56.81 | 57.69 | 60.39 | 236 | Steelers |
| 107 | 22 | Benson Mayowa | 56.78 | 64.32 | 53.71 | 353 | Raiders |
| 108 | 23 | Scott Solomon | 56.63 | 60.65 | 56.16 | 233 | Browns |
| 109 | 24 | Tourek Williams | 56.61 | 58.77 | 54.14 | 132 | Chargers |
| 110 | 25 | Malliciah Goodman | 55.07 | 57.74 | 49.90 | 581 | Falcons |
| 111 | 26 | Kerry Wynn | 54.72 | 61.81 | 61.79 | 185 | Giants |
| 112 | 27 | T.J. Fatinikun | 54.57 | 58.73 | 53.88 | 151 | Buccaneers |
| 113 | 28 | Quanterus Smith | 54.43 | 56.48 | 49.93 | 304 | Broncos |
| 114 | 29 | Jarret Johnson | 54.41 | 47.90 | 55.62 | 547 | Chargers |
| 115 | 30 | Larry English | 54.04 | 55.14 | 53.31 | 254 | Buccaneers |
| 116 | 31 | Andy Studebaker | 53.68 | 56.73 | 49.56 | 199 | Colts |
| 117 | 32 | Kamerion Wimbley | 53.62 | 49.35 | 55.44 | 539 | Titans |
| 118 | 33 | Kareem Martin | 53.47 | 57.31 | 51.94 | 182 | Cardinals |
| 119 | 34 | Cliff Matthews | 52.35 | 56.42 | 52.77 | 116 | Falcons |
| 120 | 35 | Jason Jones | 51.59 | 46.39 | 54.96 | 686 | Lions |
| 121 | 36 | Robert Geathers | 50.73 | 47.24 | 53.26 | 624 | Bengals |
| 122 | 37 | Corey Lemonier | 48.56 | 57.23 | 46.94 | 143 | 49ers |
| 123 | 38 | Lamarr Woodley | 46.26 | 53.52 | 50.67 | 285 | Raiders |
| 124 | 39 | Jadeveon Clowney | 45.87 | 61.44 | 49.82 | 143 | Texans |

## G — Guard

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Marshal Yanda | 97.88 | 93.00 | 96.96 | 1196 | Ravens |
| 2 | 2 | Evan Mathis | 94.16 | 86.79 | 94.90 | 597 | Eagles |
| 3 | 3 | Kevin Zeitler | 92.69 | 87.37 | 92.07 | 799 | Bengals |
| 4 | 4 | Josh Sitton | 91.55 | 87.10 | 90.35 | 1133 | Packers |
| 5 | 5 | Kelechi Osemele | 91.54 | 85.00 | 91.74 | 1069 | Ravens |
| 6 | 6 | Zack Martin | 91.23 | 86.40 | 90.28 | 1172 | Cowboys |
| 7 | 7 | Brandon Brooks | 91.05 | 85.50 | 90.59 | 985 | Texans |
| 8 | 8 | T.J. Lang | 90.03 | 85.90 | 88.61 | 1085 | Packers |
| 9 | 9 | Chance Warmack | 88.12 | 81.17 | 88.58 | 965 | Titans |
| 10 | 10 | Mike Iupati | 86.92 | 80.45 | 87.06 | 946 | 49ers |
| 11 | 11 | Orlando Franklin | 86.75 | 78.00 | 88.42 | 1162 | Broncos |
| 12 | 12 | John Greco | 86.55 | 79.90 | 86.82 | 1052 | Browns |
| 13 | 13 | Joel Bitonio | 86.29 | 79.40 | 86.72 | 1052 | Browns |
| 14 | 14 | Ron Leary | 84.77 | 78.00 | 85.11 | 1110 | Cowboys |
| 15 | 15 | David DeCastro | 84.66 | 78.10 | 84.86 | 1185 | Steelers |
| 16 | 16 | Kyle Long | 84.26 | 76.20 | 85.46 | 995 | Bears |
| 17 | 17 | Mike Pollak | 84.14 | 75.51 | 85.73 | 446 | Bengals |
| 18 | 18 | Logan Mankins | 83.85 | 74.36 | 86.01 | 910 | Buccaneers |
| 19 | 19 | Louis Vasquez | 83.44 | 75.50 | 84.56 | 1192 | Broncos |
| 20 | 20 | J.R. Sweezy | 82.29 | 73.50 | 83.98 | 1231 | Seahawks |
| 21 | 21 | Jon Asamoah | 81.99 | 74.62 | 82.74 | 945 | Falcons |
| 22 | 22 | Joe Berger | 81.68 | 72.93 | 83.34 | 614 | Vikings |
| 23 | 23 | Alex Boone | 80.99 | 73.03 | 82.13 | 943 | 49ers |
| 24 | 24 | Andy Levitre | 80.98 | 71.33 | 83.25 | 966 | Titans |
| 25 | 25 | Rob Sims | 80.82 | 73.50 | 81.53 | 1160 | Lions |
| 26 | 26 | Andrew Norwell | 80.72 | 72.56 | 81.99 | 828 | Panthers |
| 27 | 27 | Clint Boling | 80.13 | 72.30 | 81.19 | 1124 | Bengals |

### Good (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Justin Blalock | 79.55 | 71.94 | 80.45 | 970 | Falcons |
| 29 | 2 | Larry Warford | 79.44 | 72.07 | 80.18 | 784 | Lions |
| 30 | 3 | Matt Slauson | 79.39 | 67.82 | 82.93 | 264 | Bears |
| 31 | 4 | Brandon Fusco | 78.91 | 67.08 | 82.63 | 171 | Vikings |
| 32 | 5 | Chris Chester | 78.81 | 71.10 | 79.79 | 1051 | Commanders |
| 33 | 6 | Jahri Evans | 78.80 | 70.70 | 80.03 | 1139 | Saints |
| 34 | 7 | Oday Aboushi | 78.77 | 70.02 | 80.44 | 722 | Jets |
| 35 | 8 | Ben Grubbs | 78.70 | 71.30 | 79.46 | 1133 | Saints |
| 36 | 9 | Gabe Jackson | 78.45 | 69.09 | 80.53 | 813 | Raiders |
| 37 | 10 | Jack Mewhort | 77.97 | 69.40 | 79.51 | 1191 | Colts |
| 38 | 11 | Jeremiah Sirles | 77.65 | 66.20 | 81.11 | 112 | Chargers |
| 39 | 12 | John Urschel | 77.42 | 69.74 | 78.37 | 356 | Ravens |
| 40 | 13 | Trai Turner | 77.29 | 71.06 | 77.27 | 809 | Panthers |
| 41 | 14 | Ramon Foster | 77.29 | 69.10 | 78.59 | 1037 | Steelers |
| 42 | 15 | Patrick Omameh | 77.10 | 67.66 | 79.23 | 902 | Buccaneers |
| 43 | 16 | Josh Kline | 76.91 | 65.21 | 80.55 | 408 | Patriots |
| 44 | 17 | John Jerry | 76.46 | 67.30 | 78.40 | 1127 | Giants |
| 45 | 18 | Willie Colon | 76.34 | 64.70 | 79.93 | 1080 | Jets |
| 46 | 19 | Hugh Thornton | 76.02 | 64.62 | 79.46 | 566 | Colts |
| 47 | 20 | Rodger Saffold | 76.02 | 66.66 | 78.10 | 918 | Rams |
| 48 | 21 | Shawn Lauvao | 75.81 | 66.58 | 77.79 | 951 | Commanders |
| 49 | 22 | Ted Larsen | 75.45 | 63.30 | 79.38 | 1085 | Cardinals |
| 50 | 23 | Todd Herremans | 75.04 | 63.44 | 78.61 | 577 | Eagles |
| 51 | 24 | Matt Tobin | 74.64 | 63.28 | 78.05 | 523 | Eagles |
| 52 | 25 | Zane Beadles | 74.31 | 65.00 | 76.35 | 1037 | Jaguars |
| 53 | 26 | Ryan Groy | 74.28 | 62.25 | 78.14 | 226 | Bears |
| 54 | 27 | Lance Louis | 74.07 | 62.07 | 77.90 | 731 | Colts |

### Starter (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 55 | 1 | James Carpenter | 73.76 | 62.10 | 77.36 | 994 | Seahawks |
| 56 | 2 | Joe Reitz | 73.74 | 64.93 | 75.45 | 475 | Colts |
| 57 | 3 | Zach Fulton | 73.51 | 64.20 | 75.55 | 996 | Chiefs |
| 58 | 4 | Chad Rinehart | 72.22 | 62.70 | 74.40 | 1067 | Chargers |
| 59 | 5 | Adam Snyder | 71.81 | 59.40 | 75.91 | 101 | Giants |
| 60 | 6 | Kraig Urbik | 71.63 | 61.31 | 74.34 | 621 | Bills |
| 61 | 7 | Jonathan Cooper | 71.52 | 62.12 | 73.62 | 184 | Cardinals |
| 62 | 8 | Charlie Johnson | 71.38 | 61.78 | 73.62 | 863 | Vikings |
| 63 | 9 | Fernando Velasco | 70.98 | 63.87 | 71.55 | 397 | Panthers |
| 64 | 10 | Johnnie Troutman | 70.94 | 60.27 | 73.88 | 773 | Chargers |
| 65 | 11 | Paul Fanaika | 70.66 | 58.83 | 74.38 | 936 | Cardinals |
| 66 | 12 | Joe Looney | 70.04 | 60.98 | 71.91 | 330 | 49ers |
| 67 | 13 | Brian Winters | 70.02 | 57.48 | 74.22 | 371 | Jets |
| 68 | 14 | Cyril Richardson | 69.98 | 55.16 | 75.70 | 312 | Bills |
| 69 | 15 | Vladimir Ducasse | 69.62 | 58.58 | 72.82 | 408 | Vikings |
| 70 | 16 | Amini Silatolu | 69.28 | 57.37 | 73.06 | 404 | Panthers |
| 71 | 17 | Dan Connolly | 69.27 | 56.80 | 73.41 | 1053 | Patriots |
| 72 | 18 | Xavier Su'a-Filo | 68.75 | 56.88 | 72.50 | 127 | Texans |
| 73 | 19 | Davin Joseph | 68.59 | 55.76 | 72.98 | 876 | Rams |
| 74 | 20 | Shelley Smith | 67.93 | 56.62 | 71.31 | 359 | Dolphins |
| 75 | 21 | Daryn Colledge | 66.99 | 54.71 | 71.01 | 740 | Dolphins |
| 76 | 22 | Chris Williams | 65.62 | 54.64 | 68.78 | 131 | Bills |
| 77 | 23 | Mike McGlynn | 64.57 | 51.32 | 69.24 | 805 | Chiefs |
| 78 | 24 | Dallas Thomas | 62.79 | 49.83 | 67.27 | 673 | Dolphins |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 79 | 1 | Garrett Gilkey | 61.91 | 46.26 | 68.18 | 204 | Buccaneers |
| 80 | 2 | Lane Taylor | 59.82 | 52.52 | 60.52 | 128 | Packers |

## HB — Running Back

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Marshawn Lynch | 81.89 | 90.23 | 72.17 | 361 | Seahawks |

### Good (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 2 | 1 | DeMarco Murray | 77.59 | 81.62 | 70.74 | 303 | Cowboys |
| 3 | 2 | Eddie Lacy | 76.79 | 82.25 | 68.98 | 340 | Packers |
| 4 | 3 | C.J. Anderson | 76.60 | 80.72 | 69.69 | 245 | Broncos |
| 5 | 4 | Jonathan Stewart | 75.52 | 76.86 | 70.46 | 246 | Panthers |
| 6 | 5 | Le'Veon Bell | 74.80 | 80.80 | 66.63 | 493 | Steelers |

### Starter (46 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Jamaal Charles | 73.78 | 71.26 | 71.30 | 334 | Chiefs |
| 8 | 2 | Pierre Thomas | 73.71 | 73.80 | 69.49 | 206 | Saints |
| 9 | 3 | Darren Sproles | 73.06 | 65.66 | 73.82 | 234 | Eagles |
| 10 | 4 | Fred Jackson | 73.03 | 70.37 | 70.63 | 283 | Bills |
| 11 | 5 | Chris Ivory | 72.99 | 67.51 | 72.48 | 149 | Jets |
| 12 | 6 | Justin Forsett | 72.55 | 71.45 | 69.12 | 346 | Ravens |
| 13 | 7 | Ahmad Bradshaw | 72.44 | 72.55 | 68.20 | 177 | Colts |
| 14 | 8 | Arian Foster | 70.76 | 70.59 | 66.71 | 269 | Texans |
| 15 | 9 | Jerick McKinnon | 69.91 | 65.35 | 68.79 | 154 | Vikings |
| 16 | 10 | Giovani Bernard | 69.77 | 69.60 | 65.71 | 244 | Bengals |
| 17 | 11 | Lamar Miller | 69.60 | 72.14 | 63.74 | 290 | Dolphins |
| 18 | 12 | Steven Jackson | 69.43 | 72.75 | 63.05 | 136 | Falcons |
| 19 | 13 | Reggie Bush | 69.42 | 69.52 | 65.19 | 187 | Lions |
| 20 | 14 | Joique Bell | 69.40 | 65.62 | 67.75 | 287 | Lions |
| 21 | 15 | Jeremy Hill | 69.23 | 62.12 | 69.80 | 178 | Bengals |
| 22 | 16 | Leon Washington | 69.02 | 60.27 | 70.68 | 167 | Titans |
| 23 | 17 | Carlos Hyde | 69.01 | 62.78 | 68.99 | 113 | 49ers |
| 24 | 18 | Roy Helu | 68.75 | 64.91 | 67.15 | 217 | Commanders |
| 25 | 19 | Matt Forte | 68.60 | 63.60 | 67.77 | 509 | Bears |
| 26 | 20 | Jacquizz Rodgers | 67.97 | 67.22 | 64.31 | 187 | Falcons |
| 27 | 21 | Alfred Morris | 67.95 | 62.34 | 67.53 | 202 | Commanders |
| 28 | 22 | Mark Ingram II | 67.88 | 67.38 | 64.04 | 165 | Saints |
| 29 | 23 | Branden Oliver | 67.32 | 69.87 | 61.45 | 158 | Chargers |
| 30 | 24 | Theo Riddick | 67.27 | 71.88 | 60.03 | 147 | Lions |
| 31 | 25 | LeSean McCoy | 66.85 | 57.93 | 68.63 | 363 | Eagles |
| 32 | 26 | Shane Vereen | 66.14 | 65.90 | 62.13 | 476 | Patriots |
| 33 | 27 | Devonta Freeman | 65.70 | 65.22 | 61.86 | 125 | Falcons |
| 34 | 28 | Bishop Sankey | 65.55 | 62.91 | 63.14 | 146 | Titans |
| 35 | 29 | Frank Gore | 65.52 | 62.30 | 63.50 | 217 | 49ers |
| 36 | 30 | Latavius Murray | 65.45 | 63.14 | 62.83 | 133 | Raiders |
| 37 | 31 | Dexter McCluster | 65.20 | 64.55 | 61.46 | 165 | Titans |
| 38 | 32 | Benny Cunningham | 65.18 | 63.93 | 61.84 | 235 | Rams |
| 39 | 33 | James Starks | 65.12 | 56.68 | 66.58 | 149 | Packers |
| 40 | 34 | Tre Mason | 65.10 | 63.26 | 62.16 | 120 | Rams |
| 41 | 35 | Dan Herron | 65.07 | 61.81 | 63.07 | 199 | Colts |
| 42 | 36 | Doug Martin | 64.82 | 59.04 | 64.51 | 121 | Buccaneers |
| 43 | 37 | Donald Brown | 64.73 | 57.54 | 65.36 | 187 | Chargers |
| 44 | 38 | Darren McFadden | 64.52 | 60.15 | 63.26 | 203 | Raiders |
| 45 | 39 | Bobby Rainey Jr. | 64.32 | 58.24 | 64.20 | 189 | Buccaneers |
| 46 | 40 | Toby Gerhart | 64.32 | 53.26 | 67.53 | 126 | Jaguars |
| 47 | 41 | Rashad Jennings | 64.28 | 58.79 | 63.77 | 172 | Giants |
| 48 | 42 | Jordan Todman | 63.85 | 64.45 | 59.28 | 173 | Jaguars |
| 49 | 43 | Robert Turbin | 63.66 | 63.05 | 59.90 | 131 | Seahawks |
| 50 | 44 | Trent Richardson | 63.38 | 61.08 | 60.75 | 196 | Colts |
| 51 | 45 | Andre Ellington | 63.25 | 55.71 | 64.11 | 239 | Cardinals |
| 52 | 46 | Chris Johnson | 62.64 | 56.11 | 62.83 | 163 | Jets |

### Rotation/backup (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 53 | 1 | Terrance West | 61.90 | 59.33 | 59.44 | 142 | Browns |
| 54 | 2 | Daniel Thomas | 61.87 | 63.54 | 56.59 | 120 | Dolphins |
| 55 | 3 | Denard Robinson | 61.84 | 57.58 | 60.52 | 132 | Jaguars |
| 56 | 4 | Maurice Jones-Drew | 61.74 | 51.16 | 64.62 | 111 | Raiders |
| 57 | 5 | Bilal Powell | 61.26 | 60.92 | 57.32 | 127 | Jets |
| 58 | 6 | Andre Williams | 60.64 | 60.07 | 56.85 | 203 | Giants |
| 59 | 7 | Travaris Cadet | 58.92 | 59.86 | 54.13 | 176 | Saints |
| 60 | 8 | Ronnie Hillman | 58.81 | 54.41 | 57.58 | 157 | Broncos |
| 61 | 9 | Matt Asiata | 58.72 | 54.91 | 57.09 | 226 | Vikings |
| 62 | 10 | Isaiah Crowell | 58.72 | 55.13 | 56.94 | 156 | Browns |
| 63 | 11 | Alfred Blue | 58.17 | 57.36 | 54.55 | 110 | Texans |
| 64 | 12 | Charles Sims | 57.18 | 55.87 | 53.88 | 115 | Buccaneers |
| 65 | 13 | Knile Davis | 56.57 | 51.51 | 55.78 | 125 | Chiefs |

## LB — Linebacker

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Luke Kuechly | 87.71 | 90.20 | 81.89 | 1089 | Panthers |
| 2 | 2 | Karlos Dansby | 82.57 | 86.25 | 78.03 | 815 | Browns |
| 3 | 3 | Dont'a Hightower | 81.37 | 82.30 | 77.10 | 1015 | Patriots |
| 4 | 4 | Mychal Kendricks | 81.10 | 87.67 | 74.84 | 762 | Eagles |
| 5 | 5 | Jamie Collins Sr. | 80.21 | 85.30 | 72.65 | 1110 | Patriots |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Thomas Davis Sr. | 79.83 | 82.20 | 74.29 | 1023 | Panthers |
| 7 | 2 | Bobby Wagner | 78.28 | 78.90 | 74.73 | 872 | Seahawks |
| 8 | 3 | Chris Borland | 78.24 | 82.63 | 78.45 | 476 | 49ers |
| 9 | 4 | Rolando McClain | 77.89 | 80.38 | 74.34 | 662 | Cowboys |
| 10 | 5 | Daryl Smith | 75.28 | 76.40 | 73.28 | 1186 | Ravens |
| 11 | 6 | Brandon Marshall | 75.25 | 77.30 | 74.92 | 906 | Broncos |
| 12 | 7 | Brandon Spikes | 75.10 | 75.40 | 70.74 | 503 | Bills |

### Starter (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Lavonte David | 72.94 | 71.10 | 71.04 | 919 | Buccaneers |
| 14 | 2 | C.J. Mosley | 72.82 | 70.20 | 70.40 | 1207 | Ravens |
| 15 | 3 | DeAndre Levy | 72.66 | 70.90 | 70.09 | 1107 | Lions |
| 16 | 4 | Koa Misi | 71.58 | 73.07 | 69.76 | 573 | Dolphins |
| 17 | 5 | Jerrell Freeman | 71.24 | 67.50 | 70.08 | 962 | Colts |
| 18 | 6 | Patrick Willis | 70.65 | 71.58 | 71.06 | 342 | 49ers |
| 19 | 7 | Craig Robertson | 69.88 | 67.73 | 67.78 | 663 | Browns |
| 20 | 8 | Bruce Irvin | 69.81 | 66.80 | 67.65 | 872 | Seahawks |
| 21 | 9 | Preston Brown | 69.79 | 65.50 | 68.48 | 1020 | Bills |
| 22 | 10 | Mike Mohamed | 69.77 | 68.20 | 71.44 | 510 | Texans |
| 23 | 11 | David Harris | 69.77 | 66.10 | 68.05 | 1012 | Jets |
| 24 | 12 | K.J. Wright | 69.57 | 66.60 | 67.70 | 1130 | Seahawks |
| 25 | 13 | Lawrence Timmons | 69.15 | 67.10 | 66.35 | 1036 | Steelers |
| 26 | 14 | Jasper Brinkley | 68.97 | 65.77 | 69.44 | 461 | Vikings |
| 27 | 15 | Prince Shembo | 68.58 | 64.38 | 68.25 | 340 | Falcons |
| 28 | 16 | Gerald Hodges | 68.47 | 69.31 | 71.56 | 502 | Vikings |
| 29 | 17 | Vince Williams | 68.24 | 65.54 | 67.58 | 257 | Steelers |
| 30 | 18 | Geno Hayes | 68.22 | 66.85 | 67.25 | 573 | Jaguars |
| 31 | 19 | Avery Williamson | 68.18 | 64.18 | 67.72 | 812 | Titans |
| 32 | 20 | Telvin Smith Sr. | 67.72 | 63.44 | 66.41 | 708 | Jaguars |
| 33 | 21 | Nigel Bradham | 67.67 | 64.96 | 67.07 | 806 | Bills |
| 34 | 22 | Anthony Barr | 67.53 | 70.12 | 65.81 | 776 | Vikings |
| 35 | 23 | James Laurinaitis | 67.33 | 61.00 | 67.39 | 1042 | Rams |
| 36 | 24 | Christian Kirksey | 67.21 | 61.64 | 66.75 | 681 | Browns |
| 37 | 25 | Demario Davis | 67.08 | 62.20 | 66.36 | 1007 | Jets |
| 38 | 26 | Tahir Whitehead | 66.58 | 61.35 | 65.90 | 759 | Lions |
| 39 | 27 | Devon Kennard | 66.52 | 65.21 | 69.47 | 331 | Giants |
| 40 | 28 | Akeem Dent | 66.03 | 65.63 | 67.54 | 223 | Texans |
| 41 | 29 | Jason Trusnik | 65.89 | 63.47 | 66.99 | 396 | Dolphins |
| 42 | 30 | Vincent Rey | 65.54 | 59.00 | 66.56 | 1005 | Bengals |
| 43 | 31 | Kelvin Sheppard | 65.52 | 63.64 | 67.81 | 120 | Dolphins |
| 44 | 32 | Philip Wheeler | 65.48 | 61.77 | 64.31 | 379 | Dolphins |
| 45 | 33 | DeMeco Ryans | 65.22 | 62.62 | 66.96 | 513 | Eagles |
| 46 | 34 | Todd Davis | 65.05 | 64.83 | 69.36 | 177 | Broncos |
| 47 | 35 | Jelani Jenkins | 64.92 | 61.30 | 65.38 | 901 | Dolphins |
| 48 | 36 | Danny Lansanah | 64.71 | 63.86 | 67.63 | 630 | Buccaneers |
| 49 | 37 | Rey Maualuga | 64.59 | 61.94 | 64.37 | 458 | Bengals |
| 50 | 38 | Vontaze Burfict | 64.53 | 61.13 | 68.36 | 217 | Bengals |
| 51 | 39 | Manti Te'o | 64.48 | 62.99 | 65.61 | 457 | Chargers |
| 52 | 40 | Joplo Bartu | 64.01 | 57.84 | 64.34 | 486 | Falcons |
| 53 | 41 | Jerod Mayo | 64.01 | 64.46 | 67.88 | 334 | Patriots |
| 54 | 42 | Josh Bynes | 63.58 | 60.26 | 65.89 | 226 | Lions |
| 55 | 43 | Stephen Tulloch | 63.49 | 65.49 | 64.76 | 138 | Lions |
| 56 | 44 | Paul Posluszny | 63.32 | 61.55 | 65.33 | 488 | Jaguars |
| 57 | 45 | A.J. Klein | 63.29 | 59.65 | 65.45 | 282 | Panthers |
| 58 | 46 | Christian Jones | 63.20 | 60.73 | 65.88 | 434 | Bears |
| 59 | 47 | Ashlee Palmer | 62.94 | 60.58 | 60.77 | 193 | Lions |
| 60 | 48 | Michael Wilhoite | 62.81 | 61.70 | 65.43 | 1015 | 49ers |
| 61 | 49 | Nate Irving | 62.73 | 60.65 | 65.58 | 347 | Broncos |
| 62 | 50 | Kyle Wilber | 62.52 | 60.92 | 64.10 | 241 | Cowboys |
| 63 | 51 | Lance Briggs | 62.40 | 62.23 | 64.69 | 453 | Bears |
| 64 | 52 | Emmanuel Acho | 62.33 | 59.49 | 63.19 | 265 | Eagles |
| 65 | 53 | Mark Herzlich | 62.22 | 56.91 | 65.76 | 313 | Giants |
| 66 | 54 | Kevin Minter | 62.08 | 60.77 | 64.66 | 336 | Cardinals |

### Rotation/backup (60 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 67 | 1 | Jon Bostic | 61.43 | 57.09 | 64.05 | 679 | Bears |
| 68 | 2 | Keenan Robinson | 61.38 | 56.92 | 62.14 | 812 | Commanders |
| 69 | 3 | Shea McClellin | 60.86 | 54.97 | 64.79 | 424 | Bears |
| 70 | 4 | Zaviar Gooden | 60.71 | 60.22 | 64.94 | 157 | Titans |
| 71 | 5 | Justin Durant | 60.66 | 59.37 | 64.43 | 324 | Cowboys |
| 72 | 6 | Wesley Woodyard | 60.23 | 54.60 | 59.81 | 879 | Titans |
| 73 | 7 | Mason Foster | 60.17 | 54.81 | 63.01 | 556 | Buccaneers |
| 74 | 8 | D'Qwell Jackson | 60.04 | 52.30 | 61.03 | 1196 | Colts |
| 75 | 9 | Will Compton | 59.80 | 55.10 | 61.37 | 359 | Commanders |
| 76 | 10 | Kavell Conner | 59.64 | 55.31 | 62.01 | 348 | Chargers |
| 77 | 11 | Sam Barrington | 59.50 | 57.72 | 63.69 | 474 | Packers |
| 78 | 12 | Jonathan Casillas | 59.47 | 56.88 | 61.62 | 273 | Patriots |
| 79 | 13 | Jacquian Williams | 59.24 | 55.28 | 62.61 | 563 | Giants |
| 80 | 14 | Perry Riley | 59.14 | 52.30 | 60.57 | 888 | Commanders |
| 81 | 15 | A.J. Hawk | 58.98 | 50.10 | 60.73 | 875 | Packers |
| 82 | 16 | Curtis Lofton | 58.73 | 50.10 | 60.32 | 1044 | Saints |
| 83 | 17 | Alec Ogletree | 58.66 | 52.70 | 58.47 | 1045 | Rams |
| 84 | 18 | Brian Cushing | 58.61 | 58.17 | 60.89 | 725 | Texans |
| 85 | 19 | Jamari Lattimore | 58.34 | 60.00 | 62.44 | 281 | Packers |
| 86 | 20 | Anthony Hitchens | 58.31 | 46.41 | 62.07 | 607 | Cowboys |
| 87 | 21 | Joe Mays | 58.25 | 57.56 | 63.40 | 117 | Chiefs |
| 88 | 22 | Josh Mauga | 57.95 | 54.00 | 60.27 | 1005 | Chiefs |
| 89 | 23 | Andrew Gachkar | 57.84 | 58.36 | 59.99 | 384 | Chargers |
| 90 | 24 | Sean Spence | 57.62 | 51.39 | 57.61 | 521 | Steelers |
| 91 | 25 | David Hawthorne | 57.51 | 53.69 | 59.02 | 742 | Saints |
| 92 | 26 | Jameel McClain | 57.50 | 51.20 | 60.03 | 972 | Giants |
| 93 | 27 | Casey Matthews | 57.19 | 52.31 | 58.26 | 432 | Eagles |
| 94 | 28 | Corey Nelson | 56.99 | 57.84 | 65.68 | 108 | Broncos |
| 95 | 29 | Ray-Ray Armstrong | 56.88 | 58.28 | 61.03 | 238 | Raiders |
| 96 | 30 | Keith Rivers | 56.83 | 55.35 | 58.34 | 185 | Bills |
| 97 | 31 | Sio Moore | 56.77 | 52.35 | 59.20 | 697 | Raiders |
| 98 | 32 | D.J. Williams | 56.44 | 51.74 | 62.28 | 413 | Bears |
| 99 | 33 | Josh McNary | 56.37 | 55.50 | 60.20 | 260 | Colts |
| 100 | 34 | Ryan Shazier | 56.12 | 55.43 | 59.72 | 281 | Steelers |
| 101 | 35 | Emmanuel Lamur | 55.91 | 50.70 | 58.60 | 953 | Bengals |
| 102 | 36 | Paul Worrilow | 55.84 | 47.00 | 58.35 | 1079 | Falcons |
| 103 | 37 | Darryl Sharpton | 55.78 | 53.48 | 61.90 | 104 | Bears |
| 104 | 38 | Khaseem Greene | 54.96 | 54.76 | 60.82 | 117 | Bears |
| 105 | 39 | Dekoda Watson | 54.84 | 55.08 | 56.76 | 100 | Cowboys |
| 106 | 40 | LaRoy Reynolds | 54.57 | 55.56 | 61.07 | 116 | Jaguars |
| 107 | 41 | James-Michael Johnson | 54.55 | 49.20 | 61.04 | 437 | Chiefs |
| 108 | 42 | Steven Johnson | 54.14 | 54.00 | 57.77 | 244 | Broncos |
| 109 | 43 | Chad Greenway | 53.97 | 46.43 | 56.91 | 760 | Vikings |
| 110 | 44 | Jon Beason | 53.97 | 55.16 | 58.38 | 159 | Giants |
| 111 | 45 | Orie Lemon | 53.80 | 56.00 | 60.47 | 190 | Buccaneers |
| 112 | 46 | Jackson Jeffcoat | 53.72 | 59.12 | 65.95 | 117 | Commanders |
| 113 | 47 | Bruce Carter | 53.29 | 45.00 | 56.54 | 644 | Cowboys |
| 114 | 48 | JoLonn Dunbar | 52.81 | 44.65 | 55.85 | 422 | Rams |
| 115 | 49 | J.T. Thomas | 52.46 | 46.85 | 57.96 | 713 | Jaguars |
| 116 | 50 | Reggie Walker | 52.45 | 49.81 | 55.24 | 143 | Chargers |
| 117 | 51 | Ramon Humber | 52.44 | 44.01 | 55.88 | 445 | Saints |
| 118 | 52 | Deontae Skinner | 52.38 | 55.84 | 59.32 | 103 | Patriots |
| 119 | 53 | Donald Butler | 52.15 | 41.36 | 57.36 | 704 | Chargers |
| 120 | 54 | Dane Fletcher | 51.97 | 45.74 | 57.06 | 351 | Buccaneers |
| 121 | 55 | Malcolm Smith | 51.54 | 43.64 | 57.22 | 274 | Seahawks |
| 122 | 56 | Brad Jones | 51.31 | 42.23 | 56.85 | 216 | Packers |
| 123 | 57 | Justin Tuggle | 50.44 | 46.71 | 56.17 | 269 | Texans |
| 124 | 58 | Larry Foote | 49.90 | 40.00 | 57.02 | 1074 | Cardinals |
| 125 | 59 | Miles Burris | 49.87 | 40.00 | 55.41 | 1060 | Raiders |
| 126 | 60 | Nick Moody | 49.61 | 56.42 | 59.41 | 164 | 49ers |

## QB — Quarterback

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Rodgers | 86.78 | 90.30 | 81.21 | 693 | Packers |
| 2 | 2 | Drew Brees | 83.73 | 87.53 | 76.08 | 740 | Saints |
| 3 | 3 | Ben Roethlisberger | 83.48 | 85.86 | 77.34 | 741 | Steelers |
| 4 | 4 | Tony Romo | 82.27 | 80.71 | 80.64 | 569 | Cowboys |
| 5 | 5 | Peyton Manning | 82.09 | 82.18 | 77.48 | 708 | Broncos |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Matt Ryan | 79.69 | 81.86 | 73.37 | 720 | Falcons |
| 7 | 2 | Philip Rivers | 79.42 | 79.75 | 74.65 | 670 | Chargers |
| 8 | 3 | Tom Brady | 79.42 | 83.84 | 71.25 | 802 | Patriots |
| 9 | 4 | Russell Wilson | 78.09 | 76.41 | 75.26 | 661 | Seahawks |
| 10 | 5 | Andrew Luck | 75.84 | 75.07 | 71.93 | 855 | Colts |
| 11 | 6 | Ryan Tannehill | 74.49 | 76.80 | 68.28 | 684 | Dolphins |
| 12 | 7 | Joe Flacco | 74.19 | 73.60 | 70.13 | 709 | Ravens |

### Starter (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Eli Manning | 72.34 | 70.73 | 69.33 | 670 | Giants |
| 14 | 2 | Matthew Stafford | 72.04 | 72.10 | 67.45 | 751 | Lions |
| 15 | 3 | Cam Newton | 71.97 | 73.37 | 66.88 | 632 | Panthers |
| 16 | 4 | Alex Smith | 71.70 | 73.03 | 68.32 | 577 | Chiefs |
| 17 | 5 | Colin Kaepernick | 70.18 | 67.75 | 69.06 | 624 | 49ers |
| 18 | 6 | Carson Palmer | 69.62 | 68.12 | 73.53 | 255 | Cardinals |
| 19 | 7 | Ryan Fitzpatrick | 69.54 | 68.48 | 72.28 | 391 | Texans |
| 20 | 8 | Andy Dalton | 69.45 | 68.41 | 66.67 | 599 | Bengals |
| 21 | 9 | Jay Cutler | 68.67 | 66.90 | 67.55 | 665 | Bears |
| 22 | 10 | Nick Foles | 66.31 | 68.53 | 67.88 | 355 | Eagles |
| 23 | 11 | Teddy Bridgewater | 65.41 | 73.62 | 66.99 | 501 | Vikings |
| 24 | 12 | Robert Griffin III | 63.78 | 61.55 | 68.68 | 281 | Commanders |
| 25 | 13 | Derek Anderson | 62.28 | 66.38 | 73.08 | 107 | Panthers |

### Rotation/backup (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Colt McCoy | 61.74 | 61.10 | 75.22 | 160 | Commanders |
| 27 | 2 | Mike Glennon | 61.57 | 64.36 | 65.01 | 246 | Buccaneers |
| 28 | 3 | Geno Smith | 61.31 | 60.40 | 62.19 | 448 | Jets |
| 29 | 4 | Brian Hoyer | 61.30 | 60.78 | 66.23 | 499 | Browns |
| 30 | 5 | Mark Sanchez | 61.18 | 58.66 | 68.58 | 356 | Eagles |
| 31 | 6 | Charlie Whitehurst | 61.17 | 65.42 | 69.32 | 228 | Titans |
| 32 | 7 | Kirk Cousins | 60.80 | 59.69 | 71.33 | 228 | Commanders |
| 33 | 8 | Kyle Orton | 60.63 | 57.93 | 67.32 | 509 | Bills |
| 34 | 9 | Shaun Hill | 60.02 | 60.98 | 67.65 | 272 | Rams |
| 35 | 10 | Zach Mettenberger | 59.82 | 57.92 | 68.71 | 210 | Titans |
| 36 | 11 | Austin Davis | 59.69 | 59.94 | 65.26 | 337 | Rams |
| 37 | 12 | Derek Carr | 58.79 | 54.40 | 56.92 | 666 | Raiders |
| 38 | 13 | Josh McCown | 58.73 | 58.95 | 61.80 | 405 | Buccaneers |
| 39 | 14 | E.J. Manuel | 57.71 | 59.09 | 60.79 | 153 | Bills |
| 40 | 15 | Drew Stanton | 57.65 | 53.57 | 63.89 | 292 | Cardinals |
| 41 | 16 | Chad Henne | 56.17 | 58.46 | 60.88 | 101 | Jaguars |
| 42 | 17 | Blake Bortles | 55.76 | 45.53 | 56.41 | 587 | Jaguars |
| 43 | 18 | Jake Locker | 55.41 | 58.22 | 59.08 | 186 | Titans |
| 44 | 19 | Ryan Lindley | 54.54 | 49.27 | 55.80 | 136 | Cardinals |
| 45 | 20 | Michael Vick | 51.55 | 50.89 | 56.95 | 158 | Jets |

## S — Safety

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Eric Weddle | 93.49 | 90.00 | 91.65 | 1020 | Chargers |
| 2 | 2 | Devin McCourty | 92.58 | 90.20 | 90.00 | 1176 | Patriots |
| 3 | 3 | Glover Quin | 91.06 | 90.40 | 87.34 | 1100 | Lions |
| 4 | 4 | Antoine Bethea | 90.10 | 89.60 | 86.27 | 1040 | 49ers |
| 5 | 5 | Earl Thomas III | 89.91 | 89.10 | 86.28 | 1166 | Seahawks |
| 6 | 6 | Mike Adams | 86.15 | 85.20 | 82.61 | 1222 | Colts |
| 7 | 7 | Reggie Nelson | 84.59 | 82.90 | 81.75 | 1170 | Bengals |
| 8 | 8 | Jim Leonhard | 82.25 | 79.65 | 80.13 | 505 | Browns |
| 9 | 9 | Will Hill III | 81.74 | 82.70 | 81.71 | 703 | Ravens |
| 10 | 10 | Husain Abdullah | 81.30 | 80.50 | 79.14 | 1025 | Chiefs |
| 11 | 11 | Dawan Landry | 81.13 | 75.60 | 80.65 | 945 | Jets |
| 12 | 12 | Harrison Smith | 81.04 | 74.80 | 83.53 | 1070 | Vikings |
| 13 | 13 | Kam Chancellor | 80.11 | 80.30 | 75.81 | 1044 | Seahawks |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Tashaun Gipson Sr. | 79.47 | 79.25 | 79.30 | 772 | Browns |
| 15 | 2 | Ryan Mundy | 79.34 | 75.70 | 79.37 | 947 | Bears |
| 16 | 3 | James Ihedigbo | 79.16 | 80.30 | 75.49 | 885 | Lions |
| 17 | 4 | Malcolm Jenkins | 78.85 | 75.70 | 77.41 | 1158 | Eagles |
| 18 | 5 | Jaiquawn Jarrett | 77.87 | 75.67 | 76.20 | 383 | Jets |
| 19 | 6 | Duron Harmon | 77.19 | 67.69 | 79.74 | 317 | Patriots |
| 20 | 7 | Kurt Coleman | 76.13 | 70.60 | 79.71 | 391 | Chiefs |
| 21 | 8 | David Bruton | 76.12 | 69.83 | 81.25 | 228 | Broncos |
| 22 | 9 | Donte Whitner | 75.95 | 70.50 | 75.41 | 1152 | Browns |
| 23 | 10 | Patrick Chung | 75.43 | 70.90 | 75.64 | 976 | Patriots |
| 24 | 11 | Nate Allen | 75.30 | 72.50 | 73.94 | 1081 | Eagles |
| 25 | 12 | Reshad Jones | 74.47 | 64.68 | 78.91 | 757 | Dolphins |
| 26 | 13 | Da'Norris Searcy | 74.37 | 65.41 | 77.32 | 648 | Bills |
| 27 | 14 | Rahim Moore | 74.12 | 74.00 | 71.91 | 1126 | Broncos |

### Starter (48 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Ha Ha Clinton-Dix | 73.99 | 69.20 | 73.02 | 1069 | Packers |
| 29 | 2 | Tyrann Mathieu | 73.46 | 69.80 | 73.03 | 448 | Cardinals |
| 30 | 3 | Jeromy Miles | 73.23 | 71.15 | 76.18 | 355 | Ravens |
| 31 | 4 | Tavon Wilson | 72.19 | 65.92 | 75.64 | 211 | Patriots |
| 32 | 5 | Darian Stewart | 71.96 | 70.10 | 72.26 | 873 | Ravens |
| 33 | 6 | Bradley McDougald | 71.92 | 68.15 | 71.57 | 445 | Buccaneers |
| 34 | 7 | Rodney McLeod | 71.69 | 71.20 | 70.77 | 1025 | Rams |
| 35 | 8 | Danieal Manning | 71.56 | 65.78 | 74.38 | 579 | Texans |
| 36 | 9 | Tre Boston | 71.50 | 65.36 | 76.63 | 458 | Panthers |
| 37 | 10 | George Iloka | 71.46 | 64.20 | 72.14 | 1184 | Bengals |
| 38 | 11 | Troy Polamalu | 71.03 | 69.25 | 71.48 | 754 | Steelers |
| 39 | 12 | Isa Abdul-Quddus | 70.69 | 66.34 | 72.66 | 303 | Lions |
| 40 | 13 | Bernard Pollard | 70.27 | 66.72 | 74.20 | 344 | Titans |
| 41 | 14 | Ron Parker | 69.77 | 68.90 | 66.19 | 1013 | Chiefs |
| 42 | 15 | Micah Hyde | 68.78 | 64.32 | 67.58 | 796 | Packers |
| 43 | 16 | Charles Woodson | 68.51 | 63.30 | 67.82 | 1100 | Raiders |
| 44 | 17 | George Wilson | 68.32 | 64.58 | 66.65 | 812 | Titans |
| 45 | 18 | Chris Conte | 68.30 | 63.07 | 69.90 | 463 | Bears |
| 46 | 19 | Eric Reid | 68.21 | 61.40 | 69.24 | 879 | 49ers |
| 47 | 20 | Sergio Brown | 68.13 | 67.07 | 69.99 | 532 | Colts |
| 48 | 21 | Marcus Gilchrist | 67.95 | 58.50 | 70.09 | 987 | Chargers |
| 49 | 22 | Brock Vereen | 67.79 | 64.13 | 69.19 | 502 | Bears |
| 50 | 23 | Duke Williams | 67.57 | 65.20 | 68.88 | 528 | Bills |
| 51 | 24 | Morgan Burnett | 67.41 | 55.00 | 72.15 | 1070 | Packers |
| 52 | 25 | Barry Church | 67.08 | 62.70 | 68.53 | 997 | Cowboys |
| 53 | 26 | Andrew Sendejo | 67.07 | 61.86 | 73.05 | 141 | Vikings |
| 54 | 27 | Dwight Lowery | 67.04 | 62.00 | 69.05 | 1029 | Falcons |
| 55 | 28 | Aaron Williams | 67.03 | 62.90 | 66.77 | 901 | Bills |
| 56 | 29 | Robert Blanton | 66.84 | 60.00 | 72.01 | 948 | Vikings |
| 57 | 30 | Kendrick Lewis | 66.13 | 59.90 | 67.59 | 1077 | Texans |
| 58 | 31 | Louis Delmas | 66.06 | 63.70 | 66.70 | 834 | Dolphins |
| 59 | 32 | Trenton Robinson | 65.90 | 62.16 | 70.67 | 101 | Commanders |
| 60 | 33 | Calvin Pryor | 65.78 | 63.00 | 64.50 | 680 | Jets |
| 61 | 34 | Johnathan Cyprien | 65.76 | 61.40 | 65.53 | 985 | Jaguars |
| 62 | 35 | Jordan Poyer | 65.75 | 62.24 | 68.47 | 116 | Browns |
| 63 | 36 | Darrell Stuckey | 65.58 | 61.33 | 67.58 | 173 | Chargers |
| 64 | 37 | Jahleel Addae | 65.22 | 61.15 | 67.01 | 427 | Chargers |
| 65 | 38 | Jimmy Wilson | 65.17 | 64.00 | 62.82 | 786 | Dolphins |
| 66 | 39 | LaRon Landry | 64.76 | 55.55 | 68.40 | 604 | Colts |
| 67 | 40 | Sherrod Martin | 64.66 | 64.43 | 66.48 | 135 | Jaguars |
| 68 | 41 | Jamarca Sanford | 64.41 | 62.60 | 68.64 | 109 | Saints |
| 69 | 42 | Jeff Heath | 64.27 | 60.30 | 65.62 | 156 | Cowboys |
| 70 | 43 | Don Carey | 64.15 | 61.57 | 68.06 | 133 | Lions |
| 71 | 44 | Thomas DeCoud | 64.14 | 61.78 | 63.95 | 665 | Panthers |
| 72 | 45 | Usama Young | 63.25 | 60.05 | 69.86 | 216 | Raiders |
| 73 | 46 | D.J. Swearinger Sr. | 62.86 | 58.90 | 61.34 | 1018 | Texans |
| 74 | 47 | T.J. McDonald | 62.28 | 54.70 | 65.52 | 1049 | Rams |
| 75 | 48 | Quintin Demps | 62.24 | 57.56 | 62.55 | 629 | Giants |

### Rotation/backup (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 76 | 1 | T.J. Ward | 61.98 | 55.40 | 62.62 | 1075 | Broncos |
| 77 | 2 | Major Wright | 61.25 | 54.99 | 64.18 | 505 | Buccaneers |
| 78 | 3 | Kenny Vaccaro | 60.88 | 58.90 | 59.47 | 980 | Saints |
| 79 | 4 | Craig Dahl | 60.49 | 59.48 | 61.89 | 182 | 49ers |
| 80 | 5 | Kemal Ishmael | 60.44 | 50.62 | 68.68 | 805 | Falcons |
| 81 | 6 | Antrel Rolle | 60.25 | 56.90 | 58.32 | 1048 | Giants |
| 82 | 7 | Rashad Johnson | 60.13 | 55.20 | 61.65 | 1131 | Cardinals |
| 83 | 8 | Eric Berry | 60.06 | 55.54 | 64.10 | 361 | Chiefs |
| 84 | 9 | Rafael Bush | 59.67 | 52.02 | 66.12 | 468 | Saints |
| 85 | 10 | Stevie Brown | 59.61 | 48.64 | 64.84 | 576 | Giants |
| 86 | 11 | Tyvon Branch | 59.48 | 59.33 | 66.98 | 190 | Raiders |
| 87 | 12 | Jairus Byrd | 58.54 | 59.11 | 61.81 | 267 | Saints |
| 88 | 13 | Sean Richardson | 58.00 | 56.67 | 61.70 | 133 | Packers |
| 89 | 14 | Mike Mitchell | 57.83 | 46.40 | 61.49 | 1010 | Steelers |
| 90 | 15 | Roman Harper | 57.09 | 48.70 | 60.08 | 1024 | Panthers |
| 91 | 16 | J.J. Wilcox | 56.96 | 52.80 | 57.14 | 1109 | Cowboys |
| 92 | 17 | Ryan Clark | 56.65 | 50.30 | 56.92 | 1013 | Commanders |
| 93 | 18 | Will Allen | 56.63 | 53.61 | 57.40 | 324 | Steelers |
| 94 | 19 | Michael Griffin | 56.06 | 46.40 | 58.96 | 1132 | Titans |
| 95 | 20 | Quinton Carter | 54.65 | 55.60 | 58.09 | 216 | Broncos |
| 96 | 21 | Danny McCray | 53.74 | 51.20 | 60.75 | 168 | Bears |
| 97 | 22 | Daimion Stafford | 53.52 | 53.51 | 57.70 | 277 | Titans |
| 98 | 23 | Tony Jefferson | 53.19 | 43.10 | 59.27 | 727 | Cardinals |
| 99 | 24 | Brandon Meriweather | 53.11 | 45.79 | 61.00 | 597 | Commanders |
| 100 | 25 | Larry Asante | 52.38 | 58.28 | 58.08 | 161 | Raiders |
| 101 | 26 | William Moore | 52.26 | 47.82 | 56.16 | 322 | Falcons |
| 102 | 27 | Josh Evans | 52.13 | 43.80 | 53.90 | 971 | Jaguars |
| 103 | 28 | Terrence Brooks | 52.00 | 53.24 | 57.87 | 234 | Ravens |
| 104 | 29 | Dashon Goldson | 49.85 | 40.00 | 54.24 | 780 | Buccaneers |
| 105 | 30 | Matt Elam | 49.62 | 40.00 | 51.87 | 680 | Ravens |
| 106 | 31 | Charles Godfrey | 48.47 | 47.29 | 54.35 | 205 | Falcons |
| 107 | 32 | Bacarri Rambo | 48.43 | 48.20 | 53.67 | 128 | Bills |

## T — Tackle

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Andrew Whitworth | 97.13 | 92.30 | 96.19 | 1091 | Bengals |
| 2 | 2 | Jason Peters | 96.41 | 92.40 | 94.92 | 1146 | Eagles |
| 3 | 3 | Joe Thomas | 95.96 | 91.00 | 95.10 | 1052 | Browns |
| 4 | 4 | Joe Staley | 94.49 | 89.60 | 93.58 | 1056 | 49ers |
| 5 | 5 | Tyron Smith | 92.11 | 87.20 | 91.22 | 1178 | Cowboys |
| 6 | 6 | Donald Penn | 90.63 | 88.60 | 87.81 | 1021 | Raiders |
| 7 | 7 | Lane Johnson | 90.27 | 82.80 | 91.09 | 889 | Eagles |
| 8 | 8 | Trent Williams | 88.84 | 81.55 | 89.53 | 877 | Commanders |
| 9 | 9 | Jared Veldheer | 88.75 | 83.20 | 88.28 | 1104 | Cardinals |
| 10 | 10 | Branden Albert | 88.65 | 80.80 | 89.71 | 543 | Dolphins |
| 11 | 11 | Anthony Castonzo | 88.30 | 84.40 | 86.73 | 1360 | Colts |
| 12 | 12 | Kelvin Beachum | 87.79 | 82.70 | 87.02 | 1185 | Steelers |
| 13 | 13 | King Dunlap | 87.45 | 82.60 | 86.52 | 1058 | Chargers |
| 14 | 14 | Duane Brown | 87.05 | 80.30 | 87.39 | 1100 | Texans |
| 15 | 15 | Rick Wagner | 86.70 | 80.04 | 86.98 | 969 | Ravens |
| 16 | 16 | Derek Newton | 86.58 | 78.80 | 87.60 | 1108 | Texans |
| 17 | 17 | Will Beatty | 86.30 | 80.00 | 86.34 | 1114 | Giants |
| 18 | 18 | Cordy Glenn | 86.23 | 79.90 | 86.28 | 1045 | Bills |
| 19 | 19 | Taylor Lewan | 85.88 | 72.57 | 90.59 | 353 | Titans |
| 20 | 20 | Michael Roos | 85.55 | 74.72 | 88.60 | 289 | Titans |
| 21 | 21 | Doug Free | 85.25 | 77.57 | 86.21 | 700 | Cowboys |
| 22 | 22 | Terron Armstead | 84.88 | 77.14 | 85.87 | 836 | Saints |
| 23 | 23 | Riley Reiff | 84.29 | 78.50 | 83.98 | 1008 | Lions |
| 24 | 24 | Jermey Parnell | 84.12 | 73.72 | 86.89 | 498 | Cowboys |
| 25 | 25 | Zach Strief | 83.47 | 76.40 | 84.02 | 1058 | Saints |
| 26 | 26 | Sebastian Vollmer | 83.45 | 76.10 | 84.19 | 1223 | Patriots |
| 27 | 27 | Bryan Bulaga | 83.05 | 74.40 | 84.65 | 1064 | Packers |
| 28 | 28 | Marcus Gilbert | 82.85 | 73.50 | 84.91 | 820 | Steelers |
| 29 | 29 | Nate Solder | 82.59 | 74.10 | 84.09 | 1256 | Patriots |
| 30 | 30 | Joe Barksdale | 82.56 | 73.00 | 84.76 | 996 | Rams |
| 31 | 31 | Phil Loadholt | 82.54 | 74.03 | 84.04 | 715 | Vikings |
| 32 | 32 | Ryan Schraeder | 82.40 | 72.07 | 85.12 | 646 | Falcons |
| 33 | 33 | Demar Dotson | 82.36 | 73.48 | 84.11 | 979 | Buccaneers |
| 34 | 34 | D'Brickashaw Ferguson | 82.04 | 75.30 | 82.37 | 1088 | Jets |
| 35 | 35 | Ryan Clady | 81.94 | 74.10 | 83.00 | 1124 | Broncos |
| 36 | 36 | Mitchell Schwartz | 81.23 | 72.70 | 82.75 | 1052 | Browns |
| 37 | 37 | Russell Okung | 81.12 | 71.40 | 83.43 | 1050 | Seahawks |
| 38 | 38 | Bobby Massie | 80.13 | 70.90 | 82.11 | 1104 | Cardinals |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Breno Giacomini | 79.93 | 70.30 | 82.19 | 1088 | Jets |
| 40 | 2 | Anthony Davis | 79.74 | 66.45 | 84.44 | 429 | 49ers |
| 41 | 3 | David Bakhtiari | 79.73 | 72.30 | 80.52 | 1144 | Packers |
| 42 | 4 | Ryan Harris | 79.68 | 70.05 | 81.94 | 956 | Chiefs |
| 43 | 5 | LaAdrian Waddle | 78.73 | 67.55 | 82.02 | 550 | Lions |
| 44 | 6 | Eric Fisher | 78.49 | 67.30 | 81.78 | 1005 | Chiefs |
| 45 | 7 | Jake Long | 78.12 | 69.42 | 79.76 | 434 | Rams |
| 46 | 8 | Chris Clark | 77.76 | 65.34 | 81.88 | 486 | Broncos |
| 47 | 9 | Matt Kalil | 77.03 | 65.70 | 80.41 | 1025 | Vikings |
| 48 | 10 | Byron Stingily | 76.96 | 63.23 | 81.95 | 243 | Titans |
| 49 | 11 | Anthony Collins | 76.76 | 64.64 | 80.68 | 621 | Buccaneers |
| 50 | 12 | Jordan Mills | 76.48 | 62.45 | 81.67 | 814 | Bears |
| 51 | 13 | Eugene Monroe | 76.37 | 65.31 | 79.58 | 728 | Ravens |
| 52 | 14 | Michael Ola | 75.90 | 64.20 | 79.53 | 823 | Bears |
| 53 | 15 | Greg Robinson | 74.91 | 61.11 | 79.94 | 724 | Rams |
| 54 | 16 | Ja'Wuan James | 74.47 | 63.50 | 77.61 | 1038 | Dolphins |
| 55 | 17 | Mike Remmers | 74.34 | 66.02 | 75.72 | 506 | Panthers |
| 56 | 18 | Tom Compton | 74.24 | 62.76 | 77.72 | 650 | Commanders |
| 57 | 19 | Michael Oher | 74.20 | 63.33 | 77.28 | 651 | Titans |
| 58 | 20 | Eric Winston | 74.09 | 61.11 | 78.58 | 235 | Bengals |

### Starter (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | Austin Pasztor | 73.79 | 61.41 | 77.88 | 492 | Jaguars |
| 60 | 2 | Gosder Cherilus | 73.65 | 61.48 | 77.60 | 950 | Colts |
| 61 | 3 | Cam Fleming | 73.35 | 61.24 | 77.26 | 244 | Patriots |
| 62 | 4 | Josh Wells | 73.31 | 63.64 | 75.59 | 102 | Jaguars |
| 63 | 5 | Jason Fox | 73.10 | 59.63 | 77.91 | 214 | Dolphins |
| 64 | 6 | Jonathan Martin | 72.99 | 61.62 | 76.41 | 646 | 49ers |
| 65 | 7 | Menelik Watson | 72.75 | 58.87 | 77.84 | 487 | Raiders |
| 66 | 8 | Cornelius Lucas | 72.62 | 60.87 | 76.28 | 515 | Lions |
| 67 | 9 | Paul Cornick | 72.15 | 61.71 | 74.95 | 300 | Broncos |
| 68 | 10 | Khalif Barnes | 71.98 | 57.97 | 77.16 | 766 | Raiders |
| 69 | 11 | Tyler Polumbus | 71.92 | 60.08 | 75.65 | 471 | Commanders |
| 70 | 12 | Lamar Holmes | 71.78 | 59.81 | 75.60 | 226 | Falcons |
| 71 | 13 | Jake Matthews | 71.70 | 59.80 | 75.46 | 944 | Falcons |
| 72 | 14 | Byron Bell | 71.62 | 59.80 | 75.33 | 1157 | Panthers |
| 73 | 15 | Marshall Newhouse | 71.53 | 59.00 | 75.72 | 381 | Bengals |
| 74 | 16 | Marcus Cannon | 71.46 | 57.64 | 76.51 | 446 | Patriots |
| 75 | 17 | Nate Chandler | 71.28 | 58.83 | 75.42 | 691 | Panthers |
| 76 | 18 | Sam Young | 70.74 | 58.35 | 74.84 | 396 | Jaguars |
| 77 | 19 | Jamon Meredith | 70.24 | 56.81 | 75.02 | 210 | Titans |
| 78 | 20 | Jeff Linkenbach | 70.24 | 57.40 | 74.63 | 220 | Chiefs |
| 79 | 21 | Seantrel Henderson | 69.89 | 57.30 | 74.11 | 1061 | Bills |
| 80 | 22 | Bryce Harris | 69.89 | 57.99 | 73.65 | 387 | Saints |
| 81 | 23 | Mike Adams | 69.54 | 57.82 | 73.19 | 381 | Steelers |
| 82 | 24 | Will Svitek | 69.37 | 58.34 | 72.56 | 188 | Titans |
| 83 | 25 | James Hurst | 67.02 | 52.99 | 72.21 | 514 | Ravens |
| 84 | 26 | Morgan Moses | 66.25 | 57.76 | 67.75 | 126 | Commanders |
| 85 | 27 | David Foucault | 64.98 | 52.20 | 69.34 | 127 | Panthers |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 86 | 1 | Matt McCants | 61.94 | 53.56 | 63.36 | 112 | Raiders |

## TE — Tight End

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 88.05 | 91.30 | 81.71 | 679 | Patriots |
| 2 | 2 | Jason Witten | 81.78 | 82.97 | 76.82 | 603 | Cowboys |
| 3 | 3 | Travis Kelce | 81.74 | 79.84 | 78.84 | 426 | Chiefs |
| 4 | 4 | Greg Olsen | 81.11 | 84.02 | 75.00 | 644 | Panthers |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Ladarius Green | 79.25 | 68.75 | 82.09 | 146 | Chargers |
| 6 | 2 | Julius Thomas | 77.21 | 73.39 | 75.59 | 478 | Broncos |
| 7 | 3 | Jimmy Graham | 77.20 | 74.11 | 75.10 | 574 | Saints |
| 8 | 4 | Jared Cook | 76.21 | 70.15 | 76.08 | 435 | Rams |
| 9 | 5 | Brent Celek | 76.15 | 74.25 | 73.25 | 377 | Eagles |
| 10 | 6 | Dwayne Allen | 76.13 | 71.84 | 74.83 | 406 | Colts |
| 11 | 7 | Zach Ertz | 76.12 | 77.06 | 71.32 | 440 | Eagles |
| 12 | 8 | Martellus Bennett | 75.69 | 79.38 | 69.07 | 647 | Bears |
| 13 | 9 | Delanie Walker | 75.38 | 72.56 | 73.09 | 524 | Titans |
| 14 | 10 | Charles Clay | 74.29 | 75.39 | 69.39 | 458 | Dolphins |
| 15 | 11 | Antonio Gates | 74.27 | 71.91 | 71.68 | 576 | Chargers |
| 16 | 12 | Daniel Fells | 74.06 | 68.86 | 73.36 | 145 | Giants |

### Starter (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Owen Daniels | 73.95 | 73.16 | 70.31 | 562 | Ravens |
| 18 | 2 | Chase Ford | 73.91 | 67.28 | 74.17 | 239 | Vikings |
| 19 | 3 | Luke Willson | 73.50 | 67.09 | 73.60 | 368 | Seahawks |
| 20 | 4 | Josh Hill | 73.41 | 65.05 | 74.81 | 106 | Saints |
| 21 | 5 | Heath Miller | 73.14 | 72.60 | 69.33 | 721 | Steelers |
| 22 | 6 | Larry Donnell | 72.90 | 65.97 | 73.35 | 540 | Giants |
| 23 | 7 | Jermaine Gresham | 72.57 | 67.59 | 71.72 | 464 | Bengals |
| 24 | 8 | Niles Paul | 72.49 | 63.29 | 74.46 | 310 | Commanders |
| 25 | 9 | Jim Dray | 72.09 | 64.77 | 72.81 | 241 | Browns |
| 26 | 10 | Coby Fleener | 71.89 | 71.74 | 67.83 | 621 | Colts |
| 27 | 11 | Cooper Helfet | 71.49 | 65.62 | 71.24 | 147 | Seahawks |
| 28 | 12 | Richard Rodgers | 71.46 | 61.81 | 73.73 | 240 | Packers |
| 29 | 13 | Vernon Davis | 71.23 | 54.32 | 78.34 | 488 | 49ers |
| 30 | 14 | Jordan Reed | 71.16 | 66.55 | 70.06 | 272 | Commanders |
| 31 | 15 | Dennis Pitta | 70.58 | 65.06 | 70.09 | 106 | Ravens |
| 32 | 16 | Benjamin Watson | 70.35 | 63.04 | 71.05 | 293 | Saints |
| 33 | 17 | Scott Chandler | 70.33 | 60.47 | 72.73 | 532 | Bills |
| 34 | 18 | Jacob Tamme | 70.15 | 57.44 | 74.46 | 200 | Broncos |
| 35 | 19 | Jack Doyle | 69.83 | 65.97 | 68.24 | 127 | Colts |
| 36 | 20 | Tim Wright | 69.75 | 60.78 | 71.56 | 276 | Patriots |
| 37 | 21 | Marcedes Lewis | 69.50 | 57.70 | 73.20 | 281 | Jaguars |
| 38 | 22 | Gavin Escobar | 69.46 | 61.17 | 70.82 | 144 | Cowboys |
| 39 | 23 | Gary Barnidge | 69.27 | 63.21 | 69.15 | 195 | Browns |
| 40 | 24 | Lance Kendricks | 69.18 | 66.91 | 66.53 | 256 | Rams |
| 41 | 25 | Tony Moeaki | 69.13 | 64.97 | 67.73 | 118 | Seahawks |
| 42 | 26 | Jordan Cameron | 68.81 | 62.15 | 69.08 | 250 | Browns |
| 43 | 27 | Rhett Ellison | 68.68 | 66.91 | 65.69 | 212 | Vikings |
| 44 | 28 | Kyle Rudolph | 68.49 | 62.49 | 68.32 | 261 | Vikings |
| 45 | 29 | Virgil Green | 67.98 | 65.16 | 65.70 | 145 | Broncos |
| 46 | 30 | Andrew Quarless | 67.50 | 62.79 | 66.48 | 375 | Packers |
| 47 | 31 | Garrett Graham | 67.45 | 58.18 | 69.47 | 324 | Texans |
| 48 | 32 | Dion Sims | 67.28 | 64.59 | 64.91 | 268 | Dolphins |
| 49 | 33 | Clay Harbor | 67.18 | 63.59 | 65.41 | 302 | Jaguars |
| 50 | 34 | Anthony Fasano | 67.10 | 60.94 | 67.04 | 375 | Chiefs |
| 51 | 35 | Austin Seferian-Jenkins | 66.85 | 61.07 | 66.54 | 296 | Buccaneers |
| 52 | 36 | Jace Amaro | 66.53 | 64.87 | 63.47 | 251 | Jets |
| 53 | 37 | Michael Hoomanawanui | 65.91 | 55.05 | 68.99 | 246 | Patriots |
| 54 | 38 | Matt Spaeth | 65.90 | 62.09 | 64.28 | 149 | Steelers |
| 55 | 39 | Ryan Griffin | 65.84 | 56.27 | 68.05 | 177 | Texans |
| 56 | 40 | Rob Housler | 65.61 | 60.51 | 64.85 | 143 | Cardinals |
| 57 | 41 | Ed Dickson | 65.40 | 58.96 | 65.53 | 199 | Panthers |
| 58 | 42 | Lee Smith | 65.24 | 61.97 | 63.26 | 121 | Bills |
| 59 | 43 | Logan Paulsen | 65.11 | 56.43 | 66.73 | 158 | Commanders |
| 60 | 44 | Brian Leonhardt | 65.01 | 57.96 | 65.54 | 106 | Raiders |
| 61 | 45 | Mychal Rivera | 64.87 | 53.66 | 68.18 | 557 | Raiders |
| 62 | 46 | Brandon Pettigrew | 64.31 | 56.40 | 65.41 | 316 | Lions |
| 63 | 47 | Brandon Myers | 63.67 | 56.09 | 64.56 | 268 | Buccaneers |
| 64 | 48 | Eric Ebron | 63.21 | 57.31 | 62.97 | 330 | Lions |
| 65 | 49 | Jeff Cumberland | 63.00 | 50.53 | 67.15 | 480 | Jets |
| 66 | 50 | Dante Rosario | 62.77 | 58.98 | 61.13 | 120 | Bears |
| 67 | 51 | C.J. Fiedorowicz | 62.65 | 54.76 | 63.75 | 185 | Texans |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 68 | 1 | Levine Toilolo | 61.49 | 54.25 | 62.15 | 568 | Falcons |
| 69 | 2 | John Carlson | 61.09 | 52.50 | 62.65 | 469 | Cardinals |
| 70 | 3 | Luke Stocker | 60.63 | 58.14 | 58.12 | 158 | Buccaneers |

## WR — Wide Receiver

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Odell Beckham Jr. | 88.06 | 86.51 | 84.93 | 489 | Giants |
| 2 | 2 | Antonio Brown | 87.55 | 91.10 | 81.02 | 722 | Steelers |
| 3 | 3 | Julio Jones | 87.55 | 89.15 | 82.31 | 613 | Falcons |
| 4 | 4 | Calvin Johnson | 86.56 | 87.77 | 81.58 | 537 | Lions |
| 5 | 5 | Demaryius Thomas | 86.41 | 87.90 | 81.25 | 674 | Broncos |
| 6 | 6 | Dez Bryant | 85.99 | 89.00 | 79.82 | 582 | Cowboys |
| 7 | 7 | Jordy Nelson | 85.21 | 86.90 | 79.91 | 686 | Packers |
| 8 | 8 | DeAndre Hopkins | 83.65 | 82.42 | 80.30 | 576 | Texans |
| 9 | 9 | T.Y. Hilton | 83.63 | 83.20 | 79.75 | 716 | Colts |
| 10 | 10 | Emmanuel Sanders | 83.58 | 86.40 | 77.54 | 661 | Broncos |
| 11 | 11 | Randall Cobb | 83.43 | 84.60 | 78.49 | 664 | Packers |
| 12 | 12 | A.J. Green | 83.20 | 81.13 | 80.41 | 380 | Bengals |
| 13 | 13 | DeSean Jackson | 83.08 | 73.81 | 85.09 | 509 | Commanders |
| 14 | 14 | Mike Evans | 82.54 | 82.33 | 78.51 | 544 | Buccaneers |
| 15 | 15 | Golden Tate | 82.28 | 80.00 | 79.63 | 706 | Lions |
| 16 | 16 | Jeremy Maclin | 81.77 | 80.23 | 78.63 | 652 | Eagles |
| 17 | 17 | Josh Gordon | 81.56 | 72.53 | 83.42 | 145 | Browns |
| 18 | 18 | Martavis Bryant | 80.79 | 69.40 | 84.22 | 256 | Steelers |
| 19 | 19 | Alshon Jeffery | 80.70 | 76.70 | 79.20 | 665 | Bears |
| 20 | 20 | Anquan Boldin | 80.47 | 78.66 | 77.51 | 589 | 49ers |
| 21 | 21 | Steve Smith | 80.25 | 78.48 | 77.26 | 601 | Ravens |
| 22 | 22 | Andrew Hawkins | 80.06 | 78.84 | 76.71 | 424 | Browns |

### Good (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Doug Baldwin | 79.68 | 77.98 | 76.64 | 611 | Seahawks |
| 24 | 2 | Vincent Jackson | 79.38 | 75.42 | 77.86 | 621 | Buccaneers |
| 25 | 3 | Taylor Gabriel | 79.30 | 70.77 | 80.82 | 354 | Browns |
| 26 | 4 | Kenny Stills | 79.22 | 75.56 | 77.50 | 472 | Saints |
| 27 | 5 | Ricardo Lockette | 78.66 | 64.40 | 84.00 | 133 | Seahawks |
| 28 | 6 | Malcom Floyd | 78.41 | 74.46 | 76.88 | 639 | Chargers |
| 29 | 7 | Andre Johnson | 78.22 | 74.22 | 76.72 | 514 | Texans |
| 30 | 8 | Eric Decker | 78.20 | 76.64 | 75.07 | 476 | Jets |
| 31 | 9 | Rueben Randle | 77.83 | 76.46 | 74.57 | 626 | Giants |
| 32 | 10 | Keenan Allen | 77.69 | 75.07 | 75.27 | 527 | Chargers |
| 33 | 11 | Larry Fitzgerald | 77.66 | 76.58 | 74.22 | 583 | Cardinals |
| 34 | 12 | Steve Johnson | 77.59 | 77.63 | 73.40 | 225 | 49ers |
| 35 | 13 | Torrey Smith | 76.90 | 71.02 | 76.65 | 572 | Ravens |
| 36 | 14 | Sammy Watkins | 76.58 | 69.22 | 77.32 | 649 | Bills |
| 37 | 15 | Charles Johnson | 76.55 | 65.02 | 80.07 | 288 | Vikings |
| 38 | 16 | Jordan Matthews | 76.51 | 70.72 | 76.20 | 510 | Eagles |
| 39 | 17 | Donte Moncrief | 76.33 | 68.19 | 77.59 | 353 | Colts |
| 40 | 18 | Julian Edelman | 76.24 | 78.00 | 70.90 | 693 | Patriots |
| 41 | 19 | Jarvis Landry | 76.07 | 74.90 | 72.69 | 452 | Dolphins |
| 42 | 20 | Kelvin Benjamin | 75.85 | 74.78 | 72.39 | 641 | Panthers |
| 43 | 21 | Brandon Marshall | 75.84 | 73.76 | 73.06 | 515 | Bears |
| 44 | 22 | Percy Harvin | 75.81 | 72.73 | 73.69 | 294 | Jets |
| 45 | 23 | Michael Floyd | 75.66 | 69.60 | 75.53 | 668 | Cardinals |
| 46 | 24 | Stedman Bailey | 75.63 | 69.83 | 75.33 | 291 | Rams |
| 47 | 25 | Terrance Williams | 75.57 | 65.47 | 78.14 | 548 | Cowboys |
| 48 | 26 | Jermaine Kearse | 75.38 | 66.72 | 76.98 | 545 | Seahawks |
| 49 | 27 | Pierre Garcon | 75.33 | 70.02 | 74.70 | 583 | Commanders |
| 50 | 28 | Jarius Wright | 75.25 | 68.22 | 75.77 | 382 | Vikings |
| 51 | 29 | Kendall Wright | 75.22 | 69.82 | 74.66 | 475 | Titans |
| 52 | 30 | Kenny Britt | 75.20 | 72.71 | 72.69 | 500 | Rams |
| 53 | 31 | Eddie Royal | 75.17 | 71.72 | 73.30 | 545 | Chargers |
| 54 | 32 | John Brown | 75.11 | 68.30 | 75.48 | 526 | Cardinals |
| 55 | 33 | Brandon LaFell | 75.04 | 72.60 | 72.50 | 715 | Patriots |
| 56 | 34 | Mike Wallace | 74.95 | 71.82 | 72.87 | 531 | Dolphins |
| 57 | 35 | Miles Austin | 74.94 | 72.61 | 72.33 | 330 | Browns |
| 58 | 36 | Ted Ginn Jr. | 74.77 | 64.76 | 77.27 | 134 | Cardinals |
| 59 | 37 | Albert Wilson | 74.60 | 64.29 | 77.31 | 172 | Chiefs |
| 60 | 38 | James Wright | 74.41 | 60.34 | 79.63 | 118 | Bengals |
| 61 | 39 | Cole Beasley | 74.28 | 67.93 | 74.34 | 369 | Cowboys |
| 62 | 40 | Brian Quick | 74.15 | 67.02 | 74.74 | 205 | Rams |
| 63 | 41 | Greg Jennings | 74.01 | 71.54 | 71.49 | 595 | Vikings |

### Starter (72 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 64 | 1 | Dwayne Bowe | 73.93 | 70.83 | 71.83 | 507 | Chiefs |
| 65 | 2 | Marques Colston | 73.48 | 65.18 | 74.84 | 613 | Saints |
| 66 | 3 | Allen Robinson II | 73.45 | 66.91 | 73.64 | 357 | Jaguars |
| 67 | 4 | Nate Washington | 73.40 | 64.53 | 75.14 | 540 | Titans |
| 68 | 5 | Robert Meachem | 73.21 | 63.72 | 75.37 | 143 | Saints |
| 69 | 6 | Andre Holmes | 73.11 | 63.38 | 75.43 | 497 | Raiders |
| 70 | 7 | Travis Benjamin | 73.02 | 64.87 | 74.29 | 202 | Browns |
| 71 | 8 | Markus Wheaton | 73.02 | 68.78 | 71.68 | 552 | Steelers |
| 72 | 9 | Corey Brown | 73.00 | 65.35 | 73.93 | 228 | Panthers |
| 73 | 10 | Harry Douglas | 72.74 | 68.04 | 71.70 | 418 | Falcons |
| 74 | 11 | Justin Hunter | 72.57 | 62.25 | 75.29 | 399 | Titans |
| 75 | 12 | Michael Crabtree | 72.45 | 65.64 | 72.83 | 505 | 49ers |
| 76 | 13 | Roddy White | 72.23 | 68.34 | 70.66 | 620 | Falcons |
| 77 | 14 | Chris Givens | 72.03 | 58.31 | 77.01 | 130 | Rams |
| 78 | 15 | Victor Cruz | 72.02 | 61.32 | 74.98 | 206 | Giants |
| 79 | 16 | Brandin Cooks | 71.59 | 66.69 | 70.69 | 373 | Saints |
| 80 | 17 | Greg Salas | 71.31 | 61.23 | 73.87 | 137 | Jets |
| 81 | 18 | James Jones | 71.21 | 65.39 | 70.92 | 547 | Raiders |
| 82 | 19 | Devin Hester | 71.21 | 62.97 | 72.53 | 315 | Falcons |
| 83 | 20 | Hakeem Nicks | 71.07 | 62.02 | 72.93 | 491 | Colts |
| 84 | 21 | Brian Hartline | 71.05 | 62.98 | 72.27 | 515 | Dolphins |
| 85 | 22 | Cordarrelle Patterson | 70.88 | 65.54 | 70.28 | 380 | Vikings |
| 86 | 23 | Brandon Lloyd | 70.88 | 62.54 | 72.28 | 250 | 49ers |
| 87 | 24 | Kamar Aiken | 70.83 | 65.22 | 70.40 | 187 | Ravens |
| 88 | 25 | Brenton Bersin | 70.46 | 64.25 | 70.43 | 117 | Panthers |
| 89 | 26 | Robert Woods | 70.39 | 68.25 | 67.65 | 608 | Bills |
| 90 | 27 | Mohamed Sanu | 70.39 | 66.97 | 68.51 | 602 | Bengals |
| 91 | 28 | Brice Butler | 70.38 | 65.49 | 69.48 | 195 | Raiders |
| 92 | 29 | Jaron Brown | 70.23 | 66.36 | 68.64 | 138 | Cardinals |
| 93 | 30 | Brandon Tate | 70.20 | 61.41 | 71.90 | 327 | Bengals |
| 94 | 31 | Cecil Shorts | 70.18 | 62.22 | 71.32 | 507 | Jaguars |
| 95 | 32 | Marlon Brown | 70.14 | 67.27 | 67.88 | 283 | Ravens |
| 96 | 33 | Davante Adams | 70.09 | 62.99 | 70.66 | 575 | Packers |
| 97 | 34 | Jeremy Kerley | 70.08 | 61.39 | 71.70 | 441 | Jets |
| 98 | 35 | Jerricho Cotchery | 70.04 | 64.42 | 69.62 | 576 | Panthers |
| 99 | 36 | Reggie Wayne | 70.04 | 61.70 | 71.43 | 686 | Colts |
| 100 | 37 | Rishard Matthews | 69.63 | 62.02 | 70.53 | 112 | Dolphins |
| 101 | 38 | Allen Hurns | 69.54 | 61.81 | 70.52 | 541 | Jaguars |
| 102 | 39 | Chris Hogan | 69.36 | 63.32 | 69.22 | 360 | Bills |
| 103 | 40 | Donnie Avery | 69.03 | 60.87 | 70.30 | 154 | Chiefs |
| 104 | 41 | Corey Fuller | 68.96 | 59.61 | 71.03 | 278 | Lions |
| 105 | 42 | Kenbrell Thompkins | 68.94 | 60.13 | 70.64 | 272 | Raiders |
| 106 | 43 | Riley Cooper | 68.79 | 59.15 | 71.05 | 586 | Eagles |
| 107 | 44 | Preston Parker | 68.73 | 62.01 | 69.05 | 425 | Giants |
| 108 | 45 | Lance Moore | 68.68 | 57.93 | 71.68 | 210 | Steelers |
| 109 | 46 | Jacoby Jones | 68.60 | 57.23 | 72.02 | 132 | Ravens |
| 110 | 47 | Vincent Brown | 68.55 | 60.44 | 69.79 | 106 | Raiders |
| 111 | 48 | Paul Richardson Jr. | 68.44 | 65.38 | 66.31 | 322 | Seahawks |
| 112 | 49 | Wes Welker | 68.36 | 60.85 | 69.20 | 546 | Broncos |
| 113 | 50 | Denarius Moore | 68.26 | 54.20 | 73.47 | 164 | Raiders |
| 114 | 51 | Jason Avant | 67.95 | 62.40 | 67.49 | 386 | Chiefs |
| 115 | 52 | Eric Weems | 67.87 | 60.71 | 68.47 | 116 | Falcons |
| 116 | 53 | Andre Roberts | 67.54 | 61.04 | 67.70 | 495 | Commanders |
| 117 | 54 | Nick Toon | 67.51 | 62.30 | 66.81 | 180 | Saints |
| 118 | 55 | A.J. Jenkins | 67.41 | 56.83 | 70.29 | 157 | Chiefs |
| 119 | 56 | Marqise Lee | 67.40 | 61.02 | 67.48 | 348 | Jaguars |
| 120 | 57 | Jeremy Ross | 67.37 | 58.94 | 68.82 | 519 | Lions |
| 121 | 58 | Derek Hagan | 67.21 | 59.38 | 68.26 | 213 | Titans |
| 122 | 59 | Danny Amendola | 66.75 | 59.04 | 67.72 | 408 | Patriots |
| 123 | 60 | Louis Murphy Jr. | 66.74 | 60.37 | 66.82 | 353 | Buccaneers |
| 124 | 61 | Dane Sanzenbacher | 66.07 | 55.68 | 68.83 | 128 | Bengals |
| 125 | 62 | Josh Huff | 65.69 | 59.29 | 65.79 | 114 | Eagles |
| 126 | 63 | Brandon Gibson | 65.69 | 57.20 | 67.19 | 330 | Dolphins |
| 127 | 64 | Keshawn Martin | 65.34 | 58.41 | 65.80 | 110 | Texans |
| 128 | 65 | Tavon Austin | 64.99 | 59.50 | 64.48 | 342 | Rams |
| 129 | 66 | Jarrett Boykin | 64.59 | 49.24 | 70.66 | 149 | Packers |
| 130 | 67 | Damaris Johnson | 64.27 | 55.91 | 65.67 | 352 | Texans |
| 131 | 68 | Junior Hemingway | 63.19 | 56.71 | 63.35 | 150 | Chiefs |
| 132 | 69 | Ryan Grant | 62.93 | 57.71 | 62.25 | 114 | Commanders |
| 133 | 70 | Josh Morgan | 62.66 | 55.19 | 63.47 | 271 | Bears |
| 134 | 71 | Seyi Ajirotutu | 62.54 | 55.51 | 63.06 | 106 | Chargers |
| 135 | 72 | Andre Caldwell | 62.41 | 55.53 | 62.83 | 107 | Broncos |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 136 | 1 | Marquess Wilson | 60.37 | 55.79 | 59.26 | 268 | Bears |
| 137 | 2 | Frankie Hammond | 60.09 | 53.37 | 60.41 | 157 | Chiefs |
