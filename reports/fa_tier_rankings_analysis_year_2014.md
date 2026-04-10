# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:27:04Z
- **Requested analysis_year:** 2014 (clamped to 2014)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Travis Frederick | 94.52 | 89.30 | 93.83 | 1178 | Cowboys |
| 2 | 2 | Nick Mangold | 91.40 | 84.60 | 91.77 | 979 | Jets |
| 3 | 3 | Jason Kelce | 86.90 | 78.20 | 88.53 | 828 | Eagles |
| 4 | 4 | Max Unger | 85.13 | 77.30 | 86.19 | 548 | Seahawks |
| 5 | 5 | Maurkice Pouncey | 84.42 | 76.30 | 85.66 | 1178 | Steelers |
| 6 | 6 | Jeremy Zuttah | 84.36 | 76.20 | 85.63 | 1201 | Ravens |
| 7 | 7 | Corey Linsley | 84.29 | 76.80 | 85.11 | 1188 | Packers |
| 8 | 8 | Stefen Wisniewski | 84.14 | 74.40 | 86.47 | 1014 | Raiders |
| 9 | 9 | Rodney Hudson | 83.97 | 76.30 | 84.92 | 1006 | Chiefs |
| 10 | 10 | Will Montgomery | 83.77 | 74.80 | 85.59 | 651 | Broncos |
| 11 | 11 | Alex Mack | 83.34 | 74.40 | 85.13 | 297 | Browns |
| 12 | 12 | Chris Myers | 83.32 | 74.80 | 84.84 | 1102 | Texans |
| 13 | 13 | Eric Wood | 82.25 | 73.70 | 83.79 | 1058 | Bills |
| 14 | 14 | Kory Lichtensteiger | 81.48 | 73.20 | 82.84 | 1052 | Commanders |
| 15 | 15 | Ryan Kalil | 81.41 | 73.10 | 82.78 | 1225 | Panthers |
| 16 | 16 | Daniel Kilgore | 80.04 | 78.00 | 77.23 | 450 | 49ers |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Evan Smith | 79.87 | 67.60 | 83.89 | 930 | Buccaneers |
| 18 | 2 | Roberto Garza | 79.68 | 70.30 | 81.77 | 747 | Bears |
| 19 | 3 | John Sullivan | 79.25 | 70.20 | 81.11 | 975 | Vikings |
| 20 | 4 | A.Q. Shipley | 78.35 | 72.20 | 78.28 | 423 | Colts |
| 21 | 5 | Brian De La Puente | 78.23 | 69.20 | 80.09 | 488 | Bears |
| 22 | 6 | J.D. Walton | 78.07 | 67.50 | 80.95 | 1118 | Giants |
| 23 | 7 | Jonathan Goodwin | 77.68 | 67.60 | 80.23 | 853 | Saints |
| 24 | 8 | Russell Bodine | 76.21 | 65.60 | 79.11 | 1124 | Bengals |
| 25 | 9 | Bryan Stork | 75.48 | 64.50 | 78.63 | 891 | Patriots |
| 26 | 10 | Luke Bowanko | 75.31 | 64.30 | 78.49 | 922 | Jaguars |
| 27 | 11 | Brian Schwenke | 75.11 | 63.60 | 78.61 | 647 | Titans |
| 28 | 12 | Samson Satele | 74.77 | 63.40 | 78.19 | 1075 | Dolphins |
| 29 | 13 | Dominic Raiola | 74.44 | 63.60 | 77.50 | 1091 | Lions |

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Jonotthan Harrison | 73.89 | 61.10 | 78.25 | 693 | Colts |
| 31 | 2 | Trevor Robinson | 72.53 | 60.60 | 76.32 | 169 | Chargers |
| 32 | 3 | David Molk | 72.26 | 63.40 | 74.00 | 403 | Eagles |
| 33 | 4 | Joe Hawley | 70.69 | 62.80 | 71.79 | 243 | Falcons |
| 34 | 5 | Scott Wells | 69.37 | 55.10 | 74.71 | 979 | Rams |
| 35 | 6 | Peter Konz | 69.36 | 56.50 | 73.77 | 183 | Falcons |
| 36 | 7 | Cody Wallace | 68.82 | 56.40 | 72.94 | 148 | Steelers |
| 37 | 8 | James Stone | 67.67 | 60.00 | 68.61 | 672 | Falcons |
| 38 | 9 | Dalton Freeman | 66.95 | 54.40 | 71.15 | 110 | Jets |
| 39 | 10 | Khaled Holmes | 66.79 | 53.80 | 71.28 | 376 | Colts |
| 40 | 11 | Travis Swanson | 66.33 | 54.70 | 69.91 | 373 | Lions |
| 41 | 12 | Lyle Sendlein | 66.32 | 53.90 | 70.44 | 1104 | Cardinals |
| 42 | 13 | Rich Ohrnberger | 65.06 | 56.20 | 66.80 | 447 | Chargers |
| 43 | 14 | Patrick Lewis | 63.34 | 52.10 | 66.67 | 274 | Seahawks |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 44 | 1 | Doug Legursky | 60.58 | 40.70 | 69.67 | 133 | Chargers |
| 45 | 2 | Nick McDonald | 58.81 | 39.80 | 67.32 | 469 | Browns |

## CB — Cornerback

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Vontae Davis | 94.12 | 92.50 | 92.06 | 1022 | Colts |
| 2 | 2 | Richard Sherman | 93.37 | 89.50 | 91.78 | 1177 | Seahawks |
| 3 | 3 | Chris Harris Jr. | 92.63 | 92.20 | 88.75 | 1056 | Broncos |
| 4 | 4 | K'Waun Williams | 92.20 | 90.30 | 92.44 | 350 | Browns |
| 5 | 5 | Sean Smith | 88.53 | 87.50 | 85.05 | 1038 | Chiefs |
| 6 | 6 | Casey Hayward Jr. | 88.29 | 86.00 | 89.71 | 461 | Packers |
| 7 | 7 | Darrelle Revis | 87.23 | 83.20 | 88.66 | 1188 | Patriots |
| 8 | 8 | Kareem Jackson | 86.00 | 85.90 | 84.08 | 774 | Texans |
| 9 | 9 | Desmond Trufant | 85.05 | 79.00 | 84.91 | 1076 | Falcons |
| 10 | 10 | Jimmy Smith | 83.44 | 82.20 | 84.89 | 461 | Ravens |
| 11 | 11 | Orlando Scandrick | 82.69 | 81.90 | 80.08 | 1000 | Cowboys |
| 12 | 12 | Rashean Mathis | 82.41 | 80.30 | 80.80 | 1045 | Lions |
| 13 | 13 | Dominique Rodgers-Cromartie | 82.34 | 76.20 | 82.26 | 752 | Giants |
| 14 | 14 | Brandon Flowers | 81.68 | 78.50 | 81.52 | 818 | Chargers |
| 15 | 15 | Brandon Boykin | 81.61 | 73.70 | 82.71 | 499 | Eagles |
| 16 | 16 | Chris Culliver | 80.90 | 75.50 | 81.58 | 822 | 49ers |
| 17 | 17 | Bene Benwikere | 80.35 | 80.10 | 80.51 | 542 | Panthers |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Johnathan Joseph | 79.97 | 74.20 | 79.97 | 859 | Texans |
| 19 | 2 | Xavier Rhodes | 79.97 | 71.20 | 82.81 | 1027 | Vikings |
| 20 | 3 | Jerraud Powers | 79.88 | 75.70 | 80.16 | 774 | Cardinals |
| 21 | 4 | Trumaine McBride | 79.01 | 80.50 | 82.38 | 212 | Giants |
| 22 | 5 | Malcolm Butler | 78.81 | 73.70 | 85.35 | 217 | Patriots |
| 23 | 6 | Joe Haden | 78.58 | 69.70 | 82.21 | 1023 | Browns |
| 24 | 7 | Jason Verrett | 78.50 | 82.50 | 85.08 | 219 | Chargers |
| 25 | 8 | William Gay | 77.88 | 73.90 | 76.36 | 919 | Steelers |
| 26 | 9 | Dre Kirkpatrick | 77.73 | 75.20 | 79.61 | 276 | Bengals |
| 27 | 10 | Stephon Gilmore | 76.97 | 73.60 | 77.65 | 838 | Bills |
| 28 | 11 | E.J. Gaines | 76.67 | 70.30 | 77.78 | 937 | Rams |
| 29 | 12 | Tramon Williams | 76.43 | 67.70 | 78.08 | 1138 | Packers |
| 30 | 13 | Brandon Harris | 75.77 | 71.70 | 81.62 | 106 | Titans |
| 31 | 14 | Bradley Roby | 75.69 | 69.90 | 75.39 | 844 | Broncos |
| 32 | 15 | Nolan Carroll | 75.64 | 68.90 | 76.70 | 374 | Eagles |
| 33 | 16 | Janoris Jenkins | 75.21 | 70.00 | 75.77 | 838 | Rams |
| 34 | 17 | Darius Slay | 75.07 | 70.00 | 76.23 | 1073 | Lions |
| 35 | 18 | Josh Norman | 75.04 | 72.50 | 77.77 | 730 | Panthers |
| 36 | 19 | Aqib Talib | 74.95 | 66.10 | 77.83 | 988 | Broncos |
| 37 | 20 | Tim Jennings | 74.62 | 67.80 | 75.42 | 1006 | Bears |
| 38 | 21 | Cary Williams | 74.46 | 65.60 | 76.20 | 1154 | Eagles |
| 39 | 22 | Sterling Moore | 74.17 | 70.70 | 77.00 | 839 | Cowboys |
| 40 | 23 | Jumal Rolle | 74.08 | 70.10 | 79.94 | 202 | Texans |

### Starter (63 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 41 | 1 | Josh Gordy | 73.33 | 66.60 | 75.74 | 285 | Colts |
| 42 | 2 | Captain Munnerlyn | 73.30 | 64.70 | 74.86 | 1062 | Vikings |
| 43 | 3 | Davon House | 73.23 | 67.30 | 76.76 | 405 | Packers |
| 44 | 4 | Alterraun Verner | 73.23 | 63.80 | 76.39 | 859 | Buccaneers |
| 45 | 5 | Steve Williams | 73.21 | 71.80 | 77.29 | 130 | Chargers |
| 46 | 6 | Perrish Cox | 73.20 | 64.00 | 75.90 | 941 | 49ers |
| 47 | 7 | Justin Gilbert | 73.08 | 64.30 | 77.90 | 362 | Browns |
| 48 | 8 | Prince Amukamara | 72.99 | 66.40 | 78.00 | 458 | Giants |
| 49 | 9 | Josh Robinson | 72.78 | 65.10 | 73.73 | 671 | Vikings |
| 50 | 10 | Adam Jones | 72.39 | 65.50 | 72.82 | 838 | Bengals |
| 51 | 11 | Mike Harris | 72.33 | 70.50 | 75.74 | 216 | Giants |
| 52 | 12 | Leon Hall | 72.30 | 66.90 | 75.38 | 951 | Bengals |
| 53 | 13 | Sam Shields | 72.23 | 63.20 | 75.23 | 929 | Packers |
| 54 | 14 | Kayvon Webster | 72.21 | 66.60 | 76.34 | 130 | Broncos |
| 55 | 15 | Antonio Cromartie | 72.16 | 62.70 | 74.30 | 1064 | Cardinals |
| 56 | 16 | Tyler Patmon | 72.13 | 73.20 | 73.50 | 113 | Cowboys |
| 57 | 17 | Patrick Robinson | 72.11 | 66.10 | 77.36 | 612 | Saints |
| 58 | 18 | Brent Grimes | 71.97 | 64.80 | 75.71 | 1014 | Dolphins |
| 59 | 19 | Jason McCourty | 71.77 | 59.40 | 75.85 | 1077 | Titans |
| 60 | 20 | Phillip Gaines | 71.29 | 70.50 | 75.99 | 370 | Chiefs |
| 61 | 21 | Logan Ryan | 71.27 | 58.50 | 75.61 | 571 | Patriots |
| 62 | 22 | Trumaine Johnson | 70.79 | 63.30 | 75.46 | 433 | Rams |
| 63 | 23 | Kyle Arrington | 70.73 | 60.80 | 73.19 | 565 | Patriots |
| 64 | 24 | Byron Maxwell | 70.68 | 60.60 | 76.25 | 842 | Seahawks |
| 65 | 25 | T.J. Carrie | 70.58 | 64.80 | 74.44 | 543 | Raiders |
| 66 | 26 | Patrick Peterson | 70.40 | 62.00 | 71.83 | 1042 | Cardinals |
| 67 | 27 | Brice McCain | 70.17 | 63.30 | 72.46 | 659 | Steelers |
| 68 | 28 | Jeremy Lane | 69.94 | 67.60 | 74.63 | 274 | Seahawks |
| 69 | 29 | Marcus Burley | 69.35 | 66.00 | 72.61 | 321 | Seahawks |
| 70 | 30 | Tarell Brown | 69.31 | 57.70 | 73.92 | 959 | Raiders |
| 71 | 31 | Valentino Blake | 69.31 | 64.50 | 76.30 | 287 | Steelers |
| 72 | 32 | Nickell Robey-Coleman | 69.10 | 60.20 | 70.87 | 642 | Bills |
| 73 | 33 | A.J. Bouye | 69.00 | 65.20 | 74.53 | 634 | Texans |
| 74 | 34 | Johnthan Banks | 68.84 | 62.30 | 69.68 | 910 | Buccaneers |
| 75 | 35 | Leodis McKelvin | 68.62 | 62.70 | 73.30 | 535 | Bills |
| 76 | 36 | Josh Wilson | 68.59 | 59.70 | 71.92 | 448 | Falcons |
| 77 | 37 | Lardarius Webb | 68.58 | 60.40 | 72.46 | 916 | Ravens |
| 78 | 38 | Demetrius McCray | 68.55 | 64.00 | 70.93 | 815 | Jaguars |
| 79 | 39 | Danny Gorrer | 68.47 | 66.50 | 74.26 | 321 | Ravens |
| 80 | 40 | Dontae Johnson | 68.45 | 62.00 | 71.71 | 490 | 49ers |
| 81 | 41 | Buster Skrine | 68.40 | 56.50 | 72.17 | 1130 | Browns |
| 82 | 42 | Pierre Desir | 68.05 | 68.90 | 83.31 | 118 | Browns |
| 83 | 43 | Alan Ball | 67.91 | 61.70 | 74.55 | 499 | Jaguars |
| 84 | 44 | Terence Newman | 67.88 | 58.70 | 71.82 | 907 | Bengals |
| 85 | 45 | Cortland Finnegan | 66.89 | 63.00 | 70.22 | 705 | Dolphins |
| 86 | 46 | Darryl Morris | 66.85 | 64.90 | 72.59 | 259 | Texans |
| 87 | 47 | Marcus Williams | 65.42 | 60.80 | 72.66 | 446 | Jets |
| 88 | 48 | Kyle Wilson | 64.84 | 55.70 | 67.29 | 309 | Jets |
| 89 | 49 | D.J. Hayden | 64.83 | 60.00 | 70.91 | 581 | Raiders |
| 90 | 50 | Darrin Walls | 64.48 | 57.00 | 70.93 | 753 | Jets |
| 91 | 51 | Keenan Lewis | 64.46 | 51.80 | 68.74 | 899 | Saints |
| 92 | 52 | Brandon Carr | 64.35 | 52.20 | 68.29 | 1138 | Cowboys |
| 93 | 53 | Brandon Browner | 64.15 | 52.00 | 73.09 | 719 | Patriots |
| 94 | 54 | Carlos Rogers | 63.76 | 57.00 | 68.79 | 460 | Raiders |
| 95 | 55 | Dee Milliner | 63.50 | 58.30 | 72.44 | 116 | Jets |
| 96 | 56 | Bashaud Breeland | 63.49 | 49.90 | 68.39 | 864 | Commanders |
| 97 | 57 | Coty Sensabaugh | 63.49 | 56.30 | 66.71 | 721 | Titans |
| 98 | 58 | Aaron Colvin | 63.21 | 68.30 | 69.06 | 277 | Jaguars |
| 99 | 59 | Brandon Dixon | 63.20 | 56.50 | 70.80 | 163 | Buccaneers |
| 100 | 60 | Zackary Bowman | 63.00 | 51.70 | 69.80 | 450 | Giants |
| 101 | 61 | Marcus Roberson | 62.67 | 62.60 | 77.05 | 130 | Rams |
| 102 | 62 | Robert Alford | 62.54 | 50.60 | 70.63 | 617 | Falcons |
| 103 | 63 | Bradley Fletcher | 62.19 | 46.90 | 69.37 | 1053 | Eagles |

### Rotation/backup (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 104 | 1 | DeAngelo Hall | 61.27 | 50.60 | 70.99 | 145 | Commanders |
| 105 | 2 | Brian Dixon | 61.15 | 64.80 | 57.69 | 163 | Saints |
| 106 | 3 | Greg Toler | 60.84 | 50.10 | 66.97 | 1166 | Colts |
| 107 | 4 | Ron Brooks | 60.47 | 60.10 | 64.57 | 145 | Bills |
| 108 | 5 | Robert McClain | 60.40 | 49.80 | 64.34 | 628 | Falcons |
| 109 | 6 | Melvin White | 60.33 | 51.00 | 64.08 | 512 | Panthers |
| 110 | 7 | Leonard Johnson | 59.98 | 46.30 | 65.77 | 387 | Buccaneers |
| 111 | 8 | Cassius Vaughn | 59.61 | 50.00 | 64.67 | 425 | Lions |
| 112 | 9 | Kyle Fuller | 59.06 | 46.30 | 63.40 | 858 | Bears |
| 113 | 10 | Chris Owens | 58.91 | 50.00 | 65.58 | 490 | Chiefs |
| 114 | 11 | E.J. Biggers | 58.86 | 44.50 | 65.93 | 452 | Commanders |
| 115 | 12 | Chykie Brown | 58.74 | 50.00 | 64.56 | 503 | Giants |
| 116 | 13 | Dwayne Gratz | 58.69 | 47.10 | 65.25 | 855 | Jaguars |
| 117 | 14 | Rashaan Melvin | 58.59 | 59.30 | 65.61 | 303 | Ravens |
| 118 | 15 | David Amerson | 58.30 | 43.30 | 64.79 | 905 | Commanders |
| 119 | 16 | Antoine Cason | 58.09 | 39.20 | 69.95 | 679 | Ravens |
| 120 | 17 | Tharold Simon | 57.97 | 45.40 | 66.35 | 410 | Seahawks |
| 121 | 18 | Will Davis | 57.78 | 56.30 | 65.54 | 137 | Dolphins |
| 122 | 19 | R.J. Stanford | 56.88 | 56.90 | 64.56 | 136 | Dolphins |
| 123 | 20 | Justin Bethel | 56.62 | 57.00 | 62.72 | 101 | Cardinals |
| 124 | 21 | Alfonzo Dennard | 56.34 | 43.50 | 67.08 | 236 | Patriots |
| 125 | 22 | Antonio Allen | 55.32 | 50.40 | 55.47 | 515 | Jets |
| 126 | 23 | Cortez Allen | 55.16 | 38.00 | 67.44 | 453 | Steelers |
| 127 | 24 | Phillip Adams | 54.61 | 43.30 | 62.99 | 307 | Jets |
| 128 | 25 | Will Blackmon | 54.40 | 43.40 | 64.97 | 355 | Jaguars |
| 129 | 26 | Corey White | 54.05 | 37.70 | 62.77 | 757 | Saints |
| 130 | 27 | Isaiah Frey | 53.92 | 52.50 | 55.90 | 227 | Buccaneers |
| 131 | 28 | Shareece Wright | 53.80 | 37.10 | 63.58 | 833 | Chargers |
| 132 | 29 | Demontre Hurst | 53.76 | 52.70 | 55.50 | 366 | Bears |
| 133 | 30 | Morris Claiborne | 52.42 | 39.80 | 65.00 | 150 | Cowboys |
| 134 | 31 | Jayron Hosley | 51.67 | 53.00 | 55.78 | 150 | Giants |
| 135 | 32 | Marcus Cooper | 51.45 | 31.90 | 66.19 | 287 | Chiefs |
| 136 | 33 | Blidi Wreh-Wilson | 50.57 | 33.70 | 64.42 | 669 | Titans |
| 137 | 34 | Jamar Taylor | 50.26 | 52.70 | 52.15 | 294 | Dolphins |
| 138 | 35 | Chris Davis | 49.89 | 55.00 | 53.18 | 112 | Chargers |
| 139 | 36 | Asa Jackson | 47.44 | 35.90 | 57.15 | 323 | Ravens |
| 140 | 37 | Terrence Frederick | 45.00 | 45.40 | 50.00 | 190 | Saints |
| 141 | 38 | Al Louis-Jean | 45.00 | 29.20 | 50.68 | 119 | Bears |

## DI — Defensive Interior

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | J.J. Watt | 97.46 | 89.58 | 98.54 | 1050 | Texans |
| 2 | 2 | Aaron Donald | 92.65 | 89.12 | 90.83 | 707 | Rams |
| 3 | 3 | Ryan Davis Sr. | 90.74 | 87.93 | 88.45 | 303 | Jaguars |
| 4 | 4 | Calais Campbell | 88.35 | 87.83 | 85.68 | 858 | Cardinals |
| 5 | 5 | Sheldon Richardson | 87.97 | 89.47 | 82.81 | 810 | Jets |
| 6 | 6 | Ndamukong Suh | 87.30 | 90.11 | 81.26 | 903 | Lions |
| 7 | 7 | Marcell Dareus | 86.56 | 89.09 | 81.22 | 678 | Bills |
| 8 | 8 | Muhammad Wilkerson | 86.24 | 86.47 | 83.48 | 718 | Jets |
| 9 | 9 | Geno Atkins | 85.93 | 85.15 | 84.47 | 790 | Bengals |
| 10 | 10 | Malik Jackson | 85.40 | 81.69 | 84.12 | 609 | Broncos |
| 11 | 11 | Johnathan Hankins | 84.91 | 85.61 | 82.23 | 682 | Giants |
| 12 | 12 | Jurrell Casey | 84.48 | 83.92 | 81.00 | 909 | Titans |
| 13 | 13 | Kyle Williams | 83.36 | 77.84 | 83.39 | 717 | Bills |
| 14 | 14 | Kawann Short | 82.77 | 81.86 | 79.21 | 656 | Panthers |
| 15 | 15 | Sharrif Floyd | 82.44 | 83.44 | 78.91 | 568 | Vikings |
| 16 | 16 | Gerald McCoy | 82.38 | 89.07 | 75.32 | 665 | Buccaneers |
| 17 | 17 | Star Lotulelei | 82.04 | 76.45 | 82.25 | 508 | Panthers |
| 18 | 18 | Timmy Jernigan | 80.87 | 77.81 | 81.87 | 322 | Ravens |
| 19 | 19 | Brandon Williams | 80.75 | 79.18 | 81.15 | 623 | Ravens |
| 20 | 20 | Nick Fairley | 80.73 | 79.15 | 82.72 | 286 | Lions |
| 21 | 21 | Dan Williams | 80.11 | 84.00 | 74.18 | 462 | Cardinals |
| 22 | 22 | Mike Daniels | 80.02 | 70.81 | 82.00 | 781 | Packers |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Fletcher Cox | 79.98 | 80.77 | 75.48 | 926 | Eagles |
| 24 | 2 | Cameron Heyward | 79.02 | 76.14 | 76.77 | 913 | Steelers |
| 25 | 3 | Haloti Ngata | 79.00 | 82.72 | 73.71 | 627 | Ravens |
| 26 | 4 | Damon Harrison Sr. | 78.71 | 74.90 | 79.36 | 485 | Jets |
| 27 | 5 | Linval Joseph | 78.48 | 75.62 | 76.53 | 729 | Vikings |
| 28 | 6 | Jason Hatcher | 77.90 | 70.49 | 80.56 | 503 | Commanders |
| 29 | 7 | Kenrick Ellis | 77.83 | 71.38 | 79.85 | 154 | Jets |
| 30 | 8 | Steve McLendon | 76.56 | 71.78 | 77.77 | 330 | Steelers |
| 31 | 9 | Justin Smith | 75.74 | 63.78 | 79.55 | 694 | 49ers |
| 32 | 10 | Tyrunn Walker | 75.30 | 73.78 | 75.66 | 303 | Saints |
| 33 | 11 | Tyrone Crawford | 75.01 | 70.72 | 73.70 | 718 | Cowboys |
| 34 | 12 | Terrance Knighton | 74.95 | 72.76 | 72.25 | 566 | Broncos |
| 35 | 13 | Sen'Derrick Marks | 74.88 | 72.17 | 72.94 | 718 | Jaguars |
| 36 | 14 | Leger Douzable | 74.59 | 63.48 | 77.83 | 318 | Jets |
| 37 | 15 | Ra'Shede Hageman | 74.46 | 54.26 | 83.76 | 220 | Falcons |
| 38 | 16 | Bennie Logan | 74.32 | 62.54 | 78.00 | 639 | Eagles |
| 39 | 17 | Vance Walker | 74.18 | 66.24 | 75.63 | 229 | Chiefs |
| 40 | 18 | Akiem Hicks | 74.16 | 63.17 | 78.25 | 717 | Saints |

### Starter (83 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 41 | 1 | Ian Williams | 73.82 | 64.40 | 86.67 | 212 | 49ers |
| 42 | 2 | Karl Klug | 73.81 | 63.29 | 76.65 | 330 | Titans |
| 43 | 3 | Stefan Charles | 72.90 | 55.82 | 83.25 | 335 | Bills |
| 44 | 4 | Paul Soliai | 72.64 | 62.43 | 76.12 | 505 | Falcons |
| 45 | 5 | Earl Mitchell | 72.56 | 62.00 | 75.43 | 536 | Dolphins |
| 46 | 6 | Chris Baker | 72.54 | 59.91 | 77.83 | 503 | Commanders |
| 47 | 7 | Tyson Jackson | 72.53 | 68.30 | 71.39 | 512 | Falcons |
| 48 | 8 | Clinton McDonald | 72.35 | 63.65 | 75.55 | 619 | Buccaneers |
| 49 | 9 | Michael Brockers | 72.32 | 67.76 | 71.82 | 622 | Rams |
| 50 | 10 | Ego Ferguson | 71.97 | 60.11 | 75.71 | 312 | Bears |
| 51 | 11 | Ropati Pitoitua | 71.80 | 59.44 | 78.16 | 390 | Titans |
| 52 | 12 | Randy Starks | 71.79 | 63.85 | 73.44 | 540 | Dolphins |
| 53 | 13 | Corey Liuget | 71.64 | 59.85 | 75.33 | 775 | Chargers |
| 54 | 14 | Henry Melton | 71.63 | 60.37 | 79.45 | 424 | Cowboys |
| 55 | 15 | Jared Odrick | 71.31 | 66.41 | 70.41 | 807 | Dolphins |
| 56 | 16 | Jordan Hill | 71.29 | 55.87 | 84.03 | 360 | Seahawks |
| 57 | 17 | Stephen Paea | 71.09 | 63.59 | 73.08 | 700 | Bears |
| 58 | 18 | Brandon Bair | 70.99 | 40.74 | 86.99 | 196 | Eagles |
| 59 | 19 | Brandon Mebane | 70.95 | 62.71 | 75.92 | 280 | Seahawks |
| 60 | 20 | Sealver Siliga | 70.91 | 63.73 | 80.60 | 357 | Patriots |
| 61 | 21 | Cedric Thornton | 70.49 | 56.17 | 75.87 | 641 | Eagles |
| 62 | 22 | Jaye Howard Jr. | 70.43 | 56.26 | 82.37 | 437 | Chiefs |
| 63 | 23 | C.J. Mosley | 70.28 | 62.96 | 71.00 | 528 | Lions |
| 64 | 24 | Sammie Lee Hill | 70.28 | 58.72 | 75.49 | 588 | Titans |
| 65 | 25 | Mike Martin | 70.25 | 60.30 | 75.74 | 347 | Titans |
| 66 | 26 | Sean Lissemore | 70.16 | 58.25 | 76.01 | 332 | Chargers |
| 67 | 27 | John Jenkins | 70.12 | 60.43 | 75.01 | 390 | Saints |
| 68 | 28 | Derrick Shelby | 70.01 | 48.96 | 80.91 | 417 | Dolphins |
| 69 | 29 | Al Woods | 70.00 | 57.72 | 75.58 | 292 | Titans |
| 70 | 30 | C.J. Wilson | 69.93 | 59.69 | 75.40 | 362 | Raiders |
| 71 | 31 | Dontari Poe | 69.93 | 63.24 | 70.23 | 946 | Chiefs |
| 72 | 32 | Frostee Rucker | 69.85 | 50.78 | 78.40 | 539 | Cardinals |
| 73 | 33 | Desmond Bryant | 69.58 | 56.46 | 75.93 | 733 | Browns |
| 74 | 34 | Pat Sims | 69.36 | 57.14 | 74.80 | 419 | Raiders |
| 75 | 35 | Ricky Jean Francois | 69.23 | 60.33 | 72.24 | 728 | Colts |
| 76 | 36 | Lawrence Guy Sr. | 69.18 | 57.86 | 74.45 | 272 | Ravens |
| 77 | 37 | Arthur Jones | 69.07 | 56.22 | 76.17 | 517 | Colts |
| 78 | 38 | Cory Redding | 68.55 | 51.66 | 75.84 | 904 | Colts |
| 79 | 39 | Brodrick Bunkley | 68.37 | 56.74 | 75.39 | 274 | Saints |
| 80 | 40 | John Hughes | 68.28 | 66.69 | 71.22 | 207 | Browns |
| 81 | 41 | Red Bryant | 67.89 | 60.60 | 68.59 | 522 | Jaguars |
| 82 | 42 | Tom Johnson | 67.72 | 54.65 | 73.10 | 435 | Vikings |
| 83 | 43 | Quinton Dial | 67.70 | 61.90 | 73.78 | 326 | 49ers |
| 84 | 44 | Abry Jones | 67.70 | 52.17 | 77.02 | 376 | Jaguars |
| 85 | 45 | Kendall Langford | 67.55 | 59.94 | 68.45 | 481 | Rams |
| 86 | 46 | Derek Wolfe | 67.23 | 61.24 | 68.63 | 764 | Broncos |
| 87 | 47 | Montori Hughes | 67.06 | 59.94 | 72.98 | 225 | Colts |
| 88 | 48 | Chris Canty | 66.65 | 55.78 | 73.07 | 408 | Ravens |
| 89 | 49 | Jay Ratliff | 66.64 | 63.65 | 72.60 | 461 | Bears |
| 90 | 50 | Jared Crick | 66.59 | 53.85 | 70.92 | 714 | Texans |
| 91 | 51 | Tony McDaniel | 66.59 | 52.29 | 72.99 | 492 | Seahawks |
| 92 | 52 | Tenny Palepoi | 66.59 | 50.76 | 72.98 | 280 | Chargers |
| 93 | 53 | Tyson Alualu | 66.54 | 58.74 | 67.58 | 464 | Jaguars |
| 94 | 54 | Cullen Jenkins | 66.52 | 49.64 | 75.69 | 357 | Giants |
| 95 | 55 | Corey Peters | 66.45 | 56.99 | 70.25 | 526 | Falcons |
| 96 | 56 | Billy Winn | 66.41 | 55.31 | 72.78 | 499 | Browns |
| 97 | 57 | Kevin Vickerson | 66.21 | 48.91 | 76.18 | 168 | Chiefs |
| 98 | 58 | DeAngelo Tyson | 65.87 | 52.39 | 74.34 | 288 | Ravens |
| 99 | 59 | Sylvester Williams | 65.87 | 49.69 | 72.49 | 457 | Broncos |
| 100 | 60 | Datone Jones | 65.79 | 50.41 | 72.53 | 376 | Packers |
| 101 | 61 | Antonio Smith | 65.58 | 49.40 | 72.52 | 756 | Raiders |
| 102 | 62 | Ahtyba Rubin | 65.55 | 54.92 | 71.29 | 449 | Browns |
| 103 | 63 | Josh Chapman | 65.54 | 53.17 | 70.00 | 446 | Colts |
| 104 | 64 | Allen Bailey | 65.21 | 55.60 | 70.15 | 749 | Chiefs |
| 105 | 65 | Jonathan Babineaux | 65.00 | 52.57 | 69.63 | 695 | Falcons |
| 106 | 66 | Armonty Bryant | 64.94 | 50.80 | 78.92 | 132 | Browns |
| 107 | 67 | Zach Kerr | 64.83 | 54.65 | 71.61 | 287 | Colts |
| 108 | 68 | Vince Wilfork | 64.60 | 51.77 | 72.73 | 935 | Patriots |
| 109 | 69 | Domata Peko Sr. | 64.54 | 48.12 | 71.32 | 730 | Bengals |
| 110 | 70 | Demarcus Dobbs | 64.52 | 53.15 | 72.89 | 203 | Seahawks |
| 111 | 71 | Kevin Williams | 64.42 | 51.21 | 69.37 | 545 | Seahawks |
| 112 | 72 | Letroy Guion | 64.40 | 50.60 | 70.36 | 622 | Packers |
| 113 | 73 | Alan Branch | 64.32 | 52.32 | 70.76 | 231 | Patriots |
| 114 | 74 | Tommy Kelly | 63.91 | 51.73 | 71.30 | 765 | Cardinals |
| 115 | 75 | Phil Taylor Sr. | 63.74 | 54.25 | 73.60 | 130 | Browns |
| 116 | 76 | Dwan Edwards | 63.69 | 44.95 | 73.68 | 627 | Panthers |
| 117 | 77 | Ishmaa'ily Kitchen | 63.55 | 57.67 | 66.54 | 301 | Browns |
| 118 | 78 | Cam Thomas | 63.24 | 47.45 | 69.60 | 448 | Steelers |
| 119 | 79 | Chris Jones | 63.17 | 46.87 | 70.26 | 527 | Patriots |
| 120 | 80 | Roy Miller | 63.02 | 55.13 | 65.99 | 480 | Jaguars |
| 121 | 81 | Terrell McClain | 63.01 | 61.31 | 63.52 | 342 | Cowboys |
| 122 | 82 | Josh Boyd | 62.83 | 49.96 | 69.97 | 415 | Packers |
| 123 | 83 | Brandon Thompson | 62.49 | 52.68 | 69.65 | 254 | Bengals |

### Rotation/backup (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 124 | 1 | Akeem Spence | 61.95 | 49.97 | 65.77 | 492 | Buccaneers |
| 125 | 2 | Colin Cole | 61.81 | 40.75 | 72.31 | 407 | Panthers |
| 126 | 3 | Will Sutton III | 61.54 | 51.85 | 64.87 | 459 | Bears |
| 127 | 4 | Ricardo Mathews | 61.53 | 56.25 | 62.96 | 297 | Chargers |
| 128 | 5 | Shamar Stephen | 61.48 | 54.17 | 62.19 | 404 | Vikings |
| 129 | 6 | Casey Walker | 61.40 | 46.85 | 80.35 | 168 | Ravens |
| 130 | 7 | Mike Patterson | 61.34 | 45.68 | 69.90 | 422 | Giants |
| 131 | 8 | Marvin Austin | 60.78 | 58.26 | 61.42 | 294 | Broncos |
| 132 | 9 | Beau Allen | 60.53 | 48.93 | 64.10 | 195 | Eagles |
| 133 | 10 | Kendall Reyes | 60.42 | 48.76 | 64.02 | 643 | Chargers |
| 134 | 11 | Tony Jerod-Eddie | 60.28 | 46.80 | 68.23 | 420 | 49ers |
| 135 | 12 | Ryan Pickett | 60.14 | 42.78 | 69.12 | 287 | Texans |
| 136 | 13 | Jarvis Jenkins | 60.07 | 49.24 | 64.37 | 541 | Commanders |
| 137 | 14 | Josh Mauro | 60.06 | 61.36 | 68.44 | 105 | Cardinals |
| 138 | 15 | Brett Keisel | 59.71 | 43.80 | 69.48 | 439 | Steelers |
| 139 | 16 | Alex Carrington | 59.07 | 50.30 | 68.98 | 147 | Rams |
| 140 | 17 | Stephen Bowen | 59.02 | 46.70 | 69.12 | 241 | Commanders |
| 141 | 18 | Frank Kearse | 58.94 | 50.31 | 67.10 | 259 | Commanders |
| 142 | 19 | Barry Cofield | 58.78 | 48.35 | 65.73 | 249 | Commanders |
| 143 | 20 | Justin Ellis | 58.48 | 57.97 | 54.66 | 622 | Raiders |
| 144 | 21 | Andre Fluellen | 58.35 | 49.45 | 66.99 | 180 | Lions |
| 145 | 22 | Stephon Tuitt | 57.97 | 47.09 | 61.06 | 449 | Steelers |
| 146 | 23 | Da'Quan Bowers | 57.89 | 43.97 | 68.21 | 344 | Buccaneers |
| 147 | 24 | Mike Pennel | 57.88 | 47.36 | 61.76 | 179 | Packers |
| 148 | 25 | DaQuan Jones | 57.67 | 53.37 | 67.23 | 136 | Titans |
| 149 | 26 | Corbin Bryant | 57.11 | 44.06 | 62.02 | 353 | Bills |
| 150 | 27 | Nick Hayden | 56.87 | 41.12 | 63.62 | 662 | Cowboys |
| 151 | 28 | Dominique Easley | 56.73 | 52.42 | 60.64 | 263 | Patriots |
| 152 | 29 | Jerrell Powe | 56.72 | 52.75 | 61.04 | 273 | Texans |
| 153 | 30 | Stacy McGee | 56.72 | 47.22 | 62.53 | 118 | Raiders |
| 154 | 31 | Devon Still | 56.27 | 49.60 | 62.19 | 230 | Bengals |
| 155 | 32 | Tim Jamison | 55.06 | 39.15 | 66.18 | 411 | Texans |
| 156 | 33 | Kedric Golston | 54.56 | 39.87 | 62.78 | 178 | Commanders |
| 157 | 34 | Brandon Deaderick | 54.50 | 43.19 | 60.37 | 337 | Saints |
| 158 | 35 | Markus Kuhn | 53.61 | 42.20 | 64.04 | 249 | Giants |
| 159 | 36 | Jeoffrey Pagan | 53.11 | 46.87 | 53.10 | 189 | Texans |
| 160 | 37 | Ed Stinson | 49.56 | 46.95 | 53.39 | 204 | Cardinals |
| 161 | 38 | Caraun Reid | 49.37 | 46.65 | 51.19 | 111 | Lions |
| 162 | 39 | Sione Fua | 48.79 | 42.80 | 53.20 | 252 | Browns |
| 163 | 40 | Jay Bromley | 47.66 | 50.76 | 49.76 | 112 | Giants |
| 164 | 41 | Ethan Westbrooks | 46.98 | 50.18 | 54.09 | 114 | Rams |

## ED — Edge

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 93.84 | 97.43 | 89.47 | 973 | Broncos |
| 2 | 2 | Justin Houston | 92.29 | 91.20 | 90.10 | 1033 | Chiefs |
| 3 | 3 | Brandon Graham | 91.06 | 92.83 | 85.72 | 503 | Eagles |
| 4 | 4 | Pernell McPhee | 88.50 | 88.50 | 84.34 | 591 | Ravens |
| 5 | 5 | Ezekiel Ansah | 88.44 | 93.47 | 81.71 | 704 | Lions |
| 6 | 6 | Robert Quinn | 87.62 | 92.05 | 80.50 | 786 | Rams |
| 7 | 7 | Charles Johnson | 87.09 | 84.50 | 84.97 | 876 | Panthers |
| 8 | 8 | Cameron Wake | 86.57 | 80.20 | 86.96 | 759 | Dolphins |
| 9 | 9 | Mario Williams | 86.35 | 87.19 | 81.63 | 787 | Bills |
| 10 | 10 | Jerry Hughes | 85.50 | 85.09 | 81.60 | 782 | Bills |
| 11 | 11 | Junior Galette | 83.63 | 77.75 | 83.39 | 794 | Saints |
| 12 | 12 | Connor Barwin | 83.11 | 66.14 | 90.26 | 1008 | Eagles |
| 13 | 13 | Carlos Dunlap | 82.39 | 84.94 | 76.72 | 1001 | Bengals |
| 14 | 14 | DeMarcus Ware | 82.22 | 73.11 | 85.06 | 789 | Broncos |
| 15 | 15 | Michael Bennett | 81.28 | 88.21 | 72.49 | 1022 | Seahawks |
| 16 | 16 | Cliff Avril | 80.53 | 75.79 | 79.53 | 847 | Seahawks |
| 17 | 17 | Terrell Suggs | 80.37 | 77.36 | 79.04 | 957 | Ravens |
| 18 | 18 | Khalil Mack | 80.16 | 90.18 | 69.32 | 992 | Raiders |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Elvis Dumervil | 79.77 | 62.27 | 87.27 | 693 | Ravens |
| 20 | 2 | Arthur Moats | 78.34 | 63.44 | 84.10 | 344 | Steelers |
| 21 | 3 | James Harrison | 78.03 | 63.19 | 87.93 | 485 | Steelers |
| 22 | 4 | Clay Matthews | 77.95 | 61.33 | 84.86 | 1010 | Packers |
| 23 | 5 | Jason Babin | 77.88 | 63.62 | 83.22 | 458 | Jets |
| 24 | 6 | Ryan Kerrigan | 77.19 | 66.24 | 80.32 | 978 | Commanders |
| 25 | 7 | Everson Griffen | 76.92 | 77.75 | 72.20 | 967 | Vikings |
| 26 | 8 | Paul Kruger | 76.74 | 66.53 | 79.38 | 899 | Browns |
| 27 | 9 | Jason Pierre-Paul | 76.36 | 81.53 | 70.31 | 960 | Giants |
| 28 | 10 | Chandler Jones | 75.69 | 78.41 | 71.27 | 747 | Patriots |
| 29 | 11 | Quinton Coples | 75.31 | 66.17 | 77.24 | 691 | Jets |
| 30 | 12 | Vinny Curry | 74.66 | 65.91 | 78.73 | 377 | Eagles |
| 31 | 13 | William Hayes | 74.66 | 74.17 | 71.45 | 537 | Rams |
| 32 | 14 | Damontre Moore | 74.44 | 70.38 | 74.15 | 320 | Giants |

### Starter (53 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Cameron Jordan | 73.98 | 76.95 | 67.83 | 999 | Saints |
| 34 | 2 | Jason Worilds | 73.15 | 64.44 | 74.79 | 1031 | Steelers |
| 35 | 3 | Trent Cole | 73.05 | 62.77 | 76.77 | 806 | Eagles |
| 36 | 4 | Derrick Morgan | 72.90 | 61.62 | 76.25 | 1005 | Titans |
| 37 | 5 | Aaron Lynch | 72.75 | 64.08 | 74.36 | 514 | 49ers |
| 38 | 6 | Jonathan Newsome | 72.74 | 61.06 | 76.36 | 522 | Colts |
| 39 | 7 | Jared Allen | 72.47 | 59.39 | 77.54 | 888 | Bears |
| 40 | 8 | Justin Tuck | 72.33 | 66.02 | 73.11 | 640 | Raiders |
| 41 | 9 | Erik Walden | 72.31 | 57.68 | 77.90 | 845 | Colts |
| 42 | 10 | Dan Skuta | 72.15 | 58.40 | 80.29 | 385 | 49ers |
| 43 | 11 | Robert Ayers | 71.72 | 73.61 | 68.37 | 379 | Giants |
| 44 | 12 | Olivier Vernon | 71.59 | 73.37 | 66.24 | 839 | Dolphins |
| 45 | 13 | Julius Peppers | 71.34 | 61.14 | 73.97 | 904 | Packers |
| 46 | 14 | Barkevious Mingo | 70.77 | 69.01 | 68.81 | 668 | Browns |
| 47 | 15 | Chris Long | 70.33 | 59.80 | 78.38 | 232 | Rams |
| 48 | 16 | Willie Young | 70.07 | 61.77 | 72.47 | 664 | Bears |
| 49 | 17 | Sam Acho | 69.60 | 61.80 | 70.64 | 517 | Cardinals |
| 50 | 18 | Akeem Ayers | 69.37 | 65.00 | 71.25 | 416 | Patriots |
| 51 | 19 | Nick Perry | 68.35 | 61.27 | 68.90 | 420 | Packers |
| 52 | 20 | Jabaal Sheard | 67.88 | 67.06 | 65.19 | 675 | Browns |
| 53 | 21 | Dee Ford | 67.83 | 57.91 | 75.48 | 122 | Chiefs |
| 54 | 22 | David Bass | 67.61 | 65.48 | 71.63 | 143 | Bears |
| 55 | 23 | Dion Jordan | 67.56 | 68.99 | 66.34 | 224 | Dolphins |
| 56 | 24 | Parys Haralson | 67.50 | 51.09 | 74.27 | 488 | Saints |
| 57 | 25 | Whitney Mercilus | 67.29 | 63.31 | 66.81 | 804 | Texans |
| 58 | 26 | Lerentee McCray | 67.21 | 62.10 | 68.53 | 149 | Broncos |
| 59 | 27 | Rob Ninkovich | 67.14 | 53.77 | 71.88 | 1203 | Patriots |
| 60 | 28 | Chris Clemons | 67.09 | 54.66 | 71.21 | 790 | Jaguars |
| 61 | 29 | Jonathan Massaquoi | 67.04 | 62.59 | 68.85 | 327 | Falcons |
| 62 | 30 | Michael Johnson | 66.79 | 67.82 | 62.97 | 628 | Buccaneers |
| 63 | 31 | Jeremy Mincey | 66.25 | 61.91 | 64.98 | 790 | Cowboys |
| 64 | 32 | Jacquies Smith | 66.24 | 58.50 | 68.26 | 455 | Buccaneers |
| 65 | 33 | Alex Okafor | 66.13 | 57.00 | 75.22 | 770 | Cardinals |
| 66 | 34 | Anthony Spencer | 66.05 | 58.91 | 72.28 | 444 | Cowboys |
| 67 | 35 | Melvin Ingram III | 66.02 | 65.79 | 68.77 | 497 | Chargers |
| 68 | 36 | Mario Addison | 65.76 | 57.10 | 67.37 | 471 | Panthers |
| 69 | 37 | Aldon Smith | 65.38 | 68.23 | 70.18 | 418 | 49ers |
| 70 | 38 | Osi Umenyiora | 65.20 | 58.30 | 65.63 | 338 | Falcons |
| 71 | 39 | O'Brien Schofield | 65.11 | 56.57 | 69.04 | 426 | Seahawks |
| 72 | 40 | Jeremiah Attaochu | 65.09 | 63.89 | 66.93 | 178 | Chargers |
| 73 | 41 | Kony Ealy | 64.73 | 58.13 | 64.96 | 400 | Panthers |
| 74 | 42 | Kasim Edebali | 64.34 | 54.25 | 68.99 | 179 | Saints |
| 75 | 43 | Tamba Hali | 64.20 | 57.19 | 64.71 | 975 | Chiefs |
| 76 | 44 | Brian Robison | 64.20 | 54.19 | 66.71 | 908 | Vikings |
| 77 | 45 | Mike Neal | 63.72 | 59.59 | 62.31 | 1442 | Packers |
| 78 | 46 | Shaun Phillips | 63.70 | 46.37 | 71.08 | 486 | Colts |
| 79 | 47 | Devin Taylor | 63.63 | 61.79 | 61.47 | 241 | Lions |
| 80 | 48 | Ahmad Brooks | 63.56 | 51.85 | 70.34 | 603 | 49ers |
| 81 | 49 | Calvin Pace | 63.28 | 48.92 | 68.68 | 823 | Jets |
| 82 | 50 | Darryl Tapp | 63.10 | 55.24 | 66.36 | 297 | Lions |
| 83 | 51 | Manny Lawson | 62.49 | 47.06 | 68.93 | 341 | Bills |
| 84 | 52 | Brooks Reed | 62.36 | 57.18 | 62.06 | 787 | Texans |
| 85 | 53 | DeMarcus Lawrence | 62.24 | 75.05 | 56.83 | 270 | Cowboys |

### Rotation/backup (39 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 86 | 1 | John Simon | 61.97 | 62.97 | 66.90 | 233 | Texans |
| 87 | 2 | Dwight Freeney | 61.91 | 48.16 | 66.91 | 572 | Chargers |
| 88 | 3 | Bjoern Werner | 61.70 | 58.04 | 59.97 | 807 | Colts |
| 89 | 4 | Trent Murphy | 61.53 | 62.48 | 57.77 | 579 | Commanders |
| 90 | 5 | Wallace Gilberry | 61.00 | 46.27 | 66.85 | 883 | Bengals |
| 91 | 6 | Kroy Biermann | 60.92 | 58.75 | 62.56 | 848 | Falcons |
| 92 | 7 | William Gholston | 60.76 | 61.45 | 58.35 | 570 | Buccaneers |
| 93 | 8 | George Johnson | 60.47 | 50.88 | 65.83 | 520 | Lions |
| 94 | 9 | George Selvie | 59.79 | 51.46 | 62.65 | 564 | Cowboys |
| 95 | 10 | Andre Branch | 59.68 | 62.91 | 57.62 | 330 | Jaguars |
| 96 | 11 | Jarius Wynn | 59.64 | 57.75 | 61.93 | 311 | Bills |
| 97 | 12 | Jack Crawford | 59.57 | 56.25 | 71.04 | 143 | Cowboys |
| 98 | 13 | Wes Horton | 59.30 | 57.81 | 58.47 | 498 | Panthers |
| 99 | 14 | Lamarr Houston | 58.86 | 61.82 | 61.05 | 396 | Bears |
| 100 | 15 | Matt Shaughnessy | 58.77 | 55.95 | 60.13 | 357 | Cardinals |
| 101 | 16 | Brian Orakpo | 58.64 | 61.99 | 63.11 | 388 | Commanders |
| 102 | 17 | Denico Autry | 58.59 | 60.75 | 59.24 | 130 | Raiders |
| 103 | 18 | Mathias Kiwanuka | 58.58 | 48.18 | 63.94 | 552 | Giants |
| 104 | 19 | Benson Mayowa | 57.69 | 66.59 | 53.71 | 353 | Raiders |
| 105 | 20 | Eugene Sims | 57.01 | 53.68 | 55.70 | 490 | Rams |
| 106 | 21 | Scott Solomon | 56.85 | 61.22 | 56.16 | 233 | Browns |
| 107 | 22 | Jarvis Jones | 56.01 | 55.70 | 60.39 | 236 | Steelers |
| 108 | 23 | Tourek Williams | 55.88 | 56.93 | 54.14 | 132 | Chargers |
| 109 | 24 | Kerry Wynn | 55.52 | 63.81 | 61.79 | 185 | Giants |
| 110 | 25 | Malliciah Goodman | 54.89 | 57.31 | 49.90 | 581 | Falcons |
| 111 | 26 | Quentin Groves | 54.39 | 46.43 | 57.62 | 242 | Titans |
| 112 | 27 | T.J. Fatinikun | 53.89 | 57.04 | 53.88 | 151 | Buccaneers |
| 113 | 28 | Quanterus Smith | 53.53 | 54.22 | 49.93 | 304 | Broncos |
| 114 | 29 | Jarret Johnson | 53.32 | 45.18 | 55.62 | 547 | Chargers |
| 115 | 30 | Kamerion Wimbley | 52.63 | 46.87 | 55.44 | 539 | Titans |
| 116 | 31 | Larry English | 52.49 | 51.27 | 53.31 | 254 | Buccaneers |
| 117 | 32 | Andy Studebaker | 52.33 | 53.36 | 49.56 | 199 | Colts |
| 118 | 33 | Kareem Martin | 52.26 | 54.30 | 51.94 | 182 | Cardinals |
| 119 | 34 | Jason Jones | 51.08 | 45.12 | 54.96 | 686 | Lions |
| 120 | 35 | Cliff Matthews | 50.20 | 51.04 | 52.77 | 116 | Falcons |
| 121 | 36 | Robert Geathers | 49.98 | 45.37 | 53.26 | 624 | Bengals |
| 122 | 37 | Corey Lemonier | 47.01 | 53.36 | 46.94 | 143 | 49ers |
| 123 | 38 | Jadeveon Clowney | 46.68 | 63.46 | 49.82 | 143 | Texans |
| 124 | 39 | Lamarr Woodley | 45.00 | 49.01 | 50.67 | 285 | Raiders |

## G — Guard

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Marshal Yanda | 97.88 | 93.00 | 96.96 | 1196 | Ravens |
| 2 | 2 | Evan Mathis | 95.80 | 90.90 | 94.90 | 597 | Eagles |
| 3 | 3 | Kevin Zeitler | 93.38 | 89.10 | 92.07 | 799 | Bengals |
| 4 | 4 | Josh Sitton | 91.55 | 87.10 | 90.35 | 1133 | Packers |
| 5 | 5 | Kelechi Osemele | 91.54 | 85.00 | 91.74 | 1069 | Ravens |
| 6 | 6 | Zack Martin | 91.23 | 86.40 | 90.28 | 1172 | Cowboys |
| 7 | 7 | Brandon Brooks | 91.05 | 85.50 | 90.59 | 985 | Texans |
| 8 | 8 | T.J. Lang | 90.03 | 85.90 | 88.61 | 1085 | Packers |
| 9 | 9 | Chance Warmack | 88.17 | 81.30 | 88.58 | 965 | Titans |
| 10 | 10 | Mike Iupati | 87.02 | 80.70 | 87.06 | 946 | 49ers |
| 11 | 11 | Orlando Franklin | 86.75 | 78.00 | 88.42 | 1162 | Broncos |
| 12 | 12 | John Greco | 86.55 | 79.90 | 86.82 | 1052 | Browns |
| 13 | 13 | Joel Bitonio | 86.29 | 79.40 | 86.72 | 1052 | Browns |
| 14 | 14 | Mike Pollak | 85.66 | 79.30 | 85.73 | 446 | Bengals |
| 15 | 15 | Ron Leary | 84.77 | 78.00 | 85.11 | 1110 | Cowboys |
| 16 | 16 | David DeCastro | 84.66 | 78.10 | 84.86 | 1185 | Steelers |
| 17 | 17 | Kyle Long | 84.26 | 76.20 | 85.46 | 995 | Bears |
| 18 | 18 | Logan Mankins | 83.99 | 74.70 | 86.01 | 910 | Buccaneers |
| 19 | 19 | Louis Vasquez | 83.44 | 75.50 | 84.56 | 1192 | Broncos |
| 20 | 20 | Joe Berger | 82.42 | 74.80 | 83.34 | 614 | Vikings |
| 21 | 21 | J.R. Sweezy | 82.29 | 73.50 | 83.98 | 1231 | Seahawks |
| 22 | 22 | Jon Asamoah | 82.06 | 74.80 | 82.74 | 945 | Falcons |
| 23 | 23 | Jeremiah Sirles | 81.37 | 75.50 | 81.11 | 112 | Chargers |
| 24 | 24 | Andrew Norwell | 81.17 | 73.70 | 81.99 | 828 | Panthers |
| 25 | 25 | Alex Boone | 81.06 | 73.20 | 82.13 | 943 | 49ers |
| 26 | 26 | Andy Levitre | 81.01 | 71.40 | 83.25 | 966 | Titans |
| 27 | 27 | Rob Sims | 80.82 | 73.50 | 81.53 | 1160 | Lions |
| 28 | 28 | Matt Slauson | 80.66 | 71.00 | 82.93 | 264 | Bears |
| 29 | 29 | Brandon Fusco | 80.44 | 70.90 | 82.63 | 171 | Vikings |
| 30 | 30 | Clint Boling | 80.13 | 72.30 | 81.19 | 1124 | Bengals |
| 31 | 31 | John Urschel | 80.00 | 76.20 | 78.37 | 356 | Ravens |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 32 | 1 | Larry Warford | 79.77 | 72.90 | 80.18 | 784 | Lions |
| 33 | 2 | Justin Blalock | 79.57 | 72.00 | 80.45 | 970 | Falcons |
| 34 | 3 | Oday Aboushi | 79.44 | 71.70 | 80.44 | 722 | Jets |
| 35 | 4 | Gabe Jackson | 78.82 | 70.00 | 80.53 | 813 | Raiders |
| 36 | 5 | Chris Chester | 78.81 | 71.10 | 79.79 | 1051 | Commanders |
| 37 | 6 | Jahri Evans | 78.80 | 70.70 | 80.03 | 1139 | Saints |
| 38 | 7 | Ben Grubbs | 78.70 | 71.30 | 79.46 | 1133 | Saints |
| 39 | 8 | Josh Kline | 78.07 | 68.10 | 80.55 | 408 | Patriots |
| 40 | 9 | Jack Mewhort | 77.97 | 69.40 | 79.51 | 1191 | Colts |
| 41 | 10 | Trai Turner | 77.74 | 72.20 | 77.27 | 809 | Panthers |
| 42 | 11 | Ramon Foster | 77.29 | 69.10 | 78.59 | 1037 | Steelers |
| 43 | 12 | Patrick Omameh | 77.24 | 68.00 | 79.23 | 902 | Buccaneers |
| 44 | 13 | Hugh Thornton | 76.62 | 66.10 | 79.46 | 566 | Colts |
| 45 | 14 | John Jerry | 76.46 | 67.30 | 78.40 | 1127 | Giants |
| 46 | 15 | Willie Colon | 76.34 | 64.70 | 79.93 | 1080 | Jets |
| 47 | 16 | Rodger Saffold | 76.12 | 66.90 | 78.10 | 918 | Rams |
| 48 | 17 | Shawn Lauvao | 75.85 | 66.70 | 77.79 | 951 | Commanders |
| 49 | 18 | Ted Larsen | 75.45 | 63.30 | 79.38 | 1085 | Cardinals |
| 50 | 19 | Todd Herremans | 75.27 | 64.00 | 78.61 | 577 | Eagles |
| 51 | 20 | Ryan Groy | 75.26 | 64.70 | 78.14 | 226 | Bears |
| 52 | 21 | Matt Tobin | 75.13 | 64.50 | 78.05 | 523 | Eagles |
| 53 | 22 | Joe Reitz | 74.61 | 67.10 | 75.45 | 475 | Colts |
| 54 | 23 | Zane Beadles | 74.31 | 65.00 | 76.35 | 1037 | Jaguars |
| 55 | 24 | Lance Louis | 74.20 | 62.40 | 77.90 | 731 | Colts |

### Starter (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 56 | 1 | James Carpenter | 73.76 | 62.10 | 77.36 | 994 | Seahawks |
| 57 | 2 | Zach Fulton | 73.51 | 64.20 | 75.55 | 996 | Chiefs |
| 58 | 3 | Jonathan Cooper | 72.63 | 64.90 | 73.62 | 184 | Cardinals |
| 59 | 4 | Chad Rinehart | 72.22 | 62.70 | 74.40 | 1067 | Chargers |
| 60 | 5 | Fernando Velasco | 71.87 | 66.10 | 71.55 | 397 | Panthers |
| 61 | 6 | Kraig Urbik | 71.70 | 61.50 | 74.34 | 621 | Bills |
| 62 | 7 | Adam Snyder | 71.45 | 58.50 | 75.91 | 101 | Giants |
| 63 | 8 | Charlie Johnson | 71.43 | 61.90 | 73.62 | 863 | Vikings |
| 64 | 9 | Johnnie Troutman | 70.95 | 60.30 | 73.88 | 773 | Chargers |
| 65 | 10 | Paul Fanaika | 70.65 | 58.80 | 74.38 | 936 | Cardinals |
| 66 | 11 | Joe Looney | 70.33 | 61.70 | 71.91 | 330 | 49ers |
| 67 | 12 | Brian Winters | 69.39 | 55.90 | 74.22 | 371 | Jets |
| 68 | 13 | Vladimir Ducasse | 69.31 | 57.80 | 72.82 | 408 | Vikings |
| 69 | 14 | Dan Connolly | 69.27 | 56.80 | 73.41 | 1053 | Patriots |
| 70 | 15 | Amini Silatolu | 68.70 | 55.90 | 73.06 | 404 | Panthers |
| 71 | 16 | Davin Joseph | 68.49 | 55.50 | 72.98 | 876 | Rams |
| 72 | 17 | Cyril Richardson | 68.48 | 51.40 | 75.70 | 312 | Bills |
| 73 | 18 | Shelley Smith | 67.05 | 54.40 | 71.31 | 359 | Dolphins |
| 74 | 19 | Xavier Su'a-Filo | 66.88 | 52.20 | 72.50 | 127 | Texans |
| 75 | 20 | Daryn Colledge | 66.67 | 53.90 | 71.01 | 740 | Dolphins |
| 76 | 21 | Mike McGlynn | 64.20 | 50.40 | 69.24 | 805 | Chiefs |
| 77 | 22 | Chris Williams | 62.41 | 46.60 | 68.78 | 131 | Bills |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 78 | 1 | Dallas Thomas | 61.94 | 47.70 | 67.27 | 673 | Dolphins |
| 79 | 2 | Lane Taylor | 55.33 | 41.30 | 60.52 | 128 | Packers |
| 80 | 3 | Garrett Gilkey | 55.33 | 29.80 | 68.18 | 204 | Buccaneers |

## HB — Running Back

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Marshawn Lynch | 82.08 | 90.70 | 72.17 | 361 | Seahawks |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 2 | 1 | C.J. Anderson | 78.63 | 85.80 | 69.69 | 245 | Broncos |
| 3 | 2 | DeMarco Murray | 78.18 | 83.10 | 70.74 | 303 | Cowboys |
| 4 | 3 | Eddie Lacy | 77.09 | 83.00 | 68.98 | 340 | Packers |
| 5 | 4 | Jonathan Stewart | 76.42 | 79.10 | 70.46 | 246 | Panthers |
| 6 | 5 | Le'Veon Bell | 74.80 | 80.80 | 66.63 | 493 | Steelers |
| 7 | 6 | Chris Ivory | 74.79 | 72.00 | 72.48 | 149 | Jets |
| 8 | 7 | Pierre Thomas | 74.75 | 76.40 | 69.49 | 206 | Saints |

### Starter (45 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Jamaal Charles | 73.96 | 71.70 | 71.30 | 334 | Chiefs |
| 10 | 2 | Ahmad Bradshaw | 73.62 | 75.50 | 68.20 | 177 | Colts |
| 11 | 3 | Fred Jackson | 73.40 | 71.30 | 70.63 | 283 | Bills |
| 12 | 4 | Darren Sproles | 73.39 | 66.50 | 73.82 | 234 | Eagles |
| 13 | 5 | Justin Forsett | 72.77 | 72.00 | 69.12 | 346 | Ravens |
| 14 | 6 | Arian Foster | 71.21 | 71.70 | 66.71 | 269 | Texans |
| 15 | 7 | Jerick McKinnon | 71.13 | 68.40 | 68.79 | 154 | Vikings |
| 16 | 8 | Steven Jackson | 71.05 | 76.80 | 63.05 | 136 | Falcons |
| 17 | 9 | Lamar Miller | 70.30 | 73.90 | 63.74 | 290 | Dolphins |
| 18 | 10 | Giovani Bernard | 70.29 | 70.90 | 65.71 | 244 | Bengals |
| 19 | 11 | Reggie Bush | 70.25 | 71.60 | 65.19 | 187 | Lions |
| 20 | 12 | Theo Riddick | 70.16 | 79.10 | 60.03 | 147 | Lions |
| 21 | 13 | Carlos Hyde | 69.93 | 65.10 | 68.99 | 113 | 49ers |
| 22 | 14 | Jeremy Hill | 69.62 | 63.10 | 69.80 | 178 | Bengals |
| 23 | 15 | Joique Bell | 69.59 | 66.10 | 67.75 | 287 | Lions |
| 24 | 16 | Branden Oliver | 69.49 | 75.30 | 61.45 | 158 | Chargers |
| 25 | 17 | Mark Ingram II | 69.40 | 71.20 | 64.04 | 165 | Saints |
| 26 | 18 | Roy Helu | 69.39 | 66.50 | 67.15 | 217 | Commanders |
| 27 | 19 | Leon Washington | 69.07 | 60.40 | 70.68 | 167 | Titans |
| 28 | 20 | Jacquizz Rodgers | 68.61 | 68.80 | 64.31 | 187 | Falcons |
| 29 | 21 | Matt Forte | 68.60 | 63.60 | 67.77 | 509 | Bears |
| 30 | 22 | Alfred Morris | 68.14 | 62.80 | 67.53 | 202 | Commanders |
| 31 | 23 | Devonta Freeman | 67.26 | 69.10 | 61.86 | 125 | Falcons |
| 32 | 24 | LeSean McCoy | 66.84 | 57.90 | 68.63 | 363 | Eagles |
| 33 | 25 | Latavius Murray | 66.32 | 65.30 | 62.83 | 133 | Raiders |
| 34 | 26 | Bishop Sankey | 66.26 | 64.70 | 63.14 | 146 | Titans |
| 35 | 27 | Shane Vereen | 66.14 | 65.90 | 62.13 | 476 | Patriots |
| 36 | 28 | Dexter McCluster | 66.14 | 66.90 | 61.46 | 165 | Titans |
| 37 | 29 | Tre Mason | 66.12 | 65.80 | 62.16 | 120 | Rams |
| 38 | 30 | Frank Gore | 65.68 | 62.70 | 63.50 | 217 | 49ers |
| 39 | 31 | Benny Cunningham | 65.60 | 65.00 | 61.84 | 235 | Rams |
| 40 | 32 | Dan Herron | 65.34 | 62.50 | 63.07 | 199 | Colts |
| 41 | 33 | Jordan Todman | 64.71 | 66.60 | 59.28 | 173 | Jaguars |
| 42 | 34 | Doug Martin | 64.69 | 58.70 | 64.51 | 121 | Buccaneers |
| 43 | 35 | Darren McFadden | 64.54 | 60.20 | 63.26 | 203 | Raiders |
| 44 | 36 | Robert Turbin | 64.52 | 65.20 | 59.90 | 131 | Seahawks |
| 45 | 37 | James Starks | 64.33 | 54.70 | 66.58 | 149 | Packers |
| 46 | 38 | Donald Brown | 64.32 | 56.50 | 65.36 | 187 | Chargers |
| 47 | 39 | Rashad Jennings | 64.16 | 58.50 | 63.77 | 172 | Giants |
| 48 | 40 | Bobby Rainey Jr. | 64.02 | 57.50 | 64.20 | 189 | Buccaneers |
| 49 | 41 | Trent Richardson | 63.47 | 61.30 | 60.75 | 196 | Colts |
| 50 | 42 | Andre Ellington | 63.01 | 55.10 | 64.11 | 239 | Cardinals |
| 51 | 43 | Daniel Thomas | 62.97 | 66.30 | 56.59 | 120 | Dolphins |
| 52 | 44 | Toby Gerhart | 62.34 | 48.30 | 67.53 | 126 | Jaguars |
| 53 | 45 | Chris Johnson | 62.24 | 55.10 | 62.83 | 163 | Jets |

### Rotation/backup (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 54 | 1 | Terrance West | 61.72 | 58.90 | 59.44 | 142 | Browns |
| 55 | 2 | Bilal Powell | 61.53 | 61.60 | 57.32 | 127 | Jets |
| 56 | 3 | Denard Robinson | 61.17 | 55.90 | 60.52 | 132 | Jaguars |
| 57 | 4 | Andre Williams | 60.65 | 60.10 | 56.85 | 203 | Giants |
| 58 | 5 | Maurice Jones-Drew | 60.39 | 47.80 | 64.62 | 111 | Raiders |
| 59 | 6 | Travaris Cadet | 58.90 | 59.80 | 54.13 | 176 | Saints |
| 60 | 7 | Matt Asiata | 58.11 | 53.40 | 57.09 | 226 | Vikings |
| 61 | 8 | Isaiah Crowell | 57.62 | 52.40 | 56.94 | 156 | Browns |
| 62 | 9 | Ronnie Hillman | 57.57 | 51.30 | 57.58 | 157 | Broncos |
| 63 | 10 | Alfred Blue | 57.27 | 55.10 | 54.55 | 110 | Texans |
| 64 | 11 | Charles Sims | 55.83 | 52.50 | 53.88 | 115 | Buccaneers |
| 65 | 12 | Knile Davis | 54.05 | 45.20 | 55.78 | 125 | Chiefs |

## LB — Linebacker

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Luke Kuechly | 87.71 | 90.20 | 81.89 | 1089 | Panthers |
| 2 | 2 | Karlos Dansby | 82.59 | 86.30 | 78.03 | 815 | Browns |
| 3 | 3 | Mychal Kendricks | 81.51 | 88.70 | 74.84 | 762 | Eagles |
| 4 | 4 | Dont'a Hightower | 81.37 | 82.30 | 77.10 | 1015 | Patriots |
| 5 | 5 | Chris Borland | 81.07 | 89.70 | 78.45 | 476 | 49ers |
| 6 | 6 | Jamie Collins Sr. | 80.21 | 85.30 | 72.65 | 1110 | Patriots |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Thomas Davis Sr. | 79.83 | 82.20 | 74.29 | 1023 | Panthers |
| 8 | 2 | Rolando McClain | 78.41 | 81.70 | 74.34 | 662 | Cowboys |
| 9 | 3 | Bobby Wagner | 78.28 | 78.90 | 74.73 | 872 | Seahawks |
| 10 | 4 | Brandon Spikes | 76.02 | 77.70 | 70.74 | 503 | Bills |
| 11 | 5 | Daryl Smith | 75.28 | 76.40 | 73.28 | 1186 | Ravens |
| 12 | 6 | Brandon Marshall | 75.25 | 77.30 | 74.92 | 906 | Broncos |

### Starter (53 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Lavonte David | 72.94 | 71.10 | 71.04 | 919 | Buccaneers |
| 14 | 2 | C.J. Mosley | 72.82 | 70.20 | 70.40 | 1207 | Ravens |
| 15 | 3 | DeAndre Levy | 72.66 | 70.90 | 70.09 | 1107 | Lions |
| 16 | 4 | Koa Misi | 72.16 | 74.50 | 69.76 | 573 | Dolphins |
| 17 | 5 | Patrick Willis | 71.90 | 74.70 | 71.06 | 342 | 49ers |
| 18 | 6 | Jerrell Freeman | 71.24 | 67.50 | 70.08 | 962 | Colts |
| 19 | 7 | Mike Mohamed | 70.65 | 70.40 | 71.44 | 510 | Texans |
| 20 | 8 | Craig Robertson | 70.23 | 68.60 | 67.78 | 663 | Browns |
| 21 | 9 | Vince Williams | 69.99 | 69.90 | 67.58 | 257 | Steelers |
| 22 | 10 | Bruce Irvin | 69.81 | 66.80 | 67.65 | 872 | Seahawks |
| 23 | 11 | Preston Brown | 69.79 | 65.50 | 68.48 | 1020 | Bills |
| 24 | 12 | David Harris | 69.77 | 66.10 | 68.05 | 1012 | Jets |
| 25 | 13 | Jasper Brinkley | 69.74 | 67.70 | 69.44 | 461 | Vikings |
| 26 | 14 | K.J. Wright | 69.57 | 66.60 | 67.70 | 1130 | Seahawks |
| 27 | 15 | Prince Shembo | 69.55 | 66.80 | 68.25 | 340 | Falcons |
| 28 | 16 | Gerald Hodges | 69.51 | 71.90 | 71.56 | 502 | Vikings |
| 29 | 17 | Lawrence Timmons | 69.15 | 67.10 | 66.35 | 1036 | Steelers |
| 30 | 18 | Geno Hayes | 68.76 | 68.20 | 67.25 | 573 | Jaguars |
| 31 | 19 | Avery Williamson | 68.19 | 64.20 | 67.72 | 812 | Titans |
| 32 | 20 | Akeem Dent | 68.09 | 70.80 | 67.54 | 223 | Texans |
| 33 | 21 | Telvin Smith Sr. | 67.83 | 63.70 | 66.41 | 708 | Jaguars |
| 34 | 22 | Devon Kennard | 67.71 | 68.20 | 69.47 | 331 | Giants |
| 35 | 23 | Kelvin Sheppard | 67.71 | 69.10 | 67.81 | 120 | Dolphins |
| 36 | 24 | Nigel Bradham | 67.68 | 65.00 | 67.07 | 806 | Bills |
| 37 | 25 | Anthony Barr | 67.65 | 70.40 | 65.81 | 776 | Vikings |
| 38 | 26 | James Laurinaitis | 67.33 | 61.00 | 67.39 | 1042 | Rams |
| 39 | 27 | Todd Davis | 67.28 | 70.40 | 69.36 | 177 | Broncos |
| 40 | 28 | Christian Kirksey | 67.27 | 61.80 | 66.75 | 681 | Browns |
| 41 | 29 | Demario Davis | 67.08 | 62.20 | 66.36 | 1007 | Jets |
| 42 | 30 | Tahir Whitehead | 66.60 | 61.40 | 65.90 | 759 | Lions |
| 43 | 31 | Jason Trusnik | 66.50 | 65.00 | 66.99 | 396 | Dolphins |
| 44 | 32 | Philip Wheeler | 65.82 | 62.60 | 64.31 | 379 | Dolphins |
| 45 | 33 | Vincent Rey | 65.54 | 59.00 | 66.56 | 1005 | Bengals |
| 46 | 34 | DeMeco Ryans | 65.38 | 63.00 | 66.96 | 513 | Eagles |
| 47 | 35 | Danny Lansanah | 64.93 | 64.40 | 67.63 | 630 | Buccaneers |
| 48 | 36 | Jelani Jenkins | 64.92 | 61.30 | 65.38 | 901 | Dolphins |
| 49 | 37 | Manti Te'o | 64.89 | 64.00 | 65.61 | 457 | Chargers |
| 50 | 38 | Rey Maualuga | 64.85 | 62.60 | 64.37 | 458 | Bengals |
| 51 | 39 | Vontaze Burfict | 64.72 | 61.60 | 68.36 | 217 | Bengals |
| 52 | 40 | Stephen Tulloch | 64.70 | 68.50 | 64.76 | 138 | Lions |
| 53 | 41 | Jerod Mayo | 64.51 | 65.70 | 67.88 | 334 | Patriots |
| 54 | 42 | Joplo Bartu | 63.75 | 57.20 | 64.34 | 486 | Falcons |
| 55 | 43 | Josh Bynes | 63.67 | 60.50 | 65.89 | 226 | Lions |
| 56 | 44 | Paul Posluszny | 63.42 | 61.80 | 65.33 | 488 | Jaguars |
| 57 | 45 | Christian Jones | 63.31 | 61.00 | 65.88 | 434 | Bears |
| 58 | 46 | Ashlee Palmer | 63.19 | 61.20 | 60.77 | 193 | Lions |
| 59 | 47 | A.J. Klein | 63.19 | 59.40 | 65.45 | 282 | Panthers |
| 60 | 48 | Nate Irving | 62.87 | 61.00 | 65.58 | 347 | Broncos |
| 61 | 49 | Kyle Wilber | 62.83 | 61.70 | 64.10 | 241 | Cowboys |
| 62 | 50 | Michael Wilhoite | 62.81 | 61.70 | 65.43 | 1015 | 49ers |
| 63 | 51 | Lance Briggs | 62.70 | 63.00 | 64.69 | 453 | Bears |
| 64 | 52 | Kevin Minter | 62.26 | 61.20 | 64.66 | 336 | Cardinals |
| 65 | 53 | Emmanuel Acho | 62.17 | 59.10 | 63.19 | 265 | Eagles |

### Rotation/backup (61 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 66 | 1 | Mark Herzlich | 61.46 | 55.00 | 65.76 | 313 | Giants |
| 67 | 2 | Keenan Robinson | 61.37 | 56.90 | 62.14 | 812 | Commanders |
| 68 | 3 | Jon Bostic | 61.31 | 56.80 | 64.05 | 679 | Bears |
| 69 | 4 | Zaviar Gooden | 60.82 | 60.50 | 64.94 | 157 | Titans |
| 70 | 5 | Justin Durant | 60.51 | 59.00 | 64.43 | 324 | Cowboys |
| 71 | 6 | Wesley Woodyard | 60.23 | 54.60 | 59.81 | 879 | Titans |
| 72 | 7 | Shea McClellin | 60.07 | 53.00 | 64.79 | 424 | Bears |
| 73 | 8 | D'Qwell Jackson | 60.04 | 52.30 | 61.03 | 1196 | Colts |
| 74 | 9 | Mason Foster | 59.73 | 53.70 | 63.01 | 556 | Buccaneers |
| 75 | 10 | Sam Barrington | 59.21 | 57.00 | 63.69 | 474 | Packers |
| 76 | 11 | Perry Riley | 59.14 | 52.30 | 60.57 | 888 | Commanders |
| 77 | 12 | A.J. Hawk | 58.98 | 50.10 | 60.73 | 875 | Packers |
| 78 | 13 | Jacquian Williams | 58.85 | 54.30 | 62.61 | 563 | Giants |
| 79 | 14 | Will Compton | 58.80 | 52.60 | 61.37 | 359 | Commanders |
| 80 | 15 | Curtis Lofton | 58.73 | 50.10 | 60.32 | 1044 | Saints |
| 81 | 16 | Alec Ogletree | 58.66 | 52.70 | 58.47 | 1045 | Rams |
| 82 | 17 | Kavell Conner | 58.64 | 52.80 | 62.01 | 348 | Chargers |
| 83 | 18 | Brian Cushing | 58.58 | 58.10 | 60.89 | 725 | Texans |
| 84 | 19 | Jonathan Casillas | 58.56 | 54.60 | 61.62 | 273 | Patriots |
| 85 | 20 | Jamari Lattimore | 58.34 | 60.00 | 62.44 | 281 | Packers |
| 86 | 21 | Josh Mauga | 57.95 | 54.00 | 60.27 | 1005 | Chiefs |
| 87 | 22 | Andrew Gachkar | 57.53 | 57.60 | 59.99 | 384 | Chargers |
| 88 | 23 | Jameel McClain | 57.50 | 51.20 | 60.03 | 972 | Giants |
| 89 | 24 | David Hawthorne | 57.43 | 53.50 | 59.02 | 742 | Saints |
| 90 | 25 | Anthony Hitchens | 57.42 | 44.20 | 62.07 | 607 | Cowboys |
| 91 | 26 | Joe Mays | 56.79 | 53.90 | 63.40 | 117 | Chiefs |
| 92 | 27 | Sean Spence | 56.75 | 49.20 | 57.61 | 521 | Steelers |
| 93 | 28 | Sio Moore | 56.51 | 51.70 | 59.20 | 697 | Raiders |
| 94 | 29 | Ray-Ray Armstrong | 56.29 | 56.80 | 61.03 | 238 | Raiders |
| 95 | 30 | Casey Matthews | 56.03 | 49.40 | 58.26 | 432 | Eagles |
| 96 | 31 | Emmanuel Lamur | 55.91 | 50.70 | 58.60 | 953 | Bengals |
| 97 | 32 | Paul Worrilow | 55.84 | 47.00 | 58.35 | 1079 | Falcons |
| 98 | 33 | D.J. Williams | 55.75 | 50.00 | 62.28 | 413 | Bears |
| 99 | 34 | Corey Nelson | 55.70 | 54.60 | 65.68 | 108 | Broncos |
| 100 | 35 | Josh McNary | 54.97 | 52.00 | 60.20 | 260 | Colts |
| 101 | 36 | Ryan Shazier | 54.83 | 52.20 | 59.72 | 281 | Steelers |
| 102 | 37 | Keith Rivers | 54.77 | 50.20 | 58.34 | 185 | Bills |
| 103 | 38 | Chad Greenway | 53.76 | 45.90 | 56.91 | 760 | Vikings |
| 104 | 39 | Jackson Jeffcoat | 53.19 | 57.80 | 65.95 | 117 | Commanders |
| 105 | 40 | James-Michael Johnson | 52.95 | 45.20 | 61.04 | 437 | Chiefs |
| 106 | 41 | Bruce Carter | 52.85 | 43.90 | 56.54 | 644 | Cowboys |
| 107 | 42 | Steven Johnson | 52.14 | 49.00 | 57.77 | 244 | Broncos |
| 108 | 43 | Orie Lemon | 52.08 | 51.70 | 60.47 | 190 | Buccaneers |
| 109 | 44 | J.T. Thomas | 52.08 | 45.90 | 57.96 | 713 | Jaguars |
| 110 | 45 | LaRoy Reynolds | 51.90 | 48.90 | 61.07 | 116 | Jaguars |
| 111 | 46 | Dekoda Watson | 51.89 | 47.70 | 56.76 | 100 | Cowboys |
| 112 | 47 | Darryl Sharpton | 51.87 | 43.70 | 61.90 | 104 | Bears |
| 113 | 48 | Khaseem Greene | 51.81 | 46.90 | 60.82 | 117 | Bears |
| 114 | 49 | Donald Butler | 51.81 | 40.50 | 57.36 | 704 | Chargers |
| 115 | 50 | Jon Beason | 51.51 | 49.00 | 58.38 | 159 | Giants |
| 116 | 51 | JoLonn Dunbar | 50.39 | 38.60 | 55.85 | 422 | Rams |
| 117 | 52 | Ramon Humber | 50.16 | 38.30 | 55.88 | 445 | Saints |
| 118 | 53 | Deontae Skinner | 49.88 | 49.60 | 59.32 | 103 | Patriots |
| 119 | 54 | Larry Foote | 49.82 | 39.80 | 57.02 | 1074 | Cardinals |
| 120 | 55 | Malcolm Smith | 49.32 | 38.10 | 57.22 | 274 | Seahawks |
| 121 | 56 | Dane Fletcher | 48.96 | 38.20 | 57.06 | 351 | Buccaneers |
| 122 | 57 | Brad Jones | 48.38 | 34.90 | 56.85 | 216 | Packers |
| 123 | 58 | Nick Moody | 47.85 | 52.00 | 59.41 | 164 | 49ers |
| 124 | 59 | Reggie Walker | 46.76 | 35.60 | 55.24 | 143 | Chargers |
| 125 | 60 | Justin Tuggle | 46.47 | 36.80 | 56.17 | 269 | Texans |
| 126 | 61 | Miles Burris | 45.59 | 29.30 | 55.41 | 1060 | Raiders |

## QB — Quarterback

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Rodgers | 86.78 | 90.30 | 81.21 | 693 | Packers |
| 2 | 2 | Drew Brees | 83.73 | 87.53 | 76.08 | 740 | Saints |
| 3 | 3 | Ben Roethlisberger | 83.48 | 85.86 | 77.34 | 741 | Steelers |
| 4 | 4 | Tony Romo | 82.40 | 81.00 | 80.64 | 569 | Cowboys |
| 5 | 5 | Peyton Manning | 82.09 | 82.18 | 77.48 | 708 | Broncos |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Matt Ryan | 79.69 | 81.86 | 73.37 | 720 | Falcons |
| 7 | 2 | Philip Rivers | 79.42 | 79.75 | 74.65 | 670 | Chargers |
| 8 | 3 | Tom Brady | 79.41 | 83.84 | 71.25 | 802 | Patriots |
| 9 | 4 | Russell Wilson | 78.09 | 76.41 | 75.26 | 661 | Seahawks |
| 10 | 5 | Andrew Luck | 75.84 | 75.07 | 71.93 | 855 | Colts |
| 11 | 6 | Ryan Tannehill | 74.49 | 76.80 | 68.28 | 684 | Dolphins |
| 12 | 7 | Joe Flacco | 74.19 | 73.60 | 70.13 | 709 | Ravens |

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Eli Manning | 72.34 | 70.73 | 69.33 | 670 | Giants |
| 14 | 2 | Matthew Stafford | 72.04 | 72.10 | 67.45 | 751 | Lions |
| 15 | 3 | Cam Newton | 71.97 | 73.37 | 66.88 | 632 | Panthers |
| 16 | 4 | Alex Smith | 71.75 | 73.13 | 68.32 | 577 | Chiefs |
| 17 | 5 | Carson Palmer | 70.58 | 70.26 | 73.53 | 255 | Cardinals |
| 18 | 6 | Ryan Fitzpatrick | 70.45 | 70.51 | 72.28 | 391 | Texans |
| 19 | 7 | Colin Kaepernick | 70.18 | 67.75 | 69.06 | 624 | 49ers |
| 20 | 8 | Andy Dalton | 69.45 | 68.41 | 66.67 | 599 | Bengals |
| 21 | 9 | Jay Cutler | 68.67 | 66.90 | 67.55 | 665 | Bears |
| 22 | 10 | Nick Foles | 66.93 | 69.91 | 67.88 | 355 | Eagles |
| 23 | 11 | Teddy Bridgewater | 65.69 | 74.80 | 66.99 | 501 | Vikings |
| 24 | 12 | Derek Anderson | 63.99 | 74.80 | 73.08 | 107 | Panthers |
| 25 | 13 | Robert Griffin III | 63.94 | 61.91 | 68.68 | 281 | Commanders |
| 26 | 14 | Mike Glennon | 62.44 | 66.88 | 65.01 | 246 | Buccaneers |

### Rotation/backup (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | Colt McCoy | 61.94 | 62.12 | 75.22 | 160 | Commanders |
| 28 | 2 | Charlie Whitehurst | 61.84 | 68.73 | 69.32 | 228 | Titans |
| 29 | 3 | Geno Smith | 61.34 | 60.46 | 62.19 | 448 | Jets |
| 30 | 4 | Brian Hoyer | 61.32 | 60.85 | 66.23 | 499 | Browns |
| 31 | 5 | Mark Sanchez | 61.09 | 58.46 | 68.58 | 356 | Eagles |
| 32 | 6 | Kirk Cousins | 60.76 | 59.50 | 71.33 | 228 | Commanders |
| 33 | 7 | Kyle Orton | 60.59 | 57.77 | 67.32 | 509 | Bills |
| 34 | 8 | Shaun Hill | 60.07 | 61.23 | 67.65 | 272 | Rams |
| 35 | 9 | Austin Davis | 59.69 | 59.93 | 65.26 | 337 | Rams |
| 36 | 10 | Zach Mettenberger | 59.54 | 56.50 | 68.71 | 210 | Titans |
| 37 | 11 | Derek Carr | 58.79 | 54.40 | 56.92 | 666 | Raiders |
| 38 | 12 | Josh McCown | 58.66 | 58.73 | 61.80 | 405 | Buccaneers |
| 39 | 13 | E.J. Manuel | 57.49 | 58.22 | 60.79 | 153 | Bills |
| 40 | 14 | Drew Stanton | 57.04 | 50.60 | 63.89 | 292 | Cardinals |
| 41 | 15 | Blake Bortles | 55.73 | 45.40 | 56.41 | 587 | Jaguars |
| 42 | 16 | Chad Henne | 55.20 | 56.31 | 60.88 | 101 | Jaguars |
| 43 | 17 | Jake Locker | 54.89 | 56.83 | 59.08 | 186 | Titans |
| 44 | 18 | Ryan Lindley | 52.22 | 37.81 | 55.80 | 136 | Cardinals |
| 45 | 19 | Michael Vick | 50.20 | 47.23 | 56.95 | 158 | Jets |

## S — Safety

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Eric Weddle | 93.49 | 90.00 | 91.65 | 1020 | Chargers |
| 2 | 2 | Devin McCourty | 92.58 | 90.20 | 90.00 | 1176 | Patriots |
| 3 | 3 | Glover Quin | 91.06 | 90.40 | 87.34 | 1100 | Lions |
| 4 | 4 | Antoine Bethea | 90.10 | 89.60 | 86.27 | 1040 | 49ers |
| 5 | 5 | Earl Thomas III | 89.91 | 89.10 | 86.28 | 1166 | Seahawks |
| 6 | 6 | Mike Adams | 86.15 | 85.20 | 82.61 | 1222 | Colts |
| 7 | 7 | Reggie Nelson | 84.59 | 82.90 | 81.75 | 1170 | Bengals |
| 8 | 8 | Jim Leonhard | 83.43 | 82.60 | 80.13 | 505 | Browns |
| 9 | 9 | Will Hill III | 82.18 | 83.80 | 81.71 | 703 | Ravens |
| 10 | 10 | Husain Abdullah | 81.30 | 80.50 | 79.14 | 1025 | Chiefs |
| 11 | 11 | Dawan Landry | 81.13 | 75.60 | 80.65 | 945 | Jets |
| 12 | 12 | Harrison Smith | 81.04 | 74.80 | 83.53 | 1070 | Vikings |
| 13 | 13 | Jaiquawn Jarrett | 80.80 | 83.00 | 76.20 | 383 | Jets |
| 14 | 14 | Kam Chancellor | 80.11 | 80.30 | 75.81 | 1044 | Seahawks |

### Good (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Tashaun Gipson Sr. | 79.73 | 79.90 | 79.30 | 772 | Browns |
| 16 | 2 | David Bruton | 79.67 | 78.70 | 81.25 | 228 | Broncos |
| 17 | 3 | Ryan Mundy | 79.34 | 75.70 | 79.37 | 947 | Bears |
| 18 | 4 | James Ihedigbo | 79.16 | 80.30 | 75.49 | 885 | Lions |
| 19 | 5 | Duron Harmon | 79.07 | 72.40 | 79.74 | 317 | Patriots |
| 20 | 6 | Malcolm Jenkins | 78.85 | 75.70 | 77.41 | 1158 | Eagles |
| 21 | 7 | Kurt Coleman | 78.05 | 75.40 | 79.71 | 391 | Chiefs |
| 22 | 8 | Donte Whitner | 75.95 | 70.50 | 75.41 | 1152 | Browns |
| 23 | 9 | Jeromy Miles | 75.57 | 77.00 | 76.18 | 355 | Ravens |
| 24 | 10 | Patrick Chung | 75.43 | 70.90 | 75.64 | 976 | Patriots |
| 25 | 11 | Nate Allen | 75.30 | 72.50 | 73.94 | 1081 | Eagles |
| 26 | 12 | Tyrann Mathieu | 74.86 | 73.30 | 73.03 | 448 | Cardinals |
| 27 | 13 | Da'Norris Searcy | 74.64 | 66.10 | 77.32 | 648 | Bills |
| 28 | 14 | Reshad Jones | 74.52 | 64.80 | 78.91 | 757 | Dolphins |
| 29 | 15 | Tavon Wilson | 74.50 | 71.70 | 75.64 | 211 | Patriots |
| 30 | 16 | Rahim Moore | 74.12 | 74.00 | 71.91 | 1126 | Broncos |

### Starter (45 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Ha Ha Clinton-Dix | 73.99 | 69.20 | 73.02 | 1069 | Packers |
| 32 | 2 | Bradley McDougald | 73.10 | 71.10 | 71.57 | 445 | Buccaneers |
| 33 | 3 | Tre Boston | 72.24 | 67.20 | 76.63 | 458 | Panthers |
| 34 | 4 | Danieal Manning | 72.01 | 66.90 | 74.38 | 579 | Texans |
| 35 | 5 | Darian Stewart | 71.96 | 70.10 | 72.26 | 873 | Ravens |
| 36 | 6 | Bernard Pollard | 71.74 | 70.40 | 74.20 | 344 | Titans |
| 37 | 7 | Rodney McLeod | 71.69 | 71.20 | 70.77 | 1025 | Rams |
| 38 | 8 | Isa Abdul-Quddus | 71.48 | 68.30 | 72.66 | 303 | Lions |
| 39 | 9 | George Iloka | 71.46 | 64.20 | 72.14 | 1184 | Bengals |
| 40 | 10 | Troy Polamalu | 71.13 | 69.50 | 71.48 | 754 | Steelers |
| 41 | 11 | Ron Parker | 69.77 | 68.90 | 66.19 | 1013 | Chiefs |
| 42 | 12 | Sergio Brown | 68.82 | 68.80 | 69.99 | 532 | Colts |
| 43 | 13 | Micah Hyde | 68.81 | 64.40 | 67.58 | 796 | Packers |
| 44 | 14 | Chris Conte | 68.71 | 64.10 | 69.90 | 463 | Bears |
| 45 | 15 | Charles Woodson | 68.51 | 63.30 | 67.82 | 1100 | Raiders |
| 46 | 16 | George Wilson | 68.33 | 64.60 | 66.65 | 812 | Titans |
| 47 | 17 | Brock Vereen | 68.25 | 65.30 | 69.19 | 502 | Bears |
| 48 | 18 | Eric Reid | 68.21 | 61.40 | 69.24 | 879 | 49ers |
| 49 | 19 | Andrew Sendejo | 68.13 | 64.50 | 73.05 | 141 | Vikings |
| 50 | 20 | Duke Williams | 68.09 | 66.50 | 68.88 | 528 | Bills |
| 51 | 21 | Marcus Gilchrist | 67.95 | 58.50 | 70.09 | 987 | Chargers |
| 52 | 22 | Morgan Burnett | 67.41 | 55.00 | 72.15 | 1070 | Packers |
| 53 | 23 | Trenton Robinson | 67.19 | 65.40 | 70.67 | 101 | Commanders |
| 54 | 24 | Jordan Poyer | 67.09 | 65.60 | 68.47 | 116 | Browns |
| 55 | 25 | Barry Church | 67.08 | 62.70 | 68.53 | 997 | Cowboys |
| 56 | 26 | Dwight Lowery | 67.04 | 62.00 | 69.05 | 1029 | Falcons |
| 57 | 27 | Aaron Williams | 67.03 | 62.90 | 66.77 | 901 | Bills |
| 58 | 28 | Robert Blanton | 66.84 | 60.00 | 72.01 | 948 | Vikings |
| 59 | 29 | Darrell Stuckey | 66.21 | 62.90 | 67.58 | 173 | Chargers |
| 60 | 30 | Kendrick Lewis | 66.13 | 59.90 | 67.59 | 1077 | Texans |
| 61 | 31 | Louis Delmas | 66.06 | 63.70 | 66.70 | 834 | Dolphins |
| 62 | 32 | Jamarca Sanford | 65.97 | 66.50 | 68.64 | 109 | Saints |
| 63 | 33 | Calvin Pryor | 65.90 | 63.30 | 64.50 | 680 | Jets |
| 64 | 34 | Johnathan Cyprien | 65.76 | 61.40 | 65.53 | 985 | Jaguars |
| 65 | 35 | Sherrod Martin | 65.65 | 66.90 | 66.48 | 135 | Jaguars |
| 66 | 36 | Jahleel Addae | 65.40 | 61.60 | 67.01 | 427 | Chargers |
| 67 | 37 | Jimmy Wilson | 65.21 | 64.10 | 62.82 | 786 | Dolphins |
| 68 | 38 | Don Carey | 65.09 | 63.90 | 68.06 | 133 | Lions |
| 69 | 39 | LaRon Landry | 64.46 | 54.80 | 68.40 | 604 | Colts |
| 70 | 40 | Jeff Heath | 64.43 | 60.70 | 65.62 | 156 | Cowboys |
| 71 | 41 | Thomas DeCoud | 64.19 | 61.90 | 63.95 | 665 | Panthers |
| 72 | 42 | Usama Young | 63.27 | 60.10 | 69.86 | 216 | Raiders |
| 73 | 43 | D.J. Swearinger Sr. | 62.86 | 58.90 | 61.34 | 1018 | Texans |
| 74 | 44 | T.J. McDonald | 62.28 | 54.70 | 65.52 | 1049 | Rams |
| 75 | 45 | Quintin Demps | 62.10 | 57.20 | 62.55 | 629 | Giants |

### Rotation/backup (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 76 | 1 | T.J. Ward | 61.98 | 55.40 | 62.62 | 1075 | Broncos |
| 77 | 2 | Kenny Vaccaro | 60.88 | 58.90 | 59.47 | 980 | Saints |
| 78 | 3 | Major Wright | 60.70 | 53.60 | 64.18 | 505 | Buccaneers |
| 79 | 4 | Kemal Ishmael | 60.39 | 50.50 | 68.68 | 805 | Falcons |
| 80 | 5 | Craig Dahl | 60.25 | 58.90 | 61.89 | 182 | 49ers |
| 81 | 6 | Antrel Rolle | 60.25 | 56.90 | 58.32 | 1048 | Giants |
| 82 | 7 | Rashad Johnson | 60.13 | 55.20 | 61.65 | 1131 | Cardinals |
| 83 | 8 | Eric Berry | 59.60 | 54.40 | 64.10 | 361 | Chiefs |
| 84 | 9 | Tyvon Branch | 59.19 | 58.60 | 66.98 | 190 | Raiders |
| 85 | 10 | Stevie Brown | 59.11 | 47.40 | 64.84 | 576 | Giants |
| 86 | 11 | Rafael Bush | 58.62 | 49.40 | 66.12 | 468 | Saints |
| 87 | 12 | Jairus Byrd | 58.42 | 58.80 | 61.81 | 267 | Saints |
| 88 | 13 | Mike Mitchell | 57.83 | 46.40 | 61.49 | 1010 | Steelers |
| 89 | 14 | Roman Harper | 57.09 | 48.70 | 60.08 | 1024 | Panthers |
| 90 | 15 | J.J. Wilcox | 56.96 | 52.80 | 57.14 | 1109 | Cowboys |
| 91 | 16 | Ryan Clark | 56.65 | 50.30 | 56.92 | 1013 | Commanders |
| 92 | 17 | Michael Griffin | 56.06 | 46.40 | 58.96 | 1132 | Titans |
| 93 | 18 | Sean Richardson | 56.01 | 51.70 | 61.70 | 133 | Packers |
| 94 | 19 | Will Allen | 55.11 | 49.80 | 57.40 | 324 | Steelers |
| 95 | 20 | Quinton Carter | 52.97 | 51.40 | 58.09 | 216 | Broncos |
| 96 | 21 | Tony Jefferson | 52.75 | 42.00 | 59.27 | 727 | Cardinals |
| 97 | 22 | Josh Evans | 52.13 | 43.80 | 53.90 | 971 | Jaguars |
| 98 | 23 | Brandon Meriweather | 52.11 | 43.30 | 61.00 | 597 | Commanders |
| 99 | 24 | Daimion Stafford | 51.64 | 48.80 | 57.70 | 277 | Titans |
| 100 | 25 | Larry Asante | 51.51 | 56.10 | 58.08 | 161 | Raiders |
| 101 | 26 | Terrence Brooks | 49.62 | 47.30 | 57.87 | 234 | Ravens |
| 102 | 27 | Danny McCray | 49.46 | 40.50 | 60.75 | 168 | Bears |
| 103 | 28 | William Moore | 49.34 | 40.50 | 56.16 | 322 | Falcons |
| 104 | 29 | Matt Elam | 47.18 | 33.90 | 51.87 | 680 | Ravens |
| 105 | 30 | Dashon Goldson | 45.81 | 29.90 | 54.24 | 780 | Buccaneers |
| 106 | 31 | Bacarri Rambo | 45.00 | 30.50 | 53.67 | 128 | Bills |
| 107 | 32 | Charles Godfrey | 45.00 | 34.50 | 54.35 | 205 | Falcons |

## T — Tackle

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (39 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Andrew Whitworth | 97.13 | 92.30 | 96.19 | 1091 | Bengals |
| 2 | 2 | Jason Peters | 96.41 | 92.40 | 94.92 | 1146 | Eagles |
| 3 | 3 | Joe Thomas | 95.96 | 91.00 | 95.10 | 1052 | Browns |
| 4 | 4 | Joe Staley | 94.49 | 89.60 | 93.58 | 1056 | 49ers |
| 5 | 5 | Tyron Smith | 92.11 | 87.20 | 91.22 | 1178 | Cowboys |
| 6 | 6 | Donald Penn | 90.63 | 88.60 | 87.81 | 1021 | Raiders |
| 7 | 7 | Lane Johnson | 90.55 | 83.50 | 91.09 | 889 | Eagles |
| 8 | 8 | Branden Albert | 90.17 | 84.60 | 89.71 | 543 | Dolphins |
| 9 | 9 | Taylor Lewan | 89.25 | 81.00 | 90.59 | 353 | Titans |
| 10 | 10 | Trent Williams | 89.14 | 82.30 | 89.53 | 877 | Commanders |
| 11 | 11 | Jared Veldheer | 88.75 | 83.20 | 88.28 | 1104 | Cardinals |
| 12 | 12 | Anthony Castonzo | 88.30 | 84.40 | 86.73 | 1360 | Colts |
| 13 | 13 | Kelvin Beachum | 87.79 | 82.70 | 87.02 | 1185 | Steelers |
| 14 | 14 | Michael Roos | 87.78 | 80.30 | 88.60 | 289 | Titans |
| 15 | 15 | King Dunlap | 87.45 | 82.60 | 86.52 | 1058 | Chargers |
| 16 | 16 | Duane Brown | 87.05 | 80.30 | 87.39 | 1100 | Texans |
| 17 | 17 | Rick Wagner | 86.77 | 80.20 | 86.98 | 969 | Ravens |
| 18 | 18 | Derek Newton | 86.58 | 78.80 | 87.60 | 1108 | Texans |
| 19 | 19 | Jermey Parnell | 86.35 | 79.30 | 86.89 | 498 | Cowboys |
| 20 | 20 | Will Beatty | 86.30 | 80.00 | 86.34 | 1114 | Giants |
| 21 | 21 | Cordy Glenn | 86.23 | 79.90 | 86.28 | 1045 | Bills |
| 22 | 22 | Doug Free | 85.99 | 79.40 | 86.21 | 700 | Cowboys |
| 23 | 23 | Terron Armstead | 85.46 | 78.60 | 85.87 | 836 | Saints |
| 24 | 24 | Riley Reiff | 84.29 | 78.50 | 83.98 | 1008 | Lions |
| 25 | 25 | Ryan Schraeder | 83.53 | 74.90 | 85.12 | 646 | Falcons |
| 26 | 26 | Zach Strief | 83.47 | 76.40 | 84.02 | 1058 | Saints |
| 27 | 27 | Sebastian Vollmer | 83.45 | 76.10 | 84.19 | 1223 | Patriots |
| 28 | 28 | Marcus Gilbert | 83.37 | 74.80 | 84.91 | 820 | Steelers |
| 29 | 29 | Phil Loadholt | 83.08 | 75.40 | 84.04 | 715 | Vikings |
| 30 | 30 | Bryan Bulaga | 83.05 | 74.40 | 84.65 | 1064 | Packers |
| 31 | 31 | Nate Solder | 82.59 | 74.10 | 84.09 | 1256 | Patriots |
| 32 | 32 | Joe Barksdale | 82.56 | 73.00 | 84.76 | 996 | Rams |
| 33 | 33 | Demar Dotson | 82.37 | 73.50 | 84.11 | 979 | Buccaneers |
| 34 | 34 | D'Brickashaw Ferguson | 82.04 | 75.30 | 82.37 | 1088 | Jets |
| 35 | 35 | Ryan Clady | 81.94 | 74.10 | 83.00 | 1124 | Broncos |
| 36 | 36 | Mitchell Schwartz | 81.23 | 72.70 | 82.75 | 1052 | Browns |
| 37 | 37 | Russell Okung | 81.12 | 71.40 | 83.43 | 1050 | Seahawks |
| 38 | 38 | Anthony Davis | 80.40 | 68.10 | 84.44 | 429 | 49ers |
| 39 | 39 | Bobby Massie | 80.13 | 70.90 | 82.11 | 1104 | Cardinals |

### Good (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 40 | 1 | Breno Giacomini | 79.93 | 70.30 | 82.19 | 1088 | Jets |
| 41 | 2 | LaAdrian Waddle | 79.75 | 70.10 | 82.02 | 550 | Lions |
| 42 | 3 | Ryan Harris | 79.74 | 70.20 | 81.94 | 956 | Chiefs |
| 43 | 4 | David Bakhtiari | 79.73 | 72.30 | 80.52 | 1144 | Packers |
| 44 | 5 | Jake Long | 79.08 | 71.80 | 79.76 | 434 | Rams |
| 45 | 6 | Eric Fisher | 78.49 | 67.30 | 81.78 | 1005 | Chiefs |
| 46 | 7 | Byron Stingily | 78.27 | 66.50 | 81.95 | 243 | Titans |
| 47 | 8 | Chris Clark | 78.23 | 66.50 | 81.88 | 486 | Broncos |
| 48 | 9 | Matt Kalil | 77.03 | 65.70 | 80.41 | 1025 | Vikings |
| 49 | 10 | Anthony Collins | 77.03 | 65.30 | 80.68 | 621 | Buccaneers |
| 50 | 11 | Jordan Mills | 76.58 | 62.70 | 81.67 | 814 | Bears |
| 51 | 12 | Eugene Monroe | 76.57 | 65.80 | 79.58 | 728 | Ravens |
| 52 | 13 | Michael Ola | 76.06 | 64.60 | 79.53 | 823 | Bears |
| 53 | 14 | Josh Wells | 75.49 | 69.10 | 75.59 | 102 | Jaguars |
| 54 | 15 | Mike Remmers | 75.29 | 68.40 | 75.72 | 506 | Panthers |
| 55 | 16 | Greg Robinson | 74.98 | 61.30 | 79.94 | 724 | Rams |
| 56 | 17 | Michael Oher | 74.51 | 64.10 | 77.28 | 651 | Titans |
| 57 | 18 | Tom Compton | 74.49 | 63.40 | 77.72 | 650 | Commanders |
| 58 | 19 | Ja'Wuan James | 74.47 | 63.50 | 77.61 | 1038 | Dolphins |
| 59 | 20 | Eric Winston | 74.29 | 61.60 | 78.58 | 235 | Bengals |
| 60 | 21 | Austin Pasztor | 74.03 | 62.00 | 77.88 | 492 | Jaguars |

### Starter (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 61 | 1 | Cam Fleming | 73.86 | 62.50 | 77.26 | 244 | Patriots |
| 62 | 2 | Gosder Cherilus | 73.66 | 61.50 | 77.60 | 950 | Colts |
| 63 | 3 | Jonathan Martin | 73.15 | 62.00 | 76.41 | 646 | 49ers |
| 64 | 4 | Jason Fox | 72.93 | 59.20 | 77.91 | 214 | Dolphins |
| 65 | 5 | Cornelius Lucas | 72.75 | 61.20 | 76.28 | 515 | Lions |
| 66 | 6 | Paul Cornick | 72.71 | 63.10 | 74.95 | 300 | Broncos |
| 67 | 7 | Menelik Watson | 72.56 | 58.40 | 77.84 | 487 | Raiders |
| 68 | 8 | Tyler Polumbus | 71.93 | 60.10 | 75.65 | 471 | Commanders |
| 69 | 9 | Khalif Barnes | 71.88 | 57.70 | 77.16 | 766 | Raiders |
| 70 | 10 | Lamar Holmes | 71.70 | 59.60 | 75.60 | 226 | Falcons |
| 71 | 11 | Jake Matthews | 71.70 | 59.80 | 75.46 | 944 | Falcons |
| 72 | 12 | Byron Bell | 71.62 | 59.80 | 75.33 | 1157 | Panthers |
| 73 | 13 | Marshall Newhouse | 71.29 | 58.40 | 75.72 | 381 | Bengals |
| 74 | 14 | Nate Chandler | 71.19 | 58.60 | 75.42 | 691 | Panthers |
| 75 | 15 | Marcus Cannon | 71.01 | 56.50 | 76.51 | 446 | Patriots |
| 76 | 16 | Sam Young | 70.36 | 57.40 | 74.84 | 396 | Jaguars |
| 77 | 17 | Seantrel Henderson | 69.89 | 57.30 | 74.11 | 1061 | Bills |
| 78 | 18 | Bryce Harris | 69.41 | 56.80 | 73.65 | 387 | Saints |
| 79 | 19 | Jeff Linkenbach | 69.08 | 54.50 | 74.63 | 220 | Chiefs |
| 80 | 20 | Mike Adams | 69.01 | 56.50 | 73.19 | 381 | Steelers |
| 81 | 21 | Jamon Meredith | 68.75 | 53.10 | 75.02 | 210 | Titans |
| 82 | 22 | Will Svitek | 68.52 | 56.20 | 72.56 | 188 | Titans |
| 83 | 23 | James Hurst | 65.95 | 50.30 | 72.21 | 514 | Ravens |
| 84 | 24 | Morgan Moses | 64.91 | 54.40 | 67.75 | 126 | Commanders |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 85 | 1 | David Foucault | 60.30 | 40.50 | 69.34 | 127 | Panthers |
| 86 | 2 | Matt McCants | 58.08 | 43.90 | 63.36 | 112 | Raiders |

## TE — Tight End

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 88.05 | 91.30 | 81.71 | 679 | Patriots |
| 2 | 2 | Travis Kelce | 83.68 | 84.70 | 78.84 | 426 | Chiefs |
| 3 | 3 | Ladarius Green | 83.19 | 78.60 | 82.09 | 146 | Chargers |
| 4 | 4 | Jason Witten | 82.03 | 83.60 | 76.82 | 603 | Cowboys |
| 5 | 5 | Greg Olsen | 81.18 | 84.20 | 75.00 | 644 | Panthers |

### Good (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Daniel Fells | 78.08 | 78.90 | 73.36 | 145 | Giants |
| 7 | 2 | Julius Thomas | 77.73 | 74.70 | 75.59 | 478 | Broncos |
| 8 | 3 | Zach Ertz | 77.65 | 80.90 | 71.32 | 440 | Eagles |
| 9 | 4 | Jimmy Graham | 77.44 | 74.70 | 75.10 | 574 | Saints |
| 10 | 5 | Jared Cook | 77.15 | 72.50 | 76.08 | 435 | Rams |
| 11 | 6 | Brent Celek | 77.13 | 76.70 | 73.25 | 377 | Eagles |
| 12 | 7 | Dwayne Allen | 76.84 | 73.60 | 74.83 | 406 | Colts |
| 13 | 8 | Josh Hill | 76.43 | 72.60 | 74.81 | 106 | Saints |
| 14 | 9 | Delanie Walker | 75.99 | 74.10 | 73.09 | 524 | Titans |
| 15 | 10 | Chase Ford | 75.84 | 72.10 | 74.17 | 239 | Vikings |
| 16 | 11 | Martellus Bennett | 75.74 | 79.50 | 69.07 | 647 | Bears |
| 17 | 12 | Charles Clay | 74.97 | 77.10 | 69.39 | 458 | Dolphins |
| 18 | 13 | Antonio Gates | 74.47 | 72.40 | 71.68 | 576 | Chargers |
| 19 | 14 | Luke Willson | 74.46 | 69.50 | 73.60 | 368 | Seahawks |
| 20 | 15 | Owen Daniels | 74.21 | 73.80 | 70.31 | 562 | Ravens |
| 21 | 16 | Cooper Helfet | 74.00 | 71.90 | 71.24 | 147 | Seahawks |

### Starter (45 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Jim Dray | 73.35 | 67.90 | 72.81 | 241 | Browns |
| 23 | 2 | Larry Donnell | 73.15 | 66.60 | 73.35 | 540 | Giants |
| 24 | 3 | Heath Miller | 73.14 | 72.60 | 69.33 | 721 | Steelers |
| 25 | 4 | Niles Paul | 73.10 | 64.80 | 74.46 | 310 | Commanders |
| 26 | 5 | Jermaine Gresham | 72.89 | 68.40 | 71.72 | 464 | Bengals |
| 27 | 6 | Jack Doyle | 72.88 | 73.60 | 68.24 | 127 | Colts |
| 28 | 7 | Jordan Reed | 72.62 | 70.20 | 70.06 | 272 | Commanders |
| 29 | 8 | Coby Fleener | 72.04 | 72.10 | 67.83 | 621 | Colts |
| 30 | 9 | Richard Rodgers | 71.94 | 63.00 | 73.73 | 240 | Packers |
| 31 | 10 | Dennis Pitta | 71.71 | 67.90 | 70.09 | 106 | Ravens |
| 32 | 11 | Vernon Davis | 71.02 | 53.80 | 78.34 | 488 | 49ers |
| 33 | 12 | Lance Kendricks | 70.86 | 71.10 | 66.53 | 256 | Rams |
| 34 | 13 | Rhett Ellison | 70.79 | 72.20 | 65.69 | 212 | Vikings |
| 35 | 14 | Benjamin Watson | 70.65 | 63.80 | 71.05 | 293 | Saints |
| 36 | 15 | Gary Barnidge | 70.35 | 65.90 | 69.15 | 195 | Browns |
| 37 | 16 | Scott Chandler | 70.34 | 60.50 | 72.73 | 532 | Bills |
| 38 | 17 | Virgil Green | 70.32 | 71.00 | 65.70 | 145 | Broncos |
| 39 | 18 | Tony Moeaki | 70.18 | 67.60 | 67.73 | 118 | Seahawks |
| 40 | 19 | Gavin Escobar | 69.99 | 62.50 | 70.82 | 144 | Cowboys |
| 41 | 20 | Tim Wright | 69.92 | 61.20 | 71.56 | 276 | Patriots |
| 42 | 21 | Jacob Tamme | 69.78 | 56.50 | 74.46 | 200 | Broncos |
| 43 | 22 | Jordan Cameron | 69.35 | 63.50 | 69.08 | 250 | Browns |
| 44 | 23 | Marcedes Lewis | 69.26 | 57.10 | 73.20 | 281 | Jaguars |
| 45 | 24 | Kyle Rudolph | 68.77 | 63.20 | 68.32 | 261 | Vikings |
| 46 | 25 | Dion Sims | 68.33 | 67.20 | 64.91 | 268 | Dolphins |
| 47 | 26 | Clay Harbor | 67.87 | 65.30 | 65.41 | 302 | Jaguars |
| 48 | 27 | Andrew Quarless | 67.87 | 63.70 | 66.48 | 375 | Packers |
| 49 | 28 | Jace Amaro | 67.74 | 67.90 | 63.47 | 251 | Jets |
| 50 | 29 | Anthony Fasano | 67.16 | 61.10 | 67.04 | 375 | Chiefs |
| 51 | 30 | Garrett Graham | 67.14 | 57.40 | 69.47 | 324 | Texans |
| 52 | 31 | Austin Seferian-Jenkins | 67.06 | 61.60 | 66.54 | 296 | Buccaneers |
| 53 | 32 | Matt Spaeth | 66.83 | 64.40 | 64.28 | 149 | Steelers |
| 54 | 33 | Lee Smith | 66.30 | 64.60 | 63.26 | 121 | Bills |
| 55 | 34 | Rob Housler | 65.85 | 61.10 | 64.85 | 143 | Cardinals |
| 56 | 35 | Ed Dickson | 65.06 | 58.10 | 65.53 | 199 | Panthers |
| 57 | 36 | Michael Hoomanawanui | 64.65 | 51.90 | 68.99 | 246 | Patriots |
| 58 | 37 | Mychal Rivera | 64.65 | 53.10 | 68.18 | 557 | Raiders |
| 59 | 38 | Ryan Griffin | 64.45 | 52.80 | 68.05 | 177 | Texans |
| 60 | 39 | Brian Leonhardt | 63.78 | 54.90 | 65.54 | 106 | Raiders |
| 61 | 40 | Brandon Pettigrew | 63.67 | 54.80 | 65.41 | 316 | Lions |
| 62 | 41 | Logan Paulsen | 63.62 | 52.70 | 66.73 | 158 | Commanders |
| 63 | 42 | Brandon Myers | 63.24 | 55.00 | 64.56 | 268 | Buccaneers |
| 64 | 43 | Eric Ebron | 62.76 | 56.20 | 62.97 | 330 | Lions |
| 65 | 44 | Jeff Cumberland | 62.35 | 48.90 | 67.15 | 480 | Jets |
| 66 | 45 | Dante Rosario | 62.22 | 57.60 | 61.13 | 120 | Bears |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 67 | 1 | Levine Toilolo | 61.31 | 53.80 | 62.15 | 568 | Falcons |
| 68 | 2 | C.J. Fiedorowicz | 60.79 | 50.10 | 63.75 | 185 | Texans |
| 69 | 3 | John Carlson | 60.53 | 51.10 | 62.65 | 469 | Cardinals |
| 70 | 4 | Luke Stocker | 59.85 | 56.20 | 58.12 | 158 | Buccaneers |

## WR — Wide Receiver

- **Season used:** `2014`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Odell Beckham Jr. | 89.78 | 90.80 | 84.93 | 489 | Giants |
| 2 | 2 | Julio Jones | 87.81 | 89.80 | 82.31 | 613 | Falcons |
| 3 | 3 | Antonio Brown | 87.55 | 91.10 | 81.02 | 722 | Steelers |
| 4 | 4 | Calvin Johnson | 87.25 | 89.50 | 81.58 | 537 | Lions |
| 5 | 5 | Dez Bryant | 86.43 | 90.10 | 79.82 | 582 | Cowboys |
| 6 | 6 | Demaryius Thomas | 86.41 | 87.90 | 81.25 | 674 | Broncos |
| 7 | 7 | Jordy Nelson | 85.21 | 86.90 | 79.91 | 686 | Packers |
| 8 | 8 | A.J. Green | 84.63 | 84.70 | 80.41 | 380 | Bengals |
| 9 | 9 | DeAndre Hopkins | 84.28 | 84.00 | 80.30 | 576 | Texans |
| 10 | 10 | Josh Gordon | 83.91 | 78.40 | 83.42 | 145 | Browns |
| 11 | 11 | T.Y. Hilton | 83.63 | 83.20 | 79.75 | 716 | Colts |
| 12 | 12 | Emmanuel Sanders | 83.58 | 86.40 | 77.54 | 661 | Broncos |
| 13 | 13 | DeSean Jackson | 83.51 | 74.90 | 85.09 | 509 | Commanders |
| 14 | 14 | Mike Evans | 83.45 | 84.60 | 78.51 | 544 | Buccaneers |
| 15 | 15 | Randall Cobb | 83.43 | 84.60 | 78.49 | 664 | Packers |
| 16 | 16 | Martavis Bryant | 83.07 | 75.10 | 84.22 | 256 | Steelers |
| 17 | 17 | Golden Tate | 82.28 | 80.00 | 79.63 | 706 | Lions |
| 18 | 18 | Andrew Hawkins | 81.93 | 83.50 | 76.71 | 424 | Browns |
| 19 | 19 | Jeremy Maclin | 81.80 | 80.30 | 78.63 | 652 | Eagles |
| 20 | 20 | Taylor Gabriel | 80.87 | 74.70 | 80.82 | 354 | Browns |
| 21 | 21 | Ricardo Lockette | 80.82 | 69.80 | 84.00 | 133 | Seahawks |
| 22 | 22 | Anquan Boldin | 80.73 | 79.30 | 77.51 | 589 | 49ers |
| 23 | 23 | Alshon Jeffery | 80.70 | 76.70 | 79.20 | 665 | Bears |
| 24 | 24 | Steve Smith | 80.46 | 79.00 | 77.26 | 601 | Ravens |
| 25 | 25 | Kenny Stills | 80.36 | 78.40 | 77.50 | 472 | Saints |

### Good (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Steve Johnson | 79.94 | 83.50 | 73.40 | 225 | 49ers |
| 27 | 2 | Doug Baldwin | 79.84 | 78.40 | 76.64 | 611 | Seahawks |
| 28 | 3 | Vincent Jackson | 79.50 | 75.70 | 77.86 | 621 | Buccaneers |
| 29 | 4 | Eric Decker | 78.86 | 78.30 | 75.07 | 476 | Jets |
| 30 | 5 | Andre Johnson | 78.65 | 75.30 | 76.72 | 514 | Texans |
| 31 | 6 | Malcom Floyd | 78.47 | 74.60 | 76.88 | 639 | Chargers |
| 32 | 7 | Keenan Allen | 78.10 | 76.10 | 75.27 | 527 | Chargers |
| 33 | 8 | Rueben Randle | 78.00 | 76.90 | 74.57 | 626 | Giants |
| 34 | 9 | Larry Fitzgerald | 77.91 | 77.20 | 74.22 | 583 | Cardinals |
| 35 | 10 | Stedman Bailey | 77.62 | 74.80 | 75.33 | 291 | Rams |
| 36 | 11 | Charles Johnson | 77.58 | 67.60 | 80.07 | 288 | Vikings |
| 37 | 12 | Donte Moncrief | 77.53 | 71.20 | 77.59 | 353 | Colts |
| 38 | 13 | Jarvis Landry | 77.31 | 78.00 | 72.69 | 452 | Dolphins |
| 39 | 14 | Jordan Matthews | 77.10 | 72.20 | 76.20 | 510 | Eagles |
| 40 | 15 | Torrey Smith | 77.09 | 71.50 | 76.65 | 572 | Ravens |
| 41 | 16 | Percy Harvin | 77.07 | 75.90 | 73.69 | 294 | Jets |
| 42 | 17 | Sammy Watkins | 76.61 | 69.30 | 77.32 | 649 | Bills |
| 43 | 18 | Brian Quick | 76.38 | 72.60 | 74.74 | 205 | Rams |
| 44 | 19 | Jarius Wright | 76.28 | 70.80 | 75.77 | 382 | Vikings |
| 45 | 20 | Brandon Marshall | 76.26 | 74.80 | 73.06 | 515 | Bears |
| 46 | 21 | Albert Wilson | 76.25 | 68.40 | 77.31 | 172 | Chiefs |
| 47 | 22 | Julian Edelman | 76.24 | 78.00 | 70.90 | 693 | Patriots |
| 48 | 23 | Miles Austin | 76.02 | 75.30 | 72.33 | 330 | Browns |
| 49 | 24 | Kenny Britt | 75.95 | 74.60 | 72.69 | 500 | Rams |
| 50 | 25 | Kelvin Benjamin | 75.93 | 75.00 | 72.39 | 641 | Panthers |
| 51 | 26 | Terrance Williams | 75.78 | 66.00 | 78.14 | 548 | Cowboys |
| 52 | 27 | Ted Ginn Jr. | 75.70 | 67.10 | 77.27 | 134 | Cardinals |
| 53 | 28 | Michael Floyd | 75.66 | 69.60 | 75.53 | 668 | Cardinals |
| 54 | 29 | Jermaine Kearse | 75.65 | 67.40 | 76.98 | 545 | Seahawks |
| 55 | 30 | Kendall Wright | 75.62 | 70.80 | 74.66 | 475 | Titans |
| 56 | 31 | John Brown | 75.51 | 69.30 | 75.48 | 526 | Cardinals |
| 57 | 32 | Pierre Garcon | 75.48 | 70.40 | 74.70 | 583 | Commanders |
| 58 | 33 | Eddie Royal | 75.44 | 72.40 | 73.30 | 545 | Chargers |
| 59 | 34 | Cole Beasley | 75.34 | 70.60 | 74.34 | 369 | Cowboys |
| 60 | 35 | Mike Wallace | 75.26 | 72.60 | 72.87 | 531 | Dolphins |
| 61 | 36 | Brandon LaFell | 75.04 | 72.60 | 72.50 | 715 | Patriots |
| 62 | 37 | Robert Meachem | 74.92 | 68.00 | 75.37 | 143 | Saints |
| 63 | 38 | James Wright | 74.60 | 60.80 | 79.63 | 118 | Bengals |
| 64 | 39 | Travis Benjamin | 74.59 | 68.80 | 74.29 | 202 | Browns |
| 65 | 40 | Corey Brown | 74.50 | 69.10 | 73.93 | 228 | Panthers |
| 66 | 41 | Allen Robinson II | 74.44 | 69.40 | 73.64 | 357 | Jaguars |
| 67 | 42 | Dwayne Bowe | 74.28 | 71.70 | 71.83 | 507 | Chiefs |
| 68 | 43 | Greg Jennings | 74.15 | 71.90 | 71.49 | 595 | Vikings |

### Starter (62 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 69 | 1 | Harry Douglas | 73.56 | 70.10 | 71.70 | 418 | Falcons |
| 70 | 2 | Marques Colston | 73.52 | 65.30 | 74.84 | 613 | Saints |
| 71 | 3 | Nate Washington | 73.50 | 64.80 | 75.14 | 540 | Titans |
| 72 | 4 | Markus Wheaton | 73.35 | 69.60 | 71.68 | 552 | Steelers |
| 73 | 5 | Andre Holmes | 73.32 | 63.90 | 75.43 | 497 | Raiders |
| 74 | 6 | Jaron Brown | 73.24 | 73.90 | 68.64 | 138 | Cardinals |
| 75 | 7 | Justin Hunter | 72.83 | 62.90 | 75.29 | 399 | Titans |
| 76 | 8 | Brenton Bersin | 72.80 | 70.10 | 70.43 | 117 | Panthers |
| 77 | 9 | Kamar Aiken | 72.66 | 69.80 | 70.40 | 187 | Ravens |
| 78 | 10 | Michael Crabtree | 72.64 | 66.10 | 72.83 | 505 | 49ers |
| 79 | 11 | Brandin Cooks | 72.47 | 68.90 | 70.69 | 373 | Saints |
| 80 | 12 | Roddy White | 72.30 | 68.50 | 70.66 | 620 | Falcons |
| 81 | 13 | Brice Butler | 72.23 | 70.10 | 69.48 | 195 | Raiders |
| 82 | 14 | Victor Cruz | 72.21 | 61.80 | 74.98 | 206 | Giants |
| 83 | 15 | Greg Salas | 71.90 | 62.70 | 73.87 | 137 | Jets |
| 84 | 16 | Devin Hester | 71.74 | 64.30 | 72.53 | 315 | Falcons |
| 85 | 17 | Marlon Brown | 71.67 | 71.10 | 67.88 | 283 | Ravens |
| 86 | 18 | Cordarrelle Patterson | 71.59 | 67.30 | 70.28 | 380 | Vikings |
| 87 | 19 | James Jones | 71.33 | 65.70 | 70.92 | 547 | Raiders |
| 88 | 20 | Brandon Lloyd | 71.19 | 63.30 | 72.28 | 250 | 49ers |
| 89 | 21 | Chris Givens | 71.19 | 56.20 | 77.01 | 130 | Rams |
| 90 | 22 | Brian Hartline | 71.14 | 63.20 | 72.27 | 515 | Dolphins |
| 91 | 23 | Hakeem Nicks | 71.14 | 62.20 | 72.93 | 491 | Colts |
| 92 | 24 | Rishard Matthews | 70.78 | 64.90 | 70.53 | 112 | Dolphins |
| 93 | 25 | Robert Woods | 70.53 | 68.60 | 67.65 | 608 | Bills |
| 94 | 26 | Mohamed Sanu | 70.53 | 67.30 | 68.51 | 602 | Bengals |
| 95 | 27 | Brandon Tate | 70.44 | 62.00 | 71.90 | 327 | Bengals |
| 96 | 28 | Cecil Shorts | 70.25 | 62.40 | 71.32 | 507 | Jaguars |
| 97 | 29 | Jeremy Kerley | 70.20 | 61.70 | 71.70 | 441 | Jets |
| 98 | 30 | Davante Adams | 70.18 | 63.20 | 70.66 | 575 | Packers |
| 99 | 31 | Jerricho Cotchery | 70.11 | 64.60 | 69.62 | 576 | Panthers |
| 100 | 32 | Reggie Wayne | 70.04 | 61.70 | 71.43 | 686 | Colts |
| 101 | 33 | Chris Hogan | 69.83 | 64.50 | 69.22 | 360 | Bills |
| 102 | 34 | Allen Hurns | 69.61 | 62.00 | 70.52 | 541 | Jaguars |
| 103 | 35 | Donnie Avery | 69.40 | 61.80 | 70.30 | 154 | Chiefs |
| 104 | 36 | Paul Richardson Jr. | 69.37 | 67.70 | 66.31 | 322 | Seahawks |
| 105 | 37 | Kenbrell Thompkins | 68.96 | 60.20 | 70.64 | 272 | Raiders |
| 106 | 38 | Preston Parker | 68.93 | 62.50 | 69.05 | 425 | Giants |
| 107 | 39 | Corey Fuller | 68.88 | 59.40 | 71.03 | 278 | Lions |
| 108 | 40 | Vincent Brown | 68.81 | 61.10 | 69.79 | 106 | Raiders |
| 109 | 41 | Riley Cooper | 68.77 | 59.10 | 71.05 | 586 | Eagles |
| 110 | 42 | Lance Moore | 68.39 | 57.20 | 71.68 | 210 | Steelers |
| 111 | 43 | Wes Welker | 68.38 | 60.90 | 69.20 | 546 | Broncos |
| 112 | 44 | Nick Toon | 68.35 | 64.40 | 66.81 | 180 | Saints |
| 113 | 45 | Eric Weems | 68.26 | 61.70 | 68.47 | 116 | Falcons |
| 114 | 46 | Jason Avant | 68.11 | 62.80 | 67.49 | 386 | Chiefs |
| 115 | 47 | Andre Roberts | 67.60 | 61.20 | 67.70 | 495 | Commanders |
| 116 | 48 | Marqise Lee | 67.55 | 61.40 | 67.48 | 348 | Jaguars |
| 117 | 49 | Jeremy Ross | 67.31 | 58.80 | 68.82 | 519 | Lions |
| 118 | 50 | Denarius Moore | 67.26 | 51.70 | 73.47 | 164 | Raiders |
| 119 | 51 | Jacoby Jones | 67.23 | 53.80 | 72.02 | 132 | Ravens |
| 120 | 52 | Derek Hagan | 67.02 | 58.90 | 68.26 | 213 | Titans |
| 121 | 53 | Louis Murphy Jr. | 66.79 | 60.50 | 66.82 | 353 | Buccaneers |
| 122 | 54 | Danny Amendola | 66.69 | 58.90 | 67.72 | 408 | Patriots |
| 123 | 55 | A.J. Jenkins | 66.07 | 53.50 | 70.29 | 157 | Chiefs |
| 124 | 56 | Brandon Gibson | 65.45 | 56.60 | 67.19 | 330 | Dolphins |
| 125 | 57 | Josh Huff | 65.29 | 58.30 | 65.79 | 114 | Eagles |
| 126 | 58 | Tavon Austin | 64.91 | 59.30 | 64.48 | 342 | Rams |
| 127 | 59 | Keshawn Martin | 64.42 | 56.10 | 65.80 | 110 | Texans |
| 128 | 60 | Dane Sanzenbacher | 63.88 | 50.20 | 68.83 | 128 | Bengals |
| 129 | 61 | Damaris Johnson | 63.66 | 54.40 | 65.67 | 352 | Texans |
| 130 | 62 | Jarrett Boykin | 62.62 | 44.30 | 70.66 | 149 | Packers |

### Rotation/backup (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 131 | 1 | Junior Hemingway | 61.75 | 53.10 | 63.35 | 150 | Chiefs |
| 132 | 2 | Ryan Grant | 61.65 | 54.50 | 62.25 | 114 | Commanders |
| 133 | 3 | Josh Morgan | 61.58 | 52.50 | 63.47 | 271 | Bears |
| 134 | 4 | Seyi Ajirotutu | 59.86 | 48.80 | 63.06 | 106 | Chargers |
| 135 | 5 | Andre Caldwell | 59.76 | 48.90 | 62.83 | 107 | Broncos |
| 136 | 6 | Marquess Wilson | 59.42 | 53.40 | 59.26 | 268 | Bears |
| 137 | 7 | Frankie Hammond | 57.31 | 46.40 | 60.41 | 157 | Chiefs |
