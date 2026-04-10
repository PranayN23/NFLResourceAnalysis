# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:27:09Z
- **Requested analysis_year:** 2012 (clamped to 2012)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | John Sullivan | 91.18 | 84.70 | 91.34 | 1096 | Vikings |
| 2 | 2 | Ryan Wendell | 91.13 | 84.90 | 91.11 | 1383 | Patriots |
| 3 | 3 | Brian De La Puente | 90.98 | 84.40 | 91.20 | 1107 | Saints |
| 4 | 4 | Nick Mangold | 90.89 | 84.80 | 90.79 | 1066 | Jets |
| 5 | 5 | Chris Myers | 90.26 | 84.00 | 90.26 | 1284 | Texans |
| 6 | 6 | Max Unger | 89.60 | 86.20 | 87.70 | 1115 | Seahawks |
| 7 | 7 | Ryan Lilja | 88.95 | 82.10 | 89.35 | 990 | Chiefs |
| 8 | 8 | Matt Birk | 88.77 | 82.20 | 88.99 | 1299 | Ravens |
| 9 | 9 | Alex Mack | 88.64 | 79.70 | 90.43 | 1031 | Browns |
| 10 | 10 | Jason Kelce | 88.17 | 80.20 | 89.32 | 136 | Eagles |
| 11 | 11 | Mike Pouncey | 86.56 | 78.50 | 87.77 | 1032 | Dolphins |
| 12 | 12 | Phil Costa | 86.24 | 78.20 | 87.44 | 121 | Cowboys |
| 13 | 13 | Jonathan Goodwin | 86.05 | 78.80 | 86.71 | 1171 | 49ers |
| 14 | 14 | Will Montgomery | 85.62 | 77.70 | 86.73 | 1082 | Commanders |
| 15 | 15 | J.D. Walton | 84.86 | 75.40 | 87.00 | 248 | Broncos |
| 16 | 16 | Maurkice Pouncey | 83.99 | 75.60 | 85.41 | 930 | Steelers |
| 17 | 17 | Todd McClure | 82.36 | 73.70 | 83.96 | 1175 | Falcons |
| 18 | 18 | David Baas | 82.36 | 73.40 | 84.17 | 1003 | Giants |
| 19 | 19 | Dominic Raiola | 81.67 | 73.10 | 83.21 | 1199 | Lions |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Rodney Hudson | 79.72 | 77.00 | 77.36 | 183 | Chiefs |
| 21 | 2 | Robert Turner | 79.71 | 70.00 | 82.02 | 1044 | Rams |
| 22 | 3 | Evan Smith | 79.54 | 69.40 | 82.13 | 588 | Packers |
| 23 | 4 | Ryan Cook | 79.39 | 68.80 | 82.28 | 813 | Cowboys |
| 24 | 5 | Samson Satele | 79.19 | 68.60 | 82.08 | 720 | Colts |
| 25 | 6 | Trevor Robinson | 78.65 | 69.50 | 80.59 | 448 | Bengals |
| 26 | 7 | Eric Wood | 77.97 | 68.50 | 80.12 | 872 | Bills |
| 27 | 8 | A.Q. Shipley | 77.71 | 72.00 | 77.35 | 468 | Colts |
| 28 | 9 | Dan Koppen | 76.46 | 66.60 | 78.86 | 981 | Broncos |
| 29 | 10 | Nick Hardwick | 75.99 | 65.40 | 78.88 | 1019 | Chargers |
| 30 | 11 | Roberto Garza | 75.80 | 64.80 | 78.96 | 1046 | Bears |
| 31 | 12 | Lyle Sendlein | 75.28 | 65.70 | 77.50 | 756 | Cardinals |
| 32 | 13 | Scott Wells | 75.23 | 63.80 | 78.69 | 421 | Rams |

### Starter (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Brad Meester | 73.43 | 63.20 | 76.09 | 1062 | Jaguars |
| 34 | 2 | Jeff Saturday | 72.49 | 62.50 | 74.99 | 962 | Packers |
| 35 | 3 | Steve Vallos | 71.77 | 62.30 | 73.91 | 124 | Jaguars |
| 36 | 4 | Ryan Kalil | 71.42 | 59.00 | 75.53 | 286 | Panthers |
| 37 | 5 | Kyle Cook | 70.32 | 57.70 | 74.57 | 205 | Bengals |
| 38 | 6 | Doug Legursky | 69.12 | 56.90 | 73.10 | 408 | Steelers |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Alex Parsons | 61.62 | 51.90 | 63.94 | 120 | Raiders |

## CB — Cornerback

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Casey Hayward Jr. | 96.35 | 91.60 | 95.35 | 748 | Packers |
| 2 | 2 | Richard Sherman | 95.29 | 92.50 | 93.77 | 1067 | Seahawks |
| 3 | 3 | Charles Tillman | 93.13 | 91.90 | 89.79 | 921 | Bears |
| 4 | 4 | Sam Shields | 90.64 | 90.60 | 88.58 | 720 | Packers |
| 5 | 5 | Phillip Adams | 90.24 | 90.70 | 89.83 | 175 | Raiders |
| 6 | 6 | Chris Harris Jr. | 90.08 | 90.00 | 86.75 | 983 | Broncos |
| 7 | 7 | Tarell Brown | 88.94 | 84.50 | 88.36 | 1224 | 49ers |
| 8 | 8 | Brandon Flowers | 87.49 | 85.70 | 85.03 | 864 | Chiefs |
| 9 | 9 | Greg Toler | 86.85 | 85.80 | 88.07 | 302 | Cardinals |
| 10 | 10 | Cortez Allen | 85.43 | 84.70 | 85.14 | 543 | Steelers |
| 11 | 11 | Kareem Jackson | 85.37 | 80.10 | 84.71 | 1121 | Texans |
| 12 | 12 | Robert McClain | 84.53 | 81.20 | 82.58 | 663 | Falcons |
| 13 | 13 | Asante Samuel | 83.82 | 80.10 | 83.59 | 924 | Falcons |
| 14 | 14 | Leon Hall | 82.82 | 80.10 | 83.17 | 960 | Bengals |
| 15 | 15 | Tim Jennings | 82.71 | 78.90 | 82.11 | 886 | Bears |
| 16 | 16 | Joselio Hanson | 82.14 | 79.00 | 80.59 | 553 | Raiders |
| 17 | 17 | Trumaine Johnson | 81.92 | 77.20 | 81.94 | 355 | Rams |
| 18 | 18 | Patrick Peterson | 81.70 | 77.80 | 80.14 | 1042 | Cardinals |
| 19 | 19 | Joe Haden | 81.45 | 75.00 | 84.50 | 774 | Browns |
| 20 | 20 | Antonio Cromartie | 81.29 | 75.10 | 81.25 | 1036 | Jets |
| 21 | 21 | Adam Jones | 81.21 | 80.30 | 82.44 | 613 | Bengals |
| 22 | 22 | E.J. Biggers | 80.79 | 78.70 | 79.59 | 796 | Buccaneers |
| 23 | 23 | Chris Owens | 80.58 | 78.50 | 82.58 | 170 | Falcons |
| 24 | 24 | Jason McCourty | 80.21 | 74.00 | 81.33 | 1126 | Titans |
| 25 | 25 | Terence Newman | 80.16 | 73.80 | 80.86 | 944 | Bengals |
| 26 | 26 | Sheldon Brown | 80.00 | 72.30 | 81.48 | 879 | Browns |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | Brandon Browner | 79.58 | 72.50 | 81.43 | 861 | Seahawks |
| 28 | 2 | Vontae Davis | 78.89 | 75.50 | 80.83 | 645 | Colts |
| 29 | 3 | Alterraun Verner | 78.70 | 73.80 | 77.80 | 1045 | Titans |
| 30 | 4 | Alfonzo Dennard | 78.67 | 75.50 | 80.79 | 734 | Patriots |
| 31 | 5 | Tramon Williams | 78.43 | 68.90 | 80.61 | 1209 | Packers |
| 32 | 6 | Champ Bailey | 78.40 | 68.30 | 81.48 | 1090 | Broncos |
| 33 | 7 | Tony Carter | 77.52 | 72.00 | 78.06 | 541 | Broncos |
| 34 | 8 | Jabari Greer | 77.42 | 71.40 | 78.52 | 818 | Saints |
| 35 | 9 | Johnathan Joseph | 77.36 | 70.30 | 78.74 | 940 | Texans |
| 36 | 10 | Cortland Finnegan | 77.25 | 71.40 | 76.98 | 1005 | Rams |
| 37 | 11 | Josh Wilson | 77.18 | 69.80 | 78.13 | 1097 | Commanders |
| 38 | 12 | Antoine Winfield | 76.62 | 74.10 | 77.57 | 1076 | Vikings |
| 39 | 13 | Brandon Boykin | 76.56 | 70.30 | 76.56 | 509 | Eagles |
| 40 | 14 | Chris Culliver | 75.93 | 67.80 | 77.56 | 798 | 49ers |
| 41 | 15 | Morris Claiborne | 75.88 | 69.00 | 77.33 | 879 | Cowboys |
| 42 | 16 | Chris Houston | 75.81 | 69.10 | 77.82 | 899 | Lions |
| 43 | 17 | Davon House | 75.50 | 72.40 | 78.60 | 311 | Packers |
| 44 | 18 | D.J. Moore | 75.44 | 69.90 | 77.98 | 363 | Bears |
| 45 | 19 | Javier Arenas | 75.31 | 66.80 | 77.13 | 710 | Chiefs |
| 46 | 20 | DeAngelo Hall | 75.19 | 66.90 | 76.55 | 1114 | Commanders |
| 47 | 21 | Bradley Fletcher | 75.18 | 69.90 | 78.28 | 363 | Rams |
| 48 | 22 | Carlos Rogers | 74.99 | 66.50 | 77.31 | 1230 | 49ers |
| 49 | 23 | Leonard Johnson | 74.79 | 70.40 | 77.71 | 581 | Buccaneers |
| 50 | 24 | Captain Munnerlyn | 74.49 | 67.40 | 75.69 | 913 | Panthers |
| 51 | 25 | Shareece Wright | 74.30 | 74.40 | 80.48 | 116 | Chargers |

### Starter (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 52 | 1 | Brandon Carr | 73.95 | 64.50 | 76.08 | 1010 | Cowboys |
| 53 | 2 | Dominique Rodgers-Cromartie | 73.58 | 64.90 | 76.14 | 993 | Eagles |
| 54 | 3 | Dominique Franks | 73.45 | 69.40 | 81.79 | 203 | Falcons |
| 55 | 4 | Prince Amukamara | 73.36 | 71.10 | 75.38 | 728 | Giants |
| 56 | 5 | Elbert Mack | 73.06 | 74.10 | 76.67 | 254 | Saints |
| 57 | 6 | Cary Williams | 72.84 | 63.10 | 77.86 | 1405 | Ravens |
| 58 | 7 | Kyle Arrington | 72.69 | 62.80 | 75.11 | 929 | Patriots |
| 59 | 8 | Orlando Scandrick | 72.12 | 69.50 | 73.25 | 326 | Cowboys |
| 60 | 9 | Keenan Lewis | 71.87 | 63.60 | 75.71 | 918 | Steelers |
| 61 | 10 | Leodis McKelvin | 71.86 | 66.50 | 76.17 | 347 | Bills |
| 62 | 11 | Derek Cox | 71.69 | 68.20 | 75.68 | 764 | Jaguars |
| 63 | 12 | Stephon Gilmore | 71.51 | 61.40 | 74.08 | 1055 | Bills |
| 64 | 13 | Sean Smith | 71.46 | 61.60 | 74.07 | 1046 | Dolphins |
| 65 | 14 | Janoris Jenkins | 71.30 | 63.00 | 73.70 | 955 | Rams |
| 66 | 15 | Jonte Green | 70.94 | 65.70 | 74.43 | 396 | Lions |
| 67 | 16 | Lardarius Webb | 70.89 | 64.90 | 75.92 | 376 | Ravens |
| 68 | 17 | Chris Cook | 70.88 | 68.10 | 76.38 | 677 | Vikings |
| 69 | 18 | Aqib Talib | 70.30 | 61.00 | 76.40 | 658 | Patriots |
| 70 | 19 | Kyle Wilson | 69.52 | 60.40 | 71.63 | 947 | Jets |
| 71 | 20 | Dunta Robinson | 69.34 | 61.10 | 70.67 | 1045 | Falcons |
| 72 | 21 | Cedric Griffin | 69.14 | 63.60 | 75.02 | 377 | Commanders |
| 73 | 22 | Buster Skrine | 69.04 | 60.40 | 72.59 | 722 | Browns |
| 74 | 23 | Chris Gamble | 68.95 | 64.60 | 75.29 | 280 | Panthers |
| 75 | 24 | Mike Harris | 68.87 | 58.00 | 75.08 | 529 | Jaguars |
| 76 | 25 | Eric Wright | 68.87 | 62.70 | 72.57 | 494 | Buccaneers |
| 77 | 26 | Jerraud Powers | 68.78 | 65.70 | 73.34 | 505 | Colts |
| 78 | 27 | Alan Ball | 68.47 | 60.60 | 73.71 | 105 | Texans |
| 79 | 28 | Marcus Trufant | 68.18 | 62.90 | 72.32 | 393 | Seahawks |
| 80 | 29 | Richard Crawford | 68.08 | 67.40 | 69.57 | 199 | Commanders |
| 81 | 30 | Jacob Lacey | 68.03 | 64.50 | 70.07 | 575 | Lions |
| 82 | 31 | Stanford Routt | 67.44 | 60.00 | 70.41 | 406 | Texans |
| 83 | 32 | Bryan McCann | 67.19 | 65.20 | 74.86 | 141 | Dolphins |
| 84 | 33 | Byron Maxwell | 67.07 | 71.80 | 76.75 | 144 | Seahawks |
| 85 | 34 | Antoine Cason | 67.01 | 52.70 | 72.39 | 1021 | Chargers |
| 86 | 35 | Aaron Ross | 66.99 | 55.90 | 71.47 | 671 | Jaguars |
| 87 | 36 | Kelvin Hayden | 66.48 | 61.50 | 70.21 | 462 | Bears |
| 88 | 37 | Josh Thomas | 66.32 | 59.80 | 74.32 | 515 | Panthers |
| 89 | 38 | Nolan Carroll | 66.22 | 57.80 | 72.56 | 641 | Dolphins |
| 90 | 39 | Drayton Florence | 66.19 | 57.50 | 71.99 | 303 | Lions |
| 91 | 40 | Tracy Porter | 66.10 | 60.70 | 71.37 | 301 | Broncos |
| 92 | 41 | Ellis Lankster | 65.55 | 55.10 | 69.00 | 326 | Jets |
| 93 | 42 | Perrish Cox | 65.44 | 53.70 | 69.48 | 180 | 49ers |
| 94 | 43 | Mike Jenkins | 65.43 | 56.30 | 70.16 | 356 | Cowboys |
| 95 | 44 | Shawntae Spencer | 65.39 | 64.30 | 72.05 | 111 | Raiders |
| 96 | 45 | Brandian Ross | 65.32 | 67.70 | 70.44 | 175 | Raiders |
| 97 | 46 | Josh Norman | 64.91 | 57.50 | 68.82 | 771 | Panthers |
| 98 | 47 | Ron Brooks | 64.38 | 60.50 | 73.66 | 161 | Bills |
| 99 | 48 | Patrick Robinson | 64.20 | 52.00 | 69.64 | 1093 | Saints |
| 100 | 49 | Nnamdi Asomugha | 63.59 | 47.20 | 70.77 | 984 | Eagles |
| 101 | 50 | Marquice Cole | 63.51 | 59.20 | 67.42 | 227 | Patriots |
| 102 | 51 | Rashean Mathis | 63.50 | 54.40 | 69.67 | 476 | Jaguars |
| 103 | 52 | A.J. Jefferson | 63.37 | 54.40 | 69.35 | 631 | Vikings |
| 104 | 53 | Anthony Gaitor | 63.32 | 59.80 | 73.62 | 123 | Buccaneers |
| 105 | 54 | Coty Sensabaugh | 63.15 | 54.20 | 67.03 | 313 | Titans |
| 106 | 55 | Corey Webster | 62.83 | 47.00 | 69.41 | 1016 | Giants |
| 107 | 56 | Richard Marshall | 62.08 | 54.00 | 69.55 | 237 | Dolphins |

### Rotation/backup (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 108 | 1 | Isaiah Trufant | 61.86 | 58.80 | 71.97 | 118 | Jets |
| 109 | 2 | William Middleton | 61.77 | 58.50 | 66.96 | 196 | Jaguars |
| 110 | 3 | James Dockery | 61.75 | 62.80 | 72.38 | 158 | Panthers |
| 111 | 4 | William Gay | 61.28 | 49.00 | 65.30 | 1006 | Cardinals |
| 112 | 5 | Brice McCain | 60.70 | 49.00 | 67.46 | 450 | Texans |
| 113 | 6 | Sterling Moore | 60.62 | 58.00 | 61.33 | 327 | Cowboys |
| 114 | 7 | Corey White | 60.13 | 56.00 | 66.02 | 519 | Saints |
| 115 | 8 | Jimmy Smith | 59.54 | 49.20 | 66.16 | 519 | Ravens |
| 116 | 9 | Danny Gorrer | 59.20 | 52.30 | 70.05 | 185 | Buccaneers |
| 117 | 10 | Justin Rogers | 58.92 | 44.40 | 67.82 | 537 | Bills |
| 118 | 11 | Ronald Bartell | 58.35 | 52.00 | 68.00 | 413 | Lions |
| 119 | 12 | Jeremy Lane | 57.76 | 64.20 | 67.80 | 163 | Seahawks |
| 120 | 13 | Brandon Harris | 57.73 | 48.40 | 70.47 | 237 | Texans |
| 121 | 14 | Chykie Brown | 57.70 | 50.30 | 66.54 | 440 | Ravens |
| 122 | 15 | Cassius Vaughn | 57.65 | 47.50 | 66.40 | 861 | Colts |
| 123 | 16 | Trevin Wade | 56.19 | 55.50 | 56.65 | 196 | Browns |
| 124 | 17 | Terrence McGee | 53.27 | 46.80 | 62.69 | 142 | Bills |
| 125 | 18 | R.J. Stanford | 52.30 | 41.70 | 60.80 | 144 | Dolphins |
| 126 | 19 | Jayron Hosley | 50.68 | 43.50 | 55.47 | 452 | Giants |
| 127 | 20 | Kevin Rutland | 50.38 | 53.60 | 54.10 | 103 | Jaguars |
| 128 | 21 | Johnny Patrick | 50.22 | 36.80 | 59.95 | 215 | Saints |
| 129 | 22 | Ryan Mouton | 50.14 | 45.60 | 56.30 | 387 | Titans |
| 130 | 23 | Jalil Brown | 48.50 | 46.30 | 54.26 | 363 | Chiefs |

## DI — Defensive Interior

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | J.J. Watt | 97.03 | 90.16 | 97.45 | 1050 | Texans |
| 2 | 2 | Geno Atkins | 91.19 | 87.53 | 89.47 | 848 | Bengals |
| 3 | 3 | Steve McLendon | 87.47 | 86.20 | 86.96 | 135 | Steelers |
| 4 | 4 | Nick Fairley | 86.22 | 84.10 | 87.37 | 494 | Lions |
| 5 | 5 | Calais Campbell | 85.82 | 83.07 | 85.25 | 754 | Cardinals |
| 6 | 6 | Muhammad Wilkerson | 85.64 | 83.75 | 82.73 | 910 | Jets |
| 7 | 7 | Mike Martin | 85.20 | 78.18 | 85.72 | 429 | Titans |
| 8 | 8 | Fletcher Cox | 83.67 | 80.12 | 82.91 | 509 | Eagles |
| 9 | 9 | Desmond Bryant | 81.97 | 77.90 | 80.90 | 630 | Raiders |
| 10 | 10 | Ndamukong Suh | 81.81 | 80.36 | 78.93 | 879 | Lions |
| 11 | 11 | Kyle Williams | 80.15 | 76.02 | 82.17 | 793 | Bills |
| 12 | 12 | Jurrell Casey | 80.07 | 79.41 | 76.35 | 779 | Titans |

### Good (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Haloti Ngata | 79.69 | 79.75 | 75.48 | 1053 | Ravens |
| 14 | 2 | Marcell Dareus | 79.29 | 77.22 | 76.51 | 787 | Bills |
| 15 | 3 | Justin Smith | 78.67 | 68.76 | 81.11 | 996 | 49ers |
| 16 | 4 | Fred Evans | 77.37 | 68.46 | 80.81 | 360 | Vikings |
| 17 | 5 | Henry Melton | 77.05 | 66.58 | 81.21 | 607 | Bears |
| 18 | 6 | Alex Carrington | 77.02 | 74.71 | 77.13 | 342 | Bills |
| 19 | 7 | Gerald McCoy | 76.83 | 88.75 | 68.46 | 938 | Buccaneers |
| 20 | 8 | Brodrick Bunkley | 75.78 | 68.16 | 77.43 | 365 | Saints |
| 21 | 9 | Chris Canty | 75.04 | 75.03 | 74.53 | 297 | Giants |
| 22 | 10 | Richard Seymour | 75.01 | 72.63 | 77.21 | 348 | Raiders |
| 23 | 11 | Dan Williams | 74.80 | 79.32 | 70.22 | 419 | Cardinals |
| 24 | 12 | Linval Joseph | 74.59 | 69.86 | 75.86 | 692 | Giants |
| 25 | 13 | Paul Soliai | 74.53 | 63.77 | 77.53 | 616 | Dolphins |
| 26 | 14 | Antonio Garay | 74.28 | 57.10 | 85.74 | 147 | Chargers |
| 27 | 15 | Karl Klug | 74.12 | 66.81 | 74.82 | 250 | Titans |

### Starter (85 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Cameron Heyward | 73.55 | 63.06 | 76.37 | 262 | Steelers |
| 29 | 2 | Mike Devito | 73.33 | 66.28 | 75.11 | 629 | Jets |
| 30 | 3 | Jason Hatcher | 73.28 | 63.07 | 77.49 | 757 | Cowboys |
| 31 | 4 | Vince Wilfork | 73.23 | 63.92 | 75.27 | 1019 | Patriots |
| 32 | 5 | Michael Brockers | 73.12 | 65.79 | 76.97 | 602 | Rams |
| 33 | 6 | Sammie Lee Hill | 72.86 | 64.48 | 75.01 | 403 | Lions |
| 34 | 7 | Brandon Mebane | 72.33 | 63.29 | 74.61 | 706 | Seahawks |
| 35 | 8 | Jason Jones | 72.21 | 55.12 | 82.44 | 319 | Seahawks |
| 36 | 9 | Corey Liuget | 71.99 | 64.43 | 73.25 | 716 | Chargers |
| 37 | 10 | Antonio Smith | 71.94 | 64.25 | 72.90 | 904 | Texans |
| 38 | 11 | Tyson Jackson | 71.61 | 65.19 | 72.88 | 595 | Chiefs |
| 39 | 12 | Cam Thomas | 71.39 | 63.47 | 74.58 | 395 | Chargers |
| 40 | 13 | Kevin Williams | 71.36 | 69.14 | 69.31 | 859 | Vikings |
| 41 | 14 | Vance Walker | 71.36 | 60.20 | 74.63 | 589 | Falcons |
| 42 | 15 | Josh Price-Brent | 71.20 | 77.26 | 66.64 | 309 | Cowboys |
| 43 | 16 | Ricky Jean Francois | 70.90 | 59.62 | 74.26 | 322 | 49ers |
| 44 | 17 | Ahtyba Rubin | 70.90 | 65.99 | 71.57 | 670 | Browns |
| 45 | 18 | Earl Mitchell | 70.80 | 60.45 | 73.74 | 426 | Texans |
| 46 | 19 | C.J. Mosley | 70.61 | 65.06 | 72.13 | 615 | Jaguars |
| 47 | 20 | Aubrayo Franklin | 70.45 | 67.11 | 71.11 | 279 | Chargers |
| 48 | 21 | Cullen Jenkins | 70.34 | 51.12 | 79.18 | 625 | Eagles |
| 49 | 22 | Ray McDonald | 70.34 | 65.16 | 69.63 | 1138 | 49ers |
| 50 | 23 | Glenn Dorsey | 70.31 | 71.40 | 71.99 | 112 | Chiefs |
| 51 | 24 | Sione Pouha | 70.04 | 60.34 | 74.43 | 304 | Jets |
| 52 | 25 | Alan Branch | 69.99 | 62.79 | 70.94 | 630 | Seahawks |
| 53 | 26 | Derek Landri | 69.01 | 56.93 | 74.15 | 484 | Eagles |
| 54 | 27 | Christian Ballard | 68.96 | 59.97 | 70.79 | 402 | Vikings |
| 55 | 28 | C.J. Wilson | 68.66 | 53.68 | 76.05 | 351 | Packers |
| 56 | 29 | Kenyon Coleman | 68.60 | 57.91 | 77.43 | 164 | Cowboys |
| 57 | 30 | Phil Taylor Sr. | 68.52 | 62.70 | 73.44 | 263 | Browns |
| 58 | 31 | Cory Redding | 68.47 | 48.44 | 78.17 | 634 | Colts |
| 59 | 32 | Randy Starks | 68.38 | 59.37 | 70.22 | 810 | Dolphins |
| 60 | 33 | Domata Peko Sr. | 68.36 | 53.69 | 73.98 | 705 | Bengals |
| 61 | 34 | Terrance Knighton | 68.26 | 62.57 | 68.82 | 656 | Jaguars |
| 62 | 35 | Derek Wolfe | 68.10 | 61.21 | 68.53 | 973 | Broncos |
| 63 | 36 | Ropati Pitoitua | 67.99 | 51.70 | 76.12 | 494 | Chiefs |
| 64 | 37 | Akiem Hicks | 67.83 | 63.19 | 68.84 | 372 | Saints |
| 65 | 38 | Kendall Reyes | 67.61 | 57.93 | 69.89 | 531 | Chargers |
| 66 | 39 | Vonnie Holliday | 67.58 | 40.58 | 81.62 | 195 | Cardinals |
| 67 | 40 | Arthur Jones | 67.52 | 57.73 | 72.80 | 694 | Ravens |
| 68 | 41 | Jared Crick | 67.42 | 60.22 | 68.06 | 228 | Texans |
| 69 | 42 | Ryan Pickett | 67.38 | 56.21 | 70.98 | 645 | Packers |
| 70 | 43 | Gary Gibson | 67.14 | 58.80 | 68.53 | 278 | Buccaneers |
| 71 | 44 | Tommy Kelly | 66.90 | 54.99 | 70.67 | 758 | Raiders |
| 72 | 45 | Tom Johnson | 66.90 | 56.72 | 70.55 | 419 | Saints |
| 73 | 46 | David Carter | 66.89 | 57.48 | 69.64 | 289 | Cardinals |
| 74 | 47 | Sean Lissemore | 66.75 | 54.28 | 74.79 | 317 | Cowboys |
| 75 | 48 | Jared Odrick | 66.61 | 59.24 | 70.49 | 932 | Dolphins |
| 76 | 49 | Kenrick Ellis | 66.53 | 57.36 | 75.38 | 232 | Jets |
| 77 | 50 | Stephen Paea | 66.34 | 59.82 | 69.12 | 595 | Bears |
| 78 | 51 | Jonathan Babineaux | 66.24 | 56.05 | 69.50 | 938 | Falcons |
| 79 | 52 | Kyle Love | 65.97 | 57.76 | 68.53 | 579 | Patriots |
| 80 | 53 | Kellen Heard | 65.91 | 51.25 | 75.16 | 179 | Colts |
| 81 | 54 | Dontari Poe | 65.77 | 57.23 | 67.30 | 743 | Chiefs |
| 82 | 55 | Spencer Johnson | 65.62 | 48.22 | 75.13 | 264 | Bills |
| 83 | 56 | Mike Daniels | 65.47 | 49.88 | 71.69 | 272 | Packers |
| 84 | 57 | Corey Williams | 65.42 | 61.22 | 68.73 | 219 | Lions |
| 85 | 58 | Pat Sims | 65.38 | 54.49 | 74.11 | 205 | Bengals |
| 86 | 59 | Tyson Alualu | 65.10 | 55.02 | 67.65 | 836 | Jaguars |
| 87 | 60 | Kevin Vickerson | 64.96 | 54.07 | 72.35 | 518 | Broncos |
| 88 | 61 | B.J. Raji | 64.94 | 59.43 | 64.44 | 751 | Packers |
| 89 | 62 | Sedrick Ellis | 64.91 | 61.16 | 63.25 | 709 | Saints |
| 90 | 63 | Letroy Guion | 64.76 | 51.80 | 69.43 | 520 | Vikings |
| 91 | 64 | Barry Cofield | 64.72 | 56.98 | 65.72 | 773 | Commanders |
| 92 | 65 | Mike Patterson | 64.71 | 56.55 | 72.04 | 133 | Eagles |
| 93 | 66 | Shaun Smith | 64.65 | 48.86 | 75.49 | 131 | Chiefs |
| 94 | 67 | Stephen Bowen | 64.64 | 50.20 | 70.10 | 822 | Commanders |
| 95 | 68 | Amobi Okoye | 64.54 | 56.23 | 69.57 | 229 | Bears |
| 96 | 69 | Brett Keisel | 64.47 | 50.96 | 70.05 | 860 | Steelers |
| 97 | 70 | John Hughes | 64.29 | 50.82 | 69.10 | 514 | Browns |
| 98 | 71 | Jay Ratliff | 64.18 | 57.28 | 69.81 | 261 | Cowboys |
| 99 | 72 | Cedric Thornton | 64.10 | 44.50 | 73.00 | 395 | Eagles |
| 100 | 73 | Shaun Cody | 63.68 | 49.06 | 69.78 | 288 | Texans |
| 101 | 74 | Dwan Edwards | 63.67 | 45.36 | 73.01 | 703 | Panthers |
| 102 | 75 | Billy Winn | 63.55 | 53.57 | 66.03 | 702 | Browns |
| 103 | 76 | Lawrence Guy Sr. | 63.55 | 51.82 | 73.46 | 222 | Colts |
| 104 | 77 | Darnell Dockett | 63.33 | 45.32 | 71.91 | 794 | Cardinals |
| 105 | 78 | Marcus Spears | 63.22 | 49.06 | 70.67 | 384 | Cowboys |
| 106 | 79 | Tyrone Crawford | 63.21 | 53.52 | 65.50 | 296 | Cowboys |
| 107 | 80 | Rocky Bernard | 63.02 | 50.25 | 69.86 | 386 | Giants |
| 108 | 81 | Isaac Sopoaga | 62.61 | 47.28 | 68.67 | 393 | 49ers |
| 109 | 82 | Kendall Langford | 62.42 | 52.30 | 65.00 | 747 | Rams |
| 110 | 83 | Nate Collins | 62.17 | 62.26 | 65.25 | 241 | Bears |
| 111 | 84 | Chris Baker | 62.06 | 51.04 | 71.75 | 211 | Commanders |
| 112 | 85 | Sen'Derrick Marks | 62.02 | 50.75 | 67.25 | 678 | Titans |

### Rotation/backup (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 113 | 1 | Justin Bannan | 61.99 | 52.06 | 64.44 | 574 | Broncos |
| 114 | 2 | Tony McDaniel | 61.96 | 48.20 | 71.03 | 248 | Dolphins |
| 115 | 3 | Ishmaa'ily Kitchen | 61.93 | 51.66 | 65.65 | 208 | Browns |
| 116 | 4 | Clinton McDonald | 60.97 | 52.93 | 64.14 | 338 | Seahawks |
| 117 | 5 | Jermelle Cudjo | 60.85 | 49.28 | 67.40 | 341 | Rams |
| 118 | 6 | Peria Jerry | 60.71 | 48.23 | 64.87 | 540 | Falcons |
| 119 | 7 | Red Bryant | 60.46 | 50.28 | 64.97 | 716 | Seahawks |
| 120 | 8 | Jerrell Powe | 60.35 | 54.85 | 70.26 | 125 | Chiefs |
| 121 | 9 | Terrence Cody | 60.33 | 51.30 | 62.38 | 453 | Ravens |
| 122 | 10 | Jerel Worthy | 60.27 | 50.64 | 64.61 | 452 | Packers |
| 123 | 11 | Corey Peters | 60.21 | 45.51 | 67.93 | 504 | Falcons |
| 124 | 12 | Drake Nevis | 60.20 | 59.45 | 65.39 | 259 | Colts |
| 125 | 13 | Jarvis Jenkins | 59.71 | 52.01 | 60.67 | 582 | Commanders |
| 126 | 14 | Andre Neblett | 59.68 | 47.16 | 69.59 | 245 | Panthers |
| 127 | 15 | Roy Miller | 59.36 | 48.57 | 62.90 | 492 | Buccaneers |
| 128 | 16 | Devon Still | 59.35 | 59.88 | 63.17 | 156 | Bengals |
| 129 | 17 | DeAngelo Tyson | 59.12 | 47.72 | 65.69 | 220 | Ravens |
| 130 | 18 | Brandon Deaderick | 58.95 | 47.43 | 64.64 | 462 | Patriots |
| 131 | 19 | Andre Fluellen | 58.73 | 51.93 | 65.77 | 106 | Lions |
| 132 | 20 | Kedric Golston | 58.37 | 48.14 | 63.84 | 387 | Commanders |
| 133 | 21 | Greg Scruggs | 57.89 | 50.45 | 61.82 | 247 | Seahawks |
| 134 | 22 | Casey Hampton | 57.88 | 46.56 | 61.90 | 494 | Steelers |
| 135 | 23 | Fili Moala | 57.27 | 45.69 | 65.60 | 312 | Colts |
| 136 | 24 | Mitch Unrein | 56.73 | 48.74 | 58.27 | 405 | Broncos |
| 137 | 25 | Antonio Johnson | 56.73 | 45.23 | 61.17 | 483 | Colts |
| 138 | 26 | Ricardo Mathews | 56.58 | 50.15 | 60.03 | 494 | Colts |
| 139 | 27 | Vaughn Martin | 56.44 | 45.83 | 62.68 | 460 | Chargers |
| 140 | 28 | Ma'ake Kemoeatu | 56.21 | 41.48 | 62.64 | 541 | Ravens |
| 141 | 29 | Allen Bailey | 55.77 | 50.37 | 60.80 | 164 | Chiefs |
| 142 | 30 | Anthony Toribio | 55.21 | 50.13 | 61.98 | 135 | Chiefs |
| 143 | 31 | Christo Bilukidi | 55.08 | 51.78 | 56.24 | 242 | Raiders |
| 144 | 32 | Marvin Austin | 54.54 | 56.13 | 57.64 | 102 | Giants |
| 145 | 33 | Frank Kearse | 54.16 | 51.87 | 60.63 | 157 | Panthers |
| 146 | 34 | Ron Edwards | 53.66 | 43.11 | 59.78 | 315 | Panthers |
| 147 | 35 | Malik Jackson | 51.56 | 46.57 | 52.80 | 116 | Broncos |
| 148 | 36 | Sione Fua | 51.34 | 43.46 | 56.32 | 252 | Panthers |
| 149 | 37 | Nick Eason | 50.77 | 37.03 | 55.77 | 218 | Cardinals |
| 150 | 38 | Martin Tevaseu | 48.97 | 51.34 | 50.82 | 228 | Colts |
| 151 | 39 | Kheeston Randall | 48.48 | 45.97 | 50.15 | 145 | Dolphins |
| 152 | 40 | Markus Kuhn | 45.00 | 42.74 | 47.52 | 169 | Giants |

## ED — Edge

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 96.41 | 97.39 | 91.59 | 1037 | Broncos |
| 2 | 2 | Charles Johnson | 89.18 | 88.24 | 85.96 | 833 | Panthers |
| 3 | 3 | Cameron Wake | 88.79 | 84.72 | 87.33 | 914 | Dolphins |
| 4 | 4 | DeMarcus Ware | 88.02 | 80.77 | 88.68 | 868 | Cowboys |
| 5 | 5 | Brandon Graham | 85.79 | 91.94 | 82.21 | 421 | Eagles |
| 6 | 6 | Carlos Dunlap | 84.35 | 86.86 | 80.80 | 640 | Bengals |
| 7 | 7 | Mario Williams | 84.15 | 88.26 | 81.31 | 929 | Bills |
| 8 | 8 | Jason Babin | 83.45 | 73.32 | 86.04 | 653 | Jaguars |
| 9 | 9 | Chris Long | 82.49 | 78.44 | 81.03 | 876 | Rams |
| 10 | 10 | Jared Allen | 81.62 | 76.28 | 81.02 | 1117 | Vikings |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Anthony Spencer | 79.97 | 80.78 | 76.29 | 847 | Cowboys |
| 12 | 2 | Justin Houston | 79.77 | 69.91 | 82.17 | 993 | Chiefs |
| 13 | 3 | Jason Pierre-Paul | 79.63 | 86.67 | 70.77 | 875 | Giants |
| 14 | 4 | Julius Peppers | 78.67 | 75.91 | 76.34 | 783 | Bears |
| 15 | 5 | Chris Clemons | 77.05 | 67.56 | 79.21 | 893 | Seahawks |
| 16 | 6 | Cliff Avril | 76.72 | 69.02 | 78.32 | 686 | Lions |
| 17 | 7 | Greg Hardy | 76.19 | 77.04 | 72.19 | 747 | Panthers |
| 18 | 8 | Terrell Suggs | 76.08 | 71.13 | 77.29 | 676 | Ravens |
| 19 | 9 | Bruce Irvin | 74.30 | 61.31 | 78.80 | 514 | Seahawks |

### Starter (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Ryan Kerrigan | 73.24 | 63.23 | 75.75 | 1135 | Commanders |
| 21 | 2 | Everson Griffen | 73.04 | 71.76 | 71.19 | 656 | Vikings |
| 22 | 3 | Michael Bennett | 72.68 | 78.21 | 66.07 | 954 | Buccaneers |
| 23 | 4 | Juqua Parker | 71.35 | 64.38 | 73.49 | 528 | Browns |
| 24 | 5 | Phillip Hunt | 71.24 | 64.36 | 76.99 | 148 | Eagles |
| 25 | 6 | Melvin Ingram III | 70.95 | 71.78 | 66.23 | 459 | Chargers |
| 26 | 7 | Lawrence Jackson | 70.50 | 66.27 | 71.97 | 384 | Lions |
| 27 | 8 | Justin Tuck | 70.09 | 61.93 | 71.88 | 645 | Giants |
| 28 | 9 | Chandler Jones | 69.84 | 74.81 | 62.36 | 765 | Patriots |
| 29 | 10 | Mark Anderson | 69.16 | 57.17 | 78.92 | 244 | Bills |
| 30 | 11 | Wallace Gilberry | 68.88 | 53.81 | 75.27 | 346 | Bengals |
| 31 | 12 | Jabaal Sheard | 68.12 | 65.78 | 65.52 | 986 | Browns |
| 32 | 13 | Robert Quinn | 67.78 | 62.96 | 67.21 | 824 | Rams |
| 33 | 14 | William Hayes | 67.46 | 67.36 | 65.65 | 368 | Rams |
| 34 | 15 | Shaun Phillips | 67.11 | 54.09 | 72.88 | 835 | Chargers |
| 35 | 16 | Michael Johnson | 67.10 | 65.96 | 63.69 | 905 | Bengals |
| 36 | 17 | Brian Robison | 67.04 | 62.08 | 66.18 | 887 | Vikings |
| 37 | 18 | Israel Idonije | 66.93 | 62.76 | 65.54 | 712 | Bears |
| 38 | 19 | Brooks Reed | 66.55 | 64.46 | 65.08 | 672 | Texans |
| 39 | 20 | Darryl Tapp | 66.32 | 64.08 | 66.67 | 247 | Eagles |
| 40 | 21 | Cameron Jordan | 65.92 | 70.65 | 58.60 | 1038 | Saints |
| 41 | 22 | Mathias Kiwanuka | 65.47 | 55.22 | 70.83 | 526 | Giants |
| 42 | 23 | O'Brien Schofield | 64.38 | 56.14 | 70.80 | 490 | Cardinals |
| 43 | 24 | Jerry Hughes | 63.62 | 60.65 | 64.45 | 612 | Colts |
| 44 | 25 | Robert Ayers | 63.38 | 64.57 | 59.46 | 337 | Broncos |
| 45 | 26 | Jeremy Mincey | 63.32 | 60.01 | 61.56 | 954 | Jaguars |
| 46 | 27 | Matt Shaughnessy | 63.29 | 61.57 | 64.34 | 667 | Raiders |
| 47 | 28 | Derrick Shelby | 63.12 | 60.69 | 60.57 | 219 | Dolphins |
| 48 | 29 | Kroy Biermann | 62.81 | 60.80 | 59.98 | 762 | Falcons |
| 49 | 30 | Olivier Vernon | 62.13 | 64.33 | 56.49 | 431 | Dolphins |
| 50 | 31 | Frank Alexander | 62.12 | 60.09 | 59.31 | 551 | Panthers |

### Rotation/backup (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 51 | 1 | Jermaine Cunningham | 61.79 | 57.32 | 61.64 | 476 | Patriots |
| 52 | 2 | Antwan Applewhite | 61.65 | 55.64 | 68.68 | 100 | Panthers |
| 53 | 3 | Frostee Rucker | 61.35 | 56.38 | 61.96 | 594 | Browns |
| 54 | 4 | Austen Lane | 61.12 | 63.81 | 61.93 | 372 | Jaguars |
| 55 | 5 | Turk McBride | 60.97 | 58.82 | 65.63 | 140 | Saints |
| 56 | 6 | Adrian Clayborn | 60.62 | 58.93 | 66.04 | 176 | Buccaneers |
| 57 | 7 | Justin Francis | 59.89 | 58.82 | 61.63 | 295 | Patriots |
| 58 | 8 | Chris Kelsay | 57.89 | 49.83 | 64.00 | 289 | Bills |
| 59 | 9 | Cliff Matthews | 57.34 | 52.69 | 60.83 | 138 | Falcons |
| 60 | 10 | Kyle Vanden Bosch | 57.07 | 45.98 | 61.33 | 641 | Lions |
| 61 | 11 | Trevor Scott | 56.68 | 58.43 | 53.94 | 286 | Patriots |
| 62 | 12 | Robert Geathers | 56.25 | 52.99 | 54.58 | 676 | Bengals |
| 63 | 13 | Daniel Te'o-Nesheim | 55.75 | 56.85 | 55.01 | 728 | Buccaneers |
| 64 | 14 | Emmanuel Stephens | 55.55 | 62.27 | 54.45 | 146 | Browns |
| 65 | 15 | Kyle Moore | 55.46 | 58.68 | 56.84 | 491 | Bills |
| 66 | 16 | Andre Branch | 55.39 | 58.92 | 52.00 | 412 | Jaguars |
| 67 | 17 | Andre Carter | 55.38 | 48.78 | 58.22 | 316 | Raiders |
| 68 | 18 | George Selvie | 54.79 | 51.79 | 56.27 | 237 | Jaguars |
| 69 | 19 | Eugene Sims | 53.98 | 54.72 | 53.29 | 399 | Rams |
| 70 | 20 | Dave Tollefson | 53.41 | 47.62 | 55.18 | 198 | Raiders |
| 71 | 21 | Aaron Morgan | 53.26 | 60.08 | 53.02 | 124 | Buccaneers |
| 72 | 22 | Scott Solomon | 52.16 | 56.77 | 48.05 | 168 | Titans |
| 73 | 23 | George Johnson | 45.00 | 47.76 | 46.38 | 116 | Vikings |

## G — Guard

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Evan Mathis | 97.50 | 92.80 | 96.47 | 1124 | Eagles |
| 2 | 2 | Logan Mankins | 92.10 | 84.90 | 92.73 | 895 | Patriots |
| 3 | 3 | Marshal Yanda | 91.00 | 84.20 | 91.36 | 1194 | Ravens |
| 4 | 4 | Mike Iupati | 89.02 | 81.80 | 89.66 | 1173 | 49ers |
| 5 | 5 | Alex Boone | 88.68 | 82.40 | 88.70 | 1195 | 49ers |
| 6 | 6 | Jon Asamoah | 88.45 | 83.50 | 87.58 | 989 | Chiefs |
| 7 | 7 | Ben Grubbs | 88.09 | 81.60 | 88.25 | 1107 | Saints |
| 8 | 8 | Donald Thomas | 88.07 | 81.60 | 88.21 | 609 | Patriots |
| 9 | 9 | Kevin Boothe | 86.84 | 80.70 | 86.77 | 1012 | Giants |
| 10 | 10 | Brandon Moore | 86.66 | 80.20 | 86.80 | 1063 | Jets |
| 11 | 11 | John Greco | 86.53 | 77.60 | 88.31 | 691 | Browns |
| 12 | 12 | Geoff Schwartz | 86.17 | 79.90 | 86.19 | 157 | Vikings |
| 13 | 13 | Andy Levitre | 85.90 | 78.60 | 86.60 | 1011 | Bills |
| 14 | 14 | Stephen Peterman | 84.68 | 76.50 | 85.96 | 1199 | Lions |
| 15 | 15 | Josh Sitton | 84.67 | 77.40 | 85.35 | 1227 | Packers |
| 16 | 16 | Jahri Evans | 84.59 | 77.40 | 85.22 | 1105 | Saints |
| 17 | 17 | Rob Sims | 84.58 | 77.50 | 85.13 | 1199 | Lions |
| 18 | 18 | Carl Nicks | 84.41 | 77.40 | 84.91 | 442 | Buccaneers |
| 19 | 19 | Chris Chester | 84.16 | 76.70 | 84.97 | 1088 | Commanders |
| 20 | 20 | Chris Snee | 83.87 | 76.40 | 84.68 | 952 | Giants |
| 21 | 21 | Kevin Zeitler | 83.02 | 75.40 | 83.93 | 1099 | Bengals |
| 22 | 22 | Nate Livings | 82.87 | 73.70 | 84.81 | 1094 | Cowboys |
| 23 | 23 | Zane Beadles | 82.46 | 74.20 | 83.80 | 1236 | Broncos |
| 24 | 24 | Richie Incognito | 82.34 | 73.80 | 83.86 | 1027 | Dolphins |
| 25 | 25 | Wade Smith | 82.33 | 74.90 | 83.11 | 1270 | Texans |
| 26 | 26 | Vladimir Ducasse | 81.91 | 73.60 | 83.28 | 274 | Jets |
| 27 | 27 | Harvey Dahl | 81.45 | 72.60 | 83.19 | 915 | Rams |
| 28 | 28 | Kraig Urbik | 81.33 | 72.70 | 82.92 | 783 | Bills |
| 29 | 29 | Matt Slauson | 80.53 | 72.20 | 81.91 | 820 | Jets |

### Good (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Louis Vasquez | 79.88 | 72.90 | 80.37 | 1020 | Chargers |
| 31 | 2 | Jason Pinkston | 79.74 | 70.20 | 81.94 | 332 | Browns |
| 32 | 3 | Willie Colon | 79.56 | 68.50 | 82.76 | 711 | Steelers |
| 33 | 4 | Cooper Carlisle | 79.28 | 71.10 | 80.56 | 1076 | Raiders |
| 34 | 5 | Dan Connolly | 79.00 | 71.00 | 80.16 | 1039 | Patriots |
| 35 | 6 | Ramon Foster | 78.59 | 70.10 | 80.09 | 1010 | Steelers |
| 36 | 7 | Manuel Ramirez | 78.55 | 69.40 | 80.49 | 829 | Broncos |
| 37 | 8 | Ramon Harewood | 78.20 | 68.50 | 80.50 | 341 | Ravens |
| 38 | 9 | Uche Nwaneri | 78.03 | 68.50 | 80.21 | 926 | Jaguars |
| 39 | 10 | Clint Boling | 77.85 | 69.30 | 79.39 | 1104 | Bengals |
| 40 | 11 | John Moffitt | 77.73 | 68.20 | 79.91 | 461 | Seahawks |
| 41 | 12 | Jake Scott | 77.66 | 64.80 | 82.06 | 462 | Eagles |
| 42 | 13 | T.J. Lang | 77.12 | 67.70 | 79.23 | 1125 | Packers |
| 43 | 14 | Justin Blalock | 77.09 | 68.40 | 78.72 | 1189 | Falcons |
| 44 | 15 | Daryn Colledge | 76.63 | 66.50 | 79.22 | 1052 | Cardinals |
| 45 | 16 | Tyronne Green | 76.31 | 66.80 | 78.48 | 735 | Chargers |
| 46 | 17 | Chris Spencer | 76.29 | 65.80 | 79.12 | 345 | Bears |
| 47 | 18 | Russ Hochstein | 76.10 | 64.90 | 79.40 | 115 | Chiefs |
| 48 | 19 | Garry Williams | 76.05 | 66.40 | 78.32 | 606 | Panthers |
| 49 | 20 | Steve Hutchinson | 75.76 | 66.10 | 78.03 | 686 | Titans |
| 50 | 21 | Brandon Brooks | 74.79 | 66.50 | 76.15 | 173 | Texans |
| 51 | 22 | Rex Hadnot | 74.63 | 65.70 | 76.41 | 290 | Chargers |
| 52 | 23 | John Jerry | 74.52 | 64.50 | 77.03 | 1034 | Dolphins |
| 53 | 24 | Charlie Johnson | 74.23 | 63.80 | 77.02 | 1093 | Vikings |
| 54 | 25 | Shawn Lauvao | 74.05 | 62.90 | 77.31 | 1031 | Browns |
| 55 | 26 | Lance Louis | 74.03 | 63.70 | 76.75 | 692 | Bears |

### Starter (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 56 | 1 | Garrett Reynolds | 73.86 | 63.80 | 76.40 | 394 | Falcons |
| 57 | 2 | Chad Rinehart | 73.79 | 67.30 | 73.95 | 164 | Bills |
| 58 | 3 | Chris Kuper | 73.74 | 63.90 | 76.14 | 404 | Broncos |
| 59 | 4 | Brandon Fusco | 73.57 | 63.20 | 76.31 | 943 | Vikings |
| 60 | 5 | Danny Watkins | 73.53 | 62.10 | 76.98 | 448 | Eagles |
| 61 | 6 | Jeff Allen | 73.21 | 62.80 | 75.98 | 814 | Chiefs |
| 62 | 7 | Shelley Smith | 72.67 | 56.80 | 79.09 | 344 | Rams |
| 63 | 8 | Mike Brisiel | 72.48 | 60.30 | 76.43 | 961 | Raiders |
| 64 | 9 | Amini Silatolu | 72.21 | 59.50 | 76.51 | 882 | Panthers |
| 65 | 10 | David Snow | 72.06 | 66.90 | 71.34 | 133 | Bills |
| 66 | 11 | Mike McGlynn | 70.84 | 59.80 | 74.03 | 1258 | Colts |
| 67 | 12 | Joe Reitz | 70.59 | 59.70 | 73.69 | 475 | Colts |
| 68 | 13 | Chilo Rachal | 69.57 | 55.60 | 74.71 | 512 | Bears |
| 69 | 14 | James Carpenter | 68.54 | 56.10 | 72.66 | 343 | Seahawks |
| 70 | 15 | Antoine Caldwell | 67.73 | 55.50 | 71.71 | 346 | Texans |
| 71 | 16 | Adam Snyder | 67.01 | 55.20 | 70.71 | 866 | Cardinals |
| 72 | 17 | David DeCastro | 66.61 | 58.10 | 68.12 | 136 | Steelers |
| 73 | 18 | James Brown | 66.45 | 53.10 | 71.18 | 215 | Bears |
| 74 | 19 | Bobbie Williams | 66.32 | 52.80 | 71.17 | 355 | Ravens |
| 75 | 20 | Leonard Davis | 65.25 | 50.50 | 70.91 | 137 | 49ers |
| 76 | 21 | J.R. Sweezy | 64.41 | 50.80 | 69.31 | 376 | Seahawks |
| 77 | 22 | Seth Olsen | 63.88 | 52.60 | 67.23 | 288 | Colts |
| 78 | 23 | Derrick Dockery | 62.84 | 52.50 | 65.56 | 174 | Cowboys |

### Rotation/backup (0 players)

_None._

## HB — Running Back

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Adrian Peterson | 86.10 | 92.40 | 77.73 | 308 | Vikings |
| 2 | 2 | C.J. Spiller | 85.23 | 91.70 | 76.75 | 241 | Bills |
| 3 | 3 | Alfred Morris | 81.24 | 91.00 | 70.57 | 202 | Commanders |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Doug Martin | 78.85 | 79.10 | 74.52 | 334 | Buccaneers |
| 5 | 2 | Pierre Thomas | 78.30 | 82.50 | 71.33 | 220 | Saints |
| 6 | 3 | Marshawn Lynch | 78.07 | 84.40 | 69.69 | 246 | Seahawks |
| 7 | 4 | Isaac Redman | 77.17 | 75.80 | 73.91 | 112 | Steelers |
| 8 | 5 | Joique Bell | 76.98 | 72.60 | 75.73 | 245 | Lions |
| 9 | 6 | Jamaal Charles | 76.36 | 77.50 | 71.44 | 204 | Chiefs |
| 10 | 7 | LeSean McCoy | 76.22 | 77.90 | 70.94 | 280 | Eagles |
| 11 | 8 | Darren Sproles | 75.45 | 66.00 | 77.59 | 335 | Saints |
| 12 | 9 | Ahmad Bradshaw | 75.38 | 79.90 | 68.20 | 185 | Giants |

### Starter (39 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Frank Gore | 73.39 | 78.20 | 66.02 | 321 | 49ers |
| 14 | 2 | Bryce Brown | 73.25 | 65.00 | 74.58 | 147 | Eagles |
| 15 | 3 | Fred Jackson | 73.20 | 66.20 | 73.70 | 159 | Bills |
| 16 | 4 | Ryan Mathews | 73.13 | 73.30 | 68.85 | 173 | Chargers |
| 17 | 5 | Jacquizz Rodgers | 72.87 | 72.70 | 68.82 | 299 | Falcons |
| 18 | 6 | Arian Foster | 72.73 | 75.40 | 66.78 | 429 | Texans |
| 19 | 7 | DeMarco Murray | 72.40 | 70.20 | 69.70 | 205 | Cowboys |
| 20 | 8 | Jonathan Stewart | 72.30 | 69.30 | 70.14 | 121 | Panthers |
| 21 | 9 | Ray Rice | 71.20 | 73.70 | 65.37 | 410 | Ravens |
| 22 | 10 | Steven Jackson | 71.08 | 75.20 | 64.17 | 323 | Rams |
| 23 | 11 | Knowshon Moreno | 70.52 | 75.30 | 63.16 | 133 | Broncos |
| 24 | 12 | Matt Forte | 70.18 | 67.60 | 67.74 | 274 | Bears |
| 25 | 13 | DeAngelo Williams | 70.03 | 66.40 | 68.29 | 147 | Panthers |
| 26 | 14 | Maurice Jones-Drew | 69.49 | 65.80 | 67.78 | 101 | Jaguars |
| 27 | 15 | Michael Turner | 69.38 | 67.80 | 66.27 | 154 | Falcons |
| 28 | 16 | Willis McGahee | 69.10 | 70.70 | 63.87 | 157 | Broncos |
| 29 | 17 | Evan Royster | 68.94 | 56.20 | 73.26 | 161 | Commanders |
| 30 | 18 | Robert Turbin | 68.88 | 72.80 | 62.10 | 109 | Seahawks |
| 31 | 19 | Stevan Ridley | 68.78 | 70.30 | 63.60 | 199 | Patriots |
| 32 | 20 | Jonathan Dwyer | 68.11 | 69.70 | 62.88 | 143 | Steelers |
| 33 | 21 | Trent Richardson | 68.07 | 73.00 | 60.61 | 317 | Browns |
| 34 | 22 | Vick Ballard | 67.98 | 71.40 | 61.54 | 192 | Colts |
| 35 | 23 | Shane Vereen | 67.96 | 73.30 | 60.23 | 112 | Patriots |
| 36 | 24 | Reggie Bush | 67.79 | 65.20 | 65.35 | 231 | Dolphins |
| 37 | 25 | Donald Brown | 66.80 | 63.80 | 64.63 | 106 | Colts |
| 38 | 26 | Daryl Richardson | 66.49 | 56.90 | 68.72 | 148 | Rams |
| 39 | 27 | Felix Jones | 66.36 | 60.30 | 66.23 | 162 | Cowboys |
| 40 | 28 | Jason Snelling | 66.16 | 64.00 | 63.43 | 136 | Falcons |
| 41 | 29 | Chris Johnson | 65.96 | 62.30 | 64.23 | 366 | Titans |
| 42 | 30 | Ronnie Brown | 65.91 | 70.50 | 58.69 | 218 | Chargers |
| 43 | 31 | Danny Woodhead | 65.49 | 67.40 | 60.05 | 294 | Patriots |
| 44 | 32 | Shonn Greene | 65.32 | 66.30 | 60.50 | 169 | Jets |
| 45 | 33 | Kevin Smith | 65.13 | 63.40 | 62.12 | 103 | Lions |
| 46 | 34 | LaRod Stephens-Howling | 64.97 | 69.50 | 57.78 | 155 | Cardinals |
| 47 | 35 | Mikel Leshoure | 64.90 | 69.40 | 57.74 | 220 | Lions |
| 48 | 36 | Rashad Jennings | 64.42 | 59.60 | 63.47 | 119 | Jaguars |
| 49 | 37 | Darren McFadden | 64.15 | 52.70 | 67.61 | 258 | Raiders |
| 50 | 38 | Toby Gerhart | 64.10 | 55.40 | 65.73 | 127 | Vikings |
| 51 | 39 | BenJarvus Green-Ellis | 63.78 | 63.50 | 59.80 | 250 | Bengals |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 52 | 1 | Bilal Powell | 61.16 | 64.40 | 54.83 | 173 | Jets |
| 53 | 2 | Alex Green | 59.94 | 62.10 | 54.34 | 105 | Packers |
| 54 | 3 | Shaun Draughn | 58.93 | 54.30 | 57.85 | 134 | Chiefs |
| 55 | 4 | Daniel Thomas | 58.71 | 56.50 | 56.01 | 132 | Dolphins |

## LB — Linebacker

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Patrick Willis | 87.71 | 90.40 | 82.26 | 1173 | 49ers |
| 2 | 2 | NaVorro Bowman | 85.64 | 88.10 | 80.87 | 1194 | 49ers |
| 3 | 3 | Sean Lee | 84.11 | 90.60 | 81.77 | 314 | Cowboys |
| 4 | 4 | Lawrence Timmons | 83.91 | 87.70 | 77.22 | 983 | Steelers |
| 5 | 5 | Bobby Wagner | 83.68 | 84.10 | 79.24 | 964 | Seahawks |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Derrick Johnson | 78.53 | 79.70 | 73.59 | 963 | Chiefs |
| 7 | 2 | Daryl Washington | 78.31 | 82.80 | 71.46 | 1058 | Cardinals |
| 8 | 3 | Karlos Dansby | 77.35 | 76.00 | 74.50 | 1101 | Dolphins |
| 9 | 4 | D.J. Williams | 77.25 | 79.00 | 76.40 | 159 | Broncos |
| 10 | 5 | Kaluka Maiava | 76.75 | 78.60 | 72.28 | 485 | Browns |
| 11 | 6 | Brad Jones | 76.35 | 77.90 | 72.72 | 811 | Packers |
| 12 | 7 | Jerod Mayo | 75.84 | 74.70 | 72.43 | 1200 | Patriots |
| 13 | 8 | Luke Kuechly | 75.50 | 72.50 | 73.33 | 921 | Panthers |
| 14 | 9 | Rolando McClain | 74.62 | 76.50 | 72.34 | 495 | Raiders |
| 15 | 10 | D'Qwell Jackson | 74.17 | 70.60 | 72.38 | 1120 | Browns |

### Starter (72 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | DeMeco Ryans | 73.64 | 73.40 | 71.72 | 1045 | Eagles |
| 17 | 2 | Wesley Woodyard | 73.39 | 72.20 | 71.90 | 956 | Broncos |
| 18 | 3 | Paul Posluszny | 72.86 | 69.40 | 71.41 | 1126 | Jaguars |
| 19 | 4 | Lavonte David | 72.85 | 69.10 | 71.18 | 1057 | Buccaneers |
| 20 | 5 | Brendon Ayanbadejo | 72.83 | 77.80 | 72.44 | 187 | Ravens |
| 21 | 6 | Kevin Burnett | 72.81 | 71.00 | 69.85 | 1073 | Dolphins |
| 22 | 7 | Jerrell Freeman | 72.73 | 70.30 | 70.19 | 1098 | Colts |
| 23 | 8 | Brian Cushing | 72.53 | 74.70 | 73.48 | 246 | Texans |
| 24 | 9 | Lance Briggs | 72.52 | 70.00 | 70.03 | 1007 | Bears |
| 25 | 10 | Bront Bird | 72.27 | 82.80 | 75.08 | 110 | Chargers |
| 26 | 11 | Koa Misi | 72.22 | 72.50 | 68.90 | 550 | Dolphins |
| 27 | 12 | A.J. Hawk | 72.20 | 69.90 | 69.89 | 832 | Packers |
| 28 | 13 | Dont'a Hightower | 72.12 | 71.20 | 68.56 | 654 | Patriots |
| 29 | 14 | Thomas Davis Sr. | 72.10 | 72.80 | 71.21 | 1556 | Panthers |
| 30 | 15 | James Laurinaitis | 72.05 | 68.40 | 70.31 | 1075 | Rams |
| 31 | 16 | Moise Fokou | 71.60 | 70.10 | 70.51 | 394 | Colts |
| 32 | 17 | Tim Dobbins | 71.60 | 69.70 | 71.17 | 389 | Texans |
| 33 | 18 | Philip Wheeler | 71.37 | 71.40 | 68.12 | 1015 | Raiders |
| 34 | 19 | Kelvin Sheppard | 71.36 | 69.80 | 69.40 | 503 | Bills |
| 35 | 20 | Brandon Spikes | 71.11 | 67.90 | 71.27 | 843 | Patriots |
| 36 | 21 | Takeo Spikes | 70.87 | 66.70 | 69.49 | 699 | Chargers |
| 37 | 22 | Kavell Conner | 70.68 | 72.90 | 67.11 | 328 | Colts |
| 38 | 23 | Stephen Tulloch | 70.52 | 68.70 | 67.56 | 1023 | Lions |
| 39 | 24 | K.J. Wright | 70.43 | 67.00 | 68.94 | 982 | Seahawks |
| 40 | 25 | Adam Hayward | 70.35 | 70.20 | 70.76 | 148 | Buccaneers |
| 41 | 26 | Nick Barnett | 70.26 | 68.20 | 69.96 | 999 | Bills |
| 42 | 27 | Bruce Carter | 70.00 | 72.00 | 71.67 | 607 | Cowboys |
| 43 | 28 | Bryan Scott | 69.68 | 67.20 | 67.17 | 591 | Bills |
| 44 | 29 | Zach Brown | 69.50 | 66.90 | 67.06 | 742 | Titans |
| 45 | 30 | Donald Butler | 69.10 | 67.30 | 68.73 | 707 | Chargers |
| 46 | 31 | Dannell Ellerbe | 68.95 | 67.90 | 67.67 | 960 | Ravens |
| 47 | 32 | Mason Foster | 68.39 | 63.80 | 67.28 | 737 | Buccaneers |
| 48 | 33 | Jonathan Casillas | 68.38 | 66.90 | 67.66 | 250 | Saints |
| 49 | 34 | Vontaze Burfict | 68.35 | 63.20 | 67.62 | 964 | Bengals |
| 50 | 35 | Emmanuel Lamur | 68.07 | 70.00 | 69.91 | 135 | Bengals |
| 51 | 36 | Chad Greenway | 67.57 | 63.10 | 66.39 | 1192 | Vikings |
| 52 | 37 | Malcolm Smith | 67.49 | 69.00 | 70.00 | 181 | Seahawks |
| 53 | 38 | Akeem Dent | 67.43 | 67.00 | 68.63 | 557 | Falcons |
| 54 | 39 | Scott Fujita | 67.43 | 70.80 | 69.15 | 122 | Browns |
| 55 | 40 | Curtis Lofton | 67.31 | 62.50 | 66.35 | 1121 | Saints |
| 56 | 41 | Leroy Hill | 67.30 | 63.60 | 66.12 | 564 | Seahawks |
| 57 | 42 | Keith Brooking | 67.05 | 60.90 | 66.98 | 488 | Broncos |
| 58 | 43 | Brian Urlacher | 66.69 | 60.50 | 68.74 | 716 | Bears |
| 59 | 44 | Demorrio Williams | 66.66 | 66.70 | 66.01 | 357 | Chargers |
| 60 | 45 | Perry Riley | 66.19 | 66.00 | 66.83 | 1128 | Commanders |
| 61 | 46 | Geno Hayes | 66.02 | 67.90 | 68.93 | 138 | Bears |
| 62 | 47 | Josh Bynes | 65.99 | 67.40 | 68.25 | 211 | Ravens |
| 63 | 48 | Jovan Belcher | 65.95 | 63.10 | 66.28 | 332 | Chiefs |
| 64 | 49 | Tim Shaw | 65.71 | 68.20 | 68.21 | 223 | Titans |
| 65 | 50 | Bradie James | 65.55 | 59.30 | 65.55 | 752 | Texans |
| 66 | 51 | Arthur Moats | 65.48 | 64.20 | 65.82 | 121 | Bills |
| 67 | 52 | Nick Roach | 65.45 | 62.10 | 64.98 | 697 | Bears |
| 68 | 53 | David Harris | 65.02 | 59.50 | 64.53 | 1062 | Jets |
| 69 | 54 | Larry Foote | 64.99 | 59.70 | 64.35 | 967 | Steelers |
| 70 | 55 | Dan Connor | 64.89 | 61.90 | 65.74 | 340 | Cowboys |
| 71 | 56 | JoLonn Dunbar | 64.78 | 62.40 | 63.77 | 1059 | Rams |
| 72 | 57 | Russell Allen | 64.63 | 60.00 | 65.63 | 997 | Jaguars |
| 73 | 58 | Nigel Bradham | 64.57 | 61.50 | 64.53 | 395 | Bills |
| 74 | 59 | Craig Robertson | 64.11 | 61.30 | 61.81 | 612 | Browns |
| 75 | 60 | Jameel McClain | 63.60 | 60.20 | 63.26 | 738 | Ravens |
| 76 | 61 | Will Witherspoon | 63.60 | 62.10 | 63.03 | 388 | Titans |
| 77 | 62 | Sean Weatherspoon | 63.41 | 61.00 | 62.20 | 947 | Falcons |
| 78 | 63 | Bart Scott | 63.38 | 61.60 | 60.91 | 572 | Jets |
| 79 | 64 | Jacquian Williams | 63.24 | 58.60 | 66.07 | 286 | Giants |
| 80 | 65 | Akeem Ayers | 63.11 | 59.70 | 61.21 | 861 | Titans |
| 81 | 66 | Danny Trevathan | 63.10 | 57.00 | 65.08 | 239 | Broncos |
| 82 | 67 | Michael Morgan | 62.73 | 64.30 | 67.93 | 110 | Seahawks |
| 83 | 68 | Justin Durant | 62.59 | 58.50 | 63.03 | 851 | Lions |
| 84 | 69 | Spencer Paysinger | 62.52 | 64.90 | 66.02 | 134 | Giants |
| 85 | 70 | Michael Boley | 62.28 | 55.40 | 62.70 | 839 | Giants |
| 86 | 71 | Demario Davis | 62.15 | 58.00 | 61.79 | 309 | Jets |
| 87 | 72 | Akeem Jordan | 62.05 | 58.90 | 64.25 | 331 | Eagles |

### Rotation/backup (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 88 | 1 | Alex Albright | 61.87 | 58.20 | 63.40 | 179 | Cowboys |
| 89 | 2 | Brandon Siler | 61.76 | 63.10 | 66.20 | 174 | Chiefs |
| 90 | 3 | D.J. Smith | 61.67 | 64.20 | 66.61 | 369 | Packers |
| 91 | 4 | Erin Henderson | 61.37 | 55.00 | 64.59 | 743 | Vikings |
| 92 | 5 | DeAndre Levy | 60.74 | 55.40 | 62.21 | 699 | Lions |
| 93 | 6 | Julian Stanford | 60.68 | 53.80 | 62.14 | 350 | Jaguars |
| 94 | 7 | Dekoda Watson | 60.31 | 52.00 | 64.50 | 126 | Buccaneers |
| 95 | 8 | Chase Blackburn | 60.19 | 59.00 | 61.82 | 783 | Giants |
| 96 | 9 | Mychal Kendricks | 60.18 | 55.40 | 60.24 | 927 | Eagles |
| 97 | 10 | Dan Skuta | 60.14 | 55.30 | 61.15 | 114 | Bengals |
| 98 | 11 | Kyle Bosworth | 59.68 | 54.70 | 61.44 | 250 | Jaguars |
| 99 | 12 | Pat Angerer | 59.67 | 52.20 | 63.19 | 339 | Colts |
| 100 | 13 | London Fletcher | 59.63 | 53.50 | 59.55 | 1078 | Commanders |
| 101 | 14 | Jonathan Vilma | 59.57 | 57.70 | 60.20 | 406 | Saints |
| 102 | 15 | Jason Trusnik | 59.57 | 55.70 | 61.88 | 112 | Dolphins |
| 103 | 16 | Keith Rivers | 59.53 | 55.70 | 63.25 | 233 | Giants |
| 104 | 17 | James-Michael Johnson | 59.25 | 56.80 | 64.01 | 287 | Browns |
| 105 | 18 | Jason Phillips | 59.20 | 62.70 | 63.38 | 110 | Panthers |
| 106 | 19 | Rocky McIntosh | 59.16 | 53.90 | 59.97 | 446 | Rams |
| 107 | 20 | Ray Lewis | 58.95 | 48.70 | 65.36 | 771 | Ravens |
| 108 | 21 | Mike Peterson | 58.71 | 58.20 | 61.24 | 119 | Falcons |
| 109 | 22 | Ashlee Palmer | 58.55 | 54.30 | 62.81 | 116 | Lions |
| 110 | 23 | James Anderson | 58.43 | 51.70 | 60.83 | 518 | Panthers |
| 111 | 24 | Miles Burris | 58.43 | 51.70 | 58.75 | 868 | Raiders |
| 112 | 25 | Paris Lenon | 58.13 | 50.90 | 58.78 | 1014 | Cardinals |
| 113 | 26 | Quincy Black | 58.05 | 54.10 | 61.83 | 292 | Buccaneers |
| 114 | 27 | Mark Herzlich | 57.71 | 57.50 | 62.40 | 175 | Giants |
| 115 | 28 | Stephen Nicholas | 57.66 | 51.10 | 59.75 | 979 | Falcons |
| 116 | 29 | Barrett Ruud | 57.65 | 51.70 | 63.28 | 154 | Texans |
| 117 | 30 | Will Herring | 57.38 | 55.00 | 62.20 | 102 | Saints |
| 118 | 31 | Daryl Smith | 56.68 | 55.00 | 60.93 | 116 | Jaguars |
| 119 | 32 | Vincent Rey | 56.05 | 51.50 | 61.36 | 110 | Bengals |
| 120 | 33 | David Hawthorne | 55.86 | 49.00 | 59.91 | 320 | Saints |
| 121 | 34 | Rey Maualuga | 55.71 | 42.40 | 61.05 | 1087 | Bengals |
| 122 | 35 | Omar Gaither | 55.42 | 56.00 | 61.90 | 144 | Raiders |
| 123 | 36 | Scott Shanle | 55.00 | 49.60 | 59.33 | 218 | Saints |
| 124 | 37 | Ernie Sims | 54.58 | 48.20 | 58.56 | 364 | Cowboys |
| 125 | 38 | Jasper Brinkley | 54.26 | 46.90 | 60.09 | 869 | Vikings |
| 126 | 39 | Joe Mays | 53.40 | 58.90 | 64.06 | 292 | Broncos |
| 127 | 40 | Jon Beason | 53.35 | 54.20 | 59.55 | 262 | Panthers |
| 128 | 41 | Jamar Chaney | 49.89 | 43.00 | 57.09 | 231 | Eagles |
| 129 | 42 | Colin McCarthy | 48.38 | 40.20 | 57.48 | 385 | Titans |

## QB — Quarterback

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Rodgers | 87.67 | 89.88 | 81.32 | 775 | Packers |
| 2 | 2 | Peyton Manning | 85.04 | 90.57 | 80.22 | 696 | Broncos |
| 3 | 3 | Drew Brees | 83.80 | 86.62 | 76.95 | 725 | Saints |
| 4 | 4 | Tom Brady | 83.79 | 89.03 | 75.06 | 808 | Patriots |
| 5 | 5 | Matt Ryan | 83.66 | 86.41 | 76.86 | 778 | Falcons |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Ben Roethlisberger | 79.71 | 82.63 | 75.23 | 509 | Steelers |
| 7 | 2 | Eli Manning | 79.35 | 83.06 | 72.76 | 608 | Giants |
| 8 | 3 | Joe Flacco | 76.48 | 74.48 | 73.58 | 756 | Ravens |
| 9 | 4 | Tony Romo | 76.33 | 75.17 | 74.21 | 728 | Cowboys |
| 10 | 5 | Russell Wilson | 75.01 | 88.50 | 78.19 | 590 | Seahawks |
| 11 | 6 | Matt Schaub | 74.77 | 76.16 | 71.28 | 693 | Texans |
| 12 | 7 | Philip Rivers | 74.25 | 72.39 | 71.68 | 619 | Chargers |

### Starter (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Alex Smith | 72.51 | 72.08 | 75.96 | 267 | 49ers |
| 14 | 2 | Cam Newton | 72.18 | 67.56 | 72.61 | 588 | Panthers |
| 15 | 3 | Matthew Stafford | 71.46 | 73.17 | 67.02 | 801 | Lions |
| 16 | 4 | Robert Griffin III | 70.88 | 85.50 | 74.66 | 511 | Commanders |
| 17 | 5 | Carson Palmer | 69.51 | 67.65 | 68.66 | 625 | Raiders |
| 18 | 6 | Jay Cutler | 68.97 | 70.17 | 67.72 | 529 | Bears |
| 19 | 7 | Colin Kaepernick | 68.40 | 82.63 | 81.11 | 372 | 49ers |
| 20 | 8 | Josh Freeman | 68.15 | 63.49 | 67.71 | 632 | Buccaneers |
| 21 | 9 | Sam Bradford | 67.84 | 69.74 | 63.37 | 638 | Rams |
| 22 | 10 | Andy Dalton | 67.81 | 63.37 | 66.90 | 659 | Bengals |
| 23 | 11 | Andrew Luck | 66.00 | 64.70 | 64.46 | 813 | Colts |
| 24 | 12 | Michael Vick | 65.91 | 67.02 | 65.28 | 434 | Eagles |
| 25 | 13 | Matt Hasselbeck | 64.69 | 68.45 | 64.82 | 252 | Titans |
| 26 | 14 | Ryan Tannehill | 64.59 | 70.20 | 63.51 | 567 | Dolphins |
| 27 | 15 | Ryan Fitzpatrick | 63.75 | 58.55 | 64.63 | 584 | Bills |

### Rotation/backup (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Jake Locker | 61.30 | 67.49 | 63.70 | 387 | Titans |
| 29 | 2 | Kevin Kolb | 61.19 | 62.93 | 66.93 | 229 | Cardinals |
| 30 | 3 | Nick Foles | 59.25 | 57.60 | 63.86 | 313 | Eagles |
| 31 | 4 | Christian Ponder | 59.07 | 53.87 | 62.15 | 582 | Vikings |
| 32 | 5 | Chad Henne | 58.89 | 60.83 | 61.56 | 372 | Jaguars |
| 33 | 6 | Mark Sanchez | 58.86 | 54.13 | 60.43 | 530 | Jets |
| 34 | 7 | Brandon Weeden | 58.12 | 51.40 | 58.48 | 578 | Browns |
| 35 | 8 | Matt Cassel | 57.59 | 61.52 | 58.08 | 328 | Chiefs |
| 36 | 9 | Blaine Gabbert | 54.36 | 51.54 | 58.16 | 335 | Jaguars |
| 37 | 10 | Brady Quinn | 54.23 | 48.39 | 54.93 | 239 | Chiefs |
| 38 | 11 | Ryan Lindley | 51.92 | 37.00 | 54.30 | 193 | Cardinals |
| 39 | 12 | John Skelton | 48.25 | 38.47 | 55.77 | 225 | Cardinals |

## S — Safety

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jairus Byrd | 96.23 | 94.70 | 93.08 | 1022 | Bills |
| 2 | 2 | Eric Weddle | 94.75 | 91.70 | 92.62 | 1032 | Chargers |
| 3 | 3 | Devin McCourty | 91.95 | 89.80 | 89.22 | 1225 | Patriots |
| 4 | 4 | Reggie Nelson | 91.91 | 90.30 | 90.38 | 969 | Bengals |
| 5 | 5 | Harrison Smith | 90.68 | 89.60 | 87.23 | 1102 | Vikings |
| 6 | 6 | Reshad Jones | 90.06 | 91.00 | 87.97 | 1117 | Dolphins |
| 7 | 7 | Kerry Rhodes | 87.06 | 85.70 | 87.13 | 1012 | Cardinals |
| 8 | 8 | Ryan Clark | 83.15 | 79.50 | 81.93 | 882 | Steelers |
| 9 | 9 | Rafael Bush | 82.12 | 83.60 | 86.55 | 122 | Saints |
| 10 | 10 | Jim Leonhard | 81.47 | 79.10 | 80.87 | 277 | Broncos |
| 11 | 11 | George Wilson | 80.05 | 73.90 | 81.13 | 897 | Bills |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Isa Abdul-Quddus | 79.30 | 82.30 | 76.52 | 473 | Saints |
| 13 | 2 | Rashad Johnson | 78.58 | 81.50 | 77.36 | 161 | Cardinals |
| 14 | 3 | Dwight Lowery | 78.13 | 76.80 | 79.44 | 545 | Jaguars |
| 15 | 4 | Kenny Phillips | 77.71 | 72.10 | 81.97 | 293 | Giants |
| 16 | 5 | Troy Polamalu | 77.63 | 73.80 | 80.70 | 386 | Steelers |
| 17 | 6 | Chris Clemons | 76.93 | 75.00 | 78.32 | 1097 | Dolphins |
| 18 | 7 | Stevie Brown | 76.74 | 77.10 | 74.42 | 829 | Giants |
| 19 | 8 | Dashon Goldson | 74.29 | 71.20 | 72.19 | 1221 | 49ers |

### Starter (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Morgan Burnett | 73.76 | 69.20 | 75.13 | 1228 | Packers |
| 21 | 2 | Quintin Mikell | 73.70 | 63.00 | 76.67 | 1021 | Rams |
| 22 | 3 | William Moore | 73.19 | 67.90 | 74.53 | 877 | Falcons |
| 23 | 4 | Thomas DeCoud | 72.73 | 72.10 | 68.99 | 1139 | Falcons |
| 24 | 5 | Kam Chancellor | 72.45 | 66.10 | 72.84 | 1094 | Seahawks |
| 25 | 6 | Eric Smith | 72.40 | 67.60 | 73.52 | 321 | Jets |
| 26 | 7 | Chris Crocker | 72.31 | 69.30 | 73.17 | 600 | Bengals |
| 27 | 8 | Earl Thomas III | 72.04 | 70.10 | 69.17 | 1077 | Seahawks |
| 28 | 9 | T.J. Ward | 71.85 | 64.60 | 76.07 | 981 | Browns |
| 29 | 10 | Don Carey | 71.65 | 67.80 | 74.83 | 358 | Lions |
| 30 | 11 | Tyvon Branch | 71.31 | 64.40 | 72.78 | 834 | Raiders |
| 31 | 12 | Jamarca Sanford | 71.13 | 66.50 | 72.45 | 834 | Vikings |
| 32 | 13 | Abram Elam | 70.74 | 63.30 | 74.67 | 451 | Chiefs |
| 33 | 14 | Rahim Moore | 70.42 | 61.30 | 73.50 | 1120 | Broncos |
| 34 | 15 | Major Wright | 70.34 | 65.30 | 71.61 | 1028 | Bears |
| 35 | 16 | Adrian Wilson | 70.25 | 63.70 | 70.96 | 846 | Cardinals |
| 36 | 17 | Ahmad Black | 69.75 | 63.70 | 69.61 | 416 | Buccaneers |
| 37 | 18 | Glover Quin | 69.74 | 59.70 | 72.26 | 1143 | Texans |
| 38 | 19 | Eric Frampton | 69.52 | 68.60 | 72.73 | 194 | Cowboys |
| 39 | 20 | LaRon Landry | 69.49 | 66.30 | 71.41 | 1019 | Jets |
| 40 | 21 | Taylor Mays | 68.80 | 61.90 | 74.87 | 251 | Bengals |
| 41 | 22 | Jerron McMillian | 68.66 | 65.90 | 66.33 | 591 | Packers |
| 42 | 23 | Tavon Wilson | 68.62 | 60.30 | 70.00 | 487 | Patriots |
| 43 | 24 | Amari Spievey | 68.54 | 67.50 | 71.64 | 190 | Lions |
| 44 | 25 | Usama Young | 68.50 | 60.40 | 73.07 | 669 | Browns |
| 45 | 26 | Yeremiah Bell | 67.73 | 60.40 | 68.45 | 1058 | Jets |
| 46 | 27 | Jeron Johnson | 67.66 | 65.30 | 70.54 | 133 | Seahawks |
| 47 | 28 | Da'Norris Searcy | 67.39 | 58.80 | 73.64 | 272 | Bills |
| 48 | 29 | Patrick Chung | 67.36 | 60.20 | 70.79 | 534 | Patriots |
| 49 | 30 | James Ihedigbo | 67.18 | 64.90 | 65.47 | 293 | Ravens |
| 50 | 31 | Will Hill III | 66.86 | 63.80 | 72.03 | 212 | Giants |
| 51 | 32 | Haruki Nakamura | 66.83 | 59.60 | 69.37 | 591 | Panthers |
| 52 | 33 | M.D. Jennings | 65.93 | 63.60 | 69.18 | 603 | Packers |
| 53 | 34 | Craig Dahl | 65.72 | 62.90 | 63.64 | 1035 | Rams |
| 54 | 35 | Gerald Sensabaugh | 65.67 | 62.30 | 64.26 | 951 | Cowboys |
| 55 | 36 | Steve Gregory | 65.56 | 64.10 | 65.18 | 884 | Patriots |
| 56 | 37 | Chris Conte | 65.43 | 61.50 | 66.89 | 855 | Bears |
| 57 | 38 | Mike Adams | 65.43 | 53.50 | 69.42 | 1056 | Broncos |
| 58 | 39 | Donte Whitner | 65.37 | 57.70 | 66.31 | 1232 | 49ers |
| 59 | 40 | DeJon Gomes | 65.15 | 65.80 | 66.67 | 385 | Commanders |
| 60 | 41 | Troy Nolan | 65.01 | 60.00 | 68.04 | 105 | Bears |
| 61 | 42 | Will Allen | 64.94 | 61.70 | 66.69 | 423 | Steelers |
| 62 | 43 | Corey Lynch | 64.78 | 58.10 | 68.92 | 494 | Chargers |
| 63 | 44 | Charles Godfrey | 64.75 | 62.00 | 63.57 | 984 | Panthers |
| 64 | 45 | Bernard Pollard | 64.55 | 55.30 | 66.75 | 1223 | Ravens |
| 65 | 46 | Tysyn Hartman | 64.54 | 60.10 | 70.63 | 237 | Chiefs |
| 66 | 47 | Reed Doughty | 63.96 | 54.60 | 66.76 | 428 | Commanders |
| 67 | 48 | Antoine Bethea | 63.70 | 52.90 | 66.73 | 1107 | Colts |
| 68 | 49 | Dawan Landry | 63.52 | 53.20 | 66.24 | 1138 | Jaguars |
| 69 | 50 | Eric Berry | 63.50 | 57.60 | 67.95 | 993 | Chiefs |
| 70 | 51 | Sherrod Martin | 63.14 | 61.50 | 63.92 | 271 | Panthers |
| 71 | 52 | James Sanders | 62.77 | 54.90 | 69.58 | 120 | Cardinals |
| 72 | 53 | Shiloh Keo | 62.63 | 61.10 | 69.52 | 109 | Texans |
| 73 | 54 | Ryan Mundy | 62.08 | 60.80 | 62.93 | 283 | Steelers |

### Rotation/backup (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 74 | 1 | Craig Steltz | 61.83 | 60.10 | 66.00 | 109 | Bears |
| 75 | 2 | Tom Zbikowski | 61.11 | 60.00 | 64.35 | 695 | Colts |
| 76 | 3 | Danieal Manning | 60.97 | 50.00 | 64.43 | 1164 | Texans |
| 77 | 4 | Eric Hagg | 60.91 | 61.30 | 62.46 | 352 | Browns |
| 78 | 5 | Tashaun Gipson Sr. | 60.83 | 53.90 | 67.54 | 367 | Browns |
| 79 | 6 | Antrel Rolle | 60.37 | 51.00 | 62.45 | 1015 | Giants |
| 80 | 7 | Jordan Babineaux | 60.29 | 51.30 | 62.76 | 763 | Titans |
| 81 | 8 | Chris Prosinski | 59.40 | 54.50 | 60.07 | 674 | Jaguars |
| 82 | 9 | Madieu Williams | 59.26 | 54.20 | 62.31 | 1094 | Commanders |
| 83 | 10 | Jordan Pugh | 58.60 | 57.80 | 59.33 | 296 | Commanders |
| 84 | 11 | Quintin Demps | 58.55 | 52.70 | 63.24 | 347 | Texans |
| 85 | 12 | Robert Johnson | 57.61 | 53.10 | 58.54 | 279 | Titans |
| 86 | 13 | Matt Giordano | 57.46 | 50.60 | 61.10 | 802 | Raiders |
| 87 | 14 | Louis Delmas | 57.41 | 50.90 | 63.21 | 434 | Lions |
| 88 | 15 | Ricardo Silva | 57.24 | 56.00 | 62.23 | 414 | Lions |
| 89 | 16 | Erik Coleman | 56.78 | 56.50 | 61.77 | 462 | Lions |
| 90 | 17 | Danny McCray | 56.67 | 52.60 | 59.06 | 638 | Cowboys |
| 91 | 18 | Nate Allen | 56.36 | 49.70 | 58.61 | 846 | Eagles |
| 92 | 19 | Travis Daniels | 55.84 | 54.20 | 56.28 | 282 | Chiefs |
| 93 | 20 | Charlie Peprah | 55.05 | 53.70 | 57.52 | 180 | Cowboys |
| 94 | 21 | Jeromy Miles | 54.67 | 52.70 | 56.36 | 121 | Bengals |
| 95 | 22 | Chris Hope | 53.74 | 51.00 | 56.91 | 270 | Falcons |
| 96 | 23 | Anthony Walters | 52.67 | 59.70 | 57.64 | 110 | Bears |
| 97 | 24 | Colt Anderson | 51.63 | 47.20 | 58.12 | 292 | Eagles |
| 98 | 25 | Michael Griffin | 51.53 | 38.90 | 55.78 | 1115 | Titans |
| 99 | 26 | Kendrick Lewis | 51.46 | 43.60 | 56.80 | 553 | Chiefs |
| 100 | 27 | Kurt Coleman | 51.14 | 40.00 | 56.49 | 880 | Eagles |
| 101 | 28 | Barry Church | 51.03 | 49.30 | 56.66 | 103 | Cowboys |
| 102 | 29 | Mike Mitchell | 50.75 | 32.70 | 60.29 | 330 | Raiders |
| 103 | 30 | Malcolm Jenkins | 49.99 | 37.80 | 55.72 | 873 | Saints |
| 104 | 31 | Atari Bigby | 49.96 | 41.10 | 57.12 | 615 | Chargers |
| 105 | 32 | Roman Harper | 49.70 | 30.60 | 58.26 | 1097 | Saints |
| 106 | 33 | John Wendling | 48.58 | 49.30 | 56.44 | 166 | Lions |
| 107 | 34 | D.J. Campbell | 48.40 | 53.90 | 56.53 | 245 | Panthers |
| 108 | 35 | Mistral Raymond | 46.35 | 31.30 | 58.60 | 402 | Vikings |

## T — Tackle

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Joe Staley | 95.62 | 92.20 | 93.74 | 1152 | 49ers |
| 2 | 2 | Duane Brown | 94.93 | 90.10 | 93.99 | 1265 | Texans |
| 3 | 3 | Andre Smith | 88.76 | 81.20 | 89.63 | 1095 | Bengals |
| 4 | 4 | Jason Smith | 88.76 | 85.40 | 86.84 | 257 | Jets |
| 5 | 5 | Michael Roos | 88.45 | 83.30 | 87.72 | 935 | Titans |
| 6 | 6 | Will Beatty | 87.93 | 80.70 | 88.59 | 953 | Giants |
| 7 | 7 | Jared Gaither | 87.61 | 80.00 | 88.51 | 243 | Chargers |
| 8 | 8 | Trent Williams | 87.35 | 80.40 | 87.81 | 1025 | Commanders |
| 9 | 9 | Nate Solder | 87.34 | 81.00 | 87.40 | 1385 | Patriots |
| 10 | 10 | Anthony Davis | 87.32 | 80.10 | 87.96 | 1198 | 49ers |
| 11 | 11 | Riley Reiff | 87.08 | 87.80 | 82.43 | 326 | Lions |
| 12 | 12 | Jordan Gross | 87.07 | 80.90 | 87.02 | 1024 | Panthers |
| 13 | 13 | Eugene Monroe | 86.95 | 80.50 | 87.09 | 1062 | Jaguars |
| 14 | 14 | Ryan Clady | 86.88 | 81.10 | 86.57 | 1206 | Broncos |
| 15 | 15 | Russell Okung | 86.58 | 78.40 | 87.86 | 1050 | Seahawks |
| 16 | 16 | Tyson Clabo | 86.21 | 77.90 | 87.59 | 1179 | Falcons |
| 17 | 17 | Anthony Castonzo | 86.07 | 78.20 | 87.15 | 1259 | Colts |
| 18 | 18 | Joe Thomas | 86.04 | 80.10 | 85.83 | 1031 | Browns |
| 19 | 19 | Jared Veldheer | 85.91 | 79.70 | 85.89 | 1078 | Raiders |
| 20 | 20 | Matt Kalil | 85.82 | 78.60 | 86.46 | 1096 | Vikings |
| 21 | 21 | D'Brickashaw Ferguson | 85.42 | 79.00 | 85.53 | 1074 | Jets |
| 22 | 22 | Phil Loadholt | 85.10 | 74.80 | 87.80 | 1096 | Vikings |
| 23 | 23 | Donald Penn | 85.04 | 78.10 | 85.50 | 1047 | Buccaneers |
| 24 | 24 | Branden Albert | 84.99 | 78.00 | 85.48 | 705 | Chiefs |
| 25 | 25 | Sebastian Vollmer | 84.00 | 76.30 | 84.96 | 1248 | Patriots |
| 26 | 26 | Ryan Harris | 83.90 | 75.90 | 85.07 | 460 | Texans |
| 27 | 27 | Gosder Cherilus | 83.59 | 74.70 | 85.35 | 1198 | Lions |
| 28 | 28 | Chris Hairston | 83.26 | 73.90 | 85.33 | 568 | Bills |
| 29 | 29 | Andrew Whitworth | 83.01 | 75.70 | 83.72 | 1042 | Bengals |
| 30 | 30 | Eric Winston | 82.60 | 73.00 | 84.84 | 1051 | Chiefs |
| 31 | 31 | Bryant McKinnie | 82.37 | 74.50 | 83.45 | 410 | Ravens |
| 32 | 32 | Tyron Smith | 82.05 | 71.20 | 85.11 | 950 | Cowboys |
| 33 | 33 | Austin Howard | 82.04 | 73.00 | 83.90 | 1073 | Jets |
| 34 | 34 | Dennis Roland | 82.00 | 73.80 | 83.30 | 104 | Bengals |
| 35 | 35 | Sam Baker | 81.74 | 75.60 | 81.67 | 1189 | Falcons |
| 36 | 36 | David Stewart | 81.58 | 71.60 | 84.07 | 675 | Titans |
| 37 | 37 | Mitchell Schwartz | 81.10 | 72.30 | 82.80 | 1031 | Browns |
| 38 | 38 | Jermon Bushrod | 80.67 | 71.80 | 82.42 | 1103 | Saints |
| 39 | 39 | Cordy Glenn | 80.66 | 71.10 | 82.87 | 803 | Bills |
| 40 | 40 | King Dunlap | 80.63 | 72.20 | 82.08 | 818 | Eagles |
| 41 | 41 | Barry Richardson | 80.33 | 70.70 | 82.59 | 1023 | Rams |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 42 | 1 | J'Marcus Webb | 78.93 | 69.60 | 80.99 | 1046 | Bears |
| 43 | 2 | Jake Long | 78.26 | 67.70 | 81.14 | 730 | Dolphins |
| 44 | 3 | Demar Dotson | 77.95 | 67.70 | 80.62 | 986 | Buccaneers |
| 45 | 4 | Marcus Gilbert | 77.49 | 66.50 | 80.65 | 240 | Steelers |
| 46 | 5 | Marcus Cannon | 77.36 | 67.60 | 79.70 | 179 | Patriots |
| 47 | 6 | Charles Brown | 77.21 | 65.80 | 80.65 | 120 | Saints |
| 48 | 7 | Mike Adams | 76.98 | 66.30 | 79.94 | 487 | Steelers |
| 49 | 8 | Byron Stingily | 76.79 | 71.00 | 76.49 | 114 | Titans |
| 50 | 9 | Jeff Backus | 76.22 | 68.40 | 77.26 | 1064 | Lions |
| 51 | 10 | Michael Oher | 76.11 | 66.70 | 78.22 | 1340 | Ravens |
| 52 | 11 | Bobby Massie | 75.98 | 63.70 | 80.00 | 1052 | Cardinals |
| 53 | 12 | Marshall Newhouse | 75.98 | 67.60 | 77.40 | 1229 | Packers |
| 54 | 13 | Winston Justice | 75.92 | 64.40 | 79.43 | 798 | Colts |
| 55 | 14 | Breno Giacomini | 75.89 | 63.30 | 80.11 | 1144 | Seahawks |
| 56 | 15 | Sam Young | 75.79 | 63.40 | 79.88 | 333 | Bills |
| 57 | 16 | Doug Free | 75.78 | 62.80 | 80.26 | 1022 | Cowboys |
| 58 | 17 | Pat McQuistan | 75.71 | 64.60 | 78.95 | 140 | Cardinals |
| 59 | 18 | Derek Newton | 75.51 | 63.40 | 79.41 | 864 | Texans |
| 60 | 19 | Jermey Parnell | 75.14 | 65.70 | 77.26 | 261 | Cowboys |
| 61 | 20 | Sean Locklear | 74.67 | 62.80 | 78.42 | 647 | Giants |
| 62 | 21 | Bryan Bulaga | 74.62 | 62.90 | 78.26 | 577 | Packers |
| 63 | 22 | Zach Strief | 74.45 | 62.50 | 78.25 | 786 | Saints |

### Starter (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 64 | 1 | Donald Stephenson | 73.97 | 62.00 | 77.78 | 362 | Chiefs |
| 65 | 2 | David Diehl | 73.66 | 62.40 | 77.00 | 480 | Giants |
| 66 | 3 | Cameron Bradfield | 73.34 | 61.80 | 76.86 | 750 | Jaguars |
| 67 | 4 | Joe Barksdale | 73.30 | 61.10 | 77.27 | 118 | Rams |
| 68 | 5 | Khalif Barnes | 73.24 | 60.70 | 77.44 | 558 | Raiders |
| 69 | 6 | Don Barclay | 73.20 | 60.30 | 77.64 | 447 | Packers |
| 70 | 7 | Frank Omiyale | 73.16 | 58.40 | 78.83 | 114 | Seahawks |
| 71 | 8 | Tyler Polumbus | 73.15 | 61.60 | 76.69 | 992 | Commanders |
| 72 | 9 | Max Starks | 73.04 | 62.50 | 75.90 | 1065 | Steelers |
| 73 | 10 | William Robinson | 72.93 | 62.70 | 75.58 | 178 | Saints |
| 74 | 11 | Byron Bell | 72.86 | 61.20 | 76.46 | 943 | Panthers |
| 75 | 12 | Jonathan Scott | 72.15 | 59.50 | 76.41 | 334 | Bears |
| 76 | 13 | Kevin Haslam | 71.98 | 59.20 | 76.33 | 278 | Chargers |
| 77 | 14 | Wayne Hunter | 71.94 | 60.40 | 75.47 | 334 | Rams |
| 78 | 15 | Dennis Kelly | 70.70 | 57.40 | 75.40 | 685 | Eagles |
| 79 | 16 | Jonathan Martin | 70.69 | 56.90 | 75.71 | 1032 | Dolphins |
| 80 | 17 | Erik Pears | 69.92 | 55.80 | 75.17 | 378 | Bills |
| 81 | 18 | Willie Smith | 68.23 | 51.80 | 75.02 | 504 | Raiders |
| 82 | 19 | Nate Potter | 67.51 | 54.70 | 71.88 | 428 | Cardinals |
| 83 | 20 | Bradley Sowell | 66.63 | 53.50 | 71.21 | 131 | Colts |
| 84 | 21 | Jordan Black | 62.32 | 50.10 | 66.30 | 106 | Commanders |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 85 | 1 | Demetress Bell | 61.90 | 41.70 | 71.20 | 446 | Eagles |

## TE — Tight End

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 86.94 | 90.90 | 80.14 | 418 | Patriots |
| 2 | 2 | Tony Gonzalez | 82.24 | 89.20 | 73.43 | 725 | Falcons |
| 3 | 3 | Vernon Davis | 81.60 | 81.90 | 77.24 | 591 | 49ers |
| 4 | 4 | Dwayne Allen | 81.58 | 85.90 | 74.54 | 506 | Colts |
| 5 | 5 | Heath Miller | 81.15 | 87.10 | 73.02 | 591 | Steelers |
| 6 | 6 | Jimmy Graham | 80.91 | 79.20 | 77.89 | 562 | Saints |
| 7 | 7 | Jason Witten | 80.25 | 84.50 | 73.25 | 708 | Cowboys |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Jacob Tamme | 79.89 | 80.20 | 75.52 | 362 | Broncos |
| 9 | 2 | Zach Miller | 79.17 | 84.90 | 71.18 | 506 | Seahawks |
| 10 | 3 | Greg Olsen | 79.10 | 83.30 | 72.13 | 504 | Panthers |
| 11 | 4 | Joel Dreessen | 78.42 | 74.40 | 76.93 | 459 | Broncos |
| 12 | 5 | Marcedes Lewis | 78.41 | 78.90 | 73.92 | 474 | Jaguars |
| 13 | 6 | Daniel Fells | 77.95 | 74.40 | 76.15 | 124 | Patriots |
| 14 | 7 | Martellus Bennett | 77.91 | 81.30 | 71.49 | 555 | Giants |
| 15 | 8 | Fred Davis | 77.64 | 75.40 | 74.96 | 190 | Commanders |
| 16 | 9 | Michael Hoomanawanui | 77.48 | 75.30 | 74.76 | 135 | Patriots |
| 17 | 10 | Dennis Pitta | 76.81 | 78.50 | 71.51 | 534 | Ravens |
| 18 | 11 | Aaron Hernandez | 76.51 | 73.00 | 74.68 | 428 | Patriots |
| 19 | 12 | Scott Chandler | 75.83 | 73.20 | 73.41 | 456 | Bills |
| 20 | 13 | Owen Daniels | 75.54 | 74.80 | 71.87 | 585 | Texans |
| 21 | 14 | Jeff Cumberland | 75.37 | 69.50 | 75.11 | 328 | Jets |
| 22 | 15 | Anthony Fasano | 75.11 | 71.40 | 73.41 | 448 | Dolphins |
| 23 | 16 | Garrett Graham | 74.88 | 76.30 | 69.76 | 333 | Texans |
| 24 | 17 | Benjamin Watson | 74.77 | 73.10 | 71.72 | 485 | Browns |
| 25 | 18 | Brent Celek | 74.73 | 71.50 | 72.71 | 533 | Eagles |
| 26 | 19 | Matthew Mulligan | 74.69 | 78.10 | 68.25 | 136 | Rams |
| 27 | 20 | Kyle Rudolph | 74.60 | 76.00 | 69.50 | 511 | Vikings |
| 28 | 21 | Jermichael Finley | 74.50 | 67.40 | 75.07 | 547 | Packers |
| 29 | 22 | Delanie Walker | 74.47 | 67.00 | 75.28 | 334 | 49ers |

### Starter (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Steve Maneri | 73.75 | 73.70 | 69.62 | 147 | Chiefs |
| 31 | 2 | Dustin Keller | 73.65 | 67.00 | 73.91 | 235 | Jets |
| 32 | 3 | Antonio Gates | 73.49 | 65.50 | 74.65 | 555 | Chargers |
| 33 | 4 | Brandon Myers | 73.34 | 72.50 | 69.73 | 606 | Raiders |
| 34 | 5 | Luke Stocker | 72.70 | 77.30 | 65.46 | 246 | Buccaneers |
| 35 | 6 | Tony Moeaki | 72.62 | 65.70 | 73.07 | 498 | Chiefs |
| 36 | 7 | Jared Cook | 72.51 | 67.10 | 71.95 | 380 | Titans |
| 37 | 8 | Matt Spaeth | 72.29 | 77.70 | 64.51 | 138 | Bears |
| 38 | 9 | Lance Kendricks | 71.97 | 68.40 | 70.19 | 430 | Rams |
| 39 | 10 | Craig Stevens | 71.40 | 62.70 | 73.04 | 254 | Titans |
| 40 | 11 | Will Heller | 71.32 | 68.20 | 69.24 | 216 | Lions |
| 41 | 12 | Jeff King | 71.27 | 68.00 | 69.29 | 193 | Cardinals |
| 42 | 13 | Logan Paulsen | 70.40 | 70.90 | 65.90 | 311 | Commanders |
| 43 | 14 | Anthony McCoy | 70.37 | 67.40 | 68.19 | 245 | Seahawks |
| 44 | 15 | Randy McMichael | 70.24 | 67.10 | 68.17 | 133 | Chargers |
| 45 | 16 | Jordan Cameron | 69.94 | 62.20 | 70.93 | 199 | Browns |
| 46 | 17 | Charles Clay | 69.85 | 60.00 | 72.25 | 164 | Dolphins |
| 47 | 18 | Dallas Clark | 69.78 | 69.30 | 65.93 | 414 | Buccaneers |
| 48 | 19 | Jermaine Gresham | 69.45 | 65.10 | 68.19 | 635 | Bengals |
| 49 | 20 | Brandon Pettigrew | 68.98 | 60.80 | 70.27 | 525 | Lions |
| 50 | 21 | Coby Fleener | 68.69 | 66.30 | 66.12 | 312 | Colts |
| 51 | 22 | Rob Housler | 68.10 | 62.10 | 67.93 | 399 | Cardinals |
| 52 | 23 | Tony Scheffler | 67.73 | 64.50 | 65.72 | 436 | Lions |
| 53 | 24 | Tom Crabtree | 67.65 | 57.40 | 70.31 | 119 | Packers |
| 54 | 25 | Kellen Davis | 67.41 | 60.40 | 67.92 | 519 | Bears |
| 55 | 26 | Ed Dickson | 66.82 | 61.00 | 66.53 | 309 | Ravens |
| 56 | 27 | John Phillips | 65.73 | 65.30 | 61.85 | 132 | Cowboys |
| 57 | 28 | Clay Harbor | 65.43 | 61.60 | 63.82 | 209 | Eagles |
| 58 | 29 | Dave Thomas | 62.51 | 61.30 | 59.15 | 160 | Saints |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | John Carlson | 60.63 | 56.60 | 59.15 | 126 | Vikings |
| 60 | 2 | David Paulson | 59.90 | 51.70 | 61.20 | 153 | Steelers |

## WR — Wide Receiver

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Andre Johnson | 88.27 | 91.60 | 81.88 | 649 | Texans |
| 2 | 2 | Vincent Jackson | 87.86 | 89.60 | 82.53 | 615 | Buccaneers |
| 3 | 3 | Calvin Johnson | 87.57 | 89.80 | 81.91 | 796 | Lions |
| 4 | 4 | Percy Harvin | 86.70 | 90.80 | 79.80 | 258 | Vikings |
| 5 | 5 | A.J. Green | 85.56 | 88.90 | 79.17 | 647 | Bengals |
| 6 | 6 | Brandon Marshall | 85.23 | 90.40 | 77.62 | 576 | Bears |
| 7 | 7 | Demaryius Thomas | 85.18 | 85.10 | 81.06 | 654 | Broncos |
| 8 | 8 | Julio Jones | 85.08 | 86.10 | 80.24 | 676 | Falcons |
| 9 | 9 | Michael Crabtree | 85.05 | 91.30 | 76.71 | 544 | 49ers |
| 10 | 10 | Roddy White | 84.41 | 89.30 | 76.99 | 757 | Falcons |
| 11 | 11 | Anquan Boldin | 82.97 | 84.20 | 77.98 | 705 | Ravens |
| 12 | 12 | Reggie Wayne | 82.88 | 88.60 | 74.90 | 795 | Colts |
| 13 | 13 | Golden Tate | 82.68 | 82.80 | 78.43 | 470 | Seahawks |
| 14 | 14 | Domenik Hixon | 82.34 | 81.60 | 78.66 | 279 | Giants |
| 15 | 15 | Danario Alexander | 82.30 | 78.90 | 80.40 | 329 | Chargers |
| 16 | 16 | Hakeem Nicks | 82.13 | 80.10 | 79.32 | 424 | Giants |
| 17 | 17 | Steve Smith | 81.71 | 80.90 | 78.08 | 701 | Panthers |
| 18 | 18 | Dwayne Bowe | 81.64 | 81.10 | 77.83 | 419 | Chiefs |
| 19 | 19 | Malcom Floyd | 81.61 | 79.00 | 79.18 | 549 | Chargers |
| 20 | 20 | Antonio Brown | 81.60 | 79.60 | 78.76 | 449 | Steelers |
| 21 | 21 | Pierre Garcon | 81.57 | 84.40 | 75.51 | 238 | Commanders |
| 22 | 22 | Lance Moore | 81.43 | 83.20 | 76.09 | 503 | Saints |
| 23 | 23 | Joe Morgan | 81.27 | 76.00 | 80.61 | 190 | Saints |
| 24 | 24 | Cecil Shorts | 81.02 | 75.30 | 80.66 | 456 | Jaguars |
| 25 | 25 | Damaris Johnson | 80.78 | 77.70 | 78.67 | 165 | Eagles |
| 26 | 26 | Victor Cruz | 80.66 | 77.50 | 78.60 | 579 | Giants |
| 27 | 27 | Danny Amendola | 80.63 | 86.50 | 72.55 | 342 | Rams |
| 28 | 28 | Rueben Randle | 80.58 | 76.10 | 79.40 | 169 | Giants |
| 29 | 29 | Sidney Rice | 80.43 | 81.60 | 75.48 | 519 | Seahawks |
| 30 | 30 | Wes Welker | 80.27 | 82.10 | 74.89 | 751 | Patriots |
| 31 | 31 | Jordy Nelson | 80.21 | 75.50 | 79.19 | 484 | Packers |
| 32 | 32 | DeSean Jackson | 80.19 | 76.20 | 78.68 | 471 | Eagles |

### Good (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Dez Bryant | 79.89 | 76.50 | 77.99 | 681 | Cowboys |
| 34 | 2 | Torrey Smith | 79.58 | 75.60 | 78.06 | 705 | Ravens |
| 35 | 3 | Randy Moss | 79.01 | 75.20 | 77.38 | 293 | 49ers |
| 36 | 4 | Mario Manningham | 78.96 | 78.30 | 75.24 | 261 | 49ers |
| 37 | 5 | Chris Givens | 78.92 | 71.40 | 79.77 | 387 | Rams |
| 38 | 6 | Brian Hartline | 78.92 | 77.40 | 75.77 | 554 | Dolphins |
| 39 | 7 | Marques Colston | 78.69 | 76.00 | 76.31 | 649 | Saints |
| 40 | 8 | Steve Johnson | 78.58 | 78.10 | 74.74 | 568 | Bills |
| 41 | 9 | Mike Williams | 78.42 | 78.80 | 74.00 | 597 | Buccaneers |
| 42 | 10 | Miles Austin | 78.24 | 76.10 | 75.50 | 601 | Cowboys |
| 43 | 11 | Randall Cobb | 78.24 | 79.20 | 73.43 | 479 | Packers |
| 44 | 12 | Leonard Hankerson | 77.96 | 75.00 | 75.76 | 328 | Commanders |
| 45 | 13 | Greg Jennings | 77.92 | 72.80 | 77.17 | 377 | Packers |
| 46 | 14 | Brandon Lloyd | 77.66 | 77.00 | 73.93 | 732 | Patriots |
| 47 | 15 | Jason Avant | 77.66 | 78.60 | 72.87 | 459 | Eagles |
| 48 | 16 | Josh Gordon | 77.41 | 69.50 | 78.51 | 523 | Browns |
| 49 | 17 | Aldrick Robinson | 77.22 | 69.30 | 78.34 | 108 | Commanders |
| 50 | 18 | Ryan Broyles | 76.80 | 71.80 | 75.96 | 191 | Lions |
| 51 | 19 | Brandon Stokley | 76.68 | 76.20 | 72.83 | 447 | Broncos |
| 52 | 20 | Mike Wallace | 76.44 | 67.30 | 78.36 | 571 | Steelers |
| 53 | 21 | Brandon LaFell | 76.42 | 70.00 | 76.53 | 470 | Panthers |
| 54 | 22 | Doug Baldwin | 76.28 | 72.60 | 74.57 | 303 | Seahawks |
| 55 | 23 | Jacoby Jones | 76.01 | 72.50 | 74.18 | 375 | Ravens |
| 56 | 24 | Larry Fitzgerald | 75.97 | 69.40 | 76.18 | 701 | Cardinals |
| 57 | 25 | James Jones | 75.91 | 73.20 | 73.55 | 744 | Packers |
| 58 | 26 | Jeremy Kerley | 75.87 | 67.50 | 77.28 | 466 | Jets |
| 59 | 27 | Earl Bennett | 75.76 | 67.30 | 77.24 | 295 | Bears |
| 60 | 28 | Eric Decker | 75.49 | 76.60 | 70.58 | 676 | Broncos |
| 61 | 29 | Dwayne Harris | 75.35 | 72.00 | 73.42 | 197 | Cowboys |
| 62 | 30 | Denarius Moore | 75.28 | 68.60 | 75.56 | 520 | Raiders |
| 63 | 31 | Emmanuel Sanders | 75.12 | 69.10 | 74.97 | 440 | Steelers |
| 64 | 32 | Santonio Holmes | 75.05 | 71.30 | 73.38 | 135 | Jets |
| 65 | 33 | Davone Bess | 75.03 | 74.40 | 71.28 | 456 | Dolphins |
| 66 | 34 | Brandon Gibson | 74.73 | 76.90 | 69.11 | 500 | Rams |
| 67 | 35 | Mohamed Sanu | 74.57 | 71.70 | 72.31 | 115 | Bengals |
| 68 | 36 | Darrius Heyward-Bey | 74.34 | 68.90 | 73.80 | 536 | Raiders |
| 69 | 37 | T.Y. Hilton | 74.25 | 65.50 | 75.92 | 557 | Colts |
| 70 | 38 | Brandon Tate | 74.23 | 62.60 | 77.82 | 179 | Bengals |

### Starter (64 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 71 | 1 | Santana Moss | 73.79 | 68.60 | 73.08 | 350 | Commanders |
| 72 | 2 | Ramses Barden | 73.71 | 64.80 | 75.48 | 155 | Giants |
| 73 | 3 | Jarius Wright | 73.69 | 65.40 | 75.05 | 161 | Vikings |
| 74 | 4 | Travis Benjamin | 73.66 | 65.80 | 74.74 | 223 | Browns |
| 75 | 5 | Alshon Jeffery | 73.59 | 66.00 | 74.48 | 282 | Bears |
| 76 | 6 | Nate Washington | 73.52 | 64.40 | 75.43 | 563 | Titans |
| 77 | 7 | Rod Streater | 72.95 | 67.40 | 72.48 | 405 | Raiders |
| 78 | 8 | Juron Criner | 72.82 | 72.40 | 68.94 | 128 | Raiders |
| 79 | 9 | Devin Aromashodu | 72.72 | 64.00 | 74.37 | 234 | Vikings |
| 80 | 10 | Kenny Britt | 72.43 | 62.50 | 74.88 | 415 | Titans |
| 81 | 11 | Lestar Jean | 72.33 | 61.60 | 75.32 | 103 | Texans |
| 82 | 12 | Andrew Hawkins | 72.27 | 67.60 | 71.22 | 430 | Bengals |
| 83 | 13 | Justin Blackmon | 72.23 | 65.70 | 72.41 | 683 | Jaguars |
| 84 | 14 | Michael Jenkins | 71.93 | 65.90 | 71.79 | 487 | Vikings |
| 85 | 15 | Josh Morgan | 71.64 | 68.20 | 69.77 | 414 | Commanders |
| 86 | 16 | Mohamed Massaquoi | 71.41 | 65.90 | 70.92 | 170 | Browns |
| 87 | 17 | Derek Hagan | 70.90 | 68.50 | 68.34 | 180 | Raiders |
| 88 | 18 | Kendall Wright | 70.82 | 67.70 | 68.73 | 417 | Titans |
| 89 | 19 | Kevin Ogletree | 70.76 | 64.10 | 71.03 | 361 | Cowboys |
| 90 | 20 | Jeremy Maclin | 70.68 | 62.10 | 72.24 | 661 | Eagles |
| 91 | 21 | Matt Willis | 70.66 | 65.50 | 69.93 | 111 | Broncos |
| 92 | 22 | Rishard Matthews | 70.54 | 60.10 | 73.34 | 160 | Dolphins |
| 93 | 23 | Louis Murphy Jr. | 70.50 | 60.60 | 72.93 | 420 | Panthers |
| 94 | 24 | LaVon Brazill | 70.32 | 57.60 | 74.63 | 185 | Colts |
| 95 | 25 | Damian Williams | 70.21 | 63.70 | 70.38 | 260 | Titans |
| 96 | 26 | Kyle Williams | 70.16 | 60.30 | 72.56 | 153 | 49ers |
| 97 | 27 | Tiquan Underwood | 70.05 | 62.80 | 70.71 | 343 | Buccaneers |
| 98 | 28 | Jonathan Baldwin | 70.04 | 61.20 | 71.77 | 312 | Chiefs |
| 99 | 29 | Eddie Royal | 69.94 | 65.70 | 68.60 | 204 | Chargers |
| 100 | 30 | Michael Floyd | 69.87 | 63.10 | 70.22 | 432 | Cardinals |
| 101 | 31 | Jordan Shipley | 69.76 | 63.10 | 70.03 | 197 | Jaguars |
| 102 | 32 | Jerricho Cotchery | 69.53 | 65.00 | 68.38 | 156 | Steelers |
| 103 | 33 | Kris Durham | 69.46 | 61.40 | 70.66 | 153 | Lions |
| 104 | 34 | Jerome Simpson | 69.34 | 62.30 | 69.86 | 299 | Vikings |
| 105 | 35 | Marlon Moore | 69.24 | 58.50 | 72.24 | 100 | Dolphins |
| 106 | 36 | Kevin Walter | 69.20 | 63.10 | 69.10 | 555 | Texans |
| 107 | 37 | Tandon Doss | 69.20 | 57.20 | 73.03 | 167 | Ravens |
| 108 | 38 | Austin Pettis | 69.15 | 69.10 | 65.01 | 282 | Rams |
| 109 | 39 | Julian Edelman | 68.92 | 62.90 | 68.77 | 191 | Patriots |
| 110 | 40 | Brian Quick | 68.58 | 59.20 | 70.66 | 124 | Rams |
| 111 | 41 | Brad Smith | 68.47 | 64.90 | 66.68 | 179 | Bills |
| 112 | 42 | Greg Little | 68.37 | 65.30 | 66.25 | 576 | Browns |
| 113 | 43 | Donald Jones | 68.29 | 65.40 | 66.05 | 410 | Bills |
| 114 | 44 | Nate Burleson | 68.26 | 63.40 | 67.33 | 248 | Lions |
| 115 | 45 | Robert Meachem | 67.90 | 55.70 | 71.87 | 261 | Chargers |
| 116 | 46 | Riley Cooper | 67.85 | 60.60 | 68.52 | 335 | Eagles |
| 117 | 47 | Deion Branch | 67.81 | 53.50 | 73.18 | 353 | Patriots |
| 118 | 48 | Chaz Schilens | 67.57 | 63.10 | 66.39 | 253 | Jets |
| 119 | 49 | Andre Roberts | 67.57 | 61.40 | 67.51 | 623 | Cardinals |
| 120 | 50 | Armon Binns | 66.96 | 56.80 | 69.57 | 265 | Dolphins |
| 121 | 51 | Harry Douglas | 66.49 | 58.60 | 67.59 | 497 | Falcons |
| 122 | 52 | Marvin Jones Jr. | 66.45 | 62.10 | 65.19 | 266 | Bengals |
| 123 | 53 | Steve Breaston | 66.38 | 51.30 | 72.27 | 159 | Chiefs |
| 124 | 54 | Donnie Avery | 66.32 | 60.10 | 66.30 | 750 | Colts |
| 125 | 55 | Cole Beasley | 65.79 | 62.80 | 63.61 | 118 | Cowboys |
| 126 | 56 | Titus Young | 65.76 | 58.60 | 66.36 | 422 | Lions |
| 127 | 57 | Devin Hester | 65.26 | 55.10 | 67.87 | 262 | Bears |
| 128 | 58 | Micheal Spurlock | 65.12 | 62.40 | 62.76 | 164 | Chargers |
| 129 | 59 | DeVier Posey | 64.91 | 55.50 | 67.01 | 144 | Texans |
| 130 | 60 | Devery Henderson | 64.72 | 50.50 | 70.03 | 475 | Saints |
| 131 | 61 | Laurent Robinson | 64.29 | 57.60 | 64.58 | 199 | Jaguars |
| 132 | 62 | Donald Driver | 63.25 | 53.30 | 65.71 | 117 | Packers |
| 133 | 63 | Stephen Hill | 62.49 | 56.60 | 62.25 | 283 | Jets |
| 134 | 64 | Kevin Elliott | 62.16 | 60.00 | 59.43 | 160 | Bills |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 135 | 1 | T.J. Graham | 61.55 | 52.40 | 63.48 | 429 | Bills |
| 136 | 2 | Mike Thomas | 60.29 | 47.40 | 64.72 | 311 | Lions |
| 137 | 3 | Early Doucet | 60.23 | 47.10 | 64.82 | 319 | Cardinals |
| 138 | 4 | Keshawn Martin | 57.88 | 51.90 | 57.70 | 168 | Texans |
