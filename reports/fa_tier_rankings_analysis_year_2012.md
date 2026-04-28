# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:38:28Z
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
| 10 | 10 | Mike Pouncey | 86.56 | 78.50 | 87.77 | 1032 | Dolphins |
| 11 | 11 | Jonathan Goodwin | 86.05 | 78.80 | 86.71 | 1171 | 49ers |
| 12 | 12 | Will Montgomery | 85.62 | 77.70 | 86.73 | 1082 | Commanders |
| 13 | 13 | Maurkice Pouncey | 83.81 | 75.16 | 85.41 | 930 | Steelers |
| 14 | 14 | Jason Kelce | 83.32 | 68.08 | 89.32 | 136 | Eagles |
| 15 | 15 | Todd McClure | 82.36 | 73.70 | 83.96 | 1175 | Falcons |
| 16 | 16 | David Baas | 82.36 | 73.40 | 84.17 | 1003 | Giants |
| 17 | 17 | Phil Costa | 81.88 | 67.28 | 87.44 | 121 | Cowboys |
| 18 | 18 | J.D. Walton | 81.79 | 67.73 | 87.00 | 248 | Broncos |
| 19 | 19 | Dominic Raiola | 81.67 | 73.10 | 83.21 | 1199 | Lions |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Robert Turner | 79.71 | 70.00 | 82.02 | 1044 | Rams |
| 21 | 2 | Ryan Cook | 79.06 | 67.99 | 82.28 | 813 | Cowboys |
| 22 | 3 | Samson Satele | 78.69 | 67.35 | 82.08 | 720 | Colts |
| 23 | 4 | Evan Smith | 78.68 | 67.26 | 82.13 | 588 | Packers |
| 24 | 5 | Eric Wood | 77.85 | 68.20 | 80.12 | 872 | Bills |
| 25 | 6 | Trevor Robinson | 77.42 | 66.41 | 80.59 | 448 | Bengals |
| 26 | 7 | Dan Koppen | 76.45 | 66.59 | 78.86 | 981 | Broncos |
| 27 | 8 | A.Q. Shipley | 76.22 | 68.27 | 77.35 | 468 | Colts |
| 28 | 9 | Nick Hardwick | 75.99 | 65.40 | 78.88 | 1019 | Chargers |
| 29 | 10 | Rodney Hudson | 75.85 | 67.33 | 77.36 | 183 | Chiefs |
| 30 | 11 | Roberto Garza | 75.80 | 64.80 | 78.96 | 1046 | Bears |
| 31 | 12 | Lyle Sendlein | 75.00 | 64.99 | 77.50 | 756 | Cardinals |
| 32 | 13 | Scott Wells | 74.92 | 63.01 | 78.69 | 421 | Rams |

### Starter (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Brad Meester | 73.43 | 63.20 | 76.09 | 1062 | Jaguars |
| 34 | 2 | Jeff Saturday | 72.49 | 62.48 | 74.99 | 962 | Packers |
| 35 | 3 | Ryan Kalil | 71.53 | 59.28 | 75.53 | 286 | Panthers |
| 36 | 4 | Steve Vallos | 71.21 | 60.92 | 73.91 | 124 | Jaguars |
| 37 | 5 | Kyle Cook | 70.62 | 58.45 | 74.57 | 205 | Bengals |
| 38 | 6 | Doug Legursky | 69.56 | 58.00 | 73.10 | 408 | Steelers |
| 39 | 7 | Alex Parsons | 63.57 | 56.76 | 63.94 | 120 | Raiders |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Casey Hayward Jr. | 95.75 | 90.09 | 95.35 | 748 | Packers |
| 2 | 2 | Richard Sherman | 95.29 | 92.50 | 93.77 | 1067 | Seahawks |
| 3 | 3 | Charles Tillman | 93.13 | 91.90 | 89.79 | 921 | Bears |
| 4 | 4 | Chris Harris Jr. | 90.08 | 90.00 | 86.75 | 983 | Broncos |
| 5 | 5 | Sam Shields | 89.83 | 88.59 | 88.58 | 720 | Packers |
| 6 | 6 | Tarell Brown | 88.94 | 84.50 | 88.36 | 1224 | 49ers |
| 7 | 7 | Brandon Flowers | 87.49 | 85.70 | 85.03 | 864 | Chiefs |
| 8 | 8 | Kareem Jackson | 85.37 | 80.10 | 84.71 | 1121 | Texans |
| 9 | 9 | Asante Samuel | 83.82 | 80.10 | 83.59 | 924 | Falcons |
| 10 | 10 | Robert McClain | 83.65 | 79.00 | 82.58 | 663 | Falcons |
| 11 | 11 | Phillip Adams | 83.61 | 74.14 | 89.83 | 175 | Raiders |
| 12 | 12 | Cortez Allen | 83.57 | 80.04 | 85.14 | 543 | Steelers |
| 13 | 13 | Leon Hall | 82.82 | 80.10 | 83.17 | 960 | Bengals |
| 14 | 14 | Greg Toler | 82.78 | 75.61 | 88.07 | 302 | Cardinals |
| 15 | 15 | Tim Jennings | 82.71 | 78.90 | 82.11 | 886 | Bears |
| 16 | 16 | Patrick Peterson | 81.70 | 77.80 | 80.14 | 1042 | Cardinals |
| 17 | 17 | Joe Haden | 81.34 | 74.72 | 84.50 | 774 | Browns |
| 18 | 18 | Joselio Hanson | 81.32 | 76.93 | 80.59 | 553 | Raiders |
| 19 | 19 | Antonio Cromartie | 81.29 | 75.10 | 81.25 | 1036 | Jets |
| 20 | 20 | E.J. Biggers | 80.66 | 78.37 | 79.59 | 796 | Buccaneers |
| 21 | 21 | Jason McCourty | 80.21 | 74.00 | 81.33 | 1126 | Titans |
| 22 | 22 | Terence Newman | 80.16 | 73.80 | 80.86 | 944 | Bengals |
| 23 | 23 | Adam Jones | 80.09 | 77.50 | 82.44 | 613 | Bengals |
| 24 | 24 | Sheldon Brown | 80.00 | 72.30 | 81.48 | 879 | Browns |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 25 | 1 | Brandon Browner | 79.58 | 72.50 | 81.43 | 861 | Seahawks |
| 26 | 2 | Trumaine Johnson | 79.56 | 71.28 | 81.94 | 355 | Rams |
| 27 | 3 | Alterraun Verner | 78.70 | 73.80 | 77.80 | 1045 | Titans |
| 28 | 4 | Vontae Davis | 78.46 | 74.42 | 80.83 | 645 | Colts |
| 29 | 5 | Tramon Williams | 78.43 | 68.90 | 80.61 | 1209 | Packers |
| 30 | 6 | Champ Bailey | 78.40 | 68.30 | 81.48 | 1090 | Broncos |
| 31 | 7 | Alfonzo Dennard | 78.32 | 74.62 | 80.79 | 734 | Patriots |
| 32 | 8 | Jabari Greer | 77.41 | 71.37 | 78.52 | 818 | Saints |
| 33 | 9 | Johnathan Joseph | 77.36 | 70.30 | 78.74 | 940 | Texans |
| 34 | 10 | Cortland Finnegan | 77.25 | 71.40 | 76.98 | 1005 | Rams |
| 35 | 11 | Josh Wilson | 77.18 | 69.80 | 78.13 | 1097 | Commanders |
| 36 | 12 | Antoine Winfield | 76.62 | 74.10 | 77.57 | 1076 | Vikings |
| 37 | 13 | Tony Carter | 76.60 | 69.72 | 78.06 | 541 | Broncos |
| 38 | 14 | Chris Owens | 76.54 | 68.40 | 82.58 | 170 | Falcons |
| 39 | 15 | Morris Claiborne | 75.88 | 69.00 | 77.33 | 879 | Cowboys |
| 40 | 16 | Chris Culliver | 75.87 | 67.67 | 77.56 | 798 | 49ers |
| 41 | 17 | Chris Houston | 75.81 | 69.10 | 77.82 | 899 | Lions |
| 42 | 18 | Brandon Boykin | 75.67 | 68.09 | 76.56 | 509 | Eagles |
| 43 | 19 | DeAngelo Hall | 75.19 | 66.90 | 76.55 | 1114 | Commanders |
| 44 | 20 | Javier Arenas | 75.11 | 66.31 | 77.13 | 710 | Chiefs |
| 45 | 21 | Carlos Rogers | 74.99 | 66.50 | 77.31 | 1230 | 49ers |
| 46 | 22 | Captain Munnerlyn | 74.49 | 67.40 | 75.69 | 913 | Panthers |
| 47 | 23 | Leonard Johnson | 74.12 | 68.73 | 77.71 | 581 | Buccaneers |
| 48 | 24 | D.J. Moore | 74.11 | 66.57 | 77.98 | 363 | Bears |

### Starter (61 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 49 | 1 | Brandon Carr | 73.95 | 64.50 | 76.08 | 1010 | Cowboys |
| 50 | 2 | Bradley Fletcher | 73.85 | 66.57 | 78.28 | 363 | Rams |
| 51 | 3 | Davon House | 73.58 | 67.61 | 78.60 | 311 | Packers |
| 52 | 4 | Dominique Rodgers-Cromartie | 73.58 | 64.90 | 76.14 | 993 | Eagles |
| 53 | 5 | Prince Amukamara | 73.09 | 70.43 | 75.38 | 728 | Giants |
| 54 | 6 | Cary Williams | 72.84 | 63.10 | 77.86 | 1405 | Ravens |
| 55 | 7 | Kyle Arrington | 72.69 | 62.80 | 75.11 | 929 | Patriots |
| 56 | 8 | Keenan Lewis | 71.87 | 63.60 | 75.71 | 918 | Steelers |
| 57 | 9 | Dominique Franks | 71.56 | 64.66 | 81.79 | 203 | Falcons |
| 58 | 10 | Derek Cox | 71.56 | 67.89 | 75.68 | 764 | Jaguars |
| 59 | 11 | Stephon Gilmore | 71.51 | 61.40 | 74.08 | 1055 | Bills |
| 60 | 12 | Sean Smith | 71.46 | 61.60 | 74.07 | 1046 | Dolphins |
| 61 | 13 | Leodis McKelvin | 71.31 | 65.13 | 76.17 | 347 | Bills |
| 62 | 14 | Janoris Jenkins | 71.30 | 63.00 | 73.70 | 955 | Rams |
| 63 | 15 | Shareece Wright | 70.84 | 65.76 | 80.48 | 116 | Chargers |
| 64 | 16 | Orlando Scandrick | 70.71 | 65.97 | 73.25 | 326 | Cowboys |
| 65 | 17 | Chris Cook | 70.57 | 67.34 | 76.38 | 677 | Vikings |
| 66 | 18 | Elbert Mack | 70.55 | 67.82 | 76.67 | 254 | Saints |
| 67 | 19 | Lardarius Webb | 70.51 | 63.94 | 75.92 | 376 | Ravens |
| 68 | 20 | Aqib Talib | 70.28 | 60.94 | 76.40 | 658 | Patriots |
| 69 | 21 | Jonte Green | 70.24 | 63.95 | 74.43 | 396 | Lions |
| 70 | 22 | Kyle Wilson | 69.52 | 60.40 | 71.63 | 947 | Jets |
| 71 | 23 | Dunta Robinson | 69.34 | 61.10 | 70.67 | 1045 | Falcons |
| 72 | 24 | Buster Skrine | 69.03 | 60.37 | 72.59 | 722 | Browns |
| 73 | 25 | Mike Harris | 69.03 | 58.40 | 75.08 | 529 | Jaguars |
| 74 | 26 | Cedric Griffin | 68.67 | 62.43 | 75.02 | 377 | Commanders |
| 75 | 27 | Eric Wright | 68.63 | 62.09 | 72.57 | 494 | Buccaneers |
| 76 | 28 | Chris Gamble | 68.49 | 63.45 | 75.29 | 280 | Panthers |
| 77 | 29 | Alan Ball | 68.32 | 60.24 | 73.71 | 105 | Texans |
| 78 | 30 | Jerraud Powers | 68.29 | 64.46 | 73.34 | 505 | Colts |
| 79 | 31 | Marcus Trufant | 67.82 | 62.00 | 72.32 | 393 | Seahawks |
| 80 | 32 | Jacob Lacey | 67.74 | 63.76 | 70.07 | 575 | Lions |
| 81 | 33 | Stanford Routt | 67.44 | 60.00 | 70.41 | 406 | Texans |
| 82 | 34 | Aaron Ross | 67.15 | 56.30 | 71.47 | 671 | Jaguars |
| 83 | 35 | Antoine Cason | 67.01 | 52.70 | 72.39 | 1021 | Chargers |
| 84 | 36 | Perrish Cox | 66.78 | 57.06 | 69.48 | 180 | 49ers |
| 85 | 37 | Drayton Florence | 66.59 | 58.48 | 71.99 | 303 | Lions |
| 86 | 38 | Richard Crawford | 66.57 | 63.63 | 69.57 | 199 | Commanders |
| 87 | 39 | Kelvin Hayden | 66.38 | 61.27 | 70.21 | 462 | Bears |
| 88 | 40 | Josh Thomas | 66.34 | 59.84 | 74.32 | 515 | Panthers |
| 89 | 41 | Nolan Carroll | 66.32 | 58.06 | 72.56 | 641 | Dolphins |
| 90 | 42 | Ellis Lankster | 66.28 | 56.92 | 69.00 | 326 | Jets |
| 91 | 43 | Tracy Porter | 65.99 | 60.42 | 71.37 | 301 | Broncos |
| 92 | 44 | Bryan McCann | 65.97 | 62.15 | 74.86 | 141 | Dolphins |
| 93 | 45 | Mike Jenkins | 65.93 | 57.57 | 70.16 | 356 | Cowboys |
| 94 | 46 | Josh Norman | 64.94 | 57.58 | 68.82 | 771 | Panthers |
| 95 | 47 | Shawntae Spencer | 64.36 | 61.72 | 72.05 | 111 | Raiders |
| 96 | 48 | Byron Maxwell | 64.32 | 64.93 | 76.75 | 144 | Seahawks |
| 97 | 49 | Ron Brooks | 64.26 | 60.22 | 73.66 | 161 | Bills |
| 98 | 50 | Patrick Robinson | 64.20 | 52.00 | 69.64 | 1093 | Saints |
| 99 | 51 | Coty Sensabaugh | 64.04 | 56.43 | 67.03 | 313 | Titans |
| 100 | 52 | Rashean Mathis | 64.04 | 55.75 | 69.67 | 476 | Jaguars |
| 101 | 53 | Marquice Cole | 63.66 | 59.58 | 67.42 | 227 | Patriots |
| 102 | 54 | Brandian Ross | 63.66 | 63.55 | 70.44 | 175 | Raiders |
| 103 | 55 | A.J. Jefferson | 63.65 | 55.10 | 69.35 | 631 | Vikings |
| 104 | 56 | Nnamdi Asomugha | 63.59 | 47.20 | 70.77 | 984 | Eagles |
| 105 | 57 | Anthony Gaitor | 63.37 | 59.92 | 73.62 | 123 | Buccaneers |
| 106 | 58 | Richard Marshall | 63.19 | 56.78 | 69.55 | 237 | Dolphins |
| 107 | 59 | Corey Webster | 62.83 | 47.00 | 69.41 | 1016 | Giants |
| 108 | 60 | Isaiah Trufant | 62.15 | 59.52 | 71.97 | 118 | Jets |
| 109 | 61 | William Middleton | 62.07 | 59.27 | 66.96 | 196 | Jaguars |

### Rotation/backup (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 110 | 1 | Brice McCain | 61.39 | 50.73 | 67.46 | 450 | Texans |
| 111 | 2 | William Gay | 61.28 | 49.00 | 65.30 | 1006 | Cardinals |
| 112 | 3 | James Dockery | 61.12 | 61.23 | 72.38 | 158 | Panthers |
| 113 | 4 | Sterling Moore | 60.91 | 58.74 | 61.33 | 327 | Cowboys |
| 114 | 5 | Danny Gorrer | 60.82 | 56.35 | 70.05 | 185 | Buccaneers |
| 115 | 6 | Corey White | 60.46 | 56.83 | 66.02 | 519 | Saints |
| 116 | 7 | Jimmy Smith | 60.43 | 51.43 | 66.16 | 519 | Ravens |
| 117 | 8 | Justin Rogers | 60.13 | 47.41 | 67.82 | 537 | Bills |
| 118 | 9 | Brandon Harris | 59.88 | 53.78 | 70.47 | 237 | Texans |
| 119 | 10 | Ronald Bartell | 59.29 | 54.34 | 68.00 | 413 | Lions |
| 120 | 11 | Chykie Brown | 58.75 | 52.92 | 66.54 | 440 | Ravens |
| 121 | 12 | Cassius Vaughn | 57.65 | 47.50 | 66.40 | 861 | Colts |
| 122 | 13 | Trevin Wade | 57.11 | 57.81 | 56.65 | 196 | Browns |
| 123 | 14 | Jeremy Lane | 56.83 | 61.87 | 67.80 | 163 | Seahawks |
| 124 | 15 | R.J. Stanford | 56.56 | 52.35 | 60.80 | 144 | Dolphins |
| 125 | 16 | Terrence McGee | 56.36 | 54.52 | 62.69 | 142 | Bills |
| 126 | 17 | Johnny Patrick | 54.76 | 48.16 | 59.95 | 215 | Saints |
| 127 | 18 | Jayron Hosley | 52.40 | 47.79 | 55.47 | 452 | Giants |
| 128 | 19 | Ryan Mouton | 51.96 | 50.14 | 56.30 | 387 | Titans |
| 129 | 20 | Kevin Rutland | 51.92 | 57.44 | 54.10 | 103 | Jaguars |
| 130 | 21 | Jalil Brown | 50.34 | 50.91 | 54.26 | 363 | Chiefs |

## DI — Defensive Interior

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | J.J. Watt | 97.03 | 90.16 | 97.45 | 1050 | Texans |
| 2 | 2 | Geno Atkins | 91.19 | 87.53 | 89.47 | 848 | Bengals |
| 3 | 3 | Muhammad Wilkerson | 85.64 | 83.75 | 82.73 | 910 | Jets |
| 4 | 4 | Calais Campbell | 85.44 | 82.12 | 85.25 | 754 | Cardinals |
| 5 | 5 | Nick Fairley | 84.07 | 78.71 | 87.37 | 494 | Lions |
| 6 | 6 | Mike Martin | 83.19 | 73.15 | 85.72 | 429 | Titans |
| 7 | 7 | Fletcher Cox | 81.97 | 75.85 | 82.91 | 509 | Eagles |
| 8 | 8 | Ndamukong Suh | 81.81 | 80.36 | 78.93 | 879 | Lions |
| 9 | 9 | Steve McLendon | 81.24 | 70.63 | 86.96 | 135 | Steelers |
| 10 | 10 | Desmond Bryant | 81.09 | 75.69 | 80.90 | 630 | Raiders |
| 11 | 11 | Kyle Williams | 80.09 | 75.86 | 82.17 | 793 | Bills |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Jurrell Casey | 79.96 | 79.12 | 76.35 | 779 | Titans |
| 13 | 2 | Haloti Ngata | 79.69 | 79.75 | 75.48 | 1053 | Ravens |
| 14 | 3 | Marcell Dareus | 79.15 | 76.87 | 76.51 | 787 | Bills |
| 15 | 4 | Justin Smith | 78.67 | 68.76 | 81.11 | 996 | 49ers |
| 16 | 5 | Gerald McCoy | 76.83 | 88.75 | 68.46 | 938 | Buccaneers |
| 17 | 6 | Henry Melton | 76.68 | 65.66 | 81.21 | 607 | Bears |
| 18 | 7 | Fred Evans | 76.23 | 65.61 | 80.81 | 360 | Vikings |
| 19 | 8 | Brodrick Bunkley | 75.13 | 66.53 | 77.43 | 365 | Saints |
| 20 | 9 | Alex Carrington | 74.94 | 69.50 | 77.13 | 342 | Bills |
| 21 | 10 | Antonio Garay | 74.68 | 58.10 | 85.74 | 147 | Chargers |
| 22 | 11 | Paul Soliai | 74.33 | 63.27 | 77.53 | 616 | Dolphins |
| 23 | 12 | Linval Joseph | 74.27 | 69.06 | 75.86 | 692 | Giants |

### Starter (93 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Chris Canty | 73.60 | 71.44 | 74.53 | 297 | Giants |
| 25 | 2 | Richard Seymour | 73.25 | 68.23 | 77.21 | 348 | Raiders |
| 26 | 3 | Jason Hatcher | 73.23 | 62.95 | 77.49 | 757 | Cowboys |
| 27 | 4 | Vince Wilfork | 73.23 | 63.92 | 75.27 | 1019 | Patriots |
| 28 | 5 | Mike Devito | 73.14 | 65.81 | 75.11 | 629 | Jets |
| 29 | 6 | Cameron Heyward | 73.01 | 61.73 | 76.37 | 262 | Steelers |
| 30 | 7 | Karl Klug | 72.90 | 63.76 | 74.82 | 250 | Titans |
| 31 | 8 | Michael Brockers | 72.79 | 64.96 | 76.97 | 602 | Rams |
| 32 | 9 | Jason Jones | 72.65 | 56.22 | 82.44 | 319 | Seahawks |
| 33 | 10 | Dan Williams | 72.60 | 73.81 | 70.22 | 419 | Cardinals |
| 34 | 11 | Sammie Lee Hill | 72.32 | 63.14 | 75.01 | 403 | Lions |
| 35 | 12 | Brandon Mebane | 72.24 | 63.05 | 74.61 | 706 | Seahawks |
| 36 | 13 | Antonio Smith | 71.94 | 64.25 | 72.90 | 904 | Texans |
| 37 | 14 | Corey Liuget | 71.88 | 64.14 | 73.25 | 716 | Chargers |
| 38 | 15 | Vonnie Holliday | 71.56 | 50.53 | 81.62 | 195 | Cardinals |
| 39 | 16 | Kevin Williams | 71.36 | 69.14 | 69.31 | 859 | Vikings |
| 40 | 17 | Vance Walker | 71.35 | 60.17 | 74.63 | 589 | Falcons |
| 41 | 18 | Tyson Jackson | 71.31 | 64.42 | 72.88 | 595 | Chiefs |
| 42 | 19 | Ricky Jean Francois | 70.96 | 59.76 | 74.26 | 322 | 49ers |
| 43 | 20 | Cam Thomas | 70.96 | 62.41 | 74.58 | 395 | Chargers |
| 44 | 21 | Cullen Jenkins | 70.79 | 52.25 | 79.18 | 625 | Eagles |
| 45 | 22 | Earl Mitchell | 70.75 | 60.32 | 73.74 | 426 | Texans |
| 46 | 23 | Ahtyba Rubin | 70.67 | 65.41 | 71.57 | 670 | Browns |
| 47 | 24 | Ray McDonald | 70.34 | 65.16 | 69.63 | 1138 | 49ers |
| 48 | 25 | C.J. Mosley | 70.34 | 64.38 | 72.13 | 615 | Jaguars |
| 49 | 26 | Sione Pouha | 70.01 | 60.26 | 74.43 | 304 | Jets |
| 50 | 27 | Alan Branch | 69.91 | 62.58 | 70.94 | 630 | Seahawks |
| 51 | 28 | Aubrayo Franklin | 69.74 | 65.33 | 71.11 | 279 | Chargers |
| 52 | 29 | C.J. Wilson | 69.54 | 55.87 | 76.05 | 351 | Packers |
| 53 | 30 | Derek Landri | 69.30 | 57.64 | 74.15 | 484 | Eagles |
| 54 | 31 | Kenyon Coleman | 69.07 | 59.07 | 77.43 | 164 | Cowboys |
| 55 | 32 | Cory Redding | 69.03 | 49.84 | 78.17 | 634 | Colts |
| 56 | 33 | Christian Ballard | 68.97 | 59.98 | 70.79 | 402 | Vikings |
| 57 | 34 | Ropati Pitoitua | 68.74 | 53.56 | 76.12 | 494 | Chiefs |
| 58 | 35 | Domata Peko Sr. | 68.55 | 54.15 | 73.98 | 705 | Bengals |
| 59 | 36 | Josh Price-Brent | 68.53 | 70.60 | 66.64 | 309 | Cowboys |
| 60 | 37 | Randy Starks | 68.38 | 59.37 | 70.22 | 810 | Dolphins |
| 61 | 38 | Terrance Knighton | 68.15 | 62.30 | 68.82 | 656 | Jaguars |
| 62 | 39 | Derek Wolfe | 68.10 | 61.21 | 68.53 | 973 | Broncos |
| 63 | 40 | Phil Taylor Sr. | 68.06 | 61.53 | 73.44 | 263 | Browns |
| 64 | 41 | Kellen Heard | 67.77 | 55.91 | 75.16 | 179 | Colts |
| 65 | 42 | Kendall Reyes | 67.77 | 58.33 | 69.89 | 531 | Chargers |
| 66 | 43 | Spencer Johnson | 67.66 | 53.32 | 75.13 | 264 | Bills |
| 67 | 44 | Sean Lissemore | 67.61 | 56.44 | 74.79 | 317 | Cowboys |
| 68 | 45 | Arthur Jones | 67.59 | 57.91 | 72.80 | 694 | Ravens |
| 69 | 46 | Glenn Dorsey | 67.58 | 64.56 | 71.99 | 112 | Chiefs |
| 70 | 47 | Ryan Pickett | 67.55 | 56.64 | 70.98 | 645 | Packers |
| 71 | 48 | Akiem Hicks | 67.41 | 62.15 | 68.84 | 372 | Saints |
| 72 | 49 | Jared Crick | 67.38 | 60.12 | 68.06 | 228 | Texans |
| 73 | 50 | Gary Gibson | 67.34 | 59.30 | 68.53 | 278 | Buccaneers |
| 74 | 51 | Shaun Smith | 67.32 | 55.54 | 75.49 | 131 | Chiefs |
| 75 | 52 | David Carter | 67.29 | 58.50 | 69.64 | 289 | Cardinals |
| 76 | 53 | Tom Johnson | 67.27 | 57.66 | 70.55 | 419 | Saints |
| 77 | 54 | Mike Daniels | 67.18 | 54.17 | 71.69 | 272 | Packers |
| 78 | 55 | Kenrick Ellis | 67.03 | 58.60 | 75.38 | 232 | Jets |
| 79 | 56 | Tommy Kelly | 66.97 | 55.18 | 70.67 | 758 | Raiders |
| 80 | 57 | Jared Odrick | 66.61 | 59.24 | 70.49 | 932 | Dolphins |
| 81 | 58 | Pat Sims | 66.49 | 57.25 | 74.11 | 205 | Bengals |
| 82 | 59 | Stephen Paea | 66.35 | 59.85 | 69.12 | 595 | Bears |
| 83 | 60 | Jonathan Babineaux | 66.24 | 56.05 | 69.50 | 938 | Falcons |
| 84 | 61 | Kyle Love | 66.12 | 58.12 | 68.53 | 579 | Patriots |
| 85 | 62 | Cedric Thornton | 66.00 | 49.24 | 73.00 | 395 | Eagles |
| 86 | 63 | Dontari Poe | 65.82 | 57.36 | 67.30 | 743 | Chiefs |
| 87 | 64 | Mike Patterson | 65.54 | 58.61 | 72.04 | 133 | Eagles |
| 88 | 65 | Shaun Cody | 65.47 | 53.52 | 69.78 | 288 | Texans |
| 89 | 66 | Kevin Vickerson | 65.45 | 55.29 | 72.35 | 518 | Broncos |
| 90 | 67 | Letroy Guion | 65.43 | 53.47 | 69.43 | 520 | Vikings |
| 91 | 68 | Amobi Okoye | 65.26 | 58.01 | 69.57 | 229 | Bears |
| 92 | 69 | Corey Williams | 65.18 | 60.63 | 68.73 | 219 | Lions |
| 93 | 70 | Lawrence Guy Sr. | 65.12 | 55.74 | 73.46 | 222 | Colts |
| 94 | 71 | Tyson Alualu | 65.10 | 55.02 | 67.65 | 836 | Jaguars |
| 95 | 72 | John Hughes | 65.05 | 52.73 | 69.10 | 514 | Browns |
| 96 | 73 | B.J. Raji | 64.94 | 59.45 | 64.44 | 751 | Packers |
| 97 | 74 | Sedrick Ellis | 64.88 | 61.08 | 63.25 | 709 | Saints |
| 98 | 75 | Barry Cofield | 64.74 | 57.03 | 65.72 | 773 | Commanders |
| 99 | 76 | Stephen Bowen | 64.64 | 50.20 | 70.10 | 822 | Commanders |
| 100 | 77 | Marcus Spears | 64.60 | 52.51 | 70.67 | 384 | Cowboys |
| 101 | 78 | Brett Keisel | 64.47 | 50.96 | 70.05 | 860 | Steelers |
| 102 | 79 | Jay Ratliff | 64.46 | 57.99 | 69.81 | 261 | Cowboys |
| 103 | 80 | Rocky Bernard | 64.24 | 53.31 | 69.86 | 386 | Giants |
| 104 | 81 | Tyrone Crawford | 64.24 | 56.11 | 65.50 | 296 | Cowboys |
| 105 | 82 | Isaac Sopoaga | 64.18 | 51.19 | 68.67 | 393 | 49ers |
| 106 | 83 | Dwan Edwards | 64.10 | 46.44 | 73.01 | 703 | Panthers |
| 107 | 84 | Tony McDaniel | 64.08 | 53.51 | 71.03 | 248 | Dolphins |
| 108 | 85 | Chris Baker | 63.82 | 55.45 | 71.75 | 211 | Commanders |
| 109 | 86 | Billy Winn | 63.74 | 54.05 | 66.03 | 702 | Browns |
| 110 | 87 | Ishmaa'ily Kitchen | 63.59 | 55.80 | 65.65 | 208 | Browns |
| 111 | 88 | Darnell Dockett | 63.43 | 45.55 | 71.91 | 794 | Cardinals |
| 112 | 89 | Kendall Langford | 62.56 | 52.65 | 65.00 | 747 | Rams |
| 113 | 90 | Justin Bannan | 62.51 | 53.36 | 64.44 | 574 | Broncos |
| 114 | 91 | Jermelle Cudjo | 62.38 | 53.09 | 67.40 | 341 | Rams |
| 115 | 92 | Sen'Derrick Marks | 62.36 | 51.59 | 67.25 | 678 | Titans |
| 116 | 93 | Andre Neblett | 62.01 | 52.98 | 69.59 | 245 | Panthers |

### Rotation/backup (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 117 | 1 | Clinton McDonald | 61.98 | 55.46 | 64.14 | 338 | Seahawks |
| 118 | 2 | Nate Collins | 61.76 | 61.23 | 65.25 | 241 | Bears |
| 119 | 3 | Peria Jerry | 61.60 | 50.45 | 64.87 | 540 | Falcons |
| 120 | 4 | Jerrell Powe | 61.58 | 57.94 | 70.26 | 125 | Chiefs |
| 121 | 5 | DeAngelo Tyson | 61.49 | 53.64 | 65.69 | 220 | Ravens |
| 122 | 6 | Corey Peters | 61.46 | 48.64 | 67.93 | 504 | Falcons |
| 123 | 7 | Jerel Worthy | 61.24 | 53.05 | 64.61 | 452 | Packers |
| 124 | 8 | Terrence Cody | 61.22 | 53.53 | 62.38 | 453 | Ravens |
| 125 | 9 | Red Bryant | 60.72 | 50.92 | 64.97 | 716 | Seahawks |
| 126 | 10 | Andre Fluellen | 60.67 | 56.77 | 65.77 | 106 | Lions |
| 127 | 11 | Roy Miller | 60.39 | 51.15 | 62.90 | 492 | Buccaneers |
| 128 | 12 | Drake Nevis | 60.30 | 59.69 | 65.39 | 259 | Colts |
| 129 | 13 | Jarvis Jenkins | 60.21 | 53.27 | 60.67 | 582 | Commanders |
| 130 | 14 | Brandon Deaderick | 60.20 | 50.56 | 64.64 | 462 | Patriots |
| 131 | 15 | Kedric Golston | 59.85 | 51.85 | 63.84 | 387 | Commanders |
| 132 | 16 | Greg Scruggs | 59.62 | 54.76 | 61.82 | 247 | Seahawks |
| 133 | 17 | Fili Moala | 59.46 | 51.17 | 65.60 | 312 | Colts |
| 134 | 18 | Devon Still | 59.38 | 59.95 | 63.17 | 156 | Bengals |
| 135 | 19 | Casey Hampton | 59.09 | 49.57 | 61.90 | 494 | Steelers |
| 136 | 20 | Antonio Johnson | 58.11 | 48.66 | 61.17 | 483 | Colts |
| 137 | 21 | Mitch Unrein | 58.07 | 52.09 | 58.27 | 405 | Broncos |
| 138 | 22 | Allen Bailey | 57.90 | 55.69 | 60.80 | 164 | Chiefs |
| 139 | 23 | Vaughn Martin | 57.86 | 49.39 | 62.68 | 460 | Chargers |
| 140 | 24 | Ma'ake Kemoeatu | 57.60 | 44.96 | 62.64 | 541 | Ravens |
| 141 | 25 | Anthony Toribio | 57.56 | 56.00 | 61.98 | 135 | Chiefs |
| 142 | 26 | Ricardo Mathews | 57.46 | 52.35 | 60.03 | 494 | Colts |
| 143 | 27 | Christo Bilukidi | 56.58 | 55.53 | 56.24 | 242 | Raiders |
| 144 | 28 | Ron Edwards | 56.23 | 49.53 | 59.78 | 315 | Panthers |
| 145 | 29 | Frank Kearse | 55.98 | 56.44 | 60.63 | 157 | Panthers |
| 146 | 30 | Marvin Austin | 55.46 | 58.45 | 57.64 | 102 | Giants |
| 147 | 31 | Nick Eason | 55.23 | 48.16 | 55.77 | 218 | Cardinals |
| 148 | 32 | Malik Jackson | 54.78 | 54.63 | 52.80 | 116 | Broncos |
| 149 | 33 | Sione Fua | 54.28 | 50.83 | 56.32 | 252 | Panthers |
| 150 | 34 | Kheeston Randall | 51.73 | 54.10 | 50.15 | 145 | Dolphins |
| 151 | 35 | Martin Tevaseu | 50.60 | 55.43 | 50.82 | 228 | Colts |
| 152 | 36 | Markus Kuhn | 48.13 | 52.16 | 47.52 | 169 | Giants |

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
| 5 | 5 | Mario Williams | 84.15 | 88.26 | 81.31 | 929 | Bills |
| 6 | 6 | Carlos Dunlap | 83.60 | 84.98 | 80.80 | 640 | Bengals |
| 7 | 7 | Jason Babin | 83.11 | 72.46 | 86.04 | 653 | Jaguars |
| 8 | 8 | Chris Long | 82.49 | 78.44 | 81.03 | 876 | Rams |
| 9 | 9 | Brandon Graham | 82.17 | 82.89 | 82.21 | 421 | Eagles |
| 10 | 10 | Jared Allen | 81.62 | 76.28 | 81.02 | 1117 | Vikings |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Anthony Spencer | 79.97 | 80.78 | 76.29 | 847 | Cowboys |
| 12 | 2 | Justin Houston | 79.77 | 69.91 | 82.17 | 993 | Chiefs |
| 13 | 3 | Jason Pierre-Paul | 79.63 | 86.67 | 70.77 | 875 | Giants |
| 14 | 4 | Julius Peppers | 78.58 | 75.69 | 76.34 | 783 | Bears |
| 15 | 5 | Chris Clemons | 77.05 | 67.56 | 79.21 | 893 | Seahawks |
| 16 | 6 | Cliff Avril | 76.54 | 68.56 | 78.32 | 686 | Lions |
| 17 | 7 | Greg Hardy | 75.88 | 76.26 | 72.19 | 747 | Panthers |
| 18 | 8 | Terrell Suggs | 75.83 | 70.52 | 77.29 | 676 | Ravens |
| 19 | 9 | Bruce Irvin | 74.20 | 61.04 | 78.80 | 514 | Seahawks |

### Starter (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Ryan Kerrigan | 73.24 | 63.23 | 75.75 | 1135 | Commanders |
| 21 | 2 | Michael Bennett | 72.68 | 78.21 | 66.07 | 954 | Buccaneers |
| 22 | 3 | Everson Griffen | 72.54 | 70.52 | 71.19 | 656 | Vikings |
| 23 | 4 | Juqua Parker | 71.14 | 63.86 | 73.49 | 528 | Browns |
| 24 | 5 | Phillip Hunt | 70.23 | 61.85 | 76.99 | 148 | Eagles |
| 25 | 6 | Justin Tuck | 70.04 | 61.80 | 71.88 | 645 | Giants |
| 26 | 7 | Melvin Ingram III | 69.76 | 68.81 | 66.23 | 459 | Chargers |
| 27 | 8 | Wallace Gilberry | 69.74 | 55.98 | 75.27 | 346 | Bengals |
| 28 | 9 | Lawrence Jackson | 69.71 | 64.29 | 71.97 | 384 | Lions |
| 29 | 10 | Chandler Jones | 69.64 | 74.30 | 62.36 | 765 | Patriots |
| 30 | 11 | Mark Anderson | 69.47 | 57.94 | 78.92 | 244 | Bills |
| 31 | 12 | Jabaal Sheard | 68.12 | 65.78 | 65.52 | 986 | Browns |
| 32 | 13 | Robert Quinn | 67.78 | 62.96 | 67.21 | 824 | Rams |
| 33 | 14 | Shaun Phillips | 67.11 | 54.09 | 72.88 | 835 | Chargers |
| 34 | 15 | Michael Johnson | 67.10 | 65.96 | 63.69 | 905 | Bengals |
| 35 | 16 | Brian Robison | 67.04 | 62.08 | 66.18 | 887 | Vikings |
| 36 | 17 | Israel Idonije | 66.85 | 62.57 | 65.54 | 712 | Bears |
| 37 | 18 | William Hayes | 66.49 | 64.93 | 65.65 | 368 | Rams |
| 38 | 19 | Brooks Reed | 66.45 | 64.21 | 65.08 | 672 | Texans |
| 39 | 20 | Cameron Jordan | 65.92 | 70.65 | 58.60 | 1038 | Saints |
| 40 | 21 | Mathias Kiwanuka | 65.85 | 56.17 | 70.83 | 526 | Giants |
| 41 | 22 | Darryl Tapp | 65.59 | 62.24 | 66.67 | 247 | Eagles |
| 42 | 23 | O'Brien Schofield | 64.73 | 57.02 | 70.80 | 490 | Cardinals |
| 43 | 24 | Jerry Hughes | 63.58 | 60.56 | 64.45 | 612 | Colts |
| 44 | 25 | Jeremy Mincey | 63.32 | 60.01 | 61.56 | 954 | Jaguars |
| 45 | 26 | Matt Shaughnessy | 63.26 | 61.48 | 64.34 | 667 | Raiders |
| 46 | 27 | Derrick Shelby | 62.99 | 60.36 | 60.57 | 219 | Dolphins |
| 47 | 28 | Kroy Biermann | 62.80 | 60.77 | 59.98 | 762 | Falcons |
| 48 | 29 | Robert Ayers | 62.73 | 62.93 | 59.46 | 337 | Broncos |
| 49 | 30 | Antwan Applewhite | 62.70 | 58.26 | 68.68 | 100 | Panthers |
| 50 | 31 | Frank Alexander | 62.11 | 60.07 | 59.31 | 551 | Panthers |
| 51 | 32 | Jermaine Cunningham | 62.05 | 57.96 | 61.64 | 476 | Patriots |

### Rotation/backup (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 52 | 1 | Olivier Vernon | 61.65 | 63.14 | 56.49 | 431 | Dolphins |
| 53 | 2 | Frostee Rucker | 61.56 | 56.92 | 61.96 | 594 | Browns |
| 54 | 3 | Turk McBride | 61.24 | 59.51 | 65.63 | 140 | Saints |
| 55 | 4 | Adrian Clayborn | 60.84 | 59.50 | 66.04 | 176 | Buccaneers |
| 56 | 5 | Austen Lane | 60.63 | 62.57 | 61.93 | 372 | Jaguars |
| 57 | 6 | Justin Francis | 60.07 | 59.29 | 61.63 | 295 | Patriots |
| 58 | 7 | Chris Kelsay | 59.54 | 53.96 | 64.00 | 289 | Bills |
| 59 | 8 | Cliff Matthews | 59.07 | 57.00 | 60.83 | 138 | Falcons |
| 60 | 9 | Kyle Vanden Bosch | 57.72 | 47.60 | 61.33 | 641 | Lions |
| 61 | 10 | Andre Carter | 57.08 | 53.03 | 58.22 | 316 | Raiders |
| 62 | 11 | Trevor Scott | 56.93 | 59.07 | 53.94 | 286 | Patriots |
| 63 | 12 | Robert Geathers | 56.51 | 53.64 | 54.58 | 676 | Bengals |
| 64 | 13 | George Selvie | 56.31 | 55.59 | 56.27 | 237 | Jaguars |
| 65 | 14 | Dave Tollefson | 55.93 | 53.92 | 55.18 | 198 | Raiders |
| 66 | 15 | Daniel Te'o-Nesheim | 55.82 | 57.03 | 55.01 | 728 | Buccaneers |
| 67 | 16 | Kyle Moore | 55.58 | 58.98 | 56.84 | 491 | Bills |
| 68 | 17 | Andre Branch | 55.51 | 59.23 | 52.00 | 412 | Jaguars |
| 69 | 18 | Emmanuel Stephens | 55.02 | 60.96 | 54.45 | 146 | Browns |
| 70 | 19 | Eugene Sims | 54.62 | 56.32 | 53.29 | 399 | Rams |
| 71 | 20 | Aaron Morgan | 53.24 | 60.03 | 53.02 | 124 | Buccaneers |
| 72 | 21 | Scott Solomon | 52.87 | 58.54 | 48.05 | 168 | Titans |
| 73 | 22 | George Johnson | 47.45 | 55.10 | 46.38 | 116 | Vikings |

## G — Guard

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Evan Mathis | 97.50 | 92.80 | 96.47 | 1124 | Eagles |
| 2 | 2 | Logan Mankins | 91.82 | 84.20 | 92.73 | 895 | Patriots |
| 3 | 3 | Marshal Yanda | 91.00 | 84.20 | 91.36 | 1194 | Ravens |
| 4 | 4 | Mike Iupati | 89.02 | 81.80 | 89.66 | 1173 | 49ers |
| 5 | 5 | Alex Boone | 88.68 | 82.40 | 88.70 | 1195 | 49ers |
| 6 | 6 | Jon Asamoah | 88.45 | 83.50 | 87.58 | 989 | Chiefs |
| 7 | 7 | Ben Grubbs | 88.09 | 81.60 | 88.25 | 1107 | Saints |
| 8 | 8 | Kevin Boothe | 86.84 | 80.70 | 86.77 | 1012 | Giants |
| 9 | 9 | Brandon Moore | 86.66 | 80.20 | 86.80 | 1063 | Jets |
| 10 | 10 | Donald Thomas | 86.22 | 76.98 | 88.21 | 609 | Patriots |
| 11 | 11 | Andy Levitre | 85.90 | 78.60 | 86.60 | 1011 | Bills |
| 12 | 12 | John Greco | 85.38 | 74.74 | 88.31 | 691 | Browns |
| 13 | 13 | Stephen Peterman | 84.68 | 76.50 | 85.96 | 1199 | Lions |
| 14 | 14 | Josh Sitton | 84.67 | 77.40 | 85.35 | 1227 | Packers |
| 15 | 15 | Jahri Evans | 84.59 | 77.40 | 85.22 | 1105 | Saints |
| 16 | 16 | Rob Sims | 84.58 | 77.50 | 85.13 | 1199 | Lions |
| 17 | 17 | Chris Chester | 84.16 | 76.70 | 84.97 | 1088 | Commanders |
| 18 | 18 | Chris Snee | 83.80 | 76.23 | 84.68 | 952 | Giants |
| 19 | 19 | Geoff Schwartz | 83.31 | 72.74 | 86.19 | 157 | Vikings |
| 20 | 20 | Carl Nicks | 83.03 | 73.95 | 84.91 | 442 | Buccaneers |
| 21 | 21 | Kevin Zeitler | 83.02 | 75.40 | 83.93 | 1099 | Bengals |
| 22 | 22 | Nate Livings | 82.87 | 73.70 | 84.81 | 1094 | Cowboys |
| 23 | 23 | Zane Beadles | 82.46 | 74.20 | 83.80 | 1236 | Broncos |
| 24 | 24 | Richie Incognito | 82.34 | 73.80 | 83.86 | 1027 | Dolphins |
| 25 | 25 | Wade Smith | 82.33 | 74.90 | 83.11 | 1270 | Texans |
| 26 | 26 | Harvey Dahl | 81.35 | 72.33 | 83.19 | 915 | Rams |
| 27 | 27 | Kraig Urbik | 81.00 | 71.87 | 82.92 | 783 | Bills |
| 28 | 28 | Matt Slauson | 80.10 | 71.13 | 81.91 | 820 | Jets |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 29 | 1 | Louis Vasquez | 79.88 | 72.90 | 80.37 | 1020 | Chargers |
| 30 | 2 | Vladimir Ducasse | 79.34 | 67.17 | 83.28 | 274 | Jets |
| 31 | 3 | Cooper Carlisle | 79.28 | 71.10 | 80.56 | 1076 | Raiders |
| 32 | 4 | Willie Colon | 79.04 | 67.22 | 82.76 | 711 | Steelers |
| 33 | 5 | Dan Connolly | 79.00 | 71.00 | 80.16 | 1039 | Patriots |
| 34 | 6 | Ramon Foster | 78.59 | 70.10 | 80.09 | 1010 | Steelers |
| 35 | 7 | Manuel Ramirez | 78.24 | 68.62 | 80.49 | 829 | Broncos |
| 36 | 8 | Jason Pinkston | 78.03 | 65.92 | 81.94 | 332 | Browns |
| 37 | 9 | Uche Nwaneri | 77.96 | 68.34 | 80.21 | 926 | Jaguars |
| 38 | 10 | Clint Boling | 77.85 | 69.30 | 79.39 | 1104 | Bengals |
| 39 | 11 | Jake Scott | 77.29 | 63.89 | 82.06 | 462 | Eagles |
| 40 | 12 | T.J. Lang | 77.12 | 67.70 | 79.23 | 1125 | Packers |
| 41 | 13 | Justin Blalock | 77.09 | 68.40 | 78.72 | 1189 | Falcons |
| 42 | 14 | Ramon Harewood | 76.80 | 65.00 | 80.50 | 341 | Ravens |
| 43 | 15 | John Moffitt | 76.69 | 65.61 | 79.91 | 461 | Seahawks |
| 44 | 16 | Daryn Colledge | 76.63 | 66.50 | 79.22 | 1052 | Cardinals |
| 45 | 17 | Tyronne Green | 75.94 | 65.87 | 78.48 | 735 | Chargers |
| 46 | 18 | Steve Hutchinson | 75.51 | 65.49 | 78.03 | 686 | Titans |
| 47 | 19 | Garry Williams | 75.50 | 65.02 | 78.32 | 606 | Panthers |
| 48 | 20 | Chris Spencer | 75.34 | 63.43 | 79.12 | 345 | Bears |
| 49 | 21 | Russ Hochstein | 74.92 | 61.96 | 79.40 | 115 | Chiefs |
| 50 | 22 | John Jerry | 74.52 | 64.50 | 77.03 | 1034 | Dolphins |
| 51 | 23 | Charlie Johnson | 74.23 | 63.80 | 77.02 | 1093 | Vikings |
| 52 | 24 | Shawn Lauvao | 74.05 | 62.90 | 77.31 | 1031 | Browns |

### Starter (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 53 | 1 | Lance Louis | 73.79 | 63.10 | 76.75 | 692 | Bears |
| 54 | 2 | Rex Hadnot | 73.58 | 63.09 | 76.41 | 290 | Chargers |
| 55 | 3 | Brandon Fusco | 73.54 | 63.13 | 76.31 | 943 | Vikings |
| 56 | 4 | Garrett Reynolds | 73.30 | 62.40 | 76.40 | 394 | Falcons |
| 57 | 5 | Brandon Brooks | 73.28 | 62.72 | 76.15 | 173 | Texans |
| 58 | 6 | Danny Watkins | 73.26 | 61.42 | 76.98 | 448 | Eagles |
| 59 | 7 | Shelley Smith | 73.20 | 58.11 | 79.09 | 344 | Rams |
| 60 | 8 | Chris Kuper | 73.18 | 62.50 | 76.14 | 404 | Broncos |
| 61 | 9 | Jeff Allen | 73.11 | 62.55 | 75.98 | 814 | Chiefs |
| 62 | 10 | Chad Rinehart | 72.75 | 64.71 | 73.95 | 164 | Bills |
| 63 | 11 | Mike Brisiel | 72.48 | 60.30 | 76.43 | 961 | Raiders |
| 64 | 12 | Amini Silatolu | 72.22 | 59.53 | 76.51 | 882 | Panthers |
| 65 | 13 | Mike McGlynn | 70.84 | 59.80 | 74.03 | 1258 | Colts |
| 66 | 14 | Joe Reitz | 70.63 | 59.79 | 73.69 | 475 | Colts |
| 67 | 15 | David Snow | 70.41 | 62.76 | 71.34 | 133 | Bills |
| 68 | 16 | Chilo Rachal | 69.86 | 56.34 | 74.71 | 512 | Bears |
| 69 | 17 | James Carpenter | 69.18 | 57.70 | 72.66 | 343 | Seahawks |
| 70 | 18 | Antoine Caldwell | 68.46 | 57.33 | 71.71 | 346 | Texans |
| 71 | 19 | James Brown | 67.92 | 56.78 | 71.18 | 215 | Bears |
| 72 | 20 | Leonard Davis | 67.53 | 56.20 | 70.91 | 137 | 49ers |
| 73 | 21 | Adam Snyder | 67.13 | 55.50 | 70.71 | 866 | Cardinals |
| 74 | 22 | David DeCastro | 67.07 | 59.24 | 68.12 | 136 | Steelers |
| 75 | 23 | Bobbie Williams | 67.01 | 54.53 | 71.17 | 355 | Ravens |
| 76 | 24 | J.R. Sweezy | 65.81 | 54.32 | 69.31 | 376 | Seahawks |
| 77 | 25 | Seth Olsen | 65.24 | 56.00 | 67.23 | 288 | Colts |
| 78 | 26 | Derrick Dockery | 64.58 | 56.85 | 65.56 | 174 | Cowboys |

### Rotation/backup (0 players)

_None._

## HB — Running Back

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Adrian Peterson | 85.32 | 90.46 | 77.73 | 308 | Vikings |
| 2 | 2 | C.J. Spiller | 83.68 | 87.83 | 76.75 | 241 | Bills |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 3 | 1 | Doug Martin | 78.38 | 77.91 | 74.52 | 334 | Buccaneers |
| 4 | 2 | Alfred Morris | 77.88 | 82.60 | 70.57 | 202 | Commanders |
| 5 | 3 | Pierre Thomas | 77.01 | 79.27 | 71.33 | 220 | Saints |
| 6 | 4 | Marshawn Lynch | 76.17 | 79.63 | 69.69 | 246 | Seahawks |
| 7 | 5 | Joique Bell | 75.99 | 70.12 | 75.73 | 245 | Lions |
| 8 | 6 | LeSean McCoy | 75.62 | 76.38 | 70.94 | 280 | Eagles |
| 9 | 7 | Darren Sproles | 75.37 | 65.78 | 77.59 | 335 | Saints |
| 10 | 8 | Jamaal Charles | 75.24 | 74.69 | 71.44 | 204 | Chiefs |
| 11 | 9 | Isaac Redman | 74.28 | 68.58 | 73.91 | 112 | Steelers |

### Starter (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Ahmad Bradshaw | 73.94 | 76.29 | 68.20 | 185 | Giants |
| 13 | 2 | Frank Gore | 72.80 | 76.73 | 66.02 | 321 | 49ers |
| 14 | 3 | Arian Foster | 72.73 | 75.40 | 66.78 | 429 | Texans |
| 15 | 4 | Fred Jackson | 72.68 | 64.89 | 73.70 | 159 | Bills |
| 16 | 5 | Bryce Brown | 72.49 | 63.11 | 74.58 | 147 | Eagles |
| 17 | 6 | Jacquizz Rodgers | 72.30 | 71.27 | 68.82 | 299 | Falcons |
| 18 | 7 | Ryan Mathews | 72.09 | 70.70 | 68.85 | 173 | Chargers |
| 19 | 8 | Jonathan Stewart | 71.33 | 66.87 | 70.14 | 121 | Panthers |
| 20 | 9 | DeMarco Murray | 71.32 | 67.49 | 69.70 | 205 | Cowboys |
| 21 | 10 | Ray Rice | 71.20 | 73.70 | 65.37 | 410 | Ravens |
| 22 | 11 | Steven Jackson | 70.61 | 74.01 | 64.17 | 323 | Rams |
| 23 | 12 | Matt Forte | 69.91 | 66.91 | 67.74 | 274 | Bears |
| 24 | 13 | Evan Royster | 69.47 | 57.53 | 73.26 | 161 | Commanders |
| 25 | 14 | DeAngelo Williams | 69.07 | 63.98 | 68.29 | 147 | Panthers |
| 26 | 15 | Maurice Jones-Drew | 68.81 | 64.11 | 67.78 | 101 | Jaguars |
| 27 | 16 | Michael Turner | 68.25 | 64.97 | 66.27 | 154 | Falcons |
| 28 | 17 | Knowshon Moreno | 68.02 | 69.05 | 63.16 | 133 | Broncos |
| 29 | 18 | Stevan Ridley | 67.64 | 67.45 | 63.60 | 199 | Patriots |
| 30 | 19 | Trent Richardson | 67.61 | 71.87 | 60.61 | 317 | Browns |
| 31 | 20 | Willis McGahee | 67.57 | 66.88 | 63.87 | 157 | Broncos |
| 32 | 21 | Reggie Bush | 67.33 | 64.05 | 65.35 | 231 | Dolphins |
| 33 | 22 | Daryl Richardson | 66.96 | 58.07 | 68.72 | 148 | Rams |
| 34 | 23 | Vick Ballard | 66.66 | 68.10 | 61.54 | 192 | Colts |
| 35 | 24 | Jonathan Dwyer | 66.61 | 65.95 | 62.88 | 143 | Steelers |
| 36 | 25 | Robert Turbin | 66.50 | 66.86 | 62.10 | 109 | Seahawks |
| 37 | 26 | Felix Jones | 66.32 | 60.20 | 66.23 | 162 | Cowboys |
| 38 | 27 | Donald Brown | 66.08 | 62.01 | 64.63 | 106 | Colts |
| 39 | 28 | Chris Johnson | 65.95 | 62.27 | 64.23 | 366 | Titans |
| 40 | 29 | Shane Vereen | 65.53 | 67.22 | 60.23 | 112 | Patriots |
| 41 | 30 | Jason Snelling | 65.51 | 62.39 | 63.43 | 136 | Falcons |
| 42 | 31 | Danny Woodhead | 65.13 | 66.51 | 60.05 | 294 | Patriots |
| 43 | 32 | Ronnie Brown | 64.89 | 67.95 | 58.69 | 218 | Chargers |
| 44 | 33 | Toby Gerhart | 64.87 | 57.34 | 65.73 | 127 | Vikings |
| 45 | 34 | Darren McFadden | 64.66 | 53.98 | 67.61 | 258 | Raiders |
| 46 | 35 | Rashad Jennings | 64.49 | 59.78 | 63.47 | 119 | Jaguars |
| 47 | 36 | Shonn Greene | 64.48 | 64.20 | 60.50 | 169 | Jets |
| 48 | 37 | Kevin Smith | 64.48 | 61.77 | 62.12 | 103 | Lions |
| 49 | 38 | Mikel Leshoure | 64.00 | 67.15 | 57.74 | 220 | Lions |
| 50 | 39 | LaRod Stephens-Howling | 63.60 | 66.07 | 57.78 | 155 | Cardinals |
| 51 | 40 | BenJarvus Green-Ellis | 63.52 | 62.84 | 59.80 | 250 | Bengals |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 52 | 1 | Bilal Powell | 60.59 | 62.97 | 54.83 | 173 | Jets |
| 53 | 2 | Shaun Draughn | 59.86 | 56.62 | 57.85 | 134 | Chiefs |
| 54 | 3 | Alex Green | 59.54 | 61.10 | 54.34 | 105 | Packers |
| 55 | 4 | Daniel Thomas | 59.28 | 57.94 | 56.01 | 132 | Dolphins |

## LB — Linebacker

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Patrick Willis | 87.71 | 90.40 | 82.26 | 1173 | 49ers |
| 2 | 2 | NaVorro Bowman | 85.64 | 88.10 | 80.87 | 1194 | 49ers |
| 3 | 3 | Lawrence Timmons | 83.91 | 87.70 | 77.22 | 983 | Steelers |
| 4 | 4 | Bobby Wagner | 83.68 | 84.10 | 79.24 | 964 | Seahawks |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Sean Lee | 79.45 | 78.94 | 81.77 | 314 | Cowboys |
| 6 | 2 | Derrick Johnson | 78.53 | 79.70 | 73.59 | 963 | Chiefs |
| 7 | 3 | Daryl Washington | 78.31 | 82.80 | 71.46 | 1058 | Cardinals |
| 8 | 4 | Karlos Dansby | 77.35 | 76.00 | 74.50 | 1101 | Dolphins |
| 9 | 5 | Brad Jones | 76.31 | 77.80 | 72.72 | 811 | Packers |
| 10 | 6 | Jerod Mayo | 75.84 | 74.70 | 72.43 | 1200 | Patriots |
| 11 | 7 | Luke Kuechly | 75.50 | 72.50 | 73.33 | 921 | Panthers |
| 12 | 8 | Kaluka Maiava | 75.03 | 74.30 | 72.28 | 485 | Browns |
| 13 | 9 | D.J. Williams | 74.70 | 72.62 | 76.40 | 159 | Broncos |
| 14 | 10 | D'Qwell Jackson | 74.17 | 70.60 | 72.38 | 1120 | Browns |

### Starter (73 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Rolando McClain | 73.74 | 74.29 | 72.34 | 495 | Raiders |
| 16 | 2 | DeMeco Ryans | 73.64 | 73.40 | 71.72 | 1045 | Eagles |
| 17 | 3 | Wesley Woodyard | 73.39 | 72.20 | 71.90 | 956 | Broncos |
| 18 | 4 | Paul Posluszny | 72.86 | 69.40 | 71.41 | 1126 | Jaguars |
| 19 | 5 | Lavonte David | 72.85 | 69.10 | 71.18 | 1057 | Buccaneers |
| 20 | 6 | Kevin Burnett | 72.81 | 71.00 | 69.85 | 1073 | Dolphins |
| 21 | 7 | Jerrell Freeman | 72.73 | 70.30 | 70.19 | 1098 | Colts |
| 22 | 8 | Lance Briggs | 72.52 | 70.00 | 70.03 | 1007 | Bears |
| 23 | 9 | A.J. Hawk | 72.20 | 69.90 | 69.89 | 832 | Packers |
| 24 | 10 | Thomas Davis Sr. | 72.10 | 72.80 | 71.21 | 1556 | Panthers |
| 25 | 11 | James Laurinaitis | 72.05 | 68.40 | 70.31 | 1075 | Rams |
| 26 | 12 | Dont'a Hightower | 71.64 | 70.00 | 68.56 | 654 | Patriots |
| 27 | 13 | Philip Wheeler | 71.37 | 71.40 | 68.12 | 1015 | Raiders |
| 28 | 14 | Koa Misi | 71.32 | 70.24 | 68.90 | 550 | Dolphins |
| 29 | 15 | Brandon Spikes | 71.11 | 67.90 | 71.27 | 843 | Patriots |
| 30 | 16 | Brian Cushing | 70.93 | 70.71 | 73.48 | 246 | Texans |
| 31 | 17 | Takeo Spikes | 70.75 | 66.39 | 69.49 | 699 | Chargers |
| 32 | 18 | Stephen Tulloch | 70.52 | 68.70 | 67.56 | 1023 | Lions |
| 33 | 19 | Kelvin Sheppard | 70.51 | 67.68 | 69.40 | 503 | Bills |
| 34 | 20 | K.J. Wright | 70.43 | 67.00 | 68.94 | 982 | Seahawks |
| 35 | 21 | Tim Dobbins | 70.39 | 66.68 | 71.17 | 389 | Texans |
| 36 | 22 | Moise Fokou | 70.36 | 67.00 | 70.51 | 394 | Colts |
| 37 | 23 | Nick Barnett | 70.26 | 68.20 | 69.96 | 999 | Bills |
| 38 | 24 | Zach Brown | 69.36 | 66.56 | 67.06 | 742 | Titans |
| 39 | 25 | Bruce Carter | 69.33 | 70.32 | 71.67 | 607 | Cowboys |
| 40 | 26 | Bryan Scott | 69.25 | 66.11 | 67.17 | 591 | Bills |
| 41 | 27 | Brendon Ayanbadejo | 69.11 | 68.50 | 72.44 | 187 | Ravens |
| 42 | 28 | Donald Butler | 68.97 | 66.99 | 68.73 | 707 | Chargers |
| 43 | 29 | Dannell Ellerbe | 68.95 | 67.90 | 67.67 | 960 | Ravens |
| 44 | 30 | Kavell Conner | 68.78 | 68.16 | 67.11 | 328 | Colts |
| 45 | 31 | Vontaze Burfict | 68.35 | 63.20 | 67.62 | 964 | Bengals |
| 46 | 32 | Mason Foster | 68.31 | 63.60 | 67.28 | 737 | Buccaneers |
| 47 | 33 | Adam Hayward | 68.00 | 64.33 | 70.76 | 148 | Buccaneers |
| 48 | 34 | Chad Greenway | 67.57 | 63.10 | 66.39 | 1192 | Vikings |
| 49 | 35 | Curtis Lofton | 67.31 | 62.50 | 66.35 | 1121 | Saints |
| 50 | 36 | Jonathan Casillas | 67.14 | 63.81 | 67.66 | 250 | Saints |
| 51 | 37 | Leroy Hill | 67.06 | 62.99 | 66.12 | 564 | Seahawks |
| 52 | 38 | Keith Brooking | 66.96 | 60.69 | 66.98 | 488 | Broncos |
| 53 | 39 | Akeem Dent | 66.94 | 65.77 | 68.63 | 557 | Falcons |
| 54 | 40 | Bront Bird | 66.80 | 69.12 | 75.08 | 110 | Chargers |
| 55 | 41 | Brian Urlacher | 66.69 | 60.48 | 68.74 | 716 | Bears |
| 56 | 42 | Perry Riley | 66.19 | 66.00 | 66.83 | 1128 | Commanders |
| 57 | 43 | Demorrio Williams | 65.74 | 64.42 | 66.01 | 357 | Chargers |
| 58 | 44 | Emmanuel Lamur | 65.69 | 64.06 | 69.91 | 135 | Bengals |
| 59 | 45 | Malcolm Smith | 65.58 | 64.23 | 70.00 | 181 | Seahawks |
| 60 | 46 | Bradie James | 65.56 | 59.32 | 65.55 | 752 | Texans |
| 61 | 47 | Jovan Belcher | 65.50 | 61.97 | 66.28 | 332 | Chiefs |
| 62 | 48 | Nick Roach | 65.38 | 61.94 | 64.98 | 697 | Bears |
| 63 | 49 | David Harris | 65.02 | 59.50 | 64.53 | 1062 | Jets |
| 64 | 50 | Larry Foote | 64.99 | 59.70 | 64.35 | 967 | Steelers |
| 65 | 51 | Scott Fujita | 64.84 | 64.32 | 69.15 | 122 | Browns |
| 66 | 52 | JoLonn Dunbar | 64.78 | 62.40 | 63.77 | 1059 | Rams |
| 67 | 53 | Russell Allen | 64.63 | 60.00 | 65.63 | 997 | Jaguars |
| 68 | 54 | Dan Connor | 64.62 | 61.22 | 65.74 | 340 | Cowboys |
| 69 | 55 | Josh Bynes | 64.53 | 63.75 | 68.25 | 211 | Ravens |
| 70 | 56 | Arthur Moats | 64.47 | 61.68 | 65.82 | 121 | Bills |
| 71 | 57 | Nigel Bradham | 64.38 | 61.04 | 64.53 | 395 | Bills |
| 72 | 58 | Geno Hayes | 64.15 | 63.24 | 68.93 | 138 | Bears |
| 73 | 59 | Tim Shaw | 64.14 | 64.28 | 68.21 | 223 | Titans |
| 74 | 60 | Craig Robertson | 64.03 | 61.12 | 61.81 | 612 | Browns |
| 75 | 61 | Danny Trevathan | 63.65 | 58.38 | 65.08 | 239 | Broncos |
| 76 | 62 | Jameel McClain | 63.59 | 60.19 | 63.26 | 738 | Ravens |
| 77 | 63 | Jacquian Williams | 63.47 | 59.17 | 66.07 | 286 | Giants |
| 78 | 64 | Sean Weatherspoon | 63.41 | 61.00 | 62.20 | 947 | Falcons |
| 79 | 65 | Will Witherspoon | 63.33 | 61.44 | 63.03 | 388 | Titans |
| 80 | 66 | Bart Scott | 63.31 | 61.44 | 60.91 | 572 | Jets |
| 81 | 67 | Akeem Ayers | 63.11 | 59.70 | 61.21 | 861 | Titans |
| 82 | 68 | Justin Durant | 62.59 | 58.50 | 63.03 | 851 | Lions |
| 83 | 69 | Demario Davis | 62.46 | 58.77 | 61.79 | 309 | Jets |
| 84 | 70 | Michael Boley | 62.28 | 55.40 | 62.70 | 839 | Giants |
| 85 | 71 | Alex Albright | 62.25 | 59.16 | 63.40 | 179 | Cowboys |
| 86 | 72 | Dekoda Watson | 62.23 | 56.80 | 64.50 | 126 | Buccaneers |
| 87 | 73 | Akeem Jordan | 62.21 | 59.30 | 64.25 | 331 | Eagles |

### Rotation/backup (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 88 | 1 | Michael Morgan | 61.70 | 61.72 | 67.93 | 110 | Seahawks |
| 89 | 2 | Julian Stanford | 61.54 | 55.95 | 62.14 | 350 | Jaguars |
| 90 | 3 | Erin Henderson | 61.43 | 55.14 | 64.59 | 743 | Vikings |
| 91 | 4 | Spencer Paysinger | 61.35 | 61.98 | 66.02 | 134 | Giants |
| 92 | 5 | Dan Skuta | 61.27 | 58.12 | 61.15 | 114 | Bengals |
| 93 | 6 | D.J. Smith | 61.11 | 62.82 | 66.61 | 369 | Packers |
| 94 | 7 | Brandon Siler | 61.09 | 61.43 | 66.20 | 174 | Chiefs |
| 95 | 8 | DeAndre Levy | 60.88 | 55.75 | 62.21 | 699 | Lions |
| 96 | 9 | Kyle Bosworth | 60.63 | 57.07 | 61.44 | 250 | Jaguars |
| 97 | 10 | Jason Trusnik | 60.60 | 58.28 | 61.88 | 112 | Dolphins |
| 98 | 11 | Pat Angerer | 60.34 | 53.87 | 63.19 | 339 | Colts |
| 99 | 12 | Keith Rivers | 60.33 | 57.71 | 63.25 | 233 | Giants |
| 100 | 13 | Chase Blackburn | 60.20 | 59.02 | 61.82 | 783 | Giants |
| 101 | 14 | Mychal Kendricks | 60.18 | 55.40 | 60.24 | 927 | Eagles |
| 102 | 15 | Ashlee Palmer | 59.91 | 57.72 | 62.81 | 116 | Lions |
| 103 | 16 | Jonathan Vilma | 59.84 | 58.38 | 60.20 | 406 | Saints |
| 104 | 17 | Rocky McIntosh | 59.80 | 55.50 | 59.97 | 446 | Rams |
| 105 | 18 | James-Michael Johnson | 59.77 | 58.11 | 64.01 | 287 | Browns |
| 106 | 19 | London Fletcher | 59.63 | 53.50 | 59.55 | 1078 | Commanders |
| 107 | 20 | Barrett Ruud | 59.53 | 56.40 | 63.28 | 154 | Texans |
| 108 | 21 | Mike Peterson | 59.15 | 59.28 | 61.24 | 119 | Falcons |
| 109 | 22 | James Anderson | 59.11 | 53.40 | 60.83 | 518 | Panthers |
| 110 | 23 | Ray Lewis | 59.03 | 48.91 | 65.36 | 771 | Ravens |
| 111 | 24 | Quincy Black | 59.00 | 56.48 | 61.83 | 292 | Buccaneers |
| 112 | 25 | Will Herring | 58.58 | 58.00 | 62.20 | 102 | Saints |
| 113 | 26 | Jason Phillips | 58.55 | 61.08 | 63.38 | 110 | Panthers |
| 114 | 27 | Miles Burris | 58.43 | 51.70 | 58.75 | 868 | Raiders |
| 115 | 28 | Mark Herzlich | 58.25 | 58.85 | 62.40 | 175 | Giants |
| 116 | 29 | Paris Lenon | 58.13 | 50.90 | 58.78 | 1014 | Cardinals |
| 117 | 30 | Vincent Rey | 58.09 | 56.60 | 61.36 | 110 | Bengals |
| 118 | 31 | Stephen Nicholas | 57.66 | 51.10 | 59.75 | 979 | Falcons |
| 119 | 32 | Daryl Smith | 57.40 | 56.80 | 60.93 | 116 | Jaguars |
| 120 | 33 | Scott Shanle | 57.01 | 54.64 | 59.33 | 218 | Saints |
| 121 | 34 | David Hawthorne | 56.85 | 51.48 | 59.91 | 320 | Saints |
| 122 | 35 | Omar Gaither | 56.35 | 58.32 | 61.90 | 144 | Raiders |
| 123 | 36 | Ernie Sims | 56.15 | 52.14 | 58.56 | 364 | Cowboys |
| 124 | 37 | Rey Maualuga | 55.71 | 42.40 | 61.05 | 1087 | Bengals |
| 125 | 38 | Jon Beason | 54.36 | 56.72 | 59.55 | 262 | Panthers |
| 126 | 39 | Jasper Brinkley | 54.26 | 46.90 | 60.09 | 869 | Vikings |
| 127 | 40 | Joe Mays | 53.57 | 59.34 | 64.06 | 292 | Broncos |
| 128 | 41 | Jamar Chaney | 53.09 | 50.98 | 57.09 | 231 | Eagles |
| 129 | 42 | Colin McCarthy | 50.87 | 46.43 | 57.48 | 385 | Titans |

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
| 6 | 1 | Eli Manning | 79.36 | 83.06 | 72.76 | 608 | Giants |
| 7 | 2 | Ben Roethlisberger | 79.32 | 81.76 | 75.23 | 509 | Steelers |
| 8 | 3 | Joe Flacco | 76.49 | 74.48 | 73.58 | 756 | Ravens |
| 9 | 4 | Tony Romo | 76.33 | 75.17 | 74.21 | 728 | Cowboys |
| 10 | 5 | Russell Wilson | 74.96 | 88.32 | 78.19 | 590 | Seahawks |
| 11 | 6 | Matt Schaub | 74.77 | 76.16 | 71.28 | 693 | Texans |
| 12 | 7 | Philip Rivers | 74.25 | 72.39 | 71.68 | 619 | Chargers |

### Starter (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Cam Newton | 72.16 | 67.53 | 72.61 | 588 | Panthers |
| 14 | 2 | Alex Smith | 71.46 | 69.73 | 75.96 | 267 | 49ers |
| 15 | 3 | Matthew Stafford | 71.46 | 73.17 | 67.02 | 801 | Lions |
| 16 | 4 | Robert Griffin III | 70.46 | 83.80 | 74.66 | 511 | Commanders |
| 17 | 5 | Carson Palmer | 69.51 | 67.65 | 68.66 | 625 | Raiders |
| 18 | 6 | Jay Cutler | 68.70 | 69.57 | 67.72 | 529 | Bears |
| 19 | 7 | Josh Freeman | 68.15 | 63.49 | 67.71 | 632 | Buccaneers |
| 20 | 8 | Sam Bradford | 67.84 | 69.74 | 63.37 | 638 | Rams |
| 21 | 9 | Andy Dalton | 67.81 | 63.37 | 66.90 | 659 | Bengals |
| 22 | 10 | Colin Kaepernick | 67.46 | 77.99 | 81.11 | 372 | 49ers |
| 23 | 11 | Andrew Luck | 66.00 | 64.70 | 64.46 | 813 | Colts |
| 24 | 12 | Michael Vick | 65.66 | 66.45 | 65.28 | 434 | Eagles |
| 25 | 13 | Ryan Tannehill | 64.52 | 69.94 | 63.51 | 567 | Dolphins |
| 26 | 14 | Matt Hasselbeck | 63.91 | 66.71 | 64.82 | 252 | Titans |
| 27 | 15 | Ryan Fitzpatrick | 63.75 | 58.55 | 64.63 | 584 | Bills |

### Rotation/backup (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Jake Locker | 60.99 | 66.09 | 63.70 | 387 | Titans |
| 29 | 2 | Kevin Kolb | 60.79 | 61.82 | 66.93 | 229 | Cardinals |
| 30 | 3 | Nick Foles | 59.39 | 58.27 | 63.86 | 313 | Eagles |
| 31 | 4 | Christian Ponder | 59.09 | 53.90 | 62.15 | 582 | Vikings |
| 32 | 5 | Mark Sanchez | 58.96 | 54.33 | 60.43 | 530 | Jets |
| 33 | 6 | Chad Henne | 58.82 | 60.66 | 61.56 | 372 | Jaguars |
| 34 | 7 | Brandon Weeden | 58.13 | 51.41 | 58.48 | 578 | Browns |
| 35 | 8 | Matt Cassel | 57.42 | 61.14 | 58.08 | 328 | Chiefs |
| 36 | 9 | Blaine Gabbert | 55.22 | 53.72 | 58.16 | 335 | Jaguars |
| 37 | 10 | Brady Quinn | 55.07 | 52.56 | 54.93 | 239 | Chiefs |
| 38 | 11 | Ryan Lindley | 53.91 | 46.82 | 54.30 | 193 | Cardinals |
| 39 | 12 | John Skelton | 50.99 | 46.56 | 55.77 | 225 | Cardinals |

## S — Safety

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (9 players)

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
| 9 | 9 | George Wilson | 80.05 | 73.90 | 81.13 | 897 | Bills |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Jim Leonhard | 79.54 | 74.28 | 80.87 | 277 | Broncos |
| 11 | 2 | Dwight Lowery | 77.38 | 74.91 | 79.44 | 545 | Jaguars |
| 12 | 3 | Isa Abdul-Quddus | 77.14 | 76.89 | 76.52 | 473 | Saints |
| 13 | 4 | Chris Clemons | 76.93 | 75.00 | 78.32 | 1097 | Dolphins |
| 14 | 5 | Stevie Brown | 76.74 | 77.10 | 74.42 | 829 | Giants |
| 15 | 6 | Troy Polamalu | 76.58 | 71.18 | 80.70 | 386 | Steelers |
| 16 | 7 | Kenny Phillips | 76.54 | 69.17 | 81.97 | 293 | Giants |
| 17 | 8 | Rafael Bush | 76.46 | 69.44 | 86.55 | 122 | Saints |
| 18 | 9 | Dashon Goldson | 74.29 | 71.20 | 72.19 | 1221 | 49ers |

### Starter (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Rashad Johnson | 73.78 | 69.50 | 77.36 | 161 | Cardinals |
| 20 | 2 | Morgan Burnett | 73.76 | 69.20 | 75.13 | 1228 | Packers |
| 21 | 3 | Quintin Mikell | 73.70 | 63.00 | 76.67 | 1021 | Rams |
| 22 | 4 | William Moore | 73.19 | 67.90 | 74.53 | 877 | Falcons |
| 23 | 5 | Thomas DeCoud | 72.73 | 72.10 | 68.99 | 1139 | Falcons |
| 24 | 6 | Kam Chancellor | 72.45 | 66.10 | 72.84 | 1094 | Seahawks |
| 25 | 7 | Earl Thomas III | 72.04 | 70.10 | 69.17 | 1077 | Seahawks |
| 26 | 8 | T.J. Ward | 71.85 | 64.60 | 76.07 | 981 | Browns |
| 27 | 9 | Chris Crocker | 71.76 | 67.93 | 73.17 | 600 | Bengals |
| 28 | 10 | Tyvon Branch | 71.31 | 64.40 | 72.78 | 834 | Raiders |
| 29 | 11 | Eric Smith | 71.26 | 64.74 | 73.52 | 321 | Jets |
| 30 | 12 | Jamarca Sanford | 71.13 | 66.50 | 72.45 | 834 | Vikings |
| 31 | 13 | Don Carey | 70.58 | 65.14 | 74.83 | 358 | Lions |
| 32 | 14 | Rahim Moore | 70.42 | 61.30 | 73.50 | 1120 | Broncos |
| 33 | 15 | Abram Elam | 70.40 | 62.44 | 74.67 | 451 | Chiefs |
| 34 | 16 | Major Wright | 70.34 | 65.30 | 71.61 | 1028 | Bears |
| 35 | 17 | Adrian Wilson | 70.25 | 63.70 | 70.96 | 846 | Cardinals |
| 36 | 18 | Glover Quin | 69.74 | 59.70 | 72.26 | 1143 | Texans |
| 37 | 19 | LaRon Landry | 69.49 | 66.30 | 71.41 | 1019 | Jets |
| 38 | 20 | Ahmad Black | 69.32 | 62.63 | 69.61 | 416 | Buccaneers |
| 39 | 21 | Tavon Wilson | 68.59 | 60.23 | 70.00 | 487 | Patriots |
| 40 | 22 | Usama Young | 68.49 | 60.36 | 73.07 | 669 | Browns |
| 41 | 23 | Taylor Mays | 68.46 | 61.05 | 74.87 | 251 | Bengals |
| 42 | 24 | Jerron McMillian | 68.29 | 64.99 | 66.33 | 591 | Packers |
| 43 | 25 | Eric Frampton | 67.75 | 64.17 | 72.73 | 194 | Cowboys |
| 44 | 26 | Yeremiah Bell | 67.73 | 60.40 | 68.45 | 1058 | Jets |
| 45 | 27 | Da'Norris Searcy | 67.60 | 59.31 | 73.64 | 272 | Bills |
| 46 | 28 | Patrick Chung | 67.35 | 60.16 | 70.79 | 534 | Patriots |
| 47 | 29 | Amari Spievey | 66.98 | 63.60 | 71.64 | 190 | Lions |
| 48 | 30 | Haruki Nakamura | 66.86 | 59.66 | 69.37 | 591 | Panthers |
| 49 | 31 | Jeron Johnson | 66.40 | 62.13 | 70.54 | 133 | Seahawks |
| 50 | 32 | James Ihedigbo | 66.39 | 62.92 | 65.47 | 293 | Ravens |
| 51 | 33 | Will Hill III | 66.11 | 61.93 | 72.03 | 212 | Giants |
| 52 | 34 | Craig Dahl | 65.72 | 62.90 | 63.64 | 1035 | Rams |
| 53 | 35 | M.D. Jennings | 65.72 | 63.08 | 69.18 | 603 | Packers |
| 54 | 36 | Gerald Sensabaugh | 65.67 | 62.30 | 64.26 | 951 | Cowboys |
| 55 | 37 | Steve Gregory | 65.56 | 64.10 | 65.18 | 884 | Patriots |
| 56 | 38 | Mike Adams | 65.43 | 53.50 | 69.42 | 1056 | Broncos |
| 57 | 39 | Chris Conte | 65.43 | 61.50 | 66.89 | 855 | Bears |
| 58 | 40 | Donte Whitner | 65.37 | 57.70 | 66.31 | 1232 | 49ers |
| 59 | 41 | Troy Nolan | 65.01 | 60.00 | 68.04 | 105 | Bears |
| 60 | 42 | Corey Lynch | 64.95 | 58.53 | 68.92 | 494 | Chargers |
| 61 | 43 | Charles Godfrey | 64.75 | 62.00 | 63.57 | 984 | Panthers |
| 62 | 44 | Will Allen | 64.75 | 61.22 | 66.69 | 423 | Steelers |
| 63 | 45 | Reed Doughty | 64.56 | 56.11 | 66.76 | 428 | Commanders |
| 64 | 46 | Bernard Pollard | 64.55 | 55.30 | 66.75 | 1223 | Ravens |
| 65 | 47 | Tysyn Hartman | 64.52 | 60.05 | 70.63 | 237 | Chiefs |
| 66 | 48 | DeJon Gomes | 64.42 | 63.96 | 66.67 | 385 | Commanders |
| 67 | 49 | James Sanders | 63.99 | 57.96 | 69.58 | 120 | Cardinals |
| 68 | 50 | Antoine Bethea | 63.70 | 52.90 | 66.73 | 1107 | Colts |
| 69 | 51 | Dawan Landry | 63.52 | 53.20 | 66.24 | 1138 | Jaguars |
| 70 | 52 | Eric Berry | 63.50 | 57.60 | 67.95 | 993 | Chiefs |
| 71 | 53 | Sherrod Martin | 62.99 | 61.12 | 63.92 | 271 | Panthers |
| 72 | 54 | Shiloh Keo | 62.37 | 60.44 | 69.52 | 109 | Texans |

### Rotation/backup (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 73 | 1 | Ryan Mundy | 61.95 | 60.47 | 62.93 | 283 | Steelers |
| 74 | 2 | Craig Steltz | 61.81 | 60.04 | 66.00 | 109 | Bears |
| 75 | 3 | Tashaun Gipson Sr. | 61.65 | 55.93 | 67.54 | 367 | Browns |
| 76 | 4 | Tom Zbikowski | 61.11 | 60.00 | 64.35 | 695 | Colts |
| 77 | 5 | Danieal Manning | 60.97 | 50.00 | 64.43 | 1164 | Texans |
| 78 | 6 | Eric Hagg | 60.73 | 60.85 | 62.46 | 352 | Browns |
| 79 | 7 | Jordan Babineaux | 60.42 | 51.63 | 62.76 | 763 | Titans |
| 80 | 8 | Antrel Rolle | 60.37 | 51.00 | 62.45 | 1015 | Giants |
| 81 | 9 | Chris Prosinski | 59.61 | 55.03 | 60.07 | 674 | Jaguars |
| 82 | 10 | Quintin Demps | 59.58 | 55.27 | 63.24 | 347 | Texans |
| 83 | 11 | Madieu Williams | 59.26 | 54.20 | 62.31 | 1094 | Commanders |
| 84 | 12 | Jordan Pugh | 58.95 | 58.68 | 59.33 | 296 | Commanders |
| 85 | 13 | Robert Johnson | 58.77 | 55.99 | 58.54 | 279 | Titans |
| 86 | 14 | Louis Delmas | 58.41 | 53.40 | 63.21 | 434 | Lions |
| 87 | 15 | Ricardo Silva | 57.71 | 57.17 | 62.23 | 414 | Lions |
| 88 | 16 | Matt Giordano | 57.51 | 50.73 | 61.10 | 802 | Raiders |
| 89 | 17 | Erik Coleman | 57.13 | 57.38 | 61.77 | 462 | Lions |
| 90 | 18 | Danny McCray | 57.02 | 53.49 | 59.06 | 638 | Cowboys |
| 91 | 19 | Travis Daniels | 56.80 | 56.61 | 56.28 | 282 | Chiefs |
| 92 | 20 | Jeromy Miles | 56.42 | 57.08 | 56.36 | 121 | Bengals |
| 93 | 21 | Charlie Peprah | 56.40 | 57.06 | 57.52 | 180 | Cowboys |
| 94 | 22 | Nate Allen | 56.36 | 49.70 | 58.61 | 846 | Eagles |
| 95 | 23 | Chris Hope | 55.28 | 54.85 | 56.91 | 270 | Falcons |
| 96 | 24 | Mike Mitchell | 54.77 | 42.73 | 60.29 | 330 | Raiders |
| 97 | 25 | Colt Anderson | 53.70 | 52.38 | 58.12 | 292 | Eagles |
| 98 | 26 | Barry Church | 53.59 | 55.72 | 56.66 | 103 | Cowboys |
| 99 | 27 | Roman Harper | 53.46 | 40.00 | 58.26 | 1097 | Saints |
| 100 | 28 | Anthony Walters | 52.75 | 59.88 | 57.64 | 110 | Bears |
| 101 | 29 | Kendrick Lewis | 52.65 | 46.57 | 56.80 | 553 | Chiefs |
| 102 | 30 | Michael Griffin | 51.97 | 40.00 | 55.78 | 1115 | Titans |
| 103 | 31 | Kurt Coleman | 51.14 | 40.00 | 56.49 | 880 | Eagles |
| 104 | 32 | Atari Bigby | 50.99 | 43.68 | 57.12 | 615 | Chargers |
| 105 | 33 | John Wendling | 50.94 | 55.20 | 56.44 | 166 | Lions |
| 106 | 34 | Malcolm Jenkins | 50.87 | 40.00 | 55.72 | 873 | Saints |
| 107 | 35 | Mistral Raymond | 49.83 | 40.00 | 58.60 | 402 | Vikings |
| 108 | 36 | D.J. Campbell | 49.51 | 56.68 | 56.53 | 245 | Panthers |

## T — Tackle

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Joe Staley | 95.62 | 92.20 | 93.74 | 1152 | 49ers |
| 2 | 2 | Duane Brown | 94.93 | 90.10 | 93.99 | 1265 | Texans |
| 3 | 3 | Andre Smith | 88.76 | 81.20 | 89.63 | 1095 | Bengals |
| 4 | 4 | Michael Roos | 88.31 | 82.94 | 87.72 | 935 | Titans |
| 5 | 5 | Will Beatty | 87.85 | 80.50 | 88.59 | 953 | Giants |
| 6 | 6 | Trent Williams | 87.35 | 80.40 | 87.81 | 1025 | Commanders |
| 7 | 7 | Nate Solder | 87.34 | 81.00 | 87.40 | 1385 | Patriots |
| 8 | 8 | Anthony Davis | 87.32 | 80.10 | 87.96 | 1198 | 49ers |
| 9 | 9 | Jordan Gross | 87.07 | 80.90 | 87.02 | 1024 | Panthers |
| 10 | 10 | Eugene Monroe | 86.95 | 80.50 | 87.09 | 1062 | Jaguars |
| 11 | 11 | Ryan Clady | 86.88 | 81.10 | 86.57 | 1206 | Broncos |
| 12 | 12 | Russell Okung | 86.58 | 78.40 | 87.86 | 1050 | Seahawks |
| 13 | 13 | Tyson Clabo | 86.21 | 77.90 | 87.59 | 1179 | Falcons |
| 14 | 14 | Anthony Castonzo | 86.07 | 78.20 | 87.15 | 1259 | Colts |
| 15 | 15 | Joe Thomas | 86.04 | 80.10 | 85.83 | 1031 | Browns |
| 16 | 16 | Jared Veldheer | 85.91 | 79.70 | 85.89 | 1078 | Raiders |
| 17 | 17 | Matt Kalil | 85.82 | 78.60 | 86.46 | 1096 | Vikings |
| 18 | 18 | D'Brickashaw Ferguson | 85.42 | 79.00 | 85.53 | 1074 | Jets |
| 19 | 19 | Phil Loadholt | 85.10 | 74.80 | 87.80 | 1096 | Vikings |
| 20 | 20 | Donald Penn | 85.04 | 78.10 | 85.50 | 1047 | Buccaneers |
| 21 | 21 | Branden Albert | 84.32 | 76.34 | 85.48 | 705 | Chiefs |
| 22 | 22 | Sebastian Vollmer | 84.00 | 76.30 | 84.96 | 1248 | Patriots |
| 23 | 23 | Jason Smith | 83.79 | 72.97 | 86.84 | 257 | Jets |
| 24 | 24 | Gosder Cherilus | 83.59 | 74.70 | 85.35 | 1198 | Lions |
| 25 | 25 | Jared Gaither | 83.58 | 69.93 | 88.51 | 243 | Chargers |
| 26 | 26 | Andrew Whitworth | 83.01 | 75.70 | 83.72 | 1042 | Bengals |
| 27 | 27 | Eric Winston | 82.60 | 73.00 | 84.84 | 1051 | Chiefs |
| 28 | 28 | Riley Reiff | 82.35 | 75.99 | 82.43 | 326 | Lions |
| 29 | 29 | Austin Howard | 82.04 | 73.00 | 83.90 | 1073 | Jets |
| 30 | 30 | Tyron Smith | 82.00 | 71.08 | 85.11 | 950 | Cowboys |
| 31 | 31 | Chris Hairston | 81.92 | 70.56 | 85.33 | 568 | Bills |
| 32 | 32 | Ryan Harris | 81.89 | 70.87 | 85.07 | 460 | Texans |
| 33 | 33 | Sam Baker | 81.74 | 75.60 | 81.67 | 1189 | Falcons |
| 34 | 34 | Bryant McKinnie | 81.13 | 71.41 | 83.45 | 410 | Ravens |
| 35 | 35 | Mitchell Schwartz | 81.10 | 72.30 | 82.80 | 1031 | Browns |
| 36 | 36 | David Stewart | 81.10 | 70.40 | 84.07 | 675 | Titans |
| 37 | 37 | Jermon Bushrod | 80.67 | 71.80 | 82.42 | 1103 | Saints |
| 38 | 38 | Barry Richardson | 80.33 | 70.70 | 82.59 | 1023 | Rams |
| 39 | 39 | Cordy Glenn | 80.23 | 70.02 | 82.87 | 803 | Bills |
| 40 | 40 | King Dunlap | 80.20 | 71.12 | 82.08 | 818 | Eagles |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 41 | 1 | J'Marcus Webb | 78.93 | 69.60 | 80.99 | 1046 | Bears |
| 42 | 2 | Dennis Roland | 78.69 | 65.52 | 83.30 | 104 | Bengals |
| 43 | 3 | Jake Long | 78.01 | 67.06 | 81.14 | 730 | Dolphins |
| 44 | 4 | Demar Dotson | 77.95 | 67.70 | 80.62 | 986 | Buccaneers |
| 45 | 5 | Mike Adams | 76.24 | 64.43 | 79.94 | 487 | Steelers |
| 46 | 6 | Jeff Backus | 76.22 | 68.40 | 77.26 | 1064 | Lions |
| 47 | 7 | Marcus Gilbert | 76.17 | 63.21 | 80.65 | 240 | Steelers |
| 48 | 8 | Michael Oher | 76.11 | 66.70 | 78.22 | 1340 | Ravens |
| 49 | 9 | Bobby Massie | 75.98 | 63.70 | 80.00 | 1052 | Cardinals |
| 50 | 10 | Marshall Newhouse | 75.98 | 67.60 | 77.40 | 1229 | Packers |
| 51 | 11 | Breno Giacomini | 75.89 | 63.30 | 80.11 | 1144 | Seahawks |
| 52 | 12 | Charles Brown | 75.82 | 62.32 | 80.65 | 120 | Saints |
| 53 | 13 | Doug Free | 75.78 | 62.80 | 80.26 | 1022 | Cowboys |
| 54 | 14 | Winston Justice | 75.74 | 63.96 | 79.43 | 798 | Colts |
| 55 | 15 | Marcus Cannon | 75.62 | 63.24 | 79.70 | 179 | Patriots |
| 56 | 16 | Derek Newton | 75.42 | 63.18 | 79.41 | 864 | Texans |
| 57 | 17 | Sam Young | 75.22 | 61.98 | 79.88 | 333 | Bills |
| 58 | 18 | Pat McQuistan | 74.61 | 61.84 | 78.95 | 140 | Cardinals |
| 59 | 19 | Sean Locklear | 74.54 | 62.48 | 78.42 | 647 | Giants |
| 60 | 20 | Bryan Bulaga | 74.45 | 62.49 | 78.26 | 577 | Packers |
| 61 | 21 | Zach Strief | 74.39 | 62.34 | 78.25 | 786 | Saints |
| 62 | 22 | Byron Stingily | 74.15 | 64.40 | 76.49 | 114 | Titans |
| 63 | 23 | Jermey Parnell | 74.03 | 62.93 | 77.26 | 261 | Cowboys |

### Starter (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 64 | 1 | Donald Stephenson | 73.65 | 61.21 | 77.78 | 362 | Chiefs |
| 65 | 2 | Frank Omiyale | 73.54 | 59.36 | 78.83 | 114 | Seahawks |
| 66 | 3 | David Diehl | 73.37 | 61.68 | 77.00 | 480 | Giants |
| 67 | 4 | Cameron Bradfield | 73.24 | 61.57 | 76.86 | 750 | Jaguars |
| 68 | 5 | Khalif Barnes | 73.18 | 60.53 | 77.44 | 558 | Raiders |
| 69 | 6 | Don Barclay | 73.16 | 60.20 | 77.64 | 447 | Packers |
| 70 | 7 | Tyler Polumbus | 73.15 | 61.60 | 76.69 | 992 | Commanders |
| 71 | 8 | Max Starks | 73.04 | 62.50 | 75.90 | 1065 | Steelers |
| 72 | 9 | Joe Barksdale | 73.04 | 60.44 | 77.27 | 118 | Rams |
| 73 | 10 | Byron Bell | 72.84 | 61.17 | 76.46 | 943 | Panthers |
| 74 | 11 | William Robinson | 72.31 | 61.15 | 75.58 | 178 | Saints |
| 75 | 12 | Jonathan Scott | 72.23 | 59.71 | 76.41 | 334 | Bears |
| 76 | 13 | Kevin Haslam | 72.13 | 59.57 | 76.33 | 278 | Chargers |
| 77 | 14 | Wayne Hunter | 71.87 | 60.23 | 75.47 | 334 | Rams |
| 78 | 15 | Dennis Kelly | 70.87 | 57.83 | 75.40 | 685 | Eagles |
| 79 | 16 | Jonathan Martin | 70.69 | 56.90 | 75.71 | 1032 | Dolphins |
| 80 | 17 | Erik Pears | 70.56 | 57.40 | 75.17 | 378 | Bills |
| 81 | 18 | Willie Smith | 69.16 | 54.13 | 75.02 | 504 | Raiders |
| 82 | 19 | Nate Potter | 68.23 | 56.51 | 71.88 | 428 | Cardinals |
| 83 | 20 | Bradley Sowell | 68.19 | 57.40 | 71.21 | 131 | Colts |
| 84 | 21 | Jordan Black | 64.70 | 56.04 | 66.30 | 106 | Commanders |
| 85 | 22 | Demetress Bell | 64.30 | 47.69 | 71.20 | 446 | Eagles |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Rob Gronkowski | 85.43 | 87.11 | 80.14 | 418 | Patriots |
| 2 | 2 | Tony Gonzalez | 82.24 | 89.20 | 73.43 | 725 | Falcons |
| 3 | 3 | Vernon Davis | 81.32 | 81.19 | 77.24 | 591 | 49ers |
| 4 | 4 | Heath Miller | 80.80 | 86.23 | 73.02 | 591 | Steelers |
| 5 | 5 | Jimmy Graham | 80.56 | 78.31 | 77.89 | 562 | Saints |
| 6 | 6 | Dwayne Allen | 80.30 | 82.68 | 74.54 | 506 | Colts |
| 7 | 7 | Jason Witten | 80.25 | 84.50 | 73.25 | 708 | Cowboys |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Jacob Tamme | 78.64 | 77.06 | 75.52 | 362 | Broncos |
| 9 | 2 | Zach Miller | 78.42 | 83.04 | 71.18 | 506 | Seahawks |
| 10 | 3 | Greg Olsen | 77.92 | 80.36 | 72.13 | 504 | Panthers |
| 11 | 4 | Joel Dreessen | 77.85 | 72.97 | 76.93 | 459 | Broncos |
| 12 | 5 | Marcedes Lewis | 77.72 | 77.17 | 73.92 | 474 | Jaguars |
| 13 | 6 | Martellus Bennett | 77.21 | 79.53 | 71.49 | 555 | Giants |
| 14 | 7 | Dennis Pitta | 76.06 | 76.64 | 71.51 | 534 | Ravens |
| 15 | 8 | Fred Davis | 75.92 | 71.12 | 74.96 | 190 | Commanders |
| 16 | 9 | Aaron Hernandez | 75.90 | 71.48 | 74.68 | 428 | Patriots |
| 17 | 10 | Owen Daniels | 75.33 | 74.28 | 71.87 | 585 | Texans |
| 18 | 11 | Scott Chandler | 74.93 | 70.97 | 73.41 | 456 | Bills |
| 19 | 12 | Daniel Fells | 74.69 | 66.24 | 76.15 | 124 | Patriots |
| 20 | 13 | Anthony Fasano | 74.63 | 70.20 | 73.41 | 448 | Dolphins |
| 21 | 14 | Brent Celek | 74.45 | 70.80 | 72.71 | 533 | Eagles |
| 22 | 15 | Benjamin Watson | 74.32 | 71.98 | 71.72 | 485 | Browns |
| 23 | 16 | Jeff Cumberland | 74.25 | 66.70 | 75.11 | 328 | Jets |
| 24 | 17 | Jermichael Finley | 74.24 | 66.74 | 75.07 | 547 | Packers |
| 25 | 18 | Michael Hoomanawanui | 74.12 | 66.92 | 74.76 | 135 | Patriots |

### Starter (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Kyle Rudolph | 73.83 | 74.08 | 69.50 | 511 | Vikings |
| 27 | 2 | Delanie Walker | 73.66 | 64.98 | 75.28 | 334 | 49ers |
| 28 | 3 | Antonio Gates | 73.38 | 65.23 | 74.65 | 555 | Chargers |
| 29 | 4 | Brandon Myers | 73.13 | 71.98 | 69.73 | 606 | Raiders |
| 30 | 5 | Garrett Graham | 72.99 | 71.58 | 69.76 | 333 | Texans |
| 31 | 6 | Dustin Keller | 72.52 | 64.18 | 73.91 | 235 | Jets |
| 32 | 7 | Tony Moeaki | 72.44 | 65.25 | 73.07 | 498 | Chiefs |
| 33 | 8 | Jared Cook | 71.83 | 65.39 | 71.95 | 380 | Titans |
| 34 | 9 | Lance Kendricks | 71.33 | 66.78 | 70.19 | 430 | Rams |
| 35 | 10 | Craig Stevens | 70.99 | 61.67 | 73.04 | 254 | Titans |
| 36 | 11 | Steve Maneri | 70.86 | 66.47 | 69.62 | 147 | Chiefs |
| 37 | 12 | Matthew Mulligan | 70.74 | 68.22 | 68.25 | 136 | Rams |
| 38 | 13 | Luke Stocker | 70.00 | 70.56 | 65.46 | 246 | Buccaneers |
| 39 | 14 | Will Heller | 69.92 | 64.69 | 69.24 | 216 | Lions |
| 40 | 15 | Charles Clay | 69.85 | 60.00 | 72.25 | 164 | Dolphins |
| 41 | 16 | Jeff King | 69.81 | 64.33 | 69.29 | 193 | Cardinals |
| 42 | 17 | Jordan Cameron | 69.54 | 61.21 | 70.93 | 199 | Browns |
| 43 | 18 | Jermaine Gresham | 69.43 | 65.04 | 68.19 | 635 | Bengals |
| 44 | 19 | Anthony McCoy | 69.22 | 64.51 | 68.19 | 245 | Seahawks |
| 45 | 20 | Logan Paulsen | 69.03 | 67.48 | 65.90 | 311 | Commanders |
| 46 | 21 | Dallas Clark | 69.01 | 67.37 | 65.93 | 414 | Buccaneers |
| 47 | 22 | Brandon Pettigrew | 68.95 | 60.71 | 70.27 | 525 | Lions |
| 48 | 23 | Randy McMichael | 68.68 | 63.19 | 68.17 | 133 | Chargers |
| 49 | 24 | Matt Spaeth | 68.44 | 68.09 | 64.51 | 138 | Bears |
| 50 | 25 | Tom Crabtree | 68.25 | 58.90 | 70.31 | 119 | Packers |
| 51 | 26 | Rob Housler | 67.91 | 61.63 | 67.93 | 399 | Cardinals |
| 52 | 27 | Coby Fleener | 67.90 | 64.33 | 66.12 | 312 | Colts |
| 53 | 28 | Tony Scheffler | 67.40 | 63.66 | 65.72 | 436 | Lions |
| 54 | 29 | Kellen Davis | 67.40 | 60.37 | 67.92 | 519 | Bears |
| 55 | 30 | Ed Dickson | 66.69 | 60.68 | 66.53 | 309 | Ravens |
| 56 | 31 | Clay Harbor | 65.15 | 60.90 | 63.82 | 209 | Eagles |
| 57 | 32 | John Phillips | 64.56 | 62.37 | 61.85 | 132 | Cowboys |
| 58 | 33 | Dave Thomas | 62.25 | 60.64 | 59.15 | 160 | Saints |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | David Paulson | 61.62 | 56.00 | 61.20 | 153 | Steelers |
| 60 | 2 | John Carlson | 61.39 | 58.51 | 59.15 | 126 | Vikings |

## WR — Wide Receiver

- **Season used:** `2012`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Andre Johnson | 88.20 | 91.44 | 81.88 | 649 | Texans |
| 2 | 2 | Vincent Jackson | 87.61 | 88.98 | 82.53 | 615 | Buccaneers |
| 3 | 3 | Calvin Johnson | 87.57 | 89.80 | 81.91 | 796 | Lions |
| 4 | 4 | A.J. Green | 85.49 | 88.73 | 79.17 | 647 | Bengals |
| 5 | 5 | Demaryius Thomas | 85.13 | 84.99 | 81.06 | 654 | Broncos |
| 6 | 6 | Julio Jones | 85.08 | 86.10 | 80.24 | 676 | Falcons |
| 7 | 7 | Brandon Marshall | 84.75 | 89.20 | 77.62 | 576 | Bears |
| 8 | 8 | Roddy White | 84.41 | 89.30 | 76.99 | 757 | Falcons |
| 9 | 9 | Michael Crabtree | 84.35 | 89.57 | 76.71 | 544 | 49ers |
| 10 | 10 | Percy Harvin | 83.93 | 83.87 | 79.80 | 258 | Vikings |
| 11 | 11 | Anquan Boldin | 82.97 | 84.20 | 77.98 | 705 | Ravens |
| 12 | 12 | Reggie Wayne | 82.88 | 88.60 | 74.90 | 795 | Colts |
| 13 | 13 | Steve Smith | 81.71 | 80.90 | 78.08 | 701 | Panthers |
| 14 | 14 | Golden Tate | 81.25 | 79.24 | 78.43 | 470 | Seahawks |
| 15 | 15 | Malcom Floyd | 81.21 | 78.00 | 79.18 | 549 | Chargers |
| 16 | 16 | Hakeem Nicks | 81.18 | 77.71 | 79.32 | 424 | Giants |
| 17 | 17 | Antonio Brown | 80.77 | 77.54 | 78.76 | 449 | Steelers |
| 18 | 18 | Lance Moore | 80.73 | 81.43 | 76.09 | 503 | Saints |
| 19 | 19 | Dwayne Bowe | 80.61 | 78.53 | 77.83 | 419 | Chiefs |
| 20 | 20 | Victor Cruz | 80.39 | 76.83 | 78.60 | 579 | Giants |
| 21 | 21 | Wes Welker | 80.27 | 82.10 | 74.89 | 751 | Patriots |
| 22 | 22 | Danario Alexander | 80.08 | 73.34 | 80.40 | 329 | Chargers |

### Good (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Cecil Shorts | 79.98 | 72.72 | 80.66 | 456 | Jaguars |
| 24 | 2 | Dez Bryant | 79.89 | 76.50 | 77.99 | 681 | Cowboys |
| 25 | 3 | Jordy Nelson | 79.68 | 74.16 | 79.19 | 484 | Packers |
| 26 | 4 | Torrey Smith | 79.58 | 75.60 | 78.06 | 705 | Ravens |
| 27 | 5 | Sidney Rice | 79.45 | 79.15 | 75.48 | 519 | Seahawks |
| 28 | 6 | Domenik Hixon | 79.31 | 74.04 | 78.66 | 279 | Giants |
| 29 | 7 | DeSean Jackson | 79.18 | 73.69 | 78.68 | 471 | Eagles |
| 30 | 8 | Danny Amendola | 78.85 | 82.05 | 72.55 | 342 | Rams |
| 31 | 9 | Marques Colston | 78.65 | 75.92 | 76.31 | 649 | Saints |
| 32 | 10 | Brian Hartline | 78.34 | 75.94 | 75.77 | 554 | Dolphins |
| 33 | 11 | Joe Morgan | 78.30 | 68.58 | 80.61 | 190 | Saints |
| 34 | 12 | Steve Johnson | 78.27 | 77.31 | 74.74 | 568 | Bills |
| 35 | 13 | Mike Williams | 78.20 | 78.25 | 74.00 | 597 | Buccaneers |
| 36 | 14 | Miles Austin | 77.94 | 75.36 | 75.50 | 601 | Cowboys |
| 37 | 15 | Chris Givens | 77.85 | 68.73 | 79.77 | 387 | Rams |
| 38 | 16 | Pierre Garcon | 77.67 | 74.65 | 75.51 | 238 | Commanders |
| 39 | 17 | Brandon Lloyd | 77.66 | 77.00 | 73.93 | 732 | Patriots |
| 40 | 18 | Rueben Randle | 77.40 | 68.15 | 79.40 | 169 | Giants |
| 41 | 19 | Mario Manningham | 77.33 | 74.22 | 75.24 | 261 | 49ers |
| 42 | 20 | Damaris Johnson | 77.24 | 68.85 | 78.67 | 165 | Eagles |
| 43 | 21 | Greg Jennings | 77.17 | 70.92 | 77.17 | 377 | Packers |
| 44 | 22 | Randall Cobb | 77.10 | 76.36 | 73.43 | 479 | Packers |
| 45 | 23 | Josh Gordon | 76.99 | 68.46 | 78.51 | 523 | Browns |
| 46 | 24 | Randy Moss | 76.98 | 70.13 | 77.38 | 293 | 49ers |
| 47 | 25 | Jason Avant | 76.43 | 75.51 | 72.87 | 459 | Eagles |
| 48 | 26 | Mike Wallace | 76.31 | 66.99 | 78.36 | 571 | Steelers |
| 49 | 27 | Leonard Hankerson | 76.18 | 70.57 | 75.76 | 328 | Commanders |
| 50 | 28 | Larry Fitzgerald | 75.97 | 69.40 | 76.18 | 701 | Cardinals |
| 51 | 29 | James Jones | 75.91 | 73.20 | 73.55 | 744 | Packers |
| 52 | 30 | Brandon LaFell | 75.79 | 68.44 | 76.53 | 470 | Panthers |
| 53 | 31 | Brandon Stokley | 75.53 | 73.33 | 72.83 | 447 | Broncos |
| 54 | 32 | Eric Decker | 75.49 | 76.60 | 70.58 | 676 | Broncos |
| 55 | 33 | Jeremy Kerley | 75.39 | 66.30 | 77.28 | 466 | Jets |
| 56 | 34 | Doug Baldwin | 75.31 | 70.16 | 74.57 | 303 | Seahawks |
| 57 | 35 | Aldrick Robinson | 75.01 | 63.76 | 78.34 | 108 | Commanders |
| 58 | 36 | Denarius Moore | 74.89 | 67.63 | 75.56 | 520 | Raiders |
| 59 | 37 | Earl Bennett | 74.80 | 64.88 | 77.24 | 295 | Bears |
| 60 | 38 | Jacoby Jones | 74.78 | 69.42 | 74.18 | 375 | Ravens |
| 61 | 39 | Ryan Broyles | 74.62 | 66.35 | 75.96 | 191 | Lions |
| 62 | 40 | Emmanuel Sanders | 74.45 | 67.43 | 74.97 | 440 | Steelers |
| 63 | 41 | Davone Bess | 74.44 | 72.94 | 71.28 | 456 | Dolphins |
| 64 | 42 | Darrius Heyward-Bey | 74.13 | 68.37 | 73.80 | 536 | Raiders |
| 65 | 43 | T.Y. Hilton | 74.07 | 65.05 | 75.92 | 557 | Colts |

### Starter (70 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 66 | 1 | Brandon Gibson | 73.85 | 74.71 | 69.11 | 500 | Rams |
| 67 | 2 | Brandon Tate | 73.73 | 61.35 | 77.82 | 179 | Bengals |
| 68 | 3 | Santonio Holmes | 73.56 | 67.59 | 73.38 | 135 | Jets |
| 69 | 4 | Nate Washington | 73.44 | 64.20 | 75.43 | 563 | Titans |
| 70 | 5 | Santana Moss | 73.23 | 67.20 | 73.08 | 350 | Commanders |
| 71 | 6 | Dwayne Harris | 73.18 | 66.56 | 73.42 | 197 | Cowboys |
| 72 | 7 | Alshon Jeffery | 72.76 | 63.92 | 74.48 | 282 | Bears |
| 73 | 8 | Ramses Barden | 72.72 | 62.33 | 75.48 | 155 | Giants |
| 74 | 9 | Travis Benjamin | 72.69 | 63.37 | 74.74 | 223 | Browns |
| 75 | 10 | Jarius Wright | 72.60 | 62.67 | 75.05 | 161 | Vikings |
| 76 | 11 | Rod Streater | 72.31 | 65.80 | 72.48 | 405 | Raiders |
| 77 | 12 | Justin Blackmon | 72.23 | 65.70 | 72.41 | 683 | Jaguars |
| 78 | 13 | Kenny Britt | 72.22 | 61.98 | 74.88 | 415 | Titans |
| 79 | 14 | Devin Aromashodu | 72.07 | 62.38 | 74.37 | 234 | Vikings |
| 80 | 15 | Lestar Jean | 71.95 | 60.64 | 75.32 | 103 | Texans |
| 81 | 16 | Mohamed Sanu | 71.84 | 64.88 | 72.31 | 115 | Bengals |
| 82 | 17 | Michael Jenkins | 71.73 | 65.40 | 71.79 | 487 | Vikings |
| 83 | 18 | Andrew Hawkins | 71.68 | 66.13 | 71.22 | 430 | Bengals |
| 84 | 19 | Josh Morgan | 70.96 | 66.49 | 69.77 | 414 | Commanders |
| 85 | 20 | LaVon Brazill | 70.77 | 58.73 | 74.63 | 185 | Colts |
| 86 | 21 | Jeremy Maclin | 70.68 | 62.10 | 72.24 | 661 | Eagles |
| 87 | 22 | Rishard Matthews | 70.52 | 60.05 | 73.34 | 160 | Dolphins |
| 88 | 23 | Louis Murphy Jr. | 70.45 | 60.48 | 72.93 | 420 | Panthers |
| 89 | 24 | Kevin Ogletree | 70.33 | 63.03 | 71.03 | 361 | Cowboys |
| 90 | 25 | Mohamed Massaquoi | 70.25 | 62.99 | 70.92 | 170 | Browns |
| 91 | 26 | Kendall Wright | 70.19 | 66.12 | 68.73 | 417 | Titans |
| 92 | 27 | Kyle Williams | 70.09 | 60.14 | 72.56 | 153 | 49ers |
| 93 | 28 | Juron Criner | 70.05 | 65.46 | 68.94 | 128 | Raiders |
| 94 | 29 | Jonathan Baldwin | 69.89 | 60.83 | 71.77 | 312 | Chiefs |
| 95 | 30 | Tandon Doss | 69.75 | 58.59 | 73.03 | 167 | Ravens |
| 96 | 31 | Tiquan Underwood | 69.73 | 62.02 | 70.71 | 343 | Buccaneers |
| 97 | 32 | Damian Williams | 69.66 | 62.32 | 70.38 | 260 | Titans |
| 98 | 33 | Michael Floyd | 69.64 | 62.51 | 70.22 | 432 | Cardinals |
| 99 | 34 | Marlon Moore | 69.60 | 59.40 | 72.24 | 100 | Dolphins |
| 100 | 35 | Matt Willis | 69.36 | 62.26 | 69.93 | 111 | Broncos |
| 101 | 36 | Derek Hagan | 69.28 | 64.44 | 68.34 | 180 | Raiders |
| 102 | 37 | Jordan Shipley | 69.19 | 61.69 | 70.03 | 197 | Jaguars |
| 103 | 38 | Kris Durham | 69.16 | 60.67 | 70.66 | 153 | Lions |
| 104 | 39 | Kevin Walter | 69.10 | 62.84 | 69.10 | 555 | Texans |
| 105 | 40 | Jerome Simpson | 69.04 | 61.55 | 69.86 | 299 | Vikings |
| 106 | 41 | Eddie Royal | 68.93 | 63.17 | 68.60 | 204 | Chargers |
| 107 | 42 | Brian Quick | 68.76 | 59.65 | 70.66 | 124 | Rams |
| 108 | 43 | Robert Meachem | 68.54 | 57.30 | 71.87 | 261 | Chargers |
| 109 | 44 | Jerricho Cotchery | 68.50 | 62.43 | 68.38 | 156 | Steelers |
| 110 | 45 | Julian Edelman | 68.39 | 61.56 | 68.77 | 191 | Patriots |
| 111 | 46 | Greg Little | 68.23 | 64.95 | 66.25 | 576 | Browns |
| 112 | 47 | Deion Branch | 68.23 | 54.55 | 73.18 | 353 | Patriots |
| 113 | 48 | Steve Breaston | 68.15 | 55.73 | 72.27 | 159 | Chiefs |
| 114 | 49 | Austin Pettis | 67.89 | 65.95 | 65.01 | 282 | Rams |
| 115 | 50 | Donald Jones | 67.83 | 64.26 | 66.05 | 410 | Bills |
| 116 | 51 | Riley Cooper | 67.78 | 60.43 | 68.52 | 335 | Eagles |
| 117 | 52 | Nate Burleson | 67.73 | 62.08 | 67.33 | 248 | Lions |
| 118 | 53 | Andre Roberts | 67.55 | 61.36 | 67.51 | 623 | Cardinals |
| 119 | 54 | Brad Smith | 67.53 | 62.55 | 66.68 | 179 | Bills |
| 120 | 55 | Armon Binns | 67.43 | 57.97 | 69.57 | 265 | Dolphins |
| 121 | 56 | Chaz Schilens | 67.10 | 61.92 | 66.39 | 253 | Jets |
| 122 | 57 | Harry Douglas | 66.57 | 58.79 | 67.59 | 497 | Falcons |
| 123 | 58 | Donnie Avery | 66.32 | 60.10 | 66.30 | 750 | Colts |
| 124 | 59 | Marvin Jones Jr. | 66.15 | 61.33 | 65.19 | 266 | Bengals |
| 125 | 60 | Devin Hester | 65.99 | 56.91 | 67.87 | 262 | Bears |
| 126 | 61 | Titus Young | 65.87 | 58.88 | 66.36 | 422 | Lions |
| 127 | 62 | DeVier Posey | 65.87 | 57.90 | 67.01 | 144 | Texans |
| 128 | 63 | Devery Henderson | 65.29 | 51.94 | 70.03 | 475 | Saints |
| 129 | 64 | Cole Beasley | 65.14 | 61.18 | 63.61 | 118 | Cowboys |
| 130 | 65 | Donald Driver | 64.80 | 57.18 | 65.71 | 117 | Packers |
| 131 | 66 | Laurent Robinson | 64.72 | 58.68 | 64.58 | 199 | Jaguars |
| 132 | 67 | Micheal Spurlock | 64.64 | 61.20 | 62.76 | 164 | Chargers |
| 133 | 68 | Stephen Hill | 62.96 | 57.77 | 62.25 | 283 | Jets |
| 134 | 69 | Kevin Elliott | 62.16 | 60.00 | 59.43 | 160 | Bills |
| 135 | 70 | T.J. Graham | 62.14 | 53.87 | 63.48 | 429 | Bills |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 136 | 1 | Early Doucet | 61.80 | 51.03 | 64.82 | 319 | Cardinals |
| 137 | 2 | Mike Thomas | 61.24 | 49.77 | 64.72 | 311 | Lions |
| 138 | 3 | Keshawn Martin | 59.48 | 55.91 | 57.70 | 168 | Texans |
