# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:37:47Z
- **Requested analysis_year:** 2022 (clamped to 2022)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Creed Humphrey | 96.02 | 90.00 | 95.87 | 1138 | Chiefs |
| 2 | 2 | Jason Kelce | 95.05 | 88.30 | 95.38 | 1149 | Eagles |
| 3 | 3 | Connor Williams | 88.27 | 78.20 | 90.81 | 1056 | Dolphins |
| 4 | 4 | Frank Ragnow | 86.16 | 77.90 | 87.50 | 1074 | Lions |
| 5 | 5 | Ethan Pocic | 85.95 | 77.23 | 87.59 | 819 | Browns |
| 6 | 6 | Tyler Linderbaum | 84.75 | 74.70 | 87.28 | 1092 | Ravens |
| 7 | 7 | David Andrews | 83.21 | 73.64 | 85.43 | 799 | Patriots |
| 8 | 8 | Corey Linsley | 82.67 | 73.63 | 84.53 | 858 | Chargers |
| 9 | 9 | Ben Jones | 81.40 | 70.70 | 84.37 | 682 | Titans |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Garrett Bradbury | 79.82 | 69.24 | 82.71 | 809 | Vikings |
| 11 | 2 | Connor McGovern | 79.51 | 69.60 | 81.95 | 1111 | Jets |
| 12 | 3 | Mason Cole | 77.24 | 67.10 | 79.84 | 1114 | Steelers |
| 13 | 4 | Jake Brendel | 77.01 | 64.90 | 80.92 | 1078 | 49ers |
| 14 | 5 | Ryan Kelly | 75.32 | 64.30 | 78.50 | 1092 | Colts |
| 15 | 6 | Bradley Bozeman | 74.75 | 62.79 | 78.56 | 680 | Panthers |
| 16 | 7 | Brian Allen | 74.64 | 62.92 | 78.28 | 373 | Rams |
| 17 | 8 | Sam Mustipher | 74.64 | 63.40 | 77.97 | 1020 | Bears |
| 18 | 9 | Corey Levin | 74.36 | 64.19 | 76.98 | 251 | Titans |

### Starter (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Erik McCoy | 73.57 | 61.13 | 77.69 | 798 | Saints |
| 20 | 2 | Tyler Biadasz | 73.55 | 61.70 | 77.28 | 1066 | Cowboys |
| 21 | 3 | Drew Dalman | 73.30 | 65.90 | 74.07 | 1051 | Falcons |
| 22 | 4 | Robert Hainsey | 73.27 | 66.70 | 73.49 | 1175 | Buccaneers |
| 23 | 5 | Ted Karras | 73.25 | 62.60 | 76.19 | 1100 | Bengals |
| 24 | 6 | Andre James | 73.18 | 62.77 | 75.96 | 964 | Raiders |
| 25 | 7 | Mitch Morse | 72.29 | 61.30 | 75.45 | 765 | Bills |
| 26 | 8 | Pat Elflein | 71.90 | 60.53 | 75.31 | 338 | Panthers |
| 27 | 9 | Josh Myers | 71.87 | 60.40 | 75.35 | 1091 | Packers |
| 28 | 10 | Jon Feliciano | 70.75 | 58.21 | 74.95 | 971 | Giants |
| 29 | 11 | Rodney Hudson | 70.74 | 58.75 | 74.56 | 303 | Cardinals |
| 30 | 12 | Tyler Larsen | 70.55 | 58.67 | 74.30 | 534 | Commanders |
| 31 | 13 | Lloyd Cushenberry III | 69.70 | 57.29 | 73.80 | 502 | Broncos |
| 32 | 14 | Chase Roullier | 69.32 | 57.63 | 72.94 | 150 | Commanders |
| 33 | 15 | Will Clapp | 68.53 | 56.69 | 72.25 | 333 | Chargers |
| 34 | 16 | James Ferentz | 68.51 | 56.56 | 72.31 | 268 | Patriots |
| 35 | 17 | Nick Martin | 66.90 | 53.04 | 71.97 | 156 | Commanders |
| 36 | 18 | Josh Andrews | 66.76 | 55.14 | 70.34 | 330 | Saints |
| 37 | 19 | Austin Blythe | 66.12 | 51.90 | 71.44 | 1041 | Seahawks |
| 38 | 20 | Luke Fortner | 64.29 | 49.60 | 69.91 | 1121 | Jaguars |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Scott Quessenberry | 58.83 | 40.00 | 67.22 | 990 | Texans |

## CB — Cornerback

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Sauce Gardner | 94.37 | 90.00 | 93.11 | 1114 | Jets |
| 2 | 2 | Pat Surtain II | 89.48 | 86.70 | 87.53 | 1104 | Broncos |
| 3 | 3 | Patrick Peterson | 85.18 | 82.50 | 83.99 | 1104 | Vikings |
| 4 | 4 | Tyson Campbell | 84.71 | 81.20 | 83.61 | 1138 | Jaguars |
| 5 | 5 | Tariq Woolen | 84.35 | 77.80 | 84.55 | 1135 | Seahawks |
| 6 | 6 | Jaire Alexander | 83.96 | 82.10 | 85.55 | 901 | Packers |
| 7 | 7 | James Bradberry | 83.85 | 80.20 | 82.31 | 1077 | Eagles |
| 8 | 8 | Charvarius Ward | 83.57 | 78.30 | 84.52 | 959 | 49ers |
| 9 | 9 | Jalen Ramsey | 83.44 | 77.80 | 83.54 | 1078 | Rams |
| 10 | 10 | Stephon Gilmore | 83.35 | 81.10 | 84.86 | 1064 | Colts |
| 11 | 11 | Darius Slay | 82.25 | 77.40 | 81.81 | 1002 | Eagles |
| 12 | 12 | Jamel Dean | 81.78 | 75.60 | 83.72 | 884 | Buccaneers |
| 13 | 13 | Tyler Hall | 81.66 | 73.52 | 90.54 | 218 | Raiders |
| 14 | 14 | Kendall Fuller | 81.28 | 75.40 | 81.75 | 1030 | Commanders |
| 15 | 15 | Marlon Humphrey | 80.55 | 75.60 | 81.37 | 1051 | Ravens |
| 16 | 16 | D.J. Reed | 80.37 | 77.50 | 80.45 | 1135 | Jets |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Michael Davis | 79.63 | 73.21 | 82.10 | 790 | Chargers |
| 18 | 2 | Duke Shelley | 79.60 | 77.29 | 85.13 | 398 | Vikings |
| 19 | 3 | Martin Emerson Jr. | 79.18 | 74.71 | 77.99 | 783 | Browns |
| 20 | 4 | Isaiah Rodgers | 78.86 | 75.59 | 81.13 | 434 | Colts |
| 21 | 5 | Taron Johnson | 78.12 | 74.70 | 77.02 | 969 | Bills |
| 22 | 6 | Rasul Douglas | 77.73 | 71.50 | 79.60 | 915 | Packers |
| 23 | 7 | Trevon Diggs | 77.27 | 66.10 | 81.69 | 1115 | Cowboys |
| 24 | 8 | L'Jarius Sneed | 77.20 | 74.00 | 77.21 | 1106 | Chiefs |
| 25 | 9 | Jack Jones | 77.17 | 71.60 | 80.63 | 434 | Patriots |
| 26 | 10 | Greg Newsome II | 76.60 | 72.30 | 78.37 | 907 | Browns |
| 27 | 11 | DaRon Bland | 76.44 | 72.67 | 78.71 | 597 | Cowboys |
| 28 | 12 | Mike Hilton | 76.40 | 71.72 | 77.65 | 701 | Bengals |
| 29 | 13 | Cameron Sutton | 76.07 | 70.40 | 76.46 | 931 | Steelers |
| 30 | 14 | Danny Johnson | 75.88 | 71.96 | 83.62 | 292 | Commanders |
| 31 | 15 | Desmond King II | 75.86 | 71.80 | 74.90 | 916 | Texans |
| 32 | 16 | Steven Nelson | 75.76 | 72.00 | 75.58 | 957 | Texans |
| 33 | 17 | Sean Murphy-Bunting | 75.70 | 75.09 | 78.21 | 430 | Buccaneers |
| 34 | 18 | Marcus Peters | 75.55 | 69.18 | 78.02 | 734 | Ravens |
| 35 | 19 | Michael Carter II | 75.45 | 70.55 | 75.29 | 732 | Jets |
| 36 | 20 | Trent McDuffie | 75.09 | 74.56 | 77.16 | 683 | Chiefs |
| 37 | 21 | Marshon Lattimore | 74.86 | 68.34 | 80.65 | 415 | Saints |
| 38 | 22 | Tavierre Thomas | 74.86 | 72.34 | 78.06 | 409 | Texans |
| 39 | 23 | Chidobe Awuzie | 74.49 | 71.18 | 79.49 | 471 | Bengals |
| 40 | 24 | Jaycee Horn | 74.42 | 73.10 | 78.73 | 812 | Panthers |
| 41 | 25 | Kader Kohou | 74.08 | 70.40 | 74.34 | 895 | Dolphins |

### Starter (68 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 42 | 1 | Asante Samuel Jr. | 73.46 | 65.40 | 76.50 | 971 | Chargers |
| 43 | 2 | Jonathan Jones | 73.42 | 67.90 | 76.67 | 914 | Patriots |
| 44 | 3 | Adoree' Jackson | 73.12 | 70.25 | 78.18 | 554 | Giants |
| 45 | 4 | Isaiah Oliver | 72.76 | 67.93 | 78.08 | 349 | Falcons |
| 46 | 5 | Darious Williams | 72.73 | 63.60 | 75.54 | 944 | Jaguars |
| 47 | 6 | Cobie Durant | 72.40 | 68.35 | 82.24 | 281 | Rams |
| 48 | 7 | Emmanuel Moseley | 72.39 | 68.30 | 79.44 | 312 | 49ers |
| 49 | 8 | Marcus Jones | 72.21 | 64.56 | 79.03 | 371 | Patriots |
| 50 | 9 | Samuel Womack III | 72.13 | 65.43 | 80.28 | 146 | 49ers |
| 51 | 10 | Nick McCloud | 71.78 | 63.71 | 76.06 | 537 | Giants |
| 52 | 11 | Bryce Callahan | 71.65 | 66.70 | 74.79 | 585 | Chargers |
| 53 | 12 | Avonte Maddox | 71.23 | 68.72 | 74.21 | 457 | Eagles |
| 54 | 13 | Carlton Davis III | 71.16 | 63.98 | 76.21 | 809 | Buccaneers |
| 55 | 14 | Jaylon Johnson | 71.10 | 64.64 | 75.39 | 656 | Bears |
| 56 | 15 | Casey Hayward Jr. | 70.63 | 64.44 | 76.40 | 355 | Falcons |
| 57 | 16 | Terrance Mitchell | 70.60 | 61.46 | 76.35 | 397 | Titans |
| 58 | 17 | Antonio Hamilton Sr. | 70.57 | 66.71 | 75.84 | 420 | Cardinals |
| 59 | 18 | Nik Needham | 70.26 | 63.88 | 76.17 | 294 | Dolphins |
| 60 | 19 | James Pierre | 70.19 | 64.32 | 76.64 | 260 | Steelers |
| 61 | 20 | A.J. Terrell | 69.90 | 61.78 | 73.33 | 800 | Falcons |
| 62 | 21 | Cor'Dale Flott | 69.89 | 66.82 | 74.63 | 335 | Giants |
| 63 | 22 | Rock Ya-Sin | 69.75 | 65.44 | 73.21 | 663 | Raiders |
| 64 | 23 | Damarri Mathis | 69.70 | 65.20 | 71.47 | 794 | Broncos |
| 65 | 24 | K'Waun Williams | 69.60 | 65.91 | 71.91 | 596 | Broncos |
| 66 | 25 | Levi Wallace | 69.57 | 59.33 | 74.04 | 709 | Steelers |
| 67 | 26 | Ronald Darby | 69.28 | 64.72 | 75.81 | 280 | Broncos |
| 68 | 27 | Cameron Dantzler | 69.11 | 63.68 | 74.22 | 505 | Vikings |
| 69 | 28 | Marco Wilson | 69.01 | 61.17 | 73.00 | 778 | Cardinals |
| 70 | 29 | Roger McCreary | 69.01 | 60.40 | 70.58 | 1164 | Titans |
| 71 | 30 | Denzel Ward | 68.90 | 60.39 | 73.30 | 748 | Browns |
| 72 | 31 | Dane Jackson | 68.80 | 59.90 | 73.26 | 830 | Bills |
| 73 | 32 | Byron Murphy Jr. | 68.69 | 63.31 | 72.53 | 595 | Cardinals |
| 74 | 33 | Tre'Davious White | 68.00 | 61.00 | 76.07 | 307 | Bills |
| 75 | 34 | Troy Hill | 67.60 | 60.48 | 72.09 | 703 | Rams |
| 76 | 35 | Xavien Howard | 67.33 | 55.20 | 72.52 | 973 | Dolphins |
| 77 | 36 | Shaquill Griffin | 67.21 | 61.49 | 74.45 | 336 | Jaguars |
| 78 | 37 | Tre Flowers | 66.95 | 59.22 | 71.14 | 172 | Bengals |
| 79 | 38 | Ja'Sir Taylor | 66.09 | 62.47 | 73.16 | 161 | Chargers |
| 80 | 39 | Tremon Smith | 66.08 | 62.96 | 76.50 | 201 | Texans |
| 81 | 40 | Cam Taylor-Britt | 66.06 | 60.42 | 72.52 | 590 | Bengals |
| 82 | 41 | Myles Bryant | 66.03 | 56.25 | 69.85 | 689 | Patriots |
| 83 | 42 | Joshua Williams | 65.54 | 62.55 | 67.28 | 437 | Chiefs |
| 84 | 43 | Eric Stokes | 65.26 | 60.23 | 69.71 | 477 | Packers |
| 85 | 44 | Kristian Fulton | 65.06 | 58.67 | 71.36 | 652 | Titans |
| 86 | 45 | Dee Delaney | 64.72 | 63.89 | 70.31 | 216 | Buccaneers |
| 87 | 46 | Mike Jackson | 64.71 | 58.60 | 72.15 | 1082 | Seahawks |
| 88 | 47 | Kaiir Elam | 64.45 | 59.01 | 68.81 | 477 | Bills |
| 89 | 48 | Brandon Facyson | 64.43 | 57.03 | 69.74 | 455 | Colts |
| 90 | 49 | Benjamin St-Juste | 64.20 | 60.00 | 69.57 | 655 | Commanders |
| 91 | 50 | Christian Benford | 64.15 | 59.20 | 71.13 | 363 | Bills |
| 92 | 51 | Fabian Moreau | 63.91 | 53.90 | 69.02 | 749 | Giants |
| 93 | 52 | Coby Bryant | 63.83 | 54.83 | 65.67 | 757 | Seahawks |
| 94 | 53 | Alontae Taylor | 63.82 | 56.68 | 70.29 | 663 | Saints |
| 95 | 54 | Donte Jackson | 63.55 | 56.49 | 69.90 | 442 | Panthers |
| 96 | 55 | Rashad Fenton | 63.47 | 56.93 | 70.64 | 379 | Falcons |
| 97 | 56 | Jaylen Watson | 63.44 | 56.15 | 65.11 | 604 | Chiefs |
| 98 | 57 | Anthony Brown | 63.44 | 54.93 | 68.95 | 728 | Cowboys |
| 99 | 58 | Jourdan Lewis | 63.41 | 57.47 | 69.11 | 315 | Cowboys |
| 100 | 59 | Jerry Jacobs | 63.27 | 55.54 | 70.14 | 542 | Lions |
| 101 | 60 | Chandon Sullivan | 63.22 | 53.20 | 65.74 | 944 | Vikings |
| 102 | 61 | Eli Apple | 63.21 | 53.50 | 67.00 | 908 | Bengals |
| 103 | 62 | Arthur Maulet | 63.10 | 56.26 | 64.08 | 481 | Steelers |
| 104 | 63 | Sidney Jones IV | 63.03 | 53.84 | 71.77 | 109 | Raiders |
| 105 | 64 | Rodarius Williams | 62.87 | 61.44 | 72.16 | 147 | Giants |
| 106 | 65 | Keisean Nixon | 62.66 | 60.24 | 67.36 | 290 | Packers |
| 107 | 66 | Harrison Hand | 62.47 | 58.00 | 73.79 | 111 | Bears |
| 108 | 67 | Nate Hobbs | 62.28 | 57.28 | 65.86 | 667 | Raiders |
| 109 | 68 | Cornell Armstrong | 62.12 | 55.97 | 68.45 | 372 | Falcons |

### Rotation/backup (48 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 110 | 1 | Zech McPhearson | 61.56 | 56.00 | 69.68 | 103 | Eagles |
| 111 | 2 | Amik Robertson | 61.53 | 61.18 | 62.91 | 677 | Raiders |
| 112 | 3 | Mike Hughes | 61.52 | 52.30 | 67.97 | 561 | Lions |
| 113 | 4 | Greedy Williams | 61.48 | 55.84 | 66.55 | 105 | Browns |
| 114 | 5 | David Long Jr. | 61.28 | 56.58 | 65.18 | 287 | Rams |
| 115 | 6 | Josiah Scott | 61.21 | 61.58 | 65.45 | 390 | Eagles |
| 116 | 7 | A.J. Green III | 61.04 | 52.45 | 70.08 | 142 | Browns |
| 117 | 8 | Grant Haley | 60.97 | 63.36 | 70.83 | 129 | Rams |
| 118 | 9 | Darren Hall | 60.93 | 60.09 | 60.14 | 633 | Falcons |
| 119 | 10 | Sam Webb | 60.90 | 58.11 | 66.45 | 327 | Raiders |
| 120 | 11 | Darnay Holmes | 60.80 | 51.26 | 66.57 | 572 | Giants |
| 121 | 12 | Justin Bethel | 60.77 | 57.16 | 69.67 | 125 | Dolphins |
| 122 | 13 | Bradley Roby | 60.51 | 50.40 | 67.19 | 628 | Saints |
| 123 | 14 | Kelvin Joseph | 60.37 | 52.85 | 68.94 | 167 | Cowboys |
| 124 | 15 | Trayvon Mullen | 60.15 | 55.11 | 67.77 | 160 | Cowboys |
| 125 | 16 | Greg Mabin | 60.14 | 58.80 | 68.56 | 108 | Titans |
| 126 | 17 | Anthony Averett | 59.98 | 54.78 | 66.31 | 278 | Raiders |
| 127 | 18 | Keion Crossen | 59.92 | 53.33 | 66.33 | 382 | Dolphins |
| 128 | 19 | Paulson Adebo | 59.81 | 48.68 | 65.52 | 814 | Saints |
| 129 | 20 | Kenny Moore II | 59.80 | 46.56 | 66.91 | 774 | Colts |
| 130 | 21 | Tre Herndon | 59.70 | 58.01 | 62.85 | 416 | Jaguars |
| 131 | 22 | Derek Stingley Jr. | 59.50 | 51.39 | 68.59 | 599 | Texans |
| 132 | 23 | Kindle Vildor | 59.46 | 59.04 | 61.10 | 531 | Bears |
| 133 | 24 | CJ Henderson | 59.42 | 49.31 | 65.13 | 765 | Panthers |
| 134 | 25 | Jalen Mills | 59.40 | 46.97 | 68.71 | 468 | Patriots |
| 135 | 26 | Chris Harris Jr. | 59.14 | 50.18 | 66.72 | 375 | Saints |
| 136 | 27 | Rachad Wildgoose | 58.92 | 57.03 | 67.67 | 196 | Commanders |
| 137 | 28 | Deommodore Lenoir | 58.62 | 51.00 | 63.82 | 887 | 49ers |
| 138 | 29 | Essang Bassey | 58.37 | 55.69 | 62.63 | 222 | Broncos |
| 139 | 30 | Derion Kendrick | 57.97 | 48.37 | 65.10 | 483 | Rams |
| 140 | 31 | Josh Blackwell | 57.69 | 62.58 | 66.53 | 134 | Bears |
| 141 | 32 | Kyler Gordon | 57.36 | 46.40 | 63.44 | 863 | Bears |
| 142 | 33 | Keith Taylor Jr. | 57.34 | 54.65 | 56.56 | 378 | Panthers |
| 143 | 34 | Nahshon Wright | 56.92 | 54.96 | 65.82 | 128 | Cowboys |
| 144 | 35 | Jeff Okudah | 56.47 | 54.52 | 60.76 | 789 | Lions |
| 145 | 36 | Christian Matthew | 56.34 | 53.94 | 61.62 | 237 | Cardinals |
| 146 | 37 | Jace Whittaker | 56.02 | 55.04 | 66.50 | 281 | Cardinals |
| 147 | 38 | Noah Igbinoghene | 55.83 | 53.23 | 63.66 | 238 | Dolphins |
| 148 | 39 | Amani Oruwariye | 55.66 | 40.00 | 64.78 | 474 | Lions |
| 149 | 40 | J.C. Jackson | 55.63 | 40.00 | 67.77 | 244 | Chargers |
| 150 | 41 | Ahkello Witherspoon | 55.28 | 47.46 | 66.51 | 248 | Steelers |
| 151 | 42 | Zyon McCollum | 52.34 | 55.01 | 54.25 | 278 | Buccaneers |
| 152 | 43 | Damarion Williams | 51.11 | 50.08 | 52.53 | 225 | Ravens |
| 153 | 44 | Akayleb Evans | 50.90 | 51.36 | 57.73 | 162 | Vikings |
| 154 | 45 | Caleb Farley | 46.33 | 49.88 | 52.29 | 104 | Titans |
| 155 | 46 | Andrew Booth Jr. | 45.44 | 52.68 | 57.94 | 105 | Vikings |
| 156 | 47 | Dallis Flowers | 45.00 | 55.99 | 50.68 | 175 | Colts |
| 157 | 48 | Christian Holmes | 45.00 | 55.56 | 50.00 | 104 | Commanders |

## DI — Defensive Interior

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 89.39 | 84.34 | 91.53 | 615 | Rams |
| 2 | 2 | Quinnen Williams | 88.73 | 88.34 | 86.53 | 690 | Jets |
| 3 | 3 | Jonathan Allen | 86.42 | 86.65 | 82.58 | 802 | Commanders |
| 4 | 4 | DeForest Buckner | 86.12 | 87.68 | 81.11 | 874 | Colts |
| 5 | 5 | Christian Wilkins | 86.09 | 88.45 | 80.76 | 952 | Dolphins |
| 6 | 6 | J.J. Watt | 85.67 | 72.81 | 95.79 | 816 | Cardinals |
| 7 | 7 | Cameron Heyward | 84.83 | 82.67 | 82.31 | 801 | Steelers |
| 8 | 8 | Dexter Lawrence | 84.39 | 88.24 | 78.44 | 864 | Giants |
| 9 | 9 | Chris Jones | 84.15 | 88.78 | 77.98 | 915 | Chiefs |
| 10 | 10 | Zach Sieler | 83.92 | 78.32 | 83.48 | 874 | Dolphins |
| 11 | 11 | Jeffery Simmons | 82.96 | 85.51 | 78.27 | 840 | Titans |
| 12 | 12 | Leonard Williams | 82.14 | 84.51 | 78.84 | 604 | Giants |
| 13 | 13 | Grady Jarrett | 81.90 | 78.13 | 80.24 | 856 | Falcons |
| 14 | 14 | Derrick Brown | 80.74 | 85.64 | 73.60 | 870 | Panthers |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Javon Hargrave | 79.66 | 72.65 | 80.66 | 711 | Eagles |
| 16 | 2 | Ed Oliver | 79.36 | 71.41 | 82.46 | 526 | Bills |
| 17 | 3 | Daron Payne | 78.52 | 72.77 | 78.19 | 907 | Commanders |
| 18 | 4 | Kenny Clark | 78.44 | 74.10 | 78.08 | 807 | Packers |
| 19 | 5 | Calais Campbell | 78.36 | 67.29 | 84.47 | 548 | Ravens |
| 20 | 6 | Dalvin Tomlinson | 77.80 | 77.60 | 76.02 | 550 | Vikings |
| 21 | 7 | DeMarcus Walker | 77.03 | 61.29 | 85.54 | 427 | Titans |
| 22 | 8 | B.J. Hill | 76.04 | 70.42 | 76.40 | 815 | Bengals |
| 23 | 9 | DJ Reader | 75.52 | 79.76 | 74.85 | 397 | Bengals |
| 24 | 10 | Milton Williams | 75.44 | 61.41 | 80.63 | 396 | Eagles |
| 25 | 11 | Shelby Harris | 75.00 | 64.43 | 80.20 | 560 | Seahawks |
| 26 | 12 | Vita Vea | 74.22 | 71.47 | 75.94 | 538 | Buccaneers |

### Starter (89 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | Grover Stewart | 73.29 | 66.16 | 73.88 | 781 | Colts |
| 28 | 2 | DaVon Hamilton | 72.99 | 66.82 | 74.27 | 610 | Jaguars |
| 29 | 3 | David Onyemata | 72.61 | 65.39 | 75.23 | 682 | Saints |
| 30 | 4 | Christian Barmore | 72.55 | 63.25 | 78.86 | 327 | Patriots |
| 31 | 5 | D.J. Jones | 72.49 | 60.06 | 78.01 | 558 | Broncos |
| 32 | 6 | Andrew Billings | 72.33 | 68.68 | 72.07 | 478 | Raiders |
| 33 | 7 | Poona Ford | 72.26 | 62.35 | 74.70 | 642 | Seahawks |
| 34 | 8 | Tim Settle | 71.65 | 58.50 | 77.52 | 372 | Bills |
| 35 | 9 | Larry Ogunjobi | 71.55 | 55.20 | 79.29 | 636 | Steelers |
| 36 | 10 | Harrison Phillips | 71.55 | 67.66 | 71.70 | 693 | Vikings |
| 37 | 11 | Akiem Hicks | 71.47 | 59.18 | 80.99 | 398 | Buccaneers |
| 38 | 12 | Osa Odighizuwa | 71.11 | 55.97 | 77.41 | 616 | Cowboys |
| 39 | 13 | Morgan Fox | 70.94 | 56.42 | 76.46 | 575 | Chargers |
| 40 | 14 | Alim McNeill | 70.94 | 64.87 | 70.82 | 779 | Lions |
| 41 | 15 | Fletcher Cox | 70.64 | 55.01 | 77.39 | 712 | Eagles |
| 42 | 16 | Matt Ioannidis | 70.52 | 63.42 | 76.05 | 640 | Panthers |
| 43 | 17 | Chauncey Golston | 70.43 | 58.37 | 76.27 | 237 | Cowboys |
| 44 | 18 | Dre'Mont Jones | 70.35 | 57.95 | 77.33 | 715 | Broncos |
| 45 | 19 | Naquan Jones | 69.68 | 54.35 | 80.89 | 156 | Titans |
| 46 | 20 | Shy Tuttle | 69.47 | 58.88 | 73.00 | 557 | Saints |
| 47 | 21 | DaQuan Jones | 69.41 | 64.00 | 69.34 | 643 | Bills |
| 48 | 22 | Folorunso Fatukasi | 69.37 | 54.95 | 77.08 | 446 | Jaguars |
| 49 | 23 | Jordan Davis | 69.29 | 67.95 | 69.93 | 225 | Eagles |
| 50 | 24 | Khyiris Tonga | 69.28 | 64.62 | 72.64 | 276 | Vikings |
| 51 | 25 | Sheldon Rankins | 69.25 | 60.91 | 72.75 | 558 | Jets |
| 52 | 26 | Al Woods | 68.95 | 58.21 | 74.13 | 374 | Seahawks |
| 53 | 27 | Sebastian Joseph-Day | 68.92 | 52.28 | 79.28 | 702 | Chargers |
| 54 | 28 | A'Shawn Robinson | 68.89 | 62.12 | 74.33 | 360 | Rams |
| 55 | 29 | Roy Robertson-Harris | 68.81 | 56.04 | 75.70 | 714 | Jaguars |
| 56 | 30 | Bilal Nichols | 68.28 | 55.53 | 72.61 | 801 | Raiders |
| 57 | 31 | William Gholston | 68.20 | 50.36 | 75.93 | 484 | Buccaneers |
| 58 | 32 | Greg Gaines | 67.99 | 57.48 | 71.31 | 731 | Rams |
| 59 | 33 | Kevin Givens | 67.84 | 52.52 | 77.66 | 354 | 49ers |
| 60 | 34 | Adam Gotsis | 67.44 | 55.56 | 73.93 | 293 | Jaguars |
| 61 | 35 | Dean Lowry | 67.41 | 56.57 | 71.45 | 482 | Packers |
| 62 | 36 | Jarran Reed | 67.40 | 51.44 | 73.87 | 705 | Packers |
| 63 | 37 | Lawrence Guy Sr. | 67.35 | 47.79 | 78.10 | 504 | Patriots |
| 64 | 38 | Chris Wormley | 67.34 | 62.44 | 69.60 | 338 | Steelers |
| 65 | 39 | Linval Joseph | 67.28 | 54.98 | 76.61 | 188 | Eagles |
| 66 | 40 | Teair Tart | 67.26 | 63.38 | 69.81 | 520 | Titans |
| 67 | 41 | Marquise Copeland | 67.01 | 61.64 | 70.59 | 343 | Rams |
| 68 | 42 | Myles Adams | 66.97 | 58.43 | 78.29 | 190 | Seahawks |
| 69 | 43 | Zach Allen | 66.77 | 58.88 | 75.00 | 660 | Cardinals |
| 70 | 44 | Arik Armstead | 66.70 | 57.57 | 74.21 | 349 | 49ers |
| 71 | 45 | Taven Bryan | 66.69 | 59.54 | 68.38 | 642 | Browns |
| 72 | 46 | Ndamukong Suh | 66.66 | 50.17 | 77.91 | 176 | Eagles |
| 73 | 47 | Trysten Hill | 66.60 | 58.92 | 75.04 | 229 | Cardinals |
| 74 | 48 | James Lynch | 66.56 | 56.30 | 72.85 | 276 | Vikings |
| 75 | 49 | Nathan Shepherd | 66.49 | 58.87 | 67.82 | 416 | Jets |
| 76 | 50 | Devonte Wyatt | 66.43 | 66.32 | 63.32 | 224 | Packers |
| 77 | 51 | Mike Purcell | 66.24 | 55.13 | 72.75 | 529 | Broncos |
| 78 | 52 | Davon Godchaux | 66.05 | 53.89 | 72.27 | 659 | Patriots |
| 79 | 53 | Broderick Washington | 65.92 | 59.47 | 68.60 | 482 | Ravens |
| 80 | 54 | Maliek Collins | 65.89 | 59.20 | 68.58 | 601 | Texans |
| 81 | 55 | Hassan Ridgeway | 65.74 | 54.59 | 73.34 | 285 | 49ers |
| 82 | 56 | Austin Johnson | 65.43 | 55.72 | 72.15 | 287 | Chargers |
| 83 | 57 | Armon Watts | 65.40 | 51.89 | 70.24 | 531 | Bears |
| 84 | 58 | Matt Henningsen | 65.02 | 57.91 | 65.60 | 230 | Broncos |
| 85 | 59 | Justin Jones | 64.93 | 50.80 | 72.56 | 746 | Bears |
| 86 | 60 | Kentavius Street | 64.87 | 48.17 | 72.03 | 518 | Saints |
| 87 | 61 | DeShawn Williams | 64.74 | 49.78 | 71.55 | 598 | Broncos |
| 88 | 62 | Travis Jones | 64.58 | 57.31 | 67.22 | 322 | Ravens |
| 89 | 63 | Michael Dogbe | 64.44 | 50.78 | 73.29 | 282 | Cardinals |
| 90 | 64 | Roy Lopez | 64.38 | 50.20 | 70.03 | 557 | Texans |
| 91 | 65 | Quinton Jefferson | 64.31 | 46.16 | 72.24 | 566 | Seahawks |
| 92 | 66 | Jay Tufele | 64.22 | 56.27 | 76.25 | 137 | Bengals |
| 93 | 67 | Deadrin Senat | 64.22 | 60.68 | 71.21 | 165 | Buccaneers |
| 94 | 68 | Carlos Watkins | 64.09 | 52.74 | 70.52 | 278 | Cowboys |
| 95 | 69 | Jordan Phillips | 64.04 | 53.10 | 73.44 | 347 | Bills |
| 96 | 70 | Mike Pennel | 63.93 | 47.54 | 73.18 | 363 | Bears |
| 97 | 71 | John Jenkins | 63.76 | 53.61 | 70.82 | 258 | Dolphins |
| 98 | 72 | Marquan McCall | 63.75 | 54.80 | 66.53 | 185 | Panthers |
| 99 | 73 | Jonathan Bullard | 63.65 | 53.76 | 72.96 | 318 | Vikings |
| 100 | 74 | Corey Peters | 63.48 | 53.27 | 70.91 | 264 | Jaguars |
| 101 | 75 | Logan Hall | 63.39 | 50.05 | 68.12 | 403 | Buccaneers |
| 102 | 76 | Montravius Adams | 63.27 | 54.24 | 68.86 | 281 | Steelers |
| 103 | 77 | Solomon Thomas | 63.22 | 50.73 | 70.30 | 374 | Jets |
| 104 | 78 | Neville Gallimore | 63.14 | 49.67 | 72.39 | 402 | Cowboys |
| 105 | 79 | Michael Brockers | 63.02 | 49.93 | 73.48 | 123 | Lions |
| 106 | 80 | Johnathan Hankins | 62.94 | 47.99 | 73.05 | 235 | Cowboys |
| 107 | 81 | John Ridgeway | 62.74 | 51.26 | 68.19 | 279 | Commanders |
| 108 | 82 | Rashard Lawrence | 62.68 | 59.25 | 69.90 | 112 | Cardinals |
| 109 | 83 | Brent Urban | 62.49 | 49.10 | 70.98 | 298 | Ravens |
| 110 | 84 | Tershawn Wharton | 62.31 | 54.85 | 69.00 | 149 | Chiefs |
| 111 | 85 | Benito Jones | 62.28 | 53.25 | 68.04 | 309 | Lions |
| 112 | 86 | Eyioma Uwazurike | 62.26 | 58.60 | 69.35 | 165 | Broncos |
| 113 | 87 | Christian Covington | 62.23 | 53.45 | 70.58 | 123 | Chargers |
| 114 | 88 | Derrick Nnadi | 62.07 | 50.34 | 65.93 | 388 | Chiefs |
| 115 | 89 | Khalen Saunders | 62.01 | 53.90 | 69.39 | 421 | Chiefs |

### Rotation/backup (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 116 | 1 | Josh Tupou | 61.55 | 53.68 | 65.56 | 272 | Bengals |
| 117 | 2 | Joe Gaziano | 61.45 | 54.64 | 70.28 | 114 | Chargers |
| 118 | 3 | Kurt Hinish | 61.30 | 51.27 | 65.79 | 435 | Texans |
| 119 | 4 | Angelo Blackson | 61.24 | 47.13 | 67.47 | 393 | Bears |
| 120 | 5 | Bryan Mone | 61.21 | 53.61 | 66.21 | 271 | Seahawks |
| 121 | 6 | Rakeem Nunez-Roches | 61.15 | 48.80 | 65.51 | 548 | Buccaneers |
| 122 | 7 | Jalen Dalton | 61.11 | 52.45 | 74.01 | 145 | Falcons |
| 123 | 8 | Carl Davis Jr. | 60.93 | 52.47 | 65.61 | 218 | Patriots |
| 124 | 9 | Kyle Peko | 60.76 | 55.49 | 69.25 | 153 | Raiders |
| 125 | 10 | Ross Blacklock | 60.73 | 54.47 | 64.77 | 139 | Vikings |
| 126 | 11 | Thomas Booker IV | 60.63 | 52.04 | 69.05 | 206 | Texans |
| 127 | 12 | Jerry Tillery | 60.51 | 50.04 | 64.81 | 448 | Raiders |
| 128 | 13 | Jonathan Harris | 60.22 | 54.21 | 70.87 | 211 | Broncos |
| 129 | 14 | Ta'Quon Graham | 60.09 | 57.88 | 62.54 | 471 | Falcons |
| 130 | 15 | Jaleel Johnson | 60.00 | 51.72 | 66.76 | 181 | Falcons |
| 131 | 16 | Byron Cowart | 59.91 | 48.59 | 66.43 | 229 | Colts |
| 132 | 17 | Abdullah Anderson | 59.90 | 55.78 | 66.21 | 433 | Falcons |
| 133 | 18 | T.Y. McGill | 59.86 | 52.46 | 70.83 | 180 | 49ers |
| 134 | 19 | Larrell Murchison | 59.46 | 57.90 | 63.76 | 102 | Rams |
| 135 | 20 | Isaiahh Loudermilk | 59.31 | 54.52 | 63.35 | 116 | Steelers |
| 136 | 21 | L.J. Collier | 59.25 | 54.07 | 65.00 | 149 | Seahawks |
| 137 | 22 | Bravvion Roy | 59.06 | 51.49 | 62.10 | 299 | Panthers |
| 138 | 23 | Daniel Ekuale | 58.99 | 52.90 | 64.26 | 362 | Patriots |
| 139 | 24 | Raekwon Davis | 58.96 | 49.46 | 62.49 | 583 | Dolphins |
| 140 | 25 | Henry Mondeaux | 58.90 | 50.35 | 65.22 | 249 | Giants |
| 141 | 26 | Michael Dwumfour | 58.89 | 52.30 | 69.16 | 238 | 49ers |
| 142 | 27 | Javon Kinlaw | 58.59 | 54.61 | 66.71 | 162 | 49ers |
| 143 | 28 | Jonah Williams | 58.55 | 53.57 | 63.47 | 342 | Rams |
| 144 | 29 | Tyson Alualu | 58.54 | 43.07 | 69.30 | 291 | Steelers |
| 145 | 30 | Otito Ogbonnia | 58.53 | 53.33 | 69.13 | 138 | Chargers |
| 146 | 31 | Kevin Strong | 58.51 | 53.66 | 63.39 | 305 | Titans |
| 147 | 32 | Akeem Spence | 58.41 | 49.39 | 68.49 | 147 | 49ers |
| 148 | 33 | Zach Carter | 58.26 | 47.75 | 62.08 | 395 | Bengals |
| 149 | 34 | Leki Fotu | 57.89 | 46.79 | 62.15 | 499 | Cardinals |
| 150 | 35 | Justin Ellis | 57.53 | 42.36 | 64.11 | 362 | Giants |
| 151 | 36 | Ryder Anderson | 57.36 | 53.59 | 67.01 | 152 | Giants |
| 152 | 37 | Malcolm Roach | 57.14 | 50.73 | 63.62 | 316 | Saints |
| 153 | 38 | Isaiah Buggs | 56.99 | 48.68 | 61.69 | 752 | Lions |
| 154 | 39 | Perrion Winfrey | 56.96 | 48.79 | 62.15 | 342 | Browns |
| 155 | 40 | Tommy Togiai | 56.94 | 52.35 | 62.94 | 225 | Browns |
| 156 | 41 | Kerry Hyder Jr. | 56.32 | 49.15 | 57.92 | 357 | 49ers |
| 157 | 42 | Breiden Fehoko | 56.15 | 51.68 | 63.18 | 279 | Chargers |
| 158 | 43 | Jordan Elliott | 56.04 | 44.88 | 59.61 | 703 | Browns |
| 159 | 44 | Quinton Bohanna | 55.71 | 50.09 | 58.84 | 264 | Cowboys |
| 160 | 45 | Jonathan Ledbetter | 55.54 | 50.22 | 62.64 | 275 | Cardinals |
| 161 | 46 | Bobby Brown III | 55.41 | 57.70 | 58.30 | 164 | Rams |
| 162 | 47 | Neil Farrell Jr. | 53.72 | 50.33 | 59.67 | 316 | Raiders |
| 163 | 48 | Timmy Horne | 53.44 | 49.09 | 52.17 | 385 | Falcons |
| 164 | 49 | Marlon Tuipulotu | 52.05 | 50.73 | 58.08 | 232 | Eagles |
| 165 | 50 | Earnest Brown IV | 50.00 | 53.66 | 59.66 | 136 | Rams |
| 166 | 51 | Eric Johnson | 49.89 | 52.84 | 46.69 | 127 | Colts |

## ED — Edge

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Micah Parsons | 93.09 | 90.94 | 90.35 | 917 | Cowboys |
| 2 | 2 | Nick Bosa | 91.85 | 96.14 | 88.22 | 745 | 49ers |
| 3 | 3 | Myles Garrett | 90.68 | 95.45 | 84.23 | 816 | Browns |
| 4 | 4 | T.J. Watt | 86.97 | 86.48 | 87.36 | 502 | Steelers |
| 5 | 5 | Jaelan Phillips | 85.81 | 89.59 | 79.13 | 838 | Dolphins |
| 6 | 6 | Maxx Crosby | 85.63 | 91.15 | 77.78 | 1082 | Raiders |
| 7 | 7 | Rashan Gary | 85.46 | 85.73 | 85.53 | 378 | Packers |
| 8 | 8 | Von Miller | 84.59 | 78.19 | 88.23 | 450 | Bills |
| 9 | 9 | Danielle Hunter | 84.53 | 84.19 | 83.52 | 905 | Vikings |
| 10 | 10 | Joey Bosa | 83.91 | 83.51 | 87.02 | 165 | Chargers |
| 11 | 11 | DeMarcus Lawrence | 83.09 | 88.78 | 78.06 | 696 | Cowboys |
| 12 | 12 | Greg Rousseau | 82.81 | 80.48 | 82.64 | 463 | Bills |
| 13 | 13 | Khalil Mack | 82.02 | 79.68 | 82.35 | 860 | Chargers |
| 14 | 14 | Trey Hendrickson | 81.53 | 76.39 | 82.27 | 629 | Bengals |
| 15 | 15 | Montez Sweat | 81.26 | 86.05 | 75.97 | 731 | Commanders |
| 16 | 16 | Haason Reddick | 81.21 | 69.64 | 85.05 | 816 | Eagles |

### Good (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Josh Uche | 79.24 | 68.09 | 86.43 | 373 | Patriots |
| 18 | 2 | Brandon Graham | 79.17 | 77.29 | 80.68 | 474 | Eagles |
| 19 | 3 | Josh Sweat | 78.84 | 78.31 | 76.22 | 587 | Eagles |
| 20 | 4 | Sam Williams | 78.28 | 66.75 | 83.77 | 274 | Cowboys |
| 21 | 5 | James Houston | 77.86 | 63.37 | 94.65 | 140 | Lions |
| 22 | 6 | Cameron Jordan | 77.65 | 77.76 | 74.19 | 790 | Saints |
| 23 | 7 | Marcus Davenport | 77.19 | 80.18 | 74.82 | 490 | Saints |
| 24 | 8 | Justin Houston | 77.00 | 62.01 | 84.89 | 397 | Ravens |
| 25 | 9 | Shaquil Barrett | 76.66 | 69.43 | 82.51 | 382 | Buccaneers |
| 26 | 10 | Uchenna Nwosu | 76.60 | 69.93 | 77.51 | 904 | Seahawks |
| 27 | 11 | Za'Darius Smith | 76.33 | 75.17 | 78.13 | 770 | Vikings |
| 28 | 12 | Matthew Judon | 76.12 | 59.66 | 83.35 | 858 | Patriots |
| 29 | 13 | Brian Burns | 75.87 | 64.29 | 80.12 | 951 | Panthers |
| 30 | 14 | Aidan Hutchinson | 75.46 | 76.19 | 70.81 | 953 | Lions |
| 31 | 15 | Andrew Van Ginkel | 75.04 | 63.42 | 79.10 | 333 | Dolphins |
| 32 | 16 | John Franklin-Myers | 74.64 | 73.40 | 71.60 | 643 | Jets |

### Starter (65 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Ogbo Okoronkwo | 73.69 | 65.68 | 77.30 | 517 | Texans |
| 34 | 2 | Bradley Chubb | 72.91 | 66.88 | 76.61 | 742 | Dolphins |
| 35 | 3 | Azeez Ojulari | 72.56 | 63.03 | 86.04 | 230 | Giants |
| 36 | 4 | Alex Highsmith | 72.10 | 65.98 | 72.31 | 941 | Steelers |
| 37 | 5 | Carlos Dunlap | 71.70 | 59.56 | 75.83 | 571 | Chiefs |
| 38 | 6 | Carl Lawson | 71.43 | 59.41 | 75.28 | 663 | Jets |
| 39 | 7 | Carl Granderson | 71.17 | 64.29 | 72.87 | 480 | Saints |
| 40 | 8 | Preston Smith | 71.16 | 62.53 | 73.04 | 825 | Packers |
| 41 | 9 | Melvin Ingram III | 70.81 | 61.10 | 75.59 | 509 | Dolphins |
| 42 | 10 | Darrell Taylor | 70.52 | 57.70 | 75.69 | 484 | Seahawks |
| 43 | 11 | Jadeveon Clowney | 70.42 | 74.41 | 68.60 | 494 | Browns |
| 44 | 12 | Jerry Hughes | 70.27 | 58.67 | 74.04 | 689 | Texans |
| 45 | 13 | Micheal Clemons | 70.25 | 64.63 | 70.81 | 311 | Jets |
| 46 | 14 | Denico Autry | 69.37 | 54.43 | 77.61 | 531 | Titans |
| 47 | 15 | Sam Hubbard | 69.32 | 62.55 | 71.57 | 801 | Bengals |
| 48 | 16 | Julian Okwara | 69.18 | 59.11 | 78.41 | 220 | Lions |
| 49 | 17 | Yannick Ngakoue | 69.16 | 56.99 | 74.29 | 733 | Colts |
| 50 | 18 | Dante Fowler Jr. | 68.89 | 60.32 | 71.73 | 343 | Cowboys |
| 51 | 19 | Markus Golden | 68.85 | 50.44 | 77.26 | 781 | Cardinals |
| 52 | 20 | Bryce Huff | 68.77 | 63.50 | 72.35 | 191 | Jets |
| 53 | 21 | Chandler Jones | 68.75 | 57.75 | 75.78 | 783 | Raiders |
| 54 | 22 | Jermaine Johnson | 68.65 | 62.73 | 71.37 | 312 | Jets |
| 55 | 23 | Leonard Floyd | 68.46 | 57.55 | 71.56 | 932 | Rams |
| 56 | 24 | Deatrich Wise Jr. | 68.33 | 62.70 | 68.22 | 828 | Patriots |
| 57 | 25 | Arden Key | 68.28 | 67.87 | 64.80 | 475 | Jaguars |
| 58 | 26 | Samson Ebukam | 68.04 | 61.05 | 69.52 | 559 | 49ers |
| 59 | 27 | Frank Clark | 67.96 | 56.98 | 73.18 | 716 | Chiefs |
| 60 | 28 | Kwity Paye | 67.88 | 67.26 | 67.92 | 547 | Colts |
| 61 | 29 | Randy Gregory | 67.76 | 63.48 | 74.57 | 187 | Broncos |
| 62 | 30 | Jacob Martin | 67.52 | 59.93 | 70.79 | 261 | Broncos |
| 63 | 31 | Odafe Oweh | 67.30 | 64.72 | 65.59 | 633 | Ravens |
| 64 | 32 | Robert Quinn | 67.08 | 49.72 | 76.95 | 393 | Eagles |
| 65 | 33 | Myjai Sanders | 67.01 | 59.23 | 71.94 | 260 | Cardinals |
| 66 | 34 | Trevis Gipson | 67.00 | 56.61 | 71.92 | 641 | Bears |
| 67 | 35 | A.J. Epenesa | 66.83 | 61.31 | 68.63 | 374 | Bills |
| 68 | 36 | Kayvon Thibodeaux | 66.42 | 69.72 | 62.98 | 740 | Giants |
| 69 | 37 | George Karlaftis | 66.32 | 57.87 | 67.78 | 729 | Chiefs |
| 70 | 38 | Jonathan Greenard | 66.17 | 63.16 | 70.73 | 284 | Texans |
| 71 | 39 | Shaq Lawson | 66.14 | 58.78 | 68.69 | 467 | Bills |
| 72 | 40 | Charles Omenihu | 66.01 | 59.37 | 66.85 | 572 | 49ers |
| 73 | 41 | Mario Addison | 65.94 | 47.67 | 76.62 | 367 | Texans |
| 74 | 42 | Dorance Armstrong | 65.93 | 60.10 | 66.83 | 542 | Cowboys |
| 75 | 43 | Tyus Bowser | 65.70 | 56.11 | 71.84 | 355 | Ravens |
| 76 | 44 | Chase Winovich | 65.66 | 58.69 | 73.39 | 178 | Browns |
| 77 | 45 | Payton Turner | 65.56 | 65.02 | 71.69 | 171 | Saints |
| 78 | 46 | Dawuane Smoot | 65.25 | 61.24 | 65.03 | 445 | Jaguars |
| 79 | 47 | Kingsley Enagbare | 64.55 | 59.13 | 63.99 | 465 | Packers |
| 80 | 48 | John Cominsky | 64.52 | 59.12 | 69.54 | 554 | Lions |
| 81 | 49 | Dennis Gardeck | 64.40 | 56.06 | 68.14 | 210 | Cardinals |
| 82 | 50 | Joe Tryon-Shoyinka | 64.26 | 59.41 | 63.33 | 843 | Buccaneers |
| 83 | 51 | Dayo Odeyingbo | 64.24 | 62.58 | 63.74 | 519 | Colts |
| 84 | 52 | Vinny Curry | 64.10 | 51.24 | 73.00 | 184 | Jets |
| 85 | 53 | Emmanuel Ogbah | 63.90 | 59.43 | 66.63 | 326 | Dolphins |
| 86 | 54 | Boye Mafe | 63.81 | 59.84 | 62.29 | 423 | Seahawks |
| 87 | 55 | Clelin Ferrell | 63.59 | 61.32 | 62.77 | 492 | Raiders |
| 88 | 56 | Anthony Nelson | 63.51 | 60.76 | 61.18 | 632 | Buccaneers |
| 89 | 57 | Efe Obada | 63.42 | 52.79 | 68.40 | 391 | Commanders |
| 90 | 58 | Jonathon Cooper | 62.82 | 61.98 | 61.42 | 443 | Broncos |
| 91 | 59 | Ifeadi Odenigbo | 62.56 | 55.71 | 67.00 | 262 | Buccaneers |
| 92 | 60 | Chase Young | 62.54 | 73.23 | 60.67 | 114 | Commanders |
| 93 | 61 | Bud Dupree | 62.54 | 56.42 | 68.21 | 453 | Titans |
| 94 | 62 | Lorenzo Carter | 62.27 | 56.37 | 65.21 | 909 | Falcons |
| 95 | 63 | Mike Danna | 62.26 | 61.15 | 61.42 | 471 | Chiefs |
| 96 | 64 | Travon Walker | 62.01 | 62.04 | 59.79 | 788 | Jaguars |
| 97 | 65 | Arnold Ebiketie | 62.01 | 60.86 | 59.60 | 516 | Falcons |

### Rotation/backup (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 98 | 1 | Drake Jackson | 61.94 | 59.81 | 61.16 | 315 | 49ers |
| 99 | 2 | Justin Hollins | 61.61 | 57.22 | 63.50 | 435 | Packers |
| 100 | 3 | Isaiah Thomas | 60.98 | 58.24 | 65.51 | 162 | Browns |
| 101 | 4 | Carl Nassib | 60.86 | 57.68 | 62.37 | 250 | Buccaneers |
| 102 | 5 | Marquis Haynes Sr. | 60.60 | 53.00 | 61.70 | 470 | Panthers |
| 103 | 6 | Mario Edwards Jr. | 60.52 | 58.18 | 61.15 | 463 | Titans |
| 104 | 7 | Jason Pierre-Paul | 60.50 | 46.30 | 68.73 | 524 | Ravens |
| 105 | 8 | D.J. Wonnum | 60.47 | 56.63 | 59.77 | 562 | Vikings |
| 106 | 9 | Romeo Okwara | 59.88 | 59.52 | 65.65 | 119 | Lions |
| 107 | 10 | Rasheem Green | 59.80 | 54.19 | 59.86 | 567 | Texans |
| 108 | 11 | Josh Paschal | 59.57 | 59.11 | 62.58 | 293 | Lions |
| 109 | 12 | Cam Sample | 59.17 | 59.77 | 56.69 | 411 | Bengals |
| 110 | 13 | Malik Reed | 59.09 | 56.11 | 59.26 | 396 | Steelers |
| 111 | 14 | K'Lavon Chaisson | 59.06 | 58.86 | 59.55 | 109 | Jaguars |
| 112 | 15 | Casey Toohill | 58.92 | 57.22 | 58.55 | 347 | Commanders |
| 113 | 16 | Chris Rumph II | 58.85 | 57.68 | 57.43 | 300 | Chargers |
| 114 | 17 | Bruce Irvin | 58.70 | 43.53 | 73.75 | 402 | Seahawks |
| 115 | 18 | Charles Harris | 58.53 | 57.19 | 61.28 | 259 | Lions |
| 116 | 19 | Yetur Gross-Matos | 58.44 | 57.79 | 56.43 | 847 | Panthers |
| 117 | 20 | Tarell Basham | 57.81 | 56.34 | 59.52 | 165 | Titans |
| 118 | 21 | Rashad Weaver | 56.84 | 57.04 | 58.68 | 640 | Titans |
| 119 | 22 | Jordan Willis | 56.80 | 58.18 | 59.15 | 200 | 49ers |
| 120 | 23 | Tanoh Kpassagnon | 56.73 | 56.36 | 56.44 | 356 | Saints |
| 121 | 24 | Oshane Ximines | 56.59 | 56.37 | 58.41 | 506 | Giants |
| 122 | 25 | James Smith-Williams | 56.34 | 55.02 | 55.41 | 506 | Commanders |
| 123 | 26 | Derrek Tuszka | 56.33 | 56.44 | 57.51 | 123 | Chargers |
| 124 | 27 | Al-Quadin Muhammad | 56.28 | 52.92 | 54.84 | 609 | Bears |
| 125 | 28 | Tyquan Lewis | 56.22 | 57.93 | 61.66 | 273 | Colts |
| 126 | 29 | Jonathan Garvin | 56.13 | 57.08 | 55.94 | 194 | Packers |
| 127 | 30 | Dominique Robinson | 56.13 | 53.79 | 53.52 | 549 | Bears |
| 128 | 31 | Terrell Lewis | 56.11 | 57.66 | 57.27 | 332 | Bears |
| 129 | 32 | Adetokunbo Ogundeji | 55.47 | 54.89 | 52.68 | 541 | Falcons |
| 130 | 33 | Jihad Ward | 55.27 | 47.56 | 57.97 | 657 | Giants |
| 131 | 34 | Austin Bryant | 55.06 | 57.49 | 56.16 | 207 | Lions |
| 132 | 35 | Ben Banogu | 54.88 | 56.72 | 54.08 | 116 | Colts |
| 133 | 36 | Sam Okuayinonu | 54.42 | 59.19 | 60.86 | 105 | Titans |
| 134 | 37 | Patrick Johnson | 54.09 | 55.38 | 49.68 | 216 | Eagles |
| 135 | 38 | Isaac Rochell | 53.96 | 56.35 | 54.57 | 137 | Raiders |
| 136 | 39 | Alex Wright | 53.18 | 53.72 | 48.65 | 543 | Browns |
| 137 | 40 | DeMarvin Leal | 52.65 | 56.87 | 51.56 | 175 | Steelers |
| 138 | 41 | Henry Anderson | 52.65 | 49.09 | 53.79 | 203 | Panthers |
| 139 | 42 | Victor Dimukeje | 51.79 | 56.96 | 50.79 | 251 | Cardinals |

## G — Guard

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Lindstrom | 98.58 | 95.00 | 96.80 | 1047 | Falcons |
| 2 | 2 | Joel Bitonio | 92.37 | 87.50 | 91.45 | 1172 | Browns |
| 3 | 3 | Quinn Meinerz | 84.35 | 75.47 | 86.10 | 752 | Broncos |
| 4 | 4 | Joe Thuney | 83.59 | 77.30 | 83.62 | 999 | Chiefs |
| 5 | 5 | Robert Hunt | 83.56 | 73.70 | 85.97 | 1055 | Dolphins |
| 6 | 6 | Teven Jenkins | 82.96 | 75.83 | 83.55 | 576 | Bears |
| 7 | 7 | Isaac Seumalo | 82.67 | 75.20 | 83.48 | 1135 | Eagles |
| 8 | 8 | Ezra Cleveland | 82.17 | 73.50 | 83.78 | 1134 | Vikings |
| 9 | 9 | Kevin Zeitler | 81.68 | 73.87 | 82.72 | 955 | Ravens |
| 10 | 10 | Elgton Jenkins | 81.15 | 72.14 | 82.99 | 960 | Packers |
| 11 | 11 | Zack Martin | 80.95 | 73.30 | 81.89 | 1143 | Cowboys |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Trey Smith | 79.98 | 71.50 | 81.47 | 1039 | Chiefs |
| 13 | 2 | Damien Lewis | 79.92 | 71.80 | 81.16 | 1002 | Seahawks |
| 14 | 3 | Wyatt Teller | 79.28 | 70.12 | 81.22 | 927 | Browns |
| 15 | 4 | Nate Davis | 78.87 | 68.82 | 81.41 | 682 | Titans |
| 16 | 5 | Landon Dickerson | 78.15 | 67.30 | 81.22 | 1094 | Eagles |
| 17 | 6 | Austin Corbett | 77.94 | 69.10 | 79.67 | 985 | Panthers |
| 18 | 7 | Quenton Nelson | 77.60 | 68.40 | 79.56 | 1148 | Colts |
| 19 | 8 | Shaq Mason | 77.41 | 68.90 | 78.91 | 1200 | Buccaneers |
| 20 | 9 | Alex Cappa | 76.89 | 67.60 | 78.92 | 1086 | Bengals |
| 21 | 10 | Jonah Jackson | 76.54 | 65.69 | 79.61 | 858 | Lions |
| 22 | 11 | Kevin Dotson | 76.19 | 65.40 | 79.22 | 1160 | Steelers |
| 23 | 12 | James Daniels | 76.13 | 66.90 | 78.12 | 1160 | Steelers |
| 24 | 13 | Will Hernandez | 76.00 | 65.00 | 79.17 | 843 | Cardinals |
| 25 | 14 | Zion Johnson | 75.87 | 64.80 | 79.08 | 1184 | Chargers |
| 26 | 15 | A.J. Cann | 75.67 | 66.60 | 77.55 | 1003 | Texans |
| 27 | 16 | Michael Schofield III | 75.63 | 64.49 | 78.89 | 418 | Bears |
| 28 | 17 | Elijah Wilkinson | 75.28 | 63.28 | 79.12 | 574 | Falcons |
| 29 | 18 | Oday Aboushi | 75.16 | 63.05 | 79.06 | 339 | Rams |
| 30 | 19 | Cody Whitehair | 74.84 | 64.83 | 77.34 | 661 | Bears |
| 31 | 20 | David Edwards | 74.36 | 59.13 | 80.34 | 230 | Rams |

### Starter (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 32 | 1 | Mark Glowinski | 73.73 | 63.30 | 76.52 | 1045 | Giants |
| 33 | 2 | Nick Leverett | 73.63 | 63.52 | 76.21 | 761 | Buccaneers |
| 34 | 3 | Ben Powers | 73.20 | 62.90 | 75.90 | 1094 | Ravens |
| 35 | 4 | Robert Jones | 73.16 | 61.35 | 76.87 | 449 | Dolphins |
| 36 | 5 | Aaron Banks | 73.09 | 62.68 | 75.86 | 969 | 49ers |
| 37 | 6 | Dylan Parham | 72.93 | 61.90 | 76.12 | 1036 | Raiders |
| 38 | 7 | Jon Runyan | 72.71 | 62.60 | 75.28 | 1051 | Packers |
| 39 | 8 | Ryan Bates | 72.37 | 61.76 | 75.27 | 945 | Bills |
| 40 | 9 | Justin Pugh | 72.29 | 60.71 | 75.84 | 263 | Cardinals |
| 41 | 10 | Will Fries | 72.16 | 58.71 | 76.96 | 642 | Colts |
| 42 | 11 | Ben Bartch | 72.14 | 60.27 | 75.89 | 293 | Jaguars |
| 43 | 12 | Andrew Norwell | 71.70 | 59.80 | 75.47 | 1120 | Commanders |
| 44 | 13 | Tyler Shatley | 71.40 | 60.82 | 74.28 | 819 | Jaguars |
| 45 | 14 | Aaron Brewer | 71.39 | 59.90 | 74.88 | 1031 | Titans |
| 46 | 15 | Dalton Risner | 71.05 | 61.09 | 73.53 | 967 | Broncos |
| 47 | 16 | Ben Bredeson | 71.02 | 57.55 | 75.84 | 542 | Giants |
| 48 | 17 | Greg Van Roten | 70.98 | 58.56 | 75.10 | 354 | Bills |
| 49 | 18 | Brandon Scherff | 70.21 | 59.00 | 73.51 | 1086 | Jaguars |
| 50 | 19 | Ed Ingram | 70.02 | 57.10 | 74.46 | 1168 | Vikings |
| 51 | 20 | Phil Haynes | 69.96 | 57.97 | 73.78 | 485 | Seahawks |
| 52 | 21 | Nate Herbig | 69.88 | 58.31 | 73.43 | 707 | Jets |
| 53 | 22 | Brady Christensen | 69.75 | 57.33 | 73.87 | 965 | Panthers |
| 54 | 23 | Dan Feeney | 69.49 | 56.72 | 73.84 | 109 | Jets |
| 55 | 24 | Colby Gossett | 69.42 | 57.55 | 73.16 | 267 | Falcons |
| 56 | 25 | Royce Newman | 69.36 | 58.31 | 72.56 | 451 | Packers |
| 57 | 26 | Lucas Patrick | 69.33 | 57.86 | 72.81 | 269 | Bears |
| 58 | 27 | Nick Allegretti | 69.21 | 55.90 | 73.92 | 286 | Chiefs |
| 59 | 28 | Cesar Ruiz | 68.74 | 56.81 | 72.53 | 868 | Saints |
| 60 | 29 | Laken Tomlinson | 68.20 | 56.80 | 71.63 | 1110 | Jets |
| 61 | 30 | Max Garcia | 67.83 | 55.35 | 71.98 | 542 | Cardinals |
| 62 | 31 | Cole Strange | 67.18 | 54.61 | 71.39 | 982 | Patriots |
| 63 | 32 | Jack Anderson | 67.13 | 55.16 | 70.94 | 148 | Giants |
| 64 | 33 | Gabe Jackson | 66.83 | 55.53 | 70.19 | 667 | Seahawks |
| 65 | 34 | Trai Turner | 66.75 | 53.50 | 71.41 | 766 | Commanders |
| 66 | 35 | Rashaad Coward | 66.66 | 54.68 | 70.48 | 155 | Cardinals |
| 67 | 36 | Kayode Awosika | 66.60 | 56.16 | 69.39 | 155 | Lions |
| 68 | 37 | Andrus Peat | 66.50 | 52.83 | 71.45 | 573 | Saints |
| 69 | 38 | Connor McGovern | 66.48 | 52.51 | 71.62 | 909 | Cowboys |
| 70 | 39 | Chandler Brewer | 66.20 | 57.88 | 67.58 | 228 | Rams |
| 71 | 40 | Matt Feiler | 66.17 | 53.30 | 70.59 | 1189 | Chargers |
| 72 | 41 | Jordan Roos | 65.56 | 56.69 | 67.31 | 202 | Titans |
| 73 | 42 | Justin McCray | 65.31 | 52.56 | 69.64 | 151 | Texans |
| 74 | 43 | Spencer Burford | 65.07 | 50.96 | 70.31 | 744 | 49ers |
| 75 | 44 | Cordell Volson | 64.97 | 51.60 | 69.71 | 1107 | Bengals |
| 76 | 45 | Logan Stenberg | 64.65 | 50.04 | 70.22 | 228 | Lions |
| 77 | 46 | Joshua Ezeudu | 64.60 | 52.40 | 68.57 | 290 | Giants |
| 78 | 47 | Lecitus Smith | 63.96 | 52.98 | 67.11 | 210 | Cardinals |
| 79 | 48 | Danny Pinter | 63.93 | 51.67 | 67.94 | 292 | Colts |
| 80 | 49 | Saahdiq Charles | 63.55 | 51.10 | 67.69 | 290 | Commanders |
| 81 | 50 | Luke Goedeke | 62.55 | 48.12 | 68.00 | 523 | Buccaneers |
| 82 | 51 | Dillon Radunz | 62.54 | 49.50 | 67.06 | 280 | Titans |

### Rotation/backup (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 83 | 1 | Cody Ford | 61.86 | 48.79 | 66.40 | 350 | Cardinals |
| 84 | 2 | Alex Bars | 61.71 | 46.42 | 67.73 | 852 | Raiders |
| 85 | 3 | Bobby Hart | 61.59 | 56.80 | 60.61 | 125 | Bills |
| 86 | 4 | Rodger Saffold | 61.04 | 43.70 | 68.44 | 1058 | Bills |
| 87 | 5 | Liam Eichenberg | 60.49 | 43.88 | 67.39 | 627 | Dolphins |
| 88 | 6 | Kenyon Green | 58.59 | 40.00 | 66.81 | 823 | Texans |

## HB — Running Back

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Chubb | 87.55 | 86.98 | 83.77 | 254 | Browns |
| 2 | 2 | Tony Pollard | 85.50 | 83.69 | 82.54 | 245 | Cowboys |
| 3 | 3 | Josh Jacobs | 84.29 | 90.60 | 75.91 | 341 | Raiders |
| 4 | 4 | Rhamondre Stevenson | 82.97 | 80.12 | 80.71 | 339 | Patriots |
| 5 | 5 | Aaron Jones | 82.79 | 84.72 | 77.33 | 316 | Packers |
| 6 | 6 | Christian McCaffrey | 81.69 | 88.90 | 72.72 | 405 | 49ers |
| 7 | 7 | Tyler Allgeier | 81.68 | 78.22 | 79.82 | 177 | Falcons |
| 8 | 8 | Derrick Henry | 81.48 | 81.38 | 77.38 | 191 | Titans |
| 9 | 9 | Austin Ekeler | 81.41 | 81.30 | 77.31 | 437 | Chargers |

### Good (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Breece Hall | 79.65 | 65.25 | 85.08 | 109 | Jets |
| 11 | 2 | Dameon Pierce | 78.25 | 73.43 | 77.29 | 192 | Texans |
| 12 | 3 | AJ Dillon | 78.11 | 79.02 | 73.34 | 252 | Packers |
| 13 | 4 | Devin Singletary | 77.60 | 75.75 | 74.67 | 373 | Bills |
| 14 | 5 | Travis Etienne Jr. | 77.12 | 71.03 | 77.02 | 316 | Jaguars |
| 15 | 6 | Jaylen Warren | 76.81 | 69.28 | 77.67 | 172 | Steelers |
| 16 | 7 | Kenneth Walker III | 76.77 | 73.46 | 74.81 | 247 | Seahawks |
| 17 | 8 | Saquon Barkley | 76.64 | 77.10 | 72.17 | 380 | Giants |
| 18 | 9 | Alvin Kamara | 76.46 | 71.79 | 75.40 | 295 | Saints |
| 19 | 10 | James Cook | 76.35 | 70.48 | 76.09 | 126 | Bills |
| 20 | 11 | Jonathan Taylor | 76.11 | 66.78 | 78.17 | 256 | Colts |
| 21 | 12 | Miles Sanders | 76.10 | 71.78 | 74.81 | 269 | Eagles |
| 22 | 13 | Raheem Mostert | 75.87 | 73.81 | 73.08 | 294 | Dolphins |
| 23 | 14 | Cordarrelle Patterson | 75.69 | 76.83 | 70.77 | 170 | Falcons |
| 24 | 15 | Khalil Herbert | 74.79 | 66.63 | 76.06 | 107 | Bears |
| 25 | 16 | D'Andre Swift | 74.61 | 74.26 | 70.67 | 236 | Lions |
| 26 | 17 | Kareem Hunt | 74.52 | 66.23 | 75.88 | 254 | Browns |
| 27 | 18 | Dalvin Cook | 74.25 | 67.40 | 74.65 | 395 | Vikings |
| 28 | 19 | Joe Mixon | 74.00 | 78.03 | 67.14 | 301 | Bengals |

### Starter (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 29 | 1 | Najee Harris | 73.66 | 71.93 | 70.65 | 297 | Steelers |
| 30 | 2 | Dontrell Hilliard | 72.84 | 64.15 | 74.47 | 130 | Titans |
| 31 | 3 | Latavius Murray | 72.69 | 78.57 | 64.61 | 191 | Broncos |
| 32 | 4 | Antonio Gibson | 72.51 | 74.26 | 67.17 | 238 | Commanders |
| 33 | 5 | Ezekiel Elliott | 72.39 | 70.36 | 69.57 | 234 | Cowboys |
| 34 | 6 | Eno Benjamin | 72.03 | 68.74 | 70.05 | 195 | Saints |
| 35 | 7 | Isiah Pacheco | 71.89 | 69.08 | 69.59 | 147 | Chiefs |
| 36 | 8 | Alexander Mattison | 71.54 | 67.29 | 70.21 | 158 | Vikings |
| 37 | 9 | James Conner | 71.53 | 69.32 | 68.84 | 332 | Cardinals |
| 38 | 10 | Cam Akers | 71.02 | 73.64 | 65.11 | 165 | Rams |
| 39 | 11 | Justin Jackson | 70.80 | 59.28 | 74.32 | 100 | Lions |
| 40 | 12 | Michael Carter | 70.66 | 61.09 | 72.87 | 267 | Jets |
| 41 | 13 | Samaje Perine | 70.64 | 69.28 | 67.38 | 239 | Bengals |
| 42 | 14 | David Montgomery | 70.36 | 67.16 | 68.32 | 270 | Bears |
| 43 | 15 | Clyde Edwards-Helaire | 70.07 | 65.64 | 68.86 | 124 | Chiefs |
| 44 | 16 | Leonard Fournette | 69.59 | 68.00 | 66.49 | 392 | Buccaneers |
| 45 | 17 | D'Onta Foreman | 69.29 | 67.11 | 66.58 | 112 | Panthers |
| 46 | 18 | Chuba Hubbard | 69.04 | 68.93 | 64.94 | 110 | Panthers |
| 47 | 19 | DeeJay Dallas | 68.73 | 63.72 | 67.90 | 117 | Seahawks |
| 48 | 20 | James Robinson | 68.57 | 59.78 | 70.27 | 106 | Jets |
| 49 | 21 | Justice Hill | 67.90 | 60.17 | 68.89 | 124 | Ravens |
| 50 | 22 | Jamaal Williams | 67.88 | 67.50 | 63.96 | 119 | Lions |
| 51 | 23 | Matt Breida | 67.84 | 64.01 | 66.23 | 132 | Giants |
| 52 | 24 | Jeff Wilson Jr. | 67.79 | 65.63 | 65.07 | 253 | Dolphins |
| 53 | 25 | JaMycal Hasty | 67.17 | 67.17 | 63.01 | 121 | Jaguars |
| 54 | 26 | Jerick McKinnon | 66.58 | 63.10 | 64.74 | 343 | Chiefs |
| 55 | 27 | Kenneth Gainwell | 66.36 | 57.86 | 67.86 | 193 | Eagles |
| 56 | 28 | Kenyan Drake | 66.24 | 54.54 | 69.88 | 162 | Ravens |
| 57 | 29 | Rachaad White | 65.59 | 65.26 | 61.64 | 257 | Buccaneers |
| 58 | 30 | Nyheim Hines | 65.58 | 62.50 | 63.46 | 136 | Bills |
| 59 | 31 | Ameer Abdullah | 65.02 | 63.80 | 61.66 | 134 | Raiders |
| 60 | 32 | J.D. McKissic | 64.31 | 56.31 | 65.48 | 154 | Commanders |
| 61 | 33 | Rex Burkhead | 64.26 | 59.37 | 63.36 | 189 | Texans |
| 62 | 34 | Chase Edmonds | 63.90 | 53.97 | 66.35 | 163 | Broncos |
| 63 | 35 | Joshua Kelley | 62.57 | 61.79 | 58.92 | 135 | Chargers |
| 64 | 36 | Dare Ogunbowale | 62.40 | 62.56 | 58.13 | 100 | Texans |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 65 | 1 | Deon Jackson | 61.43 | 61.77 | 57.04 | 142 | Colts |

## LB — Linebacker

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Bobby Wagner | 86.92 | 90.70 | 80.54 | 1079 | Rams |
| 2 | 2 | Lavonte David | 81.79 | 84.10 | 77.55 | 1074 | Buccaneers |
| 3 | 3 | Fred Warner | 81.65 | 83.70 | 76.42 | 1026 | 49ers |
| 4 | 4 | T.J. Edwards | 81.63 | 84.80 | 76.48 | 1040 | Eagles |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Ja'Whaun Bentley | 79.05 | 80.40 | 74.90 | 907 | Patriots |
| 6 | 2 | Germaine Pratt | 78.90 | 79.33 | 76.01 | 722 | Bengals |
| 7 | 3 | Demario Davis | 78.79 | 82.70 | 72.32 | 1132 | Saints |
| 8 | 4 | Alex Singleton | 78.36 | 78.52 | 75.57 | 771 | Broncos |
| 9 | 5 | Nick Bolton | 77.89 | 75.70 | 75.55 | 1118 | Chiefs |
| 10 | 6 | Dre Greenlaw | 77.68 | 81.20 | 76.89 | 850 | 49ers |
| 11 | 7 | Kaden Elliss | 76.81 | 78.88 | 71.26 | 632 | Saints |
| 12 | 8 | Tremaine Edmunds | 76.25 | 78.29 | 73.48 | 760 | Bills |
| 13 | 9 | Jerome Baker | 74.76 | 78.00 | 68.74 | 1010 | Dolphins |

### Starter (53 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Bobby Okereke | 73.82 | 73.30 | 70.42 | 970 | Colts |
| 15 | 2 | Frankie Luvu | 73.35 | 74.80 | 69.49 | 941 | Panthers |
| 16 | 3 | Logan Wilson | 73.19 | 72.70 | 72.34 | 954 | Bengals |
| 17 | 4 | De'Vondre Campbell | 73.05 | 74.85 | 69.94 | 694 | Packers |
| 18 | 5 | Jahlani Tavai | 73.02 | 71.26 | 72.37 | 570 | Patriots |
| 19 | 6 | Roquan Smith | 72.76 | 70.60 | 70.03 | 1039 | Ravens |
| 20 | 7 | Shaq Thompson | 72.60 | 72.30 | 69.52 | 1089 | Panthers |
| 21 | 8 | E.J. Speed | 72.20 | 71.42 | 72.55 | 316 | Colts |
| 22 | 9 | C.J. Mosley | 72.08 | 69.80 | 69.74 | 1113 | Jets |
| 23 | 10 | Ben Niemann | 72.01 | 68.45 | 70.91 | 484 | Cardinals |
| 24 | 11 | David Long Jr. | 71.74 | 75.39 | 70.91 | 740 | Titans |
| 25 | 12 | Leighton Vander Esch | 71.52 | 70.50 | 70.75 | 745 | Cowboys |
| 26 | 13 | Matt Milano | 71.23 | 73.70 | 67.94 | 946 | Bills |
| 27 | 14 | Oren Burks | 71.11 | 68.37 | 73.03 | 156 | 49ers |
| 28 | 15 | Foyesade Oluokun | 71.11 | 69.60 | 68.15 | 1145 | Jaguars |
| 29 | 16 | Denzel Perryman | 70.67 | 71.68 | 69.49 | 555 | Raiders |
| 30 | 17 | Patrick Queen | 70.26 | 70.00 | 66.26 | 1024 | Ravens |
| 31 | 18 | Cory Littleton | 69.50 | 69.81 | 67.11 | 372 | Panthers |
| 32 | 19 | Josey Jewell | 69.24 | 71.70 | 69.80 | 825 | Broncos |
| 33 | 20 | Jordan Hicks | 68.91 | 65.40 | 67.08 | 934 | Vikings |
| 34 | 21 | Anthony Walker Jr. | 68.84 | 69.08 | 72.55 | 120 | Browns |
| 35 | 22 | Brian Asamoah II | 68.35 | 67.52 | 71.60 | 121 | Vikings |
| 36 | 23 | Willie Gay | 67.71 | 68.26 | 66.81 | 607 | Chiefs |
| 37 | 24 | Ernest Jones | 67.45 | 63.38 | 68.19 | 723 | Rams |
| 38 | 25 | Kyzir White | 67.10 | 65.00 | 65.37 | 843 | Eagles |
| 39 | 26 | Sione Takitaki | 66.65 | 65.07 | 66.79 | 498 | Browns |
| 40 | 27 | Leo Chenal | 66.58 | 63.39 | 64.54 | 262 | Chiefs |
| 41 | 28 | Azeez Al-Shaair | 66.45 | 64.82 | 67.62 | 313 | 49ers |
| 42 | 29 | Jamin Davis | 66.34 | 62.90 | 65.45 | 833 | Commanders |
| 43 | 30 | Cole Holcomb | 66.22 | 65.56 | 68.94 | 446 | Commanders |
| 44 | 31 | Drue Tranquill | 65.86 | 66.50 | 65.26 | 977 | Chargers |
| 45 | 32 | Eric Kendricks | 65.61 | 61.10 | 66.08 | 1094 | Vikings |
| 46 | 33 | Jeremiah Owusu-Koramoah | 65.58 | 64.87 | 66.67 | 535 | Browns |
| 47 | 34 | Cody Barton | 65.54 | 63.70 | 66.61 | 894 | Seahawks |
| 48 | 35 | Pete Werner | 65.49 | 64.28 | 66.30 | 596 | Saints |
| 49 | 36 | Malcolm Rodriguez | 65.28 | 62.42 | 64.00 | 611 | Lions |
| 50 | 37 | Damone Clark | 64.62 | 63.83 | 67.84 | 398 | Cowboys |
| 51 | 38 | Matthew Adams | 64.58 | 64.61 | 68.24 | 189 | Bears |
| 52 | 39 | Chris Board | 64.25 | 61.93 | 62.89 | 158 | Lions |
| 53 | 40 | Kwon Alexander | 64.22 | 62.47 | 63.52 | 558 | Jets |
| 54 | 41 | Malik Harrison | 64.20 | 63.57 | 64.09 | 248 | Ravens |
| 55 | 42 | Rashaan Evans | 64.15 | 59.60 | 64.49 | 1104 | Falcons |
| 56 | 43 | Derrick Barnes | 63.76 | 61.49 | 63.92 | 346 | Lions |
| 57 | 44 | Akeem Davis-Gaither | 63.69 | 60.42 | 64.55 | 228 | Bengals |
| 58 | 45 | Jake Hansen | 63.58 | 64.55 | 64.65 | 205 | Texans |
| 59 | 46 | Ezekiel Turner | 63.58 | 61.44 | 71.31 | 108 | Cardinals |
| 60 | 47 | Mykal Walker | 63.08 | 58.74 | 62.59 | 769 | Falcons |
| 61 | 48 | Devin Bush | 63.01 | 59.01 | 64.67 | 659 | Steelers |
| 62 | 49 | Monty Rice | 62.87 | 62.20 | 67.48 | 366 | Titans |
| 63 | 50 | Zaven Collins | 62.60 | 59.80 | 62.39 | 1025 | Cardinals |
| 64 | 51 | Alex Anzalone | 62.56 | 59.20 | 61.71 | 1076 | Lions |
| 65 | 52 | Elandon Roberts | 62.30 | 57.37 | 62.05 | 676 | Dolphins |
| 66 | 53 | Jack Sanborn | 62.23 | 62.85 | 66.47 | 330 | Bears |

### Rotation/backup (48 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 67 | 1 | Duke Riley | 61.96 | 59.40 | 60.71 | 364 | Dolphins |
| 68 | 2 | Joe Thomas | 61.71 | 62.13 | 63.32 | 413 | Bears |
| 69 | 3 | Quay Walker | 61.60 | 52.00 | 63.83 | 846 | Packers |
| 70 | 4 | Darius Harris | 61.40 | 62.63 | 64.03 | 292 | Chiefs |
| 71 | 5 | Zaire Franklin | 61.31 | 57.00 | 61.49 | 1136 | Colts |
| 72 | 6 | Zach Cunningham | 61.23 | 60.15 | 64.05 | 205 | Titans |
| 73 | 7 | Chad Muma | 61.02 | 56.99 | 62.47 | 286 | Jaguars |
| 74 | 8 | Robert Spillane | 61.00 | 53.65 | 64.95 | 588 | Steelers |
| 75 | 9 | Damien Wilson | 60.97 | 58.55 | 61.99 | 204 | Panthers |
| 76 | 10 | Jarrad Davis | 60.76 | 63.24 | 64.09 | 106 | Giants |
| 77 | 11 | Anthony Barr | 60.55 | 59.73 | 63.08 | 608 | Cowboys |
| 78 | 12 | Jaylon Smith | 60.33 | 56.30 | 62.86 | 626 | Giants |
| 79 | 13 | Jordyn Brooks | 60.26 | 52.80 | 61.96 | 1026 | Seahawks |
| 80 | 14 | Nicholas Morrow | 60.24 | 54.00 | 60.65 | 1086 | Bears |
| 81 | 15 | Myles Jack | 59.86 | 53.91 | 61.64 | 692 | Steelers |
| 82 | 16 | Christian Kirksey | 59.66 | 56.10 | 60.08 | 1139 | Texans |
| 83 | 17 | Josh Bynes | 59.45 | 55.69 | 63.57 | 269 | Ravens |
| 84 | 18 | Isaiah McDuffie | 59.24 | 59.35 | 65.16 | 175 | Packers |
| 85 | 19 | Kenneth Murray Jr. | 59.21 | 48.58 | 63.89 | 718 | Chargers |
| 86 | 20 | Divine Deablo | 59.19 | 58.80 | 64.10 | 463 | Raiders |
| 87 | 21 | David Mayo | 58.29 | 56.30 | 61.98 | 202 | Commanders |
| 88 | 22 | Quincy Williams | 58.09 | 55.28 | 59.98 | 792 | Jets |
| 89 | 23 | Terrel Bernard | 58.06 | 61.84 | 65.15 | 111 | Bills |
| 90 | 24 | Devin Lloyd | 57.60 | 48.30 | 59.64 | 925 | Jaguars |
| 91 | 25 | Krys Barnes | 57.37 | 54.48 | 61.65 | 141 | Packers |
| 92 | 26 | Tony Fields II | 57.28 | 54.20 | 61.05 | 276 | Browns |
| 93 | 27 | Dylan Cole | 56.30 | 55.10 | 60.26 | 439 | Titans |
| 94 | 28 | Jordan Kunaszyk | 55.86 | 56.96 | 62.42 | 101 | Browns |
| 95 | 29 | Raekwon McMillan | 55.80 | 50.45 | 57.01 | 250 | Patriots |
| 96 | 30 | Mack Wilson Sr. | 55.65 | 53.11 | 57.43 | 234 | Patriots |
| 97 | 31 | Jonas Griffith | 55.61 | 55.46 | 60.86 | 336 | Broncos |
| 98 | 32 | Blake Cashman | 55.48 | 57.44 | 59.31 | 149 | Texans |
| 99 | 33 | Deion Jones | 55.18 | 50.28 | 57.52 | 422 | Browns |
| 100 | 34 | Devin White | 55.08 | 45.50 | 57.50 | 1075 | Buccaneers |
| 101 | 35 | Jack Gibbens | 55.00 | 60.00 | 63.76 | 214 | Titans |
| 102 | 36 | Garret Wallow | 54.77 | 53.56 | 60.11 | 124 | Texans |
| 103 | 37 | A.J. Klein | 54.30 | 55.08 | 57.75 | 104 | Bills |
| 104 | 38 | Tyrel Dodson | 53.81 | 53.99 | 59.46 | 220 | Bills |
| 105 | 39 | Micah McFadden | 53.79 | 44.49 | 57.79 | 435 | Giants |
| 106 | 40 | Jon Bostic | 53.61 | 52.13 | 58.18 | 263 | Commanders |
| 107 | 41 | Troy Andersen | 53.57 | 44.84 | 56.20 | 481 | Falcons |
| 108 | 42 | Tanner Vallejo | 53.16 | 48.92 | 56.53 | 282 | Cardinals |
| 109 | 43 | Jayon Brown | 49.20 | 43.55 | 56.52 | 423 | Raiders |
| 110 | 44 | Tae Crowder | 49.15 | 40.00 | 54.79 | 445 | Steelers |
| 111 | 45 | Jacob Phillips | 48.90 | 45.44 | 57.23 | 320 | Browns |
| 112 | 46 | Luke Masterson | 48.65 | 41.09 | 55.40 | 344 | Raiders |
| 113 | 47 | Kamu Grugier-Hill | 48.61 | 40.00 | 54.83 | 418 | Cardinals |
| 114 | 48 | Christian Harris | 48.00 | 40.00 | 54.07 | 711 | Texans |

## QB — Quarterback

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Joe Burrow | 84.38 | 88.74 | 77.11 | 722 | Bengals |
| 2 | 2 | Patrick Mahomes | 83.68 | 85.22 | 77.88 | 768 | Chiefs |
| 3 | 3 | Josh Allen | 81.14 | 83.49 | 74.67 | 696 | Bills |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Tom Brady | 78.67 | 83.23 | 70.39 | 808 | Buccaneers |
| 5 | 2 | Aaron Rodgers | 77.85 | 80.52 | 72.78 | 624 | Packers |
| 6 | 3 | Kirk Cousins | 77.46 | 78.94 | 72.05 | 744 | Vikings |
| 7 | 4 | Justin Herbert | 76.80 | 79.63 | 69.93 | 791 | Chargers |
| 8 | 5 | Jalen Hurts | 76.51 | 77.73 | 75.10 | 573 | Eagles |
| 9 | 6 | Tua Tagovailoa | 74.83 | 74.45 | 76.88 | 455 | Dolphins |

### Starter (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Geno Smith | 73.85 | 75.80 | 77.41 | 693 | Seahawks |
| 11 | 2 | Dak Prescott | 72.78 | 73.73 | 72.51 | 453 | Cowboys |
| 12 | 3 | Ryan Tannehill | 72.54 | 74.72 | 71.75 | 397 | Titans |
| 13 | 4 | Jared Goff | 72.44 | 69.40 | 71.52 | 663 | Lions |
| 14 | 5 | Trevor Lawrence | 71.58 | 68.81 | 69.48 | 672 | Jaguars |
| 15 | 6 | Andy Dalton | 71.38 | 75.35 | 71.16 | 431 | Saints |
| 16 | 7 | Derek Carr | 71.33 | 70.41 | 69.54 | 575 | Raiders |
| 17 | 8 | Matthew Stafford | 71.32 | 71.55 | 72.33 | 347 | Rams |
| 18 | 9 | Jimmy Garoppolo | 71.05 | 69.04 | 76.49 | 348 | 49ers |
| 19 | 10 | Russell Wilson | 70.01 | 68.68 | 69.26 | 613 | Broncos |
| 20 | 11 | Mac Jones | 69.70 | 70.97 | 67.79 | 523 | Patriots |
| 21 | 12 | Daniel Jones | 69.64 | 71.33 | 67.04 | 612 | Giants |
| 22 | 13 | Kyler Murray | 69.24 | 71.30 | 67.42 | 473 | Cardinals |
| 23 | 14 | Lamar Jackson | 68.70 | 69.80 | 69.86 | 399 | Ravens |
| 24 | 15 | Matt Ryan | 67.66 | 68.70 | 64.97 | 535 | Colts |
| 25 | 16 | Jacoby Brissett | 65.85 | 72.84 | 67.54 | 448 | Browns |
| 26 | 17 | Brock Purdy | 63.56 | 68.23 | 76.02 | 202 | 49ers |
| 27 | 18 | Bailey Zappe | 63.02 | 65.89 | 76.49 | 109 | Patriots |
| 28 | 19 | Davis Mills | 62.82 | 62.42 | 62.63 | 562 | Texans |
| 29 | 20 | Kenny Pickett | 62.56 | 71.58 | 61.10 | 477 | Steelers |

### Rotation/backup (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Carson Wentz | 61.21 | 61.94 | 62.94 | 337 | Commanders |
| 31 | 2 | Mitch Trubisky | 61.17 | 66.23 | 67.26 | 207 | Steelers |
| 32 | 3 | Sam Darnold | 61.09 | 59.28 | 68.72 | 173 | Panthers |
| 33 | 4 | Jameis Winston | 60.65 | 64.38 | 68.15 | 137 | Saints |
| 34 | 5 | Marcus Mariota | 60.61 | 62.54 | 67.18 | 373 | Falcons |
| 35 | 6 | Justin Fields | 60.19 | 56.36 | 65.22 | 474 | Bears |
| 36 | 7 | Jarrett Stidham | 60.06 | 60.56 | 68.89 | 107 | Raiders |
| 37 | 8 | Taylor Heinicke | 59.63 | 55.22 | 67.36 | 304 | Commanders |
| 38 | 9 | Baker Mayfield | 59.61 | 58.91 | 61.68 | 411 | Rams |
| 39 | 10 | Deshaun Watson | 59.33 | 63.19 | 63.93 | 218 | Browns |
| 40 | 11 | P.J. Walker | 58.71 | 59.40 | 64.37 | 124 | Panthers |
| 41 | 12 | Colt McCoy | 58.60 | 62.43 | 61.45 | 163 | Cardinals |
| 42 | 13 | Desmond Ridder | 58.40 | 60.05 | 62.59 | 144 | Falcons |
| 43 | 14 | Mike White | 58.01 | 58.70 | 62.12 | 189 | Jets |
| 44 | 15 | Cooper Rush | 57.57 | 57.77 | 61.09 | 182 | Cowboys |
| 45 | 16 | Tyler Huntley | 57.41 | 59.75 | 58.83 | 141 | Ravens |
| 46 | 17 | Sam Ehlinger | 56.89 | 58.31 | 57.90 | 132 | Colts |
| 47 | 18 | Skylar Thompson | 56.68 | 61.50 | 54.47 | 127 | Dolphins |
| 48 | 19 | Joe Flacco | 55.94 | 55.28 | 56.55 | 212 | Jets |
| 49 | 20 | Zach Wilson | 54.23 | 51.32 | 58.53 | 289 | Jets |

## S — Safety

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Tyrann Mathieu | 86.12 | 87.90 | 81.26 | 1128 | Saints |
| 2 | 2 | Kevin Byard | 85.72 | 81.00 | 84.70 | 1139 | Titans |
| 3 | 3 | Minkah Fitzpatrick | 85.64 | 83.30 | 84.30 | 939 | Steelers |
| 4 | 4 | Rodney McLeod | 85.59 | 85.30 | 83.42 | 1034 | Colts |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Duron Harmon | 79.73 | 77.60 | 76.98 | 1076 | Raiders |
| 6 | 2 | Kamren Curl | 79.34 | 79.53 | 79.94 | 727 | Commanders |
| 7 | 3 | Damontae Kazee | 78.94 | 73.86 | 84.58 | 273 | Steelers |
| 8 | 4 | Quandre Diggs | 78.38 | 76.40 | 75.53 | 1156 | Seahawks |
| 9 | 5 | Kyle Dugger | 78.07 | 77.80 | 76.07 | 752 | Patriots |
| 10 | 6 | Jessie Bates III | 78.05 | 72.90 | 78.40 | 1016 | Bengals |
| 11 | 7 | Taylor Rapp | 77.31 | 72.90 | 78.04 | 976 | Rams |
| 12 | 8 | Kyle Hamilton | 77.24 | 73.68 | 76.43 | 547 | Ravens |
| 13 | 9 | Malik Hooker | 76.97 | 72.40 | 79.85 | 860 | Cowboys |
| 14 | 10 | Adrian Phillips | 76.84 | 71.06 | 76.53 | 702 | Patriots |
| 15 | 11 | Jordan Poyer | 76.71 | 75.77 | 75.92 | 754 | Bills |
| 16 | 12 | Marcus Williams | 76.62 | 70.85 | 80.45 | 637 | Ravens |
| 17 | 13 | Julian Love | 76.62 | 71.50 | 76.57 | 1006 | Giants |
| 18 | 14 | Josh Metellus | 76.58 | 71.60 | 83.81 | 259 | Vikings |
| 19 | 15 | Rudy Ford | 76.50 | 73.04 | 76.11 | 443 | Packers |
| 20 | 16 | Eddie Jackson | 76.30 | 73.61 | 77.26 | 697 | Bears |
| 21 | 17 | Devin McCourty | 75.93 | 67.60 | 77.32 | 1097 | Patriots |
| 22 | 18 | Derwin James Jr. | 75.21 | 74.40 | 75.93 | 835 | Chargers |
| 23 | 19 | Harrison Smith | 74.77 | 69.50 | 76.18 | 912 | Vikings |
| 24 | 20 | Justin Simmons | 74.39 | 74.21 | 72.79 | 808 | Broncos |
| 25 | 21 | Antoine Winfield Jr. | 74.15 | 66.45 | 78.25 | 764 | Buccaneers |
| 26 | 22 | Justin Reid | 74.05 | 73.80 | 71.85 | 1112 | Chiefs |

### Starter (55 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | Budda Baker | 73.34 | 69.10 | 73.18 | 969 | Cardinals |
| 28 | 2 | Marcus Maye | 73.05 | 70.63 | 77.16 | 669 | Saints |
| 29 | 3 | Juan Thornhill | 72.46 | 68.70 | 71.29 | 1043 | Chiefs |
| 30 | 4 | Caden Sterns | 72.33 | 72.28 | 76.65 | 274 | Broncos |
| 31 | 5 | Tracy Walker III | 72.12 | 69.95 | 77.06 | 139 | Lions |
| 32 | 6 | P.J. Locke | 71.98 | 69.84 | 76.85 | 112 | Broncos |
| 33 | 7 | Jayron Kearse | 71.94 | 63.59 | 75.11 | 815 | Cowboys |
| 34 | 8 | Darrick Forrest | 71.86 | 69.00 | 75.12 | 849 | Commanders |
| 35 | 9 | Jordan Whitehead | 71.72 | 70.60 | 69.18 | 1129 | Jets |
| 36 | 10 | Jabrill Peppers | 71.46 | 65.80 | 74.51 | 398 | Patriots |
| 37 | 11 | Richie Grant | 71.28 | 69.00 | 68.63 | 1117 | Falcons |
| 38 | 12 | Andre Cisco | 71.18 | 68.00 | 72.56 | 992 | Jaguars |
| 39 | 13 | Jimmie Ward | 71.17 | 68.10 | 72.22 | 509 | 49ers |
| 40 | 14 | Terrell Edmunds | 71.15 | 68.00 | 70.26 | 886 | Steelers |
| 41 | 15 | Talanoa Hufanga | 71.09 | 67.80 | 70.22 | 1029 | 49ers |
| 42 | 16 | Nasir Adderley | 70.97 | 68.20 | 69.94 | 882 | Chargers |
| 43 | 17 | M.J. Stewart | 70.78 | 64.88 | 76.43 | 178 | Texans |
| 44 | 18 | Bobby McCain | 70.75 | 69.50 | 68.88 | 970 | Commanders |
| 45 | 19 | Andrew Adams | 70.50 | 73.88 | 71.18 | 726 | Titans |
| 46 | 20 | Reed Blankenship | 70.34 | 66.90 | 77.28 | 292 | Eagles |
| 47 | 21 | Chuck Clark | 69.97 | 61.20 | 71.95 | 1091 | Ravens |
| 48 | 22 | Jevon Holland | 69.84 | 63.80 | 70.07 | 1123 | Dolphins |
| 49 | 23 | Amani Hooker | 69.63 | 70.09 | 70.55 | 522 | Titans |
| 50 | 24 | Landon Collins | 69.53 | 73.02 | 71.48 | 160 | Giants |
| 51 | 25 | Geno Stone | 69.45 | 70.86 | 71.00 | 450 | Ravens |
| 52 | 26 | Kerby Joseph | 69.39 | 65.00 | 71.09 | 875 | Lions |
| 53 | 27 | Xavier Woods | 69.05 | 64.00 | 69.44 | 1001 | Panthers |
| 54 | 28 | Tashaun Gipson Sr. | 69.01 | 58.00 | 73.65 | 1036 | 49ers |
| 55 | 29 | DeShon Elliott | 68.92 | 64.10 | 72.67 | 859 | Lions |
| 56 | 30 | Grant Delpit | 68.84 | 62.50 | 69.48 | 1086 | Browns |
| 57 | 31 | John Johnson III | 68.74 | 66.60 | 66.58 | 1056 | Browns |
| 58 | 32 | Tony Jefferson | 68.51 | 71.35 | 72.20 | 164 | Giants |
| 59 | 33 | Jalen Thompson | 68.32 | 59.70 | 72.19 | 1098 | Cardinals |
| 60 | 34 | Jalen Pitre | 68.15 | 65.80 | 65.55 | 1088 | Texans |
| 61 | 35 | Verone McKinley III | 67.90 | 67.01 | 71.20 | 251 | Dolphins |
| 62 | 36 | Andrew Wingard | 67.87 | 62.12 | 72.89 | 217 | Jaguars |
| 63 | 37 | Vonn Bell | 67.65 | 65.50 | 65.70 | 1023 | Bengals |
| 64 | 38 | C.J. Moore | 67.27 | 66.00 | 72.77 | 106 | Lions |
| 65 | 39 | Micah Hyde | 66.96 | 65.82 | 71.12 | 101 | Bills |
| 66 | 40 | Donovan Wilson | 66.91 | 64.50 | 67.75 | 959 | Cowboys |
| 67 | 41 | Jaquan Brisker | 66.63 | 65.00 | 65.51 | 954 | Bears |
| 68 | 42 | Jeremy Reaves | 66.45 | 64.59 | 72.08 | 149 | Commanders |
| 69 | 43 | Daniel Sorensen | 66.44 | 60.67 | 69.76 | 166 | Saints |
| 70 | 44 | Bryan Cook | 66.40 | 60.58 | 67.09 | 341 | Chiefs |
| 71 | 45 | C.J. Gardner-Johnson | 66.18 | 63.95 | 66.57 | 729 | Eagles |
| 72 | 46 | Kareem Jackson | 65.80 | 60.90 | 65.49 | 1137 | Broncos |
| 73 | 47 | Justin Evans | 64.67 | 62.68 | 65.05 | 391 | Saints |
| 74 | 48 | Eric Murray | 63.89 | 59.40 | 66.76 | 118 | Texans |
| 75 | 49 | Julian Blackmon | 63.62 | 57.57 | 68.40 | 720 | Colts |
| 76 | 50 | Damar Hamlin | 63.41 | 62.80 | 64.91 | 845 | Bills |
| 77 | 51 | K'Von Wallace | 63.15 | 58.29 | 67.07 | 168 | Eagles |
| 78 | 52 | Xavier McKinney | 62.75 | 58.04 | 67.72 | 554 | Giants |
| 79 | 53 | Rodney Thomas II | 62.65 | 55.14 | 65.45 | 720 | Colts |
| 80 | 54 | Dax Hill | 62.45 | 57.76 | 63.38 | 130 | Bengals |
| 81 | 55 | Keanu Neal | 62.38 | 62.80 | 61.94 | 580 | Buccaneers |

### Rotation/backup (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 82 | 1 | DeAndre Houston-Carson | 61.95 | 59.67 | 63.27 | 414 | Bears |
| 83 | 2 | Mike Edwards | 61.59 | 57.81 | 63.63 | 814 | Buccaneers |
| 84 | 3 | Juston Burris | 60.67 | 62.63 | 64.06 | 219 | Panthers |
| 85 | 4 | Jeremy Chinn | 60.37 | 58.11 | 61.16 | 684 | Panthers |
| 86 | 5 | Jaylinn Hawkins | 60.09 | 56.10 | 62.25 | 955 | Falcons |
| 87 | 6 | Alohi Gilman | 59.52 | 54.54 | 62.94 | 474 | Chargers |
| 88 | 7 | Tony Adams | 59.34 | 60.72 | 65.56 | 118 | Jets |
| 89 | 8 | Tre'von Moehrig | 59.07 | 49.00 | 62.85 | 906 | Raiders |
| 90 | 9 | Eric Rowe | 59.02 | 51.30 | 61.46 | 567 | Dolphins |
| 91 | 10 | Roderic Teamer | 58.98 | 57.23 | 62.57 | 286 | Raiders |
| 92 | 11 | Brandon Jones | 58.31 | 55.07 | 61.79 | 347 | Dolphins |
| 93 | 12 | Adrian Amos | 58.15 | 45.60 | 62.35 | 977 | Packers |
| 94 | 13 | Will Parks | 57.84 | 57.75 | 61.77 | 210 | Jets |
| 95 | 14 | Lamarcus Joyner | 57.48 | 53.40 | 62.22 | 872 | Jets |
| 96 | 15 | Marcus Epps | 57.14 | 44.70 | 62.18 | 1095 | Eagles |
| 97 | 16 | Rayshawn Jenkins | 56.77 | 50.70 | 57.73 | 1126 | Jaguars |
| 98 | 17 | Percy Butler | 56.72 | 55.28 | 59.40 | 134 | Commanders |
| 99 | 18 | Jaquan Johnson | 56.10 | 57.48 | 60.29 | 227 | Bills |
| 100 | 19 | Jonathan Owens | 55.38 | 49.00 | 62.11 | 970 | Texans |
| 101 | 20 | Ronnie Harrison | 55.20 | 51.93 | 56.71 | 259 | Browns |
| 102 | 21 | Russ Yeast | 55.14 | 54.84 | 59.03 | 113 | Rams |
| 103 | 22 | Nick Scott | 55.06 | 43.40 | 60.61 | 984 | Rams |
| 104 | 23 | Dean Marlowe | 55.03 | 55.32 | 56.30 | 209 | Bills |
| 105 | 24 | Dane Belton | 54.18 | 46.25 | 59.22 | 390 | Giants |
| 106 | 25 | Darnell Savage | 54.09 | 43.84 | 57.45 | 819 | Packers |
| 107 | 26 | Johnathan Abram | 53.96 | 49.06 | 56.53 | 593 | Seahawks |
| 108 | 27 | Nick Cross | 53.85 | 60.32 | 59.15 | 122 | Colts |
| 109 | 28 | Josh Jones | 53.68 | 55.54 | 55.95 | 376 | Seahawks |

## T — Tackle

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Trent Williams | 96.94 | 91.82 | 96.19 | 871 | 49ers |
| 2 | 2 | Christian Darrisaw | 94.02 | 88.20 | 93.73 | 853 | Vikings |
| 3 | 3 | Andrew Thomas | 92.81 | 89.10 | 91.11 | 1049 | Giants |
| 4 | 4 | Kaleb McGary | 92.04 | 86.60 | 91.50 | 1051 | Falcons |
| 5 | 5 | Lane Johnson | 90.22 | 83.11 | 90.80 | 972 | Eagles |
| 6 | 6 | Brian O'Neill | 89.50 | 82.70 | 89.86 | 1052 | Vikings |
| 7 | 7 | Kolton Miller | 89.05 | 84.10 | 88.18 | 1035 | Raiders |
| 8 | 8 | Tristan Wirfs | 88.85 | 83.40 | 88.31 | 931 | Buccaneers |
| 9 | 9 | Penei Sewell | 88.38 | 80.60 | 89.40 | 1142 | Lions |
| 10 | 10 | Rashawn Slater | 87.94 | 75.67 | 91.95 | 175 | Chargers |
| 11 | 11 | Laremy Tunsil | 87.21 | 80.00 | 87.85 | 1061 | Texans |
| 12 | 12 | Morgan Moses | 86.94 | 78.10 | 88.66 | 1022 | Ravens |
| 13 | 13 | David Bakhtiari | 85.93 | 77.17 | 87.61 | 597 | Packers |
| 14 | 14 | Ryan Ramczyk | 85.43 | 77.63 | 86.47 | 936 | Saints |
| 15 | 15 | Jake Matthews | 84.88 | 77.20 | 85.83 | 1047 | Falcons |
| 16 | 16 | Jermaine Eluemunor | 84.54 | 74.95 | 86.76 | 940 | Raiders |
| 17 | 17 | Terron Armstead | 84.40 | 75.77 | 85.99 | 687 | Dolphins |
| 18 | 18 | Braden Smith | 84.40 | 75.50 | 86.16 | 1066 | Colts |
| 19 | 19 | Jordan Mailata | 84.06 | 76.50 | 84.94 | 1024 | Eagles |
| 20 | 20 | Braxton Jones | 83.93 | 75.40 | 85.45 | 1033 | Bears |
| 21 | 21 | Terence Steele | 83.90 | 72.67 | 87.22 | 818 | Cowboys |
| 22 | 22 | Orlando Brown Jr. | 83.02 | 75.80 | 83.66 | 1133 | Chiefs |
| 23 | 23 | Garett Bolles | 82.62 | 69.61 | 87.13 | 325 | Broncos |
| 24 | 24 | Josh Jones | 82.42 | 72.56 | 84.83 | 622 | Cardinals |
| 25 | 25 | Taylor Decker | 82.40 | 74.40 | 83.56 | 1142 | Lions |
| 26 | 26 | Rob Havenstein | 81.56 | 73.20 | 82.97 | 1019 | Rams |
| 27 | 27 | Bernhard Raimann | 81.54 | 71.28 | 84.22 | 709 | Colts |
| 28 | 28 | Cam Fleming | 81.39 | 72.54 | 83.13 | 976 | Broncos |
| 29 | 29 | Dion Dawkins | 81.34 | 73.37 | 82.49 | 953 | Bills |
| 30 | 30 | Mike McGlinchey | 81.30 | 71.50 | 83.66 | 1036 | 49ers |
| 31 | 31 | Tyler Smith | 80.56 | 71.40 | 82.50 | 1144 | Cowboys |
| 32 | 32 | Ty Nsekhe | 80.21 | 66.95 | 84.88 | 424 | Rams |

### Good (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | D.J. Humphries | 79.72 | 70.56 | 81.66 | 575 | Cardinals |
| 34 | 2 | Sam Cosmi | 79.50 | 68.94 | 82.38 | 585 | Commanders |
| 35 | 3 | Patrick Mekari | 79.09 | 68.24 | 82.15 | 378 | Ravens |
| 36 | 4 | Ronnie Stanley | 78.90 | 69.47 | 81.02 | 602 | Ravens |
| 37 | 5 | Charles Leno Jr. | 78.89 | 71.60 | 79.59 | 1189 | Commanders |
| 38 | 6 | Kelvin Beachum | 78.86 | 70.60 | 80.20 | 1178 | Cardinals |
| 39 | 7 | Taylor Moton | 78.82 | 69.30 | 81.00 | 1018 | Panthers |
| 40 | 8 | Abraham Lucas | 78.76 | 68.46 | 81.46 | 975 | Seahawks |
| 41 | 9 | Tytus Howard | 78.27 | 67.90 | 81.02 | 997 | Texans |
| 42 | 10 | Conor McDermott | 78.19 | 63.86 | 83.58 | 437 | Patriots |
| 43 | 11 | Jason Peters | 78.09 | 67.14 | 81.23 | 235 | Cowboys |
| 44 | 12 | Cornelius Lucas | 77.48 | 66.89 | 80.38 | 671 | Commanders |
| 45 | 13 | Trent Brown | 77.33 | 67.40 | 79.79 | 1030 | Patriots |
| 46 | 14 | Jack Conklin | 77.17 | 66.55 | 80.09 | 913 | Browns |
| 47 | 15 | Josh Wells | 77.15 | 63.74 | 81.92 | 326 | Buccaneers |
| 48 | 16 | Cam Robinson | 77.14 | 66.93 | 79.78 | 913 | Jaguars |
| 49 | 17 | Brandon Shell | 77.02 | 64.54 | 81.18 | 761 | Dolphins |
| 50 | 18 | Zach Tom | 77.02 | 65.85 | 80.30 | 489 | Packers |
| 51 | 19 | Ikem Ekwonu | 76.99 | 65.30 | 80.62 | 1018 | Panthers |
| 52 | 20 | Jamaree Salyer | 76.67 | 69.20 | 77.48 | 989 | Chargers |
| 53 | 21 | Joe Noteboom | 76.13 | 64.02 | 80.03 | 325 | Rams |
| 54 | 22 | Larry Borom | 75.77 | 63.44 | 79.82 | 528 | Bears |
| 55 | 23 | Charles Cross | 75.28 | 63.70 | 78.83 | 1088 | Seahawks |
| 56 | 24 | Andrew Wylie | 75.06 | 63.10 | 78.86 | 1093 | Chiefs |
| 57 | 25 | Yodny Cajuste | 74.94 | 62.55 | 79.03 | 197 | Patriots |
| 58 | 26 | Riley Reiff | 74.90 | 63.63 | 78.24 | 542 | Bears |
| 59 | 27 | Jaylon Moore | 74.67 | 62.72 | 78.47 | 184 | 49ers |
| 60 | 28 | Thayer Munford Jr. | 74.58 | 61.96 | 78.82 | 369 | Raiders |
| 61 | 29 | Calvin Anderson | 74.53 | 63.34 | 77.83 | 439 | Broncos |
| 62 | 30 | James Hudson III | 74.43 | 58.79 | 80.69 | 296 | Browns |
| 63 | 31 | Trevor Penning | 74.32 | 65.44 | 76.07 | 124 | Saints |
| 64 | 32 | Yosh Nijman | 74.20 | 62.72 | 77.69 | 756 | Packers |
| 65 | 33 | James Hurst | 74.15 | 63.18 | 77.29 | 973 | Saints |
| 66 | 34 | La'el Collins | 74.09 | 57.92 | 80.70 | 951 | Bengals |
| 67 | 35 | Jedrick Wills Jr. | 74.06 | 62.90 | 77.33 | 1152 | Browns |

### Starter (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 68 | 1 | Dan Moore Jr. | 73.92 | 62.40 | 77.43 | 1160 | Steelers |
| 69 | 2 | Marcus Cannon | 73.78 | 62.50 | 77.13 | 207 | Patriots |
| 70 | 3 | Brandon Walton | 73.53 | 58.98 | 79.07 | 234 | Buccaneers |
| 71 | 4 | Walker Little | 73.52 | 60.54 | 78.00 | 234 | Jaguars |
| 72 | 5 | Chukwuma Okorafor | 73.02 | 61.20 | 76.74 | 1159 | Steelers |
| 73 | 6 | Jawaan Taylor | 72.99 | 58.70 | 78.35 | 1095 | Jaguars |
| 74 | 7 | Dennis Kelly | 72.61 | 59.46 | 77.21 | 240 | Colts |
| 75 | 8 | Trey Pipkins III | 71.75 | 59.24 | 75.93 | 898 | Chargers |
| 76 | 9 | Duane Brown | 71.68 | 57.97 | 76.65 | 744 | Jets |
| 77 | 10 | Jonah Williams | 71.63 | 61.20 | 74.41 | 1101 | Bengals |
| 78 | 11 | Charlie Heck | 71.52 | 58.26 | 76.19 | 162 | Texans |
| 79 | 12 | David Quessenberry | 71.32 | 59.45 | 75.06 | 396 | Bills |
| 80 | 13 | Donovan Smith | 71.14 | 58.15 | 75.64 | 908 | Buccaneers |
| 81 | 14 | Tyron Smith | 71.12 | 59.00 | 75.03 | 271 | Cowboys |
| 82 | 15 | Blake Brandel | 70.94 | 57.52 | 75.72 | 274 | Vikings |
| 83 | 16 | Isaiah Wynn | 70.86 | 55.72 | 76.78 | 423 | Patriots |
| 84 | 17 | Billy Turner | 70.48 | 57.41 | 75.02 | 483 | Broncos |
| 85 | 18 | Matt Peart | 70.45 | 55.68 | 76.13 | 117 | Giants |
| 86 | 19 | Hakeem Adeniji | 70.34 | 57.21 | 74.92 | 220 | Bengals |
| 87 | 20 | Le'Raven Clark | 70.27 | 58.88 | 73.69 | 114 | Titans |
| 88 | 21 | Max Mitchell | 69.79 | 57.35 | 73.92 | 341 | Jets |
| 89 | 22 | Nicholas Petit-Frere | 68.23 | 52.49 | 74.55 | 937 | Titans |
| 90 | 23 | George Fant | 67.53 | 51.60 | 73.99 | 516 | Jets |
| 91 | 24 | Cedric Ogbuehi | 67.37 | 53.37 | 72.53 | 286 | Jets |
| 92 | 25 | Spencer Brown | 67.26 | 52.03 | 73.24 | 845 | Bills |
| 93 | 26 | Daniel Faalele | 66.82 | 55.94 | 69.90 | 169 | Ravens |
| 94 | 27 | Foster Sarell | 65.55 | 52.24 | 70.26 | 250 | Chargers |
| 95 | 28 | Landon Young | 64.50 | 50.97 | 69.35 | 207 | Saints |
| 96 | 29 | Evan Neal | 63.66 | 46.24 | 71.10 | 738 | Giants |
| 97 | 30 | Dennis Daley | 63.19 | 46.41 | 70.21 | 942 | Titans |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 98 | 1 | Greg Little | 60.91 | 41.40 | 69.75 | 528 | Dolphins |
| 99 | 2 | Stone Forsythe | 57.49 | 50.80 | 57.79 | 122 | Seahawks |

## TE — Tight End

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Travis Kelce | 85.91 | 91.10 | 78.28 | 661 | Chiefs |
| 2 | 2 | Mark Andrews | 80.56 | 78.26 | 77.92 | 457 | Ravens |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 3 | 1 | George Kittle | 78.90 | 80.22 | 73.85 | 494 | 49ers |
| 4 | 2 | Kyle Pitts | 78.21 | 69.61 | 79.77 | 233 | Falcons |
| 5 | 3 | Chigoziem Okonkwo | 77.58 | 68.02 | 79.79 | 179 | Titans |
| 6 | 4 | Pat Freiermuth | 75.80 | 74.25 | 72.66 | 494 | Steelers |
| 7 | 5 | Dallas Goedert | 75.73 | 74.20 | 72.59 | 416 | Eagles |
| 8 | 6 | David Njoku | 75.27 | 71.65 | 73.52 | 477 | Browns |
| 9 | 7 | Colby Parkinson | 74.06 | 66.27 | 75.08 | 235 | Seahawks |

### Starter (61 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Darren Waller | 73.23 | 69.66 | 71.45 | 263 | Raiders |
| 11 | 2 | MyCole Pruitt | 73.01 | 66.95 | 72.89 | 126 | Falcons |
| 12 | 3 | Will Dissly | 72.45 | 67.20 | 71.78 | 293 | Seahawks |
| 13 | 4 | Gerald Everett | 72.43 | 66.11 | 72.47 | 488 | Chargers |
| 14 | 5 | Jordan Akins | 72.21 | 68.71 | 70.37 | 326 | Texans |
| 15 | 6 | Evan Engram | 71.79 | 64.93 | 72.19 | 572 | Jaguars |
| 16 | 7 | T.J. Hockenson | 71.25 | 70.20 | 67.79 | 673 | Vikings |
| 17 | 8 | Juwan Johnson | 71.20 | 63.67 | 72.06 | 403 | Saints |
| 18 | 9 | Adam Trautman | 71.13 | 63.71 | 71.91 | 202 | Saints |
| 19 | 10 | Cole Kmet | 71.08 | 66.65 | 69.86 | 505 | Bears |
| 20 | 11 | Marcedes Lewis | 71.08 | 64.07 | 71.59 | 195 | Packers |
| 21 | 12 | Anthony Firkser | 71.06 | 59.64 | 74.51 | 107 | Falcons |
| 22 | 13 | Foster Moreau | 70.85 | 60.87 | 73.33 | 416 | Raiders |
| 23 | 14 | Noah Fant | 70.15 | 64.65 | 69.65 | 397 | Seahawks |
| 24 | 15 | Dawson Knox | 69.93 | 64.41 | 69.44 | 493 | Bills |
| 25 | 16 | Jake Ferguson | 69.82 | 62.74 | 70.37 | 152 | Cowboys |
| 26 | 17 | Hunter Henry | 69.70 | 58.28 | 73.14 | 470 | Patriots |
| 27 | 18 | Tyler Higbee | 69.02 | 62.54 | 69.18 | 538 | Rams |
| 28 | 19 | O.J. Howard | 68.76 | 58.12 | 71.69 | 161 | Texans |
| 29 | 20 | Dalton Schultz | 68.75 | 67.07 | 65.70 | 471 | Cowboys |
| 30 | 21 | Josh Oliver | 68.62 | 66.14 | 66.10 | 188 | Ravens |
| 31 | 22 | Taysom Hill | 68.55 | 66.84 | 65.52 | 127 | Saints |
| 32 | 23 | Jonnu Smith | 68.52 | 56.07 | 72.65 | 202 | Patriots |
| 33 | 24 | Jelani Woods | 68.52 | 62.86 | 68.12 | 225 | Colts |
| 34 | 25 | Austin Hooper | 68.22 | 67.38 | 64.61 | 352 | Titans |
| 35 | 26 | Cameron Brate | 68.19 | 54.73 | 72.99 | 257 | Buccaneers |
| 36 | 27 | Harrison Bryant | 68.09 | 59.53 | 69.63 | 292 | Browns |
| 37 | 28 | Zach Ertz | 67.97 | 62.17 | 67.67 | 399 | Cardinals |
| 38 | 29 | Parker Hesse | 67.93 | 60.71 | 68.58 | 230 | Falcons |
| 39 | 30 | Teagan Quitoriano | 67.57 | 54.68 | 72.00 | 166 | Texans |
| 40 | 31 | Mike Gesicki | 67.56 | 59.91 | 68.50 | 391 | Dolphins |
| 41 | 32 | Isaiah Likely | 67.24 | 64.27 | 65.05 | 294 | Ravens |
| 42 | 33 | Chris Manhertz | 67.24 | 59.07 | 68.52 | 144 | Jaguars |
| 43 | 34 | Jody Fortson | 67.24 | 62.24 | 66.41 | 114 | Chiefs |
| 44 | 35 | Hayden Hurst | 66.83 | 63.97 | 64.57 | 417 | Bengals |
| 45 | 36 | Cade Otton | 66.72 | 56.98 | 69.05 | 519 | Buccaneers |
| 46 | 37 | Mo Alie-Cox | 66.58 | 51.78 | 72.28 | 320 | Colts |
| 47 | 38 | Greg Dulcich | 66.51 | 60.29 | 66.49 | 346 | Broncos |
| 48 | 39 | Durham Smythe | 66.18 | 60.06 | 66.10 | 249 | Dolphins |
| 49 | 40 | Trey McBride | 66.15 | 53.58 | 70.37 | 329 | Cardinals |
| 50 | 41 | Robert Tonyan | 65.88 | 58.21 | 66.83 | 400 | Packers |
| 51 | 42 | Pharaoh Brown | 65.50 | 59.57 | 65.29 | 149 | Browns |
| 52 | 43 | C.J. Uzomah | 65.43 | 61.45 | 63.91 | 263 | Jets |
| 53 | 44 | Mitchell Wilcox | 65.25 | 57.39 | 66.33 | 244 | Bengals |
| 54 | 45 | Kylen Granson | 65.23 | 59.06 | 65.18 | 257 | Colts |
| 55 | 46 | Brock Wright | 65.14 | 56.41 | 66.79 | 253 | Lions |
| 56 | 47 | Daniel Bellinger | 64.96 | 60.90 | 63.50 | 318 | Giants |
| 57 | 48 | John Bates | 64.90 | 58.36 | 65.09 | 197 | Commanders |
| 58 | 49 | Tommy Tremble | 64.83 | 53.28 | 68.36 | 270 | Panthers |
| 59 | 50 | Albert Okwuegbunam | 64.80 | 57.79 | 65.31 | 153 | Broncos |
| 60 | 51 | Josiah Deguara | 64.61 | 58.88 | 64.26 | 101 | Packers |
| 61 | 52 | Tyler Conklin | 64.46 | 58.70 | 64.13 | 568 | Jets |
| 62 | 53 | Brevin Jordan | 64.45 | 56.87 | 65.33 | 163 | Texans |
| 63 | 54 | Eric Tomlinson | 64.30 | 59.11 | 63.59 | 161 | Broncos |
| 64 | 55 | Irv Smith Jr. | 64.19 | 58.08 | 64.09 | 210 | Vikings |
| 65 | 56 | Noah Gray | 64.17 | 62.62 | 61.04 | 330 | Chiefs |
| 66 | 57 | Geoff Swaim | 64.15 | 59.20 | 63.29 | 188 | Titans |
| 67 | 58 | Jack Stoll | 63.75 | 54.17 | 65.97 | 254 | Eagles |
| 68 | 59 | Peyton Hendershot | 63.64 | 59.17 | 62.45 | 127 | Cowboys |
| 69 | 60 | Quintin Morris | 62.75 | 52.05 | 65.71 | 159 | Bills |
| 70 | 61 | Ian Thomas | 62.65 | 53.78 | 64.40 | 236 | Panthers |

### Rotation/backup (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 71 | 1 | Johnny Mundt | 61.61 | 55.10 | 61.79 | 205 | Vikings |
| 72 | 2 | Logan Thomas | 61.32 | 53.66 | 62.26 | 415 | Commanders |
| 73 | 3 | Nick Vannett | 61.21 | 53.71 | 62.05 | 106 | Giants |
| 74 | 4 | Zach Gentry | 60.76 | 52.22 | 62.29 | 256 | Steelers |
| 75 | 5 | Shane Zylstra | 60.45 | 59.63 | 56.83 | 141 | Lions |
| 76 | 6 | Eric Saubert | 60.31 | 54.46 | 60.04 | 220 | Broncos |
| 77 | 7 | Tre' McKitty | 53.61 | 41.53 | 57.49 | 283 | Chargers |

## WR — Wide Receiver

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Davante Adams | 88.18 | 90.02 | 82.78 | 654 | Raiders |
| 2 | 2 | Justin Jefferson | 88.09 | 90.40 | 82.38 | 729 | Vikings |
| 3 | 3 | A.J. Brown | 87.42 | 87.35 | 83.30 | 610 | Eagles |
| 4 | 4 | Tyreek Hill | 87.12 | 90.50 | 80.70 | 555 | Dolphins |
| 5 | 5 | Stefon Diggs | 84.64 | 89.36 | 77.32 | 607 | Bills |
| 6 | 6 | Rashid Shaheed | 84.10 | 71.22 | 88.52 | 192 | Saints |
| 7 | 7 | Cooper Kupp | 83.85 | 81.96 | 80.95 | 347 | Rams |
| 8 | 8 | Amon-Ra St. Brown | 83.70 | 88.41 | 76.39 | 506 | Lions |
| 9 | 9 | CeeDee Lamb | 83.40 | 85.52 | 77.82 | 596 | Cowboys |
| 10 | 10 | Ja'Marr Chase | 83.17 | 82.12 | 79.71 | 553 | Bengals |
| 11 | 11 | Jaylen Waddle | 82.92 | 82.67 | 78.92 | 552 | Dolphins |
| 12 | 12 | Chris Olave | 82.51 | 78.58 | 80.97 | 450 | Saints |
| 13 | 13 | Garrett Wilson | 82.35 | 81.91 | 78.48 | 615 | Jets |
| 14 | 14 | Terry McLaurin | 82.19 | 79.53 | 79.80 | 620 | Commanders |
| 15 | 15 | DeVonta Smith | 81.05 | 80.78 | 77.07 | 637 | Eagles |
| 16 | 16 | Amari Cooper | 80.89 | 80.63 | 76.90 | 602 | Browns |
| 17 | 17 | Mike Williams | 80.79 | 77.24 | 78.99 | 481 | Chargers |
| 18 | 18 | Brandon Aiyuk | 80.15 | 79.40 | 76.48 | 566 | 49ers |
| 19 | 19 | Drake London | 80.10 | 78.92 | 76.72 | 439 | Falcons |

### Good (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Tyler Lockett | 79.84 | 78.01 | 76.90 | 560 | Seahawks |
| 21 | 2 | Tee Higgins | 79.63 | 76.33 | 77.66 | 565 | Bengals |
| 22 | 3 | Jerry Jeudy | 79.28 | 75.68 | 77.52 | 479 | Broncos |
| 23 | 4 | Keenan Allen | 79.25 | 81.03 | 73.89 | 368 | Chargers |
| 24 | 5 | Christian Watson | 78.69 | 71.16 | 79.54 | 281 | Packers |
| 25 | 6 | Deebo Samuel | 78.64 | 70.35 | 80.00 | 347 | 49ers |
| 26 | 7 | DJ Moore | 78.04 | 73.10 | 77.17 | 539 | Panthers |
| 27 | 8 | Michael Thomas | 77.77 | 71.14 | 78.03 | 109 | Saints |
| 28 | 9 | Mike Evans | 77.62 | 74.00 | 75.87 | 665 | Buccaneers |
| 29 | 10 | D.K. Metcalf | 77.52 | 74.97 | 75.05 | 613 | Seahawks |
| 30 | 11 | DeAndre Hopkins | 77.26 | 71.09 | 77.20 | 388 | Cardinals |
| 31 | 12 | Treylon Burks | 76.97 | 68.93 | 78.16 | 265 | Titans |
| 32 | 13 | Chris Godwin | 76.86 | 74.82 | 74.06 | 620 | Buccaneers |
| 33 | 14 | Jakobi Meyers | 76.77 | 73.96 | 74.48 | 449 | Patriots |
| 34 | 15 | Christian Kirk | 76.43 | 74.12 | 73.80 | 648 | Jaguars |
| 35 | 16 | DeVante Parker | 76.26 | 71.70 | 75.13 | 339 | Patriots |
| 36 | 17 | Tyler Boyd | 76.05 | 70.80 | 75.38 | 602 | Bengals |
| 37 | 18 | Tutu Atwell | 75.87 | 65.35 | 78.71 | 185 | Rams |
| 38 | 19 | Brandin Cooks | 75.61 | 71.01 | 74.51 | 449 | Texans |
| 39 | 20 | Kalif Raymond | 75.42 | 68.71 | 75.72 | 342 | Lions |
| 40 | 21 | Nico Collins | 75.18 | 68.43 | 75.52 | 305 | Texans |
| 41 | 22 | Randall Cobb | 74.84 | 67.83 | 75.34 | 258 | Packers |
| 42 | 23 | Trenton Irwin | 74.59 | 64.44 | 77.19 | 203 | Bengals |
| 43 | 24 | DJ Chark Jr. | 74.53 | 68.04 | 74.69 | 351 | Lions |
| 44 | 25 | JuJu Smith-Schuster | 74.42 | 70.28 | 73.01 | 559 | Chiefs |
| 45 | 26 | Jahan Dotson | 74.38 | 68.20 | 74.33 | 403 | Commanders |
| 46 | 27 | Julio Jones | 74.29 | 63.98 | 76.99 | 264 | Buccaneers |
| 47 | 28 | Darnell Mooney | 74.26 | 67.60 | 74.54 | 332 | Bears |
| 48 | 29 | Corey Davis | 74.20 | 65.17 | 76.05 | 416 | Jets |
| 49 | 30 | Isaiah Hodgins | 74.03 | 70.75 | 72.05 | 277 | Giants |

### Starter (91 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 50 | 1 | Allen Lazard | 73.92 | 67.97 | 73.72 | 517 | Packers |
| 51 | 2 | George Pickens | 73.78 | 68.52 | 73.12 | 619 | Steelers |
| 52 | 3 | Ashton Dulin | 73.51 | 65.17 | 74.90 | 131 | Colts |
| 53 | 4 | Donovan Peoples-Jones | 73.49 | 64.73 | 75.17 | 615 | Browns |
| 54 | 5 | Michael Pittman Jr. | 73.38 | 70.60 | 71.06 | 674 | Colts |
| 55 | 6 | Courtland Sutton | 73.37 | 68.93 | 72.17 | 576 | Broncos |
| 56 | 7 | Darius Slayton | 73.24 | 66.86 | 73.33 | 430 | Giants |
| 57 | 8 | Richie James | 73.20 | 68.65 | 72.07 | 367 | Giants |
| 58 | 9 | Skyy Moore | 73.19 | 65.69 | 74.02 | 180 | Chiefs |
| 59 | 10 | Terrace Marshall Jr. | 73.11 | 65.56 | 73.97 | 344 | Panthers |
| 60 | 11 | Kendrick Bourne | 73.10 | 62.92 | 75.72 | 310 | Patriots |
| 61 | 12 | Tom Kennedy | 73.09 | 59.68 | 77.86 | 138 | Lions |
| 62 | 13 | Jarvis Landry | 73.06 | 65.45 | 73.97 | 221 | Saints |
| 63 | 14 | Marquise Brown | 72.80 | 68.17 | 71.72 | 521 | Cardinals |
| 64 | 15 | Gabe Davis | 72.79 | 64.06 | 74.45 | 617 | Bills |
| 65 | 16 | Mecole Hardman Jr. | 72.79 | 64.51 | 74.15 | 215 | Chiefs |
| 66 | 17 | Marquez Valdes-Scantling | 72.74 | 62.46 | 75.42 | 589 | Chiefs |
| 67 | 18 | Curtis Samuel | 72.71 | 69.83 | 70.46 | 528 | Commanders |
| 68 | 19 | Equanimeous St. Brown | 72.55 | 62.86 | 74.84 | 306 | Bears |
| 69 | 20 | Marquise Goodwin | 72.50 | 63.67 | 74.22 | 316 | Seahawks |
| 70 | 21 | Diontae Johnson | 72.43 | 69.60 | 70.15 | 659 | Steelers |
| 71 | 22 | Khalil Shakir | 72.37 | 61.15 | 75.69 | 151 | Bills |
| 72 | 23 | Van Jefferson | 72.31 | 65.76 | 72.51 | 310 | Rams |
| 73 | 24 | Olamide Zaccheaus | 72.18 | 63.51 | 73.79 | 385 | Falcons |
| 74 | 25 | Hunter Renfrow | 72.12 | 64.53 | 73.02 | 307 | Raiders |
| 75 | 26 | K.J. Osborn | 71.99 | 65.00 | 72.48 | 634 | Vikings |
| 76 | 27 | Sammy Watkins | 71.94 | 62.08 | 74.34 | 216 | Ravens |
| 77 | 28 | Tre'Quan Smith | 71.51 | 64.83 | 71.80 | 190 | Saints |
| 78 | 29 | Robert Woods | 71.28 | 67.32 | 69.75 | 487 | Titans |
| 79 | 30 | Damiere Byrd | 71.27 | 63.39 | 72.36 | 174 | Falcons |
| 80 | 31 | Cedrick Wilson Jr. | 71.24 | 61.62 | 73.49 | 142 | Dolphins |
| 81 | 32 | Trent Sherfield | 71.21 | 62.40 | 72.91 | 422 | Dolphins |
| 82 | 33 | Wan'Dale Robinson | 71.08 | 65.53 | 70.62 | 140 | Giants |
| 83 | 34 | Allen Robinson II | 70.99 | 64.94 | 70.86 | 375 | Rams |
| 84 | 35 | Josh Reynolds | 70.94 | 63.21 | 71.93 | 405 | Lions |
| 85 | 36 | Chase Claypool | 70.87 | 61.62 | 72.87 | 454 | Bears |
| 86 | 37 | Denzel Mims | 70.76 | 59.90 | 73.83 | 167 | Jets |
| 87 | 38 | Greg Dortch | 70.65 | 63.11 | 71.51 | 361 | Cardinals |
| 88 | 39 | Adam Thielen | 70.54 | 65.00 | 70.07 | 717 | Vikings |
| 89 | 40 | Chris Moore | 70.45 | 62.68 | 71.46 | 464 | Texans |
| 90 | 41 | Zay Jones | 70.39 | 67.00 | 68.49 | 607 | Jaguars |
| 91 | 42 | Russell Gage | 70.29 | 66.63 | 68.57 | 390 | Buccaneers |
| 92 | 43 | Alec Pierce | 69.97 | 61.13 | 71.69 | 501 | Colts |
| 93 | 44 | Joshua Palmer | 69.95 | 65.42 | 68.80 | 642 | Chargers |
| 94 | 45 | Byron Pringle | 69.81 | 61.43 | 71.23 | 161 | Bears |
| 95 | 46 | Nick Westbrook-Ikhine | 69.80 | 59.84 | 72.28 | 442 | Titans |
| 96 | 47 | Jauan Jennings | 69.79 | 64.41 | 69.21 | 323 | 49ers |
| 97 | 48 | Rashod Bateman | 69.74 | 60.70 | 71.60 | 127 | Ravens |
| 98 | 49 | Rondale Moore | 69.60 | 62.27 | 70.32 | 294 | Cardinals |
| 99 | 50 | Amari Rodgers | 69.56 | 61.07 | 71.06 | 172 | Texans |
| 100 | 51 | Mack Hollins | 69.54 | 63.76 | 69.23 | 646 | Raiders |
| 101 | 52 | Devin Duvernay | 69.41 | 65.42 | 67.91 | 374 | Ravens |
| 102 | 53 | Quez Watkins | 69.41 | 56.06 | 74.14 | 426 | Eagles |
| 103 | 54 | Marquez Callaway | 69.31 | 59.84 | 71.45 | 197 | Saints |
| 104 | 55 | Michael Gallup | 69.08 | 62.50 | 69.30 | 444 | Cowboys |
| 105 | 56 | Nelson Agholor | 68.98 | 57.07 | 72.75 | 312 | Patriots |
| 106 | 57 | DeAndre Carter | 68.94 | 61.41 | 69.79 | 513 | Chargers |
| 107 | 58 | Breshad Perriman | 68.79 | 54.35 | 74.25 | 145 | Buccaneers |
| 108 | 59 | Brandon Powell | 68.71 | 62.66 | 68.58 | 139 | Rams |
| 109 | 60 | KJ Hamler | 68.70 | 58.75 | 71.16 | 164 | Broncos |
| 110 | 61 | Elijah Moore | 68.67 | 57.79 | 71.76 | 514 | Jets |
| 111 | 62 | Zach Pascal | 68.63 | 61.82 | 69.01 | 164 | Eagles |
| 112 | 63 | Kenny Golladay | 68.60 | 56.51 | 72.50 | 162 | Giants |
| 113 | 64 | Braxton Berrios | 68.56 | 61.49 | 69.10 | 186 | Jets |
| 114 | 65 | Marvin Jones Jr. | 67.88 | 60.37 | 68.72 | 496 | Jaguars |
| 115 | 66 | Noah Brown | 67.84 | 59.57 | 69.19 | 488 | Cowboys |
| 116 | 67 | Kendall Hinton | 67.82 | 58.97 | 69.56 | 314 | Broncos |
| 117 | 68 | Sterling Shepard | 67.82 | 59.04 | 69.51 | 108 | Giants |
| 118 | 69 | Phillip Dorsett | 67.81 | 57.07 | 70.80 | 306 | Texans |
| 119 | 70 | Cam Sims | 67.78 | 54.94 | 72.18 | 183 | Commanders |
| 120 | 71 | Romeo Doubs | 67.63 | 61.84 | 67.33 | 331 | Packers |
| 121 | 72 | Isaiah McKenzie | 67.60 | 64.53 | 65.48 | 403 | Bills |
| 122 | 73 | Parris Campbell | 67.56 | 60.78 | 67.91 | 635 | Colts |
| 123 | 74 | Ray-Ray McCloud III | 67.29 | 64.90 | 64.71 | 179 | 49ers |
| 124 | 75 | Justin Watson | 67.22 | 54.96 | 71.23 | 315 | Chiefs |
| 125 | 76 | A.J. Green | 67.21 | 56.23 | 70.36 | 384 | Cardinals |
| 126 | 77 | Demarcus Robinson | 67.08 | 63.68 | 65.18 | 373 | Ravens |
| 127 | 78 | Shi Smith | 67.08 | 54.35 | 71.40 | 338 | Panthers |
| 128 | 79 | Scott Miller | 66.53 | 57.74 | 68.22 | 200 | Buccaneers |
| 129 | 80 | Marcus Johnson | 66.41 | 56.59 | 68.79 | 200 | Giants |
| 130 | 81 | Ben Skowronek | 65.86 | 57.76 | 67.09 | 424 | Rams |
| 131 | 82 | D'Wayne Eskridge | 65.47 | 57.64 | 66.52 | 101 | Seahawks |
| 132 | 83 | Michael Woods II | 65.17 | 57.40 | 66.18 | 103 | Browns |
| 133 | 84 | Dante Pettis | 64.69 | 55.28 | 66.79 | 327 | Bears |
| 134 | 85 | Keelan Cole Sr. | 64.40 | 50.60 | 69.44 | 260 | Raiders |
| 135 | 86 | Steven Sims | 64.09 | 57.55 | 64.28 | 158 | Steelers |
| 136 | 87 | Tyquan Thornton | 64.01 | 56.51 | 64.84 | 348 | Patriots |
| 137 | 88 | David Sills V | 63.66 | 58.11 | 63.19 | 140 | Giants |
| 138 | 89 | Brandon Johnson | 62.86 | 56.25 | 63.10 | 138 | Broncos |
| 139 | 90 | James Proche II | 62.77 | 52.90 | 65.18 | 142 | Ravens |
| 140 | 91 | Michael Bandy | 62.15 | 56.99 | 61.43 | 151 | Chargers |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 141 | 1 | David Bell | 61.79 | 54.72 | 62.33 | 327 | Browns |
