# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:26:38Z
- **Requested analysis_year:** 2022 (clamped to 2022)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Creed Humphrey | 96.02 | 90.00 | 95.87 | 1138 | Chiefs |
| 2 | 2 | Jason Kelce | 95.05 | 88.30 | 95.38 | 1149 | Eagles |
| 3 | 3 | Connor Williams | 88.27 | 78.20 | 90.81 | 1056 | Dolphins |
| 4 | 4 | Ethan Pocic | 86.61 | 78.90 | 87.59 | 819 | Browns |
| 5 | 5 | Frank Ragnow | 86.16 | 77.90 | 87.50 | 1074 | Lions |
| 6 | 6 | Tyler Linderbaum | 84.75 | 74.70 | 87.28 | 1092 | Ravens |
| 7 | 7 | David Andrews | 83.56 | 74.50 | 85.43 | 799 | Patriots |
| 8 | 8 | Corey Linsley | 82.90 | 74.20 | 84.53 | 858 | Chargers |
| 9 | 9 | Ben Jones | 81.88 | 71.90 | 84.37 | 682 | Titans |
| 10 | 10 | Garrett Bradbury | 80.21 | 70.20 | 82.71 | 809 | Vikings |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Connor McGovern | 79.51 | 69.60 | 81.95 | 1111 | Jets |
| 12 | 2 | Mason Cole | 77.24 | 67.10 | 79.84 | 1114 | Steelers |
| 13 | 3 | Jake Brendel | 77.01 | 64.90 | 80.92 | 1078 | 49ers |
| 14 | 4 | Corey Levin | 76.01 | 68.30 | 76.98 | 251 | Titans |
| 15 | 5 | Ryan Kelly | 75.32 | 64.30 | 78.50 | 1092 | Colts |
| 16 | 6 | Brian Allen | 74.99 | 63.80 | 78.28 | 373 | Rams |
| 17 | 7 | Bradley Bozeman | 74.88 | 63.10 | 78.56 | 680 | Panthers |
| 18 | 8 | Sam Mustipher | 74.64 | 63.40 | 77.97 | 1020 | Bears |

### Starter (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Erik McCoy | 73.59 | 61.20 | 77.69 | 798 | Saints |
| 20 | 2 | Tyler Biadasz | 73.55 | 61.70 | 77.28 | 1066 | Cowboys |
| 21 | 3 | Drew Dalman | 73.30 | 65.90 | 74.07 | 1051 | Falcons |
| 22 | 4 | Robert Hainsey | 73.27 | 66.70 | 73.49 | 1175 | Buccaneers |
| 23 | 5 | Ted Karras | 73.25 | 62.60 | 76.19 | 1100 | Bengals |
| 24 | 6 | Andre James | 73.20 | 62.80 | 75.96 | 964 | Raiders |
| 25 | 7 | Mitch Morse | 72.33 | 61.40 | 75.45 | 765 | Bills |
| 26 | 8 | Pat Elflein | 72.05 | 60.90 | 75.31 | 338 | Panthers |
| 27 | 9 | Josh Myers | 71.87 | 60.40 | 75.35 | 1091 | Packers |
| 28 | 10 | Jon Feliciano | 70.75 | 58.20 | 74.95 | 971 | Giants |
| 29 | 11 | Rodney Hudson | 70.56 | 58.30 | 74.56 | 303 | Cardinals |
| 30 | 12 | Tyler Larsen | 70.36 | 58.20 | 74.30 | 534 | Commanders |
| 31 | 13 | Lloyd Cushenberry III | 69.26 | 56.20 | 73.80 | 502 | Broncos |
| 32 | 14 | Chase Roullier | 68.78 | 56.30 | 72.94 | 150 | Commanders |
| 33 | 15 | Will Clapp | 67.57 | 54.30 | 72.25 | 333 | Chargers |
| 34 | 16 | James Ferentz | 67.25 | 53.40 | 72.31 | 268 | Patriots |
| 35 | 17 | Austin Blythe | 66.12 | 51.90 | 71.44 | 1041 | Seahawks |
| 36 | 18 | Josh Andrews | 65.34 | 51.60 | 70.34 | 330 | Saints |
| 37 | 19 | Luke Fortner | 64.29 | 49.60 | 69.91 | 1121 | Jaguars |
| 38 | 20 | Nick Martin | 62.72 | 42.60 | 71.97 | 156 | Commanders |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Scott Quessenberry | 57.47 | 36.60 | 67.22 | 990 | Texans |

## CB — Cornerback

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Sauce Gardner | 94.37 | 90.00 | 93.11 | 1114 | Jets |
| 2 | 2 | Pat Surtain II | 89.48 | 86.70 | 87.53 | 1104 | Broncos |
| 3 | 3 | Tyler Hall | 86.77 | 86.30 | 90.54 | 218 | Raiders |
| 4 | 4 | Patrick Peterson | 85.18 | 82.50 | 83.99 | 1104 | Vikings |
| 5 | 5 | Tyson Campbell | 84.71 | 81.20 | 83.61 | 1138 | Jaguars |
| 6 | 6 | Tariq Woolen | 84.35 | 77.80 | 84.55 | 1135 | Seahawks |
| 7 | 7 | Jaire Alexander | 83.96 | 82.10 | 85.55 | 901 | Packers |
| 8 | 8 | James Bradberry | 83.85 | 80.20 | 82.31 | 1077 | Eagles |
| 9 | 9 | Charvarius Ward | 83.57 | 78.30 | 84.52 | 959 | 49ers |
| 10 | 10 | Jalen Ramsey | 83.44 | 77.80 | 83.54 | 1078 | Rams |
| 11 | 11 | Stephon Gilmore | 83.35 | 81.10 | 84.86 | 1064 | Colts |
| 12 | 12 | Duke Shelley | 82.65 | 84.90 | 85.13 | 398 | Vikings |
| 13 | 13 | Darius Slay | 82.25 | 77.40 | 81.81 | 1002 | Eagles |
| 14 | 14 | Jamel Dean | 81.78 | 75.60 | 83.72 | 884 | Buccaneers |
| 15 | 15 | Kendall Fuller | 81.28 | 75.40 | 81.75 | 1030 | Commanders |
| 16 | 16 | Isaiah Rodgers | 81.23 | 81.50 | 81.13 | 434 | Colts |
| 17 | 17 | Marlon Humphrey | 80.55 | 75.60 | 81.37 | 1051 | Ravens |
| 18 | 18 | D.J. Reed | 80.37 | 77.50 | 80.45 | 1135 | Jets |

### Good (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Michael Davis | 79.75 | 73.50 | 82.10 | 790 | Chargers |
| 20 | 2 | Martin Emerson Jr. | 79.33 | 75.10 | 77.99 | 783 | Browns |
| 21 | 3 | Danny Johnson | 79.13 | 80.10 | 83.62 | 292 | Commanders |
| 22 | 4 | Jack Jones | 78.93 | 76.00 | 80.63 | 434 | Patriots |
| 23 | 5 | Taron Johnson | 78.12 | 74.70 | 77.02 | 969 | Bills |
| 24 | 6 | Sean Murphy-Bunting | 78.03 | 80.90 | 78.21 | 430 | Buccaneers |
| 25 | 7 | Rasul Douglas | 77.73 | 71.50 | 79.60 | 915 | Packers |
| 26 | 8 | DaRon Bland | 77.34 | 74.90 | 78.71 | 597 | Cowboys |
| 27 | 9 | Trevon Diggs | 77.27 | 66.10 | 81.69 | 1115 | Cowboys |
| 28 | 10 | L'Jarius Sneed | 77.20 | 74.00 | 77.21 | 1106 | Chiefs |
| 29 | 11 | Mike Hilton | 76.63 | 72.30 | 77.65 | 701 | Bengals |
| 30 | 12 | Greg Newsome II | 76.60 | 72.30 | 78.37 | 907 | Browns |
| 31 | 13 | Cameron Sutton | 76.07 | 70.40 | 76.46 | 931 | Steelers |
| 32 | 14 | Tavierre Thomas | 75.93 | 75.00 | 78.06 | 409 | Texans |
| 33 | 15 | Desmond King II | 75.86 | 71.80 | 74.90 | 916 | Texans |
| 34 | 16 | Steven Nelson | 75.76 | 72.00 | 75.58 | 957 | Texans |
| 35 | 17 | Michael Carter II | 75.71 | 71.20 | 75.29 | 732 | Jets |
| 36 | 18 | Marcus Peters | 75.68 | 69.50 | 78.02 | 734 | Ravens |
| 37 | 19 | Trent McDuffie | 75.67 | 76.00 | 77.16 | 683 | Chiefs |
| 38 | 20 | Marshon Lattimore | 75.56 | 70.10 | 80.65 | 415 | Saints |
| 39 | 21 | Chidobe Awuzie | 75.25 | 73.10 | 79.49 | 471 | Bengals |
| 40 | 22 | Samuel Womack III | 75.12 | 72.90 | 80.28 | 146 | 49ers |
| 41 | 23 | Cobie Durant | 74.78 | 74.30 | 82.24 | 281 | Rams |
| 42 | 24 | Emmanuel Moseley | 74.47 | 73.50 | 79.44 | 312 | 49ers |
| 43 | 25 | Isaiah Oliver | 74.47 | 72.20 | 78.08 | 349 | Falcons |
| 44 | 26 | Jaycee Horn | 74.46 | 73.20 | 78.73 | 812 | Panthers |
| 45 | 27 | Kader Kohou | 74.08 | 70.40 | 74.34 | 895 | Dolphins |

### Starter (62 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 46 | 1 | Adoree' Jackson | 73.62 | 71.50 | 78.18 | 554 | Giants |
| 47 | 2 | Asante Samuel Jr. | 73.46 | 65.40 | 76.50 | 971 | Chargers |
| 48 | 3 | Jonathan Jones | 73.42 | 67.90 | 76.67 | 914 | Patriots |
| 49 | 4 | Marcus Jones | 73.11 | 66.80 | 79.03 | 371 | Patriots |
| 50 | 5 | Darious Williams | 72.73 | 63.60 | 75.54 | 944 | Jaguars |
| 51 | 6 | Nick McCloud | 72.14 | 64.60 | 76.06 | 537 | Giants |
| 52 | 7 | Bryce Callahan | 71.93 | 67.40 | 74.79 | 585 | Chargers |
| 53 | 8 | Avonte Maddox | 71.87 | 70.30 | 74.21 | 457 | Eagles |
| 54 | 9 | Antonio Hamilton Sr. | 71.64 | 69.40 | 75.84 | 420 | Cardinals |
| 55 | 10 | James Pierre | 71.54 | 67.70 | 76.64 | 260 | Steelers |
| 56 | 11 | Cor'Dale Flott | 71.44 | 70.70 | 74.63 | 335 | Giants |
| 57 | 12 | Jaylon Johnson | 71.32 | 65.20 | 75.39 | 656 | Bears |
| 58 | 13 | Nik Needham | 71.31 | 66.50 | 76.17 | 294 | Dolphins |
| 59 | 14 | Carlton Davis III | 71.17 | 64.00 | 76.21 | 809 | Buccaneers |
| 60 | 15 | Casey Hayward Jr. | 71.09 | 65.60 | 76.40 | 355 | Falcons |
| 61 | 16 | Terrance Mitchell | 70.86 | 62.10 | 76.35 | 397 | Titans |
| 62 | 17 | Ronald Darby | 69.92 | 66.30 | 75.81 | 280 | Broncos |
| 63 | 18 | A.J. Terrell | 69.91 | 61.80 | 73.33 | 800 | Falcons |
| 64 | 19 | Rock Ya-Sin | 69.90 | 65.80 | 73.21 | 663 | Raiders |
| 65 | 20 | K'Waun Williams | 69.84 | 66.50 | 71.91 | 596 | Broncos |
| 66 | 21 | Damarri Mathis | 69.74 | 65.30 | 71.47 | 794 | Broncos |
| 67 | 22 | Levi Wallace | 69.55 | 59.30 | 74.04 | 709 | Steelers |
| 68 | 23 | Cameron Dantzler | 69.52 | 64.70 | 74.22 | 505 | Vikings |
| 69 | 24 | Marco Wilson | 69.02 | 61.20 | 73.00 | 778 | Cardinals |
| 70 | 25 | Roger McCreary | 69.01 | 60.40 | 70.58 | 1164 | Titans |
| 71 | 26 | Byron Murphy Jr. | 68.93 | 63.90 | 72.53 | 595 | Cardinals |
| 72 | 27 | Denzel Ward | 68.90 | 60.40 | 73.30 | 748 | Browns |
| 73 | 28 | Dane Jackson | 68.80 | 59.90 | 73.26 | 830 | Bills |
| 74 | 29 | Tre'Davious White | 68.12 | 61.30 | 76.07 | 307 | Bills |
| 75 | 30 | Troy Hill | 67.60 | 60.50 | 72.09 | 703 | Rams |
| 76 | 31 | Shaquill Griffin | 67.37 | 61.90 | 74.45 | 336 | Jaguars |
| 77 | 32 | Ja'Sir Taylor | 67.35 | 65.60 | 73.16 | 161 | Chargers |
| 78 | 33 | Xavien Howard | 67.33 | 55.20 | 72.52 | 973 | Dolphins |
| 79 | 34 | Tremon Smith | 67.30 | 66.00 | 76.50 | 201 | Texans |
| 80 | 35 | Tre Flowers | 66.58 | 58.30 | 71.14 | 172 | Bengals |
| 81 | 36 | Dee Delaney | 66.21 | 67.60 | 70.31 | 216 | Buccaneers |
| 82 | 37 | Cam Taylor-Britt | 66.09 | 60.50 | 72.52 | 590 | Bengals |
| 83 | 38 | Joshua Williams | 65.92 | 63.50 | 67.28 | 437 | Chiefs |
| 84 | 39 | Myles Bryant | 65.89 | 55.90 | 69.85 | 689 | Patriots |
| 85 | 40 | Eric Stokes | 65.29 | 60.30 | 69.71 | 477 | Packers |
| 86 | 41 | Kristian Fulton | 65.00 | 58.50 | 71.36 | 652 | Titans |
| 87 | 42 | Mike Jackson | 64.71 | 58.60 | 72.15 | 1082 | Seahawks |
| 88 | 43 | Kaiir Elam | 64.33 | 58.70 | 68.81 | 477 | Bills |
| 89 | 44 | Benjamin St-Juste | 64.20 | 60.00 | 69.57 | 655 | Commanders |
| 90 | 45 | Brandon Facyson | 64.01 | 56.00 | 69.74 | 455 | Colts |
| 91 | 46 | Christian Benford | 63.99 | 58.80 | 71.13 | 363 | Bills |
| 92 | 47 | Fabian Moreau | 63.79 | 53.60 | 69.02 | 749 | Giants |
| 93 | 48 | Coby Bryant | 63.74 | 54.60 | 65.67 | 757 | Seahawks |
| 94 | 49 | Rodarius Williams | 63.66 | 63.40 | 72.16 | 147 | Giants |
| 95 | 50 | Alontae Taylor | 63.66 | 56.30 | 70.29 | 663 | Saints |
| 96 | 51 | Anthony Brown | 63.31 | 54.60 | 68.95 | 728 | Cowboys |
| 97 | 52 | Chandon Sullivan | 63.22 | 53.20 | 65.74 | 944 | Vikings |
| 98 | 53 | Eli Apple | 63.21 | 53.50 | 67.00 | 908 | Bengals |
| 99 | 54 | Jaylen Watson | 63.18 | 55.50 | 65.11 | 604 | Chiefs |
| 100 | 55 | Rashad Fenton | 63.17 | 56.20 | 70.64 | 379 | Falcons |
| 101 | 56 | Donte Jackson | 63.03 | 55.20 | 69.90 | 442 | Panthers |
| 102 | 57 | Grant Haley | 62.99 | 68.40 | 70.83 | 129 | Rams |
| 103 | 58 | Jerry Jacobs | 62.85 | 54.50 | 70.14 | 542 | Lions |
| 104 | 59 | Jourdan Lewis | 62.79 | 55.90 | 69.11 | 315 | Cowboys |
| 105 | 60 | Keisean Nixon | 62.73 | 60.40 | 67.36 | 290 | Packers |
| 106 | 61 | Arthur Maulet | 62.64 | 55.10 | 64.08 | 481 | Steelers |
| 107 | 62 | Nate Hobbs | 62.21 | 57.10 | 65.86 | 667 | Raiders |

### Rotation/backup (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 108 | 1 | Amik Robertson | 61.58 | 61.30 | 62.91 | 677 | Raiders |
| 109 | 2 | Josiah Scott | 61.50 | 62.30 | 65.45 | 390 | Eagles |
| 110 | 3 | Cornell Armstrong | 61.33 | 54.00 | 68.45 | 372 | Falcons |
| 111 | 4 | Harrison Hand | 61.27 | 55.00 | 73.79 | 111 | Bears |
| 112 | 5 | Mike Hughes | 61.16 | 51.40 | 67.97 | 561 | Lions |
| 113 | 6 | Darren Hall | 60.93 | 60.10 | 60.14 | 633 | Falcons |
| 114 | 7 | Sam Webb | 60.46 | 57.00 | 66.45 | 327 | Raiders |
| 115 | 8 | David Long Jr. | 60.33 | 54.20 | 65.18 | 287 | Rams |
| 116 | 9 | Bradley Roby | 60.19 | 49.60 | 67.19 | 628 | Saints |
| 117 | 10 | Darnay Holmes | 60.09 | 49.50 | 66.57 | 572 | Giants |
| 118 | 11 | Paulson Adebo | 59.78 | 48.60 | 65.52 | 814 | Saints |
| 119 | 12 | Kenny Moore II | 59.70 | 46.30 | 66.91 | 774 | Colts |
| 120 | 13 | Greg Mabin | 59.42 | 57.00 | 68.56 | 108 | Titans |
| 121 | 14 | Tre Herndon | 59.38 | 57.20 | 62.85 | 416 | Jaguars |
| 122 | 15 | Kindle Vildor | 59.36 | 58.80 | 61.10 | 531 | Bears |
| 123 | 16 | Sidney Jones IV | 59.33 | 44.60 | 71.77 | 109 | Raiders |
| 124 | 17 | CJ Henderson | 59.26 | 48.90 | 65.13 | 765 | Panthers |
| 125 | 18 | Josh Blackwell | 59.22 | 66.40 | 66.53 | 134 | Bears |
| 126 | 19 | Zech McPhearson | 59.16 | 50.00 | 69.68 | 103 | Eagles |
| 127 | 20 | Justin Bethel | 59.06 | 52.90 | 69.67 | 125 | Dolphins |
| 128 | 21 | Greedy Williams | 58.98 | 49.60 | 66.55 | 105 | Browns |
| 129 | 22 | Derek Stingley Jr. | 58.90 | 49.90 | 68.59 | 599 | Texans |
| 130 | 23 | Keion Crossen | 58.67 | 50.20 | 66.33 | 382 | Dolphins |
| 131 | 24 | Deommodore Lenoir | 58.62 | 51.00 | 63.82 | 887 | 49ers |
| 132 | 25 | Anthony Averett | 58.47 | 51.00 | 66.31 | 278 | Raiders |
| 133 | 26 | Chris Harris Jr. | 58.19 | 47.80 | 66.72 | 375 | Saints |
| 134 | 27 | Jalen Mills | 57.70 | 42.70 | 68.71 | 468 | Patriots |
| 135 | 28 | Rachad Wildgoose | 57.67 | 53.90 | 67.67 | 196 | Commanders |
| 136 | 29 | Trayvon Mullen | 57.66 | 48.90 | 67.77 | 160 | Cowboys |
| 137 | 30 | Kyler Gordon | 57.36 | 46.40 | 63.44 | 863 | Bears |
| 138 | 31 | Kelvin Joseph | 56.87 | 44.10 | 68.94 | 167 | Cowboys |
| 139 | 32 | A.J. Green III | 56.78 | 41.80 | 70.08 | 142 | Browns |
| 140 | 33 | Essang Bassey | 56.78 | 51.70 | 62.63 | 222 | Broncos |
| 141 | 34 | Derion Kendrick | 56.54 | 44.80 | 65.10 | 483 | Rams |
| 142 | 35 | Jeff Okudah | 56.43 | 54.40 | 60.76 | 789 | Lions |
| 143 | 36 | Keith Taylor Jr. | 56.32 | 52.10 | 56.56 | 378 | Panthers |
| 144 | 37 | Jace Whittaker | 54.60 | 51.50 | 66.50 | 281 | Cardinals |
| 145 | 38 | Christian Matthew | 54.24 | 48.70 | 61.62 | 237 | Cardinals |
| 146 | 39 | Nahshon Wright | 53.89 | 47.40 | 65.82 | 128 | Cowboys |
| 147 | 40 | Noah Igbinoghene | 53.50 | 47.40 | 63.66 | 238 | Dolphins |
| 148 | 41 | Ahkello Witherspoon | 53.42 | 42.80 | 66.51 | 248 | Steelers |
| 149 | 42 | Amani Oruwariye | 52.18 | 31.30 | 64.78 | 474 | Lions |
| 150 | 43 | Zyon McCollum | 50.90 | 51.40 | 54.25 | 278 | Buccaneers |
| 151 | 44 | J.C. Jackson | 50.87 | 28.10 | 67.77 | 244 | Chargers |
| 152 | 45 | Damarion Williams | 47.48 | 41.00 | 52.53 | 225 | Ravens |
| 153 | 46 | Akayleb Evans | 46.56 | 40.50 | 57.73 | 162 | Vikings |
| 154 | 47 | Christian Holmes | 45.00 | 48.90 | 50.00 | 104 | Commanders |
| 155 | 48 | Caleb Farley | 45.00 | 34.70 | 52.29 | 104 | Titans |
| 156 | 49 | Andrew Booth Jr. | 45.00 | 41.70 | 57.94 | 105 | Vikings |
| 157 | 50 | Dallis Flowers | 45.00 | 51.30 | 50.68 | 175 | Colts |

## DI — Defensive Interior

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 90.25 | 86.47 | 91.53 | 615 | Rams |
| 2 | 2 | Quinnen Williams | 89.33 | 89.82 | 86.53 | 690 | Jets |
| 3 | 3 | Jonathan Allen | 86.49 | 86.83 | 82.58 | 802 | Commanders |
| 4 | 4 | DeForest Buckner | 86.12 | 87.68 | 81.11 | 874 | Colts |
| 5 | 5 | Christian Wilkins | 86.09 | 88.45 | 80.76 | 952 | Dolphins |
| 6 | 6 | J.J. Watt | 85.68 | 72.83 | 95.79 | 816 | Cardinals |
| 7 | 7 | Cameron Heyward | 84.90 | 82.83 | 82.31 | 801 | Steelers |
| 8 | 8 | Dexter Lawrence | 84.39 | 88.24 | 78.44 | 864 | Giants |
| 9 | 9 | Chris Jones | 84.15 | 88.78 | 77.98 | 915 | Chiefs |
| 10 | 10 | Zach Sieler | 83.92 | 78.32 | 83.48 | 874 | Dolphins |
| 11 | 11 | Leonard Williams | 83.05 | 86.79 | 78.84 | 604 | Giants |
| 12 | 12 | Jeffery Simmons | 82.96 | 85.51 | 78.27 | 840 | Titans |
| 13 | 13 | Grady Jarrett | 81.90 | 78.13 | 80.24 | 856 | Falcons |
| 14 | 14 | Derrick Brown | 80.74 | 85.64 | 73.60 | 870 | Panthers |
| 15 | 15 | Ed Oliver | 80.49 | 74.24 | 82.46 | 526 | Bills |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Javon Hargrave | 79.87 | 73.19 | 80.66 | 711 | Eagles |
| 17 | 2 | Calais Campbell | 78.72 | 68.19 | 84.47 | 548 | Ravens |
| 18 | 3 | Dalvin Tomlinson | 78.66 | 79.74 | 76.02 | 550 | Vikings |
| 19 | 4 | Daron Payne | 78.52 | 72.77 | 78.19 | 907 | Commanders |
| 20 | 5 | Kenny Clark | 78.47 | 74.17 | 78.08 | 807 | Packers |
| 21 | 6 | DJ Reader | 77.29 | 84.17 | 74.85 | 397 | Bengals |
| 22 | 7 | DeMarcus Walker | 77.23 | 61.79 | 85.54 | 427 | Titans |
| 23 | 8 | B.J. Hill | 76.05 | 70.44 | 76.40 | 815 | Bengals |
| 24 | 9 | Milton Williams | 75.69 | 62.03 | 80.63 | 396 | Eagles |
| 25 | 10 | Shelby Harris | 75.21 | 64.94 | 80.20 | 560 | Seahawks |
| 26 | 11 | Vita Vea | 74.81 | 72.95 | 75.94 | 538 | Buccaneers |

### Starter (70 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | DaVon Hamilton | 73.43 | 67.91 | 74.27 | 610 | Jaguars |
| 28 | 2 | Andrew Billings | 73.41 | 71.37 | 72.07 | 478 | Raiders |
| 29 | 3 | Grover Stewart | 73.33 | 66.25 | 73.88 | 781 | Colts |
| 30 | 4 | Christian Barmore | 73.30 | 65.14 | 78.86 | 327 | Patriots |
| 31 | 5 | David Onyemata | 72.73 | 65.69 | 75.23 | 682 | Saints |
| 32 | 6 | D.J. Jones | 72.49 | 60.07 | 78.01 | 558 | Broncos |
| 33 | 7 | Poona Ford | 72.33 | 62.52 | 74.70 | 642 | Seahawks |
| 34 | 8 | Jordan Davis | 72.18 | 75.18 | 69.93 | 225 | Eagles |
| 35 | 9 | Harrison Phillips | 71.82 | 68.33 | 71.70 | 693 | Vikings |
| 36 | 10 | Akiem Hicks | 71.39 | 59.00 | 80.99 | 398 | Buccaneers |
| 37 | 11 | Tim Settle | 71.36 | 57.77 | 77.52 | 372 | Bills |
| 38 | 12 | Larry Ogunjobi | 71.29 | 54.55 | 79.29 | 636 | Steelers |
| 39 | 13 | Alim McNeill | 70.99 | 65.00 | 70.82 | 779 | Lions |
| 40 | 14 | Osa Odighizuwa | 70.87 | 55.35 | 77.41 | 616 | Cowboys |
| 41 | 15 | Matt Ioannidis | 70.70 | 63.87 | 76.05 | 640 | Panthers |
| 42 | 16 | Morgan Fox | 70.67 | 55.73 | 76.46 | 575 | Chargers |
| 43 | 17 | Khyiris Tonga | 70.62 | 67.97 | 72.64 | 276 | Vikings |
| 44 | 18 | Fletcher Cox | 70.55 | 54.80 | 77.39 | 712 | Eagles |
| 45 | 19 | Dre'Mont Jones | 70.32 | 57.87 | 77.33 | 715 | Broncos |
| 46 | 20 | Chauncey Golston | 69.87 | 56.96 | 76.27 | 237 | Cowboys |
| 47 | 21 | DaQuan Jones | 69.53 | 64.29 | 69.34 | 643 | Bills |
| 48 | 22 | Shy Tuttle | 69.38 | 58.64 | 73.00 | 557 | Saints |
| 49 | 23 | Sheldon Rankins | 69.30 | 61.02 | 72.75 | 558 | Jets |
| 50 | 24 | A'Shawn Robinson | 69.10 | 62.66 | 74.33 | 360 | Rams |
| 51 | 25 | Folorunso Fatukasi | 68.99 | 54.01 | 77.08 | 446 | Jaguars |
| 52 | 26 | Sebastian Joseph-Day | 68.78 | 51.92 | 79.28 | 702 | Chargers |
| 53 | 27 | Al Woods | 68.78 | 57.78 | 74.13 | 374 | Seahawks |
| 54 | 28 | Devonte Wyatt | 68.74 | 72.10 | 63.32 | 224 | Packers |
| 55 | 29 | Roy Robertson-Harris | 68.69 | 55.76 | 75.70 | 714 | Jaguars |
| 56 | 30 | Bilal Nichols | 68.27 | 55.50 | 72.61 | 801 | Raiders |
| 57 | 31 | Greg Gaines | 67.93 | 57.33 | 71.31 | 731 | Rams |
| 58 | 32 | Chris Wormley | 67.88 | 63.80 | 69.60 | 338 | Steelers |
| 59 | 33 | Teair Tart | 67.60 | 64.24 | 69.81 | 520 | Titans |
| 60 | 34 | Marquise Copeland | 67.37 | 62.53 | 70.59 | 343 | Rams |
| 61 | 35 | Dean Lowry | 67.18 | 56.01 | 71.45 | 482 | Packers |
| 62 | 36 | Jarran Reed | 67.13 | 50.77 | 73.87 | 705 | Packers |
| 63 | 37 | William Gholston | 67.04 | 47.45 | 75.93 | 484 | Buccaneers |
| 64 | 38 | Naquan Jones | 66.76 | 47.04 | 80.89 | 156 | Titans |
| 65 | 39 | Zach Allen | 66.72 | 58.75 | 75.00 | 660 | Cardinals |
| 66 | 40 | Taven Bryan | 66.68 | 59.51 | 68.38 | 642 | Browns |
| 67 | 41 | Lawrence Guy Sr. | 66.62 | 45.97 | 78.10 | 504 | Patriots |
| 68 | 42 | Arik Armstead | 66.45 | 56.93 | 74.21 | 349 | 49ers |
| 69 | 43 | Linval Joseph | 66.37 | 52.70 | 76.61 | 188 | Eagles |
| 70 | 44 | Nathan Shepherd | 66.31 | 58.42 | 67.82 | 416 | Jets |
| 71 | 45 | Myles Adams | 66.29 | 56.74 | 78.29 | 190 | Seahawks |
| 72 | 46 | Kevin Givens | 66.28 | 48.61 | 77.66 | 354 | 49ers |
| 73 | 47 | Adam Gotsis | 66.25 | 52.57 | 73.93 | 293 | Jaguars |
| 74 | 48 | Trysten Hill | 66.21 | 57.95 | 75.04 | 229 | Cardinals |
| 75 | 49 | Davon Godchaux | 65.88 | 53.48 | 72.27 | 659 | Patriots |
| 76 | 50 | Broderick Washington | 65.85 | 59.31 | 68.60 | 482 | Ravens |
| 77 | 51 | Maliek Collins | 65.83 | 59.06 | 68.58 | 601 | Texans |
| 78 | 52 | Mike Purcell | 65.77 | 53.94 | 72.75 | 529 | Broncos |
| 79 | 53 | James Lynch | 65.49 | 53.62 | 72.85 | 276 | Vikings |
| 80 | 54 | Ndamukong Suh | 64.80 | 45.50 | 77.91 | 176 | Eagles |
| 81 | 55 | Justin Jones | 64.75 | 50.35 | 72.56 | 746 | Bears |
| 82 | 56 | Armon Watts | 64.61 | 49.92 | 70.24 | 531 | Bears |
| 83 | 57 | Deadrin Senat | 64.55 | 61.51 | 71.21 | 165 | Buccaneers |
| 84 | 58 | Matt Henningsen | 64.28 | 56.06 | 65.60 | 230 | Broncos |
| 85 | 59 | Austin Johnson | 64.25 | 52.77 | 72.15 | 287 | Chargers |
| 86 | 60 | Hassan Ridgeway | 64.23 | 50.82 | 73.34 | 285 | 49ers |
| 87 | 61 | DeShawn Williams | 64.04 | 48.03 | 71.55 | 598 | Broncos |
| 88 | 62 | Travis Jones | 63.93 | 55.70 | 67.22 | 322 | Ravens |
| 89 | 63 | Quinton Jefferson | 63.68 | 44.60 | 72.24 | 566 | Seahawks |
| 90 | 64 | Kentavius Street | 63.64 | 45.11 | 72.03 | 518 | Saints |
| 91 | 65 | Roy Lopez | 63.54 | 48.11 | 70.03 | 557 | Texans |
| 92 | 66 | Jordan Phillips | 62.56 | 49.40 | 73.44 | 347 | Bills |
| 93 | 67 | Rashard Lawrence | 62.23 | 58.12 | 69.90 | 112 | Cardinals |
| 94 | 68 | Jonathan Bullard | 62.14 | 49.98 | 72.96 | 318 | Vikings |
| 95 | 69 | Jay Tufele | 62.06 | 50.87 | 76.25 | 137 | Bengals |
| 96 | 70 | Carlos Watkins | 62.00 | 47.53 | 70.52 | 278 | Cowboys |

### Rotation/backup (70 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 97 | 1 | Michael Dogbe | 61.83 | 44.27 | 73.29 | 282 | Cardinals |
| 98 | 2 | John Jenkins | 61.76 | 48.61 | 70.82 | 258 | Dolphins |
| 99 | 3 | Logan Hall | 61.69 | 45.80 | 68.12 | 403 | Buccaneers |
| 100 | 4 | Montravius Adams | 61.64 | 50.16 | 68.86 | 281 | Steelers |
| 101 | 5 | Eyioma Uwazurike | 61.58 | 56.89 | 69.35 | 165 | Broncos |
| 102 | 6 | Marquan McCall | 61.45 | 49.06 | 66.53 | 185 | Panthers |
| 103 | 7 | Solomon Thomas | 61.44 | 46.28 | 70.30 | 374 | Jets |
| 104 | 8 | Mike Pennel | 61.43 | 41.28 | 73.18 | 363 | Bears |
| 105 | 9 | Corey Peters | 61.43 | 48.14 | 70.91 | 264 | Jaguars |
| 106 | 10 | Neville Gallimore | 61.37 | 45.25 | 72.39 | 402 | Cowboys |
| 107 | 11 | Johnathan Hankins | 61.08 | 43.35 | 73.05 | 235 | Cowboys |
| 108 | 12 | Khalen Saunders | 61.05 | 51.49 | 69.39 | 421 | Chiefs |
| 109 | 13 | Michael Brockers | 60.75 | 44.26 | 73.48 | 123 | Lions |
| 110 | 14 | Benito Jones | 60.58 | 49.00 | 68.04 | 309 | Lions |
| 111 | 15 | Derrick Nnadi | 60.32 | 45.95 | 65.93 | 388 | Chiefs |
| 112 | 16 | John Ridgeway | 60.24 | 45.02 | 68.19 | 279 | Commanders |
| 113 | 17 | Rakeem Nunez-Roches | 60.15 | 46.30 | 65.51 | 548 | Buccaneers |
| 114 | 18 | Kurt Hinish | 60.00 | 48.02 | 65.79 | 435 | Texans |
| 115 | 19 | Ta'Quon Graham | 59.81 | 57.20 | 62.54 | 471 | Falcons |
| 116 | 20 | Josh Tupou | 59.68 | 49.02 | 65.56 | 272 | Bengals |
| 117 | 21 | Brent Urban | 59.62 | 41.92 | 70.98 | 298 | Ravens |
| 118 | 22 | Tershawn Wharton | 59.53 | 47.91 | 69.00 | 149 | Chiefs |
| 119 | 23 | Bryan Mone | 59.32 | 48.88 | 66.21 | 271 | Seahawks |
| 120 | 24 | Abdullah Anderson | 59.26 | 54.19 | 66.21 | 433 | Falcons |
| 121 | 25 | Jerry Tillery | 59.10 | 46.52 | 64.81 | 448 | Raiders |
| 122 | 26 | Angelo Blackson | 58.96 | 41.41 | 67.47 | 393 | Bears |
| 123 | 27 | Kyle Peko | 58.39 | 49.57 | 69.25 | 153 | Raiders |
| 124 | 28 | Christian Covington | 58.30 | 43.62 | 70.58 | 123 | Chargers |
| 125 | 29 | Joe Gaziano | 58.23 | 46.59 | 70.28 | 114 | Chargers |
| 126 | 30 | Larrell Murchison | 58.20 | 54.75 | 63.76 | 102 | Rams |
| 127 | 31 | Raekwon Davis | 58.17 | 47.50 | 62.49 | 583 | Dolphins |
| 128 | 32 | Carl Davis Jr. | 58.10 | 45.39 | 65.61 | 218 | Patriots |
| 129 | 33 | Jonathan Harris | 57.97 | 48.59 | 70.87 | 211 | Broncos |
| 130 | 34 | Ross Blacklock | 57.57 | 46.56 | 64.77 | 139 | Vikings |
| 131 | 35 | Daniel Ekuale | 57.55 | 49.32 | 64.26 | 362 | Patriots |
| 132 | 36 | Thomas Booker IV | 57.45 | 44.11 | 69.05 | 206 | Texans |
| 133 | 37 | Jonah Williams | 57.14 | 50.05 | 63.47 | 342 | Rams |
| 134 | 38 | Jalen Dalton | 56.94 | 42.04 | 74.01 | 145 | Falcons |
| 135 | 39 | Kevin Strong | 56.88 | 49.60 | 63.39 | 305 | Titans |
| 136 | 40 | Bravvion Roy | 56.82 | 45.91 | 62.10 | 299 | Panthers |
| 137 | 41 | Isaiah Buggs | 56.79 | 48.18 | 61.69 | 752 | Lions |
| 138 | 42 | T.Y. McGill | 56.44 | 43.90 | 70.83 | 180 | 49ers |
| 139 | 43 | Leki Fotu | 56.40 | 43.07 | 62.15 | 499 | Cardinals |
| 140 | 44 | Tyson Alualu | 56.37 | 37.65 | 69.30 | 291 | Steelers |
| 141 | 45 | Jaleel Johnson | 56.26 | 42.37 | 66.76 | 181 | Falcons |
| 142 | 46 | Michael Dwumfour | 56.25 | 45.70 | 69.16 | 238 | 49ers |
| 143 | 47 | Zach Carter | 56.10 | 42.35 | 62.08 | 395 | Bengals |
| 144 | 48 | L.J. Collier | 56.06 | 46.09 | 65.00 | 149 | Seahawks |
| 145 | 49 | Isaiahh Loudermilk | 56.02 | 46.30 | 63.35 | 116 | Steelers |
| 146 | 50 | Javon Kinlaw | 55.89 | 47.87 | 66.71 | 162 | 49ers |
| 147 | 51 | Byron Cowart | 55.84 | 38.40 | 66.43 | 229 | Colts |
| 148 | 52 | Justin Ellis | 55.75 | 37.91 | 64.11 | 362 | Giants |
| 149 | 53 | Henry Mondeaux | 55.75 | 42.48 | 65.22 | 249 | Giants |
| 150 | 54 | Jordan Elliott | 55.55 | 43.67 | 59.61 | 703 | Browns |
| 151 | 55 | Malcolm Roach | 54.88 | 45.07 | 63.62 | 316 | Saints |
| 152 | 56 | Otito Ogbonnia | 54.69 | 43.74 | 69.13 | 138 | Chargers |
| 153 | 57 | Perrion Winfrey | 54.50 | 42.64 | 62.15 | 342 | Browns |
| 154 | 58 | Bobby Brown III | 54.27 | 54.86 | 58.30 | 164 | Rams |
| 155 | 59 | Tommy Togiai | 54.16 | 45.39 | 62.94 | 225 | Browns |
| 156 | 60 | Kerry Hyder Jr. | 54.09 | 43.56 | 57.92 | 357 | 49ers |
| 157 | 61 | Ryder Anderson | 53.97 | 45.12 | 67.01 | 152 | Giants |
| 158 | 62 | Breiden Fehoko | 53.77 | 45.74 | 63.18 | 279 | Chargers |
| 159 | 63 | Jonathan Ledbetter | 52.70 | 43.12 | 62.64 | 275 | Cardinals |
| 160 | 64 | Quinton Bohanna | 52.69 | 42.54 | 58.84 | 264 | Cowboys |
| 161 | 65 | Akeem Spence | 52.63 | 34.95 | 68.49 | 147 | 49ers |
| 162 | 66 | Timmy Horne | 51.43 | 44.08 | 52.17 | 385 | Falcons |
| 163 | 67 | Neil Farrell Jr. | 51.36 | 44.42 | 59.67 | 316 | Raiders |
| 164 | 68 | Marlon Tuipulotu | 48.79 | 42.58 | 58.08 | 232 | Eagles |
| 165 | 69 | Earnest Brown IV | 46.30 | 44.42 | 59.66 | 136 | Rams |
| 166 | 70 | Eric Johnson | 45.59 | 42.10 | 46.69 | 127 | Colts |

## ED — Edge

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Micah Parsons | 93.09 | 90.94 | 90.35 | 917 | Cowboys |
| 2 | 2 | Nick Bosa | 92.26 | 97.18 | 88.22 | 745 | 49ers |
| 3 | 3 | Myles Garrett | 90.70 | 95.50 | 84.23 | 816 | Browns |
| 4 | 4 | T.J. Watt | 88.56 | 90.46 | 87.36 | 502 | Steelers |
| 5 | 5 | Joey Bosa | 88.55 | 95.13 | 87.02 | 165 | Chargers |
| 6 | 6 | Rashan Gary | 87.92 | 91.87 | 85.53 | 378 | Packers |
| 7 | 7 | Von Miller | 85.93 | 81.54 | 88.23 | 450 | Bills |
| 8 | 8 | Jaelan Phillips | 85.81 | 89.59 | 79.13 | 838 | Dolphins |
| 9 | 9 | Maxx Crosby | 85.63 | 91.15 | 77.78 | 1082 | Raiders |
| 10 | 10 | Greg Rousseau | 85.52 | 87.26 | 82.64 | 463 | Bills |
| 11 | 11 | Danielle Hunter | 84.53 | 84.19 | 83.52 | 905 | Vikings |
| 12 | 12 | DeMarcus Lawrence | 83.66 | 90.21 | 78.06 | 696 | Cowboys |
| 13 | 13 | Trey Hendrickson | 82.06 | 77.71 | 82.27 | 629 | Bengals |
| 14 | 14 | Khalil Mack | 82.02 | 79.68 | 82.35 | 860 | Chargers |
| 15 | 15 | Montez Sweat | 81.62 | 86.95 | 75.97 | 731 | Commanders |
| 16 | 16 | Haason Reddick | 81.21 | 69.66 | 85.05 | 816 | Eagles |
| 17 | 17 | Josh Uche | 80.80 | 71.99 | 86.43 | 373 | Patriots |
| 18 | 18 | Brandon Graham | 80.33 | 80.19 | 80.68 | 474 | Eagles |
| 19 | 19 | Sam Williams | 80.25 | 71.68 | 83.77 | 274 | Cowboys |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | James Houston | 79.77 | 68.15 | 94.65 | 140 | Lions |
| 21 | 2 | Josh Sweat | 79.58 | 80.17 | 76.22 | 587 | Eagles |
| 22 | 3 | Marcus Davenport | 78.47 | 83.36 | 74.82 | 490 | Saints |
| 23 | 4 | Cameron Jordan | 77.73 | 77.96 | 74.19 | 790 | Saints |
| 24 | 5 | Shaquil Barrett | 77.55 | 71.65 | 82.51 | 382 | Buccaneers |
| 25 | 6 | Justin Houston | 77.18 | 62.46 | 84.89 | 397 | Ravens |
| 26 | 7 | Uchenna Nwosu | 76.60 | 69.93 | 77.51 | 904 | Seahawks |
| 27 | 8 | Za'Darius Smith | 76.44 | 75.46 | 78.13 | 770 | Vikings |
| 28 | 9 | Matthew Judon | 76.12 | 59.66 | 83.35 | 858 | Patriots |
| 29 | 10 | Brian Burns | 75.87 | 64.29 | 80.12 | 951 | Panthers |
| 30 | 11 | Aidan Hutchinson | 75.46 | 76.19 | 70.81 | 953 | Lions |
| 31 | 12 | Andrew Van Ginkel | 75.42 | 64.37 | 79.10 | 333 | Dolphins |
| 32 | 13 | John Franklin-Myers | 75.04 | 74.39 | 71.60 | 643 | Jets |
| 33 | 14 | Ogbo Okoronkwo | 74.28 | 67.15 | 77.30 | 517 | Texans |

### Starter (63 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Azeez Ojulari | 73.63 | 65.72 | 86.04 | 230 | Giants |
| 35 | 2 | Bradley Chubb | 73.05 | 67.23 | 76.61 | 742 | Dolphins |
| 36 | 3 | Alex Highsmith | 72.10 | 65.98 | 72.31 | 941 | Steelers |
| 37 | 4 | Carl Granderson | 71.70 | 65.61 | 72.87 | 480 | Saints |
| 38 | 5 | Carlos Dunlap | 71.68 | 59.51 | 75.83 | 571 | Chiefs |
| 39 | 6 | Carl Lawson | 71.42 | 59.37 | 75.28 | 663 | Jets |
| 40 | 7 | Micheal Clemons | 71.40 | 67.51 | 70.81 | 311 | Jets |
| 41 | 8 | Jadeveon Clowney | 71.32 | 76.64 | 68.60 | 494 | Browns |
| 42 | 9 | Preston Smith | 71.16 | 62.53 | 73.04 | 825 | Packers |
| 43 | 10 | Melvin Ingram III | 70.88 | 61.26 | 75.59 | 509 | Dolphins |
| 44 | 11 | Bryce Huff | 70.27 | 67.25 | 72.35 | 191 | Jets |
| 45 | 12 | Darrell Taylor | 70.25 | 57.01 | 75.69 | 484 | Seahawks |
| 46 | 13 | Jerry Hughes | 70.24 | 58.60 | 74.04 | 689 | Texans |
| 47 | 14 | Jermaine Johnson | 69.33 | 64.43 | 71.37 | 312 | Jets |
| 48 | 15 | Sam Hubbard | 69.33 | 62.58 | 71.57 | 801 | Bengals |
| 49 | 16 | Arden Key | 69.27 | 70.34 | 64.80 | 475 | Jaguars |
| 50 | 17 | Yannick Ngakoue | 69.12 | 56.89 | 74.29 | 733 | Colts |
| 51 | 18 | Dante Fowler Jr. | 68.92 | 60.41 | 71.73 | 343 | Cowboys |
| 52 | 19 | Julian Okwara | 68.85 | 58.28 | 78.41 | 220 | Lions |
| 53 | 20 | Denico Autry | 68.83 | 53.08 | 77.61 | 531 | Titans |
| 54 | 21 | Markus Golden | 68.80 | 50.30 | 77.26 | 781 | Cardinals |
| 55 | 22 | Chandler Jones | 68.74 | 57.72 | 75.78 | 783 | Raiders |
| 56 | 23 | Kwity Paye | 68.53 | 68.89 | 67.92 | 547 | Colts |
| 57 | 24 | Leonard Floyd | 68.46 | 57.55 | 71.56 | 932 | Rams |
| 58 | 25 | Randy Gregory | 68.40 | 65.07 | 74.57 | 187 | Broncos |
| 59 | 26 | Deatrich Wise Jr. | 68.33 | 62.70 | 68.22 | 828 | Patriots |
| 60 | 27 | Samson Ebukam | 68.13 | 61.27 | 69.52 | 559 | 49ers |
| 61 | 28 | Payton Turner | 67.95 | 70.99 | 71.69 | 171 | Saints |
| 62 | 29 | Frank Clark | 67.91 | 56.86 | 73.18 | 716 | Chiefs |
| 63 | 30 | Odafe Oweh | 67.56 | 65.37 | 65.59 | 633 | Ravens |
| 64 | 31 | Jacob Martin | 67.50 | 59.88 | 70.79 | 261 | Broncos |
| 65 | 32 | A.J. Epenesa | 67.08 | 61.94 | 68.63 | 374 | Bills |
| 66 | 33 | Trevis Gipson | 66.82 | 56.17 | 71.92 | 641 | Bears |
| 67 | 34 | Myjai Sanders | 66.77 | 58.63 | 71.94 | 260 | Cardinals |
| 68 | 35 | Kayvon Thibodeaux | 66.62 | 70.23 | 62.98 | 740 | Giants |
| 69 | 36 | Jonathan Greenard | 66.58 | 64.19 | 70.73 | 284 | Texans |
| 70 | 37 | George Karlaftis | 66.26 | 57.74 | 67.78 | 729 | Chiefs |
| 71 | 38 | Robert Quinn | 66.15 | 47.39 | 76.95 | 393 | Eagles |
| 72 | 39 | Shaq Lawson | 65.98 | 58.39 | 68.69 | 467 | Bills |
| 73 | 40 | Charles Omenihu | 65.96 | 59.25 | 66.85 | 572 | 49ers |
| 74 | 41 | Dorance Armstrong | 65.94 | 60.12 | 66.83 | 542 | Cowboys |
| 75 | 42 | Chase Young | 65.52 | 80.67 | 60.67 | 114 | Commanders |
| 76 | 43 | Dawuane Smoot | 65.43 | 61.69 | 65.03 | 445 | Jaguars |
| 77 | 44 | Tyus Bowser | 65.29 | 55.10 | 71.84 | 355 | Ravens |
| 78 | 45 | Chase Winovich | 65.06 | 57.19 | 73.39 | 178 | Browns |
| 79 | 46 | Mario Addison | 64.72 | 44.62 | 76.62 | 367 | Texans |
| 80 | 47 | Dayo Odeyingbo | 64.50 | 63.24 | 63.74 | 519 | Colts |
| 81 | 48 | John Cominsky | 64.45 | 58.93 | 69.54 | 554 | Lions |
| 82 | 49 | Kingsley Enagbare | 64.43 | 58.85 | 63.99 | 465 | Packers |
| 83 | 50 | Joe Tryon-Shoyinka | 64.26 | 59.41 | 63.33 | 843 | Buccaneers |
| 84 | 51 | Emmanuel Ogbah | 63.84 | 59.27 | 66.63 | 326 | Dolphins |
| 85 | 52 | Boye Mafe | 63.79 | 59.78 | 62.29 | 423 | Seahawks |
| 86 | 53 | Clelin Ferrell | 63.75 | 61.71 | 62.77 | 492 | Raiders |
| 87 | 54 | Anthony Nelson | 63.56 | 60.87 | 61.18 | 632 | Buccaneers |
| 88 | 55 | Jonathon Cooper | 63.11 | 62.69 | 61.42 | 443 | Broncos |
| 89 | 56 | Dennis Gardeck | 62.86 | 52.21 | 68.14 | 210 | Cardinals |
| 90 | 57 | Vinny Curry | 62.48 | 47.20 | 73.00 | 184 | Jets |
| 91 | 58 | Mike Danna | 62.41 | 61.52 | 61.42 | 471 | Chiefs |
| 92 | 59 | Bud Dupree | 62.28 | 55.77 | 68.21 | 453 | Titans |
| 93 | 60 | Lorenzo Carter | 62.27 | 56.37 | 65.21 | 909 | Falcons |
| 94 | 61 | Efe Obada | 62.12 | 49.56 | 68.40 | 391 | Commanders |
| 95 | 62 | Arnold Ebiketie | 62.11 | 61.09 | 59.60 | 516 | Falcons |
| 96 | 63 | Travon Walker | 62.03 | 62.08 | 59.79 | 788 | Jaguars |

### Rotation/backup (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 97 | 1 | Ifeadi Odenigbo | 61.96 | 54.19 | 67.00 | 262 | Buccaneers |
| 98 | 2 | Drake Jackson | 61.90 | 59.70 | 61.16 | 315 | 49ers |
| 99 | 3 | Justin Hollins | 61.19 | 56.18 | 63.50 | 435 | Packers |
| 100 | 4 | Mario Edwards Jr. | 60.28 | 57.58 | 61.15 | 463 | Titans |
| 101 | 5 | D.J. Wonnum | 60.19 | 55.93 | 59.77 | 562 | Vikings |
| 102 | 6 | Carl Nassib | 60.11 | 55.80 | 62.37 | 250 | Buccaneers |
| 103 | 7 | Isaiah Thomas | 60.10 | 56.03 | 65.51 | 162 | Browns |
| 104 | 8 | Jason Pierre-Paul | 59.75 | 44.43 | 68.73 | 524 | Ravens |
| 105 | 9 | Marquis Haynes Sr. | 59.70 | 50.76 | 61.70 | 470 | Panthers |
| 106 | 10 | Romeo Okwara | 59.59 | 58.81 | 65.65 | 119 | Lions |
| 107 | 11 | Rasheem Green | 59.33 | 53.01 | 59.86 | 567 | Texans |
| 108 | 12 | Josh Paschal | 59.33 | 58.51 | 62.58 | 293 | Lions |
| 109 | 13 | Cam Sample | 59.13 | 59.67 | 56.69 | 411 | Bengals |
| 110 | 14 | Yetur Gross-Matos | 58.44 | 57.79 | 56.43 | 847 | Panthers |
| 111 | 15 | Malik Reed | 58.41 | 54.40 | 59.26 | 396 | Steelers |
| 112 | 16 | K'Lavon Chaisson | 58.38 | 57.14 | 59.55 | 109 | Jaguars |
| 113 | 17 | Casey Toohill | 58.32 | 55.72 | 58.55 | 347 | Commanders |
| 114 | 18 | Chris Rumph II | 58.24 | 56.16 | 57.43 | 300 | Chargers |
| 115 | 19 | Charles Harris | 57.66 | 55.00 | 61.28 | 259 | Lions |
| 116 | 20 | Bruce Irvin | 57.26 | 39.92 | 73.75 | 402 | Seahawks |
| 117 | 21 | Tarell Basham | 57.08 | 54.53 | 59.52 | 165 | Titans |
| 118 | 22 | Rashad Weaver | 56.69 | 56.65 | 58.68 | 640 | Titans |
| 119 | 23 | Oshane Ximines | 56.20 | 55.38 | 58.41 | 506 | Giants |
| 120 | 24 | Jordan Willis | 56.05 | 56.31 | 59.15 | 200 | 49ers |
| 121 | 25 | Tanoh Kpassagnon | 55.97 | 54.47 | 56.44 | 356 | Saints |
| 122 | 26 | Al-Quadin Muhammad | 55.83 | 51.79 | 54.84 | 609 | Bears |
| 123 | 27 | James Smith-Williams | 55.80 | 53.66 | 55.41 | 506 | Commanders |
| 124 | 28 | Tyquan Lewis | 55.61 | 56.42 | 61.66 | 273 | Colts |
| 125 | 29 | Dominique Robinson | 55.58 | 52.41 | 53.52 | 549 | Bears |
| 126 | 30 | Terrell Lewis | 55.57 | 56.32 | 57.27 | 332 | Bears |
| 127 | 31 | Adetokunbo Ogundeji | 55.00 | 53.71 | 52.68 | 541 | Falcons |
| 128 | 32 | Jonathan Garvin | 54.89 | 53.99 | 55.94 | 194 | Packers |
| 129 | 33 | Jihad Ward | 54.68 | 46.10 | 57.97 | 657 | Giants |
| 130 | 34 | Derrek Tuszka | 54.19 | 51.09 | 57.51 | 123 | Chargers |
| 131 | 35 | Austin Bryant | 54.07 | 55.01 | 56.16 | 207 | Lions |
| 132 | 36 | Sam Okuayinonu | 53.94 | 57.98 | 60.86 | 105 | Titans |
| 133 | 37 | Ben Banogu | 52.91 | 51.81 | 54.08 | 116 | Colts |
| 134 | 38 | Alex Wright | 52.60 | 52.28 | 48.65 | 543 | Browns |
| 135 | 39 | Patrick Johnson | 52.33 | 50.99 | 49.68 | 216 | Eagles |
| 136 | 40 | Isaac Rochell | 51.85 | 51.08 | 54.57 | 137 | Raiders |
| 137 | 41 | DeMarvin Leal | 51.19 | 53.22 | 51.56 | 175 | Steelers |
| 138 | 42 | Victor Dimukeje | 50.80 | 54.50 | 50.79 | 251 | Cardinals |
| 139 | 43 | Henry Anderson | 48.25 | 38.08 | 53.79 | 203 | Panthers |

## G — Guard

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Lindstrom | 98.58 | 95.00 | 96.80 | 1047 | Falcons |
| 2 | 2 | Joel Bitonio | 92.37 | 87.50 | 91.45 | 1172 | Browns |
| 3 | 3 | Quinn Meinerz | 85.24 | 77.70 | 86.10 | 752 | Broncos |
| 4 | 4 | Teven Jenkins | 84.91 | 80.70 | 83.55 | 576 | Bears |
| 5 | 5 | Joe Thuney | 83.59 | 77.30 | 83.62 | 999 | Chiefs |
| 6 | 6 | Robert Hunt | 83.56 | 73.70 | 85.97 | 1055 | Dolphins |
| 7 | 7 | Isaac Seumalo | 82.67 | 75.20 | 83.48 | 1135 | Eagles |
| 8 | 8 | Ezra Cleveland | 82.17 | 73.50 | 83.78 | 1134 | Vikings |
| 9 | 9 | Kevin Zeitler | 81.73 | 74.00 | 82.72 | 955 | Ravens |
| 10 | 10 | Elgton Jenkins | 81.21 | 72.30 | 82.99 | 960 | Packers |
| 11 | 11 | Zack Martin | 80.95 | 73.30 | 81.89 | 1143 | Cowboys |

### Good (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Trey Smith | 79.98 | 71.50 | 81.47 | 1039 | Chiefs |
| 13 | 2 | Damien Lewis | 79.92 | 71.80 | 81.16 | 1002 | Seahawks |
| 14 | 3 | Nate Davis | 79.59 | 70.60 | 81.41 | 682 | Titans |
| 15 | 4 | Wyatt Teller | 79.35 | 70.30 | 81.22 | 927 | Browns |
| 16 | 5 | Landon Dickerson | 78.15 | 67.30 | 81.22 | 1094 | Eagles |
| 17 | 6 | Austin Corbett | 77.94 | 69.10 | 79.67 | 985 | Panthers |
| 18 | 7 | Quenton Nelson | 77.60 | 68.40 | 79.56 | 1148 | Colts |
| 19 | 8 | Shaq Mason | 77.41 | 68.90 | 78.91 | 1200 | Buccaneers |
| 20 | 9 | Alex Cappa | 76.89 | 67.60 | 78.92 | 1086 | Bengals |
| 21 | 10 | Jonah Jackson | 76.71 | 66.10 | 79.61 | 858 | Lions |
| 22 | 11 | Michael Schofield III | 76.59 | 66.90 | 78.89 | 418 | Bears |
| 23 | 12 | Kevin Dotson | 76.19 | 65.40 | 79.22 | 1160 | Steelers |
| 24 | 13 | Will Hernandez | 76.16 | 65.40 | 79.17 | 843 | Cardinals |
| 25 | 14 | James Daniels | 76.13 | 66.90 | 78.12 | 1160 | Steelers |
| 26 | 15 | Oday Aboushi | 76.02 | 65.20 | 79.06 | 339 | Rams |
| 27 | 16 | Zion Johnson | 75.87 | 64.80 | 79.08 | 1184 | Chargers |
| 28 | 17 | Elijah Wilkinson | 75.69 | 64.30 | 79.12 | 574 | Falcons |
| 29 | 18 | A.J. Cann | 75.67 | 66.60 | 77.55 | 1003 | Texans |
| 30 | 19 | Cody Whitehair | 75.26 | 65.90 | 77.34 | 661 | Bears |

### Starter (45 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | David Edwards | 73.98 | 58.20 | 80.34 | 230 | Rams |
| 32 | 2 | Nick Leverett | 73.83 | 64.00 | 76.21 | 761 | Buccaneers |
| 33 | 3 | Mark Glowinski | 73.73 | 63.30 | 76.52 | 1045 | Giants |
| 34 | 4 | Robert Jones | 73.42 | 62.00 | 76.87 | 449 | Dolphins |
| 35 | 5 | Ben Powers | 73.20 | 62.90 | 75.90 | 1094 | Ravens |
| 36 | 6 | Aaron Banks | 73.10 | 62.70 | 75.86 | 969 | 49ers |
| 37 | 7 | Dylan Parham | 72.93 | 61.90 | 76.12 | 1036 | Raiders |
| 38 | 8 | Jon Runyan | 72.71 | 62.60 | 75.28 | 1051 | Packers |
| 39 | 9 | Justin Pugh | 72.40 | 61.00 | 75.84 | 263 | Cardinals |
| 40 | 10 | Ryan Bates | 72.38 | 61.80 | 75.27 | 945 | Bills |
| 41 | 11 | Ben Bartch | 72.23 | 60.50 | 75.89 | 293 | Jaguars |
| 42 | 12 | Will Fries | 72.04 | 58.40 | 76.96 | 642 | Colts |
| 43 | 13 | Andrew Norwell | 71.70 | 59.80 | 75.47 | 1120 | Commanders |
| 44 | 14 | Tyler Shatley | 71.43 | 60.90 | 74.28 | 819 | Jaguars |
| 45 | 15 | Aaron Brewer | 71.39 | 59.90 | 74.88 | 1031 | Titans |
| 46 | 16 | Dalton Risner | 71.06 | 61.10 | 73.53 | 967 | Broncos |
| 47 | 17 | Ben Bredeson | 70.68 | 56.70 | 75.84 | 542 | Giants |
| 48 | 18 | Greg Van Roten | 70.60 | 57.60 | 75.10 | 354 | Bills |
| 49 | 19 | Brandon Scherff | 70.21 | 59.00 | 73.51 | 1086 | Jaguars |
| 50 | 20 | Ed Ingram | 70.02 | 57.10 | 74.46 | 1168 | Vikings |
| 51 | 21 | Nate Herbig | 69.76 | 58.00 | 73.43 | 707 | Jets |
| 52 | 22 | Brady Christensen | 69.74 | 57.30 | 73.87 | 965 | Panthers |
| 53 | 23 | Phil Haynes | 69.61 | 57.10 | 73.78 | 485 | Seahawks |
| 54 | 24 | Royce Newman | 69.04 | 57.50 | 72.56 | 451 | Packers |
| 55 | 25 | Cesar Ruiz | 68.66 | 56.60 | 72.53 | 868 | Saints |
| 56 | 26 | Lucas Patrick | 68.55 | 55.90 | 72.81 | 269 | Bears |
| 57 | 27 | Colby Gossett | 68.52 | 55.30 | 73.16 | 267 | Falcons |
| 58 | 28 | Laken Tomlinson | 68.20 | 56.80 | 71.63 | 1110 | Jets |
| 59 | 29 | Nick Allegretti | 67.81 | 52.40 | 73.92 | 286 | Chiefs |
| 60 | 30 | Dan Feeney | 67.52 | 51.80 | 73.84 | 109 | Jets |
| 61 | 31 | Max Garcia | 67.49 | 54.50 | 71.98 | 542 | Cardinals |
| 62 | 32 | Cole Strange | 67.17 | 54.60 | 71.39 | 982 | Patriots |
| 63 | 33 | Gabe Jackson | 66.61 | 55.00 | 70.19 | 667 | Seahawks |
| 64 | 34 | Trai Turner | 66.55 | 53.00 | 71.41 | 766 | Commanders |
| 65 | 35 | Connor McGovern | 66.35 | 52.20 | 71.62 | 909 | Cowboys |
| 66 | 36 | Matt Feiler | 66.17 | 53.30 | 70.59 | 1189 | Chargers |
| 67 | 37 | Andrus Peat | 65.61 | 50.60 | 71.45 | 573 | Saints |
| 68 | 38 | Chandler Brewer | 65.29 | 55.60 | 67.58 | 228 | Rams |
| 69 | 39 | Cordell Volson | 64.97 | 51.60 | 69.71 | 1107 | Bengals |
| 70 | 40 | Spencer Burford | 64.53 | 49.60 | 70.31 | 744 | 49ers |
| 71 | 41 | Kayode Awosika | 64.29 | 50.40 | 69.39 | 155 | Lions |
| 72 | 42 | Jack Anderson | 64.22 | 47.90 | 70.94 | 148 | Giants |
| 73 | 43 | Jordan Roos | 63.97 | 52.70 | 67.31 | 202 | Titans |
| 74 | 44 | Rashaad Coward | 63.47 | 46.70 | 70.48 | 155 | Cardinals |
| 75 | 45 | Joshua Ezeudu | 62.04 | 46.00 | 68.57 | 290 | Giants |

### Rotation/backup (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 76 | 1 | Alex Bars | 61.30 | 45.40 | 67.73 | 852 | Raiders |
| 77 | 2 | Danny Pinter | 61.14 | 44.70 | 67.94 | 292 | Colts |
| 78 | 3 | Rodger Saffold | 61.04 | 43.70 | 68.44 | 1058 | Bills |
| 79 | 4 | Justin McCray | 60.84 | 41.40 | 69.64 | 151 | Texans |
| 80 | 5 | Luke Goedeke | 60.78 | 43.70 | 68.00 | 523 | Buccaneers |
| 81 | 6 | Lecitus Smith | 60.69 | 44.80 | 67.11 | 210 | Cardinals |
| 82 | 7 | Saahdiq Charles | 60.55 | 43.60 | 67.69 | 290 | Commanders |
| 83 | 8 | Logan Stenberg | 60.35 | 39.30 | 70.22 | 228 | Lions |
| 84 | 9 | Bobby Hart | 59.67 | 52.00 | 60.61 | 125 | Bills |
| 85 | 10 | Dillon Radunz | 58.86 | 40.30 | 67.06 | 280 | Titans |
| 86 | 11 | Liam Eichenberg | 58.85 | 39.80 | 67.39 | 627 | Dolphins |
| 87 | 12 | Cody Ford | 58.82 | 41.20 | 66.40 | 350 | Cardinals |
| 88 | 13 | Kenyon Green | 57.67 | 37.70 | 66.81 | 823 | Texans |

## HB — Running Back

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Chubb | 88.88 | 90.30 | 83.77 | 254 | Browns |
| 2 | 2 | Tony Pollard | 87.82 | 89.50 | 82.54 | 245 | Cowboys |
| 3 | 3 | Tyler Allgeier | 85.07 | 86.70 | 79.82 | 177 | Falcons |
| 4 | 4 | Josh Jacobs | 84.69 | 91.60 | 75.91 | 341 | Raiders |
| 5 | 5 | Rhamondre Stevenson | 83.45 | 81.30 | 80.71 | 339 | Patriots |
| 6 | 6 | Aaron Jones | 83.34 | 86.10 | 77.33 | 316 | Packers |
| 7 | 7 | Derrick Henry | 83.29 | 85.90 | 77.38 | 191 | Titans |
| 8 | 8 | Christian McCaffrey | 81.69 | 88.90 | 72.72 | 405 | 49ers |
| 9 | 9 | Breece Hall | 81.47 | 69.80 | 85.08 | 109 | Jets |
| 10 | 10 | Austin Ekeler | 81.41 | 81.30 | 77.31 | 437 | Chargers |
| 11 | 11 | Dameon Pierce | 80.43 | 78.90 | 77.29 | 192 | Texans |

### Good (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | James Cook | 79.43 | 78.20 | 76.09 | 126 | Bills |
| 13 | 2 | AJ Dillon | 79.06 | 81.40 | 73.34 | 252 | Packers |
| 14 | 3 | Jaylen Warren | 78.62 | 73.80 | 77.67 | 172 | Steelers |
| 15 | 4 | Kenneth Walker III | 78.07 | 76.70 | 74.81 | 247 | Seahawks |
| 16 | 5 | Devin Singletary | 77.66 | 75.90 | 74.67 | 373 | Bills |
| 17 | 6 | Travis Etienne Jr. | 77.55 | 72.10 | 77.02 | 316 | Jaguars |
| 18 | 7 | Cordarrelle Patterson | 77.36 | 81.00 | 70.77 | 170 | Falcons |
| 19 | 8 | Khalil Herbert | 77.14 | 72.50 | 76.06 | 107 | Bears |
| 20 | 9 | Miles Sanders | 76.99 | 74.00 | 74.81 | 269 | Eagles |
| 21 | 10 | Alvin Kamara | 76.82 | 72.70 | 75.40 | 295 | Saints |
| 22 | 11 | Saquon Barkley | 76.64 | 77.10 | 72.17 | 380 | Giants |
| 23 | 12 | Raheem Mostert | 76.63 | 75.70 | 73.08 | 294 | Dolphins |
| 24 | 13 | Jonathan Taylor | 76.44 | 67.60 | 78.17 | 256 | Colts |
| 25 | 14 | D'Andre Swift | 76.14 | 78.10 | 70.67 | 236 | Lions |
| 26 | 15 | Kareem Hunt | 74.83 | 67.00 | 75.88 | 254 | Browns |
| 27 | 16 | Joe Mixon | 74.50 | 79.30 | 67.14 | 301 | Bengals |
| 28 | 17 | Najee Harris | 74.29 | 73.50 | 70.65 | 297 | Steelers |
| 29 | 18 | Latavius Murray | 74.27 | 82.50 | 64.61 | 191 | Broncos |
| 30 | 19 | Dalvin Cook | 74.25 | 67.40 | 74.65 | 395 | Vikings |
| 31 | 20 | Isiah Pacheco | 74.09 | 74.60 | 69.59 | 147 | Chiefs |
| 32 | 21 | Dontrell Hilliard | 74.02 | 67.10 | 74.47 | 130 | Titans |

### Starter (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Cam Akers | 73.85 | 80.70 | 65.11 | 165 | Rams |
| 34 | 2 | Eno Benjamin | 73.41 | 72.20 | 70.05 | 195 | Saints |
| 35 | 3 | Antonio Gibson | 73.32 | 76.30 | 67.17 | 238 | Commanders |
| 36 | 4 | Alexander Mattison | 73.15 | 71.30 | 70.21 | 158 | Vikings |
| 37 | 5 | Ezekiel Elliott | 73.00 | 71.90 | 69.57 | 234 | Cowboys |
| 38 | 6 | Chuba Hubbard | 72.10 | 76.60 | 64.94 | 110 | Panthers |
| 39 | 7 | D'Onta Foreman | 71.69 | 73.10 | 66.58 | 112 | Panthers |
| 40 | 8 | James Conner | 71.68 | 69.70 | 68.84 | 332 | Cardinals |
| 41 | 9 | Samaje Perine | 71.61 | 71.70 | 67.38 | 239 | Bengals |
| 42 | 10 | Clyde Edwards-Helaire | 70.86 | 67.60 | 68.86 | 124 | Chiefs |
| 43 | 11 | Michael Carter | 70.74 | 61.30 | 72.87 | 267 | Jets |
| 44 | 12 | David Montgomery | 70.65 | 67.90 | 68.32 | 270 | Bears |
| 45 | 13 | Justin Jackson | 70.53 | 58.60 | 74.32 | 100 | Lions |
| 46 | 14 | Jamaal Williams | 70.24 | 73.40 | 63.96 | 119 | Lions |
| 47 | 15 | DeeJay Dallas | 69.92 | 66.70 | 67.90 | 117 | Seahawks |
| 48 | 16 | Leonard Fournette | 69.59 | 68.00 | 66.49 | 392 | Buccaneers |
| 49 | 17 | JaMycal Hasty | 69.39 | 72.70 | 63.01 | 121 | Jaguars |
| 50 | 18 | Matt Breida | 68.96 | 66.80 | 66.23 | 132 | Giants |
| 51 | 19 | James Robinson | 68.54 | 59.70 | 70.27 | 106 | Jets |
| 52 | 20 | Jeff Wilson Jr. | 68.30 | 66.90 | 65.07 | 253 | Dolphins |
| 53 | 21 | Justice Hill | 67.95 | 60.30 | 68.89 | 124 | Ravens |
| 54 | 22 | Jerick McKinnon | 66.62 | 63.20 | 64.74 | 343 | Chiefs |
| 55 | 23 | Ameer Abdullah | 66.06 | 66.40 | 61.66 | 134 | Raiders |
| 56 | 24 | Rachaad White | 66.04 | 66.40 | 61.64 | 257 | Buccaneers |
| 57 | 25 | Kenneth Gainwell | 66.02 | 57.00 | 67.86 | 193 | Eagles |
| 58 | 26 | Nyheim Hines | 65.90 | 63.30 | 63.46 | 136 | Bills |
| 59 | 27 | Kenyan Drake | 65.67 | 53.10 | 69.88 | 162 | Ravens |
| 60 | 28 | Rex Burkhead | 64.16 | 59.10 | 63.36 | 189 | Texans |
| 61 | 29 | J.D. McKissic | 63.47 | 54.20 | 65.48 | 154 | Commanders |
| 62 | 30 | Dare Ogunbowale | 63.38 | 65.00 | 58.13 | 100 | Texans |
| 63 | 31 | Joshua Kelley | 63.05 | 63.00 | 58.92 | 135 | Chargers |
| 64 | 32 | Chase Edmonds | 62.63 | 50.80 | 66.35 | 163 | Broncos |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 65 | 1 | Deon Jackson | 61.88 | 62.90 | 57.04 | 142 | Colts |

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

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Germaine Pratt | 79.41 | 80.60 | 76.01 | 722 | Bengals |
| 6 | 2 | Ja'Whaun Bentley | 79.05 | 80.40 | 74.90 | 907 | Patriots |
| 7 | 3 | Demario Davis | 78.79 | 82.70 | 72.32 | 1132 | Saints |
| 8 | 4 | Alex Singleton | 78.59 | 79.10 | 75.57 | 771 | Broncos |
| 9 | 5 | Nick Bolton | 77.89 | 75.70 | 75.55 | 1118 | Chiefs |
| 10 | 6 | Kaden Elliss | 77.86 | 81.50 | 71.26 | 632 | Saints |
| 11 | 7 | Dre Greenlaw | 77.68 | 81.20 | 76.89 | 850 | 49ers |
| 12 | 8 | Tremaine Edmunds | 76.54 | 79.00 | 73.48 | 760 | Bills |
| 13 | 9 | Oren Burks | 75.44 | 79.20 | 73.03 | 156 | 49ers |
| 14 | 10 | E.J. Speed | 74.99 | 78.40 | 72.55 | 316 | Colts |
| 15 | 11 | Jerome Baker | 74.76 | 78.00 | 68.74 | 1010 | Dolphins |
| 16 | 12 | Anthony Walker Jr. | 74.29 | 82.70 | 72.55 | 120 | Browns |

### Starter (53 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Jahlani Tavai | 73.91 | 73.50 | 72.37 | 570 | Patriots |
| 18 | 2 | Bobby Okereke | 73.82 | 73.30 | 70.42 | 970 | Colts |
| 19 | 3 | Frankie Luvu | 73.35 | 74.80 | 69.49 | 941 | Panthers |
| 20 | 4 | De'Vondre Campbell | 73.35 | 75.60 | 69.94 | 694 | Packers |
| 21 | 5 | Logan Wilson | 73.19 | 72.70 | 72.34 | 954 | Bengals |
| 22 | 6 | Ben Niemann | 73.03 | 71.00 | 70.91 | 484 | Cardinals |
| 23 | 7 | Brian Asamoah II | 72.86 | 78.80 | 71.60 | 121 | Vikings |
| 24 | 8 | Roquan Smith | 72.76 | 70.60 | 70.03 | 1039 | Ravens |
| 25 | 9 | Shaq Thompson | 72.60 | 72.30 | 69.52 | 1089 | Panthers |
| 26 | 10 | C.J. Mosley | 72.08 | 69.80 | 69.74 | 1113 | Jets |
| 27 | 11 | David Long Jr. | 72.07 | 76.20 | 70.91 | 740 | Titans |
| 28 | 12 | Denzel Perryman | 71.67 | 74.20 | 69.49 | 555 | Raiders |
| 29 | 13 | Leighton Vander Esch | 71.64 | 70.80 | 70.75 | 745 | Cowboys |
| 30 | 14 | Matt Milano | 71.23 | 73.70 | 67.94 | 946 | Bills |
| 31 | 15 | Foyesade Oluokun | 71.11 | 69.60 | 68.15 | 1145 | Jaguars |
| 32 | 16 | Cory Littleton | 70.46 | 72.20 | 67.11 | 372 | Panthers |
| 33 | 17 | Patrick Queen | 70.26 | 70.00 | 66.26 | 1024 | Ravens |
| 34 | 18 | Josey Jewell | 69.24 | 71.70 | 69.80 | 825 | Broncos |
| 35 | 19 | Jordan Hicks | 68.91 | 65.40 | 67.08 | 934 | Vikings |
| 36 | 20 | Willie Gay | 68.25 | 69.60 | 66.81 | 607 | Chiefs |
| 37 | 21 | Azeez Al-Shaair | 67.64 | 67.80 | 67.62 | 313 | 49ers |
| 38 | 22 | Leo Chenal | 67.62 | 66.00 | 64.54 | 262 | Chiefs |
| 39 | 23 | Ernest Jones | 67.53 | 63.60 | 68.19 | 723 | Rams |
| 40 | 24 | Sione Takitaki | 67.22 | 66.50 | 66.79 | 498 | Browns |
| 41 | 25 | Kyzir White | 67.10 | 65.00 | 65.37 | 843 | Eagles |
| 42 | 26 | Cole Holcomb | 66.63 | 66.60 | 68.94 | 446 | Commanders |
| 43 | 27 | Matthew Adams | 66.57 | 69.60 | 68.24 | 189 | Bears |
| 44 | 28 | Jamin Davis | 66.34 | 62.90 | 65.45 | 833 | Commanders |
| 45 | 29 | Drue Tranquill | 65.86 | 66.50 | 65.26 | 977 | Chargers |
| 46 | 30 | Jeremiah Owusu-Koramoah | 65.83 | 65.50 | 66.67 | 535 | Browns |
| 47 | 31 | Pete Werner | 65.66 | 64.70 | 66.30 | 596 | Saints |
| 48 | 32 | Eric Kendricks | 65.61 | 61.10 | 66.08 | 1094 | Vikings |
| 49 | 33 | Cody Barton | 65.54 | 63.70 | 66.61 | 894 | Seahawks |
| 50 | 34 | Malcolm Rodriguez | 65.43 | 62.80 | 64.00 | 611 | Lions |
| 51 | 35 | Jake Hansen | 65.40 | 69.10 | 64.65 | 205 | Texans |
| 52 | 36 | Malik Harrison | 65.37 | 66.50 | 64.09 | 248 | Ravens |
| 53 | 37 | Damone Clark | 65.28 | 65.50 | 67.84 | 398 | Cowboys |
| 54 | 38 | Chris Board | 65.23 | 64.40 | 62.89 | 158 | Lions |
| 55 | 39 | Ezekiel Turner | 64.45 | 63.60 | 71.31 | 108 | Cardinals |
| 56 | 40 | Kwon Alexander | 64.43 | 63.00 | 63.52 | 558 | Jets |
| 57 | 41 | Rashaan Evans | 64.15 | 59.60 | 64.49 | 1104 | Falcons |
| 58 | 42 | Derrick Barnes | 64.08 | 62.30 | 63.92 | 346 | Lions |
| 59 | 43 | Akeem Davis-Gaither | 63.84 | 60.80 | 64.55 | 228 | Bengals |
| 60 | 44 | Monty Rice | 63.31 | 63.30 | 67.48 | 366 | Titans |
| 61 | 45 | Mykal Walker | 63.06 | 58.70 | 62.59 | 769 | Falcons |
| 62 | 46 | Devin Bush | 62.96 | 58.90 | 64.67 | 659 | Steelers |
| 63 | 47 | Jack Sanborn | 62.89 | 64.50 | 66.47 | 330 | Bears |
| 64 | 48 | Jarrad Davis | 62.70 | 68.10 | 64.09 | 106 | Giants |
| 65 | 49 | Zaven Collins | 62.60 | 59.80 | 62.39 | 1025 | Cardinals |
| 66 | 50 | Alex Anzalone | 62.56 | 59.20 | 61.71 | 1076 | Lions |
| 67 | 51 | Elandon Roberts | 62.19 | 57.10 | 62.05 | 676 | Dolphins |
| 68 | 52 | Darius Harris | 62.11 | 64.40 | 64.03 | 292 | Chiefs |
| 69 | 53 | Joe Thomas | 62.06 | 63.00 | 63.32 | 413 | Bears |

### Rotation/backup (45 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 70 | 1 | Duke Riley | 61.84 | 59.10 | 60.71 | 364 | Dolphins |
| 71 | 2 | Quay Walker | 61.60 | 52.00 | 63.83 | 846 | Packers |
| 72 | 3 | Zaire Franklin | 61.31 | 57.00 | 61.49 | 1136 | Colts |
| 73 | 4 | Zach Cunningham | 61.29 | 60.30 | 64.05 | 205 | Titans |
| 74 | 5 | Robert Spillane | 60.54 | 52.50 | 64.95 | 588 | Steelers |
| 75 | 6 | Anthony Barr | 60.54 | 59.70 | 63.08 | 608 | Cowboys |
| 76 | 7 | Damien Wilson | 60.39 | 57.10 | 61.99 | 204 | Panthers |
| 77 | 8 | Jordyn Brooks | 60.26 | 52.80 | 61.96 | 1026 | Seahawks |
| 78 | 9 | Nicholas Morrow | 60.24 | 54.00 | 60.65 | 1086 | Bears |
| 79 | 10 | Jaylon Smith | 60.21 | 56.00 | 62.86 | 626 | Giants |
| 80 | 11 | Chad Muma | 60.18 | 54.90 | 62.47 | 286 | Jaguars |
| 81 | 12 | Myles Jack | 59.73 | 53.60 | 61.64 | 692 | Steelers |
| 82 | 13 | Christian Kirksey | 59.66 | 56.10 | 60.08 | 1139 | Texans |
| 83 | 14 | Terrel Bernard | 59.16 | 64.60 | 65.15 | 111 | Bills |
| 84 | 15 | Divine Deablo | 59.03 | 58.40 | 64.10 | 463 | Raiders |
| 85 | 16 | Isaiah McDuffie | 58.94 | 58.60 | 65.16 | 175 | Packers |
| 86 | 17 | Kenneth Murray Jr. | 58.89 | 47.80 | 63.89 | 718 | Chargers |
| 87 | 18 | Josh Bynes | 58.85 | 54.20 | 63.57 | 269 | Ravens |
| 88 | 19 | Quincy Williams | 58.06 | 55.20 | 59.98 | 792 | Jets |
| 89 | 20 | David Mayo | 57.65 | 54.70 | 61.98 | 202 | Commanders |
| 90 | 21 | Devin Lloyd | 57.60 | 48.30 | 59.64 | 925 | Jaguars |
| 91 | 22 | Tony Fields II | 55.60 | 50.00 | 61.05 | 276 | Browns |
| 92 | 23 | Dylan Cole | 55.58 | 53.30 | 60.26 | 439 | Titans |
| 93 | 24 | Devin White | 55.08 | 45.50 | 57.50 | 1075 | Buccaneers |
| 94 | 25 | Jack Gibbens | 55.00 | 60.00 | 63.76 | 214 | Titans |
| 95 | 26 | Jonas Griffith | 54.59 | 52.90 | 60.86 | 336 | Broncos |
| 96 | 27 | Deion Jones | 54.39 | 48.30 | 57.52 | 422 | Browns |
| 97 | 28 | Krys Barnes | 54.26 | 46.70 | 61.65 | 141 | Packers |
| 98 | 29 | Blake Cashman | 54.11 | 54.00 | 59.31 | 149 | Texans |
| 99 | 30 | Jordan Kunaszyk | 54.03 | 52.40 | 62.42 | 101 | Browns |
| 100 | 31 | Mack Wilson Sr. | 53.25 | 47.10 | 57.43 | 234 | Patriots |
| 101 | 32 | Raekwon McMillan | 52.70 | 42.70 | 57.01 | 250 | Patriots |
| 102 | 33 | Troy Andersen | 51.71 | 40.20 | 56.20 | 481 | Falcons |
| 103 | 34 | Tyrel Dodson | 51.58 | 48.40 | 59.46 | 220 | Bills |
| 104 | 35 | Micah McFadden | 51.47 | 38.70 | 57.79 | 435 | Giants |
| 105 | 36 | A.J. Klein | 51.35 | 47.70 | 57.75 | 104 | Bills |
| 106 | 37 | Jon Bostic | 51.20 | 46.10 | 58.18 | 263 | Commanders |
| 107 | 38 | Garret Wallow | 50.91 | 43.90 | 60.11 | 124 | Texans |
| 108 | 39 | Tanner Vallejo | 50.03 | 41.10 | 56.53 | 282 | Cardinals |
| 109 | 40 | Jayon Brown | 47.86 | 40.20 | 56.52 | 423 | Raiders |
| 110 | 41 | Jacob Phillips | 45.41 | 36.70 | 57.23 | 320 | Browns |
| 111 | 42 | Tae Crowder | 45.00 | 29.60 | 54.79 | 445 | Steelers |
| 112 | 43 | Christian Harris | 45.00 | 28.30 | 54.07 | 711 | Texans |
| 113 | 44 | Kamu Grugier-Hill | 45.00 | 29.60 | 54.83 | 418 | Cardinals |
| 114 | 45 | Luke Masterson | 45.00 | 30.80 | 55.40 | 344 | Raiders |

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
| 8 | 5 | Jalen Hurts | 76.59 | 77.90 | 75.10 | 573 | Eagles |
| 9 | 6 | Tua Tagovailoa | 75.73 | 76.45 | 76.88 | 455 | Dolphins |

### Starter (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Geno Smith | 73.84 | 75.80 | 77.41 | 693 | Seahawks |
| 11 | 2 | Ryan Tannehill | 73.35 | 76.53 | 71.75 | 397 | Titans |
| 12 | 3 | Dak Prescott | 73.27 | 74.83 | 72.51 | 453 | Cowboys |
| 13 | 4 | Jared Goff | 72.44 | 69.40 | 71.52 | 663 | Lions |
| 14 | 5 | Matthew Stafford | 72.12 | 73.34 | 72.33 | 347 | Rams |
| 15 | 6 | Andy Dalton | 72.01 | 76.75 | 71.16 | 431 | Saints |
| 16 | 7 | Jimmy Garoppolo | 71.71 | 70.51 | 76.49 | 348 | 49ers |
| 17 | 8 | Trevor Lawrence | 71.58 | 68.81 | 69.48 | 672 | Jaguars |
| 18 | 9 | Derek Carr | 71.37 | 70.50 | 69.54 | 575 | Raiders |
| 19 | 10 | Russell Wilson | 70.01 | 68.68 | 69.26 | 613 | Broncos |
| 20 | 11 | Mac Jones | 69.88 | 71.37 | 67.79 | 523 | Patriots |
| 21 | 12 | Daniel Jones | 69.64 | 71.33 | 67.04 | 612 | Giants |
| 22 | 13 | Kyler Murray | 69.60 | 72.10 | 67.42 | 473 | Cardinals |
| 23 | 14 | Lamar Jackson | 69.23 | 70.98 | 69.86 | 399 | Ravens |
| 24 | 15 | Matt Ryan | 67.77 | 68.95 | 64.97 | 535 | Colts |
| 25 | 16 | Jacoby Brissett | 66.56 | 74.89 | 67.54 | 448 | Browns |
| 26 | 17 | Brock Purdy | 64.77 | 74.20 | 76.02 | 202 | 49ers |
| 27 | 18 | Bailey Zappe | 64.67 | 74.00 | 76.49 | 109 | Patriots |
| 28 | 19 | Kenny Pickett | 62.88 | 73.00 | 61.10 | 477 | Steelers |
| 29 | 20 | Davis Mills | 62.84 | 62.49 | 62.63 | 562 | Texans |
| 30 | 21 | Mitch Trubisky | 62.34 | 70.56 | 67.26 | 207 | Steelers |

### Rotation/backup (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Carson Wentz | 61.37 | 62.28 | 62.94 | 337 | Commanders |
| 32 | 2 | Jameis Winston | 61.05 | 66.36 | 68.15 | 137 | Saints |
| 33 | 3 | Sam Darnold | 60.81 | 58.66 | 68.72 | 173 | Panthers |
| 34 | 4 | Marcus Mariota | 60.68 | 62.90 | 67.18 | 373 | Falcons |
| 35 | 5 | Jarrett Stidham | 60.22 | 61.32 | 68.89 | 107 | Raiders |
| 36 | 6 | Justin Fields | 60.01 | 55.91 | 65.22 | 474 | Bears |
| 37 | 7 | Deshaun Watson | 59.76 | 64.19 | 63.93 | 218 | Browns |
| 38 | 8 | Baker Mayfield | 59.56 | 58.79 | 61.68 | 411 | Rams |
| 39 | 9 | Colt McCoy | 59.05 | 64.63 | 61.45 | 163 | Cardinals |
| 40 | 10 | Taylor Heinicke | 58.81 | 53.38 | 67.36 | 304 | Commanders |
| 41 | 11 | P.J. Walker | 58.56 | 58.69 | 64.37 | 124 | Panthers |
| 42 | 12 | Desmond Ridder | 58.41 | 60.10 | 62.59 | 144 | Falcons |
| 43 | 13 | Mike White | 57.82 | 57.75 | 62.12 | 189 | Jets |
| 44 | 14 | Tyler Huntley | 57.36 | 59.49 | 58.83 | 141 | Ravens |
| 45 | 15 | Cooper Rush | 57.20 | 55.97 | 61.09 | 182 | Cowboys |
| 46 | 16 | Skylar Thompson | 57.05 | 63.30 | 54.47 | 127 | Dolphins |
| 47 | 17 | Sam Ehlinger | 56.51 | 56.42 | 57.90 | 132 | Colts |
| 48 | 18 | Joe Flacco | 55.64 | 53.80 | 56.55 | 212 | Jets |
| 49 | 19 | Zach Wilson | 52.91 | 47.70 | 58.53 | 289 | Jets |

## S — Safety

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Tyrann Mathieu | 86.12 | 87.90 | 81.26 | 1128 | Saints |
| 2 | 2 | Kevin Byard | 85.72 | 81.00 | 84.70 | 1139 | Titans |
| 3 | 3 | Minkah Fitzpatrick | 85.64 | 83.30 | 84.30 | 939 | Steelers |
| 4 | 4 | Rodney McLeod | 85.59 | 85.30 | 83.42 | 1034 | Colts |
| 5 | 5 | Damontae Kazee | 80.84 | 78.60 | 84.58 | 273 | Steelers |
| 6 | 6 | Josh Metellus | 80.22 | 80.70 | 83.81 | 259 | Vikings |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Kamren Curl | 79.84 | 80.80 | 79.94 | 727 | Commanders |
| 8 | 2 | Duron Harmon | 79.73 | 77.60 | 76.98 | 1076 | Raiders |
| 9 | 3 | Kyle Hamilton | 78.49 | 76.80 | 76.43 | 547 | Ravens |
| 10 | 4 | Rudy Ford | 78.41 | 77.80 | 76.11 | 443 | Packers |
| 11 | 5 | Quandre Diggs | 78.38 | 76.40 | 75.53 | 1156 | Seahawks |
| 12 | 6 | Kyle Dugger | 78.27 | 78.30 | 76.07 | 752 | Patriots |
| 13 | 7 | Jessie Bates III | 78.05 | 72.90 | 78.40 | 1016 | Bengals |
| 14 | 8 | P.J. Locke | 77.88 | 84.60 | 76.85 | 112 | Broncos |
| 15 | 9 | Taylor Rapp | 77.31 | 72.90 | 78.04 | 976 | Rams |
| 16 | 10 | Adrian Phillips | 77.06 | 71.60 | 76.53 | 702 | Patriots |
| 17 | 11 | Malik Hooker | 76.97 | 72.40 | 79.85 | 860 | Cowboys |
| 18 | 12 | Marcus Williams | 76.96 | 71.70 | 80.45 | 637 | Ravens |
| 19 | 13 | Jordan Poyer | 76.88 | 76.20 | 75.92 | 754 | Bills |
| 20 | 14 | Julian Love | 76.62 | 71.50 | 76.57 | 1006 | Giants |
| 21 | 15 | Eddie Jackson | 76.58 | 74.30 | 77.26 | 697 | Bears |
| 22 | 16 | Caden Sterns | 75.94 | 81.30 | 76.65 | 274 | Broncos |
| 23 | 17 | Devin McCourty | 75.93 | 67.60 | 77.32 | 1097 | Patriots |
| 24 | 18 | Derwin James Jr. | 75.21 | 74.40 | 75.93 | 835 | Chargers |
| 25 | 19 | Harrison Smith | 74.77 | 69.50 | 76.18 | 912 | Vikings |
| 26 | 20 | Justin Simmons | 74.42 | 74.30 | 72.79 | 808 | Broncos |
| 27 | 21 | Tracy Walker III | 74.30 | 75.40 | 77.06 | 139 | Lions |
| 28 | 22 | Antoine Winfield Jr. | 74.21 | 66.60 | 78.25 | 764 | Buccaneers |
| 29 | 23 | Justin Reid | 74.05 | 73.80 | 71.85 | 1112 | Chiefs |

### Starter (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Budda Baker | 73.34 | 69.10 | 73.18 | 969 | Cardinals |
| 31 | 2 | Marcus Maye | 73.32 | 71.30 | 77.16 | 669 | Saints |
| 32 | 3 | M.J. Stewart | 73.03 | 70.50 | 76.43 | 178 | Texans |
| 33 | 4 | Juan Thornhill | 72.46 | 68.70 | 71.29 | 1043 | Chiefs |
| 34 | 5 | Reed Blankenship | 72.22 | 71.60 | 77.28 | 292 | Eagles |
| 35 | 6 | Landon Collins | 72.16 | 79.60 | 71.48 | 160 | Giants |
| 36 | 7 | Jabrill Peppers | 71.98 | 67.10 | 74.51 | 398 | Patriots |
| 37 | 8 | Jayron Kearse | 71.95 | 63.60 | 75.11 | 815 | Cowboys |
| 38 | 9 | Darrick Forrest | 71.86 | 69.00 | 75.12 | 849 | Commanders |
| 39 | 10 | Jordan Whitehead | 71.72 | 70.60 | 69.18 | 1129 | Jets |
| 40 | 11 | Jimmie Ward | 71.65 | 69.30 | 72.22 | 509 | 49ers |
| 41 | 12 | Richie Grant | 71.28 | 69.00 | 68.63 | 1117 | Falcons |
| 42 | 13 | Andre Cisco | 71.18 | 68.00 | 72.56 | 992 | Jaguars |
| 43 | 14 | Terrell Edmunds | 71.15 | 68.00 | 70.26 | 886 | Steelers |
| 44 | 15 | Talanoa Hufanga | 71.09 | 67.80 | 70.22 | 1029 | 49ers |
| 45 | 16 | Geno Stone | 70.99 | 74.70 | 71.00 | 450 | Ravens |
| 46 | 17 | Nasir Adderley | 70.97 | 68.20 | 69.94 | 882 | Chargers |
| 47 | 18 | Andrew Adams | 70.87 | 74.80 | 71.18 | 726 | Titans |
| 48 | 19 | C.J. Moore | 70.87 | 75.00 | 72.77 | 106 | Lions |
| 49 | 20 | Tony Jefferson | 70.77 | 77.00 | 72.20 | 164 | Giants |
| 50 | 21 | Bobby McCain | 70.75 | 69.50 | 68.88 | 970 | Commanders |
| 51 | 22 | Amani Hooker | 70.19 | 71.50 | 70.55 | 522 | Titans |
| 52 | 23 | Verone McKinley III | 70.18 | 72.70 | 71.20 | 251 | Dolphins |
| 53 | 24 | Chuck Clark | 69.97 | 61.20 | 71.95 | 1091 | Ravens |
| 54 | 25 | Jevon Holland | 69.84 | 63.80 | 70.07 | 1123 | Dolphins |
| 55 | 26 | Kerby Joseph | 69.39 | 65.00 | 71.09 | 875 | Lions |
| 56 | 27 | Xavier Woods | 69.05 | 64.00 | 69.44 | 1001 | Panthers |
| 57 | 28 | Tashaun Gipson Sr. | 69.01 | 58.00 | 73.65 | 1036 | 49ers |
| 58 | 29 | Jeremy Reaves | 68.94 | 70.80 | 72.08 | 149 | Commanders |
| 59 | 30 | DeShon Elliott | 68.92 | 64.10 | 72.67 | 859 | Lions |
| 60 | 31 | Grant Delpit | 68.84 | 62.50 | 69.48 | 1086 | Browns |
| 61 | 32 | John Johnson III | 68.74 | 66.60 | 66.58 | 1056 | Browns |
| 62 | 33 | Jalen Thompson | 68.32 | 59.70 | 72.19 | 1098 | Cardinals |
| 63 | 34 | Micah Hyde | 68.27 | 69.10 | 71.12 | 101 | Bills |
| 64 | 35 | Andrew Wingard | 68.22 | 63.00 | 72.89 | 217 | Jaguars |
| 65 | 36 | Jalen Pitre | 68.15 | 65.80 | 65.55 | 1088 | Texans |
| 66 | 37 | Vonn Bell | 67.65 | 65.50 | 65.70 | 1023 | Bengals |
| 67 | 38 | Donovan Wilson | 66.91 | 64.50 | 67.75 | 959 | Cowboys |
| 68 | 39 | Jaquan Brisker | 66.63 | 65.00 | 65.51 | 954 | Bears |
| 69 | 40 | Daniel Sorensen | 66.58 | 61.00 | 69.76 | 166 | Saints |
| 70 | 41 | Bryan Cook | 66.52 | 60.90 | 67.09 | 341 | Chiefs |
| 71 | 42 | C.J. Gardner-Johnson | 66.28 | 64.20 | 66.57 | 729 | Eagles |
| 72 | 43 | Kareem Jackson | 65.80 | 60.90 | 65.49 | 1137 | Broncos |
| 73 | 44 | Justin Evans | 65.16 | 63.90 | 65.05 | 391 | Saints |
| 74 | 45 | Julian Blackmon | 63.55 | 57.40 | 68.40 | 720 | Colts |
| 75 | 46 | Eric Murray | 63.53 | 58.50 | 66.76 | 118 | Texans |
| 76 | 47 | Damar Hamlin | 63.41 | 62.80 | 64.91 | 845 | Bills |
| 77 | 48 | Xavier McKinney | 62.65 | 57.80 | 67.72 | 554 | Giants |
| 78 | 49 | Rodney Thomas II | 62.51 | 54.80 | 65.45 | 720 | Colts |
| 79 | 50 | Keanu Neal | 62.50 | 63.10 | 61.94 | 580 | Buccaneers |
| 80 | 51 | K'Von Wallace | 62.31 | 56.20 | 67.07 | 168 | Eagles |

### Rotation/backup (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 81 | 1 | DeAndre Houston-Carson | 61.92 | 59.60 | 63.27 | 414 | Bears |
| 82 | 2 | Juston Burris | 61.66 | 65.10 | 64.06 | 219 | Panthers |
| 83 | 3 | Mike Edwards | 61.59 | 57.80 | 63.63 | 814 | Buccaneers |
| 84 | 4 | Dax Hill | 61.11 | 54.40 | 63.38 | 130 | Bengals |
| 85 | 5 | Jeremy Chinn | 60.33 | 58.00 | 61.16 | 684 | Panthers |
| 86 | 6 | Jaylinn Hawkins | 60.09 | 56.10 | 62.25 | 955 | Falcons |
| 87 | 7 | Tony Adams | 59.78 | 61.80 | 65.56 | 118 | Jets |
| 88 | 8 | Tre'von Moehrig | 59.07 | 49.00 | 62.85 | 906 | Raiders |
| 89 | 9 | Alohi Gilman | 58.82 | 52.80 | 62.94 | 474 | Chargers |
| 90 | 10 | Eric Rowe | 58.30 | 49.50 | 61.46 | 567 | Dolphins |
| 91 | 11 | Roderic Teamer | 58.21 | 55.30 | 62.57 | 286 | Raiders |
| 92 | 12 | Adrian Amos | 58.15 | 45.60 | 62.35 | 977 | Packers |
| 93 | 13 | Lamarcus Joyner | 57.48 | 53.40 | 62.22 | 872 | Jets |
| 94 | 14 | Will Parks | 57.46 | 56.80 | 61.77 | 210 | Jets |
| 95 | 15 | Brandon Jones | 57.24 | 52.40 | 61.79 | 347 | Dolphins |
| 96 | 16 | Marcus Epps | 57.14 | 44.70 | 62.18 | 1095 | Eagles |
| 97 | 17 | Rayshawn Jenkins | 56.77 | 50.70 | 57.73 | 1126 | Jaguars |
| 98 | 18 | Jonathan Owens | 55.38 | 49.00 | 62.11 | 970 | Texans |
| 99 | 19 | Jaquan Johnson | 55.18 | 55.20 | 60.29 | 227 | Bills |
| 100 | 20 | Nick Scott | 55.06 | 43.40 | 60.61 | 984 | Rams |
| 101 | 21 | Darnell Savage | 54.07 | 43.80 | 57.45 | 819 | Packers |
| 102 | 22 | Nick Cross | 54.04 | 60.80 | 59.15 | 122 | Colts |
| 103 | 23 | Percy Butler | 53.93 | 48.30 | 59.40 | 134 | Commanders |
| 104 | 24 | Dean Marlowe | 53.18 | 50.70 | 56.30 | 209 | Bills |
| 105 | 25 | Johnathan Abram | 53.18 | 47.10 | 56.53 | 593 | Seahawks |
| 106 | 26 | Josh Jones | 52.82 | 53.40 | 55.95 | 376 | Seahawks |
| 107 | 27 | Ronnie Harrison | 52.67 | 45.60 | 56.71 | 259 | Browns |
| 108 | 28 | Russ Yeast | 52.05 | 47.10 | 59.03 | 113 | Rams |
| 109 | 29 | Dane Belton | 51.68 | 40.00 | 59.22 | 390 | Giants |

## T — Tackle

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Trent Williams | 97.41 | 93.00 | 96.19 | 871 | 49ers |
| 2 | 2 | Christian Darrisaw | 94.86 | 90.30 | 93.73 | 853 | Vikings |
| 3 | 3 | Andrew Thomas | 92.81 | 89.10 | 91.11 | 1049 | Giants |
| 4 | 4 | Kaleb McGary | 92.04 | 86.60 | 91.50 | 1051 | Falcons |
| 5 | 5 | Rashawn Slater | 91.27 | 84.00 | 91.95 | 175 | Chargers |
| 6 | 6 | Lane Johnson | 90.26 | 83.20 | 90.80 | 972 | Eagles |
| 7 | 7 | Brian O'Neill | 89.50 | 82.70 | 89.86 | 1052 | Vikings |
| 8 | 8 | Kolton Miller | 89.05 | 84.10 | 88.18 | 1035 | Raiders |
| 9 | 9 | Tristan Wirfs | 89.01 | 83.80 | 88.31 | 931 | Buccaneers |
| 10 | 10 | Penei Sewell | 88.38 | 80.60 | 89.40 | 1142 | Lions |
| 11 | 11 | Laremy Tunsil | 87.21 | 80.00 | 87.85 | 1061 | Texans |
| 12 | 12 | David Bakhtiari | 86.99 | 79.80 | 87.61 | 597 | Packers |
| 13 | 13 | Morgan Moses | 86.94 | 78.10 | 88.66 | 1022 | Ravens |
| 14 | 14 | Ryan Ramczyk | 85.54 | 77.90 | 86.47 | 936 | Saints |
| 15 | 15 | Terron Armstead | 85.09 | 77.50 | 85.99 | 687 | Dolphins |
| 16 | 16 | Jake Matthews | 84.88 | 77.20 | 85.83 | 1047 | Falcons |
| 17 | 17 | Jermaine Eluemunor | 84.68 | 75.30 | 86.76 | 940 | Raiders |
| 18 | 18 | Braden Smith | 84.40 | 75.50 | 86.16 | 1066 | Colts |
| 19 | 19 | Terence Steele | 84.39 | 73.90 | 87.22 | 818 | Cowboys |
| 20 | 20 | Jordan Mailata | 84.06 | 76.50 | 84.94 | 1024 | Eagles |
| 21 | 21 | Garett Bolles | 83.94 | 72.90 | 87.13 | 325 | Broncos |
| 22 | 22 | Braxton Jones | 83.93 | 75.40 | 85.45 | 1033 | Bears |
| 23 | 23 | Josh Jones | 83.72 | 75.80 | 84.83 | 622 | Cardinals |
| 24 | 24 | Orlando Brown Jr. | 83.02 | 75.80 | 83.66 | 1133 | Chiefs |
| 25 | 25 | Taylor Decker | 82.40 | 74.40 | 83.56 | 1142 | Lions |
| 26 | 26 | Bernhard Raimann | 82.35 | 73.30 | 84.22 | 709 | Colts |
| 27 | 27 | Ty Nsekhe | 81.67 | 70.60 | 84.88 | 424 | Rams |
| 28 | 28 | Rob Havenstein | 81.56 | 73.20 | 82.97 | 1019 | Rams |
| 29 | 29 | Cam Fleming | 81.42 | 72.60 | 83.13 | 976 | Broncos |
| 30 | 30 | Dion Dawkins | 81.39 | 73.50 | 82.49 | 953 | Bills |
| 31 | 31 | Mike McGlinchey | 81.30 | 71.50 | 83.66 | 1036 | 49ers |
| 32 | 32 | Patrick Mekari | 81.11 | 73.30 | 82.15 | 378 | Ravens |
| 33 | 33 | Sam Cosmi | 80.57 | 71.60 | 82.38 | 585 | Commanders |
| 34 | 34 | Tyler Smith | 80.56 | 71.40 | 82.50 | 1144 | Cowboys |
| 35 | 35 | D.J. Humphries | 80.42 | 72.30 | 81.66 | 575 | Cardinals |

### Good (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 36 | 1 | Ronnie Stanley | 79.47 | 70.90 | 81.02 | 602 | Ravens |
| 37 | 2 | Jason Peters | 79.36 | 70.30 | 81.23 | 235 | Cowboys |
| 38 | 3 | Conor McDermott | 78.97 | 65.80 | 83.58 | 437 | Patriots |
| 39 | 4 | Charles Leno Jr. | 78.89 | 71.60 | 79.59 | 1189 | Commanders |
| 40 | 5 | Kelvin Beachum | 78.86 | 70.60 | 80.20 | 1178 | Cardinals |
| 41 | 6 | Taylor Moton | 78.82 | 69.30 | 81.00 | 1018 | Panthers |
| 42 | 7 | Abraham Lucas | 78.78 | 68.50 | 81.46 | 975 | Seahawks |
| 43 | 8 | Tytus Howard | 78.27 | 67.90 | 81.02 | 997 | Texans |
| 44 | 9 | Josh Wells | 78.25 | 66.50 | 81.92 | 326 | Buccaneers |
| 45 | 10 | Zach Tom | 78.00 | 68.30 | 80.30 | 489 | Packers |
| 46 | 11 | Cornelius Lucas | 77.81 | 67.70 | 80.38 | 671 | Commanders |
| 47 | 12 | Trevor Penning | 77.58 | 73.60 | 76.07 | 124 | Saints |
| 48 | 13 | Trent Brown | 77.33 | 67.40 | 79.79 | 1030 | Patriots |
| 49 | 14 | Joe Noteboom | 77.32 | 67.00 | 80.03 | 325 | Rams |
| 50 | 15 | Cam Robinson | 77.25 | 67.20 | 79.78 | 913 | Jaguars |
| 51 | 16 | Jack Conklin | 77.23 | 66.70 | 80.09 | 913 | Browns |
| 52 | 17 | Brandon Shell | 77.17 | 64.90 | 81.18 | 761 | Dolphins |
| 53 | 18 | Ikem Ekwonu | 76.99 | 65.30 | 80.62 | 1018 | Panthers |
| 54 | 19 | Jamaree Salyer | 76.67 | 69.20 | 77.48 | 989 | Chargers |
| 55 | 20 | Larry Borom | 76.27 | 64.70 | 79.82 | 528 | Bears |
| 56 | 21 | Yodny Cajuste | 76.20 | 65.70 | 79.03 | 197 | Patriots |
| 57 | 22 | Jaylon Moore | 76.10 | 66.30 | 78.47 | 184 | 49ers |
| 58 | 23 | Charles Cross | 75.28 | 63.70 | 78.83 | 1088 | Seahawks |
| 59 | 24 | Calvin Anderson | 75.20 | 65.00 | 77.83 | 439 | Broncos |
| 60 | 25 | Riley Reiff | 75.16 | 64.30 | 78.24 | 542 | Bears |
| 61 | 26 | Thayer Munford Jr. | 75.07 | 63.20 | 78.82 | 369 | Raiders |
| 62 | 27 | Andrew Wylie | 75.06 | 63.10 | 78.86 | 1093 | Chiefs |
| 63 | 28 | Yosh Nijman | 74.35 | 63.10 | 77.69 | 756 | Packers |
| 64 | 29 | Marcus Cannon | 74.26 | 63.70 | 77.13 | 207 | Patriots |
| 65 | 30 | James Hurst | 74.15 | 63.20 | 77.29 | 973 | Saints |
| 66 | 31 | La'el Collins | 74.08 | 57.90 | 80.70 | 951 | Bengals |
| 67 | 32 | Jedrick Wills Jr. | 74.06 | 62.90 | 77.33 | 1152 | Browns |
| 68 | 33 | James Hudson III | 74.03 | 57.80 | 80.69 | 296 | Browns |

### Starter (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 69 | 1 | Dan Moore Jr. | 73.92 | 62.40 | 77.43 | 1160 | Steelers |
| 70 | 2 | Walker Little | 73.74 | 61.10 | 78.00 | 234 | Jaguars |
| 71 | 3 | Brandon Walton | 73.10 | 57.90 | 79.07 | 234 | Buccaneers |
| 72 | 4 | Chukwuma Okorafor | 73.02 | 61.20 | 76.74 | 1159 | Steelers |
| 73 | 5 | Jawaan Taylor | 72.99 | 58.70 | 78.35 | 1095 | Jaguars |
| 74 | 6 | Dennis Kelly | 72.39 | 58.90 | 77.21 | 240 | Colts |
| 75 | 7 | Trey Pipkins III | 71.74 | 59.20 | 75.93 | 898 | Chargers |
| 76 | 8 | Jonah Williams | 71.63 | 61.20 | 74.41 | 1101 | Bengals |
| 77 | 9 | Duane Brown | 71.61 | 57.80 | 76.65 | 744 | Jets |
| 78 | 10 | David Quessenberry | 71.26 | 59.30 | 75.06 | 396 | Bills |
| 79 | 11 | Donovan Smith | 71.12 | 58.10 | 75.64 | 908 | Buccaneers |
| 80 | 12 | Tyron Smith | 70.96 | 58.60 | 75.03 | 271 | Cowboys |
| 81 | 13 | Charlie Heck | 70.49 | 55.70 | 76.19 | 162 | Texans |
| 82 | 14 | Isaiah Wynn | 70.41 | 54.60 | 76.78 | 423 | Patriots |
| 83 | 15 | Blake Brandel | 70.05 | 55.30 | 75.72 | 274 | Vikings |
| 84 | 16 | Billy Turner | 70.03 | 56.30 | 75.02 | 483 | Broncos |
| 85 | 17 | Le'Raven Clark | 69.59 | 57.20 | 73.69 | 114 | Titans |
| 86 | 18 | Hakeem Adeniji | 69.09 | 54.10 | 74.92 | 220 | Bengals |
| 87 | 19 | Max Mitchell | 69.05 | 55.50 | 73.92 | 341 | Jets |
| 88 | 20 | Nicholas Petit-Frere | 68.15 | 52.30 | 74.55 | 937 | Titans |
| 89 | 21 | Matt Peart | 67.86 | 49.20 | 76.13 | 117 | Giants |
| 90 | 22 | Spencer Brown | 67.00 | 51.40 | 73.24 | 845 | Bills |
| 91 | 23 | George Fant | 66.25 | 48.40 | 73.99 | 516 | Jets |
| 92 | 24 | Cedric Ogbuehi | 65.10 | 47.70 | 72.53 | 286 | Jets |
| 93 | 25 | Daniel Faalele | 64.52 | 50.20 | 69.90 | 169 | Ravens |
| 94 | 26 | Dennis Daley | 63.07 | 46.10 | 70.21 | 942 | Titans |
| 95 | 27 | Evan Neal | 62.80 | 44.10 | 71.10 | 738 | Giants |
| 96 | 28 | Foster Sarell | 62.50 | 44.60 | 70.26 | 250 | Chargers |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 97 | 1 | Landon Young | 60.23 | 40.30 | 69.35 | 207 | Saints |
| 98 | 2 | Greg Little | 58.19 | 34.60 | 69.75 | 528 | Dolphins |
| 99 | 3 | Stone Forsythe | 51.97 | 37.00 | 57.79 | 122 | Seahawks |

## TE — Tight End

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Travis Kelce | 85.91 | 91.10 | 78.28 | 661 | Chiefs |
| 2 | 2 | Mark Andrews | 81.37 | 80.30 | 77.92 | 457 | Ravens |
| 3 | 3 | Chigoziem Okonkwo | 80.53 | 75.40 | 79.79 | 179 | Titans |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | George Kittle | 79.61 | 82.00 | 73.85 | 494 | 49ers |
| 5 | 2 | Kyle Pitts | 79.44 | 72.70 | 79.77 | 233 | Falcons |
| 6 | 3 | MyCole Pruitt | 76.59 | 75.90 | 72.89 | 126 | Falcons |
| 7 | 4 | Dallas Goedert | 76.53 | 76.20 | 72.59 | 416 | Eagles |
| 8 | 5 | Pat Freiermuth | 76.30 | 75.50 | 72.66 | 494 | Steelers |
| 9 | 6 | David Njoku | 76.09 | 73.70 | 73.52 | 477 | Browns |
| 10 | 7 | Colby Parkinson | 75.75 | 70.50 | 75.08 | 235 | Seahawks |
| 11 | 8 | Darren Waller | 74.33 | 72.40 | 71.45 | 263 | Raiders |

### Starter (57 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Will Dissly | 73.89 | 70.80 | 71.78 | 293 | Seahawks |
| 13 | 2 | Jordan Akins | 73.68 | 72.40 | 70.37 | 326 | Texans |
| 14 | 3 | Gerald Everett | 72.82 | 67.10 | 72.47 | 488 | Chargers |
| 15 | 4 | Adam Trautman | 72.33 | 66.70 | 71.91 | 202 | Saints |
| 16 | 5 | Taysom Hill | 72.05 | 75.60 | 65.52 | 127 | Saints |
| 17 | 6 | Evan Engram | 71.93 | 65.30 | 72.19 | 572 | Jaguars |
| 18 | 7 | Marcedes Lewis | 71.69 | 65.60 | 71.59 | 195 | Packers |
| 19 | 8 | Juwan Johnson | 71.62 | 64.70 | 72.06 | 403 | Saints |
| 20 | 9 | Cole Kmet | 71.46 | 67.60 | 69.86 | 505 | Bears |
| 21 | 10 | T.J. Hockenson | 71.25 | 70.20 | 67.79 | 673 | Vikings |
| 22 | 11 | Jake Ferguson | 71.00 | 65.70 | 70.37 | 152 | Cowboys |
| 23 | 12 | Foster Moreau | 70.94 | 61.10 | 73.33 | 416 | Raiders |
| 24 | 13 | Anthony Firkser | 70.85 | 59.10 | 74.51 | 107 | Falcons |
| 25 | 14 | Josh Oliver | 70.76 | 71.50 | 66.10 | 188 | Ravens |
| 26 | 15 | Noah Fant | 70.69 | 66.00 | 69.65 | 397 | Seahawks |
| 27 | 16 | Dawson Knox | 70.20 | 65.10 | 69.44 | 493 | Bills |
| 28 | 17 | Hunter Henry | 69.62 | 58.10 | 73.14 | 470 | Patriots |
| 29 | 18 | Jelani Woods | 69.33 | 64.90 | 68.12 | 225 | Colts |
| 30 | 19 | Tyler Higbee | 69.09 | 62.70 | 69.18 | 538 | Rams |
| 31 | 20 | Dalton Schultz | 69.04 | 67.80 | 65.70 | 471 | Cowboys |
| 32 | 21 | Austin Hooper | 68.79 | 68.80 | 64.61 | 352 | Titans |
| 33 | 22 | Jody Fortson | 68.51 | 65.40 | 66.41 | 114 | Chiefs |
| 34 | 23 | Parker Hesse | 68.13 | 61.20 | 68.58 | 230 | Falcons |
| 35 | 24 | Zach Ertz | 68.10 | 62.50 | 67.67 | 399 | Cardinals |
| 36 | 25 | Isaiah Likely | 68.09 | 66.40 | 65.05 | 294 | Ravens |
| 37 | 26 | Harrison Bryant | 68.00 | 59.30 | 69.63 | 292 | Browns |
| 38 | 27 | O.J. Howard | 67.99 | 56.20 | 71.69 | 161 | Texans |
| 39 | 28 | Cameron Brate | 67.57 | 53.20 | 72.99 | 257 | Buccaneers |
| 40 | 29 | Mike Gesicki | 67.56 | 59.90 | 68.50 | 391 | Dolphins |
| 41 | 30 | Jonnu Smith | 67.25 | 52.90 | 72.65 | 202 | Patriots |
| 42 | 31 | Hayden Hurst | 67.24 | 65.00 | 64.57 | 417 | Bengals |
| 43 | 32 | Chris Manhertz | 66.81 | 58.00 | 68.52 | 144 | Jaguars |
| 44 | 33 | Cade Otton | 66.57 | 56.60 | 69.05 | 519 | Buccaneers |
| 45 | 34 | Greg Dulcich | 66.55 | 60.40 | 66.49 | 346 | Broncos |
| 46 | 35 | Durham Smythe | 66.20 | 60.10 | 66.10 | 249 | Dolphins |
| 47 | 36 | C.J. Uzomah | 65.77 | 62.30 | 63.91 | 263 | Jets |
| 48 | 37 | Robert Tonyan | 65.68 | 57.70 | 66.83 | 400 | Packers |
| 49 | 38 | Teagan Quitoriano | 65.46 | 49.40 | 72.00 | 166 | Texans |
| 50 | 39 | Pharaoh Brown | 65.31 | 59.10 | 65.29 | 149 | Browns |
| 51 | 40 | Mo Alie-Cox | 65.15 | 48.20 | 72.28 | 320 | Colts |
| 52 | 41 | Daniel Bellinger | 65.12 | 61.30 | 63.50 | 318 | Giants |
| 53 | 42 | Trey McBride | 65.08 | 50.90 | 70.37 | 329 | Cardinals |
| 54 | 43 | Kylen Granson | 65.01 | 58.50 | 65.18 | 257 | Colts |
| 55 | 44 | Noah Gray | 64.60 | 63.70 | 61.04 | 330 | Chiefs |
| 56 | 45 | Mitchell Wilcox | 64.58 | 55.70 | 66.33 | 244 | Bengals |
| 57 | 46 | Tyler Conklin | 64.42 | 58.60 | 64.13 | 568 | Jets |
| 58 | 47 | John Bates | 64.35 | 57.00 | 65.09 | 197 | Commanders |
| 59 | 48 | Brock Wright | 64.25 | 54.20 | 66.79 | 253 | Lions |
| 60 | 49 | Josiah Deguara | 63.94 | 57.20 | 64.26 | 101 | Packers |
| 61 | 50 | Eric Tomlinson | 63.93 | 58.20 | 63.59 | 161 | Broncos |
| 62 | 51 | Geoff Swaim | 63.87 | 58.50 | 63.29 | 188 | Titans |
| 63 | 52 | Albert Okwuegbunam | 63.85 | 55.40 | 65.31 | 153 | Broncos |
| 64 | 53 | Irv Smith Jr. | 63.59 | 56.60 | 64.09 | 210 | Vikings |
| 65 | 54 | Tommy Tremble | 63.32 | 49.50 | 68.36 | 270 | Panthers |
| 66 | 55 | Peyton Hendershot | 63.21 | 58.10 | 62.45 | 127 | Cowboys |
| 67 | 56 | Brevin Jordan | 63.18 | 53.70 | 65.33 | 163 | Texans |
| 68 | 57 | Jack Stoll | 62.32 | 50.60 | 65.97 | 254 | Eagles |

### Rotation/backup (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 69 | 1 | Ian Thomas | 60.98 | 49.60 | 64.40 | 236 | Panthers |
| 70 | 2 | Logan Thomas | 60.66 | 52.00 | 62.26 | 415 | Commanders |
| 71 | 3 | Shane Zylstra | 60.28 | 59.20 | 56.83 | 141 | Lions |
| 72 | 4 | Johnny Mundt | 60.05 | 51.20 | 61.79 | 205 | Vikings |
| 73 | 5 | Quintin Morris | 59.45 | 43.80 | 65.71 | 159 | Bills |
| 74 | 6 | Zach Gentry | 58.87 | 47.50 | 62.29 | 256 | Steelers |
| 75 | 7 | Eric Saubert | 58.68 | 50.40 | 60.04 | 220 | Broncos |
| 76 | 8 | Nick Vannett | 57.45 | 44.30 | 62.05 | 106 | Giants |
| 77 | 9 | Tre' McKitty | 49.71 | 31.80 | 57.49 | 283 | Chargers |

## WR — Wide Receiver

- **Season used:** `2022`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Davante Adams | 88.21 | 90.10 | 82.78 | 654 | Raiders |
| 2 | 2 | Justin Jefferson | 88.09 | 90.40 | 82.38 | 729 | Vikings |
| 3 | 3 | Rashid Shaheed | 87.93 | 80.80 | 88.52 | 192 | Saints |
| 4 | 4 | Tyreek Hill | 87.76 | 92.10 | 80.70 | 555 | Dolphins |
| 5 | 5 | A.J. Brown | 87.68 | 88.00 | 83.30 | 610 | Eagles |
| 6 | 6 | Cooper Kupp | 85.59 | 86.30 | 80.95 | 347 | Rams |
| 7 | 7 | Stefon Diggs | 84.93 | 90.10 | 77.32 | 607 | Bills |
| 8 | 8 | Amon-Ra St. Brown | 84.61 | 90.70 | 76.39 | 506 | Lions |
| 9 | 9 | Chris Olave | 84.08 | 82.50 | 80.97 | 450 | Saints |
| 10 | 10 | CeeDee Lamb | 83.71 | 86.30 | 77.82 | 596 | Cowboys |
| 11 | 11 | Ja'Marr Chase | 83.65 | 83.30 | 79.71 | 553 | Bengals |
| 12 | 12 | Jaylen Waddle | 83.41 | 83.90 | 78.92 | 552 | Dolphins |
| 13 | 13 | Garrett Wilson | 82.67 | 82.70 | 78.48 | 615 | Jets |
| 14 | 14 | Terry McLaurin | 82.34 | 79.90 | 79.80 | 620 | Commanders |
| 15 | 15 | Drake London | 81.81 | 83.20 | 76.72 | 439 | Falcons |
| 16 | 16 | Mike Williams | 81.45 | 78.90 | 78.99 | 481 | Chargers |
| 17 | 17 | DeVonta Smith | 81.14 | 81.00 | 77.07 | 637 | Eagles |
| 18 | 18 | Amari Cooper | 81.12 | 81.20 | 76.90 | 602 | Browns |
| 19 | 19 | Christian Watson | 81.06 | 77.10 | 79.54 | 281 | Packers |
| 20 | 20 | Keenan Allen | 80.75 | 84.80 | 73.89 | 368 | Chargers |
| 21 | 21 | Brandon Aiyuk | 80.51 | 80.30 | 76.48 | 566 | 49ers |
| 22 | 22 | Jerry Jeudy | 80.37 | 78.40 | 77.52 | 479 | Broncos |
| 23 | 23 | Michael Thomas | 80.24 | 77.30 | 78.03 | 109 | Saints |
| 24 | 24 | Tyler Lockett | 80.20 | 78.90 | 76.90 | 560 | Seahawks |

### Good (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 25 | 1 | Tee Higgins | 79.94 | 77.10 | 77.66 | 565 | Bengals |
| 26 | 2 | Deebo Samuel | 79.46 | 72.40 | 80.00 | 347 | 49ers |
| 27 | 3 | Treylon Burks | 79.04 | 74.10 | 78.16 | 265 | Titans |
| 28 | 4 | DJ Moore | 78.36 | 73.90 | 77.17 | 539 | Panthers |
| 29 | 5 | DeAndre Hopkins | 77.98 | 72.90 | 77.20 | 388 | Cardinals |
| 30 | 6 | Tutu Atwell | 77.77 | 70.10 | 78.71 | 185 | Rams |
| 31 | 7 | D.K. Metcalf | 77.65 | 75.30 | 75.05 | 613 | Seahawks |
| 32 | 8 | Mike Evans | 77.62 | 74.00 | 75.87 | 665 | Buccaneers |
| 33 | 9 | Jakobi Meyers | 77.43 | 75.60 | 74.48 | 449 | Patriots |
| 34 | 10 | DeVante Parker | 77.22 | 74.10 | 75.13 | 339 | Patriots |
| 35 | 11 | Chris Godwin | 76.98 | 75.10 | 74.06 | 620 | Buccaneers |
| 36 | 12 | Nico Collins | 76.77 | 72.40 | 75.52 | 305 | Texans |
| 37 | 13 | Kalif Raymond | 76.77 | 72.10 | 75.72 | 342 | Lions |
| 38 | 14 | Christian Kirk | 76.46 | 74.20 | 73.80 | 648 | Jaguars |
| 39 | 15 | Isaiah Hodgins | 76.37 | 76.60 | 72.05 | 277 | Giants |
| 40 | 16 | Tyler Boyd | 76.17 | 71.10 | 75.38 | 602 | Bengals |
| 41 | 17 | Brandin Cooks | 76.13 | 72.30 | 74.51 | 449 | Texans |
| 42 | 18 | Ashton Dulin | 76.08 | 71.60 | 74.90 | 131 | Colts |
| 43 | 19 | Trenton Irwin | 76.01 | 68.00 | 77.19 | 203 | Bengals |
| 44 | 20 | Randall Cobb | 75.74 | 70.10 | 75.34 | 258 | Packers |
| 45 | 21 | Jahan Dotson | 75.30 | 70.50 | 74.33 | 403 | Commanders |
| 46 | 22 | Skyy Moore | 75.27 | 70.90 | 74.02 | 180 | Chiefs |
| 47 | 23 | DJ Chark Jr. | 75.15 | 69.60 | 74.69 | 351 | Lions |
| 48 | 24 | Darnell Mooney | 74.90 | 69.20 | 74.54 | 332 | Bears |
| 49 | 25 | Julio Jones | 74.73 | 65.10 | 76.99 | 264 | Buccaneers |
| 50 | 26 | JuJu Smith-Schuster | 74.63 | 70.80 | 73.01 | 559 | Chiefs |
| 51 | 27 | Corey Davis | 74.49 | 65.90 | 76.05 | 416 | Jets |
| 52 | 28 | Richie James | 74.38 | 71.60 | 72.07 | 367 | Giants |
| 53 | 29 | Allen Lazard | 74.33 | 69.00 | 73.72 | 517 | Packers |
| 54 | 30 | Mecole Hardman Jr. | 74.15 | 67.90 | 74.15 | 215 | Chiefs |

### Starter (83 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 55 | 1 | Terrace Marshall Jr. | 73.96 | 67.70 | 73.97 | 344 | Panthers |
| 56 | 2 | Darius Slayton | 73.90 | 68.50 | 73.33 | 430 | Giants |
| 57 | 3 | George Pickens | 73.89 | 68.80 | 73.12 | 619 | Steelers |
| 58 | 4 | Jarvis Landry | 73.80 | 67.30 | 73.97 | 221 | Saints |
| 59 | 5 | Wan'Dale Robinson | 73.67 | 72.00 | 70.62 | 140 | Giants |
| 60 | 6 | Donovan Peoples-Jones | 73.56 | 64.90 | 75.17 | 615 | Browns |
| 61 | 7 | Courtland Sutton | 73.52 | 69.30 | 72.17 | 576 | Broncos |
| 62 | 8 | Michael Pittman Jr. | 73.38 | 70.60 | 71.06 | 674 | Colts |
| 63 | 9 | Van Jefferson | 73.37 | 68.40 | 72.51 | 310 | Rams |
| 64 | 10 | Kendrick Bourne | 73.37 | 63.60 | 75.72 | 310 | Patriots |
| 65 | 11 | Marquise Brown | 73.21 | 69.20 | 71.72 | 521 | Cardinals |
| 66 | 12 | Tre'Quan Smith | 73.18 | 69.00 | 71.80 | 190 | Saints |
| 67 | 13 | Equanimeous St. Brown | 73.08 | 64.20 | 74.84 | 306 | Bears |
| 68 | 14 | Curtis Samuel | 72.98 | 70.50 | 70.46 | 528 | Commanders |
| 69 | 15 | Tom Kennedy | 72.94 | 59.30 | 77.86 | 138 | Lions |
| 70 | 16 | Khalil Shakir | 72.87 | 62.40 | 75.69 | 151 | Bills |
| 71 | 17 | Gabe Davis | 72.85 | 64.20 | 74.45 | 617 | Bills |
| 72 | 18 | Marquise Goodwin | 72.83 | 64.50 | 74.22 | 316 | Seahawks |
| 73 | 19 | Marquez Valdes-Scantling | 72.79 | 62.60 | 75.42 | 589 | Chiefs |
| 74 | 20 | Olamide Zaccheaus | 72.61 | 64.60 | 73.79 | 385 | Falcons |
| 75 | 21 | Damiere Byrd | 72.56 | 66.60 | 72.36 | 174 | Falcons |
| 76 | 22 | Hunter Renfrow | 72.55 | 65.60 | 73.02 | 307 | Raiders |
| 77 | 23 | Diontae Johnson | 72.43 | 69.60 | 70.15 | 659 | Steelers |
| 78 | 24 | Sammy Watkins | 72.22 | 62.80 | 74.34 | 216 | Ravens |
| 79 | 25 | K.J. Osborn | 72.03 | 65.10 | 72.48 | 634 | Vikings |
| 80 | 26 | Cedrick Wilson Jr. | 71.99 | 63.50 | 73.49 | 142 | Dolphins |
| 81 | 27 | Robert Woods | 71.55 | 68.00 | 69.75 | 487 | Titans |
| 82 | 28 | Trent Sherfield | 71.45 | 63.00 | 72.91 | 422 | Dolphins |
| 83 | 29 | Allen Robinson II | 71.34 | 65.80 | 70.86 | 375 | Rams |
| 84 | 30 | Josh Reynolds | 71.30 | 64.10 | 71.93 | 405 | Lions |
| 85 | 31 | Greg Dortch | 71.09 | 64.20 | 71.51 | 361 | Cardinals |
| 86 | 32 | Chase Claypool | 70.94 | 61.80 | 72.87 | 454 | Bears |
| 87 | 33 | Denzel Mims | 70.72 | 59.80 | 73.83 | 167 | Jets |
| 88 | 34 | Russell Gage | 70.72 | 67.70 | 68.57 | 390 | Buccaneers |
| 89 | 35 | Chris Moore | 70.66 | 63.20 | 71.46 | 464 | Texans |
| 90 | 36 | Jauan Jennings | 70.55 | 66.30 | 69.21 | 323 | 49ers |
| 91 | 37 | Adam Thielen | 70.54 | 65.00 | 70.07 | 717 | Vikings |
| 92 | 38 | Zay Jones | 70.51 | 67.30 | 68.49 | 607 | Jaguars |
| 93 | 39 | Byron Pringle | 70.40 | 62.90 | 71.23 | 161 | Bears |
| 94 | 40 | Devin Duvernay | 70.13 | 67.20 | 67.91 | 374 | Ravens |
| 95 | 41 | Rashod Bateman | 70.10 | 61.60 | 71.60 | 127 | Ravens |
| 96 | 42 | Rondale Moore | 70.05 | 63.40 | 70.32 | 294 | Cardinals |
| 97 | 43 | Alec Pierce | 70.03 | 61.30 | 71.69 | 501 | Colts |
| 98 | 44 | Joshua Palmer | 69.98 | 65.50 | 68.80 | 642 | Chargers |
| 99 | 45 | Amari Rodgers | 69.98 | 62.10 | 71.06 | 172 | Texans |
| 100 | 46 | Brandon Powell | 69.97 | 65.80 | 68.58 | 139 | Rams |
| 101 | 47 | Nick Westbrook-Ikhine | 69.79 | 59.80 | 72.28 | 442 | Titans |
| 102 | 48 | Mack Hollins | 69.56 | 63.80 | 69.23 | 646 | Raiders |
| 103 | 49 | Marquez Callaway | 69.25 | 59.70 | 71.45 | 197 | Saints |
| 104 | 50 | Michael Gallup | 69.20 | 62.80 | 69.30 | 444 | Cowboys |
| 105 | 51 | Ray-Ray McCloud III | 69.09 | 69.40 | 64.71 | 179 | 49ers |
| 106 | 52 | Braxton Berrios | 69.08 | 62.80 | 69.10 | 186 | Jets |
| 107 | 53 | Quez Watkins | 69.02 | 55.10 | 74.14 | 426 | Eagles |
| 108 | 54 | DeAndre Carter | 69.01 | 61.60 | 69.79 | 513 | Chargers |
| 109 | 55 | Zach Pascal | 68.95 | 62.60 | 69.01 | 164 | Eagles |
| 110 | 56 | Nelson Agholor | 68.71 | 56.40 | 72.75 | 312 | Patriots |
| 111 | 57 | Elijah Moore | 68.56 | 57.50 | 71.76 | 514 | Jets |
| 112 | 58 | KJ Hamler | 68.20 | 57.50 | 71.16 | 164 | Broncos |
| 113 | 59 | Isaiah McKenzie | 68.11 | 65.80 | 65.48 | 403 | Bills |
| 114 | 60 | Kenny Golladay | 68.00 | 55.00 | 72.50 | 162 | Giants |
| 115 | 61 | Romeo Doubs | 67.94 | 62.60 | 67.33 | 331 | Packers |
| 116 | 62 | Marvin Jones Jr. | 67.89 | 60.40 | 68.72 | 496 | Jaguars |
| 117 | 63 | Noah Brown | 67.81 | 59.50 | 69.19 | 488 | Cowboys |
| 118 | 64 | Breshad Perriman | 67.73 | 51.70 | 74.25 | 145 | Buccaneers |
| 119 | 65 | Kendall Hinton | 67.64 | 58.50 | 69.56 | 314 | Broncos |
| 120 | 66 | Sterling Shepard | 67.61 | 58.50 | 69.51 | 108 | Giants |
| 121 | 67 | Demarcus Robinson | 67.57 | 64.90 | 65.18 | 373 | Ravens |
| 122 | 68 | Parris Campbell | 67.57 | 60.80 | 67.91 | 635 | Colts |
| 123 | 69 | Phillip Dorsett | 67.26 | 55.70 | 70.80 | 306 | Texans |
| 124 | 70 | A.J. Green | 66.96 | 55.60 | 70.36 | 384 | Cardinals |
| 125 | 71 | Justin Watson | 66.32 | 52.70 | 71.23 | 315 | Chiefs |
| 126 | 72 | Shi Smith | 66.18 | 52.10 | 71.40 | 338 | Panthers |
| 127 | 73 | Cam Sims | 65.97 | 50.40 | 72.18 | 183 | Commanders |
| 128 | 74 | Scott Miller | 65.79 | 55.90 | 68.22 | 200 | Buccaneers |
| 129 | 75 | Ben Skowronek | 65.63 | 57.20 | 67.09 | 424 | Rams |
| 130 | 76 | Marcus Johnson | 65.29 | 53.80 | 68.79 | 200 | Giants |
| 131 | 77 | D'Wayne Eskridge | 64.05 | 54.10 | 66.52 | 101 | Seahawks |
| 132 | 78 | Dante Pettis | 63.89 | 53.30 | 66.79 | 327 | Bears |
| 133 | 79 | Michael Woods II | 63.61 | 53.50 | 66.18 | 103 | Browns |
| 134 | 80 | Tyquan Thornton | 63.48 | 55.20 | 64.84 | 348 | Patriots |
| 135 | 81 | Keelan Cole Sr. | 63.32 | 47.90 | 69.44 | 260 | Raiders |
| 136 | 82 | Steven Sims | 63.07 | 55.00 | 64.28 | 158 | Steelers |
| 137 | 83 | David Sills V | 62.77 | 55.90 | 63.19 | 140 | Giants |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 138 | 1 | Brandon Johnson | 61.08 | 51.80 | 63.10 | 138 | Broncos |
| 139 | 2 | David Bell | 60.90 | 52.50 | 62.33 | 327 | Browns |
| 140 | 3 | Michael Bandy | 60.84 | 53.70 | 61.43 | 151 | Chargers |
| 141 | 4 | James Proche II | 59.49 | 44.70 | 65.18 | 142 | Ravens |
