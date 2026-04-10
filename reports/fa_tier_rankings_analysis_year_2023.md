# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:37:43Z
- **Requested analysis_year:** 2023 (clamped to 2023)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Frank Ragnow | 94.49 | 88.80 | 94.11 | 1190 | Lions |
| 2 | 2 | Connor Williams | 91.90 | 81.89 | 94.41 | 497 | Dolphins |
| 3 | 3 | Creed Humphrey | 89.66 | 81.40 | 91.00 | 1380 | Chiefs |
| 4 | 4 | Drew Dalman | 89.15 | 81.69 | 89.96 | 932 | Falcons |
| 5 | 5 | Erik McCoy | 88.94 | 79.40 | 91.14 | 1152 | Saints |
| 6 | 6 | Jason Kelce | 88.42 | 78.60 | 90.80 | 1165 | Eagles |
| 7 | 7 | Tyler Linderbaum | 86.74 | 78.50 | 88.06 | 1043 | Ravens |
| 8 | 8 | Ryan Kelly | 85.53 | 76.65 | 87.29 | 882 | Colts |
| 9 | 9 | Andre James | 83.16 | 74.44 | 84.81 | 963 | Raiders |
| 10 | 10 | Lloyd Cushenberry III | 82.89 | 73.20 | 85.19 | 1070 | Broncos |
| 11 | 11 | Aaron Brewer | 81.04 | 71.60 | 83.16 | 1050 | Titans |
| 12 | 12 | David Andrews | 80.87 | 71.20 | 83.15 | 1050 | Patriots |
| 13 | 13 | Ethan Pocic | 80.00 | 70.80 | 81.96 | 1070 | Browns |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Tyler Biadasz | 78.52 | 69.20 | 80.56 | 1123 | Cowboys |
| 15 | 2 | Jake Brendel | 77.23 | 66.30 | 80.35 | 1229 | 49ers |
| 16 | 3 | Ted Karras | 77.04 | 67.40 | 79.30 | 1075 | Bengals |
| 17 | 4 | Sam Mustipher | 76.27 | 63.17 | 80.84 | 202 | Ravens |
| 18 | 5 | Hjalte Froholdt | 75.99 | 64.10 | 79.75 | 1123 | Cardinals |
| 19 | 6 | Mitch Morse | 75.00 | 63.90 | 78.23 | 1272 | Bills |
| 20 | 7 | Corey Linsley | 74.29 | 62.38 | 78.07 | 214 | Chargers |
| 21 | 8 | Joe Tippmann | 74.10 | 60.93 | 78.71 | 852 | Jets |
| 22 | 9 | Bradley Bozeman | 74.01 | 62.20 | 77.72 | 1148 | Panthers |

### Starter (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Garrett Bradbury | 73.70 | 60.85 | 78.10 | 874 | Vikings |
| 24 | 2 | Jarrett Patterson | 69.65 | 60.27 | 71.73 | 464 | Texans |
| 25 | 3 | Will Clapp | 69.23 | 57.21 | 73.08 | 702 | Chargers |
| 26 | 4 | Mason Cole | 68.76 | 53.20 | 74.96 | 1135 | Steelers |
| 27 | 5 | Tyler Larsen | 67.64 | 53.33 | 73.02 | 466 | Commanders |
| 28 | 6 | Josh Myers | 67.42 | 54.70 | 71.73 | 1212 | Packers |
| 29 | 7 | Connor McGovern | 67.31 | 51.17 | 73.91 | 371 | Jets |
| 30 | 8 | Nick Harris | 67.17 | 59.72 | 67.97 | 313 | Browns |
| 31 | 9 | Ryan Neuzil | 64.79 | 57.87 | 65.23 | 203 | Falcons |
| 32 | 10 | Robert Hainsey | 64.77 | 50.20 | 70.31 | 1236 | Buccaneers |
| 33 | 11 | Olusegun Oluwatimi | 64.43 | 57.24 | 65.06 | 128 | Seahawks |
| 34 | 12 | Wesley French | 63.29 | 53.93 | 65.37 | 270 | Colts |
| 35 | 13 | Brock Hoffman | 63.20 | 57.01 | 63.16 | 222 | Cowboys |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 36 | 1 | Luke Fortner | 61.69 | 44.30 | 69.11 | 1163 | Jaguars |
| 37 | 2 | John Michael Schmitz Jr. | 61.04 | 43.72 | 68.42 | 755 | Giants |

## CB — Cornerback

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Sauce Gardner | 93.13 | 90.80 | 91.14 | 1049 | Jets |
| 2 | 2 | Charvarius Ward | 89.19 | 86.50 | 87.60 | 1148 | 49ers |
| 3 | 3 | Jaylon Johnson | 89.15 | 90.10 | 87.98 | 809 | Bears |
| 4 | 4 | Darious Williams | 88.68 | 85.30 | 87.35 | 1035 | Jaguars |
| 5 | 5 | DaRon Bland | 88.49 | 86.40 | 87.19 | 1020 | Cowboys |
| 6 | 6 | Kendall Fuller | 86.56 | 82.80 | 86.09 | 1020 | Commanders |
| 7 | 7 | Christian Benford | 85.87 | 83.30 | 86.97 | 837 | Bills |
| 8 | 8 | Derek Stingley Jr. | 85.20 | 85.10 | 86.50 | 812 | Texans |
| 9 | 9 | Rasul Douglas | 84.63 | 81.80 | 83.34 | 1040 | Bills |
| 10 | 10 | Paulson Adebo | 84.04 | 80.50 | 84.38 | 948 | Saints |
| 11 | 11 | Devon Witherspoon | 83.18 | 79.70 | 84.26 | 883 | Seahawks |
| 12 | 12 | Taron Johnson | 83.06 | 81.00 | 80.75 | 1044 | Bills |
| 13 | 13 | Michael Carter II | 83.04 | 81.01 | 81.59 | 671 | Jets |
| 14 | 14 | D.J. Reed | 82.77 | 79.50 | 82.35 | 993 | Jets |
| 15 | 15 | Mike Hilton | 82.36 | 80.40 | 80.39 | 876 | Bengals |
| 16 | 16 | Tariq Woolen | 81.65 | 75.00 | 82.53 | 940 | Seahawks |
| 17 | 17 | Trent McDuffie | 81.14 | 81.50 | 78.93 | 1243 | Chiefs |
| 18 | 18 | Kenny Moore II | 80.26 | 78.40 | 79.30 | 1089 | Colts |
| 19 | 19 | Asante Samuel Jr. | 80.24 | 75.60 | 80.15 | 1111 | Chargers |
| 20 | 20 | Mekhi Blackmon | 80.13 | 68.56 | 85.64 | 434 | Vikings |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Jaycee Horn | 79.58 | 77.17 | 86.34 | 275 | Panthers |
| 22 | 2 | Joshua Williams | 79.53 | 70.85 | 82.62 | 420 | Chiefs |
| 23 | 3 | Steven Nelson | 79.26 | 74.00 | 79.39 | 1192 | Texans |
| 24 | 4 | Trevon Diggs | 79.11 | 69.12 | 89.16 | 101 | Cowboys |
| 25 | 5 | A.J. Terrell | 79.11 | 74.60 | 79.03 | 1065 | Falcons |
| 26 | 6 | Jaire Alexander | 79.06 | 76.63 | 83.28 | 560 | Packers |
| 27 | 7 | Jamel Dean | 79.05 | 72.50 | 81.22 | 824 | Buccaneers |
| 28 | 8 | L'Jarius Sneed | 78.92 | 73.80 | 78.56 | 1260 | Chiefs |
| 29 | 9 | Desmond King II | 78.49 | 74.41 | 80.68 | 400 | Texans |
| 30 | 10 | Jonathan Jones | 78.24 | 74.53 | 80.47 | 724 | Patriots |
| 31 | 11 | Mike Jackson | 77.89 | 72.81 | 81.02 | 474 | Seahawks |
| 32 | 12 | Nick McCloud | 77.20 | 72.73 | 79.44 | 312 | Giants |
| 33 | 13 | Isaac Yiadom | 77.19 | 76.15 | 80.87 | 517 | Saints |
| 34 | 14 | Deommodore Lenoir | 76.97 | 74.30 | 76.84 | 1198 | 49ers |
| 35 | 15 | Stephon Gilmore | 76.95 | 69.30 | 79.95 | 1055 | Cowboys |
| 36 | 16 | Denzel Ward | 75.38 | 68.79 | 78.36 | 657 | Browns |
| 37 | 17 | Roger McCreary | 75.30 | 72.20 | 74.44 | 934 | Titans |
| 38 | 18 | Dane Jackson | 75.21 | 69.12 | 77.96 | 578 | Bills |
| 39 | 19 | Pat Surtain II | 75.13 | 64.70 | 78.11 | 1121 | Broncos |
| 40 | 20 | Greg Newsome II | 75.12 | 69.79 | 77.05 | 795 | Browns |
| 41 | 21 | Tre'Davious White | 74.82 | 72.95 | 82.69 | 182 | Bills |
| 42 | 22 | Joey Porter Jr. | 74.64 | 66.60 | 75.84 | 855 | Steelers |
| 43 | 23 | Ronald Darby | 74.62 | 71.50 | 79.20 | 554 | Ravens |
| 44 | 24 | Jack Jones | 74.38 | 68.99 | 78.95 | 471 | Raiders |
| 45 | 25 | Marshon Lattimore | 74.18 | 68.38 | 80.44 | 621 | Saints |

### Starter (69 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 46 | 1 | Dee Alford | 73.87 | 70.65 | 75.76 | 571 | Falcons |
| 47 | 2 | Tyler Hall | 73.71 | 65.26 | 82.04 | 161 | Raiders |
| 48 | 3 | Darius Slay | 73.50 | 65.40 | 76.88 | 874 | Eagles |
| 49 | 4 | Jaylen Watson | 73.41 | 64.98 | 75.23 | 533 | Chiefs |
| 50 | 5 | Cam Taylor-Britt | 73.34 | 69.16 | 77.60 | 653 | Bengals |
| 51 | 6 | Isaiah Oliver | 73.00 | 68.51 | 75.85 | 503 | 49ers |
| 52 | 7 | Jalen Ramsey | 72.74 | 64.68 | 77.08 | 659 | Dolphins |
| 53 | 8 | Martin Emerson Jr. | 72.46 | 63.60 | 74.20 | 897 | Browns |
| 54 | 9 | Antonio Hamilton Sr. | 72.43 | 67.49 | 76.25 | 559 | Cardinals |
| 55 | 10 | Kyler Gordon | 71.84 | 67.26 | 74.28 | 646 | Bears |
| 56 | 11 | Donte Jackson | 71.70 | 66.60 | 74.75 | 902 | Panthers |
| 57 | 12 | Troy Hill | 71.12 | 65.96 | 73.32 | 493 | Panthers |
| 58 | 13 | Shaquill Griffin | 71.11 | 65.93 | 77.95 | 459 | Panthers |
| 59 | 14 | Chandon Sullivan | 70.90 | 64.32 | 71.12 | 442 | Steelers |
| 60 | 15 | Tyrique Stevenson | 70.59 | 59.10 | 75.06 | 830 | Bears |
| 61 | 16 | Levi Wallace | 70.33 | 60.39 | 73.38 | 762 | Steelers |
| 62 | 17 | Rock Ya-Sin | 70.33 | 63.00 | 76.53 | 281 | Ravens |
| 63 | 18 | Fabian Moreau | 70.22 | 62.74 | 73.10 | 739 | Broncos |
| 64 | 19 | Carlton Davis III | 70.03 | 63.30 | 74.36 | 847 | Buccaneers |
| 65 | 20 | Chidobe Awuzie | 69.96 | 62.21 | 75.18 | 722 | Bengals |
| 66 | 21 | Myles Bryant | 69.94 | 63.10 | 71.32 | 852 | Patriots |
| 67 | 22 | Christian Gonzalez | 69.65 | 69.71 | 83.94 | 209 | Patriots |
| 68 | 23 | Nate Hobbs | 69.62 | 67.95 | 70.68 | 775 | Raiders |
| 69 | 24 | Marlon Humphrey | 69.47 | 62.85 | 73.64 | 553 | Ravens |
| 70 | 25 | Darnay Holmes | 69.36 | 63.48 | 75.30 | 123 | Giants |
| 71 | 26 | Tavierre Thomas | 69.24 | 67.52 | 73.57 | 352 | Texans |
| 72 | 27 | Terell Smith | 69.23 | 63.65 | 74.66 | 377 | Bears |
| 73 | 28 | Kelee Ringo | 69.11 | 61.86 | 81.07 | 233 | Eagles |
| 74 | 29 | Tre Herndon | 68.91 | 70.01 | 68.43 | 482 | Jaguars |
| 75 | 30 | Ambry Thomas | 68.88 | 61.25 | 75.10 | 573 | 49ers |
| 76 | 31 | Patrick Peterson | 68.52 | 59.80 | 70.95 | 1162 | Steelers |
| 77 | 32 | JuJu Brents | 68.52 | 63.49 | 75.55 | 497 | Colts |
| 78 | 33 | Arthur Maulet | 68.48 | 63.13 | 68.76 | 458 | Ravens |
| 79 | 34 | Derion Kendrick | 67.83 | 60.40 | 70.45 | 871 | Rams |
| 80 | 35 | Tre Brown | 67.75 | 63.51 | 73.86 | 603 | Seahawks |
| 81 | 36 | Kader Kohou | 67.75 | 62.00 | 68.15 | 985 | Dolphins |
| 82 | 37 | Justin Bethel | 67.65 | 60.80 | 76.28 | 126 | Dolphins |
| 83 | 38 | Carrington Valentine | 67.45 | 59.10 | 68.85 | 846 | Packers |
| 84 | 39 | Dallis Flowers | 67.11 | 66.43 | 75.89 | 304 | Colts |
| 85 | 40 | Artie Burns | 66.96 | 64.74 | 75.54 | 232 | Seahawks |
| 86 | 41 | Tyson Campbell | 66.52 | 56.73 | 72.21 | 589 | Jaguars |
| 87 | 42 | Andrew Booth Jr. | 66.36 | 63.89 | 73.64 | 151 | Vikings |
| 88 | 43 | Alex Austin | 66.20 | 61.79 | 76.27 | 216 | Patriots |
| 89 | 44 | Grayland Arnold | 66.16 | 67.08 | 71.76 | 143 | Texans |
| 90 | 45 | Ja'Sir Taylor | 66.03 | 58.39 | 72.73 | 534 | Chargers |
| 91 | 46 | James Bradberry | 66.01 | 52.00 | 71.19 | 1090 | Eagles |
| 92 | 47 | Amik Robertson | 65.90 | 65.24 | 65.80 | 674 | Raiders |
| 93 | 48 | Cameron Mitchell | 65.87 | 60.29 | 68.36 | 283 | Browns |
| 94 | 49 | Ahkello Witherspoon | 65.85 | 60.00 | 70.99 | 1115 | Rams |
| 95 | 50 | Jaylon Jones | 65.68 | 55.90 | 70.00 | 788 | Colts |
| 96 | 51 | Ka'dar Hollman | 65.64 | 59.40 | 69.06 | 131 | Texans |
| 97 | 52 | Cobie Durant | 65.39 | 55.18 | 71.71 | 683 | Rams |
| 98 | 53 | Bryce Hall | 65.33 | 61.15 | 74.25 | 138 | Jets |
| 99 | 54 | Benjamin St-Juste | 65.12 | 56.40 | 70.69 | 1063 | Commanders |
| 100 | 55 | Michael Davis | 64.89 | 54.30 | 70.23 | 886 | Chargers |
| 101 | 56 | Kaiir Elam | 64.82 | 62.93 | 72.33 | 210 | Bills |
| 102 | 57 | Byron Murphy Jr. | 64.82 | 58.20 | 69.08 | 906 | Vikings |
| 103 | 58 | Keisean Nixon | 64.81 | 60.40 | 67.32 | 937 | Packers |
| 104 | 59 | Xavien Howard | 64.43 | 52.15 | 71.20 | 743 | Dolphins |
| 105 | 60 | Greg Stroman Jr. | 64.41 | 63.24 | 76.53 | 150 | Bears |
| 106 | 61 | Cor'Dale Flott | 64.37 | 59.52 | 67.86 | 519 | Giants |
| 107 | 62 | Bradley Roby | 64.36 | 57.58 | 69.91 | 379 | Eagles |
| 108 | 63 | Ja'Quan McMillian | 64.03 | 61.89 | 68.39 | 669 | Broncos |
| 109 | 64 | Darrell Baker Jr. | 63.45 | 55.78 | 70.28 | 469 | Colts |
| 110 | 65 | Emmanuel Forbes | 63.15 | 58.17 | 65.23 | 482 | Commanders |
| 111 | 66 | Jerry Jacobs | 62.77 | 54.78 | 68.83 | 743 | Lions |
| 112 | 67 | Deane Leonard | 62.48 | 60.00 | 72.22 | 222 | Chargers |
| 113 | 68 | Brandin Echols | 62.38 | 57.17 | 66.21 | 143 | Jets |
| 114 | 69 | DJ Turner II | 62.16 | 48.40 | 67.16 | 827 | Bengals |

### Rotation/backup (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 115 | 1 | Alontae Taylor | 61.85 | 51.80 | 66.58 | 950 | Saints |
| 116 | 2 | Garrett Williams | 61.85 | 58.15 | 68.00 | 360 | Cardinals |
| 117 | 3 | Cameron Sutton | 61.82 | 49.00 | 66.69 | 1261 | Lions |
| 118 | 4 | Essang Bassey | 61.71 | 59.15 | 65.02 | 353 | Chargers |
| 119 | 5 | Sean Murphy-Bunting | 61.68 | 54.40 | 67.77 | 840 | Titans |
| 120 | 6 | Clark Phillips III | 61.36 | 58.94 | 70.11 | 414 | Falcons |
| 121 | 7 | Deonte Banks | 60.41 | 48.60 | 66.08 | 844 | Giants |
| 122 | 8 | Dicaprio Bootle | 60.31 | 61.93 | 70.57 | 183 | Panthers |
| 123 | 9 | Corey Ballentine | 60.23 | 61.85 | 61.55 | 534 | Packers |
| 124 | 10 | Shaun Wade | 60.16 | 59.42 | 66.49 | 348 | Patriots |
| 125 | 11 | Mike Hughes | 60.11 | 51.48 | 64.35 | 333 | Falcons |
| 126 | 12 | Akayleb Evans | 59.18 | 52.60 | 64.30 | 855 | Vikings |
| 127 | 13 | Tre Flowers | 59.06 | 51.09 | 65.50 | 200 | Falcons |
| 128 | 14 | Eli Ricks | 58.53 | 49.79 | 62.16 | 316 | Eagles |
| 129 | 15 | CJ Henderson | 58.45 | 48.69 | 64.71 | 407 | Panthers |
| 130 | 16 | Zyon McCollum | 58.26 | 46.30 | 65.00 | 870 | Buccaneers |
| 131 | 17 | Adoree' Jackson | 57.97 | 46.07 | 66.06 | 792 | Giants |
| 132 | 18 | Kristian Fulton | 57.95 | 48.43 | 65.62 | 644 | Titans |
| 133 | 19 | Kei'Trel Clark | 57.85 | 55.05 | 64.36 | 464 | Cardinals |
| 134 | 20 | Jeff Okudah | 57.47 | 48.10 | 65.23 | 596 | Falcons |
| 135 | 21 | Jakorian Bennett | 57.39 | 47.50 | 64.72 | 361 | Raiders |
| 136 | 22 | Eli Apple | 57.39 | 45.56 | 64.84 | 624 | Dolphins |
| 137 | 23 | Josh Jobe | 57.03 | 50.67 | 67.15 | 240 | Eagles |
| 138 | 24 | Jourdan Lewis | 56.22 | 40.18 | 66.18 | 771 | Cowboys |
| 139 | 25 | Marco Wilson | 56.05 | 41.89 | 65.49 | 704 | Patriots |
| 140 | 26 | J.C. Jackson | 55.62 | 40.00 | 68.83 | 524 | Patriots |
| 141 | 27 | Avonte Maddox | 54.50 | 45.44 | 64.81 | 211 | Eagles |
| 142 | 28 | Damarri Mathis | 54.20 | 42.98 | 61.68 | 440 | Broncos |
| 143 | 29 | Kindle Vildor | 53.74 | 50.06 | 57.91 | 388 | Lions |
| 144 | 30 | Eric Stokes | 53.59 | 54.36 | 58.81 | 110 | Packers |
| 145 | 31 | Tre Hawkins III | 53.49 | 51.71 | 57.37 | 346 | Giants |
| 146 | 32 | D'Shawn Jamison | 53.27 | 53.40 | 60.31 | 107 | Panthers |
| 147 | 33 | Montaric Brown | 52.42 | 55.83 | 55.66 | 475 | Jaguars |
| 148 | 34 | Starling Thomas V | 51.65 | 47.96 | 55.82 | 473 | Cardinals |

## DI — Defensive Interior

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Quinnen Williams | 90.70 | 89.48 | 88.03 | 778 | Jets |
| 2 | 2 | Aaron Donald | 90.09 | 85.22 | 90.93 | 916 | Rams |
| 3 | 3 | Kobie Turner | 87.33 | 83.35 | 85.82 | 729 | Rams |
| 4 | 4 | DeForest Buckner | 86.31 | 86.73 | 81.86 | 841 | Colts |
| 5 | 5 | Christian Wilkins | 85.50 | 86.58 | 80.61 | 968 | Dolphins |
| 6 | 6 | Dexter Lawrence | 85.40 | 88.30 | 80.28 | 709 | Giants |
| 7 | 7 | Chris Jones | 84.36 | 88.33 | 78.13 | 947 | Chiefs |
| 8 | 8 | Christian Barmore | 83.75 | 80.81 | 83.61 | 750 | Patriots |
| 9 | 9 | Derrick Brown | 83.58 | 88.70 | 76.20 | 938 | Panthers |
| 10 | 10 | Leonard Williams | 81.50 | 81.45 | 78.83 | 884 | Seahawks |
| 11 | 11 | Cameron Heyward | 80.79 | 75.68 | 82.48 | 497 | Steelers |
| 12 | 12 | Jeffery Simmons | 80.74 | 80.50 | 79.76 | 657 | Titans |
| 13 | 13 | Jalen Carter | 80.54 | 84.36 | 73.82 | 599 | Eagles |
| 14 | 14 | Ed Oliver | 80.32 | 70.87 | 83.63 | 817 | Bills |
| 15 | 15 | Zach Sieler | 80.12 | 69.86 | 82.79 | 924 | Dolphins |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Kenny Clark | 79.79 | 76.71 | 77.87 | 913 | Packers |
| 17 | 2 | Jonathan Allen | 79.75 | 70.58 | 82.48 | 867 | Commanders |
| 18 | 3 | Vita Vea | 79.74 | 78.35 | 77.58 | 691 | Buccaneers |
| 19 | 4 | Javon Hargrave | 78.84 | 70.08 | 80.71 | 775 | 49ers |
| 20 | 5 | DJ Reader | 77.72 | 81.25 | 75.11 | 535 | Bengals |
| 21 | 6 | Desjuan Johnson | 77.65 | 59.06 | 90.78 | 105 | Rams |
| 22 | 7 | Milton Williams | 77.39 | 65.94 | 80.86 | 522 | Eagles |
| 23 | 8 | Grady Jarrett | 77.27 | 73.59 | 79.98 | 318 | Falcons |
| 24 | 9 | Zach Allen | 76.78 | 73.75 | 78.31 | 913 | Broncos |
| 25 | 10 | Osa Odighizuwa | 76.70 | 67.84 | 78.64 | 676 | Cowboys |
| 26 | 11 | Michael Pierce | 76.38 | 75.92 | 78.41 | 698 | Ravens |
| 27 | 12 | Alim McNeill | 76.33 | 77.59 | 71.80 | 682 | Lions |
| 28 | 13 | Shelby Harris | 76.29 | 64.91 | 80.50 | 462 | Browns |
| 29 | 14 | David Onyemata | 75.91 | 72.95 | 76.37 | 594 | Falcons |
| 30 | 15 | Daron Payne | 75.75 | 65.01 | 78.74 | 924 | Commanders |
| 31 | 16 | Jordan Davis | 75.37 | 75.29 | 72.73 | 561 | Eagles |
| 32 | 17 | Devonte Wyatt | 74.42 | 62.42 | 78.62 | 644 | Packers |
| 33 | 18 | B.J. Hill | 74.04 | 67.46 | 74.74 | 776 | Bengals |

### Starter (85 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Fletcher Cox | 73.90 | 64.09 | 76.96 | 721 | Eagles |
| 35 | 2 | Calijah Kancey | 73.79 | 53.42 | 84.18 | 663 | Buccaneers |
| 36 | 3 | Grover Stewart | 72.86 | 67.23 | 75.38 | 445 | Colts |
| 37 | 4 | Chauncey Golston | 72.53 | 60.33 | 77.23 | 322 | Cowboys |
| 38 | 5 | Dre'Mont Jones | 72.21 | 64.31 | 74.67 | 762 | Seahawks |
| 39 | 6 | Arik Armstead | 72.19 | 67.09 | 74.75 | 661 | 49ers |
| 40 | 7 | Keeanu Benton | 71.89 | 73.08 | 66.93 | 516 | Steelers |
| 41 | 8 | Travis Jones | 71.80 | 63.56 | 73.86 | 518 | Ravens |
| 42 | 9 | Karl Brooks | 71.74 | 61.81 | 74.20 | 440 | Packers |
| 43 | 10 | Dalvin Tomlinson | 71.52 | 62.44 | 74.77 | 650 | Browns |
| 44 | 11 | Mario Edwards Jr. | 70.99 | 59.04 | 77.46 | 393 | Seahawks |
| 45 | 12 | D.J. Jones | 70.71 | 55.33 | 77.88 | 568 | Broncos |
| 46 | 13 | DaQuan Jones | 70.46 | 70.91 | 70.21 | 240 | Bills |
| 47 | 14 | Folorunso Fatukasi | 70.01 | 55.81 | 77.08 | 415 | Jaguars |
| 48 | 15 | Harrison Phillips | 69.65 | 60.08 | 72.45 | 838 | Vikings |
| 49 | 16 | Morgan Fox | 69.46 | 54.01 | 75.59 | 437 | Chargers |
| 50 | 17 | Andrew Billings | 69.37 | 61.53 | 71.32 | 506 | Bears |
| 51 | 18 | A'Shawn Robinson | 69.03 | 56.20 | 75.49 | 515 | Giants |
| 52 | 19 | Sheldon Rankins | 68.89 | 55.82 | 74.22 | 673 | Texans |
| 53 | 20 | Roy Lopez | 68.73 | 57.82 | 73.50 | 395 | Cardinals |
| 54 | 21 | Armon Watts | 68.68 | 61.15 | 70.01 | 286 | Steelers |
| 55 | 22 | Dante Stills | 68.51 | 55.74 | 74.83 | 533 | Cardinals |
| 56 | 23 | Larry Ogunjobi | 68.51 | 49.57 | 77.45 | 816 | Steelers |
| 57 | 24 | William Gholston | 68.34 | 52.03 | 75.04 | 286 | Buccaneers |
| 58 | 25 | Jarran Reed | 68.28 | 53.24 | 74.63 | 809 | Seahawks |
| 59 | 26 | Bryan Bresee | 68.10 | 51.62 | 74.92 | 539 | Saints |
| 60 | 27 | Sebastian Joseph-Day | 68.07 | 50.57 | 77.82 | 623 | 49ers |
| 61 | 28 | Khyiris Tonga | 68.05 | 61.68 | 72.25 | 188 | Vikings |
| 62 | 29 | Roy Robertson-Harris | 67.97 | 54.37 | 73.45 | 683 | Jaguars |
| 63 | 30 | Tim Settle | 67.95 | 53.71 | 74.06 | 413 | Bills |
| 64 | 31 | Cameron Young | 67.80 | 56.77 | 72.96 | 201 | Seahawks |
| 65 | 32 | Poona Ford | 67.76 | 57.53 | 74.83 | 151 | Bills |
| 66 | 33 | Naquan Jones | 67.70 | 53.29 | 78.14 | 171 | Cardinals |
| 67 | 34 | Zacch Pickens | 67.56 | 53.68 | 72.64 | 264 | Bears |
| 68 | 35 | Adam Gotsis | 67.14 | 52.79 | 73.33 | 427 | Jaguars |
| 69 | 36 | Kevin Givens | 67.05 | 51.17 | 75.44 | 454 | 49ers |
| 70 | 37 | Bobby Brown III | 66.99 | 64.34 | 70.38 | 335 | Rams |
| 71 | 38 | Maurice Hurst | 66.99 | 59.16 | 75.46 | 302 | Browns |
| 72 | 39 | Shy Tuttle | 66.87 | 54.58 | 70.90 | 547 | Panthers |
| 73 | 40 | Teair Tart | 66.84 | 55.97 | 73.36 | 378 | Texans |
| 74 | 41 | Bilal Nichols | 66.79 | 51.96 | 72.51 | 616 | Raiders |
| 75 | 42 | Maliek Collins | 66.47 | 56.30 | 70.06 | 780 | Texans |
| 76 | 43 | Solomon Thomas | 66.19 | 48.83 | 73.60 | 483 | Jets |
| 77 | 44 | Colby Wooden | 66.06 | 55.36 | 69.02 | 298 | Packers |
| 78 | 45 | Da'Shawn Hand | 66.03 | 66.00 | 69.64 | 219 | Dolphins |
| 79 | 46 | Kentavius Street | 66.02 | 52.06 | 73.13 | 267 | Falcons |
| 80 | 47 | DaVon Hamilton | 66.00 | 54.34 | 74.20 | 190 | Jaguars |
| 81 | 48 | Gervon Dexter Sr. | 65.99 | 52.08 | 71.09 | 433 | Bears |
| 82 | 49 | Khalil Davis | 65.84 | 48.82 | 78.48 | 481 | Texans |
| 83 | 50 | Khalen Saunders | 65.79 | 55.38 | 70.81 | 522 | Saints |
| 84 | 51 | Brent Urban | 65.62 | 52.82 | 72.44 | 309 | Ravens |
| 85 | 52 | Malcolm Roach | 65.60 | 58.80 | 71.55 | 290 | Saints |
| 86 | 53 | Quinton Jefferson | 65.47 | 48.50 | 74.09 | 468 | Jets |
| 87 | 54 | Taven Bryan | 65.43 | 54.95 | 68.93 | 343 | Colts |
| 88 | 55 | Lawrence Guy Sr. | 65.36 | 42.89 | 77.06 | 522 | Patriots |
| 89 | 56 | Davon Godchaux | 65.30 | 51.09 | 70.61 | 685 | Patriots |
| 90 | 57 | Adam Butler | 65.14 | 49.76 | 71.43 | 526 | Raiders |
| 91 | 58 | Tyler Lacy | 65.09 | 56.12 | 68.87 | 145 | Jaguars |
| 92 | 59 | Neville Gallimore | 65.07 | 53.15 | 71.50 | 304 | Cowboys |
| 93 | 60 | Nathan Shepherd | 64.98 | 50.95 | 70.17 | 593 | Saints |
| 94 | 61 | DeMarvin Leal | 64.70 | 51.56 | 74.20 | 206 | Steelers |
| 95 | 62 | Justin Jones | 64.68 | 47.75 | 72.98 | 740 | Bears |
| 96 | 63 | Jerry Tillery | 64.61 | 58.12 | 65.56 | 504 | Raiders |
| 97 | 64 | Levi Onwuzurike | 64.57 | 56.27 | 68.75 | 164 | Lions |
| 98 | 65 | Greg Gaines | 64.47 | 52.57 | 68.53 | 525 | Buccaneers |
| 99 | 66 | Mike Purcell | 64.16 | 49.12 | 71.28 | 463 | Broncos |
| 100 | 67 | Mike Pennel | 64.01 | 52.92 | 73.50 | 160 | Chiefs |
| 101 | 68 | Marlon Tuipulotu | 63.98 | 55.34 | 71.26 | 178 | Eagles |
| 102 | 69 | Jonathan Bullard | 63.52 | 48.10 | 72.66 | 643 | Vikings |
| 103 | 70 | DeShawn Williams | 63.51 | 48.42 | 70.28 | 443 | Panthers |
| 104 | 71 | Matt Henningsen | 63.32 | 53.60 | 65.64 | 226 | Broncos |
| 105 | 72 | Javon Kinlaw | 63.27 | 52.49 | 72.08 | 544 | 49ers |
| 106 | 73 | Abdullah Anderson | 62.99 | 59.06 | 69.38 | 113 | Commanders |
| 107 | 74 | John Jenkins | 62.98 | 47.38 | 71.46 | 595 | Raiders |
| 108 | 75 | LaCale London | 62.95 | 58.88 | 73.51 | 204 | Falcons |
| 109 | 76 | Jonah Williams | 62.93 | 50.00 | 70.32 | 636 | Rams |
| 110 | 77 | Broderick Washington | 62.87 | 51.04 | 67.18 | 452 | Ravens |
| 111 | 78 | Al Woods | 62.81 | 49.43 | 74.03 | 140 | Jets |
| 112 | 79 | Montravius Adams | 62.59 | 51.98 | 68.34 | 427 | Steelers |
| 113 | 80 | Jonathan Harris | 62.57 | 49.32 | 72.64 | 529 | Broncos |
| 114 | 81 | Johnathan Hankins | 62.52 | 46.30 | 72.80 | 387 | Cowboys |
| 115 | 82 | John Cominsky | 62.47 | 45.25 | 70.95 | 663 | Lions |
| 116 | 83 | Jordan Phillips | 62.47 | 47.60 | 72.73 | 391 | Bills |
| 117 | 84 | Linval Joseph | 62.46 | 45.63 | 76.66 | 189 | Bills |
| 118 | 85 | Austin Johnson | 62.42 | 48.27 | 70.34 | 641 | Chargers |

### Rotation/backup (44 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 119 | 1 | Dean Lowry | 61.83 | 49.57 | 70.35 | 237 | Vikings |
| 120 | 2 | Nick Thurman | 61.72 | 49.46 | 71.61 | 368 | Panthers |
| 121 | 3 | Kurt Hinish | 61.63 | 49.20 | 66.49 | 542 | Texans |
| 122 | 4 | Mazi Smith | 61.24 | 52.26 | 63.06 | 308 | Cowboys |
| 123 | 5 | Raekwon Davis | 61.16 | 50.78 | 64.80 | 531 | Dolphins |
| 124 | 6 | Ta'Quon Graham | 61.14 | 56.36 | 63.69 | 364 | Falcons |
| 125 | 7 | Tershawn Wharton | 61.12 | 51.17 | 67.12 | 525 | Chiefs |
| 126 | 8 | Sheldon Day | 61.08 | 55.21 | 69.16 | 125 | Vikings |
| 127 | 9 | Angelo Blackson | 61.03 | 50.40 | 67.49 | 215 | Jaguars |
| 128 | 10 | Leki Fotu | 61.02 | 50.79 | 66.60 | 297 | Cardinals |
| 129 | 11 | John Ridgeway | 61.00 | 49.69 | 65.11 | 355 | Commanders |
| 130 | 12 | Isaiahh Loudermilk | 60.94 | 54.43 | 64.05 | 184 | Steelers |
| 131 | 13 | Jonathan Ledbetter | 60.46 | 48.12 | 70.98 | 511 | Cardinals |
| 132 | 14 | Larrell Murchison | 60.37 | 53.76 | 64.92 | 261 | Rams |
| 133 | 15 | Derrick Nnadi | 60.27 | 46.92 | 65.01 | 507 | Chiefs |
| 134 | 16 | Josh Tupou | 60.26 | 51.21 | 65.36 | 287 | Bengals |
| 135 | 17 | Logan Hall | 60.11 | 49.79 | 62.83 | 600 | Buccaneers |
| 136 | 18 | D.J. Davidson | 59.35 | 54.28 | 64.20 | 244 | Giants |
| 137 | 19 | Ben Stille | 59.32 | 55.48 | 68.74 | 134 | Cardinals |
| 138 | 20 | Rakeem Nunez-Roches | 59.09 | 45.15 | 64.90 | 461 | Giants |
| 139 | 21 | Jeremiah Ledbetter | 58.94 | 55.86 | 64.24 | 369 | Jaguars |
| 140 | 22 | Jaleel Johnson | 58.74 | 49.78 | 66.33 | 270 | Titans |
| 141 | 23 | Otito Ogbonnia | 58.73 | 52.48 | 67.92 | 223 | Chargers |
| 142 | 24 | Jordan Elliott | 58.70 | 49.15 | 61.10 | 466 | Browns |
| 143 | 25 | Quinton Bohanna | 58.67 | 55.61 | 63.69 | 113 | Titans |
| 144 | 26 | Kyle Peko | 58.64 | 47.70 | 68.14 | 342 | Titans |
| 145 | 27 | Adetomiwa Adebawore | 58.42 | 54.52 | 64.71 | 132 | Colts |
| 146 | 28 | Matt Dickerson | 58.22 | 51.30 | 65.11 | 206 | Chiefs |
| 147 | 29 | Zach Carter | 57.75 | 47.61 | 60.71 | 500 | Bengals |
| 148 | 30 | Benito Jones | 57.23 | 45.70 | 62.84 | 602 | Lions |
| 149 | 31 | Marlon Davidson | 56.67 | 56.78 | 61.74 | 163 | Titans |
| 150 | 32 | Eric Johnson | 55.97 | 50.60 | 58.32 | 265 | Colts |
| 151 | 33 | Tyson Alualu | 55.86 | 42.96 | 69.11 | 152 | Lions |
| 152 | 34 | Jordon Riley | 55.44 | 53.27 | 61.54 | 135 | Giants |
| 153 | 35 | Phil Hoskins | 55.07 | 54.14 | 64.02 | 124 | Cardinals |
| 154 | 36 | LaBryan Ray | 54.49 | 47.16 | 55.21 | 356 | Panthers |
| 155 | 37 | Sam Roberts | 53.97 | 53.59 | 61.32 | 101 | Patriots |
| 156 | 38 | Scott Matlock | 53.80 | 49.53 | 57.38 | 266 | Chargers |
| 157 | 39 | Albert Huggins | 49.93 | 46.08 | 55.77 | 317 | Falcons |
| 158 | 40 | TK McLendon Jr. | 49.84 | 53.31 | 54.66 | 101 | Titans |
| 159 | 41 | Mike Greene | 48.45 | 50.05 | 50.09 | 168 | Buccaneers |
| 160 | 42 | Phidarian Mathis | 47.94 | 53.17 | 50.45 | 202 | Commanders |
| 161 | 43 | Keondre Coburn | 45.72 | 58.35 | 49.40 | 107 | Titans |
| 162 | 44 | Siaki Ika | 45.00 | 51.77 | 46.54 | 103 | Browns |

## ED — Edge

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Bosa | 94.26 | 97.15 | 88.47 | 1022 | 49ers |
| 2 | 2 | Micah Parsons | 94.06 | 92.27 | 91.08 | 910 | Cowboys |
| 3 | 3 | Myles Garrett | 91.76 | 95.24 | 85.57 | 844 | Browns |
| 4 | 4 | T.J. Watt | 91.53 | 93.98 | 88.18 | 930 | Steelers |
| 5 | 5 | Aidan Hutchinson | 88.28 | 93.26 | 80.79 | 1146 | Lions |
| 6 | 6 | Will Anderson Jr. | 87.72 | 91.99 | 80.71 | 695 | Texans |
| 7 | 7 | Rashan Gary | 87.45 | 88.41 | 85.20 | 667 | Packers |
| 8 | 8 | Maxx Crosby | 87.36 | 92.25 | 79.94 | 1080 | Raiders |
| 9 | 9 | Greg Rousseau | 86.75 | 89.99 | 81.61 | 660 | Bills |
| 10 | 10 | Khalil Mack | 86.72 | 86.47 | 84.68 | 934 | Chargers |
| 11 | 11 | Danielle Hunter | 84.38 | 81.88 | 83.85 | 1004 | Vikings |
| 12 | 12 | Joey Bosa | 84.17 | 85.36 | 86.86 | 320 | Chargers |
| 13 | 13 | Trey Hendrickson | 83.21 | 76.43 | 84.35 | 742 | Bengals |
| 14 | 14 | DeMarcus Lawrence | 82.08 | 86.08 | 77.21 | 647 | Cowboys |
| 15 | 15 | Montez Sweat | 81.91 | 81.47 | 79.41 | 764 | Bears |
| 16 | 16 | Jaelan Phillips | 81.38 | 81.89 | 81.29 | 366 | Dolphins |
| 17 | 17 | Bradley Chubb | 80.31 | 78.28 | 80.24 | 837 | Dolphins |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Sam Williams | 78.92 | 68.38 | 82.52 | 314 | Cowboys |
| 19 | 2 | Tuli Tuipulotu | 78.85 | 77.93 | 75.30 | 852 | Chargers |
| 20 | 3 | Von Miller | 78.56 | 64.42 | 87.46 | 298 | Bills |
| 21 | 4 | Malcolm Koonce | 78.20 | 68.02 | 84.55 | 501 | Raiders |
| 22 | 5 | Alex Highsmith | 78.04 | 78.35 | 73.87 | 974 | Steelers |
| 23 | 6 | Haason Reddick | 78.00 | 65.38 | 82.44 | 910 | Eagles |
| 24 | 7 | Brandon Graham | 77.68 | 71.74 | 80.41 | 427 | Eagles |
| 25 | 8 | Jermaine Johnson | 77.63 | 73.85 | 77.09 | 748 | Jets |
| 26 | 9 | Shaquil Barrett | 77.37 | 70.34 | 80.93 | 746 | Buccaneers |
| 27 | 10 | Will McDonald IV | 77.34 | 63.39 | 84.44 | 183 | Jets |
| 28 | 11 | Za'Darius Smith | 76.38 | 73.15 | 77.80 | 603 | Browns |
| 29 | 12 | Josh Sweat | 76.31 | 72.00 | 75.50 | 875 | Eagles |
| 30 | 13 | Brian Burns | 75.84 | 64.85 | 79.79 | 814 | Panthers |
| 31 | 14 | Jadeveon Clowney | 74.81 | 77.77 | 70.73 | 747 | Ravens |
| 32 | 15 | George Karlaftis | 74.80 | 63.65 | 78.07 | 973 | Chiefs |
| 33 | 16 | Cameron Jordan | 74.35 | 70.21 | 73.42 | 770 | Saints |
| 34 | 17 | BJ Ojulari | 74.18 | 59.96 | 79.49 | 409 | Cardinals |

### Starter (58 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Jonathan Greenard | 73.52 | 68.11 | 76.60 | 697 | Texans |
| 36 | 2 | Boye Mafe | 73.41 | 68.84 | 72.90 | 808 | Seahawks |
| 37 | 3 | Arnold Ebiketie | 72.98 | 63.36 | 75.59 | 385 | Falcons |
| 38 | 4 | Matthew Judon | 72.73 | 59.54 | 83.73 | 184 | Patriots |
| 39 | 5 | Odafe Oweh | 72.58 | 73.33 | 69.28 | 491 | Ravens |
| 40 | 6 | Harold Landry III | 72.27 | 61.60 | 75.22 | 840 | Titans |
| 41 | 7 | Carl Granderson | 72.02 | 65.47 | 72.91 | 874 | Saints |
| 42 | 8 | Chase Young | 71.65 | 79.55 | 67.90 | 891 | 49ers |
| 43 | 9 | Dennis Gardeck | 71.40 | 54.88 | 79.71 | 510 | Cardinals |
| 44 | 10 | Markus Golden | 71.27 | 55.14 | 78.05 | 281 | Steelers |
| 45 | 11 | Lukas Van Ness | 71.24 | 61.50 | 73.56 | 444 | Packers |
| 46 | 12 | Ogbo Okoronkwo | 71.13 | 59.69 | 76.35 | 459 | Browns |
| 47 | 13 | Samson Ebukam | 71.04 | 65.76 | 70.98 | 703 | Colts |
| 48 | 14 | A.J. Epenesa | 70.67 | 61.78 | 73.61 | 438 | Bills |
| 49 | 15 | Calais Campbell | 70.64 | 47.69 | 81.77 | 712 | Falcons |
| 50 | 16 | Yaya Diaby | 70.31 | 60.88 | 72.43 | 590 | Buccaneers |
| 51 | 17 | Travon Walker | 70.20 | 64.80 | 70.37 | 869 | Jaguars |
| 52 | 18 | Jonathon Cooper | 70.04 | 64.26 | 70.81 | 836 | Broncos |
| 53 | 19 | Preston Smith | 70.03 | 58.72 | 73.61 | 838 | Packers |
| 54 | 20 | Dante Fowler Jr. | 70.03 | 59.68 | 73.34 | 275 | Cowboys |
| 55 | 21 | Nolan Smith | 69.75 | 60.40 | 71.82 | 203 | Eagles |
| 56 | 22 | John Franklin-Myers | 69.63 | 62.36 | 70.51 | 626 | Jets |
| 57 | 23 | Julian Okwara | 69.59 | 60.24 | 78.43 | 120 | Lions |
| 58 | 24 | Marcus Davenport | 69.55 | 67.71 | 74.74 | 118 | Vikings |
| 59 | 25 | Dorance Armstrong | 69.39 | 61.97 | 70.96 | 468 | Cowboys |
| 60 | 26 | Darrell Taylor | 69.05 | 55.77 | 74.22 | 522 | Seahawks |
| 61 | 27 | Tyquan Lewis | 68.49 | 62.80 | 72.71 | 437 | Colts |
| 62 | 28 | Kayvon Thibodeaux | 68.35 | 64.85 | 67.62 | 981 | Giants |
| 63 | 29 | Dayo Odeyingbo | 68.31 | 62.27 | 69.54 | 623 | Colts |
| 64 | 30 | Leonard Floyd | 68.27 | 55.49 | 72.63 | 627 | Bills |
| 65 | 31 | Jacob Martin | 68.17 | 59.11 | 71.23 | 192 | Colts |
| 66 | 32 | Sam Hubbard | 67.79 | 59.46 | 70.94 | 713 | Bengals |
| 67 | 33 | Jerry Hughes | 67.53 | 52.18 | 73.59 | 474 | Texans |
| 68 | 34 | Kwity Paye | 67.51 | 66.61 | 66.30 | 700 | Colts |
| 69 | 35 | Zack Baun | 67.39 | 56.87 | 70.23 | 303 | Saints |
| 70 | 36 | Denico Autry | 67.34 | 47.58 | 77.82 | 767 | Titans |
| 71 | 37 | Victor Dimukeje | 67.28 | 59.58 | 72.16 | 385 | Cardinals |
| 72 | 38 | Azeez Ojulari | 67.03 | 59.81 | 75.02 | 424 | Giants |
| 73 | 39 | Byron Young | 66.87 | 57.97 | 68.64 | 1021 | Rams |
| 74 | 40 | Joe Tryon-Shoyinka | 66.58 | 61.00 | 66.14 | 625 | Buccaneers |
| 75 | 41 | Carl Lawson | 66.56 | 56.79 | 74.30 | 101 | Jets |
| 76 | 42 | Deatrich Wise Jr. | 66.26 | 59.20 | 67.49 | 615 | Patriots |
| 77 | 43 | Charles Omenihu | 66.23 | 56.80 | 70.22 | 502 | Chiefs |
| 78 | 44 | Zach Harrison | 66.01 | 60.51 | 66.50 | 343 | Falcons |
| 79 | 45 | Myles Murphy | 65.89 | 59.39 | 66.05 | 304 | Bengals |
| 80 | 46 | Arden Key | 65.88 | 62.89 | 63.71 | 727 | Titans |
| 81 | 47 | Yannick Ngakoue | 65.52 | 53.13 | 72.16 | 592 | Bears |
| 82 | 48 | Kingsley Enagbare | 65.50 | 60.53 | 64.65 | 493 | Packers |
| 83 | 49 | Randy Gregory | 65.15 | 55.00 | 71.96 | 488 | 49ers |
| 84 | 50 | Melvin Ingram III | 64.68 | 52.69 | 75.28 | 150 | Dolphins |
| 85 | 51 | Clelin Ferrell | 64.45 | 60.27 | 63.55 | 471 | 49ers |
| 86 | 52 | Shaq Lawson | 64.31 | 55.46 | 67.05 | 354 | Bills |
| 87 | 53 | Lorenzo Carter | 64.11 | 57.35 | 65.03 | 431 | Falcons |
| 88 | 54 | Emmanuel Ogbah | 63.97 | 57.57 | 67.40 | 286 | Dolphins |
| 89 | 55 | Derek Barnett | 63.61 | 60.90 | 66.65 | 415 | Texans |
| 90 | 56 | Amare Barno | 62.92 | 59.71 | 66.89 | 189 | Panthers |
| 91 | 57 | Bud Dupree | 62.45 | 53.19 | 67.89 | 725 | Falcons |
| 92 | 58 | Mike Danna | 62.14 | 58.96 | 61.28 | 932 | Chiefs |

### Rotation/backup (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 93 | 1 | Micheal Clemons | 61.78 | 59.82 | 59.90 | 368 | Jets |
| 94 | 2 | Anthony Nelson | 61.78 | 57.73 | 60.31 | 446 | Buccaneers |
| 95 | 3 | D.J. Wonnum | 61.77 | 56.77 | 62.21 | 826 | Vikings |
| 96 | 4 | Jesse Luketa | 61.65 | 57.87 | 64.91 | 132 | Cardinals |
| 97 | 5 | DeMarcus Walker | 61.40 | 51.58 | 63.78 | 714 | Bears |
| 98 | 6 | Myjai Sanders | 61.21 | 59.09 | 64.83 | 189 | Texans |
| 99 | 7 | Tyree Wilson | 61.08 | 57.88 | 59.04 | 493 | Raiders |
| 100 | 8 | Cam Sample | 60.92 | 60.83 | 57.89 | 375 | Bengals |
| 101 | 9 | Romeo Okwara | 60.53 | 57.30 | 64.60 | 330 | Lions |
| 102 | 10 | Dawuane Smoot | 60.43 | 55.65 | 62.68 | 340 | Jaguars |
| 103 | 11 | Marquis Haynes Sr. | 60.24 | 57.15 | 63.03 | 142 | Panthers |
| 104 | 12 | Yetur Gross-Matos | 60.23 | 59.58 | 59.53 | 465 | Panthers |
| 105 | 13 | K'Lavon Chaisson | 60.19 | 58.60 | 59.84 | 283 | Jaguars |
| 106 | 14 | Drake Jackson | 59.70 | 59.35 | 62.01 | 199 | 49ers |
| 107 | 15 | Justin Hollins | 59.47 | 56.39 | 62.36 | 197 | Chargers |
| 108 | 16 | Rasheem Green | 59.46 | 54.61 | 58.83 | 385 | Bears |
| 109 | 17 | Felix Anudike-Uzomah | 59.32 | 58.61 | 55.62 | 225 | Chiefs |
| 110 | 18 | Charles Harris | 59.09 | 56.35 | 61.95 | 291 | Lions |
| 111 | 19 | Tanoh Kpassagnon | 58.73 | 55.91 | 58.79 | 406 | Saints |
| 112 | 20 | Keion White | 58.28 | 59.43 | 54.33 | 522 | Patriots |
| 113 | 21 | Alex Wright | 58.07 | 57.44 | 54.32 | 407 | Browns |
| 114 | 22 | Tavius Robinson | 57.67 | 56.34 | 54.39 | 338 | Ravens |
| 115 | 23 | Casey Toohill | 57.46 | 53.88 | 56.67 | 494 | Commanders |
| 116 | 24 | Derick Hall | 57.31 | 56.02 | 54.00 | 308 | Seahawks |
| 117 | 25 | Josh Paschal | 56.85 | 58.66 | 55.27 | 510 | Lions |
| 118 | 26 | Rashad Weaver | 56.27 | 56.45 | 56.20 | 240 | Titans |
| 119 | 27 | Jihad Ward | 55.91 | 45.77 | 58.81 | 661 | Giants |
| 120 | 28 | Chris Rumph II | 55.69 | 58.09 | 56.29 | 103 | Chargers |
| 121 | 29 | James Smith-Williams | 55.34 | 54.97 | 54.86 | 418 | Commanders |
| 122 | 30 | KJ Henry | 54.11 | 57.32 | 55.66 | 281 | Commanders |
| 123 | 31 | Malik Herring | 54.11 | 54.96 | 55.26 | 213 | Chiefs |
| 124 | 32 | Dominique Robinson | 53.40 | 54.32 | 52.31 | 242 | Bears |
| 125 | 33 | Dylan Horton | 53.00 | 57.28 | 52.85 | 175 | Texans |
| 126 | 34 | DJ Johnson | 52.98 | 56.45 | 50.42 | 231 | Panthers |
| 127 | 35 | Ronnie Perkins | 52.57 | 55.94 | 52.55 | 149 | Broncos |
| 128 | 36 | Andre Jones Jr. | 50.22 | 56.68 | 49.59 | 171 | Commanders |
| 129 | 37 | Jeremiah Moon | 48.79 | 56.13 | 49.47 | 102 | Ravens |

## G — Guard

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Lindstrom | 93.44 | 87.60 | 93.16 | 1066 | Falcons |
| 2 | 2 | Quinn Meinerz | 91.16 | 83.70 | 91.97 | 1038 | Broncos |
| 3 | 3 | Kevin Dotson | 90.73 | 83.82 | 91.17 | 939 | Rams |
| 4 | 4 | Sam Cosmi | 87.61 | 80.60 | 88.11 | 1103 | Commanders |
| 5 | 5 | David Edwards | 86.41 | 73.40 | 90.91 | 194 | Bills |
| 6 | 6 | Tyler Smith | 85.29 | 74.40 | 88.38 | 1037 | Cowboys |
| 7 | 7 | Robert Hunt | 84.15 | 74.29 | 86.55 | 608 | Dolphins |
| 8 | 8 | Graham Glasgow | 83.65 | 74.90 | 85.32 | 1262 | Lions |
| 9 | 9 | Trey Smith | 83.11 | 74.60 | 84.61 | 1374 | Chiefs |
| 10 | 10 | Greg Van Roten | 81.98 | 75.30 | 82.26 | 1025 | Raiders |
| 11 | 11 | Isaac Seumalo | 81.87 | 73.90 | 83.02 | 1104 | Steelers |
| 12 | 12 | Wyatt Teller | 81.58 | 72.70 | 83.33 | 1254 | Browns |
| 13 | 13 | Joe Thuney | 81.49 | 74.90 | 81.72 | 1212 | Chiefs |
| 14 | 14 | Teven Jenkins | 80.58 | 71.55 | 82.43 | 731 | Bears |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Quenton Nelson | 79.17 | 70.80 | 80.59 | 1141 | Colts |
| 16 | 2 | Kevin Zeitler | 79.16 | 71.60 | 80.03 | 1101 | Ravens |
| 17 | 3 | Landon Dickerson | 78.50 | 69.40 | 80.40 | 1102 | Eagles |
| 18 | 4 | Brandon Scherff | 77.24 | 67.30 | 79.70 | 1079 | Jaguars |
| 19 | 5 | Zack Martin | 76.97 | 68.00 | 78.78 | 1003 | Cowboys |
| 20 | 6 | Shaq Mason | 76.09 | 65.60 | 78.91 | 1221 | Texans |
| 21 | 7 | Joel Bitonio | 75.82 | 67.90 | 76.94 | 1107 | Browns |
| 22 | 8 | Halapoulivaati Vaitai | 75.78 | 63.66 | 79.70 | 192 | Lions |
| 23 | 9 | Alex Cappa | 75.69 | 64.90 | 78.72 | 1066 | Bengals |
| 24 | 10 | Sidy Sow | 75.54 | 63.90 | 79.14 | 772 | Patriots |
| 25 | 11 | Ben Cleveland | 75.23 | 62.42 | 79.60 | 171 | Ravens |
| 26 | 12 | Will Hernandez | 75.17 | 66.20 | 76.99 | 1109 | Cardinals |
| 27 | 13 | Nate Herbig | 74.53 | 63.04 | 78.03 | 156 | Steelers |
| 28 | 14 | Cole Strange | 74.34 | 63.48 | 77.41 | 564 | Patriots |

### Starter (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 29 | 1 | Elgton Jenkins | 73.72 | 63.80 | 76.16 | 1019 | Packers |
| 30 | 2 | Mark Glowinski | 73.70 | 62.76 | 76.82 | 521 | Giants |
| 31 | 3 | Nick Allegretti | 73.10 | 62.94 | 75.71 | 253 | Chiefs |
| 32 | 4 | Cade Mays | 73.02 | 58.87 | 78.28 | 434 | Panthers |
| 33 | 5 | Will Fries | 72.54 | 61.20 | 75.94 | 1125 | Colts |
| 34 | 6 | Ben Powers | 72.32 | 61.50 | 75.36 | 1068 | Broncos |
| 35 | 7 | Ben Bartch | 72.05 | 60.59 | 75.52 | 240 | 49ers |
| 36 | 8 | Jonah Jackson | 72.02 | 60.95 | 75.23 | 881 | Lions |
| 37 | 9 | James Daniels | 72.01 | 61.10 | 75.11 | 1010 | Steelers |
| 38 | 10 | Dylan Parham | 71.58 | 60.40 | 74.86 | 1042 | Raiders |
| 39 | 11 | Cordell Volson | 71.27 | 58.30 | 75.75 | 1087 | Bengals |
| 40 | 12 | Ezra Cleveland | 71.24 | 59.54 | 74.88 | 749 | Jaguars |
| 41 | 13 | Damien Lewis | 71.14 | 59.61 | 74.66 | 926 | Seahawks |
| 42 | 14 | Steve Avila | 71.12 | 60.50 | 74.04 | 1205 | Rams |
| 43 | 15 | Robert Jones | 70.96 | 58.33 | 75.22 | 479 | Dolphins |
| 44 | 16 | Ed Ingram | 70.83 | 59.51 | 74.21 | 958 | Vikings |
| 45 | 17 | Matthew Bergeron | 70.75 | 59.10 | 74.35 | 1127 | Falcons |
| 46 | 18 | Gabe Jackson | 70.46 | 58.73 | 74.12 | 194 | Panthers |
| 47 | 19 | Zion Johnson | 69.71 | 57.60 | 73.62 | 1006 | Chargers |
| 48 | 20 | John Simpson | 69.52 | 56.30 | 74.16 | 1242 | Ravens |
| 49 | 21 | Aaron Stinnie | 69.25 | 56.75 | 73.41 | 851 | Buccaneers |
| 50 | 22 | Max Garcia | 68.85 | 55.70 | 73.45 | 320 | Saints |
| 51 | 23 | Kayode Awosika | 68.81 | 57.36 | 72.28 | 370 | Lions |
| 52 | 24 | Jon Runyan | 68.79 | 56.50 | 72.81 | 1009 | Packers |
| 53 | 25 | Matt Feiler | 68.50 | 56.68 | 72.22 | 386 | Buccaneers |
| 54 | 26 | O'Cyrus Torrence | 68.01 | 54.90 | 72.59 | 1307 | Bills |
| 55 | 27 | Dalton Risner | 68.00 | 57.48 | 70.84 | 745 | Vikings |
| 56 | 28 | Connor McGovern | 67.91 | 55.40 | 72.09 | 1278 | Bills |
| 57 | 29 | Saahdiq Charles | 67.66 | 56.36 | 71.03 | 643 | Commanders |
| 58 | 30 | Phil Haynes | 67.43 | 54.60 | 71.81 | 437 | Seahawks |
| 59 | 31 | Anthony Bradford | 67.21 | 53.21 | 72.37 | 659 | Seahawks |
| 60 | 32 | Wes Schweitzer | 67.15 | 57.08 | 69.70 | 149 | Jets |
| 61 | 33 | Spencer Burford | 67.08 | 50.73 | 73.81 | 900 | 49ers |
| 62 | 34 | Jake Hanson | 66.92 | 55.92 | 70.08 | 244 | Jets |
| 63 | 35 | Jordan McFadden | 66.77 | 56.75 | 69.28 | 163 | Chargers |
| 64 | 36 | Nate Davis | 66.65 | 54.17 | 70.80 | 663 | Bears |
| 65 | 37 | Xavier Newman | 66.62 | 53.36 | 71.30 | 292 | Jets |
| 66 | 38 | Aaron Banks | 66.62 | 52.80 | 71.67 | 1042 | 49ers |
| 67 | 39 | Laken Tomlinson | 66.16 | 55.00 | 69.44 | 1099 | Jets |
| 68 | 40 | Sua Opeta | 65.84 | 54.79 | 69.04 | 530 | Eagles |
| 69 | 41 | Elijah Wilkinson | 65.26 | 50.16 | 71.16 | 501 | Cardinals |
| 70 | 42 | Cesar Ruiz | 64.88 | 51.20 | 69.83 | 1050 | Saints |
| 71 | 43 | Tyler Shatley | 64.71 | 50.94 | 69.72 | 518 | Jaguars |
| 72 | 44 | Austin Corbett | 64.71 | 53.82 | 67.81 | 257 | Panthers |
| 73 | 45 | Marcus McKethan | 64.60 | 50.96 | 69.52 | 378 | Giants |
| 74 | 46 | Ja'Tyre Carter | 64.50 | 55.36 | 66.42 | 175 | Bears |
| 75 | 47 | Lester Cotton | 63.66 | 49.24 | 69.10 | 616 | Dolphins |
| 76 | 48 | Royce Newman | 63.59 | 51.70 | 67.35 | 186 | Packers |
| 77 | 49 | Dennis Daley | 62.91 | 51.36 | 66.45 | 144 | Cardinals |
| 78 | 50 | Cody Whitehair | 62.45 | 46.59 | 68.86 | 787 | Bears |

### Rotation/backup (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 79 | 1 | Chris Paul | 60.81 | 45.85 | 66.61 | 439 | Commanders |
| 80 | 2 | Nash Jensen | 60.35 | 45.99 | 65.75 | 302 | Panthers |
| 81 | 3 | Ben Bredeson | 59.97 | 42.50 | 67.45 | 1014 | Giants |
| 82 | 4 | Justin Pugh | 59.97 | 42.92 | 67.17 | 763 | Giants |
| 83 | 5 | Atonio Mafi | 58.23 | 41.11 | 65.47 | 458 | Patriots |
| 84 | 6 | Chandler Zavala | 57.10 | 40.00 | 64.33 | 374 | Panthers |

## HB — Running Back

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | De'Von Achane | 89.56 | 82.52 | 90.08 | 187 | Dolphins |
| 2 | 2 | Breece Hall | 84.27 | 81.06 | 82.24 | 336 | Jets |
| 3 | 3 | Christian McCaffrey | 83.54 | 90.30 | 74.87 | 525 | 49ers |
| 4 | 4 | Jaylen Warren | 82.97 | 75.68 | 83.66 | 273 | Steelers |
| 5 | 5 | Derrick Henry | 82.33 | 84.30 | 76.85 | 175 | Titans |
| 6 | 6 | Tony Pollard | 81.15 | 77.50 | 79.42 | 439 | Cowboys |
| 7 | 7 | Tyjae Spears | 80.25 | 75.18 | 79.47 | 303 | Titans |
| 8 | 8 | Jahmyr Gibbs | 80.17 | 76.13 | 78.69 | 372 | Lions |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Aaron Jones | 79.92 | 77.50 | 77.36 | 202 | Packers |
| 10 | 2 | Raheem Mostert | 79.76 | 82.51 | 73.76 | 276 | Dolphins |
| 11 | 3 | James Conner | 79.35 | 84.42 | 71.81 | 201 | Cardinals |
| 12 | 4 | Kenneth Walker III | 78.75 | 79.63 | 74.00 | 200 | Seahawks |
| 13 | 5 | Kyren Williams | 78.59 | 79.32 | 73.93 | 341 | Rams |
| 14 | 6 | Travis Etienne Jr. | 77.82 | 77.00 | 74.20 | 404 | Jaguars |
| 15 | 7 | Jonathan Taylor | 77.48 | 72.08 | 76.92 | 178 | Colts |
| 16 | 8 | Tyler Allgeier | 77.45 | 76.48 | 73.93 | 108 | Falcons |
| 17 | 9 | Rhamondre Stevenson | 76.75 | 70.03 | 77.07 | 251 | Patriots |
| 18 | 10 | Isiah Pacheco | 76.39 | 79.50 | 70.15 | 365 | Chiefs |
| 19 | 11 | Bijan Robinson | 76.38 | 69.10 | 77.06 | 421 | Falcons |
| 20 | 12 | Khalil Herbert | 76.20 | 71.10 | 75.44 | 146 | Bears |
| 21 | 13 | Devin Singletary | 76.14 | 74.47 | 73.08 | 329 | Texans |
| 22 | 14 | Emari Demercado | 76.08 | 68.80 | 76.77 | 164 | Cardinals |
| 23 | 15 | Alvin Kamara | 76.03 | 72.76 | 74.04 | 269 | Saints |
| 24 | 16 | David Montgomery | 74.77 | 76.56 | 69.41 | 209 | Lions |
| 25 | 17 | James Cook | 74.40 | 71.12 | 72.42 | 343 | Bills |
| 26 | 18 | Najee Harris | 74.40 | 73.93 | 70.55 | 223 | Steelers |

### Starter (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | AJ Dillon | 73.97 | 73.38 | 70.20 | 216 | Packers |
| 28 | 2 | Gus Edwards | 73.57 | 69.43 | 72.17 | 180 | Ravens |
| 29 | 3 | Saquon Barkley | 72.98 | 69.57 | 71.09 | 306 | Giants |
| 30 | 4 | Kareem Hunt | 72.86 | 67.63 | 72.18 | 151 | Browns |
| 31 | 5 | Josh Jacobs | 72.02 | 64.37 | 72.96 | 238 | Raiders |
| 32 | 6 | Chuba Hubbard | 71.98 | 75.15 | 65.70 | 295 | Panthers |
| 33 | 7 | Javonte Williams | 71.93 | 62.64 | 73.95 | 170 | Broncos |
| 34 | 8 | Austin Ekeler | 71.66 | 60.20 | 75.13 | 349 | Chargers |
| 35 | 9 | Zach Charbonnet | 71.38 | 68.35 | 69.23 | 276 | Seahawks |
| 36 | 10 | Brian Robinson | 71.31 | 71.62 | 66.94 | 219 | Commanders |
| 37 | 11 | Justice Hill | 71.10 | 68.79 | 68.47 | 277 | Ravens |
| 38 | 12 | Joe Mixon | 71.04 | 71.04 | 66.87 | 373 | Bengals |
| 39 | 13 | Samaje Perine | 70.92 | 68.56 | 68.33 | 218 | Broncos |
| 40 | 14 | D'Andre Swift | 70.82 | 66.07 | 69.82 | 301 | Eagles |
| 41 | 15 | Jerome Ford | 70.56 | 67.95 | 68.13 | 349 | Browns |
| 42 | 16 | Ezekiel Elliott | 70.55 | 66.72 | 68.94 | 247 | Patriots |
| 43 | 17 | Antonio Gibson | 70.48 | 66.26 | 69.13 | 318 | Commanders |
| 44 | 18 | Miles Sanders | 70.46 | 60.68 | 72.81 | 211 | Panthers |
| 45 | 19 | Zack Moss | 70.01 | 65.40 | 68.92 | 233 | Colts |
| 46 | 20 | Michael Carter | 70.00 | 60.50 | 72.17 | 151 | Cardinals |
| 47 | 21 | Ty Chandler | 68.97 | 69.24 | 64.63 | 122 | Vikings |
| 48 | 22 | Clyde Edwards-Helaire | 68.89 | 64.31 | 67.78 | 155 | Chiefs |
| 49 | 23 | Dameon Pierce | 68.82 | 64.27 | 67.68 | 111 | Texans |
| 50 | 24 | Alexander Mattison | 68.68 | 61.54 | 69.27 | 278 | Vikings |
| 51 | 25 | Rachaad White | 68.39 | 67.50 | 64.82 | 504 | Buccaneers |
| 52 | 26 | Roschon Johnson | 67.77 | 64.10 | 66.05 | 190 | Bears |
| 53 | 27 | D'Ernest Johnson | 67.75 | 58.67 | 69.63 | 108 | Jaguars |
| 54 | 28 | Rico Dowdle | 67.63 | 63.74 | 66.05 | 122 | Cowboys |
| 55 | 29 | Latavius Murray | 66.96 | 64.92 | 64.16 | 198 | Bills |
| 56 | 30 | Jerick McKinnon | 65.95 | 62.56 | 64.04 | 190 | Chiefs |
| 57 | 31 | Kenneth Gainwell | 65.71 | 56.33 | 67.80 | 232 | Eagles |
| 58 | 32 | Patrick Taylor | 64.98 | 60.24 | 63.98 | 136 | Packers |
| 59 | 33 | Jamaal Williams | 64.04 | 59.66 | 62.79 | 120 | Saints |
| 60 | 34 | Matt Breida | 63.54 | 55.13 | 64.98 | 156 | Giants |
| 61 | 35 | Ameer Abdullah | 62.58 | 57.24 | 61.97 | 143 | Raiders |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 62 | 1 | Joshua Kelley | 60.24 | 54.60 | 59.84 | 173 | Chargers |

## LB — Linebacker

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Demario Davis | 85.41 | 89.60 | 78.65 | 1074 | Saints |
| 2 | 2 | Fred Warner | 85.11 | 90.00 | 77.89 | 1200 | 49ers |
| 3 | 3 | Jahlani Tavai | 82.78 | 86.60 | 77.64 | 838 | Patriots |
| 4 | 4 | Tyrel Dodson | 81.04 | 85.60 | 79.14 | 589 | Bills |
| 5 | 5 | C.J. Mosley | 80.87 | 82.90 | 75.55 | 1127 | Jets |
| 6 | 6 | Bobby Wagner | 80.48 | 82.40 | 75.23 | 1170 | Seahawks |

### Good (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Quincy Williams | 79.79 | 81.10 | 75.53 | 1092 | Jets |
| 8 | 2 | Bobby Okereke | 79.63 | 79.00 | 75.88 | 1128 | Giants |
| 9 | 3 | Roquan Smith | 79.15 | 77.90 | 75.81 | 1192 | Ravens |
| 10 | 4 | T.J. Edwards | 78.42 | 79.60 | 73.66 | 1042 | Bears |
| 11 | 5 | Leo Chenal | 78.42 | 79.24 | 73.71 | 527 | Chiefs |
| 12 | 6 | Devin Lloyd | 77.51 | 78.10 | 74.18 | 966 | Jaguars |
| 13 | 7 | Blake Cashman | 77.18 | 81.08 | 75.42 | 746 | Texans |
| 14 | 8 | Ernest Jones | 75.87 | 78.80 | 71.42 | 988 | Rams |
| 15 | 9 | Kaden Elliss | 75.60 | 75.40 | 71.57 | 1082 | Falcons |
| 16 | 10 | Robert Spillane | 75.51 | 77.10 | 71.55 | 1100 | Raiders |
| 17 | 11 | Foyesade Oluokun | 75.41 | 75.20 | 71.39 | 1110 | Jaguars |
| 18 | 12 | Frankie Luvu | 75.33 | 80.00 | 68.84 | 989 | Panthers |
| 19 | 13 | Mack Wilson Sr. | 75.03 | 73.11 | 74.90 | 305 | Patriots |
| 20 | 14 | Ivan Pace Jr. | 74.59 | 75.84 | 69.59 | 704 | Vikings |
| 21 | 15 | Luke Masterson | 74.24 | 72.67 | 75.17 | 182 | Raiders |

### Starter (52 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Lavonte David | 73.90 | 72.30 | 71.79 | 1086 | Buccaneers |
| 23 | 2 | Jalen Reeves-Maybin | 73.86 | 71.44 | 76.61 | 121 | Lions |
| 24 | 3 | Jordan Hicks | 73.73 | 74.56 | 70.97 | 813 | Vikings |
| 25 | 4 | Jeremiah Owusu-Koramoah | 73.58 | 76.69 | 69.69 | 802 | Browns |
| 26 | 5 | Dre Greenlaw | 73.06 | 73.10 | 72.20 | 999 | 49ers |
| 27 | 6 | Elandon Roberts | 72.94 | 70.63 | 70.31 | 622 | Steelers |
| 28 | 7 | Patrick Queen | 72.65 | 73.00 | 68.25 | 1246 | Ravens |
| 29 | 8 | Eric Kendricks | 72.44 | 72.30 | 69.73 | 847 | Chargers |
| 30 | 9 | Nicholas Morrow | 70.35 | 68.10 | 68.17 | 898 | Eagles |
| 31 | 10 | Nate Landman | 69.99 | 71.92 | 69.93 | 809 | Falcons |
| 32 | 11 | Alex Anzalone | 69.91 | 69.80 | 66.40 | 1189 | Lions |
| 33 | 12 | Sione Takitaki | 69.78 | 69.21 | 68.35 | 608 | Browns |
| 34 | 13 | Isaiah Simmons | 69.51 | 66.04 | 67.66 | 378 | Giants |
| 35 | 14 | Malik Harrison | 69.04 | 67.51 | 68.74 | 226 | Ravens |
| 36 | 15 | Jamin Davis | 68.82 | 67.23 | 68.16 | 742 | Commanders |
| 37 | 16 | Duke Riley | 68.77 | 68.73 | 65.52 | 473 | Dolphins |
| 38 | 17 | Drue Tranquill | 68.75 | 67.88 | 65.74 | 721 | Chiefs |
| 39 | 18 | Jack Gibbens | 67.94 | 69.80 | 68.78 | 628 | Titans |
| 40 | 19 | Jerome Baker | 67.68 | 66.33 | 66.57 | 713 | Dolphins |
| 41 | 20 | E.J. Speed | 67.66 | 64.72 | 66.54 | 730 | Colts |
| 42 | 21 | Azeez Al-Shaair | 67.40 | 64.70 | 67.29 | 1101 | Titans |
| 43 | 22 | Ja'Whaun Bentley | 67.40 | 65.80 | 64.99 | 984 | Patriots |
| 44 | 23 | Zach Cunningham | 67.25 | 67.45 | 68.25 | 787 | Eagles |
| 45 | 24 | Jack Sanborn | 67.17 | 65.17 | 67.65 | 412 | Bears |
| 46 | 25 | Josey Jewell | 66.75 | 67.09 | 66.95 | 796 | Broncos |
| 47 | 26 | Damone Clark | 66.52 | 62.70 | 67.46 | 834 | Cowboys |
| 48 | 27 | Anthony Walker Jr. | 66.40 | 68.26 | 68.34 | 454 | Browns |
| 49 | 28 | David Long Jr. | 66.40 | 62.50 | 67.68 | 899 | Dolphins |
| 50 | 29 | Matt Milano | 66.22 | 67.68 | 67.75 | 211 | Bills |
| 51 | 30 | Logan Wilson | 66.20 | 62.60 | 65.80 | 1068 | Bengals |
| 52 | 31 | Germaine Pratt | 66.16 | 63.30 | 64.88 | 975 | Bengals |
| 53 | 32 | Leighton Vander Esch | 65.99 | 64.09 | 69.86 | 269 | Cowboys |
| 54 | 33 | De'Vondre Campbell | 65.94 | 64.85 | 65.83 | 690 | Packers |
| 55 | 34 | Micah McFadden | 65.70 | 65.31 | 63.14 | 736 | Giants |
| 56 | 35 | Christian Harris | 65.51 | 65.00 | 63.51 | 869 | Texans |
| 57 | 36 | Alex Singleton | 64.99 | 61.20 | 64.14 | 1089 | Broncos |
| 58 | 37 | K.J. Britt | 64.86 | 65.82 | 69.65 | 252 | Buccaneers |
| 59 | 38 | Deion Jones | 64.78 | 64.86 | 64.47 | 313 | Panthers |
| 60 | 39 | Nick Niemann | 64.47 | 66.37 | 69.34 | 247 | Chargers |
| 61 | 40 | Quay Walker | 64.46 | 58.50 | 64.88 | 973 | Packers |
| 62 | 41 | Zaire Franklin | 64.43 | 60.90 | 63.10 | 1090 | Colts |
| 63 | 42 | Terrel Bernard | 64.41 | 65.90 | 63.30 | 1031 | Bills |
| 64 | 43 | Cole Holcomb | 64.37 | 64.64 | 67.57 | 447 | Steelers |
| 65 | 44 | Derrick Barnes | 64.18 | 60.69 | 63.71 | 791 | Lions |
| 66 | 45 | Andre Smith | 63.96 | 66.84 | 69.89 | 113 | Falcons |
| 67 | 46 | Nick Bolton | 63.74 | 59.43 | 65.09 | 708 | Chiefs |
| 68 | 47 | Oren Burks | 63.60 | 59.20 | 64.43 | 433 | 49ers |
| 69 | 48 | Kyzir White | 63.26 | 58.98 | 64.88 | 708 | Cardinals |
| 70 | 49 | Khaleke Hudson | 62.87 | 63.23 | 66.49 | 405 | Commanders |
| 71 | 50 | Divine Deablo | 62.85 | 60.68 | 65.53 | 771 | Raiders |
| 72 | 51 | Pete Werner | 62.46 | 57.50 | 64.15 | 919 | Saints |
| 73 | 52 | Nakobe Dean | 62.35 | 60.80 | 69.52 | 182 | Eagles |

### Rotation/backup (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 74 | 1 | Jack Campbell | 61.79 | 52.47 | 63.84 | 745 | Lions |
| 75 | 2 | Malcolm Rodriguez | 61.68 | 58.84 | 61.60 | 162 | Lions |
| 76 | 3 | Denzel Perryman | 61.65 | 56.66 | 64.14 | 633 | Texans |
| 77 | 4 | Willie Gay | 61.64 | 56.22 | 63.23 | 698 | Chiefs |
| 78 | 5 | Kwon Alexander | 61.27 | 59.47 | 63.21 | 362 | Steelers |
| 79 | 6 | Devin Bush | 61.22 | 58.62 | 63.30 | 251 | Seahawks |
| 80 | 7 | Tremaine Edmunds | 61.00 | 56.60 | 62.31 | 876 | Bears |
| 81 | 8 | Jordyn Brooks | 60.88 | 57.33 | 59.86 | 802 | Seahawks |
| 82 | 9 | SirVocea Dennis | 60.70 | 61.88 | 62.62 | 104 | Buccaneers |
| 83 | 10 | Cody Barton | 60.68 | 53.90 | 64.57 | 844 | Commanders |
| 84 | 11 | Krys Barnes | 60.63 | 60.07 | 63.70 | 408 | Cardinals |
| 85 | 12 | Dorian Williams | 60.31 | 56.17 | 61.83 | 238 | Bills |
| 86 | 13 | Isaiah McDuffie | 59.97 | 57.62 | 63.55 | 551 | Packers |
| 87 | 14 | Tony Fields II | 59.44 | 54.78 | 60.58 | 253 | Browns |
| 88 | 15 | Eric Wilson | 59.31 | 58.53 | 64.29 | 144 | Packers |
| 89 | 16 | Troy Dye | 58.91 | 60.28 | 64.23 | 112 | Vikings |
| 90 | 17 | David Mayo | 58.91 | 56.60 | 63.64 | 349 | Commanders |
| 91 | 18 | Myles Jack | 58.25 | 56.89 | 62.34 | 157 | Steelers |
| 92 | 19 | Kenneth Murray Jr. | 57.89 | 52.10 | 59.73 | 968 | Chargers |
| 93 | 20 | Devin White | 57.71 | 47.40 | 60.90 | 933 | Buccaneers |
| 94 | 21 | Mykal Walker | 57.48 | 55.18 | 59.26 | 321 | Steelers |
| 95 | 22 | Mark Robinson | 57.18 | 55.91 | 63.91 | 173 | Steelers |
| 96 | 23 | Kamu Grugier-Hill | 56.91 | 52.01 | 58.86 | 403 | Panthers |
| 97 | 24 | Owen Pappoe | 56.90 | 62.12 | 63.03 | 114 | Cardinals |
| 98 | 25 | Demetrius Flannigan-Fowles | 56.72 | 56.59 | 60.61 | 174 | 49ers |
| 99 | 26 | Jack Cochrane | 56.52 | 57.17 | 62.09 | 183 | Chiefs |
| 100 | 27 | Segun Olubi | 56.19 | 58.48 | 62.34 | 115 | Colts |
| 101 | 28 | Henry To'oTo'o | 54.89 | 48.78 | 57.73 | 459 | Texans |
| 102 | 29 | Troy Reeder | 54.53 | 53.04 | 57.93 | 194 | Rams |
| 103 | 30 | Christian Rozeboom | 54.43 | 49.83 | 57.15 | 579 | Rams |
| 104 | 31 | Troy Andersen | 54.01 | 56.13 | 57.99 | 139 | Falcons |
| 105 | 32 | Chad Muma | 53.97 | 50.76 | 56.72 | 146 | Jaguars |
| 106 | 33 | Drew Sanders | 53.27 | 47.16 | 55.15 | 260 | Broncos |
| 107 | 34 | Josh Woods | 45.69 | 40.00 | 54.91 | 568 | Cardinals |

## QB — Quarterback

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Tua Tagovailoa | 82.64 | 85.67 | 77.75 | 683 | Dolphins |
| 2 | 2 | Dak Prescott | 82.29 | 83.64 | 78.16 | 784 | Cowboys |
| 3 | 3 | Josh Allen | 81.83 | 86.40 | 73.55 | 775 | Bills |
| 4 | 4 | Patrick Mahomes | 80.49 | 85.21 | 72.08 | 900 | Chiefs |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Brock Purdy | 79.35 | 80.73 | 81.97 | 647 | 49ers |
| 6 | 2 | Matthew Stafford | 79.11 | 82.34 | 74.85 | 643 | Rams |
| 7 | 3 | Jared Goff | 78.52 | 80.38 | 72.89 | 794 | Lions |
| 8 | 4 | Lamar Jackson | 78.47 | 79.72 | 75.52 | 675 | Ravens |
| 9 | 5 | Jalen Hurts | 78.01 | 82.31 | 71.06 | 693 | Eagles |
| 10 | 6 | Geno Smith | 77.17 | 79.97 | 73.53 | 591 | Seahawks |
| 11 | 7 | Justin Herbert | 76.33 | 81.18 | 70.13 | 552 | Chargers |
| 12 | 8 | Kirk Cousins | 75.92 | 78.81 | 74.87 | 353 | Vikings |
| 13 | 9 | Joe Burrow | 75.41 | 80.80 | 71.05 | 422 | Bengals |
| 14 | 10 | Trevor Lawrence | 74.71 | 75.14 | 69.89 | 681 | Jaguars |
| 15 | 11 | Jordan Love | 74.71 | 82.34 | 73.93 | 725 | Packers |
| 16 | 12 | Derek Carr | 74.35 | 74.26 | 71.66 | 609 | Saints |
| 17 | 13 | C.J. Stroud | 74.01 | 80.40 | 76.65 | 653 | Texans |

### Starter (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Russell Wilson | 72.33 | 70.04 | 72.84 | 568 | Broncos |
| 19 | 2 | Baker Mayfield | 71.84 | 70.28 | 70.83 | 777 | Buccaneers |
| 20 | 3 | Kyler Murray | 65.44 | 67.91 | 66.84 | 326 | Cardinals |
| 21 | 4 | Justin Fields | 64.47 | 63.15 | 66.71 | 478 | Bears |
| 22 | 5 | Ryan Tannehill | 64.38 | 68.63 | 65.27 | 291 | Titans |
| 23 | 6 | Jake Browning | 63.87 | 70.14 | 75.45 | 298 | Bengals |
| 24 | 7 | Kenny Pickett | 63.50 | 68.97 | 62.53 | 386 | Steelers |
| 25 | 8 | Tyrod Taylor | 62.79 | 68.57 | 72.70 | 242 | Giants |
| 26 | 9 | Mason Rudolph | 62.66 | 62.31 | 77.93 | 130 | Steelers |
| 27 | 10 | Jimmy Garoppolo | 62.38 | 65.97 | 65.93 | 203 | Raiders |

### Rotation/backup (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Nick Mullens | 61.81 | 60.20 | 76.23 | 172 | Vikings |
| 29 | 2 | Joe Flacco | 61.50 | 65.27 | 67.73 | 277 | Browns |
| 30 | 3 | Will Levis | 60.83 | 61.16 | 67.84 | 310 | Titans |
| 31 | 4 | Aidan O'Connell | 60.72 | 63.74 | 63.06 | 380 | Raiders |
| 32 | 5 | Gardner Minshew | 60.59 | 60.36 | 63.95 | 580 | Colts |
| 33 | 6 | Sam Howell | 60.40 | 57.65 | 63.07 | 750 | Commanders |
| 34 | 7 | Mac Jones | 60.36 | 60.30 | 62.03 | 395 | Patriots |
| 35 | 8 | Tommy DeVito | 59.73 | 62.86 | 63.54 | 243 | Giants |
| 36 | 9 | Desmond Ridder | 59.14 | 54.64 | 65.63 | 464 | Falcons |
| 37 | 10 | Deshaun Watson | 58.85 | 61.83 | 62.26 | 218 | Browns |
| 38 | 11 | Easton Stick | 58.82 | 60.97 | 63.53 | 208 | Chargers |
| 39 | 12 | Anthony Richardson | 58.63 | 58.54 | 64.74 | 107 | Colts |
| 40 | 13 | Joshua Dobbs | 58.30 | 59.77 | 59.37 | 524 | Vikings |
| 41 | 14 | Daniel Jones | 58.13 | 63.01 | 58.80 | 220 | Giants |
| 42 | 15 | Bryce Young | 57.91 | 53.00 | 56.36 | 663 | Panthers |
| 43 | 16 | Bailey Zappe | 56.99 | 56.14 | 58.85 | 253 | Patriots |
| 44 | 17 | Mitch Trubisky | 56.90 | 57.91 | 58.25 | 132 | Steelers |
| 45 | 18 | Tyson Bagent | 56.09 | 55.10 | 57.31 | 166 | Bears |
| 46 | 19 | Zach Wilson | 55.95 | 55.14 | 59.01 | 463 | Jets |
| 47 | 20 | P.J. Walker | 55.11 | 49.64 | 57.83 | 135 | Browns |
| 48 | 21 | Dorian Thompson-Robinson | 54.87 | 55.03 | 52.42 | 132 | Browns |
| 49 | 22 | Trevor Siemian | 54.40 | 49.65 | 54.93 | 177 | Jets |
| 50 | 23 | Taylor Heinicke | 54.03 | 55.34 | 60.04 | 161 | Falcons |

## S — Safety

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jessie Bates III | 92.67 | 90.20 | 90.83 | 1134 | Falcons |
| 2 | 2 | Xavier McKinney | 91.52 | 91.20 | 89.91 | 1128 | Giants |
| 3 | 3 | Jevon Holland | 90.40 | 88.57 | 90.10 | 707 | Dolphins |
| 4 | 4 | Kyle Hamilton | 88.58 | 90.10 | 83.77 | 1065 | Ravens |
| 5 | 5 | Antoine Winfield Jr. | 87.71 | 84.00 | 87.98 | 1230 | Buccaneers |
| 6 | 6 | Alohi Gilman | 87.22 | 89.20 | 84.39 | 928 | Chargers |
| 7 | 7 | Tyrann Mathieu | 86.96 | 87.40 | 82.70 | 1096 | Saints |
| 8 | 8 | Jabrill Peppers | 81.90 | 83.20 | 80.00 | 955 | Patriots |
| 9 | 9 | Xavier Woods | 81.89 | 81.46 | 80.07 | 795 | Panthers |
| 10 | 10 | Geno Stone | 80.44 | 85.30 | 75.28 | 1000 | Ravens |
| 11 | 11 | Julian Love | 80.39 | 80.40 | 76.52 | 937 | Seahawks |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Jordan Battle | 79.99 | 73.07 | 80.43 | 524 | Bengals |
| 13 | 2 | Jalen Thompson | 78.21 | 77.00 | 75.84 | 938 | Cardinals |
| 14 | 3 | Juanyeh Thomas | 77.74 | 72.49 | 78.90 | 192 | Cowboys |
| 15 | 4 | Reed Blankenship | 77.50 | 78.00 | 77.54 | 942 | Eagles |
| 16 | 5 | Malik Hooker | 77.44 | 71.70 | 77.78 | 862 | Cowboys |
| 17 | 6 | Marcus Williams | 76.09 | 76.04 | 76.17 | 765 | Ravens |
| 18 | 7 | Kevin Byard | 76.06 | 68.00 | 77.27 | 1187 | Eagles |
| 19 | 8 | Andrew Wingard | 75.50 | 71.77 | 76.57 | 330 | Jaguars |
| 20 | 9 | Rudy Ford | 74.96 | 71.24 | 76.61 | 626 | Packers |
| 21 | 10 | Julian Blackmon | 74.76 | 72.60 | 76.05 | 987 | Colts |
| 22 | 11 | Darnell Savage | 74.36 | 73.53 | 73.49 | 701 | Packers |
| 23 | 12 | Jordan Poyer | 74.33 | 69.50 | 75.05 | 1103 | Bills |
| 24 | 13 | Grant Delpit | 74.23 | 74.66 | 72.13 | 738 | Browns |
| 25 | 14 | Camryn Bynum | 74.13 | 69.70 | 72.92 | 1120 | Vikings |

### Starter (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Minkah Fitzpatrick | 73.53 | 67.26 | 77.28 | 616 | Steelers |
| 27 | 2 | Andre Cisco | 73.45 | 71.50 | 73.34 | 848 | Jaguars |
| 28 | 3 | Tre'von Moehrig | 73.38 | 68.00 | 73.39 | 1105 | Raiders |
| 29 | 4 | Kareem Jackson | 73.08 | 73.57 | 71.92 | 530 | Texans |
| 30 | 5 | Ji'Ayir Brown | 73.07 | 69.36 | 76.28 | 546 | 49ers |
| 31 | 6 | Harrison Smith | 72.54 | 69.50 | 71.66 | 1111 | Vikings |
| 32 | 7 | Brandon Jones | 71.96 | 73.13 | 71.82 | 542 | Dolphins |
| 33 | 8 | D'Anthony Bell | 71.40 | 65.46 | 78.06 | 256 | Browns |
| 34 | 9 | Ashtyn Davis | 71.25 | 67.81 | 73.40 | 218 | Jets |
| 35 | 10 | Talanoa Hufanga | 70.94 | 65.27 | 74.57 | 577 | 49ers |
| 36 | 11 | K'Von Wallace | 70.75 | 70.58 | 71.30 | 807 | Titans |
| 37 | 12 | Jordan Whitehead | 70.67 | 68.90 | 68.27 | 1076 | Jets |
| 38 | 13 | Ronnie Hickman Jr. | 70.47 | 66.37 | 74.92 | 322 | Browns |
| 39 | 14 | Jimmie Ward | 70.36 | 66.79 | 73.68 | 506 | Texans |
| 40 | 15 | Micah Hyde | 69.88 | 69.80 | 70.66 | 912 | Bills |
| 41 | 16 | Tashaun Gipson Sr. | 69.72 | 63.80 | 70.49 | 1194 | 49ers |
| 42 | 17 | Adrian Amos | 69.51 | 60.57 | 72.28 | 395 | Texans |
| 43 | 18 | Kamren Curl | 69.40 | 67.80 | 68.75 | 1088 | Commanders |
| 44 | 19 | Jason Pinnock | 69.29 | 64.80 | 69.10 | 1011 | Giants |
| 45 | 20 | Justin Simmons | 69.18 | 63.40 | 71.31 | 985 | Broncos |
| 46 | 21 | Jordan Howden | 68.86 | 64.32 | 69.69 | 569 | Saints |
| 47 | 22 | Budda Baker | 68.62 | 62.98 | 71.25 | 763 | Cardinals |
| 48 | 23 | Jaquan Brisker | 68.44 | 62.40 | 70.26 | 896 | Bears |
| 49 | 24 | Donovan Wilson | 68.41 | 66.40 | 67.65 | 776 | Cowboys |
| 50 | 25 | Duron Harmon | 68.41 | 61.64 | 73.65 | 222 | Browns |
| 51 | 26 | Juan Thornhill | 68.30 | 62.42 | 70.80 | 643 | Browns |
| 52 | 27 | Amani Hooker | 67.84 | 65.00 | 70.86 | 867 | Titans |
| 53 | 28 | Vonn Bell | 67.45 | 63.73 | 68.21 | 777 | Panthers |
| 54 | 29 | Josh Metellus | 67.44 | 64.50 | 68.97 | 1063 | Vikings |
| 55 | 30 | Terrell Edmunds | 67.15 | 60.26 | 69.15 | 475 | Titans |
| 56 | 31 | Jordan Fuller | 67.07 | 63.30 | 69.74 | 1057 | Rams |
| 57 | 32 | Jalen Pitre | 67.05 | 61.60 | 66.52 | 1032 | Texans |
| 58 | 33 | Marcus Epps | 66.79 | 62.30 | 65.82 | 1030 | Raiders |
| 59 | 34 | DeShon Elliott | 66.52 | 60.30 | 70.04 | 987 | Dolphins |
| 60 | 35 | Tony Adams | 66.34 | 65.20 | 67.84 | 879 | Jets |
| 61 | 36 | John Johnson III | 66.08 | 58.47 | 68.86 | 574 | Rams |
| 62 | 37 | Rayshawn Jenkins | 65.71 | 59.60 | 66.20 | 1099 | Jaguars |
| 63 | 38 | Sydney Brown | 65.47 | 63.12 | 66.79 | 334 | Eagles |
| 64 | 39 | Taylor Rapp | 65.43 | 58.01 | 67.98 | 422 | Bills |
| 65 | 40 | Mike Brown | 65.37 | 59.48 | 70.65 | 113 | Titans |
| 66 | 41 | Justin Evans | 65.30 | 60.59 | 71.83 | 197 | Eagles |
| 67 | 42 | Nick Cross | 65.27 | 61.90 | 69.23 | 292 | Colts |
| 68 | 43 | Miles Killebrew | 65.13 | 60.44 | 67.52 | 111 | Steelers |
| 69 | 44 | Bryan Cook | 65.03 | 62.63 | 65.90 | 593 | Chiefs |
| 70 | 45 | Damontae Kazee | 64.55 | 62.27 | 65.24 | 791 | Steelers |
| 71 | 46 | Rodney Thomas II | 64.25 | 60.10 | 63.59 | 962 | Colts |
| 72 | 47 | Jonathan Owens | 64.04 | 59.60 | 65.18 | 927 | Packers |
| 73 | 48 | Derwin James Jr. | 63.49 | 57.00 | 65.42 | 1001 | Chargers |
| 74 | 49 | Eddie Jackson | 63.34 | 60.00 | 65.92 | 646 | Bears |
| 75 | 50 | C.J. Gardner-Johnson | 62.76 | 62.49 | 65.64 | 291 | Lions |
| 76 | 51 | DeMarcco Hellams | 62.58 | 60.87 | 65.43 | 370 | Falcons |
| 77 | 52 | Kaevon Merriweather | 62.24 | 63.17 | 65.31 | 164 | Buccaneers |
| 78 | 53 | Isaiah Pola-Mao | 62.18 | 62.64 | 65.30 | 130 | Raiders |
| 79 | 54 | Percy Butler | 62.12 | 58.00 | 64.14 | 835 | Commanders |

### Rotation/backup (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 80 | 1 | DeAndre Houston-Carson | 61.87 | 56.73 | 64.57 | 589 | Texans |
| 81 | 2 | Mike Edwards | 61.85 | 58.61 | 61.61 | 809 | Chiefs |
| 82 | 3 | Eric Rowe | 61.81 | 60.91 | 65.50 | 212 | Steelers |
| 83 | 4 | Quandre Diggs | 61.56 | 55.40 | 61.50 | 1155 | Seahawks |
| 84 | 5 | Trenton Thompson | 61.51 | 61.93 | 66.80 | 212 | Steelers |
| 85 | 6 | Dane Belton | 61.24 | 59.40 | 59.76 | 295 | Giants |
| 86 | 7 | Tracy Walker III | 61.20 | 58.05 | 66.10 | 541 | Lions |
| 87 | 8 | Jaylinn Hawkins | 60.94 | 56.17 | 63.77 | 137 | Chargers |
| 88 | 9 | Kyle Dugger | 60.62 | 50.00 | 64.51 | 1116 | Patriots |
| 89 | 10 | Dean Marlowe | 60.46 | 58.50 | 64.57 | 298 | Chargers |
| 90 | 11 | P.J. Locke | 60.42 | 56.12 | 64.71 | 538 | Broncos |
| 91 | 12 | Justin Reid | 60.35 | 51.80 | 62.67 | 1247 | Chiefs |
| 92 | 13 | Elijah Campbell | 60.05 | 59.13 | 61.20 | 141 | Dolphins |
| 93 | 14 | Sam Franklin Jr. | 59.75 | 63.20 | 63.08 | 289 | Panthers |
| 94 | 15 | Eric Murray | 59.41 | 61.66 | 61.39 | 176 | Texans |
| 95 | 16 | Kerby Joseph | 58.64 | 50.90 | 60.74 | 1043 | Lions |
| 96 | 17 | Keanu Neal | 58.59 | 57.00 | 60.20 | 430 | Steelers |
| 97 | 18 | Darrick Forrest | 58.55 | 58.99 | 62.91 | 328 | Commanders |
| 98 | 19 | Russ Yeast | 58.33 | 55.60 | 58.91 | 836 | Rams |
| 99 | 20 | Jeremy Chinn | 58.06 | 53.68 | 61.23 | 285 | Panthers |
| 100 | 21 | Marcus Maye | 58.05 | 57.98 | 63.04 | 444 | Saints |
| 101 | 22 | Rodney McLeod | 57.54 | 54.38 | 59.70 | 280 | Browns |
| 102 | 23 | M.J. Stewart | 56.56 | 57.71 | 60.58 | 166 | Texans |
| 103 | 24 | Jayron Kearse | 55.38 | 43.60 | 60.15 | 860 | Cowboys |
| 104 | 25 | Adrian Phillips | 54.72 | 50.50 | 55.33 | 139 | Patriots |
| 105 | 26 | Johnathan Abram | 54.58 | 54.31 | 56.77 | 209 | Saints |
| 106 | 27 | Richie Grant | 54.22 | 42.40 | 57.93 | 945 | Falcons |
| 107 | 28 | Nick Scott | 53.49 | 42.39 | 57.03 | 569 | Bengals |
| 108 | 29 | Ryan Neal | 53.25 | 41.96 | 57.09 | 615 | Buccaneers |
| 109 | 30 | Terrell Burgess | 52.65 | 53.76 | 59.71 | 120 | Commanders |
| 110 | 31 | Delarrin Turner-Yell | 51.99 | 54.07 | 56.60 | 212 | Broncos |
| 111 | 32 | Jamal Adams | 51.61 | 51.86 | 56.87 | 518 | Seahawks |
| 112 | 33 | Alex Cook | 48.87 | 58.53 | 56.77 | 155 | Panthers |

## T — Tackle

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Trent Williams | 96.79 | 92.60 | 95.42 | 1013 | 49ers |
| 2 | 2 | Penei Sewell | 96.49 | 92.80 | 94.78 | 1379 | Lions |
| 3 | 3 | Jordan Mailata | 90.60 | 84.80 | 90.30 | 1206 | Eagles |
| 4 | 4 | Braden Smith | 89.63 | 80.00 | 91.89 | 575 | Colts |
| 5 | 5 | Tristan Wirfs | 88.79 | 83.10 | 88.42 | 1233 | Buccaneers |
| 6 | 6 | Tyron Smith | 88.15 | 83.39 | 87.15 | 942 | Cowboys |
| 7 | 7 | Christian Darrisaw | 87.86 | 82.38 | 87.34 | 982 | Vikings |
| 8 | 8 | Rob Havenstein | 87.61 | 79.36 | 88.95 | 914 | Rams |
| 9 | 9 | Bernhard Raimann | 87.55 | 82.70 | 86.61 | 1012 | Colts |
| 10 | 10 | Morgan Moses | 87.11 | 79.87 | 87.77 | 901 | Ravens |
| 11 | 11 | Zach Tom | 87.07 | 79.70 | 87.81 | 1162 | Packers |
| 12 | 12 | Lane Johnson | 86.61 | 80.10 | 86.78 | 1038 | Eagles |
| 13 | 13 | Taylor Decker | 86.24 | 81.10 | 85.50 | 1243 | Lions |
| 14 | 14 | Terron Armstead | 86.10 | 76.64 | 88.24 | 585 | Dolphins |
| 15 | 15 | Trent Brown | 86.00 | 77.37 | 87.58 | 579 | Patriots |
| 16 | 16 | Kolton Miller | 85.76 | 78.33 | 86.55 | 705 | Raiders |
| 17 | 17 | Kaleb McGary | 84.20 | 74.82 | 86.29 | 847 | Falcons |
| 18 | 18 | Laremy Tunsil | 83.83 | 75.41 | 85.28 | 965 | Texans |
| 19 | 19 | Garett Bolles | 83.14 | 75.90 | 83.80 | 1073 | Broncos |
| 20 | 20 | Dion Dawkins | 82.73 | 74.90 | 83.78 | 1264 | Bills |
| 21 | 21 | Rashawn Slater | 82.51 | 76.60 | 82.29 | 1154 | Chargers |
| 22 | 22 | Brian O'Neill | 82.49 | 74.04 | 83.95 | 884 | Vikings |
| 23 | 23 | Taylor Moton | 82.15 | 74.60 | 83.01 | 1148 | Panthers |
| 24 | 24 | Andrew Thomas | 81.86 | 73.83 | 83.05 | 576 | Giants |
| 25 | 25 | Luke Goedeke | 81.85 | 73.40 | 83.31 | 1236 | Buccaneers |
| 26 | 26 | Ryan Ramczyk | 81.14 | 72.63 | 82.65 | 785 | Saints |
| 27 | 27 | Spencer Brown | 80.77 | 70.10 | 83.71 | 1304 | Bills |
| 28 | 28 | Thayer Munford Jr. | 80.65 | 70.25 | 83.41 | 521 | Raiders |
| 29 | 29 | Charles Leno Jr. | 80.36 | 72.09 | 81.70 | 880 | Commanders |
| 30 | 30 | Jake Matthews | 80.05 | 71.20 | 81.78 | 1061 | Falcons |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Ikem Ekwonu | 79.66 | 67.40 | 83.67 | 1148 | Panthers |
| 32 | 2 | Jermaine Eluemunor | 79.57 | 68.48 | 82.79 | 905 | Raiders |
| 33 | 3 | Alijah Vera-Tucker | 79.35 | 65.89 | 84.15 | 250 | Jets |
| 34 | 4 | Patrick Mekari | 79.12 | 68.53 | 82.02 | 593 | Ravens |
| 35 | 5 | Mike McGlinchey | 78.86 | 67.41 | 82.32 | 947 | Broncos |
| 36 | 6 | Braxton Jones | 78.82 | 68.05 | 81.84 | 724 | Bears |
| 37 | 7 | Austin Jackson | 78.74 | 66.90 | 82.46 | 1050 | Dolphins |
| 38 | 8 | Andrew Wylie | 78.63 | 69.16 | 80.77 | 977 | Commanders |
| 39 | 9 | Jaylon Moore | 78.43 | 66.19 | 82.42 | 227 | 49ers |
| 40 | 10 | Storm Norton | 77.85 | 64.45 | 82.62 | 283 | Falcons |
| 41 | 11 | Ronnie Stanley | 77.78 | 67.71 | 80.33 | 834 | Ravens |
| 42 | 12 | Chris Hubbard | 76.47 | 65.68 | 79.50 | 473 | Titans |
| 43 | 13 | Alaric Jackson | 76.37 | 66.60 | 78.72 | 1026 | Rams |
| 44 | 14 | Rasheed Walker | 76.11 | 66.26 | 78.51 | 974 | Packers |
| 45 | 15 | Charles Cross | 76.06 | 66.98 | 77.94 | 832 | Seahawks |
| 46 | 16 | Conor McDermott | 76.00 | 63.74 | 80.00 | 227 | Patriots |
| 47 | 17 | Colton McKivitz | 75.68 | 65.20 | 78.50 | 1245 | 49ers |
| 48 | 18 | Cam Robinson | 75.23 | 63.98 | 78.56 | 535 | Jaguars |
| 49 | 19 | Orlando Brown Jr. | 75.03 | 66.10 | 76.82 | 1058 | Bengals |
| 50 | 20 | Darnell Wright | 74.99 | 62.40 | 79.22 | 1127 | Bears |
| 51 | 21 | Dawand Jones | 74.87 | 64.00 | 77.95 | 712 | Browns |
| 52 | 22 | David Quessenberry | 74.60 | 63.59 | 77.78 | 331 | Vikings |
| 53 | 23 | D.J. Humphries | 74.49 | 62.45 | 78.35 | 922 | Cardinals |
| 54 | 24 | Chukwuma Okorafor | 74.48 | 60.27 | 79.78 | 436 | Steelers |
| 55 | 25 | Kendall Lamm | 74.16 | 63.39 | 77.18 | 613 | Dolphins |

### Starter (45 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 56 | 1 | George Fant | 73.10 | 61.80 | 76.47 | 1004 | Texans |
| 57 | 2 | Cornelius Lucas | 72.97 | 61.93 | 76.17 | 231 | Commanders |
| 58 | 3 | Chuma Edoga | 72.92 | 60.91 | 76.76 | 420 | Cowboys |
| 59 | 4 | Stone Forsythe | 72.73 | 58.72 | 77.90 | 497 | Seahawks |
| 60 | 5 | Cam Fleming | 72.71 | 59.62 | 77.27 | 126 | Broncos |
| 61 | 6 | Trey Pipkins III | 72.63 | 62.80 | 75.01 | 1116 | Chargers |
| 62 | 7 | Paris Johnson Jr. | 72.37 | 60.10 | 76.38 | 1130 | Cardinals |
| 63 | 8 | Jason Peters | 72.06 | 55.85 | 78.70 | 215 | Seahawks |
| 64 | 9 | Walker Little | 71.73 | 59.02 | 76.04 | 659 | Jaguars |
| 65 | 10 | Kelvin Beachum | 71.69 | 60.81 | 74.77 | 212 | Cardinals |
| 66 | 11 | Yosh Nijman | 71.44 | 58.67 | 75.78 | 259 | Packers |
| 67 | 12 | Jonah Williams | 71.04 | 58.50 | 75.23 | 1087 | Bengals |
| 68 | 13 | Broderick Jones | 70.87 | 57.52 | 75.61 | 832 | Steelers |
| 69 | 14 | Jake Curhan | 70.62 | 57.64 | 75.10 | 296 | Seahawks |
| 70 | 15 | Cameron Erving | 70.28 | 54.95 | 76.33 | 208 | Saints |
| 71 | 16 | Donovan Smith | 69.91 | 57.80 | 73.81 | 1037 | Chiefs |
| 72 | 17 | Abraham Lucas | 69.50 | 56.37 | 74.08 | 273 | Seahawks |
| 73 | 18 | Jedrick Wills Jr. | 69.39 | 55.44 | 74.53 | 569 | Browns |
| 74 | 19 | Wanya Morris | 69.28 | 57.41 | 73.03 | 340 | Chiefs |
| 75 | 20 | Mekhi Becton | 69.05 | 53.20 | 75.45 | 985 | Jets |
| 76 | 21 | Dan Moore Jr. | 68.91 | 54.40 | 74.41 | 1017 | Steelers |
| 77 | 22 | Trevor Penning | 68.83 | 55.84 | 73.33 | 417 | Saints |
| 78 | 23 | Anton Harrison | 68.81 | 53.00 | 75.19 | 1112 | Jaguars |
| 79 | 24 | Tyre Phillips | 68.80 | 54.54 | 74.14 | 552 | Giants |
| 80 | 25 | Landon Young | 68.21 | 55.07 | 72.80 | 213 | Saints |
| 81 | 26 | Josh Jones | 68.18 | 52.81 | 74.26 | 233 | Texans |
| 82 | 27 | Terence Steele | 68.02 | 52.30 | 74.33 | 1273 | Cowboys |
| 83 | 28 | Charlie Heck | 67.45 | 53.61 | 72.51 | 253 | Texans |
| 84 | 29 | Billy Turner | 67.28 | 55.82 | 70.76 | 208 | Jets |
| 85 | 30 | Joe Noteboom | 67.24 | 54.13 | 71.82 | 573 | Rams |
| 86 | 31 | Matt Peart | 67.15 | 53.16 | 72.31 | 133 | Giants |
| 87 | 32 | Jawaan Taylor | 66.95 | 49.80 | 74.21 | 1364 | Chiefs |
| 88 | 33 | Daniel Faalele | 66.88 | 55.02 | 70.62 | 191 | Ravens |
| 89 | 34 | Calvin Anderson | 66.80 | 53.84 | 71.27 | 154 | Patriots |
| 90 | 35 | Andre Dillard | 66.79 | 53.20 | 71.69 | 562 | Titans |
| 91 | 36 | Max Mitchell | 66.13 | 52.58 | 70.99 | 474 | Jets |
| 92 | 37 | Carter Warren | 65.97 | 51.64 | 71.35 | 401 | Jets |
| 93 | 38 | Larry Borom | 65.78 | 52.25 | 70.64 | 411 | Bears |
| 94 | 39 | Duane Brown | 65.59 | 52.26 | 70.31 | 111 | Jets |
| 95 | 40 | Nicholas Petit-Frere | 64.57 | 50.20 | 69.98 | 117 | Titans |
| 96 | 41 | Geron Christian | 64.47 | 47.79 | 71.43 | 708 | Browns |
| 97 | 42 | James Hudson III | 64.34 | 47.84 | 71.18 | 622 | Browns |
| 98 | 43 | Evan Neal | 63.77 | 46.20 | 71.32 | 460 | Giants |
| 99 | 44 | Blake Freeland | 63.41 | 46.67 | 70.41 | 701 | Colts |
| 100 | 45 | Vederian Lowe | 63.05 | 47.28 | 69.39 | 476 | Patriots |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 101 | 1 | Jaelyn Duncan | 60.51 | 43.53 | 67.66 | 364 | Titans |
| 102 | 2 | Blake Hance | 58.60 | 47.20 | 62.04 | 152 | Jaguars |

## TE — Tight End

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | George Kittle | 85.19 | 87.33 | 79.60 | 631 | 49ers |
| 2 | 2 | Travis Kelce | 83.52 | 82.60 | 79.96 | 719 | Chiefs |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 3 | 1 | Mark Andrews | 79.77 | 75.30 | 78.58 | 314 | Ravens |
| 4 | 2 | Kyle Pitts | 77.28 | 67.45 | 79.66 | 495 | Falcons |
| 5 | 3 | Dallas Goedert | 76.88 | 70.21 | 77.16 | 512 | Eagles |
| 6 | 4 | T.J. Hockenson | 76.80 | 77.94 | 71.87 | 563 | Vikings |
| 7 | 5 | Jake Ferguson | 76.18 | 74.26 | 73.30 | 638 | Cowboys |
| 8 | 6 | Sam LaPorta | 75.83 | 77.00 | 70.89 | 683 | Lions |
| 9 | 7 | Hunter Henry | 74.80 | 68.13 | 75.08 | 413 | Patriots |
| 10 | 8 | Will Dissly | 74.35 | 65.68 | 75.96 | 167 | Seahawks |
| 11 | 9 | Andrew Ogletree | 74.17 | 63.03 | 77.43 | 158 | Colts |

### Starter (57 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Darren Waller | 73.87 | 68.12 | 73.53 | 394 | Giants |
| 13 | 2 | Cole Kmet | 73.15 | 71.96 | 69.78 | 503 | Bears |
| 14 | 3 | Marcedes Lewis | 73.11 | 70.94 | 70.39 | 103 | Bears |
| 15 | 4 | Trey McBride | 72.92 | 73.43 | 68.42 | 448 | Cardinals |
| 16 | 5 | David Njoku | 72.82 | 68.15 | 71.76 | 647 | Browns |
| 17 | 6 | Pat Freiermuth | 72.81 | 64.82 | 73.97 | 340 | Steelers |
| 18 | 7 | Evan Engram | 72.52 | 71.60 | 68.97 | 675 | Jaguars |
| 19 | 8 | Dalton Schultz | 72.17 | 70.76 | 68.95 | 526 | Texans |
| 20 | 9 | Noah Fant | 71.79 | 62.11 | 74.08 | 350 | Seahawks |
| 21 | 10 | Brevin Jordan | 71.22 | 65.23 | 71.04 | 184 | Texans |
| 22 | 11 | Donald Parham Jr. | 71.02 | 59.80 | 74.34 | 300 | Chargers |
| 23 | 12 | Davis Allen | 70.65 | 61.91 | 72.31 | 109 | Rams |
| 24 | 13 | Jonnu Smith | 70.52 | 59.28 | 73.84 | 426 | Falcons |
| 25 | 14 | Chigoziem Okonkwo | 70.42 | 61.73 | 72.04 | 448 | Titans |
| 26 | 15 | Tyler Conklin | 70.24 | 65.05 | 69.54 | 536 | Jets |
| 27 | 16 | Gerald Everett | 70.14 | 64.15 | 69.97 | 363 | Chargers |
| 28 | 17 | Austin Hooper | 70.03 | 58.52 | 73.54 | 327 | Raiders |
| 29 | 18 | Pharaoh Brown | 69.87 | 66.40 | 68.01 | 153 | Patriots |
| 30 | 19 | Dalton Kincaid | 69.56 | 67.98 | 66.44 | 543 | Bills |
| 31 | 20 | Luke Musgrave | 69.35 | 65.75 | 67.58 | 333 | Packers |
| 32 | 21 | Mike Gesicki | 69.32 | 56.39 | 73.78 | 354 | Patriots |
| 33 | 22 | Will Mallory | 69.25 | 63.10 | 69.19 | 133 | Colts |
| 34 | 23 | Tanner Hudson | 69.07 | 67.82 | 65.73 | 239 | Bengals |
| 35 | 24 | Colby Parkinson | 68.81 | 58.54 | 71.49 | 244 | Seahawks |
| 36 | 25 | Tucker Kraft | 68.79 | 60.31 | 70.28 | 397 | Packers |
| 37 | 26 | MyCole Pruitt | 68.78 | 59.71 | 70.66 | 149 | Falcons |
| 38 | 27 | Josh Oliver | 68.71 | 68.80 | 64.48 | 224 | Vikings |
| 39 | 28 | Mo Alie-Cox | 68.38 | 60.44 | 69.51 | 203 | Colts |
| 40 | 29 | Isaiah Likely | 68.08 | 64.16 | 66.52 | 365 | Ravens |
| 41 | 30 | Harrison Bryant | 68.04 | 58.48 | 70.25 | 169 | Browns |
| 42 | 31 | Dawson Knox | 68.02 | 56.27 | 71.68 | 303 | Bills |
| 43 | 32 | Tyler Higbee | 67.79 | 57.72 | 70.33 | 555 | Rams |
| 44 | 33 | Luke Farrell | 67.75 | 59.05 | 69.39 | 164 | Jaguars |
| 45 | 34 | Kylen Granson | 67.29 | 57.79 | 69.45 | 316 | Colts |
| 46 | 35 | Johnny Mundt | 66.94 | 64.81 | 64.19 | 131 | Vikings |
| 47 | 36 | Logan Thomas | 66.78 | 57.31 | 68.93 | 567 | Commanders |
| 48 | 37 | Connor Heyward | 66.51 | 57.34 | 68.46 | 230 | Steelers |
| 49 | 38 | Foster Moreau | 66.16 | 57.39 | 67.84 | 204 | Saints |
| 50 | 39 | Hayden Hurst | 66.06 | 50.62 | 72.18 | 248 | Panthers |
| 51 | 40 | Juwan Johnson | 65.99 | 59.13 | 66.39 | 344 | Saints |
| 52 | 41 | John Bates | 65.98 | 54.71 | 69.32 | 275 | Commanders |
| 53 | 42 | Chris Manhertz | 65.88 | 58.46 | 66.66 | 127 | Broncos |
| 54 | 43 | Robert Tonyan | 65.84 | 54.60 | 69.17 | 178 | Bears |
| 55 | 44 | Michael Mayer | 65.53 | 58.78 | 65.87 | 341 | Raiders |
| 56 | 45 | Jordan Akins | 65.32 | 57.08 | 66.64 | 137 | Browns |
| 57 | 46 | C.J. Uzomah | 65.32 | 62.87 | 62.79 | 133 | Jets |
| 58 | 47 | Noah Gray | 65.26 | 62.35 | 63.03 | 435 | Chiefs |
| 59 | 48 | Lucas Krull | 65.25 | 60.23 | 64.43 | 139 | Broncos |
| 60 | 49 | Drew Sample | 64.96 | 60.00 | 64.10 | 194 | Bengals |
| 61 | 50 | Adam Trautman | 64.80 | 54.49 | 67.50 | 434 | Broncos |
| 62 | 51 | Charlie Woerner | 64.35 | 61.52 | 62.07 | 125 | 49ers |
| 63 | 52 | Durham Smythe | 64.34 | 54.38 | 66.81 | 525 | Dolphins |
| 64 | 53 | Brock Wright | 63.92 | 53.25 | 66.87 | 209 | Lions |
| 65 | 54 | Geoff Swaim | 63.90 | 55.44 | 65.37 | 143 | Cardinals |
| 66 | 55 | Tommy Tremble | 63.78 | 57.11 | 64.06 | 285 | Panthers |
| 67 | 56 | Jeremy Ruckert | 62.86 | 58.75 | 61.43 | 165 | Jets |
| 68 | 57 | Cade Otton | 62.85 | 56.80 | 62.72 | 748 | Buccaneers |

### Rotation/backup (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 69 | 1 | Stone Smartt | 61.72 | 51.88 | 64.11 | 172 | Chargers |
| 70 | 2 | Brenton Strange | 61.50 | 55.18 | 61.55 | 118 | Jaguars |
| 71 | 3 | Darnell Washington | 61.34 | 53.15 | 62.64 | 195 | Steelers |
| 72 | 4 | Jack Stoll | 61.30 | 55.14 | 61.24 | 184 | Eagles |
| 73 | 5 | Daniel Bellinger | 61.08 | 52.02 | 62.95 | 354 | Giants |
| 74 | 6 | Irv Smith Jr. | 60.87 | 50.49 | 63.62 | 239 | Bengals |
| 75 | 7 | Blake Bell | 60.56 | 52.97 | 61.45 | 113 | Chiefs |
| 76 | 8 | Luke Schoonmaker | 60.26 | 54.96 | 59.62 | 161 | Cowboys |
| 77 | 9 | Julian Hill | 59.66 | 51.24 | 61.10 | 153 | Dolphins |

## WR — Wide Receiver

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Tyreek Hill | 89.32 | 91.06 | 84.00 | 515 | Dolphins |
| 2 | 2 | Brandon Aiyuk | 88.32 | 90.52 | 82.69 | 593 | 49ers |
| 3 | 3 | Nico Collins | 88.20 | 87.25 | 84.67 | 497 | Texans |
| 4 | 4 | A.J. Brown | 87.77 | 89.25 | 82.62 | 606 | Eagles |
| 5 | 5 | Puka Nacua | 87.52 | 90.15 | 81.60 | 658 | Rams |
| 6 | 6 | CeeDee Lamb | 87.30 | 90.90 | 80.73 | 736 | Cowboys |
| 7 | 7 | Justin Jefferson | 87.05 | 86.78 | 83.06 | 390 | Vikings |
| 8 | 8 | Amon-Ra St. Brown | 86.87 | 91.20 | 79.82 | 717 | Lions |
| 9 | 9 | DJ Moore | 85.69 | 88.66 | 79.55 | 613 | Bears |
| 10 | 10 | Jaylen Waddle | 84.85 | 86.71 | 79.45 | 425 | Dolphins |
| 11 | 11 | Deebo Samuel | 84.51 | 82.88 | 81.43 | 460 | 49ers |
| 12 | 12 | Ja'Marr Chase | 83.69 | 84.92 | 78.71 | 627 | Bengals |
| 13 | 13 | Rashee Rice | 83.62 | 83.28 | 79.68 | 577 | Chiefs |
| 14 | 14 | Mike Evans | 82.24 | 81.50 | 78.56 | 663 | Buccaneers |
| 15 | 15 | Amari Cooper | 82.06 | 79.60 | 79.54 | 627 | Browns |
| 16 | 16 | Chris Olave | 81.98 | 81.40 | 78.20 | 557 | Saints |
| 17 | 17 | Keenan Allen | 81.60 | 85.02 | 75.16 | 557 | Chargers |
| 18 | 18 | D.K. Metcalf | 80.89 | 79.30 | 77.79 | 585 | Seahawks |
| 19 | 19 | DeAndre Hopkins | 80.78 | 79.79 | 77.27 | 539 | Titans |
| 20 | 20 | Davante Adams | 80.69 | 78.55 | 77.95 | 598 | Raiders |
| 21 | 21 | Tank Dell | 80.20 | 76.65 | 78.40 | 334 | Texans |
| 22 | 22 | George Pickens | 80.16 | 73.63 | 80.35 | 617 | Steelers |

### Good (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Khalil Shakir | 79.39 | 73.21 | 79.35 | 413 | Bills |
| 24 | 2 | Dontayvion Wicks | 79.33 | 72.55 | 79.69 | 328 | Packers |
| 25 | 3 | Drake London | 79.15 | 77.54 | 76.05 | 511 | Falcons |
| 26 | 4 | Stefon Diggs | 78.99 | 79.10 | 74.75 | 695 | Bills |
| 27 | 5 | Terry McLaurin | 78.95 | 75.10 | 77.35 | 676 | Commanders |
| 28 | 6 | DeVonta Smith | 78.82 | 75.40 | 76.93 | 664 | Eagles |
| 29 | 7 | Tyler Lockett | 78.68 | 77.55 | 75.27 | 595 | Seahawks |
| 30 | 8 | Mike Williams | 78.16 | 69.46 | 79.79 | 113 | Chargers |
| 31 | 9 | Cooper Kupp | 78.14 | 70.37 | 79.15 | 461 | Rams |
| 32 | 10 | Chris Godwin | 78.13 | 77.50 | 74.38 | 686 | Buccaneers |
| 33 | 11 | Odell Beckham Jr. | 77.31 | 72.99 | 76.03 | 361 | Ravens |
| 34 | 12 | Rashid Shaheed | 77.22 | 67.37 | 79.62 | 443 | Saints |
| 35 | 13 | Michael Pittman Jr. | 77.20 | 77.26 | 72.99 | 606 | Colts |
| 36 | 14 | Jayden Reed | 77.07 | 70.91 | 77.01 | 444 | Packers |
| 37 | 15 | Diontae Johnson | 76.96 | 75.86 | 73.53 | 433 | Steelers |
| 38 | 16 | Tee Higgins | 76.92 | 70.62 | 76.95 | 419 | Bengals |
| 39 | 17 | Kalif Raymond | 76.73 | 69.36 | 77.48 | 244 | Lions |
| 40 | 18 | Zay Flowers | 76.67 | 75.87 | 73.03 | 626 | Ravens |
| 41 | 19 | Demario Douglas | 76.07 | 70.24 | 75.79 | 334 | Patriots |
| 42 | 20 | Deonte Harty | 75.91 | 63.95 | 79.72 | 130 | Bills |
| 43 | 21 | Jerry Jeudy | 75.88 | 67.14 | 77.54 | 487 | Broncos |
| 44 | 22 | Garrett Wilson | 75.81 | 72.90 | 73.58 | 712 | Jets |
| 45 | 23 | Christian Kirk | 75.60 | 69.66 | 75.39 | 405 | Jaguars |
| 46 | 24 | Marvin Mims Jr. | 75.55 | 61.95 | 80.45 | 262 | Broncos |
| 47 | 25 | Calvin Ridley | 75.32 | 71.40 | 73.77 | 688 | Jaguars |
| 48 | 26 | Courtland Sutton | 75.30 | 74.44 | 71.70 | 506 | Broncos |
| 49 | 27 | Romeo Doubs | 75.21 | 72.83 | 72.63 | 587 | Packers |
| 50 | 28 | Christian Watson | 75.05 | 65.67 | 77.14 | 316 | Packers |
| 51 | 29 | Michael Wilson | 74.98 | 66.81 | 76.26 | 444 | Cardinals |
| 52 | 30 | Josh Downs | 74.86 | 69.05 | 74.56 | 510 | Colts |
| 53 | 31 | Kendrick Bourne | 74.72 | 65.52 | 76.68 | 247 | Patriots |
| 54 | 32 | Noah Brown | 74.61 | 69.16 | 74.07 | 328 | Texans |
| 55 | 33 | Gabe Davis | 74.51 | 67.51 | 75.01 | 596 | Bills |
| 56 | 34 | Jordan Addison | 74.51 | 68.46 | 74.37 | 639 | Vikings |
| 57 | 35 | Darius Slayton | 74.45 | 67.05 | 75.21 | 584 | Giants |
| 58 | 36 | Josh Reynolds | 74.27 | 69.69 | 73.15 | 608 | Lions |

### Starter (91 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | Jakobi Meyers | 73.81 | 69.43 | 72.56 | 560 | Raiders |
| 60 | 2 | Adam Thielen | 73.52 | 72.70 | 69.90 | 677 | Panthers |
| 61 | 3 | Tre Tucker | 73.51 | 63.10 | 76.29 | 235 | Raiders |
| 62 | 4 | Jauan Jennings | 73.14 | 69.91 | 71.12 | 326 | 49ers |
| 63 | 5 | Chris Moore | 72.89 | 63.46 | 75.01 | 330 | Titans |
| 64 | 6 | Brandin Cooks | 72.89 | 68.60 | 71.59 | 635 | Cowboys |
| 65 | 7 | Tutu Atwell | 72.80 | 66.78 | 72.64 | 441 | Rams |
| 66 | 8 | Alex Erickson | 72.40 | 61.54 | 75.47 | 231 | Chargers |
| 67 | 9 | Kyle Philips | 72.37 | 65.41 | 72.84 | 118 | Titans |
| 68 | 10 | Cedrick Wilson Jr. | 72.34 | 63.68 | 73.94 | 318 | Dolphins |
| 69 | 11 | Joshua Palmer | 72.00 | 65.67 | 72.05 | 368 | Chargers |
| 70 | 12 | Greg Dortch | 71.99 | 65.12 | 72.41 | 257 | Cardinals |
| 71 | 13 | Michael Thomas | 71.96 | 66.11 | 71.69 | 333 | Saints |
| 72 | 14 | Curtis Samuel | 71.65 | 68.52 | 69.57 | 420 | Commanders |
| 73 | 15 | Lil'Jordan Humphrey | 71.53 | 60.24 | 74.89 | 242 | Broncos |
| 74 | 16 | KhaDarel Hodge | 71.51 | 61.29 | 74.16 | 191 | Falcons |
| 75 | 17 | Marquise Brown | 71.42 | 66.17 | 70.76 | 498 | Cardinals |
| 76 | 18 | Trenton Irwin | 70.96 | 58.05 | 75.40 | 279 | Bengals |
| 77 | 19 | Jake Bobo | 70.84 | 66.56 | 69.52 | 156 | Seahawks |
| 78 | 20 | A.T. Perry | 70.77 | 60.57 | 73.41 | 215 | Saints |
| 79 | 21 | DeVante Parker | 70.76 | 61.01 | 73.09 | 362 | Patriots |
| 80 | 22 | Byron Pringle | 70.70 | 63.25 | 71.50 | 112 | Commanders |
| 81 | 23 | Demarcus Robinson | 70.64 | 66.41 | 69.30 | 271 | Rams |
| 82 | 24 | Brandon Johnson | 70.34 | 62.43 | 71.45 | 232 | Broncos |
| 83 | 25 | Dyami Brown | 70.06 | 59.48 | 72.95 | 178 | Commanders |
| 84 | 26 | Mack Hollins | 70.03 | 62.32 | 71.00 | 168 | Falcons |
| 85 | 27 | Jamal Agnew | 69.86 | 59.95 | 72.30 | 156 | Jaguars |
| 86 | 28 | Equanimeous St. Brown | 69.84 | 57.93 | 73.62 | 118 | Bears |
| 87 | 29 | Justin Watson | 69.83 | 62.07 | 70.84 | 490 | Chiefs |
| 88 | 30 | Quez Watkins | 69.60 | 59.52 | 72.15 | 239 | Eagles |
| 89 | 31 | Tyler Boyd | 69.58 | 59.51 | 72.13 | 610 | Bengals |
| 90 | 32 | Jameson Williams | 69.48 | 62.09 | 70.24 | 342 | Lions |
| 91 | 33 | DJ Chark Jr. | 69.42 | 60.00 | 71.53 | 521 | Panthers |
| 92 | 34 | Michael Gallup | 69.37 | 63.09 | 69.39 | 474 | Cowboys |
| 93 | 35 | Mecole Hardman Jr. | 69.34 | 57.71 | 72.92 | 150 | Chiefs |
| 94 | 36 | Wan'Dale Robinson | 69.12 | 64.38 | 68.12 | 418 | Giants |
| 95 | 37 | Alec Pierce | 68.92 | 58.12 | 71.96 | 649 | Colts |
| 96 | 38 | Scott Miller | 68.84 | 61.98 | 69.24 | 133 | Falcons |
| 97 | 39 | JuJu Smith-Schuster | 68.65 | 57.68 | 71.80 | 256 | Patriots |
| 98 | 40 | Nick Westbrook-Ikhine | 68.65 | 59.56 | 70.55 | 362 | Titans |
| 99 | 41 | Donovan Peoples-Jones | 68.61 | 55.19 | 73.39 | 286 | Lions |
| 100 | 42 | Nelson Agholor | 68.23 | 61.40 | 68.61 | 419 | Ravens |
| 101 | 43 | Darnell Mooney | 68.15 | 55.62 | 72.34 | 482 | Bears |
| 102 | 44 | Richie James | 67.95 | 58.36 | 70.17 | 130 | Chiefs |
| 103 | 45 | Jaxon Smith-Njigba | 67.93 | 62.89 | 67.12 | 507 | Seahawks |
| 104 | 46 | Zay Jones | 67.64 | 60.84 | 68.00 | 325 | Jaguars |
| 105 | 47 | Ray-Ray McCloud III | 67.63 | 61.50 | 67.55 | 154 | 49ers |
| 106 | 48 | Chase Claypool | 67.60 | 56.77 | 70.66 | 123 | Dolphins |
| 107 | 49 | Skyy Moore | 67.59 | 56.47 | 70.83 | 305 | Chiefs |
| 108 | 50 | Jalin Hyatt | 67.55 | 59.77 | 68.57 | 403 | Giants |
| 109 | 51 | Quentin Johnston | 67.48 | 59.03 | 68.95 | 514 | Chargers |
| 110 | 52 | Rashod Bateman | 67.43 | 60.79 | 67.69 | 409 | Ravens |
| 111 | 53 | Elijah Moore | 67.28 | 58.92 | 68.69 | 631 | Browns |
| 112 | 54 | Olamide Zaccheaus | 67.11 | 53.54 | 71.99 | 348 | Eagles |
| 113 | 55 | Robert Woods | 67.05 | 60.63 | 67.17 | 470 | Texans |
| 114 | 56 | Treylon Burks | 66.97 | 55.00 | 70.79 | 286 | Titans |
| 115 | 57 | Brandon Powell | 66.91 | 61.16 | 66.58 | 305 | Vikings |
| 116 | 58 | Isaiah Hodgins | 66.90 | 57.59 | 68.94 | 296 | Giants |
| 117 | 59 | Kadarius Toney | 66.89 | 61.20 | 66.52 | 151 | Chiefs |
| 118 | 60 | Marquez Valdes-Scantling | 66.85 | 52.42 | 72.30 | 608 | Chiefs |
| 119 | 61 | Julio Jones | 66.74 | 56.45 | 69.44 | 176 | Eagles |
| 120 | 62 | Xavier Gipson | 66.58 | 57.19 | 68.67 | 360 | Jets |
| 121 | 63 | Jalen Tolbert | 66.23 | 57.56 | 67.84 | 321 | Cowboys |
| 122 | 64 | Calvin Austin III | 66.10 | 57.86 | 67.43 | 262 | Steelers |
| 123 | 65 | Terrace Marshall Jr. | 66.06 | 56.94 | 67.97 | 229 | Panthers |
| 124 | 66 | Trent Sherfield | 65.99 | 54.58 | 69.43 | 245 | Bills |
| 125 | 67 | Allen Lazard | 65.81 | 53.76 | 69.68 | 483 | Jets |
| 126 | 68 | Jahan Dotson | 65.74 | 57.70 | 66.93 | 675 | Commanders |
| 127 | 69 | David Bell | 65.67 | 61.82 | 64.07 | 168 | Browns |
| 128 | 70 | Randall Cobb | 65.64 | 51.40 | 70.97 | 158 | Jets |
| 129 | 71 | Braxton Berrios | 65.40 | 58.26 | 66.00 | 348 | Dolphins |
| 130 | 72 | K.J. Osborn | 65.38 | 54.21 | 68.66 | 594 | Vikings |
| 131 | 73 | Cedric Tillman | 65.37 | 56.58 | 67.06 | 382 | Browns |
| 132 | 74 | Jason Brownlee | 65.17 | 53.20 | 68.98 | 202 | Jets |
| 133 | 75 | Jalen Reagor | 65.16 | 55.22 | 67.62 | 204 | Patriots |
| 134 | 76 | John Metchie III | 65.13 | 58.71 | 65.25 | 226 | Texans |
| 135 | 77 | Trey Palmer | 65.06 | 53.96 | 68.30 | 537 | Buccaneers |
| 136 | 78 | Parris Campbell | 64.57 | 55.90 | 66.18 | 150 | Giants |
| 137 | 79 | Jalen Guyton | 64.46 | 54.49 | 66.94 | 185 | Chargers |
| 138 | 80 | Deven Thompkins | 64.41 | 59.67 | 63.41 | 149 | Buccaneers |
| 139 | 81 | Van Jefferson | 64.37 | 52.34 | 68.23 | 395 | Falcons |
| 140 | 82 | Hunter Renfrow | 64.21 | 51.07 | 68.81 | 270 | Raiders |
| 141 | 83 | Jonathan Mingo | 63.60 | 55.04 | 65.14 | 577 | Panthers |
| 142 | 84 | Tyquan Thornton | 63.58 | 58.89 | 62.54 | 141 | Patriots |
| 143 | 85 | Rondale Moore | 63.36 | 54.49 | 65.10 | 489 | Cardinals |
| 144 | 86 | Malik Heath | 63.17 | 60.04 | 61.09 | 121 | Packers |
| 145 | 87 | Allen Robinson II | 63.15 | 53.57 | 65.37 | 445 | Steelers |
| 146 | 88 | Tim Jones | 62.89 | 55.63 | 63.56 | 197 | Jaguars |
| 147 | 89 | Zach Pascal | 62.54 | 52.06 | 65.36 | 103 | Cardinals |
| 148 | 90 | Andrei Iosivas | 62.42 | 60.59 | 59.47 | 159 | Bengals |
| 149 | 91 | Tyler Scott | 62.00 | 55.32 | 62.29 | 271 | Bears |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 150 | 1 | Parker Washington | 61.55 | 55.28 | 61.56 | 186 | Jaguars |
| 151 | 2 | Xavier Hutchinson | 60.96 | 53.01 | 62.10 | 200 | Texans |
