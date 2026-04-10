# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:26:35Z
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
| 2 | 2 | Connor Williams | 93.75 | 86.50 | 94.41 | 497 | Dolphins |
| 3 | 3 | Creed Humphrey | 89.66 | 81.40 | 91.00 | 1380 | Chiefs |
| 4 | 4 | Drew Dalman | 89.40 | 82.30 | 89.96 | 932 | Falcons |
| 5 | 5 | Erik McCoy | 88.94 | 79.40 | 91.14 | 1152 | Saints |
| 6 | 6 | Jason Kelce | 88.42 | 78.60 | 90.80 | 1165 | Eagles |
| 7 | 7 | Tyler Linderbaum | 86.74 | 78.50 | 88.06 | 1043 | Ravens |
| 8 | 8 | Ryan Kelly | 85.75 | 77.20 | 87.29 | 882 | Colts |
| 9 | 9 | Andre James | 83.23 | 74.60 | 84.81 | 963 | Raiders |
| 10 | 10 | Lloyd Cushenberry III | 82.89 | 73.20 | 85.19 | 1070 | Broncos |
| 11 | 11 | Aaron Brewer | 81.04 | 71.60 | 83.16 | 1050 | Titans |
| 12 | 12 | David Andrews | 80.87 | 71.20 | 83.15 | 1050 | Patriots |
| 13 | 13 | Ethan Pocic | 80.00 | 70.80 | 81.96 | 1070 | Browns |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Tyler Biadasz | 78.52 | 69.20 | 80.56 | 1123 | Cowboys |
| 15 | 2 | Sam Mustipher | 77.80 | 67.00 | 80.84 | 202 | Ravens |
| 16 | 3 | Jake Brendel | 77.23 | 66.30 | 80.35 | 1229 | 49ers |
| 17 | 4 | Ted Karras | 77.04 | 67.40 | 79.30 | 1075 | Bengals |
| 18 | 5 | Hjalte Froholdt | 75.99 | 64.10 | 79.75 | 1123 | Cardinals |
| 19 | 6 | Mitch Morse | 75.00 | 63.90 | 78.23 | 1272 | Bills |
| 20 | 7 | Corey Linsley | 74.74 | 63.50 | 78.07 | 214 | Chargers |
| 21 | 8 | Joe Tippmann | 74.13 | 61.00 | 78.71 | 852 | Jets |
| 22 | 9 | Bradley Bozeman | 74.01 | 62.20 | 77.72 | 1148 | Panthers |

### Starter (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Garrett Bradbury | 73.72 | 60.90 | 78.10 | 874 | Vikings |
| 24 | 2 | Jarrett Patterson | 69.70 | 60.40 | 71.73 | 464 | Texans |
| 25 | 3 | Will Clapp | 69.03 | 56.70 | 73.08 | 702 | Chargers |
| 26 | 4 | Mason Cole | 68.76 | 53.20 | 74.96 | 1135 | Steelers |
| 27 | 5 | Josh Myers | 67.42 | 54.70 | 71.73 | 1212 | Packers |
| 28 | 6 | Nick Harris | 67.08 | 59.50 | 67.97 | 313 | Browns |
| 29 | 7 | Tyler Larsen | 66.43 | 50.30 | 73.02 | 466 | Commanders |
| 30 | 8 | Connor McGovern | 66.25 | 48.50 | 73.91 | 371 | Jets |
| 31 | 9 | Robert Hainsey | 64.77 | 50.20 | 70.31 | 1236 | Buccaneers |
| 32 | 10 | Ryan Neuzil | 63.76 | 55.30 | 65.23 | 203 | Falcons |
| 33 | 11 | Olusegun Oluwatimi | 62.78 | 53.10 | 65.06 | 128 | Seahawks |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Brock Hoffman | 61.88 | 53.70 | 63.16 | 222 | Cowboys |
| 35 | 2 | Luke Fortner | 61.69 | 44.30 | 69.11 | 1163 | Jaguars |
| 36 | 3 | Wesley French | 61.08 | 48.40 | 65.37 | 270 | Colts |
| 37 | 4 | John Michael Schmitz Jr. | 60.11 | 41.40 | 68.42 | 755 | Giants |

## CB — Cornerback

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Sauce Gardner | 93.13 | 90.80 | 91.14 | 1049 | Jets |
| 2 | 2 | Jaylon Johnson | 89.27 | 90.40 | 87.98 | 809 | Bears |
| 3 | 3 | Charvarius Ward | 89.19 | 86.50 | 87.60 | 1148 | 49ers |
| 4 | 4 | Darious Williams | 88.68 | 85.30 | 87.35 | 1035 | Jaguars |
| 5 | 5 | DaRon Bland | 88.49 | 86.40 | 87.19 | 1020 | Cowboys |
| 6 | 6 | Kendall Fuller | 86.56 | 82.80 | 86.09 | 1020 | Commanders |
| 7 | 7 | Christian Benford | 85.87 | 83.30 | 86.97 | 837 | Bills |
| 8 | 8 | Derek Stingley Jr. | 85.28 | 85.30 | 86.50 | 812 | Texans |
| 9 | 9 | Rasul Douglas | 84.63 | 81.80 | 83.34 | 1040 | Bills |
| 10 | 10 | Trevon Diggs | 84.59 | 82.80 | 89.16 | 101 | Cowboys |
| 11 | 11 | Paulson Adebo | 84.04 | 80.50 | 84.38 | 948 | Saints |
| 12 | 12 | Michael Carter II | 83.95 | 83.30 | 81.59 | 671 | Jets |
| 13 | 13 | Devon Witherspoon | 83.18 | 79.70 | 84.26 | 883 | Seahawks |
| 14 | 14 | Taron Johnson | 83.06 | 81.00 | 80.75 | 1044 | Bills |
| 15 | 15 | D.J. Reed | 82.77 | 79.50 | 82.35 | 993 | Jets |
| 16 | 16 | Mike Hilton | 82.36 | 80.40 | 80.39 | 876 | Bengals |
| 17 | 17 | Jaycee Horn | 81.91 | 83.00 | 86.34 | 275 | Panthers |
| 18 | 18 | Tariq Woolen | 81.65 | 75.00 | 82.53 | 940 | Seahawks |
| 19 | 19 | Mekhi Blackmon | 81.42 | 71.80 | 85.64 | 434 | Vikings |
| 20 | 20 | Joshua Williams | 81.27 | 75.20 | 82.62 | 420 | Chiefs |
| 21 | 21 | Trent McDuffie | 81.14 | 81.50 | 78.93 | 1243 | Chiefs |
| 22 | 22 | Desmond King II | 81.01 | 80.70 | 80.68 | 400 | Texans |
| 23 | 23 | Nick McCloud | 80.38 | 80.70 | 79.44 | 312 | Giants |
| 24 | 24 | Kenny Moore II | 80.26 | 78.40 | 79.30 | 1089 | Colts |
| 25 | 25 | Asante Samuel Jr. | 80.24 | 75.60 | 80.15 | 1111 | Chargers |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Jaire Alexander | 79.85 | 78.60 | 83.28 | 560 | Packers |
| 27 | 2 | Mike Jackson | 79.52 | 76.90 | 81.02 | 474 | Seahawks |
| 28 | 3 | Steven Nelson | 79.26 | 74.00 | 79.39 | 1192 | Texans |
| 29 | 4 | A.J. Terrell | 79.11 | 74.60 | 79.03 | 1065 | Falcons |
| 30 | 5 | Jamel Dean | 79.05 | 72.50 | 81.22 | 824 | Buccaneers |
| 31 | 6 | L'Jarius Sneed | 78.92 | 73.80 | 78.56 | 1260 | Chiefs |
| 32 | 7 | Isaac Yiadom | 78.89 | 80.40 | 80.87 | 517 | Saints |
| 33 | 8 | Jonathan Jones | 78.47 | 75.10 | 80.47 | 724 | Patriots |
| 34 | 9 | Tre'Davious White | 77.24 | 79.00 | 82.69 | 182 | Bills |
| 35 | 10 | Deommodore Lenoir | 76.97 | 74.30 | 76.84 | 1198 | 49ers |
| 36 | 11 | Stephon Gilmore | 76.95 | 69.30 | 79.95 | 1055 | Cowboys |
| 37 | 12 | Tyler Hall | 76.36 | 71.90 | 82.04 | 161 | Raiders |
| 38 | 13 | Dane Jackson | 75.93 | 70.90 | 77.96 | 578 | Bills |
| 39 | 14 | Denzel Ward | 75.63 | 69.40 | 78.36 | 657 | Browns |
| 40 | 15 | Jack Jones | 75.54 | 71.90 | 78.95 | 471 | Raiders |
| 41 | 16 | Roger McCreary | 75.30 | 72.20 | 74.44 | 934 | Titans |
| 42 | 17 | Ronald Darby | 75.18 | 72.90 | 79.20 | 554 | Ravens |
| 43 | 18 | Greg Newsome II | 75.16 | 69.90 | 77.05 | 795 | Browns |
| 44 | 19 | Pat Surtain II | 75.13 | 64.70 | 78.11 | 1121 | Broncos |
| 45 | 20 | Dee Alford | 74.73 | 72.80 | 75.76 | 571 | Falcons |
| 46 | 21 | Joey Porter Jr. | 74.64 | 66.60 | 75.84 | 855 | Steelers |
| 47 | 22 | Marshon Lattimore | 74.46 | 69.10 | 80.44 | 621 | Saints |

### Starter (66 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 48 | 1 | Isaiah Oliver | 73.96 | 70.90 | 75.85 | 503 | 49ers |
| 49 | 2 | Jaylen Watson | 73.90 | 66.20 | 75.23 | 533 | Chiefs |
| 50 | 3 | Cam Taylor-Britt | 73.80 | 70.30 | 77.60 | 653 | Bengals |
| 51 | 4 | Darius Slay | 73.50 | 65.40 | 76.88 | 874 | Eagles |
| 52 | 5 | Christian Gonzalez | 73.48 | 79.30 | 83.94 | 209 | Patriots |
| 53 | 6 | Antonio Hamilton Sr. | 73.07 | 69.10 | 76.25 | 559 | Cardinals |
| 54 | 7 | Jalen Ramsey | 72.87 | 65.00 | 77.08 | 659 | Dolphins |
| 55 | 8 | Martin Emerson Jr. | 72.46 | 63.60 | 74.20 | 897 | Browns |
| 56 | 9 | Kyler Gordon | 72.22 | 68.20 | 74.28 | 646 | Bears |
| 57 | 10 | Donte Jackson | 71.70 | 66.60 | 74.75 | 902 | Panthers |
| 58 | 11 | Shaquill Griffin | 71.54 | 67.00 | 77.95 | 459 | Panthers |
| 59 | 12 | Chandon Sullivan | 71.53 | 65.90 | 71.12 | 442 | Steelers |
| 60 | 13 | Troy Hill | 71.49 | 66.90 | 73.32 | 493 | Panthers |
| 61 | 14 | Darnay Holmes | 71.45 | 68.70 | 75.30 | 123 | Giants |
| 62 | 15 | Rock Ya-Sin | 70.73 | 64.00 | 76.53 | 281 | Ravens |
| 63 | 16 | Tyrique Stevenson | 70.59 | 59.10 | 75.06 | 830 | Bears |
| 64 | 17 | Levi Wallace | 70.34 | 60.40 | 73.38 | 762 | Steelers |
| 65 | 18 | Fabian Moreau | 70.28 | 62.90 | 73.10 | 739 | Broncos |
| 66 | 19 | Tre Herndon | 70.15 | 73.10 | 68.43 | 482 | Jaguars |
| 67 | 20 | Grayland Arnold | 70.13 | 77.00 | 71.76 | 143 | Texans |
| 68 | 21 | Carlton Davis III | 70.03 | 63.30 | 74.36 | 847 | Buccaneers |
| 69 | 22 | Tavierre Thomas | 70.03 | 69.50 | 73.57 | 352 | Texans |
| 70 | 23 | Chidobe Awuzie | 70.00 | 62.30 | 75.18 | 722 | Bengals |
| 71 | 24 | Myles Bryant | 69.94 | 63.10 | 71.32 | 852 | Patriots |
| 72 | 25 | Terell Smith | 69.93 | 65.40 | 74.66 | 377 | Bears |
| 73 | 26 | Kelee Ringo | 69.76 | 63.50 | 81.07 | 233 | Eagles |
| 74 | 27 | Nate Hobbs | 69.68 | 68.10 | 70.68 | 775 | Raiders |
| 75 | 28 | Marlon Humphrey | 69.61 | 63.20 | 73.64 | 553 | Ravens |
| 76 | 29 | Ambry Thomas | 68.98 | 61.50 | 75.10 | 573 | 49ers |
| 77 | 30 | JuJu Brents | 68.92 | 64.50 | 75.55 | 497 | Colts |
| 78 | 31 | Arthur Maulet | 68.91 | 64.20 | 68.76 | 458 | Ravens |
| 79 | 32 | Dallis Flowers | 68.77 | 70.60 | 75.89 | 304 | Colts |
| 80 | 33 | Patrick Peterson | 68.52 | 59.80 | 70.95 | 1162 | Steelers |
| 81 | 34 | Andrew Booth Jr. | 68.44 | 69.10 | 73.64 | 151 | Vikings |
| 82 | 35 | Justin Bethel | 68.13 | 62.00 | 76.28 | 126 | Dolphins |
| 83 | 36 | Tre Brown | 67.99 | 64.10 | 73.86 | 603 | Seahawks |
| 84 | 37 | Derion Kendrick | 67.83 | 60.40 | 70.45 | 871 | Rams |
| 85 | 38 | Kader Kohou | 67.75 | 62.00 | 68.15 | 985 | Dolphins |
| 86 | 39 | Artie Burns | 67.70 | 66.60 | 75.54 | 232 | Seahawks |
| 87 | 40 | Carrington Valentine | 67.45 | 59.10 | 68.85 | 846 | Packers |
| 88 | 41 | Alex Austin | 66.88 | 63.50 | 76.27 | 216 | Patriots |
| 89 | 42 | Tyson Campbell | 66.39 | 56.40 | 72.21 | 589 | Jaguars |
| 90 | 43 | Greg Stroman Jr. | 66.16 | 67.60 | 76.53 | 150 | Bears |
| 91 | 44 | Amik Robertson | 66.12 | 65.80 | 65.80 | 674 | Raiders |
| 92 | 45 | James Bradberry | 66.01 | 52.00 | 71.19 | 1090 | Eagles |
| 93 | 46 | Bryce Hall | 65.99 | 62.80 | 74.25 | 138 | Jets |
| 94 | 47 | Kaiir Elam | 65.97 | 65.80 | 72.33 | 210 | Bills |
| 95 | 48 | Cameron Mitchell | 65.96 | 60.50 | 68.36 | 283 | Browns |
| 96 | 49 | Ja'Sir Taylor | 65.88 | 58.00 | 72.73 | 534 | Chargers |
| 97 | 50 | Ahkello Witherspoon | 65.85 | 60.00 | 70.99 | 1115 | Rams |
| 98 | 51 | Jaylon Jones | 65.64 | 55.80 | 70.00 | 788 | Colts |
| 99 | 52 | Ka'dar Hollman | 65.28 | 58.50 | 69.06 | 131 | Texans |
| 100 | 53 | Cobie Durant | 65.20 | 54.70 | 71.71 | 683 | Rams |
| 101 | 54 | Benjamin St-Juste | 65.12 | 56.40 | 70.69 | 1063 | Commanders |
| 102 | 55 | Michael Davis | 64.89 | 54.30 | 70.23 | 886 | Chargers |
| 103 | 56 | Byron Murphy Jr. | 64.82 | 58.20 | 69.08 | 906 | Vikings |
| 104 | 57 | Keisean Nixon | 64.81 | 60.40 | 67.32 | 937 | Packers |
| 105 | 58 | Xavien Howard | 64.33 | 51.90 | 71.20 | 743 | Dolphins |
| 106 | 59 | Cor'Dale Flott | 64.33 | 59.40 | 67.86 | 519 | Giants |
| 107 | 60 | Bradley Roby | 64.13 | 57.00 | 69.91 | 379 | Eagles |
| 108 | 61 | Ja'Quan McMillian | 64.11 | 62.10 | 68.39 | 669 | Broncos |
| 109 | 62 | Emmanuel Forbes | 62.92 | 57.60 | 65.23 | 482 | Commanders |
| 110 | 63 | Darrell Baker Jr. | 62.90 | 54.40 | 70.28 | 469 | Colts |
| 111 | 64 | Jerry Jacobs | 62.66 | 54.50 | 68.83 | 743 | Lions |
| 112 | 65 | Deane Leonard | 62.48 | 60.00 | 72.22 | 222 | Chargers |
| 113 | 66 | DJ Turner II | 62.16 | 48.40 | 67.16 | 827 | Bengals |

### Rotation/backup (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 114 | 1 | Alontae Taylor | 61.85 | 51.80 | 66.58 | 950 | Saints |
| 115 | 2 | Cameron Sutton | 61.82 | 49.00 | 66.69 | 1261 | Lions |
| 116 | 3 | Sean Murphy-Bunting | 61.68 | 54.40 | 67.77 | 840 | Titans |
| 117 | 4 | Essang Bassey | 61.53 | 58.70 | 65.02 | 353 | Chargers |
| 118 | 5 | Garrett Williams | 61.47 | 57.20 | 68.00 | 360 | Cardinals |
| 119 | 6 | Clark Phillips III | 61.19 | 58.50 | 70.11 | 414 | Falcons |
| 120 | 7 | Dicaprio Bootle | 61.18 | 64.10 | 70.57 | 183 | Panthers |
| 121 | 8 | Brandin Echols | 60.80 | 53.20 | 66.21 | 143 | Jets |
| 122 | 9 | Deonte Banks | 60.41 | 48.60 | 66.08 | 844 | Giants |
| 123 | 10 | Corey Ballentine | 60.41 | 62.30 | 61.55 | 534 | Packers |
| 124 | 11 | Shaun Wade | 60.03 | 59.10 | 66.49 | 348 | Patriots |
| 125 | 12 | Akayleb Evans | 59.18 | 52.60 | 64.30 | 855 | Vikings |
| 126 | 13 | Mike Hughes | 59.16 | 49.10 | 64.35 | 333 | Falcons |
| 127 | 14 | Zyon McCollum | 58.26 | 46.30 | 65.00 | 870 | Buccaneers |
| 128 | 15 | Adoree' Jackson | 57.91 | 45.90 | 66.06 | 792 | Giants |
| 129 | 16 | Kristian Fulton | 57.34 | 46.90 | 65.62 | 644 | Titans |
| 130 | 17 | Kei'Trel Clark | 57.19 | 53.40 | 64.36 | 464 | Cardinals |
| 131 | 18 | Jeff Okudah | 56.63 | 46.00 | 65.23 | 596 | Falcons |
| 132 | 19 | CJ Henderson | 56.54 | 43.90 | 64.71 | 407 | Panthers |
| 133 | 20 | Eli Apple | 56.52 | 43.40 | 64.84 | 624 | Dolphins |
| 134 | 21 | Eli Ricks | 56.02 | 43.50 | 62.16 | 316 | Eagles |
| 135 | 22 | Jourdan Lewis | 55.95 | 39.50 | 66.18 | 771 | Cowboys |
| 136 | 23 | Marco Wilson | 55.45 | 40.40 | 65.49 | 704 | Patriots |
| 137 | 24 | Tre Flowers | 55.38 | 41.90 | 65.50 | 200 | Falcons |
| 138 | 25 | Jakorian Bennett | 54.83 | 41.10 | 64.72 | 361 | Raiders |
| 139 | 26 | Josh Jobe | 53.84 | 42.70 | 67.15 | 240 | Eagles |
| 140 | 27 | Avonte Maddox | 52.05 | 39.30 | 64.81 | 211 | Eagles |
| 141 | 28 | J.C. Jackson | 51.94 | 30.80 | 68.83 | 524 | Patriots |
| 142 | 29 | Kindle Vildor | 51.92 | 45.50 | 57.91 | 388 | Lions |
| 143 | 30 | Montaric Brown | 51.89 | 54.50 | 55.66 | 475 | Jaguars |
| 144 | 31 | Damarri Mathis | 51.69 | 36.70 | 61.68 | 440 | Broncos |
| 145 | 32 | Tre Hawkins III | 51.68 | 47.20 | 57.37 | 346 | Giants |
| 146 | 33 | Eric Stokes | 50.21 | 45.90 | 58.81 | 110 | Packers |
| 147 | 34 | Starling Thomas V | 50.10 | 44.10 | 55.82 | 473 | Cardinals |
| 148 | 35 | D'Shawn Jamison | 49.31 | 43.50 | 60.31 | 107 | Panthers |

## DI — Defensive Interior

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Quinnen Williams | 90.89 | 89.95 | 88.03 | 778 | Jets |
| 2 | 2 | Aaron Donald | 90.09 | 85.22 | 90.93 | 916 | Rams |
| 3 | 3 | Kobie Turner | 87.90 | 84.76 | 85.82 | 729 | Rams |
| 4 | 4 | DeForest Buckner | 86.31 | 86.73 | 81.86 | 841 | Colts |
| 5 | 5 | Dexter Lawrence | 85.89 | 89.54 | 80.28 | 709 | Giants |
| 6 | 6 | Christian Wilkins | 85.50 | 86.58 | 80.61 | 968 | Dolphins |
| 7 | 7 | Chris Jones | 84.36 | 88.33 | 78.13 | 947 | Chiefs |
| 8 | 8 | Christian Barmore | 84.13 | 81.76 | 83.61 | 750 | Patriots |
| 9 | 9 | Derrick Brown | 83.58 | 88.70 | 76.20 | 938 | Panthers |
| 10 | 10 | Jalen Carter | 82.19 | 88.50 | 73.82 | 599 | Eagles |
| 11 | 11 | Cameron Heyward | 81.75 | 78.08 | 82.48 | 497 | Steelers |
| 12 | 12 | Leonard Williams | 81.50 | 81.45 | 78.83 | 884 | Seahawks |
| 13 | 13 | Jeffery Simmons | 81.29 | 81.88 | 79.76 | 657 | Titans |
| 14 | 14 | Ed Oliver | 80.32 | 70.89 | 83.63 | 817 | Bills |
| 15 | 15 | Zach Sieler | 80.12 | 69.86 | 82.79 | 924 | Dolphins |
| 16 | 16 | Vita Vea | 80.12 | 79.30 | 77.58 | 691 | Buccaneers |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Kenny Clark | 79.79 | 76.71 | 77.87 | 913 | Packers |
| 18 | 2 | Jonathan Allen | 79.75 | 70.58 | 82.48 | 867 | Commanders |
| 19 | 3 | Javon Hargrave | 78.91 | 70.25 | 80.71 | 775 | 49ers |
| 20 | 4 | Grady Jarrett | 78.87 | 77.57 | 79.98 | 318 | Falcons |
| 21 | 5 | DJ Reader | 78.82 | 84.02 | 75.11 | 535 | Bengals |
| 22 | 6 | Milton Williams | 77.99 | 67.44 | 80.86 | 522 | Eagles |
| 23 | 7 | Desjuan Johnson | 77.09 | 57.65 | 90.78 | 105 | Rams |
| 24 | 8 | Osa Odighizuwa | 77.02 | 68.64 | 78.64 | 676 | Cowboys |
| 25 | 9 | Alim McNeill | 77.01 | 79.29 | 71.80 | 682 | Lions |
| 26 | 10 | Zach Allen | 76.78 | 73.75 | 78.31 | 913 | Broncos |
| 27 | 11 | Michael Pierce | 76.69 | 76.69 | 78.41 | 698 | Ravens |
| 28 | 12 | Jordan Davis | 76.65 | 78.48 | 72.73 | 561 | Eagles |
| 29 | 13 | Shelby Harris | 76.64 | 65.77 | 80.50 | 462 | Browns |
| 30 | 14 | David Onyemata | 76.42 | 74.22 | 76.37 | 594 | Falcons |
| 31 | 15 | Daron Payne | 75.75 | 65.01 | 78.74 | 924 | Commanders |
| 32 | 16 | Devonte Wyatt | 74.54 | 62.73 | 78.62 | 644 | Packers |
| 33 | 17 | B.J. Hill | 74.09 | 67.58 | 74.74 | 776 | Bengals |

### Starter (71 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Fletcher Cox | 73.97 | 64.25 | 76.96 | 721 | Eagles |
| 35 | 2 | Calijah Kancey | 73.49 | 52.68 | 84.18 | 663 | Buccaneers |
| 36 | 3 | Grover Stewart | 73.40 | 68.59 | 75.38 | 445 | Colts |
| 37 | 4 | Keeanu Benton | 73.25 | 76.49 | 66.93 | 516 | Steelers |
| 38 | 5 | Chauncey Golston | 72.61 | 60.52 | 77.23 | 322 | Cowboys |
| 39 | 6 | Arik Armstead | 72.37 | 67.55 | 74.75 | 661 | 49ers |
| 40 | 7 | Dre'Mont Jones | 72.25 | 64.41 | 74.67 | 762 | Seahawks |
| 41 | 8 | Travis Jones | 72.17 | 64.48 | 73.86 | 518 | Ravens |
| 42 | 9 | DaQuan Jones | 72.12 | 75.06 | 70.21 | 240 | Bills |
| 43 | 10 | Karl Brooks | 72.01 | 62.47 | 74.20 | 440 | Packers |
| 44 | 11 | Dalvin Tomlinson | 71.59 | 62.61 | 74.77 | 650 | Browns |
| 45 | 12 | Mario Edwards Jr. | 70.90 | 58.82 | 77.46 | 393 | Seahawks |
| 46 | 13 | D.J. Jones | 70.50 | 54.81 | 77.88 | 568 | Broncos |
| 47 | 14 | Folorunso Fatukasi | 69.66 | 54.93 | 77.08 | 415 | Jaguars |
| 48 | 15 | Harrison Phillips | 69.65 | 60.08 | 72.45 | 838 | Vikings |
| 49 | 16 | Andrew Billings | 69.47 | 61.76 | 71.32 | 506 | Bears |
| 50 | 17 | Armon Watts | 69.00 | 61.95 | 70.01 | 286 | Steelers |
| 51 | 18 | A'Shawn Robinson | 68.82 | 55.66 | 75.49 | 515 | Giants |
| 52 | 19 | Sheldon Rankins | 68.79 | 55.57 | 74.22 | 673 | Texans |
| 53 | 20 | Khyiris Tonga | 68.78 | 63.51 | 72.25 | 188 | Vikings |
| 54 | 21 | Morgan Fox | 68.57 | 51.79 | 75.59 | 437 | Chargers |
| 55 | 22 | Larry Ogunjobi | 68.50 | 49.54 | 77.45 | 816 | Steelers |
| 56 | 23 | Roy Lopez | 68.34 | 56.86 | 73.50 | 395 | Cardinals |
| 57 | 24 | Jarran Reed | 68.26 | 53.19 | 74.63 | 809 | Seahawks |
| 58 | 25 | Dante Stills | 68.10 | 54.71 | 74.83 | 533 | Cardinals |
| 59 | 26 | Bobby Brown III | 67.97 | 66.79 | 70.38 | 335 | Rams |
| 60 | 27 | Roy Robertson-Harris | 67.75 | 53.83 | 73.45 | 683 | Jaguars |
| 61 | 28 | Sebastian Joseph-Day | 67.75 | 49.78 | 77.82 | 623 | 49ers |
| 62 | 29 | Bryan Bresee | 67.32 | 49.67 | 74.92 | 539 | Saints |
| 63 | 30 | Poona Ford | 67.25 | 56.25 | 74.83 | 151 | Bills |
| 64 | 31 | Da'Shawn Hand | 67.01 | 68.45 | 69.64 | 219 | Dolphins |
| 65 | 32 | Tim Settle | 66.92 | 51.14 | 74.06 | 413 | Bills |
| 66 | 33 | Maurice Hurst | 66.89 | 58.90 | 75.46 | 302 | Browns |
| 67 | 34 | Bilal Nichols | 66.51 | 51.26 | 72.51 | 616 | Raiders |
| 68 | 35 | Cameron Young | 66.49 | 53.48 | 72.96 | 201 | Seahawks |
| 69 | 36 | Teair Tart | 66.46 | 55.01 | 73.36 | 378 | Texans |
| 70 | 37 | Maliek Collins | 66.43 | 56.21 | 70.06 | 780 | Texans |
| 71 | 38 | Shy Tuttle | 66.39 | 53.37 | 70.90 | 547 | Panthers |
| 72 | 39 | William Gholston | 66.13 | 46.51 | 75.04 | 286 | Buccaneers |
| 73 | 40 | Adam Gotsis | 66.03 | 50.01 | 73.33 | 427 | Jaguars |
| 74 | 41 | Kevin Givens | 65.84 | 48.13 | 75.44 | 454 | 49ers |
| 75 | 42 | Zacch Pickens | 65.63 | 48.87 | 72.64 | 264 | Bears |
| 76 | 43 | Khalen Saunders | 65.32 | 54.21 | 70.81 | 522 | Saints |
| 77 | 44 | Malcolm Roach | 65.28 | 57.99 | 71.55 | 290 | Saints |
| 78 | 45 | Davon Godchaux | 65.11 | 50.60 | 70.61 | 685 | Patriots |
| 79 | 46 | DaVon Hamilton | 64.98 | 51.79 | 74.20 | 190 | Jaguars |
| 80 | 47 | Taven Bryan | 64.88 | 53.59 | 68.93 | 343 | Colts |
| 81 | 48 | Solomon Thomas | 64.84 | 45.45 | 73.60 | 483 | Jets |
| 82 | 49 | Colby Wooden | 64.84 | 52.31 | 69.02 | 298 | Packers |
| 83 | 50 | Gervon Dexter Sr. | 64.79 | 49.10 | 71.09 | 433 | Bears |
| 84 | 51 | Quinton Jefferson | 64.68 | 46.52 | 74.09 | 468 | Jets |
| 85 | 52 | Naquan Jones | 64.50 | 45.30 | 78.14 | 171 | Cardinals |
| 86 | 53 | Khalil Davis | 64.47 | 45.40 | 78.48 | 481 | Texans |
| 87 | 54 | Justin Jones | 64.42 | 47.11 | 72.98 | 740 | Bears |
| 88 | 55 | Lawrence Guy Sr. | 64.42 | 40.53 | 77.06 | 522 | Patriots |
| 89 | 56 | Jerry Tillery | 64.41 | 57.60 | 65.56 | 504 | Raiders |
| 90 | 57 | Nathan Shepherd | 64.35 | 49.36 | 70.17 | 593 | Saints |
| 91 | 58 | Adam Butler | 64.13 | 47.22 | 71.43 | 526 | Raiders |
| 92 | 59 | Brent Urban | 63.82 | 48.31 | 72.44 | 309 | Ravens |
| 93 | 60 | Greg Gaines | 63.73 | 50.72 | 68.53 | 525 | Buccaneers |
| 94 | 61 | Kentavius Street | 63.63 | 46.08 | 73.13 | 267 | Falcons |
| 95 | 62 | Neville Gallimore | 63.31 | 48.75 | 71.50 | 304 | Cowboys |
| 96 | 63 | Tyler Lacy | 62.95 | 50.78 | 68.87 | 145 | Jaguars |
| 97 | 64 | Jonathan Bullard | 62.90 | 46.56 | 72.66 | 643 | Vikings |
| 98 | 65 | Levi Onwuzurike | 62.72 | 51.65 | 68.75 | 164 | Lions |
| 99 | 66 | Mike Purcell | 62.72 | 45.52 | 71.28 | 463 | Broncos |
| 100 | 67 | Javon Kinlaw | 62.59 | 50.78 | 72.08 | 544 | 49ers |
| 101 | 68 | LaCale London | 62.50 | 57.75 | 73.51 | 204 | Falcons |
| 102 | 69 | Abdullah Anderson | 62.43 | 57.65 | 69.38 | 113 | Commanders |
| 103 | 70 | Jonah Williams | 62.39 | 48.64 | 70.32 | 636 | Rams |
| 104 | 71 | John Jenkins | 62.10 | 45.19 | 71.46 | 595 | Raiders |

### Rotation/backup (58 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 105 | 1 | Marlon Tuipulotu | 61.85 | 50.00 | 71.26 | 178 | Eagles |
| 106 | 2 | DeShawn Williams | 61.83 | 44.24 | 70.28 | 443 | Panthers |
| 107 | 3 | John Cominsky | 61.81 | 43.60 | 70.95 | 663 | Lions |
| 108 | 4 | Austin Johnson | 61.81 | 46.73 | 70.34 | 641 | Chargers |
| 109 | 5 | Broderick Washington | 61.63 | 47.93 | 67.18 | 452 | Ravens |
| 110 | 6 | Jonathan Harris | 61.52 | 46.70 | 72.64 | 529 | Broncos |
| 111 | 7 | Montravius Adams | 61.35 | 48.89 | 68.34 | 427 | Steelers |
| 112 | 8 | DeMarvin Leal | 61.34 | 43.16 | 74.20 | 206 | Steelers |
| 113 | 9 | Johnathan Hankins | 61.25 | 43.13 | 72.80 | 387 | Cowboys |
| 114 | 10 | Matt Henningsen | 61.00 | 47.80 | 65.64 | 226 | Broncos |
| 115 | 11 | Kurt Hinish | 60.64 | 46.72 | 66.49 | 542 | Texans |
| 116 | 12 | Al Woods | 60.51 | 43.68 | 74.03 | 140 | Jets |
| 117 | 13 | Mike Pennel | 60.43 | 43.98 | 73.50 | 160 | Chiefs |
| 118 | 14 | Ta'Quon Graham | 60.41 | 54.54 | 63.69 | 364 | Falcons |
| 119 | 15 | Raekwon Davis | 60.27 | 48.54 | 64.80 | 531 | Dolphins |
| 120 | 16 | Tershawn Wharton | 60.24 | 48.97 | 67.12 | 525 | Chiefs |
| 121 | 17 | Jordan Phillips | 60.24 | 42.04 | 72.73 | 391 | Bills |
| 122 | 18 | Dean Lowry | 60.22 | 45.56 | 70.35 | 237 | Vikings |
| 123 | 19 | Linval Joseph | 59.85 | 39.12 | 76.66 | 189 | Bills |
| 124 | 20 | Nick Thurman | 59.64 | 44.26 | 71.61 | 368 | Panthers |
| 125 | 21 | Logan Hall | 59.42 | 48.06 | 62.83 | 600 | Buccaneers |
| 126 | 22 | Mazi Smith | 59.28 | 47.37 | 63.06 | 308 | Cowboys |
| 127 | 23 | Jonathan Ledbetter | 59.19 | 44.95 | 70.98 | 511 | Cardinals |
| 128 | 24 | John Ridgeway | 58.86 | 44.33 | 65.11 | 355 | Commanders |
| 129 | 25 | Derrick Nnadi | 58.85 | 43.36 | 65.01 | 507 | Chiefs |
| 130 | 26 | Leki Fotu | 58.58 | 44.70 | 66.60 | 297 | Cardinals |
| 131 | 27 | Isaiahh Loudermilk | 58.47 | 48.24 | 64.05 | 184 | Steelers |
| 132 | 28 | Larrell Murchison | 58.44 | 48.94 | 64.92 | 261 | Rams |
| 133 | 29 | Sheldon Day | 58.20 | 48.02 | 69.16 | 125 | Vikings |
| 134 | 30 | Jeremiah Ledbetter | 58.13 | 53.83 | 64.24 | 369 | Jaguars |
| 135 | 31 | Josh Tupou | 57.83 | 45.14 | 65.36 | 287 | Bengals |
| 136 | 32 | D.J. Davidson | 57.44 | 49.51 | 64.20 | 244 | Giants |
| 137 | 33 | Angelo Blackson | 57.37 | 41.25 | 67.49 | 215 | Jaguars |
| 138 | 34 | Jordan Elliott | 57.28 | 45.61 | 61.10 | 466 | Browns |
| 139 | 35 | Rakeem Nunez-Roches | 57.11 | 40.19 | 64.90 | 461 | Giants |
| 140 | 36 | Ben Stille | 56.66 | 48.83 | 68.74 | 134 | Cardinals |
| 141 | 37 | Zach Carter | 56.36 | 44.13 | 60.71 | 500 | Bengals |
| 142 | 38 | Benito Jones | 56.28 | 43.31 | 62.84 | 602 | Lions |
| 143 | 39 | Quinton Bohanna | 56.03 | 49.02 | 63.69 | 113 | Titans |
| 144 | 40 | Otito Ogbonnia | 55.97 | 45.58 | 67.92 | 223 | Chargers |
| 145 | 41 | Kyle Peko | 55.95 | 40.96 | 68.14 | 342 | Titans |
| 146 | 42 | Jaleel Johnson | 55.70 | 42.19 | 66.33 | 270 | Titans |
| 147 | 43 | Adetomiwa Adebawore | 55.15 | 46.34 | 64.71 | 132 | Colts |
| 148 | 44 | Marlon Davidson | 55.06 | 52.77 | 61.74 | 163 | Titans |
| 149 | 45 | Matt Dickerson | 54.76 | 42.65 | 65.11 | 206 | Chiefs |
| 150 | 46 | Eric Johnson | 53.12 | 43.46 | 58.32 | 265 | Colts |
| 151 | 47 | Tyson Alualu | 52.32 | 34.11 | 69.11 | 152 | Lions |
| 152 | 48 | LaBryan Ray | 51.83 | 40.52 | 55.21 | 356 | Panthers |
| 153 | 49 | Phil Hoskins | 51.55 | 45.34 | 64.02 | 124 | Cardinals |
| 154 | 50 | Jordon Riley | 51.50 | 43.42 | 61.54 | 135 | Giants |
| 155 | 51 | Scott Matlock | 50.64 | 41.62 | 57.38 | 266 | Chargers |
| 156 | 52 | Sam Roberts | 50.12 | 43.98 | 61.32 | 101 | Patriots |
| 157 | 53 | Albert Huggins | 46.55 | 37.61 | 55.77 | 317 | Falcons |
| 158 | 54 | TK McLendon Jr. | 45.82 | 43.27 | 54.66 | 101 | Titans |
| 159 | 55 | Phidarian Mathis | 45.16 | 46.23 | 50.45 | 202 | Commanders |
| 160 | 56 | Keondre Coburn | 45.00 | 55.87 | 49.40 | 107 | Titans |
| 161 | 57 | Mike Greene | 45.00 | 38.02 | 50.09 | 168 | Buccaneers |
| 162 | 58 | Siaki Ika | 45.00 | 39.43 | 46.54 | 103 | Browns |

## ED — Edge

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Bosa | 94.26 | 97.15 | 88.47 | 1022 | 49ers |
| 2 | 2 | Micah Parsons | 94.06 | 92.27 | 91.08 | 910 | Cowboys |
| 3 | 3 | Myles Garrett | 91.76 | 95.24 | 85.57 | 844 | Browns |
| 4 | 4 | T.J. Watt | 91.53 | 93.98 | 88.18 | 930 | Steelers |
| 5 | 5 | Will Anderson Jr. | 88.83 | 94.75 | 80.71 | 695 | Texans |
| 6 | 6 | Aidan Hutchinson | 88.28 | 93.26 | 80.79 | 1146 | Lions |
| 7 | 7 | Rashan Gary | 88.17 | 90.19 | 85.20 | 667 | Packers |
| 8 | 8 | Greg Rousseau | 87.54 | 91.96 | 81.61 | 660 | Bills |
| 9 | 9 | Maxx Crosby | 87.36 | 92.25 | 79.94 | 1080 | Raiders |
| 10 | 10 | Joey Bosa | 87.12 | 92.73 | 86.86 | 320 | Chargers |
| 11 | 11 | Khalil Mack | 86.72 | 86.47 | 84.68 | 934 | Chargers |
| 12 | 12 | Danielle Hunter | 84.38 | 81.88 | 83.85 | 1004 | Vikings |
| 13 | 13 | Jaelan Phillips | 83.56 | 87.33 | 81.29 | 366 | Dolphins |
| 14 | 14 | Trey Hendrickson | 83.41 | 76.93 | 84.35 | 742 | Bengals |
| 15 | 15 | DeMarcus Lawrence | 82.83 | 87.95 | 77.21 | 647 | Cowboys |
| 16 | 16 | Montez Sweat | 82.10 | 81.93 | 79.41 | 764 | Bears |
| 17 | 17 | Sam Williams | 80.99 | 73.54 | 82.52 | 314 | Cowboys |
| 18 | 18 | Bradley Chubb | 80.31 | 78.28 | 80.24 | 837 | Dolphins |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Von Miller | 79.12 | 65.80 | 87.46 | 298 | Bills |
| 20 | 2 | Malcolm Koonce | 79.09 | 70.26 | 84.55 | 501 | Raiders |
| 21 | 3 | Will McDonald IV | 78.85 | 67.17 | 84.44 | 183 | Jets |
| 22 | 4 | Tuli Tuipulotu | 78.85 | 77.93 | 75.30 | 852 | Chargers |
| 23 | 5 | Brandon Graham | 78.62 | 74.09 | 80.41 | 427 | Eagles |
| 24 | 6 | Alex Highsmith | 78.04 | 78.35 | 73.87 | 974 | Steelers |
| 25 | 7 | Haason Reddick | 78.00 | 65.38 | 82.44 | 910 | Eagles |
| 26 | 8 | Jermaine Johnson | 77.89 | 74.50 | 77.09 | 748 | Jets |
| 27 | 9 | Shaquil Barrett | 77.49 | 70.63 | 80.93 | 746 | Buccaneers |
| 28 | 10 | Za'Darius Smith | 76.87 | 74.38 | 77.80 | 603 | Browns |
| 29 | 11 | Josh Sweat | 76.31 | 72.00 | 75.50 | 875 | Eagles |
| 30 | 12 | Brian Burns | 75.85 | 64.86 | 79.79 | 814 | Panthers |
| 31 | 13 | Jadeveon Clowney | 75.01 | 78.27 | 70.73 | 747 | Ravens |
| 32 | 14 | George Karlaftis | 74.80 | 63.65 | 78.07 | 973 | Chiefs |
| 33 | 15 | Cameron Jordan | 74.42 | 70.40 | 73.42 | 770 | Saints |
| 34 | 16 | BJ Ojulari | 74.17 | 59.95 | 79.49 | 409 | Cardinals |
| 35 | 17 | Odafe Oweh | 74.14 | 77.23 | 69.28 | 491 | Ravens |

### Starter (57 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 36 | 1 | Jonathan Greenard | 73.68 | 68.51 | 76.60 | 697 | Texans |
| 37 | 2 | Arnold Ebiketie | 73.60 | 64.91 | 75.59 | 385 | Falcons |
| 38 | 3 | Boye Mafe | 73.43 | 68.91 | 72.90 | 808 | Seahawks |
| 39 | 4 | Matthew Judon | 72.65 | 59.33 | 83.73 | 184 | Patriots |
| 40 | 5 | Harold Landry III | 72.27 | 61.60 | 75.22 | 840 | Titans |
| 41 | 6 | Carl Granderson | 72.02 | 65.47 | 72.91 | 874 | Saints |
| 42 | 7 | Chase Young | 71.65 | 79.55 | 67.90 | 891 | 49ers |
| 43 | 8 | Lukas Van Ness | 71.45 | 62.04 | 73.56 | 444 | Packers |
| 44 | 9 | Marcus Davenport | 71.28 | 72.04 | 74.74 | 118 | Vikings |
| 45 | 10 | Samson Ebukam | 71.23 | 66.22 | 70.98 | 703 | Colts |
| 46 | 11 | Ogbo Okoronkwo | 71.10 | 59.63 | 76.35 | 459 | Browns |
| 47 | 12 | A.J. Epenesa | 70.93 | 62.43 | 73.61 | 438 | Bills |
| 48 | 13 | Dennis Gardeck | 70.85 | 53.51 | 79.71 | 510 | Cardinals |
| 49 | 14 | Markus Golden | 70.62 | 53.53 | 78.05 | 281 | Steelers |
| 50 | 15 | Calais Campbell | 70.43 | 47.16 | 81.77 | 712 | Falcons |
| 51 | 16 | Yaya Diaby | 70.37 | 61.04 | 72.43 | 590 | Buccaneers |
| 52 | 17 | Travon Walker | 70.20 | 64.80 | 70.37 | 869 | Jaguars |
| 53 | 18 | Jonathon Cooper | 70.04 | 64.26 | 70.81 | 836 | Broncos |
| 54 | 19 | Preston Smith | 70.03 | 58.72 | 73.61 | 838 | Packers |
| 55 | 20 | Dante Fowler Jr. | 69.98 | 59.57 | 73.34 | 275 | Cowboys |
| 56 | 21 | Nolan Smith | 69.92 | 60.81 | 71.82 | 203 | Eagles |
| 57 | 22 | Julian Okwara | 69.74 | 60.60 | 78.43 | 120 | Lions |
| 58 | 23 | John Franklin-Myers | 69.71 | 62.55 | 70.51 | 626 | Jets |
| 59 | 24 | Dorance Armstrong | 69.65 | 62.61 | 70.96 | 468 | Cowboys |
| 60 | 25 | Tyquan Lewis | 68.90 | 63.83 | 72.71 | 437 | Colts |
| 61 | 26 | Darrell Taylor | 68.62 | 54.70 | 74.22 | 522 | Seahawks |
| 62 | 27 | Dayo Odeyingbo | 68.44 | 62.60 | 69.54 | 623 | Colts |
| 63 | 28 | Kayvon Thibodeaux | 68.35 | 64.85 | 67.62 | 981 | Giants |
| 64 | 29 | Leonard Floyd | 68.01 | 54.84 | 72.63 | 627 | Bills |
| 65 | 30 | Jacob Martin | 67.80 | 58.17 | 71.23 | 192 | Colts |
| 66 | 31 | Sam Hubbard | 67.78 | 59.44 | 70.94 | 713 | Bengals |
| 67 | 32 | Kwity Paye | 67.73 | 67.15 | 66.30 | 700 | Colts |
| 68 | 33 | Denico Autry | 67.24 | 47.33 | 77.82 | 767 | Titans |
| 69 | 34 | Victor Dimukeje | 67.20 | 59.38 | 72.16 | 385 | Cardinals |
| 70 | 35 | Jerry Hughes | 67.00 | 50.87 | 73.59 | 474 | Texans |
| 71 | 36 | Azeez Ojulari | 66.99 | 59.73 | 75.02 | 424 | Giants |
| 72 | 37 | Byron Young | 66.87 | 57.97 | 68.64 | 1021 | Rams |
| 73 | 38 | Joe Tryon-Shoyinka | 66.64 | 61.14 | 66.14 | 625 | Buccaneers |
| 74 | 39 | Zack Baun | 66.58 | 54.85 | 70.23 | 303 | Saints |
| 75 | 40 | Deatrich Wise Jr. | 66.24 | 59.13 | 67.49 | 615 | Patriots |
| 76 | 41 | Zach Harrison | 66.13 | 60.79 | 66.50 | 343 | Falcons |
| 77 | 42 | Arden Key | 65.92 | 62.99 | 63.71 | 727 | Titans |
| 78 | 43 | Charles Omenihu | 65.88 | 55.91 | 70.22 | 502 | Chiefs |
| 79 | 44 | Carl Lawson | 65.83 | 54.98 | 74.30 | 101 | Jets |
| 80 | 45 | Myles Murphy | 65.73 | 58.99 | 66.05 | 304 | Bengals |
| 81 | 46 | Kingsley Enagbare | 65.56 | 60.68 | 64.65 | 493 | Packers |
| 82 | 47 | Yannick Ngakoue | 65.25 | 52.45 | 72.16 | 592 | Bears |
| 83 | 48 | Randy Gregory | 64.83 | 54.20 | 71.96 | 488 | 49ers |
| 84 | 49 | Clelin Ferrell | 64.48 | 60.36 | 63.55 | 471 | 49ers |
| 85 | 50 | Derek Barnett | 63.76 | 61.27 | 66.65 | 415 | Texans |
| 86 | 51 | Lorenzo Carter | 63.70 | 56.34 | 65.03 | 431 | Falcons |
| 87 | 52 | Emmanuel Ogbah | 63.65 | 56.78 | 67.40 | 286 | Dolphins |
| 88 | 53 | Shaq Lawson | 63.37 | 53.09 | 67.05 | 354 | Bills |
| 89 | 54 | Melvin Ingram III | 63.16 | 48.87 | 75.28 | 150 | Dolphins |
| 90 | 55 | Amare Barno | 62.79 | 59.40 | 66.89 | 189 | Panthers |
| 91 | 56 | Bud Dupree | 62.35 | 52.94 | 67.89 | 725 | Falcons |
| 92 | 57 | Mike Danna | 62.14 | 58.96 | 61.28 | 932 | Chiefs |

### Rotation/backup (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 93 | 1 | D.J. Wonnum | 61.77 | 56.77 | 62.21 | 826 | Vikings |
| 94 | 2 | Micheal Clemons | 61.74 | 59.73 | 59.90 | 368 | Jets |
| 95 | 3 | Anthony Nelson | 61.45 | 56.92 | 60.31 | 446 | Buccaneers |
| 96 | 4 | DeMarcus Walker | 61.16 | 50.98 | 63.78 | 714 | Bears |
| 97 | 5 | Cam Sample | 61.08 | 61.23 | 57.89 | 375 | Bengals |
| 98 | 6 | Tyree Wilson | 60.83 | 57.26 | 59.04 | 493 | Raiders |
| 99 | 7 | Myjai Sanders | 60.82 | 58.10 | 64.83 | 189 | Texans |
| 100 | 8 | Jesse Luketa | 60.39 | 54.70 | 64.91 | 132 | Cardinals |
| 101 | 9 | Yetur Gross-Matos | 60.17 | 59.44 | 59.53 | 465 | Panthers |
| 102 | 10 | Romeo Okwara | 59.91 | 55.75 | 64.60 | 330 | Lions |
| 103 | 11 | K'Lavon Chaisson | 59.80 | 57.61 | 59.84 | 283 | Jaguars |
| 104 | 12 | Dawuane Smoot | 59.46 | 53.24 | 62.68 | 340 | Jaguars |
| 105 | 13 | Drake Jackson | 59.43 | 58.68 | 62.01 | 199 | 49ers |
| 106 | 14 | Felix Anudike-Uzomah | 58.81 | 57.34 | 55.62 | 225 | Chiefs |
| 107 | 15 | Marquis Haynes Sr. | 58.63 | 53.14 | 63.03 | 142 | Panthers |
| 108 | 16 | Rasheem Green | 58.47 | 52.13 | 58.83 | 385 | Bears |
| 109 | 17 | Keion White | 58.22 | 59.28 | 54.33 | 522 | Patriots |
| 110 | 18 | Charles Harris | 58.10 | 53.87 | 61.95 | 291 | Lions |
| 111 | 19 | Tanoh Kpassagnon | 58.04 | 54.19 | 58.79 | 406 | Saints |
| 112 | 20 | Justin Hollins | 57.97 | 52.63 | 62.36 | 197 | Chargers |
| 113 | 21 | Alex Wright | 57.64 | 56.37 | 54.32 | 407 | Browns |
| 114 | 22 | Tavius Robinson | 56.85 | 54.30 | 54.39 | 338 | Ravens |
| 115 | 23 | Casey Toohill | 56.76 | 52.11 | 56.67 | 494 | Commanders |
| 116 | 24 | Josh Paschal | 56.70 | 58.30 | 55.27 | 510 | Lions |
| 117 | 25 | Derick Hall | 56.30 | 53.51 | 54.00 | 308 | Seahawks |
| 118 | 26 | Jihad Ward | 55.27 | 44.15 | 58.81 | 661 | Giants |
| 119 | 27 | Rashad Weaver | 55.06 | 53.43 | 56.20 | 240 | Titans |
| 120 | 28 | Chris Rumph II | 54.55 | 55.23 | 56.29 | 103 | Chargers |
| 121 | 29 | James Smith-Williams | 54.54 | 52.96 | 54.86 | 418 | Commanders |
| 122 | 30 | KJ Henry | 53.36 | 55.43 | 55.66 | 281 | Commanders |
| 123 | 31 | Malik Herring | 52.17 | 50.12 | 55.26 | 213 | Chiefs |
| 124 | 32 | Dylan Horton | 51.74 | 54.12 | 52.85 | 175 | Texans |
| 125 | 33 | DJ Johnson | 51.73 | 53.32 | 50.42 | 231 | Panthers |
| 126 | 34 | Dominique Robinson | 51.50 | 49.55 | 52.31 | 242 | Bears |
| 127 | 35 | Ronnie Perkins | 50.38 | 50.47 | 52.55 | 149 | Broncos |
| 128 | 36 | Andre Jones Jr. | 48.64 | 52.73 | 49.59 | 171 | Commanders |
| 129 | 37 | Jeremiah Moon | 46.47 | 50.33 | 49.47 | 102 | Ravens |

## G — Guard

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Lindstrom | 93.44 | 87.60 | 93.16 | 1066 | Falcons |
| 2 | 2 | David Edwards | 93.13 | 90.20 | 90.91 | 194 | Bills |
| 3 | 3 | Quinn Meinerz | 91.16 | 83.70 | 91.97 | 1038 | Broncos |
| 4 | 4 | Kevin Dotson | 90.96 | 84.40 | 91.17 | 939 | Rams |
| 5 | 5 | Sam Cosmi | 87.61 | 80.60 | 88.11 | 1103 | Commanders |
| 6 | 6 | Tyler Smith | 85.29 | 74.40 | 88.38 | 1037 | Cowboys |
| 7 | 7 | Robert Hunt | 84.99 | 76.40 | 86.55 | 608 | Dolphins |
| 8 | 8 | Graham Glasgow | 83.65 | 74.90 | 85.32 | 1262 | Lions |
| 9 | 9 | Trey Smith | 83.11 | 74.60 | 84.61 | 1374 | Chiefs |
| 10 | 10 | Greg Van Roten | 81.98 | 75.30 | 82.26 | 1025 | Raiders |
| 11 | 11 | Isaac Seumalo | 81.87 | 73.90 | 83.02 | 1104 | Steelers |
| 12 | 12 | Wyatt Teller | 81.58 | 72.70 | 83.33 | 1254 | Browns |
| 13 | 13 | Joe Thuney | 81.49 | 74.90 | 81.72 | 1212 | Chiefs |
| 14 | 14 | Teven Jenkins | 81.00 | 72.60 | 82.43 | 731 | Bears |

### Good (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Quenton Nelson | 79.17 | 70.80 | 80.59 | 1141 | Colts |
| 16 | 2 | Kevin Zeitler | 79.16 | 71.60 | 80.03 | 1101 | Ravens |
| 17 | 3 | Landon Dickerson | 78.50 | 69.40 | 80.40 | 1102 | Eagles |
| 18 | 4 | Halapoulivaati Vaitai | 77.64 | 68.30 | 79.70 | 192 | Lions |
| 19 | 5 | Brandon Scherff | 77.24 | 67.30 | 79.70 | 1079 | Jaguars |
| 20 | 6 | Zack Martin | 76.97 | 68.00 | 78.78 | 1003 | Cowboys |
| 21 | 7 | Ben Cleveland | 76.58 | 65.80 | 79.60 | 171 | Ravens |
| 22 | 8 | Nate Herbig | 76.36 | 67.60 | 78.03 | 156 | Steelers |
| 23 | 9 | Shaq Mason | 76.09 | 65.60 | 78.91 | 1221 | Texans |
| 24 | 10 | Joel Bitonio | 75.82 | 67.90 | 76.94 | 1107 | Browns |
| 25 | 11 | Sidy Sow | 75.74 | 64.40 | 79.14 | 772 | Patriots |
| 26 | 12 | Alex Cappa | 75.69 | 64.90 | 78.72 | 1066 | Bengals |
| 27 | 13 | Will Hernandez | 75.17 | 66.20 | 76.99 | 1109 | Cardinals |
| 28 | 14 | Cole Strange | 74.79 | 64.60 | 77.41 | 564 | Patriots |
| 29 | 15 | Nick Allegretti | 74.25 | 65.80 | 75.71 | 253 | Chiefs |

### Starter (45 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Mark Glowinski | 73.91 | 63.30 | 76.82 | 521 | Giants |
| 31 | 2 | Elgton Jenkins | 73.72 | 63.80 | 76.16 | 1019 | Packers |
| 32 | 3 | Cade Mays | 72.79 | 58.30 | 78.28 | 434 | Panthers |
| 33 | 4 | Will Fries | 72.54 | 61.20 | 75.94 | 1125 | Colts |
| 34 | 5 | Ben Powers | 72.32 | 61.50 | 75.36 | 1068 | Broncos |
| 35 | 6 | Ben Bartch | 72.29 | 61.20 | 75.52 | 240 | 49ers |
| 36 | 7 | Jonah Jackson | 72.04 | 61.00 | 75.23 | 881 | Lions |
| 37 | 8 | James Daniels | 72.01 | 61.10 | 75.11 | 1010 | Steelers |
| 38 | 9 | Dylan Parham | 71.58 | 60.40 | 74.86 | 1042 | Raiders |
| 39 | 10 | Cordell Volson | 71.27 | 58.30 | 75.75 | 1087 | Bengals |
| 40 | 11 | Ezra Cleveland | 71.23 | 59.50 | 74.88 | 749 | Jaguars |
| 41 | 12 | Damien Lewis | 71.14 | 59.60 | 74.66 | 926 | Seahawks |
| 42 | 13 | Steve Avila | 71.12 | 60.50 | 74.04 | 1205 | Rams |
| 43 | 14 | Ed Ingram | 70.83 | 59.50 | 74.21 | 958 | Vikings |
| 44 | 15 | Matthew Bergeron | 70.75 | 59.10 | 74.35 | 1127 | Falcons |
| 45 | 16 | Robert Jones | 70.67 | 57.60 | 75.22 | 479 | Dolphins |
| 46 | 17 | Gabe Jackson | 70.21 | 58.10 | 74.12 | 194 | Panthers |
| 47 | 18 | Zion Johnson | 69.71 | 57.60 | 73.62 | 1006 | Chargers |
| 48 | 19 | John Simpson | 69.52 | 56.30 | 74.16 | 1242 | Ravens |
| 49 | 20 | Aaron Stinnie | 69.15 | 56.50 | 73.41 | 851 | Buccaneers |
| 50 | 21 | Jon Runyan | 68.79 | 56.50 | 72.81 | 1009 | Packers |
| 51 | 22 | Max Garcia | 68.25 | 54.20 | 73.45 | 320 | Saints |
| 52 | 23 | Kayode Awosika | 68.15 | 55.70 | 72.28 | 370 | Lions |
| 53 | 24 | O'Cyrus Torrence | 68.01 | 54.90 | 72.59 | 1307 | Bills |
| 54 | 25 | Connor McGovern | 67.91 | 55.40 | 72.09 | 1278 | Bills |
| 55 | 26 | Dalton Risner | 67.84 | 57.10 | 70.84 | 745 | Vikings |
| 56 | 27 | Matt Feiler | 67.71 | 54.70 | 72.22 | 386 | Buccaneers |
| 57 | 28 | Saahdiq Charles | 67.32 | 55.50 | 71.03 | 643 | Commanders |
| 58 | 29 | Spencer Burford | 66.91 | 50.30 | 73.81 | 900 | 49ers |
| 59 | 30 | Aaron Banks | 66.62 | 52.80 | 71.67 | 1042 | 49ers |
| 60 | 31 | Anthony Bradford | 66.60 | 51.70 | 72.37 | 659 | Seahawks |
| 61 | 32 | Phil Haynes | 66.35 | 51.90 | 71.81 | 437 | Seahawks |
| 62 | 33 | Laken Tomlinson | 66.16 | 55.00 | 69.44 | 1099 | Jets |
| 63 | 34 | Nate Davis | 66.14 | 52.90 | 70.80 | 663 | Bears |
| 64 | 35 | Wes Schweitzer | 65.40 | 52.70 | 69.70 | 149 | Jets |
| 65 | 36 | Jake Hanson | 65.27 | 51.80 | 70.08 | 244 | Jets |
| 66 | 37 | Sua Opeta | 65.08 | 52.90 | 69.04 | 530 | Eagles |
| 67 | 38 | Cesar Ruiz | 64.88 | 51.20 | 69.83 | 1050 | Saints |
| 68 | 39 | Jordan McFadden | 64.87 | 52.00 | 69.28 | 163 | Chargers |
| 69 | 40 | Xavier Newman | 64.40 | 47.80 | 71.30 | 292 | Jets |
| 70 | 41 | Elijah Wilkinson | 63.68 | 46.20 | 71.16 | 501 | Cardinals |
| 71 | 42 | Tyler Shatley | 63.33 | 47.50 | 69.72 | 518 | Jaguars |
| 72 | 43 | Lester Cotton | 62.52 | 46.40 | 69.10 | 616 | Dolphins |
| 73 | 44 | Marcus McKethan | 62.37 | 45.40 | 69.52 | 378 | Giants |
| 74 | 45 | Austin Corbett | 62.35 | 47.90 | 67.81 | 257 | Panthers |

### Rotation/backup (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 75 | 1 | Ja'Tyre Carter | 61.95 | 49.00 | 66.42 | 175 | Bears |
| 76 | 2 | Cody Whitehair | 61.82 | 45.00 | 68.86 | 787 | Bears |
| 77 | 3 | Ben Bredeson | 59.97 | 42.50 | 67.45 | 1014 | Giants |
| 78 | 4 | Justin Pugh | 59.44 | 41.60 | 67.17 | 763 | Giants |
| 79 | 5 | Royce Newman | 59.27 | 40.90 | 67.35 | 186 | Packers |
| 80 | 6 | Chris Paul | 57.99 | 38.80 | 66.61 | 439 | Commanders |
| 81 | 7 | Dennis Daley | 57.73 | 38.40 | 66.45 | 144 | Cardinals |
| 82 | 8 | Nash Jensen | 55.83 | 34.70 | 65.75 | 302 | Panthers |
| 83 | 9 | Atonio Mafi | 54.70 | 32.30 | 65.47 | 458 | Patriots |
| 84 | 10 | Chandler Zavala | 51.58 | 26.20 | 64.33 | 374 | Panthers |

## HB — Running Back

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | De'Von Achane | 93.39 | 92.10 | 90.08 | 187 | Dolphins |
| 2 | 2 | Breece Hall | 84.80 | 82.40 | 82.24 | 336 | Jets |
| 3 | 3 | Derrick Henry | 84.65 | 90.10 | 76.85 | 175 | Titans |
| 4 | 4 | Jaylen Warren | 84.10 | 78.50 | 83.66 | 273 | Steelers |
| 5 | 5 | Christian McCaffrey | 83.54 | 90.30 | 74.87 | 525 | 49ers |
| 6 | 6 | Aaron Jones | 81.28 | 80.90 | 77.36 | 202 | Packers |
| 7 | 7 | James Conner | 81.27 | 89.20 | 71.81 | 201 | Cardinals |
| 8 | 8 | Tony Pollard | 81.15 | 77.50 | 79.42 | 439 | Cowboys |
| 9 | 9 | Tyjae Spears | 80.98 | 77.00 | 79.47 | 303 | Titans |
| 10 | 10 | Raheem Mostert | 80.64 | 84.70 | 73.76 | 276 | Dolphins |
| 11 | 11 | Kenneth Walker III | 80.30 | 83.50 | 74.00 | 200 | Seahawks |
| 12 | 12 | Jahmyr Gibbs | 80.23 | 76.30 | 78.69 | 372 | Lions |
| 13 | 13 | Tyler Allgeier | 80.02 | 82.90 | 73.93 | 108 | Falcons |

### Good (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Kyren Williams | 79.02 | 80.40 | 73.93 | 341 | Rams |
| 15 | 2 | Khalil Herbert | 78.92 | 77.90 | 75.44 | 146 | Bears |
| 16 | 3 | Jonathan Taylor | 78.61 | 74.90 | 76.92 | 178 | Colts |
| 17 | 4 | Emari Demercado | 77.92 | 73.40 | 76.77 | 164 | Cardinals |
| 18 | 5 | Travis Etienne Jr. | 77.82 | 77.00 | 74.20 | 404 | Jaguars |
| 19 | 6 | Rhamondre Stevenson | 77.26 | 71.30 | 77.07 | 251 | Patriots |
| 20 | 7 | Alvin Kamara | 76.56 | 74.10 | 74.04 | 269 | Saints |
| 21 | 8 | Isiah Pacheco | 76.55 | 79.90 | 70.15 | 365 | Chiefs |
| 22 | 9 | Devin Singletary | 76.39 | 75.10 | 73.08 | 329 | Texans |
| 23 | 10 | Bijan Robinson | 76.38 | 69.10 | 77.06 | 421 | Falcons |
| 24 | 11 | David Montgomery | 75.99 | 79.60 | 69.41 | 209 | Lions |
| 25 | 12 | Najee Harris | 75.31 | 76.20 | 70.55 | 223 | Steelers |
| 26 | 13 | Gus Edwards | 75.28 | 73.70 | 72.17 | 180 | Ravens |
| 27 | 14 | AJ Dillon | 74.90 | 75.70 | 70.20 | 216 | Packers |
| 28 | 15 | James Cook | 74.63 | 71.70 | 72.42 | 343 | Bills |

### Starter (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 29 | 1 | Kareem Hunt | 73.73 | 69.80 | 72.18 | 151 | Browns |
| 30 | 2 | Saquon Barkley | 73.23 | 70.20 | 71.09 | 306 | Giants |
| 31 | 3 | Chuba Hubbard | 72.80 | 77.20 | 65.70 | 295 | Panthers |
| 32 | 4 | Brian Robinson | 72.78 | 75.30 | 66.94 | 219 | Commanders |
| 33 | 5 | Josh Jacobs | 72.28 | 65.00 | 72.96 | 238 | Raiders |
| 34 | 6 | Javonte Williams | 72.19 | 63.30 | 73.95 | 170 | Broncos |
| 35 | 7 | Samaje Perine | 72.02 | 71.30 | 68.33 | 218 | Broncos |
| 36 | 8 | Zach Charbonnet | 71.96 | 69.80 | 69.23 | 276 | Seahawks |
| 37 | 9 | Ty Chandler | 71.80 | 76.30 | 64.63 | 122 | Vikings |
| 38 | 10 | Justice Hill | 71.70 | 70.30 | 68.47 | 277 | Ravens |
| 39 | 11 | Austin Ekeler | 71.66 | 60.20 | 75.13 | 349 | Chargers |
| 40 | 12 | Joe Mixon | 71.06 | 71.10 | 66.87 | 373 | Bengals |
| 41 | 13 | D'Andre Swift | 70.99 | 66.50 | 69.82 | 301 | Eagles |
| 42 | 14 | Ezekiel Elliott | 70.90 | 67.60 | 68.94 | 247 | Patriots |
| 43 | 15 | Jerome Ford | 70.70 | 68.30 | 68.13 | 349 | Browns |
| 44 | 16 | Antonio Gibson | 70.62 | 66.60 | 69.13 | 318 | Commanders |
| 45 | 17 | Zack Moss | 70.61 | 66.90 | 68.92 | 233 | Colts |
| 46 | 18 | Miles Sanders | 70.51 | 60.80 | 72.81 | 211 | Panthers |
| 47 | 19 | Michael Carter | 70.12 | 60.80 | 72.17 | 151 | Cardinals |
| 48 | 20 | Dameon Pierce | 69.47 | 65.90 | 67.68 | 111 | Texans |
| 49 | 21 | Clyde Edwards-Helaire | 69.37 | 65.50 | 67.78 | 155 | Chiefs |
| 50 | 22 | Alexander Mattison | 68.78 | 61.80 | 69.27 | 278 | Vikings |
| 51 | 23 | Rico Dowdle | 68.77 | 66.60 | 66.05 | 122 | Cowboys |
| 52 | 24 | Roschon Johnson | 68.45 | 65.80 | 66.05 | 190 | Bears |
| 53 | 25 | Rachaad White | 68.39 | 67.50 | 64.82 | 504 | Buccaneers |
| 54 | 26 | Latavius Murray | 67.36 | 65.90 | 64.16 | 198 | Bills |
| 55 | 27 | D'Ernest Johnson | 67.28 | 57.50 | 69.63 | 108 | Jaguars |
| 56 | 28 | Jerick McKinnon | 66.16 | 63.10 | 64.04 | 190 | Chiefs |
| 57 | 29 | Kenneth Gainwell | 65.30 | 55.30 | 67.80 | 232 | Eagles |
| 58 | 30 | Patrick Taylor | 65.05 | 60.40 | 63.98 | 136 | Packers |
| 59 | 31 | Jamaal Williams | 63.93 | 59.40 | 62.79 | 120 | Saints |
| 60 | 32 | Matt Breida | 62.45 | 52.40 | 64.98 | 156 | Giants |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 61 | 1 | Ameer Abdullah | 61.88 | 55.50 | 61.97 | 143 | Raiders |
| 62 | 2 | Joshua Kelley | 59.20 | 52.00 | 59.84 | 173 | Chargers |

## LB — Linebacker

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Demario Davis | 85.41 | 89.60 | 78.65 | 1074 | Saints |
| 2 | 2 | Fred Warner | 85.11 | 90.00 | 77.89 | 1200 | 49ers |
| 3 | 3 | Tyrel Dodson | 82.88 | 90.20 | 79.14 | 589 | Bills |
| 4 | 4 | Jahlani Tavai | 82.78 | 86.60 | 77.64 | 838 | Patriots |
| 5 | 5 | C.J. Mosley | 80.87 | 82.90 | 75.55 | 1127 | Jets |
| 6 | 6 | Jalen Reeves-Maybin | 80.73 | 88.60 | 76.61 | 121 | Lions |
| 7 | 7 | Bobby Wagner | 80.48 | 82.40 | 75.23 | 1170 | Seahawks |
| 8 | 8 | Leo Chenal | 80.33 | 84.00 | 73.71 | 527 | Chiefs |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Luke Masterson | 79.93 | 86.90 | 75.17 | 182 | Raiders |
| 10 | 2 | Quincy Williams | 79.79 | 81.10 | 75.53 | 1092 | Jets |
| 11 | 3 | Bobby Okereke | 79.63 | 79.00 | 75.88 | 1128 | Giants |
| 12 | 4 | Roquan Smith | 79.15 | 77.90 | 75.81 | 1192 | Ravens |
| 13 | 5 | T.J. Edwards | 78.42 | 79.60 | 73.66 | 1042 | Bears |
| 14 | 6 | Mack Wilson Sr. | 78.39 | 81.50 | 74.90 | 305 | Patriots |
| 15 | 7 | Blake Cashman | 77.59 | 82.10 | 75.42 | 746 | Texans |
| 16 | 8 | Devin Lloyd | 77.51 | 78.10 | 74.18 | 966 | Jaguars |
| 17 | 9 | Ernest Jones | 75.87 | 78.80 | 71.42 | 988 | Rams |
| 18 | 10 | Kaden Elliss | 75.60 | 75.40 | 71.57 | 1082 | Falcons |
| 19 | 11 | Robert Spillane | 75.51 | 77.10 | 71.55 | 1100 | Raiders |
| 20 | 12 | Foyesade Oluokun | 75.41 | 75.20 | 71.39 | 1110 | Jaguars |
| 21 | 13 | Frankie Luvu | 75.33 | 80.00 | 68.84 | 989 | Panthers |
| 22 | 14 | Ivan Pace Jr. | 75.09 | 77.10 | 69.59 | 704 | Vikings |

### Starter (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Lavonte David | 73.90 | 72.30 | 71.79 | 1086 | Buccaneers |
| 24 | 2 | Jordan Hicks | 73.74 | 74.60 | 70.97 | 813 | Vikings |
| 25 | 3 | Jeremiah Owusu-Koramoah | 73.62 | 76.80 | 69.69 | 802 | Browns |
| 26 | 4 | Elandon Roberts | 73.57 | 72.20 | 70.31 | 622 | Steelers |
| 27 | 5 | Dre Greenlaw | 73.06 | 73.10 | 72.20 | 999 | 49ers |
| 28 | 6 | Patrick Queen | 72.65 | 73.00 | 68.25 | 1246 | Ravens |
| 29 | 7 | Eric Kendricks | 72.44 | 72.30 | 69.73 | 847 | Chargers |
| 30 | 8 | Malik Harrison | 71.75 | 74.30 | 68.74 | 226 | Ravens |
| 31 | 9 | Isaiah Simmons | 70.66 | 68.90 | 67.66 | 378 | Giants |
| 32 | 10 | Sione Takitaki | 70.38 | 70.70 | 68.35 | 608 | Browns |
| 33 | 11 | Nicholas Morrow | 70.35 | 68.10 | 68.17 | 898 | Eagles |
| 34 | 12 | Nate Landman | 70.02 | 72.00 | 69.93 | 809 | Falcons |
| 35 | 13 | Alex Anzalone | 69.91 | 69.80 | 66.40 | 1189 | Lions |
| 36 | 14 | Duke Riley | 69.88 | 71.50 | 65.52 | 473 | Dolphins |
| 37 | 15 | Jamin Davis | 68.97 | 67.60 | 68.16 | 742 | Commanders |
| 38 | 16 | Drue Tranquill | 68.95 | 68.40 | 65.74 | 721 | Chiefs |
| 39 | 17 | Jack Gibbens | 68.50 | 71.20 | 68.78 | 628 | Titans |
| 40 | 18 | Andre Smith | 68.06 | 77.10 | 69.89 | 113 | Falcons |
| 41 | 19 | Jack Sanborn | 68.02 | 67.30 | 67.65 | 412 | Bears |
| 42 | 20 | Jerome Baker | 67.79 | 66.60 | 66.57 | 713 | Dolphins |
| 43 | 21 | E.J. Speed | 67.77 | 65.00 | 66.54 | 730 | Colts |
| 44 | 22 | Anthony Walker Jr. | 67.53 | 71.10 | 68.34 | 454 | Browns |
| 45 | 23 | Matt Milano | 67.51 | 70.90 | 67.75 | 211 | Bills |
| 46 | 24 | Ja'Whaun Bentley | 67.40 | 65.80 | 64.99 | 984 | Patriots |
| 47 | 25 | Azeez Al-Shaair | 67.40 | 64.70 | 67.29 | 1101 | Titans |
| 48 | 26 | Zach Cunningham | 67.31 | 67.60 | 68.25 | 787 | Eagles |
| 49 | 27 | Josey Jewell | 66.79 | 67.20 | 66.95 | 796 | Broncos |
| 50 | 28 | K.J. Britt | 66.73 | 70.50 | 69.65 | 252 | Buccaneers |
| 51 | 29 | Nick Niemann | 66.56 | 71.60 | 69.34 | 247 | Chargers |
| 52 | 30 | Leighton Vander Esch | 66.56 | 65.50 | 69.86 | 269 | Cowboys |
| 53 | 31 | Damone Clark | 66.52 | 62.70 | 67.46 | 834 | Cowboys |
| 54 | 32 | David Long Jr. | 66.40 | 62.50 | 67.68 | 899 | Dolphins |
| 55 | 33 | Logan Wilson | 66.20 | 62.60 | 65.80 | 1068 | Bengals |
| 56 | 34 | Germaine Pratt | 66.16 | 63.30 | 64.88 | 975 | Bengals |
| 57 | 35 | De'Vondre Campbell | 66.04 | 65.10 | 65.83 | 690 | Packers |
| 58 | 36 | Micah McFadden | 65.81 | 65.60 | 63.14 | 736 | Giants |
| 59 | 37 | Christian Harris | 65.51 | 65.00 | 63.51 | 869 | Texans |
| 60 | 38 | Deion Jones | 65.35 | 66.30 | 64.47 | 313 | Panthers |
| 61 | 39 | Alex Singleton | 64.99 | 61.20 | 64.14 | 1089 | Broncos |
| 62 | 40 | Cole Holcomb | 64.71 | 65.50 | 67.57 | 447 | Steelers |
| 63 | 41 | Quay Walker | 64.46 | 58.50 | 64.88 | 973 | Packers |
| 64 | 42 | Zaire Franklin | 64.43 | 60.90 | 63.10 | 1090 | Colts |
| 65 | 43 | Terrel Bernard | 64.41 | 65.90 | 63.30 | 1031 | Bills |
| 66 | 44 | Derrick Barnes | 64.19 | 60.70 | 63.71 | 791 | Lions |
| 67 | 45 | Nick Bolton | 63.72 | 59.40 | 65.09 | 708 | Chiefs |
| 68 | 46 | Oren Burks | 63.48 | 58.90 | 64.43 | 433 | 49ers |
| 69 | 47 | Khaleke Hudson | 63.41 | 64.60 | 66.49 | 405 | Commanders |
| 70 | 48 | Kyzir White | 63.23 | 58.90 | 64.88 | 708 | Cardinals |
| 71 | 49 | Divine Deablo | 62.86 | 60.70 | 65.53 | 771 | Raiders |
| 72 | 50 | Nakobe Dean | 62.71 | 61.70 | 69.52 | 182 | Eagles |
| 73 | 51 | Pete Werner | 62.46 | 57.50 | 64.15 | 919 | Saints |

### Rotation/backup (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 74 | 1 | SirVocea Dennis | 61.83 | 64.70 | 62.62 | 104 | Buccaneers |
| 75 | 2 | Jack Campbell | 61.64 | 52.10 | 63.84 | 745 | Lions |
| 76 | 3 | Denzel Perryman | 61.54 | 56.40 | 64.14 | 633 | Texans |
| 77 | 4 | Willie Gay | 61.51 | 55.90 | 63.23 | 698 | Chiefs |
| 78 | 5 | Kwon Alexander | 61.17 | 59.20 | 63.21 | 362 | Steelers |
| 79 | 6 | Malcolm Rodriguez | 61.10 | 57.40 | 61.60 | 162 | Lions |
| 80 | 7 | Tremaine Edmunds | 61.00 | 56.60 | 62.31 | 876 | Bears |
| 81 | 8 | Jordyn Brooks | 60.87 | 57.30 | 59.86 | 802 | Seahawks |
| 82 | 9 | Devin Bush | 60.77 | 57.50 | 63.30 | 251 | Seahawks |
| 83 | 10 | Cody Barton | 60.68 | 53.90 | 64.57 | 844 | Commanders |
| 84 | 11 | Krys Barnes | 60.64 | 60.10 | 63.70 | 408 | Cardinals |
| 85 | 12 | Isaiah McDuffie | 59.76 | 57.10 | 63.55 | 551 | Packers |
| 86 | 13 | Troy Dye | 59.08 | 60.70 | 64.23 | 112 | Vikings |
| 87 | 14 | Dorian Williams | 59.00 | 52.90 | 61.83 | 238 | Bills |
| 88 | 15 | David Mayo | 58.55 | 55.70 | 63.64 | 349 | Commanders |
| 89 | 16 | Eric Wilson | 58.49 | 56.50 | 64.29 | 144 | Packers |
| 90 | 17 | Owen Pappoe | 58.17 | 65.30 | 63.03 | 114 | Cardinals |
| 91 | 18 | Kenneth Murray Jr. | 57.89 | 52.10 | 59.73 | 968 | Chargers |
| 92 | 19 | Tony Fields II | 57.77 | 50.60 | 60.58 | 253 | Browns |
| 93 | 20 | Devin White | 57.71 | 47.40 | 60.90 | 933 | Buccaneers |
| 94 | 21 | Myles Jack | 57.61 | 55.30 | 62.34 | 157 | Steelers |
| 95 | 22 | Mykal Walker | 56.33 | 52.30 | 59.26 | 321 | Steelers |
| 96 | 23 | Kamu Grugier-Hill | 55.55 | 48.60 | 58.86 | 403 | Panthers |
| 97 | 24 | Segun Olubi | 55.27 | 56.20 | 62.34 | 115 | Colts |
| 98 | 25 | Mark Robinson | 55.26 | 51.10 | 63.91 | 173 | Steelers |
| 99 | 26 | Jack Cochrane | 55.25 | 54.00 | 62.09 | 183 | Chiefs |
| 100 | 27 | Demetrius Flannigan-Fowles | 55.13 | 52.60 | 60.61 | 174 | 49ers |
| 101 | 28 | Christian Rozeboom | 53.66 | 47.90 | 57.15 | 579 | Rams |
| 102 | 29 | Henry To'oTo'o | 53.38 | 45.00 | 57.73 | 459 | Texans |
| 103 | 30 | Troy Andersen | 51.79 | 50.60 | 57.99 | 139 | Falcons |
| 104 | 31 | Troy Reeder | 51.60 | 45.70 | 57.93 | 194 | Rams |
| 105 | 32 | Drew Sanders | 49.29 | 37.20 | 55.15 | 260 | Broncos |
| 106 | 33 | Chad Muma | 48.90 | 38.10 | 56.72 | 146 | Jaguars |
| 107 | 34 | Josh Woods | 45.00 | 31.80 | 54.91 | 568 | Cardinals |

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
| 10 | 6 | Kirk Cousins | 77.29 | 81.84 | 74.87 | 353 | Vikings |
| 11 | 7 | Geno Smith | 77.20 | 80.04 | 73.53 | 591 | Seahawks |
| 12 | 8 | Justin Herbert | 76.55 | 81.66 | 70.13 | 552 | Chargers |
| 13 | 9 | Joe Burrow | 76.35 | 82.88 | 71.05 | 422 | Bengals |
| 14 | 10 | Trevor Lawrence | 74.71 | 75.14 | 69.89 | 681 | Jaguars |
| 15 | 11 | Jordan Love | 74.71 | 82.34 | 73.93 | 725 | Packers |
| 16 | 12 | Derek Carr | 74.35 | 74.26 | 71.66 | 609 | Saints |
| 17 | 13 | C.J. Stroud | 74.01 | 80.40 | 76.65 | 653 | Texans |

### Starter (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Russell Wilson | 72.41 | 70.21 | 72.84 | 568 | Broncos |
| 19 | 2 | Baker Mayfield | 71.84 | 70.28 | 70.83 | 777 | Buccaneers |
| 20 | 3 | Kyler Murray | 66.11 | 69.40 | 66.84 | 326 | Cardinals |
| 21 | 4 | Ryan Tannehill | 65.25 | 70.56 | 65.27 | 291 | Titans |
| 22 | 5 | Jake Browning | 64.71 | 74.26 | 75.45 | 298 | Bengals |
| 23 | 6 | Justin Fields | 64.62 | 63.49 | 66.71 | 478 | Bears |
| 24 | 7 | Kenny Pickett | 63.97 | 70.15 | 62.53 | 386 | Steelers |
| 25 | 8 | Tyrod Taylor | 63.29 | 71.00 | 72.70 | 242 | Giants |
| 26 | 9 | Jimmy Garoppolo | 63.27 | 67.96 | 65.93 | 203 | Raiders |
| 27 | 10 | Mason Rudolph | 63.18 | 64.89 | 77.93 | 130 | Steelers |

### Rotation/backup (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Nick Mullens | 61.84 | 60.37 | 76.23 | 172 | Vikings |
| 29 | 2 | Joe Flacco | 61.82 | 66.48 | 67.73 | 277 | Browns |
| 30 | 3 | Will Levis | 60.92 | 61.60 | 67.84 | 310 | Titans |
| 31 | 4 | Aidan O'Connell | 60.89 | 64.60 | 63.06 | 380 | Raiders |
| 32 | 5 | Gardner Minshew | 60.60 | 60.36 | 63.95 | 580 | Colts |
| 33 | 6 | Sam Howell | 60.40 | 57.65 | 63.07 | 750 | Commanders |
| 34 | 7 | Mac Jones | 60.38 | 60.33 | 62.03 | 395 | Patriots |
| 35 | 8 | Tommy DeVito | 60.05 | 64.40 | 63.54 | 243 | Giants |
| 36 | 9 | Deshaun Watson | 58.97 | 62.39 | 62.26 | 218 | Browns |
| 37 | 10 | Easton Stick | 58.96 | 61.63 | 63.53 | 208 | Chargers |
| 38 | 11 | Desmond Ridder | 58.95 | 53.94 | 65.63 | 464 | Falcons |
| 39 | 12 | Daniel Jones | 58.55 | 63.93 | 58.80 | 220 | Giants |
| 40 | 13 | Joshua Dobbs | 58.30 | 59.76 | 59.37 | 524 | Vikings |
| 41 | 14 | Anthony Richardson | 58.21 | 56.50 | 64.74 | 107 | Colts |
| 42 | 15 | Bryce Young | 57.91 | 53.00 | 56.36 | 663 | Panthers |
| 43 | 16 | Bailey Zappe | 56.59 | 54.19 | 58.85 | 253 | Patriots |
| 44 | 17 | Mitch Trubisky | 56.41 | 55.53 | 58.25 | 132 | Steelers |
| 45 | 18 | Zach Wilson | 55.66 | 54.51 | 59.01 | 463 | Jets |
| 46 | 19 | Tyson Bagent | 55.22 | 50.80 | 57.31 | 166 | Bears |
| 47 | 20 | Dorian Thompson-Robinson | 53.73 | 49.40 | 52.42 | 132 | Browns |
| 48 | 21 | P.J. Walker | 52.79 | 38.16 | 57.83 | 135 | Browns |
| 49 | 22 | Trevor Siemian | 52.66 | 41.05 | 54.93 | 177 | Jets |
| 50 | 23 | Taylor Heinicke | 52.16 | 51.20 | 60.04 | 161 | Falcons |

## S — Safety

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jessie Bates III | 92.67 | 90.20 | 90.83 | 1134 | Falcons |
| 2 | 2 | Xavier McKinney | 91.52 | 91.20 | 89.91 | 1128 | Giants |
| 3 | 3 | Jevon Holland | 90.93 | 89.90 | 90.10 | 707 | Dolphins |
| 4 | 4 | Kyle Hamilton | 88.58 | 90.10 | 83.77 | 1065 | Ravens |
| 5 | 5 | Antoine Winfield Jr. | 87.71 | 84.00 | 87.98 | 1230 | Buccaneers |
| 6 | 6 | Alohi Gilman | 87.22 | 89.20 | 84.39 | 928 | Chargers |
| 7 | 7 | Tyrann Mathieu | 86.96 | 87.40 | 82.70 | 1096 | Saints |
| 8 | 8 | Juanyeh Thomas | 83.10 | 85.90 | 78.90 | 192 | Cowboys |
| 9 | 9 | Xavier Woods | 81.98 | 81.70 | 80.07 | 795 | Panthers |
| 10 | 10 | Jabrill Peppers | 81.90 | 83.20 | 80.00 | 955 | Patriots |
| 11 | 11 | Jordan Battle | 81.32 | 76.40 | 80.43 | 524 | Bengals |
| 12 | 12 | Geno Stone | 80.44 | 85.30 | 75.28 | 1000 | Ravens |
| 13 | 13 | Julian Love | 80.39 | 80.40 | 76.52 | 937 | Seahawks |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Jalen Thompson | 78.21 | 77.00 | 75.84 | 938 | Cardinals |
| 15 | 2 | Reed Blankenship | 77.50 | 78.00 | 77.54 | 942 | Eagles |
| 16 | 3 | Malik Hooker | 77.44 | 71.70 | 77.78 | 862 | Cowboys |
| 17 | 4 | Andrew Wingard | 76.83 | 75.10 | 76.57 | 330 | Jaguars |
| 18 | 5 | Marcus Williams | 76.23 | 76.40 | 76.17 | 765 | Ravens |
| 19 | 6 | Kevin Byard | 76.06 | 68.00 | 77.27 | 1187 | Eagles |
| 20 | 7 | Rudy Ford | 75.63 | 72.90 | 76.61 | 626 | Packers |
| 21 | 8 | Julian Blackmon | 74.76 | 72.60 | 76.05 | 987 | Colts |
| 22 | 9 | Darnell Savage | 74.62 | 74.20 | 73.49 | 701 | Packers |
| 23 | 10 | Grant Delpit | 74.57 | 75.50 | 72.13 | 738 | Browns |
| 24 | 11 | Jordan Poyer | 74.33 | 69.50 | 75.05 | 1103 | Bills |
| 25 | 12 | Ashtyn Davis | 74.21 | 75.20 | 73.40 | 218 | Jets |
| 26 | 13 | Camryn Bynum | 74.13 | 69.70 | 72.92 | 1120 | Vikings |

### Starter (55 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | Ji'Ayir Brown | 73.93 | 71.50 | 76.28 | 546 | 49ers |
| 28 | 2 | Kareem Jackson | 73.81 | 75.40 | 71.92 | 530 | Texans |
| 29 | 3 | Minkah Fitzpatrick | 73.79 | 67.90 | 77.28 | 616 | Steelers |
| 30 | 4 | Andre Cisco | 73.45 | 71.50 | 73.34 | 848 | Jaguars |
| 31 | 5 | Tre'von Moehrig | 73.38 | 68.00 | 73.39 | 1105 | Raiders |
| 32 | 6 | Brandon Jones | 73.19 | 76.20 | 71.82 | 542 | Dolphins |
| 33 | 7 | D'Anthony Bell | 73.14 | 69.80 | 78.06 | 256 | Browns |
| 34 | 8 | Harrison Smith | 72.54 | 69.50 | 71.66 | 1111 | Vikings |
| 35 | 9 | Ronnie Hickman Jr. | 72.00 | 70.20 | 74.92 | 322 | Browns |
| 36 | 10 | Talanoa Hufanga | 71.35 | 66.30 | 74.57 | 577 | 49ers |
| 37 | 11 | K'Von Wallace | 70.80 | 70.70 | 71.30 | 807 | Titans |
| 38 | 12 | Jimmie Ward | 70.77 | 67.80 | 73.68 | 506 | Texans |
| 39 | 13 | Jordan Whitehead | 70.67 | 68.90 | 68.27 | 1076 | Jets |
| 40 | 14 | Micah Hyde | 69.88 | 69.80 | 70.66 | 912 | Bills |
| 41 | 15 | Tashaun Gipson Sr. | 69.72 | 63.80 | 70.49 | 1194 | 49ers |
| 42 | 16 | Adrian Amos | 69.56 | 60.70 | 72.28 | 395 | Texans |
| 43 | 17 | Kamren Curl | 69.40 | 67.80 | 68.75 | 1088 | Commanders |
| 44 | 18 | Jason Pinnock | 69.29 | 64.80 | 69.10 | 1011 | Giants |
| 45 | 19 | Jordan Howden | 69.21 | 65.20 | 69.69 | 569 | Saints |
| 46 | 20 | Justin Simmons | 69.18 | 63.40 | 71.31 | 985 | Broncos |
| 47 | 21 | Budda Baker | 68.67 | 63.10 | 71.25 | 763 | Cardinals |
| 48 | 22 | Duron Harmon | 68.67 | 62.30 | 73.65 | 222 | Browns |
| 49 | 23 | Donovan Wilson | 68.49 | 66.60 | 67.65 | 776 | Cowboys |
| 50 | 24 | Jaquan Brisker | 68.44 | 62.40 | 70.26 | 896 | Bears |
| 51 | 25 | Juan Thornhill | 68.37 | 62.60 | 70.80 | 643 | Browns |
| 52 | 26 | Amani Hooker | 67.84 | 65.00 | 70.86 | 867 | Titans |
| 53 | 27 | Vonn Bell | 67.48 | 63.80 | 68.21 | 777 | Panthers |
| 54 | 28 | Josh Metellus | 67.44 | 64.50 | 68.97 | 1063 | Vikings |
| 55 | 29 | Terrell Edmunds | 67.17 | 60.30 | 69.15 | 475 | Titans |
| 56 | 30 | Jordan Fuller | 67.07 | 63.30 | 69.74 | 1057 | Rams |
| 57 | 31 | Jalen Pitre | 67.05 | 61.60 | 66.52 | 1032 | Texans |
| 58 | 32 | Marcus Epps | 66.79 | 62.30 | 65.82 | 1030 | Raiders |
| 59 | 33 | DeShon Elliott | 66.52 | 60.30 | 70.04 | 987 | Dolphins |
| 60 | 34 | Tony Adams | 66.34 | 65.20 | 67.84 | 879 | Jets |
| 61 | 35 | Sydney Brown | 66.18 | 64.90 | 66.79 | 334 | Eagles |
| 62 | 36 | John Johnson III | 66.02 | 58.30 | 68.86 | 574 | Rams |
| 63 | 37 | Nick Cross | 65.79 | 63.20 | 69.23 | 292 | Colts |
| 64 | 38 | Rayshawn Jenkins | 65.71 | 59.60 | 66.20 | 1099 | Jaguars |
| 65 | 39 | Justin Evans | 65.55 | 61.20 | 71.83 | 197 | Eagles |
| 66 | 40 | Miles Killebrew | 65.39 | 61.10 | 67.52 | 111 | Steelers |
| 67 | 41 | Taylor Rapp | 65.27 | 57.60 | 67.98 | 422 | Bills |
| 68 | 42 | Bryan Cook | 65.22 | 63.10 | 65.90 | 593 | Chiefs |
| 69 | 43 | Mike Brown | 65.06 | 58.70 | 70.65 | 113 | Titans |
| 70 | 44 | Damontae Kazee | 64.56 | 62.30 | 65.24 | 791 | Steelers |
| 71 | 45 | Rodney Thomas II | 64.25 | 60.10 | 63.59 | 962 | Colts |
| 72 | 46 | Jonathan Owens | 64.04 | 59.60 | 65.18 | 927 | Packers |
| 73 | 47 | Kaevon Merriweather | 63.82 | 67.10 | 65.31 | 164 | Buccaneers |
| 74 | 48 | Isaiah Pola-Mao | 63.76 | 66.60 | 65.30 | 130 | Raiders |
| 75 | 49 | Derwin James Jr. | 63.49 | 57.00 | 65.42 | 1001 | Chargers |
| 76 | 50 | C.J. Gardner-Johnson | 63.44 | 64.20 | 65.64 | 291 | Lions |
| 77 | 51 | Eddie Jackson | 63.34 | 60.00 | 65.92 | 646 | Bears |
| 78 | 52 | DeMarcco Hellams | 62.75 | 61.30 | 65.43 | 370 | Falcons |
| 79 | 53 | Trenton Thompson | 62.26 | 63.80 | 66.80 | 212 | Steelers |
| 80 | 54 | Eric Rowe | 62.17 | 61.80 | 65.50 | 212 | Steelers |
| 81 | 55 | Percy Butler | 62.12 | 58.00 | 64.14 | 835 | Commanders |

### Rotation/backup (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 82 | 1 | Mike Edwards | 61.85 | 58.60 | 61.61 | 809 | Chiefs |
| 83 | 2 | DeAndre Houston-Carson | 61.74 | 56.40 | 64.57 | 589 | Texans |
| 84 | 3 | Quandre Diggs | 61.56 | 55.40 | 61.50 | 1155 | Seahawks |
| 85 | 4 | Tracy Walker III | 61.10 | 57.80 | 66.10 | 541 | Lions |
| 86 | 5 | Dane Belton | 61.08 | 59.00 | 59.76 | 295 | Giants |
| 87 | 6 | Sam Franklin Jr. | 60.63 | 65.40 | 63.08 | 289 | Panthers |
| 88 | 7 | Kyle Dugger | 60.62 | 50.00 | 64.51 | 1116 | Patriots |
| 89 | 8 | Justin Reid | 60.35 | 51.80 | 62.67 | 1247 | Chiefs |
| 90 | 9 | Eric Murray | 60.18 | 63.60 | 61.39 | 176 | Texans |
| 91 | 10 | Dean Marlowe | 60.06 | 57.50 | 64.57 | 298 | Chargers |
| 92 | 11 | P.J. Locke | 60.06 | 55.20 | 64.71 | 538 | Broncos |
| 93 | 12 | Elijah Campbell | 59.56 | 57.90 | 61.20 | 141 | Dolphins |
| 94 | 13 | Jaylinn Hawkins | 58.71 | 50.60 | 63.77 | 137 | Chargers |
| 95 | 14 | Kerby Joseph | 58.64 | 50.90 | 60.74 | 1043 | Lions |
| 96 | 15 | Keanu Neal | 58.35 | 56.40 | 60.20 | 430 | Steelers |
| 97 | 16 | Russ Yeast | 58.33 | 55.60 | 58.91 | 836 | Rams |
| 98 | 17 | Darrick Forrest | 58.32 | 58.40 | 62.91 | 328 | Commanders |
| 99 | 18 | Marcus Maye | 57.89 | 57.60 | 63.04 | 444 | Saints |
| 100 | 19 | Jeremy Chinn | 57.23 | 51.60 | 61.23 | 285 | Panthers |
| 101 | 20 | Rodney McLeod | 56.79 | 52.50 | 59.70 | 280 | Browns |
| 102 | 21 | M.J. Stewart | 55.44 | 54.90 | 60.58 | 166 | Texans |
| 103 | 22 | Jayron Kearse | 55.38 | 43.60 | 60.15 | 860 | Cowboys |
| 104 | 23 | Richie Grant | 54.22 | 42.40 | 57.93 | 945 | Falcons |
| 105 | 24 | Adrian Phillips | 52.64 | 45.30 | 55.33 | 139 | Patriots |
| 106 | 25 | Johnathan Abram | 52.33 | 48.70 | 56.77 | 209 | Saints |
| 107 | 26 | Ryan Neal | 52.10 | 39.10 | 57.09 | 615 | Buccaneers |
| 108 | 27 | Nick Scott | 52.06 | 38.80 | 57.03 | 569 | Bengals |
| 109 | 28 | Jamal Adams | 51.14 | 50.70 | 56.87 | 518 | Seahawks |
| 110 | 29 | Delarrin Turner-Yell | 49.68 | 48.30 | 56.60 | 212 | Broncos |
| 111 | 30 | Terrell Burgess | 48.91 | 44.40 | 59.71 | 120 | Commanders |
| 112 | 31 | Alex Cook | 48.10 | 56.60 | 56.77 | 155 | Panthers |

## T — Tackle

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Trent Williams | 96.79 | 92.60 | 95.42 | 1013 | 49ers |
| 2 | 2 | Penei Sewell | 96.49 | 92.80 | 94.78 | 1379 | Lions |
| 3 | 3 | Braden Smith | 90.95 | 83.30 | 91.89 | 575 | Colts |
| 4 | 4 | Jordan Mailata | 90.60 | 84.80 | 90.30 | 1206 | Eagles |
| 5 | 5 | Tristan Wirfs | 88.79 | 83.10 | 88.42 | 1233 | Buccaneers |
| 6 | 6 | Tyron Smith | 88.27 | 83.70 | 87.15 | 942 | Cowboys |
| 7 | 7 | Christian Darrisaw | 87.86 | 82.40 | 87.34 | 982 | Vikings |
| 8 | 8 | Rob Havenstein | 87.79 | 79.80 | 88.95 | 914 | Rams |
| 9 | 9 | Bernhard Raimann | 87.55 | 82.70 | 86.61 | 1012 | Colts |
| 10 | 10 | Morgan Moses | 87.32 | 80.40 | 87.77 | 901 | Ravens |
| 11 | 11 | Terron Armstead | 87.16 | 79.30 | 88.24 | 585 | Dolphins |
| 12 | 12 | Trent Brown | 87.13 | 80.20 | 87.58 | 579 | Patriots |
| 13 | 13 | Zach Tom | 87.07 | 79.70 | 87.81 | 1162 | Packers |
| 14 | 14 | Lane Johnson | 86.61 | 80.10 | 86.78 | 1038 | Eagles |
| 15 | 15 | Kolton Miller | 86.51 | 80.20 | 86.55 | 705 | Raiders |
| 16 | 16 | Taylor Decker | 86.24 | 81.10 | 85.50 | 1243 | Lions |
| 17 | 17 | Kaleb McGary | 84.47 | 75.50 | 86.29 | 847 | Falcons |
| 18 | 18 | Laremy Tunsil | 83.87 | 75.50 | 85.28 | 965 | Texans |
| 19 | 19 | Garett Bolles | 83.14 | 75.90 | 83.80 | 1073 | Broncos |
| 20 | 20 | Andrew Thomas | 82.77 | 76.10 | 83.05 | 576 | Giants |
| 21 | 21 | Dion Dawkins | 82.73 | 74.90 | 83.78 | 1264 | Bills |
| 22 | 22 | Brian O'Neill | 82.67 | 74.50 | 83.95 | 884 | Vikings |
| 23 | 23 | Rashawn Slater | 82.51 | 76.60 | 82.29 | 1154 | Chargers |
| 24 | 24 | Thayer Munford Jr. | 82.19 | 74.10 | 83.41 | 521 | Raiders |
| 25 | 25 | Taylor Moton | 82.15 | 74.60 | 83.01 | 1148 | Panthers |
| 26 | 26 | Luke Goedeke | 81.85 | 73.40 | 83.31 | 1236 | Buccaneers |
| 27 | 27 | Alijah Vera-Tucker | 81.67 | 71.70 | 84.15 | 250 | Jets |
| 28 | 28 | Ryan Ramczyk | 81.49 | 73.50 | 82.65 | 785 | Saints |
| 29 | 29 | Jaylon Moore | 81.11 | 72.90 | 82.42 | 227 | 49ers |
| 30 | 30 | Spencer Brown | 80.77 | 70.10 | 83.71 | 1304 | Bills |
| 31 | 31 | Charles Leno Jr. | 80.52 | 72.50 | 81.70 | 880 | Commanders |
| 32 | 32 | Patrick Mekari | 80.11 | 71.00 | 82.02 | 593 | Ravens |
| 33 | 33 | Jake Matthews | 80.05 | 71.20 | 81.78 | 1061 | Falcons |

### Good (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Ikem Ekwonu | 79.66 | 67.40 | 83.67 | 1148 | Panthers |
| 35 | 2 | Jermaine Eluemunor | 79.65 | 68.70 | 82.79 | 905 | Raiders |
| 36 | 3 | Storm Norton | 79.39 | 68.30 | 82.62 | 283 | Falcons |
| 37 | 4 | Braxton Jones | 79.12 | 68.80 | 81.84 | 724 | Bears |
| 38 | 5 | Mike McGlinchey | 78.89 | 67.50 | 82.32 | 947 | Broncos |
| 39 | 6 | Austin Jackson | 78.74 | 66.90 | 82.46 | 1050 | Dolphins |
| 40 | 7 | Andrew Wylie | 78.64 | 69.20 | 80.77 | 977 | Commanders |
| 41 | 8 | Ronnie Stanley | 77.94 | 68.10 | 80.33 | 834 | Ravens |
| 42 | 9 | Conor McDermott | 77.62 | 67.80 | 80.00 | 227 | Patriots |
| 43 | 10 | Chris Hubbard | 77.48 | 68.20 | 79.50 | 473 | Titans |
| 44 | 11 | Alaric Jackson | 76.37 | 66.60 | 78.72 | 1026 | Rams |
| 45 | 12 | Charles Cross | 76.30 | 67.60 | 77.94 | 832 | Seahawks |
| 46 | 13 | Rasheed Walker | 76.13 | 66.30 | 78.51 | 974 | Packers |
| 47 | 14 | Cam Robinson | 75.80 | 65.40 | 78.56 | 535 | Jaguars |
| 48 | 15 | Colton McKivitz | 75.68 | 65.20 | 78.50 | 1245 | 49ers |
| 49 | 16 | Dawand Jones | 75.15 | 64.70 | 77.95 | 712 | Browns |
| 50 | 17 | David Quessenberry | 75.09 | 64.80 | 77.78 | 331 | Vikings |
| 51 | 18 | Orlando Brown Jr. | 75.03 | 66.10 | 76.82 | 1058 | Bengals |
| 52 | 19 | Darnell Wright | 74.99 | 62.40 | 79.22 | 1127 | Bears |
| 53 | 20 | Chukwuma Okorafor | 74.53 | 60.40 | 79.78 | 436 | Steelers |
| 54 | 21 | Kendall Lamm | 74.53 | 64.30 | 77.18 | 613 | Dolphins |
| 55 | 22 | D.J. Humphries | 74.51 | 62.50 | 78.35 | 922 | Cardinals |

### Starter (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 56 | 1 | Cornelius Lucas | 73.32 | 62.80 | 76.17 | 231 | Commanders |
| 57 | 2 | Chuma Edoga | 73.12 | 61.40 | 76.76 | 420 | Cowboys |
| 58 | 3 | George Fant | 73.10 | 61.80 | 76.47 | 1004 | Texans |
| 59 | 4 | Trey Pipkins III | 72.63 | 62.80 | 75.01 | 1116 | Chargers |
| 60 | 5 | Cam Fleming | 72.62 | 59.40 | 77.27 | 126 | Broncos |
| 61 | 6 | Stone Forsythe | 72.52 | 58.20 | 77.90 | 497 | Seahawks |
| 62 | 7 | Paris Johnson Jr. | 72.37 | 60.10 | 76.38 | 1130 | Cardinals |
| 63 | 8 | Kelvin Beachum | 71.84 | 61.20 | 74.77 | 212 | Cardinals |
| 64 | 9 | Walker Little | 71.64 | 58.80 | 76.04 | 659 | Jaguars |
| 65 | 10 | Jason Peters | 71.28 | 53.90 | 78.70 | 215 | Seahawks |
| 66 | 11 | Jonah Williams | 71.04 | 58.50 | 75.23 | 1087 | Bengals |
| 67 | 12 | Yosh Nijman | 70.93 | 57.40 | 75.78 | 259 | Packers |
| 68 | 13 | Broderick Jones | 70.79 | 57.30 | 75.61 | 832 | Steelers |
| 69 | 14 | Donovan Smith | 69.91 | 57.80 | 73.81 | 1037 | Chiefs |
| 70 | 15 | Jake Curhan | 69.84 | 55.70 | 75.10 | 296 | Seahawks |
| 71 | 16 | Mekhi Becton | 69.05 | 53.20 | 75.45 | 985 | Jets |
| 72 | 17 | Dan Moore Jr. | 68.91 | 54.40 | 74.41 | 1017 | Steelers |
| 73 | 18 | Jedrick Wills Jr. | 68.82 | 54.00 | 74.53 | 569 | Browns |
| 74 | 19 | Anton Harrison | 68.81 | 53.00 | 75.19 | 1112 | Jaguars |
| 75 | 20 | Wanya Morris | 68.56 | 55.60 | 73.03 | 340 | Chiefs |
| 76 | 21 | Abraham Lucas | 68.19 | 53.10 | 74.08 | 273 | Seahawks |
| 77 | 22 | Tyre Phillips | 68.06 | 52.70 | 74.14 | 552 | Giants |
| 78 | 23 | Terence Steele | 68.02 | 52.30 | 74.33 | 1273 | Cowboys |
| 79 | 24 | Trevor Penning | 67.94 | 53.60 | 73.33 | 417 | Saints |
| 80 | 25 | Cameron Erving | 67.90 | 49.00 | 76.33 | 208 | Saints |
| 81 | 26 | Jawaan Taylor | 66.95 | 49.80 | 74.21 | 1364 | Chiefs |
| 82 | 27 | Josh Jones | 66.90 | 49.60 | 74.26 | 233 | Texans |
| 83 | 28 | Joe Noteboom | 66.51 | 52.30 | 71.82 | 573 | Rams |
| 84 | 29 | Landon Young | 65.94 | 49.40 | 72.80 | 213 | Saints |
| 85 | 30 | Andre Dillard | 65.91 | 51.00 | 71.69 | 562 | Titans |
| 86 | 31 | Billy Turner | 65.32 | 50.90 | 70.76 | 208 | Jets |
| 87 | 32 | Charlie Heck | 64.97 | 47.40 | 72.51 | 253 | Texans |
| 88 | 33 | Max Mitchell | 64.81 | 49.30 | 70.99 | 474 | Jets |
| 89 | 34 | Daniel Faalele | 64.35 | 48.70 | 70.62 | 191 | Ravens |
| 90 | 35 | Larry Borom | 64.08 | 48.00 | 70.64 | 411 | Bears |
| 91 | 36 | Carter Warren | 64.07 | 46.90 | 71.35 | 401 | Jets |
| 92 | 37 | Duane Brown | 63.85 | 47.90 | 70.31 | 111 | Jets |
| 93 | 38 | Geron Christian | 63.60 | 45.60 | 71.43 | 708 | Browns |
| 94 | 39 | Calvin Anderson | 63.10 | 44.60 | 71.27 | 154 | Patriots |
| 95 | 40 | James Hudson III | 63.09 | 44.70 | 71.18 | 622 | Browns |
| 96 | 41 | Matt Peart | 63.05 | 42.90 | 72.31 | 133 | Giants |
| 97 | 42 | Blake Freeland | 62.43 | 44.20 | 70.41 | 701 | Colts |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 98 | 1 | Evan Neal | 61.21 | 39.80 | 71.32 | 460 | Giants |
| 99 | 2 | Vederian Lowe | 60.81 | 41.70 | 69.39 | 476 | Patriots |
| 100 | 3 | Nicholas Petit-Frere | 58.69 | 35.50 | 69.98 | 117 | Titans |
| 101 | 4 | Jaelyn Duncan | 56.26 | 32.90 | 67.66 | 364 | Titans |
| 102 | 5 | Blake Hance | 50.92 | 28.00 | 62.04 | 152 | Jaguars |

## TE — Tight End

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | George Kittle | 85.34 | 87.70 | 79.60 | 631 | 49ers |
| 2 | 2 | Travis Kelce | 83.52 | 82.60 | 79.96 | 719 | Chiefs |
| 3 | 3 | Mark Andrews | 81.17 | 78.80 | 78.58 | 314 | Ravens |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Kyle Pitts | 77.54 | 68.10 | 79.66 | 495 | Falcons |
| 5 | 2 | Dallas Goedert | 77.20 | 71.00 | 77.16 | 512 | Eagles |
| 6 | 3 | T.J. Hockenson | 77.14 | 78.80 | 71.87 | 563 | Vikings |
| 7 | 4 | Will Dissly | 76.60 | 71.30 | 75.96 | 167 | Seahawks |
| 8 | 5 | Jake Ferguson | 76.28 | 74.50 | 73.30 | 638 | Cowboys |
| 9 | 6 | Sam LaPorta | 75.83 | 77.00 | 70.89 | 683 | Lions |
| 10 | 7 | Marcedes Lewis | 75.57 | 77.10 | 70.39 | 103 | Bears |
| 11 | 8 | Andrew Ogletree | 75.44 | 66.20 | 77.43 | 158 | Colts |
| 12 | 9 | Hunter Henry | 75.27 | 69.30 | 75.08 | 413 | Patriots |
| 13 | 10 | Darren Waller | 74.38 | 69.40 | 73.53 | 394 | Giants |
| 14 | 11 | Trey McBride | 74.07 | 76.30 | 68.42 | 448 | Cardinals |

### Starter (52 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Cole Kmet | 73.85 | 73.70 | 69.78 | 503 | Bears |
| 16 | 2 | Pat Freiermuth | 73.20 | 65.80 | 73.97 | 340 | Steelers |
| 17 | 3 | Brevin Jordan | 73.08 | 69.90 | 71.04 | 184 | Texans |
| 18 | 4 | David Njoku | 72.84 | 68.20 | 71.76 | 647 | Browns |
| 19 | 5 | Pharaoh Brown | 72.63 | 73.30 | 68.01 | 153 | Patriots |
| 20 | 6 | Evan Engram | 72.52 | 71.60 | 68.97 | 675 | Jaguars |
| 21 | 7 | Dalton Schultz | 72.47 | 71.50 | 68.95 | 526 | Texans |
| 22 | 8 | Noah Fant | 72.11 | 62.90 | 74.08 | 350 | Seahawks |
| 23 | 9 | Davis Allen | 71.77 | 64.70 | 72.31 | 109 | Rams |
| 24 | 10 | Josh Oliver | 71.23 | 75.10 | 64.48 | 224 | Vikings |
| 25 | 11 | Tanner Hudson | 71.14 | 73.00 | 65.73 | 239 | Bengals |
| 26 | 12 | Donald Parham Jr. | 70.98 | 59.70 | 74.34 | 300 | Chargers |
| 27 | 13 | Will Mallory | 70.77 | 66.90 | 69.19 | 133 | Colts |
| 28 | 14 | Gerald Everett | 70.72 | 65.60 | 69.97 | 363 | Chargers |
| 29 | 15 | Chigoziem Okonkwo | 70.56 | 62.10 | 72.04 | 448 | Titans |
| 30 | 16 | Tyler Conklin | 70.46 | 65.60 | 69.54 | 536 | Jets |
| 31 | 17 | Jonnu Smith | 70.44 | 59.10 | 73.84 | 426 | Falcons |
| 32 | 18 | Luke Musgrave | 70.29 | 68.10 | 67.58 | 333 | Packers |
| 33 | 19 | Austin Hooper | 69.90 | 58.20 | 73.54 | 327 | Raiders |
| 34 | 20 | Dalton Kincaid | 69.88 | 68.80 | 66.44 | 543 | Bills |
| 35 | 21 | Johnny Mundt | 69.33 | 70.80 | 64.19 | 131 | Vikings |
| 36 | 22 | Mike Gesicki | 69.05 | 55.70 | 73.78 | 354 | Patriots |
| 37 | 23 | Tucker Kraft | 68.83 | 60.40 | 70.28 | 397 | Packers |
| 38 | 24 | MyCole Pruitt | 68.66 | 59.40 | 70.66 | 149 | Falcons |
| 39 | 25 | Isaiah Likely | 68.65 | 65.60 | 66.52 | 365 | Ravens |
| 40 | 26 | Mo Alie-Cox | 68.53 | 60.80 | 69.51 | 203 | Colts |
| 41 | 27 | Colby Parkinson | 68.43 | 57.60 | 71.49 | 244 | Seahawks |
| 42 | 28 | Tyler Higbee | 67.74 | 57.60 | 70.33 | 555 | Rams |
| 43 | 29 | Harrison Bryant | 67.45 | 57.00 | 70.25 | 169 | Browns |
| 44 | 30 | Luke Farrell | 67.37 | 58.10 | 69.39 | 164 | Jaguars |
| 45 | 31 | Dawson Knox | 67.31 | 54.50 | 71.68 | 303 | Bills |
| 46 | 32 | Kylen Granson | 66.89 | 56.80 | 69.45 | 316 | Colts |
| 47 | 33 | C.J. Uzomah | 66.73 | 66.40 | 62.79 | 133 | Jets |
| 48 | 34 | Logan Thomas | 66.70 | 57.10 | 68.93 | 567 | Commanders |
| 49 | 35 | Juwan Johnson | 65.85 | 58.80 | 66.39 | 344 | Saints |
| 50 | 36 | Connor Heyward | 65.78 | 55.50 | 68.46 | 230 | Steelers |
| 51 | 37 | Noah Gray | 65.48 | 62.90 | 63.03 | 435 | Chiefs |
| 52 | 38 | Lucas Krull | 65.36 | 60.50 | 64.43 | 139 | Broncos |
| 53 | 39 | Michael Mayer | 65.34 | 58.30 | 65.87 | 341 | Raiders |
| 54 | 40 | Foster Moreau | 65.32 | 55.30 | 67.84 | 204 | Saints |
| 55 | 41 | Charlie Woerner | 65.14 | 63.50 | 62.07 | 125 | 49ers |
| 56 | 42 | Chris Manhertz | 65.10 | 56.50 | 66.66 | 127 | Broncos |
| 57 | 43 | Drew Sample | 64.96 | 60.00 | 64.10 | 194 | Bengals |
| 58 | 44 | John Bates | 64.81 | 51.80 | 69.32 | 275 | Commanders |
| 59 | 45 | Adam Trautman | 64.28 | 53.20 | 67.50 | 434 | Broncos |
| 60 | 46 | Durham Smythe | 64.07 | 53.70 | 66.81 | 525 | Dolphins |
| 61 | 47 | Jordan Akins | 63.92 | 53.60 | 66.64 | 137 | Browns |
| 62 | 48 | Robert Tonyan | 63.84 | 49.60 | 69.17 | 178 | Bears |
| 63 | 49 | Hayden Hurst | 63.69 | 44.70 | 72.18 | 248 | Panthers |
| 64 | 50 | Tommy Tremble | 63.18 | 55.60 | 64.06 | 285 | Panthers |
| 65 | 51 | Cade Otton | 62.85 | 56.80 | 62.72 | 748 | Buccaneers |
| 66 | 52 | Jeremy Ruckert | 62.36 | 57.50 | 61.43 | 165 | Jets |

### Rotation/backup (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 67 | 1 | Brock Wright | 61.82 | 48.00 | 66.87 | 209 | Lions |
| 68 | 2 | Geoff Swaim | 61.80 | 50.20 | 65.37 | 143 | Cardinals |
| 69 | 3 | Daniel Bellinger | 59.91 | 49.10 | 62.95 | 354 | Giants |
| 70 | 4 | Jack Stoll | 59.56 | 50.80 | 61.24 | 184 | Eagles |
| 71 | 5 | Darnell Washington | 59.04 | 47.40 | 62.64 | 195 | Steelers |
| 72 | 6 | Brenton Strange | 58.87 | 48.60 | 61.55 | 118 | Jaguars |
| 73 | 7 | Stone Smartt | 58.61 | 44.10 | 64.11 | 172 | Chargers |
| 74 | 8 | Irv Smith Jr. | 58.35 | 44.20 | 63.62 | 239 | Bengals |
| 75 | 9 | Luke Schoonmaker | 58.19 | 49.80 | 59.62 | 161 | Cowboys |
| 76 | 10 | Blake Bell | 56.57 | 43.00 | 61.45 | 113 | Chiefs |
| 77 | 11 | Julian Hill | 55.88 | 41.80 | 61.10 | 153 | Dolphins |

## WR — Wide Receiver

- **Season used:** `2023`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Tyreek Hill | 90.26 | 93.40 | 84.00 | 515 | Dolphins |
| 2 | 2 | Nico Collins | 89.86 | 91.40 | 84.67 | 497 | Texans |
| 3 | 3 | Justin Jefferson | 88.78 | 91.10 | 83.06 | 390 | Vikings |
| 4 | 4 | Brandon Aiyuk | 88.71 | 91.50 | 82.69 | 593 | 49ers |
| 5 | 5 | A.J. Brown | 88.07 | 90.00 | 82.62 | 606 | Eagles |
| 6 | 6 | Puka Nacua | 87.54 | 90.20 | 81.60 | 658 | Rams |
| 7 | 7 | CeeDee Lamb | 87.30 | 90.90 | 80.73 | 736 | Cowboys |
| 8 | 8 | Amon-Ra St. Brown | 86.87 | 91.20 | 79.82 | 717 | Lions |
| 9 | 9 | Jaylen Waddle | 86.29 | 90.30 | 79.45 | 425 | Dolphins |
| 10 | 10 | DJ Moore | 85.95 | 89.30 | 79.55 | 613 | Bears |
| 11 | 11 | Deebo Samuel | 85.52 | 85.40 | 81.43 | 460 | 49ers |
| 12 | 12 | Rashee Rice | 84.27 | 84.90 | 79.68 | 577 | Chiefs |
| 13 | 13 | Ja'Marr Chase | 83.85 | 85.30 | 78.71 | 627 | Bengals |
| 14 | 14 | Tank Dell | 82.90 | 83.40 | 78.40 | 334 | Texans |
| 15 | 15 | Chris Olave | 82.42 | 82.50 | 78.20 | 557 | Saints |
| 16 | 16 | Mike Evans | 82.24 | 81.50 | 78.56 | 663 | Buccaneers |
| 17 | 17 | Amari Cooper | 82.18 | 79.90 | 79.54 | 627 | Browns |
| 18 | 18 | Keenan Allen | 82.12 | 86.30 | 75.16 | 557 | Chargers |
| 19 | 19 | Dontayvion Wicks | 81.43 | 77.80 | 79.69 | 328 | Packers |
| 20 | 20 | DeAndre Hopkins | 81.26 | 81.00 | 77.27 | 539 | Titans |
| 21 | 21 | D.K. Metcalf | 81.17 | 80.00 | 77.79 | 585 | Seahawks |
| 22 | 22 | Davante Adams | 80.91 | 79.10 | 77.95 | 598 | Raiders |
| 23 | 23 | Khalil Shakir | 80.79 | 76.70 | 79.35 | 413 | Bills |
| 24 | 24 | George Pickens | 80.35 | 74.10 | 80.35 | 617 | Steelers |
| 25 | 25 | Mike Williams | 80.21 | 74.60 | 79.79 | 113 | Chargers |

### Good (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 26 | 1 | Drake London | 79.69 | 78.90 | 76.05 | 511 | Falcons |
| 27 | 2 | Kalif Raymond | 79.15 | 75.40 | 77.48 | 244 | Lions |
| 28 | 3 | Stefon Diggs | 78.99 | 79.10 | 74.75 | 695 | Bills |
| 29 | 4 | Terry McLaurin | 78.95 | 75.10 | 77.35 | 676 | Commanders |
| 30 | 5 | Tyler Lockett | 78.90 | 78.10 | 75.27 | 595 | Seahawks |
| 31 | 6 | DeVonta Smith | 78.82 | 75.40 | 76.93 | 664 | Eagles |
| 32 | 7 | Cooper Kupp | 78.59 | 71.50 | 79.15 | 461 | Rams |
| 33 | 8 | Odell Beckham Jr. | 78.28 | 75.40 | 76.03 | 361 | Ravens |
| 34 | 9 | Chris Godwin | 78.13 | 77.50 | 74.38 | 686 | Buccaneers |
| 35 | 10 | Jayden Reed | 78.03 | 73.30 | 77.01 | 444 | Packers |
| 36 | 11 | Deonte Harty | 77.89 | 68.90 | 79.72 | 130 | Bills |
| 37 | 12 | Rashid Shaheed | 77.87 | 69.00 | 79.62 | 443 | Saints |
| 38 | 13 | Diontae Johnson | 77.78 | 77.90 | 73.53 | 433 | Steelers |
| 39 | 14 | Demario Douglas | 77.73 | 74.40 | 75.79 | 334 | Patriots |
| 40 | 15 | Tee Higgins | 77.51 | 72.10 | 76.95 | 419 | Bengals |
| 41 | 16 | Michael Pittman Jr. | 77.37 | 77.70 | 72.99 | 606 | Colts |
| 42 | 17 | Zay Flowers | 76.84 | 76.30 | 73.03 | 626 | Ravens |
| 43 | 18 | Christian Kirk | 76.17 | 71.10 | 75.39 | 405 | Jaguars |
| 44 | 19 | Noah Brown | 76.14 | 73.00 | 74.07 | 328 | Texans |
| 45 | 20 | Jerry Jeudy | 76.14 | 67.80 | 77.54 | 487 | Broncos |
| 46 | 21 | Christian Watson | 76.06 | 68.20 | 77.14 | 316 | Packers |
| 47 | 22 | Marvin Mims Jr. | 76.01 | 63.10 | 80.45 | 262 | Broncos |
| 48 | 23 | Garrett Wilson | 75.81 | 72.90 | 73.58 | 712 | Jets |
| 49 | 24 | Courtland Sutton | 75.76 | 75.60 | 71.70 | 506 | Broncos |
| 50 | 25 | Michael Wilson | 75.58 | 68.30 | 76.26 | 444 | Cardinals |
| 51 | 26 | Romeo Doubs | 75.52 | 73.60 | 72.63 | 587 | Packers |
| 52 | 27 | Kendrick Bourne | 75.39 | 67.20 | 76.68 | 247 | Patriots |
| 53 | 28 | Josh Downs | 75.36 | 70.30 | 74.56 | 510 | Colts |
| 54 | 29 | Calvin Ridley | 75.32 | 71.40 | 73.77 | 688 | Jaguars |
| 55 | 30 | Kyle Philips | 75.32 | 72.80 | 72.84 | 118 | Titans |
| 56 | 31 | Jauan Jennings | 74.81 | 74.10 | 71.12 | 326 | 49ers |
| 57 | 32 | Gabe Davis | 74.67 | 67.90 | 75.01 | 596 | Bills |
| 58 | 33 | Darius Slayton | 74.63 | 67.50 | 75.21 | 584 | Giants |
| 59 | 34 | Jordan Addison | 74.56 | 68.60 | 74.37 | 639 | Vikings |
| 60 | 35 | Josh Reynolds | 74.43 | 70.10 | 73.15 | 608 | Lions |
| 61 | 36 | Tre Tucker | 74.35 | 65.20 | 76.29 | 235 | Raiders |
| 62 | 37 | Jakobi Meyers | 74.00 | 69.90 | 72.56 | 560 | Raiders |

### Starter (84 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 63 | 1 | Jake Bobo | 73.61 | 73.50 | 69.52 | 156 | Seahawks |
| 64 | 2 | Adam Thielen | 73.52 | 72.70 | 69.90 | 677 | Panthers |
| 65 | 3 | Chris Moore | 73.47 | 64.90 | 75.01 | 330 | Titans |
| 66 | 4 | Tutu Atwell | 73.40 | 68.30 | 72.64 | 441 | Rams |
| 67 | 5 | Greg Dortch | 73.23 | 68.20 | 72.41 | 257 | Cardinals |
| 68 | 6 | Cedrick Wilson Jr. | 72.98 | 65.30 | 73.94 | 318 | Dolphins |
| 69 | 7 | Brandin Cooks | 72.93 | 68.70 | 71.59 | 635 | Cowboys |
| 70 | 8 | Alex Erickson | 72.82 | 62.60 | 75.47 | 231 | Chargers |
| 71 | 9 | Joshua Palmer | 72.77 | 67.60 | 72.05 | 368 | Chargers |
| 72 | 10 | Byron Pringle | 72.56 | 67.90 | 71.50 | 112 | Commanders |
| 73 | 11 | Michael Thomas | 72.47 | 67.40 | 71.69 | 333 | Saints |
| 74 | 12 | Curtis Samuel | 72.12 | 69.70 | 69.57 | 420 | Commanders |
| 75 | 13 | Demarcus Robinson | 72.08 | 70.00 | 69.30 | 271 | Rams |
| 76 | 14 | KhaDarel Hodge | 71.96 | 62.40 | 74.16 | 191 | Falcons |
| 77 | 15 | Marquise Brown | 71.80 | 67.10 | 70.76 | 498 | Cardinals |
| 78 | 16 | Lil'Jordan Humphrey | 71.59 | 60.40 | 74.89 | 242 | Broncos |
| 79 | 17 | Brandon Johnson | 71.01 | 64.10 | 71.45 | 232 | Broncos |
| 80 | 18 | A.T. Perry | 70.95 | 61.00 | 73.41 | 215 | Saints |
| 81 | 19 | Mack Hollins | 70.94 | 64.60 | 71.00 | 168 | Falcons |
| 82 | 20 | DeVante Parker | 70.83 | 61.20 | 73.09 | 362 | Patriots |
| 83 | 21 | Trenton Irwin | 70.54 | 57.00 | 75.40 | 279 | Bengals |
| 84 | 22 | Justin Watson | 69.96 | 62.40 | 70.84 | 490 | Chiefs |
| 85 | 23 | Dyami Brown | 69.87 | 59.00 | 72.95 | 178 | Commanders |
| 86 | 24 | Jamal Agnew | 69.84 | 59.90 | 72.30 | 156 | Jaguars |
| 87 | 25 | Scott Miller | 69.80 | 64.40 | 69.24 | 133 | Falcons |
| 88 | 26 | Jameson Williams | 69.80 | 62.90 | 70.24 | 342 | Lions |
| 89 | 27 | Tyler Boyd | 69.58 | 59.50 | 72.13 | 610 | Bengals |
| 90 | 28 | Wan'Dale Robinson | 69.57 | 65.50 | 68.12 | 418 | Giants |
| 91 | 29 | Michael Gallup | 69.49 | 63.40 | 69.39 | 474 | Cowboys |
| 92 | 30 | Quez Watkins | 69.47 | 59.20 | 72.15 | 239 | Eagles |
| 93 | 31 | DJ Chark Jr. | 69.42 | 60.00 | 71.53 | 521 | Panthers |
| 94 | 32 | Alec Pierce | 68.92 | 58.10 | 71.96 | 649 | Colts |
| 95 | 33 | Equanimeous St. Brown | 68.71 | 55.10 | 73.62 | 118 | Bears |
| 96 | 34 | Nick Westbrook-Ikhine | 68.59 | 59.40 | 70.55 | 362 | Titans |
| 97 | 35 | JuJu Smith-Schuster | 68.38 | 57.00 | 71.80 | 256 | Patriots |
| 98 | 36 | Mecole Hardman Jr. | 68.33 | 55.20 | 72.92 | 150 | Chiefs |
| 99 | 37 | Nelson Agholor | 68.31 | 61.60 | 68.61 | 419 | Ravens |
| 100 | 38 | Ray-Ray McCloud III | 68.27 | 63.10 | 67.55 | 154 | 49ers |
| 101 | 39 | Jaxon Smith-Njigba | 68.09 | 63.30 | 67.12 | 507 | Seahawks |
| 102 | 40 | Darnell Mooney | 67.98 | 55.20 | 72.34 | 482 | Bears |
| 103 | 41 | Zay Jones | 67.78 | 61.20 | 68.00 | 325 | Jaguars |
| 104 | 42 | Donovan Peoples-Jones | 67.61 | 52.70 | 73.39 | 286 | Lions |
| 105 | 43 | Jalin Hyatt | 67.52 | 59.70 | 68.57 | 403 | Giants |
| 106 | 44 | Rashod Bateman | 67.51 | 61.00 | 67.69 | 409 | Ravens |
| 107 | 45 | Quentin Johnston | 67.43 | 58.90 | 68.95 | 514 | Chargers |
| 108 | 46 | Kadarius Toney | 67.41 | 62.50 | 66.52 | 151 | Chiefs |
| 109 | 47 | Elijah Moore | 67.27 | 58.90 | 68.69 | 631 | Browns |
| 110 | 48 | Brandon Powell | 67.13 | 61.70 | 66.58 | 305 | Vikings |
| 111 | 49 | Richie James | 67.12 | 56.30 | 70.17 | 130 | Chiefs |
| 112 | 50 | Robert Woods | 67.08 | 60.70 | 67.17 | 470 | Texans |
| 113 | 51 | Chase Claypool | 66.94 | 55.10 | 70.66 | 123 | Dolphins |
| 114 | 52 | Skyy Moore | 66.92 | 54.80 | 70.83 | 305 | Chiefs |
| 115 | 53 | Marquez Valdes-Scantling | 66.72 | 52.10 | 72.30 | 608 | Chiefs |
| 116 | 54 | Isaiah Hodgins | 66.42 | 56.40 | 68.94 | 296 | Giants |
| 117 | 55 | David Bell | 66.38 | 63.60 | 64.07 | 168 | Browns |
| 118 | 56 | Xavier Gipson | 66.18 | 56.20 | 68.67 | 360 | Jets |
| 119 | 57 | Julio Jones | 66.16 | 55.00 | 69.44 | 176 | Eagles |
| 120 | 58 | Olamide Zaccheaus | 66.13 | 51.10 | 71.99 | 348 | Eagles |
| 121 | 59 | Treylon Burks | 65.93 | 52.40 | 70.79 | 286 | Titans |
| 122 | 60 | Jalen Tolbert | 65.80 | 56.50 | 67.84 | 321 | Cowboys |
| 123 | 61 | Jahan Dotson | 65.74 | 57.70 | 66.93 | 675 | Commanders |
| 124 | 62 | Calvin Austin III | 65.60 | 56.60 | 67.43 | 262 | Steelers |
| 125 | 63 | Allen Lazard | 65.39 | 52.70 | 69.68 | 483 | Jets |
| 126 | 64 | K.J. Osborn | 65.26 | 53.90 | 68.66 | 594 | Vikings |
| 127 | 65 | Terrace Marshall Jr. | 65.20 | 54.80 | 67.97 | 229 | Panthers |
| 128 | 66 | Braxton Berrios | 65.14 | 57.60 | 66.00 | 348 | Dolphins |
| 129 | 67 | Cedric Tillman | 64.94 | 55.50 | 67.06 | 382 | Browns |
| 130 | 68 | Trey Palmer | 64.80 | 53.30 | 68.30 | 537 | Buccaneers |
| 131 | 69 | John Metchie III | 64.77 | 57.80 | 65.25 | 226 | Texans |
| 132 | 70 | Trent Sherfield | 64.60 | 51.10 | 69.43 | 245 | Bills |
| 133 | 71 | Deven Thompkins | 64.27 | 59.30 | 63.41 | 149 | Buccaneers |
| 134 | 72 | Randall Cobb | 64.12 | 47.60 | 70.97 | 158 | Jets |
| 135 | 73 | Jalen Reagor | 63.63 | 51.40 | 67.62 | 204 | Patriots |
| 136 | 74 | Van Jefferson | 63.48 | 50.10 | 68.23 | 395 | Falcons |
| 137 | 75 | Jonathan Mingo | 63.46 | 54.70 | 65.14 | 577 | Panthers |
| 138 | 76 | Hunter Renfrow | 63.23 | 48.60 | 68.81 | 270 | Raiders |
| 139 | 77 | Malik Heath | 63.19 | 60.10 | 61.09 | 121 | Packers |
| 140 | 78 | Tyquan Thornton | 63.06 | 57.60 | 62.54 | 141 | Patriots |
| 141 | 79 | Rondale Moore | 63.00 | 53.60 | 65.10 | 489 | Cardinals |
| 142 | 80 | Jason Brownlee | 62.97 | 47.70 | 68.98 | 202 | Jets |
| 143 | 81 | Allen Robinson II | 62.84 | 52.80 | 65.37 | 445 | Steelers |
| 144 | 82 | Parris Campbell | 62.77 | 51.40 | 66.18 | 150 | Giants |
| 145 | 83 | Andrei Iosivas | 62.66 | 61.20 | 59.47 | 159 | Bengals |
| 146 | 84 | Jalen Guyton | 62.50 | 49.60 | 66.94 | 185 | Chargers |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 147 | 1 | Tim Jones | 61.44 | 52.00 | 63.56 | 197 | Jaguars |
| 148 | 2 | Tyler Scott | 60.95 | 52.70 | 62.29 | 271 | Bears |
| 149 | 3 | Zach Pascal | 60.76 | 47.60 | 65.36 | 103 | Cardinals |
| 150 | 4 | Parker Washington | 59.88 | 51.10 | 61.56 | 186 | Jaguars |
| 151 | 5 | Xavier Hutchinson | 58.68 | 47.30 | 62.10 | 200 | Texans |
