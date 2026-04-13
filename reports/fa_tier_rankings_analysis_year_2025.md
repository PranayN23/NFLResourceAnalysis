# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-13 03:34:04Z
- **Requested analysis_year:** 2025 (clamped to 2025)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Creed Humphrey | 97.14 | 92.30 | 97.86 | 1232 | Chiefs |
| 2 | 2 | Erik McCoy | 91.84 | 84.87 | 95.52 | 293 | Saints |
| 3 | 3 | Frank Ragnow | 90.72 | 86.10 | 91.14 | 1129 | Lions |
| 4 | 4 | Tyler Linderbaum | 85.89 | 79.90 | 88.20 | 1227 | Ravens |
| 5 | 5 | Zach Frazier | 84.10 | 77.90 | 86.70 | 1021 | Steelers |
| 6 | 6 | Drew Dalman | 82.02 | 75.98 | 84.42 | 554 | Falcons |
| 7 | 7 | Hjalte Froholdt | 81.80 | 76.10 | 83.72 | 1078 | Cardinals |
| 8 | 8 | Aaron Brewer | 80.12 | 73.30 | 83.59 | 1139 | Dolphins |

### Good (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Joe Tippmann | 79.79 | 73.40 | 82.67 | 1067 | Jets |
| 10 | 2 | Connor McGovern | 76.30 | 69.50 | 79.75 | 1164 | Bills |

### Starter (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Coleman Shelton | 73.90 | 66.40 | 78.31 | 1121 | Bears |
| 12 | 2 | Ryan Kelly | 73.45 | 66.08 | 77.68 | 601 | Colts |
| 13 | 3 | Cam Jurgens | 73.21 | 67.30 | 75.43 | 1217 | Eagles |
| 14 | 4 | Jake Brendel | 73.09 | 65.00 | 78.31 | 1072 | 49ers |
| 15 | 5 | Cooper Beebe | 72.85 | 65.40 | 77.19 | 1059 | Cowboys |
| 16 | 6 | Luke Wattenberg | 72.39 | 64.03 | 77.98 | 864 | Broncos |
| 17 | 7 | Tyler Biadasz | 71.60 | 64.20 | 75.87 | 1166 | Commanders |
| 18 | 8 | Ethan Pocic | 71.43 | 63.60 | 76.30 | 1073 | Browns |
| 19 | 9 | Alex Forsyth | 71.17 | 63.54 | 75.75 | 292 | Broncos |
| 20 | 10 | Ted Karras | 71.05 | 64.10 | 74.69 | 1136 | Bengals |
| 21 | 11 | Olusegun Oluwatimi | 70.94 | 62.79 | 76.25 | 435 | Seahawks |
| 22 | 12 | Brady Christensen | 70.86 | 62.29 | 76.75 | 399 | Panthers |
| 23 | 13 | Garrett Bradbury | 70.81 | 62.80 | 75.91 | 1191 | Vikings |
| 24 | 14 | Jarrett Patterson | 70.58 | 63.43 | 74.50 | 688 | Texans |
| 25 | 15 | Juice Scruggs | 70.51 | 62.94 | 75.00 | 944 | Texans |
| 26 | 16 | Danny Pinter | 70.33 | 63.44 | 73.90 | 138 | Colts |
| 27 | 17 | Austin Corbett | 70.10 | 61.58 | 75.91 | 291 | Panthers |
| 28 | 18 | John Michael Schmitz Jr. | 69.33 | 61.40 | 74.34 | 983 | Giants |
| 29 | 19 | Bradley Bozeman | 68.99 | 61.20 | 73.80 | 1112 | Chargers |
| 30 | 20 | Ryan Neuzil | 68.12 | 58.85 | 74.97 | 578 | Falcons |
| 31 | 21 | David Andrews | 67.24 | 59.13 | 72.48 | 193 | Patriots |
| 32 | 22 | Mitch Morse | 66.15 | 57.30 | 72.41 | 1021 | Jaguars |
| 33 | 23 | Corey Levin | 65.68 | 58.20 | 70.05 | 133 | Titans |
| 34 | 24 | Beaux Limmer | 65.56 | 55.50 | 73.50 | 1040 | Rams |
| 35 | 25 | Daniel Brunskill | 64.88 | 56.08 | 71.09 | 684 | Titans |
| 36 | 26 | Graham Barton | 64.82 | 55.60 | 71.60 | 1111 | Buccaneers |
| 37 | 27 | Lloyd Cushenberry III | 64.48 | 56.20 | 69.97 | 499 | Titans |
| 38 | 28 | Sedrick Van Pran-Granger | 64.11 | 57.84 | 66.81 | 125 | Bills |
| 39 | 29 | Andre James | 64.09 | 56.01 | 69.30 | 702 | Raiders |
| 40 | 30 | Ryan McCollum | 63.07 | 56.12 | 66.72 | 153 | Steelers |
| 41 | 31 | Josh Myers | 62.58 | 54.20 | 68.20 | 1067 | Packers |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 42 | 1 | Shane Lemieux | 61.85 | 54.79 | 65.64 | 337 | Saints |

## CB — Cornerback

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Pat Surtain II | 89.13 | 85.10 | 88.51 | 1054 | Broncos |
| 2 | 2 | Derek Stingley Jr. | 86.33 | 84.40 | 86.79 | 1119 | Texans |
| 3 | 3 | Quinyon Mitchell | 83.64 | 79.00 | 83.76 | 1104 | Eagles |
| 4 | 4 | Garrett Williams | 83.48 | 83.02 | 83.21 | 778 | Cardinals |
| 5 | 5 | Marlon Humphrey | 83.29 | 81.00 | 82.89 | 1000 | Ravens |
| 6 | 6 | Trent McDuffie | 83.12 | 80.70 | 82.10 | 1132 | Chiefs |
| 7 | 7 | Christian Benford | 82.98 | 78.60 | 85.26 | 1046 | Bills |
| 8 | 8 | Kamari Lassiter | 81.03 | 77.50 | 81.11 | 906 | Texans |
| 9 | 9 | Cooper DeJean | 80.96 | 79.00 | 77.81 | 830 | Eagles |

### Good (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Jourdan Lewis | 79.89 | 79.00 | 78.95 | 871 | Cowboys |
| 11 | 2 | Christian Gonzalez | 79.26 | 78.20 | 82.20 | 978 | Patriots |
| 12 | 3 | Darius Slay | 79.25 | 74.20 | 81.45 | 897 | Eagles |
| 13 | 4 | Sauce Gardner | 78.74 | 73.10 | 81.76 | 879 | Jets |
| 14 | 5 | Jamel Dean | 78.33 | 74.94 | 80.83 | 745 | Buccaneers |
| 15 | 6 | Renardo Green | 78.31 | 72.93 | 80.65 | 675 | 49ers |
| 16 | 7 | Jaylon Johnson | 78.06 | 74.20 | 79.98 | 1031 | Bears |
| 17 | 8 | Tarheeb Still | 77.56 | 74.80 | 78.01 | 826 | Chargers |
| 18 | 9 | Kyler Gordon | 77.02 | 74.99 | 77.62 | 724 | Bears |
| 19 | 10 | Jalen Ramsey | 76.20 | 71.90 | 78.26 | 1027 | Dolphins |
| 20 | 11 | Jaire Alexander | 75.96 | 74.58 | 82.02 | 361 | Packers |
| 21 | 12 | Byron Murphy Jr. | 75.80 | 73.50 | 76.32 | 1109 | Vikings |
| 22 | 13 | Carlton Davis III | 75.44 | 71.51 | 79.54 | 697 | Lions |
| 23 | 14 | Nate Wiggins | 74.99 | 68.69 | 77.14 | 769 | Ravens |
| 24 | 15 | Deommodore Lenoir | 74.93 | 71.70 | 74.89 | 922 | 49ers |
| 25 | 16 | D.J. Reed | 74.87 | 70.10 | 77.89 | 880 | Jets |
| 26 | 17 | Mike Jackson | 74.63 | 68.10 | 77.84 | 1204 | Panthers |
| 27 | 18 | A.J. Terrell | 74.50 | 69.50 | 75.84 | 1085 | Falcons |
| 28 | 19 | Samuel Womack III | 74.31 | 70.21 | 81.22 | 673 | Colts |
| 29 | 20 | Denzel Ward | 74.27 | 68.68 | 78.16 | 757 | Browns |
| 30 | 21 | Devon Witherspoon | 74.15 | 69.20 | 76.11 | 1103 | Seahawks |

### Starter (80 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Andru Phillips | 73.93 | 73.63 | 72.66 | 614 | Giants |
| 32 | 2 | Mike Hilton | 73.77 | 68.22 | 76.45 | 737 | Bengals |
| 33 | 3 | Carrington Valentine | 73.60 | 69.77 | 75.17 | 606 | Packers |
| 34 | 4 | Jaylon Jones | 73.44 | 67.90 | 75.63 | 1146 | Colts |
| 35 | 5 | DaRon Bland | 73.41 | 69.53 | 80.18 | 436 | Cowboys |
| 36 | 6 | Terell Smith | 73.32 | 68.26 | 82.61 | 207 | Bears |
| 37 | 7 | Tariq Woolen | 73.27 | 65.70 | 78.65 | 889 | Seahawks |
| 38 | 8 | Mike Hughes | 73.26 | 70.95 | 74.06 | 720 | Falcons |
| 39 | 9 | Jaylen Watson | 72.77 | 69.78 | 77.00 | 433 | Chiefs |
| 40 | 10 | Clark Phillips III | 72.47 | 69.93 | 75.74 | 409 | Falcons |
| 41 | 11 | Mike Sainristil | 72.05 | 64.50 | 75.73 | 1158 | Commanders |
| 42 | 12 | Isaiah Rodgers | 71.96 | 67.01 | 74.67 | 413 | Eagles |
| 43 | 13 | DJ Turner II | 71.68 | 66.59 | 77.25 | 508 | Bengals |
| 44 | 14 | Kenny Moore II | 71.67 | 68.20 | 73.35 | 1013 | Colts |
| 45 | 15 | Zyon McCollum | 71.43 | 66.10 | 74.47 | 1123 | Buccaneers |
| 46 | 16 | Kader Kohou | 71.32 | 68.24 | 71.36 | 708 | Dolphins |
| 47 | 17 | Joshua Williams | 71.06 | 65.86 | 76.16 | 411 | Chiefs |
| 48 | 18 | Kristian Fulton | 70.75 | 66.60 | 74.85 | 827 | Chargers |
| 49 | 19 | Kelee Ringo | 70.49 | 63.84 | 80.41 | 127 | Eagles |
| 50 | 20 | Cam Taylor-Britt | 70.35 | 64.30 | 76.00 | 1036 | Bengals |
| 51 | 21 | Troy Hill | 70.19 | 67.64 | 76.64 | 236 | Buccaneers |
| 52 | 22 | Marcus Jones | 70.06 | 65.98 | 78.92 | 586 | Patriots |
| 53 | 23 | Kool-Aid McKinstry | 70.03 | 66.45 | 72.76 | 680 | Saints |
| 54 | 24 | Jarrian Jones | 69.31 | 62.30 | 72.33 | 699 | Jaguars |
| 55 | 25 | Tyson Campbell | 68.97 | 63.82 | 75.33 | 767 | Jaguars |
| 56 | 26 | Amik Robertson | 68.65 | 61.92 | 72.50 | 630 | Lions |
| 57 | 27 | Adoree' Jackson | 68.58 | 63.74 | 73.91 | 426 | Giants |
| 58 | 28 | Kendall Fuller | 68.50 | 62.14 | 75.42 | 556 | Dolphins |
| 59 | 29 | Jonathan Jones | 68.29 | 61.05 | 73.02 | 712 | Patriots |
| 60 | 30 | Darious Williams | 68.07 | 59.80 | 74.58 | 865 | Rams |
| 61 | 31 | Shaquill Griffin | 68.05 | 61.46 | 76.44 | 597 | Vikings |
| 62 | 32 | Cobie Durant | 67.89 | 61.50 | 73.42 | 843 | Rams |
| 63 | 33 | Chamarri Conner | 67.83 | 62.54 | 68.75 | 679 | Chiefs |
| 64 | 34 | Stephon Gilmore | 67.82 | 59.20 | 73.71 | 904 | Vikings |
| 65 | 35 | Paulson Adebo | 67.78 | 63.26 | 76.11 | 436 | Saints |
| 66 | 36 | Isaiah Bolden | 67.76 | 68.19 | 71.86 | 141 | Patriots |
| 67 | 37 | Ahkello Witherspoon | 67.63 | 62.37 | 73.22 | 598 | Rams |
| 68 | 38 | Kris Abrams-Draine | 67.52 | 65.88 | 85.66 | 123 | Broncos |
| 69 | 39 | Darrell Baker Jr. | 67.18 | 62.44 | 72.82 | 626 | Titans |
| 70 | 40 | Tyrique Stevenson | 67.14 | 58.41 | 73.57 | 810 | Bears |
| 71 | 41 | Keisean Nixon | 67.08 | 60.70 | 70.36 | 1077 | Packers |
| 72 | 42 | Josh Newton | 66.86 | 60.70 | 72.74 | 504 | Bengals |
| 73 | 43 | Ja'Quan McMillian | 66.81 | 63.00 | 70.89 | 918 | Broncos |
| 74 | 44 | Beanie Bishop Jr. | 66.46 | 57.63 | 71.70 | 550 | Steelers |
| 75 | 45 | Charvarius Ward | 66.44 | 58.29 | 74.12 | 694 | 49ers |
| 76 | 46 | Amani Oruwariye | 66.29 | 62.30 | 73.86 | 286 | Cowboys |
| 77 | 47 | Cory Trice Jr. | 65.83 | 67.42 | 76.70 | 194 | Steelers |
| 78 | 48 | Cor'Dale Flott | 65.71 | 61.53 | 70.23 | 666 | Giants |
| 79 | 49 | Alex Austin | 65.31 | 62.08 | 77.58 | 234 | Patriots |
| 80 | 50 | Jakorian Bennett | 65.27 | 60.75 | 73.41 | 459 | Raiders |
| 81 | 51 | Avonte Maddox | 65.15 | 60.32 | 72.30 | 352 | Eagles |
| 82 | 52 | Israel Mukuamu | 65.01 | 56.00 | 71.10 | 201 | Cowboys |
| 83 | 53 | Trevon Diggs | 64.99 | 60.27 | 75.01 | 683 | Cowboys |
| 84 | 54 | Ronald Darby | 64.93 | 59.34 | 73.52 | 659 | Jaguars |
| 85 | 55 | Eric Stokes | 64.93 | 62.03 | 70.90 | 588 | Packers |
| 86 | 56 | Joey Porter Jr. | 64.75 | 56.30 | 69.53 | 1038 | Steelers |
| 87 | 57 | Myles Bryant | 64.70 | 61.17 | 70.00 | 156 | Texans |
| 88 | 58 | Cam Hart | 64.61 | 58.52 | 73.03 | 502 | Chargers |
| 89 | 59 | Jaycee Horn | 64.58 | 57.90 | 73.85 | 1034 | Panthers |
| 90 | 60 | Fabian Moreau | 64.40 | 60.20 | 73.41 | 104 | Vikings |
| 91 | 61 | Ja'Sir Taylor | 64.39 | 58.50 | 71.92 | 353 | Chargers |
| 92 | 62 | Brandin Echols | 64.12 | 60.84 | 69.77 | 406 | Jets |
| 93 | 63 | Marshon Lattimore | 64.07 | 58.10 | 74.43 | 687 | Commanders |
| 94 | 64 | Jarvis Brownlee Jr. | 63.81 | 55.90 | 67.92 | 911 | Titans |
| 95 | 65 | Kaiir Elam | 63.78 | 60.92 | 73.10 | 359 | Bills |
| 96 | 66 | Nate Hobbs | 63.68 | 61.34 | 68.06 | 554 | Raiders |
| 97 | 67 | Taron Johnson | 63.67 | 55.96 | 69.11 | 785 | Bills |
| 98 | 68 | Max Melton | 63.65 | 57.77 | 65.29 | 565 | Cardinals |
| 99 | 69 | Roger McCreary | 63.57 | 58.51 | 66.28 | 652 | Titans |
| 100 | 70 | Chidobe Awuzie | 63.53 | 58.71 | 72.89 | 373 | Titans |
| 101 | 71 | Asante Samuel Jr. | 63.48 | 58.99 | 71.91 | 234 | Chargers |
| 102 | 72 | Christian Roland-Wallace | 63.36 | 60.54 | 71.71 | 197 | Chiefs |
| 103 | 73 | Isaac Yiadom | 62.83 | 55.95 | 70.91 | 488 | 49ers |
| 104 | 74 | Dee Alford | 62.64 | 56.06 | 67.91 | 724 | Falcons |
| 105 | 75 | Rasul Douglas | 62.59 | 51.60 | 70.47 | 997 | Bills |
| 106 | 76 | Cameron Mitchell | 62.57 | 56.85 | 66.30 | 371 | Browns |
| 107 | 77 | Dax Hill | 62.56 | 64.51 | 70.42 | 262 | Bengals |
| 108 | 78 | D'Angelo Ross | 62.54 | 63.12 | 69.61 | 184 | Texans |
| 109 | 79 | Starling Thomas V | 62.13 | 60.90 | 61.02 | 817 | Cardinals |
| 110 | 80 | Greg Newsome II | 62.01 | 54.61 | 69.41 | 571 | Browns |

### Rotation/backup (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 111 | 1 | James Pierre | 61.90 | 57.40 | 70.99 | 207 | Steelers |
| 112 | 2 | Jack Jones | 61.61 | 52.90 | 70.11 | 1047 | Raiders |
| 113 | 3 | Montaric Brown | 61.58 | 61.60 | 62.54 | 855 | Jaguars |
| 114 | 4 | Nazeeh Johnson | 61.41 | 54.06 | 64.84 | 547 | Chiefs |
| 115 | 5 | Tre'Davious White | 61.32 | 55.63 | 73.29 | 445 | Ravens |
| 116 | 6 | Deantre Prince | 61.27 | 62.00 | 73.21 | 101 | Jaguars |
| 117 | 7 | Terrion Arnold | 60.80 | 50.20 | 68.21 | 1021 | Lions |
| 118 | 8 | Ka'dar Hollman | 60.75 | 58.80 | 69.47 | 116 | Texans |
| 119 | 9 | Tre Brown | 60.46 | 56.15 | 69.84 | 290 | Seahawks |
| 120 | 10 | Riley Moss | 60.24 | 56.00 | 65.91 | 912 | Broncos |
| 121 | 11 | Sean Murphy-Bunting | 59.52 | 53.36 | 66.07 | 725 | Cardinals |
| 122 | 12 | Deonte Banks | 59.48 | 50.52 | 68.30 | 788 | Giants |
| 123 | 13 | Greg Stroman Jr. | 59.24 | 59.76 | 69.71 | 130 | Giants |
| 124 | 14 | Storm Duck | 58.86 | 56.57 | 62.63 | 359 | Dolphins |
| 125 | 15 | Martin Emerson Jr. | 58.71 | 48.40 | 65.75 | 827 | Browns |
| 126 | 16 | Josh Blackwell | 58.69 | 60.48 | 66.77 | 102 | Bears |
| 127 | 17 | Shemar Jean-Charles | 58.52 | 58.17 | 70.06 | 143 | Saints |
| 128 | 18 | Marco Wilson | 58.33 | 51.17 | 64.92 | 242 | Bengals |
| 129 | 19 | Ja'Marcus Ingram | 58.21 | 54.20 | 69.06 | 217 | Bills |
| 130 | 20 | Benjamin St-Juste | 57.89 | 46.50 | 67.94 | 859 | Commanders |
| 131 | 21 | Josh Jobe | 57.85 | 52.60 | 70.09 | 443 | Seahawks |
| 132 | 22 | Josh Wallace | 57.84 | 52.58 | 62.62 | 165 | Rams |
| 133 | 23 | Michael Carter II | 57.68 | 50.89 | 63.82 | 285 | Jets |
| 134 | 24 | Darnay Holmes | 57.20 | 53.69 | 65.22 | 298 | Raiders |
| 135 | 25 | Emmanuel Forbes | 56.78 | 51.15 | 67.75 | 160 | Rams |
| 136 | 26 | Cameron Sutton | 56.59 | 48.82 | 65.37 | 273 | Steelers |
| 137 | 27 | Nick McCloud | 56.54 | 52.81 | 63.66 | 224 | 49ers |
| 138 | 28 | Chau Smith-Wade | 56.16 | 53.72 | 62.74 | 301 | Panthers |
| 139 | 29 | Donte Jackson | 55.91 | 45.10 | 66.70 | 832 | Steelers |
| 140 | 30 | Michael Davis | 55.61 | 47.59 | 66.65 | 139 | Commanders |
| 141 | 31 | Caleb Farley | 55.18 | 58.60 | 55.77 | 169 | Panthers |
| 142 | 32 | Noah Igbinoghene | 54.70 | 49.60 | 63.87 | 971 | Commanders |
| 143 | 33 | Dane Jackson | 54.66 | 46.20 | 66.38 | 282 | Panthers |
| 144 | 34 | Alontae Taylor | 52.69 | 40.00 | 64.22 | 1075 | Saints |
| 145 | 35 | Decamerion Richardson | 52.61 | 45.35 | 63.78 | 559 | Raiders |
| 146 | 36 | Kindle Vildor | 52.39 | 50.72 | 55.54 | 316 | Lions |
| 147 | 37 | L'Jarius Sneed | 51.39 | 42.31 | 64.77 | 301 | Titans |
| 148 | 38 | Tyrek Funderburk | 51.34 | 58.06 | 52.63 | 168 | Buccaneers |
| 149 | 39 | Andrew Booth Jr. | 50.43 | 47.48 | 63.00 | 118 | Cowboys |
| 150 | 40 | Cam Smith | 49.05 | 47.99 | 59.65 | 133 | Dolphins |
| 151 | 41 | Nehemiah Pritchett | 47.26 | 49.56 | 60.59 | 151 | Seahawks |
| 152 | 42 | Caelen Carson | 45.12 | 47.12 | 58.80 | 252 | Cowboys |

## DI — Defensive Interior

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Quinnen Williams | 88.93 | 86.92 | 86.77 | 722 | Jets |
| 2 | 2 | Kobie Turner | 87.05 | 84.01 | 85.09 | 919 | Rams |
| 3 | 3 | Dexter Lawrence | 84.54 | 86.58 | 81.01 | 551 | Giants |
| 4 | 4 | Jeffery Simmons | 84.45 | 86.06 | 80.65 | 806 | Titans |
| 5 | 5 | Chris Jones | 84.43 | 85.69 | 78.17 | 886 | Chiefs |
| 6 | 6 | Leonard Williams | 84.07 | 84.74 | 80.16 | 750 | Seahawks |
| 7 | 7 | Jalen Carter | 83.84 | 86.10 | 76.58 | 1026 | Eagles |
| 8 | 8 | DeForest Buckner | 83.74 | 83.04 | 82.37 | 579 | Colts |
| 9 | 9 | Zach Sieler | 82.38 | 77.67 | 83.26 | 749 | Dolphins |
| 10 | 10 | Vita Vea | 81.96 | 81.61 | 78.00 | 756 | Buccaneers |
| 11 | 11 | Cameron Heyward | 81.86 | 77.13 | 83.35 | 838 | Steelers |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Ed Oliver | 78.19 | 70.73 | 81.60 | 727 | Bills |
| 13 | 2 | Alim McNeill | 77.54 | 79.50 | 72.70 | 631 | Lions |
| 14 | 3 | Christian Wilkins | 77.16 | 75.95 | 80.43 | 246 | Raiders |
| 15 | 4 | Jalen Redmond | 76.71 | 68.30 | 83.63 | 236 | Vikings |
| 16 | 5 | Grover Stewart | 76.65 | 73.74 | 76.68 | 690 | Colts |
| 17 | 6 | T'Vondre Sweat | 76.58 | 77.85 | 70.32 | 699 | Titans |
| 18 | 7 | Milton Williams | 75.46 | 66.24 | 79.68 | 628 | Eagles |
| 19 | 8 | Zach Allen | 75.00 | 64.32 | 81.61 | 1031 | Broncos |
| 20 | 9 | Kenny Clark | 74.87 | 67.66 | 77.08 | 725 | Packers |
| 21 | 10 | Jordan Davis | 74.75 | 71.70 | 73.74 | 430 | Eagles |
| 22 | 11 | Devonte Wyatt | 74.46 | 63.63 | 81.71 | 366 | Packers |
| 23 | 12 | Osa Odighizuwa | 74.21 | 65.39 | 78.03 | 859 | Cowboys |
| 24 | 13 | Michael Pierce | 74.09 | 70.34 | 78.49 | 254 | Ravens |

### Starter (79 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 25 | 1 | B.J. Hill | 73.72 | 68.67 | 75.19 | 710 | Bengals |
| 26 | 2 | Grady Jarrett | 73.58 | 65.63 | 79.72 | 744 | Falcons |
| 27 | 3 | Braden Fiske | 72.84 | 56.54 | 84.14 | 700 | Rams |
| 28 | 4 | Calais Campbell | 72.66 | 57.02 | 84.83 | 616 | Dolphins |
| 29 | 5 | John Franklin-Myers | 72.61 | 61.73 | 78.96 | 569 | Broncos |
| 30 | 6 | Jonathan Allen | 72.19 | 61.15 | 82.35 | 421 | Commanders |
| 31 | 7 | Sebastian Joseph-Day | 71.59 | 60.17 | 78.25 | 483 | Titans |
| 32 | 8 | David Onyemata | 71.49 | 62.31 | 76.72 | 567 | Falcons |
| 33 | 9 | Travis Jones | 71.45 | 66.13 | 72.26 | 675 | Ravens |
| 34 | 10 | D.J. Jones | 71.23 | 60.52 | 77.76 | 510 | Broncos |
| 35 | 11 | DJ Reader | 71.20 | 66.48 | 74.22 | 566 | Lions |
| 36 | 12 | Daron Payne | 71.18 | 59.09 | 78.26 | 796 | Commanders |
| 37 | 13 | Poona Ford | 70.78 | 64.75 | 74.98 | 652 | Chargers |
| 38 | 14 | Christian Barmore | 70.78 | 63.35 | 82.51 | 123 | Patriots |
| 39 | 15 | Teair Tart | 70.69 | 64.14 | 73.88 | 378 | Chargers |
| 40 | 16 | Keeanu Benton | 70.64 | 68.48 | 67.80 | 671 | Steelers |
| 41 | 17 | Shelby Harris | 70.35 | 58.00 | 79.94 | 527 | Browns |
| 42 | 18 | Gervon Dexter Sr. | 70.27 | 63.40 | 73.62 | 616 | Bears |
| 43 | 19 | Jaquelin Roy | 70.07 | 60.70 | 84.74 | 141 | Patriots |
| 44 | 20 | Dalvin Tomlinson | 70.03 | 61.97 | 74.61 | 609 | Browns |
| 45 | 21 | Calijah Kancey | 69.75 | 53.58 | 84.30 | 595 | Buccaneers |
| 46 | 22 | Desjuan Johnson | 69.72 | 59.73 | 80.59 | 155 | Rams |
| 47 | 23 | Leonard Taylor III | 69.35 | 59.70 | 77.52 | 261 | Jets |
| 48 | 24 | Levi Onwuzurike | 69.07 | 64.87 | 69.90 | 697 | Lions |
| 49 | 25 | Karl Brooks | 68.93 | 56.31 | 76.55 | 459 | Packers |
| 50 | 26 | Javon Hargrave | 68.18 | 58.86 | 80.73 | 104 | 49ers |
| 51 | 27 | Jarran Reed | 67.84 | 56.36 | 74.67 | 679 | Seahawks |
| 52 | 28 | Malcolm Roach | 67.51 | 57.00 | 75.72 | 524 | Broncos |
| 53 | 29 | Byron Murphy II | 67.36 | 58.70 | 74.53 | 457 | Seahawks |
| 54 | 30 | William Gholston | 67.32 | 55.28 | 74.36 | 205 | Buccaneers |
| 55 | 31 | Mario Edwards Jr. | 66.57 | 54.39 | 76.81 | 519 | Texans |
| 56 | 32 | Tim Settle | 66.50 | 52.83 | 75.65 | 685 | Texans |
| 57 | 33 | Dante Stills | 66.28 | 55.58 | 73.59 | 532 | Cardinals |
| 58 | 34 | Jowon Briggs | 66.04 | 64.70 | 78.92 | 133 | Browns |
| 59 | 35 | Naquan Jones | 66.03 | 55.15 | 78.03 | 260 | Cardinals |
| 60 | 36 | Jer'Zhan Newton | 66.01 | 51.95 | 75.07 | 586 | Commanders |
| 61 | 37 | Zach Harrison | 65.96 | 56.56 | 70.36 | 272 | Falcons |
| 62 | 38 | Tyler Davis | 65.95 | 51.77 | 75.14 | 354 | Rams |
| 63 | 39 | DaQuan Jones | 65.88 | 59.98 | 69.84 | 629 | Bills |
| 64 | 40 | Evan Anderson | 65.68 | 57.72 | 74.51 | 267 | 49ers |
| 65 | 41 | Bobby Brown III | 65.43 | 59.60 | 69.20 | 513 | Rams |
| 66 | 42 | Roy Robertson-Harris | 65.42 | 52.35 | 73.50 | 398 | Seahawks |
| 67 | 43 | Elijah Garcia | 65.31 | 59.23 | 81.40 | 143 | Giants |
| 68 | 44 | Thomas Booker IV | 65.30 | 54.05 | 74.62 | 172 | Eagles |
| 69 | 45 | Jeremiah Pharms Jr. | 65.17 | 56.69 | 71.59 | 457 | Patriots |
| 70 | 46 | Roy Lopez | 65.12 | 54.21 | 72.67 | 464 | Cardinals |
| 71 | 47 | Kevin Givens | 64.95 | 54.35 | 76.79 | 185 | 49ers |
| 72 | 48 | Folorunso Fatukasi | 64.69 | 50.22 | 77.58 | 366 | Texans |
| 73 | 49 | Larry Ogunjobi | 64.59 | 48.16 | 76.84 | 550 | Steelers |
| 74 | 50 | Khyiris Tonga | 64.46 | 58.12 | 70.97 | 229 | Cardinals |
| 75 | 51 | Harrison Phillips | 64.38 | 53.21 | 70.55 | 701 | Vikings |
| 76 | 52 | Khalil Davis | 64.33 | 53.06 | 77.03 | 209 | 49ers |
| 77 | 53 | DaVon Hamilton | 64.11 | 51.39 | 75.02 | 626 | Jaguars |
| 78 | 54 | Javon Kinlaw | 64.07 | 53.54 | 72.18 | 695 | Jets |
| 79 | 55 | Kentavius Street | 63.94 | 53.29 | 73.35 | 280 | Falcons |
| 80 | 56 | A'Shawn Robinson | 63.69 | 49.20 | 75.42 | 761 | Panthers |
| 81 | 57 | Morgan Fox | 63.64 | 47.64 | 74.64 | 619 | Chargers |
| 82 | 58 | Taven Bryan | 63.55 | 53.65 | 68.68 | 340 | Colts |
| 83 | 59 | Solomon Thomas | 63.44 | 48.78 | 73.68 | 458 | Jets |
| 84 | 60 | Andrew Billings | 63.37 | 56.90 | 70.84 | 297 | Bears |
| 85 | 61 | Logan Hall | 63.31 | 54.90 | 66.73 | 571 | Buccaneers |
| 86 | 62 | Neville Gallimore | 63.31 | 51.55 | 70.89 | 308 | Rams |
| 87 | 63 | Da'Shawn Hand | 62.86 | 55.71 | 69.12 | 564 | Dolphins |
| 88 | 64 | Byron Cowart | 62.78 | 49.52 | 72.73 | 335 | Bears |
| 89 | 65 | James Lynch | 62.72 | 51.85 | 70.23 | 243 | Titans |
| 90 | 66 | Colby Wooden | 62.69 | 54.30 | 69.01 | 260 | Packers |
| 91 | 67 | Eddie Goldman | 62.67 | 48.30 | 73.34 | 330 | Falcons |
| 92 | 68 | Mike Pennel | 62.67 | 50.23 | 73.63 | 365 | Chiefs |
| 93 | 69 | Adetomiwa Adebawore | 62.61 | 57.00 | 71.15 | 137 | Colts |
| 94 | 70 | Bryan Bresee | 62.53 | 49.37 | 70.68 | 708 | Saints |
| 95 | 71 | Greg Gaines | 62.52 | 52.82 | 68.04 | 421 | Buccaneers |
| 96 | 72 | Maliek Collins | 62.50 | 50.45 | 70.03 | 715 | 49ers |
| 97 | 73 | Jeremiah Ledbetter | 62.49 | 53.22 | 71.70 | 441 | Jaguars |
| 98 | 74 | McKinnley Jackson | 62.26 | 53.10 | 71.12 | 248 | Bengals |
| 99 | 75 | Maurice Hurst | 62.26 | 53.28 | 76.47 | 164 | Browns |
| 100 | 76 | Khalen Saunders | 62.20 | 51.53 | 70.46 | 460 | Saints |
| 101 | 77 | Mazi Smith | 62.17 | 50.32 | 69.03 | 524 | Cowboys |
| 102 | 78 | Zacch Pickens | 62.11 | 53.55 | 71.56 | 228 | Bears |
| 103 | 79 | Sheldon Rankins | 62.01 | 52.14 | 73.24 | 287 | Bengals |

### Rotation/backup (64 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 104 | 1 | Jonathan Bullard | 61.73 | 47.55 | 72.10 | 590 | Vikings |
| 105 | 2 | Quinton Jefferson | 61.69 | 47.22 | 73.98 | 258 | Bills |
| 106 | 3 | Patrick O'Connor | 61.61 | 50.91 | 74.13 | 235 | Lions |
| 107 | 4 | Davon Godchaux | 61.52 | 47.31 | 70.73 | 680 | Patriots |
| 108 | 5 | Jonah Laulu | 61.50 | 48.57 | 71.79 | 474 | Raiders |
| 109 | 6 | Moro Ojomo | 61.49 | 60.87 | 61.09 | 465 | Eagles |
| 110 | 7 | Adam Butler | 61.43 | 45.66 | 72.21 | 856 | Raiders |
| 111 | 8 | Austin Johnson | 61.33 | 50.05 | 69.74 | 353 | Bills |
| 112 | 9 | Tommy Togiai | 61.25 | 52.19 | 73.80 | 280 | Texans |
| 113 | 10 | Broderick Washington | 61.20 | 49.86 | 67.53 | 488 | Ravens |
| 114 | 11 | Jordan Phillips | 61.20 | 50.06 | 72.51 | 185 | Bills |
| 115 | 12 | Jonah Williams | 61.19 | 53.07 | 69.37 | 108 | Lions |
| 116 | 13 | Justin Jones | 61.19 | 53.01 | 72.61 | 100 | Cardinals |
| 117 | 14 | Isaiahh Loudermilk | 60.94 | 54.28 | 64.59 | 255 | Steelers |
| 118 | 15 | Tershawn Wharton | 60.93 | 50.76 | 68.92 | 733 | Chiefs |
| 119 | 16 | Nathan Shepherd | 60.80 | 47.10 | 70.09 | 567 | Saints |
| 120 | 17 | Jalyn Holmes | 60.80 | 50.20 | 74.40 | 337 | Commanders |
| 121 | 18 | Elijah Chatman | 60.80 | 54.25 | 62.35 | 423 | Giants |
| 122 | 19 | D.J. Davidson | 60.67 | 53.53 | 69.30 | 261 | Giants |
| 123 | 20 | Ta'Quon Graham | 60.61 | 58.14 | 64.31 | 193 | Falcons |
| 124 | 21 | Montravius Adams | 60.35 | 50.98 | 68.72 | 207 | Steelers |
| 125 | 22 | Jerry Tillery | 60.34 | 51.48 | 64.68 | 482 | Vikings |
| 126 | 23 | Bilal Nichols | 60.17 | 49.81 | 72.00 | 173 | Cardinals |
| 127 | 24 | Linval Joseph | 60.15 | 43.78 | 76.45 | 264 | Cowboys |
| 128 | 25 | Carlos Watkins | 60.08 | 52.90 | 69.32 | 228 | Cowboys |
| 129 | 26 | John Ridgeway | 59.98 | 51.22 | 66.56 | 263 | Saints |
| 130 | 27 | DeShawn Williams | 59.91 | 47.63 | 70.50 | 338 | Panthers |
| 131 | 28 | Jordan Jefferson | 59.85 | 60.07 | 65.20 | 151 | Jaguars |
| 132 | 29 | Raekwon Davis | 59.72 | 50.78 | 63.89 | 349 | Colts |
| 133 | 30 | Bruce Hector | 59.71 | 55.88 | 73.54 | 118 | Jets |
| 134 | 31 | Kurt Hinish | 59.67 | 52.98 | 65.94 | 231 | Texans |
| 135 | 32 | Derrick Nnadi | 59.29 | 49.67 | 63.91 | 248 | Chiefs |
| 136 | 33 | Johnathan Hankins | 59.26 | 42.96 | 72.91 | 389 | Seahawks |
| 137 | 34 | John Jenkins | 59.18 | 42.21 | 71.38 | 606 | Raiders |
| 138 | 35 | Shy Tuttle | 58.83 | 46.45 | 67.39 | 610 | Panthers |
| 139 | 36 | C.J. Brewer | 58.72 | 51.15 | 67.17 | 159 | Buccaneers |
| 140 | 37 | Benito Jones | 58.68 | 48.15 | 64.21 | 481 | Dolphins |
| 141 | 38 | Ruke Orhorhoro | 58.44 | 56.38 | 66.08 | 147 | Falcons |
| 142 | 39 | Tyler Lacy | 58.39 | 51.69 | 63.90 | 340 | Jaguars |
| 143 | 40 | Maason Smith | 58.24 | 50.18 | 68.36 | 384 | Jaguars |
| 144 | 41 | Dean Lowry | 58.16 | 47.68 | 69.89 | 159 | Steelers |
| 145 | 42 | Zach Carter | 57.80 | 51.49 | 62.28 | 263 | Raiders |
| 146 | 43 | Sheldon Day | 57.45 | 48.83 | 68.49 | 339 | Commanders |
| 147 | 44 | Jordan Elliott | 57.16 | 49.20 | 61.29 | 440 | 49ers |
| 148 | 45 | Jay Tufele | 56.72 | 50.82 | 64.79 | 242 | Bengals |
| 149 | 46 | Jordon Riley | 56.69 | 51.09 | 64.92 | 248 | Giants |
| 150 | 47 | Ben Stille | 56.22 | 53.77 | 66.73 | 120 | Cardinals |
| 151 | 48 | Daniel Ekuale | 55.96 | 47.69 | 65.24 | 723 | Patriots |
| 152 | 49 | DeWayne Carter | 55.70 | 48.84 | 64.62 | 315 | Bills |
| 153 | 50 | Jordan Jackson | 55.45 | 46.06 | 59.83 | 329 | Broncos |
| 154 | 51 | Rakeem Nunez-Roches | 55.44 | 42.18 | 65.22 | 608 | Giants |
| 155 | 52 | Jaden Crumedy | 55.16 | 54.18 | 70.66 | 121 | Panthers |
| 156 | 53 | L.J. Collier | 55.04 | 45.72 | 67.13 | 588 | Cardinals |
| 157 | 54 | Eric Johnson | 54.94 | 52.89 | 57.28 | 178 | Patriots |
| 158 | 55 | Darius Robinson | 54.70 | 55.18 | 65.77 | 184 | Cardinals |
| 159 | 56 | Chris Williams | 54.15 | 46.47 | 63.19 | 367 | Bears |
| 160 | 57 | Otito Ogbonnia | 53.76 | 46.87 | 61.18 | 538 | Chargers |
| 161 | 58 | Kris Jenkins | 53.53 | 49.90 | 60.66 | 496 | Bengals |
| 162 | 59 | Phidarian Mathis | 53.02 | 51.38 | 58.84 | 257 | Jets |
| 163 | 60 | Keondre Coburn | 52.92 | 55.48 | 52.13 | 125 | Titans |
| 164 | 61 | LaBryan Ray | 51.66 | 43.49 | 55.56 | 626 | Panthers |
| 165 | 62 | Matthew Butler | 51.30 | 55.88 | 56.07 | 101 | Raiders |
| 166 | 63 | Mekhi Wingo | 50.60 | 52.65 | 50.62 | 177 | Lions |
| 167 | 64 | Kalia Davis | 50.29 | 49.73 | 54.98 | 259 | 49ers |

## ED — Edge

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Bosa | 92.84 | 95.17 | 88.33 | 693 | 49ers |
| 2 | 2 | Myles Garrett | 92.41 | 94.59 | 86.44 | 822 | Browns |
| 3 | 3 | Will Anderson Jr. | 91.69 | 93.89 | 86.15 | 645 | Texans |
| 4 | 4 | Micah Parsons | 91.24 | 89.10 | 90.57 | 694 | Cowboys |
| 5 | 5 | T.J. Watt | 90.64 | 92.51 | 86.17 | 1002 | Steelers |
| 6 | 6 | Jared Verse | 89.08 | 94.25 | 80.55 | 933 | Rams |
| 7 | 7 | Greg Rousseau | 86.98 | 89.64 | 81.27 | 861 | Bills |
| 8 | 8 | Khalil Mack | 86.34 | 83.39 | 84.20 | 668 | Chargers |
| 9 | 9 | Trey Hendrickson | 85.54 | 80.89 | 85.15 | 823 | Bengals |
| 10 | 10 | Rashan Gary | 85.37 | 83.05 | 84.40 | 670 | Packers |
| 11 | 11 | Aidan Hutchinson | 84.84 | 86.17 | 85.64 | 280 | Lions |
| 12 | 12 | Joey Bosa | 84.40 | 84.38 | 86.04 | 503 | Chargers |
| 13 | 13 | Maxx Crosby | 83.21 | 85.96 | 79.21 | 766 | Raiders |
| 14 | 14 | Danielle Hunter | 83.07 | 76.50 | 83.77 | 859 | Texans |
| 15 | 15 | Montez Sweat | 80.89 | 78.22 | 79.05 | 616 | Bears |
| 16 | 16 | Nik Bonitto | 80.45 | 67.07 | 86.50 | 761 | Broncos |
| 17 | 17 | Yaya Diaby | 80.19 | 78.37 | 77.16 | 841 | Buccaneers |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Alex Highsmith | 79.85 | 81.13 | 77.01 | 592 | Steelers |
| 19 | 2 | Brian Burns | 78.84 | 71.75 | 80.46 | 865 | Giants |
| 20 | 3 | Von Miller | 78.56 | 65.27 | 87.27 | 332 | Bills |
| 21 | 4 | Chop Robinson | 78.44 | 69.60 | 80.92 | 565 | Dolphins |
| 22 | 5 | Jonathan Greenard | 77.26 | 69.59 | 80.72 | 969 | Vikings |
| 23 | 6 | Nolan Smith | 77.21 | 70.46 | 78.05 | 725 | Eagles |
| 24 | 7 | Zaven Collins | 76.07 | 71.75 | 75.00 | 600 | Cardinals |
| 25 | 8 | Odafe Oweh | 75.90 | 75.75 | 72.17 | 683 | Ravens |
| 26 | 9 | Tuli Tuipulotu | 75.80 | 72.19 | 74.17 | 774 | Chargers |
| 27 | 10 | Javon Solomon | 75.74 | 64.20 | 84.54 | 141 | Bills |
| 28 | 11 | Dondrea Tillman | 75.57 | 63.13 | 85.08 | 275 | Broncos |
| 29 | 12 | Will McDonald IV | 75.48 | 64.10 | 80.74 | 756 | Jets |
| 30 | 13 | Brandon Graham | 75.46 | 67.49 | 80.40 | 311 | Eagles |
| 31 | 14 | Za'Darius Smith | 75.24 | 65.70 | 78.48 | 655 | Lions |
| 32 | 15 | Brenton Cox Jr. | 75.24 | 64.18 | 88.91 | 187 | Packers |
| 33 | 16 | DeMarcus Lawrence | 74.71 | 74.40 | 77.31 | 167 | Cowboys |
| 34 | 17 | Kyle Van Noy | 74.57 | 58.30 | 82.89 | 696 | Ravens |
| 35 | 18 | George Karlaftis | 74.52 | 65.98 | 76.77 | 953 | Chiefs |
| 36 | 19 | Boye Mafe | 74.41 | 69.94 | 74.81 | 607 | Seahawks |
| 37 | 20 | Nick Herbig | 74.19 | 65.29 | 79.87 | 433 | Steelers |

### Starter (62 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 38 | 1 | Jonathon Cooper | 73.79 | 65.75 | 76.26 | 882 | Broncos |
| 39 | 2 | Travon Walker | 73.09 | 70.93 | 70.76 | 911 | Jaguars |
| 40 | 3 | Josh Sweat | 72.93 | 65.45 | 74.56 | 775 | Eagles |
| 41 | 4 | Chase Young | 72.88 | 74.45 | 70.12 | 740 | Saints |
| 42 | 5 | Andrew Van Ginkel | 72.66 | 57.64 | 80.31 | 973 | Vikings |
| 43 | 6 | James Houston | 72.51 | 58.40 | 88.79 | 141 | Browns |
| 44 | 7 | Carl Granderson | 72.49 | 65.93 | 73.40 | 825 | Saints |
| 45 | 8 | Jaelan Phillips | 72.28 | 68.15 | 80.73 | 134 | Dolphins |
| 46 | 9 | Arnold Ebiketie | 71.88 | 62.88 | 74.71 | 543 | Falcons |
| 47 | 10 | Michael Hoecht | 71.60 | 60.69 | 75.70 | 705 | Rams |
| 48 | 11 | Isaiah McGuire | 71.43 | 67.59 | 75.76 | 469 | Browns |
| 49 | 12 | Byron Young | 71.07 | 58.98 | 76.10 | 936 | Rams |
| 50 | 13 | Jadeveon Clowney | 69.97 | 65.96 | 71.29 | 650 | Panthers |
| 51 | 14 | Haason Reddick | 69.92 | 54.70 | 81.10 | 392 | Jets |
| 52 | 15 | Dorance Armstrong | 69.82 | 60.31 | 72.82 | 747 | Commanders |
| 53 | 16 | Harold Landry III | 69.80 | 59.61 | 73.34 | 878 | Titans |
| 54 | 17 | Cameron Jordan | 69.25 | 59.38 | 72.76 | 565 | Saints |
| 55 | 18 | Uchenna Nwosu | 69.14 | 62.22 | 75.89 | 190 | Seahawks |
| 56 | 19 | Darrell Taylor | 69.13 | 55.43 | 76.16 | 374 | Bears |
| 57 | 20 | Dennis Gardeck | 68.89 | 54.78 | 81.39 | 206 | Cardinals |
| 58 | 21 | Kayvon Thibodeaux | 68.71 | 67.37 | 68.54 | 593 | Giants |
| 59 | 22 | Dante Fowler Jr. | 68.42 | 54.09 | 75.21 | 642 | Commanders |
| 60 | 23 | Matthew Judon | 68.24 | 50.11 | 82.11 | 655 | Falcons |
| 61 | 24 | Jonah Elliss | 67.95 | 60.23 | 69.56 | 441 | Broncos |
| 62 | 25 | Azeez Ojulari | 67.49 | 59.10 | 76.76 | 391 | Giants |
| 63 | 26 | Kwity Paye | 67.36 | 64.40 | 67.63 | 667 | Colts |
| 64 | 27 | Bryce Huff | 67.36 | 62.25 | 71.10 | 298 | Eagles |
| 65 | 28 | Chris Braswell | 67.18 | 60.45 | 68.01 | 335 | Buccaneers |
| 66 | 29 | Leonard Floyd | 67.16 | 52.51 | 74.20 | 604 | 49ers |
| 67 | 30 | Lukas Van Ness | 66.94 | 59.77 | 68.11 | 458 | Packers |
| 68 | 31 | Preston Smith | 66.77 | 53.67 | 72.60 | 469 | Steelers |
| 69 | 32 | Tyree Wilson | 66.75 | 65.45 | 63.97 | 524 | Raiders |
| 70 | 33 | Ogbo Okoronkwo | 66.67 | 54.33 | 73.06 | 464 | Browns |
| 71 | 34 | Derick Hall | 66.64 | 57.52 | 69.35 | 673 | Seahawks |
| 72 | 35 | Charles Snowden | 66.54 | 54.07 | 72.40 | 405 | Raiders |
| 73 | 36 | Victor Dimukeje | 66.51 | 60.12 | 71.16 | 157 | Cardinals |
| 74 | 37 | Jacob Martin | 66.17 | 57.71 | 72.36 | 222 | Bears |
| 75 | 38 | Dayo Odeyingbo | 66.12 | 60.74 | 65.88 | 746 | Colts |
| 76 | 39 | Jamin Davis | 65.94 | 57.15 | 73.11 | 107 | Jets |
| 77 | 40 | Laiatu Latu | 65.75 | 64.49 | 62.27 | 618 | Colts |
| 78 | 41 | A.J. Epenesa | 65.48 | 57.55 | 67.67 | 712 | Bills |
| 79 | 42 | Carl Lawson | 65.42 | 55.08 | 73.60 | 402 | Cowboys |
| 80 | 43 | Kingsley Enagbare | 64.71 | 59.34 | 64.47 | 538 | Packers |
| 81 | 44 | Tyquan Lewis | 64.56 | 58.73 | 70.47 | 355 | Colts |
| 82 | 45 | Dallas Turner | 64.43 | 59.16 | 64.10 | 310 | Vikings |
| 83 | 46 | Arik Armstead | 64.34 | 56.37 | 66.14 | 569 | Jaguars |
| 84 | 47 | Arden Key | 64.27 | 59.05 | 64.43 | 734 | Titans |
| 85 | 48 | Sam Hubbard | 64.17 | 55.01 | 69.52 | 521 | Bengals |
| 86 | 49 | Xavier Thomas | 64.07 | 57.87 | 67.62 | 208 | Cardinals |
| 87 | 50 | Julian Okwara | 63.97 | 56.49 | 71.48 | 286 | Cardinals |
| 88 | 51 | Derek Barnett | 63.86 | 59.51 | 66.50 | 413 | Texans |
| 89 | 52 | Felix Anudike-Uzomah | 63.84 | 60.07 | 62.33 | 344 | Chiefs |
| 90 | 53 | Keion White | 63.55 | 62.89 | 60.00 | 830 | Patriots |
| 91 | 54 | Deatrich Wise Jr. | 63.48 | 55.22 | 66.87 | 409 | Patriots |
| 92 | 55 | Clelin Ferrell | 63.37 | 57.97 | 63.36 | 443 | Commanders |
| 93 | 56 | Joseph Ossai | 63.21 | 58.60 | 62.36 | 573 | Bengals |
| 94 | 57 | Baron Browning | 62.97 | 58.61 | 66.13 | 378 | Cardinals |
| 95 | 58 | Joe Tryon-Shoyinka | 62.71 | 57.02 | 63.24 | 570 | Buccaneers |
| 96 | 59 | Jalyx Hunt | 62.67 | 59.57 | 63.79 | 320 | Eagles |
| 97 | 60 | Emmanuel Ogbah | 62.31 | 54.49 | 66.83 | 734 | Dolphins |
| 98 | 61 | Arron Mosby | 62.19 | 59.09 | 67.52 | 150 | Packers |
| 99 | 62 | Anfernee Jennings | 62.02 | 57.21 | 62.38 | 831 | Patriots |

### Rotation/backup (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 100 | 1 | Austin Booker | 61.61 | 57.65 | 60.26 | 283 | Bears |
| 101 | 2 | Charles Omenihu | 61.59 | 55.22 | 67.81 | 303 | Chiefs |
| 102 | 3 | Dre'Mont Jones | 61.35 | 50.65 | 65.30 | 617 | Seahawks |
| 103 | 4 | Mike Danna | 61.22 | 56.94 | 62.02 | 581 | Chiefs |
| 104 | 5 | Bud Dupree | 60.95 | 49.40 | 67.13 | 570 | Chargers |
| 105 | 6 | K'Lavon Chaisson | 60.93 | 57.76 | 61.68 | 508 | Raiders |
| 106 | 7 | Anthony Nelson | 60.51 | 55.42 | 60.04 | 624 | Buccaneers |
| 107 | 8 | Payton Turner | 60.45 | 60.17 | 63.87 | 335 | Saints |
| 108 | 9 | Janarius Robinson | 60.18 | 57.04 | 65.33 | 109 | Raiders |
| 109 | 10 | Sam Okuayinonu | 60.12 | 57.64 | 62.58 | 451 | 49ers |
| 110 | 11 | Lorenzo Carter | 60.04 | 53.62 | 62.72 | 410 | Falcons |
| 111 | 12 | D.J. Wonnum | 59.85 | 56.42 | 63.66 | 453 | Panthers |
| 112 | 13 | Tyus Bowser | 59.76 | 49.90 | 68.92 | 276 | Dolphins |
| 113 | 14 | Myles Murphy | 59.45 | 58.35 | 58.47 | 353 | Bengals |
| 114 | 15 | Yetur Gross-Matos | 59.17 | 57.91 | 60.43 | 367 | 49ers |
| 115 | 16 | Micheal Clemons | 58.64 | 54.04 | 58.30 | 624 | Jets |
| 116 | 17 | Tavius Robinson | 58.36 | 55.14 | 56.42 | 548 | Ravens |
| 117 | 18 | Ali Gaye | 58.18 | 54.73 | 57.75 | 177 | Titans |
| 118 | 19 | DeMarcus Walker | 58.05 | 46.76 | 62.45 | 738 | Bears |
| 119 | 20 | Dawuane Smoot | 57.93 | 52.64 | 61.19 | 386 | Bills |
| 120 | 21 | Javontae Jean-Baptiste | 57.15 | 56.73 | 55.12 | 248 | Commanders |
| 121 | 22 | Josh Paschal | 56.68 | 57.32 | 54.86 | 613 | Lions |
| 122 | 23 | Charles Harris | 56.54 | 52.87 | 61.16 | 474 | Eagles |
| 123 | 24 | Casey Toohill | 56.52 | 53.62 | 56.44 | 249 | Bills |
| 124 | 25 | Quinton Bell | 55.78 | 54.02 | 55.94 | 258 | Dolphins |
| 125 | 26 | Cam Gill | 55.71 | 55.19 | 57.01 | 222 | Panthers |
| 126 | 27 | David Ojabo | 55.56 | 57.68 | 58.03 | 292 | Ravens |
| 127 | 28 | Robert Beal Jr. | 55.52 | 57.55 | 55.76 | 149 | 49ers |
| 128 | 29 | Al-Quadin Muhammad | 55.42 | 54.44 | 55.61 | 293 | Lions |
| 129 | 30 | Tyrus Wheat | 55.35 | 57.25 | 59.90 | 165 | Cowboys |
| 130 | 31 | Alex Wright | 55.14 | 58.73 | 54.68 | 103 | Browns |
| 131 | 32 | Eric Watts | 54.61 | 57.99 | 50.64 | 231 | Jets |
| 132 | 33 | DJ Johnson | 53.98 | 54.98 | 52.29 | 392 | Panthers |
| 133 | 34 | Dylan Horton | 53.98 | 57.18 | 51.08 | 217 | Texans |
| 134 | 35 | Jaylen Harrell | 53.87 | 54.88 | 48.61 | 286 | Titans |
| 135 | 36 | Tomon Fox | 53.80 | 57.20 | 54.03 | 207 | Giants |
| 136 | 37 | Malik Herring | 53.77 | 55.57 | 54.52 | 193 | Chiefs |
| 137 | 38 | Marshawn Kneeland | 53.34 | 57.18 | 52.17 | 255 | Cowboys |
| 138 | 39 | James Smith-Williams | 52.95 | 52.07 | 54.74 | 306 | Falcons |
| 139 | 40 | Brent Urban | 51.84 | 45.70 | 53.25 | 209 | Ravens |
| 140 | 41 | Jeremiah Moon | 49.44 | 55.50 | 47.88 | 117 | Steelers |
| 141 | 42 | Demone Harris | 48.89 | 52.89 | 48.43 | 216 | Falcons |
| 142 | 43 | Myles Cole | 48.39 | 56.12 | 47.29 | 135 | Jaguars |

## G — Guard

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Lindstrom | 95.45 | 93.50 | 92.20 | 1099 | Falcons |
| 2 | 2 | Quinn Meinerz | 91.74 | 86.90 | 92.48 | 1131 | Broncos |
| 3 | 3 | Kevin Zeitler | 90.93 | 86.80 | 90.67 | 1047 | Lions |
| 4 | 4 | Landon Dickerson | 86.95 | 82.30 | 87.41 | 1157 | Eagles |
| 5 | 5 | Quenton Nelson | 86.93 | 81.30 | 88.75 | 1083 | Colts |
| 6 | 6 | Joe Thuney | 84.69 | 80.20 | 84.94 | 1232 | Chiefs |
| 7 | 7 | James Daniels | 84.55 | 75.15 | 91.58 | 209 | Steelers |
| 8 | 8 | John Simpson | 83.68 | 77.30 | 86.53 | 1020 | Jets |
| 9 | 9 | Christian Mahogany | 83.62 | 72.60 | 92.88 | 144 | Lions |
| 10 | 10 | Dominick Puni | 83.57 | 80.50 | 81.86 | 1078 | 49ers |
| 11 | 11 | Kevin Dotson | 83.27 | 77.70 | 85.02 | 1145 | Rams |
| 12 | 12 | Jordan Meredith | 83.25 | 75.88 | 87.47 | 574 | Raiders |
| 13 | 13 | Alijah Vera-Tucker | 83.13 | 77.07 | 85.55 | 916 | Jets |
| 14 | 14 | Will Fries | 82.86 | 74.03 | 89.09 | 268 | Colts |
| 15 | 15 | Trey Smith | 81.64 | 75.30 | 84.45 | 1232 | Chiefs |
| 16 | 16 | Damien Lewis | 81.56 | 75.16 | 84.45 | 942 | Panthers |
| 17 | 17 | Tyler Smith | 81.38 | 75.00 | 84.24 | 1052 | Cowboys |
| 18 | 18 | Teven Jenkins | 80.80 | 74.16 | 84.02 | 738 | Bears |
| 19 | 19 | Cody Mauch | 80.33 | 74.60 | 82.28 | 1178 | Buccaneers |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Dylan Parham | 79.63 | 73.53 | 82.10 | 882 | Raiders |
| 21 | 2 | Mekhi Becton | 77.88 | 72.50 | 79.36 | 1097 | Eagles |
| 22 | 3 | Matthew Bergeron | 77.86 | 70.90 | 81.53 | 1106 | Falcons |
| 23 | 4 | Matt Pryor | 76.59 | 69.90 | 79.87 | 1005 | Bears |
| 24 | 5 | Sam Cosmi | 75.26 | 67.80 | 79.62 | 1259 | Commanders |
| 25 | 6 | Chandler Zavala | 75.18 | 65.02 | 83.25 | 198 | Panthers |
| 26 | 7 | Robert Hunt | 74.89 | 67.66 | 78.92 | 966 | Panthers |
| 27 | 8 | Cesar Ruiz | 74.35 | 66.90 | 78.69 | 813 | Saints |

### Starter (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Isaac Seumalo | 73.88 | 66.08 | 78.70 | 872 | Steelers |
| 29 | 2 | Jake Hanson | 73.67 | 63.96 | 81.13 | 103 | Jets |
| 30 | 3 | Elgton Jenkins | 73.63 | 65.50 | 78.91 | 1073 | Packers |
| 31 | 4 | Jackson Powers-Johnson | 73.61 | 63.84 | 81.15 | 956 | Raiders |
| 32 | 5 | Will Hernandez | 73.50 | 64.96 | 79.33 | 280 | Cardinals |
| 33 | 6 | David Edwards | 73.35 | 66.10 | 77.41 | 2360 | Bills |
| 34 | 7 | Aaron Banks | 73.02 | 64.79 | 78.43 | 775 | 49ers |
| 35 | 8 | Jack Driscoll | 72.97 | 64.52 | 78.68 | 110 | Eagles |
| 36 | 9 | Dalton Risner | 72.76 | 66.38 | 75.62 | 611 | Vikings |
| 37 | 10 | Jonah Jackson | 72.62 | 63.96 | 78.62 | 267 | Rams |
| 38 | 11 | Brandon Scherff | 72.14 | 64.70 | 76.47 | 1013 | Jaguars |
| 39 | 12 | Ben Powers | 71.97 | 64.40 | 76.47 | 1130 | Broncos |
| 40 | 13 | Ezra Cleveland | 71.88 | 64.79 | 75.72 | 911 | Jaguars |
| 41 | 14 | Zack Martin | 71.75 | 64.94 | 75.19 | 638 | Cowboys |
| 42 | 15 | Zion Johnson | 71.50 | 64.40 | 75.35 | 1102 | Chargers |
| 43 | 16 | Joel Bitonio | 71.41 | 63.90 | 75.83 | 1178 | Browns |
| 44 | 17 | T.J. Bass | 71.27 | 61.70 | 78.54 | 315 | Cowboys |
| 45 | 18 | Evan Brown | 71.13 | 65.90 | 72.41 | 1070 | Cardinals |
| 46 | 19 | Greg Van Roten | 71.02 | 63.40 | 75.59 | 1121 | Giants |
| 47 | 20 | Wyatt Teller | 70.50 | 62.52 | 75.56 | 885 | Browns |
| 48 | 21 | Laken Tomlinson | 69.65 | 62.10 | 74.12 | 1094 | Seahawks |
| 49 | 22 | Sean Rhyan | 69.09 | 61.30 | 73.89 | 1027 | Packers |
| 50 | 23 | Peter Skoronski | 68.47 | 60.30 | 73.80 | 1095 | Titans |
| 51 | 24 | Cordell Volson | 68.12 | 59.30 | 74.34 | 984 | Bengals |
| 52 | 25 | Nick Allegretti | 67.94 | 59.40 | 73.78 | 1372 | Commanders |
| 53 | 26 | Shaq Mason | 67.90 | 60.50 | 72.17 | 999 | Texans |
| 54 | 27 | Trey Pipkins III | 67.03 | 57.97 | 73.58 | 838 | Chargers |
| 55 | 28 | Nick Zakelj | 66.89 | 59.47 | 71.19 | 162 | 49ers |
| 56 | 29 | Spencer Burford | 66.69 | 59.04 | 71.30 | 113 | 49ers |
| 57 | 30 | Spencer Anderson | 66.50 | 58.01 | 72.28 | 357 | Steelers |
| 58 | 31 | Mason McCormick | 66.17 | 57.76 | 71.83 | 936 | Steelers |
| 59 | 32 | Jordan Morgan | 66.00 | 59.65 | 68.81 | 186 | Packers |
| 60 | 33 | Jake Kubas | 65.92 | 56.51 | 72.97 | 197 | Giants |
| 61 | 34 | Patrick Mekari | 65.86 | 59.00 | 69.39 | 1131 | Ravens |
| 62 | 35 | Graham Glasgow | 65.70 | 57.20 | 71.48 | 1149 | Lions |
| 63 | 36 | Blake Brandel | 65.09 | 55.70 | 72.11 | 1191 | Vikings |
| 64 | 37 | Robert Jones | 65.07 | 56.10 | 71.50 | 1080 | Dolphins |
| 65 | 38 | Nick Saldiveri | 65.02 | 57.64 | 69.25 | 344 | Saints |
| 66 | 39 | Jon Runyan | 64.71 | 56.39 | 70.24 | 842 | Giants |
| 67 | 40 | Andrew Vorhees | 64.69 | 58.54 | 67.23 | 268 | Ravens |
| 68 | 41 | Ben Bredeson | 64.58 | 56.00 | 70.48 | 1173 | Buccaneers |
| 69 | 42 | Kayode Awosika | 64.13 | 56.52 | 68.68 | 145 | Lions |
| 70 | 43 | O'Cyrus Torrence | 64.13 | 55.50 | 70.09 | 1221 | Bills |
| 71 | 44 | Ed Ingram | 64.08 | 55.40 | 70.11 | 580 | Vikings |
| 72 | 45 | Mark Glowinski | 63.32 | 54.98 | 68.88 | 355 | Colts |
| 73 | 46 | Logan Bruss | 63.16 | 53.24 | 70.91 | 195 | Titans |
| 74 | 47 | Dalton Tucker | 62.92 | 55.40 | 67.36 | 464 | Colts |
| 75 | 48 | Aaron Stinnie | 62.92 | 54.64 | 68.41 | 193 | Giants |
| 76 | 49 | Isaiah Wynn | 62.90 | 55.60 | 67.02 | 103 | Dolphins |
| 77 | 50 | Michael Dunn | 62.70 | 54.38 | 68.23 | 171 | Browns |

### Rotation/backup (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 78 | 1 | Anthony Bradford | 61.69 | 51.50 | 69.81 | 578 | Seahawks |
| 79 | 2 | Mike Caliendo | 61.58 | 53.65 | 66.58 | 354 | Chiefs |
| 80 | 3 | Zak Zinter | 61.19 | 52.17 | 67.69 | 233 | Browns |
| 81 | 4 | Christian Haynes | 61.06 | 55.26 | 63.11 | 167 | Seahawks |
| 82 | 5 | Alex Cappa | 60.59 | 50.50 | 68.58 | 1132 | Bengals |
| 83 | 6 | Sidy Sow | 57.27 | 47.92 | 64.22 | 155 | Patriots |
| 84 | 7 | Layden Robinson | 57.19 | 47.18 | 65.06 | 602 | Patriots |
| 85 | 8 | Sataoa Laumea | 56.76 | 46.13 | 65.48 | 355 | Seahawks |
| 86 | 9 | Kenyon Green | 55.08 | 43.55 | 65.05 | 582 | Texans |

## HB — Running Back

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Bucky Irving | 87.79 | 84.78 | 85.71 | 246 | Buccaneers |
| 2 | 2 | Jahmyr Gibbs | 87.14 | 89.30 | 80.82 | 347 | Lions |
| 3 | 3 | Bijan Robinson | 86.80 | 92.80 | 77.35 | 389 | Falcons |
| 4 | 4 | De'Von Achane | 86.03 | 81.60 | 85.11 | 408 | Dolphins |
| 5 | 5 | Derrick Henry | 85.23 | 88.45 | 78.05 | 197 | Ravens |
| 6 | 6 | Josh Jacobs | 83.50 | 89.10 | 74.37 | 265 | Packers |
| 7 | 7 | Saquon Barkley | 82.41 | 87.00 | 74.10 | 353 | Eagles |
| 8 | 8 | James Conner | 82.13 | 87.51 | 73.18 | 269 | Cardinals |
| 9 | 9 | Kenneth Walker III | 81.78 | 84.44 | 75.05 | 224 | Seahawks |

### Good (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | James Cook | 79.77 | 81.59 | 73.74 | 258 | Bills |
| 11 | 2 | Aaron Jones | 78.30 | 74.99 | 76.47 | 347 | Vikings |
| 12 | 3 | David Montgomery | 77.31 | 80.38 | 70.25 | 158 | Lions |
| 13 | 4 | Chase Brown | 76.97 | 74.83 | 74.17 | 339 | Bengals |
| 14 | 5 | Justice Hill | 76.77 | 76.25 | 72.65 | 264 | Ravens |
| 15 | 6 | Jordan Mason | 76.62 | 68.49 | 78.73 | 170 | 49ers |
| 16 | 7 | Zach Charbonnet | 76.45 | 75.13 | 72.99 | 284 | Seahawks |
| 17 | 8 | Jaylen Warren | 76.20 | 63.74 | 81.84 | 232 | Steelers |
| 18 | 9 | Tony Pollard | 76.08 | 68.13 | 78.04 | 301 | Titans |
| 19 | 10 | Alvin Kamara | 75.73 | 72.92 | 73.49 | 311 | Saints |
| 20 | 11 | Najee Harris | 75.17 | 74.96 | 70.79 | 233 | Steelers |
| 21 | 12 | Jerome Ford | 74.74 | 70.02 | 74.05 | 304 | Browns |
| 22 | 13 | Chuba Hubbard | 74.69 | 75.33 | 69.63 | 336 | Panthers |
| 23 | 14 | Rhamondre Stevenson | 74.54 | 68.72 | 74.76 | 273 | Patriots |
| 24 | 15 | Austin Ekeler | 74.54 | 68.99 | 74.54 | 283 | Commanders |

### Starter (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 25 | 1 | Rico Dowdle | 73.98 | 72.00 | 71.05 | 283 | Cowboys |
| 26 | 2 | Kareem Hunt | 73.97 | 72.51 | 70.61 | 238 | Chiefs |
| 27 | 3 | Tyjae Spears | 73.96 | 66.22 | 75.74 | 167 | Titans |
| 28 | 4 | Breece Hall | 73.78 | 62.00 | 78.87 | 384 | Jets |
| 29 | 5 | Antonio Gibson | 73.75 | 70.01 | 72.26 | 164 | Patriots |
| 30 | 6 | Joe Mixon | 73.25 | 75.08 | 67.20 | 273 | Texans |
| 31 | 7 | Raheem Mostert | 72.94 | 67.28 | 73.03 | 155 | Dolphins |
| 32 | 8 | Ty Johnson | 72.73 | 70.37 | 70.11 | 231 | Bills |
| 33 | 9 | Emanuel Wilson | 72.66 | 72.26 | 68.44 | 109 | Packers |
| 34 | 10 | Jaleel McLaughlin | 72.64 | 63.22 | 75.81 | 146 | Broncos |
| 35 | 11 | Kyren Williams | 72.47 | 69.10 | 70.68 | 402 | Rams |
| 36 | 12 | J.K. Dobbins | 72.40 | 65.10 | 73.83 | 227 | Chargers |
| 37 | 13 | Jeremy McNichols | 72.06 | 67.18 | 71.51 | 136 | Commanders |
| 38 | 14 | Miles Sanders | 72.03 | 66.01 | 72.41 | 130 | Panthers |
| 39 | 15 | Rachaad White | 72.02 | 72.48 | 67.09 | 311 | Buccaneers |
| 40 | 16 | Tank Bigsby | 71.71 | 64.72 | 72.89 | 129 | Jaguars |
| 41 | 17 | Brian Robinson | 70.96 | 69.48 | 67.62 | 237 | Commanders |
| 42 | 18 | Ray Davis | 70.58 | 65.65 | 70.06 | 112 | Bills |
| 43 | 19 | Isaac Guerendo | 70.26 | 62.44 | 72.11 | 107 | 49ers |
| 44 | 20 | Devin Singletary | 70.11 | 61.67 | 72.47 | 164 | Giants |
| 45 | 21 | Samaje Perine | 69.62 | 65.74 | 68.25 | 217 | Chiefs |
| 46 | 22 | Javonte Williams | 69.50 | 61.59 | 71.43 | 300 | Broncos |
| 47 | 23 | Jonathan Taylor | 69.36 | 57.19 | 74.77 | 270 | Colts |
| 48 | 24 | Travis Etienne Jr. | 68.76 | 60.62 | 70.87 | 254 | Jaguars |
| 49 | 25 | Pierre Strong Jr. | 68.71 | 59.05 | 72.07 | 134 | Browns |
| 50 | 26 | Cam Akers | 68.17 | 67.03 | 64.56 | 126 | Vikings |
| 51 | 27 | Isiah Pacheco | 67.93 | 63.07 | 67.36 | 115 | Chiefs |
| 52 | 28 | Tyrone Tracy | 67.83 | 58.55 | 70.87 | 310 | Giants |
| 53 | 29 | D'Ernest Johnson | 67.75 | 60.39 | 69.23 | 117 | Jaguars |
| 54 | 30 | Braelon Allen | 67.68 | 68.08 | 62.80 | 134 | Jets |
| 55 | 31 | D'Andre Swift | 67.53 | 61.27 | 68.10 | 353 | Bears |
| 56 | 32 | Ameer Abdullah | 67.23 | 66.84 | 63.00 | 258 | Raiders |
| 57 | 33 | Alexander Mattison | 67.14 | 61.06 | 67.57 | 218 | Raiders |
| 58 | 34 | Trey Sermon | 66.34 | 56.66 | 69.71 | 122 | Colts |
| 59 | 35 | Zack Moss | 65.84 | 58.85 | 67.01 | 155 | Bengals |
| 60 | 36 | Kenneth Gainwell | 65.61 | 58.59 | 66.81 | 132 | Eagles |
| 61 | 37 | Roschon Johnson | 65.12 | 65.68 | 60.11 | 136 | Bears |
| 62 | 38 | Dare Ogunbowale | 62.46 | 60.45 | 59.56 | 213 | Texans |

### Rotation/backup (0 players)

_None._

## LB — Linebacker

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Fred Warner | 85.51 | 89.20 | 78.15 | 997 | 49ers |
| 2 | 2 | Zack Baun | 85.01 | 90.20 | 78.58 | 1150 | Eagles |
| 3 | 3 | Bobby Wagner | 83.79 | 88.30 | 75.79 | 1258 | Commanders |
| 4 | 4 | Leo Chenal | 82.26 | 81.24 | 78.59 | 497 | Chiefs |

### Good (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Jack Campbell | 79.47 | 78.70 | 75.61 | 1047 | Lions |
| 6 | 2 | Edgerrin Cooper | 78.31 | 81.03 | 73.82 | 549 | Packers |
| 7 | 3 | Devin Lloyd | 76.60 | 76.70 | 73.22 | 884 | Jaguars |
| 8 | 4 | Elandon Roberts | 74.95 | 77.34 | 68.61 | 525 | Steelers |

### Starter (64 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Devin Bush | 73.89 | 74.95 | 71.64 | 497 | Browns |
| 10 | 2 | Jordan Hicks | 73.55 | 75.91 | 71.13 | 602 | Browns |
| 11 | 3 | Demario Davis | 73.51 | 73.20 | 69.81 | 1090 | Saints |
| 12 | 4 | Eric Kendricks | 73.44 | 75.20 | 69.28 | 918 | Cowboys |
| 13 | 5 | Payton Wilson | 73.23 | 71.71 | 69.96 | 520 | Steelers |
| 14 | 6 | Bobby Okereke | 72.64 | 74.42 | 69.40 | 734 | Giants |
| 15 | 7 | Nakobe Dean | 72.57 | 76.60 | 70.92 | 880 | Eagles |
| 16 | 8 | Oren Burks | 72.31 | 73.72 | 71.36 | 322 | Eagles |
| 17 | 9 | Jordyn Brooks | 71.90 | 71.30 | 68.42 | 1039 | Dolphins |
| 18 | 10 | Jeremiah Owusu-Koramoah | 71.87 | 77.50 | 68.96 | 460 | Browns |
| 19 | 11 | Logan Wilson | 71.79 | 71.94 | 70.78 | 743 | Bengals |
| 20 | 12 | Blake Cashman | 71.78 | 72.00 | 69.45 | 947 | Vikings |
| 21 | 13 | Robert Spillane | 71.56 | 68.40 | 69.79 | 1093 | Raiders |
| 22 | 14 | Malcolm Rodriguez | 71.25 | 68.97 | 73.94 | 318 | Lions |
| 23 | 15 | Jack Gibbens | 71.10 | 72.39 | 73.82 | 234 | Titans |
| 24 | 16 | Roquan Smith | 70.71 | 66.80 | 69.31 | 1099 | Ravens |
| 25 | 17 | Kaden Elliss | 70.51 | 71.10 | 65.59 | 1097 | Falcons |
| 26 | 18 | Christian Elliss | 70.39 | 69.98 | 68.36 | 514 | Patriots |
| 27 | 19 | Quincy Williams | 70.13 | 68.00 | 67.77 | 1136 | Jets |
| 28 | 20 | Lavonte David | 69.54 | 67.90 | 66.36 | 1149 | Buccaneers |
| 29 | 21 | Pete Werner | 69.39 | 68.70 | 68.94 | 731 | Saints |
| 30 | 22 | Foyesade Oluokun | 69.35 | 68.48 | 67.67 | 815 | Jaguars |
| 31 | 23 | Daiyan Henley | 69.25 | 69.90 | 68.62 | 1071 | Chargers |
| 32 | 24 | Azeez Al-Shaair | 69.14 | 68.06 | 68.66 | 672 | Texans |
| 33 | 25 | Omar Speights | 68.66 | 67.37 | 69.40 | 504 | Rams |
| 34 | 26 | Tyrel Dodson | 68.42 | 67.30 | 68.62 | 854 | Dolphins |
| 35 | 27 | Drue Tranquill | 68.05 | 66.00 | 65.20 | 902 | Chiefs |
| 36 | 28 | Cody Barton | 67.88 | 63.70 | 67.96 | 1129 | Broncos |
| 37 | 29 | C.J. Mosley | 67.85 | 69.41 | 68.99 | 110 | Jets |
| 38 | 30 | Alex Anzalone | 67.66 | 66.29 | 67.42 | 681 | Lions |
| 39 | 31 | Jeremiah Trotter Jr. | 66.69 | 66.16 | 72.09 | 109 | Eagles |
| 40 | 32 | Chazz Surratt | 66.65 | 65.19 | 70.36 | 137 | Jets |
| 41 | 33 | Jake Hansen | 66.61 | 67.33 | 70.94 | 136 | Texans |
| 42 | 34 | Dee Winters | 66.15 | 64.46 | 67.09 | 398 | 49ers |
| 43 | 35 | Derrick Barnes | 65.96 | 64.72 | 70.67 | 120 | Lions |
| 44 | 36 | T.J. Edwards | 65.77 | 61.40 | 64.74 | 1054 | Bears |
| 45 | 37 | Nate Landman | 65.67 | 65.06 | 66.82 | 543 | Falcons |
| 46 | 38 | Nick Bolton | 65.56 | 62.50 | 65.07 | 1076 | Chiefs |
| 47 | 39 | Frankie Luvu | 65.55 | 64.20 | 62.57 | 1239 | Commanders |
| 48 | 40 | Mack Wilson Sr. | 65.45 | 63.66 | 64.28 | 760 | Cardinals |
| 49 | 41 | Ernest Jones | 65.12 | 60.70 | 64.97 | 995 | Seahawks |
| 50 | 42 | Tyrice Knight | 65.04 | 64.42 | 66.32 | 550 | Seahawks |
| 51 | 43 | Neville Hewitt | 65.01 | 67.79 | 67.91 | 351 | Texans |
| 52 | 44 | Grant Stuard | 64.97 | 64.97 | 68.49 | 229 | Colts |
| 53 | 45 | Henry To'oTo'o | 64.96 | 62.20 | 63.84 | 936 | Texans |
| 54 | 46 | Krys Barnes | 64.89 | 61.55 | 67.57 | 205 | Cardinals |
| 55 | 47 | Ivan Pace Jr. | 64.89 | 62.54 | 65.55 | 454 | Vikings |
| 56 | 48 | Damone Clark | 64.71 | 63.66 | 66.21 | 163 | Cowboys |
| 57 | 49 | Joe Andreessen | 64.69 | 64.84 | 69.55 | 116 | Bills |
| 58 | 50 | Zaire Franklin | 64.55 | 60.30 | 63.75 | 1157 | Colts |
| 59 | 51 | J.J. Russell | 64.48 | 65.29 | 69.67 | 271 | Buccaneers |
| 60 | 52 | Troy Dye | 64.29 | 63.68 | 66.50 | 355 | Chargers |
| 61 | 53 | Jack Sanborn | 64.10 | 61.66 | 63.44 | 235 | Bears |
| 62 | 54 | Germaine Pratt | 63.93 | 60.20 | 62.82 | 1075 | Bengals |
| 63 | 55 | Dorian Williams | 63.79 | 58.63 | 64.55 | 680 | Bills |
| 64 | 56 | Micah McFadden | 63.72 | 62.53 | 62.49 | 668 | Giants |
| 65 | 57 | Tremaine Edmunds | 63.53 | 59.30 | 63.86 | 1055 | Bears |
| 66 | 58 | Eric Wilson | 63.18 | 63.55 | 63.26 | 559 | Packers |
| 67 | 59 | Sione Takitaki | 62.72 | 61.46 | 63.76 | 194 | Patriots |
| 68 | 60 | Jerome Baker | 62.59 | 60.90 | 64.39 | 566 | Titans |
| 69 | 61 | DeMarvion Overshown | 62.35 | 61.49 | 61.19 | 708 | Cowboys |
| 70 | 62 | Quay Walker | 62.30 | 57.43 | 63.55 | 804 | Packers |
| 71 | 63 | Chris Board | 62.23 | 62.80 | 65.08 | 213 | Ravens |
| 72 | 64 | Shaq Thompson | 62.17 | 65.39 | 66.73 | 245 | Panthers |

### Rotation/backup (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 73 | 1 | Yasir Abdullah | 61.95 | 58.45 | 63.52 | 170 | Jaguars |
| 74 | 2 | Akeem Davis-Gaither | 61.92 | 59.19 | 61.70 | 535 | Bengals |
| 75 | 3 | Denzel Perryman | 61.87 | 61.81 | 62.08 | 343 | Chargers |
| 76 | 4 | E.J. Speed | 61.82 | 56.70 | 62.74 | 1011 | Colts |
| 77 | 5 | Alex Singleton | 61.47 | 62.00 | 64.35 | 190 | Broncos |
| 78 | 6 | Trenton Simpson | 61.46 | 58.84 | 64.04 | 654 | Ravens |
| 79 | 7 | Chad Muma | 61.22 | 58.54 | 63.48 | 260 | Jaguars |
| 80 | 8 | Jahlani Tavai | 61.21 | 55.50 | 61.23 | 916 | Patriots |
| 81 | 9 | Owen Pappoe | 61.21 | 61.20 | 64.36 | 131 | Cardinals |
| 82 | 10 | Patrick Queen | 60.92 | 56.80 | 59.69 | 1164 | Steelers |
| 83 | 11 | Darius Muasau | 60.51 | 57.67 | 62.48 | 435 | Giants |
| 84 | 12 | SirVocea Dennis | 60.23 | 62.92 | 64.93 | 105 | Buccaneers |
| 85 | 13 | Devin White | 60.21 | 58.61 | 62.58 | 176 | Texans |
| 86 | 14 | Isaiah McDuffie | 60.09 | 55.67 | 61.21 | 728 | Packers |
| 87 | 15 | De'Vondre Campbell | 60.09 | 58.36 | 61.19 | 719 | 49ers |
| 88 | 16 | Josey Jewell | 59.62 | 56.55 | 61.36 | 796 | Panthers |
| 89 | 17 | Malik Harrison | 59.57 | 53.86 | 60.33 | 438 | Ravens |
| 90 | 18 | Divine Deablo | 59.44 | 57.53 | 60.56 | 689 | Raiders |
| 91 | 19 | Winston Reid | 59.43 | 56.56 | 59.33 | 144 | Browns |
| 92 | 20 | Jalen Reeves-Maybin | 59.34 | 58.88 | 61.75 | 165 | Lions |
| 93 | 21 | Ventrell Miller | 59.31 | 53.87 | 60.45 | 482 | Jaguars |
| 94 | 22 | Ben Niemann | 59.08 | 57.76 | 63.53 | 178 | Lions |
| 95 | 23 | Anfernee Orji | 58.90 | 57.97 | 60.41 | 147 | Saints |
| 96 | 24 | Troy Andersen | 58.86 | 60.24 | 63.51 | 287 | Falcons |
| 97 | 25 | Ty Summers | 58.84 | 61.68 | 66.91 | 113 | Giants |
| 98 | 26 | Christian Rozeboom | 58.61 | 53.50 | 60.90 | 956 | Rams |
| 99 | 27 | Luke Gifford | 58.58 | 60.10 | 65.53 | 203 | Titans |
| 100 | 28 | Kyzir White | 58.49 | 48.80 | 63.54 | 1015 | Cardinals |
| 101 | 29 | Trevin Wallace | 58.39 | 56.63 | 61.62 | 582 | Panthers |
| 102 | 30 | Marist Liufau | 57.83 | 52.12 | 57.86 | 520 | Cowboys |
| 103 | 31 | Troy Reeder | 57.70 | 58.59 | 62.73 | 372 | Rams |
| 104 | 32 | Isaiah Simmons | 57.57 | 53.23 | 59.14 | 181 | Giants |
| 105 | 33 | Claudin Cherelus | 57.44 | 61.45 | 62.52 | 158 | Panthers |
| 106 | 34 | Amari Burney | 57.10 | 58.88 | 60.29 | 101 | Raiders |
| 107 | 35 | Ezekiel Turner | 56.76 | 55.88 | 64.44 | 111 | Lions |
| 108 | 36 | JD Bertrand | 56.58 | 55.54 | 62.38 | 157 | Falcons |
| 109 | 37 | Luke Masterson | 56.22 | 54.68 | 62.00 | 102 | Raiders |
| 110 | 38 | Nick Vigil | 56.15 | 55.32 | 61.37 | 127 | Cowboys |
| 111 | 39 | Terrel Bernard | 55.98 | 48.20 | 60.46 | 917 | Bills |
| 112 | 40 | Mohamoud Diabate | 55.89 | 53.69 | 60.11 | 581 | Browns |
| 113 | 41 | Anthony Walker Jr. | 55.31 | 50.48 | 62.30 | 516 | Dolphins |
| 114 | 42 | Willie Gay | 55.20 | 50.64 | 57.27 | 277 | Saints |
| 115 | 43 | Kenneth Murray Jr. | 54.93 | 45.94 | 59.74 | 815 | Titans |
| 116 | 44 | Raekwon McMillan | 54.53 | 49.04 | 59.52 | 267 | Titans |
| 117 | 45 | Matt Milano | 54.13 | 54.76 | 58.62 | 333 | Bills |
| 118 | 46 | Junior Colson | 53.02 | 47.55 | 59.15 | 234 | Chargers |
| 119 | 47 | K.J. Britt | 53.00 | 47.27 | 58.83 | 632 | Buccaneers |
| 120 | 48 | Kamu Grugier-Hill | 52.70 | 48.88 | 55.97 | 182 | Vikings |
| 121 | 49 | Christian Harris | 52.29 | 50.21 | 56.82 | 180 | Texans |
| 122 | 50 | Justin Strnad | 51.81 | 50.43 | 57.03 | 736 | Broncos |
| 123 | 51 | Jacoby Windmon | 50.82 | 56.20 | 59.56 | 128 | Panthers |
| 124 | 52 | Demetrius Flannigan-Fowles | 48.46 | 47.38 | 54.40 | 151 | 49ers |
| 125 | 53 | Baylon Spector | 45.08 | 42.19 | 54.77 | 291 | Bills |
| 126 | 54 | Chandler Wooten | 45.00 | 44.34 | 54.02 | 212 | Panthers |

## QB — Quarterback

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Lamar Jackson | 85.55 | 89.15 | 80.59 | 635 | Ravens |
| 2 | 2 | Joe Burrow | 83.48 | 89.92 | 76.58 | 775 | Bengals |
| 3 | 3 | Josh Allen | 80.96 | 85.33 | 74.08 | 686 | Bills |
| 4 | 4 | Jared Goff | 80.83 | 78.41 | 78.60 | 648 | Lions |
| 5 | 5 | Justin Herbert | 80.04 | 87.09 | 72.46 | 662 | Chargers |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Baker Mayfield | 79.96 | 79.34 | 76.87 | 721 | Buccaneers |
| 7 | 2 | Geno Smith | 78.73 | 81.34 | 73.43 | 704 | Seahawks |
| 8 | 3 | Patrick Mahomes | 78.61 | 82.91 | 71.58 | 776 | Chiefs |
| 9 | 4 | Brock Purdy | 78.36 | 77.54 | 77.92 | 567 | 49ers |
| 10 | 5 | Jayden Daniels | 77.27 | 84.70 | 74.56 | 781 | Commanders |
| 11 | 6 | Tua Tagovailoa | 76.37 | 76.11 | 76.46 | 460 | Dolphins |
| 12 | 7 | C.J. Stroud | 76.18 | 78.24 | 70.84 | 742 | Texans |
| 13 | 8 | Jalen Hurts | 76.09 | 74.54 | 75.22 | 558 | Eagles |
| 14 | 9 | Matthew Stafford | 75.41 | 76.21 | 72.07 | 667 | Rams |
| 15 | 10 | Sam Darnold | 75.12 | 76.79 | 75.22 | 725 | Vikings |
| 16 | 11 | Jordan Love | 74.14 | 76.76 | 72.38 | 528 | Packers |

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Derek Carr | 73.24 | 76.93 | 73.05 | 319 | Saints |
| 18 | 2 | Kyler Murray | 73.24 | 75.31 | 70.96 | 656 | Cardinals |
| 19 | 3 | Kirk Cousins | 73.06 | 74.50 | 72.67 | 521 | Falcons |
| 20 | 4 | Russell Wilson | 72.20 | 73.38 | 71.88 | 444 | Steelers |
| 21 | 5 | Dak Prescott | 71.80 | 73.35 | 72.66 | 344 | Cowboys |
| 22 | 6 | Aaron Rodgers | 70.98 | 76.21 | 67.44 | 684 | Jets |
| 23 | 7 | Bo Nix | 70.09 | 73.80 | 69.08 | 712 | Broncos |
| 24 | 8 | Trevor Lawrence | 69.56 | 73.19 | 68.53 | 332 | Jaguars |
| 25 | 9 | Caleb Williams | 65.11 | 62.90 | 65.14 | 741 | Bears |
| 26 | 10 | Joe Flacco | 64.08 | 67.59 | 68.99 | 290 | Colts |
| 27 | 11 | Bryce Young | 63.14 | 66.61 | 60.48 | 477 | Panthers |
| 28 | 12 | Justin Fields | 62.54 | 62.70 | 67.53 | 215 | Steelers |
| 29 | 13 | Drake Maye | 62.53 | 64.25 | 67.77 | 461 | Patriots |
| 30 | 14 | Andy Dalton | 62.10 | 68.66 | 67.14 | 185 | Panthers |

### Rotation/backup (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Daniel Jones | 61.97 | 65.56 | 62.73 | 418 | Vikings |
| 32 | 2 | Gardner Minshew | 61.49 | 60.92 | 64.84 | 370 | Raiders |
| 33 | 3 | Michael Penix Jr. | 61.44 | 72.45 | 65.36 | 120 | Falcons |
| 34 | 4 | Jameis Winston | 61.26 | 67.75 | 64.90 | 347 | Browns |
| 35 | 5 | Aidan O'Connell | 60.18 | 60.34 | 65.27 | 276 | Raiders |
| 36 | 6 | Mason Rudolph | 59.90 | 61.35 | 66.03 | 276 | Titans |
| 37 | 7 | Will Levis | 59.85 | 57.00 | 65.33 | 384 | Titans |
| 38 | 8 | Mac Jones | 59.68 | 60.22 | 63.23 | 309 | Jaguars |
| 39 | 9 | Tyler Huntley | 58.77 | 60.82 | 63.22 | 182 | Dolphins |
| 40 | 10 | Cooper Rush | 58.43 | 59.51 | 62.20 | 352 | Cowboys |
| 41 | 11 | Deshaun Watson | 58.40 | 62.52 | 60.90 | 290 | Browns |
| 42 | 12 | Anthony Richardson | 58.30 | 59.58 | 60.23 | 317 | Colts |
| 43 | 13 | Desmond Ridder | 58.02 | 56.68 | 65.43 | 105 | Raiders |
| 44 | 14 | Jacoby Brissett | 57.90 | 62.51 | 61.34 | 200 | Patriots |
| 45 | 15 | Drew Lock | 56.54 | 54.03 | 59.51 | 216 | Giants |
| 46 | 16 | Spencer Rattler | 56.44 | 52.72 | 57.35 | 284 | Saints |
| 47 | 17 | Dorian Thompson-Robinson | 54.78 | 51.21 | 54.87 | 142 | Browns |

## S — Safety

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Kerby Joseph | 93.53 | 91.10 | 91.64 | 1158 | Lions |
| 2 | 2 | Xavier McKinney | 90.97 | 90.20 | 88.69 | 1125 | Packers |
| 3 | 3 | Brandon Jones | 88.01 | 89.80 | 84.83 | 1042 | Broncos |
| 4 | 4 | Kyle Hamilton | 86.90 | 88.40 | 80.93 | 1150 | Ravens |
| 5 | 5 | Ar'Darius Washington | 85.80 | 86.10 | 80.72 | 830 | Ravens |
| 6 | 6 | Jessie Bates III | 82.71 | 81.40 | 79.34 | 1095 | Falcons |
| 7 | 7 | Julian Love | 81.03 | 76.10 | 81.00 | 1079 | Seahawks |
| 8 | 8 | Brian Branch | 80.61 | 77.80 | 78.40 | 982 | Lions |
| 9 | 9 | Justin Reid | 80.51 | 77.00 | 78.95 | 1112 | Chiefs |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | C.J. Gardner-Johnson | 79.73 | 83.10 | 76.67 | 1118 | Eagles |
| 11 | 2 | Derwin James Jr. | 79.67 | 76.30 | 79.00 | 1059 | Chargers |
| 12 | 3 | Jabrill Peppers | 78.54 | 77.42 | 81.67 | 372 | Patriots |
| 13 | 4 | Jaden Hicks | 77.62 | 69.67 | 80.16 | 430 | Chiefs |
| 14 | 5 | Budda Baker | 76.11 | 74.70 | 74.76 | 1064 | Cardinals |
| 15 | 6 | Kamren Kinchens | 75.52 | 71.91 | 74.04 | 623 | Rams |
| 16 | 7 | Andrew Wingard | 75.14 | 72.44 | 80.86 | 216 | Jaguars |
| 17 | 8 | Julian Blackmon | 74.45 | 73.50 | 72.45 | 1084 | Colts |
| 18 | 9 | Jimmie Ward | 74.37 | 72.98 | 78.30 | 461 | Texans |
| 19 | 10 | Dell Pettus | 74.26 | 67.78 | 77.70 | 341 | Patriots |

### Starter (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Quandre Diggs | 73.85 | 68.77 | 78.82 | 419 | Titans |
| 21 | 2 | Evan Williams | 72.55 | 71.09 | 73.60 | 533 | Packers |
| 22 | 3 | Thomas Harper | 72.26 | 67.89 | 76.00 | 191 | Raiders |
| 23 | 4 | Jalen Pitre | 72.25 | 70.11 | 72.24 | 660 | Texans |
| 24 | 5 | Reed Blankenship | 72.19 | 71.40 | 70.83 | 1030 | Eagles |
| 25 | 6 | Ronnie Hickman Jr. | 71.98 | 68.47 | 75.08 | 463 | Browns |
| 26 | 7 | Jordan Howden | 71.43 | 62.86 | 75.38 | 550 | Saints |
| 27 | 8 | DeShon Elliott | 71.09 | 63.30 | 75.05 | 895 | Steelers |
| 28 | 9 | Harrison Smith | 70.78 | 65.30 | 71.71 | 1062 | Vikings |
| 29 | 10 | Minkah Fitzpatrick | 70.42 | 64.90 | 73.19 | 1158 | Steelers |
| 30 | 11 | Ji'Ayir Brown | 70.42 | 69.50 | 68.58 | 886 | 49ers |
| 31 | 12 | Kevin Byard | 70.40 | 60.50 | 74.74 | 1055 | Bears |
| 32 | 13 | Jalen Thompson | 70.33 | 64.20 | 72.99 | 941 | Cardinals |
| 33 | 14 | Dadrion Taylor-Demerson | 70.18 | 65.59 | 71.87 | 258 | Cardinals |
| 34 | 15 | Tony Jefferson | 69.59 | 67.15 | 78.11 | 261 | Chargers |
| 35 | 16 | Mike Brown | 69.20 | 65.19 | 73.97 | 384 | Titans |
| 36 | 17 | Ashtyn Davis | 68.79 | 64.38 | 72.81 | 260 | Jets |
| 37 | 18 | Tony Adams | 68.53 | 67.22 | 69.57 | 764 | Jets |
| 38 | 19 | Jeremy Chinn | 68.48 | 64.50 | 70.40 | 1207 | Commanders |
| 39 | 20 | Mike Edwards | 68.36 | 62.41 | 75.04 | 251 | Buccaneers |
| 40 | 21 | Marcus Maye | 68.26 | 67.55 | 71.92 | 405 | Chargers |
| 41 | 22 | Amani Hooker | 68.03 | 66.30 | 69.69 | 848 | Titans |
| 42 | 23 | Nick Cross | 68.00 | 64.30 | 70.10 | 1156 | Colts |
| 43 | 24 | Kamren Curl | 67.71 | 63.70 | 68.07 | 1112 | Rams |
| 44 | 25 | Jaylen McCollough | 67.64 | 61.09 | 68.88 | 382 | Rams |
| 45 | 26 | Zayne Anderson | 67.46 | 66.24 | 78.21 | 122 | Packers |
| 46 | 27 | Jordan Poyer | 67.28 | 61.40 | 69.59 | 964 | Dolphins |
| 47 | 28 | Alohi Gilman | 66.96 | 64.34 | 68.41 | 731 | Chargers |
| 48 | 29 | Grant Delpit | 66.57 | 60.90 | 68.91 | 976 | Browns |
| 49 | 30 | Dane Belton | 66.36 | 60.97 | 68.57 | 460 | Giants |
| 50 | 31 | Malik Mustapha | 65.62 | 60.10 | 68.18 | 755 | 49ers |
| 51 | 32 | Camryn Bynum | 65.52 | 58.60 | 67.10 | 1056 | Vikings |
| 52 | 33 | Malik Hooker | 65.43 | 57.80 | 67.89 | 1062 | Cowboys |
| 53 | 34 | Vonn Bell | 64.97 | 62.00 | 64.48 | 705 | Bengals |
| 54 | 35 | Eric Murray | 64.77 | 64.90 | 64.70 | 961 | Texans |
| 55 | 36 | Damontae Kazee | 64.73 | 57.23 | 69.91 | 313 | Steelers |
| 56 | 37 | Juan Thornhill | 64.54 | 63.68 | 65.84 | 401 | Browns |
| 57 | 38 | Tre'von Moehrig | 64.52 | 54.40 | 69.51 | 1099 | Raiders |
| 58 | 39 | Will Harris | 64.31 | 61.60 | 64.27 | 860 | Saints |
| 59 | 40 | Jaquan Brisker | 63.81 | 62.09 | 68.52 | 293 | Bears |
| 60 | 41 | Justin Simmons | 63.59 | 60.80 | 63.75 | 1017 | Falcons |
| 61 | 42 | Devon Key | 63.58 | 58.06 | 65.56 | 253 | Broncos |
| 62 | 43 | Quentin Lake | 63.24 | 58.40 | 65.85 | 1207 | Rams |
| 63 | 44 | Kaevon Merriweather | 63.18 | 61.67 | 66.69 | 274 | Buccaneers |
| 64 | 45 | Tyler Nubin | 62.98 | 58.14 | 67.15 | 789 | Giants |
| 65 | 46 | George Odum | 62.95 | 62.75 | 69.75 | 139 | 49ers |
| 66 | 47 | Tyrann Mathieu | 62.50 | 57.80 | 62.04 | 1015 | Saints |
| 67 | 48 | Xavier Woods | 62.41 | 55.40 | 65.53 | 1216 | Panthers |
| 68 | 49 | Donovan Wilson | 62.23 | 56.60 | 62.96 | 1008 | Cowboys |
| 69 | 50 | Josh Metellus | 62.20 | 52.20 | 67.52 | 1030 | Vikings |

### Rotation/backup (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 70 | 1 | Andre Cisco | 61.89 | 56.70 | 63.57 | 979 | Jaguars |
| 71 | 2 | Jaylinn Hawkins | 61.70 | 56.81 | 64.24 | 613 | Patriots |
| 72 | 3 | Jevon Holland | 61.19 | 57.10 | 62.98 | 854 | Dolphins |
| 73 | 4 | Rayshawn Jenkins | 60.09 | 58.37 | 59.14 | 550 | Seahawks |
| 74 | 5 | Jordan Whitehead | 59.96 | 53.73 | 63.73 | 731 | Buccaneers |
| 75 | 6 | Jonathan Owens | 59.53 | 55.60 | 62.32 | 429 | Bears |
| 76 | 7 | Jordan Battle | 59.43 | 54.05 | 61.01 | 464 | Bengals |
| 77 | 8 | Geno Stone | 58.76 | 53.70 | 59.29 | 1100 | Bengals |
| 78 | 9 | Christian Izien | 58.70 | 55.68 | 58.80 | 697 | Buccaneers |
| 79 | 10 | Antoine Winfield Jr. | 58.25 | 51.61 | 64.44 | 601 | Buccaneers |
| 80 | 11 | Talanoa Hufanga | 57.95 | 55.36 | 63.57 | 308 | 49ers |
| 81 | 12 | Demani Richardson | 57.89 | 59.02 | 65.08 | 403 | Panthers |
| 82 | 13 | Bryan Cook | 57.74 | 48.10 | 63.76 | 1056 | Chiefs |
| 83 | 14 | Nick Scott | 57.29 | 57.93 | 58.33 | 324 | Panthers |
| 84 | 15 | Taylor Rapp | 56.95 | 43.20 | 66.65 | 840 | Bills |
| 85 | 16 | Javon Bullard | 56.09 | 49.06 | 58.91 | 816 | Packers |
| 86 | 17 | Eddie Jackson | 56.03 | 51.96 | 60.63 | 390 | Chargers |
| 87 | 18 | P.J. Locke | 55.72 | 50.70 | 58.71 | 1076 | Broncos |
| 88 | 19 | Rodney McLeod | 55.23 | 48.52 | 59.01 | 565 | Browns |
| 89 | 20 | Antonio Johnson | 54.45 | 44.51 | 58.82 | 685 | Jaguars |
| 90 | 21 | Cole Bishop | 54.31 | 51.53 | 55.45 | 464 | Bills |
| 91 | 22 | Richie Grant | 54.30 | 52.17 | 54.84 | 165 | Falcons |
| 92 | 23 | Jordan Fuller | 53.85 | 49.16 | 61.07 | 574 | Panthers |
| 93 | 24 | Percy Butler | 53.73 | 46.51 | 57.62 | 448 | Commanders |
| 94 | 25 | Chuck Clark | 53.58 | 45.75 | 58.83 | 709 | Jets |
| 95 | 26 | Jason Pinnock | 53.55 | 45.50 | 57.31 | 976 | Giants |
| 96 | 27 | K'Von Wallace | 52.92 | 52.72 | 54.41 | 127 | Seahawks |
| 97 | 28 | Marcus Epps | 52.38 | 52.61 | 55.29 | 176 | Raiders |
| 98 | 29 | Isaiah Pola-Mao | 52.35 | 45.10 | 58.64 | 952 | Raiders |
| 99 | 30 | Damar Hamlin | 52.22 | 41.50 | 62.17 | 1042 | Bills |
| 100 | 31 | Calen Bullock | 52.13 | 40.00 | 58.51 | 1083 | Texans |
| 101 | 32 | Marcus Williams | 51.59 | 40.00 | 63.80 | 601 | Ravens |
| 102 | 33 | Darnell Savage | 50.82 | 41.14 | 59.14 | 764 | Jaguars |
| 103 | 34 | Kyle Dugger | 48.95 | 40.00 | 55.12 | 759 | Patriots |

## T — Tackle

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jordan Mailata | 98.23 | 95.20 | 96.47 | 995 | Eagles |
| 2 | 2 | Penei Sewell | 93.60 | 89.60 | 93.17 | 1213 | Lions |
| 3 | 3 | Rashawn Slater | 93.55 | 90.65 | 91.61 | 959 | Chargers |
| 4 | 4 | Lane Johnson | 92.68 | 87.50 | 93.87 | 1123 | Eagles |
| 5 | 5 | Terron Armstead | 92.04 | 87.86 | 91.85 | 821 | Dolphins |
| 6 | 6 | Zach Tom | 90.88 | 85.80 | 91.95 | 1134 | Packers |
| 7 | 7 | Bernhard Raimann | 88.84 | 84.08 | 89.46 | 856 | Colts |
| 8 | 8 | Trent Williams | 88.45 | 82.71 | 90.43 | 649 | 49ers |
| 9 | 9 | Tristan Wirfs | 87.02 | 82.50 | 87.32 | 1061 | Buccaneers |
| 10 | 10 | Charles Cross | 86.94 | 82.50 | 87.11 | 1094 | Seahawks |
| 11 | 11 | Garett Bolles | 86.32 | 80.20 | 88.81 | 1111 | Broncos |
| 12 | 12 | Darnell Wright | 85.71 | 79.30 | 88.62 | 1021 | Bears |
| 13 | 13 | Spencer Brown | 85.71 | 77.90 | 90.54 | 1140 | Bills |
| 14 | 14 | Laremy Tunsil | 85.32 | 78.10 | 89.33 | 1167 | Texans |
| 15 | 15 | Kolton Miller | 85.18 | 80.60 | 85.55 | 1075 | Raiders |
| 16 | 16 | Brian O'Neill | 85.08 | 79.30 | 87.11 | 1151 | Vikings |
| 17 | 17 | Jake Matthews | 84.92 | 79.80 | 86.03 | 1119 | Falcons |
| 18 | 18 | Paris Johnson Jr. | 84.81 | 79.49 | 86.20 | 865 | Cardinals |
| 19 | 19 | Christian Darrisaw | 84.55 | 76.66 | 89.49 | 392 | Vikings |
| 20 | 20 | Alaric Jackson | 84.55 | 78.40 | 87.09 | 1017 | Rams |
| 21 | 21 | Taylor Moton | 82.82 | 76.44 | 85.69 | 846 | Panthers |
| 22 | 22 | Taylor Decker | 82.46 | 77.08 | 83.94 | 963 | Lions |
| 23 | 23 | Luke Goedeke | 82.28 | 74.06 | 87.67 | 952 | Buccaneers |
| 24 | 24 | Joe Alt | 82.23 | 75.90 | 85.02 | 1066 | Chargers |
| 25 | 25 | Rob Havenstein | 82.19 | 74.89 | 86.32 | 805 | Rams |
| 26 | 26 | Braxton Jones | 82.03 | 75.88 | 84.58 | 719 | Bears |
| 27 | 27 | Kaleb McGary | 80.86 | 73.90 | 84.52 | 1042 | Falcons |
| 28 | 28 | Dion Dawkins | 80.02 | 72.40 | 84.59 | 1164 | Bills |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 29 | 1 | Mike McGlinchey | 79.96 | 72.23 | 84.68 | 891 | Broncos |
| 30 | 2 | Ikem Ekwonu | 79.73 | 71.24 | 85.50 | 909 | Panthers |
| 31 | 3 | Andrew Thomas | 79.65 | 72.16 | 84.03 | 416 | Giants |
| 32 | 4 | Tyron Smith | 79.40 | 71.85 | 83.88 | 592 | Jets |
| 33 | 5 | Colton McKivitz | 79.32 | 72.20 | 83.20 | 1062 | 49ers |
| 34 | 6 | Cornelius Lucas | 79.31 | 71.45 | 84.21 | 464 | Commanders |
| 35 | 7 | Walker Little | 78.44 | 69.19 | 85.26 | 508 | Jaguars |
| 36 | 8 | Ronnie Stanley | 77.86 | 70.70 | 81.80 | 1221 | Ravens |
| 37 | 9 | Tytus Howard | 77.09 | 70.20 | 80.66 | 1157 | Texans |
| 38 | 10 | Jaylon Moore | 76.81 | 67.82 | 83.27 | 271 | 49ers |
| 39 | 11 | Kendall Lamm | 76.67 | 69.16 | 81.09 | 512 | Dolphins |
| 40 | 12 | Jonah Williams | 76.63 | 68.07 | 82.51 | 343 | Cardinals |
| 41 | 13 | Terence Steele | 76.56 | 67.00 | 83.82 | 1168 | Cowboys |
| 42 | 14 | Braden Smith | 76.04 | 65.68 | 84.40 | 731 | Colts |
| 43 | 15 | Rasheed Walker | 75.30 | 68.60 | 78.60 | 1139 | Packers |
| 44 | 16 | Justin Skule | 74.80 | 65.58 | 81.58 | 362 | Buccaneers |
| 45 | 17 | Taliese Fuaga | 74.72 | 65.70 | 81.23 | 1070 | Saints |
| 46 | 18 | DJ Glaze | 74.58 | 66.10 | 80.34 | 998 | Raiders |
| 47 | 19 | Dan Moore Jr. | 74.43 | 67.20 | 78.46 | 1128 | Steelers |
| 48 | 20 | John Ojukwu | 74.36 | 64.87 | 81.51 | 264 | Titans |

### Starter (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 49 | 1 | Roger Rosengarten | 73.98 | 66.00 | 79.05 | 1066 | Ravens |
| 50 | 2 | Jack Conklin | 73.56 | 65.87 | 78.23 | 818 | Browns |
| 51 | 3 | Trent Brown | 73.47 | 63.33 | 81.51 | 139 | Bengals |
| 52 | 4 | Matt Goncalves | 73.41 | 64.47 | 79.81 | 566 | Colts |
| 53 | 5 | Morgan Moses | 73.03 | 63.02 | 80.89 | 723 | Jets |
| 54 | 6 | Evan Neal | 72.88 | 60.82 | 83.58 | 459 | Giants |
| 55 | 7 | Kelvin Beachum | 72.77 | 63.78 | 79.23 | 742 | Cardinals |
| 56 | 8 | Matt Peart | 72.55 | 63.25 | 79.45 | 190 | Broncos |
| 57 | 9 | Anton Harrison | 72.41 | 64.11 | 77.91 | 943 | Jaguars |
| 58 | 10 | Alex Palczewski | 72.01 | 61.45 | 80.64 | 179 | Broncos |
| 59 | 11 | Cam Robinson | 71.70 | 63.20 | 77.48 | 1073 | Vikings |
| 60 | 12 | Storm Norton | 71.30 | 60.76 | 79.90 | 128 | Falcons |
| 61 | 13 | Abraham Lucas | 71.10 | 61.22 | 78.80 | 406 | Seahawks |
| 62 | 14 | Joshua Ezeudu | 70.84 | 61.16 | 78.25 | 182 | Giants |
| 63 | 15 | Andrew Wylie | 70.83 | 61.70 | 77.48 | 1115 | Commanders |
| 64 | 16 | JC Latham | 70.81 | 61.80 | 77.31 | 1095 | Titans |
| 65 | 17 | Austin Jackson | 70.74 | 60.00 | 79.63 | 542 | Dolphins |
| 66 | 18 | Olumuyiwa Fashanu | 70.65 | 60.88 | 78.18 | 534 | Jets |
| 67 | 19 | Cole Van Lanen | 70.61 | 61.16 | 77.70 | 252 | Jaguars |
| 68 | 20 | Trevor Penning | 70.60 | 60.20 | 79.00 | 1081 | Saints |
| 69 | 21 | Warren McClendon Jr. | 70.43 | 59.88 | 79.04 | 333 | Rams |
| 70 | 22 | Joe Noteboom | 70.13 | 60.00 | 78.16 | 220 | Rams |
| 71 | 23 | Jawaan Taylor | 70.12 | 59.80 | 78.43 | 1209 | Chiefs |
| 72 | 24 | Brandon Coleman | 69.65 | 59.80 | 77.29 | 1013 | Commanders |
| 73 | 25 | Broderick Jones | 69.10 | 58.70 | 77.52 | 1117 | Steelers |
| 74 | 26 | Jackson Barton | 68.77 | 61.56 | 72.77 | 157 | Cardinals |
| 75 | 27 | Orlando Brown Jr. | 68.49 | 58.41 | 76.46 | 637 | Bengals |
| 76 | 28 | Amarius Mims | 67.69 | 57.97 | 75.15 | 835 | Bengals |
| 77 | 29 | Yosh Nijman | 67.66 | 59.08 | 73.56 | 187 | Panthers |
| 78 | 30 | Chuma Edoga | 66.68 | 56.41 | 74.92 | 226 | Cowboys |
| 79 | 31 | Jedrick Wills Jr. | 66.61 | 56.46 | 74.67 | 245 | Browns |
| 80 | 32 | Devin Cochran | 66.37 | 56.24 | 74.40 | 152 | Bengals |
| 81 | 33 | Dan Skipper | 66.30 | 57.25 | 72.84 | 324 | Lions |
| 82 | 34 | David Quessenberry | 66.02 | 56.93 | 72.61 | 133 | Vikings |
| 83 | 35 | James Hudson III | 65.71 | 55.44 | 73.95 | 222 | Browns |
| 84 | 36 | Larry Borom | 65.55 | 56.42 | 72.20 | 329 | Bears |
| 85 | 37 | Chris Hubbard | 65.53 | 53.16 | 76.67 | 257 | Giants |
| 86 | 38 | Vederian Lowe | 65.49 | 54.58 | 74.61 | 803 | Patriots |
| 87 | 39 | Wanya Morris | 65.08 | 53.97 | 74.47 | 732 | Chiefs |
| 88 | 40 | Caedan Wallace | 64.82 | 53.64 | 74.31 | 129 | Patriots |
| 89 | 41 | Carter Warren | 64.42 | 53.84 | 73.08 | 141 | Jets |
| 90 | 42 | Ryan Van Demark | 63.77 | 56.99 | 67.18 | 199 | Bills |
| 91 | 43 | Trent Scott | 63.53 | 52.48 | 72.83 | 288 | Commanders |
| 92 | 44 | Tyler Guyton | 63.46 | 51.27 | 74.34 | 668 | Cowboys |
| 93 | 45 | Mike Jerrell | 63.34 | 53.15 | 71.46 | 250 | Seahawks |
| 94 | 46 | Thayer Munford Jr. | 63.12 | 53.63 | 70.27 | 201 | Raiders |
| 95 | 47 | Fred Johnson | 62.98 | 52.45 | 71.56 | 490 | Eagles |
| 96 | 48 | Patrick Paul | 62.55 | 51.15 | 72.34 | 338 | Dolphins |
| 97 | 49 | Dawand Jones | 62.40 | 50.20 | 73.30 | 511 | Browns |
| 98 | 50 | Charlie Heck | 62.27 | 52.48 | 69.83 | 117 | 49ers |

### Rotation/backup (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 99 | 1 | Nicholas Petit-Frere | 61.34 | 49.28 | 72.05 | 621 | Titans |
| 100 | 2 | Kingsley Suamataia | 60.73 | 50.09 | 69.48 | 198 | Chiefs |
| 101 | 3 | Blake Fisher | 60.68 | 49.34 | 70.38 | 478 | Texans |
| 102 | 4 | Stone Forsythe | 60.64 | 49.04 | 70.71 | 414 | Seahawks |
| 103 | 5 | Kiran Amegadjie | 60.52 | 52.12 | 66.17 | 126 | Bears |
| 104 | 6 | Demontrey Jacobs | 55.26 | 40.00 | 70.39 | 867 | Patriots |

## TE — Tight End

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | George Kittle | 85.86 | 89.42 | 77.99 | 489 | 49ers |
| 2 | 2 | Trey McBride | 82.35 | 85.81 | 74.57 | 581 | Cardinals |
| 3 | 3 | Brock Bowers | 80.47 | 84.99 | 71.74 | 654 | Raiders |
| 4 | 4 | Mark Andrews | 80.37 | 80.75 | 75.32 | 455 | Ravens |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Travis Kelce | 77.28 | 72.70 | 76.62 | 698 | Chiefs |
| 6 | 2 | Jonnu Smith | 76.68 | 75.62 | 72.90 | 486 | Dolphins |
| 7 | 3 | Dallas Goedert | 74.75 | 70.92 | 73.42 | 325 | Eagles |
| 8 | 4 | Sam LaPorta | 74.34 | 72.86 | 70.93 | 546 | Lions |
| 9 | 5 | Austin Hooper | 74.27 | 73.07 | 70.62 | 335 | Patriots |
| 10 | 6 | T.J. Hockenson | 74.27 | 72.53 | 71.09 | 366 | Vikings |

### Starter (58 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Foster Moreau | 73.73 | 68.59 | 73.58 | 395 | Saints |
| 12 | 2 | Evan Engram | 73.50 | 67.85 | 73.80 | 260 | Jaguars |
| 13 | 3 | Dalton Kincaid | 72.84 | 68.55 | 71.92 | 365 | Bills |
| 14 | 4 | Hunter Henry | 72.64 | 69.47 | 70.73 | 548 | Patriots |
| 15 | 5 | Isaiah Likely | 72.30 | 71.93 | 67.92 | 386 | Ravens |
| 16 | 6 | Zach Ertz | 72.13 | 66.99 | 71.98 | 658 | Commanders |
| 17 | 7 | Jordan Akins | 71.92 | 65.23 | 73.14 | 348 | Browns |
| 18 | 8 | Tucker Kraft | 71.88 | 67.04 | 71.46 | 538 | Packers |
| 19 | 9 | Mike Gesicki | 71.76 | 70.29 | 68.34 | 449 | Bengals |
| 20 | 10 | Noah Gray | 71.75 | 68.95 | 69.52 | 380 | Chiefs |
| 21 | 11 | Andrew Ogletree | 71.68 | 64.66 | 73.18 | 173 | Colts |
| 22 | 12 | David Njoku | 71.44 | 63.50 | 73.76 | 415 | Browns |
| 23 | 13 | Will Dissly | 71.28 | 65.25 | 71.91 | 361 | Chargers |
| 24 | 14 | Mo Alie-Cox | 71.08 | 62.64 | 73.84 | 227 | Colts |
| 25 | 15 | Josh Oliver | 70.71 | 68.87 | 67.63 | 254 | Vikings |
| 26 | 16 | Stone Smartt | 70.19 | 62.93 | 71.92 | 138 | Chargers |
| 27 | 17 | Darnell Washington | 69.53 | 66.99 | 67.07 | 257 | Steelers |
| 28 | 18 | Pat Freiermuth | 69.52 | 66.70 | 67.30 | 516 | Steelers |
| 29 | 19 | Colby Parkinson | 68.77 | 62.46 | 69.64 | 390 | Rams |
| 30 | 20 | Cole Kmet | 68.75 | 60.59 | 71.26 | 637 | Bears |
| 31 | 21 | Hunter Long | 68.71 | 62.69 | 69.33 | 110 | Rams |
| 32 | 22 | Noah Fant | 68.70 | 64.82 | 67.43 | 426 | Seahawks |
| 33 | 23 | Taysom Hill | 68.58 | 64.72 | 67.28 | 103 | Saints |
| 34 | 24 | Erick All | 68.58 | 59.96 | 71.51 | 112 | Bengals |
| 35 | 25 | Kylen Granson | 68.56 | 58.92 | 72.39 | 268 | Colts |
| 36 | 26 | Josh Whyle | 68.54 | 62.67 | 69.03 | 204 | Titans |
| 37 | 27 | Tyler Conklin | 68.46 | 58.90 | 72.23 | 556 | Jets |
| 38 | 28 | Chigoziem Okonkwo | 68.35 | 59.92 | 71.11 | 425 | Titans |
| 39 | 29 | Kyle Pitts | 68.07 | 59.63 | 70.83 | 511 | Falcons |
| 40 | 30 | Chris Manhertz | 67.99 | 64.27 | 66.57 | 139 | Giants |
| 41 | 31 | Payne Durham | 67.98 | 60.43 | 69.95 | 191 | Buccaneers |
| 42 | 32 | Juwan Johnson | 67.96 | 65.64 | 65.30 | 467 | Saints |
| 43 | 33 | Brenton Strange | 67.86 | 64.18 | 66.41 | 320 | Jaguars |
| 44 | 34 | Cade Otton | 67.70 | 63.82 | 66.42 | 573 | Buccaneers |
| 45 | 35 | Luke Farrell | 67.54 | 58.90 | 70.49 | 152 | Jaguars |
| 46 | 36 | Ben Sinnott | 66.46 | 57.08 | 70.07 | 122 | Commanders |
| 47 | 37 | Nate Adkins | 66.10 | 62.04 | 64.99 | 181 | Broncos |
| 48 | 38 | Adam Trautman | 66.02 | 57.95 | 68.45 | 288 | Broncos |
| 49 | 39 | Brevyn Spann-Ford | 65.86 | 57.04 | 68.96 | 146 | Cowboys |
| 50 | 40 | Lucas Krull | 65.83 | 55.30 | 70.45 | 252 | Broncos |
| 51 | 41 | Dalton Schultz | 65.62 | 60.80 | 65.17 | 648 | Texans |
| 52 | 42 | Dawson Knox | 65.54 | 57.77 | 67.72 | 392 | Bills |
| 53 | 43 | Michael Mayer | 65.51 | 58.49 | 67.01 | 284 | Raiders |
| 54 | 44 | AJ Barner | 64.60 | 60.62 | 63.42 | 253 | Seahawks |
| 55 | 45 | Gerald Everett | 64.57 | 52.32 | 70.71 | 133 | Bears |
| 56 | 46 | Johnny Mundt | 64.50 | 58.33 | 65.25 | 236 | Vikings |
| 57 | 47 | Charlie Woerner | 64.04 | 59.51 | 63.34 | 131 | Falcons |
| 58 | 48 | Harrison Bryant | 64.01 | 60.00 | 62.84 | 101 | Raiders |
| 59 | 49 | Jake Ferguson | 63.60 | 55.15 | 66.37 | 427 | Cowboys |
| 60 | 50 | Daniel Bellinger | 63.22 | 58.43 | 62.75 | 207 | Giants |
| 61 | 51 | Luke Schoonmaker | 63.22 | 58.98 | 62.27 | 213 | Cowboys |
| 62 | 52 | Theo Johnson | 63.20 | 54.74 | 65.99 | 446 | Giants |
| 63 | 53 | Grant Calcaterra | 63.20 | 55.07 | 65.70 | 347 | Eagles |
| 64 | 54 | Nick Vannett | 62.89 | 57.33 | 63.10 | 145 | Titans |
| 65 | 55 | Jeremy Ruckert | 62.61 | 52.72 | 66.67 | 198 | Jets |
| 66 | 56 | Brock Wright | 62.52 | 55.23 | 64.26 | 235 | Lions |
| 67 | 57 | Ja'Tavion Sanders | 62.45 | 54.47 | 64.81 | 359 | Panthers |
| 68 | 58 | Pharaoh Brown | 62.43 | 54.20 | 65.01 | 107 | Seahawks |

### Rotation/backup (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 69 | 1 | Tommy Tremble | 61.19 | 56.21 | 60.88 | 303 | Panthers |
| 70 | 2 | John Bates | 61.19 | 52.15 | 64.49 | 252 | Commanders |
| 71 | 3 | Durham Smythe | 60.76 | 51.88 | 63.91 | 156 | Dolphins |
| 72 | 4 | Cade Stover | 60.51 | 55.74 | 60.03 | 192 | Texans |
| 73 | 5 | Tip Reiman | 60.20 | 54.59 | 60.46 | 169 | Cardinals |
| 74 | 6 | Greg Dulcich | 59.91 | 50.22 | 63.78 | 127 | Giants |
| 75 | 7 | Hayden Hurst | 59.43 | 50.92 | 62.26 | 103 | Chargers |
| 76 | 8 | Eric Saubert | 59.42 | 55.44 | 58.24 | 177 | 49ers |
| 77 | 9 | Davis Allen | 59.33 | 52.98 | 60.25 | 176 | Rams |
| 78 | 10 | Drew Sample | 58.61 | 52.80 | 59.05 | 278 | Bengals |
| 79 | 11 | Julian Hill | 56.48 | 46.60 | 60.52 | 228 | Dolphins |

## WR — Wide Receiver

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nico Collins | 89.09 | 88.87 | 85.37 | 447 | Texans |
| 2 | 2 | A.J. Brown | 87.91 | 87.51 | 84.28 | 473 | Eagles |
| 3 | 3 | Puka Nacua | 86.83 | 87.60 | 82.57 | 370 | Rams |
| 4 | 4 | Justin Jefferson | 86.47 | 86.60 | 82.55 | 700 | Vikings |
| 5 | 5 | Amon-Ra St. Brown | 85.33 | 88.63 | 79.71 | 611 | Lions |
| 6 | 6 | Ladd McConkey | 84.81 | 82.24 | 82.35 | 553 | Chargers |
| 7 | 7 | Mike Evans | 84.79 | 87.44 | 79.51 | 463 | Buccaneers |
| 8 | 8 | Brian Thomas Jr. | 84.17 | 80.12 | 82.50 | 552 | Jaguars |
| 9 | 9 | Ja'Marr Chase | 83.71 | 85.80 | 78.74 | 745 | Bengals |
| 10 | 10 | Drake London | 83.10 | 86.96 | 77.17 | 595 | Falcons |
| 11 | 11 | Tee Higgins | 82.32 | 85.65 | 76.68 | 476 | Bengals |
| 12 | 12 | Terry McLaurin | 81.75 | 82.10 | 77.71 | 717 | Commanders |
| 13 | 13 | Jordan Whittington | 81.59 | 68.09 | 85.01 | 129 | Rams |
| 14 | 14 | Malik Nabers | 81.13 | 85.46 | 74.96 | 600 | Giants |
| 15 | 15 | Zay Flowers | 81.04 | 80.37 | 77.55 | 499 | Ravens |
| 16 | 16 | Jameson Williams | 80.93 | 73.05 | 81.33 | 535 | Lions |
| 17 | 17 | Tyreek Hill | 80.74 | 72.22 | 81.48 | 580 | Dolphins |
| 18 | 18 | Khalil Shakir | 80.57 | 77.38 | 78.44 | 495 | Bills |
| 19 | 19 | Brandon Aiyuk | 80.52 | 70.95 | 81.83 | 225 | 49ers |
| 20 | 20 | CeeDee Lamb | 80.48 | 76.53 | 78.76 | 566 | Cowboys |
| 21 | 21 | George Pickens | 80.47 | 77.13 | 78.42 | 498 | Steelers |
| 22 | 22 | Chris Olave | 80.04 | 76.36 | 78.17 | 200 | Saints |

### Good (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | DeVonta Smith | 79.89 | 79.33 | 76.34 | 505 | Eagles |
| 24 | 2 | Chris Godwin | 79.87 | 80.52 | 75.68 | 265 | Buccaneers |
| 25 | 3 | Tylan Wallace | 79.36 | 62.94 | 84.35 | 148 | Ravens |
| 26 | 4 | Josh Downs | 79.22 | 78.84 | 75.58 | 381 | Colts |
| 27 | 5 | Jauan Jennings | 79.17 | 79.26 | 75.28 | 459 | 49ers |
| 28 | 6 | Marvin Harrison Jr. | 79.09 | 76.58 | 76.60 | 579 | Cardinals |
| 29 | 7 | D.K. Metcalf | 79.09 | 73.83 | 78.08 | 590 | Seahawks |
| 30 | 8 | Christian Watson | 78.77 | 66.08 | 81.76 | 295 | Packers |
| 31 | 9 | Jaxon Smith-Njigba | 78.51 | 81.00 | 73.33 | 666 | Seahawks |
| 32 | 10 | Davante Adams | 78.28 | 75.03 | 76.18 | 557 | Jets |
| 33 | 11 | Jakobi Meyers | 78.21 | 77.43 | 74.79 | 627 | Raiders |
| 34 | 12 | Rashid Shaheed | 78.07 | 65.08 | 81.22 | 181 | Saints |
| 35 | 13 | Jalen Coker | 78.06 | 68.59 | 79.31 | 297 | Panthers |
| 36 | 14 | Jaylen Waddle | 77.79 | 71.24 | 77.47 | 513 | Dolphins |
| 37 | 15 | Garrett Wilson | 77.69 | 78.90 | 73.19 | 691 | Jets |
| 38 | 16 | DJ Moore | 77.61 | 73.50 | 75.98 | 722 | Bears |
| 39 | 17 | Marvin Mims Jr. | 77.50 | 64.71 | 80.54 | 198 | Broncos |
| 40 | 18 | DeAndre Hopkins | 77.45 | 75.54 | 74.63 | 419 | Chiefs |
| 41 | 19 | Kalif Raymond | 77.44 | 65.43 | 80.06 | 158 | Lions |
| 42 | 20 | Alec Pierce | 77.39 | 72.26 | 76.31 | 485 | Colts |
| 43 | 21 | Jayden Reed | 77.38 | 70.35 | 77.32 | 430 | Packers |
| 44 | 22 | Deebo Samuel | 77.30 | 69.48 | 77.67 | 405 | 49ers |
| 45 | 23 | Jerry Jeudy | 77.23 | 73.50 | 75.39 | 757 | Browns |
| 46 | 24 | Stefon Diggs | 77.18 | 75.05 | 74.48 | 282 | Texans |
| 47 | 25 | Darnell Mooney | 77.10 | 73.32 | 75.29 | 557 | Falcons |
| 48 | 26 | Keon Coleman | 76.67 | 66.49 | 78.31 | 404 | Bills |
| 49 | 27 | Calvin Ridley | 76.36 | 72.91 | 74.37 | 582 | Titans |
| 50 | 28 | Jordan Addison | 76.32 | 72.30 | 74.64 | 600 | Vikings |
| 51 | 29 | Cooper Kupp | 76.04 | 70.24 | 75.31 | 455 | Rams |
| 52 | 30 | Courtland Sutton | 75.62 | 75.43 | 71.87 | 650 | Broncos |
| 53 | 31 | Rashod Bateman | 75.60 | 70.49 | 74.51 | 522 | Ravens |
| 54 | 32 | Amari Cooper | 75.56 | 67.80 | 75.90 | 451 | Bills |
| 55 | 33 | Tank Dell | 75.51 | 71.85 | 73.63 | 467 | Texans |
| 56 | 34 | Jermaine Burton | 74.87 | 59.36 | 79.37 | 100 | Bengals |
| 57 | 35 | Michael Pittman Jr. | 74.83 | 71.32 | 72.87 | 511 | Colts |
| 58 | 36 | Tutu Atwell | 74.77 | 68.23 | 74.44 | 277 | Rams |
| 59 | 37 | Christian Kirk | 74.74 | 65.96 | 75.62 | 230 | Jaguars |
| 60 | 38 | KaVontae Turpin | 74.73 | 66.32 | 75.41 | 210 | Cowboys |
| 61 | 39 | Adam Thielen | 74.48 | 73.40 | 71.21 | 319 | Panthers |
| 62 | 40 | Noah Brown | 74.14 | 67.15 | 74.05 | 295 | Commanders |

### Starter (81 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 63 | 1 | Dyami Brown | 73.93 | 64.91 | 74.94 | 377 | Commanders |
| 64 | 2 | Tim Patrick | 73.93 | 65.78 | 74.47 | 371 | Lions |
| 65 | 3 | Demario Douglas | 73.79 | 68.50 | 72.79 | 477 | Patriots |
| 66 | 4 | KhaDarel Hodge | 73.62 | 61.20 | 76.46 | 121 | Falcons |
| 67 | 5 | Romeo Doubs | 73.25 | 68.44 | 71.99 | 406 | Packers |
| 68 | 6 | Calvin Austin III | 73.04 | 63.04 | 74.58 | 423 | Steelers |
| 69 | 7 | Mike Williams | 72.79 | 58.98 | 76.38 | 366 | Steelers |
| 70 | 8 | Joshua Palmer | 72.67 | 65.64 | 72.61 | 428 | Chargers |
| 71 | 9 | Demarcus Robinson | 72.53 | 64.84 | 72.83 | 618 | Rams |
| 72 | 10 | Xavier Worthy | 72.50 | 68.32 | 70.90 | 603 | Chiefs |
| 73 | 11 | Quentin Johnston | 72.36 | 66.20 | 71.83 | 464 | Chargers |
| 74 | 12 | Marquez Valdes-Scantling | 72.11 | 61.17 | 74.15 | 315 | Saints |
| 75 | 13 | Ricky Pearsall | 72.06 | 62.73 | 73.24 | 324 | 49ers |
| 76 | 14 | Diontae Johnson | 72.05 | 64.02 | 72.53 | 277 | Texans |
| 77 | 15 | Rakim Jarrett | 71.85 | 59.79 | 74.50 | 117 | Buccaneers |
| 78 | 16 | Rome Odunze | 71.81 | 63.80 | 72.28 | 677 | Bears |
| 79 | 17 | Keenan Allen | 71.72 | 64.27 | 71.89 | 596 | Bears |
| 80 | 18 | Tyler Lockett | 71.55 | 65.03 | 71.22 | 589 | Seahawks |
| 81 | 19 | Bo Melton | 71.51 | 61.35 | 73.13 | 118 | Packers |
| 82 | 20 | Jalen Brooks | 71.47 | 59.52 | 74.06 | 236 | Cowboys |
| 83 | 21 | Josh Reynolds | 71.35 | 60.69 | 73.24 | 220 | Jaguars |
| 84 | 22 | Olamide Zaccheaus | 71.31 | 67.19 | 69.69 | 403 | Commanders |
| 85 | 23 | JuJu Smith-Schuster | 71.29 | 60.00 | 73.52 | 321 | Chiefs |
| 86 | 24 | Nick Westbrook-Ikhine | 71.17 | 62.87 | 71.80 | 469 | Titans |
| 87 | 25 | Tyler Johnson | 71.00 | 64.00 | 70.93 | 215 | Rams |
| 88 | 26 | Michael Wilson | 70.96 | 62.62 | 71.60 | 538 | Cardinals |
| 89 | 27 | Dontayvion Wicks | 70.95 | 63.69 | 71.02 | 345 | Packers |
| 90 | 28 | Ray-Ray McCloud III | 70.87 | 62.57 | 71.49 | 598 | Falcons |
| 91 | 29 | Kendrick Bourne | 70.63 | 61.71 | 71.58 | 317 | Patriots |
| 92 | 30 | Devaughn Vele | 70.54 | 66.76 | 68.73 | 349 | Broncos |
| 93 | 31 | Mack Hollins | 70.53 | 61.39 | 71.61 | 495 | Bills |
| 94 | 32 | Greg Dortch | 70.51 | 62.62 | 70.91 | 299 | Cardinals |
| 95 | 33 | Darius Slayton | 70.50 | 59.07 | 72.81 | 575 | Giants |
| 96 | 34 | Nelson Agholor | 70.29 | 62.62 | 70.57 | 250 | Ravens |
| 97 | 35 | Adonai Mitchell | 70.17 | 58.78 | 72.46 | 221 | Colts |
| 98 | 36 | Curtis Samuel | 70.08 | 64.28 | 69.36 | 263 | Bills |
| 99 | 37 | Chris Conley | 69.87 | 59.91 | 71.39 | 124 | 49ers |
| 100 | 38 | Brandin Cooks | 69.80 | 62.61 | 69.82 | 317 | Cowboys |
| 101 | 39 | Kayshon Boutte | 69.59 | 61.23 | 70.24 | 507 | Patriots |
| 102 | 40 | Gabe Davis | 69.48 | 55.26 | 73.29 | 264 | Jaguars |
| 103 | 41 | Cedrick Wilson Jr. | 69.46 | 62.02 | 69.62 | 220 | Saints |
| 104 | 42 | Ryan Flournoy | 69.36 | 61.56 | 69.72 | 103 | Cowboys |
| 105 | 43 | Allen Lazard | 69.33 | 62.23 | 69.30 | 451 | Jets |
| 106 | 44 | Tyler Boyd | 69.08 | 60.00 | 70.12 | 464 | Titans |
| 107 | 45 | Jalen Nailor | 68.89 | 59.41 | 70.15 | 462 | Vikings |
| 108 | 46 | Cedric Tillman | 68.78 | 62.43 | 68.36 | 300 | Browns |
| 109 | 47 | Dante Pettis | 68.66 | 64.08 | 67.28 | 104 | Saints |
| 110 | 48 | Jalen McMillan | 68.55 | 60.65 | 68.96 | 430 | Buccaneers |
| 111 | 49 | Trey Palmer | 68.51 | 59.04 | 69.76 | 188 | Buccaneers |
| 112 | 50 | Jalen Tolbert | 68.49 | 60.67 | 68.86 | 597 | Cowboys |
| 113 | 51 | Tre Tucker | 68.46 | 57.50 | 70.51 | 683 | Raiders |
| 114 | 52 | Kevin Austin Jr. | 68.21 | 57.74 | 70.00 | 210 | Saints |
| 115 | 53 | Devin Duvernay | 67.97 | 59.68 | 68.58 | 141 | Jaguars |
| 116 | 54 | David Moore | 67.50 | 62.28 | 66.46 | 358 | Panthers |
| 117 | 55 | Simi Fehoko | 67.37 | 57.73 | 68.71 | 136 | Chargers |
| 118 | 56 | Wan'Dale Robinson | 67.17 | 63.29 | 65.42 | 618 | Giants |
| 119 | 57 | Ryan Miller | 67.16 | 59.52 | 67.42 | 151 | Buccaneers |
| 120 | 58 | DJ Turner | 67.04 | 56.89 | 68.66 | 246 | Raiders |
| 121 | 59 | John Metchie III | 66.90 | 59.52 | 67.03 | 314 | Texans |
| 122 | 60 | Parker Washington | 66.83 | 59.77 | 66.79 | 404 | Jaguars |
| 123 | 61 | Justin Watson | 66.79 | 54.59 | 69.52 | 430 | Chiefs |
| 124 | 62 | Brandon Powell | 66.68 | 59.51 | 66.70 | 130 | Vikings |
| 125 | 63 | Robert Woods | 66.37 | 59.10 | 66.44 | 226 | Texans |
| 126 | 64 | K.J. Osborn | 66.19 | 56.37 | 67.63 | 143 | Commanders |
| 127 | 65 | Van Jefferson | 66.06 | 57.71 | 66.71 | 442 | Steelers |
| 128 | 66 | Troy Franklin | 65.77 | 56.18 | 67.09 | 297 | Broncos |
| 129 | 67 | Jahan Dotson | 65.57 | 54.56 | 67.65 | 492 | Eagles |
| 130 | 68 | Elijah Moore | 65.47 | 58.54 | 65.36 | 623 | Browns |
| 131 | 69 | Xavier Hutchinson | 65.34 | 58.45 | 65.21 | 326 | Texans |
| 132 | 70 | Zay Jones | 65.33 | 56.36 | 66.31 | 184 | Cardinals |
| 133 | 71 | Luke McCaffrey | 65.23 | 56.27 | 66.21 | 283 | Commanders |
| 134 | 72 | Malik Washington | 64.75 | 58.53 | 64.25 | 270 | Dolphins |
| 135 | 73 | Jake Bobo | 64.50 | 57.71 | 64.31 | 163 | Seahawks |
| 136 | 74 | Xavier Gipson | 64.45 | 54.79 | 65.81 | 138 | Jets |
| 137 | 75 | Xavier Legette | 64.34 | 59.43 | 63.13 | 441 | Panthers |
| 138 | 76 | Andrei Iosivas | 63.92 | 52.92 | 65.99 | 620 | Bengals |
| 139 | 77 | Jalin Hyatt | 63.83 | 52.92 | 65.86 | 230 | Giants |
| 140 | 78 | Jonathan Mingo | 63.49 | 53.36 | 65.10 | 285 | Cowboys |
| 141 | 79 | Jamison Crowder | 63.13 | 56.62 | 62.79 | 127 | Commanders |
| 142 | 80 | Sterling Shepard | 62.94 | 55.51 | 63.09 | 393 | Buccaneers |
| 143 | 81 | Michael Woods II | 62.94 | 53.55 | 64.15 | 211 | Browns |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 144 | 1 | Johnny Wilson | 60.33 | 54.72 | 59.50 | 177 | Eagles |
| 145 | 2 | Mason Tipton | 58.48 | 52.57 | 57.81 | 253 | Saints |
| 146 | 3 | Ja'Lynn Polk | 57.44 | 49.15 | 58.05 | 272 | Patriots |
