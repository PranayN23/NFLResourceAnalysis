# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:37:32Z
- **Requested analysis_year:** 2025 (clamped to 2025)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Creed Humphrey | 98.17 | 92.30 | 97.92 | 1232 | Chiefs |
| 2 | 2 | Erik McCoy | 94.36 | 84.87 | 96.52 | 293 | Saints |
| 3 | 3 | Frank Ragnow | 92.86 | 86.10 | 93.20 | 1129 | Lions |
| 4 | 4 | Tyler Linderbaum | 88.56 | 79.90 | 90.17 | 1227 | Ravens |
| 5 | 5 | Zach Frazier | 86.64 | 77.90 | 88.30 | 1021 | Steelers |
| 6 | 6 | Drew Dalman | 84.83 | 75.98 | 86.56 | 554 | Falcons |
| 7 | 7 | Hjalte Froholdt | 84.43 | 76.10 | 85.81 | 1078 | Cardinals |
| 8 | 8 | Aaron Brewer | 82.89 | 73.30 | 85.11 | 1139 | Dolphins |
| 9 | 9 | Joe Tippmann | 82.83 | 73.40 | 84.95 | 1067 | Jets |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Connor McGovern | 79.37 | 69.50 | 81.78 | 1164 | Bills |
| 11 | 2 | Coleman Shelton | 77.25 | 66.40 | 80.31 | 1121 | Bears |
| 12 | 3 | Jake Brendel | 77.11 | 65.00 | 81.02 | 1072 | 49ers |
| 13 | 4 | Cam Jurgens | 76.67 | 67.30 | 78.75 | 1217 | Eagles |
| 14 | 5 | Ryan Kelly | 76.62 | 66.08 | 79.48 | 601 | Colts |
| 15 | 6 | Cooper Beebe | 76.32 | 65.40 | 79.44 | 1059 | Cowboys |
| 16 | 7 | Luke Wattenberg | 75.40 | 64.03 | 78.81 | 864 | Broncos |
| 17 | 8 | Garrett Bradbury | 75.11 | 62.80 | 79.15 | 1191 | Vikings |
| 18 | 9 | Alex Forsyth | 75.07 | 63.54 | 78.59 | 292 | Broncos |
| 19 | 10 | Ethan Pocic | 75.02 | 63.60 | 78.46 | 1073 | Browns |
| 20 | 11 | Tyler Biadasz | 75.00 | 64.20 | 78.03 | 1166 | Commanders |
| 21 | 12 | Juice Scruggs | 74.19 | 62.94 | 77.53 | 944 | Texans |

### Starter (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Ted Karras | 73.77 | 64.10 | 76.05 | 1136 | Bengals |
| 23 | 2 | Jarrett Patterson | 73.75 | 63.43 | 76.47 | 688 | Texans |
| 24 | 3 | Austin Corbett | 73.66 | 61.58 | 77.55 | 291 | Panthers |
| 25 | 4 | John Michael Schmitz Jr. | 73.58 | 61.40 | 77.53 | 983 | Giants |
| 26 | 5 | Bradley Bozeman | 73.08 | 61.20 | 76.83 | 1112 | Chargers |
| 27 | 6 | Ryan Neuzil | 72.27 | 58.85 | 77.05 | 578 | Falcons |
| 28 | 7 | Brady Christensen | 72.04 | 62.29 | 74.38 | 399 | Panthers |
| 29 | 8 | Olusegun Oluwatimi | 71.60 | 62.79 | 73.31 | 435 | Seahawks |
| 30 | 9 | David Andrews | 71.28 | 59.13 | 75.22 | 193 | Patriots |
| 31 | 10 | Beaux Limmer | 70.73 | 55.50 | 76.71 | 1040 | Rams |
| 32 | 11 | Danny Pinter | 70.45 | 63.44 | 70.96 | 138 | Colts |
| 33 | 12 | Mitch Morse | 70.08 | 57.30 | 74.44 | 1021 | Jaguars |
| 34 | 13 | Corey Levin | 69.50 | 58.20 | 72.86 | 133 | Titans |
| 35 | 14 | Graham Barton | 69.02 | 55.60 | 73.80 | 1111 | Buccaneers |
| 36 | 15 | Lloyd Cushenberry III | 68.59 | 56.20 | 72.69 | 499 | Titans |
| 37 | 16 | Andre James | 67.82 | 56.01 | 71.53 | 702 | Raiders |
| 38 | 17 | Ryan McCollum | 66.93 | 56.12 | 69.97 | 153 | Steelers |
| 39 | 18 | Josh Myers | 66.68 | 54.20 | 70.84 | 1067 | Packers |
| 40 | 19 | Shane Lemieux | 65.80 | 54.79 | 68.98 | 337 | Saints |
| 41 | 20 | Daniel Brunskill | 65.78 | 56.08 | 68.08 | 684 | Titans |
| 42 | 21 | Sedrick Van Pran-Granger | 64.08 | 57.84 | 64.07 | 125 | Bills |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Pat Surtain II | 89.65 | 85.10 | 88.51 | 1054 | Broncos |
| 2 | 2 | Derek Stingley Jr. | 86.68 | 84.40 | 86.79 | 1119 | Texans |
| 3 | 3 | Quinyon Mitchell | 84.36 | 79.00 | 83.76 | 1104 | Eagles |
| 4 | 4 | Christian Benford | 83.98 | 78.60 | 85.26 | 1046 | Bills |
| 5 | 5 | Marlon Humphrey | 83.57 | 81.00 | 82.89 | 1000 | Ravens |
| 6 | 6 | Garrett Williams | 83.50 | 83.02 | 83.21 | 778 | Cardinals |
| 7 | 7 | Trent McDuffie | 83.33 | 80.70 | 82.10 | 1132 | Chiefs |
| 8 | 8 | Kamari Lassiter | 81.58 | 77.50 | 81.11 | 906 | Texans |
| 9 | 9 | Cooper DeJean | 80.79 | 79.00 | 77.81 | 830 | Eagles |
| 10 | 10 | Darius Slay | 80.34 | 74.20 | 81.45 | 897 | Eagles |
| 11 | 11 | Sauce Gardner | 80.04 | 73.10 | 81.76 | 879 | Jets |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Jourdan Lewis | 79.88 | 79.00 | 78.95 | 871 | Cowboys |
| 13 | 2 | Christian Gonzalez | 79.86 | 78.20 | 82.20 | 978 | Patriots |
| 14 | 3 | Renardo Green | 79.47 | 72.93 | 80.65 | 675 | 49ers |
| 15 | 4 | Jamel Dean | 79.21 | 74.94 | 80.83 | 745 | Buccaneers |
| 16 | 5 | Jaylon Johnson | 78.93 | 74.20 | 79.98 | 1031 | Bears |
| 17 | 6 | Tarheeb Still | 78.05 | 74.80 | 78.01 | 826 | Chargers |
| 18 | 7 | Kyler Gordon | 77.42 | 74.99 | 77.62 | 724 | Bears |
| 19 | 8 | Jalen Ramsey | 77.16 | 71.90 | 78.26 | 1027 | Dolphins |
| 20 | 9 | Jaire Alexander | 77.07 | 74.58 | 82.02 | 361 | Packers |
| 21 | 10 | Carlton Davis III | 76.65 | 71.51 | 79.54 | 697 | Lions |
| 22 | 11 | Nate Wiggins | 76.26 | 68.69 | 77.14 | 769 | Ravens |
| 23 | 12 | Byron Murphy Jr. | 76.22 | 73.50 | 76.32 | 1109 | Vikings |
| 24 | 13 | Mike Jackson | 76.09 | 68.10 | 77.84 | 1204 | Panthers |
| 25 | 14 | D.J. Reed | 76.03 | 70.10 | 77.89 | 880 | Jets |
| 26 | 15 | Samuel Womack III | 75.97 | 70.21 | 81.22 | 673 | Colts |
| 27 | 16 | Denzel Ward | 75.69 | 68.68 | 78.16 | 757 | Browns |
| 28 | 17 | Terell Smith | 75.47 | 68.26 | 82.61 | 207 | Bears |
| 29 | 18 | A.J. Terrell | 75.45 | 69.50 | 75.84 | 1085 | Falcons |
| 30 | 19 | Deommodore Lenoir | 75.40 | 71.70 | 74.89 | 922 | 49ers |
| 31 | 20 | Tariq Woolen | 75.21 | 65.70 | 78.65 | 889 | Seahawks |
| 32 | 21 | Devon Witherspoon | 75.19 | 69.20 | 76.11 | 1103 | Seahawks |
| 33 | 22 | Mike Hilton | 75.01 | 68.22 | 76.45 | 737 | Bengals |
| 34 | 23 | DaRon Bland | 75.01 | 69.53 | 80.18 | 436 | Cowboys |
| 35 | 24 | Jaylon Jones | 74.60 | 67.90 | 75.63 | 1146 | Colts |
| 36 | 25 | Carrington Valentine | 74.41 | 69.77 | 75.17 | 606 | Packers |

### Starter (83 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 37 | 1 | Jaylen Watson | 73.85 | 69.78 | 77.00 | 433 | Chiefs |
| 38 | 2 | Andru Phillips | 73.79 | 73.63 | 72.66 | 614 | Giants |
| 39 | 3 | Mike Sainristil | 73.74 | 64.50 | 75.73 | 1158 | Commanders |
| 40 | 4 | Mike Hughes | 73.73 | 70.95 | 74.06 | 720 | Falcons |
| 41 | 5 | Clark Phillips III | 73.35 | 69.93 | 75.74 | 409 | Falcons |
| 42 | 6 | DJ Turner II | 73.28 | 66.59 | 77.25 | 508 | Bengals |
| 43 | 7 | Isaiah Rodgers | 73.11 | 67.01 | 74.67 | 413 | Eagles |
| 44 | 8 | Kelee Ringo | 72.97 | 63.84 | 80.41 | 127 | Eagles |
| 45 | 9 | Zyon McCollum | 72.68 | 66.10 | 74.47 | 1123 | Buccaneers |
| 46 | 10 | Joshua Williams | 72.60 | 65.86 | 76.16 | 411 | Chiefs |
| 47 | 11 | Kenny Moore II | 72.44 | 68.20 | 73.35 | 1013 | Colts |
| 48 | 12 | Cam Taylor-Britt | 72.11 | 64.30 | 76.00 | 1036 | Bengals |
| 49 | 13 | Marcus Jones | 72.00 | 65.98 | 78.92 | 586 | Patriots |
| 50 | 14 | Kristian Fulton | 71.99 | 66.60 | 74.85 | 827 | Chargers |
| 51 | 15 | Kader Kohou | 71.79 | 68.24 | 71.36 | 708 | Dolphins |
| 52 | 16 | Troy Hill | 71.54 | 67.64 | 76.64 | 236 | Buccaneers |
| 53 | 17 | Kool-Aid McKinstry | 70.98 | 66.45 | 72.76 | 680 | Saints |
| 54 | 18 | Jarrian Jones | 70.82 | 62.30 | 72.33 | 699 | Jaguars |
| 55 | 19 | Tyson Campbell | 70.70 | 63.82 | 75.33 | 767 | Jaguars |
| 56 | 20 | Kendall Fuller | 70.49 | 62.14 | 75.42 | 556 | Dolphins |
| 57 | 21 | Kris Abrams-Draine | 70.49 | 65.88 | 85.66 | 123 | Broncos |
| 58 | 22 | Shaquill Griffin | 70.30 | 61.46 | 76.44 | 597 | Vikings |
| 59 | 23 | Darious Williams | 70.29 | 59.80 | 74.58 | 865 | Rams |
| 60 | 24 | Amik Robertson | 70.24 | 61.92 | 72.50 | 630 | Lions |
| 61 | 25 | Adoree' Jackson | 70.10 | 63.74 | 73.91 | 426 | Giants |
| 62 | 26 | Jonathan Jones | 70.08 | 61.05 | 73.02 | 712 | Patriots |
| 63 | 27 | Stephon Gilmore | 70.00 | 59.20 | 73.71 | 904 | Vikings |
| 64 | 28 | Paulson Adebo | 69.71 | 63.26 | 76.11 | 436 | Saints |
| 65 | 29 | Cobie Durant | 69.68 | 61.50 | 73.42 | 843 | Rams |
| 66 | 30 | Tyrique Stevenson | 69.42 | 58.41 | 73.57 | 810 | Bears |
| 67 | 31 | Ahkello Witherspoon | 69.26 | 62.37 | 73.22 | 598 | Rams |
| 68 | 32 | Charvarius Ward | 68.82 | 58.29 | 74.12 | 694 | 49ers |
| 69 | 33 | Chamarri Conner | 68.77 | 62.54 | 68.75 | 679 | Chiefs |
| 70 | 34 | Darrell Baker Jr. | 68.74 | 62.44 | 72.82 | 626 | Titans |
| 71 | 35 | Josh Newton | 68.66 | 60.70 | 72.74 | 504 | Bengals |
| 72 | 36 | Beanie Bishop Jr. | 68.57 | 57.63 | 71.70 | 550 | Steelers |
| 73 | 37 | Keisean Nixon | 68.53 | 60.70 | 70.36 | 1077 | Packers |
| 74 | 38 | Isaiah Bolden | 68.31 | 68.19 | 71.86 | 141 | Patriots |
| 75 | 39 | Amani Oruwariye | 68.03 | 62.30 | 73.86 | 286 | Cowboys |
| 76 | 40 | Ja'Quan McMillian | 67.99 | 63.00 | 70.89 | 918 | Broncos |
| 77 | 41 | Alex Austin | 67.63 | 62.08 | 77.58 | 234 | Patriots |
| 78 | 42 | Israel Mukuamu | 67.27 | 56.00 | 71.10 | 201 | Cowboys |
| 79 | 43 | Cory Trice Jr. | 67.22 | 67.42 | 76.70 | 194 | Steelers |
| 80 | 44 | Trevon Diggs | 67.20 | 60.27 | 75.01 | 683 | Cowboys |
| 81 | 45 | Jakorian Bennett | 67.17 | 60.75 | 73.41 | 459 | Raiders |
| 82 | 46 | Ronald Darby | 67.06 | 59.34 | 73.52 | 659 | Jaguars |
| 83 | 47 | Cor'Dale Flott | 67.01 | 61.53 | 70.23 | 666 | Giants |
| 84 | 48 | Jaycee Horn | 66.97 | 57.90 | 73.85 | 1034 | Panthers |
| 85 | 49 | Avonte Maddox | 66.95 | 60.32 | 72.30 | 352 | Eagles |
| 86 | 50 | Cam Hart | 66.79 | 58.52 | 73.03 | 502 | Chargers |
| 87 | 51 | Joey Porter Jr. | 66.74 | 56.30 | 69.53 | 1038 | Steelers |
| 88 | 52 | Marshon Lattimore | 66.52 | 58.10 | 74.43 | 687 | Commanders |
| 89 | 53 | Ja'Sir Taylor | 66.40 | 58.50 | 71.92 | 353 | Chargers |
| 90 | 54 | Fabian Moreau | 66.39 | 60.20 | 73.41 | 104 | Vikings |
| 91 | 55 | Eric Stokes | 66.26 | 62.03 | 70.90 | 588 | Packers |
| 92 | 56 | Myles Bryant | 66.03 | 61.17 | 70.00 | 156 | Texans |
| 93 | 57 | Chidobe Awuzie | 65.66 | 58.71 | 72.89 | 373 | Titans |
| 94 | 58 | Taron Johnson | 65.64 | 55.96 | 69.11 | 785 | Bills |
| 95 | 59 | Jarvis Brownlee Jr. | 65.61 | 55.90 | 67.92 | 911 | Titans |
| 96 | 60 | Kaiir Elam | 65.61 | 60.92 | 73.10 | 359 | Bills |
| 97 | 61 | Brandin Echols | 65.46 | 60.84 | 69.77 | 406 | Jets |
| 98 | 62 | Rasul Douglas | 65.42 | 51.60 | 70.47 | 997 | Bills |
| 99 | 63 | Asante Samuel Jr. | 65.42 | 58.99 | 71.91 | 234 | Chargers |
| 100 | 64 | Isaac Yiadom | 65.08 | 55.95 | 70.91 | 488 | 49ers |
| 101 | 65 | Christian Roland-Wallace | 65.03 | 60.54 | 71.71 | 197 | Chiefs |
| 102 | 66 | Max Melton | 64.78 | 57.77 | 65.29 | 565 | Cardinals |
| 103 | 67 | Roger McCreary | 64.73 | 58.51 | 66.28 | 652 | Titans |
| 104 | 68 | Nate Hobbs | 64.69 | 61.34 | 68.06 | 554 | Raiders |
| 105 | 69 | Dee Alford | 64.42 | 56.06 | 67.91 | 724 | Falcons |
| 106 | 70 | Greg Newsome II | 64.23 | 54.61 | 69.41 | 571 | Browns |
| 107 | 71 | Jack Jones | 64.20 | 52.90 | 70.11 | 1047 | Raiders |
| 108 | 72 | Cameron Mitchell | 63.99 | 56.85 | 66.30 | 371 | Browns |
| 109 | 73 | Tre'Davious White | 63.97 | 55.63 | 73.29 | 445 | Ravens |
| 110 | 74 | James Pierre | 63.93 | 57.40 | 70.99 | 207 | Steelers |
| 111 | 75 | Terrion Arnold | 63.51 | 50.20 | 68.21 | 1021 | Lions |
| 112 | 76 | D'Angelo Ross | 63.51 | 63.12 | 69.61 | 184 | Texans |
| 113 | 77 | Dax Hill | 63.45 | 64.51 | 70.42 | 262 | Bengals |
| 114 | 78 | Nazeeh Johnson | 63.03 | 54.06 | 64.84 | 547 | Chiefs |
| 115 | 79 | Deantre Prince | 62.96 | 62.00 | 73.21 | 101 | Jaguars |
| 116 | 80 | Tre Brown | 62.51 | 56.15 | 69.84 | 290 | Seahawks |
| 117 | 81 | Ka'dar Hollman | 62.35 | 58.80 | 69.47 | 116 | Texans |
| 118 | 82 | Deonte Banks | 62.15 | 50.52 | 68.30 | 788 | Giants |
| 119 | 83 | Starling Thomas V | 62.15 | 60.90 | 61.02 | 817 | Cardinals |

### Rotation/backup (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 120 | 1 | Riley Moss | 61.73 | 56.00 | 65.91 | 912 | Broncos |
| 121 | 2 | Montaric Brown | 61.72 | 61.60 | 62.54 | 855 | Jaguars |
| 122 | 3 | Sean Murphy-Bunting | 61.43 | 53.36 | 66.07 | 725 | Cardinals |
| 123 | 4 | Martin Emerson Jr. | 61.31 | 48.40 | 65.75 | 827 | Browns |
| 124 | 5 | Benjamin St-Juste | 61.10 | 46.50 | 67.94 | 859 | Commanders |
| 125 | 6 | Greg Stroman Jr. | 60.73 | 59.76 | 69.71 | 130 | Giants |
| 126 | 7 | Josh Jobe | 60.47 | 52.60 | 70.09 | 443 | Seahawks |
| 127 | 8 | Ja'Marcus Ingram | 60.44 | 54.20 | 69.06 | 217 | Bills |
| 128 | 9 | Marco Wilson | 60.39 | 51.17 | 64.92 | 242 | Bengals |
| 129 | 10 | Shemar Jean-Charles | 60.30 | 58.17 | 70.06 | 143 | Saints |
| 130 | 11 | Storm Duck | 59.77 | 56.57 | 62.63 | 359 | Dolphins |
| 131 | 12 | Josh Blackwell | 59.63 | 60.48 | 66.77 | 102 | Bears |
| 132 | 13 | Michael Carter II | 59.62 | 50.89 | 63.82 | 285 | Jets |
| 133 | 14 | Josh Wallace | 59.34 | 52.58 | 62.62 | 165 | Rams |
| 134 | 15 | Emmanuel Forbes | 59.27 | 51.15 | 67.75 | 160 | Rams |
| 135 | 16 | Donte Jackson | 59.15 | 45.10 | 66.70 | 832 | Steelers |
| 136 | 17 | Cameron Sutton | 59.07 | 48.82 | 65.37 | 273 | Steelers |
| 137 | 18 | Darnay Holmes | 58.93 | 53.69 | 65.22 | 298 | Raiders |
| 138 | 19 | Michael Davis | 58.47 | 47.59 | 66.65 | 139 | Commanders |
| 139 | 20 | Nick McCloud | 58.17 | 52.81 | 63.66 | 224 | 49ers |
| 140 | 21 | Dane Jackson | 57.69 | 46.20 | 66.38 | 282 | Panthers |
| 141 | 22 | Chau Smith-Wade | 57.51 | 53.72 | 62.74 | 301 | Panthers |
| 142 | 23 | Noah Igbinoghene | 56.84 | 49.60 | 63.87 | 971 | Commanders |
| 143 | 24 | Alontae Taylor | 56.32 | 40.00 | 64.22 | 1075 | Saints |
| 144 | 25 | Decamerion Richardson | 55.38 | 45.35 | 63.78 | 559 | Raiders |
| 145 | 26 | L'Jarius Sneed | 54.76 | 42.31 | 64.77 | 301 | Titans |
| 146 | 27 | Caleb Farley | 54.75 | 58.60 | 55.77 | 169 | Panthers |
| 147 | 28 | Kindle Vildor | 53.11 | 50.72 | 55.54 | 316 | Lions |
| 148 | 29 | Andrew Booth Jr. | 52.76 | 47.48 | 63.00 | 118 | Cowboys |
| 149 | 30 | Cam Smith | 50.80 | 47.99 | 59.65 | 133 | Dolphins |
| 150 | 31 | Tyrek Funderburk | 50.52 | 58.06 | 52.63 | 168 | Buccaneers |
| 151 | 32 | Nehemiah Pritchett | 48.92 | 49.56 | 60.59 | 151 | Seahawks |
| 152 | 33 | Caelen Carson | 46.87 | 47.12 | 58.80 | 252 | Cowboys |

## DI — Defensive Interior

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Quinnen Williams | 88.92 | 86.92 | 86.77 | 722 | Jets |
| 2 | 2 | Kobie Turner | 87.16 | 84.01 | 85.09 | 919 | Rams |
| 3 | 3 | Dexter Lawrence | 83.98 | 86.58 | 81.01 | 551 | Giants |
| 4 | 4 | Jeffery Simmons | 83.90 | 86.06 | 80.65 | 806 | Titans |
| 5 | 5 | Chris Jones | 83.68 | 85.69 | 78.17 | 886 | Chiefs |
| 6 | 6 | DeForest Buckner | 83.67 | 83.04 | 82.37 | 579 | Colts |
| 7 | 7 | Leonard Williams | 83.61 | 84.74 | 80.16 | 750 | Seahawks |
| 8 | 8 | Zach Sieler | 82.93 | 77.67 | 83.26 | 749 | Dolphins |
| 9 | 9 | Jalen Carter | 82.89 | 86.10 | 76.58 | 1026 | Eagles |
| 10 | 10 | Cameron Heyward | 82.48 | 77.13 | 83.35 | 838 | Steelers |
| 11 | 11 | Vita Vea | 81.59 | 81.61 | 78.00 | 756 | Buccaneers |

### Good (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Ed Oliver | 79.28 | 70.73 | 81.60 | 727 | Bills |
| 13 | 2 | Jalen Redmond | 78.24 | 68.30 | 83.63 | 236 | Vikings |
| 14 | 3 | Christian Wilkins | 77.61 | 75.95 | 80.43 | 246 | Raiders |
| 15 | 4 | Grover Stewart | 76.94 | 73.74 | 76.68 | 690 | Colts |
| 16 | 5 | Alim McNeill | 76.86 | 79.50 | 72.70 | 631 | Lions |
| 17 | 6 | Milton Williams | 76.80 | 66.24 | 79.68 | 628 | Eagles |
| 18 | 7 | Zach Allen | 76.72 | 64.32 | 81.61 | 1031 | Broncos |
| 19 | 8 | Devonte Wyatt | 76.27 | 63.63 | 81.71 | 366 | Packers |
| 20 | 9 | T'Vondre Sweat | 75.83 | 77.85 | 70.32 | 699 | Titans |
| 21 | 10 | Kenny Clark | 75.81 | 67.66 | 77.08 | 725 | Packers |
| 22 | 11 | Braden Fiske | 75.60 | 56.54 | 84.14 | 700 | Rams |
| 23 | 12 | Osa Odighizuwa | 75.47 | 65.39 | 78.03 | 859 | Cowboys |
| 24 | 13 | Calais Campbell | 75.45 | 57.02 | 84.83 | 616 | Dolphins |
| 25 | 14 | Grady Jarrett | 74.99 | 65.63 | 79.72 | 744 | Falcons |
| 26 | 15 | Jordan Davis | 74.95 | 71.70 | 73.74 | 430 | Eagles |
| 27 | 16 | Michael Pierce | 74.91 | 70.34 | 78.49 | 254 | Ravens |
| 28 | 17 | B.J. Hill | 74.37 | 68.67 | 75.19 | 710 | Bengals |
| 29 | 18 | John Franklin-Myers | 74.34 | 61.73 | 78.96 | 569 | Broncos |
| 30 | 19 | Jonathan Allen | 74.31 | 61.15 | 82.35 | 421 | Commanders |

### Starter (95 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Sebastian Joseph-Day | 73.40 | 60.17 | 78.25 | 483 | Titans |
| 32 | 2 | Daron Payne | 73.09 | 59.09 | 78.26 | 796 | Commanders |
| 33 | 3 | D.J. Jones | 72.95 | 60.52 | 77.76 | 510 | Broncos |
| 34 | 4 | David Onyemata | 72.93 | 62.31 | 76.72 | 567 | Falcons |
| 35 | 5 | Calijah Kancey | 72.82 | 53.58 | 84.30 | 595 | Buccaneers |
| 36 | 6 | Christian Barmore | 72.70 | 63.35 | 82.51 | 123 | Patriots |
| 37 | 7 | Shelby Harris | 72.54 | 58.00 | 79.94 | 527 | Browns |
| 38 | 8 | Jaquelin Roy | 72.47 | 60.70 | 84.74 | 141 | Patriots |
| 39 | 9 | Travis Jones | 72.07 | 66.13 | 72.26 | 675 | Ravens |
| 40 | 10 | DJ Reader | 71.97 | 66.48 | 74.22 | 566 | Lions |
| 41 | 11 | Desjuan Johnson | 71.81 | 59.73 | 80.59 | 155 | Rams |
| 42 | 12 | Poona Ford | 71.80 | 64.75 | 74.98 | 652 | Chargers |
| 43 | 13 | Teair Tart | 71.66 | 64.14 | 73.88 | 378 | Chargers |
| 44 | 14 | Gervon Dexter Sr. | 71.29 | 63.40 | 73.62 | 616 | Bears |
| 45 | 15 | Dalvin Tomlinson | 71.29 | 61.97 | 74.61 | 609 | Browns |
| 46 | 16 | Leonard Taylor III | 71.13 | 59.70 | 77.52 | 261 | Jets |
| 47 | 17 | Karl Brooks | 70.95 | 56.31 | 76.55 | 459 | Packers |
| 48 | 18 | Keeanu Benton | 70.57 | 68.48 | 67.80 | 671 | Steelers |
| 49 | 19 | Javon Hargrave | 70.36 | 58.86 | 80.73 | 104 | 49ers |
| 50 | 20 | Jarran Reed | 69.67 | 56.36 | 74.67 | 679 | Seahawks |
| 51 | 21 | Levi Onwuzurike | 69.57 | 64.87 | 69.90 | 697 | Lions |
| 52 | 22 | Malcolm Roach | 69.38 | 57.00 | 75.72 | 524 | Broncos |
| 53 | 23 | William Gholston | 69.23 | 55.28 | 74.36 | 205 | Buccaneers |
| 54 | 24 | Byron Murphy II | 68.94 | 58.70 | 74.53 | 457 | Seahawks |
| 55 | 25 | Mario Edwards Jr. | 68.81 | 54.39 | 76.81 | 519 | Texans |
| 56 | 26 | Tim Settle | 68.78 | 52.83 | 75.65 | 685 | Texans |
| 57 | 27 | Naquan Jones | 68.32 | 55.15 | 78.03 | 260 | Cardinals |
| 58 | 28 | Jer'Zhan Newton | 68.32 | 51.95 | 75.07 | 586 | Commanders |
| 59 | 29 | Tyler Davis | 68.29 | 51.77 | 75.14 | 354 | Rams |
| 60 | 30 | Dante Stills | 68.08 | 55.58 | 73.59 | 532 | Cardinals |
| 61 | 31 | Roy Robertson-Harris | 67.54 | 52.35 | 73.50 | 398 | Seahawks |
| 62 | 32 | Elijah Garcia | 67.53 | 59.23 | 81.40 | 143 | Giants |
| 63 | 33 | Larry Ogunjobi | 67.46 | 48.16 | 76.84 | 550 | Steelers |
| 64 | 34 | Jowon Briggs | 67.46 | 64.70 | 78.92 | 133 | Browns |
| 65 | 35 | Folorunso Fatukasi | 67.43 | 50.22 | 77.58 | 366 | Texans |
| 66 | 36 | Thomas Booker IV | 67.35 | 54.05 | 74.62 | 172 | Eagles |
| 67 | 37 | Evan Anderson | 67.35 | 57.72 | 74.51 | 267 | 49ers |
| 68 | 38 | Zach Harrison | 67.34 | 56.56 | 70.36 | 272 | Falcons |
| 69 | 39 | Kevin Givens | 67.19 | 54.35 | 76.79 | 185 | 49ers |
| 70 | 40 | Roy Lopez | 66.97 | 54.21 | 72.67 | 464 | Cardinals |
| 71 | 41 | DaQuan Jones | 66.87 | 59.98 | 69.84 | 629 | Bills |
| 72 | 42 | Khalil Davis | 66.72 | 53.06 | 77.03 | 209 | 49ers |
| 73 | 43 | Jeremiah Pharms Jr. | 66.66 | 56.69 | 71.59 | 457 | Patriots |
| 74 | 44 | DaVon Hamilton | 66.48 | 51.39 | 75.02 | 626 | Jaguars |
| 75 | 45 | Bobby Brown III | 66.39 | 59.60 | 69.20 | 513 | Rams |
| 76 | 46 | Morgan Fox | 66.34 | 47.64 | 74.64 | 619 | Chargers |
| 77 | 47 | A'Shawn Robinson | 66.31 | 49.20 | 75.42 | 761 | Panthers |
| 78 | 48 | Harrison Phillips | 66.11 | 53.21 | 70.55 | 701 | Vikings |
| 79 | 49 | Kentavius Street | 65.95 | 53.29 | 73.35 | 280 | Falcons |
| 80 | 50 | Javon Kinlaw | 65.93 | 53.54 | 72.18 | 695 | Jets |
| 81 | 51 | Solomon Thomas | 65.93 | 48.78 | 73.68 | 458 | Jets |
| 82 | 52 | Khyiris Tonga | 65.74 | 58.12 | 70.97 | 229 | Cardinals |
| 83 | 53 | Neville Gallimore | 65.24 | 51.55 | 70.89 | 308 | Rams |
| 84 | 54 | Eddie Goldman | 65.17 | 48.30 | 73.34 | 330 | Falcons |
| 85 | 55 | Byron Cowart | 65.11 | 49.52 | 72.73 | 335 | Bears |
| 86 | 56 | Taven Bryan | 65.05 | 53.65 | 68.68 | 340 | Colts |
| 87 | 57 | Mike Pennel | 65.01 | 50.23 | 73.63 | 365 | Chiefs |
| 88 | 58 | Andrew Billings | 64.76 | 56.90 | 70.84 | 297 | Bears |
| 89 | 59 | Bryan Bresee | 64.66 | 49.37 | 70.68 | 708 | Saints |
| 90 | 60 | Maurice Hurst | 64.57 | 53.28 | 76.47 | 164 | Browns |
| 91 | 61 | James Lynch | 64.56 | 51.85 | 70.23 | 243 | Titans |
| 92 | 62 | Logan Hall | 64.50 | 54.90 | 66.73 | 571 | Buccaneers |
| 93 | 63 | Maliek Collins | 64.46 | 50.45 | 70.03 | 715 | 49ers |
| 94 | 64 | Quinton Jefferson | 64.37 | 47.22 | 73.98 | 258 | Bills |
| 95 | 65 | Jeremiah Ledbetter | 64.34 | 53.22 | 71.70 | 441 | Jaguars |
| 96 | 66 | Da'Shawn Hand | 64.20 | 55.71 | 69.12 | 564 | Dolphins |
| 97 | 67 | Jonathan Bullard | 64.19 | 47.55 | 72.10 | 590 | Vikings |
| 98 | 68 | Colby Wooden | 64.16 | 54.30 | 69.01 | 260 | Packers |
| 99 | 69 | Sheldon Rankins | 64.12 | 52.14 | 73.24 | 287 | Bengals |
| 100 | 70 | Khalen Saunders | 64.10 | 51.53 | 70.46 | 460 | Saints |
| 101 | 71 | Adam Butler | 64.09 | 45.66 | 72.21 | 856 | Raiders |
| 102 | 72 | McKinnley Jackson | 64.06 | 53.10 | 71.12 | 248 | Bengals |
| 103 | 73 | Mazi Smith | 64.05 | 50.32 | 69.03 | 524 | Cowboys |
| 104 | 74 | Greg Gaines | 64.04 | 52.82 | 68.04 | 421 | Buccaneers |
| 105 | 75 | Adetomiwa Adebawore | 64.02 | 57.00 | 71.15 | 137 | Colts |
| 106 | 76 | Patrick O'Connor | 63.93 | 50.91 | 74.13 | 235 | Lions |
| 107 | 77 | Zacch Pickens | 63.92 | 53.55 | 71.56 | 228 | Bears |
| 108 | 78 | Davon Godchaux | 63.86 | 47.31 | 70.73 | 680 | Patriots |
| 109 | 79 | Jonah Laulu | 63.82 | 48.57 | 71.79 | 474 | Raiders |
| 110 | 80 | Jordan Phillips | 63.44 | 50.06 | 72.51 | 185 | Bills |
| 111 | 81 | Tommy Togiai | 63.42 | 52.19 | 73.80 | 280 | Texans |
| 112 | 82 | Linval Joseph | 63.41 | 43.78 | 76.45 | 264 | Cowboys |
| 113 | 83 | Austin Johnson | 63.30 | 50.05 | 69.74 | 353 | Bills |
| 114 | 84 | Jalyn Holmes | 63.22 | 50.20 | 74.40 | 337 | Commanders |
| 115 | 85 | Justin Jones | 63.15 | 53.01 | 72.61 | 100 | Cardinals |
| 116 | 86 | Nathan Shepherd | 63.10 | 47.10 | 70.09 | 567 | Saints |
| 117 | 87 | Broderick Washington | 62.96 | 49.86 | 67.53 | 488 | Ravens |
| 118 | 88 | Jonah Williams | 62.82 | 53.07 | 69.37 | 108 | Lions |
| 119 | 89 | Tershawn Wharton | 62.75 | 50.76 | 68.92 | 733 | Chiefs |
| 120 | 90 | Bilal Nichols | 62.38 | 49.81 | 72.00 | 173 | Cardinals |
| 121 | 91 | D.J. Davidson | 62.25 | 53.53 | 69.30 | 261 | Giants |
| 122 | 92 | Johnathan Hankins | 62.25 | 42.96 | 72.91 | 389 | Seahawks |
| 123 | 93 | DeShawn Williams | 62.20 | 47.63 | 70.50 | 338 | Panthers |
| 124 | 94 | Montravius Adams | 62.12 | 50.98 | 68.72 | 207 | Steelers |
| 125 | 95 | John Jenkins | 62.09 | 42.21 | 71.38 | 606 | Raiders |

### Rotation/backup (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 126 | 1 | Isaiahh Loudermilk | 61.97 | 54.28 | 64.59 | 255 | Steelers |
| 127 | 2 | Carlos Watkins | 61.72 | 52.90 | 69.32 | 228 | Cowboys |
| 128 | 3 | Jerry Tillery | 61.66 | 51.48 | 64.68 | 482 | Vikings |
| 129 | 4 | Elijah Chatman | 61.61 | 54.25 | 62.35 | 423 | Giants |
| 130 | 5 | John Ridgeway | 61.51 | 51.22 | 66.56 | 263 | Saints |
| 131 | 6 | Moro Ojomo | 61.51 | 60.87 | 61.09 | 465 | Eagles |
| 132 | 7 | Bruce Hector | 61.48 | 55.88 | 73.54 | 118 | Jets |
| 133 | 8 | Ta'Quon Graham | 61.22 | 58.14 | 64.31 | 193 | Falcons |
| 134 | 9 | Raekwon Davis | 61.03 | 50.78 | 63.89 | 349 | Colts |
| 135 | 10 | Kurt Hinish | 60.97 | 52.98 | 65.94 | 231 | Texans |
| 136 | 11 | Shy Tuttle | 60.92 | 46.45 | 67.39 | 610 | Panthers |
| 137 | 12 | Derrick Nnadi | 60.71 | 49.67 | 63.91 | 248 | Chiefs |
| 138 | 13 | Dean Lowry | 60.39 | 47.68 | 69.89 | 159 | Steelers |
| 139 | 14 | Jordan Jefferson | 60.36 | 60.07 | 65.20 | 151 | Jaguars |
| 140 | 15 | C.J. Brewer | 60.32 | 51.15 | 67.17 | 159 | Buccaneers |
| 141 | 16 | Benito Jones | 60.29 | 48.15 | 64.21 | 481 | Dolphins |
| 142 | 17 | Maason Smith | 60.06 | 50.18 | 68.36 | 384 | Jaguars |
| 143 | 18 | Tyler Lacy | 59.61 | 51.69 | 63.90 | 340 | Jaguars |
| 144 | 19 | Sheldon Day | 59.42 | 48.83 | 68.49 | 339 | Commanders |
| 145 | 20 | Ruke Orhorhoro | 59.41 | 56.38 | 66.08 | 147 | Falcons |
| 146 | 21 | Zach Carter | 58.87 | 51.49 | 62.28 | 263 | Raiders |
| 147 | 22 | Jordan Elliott | 58.36 | 49.20 | 61.29 | 440 | 49ers |
| 148 | 23 | Jay Tufele | 58.11 | 50.82 | 64.79 | 242 | Bengals |
| 149 | 24 | Jordon Riley | 58.07 | 51.09 | 64.92 | 248 | Giants |
| 150 | 25 | Rakeem Nunez-Roches | 57.74 | 42.18 | 65.22 | 608 | Giants |
| 151 | 26 | Daniel Ekuale | 57.72 | 47.69 | 65.24 | 723 | Patriots |
| 152 | 27 | Ben Stille | 57.52 | 53.77 | 66.73 | 120 | Cardinals |
| 153 | 28 | DeWayne Carter | 57.28 | 48.84 | 64.62 | 315 | Bills |
| 154 | 29 | L.J. Collier | 57.19 | 45.72 | 67.13 | 588 | Cardinals |
| 155 | 30 | Jordan Jackson | 56.82 | 46.06 | 59.83 | 329 | Broncos |
| 156 | 31 | Jaden Crumedy | 56.81 | 54.18 | 70.66 | 121 | Panthers |
| 157 | 32 | Chris Williams | 55.82 | 46.47 | 63.19 | 367 | Bears |
| 158 | 33 | Darius Robinson | 55.76 | 55.18 | 65.77 | 184 | Cardinals |
| 159 | 34 | Eric Johnson | 55.37 | 52.89 | 57.28 | 178 | Patriots |
| 160 | 35 | Otito Ogbonnia | 55.20 | 46.87 | 61.18 | 538 | Chargers |
| 161 | 36 | Kris Jenkins | 54.61 | 49.90 | 60.66 | 496 | Bengals |
| 162 | 37 | Phidarian Mathis | 53.77 | 51.38 | 58.84 | 257 | Jets |
| 163 | 38 | LaBryan Ray | 52.86 | 43.49 | 55.56 | 626 | Panthers |
| 164 | 39 | Keondre Coburn | 52.59 | 55.48 | 52.13 | 125 | Titans |
| 165 | 40 | Matthew Butler | 51.31 | 55.88 | 56.07 | 101 | Raiders |
| 166 | 41 | Kalia Davis | 50.82 | 49.73 | 54.98 | 259 | 49ers |
| 167 | 42 | Mekhi Wingo | 50.40 | 52.65 | 50.62 | 177 | Lions |

## ED — Edge

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Bosa | 92.57 | 95.17 | 88.33 | 693 | 49ers |
| 2 | 2 | Myles Garrett | 92.08 | 94.59 | 86.44 | 822 | Browns |
| 3 | 3 | Will Anderson Jr. | 91.38 | 93.89 | 86.15 | 645 | Texans |
| 4 | 4 | Micah Parsons | 91.30 | 89.10 | 90.57 | 694 | Cowboys |
| 5 | 5 | T.J. Watt | 90.39 | 92.51 | 86.17 | 1002 | Steelers |
| 6 | 6 | Jared Verse | 88.53 | 94.25 | 80.55 | 933 | Rams |
| 7 | 7 | Greg Rousseau | 86.65 | 89.64 | 81.27 | 861 | Bills |
| 8 | 8 | Khalil Mack | 86.38 | 83.39 | 84.20 | 668 | Chargers |
| 9 | 9 | Trey Hendrickson | 85.71 | 80.89 | 85.15 | 823 | Bengals |
| 10 | 10 | Rashan Gary | 85.42 | 83.05 | 84.40 | 670 | Packers |
| 11 | 11 | Aidan Hutchinson | 84.82 | 86.17 | 85.64 | 280 | Lions |
| 12 | 12 | Joey Bosa | 84.47 | 84.38 | 86.04 | 503 | Chargers |
| 13 | 13 | Danielle Hunter | 83.36 | 76.50 | 83.77 | 859 | Texans |
| 14 | 14 | Maxx Crosby | 82.94 | 85.96 | 79.21 | 766 | Raiders |
| 15 | 15 | Nik Bonitto | 81.23 | 67.07 | 86.50 | 761 | Broncos |
| 16 | 16 | Montez Sweat | 80.93 | 78.22 | 79.05 | 616 | Bears |
| 17 | 17 | Yaya Diaby | 80.14 | 78.37 | 77.16 | 841 | Buccaneers |

### Good (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Alex Highsmith | 79.69 | 81.13 | 77.01 | 592 | Steelers |
| 19 | 2 | Von Miller | 79.44 | 65.27 | 87.27 | 332 | Bills |
| 20 | 3 | Brian Burns | 79.19 | 71.75 | 80.46 | 865 | Giants |
| 21 | 4 | Chop Robinson | 78.89 | 69.60 | 80.92 | 565 | Dolphins |
| 22 | 5 | Jonathan Greenard | 77.71 | 69.59 | 80.72 | 969 | Vikings |
| 23 | 6 | Nolan Smith | 77.51 | 70.46 | 78.05 | 725 | Eagles |
| 24 | 7 | Javon Solomon | 76.55 | 64.20 | 84.54 | 141 | Bills |
| 25 | 8 | Dondrea Tillman | 76.45 | 63.13 | 85.08 | 275 | Broncos |
| 26 | 9 | Brenton Cox Jr. | 76.23 | 64.18 | 88.91 | 187 | Packers |
| 27 | 10 | Zaven Collins | 76.20 | 71.75 | 75.00 | 600 | Cardinals |
| 28 | 11 | Will McDonald IV | 76.14 | 64.10 | 80.74 | 756 | Jets |
| 29 | 12 | Brandon Graham | 75.98 | 67.49 | 80.40 | 311 | Eagles |
| 30 | 13 | Tuli Tuipulotu | 75.88 | 72.19 | 74.17 | 774 | Chargers |
| 31 | 14 | Za'Darius Smith | 75.75 | 65.70 | 78.48 | 655 | Lions |
| 32 | 15 | Odafe Oweh | 75.75 | 75.75 | 72.17 | 683 | Ravens |
| 33 | 16 | Kyle Van Noy | 75.55 | 58.30 | 82.89 | 696 | Ravens |
| 34 | 17 | George Karlaftis | 74.95 | 65.98 | 76.77 | 953 | Chiefs |
| 35 | 18 | DeMarcus Lawrence | 74.83 | 74.40 | 77.31 | 167 | Cowboys |
| 36 | 19 | Nick Herbig | 74.78 | 65.29 | 79.87 | 433 | Steelers |
| 37 | 20 | Boye Mafe | 74.60 | 69.94 | 74.81 | 607 | Seahawks |
| 38 | 21 | Jonathon Cooper | 74.21 | 65.75 | 76.26 | 882 | Broncos |

### Starter (62 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | James Houston | 73.72 | 58.40 | 88.79 | 141 | Browns |
| 40 | 2 | Andrew Van Ginkel | 73.56 | 57.64 | 80.31 | 973 | Vikings |
| 41 | 3 | Josh Sweat | 73.30 | 65.45 | 74.56 | 775 | Eagles |
| 42 | 4 | Travon Walker | 73.09 | 70.93 | 70.76 | 911 | Jaguars |
| 43 | 5 | Jaelan Phillips | 72.79 | 68.15 | 80.73 | 134 | Dolphins |
| 44 | 6 | Carl Granderson | 72.79 | 65.93 | 73.40 | 825 | Saints |
| 45 | 7 | Chase Young | 72.70 | 74.45 | 70.12 | 740 | Saints |
| 46 | 8 | Arnold Ebiketie | 72.36 | 62.88 | 74.71 | 543 | Falcons |
| 47 | 9 | Michael Hoecht | 72.20 | 60.69 | 75.70 | 705 | Rams |
| 48 | 10 | Isaiah McGuire | 71.75 | 67.59 | 75.76 | 469 | Browns |
| 49 | 11 | Byron Young | 71.75 | 58.98 | 76.10 | 936 | Rams |
| 50 | 12 | Haason Reddick | 70.98 | 54.70 | 81.10 | 392 | Jets |
| 51 | 13 | Harold Landry III | 70.35 | 59.61 | 73.34 | 878 | Titans |
| 52 | 14 | Dorance Armstrong | 70.32 | 60.31 | 72.82 | 747 | Commanders |
| 53 | 15 | Jadeveon Clowney | 70.19 | 65.96 | 71.29 | 650 | Panthers |
| 54 | 16 | Darrell Taylor | 69.96 | 55.43 | 76.16 | 374 | Bears |
| 55 | 17 | Dennis Gardeck | 69.96 | 54.78 | 81.39 | 206 | Cardinals |
| 56 | 18 | Cameron Jordan | 69.79 | 59.38 | 72.76 | 565 | Saints |
| 57 | 19 | Uchenna Nwosu | 69.68 | 62.22 | 75.89 | 190 | Seahawks |
| 58 | 20 | Matthew Judon | 69.52 | 50.11 | 82.11 | 655 | Falcons |
| 59 | 21 | Dante Fowler Jr. | 69.26 | 54.09 | 75.21 | 642 | Commanders |
| 60 | 22 | Kayvon Thibodeaux | 68.75 | 67.37 | 68.54 | 593 | Giants |
| 61 | 23 | Jonah Elliss | 68.33 | 60.23 | 69.56 | 441 | Broncos |
| 62 | 24 | Azeez Ojulari | 68.20 | 59.10 | 76.76 | 391 | Giants |
| 63 | 25 | Leonard Floyd | 68.02 | 52.51 | 74.20 | 604 | 49ers |
| 64 | 26 | Bryce Huff | 67.71 | 62.25 | 71.10 | 298 | Eagles |
| 65 | 27 | Preston Smith | 67.53 | 53.67 | 72.60 | 469 | Steelers |
| 66 | 28 | Kwity Paye | 67.49 | 64.40 | 67.63 | 667 | Colts |
| 67 | 29 | Chris Braswell | 67.49 | 60.45 | 68.01 | 335 | Buccaneers |
| 68 | 30 | Ogbo Okoronkwo | 67.42 | 54.33 | 73.06 | 464 | Browns |
| 69 | 31 | Charles Snowden | 67.28 | 54.07 | 72.40 | 405 | Raiders |
| 70 | 32 | Lukas Van Ness | 67.27 | 59.77 | 68.11 | 458 | Packers |
| 71 | 33 | Derick Hall | 67.12 | 57.52 | 69.35 | 673 | Seahawks |
| 72 | 34 | Victor Dimukeje | 66.95 | 60.12 | 71.16 | 157 | Cardinals |
| 73 | 35 | Jacob Martin | 66.76 | 57.71 | 72.36 | 222 | Bears |
| 74 | 36 | Tyree Wilson | 66.69 | 65.45 | 63.97 | 524 | Raiders |
| 75 | 37 | Jamin Davis | 66.58 | 57.15 | 73.11 | 107 | Jets |
| 76 | 38 | Dayo Odeyingbo | 66.32 | 60.74 | 65.88 | 746 | Colts |
| 77 | 39 | Carl Lawson | 66.16 | 55.08 | 73.60 | 402 | Cowboys |
| 78 | 40 | A.J. Epenesa | 65.88 | 57.55 | 67.67 | 712 | Bills |
| 79 | 41 | Laiatu Latu | 65.66 | 64.49 | 62.27 | 618 | Colts |
| 80 | 42 | Tyquan Lewis | 65.03 | 58.73 | 70.47 | 355 | Colts |
| 81 | 43 | Kingsley Enagbare | 64.92 | 59.34 | 64.47 | 538 | Packers |
| 82 | 44 | Sam Hubbard | 64.75 | 55.01 | 69.52 | 521 | Bengals |
| 83 | 45 | Arik Armstead | 64.73 | 56.37 | 66.14 | 569 | Jaguars |
| 84 | 46 | Dallas Turner | 64.62 | 59.16 | 64.10 | 310 | Vikings |
| 85 | 47 | Julian Okwara | 64.57 | 56.49 | 71.48 | 286 | Cardinals |
| 86 | 48 | Arden Key | 64.49 | 59.05 | 64.43 | 734 | Titans |
| 87 | 49 | Xavier Thomas | 64.46 | 57.87 | 67.62 | 208 | Cardinals |
| 88 | 50 | Derek Barnett | 64.14 | 59.51 | 66.50 | 413 | Texans |
| 89 | 51 | Deatrich Wise Jr. | 63.95 | 55.22 | 66.87 | 409 | Patriots |
| 90 | 52 | Felix Anudike-Uzomah | 63.93 | 60.07 | 62.33 | 344 | Chiefs |
| 91 | 53 | Clelin Ferrell | 63.58 | 57.97 | 63.36 | 443 | Commanders |
| 92 | 54 | Keion White | 63.44 | 62.89 | 60.00 | 830 | Patriots |
| 93 | 55 | Joseph Ossai | 63.36 | 58.60 | 62.36 | 573 | Bengals |
| 94 | 56 | Baron Browning | 63.27 | 58.61 | 66.13 | 378 | Cardinals |
| 95 | 57 | Joe Tryon-Shoyinka | 62.96 | 57.02 | 63.24 | 570 | Buccaneers |
| 96 | 58 | Jalyx Hunt | 62.84 | 59.57 | 63.79 | 320 | Eagles |
| 97 | 59 | Emmanuel Ogbah | 62.80 | 54.49 | 66.83 | 734 | Dolphins |
| 98 | 60 | Arron Mosby | 62.53 | 59.09 | 67.52 | 150 | Packers |
| 99 | 61 | Anfernee Jennings | 62.22 | 57.21 | 62.38 | 831 | Patriots |
| 100 | 62 | Charles Omenihu | 62.09 | 55.22 | 67.81 | 303 | Chiefs |

### Rotation/backup (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 101 | 1 | Dre'Mont Jones | 61.94 | 50.65 | 65.30 | 617 | Seahawks |
| 102 | 2 | Austin Booker | 61.72 | 57.65 | 60.26 | 283 | Bears |
| 103 | 3 | Bud Dupree | 61.66 | 49.40 | 67.13 | 570 | Chargers |
| 104 | 4 | Mike Danna | 61.43 | 56.94 | 62.02 | 581 | Chiefs |
| 105 | 5 | K'Lavon Chaisson | 61.08 | 57.76 | 61.68 | 508 | Raiders |
| 106 | 6 | Anthony Nelson | 60.69 | 55.42 | 60.04 | 624 | Buccaneers |
| 107 | 7 | Payton Turner | 60.60 | 60.17 | 63.87 | 335 | Saints |
| 108 | 8 | Tyus Bowser | 60.52 | 49.90 | 68.92 | 276 | Dolphins |
| 109 | 9 | Janarius Robinson | 60.51 | 57.04 | 65.33 | 109 | Raiders |
| 110 | 10 | Lorenzo Carter | 60.40 | 53.62 | 62.72 | 410 | Falcons |
| 111 | 11 | Sam Okuayinonu | 60.31 | 57.64 | 62.58 | 451 | 49ers |
| 112 | 12 | D.J. Wonnum | 60.14 | 56.42 | 63.66 | 453 | Panthers |
| 113 | 13 | Myles Murphy | 59.45 | 58.35 | 58.47 | 353 | Bengals |
| 114 | 14 | Yetur Gross-Matos | 59.27 | 57.91 | 60.43 | 367 | 49ers |
| 115 | 15 | Micheal Clemons | 58.81 | 54.04 | 58.30 | 624 | Jets |
| 116 | 16 | DeMarcus Walker | 58.67 | 46.76 | 62.45 | 738 | Bears |
| 117 | 17 | Tavius Robinson | 58.41 | 55.14 | 56.42 | 548 | Ravens |
| 118 | 18 | Ali Gaye | 58.30 | 54.73 | 57.75 | 177 | Titans |
| 119 | 19 | Dawuane Smoot | 58.27 | 52.64 | 61.19 | 386 | Bills |
| 120 | 20 | Javontae Jean-Baptiste | 57.08 | 56.73 | 55.12 | 248 | Commanders |
| 121 | 21 | Charles Harris | 56.87 | 52.87 | 61.16 | 474 | Eagles |
| 122 | 22 | Casey Toohill | 56.63 | 53.62 | 56.44 | 249 | Bills |
| 123 | 23 | Josh Paschal | 56.58 | 57.32 | 54.86 | 613 | Lions |
| 124 | 24 | Quinton Bell | 55.85 | 54.02 | 55.94 | 258 | Dolphins |
| 125 | 25 | Cam Gill | 55.78 | 55.19 | 57.01 | 222 | Panthers |
| 126 | 26 | David Ojabo | 55.57 | 57.68 | 58.03 | 292 | Ravens |
| 127 | 27 | Al-Quadin Muhammad | 55.46 | 54.44 | 55.61 | 293 | Lions |
| 128 | 28 | Tyrus Wheat | 55.46 | 57.25 | 59.90 | 165 | Cowboys |
| 129 | 29 | Robert Beal Jr. | 55.45 | 57.55 | 55.76 | 149 | 49ers |
| 130 | 30 | Alex Wright | 54.98 | 58.73 | 54.68 | 103 | Browns |
| 131 | 31 | Eric Watts | 54.32 | 57.99 | 50.64 | 231 | Jets |
| 132 | 32 | DJ Johnson | 53.88 | 54.98 | 52.29 | 392 | Panthers |
| 133 | 33 | Dylan Horton | 53.74 | 57.18 | 51.08 | 217 | Texans |
| 134 | 34 | Malik Herring | 53.73 | 55.57 | 54.52 | 193 | Chiefs |
| 135 | 35 | Tomon Fox | 53.68 | 57.20 | 54.03 | 207 | Giants |
| 136 | 36 | Jaylen Harrell | 53.62 | 54.88 | 48.61 | 286 | Titans |
| 137 | 37 | Marshawn Kneeland | 53.14 | 57.18 | 52.17 | 255 | Cowboys |
| 138 | 38 | James Smith-Williams | 53.05 | 52.07 | 54.74 | 306 | Falcons |
| 139 | 39 | Brent Urban | 52.14 | 45.70 | 53.25 | 209 | Ravens |
| 140 | 40 | Jeremiah Moon | 49.14 | 55.50 | 47.88 | 117 | Steelers |
| 141 | 41 | Demone Harris | 48.71 | 52.89 | 48.43 | 216 | Falcons |
| 142 | 42 | Myles Cole | 48.03 | 56.12 | 47.29 | 135 | Jaguars |

## G — Guard

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Lindstrom | 95.90 | 93.50 | 93.34 | 1099 | Falcons |
| 2 | 2 | Quinn Meinerz | 93.00 | 86.90 | 92.90 | 1131 | Broncos |
| 3 | 3 | Kevin Zeitler | 92.17 | 86.80 | 91.59 | 1047 | Lions |
| 4 | 4 | Quenton Nelson | 88.69 | 81.30 | 89.45 | 1083 | Colts |
| 5 | 5 | Landon Dickerson | 88.53 | 82.30 | 88.52 | 1157 | Eagles |
| 6 | 6 | James Daniels | 88.25 | 75.15 | 92.81 | 209 | Steelers |
| 7 | 7 | Christian Mahogany | 87.83 | 72.60 | 93.81 | 144 | Lions |
| 8 | 8 | Will Fries | 86.14 | 74.03 | 90.05 | 268 | Colts |
| 9 | 9 | Joe Thuney | 85.94 | 80.20 | 85.60 | 1232 | Chiefs |
| 10 | 10 | John Simpson | 85.84 | 77.30 | 87.37 | 1020 | Jets |
| 11 | 11 | Jordan Meredith | 85.71 | 75.88 | 88.09 | 574 | Raiders |
| 12 | 12 | Kevin Dotson | 85.39 | 77.70 | 86.35 | 1145 | Rams |
| 13 | 13 | Alijah Vera-Tucker | 85.11 | 77.07 | 86.31 | 916 | Jets |
| 14 | 14 | Dominick Puni | 84.51 | 80.50 | 83.02 | 1078 | 49ers |
| 15 | 15 | Trey Smith | 83.91 | 75.30 | 85.49 | 1232 | Chiefs |
| 16 | 16 | Damien Lewis | 83.83 | 75.16 | 85.45 | 942 | Panthers |
| 17 | 17 | Tyler Smith | 83.51 | 75.00 | 85.02 | 1052 | Cowboys |
| 18 | 18 | Teven Jenkins | 83.09 | 74.16 | 84.88 | 738 | Bears |
| 19 | 19 | Cody Mauch | 82.10 | 74.60 | 82.94 | 1178 | Buccaneers |
| 20 | 20 | Dylan Parham | 81.94 | 73.53 | 83.38 | 882 | Raiders |
| 21 | 21 | Matthew Bergeron | 80.47 | 70.90 | 82.69 | 1106 | Falcons |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Mekhi Becton | 79.97 | 72.50 | 80.78 | 1097 | Eagles |
| 23 | 2 | Chandler Zavala | 78.93 | 65.02 | 84.03 | 198 | Panthers |
| 24 | 3 | Matt Pryor | 78.89 | 69.90 | 80.71 | 1005 | Bears |
| 25 | 4 | Jake Hanson | 78.05 | 63.96 | 83.28 | 103 | Jets |
| 26 | 5 | Sam Cosmi | 77.87 | 67.80 | 80.42 | 1259 | Commanders |
| 27 | 6 | Robert Hunt | 77.83 | 67.66 | 80.44 | 966 | Panthers |
| 28 | 7 | Jackson Powers-Johnson | 77.33 | 63.84 | 82.16 | 956 | Raiders |
| 29 | 8 | Cesar Ruiz | 77.03 | 66.90 | 79.61 | 813 | Saints |
| 30 | 9 | Isaac Seumalo | 76.94 | 66.08 | 80.02 | 872 | Steelers |
| 31 | 10 | Will Hernandez | 76.65 | 64.96 | 80.28 | 280 | Cardinals |
| 32 | 11 | Elgton Jenkins | 76.41 | 65.50 | 79.52 | 1073 | Packers |
| 33 | 12 | Jack Driscoll | 76.28 | 64.52 | 79.95 | 110 | Eagles |
| 34 | 13 | Aaron Banks | 76.27 | 64.79 | 79.76 | 775 | 49ers |
| 35 | 14 | David Edwards | 76.09 | 66.10 | 78.58 | 2360 | Bills |
| 36 | 15 | Jonah Jackson | 76.04 | 63.96 | 79.92 | 267 | Rams |
| 37 | 16 | T.J. Bass | 75.17 | 61.70 | 79.99 | 315 | Cowboys |
| 38 | 17 | Dalton Risner | 75.00 | 66.38 | 76.58 | 611 | Vikings |
| 39 | 18 | Ben Powers | 74.77 | 64.40 | 77.51 | 1130 | Broncos |
| 40 | 19 | Brandon Scherff | 74.68 | 64.70 | 77.16 | 1013 | Jaguars |
| 41 | 20 | Ezra Cleveland | 74.46 | 64.79 | 76.74 | 911 | Jaguars |
| 42 | 21 | Zack Martin | 74.44 | 64.94 | 76.60 | 638 | Cowboys |
| 43 | 22 | Zion Johnson | 74.31 | 64.40 | 76.75 | 1102 | Chargers |
| 44 | 23 | Joel Bitonio | 74.15 | 63.90 | 76.82 | 1178 | Browns |

### Starter (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 45 | 1 | Greg Van Roten | 73.96 | 63.40 | 76.84 | 1121 | Giants |
| 46 | 2 | Wyatt Teller | 73.57 | 62.52 | 76.77 | 885 | Browns |
| 47 | 3 | Evan Brown | 72.79 | 65.90 | 73.22 | 1070 | Cardinals |
| 48 | 4 | Laken Tomlinson | 72.59 | 62.10 | 75.41 | 1094 | Seahawks |
| 49 | 5 | Sean Rhyan | 72.10 | 61.30 | 75.13 | 1027 | Packers |
| 50 | 6 | Cordell Volson | 71.87 | 59.30 | 76.08 | 984 | Bengals |
| 51 | 7 | Nick Allegretti | 71.46 | 59.40 | 75.34 | 1372 | Commanders |
| 52 | 8 | Peter Skoronski | 71.43 | 60.30 | 74.69 | 1095 | Titans |
| 53 | 9 | Trey Pipkins III | 70.86 | 57.97 | 75.29 | 838 | Chargers |
| 54 | 10 | Shaq Mason | 70.83 | 60.50 | 73.55 | 999 | Texans |
| 55 | 11 | Spencer Anderson | 69.83 | 58.01 | 73.55 | 357 | Steelers |
| 56 | 12 | Spencer Burford | 69.62 | 59.04 | 72.50 | 113 | 49ers |
| 57 | 13 | Mason McCormick | 69.40 | 57.76 | 73.00 | 936 | Steelers |
| 58 | 14 | Graham Glasgow | 69.33 | 57.20 | 73.25 | 1149 | Lions |
| 59 | 15 | Blake Brandel | 68.92 | 55.70 | 73.56 | 1191 | Vikings |
| 60 | 16 | Robert Jones | 68.70 | 56.10 | 72.94 | 1080 | Dolphins |
| 61 | 17 | Patrick Mekari | 68.39 | 59.00 | 70.48 | 1131 | Ravens |
| 62 | 18 | Jake Kubas | 68.26 | 56.51 | 71.93 | 197 | Giants |
| 63 | 19 | Ben Bredeson | 68.00 | 56.00 | 71.84 | 1173 | Buccaneers |
| 64 | 20 | Jon Runyan | 67.99 | 56.39 | 71.56 | 842 | Giants |
| 65 | 21 | Ed Ingram | 67.81 | 55.40 | 71.92 | 580 | Vikings |
| 66 | 22 | Nick Zakelj | 67.75 | 59.47 | 69.11 | 162 | 49ers |
| 67 | 23 | O'Cyrus Torrence | 67.62 | 55.50 | 71.53 | 1221 | Bills |
| 68 | 24 | Logan Bruss | 67.61 | 53.24 | 73.03 | 195 | Titans |
| 69 | 25 | Kayode Awosika | 67.28 | 56.52 | 70.28 | 145 | Lions |
| 70 | 26 | Mark Glowinski | 66.92 | 54.98 | 70.71 | 355 | Colts |
| 71 | 27 | Jordan Morgan | 66.88 | 59.65 | 67.54 | 186 | Packers |
| 72 | 28 | Nick Saldiveri | 66.54 | 57.64 | 68.31 | 344 | Saints |
| 73 | 29 | Michael Dunn | 66.29 | 54.38 | 70.06 | 171 | Browns |
| 74 | 30 | Aaron Stinnie | 66.28 | 54.64 | 69.88 | 193 | Giants |
| 75 | 31 | Dalton Tucker | 66.19 | 55.40 | 69.22 | 464 | Colts |
| 76 | 32 | Anthony Bradford | 66.09 | 51.50 | 71.65 | 578 | Seahawks |
| 77 | 33 | Isaiah Wynn | 66.09 | 55.60 | 68.91 | 103 | Dolphins |
| 78 | 34 | Andrew Vorhees | 65.24 | 58.54 | 65.54 | 268 | Ravens |
| 79 | 35 | Zak Zinter | 65.12 | 52.17 | 69.58 | 233 | Browns |
| 80 | 36 | Mike Caliendo | 64.95 | 53.65 | 68.31 | 354 | Chiefs |
| 81 | 37 | Alex Cappa | 64.93 | 50.50 | 70.39 | 1132 | Bengals |
| 82 | 38 | Christian Haynes | 63.45 | 55.26 | 64.75 | 167 | Seahawks |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 83 | 1 | Layden Robinson | 61.57 | 47.18 | 67.00 | 602 | Patriots |
| 84 | 2 | Sataoa Laumea | 61.51 | 46.13 | 67.59 | 355 | Seahawks |
| 85 | 3 | Sidy Sow | 61.43 | 47.92 | 66.27 | 155 | Patriots |
| 86 | 4 | Kenyon Green | 60.09 | 43.55 | 66.95 | 582 | Texans |

## HB — Running Back

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Bucky Irving | 87.84 | 84.78 | 85.71 | 246 | Buccaneers |
| 2 | 2 | Jahmyr Gibbs | 86.71 | 89.30 | 80.82 | 347 | Lions |
| 3 | 3 | De'Von Achane | 86.21 | 81.60 | 85.11 | 408 | Dolphins |
| 4 | 4 | Bijan Robinson | 86.03 | 92.80 | 77.35 | 389 | Falcons |
| 5 | 5 | Derrick Henry | 84.71 | 88.45 | 78.05 | 197 | Ravens |
| 6 | 6 | Josh Jacobs | 82.76 | 89.10 | 74.37 | 265 | Packers |
| 7 | 7 | Saquon Barkley | 81.76 | 87.00 | 74.10 | 353 | Eagles |
| 8 | 8 | James Conner | 81.41 | 87.51 | 73.18 | 269 | Cardinals |
| 9 | 9 | Kenneth Walker III | 81.31 | 84.44 | 75.05 | 224 | Seahawks |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | James Cook | 79.38 | 81.59 | 73.74 | 258 | Bills |
| 11 | 2 | Aaron Jones | 78.38 | 74.99 | 76.47 | 347 | Vikings |
| 12 | 3 | Jordan Mason | 77.13 | 68.49 | 78.73 | 170 | 49ers |
| 13 | 4 | Jaylen Warren | 77.10 | 63.74 | 81.84 | 232 | Steelers |
| 14 | 5 | Chase Brown | 76.93 | 74.83 | 74.17 | 339 | Bengals |
| 15 | 6 | David Montgomery | 76.80 | 80.38 | 70.25 | 158 | Lions |
| 16 | 7 | Justice Hill | 76.59 | 76.25 | 72.65 | 264 | Ravens |
| 17 | 8 | Tony Pollard | 76.58 | 68.13 | 78.04 | 301 | Titans |
| 18 | 9 | Zach Charbonnet | 76.35 | 75.13 | 72.99 | 284 | Seahawks |
| 19 | 10 | Alvin Kamara | 75.76 | 72.92 | 73.49 | 311 | Saints |
| 20 | 11 | Najee Harris | 74.96 | 74.96 | 70.79 | 233 | Steelers |
| 21 | 12 | Jerome Ford | 74.94 | 70.02 | 74.05 | 304 | Browns |
| 22 | 13 | Rhamondre Stevenson | 74.84 | 68.72 | 74.76 | 273 | Patriots |
| 23 | 14 | Austin Ekeler | 74.82 | 68.99 | 74.54 | 283 | Commanders |
| 24 | 15 | Breece Hall | 74.62 | 62.00 | 78.87 | 384 | Jets |
| 25 | 16 | Tyjae Spears | 74.43 | 66.22 | 75.74 | 167 | Titans |
| 26 | 17 | Chuba Hubbard | 74.41 | 75.33 | 69.63 | 336 | Panthers |

### Starter (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | Rico Dowdle | 73.93 | 72.00 | 71.05 | 283 | Cowboys |
| 28 | 2 | Kareem Hunt | 73.87 | 72.51 | 70.61 | 238 | Chiefs |
| 29 | 3 | Antonio Gibson | 73.86 | 70.01 | 72.26 | 164 | Patriots |
| 30 | 4 | Jaleel McLaughlin | 73.27 | 63.22 | 75.81 | 146 | Broncos |
| 31 | 5 | Raheem Mostert | 73.23 | 67.28 | 73.03 | 155 | Dolphins |
| 32 | 6 | Joe Mixon | 72.85 | 75.08 | 67.20 | 273 | Texans |
| 33 | 7 | J.K. Dobbins | 72.84 | 65.10 | 73.83 | 227 | Chargers |
| 34 | 8 | Ty Johnson | 72.71 | 70.37 | 70.11 | 231 | Bills |
| 35 | 9 | Kyren Williams | 72.55 | 69.10 | 70.68 | 402 | Rams |
| 36 | 10 | Emanuel Wilson | 72.47 | 72.26 | 68.44 | 109 | Packers |
| 37 | 11 | Miles Sanders | 72.35 | 66.01 | 72.41 | 130 | Panthers |
| 38 | 12 | Jeremy McNichols | 72.28 | 67.18 | 71.51 | 136 | Commanders |
| 39 | 13 | Tank Bigsby | 72.12 | 64.72 | 72.89 | 129 | Jaguars |
| 40 | 14 | Rachaad White | 71.75 | 72.48 | 67.09 | 311 | Buccaneers |
| 41 | 15 | Brian Robinson | 70.86 | 69.48 | 67.62 | 237 | Commanders |
| 42 | 16 | Ray Davis | 70.80 | 65.65 | 70.06 | 112 | Bills |
| 43 | 17 | Isaac Guerendo | 70.74 | 62.44 | 72.11 | 107 | 49ers |
| 44 | 18 | Devin Singletary | 70.65 | 61.67 | 72.47 | 164 | Giants |
| 45 | 19 | Jonathan Taylor | 70.24 | 57.19 | 74.77 | 270 | Colts |
| 46 | 20 | Javonte Williams | 69.99 | 61.59 | 71.43 | 300 | Broncos |
| 47 | 21 | Samaje Perine | 69.75 | 65.74 | 68.25 | 217 | Chiefs |
| 48 | 22 | Pierre Strong Jr. | 69.36 | 59.05 | 72.07 | 134 | Browns |
| 49 | 23 | Travis Etienne Jr. | 69.27 | 60.62 | 70.87 | 254 | Jaguars |
| 50 | 24 | Tyrone Tracy | 68.44 | 58.55 | 70.87 | 310 | Giants |
| 51 | 25 | D'Ernest Johnson | 68.19 | 60.39 | 69.23 | 117 | Jaguars |
| 52 | 26 | Isiah Pacheco | 68.14 | 63.07 | 67.36 | 115 | Chiefs |
| 53 | 27 | Cam Akers | 68.05 | 67.03 | 64.56 | 126 | Vikings |
| 54 | 28 | D'Andre Swift | 67.87 | 61.27 | 68.10 | 353 | Bears |
| 55 | 29 | Alexander Mattison | 67.47 | 61.06 | 67.57 | 218 | Raiders |
| 56 | 30 | Braelon Allen | 67.41 | 68.08 | 62.80 | 134 | Jets |
| 57 | 31 | Ameer Abdullah | 67.04 | 66.84 | 63.00 | 258 | Raiders |
| 58 | 32 | Trey Sermon | 66.99 | 56.66 | 69.71 | 122 | Colts |
| 59 | 33 | Zack Moss | 66.25 | 58.85 | 67.01 | 155 | Bengals |
| 60 | 34 | Kenneth Gainwell | 66.02 | 58.59 | 66.81 | 132 | Eagles |
| 61 | 35 | Roschon Johnson | 64.84 | 65.68 | 60.11 | 136 | Bears |
| 62 | 36 | Dare Ogunbowale | 62.42 | 60.45 | 59.56 | 213 | Texans |

### Rotation/backup (0 players)

_None._

## LB — Linebacker

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Fred Warner | 85.07 | 89.20 | 78.15 | 997 | 49ers |
| 2 | 2 | Zack Baun | 84.55 | 90.20 | 78.58 | 1150 | Eagles |
| 3 | 3 | Bobby Wagner | 83.29 | 88.30 | 75.79 | 1258 | Commanders |
| 4 | 4 | Leo Chenal | 82.15 | 81.24 | 78.59 | 497 | Chiefs |

### Good (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Jack Campbell | 79.35 | 78.70 | 75.61 | 1047 | Lions |
| 6 | 2 | Edgerrin Cooper | 78.02 | 81.03 | 73.82 | 549 | Packers |
| 7 | 3 | Devin Lloyd | 76.46 | 76.70 | 73.22 | 884 | Jaguars |
| 8 | 4 | Elandon Roberts | 74.60 | 77.34 | 68.61 | 525 | Steelers |

### Starter (67 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Devin Bush | 73.75 | 74.95 | 71.64 | 497 | Browns |
| 10 | 2 | Demario Davis | 73.38 | 73.20 | 69.81 | 1090 | Saints |
| 11 | 3 | Jordan Hicks | 73.36 | 75.91 | 71.13 | 602 | Browns |
| 12 | 4 | Eric Kendricks | 73.21 | 75.20 | 69.28 | 918 | Cowboys |
| 13 | 5 | Payton Wilson | 73.16 | 71.71 | 69.96 | 520 | Steelers |
| 14 | 6 | Bobby Okereke | 72.44 | 74.42 | 69.40 | 734 | Giants |
| 15 | 7 | Nakobe Dean | 72.34 | 76.60 | 70.92 | 880 | Eagles |
| 16 | 8 | Oren Burks | 72.21 | 73.72 | 71.36 | 322 | Eagles |
| 17 | 9 | Jordyn Brooks | 71.78 | 71.30 | 68.42 | 1039 | Dolphins |
| 18 | 10 | Logan Wilson | 71.74 | 71.94 | 70.78 | 743 | Bengals |
| 19 | 11 | Blake Cashman | 71.68 | 72.00 | 69.45 | 947 | Vikings |
| 20 | 12 | Robert Spillane | 71.61 | 68.40 | 69.79 | 1093 | Raiders |
| 21 | 13 | Jeremiah Owusu-Koramoah | 71.53 | 77.50 | 68.96 | 460 | Browns |
| 22 | 14 | Malcolm Rodriguez | 71.45 | 68.97 | 73.94 | 318 | Lions |
| 23 | 15 | Jack Gibbens | 71.16 | 72.39 | 73.82 | 234 | Titans |
| 24 | 16 | Roquan Smith | 70.81 | 66.80 | 69.31 | 1099 | Ravens |
| 25 | 17 | Christian Elliss | 70.33 | 69.98 | 68.36 | 514 | Patriots |
| 26 | 18 | Kaden Elliss | 70.29 | 71.10 | 65.59 | 1097 | Falcons |
| 27 | 19 | Quincy Williams | 70.12 | 68.00 | 67.77 | 1136 | Jets |
| 28 | 20 | Lavonte David | 69.48 | 67.90 | 66.36 | 1149 | Buccaneers |
| 29 | 21 | Pete Werner | 69.40 | 68.70 | 68.94 | 731 | Saints |
| 30 | 22 | Foyesade Oluokun | 69.31 | 68.48 | 67.67 | 815 | Jaguars |
| 31 | 23 | Daiyan Henley | 69.20 | 69.90 | 68.62 | 1071 | Chargers |
| 32 | 24 | Azeez Al-Shaair | 69.16 | 68.06 | 68.66 | 672 | Texans |
| 33 | 25 | Omar Speights | 68.74 | 67.37 | 69.40 | 504 | Rams |
| 34 | 26 | Tyrel Dodson | 68.47 | 67.30 | 68.62 | 854 | Dolphins |
| 35 | 27 | Cody Barton | 68.05 | 63.70 | 67.96 | 1129 | Broncos |
| 36 | 28 | Drue Tranquill | 68.02 | 66.00 | 65.20 | 902 | Chiefs |
| 37 | 29 | C.J. Mosley | 67.84 | 69.41 | 68.99 | 110 | Jets |
| 38 | 30 | Alex Anzalone | 67.71 | 66.29 | 67.42 | 681 | Lions |
| 39 | 31 | Jeremiah Trotter Jr. | 66.93 | 66.16 | 72.09 | 109 | Eagles |
| 40 | 32 | Chazz Surratt | 66.85 | 65.19 | 70.36 | 137 | Jets |
| 41 | 33 | Jake Hansen | 66.76 | 67.33 | 70.94 | 136 | Texans |
| 42 | 34 | Dee Winters | 66.26 | 64.46 | 67.09 | 398 | 49ers |
| 43 | 35 | Derrick Barnes | 66.20 | 64.72 | 70.67 | 120 | Lions |
| 44 | 36 | T.J. Edwards | 65.90 | 61.40 | 64.74 | 1054 | Bears |
| 45 | 37 | Nate Landman | 65.74 | 65.06 | 66.82 | 543 | Falcons |
| 46 | 38 | Nick Bolton | 65.66 | 62.50 | 65.07 | 1076 | Chiefs |
| 47 | 39 | Frankie Luvu | 65.48 | 64.20 | 62.57 | 1239 | Commanders |
| 48 | 40 | Mack Wilson Sr. | 65.47 | 63.66 | 64.28 | 760 | Cardinals |
| 49 | 41 | Ernest Jones | 65.29 | 60.70 | 64.97 | 995 | Seahawks |
| 50 | 42 | Krys Barnes | 65.13 | 61.55 | 67.57 | 205 | Cardinals |
| 51 | 43 | Tyrice Knight | 65.12 | 64.42 | 66.32 | 550 | Seahawks |
| 52 | 44 | Grant Stuard | 65.11 | 64.97 | 68.49 | 229 | Colts |
| 53 | 45 | Henry To'oTo'o | 65.02 | 62.20 | 63.84 | 936 | Texans |
| 54 | 46 | Neville Hewitt | 65.01 | 67.79 | 67.91 | 351 | Texans |
| 55 | 47 | Ivan Pace Jr. | 65.01 | 62.54 | 65.55 | 454 | Vikings |
| 56 | 48 | Joe Andreessen | 64.88 | 64.84 | 69.55 | 116 | Bills |
| 57 | 49 | Damone Clark | 64.81 | 63.66 | 66.21 | 163 | Cowboys |
| 58 | 50 | Zaire Franklin | 64.69 | 60.30 | 63.75 | 1157 | Colts |
| 59 | 51 | J.J. Russell | 64.66 | 65.29 | 69.67 | 271 | Buccaneers |
| 60 | 52 | Troy Dye | 64.40 | 63.68 | 66.50 | 355 | Chargers |
| 61 | 53 | Jack Sanborn | 64.17 | 61.66 | 63.44 | 235 | Bears |
| 62 | 54 | Germaine Pratt | 64.03 | 60.20 | 62.82 | 1075 | Bengals |
| 63 | 55 | Dorian Williams | 64.02 | 58.63 | 64.55 | 680 | Bills |
| 64 | 56 | Tremaine Edmunds | 63.72 | 59.30 | 63.86 | 1055 | Bears |
| 65 | 57 | Micah McFadden | 63.72 | 62.53 | 62.49 | 668 | Giants |
| 66 | 58 | Eric Wilson | 63.17 | 63.55 | 63.26 | 559 | Packers |
| 67 | 59 | Sione Takitaki | 62.81 | 61.46 | 63.76 | 194 | Patriots |
| 68 | 60 | Jerome Baker | 62.73 | 60.90 | 64.39 | 566 | Titans |
| 69 | 61 | Quay Walker | 62.54 | 57.43 | 63.55 | 804 | Packers |
| 70 | 62 | DeMarvion Overshown | 62.34 | 61.49 | 61.19 | 708 | Cowboys |
| 71 | 63 | Chris Board | 62.32 | 62.80 | 65.08 | 213 | Ravens |
| 72 | 64 | Shaq Thompson | 62.22 | 65.39 | 66.73 | 245 | Panthers |
| 73 | 65 | Yasir Abdullah | 62.15 | 58.45 | 63.52 | 170 | Jaguars |
| 74 | 66 | E.J. Speed | 62.06 | 56.70 | 62.74 | 1011 | Colts |
| 75 | 67 | Akeem Davis-Gaither | 62.02 | 59.19 | 61.70 | 535 | Bengals |

### Rotation/backup (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 76 | 1 | Denzel Perryman | 61.88 | 61.81 | 62.08 | 343 | Chargers |
| 77 | 2 | Trenton Simpson | 61.67 | 58.84 | 64.04 | 654 | Ravens |
| 78 | 3 | Alex Singleton | 61.56 | 62.00 | 64.35 | 190 | Broncos |
| 79 | 4 | Jahlani Tavai | 61.44 | 55.50 | 61.23 | 916 | Patriots |
| 80 | 5 | Chad Muma | 61.41 | 58.54 | 63.48 | 260 | Jaguars |
| 81 | 6 | Owen Pappoe | 61.34 | 61.20 | 64.36 | 131 | Cardinals |
| 82 | 7 | Patrick Queen | 61.03 | 56.80 | 59.69 | 1164 | Steelers |
| 83 | 8 | Darius Muasau | 60.71 | 57.67 | 62.48 | 435 | Giants |
| 84 | 9 | Devin White | 60.37 | 58.61 | 62.58 | 176 | Texans |
| 85 | 10 | Isaiah McDuffie | 60.31 | 55.67 | 61.21 | 728 | Packers |
| 86 | 11 | SirVocea Dennis | 60.31 | 62.92 | 64.93 | 105 | Buccaneers |
| 87 | 12 | De'Vondre Campbell | 60.21 | 58.36 | 61.19 | 719 | 49ers |
| 88 | 13 | Malik Harrison | 59.83 | 53.86 | 60.33 | 438 | Ravens |
| 89 | 14 | Josey Jewell | 59.82 | 56.55 | 61.36 | 796 | Panthers |
| 90 | 15 | Ventrell Miller | 59.58 | 53.87 | 60.45 | 482 | Jaguars |
| 91 | 16 | Divine Deablo | 59.56 | 57.53 | 60.56 | 689 | Raiders |
| 92 | 17 | Winston Reid | 59.54 | 56.56 | 59.33 | 144 | Browns |
| 93 | 18 | Jalen Reeves-Maybin | 59.45 | 58.88 | 61.75 | 165 | Lions |
| 94 | 19 | Ben Niemann | 59.31 | 57.76 | 63.53 | 178 | Lions |
| 95 | 20 | Kyzir White | 59.08 | 48.80 | 63.54 | 1015 | Cardinals |
| 96 | 21 | Ty Summers | 59.05 | 61.68 | 66.91 | 113 | Giants |
| 97 | 22 | Troy Andersen | 58.99 | 60.24 | 63.51 | 287 | Falcons |
| 98 | 23 | Anfernee Orji | 58.99 | 57.97 | 60.41 | 147 | Saints |
| 99 | 24 | Christian Rozeboom | 58.91 | 53.50 | 60.90 | 956 | Rams |
| 100 | 25 | Luke Gifford | 58.80 | 60.10 | 65.53 | 203 | Titans |
| 101 | 26 | Trevin Wallace | 58.59 | 56.63 | 61.62 | 582 | Panthers |
| 102 | 27 | Marist Liufau | 58.06 | 52.12 | 57.86 | 520 | Cowboys |
| 103 | 28 | Troy Reeder | 57.86 | 58.59 | 62.73 | 372 | Rams |
| 104 | 29 | Isaiah Simmons | 57.81 | 53.23 | 59.14 | 181 | Giants |
| 105 | 30 | Claudin Cherelus | 57.48 | 61.45 | 62.52 | 158 | Panthers |
| 106 | 31 | Amari Burney | 57.16 | 58.88 | 60.29 | 101 | Raiders |
| 107 | 32 | Ezekiel Turner | 57.11 | 55.88 | 64.44 | 111 | Lions |
| 108 | 33 | JD Bertrand | 56.85 | 55.54 | 62.38 | 157 | Falcons |
| 109 | 34 | Luke Masterson | 56.51 | 54.68 | 62.00 | 102 | Raiders |
| 110 | 35 | Terrel Bernard | 56.47 | 48.20 | 60.46 | 917 | Bills |
| 111 | 36 | Nick Vigil | 56.39 | 55.32 | 61.37 | 127 | Cowboys |
| 112 | 37 | Mohamoud Diabate | 56.14 | 53.69 | 60.11 | 581 | Browns |
| 113 | 38 | Anthony Walker Jr. | 55.78 | 50.48 | 62.30 | 516 | Dolphins |
| 114 | 39 | Kenneth Murray Jr. | 55.48 | 45.94 | 59.74 | 815 | Titans |
| 115 | 40 | Willie Gay | 55.47 | 50.64 | 57.27 | 277 | Saints |
| 116 | 41 | Raekwon McMillan | 54.95 | 49.04 | 59.52 | 267 | Titans |
| 117 | 42 | Matt Milano | 54.29 | 54.76 | 58.62 | 333 | Bills |
| 118 | 43 | Junior Colson | 53.48 | 47.55 | 59.15 | 234 | Chargers |
| 119 | 44 | K.J. Britt | 53.47 | 47.27 | 58.83 | 632 | Buccaneers |
| 120 | 45 | Kamu Grugier-Hill | 52.98 | 48.88 | 55.97 | 182 | Vikings |
| 121 | 46 | Christian Harris | 52.56 | 50.21 | 56.82 | 180 | Texans |
| 122 | 47 | Justin Strnad | 52.07 | 50.43 | 57.03 | 736 | Broncos |
| 123 | 48 | Jacoby Windmon | 50.96 | 56.20 | 59.56 | 128 | Panthers |
| 124 | 49 | Demetrius Flannigan-Fowles | 48.74 | 47.38 | 54.40 | 151 | 49ers |
| 125 | 50 | Baylon Spector | 45.59 | 42.19 | 54.77 | 291 | Bills |
| 126 | 51 | Chandler Wooten | 45.00 | 44.34 | 54.02 | 212 | Panthers |

## QB — Quarterback

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Lamar Jackson | 87.60 | 89.15 | 83.54 | 635 | Ravens |
| 2 | 2 | Joe Burrow | 85.05 | 89.92 | 78.23 | 775 | Bengals |
| 3 | 3 | Josh Allen | 83.44 | 85.33 | 77.56 | 686 | Bills |
| 4 | 4 | Justin Herbert | 81.71 | 87.09 | 74.17 | 662 | Chargers |
| 5 | 5 | Jared Goff | 80.56 | 78.41 | 78.11 | 648 | Lions |
| 6 | 6 | Baker Mayfield | 80.15 | 79.34 | 77.00 | 721 | Buccaneers |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Geno Smith | 79.61 | 81.34 | 74.30 | 704 | Seahawks |
| 8 | 2 | Patrick Mahomes | 78.50 | 82.91 | 70.35 | 776 | Chiefs |
| 9 | 3 | Brock Purdy | 77.25 | 77.54 | 75.94 | 567 | 49ers |
| 10 | 4 | Jayden Daniels | 77.00 | 84.70 | 73.05 | 781 | Commanders |
| 11 | 5 | Jalen Hurts | 76.57 | 74.54 | 76.16 | 558 | Eagles |
| 12 | 6 | Matthew Stafford | 76.23 | 76.21 | 73.20 | 667 | Rams |
| 13 | 7 | Tua Tagovailoa | 76.10 | 76.11 | 76.00 | 460 | Dolphins |
| 14 | 8 | Sam Darnold | 76.02 | 76.79 | 76.74 | 725 | Vikings |
| 15 | 9 | C.J. Stroud | 75.87 | 78.24 | 69.61 | 742 | Texans |
| 16 | 10 | Derek Carr | 74.41 | 76.93 | 74.82 | 319 | Saints |
| 17 | 11 | Jordan Love | 74.30 | 76.76 | 72.26 | 528 | Packers |

### Starter (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Russell Wilson | 73.51 | 73.38 | 74.13 | 444 | Steelers |
| 19 | 2 | Kyler Murray | 73.43 | 75.31 | 70.91 | 656 | Cardinals |
| 20 | 3 | Kirk Cousins | 71.78 | 74.50 | 70.17 | 521 | Falcons |
| 21 | 4 | Aaron Rodgers | 71.60 | 76.21 | 67.78 | 684 | Jets |
| 22 | 5 | Dak Prescott | 70.64 | 73.35 | 70.49 | 344 | Cowboys |
| 23 | 6 | Bo Nix | 69.90 | 73.80 | 68.20 | 712 | Broncos |
| 24 | 7 | Trevor Lawrence | 69.73 | 73.19 | 68.40 | 332 | Jaguars |
| 25 | 8 | Bryce Young | 64.89 | 66.61 | 63.10 | 477 | Panthers |
| 26 | 9 | Caleb Williams | 64.73 | 62.90 | 64.42 | 741 | Bears |
| 27 | 10 | Joe Flacco | 63.64 | 67.59 | 68.14 | 290 | Colts |
| 28 | 11 | Justin Fields | 62.76 | 62.70 | 68.38 | 215 | Steelers |
| 29 | 12 | Michael Penix Jr. | 62.65 | 72.45 | 69.61 | 120 | Falcons |
| 30 | 13 | Drake Maye | 62.19 | 64.25 | 66.76 | 461 | Patriots |

### Rotation/backup (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Jameis Winston | 61.31 | 67.75 | 64.81 | 347 | Browns |
| 32 | 2 | Andy Dalton | 61.07 | 68.66 | 64.43 | 185 | Panthers |
| 33 | 3 | Daniel Jones | 60.67 | 65.56 | 60.10 | 418 | Vikings |
| 34 | 4 | Aidan O'Connell | 60.26 | 60.34 | 65.94 | 276 | Raiders |
| 35 | 5 | Gardner Minshew | 59.98 | 60.92 | 62.44 | 370 | Raiders |
| 36 | 6 | Will Levis | 59.72 | 57.00 | 65.74 | 384 | Titans |
| 37 | 7 | Mason Rudolph | 59.20 | 61.35 | 63.63 | 276 | Titans |
| 38 | 8 | Anthony Richardson | 58.81 | 59.58 | 62.34 | 317 | Colts |
| 39 | 9 | Mac Jones | 58.67 | 60.22 | 61.68 | 309 | Jaguars |
| 40 | 10 | Cooper Rush | 57.88 | 59.51 | 60.73 | 352 | Cowboys |
| 41 | 11 | Tyler Huntley | 57.88 | 60.82 | 59.84 | 182 | Dolphins |
| 42 | 12 | Deshaun Watson | 57.63 | 62.52 | 58.91 | 290 | Browns |
| 43 | 13 | Jacoby Brissett | 56.34 | 62.51 | 57.19 | 200 | Patriots |
| 44 | 14 | Drew Lock | 56.05 | 54.03 | 58.01 | 216 | Giants |
| 45 | 15 | Spencer Rattler | 55.87 | 52.72 | 55.49 | 284 | Saints |
| 46 | 16 | Desmond Ridder | 55.46 | 56.68 | 60.03 | 105 | Raiders |
| 47 | 17 | Dorian Thompson-Robinson | 53.99 | 51.21 | 52.01 | 142 | Browns |

## S — Safety

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Kerby Joseph | 93.57 | 91.10 | 91.64 | 1158 | Lions |
| 2 | 2 | Xavier McKinney | 90.85 | 90.20 | 88.69 | 1125 | Packers |
| 3 | 3 | Brandon Jones | 87.61 | 89.80 | 84.83 | 1042 | Broncos |
| 4 | 4 | Kyle Hamilton | 86.30 | 88.40 | 80.93 | 1150 | Ravens |
| 5 | 5 | Ar'Darius Washington | 85.37 | 86.10 | 80.72 | 830 | Ravens |
| 6 | 6 | Jessie Bates III | 82.54 | 81.40 | 79.34 | 1095 | Falcons |
| 7 | 7 | Julian Love | 81.42 | 76.10 | 81.00 | 1079 | Seahawks |
| 8 | 8 | Justin Reid | 80.67 | 77.00 | 78.95 | 1112 | Chiefs |
| 9 | 9 | Brian Branch | 80.66 | 77.80 | 78.40 | 982 | Lions |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Derwin James Jr. | 79.89 | 76.30 | 79.00 | 1059 | Chargers |
| 11 | 2 | C.J. Gardner-Johnson | 79.21 | 83.10 | 76.67 | 1118 | Eagles |
| 12 | 3 | Jabrill Peppers | 78.88 | 77.42 | 81.67 | 372 | Patriots |
| 13 | 4 | Jaden Hicks | 78.46 | 69.67 | 80.16 | 430 | Chiefs |
| 14 | 5 | Budda Baker | 76.12 | 74.70 | 74.76 | 1064 | Cardinals |
| 15 | 6 | Andrew Wingard | 75.81 | 72.44 | 80.86 | 216 | Jaguars |
| 16 | 7 | Kamren Kinchens | 75.69 | 71.91 | 74.04 | 623 | Rams |
| 17 | 8 | Dell Pettus | 75.05 | 67.78 | 77.70 | 341 | Patriots |
| 18 | 9 | Jimmie Ward | 74.79 | 72.98 | 78.30 | 461 | Texans |
| 19 | 10 | Quandre Diggs | 74.65 | 68.77 | 78.82 | 419 | Titans |
| 20 | 11 | Julian Blackmon | 74.37 | 73.50 | 72.45 | 1084 | Colts |

### Starter (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Thomas Harper | 72.91 | 67.89 | 76.00 | 191 | Raiders |
| 22 | 2 | Evan Williams | 72.75 | 71.09 | 73.60 | 533 | Packers |
| 23 | 3 | Ronnie Hickman Jr. | 72.51 | 68.47 | 75.08 | 463 | Browns |
| 24 | 4 | Jordan Howden | 72.43 | 62.86 | 75.38 | 550 | Saints |
| 25 | 5 | Jalen Pitre | 72.42 | 70.11 | 72.24 | 660 | Texans |
| 26 | 6 | Reed Blankenship | 72.15 | 71.40 | 70.83 | 1030 | Eagles |
| 27 | 7 | DeShon Elliott | 72.03 | 63.30 | 75.05 | 895 | Steelers |
| 28 | 8 | Kevin Byard | 71.54 | 60.50 | 74.74 | 1055 | Bears |
| 29 | 9 | Harrison Smith | 71.30 | 65.30 | 71.71 | 1062 | Vikings |
| 30 | 10 | Minkah Fitzpatrick | 71.08 | 64.90 | 73.19 | 1158 | Steelers |
| 31 | 11 | Jalen Thompson | 71.03 | 64.20 | 72.99 | 941 | Cardinals |
| 32 | 12 | Dadrion Taylor-Demerson | 70.68 | 65.59 | 71.87 | 258 | Cardinals |
| 33 | 13 | Tony Jefferson | 70.47 | 67.15 | 78.11 | 261 | Chargers |
| 34 | 14 | Ji'Ayir Brown | 70.35 | 69.50 | 68.58 | 886 | 49ers |
| 35 | 15 | Mike Brown | 69.90 | 65.19 | 73.97 | 384 | Titans |
| 36 | 16 | Ashtyn Davis | 69.47 | 64.38 | 72.81 | 260 | Jets |
| 37 | 17 | Mike Edwards | 69.37 | 62.41 | 75.04 | 251 | Buccaneers |
| 38 | 18 | Jeremy Chinn | 68.95 | 64.50 | 70.40 | 1207 | Commanders |
| 39 | 19 | Tony Adams | 68.72 | 67.22 | 69.57 | 764 | Jets |
| 40 | 20 | Marcus Maye | 68.61 | 67.55 | 71.92 | 405 | Chargers |
| 41 | 21 | Nick Cross | 68.46 | 64.30 | 70.10 | 1156 | Colts |
| 42 | 22 | Zayne Anderson | 68.42 | 66.24 | 78.21 | 122 | Packers |
| 43 | 23 | Amani Hooker | 68.30 | 66.30 | 69.69 | 848 | Titans |
| 44 | 24 | Jaylen McCollough | 68.26 | 61.09 | 68.88 | 382 | Rams |
| 45 | 25 | Kamren Curl | 68.06 | 63.70 | 68.07 | 1112 | Rams |
| 46 | 26 | Jordan Poyer | 67.93 | 61.40 | 69.59 | 964 | Dolphins |
| 47 | 27 | Alohi Gilman | 67.28 | 64.34 | 68.41 | 731 | Chargers |
| 48 | 28 | Grant Delpit | 67.21 | 60.90 | 68.91 | 976 | Browns |
| 49 | 29 | Dane Belton | 66.97 | 60.97 | 68.57 | 460 | Giants |
| 50 | 30 | Malik Mustapha | 66.27 | 60.10 | 68.18 | 755 | 49ers |
| 51 | 31 | Malik Hooker | 66.23 | 57.80 | 67.89 | 1062 | Cowboys |
| 52 | 32 | Camryn Bynum | 66.20 | 58.60 | 67.10 | 1056 | Vikings |
| 53 | 33 | Damontae Kazee | 65.75 | 57.23 | 69.91 | 313 | Steelers |
| 54 | 34 | Tre'von Moehrig | 65.73 | 54.40 | 69.51 | 1099 | Raiders |
| 55 | 35 | Vonn Bell | 65.17 | 62.00 | 64.48 | 705 | Bengals |
| 56 | 36 | Eric Murray | 64.75 | 64.90 | 64.70 | 961 | Texans |
| 57 | 37 | Juan Thornhill | 64.72 | 63.68 | 65.84 | 401 | Browns |
| 58 | 38 | Will Harris | 64.52 | 61.60 | 64.27 | 860 | Saints |
| 59 | 39 | Jaquan Brisker | 64.33 | 62.09 | 68.52 | 293 | Bears |
| 60 | 40 | Devon Key | 64.18 | 58.06 | 65.56 | 253 | Broncos |
| 61 | 41 | Quentin Lake | 63.84 | 58.40 | 65.85 | 1207 | Rams |
| 62 | 42 | Justin Simmons | 63.83 | 60.80 | 63.75 | 1017 | Falcons |
| 63 | 43 | Tyler Nubin | 63.70 | 58.14 | 67.15 | 789 | Giants |
| 64 | 44 | Kaevon Merriweather | 63.58 | 61.67 | 66.69 | 274 | Buccaneers |
| 65 | 45 | George Odum | 63.51 | 62.75 | 69.75 | 139 | 49ers |
| 66 | 46 | Josh Metellus | 63.42 | 52.20 | 67.52 | 1030 | Vikings |
| 67 | 47 | Xavier Woods | 63.22 | 55.40 | 65.53 | 1216 | Panthers |
| 68 | 48 | Tyrann Mathieu | 62.84 | 57.80 | 62.04 | 1015 | Saints |
| 69 | 49 | Donovan Wilson | 62.74 | 56.60 | 62.96 | 1008 | Cowboys |
| 70 | 50 | Andre Cisco | 62.44 | 56.70 | 63.57 | 979 | Jaguars |
| 71 | 51 | Jaylinn Hawkins | 62.30 | 56.81 | 64.24 | 613 | Patriots |

### Rotation/backup (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 72 | 1 | Jevon Holland | 61.66 | 57.10 | 62.98 | 854 | Dolphins |
| 73 | 2 | Jordan Whitehead | 60.76 | 53.73 | 63.73 | 731 | Buccaneers |
| 74 | 3 | Rayshawn Jenkins | 60.15 | 58.37 | 59.14 | 550 | Seahawks |
| 75 | 4 | Jonathan Owens | 60.07 | 55.60 | 62.32 | 429 | Bears |
| 76 | 5 | Jordan Battle | 59.99 | 54.05 | 61.01 | 464 | Bengals |
| 77 | 6 | Antoine Winfield Jr. | 59.28 | 51.61 | 64.44 | 601 | Buccaneers |
| 78 | 7 | Geno Stone | 59.20 | 53.70 | 59.29 | 1100 | Bengals |
| 79 | 8 | Bryan Cook | 59.00 | 48.10 | 63.76 | 1056 | Chiefs |
| 80 | 9 | Christian Izien | 58.95 | 55.68 | 58.80 | 697 | Buccaneers |
| 81 | 10 | Taylor Rapp | 58.83 | 43.20 | 66.65 | 840 | Bills |
| 82 | 11 | Talanoa Hufanga | 58.61 | 55.36 | 63.57 | 308 | 49ers |
| 83 | 12 | Demani Richardson | 58.38 | 59.02 | 65.08 | 403 | Panthers |
| 84 | 13 | Nick Scott | 57.32 | 57.93 | 58.33 | 324 | Panthers |
| 85 | 14 | Javon Bullard | 56.88 | 49.06 | 58.91 | 816 | Packers |
| 86 | 15 | Eddie Jackson | 56.72 | 51.96 | 60.63 | 390 | Chargers |
| 87 | 16 | P.J. Locke | 56.36 | 50.70 | 58.71 | 1076 | Broncos |
| 88 | 17 | Rodney McLeod | 56.07 | 48.52 | 59.01 | 565 | Browns |
| 89 | 18 | Antonio Johnson | 55.60 | 44.51 | 58.82 | 685 | Jaguars |
| 90 | 19 | Jordan Fuller | 54.81 | 49.16 | 61.07 | 574 | Panthers |
| 91 | 20 | Chuck Clark | 54.63 | 45.75 | 58.83 | 709 | Jets |
| 92 | 21 | Percy Butler | 54.62 | 46.51 | 57.62 | 448 | Commanders |
| 93 | 22 | Cole Bishop | 54.62 | 51.53 | 55.45 | 464 | Bills |
| 94 | 23 | Richie Grant | 54.51 | 52.17 | 54.84 | 165 | Falcons |
| 95 | 24 | Jason Pinnock | 54.50 | 45.50 | 57.31 | 976 | Giants |
| 96 | 25 | Damar Hamlin | 53.87 | 41.50 | 62.17 | 1042 | Bills |
| 97 | 26 | Calen Bullock | 53.61 | 40.00 | 58.51 | 1083 | Texans |
| 98 | 27 | Marcus Williams | 53.49 | 40.00 | 63.80 | 601 | Ravens |
| 99 | 28 | Isaiah Pola-Mao | 53.43 | 45.10 | 58.64 | 952 | Raiders |
| 100 | 29 | K'Von Wallace | 53.05 | 52.72 | 54.41 | 127 | Seahawks |
| 101 | 30 | Marcus Epps | 52.60 | 52.61 | 55.29 | 176 | Raiders |
| 102 | 31 | Darnell Savage | 52.26 | 41.14 | 59.14 | 764 | Jaguars |
| 103 | 32 | Kyle Dugger | 50.16 | 40.00 | 55.12 | 759 | Patriots |

## T — Tackle

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jordan Mailata | 98.46 | 95.20 | 96.47 | 995 | Eagles |
| 2 | 2 | Penei Sewell | 94.24 | 89.60 | 93.17 | 1213 | Lions |
| 3 | 3 | Lane Johnson | 93.82 | 87.50 | 93.87 | 1123 | Eagles |
| 4 | 4 | Rashawn Slater | 93.73 | 90.65 | 91.61 | 959 | Chargers |
| 5 | 5 | Terron Armstead | 92.75 | 87.86 | 91.85 | 821 | Dolphins |
| 6 | 6 | Zach Tom | 91.99 | 85.80 | 91.95 | 1134 | Packers |
| 7 | 7 | Trent Williams | 89.84 | 82.71 | 90.43 | 649 | 49ers |
| 8 | 8 | Bernhard Raimann | 89.81 | 84.08 | 89.46 | 856 | Colts |
| 9 | 9 | Spencer Brown | 87.98 | 77.90 | 90.54 | 1140 | Bills |
| 10 | 10 | Tristan Wirfs | 87.89 | 82.50 | 87.32 | 1061 | Buccaneers |
| 11 | 11 | Garett Bolles | 87.87 | 80.20 | 88.81 | 1111 | Broncos |
| 12 | 12 | Charles Cross | 87.77 | 82.50 | 87.11 | 1094 | Seahawks |
| 13 | 13 | Darnell Wright | 87.39 | 79.30 | 88.62 | 1021 | Bears |
| 14 | 14 | Laremy Tunsil | 87.34 | 78.10 | 89.33 | 1167 | Texans |
| 15 | 15 | Christian Darrisaw | 86.86 | 76.66 | 89.49 | 392 | Vikings |
| 16 | 16 | Brian O'Neill | 86.49 | 79.30 | 87.11 | 1151 | Vikings |
| 17 | 17 | Alaric Jackson | 86.11 | 78.40 | 87.09 | 1017 | Rams |
| 18 | 18 | Kolton Miller | 86.07 | 80.60 | 85.55 | 1075 | Raiders |
| 19 | 19 | Jake Matthews | 86.04 | 79.80 | 86.03 | 1119 | Falcons |
| 20 | 20 | Paris Johnson Jr. | 86.02 | 79.49 | 86.20 | 865 | Cardinals |
| 21 | 21 | Luke Goedeke | 84.73 | 74.06 | 87.67 | 952 | Buccaneers |
| 22 | 22 | Taylor Moton | 84.49 | 76.44 | 85.69 | 846 | Panthers |
| 23 | 23 | Rob Havenstein | 84.25 | 74.89 | 86.32 | 805 | Rams |
| 24 | 24 | Joe Alt | 83.87 | 75.90 | 85.02 | 1066 | Chargers |
| 25 | 25 | Taylor Decker | 83.70 | 77.08 | 83.94 | 963 | Lions |
| 26 | 26 | Braxton Jones | 83.60 | 75.88 | 84.58 | 719 | Bears |
| 27 | 27 | Kaleb McGary | 82.77 | 73.90 | 84.52 | 1042 | Falcons |
| 28 | 28 | Ikem Ekwonu | 82.30 | 71.24 | 85.50 | 909 | Panthers |
| 29 | 29 | Dion Dawkins | 82.21 | 72.40 | 84.59 | 1164 | Bills |
| 30 | 30 | Mike McGlinchey | 82.20 | 72.23 | 84.68 | 891 | Broncos |
| 31 | 31 | Andrew Thomas | 81.78 | 72.16 | 84.03 | 416 | Giants |
| 32 | 32 | Cornelius Lucas | 81.61 | 71.45 | 84.21 | 464 | Commanders |
| 33 | 33 | Tyron Smith | 81.57 | 71.85 | 83.88 | 592 | Jets |
| 34 | 34 | Walker Little | 81.33 | 69.19 | 85.26 | 508 | Jaguars |
| 35 | 35 | Colton McKivitz | 81.30 | 72.20 | 83.20 | 1062 | 49ers |

### Good (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 36 | 1 | Ronnie Stanley | 79.86 | 70.70 | 81.80 | 1221 | Ravens |
| 37 | 2 | Jaylon Moore | 79.59 | 67.82 | 83.27 | 271 | 49ers |
| 38 | 3 | Terence Steele | 79.59 | 67.00 | 83.82 | 1168 | Cowboys |
| 39 | 4 | Braden Smith | 79.41 | 65.68 | 84.40 | 731 | Colts |
| 40 | 5 | Jonah Williams | 79.23 | 68.07 | 82.51 | 343 | Cardinals |
| 41 | 6 | Tytus Howard | 78.98 | 70.20 | 80.66 | 1157 | Texans |
| 42 | 7 | Kendall Lamm | 78.82 | 69.16 | 81.09 | 512 | Dolphins |
| 43 | 8 | Justin Skule | 77.68 | 65.58 | 81.58 | 362 | Buccaneers |
| 44 | 9 | Taliese Fuaga | 77.52 | 65.70 | 81.23 | 1070 | Saints |
| 45 | 10 | John Ojukwu | 77.35 | 64.87 | 81.51 | 264 | Titans |
| 46 | 11 | DJ Glaze | 77.14 | 66.10 | 80.34 | 998 | Raiders |
| 47 | 12 | Rasheed Walker | 77.10 | 68.60 | 78.60 | 1139 | Packers |
| 48 | 13 | Evan Neal | 76.98 | 60.82 | 83.58 | 459 | Giants |
| 49 | 14 | Trent Brown | 76.74 | 63.33 | 81.51 | 139 | Bengals |
| 50 | 15 | Dan Moore Jr. | 76.46 | 67.20 | 78.46 | 1128 | Steelers |
| 51 | 16 | Roger Rosengarten | 76.33 | 66.00 | 79.05 | 1066 | Ravens |
| 52 | 17 | Morgan Moses | 76.24 | 63.02 | 80.89 | 723 | Jets |
| 53 | 18 | Matt Goncalves | 76.17 | 64.47 | 79.81 | 566 | Colts |
| 54 | 19 | Jack Conklin | 75.79 | 65.87 | 78.23 | 818 | Browns |
| 55 | 20 | Kelvin Beachum | 75.55 | 63.78 | 79.23 | 742 | Cardinals |
| 56 | 21 | Matt Peart | 75.47 | 63.25 | 79.45 | 190 | Broncos |
| 57 | 22 | Alex Palczewski | 75.46 | 61.45 | 80.64 | 179 | Broncos |
| 58 | 23 | Anton Harrison | 74.89 | 64.11 | 77.91 | 943 | Jaguars |
| 59 | 24 | Storm Norton | 74.74 | 60.76 | 79.90 | 128 | Falcons |
| 60 | 25 | Austin Jackson | 74.28 | 60.00 | 79.63 | 542 | Dolphins |
| 61 | 26 | Abraham Lucas | 74.27 | 61.22 | 78.80 | 406 | Seahawks |
| 62 | 27 | Cam Robinson | 74.27 | 63.20 | 77.48 | 1073 | Vikings |

### Starter (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 63 | 1 | Trevor Penning | 73.98 | 60.20 | 79.00 | 1081 | Saints |
| 64 | 2 | Joshua Ezeudu | 73.91 | 61.16 | 78.25 | 182 | Giants |
| 65 | 3 | Warren McClendon Jr. | 73.88 | 59.88 | 79.04 | 333 | Rams |
| 66 | 4 | Olumuyiwa Fashanu | 73.76 | 60.88 | 78.18 | 534 | Jets |
| 67 | 5 | Andrew Wylie | 73.67 | 61.70 | 77.48 | 1115 | Commanders |
| 68 | 6 | JC Latham | 73.61 | 61.80 | 77.31 | 1095 | Titans |
| 69 | 7 | Cole Van Lanen | 73.58 | 61.16 | 77.70 | 252 | Jaguars |
| 70 | 8 | Jawaan Taylor | 73.48 | 59.80 | 78.43 | 1209 | Chiefs |
| 71 | 9 | Joe Noteboom | 73.40 | 60.00 | 78.16 | 220 | Rams |
| 72 | 10 | Brandon Coleman | 72.79 | 59.80 | 77.29 | 1013 | Commanders |
| 73 | 11 | Broderick Jones | 72.49 | 58.70 | 77.52 | 1117 | Steelers |
| 74 | 12 | Orlando Brown Jr. | 71.74 | 58.41 | 76.46 | 637 | Bengals |
| 75 | 13 | Jackson Barton | 70.79 | 61.56 | 72.77 | 157 | Cardinals |
| 76 | 14 | Amarius Mims | 70.78 | 57.97 | 75.15 | 835 | Bengals |
| 77 | 15 | Yosh Nijman | 70.27 | 59.08 | 73.56 | 187 | Panthers |
| 78 | 16 | Chuma Edoga | 70.02 | 56.41 | 74.92 | 226 | Cowboys |
| 79 | 17 | Jedrick Wills Jr. | 69.89 | 56.46 | 74.67 | 245 | Browns |
| 80 | 18 | Chris Hubbard | 69.77 | 53.16 | 76.67 | 257 | Giants |
| 81 | 19 | Devin Cochran | 69.64 | 56.24 | 74.40 | 152 | Bengals |
| 82 | 20 | Vederian Lowe | 69.10 | 54.58 | 74.61 | 803 | Patriots |
| 83 | 21 | Dan Skipper | 69.10 | 57.25 | 72.84 | 324 | Lions |
| 84 | 22 | James Hudson III | 69.05 | 55.44 | 73.95 | 222 | Browns |
| 85 | 23 | David Quessenberry | 68.84 | 56.93 | 72.61 | 133 | Vikings |
| 86 | 24 | Wanya Morris | 68.77 | 53.97 | 74.47 | 732 | Chiefs |
| 87 | 25 | Caedan Wallace | 68.54 | 53.64 | 74.31 | 129 | Patriots |
| 88 | 26 | Larry Borom | 68.39 | 56.42 | 72.20 | 329 | Bears |
| 89 | 27 | Carter Warren | 67.88 | 53.84 | 73.08 | 141 | Jets |
| 90 | 28 | Tyler Guyton | 67.61 | 51.27 | 74.34 | 668 | Cowboys |
| 91 | 29 | Trent Scott | 67.19 | 52.48 | 72.83 | 288 | Commanders |
| 92 | 30 | Mike Jerrell | 66.64 | 53.15 | 71.46 | 250 | Seahawks |
| 93 | 31 | Dawand Jones | 66.56 | 50.20 | 73.30 | 511 | Browns |
| 94 | 32 | Fred Johnson | 66.42 | 52.45 | 71.56 | 490 | Eagles |
| 95 | 33 | Patrick Paul | 66.36 | 51.15 | 72.34 | 338 | Dolphins |
| 96 | 34 | Thayer Munford Jr. | 66.11 | 53.63 | 70.27 | 201 | Raiders |
| 97 | 35 | Ryan Van Demark | 65.60 | 56.99 | 67.18 | 199 | Bills |
| 98 | 36 | Nicholas Petit-Frere | 65.44 | 49.28 | 72.05 | 621 | Titans |
| 99 | 37 | Charlie Heck | 65.39 | 52.48 | 69.83 | 117 | 49ers |
| 100 | 38 | Stone Forsythe | 64.54 | 49.04 | 70.71 | 414 | Seahawks |
| 101 | 39 | Blake Fisher | 64.46 | 49.34 | 70.38 | 478 | Texans |
| 102 | 40 | Kingsley Suamataia | 64.22 | 50.09 | 69.48 | 198 | Chiefs |
| 103 | 41 | Kiran Amegadjie | 63.05 | 52.12 | 66.17 | 126 | Bears |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 104 | 1 | Demontrey Jacobs | 60.73 | 40.00 | 70.39 | 867 | Patriots |

## TE — Tight End

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | George Kittle | 85.06 | 89.42 | 77.99 | 489 | 49ers |
| 2 | 2 | Trey McBride | 81.57 | 85.81 | 74.57 | 581 | Cardinals |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 3 | 1 | Mark Andrews | 79.99 | 80.75 | 75.32 | 455 | Ravens |
| 4 | 2 | Brock Bowers | 79.54 | 84.99 | 71.74 | 654 | Raiders |
| 5 | 3 | Travis Kelce | 77.55 | 72.70 | 76.62 | 698 | Chiefs |
| 6 | 4 | Jonnu Smith | 76.49 | 75.62 | 72.90 | 486 | Dolphins |
| 7 | 5 | Dallas Goedert | 74.92 | 70.92 | 73.42 | 325 | Eagles |
| 8 | 6 | Sam LaPorta | 74.20 | 72.86 | 70.93 | 546 | Lions |
| 9 | 7 | T.J. Hockenson | 74.17 | 72.53 | 71.09 | 366 | Vikings |
| 10 | 8 | Austin Hooper | 74.10 | 73.07 | 70.62 | 335 | Patriots |
| 11 | 9 | Foster Moreau | 74.08 | 68.59 | 73.58 | 395 | Saints |

### Starter (58 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Evan Engram | 73.92 | 67.85 | 73.80 | 260 | Jaguars |
| 13 | 2 | Dalton Kincaid | 73.07 | 68.55 | 71.92 | 365 | Bills |
| 14 | 3 | Hunter Henry | 72.73 | 69.47 | 70.73 | 548 | Patriots |
| 15 | 4 | Zach Ertz | 72.48 | 66.99 | 71.98 | 658 | Commanders |
| 16 | 5 | Jordan Akins | 72.48 | 65.23 | 73.14 | 348 | Browns |
| 17 | 6 | Andrew Ogletree | 72.27 | 64.66 | 73.18 | 173 | Colts |
| 18 | 7 | Tucker Kraft | 72.19 | 67.04 | 71.46 | 538 | Packers |
| 19 | 8 | David Njoku | 72.16 | 63.50 | 73.76 | 415 | Browns |
| 20 | 9 | Isaiah Likely | 72.02 | 71.93 | 67.92 | 386 | Ravens |
| 21 | 10 | Mo Alie-Cox | 71.86 | 62.64 | 73.84 | 227 | Colts |
| 22 | 11 | Noah Gray | 71.79 | 68.95 | 69.52 | 380 | Chiefs |
| 23 | 12 | Will Dissly | 71.75 | 65.25 | 71.91 | 361 | Chargers |
| 24 | 13 | Mike Gesicki | 71.62 | 70.29 | 68.34 | 449 | Bengals |
| 25 | 14 | Stone Smartt | 70.82 | 62.93 | 71.92 | 138 | Chargers |
| 26 | 15 | Josh Oliver | 70.63 | 68.87 | 67.63 | 254 | Vikings |
| 27 | 16 | Pat Freiermuth | 69.56 | 66.70 | 67.30 | 516 | Steelers |
| 28 | 17 | Darnell Washington | 69.54 | 66.99 | 67.07 | 257 | Steelers |
| 29 | 18 | Kylen Granson | 69.50 | 58.92 | 72.39 | 268 | Colts |
| 30 | 19 | Cole Kmet | 69.49 | 60.59 | 71.26 | 637 | Bears |
| 31 | 20 | Tyler Conklin | 69.40 | 58.90 | 72.23 | 556 | Jets |
| 32 | 21 | Erick All | 69.39 | 59.96 | 71.51 | 112 | Bengals |
| 33 | 22 | Colby Parkinson | 69.27 | 62.46 | 69.64 | 390 | Rams |
| 34 | 23 | Hunter Long | 69.17 | 62.69 | 69.33 | 110 | Rams |
| 35 | 24 | Chigoziem Okonkwo | 69.13 | 59.92 | 71.11 | 425 | Titans |
| 36 | 25 | Josh Whyle | 68.99 | 62.67 | 69.03 | 204 | Titans |
| 37 | 26 | Noah Fant | 68.89 | 64.82 | 67.43 | 426 | Seahawks |
| 38 | 27 | Kyle Pitts | 68.85 | 59.63 | 70.83 | 511 | Falcons |
| 39 | 28 | Taysom Hill | 68.76 | 64.72 | 67.28 | 103 | Saints |
| 40 | 29 | Payne Durham | 68.64 | 60.43 | 69.95 | 191 | Buccaneers |
| 41 | 30 | Luke Farrell | 68.35 | 58.90 | 70.49 | 152 | Jaguars |
| 42 | 31 | Chris Manhertz | 68.15 | 64.27 | 66.57 | 139 | Giants |
| 43 | 32 | Brenton Strange | 68.02 | 64.18 | 66.41 | 320 | Jaguars |
| 44 | 33 | Juwan Johnson | 67.94 | 65.64 | 65.30 | 467 | Saints |
| 45 | 34 | Cade Otton | 67.88 | 63.82 | 66.42 | 573 | Buccaneers |
| 46 | 35 | Ben Sinnott | 67.37 | 57.08 | 70.07 | 122 | Commanders |
| 47 | 36 | Lucas Krull | 66.89 | 55.30 | 70.45 | 252 | Broncos |
| 48 | 37 | Adam Trautman | 66.75 | 57.95 | 68.45 | 288 | Broncos |
| 49 | 38 | Brevyn Spann-Ford | 66.69 | 57.04 | 68.96 | 146 | Cowboys |
| 50 | 39 | Nate Adkins | 66.31 | 62.04 | 64.99 | 181 | Broncos |
| 51 | 40 | Dawson Knox | 66.24 | 57.77 | 67.72 | 392 | Bills |
| 52 | 41 | Michael Mayer | 66.10 | 58.49 | 67.01 | 284 | Raiders |
| 53 | 42 | Dalton Schultz | 65.92 | 60.80 | 65.17 | 648 | Texans |
| 54 | 43 | Gerald Everett | 65.85 | 52.32 | 70.71 | 133 | Bears |
| 55 | 44 | Johnny Mundt | 64.98 | 58.33 | 65.25 | 236 | Vikings |
| 56 | 45 | AJ Barner | 64.80 | 60.62 | 63.42 | 253 | Seahawks |
| 57 | 46 | Jake Ferguson | 64.38 | 55.15 | 66.37 | 427 | Cowboys |
| 58 | 47 | Charlie Woerner | 64.31 | 59.51 | 63.34 | 131 | Falcons |
| 59 | 48 | Harrison Bryant | 64.20 | 60.00 | 62.84 | 101 | Raiders |
| 60 | 49 | Theo Johnson | 63.99 | 54.74 | 65.99 | 446 | Giants |
| 61 | 50 | Grant Calcaterra | 63.95 | 55.07 | 65.70 | 347 | Eagles |
| 62 | 51 | Jeremy Ruckert | 63.59 | 52.72 | 66.67 | 198 | Jets |
| 63 | 52 | Daniel Bellinger | 63.52 | 58.43 | 62.75 | 207 | Giants |
| 64 | 53 | Luke Schoonmaker | 63.45 | 58.98 | 62.27 | 213 | Cowboys |
| 65 | 54 | Nick Vannett | 63.29 | 57.33 | 63.10 | 145 | Titans |
| 66 | 55 | Pharaoh Brown | 63.19 | 54.20 | 65.01 | 107 | Seahawks |
| 67 | 56 | Ja'Tavion Sanders | 63.17 | 54.47 | 64.81 | 359 | Panthers |
| 68 | 57 | Brock Wright | 63.15 | 55.23 | 64.26 | 235 | Lions |
| 69 | 58 | John Bates | 62.05 | 52.15 | 64.49 | 252 | Commanders |

### Rotation/backup (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 70 | 1 | Durham Smythe | 61.60 | 51.88 | 63.91 | 156 | Dolphins |
| 71 | 2 | Tommy Tremble | 61.51 | 56.21 | 60.88 | 303 | Panthers |
| 72 | 3 | Greg Dulcich | 60.86 | 50.22 | 63.78 | 127 | Giants |
| 73 | 4 | Cade Stover | 60.81 | 55.74 | 60.03 | 192 | Texans |
| 74 | 5 | Tip Reiman | 60.61 | 54.59 | 60.46 | 169 | Cardinals |
| 75 | 6 | Hayden Hurst | 60.22 | 50.92 | 62.26 | 103 | Chargers |
| 76 | 7 | Davis Allen | 59.84 | 52.98 | 60.25 | 176 | Rams |
| 77 | 8 | Eric Saubert | 59.62 | 55.44 | 58.24 | 177 | 49ers |
| 78 | 9 | Drew Sample | 59.05 | 52.80 | 59.05 | 278 | Bengals |
| 79 | 10 | Julian Hill | 57.45 | 46.60 | 60.52 | 228 | Dolphins |

## WR — Wide Receiver

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nico Collins | 89.27 | 88.87 | 85.37 | 447 | Texans |
| 2 | 2 | A.J. Brown | 88.07 | 87.51 | 84.28 | 473 | Eagles |
| 3 | 3 | Puka Nacua | 87.08 | 87.60 | 82.57 | 370 | Rams |
| 4 | 4 | Justin Jefferson | 86.67 | 86.60 | 82.55 | 700 | Vikings |
| 5 | 5 | Amon-Ra St. Brown | 85.78 | 88.63 | 79.71 | 611 | Lions |
| 6 | 6 | Mike Evans | 85.18 | 87.44 | 79.51 | 463 | Buccaneers |
| 7 | 7 | Ladd McConkey | 84.81 | 82.24 | 82.35 | 553 | Chargers |
| 8 | 8 | Ja'Marr Chase | 84.06 | 85.80 | 78.74 | 745 | Bengals |
| 9 | 9 | Brian Thomas Jr. | 84.05 | 80.12 | 82.50 | 552 | Jaguars |
| 10 | 10 | Drake London | 83.59 | 86.96 | 77.17 | 595 | Falcons |
| 11 | 11 | Tee Higgins | 82.77 | 85.65 | 76.68 | 476 | Bengals |
| 12 | 12 | Terry McLaurin | 81.97 | 82.10 | 77.71 | 717 | Commanders |
| 13 | 13 | Malik Nabers | 81.66 | 85.46 | 74.96 | 600 | Giants |
| 14 | 14 | Zay Flowers | 81.18 | 80.37 | 77.55 | 499 | Ravens |
| 15 | 15 | Jordan Whittington | 80.74 | 68.09 | 85.01 | 129 | Rams |
| 16 | 16 | Jameson Williams | 80.52 | 73.05 | 81.33 | 535 | Lions |
| 17 | 17 | Khalil Shakir | 80.52 | 77.38 | 78.44 | 495 | Bills |
| 18 | 18 | George Pickens | 80.40 | 77.13 | 78.42 | 498 | Steelers |
| 19 | 19 | CeeDee Lamb | 80.37 | 76.53 | 78.76 | 566 | Cowboys |
| 20 | 20 | Tyreek Hill | 80.28 | 72.22 | 81.48 | 580 | Dolphins |
| 21 | 21 | Chris Godwin | 80.12 | 80.52 | 75.68 | 265 | Buccaneers |
| 22 | 22 | DeVonta Smith | 80.04 | 79.33 | 76.34 | 505 | Eagles |

### Good (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Brandon Aiyuk | 79.98 | 70.95 | 81.83 | 225 | 49ers |
| 24 | 2 | Chris Olave | 79.95 | 76.36 | 78.17 | 200 | Saints |
| 25 | 3 | Josh Downs | 79.38 | 78.84 | 75.58 | 381 | Colts |
| 26 | 4 | Jauan Jennings | 79.37 | 79.26 | 75.28 | 459 | 49ers |
| 27 | 5 | Marvin Harrison Jr. | 79.09 | 76.58 | 76.60 | 579 | Cardinals |
| 28 | 6 | Jaxon Smith-Njigba | 78.90 | 81.00 | 73.33 | 666 | Seahawks |
| 29 | 7 | D.K. Metcalf | 78.88 | 73.83 | 78.08 | 590 | Seahawks |
| 30 | 8 | Jakobi Meyers | 78.35 | 77.43 | 74.79 | 627 | Raiders |
| 31 | 9 | Tylan Wallace | 78.29 | 62.94 | 84.35 | 148 | Ravens |
| 32 | 10 | Davante Adams | 78.22 | 75.03 | 76.18 | 557 | Jets |
| 33 | 11 | Christian Watson | 77.99 | 66.08 | 81.76 | 295 | Packers |
| 34 | 12 | Garrett Wilson | 77.97 | 78.90 | 73.19 | 691 | Jets |
| 35 | 13 | Jalen Coker | 77.52 | 68.59 | 79.31 | 297 | Panthers |
| 36 | 14 | DJ Moore | 77.49 | 73.50 | 75.98 | 722 | Bears |
| 37 | 15 | DeAndre Hopkins | 77.49 | 75.54 | 74.63 | 419 | Chiefs |
| 38 | 16 | Jaylen Waddle | 77.48 | 71.24 | 77.47 | 513 | Dolphins |
| 39 | 17 | Rashid Shaheed | 77.26 | 65.08 | 81.22 | 181 | Saints |
| 40 | 18 | Stefon Diggs | 77.21 | 75.05 | 74.48 | 282 | Texans |
| 41 | 19 | Alec Pierce | 77.19 | 72.26 | 76.31 | 485 | Colts |
| 42 | 20 | Jerry Jeudy | 77.13 | 73.50 | 75.39 | 757 | Browns |
| 43 | 21 | Jayden Reed | 77.03 | 70.35 | 77.32 | 430 | Packers |
| 44 | 22 | Darnell Mooney | 77.00 | 73.32 | 75.29 | 557 | Falcons |
| 45 | 23 | Deebo Samuel | 76.89 | 69.48 | 77.67 | 405 | 49ers |
| 46 | 24 | Marvin Mims Jr. | 76.71 | 64.71 | 80.54 | 198 | Broncos |
| 47 | 25 | Kalif Raymond | 76.71 | 65.43 | 80.06 | 158 | Lions |
| 48 | 26 | Calvin Ridley | 76.29 | 72.91 | 74.37 | 582 | Titans |
| 49 | 27 | Jordan Addison | 76.20 | 72.30 | 74.64 | 600 | Vikings |
| 50 | 28 | Keon Coleman | 76.08 | 66.49 | 78.31 | 404 | Bills |
| 51 | 29 | Courtland Sutton | 75.79 | 75.43 | 71.87 | 650 | Broncos |
| 52 | 30 | Cooper Kupp | 75.78 | 70.24 | 75.31 | 455 | Rams |
| 53 | 31 | Tank Dell | 75.42 | 71.85 | 73.63 | 467 | Texans |
| 54 | 32 | Rashod Bateman | 75.40 | 70.49 | 74.51 | 522 | Ravens |
| 55 | 33 | Amari Cooper | 75.16 | 67.80 | 75.90 | 451 | Bills |
| 56 | 34 | Michael Pittman Jr. | 74.75 | 71.32 | 72.87 | 511 | Colts |
| 57 | 35 | Adam Thielen | 74.59 | 73.40 | 71.21 | 319 | Panthers |
| 58 | 36 | Tutu Atwell | 74.46 | 68.23 | 74.44 | 277 | Rams |
| 59 | 37 | KaVontae Turpin | 74.27 | 66.32 | 75.41 | 210 | Cowboys |
| 60 | 38 | Christian Kirk | 74.26 | 65.96 | 75.62 | 230 | Jaguars |

### Starter (83 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 61 | 1 | Jermaine Burton | 73.87 | 59.36 | 79.37 | 100 | Bengals |
| 62 | 2 | Noah Brown | 73.79 | 67.15 | 74.05 | 295 | Commanders |
| 63 | 3 | Demario Douglas | 73.57 | 68.50 | 72.79 | 477 | Patriots |
| 64 | 4 | Tim Patrick | 73.49 | 65.78 | 74.47 | 371 | Lions |
| 65 | 5 | Dyami Brown | 73.43 | 64.91 | 74.94 | 377 | Commanders |
| 66 | 6 | Romeo Doubs | 73.07 | 68.44 | 71.99 | 406 | Packers |
| 67 | 7 | KhaDarel Hodge | 72.86 | 61.20 | 76.46 | 121 | Falcons |
| 68 | 8 | Calvin Austin III | 72.46 | 63.04 | 74.58 | 423 | Steelers |
| 69 | 9 | Xavier Worthy | 72.37 | 68.32 | 70.90 | 603 | Chiefs |
| 70 | 10 | Joshua Palmer | 72.32 | 65.64 | 72.61 | 428 | Chargers |
| 71 | 11 | Demarcus Robinson | 72.13 | 64.84 | 72.83 | 618 | Rams |
| 72 | 12 | Quentin Johnston | 72.08 | 66.20 | 71.83 | 464 | Chargers |
| 73 | 13 | Mike Williams | 71.92 | 58.98 | 76.38 | 366 | Steelers |
| 74 | 14 | Diontae Johnson | 71.63 | 64.02 | 72.53 | 277 | Texans |
| 75 | 15 | Ricky Pearsall | 71.54 | 62.73 | 73.24 | 324 | 49ers |
| 76 | 16 | Marquez Valdes-Scantling | 71.46 | 61.17 | 74.15 | 315 | Saints |
| 77 | 17 | Rome Odunze | 71.39 | 63.80 | 72.28 | 677 | Bears |
| 78 | 18 | Keenan Allen | 71.34 | 64.27 | 71.89 | 596 | Bears |
| 79 | 19 | Tyler Lockett | 71.24 | 65.03 | 71.22 | 589 | Seahawks |
| 80 | 20 | Olamide Zaccheaus | 71.19 | 67.19 | 69.69 | 403 | Commanders |
| 81 | 21 | Rakim Jarrett | 71.12 | 59.79 | 74.50 | 117 | Buccaneers |
| 82 | 22 | Bo Melton | 70.92 | 61.35 | 73.13 | 118 | Packers |
| 83 | 23 | Jalen Brooks | 70.74 | 59.52 | 74.06 | 236 | Cowboys |
| 84 | 24 | Nick Westbrook-Ikhine | 70.73 | 62.87 | 71.80 | 469 | Titans |
| 85 | 25 | Josh Reynolds | 70.72 | 60.69 | 73.24 | 220 | Jaguars |
| 86 | 26 | Tyler Johnson | 70.66 | 64.00 | 70.93 | 215 | Rams |
| 87 | 27 | JuJu Smith-Schuster | 70.61 | 60.00 | 73.52 | 321 | Chiefs |
| 88 | 28 | Dontayvion Wicks | 70.59 | 63.69 | 71.02 | 345 | Packers |
| 89 | 29 | Michael Wilson | 70.51 | 62.62 | 71.60 | 538 | Cardinals |
| 90 | 30 | Devaughn Vele | 70.44 | 66.76 | 68.73 | 349 | Broncos |
| 91 | 31 | Ray-Ray McCloud III | 70.42 | 62.57 | 71.49 | 598 | Falcons |
| 92 | 32 | Kendrick Bourne | 70.13 | 61.71 | 71.58 | 317 | Patriots |
| 93 | 33 | Greg Dortch | 70.09 | 62.62 | 70.91 | 299 | Cardinals |
| 94 | 34 | Mack Hollins | 70.02 | 61.39 | 71.61 | 495 | Bills |
| 95 | 35 | Nelson Agholor | 69.89 | 62.62 | 70.57 | 250 | Ravens |
| 96 | 36 | Curtis Samuel | 69.83 | 64.28 | 69.36 | 263 | Bills |
| 97 | 37 | Darius Slayton | 69.81 | 59.07 | 72.81 | 575 | Giants |
| 98 | 38 | Adonai Mitchell | 69.49 | 58.78 | 72.46 | 221 | Colts |
| 99 | 39 | Brandin Cooks | 69.44 | 62.61 | 69.82 | 317 | Cowboys |
| 100 | 40 | Chris Conley | 69.30 | 59.91 | 71.39 | 124 | 49ers |
| 101 | 41 | Kayshon Boutte | 69.14 | 61.23 | 70.24 | 507 | Patriots |
| 102 | 42 | Cedrick Wilson Jr. | 69.08 | 62.02 | 69.62 | 220 | Saints |
| 103 | 43 | Allen Lazard | 68.97 | 62.23 | 69.30 | 451 | Jets |
| 104 | 44 | Ryan Flournoy | 68.96 | 61.56 | 69.72 | 103 | Cowboys |
| 105 | 45 | Gabe Davis | 68.58 | 55.26 | 73.29 | 264 | Jaguars |
| 106 | 46 | Tyler Boyd | 68.57 | 60.00 | 70.12 | 464 | Titans |
| 107 | 47 | Dante Pettis | 68.50 | 64.08 | 67.28 | 104 | Saints |
| 108 | 48 | Cedric Tillman | 68.49 | 62.43 | 68.36 | 300 | Browns |
| 109 | 49 | Jalen Nailor | 68.35 | 59.41 | 70.15 | 462 | Vikings |
| 110 | 50 | Jalen McMillan | 68.14 | 60.65 | 68.96 | 430 | Buccaneers |
| 111 | 51 | Jalen Tolbert | 68.08 | 60.67 | 68.86 | 597 | Cowboys |
| 112 | 52 | Trey Palmer | 67.97 | 59.04 | 69.76 | 188 | Buccaneers |
| 113 | 53 | Tre Tucker | 67.81 | 57.50 | 70.51 | 683 | Raiders |
| 114 | 54 | Kevin Austin Jr. | 67.60 | 57.74 | 70.00 | 210 | Saints |
| 115 | 55 | Devin Duvernay | 67.52 | 59.68 | 68.58 | 141 | Jaguars |
| 116 | 56 | David Moore | 67.29 | 62.28 | 66.46 | 358 | Panthers |
| 117 | 57 | Wan'Dale Robinson | 67.07 | 63.29 | 65.42 | 618 | Giants |
| 118 | 58 | Simi Fehoko | 66.82 | 57.73 | 68.71 | 136 | Chargers |
| 119 | 59 | Ryan Miller | 66.76 | 59.52 | 67.42 | 151 | Buccaneers |
| 120 | 60 | John Metchie III | 66.53 | 59.52 | 67.03 | 314 | Texans |
| 121 | 61 | Parker Washington | 66.48 | 59.77 | 66.79 | 404 | Jaguars |
| 122 | 62 | DJ Turner | 66.45 | 56.89 | 68.66 | 246 | Raiders |
| 123 | 63 | Brandon Powell | 66.32 | 59.51 | 66.70 | 130 | Vikings |
| 124 | 64 | Justin Watson | 66.05 | 54.59 | 69.52 | 430 | Chiefs |
| 125 | 65 | Robert Woods | 66.00 | 59.10 | 66.44 | 226 | Texans |
| 126 | 66 | K.J. Osborn | 65.63 | 56.37 | 67.63 | 143 | Commanders |
| 127 | 67 | Van Jefferson | 65.61 | 57.71 | 66.71 | 442 | Steelers |
| 128 | 68 | Troy Franklin | 65.23 | 56.18 | 67.09 | 297 | Broncos |
| 129 | 69 | Elijah Moore | 65.13 | 58.54 | 65.36 | 623 | Browns |
| 130 | 70 | Xavier Hutchinson | 65.01 | 58.45 | 65.21 | 326 | Texans |
| 131 | 71 | Jahan Dotson | 64.91 | 54.56 | 67.65 | 492 | Eagles |
| 132 | 72 | Zay Jones | 64.83 | 56.36 | 66.31 | 184 | Cardinals |
| 133 | 73 | Luke McCaffrey | 64.73 | 56.27 | 66.21 | 283 | Commanders |
| 134 | 74 | Malik Washington | 64.46 | 58.53 | 64.25 | 270 | Dolphins |
| 135 | 75 | Jake Bobo | 64.17 | 57.71 | 64.31 | 163 | Seahawks |
| 136 | 76 | Xavier Legette | 64.15 | 59.43 | 63.13 | 441 | Panthers |
| 137 | 77 | Xavier Gipson | 63.90 | 54.79 | 65.81 | 138 | Jets |
| 138 | 78 | Andrei Iosivas | 63.26 | 52.92 | 65.99 | 620 | Bengals |
| 139 | 79 | Jalin Hyatt | 63.18 | 52.92 | 65.86 | 230 | Giants |
| 140 | 80 | Jonathan Mingo | 62.90 | 53.36 | 65.10 | 285 | Cowboys |
| 141 | 81 | Jamison Crowder | 62.82 | 56.62 | 62.79 | 127 | Commanders |
| 142 | 82 | Sterling Shepard | 62.56 | 55.51 | 63.09 | 393 | Buccaneers |
| 143 | 83 | Michael Woods II | 62.41 | 53.55 | 64.15 | 211 | Browns |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 144 | 1 | Johnny Wilson | 60.09 | 54.72 | 59.50 | 177 | Eagles |
| 145 | 2 | Mason Tipton | 58.21 | 52.57 | 57.81 | 253 | Saints |
| 146 | 3 | Ja'Lynn Polk | 56.99 | 49.15 | 58.05 | 272 | Patriots |
