# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:26:32Z
- **Requested analysis_year:** 2024 (clamped to 2024)
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
| 2 | 2 | Erik McCoy | 98.09 | 94.20 | 96.52 | 293 | Saints |
| 3 | 3 | Frank Ragnow | 92.86 | 86.10 | 93.20 | 1129 | Lions |
| 4 | 4 | Tyler Linderbaum | 88.56 | 79.90 | 90.17 | 1227 | Ravens |
| 5 | 5 | Zach Frazier | 86.64 | 77.90 | 88.30 | 1021 | Steelers |
| 6 | 6 | Drew Dalman | 85.96 | 78.80 | 86.56 | 554 | Falcons |
| 7 | 7 | Hjalte Froholdt | 84.43 | 76.10 | 85.81 | 1078 | Cardinals |
| 8 | 8 | Aaron Brewer | 82.89 | 73.30 | 85.11 | 1139 | Dolphins |
| 9 | 9 | Joe Tippmann | 82.83 | 73.40 | 84.95 | 1067 | Jets |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Connor McGovern | 79.37 | 69.50 | 81.78 | 1164 | Bills |
| 11 | 2 | Coleman Shelton | 77.25 | 66.40 | 80.31 | 1121 | Bears |
| 12 | 3 | Jake Brendel | 77.11 | 65.00 | 81.02 | 1072 | 49ers |
| 13 | 4 | Ryan Kelly | 76.99 | 67.00 | 79.48 | 601 | Colts |
| 14 | 5 | Cam Jurgens | 76.67 | 67.30 | 78.75 | 1217 | Eagles |
| 15 | 6 | Cooper Beebe | 76.32 | 65.40 | 79.44 | 1059 | Cowboys |
| 16 | 7 | Alex Forsyth | 76.25 | 66.50 | 78.59 | 292 | Broncos |
| 17 | 8 | Luke Wattenberg | 75.51 | 64.30 | 78.81 | 864 | Broncos |
| 18 | 9 | Garrett Bradbury | 75.11 | 62.80 | 79.15 | 1191 | Vikings |
| 19 | 10 | Ethan Pocic | 75.02 | 63.60 | 78.46 | 1073 | Browns |
| 20 | 11 | Tyler Biadasz | 75.00 | 64.20 | 78.03 | 1166 | Commanders |
| 21 | 12 | Juice Scruggs | 74.22 | 63.00 | 77.53 | 944 | Texans |
| 22 | 13 | Austin Corbett | 74.19 | 62.90 | 77.55 | 291 | Panthers |
| 23 | 14 | Jarrett Patterson | 74.02 | 64.10 | 76.47 | 688 | Texans |

### Starter (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Ted Karras | 73.77 | 64.10 | 76.05 | 1136 | Bengals |
| 25 | 2 | John Michael Schmitz Jr. | 73.58 | 61.40 | 77.53 | 983 | Giants |
| 26 | 3 | Bradley Bozeman | 73.08 | 61.20 | 76.83 | 1112 | Chargers |
| 27 | 4 | Brady Christensen | 72.57 | 63.60 | 74.38 | 399 | Panthers |
| 28 | 5 | Danny Pinter | 72.52 | 68.60 | 70.96 | 138 | Colts |
| 29 | 6 | Olusegun Oluwatimi | 72.17 | 64.20 | 73.31 | 435 | Seahawks |
| 30 | 7 | Ryan Neuzil | 72.13 | 58.50 | 77.05 | 578 | Falcons |
| 31 | 8 | David Andrews | 71.11 | 58.70 | 75.22 | 193 | Patriots |
| 32 | 9 | Beaux Limmer | 70.73 | 55.50 | 76.71 | 1040 | Rams |
| 33 | 10 | Mitch Morse | 70.08 | 57.30 | 74.44 | 1021 | Jaguars |
| 34 | 11 | Graham Barton | 69.02 | 55.60 | 73.80 | 1111 | Buccaneers |
| 35 | 12 | Corey Levin | 68.42 | 55.50 | 72.86 | 133 | Titans |
| 36 | 13 | Lloyd Cushenberry III | 68.27 | 55.40 | 72.69 | 499 | Titans |
| 37 | 14 | Andre James | 67.66 | 55.60 | 71.53 | 702 | Raiders |
| 38 | 15 | Josh Myers | 66.68 | 54.20 | 70.84 | 1067 | Packers |
| 39 | 16 | Daniel Brunskill | 65.47 | 55.30 | 68.08 | 684 | Titans |
| 40 | 17 | Ryan McCollum | 64.60 | 50.30 | 69.97 | 153 | Steelers |
| 41 | 18 | Shane Lemieux | 64.33 | 51.10 | 68.98 | 337 | Saints |
| 42 | 19 | Sedrick Van Pran-Granger | 62.78 | 54.60 | 64.07 | 125 | Bills |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Pat Surtain II | 89.65 | 85.10 | 88.51 | 1054 | Broncos |
| 2 | 2 | Derek Stingley Jr. | 86.68 | 84.40 | 86.79 | 1119 | Texans |
| 3 | 3 | Quinyon Mitchell | 84.36 | 79.00 | 83.76 | 1104 | Eagles |
| 4 | 4 | Christian Benford | 83.98 | 78.60 | 85.26 | 1046 | Bills |
| 5 | 5 | Garrett Williams | 83.78 | 83.70 | 83.21 | 778 | Cardinals |
| 6 | 6 | Marlon Humphrey | 83.57 | 81.00 | 82.89 | 1000 | Ravens |
| 7 | 7 | Trent McDuffie | 83.33 | 80.70 | 82.10 | 1132 | Chiefs |
| 8 | 8 | Kamari Lassiter | 81.58 | 77.50 | 81.11 | 906 | Texans |
| 9 | 9 | Cooper DeJean | 80.79 | 79.00 | 77.81 | 830 | Eagles |
| 10 | 10 | Darius Slay | 80.34 | 74.20 | 81.45 | 897 | Eagles |
| 11 | 11 | Sauce Gardner | 80.04 | 73.10 | 81.76 | 879 | Jets |
| 12 | 12 | Renardo Green | 80.02 | 74.30 | 80.65 | 675 | 49ers |

### Good (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Jourdan Lewis | 79.88 | 79.00 | 78.95 | 871 | Cowboys |
| 14 | 2 | Christian Gonzalez | 79.86 | 78.20 | 82.20 | 978 | Patriots |
| 15 | 3 | Jamel Dean | 79.40 | 75.40 | 80.83 | 745 | Buccaneers |
| 16 | 4 | Jaylon Johnson | 78.93 | 74.20 | 79.98 | 1031 | Bears |
| 17 | 5 | Terell Smith | 78.77 | 76.50 | 82.61 | 207 | Bears |
| 18 | 6 | Jaire Alexander | 78.56 | 78.30 | 82.02 | 361 | Packers |
| 19 | 7 | Tarheeb Still | 78.05 | 74.80 | 78.01 | 826 | Chargers |
| 20 | 8 | Kyler Gordon | 77.82 | 76.00 | 77.62 | 724 | Bears |
| 21 | 9 | Jalen Ramsey | 77.16 | 71.90 | 78.26 | 1027 | Dolphins |
| 22 | 10 | Carlton Davis III | 76.88 | 72.10 | 79.54 | 697 | Lions |
| 23 | 11 | Samuel Womack III | 76.40 | 71.30 | 81.22 | 673 | Colts |
| 24 | 12 | Nate Wiggins | 76.38 | 69.00 | 77.14 | 769 | Ravens |
| 25 | 13 | Byron Murphy Jr. | 76.22 | 73.50 | 76.32 | 1109 | Vikings |
| 26 | 14 | Mike Jackson | 76.09 | 68.10 | 77.84 | 1204 | Panthers |
| 27 | 15 | D.J. Reed | 76.03 | 70.10 | 77.89 | 880 | Jets |
| 28 | 16 | Denzel Ward | 75.78 | 68.90 | 78.16 | 757 | Browns |
| 29 | 17 | DaRon Bland | 75.76 | 71.40 | 80.18 | 436 | Cowboys |
| 30 | 18 | A.J. Terrell | 75.45 | 69.50 | 75.84 | 1085 | Falcons |
| 31 | 19 | Deommodore Lenoir | 75.40 | 71.70 | 74.89 | 922 | 49ers |
| 32 | 20 | Jaylen Watson | 75.34 | 73.50 | 77.00 | 433 | Chiefs |
| 33 | 21 | Kelee Ringo | 75.28 | 69.60 | 80.41 | 127 | Eagles |
| 34 | 22 | Tariq Woolen | 75.21 | 65.70 | 78.65 | 889 | Seahawks |
| 35 | 23 | Devon Witherspoon | 75.19 | 69.20 | 76.11 | 1103 | Seahawks |
| 36 | 24 | Mike Hilton | 75.12 | 68.50 | 76.45 | 737 | Bengals |
| 37 | 25 | Carrington Valentine | 75.06 | 71.40 | 75.17 | 606 | Packers |
| 38 | 26 | Clark Phillips III | 75.01 | 74.10 | 75.74 | 409 | Falcons |
| 39 | 27 | Andru Phillips | 74.66 | 75.80 | 72.66 | 614 | Giants |
| 40 | 28 | Jaylon Jones | 74.60 | 67.90 | 75.63 | 1146 | Colts |
| 41 | 29 | Kris Abrams-Draine | 74.02 | 74.70 | 85.66 | 123 | Broncos |
| 42 | 30 | DJ Turner II | 74.00 | 68.40 | 77.25 | 508 | Bengals |

### Starter (75 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 43 | 1 | Mike Hughes | 73.91 | 71.40 | 74.06 | 720 | Falcons |
| 44 | 2 | Mike Sainristil | 73.74 | 64.50 | 75.73 | 1158 | Commanders |
| 45 | 3 | Isaiah Rodgers | 73.70 | 68.50 | 74.67 | 413 | Eagles |
| 46 | 4 | Joshua Williams | 73.58 | 68.30 | 76.16 | 411 | Chiefs |
| 47 | 5 | Isaiah Bolden | 72.96 | 79.80 | 71.86 | 141 | Patriots |
| 48 | 6 | Troy Hill | 72.72 | 70.60 | 76.64 | 236 | Buccaneers |
| 49 | 7 | Zyon McCollum | 72.68 | 66.10 | 74.47 | 1123 | Buccaneers |
| 50 | 8 | Marcus Jones | 72.45 | 67.10 | 78.92 | 586 | Patriots |
| 51 | 9 | Kenny Moore II | 72.44 | 68.20 | 73.35 | 1013 | Colts |
| 52 | 10 | Cam Taylor-Britt | 72.11 | 64.30 | 76.00 | 1036 | Bengals |
| 53 | 11 | Kader Kohou | 72.06 | 68.90 | 71.36 | 708 | Dolphins |
| 54 | 12 | Kristian Fulton | 71.99 | 66.60 | 74.85 | 827 | Chargers |
| 55 | 13 | Kool-Aid McKinstry | 71.24 | 67.10 | 72.76 | 680 | Saints |
| 56 | 14 | Jarrian Jones | 70.90 | 62.50 | 72.33 | 699 | Jaguars |
| 57 | 15 | Tyson Campbell | 70.73 | 63.90 | 75.33 | 767 | Jaguars |
| 58 | 16 | Kendall Fuller | 70.59 | 62.40 | 75.42 | 556 | Dolphins |
| 59 | 17 | Adoree' Jackson | 70.41 | 64.50 | 73.91 | 426 | Giants |
| 60 | 18 | Cory Trice Jr. | 70.37 | 75.30 | 76.70 | 194 | Steelers |
| 61 | 19 | Amik Robertson | 70.35 | 62.20 | 72.50 | 630 | Lions |
| 62 | 20 | Shaquill Griffin | 70.35 | 61.60 | 76.44 | 597 | Vikings |
| 63 | 21 | Darious Williams | 70.29 | 59.80 | 74.58 | 865 | Rams |
| 64 | 22 | Jonathan Jones | 70.10 | 61.10 | 73.02 | 712 | Patriots |
| 65 | 23 | Stephon Gilmore | 70.00 | 59.20 | 73.71 | 904 | Vikings |
| 66 | 24 | Paulson Adebo | 69.97 | 63.90 | 76.11 | 436 | Saints |
| 67 | 25 | Cobie Durant | 69.68 | 61.50 | 73.42 | 843 | Rams |
| 68 | 26 | Tyrique Stevenson | 69.41 | 58.40 | 73.57 | 810 | Bears |
| 69 | 27 | Ahkello Witherspoon | 69.35 | 62.60 | 73.22 | 598 | Rams |
| 70 | 28 | Darrell Baker Jr. | 68.88 | 62.80 | 72.82 | 626 | Titans |
| 71 | 29 | Chamarri Conner | 68.87 | 62.80 | 68.75 | 679 | Chiefs |
| 72 | 30 | Charvarius Ward | 68.78 | 58.20 | 74.12 | 694 | 49ers |
| 73 | 31 | Josh Newton | 68.74 | 60.90 | 72.74 | 504 | Bengals |
| 74 | 32 | Amani Oruwariye | 68.67 | 63.90 | 73.86 | 286 | Cowboys |
| 75 | 33 | Keisean Nixon | 68.53 | 60.70 | 70.36 | 1077 | Packers |
| 76 | 34 | Beanie Bishop Jr. | 68.36 | 57.10 | 71.70 | 550 | Steelers |
| 77 | 35 | Alex Austin | 68.36 | 63.90 | 77.58 | 234 | Patriots |
| 78 | 36 | Ja'Quan McMillian | 67.99 | 63.00 | 70.89 | 918 | Broncos |
| 79 | 37 | Jakorian Bennett | 67.27 | 61.00 | 73.41 | 459 | Raiders |
| 80 | 38 | Trevon Diggs | 67.22 | 60.30 | 75.01 | 683 | Cowboys |
| 81 | 39 | Cor'Dale Flott | 67.08 | 61.70 | 70.23 | 666 | Giants |
| 82 | 40 | Ronald Darby | 67.04 | 59.30 | 73.52 | 659 | Jaguars |
| 83 | 41 | Avonte Maddox | 66.98 | 60.40 | 72.30 | 352 | Eagles |
| 84 | 42 | Jaycee Horn | 66.97 | 57.90 | 73.85 | 1034 | Panthers |
| 85 | 43 | Joey Porter Jr. | 66.74 | 56.30 | 69.53 | 1038 | Steelers |
| 86 | 44 | Myles Bryant | 66.64 | 62.70 | 70.00 | 156 | Texans |
| 87 | 45 | Cam Hart | 66.62 | 58.10 | 73.03 | 502 | Chargers |
| 88 | 46 | Fabian Moreau | 66.51 | 60.50 | 73.41 | 104 | Vikings |
| 89 | 47 | Marshon Lattimore | 66.48 | 58.00 | 74.43 | 687 | Commanders |
| 90 | 48 | Eric Stokes | 66.41 | 62.40 | 70.90 | 588 | Packers |
| 91 | 49 | Ja'Sir Taylor | 66.08 | 57.70 | 71.92 | 353 | Chargers |
| 92 | 50 | Kaiir Elam | 65.80 | 61.40 | 73.10 | 359 | Bills |
| 93 | 51 | Israel Mukuamu | 65.63 | 51.90 | 71.10 | 201 | Cowboys |
| 94 | 52 | Taron Johnson | 65.62 | 55.90 | 69.11 | 785 | Bills |
| 95 | 53 | Jarvis Brownlee Jr. | 65.61 | 55.90 | 67.92 | 911 | Titans |
| 96 | 54 | Brandin Echols | 65.60 | 61.20 | 69.77 | 406 | Jets |
| 97 | 55 | Chidobe Awuzie | 65.53 | 58.40 | 72.89 | 373 | Titans |
| 98 | 56 | Rasul Douglas | 65.42 | 51.60 | 70.47 | 997 | Bills |
| 99 | 57 | Asante Samuel Jr. | 65.27 | 58.60 | 71.91 | 234 | Chargers |
| 100 | 58 | Christian Roland-Wallace | 65.26 | 61.10 | 71.71 | 197 | Chiefs |
| 101 | 59 | D'Angelo Ross | 64.91 | 66.60 | 69.61 | 184 | Texans |
| 102 | 60 | Dax Hill | 64.84 | 68.00 | 70.42 | 262 | Bengals |
| 103 | 61 | Isaac Yiadom | 64.82 | 55.30 | 70.91 | 488 | 49ers |
| 104 | 62 | Nate Hobbs | 64.76 | 61.50 | 68.06 | 554 | Raiders |
| 105 | 63 | Roger McCreary | 64.69 | 58.40 | 66.28 | 652 | Titans |
| 106 | 64 | Max Melton | 64.59 | 57.30 | 65.29 | 565 | Cardinals |
| 107 | 65 | Dee Alford | 64.36 | 55.90 | 67.91 | 724 | Falcons |
| 108 | 66 | Jack Jones | 64.20 | 52.90 | 70.11 | 1047 | Raiders |
| 109 | 67 | Deantre Prince | 64.16 | 65.00 | 73.21 | 101 | Jaguars |
| 110 | 68 | Greg Newsome II | 63.99 | 54.00 | 69.41 | 571 | Browns |
| 111 | 69 | Tre'Davious White | 63.63 | 54.80 | 73.29 | 445 | Ravens |
| 112 | 70 | Terrion Arnold | 63.51 | 50.20 | 68.21 | 1021 | Lions |
| 113 | 71 | Cameron Mitchell | 63.37 | 55.30 | 66.30 | 371 | Browns |
| 114 | 72 | James Pierre | 62.89 | 54.80 | 70.99 | 207 | Steelers |
| 115 | 73 | Nazeeh Johnson | 62.48 | 52.70 | 64.84 | 547 | Chiefs |
| 116 | 74 | Starling Thomas V | 62.15 | 60.90 | 61.02 | 817 | Cardinals |
| 117 | 75 | Deonte Banks | 62.06 | 50.30 | 68.30 | 788 | Giants |

### Rotation/backup (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 118 | 1 | Riley Moss | 61.73 | 56.00 | 65.91 | 912 | Broncos |
| 119 | 2 | Montaric Brown | 61.72 | 61.60 | 62.54 | 855 | Jaguars |
| 120 | 3 | Ka'dar Hollman | 61.63 | 57.00 | 69.47 | 116 | Texans |
| 121 | 4 | Tre Brown | 61.45 | 53.50 | 69.84 | 290 | Seahawks |
| 122 | 5 | Sean Murphy-Bunting | 61.32 | 53.10 | 66.07 | 725 | Cardinals |
| 123 | 6 | Martin Emerson Jr. | 61.31 | 48.40 | 65.75 | 827 | Browns |
| 124 | 7 | Benjamin St-Juste | 61.10 | 46.50 | 67.94 | 859 | Commanders |
| 125 | 8 | Greg Stroman Jr. | 60.59 | 59.40 | 69.71 | 130 | Giants |
| 126 | 9 | Josh Blackwell | 59.92 | 61.20 | 66.77 | 102 | Bears |
| 127 | 10 | Josh Jobe | 59.39 | 49.90 | 70.09 | 443 | Seahawks |
| 128 | 11 | Shemar Jean-Charles | 59.28 | 55.60 | 70.06 | 143 | Saints |
| 129 | 12 | Donte Jackson | 59.15 | 45.10 | 66.70 | 832 | Steelers |
| 130 | 13 | Storm Duck | 59.06 | 54.80 | 62.63 | 359 | Dolphins |
| 131 | 14 | Michael Carter II | 58.42 | 47.90 | 63.82 | 285 | Jets |
| 132 | 15 | Ja'Marcus Ingram | 58.24 | 48.70 | 69.06 | 217 | Bills |
| 133 | 16 | Cameron Sutton | 57.54 | 45.00 | 65.37 | 273 | Steelers |
| 134 | 17 | Marco Wilson | 57.40 | 43.70 | 64.92 | 242 | Bengals |
| 135 | 18 | Darnay Holmes | 57.25 | 49.50 | 65.22 | 298 | Raiders |
| 136 | 19 | Noah Igbinoghene | 56.84 | 49.60 | 63.87 | 971 | Commanders |
| 137 | 20 | Chau Smith-Wade | 55.86 | 49.60 | 62.74 | 301 | Panthers |
| 138 | 21 | Michael Davis | 55.75 | 40.80 | 66.65 | 139 | Commanders |
| 139 | 22 | Josh Wallace | 55.67 | 43.40 | 62.62 | 165 | Rams |
| 140 | 23 | Nick McCloud | 55.53 | 46.20 | 63.66 | 224 | 49ers |
| 141 | 24 | Emmanuel Forbes | 54.77 | 39.90 | 67.75 | 160 | Rams |
| 142 | 25 | Decamerion Richardson | 54.12 | 42.20 | 63.78 | 559 | Raiders |
| 143 | 26 | Caleb Farley | 54.07 | 56.90 | 55.77 | 169 | Panthers |
| 144 | 27 | Alontae Taylor | 54.00 | 34.20 | 64.22 | 1075 | Saints |
| 145 | 28 | Dane Jackson | 53.77 | 36.40 | 66.38 | 282 | Panthers |
| 146 | 29 | L'Jarius Sneed | 52.55 | 36.80 | 64.77 | 301 | Titans |
| 147 | 30 | Kindle Vildor | 50.82 | 45.00 | 55.54 | 316 | Lions |
| 148 | 31 | Tyrek Funderburk | 49.58 | 55.70 | 52.63 | 168 | Buccaneers |
| 149 | 32 | Andrew Booth Jr. | 45.25 | 28.70 | 63.00 | 118 | Cowboys |
| 150 | 33 | Caelen Carson | 45.00 | 36.70 | 58.80 | 252 | Cowboys |
| 151 | 34 | Cam Smith | 45.00 | 30.10 | 59.65 | 133 | Dolphins |
| 152 | 35 | Nehemiah Pritchett | 45.00 | 35.60 | 60.59 | 151 | Seahawks |

## DI — Defensive Interior

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Quinnen Williams | 89.33 | 87.95 | 86.77 | 722 | Jets |
| 2 | 2 | Kobie Turner | 87.16 | 84.01 | 85.09 | 919 | Rams |
| 3 | 3 | Dexter Lawrence | 85.27 | 89.80 | 81.01 | 551 | Giants |
| 4 | 4 | DeForest Buckner | 84.64 | 85.48 | 82.37 | 579 | Colts |
| 5 | 5 | Jeffery Simmons | 83.96 | 86.19 | 80.65 | 806 | Titans |
| 6 | 6 | Leonard Williams | 83.88 | 85.41 | 80.16 | 750 | Seahawks |
| 7 | 7 | Chris Jones | 83.68 | 85.69 | 78.17 | 886 | Chiefs |
| 8 | 8 | Zach Sieler | 83.13 | 78.15 | 83.26 | 749 | Dolphins |
| 9 | 9 | Jalen Carter | 82.89 | 86.10 | 76.58 | 1026 | Eagles |
| 10 | 10 | Cameron Heyward | 82.48 | 77.13 | 83.35 | 838 | Steelers |
| 11 | 11 | Vita Vea | 81.81 | 82.14 | 78.00 | 756 | Buccaneers |
| 12 | 12 | Jalen Redmond | 81.11 | 75.48 | 83.63 | 236 | Vikings |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Christian Wilkins | 79.98 | 81.89 | 80.43 | 246 | Raiders |
| 14 | 2 | Ed Oliver | 79.55 | 71.40 | 81.60 | 727 | Bills |
| 15 | 3 | Alim McNeill | 77.48 | 81.05 | 72.70 | 631 | Lions |
| 16 | 4 | Grover Stewart | 77.23 | 74.46 | 76.68 | 690 | Colts |
| 17 | 5 | Milton Williams | 77.16 | 67.13 | 79.68 | 628 | Eagles |
| 18 | 6 | Devonte Wyatt | 76.99 | 65.43 | 81.71 | 366 | Packers |
| 19 | 7 | Jordan Davis | 76.74 | 76.16 | 73.74 | 430 | Eagles |
| 20 | 8 | Zach Allen | 76.72 | 64.32 | 81.61 | 1031 | Broncos |
| 21 | 9 | T'Vondre Sweat | 76.42 | 79.33 | 70.32 | 699 | Titans |
| 22 | 10 | Michael Pierce | 76.41 | 74.09 | 78.49 | 254 | Ravens |
| 23 | 11 | Kenny Clark | 75.92 | 67.94 | 77.08 | 725 | Packers |
| 24 | 12 | Braden Fiske | 75.49 | 56.26 | 84.14 | 700 | Rams |
| 25 | 13 | Osa Odighizuwa | 75.47 | 65.39 | 78.03 | 859 | Cowboys |
| 26 | 14 | Calais Campbell | 75.34 | 56.76 | 84.83 | 616 | Dolphins |
| 27 | 15 | Grady Jarrett | 75.06 | 65.79 | 79.72 | 744 | Falcons |
| 28 | 16 | B.J. Hill | 74.52 | 69.05 | 75.19 | 710 | Bengals |
| 29 | 17 | John Franklin-Myers | 74.48 | 62.08 | 78.96 | 569 | Broncos |
| 30 | 18 | Jonathan Allen | 74.40 | 61.38 | 82.35 | 421 | Commanders |

### Starter (77 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Christian Barmore | 73.45 | 65.23 | 82.51 | 123 | Patriots |
| 32 | 2 | Sebastian Joseph-Day | 73.41 | 60.20 | 78.25 | 483 | Titans |
| 33 | 3 | Daron Payne | 73.09 | 59.08 | 78.26 | 796 | Commanders |
| 34 | 4 | David Onyemata | 73.03 | 62.57 | 76.72 | 567 | Falcons |
| 35 | 5 | D.J. Jones | 72.98 | 60.59 | 77.76 | 510 | Broncos |
| 36 | 6 | Jaquelin Roy | 72.87 | 61.69 | 84.74 | 141 | Patriots |
| 37 | 7 | Shelby Harris | 72.44 | 57.73 | 79.94 | 527 | Browns |
| 38 | 8 | Calijah Kancey | 72.37 | 52.46 | 84.30 | 595 | Buccaneers |
| 39 | 9 | Travis Jones | 72.32 | 66.76 | 72.26 | 675 | Ravens |
| 40 | 10 | DJ Reader | 72.27 | 67.21 | 74.22 | 566 | Lions |
| 41 | 11 | Teair Tart | 72.06 | 65.13 | 73.88 | 378 | Chargers |
| 42 | 12 | Poona Ford | 71.93 | 65.08 | 74.98 | 652 | Chargers |
| 43 | 13 | Desjuan Johnson | 71.67 | 59.39 | 80.59 | 155 | Rams |
| 44 | 14 | Gervon Dexter Sr. | 71.50 | 63.92 | 73.62 | 616 | Bears |
| 45 | 15 | Dalvin Tomlinson | 71.37 | 62.15 | 74.61 | 609 | Browns |
| 46 | 16 | Leonard Taylor III | 71.04 | 59.46 | 77.52 | 261 | Jets |
| 47 | 17 | Keeanu Benton | 70.78 | 69.00 | 67.80 | 671 | Steelers |
| 48 | 18 | Karl Brooks | 70.46 | 55.07 | 76.55 | 459 | Packers |
| 49 | 19 | Jowon Briggs | 70.25 | 71.68 | 78.92 | 133 | Browns |
| 50 | 20 | Javon Hargrave | 70.11 | 58.22 | 80.73 | 104 | 49ers |
| 51 | 21 | Levi Onwuzurike | 69.73 | 65.28 | 69.90 | 697 | Lions |
| 52 | 22 | Jarran Reed | 69.52 | 56.00 | 74.67 | 679 | Seahawks |
| 53 | 23 | Malcolm Roach | 69.08 | 56.25 | 75.72 | 524 | Broncos |
| 54 | 24 | Byron Murphy II | 68.76 | 58.26 | 74.53 | 457 | Seahawks |
| 55 | 25 | Tim Settle | 68.51 | 52.16 | 75.65 | 685 | Texans |
| 56 | 26 | Mario Edwards Jr. | 68.50 | 53.61 | 76.81 | 519 | Texans |
| 57 | 27 | Jer'Zhan Newton | 67.73 | 50.48 | 75.07 | 586 | Commanders |
| 58 | 28 | Dante Stills | 67.65 | 54.51 | 73.59 | 532 | Cardinals |
| 59 | 29 | William Gholston | 67.34 | 50.56 | 74.36 | 205 | Buccaneers |
| 60 | 30 | Elijah Garcia | 67.10 | 58.15 | 81.40 | 143 | Giants |
| 61 | 31 | DaQuan Jones | 66.87 | 59.98 | 69.84 | 629 | Bills |
| 62 | 32 | Naquan Jones | 66.81 | 51.39 | 78.03 | 260 | Cardinals |
| 63 | 33 | Evan Anderson | 66.67 | 56.01 | 74.51 | 267 | 49ers |
| 64 | 34 | Tyler Davis | 66.57 | 47.47 | 75.14 | 354 | Rams |
| 65 | 35 | Folorunso Fatukasi | 66.45 | 47.79 | 77.58 | 366 | Texans |
| 66 | 36 | Larry Ogunjobi | 66.41 | 45.54 | 76.84 | 550 | Steelers |
| 67 | 37 | Bobby Brown III | 66.35 | 59.49 | 69.20 | 513 | Rams |
| 68 | 38 | Zach Harrison | 66.32 | 54.02 | 70.36 | 272 | Falcons |
| 69 | 39 | A'Shawn Robinson | 66.22 | 48.96 | 75.42 | 761 | Panthers |
| 70 | 40 | Roy Robertson-Harris | 66.21 | 49.02 | 73.50 | 398 | Seahawks |
| 71 | 41 | Jeremiah Pharms Jr. | 66.21 | 55.56 | 71.59 | 457 | Patriots |
| 72 | 42 | Roy Lopez | 66.20 | 52.30 | 72.67 | 464 | Cardinals |
| 73 | 43 | DaVon Hamilton | 66.19 | 50.68 | 75.02 | 626 | Jaguars |
| 74 | 44 | Harrison Phillips | 65.99 | 52.89 | 70.55 | 701 | Vikings |
| 75 | 45 | Javon Kinlaw | 65.71 | 52.98 | 72.18 | 695 | Jets |
| 76 | 46 | Morgan Fox | 65.59 | 45.77 | 74.64 | 619 | Chargers |
| 77 | 47 | Khyiris Tonga | 65.07 | 56.45 | 70.97 | 229 | Cardinals |
| 78 | 48 | Kevin Givens | 64.70 | 48.11 | 76.79 | 185 | 49ers |
| 79 | 49 | Thomas Booker IV | 64.54 | 47.01 | 74.62 | 172 | Eagles |
| 80 | 50 | Solomon Thomas | 64.41 | 44.99 | 73.68 | 458 | Jets |
| 81 | 51 | Andrew Billings | 64.38 | 55.93 | 70.84 | 297 | Bears |
| 82 | 52 | Taven Bryan | 64.36 | 51.92 | 68.68 | 340 | Colts |
| 83 | 53 | Bryan Bresee | 64.33 | 48.56 | 70.68 | 708 | Saints |
| 84 | 54 | Maliek Collins | 64.19 | 49.77 | 70.03 | 715 | 49ers |
| 85 | 55 | Adam Butler | 64.09 | 45.66 | 72.21 | 856 | Raiders |
| 86 | 56 | Logan Hall | 64.09 | 53.89 | 66.73 | 571 | Buccaneers |
| 87 | 57 | Kentavius Street | 64.04 | 48.52 | 73.35 | 280 | Falcons |
| 88 | 58 | Da'Shawn Hand | 64.00 | 55.22 | 69.12 | 564 | Dolphins |
| 89 | 59 | Khalil Davis | 64.00 | 46.25 | 77.03 | 209 | 49ers |
| 90 | 60 | Eddie Goldman | 63.86 | 45.01 | 73.34 | 330 | Falcons |
| 91 | 61 | Davon Godchaux | 63.57 | 46.59 | 70.73 | 680 | Patriots |
| 92 | 62 | Jeremiah Ledbetter | 63.35 | 50.75 | 71.70 | 441 | Jaguars |
| 93 | 63 | Jonathan Bullard | 63.30 | 45.32 | 72.10 | 590 | Vikings |
| 94 | 64 | Maurice Hurst | 63.24 | 49.94 | 76.47 | 164 | Browns |
| 95 | 65 | Neville Gallimore | 63.11 | 46.22 | 70.89 | 308 | Rams |
| 96 | 66 | Sheldon Rankins | 63.10 | 49.59 | 73.24 | 287 | Bengals |
| 97 | 67 | Mazi Smith | 63.07 | 47.89 | 69.03 | 524 | Cowboys |
| 98 | 68 | Mike Pennel | 63.06 | 45.35 | 73.63 | 365 | Chiefs |
| 99 | 69 | Khalen Saunders | 62.96 | 48.69 | 70.46 | 460 | Saints |
| 100 | 70 | Greg Gaines | 62.91 | 49.98 | 68.04 | 421 | Buccaneers |
| 101 | 71 | Byron Cowart | 62.74 | 43.61 | 72.73 | 335 | Bears |
| 102 | 72 | Quinton Jefferson | 62.54 | 42.65 | 73.98 | 258 | Bills |
| 103 | 73 | Tershawn Wharton | 62.53 | 50.23 | 68.92 | 733 | Chiefs |
| 104 | 74 | Colby Wooden | 62.39 | 49.88 | 69.01 | 260 | Packers |
| 105 | 75 | Jonah Laulu | 62.38 | 44.97 | 71.79 | 474 | Raiders |
| 106 | 76 | Adetomiwa Adebawore | 62.29 | 52.67 | 71.15 | 137 | Colts |
| 107 | 77 | Nathan Shepherd | 62.06 | 44.49 | 70.09 | 567 | Saints |

### Rotation/backup (60 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 108 | 1 | James Lynch | 61.83 | 45.02 | 70.23 | 243 | Titans |
| 109 | 2 | McKinnley Jackson | 61.81 | 47.46 | 71.12 | 248 | Bengals |
| 110 | 3 | Broderick Washington | 61.76 | 46.85 | 67.53 | 488 | Ravens |
| 111 | 4 | Moro Ojomo | 61.62 | 61.15 | 61.09 | 465 | Eagles |
| 112 | 5 | Zacch Pickens | 61.60 | 47.77 | 71.56 | 228 | Bears |
| 113 | 6 | Austin Johnson | 61.22 | 44.84 | 69.74 | 353 | Bills |
| 114 | 7 | Tommy Togiai | 61.20 | 46.64 | 73.80 | 280 | Texans |
| 115 | 8 | Linval Joseph | 61.14 | 38.10 | 76.45 | 264 | Cowboys |
| 116 | 9 | Jalyn Holmes | 61.03 | 44.72 | 74.40 | 337 | Commanders |
| 117 | 10 | John Jenkins | 60.93 | 39.31 | 71.38 | 606 | Raiders |
| 118 | 11 | Patrick O'Connor | 60.78 | 43.02 | 74.13 | 235 | Lions |
| 119 | 12 | Elijah Chatman | 60.71 | 51.99 | 62.35 | 423 | Giants |
| 120 | 13 | Johnathan Hankins | 60.69 | 39.05 | 72.91 | 389 | Seahawks |
| 121 | 14 | Jerry Tillery | 60.62 | 48.89 | 64.68 | 482 | Vikings |
| 122 | 15 | Bilal Nichols | 60.43 | 44.92 | 72.00 | 173 | Cardinals |
| 123 | 16 | Ta'Quon Graham | 60.43 | 56.16 | 64.31 | 193 | Falcons |
| 124 | 17 | Jordan Jefferson | 60.39 | 60.16 | 65.20 | 151 | Jaguars |
| 125 | 18 | D.J. Davidson | 60.25 | 48.53 | 69.30 | 261 | Giants |
| 126 | 19 | Isaiahh Loudermilk | 60.15 | 49.75 | 64.59 | 255 | Steelers |
| 127 | 20 | Shy Tuttle | 60.06 | 44.29 | 67.39 | 610 | Panthers |
| 128 | 21 | DeShawn Williams | 59.44 | 40.73 | 70.50 | 338 | Panthers |
| 129 | 22 | Carlos Watkins | 59.18 | 46.54 | 69.32 | 228 | Cowboys |
| 130 | 23 | Raekwon Davis | 59.06 | 45.86 | 63.89 | 349 | Colts |
| 131 | 24 | Jordan Phillips | 59.04 | 39.07 | 72.51 | 185 | Bills |
| 132 | 25 | Bruce Hector | 59.01 | 49.71 | 73.54 | 118 | Jets |
| 133 | 26 | Justin Jones | 58.95 | 42.52 | 72.61 | 100 | Cardinals |
| 134 | 27 | Benito Jones | 58.84 | 44.53 | 64.21 | 481 | Dolphins |
| 135 | 28 | John Ridgeway | 58.83 | 44.50 | 66.56 | 263 | Saints |
| 136 | 29 | Jonah Williams | 58.66 | 42.67 | 69.37 | 108 | Lions |
| 137 | 30 | Montravius Adams | 58.55 | 42.05 | 68.72 | 207 | Steelers |
| 138 | 31 | Kurt Hinish | 58.49 | 46.78 | 65.94 | 231 | Texans |
| 139 | 32 | Maason Smith | 58.25 | 45.65 | 68.36 | 384 | Jaguars |
| 140 | 33 | Dean Lowry | 57.89 | 41.45 | 69.89 | 159 | Steelers |
| 141 | 34 | Tyler Lacy | 57.77 | 47.09 | 63.90 | 340 | Jaguars |
| 142 | 35 | Ruke Orhorhoro | 57.44 | 51.46 | 66.08 | 147 | Falcons |
| 143 | 36 | Daniel Ekuale | 57.40 | 46.89 | 65.24 | 723 | Patriots |
| 144 | 37 | Derrick Nnadi | 57.33 | 41.21 | 63.91 | 248 | Chiefs |
| 145 | 38 | Sheldon Day | 56.94 | 42.63 | 68.49 | 339 | Commanders |
| 146 | 39 | Jordan Elliott | 56.79 | 45.26 | 61.29 | 440 | 49ers |
| 147 | 40 | Rakeem Nunez-Roches | 56.60 | 39.31 | 65.22 | 608 | Giants |
| 148 | 41 | Zach Carter | 56.27 | 44.97 | 62.28 | 263 | Raiders |
| 149 | 42 | L.J. Collier | 56.15 | 43.14 | 67.13 | 588 | Cardinals |
| 150 | 43 | C.J. Brewer | 55.82 | 39.90 | 67.17 | 159 | Buccaneers |
| 151 | 44 | Jordon Riley | 55.15 | 43.80 | 64.92 | 248 | Giants |
| 152 | 45 | Jay Tufele | 55.03 | 43.11 | 64.79 | 242 | Bengals |
| 153 | 46 | DeWayne Carter | 54.54 | 42.00 | 64.62 | 315 | Bills |
| 154 | 47 | Otito Ogbonnia | 53.96 | 43.79 | 61.18 | 538 | Chargers |
| 155 | 48 | Ben Stille | 53.78 | 44.43 | 66.73 | 120 | Cardinals |
| 156 | 49 | Darius Robinson | 53.62 | 49.82 | 65.77 | 184 | Cardinals |
| 157 | 50 | Jordan Jackson | 53.60 | 38.00 | 59.83 | 329 | Broncos |
| 158 | 51 | Kris Jenkins | 53.45 | 47.01 | 60.66 | 496 | Bengals |
| 159 | 52 | Jaden Crumedy | 53.32 | 45.46 | 70.66 | 121 | Panthers |
| 160 | 53 | Chris Williams | 53.15 | 39.78 | 63.19 | 367 | Bears |
| 161 | 54 | Eric Johnson | 52.11 | 44.73 | 57.28 | 178 | Patriots |
| 162 | 55 | LaBryan Ray | 51.91 | 41.10 | 55.56 | 626 | Panthers |
| 163 | 56 | Phidarian Mathis | 51.06 | 44.61 | 58.84 | 257 | Jets |
| 164 | 57 | Keondre Coburn | 49.88 | 48.71 | 52.13 | 125 | Titans |
| 165 | 58 | Matthew Butler | 48.84 | 49.70 | 56.07 | 101 | Raiders |
| 166 | 59 | Kalia Davis | 47.62 | 41.72 | 54.98 | 259 | 49ers |
| 167 | 60 | Mekhi Wingo | 47.02 | 44.19 | 50.62 | 177 | Lions |

## ED — Edge

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Bosa | 93.28 | 96.96 | 88.33 | 693 | 49ers |
| 2 | 2 | Will Anderson Jr. | 92.36 | 96.36 | 86.15 | 645 | Texans |
| 3 | 3 | Myles Garrett | 92.08 | 94.59 | 86.44 | 822 | Browns |
| 4 | 4 | Micah Parsons | 91.89 | 90.57 | 90.57 | 694 | Cowboys |
| 5 | 5 | T.J. Watt | 90.39 | 92.51 | 86.17 | 1002 | Steelers |
| 6 | 6 | Jared Verse | 88.53 | 94.25 | 80.55 | 933 | Rams |
| 7 | 7 | Aidan Hutchinson | 88.30 | 94.87 | 85.64 | 280 | Lions |
| 8 | 8 | Khalil Mack | 86.96 | 84.84 | 84.20 | 668 | Chargers |
| 9 | 9 | Greg Rousseau | 86.65 | 89.64 | 81.27 | 861 | Bills |
| 10 | 10 | Rashan Gary | 85.98 | 84.46 | 84.40 | 670 | Packers |
| 11 | 11 | Joey Bosa | 85.92 | 88.02 | 86.04 | 503 | Chargers |
| 12 | 12 | Trey Hendrickson | 85.71 | 80.89 | 85.15 | 823 | Bengals |
| 13 | 13 | Danielle Hunter | 83.36 | 76.50 | 83.77 | 859 | Texans |
| 14 | 14 | Maxx Crosby | 83.15 | 86.49 | 79.21 | 766 | Raiders |
| 15 | 15 | Montez Sweat | 81.56 | 79.80 | 79.05 | 616 | Bears |
| 16 | 16 | Nik Bonitto | 81.34 | 67.34 | 86.50 | 761 | Broncos |
| 17 | 17 | Alex Highsmith | 80.53 | 83.23 | 77.01 | 592 | Steelers |
| 18 | 18 | Yaya Diaby | 80.14 | 78.37 | 77.16 | 841 | Buccaneers |
| 19 | 19 | Von Miller | 80.03 | 66.74 | 87.27 | 332 | Bills |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Chop Robinson | 79.68 | 71.56 | 80.92 | 565 | Dolphins |
| 21 | 2 | Brian Burns | 79.19 | 71.75 | 80.46 | 865 | Giants |
| 22 | 3 | Javon Solomon | 78.92 | 70.12 | 84.54 | 141 | Bills |
| 23 | 4 | Brenton Cox Jr. | 78.06 | 68.76 | 88.91 | 187 | Packers |
| 24 | 5 | Nolan Smith | 77.78 | 71.12 | 78.05 | 725 | Eagles |
| 25 | 6 | Jonathan Greenard | 77.71 | 69.59 | 80.72 | 969 | Vikings |
| 26 | 7 | DeMarcus Lawrence | 77.65 | 81.47 | 77.31 | 167 | Cowboys |
| 27 | 8 | Dondrea Tillman | 77.36 | 65.41 | 85.08 | 275 | Broncos |
| 28 | 9 | Zaven Collins | 77.00 | 73.74 | 75.00 | 600 | Cardinals |
| 29 | 10 | Brandon Graham | 76.87 | 69.73 | 80.40 | 311 | Eagles |
| 30 | 11 | Will McDonald IV | 76.21 | 64.27 | 80.74 | 756 | Jets |
| 31 | 12 | Odafe Oweh | 76.10 | 76.62 | 72.17 | 683 | Ravens |
| 32 | 13 | Tuli Tuipulotu | 76.02 | 72.55 | 74.17 | 774 | Chargers |
| 33 | 14 | Za'Darius Smith | 75.90 | 66.09 | 78.48 | 655 | Lions |
| 34 | 15 | Nick Herbig | 75.57 | 67.28 | 79.87 | 433 | Steelers |
| 35 | 16 | Kyle Van Noy | 75.50 | 58.16 | 82.89 | 696 | Ravens |
| 36 | 17 | Boye Mafe | 74.97 | 70.85 | 74.81 | 607 | Seahawks |
| 37 | 18 | George Karlaftis | 74.95 | 65.98 | 76.77 | 953 | Chiefs |
| 38 | 19 | Jaelan Phillips | 74.60 | 72.69 | 80.73 | 134 | Dolphins |
| 39 | 20 | Jonathon Cooper | 74.21 | 65.75 | 76.26 | 882 | Broncos |

### Starter (60 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 40 | 1 | Andrew Van Ginkel | 73.56 | 57.64 | 80.31 | 973 | Vikings |
| 41 | 2 | Josh Sweat | 73.33 | 65.54 | 74.56 | 775 | Eagles |
| 42 | 3 | Travon Walker | 73.09 | 70.93 | 70.76 | 911 | Jaguars |
| 43 | 4 | Chase Young | 72.88 | 74.90 | 70.12 | 740 | Saints |
| 44 | 5 | James Houston | 72.82 | 56.13 | 88.79 | 141 | Browns |
| 45 | 6 | Carl Granderson | 72.79 | 65.93 | 73.40 | 825 | Saints |
| 46 | 7 | Isaiah McGuire | 72.73 | 70.03 | 75.76 | 469 | Browns |
| 47 | 8 | Arnold Ebiketie | 72.62 | 63.54 | 74.71 | 543 | Falcons |
| 48 | 9 | Michael Hoecht | 72.22 | 60.74 | 75.70 | 705 | Rams |
| 49 | 10 | Byron Young | 71.75 | 58.98 | 76.10 | 936 | Rams |
| 50 | 11 | Haason Reddick | 70.50 | 53.49 | 81.10 | 392 | Jets |
| 51 | 12 | Jadeveon Clowney | 70.36 | 66.38 | 71.29 | 650 | Panthers |
| 52 | 13 | Harold Landry III | 70.35 | 59.61 | 73.34 | 878 | Titans |
| 53 | 14 | Dorance Armstrong | 70.32 | 60.32 | 72.82 | 747 | Commanders |
| 54 | 15 | Uchenna Nwosu | 70.08 | 63.22 | 75.89 | 190 | Seahawks |
| 55 | 16 | Cameron Jordan | 69.76 | 59.31 | 72.76 | 565 | Saints |
| 56 | 17 | Kayvon Thibodeaux | 69.27 | 68.67 | 68.54 | 593 | Giants |
| 57 | 18 | Matthew Judon | 69.25 | 49.44 | 82.11 | 655 | Falcons |
| 58 | 19 | Dante Fowler Jr. | 69.09 | 53.65 | 75.21 | 642 | Commanders |
| 59 | 20 | Darrell Taylor | 69.08 | 53.23 | 76.16 | 374 | Bears |
| 60 | 21 | Jonah Elliss | 68.36 | 60.31 | 69.56 | 441 | Broncos |
| 61 | 22 | Bryce Huff | 68.31 | 63.74 | 71.10 | 298 | Eagles |
| 62 | 23 | Azeez Ojulari | 68.03 | 58.69 | 76.76 | 391 | Giants |
| 63 | 24 | Dennis Gardeck | 67.88 | 49.59 | 81.39 | 206 | Cardinals |
| 64 | 25 | Kwity Paye | 67.60 | 64.68 | 67.63 | 667 | Colts |
| 65 | 26 | Chris Braswell | 67.59 | 60.70 | 68.01 | 335 | Buccaneers |
| 66 | 27 | Leonard Floyd | 67.53 | 51.27 | 74.20 | 604 | 49ers |
| 67 | 28 | Tyree Wilson | 67.24 | 66.82 | 63.97 | 524 | Raiders |
| 68 | 29 | Lukas Van Ness | 67.24 | 59.69 | 68.11 | 458 | Packers |
| 69 | 30 | Preston Smith | 67.09 | 52.58 | 72.60 | 469 | Steelers |
| 70 | 31 | Victor Dimukeje | 67.02 | 60.28 | 71.16 | 157 | Cardinals |
| 71 | 32 | Ogbo Okoronkwo | 67.02 | 53.34 | 73.06 | 464 | Browns |
| 72 | 33 | Derick Hall | 67.01 | 57.26 | 69.35 | 673 | Seahawks |
| 73 | 34 | Dayo Odeyingbo | 66.34 | 60.78 | 65.88 | 746 | Colts |
| 74 | 35 | Charles Snowden | 66.27 | 51.56 | 72.40 | 405 | Raiders |
| 75 | 36 | Laiatu Latu | 65.93 | 65.17 | 62.27 | 618 | Colts |
| 76 | 37 | Jacob Martin | 65.91 | 55.59 | 72.36 | 222 | Bears |
| 77 | 38 | A.J. Epenesa | 65.81 | 57.37 | 67.67 | 712 | Bills |
| 78 | 39 | Carl Lawson | 65.73 | 54.00 | 73.60 | 402 | Cowboys |
| 79 | 40 | Kingsley Enagbare | 64.86 | 59.19 | 64.47 | 538 | Packers |
| 80 | 41 | Jamin Davis | 64.86 | 52.87 | 73.11 | 107 | Jets |
| 81 | 42 | Tyquan Lewis | 64.77 | 58.07 | 70.47 | 355 | Colts |
| 82 | 43 | Arik Armstead | 64.57 | 55.97 | 66.14 | 569 | Jaguars |
| 83 | 44 | Arden Key | 64.48 | 59.02 | 64.43 | 734 | Titans |
| 84 | 45 | Sam Hubbard | 64.47 | 54.32 | 69.52 | 521 | Bengals |
| 85 | 46 | Dallas Turner | 64.41 | 58.63 | 64.10 | 310 | Vikings |
| 86 | 47 | Derek Barnett | 64.10 | 59.41 | 66.50 | 413 | Texans |
| 87 | 48 | Felix Anudike-Uzomah | 63.94 | 60.11 | 62.33 | 344 | Chiefs |
| 88 | 49 | Xavier Thomas | 63.62 | 55.78 | 67.62 | 208 | Cardinals |
| 89 | 50 | Julian Okwara | 63.60 | 54.06 | 71.48 | 286 | Cardinals |
| 90 | 51 | Deatrich Wise Jr. | 63.54 | 54.20 | 66.87 | 409 | Patriots |
| 91 | 52 | Keion White | 63.44 | 62.89 | 60.00 | 830 | Patriots |
| 92 | 53 | Clelin Ferrell | 63.29 | 57.24 | 63.36 | 443 | Commanders |
| 93 | 54 | Joseph Ossai | 63.25 | 58.33 | 62.36 | 573 | Bengals |
| 94 | 55 | Baron Browning | 63.01 | 57.96 | 66.13 | 378 | Cardinals |
| 95 | 56 | Jalyx Hunt | 62.74 | 59.31 | 63.79 | 320 | Eagles |
| 96 | 57 | Emmanuel Ogbah | 62.73 | 54.31 | 66.83 | 734 | Dolphins |
| 97 | 58 | Joe Tryon-Shoyinka | 62.72 | 56.42 | 63.24 | 570 | Buccaneers |
| 98 | 59 | Anfernee Jennings | 62.22 | 57.21 | 62.38 | 831 | Patriots |
| 99 | 60 | Arron Mosby | 62.04 | 57.87 | 67.52 | 150 | Packers |

### Rotation/backup (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 100 | 1 | Dre'Mont Jones | 61.37 | 49.22 | 65.30 | 617 | Seahawks |
| 101 | 2 | Mike Danna | 61.20 | 56.37 | 62.02 | 581 | Chiefs |
| 102 | 3 | Bud Dupree | 61.19 | 48.22 | 67.13 | 570 | Chargers |
| 103 | 4 | Austin Booker | 61.06 | 56.00 | 60.26 | 283 | Bears |
| 104 | 5 | Charles Omenihu | 60.86 | 52.14 | 67.81 | 303 | Chiefs |
| 105 | 6 | K'Lavon Chaisson | 60.84 | 57.16 | 61.68 | 508 | Raiders |
| 106 | 7 | Payton Turner | 60.64 | 60.26 | 63.87 | 335 | Saints |
| 107 | 8 | Anthony Nelson | 60.42 | 54.75 | 60.04 | 624 | Buccaneers |
| 108 | 9 | Sam Okuayinonu | 59.99 | 56.82 | 62.58 | 451 | 49ers |
| 109 | 10 | D.J. Wonnum | 59.65 | 55.18 | 63.66 | 453 | Panthers |
| 110 | 11 | Lorenzo Carter | 59.34 | 50.98 | 62.72 | 410 | Falcons |
| 111 | 12 | Tyus Bowser | 59.16 | 46.50 | 68.92 | 276 | Dolphins |
| 112 | 13 | Myles Murphy | 59.11 | 57.49 | 58.47 | 353 | Bengals |
| 113 | 14 | Yetur Gross-Matos | 58.86 | 56.88 | 60.43 | 367 | 49ers |
| 114 | 15 | Janarius Robinson | 58.74 | 52.61 | 65.33 | 109 | Raiders |
| 115 | 16 | Micheal Clemons | 58.46 | 53.17 | 58.30 | 624 | Jets |
| 116 | 17 | DeMarcus Walker | 58.39 | 46.04 | 62.45 | 738 | Bears |
| 117 | 18 | Tavius Robinson | 57.98 | 54.06 | 56.42 | 548 | Ravens |
| 118 | 19 | Dawuane Smoot | 56.92 | 49.27 | 61.19 | 386 | Bills |
| 119 | 20 | Josh Paschal | 56.42 | 56.90 | 54.86 | 613 | Lions |
| 120 | 21 | Javontae Jean-Baptiste | 56.01 | 54.05 | 55.12 | 248 | Commanders |
| 121 | 22 | Charles Harris | 55.97 | 50.62 | 61.16 | 474 | Eagles |
| 122 | 23 | Ali Gaye | 55.87 | 48.65 | 57.75 | 177 | Titans |
| 123 | 24 | David Ojabo | 54.95 | 56.12 | 58.03 | 292 | Ravens |
| 124 | 25 | Casey Toohill | 54.56 | 48.43 | 56.44 | 249 | Bills |
| 125 | 26 | Alex Wright | 54.22 | 56.83 | 54.68 | 103 | Browns |
| 126 | 27 | Robert Beal Jr. | 54.13 | 54.26 | 55.76 | 149 | 49ers |
| 127 | 28 | Tyrus Wheat | 54.11 | 53.87 | 59.90 | 165 | Cowboys |
| 128 | 29 | Cam Gill | 54.01 | 50.75 | 57.01 | 222 | Panthers |
| 129 | 30 | Quinton Bell | 53.98 | 49.34 | 55.94 | 258 | Dolphins |
| 130 | 31 | Al-Quadin Muhammad | 53.97 | 50.70 | 55.61 | 293 | Lions |
| 131 | 32 | Eric Watts | 53.61 | 56.22 | 50.64 | 231 | Jets |
| 132 | 33 | DJ Johnson | 52.98 | 52.74 | 52.29 | 392 | Panthers |
| 133 | 34 | Dylan Horton | 52.68 | 54.52 | 51.08 | 217 | Texans |
| 134 | 35 | Tomon Fox | 52.57 | 54.42 | 54.03 | 207 | Giants |
| 135 | 36 | Marshawn Kneeland | 52.25 | 54.94 | 52.17 | 255 | Cowboys |
| 136 | 37 | Jaylen Harrell | 52.20 | 51.33 | 48.61 | 286 | Titans |
| 137 | 38 | Malik Herring | 51.85 | 50.86 | 54.52 | 193 | Chiefs |
| 138 | 39 | James Smith-Williams | 51.03 | 47.02 | 54.74 | 306 | Falcons |
| 139 | 40 | Brent Urban | 46.53 | 31.67 | 53.25 | 209 | Ravens |
| 140 | 41 | Jeremiah Moon | 46.44 | 48.76 | 47.88 | 117 | Steelers |
| 141 | 42 | Demone Harris | 46.01 | 46.14 | 48.43 | 216 | Falcons |
| 142 | 43 | Myles Cole | 45.76 | 50.43 | 47.29 | 135 | Jaguars |

## G — Guard

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Lindstrom | 95.90 | 93.50 | 93.34 | 1099 | Falcons |
| 2 | 2 | Christian Mahogany | 95.39 | 91.50 | 93.81 | 144 | Lions |
| 3 | 3 | James Daniels | 95.35 | 92.90 | 92.81 | 209 | Steelers |
| 4 | 4 | Quinn Meinerz | 93.00 | 86.90 | 92.90 | 1131 | Broncos |
| 5 | 5 | Kevin Zeitler | 92.17 | 86.80 | 91.59 | 1047 | Lions |
| 6 | 6 | Will Fries | 91.29 | 86.90 | 90.05 | 268 | Colts |
| 7 | 7 | Quenton Nelson | 88.69 | 81.30 | 89.45 | 1083 | Colts |
| 8 | 8 | Landon Dickerson | 88.53 | 82.30 | 88.52 | 1157 | Eagles |
| 9 | 9 | Jordan Meredith | 87.67 | 80.80 | 88.09 | 574 | Raiders |
| 10 | 10 | Joe Thuney | 85.94 | 80.20 | 85.60 | 1232 | Chiefs |
| 11 | 11 | John Simpson | 85.84 | 77.30 | 87.37 | 1020 | Jets |
| 12 | 12 | Kevin Dotson | 85.39 | 77.70 | 86.35 | 1145 | Rams |
| 13 | 13 | Alijah Vera-Tucker | 85.37 | 77.70 | 86.31 | 916 | Jets |
| 14 | 14 | Dominick Puni | 84.51 | 80.50 | 83.02 | 1078 | 49ers |
| 15 | 15 | Damien Lewis | 83.97 | 75.50 | 85.45 | 942 | Panthers |
| 16 | 16 | Trey Smith | 83.91 | 75.30 | 85.49 | 1232 | Chiefs |
| 17 | 17 | Teven Jenkins | 83.59 | 75.40 | 84.88 | 738 | Bears |
| 18 | 18 | Tyler Smith | 83.51 | 75.00 | 85.02 | 1052 | Cowboys |
| 19 | 19 | Dylan Parham | 82.25 | 74.30 | 83.38 | 882 | Raiders |
| 20 | 20 | Cody Mauch | 82.10 | 74.60 | 82.94 | 1178 | Buccaneers |
| 21 | 21 | Chandler Zavala | 81.40 | 71.20 | 84.03 | 198 | Panthers |
| 22 | 22 | Matthew Bergeron | 80.47 | 70.90 | 82.69 | 1106 | Falcons |
| 23 | 23 | Jake Hanson | 80.43 | 69.90 | 83.28 | 103 | Jets |

### Good (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Mekhi Becton | 79.97 | 72.50 | 80.78 | 1097 | Eagles |
| 25 | 2 | Jack Driscoll | 78.99 | 71.30 | 79.95 | 110 | Eagles |
| 26 | 3 | Matt Pryor | 78.89 | 69.90 | 80.71 | 1005 | Bears |
| 27 | 4 | Will Hernandez | 78.39 | 69.30 | 80.28 | 280 | Cardinals |
| 28 | 5 | Sam Cosmi | 77.87 | 67.80 | 80.42 | 1259 | Commanders |
| 29 | 6 | Robert Hunt | 77.84 | 67.70 | 80.44 | 966 | Panthers |
| 30 | 7 | Jonah Jackson | 77.49 | 67.60 | 79.92 | 267 | Rams |
| 31 | 8 | Jackson Powers-Johnson | 77.36 | 63.90 | 82.16 | 956 | Raiders |
| 32 | 9 | Cesar Ruiz | 77.31 | 67.60 | 79.61 | 813 | Saints |
| 33 | 10 | Isaac Seumalo | 77.03 | 66.30 | 80.02 | 872 | Steelers |
| 34 | 11 | Aaron Banks | 76.52 | 65.40 | 79.76 | 775 | 49ers |
| 35 | 12 | Elgton Jenkins | 76.41 | 65.50 | 79.52 | 1073 | Packers |
| 36 | 13 | David Edwards | 76.09 | 66.10 | 78.58 | 2360 | Bills |
| 37 | 14 | Dalton Risner | 75.69 | 68.10 | 76.58 | 611 | Vikings |
| 38 | 15 | T.J. Bass | 75.69 | 63.00 | 79.99 | 315 | Cowboys |
| 39 | 16 | Ben Powers | 74.77 | 64.40 | 77.51 | 1130 | Broncos |
| 40 | 17 | Zack Martin | 74.70 | 65.60 | 76.60 | 638 | Cowboys |
| 41 | 18 | Brandon Scherff | 74.68 | 64.70 | 77.16 | 1013 | Jaguars |
| 42 | 19 | Ezra Cleveland | 74.50 | 64.90 | 76.74 | 911 | Jaguars |
| 43 | 20 | Zion Johnson | 74.31 | 64.40 | 76.75 | 1102 | Chargers |
| 44 | 21 | Joel Bitonio | 74.15 | 63.90 | 76.82 | 1178 | Browns |

### Starter (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 45 | 1 | Greg Van Roten | 73.96 | 63.40 | 76.84 | 1121 | Giants |
| 46 | 2 | Wyatt Teller | 73.60 | 62.60 | 76.77 | 885 | Browns |
| 47 | 3 | Evan Brown | 72.79 | 65.90 | 73.22 | 1070 | Cardinals |
| 48 | 4 | Laken Tomlinson | 72.59 | 62.10 | 75.41 | 1094 | Seahawks |
| 49 | 5 | Sean Rhyan | 72.10 | 61.30 | 75.13 | 1027 | Packers |
| 50 | 6 | Cordell Volson | 71.87 | 59.30 | 76.08 | 984 | Bengals |
| 51 | 7 | Nick Allegretti | 71.46 | 59.40 | 75.34 | 1372 | Commanders |
| 52 | 8 | Peter Skoronski | 71.43 | 60.30 | 74.69 | 1095 | Titans |
| 53 | 9 | Shaq Mason | 70.83 | 60.50 | 73.55 | 999 | Texans |
| 54 | 10 | Trey Pipkins III | 70.79 | 57.80 | 75.29 | 838 | Chargers |
| 55 | 11 | Mason McCormick | 69.38 | 57.70 | 73.00 | 936 | Steelers |
| 56 | 12 | Graham Glasgow | 69.33 | 57.20 | 73.25 | 1149 | Lions |
| 57 | 13 | Spencer Anderson | 69.31 | 56.70 | 73.55 | 357 | Steelers |
| 58 | 14 | Spencer Burford | 69.04 | 57.60 | 72.50 | 113 | 49ers |
| 59 | 15 | Blake Brandel | 68.92 | 55.70 | 73.56 | 1191 | Vikings |
| 60 | 16 | Robert Jones | 68.70 | 56.10 | 72.94 | 1080 | Dolphins |
| 61 | 17 | Patrick Mekari | 68.39 | 59.00 | 70.48 | 1131 | Ravens |
| 62 | 18 | Ben Bredeson | 68.00 | 56.00 | 71.84 | 1173 | Buccaneers |
| 63 | 19 | Jon Runyan | 67.88 | 56.10 | 71.56 | 842 | Giants |
| 64 | 20 | O'Cyrus Torrence | 67.62 | 55.50 | 71.53 | 1221 | Bills |
| 65 | 21 | Nick Zakelj | 67.45 | 58.70 | 69.11 | 162 | 49ers |
| 66 | 22 | Ed Ingram | 67.25 | 54.00 | 71.92 | 580 | Vikings |
| 67 | 23 | Jordan Morgan | 66.70 | 59.20 | 67.54 | 186 | Packers |
| 68 | 24 | Jake Kubas | 66.54 | 52.20 | 71.93 | 197 | Giants |
| 69 | 25 | Mark Glowinski | 66.29 | 53.40 | 70.71 | 355 | Colts |
| 70 | 26 | Nick Saldiveri | 65.89 | 56.00 | 68.31 | 344 | Saints |
| 71 | 27 | Dalton Tucker | 65.35 | 53.30 | 69.22 | 464 | Colts |
| 72 | 28 | Kayode Awosika | 65.19 | 51.30 | 70.28 | 145 | Lions |
| 73 | 29 | Anthony Bradford | 65.05 | 48.90 | 71.65 | 578 | Seahawks |
| 74 | 30 | Alex Cappa | 64.93 | 50.50 | 70.39 | 1132 | Bengals |
| 75 | 31 | Andrew Vorhees | 64.70 | 57.20 | 65.54 | 268 | Ravens |
| 76 | 32 | Logan Bruss | 64.24 | 44.80 | 73.03 | 195 | Titans |
| 77 | 33 | Aaron Stinnie | 63.59 | 47.90 | 69.88 | 193 | Giants |
| 78 | 34 | Isaiah Wynn | 63.45 | 49.00 | 68.91 | 103 | Dolphins |
| 79 | 35 | Mike Caliendo | 63.25 | 49.40 | 68.31 | 354 | Chiefs |
| 80 | 36 | Michael Dunn | 63.14 | 46.50 | 70.06 | 171 | Browns |

### Rotation/backup (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 81 | 1 | Zak Zinter | 61.81 | 43.90 | 69.58 | 233 | Browns |
| 82 | 2 | Christian Haynes | 60.75 | 48.50 | 64.75 | 167 | Seahawks |
| 83 | 3 | Layden Robinson | 60.14 | 43.60 | 67.00 | 602 | Patriots |
| 84 | 4 | Kenyon Green | 58.11 | 38.60 | 66.95 | 582 | Texans |
| 85 | 5 | Sataoa Laumea | 57.81 | 36.90 | 67.59 | 355 | Seahawks |
| 86 | 6 | Sidy Sow | 54.18 | 29.80 | 66.27 | 155 | Patriots |

## HB — Running Back

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Bucky Irving | 90.25 | 90.80 | 85.71 | 246 | Buccaneers |
| 2 | 2 | Jahmyr Gibbs | 87.03 | 90.10 | 80.82 | 347 | Lions |
| 3 | 3 | Derrick Henry | 87.01 | 94.20 | 78.05 | 197 | Ravens |
| 4 | 4 | De'Von Achane | 86.21 | 81.60 | 85.11 | 408 | Dolphins |
| 5 | 5 | Bijan Robinson | 86.03 | 92.80 | 77.35 | 389 | Falcons |
| 6 | 6 | Josh Jacobs | 84.04 | 92.30 | 74.37 | 265 | Packers |
| 7 | 7 | Kenneth Walker III | 82.89 | 88.40 | 75.05 | 224 | Seahawks |
| 8 | 8 | James Conner | 82.57 | 90.40 | 73.18 | 269 | Cardinals |
| 9 | 9 | Saquon Barkley | 82.00 | 87.60 | 74.10 | 353 | Eagles |
| 10 | 10 | James Cook | 81.22 | 86.20 | 73.74 | 258 | Bills |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | David Montgomery | 79.01 | 85.90 | 70.25 | 158 | Lions |
| 12 | 2 | Jordan Mason | 78.82 | 72.70 | 78.73 | 170 | 49ers |
| 13 | 3 | Aaron Jones | 78.54 | 75.40 | 76.47 | 347 | Vikings |
| 14 | 4 | Justice Hill | 77.89 | 79.50 | 72.65 | 264 | Ravens |
| 15 | 5 | Jaylen Warren | 77.32 | 64.30 | 81.84 | 232 | Steelers |
| 16 | 6 | Zach Charbonnet | 77.29 | 77.50 | 72.99 | 284 | Seahawks |
| 17 | 7 | Chase Brown | 77.28 | 75.70 | 74.17 | 339 | Bengals |
| 18 | 8 | Tony Pollard | 76.80 | 68.70 | 78.04 | 301 | Titans |
| 19 | 9 | Emanuel Wilson | 76.72 | 82.90 | 68.44 | 109 | Packers |
| 20 | 10 | Alvin Kamara | 76.07 | 73.70 | 73.49 | 311 | Saints |
| 21 | 11 | Najee Harris | 75.85 | 77.20 | 70.79 | 233 | Steelers |
| 22 | 12 | Jerome Ford | 75.41 | 71.20 | 74.05 | 304 | Browns |
| 23 | 13 | Rhamondre Stevenson | 75.20 | 69.60 | 74.76 | 273 | Patriots |
| 24 | 14 | Austin Ekeler | 75.14 | 69.80 | 74.54 | 283 | Commanders |
| 25 | 15 | Tyjae Spears | 75.06 | 67.80 | 75.74 | 167 | Titans |
| 26 | 16 | Antonio Gibson | 74.90 | 72.60 | 72.26 | 164 | Patriots |
| 27 | 17 | Rico Dowdle | 74.69 | 73.90 | 71.05 | 283 | Cowboys |
| 28 | 18 | Chuba Hubbard | 74.64 | 75.90 | 69.63 | 336 | Panthers |
| 29 | 19 | Breece Hall | 74.62 | 62.00 | 78.87 | 384 | Jets |
| 30 | 20 | Kareem Hunt | 74.59 | 74.30 | 70.61 | 238 | Chiefs |
| 31 | 21 | Jeremy McNichols | 74.21 | 72.00 | 71.51 | 136 | Commanders |
| 32 | 22 | Jaleel McLaughlin | 74.07 | 65.20 | 75.81 | 146 | Broncos |
| 33 | 23 | Raheem Mostert | 74.04 | 69.30 | 73.03 | 155 | Dolphins |

### Starter (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Ty Johnson | 73.89 | 73.30 | 70.11 | 231 | Bills |
| 35 | 2 | Tank Bigsby | 73.47 | 68.10 | 72.89 | 129 | Jaguars |
| 36 | 3 | Joe Mixon | 73.46 | 76.60 | 67.20 | 273 | Texans |
| 37 | 4 | J.K. Dobbins | 73.44 | 66.60 | 73.83 | 227 | Chargers |
| 38 | 5 | Miles Sanders | 73.15 | 68.00 | 72.41 | 130 | Panthers |
| 39 | 6 | Ray Davis | 72.70 | 70.40 | 70.06 | 112 | Bills |
| 40 | 7 | Kyren Williams | 72.55 | 69.10 | 70.68 | 402 | Rams |
| 41 | 8 | Rachaad White | 72.27 | 73.80 | 67.09 | 311 | Buccaneers |
| 42 | 9 | Brian Robinson | 71.87 | 72.00 | 67.62 | 237 | Commanders |
| 43 | 10 | Isaac Guerendo | 71.61 | 64.60 | 72.11 | 107 | 49ers |
| 44 | 11 | Devin Singletary | 70.82 | 62.10 | 72.47 | 164 | Giants |
| 45 | 12 | Samaje Perine | 70.49 | 67.60 | 68.25 | 217 | Chiefs |
| 46 | 13 | Cam Akers | 70.12 | 72.20 | 64.56 | 126 | Vikings |
| 47 | 14 | Jonathan Taylor | 70.12 | 56.90 | 74.77 | 270 | Colts |
| 48 | 15 | Javonte Williams | 70.04 | 61.70 | 71.43 | 300 | Broncos |
| 49 | 16 | Braelon Allen | 69.62 | 73.60 | 62.80 | 134 | Jets |
| 50 | 17 | Travis Etienne Jr. | 69.30 | 60.70 | 70.87 | 254 | Jaguars |
| 51 | 18 | Pierre Strong Jr. | 69.10 | 58.40 | 72.07 | 134 | Browns |
| 52 | 19 | Isiah Pacheco | 68.60 | 64.20 | 67.36 | 115 | Chiefs |
| 53 | 20 | Tyrone Tracy | 68.38 | 58.40 | 70.87 | 310 | Giants |
| 54 | 21 | D'Ernest Johnson | 68.32 | 60.70 | 69.23 | 117 | Jaguars |
| 55 | 22 | D'Andre Swift | 67.88 | 61.30 | 68.10 | 353 | Bears |
| 56 | 23 | Ameer Abdullah | 67.62 | 68.30 | 63.00 | 258 | Raiders |
| 57 | 24 | Alexander Mattison | 67.60 | 61.40 | 67.57 | 218 | Raiders |
| 58 | 25 | Roschon Johnson | 66.37 | 69.50 | 60.11 | 136 | Bears |
| 59 | 26 | Zack Moss | 65.99 | 58.20 | 67.01 | 155 | Bengals |
| 60 | 27 | Trey Sermon | 65.97 | 54.10 | 69.71 | 122 | Colts |
| 61 | 28 | Kenneth Gainwell | 65.63 | 57.60 | 66.81 | 132 | Eagles |
| 62 | 29 | Dare Ogunbowale | 62.48 | 60.60 | 59.56 | 213 | Texans |

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
| 3 | 3 | Leo Chenal | 83.45 | 84.50 | 78.59 | 497 | Chiefs |
| 4 | 4 | Bobby Wagner | 83.29 | 88.30 | 75.79 | 1258 | Commanders |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Edgerrin Cooper | 79.89 | 85.70 | 73.82 | 549 | Packers |
| 6 | 2 | Jack Campbell | 79.35 | 78.70 | 75.61 | 1047 | Lions |
| 7 | 3 | Devin Lloyd | 76.46 | 76.70 | 73.22 | 884 | Jaguars |
| 8 | 4 | Elandon Roberts | 75.55 | 79.70 | 68.61 | 525 | Steelers |
| 9 | 5 | Oren Burks | 75.49 | 81.90 | 71.36 | 322 | Eagles |
| 10 | 6 | Jack Gibbens | 75.48 | 83.20 | 73.82 | 234 | Titans |
| 11 | 7 | Devin Bush | 75.45 | 79.20 | 71.64 | 497 | Browns |
| 12 | 8 | Payton Wilson | 74.36 | 74.70 | 69.96 | 520 | Steelers |

### Starter (64 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Jordan Hicks | 73.96 | 77.40 | 71.13 | 602 | Browns |
| 14 | 2 | Malcolm Rodriguez | 73.62 | 74.40 | 73.94 | 318 | Lions |
| 15 | 3 | Demario Davis | 73.38 | 73.20 | 69.81 | 1090 | Saints |
| 16 | 4 | Eric Kendricks | 73.21 | 75.20 | 69.28 | 918 | Cowboys |
| 17 | 5 | Jeremiah Owusu-Koramoah | 72.77 | 80.60 | 68.96 | 460 | Browns |
| 18 | 6 | Bobby Okereke | 72.63 | 74.90 | 69.40 | 734 | Giants |
| 19 | 7 | Nakobe Dean | 72.34 | 76.60 | 70.92 | 880 | Eagles |
| 20 | 8 | Logan Wilson | 71.89 | 72.30 | 70.78 | 743 | Bengals |
| 21 | 9 | Jordyn Brooks | 71.78 | 71.30 | 68.42 | 1039 | Dolphins |
| 22 | 10 | Blake Cashman | 71.68 | 72.00 | 69.45 | 947 | Vikings |
| 23 | 11 | Robert Spillane | 71.61 | 68.40 | 69.79 | 1093 | Raiders |
| 24 | 12 | Christian Elliss | 71.38 | 72.60 | 68.36 | 514 | Patriots |
| 25 | 13 | Jake Hansen | 71.02 | 78.00 | 70.94 | 136 | Texans |
| 26 | 14 | Roquan Smith | 70.81 | 66.80 | 69.31 | 1099 | Ravens |
| 27 | 15 | Jeremiah Trotter Jr. | 70.62 | 75.40 | 72.09 | 109 | Eagles |
| 28 | 16 | Kaden Elliss | 70.29 | 71.10 | 65.59 | 1097 | Falcons |
| 29 | 17 | Quincy Williams | 70.12 | 68.00 | 67.77 | 1136 | Jets |
| 30 | 18 | C.J. Mosley | 69.95 | 74.70 | 68.99 | 110 | Jets |
| 31 | 19 | Chazz Surratt | 69.86 | 72.70 | 70.36 | 137 | Jets |
| 32 | 20 | Omar Speights | 69.55 | 69.40 | 69.40 | 504 | Rams |
| 33 | 21 | Pete Werner | 69.52 | 69.00 | 68.94 | 731 | Saints |
| 34 | 22 | Azeez Al-Shaair | 69.50 | 68.90 | 68.66 | 672 | Texans |
| 35 | 23 | Lavonte David | 69.48 | 67.90 | 66.36 | 1149 | Buccaneers |
| 36 | 24 | Foyesade Oluokun | 69.32 | 68.50 | 67.67 | 815 | Jaguars |
| 37 | 25 | Daiyan Henley | 69.20 | 69.90 | 68.62 | 1071 | Chargers |
| 38 | 26 | Derrick Barnes | 69.03 | 71.80 | 70.67 | 120 | Lions |
| 39 | 27 | Tyrel Dodson | 68.47 | 67.30 | 68.62 | 854 | Dolphins |
| 40 | 28 | Cody Barton | 68.05 | 63.70 | 67.96 | 1129 | Broncos |
| 41 | 29 | Drue Tranquill | 68.02 | 66.00 | 65.20 | 902 | Chiefs |
| 42 | 30 | Alex Anzalone | 67.95 | 66.90 | 67.42 | 681 | Lions |
| 43 | 31 | Joe Andreessen | 67.78 | 72.10 | 69.55 | 116 | Bills |
| 44 | 32 | Dee Winters | 67.03 | 66.40 | 67.09 | 398 | 49ers |
| 45 | 33 | Grant Stuard | 66.88 | 69.40 | 68.49 | 229 | Colts |
| 46 | 34 | Neville Hewitt | 66.66 | 71.90 | 67.91 | 351 | Texans |
| 47 | 35 | Damone Clark | 66.63 | 68.20 | 66.21 | 163 | Cowboys |
| 48 | 36 | J.J. Russell | 66.22 | 69.20 | 69.67 | 271 | Buccaneers |
| 49 | 37 | Nate Landman | 65.99 | 65.70 | 66.82 | 543 | Falcons |
| 50 | 38 | T.J. Edwards | 65.90 | 61.40 | 64.74 | 1054 | Bears |
| 51 | 39 | Krys Barnes | 65.75 | 63.10 | 67.57 | 205 | Cardinals |
| 52 | 40 | Nick Bolton | 65.66 | 62.50 | 65.07 | 1076 | Chiefs |
| 53 | 41 | Mack Wilson Sr. | 65.53 | 63.80 | 64.28 | 760 | Cardinals |
| 54 | 42 | Tyrice Knight | 65.51 | 65.40 | 66.32 | 550 | Seahawks |
| 55 | 43 | Frankie Luvu | 65.48 | 64.20 | 62.57 | 1239 | Commanders |
| 56 | 44 | Ernest Jones | 65.29 | 60.70 | 64.97 | 995 | Seahawks |
| 57 | 45 | Ivan Pace Jr. | 65.19 | 63.00 | 65.55 | 454 | Vikings |
| 58 | 46 | Troy Dye | 65.17 | 65.60 | 66.50 | 355 | Chargers |
| 59 | 47 | Henry To'oTo'o | 65.02 | 62.20 | 63.84 | 936 | Texans |
| 60 | 48 | Jack Sanborn | 64.74 | 63.10 | 63.44 | 235 | Bears |
| 61 | 49 | Zaire Franklin | 64.69 | 60.30 | 63.75 | 1157 | Colts |
| 62 | 50 | Germaine Pratt | 64.03 | 60.20 | 62.82 | 1075 | Bengals |
| 63 | 51 | Dorian Williams | 63.97 | 58.50 | 64.55 | 680 | Bills |
| 64 | 52 | Micah McFadden | 63.82 | 62.80 | 62.49 | 668 | Giants |
| 65 | 53 | Tremaine Edmunds | 63.72 | 59.30 | 63.86 | 1055 | Bears |
| 66 | 54 | Eric Wilson | 63.47 | 64.30 | 63.26 | 559 | Packers |
| 67 | 55 | Sione Takitaki | 63.43 | 63.00 | 63.76 | 194 | Patriots |
| 68 | 56 | Chris Board | 63.40 | 65.50 | 65.08 | 213 | Ravens |
| 69 | 57 | Shaq Thompson | 63.03 | 67.40 | 66.73 | 245 | Panthers |
| 70 | 58 | Jerome Baker | 62.77 | 61.00 | 64.39 | 566 | Titans |
| 71 | 59 | Quay Walker | 62.53 | 57.40 | 63.55 | 804 | Packers |
| 72 | 60 | DeMarvion Overshown | 62.38 | 61.60 | 61.19 | 708 | Cowboys |
| 73 | 61 | Denzel Perryman | 62.08 | 62.30 | 62.08 | 343 | Chargers |
| 74 | 62 | E.J. Speed | 62.06 | 56.70 | 62.74 | 1011 | Colts |
| 75 | 63 | Owen Pappoe | 62.06 | 63.00 | 64.36 | 131 | Cardinals |
| 76 | 64 | SirVocea Dennis | 62.06 | 67.30 | 64.93 | 105 | Buccaneers |

### Rotation/backup (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 77 | 1 | Akeem Davis-Gaither | 61.94 | 59.00 | 61.70 | 535 | Bengals |
| 78 | 2 | Alex Singleton | 61.92 | 62.90 | 64.35 | 190 | Broncos |
| 79 | 3 | Trenton Simpson | 61.61 | 58.70 | 64.04 | 654 | Ravens |
| 80 | 4 | Jahlani Tavai | 61.44 | 55.50 | 61.23 | 916 | Patriots |
| 81 | 5 | Yasir Abdullah | 61.41 | 56.60 | 63.52 | 170 | Jaguars |
| 82 | 6 | Patrick Queen | 61.03 | 56.80 | 59.69 | 1164 | Steelers |
| 83 | 7 | Chad Muma | 60.96 | 57.40 | 63.48 | 260 | Jaguars |
| 84 | 8 | Darius Muasau | 60.36 | 56.80 | 62.48 | 435 | Giants |
| 85 | 9 | Isaiah McDuffie | 60.21 | 55.40 | 61.21 | 728 | Packers |
| 86 | 10 | De'Vondre Campbell | 60.18 | 58.30 | 61.19 | 719 | 49ers |
| 87 | 11 | Ty Summers | 60.06 | 64.20 | 66.91 | 113 | Giants |
| 88 | 12 | Josey Jewell | 59.80 | 56.50 | 61.36 | 796 | Panthers |
| 89 | 13 | Devin White | 59.73 | 57.00 | 62.58 | 176 | Texans |
| 90 | 14 | Divine Deablo | 59.47 | 57.30 | 60.56 | 689 | Raiders |
| 91 | 15 | Kyzir White | 59.08 | 48.80 | 63.54 | 1015 | Cardinals |
| 92 | 16 | Troy Andersen | 59.06 | 60.40 | 63.51 | 287 | Falcons |
| 93 | 17 | Malik Harrison | 58.93 | 51.60 | 60.33 | 438 | Ravens |
| 94 | 18 | Christian Rozeboom | 58.91 | 53.50 | 60.90 | 956 | Rams |
| 95 | 19 | Jalen Reeves-Maybin | 58.90 | 57.50 | 61.75 | 165 | Lions |
| 96 | 20 | Luke Gifford | 58.84 | 60.20 | 65.53 | 203 | Titans |
| 97 | 21 | Ventrell Miller | 58.83 | 52.00 | 60.45 | 482 | Jaguars |
| 98 | 22 | Trevin Wallace | 58.34 | 56.00 | 61.62 | 582 | Panthers |
| 99 | 23 | Ben Niemann | 58.29 | 55.20 | 63.53 | 178 | Lions |
| 100 | 24 | Claudin Cherelus | 58.22 | 63.30 | 62.52 | 158 | Panthers |
| 101 | 25 | Anfernee Orji | 57.89 | 55.20 | 60.41 | 147 | Saints |
| 102 | 26 | Winston Reid | 57.64 | 51.80 | 59.33 | 144 | Browns |
| 103 | 27 | Troy Reeder | 57.59 | 57.90 | 62.73 | 372 | Rams |
| 104 | 28 | Marist Liufau | 57.26 | 50.10 | 57.86 | 520 | Cowboys |
| 105 | 29 | Amari Burney | 56.48 | 57.20 | 60.29 | 101 | Raiders |
| 106 | 30 | Terrel Bernard | 56.47 | 48.20 | 60.46 | 917 | Bills |
| 107 | 31 | Mohamoud Diabate | 55.67 | 52.50 | 60.11 | 581 | Browns |
| 108 | 32 | Kenneth Murray Jr. | 55.46 | 45.90 | 59.74 | 815 | Titans |
| 109 | 33 | Anthony Walker Jr. | 54.79 | 48.00 | 62.30 | 516 | Dolphins |
| 110 | 34 | Isaiah Simmons | 54.75 | 45.60 | 59.14 | 181 | Giants |
| 111 | 35 | Ezekiel Turner | 54.63 | 49.70 | 64.44 | 111 | Lions |
| 112 | 36 | JD Bertrand | 54.56 | 49.80 | 62.38 | 157 | Falcons |
| 113 | 37 | Matt Milano | 53.70 | 53.30 | 58.62 | 333 | Bills |
| 114 | 38 | Nick Vigil | 53.58 | 48.30 | 61.37 | 127 | Cowboys |
| 115 | 39 | Luke Masterson | 53.32 | 46.70 | 62.00 | 102 | Raiders |
| 116 | 40 | Willie Gay | 52.77 | 43.90 | 57.27 | 277 | Saints |
| 117 | 41 | K.J. Britt | 52.76 | 45.50 | 58.83 | 632 | Buccaneers |
| 118 | 42 | Justin Strnad | 51.86 | 49.90 | 57.03 | 736 | Broncos |
| 119 | 43 | Raekwon McMillan | 51.65 | 40.80 | 59.52 | 267 | Titans |
| 120 | 44 | Junior Colson | 49.14 | 36.70 | 59.15 | 234 | Chargers |
| 121 | 45 | Jacoby Windmon | 48.68 | 50.50 | 59.56 | 128 | Panthers |
| 122 | 46 | Christian Harris | 48.11 | 39.10 | 56.82 | 180 | Texans |
| 123 | 47 | Kamu Grugier-Hill | 47.99 | 36.40 | 55.97 | 182 | Vikings |
| 124 | 48 | Baylon Spector | 45.00 | 30.10 | 54.77 | 291 | Bills |
| 125 | 49 | Chandler Wooten | 45.00 | 29.20 | 54.02 | 212 | Panthers |
| 126 | 50 | Demetrius Flannigan-Fowles | 45.00 | 30.60 | 54.40 | 151 | 49ers |

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

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Geno Smith | 79.61 | 81.34 | 74.30 | 704 | Seahawks |
| 8 | 2 | Patrick Mahomes | 78.50 | 82.91 | 70.35 | 776 | Chiefs |
| 9 | 3 | Brock Purdy | 77.35 | 77.76 | 75.94 | 567 | 49ers |
| 10 | 4 | Jayden Daniels | 77.00 | 84.70 | 73.05 | 781 | Commanders |
| 11 | 5 | Jalen Hurts | 76.69 | 74.80 | 76.16 | 558 | Eagles |
| 12 | 6 | Tua Tagovailoa | 76.65 | 77.34 | 76.00 | 460 | Dolphins |
| 13 | 7 | Matthew Stafford | 76.24 | 76.21 | 73.20 | 667 | Rams |
| 14 | 8 | Sam Darnold | 76.02 | 76.79 | 76.74 | 725 | Vikings |
| 15 | 9 | C.J. Stroud | 75.87 | 78.24 | 69.61 | 742 | Texans |
| 16 | 10 | Derek Carr | 75.87 | 80.18 | 74.82 | 319 | Saints |
| 17 | 11 | Jordan Love | 74.59 | 77.41 | 72.26 | 528 | Packers |
| 18 | 12 | Russell Wilson | 74.01 | 74.49 | 74.13 | 444 | Steelers |

### Starter (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Kyler Murray | 73.43 | 75.31 | 70.91 | 656 | Cardinals |
| 20 | 2 | Kirk Cousins | 72.06 | 75.13 | 70.17 | 521 | Falcons |
| 21 | 3 | Dak Prescott | 71.68 | 75.68 | 70.49 | 344 | Cowboys |
| 22 | 4 | Aaron Rodgers | 71.61 | 76.21 | 67.78 | 684 | Jets |
| 23 | 5 | Trevor Lawrence | 70.76 | 75.49 | 68.40 | 332 | Jaguars |
| 24 | 6 | Bo Nix | 69.90 | 73.80 | 68.20 | 712 | Broncos |
| 25 | 7 | Michael Penix Jr. | 65.72 | 87.60 | 69.61 | 120 | Falcons |
| 26 | 8 | Bryce Young | 65.23 | 67.37 | 63.10 | 477 | Panthers |
| 27 | 9 | Caleb Williams | 64.73 | 62.90 | 64.42 | 741 | Bears |
| 28 | 10 | Joe Flacco | 64.27 | 69.29 | 68.14 | 290 | Colts |
| 29 | 11 | Justin Fields | 63.60 | 64.55 | 68.38 | 215 | Steelers |
| 30 | 12 | Drake Maye | 62.32 | 64.90 | 66.76 | 461 | Patriots |
| 31 | 13 | Andy Dalton | 62.10 | 71.78 | 64.43 | 185 | Panthers |

### Rotation/backup (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 32 | 1 | Jameis Winston | 61.63 | 69.02 | 64.81 | 347 | Browns |
| 33 | 2 | Daniel Jones | 60.94 | 66.18 | 60.10 | 418 | Vikings |
| 34 | 3 | Aidan O'Connell | 60.31 | 60.50 | 65.94 | 276 | Raiders |
| 35 | 4 | Gardner Minshew | 60.09 | 61.18 | 62.44 | 370 | Raiders |
| 36 | 5 | Will Levis | 59.48 | 56.27 | 65.74 | 384 | Titans |
| 37 | 6 | Mason Rudolph | 59.33 | 61.99 | 63.63 | 276 | Titans |
| 38 | 7 | Anthony Richardson | 58.78 | 59.43 | 62.34 | 317 | Colts |
| 39 | 8 | Mac Jones | 58.69 | 60.27 | 61.68 | 309 | Jaguars |
| 40 | 9 | Tyler Huntley | 58.02 | 61.50 | 59.84 | 182 | Dolphins |
| 41 | 10 | Cooper Rush | 57.84 | 59.36 | 60.73 | 352 | Cowboys |
| 42 | 11 | Deshaun Watson | 57.83 | 63.08 | 58.91 | 290 | Browns |
| 43 | 12 | Jacoby Brissett | 56.60 | 63.35 | 57.19 | 200 | Patriots |
| 44 | 13 | Drew Lock | 55.25 | 50.10 | 58.01 | 216 | Giants |
| 45 | 14 | Spencer Rattler | 55.20 | 49.40 | 55.49 | 284 | Saints |
| 46 | 15 | Desmond Ridder | 53.95 | 52.23 | 60.03 | 105 | Raiders |
| 47 | 16 | Dorian Thompson-Robinson | 52.12 | 41.97 | 52.01 | 142 | Browns |

## S — Safety

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (10 players)

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
| 10 | 10 | Jabrill Peppers | 80.59 | 81.70 | 81.67 | 372 | Patriots |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Jaden Hicks | 79.96 | 73.40 | 80.16 | 430 | Chiefs |
| 12 | 2 | Derwin James Jr. | 79.89 | 76.30 | 79.00 | 1059 | Chargers |
| 13 | 3 | C.J. Gardner-Johnson | 79.21 | 83.10 | 76.67 | 1118 | Eagles |
| 14 | 4 | Andrew Wingard | 77.88 | 77.60 | 80.86 | 216 | Jaguars |
| 15 | 5 | Dell Pettus | 76.78 | 72.10 | 77.70 | 341 | Patriots |
| 16 | 6 | Kamren Kinchens | 76.40 | 73.70 | 74.04 | 623 | Rams |
| 17 | 7 | Thomas Harper | 76.31 | 76.40 | 76.00 | 191 | Raiders |
| 18 | 8 | Budda Baker | 76.12 | 74.70 | 74.76 | 1064 | Cardinals |
| 19 | 9 | Jimmie Ward | 75.72 | 75.30 | 78.30 | 461 | Texans |
| 20 | 10 | Quandre Diggs | 75.38 | 70.60 | 78.82 | 419 | Titans |
| 21 | 11 | Julian Blackmon | 74.37 | 73.50 | 72.45 | 1084 | Colts |

### Starter (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Evan Williams | 73.83 | 73.80 | 73.60 | 533 | Packers |
| 23 | 2 | Ronnie Hickman Jr. | 73.64 | 71.30 | 75.08 | 463 | Browns |
| 24 | 3 | Jalen Pitre | 72.89 | 71.30 | 72.24 | 660 | Texans |
| 25 | 4 | Jordan Howden | 72.69 | 63.50 | 75.38 | 550 | Saints |
| 26 | 5 | Dadrion Taylor-Demerson | 72.44 | 70.00 | 71.87 | 258 | Cardinals |
| 27 | 6 | Zayne Anderson | 72.17 | 75.60 | 78.21 | 122 | Packers |
| 28 | 7 | Reed Blankenship | 72.15 | 71.40 | 70.83 | 1030 | Eagles |
| 29 | 8 | DeShon Elliott | 72.03 | 63.30 | 75.05 | 895 | Steelers |
| 30 | 9 | Kevin Byard | 71.54 | 60.50 | 74.74 | 1055 | Bears |
| 31 | 10 | Tony Jefferson | 71.49 | 69.70 | 78.11 | 261 | Chargers |
| 32 | 11 | Harrison Smith | 71.30 | 65.30 | 71.71 | 1062 | Vikings |
| 33 | 12 | Minkah Fitzpatrick | 71.08 | 64.90 | 73.19 | 1158 | Steelers |
| 34 | 13 | Jalen Thompson | 71.03 | 64.20 | 72.99 | 941 | Cardinals |
| 35 | 14 | Mike Brown | 70.86 | 67.60 | 73.97 | 384 | Titans |
| 36 | 15 | Ashtyn Davis | 70.84 | 67.80 | 72.81 | 260 | Jets |
| 37 | 16 | Ji'Ayir Brown | 70.35 | 69.50 | 68.58 | 886 | 49ers |
| 38 | 17 | Mike Edwards | 69.72 | 63.30 | 75.04 | 251 | Buccaneers |
| 39 | 18 | Marcus Maye | 69.27 | 69.20 | 71.92 | 405 | Chargers |
| 40 | 19 | Jeremy Chinn | 68.95 | 64.50 | 70.40 | 1207 | Commanders |
| 41 | 20 | Tony Adams | 68.83 | 67.50 | 69.57 | 764 | Jets |
| 42 | 21 | Jaylen McCollough | 68.47 | 61.60 | 68.88 | 382 | Rams |
| 43 | 22 | Nick Cross | 68.46 | 64.30 | 70.10 | 1156 | Colts |
| 44 | 23 | Amani Hooker | 68.30 | 66.30 | 69.69 | 848 | Titans |
| 45 | 24 | Kamren Curl | 68.06 | 63.70 | 68.07 | 1112 | Rams |
| 46 | 25 | Jordan Poyer | 67.93 | 61.40 | 69.59 | 964 | Dolphins |
| 47 | 26 | Alohi Gilman | 67.35 | 64.50 | 68.41 | 731 | Chargers |
| 48 | 27 | Grant Delpit | 67.21 | 60.90 | 68.91 | 976 | Browns |
| 49 | 28 | Dane Belton | 67.10 | 61.30 | 68.57 | 460 | Giants |
| 50 | 29 | Malik Mustapha | 66.27 | 60.10 | 68.18 | 755 | 49ers |
| 51 | 30 | Malik Hooker | 66.23 | 57.80 | 67.89 | 1062 | Cowboys |
| 52 | 31 | Camryn Bynum | 66.20 | 58.60 | 67.10 | 1056 | Vikings |
| 53 | 32 | Tre'von Moehrig | 65.73 | 54.40 | 69.51 | 1099 | Raiders |
| 54 | 33 | Damontae Kazee | 65.42 | 56.40 | 69.91 | 313 | Steelers |
| 55 | 34 | Vonn Bell | 65.21 | 62.10 | 64.48 | 705 | Bengals |
| 56 | 35 | George Odum | 65.09 | 66.70 | 69.75 | 139 | 49ers |
| 57 | 36 | Juan Thornhill | 65.04 | 64.50 | 65.84 | 401 | Browns |
| 58 | 37 | Jaquan Brisker | 64.89 | 63.50 | 68.52 | 293 | Bears |
| 59 | 38 | Eric Murray | 64.75 | 64.90 | 64.70 | 961 | Texans |
| 60 | 39 | Will Harris | 64.52 | 61.60 | 64.27 | 860 | Saints |
| 61 | 40 | Kaevon Merriweather | 64.07 | 62.90 | 66.69 | 274 | Buccaneers |
| 62 | 41 | Quentin Lake | 63.84 | 58.40 | 65.85 | 1207 | Rams |
| 63 | 42 | Justin Simmons | 63.83 | 60.80 | 63.75 | 1017 | Falcons |
| 64 | 43 | Tyler Nubin | 63.68 | 58.10 | 67.15 | 789 | Giants |
| 65 | 44 | Devon Key | 63.56 | 56.50 | 65.56 | 253 | Broncos |
| 66 | 45 | Josh Metellus | 63.42 | 52.20 | 67.52 | 1030 | Vikings |
| 67 | 46 | Xavier Woods | 63.22 | 55.40 | 65.53 | 1216 | Panthers |
| 68 | 47 | Tyrann Mathieu | 62.84 | 57.80 | 62.04 | 1015 | Saints |
| 69 | 48 | Donovan Wilson | 62.74 | 56.60 | 62.96 | 1008 | Cowboys |
| 70 | 49 | Andre Cisco | 62.44 | 56.70 | 63.57 | 979 | Jaguars |
| 71 | 50 | Jaylinn Hawkins | 62.09 | 56.30 | 64.24 | 613 | Patriots |

### Rotation/backup (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 72 | 1 | Jevon Holland | 61.66 | 57.10 | 62.98 | 854 | Dolphins |
| 73 | 2 | Jordan Whitehead | 60.67 | 53.50 | 63.73 | 731 | Buccaneers |
| 74 | 3 | Rayshawn Jenkins | 60.00 | 58.00 | 59.14 | 550 | Seahawks |
| 75 | 4 | Jordan Battle | 59.57 | 53.00 | 61.01 | 464 | Bengals |
| 76 | 5 | Jonathan Owens | 59.39 | 53.90 | 62.32 | 429 | Bears |
| 77 | 6 | Geno Stone | 59.20 | 53.70 | 59.29 | 1100 | Bengals |
| 78 | 7 | Bryan Cook | 59.00 | 48.10 | 63.76 | 1056 | Chiefs |
| 79 | 8 | Antoine Winfield Jr. | 58.95 | 50.80 | 64.44 | 601 | Buccaneers |
| 80 | 9 | Taylor Rapp | 58.83 | 43.20 | 66.65 | 840 | Bills |
| 81 | 10 | Christian Izien | 58.80 | 55.30 | 58.80 | 697 | Buccaneers |
| 82 | 11 | Demani Richardson | 58.21 | 58.60 | 65.08 | 403 | Panthers |
| 83 | 12 | Talanoa Hufanga | 57.42 | 52.40 | 63.57 | 308 | 49ers |
| 84 | 13 | Javon Bullard | 56.86 | 49.00 | 58.91 | 816 | Packers |
| 85 | 14 | Nick Scott | 56.83 | 56.70 | 58.33 | 324 | Panthers |
| 86 | 15 | P.J. Locke | 56.36 | 50.70 | 58.71 | 1076 | Broncos |
| 87 | 16 | Eddie Jackson | 55.98 | 50.10 | 60.63 | 390 | Chargers |
| 88 | 17 | Rodney McLeod | 55.55 | 47.20 | 59.01 | 565 | Browns |
| 89 | 18 | Antonio Johnson | 54.99 | 43.00 | 58.82 | 685 | Jaguars |
| 90 | 19 | Jason Pinnock | 54.50 | 45.50 | 57.31 | 976 | Giants |
| 91 | 20 | Chuck Clark | 54.37 | 45.10 | 58.83 | 709 | Jets |
| 92 | 21 | Jordan Fuller | 53.94 | 47.00 | 61.07 | 574 | Panthers |
| 93 | 22 | Damar Hamlin | 53.87 | 41.50 | 62.17 | 1042 | Bills |
| 94 | 23 | Cole Bishop | 53.49 | 48.70 | 55.45 | 464 | Bills |
| 95 | 24 | Isaiah Pola-Mao | 53.43 | 45.10 | 58.64 | 952 | Raiders |
| 96 | 25 | Percy Butler | 52.69 | 41.70 | 57.62 | 448 | Commanders |
| 97 | 26 | Marcus Williams | 52.41 | 37.30 | 63.80 | 601 | Ravens |
| 98 | 27 | Calen Bullock | 52.09 | 36.20 | 58.51 | 1083 | Texans |
| 99 | 28 | Darnell Savage | 52.08 | 40.70 | 59.14 | 764 | Jaguars |
| 100 | 29 | Richie Grant | 50.64 | 42.50 | 54.84 | 165 | Falcons |
| 101 | 30 | Marcus Epps | 49.15 | 44.00 | 55.29 | 176 | Raiders |
| 102 | 31 | K'Von Wallace | 48.69 | 41.80 | 54.41 | 127 | Seahawks |
| 103 | 32 | Kyle Dugger | 48.08 | 34.80 | 55.12 | 759 | Patriots |

## T — Tackle

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jordan Mailata | 98.46 | 95.20 | 96.47 | 995 | Eagles |
| 2 | 2 | Penei Sewell | 94.24 | 89.60 | 93.17 | 1213 | Lions |
| 3 | 3 | Rashawn Slater | 93.83 | 90.90 | 91.61 | 959 | Chargers |
| 4 | 4 | Lane Johnson | 93.82 | 87.50 | 93.87 | 1123 | Eagles |
| 5 | 5 | Terron Armstead | 93.37 | 89.40 | 91.85 | 821 | Dolphins |
| 6 | 6 | Zach Tom | 91.99 | 85.80 | 91.95 | 1134 | Packers |
| 7 | 7 | Trent Williams | 91.00 | 85.60 | 90.43 | 649 | 49ers |
| 8 | 8 | Bernhard Raimann | 90.22 | 85.10 | 89.46 | 856 | Colts |
| 9 | 9 | Christian Darrisaw | 88.75 | 81.40 | 89.49 | 392 | Vikings |
| 10 | 10 | Spencer Brown | 87.98 | 77.90 | 90.54 | 1140 | Bills |
| 11 | 11 | Tristan Wirfs | 87.89 | 82.50 | 87.32 | 1061 | Buccaneers |
| 12 | 12 | Garett Bolles | 87.87 | 80.20 | 88.81 | 1111 | Broncos |
| 13 | 13 | Charles Cross | 87.77 | 82.50 | 87.11 | 1094 | Seahawks |
| 14 | 14 | Darnell Wright | 87.39 | 79.30 | 88.62 | 1021 | Bears |
| 15 | 15 | Laremy Tunsil | 87.34 | 78.10 | 89.33 | 1167 | Texans |
| 16 | 16 | Paris Johnson Jr. | 86.54 | 80.80 | 86.20 | 865 | Cardinals |
| 17 | 17 | Brian O'Neill | 86.49 | 79.30 | 87.11 | 1151 | Vikings |
| 18 | 18 | Alaric Jackson | 86.11 | 78.40 | 87.09 | 1017 | Rams |
| 19 | 19 | Kolton Miller | 86.07 | 80.60 | 85.55 | 1075 | Raiders |
| 20 | 20 | Jake Matthews | 86.04 | 79.80 | 86.03 | 1119 | Falcons |
| 21 | 21 | Taylor Moton | 84.79 | 77.20 | 85.69 | 846 | Panthers |
| 22 | 22 | Luke Goedeke | 84.78 | 74.20 | 87.67 | 952 | Buccaneers |
| 23 | 23 | Rob Havenstein | 84.61 | 75.80 | 86.32 | 805 | Rams |
| 24 | 24 | Braxton Jones | 84.21 | 77.40 | 84.58 | 719 | Bears |
| 25 | 25 | Joe Alt | 83.87 | 75.90 | 85.02 | 1066 | Chargers |
| 26 | 26 | Taylor Decker | 83.74 | 77.20 | 83.94 | 963 | Lions |
| 27 | 27 | Andrew Thomas | 83.08 | 75.40 | 84.03 | 416 | Giants |
| 28 | 28 | Walker Little | 82.78 | 72.80 | 85.26 | 508 | Jaguars |
| 29 | 29 | Kaleb McGary | 82.77 | 73.90 | 84.52 | 1042 | Falcons |
| 30 | 30 | Cornelius Lucas | 82.67 | 74.10 | 84.21 | 464 | Commanders |
| 31 | 31 | Ikem Ekwonu | 82.48 | 71.70 | 85.50 | 909 | Panthers |
| 32 | 32 | Jaylon Moore | 82.42 | 74.90 | 83.27 | 271 | 49ers |
| 33 | 33 | Mike McGlinchey | 82.35 | 72.60 | 84.68 | 891 | Broncos |
| 34 | 34 | Tyron Smith | 82.31 | 73.70 | 83.88 | 592 | Jets |
| 35 | 35 | Dion Dawkins | 82.21 | 72.40 | 84.59 | 1164 | Bills |
| 36 | 36 | Colton McKivitz | 81.30 | 72.20 | 83.20 | 1062 | 49ers |
| 37 | 37 | Jonah Williams | 80.29 | 70.70 | 82.51 | 343 | Cardinals |
| 38 | 38 | Kendall Lamm | 80.23 | 72.70 | 81.09 | 512 | Dolphins |

### Good (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Ronnie Stanley | 79.86 | 70.70 | 81.80 | 1221 | Ravens |
| 40 | 2 | Braden Smith | 79.62 | 66.20 | 84.40 | 731 | Colts |
| 41 | 3 | Terence Steele | 79.59 | 67.00 | 83.82 | 1168 | Cowboys |
| 42 | 4 | John Ojukwu | 79.17 | 69.40 | 81.51 | 264 | Titans |
| 43 | 5 | Justin Skule | 79.13 | 69.20 | 81.58 | 362 | Buccaneers |
| 44 | 6 | Tytus Howard | 78.98 | 70.20 | 80.66 | 1157 | Texans |
| 45 | 7 | Taliese Fuaga | 77.52 | 65.70 | 81.23 | 1070 | Saints |
| 46 | 8 | Trent Brown | 77.49 | 65.20 | 81.51 | 139 | Bengals |
| 47 | 9 | DJ Glaze | 77.14 | 66.10 | 80.34 | 998 | Raiders |
| 48 | 10 | Matt Peart | 77.13 | 67.40 | 79.45 | 190 | Broncos |
| 49 | 11 | Evan Neal | 77.13 | 61.20 | 83.58 | 459 | Giants |
| 50 | 12 | Rasheed Walker | 77.10 | 68.60 | 78.60 | 1139 | Packers |
| 51 | 13 | Matt Goncalves | 76.75 | 65.90 | 79.81 | 566 | Colts |
| 52 | 14 | Dan Moore Jr. | 76.46 | 67.20 | 78.46 | 1128 | Steelers |
| 53 | 15 | Morgan Moses | 76.35 | 63.30 | 80.89 | 723 | Jets |
| 54 | 16 | Roger Rosengarten | 76.33 | 66.00 | 79.05 | 1066 | Ravens |
| 55 | 17 | Alex Palczewski | 76.24 | 63.40 | 80.64 | 179 | Broncos |
| 56 | 18 | Jack Conklin | 75.92 | 66.20 | 78.23 | 818 | Browns |
| 57 | 19 | Kelvin Beachum | 75.68 | 64.10 | 79.23 | 742 | Cardinals |
| 58 | 20 | Storm Norton | 75.20 | 61.90 | 79.90 | 128 | Falcons |
| 59 | 21 | Anton Harrison | 74.93 | 64.20 | 77.91 | 943 | Jaguars |
| 60 | 22 | Abraham Lucas | 74.54 | 61.90 | 78.80 | 406 | Seahawks |
| 61 | 23 | Joshua Ezeudu | 74.53 | 62.70 | 78.25 | 182 | Giants |
| 62 | 24 | Austin Jackson | 74.28 | 60.00 | 79.63 | 542 | Dolphins |
| 63 | 25 | Cam Robinson | 74.27 | 63.20 | 77.48 | 1073 | Vikings |
| 64 | 26 | Cole Van Lanen | 74.04 | 62.30 | 77.70 | 252 | Jaguars |

### Starter (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 65 | 1 | Trevor Penning | 73.98 | 60.20 | 79.00 | 1081 | Saints |
| 66 | 2 | Olumuyiwa Fashanu | 73.89 | 61.20 | 78.18 | 534 | Jets |
| 67 | 3 | Warren McClendon Jr. | 73.84 | 59.80 | 79.04 | 333 | Rams |
| 68 | 4 | Andrew Wylie | 73.67 | 61.70 | 77.48 | 1115 | Commanders |
| 69 | 5 | JC Latham | 73.61 | 61.80 | 77.31 | 1095 | Titans |
| 70 | 6 | Jawaan Taylor | 73.48 | 59.80 | 78.43 | 1209 | Chiefs |
| 71 | 7 | Joe Noteboom | 73.40 | 60.00 | 78.16 | 220 | Rams |
| 72 | 8 | Brandon Coleman | 72.79 | 59.80 | 77.29 | 1013 | Commanders |
| 73 | 9 | Broderick Jones | 72.49 | 58.70 | 77.52 | 1117 | Steelers |
| 74 | 10 | Jackson Barton | 71.72 | 63.90 | 72.77 | 157 | Cardinals |
| 75 | 11 | Orlando Brown Jr. | 71.66 | 58.20 | 76.46 | 637 | Bengals |
| 76 | 12 | Amarius Mims | 70.71 | 57.80 | 75.15 | 835 | Bengals |
| 77 | 13 | Yosh Nijman | 69.80 | 57.90 | 73.56 | 187 | Panthers |
| 78 | 14 | Vederian Lowe | 68.87 | 54.00 | 74.61 | 803 | Patriots |
| 79 | 15 | Jedrick Wills Jr. | 68.46 | 52.90 | 74.67 | 245 | Browns |
| 80 | 16 | Chuma Edoga | 68.45 | 52.50 | 74.92 | 226 | Cowboys |
| 81 | 17 | Wanya Morris | 68.38 | 53.00 | 74.47 | 732 | Chiefs |
| 82 | 18 | Dan Skipper | 68.28 | 55.20 | 72.84 | 324 | Lions |
| 83 | 19 | David Quessenberry | 68.15 | 55.20 | 72.61 | 133 | Vikings |
| 84 | 20 | Devin Cochran | 67.38 | 50.60 | 74.40 | 152 | Bengals |
| 85 | 21 | Larry Borom | 67.34 | 53.80 | 72.20 | 329 | Bears |
| 86 | 22 | Chris Hubbard | 67.14 | 46.60 | 76.67 | 257 | Giants |
| 87 | 23 | James Hudson III | 67.03 | 50.40 | 73.95 | 222 | Browns |
| 88 | 24 | Tyler Guyton | 66.86 | 49.40 | 74.34 | 668 | Cowboys |
| 89 | 25 | Fred Johnson | 65.16 | 49.30 | 71.56 | 490 | Eagles |
| 90 | 26 | Dawand Jones | 65.04 | 46.40 | 73.30 | 511 | Browns |
| 91 | 27 | Caedan Wallace | 64.73 | 44.10 | 74.31 | 129 | Patriots |
| 92 | 28 | Trent Scott | 64.64 | 46.10 | 72.83 | 288 | Commanders |
| 93 | 29 | Nicholas Petit-Frere | 64.33 | 46.50 | 72.05 | 621 | Titans |
| 94 | 30 | Carter Warren | 64.19 | 44.60 | 73.08 | 141 | Jets |
| 95 | 31 | Ryan Van Demark | 64.13 | 53.30 | 67.18 | 199 | Bills |
| 96 | 32 | Mike Jerrell | 63.94 | 46.40 | 71.46 | 250 | Seahawks |
| 97 | 33 | Patrick Paul | 63.86 | 44.90 | 72.34 | 338 | Dolphins |
| 98 | 34 | Thayer Munford Jr. | 63.02 | 45.90 | 70.27 | 201 | Raiders |
| 99 | 35 | Blake Fisher | 62.61 | 44.70 | 70.38 | 478 | Texans |
| 100 | 36 | Stone Forsythe | 62.17 | 43.10 | 70.71 | 414 | Seahawks |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 101 | 1 | Charlie Heck | 60.88 | 41.20 | 69.83 | 117 | 49ers |
| 102 | 2 | Demontrey Jacobs | 60.09 | 38.40 | 70.39 | 867 | Patriots |
| 103 | 3 | Kingsley Suamataia | 59.35 | 37.90 | 69.48 | 198 | Chiefs |
| 104 | 4 | Kiran Amegadjie | 58.32 | 40.30 | 66.17 | 126 | Bears |

## TE — Tight End

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | George Kittle | 86.13 | 92.10 | 77.99 | 489 | 49ers |
| 2 | 2 | Trey McBride | 81.96 | 86.80 | 74.57 | 581 | Cardinals |
| 3 | 3 | Mark Andrews | 80.93 | 83.10 | 75.32 | 455 | Ravens |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Brock Bowers | 79.58 | 85.10 | 71.74 | 654 | Raiders |
| 5 | 2 | Travis Kelce | 77.55 | 72.70 | 76.62 | 698 | Chiefs |
| 6 | 3 | Jonnu Smith | 77.52 | 78.20 | 72.90 | 486 | Dolphins |
| 7 | 4 | Dallas Goedert | 75.87 | 73.30 | 73.42 | 325 | Eagles |
| 8 | 5 | Evan Engram | 75.78 | 72.50 | 73.80 | 260 | Jaguars |
| 9 | 6 | Austin Hooper | 75.19 | 75.80 | 70.62 | 335 | Patriots |
| 10 | 7 | Foster Moreau | 75.09 | 71.10 | 73.58 | 395 | Saints |
| 11 | 8 | T.J. Hockenson | 75.07 | 74.80 | 71.09 | 366 | Vikings |
| 12 | 9 | Sam LaPorta | 74.50 | 73.60 | 70.93 | 546 | Lions |
| 13 | 10 | Dalton Kincaid | 74.25 | 71.50 | 71.92 | 365 | Bills |
| 14 | 11 | Andrew Ogletree | 74.05 | 69.10 | 73.18 | 173 | Colts |

### Starter (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Isaiah Likely | 73.49 | 75.60 | 67.92 | 386 | Ravens |
| 16 | 2 | Jordan Akins | 73.26 | 67.20 | 73.14 | 348 | Browns |
| 17 | 3 | Hunter Henry | 72.94 | 70.00 | 70.73 | 548 | Patriots |
| 18 | 4 | Noah Gray | 72.93 | 71.80 | 69.52 | 380 | Chiefs |
| 19 | 5 | Josh Oliver | 72.80 | 74.30 | 67.63 | 254 | Vikings |
| 20 | 6 | Mo Alie-Cox | 72.60 | 64.50 | 73.84 | 227 | Colts |
| 21 | 7 | Tucker Kraft | 72.50 | 67.80 | 71.46 | 538 | Packers |
| 22 | 8 | Will Dissly | 72.49 | 67.10 | 71.91 | 361 | Chargers |
| 23 | 9 | Zach Ertz | 72.49 | 67.00 | 71.98 | 658 | Commanders |
| 24 | 10 | David Njoku | 72.36 | 64.00 | 73.76 | 415 | Browns |
| 25 | 11 | Stone Smartt | 72.21 | 66.40 | 71.92 | 138 | Chargers |
| 26 | 12 | Mike Gesicki | 72.10 | 71.50 | 68.34 | 449 | Bengals |
| 27 | 13 | Taysom Hill | 71.59 | 71.80 | 67.28 | 103 | Saints |
| 28 | 14 | Darnell Washington | 71.22 | 71.20 | 67.07 | 257 | Steelers |
| 29 | 15 | Hunter Long | 70.74 | 66.60 | 69.33 | 110 | Rams |
| 30 | 16 | Chris Manhertz | 70.16 | 69.30 | 66.57 | 139 | Giants |
| 31 | 17 | Josh Whyle | 69.84 | 64.80 | 69.03 | 204 | Titans |
| 32 | 18 | Pat Freiermuth | 69.76 | 67.20 | 67.30 | 516 | Steelers |
| 33 | 19 | Colby Parkinson | 69.56 | 63.20 | 69.64 | 390 | Rams |
| 34 | 20 | Cole Kmet | 69.50 | 60.60 | 71.26 | 637 | Bears |
| 35 | 21 | Erick All | 69.37 | 59.90 | 71.51 | 112 | Bengals |
| 36 | 22 | Tyler Conklin | 69.36 | 58.80 | 72.23 | 556 | Jets |
| 37 | 23 | Noah Fant | 69.36 | 66.00 | 67.43 | 426 | Seahawks |
| 38 | 24 | Kylen Granson | 69.25 | 58.30 | 72.39 | 268 | Colts |
| 39 | 25 | Chigoziem Okonkwo | 69.13 | 59.90 | 71.11 | 425 | Titans |
| 40 | 26 | Kyle Pitts | 68.84 | 59.60 | 70.83 | 511 | Falcons |
| 41 | 27 | Payne Durham | 68.79 | 60.80 | 69.95 | 191 | Buccaneers |
| 42 | 28 | Brenton Strange | 68.75 | 66.00 | 66.41 | 320 | Jaguars |
| 43 | 29 | Juwan Johnson | 68.36 | 66.70 | 65.30 | 467 | Saints |
| 44 | 30 | Cade Otton | 67.99 | 64.10 | 66.42 | 573 | Buccaneers |
| 45 | 31 | Luke Farrell | 67.87 | 57.70 | 70.49 | 152 | Jaguars |
| 46 | 32 | Nate Adkins | 67.05 | 63.90 | 64.99 | 181 | Broncos |
| 47 | 33 | Adam Trautman | 66.33 | 56.90 | 68.45 | 288 | Broncos |
| 48 | 34 | Dawson Knox | 65.97 | 57.10 | 67.72 | 392 | Bills |
| 49 | 35 | Dalton Schultz | 65.92 | 60.80 | 65.17 | 648 | Texans |
| 50 | 36 | Ben Sinnott | 65.82 | 53.20 | 70.07 | 122 | Commanders |
| 51 | 37 | Michael Mayer | 65.79 | 57.70 | 67.01 | 284 | Raiders |
| 52 | 38 | Lucas Krull | 65.73 | 52.40 | 70.45 | 252 | Broncos |
| 53 | 39 | Brevyn Spann-Ford | 65.36 | 53.70 | 68.96 | 146 | Cowboys |
| 54 | 40 | AJ Barner | 64.95 | 61.00 | 63.42 | 253 | Seahawks |
| 55 | 41 | Johnny Mundt | 64.53 | 57.20 | 65.25 | 236 | Vikings |
| 56 | 42 | Harrison Bryant | 64.20 | 60.00 | 62.84 | 101 | Raiders |
| 57 | 43 | Jake Ferguson | 64.12 | 54.50 | 66.37 | 427 | Cowboys |
| 58 | 44 | Charlie Woerner | 64.06 | 58.90 | 63.34 | 131 | Falcons |
| 59 | 45 | Theo Johnson | 63.53 | 53.60 | 65.99 | 446 | Giants |
| 60 | 46 | Grant Calcaterra | 63.20 | 53.20 | 65.70 | 347 | Eagles |
| 61 | 47 | Luke Schoonmaker | 63.14 | 58.20 | 62.27 | 213 | Cowboys |
| 62 | 48 | Daniel Bellinger | 63.03 | 57.20 | 62.75 | 207 | Giants |
| 63 | 49 | Ja'Tavion Sanders | 62.39 | 52.50 | 64.81 | 359 | Panthers |
| 64 | 50 | Gerald Everett | 62.09 | 42.90 | 70.71 | 133 | Bears |
| 65 | 51 | Nick Vannett | 62.08 | 54.30 | 63.10 | 145 | Titans |

### Rotation/backup (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 66 | 1 | Brock Wright | 61.86 | 52.00 | 64.26 | 235 | Lions |
| 67 | 2 | Jeremy Ruckert | 61.18 | 46.70 | 66.67 | 198 | Jets |
| 68 | 3 | Tommy Tremble | 60.79 | 54.40 | 60.88 | 303 | Panthers |
| 69 | 4 | John Bates | 60.11 | 47.30 | 64.49 | 252 | Commanders |
| 70 | 5 | Pharaoh Brown | 59.75 | 45.60 | 65.01 | 107 | Seahawks |
| 71 | 6 | Cade Stover | 59.36 | 52.10 | 60.03 | 192 | Texans |
| 72 | 7 | Tip Reiman | 58.50 | 49.30 | 60.46 | 169 | Cardinals |
| 73 | 8 | Durham Smythe | 58.17 | 43.30 | 63.91 | 156 | Dolphins |
| 74 | 9 | Eric Saubert | 57.92 | 51.20 | 58.24 | 177 | 49ers |
| 75 | 10 | Drew Sample | 57.49 | 48.90 | 59.05 | 278 | Bengals |
| 76 | 11 | Davis Allen | 57.21 | 46.40 | 60.25 | 176 | Rams |
| 77 | 12 | Greg Dulcich | 55.85 | 37.70 | 63.78 | 127 | Giants |
| 78 | 13 | Hayden Hurst | 54.78 | 37.30 | 62.26 | 103 | Chargers |
| 79 | 14 | Julian Hill | 53.69 | 37.20 | 60.52 | 228 | Dolphins |

## WR — Wide Receiver

- **Season used:** `2024`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nico Collins | 90.64 | 92.30 | 85.37 | 447 | Texans |
| 2 | 2 | A.J. Brown | 89.19 | 90.30 | 84.28 | 473 | Eagles |
| 3 | 3 | Puka Nacua | 89.04 | 92.50 | 82.57 | 370 | Rams |
| 4 | 4 | Justin Jefferson | 86.67 | 86.60 | 82.55 | 700 | Vikings |
| 5 | 5 | Mike Evans | 86.37 | 90.40 | 79.51 | 463 | Buccaneers |
| 6 | 6 | Amon-Ra St. Brown | 86.05 | 89.30 | 79.71 | 611 | Lions |
| 7 | 7 | Ladd McConkey | 85.63 | 84.30 | 82.35 | 553 | Chargers |
| 8 | 8 | Jordan Whittington | 84.83 | 78.30 | 85.01 | 129 | Rams |
| 9 | 9 | Brian Thomas Jr. | 84.80 | 82.00 | 82.50 | 552 | Jaguars |
| 10 | 10 | Ja'Marr Chase | 84.06 | 85.80 | 78.74 | 745 | Bengals |
| 11 | 11 | Drake London | 83.92 | 87.80 | 77.17 | 595 | Falcons |
| 12 | 12 | Tee Higgins | 83.79 | 88.20 | 76.68 | 476 | Bengals |
| 13 | 13 | Chris Godwin | 82.43 | 86.30 | 75.68 | 265 | Buccaneers |
| 14 | 14 | Chris Olave | 82.36 | 82.40 | 78.17 | 200 | Saints |
| 15 | 15 | Malik Nabers | 82.16 | 86.70 | 74.96 | 600 | Giants |
| 16 | 16 | Terry McLaurin | 81.97 | 82.10 | 77.71 | 717 | Commanders |
| 17 | 17 | Zay Flowers | 81.87 | 82.10 | 77.55 | 499 | Ravens |
| 18 | 18 | Josh Downs | 81.77 | 84.80 | 75.58 | 381 | Colts |
| 19 | 19 | Brandon Aiyuk | 81.44 | 74.60 | 81.83 | 225 | 49ers |
| 20 | 20 | Khalil Shakir | 81.12 | 78.90 | 78.44 | 495 | Bills |
| 21 | 21 | Jameson Williams | 81.10 | 74.50 | 81.33 | 535 | Lions |
| 22 | 22 | George Pickens | 80.99 | 78.60 | 78.42 | 498 | Steelers |
| 23 | 23 | Jauan Jennings | 80.91 | 83.10 | 75.28 | 459 | 49ers |
| 24 | 24 | CeeDee Lamb | 80.68 | 77.30 | 78.76 | 566 | Cowboys |
| 25 | 25 | DeVonta Smith | 80.66 | 80.90 | 76.34 | 505 | Eagles |
| 26 | 26 | Tyreek Hill | 80.47 | 72.70 | 81.48 | 580 | Dolphins |

### Good (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | Tylan Wallace | 79.59 | 66.20 | 84.35 | 148 | Ravens |
| 28 | 2 | Marvin Harrison Jr. | 79.54 | 77.70 | 76.60 | 579 | Cardinals |
| 29 | 3 | Jalen Coker | 79.21 | 72.80 | 79.31 | 297 | Panthers |
| 30 | 4 | Christian Watson | 79.20 | 69.10 | 81.76 | 295 | Packers |
| 31 | 5 | Rashid Shaheed | 79.11 | 69.70 | 81.22 | 181 | Saints |
| 32 | 6 | D.K. Metcalf | 79.07 | 74.30 | 78.08 | 590 | Seahawks |
| 33 | 7 | Kalif Raymond | 78.98 | 71.10 | 80.06 | 158 | Lions |
| 34 | 8 | Jaxon Smith-Njigba | 78.90 | 81.00 | 73.33 | 666 | Seahawks |
| 35 | 9 | Stefon Diggs | 78.79 | 79.00 | 74.48 | 282 | Texans |
| 36 | 10 | Davante Adams | 78.53 | 75.80 | 76.18 | 557 | Jets |
| 37 | 11 | Jakobi Meyers | 78.45 | 77.70 | 74.79 | 627 | Raiders |
| 38 | 12 | DeAndre Hopkins | 78.36 | 77.70 | 74.63 | 419 | Chiefs |
| 39 | 13 | Marvin Mims Jr. | 78.26 | 68.60 | 80.54 | 198 | Broncos |
| 40 | 14 | Alec Pierce | 78.01 | 74.30 | 76.31 | 485 | Colts |
| 41 | 15 | Garrett Wilson | 77.97 | 78.90 | 73.19 | 691 | Jets |
| 42 | 16 | Jaylen Waddle | 77.82 | 72.10 | 77.47 | 513 | Dolphins |
| 43 | 17 | Jayden Reed | 77.57 | 71.70 | 77.32 | 430 | Packers |
| 44 | 18 | DJ Moore | 77.49 | 73.50 | 75.98 | 722 | Bears |
| 45 | 19 | Deebo Samuel | 77.46 | 70.90 | 77.67 | 405 | 49ers |
| 46 | 20 | Darnell Mooney | 77.27 | 74.00 | 75.29 | 557 | Falcons |
| 47 | 21 | Jerry Jeudy | 77.13 | 73.50 | 75.39 | 757 | Browns |
| 48 | 22 | Keon Coleman | 76.81 | 68.30 | 78.31 | 404 | Bills |
| 49 | 23 | Calvin Ridley | 76.48 | 73.40 | 74.37 | 582 | Titans |
| 50 | 24 | Jordan Addison | 76.44 | 72.90 | 74.64 | 600 | Vikings |
| 51 | 25 | Cooper Kupp | 76.25 | 71.40 | 75.31 | 455 | Rams |
| 52 | 26 | Tutu Atwell | 76.24 | 72.70 | 74.44 | 277 | Rams |
| 53 | 27 | KaVontae Turpin | 76.23 | 71.20 | 75.41 | 210 | Cowboys |
| 54 | 28 | Rashod Bateman | 75.93 | 71.80 | 74.51 | 522 | Ravens |
| 55 | 29 | Tank Dell | 75.92 | 73.10 | 73.63 | 467 | Texans |
| 56 | 30 | Courtland Sutton | 75.82 | 75.50 | 71.87 | 650 | Broncos |
| 57 | 31 | Adam Thielen | 75.79 | 76.40 | 71.21 | 319 | Panthers |
| 58 | 32 | Amari Cooper | 75.52 | 68.70 | 75.90 | 451 | Bills |
| 59 | 33 | Noah Brown | 75.21 | 70.70 | 74.05 | 295 | Commanders |
| 60 | 34 | Michael Pittman Jr. | 75.10 | 72.20 | 72.87 | 511 | Colts |
| 61 | 35 | Christian Kirk | 75.03 | 67.90 | 75.62 | 230 | Jaguars |
| 62 | 36 | Demario Douglas | 74.17 | 70.00 | 72.79 | 477 | Patriots |
| 63 | 37 | Dyami Brown | 74.06 | 66.50 | 74.94 | 377 | Commanders |

### Starter (76 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 64 | 1 | Tim Patrick | 73.90 | 66.80 | 74.47 | 371 | Lions |
| 65 | 2 | Romeo Doubs | 73.57 | 69.70 | 71.99 | 406 | Packers |
| 66 | 3 | KhaDarel Hodge | 73.50 | 62.80 | 76.46 | 121 | Falcons |
| 67 | 4 | Jermaine Burton | 73.48 | 58.40 | 79.37 | 100 | Bengals |
| 68 | 5 | Joshua Palmer | 72.87 | 67.00 | 72.61 | 428 | Chargers |
| 69 | 6 | Calvin Austin III | 72.77 | 63.80 | 74.58 | 423 | Steelers |
| 70 | 7 | Quentin Johnston | 72.56 | 67.40 | 71.83 | 464 | Chargers |
| 71 | 8 | Xavier Worthy | 72.52 | 68.70 | 70.90 | 603 | Chiefs |
| 72 | 9 | Demarcus Robinson | 72.20 | 65.00 | 72.83 | 618 | Rams |
| 73 | 10 | Diontae Johnson | 72.06 | 65.10 | 72.53 | 277 | Texans |
| 74 | 11 | Ricky Pearsall | 72.00 | 63.90 | 73.24 | 324 | 49ers |
| 75 | 12 | Olamide Zaccheaus | 71.99 | 69.20 | 69.69 | 403 | Commanders |
| 76 | 13 | Tyler Johnson | 71.86 | 67.00 | 70.93 | 215 | Rams |
| 77 | 14 | Mike Williams | 71.85 | 58.80 | 76.38 | 366 | Steelers |
| 78 | 15 | Marquez Valdes-Scantling | 71.67 | 61.70 | 74.15 | 315 | Saints |
| 79 | 16 | Bo Melton | 71.66 | 63.20 | 73.13 | 118 | Packers |
| 80 | 17 | Devaughn Vele | 71.46 | 69.30 | 68.73 | 349 | Broncos |
| 81 | 18 | Rome Odunze | 71.39 | 63.80 | 72.28 | 677 | Bears |
| 82 | 19 | Keenan Allen | 71.39 | 64.40 | 71.89 | 596 | Bears |
| 83 | 20 | Tyler Lockett | 71.31 | 65.20 | 71.22 | 589 | Seahawks |
| 84 | 21 | Dontayvion Wicks | 71.15 | 65.10 | 71.02 | 345 | Packers |
| 85 | 22 | Rakim Jarrett | 71.00 | 59.50 | 74.50 | 117 | Buccaneers |
| 86 | 23 | Dante Pettis | 70.95 | 70.20 | 67.28 | 104 | Saints |
| 87 | 24 | Nick Westbrook-Ikhine | 70.94 | 63.40 | 71.80 | 469 | Titans |
| 88 | 25 | Josh Reynolds | 70.92 | 61.20 | 73.24 | 220 | Jaguars |
| 89 | 26 | Michael Wilson | 70.62 | 62.90 | 71.60 | 538 | Cardinals |
| 90 | 27 | Jalen Brooks | 70.62 | 59.20 | 74.06 | 236 | Cowboys |
| 91 | 28 | JuJu Smith-Schuster | 70.61 | 60.00 | 73.52 | 321 | Chiefs |
| 92 | 29 | Greg Dortch | 70.61 | 63.90 | 70.91 | 299 | Cardinals |
| 93 | 30 | Ray-Ray McCloud III | 70.47 | 62.70 | 71.49 | 598 | Falcons |
| 94 | 31 | Curtis Samuel | 70.32 | 65.50 | 69.36 | 263 | Bills |
| 95 | 32 | Kendrick Bourne | 70.29 | 62.10 | 71.58 | 317 | Patriots |
| 96 | 33 | Nelson Agholor | 70.20 | 63.40 | 70.57 | 250 | Ravens |
| 97 | 34 | Mack Hollins | 70.11 | 61.60 | 71.61 | 495 | Bills |
| 98 | 35 | Ryan Flournoy | 69.89 | 63.90 | 69.72 | 103 | Cowboys |
| 99 | 36 | Darius Slayton | 69.79 | 59.00 | 72.81 | 575 | Giants |
| 100 | 37 | Brandin Cooks | 69.67 | 63.20 | 69.82 | 317 | Cowboys |
| 101 | 38 | Cedrick Wilson Jr. | 69.67 | 63.50 | 69.62 | 220 | Saints |
| 102 | 39 | Chris Conley | 69.25 | 59.80 | 71.39 | 124 | 49ers |
| 103 | 40 | Kayshon Boutte | 69.20 | 61.40 | 70.24 | 507 | Patriots |
| 104 | 41 | Allen Lazard | 69.16 | 62.70 | 69.30 | 451 | Jets |
| 105 | 42 | Adonai Mitchell | 69.14 | 57.90 | 72.46 | 221 | Colts |
| 106 | 43 | Cedric Tillman | 68.96 | 63.60 | 68.36 | 300 | Browns |
| 107 | 44 | Tyler Boyd | 68.57 | 60.00 | 70.12 | 464 | Titans |
| 108 | 45 | Jalen Nailor | 68.31 | 59.30 | 70.15 | 462 | Vikings |
| 109 | 46 | Jalen McMillan | 68.20 | 60.80 | 68.96 | 430 | Buccaneers |
| 110 | 47 | Jalen Tolbert | 68.10 | 60.70 | 68.86 | 597 | Cowboys |
| 111 | 48 | Tre Tucker | 67.81 | 57.50 | 70.51 | 683 | Raiders |
| 112 | 49 | Trey Palmer | 67.64 | 58.20 | 69.76 | 188 | Buccaneers |
| 113 | 50 | David Moore | 67.62 | 63.10 | 66.46 | 358 | Panthers |
| 114 | 51 | Gabe Davis | 67.47 | 52.50 | 73.29 | 264 | Jaguars |
| 115 | 52 | Devin Duvernay | 67.37 | 59.30 | 68.58 | 141 | Jaguars |
| 116 | 53 | Wan'Dale Robinson | 67.11 | 63.40 | 65.42 | 618 | Giants |
| 117 | 54 | Kevin Austin Jr. | 66.90 | 56.00 | 70.00 | 210 | Saints |
| 118 | 55 | Ryan Miller | 66.55 | 59.00 | 67.42 | 151 | Buccaneers |
| 119 | 56 | Parker Washington | 66.45 | 59.70 | 66.79 | 404 | Jaguars |
| 120 | 57 | John Metchie III | 66.44 | 59.30 | 67.03 | 314 | Texans |
| 121 | 58 | Brandon Powell | 66.08 | 58.90 | 66.70 | 130 | Vikings |
| 122 | 59 | Robert Woods | 65.88 | 58.80 | 66.44 | 226 | Texans |
| 123 | 60 | Simi Fehoko | 65.73 | 55.00 | 68.71 | 136 | Chargers |
| 124 | 61 | DJ Turner | 65.66 | 54.90 | 68.66 | 246 | Raiders |
| 125 | 62 | Justin Watson | 65.53 | 53.30 | 69.52 | 430 | Chiefs |
| 126 | 63 | Van Jefferson | 65.41 | 57.20 | 66.71 | 442 | Steelers |
| 127 | 64 | Elijah Moore | 65.12 | 58.50 | 65.36 | 623 | Browns |
| 128 | 65 | Xavier Hutchinson | 64.75 | 57.80 | 65.21 | 326 | Texans |
| 129 | 66 | Jahan Dotson | 64.57 | 53.70 | 67.65 | 492 | Eagles |
| 130 | 67 | Troy Franklin | 64.47 | 54.30 | 67.09 | 297 | Broncos |
| 131 | 68 | Malik Washington | 64.13 | 57.70 | 64.25 | 270 | Dolphins |
| 132 | 69 | Xavier Legette | 64.10 | 59.30 | 63.13 | 441 | Panthers |
| 133 | 70 | K.J. Osborn | 63.96 | 52.20 | 67.63 | 143 | Commanders |
| 134 | 71 | Luke McCaffrey | 63.95 | 54.30 | 66.21 | 283 | Commanders |
| 135 | 72 | Zay Jones | 63.53 | 53.10 | 66.31 | 184 | Cardinals |
| 136 | 73 | Jake Bobo | 63.25 | 55.40 | 64.31 | 163 | Seahawks |
| 137 | 74 | Andrei Iosivas | 63.17 | 52.70 | 65.99 | 620 | Bengals |
| 138 | 75 | Sterling Shepard | 62.27 | 54.80 | 63.09 | 393 | Buccaneers |
| 139 | 76 | Jamison Crowder | 62.13 | 54.90 | 62.79 | 127 | Commanders |

### Rotation/backup (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 140 | 1 | Jonathan Mingo | 61.52 | 49.90 | 65.10 | 285 | Cowboys |
| 141 | 2 | Xavier Gipson | 61.43 | 48.60 | 65.81 | 138 | Jets |
| 142 | 3 | Jalin Hyatt | 61.22 | 48.00 | 65.86 | 230 | Giants |
| 143 | 4 | Michael Woods II | 60.43 | 48.60 | 64.15 | 211 | Browns |
| 144 | 5 | Johnny Wilson | 58.12 | 49.80 | 59.50 | 177 | Eagles |
| 145 | 6 | Mason Tipton | 56.39 | 48.00 | 57.81 | 253 | Saints |
| 146 | 7 | Ja'Lynn Polk | 54.57 | 43.10 | 58.05 | 272 | Patriots |
