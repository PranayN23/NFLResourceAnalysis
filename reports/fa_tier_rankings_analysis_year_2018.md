# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:38:04Z
- **Requested analysis_year:** 2018 (clamped to 2018)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jason Kelce | 95.51 | 88.80 | 95.81 | 1037 | Eagles |
| 2 | 2 | Rodney Hudson | 88.43 | 81.90 | 88.62 | 1042 | Raiders |
| 3 | 3 | Alex Mack | 88.13 | 80.00 | 89.38 | 1057 | Falcons |
| 4 | 4 | Matt Paradis | 87.84 | 78.32 | 90.02 | 569 | Broncos |
| 5 | 5 | Corey Linsley | 85.58 | 78.60 | 86.06 | 1074 | Packers |
| 6 | 6 | Brandon Linder | 85.37 | 76.28 | 87.27 | 507 | Jaguars |
| 7 | 7 | J.C. Tretter | 84.29 | 76.90 | 85.05 | 1091 | Browns |
| 8 | 8 | Ryan Kelly | 83.08 | 73.53 | 85.28 | 777 | Colts |
| 9 | 9 | Austin Reiter | 82.59 | 69.67 | 87.03 | 266 | Chiefs |
| 10 | 10 | David Andrews | 82.23 | 73.30 | 84.02 | 1104 | Patriots |
| 11 | 11 | Ben Jones | 81.19 | 72.70 | 82.69 | 986 | Titans |
| 12 | 12 | Graham Glasgow | 80.84 | 71.10 | 83.16 | 1076 | Lions |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Maurkice Pouncey | 79.57 | 71.30 | 80.91 | 1101 | Steelers |
| 14 | 2 | Mitch Morse | 79.53 | 69.25 | 82.22 | 678 | Chiefs |
| 15 | 3 | Cody Whitehair | 78.05 | 75.20 | 75.78 | 1075 | Bears |
| 16 | 4 | Ryan Kalil | 76.35 | 66.60 | 78.69 | 1028 | Panthers |
| 17 | 5 | Travis Swanson | 75.39 | 65.13 | 78.06 | 644 | Dolphins |
| 18 | 6 | Nick Martin | 75.23 | 65.40 | 77.62 | 1094 | Texans |
| 19 | 7 | Chase Roullier | 75.18 | 65.00 | 77.80 | 1020 | Commanders |
| 20 | 8 | Max Unger | 74.65 | 64.50 | 77.25 | 1012 | Saints |

### Starter (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Justin Britt | 73.96 | 62.00 | 77.76 | 989 | Seahawks |
| 22 | 2 | Russell Bodine | 73.43 | 61.78 | 77.03 | 588 | Bills |
| 23 | 3 | John Greco | 73.03 | 60.42 | 77.27 | 488 | Giants |
| 24 | 4 | Jon Halapio | 72.92 | 66.56 | 73.00 | 116 | Giants |
| 25 | 5 | Mike Pouncey | 71.69 | 59.11 | 75.91 | 954 | Chargers |
| 26 | 6 | Joey Hunt | 71.23 | 59.32 | 75.01 | 115 | Seahawks |
| 27 | 7 | Jonotthan Harrison | 71.07 | 57.63 | 75.87 | 506 | Jets |
| 28 | 8 | Matt Skura | 71.03 | 59.20 | 74.75 | 1188 | Ravens |
| 29 | 9 | Billy Price | 70.63 | 56.69 | 75.75 | 558 | Bengals |
| 30 | 10 | Spencer Pulley | 70.08 | 58.71 | 73.49 | 567 | Giants |
| 31 | 11 | Ryan Jensen | 69.77 | 56.60 | 74.39 | 1116 | Buccaneers |
| 32 | 12 | Weston Richburg | 69.18 | 56.62 | 73.38 | 968 | 49ers |
| 33 | 13 | Jake Brendel | 69.04 | 59.49 | 71.24 | 176 | Dolphins |
| 34 | 14 | Daniel Kilgore | 68.78 | 57.72 | 71.98 | 182 | Dolphins |
| 35 | 15 | Brett Jones | 68.52 | 56.96 | 72.06 | 191 | Vikings |
| 36 | 16 | Trey Hopkins | 68.33 | 60.00 | 69.72 | 589 | Bengals |
| 37 | 17 | Mason Cole | 67.93 | 53.74 | 73.22 | 942 | Cardinals |
| 38 | 18 | Joe Looney | 67.44 | 55.10 | 71.50 | 1076 | Cowboys |
| 39 | 19 | John Sullivan | 67.12 | 53.80 | 71.84 | 1054 | Rams |
| 40 | 20 | Spencer Long | 62.79 | 48.34 | 68.25 | 805 | Jets |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 41 | 1 | Pat Elflein | 60.39 | 43.06 | 67.78 | 863 | Vikings |

## CB — Cornerback

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Stephon Gilmore | 93.18 | 90.90 | 90.73 | 1013 | Patriots |
| 2 | 2 | Kyle Fuller | 87.87 | 84.10 | 86.21 | 1014 | Bears |
| 3 | 3 | Byron Jones | 87.28 | 83.30 | 85.76 | 1021 | Cowboys |
| 4 | 4 | Chris Harris Jr. | 86.17 | 83.69 | 85.74 | 747 | Broncos |
| 5 | 5 | Patrick Peterson | 85.95 | 83.70 | 83.28 | 1107 | Cardinals |
| 6 | 6 | Casey Hayward Jr. | 85.19 | 77.70 | 86.01 | 1016 | Chargers |
| 7 | 7 | Jason McCourty | 84.14 | 80.00 | 83.76 | 835 | Patriots |
| 8 | 8 | Darius Slay | 83.78 | 78.20 | 84.27 | 875 | Lions |
| 9 | 9 | Nickell Robey-Coleman | 83.71 | 79.01 | 82.67 | 556 | Rams |
| 10 | 10 | A.J. Bouye | 83.40 | 79.30 | 83.54 | 827 | Jaguars |
| 11 | 11 | Denzel Ward | 83.33 | 83.60 | 82.12 | 841 | Browns |
| 12 | 12 | Marlon Humphrey | 83.09 | 79.29 | 82.75 | 718 | Ravens |
| 13 | 13 | Johnathan Joseph | 82.51 | 80.59 | 80.87 | 811 | Texans |
| 14 | 14 | Bryce Callahan | 82.14 | 80.09 | 83.19 | 676 | Bears |
| 15 | 15 | J.C. Jackson | 80.91 | 72.66 | 85.37 | 395 | Patriots |
| 16 | 16 | Marshon Lattimore | 80.37 | 71.70 | 82.36 | 907 | Saints |
| 17 | 17 | Prince Amukamara | 80.32 | 76.20 | 80.46 | 911 | Bears |
| 18 | 18 | William Jackson III | 80.02 | 72.90 | 81.39 | 1063 | Bengals |

### Good (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Jalen Ramsey | 79.91 | 71.90 | 81.08 | 1019 | Jaguars |
| 20 | 2 | Aqib Talib | 79.87 | 73.71 | 84.91 | 388 | Rams |
| 21 | 3 | Justin Coleman | 79.78 | 76.76 | 79.81 | 672 | Seahawks |
| 22 | 4 | Xavien Howard | 79.77 | 75.09 | 82.48 | 803 | Dolphins |
| 23 | 5 | Levi Wallace | 79.25 | 77.97 | 86.81 | 416 | Bills |
| 24 | 6 | Pierre Desir | 79.20 | 73.10 | 82.95 | 903 | Colts |
| 25 | 7 | Trumaine Johnson | 78.87 | 74.20 | 81.36 | 670 | Jets |
| 26 | 8 | Holton Hill | 78.79 | 71.14 | 81.80 | 376 | Vikings |
| 27 | 9 | Mike Hilton | 78.35 | 70.91 | 79.79 | 594 | Steelers |
| 28 | 10 | Sherrick McManis | 78.31 | 73.83 | 83.18 | 237 | Bears |
| 29 | 11 | Josh Norman | 78.20 | 72.00 | 78.80 | 1028 | Commanders |
| 30 | 12 | Malcolm Butler | 77.93 | 69.80 | 79.18 | 836 | Titans |
| 31 | 13 | Logan Ryan | 77.82 | 70.10 | 79.84 | 855 | Titans |
| 32 | 14 | Steven Nelson | 77.77 | 73.60 | 78.26 | 1164 | Chiefs |
| 33 | 15 | Adoree' Jackson | 77.06 | 69.00 | 78.27 | 959 | Titans |
| 34 | 16 | Mackensie Alexander | 76.82 | 70.34 | 78.95 | 564 | Vikings |
| 35 | 17 | Brandon Carr | 76.60 | 69.10 | 77.43 | 876 | Ravens |
| 36 | 18 | Joe Haden | 76.56 | 70.90 | 78.56 | 937 | Steelers |
| 37 | 19 | Desmond Trufant | 76.13 | 67.20 | 79.38 | 1061 | Falcons |
| 38 | 20 | D.J. Hayden | 76.11 | 71.82 | 78.97 | 456 | Jaguars |
| 39 | 21 | Richard Sherman | 75.99 | 68.10 | 80.31 | 836 | 49ers |
| 40 | 22 | Jaire Alexander | 75.70 | 72.49 | 76.80 | 761 | Packers |
| 41 | 23 | Kendall Fuller | 75.55 | 70.20 | 76.30 | 1078 | Chiefs |
| 42 | 24 | Isaiah Oliver | 74.87 | 65.19 | 82.35 | 241 | Falcons |
| 43 | 25 | Jourdan Lewis | 74.12 | 67.00 | 77.70 | 187 | Cowboys |
| 44 | 26 | Donte Jackson | 74.09 | 67.20 | 74.52 | 895 | Panthers |

### Starter (71 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 45 | 1 | Kenny Moore II | 73.95 | 68.40 | 77.27 | 911 | Colts |
| 46 | 2 | Orlando Scandrick | 73.76 | 69.96 | 74.82 | 788 | Chiefs |
| 47 | 3 | Ronald Darby | 73.66 | 69.22 | 78.09 | 542 | Eagles |
| 48 | 4 | Tre'Davious White | 73.46 | 62.50 | 76.60 | 961 | Bills |
| 49 | 5 | Janoris Jenkins | 73.36 | 66.30 | 76.08 | 1088 | Giants |
| 50 | 6 | Trae Waynes | 73.21 | 64.22 | 76.29 | 693 | Vikings |
| 51 | 7 | Quinton Dunbar | 73.06 | 65.11 | 80.34 | 373 | Commanders |
| 52 | 8 | Josh Jackson | 72.62 | 61.87 | 75.62 | 721 | Packers |
| 53 | 9 | Rasul Douglas | 72.51 | 66.42 | 75.27 | 544 | Eagles |
| 54 | 10 | Chidobe Awuzie | 72.43 | 66.10 | 75.87 | 886 | Cowboys |
| 55 | 11 | T.J. Carrie | 72.37 | 63.90 | 74.05 | 908 | Browns |
| 56 | 12 | James Bradberry | 72.08 | 64.90 | 73.33 | 994 | Panthers |
| 57 | 13 | Anthony Brown | 72.02 | 62.65 | 74.62 | 690 | Cowboys |
| 58 | 14 | Jimmy Smith | 71.93 | 65.50 | 76.41 | 610 | Ravens |
| 59 | 15 | K'Waun Williams | 71.69 | 65.46 | 73.96 | 595 | 49ers |
| 60 | 16 | Brent Grimes | 71.25 | 60.99 | 76.42 | 791 | Buccaneers |
| 61 | 17 | Briean Boddy-Calhoun | 70.75 | 62.81 | 74.28 | 656 | Browns |
| 62 | 18 | Marcus Peters | 70.67 | 58.10 | 75.20 | 914 | Rams |
| 63 | 19 | Gareon Conley | 70.66 | 61.36 | 79.46 | 679 | Raiders |
| 64 | 20 | Terrance Mitchell | 70.29 | 63.16 | 77.74 | 445 | Browns |
| 65 | 21 | Tavon Young | 70.13 | 62.37 | 71.65 | 602 | Ravens |
| 66 | 22 | Dre Kirkpatrick | 69.72 | 62.52 | 72.76 | 774 | Bengals |
| 67 | 23 | Taron Johnson | 69.52 | 66.73 | 72.42 | 405 | Bills |
| 68 | 24 | Captain Munnerlyn | 69.45 | 60.28 | 71.91 | 630 | Panthers |
| 69 | 25 | Michael Davis | 69.44 | 64.97 | 74.51 | 627 | Chargers |
| 70 | 26 | Jonathan Jones | 69.34 | 60.79 | 73.28 | 516 | Patriots |
| 71 | 27 | Coty Sensabaugh | 69.30 | 64.47 | 73.25 | 745 | Steelers |
| 72 | 28 | Cre'Von LeBlanc | 69.19 | 62.65 | 75.64 | 382 | Eagles |
| 73 | 29 | Bradley Roby | 69.06 | 58.60 | 72.39 | 926 | Broncos |
| 74 | 30 | B.W. Webb | 68.88 | 60.40 | 72.65 | 1004 | Giants |
| 75 | 31 | Patrick Robinson | 68.86 | 63.90 | 76.65 | 110 | Saints |
| 76 | 32 | Eli Apple | 68.55 | 62.30 | 71.15 | 905 | Saints |
| 77 | 33 | Darqueze Dennard | 68.53 | 60.66 | 71.38 | 675 | Bengals |
| 78 | 34 | Morris Claiborne | 68.25 | 60.30 | 71.88 | 1002 | Jets |
| 79 | 35 | Greg Stroman Jr. | 67.91 | 65.89 | 68.22 | 387 | Commanders |
| 80 | 36 | Ryan Smith | 67.76 | 61.28 | 71.24 | 419 | Buccaneers |
| 81 | 37 | Xavier Rhodes | 67.61 | 55.20 | 73.18 | 771 | Vikings |
| 82 | 38 | Tramaine Brock Sr. | 67.53 | 60.42 | 73.31 | 436 | Broncos |
| 83 | 39 | Quincy Wilson | 67.26 | 62.11 | 72.00 | 435 | Colts |
| 84 | 40 | Avonte Maddox | 67.20 | 60.57 | 73.71 | 541 | Eagles |
| 85 | 41 | Javien Elliott | 66.89 | 65.22 | 70.60 | 351 | Buccaneers |
| 86 | 42 | Rashaan Melvin | 66.82 | 56.62 | 72.78 | 604 | Raiders |
| 87 | 43 | Artie Burns | 66.71 | 58.01 | 72.00 | 308 | Steelers |
| 88 | 44 | Robert Alford | 66.70 | 53.40 | 71.92 | 959 | Falcons |
| 89 | 45 | Troy Hill | 66.66 | 58.71 | 72.37 | 427 | Rams |
| 90 | 46 | Trevor Williams | 66.10 | 58.68 | 72.51 | 410 | Chargers |
| 91 | 47 | Bashaud Breeland | 66.02 | 57.82 | 72.73 | 330 | Packers |
| 92 | 48 | E.J. Gaines | 65.84 | 61.70 | 72.96 | 181 | Browns |
| 93 | 49 | Eric Rowe | 65.71 | 59.39 | 74.40 | 136 | Patriots |
| 94 | 50 | Grant Haley | 65.55 | 67.57 | 66.29 | 429 | Giants |
| 95 | 51 | Shaquill Griffin | 65.13 | 51.90 | 70.17 | 941 | Seahawks |
| 96 | 52 | Tre Flowers | 65.11 | 56.50 | 67.72 | 903 | Seahawks |
| 97 | 53 | David Amerson | 65.04 | 58.64 | 73.47 | 293 | Cardinals |
| 98 | 54 | Nevin Lawson | 64.99 | 54.70 | 68.52 | 877 | Lions |
| 99 | 55 | Shareece Wright | 64.99 | 58.83 | 69.09 | 506 | Texans |
| 100 | 56 | Sam Shields | 64.65 | 57.72 | 72.51 | 340 | Rams |
| 101 | 57 | Leon Hall | 64.61 | 57.68 | 71.00 | 366 | Raiders |
| 102 | 58 | Jalen Mills | 64.16 | 56.87 | 73.19 | 457 | Eagles |
| 103 | 59 | Phillip Gaines | 63.85 | 57.75 | 70.32 | 408 | Browns |
| 104 | 60 | Mike Hughes | 63.74 | 62.50 | 73.81 | 244 | Vikings |
| 105 | 61 | Carlton Davis III | 63.71 | 59.00 | 65.81 | 1436 | Buccaneers |
| 106 | 62 | Bobby McCain | 63.36 | 54.00 | 66.46 | 823 | Dolphins |
| 107 | 63 | P.J. Williams | 63.36 | 51.38 | 70.62 | 693 | Saints |
| 108 | 64 | Kevin King | 63.20 | 59.94 | 70.45 | 305 | Packers |
| 109 | 65 | Daryl Worley | 62.93 | 53.04 | 68.69 | 505 | Raiders |
| 110 | 66 | LeShaun Sims | 62.70 | 58.37 | 65.79 | 215 | Titans |
| 111 | 67 | Ken Crawley | 62.58 | 50.87 | 70.29 | 409 | Saints |
| 112 | 68 | Tyler Patmon | 62.49 | 58.94 | 68.83 | 231 | Jaguars |
| 113 | 69 | Buster Skrine | 62.30 | 51.57 | 67.06 | 693 | Jets |
| 114 | 70 | Charvarius Ward | 62.29 | 59.75 | 78.32 | 140 | Chiefs |
| 115 | 71 | Bene Benwikere | 62.12 | 55.82 | 69.23 | 592 | Raiders |

### Rotation/backup (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 116 | 1 | Kevin Toliver II | 61.82 | 57.89 | 66.53 | 136 | Bears |
| 117 | 2 | Isaac Yiadom | 61.74 | 57.51 | 65.59 | 264 | Broncos |
| 118 | 3 | Fabian Moreau | 61.48 | 58.30 | 63.34 | 840 | Commanders |
| 119 | 4 | Tony Brown | 59.76 | 53.30 | 66.15 | 290 | Packers |
| 120 | 5 | Nate Hairston | 59.38 | 51.79 | 64.95 | 413 | Colts |
| 121 | 6 | Aaron Colvin | 59.38 | 54.83 | 63.67 | 317 | Texans |
| 122 | 7 | Ryan Lewis | 59.21 | 59.36 | 70.91 | 150 | Bills |
| 123 | 8 | Greg Mabin | 57.83 | 55.07 | 65.00 | 163 | 49ers |
| 124 | 9 | Darius Phillips | 57.48 | 56.92 | 58.88 | 232 | Bengals |
| 125 | 10 | KeiVarae Russell | 57.39 | 58.08 | 64.01 | 137 | Bengals |
| 126 | 11 | Cameron Sutton | 56.37 | 57.79 | 59.20 | 240 | Steelers |
| 127 | 12 | M.J. Stewart | 56.04 | 57.65 | 59.13 | 300 | Buccaneers |
| 128 | 13 | Ahkello Witherspoon | 54.41 | 40.76 | 62.86 | 700 | 49ers |
| 129 | 14 | Torry McTyer | 54.32 | 52.35 | 58.50 | 347 | Dolphins |
| 130 | 15 | Sidney Jones IV | 53.82 | 52.27 | 61.10 | 321 | Eagles |
| 131 | 16 | Jamal Agnew | 53.79 | 54.24 | 61.56 | 117 | Lions |
| 132 | 17 | Jamar Taylor | 52.65 | 44.24 | 56.69 | 305 | Broncos |
| 133 | 18 | Parry Nickerson | 50.81 | 51.87 | 54.27 | 213 | Jets |
| 134 | 19 | Nick Nelson | 45.87 | 44.53 | 48.84 | 311 | Raiders |

## DI — Defensive Interior

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 94.35 | 90.15 | 93.30 | 913 | Rams |
| 2 | 2 | Geno Atkins | 87.61 | 85.07 | 85.13 | 795 | Bengals |
| 3 | 3 | Kawann Short | 87.45 | 84.09 | 86.56 | 584 | Panthers |
| 4 | 4 | Jurrell Casey | 86.23 | 86.95 | 82.32 | 745 | Titans |
| 5 | 5 | Leonard Williams | 86.00 | 89.11 | 79.76 | 866 | Jets |
| 6 | 6 | Fletcher Cox | 85.72 | 89.24 | 79.21 | 831 | Eagles |
| 7 | 7 | DeForest Buckner | 85.51 | 89.65 | 78.78 | 852 | 49ers |
| 8 | 8 | Damon Harrison Sr. | 84.90 | 82.38 | 82.42 | 606 | Lions |
| 9 | 9 | Akiem Hicks | 84.70 | 82.61 | 81.93 | 781 | Bears |
| 10 | 10 | Ndamukong Suh | 84.29 | 84.83 | 79.76 | 887 | Rams |
| 11 | 11 | Kenny Clark | 83.88 | 86.21 | 80.04 | 720 | Packers |
| 12 | 12 | Grady Jarrett | 83.79 | 82.38 | 81.60 | 712 | Falcons |
| 13 | 13 | Bilal Nichols | 82.70 | 70.72 | 88.60 | 329 | Bears |
| 14 | 14 | Cameron Heyward | 82.50 | 83.82 | 79.34 | 842 | Steelers |
| 15 | 15 | Javon Hargrave | 82.14 | 80.22 | 79.26 | 455 | Steelers |
| 16 | 16 | Michael Pierce | 81.73 | 78.35 | 80.85 | 388 | Ravens |
| 17 | 17 | Shelby Harris | 81.51 | 78.44 | 81.28 | 391 | Broncos |
| 18 | 18 | Marcell Dareus | 81.01 | 80.55 | 79.33 | 563 | Jaguars |
| 19 | 19 | Malik Jackson | 80.77 | 73.47 | 81.47 | 628 | Jaguars |
| 20 | 20 | Jonathan Allen | 80.72 | 82.22 | 79.86 | 780 | Commanders |
| 21 | 21 | Linval Joseph | 80.33 | 78.93 | 77.61 | 671 | Vikings |

### Good (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Gerald McCoy | 79.94 | 83.92 | 74.69 | 732 | Buccaneers |
| 23 | 2 | Sheldon Richardson | 79.68 | 75.28 | 79.79 | 719 | Vikings |
| 24 | 3 | Stephon Tuitt | 79.49 | 83.05 | 74.93 | 694 | Steelers |
| 25 | 4 | Henry Anderson | 79.24 | 78.47 | 78.82 | 668 | Jets |
| 26 | 5 | Vincent Taylor | 79.12 | 71.39 | 86.87 | 204 | Dolphins |
| 27 | 6 | Mike Daniels | 78.81 | 72.86 | 82.36 | 419 | Packers |
| 28 | 7 | Da'Shawn Hand | 78.75 | 78.27 | 78.03 | 455 | Lions |
| 29 | 8 | B.J. Hill | 78.65 | 72.29 | 78.72 | 642 | Giants |
| 30 | 9 | Eddie Goldman | 78.63 | 78.11 | 77.21 | 552 | Bears |
| 31 | 10 | Taven Bryan | 78.02 | 72.35 | 77.63 | 301 | Jaguars |
| 32 | 11 | Sheldon Rankins | 77.71 | 78.60 | 74.41 | 642 | Saints |
| 33 | 12 | Chris Jones | 77.10 | 73.35 | 76.07 | 773 | Chiefs |
| 34 | 13 | Larry Ogunjobi | 76.65 | 67.47 | 79.38 | 930 | Browns |
| 35 | 14 | Dean Lowry | 76.62 | 71.66 | 75.76 | 698 | Packers |
| 36 | 15 | Dalvin Tomlinson | 76.60 | 74.53 | 73.81 | 628 | Giants |
| 37 | 16 | Vita Vea | 76.58 | 77.64 | 74.84 | 493 | Buccaneers |
| 38 | 17 | Lawrence Guy Sr. | 76.48 | 68.06 | 77.92 | 519 | Patriots |
| 39 | 18 | Denico Autry | 76.37 | 71.03 | 77.85 | 555 | Colts |
| 40 | 19 | A'Shawn Robinson | 75.90 | 73.48 | 74.91 | 415 | Lions |
| 41 | 20 | Brandon Williams | 75.78 | 69.62 | 76.97 | 518 | Ravens |
| 42 | 21 | Jarran Reed | 75.51 | 67.07 | 77.28 | 773 | Seahawks |
| 43 | 22 | Poona Ford | 75.41 | 70.78 | 79.53 | 231 | Seahawks |
| 44 | 23 | Mike Pennel | 75.26 | 69.45 | 74.96 | 358 | Jets |
| 45 | 24 | Michael Brockers | 75.22 | 72.09 | 73.56 | 679 | Rams |
| 46 | 25 | Muhammad Wilkerson | 75.10 | 69.69 | 82.24 | 115 | Packers |
| 47 | 26 | Daron Payne | 75.03 | 68.71 | 75.07 | 797 | Commanders |
| 48 | 27 | Steve McLendon | 75.02 | 65.50 | 78.24 | 471 | Jets |
| 49 | 28 | DJ Reader | 74.98 | 74.45 | 71.80 | 638 | Texans |
| 50 | 29 | Derek Wolfe | 74.90 | 69.83 | 76.10 | 710 | Broncos |
| 51 | 30 | Danny Shelton | 74.70 | 73.82 | 73.30 | 323 | Patriots |
| 52 | 31 | David Onyemata | 74.61 | 74.38 | 70.59 | 618 | Saints |

### Starter (71 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 53 | 1 | Roy Robertson-Harris | 73.35 | 63.85 | 77.08 | 354 | Bears |
| 54 | 2 | Fadol Brown | 73.35 | 60.44 | 81.95 | 215 | Packers |
| 55 | 3 | Johnathan Hankins | 73.31 | 65.02 | 75.50 | 573 | Raiders |
| 56 | 4 | Christian Covington | 73.26 | 71.59 | 75.11 | 257 | Texans |
| 57 | 5 | Kyle Williams | 73.23 | 59.25 | 78.59 | 657 | Bills |
| 58 | 6 | Abry Jones | 72.91 | 63.07 | 76.04 | 498 | Jaguars |
| 59 | 7 | Rodney Gunter | 72.90 | 62.73 | 75.51 | 641 | Cardinals |
| 60 | 8 | Malcom Brown | 72.61 | 62.05 | 76.00 | 456 | Patriots |
| 61 | 9 | Mario Edwards Jr. | 72.54 | 66.66 | 76.15 | 232 | Giants |
| 62 | 10 | Adam Gotsis | 72.46 | 63.86 | 74.03 | 513 | Broncos |
| 63 | 11 | Xavier Williams | 72.42 | 64.88 | 76.52 | 424 | Chiefs |
| 64 | 12 | Olsen Pierre | 72.20 | 53.60 | 85.12 | 246 | Cardinals |
| 65 | 13 | Maurice Hurst | 72.15 | 70.98 | 71.89 | 472 | Raiders |
| 66 | 14 | Bennie Logan | 71.90 | 55.62 | 79.74 | 230 | Titans |
| 67 | 15 | Zach Kerr | 71.62 | 61.69 | 76.48 | 394 | Broncos |
| 68 | 16 | Tyler Lancaster | 71.59 | 67.78 | 75.16 | 272 | Packers |
| 69 | 17 | Margus Hunt | 70.85 | 59.63 | 74.82 | 724 | Colts |
| 70 | 18 | Matt Ioannidis | 70.35 | 66.60 | 71.81 | 439 | Commanders |
| 71 | 19 | Tyrone Crawford | 70.13 | 59.96 | 73.26 | 633 | Cowboys |
| 72 | 20 | Andrew Billings | 70.12 | 60.50 | 72.75 | 632 | Bengals |
| 73 | 21 | Davon Godchaux | 70.03 | 61.32 | 72.05 | 675 | Dolphins |
| 74 | 22 | Darius Philon | 69.68 | 60.24 | 72.23 | 607 | Chargers |
| 75 | 23 | Corey Liuget | 69.62 | 64.50 | 75.32 | 206 | Chargers |
| 76 | 24 | DaQuan Jones | 69.61 | 65.11 | 69.69 | 587 | Titans |
| 77 | 25 | Dan McCullers | 69.48 | 62.60 | 74.68 | 111 | Steelers |
| 78 | 26 | Treyvon Hester | 68.98 | 69.00 | 68.18 | 226 | Eagles |
| 79 | 27 | Josh Mauro | 68.89 | 59.13 | 76.33 | 270 | Giants |
| 80 | 28 | Ricky Jean Francois | 68.46 | 55.38 | 73.33 | 405 | Lions |
| 81 | 29 | Tim Settle | 68.40 | 58.53 | 72.90 | 135 | Commanders |
| 82 | 30 | Sheldon Day | 67.92 | 57.83 | 73.82 | 275 | 49ers |
| 83 | 31 | Jonathan Bullard | 67.81 | 57.45 | 70.97 | 298 | Bears |
| 84 | 32 | Clinton McDonald | 67.78 | 52.81 | 75.57 | 419 | Raiders |
| 85 | 33 | Dontari Poe | 67.69 | 62.67 | 66.87 | 515 | Panthers |
| 86 | 34 | Brent Urban | 67.48 | 61.85 | 71.14 | 523 | Ravens |
| 87 | 35 | Domata Peko Sr. | 67.42 | 57.45 | 70.53 | 523 | Broncos |
| 88 | 36 | Robert Nkemdiche | 67.32 | 57.44 | 76.40 | 426 | Cardinals |
| 89 | 37 | Ethan Westbrooks | 67.26 | 54.82 | 73.47 | 180 | Rams |
| 90 | 38 | Adolphus Washington | 67.09 | 60.94 | 72.95 | 134 | Bengals |
| 91 | 39 | Deadrin Senat | 66.84 | 61.68 | 67.15 | 370 | Falcons |
| 92 | 40 | Beau Allen | 66.68 | 54.86 | 71.42 | 386 | Buccaneers |
| 93 | 41 | Star Lotulelei | 66.57 | 53.68 | 70.99 | 476 | Bills |
| 94 | 42 | Allen Bailey | 66.39 | 56.24 | 71.59 | 847 | Chiefs |
| 95 | 43 | Al Woods | 66.29 | 54.33 | 71.98 | 375 | Colts |
| 96 | 44 | Corey Peters | 66.20 | 55.67 | 71.04 | 735 | Cardinals |
| 97 | 45 | Vernon Butler | 65.90 | 60.62 | 67.85 | 329 | Panthers |
| 98 | 46 | Tom Johnson | 65.87 | 49.58 | 74.03 | 381 | Vikings |
| 99 | 47 | Tyson Alualu | 65.62 | 55.92 | 68.86 | 311 | Steelers |
| 100 | 48 | Adam Butler | 65.39 | 52.97 | 69.51 | 380 | Patriots |
| 101 | 49 | Stacy McGee | 65.38 | 58.54 | 71.19 | 137 | Commanders |
| 102 | 50 | Haloti Ngata | 65.29 | 55.79 | 72.88 | 368 | Eagles |
| 103 | 51 | Akeem Spence | 65.10 | 50.56 | 70.62 | 665 | Dolphins |
| 104 | 52 | Jordan Phillips | 65.01 | 55.01 | 68.45 | 393 | Bills |
| 105 | 53 | Jack Crawford | 64.83 | 53.79 | 71.77 | 624 | Falcons |
| 106 | 54 | Earl Mitchell | 64.80 | 52.97 | 71.34 | 363 | 49ers |
| 107 | 55 | Harrison Phillips | 64.79 | 58.00 | 65.15 | 389 | Bills |
| 108 | 56 | Brandon Dunn | 64.62 | 58.70 | 66.06 | 347 | Texans |
| 109 | 57 | Derrick Nnadi | 64.40 | 60.50 | 62.84 | 448 | Chiefs |
| 110 | 58 | Sylvester Williams | 64.36 | 54.73 | 67.64 | 376 | Dolphins |
| 111 | 59 | Caraun Reid | 64.34 | 61.34 | 71.86 | 185 | Cowboys |
| 112 | 60 | Brandon Mebane | 64.33 | 51.22 | 72.24 | 405 | Chargers |
| 113 | 61 | Quinton Jefferson | 64.28 | 57.12 | 70.72 | 558 | Seahawks |
| 114 | 62 | Austin Johnson | 64.24 | 57.95 | 65.51 | 399 | Titans |
| 115 | 63 | Montravius Adams | 64.20 | 58.42 | 67.41 | 213 | Packers |
| 116 | 64 | Daniel Ross | 63.98 | 55.67 | 72.39 | 250 | Cowboys |
| 117 | 65 | D.J. Jones | 63.42 | 56.48 | 70.52 | 239 | 49ers |
| 118 | 66 | Christian Ringo | 63.02 | 58.85 | 71.44 | 152 | Bengals |
| 119 | 67 | Nazair Jones | 62.96 | 58.03 | 68.60 | 132 | Seahawks |
| 120 | 68 | Kyle Love | 62.94 | 51.22 | 67.84 | 467 | Panthers |
| 121 | 69 | Maliek Collins | 62.61 | 54.54 | 65.39 | 498 | Cowboys |
| 122 | 70 | Justin Ellis | 62.61 | 56.93 | 67.43 | 133 | Raiders |
| 123 | 71 | Tyeler Davison | 62.22 | 57.85 | 62.21 | 422 | Saints |

### Rotation/backup (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 124 | 1 | Grover Stewart | 61.94 | 55.30 | 63.23 | 292 | Colts |
| 125 | 2 | Chris Wormley | 61.89 | 59.10 | 63.10 | 401 | Ravens |
| 126 | 3 | P.J. Hall | 61.80 | 60.31 | 60.71 | 511 | Raiders |
| 127 | 4 | Antwaun Woods | 61.62 | 57.08 | 66.99 | 585 | Cowboys |
| 128 | 5 | Brian Price | 61.29 | 53.77 | 66.40 | 210 | Browns |
| 129 | 6 | Trevon Coley | 61.26 | 48.99 | 65.66 | 614 | Browns |
| 130 | 7 | Terrell McClain | 60.89 | 48.64 | 67.70 | 374 | Falcons |
| 131 | 8 | Angelo Blackson | 60.76 | 54.77 | 63.40 | 430 | Texans |
| 132 | 9 | Shamar Stephen | 60.41 | 53.18 | 61.58 | 494 | Seahawks |
| 133 | 10 | Damion Square | 60.33 | 49.81 | 64.21 | 530 | Chargers |
| 134 | 11 | Jihad Ward | 60.00 | 61.58 | 63.43 | 144 | Colts |
| 135 | 12 | Nathan Shepherd | 58.93 | 60.88 | 53.46 | 343 | Jets |
| 136 | 13 | Jaleel Johnson | 58.50 | 53.96 | 60.88 | 261 | Vikings |
| 137 | 14 | Taylor Stallworth | 57.70 | 53.69 | 58.29 | 318 | Saints |
| 138 | 15 | Justin Jones | 57.49 | 51.54 | 58.32 | 300 | Chargers |
| 139 | 16 | Destiny Vaeao | 57.45 | 52.92 | 62.75 | 157 | Jets |
| 140 | 17 | Josh Tupou | 57.43 | 61.89 | 63.70 | 154 | Bengals |
| 141 | 18 | William Gholston | 55.35 | 45.30 | 57.88 | 402 | Buccaneers |
| 142 | 19 | Darius Kilgo | 54.22 | 56.20 | 54.72 | 131 | Titans |

## ED — Edge

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 93.49 | 93.77 | 89.13 | 844 | Broncos |
| 2 | 2 | Khalil Mack | 90.98 | 95.29 | 84.97 | 756 | Bears |
| 3 | 3 | DeMarcus Lawrence | 88.19 | 95.31 | 80.52 | 736 | Cowboys |
| 4 | 4 | Myles Garrett | 87.12 | 95.00 | 79.65 | 1012 | Browns |
| 5 | 5 | Justin Houston | 86.35 | 83.93 | 87.96 | 719 | Chiefs |
| 6 | 6 | Joey Bosa | 86.34 | 87.10 | 87.19 | 314 | Chargers |
| 7 | 7 | T.J. Watt | 86.01 | 84.66 | 82.74 | 903 | Steelers |
| 8 | 8 | Danielle Hunter | 85.90 | 84.74 | 82.50 | 879 | Vikings |
| 9 | 9 | Brandon Graham | 85.26 | 87.70 | 79.47 | 755 | Eagles |
| 10 | 10 | Bradley Chubb | 84.07 | 78.97 | 83.31 | 844 | Broncos |
| 11 | 11 | Cameron Jordan | 83.25 | 91.91 | 73.31 | 884 | Saints |
| 12 | 12 | Cameron Wake | 82.14 | 71.10 | 86.36 | 517 | Dolphins |
| 13 | 13 | Trey Flowers | 81.47 | 82.57 | 77.09 | 732 | Patriots |
| 14 | 14 | Jadeveon Clowney | 81.02 | 94.17 | 68.60 | 902 | Texans |
| 15 | 15 | Jerry Hughes | 80.78 | 83.33 | 74.92 | 668 | Bills |
| 16 | 16 | Ryan Kerrigan | 80.75 | 70.89 | 83.16 | 820 | Commanders |
| 17 | 17 | J.J. Watt | 80.39 | 71.17 | 82.37 | 963 | Texans |
| 18 | 18 | Frank Clark | 80.24 | 74.00 | 80.24 | 728 | Seahawks |

### Good (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Ezekiel Ansah | 79.31 | 75.62 | 83.34 | 146 | Lions |
| 20 | 2 | Carlos Dunlap | 79.28 | 77.91 | 76.02 | 839 | Bengals |
| 21 | 3 | Michael Bennett | 79.26 | 78.96 | 75.92 | 716 | Eagles |
| 22 | 4 | Melvin Ingram III | 78.97 | 74.36 | 77.87 | 915 | Chargers |
| 23 | 5 | Chandler Jones | 78.30 | 74.86 | 76.43 | 969 | Cardinals |
| 24 | 6 | Olivier Vernon | 78.15 | 83.85 | 74.03 | 665 | Giants |
| 25 | 7 | Calais Campbell | 78.05 | 62.91 | 83.98 | 816 | Jaguars |
| 26 | 8 | Robert Quinn | 77.13 | 73.56 | 76.81 | 635 | Dolphins |
| 27 | 9 | Jabaal Sheard | 77.07 | 79.93 | 70.99 | 814 | Colts |
| 28 | 10 | Takk McKinley | 76.94 | 70.30 | 77.85 | 619 | Falcons |
| 29 | 11 | Yannick Ngakoue | 76.23 | 68.89 | 76.96 | 766 | Jaguars |
| 30 | 12 | Dee Ford | 76.02 | 71.68 | 77.88 | 1022 | Chiefs |
| 31 | 13 | Shaquil Barrett | 75.51 | 72.06 | 75.21 | 276 | Broncos |
| 32 | 14 | Pernell McPhee | 74.93 | 65.04 | 81.32 | 203 | Commanders |
| 33 | 15 | Mario Addison | 74.40 | 61.84 | 79.03 | 666 | Panthers |
| 34 | 16 | Whitney Mercilus | 74.28 | 65.92 | 79.12 | 785 | Texans |

### Starter (58 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Terrell Suggs | 73.65 | 60.27 | 78.61 | 743 | Ravens |
| 36 | 2 | Matthew Judon | 73.51 | 61.18 | 77.98 | 674 | Ravens |
| 37 | 3 | Everson Griffen | 73.00 | 67.63 | 75.01 | 585 | Vikings |
| 38 | 4 | Jacob Martin | 72.76 | 63.14 | 75.01 | 225 | Seahawks |
| 39 | 5 | Marcus Davenport | 72.59 | 72.65 | 71.51 | 416 | Saints |
| 40 | 6 | Kemoko Turay | 71.40 | 61.05 | 76.21 | 383 | Colts |
| 41 | 7 | Chris Long | 71.40 | 60.42 | 74.55 | 612 | Eagles |
| 42 | 8 | Carl Lawson | 70.85 | 63.55 | 77.41 | 225 | Bengals |
| 43 | 9 | Kyler Fackrell | 70.71 | 56.70 | 76.30 | 626 | Packers |
| 44 | 10 | Tyus Bowser | 70.38 | 59.91 | 73.85 | 164 | Ravens |
| 45 | 11 | Derek Barnett | 70.34 | 70.65 | 72.48 | 234 | Eagles |
| 46 | 12 | Leonard Floyd | 70.33 | 63.64 | 73.32 | 794 | Bears |
| 47 | 13 | Jason Pierre-Paul | 70.03 | 64.18 | 70.60 | 933 | Buccaneers |
| 48 | 14 | Markus Golden | 69.69 | 59.73 | 78.51 | 393 | Cardinals |
| 49 | 15 | Preston Smith | 69.52 | 62.96 | 69.73 | 834 | Commanders |
| 50 | 16 | Aaron Lynch | 68.94 | 62.35 | 75.42 | 353 | Bears |
| 51 | 17 | Za'Darius Smith | 68.76 | 65.39 | 68.09 | 691 | Ravens |
| 52 | 18 | Julius Peppers | 68.42 | 53.50 | 74.20 | 506 | Panthers |
| 53 | 19 | Nick Perry | 68.40 | 60.00 | 74.73 | 302 | Packers |
| 54 | 20 | Sam Hubbard | 68.23 | 64.45 | 66.58 | 508 | Bengals |
| 55 | 21 | Jordan Jenkins | 68.16 | 59.18 | 70.40 | 660 | Jets |
| 56 | 22 | Derrick Morgan | 68.15 | 55.76 | 74.01 | 532 | Titans |
| 57 | 23 | Clay Matthews | 68.08 | 52.81 | 74.92 | 756 | Packers |
| 58 | 24 | Dante Fowler Jr. | 67.98 | 63.17 | 67.54 | 577 | Rams |
| 59 | 25 | Dion Jordan | 67.79 | 63.42 | 72.06 | 295 | Seahawks |
| 60 | 26 | Bruce Irvin | 67.72 | 49.06 | 75.99 | 471 | Falcons |
| 61 | 27 | Efe Obada | 67.54 | 59.16 | 75.21 | 189 | Panthers |
| 62 | 28 | Vinny Curry | 67.44 | 58.16 | 71.54 | 445 | Buccaneers |
| 63 | 29 | Lerentee McCray | 67.31 | 59.33 | 73.47 | 100 | Jaguars |
| 64 | 30 | Brian Orakpo | 67.20 | 52.74 | 74.24 | 572 | Titans |
| 65 | 31 | Alex Okafor | 67.01 | 59.69 | 69.81 | 658 | Saints |
| 66 | 32 | Vic Beasley Jr. | 66.87 | 58.48 | 68.30 | 702 | Falcons |
| 67 | 33 | Deatrich Wise Jr. | 66.39 | 61.75 | 65.32 | 431 | Patriots |
| 68 | 34 | Adrian Clayborn | 66.27 | 62.55 | 66.05 | 318 | Patriots |
| 69 | 35 | Shaq Lawson | 66.23 | 66.25 | 65.90 | 440 | Bills |
| 70 | 36 | Lorenzo Carter | 66.02 | 61.27 | 66.06 | 442 | Giants |
| 71 | 37 | John Franklin-Myers | 65.80 | 61.59 | 64.44 | 301 | Rams |
| 72 | 38 | John Simon | 65.71 | 60.25 | 71.01 | 185 | Patriots |
| 73 | 39 | Jeremiah Attaochu | 65.70 | 61.99 | 72.86 | 171 | Jets |
| 74 | 40 | Barkevious Mingo | 65.55 | 56.58 | 69.24 | 517 | Seahawks |
| 75 | 41 | Zach Moore | 65.47 | 53.19 | 72.62 | 246 | Cardinals |
| 76 | 42 | Trent Murphy | 65.41 | 54.54 | 70.06 | 441 | Bills |
| 77 | 43 | Samson Ebukam | 65.11 | 61.52 | 63.33 | 692 | Rams |
| 78 | 44 | Ronald Blair III | 64.88 | 59.45 | 67.46 | 534 | 49ers |
| 79 | 45 | Randy Gregory | 64.55 | 60.31 | 67.17 | 457 | Cowboys |
| 80 | 46 | Kerry Hyder Jr. | 64.48 | 58.87 | 71.87 | 153 | Lions |
| 81 | 47 | Shane Ray | 64.39 | 58.07 | 69.54 | 253 | Broncos |
| 82 | 48 | Arik Armstead | 64.29 | 61.95 | 64.82 | 608 | 49ers |
| 83 | 49 | Reggie Gilbert | 63.83 | 59.80 | 67.81 | 487 | Packers |
| 84 | 50 | Cassius Marsh | 63.73 | 60.44 | 62.08 | 550 | 49ers |
| 85 | 51 | Bud Dupree | 63.63 | 57.09 | 65.07 | 869 | Steelers |
| 86 | 52 | Connor Barwin | 63.55 | 51.12 | 68.51 | 289 | Giants |
| 87 | 53 | Brandon Copeland | 63.16 | 54.19 | 64.98 | 611 | Jets |
| 88 | 54 | Tim Williams | 62.97 | 63.27 | 67.59 | 120 | Ravens |
| 89 | 55 | Charles Harris | 62.67 | 60.79 | 63.00 | 347 | Dolphins |
| 90 | 56 | Ryan Anderson | 62.62 | 63.61 | 60.52 | 164 | Commanders |
| 91 | 57 | Chris Smith | 62.28 | 58.28 | 62.87 | 336 | Browns |
| 92 | 58 | Matt Longacre | 62.06 | 59.72 | 63.72 | 281 | Rams |

### Rotation/backup (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 93 | 1 | Anthony Chickillo | 61.38 | 56.94 | 60.80 | 295 | Steelers |
| 94 | 2 | Solomon Thomas | 61.02 | 63.68 | 55.87 | 644 | 49ers |
| 95 | 3 | Devon Kennard | 60.97 | 49.16 | 65.51 | 864 | Lions |
| 96 | 4 | Anthony Zettel | 60.61 | 57.59 | 59.39 | 158 | Browns |
| 97 | 5 | Tanoh Kpassagnon | 60.45 | 61.26 | 62.64 | 115 | Chiefs |
| 98 | 6 | Brooks Reed | 60.20 | 51.96 | 61.53 | 458 | Falcons |
| 99 | 7 | Carl Nassib | 60.20 | 59.16 | 57.66 | 598 | Buccaneers |
| 100 | 8 | Benson Mayowa | 60.17 | 58.77 | 58.51 | 550 | Cardinals |
| 101 | 9 | Jonathan Woodard | 60.14 | 57.98 | 66.93 | 128 | Dolphins |
| 102 | 10 | Eli Harold | 60.00 | 57.85 | 59.87 | 184 | Lions |
| 103 | 11 | Emmanuel Ogbah | 59.42 | 59.61 | 58.05 | 806 | Browns |
| 104 | 12 | Romeo Okwara | 59.14 | 59.53 | 58.36 | 716 | Lions |
| 105 | 13 | Jordan Willis | 59.04 | 58.15 | 55.47 | 537 | Bengals |
| 106 | 14 | Arden Key | 59.02 | 57.31 | 56.00 | 644 | Raiders |
| 107 | 15 | Trey Hendrickson | 58.98 | 61.54 | 61.05 | 136 | Saints |
| 108 | 16 | Michael Johnson | 58.96 | 51.94 | 60.30 | 467 | Bengals |
| 109 | 17 | Taco Charlton | 58.84 | 60.11 | 57.08 | 402 | Cowboys |
| 110 | 18 | Eddie Yarbrough | 58.48 | 57.61 | 55.55 | 307 | Bills |
| 111 | 19 | Stephen Weatherly | 58.43 | 58.11 | 59.57 | 524 | Vikings |
| 112 | 20 | Dawuane Smoot | 58.41 | 61.93 | 57.09 | 171 | Jaguars |
| 113 | 21 | Andre Branch | 58.35 | 53.67 | 58.97 | 483 | Dolphins |
| 114 | 22 | Steven Means | 58.23 | 58.68 | 63.45 | 162 | Falcons |
| 115 | 23 | Wes Horton | 58.07 | 52.06 | 58.95 | 471 | Panthers |
| 116 | 24 | Kareem Martin | 57.63 | 55.98 | 56.44 | 610 | Giants |
| 117 | 25 | Tyquan Lewis | 57.59 | 58.35 | 61.25 | 337 | Colts |
| 118 | 26 | Kerry Wynn | 57.45 | 56.95 | 55.39 | 393 | Giants |
| 119 | 27 | Isaiah Irving | 57.37 | 59.21 | 61.23 | 117 | Bears |
| 120 | 28 | Dorance Armstrong | 57.29 | 57.21 | 54.21 | 273 | Cowboys |
| 121 | 29 | Duke Ejiofor | 55.79 | 57.09 | 54.93 | 158 | Texans |
| 122 | 30 | Derrick Shelby | 55.70 | 57.03 | 56.83 | 135 | Falcons |
| 123 | 31 | Shilique Calhoun | 54.92 | 58.07 | 57.09 | 138 | Raiders |
| 124 | 32 | Branden Jackson | 54.38 | 57.72 | 55.59 | 258 | Seahawks |
| 125 | 33 | Rasheem Green | 54.32 | 57.19 | 54.49 | 201 | Seahawks |
| 126 | 34 | Al-Quadin Muhammad | 52.89 | 57.20 | 51.58 | 415 | Colts |
| 127 | 35 | Bryan Cox Jr. | 52.55 | 57.23 | 51.64 | 200 | Panthers |
| 128 | 36 | Cameron Malveaux | 52.20 | 57.97 | 53.43 | 174 | Cardinals |
| 129 | 37 | Keionta Davis | 51.18 | 55.85 | 53.41 | 182 | Patriots |

## G — Guard

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Marshal Yanda | 90.53 | 85.00 | 90.05 | 1162 | Ravens |
| 2 | 2 | Shaq Mason | 90.10 | 84.76 | 89.49 | 954 | Patriots |
| 3 | 3 | Quenton Nelson | 87.10 | 79.70 | 87.86 | 1136 | Colts |
| 4 | 4 | Zack Martin | 86.95 | 81.50 | 86.42 | 1754 | Cowboys |
| 5 | 5 | Joel Bitonio | 86.92 | 80.30 | 87.17 | 1091 | Browns |
| 6 | 6 | Brandon Brooks | 85.27 | 78.10 | 85.88 | 1087 | Eagles |
| 7 | 7 | Rodger Saffold | 84.78 | 76.90 | 85.86 | 1069 | Rams |
| 8 | 8 | Kevin Zeitler | 84.48 | 77.20 | 85.16 | 1091 | Browns |
| 9 | 9 | Mark Glowinski | 83.66 | 73.83 | 86.05 | 601 | Colts |
| 10 | 10 | Joe Thuney | 82.95 | 75.70 | 83.62 | 1120 | Patriots |
| 11 | 11 | Ali Marpet | 82.80 | 75.90 | 83.23 | 1117 | Buccaneers |
| 12 | 12 | David DeCastro | 81.45 | 74.68 | 81.79 | 958 | Steelers |
| 13 | 13 | Gabe Jackson | 81.07 | 73.14 | 82.19 | 855 | Raiders |
| 14 | 14 | T.J. Lang | 81.06 | 71.61 | 83.20 | 282 | Lions |
| 15 | 15 | Ben Garland | 80.16 | 67.12 | 84.68 | 371 | Falcons |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Austin Blythe | 79.81 | 76.40 | 77.91 | 1101 | Rams |
| 17 | 2 | Brandon Scherff | 79.56 | 70.38 | 81.52 | 506 | Commanders |
| 18 | 3 | Ramon Foster | 79.31 | 72.30 | 79.82 | 1116 | Steelers |
| 19 | 4 | Mike Person | 79.10 | 71.40 | 80.07 | 1000 | 49ers |
| 20 | 5 | Andrew Norwell | 78.54 | 69.61 | 80.32 | 726 | Jaguars |
| 21 | 6 | Quinton Spain | 77.44 | 67.39 | 79.97 | 856 | Titans |
| 22 | 7 | Larry Warford | 77.04 | 67.99 | 78.91 | 980 | Saints |
| 23 | 8 | Laken Tomlinson | 76.92 | 67.50 | 79.03 | 1028 | 49ers |
| 24 | 9 | Ted Karras | 76.90 | 65.87 | 80.09 | 171 | Patriots |
| 25 | 10 | Isaac Seumalo | 76.79 | 66.04 | 79.79 | 548 | Eagles |
| 26 | 11 | Will Hernandez | 76.76 | 67.90 | 78.50 | 1027 | Giants |
| 27 | 12 | Mike Iupati | 76.72 | 64.66 | 80.59 | 477 | Cardinals |
| 28 | 13 | Frank Ragnow | 76.34 | 66.50 | 78.73 | 1076 | Lions |
| 29 | 14 | Trai Turner | 76.23 | 67.33 | 78.00 | 762 | Panthers |
| 30 | 15 | Wes Schweitzer | 76.08 | 66.60 | 78.24 | 901 | Falcons |
| 31 | 16 | John Miller | 75.85 | 67.49 | 77.25 | 885 | Bills |
| 32 | 17 | Brandon Fusco | 75.64 | 64.32 | 79.02 | 436 | Falcons |
| 33 | 18 | Brian Winters | 75.59 | 65.90 | 77.89 | 1001 | Jets |
| 34 | 19 | Kyle Long | 75.51 | 65.49 | 78.03 | 511 | Bears |
| 35 | 20 | Matt Slauson | 75.06 | 65.24 | 77.44 | 376 | Colts |
| 36 | 21 | Andrew Wylie | 75.03 | 65.01 | 77.55 | 687 | Chiefs |
| 37 | 22 | Laurent Duvernay-Tardif | 74.74 | 62.47 | 78.76 | 331 | Chiefs |
| 38 | 23 | Lane Taylor | 74.32 | 65.68 | 75.91 | 882 | Packers |

### Starter (49 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Ron Leary | 73.94 | 62.55 | 77.37 | 383 | Broncos |
| 40 | 2 | Tom Compton | 73.75 | 62.21 | 77.28 | 837 | Vikings |
| 41 | 3 | Justin McCray | 73.49 | 63.42 | 76.03 | 481 | Packers |
| 42 | 4 | Billy Turner | 73.29 | 64.02 | 75.31 | 824 | Broncos |
| 43 | 5 | Connor McGovern | 73.03 | 60.00 | 77.55 | 1056 | Broncos |
| 44 | 6 | Oday Aboushi | 72.97 | 62.38 | 75.86 | 407 | Cardinals |
| 45 | 7 | Alex Redmond | 72.96 | 60.78 | 76.91 | 928 | Bengals |
| 46 | 8 | Michael Schofield III | 72.89 | 63.89 | 74.72 | 978 | Chargers |
| 47 | 9 | A.J. Cann | 72.43 | 62.66 | 74.78 | 934 | Jaguars |
| 48 | 10 | B.J. Finney | 72.32 | 64.71 | 73.22 | 165 | Steelers |
| 49 | 11 | Mike Remmers | 72.14 | 61.10 | 75.33 | 1048 | Vikings |
| 50 | 12 | Greg Van Roten | 71.42 | 61.20 | 74.06 | 1059 | Panthers |
| 51 | 13 | D.J. Fluker | 71.40 | 56.94 | 76.88 | 607 | Seahawks |
| 52 | 14 | Clint Boling | 71.17 | 62.59 | 72.72 | 969 | Bengals |
| 53 | 15 | Josh Kline | 71.10 | 60.90 | 73.73 | 975 | Titans |
| 54 | 16 | Danny Isidora | 71.03 | 58.60 | 75.15 | 214 | Vikings |
| 55 | 17 | Dakota Dozier | 70.89 | 58.84 | 74.75 | 106 | Jets |
| 56 | 18 | Kelechi Osemele | 70.61 | 58.62 | 74.43 | 735 | Raiders |
| 57 | 19 | Connor Williams | 70.44 | 60.00 | 73.23 | 688 | Cowboys |
| 58 | 20 | James Carpenter | 70.37 | 58.68 | 74.00 | 624 | Jets |
| 59 | 21 | Jesse Davis | 70.21 | 59.23 | 73.36 | 921 | Dolphins |
| 60 | 22 | Senio Kelemete | 69.68 | 58.95 | 72.66 | 895 | Texans |
| 61 | 23 | Stefen Wisniewski | 69.39 | 57.58 | 73.09 | 643 | Eagles |
| 62 | 24 | Zane Beadles | 69.36 | 58.71 | 72.30 | 279 | Falcons |
| 63 | 25 | Jeff Allen | 68.93 | 57.60 | 72.32 | 224 | Chiefs |
| 64 | 26 | Corey Levin | 68.81 | 57.72 | 72.03 | 140 | Titans |
| 65 | 27 | Wyatt Teller | 68.79 | 60.14 | 70.39 | 476 | Bills |
| 66 | 28 | Jeremiah Sirles | 68.60 | 57.68 | 71.72 | 140 | Bills |
| 67 | 29 | Bryan Witzmann | 68.58 | 57.43 | 71.84 | 533 | Bears |
| 68 | 30 | Jonathan Cooper | 68.31 | 57.65 | 71.25 | 201 | Commanders |
| 69 | 31 | Kenny Wiggins | 68.23 | 57.39 | 71.29 | 798 | Lions |
| 70 | 32 | Patrick Omameh | 68.12 | 56.76 | 71.52 | 679 | Jaguars |
| 71 | 33 | Vladimir Ducasse | 67.92 | 53.64 | 73.28 | 564 | Bills |
| 72 | 34 | Shawn Lauvao | 67.92 | 56.40 | 71.43 | 284 | Commanders |
| 73 | 35 | Zach Fulton | 67.51 | 55.36 | 71.44 | 817 | Texans |
| 74 | 36 | Max Garcia | 67.40 | 55.61 | 71.09 | 242 | Broncos |
| 75 | 37 | Lucas Patrick | 65.84 | 54.73 | 69.08 | 279 | Packers |
| 76 | 38 | Justin Pugh | 65.59 | 54.80 | 68.61 | 343 | Cardinals |
| 77 | 39 | Eric Kush | 65.25 | 58.70 | 65.45 | 344 | Bears |
| 78 | 40 | Alex Lewis | 65.25 | 52.29 | 69.73 | 707 | Ravens |
| 79 | 41 | Jon Feliciano | 65.16 | 54.96 | 67.80 | 227 | Raiders |
| 80 | 42 | Colby Gossett | 64.36 | 52.51 | 68.09 | 282 | Cardinals |
| 81 | 43 | Byron Bell | 64.29 | 51.95 | 68.35 | 528 | Packers |
| 82 | 44 | J.R. Sweezy | 64.08 | 49.92 | 69.35 | 948 | Seahawks |
| 83 | 45 | Dan Feeney | 64.03 | 49.20 | 69.75 | 995 | Chargers |
| 84 | 46 | Ethan Pocic | 63.71 | 52.27 | 67.17 | 296 | Seahawks |
| 85 | 47 | Cameron Erving | 63.08 | 47.06 | 69.59 | 830 | Chiefs |
| 86 | 48 | Xavier Su'a-Filo | 62.23 | 48.74 | 67.05 | 494 | Cowboys |
| 87 | 49 | Andrus Peat | 62.04 | 45.97 | 68.58 | 739 | Saints |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 88 | 1 | Josh LeRibeus | 61.81 | 52.03 | 64.16 | 179 | Saints |
| 89 | 2 | Alex Cappa | 61.68 | 52.04 | 63.94 | 106 | Buccaneers |
| 90 | 3 | Caleb Benenoch | 61.19 | 45.93 | 67.20 | 844 | Buccaneers |
| 91 | 4 | Ted Larsen | 60.86 | 45.40 | 67.00 | 752 | Dolphins |
| 92 | 5 | Cameron Tom | 59.58 | 53.24 | 59.64 | 178 | Saints |

## HB — Running Back

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Saquon Barkley | 85.96 | 85.20 | 82.30 | 470 | Giants |
| 2 | 2 | Nick Chubb | 83.85 | 76.14 | 84.83 | 142 | Browns |
| 3 | 3 | Alvin Kamara | 83.69 | 82.04 | 80.62 | 334 | Saints |
| 4 | 4 | Austin Ekeler | 81.63 | 77.11 | 80.48 | 190 | Chargers |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Christian McCaffrey | 79.18 | 82.80 | 72.60 | 493 | Panthers |
| 6 | 2 | Kerryon Johnson | 78.98 | 73.47 | 78.48 | 169 | Lions |
| 7 | 3 | Derrick Henry | 77.34 | 73.92 | 75.46 | 115 | Titans |
| 8 | 4 | Aaron Jones | 76.56 | 75.17 | 73.32 | 204 | Packers |
| 9 | 5 | Melvin Gordon III | 75.87 | 83.35 | 66.72 | 241 | Chargers |
| 10 | 6 | Chris Carson | 75.79 | 73.88 | 72.90 | 136 | Seahawks |
| 11 | 7 | Todd Gurley II | 75.21 | 77.80 | 69.31 | 428 | Rams |
| 12 | 8 | Duke Johnson Jr. | 75.21 | 71.29 | 73.66 | 300 | Browns |
| 13 | 9 | Kenyan Drake | 74.67 | 65.23 | 76.80 | 300 | Dolphins |
| 14 | 10 | Dalvin Cook | 74.54 | 70.10 | 73.34 | 244 | Vikings |
| 15 | 11 | Ty Montgomery | 74.29 | 66.70 | 75.18 | 152 | Ravens |

### Starter (47 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Ezekiel Elliott | 73.98 | 72.20 | 71.00 | 412 | Cowboys |
| 17 | 2 | Jalen Richard | 73.96 | 65.55 | 75.40 | 269 | Raiders |
| 18 | 3 | Phillip Lindsay | 73.68 | 75.63 | 68.22 | 185 | Broncos |
| 19 | 4 | Adrian Peterson | 73.50 | 72.75 | 69.83 | 158 | Commanders |
| 20 | 5 | Mark Ingram II | 72.99 | 72.19 | 69.36 | 133 | Saints |
| 21 | 6 | Spencer Ware | 72.95 | 67.59 | 72.35 | 142 | Chiefs |
| 22 | 7 | James Conner | 72.16 | 72.10 | 68.04 | 380 | Steelers |
| 23 | 8 | Tarik Cohen | 72.03 | 71.86 | 67.98 | 307 | Bears |
| 24 | 9 | Joe Mixon | 72.01 | 75.11 | 65.78 | 274 | Bengals |
| 25 | 10 | Damien Williams | 71.39 | 70.65 | 67.72 | 117 | Chiefs |
| 26 | 11 | Dion Lewis | 71.28 | 65.02 | 71.29 | 315 | Titans |
| 27 | 12 | Matt Breida | 70.84 | 70.25 | 67.07 | 160 | 49ers |
| 28 | 13 | Chris Ivory | 70.53 | 67.82 | 68.17 | 117 | Bills |
| 29 | 14 | Frank Gore | 70.41 | 75.19 | 63.06 | 128 | Dolphins |
| 30 | 15 | Lamar Miller | 69.83 | 72.49 | 63.89 | 276 | Texans |
| 31 | 16 | Theo Riddick | 69.80 | 71.16 | 64.73 | 272 | Lions |
| 32 | 17 | Jordan Wilkins | 69.50 | 61.74 | 70.51 | 100 | Colts |
| 33 | 18 | Chris Thompson | 69.43 | 59.12 | 72.13 | 206 | Commanders |
| 34 | 19 | Marlon Mack | 69.33 | 65.77 | 67.53 | 171 | Colts |
| 35 | 20 | Latavius Murray | 69.22 | 71.97 | 63.22 | 221 | Vikings |
| 36 | 21 | Jordan Howard | 68.99 | 66.86 | 66.25 | 211 | Bears |
| 37 | 22 | Royce Freeman | 68.48 | 61.68 | 68.85 | 119 | Broncos |
| 38 | 23 | Isaiah Crowell | 68.40 | 66.42 | 65.55 | 145 | Jets |
| 39 | 24 | Giovani Bernard | 68.31 | 66.72 | 65.20 | 195 | Bengals |
| 40 | 25 | Jaylen Samuels | 68.16 | 65.15 | 66.00 | 130 | Steelers |
| 41 | 26 | Tevin Coleman | 68.00 | 65.03 | 65.81 | 296 | Falcons |
| 42 | 27 | Nyheim Hines | 67.94 | 71.07 | 61.68 | 346 | Colts |
| 43 | 28 | Ito Smith | 67.75 | 65.78 | 64.89 | 164 | Falcons |
| 44 | 29 | James White | 67.69 | 71.00 | 61.31 | 411 | Patriots |
| 45 | 30 | David Johnson | 67.30 | 64.03 | 65.31 | 360 | Cardinals |
| 46 | 31 | Doug Martin | 67.10 | 65.00 | 64.34 | 135 | Raiders |
| 47 | 32 | Mike Davis | 67.02 | 70.56 | 60.50 | 186 | Seahawks |
| 48 | 33 | LeSean McCoy | 66.83 | 60.78 | 66.69 | 225 | Bills |
| 49 | 34 | Alex Collins | 66.57 | 59.89 | 66.85 | 123 | Ravens |
| 50 | 35 | Jacquizz Rodgers | 66.44 | 64.44 | 63.61 | 234 | Buccaneers |
| 51 | 36 | Devontae Booker | 66.05 | 63.55 | 63.55 | 226 | Broncos |
| 52 | 37 | Peyton Barber | 65.88 | 65.73 | 61.81 | 270 | Buccaneers |
| 53 | 38 | Jamaal Williams | 65.26 | 66.38 | 60.35 | 275 | Packers |
| 54 | 39 | Corey Clement | 64.46 | 58.74 | 64.11 | 125 | Eagles |
| 55 | 40 | Carlos Hyde | 64.22 | 53.99 | 66.88 | 149 | Jaguars |
| 56 | 41 | T.J. Yeldon | 63.71 | 56.39 | 64.42 | 310 | Jaguars |
| 57 | 42 | Trenton Cannon | 62.98 | 57.47 | 62.48 | 101 | Jets |
| 58 | 43 | Wendell Smallwood | 62.84 | 57.26 | 62.39 | 197 | Eagles |
| 59 | 44 | Kapri Bibbs | 62.84 | 59.19 | 61.10 | 128 | Packers |
| 60 | 45 | Jeff Wilson Jr. | 62.84 | 57.20 | 62.44 | 106 | 49ers |
| 61 | 46 | Javorius Allen | 62.61 | 61.94 | 58.89 | 183 | Ravens |
| 62 | 47 | Elijah McGuire | 62.20 | 56.75 | 61.67 | 154 | Jets |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 63 | 1 | Alfred Blue | 61.70 | 62.29 | 57.14 | 208 | Texans |
| 64 | 2 | Chase Edmonds | 61.46 | 60.27 | 58.08 | 114 | Cardinals |

## LB — Linebacker

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Bobby Wagner | 90.01 | 91.70 | 85.24 | 925 | Seahawks |
| 2 | 2 | Luke Kuechly | 87.38 | 90.50 | 82.38 | 927 | Panthers |
| 3 | 3 | Zach Brown | 84.61 | 86.95 | 79.81 | 703 | Commanders |
| 4 | 4 | Leighton Vander Esch | 82.84 | 83.87 | 77.99 | 785 | Cowboys |
| 5 | 5 | Lorenzo Alexander | 82.32 | 83.30 | 77.50 | 629 | Bills |
| 6 | 6 | Benardrick McKinney | 80.41 | 80.90 | 75.91 | 919 | Texans |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Jaylon Smith | 79.80 | 84.00 | 72.84 | 978 | Cowboys |
| 8 | 2 | Jayon Brown | 78.35 | 81.20 | 72.29 | 852 | Titans |
| 9 | 3 | Lavonte David | 76.17 | 76.10 | 74.04 | 922 | Buccaneers |
| 10 | 4 | Avery Williamson | 75.10 | 72.80 | 72.46 | 1114 | Jets |
| 11 | 5 | Blake Martinez | 75.07 | 73.90 | 71.68 | 1050 | Packers |
| 12 | 6 | Demario Davis | 74.21 | 74.60 | 69.78 | 877 | Saints |

### Starter (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Jordan Hicks | 73.72 | 74.44 | 73.97 | 705 | Eagles |
| 14 | 2 | Wesley Woodyard | 73.68 | 71.71 | 71.86 | 714 | Titans |
| 15 | 3 | Joe Schobert | 73.51 | 76.80 | 68.71 | 897 | Browns |
| 16 | 4 | Brennan Scarlett | 72.82 | 71.96 | 73.39 | 108 | Texans |
| 17 | 5 | Matt Milano | 72.59 | 75.30 | 69.35 | 741 | Bills |
| 18 | 6 | Jerome Baker | 72.27 | 69.73 | 69.80 | 678 | Dolphins |
| 19 | 7 | C.J. Mosley | 72.09 | 70.10 | 70.18 | 874 | Ravens |
| 20 | 8 | Danny Trevathan | 72.06 | 71.90 | 70.70 | 987 | Bears |
| 21 | 9 | Thomas Davis Sr. | 72.02 | 72.98 | 69.30 | 649 | Panthers |
| 22 | 10 | Josh Bynes | 71.89 | 73.93 | 71.15 | 726 | Cardinals |
| 23 | 11 | Todd Davis | 71.68 | 70.60 | 68.86 | 842 | Broncos |
| 24 | 12 | Darron Lee | 71.54 | 72.31 | 69.87 | 808 | Jets |
| 25 | 13 | Anthony Barr | 71.07 | 70.86 | 68.61 | 810 | Vikings |
| 26 | 14 | Cory Littleton | 70.36 | 66.00 | 70.35 | 964 | Rams |
| 27 | 15 | Deion Jones | 70.12 | 71.11 | 70.49 | 384 | Falcons |
| 28 | 16 | Myles Jack | 70.12 | 68.30 | 67.59 | 1024 | Jaguars |
| 29 | 17 | Shaq Thompson | 69.45 | 67.94 | 68.05 | 599 | Panthers |
| 30 | 18 | A.J. Klein | 69.34 | 68.23 | 67.79 | 670 | Saints |
| 31 | 19 | Nigel Bradham | 68.96 | 66.60 | 66.89 | 919 | Eagles |
| 32 | 20 | Dont'a Hightower | 68.47 | 64.91 | 67.20 | 774 | Patriots |
| 33 | 21 | L.J. Fort | 68.40 | 67.56 | 70.73 | 305 | Steelers |
| 34 | 22 | Christian Jones | 68.40 | 64.25 | 67.31 | 643 | Lions |
| 35 | 23 | Eric Kendricks | 68.30 | 64.50 | 67.92 | 877 | Vikings |
| 36 | 24 | Manti Te'o | 68.20 | 68.46 | 72.29 | 141 | Saints |
| 37 | 25 | Fred Warner | 67.84 | 64.10 | 66.17 | 1060 | 49ers |
| 38 | 26 | Foyesade Oluokun | 67.81 | 64.56 | 65.81 | 525 | Falcons |
| 39 | 27 | Zach Cunningham | 67.77 | 64.98 | 66.76 | 753 | Texans |
| 40 | 28 | Jason Cabinda | 67.76 | 65.99 | 71.03 | 164 | Raiders |
| 41 | 29 | Kyle Van Noy | 67.69 | 65.30 | 65.12 | 946 | Patriots |
| 42 | 30 | Roquan Smith | 67.59 | 64.20 | 65.68 | 880 | Bears |
| 43 | 31 | Ben Gedeon | 67.49 | 62.16 | 67.52 | 311 | Vikings |
| 44 | 32 | Telvin Smith Sr. | 66.96 | 63.70 | 64.96 | 1020 | Jaguars |
| 45 | 33 | Rashaan Evans | 66.94 | 64.27 | 66.64 | 494 | Titans |
| 46 | 34 | K.J. Wright | 66.84 | 67.27 | 68.44 | 223 | Seahawks |
| 47 | 35 | Alex Anzalone | 66.69 | 68.08 | 66.28 | 486 | Saints |
| 48 | 36 | Leon Jacobs | 66.43 | 63.71 | 68.25 | 146 | Jaguars |
| 49 | 37 | Raekwon McMillan | 65.99 | 60.20 | 65.68 | 831 | Dolphins |
| 50 | 38 | Josey Jewell | 65.93 | 61.27 | 64.87 | 460 | Broncos |
| 51 | 39 | Elandon Roberts | 65.90 | 62.31 | 64.33 | 429 | Patriots |
| 52 | 40 | Vince Williams | 65.60 | 64.76 | 64.07 | 744 | Steelers |
| 53 | 41 | Jon Bostic | 65.50 | 59.59 | 65.90 | 560 | Steelers |
| 54 | 42 | Denzel Perryman | 65.25 | 66.93 | 67.27 | 386 | Chargers |
| 55 | 43 | B.J. Goodson | 65.06 | 64.90 | 66.83 | 513 | Giants |
| 56 | 44 | Malcolm Smith | 65.05 | 62.04 | 64.97 | 336 | 49ers |
| 57 | 45 | Marquel Lee | 64.99 | 60.74 | 65.22 | 448 | Raiders |
| 58 | 46 | Haason Reddick | 64.28 | 60.40 | 62.70 | 846 | Cardinals |
| 59 | 47 | Eric Wilson | 64.25 | 61.73 | 63.07 | 336 | Vikings |
| 60 | 48 | Brandon Marshall | 64.09 | 62.99 | 64.31 | 468 | Broncos |
| 61 | 49 | De'Vondre Campbell | 63.61 | 56.70 | 64.46 | 902 | Falcons |
| 62 | 50 | Kamu Grugier-Hill | 63.10 | 61.90 | 65.98 | 330 | Eagles |
| 63 | 51 | Patrick Onwuasor | 62.77 | 58.91 | 62.84 | 435 | Ravens |
| 64 | 52 | Jamie Collins Sr. | 62.73 | 61.20 | 62.91 | 1067 | Browns |
| 65 | 53 | Mychal Kendricks | 62.23 | 64.37 | 63.08 | 183 | Seahawks |
| 66 | 54 | Nick Kwiatkoski | 62.00 | 57.50 | 65.61 | 112 | Bears |

### Rotation/backup (52 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 67 | 1 | Mason Foster | 61.97 | 57.30 | 64.35 | 1014 | Commanders |
| 68 | 2 | Ramik Wilson | 61.86 | 60.86 | 66.59 | 158 | Rams |
| 69 | 3 | Ja'Whaun Bentley | 61.70 | 68.49 | 73.01 | 138 | Patriots |
| 70 | 4 | Kenny Young | 61.63 | 56.70 | 60.75 | 371 | Ravens |
| 71 | 5 | Tahir Whitehead | 61.40 | 54.80 | 61.63 | 1025 | Raiders |
| 72 | 6 | Nick Vigil | 61.40 | 60.00 | 63.36 | 672 | Bengals |
| 73 | 7 | Kyle Emanuel | 61.30 | 56.97 | 61.05 | 216 | Chargers |
| 74 | 8 | Tremaine Edmunds | 60.83 | 57.00 | 60.25 | 927 | Bills |
| 75 | 9 | Elijah Lee | 60.79 | 58.63 | 65.36 | 476 | 49ers |
| 76 | 10 | Anthony Walker Jr. | 60.77 | 60.00 | 62.85 | 695 | Colts |
| 77 | 11 | Antonio Morrison | 60.66 | 57.45 | 59.36 | 302 | Packers |
| 78 | 12 | Sean Lee | 60.53 | 59.79 | 63.10 | 221 | Cowboys |
| 79 | 13 | Jalen Reeves-Maybin | 60.04 | 59.32 | 62.47 | 111 | Lions |
| 80 | 14 | Matthew Adams | 59.95 | 56.36 | 62.34 | 215 | Colts |
| 81 | 15 | Reggie Ragland | 59.63 | 54.61 | 59.74 | 582 | Chiefs |
| 82 | 16 | Gerald Hodges | 59.61 | 56.82 | 63.24 | 356 | Cardinals |
| 83 | 17 | Jatavis Brown | 59.55 | 53.87 | 60.52 | 637 | Chargers |
| 84 | 18 | David Mayo | 59.52 | 59.45 | 62.27 | 145 | Panthers |
| 85 | 19 | Zaire Franklin | 59.49 | 55.92 | 62.91 | 176 | Colts |
| 86 | 20 | Preston Brown | 59.28 | 55.67 | 62.21 | 375 | Bengals |
| 87 | 21 | Jordan Evans | 59.00 | 52.67 | 62.30 | 510 | Bengals |
| 88 | 22 | Jarrad Davis | 58.84 | 51.00 | 60.68 | 976 | Lions |
| 89 | 23 | Kiko Alonso | 58.39 | 47.70 | 61.86 | 1004 | Dolphins |
| 90 | 24 | Kwon Alexander | 57.65 | 58.26 | 59.52 | 366 | Buccaneers |
| 91 | 25 | Nicholas Morrow | 57.24 | 53.52 | 57.25 | 416 | Raiders |
| 92 | 26 | Emmanuel Lamur | 56.81 | 57.01 | 59.70 | 145 | Jets |
| 93 | 27 | Nathan Gerry | 56.55 | 56.65 | 61.69 | 137 | Eagles |
| 94 | 28 | Devante Bond | 56.21 | 52.91 | 59.24 | 248 | Buccaneers |
| 95 | 29 | Josh Harvey-Clemons | 56.13 | 55.80 | 57.39 | 196 | Commanders |
| 96 | 30 | Dylan Cole | 56.06 | 55.72 | 58.88 | 120 | Texans |
| 97 | 31 | Alec Ogletree | 56.04 | 49.30 | 57.93 | 885 | Giants |
| 98 | 32 | Ray-Ray Armstrong | 56.01 | 54.74 | 60.71 | 214 | Browns |
| 99 | 33 | Vincent Rey | 55.78 | 51.64 | 58.13 | 178 | Bengals |
| 100 | 34 | Damien Wilson | 55.77 | 49.53 | 56.40 | 287 | Cowboys |
| 101 | 35 | Oren Burks | 55.71 | 53.68 | 57.06 | 126 | Packers |
| 102 | 36 | Hardy Nickerson | 55.54 | 48.34 | 57.74 | 538 | Bengals |
| 103 | 37 | Deone Bucannon | 55.49 | 49.88 | 58.49 | 389 | Cardinals |
| 104 | 38 | Mark Barron | 55.27 | 47.17 | 58.90 | 569 | Rams |
| 105 | 39 | Kyzir White | 55.12 | 62.33 | 66.14 | 142 | Chargers |
| 106 | 40 | Duke Riley | 54.61 | 47.30 | 57.79 | 408 | Falcons |
| 107 | 41 | Mark Nzeocha | 54.06 | 56.81 | 57.22 | 175 | 49ers |
| 108 | 42 | Shaun Dion Hamilton | 53.98 | 57.36 | 60.97 | 129 | Commanders |
| 109 | 43 | Vontaze Burfict | 53.71 | 50.86 | 59.04 | 298 | Bengals |
| 110 | 44 | Neville Hewitt | 53.05 | 49.94 | 58.35 | 268 | Jets |
| 111 | 45 | Christian Kirksey | 52.97 | 48.37 | 56.56 | 474 | Browns |
| 112 | 46 | Tae Davis | 52.66 | 47.43 | 57.18 | 344 | Giants |
| 113 | 47 | Anthony Hitchens | 52.40 | 40.00 | 58.26 | 944 | Chiefs |
| 114 | 48 | Adarius Taylor | 51.28 | 43.37 | 57.48 | 635 | Buccaneers |
| 115 | 49 | Riley Bullough | 51.00 | 51.56 | 58.69 | 126 | Buccaneers |
| 116 | 50 | Terrance Smith | 50.69 | 52.15 | 57.02 | 173 | Chiefs |
| 117 | 51 | Tanner Vallejo | 50.30 | 51.34 | 57.03 | 145 | Browns |
| 118 | 52 | Austin Calitro | 50.11 | 41.94 | 56.59 | 282 | Seahawks |

## QB — Quarterback

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Drew Brees | 87.76 | 89.98 | 83.31 | 542 | Saints |
| 2 | 2 | Tom Brady | 85.14 | 90.57 | 76.39 | 625 | Patriots |
| 3 | 3 | Philip Rivers | 83.77 | 85.04 | 79.51 | 579 | Chargers |
| 4 | 4 | Matt Ryan | 83.03 | 83.22 | 78.33 | 706 | Falcons |
| 5 | 5 | Russell Wilson | 81.80 | 80.91 | 79.91 | 546 | Seahawks |
| 6 | 6 | Andrew Luck | 81.33 | 88.94 | 74.81 | 725 | Colts |
| 7 | 7 | Patrick Mahomes | 80.89 | 92.32 | 84.20 | 683 | Chiefs |
| 8 | 8 | Aaron Rodgers | 80.23 | 85.62 | 73.76 | 724 | Packers |
| 9 | 9 | Jared Goff | 80.20 | 80.24 | 77.10 | 653 | Rams |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Ben Roethlisberger | 79.21 | 77.78 | 75.84 | 761 | Steelers |
| 11 | 2 | Deshaun Watson | 78.47 | 78.89 | 77.85 | 673 | Texans |
| 12 | 3 | Kirk Cousins | 77.89 | 76.83 | 74.23 | 705 | Vikings |
| 13 | 4 | Carson Wentz | 76.52 | 77.53 | 75.05 | 472 | Eagles |
| 14 | 5 | Matthew Stafford | 76.07 | 76.82 | 70.91 | 649 | Lions |
| 15 | 6 | Derek Carr | 74.50 | 72.97 | 71.73 | 647 | Raiders |
| 16 | 7 | Dak Prescott | 74.41 | 72.85 | 71.70 | 653 | Cowboys |

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Ryan Fitzpatrick | 73.24 | 72.73 | 79.09 | 302 | Buccaneers |
| 18 | 2 | Andy Dalton | 71.76 | 74.39 | 69.59 | 428 | Bengals |
| 19 | 3 | Marcus Mariota | 71.03 | 69.63 | 71.97 | 431 | Titans |
| 20 | 4 | Baker Mayfield | 70.99 | 79.38 | 75.76 | 575 | Browns |
| 21 | 5 | Jameis Winston | 70.61 | 69.73 | 70.84 | 471 | Buccaneers |
| 22 | 6 | Cam Newton | 70.50 | 69.16 | 68.74 | 558 | Panthers |
| 23 | 7 | Eli Manning | 69.74 | 64.64 | 69.36 | 657 | Giants |
| 24 | 8 | Alex Smith | 69.70 | 72.64 | 67.38 | 396 | Commanders |
| 25 | 9 | Case Keenum | 68.16 | 69.03 | 63.67 | 680 | Broncos |
| 26 | 10 | Mitch Trubisky | 66.24 | 59.92 | 71.33 | 539 | Bears |
| 27 | 11 | Joe Flacco | 65.32 | 68.37 | 62.51 | 427 | Ravens |
| 28 | 12 | Blake Bortles | 64.42 | 63.80 | 62.94 | 493 | Jaguars |
| 29 | 13 | Nick Foles | 63.69 | 68.94 | 73.12 | 221 | Eagles |
| 30 | 14 | Nick Mullens | 62.60 | 64.84 | 71.44 | 316 | 49ers |

### Rotation/backup (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Jimmy Garoppolo | 61.74 | 65.00 | 72.01 | 117 | 49ers |
| 32 | 2 | Sam Darnold | 61.25 | 62.63 | 62.15 | 477 | Jets |
| 33 | 3 | C.J. Beathard | 59.19 | 61.52 | 63.04 | 207 | 49ers |
| 34 | 4 | Lamar Jackson | 59.14 | 59.58 | 64.44 | 217 | Ravens |
| 35 | 5 | Josh Allen | 59.13 | 58.32 | 59.65 | 429 | Bills |
| 36 | 6 | Ryan Tannehill | 57.83 | 49.55 | 69.72 | 343 | Dolphins |
| 37 | 7 | Tyrod Taylor | 57.05 | 64.38 | 56.89 | 115 | Browns |
| 38 | 8 | Josh Rosen | 56.86 | 49.39 | 57.44 | 481 | Cardinals |
| 39 | 9 | Cody Kessler | 56.78 | 57.69 | 58.44 | 177 | Jaguars |
| 40 | 10 | Jeff Driskel | 56.69 | 54.56 | 60.17 | 223 | Bengals |
| 41 | 11 | Brock Osweiler | 55.77 | 55.01 | 62.88 | 211 | Dolphins |
| 42 | 12 | Josh Johnson | 55.32 | 51.92 | 56.79 | 126 | Commanders |
| 43 | 13 | Josh McCown | 54.65 | 58.71 | 56.92 | 126 | Jets |
| 44 | 14 | Blaine Gabbert | 54.48 | 51.49 | 58.08 | 115 | Titans |

## S — Safety

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (22 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jamal Adams | 90.83 | 89.60 | 87.49 | 1119 | Jets |
| 2 | 2 | Eddie Jackson | 90.49 | 94.70 | 84.82 | 906 | Bears |
| 3 | 3 | John Johnson III | 89.81 | 86.10 | 88.11 | 961 | Rams |
| 4 | 4 | Derwin James Jr. | 88.80 | 89.70 | 84.04 | 1027 | Chargers |
| 5 | 5 | Kevin Byard | 87.42 | 82.50 | 86.53 | 1042 | Titans |
| 6 | 6 | Micah Hyde | 86.76 | 88.60 | 81.89 | 882 | Bills |
| 7 | 7 | Earl Thomas III | 86.10 | 82.08 | 92.53 | 237 | Seahawks |
| 8 | 8 | Ha Ha Clinton-Dix | 85.42 | 82.00 | 83.54 | 1025 | Commanders |
| 9 | 9 | Anthony Harris | 85.19 | 83.57 | 89.41 | 624 | Vikings |
| 10 | 10 | Adrian Amos | 84.88 | 85.00 | 82.10 | 1029 | Bears |
| 11 | 11 | Tracy Walker III | 84.79 | 73.62 | 88.07 | 268 | Lions |
| 12 | 12 | Jessie Bates III | 83.90 | 80.90 | 81.74 | 1114 | Bengals |
| 13 | 13 | Malcolm Jenkins | 83.04 | 83.30 | 78.70 | 1038 | Eagles |
| 14 | 14 | Damontae Kazee | 82.87 | 82.70 | 78.81 | 991 | Falcons |
| 15 | 15 | Devin McCourty | 82.76 | 80.00 | 80.43 | 1004 | Patriots |
| 16 | 16 | Bradley McDougald | 82.58 | 77.90 | 81.54 | 874 | Seahawks |
| 17 | 17 | Eric Weddle | 82.33 | 76.90 | 81.79 | 1015 | Ravens |
| 18 | 18 | Malik Hooker | 81.25 | 81.60 | 81.67 | 912 | Colts |
| 19 | 19 | Justin Reid | 80.66 | 74.70 | 80.46 | 906 | Texans |
| 20 | 20 | Clayton Fejedelem | 80.51 | 72.78 | 85.76 | 167 | Bengals |
| 21 | 21 | D.J. Swearinger Sr. | 80.33 | 79.50 | 77.23 | 961 | Cardinals |
| 22 | 22 | Tre Boston | 80.00 | 78.20 | 78.28 | 950 | Cardinals |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Erik Harris | 79.44 | 74.42 | 79.65 | 433 | Raiders |
| 24 | 2 | Jabrill Peppers | 78.80 | 76.66 | 77.22 | 765 | Browns |
| 25 | 3 | Shawn Williams | 78.60 | 78.70 | 76.13 | 995 | Bengals |
| 26 | 4 | Lamarcus Joyner | 77.87 | 71.20 | 79.62 | 907 | Rams |
| 27 | 5 | Harrison Smith | 77.86 | 68.70 | 80.21 | 1025 | Vikings |
| 28 | 6 | Patrick Chung | 77.00 | 73.90 | 75.41 | 888 | Patriots |
| 29 | 7 | Tyrann Mathieu | 76.88 | 74.10 | 75.81 | 1045 | Texans |
| 30 | 8 | Tashaun Gipson Sr. | 76.64 | 71.90 | 75.64 | 1006 | Jaguars |
| 31 | 9 | Damarious Randall | 76.23 | 71.40 | 75.80 | 1083 | Browns |
| 32 | 10 | Xavier Woods | 75.91 | 75.20 | 73.91 | 883 | Cowboys |
| 33 | 11 | Marcus Williams | 75.81 | 69.50 | 75.85 | 956 | Saints |
| 34 | 12 | Ricardo Allen | 74.99 | 74.12 | 78.17 | 205 | Falcons |
| 35 | 13 | Duron Harmon | 74.90 | 67.79 | 75.48 | 636 | Patriots |
| 36 | 14 | Will Parks | 74.61 | 73.07 | 71.67 | 572 | Broncos |
| 37 | 15 | Antoine Bethea | 74.52 | 73.50 | 71.35 | 1111 | Cardinals |
| 38 | 16 | Sean Davis | 74.44 | 71.50 | 72.75 | 980 | Steelers |
| 39 | 17 | Tony Jefferson | 74.18 | 72.50 | 72.60 | 862 | Ravens |

### Starter (46 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 40 | 1 | Reshad Jones | 73.96 | 71.70 | 74.44 | 825 | Dolphins |
| 41 | 2 | Mike Mitchell | 73.36 | 71.33 | 75.33 | 224 | Colts |
| 42 | 3 | Jordan Poyer | 73.33 | 70.90 | 72.87 | 1010 | Bills |
| 43 | 4 | Vonn Bell | 73.27 | 67.93 | 72.86 | 753 | Saints |
| 44 | 5 | Quandre Diggs | 73.03 | 67.10 | 72.82 | 948 | Lions |
| 45 | 6 | Jordan Whitehead | 72.50 | 72.88 | 69.12 | 660 | Buccaneers |
| 46 | 7 | Marcus Maye | 72.46 | 67.73 | 77.96 | 393 | Jets |
| 47 | 8 | Adrian Phillips | 71.98 | 65.74 | 72.70 | 685 | Chargers |
| 48 | 9 | Andre Hal | 71.90 | 69.96 | 73.19 | 237 | Texans |
| 49 | 10 | Rodney McLeod | 71.72 | 72.12 | 74.06 | 162 | Eagles |
| 50 | 11 | Karl Joseph | 71.57 | 66.28 | 73.85 | 509 | Raiders |
| 51 | 12 | Landon Collins | 71.43 | 68.14 | 71.86 | 804 | Giants |
| 52 | 13 | Ibraheim Campbell | 71.31 | 66.24 | 80.62 | 114 | Packers |
| 53 | 14 | Jarrod Wilson | 71.23 | 64.51 | 75.20 | 222 | Jaguars |
| 54 | 15 | Terrell Edmunds | 70.50 | 65.20 | 69.87 | 967 | Steelers |
| 55 | 16 | Tavon Wilson | 70.38 | 65.50 | 72.39 | 304 | Lions |
| 56 | 17 | Eric Murray | 70.33 | 67.57 | 69.03 | 703 | Chiefs |
| 57 | 18 | Deon Bush | 70.31 | 67.55 | 75.59 | 152 | Bears |
| 58 | 19 | Michael Thomas | 70.00 | 68.03 | 69.96 | 522 | Giants |
| 59 | 20 | Corey Graham | 69.62 | 63.36 | 71.20 | 655 | Eagles |
| 60 | 21 | Andrew Sendejo | 69.31 | 68.32 | 72.25 | 326 | Vikings |
| 61 | 22 | Rashaan Gaulden | 69.26 | 63.50 | 74.13 | 143 | Panthers |
| 62 | 23 | Ronnie Harrison | 69.15 | 65.61 | 69.43 | 328 | Jaguars |
| 63 | 24 | George Iloka | 68.82 | 61.92 | 73.94 | 117 | Vikings |
| 64 | 25 | Glover Quin | 68.79 | 63.40 | 68.21 | 829 | Lions |
| 65 | 26 | Justin Evans | 68.65 | 64.97 | 72.02 | 605 | Buccaneers |
| 66 | 27 | Morgan Burnett | 68.40 | 61.93 | 72.40 | 390 | Steelers |
| 67 | 28 | T.J. McDonald | 68.35 | 66.70 | 68.84 | 952 | Dolphins |
| 68 | 29 | Daniel Sorensen | 67.72 | 66.34 | 69.15 | 354 | Chiefs |
| 69 | 30 | Clayton Geathers | 67.39 | 64.75 | 71.96 | 715 | Colts |
| 70 | 31 | Eric Reid | 67.31 | 66.48 | 67.44 | 736 | Panthers |
| 71 | 32 | Kenny Vaccaro | 67.01 | 61.52 | 70.35 | 747 | Titans |
| 72 | 33 | Curtis Riley | 66.70 | 63.90 | 64.40 | 1048 | Giants |
| 73 | 34 | Reggie Nelson | 66.60 | 62.73 | 67.62 | 370 | Raiders |
| 74 | 35 | Andrew Adams | 66.34 | 66.03 | 66.55 | 370 | Buccaneers |
| 75 | 36 | Doug Middleton | 65.53 | 61.01 | 73.12 | 231 | Jets |
| 76 | 37 | Tedric Thompson | 65.51 | 65.71 | 67.98 | 656 | Seahawks |
| 77 | 38 | Jordan Richards | 65.27 | 63.32 | 65.42 | 429 | Falcons |
| 78 | 39 | Jahleel Addae | 64.73 | 60.10 | 65.32 | 1025 | Chargers |
| 79 | 40 | George Odum | 64.32 | 61.64 | 68.19 | 205 | Colts |
| 80 | 41 | Darian Stewart | 64.05 | 60.80 | 63.08 | 874 | Broncos |
| 81 | 42 | Matthias Farley | 64.03 | 62.57 | 68.44 | 151 | Colts |
| 82 | 43 | Sharrod Neasman | 63.80 | 65.45 | 67.50 | 436 | Falcons |
| 83 | 44 | Antone Exum Jr. | 62.89 | 62.97 | 66.91 | 594 | 49ers |
| 84 | 45 | Josh Jones | 62.83 | 59.38 | 64.62 | 502 | Packers |
| 85 | 46 | Jeff Heath | 62.21 | 59.60 | 60.71 | 1000 | Cowboys |

### Rotation/backup (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 86 | 1 | Jaquiski Tartt | 61.89 | 58.74 | 66.39 | 437 | 49ers |
| 87 | 2 | Chuck Clark | 61.54 | 58.94 | 62.75 | 255 | Ravens |
| 88 | 3 | Justin Simmons | 61.08 | 51.20 | 65.26 | 1078 | Broncos |
| 89 | 4 | Marcus Gilchrist | 60.79 | 53.50 | 62.12 | 871 | Raiders |
| 90 | 5 | Deshazor Everett | 60.68 | 57.65 | 62.50 | 136 | Commanders |
| 91 | 6 | Delano Hill | 60.42 | 57.57 | 64.41 | 320 | Seahawks |
| 92 | 7 | Jermaine Whitehead | 60.39 | 59.00 | 62.98 | 228 | Browns |
| 93 | 8 | Marqui Christian | 60.22 | 59.22 | 61.10 | 349 | Rams |
| 94 | 9 | Kurt Coleman | 59.70 | 54.99 | 59.83 | 359 | Saints |
| 95 | 10 | Tre Sullivan | 58.99 | 61.97 | 58.04 | 222 | Eagles |
| 96 | 11 | Jimmie Ward | 58.88 | 54.03 | 64.39 | 388 | 49ers |
| 97 | 12 | Sean Chandler | 58.85 | 58.88 | 59.86 | 142 | Giants |
| 98 | 13 | Chris Conte | 58.07 | 58.08 | 61.08 | 118 | Buccaneers |
| 99 | 14 | Jordan Lucas | 57.85 | 60.56 | 62.02 | 262 | Chiefs |
| 100 | 15 | Derrick Kindred | 57.55 | 54.25 | 57.04 | 498 | Browns |
| 101 | 16 | Mike Adams | 57.00 | 42.10 | 62.97 | 938 | Panthers |
| 102 | 17 | Kendrick Lewis | 56.64 | 53.70 | 62.03 | 276 | Titans |
| 103 | 18 | Kentrell Brice | 55.85 | 53.00 | 57.75 | 648 | Packers |
| 104 | 19 | Kavon Frazier | 53.41 | 52.59 | 55.72 | 186 | Cowboys |
| 105 | 20 | Montae Nicholson | 50.12 | 47.06 | 55.02 | 467 | Commanders |
| 106 | 21 | Isaiah Johnson | 49.09 | 44.46 | 57.17 | 404 | Buccaneers |
| 107 | 22 | Tyvis Powell | 48.57 | 48.24 | 56.29 | 104 | 49ers |
| 108 | 23 | Marcell Harris | 48.40 | 52.89 | 52.10 | 358 | 49ers |
| 109 | 24 | Su'a Cravens | 47.72 | 47.68 | 55.24 | 117 | Broncos |
| 110 | 25 | Adrian Colbert | 46.69 | 40.00 | 56.23 | 320 | 49ers |

## T — Tackle

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Terron Armstead | 92.92 | 86.42 | 93.09 | 602 | Saints |
| 2 | 2 | Rob Havenstein | 91.98 | 86.10 | 91.73 | 1101 | Rams |
| 3 | 3 | David Bakhtiari | 90.88 | 88.90 | 88.04 | 1032 | Packers |
| 4 | 4 | Joe Staley | 90.13 | 84.70 | 89.59 | 1006 | 49ers |
| 5 | 5 | Mitchell Schwartz | 89.41 | 84.20 | 88.72 | 1045 | Chiefs |
| 6 | 6 | Andrew Whitworth | 89.11 | 83.60 | 88.61 | 1038 | Rams |
| 7 | 7 | Ryan Ramczyk | 89.05 | 83.50 | 88.59 | 996 | Saints |
| 8 | 8 | Duane Brown | 88.50 | 83.00 | 88.00 | 1067 | Seahawks |
| 9 | 9 | Tyron Smith | 87.53 | 80.00 | 88.39 | 849 | Cowboys |
| 10 | 10 | Lane Johnson | 87.11 | 80.85 | 87.11 | 962 | Eagles |
| 11 | 11 | Alejandro Villanueva | 85.88 | 81.20 | 84.83 | 1116 | Steelers |
| 12 | 12 | Jake Matthews | 85.22 | 80.00 | 84.54 | 1057 | Falcons |
| 13 | 13 | Ronnie Stanley | 85.12 | 77.50 | 86.03 | 1084 | Ravens |
| 14 | 14 | Russell Okung | 84.46 | 77.71 | 84.79 | 866 | Chargers |
| 15 | 15 | Anthony Castonzo | 84.05 | 76.31 | 85.05 | 744 | Colts |
| 16 | 16 | Mike McGlinchey | 83.85 | 74.80 | 85.72 | 1055 | 49ers |
| 17 | 17 | Taylor Lewan | 83.74 | 75.71 | 84.92 | 852 | Titans |
| 18 | 18 | Charles Leno Jr. | 83.30 | 75.70 | 84.20 | 1067 | Bears |
| 19 | 19 | Trent Williams | 83.13 | 74.63 | 84.63 | 792 | Commanders |
| 20 | 20 | Taylor Moton | 83.11 | 76.60 | 83.29 | 1054 | Panthers |
| 21 | 21 | Laremy Tunsil | 82.86 | 73.23 | 85.11 | 820 | Dolphins |
| 22 | 22 | Nate Solder | 82.60 | 75.70 | 83.04 | 1027 | Giants |
| 23 | 23 | Marcus Cannon | 82.50 | 73.05 | 84.64 | 836 | Patriots |
| 24 | 24 | Dennis Kelly | 82.50 | 69.51 | 86.99 | 376 | Titans |
| 25 | 25 | Garett Bolles | 82.32 | 72.80 | 84.50 | 1062 | Broncos |
| 26 | 26 | Eric Fisher | 81.97 | 73.40 | 83.52 | 1042 | Chiefs |
| 27 | 27 | George Fant | 81.94 | 68.59 | 86.67 | 371 | Seahawks |
| 28 | 28 | Bryan Bulaga | 81.56 | 74.01 | 82.42 | 781 | Packers |
| 29 | 29 | La'el Collins | 81.55 | 72.50 | 83.42 | 1075 | Cowboys |
| 30 | 30 | Riley Reiff | 81.12 | 73.23 | 82.21 | 793 | Vikings |
| 31 | 31 | Ja'Wuan James | 80.98 | 71.73 | 82.98 | 816 | Dolphins |
| 32 | 32 | Bobby Massie | 80.52 | 71.80 | 82.17 | 1070 | Bears |
| 33 | 33 | Jason Peters | 80.17 | 70.69 | 82.33 | 868 | Eagles |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Ty Nsekhe | 79.55 | 64.99 | 85.09 | 403 | Commanders |
| 35 | 2 | Matt Feiler | 79.21 | 69.69 | 81.39 | 675 | Steelers |
| 36 | 3 | Trent Brown | 79.18 | 69.70 | 81.34 | 1090 | Patriots |
| 37 | 4 | Rick Wagner | 79.14 | 71.40 | 80.13 | 985 | Lions |
| 38 | 5 | Dion Dawkins | 78.90 | 69.90 | 80.73 | 1059 | Bills |
| 39 | 6 | D.J. Humphries | 78.87 | 67.35 | 82.38 | 522 | Cardinals |
| 40 | 7 | Ty Sambrailo | 78.54 | 63.43 | 84.44 | 266 | Falcons |
| 41 | 8 | Ryan Schraeder | 78.26 | 65.77 | 82.42 | 865 | Falcons |
| 42 | 9 | Taylor Decker | 78.07 | 70.60 | 78.88 | 1062 | Lions |
| 43 | 10 | Kelvin Beachum | 77.86 | 68.50 | 79.94 | 1001 | Jets |
| 44 | 11 | Jack Conklin | 77.40 | 65.62 | 81.08 | 498 | Titans |
| 45 | 12 | Marcus Gilbert | 77.31 | 67.33 | 79.80 | 362 | Steelers |
| 46 | 13 | Jared Veldheer | 77.19 | 64.35 | 81.59 | 704 | Broncos |
| 47 | 14 | Morgan Moses | 77.07 | 64.37 | 81.37 | 965 | Commanders |
| 48 | 15 | Demar Dotson | 76.67 | 67.40 | 78.68 | 1005 | Buccaneers |
| 49 | 16 | Jermey Parnell | 76.59 | 65.49 | 79.82 | 869 | Jaguars |
| 50 | 17 | Jason Spriggs | 75.96 | 62.56 | 80.73 | 292 | Packers |
| 51 | 18 | Le'Raven Clark | 75.73 | 64.63 | 78.97 | 365 | Colts |
| 52 | 19 | Brandon Shell | 75.53 | 63.44 | 79.43 | 850 | Jets |
| 53 | 20 | Donovan Smith | 74.94 | 66.40 | 76.47 | 1117 | Buccaneers |
| 54 | 21 | Ereck Flowers | 74.75 | 64.17 | 77.63 | 588 | Jaguars |
| 55 | 22 | Joe Barksdale | 74.56 | 62.02 | 78.75 | 388 | Cardinals |
| 56 | 23 | Kendall Lamm | 74.27 | 64.11 | 76.88 | 859 | Texans |
| 57 | 24 | Chris Hubbard | 74.20 | 65.10 | 76.10 | 1091 | Browns |
| 58 | 25 | Orlando Brown Jr. | 74.04 | 66.85 | 74.67 | 760 | Ravens |

### Starter (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | James Hurst | 73.09 | 60.58 | 77.26 | 675 | Ravens |
| 60 | 2 | Chris Clark | 72.86 | 61.61 | 76.20 | 818 | Panthers |
| 61 | 3 | Greg Robinson | 72.62 | 60.21 | 76.73 | 498 | Browns |
| 62 | 4 | Josh Wells | 72.59 | 61.17 | 76.04 | 305 | Jaguars |
| 63 | 5 | Joe Haeg | 72.37 | 59.39 | 76.85 | 367 | Colts |
| 64 | 6 | Andre Smith | 72.35 | 59.03 | 77.07 | 452 | Bengals |
| 65 | 7 | Cordy Glenn | 72.28 | 61.30 | 75.44 | 765 | Bengals |
| 66 | 8 | Korey Cunningham | 71.76 | 60.24 | 75.28 | 349 | Cardinals |
| 67 | 9 | Cam Fleming | 71.64 | 58.30 | 76.37 | 232 | Cowboys |
| 68 | 10 | John Wetzel | 71.41 | 58.30 | 75.98 | 339 | Cardinals |
| 69 | 11 | Marshall Newhouse | 71.40 | 58.06 | 76.12 | 211 | Panthers |
| 70 | 12 | Bobby Hart | 70.95 | 56.70 | 76.28 | 994 | Bengals |
| 71 | 13 | Desmond Harrison | 70.89 | 57.90 | 75.38 | 595 | Browns |
| 72 | 14 | Rashod Hill | 70.87 | 58.90 | 74.69 | 529 | Vikings |
| 73 | 15 | Germain Ifedi | 70.72 | 56.50 | 76.03 | 989 | Seahawks |
| 74 | 16 | Brian O'Neill | 70.41 | 62.70 | 71.39 | 800 | Vikings |
| 75 | 17 | Jordan Mills | 70.14 | 56.50 | 75.06 | 1013 | Bills |
| 76 | 18 | Sam Tevi | 68.76 | 52.76 | 75.26 | 871 | Chargers |
| 77 | 19 | Kevin Pamphile | 68.34 | 58.24 | 70.91 | 155 | Titans |
| 78 | 20 | LaAdrian Waddle | 67.96 | 55.17 | 72.32 | 342 | Patriots |
| 79 | 21 | Brent Qvale | 67.86 | 53.49 | 73.28 | 159 | Jets |
| 80 | 22 | Sam Young | 67.36 | 54.32 | 71.89 | 121 | Dolphins |
| 81 | 23 | Donald Penn | 67.18 | 52.39 | 72.87 | 188 | Raiders |
| 82 | 24 | Halapoulivaati Vaitai | 67.02 | 54.12 | 71.45 | 334 | Eagles |
| 83 | 25 | Julie'n Davenport | 66.96 | 52.70 | 72.30 | 1014 | Texans |
| 84 | 26 | Chukwuma Okorafor | 66.73 | 56.32 | 69.50 | 156 | Steelers |
| 85 | 27 | Will Holden | 65.87 | 52.19 | 70.83 | 294 | Cardinals |
| 86 | 28 | Brandon Parker | 65.82 | 48.70 | 73.07 | 780 | Raiders |
| 87 | 29 | Chad Wheeler | 65.44 | 49.65 | 71.80 | 857 | Giants |
| 88 | 30 | Kolton Miller | 64.73 | 49.60 | 70.65 | 1008 | Raiders |
| 89 | 31 | Leonard Wester | 62.59 | 54.68 | 63.70 | 118 | Buccaneers |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Travis Kelce | 84.70 | 88.00 | 78.33 | 661 | Chiefs |
| 2 | 2 | O.J. Howard | 83.74 | 79.19 | 82.60 | 291 | Buccaneers |
| 3 | 3 | George Kittle | 83.27 | 87.53 | 76.27 | 567 | 49ers |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Rob Gronkowski | 79.21 | 72.47 | 79.53 | 477 | Patriots |
| 5 | 2 | Zach Ertz | 78.19 | 75.58 | 75.76 | 657 | Eagles |
| 6 | 3 | Anthony Firkser | 76.66 | 64.41 | 80.66 | 139 | Titans |
| 7 | 4 | Chris Herndon | 76.47 | 71.07 | 75.90 | 369 | Jets |
| 8 | 5 | Mark Andrews | 75.54 | 70.39 | 74.80 | 289 | Ravens |
| 9 | 6 | Evan Engram | 75.27 | 71.77 | 73.44 | 344 | Giants |
| 10 | 7 | Jared Cook | 74.34 | 74.60 | 70.00 | 538 | Raiders |
| 11 | 8 | Antonio Gates | 74.24 | 67.77 | 74.38 | 283 | Chargers |
| 12 | 9 | Gerald Everett | 74.17 | 74.54 | 69.76 | 278 | Rams |

### Starter (62 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Benjamin Watson | 73.42 | 69.19 | 72.08 | 302 | Saints |
| 14 | 2 | Greg Olsen | 72.96 | 64.56 | 74.39 | 274 | Panthers |
| 15 | 3 | Jack Doyle | 72.93 | 68.89 | 71.45 | 178 | Colts |
| 16 | 4 | Jordan Reed | 72.92 | 71.13 | 69.95 | 370 | Commanders |
| 17 | 5 | Vance McDonald | 72.45 | 67.63 | 71.49 | 444 | Steelers |
| 18 | 6 | Eric Ebron | 72.33 | 66.40 | 72.11 | 493 | Colts |
| 19 | 7 | Vernon Davis | 72.10 | 60.76 | 75.50 | 244 | Commanders |
| 20 | 8 | Austin Hooper | 71.84 | 67.22 | 70.76 | 566 | Falcons |
| 21 | 9 | Tyler Higbee | 71.83 | 68.77 | 69.70 | 391 | Rams |
| 22 | 10 | Dallas Goedert | 71.81 | 69.36 | 69.27 | 295 | Eagles |
| 23 | 11 | Trey Burton | 71.75 | 68.57 | 69.71 | 549 | Bears |
| 24 | 12 | Levine Toilolo | 71.35 | 63.87 | 72.17 | 265 | Lions |
| 25 | 13 | Tyler Eifert | 71.21 | 67.68 | 69.40 | 103 | Bengals |
| 26 | 14 | Darren Fells | 70.60 | 62.50 | 71.84 | 165 | Browns |
| 27 | 15 | Jesse James | 70.46 | 63.73 | 70.78 | 339 | Steelers |
| 28 | 16 | Dalton Schultz | 70.44 | 62.37 | 71.66 | 143 | Cowboys |
| 29 | 17 | Hayden Hurst | 70.43 | 62.82 | 71.33 | 128 | Ravens |
| 30 | 18 | Nick O'Leary | 69.88 | 61.61 | 71.23 | 167 | Dolphins |
| 31 | 19 | Luke Stocker | 69.83 | 66.98 | 67.56 | 124 | Titans |
| 32 | 20 | Garrett Celek | 69.61 | 61.08 | 71.13 | 100 | 49ers |
| 33 | 21 | Maxx Williams | 69.41 | 63.88 | 68.93 | 105 | Ravens |
| 34 | 22 | Nick Boyle | 69.19 | 64.69 | 68.02 | 239 | Ravens |
| 35 | 23 | Ricky Seals-Jones | 69.12 | 54.38 | 74.78 | 402 | Cardinals |
| 36 | 24 | Blake Jarwin | 69.11 | 65.45 | 67.39 | 253 | Cowboys |
| 37 | 25 | David Njoku | 68.95 | 65.94 | 66.79 | 586 | Browns |
| 38 | 26 | Lee Smith | 68.87 | 67.37 | 65.70 | 106 | Raiders |
| 39 | 27 | Virgil Green | 68.81 | 58.91 | 71.24 | 307 | Chargers |
| 40 | 28 | Jimmy Graham | 68.81 | 59.61 | 70.78 | 611 | Packers |
| 41 | 29 | Jeff Heuerman | 68.76 | 60.91 | 69.82 | 321 | Broncos |
| 42 | 30 | Jordan Akins | 68.74 | 62.14 | 68.97 | 209 | Texans |
| 43 | 31 | Austin Seferian-Jenkins | 68.73 | 64.16 | 67.61 | 138 | Jaguars |
| 44 | 32 | Charles Clay | 68.66 | 57.30 | 72.06 | 283 | Bills |
| 45 | 33 | Blake Bell | 68.60 | 58.19 | 71.37 | 123 | Jaguars |
| 46 | 34 | Rhett Ellison | 68.41 | 63.05 | 67.81 | 318 | Giants |
| 47 | 35 | Kyle Rudolph | 68.26 | 63.58 | 67.21 | 647 | Vikings |
| 48 | 36 | Cameron Brate | 67.77 | 54.01 | 72.78 | 401 | Buccaneers |
| 49 | 37 | Michael Roberts | 67.68 | 58.24 | 69.81 | 101 | Lions |
| 50 | 38 | Luke Willson | 67.68 | 56.88 | 70.71 | 221 | Lions |
| 51 | 39 | Demetrius Harris | 67.30 | 60.64 | 67.58 | 220 | Chiefs |
| 52 | 40 | Logan Paulsen | 67.23 | 60.99 | 67.22 | 163 | Falcons |
| 53 | 41 | Matt LaCosse | 67.03 | 58.21 | 68.74 | 251 | Broncos |
| 54 | 42 | Jason Croom | 66.77 | 55.44 | 70.16 | 220 | Bills |
| 55 | 43 | Lance Kendricks | 66.68 | 59.81 | 67.10 | 154 | Packers |
| 56 | 44 | Jermaine Gresham | 66.62 | 52.48 | 71.88 | 200 | Cardinals |
| 57 | 45 | Jordan Leggett | 66.05 | 62.90 | 63.98 | 171 | Jets |
| 58 | 46 | Ed Dickson | 65.89 | 61.31 | 64.77 | 198 | Seahawks |
| 59 | 47 | Derek Carrier | 65.84 | 59.63 | 65.82 | 111 | Raiders |
| 60 | 48 | Jordan Thomas | 65.83 | 56.09 | 68.15 | 239 | Texans |
| 61 | 49 | Nick Vannett | 65.82 | 62.53 | 63.84 | 265 | Seahawks |
| 62 | 50 | Ian Thomas | 65.30 | 55.13 | 67.91 | 349 | Panthers |
| 63 | 51 | Scott Simonson | 64.84 | 58.72 | 64.76 | 129 | Giants |
| 64 | 52 | Josh Hill | 64.81 | 57.03 | 65.83 | 275 | Saints |
| 65 | 53 | Ryan Griffin | 64.66 | 56.69 | 65.81 | 452 | Texans |
| 66 | 54 | James O'Shaughnessy | 64.64 | 60.07 | 63.52 | 311 | Jaguars |
| 67 | 55 | Geoff Swaim | 64.44 | 57.21 | 65.09 | 291 | Cowboys |
| 68 | 56 | Dwayne Allen | 64.29 | 54.04 | 66.95 | 149 | Patriots |
| 69 | 57 | Jonnu Smith | 63.81 | 56.65 | 64.41 | 308 | Titans |
| 70 | 58 | Eric Tomlinson | 63.80 | 55.17 | 65.38 | 151 | Jets |
| 71 | 59 | C.J. Uzomah | 63.73 | 57.16 | 63.95 | 520 | Bengals |
| 72 | 60 | Antony Auclair | 62.42 | 53.26 | 64.36 | 137 | Buccaneers |
| 73 | 61 | Mike Gesicki | 62.32 | 53.70 | 63.90 | 273 | Dolphins |
| 74 | 62 | Jeremy Sprinkle | 62.16 | 52.98 | 64.12 | 159 | Commanders |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 75 | 1 | Logan Thomas | 60.01 | 55.82 | 58.64 | 167 | Bills |

## WR — Wide Receiver

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Julio Jones | 88.93 | 90.11 | 83.98 | 605 | Falcons |
| 2 | 2 | Tyreek Hill | 87.79 | 88.82 | 82.94 | 603 | Chiefs |
| 3 | 3 | DeAndre Hopkins | 87.62 | 92.40 | 80.26 | 673 | Texans |
| 4 | 4 | Michael Thomas | 86.13 | 89.85 | 79.49 | 562 | Saints |
| 5 | 5 | T.Y. Hilton | 85.92 | 85.01 | 82.36 | 535 | Colts |
| 6 | 6 | Keenan Allen | 85.54 | 88.09 | 79.67 | 509 | Chargers |
| 7 | 7 | Robert Woods | 85.54 | 87.71 | 79.93 | 645 | Rams |
| 8 | 8 | Adam Thielen | 84.81 | 89.40 | 77.59 | 694 | Vikings |
| 9 | 9 | Odell Beckham Jr. | 84.49 | 87.48 | 78.33 | 488 | Giants |
| 10 | 10 | Mike Evans | 83.69 | 84.30 | 79.11 | 679 | Buccaneers |
| 11 | 11 | Davante Adams | 83.23 | 87.80 | 76.01 | 694 | Packers |
| 12 | 12 | Antonio Brown | 82.79 | 79.30 | 80.95 | 716 | Steelers |
| 13 | 13 | JuJu Smith-Schuster | 82.59 | 81.80 | 78.95 | 728 | Steelers |
| 14 | 14 | A.J. Green | 82.54 | 80.49 | 79.74 | 313 | Bengals |
| 15 | 15 | Kenny Golladay | 82.36 | 80.17 | 79.65 | 609 | Lions |
| 16 | 16 | Brandin Cooks | 81.90 | 79.30 | 79.46 | 616 | Rams |
| 17 | 17 | DeSean Jackson | 81.61 | 76.36 | 80.94 | 360 | Buccaneers |
| 18 | 18 | Tyler Lockett | 81.56 | 78.66 | 79.33 | 494 | Seahawks |
| 19 | 19 | Chris Godwin | 81.31 | 77.45 | 79.71 | 483 | Buccaneers |
| 20 | 20 | Tyler Boyd | 80.91 | 82.04 | 75.99 | 534 | Bengals |
| 21 | 21 | Robert Foster | 80.23 | 67.90 | 84.28 | 281 | Bills |

### Good (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Josh Gordon | 79.76 | 71.73 | 80.94 | 403 | Patriots |
| 23 | 2 | Stefon Diggs | 79.66 | 80.75 | 74.76 | 624 | Vikings |
| 24 | 3 | Doug Baldwin | 79.64 | 77.61 | 76.83 | 379 | Seahawks |
| 25 | 4 | Emmanuel Sanders | 79.13 | 78.77 | 75.20 | 439 | Broncos |
| 26 | 5 | Cooper Kupp | 79.05 | 73.57 | 78.53 | 271 | Rams |
| 27 | 6 | Amari Cooper | 78.95 | 77.25 | 75.91 | 583 | Cowboys |
| 28 | 7 | Alshon Jeffery | 78.78 | 76.93 | 75.85 | 506 | Eagles |
| 29 | 8 | Mike Williams | 78.55 | 79.00 | 74.09 | 408 | Chargers |
| 30 | 9 | Dante Pettis | 78.53 | 65.57 | 83.00 | 283 | 49ers |
| 31 | 10 | Albert Wilson | 77.91 | 75.24 | 75.52 | 141 | Dolphins |
| 32 | 11 | DJ Moore | 77.66 | 69.75 | 78.76 | 451 | Panthers |
| 33 | 12 | Corey Davis | 77.22 | 74.49 | 74.87 | 515 | Titans |
| 34 | 13 | Will Fuller V | 76.98 | 73.05 | 75.44 | 250 | Texans |
| 35 | 14 | Marvin Jones Jr. | 76.93 | 70.07 | 77.33 | 365 | Lions |
| 36 | 15 | Sammy Watkins | 76.87 | 70.93 | 76.67 | 308 | Chiefs |
| 37 | 16 | Allen Robinson II | 76.07 | 73.40 | 73.69 | 480 | Bears |
| 38 | 17 | Jarvis Landry | 75.97 | 74.59 | 72.73 | 643 | Browns |
| 39 | 18 | Julian Edelman | 75.89 | 73.67 | 73.21 | 457 | Patriots |
| 40 | 19 | Mohamed Sanu | 75.61 | 71.26 | 74.35 | 597 | Falcons |
| 41 | 20 | Christian Kirk | 75.37 | 67.65 | 76.35 | 364 | Cardinals |
| 42 | 21 | Adam Humphries | 75.27 | 74.50 | 71.62 | 601 | Buccaneers |
| 43 | 22 | Equanimeous St. Brown | 75.03 | 62.72 | 79.07 | 265 | Packers |
| 44 | 23 | Jakeem Grant Sr. | 74.98 | 65.33 | 77.25 | 170 | Dolphins |
| 45 | 24 | Tyrell Williams | 74.81 | 66.26 | 76.34 | 496 | Chargers |
| 46 | 25 | Ted Ginn Jr. | 74.57 | 64.95 | 76.82 | 151 | Saints |
| 47 | 26 | Golden Tate | 74.52 | 68.88 | 74.11 | 421 | Eagles |
| 48 | 27 | Cole Beasley | 74.34 | 73.73 | 70.58 | 510 | Cowboys |
| 49 | 28 | Larry Fitzgerald | 74.25 | 72.58 | 71.19 | 561 | Cardinals |
| 50 | 29 | Marquise Goodwin | 74.01 | 62.10 | 77.78 | 263 | 49ers |

### Starter (95 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 51 | 1 | David Moore | 73.95 | 63.91 | 76.47 | 346 | Seahawks |
| 52 | 2 | Josh Reynolds | 73.87 | 66.64 | 74.53 | 367 | Rams |
| 53 | 3 | Jordy Nelson | 73.85 | 70.17 | 72.13 | 556 | Raiders |
| 54 | 4 | Tre'Quan Smith | 73.80 | 66.24 | 74.67 | 340 | Saints |
| 55 | 5 | Taylor Gabriel | 73.67 | 68.27 | 73.11 | 513 | Bears |
| 56 | 6 | DeVante Parker | 73.63 | 67.04 | 73.85 | 245 | Dolphins |
| 57 | 7 | Keith Kirkwood | 73.63 | 62.81 | 76.67 | 140 | Saints |
| 58 | 8 | Calvin Ridley | 73.55 | 67.58 | 73.36 | 490 | Falcons |
| 59 | 9 | Demaryius Thomas | 73.39 | 70.58 | 71.09 | 495 | Texans |
| 60 | 10 | Breshad Perriman | 73.27 | 65.11 | 74.54 | 145 | Browns |
| 61 | 11 | Cody Latimer | 73.25 | 64.85 | 74.68 | 149 | Giants |
| 62 | 12 | Pierre Garcon | 73.11 | 63.28 | 75.50 | 240 | 49ers |
| 63 | 13 | Kenny Stills | 72.97 | 63.98 | 74.79 | 466 | Dolphins |
| 64 | 14 | John Brown | 72.95 | 65.48 | 73.76 | 514 | Ravens |
| 65 | 15 | Devin Funchess | 72.79 | 67.42 | 72.21 | 429 | Panthers |
| 66 | 16 | Taywan Taylor | 72.60 | 66.36 | 72.59 | 267 | Titans |
| 67 | 17 | Dede Westbrook | 72.46 | 70.56 | 69.56 | 576 | Jaguars |
| 68 | 18 | Martavis Bryant | 72.31 | 62.55 | 74.65 | 153 | Raiders |
| 69 | 19 | Rashard Higgins | 72.15 | 65.85 | 72.19 | 344 | Browns |
| 70 | 20 | Jaron Brown | 71.95 | 63.00 | 73.75 | 165 | Seahawks |
| 71 | 21 | Sterling Shepard | 71.95 | 67.12 | 71.00 | 627 | Giants |
| 72 | 22 | Danny Amendola | 71.93 | 65.88 | 71.79 | 443 | Dolphins |
| 73 | 23 | Travis Benjamin | 71.92 | 60.76 | 75.19 | 192 | Chargers |
| 74 | 24 | Willie Snead IV | 71.80 | 66.45 | 71.20 | 474 | Ravens |
| 75 | 25 | Curtis Samuel | 71.79 | 67.98 | 70.17 | 318 | Panthers |
| 76 | 26 | Cordarrelle Patterson | 71.73 | 65.04 | 72.03 | 100 | Patriots |
| 77 | 27 | Quincy Enunwa | 71.53 | 65.71 | 71.25 | 336 | Jets |
| 78 | 28 | Allen Hurns | 71.50 | 63.52 | 72.65 | 268 | Cowboys |
| 79 | 29 | Michael Gallup | 71.37 | 60.25 | 74.62 | 468 | Cowboys |
| 80 | 30 | DeAndre Carter | 71.29 | 62.05 | 73.29 | 174 | Texans |
| 81 | 31 | Courtland Sutton | 71.28 | 63.73 | 72.14 | 574 | Broncos |
| 82 | 32 | Marquez Valdes-Scantling | 71.20 | 60.35 | 74.26 | 507 | Packers |
| 83 | 33 | Jamison Crowder | 71.10 | 60.72 | 73.85 | 289 | Commanders |
| 84 | 34 | Marvin Hall | 70.81 | 58.24 | 75.03 | 122 | Falcons |
| 85 | 35 | Paul Richardson Jr. | 70.77 | 62.90 | 71.85 | 252 | Commanders |
| 86 | 36 | Brandon LaFell | 70.72 | 63.02 | 71.69 | 135 | Raiders |
| 87 | 37 | Antonio Callaway | 70.72 | 63.59 | 71.31 | 533 | Browns |
| 88 | 38 | Deonte Thompson | 70.66 | 64.08 | 70.88 | 142 | Bills |
| 89 | 39 | Dontrelle Inman | 70.50 | 65.95 | 69.36 | 242 | Colts |
| 90 | 40 | Nelson Agholor | 70.45 | 64.74 | 70.09 | 631 | Eagles |
| 91 | 41 | Donte Moncrief | 70.39 | 62.20 | 71.68 | 567 | Jaguars |
| 92 | 42 | Chris Hogan | 70.34 | 56.09 | 75.67 | 520 | Patriots |
| 93 | 43 | Jordan Matthews | 70.33 | 64.06 | 70.35 | 228 | Eagles |
| 94 | 44 | Kendrick Bourne | 70.29 | 65.12 | 69.57 | 397 | 49ers |
| 95 | 45 | Keke Coutee | 70.28 | 62.42 | 71.36 | 175 | Texans |
| 96 | 46 | Keelan Cole Sr. | 70.27 | 59.45 | 73.32 | 490 | Jaguars |
| 97 | 47 | Kelvin Benjamin | 70.24 | 58.15 | 74.13 | 353 | Chiefs |
| 98 | 48 | Rishard Matthews | 70.06 | 53.65 | 76.84 | 107 | Jets |
| 99 | 49 | Phillip Dorsett | 69.94 | 64.19 | 69.60 | 236 | Patriots |
| 100 | 50 | Jake Kumerow | 69.83 | 59.55 | 72.52 | 111 | Packers |
| 101 | 51 | Josh Doctson | 69.81 | 63.01 | 70.18 | 548 | Commanders |
| 102 | 52 | Richie James | 69.81 | 59.03 | 72.83 | 128 | 49ers |
| 103 | 53 | Geronimo Allison | 69.75 | 60.73 | 71.59 | 179 | Packers |
| 104 | 54 | Cameron Batson | 69.54 | 60.96 | 71.09 | 115 | Titans |
| 105 | 55 | Jermaine Kearse | 69.51 | 62.11 | 70.28 | 404 | Jets |
| 106 | 56 | Randall Cobb | 69.37 | 60.17 | 71.34 | 358 | Packers |
| 107 | 57 | Jarius Wright | 69.27 | 62.91 | 69.34 | 408 | Panthers |
| 108 | 58 | Aldrick Robinson | 68.95 | 60.96 | 70.11 | 187 | Vikings |
| 109 | 59 | Tim Patrick | 68.82 | 60.13 | 70.44 | 261 | Broncos |
| 110 | 60 | Anthony Miller | 68.76 | 60.31 | 70.23 | 384 | Bears |
| 111 | 61 | Tajae Sharpe | 68.53 | 61.24 | 69.22 | 351 | Titans |
| 112 | 62 | Russell Shepard | 68.02 | 60.68 | 68.75 | 134 | Giants |
| 113 | 63 | Isaiah McKenzie | 67.91 | 63.44 | 66.73 | 159 | Bills |
| 114 | 64 | Zay Jones | 67.75 | 62.09 | 67.35 | 595 | Bills |
| 115 | 65 | Michael Crabtree | 67.66 | 62.54 | 66.91 | 539 | Ravens |
| 116 | 66 | Josh Bellamy | 67.62 | 58.93 | 69.25 | 170 | Bears |
| 117 | 67 | J.J. Nelson | 67.52 | 54.91 | 71.76 | 139 | Cardinals |
| 118 | 68 | Bruce Ellington | 67.46 | 64.43 | 65.32 | 188 | Lions |
| 119 | 69 | Vyncint Smith | 67.13 | 59.20 | 68.25 | 129 | Texans |
| 120 | 70 | Zach Pascal | 67.02 | 60.60 | 67.14 | 289 | Colts |
| 121 | 71 | Bennie Fowler | 66.93 | 58.88 | 68.13 | 257 | Giants |
| 122 | 72 | Chester Rogers | 66.83 | 62.62 | 65.47 | 416 | Colts |
| 123 | 73 | Seth Roberts | 66.80 | 61.64 | 66.08 | 401 | Raiders |
| 124 | 74 | Chris Moore | 66.75 | 60.00 | 67.09 | 214 | Ravens |
| 125 | 75 | Justin Hardy | 66.71 | 60.21 | 66.88 | 121 | Falcons |
| 126 | 76 | Maurice Harris | 66.70 | 57.25 | 68.84 | 298 | Commanders |
| 127 | 77 | Austin Carr | 66.61 | 57.53 | 68.50 | 161 | Saints |
| 128 | 78 | Ryan Grant | 66.55 | 59.28 | 67.23 | 422 | Colts |
| 129 | 79 | Demarcus Robinson | 66.51 | 59.11 | 67.28 | 269 | Chiefs |
| 130 | 80 | Alex Erickson | 66.27 | 59.46 | 66.64 | 239 | Bengals |
| 131 | 81 | Trent Sherfield | 65.74 | 58.87 | 66.16 | 233 | Cardinals |
| 132 | 82 | Torrey Smith | 65.58 | 58.80 | 65.93 | 222 | Panthers |
| 133 | 83 | James Washington | 65.42 | 51.57 | 70.48 | 402 | Steelers |
| 134 | 84 | Damion Ratley | 65.39 | 58.19 | 66.03 | 128 | Browns |
| 135 | 85 | Ryan Switzer | 65.36 | 61.81 | 63.56 | 239 | Steelers |
| 136 | 86 | Cody Core | 64.84 | 56.55 | 66.20 | 211 | Bengals |
| 137 | 87 | Trent Taylor | 64.58 | 57.63 | 65.04 | 257 | 49ers |
| 138 | 88 | Chris Conley | 64.33 | 56.15 | 65.61 | 555 | Chiefs |
| 139 | 89 | T.J. Jones | 64.14 | 53.05 | 67.36 | 346 | Lions |
| 140 | 90 | DaeSean Hamilton | 63.93 | 60.07 | 62.33 | 292 | Broncos |
| 141 | 91 | Andre Roberts | 63.93 | 59.19 | 62.92 | 121 | Jets |
| 142 | 92 | Andre Holmes | 63.31 | 56.95 | 63.38 | 182 | Broncos |
| 143 | 93 | Marcell Ateman | 62.65 | 56.15 | 62.81 | 232 | Raiders |
| 144 | 94 | Michael Floyd | 62.44 | 52.58 | 64.84 | 214 | Commanders |
| 145 | 95 | DJ Chark Jr. | 62.26 | 55.84 | 62.38 | 198 | Jaguars |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 146 | 1 | Andy Jones | 61.93 | 60.56 | 58.67 | 142 | Lions |
| 147 | 2 | Chad Williams | 61.63 | 55.76 | 61.38 | 299 | Cardinals |
| 148 | 3 | Darius Jennings | 61.58 | 58.17 | 59.69 | 109 | Titans |
| 149 | 4 | Laquon Treadwell | 60.79 | 53.35 | 61.59 | 386 | Vikings |
| 150 | 5 | John Ross | 59.00 | 51.75 | 59.67 | 400 | Bengals |
