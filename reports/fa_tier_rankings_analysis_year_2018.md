# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:26:51Z
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
| 2 | 2 | Matt Paradis | 89.07 | 81.40 | 90.02 | 569 | Broncos |
| 3 | 3 | Rodney Hudson | 88.43 | 81.90 | 88.62 | 1042 | Raiders |
| 4 | 4 | Alex Mack | 88.13 | 80.00 | 89.38 | 1057 | Falcons |
| 5 | 5 | Brandon Linder | 86.70 | 79.60 | 87.27 | 507 | Jaguars |
| 6 | 6 | Austin Reiter | 86.16 | 78.60 | 87.03 | 266 | Chiefs |
| 7 | 7 | Corey Linsley | 85.58 | 78.60 | 86.06 | 1074 | Packers |
| 8 | 8 | J.C. Tretter | 84.29 | 76.90 | 85.05 | 1091 | Browns |
| 9 | 9 | Ryan Kelly | 83.47 | 74.50 | 85.28 | 777 | Colts |
| 10 | 10 | David Andrews | 82.23 | 73.30 | 84.02 | 1104 | Patriots |
| 11 | 11 | Ben Jones | 81.19 | 72.70 | 82.69 | 986 | Titans |
| 12 | 12 | Graham Glasgow | 80.84 | 71.10 | 83.16 | 1076 | Lions |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Mitch Morse | 79.95 | 70.30 | 82.22 | 678 | Chiefs |
| 14 | 2 | Maurkice Pouncey | 79.57 | 71.30 | 80.91 | 1101 | Steelers |
| 15 | 3 | Cody Whitehair | 78.05 | 75.20 | 75.78 | 1075 | Bears |
| 16 | 4 | Jon Halapio | 76.86 | 76.40 | 73.00 | 116 | Giants |
| 17 | 5 | Ryan Kalil | 76.35 | 66.60 | 78.69 | 1028 | Panthers |
| 18 | 6 | Travis Swanson | 75.66 | 65.80 | 78.06 | 644 | Dolphins |
| 19 | 7 | Nick Martin | 75.23 | 65.40 | 77.62 | 1094 | Texans |
| 20 | 8 | Chase Roullier | 75.18 | 65.00 | 77.80 | 1020 | Commanders |
| 21 | 9 | Max Unger | 74.65 | 64.50 | 77.25 | 1012 | Saints |

### Starter (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Justin Britt | 73.96 | 62.00 | 77.76 | 989 | Seahawks |
| 23 | 2 | Russell Bodine | 73.64 | 62.30 | 77.03 | 588 | Bills |
| 24 | 3 | John Greco | 73.10 | 60.60 | 77.27 | 488 | Giants |
| 25 | 4 | Mike Pouncey | 71.69 | 59.10 | 75.91 | 954 | Chargers |
| 26 | 5 | Matt Skura | 71.03 | 59.20 | 74.75 | 1188 | Ravens |
| 27 | 6 | Joey Hunt | 70.83 | 58.30 | 75.01 | 115 | Seahawks |
| 28 | 7 | Jonotthan Harrison | 70.70 | 56.70 | 75.87 | 506 | Jets |
| 29 | 8 | Billy Price | 70.19 | 55.60 | 75.75 | 558 | Bengals |
| 30 | 9 | Spencer Pulley | 69.91 | 58.30 | 73.49 | 567 | Giants |
| 31 | 10 | Ryan Jensen | 69.77 | 56.60 | 74.39 | 1116 | Buccaneers |
| 32 | 11 | Weston Richburg | 69.17 | 56.60 | 73.38 | 968 | 49ers |
| 33 | 12 | Jake Brendel | 68.76 | 58.80 | 71.24 | 176 | Dolphins |
| 34 | 13 | Trey Hopkins | 68.33 | 60.00 | 69.72 | 589 | Bengals |
| 35 | 14 | Mason Cole | 67.87 | 53.60 | 73.22 | 942 | Cardinals |
| 36 | 15 | Daniel Kilgore | 67.57 | 54.70 | 71.98 | 182 | Dolphins |
| 37 | 16 | Joe Looney | 67.44 | 55.10 | 71.50 | 1076 | Cowboys |
| 38 | 17 | John Sullivan | 67.12 | 53.80 | 71.84 | 1054 | Rams |
| 39 | 18 | Brett Jones | 66.98 | 53.10 | 72.06 | 191 | Vikings |
| 40 | 19 | Spencer Long | 62.29 | 47.10 | 68.25 | 805 | Jets |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 41 | 1 | Pat Elflein | 59.93 | 41.90 | 67.78 | 863 | Vikings |

## CB — Cornerback

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Stephon Gilmore | 93.18 | 90.90 | 90.73 | 1013 | Patriots |
| 2 | 2 | Kyle Fuller | 87.87 | 84.10 | 86.21 | 1014 | Bears |
| 3 | 3 | Byron Jones | 87.28 | 83.30 | 85.76 | 1021 | Cowboys |
| 4 | 4 | Chris Harris Jr. | 86.45 | 84.40 | 85.74 | 747 | Broncos |
| 5 | 5 | Patrick Peterson | 85.95 | 83.70 | 83.28 | 1107 | Cardinals |
| 6 | 6 | Casey Hayward Jr. | 85.19 | 77.70 | 86.01 | 1016 | Chargers |
| 7 | 7 | Nickell Robey-Coleman | 84.62 | 81.30 | 82.67 | 556 | Rams |
| 8 | 8 | Jason McCourty | 84.14 | 80.00 | 83.76 | 835 | Patriots |
| 9 | 9 | Darius Slay | 83.78 | 78.20 | 84.27 | 875 | Lions |
| 10 | 10 | Marlon Humphrey | 83.41 | 80.10 | 82.75 | 718 | Ravens |
| 11 | 11 | A.J. Bouye | 83.40 | 79.30 | 83.54 | 827 | Jaguars |
| 12 | 12 | Denzel Ward | 83.33 | 83.60 | 82.12 | 841 | Browns |
| 13 | 13 | J.C. Jackson | 83.16 | 78.30 | 85.37 | 395 | Patriots |
| 14 | 14 | Sherrick McManis | 83.10 | 85.80 | 83.18 | 237 | Bears |
| 15 | 15 | Bryce Callahan | 82.62 | 81.30 | 83.19 | 676 | Bears |
| 16 | 16 | Johnathan Joseph | 82.55 | 80.70 | 80.87 | 811 | Texans |
| 17 | 17 | Levi Wallace | 82.19 | 85.30 | 86.81 | 416 | Bills |
| 18 | 18 | Aqib Talib | 81.15 | 76.90 | 84.91 | 388 | Rams |
| 19 | 19 | Holton Hill | 80.93 | 76.50 | 81.80 | 376 | Vikings |
| 20 | 20 | Marshon Lattimore | 80.37 | 71.70 | 82.36 | 907 | Saints |
| 21 | 21 | Prince Amukamara | 80.32 | 76.20 | 80.46 | 911 | Bears |
| 22 | 22 | Justin Coleman | 80.20 | 77.80 | 79.81 | 672 | Seahawks |
| 23 | 23 | William Jackson III | 80.02 | 72.90 | 81.39 | 1063 | Bengals |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Jalen Ramsey | 79.91 | 71.90 | 81.08 | 1019 | Jaguars |
| 25 | 2 | Xavien Howard | 79.86 | 75.30 | 82.48 | 803 | Dolphins |
| 26 | 3 | Trumaine Johnson | 79.23 | 75.10 | 81.36 | 670 | Jets |
| 27 | 4 | Pierre Desir | 79.20 | 73.10 | 82.95 | 903 | Colts |
| 28 | 5 | Mike Hilton | 78.78 | 72.00 | 79.79 | 594 | Steelers |
| 29 | 6 | Josh Norman | 78.20 | 72.00 | 78.80 | 1028 | Commanders |
| 30 | 7 | Malcolm Butler | 77.93 | 69.80 | 79.18 | 836 | Titans |
| 31 | 8 | Logan Ryan | 77.82 | 70.10 | 79.84 | 855 | Titans |
| 32 | 9 | Steven Nelson | 77.77 | 73.60 | 78.26 | 1164 | Chiefs |
| 33 | 10 | D.J. Hayden | 77.74 | 75.90 | 78.97 | 456 | Jaguars |
| 34 | 11 | Mackensie Alexander | 77.68 | 72.50 | 78.95 | 564 | Vikings |
| 35 | 12 | Jourdan Lewis | 77.20 | 74.70 | 77.70 | 187 | Cowboys |
| 36 | 13 | Adoree' Jackson | 77.06 | 69.00 | 78.27 | 959 | Titans |
| 37 | 14 | Isaiah Oliver | 76.63 | 69.60 | 82.35 | 241 | Falcons |
| 38 | 15 | Brandon Carr | 76.60 | 69.10 | 77.43 | 876 | Ravens |
| 39 | 16 | Joe Haden | 76.56 | 70.90 | 78.56 | 937 | Steelers |
| 40 | 17 | Desmond Trufant | 76.13 | 67.20 | 79.38 | 1061 | Falcons |
| 41 | 18 | Richard Sherman | 75.99 | 68.10 | 80.31 | 836 | 49ers |
| 42 | 19 | Jaire Alexander | 75.90 | 73.00 | 76.80 | 761 | Packers |
| 43 | 20 | Kendall Fuller | 75.55 | 70.20 | 76.30 | 1078 | Chiefs |
| 44 | 21 | Ronald Darby | 74.13 | 70.40 | 78.09 | 542 | Eagles |
| 45 | 22 | Donte Jackson | 74.09 | 67.20 | 74.52 | 895 | Panthers |
| 46 | 23 | Quinton Dunbar | 74.05 | 67.60 | 80.34 | 373 | Commanders |

### Starter (67 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 47 | 1 | Kenny Moore II | 73.95 | 68.40 | 77.27 | 911 | Colts |
| 48 | 2 | Orlando Scandrick | 73.81 | 70.10 | 74.82 | 788 | Chiefs |
| 49 | 3 | Tre'Davious White | 73.46 | 62.50 | 76.60 | 961 | Bills |
| 50 | 4 | Trae Waynes | 73.36 | 64.60 | 76.29 | 693 | Vikings |
| 51 | 5 | Janoris Jenkins | 73.36 | 66.30 | 76.08 | 1088 | Giants |
| 52 | 6 | Rasul Douglas | 73.10 | 67.90 | 75.27 | 544 | Eagles |
| 53 | 7 | Josh Jackson | 72.67 | 62.00 | 75.62 | 721 | Packers |
| 54 | 8 | Chidobe Awuzie | 72.43 | 66.10 | 75.87 | 886 | Cowboys |
| 55 | 9 | T.J. Carrie | 72.37 | 63.90 | 74.05 | 908 | Browns |
| 56 | 10 | Jimmy Smith | 72.13 | 66.00 | 76.41 | 610 | Ravens |
| 57 | 11 | Anthony Brown | 72.12 | 62.90 | 74.62 | 690 | Cowboys |
| 58 | 12 | James Bradberry | 72.08 | 64.90 | 73.33 | 994 | Panthers |
| 59 | 13 | K'Waun Williams | 71.91 | 66.00 | 73.96 | 595 | 49ers |
| 60 | 14 | Brent Grimes | 71.25 | 61.00 | 76.42 | 791 | Buccaneers |
| 61 | 15 | Briean Boddy-Calhoun | 70.83 | 63.00 | 74.28 | 656 | Browns |
| 62 | 16 | Terrance Mitchell | 70.74 | 64.30 | 77.74 | 445 | Browns |
| 63 | 17 | Gareon Conley | 70.72 | 61.50 | 79.46 | 679 | Raiders |
| 64 | 18 | Marcus Peters | 70.67 | 58.10 | 75.20 | 914 | Rams |
| 65 | 19 | Taron Johnson | 70.67 | 69.60 | 72.42 | 405 | Bills |
| 66 | 20 | Tavon Young | 70.22 | 62.60 | 71.65 | 602 | Ravens |
| 67 | 21 | Dre Kirkpatrick | 69.76 | 62.60 | 72.76 | 774 | Bengals |
| 68 | 22 | Patrick Robinson | 69.74 | 66.10 | 76.65 | 110 | Saints |
| 69 | 23 | Michael Davis | 69.74 | 65.70 | 74.51 | 627 | Chargers |
| 70 | 24 | Cre'Von LeBlanc | 69.69 | 63.90 | 75.64 | 382 | Eagles |
| 71 | 25 | Captain Munnerlyn | 69.46 | 60.30 | 71.91 | 630 | Panthers |
| 72 | 26 | Jonathan Jones | 69.43 | 61.00 | 73.28 | 516 | Patriots |
| 73 | 27 | Coty Sensabaugh | 69.39 | 64.70 | 73.25 | 745 | Steelers |
| 74 | 28 | Bradley Roby | 69.06 | 58.60 | 72.39 | 926 | Broncos |
| 75 | 29 | Greg Stroman Jr. | 68.99 | 68.60 | 68.22 | 387 | Commanders |
| 76 | 30 | B.W. Webb | 68.88 | 60.40 | 72.65 | 1004 | Giants |
| 77 | 31 | Darqueze Dennard | 68.55 | 60.70 | 71.38 | 675 | Bengals |
| 78 | 32 | Eli Apple | 68.55 | 62.30 | 71.15 | 905 | Saints |
| 79 | 33 | Morris Claiborne | 68.25 | 60.30 | 71.88 | 1002 | Jets |
| 80 | 34 | Javien Elliott | 68.00 | 68.00 | 70.60 | 351 | Buccaneers |
| 81 | 35 | Ryan Smith | 67.96 | 61.80 | 71.24 | 419 | Buccaneers |
| 82 | 36 | Quincy Wilson | 67.58 | 62.90 | 72.00 | 435 | Colts |
| 83 | 37 | Xavier Rhodes | 67.57 | 55.10 | 73.18 | 771 | Vikings |
| 84 | 38 | Tramaine Brock Sr. | 67.57 | 60.50 | 73.31 | 436 | Broncos |
| 85 | 39 | Avonte Maddox | 67.26 | 60.70 | 73.71 | 541 | Eagles |
| 86 | 40 | Grant Haley | 66.72 | 70.50 | 66.29 | 429 | Giants |
| 87 | 41 | Robert Alford | 66.70 | 53.40 | 71.92 | 959 | Falcons |
| 88 | 42 | Rashaan Melvin | 66.69 | 56.30 | 72.78 | 604 | Raiders |
| 89 | 43 | Artie Burns | 66.47 | 57.40 | 72.00 | 308 | Steelers |
| 90 | 44 | Troy Hill | 66.45 | 58.20 | 72.37 | 427 | Rams |
| 91 | 45 | E.J. Gaines | 66.16 | 62.50 | 72.96 | 181 | Browns |
| 92 | 46 | Trevor Williams | 65.99 | 58.40 | 72.51 | 410 | Chargers |
| 93 | 47 | Bashaud Breeland | 65.77 | 57.20 | 72.73 | 330 | Packers |
| 94 | 48 | Eric Rowe | 65.35 | 58.50 | 74.40 | 136 | Patriots |
| 95 | 49 | Shaquill Griffin | 65.13 | 51.90 | 70.17 | 941 | Seahawks |
| 96 | 50 | Tre Flowers | 65.11 | 56.50 | 67.72 | 903 | Seahawks |
| 97 | 51 | Nevin Lawson | 64.99 | 54.70 | 68.52 | 877 | Lions |
| 98 | 52 | David Amerson | 64.86 | 58.20 | 73.47 | 293 | Cardinals |
| 99 | 53 | Shareece Wright | 64.85 | 58.50 | 69.09 | 506 | Texans |
| 100 | 54 | Mike Hughes | 64.58 | 64.60 | 73.81 | 244 | Vikings |
| 101 | 55 | Sam Shields | 64.41 | 57.10 | 72.51 | 340 | Rams |
| 102 | 56 | Leon Hall | 64.38 | 57.10 | 71.00 | 366 | Raiders |
| 103 | 57 | Jalen Mills | 63.73 | 55.80 | 73.19 | 457 | Eagles |
| 104 | 58 | Carlton Davis III | 63.71 | 59.00 | 65.81 | 1436 | Buccaneers |
| 105 | 59 | Phillip Gaines | 63.47 | 56.80 | 70.32 | 408 | Browns |
| 106 | 60 | Bobby McCain | 63.36 | 54.00 | 66.46 | 823 | Dolphins |
| 107 | 61 | Kevin King | 63.18 | 59.90 | 70.45 | 305 | Packers |
| 108 | 62 | P.J. Williams | 63.05 | 50.60 | 70.62 | 693 | Saints |
| 109 | 63 | Daryl Worley | 62.15 | 51.10 | 68.69 | 505 | Raiders |
| 110 | 64 | Charvarius Ward | 62.15 | 59.40 | 78.32 | 140 | Chiefs |
| 111 | 65 | Tyler Patmon | 62.12 | 58.00 | 68.83 | 231 | Jaguars |
| 112 | 66 | LeShaun Sims | 62.07 | 56.80 | 65.79 | 215 | Titans |
| 113 | 67 | Buster Skrine | 62.00 | 50.80 | 67.06 | 693 | Jets |

### Rotation/backup (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 114 | 1 | Bene Benwikere | 61.95 | 55.40 | 69.23 | 592 | Raiders |
| 115 | 2 | Ken Crawley | 61.79 | 48.90 | 70.29 | 409 | Saints |
| 116 | 3 | Fabian Moreau | 61.48 | 58.30 | 63.34 | 840 | Commanders |
| 117 | 4 | Isaac Yiadom | 60.97 | 55.60 | 65.59 | 264 | Broncos |
| 118 | 5 | Kevin Toliver II | 60.59 | 54.80 | 66.53 | 136 | Bears |
| 119 | 6 | Ryan Lewis | 58.87 | 58.50 | 70.91 | 150 | Bills |
| 120 | 7 | Aaron Colvin | 58.77 | 53.30 | 63.67 | 317 | Texans |
| 121 | 8 | Nate Hairston | 58.02 | 48.40 | 64.95 | 413 | Colts |
| 122 | 9 | Tony Brown | 57.92 | 48.70 | 66.15 | 290 | Packers |
| 123 | 10 | Darius Phillips | 56.39 | 54.20 | 58.88 | 232 | Bengals |
| 124 | 11 | KeiVarae Russell | 56.28 | 55.30 | 64.01 | 137 | Bengals |
| 125 | 12 | Cameron Sutton | 55.61 | 55.90 | 59.20 | 240 | Steelers |
| 126 | 13 | M.J. Stewart | 55.42 | 56.10 | 59.13 | 300 | Buccaneers |
| 127 | 14 | Greg Mabin | 55.36 | 48.90 | 65.00 | 163 | 49ers |
| 128 | 15 | Ahkello Witherspoon | 54.03 | 39.80 | 62.86 | 700 | 49ers |
| 129 | 16 | Torry McTyer | 52.66 | 48.20 | 58.50 | 347 | Dolphins |
| 130 | 17 | Sidney Jones IV | 51.95 | 47.60 | 61.10 | 321 | Eagles |
| 131 | 18 | Jamar Taylor | 50.71 | 39.40 | 56.69 | 305 | Broncos |
| 132 | 19 | Jamal Agnew | 50.34 | 45.60 | 61.56 | 117 | Lions |
| 133 | 20 | Parry Nickerson | 47.66 | 44.00 | 54.27 | 213 | Jets |
| 134 | 21 | Nick Nelson | 45.00 | 34.80 | 48.84 | 311 | Raiders |

## DI — Defensive Interior

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 94.35 | 90.15 | 93.30 | 913 | Rams |
| 2 | 2 | Kawann Short | 88.45 | 86.58 | 86.56 | 584 | Panthers |
| 3 | 3 | Geno Atkins | 87.70 | 85.30 | 85.13 | 795 | Bengals |
| 4 | 4 | Jurrell Casey | 86.54 | 87.73 | 82.32 | 745 | Titans |
| 5 | 5 | Leonard Williams | 86.00 | 89.11 | 79.76 | 866 | Jets |
| 6 | 6 | Damon Harrison Sr. | 85.73 | 84.44 | 82.42 | 606 | Lions |
| 7 | 7 | Fletcher Cox | 85.72 | 89.24 | 79.21 | 831 | Eagles |
| 8 | 8 | DeForest Buckner | 85.51 | 89.65 | 78.78 | 852 | 49ers |
| 9 | 9 | Bilal Nichols | 85.18 | 76.93 | 88.60 | 329 | Bears |
| 10 | 10 | Akiem Hicks | 84.83 | 82.94 | 81.93 | 781 | Bears |
| 11 | 11 | Kenny Clark | 84.29 | 87.24 | 80.04 | 720 | Packers |
| 12 | 12 | Ndamukong Suh | 84.29 | 84.83 | 79.76 | 887 | Rams |
| 13 | 13 | Grady Jarrett | 84.17 | 83.33 | 81.60 | 712 | Falcons |
| 14 | 14 | Vincent Taylor | 83.69 | 82.83 | 86.87 | 204 | Dolphins |
| 15 | 15 | Javon Hargrave | 83.61 | 83.88 | 79.26 | 455 | Steelers |
| 16 | 16 | Michael Pierce | 83.42 | 82.58 | 80.85 | 388 | Ravens |
| 17 | 17 | Shelby Harris | 83.19 | 82.64 | 81.28 | 391 | Broncos |
| 18 | 18 | Cameron Heyward | 82.50 | 83.82 | 79.34 | 842 | Steelers |
| 19 | 19 | Marcell Dareus | 81.95 | 82.90 | 79.33 | 563 | Jaguars |
| 20 | 20 | Da'Shawn Hand | 81.25 | 84.53 | 78.03 | 455 | Lions |
| 21 | 21 | Taven Bryan | 81.23 | 80.39 | 77.63 | 301 | Jaguars |
| 22 | 22 | Malik Jackson | 81.21 | 74.56 | 81.47 | 628 | Jaguars |
| 23 | 23 | Jonathan Allen | 80.95 | 82.78 | 79.86 | 780 | Commanders |
| 24 | 24 | Linval Joseph | 80.79 | 80.08 | 77.61 | 671 | Vikings |
| 25 | 25 | Gerald McCoy | 80.27 | 84.74 | 74.69 | 732 | Buccaneers |
| 26 | 26 | Eddie Goldman | 80.21 | 82.07 | 77.21 | 552 | Bears |
| 27 | 27 | Henry Anderson | 80.04 | 80.46 | 78.82 | 668 | Jets |

### Good (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Stephon Tuitt | 79.95 | 84.21 | 74.93 | 694 | Steelers |
| 29 | 2 | Sheldon Richardson | 79.92 | 75.89 | 79.79 | 719 | Vikings |
| 30 | 3 | Mike Daniels | 79.87 | 75.51 | 82.36 | 419 | Packers |
| 31 | 4 | B.J. Hill | 79.29 | 73.89 | 78.72 | 642 | Giants |
| 32 | 5 | Poona Ford | 79.22 | 80.31 | 79.53 | 231 | Seahawks |
| 33 | 6 | Sheldon Rankins | 78.67 | 81.02 | 74.41 | 642 | Saints |
| 34 | 7 | Vita Vea | 78.62 | 82.75 | 74.84 | 493 | Buccaneers |
| 35 | 8 | A'Shawn Robinson | 78.09 | 78.95 | 74.91 | 415 | Lions |
| 36 | 9 | Denico Autry | 77.32 | 73.41 | 77.85 | 555 | Colts |
| 37 | 10 | Muhammad Wilkerson | 77.28 | 75.14 | 82.24 | 115 | Packers |
| 38 | 11 | Mike Pennel | 77.20 | 74.30 | 74.96 | 358 | Jets |
| 39 | 12 | Chris Jones | 77.20 | 73.59 | 76.07 | 773 | Chiefs |
| 40 | 13 | Dalvin Tomlinson | 77.07 | 75.71 | 73.81 | 628 | Giants |
| 41 | 14 | Dean Lowry | 77.01 | 72.64 | 75.76 | 698 | Packers |
| 42 | 15 | Lawrence Guy Sr. | 76.93 | 69.19 | 77.92 | 519 | Patriots |
| 43 | 16 | Christian Covington | 76.91 | 80.71 | 75.11 | 257 | Texans |
| 44 | 17 | Larry Ogunjobi | 76.65 | 67.47 | 79.38 | 930 | Browns |
| 45 | 18 | Brandon Williams | 76.32 | 70.97 | 76.97 | 518 | Ravens |
| 46 | 19 | Danny Shelton | 76.29 | 77.80 | 73.30 | 323 | Patriots |
| 47 | 20 | Jarran Reed | 75.59 | 67.28 | 77.28 | 773 | Seahawks |
| 48 | 21 | Michael Brockers | 75.50 | 72.78 | 73.56 | 679 | Rams |
| 49 | 22 | David Onyemata | 75.48 | 76.56 | 70.59 | 618 | Saints |
| 50 | 23 | DJ Reader | 75.42 | 75.55 | 71.80 | 638 | Texans |
| 51 | 24 | Steve McLendon | 75.40 | 66.44 | 78.24 | 471 | Jets |
| 52 | 25 | Daron Payne | 75.07 | 68.83 | 75.07 | 797 | Commanders |
| 53 | 26 | Derek Wolfe | 75.07 | 70.26 | 76.10 | 710 | Broncos |
| 54 | 27 | Roy Robertson-Harris | 74.15 | 65.86 | 77.08 | 354 | Bears |

### Starter (65 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 55 | 1 | Tyler Lancaster | 73.88 | 73.50 | 75.16 | 272 | Packers |
| 56 | 2 | Mario Edwards Jr. | 73.58 | 69.26 | 76.15 | 232 | Giants |
| 57 | 3 | Maurice Hurst | 73.54 | 74.47 | 71.89 | 472 | Raiders |
| 58 | 4 | Johnathan Hankins | 73.53 | 65.57 | 75.50 | 573 | Raiders |
| 59 | 5 | Fadol Brown | 73.51 | 60.85 | 81.95 | 215 | Packers |
| 60 | 6 | Kyle Williams | 73.21 | 59.20 | 78.59 | 657 | Bills |
| 61 | 7 | Xavier Williams | 73.18 | 66.78 | 76.52 | 424 | Chiefs |
| 62 | 8 | Abry Jones | 73.10 | 63.54 | 76.04 | 498 | Jaguars |
| 63 | 9 | Rodney Gunter | 73.04 | 63.09 | 75.51 | 641 | Cardinals |
| 64 | 10 | Adam Gotsis | 72.87 | 64.88 | 74.03 | 513 | Broncos |
| 65 | 11 | Malcom Brown | 72.76 | 62.42 | 76.00 | 456 | Patriots |
| 66 | 12 | Treyvon Hester | 72.24 | 77.15 | 68.18 | 226 | Eagles |
| 67 | 13 | Zach Kerr | 71.92 | 62.44 | 76.48 | 394 | Broncos |
| 68 | 14 | Corey Liuget | 71.41 | 68.98 | 75.32 | 206 | Chargers |
| 69 | 15 | Matt Ioannidis | 71.31 | 69.02 | 71.81 | 439 | Commanders |
| 70 | 16 | Bennie Logan | 71.21 | 53.90 | 79.74 | 230 | Titans |
| 71 | 17 | Dan McCullers | 71.04 | 66.51 | 74.68 | 111 | Steelers |
| 72 | 18 | Margus Hunt | 70.85 | 59.62 | 74.82 | 724 | Colts |
| 73 | 19 | Andrew Billings | 70.15 | 60.57 | 72.75 | 632 | Bengals |
| 74 | 20 | Tyrone Crawford | 70.13 | 59.96 | 73.26 | 633 | Cowboys |
| 75 | 21 | Olsen Pierre | 70.09 | 48.31 | 85.12 | 246 | Cardinals |
| 76 | 22 | Davon Godchaux | 70.08 | 61.45 | 72.05 | 675 | Dolphins |
| 77 | 23 | DaQuan Jones | 69.82 | 65.63 | 69.69 | 587 | Titans |
| 78 | 24 | Darius Philon | 69.70 | 60.28 | 72.23 | 607 | Chargers |
| 79 | 25 | Josh Mauro | 68.63 | 58.48 | 76.33 | 270 | Giants |
| 80 | 26 | Dontari Poe | 67.84 | 63.05 | 66.87 | 515 | Panthers |
| 81 | 27 | Ricky Jean Francois | 67.68 | 53.43 | 73.33 | 405 | Lions |
| 82 | 28 | Brent Urban | 67.67 | 62.32 | 71.14 | 523 | Ravens |
| 83 | 29 | Adolphus Washington | 67.64 | 62.32 | 72.95 | 134 | Bengals |
| 84 | 30 | Tim Settle | 67.54 | 56.38 | 72.90 | 135 | Commanders |
| 85 | 31 | Sheldon Day | 67.29 | 56.25 | 73.82 | 275 | 49ers |
| 86 | 32 | Deadrin Senat | 67.17 | 62.50 | 67.15 | 370 | Falcons |
| 87 | 33 | Domata Peko Sr. | 67.16 | 56.81 | 70.53 | 523 | Broncos |
| 88 | 34 | Jonathan Bullard | 67.14 | 55.77 | 70.97 | 298 | Bears |
| 89 | 35 | Robert Nkemdiche | 66.92 | 56.45 | 76.40 | 426 | Cardinals |
| 90 | 36 | Clinton McDonald | 66.63 | 49.94 | 75.57 | 419 | Raiders |
| 91 | 37 | Allen Bailey | 66.39 | 56.24 | 71.59 | 847 | Chiefs |
| 92 | 38 | Star Lotulelei | 66.15 | 52.63 | 70.99 | 476 | Bills |
| 93 | 39 | Corey Peters | 66.11 | 55.43 | 71.04 | 735 | Cardinals |
| 94 | 40 | Vernon Butler | 66.04 | 60.98 | 67.85 | 329 | Panthers |
| 95 | 41 | Al Woods | 65.74 | 52.96 | 71.98 | 375 | Colts |
| 96 | 42 | Beau Allen | 65.74 | 52.51 | 71.42 | 386 | Buccaneers |
| 97 | 43 | Caraun Reid | 64.94 | 62.83 | 71.86 | 185 | Cowboys |
| 98 | 44 | Ethan Westbrooks | 64.91 | 48.95 | 73.47 | 180 | Rams |
| 99 | 45 | Haloti Ngata | 64.88 | 54.75 | 72.88 | 368 | Eagles |
| 100 | 46 | Akeem Spence | 64.68 | 49.52 | 70.62 | 665 | Dolphins |
| 101 | 47 | Tyson Alualu | 64.60 | 53.37 | 68.86 | 311 | Steelers |
| 102 | 48 | Stacy McGee | 64.54 | 56.44 | 71.19 | 137 | Commanders |
| 103 | 49 | Derrick Nnadi | 64.48 | 60.68 | 62.84 | 448 | Chiefs |
| 104 | 50 | Jack Crawford | 64.46 | 52.88 | 71.77 | 624 | Falcons |
| 105 | 51 | Harrison Phillips | 64.43 | 57.09 | 65.15 | 389 | Bills |
| 106 | 52 | Brandon Dunn | 64.34 | 58.00 | 66.06 | 347 | Texans |
| 107 | 53 | Jordan Phillips | 64.13 | 52.79 | 68.45 | 393 | Bills |
| 108 | 54 | Adam Butler | 64.08 | 49.68 | 69.51 | 380 | Patriots |
| 109 | 55 | Quinton Jefferson | 64.04 | 56.51 | 70.72 | 558 | Seahawks |
| 110 | 56 | Tom Johnson | 63.93 | 44.72 | 74.03 | 381 | Vikings |
| 111 | 57 | Austin Johnson | 63.88 | 57.06 | 65.51 | 399 | Titans |
| 112 | 58 | Montravius Adams | 63.60 | 56.90 | 67.41 | 213 | Packers |
| 113 | 59 | Brandon Mebane | 63.57 | 49.31 | 72.24 | 405 | Chargers |
| 114 | 60 | Earl Mitchell | 63.39 | 49.44 | 71.34 | 363 | 49ers |
| 115 | 61 | Sylvester Williams | 63.35 | 52.22 | 67.64 | 376 | Dolphins |
| 116 | 62 | Daniel Ross | 62.57 | 52.15 | 72.39 | 250 | Cowboys |
| 117 | 63 | Christian Ringo | 62.42 | 57.34 | 71.44 | 152 | Bengals |
| 118 | 64 | D.J. Jones | 62.22 | 53.48 | 70.52 | 239 | 49ers |
| 119 | 65 | Tyeler Davison | 62.04 | 57.41 | 62.21 | 422 | Saints |

### Rotation/backup (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 120 | 1 | Maliek Collins | 61.99 | 52.99 | 65.39 | 498 | Cowboys |
| 121 | 2 | Justin Ellis | 61.93 | 55.22 | 67.43 | 133 | Raiders |
| 122 | 3 | P.J. Hall | 61.83 | 60.39 | 60.71 | 511 | Raiders |
| 123 | 4 | Kyle Love | 61.80 | 48.37 | 67.84 | 467 | Panthers |
| 124 | 5 | Nazair Jones | 61.79 | 55.10 | 68.60 | 132 | Seahawks |
| 125 | 6 | Chris Wormley | 61.73 | 58.71 | 63.10 | 401 | Ravens |
| 126 | 7 | Antwaun Woods | 61.40 | 56.54 | 66.99 | 585 | Cowboys |
| 127 | 8 | Jihad Ward | 60.87 | 63.76 | 63.43 | 144 | Colts |
| 128 | 9 | Grover Stewart | 60.67 | 52.13 | 63.23 | 292 | Colts |
| 129 | 10 | Trevon Coley | 60.58 | 47.28 | 65.66 | 614 | Browns |
| 130 | 11 | Angelo Blackson | 59.96 | 52.78 | 63.40 | 430 | Texans |
| 131 | 12 | Shamar Stephen | 59.62 | 51.21 | 61.58 | 494 | Seahawks |
| 132 | 13 | Damion Square | 59.34 | 47.33 | 64.21 | 530 | Chargers |
| 133 | 14 | Nathan Shepherd | 59.12 | 61.36 | 53.46 | 343 | Jets |
| 134 | 15 | Brian Price | 58.86 | 47.69 | 66.40 | 210 | Browns |
| 135 | 16 | Terrell McClain | 58.70 | 43.18 | 67.70 | 374 | Falcons |
| 136 | 17 | Josh Tupou | 58.42 | 64.37 | 63.70 | 154 | Bengals |
| 137 | 18 | Jaleel Johnson | 56.63 | 49.29 | 60.88 | 261 | Vikings |
| 138 | 19 | Taylor Stallworth | 56.17 | 49.87 | 58.29 | 318 | Saints |
| 139 | 20 | Justin Jones | 55.28 | 46.01 | 58.32 | 300 | Chargers |
| 140 | 21 | Destiny Vaeao | 53.81 | 43.83 | 62.75 | 157 | Jets |
| 141 | 22 | William Gholston | 52.83 | 39.00 | 57.88 | 402 | Buccaneers |
| 142 | 23 | Darius Kilgo | 51.94 | 50.49 | 54.72 | 131 | Titans |

## ED — Edge

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 93.49 | 93.77 | 89.13 | 844 | Broncos |
| 2 | 2 | Khalil Mack | 91.32 | 96.15 | 84.97 | 756 | Bears |
| 3 | 3 | Joey Bosa | 89.56 | 95.14 | 87.19 | 314 | Chargers |
| 4 | 4 | DeMarcus Lawrence | 88.65 | 96.46 | 80.52 | 736 | Cowboys |
| 5 | 5 | Myles Garrett | 87.12 | 95.00 | 79.65 | 1012 | Browns |
| 6 | 6 | Justin Houston | 86.73 | 84.88 | 87.96 | 719 | Chiefs |
| 7 | 7 | T.J. Watt | 86.01 | 84.66 | 82.74 | 903 | Steelers |
| 8 | 8 | Danielle Hunter | 85.90 | 84.74 | 82.50 | 879 | Vikings |
| 9 | 9 | Brandon Graham | 85.54 | 88.39 | 79.47 | 755 | Eagles |
| 10 | 10 | Bradley Chubb | 84.07 | 78.97 | 83.31 | 844 | Broncos |
| 11 | 11 | Cameron Jordan | 83.25 | 91.91 | 73.31 | 884 | Saints |
| 12 | 12 | Cameron Wake | 82.76 | 72.66 | 86.36 | 517 | Dolphins |
| 13 | 13 | Ezekiel Ansah | 82.63 | 83.92 | 83.34 | 146 | Lions |
| 14 | 14 | Trey Flowers | 81.78 | 83.34 | 77.09 | 732 | Patriots |
| 15 | 15 | Jerry Hughes | 81.36 | 84.78 | 74.92 | 668 | Bills |
| 16 | 16 | Jadeveon Clowney | 81.02 | 94.17 | 68.60 | 902 | Texans |
| 17 | 17 | Ryan Kerrigan | 80.75 | 70.89 | 83.16 | 820 | Commanders |
| 18 | 18 | Frank Clark | 80.44 | 74.50 | 80.24 | 728 | Seahawks |
| 19 | 19 | J.J. Watt | 80.39 | 71.17 | 82.37 | 963 | Texans |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Michael Bennett | 79.57 | 79.74 | 75.92 | 716 | Eagles |
| 21 | 2 | Carlos Dunlap | 79.28 | 77.91 | 76.02 | 839 | Bengals |
| 22 | 3 | Melvin Ingram III | 78.97 | 74.36 | 77.87 | 915 | Chargers |
| 23 | 4 | Olivier Vernon | 78.75 | 85.36 | 74.03 | 665 | Giants |
| 24 | 5 | Chandler Jones | 78.30 | 74.86 | 76.43 | 969 | Cardinals |
| 25 | 6 | Calais Campbell | 78.05 | 62.91 | 83.98 | 816 | Jaguars |
| 26 | 7 | Takk McKinley | 77.56 | 71.85 | 77.85 | 619 | Falcons |
| 27 | 8 | Robert Quinn | 77.55 | 74.61 | 76.81 | 635 | Dolphins |
| 28 | 9 | Shaquil Barrett | 77.13 | 76.12 | 75.21 | 276 | Broncos |
| 29 | 10 | Jabaal Sheard | 77.08 | 79.97 | 70.99 | 814 | Colts |
| 30 | 11 | Yannick Ngakoue | 76.30 | 69.07 | 76.96 | 766 | Jaguars |
| 31 | 12 | Dee Ford | 76.02 | 71.68 | 77.88 | 1022 | Chiefs |
| 32 | 13 | Pernell McPhee | 75.80 | 67.21 | 81.32 | 203 | Commanders |
| 33 | 14 | Marcus Davenport | 74.63 | 77.76 | 71.51 | 416 | Saints |
| 34 | 15 | Mario Addison | 74.45 | 61.96 | 79.03 | 666 | Panthers |
| 35 | 16 | Whitney Mercilus | 74.31 | 66.00 | 79.12 | 785 | Texans |
| 36 | 17 | Derek Barnett | 74.05 | 79.93 | 72.48 | 234 | Eagles |

### Starter (53 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 37 | 1 | Jacob Martin | 73.90 | 65.99 | 75.01 | 225 | Seahawks |
| 38 | 2 | Terrell Suggs | 73.66 | 60.28 | 78.61 | 743 | Ravens |
| 39 | 3 | Matthew Judon | 73.56 | 61.30 | 77.98 | 674 | Ravens |
| 40 | 4 | Everson Griffen | 73.31 | 68.41 | 75.01 | 585 | Vikings |
| 41 | 5 | Carl Lawson | 72.13 | 66.77 | 77.41 | 225 | Bengals |
| 42 | 6 | Kemoko Turay | 71.59 | 61.54 | 76.21 | 383 | Colts |
| 43 | 7 | Chris Long | 71.41 | 60.46 | 74.55 | 612 | Eagles |
| 44 | 8 | Kyler Fackrell | 70.52 | 56.22 | 76.30 | 626 | Packers |
| 45 | 9 | Leonard Floyd | 70.35 | 63.70 | 73.32 | 794 | Bears |
| 46 | 10 | Tyus Bowser | 70.34 | 59.80 | 73.85 | 164 | Ravens |
| 47 | 11 | Jason Pierre-Paul | 70.03 | 64.18 | 70.60 | 933 | Buccaneers |
| 48 | 12 | Markus Golden | 69.66 | 59.67 | 78.51 | 393 | Cardinals |
| 49 | 13 | Preston Smith | 69.52 | 62.96 | 69.73 | 834 | Commanders |
| 50 | 14 | Aaron Lynch | 69.43 | 63.58 | 75.42 | 353 | Bears |
| 51 | 15 | Za'Darius Smith | 68.95 | 65.87 | 68.09 | 691 | Ravens |
| 52 | 16 | Dion Jordan | 68.71 | 65.70 | 72.06 | 295 | Seahawks |
| 53 | 17 | Sam Hubbard | 68.71 | 65.66 | 66.58 | 508 | Bengals |
| 54 | 18 | Nick Perry | 68.40 | 60.00 | 74.73 | 302 | Packers |
| 55 | 19 | Dante Fowler Jr. | 68.23 | 63.78 | 67.54 | 577 | Rams |
| 56 | 20 | Jordan Jenkins | 68.13 | 59.09 | 70.40 | 660 | Jets |
| 57 | 21 | Julius Peppers | 68.04 | 52.54 | 74.20 | 506 | Panthers |
| 58 | 22 | Clay Matthews | 68.00 | 52.63 | 74.92 | 756 | Packers |
| 59 | 23 | Derrick Morgan | 67.93 | 55.20 | 74.01 | 532 | Titans |
| 60 | 24 | Vinny Curry | 67.30 | 57.82 | 71.54 | 445 | Buccaneers |
| 61 | 25 | Efe Obada | 67.18 | 58.25 | 75.21 | 189 | Panthers |
| 62 | 26 | Shaq Lawson | 67.14 | 68.53 | 65.90 | 440 | Bills |
| 63 | 27 | Alex Okafor | 67.00 | 59.67 | 69.81 | 658 | Saints |
| 64 | 28 | Bruce Irvin | 66.97 | 47.20 | 75.99 | 471 | Falcons |
| 65 | 29 | Lerentee McCray | 66.91 | 58.33 | 73.47 | 100 | Jaguars |
| 66 | 30 | Brian Orakpo | 66.88 | 51.94 | 74.24 | 572 | Titans |
| 67 | 31 | Vic Beasley Jr. | 66.84 | 58.41 | 68.30 | 702 | Falcons |
| 68 | 32 | Deatrich Wise Jr. | 66.66 | 62.41 | 65.32 | 431 | Patriots |
| 69 | 33 | Adrian Clayborn | 66.57 | 63.29 | 66.05 | 318 | Patriots |
| 70 | 34 | John Franklin-Myers | 66.22 | 62.63 | 64.44 | 301 | Rams |
| 71 | 35 | Lorenzo Carter | 66.21 | 61.73 | 66.06 | 442 | Giants |
| 72 | 36 | Jeremiah Attaochu | 66.09 | 62.95 | 72.86 | 171 | Jets |
| 73 | 37 | John Simon | 65.75 | 60.37 | 71.01 | 185 | Patriots |
| 74 | 38 | Barkevious Mingo | 65.35 | 56.10 | 69.24 | 517 | Seahawks |
| 75 | 39 | Samson Ebukam | 65.16 | 61.66 | 63.33 | 692 | Rams |
| 76 | 40 | Trent Murphy | 65.00 | 53.50 | 70.06 | 441 | Bills |
| 77 | 41 | Tim Williams | 64.94 | 68.18 | 67.59 | 120 | Ravens |
| 78 | 42 | Ronald Blair III | 64.82 | 59.32 | 67.46 | 534 | 49ers |
| 79 | 43 | Randy Gregory | 64.59 | 60.41 | 67.17 | 457 | Cowboys |
| 80 | 44 | Arik Armstead | 64.42 | 62.27 | 64.82 | 608 | 49ers |
| 81 | 45 | Ryan Anderson | 64.40 | 68.07 | 60.52 | 164 | Commanders |
| 82 | 46 | Kerry Hyder Jr. | 63.89 | 57.39 | 71.87 | 153 | Lions |
| 83 | 47 | Reggie Gilbert | 63.80 | 59.74 | 67.81 | 487 | Packers |
| 84 | 48 | Shane Ray | 63.77 | 56.52 | 69.54 | 253 | Broncos |
| 85 | 49 | Cassius Marsh | 63.77 | 60.54 | 62.08 | 550 | 49ers |
| 86 | 50 | Bud Dupree | 63.63 | 57.09 | 65.07 | 869 | Steelers |
| 87 | 51 | Zach Moore | 63.22 | 47.57 | 72.62 | 246 | Cardinals |
| 88 | 52 | Charles Harris | 62.84 | 61.22 | 63.00 | 347 | Dolphins |
| 89 | 53 | Brandon Copeland | 62.80 | 53.27 | 64.98 | 611 | Jets |

### Rotation/backup (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 90 | 1 | Matt Longacre | 61.98 | 59.53 | 63.72 | 281 | Rams |
| 91 | 2 | Chris Smith | 61.90 | 57.32 | 62.87 | 336 | Browns |
| 92 | 3 | Solomon Thomas | 61.21 | 64.15 | 55.87 | 644 | 49ers |
| 93 | 4 | Tanoh Kpassagnon | 61.20 | 63.15 | 62.64 | 115 | Chiefs |
| 94 | 5 | Connor Barwin | 61.13 | 45.05 | 68.51 | 289 | Giants |
| 95 | 6 | Devon Kennard | 60.97 | 49.16 | 65.51 | 864 | Lions |
| 96 | 7 | Anthony Chickillo | 60.56 | 54.89 | 60.80 | 295 | Steelers |
| 97 | 8 | Carl Nassib | 60.14 | 59.02 | 57.66 | 598 | Buccaneers |
| 98 | 9 | Benson Mayowa | 60.07 | 58.50 | 58.51 | 550 | Cardinals |
| 99 | 10 | Trey Hendrickson | 59.87 | 63.77 | 61.05 | 136 | Saints |
| 100 | 11 | Brooks Reed | 59.63 | 50.52 | 61.53 | 458 | Falcons |
| 101 | 12 | Emmanuel Ogbah | 59.42 | 59.61 | 58.05 | 806 | Browns |
| 102 | 13 | Anthony Zettel | 59.38 | 54.52 | 59.39 | 158 | Browns |
| 103 | 14 | Dawuane Smoot | 59.33 | 64.23 | 57.09 | 171 | Jaguars |
| 104 | 15 | Romeo Okwara | 59.13 | 59.50 | 58.36 | 716 | Lions |
| 105 | 16 | Eli Harold | 59.05 | 55.47 | 59.87 | 184 | Lions |
| 106 | 17 | Jonathan Woodard | 58.93 | 54.95 | 66.93 | 128 | Dolphins |
| 107 | 18 | Arden Key | 58.88 | 56.96 | 56.00 | 644 | Raiders |
| 108 | 19 | Jordan Willis | 58.87 | 57.71 | 55.47 | 537 | Bengals |
| 109 | 20 | Taco Charlton | 58.86 | 60.16 | 57.08 | 402 | Cowboys |
| 110 | 21 | Michael Johnson | 58.40 | 50.55 | 60.30 | 467 | Bengals |
| 111 | 22 | Stephen Weatherly | 58.23 | 57.63 | 59.57 | 524 | Vikings |
| 112 | 23 | Eddie Yarbrough | 57.88 | 56.09 | 55.55 | 307 | Bills |
| 113 | 24 | Steven Means | 57.58 | 57.04 | 63.45 | 162 | Falcons |
| 114 | 25 | Andre Branch | 57.58 | 51.75 | 58.97 | 483 | Dolphins |
| 115 | 26 | Kareem Martin | 57.37 | 55.34 | 56.44 | 610 | Giants |
| 116 | 27 | Tyquan Lewis | 57.22 | 57.43 | 61.25 | 337 | Colts |
| 117 | 28 | Wes Horton | 57.06 | 49.53 | 58.95 | 471 | Panthers |
| 118 | 29 | Kerry Wynn | 56.91 | 55.60 | 55.39 | 393 | Giants |
| 119 | 30 | Isaiah Irving | 56.90 | 58.03 | 61.23 | 117 | Bears |
| 120 | 31 | Dorance Armstrong | 56.47 | 55.17 | 54.21 | 273 | Cowboys |
| 121 | 32 | Duke Ejiofor | 54.30 | 53.36 | 54.93 | 158 | Texans |
| 122 | 33 | Derrick Shelby | 53.96 | 52.68 | 56.83 | 135 | Falcons |
| 123 | 34 | Shilique Calhoun | 53.81 | 55.30 | 57.09 | 138 | Raiders |
| 124 | 35 | Branden Jackson | 53.67 | 55.93 | 55.59 | 258 | Seahawks |
| 125 | 36 | Rasheem Green | 53.18 | 54.33 | 54.49 | 201 | Seahawks |
| 126 | 37 | Al-Quadin Muhammad | 52.44 | 56.07 | 51.58 | 415 | Colts |
| 127 | 38 | Bryan Cox Jr. | 51.41 | 54.40 | 51.64 | 200 | Panthers |
| 128 | 39 | Cameron Malveaux | 51.24 | 55.59 | 53.43 | 174 | Cardinals |
| 129 | 40 | Keionta Davis | 49.32 | 51.20 | 53.41 | 182 | Patriots |

## G — Guard

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Marshal Yanda | 90.53 | 85.00 | 90.05 | 1162 | Ravens |
| 2 | 2 | Shaq Mason | 90.19 | 85.00 | 89.49 | 954 | Patriots |
| 3 | 3 | Quenton Nelson | 87.10 | 79.70 | 87.86 | 1136 | Colts |
| 4 | 4 | Zack Martin | 86.95 | 81.50 | 86.42 | 1754 | Cowboys |
| 5 | 5 | Joel Bitonio | 86.92 | 80.30 | 87.17 | 1091 | Browns |
| 6 | 6 | Brandon Brooks | 85.27 | 78.10 | 85.88 | 1087 | Eagles |
| 7 | 7 | Mark Glowinski | 85.21 | 77.70 | 86.05 | 601 | Colts |
| 8 | 8 | Rodger Saffold | 84.78 | 76.90 | 85.86 | 1069 | Rams |
| 9 | 9 | Kevin Zeitler | 84.48 | 77.20 | 85.16 | 1091 | Browns |
| 10 | 10 | Joe Thuney | 82.95 | 75.70 | 83.62 | 1120 | Patriots |
| 11 | 11 | T.J. Lang | 82.86 | 76.10 | 83.20 | 282 | Lions |
| 12 | 12 | Ali Marpet | 82.80 | 75.90 | 83.23 | 1117 | Buccaneers |
| 13 | 13 | Ben Garland | 81.95 | 71.60 | 84.68 | 371 | Falcons |
| 14 | 14 | David DeCastro | 81.49 | 74.80 | 81.79 | 958 | Steelers |
| 15 | 15 | Gabe Jackson | 81.29 | 73.70 | 82.19 | 855 | Raiders |
| 16 | 16 | Brandon Scherff | 80.41 | 72.50 | 81.52 | 506 | Commanders |
| 17 | 17 | Ted Karras | 80.19 | 74.10 | 80.09 | 171 | Patriots |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Austin Blythe | 79.81 | 76.40 | 77.91 | 1101 | Rams |
| 19 | 2 | Ramon Foster | 79.31 | 72.30 | 79.82 | 1116 | Steelers |
| 20 | 3 | Mike Person | 79.10 | 71.40 | 80.07 | 1000 | 49ers |
| 21 | 4 | Andrew Norwell | 78.89 | 70.50 | 80.32 | 726 | Jaguars |
| 22 | 5 | Isaac Seumalo | 77.61 | 68.10 | 79.79 | 548 | Eagles |
| 23 | 6 | Quinton Spain | 77.56 | 67.70 | 79.97 | 856 | Titans |
| 24 | 7 | Mike Iupati | 77.13 | 65.70 | 80.59 | 477 | Cardinals |
| 25 | 8 | Larry Warford | 77.05 | 68.00 | 78.91 | 980 | Saints |
| 26 | 9 | Laken Tomlinson | 76.92 | 67.50 | 79.03 | 1028 | 49ers |
| 27 | 10 | Will Hernandez | 76.76 | 67.90 | 78.50 | 1027 | Giants |
| 28 | 11 | Trai Turner | 76.46 | 67.90 | 78.00 | 762 | Panthers |
| 29 | 12 | Frank Ragnow | 76.34 | 66.50 | 78.73 | 1076 | Lions |
| 30 | 13 | Wes Schweitzer | 76.20 | 66.90 | 78.24 | 901 | Falcons |
| 31 | 14 | Brandon Fusco | 76.07 | 65.40 | 79.02 | 436 | Falcons |
| 32 | 15 | John Miller | 76.01 | 67.90 | 77.25 | 885 | Bills |
| 33 | 16 | Kyle Long | 75.96 | 66.60 | 78.03 | 511 | Bears |
| 34 | 17 | Matt Slauson | 75.68 | 66.80 | 77.44 | 376 | Colts |
| 35 | 18 | Brian Winters | 75.59 | 65.90 | 77.89 | 1001 | Jets |
| 36 | 19 | Andrew Wylie | 75.43 | 66.00 | 77.55 | 687 | Chiefs |
| 37 | 20 | Laurent Duvernay-Tardif | 75.08 | 63.30 | 78.76 | 331 | Chiefs |
| 38 | 21 | B.J. Finney | 75.03 | 71.50 | 73.22 | 165 | Steelers |
| 39 | 22 | Lane Taylor | 74.45 | 66.00 | 75.91 | 882 | Packers |
| 40 | 23 | Ron Leary | 74.24 | 63.30 | 77.37 | 383 | Broncos |
| 41 | 24 | Justin McCray | 74.08 | 64.90 | 76.03 | 481 | Packers |

### Starter (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 42 | 1 | Tom Compton | 73.83 | 62.40 | 77.28 | 837 | Vikings |
| 43 | 2 | Oday Aboushi | 73.50 | 63.70 | 75.86 | 407 | Cardinals |
| 44 | 3 | Billy Turner | 73.45 | 64.40 | 75.31 | 824 | Broncos |
| 45 | 4 | Connor McGovern | 73.03 | 60.00 | 77.55 | 1056 | Broncos |
| 46 | 5 | Alex Redmond | 72.97 | 60.80 | 76.91 | 928 | Bengals |
| 47 | 6 | Michael Schofield III | 72.89 | 63.90 | 74.72 | 978 | Chargers |
| 48 | 7 | A.J. Cann | 72.45 | 62.70 | 74.78 | 934 | Jaguars |
| 49 | 8 | Mike Remmers | 72.14 | 61.10 | 75.33 | 1048 | Vikings |
| 50 | 9 | Greg Van Roten | 71.42 | 61.20 | 74.06 | 1059 | Panthers |
| 51 | 10 | Clint Boling | 71.17 | 62.60 | 72.72 | 969 | Bengals |
| 52 | 11 | Josh Kline | 71.10 | 60.90 | 73.73 | 975 | Titans |
| 53 | 12 | D.J. Fluker | 71.07 | 56.10 | 76.88 | 607 | Seahawks |
| 54 | 13 | Kelechi Osemele | 70.56 | 58.50 | 74.43 | 735 | Raiders |
| 55 | 14 | Connor Williams | 70.44 | 60.00 | 73.23 | 688 | Cowboys |
| 56 | 15 | Danny Isidora | 70.39 | 57.00 | 75.15 | 214 | Vikings |
| 57 | 16 | James Carpenter | 70.30 | 58.50 | 74.00 | 624 | Jets |
| 58 | 17 | Jesse Davis | 70.20 | 59.20 | 73.36 | 921 | Dolphins |
| 59 | 18 | Dakota Dozier | 70.19 | 57.10 | 74.75 | 106 | Jets |
| 60 | 19 | Senio Kelemete | 69.66 | 58.90 | 72.66 | 895 | Texans |
| 61 | 20 | Zane Beadles | 69.16 | 58.20 | 72.30 | 279 | Falcons |
| 62 | 21 | Stefen Wisniewski | 69.15 | 57.00 | 73.09 | 643 | Eagles |
| 63 | 22 | Wyatt Teller | 68.81 | 60.20 | 70.39 | 476 | Bills |
| 64 | 23 | Jeff Allen | 68.49 | 56.50 | 72.32 | 224 | Chiefs |
| 65 | 24 | Bryan Witzmann | 68.20 | 56.50 | 71.84 | 533 | Bears |
| 66 | 25 | Kenny Wiggins | 68.11 | 57.10 | 71.29 | 798 | Lions |
| 67 | 26 | Patrick Omameh | 67.85 | 56.10 | 71.52 | 679 | Jaguars |
| 68 | 27 | Corey Levin | 67.44 | 54.30 | 72.03 | 140 | Titans |
| 69 | 28 | Zach Fulton | 67.32 | 54.90 | 71.44 | 817 | Texans |
| 70 | 29 | Jeremiah Sirles | 67.21 | 54.20 | 71.72 | 140 | Bills |
| 71 | 30 | Jonathan Cooper | 67.17 | 54.80 | 71.25 | 201 | Commanders |
| 72 | 31 | Vladimir Ducasse | 67.11 | 51.60 | 73.28 | 564 | Bills |
| 73 | 32 | Shawn Lauvao | 66.68 | 53.30 | 71.43 | 284 | Commanders |
| 74 | 33 | Max Garcia | 66.63 | 53.70 | 71.09 | 242 | Broncos |
| 75 | 34 | Justin Pugh | 64.91 | 53.10 | 68.61 | 343 | Cardinals |
| 76 | 35 | Eric Kush | 64.89 | 57.80 | 65.45 | 344 | Bears |
| 77 | 36 | Alex Lewis | 64.70 | 50.90 | 69.73 | 707 | Ravens |
| 78 | 37 | J.R. Sweezy | 64.03 | 49.80 | 69.35 | 948 | Seahawks |
| 79 | 38 | Dan Feeney | 64.03 | 49.20 | 69.75 | 995 | Chargers |
| 80 | 39 | Lucas Patrick | 63.99 | 50.10 | 69.08 | 279 | Packers |
| 81 | 40 | Byron Bell | 63.11 | 49.00 | 68.35 | 528 | Packers |
| 82 | 41 | Jon Feliciano | 62.98 | 49.50 | 67.80 | 227 | Raiders |
| 83 | 42 | Cameron Erving | 62.61 | 45.90 | 69.59 | 830 | Chiefs |

### Rotation/backup (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 84 | 1 | Colby Gossett | 61.75 | 46.00 | 68.09 | 282 | Cardinals |
| 85 | 2 | Andrus Peat | 61.17 | 43.80 | 68.58 | 739 | Saints |
| 86 | 3 | Ethan Pocic | 61.16 | 45.90 | 67.17 | 296 | Seahawks |
| 87 | 4 | Caleb Benenoch | 60.74 | 44.80 | 67.20 | 844 | Buccaneers |
| 88 | 5 | Ted Larsen | 60.38 | 44.20 | 67.00 | 752 | Dolphins |
| 89 | 6 | Xavier Su'a-Filo | 60.37 | 44.10 | 67.05 | 494 | Cowboys |
| 90 | 7 | Josh LeRibeus | 57.52 | 41.30 | 64.16 | 179 | Saints |
| 91 | 8 | Alex Cappa | 56.90 | 40.10 | 63.94 | 106 | Buccaneers |
| 92 | 9 | Cameron Tom | 55.92 | 44.10 | 59.64 | 178 | Saints |

## HB — Running Back

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Chubb | 87.96 | 86.40 | 84.83 | 142 | Browns |
| 2 | 2 | Saquon Barkley | 85.96 | 85.20 | 82.30 | 470 | Giants |
| 3 | 3 | Austin Ekeler | 84.47 | 84.20 | 80.48 | 190 | Chargers |
| 4 | 4 | Alvin Kamara | 84.03 | 82.90 | 80.62 | 334 | Saints |
| 5 | 5 | Derrick Henry | 81.90 | 85.30 | 75.46 | 115 | Titans |
| 6 | 6 | Kerryon Johnson | 81.67 | 80.20 | 78.48 | 169 | Lions |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Chris Carson | 79.52 | 83.20 | 72.90 | 136 | Seahawks |
| 8 | 2 | Christian McCaffrey | 79.18 | 82.80 | 72.60 | 493 | Panthers |
| 9 | 3 | Aaron Jones | 78.77 | 80.70 | 73.32 | 204 | Packers |
| 10 | 4 | Melvin Gordon III | 77.17 | 86.60 | 66.72 | 241 | Chargers |
| 11 | 5 | Phillip Lindsay | 76.39 | 82.40 | 68.22 | 185 | Broncos |
| 12 | 6 | Mark Ingram II | 76.36 | 80.60 | 69.36 | 133 | Saints |
| 13 | 7 | Dalvin Cook | 75.54 | 72.60 | 73.34 | 244 | Vikings |
| 14 | 8 | Duke Johnson Jr. | 75.54 | 72.10 | 73.66 | 300 | Browns |
| 15 | 9 | Todd Gurley II | 75.21 | 77.80 | 69.31 | 428 | Rams |
| 16 | 10 | Ty Montgomery | 75.05 | 68.60 | 75.18 | 152 | Ravens |
| 17 | 11 | Adrian Peterson | 74.88 | 76.20 | 69.83 | 158 | Commanders |
| 18 | 12 | Kenyan Drake | 74.82 | 65.60 | 76.80 | 300 | Dolphins |
| 19 | 13 | Damien Williams | 74.81 | 79.20 | 67.72 | 117 | Chiefs |
| 20 | 14 | Jalen Richard | 74.38 | 66.60 | 75.40 | 269 | Raiders |

### Starter (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Ezekiel Elliott | 73.98 | 72.20 | 71.00 | 412 | Cowboys |
| 22 | 2 | Spencer Ware | 73.87 | 69.90 | 72.35 | 142 | Chiefs |
| 23 | 3 | Joe Mixon | 73.09 | 77.80 | 65.78 | 274 | Bengals |
| 24 | 4 | Matt Breida | 73.06 | 75.80 | 67.07 | 160 | 49ers |
| 25 | 5 | Chris Ivory | 73.04 | 74.10 | 68.17 | 117 | Bills |
| 26 | 6 | Tarik Cohen | 72.57 | 73.20 | 67.98 | 307 | Bears |
| 27 | 7 | Frank Gore | 72.46 | 80.30 | 63.06 | 128 | Dolphins |
| 28 | 8 | James Conner | 72.16 | 72.10 | 68.04 | 380 | Steelers |
| 29 | 9 | Dion Lewis | 71.39 | 65.30 | 71.29 | 315 | Titans |
| 30 | 10 | Latavius Murray | 70.71 | 75.70 | 63.22 | 221 | Vikings |
| 31 | 11 | Marlon Mack | 70.46 | 68.60 | 67.53 | 171 | Colts |
| 32 | 12 | Lamar Miller | 70.31 | 73.70 | 63.89 | 276 | Texans |
| 33 | 13 | Theo Riddick | 70.26 | 72.30 | 64.73 | 272 | Lions |
| 34 | 14 | Jordan Wilkins | 70.17 | 63.40 | 70.51 | 100 | Colts |
| 35 | 15 | Isaiah Crowell | 69.99 | 70.40 | 65.55 | 145 | Jets |
| 36 | 16 | Jaylen Samuels | 69.62 | 68.80 | 66.00 | 130 | Steelers |
| 37 | 17 | Jordan Howard | 69.49 | 68.10 | 66.25 | 211 | Bears |
| 38 | 18 | Chris Thompson | 69.30 | 58.80 | 72.13 | 206 | Commanders |
| 39 | 19 | Royce Freeman | 69.01 | 63.00 | 68.85 | 119 | Broncos |
| 40 | 20 | Ito Smith | 68.95 | 68.80 | 64.89 | 164 | Falcons |
| 41 | 21 | Giovani Bernard | 68.86 | 68.10 | 65.20 | 195 | Bengals |
| 42 | 22 | Mike Davis | 68.84 | 75.10 | 60.50 | 186 | Seahawks |
| 43 | 23 | Tevin Coleman | 68.27 | 65.70 | 65.81 | 296 | Falcons |
| 44 | 24 | Nyheim Hines | 68.15 | 71.60 | 61.68 | 346 | Colts |
| 45 | 25 | Doug Martin | 67.74 | 66.60 | 64.34 | 135 | Raiders |
| 46 | 26 | James White | 67.69 | 71.00 | 61.31 | 411 | Patriots |
| 47 | 27 | David Johnson | 67.33 | 64.10 | 65.31 | 360 | Cardinals |
| 48 | 28 | LeSean McCoy | 66.87 | 60.90 | 66.69 | 225 | Bills |
| 49 | 29 | Jacquizz Rodgers | 66.71 | 65.10 | 63.61 | 234 | Buccaneers |
| 50 | 30 | Alex Collins | 66.53 | 59.80 | 66.85 | 123 | Ravens |
| 51 | 31 | Devontae Booker | 66.47 | 64.60 | 63.55 | 226 | Broncos |
| 52 | 32 | Peyton Barber | 66.31 | 66.80 | 61.81 | 270 | Buccaneers |
| 53 | 33 | Jamaal Williams | 65.71 | 67.50 | 60.35 | 275 | Packers |
| 54 | 34 | Corey Clement | 64.09 | 57.80 | 64.11 | 125 | Eagles |
| 55 | 35 | T.J. Yeldon | 63.55 | 56.00 | 64.42 | 310 | Jaguars |
| 56 | 36 | Javorius Allen | 62.95 | 62.80 | 58.89 | 183 | Ravens |
| 57 | 37 | Carlos Hyde | 62.79 | 50.40 | 66.88 | 149 | Jaguars |
| 58 | 38 | Kapri Bibbs | 62.60 | 58.60 | 61.10 | 128 | Packers |
| 59 | 39 | Wendell Smallwood | 62.41 | 56.20 | 62.39 | 197 | Eagles |
| 60 | 40 | Trenton Cannon | 62.03 | 55.10 | 62.48 | 101 | Jets |
| 61 | 41 | Alfred Blue | 62.02 | 63.10 | 57.14 | 208 | Texans |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 62 | 1 | Jeff Wilson Jr. | 61.84 | 54.70 | 62.44 | 106 | 49ers |
| 63 | 2 | Chase Edmonds | 61.55 | 60.50 | 58.08 | 114 | Cardinals |
| 64 | 3 | Elijah McGuire | 61.46 | 54.90 | 61.67 | 154 | Jets |

## LB — Linebacker

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Bobby Wagner | 90.01 | 91.70 | 85.24 | 925 | Seahawks |
| 2 | 2 | Luke Kuechly | 87.38 | 90.50 | 82.38 | 927 | Panthers |
| 3 | 3 | Zach Brown | 85.11 | 88.20 | 79.81 | 703 | Commanders |
| 4 | 4 | Lorenzo Alexander | 83.64 | 86.60 | 77.50 | 629 | Bills |
| 5 | 5 | Leighton Vander Esch | 83.05 | 84.40 | 77.99 | 785 | Cowboys |
| 6 | 6 | Benardrick McKinney | 80.41 | 80.90 | 75.91 | 919 | Texans |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Brennan Scarlett | 79.99 | 89.90 | 73.39 | 108 | Texans |
| 8 | 2 | Jaylon Smith | 79.80 | 84.00 | 72.84 | 978 | Cowboys |
| 9 | 3 | Jayon Brown | 78.35 | 81.20 | 72.29 | 852 | Titans |
| 10 | 4 | Lavonte David | 76.17 | 76.10 | 74.04 | 922 | Buccaneers |
| 11 | 5 | Avery Williamson | 75.10 | 72.80 | 72.46 | 1114 | Jets |
| 12 | 6 | Blake Martinez | 75.07 | 73.90 | 71.68 | 1050 | Packers |
| 13 | 7 | Demario Davis | 74.21 | 74.60 | 69.78 | 877 | Saints |

### Starter (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Jordan Hicks | 73.98 | 75.10 | 73.97 | 705 | Eagles |
| 15 | 2 | Wesley Woodyard | 73.88 | 72.20 | 71.86 | 714 | Titans |
| 16 | 3 | Joe Schobert | 73.51 | 76.80 | 68.71 | 897 | Browns |
| 17 | 4 | Manti Te'o | 72.97 | 80.40 | 72.29 | 141 | Saints |
| 18 | 5 | Matt Milano | 72.91 | 76.10 | 69.35 | 741 | Bills |
| 19 | 6 | Jerome Baker | 72.66 | 70.70 | 69.80 | 678 | Dolphins |
| 20 | 7 | Thomas Davis Sr. | 72.39 | 73.90 | 69.30 | 649 | Panthers |
| 21 | 8 | Josh Bynes | 72.24 | 74.80 | 71.15 | 726 | Cardinals |
| 22 | 9 | C.J. Mosley | 72.09 | 70.10 | 70.18 | 874 | Ravens |
| 23 | 10 | Danny Trevathan | 72.06 | 71.90 | 70.70 | 987 | Bears |
| 24 | 11 | Todd Davis | 71.68 | 70.60 | 68.86 | 842 | Broncos |
| 25 | 12 | Darron Lee | 71.57 | 72.40 | 69.87 | 808 | Jets |
| 26 | 13 | Deion Jones | 71.15 | 73.70 | 70.49 | 384 | Falcons |
| 27 | 14 | Anthony Barr | 71.09 | 70.90 | 68.61 | 810 | Vikings |
| 28 | 15 | Jason Cabinda | 70.73 | 73.40 | 71.03 | 164 | Raiders |
| 29 | 16 | Cory Littleton | 70.36 | 66.00 | 70.35 | 964 | Rams |
| 30 | 17 | L.J. Fort | 70.34 | 72.40 | 70.73 | 305 | Steelers |
| 31 | 18 | Myles Jack | 70.12 | 68.30 | 67.59 | 1024 | Jaguars |
| 32 | 19 | Shaq Thompson | 69.75 | 68.70 | 68.05 | 599 | Panthers |
| 33 | 20 | A.J. Klein | 69.68 | 69.10 | 67.79 | 670 | Saints |
| 34 | 21 | Nigel Bradham | 68.96 | 66.60 | 66.89 | 919 | Eagles |
| 35 | 22 | Christian Jones | 68.62 | 64.80 | 67.31 | 643 | Lions |
| 36 | 23 | Dont'a Hightower | 68.51 | 65.00 | 67.20 | 774 | Patriots |
| 37 | 24 | Leon Jacobs | 68.47 | 68.80 | 68.25 | 146 | Jaguars |
| 38 | 25 | Eric Kendricks | 68.30 | 64.50 | 67.92 | 877 | Vikings |
| 39 | 26 | Foyesade Oluokun | 68.27 | 65.70 | 65.81 | 525 | Falcons |
| 40 | 27 | Ben Gedeon | 68.02 | 63.50 | 67.52 | 311 | Vikings |
| 41 | 28 | K.J. Wright | 68.01 | 70.20 | 68.44 | 223 | Seahawks |
| 42 | 29 | Zach Cunningham | 67.86 | 65.20 | 66.76 | 753 | Texans |
| 43 | 30 | Fred Warner | 67.84 | 64.10 | 66.17 | 1060 | 49ers |
| 44 | 31 | Kyle Van Noy | 67.69 | 65.30 | 65.12 | 946 | Patriots |
| 45 | 32 | Alex Anzalone | 67.66 | 70.50 | 66.28 | 486 | Saints |
| 46 | 33 | Roquan Smith | 67.59 | 64.20 | 65.68 | 880 | Bears |
| 47 | 34 | Rashaan Evans | 67.43 | 65.50 | 66.64 | 494 | Titans |
| 48 | 35 | Telvin Smith Sr. | 66.96 | 63.70 | 64.96 | 1020 | Jaguars |
| 49 | 36 | Ja'Whaun Bentley | 66.59 | 80.70 | 73.01 | 138 | Patriots |
| 50 | 37 | Denzel Perryman | 66.52 | 70.10 | 67.27 | 386 | Chargers |
| 51 | 38 | Elandon Roberts | 66.26 | 63.20 | 64.33 | 429 | Patriots |
| 52 | 39 | Josey Jewell | 66.10 | 61.70 | 64.87 | 460 | Broncos |
| 53 | 40 | Raekwon McMillan | 65.99 | 60.20 | 65.68 | 831 | Dolphins |
| 54 | 41 | Vince Williams | 65.69 | 65.00 | 64.07 | 744 | Steelers |
| 55 | 42 | B.J. Goodson | 65.58 | 66.20 | 66.83 | 513 | Giants |
| 56 | 43 | Jon Bostic | 65.46 | 59.50 | 65.90 | 560 | Steelers |
| 57 | 44 | Malcolm Smith | 65.27 | 62.60 | 64.97 | 336 | 49ers |
| 58 | 45 | Marquel Lee | 65.09 | 61.00 | 65.22 | 448 | Raiders |
| 59 | 46 | Eric Wilson | 64.64 | 62.70 | 63.07 | 336 | Vikings |
| 60 | 47 | Brandon Marshall | 64.30 | 63.50 | 64.31 | 468 | Broncos |
| 61 | 48 | Haason Reddick | 64.28 | 60.40 | 62.70 | 846 | Cardinals |
| 62 | 49 | De'Vondre Campbell | 63.61 | 56.70 | 64.46 | 902 | Falcons |
| 63 | 50 | Kamu Grugier-Hill | 63.54 | 63.00 | 65.98 | 330 | Eagles |
| 64 | 51 | Mychal Kendricks | 63.04 | 66.40 | 63.08 | 183 | Seahawks |
| 65 | 52 | Jamie Collins Sr. | 62.73 | 61.20 | 62.91 | 1067 | Browns |
| 66 | 53 | Patrick Onwuasor | 62.60 | 58.50 | 62.84 | 435 | Ravens |
| 67 | 54 | Ramik Wilson | 62.03 | 61.30 | 66.59 | 158 | Rams |

### Rotation/backup (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 68 | 1 | Mason Foster | 61.97 | 57.30 | 64.35 | 1014 | Commanders |
| 69 | 2 | Nick Kwiatkoski | 61.44 | 56.10 | 65.61 | 112 | Bears |
| 70 | 3 | Tahir Whitehead | 61.40 | 54.80 | 61.63 | 1025 | Raiders |
| 71 | 4 | Nick Vigil | 61.40 | 60.00 | 63.36 | 672 | Bengals |
| 72 | 5 | Kenny Young | 60.99 | 55.10 | 60.75 | 371 | Ravens |
| 73 | 6 | Tremaine Edmunds | 60.83 | 57.00 | 60.25 | 927 | Bills |
| 74 | 7 | Anthony Walker Jr. | 60.77 | 60.00 | 62.85 | 695 | Colts |
| 75 | 8 | Elijah Lee | 60.62 | 58.20 | 65.36 | 476 | 49ers |
| 76 | 9 | Sean Lee | 60.49 | 59.70 | 63.10 | 221 | Cowboys |
| 77 | 10 | Kyle Emanuel | 60.15 | 54.10 | 61.05 | 216 | Chargers |
| 78 | 11 | Antonio Morrison | 60.00 | 55.80 | 59.36 | 302 | Packers |
| 79 | 12 | Jalen Reeves-Maybin | 59.63 | 58.30 | 62.47 | 111 | Lions |
| 80 | 13 | Jatavis Brown | 59.36 | 53.40 | 60.52 | 637 | Chargers |
| 81 | 14 | Gerald Hodges | 59.28 | 56.00 | 63.24 | 356 | Cardinals |
| 82 | 15 | David Mayo | 59.22 | 58.70 | 62.27 | 145 | Panthers |
| 83 | 16 | Reggie Ragland | 59.22 | 53.60 | 59.74 | 582 | Chiefs |
| 84 | 17 | Jarrad Davis | 58.84 | 51.00 | 60.68 | 976 | Lions |
| 85 | 18 | Matthew Adams | 58.56 | 52.90 | 62.34 | 215 | Colts |
| 86 | 19 | Preston Brown | 58.46 | 53.60 | 62.21 | 375 | Bengals |
| 87 | 20 | Kiko Alonso | 58.39 | 47.70 | 61.86 | 1004 | Dolphins |
| 88 | 21 | Jordan Evans | 58.21 | 50.70 | 62.30 | 510 | Bengals |
| 89 | 22 | Zaire Franklin | 57.61 | 51.20 | 62.91 | 176 | Colts |
| 90 | 23 | Kwon Alexander | 57.30 | 57.40 | 59.52 | 366 | Buccaneers |
| 91 | 24 | Kyzir White | 56.42 | 65.60 | 66.14 | 142 | Chargers |
| 92 | 25 | Nicholas Morrow | 56.19 | 50.90 | 57.25 | 416 | Raiders |
| 93 | 26 | Alec Ogletree | 56.04 | 49.30 | 57.93 | 885 | Giants |
| 94 | 27 | Emmanuel Lamur | 55.17 | 52.90 | 59.70 | 145 | Jets |
| 95 | 28 | Nathan Gerry | 54.61 | 51.80 | 61.69 | 137 | Eagles |
| 96 | 29 | Hardy Nickerson | 54.44 | 45.60 | 57.74 | 538 | Bengals |
| 97 | 30 | Josh Harvey-Clemons | 54.37 | 51.40 | 57.39 | 196 | Commanders |
| 98 | 31 | Mark Barron | 54.24 | 44.60 | 58.90 | 569 | Rams |
| 99 | 32 | Vincent Rey | 54.21 | 47.70 | 58.13 | 178 | Bengals |
| 100 | 33 | Ray-Ray Armstrong | 54.00 | 49.70 | 60.71 | 214 | Browns |
| 101 | 34 | Devante Bond | 53.88 | 47.10 | 59.24 | 248 | Buccaneers |
| 102 | 35 | Deone Bucannon | 53.65 | 45.30 | 58.49 | 389 | Cardinals |
| 103 | 36 | Dylan Cole | 53.49 | 49.30 | 58.88 | 120 | Texans |
| 104 | 37 | Damien Wilson | 52.88 | 42.30 | 56.40 | 287 | Cowboys |
| 105 | 38 | Mark Nzeocha | 52.57 | 53.10 | 57.22 | 175 | 49ers |
| 106 | 39 | Vontaze Burfict | 52.56 | 48.00 | 59.04 | 298 | Bengals |
| 107 | 40 | Duke Riley | 52.49 | 42.00 | 57.79 | 408 | Falcons |
| 108 | 41 | Shaun Dion Hamilton | 52.39 | 53.40 | 60.97 | 129 | Commanders |
| 109 | 42 | Anthony Hitchens | 52.32 | 39.80 | 58.26 | 944 | Chiefs |
| 110 | 43 | Oren Burks | 51.92 | 44.20 | 57.06 | 126 | Packers |
| 111 | 44 | Christian Kirksey | 51.51 | 44.70 | 56.56 | 474 | Browns |
| 112 | 45 | Adarius Taylor | 50.37 | 41.10 | 57.48 | 635 | Buccaneers |
| 113 | 46 | Neville Hewitt | 50.03 | 42.40 | 58.35 | 268 | Jets |
| 114 | 47 | Tae Davis | 49.93 | 40.60 | 57.18 | 344 | Giants |
| 115 | 48 | Terrance Smith | 46.99 | 42.90 | 57.02 | 173 | Chiefs |
| 116 | 49 | Riley Bullough | 45.93 | 38.90 | 58.69 | 126 | Buccaneers |
| 117 | 50 | Tanner Vallejo | 45.53 | 39.40 | 57.03 | 145 | Browns |
| 118 | 51 | Austin Calitro | 45.01 | 29.20 | 56.59 | 282 | Seahawks |

## QB — Quarterback

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Drew Brees | 88.14 | 90.82 | 83.31 | 542 | Saints |
| 2 | 2 | Tom Brady | 85.14 | 90.57 | 76.39 | 625 | Patriots |
| 3 | 3 | Philip Rivers | 83.87 | 85.25 | 79.51 | 579 | Chargers |
| 4 | 4 | Matt Ryan | 83.03 | 83.22 | 78.33 | 706 | Falcons |
| 5 | 5 | Russell Wilson | 82.07 | 81.51 | 79.91 | 546 | Seahawks |
| 6 | 6 | Andrew Luck | 81.33 | 88.94 | 74.81 | 725 | Colts |
| 7 | 7 | Patrick Mahomes | 80.89 | 92.32 | 84.20 | 683 | Chiefs |
| 8 | 8 | Aaron Rodgers | 80.23 | 85.62 | 73.76 | 724 | Packers |
| 9 | 9 | Jared Goff | 80.20 | 80.24 | 77.10 | 653 | Rams |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Ben Roethlisberger | 79.21 | 77.78 | 75.84 | 761 | Steelers |
| 11 | 2 | Deshaun Watson | 78.47 | 78.89 | 77.85 | 673 | Texans |
| 12 | 3 | Kirk Cousins | 77.89 | 76.83 | 74.23 | 705 | Vikings |
| 13 | 4 | Carson Wentz | 77.05 | 78.71 | 75.05 | 472 | Eagles |
| 14 | 5 | Matthew Stafford | 76.07 | 76.82 | 70.91 | 649 | Lions |
| 15 | 6 | Derek Carr | 74.50 | 72.97 | 71.73 | 647 | Raiders |
| 16 | 7 | Ryan Fitzpatrick | 74.43 | 75.38 | 79.09 | 302 | Buccaneers |
| 17 | 8 | Dak Prescott | 74.41 | 72.85 | 71.70 | 653 | Cowboys |

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Andy Dalton | 72.43 | 75.88 | 69.59 | 428 | Bengals |
| 19 | 2 | Marcus Mariota | 71.44 | 70.55 | 71.97 | 431 | Titans |
| 20 | 3 | Baker Mayfield | 71.12 | 79.90 | 75.76 | 575 | Browns |
| 21 | 4 | Jameis Winston | 70.93 | 70.44 | 70.84 | 471 | Buccaneers |
| 22 | 5 | Cam Newton | 70.57 | 69.31 | 68.74 | 558 | Panthers |
| 23 | 6 | Alex Smith | 70.40 | 74.20 | 67.38 | 396 | Commanders |
| 24 | 7 | Eli Manning | 69.74 | 64.64 | 69.36 | 657 | Giants |
| 25 | 8 | Case Keenum | 68.16 | 69.03 | 63.67 | 680 | Broncos |
| 26 | 9 | Mitch Trubisky | 66.23 | 59.91 | 71.33 | 539 | Bears |
| 27 | 10 | Joe Flacco | 65.70 | 69.23 | 62.51 | 427 | Ravens |
| 28 | 11 | Blake Bortles | 64.56 | 64.13 | 62.94 | 493 | Jaguars |
| 29 | 12 | Nick Foles | 64.35 | 71.70 | 73.12 | 221 | Eagles |
| 30 | 13 | Jimmy Garoppolo | 63.04 | 71.43 | 72.01 | 117 | 49ers |
| 31 | 14 | Nick Mullens | 62.97 | 66.70 | 71.44 | 316 | 49ers |

### Rotation/backup (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 32 | 1 | Sam Darnold | 61.31 | 62.90 | 62.15 | 477 | Jets |
| 33 | 2 | C.J. Beathard | 59.42 | 62.55 | 63.04 | 207 | 49ers |
| 34 | 3 | Lamar Jackson | 59.09 | 59.30 | 64.44 | 217 | Ravens |
| 35 | 4 | Josh Allen | 59.06 | 58.00 | 59.65 | 429 | Bills |
| 36 | 5 | Tyrod Taylor | 58.05 | 66.60 | 56.89 | 115 | Browns |
| 37 | 6 | Ryan Tannehill | 57.15 | 47.74 | 69.72 | 343 | Dolphins |
| 38 | 7 | Josh Rosen | 56.59 | 48.20 | 57.44 | 481 | Cardinals |
| 39 | 8 | Cody Kessler | 56.38 | 55.79 | 58.44 | 177 | Jaguars |
| 40 | 9 | Jeff Driskel | 55.96 | 50.96 | 60.17 | 223 | Bengals |
| 41 | 10 | Brock Osweiler | 54.25 | 51.65 | 62.88 | 211 | Dolphins |
| 42 | 11 | Josh McCown | 54.08 | 57.24 | 56.92 | 126 | Jets |
| 43 | 12 | Josh Johnson | 53.33 | 42.10 | 56.79 | 126 | Commanders |
| 44 | 13 | Blaine Gabbert | 51.71 | 40.43 | 58.08 | 115 | Titans |

## S — Safety

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jamal Adams | 90.83 | 89.60 | 87.49 | 1119 | Jets |
| 2 | 2 | Eddie Jackson | 90.49 | 94.70 | 84.82 | 906 | Bears |
| 3 | 3 | John Johnson III | 89.81 | 86.10 | 88.11 | 961 | Rams |
| 4 | 4 | Earl Thomas III | 89.51 | 90.60 | 92.53 | 237 | Seahawks |
| 5 | 5 | Tracy Walker III | 88.90 | 83.90 | 88.07 | 268 | Lions |
| 6 | 6 | Derwin James Jr. | 88.80 | 89.70 | 84.04 | 1027 | Chargers |
| 7 | 7 | Kevin Byard | 87.42 | 82.50 | 86.53 | 1042 | Titans |
| 8 | 8 | Micah Hyde | 86.76 | 88.60 | 81.89 | 882 | Bills |
| 9 | 9 | Clayton Fejedelem | 86.76 | 88.40 | 85.76 | 167 | Bengals |
| 10 | 10 | Anthony Harris | 86.61 | 87.10 | 89.41 | 624 | Vikings |
| 11 | 11 | Ha Ha Clinton-Dix | 85.42 | 82.00 | 83.54 | 1025 | Commanders |
| 12 | 12 | Adrian Amos | 84.88 | 85.00 | 82.10 | 1029 | Bears |
| 13 | 13 | Jessie Bates III | 83.90 | 80.90 | 81.74 | 1114 | Bengals |
| 14 | 14 | Malcolm Jenkins | 83.04 | 83.30 | 78.70 | 1038 | Eagles |
| 15 | 15 | Damontae Kazee | 82.87 | 82.70 | 78.81 | 991 | Falcons |
| 16 | 16 | Devin McCourty | 82.76 | 80.00 | 80.43 | 1004 | Patriots |
| 17 | 17 | Bradley McDougald | 82.58 | 77.90 | 81.54 | 874 | Seahawks |
| 18 | 18 | Eric Weddle | 82.33 | 76.90 | 81.79 | 1015 | Ravens |
| 19 | 19 | Erik Harris | 81.63 | 79.90 | 79.65 | 433 | Raiders |
| 20 | 20 | Malik Hooker | 81.25 | 81.60 | 81.67 | 912 | Colts |
| 21 | 21 | Justin Reid | 80.66 | 74.70 | 80.46 | 906 | Texans |
| 22 | 22 | D.J. Swearinger Sr. | 80.33 | 79.50 | 77.23 | 961 | Cardinals |
| 23 | 23 | Tre Boston | 80.00 | 78.20 | 78.28 | 950 | Cardinals |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Jabrill Peppers | 79.05 | 77.30 | 77.22 | 765 | Browns |
| 25 | 2 | Shawn Williams | 78.60 | 78.70 | 76.13 | 995 | Bengals |
| 26 | 3 | Lamarcus Joyner | 77.87 | 71.20 | 79.62 | 907 | Rams |
| 27 | 4 | Harrison Smith | 77.86 | 68.70 | 80.21 | 1025 | Vikings |
| 28 | 5 | Ricardo Allen | 77.42 | 80.20 | 78.17 | 205 | Falcons |
| 29 | 6 | Patrick Chung | 77.00 | 73.90 | 75.41 | 888 | Patriots |
| 30 | 7 | Tyrann Mathieu | 76.88 | 74.10 | 75.81 | 1045 | Texans |
| 31 | 8 | Tashaun Gipson Sr. | 76.64 | 71.90 | 75.64 | 1006 | Jaguars |
| 32 | 9 | Damarious Randall | 76.23 | 71.40 | 75.80 | 1083 | Browns |
| 33 | 10 | Xavier Woods | 75.91 | 75.20 | 73.91 | 883 | Cowboys |
| 34 | 11 | Marcus Williams | 75.81 | 69.50 | 75.85 | 956 | Saints |
| 35 | 12 | Will Parks | 75.66 | 75.70 | 71.67 | 572 | Broncos |
| 36 | 13 | Mike Mitchell | 75.19 | 75.90 | 75.33 | 224 | Colts |
| 37 | 14 | Duron Harmon | 75.15 | 68.40 | 75.48 | 636 | Patriots |
| 38 | 15 | Ibraheim Campbell | 75.05 | 75.60 | 80.62 | 114 | Packers |
| 39 | 16 | Antoine Bethea | 74.52 | 73.50 | 71.35 | 1111 | Cardinals |
| 40 | 17 | Sean Davis | 74.44 | 71.50 | 72.75 | 980 | Steelers |
| 41 | 18 | Deon Bush | 74.33 | 77.60 | 75.59 | 152 | Bears |
| 42 | 19 | Tony Jefferson | 74.18 | 72.50 | 72.60 | 862 | Ravens |
| 43 | 20 | Rodney McLeod | 74.16 | 78.20 | 74.06 | 162 | Eagles |

### Starter (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 44 | 1 | Reshad Jones | 73.96 | 71.70 | 74.44 | 825 | Dolphins |
| 45 | 2 | Marcus Maye | 73.85 | 71.20 | 77.96 | 393 | Jets |
| 46 | 3 | Andre Hal | 73.43 | 73.80 | 73.19 | 237 | Texans |
| 47 | 4 | Vonn Bell | 73.42 | 68.30 | 72.86 | 753 | Saints |
| 48 | 5 | Jordan Poyer | 73.33 | 70.90 | 72.87 | 1010 | Bills |
| 49 | 6 | Jordan Whitehead | 73.11 | 74.40 | 69.12 | 660 | Buccaneers |
| 50 | 7 | Quandre Diggs | 73.03 | 67.10 | 72.82 | 948 | Lions |
| 51 | 8 | Jarrod Wilson | 72.91 | 68.70 | 75.20 | 222 | Jaguars |
| 52 | 9 | Karl Joseph | 72.26 | 68.00 | 73.85 | 509 | Raiders |
| 53 | 10 | Adrian Phillips | 72.20 | 66.30 | 72.70 | 685 | Chargers |
| 54 | 11 | Landon Collins | 71.46 | 68.20 | 71.86 | 804 | Giants |
| 55 | 12 | Rashaan Gaulden | 71.22 | 68.40 | 74.13 | 143 | Panthers |
| 56 | 13 | Tavon Wilson | 71.06 | 67.20 | 72.39 | 304 | Lions |
| 57 | 14 | Michael Thomas | 70.83 | 70.10 | 69.96 | 522 | Giants |
| 58 | 15 | Eric Murray | 70.58 | 68.20 | 69.03 | 703 | Chiefs |
| 59 | 16 | Terrell Edmunds | 70.50 | 65.20 | 69.87 | 967 | Steelers |
| 60 | 17 | Ronnie Harrison | 70.47 | 68.90 | 69.43 | 328 | Jaguars |
| 61 | 18 | Andrew Sendejo | 70.26 | 70.70 | 72.25 | 326 | Vikings |
| 62 | 19 | Corey Graham | 69.72 | 63.60 | 71.20 | 655 | Eagles |
| 63 | 20 | George Iloka | 69.25 | 63.00 | 73.94 | 117 | Vikings |
| 64 | 21 | Justin Evans | 68.98 | 65.80 | 72.02 | 605 | Buccaneers |
| 65 | 22 | Glover Quin | 68.79 | 63.40 | 68.21 | 829 | Lions |
| 66 | 23 | Morgan Burnett | 68.75 | 62.80 | 72.40 | 390 | Steelers |
| 67 | 24 | Daniel Sorensen | 68.38 | 68.00 | 69.15 | 354 | Chiefs |
| 68 | 25 | T.J. McDonald | 68.35 | 66.70 | 68.84 | 952 | Dolphins |
| 69 | 26 | Andrew Adams | 67.53 | 69.00 | 66.55 | 370 | Buccaneers |
| 70 | 27 | Clayton Geathers | 67.53 | 65.10 | 71.96 | 715 | Colts |
| 71 | 28 | Eric Reid | 67.39 | 66.70 | 67.44 | 736 | Panthers |
| 72 | 29 | Kenny Vaccaro | 67.04 | 61.60 | 70.35 | 747 | Titans |
| 73 | 30 | Reggie Nelson | 66.87 | 63.40 | 67.62 | 370 | Raiders |
| 74 | 31 | Curtis Riley | 66.70 | 63.90 | 64.40 | 1048 | Giants |
| 75 | 32 | Doug Middleton | 65.88 | 61.90 | 73.12 | 231 | Jets |
| 76 | 33 | Tedric Thompson | 65.79 | 66.40 | 67.98 | 656 | Seahawks |
| 77 | 34 | Jordan Richards | 65.78 | 64.60 | 65.42 | 429 | Falcons |
| 78 | 35 | Matthias Farley | 65.40 | 66.00 | 68.44 | 151 | Colts |
| 79 | 36 | George Odum | 64.98 | 63.30 | 68.19 | 205 | Colts |
| 80 | 37 | Jahleel Addae | 64.73 | 60.10 | 65.32 | 1025 | Chargers |
| 81 | 38 | Sharrod Neasman | 64.62 | 67.50 | 67.50 | 436 | Falcons |
| 82 | 39 | Darian Stewart | 64.05 | 60.80 | 63.08 | 874 | Broncos |
| 83 | 40 | Antone Exum Jr. | 63.11 | 63.50 | 66.91 | 594 | 49ers |
| 84 | 41 | Josh Jones | 62.76 | 59.20 | 64.62 | 502 | Packers |
| 85 | 42 | Jeff Heath | 62.21 | 59.60 | 60.71 | 1000 | Cowboys |

### Rotation/backup (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 86 | 1 | Jaquiski Tartt | 61.79 | 58.50 | 66.39 | 437 | 49ers |
| 87 | 2 | Chuck Clark | 61.20 | 58.10 | 62.75 | 255 | Ravens |
| 88 | 3 | Justin Simmons | 61.08 | 51.20 | 65.26 | 1078 | Broncos |
| 89 | 4 | Marcus Gilchrist | 60.79 | 53.50 | 62.12 | 871 | Raiders |
| 90 | 5 | Marqui Christian | 60.05 | 58.80 | 61.10 | 349 | Rams |
| 91 | 6 | Jermaine Whitehead | 60.03 | 58.10 | 62.98 | 228 | Browns |
| 92 | 7 | Delano Hill | 59.84 | 56.10 | 64.41 | 320 | Seahawks |
| 93 | 8 | Tre Sullivan | 59.72 | 63.80 | 58.04 | 222 | Eagles |
| 94 | 9 | Deshazor Everett | 59.30 | 54.20 | 62.50 | 136 | Commanders |
| 95 | 10 | Kurt Coleman | 59.19 | 53.70 | 59.83 | 359 | Saints |
| 96 | 11 | Sean Chandler | 58.22 | 57.30 | 59.86 | 142 | Giants |
| 97 | 12 | Jordan Lucas | 58.02 | 61.00 | 62.02 | 262 | Chiefs |
| 98 | 13 | Jimmie Ward | 57.78 | 51.30 | 64.39 | 388 | 49ers |
| 99 | 14 | Chris Conte | 57.64 | 57.00 | 61.08 | 118 | Buccaneers |
| 100 | 15 | Mike Adams | 57.00 | 42.10 | 62.97 | 938 | Panthers |
| 101 | 16 | Derrick Kindred | 56.89 | 52.60 | 57.04 | 498 | Browns |
| 102 | 17 | Kentrell Brice | 55.49 | 52.10 | 57.75 | 648 | Packers |
| 103 | 18 | Kendrick Lewis | 54.80 | 49.10 | 62.03 | 276 | Titans |
| 104 | 19 | Kavon Frazier | 50.13 | 44.40 | 55.72 | 186 | Cowboys |
| 105 | 20 | Montae Nicholson | 48.41 | 42.80 | 55.02 | 467 | Commanders |
| 106 | 21 | Marcell Harris | 46.92 | 49.20 | 52.10 | 358 | 49ers |
| 107 | 22 | Isaiah Johnson | 46.42 | 37.80 | 57.17 | 404 | Buccaneers |
| 108 | 23 | Tyvis Powell | 45.00 | 30.60 | 56.29 | 104 | 49ers |
| 109 | 24 | Su'a Cravens | 45.00 | 29.20 | 55.24 | 117 | Broncos |
| 110 | 25 | Adrian Colbert | 45.00 | 32.10 | 56.23 | 320 | 49ers |

## T — Tackle

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (35 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Terron Armstead | 94.51 | 90.40 | 93.09 | 602 | Saints |
| 2 | 2 | Rob Havenstein | 91.98 | 86.10 | 91.73 | 1101 | Rams |
| 3 | 3 | David Bakhtiari | 90.88 | 88.90 | 88.04 | 1032 | Packers |
| 4 | 4 | Joe Staley | 90.13 | 84.70 | 89.59 | 1006 | 49ers |
| 5 | 5 | Mitchell Schwartz | 89.41 | 84.20 | 88.72 | 1045 | Chiefs |
| 6 | 6 | Andrew Whitworth | 89.11 | 83.60 | 88.61 | 1038 | Rams |
| 7 | 7 | Ryan Ramczyk | 89.05 | 83.50 | 88.59 | 996 | Saints |
| 8 | 8 | Duane Brown | 88.50 | 83.00 | 88.00 | 1067 | Seahawks |
| 9 | 9 | Tyron Smith | 87.89 | 80.90 | 88.39 | 849 | Cowboys |
| 10 | 10 | Lane Johnson | 87.17 | 81.00 | 87.11 | 962 | Eagles |
| 11 | 11 | Alejandro Villanueva | 85.88 | 81.20 | 84.83 | 1116 | Steelers |
| 12 | 12 | Jake Matthews | 85.22 | 80.00 | 84.54 | 1057 | Falcons |
| 13 | 13 | Ronnie Stanley | 85.12 | 77.50 | 86.03 | 1084 | Ravens |
| 14 | 14 | Dennis Kelly | 84.85 | 75.40 | 86.99 | 376 | Titans |
| 15 | 15 | Russell Okung | 84.73 | 78.40 | 84.79 | 866 | Chargers |
| 16 | 16 | Anthony Castonzo | 84.61 | 77.70 | 85.05 | 744 | Colts |
| 17 | 17 | George Fant | 84.10 | 74.00 | 86.67 | 371 | Seahawks |
| 18 | 18 | Taylor Lewan | 84.01 | 76.40 | 84.92 | 852 | Titans |
| 19 | 19 | Mike McGlinchey | 83.85 | 74.80 | 85.72 | 1055 | 49ers |
| 20 | 20 | Trent Williams | 83.52 | 75.60 | 84.63 | 792 | Commanders |
| 21 | 21 | Laremy Tunsil | 83.37 | 74.50 | 85.11 | 820 | Dolphins |
| 22 | 22 | Charles Leno Jr. | 83.30 | 75.70 | 84.20 | 1067 | Bears |
| 23 | 23 | Taylor Moton | 83.11 | 76.60 | 83.29 | 1054 | Panthers |
| 24 | 24 | Marcus Cannon | 82.76 | 73.70 | 84.64 | 836 | Patriots |
| 25 | 25 | Nate Solder | 82.60 | 75.70 | 83.04 | 1027 | Giants |
| 26 | 26 | Garett Bolles | 82.32 | 72.80 | 84.50 | 1062 | Broncos |
| 27 | 27 | Eric Fisher | 81.97 | 73.40 | 83.52 | 1042 | Chiefs |
| 28 | 28 | Bryan Bulaga | 81.95 | 75.00 | 82.42 | 781 | Packers |
| 29 | 29 | La'el Collins | 81.55 | 72.50 | 83.42 | 1075 | Cowboys |
| 30 | 30 | Riley Reiff | 81.47 | 74.10 | 82.21 | 793 | Vikings |
| 31 | 31 | Ja'Wuan James | 81.25 | 72.40 | 82.98 | 816 | Dolphins |
| 32 | 32 | Ty Nsekhe | 80.67 | 67.80 | 85.09 | 403 | Commanders |
| 33 | 33 | Bobby Massie | 80.52 | 71.80 | 82.17 | 1070 | Bears |
| 34 | 34 | Jason Peters | 80.34 | 71.10 | 82.33 | 868 | Eagles |
| 35 | 35 | Matt Feiler | 80.01 | 71.70 | 81.39 | 675 | Steelers |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 36 | 1 | D.J. Humphries | 79.97 | 70.10 | 82.38 | 522 | Cardinals |
| 37 | 2 | Ty Sambrailo | 79.80 | 66.60 | 84.44 | 266 | Falcons |
| 38 | 3 | Trent Brown | 79.18 | 69.70 | 81.34 | 1090 | Patriots |
| 39 | 4 | Rick Wagner | 79.14 | 71.40 | 80.13 | 985 | Lions |
| 40 | 5 | Dion Dawkins | 78.90 | 69.90 | 80.73 | 1059 | Bills |
| 41 | 6 | Ryan Schraeder | 78.35 | 66.00 | 82.42 | 865 | Falcons |
| 42 | 7 | Marcus Gilbert | 78.22 | 69.60 | 79.80 | 362 | Steelers |
| 43 | 8 | Taylor Decker | 78.07 | 70.60 | 78.88 | 1062 | Lions |
| 44 | 9 | Jack Conklin | 77.87 | 66.80 | 81.08 | 498 | Titans |
| 45 | 10 | Kelvin Beachum | 77.86 | 68.50 | 79.94 | 1001 | Jets |
| 46 | 11 | Jared Veldheer | 77.37 | 64.80 | 81.59 | 704 | Broncos |
| 47 | 12 | Morgan Moses | 77.08 | 64.40 | 81.37 | 965 | Commanders |
| 48 | 13 | Le'Raven Clark | 76.92 | 67.60 | 78.97 | 365 | Colts |
| 49 | 14 | Jason Spriggs | 76.82 | 64.70 | 80.73 | 292 | Packers |
| 50 | 15 | Jermey Parnell | 76.67 | 65.70 | 79.82 | 869 | Jaguars |
| 51 | 16 | Demar Dotson | 76.67 | 67.40 | 78.68 | 1005 | Buccaneers |
| 52 | 17 | Brandon Shell | 75.64 | 63.70 | 79.43 | 850 | Jets |
| 53 | 18 | Ereck Flowers | 75.24 | 65.40 | 77.63 | 588 | Jaguars |
| 54 | 19 | Donovan Smith | 74.94 | 66.40 | 76.47 | 1117 | Buccaneers |
| 55 | 20 | Joe Barksdale | 74.79 | 62.60 | 78.75 | 388 | Cardinals |
| 56 | 21 | Orlando Brown Jr. | 74.42 | 67.80 | 74.67 | 760 | Ravens |
| 57 | 22 | Kendall Lamm | 74.39 | 64.40 | 76.88 | 859 | Texans |
| 58 | 23 | Chris Hubbard | 74.20 | 65.10 | 76.10 | 1091 | Browns |

### Starter (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | James Hurst | 73.14 | 60.70 | 77.26 | 675 | Ravens |
| 60 | 2 | Josh Wells | 72.96 | 62.10 | 76.04 | 305 | Jaguars |
| 61 | 3 | Chris Clark | 72.90 | 61.70 | 76.20 | 818 | Panthers |
| 62 | 4 | Greg Robinson | 72.66 | 60.30 | 76.73 | 498 | Browns |
| 63 | 5 | Cordy Glenn | 72.32 | 61.40 | 75.44 | 765 | Bengals |
| 64 | 6 | Andre Smith | 72.26 | 58.80 | 77.07 | 452 | Bengals |
| 65 | 7 | Joe Haeg | 72.21 | 59.00 | 76.85 | 367 | Colts |
| 66 | 8 | Korey Cunningham | 71.83 | 60.40 | 75.28 | 349 | Cardinals |
| 67 | 9 | Bobby Hart | 70.95 | 56.70 | 76.28 | 994 | Bengals |
| 68 | 10 | John Wetzel | 70.93 | 57.10 | 75.98 | 339 | Cardinals |
| 69 | 11 | Cam Fleming | 70.92 | 56.50 | 76.37 | 232 | Cowboys |
| 70 | 12 | Germain Ifedi | 70.72 | 56.50 | 76.03 | 989 | Seahawks |
| 71 | 13 | Rashod Hill | 70.71 | 58.50 | 74.69 | 529 | Vikings |
| 72 | 14 | Desmond Harrison | 70.65 | 57.30 | 75.38 | 595 | Browns |
| 73 | 15 | Brian O'Neill | 70.53 | 63.00 | 71.39 | 800 | Vikings |
| 74 | 16 | Marshall Newhouse | 70.49 | 55.80 | 76.12 | 211 | Panthers |
| 75 | 17 | Jordan Mills | 70.14 | 56.50 | 75.06 | 1013 | Bills |
| 76 | 18 | Sam Tevi | 68.58 | 52.30 | 75.26 | 871 | Chargers |
| 77 | 19 | Kevin Pamphile | 67.29 | 55.60 | 70.91 | 155 | Titans |
| 78 | 20 | Julie'n Davenport | 66.96 | 52.70 | 72.30 | 1014 | Texans |
| 79 | 21 | LaAdrian Waddle | 66.61 | 51.80 | 72.32 | 342 | Patriots |
| 80 | 22 | Donald Penn | 65.62 | 48.50 | 72.87 | 188 | Raiders |
| 81 | 23 | Halapoulivaati Vaitai | 65.33 | 49.90 | 71.45 | 334 | Eagles |
| 82 | 24 | Brandon Parker | 65.26 | 47.30 | 73.07 | 780 | Raiders |
| 83 | 25 | Chad Wheeler | 65.14 | 48.90 | 71.80 | 857 | Giants |
| 84 | 26 | Kolton Miller | 64.73 | 49.60 | 70.65 | 1008 | Raiders |
| 85 | 27 | Chukwuma Okorafor | 64.52 | 50.80 | 69.50 | 156 | Steelers |
| 86 | 28 | Brent Qvale | 63.99 | 43.80 | 73.28 | 159 | Jets |
| 87 | 29 | Sam Young | 63.95 | 45.80 | 71.89 | 121 | Dolphins |
| 88 | 30 | Will Holden | 63.28 | 45.70 | 70.83 | 294 | Cardinals |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 89 | 1 | Leonard Wester | 59.40 | 46.70 | 63.70 | 118 | Buccaneers |

## TE — Tight End

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | O.J. Howard | 87.62 | 88.90 | 82.60 | 291 | Buccaneers |
| 2 | 2 | Travis Kelce | 84.70 | 88.00 | 78.33 | 661 | Chiefs |
| 3 | 3 | George Kittle | 84.14 | 89.70 | 76.27 | 567 | 49ers |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Rob Gronkowski | 79.70 | 73.70 | 79.53 | 477 | Patriots |
| 5 | 2 | Anthony Firkser | 78.74 | 69.60 | 80.66 | 139 | Titans |
| 6 | 3 | Zach Ertz | 78.20 | 75.60 | 75.76 | 657 | Eagles |
| 7 | 4 | Chris Herndon | 77.96 | 74.80 | 75.90 | 369 | Jets |
| 8 | 5 | Mark Andrews | 77.66 | 75.70 | 74.80 | 289 | Ravens |
| 9 | 6 | Gerald Everett | 77.32 | 82.40 | 69.76 | 278 | Rams |
| 10 | 7 | Evan Engram | 77.08 | 76.30 | 73.44 | 344 | Giants |
| 11 | 8 | Antonio Gates | 75.05 | 69.80 | 74.38 | 283 | Chargers |
| 12 | 9 | Jared Cook | 74.70 | 75.50 | 70.00 | 538 | Raiders |
| 13 | 10 | Jack Doyle | 74.37 | 72.50 | 71.45 | 178 | Colts |
| 14 | 11 | Benjamin Watson | 74.31 | 71.40 | 72.08 | 302 | Saints |

### Starter (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Jordan Reed | 73.71 | 73.10 | 69.95 | 370 | Commanders |
| 16 | 2 | Dallas Goedert | 73.66 | 74.00 | 69.27 | 295 | Eagles |
| 17 | 3 | Luke Stocker | 73.48 | 76.10 | 67.56 | 124 | Titans |
| 18 | 4 | Greg Olsen | 73.45 | 65.80 | 74.39 | 274 | Panthers |
| 19 | 5 | Lee Smith | 73.28 | 78.40 | 65.70 | 106 | Raiders |
| 20 | 6 | Vance McDonald | 73.11 | 69.30 | 71.49 | 444 | Steelers |
| 21 | 7 | Tyler Eifert | 72.94 | 72.00 | 69.40 | 103 | Bengals |
| 22 | 8 | Tyler Higbee | 72.88 | 71.40 | 69.70 | 391 | Rams |
| 23 | 9 | Eric Ebron | 72.73 | 67.40 | 72.11 | 493 | Colts |
| 24 | 10 | Levine Toilolo | 72.24 | 66.10 | 72.17 | 265 | Lions |
| 25 | 11 | Vernon Davis | 72.20 | 61.00 | 75.50 | 244 | Commanders |
| 26 | 12 | Trey Burton | 72.09 | 69.40 | 69.71 | 549 | Bears |
| 27 | 13 | Austin Hooper | 72.08 | 67.80 | 70.76 | 566 | Falcons |
| 28 | 14 | Hayden Hurst | 71.86 | 66.40 | 71.33 | 128 | Ravens |
| 29 | 15 | Maxx Williams | 71.74 | 69.70 | 68.93 | 105 | Ravens |
| 30 | 16 | Darren Fells | 71.60 | 65.00 | 71.84 | 165 | Browns |
| 31 | 17 | Dalton Schultz | 71.54 | 65.10 | 71.66 | 143 | Cowboys |
| 32 | 18 | Jesse James | 71.05 | 65.20 | 70.78 | 339 | Steelers |
| 33 | 19 | Austin Seferian-Jenkins | 70.71 | 69.10 | 67.61 | 138 | Jaguars |
| 34 | 20 | Nick O'Leary | 70.52 | 63.20 | 71.23 | 167 | Dolphins |
| 35 | 21 | Blake Jarwin | 70.45 | 68.80 | 67.39 | 253 | Cowboys |
| 36 | 22 | Nick Boyle | 70.43 | 67.80 | 68.02 | 239 | Ravens |
| 37 | 23 | Garrett Celek | 70.26 | 62.70 | 71.13 | 100 | 49ers |
| 38 | 24 | Jordan Akins | 69.40 | 63.80 | 68.97 | 209 | Texans |
| 39 | 25 | David Njoku | 69.09 | 66.30 | 66.79 | 586 | Browns |
| 40 | 26 | Rhett Ellison | 68.95 | 64.40 | 67.81 | 318 | Giants |
| 41 | 27 | Jeff Heuerman | 68.91 | 61.30 | 69.82 | 321 | Broncos |
| 42 | 28 | Jimmy Graham | 68.81 | 59.60 | 70.78 | 611 | Packers |
| 43 | 29 | Virgil Green | 68.60 | 58.40 | 71.24 | 307 | Chargers |
| 44 | 30 | Ricky Seals-Jones | 68.49 | 52.80 | 74.78 | 402 | Cardinals |
| 45 | 31 | Charles Clay | 68.38 | 56.60 | 72.06 | 283 | Bills |
| 46 | 32 | Kyle Rudolph | 68.27 | 63.60 | 67.21 | 647 | Vikings |
| 47 | 33 | Blake Bell | 67.64 | 55.80 | 71.37 | 123 | Jaguars |
| 48 | 34 | Logan Paulsen | 67.63 | 62.00 | 67.22 | 163 | Falcons |
| 49 | 35 | Demetrius Harris | 67.49 | 61.10 | 67.58 | 220 | Chiefs |
| 50 | 36 | Cameron Brate | 67.41 | 53.10 | 72.78 | 401 | Buccaneers |
| 51 | 37 | Jordan Leggett | 67.17 | 65.70 | 63.98 | 171 | Jets |
| 52 | 38 | Luke Willson | 66.77 | 54.60 | 70.71 | 221 | Lions |
| 53 | 39 | Michael Roberts | 66.63 | 55.60 | 69.81 | 101 | Lions |
| 54 | 40 | Lance Kendricks | 66.60 | 59.60 | 67.10 | 154 | Packers |
| 55 | 41 | Matt LaCosse | 66.58 | 57.10 | 68.74 | 251 | Broncos |
| 56 | 42 | Nick Vannett | 66.40 | 64.00 | 63.84 | 265 | Seahawks |
| 57 | 43 | Ed Dickson | 66.32 | 62.40 | 64.77 | 198 | Seahawks |
| 58 | 44 | Derek Carrier | 65.63 | 59.10 | 65.82 | 111 | Raiders |
| 59 | 45 | Jermaine Gresham | 65.51 | 49.70 | 71.88 | 200 | Cardinals |
| 60 | 46 | Jason Croom | 65.44 | 52.10 | 70.16 | 220 | Bills |
| 61 | 47 | Jordan Thomas | 64.79 | 53.50 | 68.15 | 239 | Texans |
| 62 | 48 | James O'Shaughnessy | 64.65 | 60.10 | 63.52 | 311 | Jaguars |
| 63 | 49 | Ian Thomas | 64.57 | 53.30 | 67.91 | 349 | Panthers |
| 64 | 50 | Ryan Griffin | 64.39 | 56.00 | 65.81 | 452 | Texans |
| 65 | 51 | Scott Simonson | 64.20 | 57.10 | 64.76 | 129 | Giants |
| 66 | 52 | Josh Hill | 64.16 | 55.40 | 65.83 | 275 | Saints |
| 67 | 53 | Geoff Swaim | 63.87 | 55.80 | 65.09 | 291 | Cowboys |
| 68 | 54 | C.J. Uzomah | 63.59 | 56.80 | 63.95 | 520 | Bengals |
| 69 | 55 | Dwayne Allen | 63.19 | 51.30 | 66.95 | 149 | Patriots |
| 70 | 56 | Jonnu Smith | 63.19 | 55.10 | 64.41 | 308 | Titans |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 71 | 1 | Eric Tomlinson | 61.69 | 49.90 | 65.38 | 151 | Jets |
| 72 | 2 | Mike Gesicki | 60.92 | 50.20 | 63.90 | 273 | Dolphins |
| 73 | 3 | Jeremy Sprinkle | 59.25 | 45.70 | 64.12 | 159 | Commanders |
| 74 | 4 | Antony Auclair | 59.20 | 45.20 | 64.36 | 137 | Buccaneers |
| 75 | 5 | Logan Thomas | 58.36 | 51.70 | 58.64 | 167 | Bills |

## WR — Wide Receiver

- **Season used:** `2018`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Julio Jones | 89.25 | 90.90 | 83.98 | 605 | Falcons |
| 2 | 2 | Tyreek Hill | 88.10 | 89.60 | 82.94 | 603 | Chiefs |
| 3 | 3 | DeAndre Hopkins | 87.62 | 92.40 | 80.26 | 673 | Texans |
| 4 | 4 | Michael Thomas | 86.71 | 91.30 | 79.49 | 562 | Saints |
| 5 | 5 | T.Y. Hilton | 86.56 | 86.60 | 82.36 | 535 | Colts |
| 6 | 6 | Keenan Allen | 86.42 | 90.30 | 79.67 | 509 | Chargers |
| 7 | 7 | Robert Woods | 85.62 | 87.90 | 79.93 | 645 | Rams |
| 8 | 8 | Odell Beckham Jr. | 85.50 | 90.00 | 78.33 | 488 | Giants |
| 9 | 9 | Adam Thielen | 84.81 | 89.40 | 77.59 | 694 | Vikings |
| 10 | 10 | A.J. Green | 84.42 | 85.20 | 79.74 | 313 | Bengals |
| 11 | 11 | Mike Evans | 83.69 | 84.30 | 79.11 | 679 | Buccaneers |
| 12 | 12 | Davante Adams | 83.23 | 87.80 | 76.01 | 694 | Packers |
| 13 | 13 | DeSean Jackson | 82.82 | 79.40 | 80.94 | 360 | Buccaneers |
| 14 | 14 | Antonio Brown | 82.79 | 79.30 | 80.95 | 716 | Steelers |
| 15 | 15 | Kenny Golladay | 82.69 | 81.00 | 79.65 | 609 | Lions |
| 16 | 16 | JuJu Smith-Schuster | 82.59 | 81.80 | 78.95 | 728 | Steelers |
| 17 | 17 | Chris Godwin | 82.49 | 80.40 | 79.71 | 483 | Buccaneers |
| 18 | 18 | Tyler Lockett | 82.22 | 80.30 | 79.33 | 494 | Seahawks |
| 19 | 19 | Brandin Cooks | 82.06 | 79.70 | 79.46 | 616 | Rams |
| 20 | 20 | Robert Foster | 81.91 | 72.10 | 84.28 | 281 | Bills |
| 21 | 21 | Tyler Boyd | 81.89 | 84.50 | 75.99 | 534 | Bengals |
| 22 | 22 | Doug Baldwin | 80.84 | 80.60 | 76.83 | 379 | Seahawks |
| 23 | 23 | Albert Wilson | 80.81 | 82.50 | 75.52 | 141 | Dolphins |
| 24 | 24 | Cooper Kupp | 80.54 | 77.30 | 78.53 | 271 | Rams |
| 25 | 25 | Josh Gordon | 80.46 | 73.50 | 80.94 | 403 | Patriots |
| 26 | 26 | Will Fuller V | 80.24 | 81.20 | 75.44 | 250 | Texans |
| 27 | 27 | Emmanuel Sanders | 80.06 | 81.10 | 75.20 | 439 | Broncos |

### Good (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Stefon Diggs | 79.80 | 81.10 | 74.76 | 624 | Vikings |
| 29 | 2 | Dante Pettis | 79.70 | 68.50 | 83.00 | 283 | 49ers |
| 30 | 3 | Mike Williams | 79.67 | 81.80 | 74.09 | 408 | Chargers |
| 31 | 4 | Alshon Jeffery | 79.33 | 78.30 | 75.85 | 506 | Eagles |
| 32 | 5 | Amari Cooper | 79.21 | 77.90 | 75.91 | 583 | Cowboys |
| 33 | 6 | DJ Moore | 78.48 | 71.80 | 78.76 | 451 | Panthers |
| 34 | 7 | Corey Davis | 77.98 | 76.40 | 74.87 | 515 | Titans |
| 35 | 8 | Sammy Watkins | 77.90 | 73.50 | 76.67 | 308 | Chiefs |
| 36 | 9 | Marvin Jones Jr. | 77.66 | 71.90 | 77.33 | 365 | Lions |
| 37 | 10 | Jakeem Grant Sr. | 77.05 | 70.50 | 77.25 | 170 | Dolphins |
| 38 | 11 | Allen Robinson II | 76.59 | 74.70 | 73.69 | 480 | Bears |
| 39 | 12 | Julian Edelman | 76.51 | 75.20 | 73.21 | 457 | Patriots |
| 40 | 13 | Christian Kirk | 76.43 | 70.30 | 76.35 | 364 | Cardinals |
| 41 | 14 | Jarvis Landry | 76.02 | 74.70 | 72.73 | 643 | Browns |
| 42 | 15 | Mohamed Sanu | 75.75 | 71.60 | 74.35 | 597 | Falcons |
| 43 | 16 | Equanimeous St. Brown | 75.66 | 64.30 | 79.07 | 265 | Packers |
| 44 | 17 | Breshad Perriman | 75.58 | 70.90 | 74.54 | 145 | Browns |
| 45 | 18 | Adam Humphries | 75.55 | 75.20 | 71.62 | 601 | Buccaneers |
| 46 | 19 | Ted Ginn Jr. | 75.47 | 67.20 | 76.82 | 151 | Saints |
| 47 | 20 | Cody Latimer | 75.39 | 70.20 | 74.68 | 149 | Giants |
| 48 | 21 | Tyrell Williams | 75.02 | 66.80 | 76.34 | 496 | Chargers |
| 49 | 22 | Golden Tate | 75.01 | 70.10 | 74.11 | 421 | Eagles |
| 50 | 23 | Keith Kirkwood | 74.94 | 66.10 | 76.67 | 140 | Saints |
| 51 | 24 | Josh Reynolds | 74.78 | 68.90 | 74.53 | 367 | Rams |
| 52 | 25 | Tre'Quan Smith | 74.78 | 68.70 | 74.67 | 340 | Saints |
| 53 | 26 | Cole Beasley | 74.77 | 74.80 | 70.58 | 510 | Cowboys |
| 54 | 27 | Cordarrelle Patterson | 74.76 | 72.60 | 72.03 | 100 | Patriots |
| 55 | 28 | David Moore | 74.54 | 65.40 | 76.47 | 346 | Seahawks |
| 56 | 29 | DeVante Parker | 74.49 | 69.20 | 73.85 | 245 | Dolphins |
| 57 | 30 | Larry Fitzgerald | 74.49 | 73.20 | 71.19 | 561 | Cardinals |
| 58 | 31 | Marquise Goodwin | 74.25 | 62.70 | 77.78 | 263 | 49ers |
| 59 | 32 | Jordy Nelson | 74.06 | 70.70 | 72.13 | 556 | Raiders |
| 60 | 33 | Taywan Taylor | 74.05 | 70.00 | 72.59 | 267 | Titans |
| 61 | 34 | Calvin Ridley | 74.04 | 68.80 | 73.36 | 490 | Falcons |

### Starter (82 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 62 | 1 | Taylor Gabriel | 73.93 | 68.90 | 73.11 | 513 | Bears |
| 63 | 2 | Demaryius Thomas | 73.75 | 71.50 | 71.09 | 495 | Texans |
| 64 | 3 | Pierre Garcon | 73.52 | 64.30 | 75.50 | 240 | 49ers |
| 65 | 4 | Curtis Samuel | 73.20 | 71.50 | 70.17 | 318 | Panthers |
| 66 | 5 | Devin Funchess | 73.19 | 68.40 | 72.21 | 429 | Panthers |
| 67 | 6 | Jaron Brown | 73.15 | 66.00 | 73.75 | 165 | Seahawks |
| 68 | 7 | Kenny Stills | 73.13 | 64.40 | 74.79 | 466 | Dolphins |
| 69 | 8 | John Brown | 73.12 | 65.90 | 73.76 | 514 | Ravens |
| 70 | 9 | Rashard Higgins | 73.05 | 68.10 | 72.19 | 344 | Browns |
| 71 | 10 | Martavis Bryant | 72.77 | 63.70 | 74.65 | 153 | Raiders |
| 72 | 11 | Dede Westbrook | 72.76 | 71.30 | 69.56 | 576 | Jaguars |
| 73 | 12 | Deonte Thompson | 72.55 | 68.80 | 70.88 | 142 | Bills |
| 74 | 13 | Quincy Enunwa | 72.45 | 68.00 | 71.25 | 336 | Jets |
| 75 | 14 | Danny Amendola | 72.21 | 66.60 | 71.79 | 443 | Dolphins |
| 76 | 15 | Travis Benjamin | 72.17 | 61.40 | 75.19 | 192 | Chargers |
| 77 | 16 | DeAndre Carter | 72.07 | 64.00 | 73.29 | 174 | Texans |
| 78 | 17 | Willie Snead IV | 72.06 | 67.10 | 71.20 | 474 | Ravens |
| 79 | 18 | Sterling Shepard | 72.02 | 67.30 | 71.00 | 627 | Giants |
| 80 | 19 | Allen Hurns | 71.89 | 64.50 | 72.65 | 268 | Cowboys |
| 81 | 20 | Paul Richardson Jr. | 71.49 | 64.70 | 71.85 | 252 | Commanders |
| 82 | 21 | Michael Gallup | 71.39 | 60.30 | 74.62 | 468 | Cowboys |
| 83 | 22 | Courtland Sutton | 71.38 | 64.00 | 72.14 | 574 | Broncos |
| 84 | 23 | Brandon LaFell | 71.31 | 64.50 | 71.69 | 135 | Raiders |
| 85 | 24 | Dontrelle Inman | 71.24 | 67.80 | 69.36 | 242 | Colts |
| 86 | 25 | Marquez Valdes-Scantling | 71.22 | 60.40 | 74.26 | 507 | Packers |
| 87 | 26 | Keke Coutee | 71.20 | 64.70 | 71.36 | 175 | Texans |
| 88 | 27 | Jamison Crowder | 71.17 | 60.90 | 73.85 | 289 | Commanders |
| 89 | 28 | Phillip Dorsett | 71.06 | 67.00 | 69.60 | 236 | Patriots |
| 90 | 29 | Antonio Callaway | 70.89 | 64.00 | 71.31 | 533 | Browns |
| 91 | 30 | Kendrick Bourne | 70.88 | 66.60 | 69.57 | 397 | 49ers |
| 92 | 31 | Jordan Matthews | 70.87 | 65.40 | 70.35 | 228 | Eagles |
| 93 | 32 | Nelson Agholor | 70.47 | 64.80 | 70.09 | 631 | Eagles |
| 94 | 33 | Donte Moncrief | 70.43 | 62.30 | 71.68 | 567 | Jaguars |
| 95 | 34 | Keelan Cole Sr. | 70.25 | 59.40 | 73.32 | 490 | Jaguars |
| 96 | 35 | Chris Hogan | 70.14 | 55.60 | 75.67 | 520 | Patriots |
| 97 | 36 | Kelvin Benjamin | 70.10 | 57.80 | 74.13 | 353 | Chiefs |
| 98 | 37 | Cameron Batson | 70.07 | 62.30 | 71.09 | 115 | Titans |
| 99 | 38 | Geronimo Allison | 70.01 | 61.40 | 71.59 | 179 | Packers |
| 100 | 39 | Josh Doctson | 69.93 | 63.30 | 70.18 | 548 | Commanders |
| 101 | 40 | Marvin Hall | 69.88 | 55.90 | 75.03 | 122 | Falcons |
| 102 | 41 | Jermaine Kearse | 69.75 | 62.70 | 70.28 | 404 | Jets |
| 103 | 42 | Jarius Wright | 69.58 | 63.70 | 69.34 | 408 | Panthers |
| 104 | 43 | Jake Kumerow | 69.57 | 58.90 | 72.52 | 111 | Packers |
| 105 | 44 | Randall Cobb | 69.38 | 60.20 | 71.34 | 358 | Packers |
| 106 | 45 | Isaiah McKenzie | 69.34 | 67.00 | 66.73 | 159 | Bills |
| 107 | 46 | Richie James | 69.32 | 57.80 | 72.83 | 128 | 49ers |
| 108 | 47 | Aldrick Robinson | 69.29 | 61.80 | 70.11 | 187 | Vikings |
| 109 | 48 | Bruce Ellington | 69.01 | 68.30 | 65.32 | 188 | Lions |
| 110 | 49 | Tim Patrick | 68.84 | 60.20 | 70.44 | 261 | Broncos |
| 111 | 50 | Anthony Miller | 68.80 | 60.40 | 70.23 | 384 | Bears |
| 112 | 51 | Tajae Sharpe | 68.71 | 61.70 | 69.22 | 351 | Titans |
| 113 | 52 | Rishard Matthews | 68.64 | 50.10 | 76.84 | 107 | Jets |
| 114 | 53 | Russell Shepard | 68.35 | 61.50 | 68.75 | 134 | Giants |
| 115 | 54 | Zay Jones | 67.79 | 62.20 | 67.35 | 595 | Bills |
| 116 | 55 | Michael Crabtree | 67.73 | 62.70 | 66.91 | 539 | Ravens |
| 117 | 56 | Josh Bellamy | 67.21 | 57.90 | 69.25 | 170 | Bears |
| 118 | 57 | Zach Pascal | 67.14 | 60.90 | 67.14 | 289 | Colts |
| 119 | 58 | Chester Rogers | 67.10 | 63.30 | 65.47 | 416 | Colts |
| 120 | 59 | Seth Roberts | 66.99 | 62.10 | 66.08 | 401 | Raiders |
| 121 | 60 | Justin Hardy | 66.83 | 60.50 | 66.88 | 121 | Falcons |
| 122 | 61 | Chris Moore | 66.75 | 60.00 | 67.09 | 214 | Ravens |
| 123 | 62 | Vyncint Smith | 66.73 | 58.20 | 68.25 | 129 | Texans |
| 124 | 63 | Bennie Fowler | 66.66 | 58.20 | 68.13 | 257 | Giants |
| 125 | 64 | Ryan Grant | 66.48 | 59.10 | 67.23 | 422 | Colts |
| 126 | 65 | Demarcus Robinson | 66.31 | 58.60 | 67.28 | 269 | Chiefs |
| 127 | 66 | Maurice Harris | 66.16 | 55.90 | 68.84 | 298 | Commanders |
| 128 | 67 | Alex Erickson | 66.12 | 59.10 | 66.64 | 239 | Bengals |
| 129 | 68 | Ryan Switzer | 65.84 | 63.00 | 63.56 | 239 | Steelers |
| 130 | 69 | Austin Carr | 65.60 | 55.00 | 68.50 | 161 | Saints |
| 131 | 70 | Trent Sherfield | 65.44 | 58.10 | 66.16 | 233 | Cardinals |
| 132 | 71 | Torrey Smith | 65.42 | 58.40 | 65.93 | 222 | Panthers |
| 133 | 72 | J.J. Nelson | 65.12 | 48.90 | 71.76 | 139 | Cardinals |
| 134 | 73 | Damion Ratley | 64.48 | 55.90 | 66.03 | 128 | Browns |
| 135 | 74 | James Washington | 64.47 | 49.20 | 70.48 | 402 | Steelers |
| 136 | 75 | Chris Conley | 64.19 | 55.80 | 65.61 | 555 | Chiefs |
| 137 | 76 | Trent Taylor | 64.00 | 56.20 | 65.04 | 257 | 49ers |
| 138 | 77 | DaeSean Hamilton | 63.94 | 60.10 | 62.33 | 292 | Broncos |
| 139 | 78 | Cody Core | 63.78 | 53.90 | 66.20 | 211 | Bengals |
| 140 | 79 | Andre Roberts | 63.49 | 58.10 | 62.92 | 121 | Jets |
| 141 | 80 | T.J. Jones | 63.08 | 50.40 | 67.36 | 346 | Lions |
| 142 | 81 | Andre Holmes | 62.21 | 54.20 | 63.38 | 182 | Broncos |
| 143 | 82 | Andy Jones | 62.18 | 61.20 | 58.67 | 142 | Lions |

### Rotation/backup (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 144 | 1 | Marcell Ateman | 61.59 | 53.50 | 62.81 | 232 | Raiders |
| 145 | 2 | Michael Floyd | 61.40 | 50.00 | 64.84 | 214 | Commanders |
| 146 | 3 | DJ Chark Jr. | 60.89 | 52.40 | 62.38 | 198 | Jaguars |
| 147 | 4 | Chad Williams | 60.81 | 53.70 | 61.38 | 299 | Cardinals |
| 148 | 5 | Darius Jennings | 60.51 | 55.50 | 59.69 | 109 | Titans |
| 149 | 6 | Laquon Treadwell | 59.97 | 51.30 | 61.59 | 386 | Vikings |
| 150 | 7 | John Ross | 58.06 | 49.40 | 59.67 | 400 | Bengals |
