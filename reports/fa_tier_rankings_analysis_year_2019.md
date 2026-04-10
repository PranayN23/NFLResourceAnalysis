# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:26:48Z
- **Requested analysis_year:** 2019 (clamped to 2019)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jason Kelce | 87.78 | 81.00 | 88.14 | 1163 | Eagles |
| 2 | 2 | Ryan Jensen | 86.47 | 79.30 | 87.08 | 1139 | Buccaneers |
| 3 | 3 | Erik McCoy | 85.28 | 76.20 | 87.17 | 1058 | Saints |
| 4 | 4 | Ben Jones | 84.43 | 76.70 | 85.41 | 918 | Titans |
| 5 | 5 | Brandon Linder | 84.23 | 75.30 | 86.02 | 1083 | Jaguars |
| 6 | 6 | Ryan Kelly | 82.46 | 73.00 | 84.60 | 1018 | Colts |
| 7 | 7 | Alex Mack | 81.74 | 72.10 | 84.00 | 1156 | Falcons |
| 8 | 8 | J.C. Tretter | 80.52 | 72.00 | 82.03 | 1039 | Browns |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Rodney Hudson | 79.88 | 71.00 | 81.63 | 904 | Raiders |
| 10 | 2 | Travis Frederick | 79.40 | 70.00 | 81.50 | 1116 | Cowboys |
| 11 | 3 | Matt Skura | 79.04 | 68.70 | 81.77 | 717 | Ravens |
| 12 | 4 | Corey Linsley | 78.92 | 69.80 | 80.84 | 950 | Packers |
| 13 | 5 | James Daniels | 78.87 | 69.90 | 80.69 | 1069 | Bears |
| 14 | 6 | Chase Roullier | 78.27 | 69.30 | 80.09 | 833 | Commanders |
| 15 | 7 | Mitch Morse | 76.52 | 66.30 | 79.16 | 909 | Bills |
| 16 | 8 | Ted Karras | 75.89 | 64.50 | 79.32 | 1041 | Patriots |
| 17 | 9 | Daniel Kilgore | 75.86 | 66.30 | 78.06 | 877 | Dolphins |
| 18 | 10 | Nick Martin | 75.75 | 65.50 | 78.41 | 1020 | Texans |
| 19 | 11 | Connor McGovern | 75.47 | 60.00 | 81.61 | 1013 | Broncos |
| 20 | 12 | Matt Paradis | 75.37 | 63.40 | 79.19 | 1094 | Panthers |

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Justin Britt | 73.64 | 62.00 | 77.24 | 504 | Seahawks |
| 22 | 2 | Weston Richburg | 73.50 | 62.50 | 76.66 | 835 | 49ers |
| 23 | 3 | Austin Reiter | 73.32 | 63.00 | 76.03 | 1045 | Chiefs |
| 24 | 4 | Trey Hopkins | 72.93 | 62.40 | 75.78 | 1097 | Bengals |
| 25 | 5 | Brian Allen | 72.38 | 58.60 | 77.40 | 563 | Rams |
| 26 | 6 | Garrett Bradbury | 70.44 | 58.10 | 74.50 | 989 | Vikings |
| 27 | 7 | Tony Bergstrom | 70.39 | 61.00 | 72.48 | 226 | Commanders |
| 28 | 8 | Ryan Kalil | 69.71 | 55.50 | 75.01 | 343 | Jets |
| 29 | 9 | A.Q. Shipley | 69.66 | 57.60 | 73.54 | 1041 | Cardinals |
| 30 | 10 | Mike Pouncey | 69.65 | 58.20 | 73.12 | 305 | Chargers |
| 31 | 11 | Jon Halapio | 68.71 | 56.30 | 72.81 | 980 | Giants |
| 32 | 12 | Scott Quessenberry | 68.54 | 58.40 | 71.14 | 625 | Chargers |
| 33 | 13 | Maurkice Pouncey | 65.62 | 51.50 | 70.86 | 777 | Steelers |
| 34 | 14 | Jonotthan Harrison | 64.95 | 51.40 | 69.81 | 679 | Jets |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Andre James | 45.54 | 23.40 | 56.13 | 116 | Raiders |

## CB — Cornerback

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Richard Sherman | 92.10 | 90.10 | 91.87 | 895 | 49ers |
| 2 | 2 | Quinton Dunbar | 89.99 | 89.50 | 91.99 | 613 | Commanders |
| 3 | 3 | Marcus Peters | 88.98 | 85.50 | 87.34 | 985 | Ravens |
| 4 | 4 | Stephon Gilmore | 88.32 | 85.70 | 85.90 | 952 | Patriots |
| 5 | 5 | Casey Hayward Jr. | 87.53 | 83.00 | 86.38 | 944 | Chargers |
| 6 | 6 | Tramon Williams | 86.32 | 82.20 | 85.53 | 761 | Packers |
| 7 | 7 | Darius Phillips | 83.46 | 83.80 | 87.53 | 108 | Bengals |
| 8 | 8 | Jamel Dean | 82.93 | 78.90 | 87.70 | 372 | Buccaneers |
| 9 | 9 | Steven Nelson | 81.97 | 80.30 | 80.69 | 1011 | Steelers |
| 10 | 10 | D.J. Reed | 81.78 | 80.30 | 82.50 | 125 | 49ers |
| 11 | 11 | Brian Poole | 81.55 | 80.00 | 80.50 | 750 | Jets |
| 12 | 12 | Tre'Davious White | 81.54 | 76.00 | 81.59 | 951 | Bills |
| 13 | 13 | Byron Jones | 81.47 | 74.80 | 82.26 | 917 | Cowboys |
| 14 | 14 | Jason McCourty | 81.42 | 77.60 | 82.30 | 474 | Patriots |
| 15 | 15 | Darious Williams | 81.09 | 81.90 | 89.80 | 221 | Rams |
| 16 | 16 | Marlon Humphrey | 81.06 | 76.20 | 80.77 | 959 | Ravens |
| 17 | 17 | Jaire Alexander | 80.87 | 77.50 | 80.12 | 1027 | Packers |
| 18 | 18 | Shaquill Griffin | 80.65 | 76.00 | 80.84 | 917 | Seahawks |
| 19 | 19 | Rashad Fenton | 80.48 | 78.40 | 86.03 | 166 | Chiefs |
| 20 | 20 | D.J. Hayden | 80.40 | 77.90 | 80.30 | 647 | Jaguars |

### Good (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Adoree' Jackson | 79.01 | 75.20 | 79.98 | 575 | Titans |
| 22 | 2 | Nickell Robey-Coleman | 78.71 | 74.10 | 77.61 | 708 | Rams |
| 23 | 3 | Mike Hilton | 78.28 | 73.00 | 77.95 | 671 | Steelers |
| 24 | 4 | Jonathan Jones | 77.98 | 71.40 | 78.93 | 619 | Patriots |
| 25 | 5 | Joe Haden | 77.72 | 71.30 | 78.98 | 1055 | Steelers |
| 26 | 6 | Marshon Lattimore | 77.60 | 68.70 | 80.62 | 820 | Saints |
| 27 | 7 | K'Waun Williams | 77.58 | 75.70 | 76.23 | 603 | 49ers |
| 28 | 8 | Chidobe Awuzie | 77.35 | 70.50 | 79.52 | 1020 | Cowboys |
| 29 | 9 | Cameron Sutton | 77.21 | 74.10 | 79.19 | 268 | Steelers |
| 30 | 10 | Troy Hill | 77.16 | 72.70 | 79.51 | 538 | Rams |
| 31 | 11 | Chandon Sullivan | 76.91 | 75.00 | 78.19 | 350 | Packers |
| 32 | 12 | J.C. Jackson | 76.66 | 68.40 | 79.17 | 682 | Patriots |
| 33 | 13 | Chris Jones | 76.52 | 72.60 | 81.21 | 275 | Cardinals |
| 34 | 14 | Denzel Ward | 76.31 | 72.70 | 78.34 | 748 | Browns |
| 35 | 15 | Carlton Davis III | 76.18 | 72.10 | 77.34 | 934 | Buccaneers |
| 36 | 16 | Jalen Ramsey | 76.06 | 68.70 | 78.89 | 780 | Rams |
| 37 | 17 | Darqueze Dennard | 75.98 | 76.00 | 76.39 | 495 | Bengals |
| 38 | 18 | Prince Amukamara | 75.78 | 69.60 | 76.98 | 891 | Bears |
| 39 | 19 | Emmanuel Moseley | 75.65 | 68.00 | 76.59 | 577 | 49ers |
| 40 | 20 | Tramaine Brock Sr. | 75.61 | 72.40 | 77.95 | 745 | Titans |
| 41 | 21 | Kevin Johnson | 75.40 | 76.30 | 76.66 | 335 | Bills |
| 42 | 22 | Desmond Trufant | 75.14 | 70.30 | 77.85 | 521 | Falcons |
| 43 | 23 | Marvell Tell III | 75.06 | 67.90 | 80.86 | 254 | Colts |
| 44 | 24 | Kenny Moore II | 74.81 | 71.20 | 77.64 | 631 | Colts |
| 45 | 25 | Chris Harris Jr. | 74.77 | 66.80 | 77.17 | 1044 | Broncos |
| 46 | 26 | Janoris Jenkins | 74.40 | 68.00 | 76.49 | 973 | Saints |
| 47 | 27 | Gareon Conley | 74.36 | 65.30 | 80.82 | 768 | Texans |
| 48 | 28 | Brandon Carr | 74.03 | 66.80 | 74.68 | 748 | Ravens |

### Starter (65 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 49 | 1 | Bradley Roby | 73.78 | 69.00 | 76.23 | 654 | Texans |
| 50 | 2 | Jourdan Lewis | 73.63 | 65.50 | 76.87 | 590 | Cowboys |
| 51 | 3 | Sean Murphy-Bunting | 73.48 | 66.40 | 75.06 | 686 | Buccaneers |
| 52 | 4 | James Bradberry | 73.45 | 65.40 | 75.17 | 1020 | Panthers |
| 53 | 5 | Amani Oruwariye | 73.29 | 75.30 | 78.65 | 215 | Lions |
| 54 | 6 | Mackensie Alexander | 72.98 | 64.10 | 76.61 | 534 | Vikings |
| 55 | 7 | Levi Wallace | 72.78 | 68.30 | 75.12 | 785 | Bills |
| 56 | 8 | Malcolm Butler | 72.63 | 64.20 | 77.73 | 579 | Titans |
| 57 | 9 | Maurice Canady | 72.05 | 70.60 | 76.98 | 397 | Jets |
| 58 | 10 | Johnathan Joseph | 71.97 | 65.00 | 74.11 | 622 | Texans |
| 59 | 11 | Patrick Peterson | 71.53 | 64.30 | 75.32 | 696 | Cardinals |
| 60 | 12 | Kyle Fuller | 71.18 | 58.70 | 75.34 | 1070 | Bears |
| 61 | 13 | Charvarius Ward | 71.06 | 65.60 | 75.22 | 1048 | Chiefs |
| 62 | 14 | Tye Smith | 70.78 | 71.20 | 73.74 | 210 | Titans |
| 63 | 15 | Jimmy Smith | 70.74 | 64.20 | 76.66 | 402 | Ravens |
| 64 | 16 | Sidney Jones IV | 70.74 | 68.30 | 76.63 | 293 | Eagles |
| 65 | 17 | Logan Ryan | 70.57 | 60.60 | 73.69 | 1098 | Titans |
| 66 | 18 | Rock Ya-Sin | 70.54 | 62.20 | 72.97 | 853 | Colts |
| 67 | 19 | Daryl Worley | 70.06 | 62.60 | 73.26 | 939 | Raiders |
| 68 | 20 | Trayvon Mullen | 69.85 | 59.00 | 72.92 | 675 | Raiders |
| 69 | 21 | Ross Cockrell | 69.85 | 61.30 | 73.37 | 733 | Panthers |
| 70 | 22 | Kevin Peterson | 69.71 | 61.30 | 77.30 | 255 | Cardinals |
| 71 | 23 | Desmond King II | 69.49 | 69.60 | 66.28 | 584 | Chargers |
| 72 | 24 | Morris Claiborne | 69.39 | 63.50 | 73.84 | 198 | Chiefs |
| 73 | 25 | De'Vante Bausby | 69.03 | 72.30 | 82.69 | 133 | Broncos |
| 74 | 26 | Darius Slay | 68.98 | 56.90 | 74.22 | 858 | Lions |
| 75 | 27 | Trae Waynes | 68.84 | 61.60 | 71.16 | 769 | Vikings |
| 76 | 28 | Justin Coleman | 68.82 | 58.80 | 71.54 | 963 | Lions |
| 77 | 29 | Anthony Brown | 68.63 | 61.70 | 73.05 | 282 | Cowboys |
| 78 | 30 | Blessuan Austin | 68.60 | 69.90 | 74.44 | 388 | Jets |
| 79 | 31 | Kevin King | 68.57 | 62.30 | 73.69 | 805 | Packers |
| 80 | 32 | Darryl Roberts | 68.52 | 58.50 | 73.74 | 713 | Jets |
| 81 | 33 | Terrance Mitchell | 68.22 | 61.10 | 75.88 | 329 | Browns |
| 82 | 34 | Javien Elliott | 67.98 | 65.80 | 70.69 | 439 | Panthers |
| 83 | 35 | Eric Rowe | 67.86 | 60.20 | 73.58 | 1072 | Dolphins |
| 84 | 36 | Eli Apple | 67.78 | 59.10 | 71.48 | 933 | Saints |
| 85 | 37 | Jimmy Moreland | 67.72 | 61.30 | 69.92 | 471 | Commanders |
| 86 | 38 | Nik Needham | 67.51 | 60.00 | 72.51 | 743 | Dolphins |
| 87 | 39 | William Jackson III | 67.47 | 55.20 | 72.95 | 831 | Bengals |
| 88 | 40 | Damontae Kazee | 67.44 | 58.10 | 69.50 | 803 | Falcons |
| 89 | 41 | Blidi Wreh-Wilson | 67.39 | 64.00 | 73.72 | 336 | Falcons |
| 90 | 42 | Holton Hill | 67.39 | 62.20 | 73.32 | 151 | Vikings |
| 91 | 43 | Nevin Lawson | 67.38 | 61.50 | 70.78 | 300 | Raiders |
| 92 | 44 | A.J. Bouye | 67.24 | 55.40 | 72.95 | 931 | Jaguars |
| 93 | 45 | Dre Kirkpatrick | 66.97 | 61.20 | 73.22 | 334 | Bengals |
| 94 | 46 | Buster Skrine | 66.97 | 59.10 | 68.88 | 727 | Bears |
| 95 | 47 | Ahkello Witherspoon | 66.90 | 63.50 | 69.90 | 562 | 49ers |
| 96 | 48 | Mike Hughes | 66.84 | 59.60 | 72.70 | 500 | Vikings |
| 97 | 49 | Pierre Desir | 66.69 | 57.90 | 72.45 | 683 | Colts |
| 98 | 50 | Isaiah Oliver | 66.45 | 54.50 | 72.20 | 927 | Falcons |
| 99 | 51 | Avonte Maddox | 66.12 | 57.40 | 72.71 | 518 | Eagles |
| 100 | 52 | Taron Johnson | 65.80 | 61.00 | 69.38 | 495 | Bills |
| 101 | 53 | David Long Jr. | 65.68 | 66.70 | 76.80 | 109 | Rams |
| 102 | 54 | Xavien Howard | 65.55 | 57.00 | 74.06 | 322 | Dolphins |
| 103 | 55 | Kendall Fuller | 65.48 | 55.30 | 71.01 | 498 | Chiefs |
| 104 | 56 | Donte Jackson | 65.14 | 56.30 | 69.46 | 726 | Panthers |
| 105 | 57 | Michael Davis | 65.08 | 57.30 | 71.42 | 659 | Chargers |
| 106 | 58 | B.W. Webb | 64.59 | 54.50 | 68.09 | 834 | Bengals |
| 107 | 59 | Antonio Hamilton Sr. | 64.45 | 59.50 | 77.00 | 132 | Giants |
| 108 | 60 | Arthur Maulet | 63.99 | 64.70 | 69.12 | 349 | Jets |
| 109 | 61 | Tre Herndon | 63.58 | 54.10 | 71.06 | 902 | Jaguars |
| 110 | 62 | Fabian Moreau | 63.49 | 56.40 | 68.21 | 664 | Commanders |
| 111 | 63 | Jalen Mills | 63.42 | 56.00 | 71.89 | 501 | Eagles |
| 112 | 64 | Tre Flowers | 62.45 | 53.10 | 65.55 | 978 | Seahawks |
| 113 | 65 | Rasul Douglas | 62.07 | 50.20 | 67.26 | 583 | Eagles |

### Rotation/backup (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 114 | 1 | Trumaine Johnson | 61.77 | 50.30 | 71.82 | 314 | Jets |
| 115 | 2 | Xavier Rhodes | 61.72 | 44.70 | 70.05 | 795 | Vikings |
| 116 | 3 | Rashaan Melvin | 61.30 | 47.20 | 69.97 | 870 | Lions |
| 117 | 4 | T.J. Carrie | 60.70 | 47.60 | 65.79 | 676 | Browns |
| 118 | 5 | Greedy Williams | 60.56 | 53.60 | 65.20 | 680 | Browns |
| 119 | 6 | Byron Murphy Jr. | 60.43 | 48.50 | 64.21 | 1105 | Cardinals |
| 120 | 7 | Davontae Harris | 60.00 | 54.60 | 63.99 | 429 | Broncos |
| 121 | 8 | P.J. Williams | 59.79 | 47.80 | 64.96 | 799 | Saints |
| 122 | 9 | Kendall Sheffield | 59.50 | 48.90 | 64.48 | 697 | Falcons |
| 123 | 10 | Josh Norman | 59.46 | 43.40 | 69.02 | 603 | Commanders |
| 124 | 11 | Coty Sensabaugh | 59.40 | 52.20 | 68.17 | 152 | Commanders |
| 125 | 12 | Isaac Yiadom | 59.34 | 52.30 | 63.77 | 504 | Broncos |
| 126 | 13 | Bashaud Breeland | 59.15 | 43.90 | 68.16 | 912 | Chiefs |
| 127 | 14 | Brandon Facyson | 58.32 | 53.40 | 64.74 | 328 | Chargers |
| 128 | 15 | Duke Dawson | 57.78 | 56.70 | 58.23 | 343 | Broncos |
| 129 | 16 | M.J. Stewart | 57.58 | 61.50 | 59.14 | 628 | Buccaneers |
| 130 | 17 | Sam Beal | 57.52 | 56.40 | 63.61 | 289 | Giants |
| 131 | 18 | Jamar Taylor | 57.08 | 44.30 | 65.50 | 215 | Falcons |
| 132 | 19 | Jamal Perry | 56.01 | 41.30 | 63.74 | 600 | Dolphins |
| 133 | 20 | DeAndre Baker | 55.82 | 45.60 | 58.47 | 970 | Giants |
| 134 | 21 | LeShaun Sims | 55.44 | 46.30 | 62.05 | 335 | Titans |
| 135 | 22 | Ken Webster | 55.37 | 47.30 | 64.92 | 226 | Dolphins |
| 136 | 23 | Ronald Darby | 55.11 | 39.80 | 66.99 | 506 | Eagles |
| 137 | 24 | Kevin Toliver II | 55.04 | 46.90 | 64.50 | 175 | Bears |
| 138 | 25 | Vernon Hargreaves III | 53.01 | 40.90 | 63.59 | 844 | Texans |
| 139 | 26 | Grant Haley | 52.56 | 50.70 | 55.88 | 422 | Giants |
| 140 | 27 | Nate Hairston | 52.50 | 51.30 | 55.58 | 392 | Jets |
| 141 | 28 | Anthony Averett | 51.40 | 41.20 | 62.10 | 221 | Ravens |
| 142 | 29 | Ryan Lewis | 51.39 | 32.80 | 69.11 | 293 | Dolphins |
| 143 | 30 | Keion Crossen | 49.78 | 44.20 | 57.80 | 134 | Texans |
| 144 | 31 | Quincy Wilson | 49.69 | 32.60 | 63.90 | 122 | Colts |
| 145 | 32 | Aaron Colvin | 48.66 | 38.10 | 59.77 | 223 | Commanders |
| 146 | 33 | Tony McRae | 48.03 | 46.30 | 54.60 | 197 | Bengals |
| 147 | 34 | Tae Hayes | 47.60 | 44.60 | 66.93 | 107 | Dolphins |
| 148 | 35 | Phillip Gaines | 47.32 | 29.20 | 64.91 | 133 | Texans |
| 149 | 36 | Corey Ballentine | 45.00 | 29.80 | 54.25 | 298 | Giants |
| 150 | 37 | Lonnie Johnson Jr. | 45.00 | 31.70 | 52.99 | 531 | Texans |

## DI — Defensive Interior

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 93.56 | 89.25 | 92.46 | 926 | Rams |
| 2 | 2 | Grady Jarrett | 87.39 | 87.78 | 83.59 | 805 | Falcons |
| 3 | 3 | Geno Atkins | 85.96 | 82.64 | 84.00 | 817 | Bengals |
| 4 | 4 | Kenny Clark | 85.79 | 87.39 | 81.71 | 869 | Packers |
| 5 | 5 | DeForest Buckner | 85.70 | 89.04 | 79.30 | 811 | 49ers |
| 6 | 6 | Cameron Heyward | 84.79 | 84.99 | 80.49 | 873 | Steelers |
| 7 | 7 | Jurrell Casey | 84.58 | 84.76 | 81.65 | 707 | Titans |
| 8 | 8 | Fletcher Cox | 84.58 | 87.44 | 78.51 | 799 | Eagles |
| 9 | 9 | Calais Campbell | 84.22 | 76.06 | 85.49 | 818 | Jaguars |
| 10 | 10 | Shelby Harris | 83.62 | 81.75 | 80.70 | 636 | Broncos |
| 11 | 11 | Leonard Williams | 83.59 | 86.25 | 78.16 | 732 | Giants |
| 12 | 12 | Javon Hargrave | 83.13 | 84.18 | 78.27 | 680 | Steelers |
| 13 | 13 | Sheldon Richardson | 82.14 | 80.10 | 79.54 | 774 | Browns |
| 14 | 14 | Chris Jones | 82.11 | 85.92 | 76.97 | 646 | Chiefs |
| 15 | 15 | DeMarcus Walker | 81.28 | 70.02 | 93.61 | 220 | Broncos |
| 16 | 16 | Taven Bryan | 80.82 | 84.16 | 74.43 | 481 | Jaguars |
| 17 | 17 | Eddie Goldman | 80.20 | 80.41 | 76.63 | 467 | Bears |
| 18 | 18 | DJ Reader | 80.11 | 82.80 | 75.08 | 622 | Texans |

### Good (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Dalvin Tomlinson | 79.92 | 80.52 | 75.36 | 595 | Giants |
| 20 | 2 | Ndamukong Suh | 79.92 | 75.07 | 78.98 | 874 | Buccaneers |
| 21 | 3 | Akiem Hicks | 79.84 | 79.44 | 81.67 | 191 | Bears |
| 22 | 4 | Vita Vea | 79.83 | 86.35 | 72.48 | 749 | Buccaneers |
| 23 | 5 | Gerald McCoy | 79.28 | 80.50 | 75.14 | 696 | Panthers |
| 24 | 6 | Steve McLendon | 78.86 | 71.58 | 79.54 | 465 | Jets |
| 25 | 7 | Dexter Lawrence | 78.86 | 83.67 | 71.48 | 701 | Giants |
| 26 | 8 | Stephon Tuitt | 78.73 | 85.88 | 76.25 | 278 | Steelers |
| 27 | 9 | Linval Joseph | 78.61 | 76.78 | 77.54 | 553 | Vikings |
| 28 | 10 | B.J. Hill | 78.45 | 75.34 | 76.36 | 486 | Giants |
| 29 | 11 | Michael Pierce | 78.41 | 73.38 | 79.27 | 481 | Ravens |
| 30 | 12 | Danny Shelton | 78.33 | 80.93 | 73.78 | 492 | Patriots |
| 31 | 13 | Damon Harrison Sr. | 78.30 | 67.52 | 81.84 | 527 | Lions |
| 32 | 14 | Matt Ioannidis | 78.27 | 75.58 | 76.93 | 827 | Commanders |
| 33 | 15 | Poona Ford | 77.97 | 74.58 | 78.66 | 506 | Seahawks |
| 34 | 16 | Ed Oliver | 77.95 | 70.79 | 78.56 | 557 | Bills |
| 35 | 17 | Michael Brockers | 77.46 | 76.11 | 74.20 | 766 | Rams |
| 36 | 18 | Jonathan Allen | 77.01 | 69.92 | 80.39 | 722 | Commanders |
| 37 | 19 | Lawrence Guy Sr. | 76.89 | 67.99 | 78.65 | 524 | Patriots |
| 38 | 20 | Mike Daniels | 76.84 | 71.33 | 82.28 | 203 | Lions |
| 39 | 21 | Jeffery Simmons | 76.69 | 77.64 | 79.19 | 315 | Titans |
| 40 | 22 | Shy Tuttle | 76.22 | 71.57 | 75.15 | 340 | Saints |
| 41 | 23 | Derek Wolfe | 76.14 | 73.89 | 76.61 | 523 | Broncos |
| 42 | 24 | Quinnen Williams | 75.45 | 75.82 | 74.17 | 512 | Jets |
| 43 | 25 | Daron Payne | 75.34 | 74.36 | 72.48 | 758 | Commanders |
| 44 | 26 | Armon Watts | 75.00 | 74.71 | 81.90 | 121 | Vikings |
| 45 | 27 | Maurice Hurst | 74.36 | 74.70 | 71.13 | 522 | Raiders |
| 46 | 28 | Mike Pennel | 74.17 | 70.14 | 76.86 | 154 | Chiefs |
| 47 | 29 | Folorunso Fatukasi | 74.01 | 76.68 | 75.23 | 390 | Jets |

### Starter (74 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 48 | 1 | Malcom Brown | 73.86 | 64.87 | 76.00 | 487 | Saints |
| 49 | 2 | Denico Autry | 73.73 | 66.38 | 76.74 | 620 | Colts |
| 50 | 3 | Larry Ogunjobi | 73.39 | 59.62 | 79.34 | 779 | Browns |
| 51 | 4 | Henry Anderson | 73.33 | 66.61 | 76.66 | 446 | Jets |
| 52 | 5 | Brandon Williams | 73.24 | 66.57 | 75.40 | 525 | Ravens |
| 53 | 6 | Christian Wilkins | 73.21 | 67.77 | 72.67 | 730 | Dolphins |
| 54 | 7 | Christian Covington | 73.18 | 69.57 | 74.55 | 481 | Cowboys |
| 55 | 8 | Tim Settle | 73.13 | 64.93 | 75.86 | 314 | Commanders |
| 56 | 9 | Johnathan Hankins | 73.00 | 63.65 | 75.58 | 670 | Raiders |
| 57 | 10 | Roy Robertson-Harris | 72.58 | 65.05 | 74.79 | 544 | Bears |
| 58 | 11 | Dean Lowry | 72.37 | 62.53 | 74.76 | 637 | Packers |
| 59 | 12 | Rodney Gunter | 72.37 | 63.74 | 75.52 | 602 | Cardinals |
| 60 | 13 | Marcell Dareus | 72.16 | 63.82 | 79.07 | 206 | Jaguars |
| 61 | 14 | Sheldon Rankins | 72.09 | 68.72 | 73.31 | 323 | Saints |
| 62 | 15 | Zach Kerr | 71.80 | 63.54 | 76.27 | 328 | Cardinals |
| 63 | 16 | Tyler Lancaster | 71.66 | 63.17 | 75.11 | 381 | Packers |
| 64 | 17 | Andrew Billings | 71.64 | 67.00 | 70.77 | 657 | Bengals |
| 65 | 18 | DaQuan Jones | 71.62 | 70.63 | 68.94 | 679 | Titans |
| 66 | 19 | Zach Sieler | 71.47 | 71.08 | 78.90 | 118 | Dolphins |
| 67 | 20 | Mike Purcell | 71.29 | 62.27 | 76.99 | 416 | Broncos |
| 68 | 21 | Davon Godchaux | 70.97 | 60.84 | 73.76 | 718 | Dolphins |
| 69 | 22 | Corey Liuget | 70.63 | 66.45 | 76.33 | 180 | Bills |
| 70 | 23 | Dre'Mont Jones | 70.25 | 64.63 | 71.92 | 283 | Broncos |
| 71 | 24 | Sebastian Joseph-Day | 70.23 | 52.34 | 77.99 | 481 | Rams |
| 72 | 25 | Abry Jones | 70.22 | 57.13 | 75.09 | 558 | Jaguars |
| 73 | 26 | Timmy Jernigan | 70.07 | 63.71 | 77.32 | 274 | Eagles |
| 74 | 27 | Vernon Butler | 69.98 | 61.27 | 73.50 | 440 | Panthers |
| 75 | 28 | Bilal Nichols | 69.85 | 56.10 | 77.59 | 445 | Bears |
| 76 | 29 | Quinton Jefferson | 69.73 | 65.95 | 71.21 | 589 | Seahawks |
| 77 | 30 | A'Shawn Robinson | 69.52 | 60.17 | 74.09 | 526 | Lions |
| 78 | 31 | P.J. Hall | 69.42 | 67.82 | 67.10 | 551 | Raiders |
| 79 | 32 | Jarran Reed | 69.39 | 58.31 | 75.94 | 479 | Seahawks |
| 80 | 33 | Tyson Alualu | 69.37 | 63.63 | 69.35 | 432 | Steelers |
| 81 | 34 | Adam Butler | 69.17 | 54.75 | 74.61 | 474 | Patriots |
| 82 | 35 | Josh Tupou | 69.10 | 68.54 | 69.21 | 465 | Bengals |
| 83 | 36 | Olsen Pierre | 68.82 | 48.05 | 84.44 | 172 | Raiders |
| 84 | 37 | Greg Gaines | 68.78 | 72.85 | 68.15 | 183 | Rams |
| 85 | 38 | Nathan Shepherd | 68.71 | 73.07 | 66.19 | 232 | Jets |
| 86 | 39 | David Onyemata | 68.59 | 60.69 | 70.21 | 565 | Saints |
| 87 | 40 | Dontari Poe | 68.53 | 67.81 | 67.45 | 402 | Panthers |
| 88 | 41 | D.J. Jones | 68.00 | 60.11 | 75.03 | 304 | 49ers |
| 89 | 42 | Jordan Phillips | 67.78 | 55.42 | 72.49 | 542 | Bills |
| 90 | 43 | William Gholston | 67.52 | 52.94 | 73.07 | 493 | Buccaneers |
| 91 | 44 | Dan McCullers | 67.45 | 55.78 | 74.19 | 131 | Steelers |
| 92 | 45 | Beau Allen | 66.82 | 57.67 | 70.93 | 179 | Buccaneers |
| 93 | 46 | Adam Gotsis | 66.81 | 56.81 | 72.96 | 272 | Broncos |
| 94 | 47 | Derrick Nnadi | 66.65 | 60.32 | 66.71 | 598 | Chiefs |
| 95 | 48 | Carlos Watkins | 66.61 | 57.07 | 76.50 | 265 | Texans |
| 96 | 49 | Corey Peters | 66.53 | 55.40 | 70.94 | 805 | Cardinals |
| 97 | 50 | Treyvon Hester | 66.53 | 63.12 | 67.34 | 132 | Commanders |
| 98 | 51 | Jonathan Bullard | 66.40 | 55.23 | 73.33 | 309 | Cardinals |
| 99 | 52 | Allen Bailey | 66.39 | 54.49 | 70.89 | 511 | Falcons |
| 100 | 53 | RJ McIntosh | 66.20 | 62.39 | 71.74 | 114 | Giants |
| 101 | 54 | Isaiah Mack | 65.58 | 60.35 | 68.03 | 172 | Titans |
| 102 | 55 | Star Lotulelei | 65.45 | 52.53 | 69.90 | 482 | Bills |
| 103 | 56 | Charles Omenihu | 65.37 | 60.48 | 66.55 | 443 | Texans |
| 104 | 57 | Austin Johnson | 65.30 | 57.77 | 66.15 | 325 | Titans |
| 105 | 58 | Al Woods | 65.22 | 52.73 | 71.04 | 450 | Seahawks |
| 106 | 59 | Sheldon Day | 65.17 | 53.90 | 70.60 | 325 | 49ers |
| 107 | 60 | Da'Shawn Hand | 65.01 | 58.85 | 74.59 | 110 | Lions |
| 108 | 61 | Tyeler Davison | 65.00 | 61.18 | 64.01 | 560 | Falcons |
| 109 | 62 | Hassan Ridgeway | 64.80 | 57.28 | 74.40 | 247 | Eagles |
| 110 | 63 | Maliek Collins | 64.45 | 57.30 | 65.99 | 763 | Cowboys |
| 111 | 64 | Grover Stewart | 64.41 | 55.99 | 66.37 | 627 | Colts |
| 112 | 65 | Xavier Williams | 64.28 | 51.49 | 75.40 | 118 | Chiefs |
| 113 | 66 | Chris Wormley | 64.18 | 59.13 | 65.26 | 446 | Ravens |
| 114 | 67 | Anthony Rush | 64.05 | 54.46 | 73.58 | 150 | Eagles |
| 115 | 68 | John Jenkins | 63.97 | 54.87 | 70.35 | 480 | Dolphins |
| 116 | 69 | Brent Urban | 63.96 | 53.09 | 71.31 | 245 | Bears |
| 117 | 70 | Jullian Taylor | 63.35 | 55.12 | 78.08 | 101 | 49ers |
| 118 | 71 | Margus Hunt | 63.10 | 46.38 | 70.39 | 451 | Colts |
| 119 | 72 | Brandon Dunn | 63.03 | 52.91 | 66.24 | 399 | Texans |
| 120 | 73 | Akeem Spence | 62.61 | 47.24 | 69.21 | 369 | Jaguars |
| 121 | 74 | Rakeem Nunez-Roches | 62.05 | 51.32 | 69.10 | 293 | Buccaneers |

### Rotation/backup (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 122 | 1 | Clinton McDonald | 61.62 | 41.54 | 75.20 | 122 | Cardinals |
| 123 | 2 | Justin Jones | 61.27 | 51.28 | 66.76 | 504 | Chargers |
| 124 | 3 | Jack Crawford | 61.08 | 44.35 | 70.57 | 431 | Falcons |
| 125 | 4 | Domata Peko Sr. | 60.81 | 47.42 | 70.67 | 140 | Ravens |
| 126 | 5 | Brandon Mebane | 60.37 | 41.48 | 71.61 | 408 | Chargers |
| 127 | 6 | Jaleel Johnson | 60.21 | 46.82 | 66.86 | 408 | Vikings |
| 128 | 7 | Antwaun Woods | 60.11 | 52.02 | 67.91 | 310 | Cowboys |
| 129 | 8 | Morgan Fox | 60.05 | 46.04 | 65.22 | 353 | Rams |
| 130 | 9 | Ryan Glasgow | 59.65 | 57.36 | 66.81 | 118 | Bengals |
| 131 | 10 | Damion Square | 59.57 | 47.00 | 63.78 | 402 | Chargers |
| 132 | 11 | Khalen Saunders | 59.53 | 48.11 | 67.14 | 303 | Chiefs |
| 133 | 12 | Jerry Tillery | 58.94 | 47.22 | 63.62 | 354 | Chargers |
| 134 | 13 | Angelo Blackson | 58.76 | 47.46 | 64.11 | 427 | Texans |
| 135 | 14 | Kyle Love | 58.62 | 41.03 | 66.69 | 412 | Panthers |
| 136 | 15 | Montravius Adams | 58.52 | 47.38 | 64.70 | 187 | Packers |
| 137 | 16 | Jihad Ward | 58.29 | 55.95 | 62.14 | 398 | Ravens |
| 138 | 17 | Caraun Reid | 57.69 | 48.62 | 71.33 | 136 | Cardinals |
| 139 | 18 | Abdullah Anderson | 57.28 | 52.97 | 69.40 | 106 | Bears |
| 140 | 19 | Shamar Stephen | 57.21 | 48.34 | 59.79 | 580 | Vikings |
| 141 | 20 | Eli Ankou | 57.08 | 48.68 | 68.00 | 178 | Browns |
| 142 | 21 | Renell Wren | 56.28 | 52.01 | 60.16 | 154 | Bengals |
| 143 | 22 | Miles Brown | 54.47 | 44.44 | 64.29 | 123 | Cardinals |
| 144 | 23 | Trysten Hill | 53.88 | 49.18 | 63.71 | 121 | Cowboys |
| 145 | 24 | Tanzel Smart | 50.42 | 44.45 | 56.17 | 171 | Rams |
| 146 | 25 | Zach Allen | 50.39 | 49.03 | 65.63 | 144 | Cardinals |
| 147 | 26 | Daniel Ekuale | 50.36 | 44.78 | 60.78 | 114 | Browns |
| 148 | 27 | John Atkins | 49.96 | 48.66 | 54.73 | 409 | Lions |
| 149 | 28 | Kevin Strong | 49.71 | 48.30 | 54.82 | 172 | Lions |
| 150 | 29 | Wes Horton | 45.00 | 34.09 | 52.04 | 117 | Panthers |
| 151 | 30 | Joey Ivie | 45.00 | 39.97 | 50.31 | 115 | Titans |
| 152 | 31 | Frank Herron | 45.00 | 43.86 | 53.92 | 103 | Lions |

## ED — Edge

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 92.17 | 91.79 | 88.77 | 833 | Broncos |
| 2 | 2 | Nick Bosa | 91.87 | 96.80 | 84.41 | 777 | 49ers |
| 3 | 3 | Joey Bosa | 91.74 | 96.47 | 87.24 | 836 | Chargers |
| 4 | 4 | T.J. Watt | 91.70 | 93.77 | 86.15 | 935 | Steelers |
| 5 | 5 | Danielle Hunter | 90.12 | 92.95 | 84.06 | 883 | Vikings |
| 6 | 6 | Khalil Mack | 89.96 | 93.91 | 83.80 | 925 | Bears |
| 7 | 7 | Justin Houston | 87.93 | 84.29 | 87.44 | 674 | Colts |
| 8 | 8 | DeMarcus Lawrence | 86.52 | 92.04 | 78.67 | 668 | Cowboys |
| 9 | 9 | Myles Garrett | 86.31 | 93.10 | 81.79 | 544 | Browns |
| 10 | 10 | Shaquil Barrett | 83.53 | 81.04 | 81.95 | 889 | Buccaneers |
| 11 | 11 | Cameron Jordan | 83.28 | 89.37 | 75.06 | 877 | Saints |
| 12 | 12 | Brandon Graham | 82.86 | 81.67 | 79.49 | 775 | Eagles |
| 13 | 13 | Za'Darius Smith | 82.35 | 85.36 | 76.59 | 872 | Packers |
| 14 | 14 | Chandler Jones | 82.22 | 81.47 | 78.55 | 1069 | Cardinals |
| 15 | 15 | Marcus Davenport | 81.69 | 89.80 | 75.25 | 533 | Saints |
| 16 | 16 | Carlos Dunlap | 80.31 | 80.87 | 76.81 | 739 | Bengals |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Jadeveon Clowney | 79.14 | 92.33 | 68.07 | 605 | Seahawks |
| 18 | 2 | Dante Fowler Jr. | 78.51 | 79.64 | 73.90 | 880 | Rams |
| 19 | 3 | J.J. Watt | 78.26 | 69.87 | 83.86 | 469 | Texans |
| 20 | 4 | Trey Flowers | 78.25 | 74.41 | 77.48 | 705 | Lions |
| 21 | 5 | Michael Bennett | 77.55 | 74.94 | 75.64 | 559 | Cowboys |
| 22 | 6 | Dee Ford | 77.42 | 75.38 | 79.29 | 226 | 49ers |
| 23 | 7 | Deatrich Wise Jr. | 76.81 | 81.23 | 70.73 | 229 | Patriots |
| 24 | 8 | Frank Clark | 76.80 | 68.33 | 79.32 | 725 | Chiefs |
| 25 | 9 | Cameron Wake | 76.61 | 62.53 | 86.09 | 195 | Titans |
| 26 | 10 | Melvin Ingram III | 76.52 | 70.13 | 78.18 | 668 | Chargers |
| 27 | 11 | Matthew Judon | 76.41 | 63.93 | 80.57 | 791 | Ravens |
| 28 | 12 | Ezekiel Ansah | 75.84 | 69.71 | 81.59 | 338 | Seahawks |
| 29 | 13 | Ifeadi Odenigbo | 75.70 | 74.39 | 72.40 | 736 | Vikings |
| 30 | 14 | Olivier Vernon | 75.50 | 79.44 | 74.22 | 508 | Browns |
| 31 | 15 | Jacob Martin | 75.47 | 66.87 | 78.34 | 220 | Texans |
| 32 | 16 | Ryan Kerrigan | 75.35 | 60.90 | 82.90 | 642 | Commanders |
| 33 | 17 | Jerry Hughes | 75.17 | 71.22 | 73.63 | 663 | Bills |
| 34 | 18 | Robert Quinn | 75.00 | 65.90 | 77.93 | 647 | Cowboys |
| 35 | 19 | Shaq Lawson | 74.41 | 74.53 | 72.35 | 483 | Bills |
| 36 | 20 | Chase Winovich | 74.19 | 66.22 | 75.34 | 291 | Patriots |

### Starter (55 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 37 | 1 | Bradley Chubb | 73.91 | 67.99 | 81.51 | 233 | Broncos |
| 38 | 2 | Yannick Ngakoue | 73.88 | 65.37 | 75.90 | 791 | Jaguars |
| 39 | 3 | Sam Hubbard | 73.85 | 67.36 | 74.66 | 852 | Bengals |
| 40 | 4 | Pernell McPhee | 73.54 | 66.22 | 80.50 | 260 | Ravens |
| 41 | 5 | Arik Armstead | 73.44 | 70.84 | 73.09 | 776 | 49ers |
| 42 | 6 | Brian Burns | 73.02 | 63.10 | 75.47 | 478 | Panthers |
| 43 | 7 | Takk McKinley | 73.01 | 66.31 | 74.66 | 546 | Falcons |
| 44 | 8 | Preston Smith | 72.48 | 64.06 | 73.93 | 870 | Packers |
| 45 | 9 | Tyus Bowser | 72.32 | 62.11 | 75.27 | 389 | Ravens |
| 46 | 10 | Everson Griffen | 72.17 | 64.76 | 75.02 | 849 | Vikings |
| 47 | 11 | Rashan Gary | 72.16 | 64.88 | 72.85 | 244 | Packers |
| 48 | 12 | Jabaal Sheard | 72.13 | 70.22 | 70.81 | 569 | Colts |
| 49 | 13 | Montez Sweat | 71.97 | 62.08 | 74.39 | 724 | Commanders |
| 50 | 14 | Maxx Crosby | 71.76 | 62.97 | 73.45 | 750 | Raiders |
| 51 | 15 | John Cominsky | 71.72 | 74.66 | 72.90 | 100 | Falcons |
| 52 | 16 | Mario Addison | 71.69 | 55.98 | 78.52 | 729 | Panthers |
| 53 | 17 | Markus Golden | 71.63 | 58.73 | 80.13 | 916 | Giants |
| 54 | 18 | Derek Barnett | 71.55 | 73.17 | 70.47 | 694 | Eagles |
| 55 | 19 | Whitney Mercilus | 71.40 | 62.76 | 75.27 | 950 | Texans |
| 56 | 20 | Carl Lawson | 70.86 | 63.04 | 76.80 | 457 | Bengals |
| 57 | 21 | Lorenzo Alexander | 70.14 | 41.34 | 85.17 | 494 | Bills |
| 58 | 22 | Vinny Curry | 70.11 | 61.28 | 73.08 | 393 | Eagles |
| 59 | 23 | Aaron Lynch | 69.73 | 59.20 | 75.40 | 244 | Bears |
| 60 | 24 | Jason Pierre-Paul | 68.96 | 64.11 | 71.16 | 586 | Buccaneers |
| 61 | 25 | Kyler Fackrell | 68.67 | 53.83 | 74.40 | 415 | Packers |
| 62 | 26 | Oshane Ximines | 68.54 | 59.79 | 70.21 | 502 | Giants |
| 63 | 27 | Jordan Jenkins | 68.46 | 60.04 | 70.94 | 572 | Jets |
| 64 | 28 | Bud Dupree | 68.46 | 63.01 | 67.92 | 980 | Steelers |
| 65 | 29 | Josh Sweat | 68.07 | 67.49 | 67.43 | 352 | Eagles |
| 66 | 30 | Leonard Floyd | 67.78 | 61.90 | 68.78 | 899 | Bears |
| 67 | 31 | Terrell Suggs | 67.56 | 47.01 | 77.61 | 690 | Chiefs |
| 68 | 32 | Samson Ebukam | 67.36 | 62.68 | 66.31 | 565 | Rams |
| 69 | 33 | Vic Beasley Jr. | 67.11 | 58.24 | 68.86 | 757 | Falcons |
| 70 | 34 | Ronald Blair III | 66.01 | 63.27 | 69.41 | 199 | 49ers |
| 71 | 35 | Clay Matthews | 65.87 | 45.10 | 77.53 | 614 | Rams |
| 72 | 36 | Ogbo Okoronkwo | 65.85 | 63.06 | 70.85 | 115 | Rams |
| 73 | 37 | Adrian Clayborn | 65.76 | 59.19 | 67.12 | 439 | Falcons |
| 74 | 38 | Kamalei Correa | 65.75 | 56.04 | 68.05 | 432 | Titans |
| 75 | 39 | Lorenzo Carter | 65.74 | 62.26 | 64.92 | 723 | Giants |
| 76 | 40 | Bruce Irvin | 65.63 | 45.96 | 76.14 | 608 | Panthers |
| 77 | 41 | John Simon | 65.55 | 56.81 | 70.23 | 481 | Patriots |
| 78 | 42 | Jeremiah Attaochu | 65.39 | 60.08 | 71.94 | 322 | Broncos |
| 79 | 43 | Ryan Anderson | 64.93 | 60.77 | 64.88 | 559 | Commanders |
| 80 | 44 | Trent Murphy | 64.66 | 55.46 | 67.56 | 674 | Bills |
| 81 | 45 | Trey Hendrickson | 64.63 | 65.72 | 65.16 | 404 | Saints |
| 82 | 46 | Clelin Ferrell | 64.12 | 64.57 | 60.69 | 648 | Raiders |
| 83 | 47 | Ben Banogu | 63.50 | 57.16 | 64.59 | 272 | Colts |
| 84 | 48 | Cassius Marsh | 63.49 | 57.16 | 63.74 | 429 | Cardinals |
| 85 | 49 | Tarell Basham | 63.44 | 60.74 | 63.77 | 590 | Jets |
| 86 | 50 | Jaylon Ferguson | 63.38 | 59.37 | 63.97 | 498 | Ravens |
| 87 | 51 | Benson Mayowa | 63.15 | 58.77 | 63.15 | 302 | Raiders |
| 88 | 52 | Kerry Hyder Jr. | 62.91 | 55.31 | 66.62 | 439 | Cowboys |
| 89 | 53 | Kyle Phillips | 62.52 | 62.31 | 59.52 | 549 | Jets |
| 90 | 54 | Alex Okafor | 62.49 | 52.63 | 69.26 | 421 | Chiefs |
| 91 | 55 | Mario Edwards Jr. | 62.29 | 58.66 | 62.63 | 293 | Saints |

### Rotation/backup (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 92 | 1 | Trent Harris | 61.88 | 56.16 | 66.72 | 253 | Dolphins |
| 93 | 2 | Carl Nassib | 61.73 | 61.24 | 59.24 | 630 | Buccaneers |
| 94 | 3 | Devon Kennard | 61.70 | 48.32 | 66.97 | 935 | Lions |
| 95 | 4 | Emmanuel Ogbah | 61.57 | 62.96 | 61.47 | 410 | Chiefs |
| 96 | 5 | Malik Reed | 61.46 | 59.71 | 59.49 | 468 | Broncos |
| 97 | 6 | Stephen Weatherly | 61.43 | 57.79 | 61.15 | 422 | Vikings |
| 98 | 7 | Brennan Scarlett | 60.18 | 53.84 | 61.81 | 491 | Texans |
| 99 | 8 | Vince Biegel | 60.14 | 56.99 | 64.32 | 627 | Dolphins |
| 100 | 9 | Andrew Brown | 59.58 | 59.32 | 57.67 | 241 | Bengals |
| 101 | 10 | Solomon Thomas | 59.43 | 61.08 | 54.58 | 425 | 49ers |
| 102 | 11 | Jordan Willis | 59.11 | 63.22 | 55.86 | 162 | Jets |
| 103 | 12 | Anthony Chickillo | 58.93 | 53.12 | 61.23 | 146 | Steelers |
| 104 | 13 | Taco Charlton | 58.83 | 57.69 | 60.10 | 396 | Dolphins |
| 105 | 14 | Chad Thomas | 58.66 | 51.11 | 59.53 | 464 | Browns |
| 106 | 15 | Charles Harris | 58.63 | 58.31 | 57.27 | 429 | Dolphins |
| 107 | 16 | Dawuane Smoot | 58.62 | 58.69 | 56.90 | 404 | Jaguars |
| 108 | 17 | Darryl Johnson | 58.53 | 55.92 | 57.14 | 224 | Bills |
| 109 | 18 | Romeo Okwara | 58.27 | 59.91 | 56.45 | 605 | Lions |
| 110 | 19 | Justin Hollins | 57.60 | 57.32 | 55.70 | 266 | Broncos |
| 111 | 20 | Efe Obada | 57.48 | 55.47 | 57.01 | 306 | Panthers |
| 112 | 21 | Dorance Armstrong | 57.46 | 56.10 | 55.23 | 262 | Cowboys |
| 113 | 22 | Al-Quadin Muhammad | 57.06 | 58.13 | 55.20 | 483 | Colts |
| 114 | 23 | Rasheem Green | 57.04 | 56.39 | 55.65 | 545 | Seahawks |
| 115 | 24 | Tanoh Kpassagnon | 56.79 | 57.79 | 55.39 | 691 | Chiefs |
| 116 | 25 | Arden Key | 56.51 | 59.74 | 56.06 | 179 | Raiders |
| 117 | 26 | Nate Orchard | 55.71 | 56.86 | 59.21 | 118 | Commanders |
| 118 | 27 | Branden Jackson | 55.07 | 54.32 | 54.95 | 418 | Seahawks |
| 119 | 28 | Isaiah Irving | 55.01 | 55.88 | 56.51 | 128 | Bears |
| 120 | 29 | Isaac Rochell | 54.92 | 55.07 | 50.66 | 274 | Chargers |
| 121 | 30 | Anthony Zettel | 54.06 | 51.18 | 57.87 | 103 | 49ers |
| 122 | 31 | Bryan Cox Jr. | 53.94 | 56.29 | 56.13 | 208 | Browns |
| 123 | 32 | Shilique Calhoun | 53.40 | 53.77 | 53.99 | 266 | Patriots |
| 124 | 33 | Carl Granderson | 52.90 | 59.25 | 57.92 | 115 | Saints |
| 125 | 34 | Anthony Nelson | 52.85 | 61.83 | 50.00 | 152 | Buccaneers |
| 126 | 35 | Avery Moss | 52.31 | 56.10 | 51.22 | 348 | Dolphins |
| 127 | 36 | Marquis Haynes Sr. | 52.06 | 53.63 | 54.79 | 210 | Panthers |
| 128 | 37 | Josh Mauro | 51.47 | 50.17 | 51.31 | 282 | Raiders |
| 129 | 38 | Porter Gustin | 50.01 | 58.43 | 53.65 | 225 | Browns |
| 130 | 39 | L.J. Collier | 49.05 | 54.00 | 46.79 | 152 | Seahawks |
| 131 | 40 | Demone Harris | 45.00 | 54.29 | 50.63 | 121 | Chiefs |
| 132 | 41 | Austin Bryant | 45.00 | 56.33 | 48.99 | 133 | Lions |

## G — Guard

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Brandon Brooks | 97.32 | 92.80 | 96.17 | 1046 | Eagles |
| 2 | 2 | Quenton Nelson | 96.00 | 91.20 | 95.03 | 1044 | Colts |
| 3 | 3 | Zack Martin | 92.17 | 88.10 | 90.72 | 1114 | Cowboys |
| 4 | 4 | Marshal Yanda | 91.37 | 86.60 | 90.38 | 968 | Ravens |
| 5 | 5 | Brandon Scherff | 84.61 | 75.40 | 86.58 | 643 | Commanders |
| 6 | 6 | Nick Gates | 84.13 | 77.00 | 84.72 | 291 | Giants |
| 7 | 7 | Joe Thuney | 83.67 | 77.40 | 83.69 | 1140 | Patriots |
| 8 | 8 | Kevin Zeitler | 83.66 | 76.40 | 84.34 | 991 | Giants |
| 9 | 9 | Graham Glasgow | 82.47 | 74.10 | 83.89 | 872 | Lions |
| 10 | 10 | Larry Warford | 81.78 | 73.10 | 83.40 | 970 | Saints |
| 11 | 11 | Joel Bitonio | 81.51 | 74.20 | 82.22 | 1039 | Browns |
| 12 | 12 | Halapoulivaati Vaitai | 81.07 | 72.80 | 82.41 | 477 | Eagles |
| 13 | 13 | Shaq Mason | 80.82 | 73.00 | 81.87 | 1067 | Patriots |
| 14 | 14 | Ali Marpet | 80.27 | 72.30 | 81.42 | 1139 | Buccaneers |

### Good (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Rodger Saffold | 79.94 | 71.40 | 81.46 | 928 | Titans |
| 16 | 2 | David DeCastro | 79.39 | 71.10 | 80.75 | 995 | Steelers |
| 17 | 3 | Isaac Seumalo | 78.08 | 69.60 | 79.56 | 1162 | Eagles |
| 18 | 4 | Denzelle Good | 76.47 | 65.10 | 79.89 | 338 | Raiders |
| 19 | 5 | Andrew Norwell | 75.90 | 65.50 | 78.67 | 1088 | Jaguars |
| 20 | 6 | Cody Whitehair | 75.57 | 64.90 | 78.51 | 1069 | Bears |
| 21 | 7 | Elgton Jenkins | 75.34 | 69.20 | 75.27 | 964 | Packers |
| 22 | 8 | Jon Feliciano | 75.05 | 64.10 | 78.18 | 947 | Bills |
| 23 | 9 | Justin Pugh | 75.04 | 66.80 | 76.37 | 1022 | Cardinals |
| 24 | 10 | Billy Turner | 74.93 | 65.30 | 77.19 | 1076 | Packers |
| 25 | 11 | Pat Elflein | 74.65 | 64.70 | 77.11 | 919 | Vikings |
| 26 | 12 | Greg Van Roten | 74.59 | 65.60 | 76.41 | 704 | Panthers |
| 27 | 13 | Laken Tomlinson | 74.59 | 64.70 | 77.01 | 1061 | 49ers |
| 28 | 14 | Dalton Risner | 74.53 | 64.40 | 77.12 | 975 | Broncos |
| 29 | 15 | Bradley Bozeman | 74.53 | 63.40 | 77.79 | 1105 | Ravens |
| 30 | 16 | Ereck Flowers | 74.18 | 64.10 | 76.74 | 937 | Commanders |

### Starter (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Chris Lindstrom | 73.17 | 66.60 | 73.38 | 309 | Falcons |
| 32 | 2 | Trai Turner | 73.03 | 63.90 | 74.95 | 888 | Panthers |
| 33 | 3 | Alex Cappa | 72.93 | 62.70 | 75.59 | 869 | Buccaneers |
| 34 | 4 | Mark Glowinski | 72.67 | 60.50 | 76.61 | 1076 | Colts |
| 35 | 5 | Connor Williams | 72.53 | 60.90 | 76.11 | 727 | Cowboys |
| 36 | 6 | Josh Kline | 72.32 | 61.50 | 75.36 | 733 | Vikings |
| 37 | 7 | D.J. Fluker | 72.10 | 60.40 | 75.74 | 863 | Seahawks |
| 38 | 8 | Mike Iupati | 72.06 | 60.40 | 75.66 | 1015 | Seahawks |
| 39 | 9 | Kenny Wiggins | 72.00 | 61.80 | 74.64 | 438 | Lions |
| 40 | 10 | Michael Schofield III | 71.85 | 63.60 | 73.18 | 1057 | Chargers |
| 41 | 11 | Gabe Jackson | 71.77 | 61.80 | 74.25 | 707 | Raiders |
| 42 | 12 | Mike Person | 71.70 | 61.80 | 74.13 | 937 | 49ers |
| 43 | 13 | Brian Winters | 71.47 | 61.00 | 74.28 | 526 | Jets |
| 44 | 14 | Jamil Douglas | 71.09 | 59.00 | 74.98 | 388 | Titans |
| 45 | 15 | Xavier Su'a-Filo | 71.08 | 60.10 | 74.24 | 307 | Cowboys |
| 46 | 16 | J.R. Sweezy | 71.02 | 61.60 | 73.14 | 1001 | Cardinals |
| 47 | 17 | Earl Watford | 70.95 | 64.90 | 70.81 | 326 | Buccaneers |
| 48 | 18 | Ron Leary | 70.55 | 58.40 | 74.48 | 754 | Broncos |
| 49 | 19 | Austin Schlottmann | 70.23 | 58.20 | 74.09 | 260 | Broncos |
| 50 | 20 | John Miller | 69.84 | 58.60 | 73.17 | 779 | Bengals |
| 51 | 21 | Lane Taylor | 69.64 | 59.20 | 72.44 | 114 | Packers |
| 52 | 22 | Jordan Devey | 69.21 | 53.20 | 75.72 | 228 | Raiders |
| 53 | 23 | Will Hernandez | 68.94 | 58.40 | 71.80 | 1068 | Giants |
| 54 | 24 | Ramon Foster | 68.90 | 59.00 | 71.33 | 822 | Steelers |
| 55 | 25 | Laurent Duvernay-Tardif | 68.63 | 57.20 | 72.08 | 899 | Chiefs |
| 56 | 26 | Max Scharping | 68.33 | 56.70 | 71.91 | 938 | Texans |
| 57 | 27 | Ted Larsen | 68.15 | 55.40 | 72.48 | 168 | Bears |
| 58 | 28 | Wyatt Teller | 67.92 | 56.70 | 71.23 | 559 | Browns |
| 59 | 29 | Alex Lewis | 67.71 | 56.10 | 71.28 | 764 | Jets |
| 60 | 30 | Quinton Spain | 67.59 | 55.40 | 71.55 | 1063 | Bills |
| 61 | 31 | A.J. Cann | 67.50 | 55.30 | 71.47 | 775 | Jaguars |
| 62 | 32 | Zach Fulton | 66.35 | 52.20 | 71.62 | 955 | Texans |
| 63 | 33 | Danny Isidora | 66.02 | 53.70 | 70.06 | 127 | Dolphins |
| 64 | 34 | Wes Martin | 65.80 | 51.70 | 71.03 | 290 | Commanders |
| 65 | 35 | Alex Redmond | 65.11 | 50.60 | 70.62 | 189 | Bengals |
| 66 | 36 | Dan Feeney | 64.65 | 51.70 | 69.12 | 1032 | Chargers |
| 67 | 37 | Nick Easton | 64.40 | 48.90 | 70.57 | 409 | Saints |
| 68 | 38 | Andrus Peat | 64.10 | 49.50 | 69.66 | 575 | Saints |
| 69 | 39 | Forrest Lamp | 63.87 | 59.30 | 62.75 | 157 | Chargers |
| 70 | 40 | Tom Compton | 63.20 | 49.70 | 68.03 | 363 | Jets |
| 71 | 41 | Jamon Brown | 62.82 | 53.20 | 65.06 | 587 | Falcons |
| 72 | 42 | Evan Boehm | 62.41 | 47.40 | 68.25 | 595 | Dolphins |

### Rotation/backup (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 73 | 1 | James Carpenter | 61.27 | 45.30 | 67.75 | 675 | Falcons |
| 74 | 2 | Deion Calhoun | 60.97 | 44.20 | 67.99 | 471 | Dolphins |
| 75 | 3 | Michael Jordan | 60.97 | 43.70 | 68.31 | 648 | Bengals |
| 76 | 4 | Michael Deiter | 60.05 | 42.50 | 67.59 | 996 | Dolphins |
| 77 | 5 | Nate Davis | 59.61 | 40.90 | 67.91 | 724 | Titans |
| 78 | 6 | Eric Kush | 58.99 | 45.40 | 63.89 | 436 | Browns |
| 79 | 7 | Kyle Long | 57.52 | 38.00 | 66.37 | 250 | Bears |
| 80 | 8 | Joe Noteboom | 55.12 | 39.60 | 61.30 | 376 | Rams |
| 81 | 9 | Jamil Demby | 51.40 | 26.40 | 63.90 | 146 | Rams |

## HB — Running Back

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Chubb | 88.20 | 88.70 | 83.70 | 293 | Browns |
| 2 | 2 | Josh Jacobs | 85.52 | 87.10 | 80.30 | 147 | Raiders |
| 3 | 3 | Austin Ekeler | 84.21 | 85.20 | 79.39 | 362 | Chargers |
| 4 | 4 | Christian McCaffrey | 81.51 | 86.50 | 74.01 | 570 | Panthers |
| 5 | 5 | Aaron Jones | 81.00 | 84.80 | 74.30 | 309 | Packers |
| 6 | 6 | Kareem Hunt | 80.38 | 76.70 | 78.67 | 190 | Browns |

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Derrick Henry | 79.55 | 76.20 | 77.62 | 194 | Titans |
| 8 | 2 | Dalvin Cook | 79.38 | 81.30 | 73.94 | 269 | Vikings |
| 9 | 3 | Saquon Barkley | 79.13 | 72.20 | 79.58 | 377 | Giants |
| 10 | 4 | Chris Carson | 78.91 | 77.90 | 75.41 | 307 | Seahawks |
| 11 | 5 | Alvin Kamara | 78.61 | 70.00 | 80.18 | 345 | Saints |
| 12 | 6 | Raheem Mostert | 78.12 | 75.50 | 75.70 | 149 | 49ers |
| 13 | 7 | Duke Johnson Jr. | 77.61 | 74.20 | 75.72 | 346 | Texans |
| 14 | 8 | Mark Ingram II | 76.47 | 79.80 | 70.09 | 165 | Ravens |
| 15 | 9 | Ezekiel Elliott | 76.41 | 77.00 | 71.85 | 465 | Cowboys |
| 16 | 10 | Kenyan Drake | 76.02 | 71.20 | 75.06 | 303 | Cardinals |
| 17 | 11 | Gus Edwards | 75.22 | 70.30 | 74.34 | 126 | Ravens |
| 18 | 12 | Devin Singletary | 74.68 | 66.10 | 76.23 | 271 | Bills |
| 19 | 13 | Le'Veon Bell | 74.08 | 74.10 | 69.90 | 391 | Jets |

### Starter (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Joe Mixon | 73.35 | 75.30 | 67.88 | 241 | Bengals |
| 21 | 2 | James Conner | 72.94 | 73.60 | 68.33 | 153 | Steelers |
| 22 | 3 | Phillip Lindsay | 72.68 | 71.30 | 69.44 | 211 | Broncos |
| 23 | 4 | Carlos Hyde | 72.37 | 74.20 | 66.99 | 211 | Texans |
| 24 | 5 | Jalen Richard | 72.22 | 66.10 | 72.13 | 204 | Raiders |
| 25 | 6 | DeAndre Washington | 72.08 | 76.50 | 64.96 | 127 | Raiders |
| 26 | 7 | Kerryon Johnson | 71.88 | 66.70 | 71.17 | 104 | Lions |
| 27 | 8 | Miles Sanders | 71.66 | 62.30 | 73.74 | 317 | Eagles |
| 28 | 9 | Damien Williams | 71.61 | 65.70 | 71.38 | 193 | Chiefs |
| 29 | 10 | Adrian Peterson | 71.44 | 67.60 | 69.84 | 127 | Commanders |
| 30 | 11 | J.D. McKissic | 71.35 | 72.40 | 66.48 | 154 | Lions |
| 31 | 12 | Chris Thompson | 70.97 | 65.10 | 70.71 | 221 | Commanders |
| 32 | 13 | Jamaal Williams | 70.90 | 76.70 | 62.87 | 202 | Packers |
| 33 | 14 | Latavius Murray | 70.86 | 73.90 | 64.67 | 198 | Saints |
| 34 | 15 | Marlon Mack | 70.67 | 69.50 | 67.29 | 177 | Colts |
| 35 | 16 | Todd Gurley II | 70.63 | 67.00 | 68.88 | 391 | Rams |
| 36 | 17 | James White | 70.03 | 76.10 | 61.82 | 337 | Patriots |
| 37 | 18 | David Johnson | 69.83 | 72.20 | 64.08 | 232 | Cardinals |
| 38 | 19 | Tevin Coleman | 69.43 | 69.50 | 65.21 | 170 | 49ers |
| 39 | 20 | Rex Burkhead | 69.30 | 70.20 | 64.54 | 153 | Patriots |
| 40 | 21 | Dion Lewis | 69.28 | 60.20 | 71.16 | 198 | Titans |
| 41 | 22 | Chase Edmonds | 68.94 | 69.70 | 64.26 | 101 | Cardinals |
| 42 | 23 | Melvin Gordon III | 68.69 | 66.00 | 66.31 | 208 | Chargers |
| 43 | 24 | Royce Freeman | 68.59 | 65.80 | 66.29 | 278 | Broncos |
| 44 | 25 | Ronald Jones | 68.50 | 67.60 | 64.93 | 170 | Buccaneers |
| 45 | 26 | LeSean McCoy | 68.45 | 65.10 | 66.52 | 145 | Chiefs |
| 46 | 27 | Brian Hill | 68.08 | 58.90 | 70.04 | 108 | Falcons |
| 47 | 28 | Leonard Fournette | 67.72 | 64.00 | 66.03 | 483 | Jaguars |
| 48 | 29 | Sony Michel | 66.86 | 68.00 | 61.93 | 102 | Patriots |
| 49 | 30 | David Montgomery | 66.38 | 66.60 | 62.07 | 239 | Bears |
| 50 | 31 | Tarik Cohen | 66.23 | 60.60 | 65.82 | 361 | Bears |
| 51 | 32 | Frank Gore | 65.75 | 64.10 | 62.69 | 121 | Bills |
| 52 | 33 | Devonta Freeman | 65.23 | 60.80 | 64.01 | 337 | Falcons |
| 53 | 34 | Nyheim Hines | 65.04 | 63.30 | 62.03 | 223 | Colts |
| 54 | 35 | Peyton Barber | 64.50 | 64.90 | 60.06 | 110 | Buccaneers |
| 55 | 36 | T.J. Yeldon | 64.30 | 58.10 | 64.27 | 100 | Bills |
| 56 | 37 | Ty Johnson | 63.02 | 51.20 | 66.74 | 188 | Lions |
| 57 | 38 | Giovani Bernard | 62.07 | 51.80 | 64.75 | 262 | Bengals |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 58 | 1 | Darrel Williams | 60.88 | 63.50 | 54.97 | 104 | Chiefs |
| 59 | 2 | Jaylen Samuels | 60.31 | 54.40 | 60.08 | 206 | Steelers |
| 60 | 3 | Dare Ogunbowale | 59.87 | 60.30 | 55.42 | 223 | Buccaneers |
| 61 | 4 | Kalen Ballage | 59.64 | 58.20 | 56.44 | 111 | Dolphins |
| 62 | 5 | Patrick Laird | 56.66 | 52.40 | 55.33 | 158 | Dolphins |

## LB — Linebacker

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Demario Davis | 86.81 | 90.10 | 80.45 | 985 | Saints |
| 2 | 2 | Eric Kendricks | 85.85 | 90.20 | 79.94 | 948 | Vikings |
| 3 | 3 | Luke Kuechly | 84.37 | 85.40 | 79.51 | 1064 | Panthers |
| 4 | 4 | Lavonte David | 84.19 | 88.20 | 78.60 | 1124 | Buccaneers |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | T.J. Edwards | 79.21 | 86.60 | 74.28 | 112 | Eagles |
| 6 | 2 | Cory Littleton | 78.99 | 79.00 | 74.81 | 1039 | Rams |
| 7 | 3 | Kevin Pierre-Louis | 78.31 | 90.50 | 74.87 | 213 | Bears |
| 8 | 4 | Bobby Okereke | 77.45 | 78.40 | 72.65 | 472 | Colts |
| 9 | 5 | Bobby Wagner | 76.65 | 76.10 | 73.17 | 1054 | Seahawks |
| 10 | 6 | Shaun Dion Hamilton | 74.45 | 74.80 | 74.60 | 387 | Commanders |

### Starter (52 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Jaylon Smith | 73.75 | 70.20 | 71.95 | 991 | Cowboys |
| 12 | 2 | Vince Williams | 73.57 | 76.20 | 69.31 | 397 | Steelers |
| 13 | 3 | Deion Jones | 73.40 | 76.60 | 70.24 | 946 | Falcons |
| 14 | 4 | Josh Bynes | 73.19 | 76.60 | 71.02 | 391 | Ravens |
| 15 | 5 | Jayon Brown | 72.68 | 71.60 | 70.27 | 829 | Titans |
| 16 | 6 | Zach Cunningham | 71.59 | 69.00 | 69.78 | 943 | Texans |
| 17 | 7 | Jamie Collins Sr. | 71.38 | 75.80 | 66.35 | 813 | Patriots |
| 18 | 8 | C.J. Mosley | 70.83 | 75.00 | 71.48 | 114 | Jets |
| 19 | 9 | Kiko Alonso | 70.56 | 70.90 | 68.56 | 285 | Saints |
| 20 | 10 | David Mayo | 70.28 | 72.90 | 68.02 | 631 | Giants |
| 21 | 11 | Dont'a Hightower | 70.21 | 69.40 | 67.42 | 724 | Patriots |
| 22 | 12 | Benardrick McKinney | 70.20 | 67.60 | 68.80 | 844 | Texans |
| 23 | 13 | Eric Wilson | 69.46 | 67.60 | 67.16 | 380 | Vikings |
| 24 | 14 | Josey Jewell | 69.45 | 68.40 | 69.23 | 214 | Broncos |
| 25 | 15 | Nick Kwiatkoski | 69.26 | 72.50 | 68.66 | 512 | Bears |
| 26 | 16 | Leon Jacobs | 69.25 | 68.30 | 68.59 | 325 | Jaguars |
| 27 | 17 | Reggie Ragland | 69.01 | 68.10 | 67.11 | 235 | Chiefs |
| 28 | 18 | Todd Davis | 68.95 | 65.10 | 68.82 | 897 | Broncos |
| 29 | 19 | Dre Greenlaw | 68.73 | 64.00 | 67.71 | 725 | 49ers |
| 30 | 20 | Fred Warner | 68.39 | 66.90 | 65.21 | 985 | 49ers |
| 31 | 21 | Devin Bush | 67.96 | 62.90 | 67.17 | 889 | Steelers |
| 32 | 22 | Foyesade Oluokun | 67.74 | 62.70 | 66.94 | 310 | Falcons |
| 33 | 23 | Matt Milano | 67.32 | 65.30 | 66.39 | 893 | Bills |
| 34 | 24 | Shaq Thompson | 67.32 | 65.80 | 66.05 | 962 | Panthers |
| 35 | 25 | Drue Tranquill | 67.04 | 66.60 | 67.34 | 382 | Chargers |
| 36 | 26 | L.J. Fort | 66.98 | 70.10 | 65.64 | 254 | Ravens |
| 37 | 27 | Kyzir White | 66.83 | 66.60 | 67.90 | 372 | Chargers |
| 38 | 28 | Ja'Whaun Bentley | 66.28 | 67.60 | 66.32 | 275 | Patriots |
| 39 | 29 | Raekwon McMillan | 66.02 | 63.90 | 64.83 | 516 | Dolphins |
| 40 | 30 | Elandon Roberts | 65.56 | 61.90 | 64.86 | 202 | Patriots |
| 41 | 31 | Jahlani Tavai | 65.48 | 61.60 | 64.93 | 597 | Lions |
| 42 | 32 | Nigel Bradham | 65.28 | 64.60 | 63.96 | 717 | Eagles |
| 43 | 33 | K.J. Wright | 65.24 | 62.10 | 66.82 | 997 | Seahawks |
| 44 | 34 | B.J. Goodson | 65.18 | 62.20 | 66.24 | 254 | Packers |
| 45 | 35 | Anthony Barr | 64.93 | 60.60 | 65.63 | 930 | Vikings |
| 46 | 36 | Thomas Davis Sr. | 64.27 | 61.70 | 63.07 | 805 | Chargers |
| 47 | 37 | Blake Martinez | 64.15 | 58.90 | 63.49 | 1024 | Packers |
| 48 | 38 | Tremaine Edmunds | 64.03 | 60.60 | 62.54 | 981 | Bills |
| 49 | 39 | Anthony Walker Jr. | 63.76 | 61.00 | 64.45 | 811 | Colts |
| 50 | 40 | Joe Schobert | 63.69 | 59.10 | 63.51 | 1057 | Browns |
| 51 | 41 | Ben Gedeon | 63.55 | 60.10 | 66.16 | 102 | Vikings |
| 52 | 42 | Danny Trevathan | 63.46 | 61.90 | 64.81 | 559 | Bears |
| 53 | 43 | Jordan Hicks | 63.23 | 61.00 | 63.68 | 1133 | Cardinals |
| 54 | 44 | Sione Takitaki | 63.08 | 64.40 | 66.37 | 105 | Browns |
| 55 | 45 | Sean Lee | 63.01 | 61.50 | 63.70 | 637 | Cowboys |
| 56 | 46 | Joe Thomas | 62.61 | 64.40 | 63.50 | 246 | Cowboys |
| 57 | 47 | Cole Holcomb | 62.60 | 56.00 | 62.84 | 718 | Commanders |
| 58 | 48 | Will Compton | 62.55 | 64.30 | 65.75 | 245 | Raiders |
| 59 | 49 | Jon Bostic | 62.45 | 55.90 | 63.07 | 1031 | Commanders |
| 60 | 50 | Kevin Minter | 62.41 | 65.20 | 64.72 | 275 | Buccaneers |
| 61 | 51 | Ben Niemann | 62.40 | 59.70 | 64.58 | 400 | Chiefs |
| 62 | 52 | Damien Wilson | 62.00 | 55.20 | 62.36 | 709 | Chiefs |

### Rotation/backup (53 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 63 | 1 | Tahir Whitehead | 61.94 | 53.20 | 63.60 | 941 | Raiders |
| 64 | 2 | Mychal Kendricks | 61.82 | 62.70 | 61.85 | 649 | Seahawks |
| 65 | 3 | Wesley Woodyard | 61.64 | 56.70 | 62.43 | 325 | Titans |
| 66 | 4 | Kwon Alexander | 61.53 | 61.30 | 65.65 | 357 | 49ers |
| 67 | 5 | Mark Barron | 61.28 | 57.50 | 61.61 | 750 | Steelers |
| 68 | 6 | Kamu Grugier-Hill | 61.06 | 60.10 | 62.74 | 300 | Eagles |
| 69 | 7 | Denzel Perryman | 60.34 | 59.20 | 63.08 | 359 | Chargers |
| 70 | 8 | Leighton Vander Esch | 60.21 | 58.40 | 61.80 | 510 | Cowboys |
| 71 | 9 | De'Vondre Campbell | 59.97 | 50.10 | 62.39 | 921 | Falcons |
| 72 | 10 | Nathan Gerry | 59.92 | 58.30 | 61.62 | 620 | Eagles |
| 73 | 11 | Rashaan Evans | 59.87 | 49.90 | 63.13 | 950 | Titans |
| 74 | 12 | Travin Howard | 59.77 | 60.70 | 65.85 | 102 | Rams |
| 75 | 13 | Roquan Smith | 59.51 | 52.40 | 62.69 | 719 | Bears |
| 76 | 14 | Germaine Pratt | 59.10 | 51.00 | 61.37 | 437 | Bengals |
| 77 | 15 | Nick Vigil | 59.10 | 54.30 | 60.73 | 985 | Bengals |
| 78 | 16 | Vontaze Burfict | 58.64 | 62.20 | 62.41 | 191 | Raiders |
| 79 | 17 | Jerome Baker | 58.55 | 46.70 | 62.28 | 1080 | Dolphins |
| 80 | 18 | James Burgess | 58.26 | 54.90 | 61.02 | 662 | Jets |
| 81 | 19 | Alec Ogletree | 58.25 | 55.60 | 58.35 | 850 | Giants |
| 82 | 20 | Sam Eguavoen | 58.11 | 50.20 | 59.21 | 621 | Dolphins |
| 83 | 21 | Jalen Reeves-Maybin | 58.05 | 54.80 | 61.69 | 298 | Lions |
| 84 | 22 | Craig Robertson | 57.62 | 57.50 | 61.24 | 189 | Saints |
| 85 | 23 | Deone Bucannon | 57.59 | 53.90 | 59.74 | 244 | Giants |
| 86 | 24 | Patrick Onwuasor | 56.96 | 49.40 | 58.87 | 473 | Ravens |
| 87 | 25 | Anthony Hitchens | 56.67 | 48.80 | 59.42 | 699 | Chiefs |
| 88 | 26 | Devin White | 56.64 | 51.90 | 58.77 | 826 | Buccaneers |
| 89 | 27 | Kentrell Brothers | 56.47 | 57.80 | 63.82 | 111 | Vikings |
| 90 | 28 | Jermaine Carter | 55.40 | 54.00 | 59.20 | 261 | Panthers |
| 91 | 29 | Nicholas Morrow | 55.33 | 46.20 | 58.08 | 728 | Raiders |
| 92 | 30 | A.J. Klein | 54.69 | 47.80 | 56.46 | 754 | Saints |
| 93 | 31 | Corey Nelson | 54.22 | 53.60 | 60.15 | 106 | Broncos |
| 94 | 32 | Myles Jack | 53.96 | 45.90 | 57.76 | 613 | Jaguars |
| 95 | 33 | Christian Jones | 53.79 | 45.50 | 56.91 | 609 | Lions |
| 96 | 34 | Ryan Connelly | 53.32 | 57.10 | 65.14 | 187 | Giants |
| 97 | 35 | Preston Brown | 53.19 | 45.90 | 59.81 | 427 | Jaguars |
| 98 | 36 | Haason Reddick | 52.92 | 40.20 | 57.23 | 690 | Cardinals |
| 99 | 37 | Dylan Cole | 52.61 | 49.70 | 56.32 | 136 | Texans |
| 100 | 38 | Mack Wilson Sr. | 52.03 | 41.40 | 55.98 | 942 | Browns |
| 101 | 39 | Blake Cashman | 51.79 | 49.10 | 60.29 | 424 | Jets |
| 102 | 40 | Matthew Adams | 51.75 | 44.70 | 57.75 | 105 | Colts |
| 103 | 41 | Cody Barton | 51.58 | 48.70 | 57.67 | 151 | Seahawks |
| 104 | 42 | Christian Kirksey | 50.53 | 48.20 | 58.01 | 112 | Browns |
| 105 | 43 | Jarrad Davis | 50.29 | 38.60 | 56.94 | 654 | Lions |
| 106 | 44 | Austin Calitro | 50.08 | 44.30 | 57.59 | 234 | Jaguars |
| 107 | 45 | Joe Walker | 49.99 | 46.20 | 56.48 | 537 | Cardinals |
| 108 | 46 | Azeez Al-Shaair | 48.95 | 42.20 | 55.53 | 174 | 49ers |
| 109 | 47 | Neville Hewitt | 47.88 | 40.40 | 55.46 | 762 | Jets |
| 110 | 48 | Darron Lee | 47.70 | 36.30 | 56.24 | 161 | Chiefs |
| 111 | 49 | Najee Goode | 46.21 | 39.00 | 55.09 | 295 | Jaguars |
| 112 | 50 | Tyrell Adams | 45.15 | 45.70 | 58.01 | 108 | Texans |
| 113 | 51 | Donald Payne | 45.00 | 26.90 | 56.06 | 348 | Jaguars |
| 114 | 52 | Troy Reeder | 45.00 | 28.60 | 54.77 | 298 | Rams |
| 115 | 53 | Quincy Williams | 45.00 | 33.20 | 54.60 | 494 | Jaguars |

## QB — Quarterback

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Drew Brees | 85.53 | 91.07 | 81.42 | 415 | Saints |
| 2 | 2 | Russell Wilson | 85.41 | 87.65 | 79.78 | 659 | Seahawks |
| 3 | 3 | Patrick Mahomes | 83.71 | 85.27 | 81.18 | 591 | Chiefs |
| 4 | 4 | Kirk Cousins | 80.94 | 81.90 | 78.08 | 515 | Vikings |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Deshaun Watson | 78.50 | 77.25 | 76.42 | 628 | Texans |
| 6 | 2 | Matthew Stafford | 78.49 | 79.03 | 79.11 | 336 | Lions |
| 7 | 3 | Aaron Rodgers | 78.43 | 81.60 | 72.29 | 681 | Packers |
| 8 | 4 | Derek Carr | 78.07 | 77.62 | 75.10 | 578 | Raiders |
| 9 | 5 | Dak Prescott | 77.70 | 75.03 | 75.50 | 679 | Cowboys |
| 10 | 6 | Philip Rivers | 77.61 | 77.96 | 73.28 | 682 | Chargers |
| 11 | 7 | Matt Ryan | 76.83 | 77.30 | 71.90 | 732 | Falcons |
| 12 | 8 | Tom Brady | 76.68 | 81.90 | 67.96 | 688 | Patriots |
| 13 | 9 | Jared Goff | 75.54 | 75.54 | 71.13 | 709 | Rams |
| 14 | 10 | Carson Wentz | 75.10 | 76.76 | 70.71 | 718 | Eagles |

### Starter (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Ryan Tannehill | 73.80 | 77.98 | 84.45 | 356 | Titans |
| 16 | 2 | Lamar Jackson | 73.65 | 81.53 | 77.68 | 491 | Ravens |
| 17 | 3 | Jimmy Garoppolo | 73.53 | 75.20 | 76.52 | 565 | 49ers |
| 18 | 4 | Baker Mayfield | 72.45 | 74.64 | 67.09 | 647 | Browns |
| 19 | 5 | Ryan Fitzpatrick | 71.82 | 76.33 | 67.82 | 620 | Dolphins |
| 20 | 6 | Jameis Winston | 70.70 | 67.61 | 70.25 | 747 | Buccaneers |
| 21 | 7 | Andy Dalton | 66.47 | 69.23 | 62.05 | 609 | Bengals |
| 22 | 8 | Marcus Mariota | 66.43 | 67.83 | 70.17 | 214 | Titans |
| 23 | 9 | Gardner Minshew | 66.20 | 70.20 | 66.92 | 598 | Jaguars |
| 24 | 10 | Sam Darnold | 65.00 | 63.92 | 65.22 | 521 | Jets |
| 25 | 11 | Case Keenum | 65.00 | 63.77 | 68.11 | 290 | Commanders |
| 26 | 12 | Kyler Murray | 64.65 | 61.10 | 66.82 | 661 | Cardinals |
| 27 | 13 | Mitch Trubisky | 64.47 | 62.51 | 63.58 | 613 | Bears |
| 28 | 14 | Daniel Jones | 63.81 | 65.10 | 65.53 | 566 | Giants |
| 29 | 15 | Joe Flacco | 63.55 | 66.88 | 64.01 | 315 | Broncos |
| 30 | 16 | Eli Manning | 63.14 | 64.45 | 66.38 | 160 | Giants |
| 31 | 17 | Josh Allen | 62.94 | 61.04 | 63.07 | 576 | Bills |
| 32 | 18 | Teddy Bridgewater | 62.75 | 71.30 | 70.84 | 236 | Saints |

### Rotation/backup (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Nick Foles | 61.68 | 68.12 | 67.32 | 137 | Jaguars |
| 34 | 2 | Matt Moore | 60.57 | 59.44 | 71.85 | 104 | Chiefs |
| 35 | 3 | Jacoby Brissett | 60.47 | 58.66 | 63.79 | 545 | Colts |
| 36 | 4 | Dwayne Haskins | 60.00 | 67.00 | 60.57 | 259 | Commanders |
| 37 | 5 | Drew Lock | 58.71 | 57.40 | 65.40 | 186 | Broncos |
| 38 | 6 | Kyle Allen | 58.19 | 51.42 | 63.36 | 578 | Panthers |
| 39 | 7 | Mason Rudolph | 57.22 | 53.43 | 60.84 | 331 | Steelers |
| 40 | 8 | David Blough | 56.22 | 55.40 | 56.35 | 210 | Lions |
| 41 | 9 | Devlin Hodges | 55.92 | 49.40 | 60.36 | 196 | Steelers |
| 42 | 10 | Jeff Driskel | 55.51 | 51.12 | 58.21 | 141 | Lions |
| 43 | 11 | Brandon Allen | 54.10 | 46.74 | 56.10 | 105 | Broncos |
| 44 | 12 | Josh Rosen | 52.47 | 47.74 | 55.90 | 137 | Dolphins |
| 45 | 13 | Ryan Finley | 49.73 | 28.20 | 53.60 | 110 | Bengals |

## S — Safety

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Justin Simmons | 93.34 | 91.10 | 91.30 | 1053 | Broncos |
| 2 | 2 | Anthony Harris | 91.30 | 91.60 | 91.41 | 910 | Vikings |
| 3 | 3 | Harrison Smith | 90.22 | 91.40 | 85.79 | 971 | Vikings |
| 4 | 4 | Jamal Adams | 90.11 | 87.50 | 88.72 | 959 | Jets |
| 5 | 5 | Marcus Williams | 88.33 | 89.20 | 84.10 | 950 | Saints |
| 6 | 6 | Devin McCourty | 87.67 | 89.10 | 82.55 | 946 | Patriots |
| 7 | 7 | Tre Boston | 87.13 | 90.60 | 81.28 | 1104 | Panthers |
| 8 | 8 | Adrian Phillips | 83.75 | 86.90 | 82.38 | 282 | Chargers |
| 9 | 9 | Micah Hyde | 83.34 | 80.90 | 81.11 | 969 | Bills |
| 10 | 10 | Earl Thomas III | 81.09 | 84.70 | 79.20 | 891 | Ravens |
| 11 | 11 | Justin Reid | 80.98 | 79.30 | 78.58 | 916 | Texans |
| 12 | 12 | Kareem Jackson | 80.79 | 79.80 | 79.24 | 842 | Broncos |
| 13 | 13 | Jimmie Ward | 80.72 | 81.00 | 82.00 | 806 | 49ers |
| 14 | 14 | Adrian Amos | 80.61 | 76.00 | 80.35 | 1036 | Packers |
| 15 | 15 | Derwin James Jr. | 80.52 | 83.90 | 81.26 | 299 | Chargers |
| 16 | 16 | Chuck Clark | 80.34 | 81.90 | 77.53 | 745 | Ravens |
| 17 | 17 | Tyrann Mathieu | 80.32 | 81.60 | 75.30 | 1080 | Chiefs |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Jarrod Wilson | 79.97 | 79.60 | 77.51 | 1052 | Jaguars |
| 19 | 2 | Minkah Fitzpatrick | 79.92 | 77.40 | 77.43 | 1046 | Steelers |
| 20 | 3 | Kevin Byard | 79.62 | 71.70 | 80.74 | 1098 | Titans |
| 21 | 4 | Ha Ha Clinton-Dix | 79.10 | 75.50 | 77.33 | 1066 | Bears |
| 22 | 5 | Juan Thornhill | 77.49 | 78.00 | 72.99 | 996 | Chiefs |
| 23 | 6 | Tracy Walker III | 77.31 | 76.00 | 75.96 | 843 | Lions |
| 24 | 7 | Xavier Woods | 76.97 | 77.60 | 73.74 | 978 | Cowboys |
| 25 | 8 | Quandre Diggs | 76.52 | 79.30 | 73.63 | 606 | Seahawks |
| 26 | 9 | Duron Harmon | 76.50 | 74.70 | 73.54 | 657 | Patriots |
| 27 | 10 | Marcus Maye | 76.38 | 77.40 | 74.67 | 1089 | Jets |
| 28 | 11 | Eric Weddle | 76.29 | 70.50 | 75.99 | 1031 | Rams |
| 29 | 12 | Jordan Poyer | 75.94 | 70.10 | 75.67 | 977 | Bills |
| 30 | 13 | Andrew Sendejo | 75.71 | 78.30 | 75.02 | 384 | Vikings |
| 31 | 14 | Tavon Wilson | 74.91 | 74.00 | 73.24 | 840 | Lions |
| 32 | 15 | Damarious Randall | 74.45 | 69.90 | 76.23 | 723 | Browns |
| 33 | 16 | Marquise Blair | 74.32 | 80.80 | 73.14 | 230 | Seahawks |
| 34 | 17 | Jeff Heath | 74.28 | 68.90 | 75.46 | 719 | Cowboys |

### Starter (46 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Troy Apke | 73.26 | 74.20 | 73.02 | 210 | Commanders |
| 36 | 2 | Budda Baker | 72.88 | 64.80 | 74.10 | 1120 | Cardinals |
| 37 | 3 | Darnell Savage | 72.05 | 73.10 | 69.26 | 865 | Packers |
| 38 | 4 | C.J. Gardner-Johnson | 72.03 | 68.80 | 70.02 | 547 | Saints |
| 39 | 5 | Malcolm Jenkins | 71.96 | 67.50 | 70.77 | 1015 | Eagles |
| 40 | 6 | Ronnie Harrison | 71.51 | 68.50 | 71.43 | 833 | Jaguars |
| 41 | 7 | Clayton Fejedelem | 71.00 | 70.20 | 71.73 | 111 | Bengals |
| 42 | 8 | Jalen Thompson | 70.77 | 70.30 | 71.09 | 607 | Cardinals |
| 43 | 9 | Rodney McLeod | 70.19 | 70.60 | 69.81 | 1013 | Eagles |
| 44 | 10 | Erik Harris | 69.85 | 66.70 | 68.81 | 900 | Raiders |
| 45 | 11 | Curtis Riley | 69.81 | 67.70 | 70.30 | 275 | Raiders |
| 46 | 12 | Reshad Jones | 69.81 | 66.20 | 74.92 | 189 | Dolphins |
| 47 | 13 | Antoine Bethea | 69.57 | 62.40 | 70.39 | 1107 | Giants |
| 48 | 14 | Eddie Jackson | 69.35 | 66.10 | 67.99 | 1061 | Bears |
| 49 | 15 | Taylor Rapp | 68.52 | 61.40 | 70.13 | 823 | Rams |
| 50 | 16 | Andrew Adams | 68.45 | 63.60 | 71.69 | 616 | Buccaneers |
| 51 | 17 | Khari Willis | 68.26 | 66.40 | 67.41 | 620 | Colts |
| 52 | 18 | Jessie Bates III | 68.14 | 64.90 | 66.14 | 1059 | Bengals |
| 53 | 19 | Rayshawn Jenkins | 67.92 | 67.50 | 66.53 | 964 | Chargers |
| 54 | 20 | Jaquiski Tartt | 67.92 | 67.90 | 69.82 | 673 | 49ers |
| 55 | 21 | Karl Joseph | 67.90 | 64.70 | 70.65 | 575 | Raiders |
| 56 | 22 | Marqui Christian | 67.65 | 70.20 | 63.66 | 371 | Rams |
| 57 | 23 | Jabrill Peppers | 67.63 | 66.70 | 67.31 | 705 | Giants |
| 58 | 24 | George Odum | 67.61 | 64.10 | 69.44 | 284 | Colts |
| 59 | 25 | Michael Thomas | 67.54 | 65.70 | 66.49 | 302 | Giants |
| 60 | 26 | Bradley McDougald | 67.21 | 64.50 | 65.37 | 941 | Seahawks |
| 61 | 27 | Tashaun Gipson Sr. | 67.20 | 64.50 | 65.86 | 868 | Texans |
| 62 | 28 | Bobby McCain | 66.66 | 69.60 | 67.83 | 540 | Dolphins |
| 63 | 29 | Terrence Brooks | 66.55 | 61.30 | 71.08 | 274 | Patriots |
| 64 | 30 | Amani Hooker | 66.43 | 66.30 | 62.35 | 335 | Titans |
| 65 | 31 | Landon Collins | 66.27 | 60.60 | 67.86 | 1057 | Commanders |
| 66 | 32 | Malik Hooker | 66.18 | 62.80 | 68.33 | 789 | Colts |
| 67 | 33 | Terrell Edmunds | 65.63 | 58.30 | 66.35 | 1036 | Steelers |
| 68 | 34 | Sheldrick Redwine | 65.53 | 68.40 | 67.79 | 374 | Browns |
| 69 | 35 | Trey Marshall | 65.39 | 59.10 | 67.37 | 160 | Broncos |
| 70 | 36 | Daniel Sorensen | 65.24 | 65.40 | 63.78 | 563 | Chiefs |
| 71 | 37 | Tarvarius Moore | 65.10 | 62.10 | 66.84 | 234 | 49ers |
| 72 | 38 | Marcus Epps | 64.93 | 63.40 | 69.08 | 110 | Eagles |
| 73 | 39 | Steven Parker | 64.63 | 62.40 | 66.11 | 339 | Dolphins |
| 74 | 40 | Kenny Vaccaro | 64.42 | 56.40 | 67.36 | 1062 | Titans |
| 75 | 41 | Clayton Geathers | 63.96 | 58.70 | 67.88 | 528 | Colts |
| 76 | 42 | Darian Thompson | 63.76 | 62.90 | 65.90 | 425 | Cowboys |
| 77 | 43 | Morgan Burnett | 63.59 | 58.20 | 69.59 | 367 | Browns |
| 78 | 44 | Dean Marlowe | 62.46 | 61.80 | 71.23 | 108 | Bills |
| 79 | 45 | Adrian Colbert | 62.40 | 64.70 | 66.50 | 361 | Dolphins |
| 80 | 46 | Ricardo Allen | 62.27 | 57.30 | 65.49 | 950 | Falcons |

### Rotation/backup (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 81 | 1 | Delano Hill | 61.43 | 65.10 | 61.69 | 300 | Seahawks |
| 82 | 2 | Will Parks | 60.85 | 57.30 | 60.08 | 537 | Broncos |
| 83 | 3 | Patrick Chung | 60.83 | 56.00 | 61.77 | 642 | Patriots |
| 84 | 4 | Jeremy Reaves | 59.52 | 57.40 | 67.84 | 111 | Commanders |
| 85 | 5 | John Johnson III | 59.15 | 56.90 | 61.69 | 395 | Rams |
| 86 | 6 | Marcell Harris | 58.99 | 59.30 | 63.33 | 340 | 49ers |
| 87 | 7 | Mike Edwards | 58.30 | 52.50 | 59.03 | 614 | Buccaneers |
| 88 | 8 | Will Harris | 58.30 | 55.40 | 56.07 | 668 | Lions |
| 89 | 9 | Vonn Bell | 57.57 | 49.40 | 60.42 | 872 | Saints |
| 90 | 10 | Shawn Williams | 57.10 | 51.00 | 58.04 | 1002 | Bengals |
| 91 | 11 | Darian Stewart | 56.70 | 54.40 | 58.85 | 169 | Buccaneers |
| 92 | 12 | Jamal Carter | 56.36 | 53.50 | 59.52 | 105 | Falcons |
| 93 | 13 | Montae Nicholson | 54.96 | 49.30 | 59.66 | 873 | Commanders |
| 94 | 14 | Anthony Levine | 54.86 | 52.60 | 53.23 | 167 | Ravens |
| 95 | 15 | Will Redmond | 54.19 | 52.40 | 58.52 | 271 | Packers |
| 96 | 16 | Ibraheim Campbell | 53.94 | 54.90 | 59.34 | 181 | Packers |
| 97 | 17 | Deionte Thompson | 52.96 | 53.30 | 55.86 | 252 | Cardinals |
| 98 | 18 | Tedric Thompson | 52.64 | 48.30 | 60.11 | 387 | Seahawks |
| 99 | 19 | Kemal Ishmael | 52.62 | 45.20 | 58.50 | 282 | Falcons |
| 100 | 20 | D.J. Swearinger Sr. | 52.60 | 42.00 | 59.47 | 484 | Saints |
| 101 | 21 | Tony Jefferson | 52.16 | 48.90 | 56.51 | 281 | Ravens |
| 102 | 22 | Brandon Wilson | 52.06 | 51.00 | 53.09 | 183 | Bengals |
| 103 | 23 | Eric Reid | 51.29 | 36.10 | 58.81 | 1094 | Panthers |
| 104 | 24 | Andrew Wingard | 50.97 | 50.10 | 53.63 | 185 | Jaguars |
| 105 | 25 | Roderic Teamer | 49.78 | 49.80 | 56.46 | 377 | Chargers |
| 106 | 26 | Jordan Whitehead | 49.28 | 36.50 | 55.34 | 919 | Buccaneers |
| 107 | 27 | Keanu Neal | 47.07 | 42.00 | 57.75 | 166 | Falcons |
| 108 | 28 | Walt Aikens | 45.37 | 39.10 | 55.07 | 104 | Dolphins |

## T — Tackle

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Ryan Ramczyk | 95.89 | 90.80 | 95.11 | 1058 | Saints |
| 2 | 2 | Lane Johnson | 93.84 | 88.80 | 93.03 | 759 | Eagles |
| 3 | 3 | Ronnie Stanley | 93.38 | 88.50 | 92.46 | 938 | Ravens |
| 4 | 4 | La'el Collins | 92.30 | 86.40 | 92.07 | 1000 | Cowboys |
| 5 | 5 | Mitchell Schwartz | 89.70 | 84.00 | 89.33 | 1046 | Chiefs |
| 6 | 6 | Jason Peters | 88.19 | 82.30 | 87.95 | 872 | Eagles |
| 7 | 7 | Jake Rodgers | 87.88 | 81.00 | 88.30 | 117 | Broncos |
| 8 | 8 | Braden Smith | 87.15 | 79.80 | 87.89 | 1075 | Colts |
| 9 | 9 | Anthony Castonzo | 86.41 | 81.30 | 85.65 | 1076 | Colts |
| 10 | 10 | Bryan Bulaga | 86.18 | 77.80 | 87.60 | 898 | Packers |
| 11 | 11 | Terron Armstead | 85.95 | 80.40 | 85.49 | 935 | Saints |
| 12 | 12 | Jack Conklin | 85.85 | 77.90 | 86.99 | 933 | Titans |
| 13 | 13 | David Bakhtiari | 85.20 | 78.50 | 85.50 | 1075 | Packers |
| 14 | 14 | Laremy Tunsil | 85.20 | 75.80 | 87.30 | 915 | Texans |
| 15 | 15 | Jake Matthews | 85.09 | 79.70 | 84.51 | 1163 | Falcons |
| 16 | 16 | Taylor Lewan | 84.25 | 73.40 | 87.32 | 711 | Titans |
| 17 | 17 | Garett Bolles | 83.90 | 76.20 | 84.87 | 1013 | Broncos |
| 18 | 18 | Taylor Moton | 83.70 | 76.20 | 84.54 | 1106 | Panthers |
| 19 | 19 | Tyron Smith | 83.23 | 76.60 | 83.49 | 882 | Cowboys |
| 20 | 20 | Matt Feiler | 83.13 | 75.90 | 83.78 | 995 | Steelers |
| 21 | 21 | Taylor Decker | 83.10 | 75.90 | 83.73 | 1017 | Lions |
| 22 | 22 | Duane Brown | 82.10 | 74.10 | 83.27 | 793 | Seahawks |
| 23 | 23 | Joe Staley | 81.87 | 72.70 | 83.82 | 434 | 49ers |
| 24 | 24 | Dion Dawkins | 81.36 | 73.40 | 82.50 | 1016 | Bills |
| 25 | 25 | Brian O'Neill | 81.12 | 70.70 | 83.90 | 967 | Vikings |
| 26 | 26 | Orlando Brown Jr. | 80.79 | 72.00 | 82.49 | 1105 | Ravens |
| 27 | 27 | Cornelius Lucas | 80.76 | 72.20 | 82.30 | 507 | Bears |
| 28 | 28 | Alejandro Villanueva | 80.72 | 74.00 | 81.04 | 995 | Steelers |
| 29 | 29 | Andrew Whitworth | 80.50 | 72.80 | 81.47 | 1097 | Rams |
| 30 | 30 | Demar Dotson | 80.41 | 70.80 | 82.65 | 1045 | Buccaneers |
| 31 | 31 | Trent Brown | 80.24 | 69.10 | 83.50 | 581 | Raiders |
| 32 | 32 | Riley Reiff | 80.01 | 71.20 | 81.72 | 874 | Vikings |

### Good (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Zach Banner | 79.71 | 78.60 | 76.28 | 216 | Steelers |
| 34 | 2 | Ty Nsekhe | 79.47 | 67.30 | 83.42 | 358 | Bills |
| 35 | 3 | Dennis Kelly | 79.22 | 71.00 | 80.53 | 352 | Titans |
| 36 | 4 | Isaiah Wynn | 79.21 | 69.90 | 81.25 | 502 | Patriots |
| 37 | 5 | Donovan Smith | 78.92 | 70.40 | 80.44 | 1055 | Buccaneers |
| 38 | 6 | Mike McGlinchey | 78.67 | 67.40 | 82.01 | 777 | 49ers |
| 39 | 7 | James Hurst | 77.93 | 68.00 | 80.39 | 194 | Ravens |
| 40 | 8 | Roderick Johnson | 77.91 | 67.90 | 80.42 | 365 | Texans |
| 41 | 9 | Marcus Cannon | 77.86 | 68.00 | 80.26 | 1008 | Patriots |
| 42 | 10 | Cedric Ogbuehi | 77.76 | 67.00 | 80.77 | 156 | Jaguars |
| 43 | 11 | Greg Robinson | 77.14 | 66.30 | 80.20 | 860 | Browns |
| 44 | 12 | Cordy Glenn | 76.62 | 68.40 | 77.94 | 291 | Bengals |
| 45 | 13 | Kelvin Beachum | 76.47 | 67.10 | 78.55 | 805 | Jets |
| 46 | 14 | Morgan Moses | 76.03 | 65.20 | 79.08 | 858 | Commanders |
| 47 | 15 | David Sharpe | 75.78 | 64.90 | 78.86 | 268 | Raiders |
| 48 | 16 | Russell Okung | 75.60 | 62.40 | 80.23 | 257 | Chargers |
| 49 | 17 | Donald Penn | 75.58 | 64.00 | 79.13 | 885 | Commanders |
| 50 | 18 | D.J. Humphries | 75.53 | 64.50 | 78.71 | 1046 | Cardinals |
| 51 | 19 | Eric Fisher | 75.02 | 64.50 | 77.86 | 467 | Chiefs |
| 52 | 20 | Jawaan Taylor | 74.98 | 63.70 | 78.34 | 1087 | Jaguars |
| 53 | 21 | Kolton Miller | 74.84 | 65.00 | 77.24 | 1019 | Raiders |
| 54 | 22 | Bobby Massie | 74.82 | 63.20 | 78.40 | 612 | Bears |
| 55 | 23 | Tyrell Crosby | 74.79 | 62.40 | 78.88 | 397 | Lions |
| 56 | 24 | Mike Remmers | 74.52 | 64.20 | 77.24 | 870 | Giants |
| 57 | 25 | Justin Skule | 74.32 | 62.30 | 78.17 | 545 | 49ers |
| 58 | 26 | Justin Murray | 74.15 | 62.90 | 77.48 | 844 | Cardinals |
| 59 | 27 | Brandon Shell | 74.11 | 63.60 | 76.95 | 806 | Jets |
| 60 | 28 | Trey Pipkins III | 74.00 | 63.30 | 76.96 | 251 | Chargers |

### Starter (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 61 | 1 | Geron Christian | 73.64 | 63.00 | 76.56 | 146 | Commanders |
| 62 | 2 | Nate Solder | 73.54 | 64.90 | 75.14 | 1011 | Giants |
| 63 | 3 | George Fant | 73.32 | 62.20 | 76.56 | 472 | Seahawks |
| 64 | 4 | Marshall Newhouse | 73.10 | 62.60 | 75.94 | 729 | Patriots |
| 65 | 5 | Sam Tevi | 72.60 | 59.80 | 76.97 | 783 | Chargers |
| 66 | 6 | Elijah Wilkinson | 72.02 | 59.70 | 76.06 | 833 | Broncos |
| 67 | 7 | Charles Leno Jr. | 71.47 | 58.60 | 75.89 | 1066 | Bears |
| 68 | 8 | Jesse Davis | 71.28 | 58.90 | 75.37 | 975 | Dolphins |
| 69 | 9 | Dennis Daley | 71.24 | 57.70 | 76.10 | 686 | Panthers |
| 70 | 10 | Cam Fleming | 71.07 | 59.40 | 74.68 | 258 | Cowboys |
| 71 | 11 | Rick Wagner | 70.89 | 58.60 | 74.92 | 753 | Lions |
| 72 | 12 | Bobby Hart | 70.87 | 57.60 | 75.55 | 1086 | Bengals |
| 73 | 13 | Andre Dillard | 70.74 | 59.70 | 73.94 | 337 | Eagles |
| 74 | 14 | Germain Ifedi | 70.26 | 56.20 | 75.47 | 1107 | Seahawks |
| 75 | 15 | Patrick Omameh | 70.17 | 57.90 | 74.18 | 156 | Saints |
| 76 | 16 | Daryl Williams | 69.43 | 56.10 | 74.15 | 838 | Panthers |
| 77 | 17 | Julie'n Davenport | 68.76 | 56.50 | 72.77 | 534 | Dolphins |
| 78 | 18 | Cam Robinson | 68.71 | 54.80 | 73.82 | 870 | Jaguars |
| 79 | 19 | Kaleb McGary | 67.97 | 53.00 | 73.78 | 1105 | Falcons |
| 80 | 20 | Cody Ford | 67.21 | 52.40 | 72.91 | 739 | Bills |
| 81 | 21 | Chris Clark | 67.07 | 53.20 | 72.15 | 342 | Texans |
| 82 | 22 | Rob Havenstein | 66.63 | 50.90 | 72.95 | 616 | Rams |
| 83 | 23 | Chris Hubbard | 66.60 | 50.50 | 73.16 | 891 | Browns |
| 84 | 24 | Greg Little | 66.52 | 52.70 | 71.56 | 224 | Panthers |
| 85 | 25 | Chuma Edoga | 66.06 | 49.50 | 72.93 | 421 | Jets |
| 86 | 26 | Bobby Evans | 65.42 | 49.40 | 71.94 | 472 | Rams |
| 87 | 27 | Trent Scott | 65.09 | 49.40 | 71.39 | 827 | Chargers |
| 88 | 28 | Cameron Erving | 63.54 | 44.80 | 71.87 | 589 | Chiefs |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 89 | 1 | Brandon Parker | 60.32 | 39.30 | 70.16 | 193 | Raiders |
| 90 | 2 | Josh Wells | 60.03 | 39.00 | 69.89 | 203 | Buccaneers |
| 91 | 3 | J'Marcus Webb | 57.73 | 34.40 | 69.11 | 543 | Dolphins |
| 92 | 4 | Alex Light | 56.12 | 46.80 | 58.17 | 151 | Packers |

## TE — Tight End

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | George Kittle | 87.55 | 95.00 | 78.41 | 412 | 49ers |
| 2 | 2 | Mark Andrews | 85.27 | 90.80 | 77.42 | 311 | Ravens |
| 3 | 3 | Travis Kelce | 84.35 | 85.10 | 79.68 | 666 | Chiefs |
| 4 | 4 | Will Dissly | 83.02 | 79.60 | 81.13 | 137 | Seahawks |
| 5 | 5 | Darren Waller | 80.42 | 83.20 | 74.40 | 557 | Raiders |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Tyler Higbee | 79.46 | 86.10 | 70.86 | 402 | Rams |
| 7 | 2 | Dallas Goedert | 79.30 | 83.20 | 72.53 | 443 | Eagles |
| 8 | 3 | Hayden Hurst | 77.74 | 74.10 | 76.00 | 216 | Ravens |
| 9 | 4 | Maxx Williams | 77.68 | 79.10 | 72.56 | 226 | Cardinals |
| 10 | 5 | Austin Hooper | 76.00 | 78.30 | 70.30 | 544 | Falcons |
| 11 | 6 | Hunter Henry | 75.96 | 73.20 | 73.64 | 432 | Chargers |
| 12 | 7 | Jaeden Graham | 75.75 | 65.40 | 78.48 | 133 | Falcons |
| 13 | 8 | Mo Alie-Cox | 75.39 | 67.30 | 76.61 | 127 | Colts |
| 14 | 9 | Jared Cook | 74.99 | 75.90 | 70.22 | 375 | Saints |
| 15 | 10 | Greg Olsen | 74.76 | 66.90 | 75.83 | 498 | Panthers |
| 16 | 11 | Zach Ertz | 74.74 | 73.40 | 71.46 | 600 | Eagles |
| 17 | 12 | Marcedes Lewis | 74.26 | 71.00 | 72.27 | 226 | Packers |

### Starter (53 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Tyler Eifert | 73.82 | 65.70 | 75.06 | 408 | Bengals |
| 19 | 2 | Jack Doyle | 73.67 | 69.20 | 72.49 | 426 | Colts |
| 20 | 3 | Nick Boyle | 73.39 | 73.50 | 69.15 | 273 | Ravens |
| 21 | 4 | Gerald Everett | 73.32 | 76.00 | 67.36 | 298 | Rams |
| 22 | 5 | Kyle Rudolph | 73.17 | 73.00 | 69.11 | 436 | Vikings |
| 23 | 6 | Eric Ebron | 72.38 | 71.10 | 69.07 | 233 | Colts |
| 24 | 7 | J.P. Holtz | 72.30 | 66.90 | 71.73 | 126 | Bears |
| 25 | 8 | Blake Jarwin | 72.25 | 61.40 | 75.32 | 236 | Cowboys |
| 26 | 9 | MyCole Pruitt | 71.73 | 61.50 | 74.38 | 122 | Titans |
| 27 | 10 | Evan Engram | 71.63 | 64.10 | 72.48 | 334 | Giants |
| 28 | 11 | Jonnu Smith | 71.33 | 72.00 | 66.71 | 325 | Titans |
| 29 | 12 | Foster Moreau | 71.26 | 70.80 | 67.40 | 175 | Raiders |
| 30 | 13 | Seth DeValve | 71.12 | 57.90 | 75.77 | 153 | Jaguars |
| 31 | 14 | Rhett Ellison | 70.80 | 65.60 | 70.10 | 181 | Giants |
| 32 | 15 | James O'Shaughnessy | 70.41 | 62.50 | 71.52 | 130 | Jaguars |
| 33 | 16 | Cameron Brate | 70.16 | 66.30 | 68.56 | 313 | Buccaneers |
| 34 | 17 | Jacob Hollister | 70.13 | 67.30 | 67.85 | 347 | Seahawks |
| 35 | 18 | Josh Hill | 70.03 | 67.50 | 67.55 | 317 | Saints |
| 36 | 19 | Ricky Seals-Jones | 70.00 | 56.80 | 74.64 | 191 | Browns |
| 37 | 20 | Luke Willson | 69.67 | 61.90 | 70.68 | 113 | Seahawks |
| 38 | 21 | Jimmy Graham | 69.62 | 55.00 | 75.20 | 458 | Packers |
| 39 | 22 | O.J. Howard | 68.94 | 54.70 | 74.26 | 471 | Buccaneers |
| 40 | 23 | Delanie Walker | 68.91 | 64.00 | 68.02 | 156 | Titans |
| 41 | 24 | Nick O'Leary | 68.86 | 51.60 | 76.20 | 182 | Jaguars |
| 42 | 25 | Jeff Heuerman | 68.79 | 60.00 | 70.48 | 194 | Broncos |
| 43 | 26 | Charles Clay | 68.75 | 64.70 | 67.28 | 192 | Cardinals |
| 44 | 27 | Mike Gesicki | 68.56 | 60.50 | 69.76 | 571 | Dolphins |
| 45 | 28 | Darren Fells | 68.32 | 60.10 | 69.64 | 402 | Texans |
| 46 | 29 | Chris Manhertz | 68.16 | 62.10 | 68.04 | 140 | Panthers |
| 47 | 30 | Ryan Griffin | 67.97 | 61.90 | 67.85 | 404 | Jets |
| 48 | 31 | Irv Smith Jr. | 67.75 | 65.00 | 65.42 | 341 | Vikings |
| 49 | 32 | Jason Witten | 67.59 | 59.40 | 68.89 | 522 | Cowboys |
| 50 | 33 | T.J. Hockenson | 67.05 | 60.50 | 67.25 | 344 | Lions |
| 51 | 34 | Jordan Akins | 66.62 | 55.10 | 70.14 | 414 | Texans |
| 52 | 35 | Lee Smith | 66.57 | 60.80 | 66.25 | 124 | Bills |
| 53 | 36 | Virgil Green | 66.24 | 58.00 | 67.56 | 200 | Chargers |
| 54 | 37 | Ryan Izzo | 66.21 | 46.70 | 75.05 | 133 | Patriots |
| 55 | 38 | Dawson Knox | 66.07 | 60.00 | 65.95 | 377 | Bills |
| 56 | 39 | Benjamin Watson | 65.80 | 53.60 | 69.77 | 278 | Patriots |
| 57 | 40 | Vernon Davis | 65.72 | 49.40 | 72.43 | 148 | Commanders |
| 58 | 41 | Tyler Conklin | 65.70 | 53.70 | 69.53 | 104 | Vikings |
| 59 | 42 | Tyler Kroft | 65.53 | 54.70 | 68.59 | 150 | Bills |
| 60 | 43 | Ben Koyack | 65.21 | 55.70 | 67.39 | 105 | Jaguars |
| 61 | 44 | Luke Stocker | 65.17 | 63.00 | 62.45 | 197 | Falcons |
| 62 | 45 | Noah Fant | 65.09 | 52.00 | 69.65 | 432 | Broncos |
| 63 | 46 | Ian Thomas | 64.09 | 51.80 | 68.12 | 206 | Panthers |
| 64 | 47 | Matt LaCosse | 63.54 | 54.00 | 65.74 | 204 | Patriots |
| 65 | 48 | Logan Thomas | 63.52 | 55.10 | 64.97 | 201 | Lions |
| 66 | 49 | Kaden Smith | 62.99 | 54.50 | 64.49 | 272 | Giants |
| 67 | 50 | Durham Smythe | 62.33 | 52.70 | 64.59 | 249 | Dolphins |
| 68 | 51 | Jesse James | 62.25 | 53.70 | 63.78 | 268 | Lions |
| 69 | 52 | Vance McDonald | 62.16 | 45.30 | 69.23 | 449 | Steelers |
| 70 | 53 | Blake Bell | 62.14 | 50.00 | 66.06 | 209 | Chiefs |

### Rotation/backup (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 71 | 1 | Jeremy Sprinkle | 61.84 | 49.80 | 65.70 | 374 | Commanders |
| 72 | 2 | Trey Burton | 61.77 | 49.20 | 65.99 | 201 | Bears |
| 73 | 3 | Ross Dwelley | 61.71 | 56.10 | 61.29 | 152 | 49ers |
| 74 | 4 | Daniel Brown | 60.51 | 49.20 | 63.89 | 139 | Jets |
| 75 | 5 | C.J. Uzomah | 60.48 | 49.40 | 63.70 | 327 | Bengals |
| 76 | 6 | Nick Vannett | 59.84 | 48.20 | 63.44 | 219 | Steelers |
| 77 | 7 | Geoff Swaim | 59.49 | 46.50 | 63.99 | 116 | Jaguars |
| 78 | 8 | Demetrius Harris | 59.47 | 46.10 | 64.22 | 319 | Browns |

## WR — Wide Receiver

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Godwin | 88.61 | 90.70 | 83.05 | 633 | Buccaneers |
| 2 | 2 | A.J. Brown | 88.38 | 84.00 | 87.14 | 416 | Titans |
| 3 | 3 | Julio Jones | 87.91 | 90.60 | 81.95 | 610 | Falcons |
| 4 | 4 | Michael Thomas | 86.62 | 90.40 | 79.93 | 639 | Saints |
| 5 | 5 | Tyreek Hill | 86.55 | 85.50 | 83.08 | 381 | Chiefs |
| 6 | 6 | Mike Evans | 84.91 | 85.70 | 80.22 | 539 | Buccaneers |
| 7 | 7 | DeAndre Hopkins | 84.35 | 87.30 | 78.22 | 621 | Texans |
| 8 | 8 | Robert Woods | 83.24 | 82.20 | 79.77 | 655 | Rams |
| 9 | 9 | Terry McLaurin | 83.18 | 85.70 | 77.34 | 485 | Commanders |
| 10 | 10 | Kenny Golladay | 83.08 | 79.90 | 81.04 | 626 | Lions |
| 11 | 11 | Courtland Sutton | 82.58 | 83.10 | 78.06 | 576 | Broncos |
| 12 | 12 | Amari Cooper | 82.37 | 84.20 | 76.98 | 549 | Cowboys |
| 13 | 13 | DJ Moore | 82.31 | 82.20 | 78.22 | 607 | Panthers |
| 14 | 14 | Davante Adams | 81.73 | 83.50 | 76.39 | 456 | Packers |
| 15 | 15 | Stefon Diggs | 81.53 | 78.80 | 79.19 | 453 | Vikings |
| 16 | 16 | Keenan Allen | 81.13 | 80.80 | 77.18 | 643 | Chargers |
| 17 | 17 | T.Y. Hilton | 80.95 | 75.10 | 80.68 | 298 | Colts |
| 18 | 18 | DeVante Parker | 80.69 | 79.20 | 77.51 | 675 | Dolphins |
| 19 | 19 | Mecole Hardman Jr. | 80.44 | 69.10 | 83.83 | 330 | Chiefs |
| 20 | 20 | Andy Isabella | 80.03 | 64.20 | 86.42 | 102 | Cardinals |

### Good (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Tyler Lockett | 79.70 | 77.10 | 77.27 | 615 | Seahawks |
| 22 | 2 | Allen Robinson II | 79.70 | 80.80 | 74.80 | 670 | Bears |
| 23 | 3 | Cooper Kupp | 79.55 | 74.60 | 78.69 | 619 | Rams |
| 24 | 4 | Adam Thielen | 79.46 | 75.90 | 77.66 | 244 | Vikings |
| 25 | 5 | Mike Williams | 79.32 | 74.10 | 78.64 | 567 | Chargers |
| 26 | 6 | Emmanuel Sanders | 79.32 | 78.50 | 75.70 | 526 | 49ers |
| 27 | 7 | Jarvis Landry | 78.80 | 78.20 | 75.03 | 623 | Browns |
| 28 | 8 | Will Fuller V | 78.63 | 75.40 | 76.62 | 351 | Texans |
| 29 | 9 | Breshad Perriman | 78.55 | 72.80 | 78.22 | 472 | Buccaneers |
| 30 | 10 | John Brown | 78.47 | 75.80 | 76.09 | 572 | Bills |
| 31 | 11 | DJ Chark Jr. | 78.28 | 75.20 | 76.16 | 639 | Jaguars |
| 32 | 12 | Hunter Renfrow | 78.11 | 75.50 | 75.69 | 303 | Raiders |
| 33 | 13 | Michael Gallup | 78.02 | 74.00 | 76.54 | 541 | Cowboys |
| 34 | 14 | Deebo Samuel | 77.75 | 74.50 | 75.75 | 419 | 49ers |
| 35 | 15 | Golden Tate | 77.12 | 74.00 | 75.04 | 453 | Giants |
| 36 | 16 | Zach Pascal | 77.11 | 73.70 | 75.22 | 448 | Colts |
| 37 | 17 | Calvin Ridley | 77.00 | 74.90 | 74.23 | 553 | Falcons |
| 38 | 18 | Darius Slayton | 76.77 | 70.30 | 76.92 | 503 | Giants |
| 39 | 19 | David Moore | 76.69 | 67.00 | 78.99 | 205 | Seahawks |
| 40 | 20 | Alshon Jeffery | 76.67 | 74.80 | 73.75 | 305 | Eagles |
| 41 | 21 | Marvin Jones Jr. | 76.34 | 72.90 | 74.47 | 545 | Lions |
| 42 | 22 | Odell Beckham Jr. | 76.22 | 68.70 | 77.07 | 628 | Browns |
| 43 | 23 | Marquise Brown | 76.20 | 70.80 | 75.64 | 338 | Ravens |
| 44 | 24 | Tyler Boyd | 76.11 | 73.00 | 74.01 | 675 | Bengals |
| 45 | 25 | Kenny Stills | 75.89 | 72.20 | 74.19 | 396 | Texans |
| 46 | 26 | Allen Lazard | 75.84 | 72.30 | 74.04 | 319 | Packers |
| 47 | 27 | Brandin Cooks | 75.70 | 68.10 | 76.60 | 482 | Rams |
| 48 | 28 | D.K. Metcalf | 75.60 | 69.70 | 75.37 | 580 | Seahawks |
| 49 | 29 | Sterling Shepard | 75.56 | 75.80 | 71.24 | 416 | Giants |
| 50 | 30 | Corey Davis | 75.41 | 69.90 | 74.92 | 437 | Titans |
| 51 | 31 | James Washington | 75.30 | 69.30 | 75.13 | 448 | Steelers |
| 52 | 32 | Malik Turner | 75.19 | 68.00 | 75.81 | 143 | Seahawks |
| 53 | 33 | Josh Gordon | 75.03 | 63.70 | 78.42 | 316 | Seahawks |
| 54 | 34 | Tajae Sharpe | 74.79 | 74.20 | 71.01 | 248 | Titans |
| 55 | 35 | Julian Edelman | 74.67 | 72.00 | 72.28 | 656 | Patriots |
| 56 | 36 | Jamison Crowder | 74.49 | 73.00 | 71.32 | 562 | Jets |
| 57 | 37 | Scott Miller | 74.46 | 63.30 | 77.73 | 149 | Buccaneers |
| 58 | 38 | Tyrell Williams | 74.41 | 66.50 | 75.52 | 428 | Raiders |
| 59 | 39 | Cole Beasley | 74.31 | 73.60 | 70.61 | 499 | Bills |
| 60 | 40 | Sammy Watkins | 74.29 | 66.50 | 75.31 | 515 | Chiefs |
| 61 | 41 | Jake Kumerow | 74.17 | 63.40 | 77.19 | 196 | Packers |
| 62 | 42 | Larry Fitzgerald | 74.16 | 70.30 | 72.57 | 619 | Cardinals |

### Starter (77 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 63 | 1 | Richie James | 73.99 | 62.40 | 77.55 | 102 | 49ers |
| 64 | 2 | Auden Tate | 73.98 | 70.50 | 72.14 | 437 | Bengals |
| 65 | 3 | Josh Reynolds | 73.97 | 65.90 | 75.18 | 290 | Rams |
| 66 | 4 | Diontae Johnson | 73.93 | 67.90 | 73.78 | 453 | Steelers |
| 67 | 5 | Randall Cobb | 73.91 | 69.90 | 72.42 | 499 | Cowboys |
| 68 | 6 | JuJu Smith-Schuster | 73.81 | 63.10 | 76.79 | 406 | Steelers |
| 69 | 7 | Danny Amendola | 73.73 | 71.50 | 71.05 | 476 | Lions |
| 70 | 8 | Keelan Cole Sr. | 73.73 | 66.50 | 74.39 | 275 | Jaguars |
| 71 | 9 | Adam Humphries | 72.97 | 69.60 | 71.05 | 285 | Titans |
| 72 | 10 | Cody Latimer | 72.96 | 68.40 | 71.83 | 268 | Giants |
| 73 | 11 | Preston Williams | 72.95 | 67.70 | 72.29 | 288 | Dolphins |
| 74 | 12 | Greg Ward | 72.91 | 73.40 | 68.42 | 205 | Eagles |
| 75 | 13 | Marquise Goodwin | 72.49 | 59.60 | 76.91 | 142 | 49ers |
| 76 | 14 | Robert Foster | 72.49 | 52.70 | 81.52 | 143 | Bills |
| 77 | 15 | Damion Ratley | 72.38 | 62.90 | 74.53 | 173 | Browns |
| 78 | 16 | Phillip Dorsett | 72.36 | 66.90 | 71.84 | 387 | Patriots |
| 79 | 17 | Tre'Quan Smith | 72.30 | 65.20 | 72.86 | 310 | Saints |
| 80 | 18 | Alex Erickson | 72.28 | 67.10 | 71.56 | 411 | Bengals |
| 81 | 19 | Isaiah Ford | 72.10 | 68.90 | 70.06 | 159 | Dolphins |
| 82 | 20 | Kelvin Harmon | 72.02 | 63.90 | 73.26 | 315 | Commanders |
| 83 | 21 | Anthony Miller | 71.91 | 66.00 | 71.69 | 492 | Bears |
| 84 | 22 | Tim Patrick | 71.75 | 66.70 | 70.95 | 203 | Broncos |
| 85 | 23 | Jakobi Meyers | 71.54 | 64.70 | 71.93 | 288 | Patriots |
| 86 | 24 | Tavon Austin | 71.45 | 59.90 | 74.98 | 165 | Cowboys |
| 87 | 25 | Marquez Valdes-Scantling | 71.18 | 57.00 | 76.46 | 357 | Packers |
| 88 | 26 | Pharoh Cooper | 71.13 | 70.30 | 67.52 | 165 | Cardinals |
| 89 | 27 | Dante Pettis | 71.01 | 56.40 | 76.59 | 157 | 49ers |
| 90 | 28 | Kendrick Bourne | 70.96 | 66.70 | 69.63 | 301 | 49ers |
| 91 | 29 | Demaryius Thomas | 70.93 | 65.90 | 70.11 | 311 | Jets |
| 92 | 30 | Christian Kirk | 70.83 | 62.50 | 72.21 | 536 | Cardinals |
| 93 | 31 | Damiere Byrd | 70.72 | 64.40 | 70.77 | 325 | Cardinals |
| 94 | 32 | Isaiah McKenzie | 70.71 | 68.80 | 67.82 | 246 | Bills |
| 95 | 33 | Vyncint Smith | 70.50 | 62.40 | 71.73 | 183 | Jets |
| 96 | 34 | Justin Hardy | 70.50 | 67.30 | 68.46 | 117 | Falcons |
| 97 | 35 | Chris Conley | 70.36 | 62.70 | 71.30 | 629 | Jaguars |
| 98 | 36 | DeAndre Carter | 70.35 | 60.00 | 73.08 | 130 | Texans |
| 99 | 37 | John Ross | 70.23 | 63.10 | 70.82 | 274 | Bengals |
| 100 | 38 | Jakeem Grant Sr. | 69.96 | 62.10 | 71.03 | 156 | Dolphins |
| 101 | 39 | Albert Wilson | 69.37 | 62.30 | 69.91 | 330 | Dolphins |
| 102 | 40 | Dede Westbrook | 69.30 | 63.50 | 69.00 | 595 | Jaguars |
| 103 | 41 | Taylor Gabriel | 69.26 | 62.00 | 69.93 | 314 | Bears |
| 104 | 42 | Russell Gage | 69.06 | 66.30 | 66.73 | 402 | Falcons |
| 105 | 43 | Seth Roberts | 68.93 | 63.50 | 68.38 | 307 | Ravens |
| 106 | 44 | Jaron Brown | 68.57 | 57.20 | 71.99 | 224 | Seahawks |
| 107 | 45 | Keelan Doss | 68.52 | 58.60 | 70.97 | 102 | Raiders |
| 108 | 46 | Ted Ginn Jr. | 68.50 | 56.80 | 72.14 | 467 | Saints |
| 109 | 47 | Olabisi Johnson | 68.48 | 66.20 | 65.84 | 303 | Vikings |
| 110 | 48 | Chris Hogan | 68.33 | 54.90 | 73.12 | 112 | Panthers |
| 111 | 49 | Willie Snead IV | 68.32 | 61.00 | 69.03 | 326 | Ravens |
| 112 | 50 | Steven Sims | 68.07 | 63.60 | 66.89 | 229 | Commanders |
| 113 | 51 | Allen Hurns | 67.96 | 57.00 | 71.10 | 405 | Dolphins |
| 114 | 52 | Miles Boykin | 67.95 | 60.90 | 68.49 | 192 | Ravens |
| 115 | 53 | Marcus Johnson | 67.87 | 59.80 | 69.09 | 249 | Colts |
| 116 | 54 | Mohamed Sanu | 67.72 | 58.80 | 69.50 | 533 | Patriots |
| 117 | 55 | Dontrelle Inman | 67.61 | 62.90 | 66.58 | 136 | Colts |
| 118 | 56 | Curtis Samuel | 67.59 | 62.70 | 66.68 | 668 | Panthers |
| 119 | 57 | Demarcus Robinson | 67.55 | 57.80 | 69.88 | 529 | Chiefs |
| 120 | 58 | Cordarrelle Patterson | 66.96 | 54.70 | 70.97 | 132 | Bears |
| 121 | 59 | Paul Richardson Jr. | 66.90 | 59.70 | 67.53 | 280 | Commanders |
| 122 | 60 | Rashard Higgins | 66.61 | 53.60 | 71.12 | 113 | Browns |
| 123 | 61 | N'Keal Harry | 66.33 | 66.60 | 61.98 | 136 | Patriots |
| 124 | 62 | KhaDarel Hodge | 66.30 | 54.30 | 70.13 | 103 | Browns |
| 125 | 63 | J.J. Arcega-Whiteside | 66.22 | 53.80 | 70.34 | 323 | Eagles |
| 126 | 64 | Nelson Agholor | 65.95 | 54.40 | 69.48 | 437 | Eagles |
| 127 | 65 | Trevor Davis | 65.86 | 49.30 | 72.74 | 114 | Dolphins |
| 128 | 66 | Justin Watson | 65.84 | 63.90 | 62.97 | 153 | Buccaneers |
| 129 | 67 | Jordan Matthews | 65.31 | 52.80 | 69.49 | 102 | 49ers |
| 130 | 68 | Keke Coutee | 65.12 | 53.10 | 68.97 | 242 | Texans |
| 131 | 69 | Deon Cain | 63.94 | 55.60 | 65.33 | 211 | Steelers |
| 132 | 70 | Parris Campbell | 63.79 | 54.70 | 65.69 | 124 | Colts |
| 133 | 71 | Zay Jones | 63.73 | 53.70 | 66.25 | 394 | Raiders |
| 134 | 72 | Travis Benjamin | 63.38 | 46.40 | 70.54 | 146 | Chargers |
| 135 | 73 | DaeSean Hamilton | 63.38 | 56.40 | 63.87 | 418 | Broncos |
| 136 | 74 | Johnny Holton | 62.94 | 47.40 | 69.13 | 107 | Steelers |
| 137 | 75 | Trent Sherfield | 62.69 | 52.20 | 65.52 | 151 | Cardinals |
| 138 | 76 | Mack Hollins | 62.30 | 49.50 | 66.67 | 243 | Dolphins |
| 139 | 77 | Geronimo Allison | 62.08 | 54.10 | 63.23 | 458 | Packers |

### Rotation/backup (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 140 | 1 | Trey Quinn | 61.98 | 54.10 | 63.06 | 315 | Commanders |
| 141 | 2 | Christian Blake | 61.44 | 51.30 | 64.03 | 232 | Falcons |
| 142 | 3 | Chester Rogers | 61.04 | 50.50 | 63.90 | 242 | Colts |
| 143 | 4 | Javon Wims | 60.76 | 52.00 | 62.44 | 305 | Bears |
| 144 | 5 | KeeSean Johnson | 60.50 | 53.50 | 61.00 | 256 | Cardinals |
| 145 | 6 | Jarius Wright | 60.15 | 47.80 | 64.22 | 528 | Panthers |
| 146 | 7 | Damion Willis | 57.91 | 52.20 | 57.55 | 155 | Bengals |
| 147 | 8 | Andre Patton | 56.18 | 45.70 | 59.00 | 334 | Chargers |
