# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:38:00Z
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
| 4 | 4 | Ben Jones | 84.29 | 76.35 | 85.41 | 918 | Titans |
| 5 | 5 | Brandon Linder | 84.23 | 75.30 | 86.02 | 1083 | Jaguars |
| 6 | 6 | Ryan Kelly | 82.46 | 73.00 | 84.60 | 1018 | Colts |
| 7 | 7 | Alex Mack | 81.74 | 72.10 | 84.00 | 1156 | Falcons |
| 8 | 8 | J.C. Tretter | 80.52 | 72.00 | 82.03 | 1039 | Browns |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Rodney Hudson | 79.77 | 70.72 | 81.63 | 904 | Raiders |
| 10 | 2 | Travis Frederick | 79.40 | 70.00 | 81.50 | 1116 | Cowboys |
| 11 | 3 | Corey Linsley | 78.88 | 69.69 | 80.84 | 950 | Packers |
| 12 | 4 | James Daniels | 78.87 | 69.90 | 80.69 | 1069 | Bears |
| 13 | 5 | Matt Skura | 78.53 | 67.42 | 81.77 | 717 | Ravens |
| 14 | 6 | Chase Roullier | 77.97 | 68.55 | 80.09 | 833 | Commanders |
| 15 | 7 | Mitch Morse | 76.46 | 66.15 | 79.16 | 909 | Bills |
| 16 | 8 | Ted Karras | 75.89 | 64.50 | 79.32 | 1041 | Patriots |
| 17 | 9 | Nick Martin | 75.75 | 65.50 | 78.41 | 1020 | Texans |
| 18 | 10 | Daniel Kilgore | 75.71 | 65.94 | 78.06 | 877 | Dolphins |
| 19 | 11 | Connor McGovern | 75.47 | 60.00 | 81.61 | 1013 | Broncos |
| 20 | 12 | Matt Paradis | 75.37 | 63.40 | 79.19 | 1094 | Panthers |

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Justin Britt | 73.51 | 61.66 | 77.24 | 504 | Seahawks |
| 22 | 2 | Weston Richburg | 73.45 | 62.38 | 76.66 | 835 | 49ers |
| 23 | 3 | Austin Reiter | 73.32 | 63.00 | 76.03 | 1045 | Chiefs |
| 24 | 4 | Trey Hopkins | 72.93 | 62.40 | 75.78 | 1097 | Bengals |
| 25 | 5 | Brian Allen | 72.52 | 58.94 | 77.40 | 563 | Rams |
| 26 | 6 | Garrett Bradbury | 70.44 | 58.10 | 74.50 | 989 | Vikings |
| 27 | 7 | Tony Bergstrom | 70.18 | 60.48 | 72.48 | 226 | Commanders |
| 28 | 8 | Ryan Kalil | 70.15 | 56.61 | 75.01 | 343 | Jets |
| 29 | 9 | Mike Pouncey | 69.84 | 58.68 | 73.12 | 305 | Chargers |
| 30 | 10 | A.Q. Shipley | 69.66 | 57.60 | 73.54 | 1041 | Cardinals |
| 31 | 11 | Jon Halapio | 68.71 | 56.31 | 72.81 | 980 | Giants |
| 32 | 12 | Scott Quessenberry | 68.68 | 58.73 | 71.14 | 625 | Chargers |
| 33 | 13 | Maurkice Pouncey | 65.84 | 52.07 | 70.86 | 777 | Steelers |
| 34 | 14 | Jonotthan Harrison | 65.53 | 52.86 | 69.81 | 679 | Jets |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | Andre James | 54.32 | 45.36 | 56.13 | 116 | Raiders |

## CB — Cornerback

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Richard Sherman | 92.10 | 90.10 | 91.87 | 895 | 49ers |
| 2 | 2 | Marcus Peters | 88.98 | 85.50 | 87.34 | 985 | Ravens |
| 3 | 3 | Quinton Dunbar | 88.37 | 85.43 | 91.99 | 613 | Commanders |
| 4 | 4 | Stephon Gilmore | 88.32 | 85.70 | 85.90 | 952 | Patriots |
| 5 | 5 | Casey Hayward Jr. | 87.53 | 83.00 | 86.38 | 944 | Chargers |
| 6 | 6 | Tramon Williams | 86.11 | 81.67 | 85.53 | 761 | Packers |
| 7 | 7 | Steven Nelson | 81.97 | 80.30 | 80.69 | 1011 | Steelers |
| 8 | 8 | Tre'Davious White | 81.54 | 76.00 | 81.59 | 951 | Bills |
| 9 | 9 | Byron Jones | 81.47 | 74.80 | 82.26 | 917 | Cowboys |
| 10 | 10 | Brian Poole | 81.18 | 79.07 | 80.50 | 750 | Jets |
| 11 | 11 | Marlon Humphrey | 81.06 | 76.20 | 80.77 | 959 | Ravens |
| 12 | 12 | Jaire Alexander | 80.87 | 77.50 | 80.12 | 1027 | Packers |
| 13 | 13 | Shaquill Griffin | 80.65 | 76.00 | 80.84 | 917 | Seahawks |
| 14 | 14 | Jamel Dean | 80.45 | 72.69 | 87.70 | 372 | Buccaneers |
| 15 | 15 | Jason McCourty | 80.40 | 75.04 | 82.30 | 474 | Patriots |

### Good (30 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | D.J. Hayden | 79.58 | 75.85 | 80.30 | 647 | Jaguars |
| 17 | 2 | Nickell Robey-Coleman | 78.46 | 73.48 | 77.61 | 708 | Rams |
| 18 | 3 | Adoree' Jackson | 78.40 | 73.69 | 79.98 | 575 | Titans |
| 19 | 4 | Mike Hilton | 77.97 | 72.23 | 77.95 | 671 | Steelers |
| 20 | 5 | Darius Phillips | 77.75 | 69.52 | 87.53 | 108 | Bengals |
| 21 | 6 | Joe Haden | 77.72 | 71.30 | 78.98 | 1055 | Steelers |
| 22 | 7 | Marshon Lattimore | 77.59 | 68.68 | 80.62 | 820 | Saints |
| 23 | 8 | Jonathan Jones | 77.37 | 69.87 | 78.93 | 619 | Patriots |
| 24 | 9 | Chidobe Awuzie | 77.35 | 70.50 | 79.52 | 1020 | Cowboys |
| 25 | 10 | K'Waun Williams | 77.03 | 74.33 | 76.23 | 603 | 49ers |
| 26 | 11 | D.J. Reed | 76.91 | 68.12 | 82.50 | 125 | 49ers |
| 27 | 12 | Darious Williams | 76.86 | 71.33 | 89.80 | 221 | Rams |
| 28 | 13 | Rashad Fenton | 76.42 | 68.25 | 86.03 | 166 | Chiefs |
| 29 | 14 | J.C. Jackson | 76.36 | 67.64 | 79.17 | 682 | Patriots |
| 30 | 15 | Carlton Davis III | 76.18 | 72.10 | 77.34 | 934 | Buccaneers |
| 31 | 16 | Troy Hill | 76.18 | 70.26 | 79.51 | 538 | Rams |
| 32 | 17 | Denzel Ward | 76.17 | 72.34 | 78.34 | 748 | Browns |
| 33 | 18 | Jalen Ramsey | 76.01 | 68.56 | 78.89 | 780 | Rams |
| 34 | 19 | Prince Amukamara | 75.78 | 69.60 | 76.98 | 891 | Bears |
| 35 | 20 | Tramaine Brock Sr. | 75.46 | 72.03 | 77.95 | 745 | Titans |
| 36 | 21 | Emmanuel Moseley | 75.13 | 66.69 | 76.59 | 577 | 49ers |
| 37 | 22 | Darqueze Dennard | 75.12 | 73.84 | 76.39 | 495 | Bengals |
| 38 | 23 | Chandon Sullivan | 74.82 | 69.77 | 78.19 | 350 | Packers |
| 39 | 24 | Cameron Sutton | 74.79 | 68.04 | 79.19 | 268 | Steelers |
| 40 | 25 | Chris Harris Jr. | 74.77 | 66.80 | 77.17 | 1044 | Broncos |
| 41 | 26 | Desmond Trufant | 74.63 | 69.03 | 77.85 | 521 | Falcons |
| 42 | 27 | Janoris Jenkins | 74.40 | 68.00 | 76.49 | 973 | Saints |
| 43 | 28 | Chris Jones | 74.38 | 67.27 | 81.21 | 275 | Cardinals |
| 44 | 29 | Gareon Conley | 74.29 | 65.11 | 80.82 | 768 | Texans |
| 45 | 30 | Kenny Moore II | 74.25 | 69.80 | 77.64 | 631 | Colts |

### Starter (69 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 46 | 1 | Brandon Carr | 73.95 | 66.60 | 74.68 | 748 | Ravens |
| 47 | 2 | Marvell Tell III | 73.65 | 64.38 | 80.86 | 254 | Colts |
| 48 | 3 | Bradley Roby | 73.54 | 68.41 | 76.23 | 654 | Texans |
| 49 | 4 | James Bradberry | 73.45 | 65.40 | 75.17 | 1020 | Panthers |
| 50 | 5 | Jourdan Lewis | 73.29 | 64.65 | 76.87 | 590 | Cowboys |
| 51 | 6 | Sean Murphy-Bunting | 73.25 | 65.84 | 75.06 | 686 | Buccaneers |
| 52 | 7 | Kevin Johnson | 73.03 | 70.39 | 76.66 | 335 | Bills |
| 53 | 8 | Mackensie Alexander | 72.78 | 63.62 | 76.61 | 534 | Vikings |
| 54 | 9 | Levi Wallace | 72.73 | 68.18 | 75.12 | 785 | Bills |
| 55 | 10 | Malcolm Butler | 72.46 | 63.79 | 77.73 | 579 | Titans |
| 56 | 11 | Johnathan Joseph | 71.81 | 64.60 | 74.11 | 622 | Texans |
| 57 | 12 | Patrick Peterson | 71.45 | 64.09 | 75.32 | 696 | Cardinals |
| 58 | 13 | Kyle Fuller | 71.18 | 58.70 | 75.34 | 1070 | Bears |
| 59 | 14 | Charvarius Ward | 71.06 | 65.60 | 75.22 | 1048 | Chiefs |
| 60 | 15 | Maurice Canady | 70.75 | 67.35 | 76.98 | 397 | Jets |
| 61 | 16 | Logan Ryan | 70.57 | 60.60 | 73.69 | 1098 | Titans |
| 62 | 17 | Rock Ya-Sin | 70.54 | 62.20 | 72.97 | 853 | Colts |
| 63 | 18 | Jimmy Smith | 70.43 | 63.44 | 76.66 | 402 | Ravens |
| 64 | 19 | Amani Oruwariye | 70.29 | 67.81 | 78.65 | 215 | Lions |
| 65 | 20 | Daryl Worley | 70.06 | 62.60 | 73.26 | 939 | Raiders |
| 66 | 21 | Trayvon Mullen | 69.89 | 59.10 | 72.92 | 675 | Raiders |
| 67 | 22 | Ross Cockrell | 69.84 | 61.26 | 73.37 | 733 | Panthers |
| 68 | 23 | Kevin Peterson | 69.48 | 60.72 | 77.30 | 255 | Cardinals |
| 69 | 24 | Sidney Jones IV | 69.40 | 64.95 | 76.63 | 293 | Eagles |
| 70 | 25 | Darius Slay | 68.98 | 56.90 | 74.22 | 858 | Lions |
| 71 | 26 | Desmond King II | 68.88 | 68.08 | 66.28 | 584 | Chargers |
| 72 | 27 | Justin Coleman | 68.82 | 58.80 | 71.54 | 963 | Lions |
| 73 | 28 | Trae Waynes | 68.81 | 61.54 | 71.16 | 769 | Vikings |
| 74 | 29 | Morris Claiborne | 68.68 | 61.71 | 73.84 | 198 | Chiefs |
| 75 | 30 | Darryl Roberts | 68.57 | 58.61 | 73.74 | 713 | Jets |
| 76 | 31 | Kevin King | 68.56 | 62.27 | 73.69 | 805 | Packers |
| 77 | 32 | Tye Smith | 68.56 | 65.65 | 73.74 | 210 | Titans |
| 78 | 33 | Anthony Brown | 68.35 | 60.99 | 73.05 | 282 | Cowboys |
| 79 | 34 | Terrance Mitchell | 68.05 | 60.69 | 75.88 | 329 | Browns |
| 80 | 35 | Eric Rowe | 67.86 | 60.20 | 73.58 | 1072 | Dolphins |
| 81 | 36 | Eli Apple | 67.78 | 59.10 | 71.48 | 933 | Saints |
| 82 | 37 | Jimmy Moreland | 67.59 | 60.98 | 69.92 | 471 | Commanders |
| 83 | 38 | Nik Needham | 67.51 | 60.00 | 72.51 | 743 | Dolphins |
| 84 | 39 | William Jackson III | 67.47 | 55.20 | 72.95 | 831 | Bengals |
| 85 | 40 | Damontae Kazee | 67.45 | 58.13 | 69.50 | 803 | Falcons |
| 86 | 41 | Javien Elliott | 67.36 | 64.23 | 70.69 | 439 | Panthers |
| 87 | 42 | Blessuan Austin | 67.36 | 66.79 | 74.44 | 388 | Jets |
| 88 | 43 | A.J. Bouye | 67.24 | 55.40 | 72.95 | 931 | Jaguars |
| 89 | 44 | Nevin Lawson | 67.14 | 60.90 | 70.78 | 300 | Raiders |
| 90 | 45 | Buster Skrine | 66.99 | 59.16 | 68.88 | 727 | Bears |
| 91 | 46 | Holton Hill | 66.89 | 60.94 | 73.32 | 151 | Vikings |
| 92 | 47 | Mike Hughes | 66.88 | 59.69 | 72.70 | 500 | Vikings |
| 93 | 48 | Blidi Wreh-Wilson | 66.81 | 62.55 | 73.72 | 336 | Falcons |
| 94 | 49 | Dre Kirkpatrick | 66.80 | 60.76 | 73.22 | 334 | Bengals |
| 95 | 50 | Ahkello Witherspoon | 66.75 | 63.13 | 69.90 | 562 | 49ers |
| 96 | 51 | Pierre Desir | 66.73 | 58.01 | 72.45 | 683 | Colts |
| 97 | 52 | Isaiah Oliver | 66.45 | 54.50 | 72.20 | 927 | Falcons |
| 98 | 53 | Avonte Maddox | 66.33 | 57.94 | 72.71 | 518 | Eagles |
| 99 | 54 | De'Vante Bausby | 66.09 | 64.94 | 82.69 | 133 | Broncos |
| 100 | 55 | Xavien Howard | 65.82 | 57.68 | 74.06 | 322 | Dolphins |
| 101 | 56 | Kendall Fuller | 65.73 | 55.93 | 71.01 | 498 | Chiefs |
| 102 | 57 | Taron Johnson | 65.71 | 60.77 | 69.38 | 495 | Bills |
| 103 | 58 | Donte Jackson | 65.23 | 56.53 | 69.46 | 726 | Panthers |
| 104 | 59 | Michael Davis | 65.20 | 57.59 | 71.42 | 659 | Chargers |
| 105 | 60 | B.W. Webb | 64.59 | 54.50 | 68.09 | 834 | Bengals |
| 106 | 61 | Antonio Hamilton Sr. | 64.57 | 59.80 | 77.00 | 132 | Giants |
| 107 | 62 | David Long Jr. | 64.07 | 62.68 | 76.80 | 109 | Rams |
| 108 | 63 | Jalen Mills | 63.78 | 56.88 | 71.89 | 501 | Eagles |
| 109 | 64 | Fabian Moreau | 63.63 | 56.77 | 68.21 | 664 | Commanders |
| 110 | 65 | Tre Herndon | 63.58 | 54.10 | 71.06 | 902 | Jaguars |
| 111 | 66 | Arthur Maulet | 63.34 | 63.06 | 69.12 | 349 | Jets |
| 112 | 67 | Rasul Douglas | 62.69 | 51.76 | 67.26 | 583 | Eagles |
| 113 | 68 | Trumaine Johnson | 62.66 | 52.53 | 71.82 | 314 | Jets |
| 114 | 69 | Tre Flowers | 62.45 | 53.10 | 65.55 | 978 | Seahawks |

### Rotation/backup (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 115 | 1 | Xavier Rhodes | 61.79 | 44.87 | 70.05 | 795 | Vikings |
| 116 | 2 | Rashaan Melvin | 61.30 | 47.20 | 69.97 | 870 | Lions |
| 117 | 3 | Coty Sensabaugh | 61.18 | 56.65 | 68.17 | 152 | Commanders |
| 118 | 4 | T.J. Carrie | 60.99 | 48.31 | 65.79 | 676 | Browns |
| 119 | 5 | Greedy Williams | 60.80 | 54.19 | 65.20 | 680 | Browns |
| 120 | 6 | Davontae Harris | 60.61 | 56.11 | 63.99 | 429 | Broncos |
| 121 | 7 | Byron Murphy Jr. | 60.43 | 48.50 | 64.21 | 1105 | Cardinals |
| 122 | 8 | Josh Norman | 60.04 | 44.84 | 69.02 | 603 | Commanders |
| 123 | 9 | Isaac Yiadom | 60.01 | 53.98 | 63.77 | 504 | Broncos |
| 124 | 10 | Kendall Sheffield | 59.86 | 49.80 | 64.48 | 697 | Falcons |
| 125 | 11 | P.J. Williams | 59.86 | 47.99 | 64.96 | 799 | Saints |
| 126 | 12 | Brandon Facyson | 59.30 | 55.84 | 64.74 | 328 | Chargers |
| 127 | 13 | Bashaud Breeland | 59.15 | 43.90 | 68.16 | 912 | Chiefs |
| 128 | 14 | Jamar Taylor | 58.92 | 48.91 | 65.50 | 215 | Falcons |
| 129 | 15 | Duke Dawson | 58.25 | 57.87 | 58.23 | 343 | Broncos |
| 130 | 16 | Sam Beal | 58.10 | 57.87 | 63.61 | 289 | Giants |
| 131 | 17 | Kevin Toliver II | 57.87 | 53.97 | 64.50 | 175 | Bears |
| 132 | 18 | Ken Webster | 57.79 | 53.35 | 64.92 | 226 | Dolphins |
| 133 | 19 | M.J. Stewart | 57.51 | 61.31 | 59.14 | 628 | Buccaneers |
| 134 | 20 | LeShaun Sims | 57.43 | 51.27 | 62.05 | 335 | Titans |
| 135 | 21 | Jamal Perry | 57.11 | 44.05 | 63.74 | 600 | Dolphins |
| 136 | 22 | Quincy Wilson | 56.27 | 49.04 | 63.90 | 122 | Colts |
| 137 | 23 | Ronald Darby | 56.17 | 42.43 | 66.99 | 506 | Eagles |
| 138 | 24 | DeAndre Baker | 55.82 | 45.60 | 58.47 | 970 | Giants |
| 139 | 25 | Ryan Lewis | 55.78 | 43.79 | 69.11 | 293 | Dolphins |
| 140 | 26 | Anthony Averett | 55.03 | 50.27 | 62.10 | 221 | Ravens |
| 141 | 27 | Phillip Gaines | 54.69 | 47.63 | 64.91 | 133 | Texans |
| 142 | 28 | Grant Haley | 53.62 | 53.35 | 55.88 | 422 | Giants |
| 143 | 29 | Nate Hairston | 53.58 | 54.00 | 55.58 | 392 | Jets |
| 144 | 30 | Keion Crossen | 53.55 | 53.63 | 57.80 | 134 | Texans |
| 145 | 31 | Vernon Hargreaves III | 53.01 | 40.90 | 63.59 | 844 | Texans |
| 146 | 32 | Tae Hayes | 51.29 | 53.84 | 66.93 | 107 | Dolphins |
| 147 | 33 | Aaron Colvin | 51.19 | 44.41 | 59.77 | 223 | Commanders |
| 148 | 34 | Tony McRae | 50.83 | 53.31 | 54.60 | 197 | Bengals |
| 149 | 35 | Lonnie Johnson Jr. | 47.79 | 40.00 | 52.99 | 531 | Texans |
| 150 | 36 | Corey Ballentine | 46.79 | 41.85 | 54.25 | 298 | Giants |

## DI — Defensive Interior

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 93.56 | 89.25 | 92.46 | 926 | Rams |
| 2 | 2 | Grady Jarrett | 87.33 | 87.63 | 83.59 | 805 | Falcons |
| 3 | 3 | Geno Atkins | 85.95 | 82.62 | 84.00 | 817 | Bengals |
| 4 | 4 | Kenny Clark | 85.79 | 87.39 | 81.71 | 869 | Packers |
| 5 | 5 | DeForest Buckner | 85.66 | 88.94 | 79.30 | 811 | 49ers |
| 6 | 6 | Cameron Heyward | 84.79 | 84.99 | 80.49 | 873 | Steelers |
| 7 | 7 | Fletcher Cox | 84.50 | 87.23 | 78.51 | 799 | Eagles |
| 8 | 8 | Calais Campbell | 84.21 | 76.05 | 85.49 | 818 | Jaguars |
| 9 | 9 | Jurrell Casey | 84.16 | 83.70 | 81.65 | 707 | Titans |
| 10 | 10 | Leonard Williams | 83.24 | 85.38 | 78.16 | 732 | Giants |
| 11 | 11 | Shelby Harris | 83.00 | 80.19 | 80.70 | 636 | Broncos |
| 12 | 12 | Javon Hargrave | 82.61 | 82.88 | 78.27 | 680 | Steelers |
| 13 | 13 | Sheldon Richardson | 82.01 | 79.76 | 79.54 | 774 | Browns |
| 14 | 14 | Chris Jones | 81.41 | 84.17 | 76.97 | 646 | Chiefs |

### Good (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Ndamukong Suh | 79.92 | 75.07 | 78.98 | 874 | Buccaneers |
| 16 | 2 | Vita Vea | 79.55 | 85.65 | 72.48 | 749 | Buccaneers |
| 17 | 3 | DJ Reader | 79.40 | 81.03 | 75.08 | 622 | Texans |
| 18 | 4 | DeMarcus Walker | 79.35 | 65.19 | 93.61 | 220 | Broncos |
| 19 | 5 | Dalvin Tomlinson | 79.20 | 78.70 | 75.36 | 595 | Giants |
| 20 | 6 | Eddie Goldman | 79.00 | 77.41 | 76.63 | 467 | Bears |
| 21 | 7 | Gerald McCoy | 78.90 | 79.53 | 75.14 | 696 | Panthers |
| 22 | 8 | Taven Bryan | 78.56 | 78.50 | 74.43 | 481 | Jaguars |
| 23 | 9 | Matt Ioannidis | 78.27 | 75.58 | 76.93 | 827 | Commanders |
| 24 | 10 | Steve McLendon | 78.17 | 69.86 | 79.54 | 465 | Jets |
| 25 | 11 | Dexter Lawrence | 78.14 | 81.89 | 71.48 | 701 | Giants |
| 26 | 12 | Damon Harrison Sr. | 77.95 | 66.63 | 81.84 | 527 | Lions |
| 27 | 13 | Linval Joseph | 77.89 | 74.98 | 77.54 | 553 | Vikings |
| 28 | 14 | Michael Pierce | 77.66 | 71.50 | 79.27 | 481 | Ravens |
| 29 | 15 | Akiem Hicks | 77.43 | 73.41 | 81.67 | 191 | Bears |
| 30 | 16 | Michael Brockers | 77.34 | 75.79 | 74.20 | 766 | Rams |
| 31 | 17 | Danny Shelton | 77.20 | 78.10 | 73.78 | 492 | Patriots |
| 32 | 18 | Ed Oliver | 77.19 | 68.89 | 78.56 | 557 | Bills |
| 33 | 19 | B.J. Hill | 77.04 | 71.81 | 76.36 | 486 | Giants |
| 34 | 20 | Jonathan Allen | 76.77 | 69.31 | 80.39 | 722 | Commanders |
| 35 | 21 | Poona Ford | 76.72 | 71.45 | 78.66 | 506 | Seahawks |
| 36 | 22 | Lawrence Guy Sr. | 76.50 | 67.03 | 78.65 | 524 | Patriots |
| 37 | 23 | Stephon Tuitt | 76.14 | 79.39 | 76.25 | 278 | Steelers |
| 38 | 24 | Derek Wolfe | 75.47 | 72.21 | 76.61 | 523 | Broncos |
| 39 | 25 | Mike Daniels | 75.47 | 67.91 | 82.28 | 203 | Lions |
| 40 | 26 | Daron Payne | 75.12 | 73.81 | 72.48 | 758 | Commanders |
| 41 | 27 | Shy Tuttle | 74.57 | 67.45 | 75.15 | 340 | Saints |
| 42 | 28 | Quinnen Williams | 74.12 | 72.50 | 74.17 | 512 | Jets |
| 43 | 29 | Jeffery Simmons | 74.01 | 70.93 | 79.19 | 315 | Titans |

### Starter (82 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 44 | 1 | Malcom Brown | 73.59 | 64.20 | 76.00 | 487 | Saints |
| 45 | 2 | Denico Autry | 73.53 | 65.88 | 76.74 | 620 | Colts |
| 46 | 3 | Larry Ogunjobi | 73.40 | 59.63 | 79.34 | 779 | Browns |
| 47 | 4 | Maurice Hurst | 73.17 | 71.73 | 71.13 | 522 | Raiders |
| 48 | 5 | Christian Wilkins | 73.03 | 67.33 | 72.67 | 730 | Dolphins |
| 49 | 6 | Brandon Williams | 72.92 | 65.78 | 75.40 | 525 | Ravens |
| 50 | 7 | Johnathan Hankins | 72.91 | 63.44 | 75.58 | 670 | Raiders |
| 51 | 8 | Henry Anderson | 72.91 | 65.57 | 76.66 | 446 | Jets |
| 52 | 9 | Tim Settle | 72.38 | 63.05 | 75.86 | 314 | Commanders |
| 53 | 10 | Dean Lowry | 72.30 | 62.35 | 74.76 | 637 | Packers |
| 54 | 11 | Christian Covington | 72.28 | 67.33 | 74.55 | 481 | Cowboys |
| 55 | 12 | Roy Robertson-Harris | 72.21 | 64.11 | 74.79 | 544 | Bears |
| 56 | 13 | Rodney Gunter | 72.15 | 63.20 | 75.52 | 602 | Cardinals |
| 57 | 14 | Folorunso Fatukasi | 71.94 | 71.50 | 75.23 | 390 | Jets |
| 58 | 15 | Mike Pennel | 71.87 | 64.39 | 76.86 | 154 | Chiefs |
| 59 | 16 | Marcell Dareus | 71.70 | 62.68 | 79.07 | 206 | Jaguars |
| 60 | 17 | Armon Watts | 71.47 | 65.88 | 81.90 | 121 | Vikings |
| 61 | 18 | Olsen Pierre | 71.42 | 54.53 | 84.44 | 172 | Raiders |
| 62 | 19 | DaQuan Jones | 71.39 | 70.06 | 68.94 | 679 | Titans |
| 63 | 20 | Andrew Billings | 71.35 | 66.27 | 70.77 | 657 | Bengals |
| 64 | 21 | Sheldon Rankins | 71.31 | 66.77 | 73.31 | 323 | Saints |
| 65 | 22 | Zach Kerr | 71.28 | 62.24 | 76.27 | 328 | Cardinals |
| 66 | 23 | Tyler Lancaster | 71.26 | 62.16 | 75.11 | 381 | Packers |
| 67 | 24 | Mike Purcell | 71.03 | 61.62 | 76.99 | 416 | Broncos |
| 68 | 25 | Davon Godchaux | 70.96 | 60.81 | 73.76 | 718 | Dolphins |
| 69 | 26 | Sebastian Joseph-Day | 70.95 | 54.13 | 77.99 | 481 | Rams |
| 70 | 27 | Abry Jones | 70.34 | 57.43 | 75.09 | 558 | Jaguars |
| 71 | 28 | Bilal Nichols | 70.27 | 57.13 | 77.59 | 445 | Bears |
| 72 | 29 | Vernon Butler | 69.84 | 60.93 | 73.50 | 440 | Panthers |
| 73 | 30 | Adam Butler | 69.67 | 56.01 | 74.61 | 474 | Patriots |
| 74 | 31 | Jarran Reed | 69.55 | 58.71 | 75.94 | 479 | Seahawks |
| 75 | 32 | A'Shawn Robinson | 69.51 | 60.15 | 74.09 | 526 | Lions |
| 76 | 33 | Dre'Mont Jones | 69.49 | 62.72 | 71.92 | 283 | Broncos |
| 77 | 34 | Timmy Jernigan | 69.44 | 62.14 | 77.32 | 274 | Eagles |
| 78 | 35 | Quinton Jefferson | 69.36 | 65.04 | 71.21 | 589 | Seahawks |
| 79 | 36 | Corey Liuget | 69.26 | 63.02 | 76.33 | 180 | Bills |
| 80 | 37 | Tyson Alualu | 68.97 | 62.63 | 69.35 | 432 | Steelers |
| 81 | 38 | P.J. Hall | 68.85 | 66.41 | 67.10 | 551 | Raiders |
| 82 | 39 | Zach Sieler | 68.81 | 64.43 | 78.90 | 118 | Dolphins |
| 83 | 40 | David Onyemata | 68.56 | 60.62 | 70.21 | 565 | Saints |
| 84 | 41 | Dan McCullers | 68.46 | 58.31 | 74.19 | 131 | Steelers |
| 85 | 42 | Josh Tupou | 68.26 | 66.43 | 69.21 | 465 | Bengals |
| 86 | 43 | William Gholston | 68.15 | 54.53 | 73.07 | 493 | Buccaneers |
| 87 | 44 | Jordan Phillips | 68.13 | 56.28 | 72.49 | 542 | Bills |
| 88 | 45 | D.J. Jones | 67.99 | 60.07 | 75.03 | 304 | 49ers |
| 89 | 46 | Dontari Poe | 67.97 | 66.41 | 67.45 | 402 | Panthers |
| 90 | 47 | Adam Gotsis | 67.35 | 58.16 | 72.96 | 272 | Broncos |
| 91 | 48 | Beau Allen | 67.31 | 58.91 | 70.93 | 179 | Buccaneers |
| 92 | 49 | Jonathan Bullard | 67.14 | 57.07 | 73.33 | 309 | Cardinals |
| 93 | 50 | Carlos Watkins | 67.11 | 58.33 | 76.50 | 265 | Texans |
| 94 | 51 | Allen Bailey | 66.85 | 55.65 | 70.89 | 511 | Falcons |
| 95 | 52 | Derrick Nnadi | 66.63 | 60.27 | 66.71 | 598 | Chiefs |
| 96 | 53 | Corey Peters | 66.55 | 55.44 | 70.94 | 805 | Cardinals |
| 97 | 54 | Xavier Williams | 66.32 | 56.60 | 75.40 | 118 | Chiefs |
| 98 | 55 | Nathan Shepherd | 66.26 | 66.95 | 66.19 | 232 | Jets |
| 99 | 56 | Greg Gaines | 66.07 | 66.07 | 68.15 | 183 | Rams |
| 100 | 57 | Sheldon Day | 66.07 | 56.16 | 70.60 | 325 | 49ers |
| 101 | 58 | Clinton McDonald | 66.05 | 52.62 | 75.20 | 122 | Cardinals |
| 102 | 59 | Star Lotulelei | 65.87 | 53.58 | 69.90 | 482 | Bills |
| 103 | 60 | Treyvon Hester | 65.78 | 61.25 | 67.34 | 132 | Commanders |
| 104 | 61 | Al Woods | 65.67 | 53.86 | 71.04 | 450 | Seahawks |
| 105 | 62 | Austin Johnson | 65.63 | 58.60 | 66.15 | 325 | Titans |
| 106 | 63 | RJ McIntosh | 65.63 | 60.96 | 71.74 | 114 | Giants |
| 107 | 64 | Isaiah Mack | 65.50 | 60.16 | 68.03 | 172 | Titans |
| 108 | 65 | Anthony Rush | 65.32 | 57.63 | 73.58 | 150 | Eagles |
| 109 | 66 | Charles Omenihu | 65.32 | 60.35 | 66.55 | 443 | Texans |
| 110 | 67 | Hassan Ridgeway | 65.29 | 58.51 | 74.40 | 247 | Eagles |
| 111 | 68 | Brent Urban | 65.21 | 56.22 | 71.31 | 245 | Bears |
| 112 | 69 | Da'Shawn Hand | 65.18 | 59.26 | 74.59 | 110 | Lions |
| 113 | 70 | Tyeler Davison | 64.95 | 61.06 | 64.01 | 560 | Falcons |
| 114 | 71 | Grover Stewart | 64.61 | 56.49 | 66.37 | 627 | Colts |
| 115 | 72 | Jullian Taylor | 64.52 | 58.05 | 78.08 | 101 | 49ers |
| 116 | 73 | Maliek Collins | 64.49 | 57.40 | 65.99 | 763 | Cowboys |
| 117 | 74 | John Jenkins | 64.45 | 56.08 | 70.35 | 480 | Dolphins |
| 118 | 75 | Akeem Spence | 64.29 | 51.44 | 69.21 | 369 | Jaguars |
| 119 | 76 | Chris Wormley | 64.27 | 59.36 | 65.26 | 446 | Ravens |
| 120 | 77 | Margus Hunt | 63.94 | 48.49 | 70.39 | 451 | Colts |
| 121 | 78 | Brandon Dunn | 63.88 | 55.05 | 66.24 | 399 | Texans |
| 122 | 79 | Domata Peko Sr. | 63.76 | 54.80 | 70.67 | 140 | Ravens |
| 123 | 80 | Rakeem Nunez-Roches | 63.44 | 54.81 | 69.10 | 293 | Buccaneers |
| 124 | 81 | Jack Crawford | 62.80 | 48.65 | 70.57 | 431 | Falcons |
| 125 | 82 | Justin Jones | 62.02 | 53.16 | 66.76 | 504 | Chargers |

### Rotation/backup (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 126 | 1 | Morgan Fox | 61.97 | 50.84 | 65.22 | 353 | Rams |
| 127 | 2 | Jaleel Johnson | 61.77 | 50.70 | 66.86 | 408 | Vikings |
| 128 | 3 | Brandon Mebane | 61.68 | 44.75 | 71.61 | 408 | Chargers |
| 129 | 4 | Khalen Saunders | 61.39 | 52.77 | 67.14 | 303 | Chiefs |
| 130 | 5 | Antwaun Woods | 61.34 | 55.09 | 67.91 | 310 | Cowboys |
| 131 | 6 | Montravius Adams | 61.16 | 53.97 | 64.70 | 187 | Packers |
| 132 | 7 | Damion Square | 61.13 | 50.90 | 63.78 | 402 | Chargers |
| 133 | 8 | Kyle Love | 60.82 | 46.55 | 66.69 | 412 | Panthers |
| 134 | 9 | Jerry Tillery | 60.69 | 51.60 | 63.62 | 354 | Chargers |
| 135 | 10 | Caraun Reid | 60.39 | 55.37 | 71.33 | 136 | Cardinals |
| 136 | 11 | Ryan Glasgow | 60.28 | 58.94 | 66.81 | 118 | Bengals |
| 137 | 12 | Angelo Blackson | 60.16 | 50.95 | 64.11 | 427 | Texans |
| 138 | 13 | Eli Ankou | 59.50 | 54.73 | 68.00 | 178 | Browns |
| 139 | 14 | Abdullah Anderson | 58.97 | 57.19 | 69.40 | 106 | Bears |
| 140 | 15 | Jihad Ward | 58.79 | 57.18 | 62.14 | 398 | Ravens |
| 141 | 16 | Miles Brown | 58.21 | 53.78 | 64.29 | 123 | Cardinals |
| 142 | 17 | Renell Wren | 58.09 | 56.54 | 60.16 | 154 | Bengals |
| 143 | 18 | Shamar Stephen | 57.95 | 50.19 | 59.79 | 580 | Vikings |
| 144 | 19 | Trysten Hill | 56.47 | 55.67 | 63.71 | 121 | Cowboys |
| 145 | 20 | Daniel Ekuale | 54.01 | 53.91 | 60.78 | 114 | Browns |
| 146 | 21 | Tanzel Smart | 53.80 | 52.90 | 56.17 | 171 | Rams |
| 147 | 22 | Zach Allen | 52.94 | 55.40 | 65.63 | 144 | Cardinals |
| 148 | 23 | Kevin Strong | 52.25 | 54.64 | 54.82 | 172 | Lions |
| 149 | 24 | John Atkins | 51.29 | 51.99 | 54.73 | 409 | Lions |
| 150 | 25 | Joey Ivie | 48.48 | 51.99 | 50.31 | 115 | Titans |
| 151 | 26 | Wes Horton | 45.53 | 49.64 | 52.04 | 117 | Panthers |
| 152 | 27 | Frank Herron | 45.00 | 53.54 | 53.92 | 103 | Lions |

## ED — Edge

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Von Miller | 92.17 | 91.79 | 88.77 | 833 | Broncos |
| 2 | 2 | Joey Bosa | 91.74 | 96.47 | 87.24 | 836 | Chargers |
| 3 | 3 | T.J. Watt | 91.70 | 93.77 | 86.15 | 935 | Steelers |
| 4 | 4 | Nick Bosa | 91.47 | 95.82 | 84.41 | 777 | 49ers |
| 5 | 5 | Danielle Hunter | 90.12 | 92.95 | 84.06 | 883 | Vikings |
| 6 | 6 | Khalil Mack | 89.96 | 93.91 | 83.80 | 925 | Bears |
| 7 | 7 | Justin Houston | 87.39 | 82.93 | 87.44 | 674 | Colts |
| 8 | 8 | DeMarcus Lawrence | 85.77 | 90.17 | 78.67 | 668 | Cowboys |
| 9 | 9 | Myles Garrett | 84.84 | 89.42 | 81.79 | 544 | Browns |
| 10 | 10 | Shaquil Barrett | 83.53 | 81.04 | 81.95 | 889 | Buccaneers |
| 11 | 11 | Cameron Jordan | 83.28 | 89.37 | 75.06 | 877 | Saints |
| 12 | 12 | Brandon Graham | 82.72 | 81.31 | 79.49 | 775 | Eagles |
| 13 | 13 | Za'Darius Smith | 82.35 | 85.36 | 76.59 | 872 | Packers |
| 14 | 14 | Chandler Jones | 82.22 | 81.47 | 78.55 | 1069 | Cardinals |
| 15 | 15 | Carlos Dunlap | 80.06 | 80.24 | 76.81 | 739 | Bengals |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Marcus Davenport | 79.38 | 84.03 | 75.25 | 533 | Saints |
| 17 | 2 | Dante Fowler Jr. | 78.51 | 79.64 | 73.90 | 880 | Rams |
| 18 | 3 | Jadeveon Clowney | 78.05 | 89.59 | 68.07 | 605 | Seahawks |
| 19 | 4 | Trey Flowers | 78.00 | 73.78 | 77.48 | 705 | Lions |
| 20 | 5 | J.J. Watt | 77.69 | 68.43 | 83.86 | 469 | Texans |
| 21 | 6 | Michael Bennett | 76.93 | 73.38 | 75.64 | 559 | Cowboys |
| 22 | 7 | Frank Clark | 76.68 | 68.03 | 79.32 | 725 | Chiefs |
| 23 | 8 | Matthew Judon | 76.39 | 63.86 | 80.57 | 791 | Ravens |
| 24 | 9 | Cameron Wake | 76.29 | 61.75 | 86.09 | 195 | Titans |
| 25 | 10 | Melvin Ingram III | 76.28 | 69.54 | 78.18 | 668 | Chargers |
| 26 | 11 | Dee Ford | 75.66 | 71.00 | 79.29 | 226 | 49ers |
| 27 | 12 | Ifeadi Odenigbo | 75.39 | 73.63 | 72.40 | 736 | Vikings |
| 28 | 13 | Ryan Kerrigan | 75.33 | 60.84 | 82.90 | 642 | Commanders |
| 29 | 14 | Ezekiel Ansah | 75.00 | 67.62 | 81.59 | 338 | Seahawks |
| 30 | 15 | Jerry Hughes | 74.89 | 70.54 | 73.63 | 663 | Bills |
| 31 | 16 | Robert Quinn | 74.84 | 65.50 | 77.93 | 647 | Cowboys |
| 32 | 17 | Olivier Vernon | 74.51 | 76.96 | 74.22 | 508 | Browns |
| 33 | 18 | Jacob Martin | 74.15 | 63.56 | 78.34 | 220 | Texans |

### Starter (59 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Sam Hubbard | 73.85 | 67.36 | 74.66 | 852 | Bengals |
| 35 | 2 | Yannick Ngakoue | 73.85 | 65.31 | 75.90 | 791 | Jaguars |
| 36 | 3 | Arik Armstead | 73.37 | 70.66 | 73.09 | 776 | 49ers |
| 37 | 4 | Chase Winovich | 73.19 | 63.71 | 75.34 | 291 | Patriots |
| 38 | 5 | Shaq Lawson | 73.06 | 71.15 | 72.35 | 483 | Bills |
| 39 | 6 | Pernell McPhee | 72.89 | 64.59 | 80.50 | 260 | Ravens |
| 40 | 7 | Deatrich Wise Jr. | 72.81 | 71.22 | 70.73 | 229 | Patriots |
| 41 | 8 | Brian Burns | 72.73 | 62.37 | 75.47 | 478 | Panthers |
| 42 | 9 | Takk McKinley | 72.55 | 65.15 | 74.66 | 546 | Falcons |
| 43 | 10 | Preston Smith | 72.48 | 64.06 | 73.93 | 870 | Packers |
| 44 | 11 | Bradley Chubb | 72.42 | 64.26 | 81.51 | 233 | Broncos |
| 45 | 12 | Everson Griffen | 72.17 | 64.76 | 75.02 | 849 | Vikings |
| 46 | 13 | Tyus Bowser | 72.05 | 61.45 | 75.27 | 389 | Ravens |
| 47 | 14 | Montez Sweat | 71.91 | 61.95 | 74.39 | 724 | Commanders |
| 48 | 15 | Mario Addison | 71.75 | 56.12 | 78.52 | 729 | Panthers |
| 49 | 16 | Jabaal Sheard | 71.73 | 69.20 | 70.81 | 569 | Colts |
| 50 | 17 | Maxx Crosby | 71.71 | 62.84 | 73.45 | 750 | Raiders |
| 51 | 18 | Markus Golden | 71.63 | 58.73 | 80.13 | 916 | Giants |
| 52 | 19 | Whitney Mercilus | 71.40 | 62.76 | 75.27 | 950 | Texans |
| 53 | 20 | Rashan Gary | 71.27 | 62.66 | 72.85 | 244 | Packers |
| 54 | 21 | Lorenzo Alexander | 71.14 | 43.85 | 85.17 | 494 | Bills |
| 55 | 22 | Derek Barnett | 71.13 | 72.12 | 70.47 | 694 | Eagles |
| 56 | 23 | Carl Lawson | 70.55 | 62.27 | 76.80 | 457 | Bengals |
| 57 | 24 | Vinny Curry | 70.01 | 61.04 | 73.08 | 393 | Eagles |
| 58 | 25 | Aaron Lynch | 69.87 | 59.56 | 75.40 | 244 | Bears |
| 59 | 26 | Kyler Fackrell | 69.38 | 55.61 | 74.40 | 415 | Packers |
| 60 | 27 | Jason Pierre-Paul | 68.81 | 63.73 | 71.16 | 586 | Buccaneers |
| 61 | 28 | Oshane Ximines | 68.56 | 59.84 | 70.21 | 502 | Giants |
| 62 | 29 | Bud Dupree | 68.46 | 63.01 | 67.92 | 980 | Steelers |
| 63 | 30 | Jordan Jenkins | 68.46 | 60.03 | 70.94 | 572 | Jets |
| 64 | 31 | John Cominsky | 68.20 | 65.86 | 72.90 | 100 | Falcons |
| 65 | 32 | Terrell Suggs | 67.82 | 47.65 | 77.61 | 690 | Chiefs |
| 66 | 33 | Leonard Floyd | 67.78 | 61.90 | 68.78 | 899 | Bears |
| 67 | 34 | Samson Ebukam | 67.17 | 62.22 | 66.31 | 565 | Rams |
| 68 | 35 | Vic Beasley Jr. | 67.13 | 58.28 | 68.86 | 757 | Falcons |
| 69 | 36 | Josh Sweat | 67.04 | 64.91 | 67.43 | 352 | Eagles |
| 70 | 37 | Clay Matthews | 66.35 | 46.30 | 77.53 | 614 | Rams |
| 71 | 38 | Kamalei Correa | 66.18 | 57.13 | 68.05 | 432 | Titans |
| 72 | 39 | Bruce Irvin | 66.10 | 47.13 | 76.14 | 608 | Panthers |
| 73 | 40 | Adrian Clayborn | 65.81 | 59.32 | 67.12 | 439 | Falcons |
| 74 | 41 | John Simon | 65.73 | 57.26 | 70.23 | 481 | Patriots |
| 75 | 42 | Lorenzo Carter | 65.68 | 62.12 | 64.92 | 723 | Giants |
| 76 | 43 | Jeremiah Attaochu | 65.38 | 60.06 | 71.94 | 322 | Broncos |
| 77 | 44 | Ronald Blair III | 65.35 | 61.61 | 69.41 | 199 | 49ers |
| 78 | 45 | Ogbo Okoronkwo | 65.12 | 61.22 | 70.85 | 115 | Rams |
| 79 | 46 | Ryan Anderson | 64.87 | 60.64 | 64.88 | 559 | Commanders |
| 80 | 47 | Trent Murphy | 64.76 | 55.71 | 67.56 | 674 | Bills |
| 81 | 48 | Ben Banogu | 63.98 | 58.36 | 64.59 | 272 | Colts |
| 82 | 49 | Trey Hendrickson | 63.95 | 64.01 | 65.16 | 404 | Saints |
| 83 | 50 | Clelin Ferrell | 63.92 | 64.06 | 60.69 | 648 | Raiders |
| 84 | 51 | Cassius Marsh | 63.80 | 57.95 | 63.74 | 429 | Cardinals |
| 85 | 52 | Jaylon Ferguson | 63.44 | 59.51 | 63.97 | 498 | Ravens |
| 86 | 53 | Kerry Hyder Jr. | 63.41 | 56.57 | 66.62 | 439 | Cowboys |
| 87 | 54 | Tarell Basham | 63.39 | 60.63 | 63.77 | 590 | Jets |
| 88 | 55 | Benson Mayowa | 63.26 | 59.06 | 63.15 | 302 | Raiders |
| 89 | 56 | Alex Okafor | 62.99 | 53.88 | 69.26 | 421 | Chiefs |
| 90 | 57 | Trent Harris | 62.56 | 57.87 | 66.72 | 253 | Dolphins |
| 91 | 58 | Mario Edwards Jr. | 62.51 | 59.20 | 62.63 | 293 | Saints |
| 92 | 59 | Kyle Phillips | 62.35 | 61.89 | 59.52 | 549 | Jets |

### Rotation/backup (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 93 | 1 | Devon Kennard | 61.70 | 48.32 | 66.97 | 935 | Lions |
| 94 | 2 | Carl Nassib | 61.67 | 61.09 | 59.24 | 630 | Buccaneers |
| 95 | 3 | Stephen Weatherly | 61.67 | 58.41 | 61.15 | 422 | Vikings |
| 96 | 4 | Malik Reed | 61.49 | 59.78 | 59.49 | 468 | Broncos |
| 97 | 5 | Emmanuel Ogbah | 61.22 | 62.09 | 61.47 | 410 | Chiefs |
| 98 | 6 | Brennan Scarlett | 60.74 | 55.23 | 61.81 | 491 | Texans |
| 99 | 7 | Anthony Chickillo | 60.52 | 57.10 | 61.23 | 146 | Steelers |
| 100 | 8 | Vince Biegel | 60.29 | 57.37 | 64.32 | 627 | Dolphins |
| 101 | 9 | Andrew Brown | 59.70 | 59.63 | 57.67 | 241 | Bengals |
| 102 | 10 | Chad Thomas | 59.54 | 53.31 | 59.53 | 464 | Browns |
| 103 | 11 | Solomon Thomas | 59.31 | 60.78 | 54.58 | 425 | 49ers |
| 104 | 12 | Darryl Johnson | 59.31 | 57.87 | 57.14 | 224 | Bills |
| 105 | 13 | Taco Charlton | 59.11 | 58.39 | 60.10 | 396 | Dolphins |
| 106 | 14 | Charles Harris | 58.81 | 58.78 | 57.27 | 429 | Dolphins |
| 107 | 15 | Dawuane Smoot | 58.77 | 59.08 | 56.90 | 404 | Jaguars |
| 108 | 16 | Jordan Willis | 58.40 | 61.43 | 55.86 | 162 | Jets |
| 109 | 17 | Romeo Okwara | 58.28 | 59.92 | 56.45 | 605 | Lions |
| 110 | 18 | Efe Obada | 58.19 | 57.23 | 57.01 | 306 | Panthers |
| 111 | 19 | Dorance Armstrong | 58.14 | 57.80 | 55.23 | 262 | Cowboys |
| 112 | 20 | Justin Hollins | 58.06 | 58.47 | 55.70 | 266 | Broncos |
| 113 | 21 | Rasheem Green | 57.30 | 57.06 | 55.65 | 545 | Seahawks |
| 114 | 22 | Al-Quadin Muhammad | 57.23 | 58.56 | 55.20 | 483 | Colts |
| 115 | 23 | Tanoh Kpassagnon | 56.86 | 57.97 | 55.39 | 691 | Chiefs |
| 116 | 24 | Arden Key | 56.57 | 59.88 | 56.06 | 179 | Raiders |
| 117 | 25 | Nate Orchard | 56.46 | 58.74 | 59.21 | 118 | Commanders |
| 118 | 26 | Anthony Zettel | 56.18 | 56.47 | 57.87 | 103 | 49ers |
| 119 | 27 | Isaiah Irving | 56.00 | 58.35 | 56.51 | 128 | Bears |
| 120 | 28 | Isaac Rochell | 55.76 | 57.15 | 50.66 | 274 | Chargers |
| 121 | 29 | Branden Jackson | 55.72 | 55.94 | 54.95 | 418 | Seahawks |
| 122 | 30 | Bryan Cox Jr. | 54.68 | 58.13 | 56.13 | 208 | Browns |
| 123 | 31 | Shilique Calhoun | 54.47 | 56.45 | 53.99 | 266 | Patriots |
| 124 | 32 | Marquis Haynes Sr. | 53.32 | 56.78 | 54.79 | 210 | Panthers |
| 125 | 33 | Josh Mauro | 53.10 | 54.24 | 51.31 | 282 | Raiders |
| 126 | 34 | Carl Granderson | 53.08 | 59.70 | 57.92 | 115 | Saints |
| 127 | 35 | Avery Moss | 52.86 | 57.46 | 51.22 | 348 | Dolphins |
| 128 | 36 | Anthony Nelson | 52.44 | 60.79 | 50.00 | 152 | Buccaneers |
| 129 | 37 | L.J. Collier | 50.42 | 57.42 | 46.79 | 152 | Seahawks |
| 130 | 38 | Porter Gustin | 50.31 | 59.18 | 53.65 | 225 | Browns |
| 131 | 39 | Austin Bryant | 45.00 | 58.52 | 48.99 | 133 | Lions |
| 132 | 40 | Demone Harris | 45.00 | 57.72 | 50.63 | 121 | Chiefs |

## G — Guard

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Brandon Brooks | 97.32 | 92.80 | 96.17 | 1046 | Eagles |
| 2 | 2 | Quenton Nelson | 96.00 | 91.20 | 95.03 | 1044 | Colts |
| 3 | 3 | Zack Martin | 92.17 | 88.10 | 90.72 | 1114 | Cowboys |
| 4 | 4 | Marshal Yanda | 91.31 | 86.46 | 90.38 | 968 | Ravens |
| 5 | 5 | Brandon Scherff | 83.90 | 73.63 | 86.58 | 643 | Commanders |
| 6 | 6 | Joe Thuney | 83.67 | 77.40 | 83.69 | 1140 | Patriots |
| 7 | 7 | Kevin Zeitler | 83.66 | 76.40 | 84.34 | 991 | Giants |
| 8 | 8 | Graham Glasgow | 82.14 | 73.27 | 83.89 | 872 | Lions |
| 9 | 9 | Larry Warford | 81.76 | 73.04 | 83.40 | 970 | Saints |
| 10 | 10 | Joel Bitonio | 81.51 | 74.20 | 82.22 | 1039 | Browns |
| 11 | 11 | Nick Gates | 81.03 | 69.24 | 84.72 | 291 | Giants |
| 12 | 12 | Shaq Mason | 80.82 | 73.00 | 81.87 | 1067 | Patriots |
| 13 | 13 | Ali Marpet | 80.27 | 72.30 | 81.42 | 1139 | Buccaneers |

### Good (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Rodger Saffold | 79.86 | 71.20 | 81.46 | 928 | Titans |
| 15 | 2 | Halapoulivaati Vaitai | 79.51 | 68.91 | 82.41 | 477 | Eagles |
| 16 | 3 | David DeCastro | 79.39 | 71.10 | 80.75 | 995 | Steelers |
| 17 | 4 | Isaac Seumalo | 78.08 | 69.60 | 79.56 | 1162 | Eagles |
| 18 | 5 | Andrew Norwell | 75.90 | 65.50 | 78.67 | 1088 | Jaguars |
| 19 | 6 | Denzelle Good | 75.63 | 62.99 | 79.89 | 338 | Raiders |
| 20 | 7 | Cody Whitehair | 75.57 | 64.90 | 78.51 | 1069 | Bears |
| 21 | 8 | Elgton Jenkins | 75.30 | 69.10 | 75.27 | 964 | Packers |
| 22 | 9 | Justin Pugh | 75.04 | 66.80 | 76.37 | 1022 | Cardinals |
| 23 | 10 | Jon Feliciano | 75.02 | 64.02 | 78.18 | 947 | Bills |
| 24 | 11 | Billy Turner | 74.93 | 65.30 | 77.19 | 1076 | Packers |
| 25 | 12 | Laken Tomlinson | 74.59 | 64.70 | 77.01 | 1061 | 49ers |
| 26 | 13 | Pat Elflein | 74.58 | 64.54 | 77.11 | 919 | Vikings |
| 27 | 14 | Bradley Bozeman | 74.53 | 63.40 | 77.79 | 1105 | Ravens |
| 28 | 15 | Dalton Risner | 74.52 | 64.38 | 77.12 | 975 | Broncos |
| 29 | 16 | Greg Van Roten | 74.24 | 64.73 | 76.41 | 704 | Panthers |
| 30 | 17 | Ereck Flowers | 74.14 | 64.00 | 76.74 | 937 | Commanders |

### Starter (44 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Trai Turner | 72.98 | 63.78 | 74.95 | 888 | Panthers |
| 32 | 2 | Alex Cappa | 72.87 | 62.54 | 75.59 | 869 | Buccaneers |
| 33 | 3 | Mark Glowinski | 72.67 | 60.50 | 76.61 | 1076 | Colts |
| 34 | 4 | Connor Williams | 72.47 | 60.77 | 76.11 | 727 | Cowboys |
| 35 | 5 | Josh Kline | 72.23 | 61.29 | 75.36 | 733 | Vikings |
| 36 | 6 | D.J. Fluker | 72.09 | 60.37 | 75.74 | 863 | Seahawks |
| 37 | 7 | Mike Iupati | 72.06 | 60.40 | 75.66 | 1015 | Seahawks |
| 38 | 8 | Chris Lindstrom | 72.01 | 63.70 | 73.38 | 309 | Falcons |
| 39 | 9 | Michael Schofield III | 71.85 | 63.60 | 73.18 | 1057 | Chargers |
| 40 | 10 | Kenny Wiggins | 71.76 | 61.20 | 74.64 | 438 | Lions |
| 41 | 11 | Gabe Jackson | 71.70 | 61.63 | 74.25 | 707 | Raiders |
| 42 | 12 | Mike Person | 71.68 | 61.76 | 74.13 | 937 | 49ers |
| 43 | 13 | Brian Winters | 71.36 | 60.73 | 74.28 | 526 | Jets |
| 44 | 14 | Jamil Douglas | 71.24 | 59.37 | 74.98 | 388 | Titans |
| 45 | 15 | Xavier Su'a-Filo | 71.07 | 60.06 | 74.24 | 307 | Cowboys |
| 46 | 16 | J.R. Sweezy | 71.02 | 61.60 | 73.14 | 1001 | Cardinals |
| 47 | 17 | Jordan Devey | 70.62 | 56.73 | 75.72 | 228 | Raiders |
| 48 | 18 | Ron Leary | 70.60 | 58.52 | 74.48 | 754 | Broncos |
| 49 | 19 | Austin Schlottmann | 70.59 | 59.08 | 74.09 | 260 | Broncos |
| 50 | 20 | Earl Watford | 70.11 | 62.82 | 70.81 | 326 | Buccaneers |
| 51 | 21 | John Miller | 69.90 | 58.75 | 73.17 | 779 | Bengals |
| 52 | 22 | Lane Taylor | 69.84 | 59.68 | 72.44 | 114 | Packers |
| 53 | 23 | Will Hernandez | 68.94 | 58.40 | 71.80 | 1068 | Giants |
| 54 | 24 | Ramon Foster | 68.92 | 59.05 | 71.33 | 822 | Steelers |
| 55 | 25 | Ted Larsen | 68.80 | 57.02 | 72.48 | 168 | Bears |
| 56 | 26 | Laurent Duvernay-Tardif | 68.66 | 57.28 | 72.08 | 899 | Chiefs |
| 57 | 27 | Max Scharping | 68.36 | 56.78 | 71.91 | 938 | Texans |
| 58 | 28 | Wyatt Teller | 68.24 | 57.51 | 71.23 | 559 | Browns |
| 59 | 29 | Alex Lewis | 67.90 | 56.57 | 71.28 | 764 | Jets |
| 60 | 30 | A.J. Cann | 67.63 | 55.62 | 71.47 | 775 | Jaguars |
| 61 | 31 | Quinton Spain | 67.59 | 55.40 | 71.55 | 1063 | Bills |
| 62 | 32 | Danny Isidora | 67.53 | 57.48 | 70.06 | 127 | Dolphins |
| 63 | 33 | Wes Martin | 67.32 | 55.50 | 71.03 | 290 | Commanders |
| 64 | 34 | Alex Redmond | 67.22 | 55.88 | 70.62 | 189 | Bengals |
| 65 | 35 | Zach Fulton | 66.40 | 52.32 | 71.62 | 955 | Texans |
| 66 | 36 | Nick Easton | 65.98 | 52.85 | 70.57 | 409 | Saints |
| 67 | 37 | Andrus Peat | 65.09 | 51.98 | 69.66 | 575 | Saints |
| 68 | 38 | Tom Compton | 64.82 | 53.75 | 68.03 | 363 | Jets |
| 69 | 39 | Dan Feeney | 64.65 | 51.70 | 69.12 | 1032 | Chargers |
| 70 | 40 | Forrest Lamp | 64.04 | 59.72 | 62.75 | 157 | Chargers |
| 71 | 41 | Evan Boehm | 63.53 | 50.21 | 68.25 | 595 | Dolphins |
| 72 | 42 | Jamon Brown | 63.44 | 54.75 | 65.06 | 587 | Falcons |
| 73 | 43 | Deion Calhoun | 62.92 | 49.07 | 67.99 | 471 | Dolphins |
| 74 | 44 | Michael Jordan | 62.20 | 46.78 | 68.31 | 648 | Bengals |

### Rotation/backup (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 75 | 1 | James Carpenter | 61.88 | 46.82 | 67.75 | 675 | Falcons |
| 76 | 2 | Eric Kush | 60.95 | 50.29 | 63.89 | 436 | Browns |
| 77 | 3 | Nate Davis | 60.69 | 43.62 | 67.91 | 724 | Titans |
| 78 | 4 | Kyle Long | 60.14 | 44.55 | 66.37 | 250 | Bears |
| 79 | 5 | Michael Deiter | 60.05 | 42.50 | 67.59 | 996 | Dolphins |
| 80 | 6 | Jamil Demby | 59.46 | 46.56 | 63.90 | 146 | Rams |
| 81 | 7 | Joe Noteboom | 58.24 | 47.40 | 61.30 | 376 | Rams |

## HB — Running Back

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Chubb | 86.80 | 85.20 | 83.70 | 293 | Browns |
| 2 | 2 | Austin Ekeler | 84.07 | 84.84 | 79.39 | 362 | Chargers |
| 3 | 3 | Christian McCaffrey | 81.51 | 86.50 | 74.01 | 570 | Panthers |
| 4 | 4 | Josh Jacobs | 81.42 | 76.86 | 80.30 | 147 | Raiders |
| 5 | 5 | Aaron Jones | 80.42 | 83.34 | 74.30 | 309 | Packers |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Kareem Hunt | 79.21 | 73.77 | 78.67 | 190 | Browns |
| 7 | 2 | Saquon Barkley | 79.12 | 72.17 | 79.58 | 377 | Giants |
| 8 | 3 | Dalvin Cook | 78.57 | 79.27 | 73.94 | 269 | Vikings |
| 9 | 4 | Alvin Kamara | 78.50 | 69.72 | 80.18 | 345 | Saints |
| 10 | 5 | Chris Carson | 78.18 | 76.09 | 75.41 | 307 | Seahawks |
| 11 | 6 | Derrick Henry | 77.70 | 71.58 | 77.62 | 194 | Titans |
| 12 | 7 | Duke Johnson Jr. | 77.46 | 73.81 | 75.72 | 346 | Texans |
| 13 | 8 | Ezekiel Elliott | 76.41 | 77.00 | 71.85 | 465 | Cowboys |
| 14 | 9 | Raheem Mostert | 75.80 | 69.71 | 75.70 | 149 | 49ers |
| 15 | 10 | Kenyan Drake | 75.73 | 70.48 | 75.06 | 303 | Cardinals |
| 16 | 11 | Devin Singletary | 74.30 | 65.15 | 76.23 | 271 | Bills |
| 17 | 12 | Le'Veon Bell | 74.08 | 74.10 | 69.90 | 391 | Jets |

### Starter (40 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Mark Ingram II | 73.77 | 73.05 | 70.09 | 165 | Ravens |
| 19 | 2 | Gus Edwards | 73.48 | 65.93 | 74.34 | 126 | Ravens |
| 20 | 3 | Joe Mixon | 72.60 | 73.43 | 67.88 | 241 | Bengals |
| 21 | 4 | Phillip Lindsay | 71.99 | 69.57 | 69.44 | 211 | Broncos |
| 22 | 5 | James Conner | 71.75 | 70.62 | 68.33 | 153 | Steelers |
| 23 | 6 | Miles Sanders | 71.58 | 62.10 | 73.74 | 317 | Eagles |
| 24 | 7 | Jalen Richard | 71.57 | 64.47 | 72.13 | 204 | Raiders |
| 25 | 8 | Damien Williams | 70.95 | 64.06 | 71.38 | 193 | Chiefs |
| 26 | 9 | Carlos Hyde | 70.93 | 70.58 | 66.99 | 211 | Texans |
| 27 | 10 | Adrian Peterson | 70.68 | 65.68 | 69.84 | 127 | Commanders |
| 28 | 11 | Todd Gurley II | 70.63 | 67.00 | 68.88 | 391 | Rams |
| 29 | 12 | Kerryon Johnson | 70.61 | 63.51 | 71.17 | 104 | Lions |
| 30 | 13 | Chris Thompson | 70.48 | 63.89 | 70.71 | 221 | Commanders |
| 31 | 14 | Latavius Murray | 69.93 | 71.58 | 64.67 | 198 | Saints |
| 32 | 15 | James White | 69.81 | 75.54 | 61.82 | 337 | Patriots |
| 33 | 16 | J.D. McKissic | 69.54 | 67.89 | 66.48 | 154 | Lions |
| 34 | 17 | Marlon Mack | 69.47 | 66.48 | 67.29 | 177 | Colts |
| 35 | 18 | DeAndre Washington | 69.29 | 69.54 | 64.96 | 127 | Raiders |
| 36 | 19 | Dion Lewis | 69.26 | 60.17 | 71.16 | 198 | Titans |
| 37 | 20 | David Johnson | 69.19 | 70.60 | 64.08 | 232 | Cardinals |
| 38 | 21 | Jamaal Williams | 69.09 | 72.18 | 62.87 | 202 | Packers |
| 39 | 22 | Melvin Gordon III | 68.31 | 65.06 | 66.31 | 208 | Chargers |
| 40 | 23 | Brian Hill | 68.29 | 59.41 | 70.04 | 108 | Falcons |
| 41 | 24 | Royce Freeman | 68.26 | 64.96 | 66.29 | 278 | Broncos |
| 42 | 25 | Tevin Coleman | 68.17 | 66.35 | 65.21 | 170 | 49ers |
| 43 | 26 | LeSean McCoy | 67.98 | 63.93 | 66.52 | 145 | Chiefs |
| 44 | 27 | Rex Burkhead | 67.81 | 66.47 | 64.54 | 153 | Patriots |
| 45 | 28 | Leonard Fournette | 67.72 | 64.00 | 66.03 | 483 | Jaguars |
| 46 | 29 | Ronald Jones | 67.49 | 65.08 | 64.93 | 170 | Buccaneers |
| 47 | 30 | Chase Edmonds | 67.06 | 65.00 | 64.26 | 101 | Cardinals |
| 48 | 31 | Tarik Cohen | 66.23 | 60.59 | 65.82 | 361 | Bears |
| 49 | 32 | David Montgomery | 65.83 | 65.23 | 62.07 | 239 | Bears |
| 50 | 33 | Frank Gore | 65.33 | 63.03 | 62.69 | 121 | Bills |
| 51 | 34 | Sony Michel | 65.31 | 64.14 | 61.93 | 102 | Patriots |
| 52 | 35 | Devonta Freeman | 65.21 | 60.77 | 64.01 | 337 | Falcons |
| 53 | 36 | Nyheim Hines | 64.73 | 62.53 | 62.03 | 223 | Colts |
| 54 | 37 | T.J. Yeldon | 64.67 | 59.03 | 64.27 | 100 | Bills |
| 55 | 38 | Ty Johnson | 64.07 | 53.81 | 66.74 | 188 | Lions |
| 56 | 39 | Peyton Barber | 63.59 | 62.64 | 60.06 | 110 | Buccaneers |
| 57 | 40 | Giovani Bernard | 62.40 | 52.63 | 64.75 | 262 | Bengals |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 58 | 1 | Jaylen Samuels | 60.90 | 55.88 | 60.08 | 206 | Steelers |
| 59 | 2 | Darrel Williams | 60.21 | 61.83 | 54.97 | 104 | Chiefs |
| 60 | 3 | Kalen Ballage | 59.98 | 59.03 | 56.44 | 111 | Dolphins |
| 61 | 4 | Dare Ogunbowale | 59.84 | 60.23 | 55.42 | 223 | Buccaneers |
| 62 | 5 | Patrick Laird | 57.74 | 55.10 | 55.33 | 158 | Dolphins |

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

### Good (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Cory Littleton | 78.99 | 79.00 | 74.81 | 1039 | Rams |
| 6 | 2 | Bobby Wagner | 76.65 | 76.10 | 73.17 | 1054 | Seahawks |
| 7 | 3 | Bobby Okereke | 75.67 | 73.96 | 72.65 | 472 | Colts |

### Starter (52 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | Jaylon Smith | 73.75 | 70.20 | 71.95 | 991 | Cowboys |
| 9 | 2 | Deion Jones | 73.40 | 76.60 | 70.24 | 946 | Falcons |
| 10 | 3 | T.J. Edwards | 72.82 | 70.64 | 74.28 | 112 | Eagles |
| 11 | 4 | Jayon Brown | 72.68 | 71.60 | 70.27 | 829 | Titans |
| 12 | 5 | Shaun Dion Hamilton | 72.60 | 70.17 | 74.60 | 387 | Commanders |
| 13 | 6 | Kevin Pierre-Louis | 72.33 | 75.54 | 74.87 | 213 | Bears |
| 14 | 7 | Josh Bynes | 71.96 | 73.52 | 71.02 | 391 | Ravens |
| 15 | 8 | Zach Cunningham | 71.59 | 69.00 | 69.78 | 943 | Texans |
| 16 | 9 | Vince Williams | 71.59 | 71.27 | 69.31 | 397 | Steelers |
| 17 | 10 | Jamie Collins Sr. | 71.36 | 75.76 | 66.35 | 813 | Patriots |
| 18 | 11 | Benardrick McKinney | 70.20 | 67.60 | 68.80 | 844 | Texans |
| 19 | 12 | Dont'a Hightower | 70.08 | 69.06 | 67.42 | 724 | Patriots |
| 20 | 13 | David Mayo | 69.65 | 71.32 | 68.02 | 631 | Giants |
| 21 | 14 | Kiko Alonso | 69.48 | 68.22 | 68.56 | 285 | Saints |
| 22 | 15 | Todd Davis | 68.95 | 65.10 | 68.82 | 897 | Broncos |
| 23 | 16 | C.J. Mosley | 68.67 | 69.60 | 71.48 | 114 | Jets |
| 24 | 17 | Dre Greenlaw | 68.63 | 63.76 | 67.71 | 725 | 49ers |
| 25 | 18 | Nick Kwiatkoski | 68.63 | 70.93 | 68.66 | 512 | Bears |
| 26 | 19 | Eric Wilson | 68.48 | 65.17 | 67.16 | 380 | Vikings |
| 27 | 20 | Fred Warner | 68.39 | 66.90 | 65.21 | 985 | 49ers |
| 28 | 21 | Leon Jacobs | 68.03 | 65.23 | 68.59 | 325 | Jaguars |
| 29 | 22 | Devin Bush | 67.96 | 62.90 | 67.17 | 889 | Steelers |
| 30 | 23 | Josey Jewell | 67.80 | 64.29 | 69.23 | 214 | Broncos |
| 31 | 24 | Reggie Ragland | 67.50 | 64.34 | 67.11 | 235 | Chiefs |
| 32 | 25 | Foyesade Oluokun | 67.33 | 61.66 | 66.94 | 310 | Falcons |
| 33 | 26 | Shaq Thompson | 67.32 | 65.80 | 66.05 | 962 | Panthers |
| 34 | 27 | Matt Milano | 67.32 | 65.30 | 66.39 | 893 | Bills |
| 35 | 28 | Drue Tranquill | 66.20 | 64.50 | 67.34 | 382 | Chargers |
| 36 | 29 | Kyzir White | 65.97 | 64.45 | 67.90 | 372 | Chargers |
| 37 | 30 | Raekwon McMillan | 65.69 | 63.09 | 64.83 | 516 | Dolphins |
| 38 | 31 | Jahlani Tavai | 65.39 | 61.37 | 64.93 | 597 | Lions |
| 39 | 32 | K.J. Wright | 65.24 | 62.10 | 66.82 | 997 | Seahawks |
| 40 | 33 | Nigel Bradham | 65.20 | 64.42 | 63.96 | 717 | Eagles |
| 41 | 34 | L.J. Fort | 65.19 | 65.62 | 65.64 | 254 | Ravens |
| 42 | 35 | Elandon Roberts | 65.17 | 60.94 | 64.86 | 202 | Patriots |
| 43 | 36 | Ja'Whaun Bentley | 65.00 | 64.40 | 66.32 | 275 | Patriots |
| 44 | 37 | Anthony Barr | 64.93 | 60.60 | 65.63 | 930 | Vikings |
| 45 | 38 | B.J. Goodson | 64.79 | 61.22 | 66.24 | 254 | Packers |
| 46 | 39 | Thomas Davis Sr. | 64.27 | 61.69 | 63.07 | 805 | Chargers |
| 47 | 40 | Blake Martinez | 64.15 | 58.90 | 63.49 | 1024 | Packers |
| 48 | 41 | Tremaine Edmunds | 64.03 | 60.60 | 62.54 | 981 | Bills |
| 49 | 42 | Anthony Walker Jr. | 63.76 | 60.99 | 64.45 | 811 | Colts |
| 50 | 43 | Joe Schobert | 63.69 | 59.10 | 63.51 | 1057 | Browns |
| 51 | 44 | Ben Gedeon | 63.52 | 60.04 | 66.16 | 102 | Vikings |
| 52 | 45 | Danny Trevathan | 63.38 | 61.70 | 64.81 | 559 | Bears |
| 53 | 46 | Jordan Hicks | 63.23 | 61.00 | 63.68 | 1133 | Cardinals |
| 54 | 47 | Sean Lee | 62.97 | 61.39 | 63.70 | 637 | Cowboys |
| 55 | 48 | Cole Holcomb | 62.71 | 56.26 | 62.84 | 718 | Commanders |
| 56 | 49 | Jon Bostic | 62.45 | 55.90 | 63.07 | 1031 | Commanders |
| 57 | 50 | Ben Niemann | 62.43 | 59.79 | 64.58 | 400 | Chiefs |
| 58 | 51 | Damien Wilson | 62.13 | 55.54 | 62.36 | 709 | Chiefs |
| 59 | 52 | Sione Takitaki | 62.03 | 61.76 | 66.37 | 105 | Browns |

### Rotation/backup (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 60 | 1 | Tahir Whitehead | 61.94 | 53.20 | 63.60 | 941 | Raiders |
| 61 | 2 | Wesley Woodyard | 61.93 | 57.43 | 62.43 | 325 | Titans |
| 62 | 3 | Joe Thomas | 61.81 | 62.41 | 63.50 | 246 | Cowboys |
| 63 | 4 | Will Compton | 61.77 | 62.35 | 65.75 | 245 | Raiders |
| 64 | 5 | Mychal Kendricks | 61.75 | 62.52 | 61.85 | 649 | Seahawks |
| 65 | 6 | Kevin Minter | 61.54 | 63.01 | 64.72 | 275 | Buccaneers |
| 66 | 7 | Kwon Alexander | 61.35 | 60.86 | 65.65 | 357 | 49ers |
| 67 | 8 | Mark Barron | 61.32 | 57.61 | 61.61 | 750 | Steelers |
| 68 | 9 | Kamu Grugier-Hill | 61.05 | 60.06 | 62.74 | 300 | Eagles |
| 69 | 10 | Denzel Perryman | 60.45 | 59.47 | 63.08 | 359 | Chargers |
| 70 | 11 | Leighton Vander Esch | 60.29 | 58.60 | 61.80 | 510 | Cowboys |
| 71 | 12 | Germaine Pratt | 60.07 | 53.43 | 61.37 | 437 | Bengals |
| 72 | 13 | Nathan Gerry | 60.01 | 58.52 | 61.62 | 620 | Eagles |
| 73 | 14 | De'Vondre Campbell | 59.97 | 50.10 | 62.39 | 921 | Falcons |
| 74 | 15 | Rashaan Evans | 59.87 | 49.90 | 63.13 | 950 | Titans |
| 75 | 16 | Roquan Smith | 59.71 | 52.88 | 62.69 | 719 | Bears |
| 76 | 17 | Travin Howard | 59.60 | 60.28 | 65.85 | 102 | Rams |
| 77 | 18 | Nick Vigil | 59.10 | 54.30 | 60.73 | 985 | Bengals |
| 78 | 19 | Jalen Reeves-Maybin | 58.88 | 56.87 | 61.69 | 298 | Lions |
| 79 | 20 | Deone Bucannon | 58.70 | 56.67 | 59.74 | 244 | Giants |
| 80 | 21 | Sam Eguavoen | 58.61 | 51.47 | 59.21 | 621 | Dolphins |
| 81 | 22 | Jerome Baker | 58.55 | 46.70 | 62.28 | 1080 | Dolphins |
| 82 | 23 | James Burgess | 58.47 | 55.42 | 61.02 | 662 | Jets |
| 83 | 24 | Vontaze Burfict | 58.36 | 61.52 | 62.41 | 191 | Raiders |
| 84 | 25 | Alec Ogletree | 58.25 | 55.60 | 58.35 | 850 | Giants |
| 85 | 26 | Craig Robertson | 58.14 | 58.80 | 61.24 | 189 | Saints |
| 86 | 27 | Patrick Onwuasor | 57.98 | 51.95 | 58.87 | 473 | Ravens |
| 87 | 28 | Kentrell Brothers | 57.00 | 59.12 | 63.82 | 111 | Vikings |
| 88 | 29 | Anthony Hitchens | 56.88 | 49.32 | 59.42 | 699 | Chiefs |
| 89 | 30 | Devin White | 56.64 | 51.90 | 58.77 | 826 | Buccaneers |
| 90 | 31 | Jermaine Carter | 56.44 | 56.61 | 59.20 | 261 | Panthers |
| 91 | 32 | Corey Nelson | 55.76 | 57.44 | 60.15 | 106 | Broncos |
| 92 | 33 | Nicholas Morrow | 55.65 | 47.00 | 58.08 | 728 | Raiders |
| 93 | 34 | Matthew Adams | 55.42 | 53.88 | 57.75 | 105 | Colts |
| 94 | 35 | Dylan Cole | 55.06 | 55.81 | 56.32 | 136 | Texans |
| 95 | 36 | A.J. Klein | 54.89 | 48.30 | 56.46 | 754 | Saints |
| 96 | 37 | Preston Brown | 54.76 | 49.83 | 59.81 | 427 | Jaguars |
| 97 | 38 | Christian Jones | 54.59 | 47.50 | 56.91 | 609 | Lions |
| 98 | 39 | Myles Jack | 54.42 | 47.05 | 57.76 | 613 | Jaguars |
| 99 | 40 | Cody Barton | 54.16 | 55.15 | 57.67 | 151 | Seahawks |
| 100 | 41 | Ryan Connelly | 53.93 | 58.62 | 65.14 | 187 | Giants |
| 101 | 42 | Haason Reddick | 53.57 | 41.84 | 57.23 | 690 | Cardinals |
| 102 | 43 | Christian Kirksey | 53.36 | 55.28 | 58.01 | 112 | Browns |
| 103 | 44 | Blake Cashman | 53.02 | 52.16 | 60.29 | 424 | Jets |
| 104 | 45 | Austin Calitro | 53.01 | 51.61 | 57.59 | 234 | Jaguars |
| 105 | 46 | Azeez Al-Shaair | 52.79 | 51.80 | 55.53 | 174 | 49ers |
| 106 | 47 | Mack Wilson Sr. | 52.03 | 41.40 | 55.98 | 942 | Browns |
| 107 | 48 | Jarrad Davis | 51.21 | 40.89 | 56.94 | 654 | Lions |
| 108 | 49 | Joe Walker | 51.04 | 48.83 | 56.48 | 537 | Cardinals |
| 109 | 50 | Darron Lee | 50.87 | 44.22 | 56.24 | 161 | Chiefs |
| 110 | 51 | Troy Reeder | 49.91 | 41.07 | 54.77 | 298 | Rams |
| 111 | 52 | Najee Goode | 49.57 | 47.40 | 55.09 | 295 | Jaguars |
| 112 | 53 | Tyrell Adams | 48.58 | 54.28 | 58.01 | 108 | Texans |
| 113 | 54 | Neville Hewitt | 48.16 | 41.11 | 55.46 | 762 | Jets |
| 114 | 55 | Quincy Williams | 46.88 | 40.00 | 54.60 | 494 | Jaguars |
| 115 | 56 | Donald Payne | 46.24 | 40.00 | 56.06 | 348 | Jaguars |

## QB — Quarterback

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Russell Wilson | 85.41 | 87.65 | 79.78 | 659 | Seahawks |
| 2 | 2 | Drew Brees | 84.15 | 87.99 | 81.42 | 415 | Saints |
| 3 | 3 | Patrick Mahomes | 83.50 | 84.80 | 81.18 | 591 | Chiefs |
| 4 | 4 | Kirk Cousins | 80.51 | 80.94 | 78.08 | 515 | Vikings |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Deshaun Watson | 78.50 | 77.25 | 76.42 | 628 | Texans |
| 6 | 2 | Aaron Rodgers | 78.43 | 81.60 | 72.29 | 681 | Packers |
| 7 | 3 | Derek Carr | 78.06 | 77.58 | 75.10 | 578 | Raiders |
| 8 | 4 | Dak Prescott | 77.70 | 75.03 | 75.50 | 679 | Cowboys |
| 9 | 5 | Philip Rivers | 77.61 | 77.96 | 73.28 | 682 | Chargers |
| 10 | 6 | Matthew Stafford | 77.24 | 76.24 | 79.11 | 336 | Lions |
| 11 | 7 | Matt Ryan | 76.83 | 77.30 | 71.90 | 732 | Falcons |
| 12 | 8 | Tom Brady | 76.68 | 81.90 | 67.96 | 688 | Patriots |
| 13 | 9 | Jared Goff | 75.53 | 75.54 | 71.13 | 709 | Rams |
| 14 | 10 | Carson Wentz | 75.10 | 76.76 | 70.71 | 718 | Eagles |

### Starter (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Jimmy Garoppolo | 73.34 | 74.73 | 76.52 | 565 | 49ers |
| 16 | 2 | Lamar Jackson | 73.02 | 79.64 | 77.68 | 491 | Ravens |
| 17 | 3 | Ryan Tannehill | 73.00 | 75.53 | 84.45 | 356 | Titans |
| 18 | 4 | Baker Mayfield | 72.45 | 74.64 | 67.09 | 647 | Browns |
| 19 | 5 | Ryan Fitzpatrick | 71.82 | 76.33 | 67.82 | 620 | Dolphins |
| 20 | 6 | Jameis Winston | 70.70 | 67.61 | 70.25 | 747 | Buccaneers |
| 21 | 7 | Andy Dalton | 66.47 | 69.23 | 62.05 | 609 | Bengals |
| 22 | 8 | Gardner Minshew | 66.19 | 70.16 | 66.92 | 598 | Jaguars |
| 23 | 9 | Marcus Mariota | 65.59 | 65.96 | 70.17 | 214 | Titans |
| 24 | 10 | Sam Darnold | 64.87 | 63.65 | 65.22 | 521 | Jets |
| 25 | 11 | Case Keenum | 64.68 | 63.07 | 68.11 | 290 | Commanders |
| 26 | 12 | Kyler Murray | 64.65 | 61.10 | 66.82 | 661 | Cardinals |
| 27 | 13 | Mitch Trubisky | 64.47 | 62.51 | 63.58 | 613 | Bears |
| 28 | 14 | Daniel Jones | 63.77 | 64.95 | 65.53 | 566 | Giants |
| 29 | 15 | Joe Flacco | 63.04 | 65.74 | 64.01 | 315 | Broncos |
| 30 | 16 | Josh Allen | 62.93 | 61.03 | 63.07 | 576 | Bills |
| 31 | 17 | Eli Manning | 62.57 | 63.18 | 66.38 | 160 | Giants |
| 32 | 18 | Teddy Bridgewater | 62.24 | 68.77 | 70.84 | 236 | Saints |

### Rotation/backup (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Nick Foles | 60.98 | 65.57 | 67.32 | 137 | Jaguars |
| 34 | 2 | Matt Moore | 60.61 | 59.64 | 71.85 | 104 | Chiefs |
| 35 | 3 | Jacoby Brissett | 60.50 | 58.73 | 63.79 | 545 | Colts |
| 36 | 4 | Dwayne Haskins | 59.52 | 64.63 | 60.57 | 259 | Commanders |
| 37 | 5 | Drew Lock | 58.94 | 58.56 | 65.40 | 186 | Broncos |
| 38 | 6 | Kyle Allen | 58.20 | 51.46 | 63.36 | 578 | Panthers |
| 39 | 7 | Mason Rudolph | 57.57 | 55.13 | 60.84 | 331 | Steelers |
| 40 | 8 | Devlin Hodges | 56.84 | 53.94 | 60.36 | 196 | Steelers |
| 41 | 9 | David Blough | 56.61 | 57.29 | 56.35 | 210 | Lions |
| 42 | 10 | Jeff Driskel | 56.45 | 55.74 | 58.21 | 141 | Lions |
| 43 | 11 | Brandon Allen | 55.66 | 54.45 | 56.10 | 105 | Broncos |
| 44 | 12 | Josh Rosen | 54.33 | 54.14 | 55.90 | 137 | Dolphins |
| 45 | 13 | Ryan Finley | 53.39 | 46.30 | 53.60 | 110 | Bengals |

## S — Safety

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Justin Simmons | 93.34 | 91.10 | 91.30 | 1053 | Broncos |
| 2 | 2 | Anthony Harris | 91.30 | 91.60 | 91.41 | 910 | Vikings |
| 3 | 3 | Harrison Smith | 90.22 | 91.40 | 85.79 | 971 | Vikings |
| 4 | 4 | Jamal Adams | 90.11 | 87.50 | 88.72 | 959 | Jets |
| 5 | 5 | Marcus Williams | 88.33 | 89.20 | 84.10 | 950 | Saints |
| 6 | 6 | Devin McCourty | 87.67 | 89.10 | 82.55 | 946 | Patriots |
| 7 | 7 | Tre Boston | 87.13 | 90.60 | 81.28 | 1104 | Panthers |
| 8 | 8 | Micah Hyde | 83.34 | 80.90 | 81.11 | 969 | Bills |
| 9 | 9 | Earl Thomas III | 81.09 | 84.70 | 79.20 | 891 | Ravens |
| 10 | 10 | Justin Reid | 80.98 | 79.30 | 78.58 | 916 | Texans |
| 11 | 11 | Kareem Jackson | 80.79 | 79.80 | 79.24 | 842 | Broncos |
| 12 | 12 | Jimmie Ward | 80.62 | 80.76 | 82.00 | 806 | 49ers |
| 13 | 13 | Adrian Amos | 80.61 | 76.00 | 80.35 | 1036 | Packers |
| 14 | 14 | Tyrann Mathieu | 80.32 | 81.60 | 75.30 | 1080 | Chiefs |

### Good (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Jarrod Wilson | 79.97 | 79.60 | 77.51 | 1052 | Jaguars |
| 16 | 2 | Minkah Fitzpatrick | 79.92 | 77.40 | 77.43 | 1046 | Steelers |
| 17 | 3 | Chuck Clark | 79.90 | 80.81 | 77.53 | 745 | Ravens |
| 18 | 4 | Kevin Byard | 79.62 | 71.70 | 80.74 | 1098 | Titans |
| 19 | 5 | Adrian Phillips | 79.28 | 75.73 | 82.38 | 282 | Chargers |
| 20 | 6 | Ha Ha Clinton-Dix | 79.10 | 75.50 | 77.33 | 1066 | Bears |
| 21 | 7 | Derwin James Jr. | 78.23 | 78.19 | 81.26 | 299 | Chargers |
| 22 | 8 | Juan Thornhill | 77.49 | 78.00 | 72.99 | 996 | Chiefs |
| 23 | 9 | Tracy Walker III | 77.31 | 76.00 | 75.96 | 843 | Lions |
| 24 | 10 | Xavier Woods | 76.97 | 77.60 | 73.74 | 978 | Cowboys |
| 25 | 11 | Marcus Maye | 76.38 | 77.40 | 74.67 | 1089 | Jets |
| 26 | 12 | Eric Weddle | 76.29 | 70.50 | 75.99 | 1031 | Rams |
| 27 | 13 | Duron Harmon | 76.12 | 73.75 | 73.54 | 657 | Patriots |
| 28 | 14 | Jordan Poyer | 75.94 | 70.10 | 75.67 | 977 | Bills |
| 29 | 15 | Quandre Diggs | 75.41 | 76.54 | 73.63 | 606 | Seahawks |
| 30 | 16 | Tavon Wilson | 74.91 | 74.00 | 73.24 | 840 | Lions |
| 31 | 17 | Andrew Sendejo | 74.32 | 74.81 | 75.02 | 384 | Vikings |
| 32 | 18 | Damarious Randall | 74.20 | 69.27 | 76.23 | 723 | Browns |
| 33 | 19 | Jeff Heath | 74.14 | 68.55 | 75.46 | 719 | Cowboys |

### Starter (47 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Budda Baker | 72.88 | 64.80 | 74.10 | 1120 | Cardinals |
| 35 | 2 | Darnell Savage | 72.05 | 73.10 | 69.26 | 865 | Packers |
| 36 | 3 | Malcolm Jenkins | 71.96 | 67.50 | 70.77 | 1015 | Eagles |
| 37 | 4 | Ronnie Harrison | 71.51 | 68.50 | 71.43 | 833 | Jaguars |
| 38 | 5 | C.J. Gardner-Johnson | 71.38 | 67.17 | 70.02 | 547 | Saints |
| 39 | 6 | Troy Apke | 70.45 | 67.16 | 73.02 | 210 | Commanders |
| 40 | 7 | Marquise Blair | 70.40 | 70.98 | 73.14 | 230 | Seahawks |
| 41 | 8 | Jalen Thompson | 70.19 | 68.83 | 71.09 | 607 | Cardinals |
| 42 | 9 | Rodney McLeod | 70.19 | 70.60 | 69.81 | 1013 | Eagles |
| 43 | 10 | Erik Harris | 69.85 | 66.70 | 68.81 | 900 | Raiders |
| 44 | 11 | Antoine Bethea | 69.57 | 62.40 | 70.39 | 1107 | Giants |
| 45 | 12 | Eddie Jackson | 69.35 | 66.10 | 67.99 | 1061 | Bears |
| 46 | 13 | Reshad Jones | 69.04 | 64.26 | 74.92 | 189 | Dolphins |
| 47 | 14 | Clayton Fejedelem | 68.55 | 64.08 | 71.73 | 111 | Bengals |
| 48 | 15 | Taylor Rapp | 68.52 | 61.40 | 70.13 | 823 | Rams |
| 49 | 16 | Curtis Riley | 68.51 | 64.45 | 70.30 | 275 | Raiders |
| 50 | 17 | Andrew Adams | 68.26 | 63.11 | 71.69 | 616 | Buccaneers |
| 51 | 18 | Jessie Bates III | 68.14 | 64.90 | 66.14 | 1059 | Bengals |
| 52 | 19 | Khari Willis | 67.92 | 65.55 | 67.41 | 620 | Colts |
| 53 | 20 | Rayshawn Jenkins | 67.92 | 67.50 | 66.53 | 964 | Chargers |
| 54 | 21 | Jaquiski Tartt | 67.74 | 67.44 | 69.82 | 673 | 49ers |
| 55 | 22 | Karl Joseph | 67.59 | 63.92 | 70.65 | 575 | Raiders |
| 56 | 23 | Jabrill Peppers | 67.51 | 66.40 | 67.31 | 705 | Giants |
| 57 | 24 | Bradley McDougald | 67.21 | 64.50 | 65.37 | 941 | Seahawks |
| 58 | 25 | Tashaun Gipson Sr. | 67.20 | 64.50 | 65.86 | 868 | Texans |
| 59 | 26 | George Odum | 66.94 | 62.41 | 69.44 | 284 | Colts |
| 60 | 27 | Michael Thomas | 66.64 | 63.45 | 66.49 | 302 | Giants |
| 61 | 28 | Terrence Brooks | 66.33 | 60.75 | 71.08 | 274 | Patriots |
| 62 | 29 | Marqui Christian | 66.30 | 66.84 | 63.66 | 371 | Rams |
| 63 | 30 | Landon Collins | 66.27 | 60.60 | 67.86 | 1057 | Commanders |
| 64 | 31 | Malik Hooker | 66.16 | 62.76 | 68.33 | 789 | Colts |
| 65 | 32 | Bobby McCain | 65.93 | 67.77 | 67.83 | 540 | Dolphins |
| 66 | 33 | Terrell Edmunds | 65.63 | 58.30 | 66.35 | 1036 | Steelers |
| 67 | 34 | Trey Marshall | 65.59 | 59.60 | 67.37 | 160 | Broncos |
| 68 | 35 | Amani Hooker | 65.51 | 64.01 | 62.35 | 335 | Titans |
| 69 | 36 | Daniel Sorensen | 65.01 | 64.84 | 63.78 | 563 | Chiefs |
| 70 | 37 | Tarvarius Moore | 64.71 | 61.12 | 66.84 | 234 | 49ers |
| 71 | 38 | Sheldrick Redwine | 64.44 | 65.66 | 67.79 | 374 | Browns |
| 72 | 39 | Kenny Vaccaro | 64.42 | 56.40 | 67.36 | 1062 | Titans |
| 73 | 40 | Steven Parker | 64.28 | 61.54 | 66.11 | 339 | Dolphins |
| 74 | 41 | Marcus Epps | 64.11 | 61.36 | 69.08 | 110 | Eagles |
| 75 | 42 | Clayton Geathers | 64.06 | 58.96 | 67.88 | 528 | Colts |
| 76 | 43 | Morgan Burnett | 63.83 | 58.80 | 69.59 | 367 | Browns |
| 77 | 44 | Darian Thompson | 63.43 | 62.08 | 65.90 | 425 | Cowboys |
| 78 | 45 | Ricardo Allen | 62.27 | 57.30 | 65.49 | 950 | Falcons |
| 79 | 46 | Dean Marlowe | 62.03 | 60.72 | 71.23 | 108 | Bills |
| 80 | 47 | Adrian Colbert | 62.02 | 63.75 | 66.50 | 361 | Dolphins |

### Rotation/backup (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 81 | 1 | Will Parks | 60.97 | 57.61 | 60.08 | 537 | Broncos |
| 82 | 2 | Patrick Chung | 60.94 | 56.28 | 61.77 | 642 | Patriots |
| 83 | 3 | Delano Hill | 60.63 | 63.08 | 61.69 | 300 | Seahawks |
| 84 | 4 | Jeremy Reaves | 60.15 | 58.96 | 67.84 | 111 | Commanders |
| 85 | 5 | John Johnson III | 59.38 | 57.47 | 61.69 | 395 | Rams |
| 86 | 6 | Marcell Harris | 59.09 | 59.55 | 63.33 | 340 | 49ers |
| 87 | 7 | Mike Edwards | 58.71 | 53.53 | 59.03 | 614 | Buccaneers |
| 88 | 8 | Will Harris | 58.49 | 55.86 | 56.07 | 668 | Lions |
| 89 | 9 | Jamal Carter | 57.92 | 57.40 | 59.52 | 105 | Falcons |
| 90 | 10 | Vonn Bell | 57.57 | 49.40 | 60.42 | 872 | Saints |
| 91 | 11 | Darian Stewart | 57.44 | 56.24 | 58.85 | 169 | Buccaneers |
| 92 | 12 | Shawn Williams | 57.10 | 51.00 | 58.04 | 1002 | Bengals |
| 93 | 13 | Anthony Levine | 56.49 | 56.67 | 53.23 | 167 | Ravens |
| 94 | 14 | Will Redmond | 55.49 | 55.64 | 58.52 | 271 | Packers |
| 95 | 15 | Kemal Ishmael | 55.08 | 51.35 | 58.50 | 282 | Falcons |
| 96 | 16 | Ibraheim Campbell | 55.03 | 57.61 | 59.34 | 181 | Packers |
| 97 | 17 | Montae Nicholson | 54.96 | 49.30 | 59.66 | 873 | Commanders |
| 98 | 18 | Deionte Thompson | 54.16 | 56.30 | 55.86 | 252 | Cardinals |
| 99 | 19 | Tedric Thompson | 54.11 | 51.99 | 60.11 | 387 | Seahawks |
| 100 | 20 | Brandon Wilson | 53.97 | 55.76 | 53.09 | 183 | Bengals |
| 101 | 21 | D.J. Swearinger Sr. | 53.61 | 44.53 | 59.47 | 484 | Saints |
| 102 | 22 | Tony Jefferson | 53.26 | 51.67 | 56.51 | 281 | Ravens |
| 103 | 23 | Andrew Wingard | 53.05 | 55.31 | 53.63 | 185 | Jaguars |
| 104 | 24 | Eric Reid | 52.85 | 40.00 | 58.81 | 1094 | Panthers |
| 105 | 25 | Roderic Teamer | 51.10 | 53.10 | 56.46 | 377 | Chargers |
| 106 | 26 | Jordan Whitehead | 50.68 | 40.00 | 55.34 | 919 | Buccaneers |
| 107 | 27 | Walt Aikens | 50.39 | 51.64 | 55.07 | 104 | Dolphins |
| 108 | 28 | Keanu Neal | 49.45 | 47.96 | 57.75 | 166 | Falcons |

## T — Tackle

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Ryan Ramczyk | 95.89 | 90.80 | 95.11 | 1058 | Saints |
| 2 | 2 | Ronnie Stanley | 93.21 | 88.09 | 92.46 | 938 | Ravens |
| 3 | 3 | Lane Johnson | 92.99 | 86.69 | 93.03 | 759 | Eagles |
| 4 | 4 | La'el Collins | 92.30 | 86.40 | 92.07 | 1000 | Cowboys |
| 5 | 5 | Mitchell Schwartz | 89.70 | 84.00 | 89.33 | 1046 | Chiefs |
| 6 | 6 | Jason Peters | 87.87 | 81.51 | 87.95 | 872 | Eagles |
| 7 | 7 | Braden Smith | 87.15 | 79.80 | 87.89 | 1075 | Colts |
| 8 | 8 | Anthony Castonzo | 86.41 | 81.30 | 85.65 | 1076 | Colts |
| 9 | 9 | Bryan Bulaga | 85.99 | 77.32 | 87.60 | 898 | Packers |
| 10 | 10 | Terron Armstead | 85.83 | 80.09 | 85.49 | 935 | Saints |
| 11 | 11 | Jack Conklin | 85.74 | 77.61 | 86.99 | 933 | Titans |
| 12 | 12 | David Bakhtiari | 85.20 | 78.50 | 85.50 | 1075 | Packers |
| 13 | 13 | Jake Matthews | 85.09 | 79.70 | 84.51 | 1163 | Falcons |
| 14 | 14 | Laremy Tunsil | 85.06 | 75.46 | 87.30 | 915 | Texans |
| 15 | 15 | Garett Bolles | 83.90 | 76.20 | 84.87 | 1013 | Broncos |
| 16 | 16 | Taylor Lewan | 83.77 | 72.19 | 87.32 | 711 | Titans |
| 17 | 17 | Taylor Moton | 83.70 | 76.20 | 84.54 | 1106 | Panthers |
| 18 | 18 | Matt Feiler | 83.13 | 75.90 | 83.78 | 995 | Steelers |
| 19 | 19 | Taylor Decker | 83.10 | 75.90 | 83.73 | 1017 | Lions |
| 20 | 20 | Tyron Smith | 83.02 | 76.06 | 83.49 | 882 | Cowboys |
| 21 | 21 | Jake Rodgers | 82.84 | 68.40 | 88.30 | 117 | Broncos |
| 22 | 22 | Duane Brown | 81.75 | 73.23 | 83.27 | 793 | Seahawks |
| 23 | 23 | Dion Dawkins | 81.36 | 73.40 | 82.50 | 1016 | Bills |
| 24 | 24 | Brian O'Neill | 81.08 | 70.60 | 83.90 | 967 | Vikings |
| 25 | 25 | Joe Staley | 80.85 | 70.14 | 83.82 | 434 | 49ers |
| 26 | 26 | Orlando Brown Jr. | 80.79 | 72.00 | 82.49 | 1105 | Ravens |
| 27 | 27 | Alejandro Villanueva | 80.72 | 74.00 | 81.04 | 995 | Steelers |
| 28 | 28 | Andrew Whitworth | 80.50 | 72.80 | 81.47 | 1097 | Rams |
| 29 | 29 | Demar Dotson | 80.41 | 70.80 | 82.65 | 1045 | Buccaneers |

### Good (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 30 | 1 | Riley Reiff | 79.86 | 70.81 | 81.72 | 874 | Vikings |
| 31 | 2 | Trent Brown | 79.73 | 67.83 | 83.50 | 581 | Raiders |
| 32 | 3 | Cornelius Lucas | 79.38 | 68.75 | 82.30 | 507 | Bears |
| 33 | 4 | Donovan Smith | 78.92 | 70.40 | 80.44 | 1055 | Buccaneers |
| 34 | 5 | Mike McGlinchey | 78.47 | 66.90 | 82.01 | 777 | 49ers |
| 35 | 6 | Ty Nsekhe | 78.31 | 64.40 | 83.42 | 358 | Bills |
| 36 | 7 | Isaiah Wynn | 78.08 | 67.07 | 81.25 | 502 | Patriots |
| 37 | 8 | Marcus Cannon | 77.86 | 68.00 | 80.26 | 1008 | Patriots |
| 38 | 9 | Dennis Kelly | 77.45 | 66.58 | 80.53 | 352 | Titans |
| 39 | 10 | Greg Robinson | 76.98 | 65.89 | 80.20 | 860 | Browns |
| 40 | 11 | Roderick Johnson | 76.68 | 64.81 | 80.42 | 365 | Texans |
| 41 | 12 | Kelvin Beachum | 76.31 | 66.69 | 78.55 | 805 | Jets |
| 42 | 13 | James Hurst | 76.15 | 63.55 | 80.39 | 194 | Ravens |
| 43 | 14 | Cedric Ogbuehi | 76.08 | 62.80 | 80.77 | 156 | Jaguars |
| 44 | 15 | Morgan Moses | 75.94 | 64.99 | 79.08 | 858 | Commanders |
| 45 | 16 | Zach Banner | 75.75 | 68.71 | 76.28 | 216 | Steelers |
| 46 | 17 | Cordy Glenn | 75.70 | 66.10 | 77.94 | 291 | Bengals |
| 47 | 18 | D.J. Humphries | 75.53 | 64.50 | 78.71 | 1046 | Cardinals |
| 48 | 19 | Donald Penn | 75.53 | 63.87 | 79.13 | 885 | Commanders |
| 49 | 20 | Russell Okung | 75.32 | 61.70 | 80.23 | 257 | Chargers |
| 50 | 21 | Jawaan Taylor | 74.98 | 63.70 | 78.34 | 1087 | Jaguars |
| 51 | 22 | David Sharpe | 74.84 | 62.56 | 78.86 | 268 | Raiders |
| 52 | 23 | Kolton Miller | 74.84 | 65.00 | 77.24 | 1019 | Raiders |
| 53 | 24 | Eric Fisher | 74.68 | 63.66 | 77.86 | 467 | Chiefs |
| 54 | 25 | Bobby Massie | 74.55 | 62.52 | 78.40 | 612 | Bears |
| 55 | 26 | Tyrell Crosby | 74.44 | 61.52 | 78.88 | 397 | Lions |
| 56 | 27 | Mike Remmers | 74.42 | 63.95 | 77.24 | 870 | Giants |
| 57 | 28 | Justin Skule | 74.09 | 61.71 | 78.17 | 545 | 49ers |
| 58 | 29 | Justin Murray | 74.06 | 62.68 | 77.48 | 844 | Cardinals |

### Starter (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | Brandon Shell | 73.97 | 63.26 | 76.95 | 806 | Jets |
| 60 | 2 | Nate Solder | 73.54 | 64.90 | 75.14 | 1011 | Giants |
| 61 | 3 | Trey Pipkins III | 73.34 | 61.67 | 76.96 | 251 | Chargers |
| 62 | 4 | George Fant | 73.04 | 61.52 | 76.56 | 472 | Seahawks |
| 63 | 5 | Marshall Newhouse | 72.96 | 62.24 | 75.94 | 729 | Patriots |
| 64 | 6 | Geron Christian | 72.92 | 61.20 | 76.56 | 146 | Commanders |
| 65 | 7 | Sam Tevi | 72.61 | 59.82 | 76.97 | 783 | Chargers |
| 66 | 8 | Elijah Wilkinson | 72.02 | 59.72 | 76.06 | 833 | Broncos |
| 67 | 9 | Charles Leno Jr. | 71.47 | 58.60 | 75.89 | 1066 | Bears |
| 68 | 10 | Dennis Daley | 71.39 | 58.08 | 76.10 | 686 | Panthers |
| 69 | 11 | Jesse Davis | 71.29 | 58.91 | 75.37 | 975 | Dolphins |
| 70 | 12 | Cam Fleming | 71.18 | 59.69 | 74.68 | 258 | Cowboys |
| 71 | 13 | Rick Wagner | 70.94 | 58.71 | 74.92 | 753 | Lions |
| 72 | 14 | Bobby Hart | 70.87 | 57.60 | 75.55 | 1086 | Bengals |
| 73 | 15 | Andre Dillard | 70.79 | 59.82 | 73.94 | 337 | Eagles |
| 74 | 16 | Patrick Omameh | 70.67 | 59.16 | 74.18 | 156 | Saints |
| 75 | 17 | Germain Ifedi | 70.26 | 56.20 | 75.47 | 1107 | Seahawks |
| 76 | 18 | Daryl Williams | 69.50 | 56.28 | 74.15 | 838 | Panthers |
| 77 | 19 | Julie'n Davenport | 69.13 | 57.42 | 72.77 | 534 | Dolphins |
| 78 | 20 | Cam Robinson | 68.84 | 55.11 | 73.82 | 870 | Jaguars |
| 79 | 21 | Greg Little | 68.04 | 56.52 | 71.56 | 224 | Panthers |
| 80 | 22 | Kaleb McGary | 67.97 | 53.00 | 73.78 | 1105 | Falcons |
| 81 | 23 | Chris Clark | 67.74 | 54.88 | 72.15 | 342 | Texans |
| 82 | 24 | Cody Ford | 67.61 | 53.42 | 72.91 | 739 | Bills |
| 83 | 25 | Chuma Edoga | 67.51 | 53.14 | 72.93 | 421 | Jets |
| 84 | 26 | Rob Havenstein | 67.09 | 52.04 | 72.95 | 616 | Rams |
| 85 | 27 | Chris Hubbard | 66.78 | 50.96 | 73.16 | 891 | Browns |
| 86 | 28 | Bobby Evans | 66.73 | 52.66 | 71.94 | 472 | Rams |
| 87 | 29 | Trent Scott | 65.45 | 50.29 | 71.39 | 827 | Chargers |
| 88 | 30 | Brandon Parker | 64.93 | 50.84 | 70.16 | 193 | Raiders |
| 89 | 31 | Cameron Erving | 64.92 | 48.25 | 71.87 | 589 | Chiefs |
| 90 | 32 | Josh Wells | 64.62 | 50.47 | 69.89 | 203 | Buccaneers |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 91 | 1 | J'Marcus Webb | 60.36 | 40.99 | 69.11 | 543 | Dolphins |
| 92 | 2 | Alex Light | 59.29 | 54.72 | 58.17 | 151 | Packers |

## TE — Tight End

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | George Kittle | 85.78 | 90.59 | 78.41 | 412 | 49ers |
| 2 | 2 | Travis Kelce | 84.35 | 85.10 | 79.68 | 666 | Chiefs |
| 3 | 3 | Mark Andrews | 81.41 | 81.14 | 77.42 | 311 | Ravens |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Darren Waller | 79.66 | 81.31 | 74.40 | 557 | Raiders |
| 5 | 2 | Will Dissly | 78.75 | 68.93 | 81.13 | 137 | Seahawks |
| 6 | 3 | Dallas Goedert | 77.62 | 79.01 | 72.53 | 443 | Eagles |
| 7 | 4 | Tyler Higbee | 77.16 | 80.37 | 70.86 | 402 | Rams |
| 8 | 5 | Hunter Henry | 75.36 | 71.69 | 73.64 | 432 | Chargers |
| 9 | 6 | Hayden Hurst | 75.33 | 68.07 | 76.00 | 216 | Ravens |
| 10 | 7 | Austin Hooper | 75.32 | 76.61 | 70.30 | 544 | Falcons |
| 11 | 8 | Zach Ertz | 74.59 | 73.03 | 71.46 | 600 | Eagles |
| 12 | 9 | Jaeden Graham | 74.56 | 62.42 | 78.48 | 133 | Falcons |
| 13 | 10 | Greg Olsen | 74.54 | 66.36 | 75.83 | 498 | Panthers |
| 14 | 11 | Maxx Williams | 74.51 | 71.18 | 72.56 | 226 | Cardinals |
| 15 | 12 | Jared Cook | 74.05 | 73.55 | 70.22 | 375 | Saints |

### Starter (60 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Mo Alie-Cox | 73.75 | 63.20 | 76.61 | 127 | Colts |
| 17 | 2 | Tyler Eifert | 73.52 | 64.97 | 75.06 | 408 | Bengals |
| 18 | 3 | Jack Doyle | 73.24 | 68.11 | 72.49 | 426 | Colts |
| 19 | 4 | Marcedes Lewis | 73.17 | 68.26 | 72.27 | 226 | Packers |
| 20 | 5 | Kyle Rudolph | 72.58 | 71.54 | 69.11 | 436 | Vikings |
| 21 | 6 | Blake Jarwin | 72.03 | 60.84 | 75.32 | 236 | Cowboys |
| 22 | 7 | Seth DeValve | 71.56 | 58.99 | 75.77 | 153 | Jaguars |
| 23 | 8 | Nick Boyle | 71.46 | 68.68 | 69.15 | 273 | Ravens |
| 24 | 9 | MyCole Pruitt | 71.38 | 60.64 | 74.38 | 122 | Titans |
| 25 | 10 | Gerald Everett | 71.22 | 70.75 | 67.36 | 298 | Rams |
| 26 | 11 | Evan Engram | 71.16 | 62.92 | 72.48 | 334 | Giants |
| 27 | 12 | J.P. Holtz | 70.74 | 63.01 | 71.73 | 126 | Bears |
| 28 | 13 | Ricky Seals-Jones | 70.60 | 58.28 | 74.64 | 191 | Browns |
| 29 | 14 | Eric Ebron | 70.58 | 66.60 | 69.07 | 233 | Colts |
| 30 | 15 | Nick O'Leary | 70.46 | 55.59 | 76.20 | 182 | Jaguars |
| 31 | 16 | Jonnu Smith | 69.89 | 68.42 | 66.71 | 325 | Titans |
| 32 | 17 | James O'Shaughnessy | 69.86 | 61.11 | 71.52 | 130 | Jaguars |
| 33 | 18 | Jimmy Graham | 69.82 | 55.50 | 75.20 | 458 | Packers |
| 34 | 19 | Rhett Ellison | 69.73 | 62.93 | 70.10 | 181 | Giants |
| 35 | 20 | Cameron Brate | 69.68 | 65.12 | 68.56 | 313 | Buccaneers |
| 36 | 21 | Jacob Hollister | 69.33 | 65.29 | 67.85 | 347 | Seahawks |
| 37 | 22 | O.J. Howard | 69.26 | 55.52 | 74.26 | 471 | Buccaneers |
| 38 | 23 | Luke Willson | 69.22 | 60.79 | 70.68 | 113 | Seahawks |
| 39 | 24 | Foster Moreau | 69.16 | 65.56 | 67.40 | 175 | Raiders |
| 40 | 25 | Ryan Izzo | 69.14 | 54.03 | 75.05 | 133 | Patriots |
| 41 | 26 | Josh Hill | 69.11 | 65.20 | 67.55 | 317 | Saints |
| 42 | 27 | Jeff Heuerman | 68.79 | 60.00 | 70.48 | 194 | Broncos |
| 43 | 28 | Mike Gesicki | 68.54 | 60.47 | 69.76 | 571 | Dolphins |
| 44 | 29 | Delanie Walker | 68.42 | 62.77 | 68.02 | 156 | Titans |
| 45 | 30 | Darren Fells | 68.32 | 60.08 | 69.64 | 402 | Texans |
| 46 | 31 | Charles Clay | 68.23 | 63.40 | 67.28 | 192 | Cardinals |
| 47 | 32 | Ryan Griffin | 67.81 | 61.49 | 67.85 | 404 | Jets |
| 48 | 33 | Chris Manhertz | 67.71 | 60.97 | 68.04 | 140 | Panthers |
| 49 | 34 | Jason Witten | 67.61 | 59.44 | 68.89 | 522 | Cowboys |
| 50 | 35 | Tyler Conklin | 67.21 | 57.48 | 69.53 | 104 | Vikings |
| 51 | 36 | Irv Smith Jr. | 67.19 | 63.59 | 65.42 | 341 | Vikings |
| 52 | 37 | Vernon Davis | 67.06 | 52.75 | 72.43 | 148 | Commanders |
| 53 | 38 | Jordan Akins | 67.03 | 56.12 | 70.14 | 414 | Texans |
| 54 | 39 | T.J. Hockenson | 66.99 | 60.36 | 67.25 | 344 | Lions |
| 55 | 40 | Tyler Kroft | 66.64 | 57.47 | 68.59 | 150 | Bills |
| 56 | 41 | Virgil Green | 66.60 | 58.90 | 67.56 | 200 | Chargers |
| 57 | 42 | Lee Smith | 66.39 | 60.35 | 66.25 | 124 | Bills |
| 58 | 43 | Benjamin Watson | 66.34 | 54.95 | 69.77 | 278 | Patriots |
| 59 | 44 | Ben Koyack | 66.25 | 58.28 | 67.39 | 105 | Jaguars |
| 60 | 45 | Dawson Knox | 66.07 | 60.00 | 65.95 | 377 | Bills |
| 61 | 46 | Noah Fant | 65.70 | 53.53 | 69.65 | 432 | Broncos |
| 62 | 47 | Ian Thomas | 65.54 | 55.42 | 68.12 | 206 | Panthers |
| 63 | 48 | Luke Stocker | 64.63 | 61.64 | 62.45 | 197 | Falcons |
| 64 | 49 | Matt LaCosse | 64.61 | 56.66 | 65.74 | 204 | Patriots |
| 65 | 50 | Logan Thomas | 64.40 | 57.30 | 64.97 | 201 | Lions |
| 66 | 51 | Blake Bell | 63.88 | 54.37 | 66.06 | 209 | Chiefs |
| 67 | 52 | Kaden Smith | 63.78 | 56.47 | 64.49 | 272 | Giants |
| 68 | 53 | Trey Burton | 63.71 | 54.04 | 65.99 | 201 | Bears |
| 69 | 54 | Durham Smythe | 63.46 | 55.52 | 64.59 | 249 | Dolphins |
| 70 | 55 | Vance McDonald | 63.19 | 47.88 | 69.23 | 449 | Steelers |
| 71 | 56 | Jesse James | 63.16 | 55.99 | 63.78 | 268 | Lions |
| 72 | 57 | Daniel Brown | 62.85 | 55.04 | 63.89 | 139 | Jets |
| 73 | 58 | Jeremy Sprinkle | 62.85 | 52.32 | 65.70 | 374 | Commanders |
| 74 | 59 | Geoff Swaim | 62.63 | 54.34 | 63.99 | 116 | Jaguars |
| 75 | 60 | Ross Dwelley | 62.53 | 58.13 | 61.29 | 152 | 49ers |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 76 | 1 | Nick Vannett | 61.84 | 53.20 | 63.44 | 219 | Steelers |
| 77 | 2 | C.J. Uzomah | 61.74 | 52.54 | 63.70 | 327 | Bengals |
| 78 | 3 | Demetrius Harris | 61.17 | 50.34 | 64.22 | 319 | Browns |

## WR — Wide Receiver

- **Season used:** `2019`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Chris Godwin | 88.46 | 90.32 | 83.05 | 633 | Buccaneers |
| 2 | 2 | Julio Jones | 87.63 | 89.89 | 81.95 | 610 | Falcons |
| 3 | 3 | Michael Thomas | 86.50 | 90.11 | 79.93 | 639 | Saints |
| 4 | 4 | A.J. Brown | 86.40 | 79.05 | 87.14 | 416 | Titans |
| 5 | 5 | Tyreek Hill | 85.08 | 81.82 | 83.08 | 381 | Chiefs |
| 6 | 6 | Mike Evans | 84.32 | 84.21 | 80.22 | 539 | Buccaneers |
| 7 | 7 | DeAndre Hopkins | 84.16 | 86.81 | 78.22 | 621 | Texans |
| 8 | 8 | Robert Woods | 83.22 | 82.15 | 79.77 | 655 | Rams |
| 9 | 9 | Kenny Golladay | 82.96 | 79.59 | 81.04 | 626 | Lions |
| 10 | 10 | Courtland Sutton | 81.97 | 81.58 | 78.06 | 576 | Broncos |
| 11 | 11 | DJ Moore | 81.95 | 81.29 | 78.22 | 607 | Panthers |
| 12 | 12 | Amari Cooper | 81.86 | 82.92 | 76.98 | 549 | Cowboys |
| 13 | 13 | Terry McLaurin | 81.72 | 82.03 | 77.34 | 485 | Commanders |
| 14 | 14 | Keenan Allen | 81.06 | 80.64 | 77.18 | 643 | Chargers |
| 15 | 15 | Davante Adams | 80.78 | 81.12 | 76.39 | 456 | Packers |
| 16 | 16 | Stefon Diggs | 80.76 | 76.87 | 79.19 | 453 | Vikings |
| 17 | 17 | DeVante Parker | 80.69 | 79.20 | 77.51 | 675 | Dolphins |

### Good (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | T.Y. Hilton | 79.76 | 72.13 | 80.68 | 298 | Colts |
| 19 | 2 | Allen Robinson II | 79.70 | 80.80 | 74.80 | 670 | Bears |
| 20 | 3 | Tyler Lockett | 79.56 | 76.74 | 77.27 | 615 | Seahawks |
| 21 | 4 | Cooper Kupp | 79.44 | 74.32 | 78.69 | 619 | Rams |
| 22 | 5 | Mecole Hardman Jr. | 79.37 | 66.43 | 83.83 | 330 | Chiefs |
| 23 | 6 | Mike Williams | 79.08 | 73.48 | 78.64 | 567 | Chargers |
| 24 | 7 | Andy Isabella | 79.02 | 61.68 | 86.42 | 102 | Cardinals |
| 25 | 8 | Emmanuel Sanders | 78.84 | 77.31 | 75.70 | 526 | 49ers |
| 26 | 9 | Jarvis Landry | 78.67 | 77.89 | 75.03 | 623 | Browns |
| 27 | 10 | John Brown | 78.21 | 75.15 | 76.09 | 572 | Bills |
| 28 | 11 | DJ Chark Jr. | 78.18 | 74.96 | 76.16 | 639 | Jaguars |
| 29 | 12 | Adam Thielen | 77.96 | 72.16 | 77.66 | 244 | Vikings |
| 30 | 13 | Breshad Perriman | 77.76 | 70.82 | 78.22 | 472 | Buccaneers |
| 31 | 14 | Michael Gallup | 77.50 | 72.68 | 76.54 | 541 | Cowboys |
| 32 | 15 | Will Fuller V | 76.96 | 71.23 | 76.62 | 351 | Texans |
| 33 | 16 | Deebo Samuel | 76.57 | 71.55 | 75.75 | 419 | 49ers |
| 34 | 17 | Golden Tate | 76.55 | 72.56 | 75.04 | 453 | Giants |
| 35 | 18 | Calvin Ridley | 76.49 | 73.64 | 74.23 | 553 | Falcons |
| 36 | 19 | Darius Slayton | 76.25 | 68.99 | 76.92 | 503 | Giants |
| 37 | 20 | Odell Beckham Jr. | 76.17 | 68.57 | 77.07 | 628 | Browns |
| 38 | 21 | Zach Pascal | 76.15 | 71.29 | 75.22 | 448 | Colts |
| 39 | 22 | Tyler Boyd | 76.11 | 73.00 | 74.01 | 675 | Bengals |
| 40 | 23 | Hunter Renfrow | 76.11 | 70.50 | 75.69 | 303 | Raiders |
| 41 | 24 | Marvin Jones Jr. | 76.06 | 72.19 | 74.47 | 545 | Lions |
| 42 | 25 | Alshon Jeffery | 75.53 | 71.96 | 73.75 | 305 | Eagles |
| 43 | 26 | David Moore | 75.45 | 63.90 | 78.99 | 205 | Seahawks |
| 44 | 27 | Brandin Cooks | 75.42 | 67.39 | 76.60 | 482 | Rams |
| 45 | 28 | D.K. Metcalf | 75.36 | 69.09 | 75.37 | 580 | Seahawks |
| 46 | 29 | Kenny Stills | 75.23 | 70.55 | 74.19 | 396 | Texans |
| 47 | 30 | Marquise Brown | 74.98 | 67.73 | 75.64 | 338 | Ravens |
| 48 | 31 | Corey Davis | 74.97 | 68.79 | 74.92 | 437 | Titans |
| 49 | 32 | Josh Gordon | 74.76 | 63.02 | 78.42 | 316 | Seahawks |
| 50 | 33 | Julian Edelman | 74.66 | 71.98 | 72.28 | 656 | Patriots |
| 51 | 34 | James Washington | 74.64 | 67.66 | 75.13 | 448 | Steelers |
| 52 | 35 | Allen Lazard | 74.34 | 68.55 | 74.04 | 319 | Packers |
| 53 | 36 | Sterling Shepard | 74.26 | 72.54 | 71.24 | 416 | Giants |
| 54 | 37 | Jamison Crowder | 74.25 | 72.40 | 71.32 | 562 | Jets |
| 55 | 38 | Sammy Watkins | 74.11 | 66.05 | 75.31 | 515 | Chiefs |
| 56 | 39 | Tyrell Williams | 74.11 | 65.74 | 75.52 | 428 | Raiders |
| 57 | 40 | Larry Fitzgerald | 74.08 | 70.10 | 72.57 | 619 | Cardinals |
| 58 | 41 | Robert Foster | 74.05 | 56.60 | 81.52 | 143 | Bills |

### Starter (84 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 59 | 1 | Cole Beasley | 73.88 | 72.54 | 70.61 | 499 | Bills |
| 60 | 2 | Scott Miller | 73.77 | 61.57 | 77.73 | 149 | Buccaneers |
| 61 | 3 | JuJu Smith-Schuster | 73.65 | 62.70 | 76.79 | 406 | Steelers |
| 62 | 4 | Randall Cobb | 73.60 | 69.12 | 72.42 | 499 | Cowboys |
| 63 | 5 | Jake Kumerow | 73.55 | 61.85 | 77.19 | 196 | Packers |
| 64 | 6 | Malik Turner | 73.47 | 63.72 | 75.81 | 143 | Seahawks |
| 65 | 7 | Richie James | 73.41 | 60.96 | 77.55 | 102 | 49ers |
| 66 | 8 | Diontae Johnson | 73.38 | 66.54 | 73.78 | 453 | Steelers |
| 67 | 9 | Danny Amendola | 73.31 | 70.46 | 71.05 | 476 | Lions |
| 68 | 10 | Auden Tate | 73.20 | 68.54 | 72.14 | 437 | Bengals |
| 69 | 11 | Keelan Cole Sr. | 73.18 | 65.12 | 74.39 | 275 | Jaguars |
| 70 | 12 | Josh Reynolds | 73.17 | 63.91 | 75.18 | 290 | Rams |
| 71 | 13 | Tajae Sharpe | 72.59 | 68.70 | 71.01 | 248 | Titans |
| 72 | 14 | Marquise Goodwin | 72.54 | 59.73 | 76.91 | 142 | 49ers |
| 73 | 15 | Adam Humphries | 72.18 | 67.63 | 71.05 | 285 | Titans |
| 74 | 16 | Preston Williams | 71.91 | 65.09 | 72.29 | 288 | Dolphins |
| 75 | 17 | Damion Ratley | 71.81 | 61.48 | 74.53 | 173 | Browns |
| 76 | 18 | Dante Pettis | 71.75 | 58.24 | 76.59 | 157 | 49ers |
| 77 | 19 | Cody Latimer | 71.74 | 65.35 | 71.83 | 268 | Giants |
| 78 | 20 | Phillip Dorsett | 71.72 | 65.28 | 71.84 | 387 | Patriots |
| 79 | 21 | Alex Erickson | 71.68 | 65.60 | 71.56 | 411 | Bengals |
| 80 | 22 | Tre'Quan Smith | 71.64 | 63.56 | 72.86 | 310 | Saints |
| 81 | 23 | Anthony Miller | 71.59 | 65.18 | 71.69 | 492 | Bears |
| 82 | 24 | Kelvin Harmon | 71.53 | 62.69 | 73.26 | 315 | Commanders |
| 83 | 25 | Marquez Valdes-Scantling | 71.49 | 57.79 | 76.46 | 357 | Packers |
| 84 | 26 | Tavon Austin | 71.47 | 59.95 | 74.98 | 165 | Cowboys |
| 85 | 27 | Jakobi Meyers | 70.90 | 63.10 | 71.93 | 288 | Patriots |
| 86 | 28 | Christian Kirk | 70.73 | 62.25 | 72.21 | 536 | Cardinals |
| 87 | 29 | Tim Patrick | 70.56 | 63.72 | 70.95 | 203 | Broncos |
| 88 | 30 | Greg Ward | 70.54 | 67.47 | 68.42 | 205 | Eagles |
| 89 | 31 | Demaryius Thomas | 70.48 | 64.79 | 70.11 | 311 | Jets |
| 90 | 32 | DeAndre Carter | 70.35 | 60.00 | 73.08 | 130 | Texans |
| 91 | 33 | Chris Conley | 70.34 | 62.64 | 71.30 | 629 | Jaguars |
| 92 | 34 | Isaiah Ford | 70.28 | 64.37 | 70.06 | 159 | Dolphins |
| 93 | 35 | Damiere Byrd | 70.20 | 63.09 | 70.77 | 325 | Cardinals |
| 94 | 36 | Kendrick Bourne | 70.09 | 64.52 | 69.63 | 301 | 49ers |
| 95 | 37 | Vyncint Smith | 70.04 | 61.26 | 71.73 | 183 | Jets |
| 96 | 38 | John Ross | 69.79 | 62.00 | 70.82 | 274 | Bengals |
| 97 | 39 | Jakeem Grant Sr. | 69.53 | 61.02 | 71.03 | 156 | Dolphins |
| 98 | 40 | Chris Hogan | 69.53 | 57.90 | 73.12 | 112 | Panthers |
| 99 | 41 | Isaiah McKenzie | 69.34 | 65.37 | 67.82 | 246 | Bills |
| 100 | 42 | Dede Westbrook | 69.23 | 63.32 | 69.00 | 595 | Jaguars |
| 101 | 43 | Albert Wilson | 69.21 | 61.90 | 69.91 | 330 | Dolphins |
| 102 | 44 | Taylor Gabriel | 69.11 | 61.63 | 69.93 | 314 | Bears |
| 103 | 45 | Pharoh Cooper | 69.07 | 65.15 | 67.52 | 165 | Cardinals |
| 104 | 46 | Jaron Brown | 69.04 | 58.37 | 71.99 | 224 | Seahawks |
| 105 | 47 | Keelan Doss | 68.86 | 59.44 | 70.97 | 102 | Raiders |
| 106 | 48 | Justin Hardy | 68.80 | 63.07 | 68.46 | 117 | Falcons |
| 107 | 49 | Ted Ginn Jr. | 68.62 | 57.10 | 72.14 | 467 | Saints |
| 108 | 50 | Russell Gage | 68.51 | 64.92 | 66.73 | 402 | Falcons |
| 109 | 51 | Seth Roberts | 68.48 | 62.39 | 68.38 | 307 | Ravens |
| 110 | 52 | Trevor Davis | 68.36 | 55.55 | 72.74 | 114 | Dolphins |
| 111 | 53 | Willie Snead IV | 68.25 | 60.82 | 69.03 | 326 | Ravens |
| 112 | 54 | Cordarrelle Patterson | 68.13 | 57.63 | 70.97 | 132 | Bears |
| 113 | 55 | Allen Hurns | 68.12 | 57.39 | 71.10 | 405 | Dolphins |
| 114 | 56 | Rashard Higgins | 68.11 | 57.35 | 71.12 | 113 | Browns |
| 115 | 57 | Marcus Johnson | 67.91 | 59.88 | 69.09 | 249 | Colts |
| 116 | 58 | Miles Boykin | 67.79 | 60.49 | 68.49 | 192 | Ravens |
| 117 | 59 | Mohamed Sanu | 67.75 | 58.87 | 69.50 | 533 | Patriots |
| 118 | 60 | Olabisi Johnson | 67.68 | 64.20 | 65.84 | 303 | Vikings |
| 119 | 61 | KhaDarel Hodge | 67.67 | 57.72 | 70.13 | 103 | Browns |
| 120 | 62 | Demarcus Robinson | 67.64 | 58.03 | 69.88 | 529 | Chiefs |
| 121 | 63 | Curtis Samuel | 67.59 | 62.70 | 66.68 | 668 | Panthers |
| 122 | 64 | Steven Sims | 67.48 | 62.12 | 66.89 | 229 | Commanders |
| 123 | 65 | Dontrelle Inman | 67.23 | 61.95 | 66.58 | 136 | Colts |
| 124 | 66 | J.J. Arcega-Whiteside | 66.97 | 55.66 | 70.34 | 323 | Eagles |
| 125 | 67 | Paul Richardson Jr. | 66.94 | 59.80 | 67.53 | 280 | Commanders |
| 126 | 68 | Jordan Matthews | 66.35 | 55.39 | 69.49 | 102 | 49ers |
| 127 | 69 | Travis Benjamin | 66.26 | 53.60 | 70.54 | 146 | Chargers |
| 128 | 70 | Keke Coutee | 66.21 | 55.82 | 68.97 | 242 | Texans |
| 129 | 71 | Nelson Agholor | 66.20 | 55.03 | 69.48 | 437 | Eagles |
| 130 | 72 | Johnny Holton | 65.95 | 54.93 | 69.13 | 107 | Steelers |
| 131 | 73 | Justin Watson | 65.03 | 61.88 | 62.97 | 153 | Buccaneers |
| 132 | 74 | Parris Campbell | 64.99 | 57.70 | 65.69 | 124 | Colts |
| 133 | 75 | N'Keal Harry | 64.89 | 63.00 | 61.98 | 136 | Patriots |
| 134 | 76 | Deon Cain | 64.70 | 57.51 | 65.33 | 211 | Steelers |
| 135 | 77 | Trent Sherfield | 64.32 | 56.27 | 65.52 | 151 | Cardinals |
| 136 | 78 | Zay Jones | 64.30 | 55.13 | 66.25 | 394 | Raiders |
| 137 | 79 | Mack Hollins | 63.95 | 53.63 | 66.67 | 243 | Dolphins |
| 138 | 80 | DaeSean Hamilton | 63.68 | 57.14 | 63.87 | 418 | Broncos |
| 139 | 81 | Christian Blake | 62.85 | 54.84 | 64.03 | 232 | Falcons |
| 140 | 82 | Trey Quinn | 62.70 | 55.92 | 63.06 | 315 | Commanders |
| 141 | 83 | Chester Rogers | 62.54 | 54.25 | 63.90 | 242 | Colts |
| 142 | 84 | Geronimo Allison | 62.47 | 55.09 | 63.23 | 458 | Packers |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 143 | 1 | Javon Wims | 61.79 | 54.56 | 62.44 | 305 | Bears |
| 144 | 2 | KeeSean Johnson | 61.48 | 55.95 | 61.00 | 256 | Cardinals |
| 145 | 3 | Jarius Wright | 60.67 | 49.09 | 64.22 | 528 | Panthers |
| 146 | 4 | Damion Willis | 59.52 | 56.22 | 57.55 | 155 | Bengals |
| 147 | 5 | Andre Patton | 57.83 | 49.83 | 59.00 | 334 | Chargers |
