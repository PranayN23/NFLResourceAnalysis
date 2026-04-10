# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:37:56Z
- **Requested analysis_year:** 2020 (clamped to 2020)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Corey Linsley | 94.53 | 87.45 | 95.09 | 734 | Packers |
| 2 | 2 | Brandon Linder | 86.90 | 76.80 | 89.47 | 530 | Jaguars |
| 3 | 3 | Ben Jones | 86.71 | 78.60 | 87.95 | 1042 | Titans |
| 4 | 4 | Cody Whitehair | 85.44 | 75.83 | 87.68 | 893 | Bears |
| 5 | 5 | J.C. Tretter | 85.41 | 77.30 | 86.65 | 1061 | Browns |
| 6 | 6 | Frank Ragnow | 85.01 | 79.71 | 84.38 | 929 | Lions |
| 7 | 7 | Chase Roullier | 84.98 | 76.80 | 86.26 | 1089 | Commanders |
| 8 | 8 | Rodney Hudson | 82.66 | 73.60 | 84.53 | 1082 | Raiders |
| 9 | 9 | Jason Kelce | 82.00 | 69.60 | 86.10 | 1126 | Eagles |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Erik McCoy | 79.82 | 70.10 | 82.14 | 1073 | Saints |
| 11 | 2 | Austin Reiter | 79.62 | 70.23 | 81.72 | 867 | Chiefs |
| 12 | 3 | Ryan Kelly | 78.80 | 69.00 | 81.16 | 1007 | Colts |
| 13 | 4 | Ryan Jensen | 77.97 | 64.90 | 82.51 | 1061 | Buccaneers |
| 14 | 5 | David Andrews | 77.62 | 67.04 | 80.50 | 724 | Patriots |
| 15 | 6 | Alex Mack | 77.14 | 65.88 | 80.48 | 972 | Falcons |
| 16 | 7 | Patrick Mekari | 77.11 | 65.17 | 80.91 | 554 | Ravens |
| 17 | 8 | Mitch Morse | 76.43 | 65.61 | 79.47 | 880 | Bills |
| 18 | 9 | Daniel Kilgore | 76.28 | 64.21 | 80.16 | 236 | Chiefs |
| 19 | 10 | Ted Karras | 76.26 | 65.40 | 79.34 | 1068 | Dolphins |
| 20 | 11 | Ben Garland | 76.22 | 66.45 | 78.56 | 333 | 49ers |
| 21 | 12 | Trey Hopkins | 75.16 | 63.71 | 78.63 | 938 | Bengals |
| 22 | 13 | Connor McGovern | 74.51 | 62.19 | 78.56 | 969 | Jets |
| 23 | 14 | Garrett Bradbury | 74.06 | 61.40 | 78.34 | 1082 | Vikings |

### Starter (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Matt Paradis | 73.89 | 63.40 | 76.71 | 1029 | Panthers |
| 25 | 2 | A.Q. Shipley | 73.40 | 58.40 | 79.23 | 157 | Buccaneers |
| 26 | 3 | Trystan Colon | 72.78 | 63.04 | 75.10 | 127 | Ravens |
| 27 | 4 | Billy Price | 71.80 | 55.73 | 78.34 | 208 | Bengals |
| 28 | 5 | Sam Mustipher | 71.65 | 59.21 | 75.78 | 504 | Bears |
| 29 | 6 | Maurkice Pouncey | 71.61 | 60.48 | 74.86 | 863 | Steelers |
| 30 | 7 | Mason Cole | 69.81 | 54.61 | 75.77 | 913 | Cardinals |
| 31 | 8 | Hroniss Grasu | 69.35 | 56.45 | 73.78 | 215 | 49ers |
| 32 | 9 | Tyler Biadasz | 69.09 | 55.72 | 73.84 | 427 | Cowboys |
| 33 | 10 | Nick Martin | 68.59 | 56.11 | 72.75 | 980 | Texans |
| 34 | 11 | Matt Hennessy | 67.59 | 53.79 | 72.62 | 225 | Falcons |
| 35 | 12 | Matt Skura | 67.22 | 51.97 | 73.22 | 661 | Ravens |
| 36 | 13 | J.C. Hassenauer | 67.22 | 58.72 | 68.72 | 303 | Steelers |
| 37 | 14 | James Ferentz | 65.54 | 57.69 | 66.61 | 162 | Patriots |
| 38 | 15 | Joe Looney | 65.21 | 51.81 | 69.97 | 764 | Cowboys |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Lloyd Cushenberry III | 59.43 | 40.50 | 67.89 | 1076 | Broncos |

## CB — Cornerback

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jaire Alexander | 91.91 | 90.60 | 89.76 | 900 | Packers |
| 2 | 2 | Xavien Howard | 90.37 | 89.60 | 90.98 | 936 | Dolphins |
| 3 | 3 | Bryce Callahan | 85.94 | 84.95 | 86.19 | 655 | Broncos |
| 4 | 4 | Darious Williams | 84.45 | 79.59 | 87.43 | 824 | Rams |
| 5 | 5 | Jamel Dean | 83.80 | 78.47 | 86.84 | 711 | Buccaneers |
| 6 | 6 | Jalen Ramsey | 83.55 | 80.30 | 83.31 | 954 | Rams |
| 7 | 7 | Tre'Davious White | 83.39 | 77.90 | 84.23 | 878 | Bills |
| 8 | 8 | Jonathan Jones | 82.80 | 79.57 | 81.11 | 730 | Patriots |
| 9 | 9 | James Bradberry | 82.67 | 79.90 | 81.19 | 1021 | Giants |
| 10 | 10 | Marlon Humphrey | 82.47 | 77.60 | 82.49 | 972 | Ravens |
| 11 | 11 | Xavier Rhodes | 82.46 | 77.50 | 82.33 | 902 | Colts |
| 12 | 12 | Kenny Moore II | 80.42 | 78.10 | 79.57 | 952 | Colts |
| 13 | 13 | Ronald Darby | 80.33 | 75.90 | 82.13 | 1002 | Commanders |
| 14 | 14 | Brian Poole | 80.02 | 76.75 | 83.36 | 483 | Jets |

### Good (20 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Malcolm Butler | 79.20 | 72.70 | 81.55 | 1087 | Titans |
| 16 | 2 | Troy Hill | 79.05 | 75.70 | 78.69 | 973 | Rams |
| 17 | 3 | Denzel Ward | 78.74 | 74.34 | 81.47 | 776 | Browns |
| 18 | 4 | Rashad Fenton | 78.72 | 71.99 | 83.48 | 527 | Chiefs |
| 19 | 5 | William Jackson III | 78.57 | 72.40 | 80.19 | 886 | Bengals |
| 20 | 6 | Ahkello Witherspoon | 78.20 | 76.34 | 81.43 | 334 | 49ers |
| 21 | 7 | J.C. Jackson | 77.53 | 70.10 | 78.95 | 851 | Patriots |
| 22 | 8 | Jason Verrett | 76.70 | 75.97 | 79.27 | 803 | 49ers |
| 23 | 9 | Kyle Fuller | 76.58 | 70.10 | 76.74 | 1060 | Bears |
| 24 | 10 | Cameron Sutton | 76.52 | 71.37 | 76.82 | 552 | Steelers |
| 25 | 11 | Joe Haden | 76.49 | 69.60 | 78.16 | 846 | Steelers |
| 26 | 12 | Marcus Peters | 76.37 | 69.40 | 77.88 | 912 | Ravens |
| 27 | 13 | Jimmy Smith | 75.97 | 73.27 | 79.24 | 454 | Ravens |
| 28 | 14 | Bashaud Breeland | 75.94 | 72.24 | 78.72 | 690 | Chiefs |
| 29 | 15 | Bradley Roby | 75.72 | 71.19 | 79.78 | 613 | Texans |
| 30 | 16 | Ross Cockrell | 74.82 | 70.26 | 79.33 | 238 | Buccaneers |
| 31 | 17 | L'Jarius Sneed | 74.54 | 69.66 | 80.92 | 410 | Chiefs |
| 32 | 18 | Donte Jackson | 74.36 | 67.67 | 76.94 | 599 | Panthers |
| 33 | 19 | Steven Nelson | 74.11 | 68.10 | 74.79 | 908 | Steelers |
| 34 | 20 | Darius Phillips | 74.10 | 66.87 | 81.00 | 593 | Bengals |

### Starter (66 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 35 | 1 | D.J. Reed | 73.54 | 68.07 | 78.53 | 560 | Seahawks |
| 36 | 2 | Charvarius Ward | 73.50 | 66.62 | 77.47 | 782 | Chiefs |
| 37 | 3 | Janoris Jenkins | 73.37 | 65.96 | 76.02 | 805 | Saints |
| 38 | 4 | Carlton Davis III | 73.34 | 66.40 | 76.08 | 906 | Buccaneers |
| 39 | 5 | T.J. Carrie | 72.91 | 65.71 | 74.90 | 396 | Colts |
| 40 | 6 | Trevon Diggs | 72.75 | 63.74 | 78.75 | 758 | Cowboys |
| 41 | 7 | Josh Norman | 72.66 | 67.80 | 77.46 | 344 | Bills |
| 42 | 8 | Sidney Jones IV | 72.58 | 66.85 | 80.25 | 303 | Jaguars |
| 43 | 9 | Mackensie Alexander | 72.43 | 66.69 | 74.79 | 642 | Bengals |
| 44 | 10 | Shaquill Griffin | 72.38 | 63.58 | 76.78 | 812 | Seahawks |
| 45 | 11 | Kendall Fuller | 72.37 | 65.30 | 75.74 | 893 | Commanders |
| 46 | 12 | Mike Hilton | 72.33 | 65.95 | 74.70 | 464 | Steelers |
| 47 | 13 | Michael Davis | 72.28 | 62.90 | 76.25 | 958 | Chargers |
| 48 | 14 | Darius Slay | 72.26 | 63.10 | 75.55 | 885 | Eagles |
| 49 | 15 | Richard Sherman | 71.74 | 65.62 | 78.10 | 332 | 49ers |
| 50 | 16 | Terrance Mitchell | 71.49 | 63.40 | 76.89 | 1070 | Browns |
| 51 | 17 | Chandon Sullivan | 70.77 | 65.08 | 71.96 | 729 | Packers |
| 52 | 18 | Cameron Dantzler | 70.64 | 68.36 | 73.20 | 601 | Vikings |
| 53 | 19 | Ugo Amadi | 70.62 | 65.81 | 75.26 | 552 | Seahawks |
| 54 | 20 | Byron Murphy Jr. | 70.42 | 63.24 | 71.69 | 795 | Cardinals |
| 55 | 21 | Casey Hayward Jr. | 70.24 | 59.51 | 74.26 | 788 | Chargers |
| 56 | 22 | Breon Borders | 70.16 | 67.53 | 76.08 | 360 | Titans |
| 57 | 23 | Rasul Douglas | 69.49 | 60.20 | 72.99 | 821 | Panthers |
| 58 | 24 | Johnathan Joseph | 69.46 | 62.57 | 73.53 | 423 | Cardinals |
| 59 | 25 | Fabian Moreau | 69.12 | 61.88 | 73.12 | 158 | Commanders |
| 60 | 26 | Trayvon Mullen | 69.04 | 58.30 | 72.03 | 933 | Raiders |
| 61 | 27 | Byron Jones | 68.97 | 61.49 | 71.14 | 814 | Dolphins |
| 62 | 28 | K'Waun Williams | 68.50 | 63.61 | 72.49 | 284 | 49ers |
| 63 | 29 | Blidi Wreh-Wilson | 68.29 | 61.74 | 74.00 | 245 | Falcons |
| 64 | 30 | Stephon Gilmore | 68.04 | 58.61 | 72.76 | 632 | Patriots |
| 65 | 31 | Adoree' Jackson | 68.01 | 64.49 | 74.53 | 155 | Titans |
| 66 | 32 | Corn Elder | 67.81 | 65.79 | 66.65 | 411 | Panthers |
| 67 | 33 | Desmond King II | 67.43 | 60.83 | 68.70 | 709 | Titans |
| 68 | 34 | Dontae Johnson | 67.21 | 62.99 | 72.84 | 273 | 49ers |
| 69 | 35 | Jaylon Johnson | 67.06 | 56.10 | 73.33 | 867 | Bears |
| 70 | 36 | Marshon Lattimore | 66.96 | 53.70 | 73.30 | 871 | Saints |
| 71 | 37 | Javelin Guidry | 66.91 | 67.53 | 73.20 | 172 | Jets |
| 72 | 38 | Levi Wallace | 66.66 | 56.33 | 73.35 | 612 | Bills |
| 73 | 39 | Darqueze Dennard | 66.28 | 62.60 | 71.55 | 439 | Falcons |
| 74 | 40 | Patrick Robinson | 66.10 | 57.74 | 74.38 | 248 | Saints |
| 75 | 41 | Joejuan Williams | 65.98 | 57.12 | 74.89 | 172 | Patriots |
| 76 | 42 | Emmanuel Moseley | 65.94 | 55.80 | 70.62 | 499 | 49ers |
| 77 | 43 | Darryl Roberts | 65.76 | 56.23 | 72.31 | 469 | Lions |
| 78 | 44 | Harrison Hand | 65.73 | 61.07 | 78.08 | 163 | Vikings |
| 79 | 45 | Jason McCourty | 65.61 | 52.30 | 71.57 | 665 | Patriots |
| 80 | 46 | Chris Harris Jr. | 65.08 | 57.67 | 70.34 | 568 | Chargers |
| 81 | 47 | CJ Henderson | 65.03 | 58.71 | 73.41 | 474 | Jaguars |
| 82 | 48 | Isaiah Oliver | 64.96 | 53.30 | 69.60 | 831 | Falcons |
| 83 | 49 | A.J. Terrell | 64.84 | 57.00 | 67.98 | 908 | Falcons |
| 84 | 50 | Bryce Hall | 64.80 | 62.28 | 70.64 | 547 | Jets |
| 85 | 51 | Taron Johnson | 64.36 | 55.60 | 68.32 | 825 | Bills |
| 86 | 52 | Keion Crossen | 64.35 | 60.92 | 70.90 | 307 | Texans |
| 87 | 53 | Anthony Averett | 64.35 | 60.39 | 70.00 | 355 | Ravens |
| 88 | 54 | Tye Smith | 64.20 | 61.86 | 69.20 | 169 | Titans |
| 89 | 55 | Patrick Peterson | 64.11 | 53.10 | 69.17 | 1096 | Cardinals |
| 90 | 56 | Jamal Perry | 63.79 | 57.36 | 67.95 | 140 | Dolphins |
| 91 | 57 | Josh Jackson | 63.66 | 54.74 | 69.09 | 331 | Packers |
| 92 | 58 | Sean Murphy-Bunting | 63.60 | 55.20 | 65.41 | 884 | Buccaneers |
| 93 | 59 | Jimmy Moreland | 63.46 | 57.18 | 64.26 | 601 | Commanders |
| 94 | 60 | Nickell Robey-Coleman | 63.45 | 53.86 | 66.19 | 612 | Eagles |
| 95 | 61 | Nevin Lawson | 63.37 | 54.99 | 67.92 | 737 | Raiders |
| 96 | 62 | Chidobe Awuzie | 62.78 | 54.08 | 68.78 | 452 | Cowboys |
| 97 | 63 | A.J. Bouye | 62.60 | 53.75 | 70.26 | 410 | Broncos |
| 98 | 64 | Rashard Robinson | 62.54 | 58.05 | 72.62 | 187 | Cowboys |
| 99 | 65 | Mike Hughes | 62.26 | 58.49 | 69.58 | 173 | Vikings |
| 100 | 66 | Tre Herndon | 62.18 | 50.70 | 68.48 | 1017 | Jaguars |

### Rotation/backup (59 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 101 | 1 | Kevin King | 61.88 | 52.91 | 68.69 | 664 | Packers |
| 102 | 2 | Jamar Taylor | 61.72 | 55.05 | 69.09 | 203 | 49ers |
| 103 | 3 | Rock Ya-Sin | 61.55 | 50.37 | 67.18 | 550 | Colts |
| 104 | 4 | Tre Flowers | 61.00 | 52.47 | 65.65 | 578 | Seahawks |
| 105 | 5 | Chris Claybrooks | 60.98 | 58.25 | 62.80 | 375 | Jaguars |
| 106 | 6 | Tevaughn Campbell | 60.61 | 53.90 | 67.16 | 326 | Chargers |
| 107 | 7 | Myles Hartsfield | 60.54 | 59.92 | 64.09 | 140 | Panthers |
| 108 | 8 | Isaac Yiadom | 60.53 | 53.07 | 64.88 | 634 | Giants |
| 109 | 9 | Justin Coleman | 60.39 | 50.11 | 65.68 | 470 | Lions |
| 110 | 10 | Ka'dar Hollman | 60.38 | 57.96 | 68.90 | 108 | Packers |
| 111 | 11 | Buster Skrine | 60.32 | 48.66 | 66.42 | 557 | Bears |
| 112 | 12 | Anthony Brown | 60.29 | 50.59 | 68.10 | 534 | Cowboys |
| 113 | 13 | Amani Oruwariye | 60.24 | 51.20 | 65.61 | 1028 | Lions |
| 114 | 14 | Antonio Hamilton Sr. | 60.21 | 54.92 | 68.68 | 136 | Chiefs |
| 115 | 15 | De'Vante Bausby | 60.19 | 56.52 | 70.06 | 277 | Broncos |
| 116 | 16 | Daryl Worley | 60.09 | 48.47 | 68.36 | 346 | Raiders |
| 117 | 17 | Kevin Johnson | 60.08 | 53.91 | 65.55 | 575 | Browns |
| 118 | 18 | Duke Shelley | 59.86 | 59.40 | 66.86 | 209 | Bears |
| 119 | 19 | Phillip Gaines | 59.79 | 56.34 | 66.78 | 262 | Texans |
| 120 | 20 | Nik Needham | 59.66 | 47.29 | 66.60 | 617 | Dolphins |
| 121 | 21 | Jourdan Lewis | 59.62 | 45.87 | 66.28 | 817 | Cowboys |
| 122 | 22 | Darnay Holmes | 59.13 | 51.80 | 64.02 | 442 | Giants |
| 123 | 23 | Pierre Desir | 59.05 | 44.67 | 69.05 | 519 | Ravens |
| 124 | 24 | Chris Jones | 58.96 | 52.41 | 66.06 | 273 | Vikings |
| 125 | 25 | Quinton Dunbar | 58.85 | 47.10 | 71.17 | 397 | Seahawks |
| 126 | 26 | David Long Jr. | 58.46 | 55.80 | 62.31 | 116 | Rams |
| 127 | 27 | Isaiah Johnson | 58.15 | 48.62 | 68.28 | 181 | Raiders |
| 128 | 28 | Dre Kirkpatrick | 58.10 | 46.75 | 66.28 | 750 | Cardinals |
| 129 | 29 | Kristian Fulton | 57.89 | 58.26 | 66.89 | 203 | Titans |
| 130 | 30 | M.J. Stewart | 57.72 | 51.62 | 64.39 | 229 | Browns |
| 131 | 31 | Davontae Harris | 57.41 | 54.60 | 63.55 | 117 | Ravens |
| 132 | 32 | Essang Bassey | 57.38 | 52.11 | 62.98 | 382 | Broncos |
| 133 | 33 | Cre'Von LeBlanc | 57.36 | 51.44 | 65.79 | 217 | Eagles |
| 134 | 34 | Greg Mabin | 56.84 | 49.80 | 68.52 | 248 | Jaguars |
| 135 | 35 | Kris Boyd | 56.81 | 53.75 | 63.02 | 343 | Vikings |
| 136 | 36 | John Reid | 56.39 | 58.66 | 61.57 | 145 | Texans |
| 137 | 37 | Tavierre Thomas | 56.23 | 55.57 | 63.18 | 204 | Browns |
| 138 | 38 | Blessuan Austin | 55.75 | 48.55 | 63.15 | 681 | Jets |
| 139 | 39 | Ryan Lewis | 55.50 | 50.14 | 65.42 | 271 | Giants |
| 140 | 40 | Desmond Trufant | 55.36 | 41.92 | 67.55 | 324 | Lions |
| 141 | 41 | Avonte Maddox | 55.15 | 42.01 | 65.38 | 509 | Eagles |
| 142 | 42 | Michael Ojemudia | 55.06 | 49.90 | 56.41 | 852 | Broncos |
| 143 | 43 | Kendall Sheffield | 54.54 | 41.59 | 62.39 | 524 | Falcons |
| 144 | 44 | Vernon Hargreaves III | 54.25 | 40.00 | 63.01 | 980 | Texans |
| 145 | 45 | LeShaun Sims | 54.09 | 42.93 | 61.95 | 607 | Bengals |
| 146 | 46 | Justin Layne | 53.60 | 54.92 | 57.92 | 120 | Steelers |
| 147 | 47 | Jeff Gladney | 53.51 | 48.50 | 52.68 | 958 | Vikings |
| 148 | 48 | D.J. Hayden | 52.25 | 40.28 | 63.36 | 234 | Jaguars |
| 149 | 49 | Keisean Nixon | 51.55 | 51.63 | 56.96 | 155 | Raiders |
| 150 | 50 | Troy Pride Jr. | 51.36 | 45.35 | 53.28 | 529 | Panthers |
| 151 | 51 | Lamar Jackson | 51.24 | 49.48 | 56.58 | 453 | Jets |
| 152 | 52 | Jeff Okudah | 50.63 | 40.00 | 60.85 | 460 | Lions |
| 153 | 53 | Kindle Vildor | 49.07 | 55.86 | 56.35 | 136 | Bears |
| 154 | 54 | Noah Igbinoghene | 48.87 | 47.52 | 51.85 | 286 | Dolphins |
| 155 | 55 | Michael Jacquet | 48.57 | 50.80 | 61.41 | 160 | Eagles |
| 156 | 56 | Damon Arnette | 48.49 | 45.49 | 53.62 | 343 | Raiders |
| 157 | 57 | Corey Ballentine | 46.99 | 49.76 | 51.91 | 107 | Jets |
| 158 | 58 | Chris Jackson | 46.98 | 44.38 | 51.85 | 241 | Titans |
| 159 | 59 | Luq Barcoo | 45.00 | 47.68 | 53.50 | 152 | Jaguars |

## DI — Defensive Interior

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 92.85 | 88.40 | 91.65 | 866 | Rams |
| 2 | 2 | Quinnen Williams | 86.22 | 85.27 | 85.82 | 587 | Jets |
| 3 | 3 | DeForest Buckner | 85.98 | 89.29 | 80.13 | 751 | Colts |
| 4 | 4 | Grady Jarrett | 85.86 | 85.78 | 82.16 | 851 | Falcons |
| 5 | 5 | Leonard Williams | 85.53 | 89.40 | 79.10 | 803 | Giants |
| 6 | 6 | Cameron Heyward | 83.73 | 83.16 | 80.46 | 807 | Steelers |
| 7 | 7 | Fletcher Cox | 82.96 | 84.29 | 78.43 | 747 | Eagles |
| 8 | 8 | Dexter Lawrence | 82.76 | 86.13 | 76.35 | 655 | Giants |
| 9 | 9 | Kenny Clark | 82.74 | 83.52 | 80.24 | 595 | Packers |
| 10 | 10 | Chris Jones | 82.19 | 86.21 | 76.81 | 695 | Chiefs |
| 11 | 11 | Calais Campbell | 81.84 | 73.31 | 85.45 | 410 | Ravens |
| 12 | 12 | Jonathan Allen | 81.39 | 79.61 | 78.73 | 809 | Commanders |
| 13 | 13 | Stephon Tuitt | 80.83 | 84.87 | 78.04 | 779 | Steelers |
| 14 | 14 | Dalvin Tomlinson | 80.46 | 80.64 | 76.18 | 658 | Giants |
| 15 | 15 | Poona Ford | 80.31 | 77.58 | 79.31 | 670 | Seahawks |
| 16 | 16 | Zach Sieler | 80.29 | 71.05 | 88.02 | 532 | Dolphins |
| 17 | 17 | Shelby Harris | 80.10 | 77.34 | 80.37 | 441 | Broncos |

### Good (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Mario Edwards Jr. | 79.97 | 77.59 | 78.64 | 256 | Bears |
| 19 | 2 | Sheldon Richardson | 79.23 | 73.00 | 79.21 | 799 | Browns |
| 20 | 3 | Folorunso Fatukasi | 79.14 | 80.18 | 78.54 | 507 | Jets |
| 21 | 4 | Sebastian Joseph-Day | 77.80 | 69.26 | 79.32 | 412 | Rams |
| 22 | 5 | Dre'Mont Jones | 77.57 | 75.68 | 77.39 | 560 | Broncos |
| 23 | 6 | B.J. Hill | 77.56 | 74.48 | 75.44 | 375 | Giants |
| 24 | 7 | Akiem Hicks | 77.54 | 71.58 | 81.32 | 795 | Bears |
| 25 | 8 | Jeffery Simmons | 77.40 | 84.53 | 71.86 | 841 | Titans |
| 26 | 9 | James Smith-Williams | 77.05 | 59.10 | 86.93 | 100 | Commanders |
| 27 | 10 | Daron Payne | 76.62 | 74.36 | 74.28 | 882 | Commanders |
| 28 | 11 | David Onyemata | 76.57 | 74.05 | 74.92 | 599 | Saints |
| 29 | 12 | Jurrell Casey | 76.23 | 73.77 | 81.30 | 156 | Broncos |
| 30 | 13 | Tim Settle | 76.11 | 67.34 | 78.52 | 348 | Commanders |
| 31 | 14 | Bilal Nichols | 75.80 | 66.66 | 79.07 | 618 | Bears |
| 32 | 15 | Vita Vea | 75.50 | 80.48 | 74.36 | 224 | Buccaneers |
| 33 | 16 | Javon Hargrave | 75.19 | 64.77 | 78.49 | 602 | Eagles |
| 34 | 17 | Linval Joseph | 75.06 | 67.56 | 77.05 | 726 | Chargers |
| 35 | 18 | Christian Wilkins | 74.92 | 71.47 | 74.36 | 637 | Dolphins |
| 36 | 19 | Ndamukong Suh | 74.77 | 62.56 | 78.75 | 788 | Buccaneers |
| 37 | 20 | Henry Anderson | 74.65 | 66.97 | 76.54 | 549 | Jets |
| 38 | 21 | Zach Kerr | 74.55 | 69.20 | 76.76 | 390 | Panthers |
| 39 | 22 | Lawrence Guy Sr. | 74.50 | 63.77 | 78.52 | 503 | Patriots |
| 40 | 23 | Malcom Brown | 74.28 | 68.08 | 76.02 | 345 | Saints |

### Starter (79 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 41 | 1 | DJ Reader | 73.44 | 74.06 | 74.91 | 259 | Bengals |
| 42 | 2 | Maurice Hurst | 73.37 | 73.07 | 72.63 | 277 | Raiders |
| 43 | 3 | Ed Oliver | 73.09 | 57.98 | 78.99 | 578 | Bills |
| 44 | 4 | Greg Gaines | 73.00 | 63.79 | 77.32 | 201 | Rams |
| 45 | 5 | Geno Atkins | 72.99 | 57.24 | 83.49 | 119 | Bengals |
| 46 | 6 | Derek Wolfe | 72.92 | 64.60 | 76.58 | 621 | Ravens |
| 47 | 7 | Michael Brockers | 72.48 | 63.82 | 74.61 | 625 | Rams |
| 48 | 8 | Damon Harrison Sr. | 72.46 | 60.13 | 81.51 | 150 | Packers |
| 49 | 9 | Steve McLendon | 72.43 | 57.71 | 78.59 | 443 | Buccaneers |
| 50 | 10 | Shy Tuttle | 72.07 | 65.77 | 74.05 | 326 | Saints |
| 51 | 11 | Derrick Brown | 71.98 | 65.84 | 71.91 | 742 | Panthers |
| 52 | 12 | Johnathan Hankins | 71.98 | 61.37 | 75.08 | 665 | Raiders |
| 53 | 13 | Tyson Alualu | 71.83 | 69.64 | 69.85 | 448 | Steelers |
| 54 | 14 | Brandon Williams | 71.56 | 62.71 | 75.47 | 354 | Ravens |
| 55 | 15 | DaQuan Jones | 71.50 | 67.67 | 69.89 | 706 | Titans |
| 56 | 16 | DeMarcus Walker | 71.32 | 58.60 | 81.79 | 384 | Broncos |
| 57 | 17 | Kawann Short | 71.22 | 60.43 | 85.81 | 123 | Panthers |
| 58 | 18 | Larry Ogunjobi | 71.19 | 55.34 | 78.43 | 642 | Browns |
| 59 | 19 | Malik Jackson | 71.11 | 57.66 | 81.11 | 537 | Eagles |
| 60 | 20 | Morgan Fox | 70.95 | 56.12 | 76.67 | 403 | Rams |
| 61 | 21 | Vincent Taylor | 70.81 | 57.38 | 81.85 | 207 | Browns |
| 62 | 22 | DeShawn Williams | 70.47 | 69.39 | 73.01 | 436 | Broncos |
| 63 | 23 | Taven Bryan | 70.32 | 64.86 | 69.79 | 511 | Jaguars |
| 64 | 24 | Grover Stewart | 70.09 | 61.84 | 71.63 | 581 | Colts |
| 65 | 25 | Brent Urban | 70.05 | 59.81 | 73.65 | 370 | Bears |
| 66 | 26 | John Cominsky | 69.79 | 56.56 | 77.57 | 399 | Falcons |
| 67 | 27 | Jarran Reed | 69.52 | 58.42 | 74.64 | 847 | Seahawks |
| 68 | 28 | Raekwon Davis | 69.39 | 69.31 | 65.28 | 538 | Dolphins |
| 69 | 29 | William Gholston | 69.22 | 54.58 | 74.81 | 606 | Buccaneers |
| 70 | 30 | Derrick Nnadi | 69.10 | 64.16 | 68.74 | 460 | Chiefs |
| 71 | 31 | Mike Pennel | 68.82 | 59.31 | 74.55 | 322 | Chiefs |
| 72 | 32 | Tershawn Wharton | 68.74 | 58.49 | 71.41 | 518 | Chiefs |
| 73 | 33 | Dean Lowry | 68.60 | 57.08 | 72.12 | 601 | Packers |
| 74 | 34 | Roy Robertson-Harris | 68.55 | 61.25 | 73.74 | 245 | Bears |
| 75 | 35 | Adam Butler | 68.52 | 54.16 | 74.44 | 481 | Patriots |
| 76 | 36 | Quinton Jefferson | 68.38 | 59.20 | 70.96 | 534 | Bills |
| 77 | 37 | Mike Daniels | 68.08 | 52.01 | 80.67 | 356 | Bengals |
| 78 | 38 | Kingsley Keke | 68.07 | 62.76 | 69.66 | 414 | Packers |
| 79 | 39 | Danny Shelton | 68.02 | 58.46 | 72.93 | 498 | Lions |
| 80 | 40 | D.J. Jones | 67.89 | 57.95 | 74.20 | 420 | 49ers |
| 81 | 41 | John Franklin-Myers | 67.74 | 58.91 | 70.50 | 500 | Jets |
| 82 | 42 | Mike Purcell | 67.45 | 58.06 | 76.11 | 218 | Broncos |
| 83 | 43 | Chris Wormley | 67.37 | 66.62 | 65.27 | 148 | Steelers |
| 84 | 44 | Sheldon Rankins | 67.36 | 59.81 | 72.19 | 415 | Saints |
| 85 | 45 | Vernon Butler | 67.21 | 56.98 | 71.95 | 428 | Bills |
| 86 | 46 | Tyler Lancaster | 67.15 | 58.70 | 70.18 | 352 | Packers |
| 87 | 47 | Hassan Ridgeway | 67.13 | 61.83 | 76.30 | 138 | Eagles |
| 88 | 48 | Christian Covington | 67.06 | 54.20 | 72.30 | 559 | Bengals |
| 89 | 49 | Armon Watts | 66.89 | 60.64 | 70.40 | 392 | Vikings |
| 90 | 50 | Abry Jones | 66.82 | 58.27 | 74.29 | 159 | Jaguars |
| 91 | 51 | Austin Johnson | 66.62 | 60.27 | 66.68 | 231 | Giants |
| 92 | 52 | John Jenkins | 66.45 | 59.81 | 71.19 | 223 | Bears |
| 93 | 53 | Justin Zimmer | 66.45 | 57.24 | 78.01 | 275 | Bills |
| 94 | 54 | Neville Gallimore | 66.25 | 51.81 | 73.80 | 416 | Cowboys |
| 95 | 55 | Javon Kinlaw | 65.77 | 53.67 | 71.75 | 547 | 49ers |
| 96 | 56 | Kevin Givens | 65.73 | 55.54 | 76.18 | 387 | 49ers |
| 97 | 57 | Justin Jones | 65.58 | 58.85 | 68.92 | 527 | Chargers |
| 98 | 58 | A'Shawn Robinson | 65.54 | 55.19 | 74.00 | 111 | Rams |
| 99 | 59 | Davon Godchaux | 65.46 | 56.48 | 73.02 | 172 | Dolphins |
| 100 | 60 | Jonathan Bullard | 65.05 | 57.90 | 73.05 | 117 | Seahawks |
| 101 | 61 | Jordan Phillips | 65.03 | 53.96 | 71.89 | 266 | Cardinals |
| 102 | 62 | P.J. Hall | 64.85 | 57.83 | 68.91 | 343 | Texans |
| 103 | 63 | Allen Bailey | 64.77 | 50.29 | 70.58 | 424 | Falcons |
| 104 | 64 | Isaiah Buggs | 64.61 | 55.87 | 72.90 | 131 | Steelers |
| 105 | 65 | L.J. Collier | 64.35 | 53.52 | 67.40 | 559 | Seahawks |
| 106 | 66 | Corey Peters | 64.23 | 54.47 | 70.42 | 379 | Cardinals |
| 107 | 67 | Xavier Williams | 64.21 | 51.24 | 74.21 | 320 | Bengals |
| 108 | 68 | Nathan Shepherd | 64.00 | 57.75 | 67.24 | 336 | Jets |
| 109 | 69 | Charles Omenihu | 63.95 | 54.94 | 67.23 | 546 | Texans |
| 110 | 70 | Harrison Phillips | 63.88 | 61.22 | 67.63 | 332 | Bills |
| 111 | 71 | Da'Shawn Hand | 63.75 | 58.03 | 71.21 | 353 | Lions |
| 112 | 72 | Montravius Adams | 63.48 | 56.58 | 68.69 | 130 | Packers |
| 113 | 73 | Leki Fotu | 63.27 | 51.43 | 72.19 | 284 | Cardinals |
| 114 | 74 | T.Y. McGill | 63.13 | 58.46 | 70.51 | 127 | Eagles |
| 115 | 75 | Tyeler Davison | 62.66 | 54.24 | 64.52 | 519 | Falcons |
| 116 | 76 | Tyrone Crawford | 62.66 | 48.84 | 71.67 | 445 | Cowboys |
| 117 | 77 | Carlos Watkins | 62.46 | 50.89 | 70.37 | 542 | Texans |
| 118 | 78 | Doug Costin | 62.42 | 61.66 | 62.92 | 456 | Jaguars |
| 119 | 79 | Damion Square | 62.18 | 53.18 | 64.02 | 253 | Chargers |

### Rotation/backup (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 120 | 1 | Ross Blacklock | 61.87 | 51.72 | 65.50 | 254 | Texans |
| 121 | 2 | Akeem Spence | 61.87 | 53.19 | 69.01 | 103 | Patriots |
| 122 | 3 | John Penisini | 61.59 | 46.47 | 67.51 | 576 | Lions |
| 123 | 4 | Kendal Vickers | 60.84 | 50.54 | 64.57 | 315 | Raiders |
| 124 | 5 | Brandon Dunn | 60.80 | 50.17 | 65.71 | 451 | Texans |
| 125 | 6 | Caraun Reid | 60.63 | 56.05 | 69.52 | 144 | Jaguars |
| 126 | 7 | Maliek Collins | 60.36 | 51.63 | 64.72 | 505 | Raiders |
| 127 | 8 | DaVon Hamilton | 60.10 | 53.09 | 65.81 | 408 | Jaguars |
| 128 | 9 | Angelo Blackson | 60.01 | 47.02 | 64.82 | 550 | Cardinals |
| 129 | 10 | Antwaun Woods | 59.88 | 51.90 | 64.17 | 457 | Cowboys |
| 130 | 11 | Bravvion Roy | 59.63 | 48.46 | 63.95 | 419 | Panthers |
| 131 | 12 | Rakeem Nunez-Roches | 59.53 | 49.09 | 65.02 | 483 | Buccaneers |
| 132 | 13 | Byron Cowart | 59.50 | 52.49 | 65.99 | 419 | Patriots |
| 133 | 14 | Domata Peko Sr. | 59.47 | 48.84 | 70.41 | 177 | Cardinals |
| 134 | 15 | Jaleel Johnson | 59.35 | 45.11 | 64.68 | 654 | Vikings |
| 135 | 16 | Margus Hunt | 59.35 | 43.38 | 67.60 | 387 | Bengals |
| 136 | 17 | Trevon Coley | 58.33 | 52.82 | 65.86 | 192 | Jets |
| 137 | 18 | Sylvester Williams | 58.28 | 52.04 | 66.29 | 173 | Broncos |
| 138 | 19 | Justin Ellis | 58.04 | 51.08 | 65.92 | 358 | Ravens |
| 139 | 20 | Taylor Stallworth | 57.84 | 52.64 | 61.31 | 253 | Colts |
| 140 | 21 | Shamar Stephen | 57.79 | 48.18 | 60.55 | 662 | Vikings |
| 141 | 22 | Matt Dickerson | 57.56 | 53.51 | 65.36 | 197 | Titans |
| 142 | 23 | Malcolm Roach | 57.44 | 51.42 | 64.58 | 233 | Saints |
| 143 | 24 | Trysten Hill | 57.28 | 53.56 | 66.27 | 212 | Cowboys |
| 144 | 25 | Rashard Lawrence | 56.71 | 52.34 | 62.76 | 166 | Cardinals |
| 145 | 26 | Kevin Strong | 56.49 | 54.36 | 63.37 | 209 | Lions |
| 146 | 27 | Eli Ankou | 55.75 | 51.99 | 63.89 | 186 | Cowboys |
| 147 | 28 | Bryan Mone | 55.00 | 57.46 | 57.79 | 228 | Seahawks |
| 148 | 29 | Rasheem Green | 54.97 | 48.24 | 61.54 | 365 | Seahawks |
| 149 | 30 | McTelvin Agim | 54.29 | 53.77 | 56.72 | 141 | Broncos |
| 150 | 31 | Teair Tart | 53.75 | 54.45 | 59.99 | 155 | Titans |
| 151 | 32 | Kentavius Street | 53.59 | 48.20 | 58.75 | 380 | 49ers |
| 152 | 33 | Daniel Ekuale | 53.36 | 52.51 | 57.82 | 290 | Jaguars |
| 153 | 34 | Jordan Elliott | 52.76 | 51.65 | 49.34 | 307 | Browns |
| 154 | 35 | Larrell Murchison | 51.85 | 52.46 | 53.52 | 136 | Titans |
| 155 | 36 | Justin Hamilton | 51.50 | 54.10 | 54.77 | 236 | Cowboys |
| 156 | 37 | Marlon Davidson | 50.99 | 57.06 | 51.11 | 132 | Falcons |
| 157 | 38 | Broderick Washington | 46.89 | 51.04 | 48.29 | 161 | Ravens |

## ED — Edge

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | T.J. Watt | 93.03 | 96.33 | 87.18 | 856 | Steelers |
| 2 | 2 | Joey Bosa | 89.85 | 93.07 | 87.51 | 549 | Chargers |
| 3 | 3 | Khalil Mack | 89.24 | 93.04 | 82.96 | 894 | Bears |
| 4 | 4 | Myles Garrett | 87.47 | 94.39 | 81.61 | 758 | Browns |
| 5 | 5 | DeMarcus Lawrence | 84.63 | 87.75 | 78.38 | 668 | Cowboys |
| 6 | 6 | Shaquil Barrett | 84.33 | 82.19 | 82.74 | 824 | Buccaneers |
| 7 | 7 | Brandon Graham | 82.25 | 80.17 | 79.47 | 759 | Eagles |
| 8 | 8 | Cameron Jordan | 81.78 | 86.01 | 74.80 | 816 | Saints |
| 9 | 9 | Montez Sweat | 81.75 | 82.04 | 77.39 | 693 | Commanders |
| 10 | 10 | Justin Houston | 81.47 | 69.87 | 85.87 | 608 | Colts |
| 11 | 11 | Za'Darius Smith | 80.37 | 79.44 | 76.82 | 858 | Packers |
| 12 | 12 | Brian Burns | 80.06 | 74.97 | 79.93 | 750 | Panthers |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Rashan Gary | 77.34 | 68.11 | 79.98 | 456 | Packers |
| 14 | 2 | Bradley Chubb | 77.32 | 71.15 | 82.05 | 741 | Broncos |
| 15 | 3 | Marcus Davenport | 77.12 | 81.97 | 73.89 | 374 | Saints |
| 16 | 4 | Uchenna Nwosu | 76.91 | 65.05 | 83.79 | 356 | Chargers |
| 17 | 5 | Chase Winovich | 75.91 | 65.36 | 78.78 | 593 | Patriots |
| 18 | 6 | Chase Young | 75.19 | 87.44 | 63.89 | 770 | Commanders |
| 19 | 7 | Trey Flowers | 74.55 | 73.78 | 76.10 | 309 | Lions |
| 20 | 8 | Olivier Vernon | 74.42 | 74.39 | 74.24 | 805 | Browns |
| 21 | 9 | Carlos Dunlap | 74.05 | 66.11 | 76.32 | 593 | Seahawks |
| 22 | 10 | Matthew Judon | 74.02 | 59.47 | 80.58 | 563 | Ravens |

### Starter (68 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Carl Lawson | 73.41 | 67.77 | 76.13 | 723 | Bengals |
| 24 | 2 | Andrew Van Ginkel | 73.40 | 67.19 | 73.37 | 479 | Dolphins |
| 25 | 3 | Trey Hendrickson | 73.36 | 67.03 | 77.17 | 558 | Saints |
| 26 | 4 | Ryan Kerrigan | 73.30 | 56.02 | 81.90 | 398 | Commanders |
| 27 | 5 | Genard Avery | 73.17 | 62.00 | 84.78 | 126 | Eagles |
| 28 | 6 | Markus Golden | 73.01 | 58.14 | 79.79 | 591 | Cardinals |
| 29 | 7 | Pernell McPhee | 72.65 | 63.08 | 78.83 | 458 | Ravens |
| 30 | 8 | Frank Clark | 72.50 | 62.29 | 76.29 | 759 | Chiefs |
| 31 | 9 | Jadeveon Clowney | 72.19 | 81.93 | 66.85 | 425 | Titans |
| 32 | 10 | Chandler Jones | 71.57 | 65.20 | 77.38 | 286 | Cardinals |
| 33 | 11 | Melvin Ingram III | 71.39 | 65.30 | 76.92 | 361 | Chargers |
| 34 | 12 | Jerry Hughes | 70.92 | 61.12 | 73.81 | 629 | Bills |
| 35 | 13 | Randy Gregory | 70.58 | 68.31 | 71.48 | 270 | Cowboys |
| 36 | 14 | Shaq Lawson | 70.54 | 66.00 | 71.16 | 571 | Dolphins |
| 37 | 15 | Yannick Ngakoue | 70.45 | 59.91 | 74.14 | 657 | Ravens |
| 38 | 16 | Tyus Bowser | 70.29 | 62.13 | 71.77 | 540 | Ravens |
| 39 | 17 | Robert Quinn | 70.27 | 56.37 | 76.52 | 548 | Bears |
| 40 | 18 | Terrell Lewis | 70.26 | 64.88 | 78.02 | 124 | Rams |
| 41 | 19 | J.J. Watt | 70.10 | 61.05 | 74.46 | 1013 | Texans |
| 42 | 20 | Arik Armstead | 69.99 | 65.39 | 68.89 | 750 | 49ers |
| 43 | 21 | Mario Addison | 69.96 | 53.04 | 77.91 | 606 | Bills |
| 44 | 22 | Deatrich Wise Jr. | 69.89 | 67.31 | 68.08 | 565 | Patriots |
| 45 | 23 | Derek Barnett | 69.75 | 67.79 | 71.15 | 535 | Eagles |
| 46 | 24 | Jacob Martin | 69.75 | 59.72 | 73.94 | 375 | Texans |
| 47 | 25 | Everson Griffen | 68.92 | 57.31 | 74.90 | 528 | Lions |
| 48 | 26 | Dante Fowler Jr. | 68.82 | 61.96 | 70.48 | 601 | Falcons |
| 49 | 27 | Samson Ebukam | 68.29 | 60.44 | 69.36 | 364 | Rams |
| 50 | 28 | Sam Hubbard | 68.09 | 62.65 | 69.44 | 665 | Bengals |
| 51 | 29 | Bud Dupree | 68.07 | 63.06 | 69.84 | 609 | Steelers |
| 52 | 30 | Preston Smith | 68.06 | 58.01 | 70.59 | 814 | Packers |
| 53 | 31 | Carl Granderson | 67.91 | 62.04 | 72.21 | 291 | Saints |
| 54 | 32 | Ifeadi Odenigbo | 67.73 | 63.03 | 67.21 | 696 | Vikings |
| 55 | 33 | Clelin Ferrell | 67.70 | 76.16 | 61.55 | 461 | Raiders |
| 56 | 34 | Vinny Curry | 67.68 | 57.65 | 73.63 | 310 | Eagles |
| 57 | 35 | Jabaal Sheard | 67.49 | 63.07 | 70.34 | 275 | Giants |
| 58 | 36 | Josh Sweat | 67.44 | 63.21 | 68.79 | 422 | Eagles |
| 59 | 37 | Emmanuel Ogbah | 67.35 | 66.72 | 65.88 | 792 | Dolphins |
| 60 | 38 | Aaron Lynch | 67.24 | 57.63 | 74.27 | 152 | Jaguars |
| 61 | 39 | Maxx Crosby | 67.16 | 60.29 | 67.57 | 906 | Raiders |
| 62 | 40 | Whitney Mercilus | 67.06 | 54.50 | 72.84 | 614 | Texans |
| 63 | 41 | Jason Pierre-Paul | 67.01 | 57.86 | 70.82 | 943 | Buccaneers |
| 64 | 42 | Frankie Luvu | 66.90 | 61.40 | 69.54 | 257 | Jets |
| 65 | 43 | Jordan Jenkins | 66.65 | 59.57 | 69.91 | 528 | Jets |
| 66 | 44 | Alton Robinson | 66.58 | 62.13 | 67.46 | 336 | Seahawks |
| 67 | 45 | Romeo Okwara | 66.45 | 64.69 | 64.29 | 748 | Lions |
| 68 | 46 | Kerry Hyder Jr. | 66.14 | 55.21 | 71.14 | 722 | 49ers |
| 69 | 47 | Jeremiah Attaochu | 66.08 | 57.03 | 72.31 | 414 | Broncos |
| 70 | 48 | Josh Uche | 65.73 | 63.72 | 70.21 | 179 | Patriots |
| 71 | 49 | Ogbo Okoronkwo | 65.34 | 61.10 | 70.64 | 158 | Rams |
| 72 | 50 | Alex Highsmith | 64.99 | 60.49 | 63.83 | 440 | Steelers |
| 73 | 51 | Malik Reed | 64.74 | 60.69 | 63.65 | 785 | Broncos |
| 74 | 52 | Jihad Ward | 64.41 | 56.99 | 68.32 | 271 | Ravens |
| 75 | 53 | Harold Landry III | 64.15 | 59.67 | 62.97 | 1050 | Titans |
| 76 | 54 | Aldon Smith | 64.00 | 54.89 | 67.38 | 809 | Cowboys |
| 77 | 55 | Bruce Irvin | 63.76 | 52.79 | 75.14 | 121 | Seahawks |
| 78 | 56 | Devon Kennard | 63.74 | 55.68 | 66.72 | 362 | Cardinals |
| 79 | 57 | Kyler Fackrell | 63.68 | 52.31 | 69.18 | 608 | Giants |
| 80 | 58 | Derek Rivers | 63.66 | 57.11 | 68.03 | 115 | Rams |
| 81 | 59 | Justin Hollins | 63.60 | 59.13 | 63.20 | 349 | Rams |
| 82 | 60 | Vic Beasley Jr. | 63.51 | 56.69 | 67.02 | 199 | Raiders |
| 83 | 61 | Barkevious Mingo | 63.42 | 51.76 | 69.52 | 391 | Bears |
| 84 | 62 | Alex Okafor | 63.39 | 54.93 | 69.35 | 283 | Chiefs |
| 85 | 63 | Jaylon Ferguson | 62.97 | 59.78 | 63.01 | 303 | Ravens |
| 86 | 64 | Efe Obada | 62.91 | 56.91 | 63.99 | 415 | Panthers |
| 87 | 65 | Adrian Clayborn | 62.77 | 53.43 | 66.08 | 404 | Browns |
| 88 | 66 | Carter Coughlin | 62.58 | 58.67 | 67.27 | 193 | Giants |
| 89 | 67 | Trent Murphy | 62.52 | 56.24 | 66.29 | 343 | Bills |
| 90 | 68 | Mike Danna | 62.23 | 59.29 | 63.15 | 334 | Chiefs |

### Rotation/backup (50 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 91 | 1 | Dion Jordan | 61.92 | 55.69 | 67.00 | 375 | 49ers |
| 92 | 2 | A.J. Epenesa | 61.90 | 60.24 | 60.93 | 291 | Bills |
| 93 | 3 | Lorenzo Carter | 61.78 | 60.44 | 64.75 | 234 | Giants |
| 94 | 4 | John Simon | 61.78 | 50.31 | 66.29 | 702 | Patriots |
| 95 | 5 | Tarell Basham | 61.44 | 59.47 | 60.26 | 734 | Jets |
| 96 | 6 | James Vaughters | 61.39 | 57.25 | 62.06 | 243 | Bears |
| 97 | 7 | Chris Smith | 61.24 | 57.79 | 63.54 | 115 | Raiders |
| 98 | 8 | Benson Mayowa | 61.16 | 54.76 | 63.34 | 571 | Seahawks |
| 99 | 9 | Yetur Gross-Matos | 61.12 | 59.34 | 62.30 | 377 | Panthers |
| 100 | 10 | Oshane Ximines | 60.75 | 59.80 | 65.04 | 110 | Giants |
| 101 | 11 | Carl Nassib | 60.47 | 58.25 | 59.66 | 463 | Raiders |
| 102 | 12 | Cassius Marsh | 60.38 | 55.80 | 63.43 | 151 | Steelers |
| 103 | 13 | Damontre Moore | 59.78 | 59.79 | 66.12 | 184 | Seahawks |
| 104 | 14 | Dawuane Smoot | 59.72 | 57.00 | 59.04 | 665 | Jaguars |
| 105 | 15 | Kamalei Correa | 59.40 | 55.95 | 62.08 | 197 | Jaguars |
| 106 | 16 | Anthony Nelson | 59.10 | 65.54 | 53.38 | 324 | Buccaneers |
| 107 | 17 | Ryan Anderson | 59.02 | 57.82 | 59.92 | 146 | Commanders |
| 108 | 18 | Khalid Kareem | 58.81 | 58.04 | 55.15 | 259 | Bengals |
| 109 | 19 | Anthony Chickillo | 58.79 | 56.11 | 60.58 | 164 | Broncos |
| 110 | 20 | Charles Harris | 58.77 | 58.96 | 57.71 | 289 | Falcons |
| 111 | 21 | Bryce Huff | 58.69 | 57.76 | 57.22 | 296 | Jets |
| 112 | 22 | D.J. Wonnum | 58.56 | 57.12 | 57.44 | 471 | Vikings |
| 113 | 23 | Ben Banogu | 58.33 | 57.65 | 59.57 | 100 | Colts |
| 114 | 24 | Al-Quadin Muhammad | 58.06 | 57.88 | 54.21 | 579 | Colts |
| 115 | 25 | K'Lavon Chaisson | 58.02 | 56.43 | 54.92 | 569 | Jaguars |
| 116 | 26 | Alex Barrett | 57.78 | 59.34 | 63.44 | 120 | 49ers |
| 117 | 27 | Dorance Armstrong | 57.46 | 56.13 | 54.69 | 368 | Cowboys |
| 118 | 28 | Derick Roberson | 57.42 | 57.41 | 63.95 | 248 | Titans |
| 119 | 29 | Isaac Rochell | 57.12 | 55.72 | 53.89 | 438 | Chargers |
| 120 | 30 | Stephen Weatherly | 57.08 | 55.35 | 57.72 | 358 | Panthers |
| 121 | 31 | Arden Key | 57.07 | 59.51 | 55.13 | 435 | Raiders |
| 122 | 32 | Kyle Phillips | 56.94 | 59.68 | 57.20 | 171 | Jets |
| 123 | 33 | Darryl Johnson | 56.83 | 56.93 | 53.63 | 225 | Bills |
| 124 | 34 | Marquis Haynes Sr. | 56.53 | 55.23 | 57.82 | 390 | Panthers |
| 125 | 35 | Jordan Willis | 56.52 | 59.24 | 56.37 | 229 | 49ers |
| 126 | 36 | Zach Allen | 56.22 | 54.31 | 55.28 | 505 | Cardinals |
| 127 | 37 | Porter Gustin | 56.15 | 59.53 | 54.93 | 326 | Browns |
| 128 | 38 | Tanoh Kpassagnon | 56.00 | 54.40 | 54.36 | 720 | Chiefs |
| 129 | 39 | Tashawn Bower | 55.56 | 57.54 | 61.32 | 137 | Patriots |
| 130 | 40 | Jonathan Greenard | 54.60 | 57.21 | 52.86 | 265 | Texans |
| 131 | 41 | Steven Means | 54.17 | 48.85 | 55.22 | 646 | Falcons |
| 132 | 42 | Shilique Calhoun | 53.71 | 53.90 | 54.73 | 256 | Patriots |
| 133 | 43 | Christian Jones | 53.59 | 48.60 | 52.75 | 510 | Lions |
| 134 | 44 | Amani Bledsoe | 52.44 | 55.40 | 48.38 | 312 | Bengals |
| 135 | 45 | Adam Gotsis | 52.42 | 46.97 | 51.88 | 579 | Jaguars |
| 136 | 46 | Austin Larkin | 52.30 | 57.19 | 55.29 | 127 | Panthers |
| 137 | 47 | Olasunkanmi Adeniyi | 52.05 | 56.75 | 47.88 | 146 | Steelers |
| 138 | 48 | Austin Bryant | 50.72 | 59.11 | 52.16 | 212 | Lions |
| 139 | 49 | Kylie Fitts | 49.17 | 56.75 | 48.60 | 140 | Cardinals |
| 140 | 50 | Jabari Zuniga | 48.17 | 56.53 | 46.77 | 103 | Jets |

## G — Guard

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Zack Martin | 94.44 | 87.40 | 94.97 | 618 | Cowboys |
| 2 | 2 | Wyatt Teller | 93.15 | 87.45 | 92.78 | 694 | Browns |
| 3 | 3 | Ali Marpet | 92.98 | 85.55 | 93.77 | 849 | Buccaneers |
| 4 | 4 | Quenton Nelson | 92.29 | 86.20 | 92.18 | 1082 | Colts |
| 5 | 5 | Joel Bitonio | 90.52 | 84.60 | 90.30 | 1061 | Browns |
| 6 | 6 | Shaq Mason | 89.82 | 83.74 | 89.70 | 782 | Patriots |
| 7 | 7 | Brandon Scherff | 88.83 | 83.13 | 88.47 | 857 | Commanders |
| 8 | 8 | Laken Tomlinson | 85.65 | 78.80 | 86.05 | 1094 | 49ers |
| 9 | 9 | Chris Lindstrom | 84.49 | 77.10 | 85.25 | 1122 | Falcons |
| 10 | 10 | Joe Thuney | 81.81 | 74.18 | 82.73 | 980 | Patriots |
| 11 | 11 | Damien Lewis | 81.60 | 70.11 | 85.09 | 967 | Seahawks |
| 12 | 12 | Austin Corbett | 80.48 | 70.90 | 82.70 | 1120 | Rams |
| 13 | 13 | Connor Williams | 80.11 | 71.20 | 81.89 | 1146 | Cowboys |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Rodger Saffold | 79.88 | 69.84 | 82.40 | 872 | Titans |
| 15 | 2 | David Edwards | 79.11 | 70.30 | 80.81 | 1006 | Rams |
| 16 | 3 | Nate Davis | 78.75 | 69.70 | 80.62 | 1074 | Titans |
| 17 | 4 | Mike Iupati | 78.50 | 66.45 | 82.36 | 498 | Seahawks |
| 18 | 5 | Graham Glasgow | 78.40 | 67.89 | 81.24 | 764 | Broncos |
| 19 | 6 | Wes Schweitzer | 77.90 | 69.00 | 79.66 | 990 | Commanders |
| 20 | 7 | Elgton Jenkins | 77.60 | 67.70 | 80.04 | 1037 | Packers |
| 21 | 8 | Alex Cappa | 77.48 | 69.00 | 78.97 | 1070 | Buccaneers |
| 22 | 9 | Oday Aboushi | 77.35 | 65.24 | 81.25 | 622 | Lions |
| 23 | 10 | A.J. Cann | 77.33 | 68.82 | 78.84 | 919 | Jaguars |
| 24 | 11 | Nick Allegretti | 76.77 | 65.20 | 80.31 | 694 | Chiefs |
| 25 | 12 | Tom Compton | 76.51 | 63.24 | 81.19 | 148 | 49ers |
| 26 | 13 | Mark Glowinski | 76.40 | 67.30 | 78.30 | 1090 | Colts |
| 27 | 14 | Alex Lewis | 76.37 | 64.90 | 79.85 | 544 | Jets |
| 28 | 15 | Justin Pugh | 76.31 | 64.76 | 79.85 | 958 | Cardinals |
| 29 | 16 | Andrew Norwell | 75.91 | 67.43 | 77.40 | 801 | Jaguars |
| 30 | 17 | Jon Feliciano | 75.76 | 63.50 | 79.77 | 571 | Bills |
| 31 | 18 | Kevin Zeitler | 75.71 | 65.90 | 78.09 | 1003 | Giants |
| 32 | 19 | Lucas Patrick | 75.49 | 64.69 | 78.52 | 939 | Packers |
| 33 | 20 | Germain Ifedi | 75.22 | 65.00 | 77.86 | 1066 | Bears |
| 34 | 21 | Bradley Bozeman | 74.90 | 64.30 | 77.80 | 1017 | Ravens |
| 35 | 22 | Kelechi Osemele | 74.76 | 59.86 | 80.53 | 282 | Chiefs |
| 36 | 23 | Ereck Flowers | 74.52 | 65.50 | 76.36 | 857 | Dolphins |
| 37 | 24 | Alex Redmond | 74.36 | 61.01 | 79.10 | 448 | Bengals |

### Starter (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 38 | 1 | Gabe Jackson | 73.77 | 63.70 | 76.32 | 1062 | Raiders |
| 39 | 2 | James Daniels | 72.98 | 63.23 | 75.31 | 305 | Bears |
| 40 | 3 | Ike Boettger | 72.96 | 64.22 | 74.62 | 623 | Bills |
| 41 | 4 | Zach Fulton | 72.57 | 62.95 | 74.81 | 953 | Texans |
| 42 | 5 | Connor McGovern | 72.57 | 61.33 | 75.90 | 606 | Cowboys |
| 43 | 6 | David DeCastro | 72.48 | 63.92 | 74.02 | 845 | Steelers |
| 44 | 7 | John Miller | 72.36 | 61.15 | 75.66 | 910 | Panthers |
| 45 | 8 | Andrus Peat | 72.16 | 61.06 | 75.40 | 766 | Saints |
| 46 | 9 | Stefen Wisniewski | 71.55 | 60.55 | 74.72 | 209 | Chiefs |
| 47 | 10 | Isaac Seumalo | 71.36 | 61.85 | 73.53 | 588 | Eagles |
| 48 | 11 | Joe Dahl | 71.28 | 58.65 | 75.53 | 264 | Lions |
| 49 | 12 | Dalton Risner | 71.25 | 61.30 | 73.72 | 999 | Broncos |
| 50 | 13 | Xavier Su'a-Filo | 70.64 | 59.51 | 73.90 | 293 | Bengals |
| 51 | 14 | Quinton Spain | 70.58 | 56.99 | 75.48 | 720 | Bengals |
| 52 | 15 | Ryan Groy | 70.39 | 58.79 | 73.96 | 271 | Chargers |
| 53 | 16 | Jonah Jackson | 70.04 | 57.00 | 74.56 | 1006 | Lions |
| 54 | 17 | Michael Jordan | 69.85 | 56.30 | 74.72 | 731 | Bengals |
| 55 | 18 | Sua Opeta | 69.71 | 59.00 | 72.69 | 170 | Eagles |
| 56 | 19 | Kevin Dotson | 69.64 | 63.74 | 69.40 | 358 | Steelers |
| 57 | 20 | Will Hernandez | 69.63 | 58.61 | 72.81 | 525 | Giants |
| 58 | 21 | Justin McCray | 68.69 | 57.92 | 71.71 | 156 | Falcons |
| 59 | 22 | Senio Kelemete | 68.42 | 56.64 | 72.11 | 367 | Texans |
| 60 | 23 | Brian Winters | 68.22 | 55.72 | 72.39 | 618 | Bills |
| 61 | 24 | Matt Pryor | 68.19 | 55.92 | 72.21 | 776 | Eagles |
| 62 | 25 | James Carpenter | 67.82 | 56.30 | 71.34 | 826 | Falcons |
| 63 | 26 | Ben Bartch | 67.77 | 59.29 | 69.25 | 219 | Jaguars |
| 64 | 27 | Cesar Ruiz | 67.35 | 58.78 | 68.90 | 744 | Saints |
| 65 | 28 | Andrew Wylie | 67.34 | 54.93 | 71.44 | 972 | Chiefs |
| 66 | 29 | Max Scharping | 66.88 | 54.64 | 70.87 | 454 | Texans |
| 67 | 30 | Michael Schofield III | 66.79 | 54.92 | 70.54 | 270 | Panthers |
| 68 | 31 | J.R. Sweezy | 66.34 | 53.36 | 70.83 | 643 | Cardinals |
| 69 | 32 | Ben Powers | 66.16 | 59.57 | 66.38 | 513 | Ravens |
| 70 | 33 | Jordan Simmons | 65.89 | 53.48 | 70.00 | 593 | Seahawks |
| 71 | 34 | Solomon Kindley | 65.50 | 52.42 | 70.05 | 748 | Dolphins |
| 72 | 35 | Pat Elflein | 65.37 | 52.17 | 70.01 | 419 | Jets |
| 73 | 36 | John Simpson | 65.36 | 52.82 | 69.55 | 252 | Raiders |
| 74 | 37 | Forrest Lamp | 64.03 | 49.40 | 69.62 | 1174 | Chargers |
| 75 | 38 | Rashaad Coward | 63.85 | 51.98 | 67.59 | 333 | Bears |
| 76 | 39 | Tyre Phillips | 63.64 | 51.60 | 67.50 | 418 | Ravens |
| 77 | 40 | Wes Martin | 63.40 | 51.08 | 67.44 | 339 | Commanders |
| 78 | 41 | Jon Runyan | 63.04 | 57.50 | 62.56 | 160 | Packers |
| 79 | 42 | Netane Muti | 62.68 | 50.84 | 66.41 | 122 | Broncos |
| 80 | 43 | Austin Schlottmann | 62.58 | 49.44 | 67.17 | 269 | Broncos |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 81 | 1 | Dakota Dozier | 61.64 | 44.70 | 68.77 | 1083 | Vikings |
| 82 | 2 | Dru Samia | 59.99 | 45.86 | 65.25 | 272 | Vikings |
| 83 | 3 | Trai Turner | 58.68 | 40.00 | 66.97 | 536 | Chargers |
| 84 | 4 | Shane Lemieux | 58.33 | 40.11 | 66.31 | 504 | Giants |

## HB — Running Back

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Chubb | 85.38 | 81.01 | 84.13 | 158 | Browns |
| 2 | 2 | Derrick Henry | 83.78 | 84.82 | 78.92 | 223 | Titans |
| 3 | 3 | Alvin Kamara | 83.06 | 81.57 | 79.88 | 345 | Saints |
| 4 | 4 | Dalvin Cook | 82.49 | 86.08 | 75.93 | 263 | Vikings |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Austin Ekeler | 79.50 | 74.82 | 78.45 | 235 | Chargers |
| 6 | 2 | Aaron Jones | 79.35 | 77.26 | 76.58 | 253 | Packers |
| 7 | 3 | Tony Pollard | 79.11 | 68.47 | 82.04 | 199 | Cowboys |
| 8 | 4 | J.K. Dobbins | 78.16 | 68.69 | 80.31 | 217 | Ravens |
| 9 | 5 | Kareem Hunt | 77.80 | 72.44 | 77.21 | 252 | Browns |
| 10 | 6 | Chris Carson | 77.80 | 75.26 | 75.32 | 207 | Seahawks |
| 11 | 7 | Jonathan Taylor | 77.77 | 77.47 | 73.81 | 203 | Colts |
| 12 | 8 | David Montgomery | 76.73 | 81.80 | 69.19 | 391 | Bears |
| 13 | 9 | Gus Edwards | 76.26 | 73.50 | 73.93 | 104 | Ravens |
| 14 | 10 | Josh Jacobs | 76.18 | 72.71 | 74.32 | 231 | Raiders |
| 15 | 11 | Clyde Edwards-Helaire | 75.69 | 73.27 | 73.14 | 286 | Chiefs |
| 16 | 12 | James Robinson | 75.42 | 71.19 | 74.07 | 286 | Jaguars |
| 17 | 13 | Antonio Gibson | 74.88 | 74.46 | 70.99 | 182 | Commanders |
| 18 | 14 | Devin Singletary | 74.31 | 66.13 | 75.60 | 318 | Bills |

### Starter (45 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Darrell Henderson | 73.96 | 72.69 | 70.64 | 147 | Rams |
| 20 | 2 | Ronald Jones | 73.15 | 69.62 | 71.34 | 190 | Buccaneers |
| 21 | 3 | Miles Sanders | 73.14 | 62.64 | 75.98 | 295 | Eagles |
| 22 | 4 | Le'Veon Bell | 72.49 | 70.85 | 69.42 | 153 | Chiefs |
| 23 | 5 | Nyheim Hines | 72.06 | 78.00 | 63.94 | 264 | Colts |
| 24 | 6 | Latavius Murray | 71.95 | 76.91 | 64.47 | 133 | Saints |
| 25 | 7 | Melvin Gordon III | 71.76 | 71.55 | 67.73 | 290 | Broncos |
| 26 | 8 | Ezekiel Elliott | 71.47 | 65.30 | 71.42 | 387 | Cowboys |
| 27 | 9 | Myles Gaskin | 71.25 | 70.73 | 67.43 | 208 | Dolphins |
| 28 | 10 | Duke Johnson Jr. | 71.08 | 62.18 | 72.85 | 204 | Texans |
| 29 | 11 | Zack Moss | 70.96 | 68.30 | 68.57 | 173 | Bills |
| 30 | 12 | James Conner | 70.71 | 67.24 | 68.85 | 269 | Steelers |
| 31 | 13 | Wayne Gallman | 70.31 | 67.20 | 68.22 | 160 | Giants |
| 32 | 14 | D'Andre Swift | 70.08 | 67.25 | 67.80 | 226 | Lions |
| 33 | 15 | Jalen Richard | 70.00 | 58.75 | 73.33 | 123 | Raiders |
| 34 | 16 | Jerick McKinnon | 69.98 | 71.63 | 64.72 | 213 | 49ers |
| 35 | 17 | Jeremy McNichols | 69.71 | 58.86 | 72.78 | 152 | Titans |
| 36 | 18 | Kenyan Drake | 69.67 | 60.80 | 71.41 | 251 | Cardinals |
| 37 | 19 | Chase Edmonds | 69.61 | 68.24 | 66.35 | 305 | Cardinals |
| 38 | 20 | Boston Scott | 69.53 | 66.11 | 67.65 | 211 | Eagles |
| 39 | 21 | Mike Davis | 69.42 | 73.48 | 62.55 | 303 | Panthers |
| 40 | 22 | Rex Burkhead | 69.24 | 70.35 | 64.33 | 123 | Patriots |
| 41 | 23 | Devontae Booker | 68.67 | 64.57 | 67.23 | 105 | Raiders |
| 42 | 24 | Carlos Hyde | 68.53 | 64.65 | 66.95 | 125 | Seahawks |
| 43 | 25 | Jamaal Williams | 68.45 | 69.60 | 63.51 | 201 | Packers |
| 44 | 26 | Cam Akers | 68.31 | 64.46 | 66.71 | 107 | Rams |
| 45 | 27 | Joe Mixon | 68.25 | 63.97 | 66.93 | 128 | Bengals |
| 46 | 28 | Brian Hill | 68.16 | 58.62 | 70.36 | 165 | Falcons |
| 47 | 29 | J.D. McKissic | 68.03 | 62.70 | 67.41 | 399 | Commanders |
| 48 | 30 | Adrian Peterson | 67.93 | 59.71 | 69.24 | 110 | Lions |
| 49 | 31 | Kerryon Johnson | 67.90 | 61.03 | 68.31 | 159 | Lions |
| 50 | 32 | Jeff Wilson Jr. | 67.79 | 65.38 | 65.23 | 133 | 49ers |
| 51 | 33 | Chris Thompson | 67.52 | 57.05 | 70.34 | 127 | Jaguars |
| 52 | 34 | Ty Johnson | 67.19 | 64.67 | 64.71 | 100 | Jets |
| 53 | 35 | Dion Lewis | 66.91 | 55.57 | 70.30 | 201 | Giants |
| 54 | 36 | David Johnson | 66.57 | 62.79 | 64.93 | 333 | Texans |
| 55 | 37 | Giovani Bernard | 66.24 | 63.10 | 64.17 | 276 | Bengals |
| 56 | 38 | Leonard Fournette | 65.87 | 59.85 | 65.71 | 216 | Buccaneers |
| 57 | 39 | Todd Gurley II | 65.45 | 55.50 | 67.91 | 229 | Falcons |
| 58 | 40 | James White | 65.16 | 64.50 | 61.43 | 184 | Patriots |
| 59 | 41 | Ito Smith | 64.92 | 59.40 | 64.43 | 139 | Falcons |
| 60 | 42 | Dare Ogunbowale | 64.75 | 59.32 | 64.20 | 121 | Jaguars |
| 61 | 43 | Frank Gore | 64.19 | 60.61 | 62.41 | 136 | Jets |
| 62 | 44 | Malcolm Brown | 62.75 | 57.53 | 62.07 | 276 | Rams |
| 63 | 45 | Joshua Kelley | 62.07 | 61.81 | 58.07 | 130 | Chargers |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 64 | 1 | Kalen Ballage | 61.94 | 61.66 | 57.96 | 143 | Chargers |
| 65 | 2 | Darrel Williams | 60.92 | 60.68 | 56.92 | 177 | Chiefs |

## LB — Linebacker

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Fred Warner | 83.89 | 88.60 | 76.59 | 973 | 49ers |
| 2 | 2 | Bobby Wagner | 81.84 | 83.20 | 76.97 | 1141 | Seahawks |
| 3 | 3 | Lavonte David | 80.25 | 81.50 | 75.67 | 1058 | Buccaneers |

### Good (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Eric Kendricks | 78.29 | 82.04 | 74.95 | 754 | Vikings |
| 5 | 2 | Demario Davis | 77.43 | 78.10 | 72.82 | 1032 | Saints |
| 6 | 3 | Blake Martinez | 77.16 | 75.90 | 73.84 | 1063 | Giants |
| 7 | 4 | Denzel Perryman | 75.96 | 74.49 | 77.04 | 317 | Chargers |

### Starter (34 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 8 | 1 | K.J. Wright | 73.41 | 75.30 | 70.26 | 990 | Seahawks |
| 9 | 2 | Mykal Walker | 73.00 | 69.62 | 71.09 | 387 | Falcons |
| 10 | 3 | Myles Jack | 70.85 | 69.10 | 70.45 | 931 | Jaguars |
| 11 | 4 | Roquan Smith | 70.62 | 67.20 | 69.99 | 1016 | Bears |
| 12 | 5 | Cole Holcomb | 69.97 | 69.87 | 69.77 | 555 | Commanders |
| 13 | 6 | Josey Jewell | 69.75 | 68.10 | 68.25 | 1011 | Broncos |
| 14 | 7 | Sione Takitaki | 68.89 | 68.16 | 68.99 | 435 | Browns |
| 15 | 8 | Deion Jones | 68.84 | 68.70 | 66.85 | 1040 | Falcons |
| 16 | 9 | Willie Gay | 68.65 | 64.64 | 68.19 | 269 | Chiefs |
| 17 | 10 | B.J. Goodson | 68.28 | 65.40 | 67.91 | 848 | Browns |
| 18 | 11 | L.J. Fort | 68.22 | 66.41 | 68.69 | 381 | Ravens |
| 19 | 12 | Malcolm Smith | 67.86 | 69.67 | 68.22 | 559 | Browns |
| 20 | 13 | Isaiah Simmons | 66.99 | 59.93 | 67.53 | 376 | Cardinals |
| 21 | 14 | Nick Kwiatkoski | 66.60 | 66.64 | 67.40 | 651 | Raiders |
| 22 | 15 | Nicholas Morrow | 66.28 | 63.47 | 65.45 | 723 | Raiders |
| 23 | 16 | Kamal Martin | 66.26 | 66.40 | 68.25 | 190 | Packers |
| 24 | 17 | Jamie Collins Sr. | 66.18 | 64.20 | 64.36 | 829 | Lions |
| 25 | 18 | Jayon Brown | 66.04 | 65.89 | 65.73 | 653 | Titans |
| 26 | 19 | Kevin Pierre-Louis | 65.58 | 66.05 | 68.08 | 506 | Commanders |
| 27 | 20 | T.J. Edwards | 65.43 | 65.03 | 65.69 | 492 | Eagles |
| 28 | 21 | Jermaine Carter | 65.09 | 63.83 | 66.86 | 284 | Panthers |
| 29 | 22 | Zach Cunningham | 64.85 | 60.90 | 63.74 | 944 | Texans |
| 30 | 23 | Dre Greenlaw | 64.34 | 59.82 | 65.13 | 700 | 49ers |
| 31 | 24 | Nick Vigil | 63.73 | 60.37 | 64.93 | 312 | Chargers |
| 32 | 25 | Jarrad Davis | 63.58 | 61.39 | 63.48 | 329 | Lions |
| 33 | 26 | Rashaan Evans | 62.90 | 53.70 | 65.29 | 895 | Titans |
| 34 | 27 | Alex Singleton | 62.87 | 58.95 | 61.96 | 750 | Eagles |
| 35 | 28 | Joe Giles-Harris | 62.82 | 67.80 | 66.66 | 205 | Jaguars |
| 36 | 29 | Neville Hewitt | 62.68 | 59.30 | 63.47 | 1130 | Jets |
| 37 | 30 | Avery Williamson | 62.60 | 52.20 | 65.88 | 665 | Steelers |
| 38 | 31 | Jaylon Smith | 62.50 | 54.20 | 63.87 | 1083 | Cowboys |
| 39 | 32 | Robert Spillane | 62.23 | 64.28 | 65.30 | 379 | Steelers |
| 40 | 33 | Kyle Van Noy | 62.19 | 61.59 | 59.45 | 811 | Dolphins |
| 41 | 34 | Malik Harrison | 62.16 | 55.91 | 62.16 | 265 | Ravens |

### Rotation/backup (74 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 42 | 1 | Alex Anzalone | 61.97 | 60.72 | 63.53 | 525 | Saints |
| 43 | 2 | Azeez Al-Shaair | 61.81 | 60.61 | 62.75 | 305 | 49ers |
| 44 | 3 | Foyesade Oluokun | 61.50 | 56.60 | 61.11 | 895 | Falcons |
| 45 | 4 | Jerome Baker | 61.42 | 55.20 | 61.40 | 868 | Dolphins |
| 46 | 5 | Todd Davis | 61.19 | 60.35 | 62.36 | 281 | Vikings |
| 47 | 6 | Kenneth Murray Jr. | 61.08 | 54.40 | 61.36 | 959 | Chargers |
| 48 | 7 | Thomas Davis Sr. | 60.78 | 56.64 | 64.89 | 137 | Commanders |
| 49 | 8 | Anfernee Jennings | 60.71 | 56.77 | 62.30 | 293 | Patriots |
| 50 | 9 | Troy Reeder | 60.50 | 60.29 | 62.20 | 423 | Rams |
| 51 | 10 | Jon Bostic | 60.41 | 52.70 | 61.38 | 966 | Commanders |
| 52 | 11 | Brennan Scarlett | 60.29 | 57.46 | 62.79 | 286 | Texans |
| 53 | 12 | Eric Wilson | 60.24 | 53.50 | 60.99 | 1034 | Vikings |
| 54 | 13 | Devin Bush | 60.20 | 59.83 | 63.45 | 278 | Steelers |
| 55 | 14 | Joe Schobert | 59.96 | 54.20 | 60.26 | 1110 | Jaguars |
| 56 | 15 | Damien Wilson | 59.93 | 53.40 | 61.69 | 531 | Chiefs |
| 57 | 16 | Ja'Whaun Bentley | 59.61 | 54.14 | 63.36 | 608 | Patriots |
| 58 | 17 | Jordyn Brooks | 59.59 | 51.70 | 62.76 | 367 | Seahawks |
| 59 | 18 | Jordan Hicks | 59.23 | 50.40 | 61.79 | 1024 | Cardinals |
| 60 | 19 | Benardrick McKinney | 59.23 | 58.41 | 62.48 | 234 | Texans |
| 61 | 20 | Logan Wilson | 59.09 | 56.57 | 60.77 | 343 | Bengals |
| 62 | 21 | Anthony Walker Jr. | 58.96 | 48.94 | 61.68 | 697 | Colts |
| 63 | 22 | Josh Bynes | 58.81 | 52.66 | 61.03 | 761 | Bengals |
| 64 | 23 | Kwon Alexander | 58.80 | 58.74 | 61.34 | 668 | Saints |
| 65 | 24 | Anthony Hitchens | 58.74 | 51.31 | 61.10 | 603 | Chiefs |
| 66 | 25 | Duke Riley | 58.38 | 55.91 | 60.35 | 571 | Eagles |
| 67 | 26 | Matt Milano | 58.11 | 56.71 | 58.95 | 335 | Bills |
| 68 | 27 | Shaq Thompson | 57.89 | 49.80 | 60.15 | 1031 | Panthers |
| 69 | 28 | Reggie Ragland | 57.82 | 50.07 | 59.45 | 562 | Lions |
| 70 | 29 | De'Vondre Campbell | 57.82 | 49.00 | 59.54 | 880 | Cardinals |
| 71 | 30 | Terez Hall | 57.73 | 55.39 | 63.46 | 259 | Patriots |
| 72 | 31 | Bobby Okereke | 57.57 | 50.49 | 59.42 | 685 | Colts |
| 73 | 32 | Will Compton | 57.46 | 56.16 | 62.40 | 124 | Titans |
| 74 | 33 | Cody Barton | 57.21 | 57.72 | 61.04 | 115 | Seahawks |
| 75 | 34 | Tremaine Edmunds | 57.17 | 47.90 | 59.92 | 911 | Bills |
| 76 | 35 | David Mayo | 56.77 | 52.11 | 59.97 | 194 | Giants |
| 77 | 36 | Devante Downs | 56.74 | 52.64 | 59.48 | 233 | Giants |
| 78 | 37 | David Long Jr. | 56.56 | 56.06 | 60.15 | 379 | Titans |
| 79 | 38 | Kyzir White | 56.54 | 53.12 | 59.97 | 538 | Chargers |
| 80 | 39 | Chris Board | 56.44 | 55.13 | 58.04 | 263 | Ravens |
| 81 | 40 | Joe Thomas | 56.39 | 51.30 | 60.94 | 410 | Cowboys |
| 82 | 41 | Leighton Vander Esch | 56.24 | 52.02 | 60.20 | 460 | Cowboys |
| 83 | 42 | Vince Williams | 56.18 | 51.49 | 57.22 | 672 | Steelers |
| 84 | 43 | Kamu Grugier-Hill | 56.05 | 55.18 | 56.43 | 207 | Dolphins |
| 85 | 44 | Germaine Pratt | 56.04 | 43.08 | 60.89 | 686 | Bengals |
| 86 | 45 | Jacob Phillips | 55.75 | 53.33 | 60.49 | 169 | Browns |
| 87 | 46 | Adarius Taylor | 55.73 | 54.20 | 57.90 | 111 | Panthers |
| 88 | 47 | Ty Summers | 55.68 | 52.96 | 60.70 | 176 | Packers |
| 89 | 48 | A.J. Klein | 55.61 | 47.61 | 57.09 | 652 | Bills |
| 90 | 49 | Akeem Davis-Gaither | 55.39 | 48.06 | 56.11 | 314 | Bengals |
| 91 | 50 | Ben Niemann | 55.30 | 49.80 | 57.72 | 468 | Chiefs |
| 92 | 51 | Krys Barnes | 55.15 | 48.32 | 59.70 | 421 | Packers |
| 93 | 52 | Nathan Gerry | 54.90 | 49.07 | 60.56 | 479 | Eagles |
| 94 | 53 | Danny Trevathan | 54.76 | 44.10 | 59.89 | 832 | Bears |
| 95 | 54 | Cory Littleton | 54.31 | 46.30 | 56.51 | 849 | Raiders |
| 96 | 55 | Raekwon McMillan | 53.90 | 46.48 | 57.70 | 170 | Raiders |
| 97 | 56 | Kenny Young | 53.65 | 46.04 | 58.30 | 472 | Rams |
| 98 | 57 | Devin White | 53.24 | 43.40 | 57.45 | 993 | Buccaneers |
| 99 | 58 | Mack Wilson Sr. | 51.78 | 43.97 | 55.17 | 372 | Browns |
| 100 | 59 | Jahlani Tavai | 51.76 | 40.00 | 55.81 | 624 | Lions |
| 101 | 60 | Tahir Whitehead | 51.68 | 40.42 | 56.05 | 398 | Panthers |
| 102 | 61 | Hardy Nickerson | 51.66 | 47.96 | 58.81 | 102 | Vikings |
| 103 | 62 | Elandon Roberts | 51.31 | 40.00 | 56.86 | 402 | Dolphins |
| 104 | 63 | Sean Lee | 51.15 | 42.84 | 58.04 | 180 | Cowboys |
| 105 | 64 | Patrick Queen | 51.08 | 40.00 | 54.30 | 858 | Ravens |
| 106 | 65 | Tyrell Adams | 51.02 | 43.58 | 56.92 | 812 | Texans |
| 107 | 66 | Micah Kiser | 50.73 | 43.49 | 58.16 | 559 | Rams |
| 108 | 67 | Tyrel Dodson | 50.36 | 56.47 | 58.08 | 172 | Bills |
| 109 | 68 | Christian Kirksey | 50.30 | 46.76 | 57.34 | 548 | Packers |
| 110 | 69 | Troy Dye | 49.59 | 44.55 | 55.04 | 201 | Vikings |
| 111 | 70 | Dakota Allen | 49.37 | 47.80 | 57.92 | 103 | Jaguars |
| 112 | 71 | Tae Crowder | 49.31 | 43.46 | 55.29 | 403 | Giants |
| 113 | 72 | Harvey Langi | 49.29 | 40.15 | 55.38 | 513 | Jets |
| 114 | 73 | Bryce Hager | 46.95 | 47.77 | 54.64 | 138 | Jets |
| 115 | 74 | Darius Harris | 45.00 | 47.52 | 55.67 | 126 | Chiefs |

## QB — Quarterback

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Rodgers | 88.14 | 89.63 | 83.12 | 607 | Packers |
| 2 | 2 | Deshaun Watson | 87.32 | 86.73 | 83.39 | 687 | Texans |
| 3 | 3 | Patrick Mahomes | 86.86 | 88.83 | 81.32 | 693 | Chiefs |
| 4 | 4 | Russell Wilson | 86.09 | 88.78 | 79.64 | 695 | Seahawks |
| 5 | 5 | Tom Brady | 83.73 | 87.65 | 76.01 | 680 | Buccaneers |
| 6 | 6 | Derek Carr | 81.81 | 81.95 | 78.35 | 604 | Raiders |
| 7 | 7 | Kirk Cousins | 81.47 | 81.57 | 78.49 | 613 | Vikings |
| 8 | 8 | Josh Allen | 81.07 | 81.14 | 77.58 | 677 | Bills |
| 9 | 9 | Ryan Tannehill | 80.31 | 82.89 | 78.10 | 557 | Titans |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Matt Ryan | 78.28 | 80.09 | 72.25 | 721 | Falcons |
| 11 | 2 | Philip Rivers | 78.16 | 76.56 | 75.98 | 600 | Colts |
| 12 | 3 | Matthew Stafford | 78.14 | 78.83 | 75.67 | 626 | Lions |
| 13 | 4 | Baker Mayfield | 77.35 | 78.84 | 73.29 | 568 | Browns |
| 14 | 5 | Drew Brees | 76.50 | 76.55 | 77.79 | 421 | Saints |
| 15 | 6 | Lamar Jackson | 75.17 | 77.27 | 74.10 | 478 | Ravens |
| 16 | 7 | Kyler Murray | 74.37 | 72.78 | 71.13 | 674 | Cardinals |
| 17 | 8 | Dak Prescott | 74.32 | 74.00 | 77.18 | 247 | Cowboys |

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Jared Goff | 73.56 | 73.30 | 69.45 | 617 | Rams |
| 19 | 2 | Justin Herbert | 73.52 | 78.40 | 74.27 | 689 | Chargers |
| 20 | 3 | Ryan Fitzpatrick | 71.03 | 71.35 | 73.40 | 316 | Dolphins |
| 21 | 4 | Ben Roethlisberger | 69.07 | 68.59 | 68.97 | 672 | Steelers |
| 22 | 5 | Daniel Jones | 68.30 | 71.28 | 63.97 | 552 | Giants |
| 23 | 6 | Teddy Bridgewater | 68.14 | 67.07 | 71.70 | 594 | Panthers |
| 24 | 7 | Gardner Minshew | 67.79 | 66.19 | 69.94 | 396 | Jaguars |
| 25 | 8 | Jimmy Garoppolo | 66.27 | 67.70 | 73.09 | 159 | 49ers |
| 26 | 9 | Andy Dalton | 65.97 | 68.42 | 64.81 | 393 | Cowboys |
| 27 | 10 | Joe Burrow | 64.74 | 72.86 | 66.69 | 472 | Bengals |
| 28 | 11 | Cam Newton | 64.26 | 66.58 | 65.53 | 444 | Patriots |
| 29 | 12 | Carson Wentz | 63.89 | 66.12 | 59.92 | 544 | Eagles |
| 30 | 13 | Mitch Trubisky | 63.73 | 60.16 | 68.00 | 340 | Bears |
| 31 | 14 | Taysom Hill | 62.32 | 65.13 | 74.28 | 161 | Saints |

### Rotation/backup (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 32 | 1 | C.J. Beathard | 61.56 | 62.19 | 73.59 | 123 | 49ers |
| 33 | 2 | Nick Foles | 61.49 | 66.02 | 64.25 | 350 | Bears |
| 34 | 3 | Drew Lock | 61.38 | 62.48 | 63.19 | 510 | Broncos |
| 35 | 4 | Kyle Allen | 60.53 | 58.39 | 70.04 | 107 | Commanders |
| 36 | 5 | Tua Tagovailoa | 60.45 | 62.98 | 63.29 | 347 | Dolphins |
| 37 | 6 | Nick Mullens | 60.16 | 58.26 | 67.54 | 365 | 49ers |
| 38 | 7 | Alex Smith | 60.05 | 65.91 | 62.15 | 289 | Commanders |
| 39 | 8 | Mike Glennon | 59.68 | 65.19 | 63.54 | 198 | Jaguars |
| 40 | 9 | Joe Flacco | 58.63 | 61.21 | 63.73 | 160 | Jets |
| 41 | 10 | Jalen Hurts | 58.59 | 58.56 | 63.58 | 197 | Eagles |
| 42 | 11 | Brandon Allen | 57.66 | 56.82 | 62.24 | 167 | Bengals |
| 43 | 12 | Sam Darnold | 57.43 | 57.32 | 57.90 | 449 | Jets |
| 44 | 13 | Jake Luton | 54.58 | 49.76 | 55.56 | 123 | Jaguars |

## S — Safety

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Adrian Amos | 91.48 | 91.50 | 87.30 | 1008 | Packers |
| 2 | 2 | Jessie Bates III | 90.30 | 90.00 | 86.34 | 1050 | Bengals |
| 3 | 3 | John Johnson III | 87.13 | 85.60 | 87.11 | 1025 | Rams |
| 4 | 4 | Marcus Maye | 85.14 | 85.80 | 82.61 | 1137 | Jets |
| 5 | 5 | Minkah Fitzpatrick | 82.41 | 80.10 | 79.78 | 1021 | Steelers |
| 6 | 6 | Kareem Jackson | 82.01 | 81.30 | 79.25 | 1083 | Broncos |
| 7 | 7 | Justin Simmons | 81.78 | 79.00 | 79.46 | 1088 | Broncos |
| 8 | 8 | Jordan Poyer | 80.66 | 76.40 | 79.34 | 1010 | Bills |
| 9 | 9 | Harrison Smith | 80.55 | 76.70 | 79.26 | 1030 | Vikings |

### Good (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Marcus Williams | 79.90 | 72.40 | 82.09 | 880 | Saints |
| 11 | 2 | Darnell Savage | 77.74 | 76.70 | 75.70 | 876 | Packers |
| 12 | 3 | Terrell Edmunds | 77.61 | 76.00 | 75.03 | 865 | Steelers |
| 13 | 4 | Jimmie Ward | 76.65 | 72.50 | 78.69 | 851 | 49ers |
| 14 | 5 | Chuck Clark | 75.50 | 69.50 | 76.06 | 1063 | Ravens |
| 15 | 6 | Rodney McLeod | 75.33 | 77.30 | 74.11 | 873 | Eagles |
| 16 | 7 | Jeff Heath | 75.19 | 70.57 | 76.60 | 415 | Raiders |
| 17 | 8 | Budda Baker | 74.47 | 68.90 | 74.53 | 1005 | Cardinals |
| 18 | 9 | Ronnie Harrison | 74.12 | 69.04 | 76.99 | 325 | Browns |
| 19 | 10 | Khari Willis | 74.07 | 69.10 | 75.30 | 842 | Colts |

### Starter (47 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Marcus Epps | 73.84 | 69.84 | 77.02 | 365 | Eagles |
| 21 | 2 | Kevin Byard | 73.48 | 63.90 | 75.70 | 1103 | Titans |
| 22 | 3 | Jeremy Reaves | 72.70 | 69.54 | 81.47 | 263 | Commanders |
| 23 | 4 | Mike Edwards | 72.35 | 68.23 | 73.93 | 189 | Buccaneers |
| 24 | 5 | Vonn Bell | 72.09 | 64.60 | 73.85 | 1061 | Bengals |
| 25 | 6 | Andrew Wingard | 71.82 | 67.33 | 75.60 | 461 | Jaguars |
| 26 | 7 | Amani Hooker | 71.54 | 68.15 | 69.63 | 470 | Titans |
| 27 | 8 | Micah Hyde | 71.54 | 66.80 | 71.26 | 938 | Bills |
| 28 | 9 | Tashaun Gipson Sr. | 71.18 | 64.70 | 71.97 | 1054 | Bears |
| 29 | 10 | Anthony Harris | 70.73 | 63.60 | 72.99 | 1074 | Vikings |
| 30 | 11 | Devin McCourty | 70.35 | 64.30 | 70.21 | 960 | Patriots |
| 31 | 12 | Rayshawn Jenkins | 70.18 | 70.00 | 67.90 | 860 | Chargers |
| 32 | 13 | Donovan Wilson | 70.17 | 69.48 | 74.79 | 673 | Cowboys |
| 33 | 14 | Duron Harmon | 69.43 | 64.40 | 68.61 | 1102 | Lions |
| 34 | 15 | Eric Rowe | 69.26 | 64.10 | 68.54 | 919 | Dolphins |
| 35 | 16 | Jalen Mills | 69.20 | 66.90 | 67.09 | 1014 | Eagles |
| 36 | 17 | Taylor Rapp | 69.14 | 66.05 | 71.99 | 365 | Rams |
| 37 | 18 | Jarrod Wilson | 68.77 | 64.60 | 70.30 | 765 | Jaguars |
| 38 | 19 | Jordan Whitehead | 68.71 | 63.70 | 68.71 | 920 | Buccaneers |
| 39 | 20 | Justin Reid | 68.69 | 67.20 | 67.40 | 888 | Texans |
| 40 | 21 | Adrian Phillips | 68.10 | 62.09 | 70.76 | 747 | Patriots |
| 41 | 22 | DeShon Elliott | 68.07 | 66.20 | 68.90 | 1044 | Ravens |
| 42 | 23 | Tavon Wilson | 67.73 | 62.98 | 70.28 | 219 | Colts |
| 43 | 24 | Quandre Diggs | 67.70 | 58.60 | 71.48 | 1075 | Seahawks |
| 44 | 25 | Tyrann Mathieu | 67.49 | 60.40 | 68.57 | 982 | Chiefs |
| 45 | 26 | Xavier Woods | 67.28 | 62.50 | 67.55 | 990 | Cowboys |
| 46 | 27 | Bobby McCain | 66.95 | 63.10 | 68.09 | 923 | Dolphins |
| 47 | 28 | Keanu Neal | 66.65 | 66.20 | 70.49 | 917 | Falcons |
| 48 | 29 | Jeremy Chinn | 66.44 | 64.40 | 64.66 | 967 | Panthers |
| 49 | 30 | Jalen Thompson | 66.44 | 64.45 | 72.31 | 232 | Cardinals |
| 50 | 31 | Chris Banjo | 66.37 | 62.62 | 72.30 | 436 | Cardinals |
| 51 | 32 | Ricardo Allen | 65.34 | 60.09 | 69.45 | 604 | Falcons |
| 52 | 33 | Xavier McKinney | 65.13 | 64.65 | 74.70 | 211 | Giants |
| 53 | 34 | Antoine Winfield Jr. | 65.04 | 55.00 | 67.57 | 1034 | Buccaneers |
| 54 | 35 | Jaquiski Tartt | 64.76 | 62.81 | 69.50 | 374 | 49ers |
| 55 | 36 | Kenny Vaccaro | 64.56 | 63.60 | 63.21 | 871 | Titans |
| 56 | 37 | Brandon Jones | 64.53 | 59.45 | 63.75 | 385 | Dolphins |
| 57 | 38 | Jordan Fuller | 64.39 | 58.70 | 68.19 | 708 | Rams |
| 58 | 39 | D.J. Swearinger Sr. | 63.95 | 60.51 | 68.12 | 124 | Saints |
| 59 | 40 | Malcolm Jenkins | 63.51 | 52.00 | 67.01 | 1036 | Saints |
| 60 | 41 | Damontae Kazee | 63.49 | 61.23 | 68.65 | 241 | Falcons |
| 61 | 42 | Sharrod Neasman | 63.41 | 59.64 | 66.66 | 292 | Falcons |
| 62 | 43 | Eddie Jackson | 63.32 | 55.80 | 64.58 | 1059 | Bears |
| 63 | 44 | Jabrill Peppers | 63.06 | 57.20 | 64.88 | 912 | Giants |
| 64 | 45 | Nick Scott | 62.49 | 59.85 | 70.11 | 193 | Rams |
| 65 | 46 | Erik Harris | 62.40 | 57.98 | 62.74 | 724 | Raiders |
| 66 | 47 | Julian Love | 62.15 | 55.23 | 63.63 | 722 | Giants |

### Rotation/backup (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 67 | 1 | Troy Apke | 61.97 | 58.54 | 65.40 | 441 | Commanders |
| 68 | 2 | Juan Thornhill | 61.96 | 55.60 | 62.04 | 765 | Chiefs |
| 69 | 3 | Julian Blackmon | 61.83 | 54.50 | 63.59 | 916 | Colts |
| 70 | 4 | Deshazor Everett | 61.35 | 59.54 | 64.84 | 357 | Commanders |
| 71 | 5 | Tarvarius Moore | 61.31 | 56.84 | 63.04 | 541 | 49ers |
| 72 | 6 | Will Harris | 60.91 | 57.23 | 59.20 | 312 | Lions |
| 73 | 7 | Will Redmond | 60.72 | 61.60 | 60.65 | 340 | Packers |
| 74 | 8 | Jamal Adams | 60.67 | 53.20 | 64.19 | 784 | Seahawks |
| 75 | 9 | Jahleel Addae | 60.54 | 57.68 | 61.42 | 211 | Chargers |
| 76 | 10 | Kyle Dugger | 60.30 | 56.35 | 60.85 | 520 | Patriots |
| 77 | 11 | Adrian Colbert | 60.11 | 60.00 | 67.49 | 104 | Giants |
| 78 | 12 | Juston Burris | 59.63 | 59.51 | 58.67 | 790 | Panthers |
| 79 | 13 | Daniel Sorensen | 59.52 | 58.50 | 58.44 | 883 | Chiefs |
| 80 | 14 | Tre Boston | 58.57 | 49.90 | 60.60 | 1037 | Panthers |
| 81 | 15 | Landon Collins | 58.20 | 56.81 | 60.79 | 398 | Commanders |
| 82 | 16 | Raven Greene | 58.05 | 60.06 | 62.75 | 324 | Packers |
| 83 | 17 | Eric Murray | 57.84 | 50.40 | 59.89 | 941 | Texans |
| 84 | 18 | Daniel Thomas | 57.48 | 61.06 | 64.34 | 162 | Jaguars |
| 85 | 19 | Armani Watts | 57.36 | 59.88 | 62.44 | 102 | Chiefs |
| 86 | 20 | Matthias Farley | 57.04 | 60.25 | 62.09 | 201 | Jets |
| 87 | 21 | Deionte Thompson | 56.93 | 56.83 | 59.46 | 332 | Cardinals |
| 88 | 22 | Sam Franklin Jr. | 56.84 | 55.37 | 60.96 | 251 | Panthers |
| 89 | 23 | Karl Joseph | 56.12 | 49.62 | 60.65 | 660 | Browns |
| 90 | 24 | Dean Marlowe | 55.76 | 55.83 | 59.04 | 230 | Bills |
| 91 | 25 | K'Von Wallace | 55.39 | 57.02 | 54.31 | 203 | Eagles |
| 92 | 26 | Ashtyn Davis | 55.29 | 49.60 | 63.25 | 402 | Jets |
| 93 | 27 | Darian Thompson | 54.94 | 53.37 | 57.13 | 479 | Cowboys |
| 94 | 28 | Terrence Brooks | 54.92 | 52.95 | 56.65 | 254 | Patriots |
| 95 | 29 | Sheldrick Redwine | 54.57 | 55.08 | 57.10 | 276 | Browns |
| 96 | 30 | Bradley McDougald | 53.25 | 45.65 | 59.15 | 432 | Jets |
| 97 | 31 | Tracy Walker III | 52.37 | 41.30 | 57.05 | 755 | Lions |
| 98 | 32 | Nasir Adderley | 51.80 | 42.50 | 59.95 | 886 | Chargers |
| 99 | 33 | Josh Jones | 51.52 | 40.84 | 57.81 | 700 | Jaguars |
| 100 | 34 | Marcell Harris | 50.61 | 44.80 | 56.77 | 348 | 49ers |
| 101 | 35 | Andrew Sendejo | 49.98 | 40.90 | 56.14 | 918 | Browns |
| 102 | 36 | Johnathan Abram | 47.94 | 40.00 | 56.88 | 856 | Raiders |

## T — Tackle

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Trent Williams | 96.19 | 91.63 | 95.06 | 957 | 49ers |
| 2 | 2 | Garett Bolles | 96.02 | 90.60 | 95.47 | 1015 | Broncos |
| 3 | 3 | David Bakhtiari | 95.75 | 89.46 | 95.78 | 758 | Packers |
| 4 | 4 | D.J. Humphries | 94.18 | 88.30 | 93.94 | 1129 | Cardinals |
| 5 | 5 | Andrew Whitworth | 93.18 | 84.92 | 94.52 | 600 | Rams |
| 6 | 6 | Duane Brown | 91.56 | 87.30 | 90.24 | 1048 | Seahawks |
| 7 | 7 | Terron Armstead | 90.70 | 85.24 | 90.17 | 857 | Saints |
| 8 | 8 | Jack Conklin | 90.32 | 84.30 | 90.17 | 999 | Browns |
| 9 | 9 | Ryan Ramczyk | 88.61 | 81.50 | 89.19 | 1038 | Saints |
| 10 | 10 | Tristan Wirfs | 88.18 | 81.80 | 88.26 | 1073 | Buccaneers |
| 11 | 11 | Morgan Moses | 88.17 | 80.60 | 89.05 | 1065 | Commanders |
| 12 | 12 | Taylor Moton | 88.07 | 81.60 | 88.22 | 1032 | Panthers |
| 13 | 13 | Taylor Decker | 87.94 | 82.00 | 87.73 | 1048 | Lions |
| 14 | 14 | Mike McGlinchey | 87.81 | 79.60 | 89.12 | 1091 | 49ers |
| 15 | 15 | Rob Havenstein | 87.76 | 80.20 | 88.64 | 1117 | Rams |
| 16 | 16 | Braden Smith | 87.23 | 79.80 | 88.02 | 937 | Colts |
| 17 | 17 | Brian O'Neill | 86.91 | 78.00 | 88.68 | 1070 | Vikings |
| 18 | 18 | Isaiah Wynn | 86.84 | 78.23 | 88.42 | 641 | Patriots |
| 19 | 19 | Eric Fisher | 85.79 | 80.00 | 85.49 | 1049 | Chiefs |
| 20 | 20 | Ronnie Stanley | 85.27 | 74.68 | 88.16 | 312 | Ravens |
| 21 | 21 | Dion Dawkins | 84.96 | 78.10 | 85.36 | 1034 | Bills |
| 22 | 22 | Orlando Brown Jr. | 84.26 | 76.50 | 85.26 | 1027 | Ravens |
| 23 | 23 | Rick Wagner | 83.34 | 74.83 | 84.84 | 610 | Packers |
| 24 | 24 | Charles Leno Jr. | 82.91 | 74.60 | 84.28 | 1066 | Bears |
| 25 | 25 | Laremy Tunsil | 82.88 | 74.58 | 84.24 | 817 | Texans |
| 26 | 26 | Mekhi Becton | 82.25 | 72.06 | 84.88 | 691 | Jets |
| 27 | 27 | Jake Matthews | 81.97 | 75.50 | 82.11 | 1113 | Falcons |
| 28 | 28 | Cornelius Lucas | 81.95 | 73.43 | 83.47 | 536 | Commanders |
| 29 | 29 | Donovan Smith | 81.61 | 71.66 | 84.07 | 962 | Buccaneers |
| 30 | 30 | Mitchell Schwartz | 80.87 | 71.19 | 83.16 | 357 | Chiefs |
| 31 | 31 | Alejandro Villanueva | 80.87 | 74.60 | 80.88 | 1098 | Steelers |
| 32 | 32 | Lane Johnson | 80.70 | 69.34 | 84.10 | 405 | Eagles |
| 33 | 33 | Russell Okung | 80.55 | 70.21 | 83.27 | 406 | Panthers |
| 34 | 34 | Anthony Castonzo | 80.44 | 72.37 | 81.66 | 749 | Colts |
| 35 | 35 | Brandon Shell | 80.17 | 70.50 | 82.45 | 673 | Seahawks |
| 36 | 36 | Riley Reiff | 80.15 | 71.30 | 81.89 | 1003 | Vikings |

### Good (25 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 37 | 1 | Demar Dotson | 79.95 | 68.70 | 83.29 | 451 | Broncos |
| 38 | 2 | James Hurst | 79.77 | 66.25 | 84.62 | 377 | Saints |
| 39 | 3 | Kolton Miller | 79.70 | 72.84 | 80.10 | 961 | Raiders |
| 40 | 4 | Mike Remmers | 79.44 | 68.57 | 82.52 | 709 | Chiefs |
| 41 | 5 | Kelvin Beachum | 78.98 | 69.00 | 81.47 | 1126 | Cardinals |
| 42 | 6 | Jordan Mailata | 78.92 | 68.97 | 81.39 | 733 | Eagles |
| 43 | 7 | Bobby Massie | 78.82 | 68.70 | 81.40 | 470 | Bears |
| 44 | 8 | Kendall Lamm | 78.62 | 66.72 | 82.39 | 113 | Browns |
| 45 | 9 | Trent Brown | 78.50 | 66.42 | 82.38 | 282 | Raiders |
| 46 | 10 | Rashod Hill | 77.95 | 64.88 | 82.49 | 121 | Vikings |
| 47 | 11 | Dennis Kelly | 77.52 | 65.90 | 81.10 | 1049 | Titans |
| 48 | 12 | Matt Peart | 76.97 | 63.88 | 81.53 | 150 | Giants |
| 49 | 13 | Jonah Williams | 76.95 | 68.10 | 78.69 | 634 | Bengals |
| 50 | 14 | Bobby Hart | 76.91 | 65.93 | 80.06 | 872 | Bengals |
| 51 | 15 | Roderick Johnson | 76.70 | 62.24 | 82.18 | 245 | Texans |
| 52 | 16 | Tyron Smith | 76.69 | 64.99 | 80.33 | 154 | Cowboys |
| 53 | 17 | Ty Sambrailo | 76.14 | 63.63 | 80.31 | 415 | Titans |
| 54 | 18 | Kaleb McGary | 76.03 | 64.09 | 79.82 | 890 | Falcons |
| 55 | 19 | Chuma Edoga | 75.93 | 60.63 | 81.96 | 235 | Jets |
| 56 | 20 | Jason Peters | 75.91 | 66.32 | 78.14 | 509 | Eagles |
| 57 | 21 | Cedric Ogbuehi | 75.88 | 62.92 | 80.36 | 277 | Seahawks |
| 58 | 22 | Tyrell Crosby | 74.81 | 63.43 | 78.23 | 657 | Lions |
| 59 | 23 | Tytus Howard | 74.73 | 61.91 | 79.11 | 811 | Texans |
| 60 | 24 | Jesse Davis | 74.51 | 62.60 | 78.28 | 1055 | Dolphins |
| 61 | 25 | Jedrick Wills Jr. | 74.33 | 61.48 | 78.73 | 957 | Browns |

### Starter (31 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 62 | 1 | Justin Herron | 73.58 | 62.03 | 77.11 | 352 | Patriots |
| 63 | 2 | Taylor Lewan | 73.57 | 61.25 | 77.61 | 239 | Titans |
| 64 | 3 | Chaz Green | 73.31 | 60.88 | 77.43 | 209 | Colts |
| 65 | 4 | Josh Wells | 73.22 | 62.04 | 76.50 | 111 | Buccaneers |
| 66 | 5 | David Quessenberry | 73.18 | 61.13 | 77.05 | 437 | Titans |
| 67 | 6 | Andrew Thomas | 73.15 | 62.39 | 76.16 | 978 | Giants |
| 68 | 7 | Bryan Bulaga | 73.10 | 61.28 | 76.81 | 444 | Chargers |
| 69 | 8 | Cam Robinson | 73.10 | 61.69 | 76.54 | 973 | Jaguars |
| 70 | 9 | George Fant | 73.10 | 61.47 | 76.69 | 829 | Jets |
| 71 | 10 | Trent Scott | 72.87 | 60.12 | 77.21 | 347 | Panthers |
| 72 | 11 | Geron Christian | 72.66 | 61.46 | 75.96 | 398 | Commanders |
| 73 | 12 | Sam Young | 72.54 | 59.38 | 77.14 | 382 | Raiders |
| 74 | 13 | Cam Fleming | 72.41 | 58.46 | 77.54 | 913 | Giants |
| 75 | 14 | Chukwuma Okorafor | 71.05 | 57.50 | 75.91 | 1033 | Steelers |
| 76 | 15 | Conor McDermott | 70.71 | 57.10 | 75.62 | 247 | Jets |
| 77 | 16 | Jawaan Taylor | 70.69 | 56.50 | 75.99 | 1037 | Jaguars |
| 78 | 17 | Greg Little | 70.37 | 53.64 | 77.36 | 134 | Panthers |
| 79 | 18 | Matt Gono | 70.25 | 57.02 | 74.90 | 336 | Falcons |
| 80 | 19 | Trey Pipkins III | 69.84 | 56.04 | 74.87 | 571 | Chargers |
| 81 | 20 | David Sharpe | 69.11 | 56.07 | 73.63 | 184 | Commanders |
| 82 | 21 | Hakeem Adeniji | 68.62 | 55.91 | 72.92 | 233 | Bengals |
| 83 | 22 | Matt Nelson | 68.35 | 57.92 | 71.14 | 242 | Lions |
| 84 | 23 | Sam Tevi | 68.05 | 52.90 | 73.98 | 1024 | Chargers |
| 85 | 24 | Brandon Parker | 67.62 | 53.43 | 72.91 | 345 | Raiders |
| 86 | 25 | Le'Raven Clark | 67.57 | 55.64 | 71.36 | 148 | Colts |
| 87 | 26 | Austin Jackson | 67.28 | 53.04 | 72.61 | 848 | Dolphins |
| 88 | 27 | Justin Skule | 66.87 | 52.57 | 72.23 | 255 | 49ers |
| 89 | 28 | Fred Johnson | 66.47 | 51.81 | 72.08 | 491 | Bengals |
| 90 | 29 | Terence Steele | 66.20 | 50.37 | 72.58 | 970 | Cowboys |
| 91 | 30 | Brandon Knight | 65.59 | 49.81 | 71.95 | 774 | Cowboys |
| 92 | 31 | Calvin Anderson | 65.57 | 58.04 | 66.42 | 132 | Broncos |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Travis Kelce | 83.81 | 93.12 | 73.43 | 635 | Chiefs |
| 2 | 2 | George Kittle | 83.09 | 79.37 | 81.40 | 262 | 49ers |
| 3 | 3 | Mark Andrews | 82.00 | 78.43 | 80.21 | 364 | Ravens |
| 4 | 4 | Darren Waller | 81.84 | 86.01 | 74.89 | 620 | Raiders |

### Good (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 5 | 1 | Mo Alie-Cox | 79.13 | 70.65 | 80.62 | 226 | Colts |
| 6 | 2 | Rob Gronkowski | 78.89 | 71.15 | 79.89 | 498 | Buccaneers |
| 7 | 3 | Pharaoh Brown | 75.90 | 69.31 | 76.12 | 126 | Texans |
| 8 | 4 | Dallas Goedert | 75.26 | 76.88 | 70.01 | 390 | Eagles |
| 9 | 5 | Donald Parham Jr. | 74.61 | 62.04 | 78.82 | 136 | Chargers |

### Starter (64 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Will Dissly | 73.91 | 64.17 | 76.23 | 308 | Seahawks |
| 11 | 2 | Dan Arnold | 73.91 | 61.17 | 78.24 | 312 | Cardinals |
| 12 | 3 | Jordan Reed | 73.82 | 64.48 | 75.88 | 190 | 49ers |
| 13 | 4 | Adam Trautman | 73.32 | 69.90 | 71.44 | 172 | Saints |
| 14 | 5 | Mike Gesicki | 73.12 | 75.32 | 67.48 | 473 | Dolphins |
| 15 | 6 | Foster Moreau | 73.10 | 63.03 | 75.64 | 117 | Raiders |
| 16 | 7 | Anthony Firkser | 73.06 | 70.52 | 70.59 | 250 | Titans |
| 17 | 8 | Richard Rodgers | 72.86 | 74.68 | 67.48 | 174 | Eagles |
| 18 | 9 | Hunter Henry | 72.66 | 68.98 | 70.94 | 586 | Chargers |
| 19 | 10 | T.J. Hockenson | 72.63 | 73.99 | 67.56 | 538 | Lions |
| 20 | 11 | Tyler Higbee | 72.17 | 68.10 | 70.71 | 424 | Rams |
| 21 | 12 | Darren Fells | 71.88 | 66.54 | 71.27 | 313 | Texans |
| 22 | 13 | Robert Tonyan | 71.80 | 66.43 | 71.21 | 426 | Packers |
| 23 | 14 | Greg Olsen | 71.64 | 61.63 | 74.15 | 316 | Seahawks |
| 24 | 15 | Kyle Rudolph | 71.50 | 65.21 | 71.53 | 296 | Vikings |
| 25 | 16 | David Njoku | 71.42 | 63.95 | 72.23 | 216 | Browns |
| 26 | 17 | Noah Fant | 71.41 | 69.41 | 68.58 | 466 | Broncos |
| 27 | 18 | Jared Cook | 71.34 | 69.87 | 68.16 | 338 | Saints |
| 28 | 19 | Jack Doyle | 71.14 | 65.35 | 70.83 | 244 | Colts |
| 29 | 20 | Marcedes Lewis | 71.00 | 65.78 | 70.31 | 191 | Packers |
| 30 | 21 | Jonnu Smith | 70.94 | 71.12 | 66.65 | 353 | Titans |
| 31 | 22 | Trey Burton | 70.91 | 64.06 | 71.31 | 242 | Colts |
| 32 | 23 | Austin Hooper | 70.45 | 67.03 | 68.56 | 366 | Browns |
| 33 | 24 | Zach Ertz | 69.93 | 57.62 | 73.97 | 424 | Eagles |
| 34 | 25 | Jimmy Graham | 69.76 | 62.41 | 70.49 | 445 | Bears |
| 35 | 26 | Blake Bell | 69.74 | 61.50 | 71.07 | 154 | Cowboys |
| 36 | 27 | Jordan Akins | 69.70 | 67.97 | 66.68 | 301 | Texans |
| 37 | 28 | Tyler Kroft | 69.19 | 61.94 | 69.85 | 135 | Bills |
| 38 | 29 | Adam Shaheen | 69.12 | 62.95 | 69.06 | 160 | Dolphins |
| 39 | 30 | Chris Herndon | 68.98 | 58.04 | 72.11 | 438 | Jets |
| 40 | 31 | Cameron Brate | 68.97 | 65.68 | 66.99 | 198 | Buccaneers |
| 41 | 32 | Durham Smythe | 68.35 | 64.87 | 66.50 | 207 | Dolphins |
| 42 | 33 | Evan Engram | 68.33 | 60.56 | 69.34 | 567 | Giants |
| 43 | 34 | Dawson Knox | 68.26 | 60.86 | 69.02 | 249 | Bills |
| 44 | 35 | Gerald Everett | 68.17 | 62.05 | 68.09 | 353 | Rams |
| 45 | 36 | Jesse James | 68.14 | 60.72 | 68.92 | 237 | Lions |
| 46 | 37 | Irv Smith Jr. | 67.87 | 66.98 | 64.29 | 322 | Vikings |
| 47 | 38 | Nick Boyle | 67.50 | 65.07 | 64.95 | 151 | Ravens |
| 48 | 39 | Tyler Eifert | 67.38 | 61.93 | 66.84 | 417 | Jaguars |
| 49 | 40 | Harrison Bryant | 67.31 | 60.47 | 67.70 | 294 | Browns |
| 50 | 41 | Jason Witten | 67.05 | 60.57 | 67.21 | 179 | Raiders |
| 51 | 42 | Devin Asiasi | 66.65 | 56.59 | 69.19 | 109 | Patriots |
| 52 | 43 | Drew Sample | 66.60 | 61.23 | 66.02 | 513 | Bengals |
| 53 | 44 | Eric Ebron | 66.55 | 56.12 | 69.33 | 536 | Steelers |
| 54 | 45 | Maxx Williams | 66.42 | 63.31 | 64.32 | 122 | Cardinals |
| 55 | 46 | Darrell Daniels | 66.40 | 59.62 | 66.76 | 117 | Cardinals |
| 56 | 47 | James O'Shaughnessy | 66.34 | 58.55 | 67.37 | 264 | Jaguars |
| 57 | 48 | Logan Thomas | 66.31 | 64.50 | 63.35 | 661 | Commanders |
| 58 | 49 | Dalton Schultz | 66.27 | 63.78 | 63.76 | 620 | Cowboys |
| 59 | 50 | Ross Dwelley | 66.22 | 60.77 | 65.69 | 234 | 49ers |
| 60 | 51 | Hayden Hurst | 65.91 | 58.96 | 66.37 | 588 | Falcons |
| 61 | 52 | Cole Kmet | 65.81 | 59.09 | 66.12 | 326 | Bears |
| 62 | 53 | Kaden Smith | 65.62 | 63.83 | 62.64 | 203 | Giants |
| 63 | 54 | Ryan Izzo | 65.35 | 54.75 | 68.25 | 292 | Patriots |
| 64 | 55 | Nick Vannett | 65.31 | 58.65 | 65.58 | 154 | Broncos |
| 65 | 56 | Geoff Swaim | 65.16 | 60.96 | 63.80 | 126 | Titans |
| 66 | 57 | Ryan Griffin | 64.90 | 58.10 | 65.26 | 157 | Jets |
| 67 | 58 | Josh Hill | 64.85 | 61.25 | 63.08 | 141 | Saints |
| 68 | 59 | Levine Toilolo | 64.73 | 58.60 | 64.65 | 102 | Giants |
| 69 | 60 | Chris Manhertz | 64.72 | 58.90 | 64.43 | 220 | Panthers |
| 70 | 61 | Demetrius Harris | 64.64 | 59.39 | 63.97 | 109 | Bears |
| 71 | 62 | Jacob Hollister | 64.56 | 61.44 | 62.47 | 220 | Seahawks |
| 72 | 63 | Tyler Conklin | 63.66 | 55.38 | 65.01 | 257 | Vikings |
| 73 | 64 | Nick Keizer | 62.76 | 54.52 | 64.09 | 127 | Chiefs |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 74 | 1 | Vance McDonald | 61.53 | 51.33 | 64.16 | 239 | Steelers |
| 75 | 2 | Ian Thomas | 60.49 | 45.01 | 66.64 | 501 | Panthers |
| 76 | 3 | Luke Stocker | 59.50 | 53.26 | 59.49 | 212 | Falcons |

## WR — Wide Receiver

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Justin Jefferson | 88.47 | 88.08 | 84.57 | 563 | Vikings |
| 2 | 2 | A.J. Brown | 87.85 | 86.60 | 84.51 | 429 | Titans |
| 3 | 3 | Davante Adams | 86.48 | 89.51 | 80.30 | 489 | Packers |
| 4 | 4 | Stefon Diggs | 85.37 | 89.37 | 78.53 | 658 | Bills |
| 5 | 5 | DeAndre Hopkins | 84.75 | 87.10 | 79.01 | 663 | Cardinals |
| 6 | 6 | Julio Jones | 84.10 | 81.40 | 81.74 | 314 | Falcons |
| 7 | 7 | Tyreek Hill | 83.89 | 82.78 | 80.46 | 621 | Chiefs |
| 8 | 8 | Allen Robinson II | 83.50 | 87.94 | 76.38 | 625 | Bears |
| 9 | 9 | Will Fuller V | 83.46 | 80.55 | 81.24 | 406 | Texans |
| 10 | 10 | Corey Davis | 83.23 | 83.29 | 79.03 | 398 | Titans |
| 11 | 11 | Antonio Brown | 82.83 | 79.83 | 80.66 | 249 | Buccaneers |
| 12 | 12 | Calvin Ridley | 82.54 | 84.12 | 77.32 | 593 | Falcons |
| 13 | 13 | Chris Godwin | 82.48 | 78.31 | 81.09 | 462 | Buccaneers |
| 14 | 14 | DJ Moore | 82.05 | 78.27 | 80.41 | 568 | Panthers |
| 15 | 15 | Adam Thielen | 81.90 | 85.78 | 75.14 | 536 | Vikings |
| 16 | 16 | D.K. Metcalf | 81.73 | 82.70 | 76.91 | 667 | Seahawks |
| 17 | 17 | Michael Thomas | 81.45 | 78.71 | 79.11 | 215 | Saints |
| 18 | 18 | Keenan Allen | 81.37 | 83.65 | 75.68 | 554 | Chargers |
| 19 | 19 | Brandin Cooks | 81.20 | 80.34 | 77.61 | 583 | Texans |
| 20 | 20 | Kenny Golladay | 81.05 | 74.82 | 81.04 | 150 | Lions |
| 21 | 21 | Jarvis Landry | 80.71 | 81.37 | 76.10 | 428 | Browns |
| 22 | 22 | Cole Beasley | 80.57 | 82.92 | 74.83 | 497 | Bills |
| 23 | 23 | Terry McLaurin | 80.36 | 77.70 | 77.96 | 624 | Commanders |
| 24 | 24 | Cooper Kupp | 80.20 | 79.63 | 76.42 | 542 | Rams |

### Good (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 25 | 1 | T.Y. Hilton | 79.92 | 75.99 | 78.38 | 465 | Colts |
| 26 | 2 | Chase Claypool | 79.24 | 73.22 | 79.08 | 474 | Steelers |
| 27 | 3 | Donovan Peoples-Jones | 79.05 | 65.40 | 83.99 | 138 | Browns |
| 28 | 4 | Mike Evans | 78.91 | 73.70 | 78.21 | 599 | Buccaneers |
| 29 | 5 | Amari Cooper | 78.89 | 75.74 | 76.82 | 638 | Cowboys |
| 30 | 6 | Tim Patrick | 78.03 | 72.04 | 77.85 | 455 | Broncos |
| 31 | 7 | Mike Williams | 78.00 | 71.80 | 77.96 | 514 | Chargers |
| 32 | 8 | Tee Higgins | 77.94 | 74.21 | 76.26 | 527 | Bengals |
| 33 | 9 | Deebo Samuel | 77.91 | 74.07 | 76.31 | 177 | 49ers |
| 34 | 10 | Rashard Higgins | 77.75 | 71.85 | 77.52 | 306 | Browns |
| 35 | 11 | DeVante Parker | 77.63 | 74.99 | 75.23 | 484 | Dolphins |
| 36 | 12 | Brandon Aiyuk | 77.62 | 76.69 | 74.07 | 455 | 49ers |
| 37 | 13 | DeSean Jackson | 77.60 | 64.56 | 82.13 | 134 | Eagles |
| 38 | 14 | Bryan Edwards | 77.54 | 64.42 | 82.12 | 143 | Raiders |
| 39 | 15 | Odell Beckham Jr. | 77.32 | 71.03 | 77.34 | 189 | Browns |
| 40 | 16 | Jakobi Meyers | 77.20 | 73.47 | 75.52 | 346 | Patriots |
| 41 | 17 | Tyler Lockett | 77.18 | 76.40 | 73.53 | 660 | Seahawks |
| 42 | 18 | Tyler Boyd | 76.87 | 74.87 | 74.04 | 537 | Bengals |
| 43 | 19 | Emmanuel Sanders | 76.72 | 72.47 | 75.39 | 378 | Saints |
| 44 | 20 | Robert Woods | 76.63 | 71.05 | 76.18 | 612 | Rams |
| 45 | 21 | Hunter Renfrow | 76.58 | 71.54 | 75.77 | 401 | Raiders |
| 46 | 22 | Jamison Crowder | 75.89 | 73.36 | 73.41 | 411 | Jets |
| 47 | 23 | Sterling Shepard | 75.86 | 76.94 | 70.97 | 395 | Giants |
| 48 | 24 | John Brown | 75.81 | 68.33 | 76.63 | 306 | Bills |
| 49 | 25 | Nelson Agholor | 75.68 | 72.47 | 73.66 | 465 | Raiders |
| 50 | 26 | Marvin Jones Jr. | 75.61 | 73.60 | 72.78 | 667 | Lions |
| 51 | 27 | Mecole Hardman Jr. | 75.52 | 67.25 | 76.86 | 361 | Chiefs |
| 52 | 28 | Danny Amendola | 75.38 | 72.49 | 73.14 | 352 | Lions |
| 53 | 29 | Denzel Mims | 75.26 | 65.95 | 77.30 | 264 | Jets |
| 54 | 30 | Collin Johnson | 75.19 | 67.00 | 76.49 | 180 | Jaguars |
| 55 | 31 | DJ Chark Jr. | 75.11 | 70.40 | 74.09 | 497 | Jaguars |
| 56 | 32 | Breshad Perriman | 74.88 | 63.11 | 78.56 | 393 | Jets |
| 57 | 33 | Richie James | 74.67 | 62.75 | 78.45 | 247 | 49ers |
| 58 | 34 | Marquise Brown | 74.39 | 68.49 | 74.16 | 466 | Ravens |
| 59 | 35 | Scott Miller | 74.31 | 64.92 | 76.41 | 336 | Buccaneers |
| 60 | 36 | CeeDee Lamb | 74.30 | 70.48 | 72.68 | 539 | Cowboys |
| 61 | 37 | David Moore | 74.12 | 65.31 | 75.82 | 314 | Seahawks |
| 62 | 38 | Curtis Samuel | 74.05 | 73.88 | 69.99 | 435 | Panthers |

### Starter (87 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 63 | 1 | Randall Cobb | 73.98 | 69.15 | 73.03 | 277 | Texans |
| 64 | 2 | Quintez Cephus | 73.95 | 62.84 | 77.19 | 231 | Lions |
| 65 | 3 | Darius Slayton | 73.87 | 67.33 | 74.06 | 583 | Giants |
| 66 | 4 | Russell Gage | 73.72 | 74.94 | 68.74 | 548 | Falcons |
| 67 | 5 | Kendrick Bourne | 73.68 | 70.19 | 71.84 | 476 | 49ers |
| 68 | 6 | Allen Lazard | 73.67 | 67.58 | 73.56 | 277 | Packers |
| 69 | 7 | Braxton Berrios | 73.60 | 66.64 | 74.07 | 189 | Jets |
| 70 | 8 | Michael Gallup | 73.43 | 66.29 | 74.02 | 655 | Cowboys |
| 71 | 9 | Equanimeous St. Brown | 73.35 | 60.16 | 77.97 | 107 | Packers |
| 72 | 10 | Jerry Jeudy | 73.28 | 64.73 | 74.82 | 545 | Broncos |
| 73 | 11 | KhaDarel Hodge | 73.20 | 64.74 | 74.68 | 140 | Browns |
| 74 | 12 | Olamide Zaccheaus | 73.08 | 64.54 | 74.61 | 184 | Falcons |
| 75 | 13 | Laviska Shenault Jr. | 72.95 | 68.86 | 71.51 | 392 | Jaguars |
| 76 | 14 | Gabe Davis | 72.92 | 64.30 | 74.50 | 508 | Bills |
| 77 | 15 | Willie Snead IV | 72.85 | 69.30 | 71.05 | 296 | Ravens |
| 78 | 16 | Zach Pascal | 72.77 | 63.88 | 74.53 | 501 | Colts |
| 79 | 17 | JuJu Smith-Schuster | 72.60 | 68.20 | 71.36 | 687 | Steelers |
| 80 | 18 | Jalen Reagor | 72.51 | 62.79 | 74.82 | 321 | Eagles |
| 81 | 19 | Travis Fulgham | 72.51 | 68.58 | 70.96 | 387 | Eagles |
| 82 | 20 | Josh Reynolds | 72.43 | 65.02 | 73.20 | 511 | Rams |
| 83 | 21 | A.J. Green | 72.37 | 65.93 | 72.50 | 538 | Bengals |
| 84 | 22 | Golden Tate | 72.09 | 64.90 | 72.71 | 318 | Giants |
| 85 | 23 | Adam Humphries | 72.01 | 66.96 | 71.21 | 168 | Titans |
| 86 | 24 | Julian Edelman | 72.00 | 66.24 | 71.68 | 175 | Patriots |
| 87 | 25 | Diontae Johnson | 71.98 | 68.64 | 70.04 | 557 | Steelers |
| 88 | 26 | Marquez Valdes-Scantling | 71.80 | 58.00 | 76.84 | 497 | Packers |
| 89 | 27 | Darnell Mooney | 71.80 | 67.85 | 70.27 | 537 | Bears |
| 90 | 28 | Cam Sims | 71.76 | 61.02 | 74.76 | 410 | Commanders |
| 91 | 29 | Auden Tate | 71.71 | 65.36 | 71.77 | 101 | Bengals |
| 92 | 30 | Chris Conley | 71.58 | 66.98 | 70.48 | 315 | Jaguars |
| 93 | 31 | Keke Coutee | 71.55 | 64.82 | 71.87 | 228 | Texans |
| 94 | 32 | Byron Pringle | 71.51 | 62.29 | 73.49 | 163 | Chiefs |
| 95 | 33 | Sammy Watkins | 71.50 | 63.71 | 72.53 | 361 | Chiefs |
| 96 | 34 | Michael Pittman Jr. | 71.37 | 61.99 | 73.45 | 385 | Colts |
| 97 | 35 | Marquez Callaway | 71.25 | 64.34 | 71.69 | 150 | Saints |
| 98 | 36 | Miles Boykin | 71.25 | 61.60 | 73.51 | 272 | Ravens |
| 99 | 37 | Isaiah McKenzie | 71.16 | 65.75 | 70.60 | 149 | Bills |
| 100 | 38 | Preston Williams | 70.86 | 63.18 | 71.81 | 221 | Dolphins |
| 101 | 39 | Jakeem Grant Sr. | 70.84 | 66.67 | 69.46 | 247 | Dolphins |
| 102 | 40 | Christian Kirk | 70.75 | 62.08 | 72.36 | 540 | Cardinals |
| 103 | 41 | James Washington | 70.73 | 60.63 | 73.29 | 322 | Steelers |
| 104 | 42 | Deonte Harty | 70.50 | 65.05 | 69.96 | 125 | Saints |
| 105 | 43 | Henry Ruggs III | 70.29 | 55.53 | 75.96 | 366 | Raiders |
| 106 | 44 | Marvin Hall | 70.09 | 58.37 | 73.74 | 258 | Browns |
| 107 | 45 | Tyler Johnson | 69.67 | 59.44 | 72.33 | 171 | Buccaneers |
| 108 | 46 | Kalif Raymond | 69.54 | 57.13 | 73.64 | 129 | Titans |
| 109 | 47 | Mohamed Sanu | 69.42 | 64.00 | 68.86 | 197 | Lions |
| 110 | 48 | Keelan Cole Sr. | 69.37 | 64.28 | 68.60 | 601 | Jaguars |
| 111 | 49 | Freddie Swain | 69.33 | 56.01 | 74.05 | 241 | Seahawks |
| 112 | 50 | Damiere Byrd | 69.10 | 61.21 | 70.20 | 496 | Patriots |
| 113 | 51 | Demarcus Robinson | 68.92 | 62.58 | 68.98 | 523 | Chiefs |
| 114 | 52 | Van Jefferson | 68.78 | 63.48 | 68.14 | 159 | Rams |
| 115 | 53 | Noah Brown | 68.76 | 63.12 | 68.35 | 143 | Cowboys |
| 116 | 54 | Olabisi Johnson | 68.70 | 63.31 | 68.13 | 148 | Vikings |
| 117 | 55 | Tre'Quan Smith | 68.62 | 59.92 | 70.26 | 425 | Saints |
| 118 | 56 | Cedrick Wilson Jr. | 68.35 | 60.54 | 69.39 | 159 | Cowboys |
| 119 | 57 | Larry Fitzgerald | 68.15 | 59.64 | 69.66 | 473 | Cardinals |
| 120 | 58 | Alex Erickson | 67.99 | 57.23 | 70.99 | 116 | Bengals |
| 121 | 59 | Devin Duvernay | 67.78 | 60.28 | 68.61 | 202 | Ravens |
| 122 | 60 | Andy Isabella | 67.61 | 58.15 | 69.75 | 221 | Cardinals |
| 123 | 61 | Anthony Miller | 67.41 | 58.77 | 69.01 | 447 | Bears |
| 124 | 62 | Zay Jones | 67.15 | 61.47 | 66.77 | 170 | Raiders |
| 125 | 63 | Jalen Guyton | 66.98 | 52.46 | 72.50 | 617 | Chargers |
| 126 | 64 | Alshon Jeffery | 66.95 | 54.01 | 71.41 | 153 | Eagles |
| 127 | 65 | Chad Beebe | 66.83 | 58.34 | 68.32 | 233 | Vikings |
| 128 | 66 | Isaiah Wright | 66.75 | 59.39 | 67.49 | 247 | Commanders |
| 129 | 67 | Cameron Batson | 66.70 | 59.07 | 67.62 | 157 | Titans |
| 130 | 68 | Chad Hansen | 66.70 | 60.39 | 66.74 | 209 | Texans |
| 131 | 69 | Isaiah Ford | 66.51 | 60.71 | 66.21 | 274 | Dolphins |
| 132 | 70 | Brandon Powell | 66.11 | 58.50 | 67.01 | 136 | Falcons |
| 133 | 71 | John Hightower | 65.78 | 54.15 | 69.37 | 250 | Eagles |
| 134 | 72 | Dontrelle Inman | 65.57 | 61.14 | 64.36 | 240 | Commanders |
| 135 | 73 | DaeSean Hamilton | 65.50 | 58.88 | 65.74 | 325 | Broncos |
| 136 | 74 | KeeSean Johnson | 65.48 | 60.56 | 64.60 | 145 | Cardinals |
| 137 | 75 | Ray-Ray McCloud III | 65.30 | 64.08 | 61.94 | 105 | Steelers |
| 138 | 76 | C.J. Board | 65.21 | 63.10 | 62.45 | 116 | Giants |
| 139 | 77 | KJ Hamler | 65.17 | 57.01 | 66.44 | 351 | Broncos |
| 140 | 78 | Mike Thomas | 65.11 | 56.74 | 66.53 | 142 | Bengals |
| 141 | 79 | N'Keal Harry | 64.85 | 58.44 | 64.96 | 332 | Patriots |
| 142 | 80 | Jamal Agnew | 64.81 | 58.55 | 64.82 | 128 | Lions |
| 143 | 81 | Mack Hollins | 64.79 | 59.11 | 64.41 | 162 | Dolphins |
| 144 | 82 | Trent Taylor | 64.60 | 57.05 | 65.46 | 154 | 49ers |
| 145 | 83 | Greg Ward | 64.31 | 58.91 | 63.75 | 543 | Eagles |
| 146 | 84 | Steven Sims | 64.21 | 57.83 | 64.29 | 286 | Commanders |
| 147 | 85 | Christian Blake | 63.63 | 55.77 | 64.70 | 156 | Falcons |
| 148 | 86 | K.J. Hill | 63.51 | 58.00 | 63.01 | 100 | Chargers |
| 149 | 87 | Jeff Smith | 62.23 | 55.43 | 62.60 | 221 | Jets |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 150 | 1 | Javon Wims | 61.80 | 57.06 | 60.79 | 153 | Bears |
