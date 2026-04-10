# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:26:45Z
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
| 1 | 1 | Corey Linsley | 95.51 | 89.90 | 95.09 | 734 | Packers |
| 2 | 2 | Brandon Linder | 88.18 | 80.00 | 89.47 | 530 | Jaguars |
| 3 | 3 | Ben Jones | 86.71 | 78.60 | 87.95 | 1042 | Titans |
| 4 | 4 | Cody Whitehair | 85.63 | 76.30 | 87.68 | 893 | Bears |
| 5 | 5 | J.C. Tretter | 85.41 | 77.30 | 86.65 | 1061 | Browns |
| 6 | 6 | Frank Ragnow | 85.25 | 80.30 | 84.38 | 929 | Lions |
| 7 | 7 | Chase Roullier | 84.98 | 76.80 | 86.26 | 1089 | Commanders |
| 8 | 8 | Rodney Hudson | 82.66 | 73.60 | 84.53 | 1082 | Raiders |
| 9 | 9 | Jason Kelce | 82.00 | 69.60 | 86.10 | 1126 | Eagles |

### Good (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Austin Reiter | 79.89 | 70.90 | 81.72 | 867 | Chiefs |
| 11 | 2 | Erik McCoy | 79.82 | 70.10 | 82.14 | 1073 | Saints |
| 12 | 3 | Ryan Kelly | 78.80 | 69.00 | 81.16 | 1007 | Colts |
| 13 | 4 | Ben Garland | 78.08 | 71.10 | 78.56 | 333 | 49ers |
| 14 | 5 | Daniel Kilgore | 78.04 | 68.60 | 80.16 | 236 | Chiefs |
| 15 | 6 | Ryan Jensen | 77.97 | 64.90 | 82.51 | 1061 | Buccaneers |
| 16 | 7 | David Andrews | 77.88 | 67.70 | 80.50 | 724 | Patriots |
| 17 | 8 | Patrick Mekari | 77.81 | 66.90 | 80.91 | 554 | Ravens |
| 18 | 9 | Alex Mack | 77.15 | 65.90 | 80.48 | 972 | Falcons |
| 19 | 10 | Mitch Morse | 76.50 | 65.80 | 79.47 | 880 | Bills |
| 20 | 11 | Ted Karras | 76.26 | 65.40 | 79.34 | 1068 | Dolphins |
| 21 | 12 | Trey Hopkins | 75.20 | 63.80 | 78.63 | 938 | Bengals |
| 22 | 13 | Trystan Colon | 74.60 | 67.60 | 75.10 | 127 | Ravens |
| 23 | 14 | Connor McGovern | 74.52 | 62.20 | 78.56 | 969 | Jets |
| 24 | 15 | Garrett Bradbury | 74.06 | 61.40 | 78.34 | 1082 | Vikings |

### Starter (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 25 | 1 | Matt Paradis | 73.89 | 63.40 | 76.71 | 1029 | Panthers |
| 26 | 2 | A.Q. Shipley | 72.44 | 56.00 | 79.23 | 157 | Buccaneers |
| 27 | 3 | Maurkice Pouncey | 71.62 | 60.50 | 74.86 | 863 | Steelers |
| 28 | 4 | Sam Mustipher | 71.53 | 58.90 | 75.78 | 504 | Bears |
| 29 | 5 | Billy Price | 69.78 | 50.70 | 78.34 | 208 | Bengals |
| 30 | 6 | Mason Cole | 69.72 | 54.40 | 75.77 | 913 | Cardinals |
| 31 | 7 | Nick Martin | 68.59 | 56.10 | 72.75 | 980 | Texans |
| 32 | 8 | Tyler Biadasz | 68.20 | 53.50 | 73.84 | 427 | Cowboys |
| 33 | 9 | Hroniss Grasu | 67.73 | 52.40 | 73.78 | 215 | 49ers |
| 34 | 10 | J.C. Hassenauer | 66.81 | 57.70 | 68.72 | 303 | Steelers |
| 35 | 11 | Matt Skura | 66.51 | 50.20 | 73.22 | 661 | Ravens |
| 36 | 12 | Matt Hennessy | 64.87 | 47.00 | 72.62 | 225 | Falcons |
| 37 | 13 | Joe Looney | 64.76 | 50.70 | 69.97 | 764 | Cowboys |
| 38 | 14 | James Ferentz | 64.19 | 54.30 | 66.61 | 162 | Patriots |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Lloyd Cushenberry III | 59.43 | 40.50 | 67.89 | 1076 | Broncos |

## CB — Cornerback

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jaire Alexander | 91.91 | 90.60 | 89.76 | 900 | Packers |
| 2 | 2 | Xavien Howard | 90.37 | 89.60 | 90.98 | 936 | Dolphins |
| 3 | 3 | Bryce Callahan | 86.64 | 86.70 | 86.19 | 655 | Broncos |
| 4 | 4 | Darious Williams | 84.46 | 79.60 | 87.43 | 824 | Rams |
| 5 | 5 | Jamel Dean | 84.37 | 79.90 | 86.84 | 711 | Buccaneers |
| 6 | 6 | Jalen Ramsey | 83.55 | 80.30 | 83.31 | 954 | Rams |
| 7 | 7 | Tre'Davious White | 83.39 | 77.90 | 84.23 | 878 | Bills |
| 8 | 8 | Jonathan Jones | 83.30 | 80.80 | 81.11 | 730 | Patriots |
| 9 | 9 | James Bradberry | 82.67 | 79.90 | 81.19 | 1021 | Giants |
| 10 | 10 | Marlon Humphrey | 82.47 | 77.60 | 82.49 | 972 | Ravens |
| 11 | 11 | Xavier Rhodes | 82.46 | 77.50 | 82.33 | 902 | Colts |
| 12 | 12 | Brian Poole | 81.12 | 79.50 | 83.36 | 483 | Jets |
| 13 | 13 | Kenny Moore II | 80.42 | 78.10 | 79.57 | 952 | Colts |
| 14 | 14 | Ronald Darby | 80.33 | 75.90 | 82.13 | 1002 | Commanders |
| 15 | 15 | Ahkello Witherspoon | 80.03 | 80.90 | 81.43 | 334 | 49ers |

### Good (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Rashad Fenton | 79.93 | 75.00 | 83.48 | 527 | Chiefs |
| 17 | 2 | Malcolm Butler | 79.20 | 72.70 | 81.55 | 1087 | Titans |
| 18 | 3 | Troy Hill | 79.05 | 75.70 | 78.69 | 973 | Rams |
| 19 | 4 | Denzel Ward | 78.84 | 74.60 | 81.47 | 776 | Browns |
| 20 | 5 | William Jackson III | 78.57 | 72.40 | 80.19 | 886 | Bengals |
| 21 | 6 | Cameron Sutton | 77.53 | 73.90 | 76.82 | 552 | Steelers |
| 22 | 7 | J.C. Jackson | 77.53 | 70.10 | 78.95 | 851 | Patriots |
| 23 | 8 | Jimmy Smith | 76.94 | 75.70 | 79.24 | 454 | Ravens |
| 24 | 9 | Jason Verrett | 76.75 | 76.10 | 79.27 | 803 | 49ers |
| 25 | 10 | Kyle Fuller | 76.58 | 70.10 | 76.74 | 1060 | Bears |
| 26 | 11 | Joe Haden | 76.49 | 69.60 | 78.16 | 846 | Steelers |
| 27 | 12 | Ross Cockrell | 76.40 | 74.20 | 79.33 | 238 | Buccaneers |
| 28 | 13 | Marcus Peters | 76.37 | 69.40 | 77.88 | 912 | Ravens |
| 29 | 14 | Bashaud Breeland | 76.20 | 72.90 | 78.72 | 690 | Chiefs |
| 30 | 15 | L'Jarius Sneed | 76.15 | 73.70 | 80.92 | 410 | Chiefs |
| 31 | 16 | Bradley Roby | 76.13 | 72.20 | 79.78 | 613 | Texans |
| 32 | 17 | Donte Jackson | 74.89 | 69.00 | 76.94 | 599 | Panthers |
| 33 | 18 | Darius Phillips | 74.59 | 68.10 | 81.00 | 593 | Bengals |
| 34 | 19 | Sidney Jones IV | 74.36 | 71.30 | 80.25 | 303 | Jaguars |
| 35 | 20 | D.J. Reed | 74.23 | 69.80 | 78.53 | 560 | Seahawks |
| 36 | 21 | Steven Nelson | 74.11 | 68.10 | 74.79 | 908 | Steelers |

### Starter (61 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 37 | 1 | Charvarius Ward | 73.57 | 66.80 | 77.47 | 782 | Chiefs |
| 38 | 2 | Josh Norman | 73.50 | 69.90 | 77.46 | 344 | Bills |
| 39 | 3 | T.J. Carrie | 73.43 | 67.00 | 74.90 | 396 | Colts |
| 40 | 4 | Janoris Jenkins | 73.38 | 66.00 | 76.02 | 805 | Saints |
| 41 | 5 | Carlton Davis III | 73.34 | 66.40 | 76.08 | 906 | Buccaneers |
| 42 | 6 | Trevon Diggs | 72.81 | 63.90 | 78.75 | 758 | Cowboys |
| 43 | 7 | Mike Hilton | 72.75 | 67.00 | 74.70 | 464 | Steelers |
| 44 | 8 | Mackensie Alexander | 72.63 | 67.20 | 74.79 | 642 | Bengals |
| 45 | 9 | Shaquill Griffin | 72.39 | 63.60 | 76.78 | 812 | Seahawks |
| 46 | 10 | Richard Sherman | 72.37 | 67.20 | 78.10 | 332 | 49ers |
| 47 | 11 | Kendall Fuller | 72.37 | 65.30 | 75.74 | 893 | Commanders |
| 48 | 12 | Michael Davis | 72.28 | 62.90 | 76.25 | 958 | Chargers |
| 49 | 13 | Darius Slay | 72.26 | 63.10 | 75.55 | 885 | Eagles |
| 50 | 14 | Breon Borders | 71.71 | 71.40 | 76.08 | 360 | Titans |
| 51 | 15 | Terrance Mitchell | 71.49 | 63.40 | 76.89 | 1070 | Browns |
| 52 | 16 | Cameron Dantzler | 71.22 | 69.80 | 73.20 | 601 | Vikings |
| 53 | 17 | Ugo Amadi | 71.14 | 67.10 | 75.26 | 552 | Seahawks |
| 54 | 18 | Chandon Sullivan | 70.90 | 65.40 | 71.96 | 729 | Packers |
| 55 | 19 | Javelin Guidry | 70.50 | 76.50 | 73.20 | 172 | Jets |
| 56 | 20 | Byron Murphy Jr. | 70.44 | 63.30 | 71.69 | 795 | Cardinals |
| 57 | 21 | Casey Hayward Jr. | 70.24 | 59.50 | 74.26 | 788 | Chargers |
| 58 | 22 | Fabian Moreau | 70.09 | 64.30 | 73.12 | 158 | Commanders |
| 59 | 23 | Johnathan Joseph | 69.67 | 63.10 | 73.53 | 423 | Cardinals |
| 60 | 24 | Rasul Douglas | 69.49 | 60.20 | 72.99 | 821 | Panthers |
| 61 | 25 | Trayvon Mullen | 69.04 | 58.30 | 72.03 | 933 | Raiders |
| 62 | 26 | K'Waun Williams | 68.97 | 64.80 | 72.49 | 284 | 49ers |
| 63 | 27 | Byron Jones | 68.97 | 61.50 | 71.14 | 814 | Dolphins |
| 64 | 28 | Adoree' Jackson | 68.94 | 66.80 | 74.53 | 155 | Titans |
| 65 | 29 | Blidi Wreh-Wilson | 68.87 | 63.20 | 74.00 | 245 | Falcons |
| 66 | 30 | Corn Elder | 68.77 | 68.20 | 66.65 | 411 | Panthers |
| 67 | 31 | Dontae Johnson | 68.09 | 65.20 | 72.84 | 273 | 49ers |
| 68 | 32 | Stephon Gilmore | 68.00 | 58.50 | 72.76 | 632 | Patriots |
| 69 | 33 | Desmond King II | 67.46 | 60.90 | 68.70 | 709 | Titans |
| 70 | 34 | Jaylon Johnson | 67.06 | 56.10 | 73.33 | 867 | Bears |
| 71 | 35 | Marshon Lattimore | 66.96 | 53.70 | 73.30 | 871 | Saints |
| 72 | 36 | Levi Wallace | 66.53 | 56.00 | 73.35 | 612 | Bills |
| 73 | 37 | Darqueze Dennard | 66.48 | 63.10 | 71.55 | 439 | Falcons |
| 74 | 38 | Harrison Hand | 66.26 | 62.40 | 78.08 | 163 | Vikings |
| 75 | 39 | Patrick Robinson | 65.77 | 56.90 | 74.38 | 248 | Saints |
| 76 | 40 | Emmanuel Moseley | 65.46 | 54.60 | 70.62 | 499 | 49ers |
| 77 | 41 | Jason McCourty | 65.41 | 51.80 | 71.57 | 665 | Patriots |
| 78 | 42 | Darryl Roberts | 65.27 | 55.00 | 72.31 | 469 | Lions |
| 79 | 43 | Tye Smith | 65.10 | 64.10 | 69.20 | 169 | Titans |
| 80 | 44 | Bryce Hall | 65.00 | 62.80 | 70.64 | 547 | Jets |
| 81 | 45 | Chris Harris Jr. | 64.97 | 57.40 | 70.34 | 568 | Chargers |
| 82 | 46 | Isaiah Oliver | 64.96 | 53.30 | 69.60 | 831 | Falcons |
| 83 | 47 | CJ Henderson | 64.87 | 58.30 | 73.41 | 474 | Jaguars |
| 84 | 48 | A.J. Terrell | 64.84 | 57.00 | 67.98 | 908 | Falcons |
| 85 | 49 | Joejuan Williams | 64.61 | 53.70 | 74.89 | 172 | Patriots |
| 86 | 50 | Keion Crossen | 64.58 | 61.50 | 70.90 | 307 | Texans |
| 87 | 51 | Anthony Averett | 64.43 | 60.60 | 70.00 | 355 | Ravens |
| 88 | 52 | Taron Johnson | 64.36 | 55.60 | 68.32 | 825 | Bills |
| 89 | 53 | Patrick Peterson | 64.11 | 53.10 | 69.17 | 1096 | Cardinals |
| 90 | 54 | Sean Murphy-Bunting | 63.60 | 55.20 | 65.41 | 884 | Buccaneers |
| 91 | 55 | Jimmy Moreland | 63.27 | 56.70 | 64.26 | 601 | Commanders |
| 92 | 56 | Nevin Lawson | 63.25 | 54.70 | 67.92 | 737 | Raiders |
| 93 | 57 | Nickell Robey-Coleman | 63.22 | 53.30 | 66.19 | 612 | Eagles |
| 94 | 58 | Josh Jackson | 62.44 | 51.70 | 69.09 | 331 | Packers |
| 95 | 59 | Jamal Perry | 62.29 | 53.60 | 67.95 | 140 | Dolphins |
| 96 | 60 | Tre Herndon | 62.18 | 50.70 | 68.48 | 1017 | Jaguars |
| 97 | 61 | A.J. Bouye | 62.06 | 52.40 | 70.26 | 410 | Broncos |

### Rotation/backup (62 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 98 | 1 | Chidobe Awuzie | 61.95 | 52.00 | 68.78 | 452 | Cowboys |
| 99 | 2 | Rashard Robinson | 61.68 | 55.90 | 72.62 | 187 | Cowboys |
| 100 | 3 | Kevin King | 61.55 | 52.10 | 68.69 | 664 | Packers |
| 101 | 4 | Mike Hughes | 61.55 | 56.70 | 69.58 | 173 | Vikings |
| 102 | 5 | Jamar Taylor | 60.86 | 52.90 | 69.09 | 203 | 49ers |
| 103 | 6 | Rock Ya-Sin | 60.68 | 48.20 | 67.18 | 550 | Colts |
| 104 | 7 | Chris Claybrooks | 60.64 | 57.40 | 62.80 | 375 | Jaguars |
| 105 | 8 | Myles Hartsfield | 60.49 | 59.80 | 64.09 | 140 | Panthers |
| 106 | 9 | Tre Flowers | 60.41 | 51.00 | 65.65 | 578 | Seahawks |
| 107 | 10 | Amani Oruwariye | 60.24 | 51.20 | 65.61 | 1028 | Lions |
| 108 | 11 | Isaac Yiadom | 60.14 | 52.10 | 64.88 | 634 | Giants |
| 109 | 12 | Justin Coleman | 59.71 | 48.40 | 65.68 | 470 | Lions |
| 110 | 13 | Duke Shelley | 59.62 | 58.80 | 66.86 | 209 | Bears |
| 111 | 14 | Kevin Johnson | 59.60 | 52.70 | 65.55 | 575 | Browns |
| 112 | 15 | Jourdan Lewis | 59.59 | 45.80 | 66.28 | 817 | Cowboys |
| 113 | 16 | Anthony Brown | 59.37 | 48.30 | 68.10 | 534 | Cowboys |
| 114 | 17 | Buster Skrine | 59.33 | 46.20 | 66.42 | 557 | Bears |
| 115 | 18 | De'Vante Bausby | 59.19 | 54.00 | 70.06 | 277 | Broncos |
| 116 | 19 | Tevaughn Campbell | 59.17 | 50.30 | 67.16 | 326 | Chargers |
| 117 | 20 | Ka'dar Hollman | 59.16 | 54.90 | 68.90 | 108 | Packers |
| 118 | 21 | Nik Needham | 58.86 | 45.30 | 66.60 | 617 | Dolphins |
| 119 | 22 | Phillip Gaines | 58.66 | 53.50 | 66.78 | 262 | Texans |
| 120 | 23 | Pierre Desir | 58.18 | 42.50 | 69.05 | 519 | Ravens |
| 121 | 24 | Darnay Holmes | 57.93 | 48.80 | 64.02 | 442 | Giants |
| 122 | 25 | Dre Kirkpatrick | 57.84 | 46.10 | 66.28 | 750 | Cardinals |
| 123 | 26 | Quinton Dunbar | 57.69 | 44.20 | 71.17 | 397 | Seahawks |
| 124 | 27 | Daryl Worley | 57.59 | 42.20 | 68.36 | 346 | Raiders |
| 125 | 28 | Antonio Hamilton Sr. | 57.24 | 47.50 | 68.68 | 136 | Chiefs |
| 126 | 29 | Kristian Fulton | 57.18 | 56.50 | 66.89 | 203 | Titans |
| 127 | 30 | Chris Jones | 56.72 | 46.80 | 66.06 | 273 | Vikings |
| 128 | 31 | David Long Jr. | 55.94 | 49.50 | 62.31 | 116 | Rams |
| 129 | 32 | Essang Bassey | 55.90 | 48.40 | 62.98 | 382 | Broncos |
| 130 | 33 | John Reid | 55.64 | 56.80 | 61.57 | 145 | Texans |
| 131 | 34 | Kris Boyd | 55.43 | 50.30 | 63.02 | 343 | Vikings |
| 132 | 35 | Blessuan Austin | 55.29 | 47.40 | 63.15 | 681 | Jets |
| 133 | 36 | Michael Ojemudia | 55.06 | 49.90 | 56.41 | 852 | Broncos |
| 134 | 37 | M.J. Stewart | 54.71 | 44.10 | 64.39 | 229 | Browns |
| 135 | 38 | Tavierre Thomas | 54.44 | 51.10 | 63.18 | 204 | Browns |
| 136 | 39 | Davontae Harris | 54.17 | 46.50 | 63.55 | 117 | Ravens |
| 137 | 40 | Cre'Von LeBlanc | 54.10 | 43.30 | 65.79 | 217 | Eagles |
| 138 | 41 | Jeff Gladney | 53.51 | 48.50 | 52.68 | 958 | Vikings |
| 139 | 42 | Greg Mabin | 53.48 | 41.40 | 68.52 | 248 | Jaguars |
| 140 | 43 | Desmond Trufant | 53.27 | 36.70 | 67.55 | 324 | Lions |
| 141 | 44 | Avonte Maddox | 53.19 | 37.10 | 65.38 | 509 | Eagles |
| 142 | 45 | Vernon Hargreaves III | 53.17 | 37.30 | 63.01 | 980 | Texans |
| 143 | 46 | Isaiah Johnson | 52.98 | 35.70 | 68.28 | 181 | Raiders |
| 144 | 47 | LeShaun Sims | 52.96 | 40.10 | 61.95 | 607 | Bengals |
| 145 | 48 | Kendall Sheffield | 52.66 | 36.90 | 62.39 | 524 | Falcons |
| 146 | 49 | Ryan Lewis | 52.56 | 42.80 | 65.42 | 271 | Giants |
| 147 | 50 | Justin Layne | 50.55 | 47.30 | 57.92 | 120 | Steelers |
| 148 | 51 | Troy Pride Jr. | 49.90 | 41.70 | 53.28 | 529 | Panthers |
| 149 | 52 | Lamar Jackson | 49.77 | 45.80 | 56.58 | 453 | Jets |
| 150 | 53 | D.J. Hayden | 49.18 | 32.60 | 63.36 | 234 | Jaguars |
| 151 | 54 | Keisean Nixon | 47.18 | 40.70 | 56.96 | 155 | Raiders |
| 152 | 55 | Jeff Okudah | 46.99 | 30.90 | 60.85 | 460 | Lions |
| 153 | 56 | Kindle Vildor | 46.65 | 49.80 | 56.35 | 136 | Bears |
| 154 | 57 | Noah Igbinoghene | 45.38 | 38.80 | 51.85 | 286 | Dolphins |
| 155 | 58 | Damon Arnette | 45.29 | 37.50 | 53.62 | 343 | Raiders |
| 156 | 59 | Corey Ballentine | 45.00 | 34.40 | 51.91 | 107 | Jets |
| 157 | 60 | Michael Jacquet | 45.00 | 39.10 | 61.41 | 160 | Eagles |
| 158 | 61 | Chris Jackson | 45.00 | 31.10 | 51.85 | 241 | Titans |
| 159 | 62 | Luq Barcoo | 45.00 | 31.30 | 53.50 | 152 | Jaguars |

## DI — Defensive Interior

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 92.85 | 88.40 | 91.65 | 866 | Rams |
| 2 | 2 | Quinnen Williams | 88.06 | 89.87 | 85.82 | 587 | Jets |
| 3 | 3 | DeForest Buckner | 86.30 | 90.07 | 80.13 | 751 | Colts |
| 4 | 4 | Grady Jarrett | 85.86 | 85.78 | 82.16 | 851 | Falcons |
| 5 | 5 | Leonard Williams | 85.60 | 89.58 | 79.10 | 803 | Giants |
| 6 | 6 | Cameron Heyward | 83.77 | 83.27 | 80.46 | 807 | Steelers |
| 7 | 7 | Kenny Clark | 83.66 | 85.81 | 80.24 | 595 | Packers |
| 8 | 8 | Dexter Lawrence | 83.47 | 87.91 | 76.35 | 655 | Giants |
| 9 | 9 | Fletcher Cox | 83.24 | 84.97 | 78.43 | 747 | Eagles |
| 10 | 10 | Calais Campbell | 82.98 | 76.15 | 85.45 | 410 | Ravens |
| 11 | 11 | Chris Jones | 82.71 | 87.52 | 76.81 | 695 | Chiefs |
| 12 | 12 | Mario Edwards Jr. | 82.51 | 83.93 | 78.64 | 256 | Bears |
| 13 | 13 | Jonathan Allen | 81.44 | 79.74 | 78.73 | 809 | Commanders |
| 14 | 14 | Shelby Harris | 81.42 | 80.64 | 80.37 | 441 | Broncos |
| 15 | 15 | Zach Sieler | 81.36 | 73.72 | 88.02 | 532 | Dolphins |
| 16 | 16 | Dalvin Tomlinson | 81.02 | 82.02 | 76.18 | 658 | Giants |
| 17 | 17 | Stephon Tuitt | 80.98 | 85.25 | 78.04 | 779 | Steelers |
| 18 | 18 | Poona Ford | 80.74 | 78.66 | 79.31 | 670 | Seahawks |
| 19 | 19 | Folorunso Fatukasi | 80.32 | 83.15 | 78.54 | 507 | Jets |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Sebastian Joseph-Day | 79.32 | 73.06 | 79.32 | 412 | Rams |
| 21 | 2 | Sheldon Richardson | 79.27 | 73.10 | 79.21 | 799 | Browns |
| 22 | 3 | Jurrell Casey | 79.04 | 80.81 | 81.30 | 156 | Broncos |
| 23 | 4 | B.J. Hill | 78.95 | 77.97 | 75.44 | 375 | Giants |
| 24 | 5 | Dre'Mont Jones | 78.89 | 78.98 | 77.39 | 560 | Broncos |
| 25 | 6 | Vita Vea | 78.79 | 88.70 | 74.36 | 224 | Buccaneers |
| 26 | 7 | Tim Settle | 77.68 | 71.26 | 78.52 | 348 | Commanders |
| 27 | 8 | Akiem Hicks | 77.59 | 71.69 | 81.32 | 795 | Bears |
| 28 | 9 | Jeffery Simmons | 77.40 | 84.53 | 71.86 | 841 | Titans |
| 29 | 10 | David Onyemata | 77.11 | 75.39 | 74.92 | 599 | Saints |
| 30 | 11 | Daron Payne | 76.62 | 74.36 | 74.28 | 882 | Commanders |
| 31 | 12 | James Smith-Williams | 76.51 | 57.75 | 86.93 | 100 | Commanders |
| 32 | 13 | Zach Kerr | 76.20 | 73.34 | 76.76 | 390 | Panthers |
| 33 | 14 | Bilal Nichols | 76.20 | 67.67 | 79.07 | 618 | Bears |
| 34 | 15 | Christian Wilkins | 75.54 | 73.01 | 74.36 | 637 | Dolphins |
| 35 | 16 | DJ Reader | 75.44 | 79.07 | 74.91 | 259 | Bengals |
| 36 | 17 | Javon Hargrave | 75.37 | 65.22 | 78.49 | 602 | Eagles |
| 37 | 18 | Linval Joseph | 75.18 | 67.84 | 77.05 | 726 | Chargers |
| 38 | 19 | Malcom Brown | 75.15 | 70.24 | 76.02 | 345 | Saints |
| 39 | 20 | Maurice Hurst | 75.12 | 77.46 | 72.63 | 277 | Raiders |
| 40 | 21 | Henry Anderson | 74.99 | 67.82 | 76.54 | 549 | Jets |
| 41 | 22 | Ndamukong Suh | 74.79 | 62.59 | 78.75 | 788 | Buccaneers |
| 42 | 23 | Lawrence Guy Sr. | 74.72 | 64.33 | 78.52 | 503 | Patriots |
| 43 | 24 | Greg Gaines | 74.54 | 67.65 | 77.32 | 201 | Rams |

### Starter (71 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 44 | 1 | Shy Tuttle | 73.42 | 69.15 | 74.05 | 326 | Saints |
| 45 | 2 | Derek Wolfe | 73.07 | 64.99 | 76.58 | 621 | Ravens |
| 46 | 3 | Ed Oliver | 72.93 | 57.59 | 78.99 | 578 | Bills |
| 47 | 4 | Michael Brockers | 72.61 | 64.14 | 74.61 | 625 | Rams |
| 48 | 5 | Tyson Alualu | 72.54 | 71.43 | 69.85 | 448 | Steelers |
| 49 | 6 | Damon Harrison Sr. | 72.49 | 60.20 | 81.51 | 150 | Packers |
| 50 | 7 | Geno Atkins | 72.37 | 55.68 | 83.49 | 119 | Bengals |
| 51 | 8 | Steve McLendon | 72.26 | 57.28 | 78.59 | 443 | Buccaneers |
| 52 | 9 | Derrick Brown | 72.10 | 66.14 | 71.91 | 742 | Panthers |
| 53 | 10 | Johnathan Hankins | 72.01 | 61.46 | 75.08 | 665 | Raiders |
| 54 | 11 | DeShawn Williams | 71.87 | 72.88 | 73.01 | 436 | Broncos |
| 55 | 12 | Brandon Williams | 71.84 | 63.41 | 75.47 | 354 | Ravens |
| 56 | 13 | DaQuan Jones | 71.64 | 68.02 | 69.89 | 706 | Titans |
| 57 | 14 | Kawann Short | 71.31 | 60.67 | 85.81 | 123 | Panthers |
| 58 | 15 | DeMarcus Walker | 71.06 | 57.95 | 81.79 | 384 | Broncos |
| 59 | 16 | Malik Jackson | 70.99 | 57.36 | 81.11 | 537 | Eagles |
| 60 | 17 | Chris Wormley | 70.96 | 75.59 | 65.27 | 148 | Steelers |
| 61 | 18 | Larry Ogunjobi | 70.95 | 54.73 | 78.43 | 642 | Browns |
| 62 | 19 | Taven Bryan | 70.60 | 65.56 | 69.79 | 511 | Jaguars |
| 63 | 20 | Morgan Fox | 70.29 | 54.46 | 76.67 | 403 | Rams |
| 64 | 21 | Raekwon Davis | 70.26 | 71.49 | 65.28 | 538 | Dolphins |
| 65 | 22 | Grover Stewart | 70.23 | 62.19 | 71.63 | 581 | Colts |
| 66 | 23 | Brent Urban | 70.02 | 59.72 | 73.65 | 370 | Bears |
| 67 | 24 | Vincent Taylor | 69.77 | 54.78 | 81.85 | 207 | Browns |
| 68 | 25 | Derrick Nnadi | 69.66 | 65.56 | 68.74 | 460 | Chiefs |
| 69 | 26 | Jarran Reed | 69.52 | 58.42 | 74.64 | 847 | Seahawks |
| 70 | 27 | John Cominsky | 69.19 | 55.07 | 77.57 | 399 | Falcons |
| 71 | 28 | Roy Robertson-Harris | 68.97 | 62.29 | 73.74 | 245 | Bears |
| 72 | 29 | William Gholston | 68.86 | 53.69 | 74.81 | 606 | Buccaneers |
| 73 | 30 | Mike Pennel | 68.66 | 58.90 | 74.55 | 322 | Chiefs |
| 74 | 31 | Tershawn Wharton | 68.59 | 58.10 | 71.41 | 518 | Chiefs |
| 75 | 32 | Kingsley Keke | 68.52 | 63.88 | 69.66 | 414 | Packers |
| 76 | 33 | Dean Lowry | 68.49 | 56.80 | 72.12 | 601 | Packers |
| 77 | 34 | Quinton Jefferson | 68.33 | 59.09 | 70.96 | 534 | Bills |
| 78 | 35 | Hassan Ridgeway | 68.18 | 64.46 | 76.30 | 138 | Eagles |
| 79 | 36 | Danny Shelton | 67.93 | 58.23 | 72.93 | 498 | Lions |
| 80 | 37 | Adam Butler | 67.81 | 52.38 | 74.44 | 481 | Patriots |
| 81 | 38 | John Franklin-Myers | 67.62 | 58.61 | 70.50 | 500 | Jets |
| 82 | 39 | D.J. Jones | 67.57 | 57.14 | 74.20 | 420 | 49ers |
| 83 | 40 | Sheldon Rankins | 67.34 | 59.77 | 72.19 | 415 | Saints |
| 84 | 41 | Mike Daniels | 67.25 | 49.95 | 80.67 | 356 | Bengals |
| 85 | 42 | Armon Watts | 67.00 | 60.93 | 70.40 | 392 | Vikings |
| 86 | 43 | Tyler Lancaster | 66.87 | 58.01 | 70.18 | 352 | Packers |
| 87 | 44 | Vernon Butler | 66.75 | 55.82 | 71.95 | 428 | Bills |
| 88 | 45 | Mike Purcell | 66.72 | 56.23 | 76.11 | 218 | Broncos |
| 89 | 46 | Austin Johnson | 66.71 | 60.51 | 66.68 | 231 | Giants |
| 90 | 47 | Christian Covington | 66.57 | 52.97 | 72.30 | 559 | Bengals |
| 91 | 48 | Abry Jones | 66.47 | 57.40 | 74.29 | 159 | Jaguars |
| 92 | 49 | John Jenkins | 66.38 | 59.63 | 71.19 | 223 | Bears |
| 93 | 50 | Justin Zimmer | 65.65 | 55.24 | 78.01 | 275 | Bills |
| 94 | 51 | Justin Jones | 65.47 | 58.56 | 68.92 | 527 | Chargers |
| 95 | 52 | Javon Kinlaw | 65.20 | 52.25 | 71.75 | 547 | 49ers |
| 96 | 53 | Neville Gallimore | 64.93 | 48.50 | 73.80 | 416 | Cowboys |
| 97 | 54 | Kevin Givens | 64.92 | 53.51 | 76.18 | 387 | 49ers |
| 98 | 55 | Davon Godchaux | 64.79 | 54.79 | 73.02 | 172 | Dolphins |
| 99 | 56 | A'Shawn Robinson | 64.46 | 52.49 | 74.00 | 111 | Rams |
| 100 | 57 | P.J. Hall | 64.37 | 56.64 | 68.91 | 343 | Texans |
| 101 | 58 | Harrison Phillips | 64.16 | 61.92 | 67.63 | 332 | Bills |
| 102 | 59 | L.J. Collier | 63.80 | 52.15 | 67.40 | 559 | Seahawks |
| 103 | 60 | Jonathan Bullard | 63.79 | 54.74 | 73.05 | 117 | Seahawks |
| 104 | 61 | Da'Shawn Hand | 63.54 | 57.52 | 71.21 | 353 | Lions |
| 105 | 62 | Charles Omenihu | 63.50 | 53.80 | 67.23 | 546 | Texans |
| 106 | 63 | Nathan Shepherd | 63.50 | 56.48 | 67.24 | 336 | Jets |
| 107 | 64 | Allen Bailey | 63.26 | 46.50 | 70.58 | 424 | Falcons |
| 108 | 65 | Jordan Phillips | 63.20 | 49.40 | 71.89 | 266 | Cardinals |
| 109 | 66 | Corey Peters | 63.19 | 51.86 | 70.42 | 379 | Cardinals |
| 110 | 67 | Doug Costin | 62.64 | 62.23 | 62.92 | 456 | Jaguars |
| 111 | 68 | Tyeler Davison | 62.33 | 53.43 | 64.52 | 519 | Falcons |
| 112 | 69 | T.Y. McGill | 62.21 | 56.15 | 70.51 | 127 | Eagles |
| 113 | 70 | Isaiah Buggs | 62.13 | 49.67 | 72.90 | 131 | Steelers |
| 114 | 71 | Xavier Williams | 62.11 | 45.98 | 74.21 | 320 | Bengals |

### Rotation/backup (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 115 | 1 | Tyrone Crawford | 61.82 | 46.74 | 71.67 | 445 | Cowboys |
| 116 | 2 | Carlos Watkins | 61.62 | 48.80 | 70.37 | 542 | Texans |
| 117 | 3 | Montravius Adams | 61.42 | 51.45 | 68.69 | 130 | Packers |
| 118 | 4 | Leki Fotu | 60.87 | 45.43 | 72.19 | 284 | Cardinals |
| 119 | 5 | John Penisini | 60.55 | 43.86 | 67.51 | 576 | Lions |
| 120 | 6 | Damion Square | 60.00 | 47.73 | 64.02 | 253 | Chargers |
| 121 | 7 | Maliek Collins | 59.45 | 49.34 | 64.72 | 505 | Raiders |
| 122 | 8 | Brandon Dunn | 59.44 | 46.75 | 65.71 | 451 | Texans |
| 123 | 9 | Ross Blacklock | 59.23 | 45.12 | 65.50 | 254 | Texans |
| 124 | 10 | DaVon Hamilton | 58.95 | 50.20 | 65.81 | 408 | Jaguars |
| 125 | 11 | Angelo Blackson | 58.86 | 44.15 | 64.82 | 550 | Cardinals |
| 126 | 12 | Antwaun Woods | 58.78 | 49.15 | 64.17 | 457 | Cowboys |
| 127 | 13 | Jaleel Johnson | 58.64 | 43.33 | 64.68 | 654 | Vikings |
| 128 | 14 | Kendal Vickers | 58.52 | 44.74 | 64.57 | 315 | Raiders |
| 129 | 15 | Caraun Reid | 58.44 | 50.58 | 69.52 | 144 | Jaguars |
| 130 | 16 | Byron Cowart | 58.30 | 49.50 | 65.99 | 419 | Patriots |
| 131 | 17 | Rakeem Nunez-Roches | 58.21 | 45.79 | 65.02 | 483 | Buccaneers |
| 132 | 18 | Margus Hunt | 57.82 | 39.54 | 67.60 | 387 | Bengals |
| 133 | 19 | Bravvion Roy | 57.79 | 43.85 | 63.95 | 419 | Panthers |
| 134 | 20 | Akeem Spence | 57.79 | 42.98 | 69.01 | 103 | Patriots |
| 135 | 21 | Shamar Stephen | 57.26 | 46.85 | 60.55 | 662 | Vikings |
| 136 | 22 | Justin Ellis | 57.13 | 48.80 | 65.92 | 358 | Ravens |
| 137 | 23 | Taylor Stallworth | 55.49 | 46.75 | 61.31 | 253 | Colts |
| 138 | 24 | Trevon Coley | 55.27 | 45.17 | 65.86 | 192 | Jets |
| 139 | 25 | Matt Dickerson | 54.86 | 46.75 | 65.36 | 197 | Titans |
| 140 | 26 | Trysten Hill | 54.79 | 47.34 | 66.27 | 212 | Cowboys |
| 141 | 27 | Sylvester Williams | 54.54 | 42.68 | 66.29 | 173 | Broncos |
| 142 | 28 | Malcolm Roach | 54.43 | 43.90 | 64.58 | 233 | Saints |
| 143 | 29 | Domata Peko Sr. | 54.32 | 35.97 | 70.41 | 177 | Cardinals |
| 144 | 30 | Kevin Strong | 54.27 | 48.83 | 63.37 | 209 | Lions |
| 145 | 31 | Bryan Mone | 54.09 | 55.19 | 57.79 | 228 | Seahawks |
| 146 | 32 | Rashard Lawrence | 52.96 | 42.97 | 62.76 | 166 | Cardinals |
| 147 | 33 | Rasheem Green | 52.62 | 42.37 | 61.54 | 365 | Seahawks |
| 148 | 34 | Eli Ankou | 52.23 | 43.18 | 63.89 | 186 | Cowboys |
| 149 | 35 | Kentavius Street | 51.38 | 42.67 | 58.75 | 380 | 49ers |
| 150 | 36 | Daniel Ekuale | 51.31 | 47.40 | 57.82 | 290 | Jaguars |
| 151 | 37 | Teair Tart | 50.87 | 47.23 | 59.99 | 155 | Titans |
| 152 | 38 | McTelvin Agim | 50.77 | 44.98 | 56.72 | 141 | Broncos |
| 153 | 39 | Jordan Elliott | 50.64 | 46.35 | 49.34 | 307 | Browns |
| 154 | 40 | Justin Hamilton | 49.46 | 49.00 | 54.77 | 236 | Cowboys |
| 155 | 41 | Marlon Davidson | 49.24 | 52.68 | 51.11 | 132 | Falcons |
| 156 | 42 | Larrell Murchison | 47.45 | 41.48 | 53.52 | 136 | Titans |
| 157 | 43 | Broderick Washington | 45.00 | 39.78 | 48.29 | 161 | Ravens |

## ED — Edge

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | T.J. Watt | 93.03 | 96.33 | 87.18 | 856 | Steelers |
| 2 | 2 | Joey Bosa | 91.47 | 97.12 | 87.51 | 549 | Chargers |
| 3 | 3 | Khalil Mack | 89.24 | 93.04 | 82.96 | 894 | Bears |
| 4 | 4 | Myles Garrett | 87.80 | 95.20 | 81.61 | 758 | Browns |
| 5 | 5 | DeMarcus Lawrence | 85.32 | 89.47 | 78.38 | 668 | Cowboys |
| 6 | 6 | Shaquil Barrett | 84.33 | 82.19 | 82.74 | 824 | Buccaneers |
| 7 | 7 | Montez Sweat | 82.52 | 83.97 | 77.39 | 693 | Commanders |
| 8 | 8 | Brandon Graham | 82.44 | 80.64 | 79.47 | 759 | Eagles |
| 9 | 9 | Justin Houston | 81.83 | 70.77 | 85.87 | 608 | Colts |
| 10 | 10 | Cameron Jordan | 81.80 | 86.05 | 74.80 | 816 | Saints |
| 11 | 11 | Za'Darius Smith | 80.37 | 79.44 | 76.82 | 858 | Packers |
| 12 | 12 | Brian Burns | 80.33 | 75.65 | 79.93 | 750 | Panthers |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 13 | 1 | Marcus Davenport | 79.25 | 87.29 | 73.89 | 374 | Saints |
| 14 | 2 | Rashan Gary | 78.45 | 70.88 | 79.98 | 456 | Packers |
| 15 | 3 | Uchenna Nwosu | 77.96 | 67.66 | 83.79 | 356 | Chargers |
| 16 | 4 | Bradley Chubb | 77.55 | 71.73 | 82.05 | 741 | Broncos |
| 17 | 5 | Chase Winovich | 76.29 | 66.30 | 78.78 | 593 | Patriots |
| 18 | 6 | Trey Flowers | 76.22 | 77.94 | 76.10 | 309 | Lions |
| 19 | 7 | Chase Young | 75.54 | 88.32 | 63.89 | 770 | Commanders |
| 20 | 8 | Olivier Vernon | 74.45 | 74.47 | 74.24 | 805 | Browns |
| 21 | 9 | Genard Avery | 74.37 | 65.00 | 84.78 | 126 | Eagles |
| 22 | 10 | Andrew Van Ginkel | 74.29 | 69.41 | 73.37 | 479 | Dolphins |
| 23 | 11 | Carlos Dunlap | 74.29 | 66.71 | 76.32 | 593 | Seahawks |

### Starter (66 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 24 | 1 | Matthew Judon | 73.97 | 59.36 | 80.58 | 563 | Ravens |
| 25 | 2 | Trey Hendrickson | 73.96 | 68.52 | 77.17 | 558 | Saints |
| 26 | 3 | Jadeveon Clowney | 73.96 | 86.36 | 66.85 | 425 | Titans |
| 27 | 4 | Carl Lawson | 73.61 | 68.27 | 76.13 | 723 | Bengals |
| 28 | 5 | Terrell Lewis | 73.19 | 72.20 | 78.02 | 124 | Rams |
| 29 | 6 | Randy Gregory | 73.05 | 74.48 | 71.48 | 270 | Cowboys |
| 30 | 7 | Ryan Kerrigan | 72.94 | 55.13 | 81.90 | 398 | Commanders |
| 31 | 8 | Markus Golden | 72.93 | 57.95 | 79.79 | 591 | Cardinals |
| 32 | 9 | Pernell McPhee | 72.87 | 63.63 | 78.83 | 458 | Ravens |
| 33 | 10 | Frank Clark | 72.52 | 62.34 | 76.29 | 759 | Chiefs |
| 34 | 11 | Chandler Jones | 72.24 | 66.89 | 77.38 | 286 | Cardinals |
| 35 | 12 | Melvin Ingram III | 71.93 | 66.64 | 76.92 | 361 | Chargers |
| 36 | 13 | Shaq Lawson | 71.01 | 67.19 | 71.16 | 571 | Dolphins |
| 37 | 14 | Jerry Hughes | 70.96 | 61.21 | 73.81 | 629 | Bills |
| 38 | 15 | Tyus Bowser | 70.49 | 62.63 | 71.77 | 540 | Ravens |
| 39 | 16 | Deatrich Wise Jr. | 70.49 | 68.81 | 68.08 | 565 | Patriots |
| 40 | 17 | Derek Barnett | 70.49 | 69.65 | 71.15 | 535 | Eagles |
| 41 | 18 | Yannick Ngakoue | 70.44 | 59.90 | 74.14 | 657 | Ravens |
| 42 | 19 | J.J. Watt | 70.10 | 61.05 | 74.46 | 1013 | Texans |
| 43 | 20 | Robert Quinn | 70.09 | 55.92 | 76.52 | 548 | Bears |
| 44 | 21 | Arik Armstead | 70.05 | 65.53 | 68.89 | 750 | 49ers |
| 45 | 22 | Clelin Ferrell | 69.86 | 81.55 | 61.55 | 461 | Raiders |
| 46 | 23 | Mario Addison | 69.71 | 52.40 | 77.91 | 606 | Bills |
| 47 | 24 | Jacob Martin | 69.70 | 59.59 | 73.94 | 375 | Texans |
| 48 | 25 | Dante Fowler Jr. | 68.89 | 62.14 | 70.48 | 601 | Falcons |
| 49 | 26 | Everson Griffen | 68.78 | 56.95 | 74.90 | 528 | Lions |
| 50 | 27 | Carl Granderson | 68.46 | 63.42 | 72.21 | 291 | Saints |
| 51 | 28 | Samson Ebukam | 68.38 | 60.66 | 69.36 | 364 | Rams |
| 52 | 29 | Sam Hubbard | 68.21 | 62.94 | 69.44 | 665 | Bengals |
| 53 | 30 | Bud Dupree | 68.18 | 63.34 | 69.84 | 609 | Steelers |
| 54 | 31 | Preston Smith | 68.06 | 58.01 | 70.59 | 814 | Packers |
| 55 | 32 | Josh Sweat | 67.94 | 64.47 | 68.79 | 422 | Eagles |
| 56 | 33 | Jabaal Sheard | 67.91 | 64.11 | 70.34 | 275 | Giants |
| 57 | 34 | Ifeadi Odenigbo | 67.79 | 63.18 | 67.21 | 696 | Vikings |
| 58 | 35 | Josh Uche | 67.43 | 67.97 | 70.21 | 179 | Patriots |
| 59 | 36 | Vinny Curry | 67.40 | 56.95 | 73.63 | 310 | Eagles |
| 60 | 37 | Emmanuel Ogbah | 67.39 | 66.84 | 65.88 | 792 | Dolphins |
| 61 | 38 | Frankie Luvu | 67.34 | 62.50 | 69.54 | 257 | Jets |
| 62 | 39 | Maxx Crosby | 67.16 | 60.29 | 67.57 | 906 | Raiders |
| 63 | 40 | Alton Robinson | 67.05 | 63.32 | 67.46 | 336 | Seahawks |
| 64 | 41 | Jason Pierre-Paul | 67.01 | 57.86 | 70.82 | 943 | Buccaneers |
| 65 | 42 | Whitney Mercilus | 66.87 | 54.02 | 72.84 | 614 | Texans |
| 66 | 43 | Jordan Jenkins | 66.61 | 59.47 | 69.91 | 528 | Jets |
| 67 | 44 | Romeo Okwara | 66.54 | 64.91 | 64.29 | 748 | Lions |
| 68 | 45 | Kerry Hyder Jr. | 66.01 | 54.89 | 71.14 | 722 | 49ers |
| 69 | 46 | Aaron Lynch | 65.99 | 54.49 | 74.27 | 152 | Jaguars |
| 70 | 47 | Ogbo Okoronkwo | 65.90 | 62.50 | 70.64 | 158 | Rams |
| 71 | 48 | Jeremiah Attaochu | 65.83 | 56.40 | 72.31 | 414 | Broncos |
| 72 | 49 | Alex Highsmith | 65.07 | 60.67 | 63.83 | 440 | Steelers |
| 73 | 50 | Malik Reed | 64.74 | 60.71 | 63.65 | 785 | Broncos |
| 74 | 51 | Harold Landry III | 64.15 | 59.67 | 62.97 | 1050 | Titans |
| 75 | 52 | Aldon Smith | 63.99 | 54.86 | 67.38 | 809 | Cowboys |
| 76 | 53 | Jihad Ward | 63.52 | 54.76 | 68.32 | 271 | Ravens |
| 77 | 54 | Justin Hollins | 63.41 | 58.66 | 63.20 | 349 | Rams |
| 78 | 55 | Devon Kennard | 63.31 | 54.59 | 66.72 | 362 | Cardinals |
| 79 | 56 | Kyler Fackrell | 63.19 | 51.07 | 69.18 | 608 | Giants |
| 80 | 57 | Vic Beasley Jr. | 62.93 | 55.24 | 67.02 | 199 | Raiders |
| 81 | 58 | Jaylon Ferguson | 62.91 | 59.64 | 63.01 | 303 | Ravens |
| 82 | 59 | Alex Okafor | 62.72 | 53.26 | 69.35 | 283 | Chiefs |
| 83 | 60 | Barkevious Mingo | 62.66 | 49.88 | 69.52 | 391 | Bears |
| 84 | 61 | Efe Obada | 62.41 | 55.66 | 63.99 | 415 | Panthers |
| 85 | 62 | Adrian Clayborn | 62.20 | 52.00 | 66.08 | 404 | Browns |
| 86 | 63 | Bruce Irvin | 62.14 | 48.74 | 75.14 | 121 | Seahawks |
| 87 | 64 | Trent Murphy | 62.12 | 55.23 | 66.29 | 343 | Bills |
| 88 | 65 | Mike Danna | 62.07 | 58.89 | 63.15 | 334 | Chiefs |
| 89 | 66 | Carter Coughlin | 62.02 | 57.26 | 67.27 | 193 | Giants |

### Rotation/backup (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 90 | 1 | A.J. Epenesa | 61.97 | 60.41 | 60.93 | 291 | Bills |
| 91 | 2 | Lorenzo Carter | 61.93 | 60.83 | 64.75 | 234 | Giants |
| 92 | 3 | Derek Rivers | 61.93 | 52.78 | 68.03 | 115 | Rams |
| 93 | 4 | John Simon | 61.60 | 49.86 | 66.29 | 702 | Patriots |
| 94 | 5 | Tarell Basham | 61.44 | 59.45 | 60.26 | 734 | Jets |
| 95 | 6 | Dion Jordan | 61.09 | 53.63 | 67.00 | 375 | 49ers |
| 96 | 7 | Yetur Gross-Matos | 60.99 | 59.02 | 62.30 | 377 | Panthers |
| 97 | 8 | Benson Mayowa | 60.93 | 54.18 | 63.34 | 571 | Seahawks |
| 98 | 9 | Oshane Ximines | 60.63 | 59.50 | 65.04 | 110 | Giants |
| 99 | 10 | James Vaughters | 60.47 | 54.95 | 62.06 | 243 | Bears |
| 100 | 11 | Anthony Nelson | 60.41 | 68.81 | 53.38 | 324 | Buccaneers |
| 101 | 12 | Carl Nassib | 60.23 | 57.67 | 59.66 | 463 | Raiders |
| 102 | 13 | Chris Smith | 59.92 | 54.48 | 63.54 | 115 | Raiders |
| 103 | 14 | Damontre Moore | 59.69 | 59.56 | 66.12 | 184 | Seahawks |
| 104 | 15 | Dawuane Smoot | 59.59 | 56.67 | 59.04 | 665 | Jaguars |
| 105 | 16 | Charles Harris | 58.49 | 58.25 | 57.71 | 289 | Falcons |
| 106 | 17 | D.J. Wonnum | 58.19 | 56.20 | 57.44 | 471 | Vikings |
| 107 | 18 | Khalid Kareem | 58.19 | 56.51 | 55.15 | 259 | Bengals |
| 108 | 19 | Cassius Marsh | 58.15 | 50.22 | 63.43 | 151 | Steelers |
| 109 | 20 | Bryce Huff | 58.09 | 56.28 | 57.22 | 296 | Jets |
| 110 | 21 | Al-Quadin Muhammad | 57.90 | 57.48 | 54.21 | 579 | Colts |
| 111 | 22 | Ryan Anderson | 57.82 | 54.83 | 59.92 | 146 | Commanders |
| 112 | 23 | K'Lavon Chaisson | 57.74 | 55.71 | 54.92 | 569 | Jaguars |
| 113 | 24 | Kamalei Correa | 57.71 | 51.74 | 62.08 | 197 | Jaguars |
| 114 | 25 | Alex Barrett | 57.38 | 58.34 | 63.44 | 120 | 49ers |
| 115 | 26 | Arden Key | 57.00 | 59.33 | 55.13 | 435 | Raiders |
| 116 | 27 | Ben Banogu | 56.92 | 54.12 | 59.57 | 100 | Colts |
| 117 | 28 | Anthony Chickillo | 56.87 | 51.31 | 60.58 | 164 | Broncos |
| 118 | 29 | Kyle Phillips | 56.79 | 59.31 | 57.20 | 171 | Jets |
| 119 | 30 | Dorance Armstrong | 56.70 | 54.23 | 54.69 | 368 | Cowboys |
| 120 | 31 | Derick Roberson | 56.58 | 55.29 | 63.95 | 248 | Titans |
| 121 | 32 | Isaac Rochell | 56.49 | 54.15 | 53.89 | 438 | Chargers |
| 122 | 33 | Jordan Willis | 56.25 | 58.57 | 56.37 | 229 | 49ers |
| 123 | 34 | Stephen Weatherly | 56.13 | 52.96 | 57.72 | 358 | Panthers |
| 124 | 35 | Porter Gustin | 56.04 | 59.26 | 54.93 | 326 | Browns |
| 125 | 36 | Tanoh Kpassagnon | 55.84 | 54.02 | 54.36 | 720 | Chiefs |
| 126 | 37 | Darryl Johnson | 55.71 | 54.14 | 53.63 | 225 | Bills |
| 127 | 38 | Marquis Haynes Sr. | 55.67 | 53.08 | 57.82 | 390 | Panthers |
| 128 | 39 | Zach Allen | 55.60 | 52.75 | 55.28 | 505 | Cardinals |
| 129 | 40 | Tashawn Bower | 54.13 | 53.98 | 61.32 | 137 | Patriots |
| 130 | 41 | Jonathan Greenard | 53.75 | 55.09 | 52.86 | 265 | Texans |
| 131 | 42 | Steven Means | 53.61 | 47.44 | 55.22 | 646 | Falcons |
| 132 | 43 | Christian Jones | 52.37 | 45.55 | 52.75 | 510 | Lions |
| 133 | 44 | Shilique Calhoun | 51.78 | 49.08 | 54.73 | 256 | Patriots |
| 134 | 45 | Adam Gotsis | 51.42 | 44.49 | 51.88 | 579 | Jaguars |
| 135 | 46 | Amani Bledsoe | 51.29 | 52.54 | 48.38 | 312 | Bengals |
| 136 | 47 | Austin Larkin | 50.62 | 52.98 | 55.29 | 127 | Panthers |
| 137 | 48 | Austin Bryant | 50.37 | 58.24 | 52.16 | 212 | Lions |
| 138 | 49 | Olasunkanmi Adeniyi | 50.26 | 52.29 | 47.88 | 146 | Steelers |
| 139 | 50 | Kylie Fitts | 47.33 | 52.14 | 48.60 | 140 | Cardinals |
| 140 | 51 | Jabari Zuniga | 46.09 | 51.32 | 46.77 | 103 | Jets |

## G — Guard

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Zack Martin | 96.00 | 91.30 | 94.97 | 618 | Cowboys |
| 2 | 2 | Wyatt Teller | 95.25 | 92.70 | 92.78 | 694 | Browns |
| 3 | 3 | Ali Marpet | 93.44 | 86.70 | 93.77 | 849 | Buccaneers |
| 4 | 4 | Quenton Nelson | 92.29 | 86.20 | 92.18 | 1082 | Colts |
| 5 | 5 | Joel Bitonio | 90.52 | 84.60 | 90.30 | 1061 | Browns |
| 6 | 6 | Shaq Mason | 90.48 | 85.40 | 89.70 | 782 | Patriots |
| 7 | 7 | Brandon Scherff | 89.22 | 84.10 | 88.47 | 857 | Commanders |
| 8 | 8 | Laken Tomlinson | 85.65 | 78.80 | 86.05 | 1094 | 49ers |
| 9 | 9 | Chris Lindstrom | 84.49 | 77.10 | 85.25 | 1122 | Falcons |
| 10 | 10 | Joe Thuney | 81.82 | 74.20 | 82.73 | 980 | Patriots |
| 11 | 11 | Damien Lewis | 81.63 | 70.20 | 85.09 | 967 | Seahawks |
| 12 | 12 | Austin Corbett | 80.48 | 70.90 | 82.70 | 1120 | Rams |
| 13 | 13 | Connor Williams | 80.11 | 71.20 | 81.89 | 1146 | Cowboys |
| 14 | 14 | Rodger Saffold | 80.02 | 70.20 | 82.40 | 872 | Titans |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | David Edwards | 79.11 | 70.30 | 80.81 | 1006 | Rams |
| 16 | 2 | Mike Iupati | 79.04 | 67.80 | 82.36 | 498 | Seahawks |
| 17 | 3 | Nate Davis | 78.75 | 69.70 | 80.62 | 1074 | Titans |
| 18 | 4 | Graham Glasgow | 78.64 | 68.50 | 81.24 | 764 | Broncos |
| 19 | 5 | Tom Compton | 78.45 | 68.10 | 81.19 | 148 | 49ers |
| 20 | 6 | Wes Schweitzer | 77.90 | 69.00 | 79.66 | 990 | Commanders |
| 21 | 7 | Oday Aboushi | 77.89 | 66.60 | 81.25 | 622 | Lions |
| 22 | 8 | Elgton Jenkins | 77.60 | 67.70 | 80.04 | 1037 | Packers |
| 23 | 9 | Alex Cappa | 77.48 | 69.00 | 78.97 | 1070 | Buccaneers |
| 24 | 10 | A.J. Cann | 77.40 | 69.00 | 78.84 | 919 | Jaguars |
| 25 | 11 | Nick Allegretti | 77.17 | 66.20 | 80.31 | 694 | Chiefs |
| 26 | 12 | Alex Lewis | 77.05 | 66.60 | 79.85 | 544 | Jets |
| 27 | 13 | Mark Glowinski | 76.40 | 67.30 | 78.30 | 1090 | Colts |
| 28 | 14 | Justin Pugh | 76.33 | 64.80 | 79.85 | 958 | Cardinals |
| 29 | 15 | Jon Feliciano | 76.20 | 64.60 | 79.77 | 571 | Bills |
| 30 | 16 | Andrew Norwell | 76.10 | 67.90 | 77.40 | 801 | Jaguars |
| 31 | 17 | Kevin Zeitler | 75.71 | 65.90 | 78.09 | 1003 | Giants |
| 32 | 18 | Lucas Patrick | 75.53 | 64.80 | 78.52 | 939 | Packers |
| 33 | 19 | Germain Ifedi | 75.22 | 65.00 | 77.86 | 1066 | Bears |
| 34 | 20 | Bradley Bozeman | 74.90 | 64.30 | 77.80 | 1017 | Ravens |
| 35 | 21 | Kelechi Osemele | 74.74 | 59.80 | 80.53 | 282 | Chiefs |
| 36 | 22 | Ereck Flowers | 74.68 | 65.90 | 76.36 | 857 | Dolphins |
| 37 | 23 | Alex Redmond | 74.56 | 61.50 | 79.10 | 448 | Bengals |
| 38 | 24 | James Daniels | 74.01 | 65.80 | 75.31 | 305 | Bears |

### Starter (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 39 | 1 | Gabe Jackson | 73.77 | 63.70 | 76.32 | 1062 | Raiders |
| 40 | 2 | Ike Boettger | 73.39 | 65.30 | 74.62 | 623 | Bills |
| 41 | 3 | Connor McGovern | 72.72 | 61.70 | 75.90 | 606 | Cowboys |
| 42 | 4 | Zach Fulton | 72.59 | 63.00 | 74.81 | 953 | Texans |
| 43 | 5 | David DeCastro | 72.55 | 64.10 | 74.02 | 845 | Steelers |
| 44 | 6 | John Miller | 72.38 | 61.20 | 75.66 | 910 | Panthers |
| 45 | 7 | Andrus Peat | 72.22 | 61.20 | 75.40 | 766 | Saints |
| 46 | 8 | Stefen Wisniewski | 71.81 | 61.20 | 74.72 | 209 | Chiefs |
| 47 | 9 | Isaac Seumalo | 71.58 | 62.40 | 73.53 | 588 | Eagles |
| 48 | 10 | Dalton Risner | 71.25 | 61.30 | 73.72 | 999 | Broncos |
| 49 | 11 | Joe Dahl | 70.78 | 57.40 | 75.53 | 264 | Lions |
| 50 | 12 | Kevin Dotson | 70.62 | 66.20 | 69.40 | 358 | Steelers |
| 51 | 13 | Xavier Su'a-Filo | 70.48 | 59.10 | 73.90 | 293 | Bengals |
| 52 | 14 | Quinton Spain | 70.47 | 56.70 | 75.48 | 720 | Bengals |
| 53 | 15 | Jonah Jackson | 70.04 | 57.00 | 74.56 | 1006 | Lions |
| 54 | 16 | Ryan Groy | 69.96 | 57.70 | 73.96 | 271 | Chargers |
| 55 | 17 | Michael Jordan | 69.61 | 55.70 | 74.72 | 731 | Bengals |
| 56 | 18 | Will Hernandez | 69.43 | 58.10 | 72.81 | 525 | Giants |
| 57 | 19 | Sua Opeta | 69.15 | 57.60 | 72.69 | 170 | Eagles |
| 58 | 20 | Matt Pryor | 67.99 | 55.40 | 72.21 | 776 | Eagles |
| 59 | 21 | Brian Winters | 67.77 | 54.60 | 72.39 | 618 | Bills |
| 60 | 22 | James Carpenter | 67.74 | 56.10 | 71.34 | 826 | Falcons |
| 61 | 23 | Senio Kelemete | 67.57 | 54.50 | 72.11 | 367 | Texans |
| 62 | 24 | Ben Bartch | 67.45 | 58.50 | 69.25 | 219 | Jaguars |
| 63 | 25 | Justin McCray | 67.45 | 54.80 | 71.71 | 156 | Falcons |
| 64 | 26 | Andrew Wylie | 67.32 | 54.90 | 71.44 | 972 | Chiefs |
| 65 | 27 | Cesar Ruiz | 67.28 | 58.60 | 68.90 | 744 | Saints |
| 66 | 28 | Ben Powers | 66.09 | 59.40 | 66.38 | 513 | Ravens |
| 67 | 29 | J.R. Sweezy | 66.00 | 52.50 | 70.83 | 643 | Cardinals |
| 68 | 30 | Max Scharping | 65.86 | 52.10 | 70.87 | 454 | Texans |
| 69 | 31 | Jordan Simmons | 65.14 | 51.60 | 70.00 | 593 | Seahawks |
| 70 | 32 | Solomon Kindley | 65.05 | 51.30 | 70.05 | 748 | Dolphins |
| 71 | 33 | Michael Schofield III | 64.94 | 50.30 | 70.54 | 270 | Panthers |
| 72 | 34 | Forrest Lamp | 64.03 | 49.40 | 69.62 | 1174 | Chargers |
| 73 | 35 | Pat Elflein | 63.71 | 48.00 | 70.01 | 419 | Jets |
| 74 | 36 | John Simpson | 62.55 | 45.80 | 69.55 | 252 | Raiders |

### Rotation/backup (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 75 | 1 | Tyre Phillips | 61.84 | 47.10 | 67.50 | 418 | Ravens |
| 76 | 2 | Dakota Dozier | 61.64 | 44.70 | 68.77 | 1083 | Vikings |
| 77 | 3 | Jon Runyan | 61.56 | 53.80 | 62.56 | 160 | Packers |
| 78 | 4 | Rashaad Coward | 61.53 | 46.20 | 67.59 | 333 | Bears |
| 79 | 5 | Wes Martin | 60.88 | 44.80 | 67.44 | 339 | Commanders |
| 80 | 6 | Austin Schlottmann | 58.72 | 39.80 | 67.17 | 269 | Broncos |
| 81 | 7 | Netane Muti | 57.19 | 37.10 | 66.41 | 122 | Broncos |
| 82 | 8 | Trai Turner | 56.60 | 34.80 | 66.97 | 536 | Chargers |
| 83 | 9 | Shane Lemieux | 55.17 | 32.20 | 66.31 | 504 | Giants |
| 84 | 10 | Dru Samia | 54.89 | 33.10 | 65.25 | 272 | Vikings |

## HB — Running Back

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Chubb | 87.66 | 86.70 | 84.13 | 158 | Browns |
| 2 | 2 | Derrick Henry | 86.81 | 92.40 | 78.92 | 223 | Titans |
| 3 | 3 | Dalvin Cook | 83.66 | 89.00 | 75.93 | 263 | Vikings |
| 4 | 4 | Alvin Kamara | 83.31 | 82.20 | 79.88 | 345 | Saints |
| 5 | 5 | Gus Edwards | 81.18 | 85.80 | 73.93 | 104 | Ravens |
| 6 | 6 | Tony Pollard | 80.40 | 71.70 | 82.04 | 199 | Cowboys |
| 7 | 7 | Austin Ekeler | 80.37 | 77.00 | 78.45 | 235 | Chargers |
| 8 | 8 | Jonathan Taylor | 80.35 | 83.90 | 73.81 | 203 | Colts |
| 9 | 9 | Aaron Jones | 80.21 | 79.40 | 76.58 | 253 | Packers |

### Good (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | J.K. Dobbins | 79.29 | 71.50 | 80.31 | 217 | Ravens |
| 11 | 2 | Chris Carson | 78.93 | 78.10 | 75.32 | 207 | Seahawks |
| 12 | 3 | Kareem Hunt | 78.43 | 74.00 | 77.21 | 252 | Browns |
| 13 | 4 | Josh Jacobs | 77.61 | 76.30 | 74.32 | 231 | Raiders |
| 14 | 5 | Antonio Gibson | 77.45 | 80.90 | 70.99 | 182 | Commanders |
| 15 | 6 | Darrell Henderson | 77.04 | 80.40 | 70.64 | 147 | Rams |
| 16 | 7 | David Montgomery | 76.73 | 81.80 | 69.19 | 391 | Bears |
| 17 | 8 | Clyde Edwards-Helaire | 76.50 | 75.30 | 73.14 | 286 | Chiefs |
| 18 | 9 | James Robinson | 76.10 | 72.90 | 74.07 | 286 | Jaguars |
| 19 | 10 | Ronald Jones | 74.74 | 73.60 | 71.34 | 190 | Buccaneers |
| 20 | 11 | Devin Singletary | 74.54 | 66.70 | 75.60 | 318 | Bills |
| 21 | 12 | Latavius Murray | 74.14 | 82.40 | 64.47 | 133 | Saints |

### Starter (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 22 | 1 | Le'Veon Bell | 73.71 | 73.90 | 69.42 | 153 | Chiefs |
| 23 | 2 | Nyheim Hines | 73.50 | 81.60 | 63.94 | 264 | Colts |
| 24 | 3 | Miles Sanders | 73.29 | 63.00 | 75.98 | 295 | Eagles |
| 25 | 4 | Myles Gaskin | 72.76 | 74.50 | 67.43 | 208 | Dolphins |
| 26 | 5 | Zack Moss | 72.56 | 72.30 | 68.57 | 173 | Bills |
| 27 | 6 | Rex Burkhead | 72.38 | 78.20 | 64.33 | 123 | Patriots |
| 28 | 7 | Melvin Gordon III | 72.14 | 72.50 | 67.73 | 290 | Broncos |
| 29 | 8 | Wayne Gallman | 71.87 | 71.10 | 68.22 | 160 | Giants |
| 30 | 9 | Ezekiel Elliott | 71.47 | 65.30 | 71.42 | 387 | Cowboys |
| 31 | 10 | Duke Johnson Jr. | 71.25 | 62.60 | 72.85 | 204 | Texans |
| 32 | 11 | James Conner | 71.01 | 68.00 | 68.85 | 269 | Steelers |
| 33 | 12 | D'Andre Swift | 70.94 | 69.40 | 67.80 | 226 | Lions |
| 34 | 13 | Jerick McKinnon | 70.81 | 73.70 | 64.72 | 213 | 49ers |
| 35 | 14 | Boston Scott | 70.37 | 68.20 | 67.65 | 211 | Eagles |
| 36 | 15 | Devontae Booker | 70.32 | 68.70 | 67.23 | 105 | Raiders |
| 37 | 16 | Mike Davis | 70.07 | 75.10 | 62.55 | 303 | Panthers |
| 38 | 17 | Chase Edmonds | 69.99 | 69.20 | 66.35 | 305 | Cardinals |
| 39 | 18 | Carlos Hyde | 69.91 | 68.10 | 66.95 | 125 | Seahawks |
| 40 | 19 | Cam Akers | 69.89 | 68.40 | 66.71 | 107 | Rams |
| 41 | 20 | Jamaal Williams | 69.89 | 73.20 | 63.51 | 201 | Packers |
| 42 | 21 | Kenyan Drake | 69.71 | 60.90 | 71.41 | 251 | Cardinals |
| 43 | 22 | Jalen Richard | 69.62 | 57.80 | 73.33 | 123 | Raiders |
| 44 | 23 | Jeremy McNichols | 69.45 | 58.20 | 72.78 | 152 | Titans |
| 45 | 24 | Jeff Wilson Jr. | 69.28 | 69.10 | 65.23 | 133 | 49ers |
| 46 | 25 | Ty Johnson | 68.97 | 69.10 | 64.71 | 100 | Jets |
| 47 | 26 | Joe Mixon | 68.78 | 65.30 | 66.93 | 128 | Bengals |
| 48 | 27 | Kerryon Johnson | 68.13 | 61.60 | 68.31 | 159 | Lions |
| 49 | 28 | J.D. McKissic | 68.03 | 62.70 | 67.41 | 399 | Commanders |
| 50 | 29 | Adrian Peterson | 67.88 | 59.60 | 69.24 | 110 | Lions |
| 51 | 30 | Brian Hill | 67.88 | 57.90 | 70.36 | 165 | Falcons |
| 52 | 31 | Chris Thompson | 66.66 | 54.90 | 70.34 | 127 | Jaguars |
| 53 | 32 | David Johnson | 66.62 | 62.90 | 64.93 | 333 | Texans |
| 54 | 33 | Dion Lewis | 66.56 | 54.70 | 70.30 | 201 | Giants |
| 55 | 34 | Giovani Bernard | 66.36 | 63.40 | 64.17 | 276 | Bengals |
| 56 | 35 | Leonard Fournette | 65.85 | 59.80 | 65.71 | 216 | Buccaneers |
| 57 | 36 | James White | 65.56 | 65.50 | 61.43 | 184 | Patriots |
| 58 | 37 | Todd Gurley II | 65.17 | 54.80 | 67.91 | 229 | Falcons |
| 59 | 38 | Ito Smith | 64.76 | 59.00 | 64.43 | 139 | Falcons |
| 60 | 39 | Dare Ogunbowale | 64.54 | 58.80 | 64.20 | 121 | Jaguars |
| 61 | 40 | Frank Gore | 64.27 | 60.80 | 62.41 | 136 | Jets |
| 62 | 41 | Malcolm Brown | 62.58 | 57.10 | 62.07 | 276 | Rams |
| 63 | 42 | Joshua Kelley | 62.58 | 63.10 | 58.07 | 130 | Chargers |
| 64 | 43 | Kalen Ballage | 62.36 | 62.70 | 57.96 | 143 | Chargers |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 65 | 1 | Darrel Williams | 61.05 | 61.00 | 56.92 | 177 | Chiefs |

## LB — Linebacker

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Fred Warner | 83.89 | 88.60 | 76.59 | 973 | 49ers |
| 2 | 2 | Bobby Wagner | 81.84 | 83.20 | 76.97 | 1141 | Seahawks |
| 3 | 3 | Lavonte David | 80.25 | 81.50 | 75.67 | 1058 | Buccaneers |

### Good (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 4 | 1 | Denzel Perryman | 79.48 | 83.30 | 77.04 | 317 | Chargers |
| 5 | 2 | Eric Kendricks | 78.51 | 82.60 | 74.95 | 754 | Vikings |
| 6 | 3 | Demario Davis | 77.43 | 78.10 | 72.82 | 1032 | Saints |
| 7 | 4 | Blake Martinez | 77.16 | 75.90 | 73.84 | 1063 | Giants |
| 8 | 5 | Mykal Walker | 74.75 | 74.00 | 71.09 | 387 | Falcons |

### Starter (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | K.J. Wright | 73.41 | 75.30 | 70.26 | 990 | Seahawks |
| 10 | 2 | Myles Jack | 70.85 | 69.10 | 70.45 | 931 | Jaguars |
| 11 | 3 | Cole Holcomb | 70.82 | 72.00 | 69.77 | 555 | Commanders |
| 12 | 4 | Roquan Smith | 70.62 | 67.20 | 69.99 | 1016 | Bears |
| 13 | 5 | Sione Takitaki | 70.10 | 71.20 | 68.99 | 435 | Browns |
| 14 | 6 | Willie Gay | 70.03 | 68.10 | 68.19 | 269 | Chiefs |
| 15 | 7 | Josey Jewell | 69.75 | 68.10 | 68.25 | 1011 | Broncos |
| 16 | 8 | L.J. Fort | 69.41 | 69.40 | 68.69 | 381 | Ravens |
| 17 | 9 | Kamal Martin | 69.02 | 73.30 | 68.25 | 190 | Packers |
| 18 | 10 | Deion Jones | 68.84 | 68.70 | 66.85 | 1040 | Falcons |
| 19 | 11 | Malcolm Smith | 68.31 | 70.80 | 68.22 | 559 | Browns |
| 20 | 12 | B.J. Goodson | 68.28 | 65.40 | 67.91 | 848 | Browns |
| 21 | 13 | Isaiah Simmons | 66.98 | 59.90 | 67.53 | 376 | Cardinals |
| 22 | 14 | Nick Kwiatkoski | 66.78 | 67.10 | 67.40 | 651 | Raiders |
| 23 | 15 | Nicholas Morrow | 66.37 | 63.70 | 65.45 | 723 | Raiders |
| 24 | 16 | Kevin Pierre-Louis | 66.24 | 67.70 | 68.08 | 506 | Commanders |
| 25 | 17 | Jayon Brown | 66.21 | 66.30 | 65.73 | 653 | Titans |
| 26 | 18 | Jamie Collins Sr. | 66.18 | 64.20 | 64.36 | 829 | Lions |
| 27 | 19 | Jermaine Carter | 66.16 | 66.50 | 66.86 | 284 | Panthers |
| 28 | 20 | T.J. Edwards | 66.01 | 66.50 | 65.69 | 492 | Eagles |
| 29 | 21 | Joe Giles-Harris | 65.94 | 75.60 | 66.66 | 205 | Jaguars |
| 30 | 22 | Zach Cunningham | 64.85 | 60.90 | 63.74 | 944 | Texans |
| 31 | 23 | Dre Greenlaw | 64.33 | 59.80 | 65.13 | 700 | 49ers |
| 32 | 24 | Jarrad Davis | 63.91 | 62.20 | 63.48 | 329 | Lions |
| 33 | 25 | Nick Vigil | 63.82 | 60.60 | 64.93 | 312 | Chargers |
| 34 | 26 | Robert Spillane | 63.04 | 66.30 | 65.30 | 379 | Steelers |
| 35 | 27 | Rashaan Evans | 62.90 | 53.70 | 65.29 | 895 | Titans |
| 36 | 28 | Alex Singleton | 62.85 | 58.90 | 61.96 | 750 | Eagles |
| 37 | 29 | Neville Hewitt | 62.68 | 59.30 | 63.47 | 1130 | Jets |
| 38 | 30 | Jaylon Smith | 62.50 | 54.20 | 63.87 | 1083 | Cowboys |
| 39 | 31 | Avery Williamson | 62.40 | 51.70 | 65.88 | 665 | Steelers |
| 40 | 32 | Kyle Van Noy | 62.19 | 61.60 | 59.45 | 811 | Dolphins |
| 41 | 33 | Alex Anzalone | 62.04 | 60.90 | 63.53 | 525 | Saints |

### Rotation/backup (74 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 42 | 1 | Azeez Al-Shaair | 61.97 | 61.00 | 62.75 | 305 | 49ers |
| 43 | 2 | Foyesade Oluokun | 61.50 | 56.60 | 61.11 | 895 | Falcons |
| 44 | 3 | Jerome Baker | 61.42 | 55.20 | 61.40 | 868 | Dolphins |
| 45 | 4 | Todd Davis | 61.29 | 60.60 | 62.36 | 281 | Vikings |
| 46 | 5 | Kenneth Murray Jr. | 61.08 | 54.40 | 61.36 | 959 | Chargers |
| 47 | 6 | Malik Harrison | 60.92 | 52.80 | 62.16 | 265 | Ravens |
| 48 | 7 | Troy Reeder | 60.54 | 60.40 | 62.20 | 423 | Rams |
| 49 | 8 | Jon Bostic | 60.41 | 52.70 | 61.38 | 966 | Commanders |
| 50 | 9 | Eric Wilson | 60.24 | 53.50 | 60.99 | 1034 | Vikings |
| 51 | 10 | Devin Bush | 60.15 | 59.70 | 63.45 | 278 | Steelers |
| 52 | 11 | Thomas Davis Sr. | 60.04 | 54.80 | 64.89 | 137 | Commanders |
| 53 | 12 | Joe Schobert | 59.96 | 54.20 | 60.26 | 1110 | Jaguars |
| 54 | 13 | Anfernee Jennings | 59.84 | 54.60 | 62.30 | 293 | Patriots |
| 55 | 14 | Brennan Scarlett | 59.58 | 55.70 | 62.79 | 286 | Texans |
| 56 | 15 | Damien Wilson | 59.29 | 51.80 | 61.69 | 531 | Chiefs |
| 57 | 16 | Ja'Whaun Bentley | 59.24 | 53.20 | 63.36 | 608 | Patriots |
| 58 | 17 | Jordan Hicks | 59.23 | 50.40 | 61.79 | 1024 | Cardinals |
| 59 | 18 | Benardrick McKinney | 58.99 | 57.80 | 62.48 | 234 | Texans |
| 60 | 19 | Josh Bynes | 58.75 | 52.50 | 61.03 | 761 | Bengals |
| 61 | 20 | Kwon Alexander | 58.74 | 58.60 | 61.34 | 668 | Saints |
| 62 | 21 | Anthony Walker Jr. | 58.59 | 48.00 | 61.68 | 697 | Colts |
| 63 | 22 | Anthony Hitchens | 58.42 | 50.50 | 61.10 | 603 | Chiefs |
| 64 | 23 | Logan Wilson | 58.34 | 54.70 | 60.77 | 343 | Bengals |
| 65 | 24 | Duke Riley | 58.06 | 55.10 | 60.35 | 571 | Eagles |
| 66 | 25 | Jordyn Brooks | 57.95 | 47.60 | 62.76 | 367 | Seahawks |
| 67 | 26 | Shaq Thompson | 57.89 | 49.80 | 60.15 | 1031 | Panthers |
| 68 | 27 | De'Vondre Campbell | 57.82 | 49.00 | 59.54 | 880 | Cardinals |
| 69 | 28 | Matt Milano | 57.75 | 55.80 | 58.95 | 335 | Bills |
| 70 | 29 | Bobby Okereke | 57.21 | 49.60 | 59.42 | 685 | Colts |
| 71 | 30 | Tremaine Edmunds | 57.17 | 47.90 | 59.92 | 911 | Bills |
| 72 | 31 | Reggie Ragland | 56.99 | 48.00 | 59.45 | 562 | Lions |
| 73 | 32 | Terez Hall | 56.30 | 51.80 | 63.46 | 259 | Patriots |
| 74 | 33 | Kyzir White | 55.89 | 51.50 | 59.97 | 538 | Chargers |
| 75 | 34 | Cody Barton | 55.84 | 54.30 | 61.04 | 115 | Seahawks |
| 76 | 35 | Vince Williams | 55.82 | 50.60 | 57.22 | 672 | Steelers |
| 77 | 36 | David Long Jr. | 55.82 | 54.20 | 60.15 | 379 | Titans |
| 78 | 37 | Leighton Vander Esch | 55.67 | 50.60 | 60.20 | 460 | Cowboys |
| 79 | 38 | Germaine Pratt | 55.40 | 41.50 | 60.89 | 686 | Bengals |
| 80 | 39 | David Mayo | 55.36 | 48.60 | 59.97 | 194 | Giants |
| 81 | 40 | Will Compton | 55.16 | 50.40 | 62.40 | 124 | Titans |
| 82 | 41 | A.J. Klein | 55.00 | 46.10 | 57.09 | 652 | Bills |
| 83 | 42 | Joe Thomas | 54.95 | 47.70 | 60.94 | 410 | Cowboys |
| 84 | 43 | Chris Board | 54.94 | 51.40 | 58.04 | 263 | Ravens |
| 85 | 44 | Danny Trevathan | 54.76 | 44.10 | 59.89 | 832 | Bears |
| 86 | 45 | Cory Littleton | 54.31 | 46.30 | 56.51 | 849 | Raiders |
| 87 | 46 | Devante Downs | 54.17 | 46.20 | 59.48 | 233 | Giants |
| 88 | 47 | Kamu Grugier-Hill | 54.14 | 50.40 | 56.43 | 207 | Dolphins |
| 89 | 48 | Ben Niemann | 53.98 | 46.50 | 57.72 | 468 | Chiefs |
| 90 | 49 | Nathan Gerry | 53.56 | 45.70 | 60.56 | 479 | Eagles |
| 91 | 50 | Krys Barnes | 53.30 | 43.70 | 59.70 | 421 | Packers |
| 92 | 51 | Devin White | 53.24 | 43.40 | 57.45 | 993 | Buccaneers |
| 93 | 52 | Jacob Phillips | 52.53 | 45.30 | 60.49 | 169 | Browns |
| 94 | 53 | Akeem Davis-Gaither | 52.45 | 40.70 | 56.11 | 314 | Bengals |
| 95 | 54 | Ty Summers | 52.42 | 44.80 | 60.70 | 176 | Packers |
| 96 | 55 | Adarius Taylor | 52.25 | 45.50 | 57.90 | 111 | Panthers |
| 97 | 56 | Kenny Young | 51.87 | 41.60 | 58.30 | 472 | Rams |
| 98 | 57 | Tyrell Adams | 50.99 | 43.50 | 56.92 | 812 | Texans |
| 99 | 58 | Micah Kiser | 49.34 | 40.00 | 58.16 | 559 | Rams |
| 100 | 59 | Christian Kirksey | 49.11 | 43.80 | 57.34 | 548 | Packers |
| 101 | 60 | Tyrel Dodson | 48.69 | 52.30 | 58.08 | 172 | Bills |
| 102 | 61 | Mack Wilson Sr. | 48.67 | 36.20 | 55.17 | 372 | Browns |
| 103 | 62 | Jahlani Tavai | 48.60 | 32.10 | 55.81 | 624 | Lions |
| 104 | 63 | Tahir Whitehead | 48.27 | 31.90 | 56.05 | 398 | Panthers |
| 105 | 64 | Sean Lee | 47.93 | 34.80 | 58.04 | 180 | Cowboys |
| 106 | 65 | Raekwon McMillan | 47.43 | 30.30 | 57.70 | 170 | Raiders |
| 107 | 66 | Harvey Langi | 47.19 | 34.90 | 55.38 | 513 | Jets |
| 108 | 67 | Elandon Roberts | 47.03 | 29.30 | 56.86 | 402 | Dolphins |
| 109 | 68 | Patrick Queen | 46.96 | 29.70 | 54.30 | 858 | Ravens |
| 110 | 69 | Tae Crowder | 46.48 | 36.40 | 55.29 | 403 | Giants |
| 111 | 70 | Troy Dye | 45.00 | 28.80 | 55.04 | 201 | Vikings |
| 112 | 71 | Bryce Hager | 45.00 | 30.20 | 54.64 | 138 | Jets |
| 113 | 72 | Darius Harris | 45.00 | 28.80 | 55.67 | 126 | Chiefs |
| 114 | 73 | Hardy Nickerson | 45.00 | 29.90 | 58.81 | 102 | Vikings |
| 115 | 74 | Dakota Allen | 45.00 | 29.50 | 57.92 | 103 | Jaguars |

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
| 9 | 9 | Ryan Tannehill | 80.49 | 83.29 | 78.10 | 557 | Titans |

### Good (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Matt Ryan | 78.28 | 80.09 | 72.25 | 721 | Falcons |
| 11 | 2 | Philip Rivers | 78.16 | 76.56 | 75.98 | 600 | Colts |
| 12 | 3 | Matthew Stafford | 78.14 | 78.83 | 75.67 | 626 | Lions |
| 13 | 4 | Baker Mayfield | 77.45 | 79.07 | 73.29 | 568 | Browns |
| 14 | 5 | Drew Brees | 77.24 | 78.19 | 77.79 | 421 | Saints |
| 15 | 6 | Dak Prescott | 75.96 | 77.65 | 77.18 | 247 | Cowboys |
| 16 | 7 | Lamar Jackson | 75.64 | 78.33 | 74.10 | 478 | Ravens |
| 17 | 8 | Kyler Murray | 74.37 | 72.78 | 71.13 | 674 | Cardinals |

### Starter (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Jared Goff | 73.56 | 73.30 | 69.45 | 617 | Rams |
| 19 | 2 | Justin Herbert | 73.52 | 78.40 | 74.27 | 689 | Chargers |
| 20 | 3 | Ryan Fitzpatrick | 72.01 | 73.54 | 73.40 | 316 | Dolphins |
| 21 | 4 | Ben Roethlisberger | 69.07 | 68.59 | 68.97 | 672 | Steelers |
| 22 | 5 | Daniel Jones | 68.51 | 71.76 | 63.97 | 552 | Giants |
| 23 | 6 | Gardner Minshew | 68.39 | 67.53 | 69.94 | 396 | Jaguars |
| 24 | 7 | Teddy Bridgewater | 68.14 | 67.07 | 71.70 | 594 | Panthers |
| 25 | 8 | Jimmy Garoppolo | 67.48 | 70.76 | 73.09 | 159 | 49ers |
| 26 | 9 | Andy Dalton | 66.43 | 69.44 | 64.81 | 393 | Cowboys |
| 27 | 10 | Joe Burrow | 65.06 | 74.30 | 66.69 | 472 | Bengals |
| 28 | 11 | Cam Newton | 64.52 | 67.16 | 65.53 | 444 | Patriots |
| 29 | 12 | Carson Wentz | 63.97 | 66.29 | 59.92 | 544 | Eagles |
| 30 | 13 | Mitch Trubisky | 63.76 | 60.21 | 68.00 | 340 | Bears |
| 31 | 14 | Taysom Hill | 63.26 | 69.78 | 74.28 | 161 | Saints |
| 32 | 15 | C.J. Beathard | 62.10 | 64.85 | 73.59 | 123 | 49ers |

### Rotation/backup (12 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 33 | 1 | Nick Foles | 61.80 | 66.97 | 64.25 | 350 | Bears |
| 34 | 2 | Drew Lock | 61.44 | 62.67 | 63.19 | 510 | Broncos |
| 35 | 3 | Tua Tagovailoa | 60.64 | 63.90 | 63.29 | 347 | Dolphins |
| 36 | 4 | Alex Smith | 60.47 | 67.18 | 62.15 | 289 | Commanders |
| 37 | 5 | Mike Glennon | 60.42 | 68.87 | 63.54 | 198 | Jaguars |
| 38 | 6 | Nick Mullens | 60.01 | 57.79 | 67.54 | 365 | 49ers |
| 39 | 7 | Kyle Allen | 59.76 | 56.16 | 70.04 | 107 | Commanders |
| 40 | 8 | Joe Flacco | 58.84 | 61.72 | 63.73 | 160 | Jets |
| 41 | 9 | Jalen Hurts | 58.37 | 57.50 | 63.58 | 197 | Eagles |
| 42 | 10 | Sam Darnold | 57.25 | 56.93 | 57.90 | 449 | Jets |
| 43 | 11 | Brandon Allen | 57.08 | 53.96 | 62.24 | 167 | Bengals |
| 44 | 12 | Jake Luton | 52.13 | 37.70 | 55.56 | 123 | Jaguars |

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

### Good (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 10 | 1 | Marcus Williams | 79.90 | 72.40 | 82.09 | 880 | Saints |
| 11 | 2 | Darnell Savage | 77.74 | 76.70 | 75.70 | 876 | Packers |
| 12 | 3 | Terrell Edmunds | 77.61 | 76.00 | 75.03 | 865 | Steelers |
| 13 | 4 | Jimmie Ward | 76.65 | 72.50 | 78.69 | 851 | 49ers |
| 14 | 5 | Ronnie Harrison | 76.26 | 74.40 | 76.99 | 325 | Browns |
| 15 | 6 | Jeff Heath | 76.08 | 72.80 | 76.60 | 415 | Raiders |
| 16 | 7 | Mike Edwards | 75.94 | 77.20 | 73.93 | 189 | Buccaneers |
| 17 | 8 | Marcus Epps | 75.82 | 74.80 | 77.02 | 365 | Eagles |
| 18 | 9 | Jeremy Reaves | 75.64 | 76.90 | 81.47 | 263 | Commanders |
| 19 | 10 | Chuck Clark | 75.50 | 69.50 | 76.06 | 1063 | Ravens |
| 20 | 11 | Rodney McLeod | 75.33 | 77.30 | 74.11 | 873 | Eagles |
| 21 | 12 | Budda Baker | 74.47 | 68.90 | 74.53 | 1005 | Cardinals |
| 22 | 13 | Khari Willis | 74.07 | 69.10 | 75.30 | 842 | Colts |

### Starter (44 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 23 | 1 | Kevin Byard | 73.48 | 63.90 | 75.70 | 1103 | Titans |
| 24 | 2 | Andrew Wingard | 72.81 | 69.80 | 75.60 | 461 | Jaguars |
| 25 | 3 | Amani Hooker | 72.60 | 70.80 | 69.63 | 470 | Titans |
| 26 | 4 | Vonn Bell | 72.09 | 64.60 | 73.85 | 1061 | Bengals |
| 27 | 5 | Micah Hyde | 71.54 | 66.80 | 71.26 | 938 | Bills |
| 28 | 6 | Tashaun Gipson Sr. | 71.18 | 64.70 | 71.97 | 1054 | Bears |
| 29 | 7 | Anthony Harris | 70.73 | 63.60 | 72.99 | 1074 | Vikings |
| 30 | 8 | Donovan Wilson | 70.57 | 70.50 | 74.79 | 673 | Cowboys |
| 31 | 9 | Taylor Rapp | 70.36 | 69.10 | 71.99 | 365 | Rams |
| 32 | 10 | Devin McCourty | 70.35 | 64.30 | 70.21 | 960 | Patriots |
| 33 | 11 | Rayshawn Jenkins | 70.18 | 70.00 | 67.90 | 860 | Chargers |
| 34 | 12 | Duron Harmon | 69.43 | 64.40 | 68.61 | 1102 | Lions |
| 35 | 13 | Eric Rowe | 69.26 | 64.10 | 68.54 | 919 | Dolphins |
| 36 | 14 | Jalen Mills | 69.20 | 66.90 | 67.09 | 1014 | Eagles |
| 37 | 15 | Jarrod Wilson | 68.81 | 64.70 | 70.30 | 765 | Jaguars |
| 38 | 16 | Jordan Whitehead | 68.71 | 63.70 | 68.71 | 920 | Buccaneers |
| 39 | 17 | Justin Reid | 68.69 | 67.20 | 67.40 | 888 | Texans |
| 40 | 18 | Tavon Wilson | 68.22 | 64.20 | 70.28 | 219 | Colts |
| 41 | 19 | Adrian Phillips | 68.15 | 62.20 | 70.76 | 747 | Patriots |
| 42 | 20 | DeShon Elliott | 68.07 | 66.20 | 68.90 | 1044 | Ravens |
| 43 | 21 | Jalen Thompson | 68.02 | 68.40 | 72.31 | 232 | Cardinals |
| 44 | 22 | Quandre Diggs | 67.70 | 58.60 | 71.48 | 1075 | Seahawks |
| 45 | 23 | Tyrann Mathieu | 67.49 | 60.40 | 68.57 | 982 | Chiefs |
| 46 | 24 | Xavier Woods | 67.28 | 62.50 | 67.55 | 990 | Cowboys |
| 47 | 25 | Xavier McKinney | 66.95 | 69.20 | 74.70 | 211 | Giants |
| 48 | 26 | Bobby McCain | 66.95 | 63.10 | 68.09 | 923 | Dolphins |
| 49 | 27 | Chris Banjo | 66.76 | 63.60 | 72.30 | 436 | Cardinals |
| 50 | 28 | Keanu Neal | 66.65 | 66.20 | 70.49 | 917 | Falcons |
| 51 | 29 | Jeremy Chinn | 66.44 | 64.40 | 64.66 | 967 | Panthers |
| 52 | 30 | Ricardo Allen | 65.34 | 60.10 | 69.45 | 604 | Falcons |
| 53 | 31 | Jaquiski Tartt | 65.04 | 63.50 | 69.50 | 374 | 49ers |
| 54 | 32 | Antoine Winfield Jr. | 65.04 | 55.00 | 67.57 | 1034 | Buccaneers |
| 55 | 33 | Kenny Vaccaro | 64.56 | 63.60 | 63.21 | 871 | Titans |
| 56 | 34 | Brandon Jones | 64.43 | 59.20 | 63.75 | 385 | Dolphins |
| 57 | 35 | Jordan Fuller | 64.35 | 58.60 | 68.19 | 708 | Rams |
| 58 | 36 | D.J. Swearinger Sr. | 64.06 | 60.80 | 68.12 | 124 | Saints |
| 59 | 37 | Damontae Kazee | 63.68 | 61.70 | 68.65 | 241 | Falcons |
| 60 | 38 | Malcolm Jenkins | 63.51 | 52.00 | 67.01 | 1036 | Saints |
| 61 | 39 | Sharrod Neasman | 63.32 | 59.40 | 66.66 | 292 | Falcons |
| 62 | 40 | Eddie Jackson | 63.32 | 55.80 | 64.58 | 1059 | Bears |
| 63 | 41 | Jabrill Peppers | 63.06 | 57.20 | 64.88 | 912 | Giants |
| 64 | 42 | Nick Scott | 62.43 | 59.70 | 70.11 | 193 | Rams |
| 65 | 43 | Erik Harris | 62.36 | 57.90 | 62.74 | 724 | Raiders |
| 66 | 44 | Julian Love | 62.02 | 54.90 | 63.63 | 722 | Giants |

### Rotation/backup (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 67 | 1 | Juan Thornhill | 61.92 | 55.50 | 62.04 | 765 | Chiefs |
| 68 | 2 | Julian Blackmon | 61.83 | 54.50 | 63.59 | 916 | Colts |
| 69 | 3 | Troy Apke | 61.75 | 58.00 | 65.40 | 441 | Commanders |
| 70 | 4 | Deshazor Everett | 61.25 | 59.30 | 64.84 | 357 | Commanders |
| 71 | 5 | Will Redmond | 61.08 | 62.50 | 60.65 | 340 | Packers |
| 72 | 6 | Tarvarius Moore | 61.01 | 56.10 | 63.04 | 541 | 49ers |
| 73 | 7 | Jamal Adams | 60.63 | 53.10 | 64.19 | 784 | Seahawks |
| 74 | 8 | Will Harris | 60.22 | 55.50 | 59.20 | 312 | Lions |
| 75 | 9 | Jahleel Addae | 60.15 | 56.70 | 61.42 | 211 | Chargers |
| 76 | 10 | Adrian Colbert | 60.11 | 60.00 | 67.49 | 104 | Giants |
| 77 | 11 | Kyle Dugger | 59.92 | 55.40 | 60.85 | 520 | Patriots |
| 78 | 12 | Juston Burris | 59.62 | 59.50 | 58.67 | 790 | Panthers |
| 79 | 13 | Daniel Sorensen | 59.52 | 58.50 | 58.44 | 883 | Chiefs |
| 80 | 14 | Tre Boston | 58.57 | 49.90 | 60.60 | 1037 | Panthers |
| 81 | 15 | Raven Greene | 58.07 | 60.10 | 62.75 | 324 | Packers |
| 82 | 16 | Daniel Thomas | 58.01 | 62.40 | 64.34 | 162 | Jaguars |
| 83 | 17 | Landon Collins | 57.91 | 56.10 | 60.79 | 398 | Commanders |
| 84 | 18 | Eric Murray | 57.84 | 50.40 | 59.89 | 941 | Texans |
| 85 | 19 | Armani Watts | 57.28 | 59.70 | 62.44 | 102 | Chiefs |
| 86 | 20 | Matthias Farley | 57.14 | 60.50 | 62.09 | 201 | Jets |
| 87 | 21 | Deionte Thompson | 56.20 | 55.00 | 59.46 | 332 | Cardinals |
| 88 | 22 | Karl Joseph | 55.63 | 48.40 | 60.65 | 660 | Browns |
| 89 | 23 | Sam Franklin Jr. | 55.34 | 51.60 | 60.96 | 251 | Panthers |
| 90 | 24 | Dean Marlowe | 54.26 | 52.10 | 59.04 | 230 | Bills |
| 91 | 25 | K'Von Wallace | 54.19 | 54.00 | 54.31 | 203 | Eagles |
| 92 | 26 | Darian Thompson | 54.11 | 51.30 | 57.13 | 479 | Cowboys |
| 93 | 27 | Ashtyn Davis | 53.49 | 45.10 | 63.25 | 402 | Jets |
| 94 | 28 | Sheldrick Redwine | 53.14 | 51.50 | 57.10 | 276 | Browns |
| 95 | 29 | Terrence Brooks | 52.66 | 47.30 | 56.65 | 254 | Patriots |
| 96 | 30 | Tracy Walker III | 52.17 | 40.80 | 57.05 | 755 | Lions |
| 97 | 31 | Bradley McDougald | 52.11 | 42.80 | 59.15 | 432 | Jets |
| 98 | 32 | Nasir Adderley | 51.80 | 42.50 | 59.95 | 886 | Chargers |
| 99 | 33 | Josh Jones | 50.87 | 39.20 | 57.81 | 700 | Jaguars |
| 100 | 34 | Andrew Sendejo | 49.98 | 40.90 | 56.14 | 918 | Browns |
| 101 | 35 | Marcell Harris | 47.33 | 36.60 | 56.77 | 348 | 49ers |
| 102 | 36 | Johnathan Abram | 45.00 | 30.30 | 56.88 | 856 | Raiders |

## T — Tackle

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | David Bakhtiari | 96.69 | 91.80 | 95.78 | 758 | Packers |
| 2 | 2 | Trent Williams | 96.30 | 91.90 | 95.06 | 957 | 49ers |
| 3 | 3 | Garett Bolles | 96.02 | 90.60 | 95.47 | 1015 | Broncos |
| 4 | 4 | Andrew Whitworth | 94.69 | 88.70 | 94.52 | 600 | Rams |
| 5 | 5 | D.J. Humphries | 94.18 | 88.30 | 93.94 | 1129 | Cardinals |
| 6 | 6 | Duane Brown | 91.56 | 87.30 | 90.24 | 1048 | Seahawks |
| 7 | 7 | Terron Armstead | 91.12 | 86.30 | 90.17 | 857 | Saints |
| 8 | 8 | Jack Conklin | 90.32 | 84.30 | 90.17 | 999 | Browns |
| 9 | 9 | Ryan Ramczyk | 88.61 | 81.50 | 89.19 | 1038 | Saints |
| 10 | 10 | Isaiah Wynn | 88.59 | 82.60 | 88.42 | 641 | Patriots |
| 11 | 11 | Tristan Wirfs | 88.18 | 81.80 | 88.26 | 1073 | Buccaneers |
| 12 | 12 | Morgan Moses | 88.17 | 80.60 | 89.05 | 1065 | Commanders |
| 13 | 13 | Taylor Moton | 88.07 | 81.60 | 88.22 | 1032 | Panthers |
| 14 | 14 | Taylor Decker | 87.94 | 82.00 | 87.73 | 1048 | Lions |
| 15 | 15 | Mike McGlinchey | 87.81 | 79.60 | 89.12 | 1091 | 49ers |
| 16 | 16 | Rob Havenstein | 87.76 | 80.20 | 88.64 | 1117 | Rams |
| 17 | 17 | Ronnie Stanley | 87.36 | 79.90 | 88.16 | 312 | Ravens |
| 18 | 18 | Braden Smith | 87.35 | 80.10 | 88.02 | 937 | Colts |
| 19 | 19 | Brian O'Neill | 86.91 | 78.00 | 88.68 | 1070 | Vikings |
| 20 | 20 | Eric Fisher | 85.79 | 80.00 | 85.49 | 1049 | Chiefs |
| 21 | 21 | Dion Dawkins | 84.96 | 78.10 | 85.36 | 1034 | Bills |
| 22 | 22 | Orlando Brown Jr. | 84.26 | 76.50 | 85.26 | 1027 | Ravens |
| 23 | 23 | Rick Wagner | 84.20 | 77.00 | 84.84 | 610 | Packers |
| 24 | 24 | Cornelius Lucas | 83.86 | 78.20 | 83.47 | 536 | Commanders |
| 25 | 25 | Laremy Tunsil | 83.20 | 75.40 | 84.24 | 817 | Texans |
| 26 | 26 | Mekhi Becton | 83.19 | 74.40 | 84.88 | 691 | Jets |
| 27 | 27 | Charles Leno Jr. | 82.91 | 74.60 | 84.28 | 1066 | Bears |
| 28 | 28 | Kendall Lamm | 82.65 | 76.80 | 82.39 | 113 | Browns |
| 29 | 29 | Mitchell Schwartz | 82.28 | 74.70 | 83.16 | 357 | Chiefs |
| 30 | 30 | Jake Matthews | 81.97 | 75.50 | 82.11 | 1113 | Falcons |
| 31 | 31 | Lane Johnson | 81.72 | 71.90 | 84.10 | 405 | Eagles |
| 32 | 32 | Russell Okung | 81.66 | 73.00 | 83.27 | 406 | Panthers |
| 33 | 33 | Donovan Smith | 81.66 | 71.80 | 84.07 | 962 | Buccaneers |
| 34 | 34 | James Hurst | 81.31 | 70.10 | 84.62 | 377 | Saints |
| 35 | 35 | Brandon Shell | 81.05 | 72.70 | 82.45 | 673 | Seahawks |
| 36 | 36 | Alejandro Villanueva | 80.87 | 74.60 | 80.88 | 1098 | Steelers |
| 37 | 37 | Rashod Hill | 80.87 | 72.20 | 82.49 | 121 | Vikings |
| 38 | 38 | Anthony Castonzo | 80.86 | 73.40 | 81.66 | 749 | Colts |
| 39 | 39 | Demar Dotson | 80.79 | 70.80 | 83.29 | 451 | Broncos |
| 40 | 40 | Bobby Massie | 80.38 | 72.60 | 81.40 | 470 | Bears |
| 41 | 41 | Riley Reiff | 80.15 | 71.30 | 81.89 | 1003 | Vikings |
| 42 | 42 | Mike Remmers | 80.05 | 70.10 | 82.52 | 709 | Chiefs |

### Good (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 43 | 1 | Kolton Miller | 79.76 | 73.00 | 80.10 | 961 | Raiders |
| 44 | 2 | Jordan Mailata | 79.49 | 70.40 | 81.39 | 733 | Eagles |
| 45 | 3 | Trent Brown | 79.49 | 68.90 | 82.38 | 282 | Raiders |
| 46 | 4 | Matt Peart | 79.30 | 69.70 | 81.53 | 150 | Giants |
| 47 | 5 | Kelvin Beachum | 78.98 | 69.00 | 81.47 | 1126 | Cardinals |
| 48 | 6 | Tyron Smith | 77.82 | 67.80 | 80.33 | 154 | Cowboys |
| 49 | 7 | Jonah Williams | 77.75 | 70.10 | 78.69 | 634 | Bengals |
| 50 | 8 | Roderick Johnson | 77.61 | 64.50 | 82.18 | 245 | Texans |
| 51 | 9 | Dennis Kelly | 77.52 | 65.90 | 81.10 | 1049 | Titans |
| 52 | 10 | Bobby Hart | 77.06 | 66.30 | 80.06 | 872 | Bengals |
| 53 | 11 | Ty Sambrailo | 76.93 | 65.60 | 80.31 | 415 | Titans |
| 54 | 12 | Cedric Ogbuehi | 76.92 | 65.50 | 80.36 | 277 | Seahawks |
| 55 | 13 | Jason Peters | 76.42 | 67.60 | 78.14 | 509 | Eagles |
| 56 | 14 | Chuma Edoga | 76.20 | 61.30 | 81.96 | 235 | Jets |
| 57 | 15 | Kaleb McGary | 76.11 | 64.30 | 79.82 | 890 | Falcons |
| 58 | 16 | Tyrell Crosby | 75.12 | 64.20 | 78.23 | 657 | Lions |
| 59 | 17 | Tytus Howard | 74.81 | 62.10 | 79.11 | 811 | Texans |
| 60 | 18 | Jesse Davis | 74.51 | 62.60 | 78.28 | 1055 | Dolphins |
| 61 | 19 | Josh Wells | 74.44 | 65.10 | 76.50 | 111 | Buccaneers |
| 62 | 20 | Jedrick Wills Jr. | 74.34 | 61.50 | 78.73 | 957 | Browns |
| 63 | 21 | Justin Herron | 74.13 | 63.40 | 77.11 | 352 | Patriots |

### Starter (29 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 64 | 1 | Taylor Lewan | 73.79 | 61.80 | 77.61 | 239 | Titans |
| 65 | 2 | Chaz Green | 73.72 | 61.90 | 77.43 | 209 | Colts |
| 66 | 3 | David Quessenberry | 73.41 | 61.70 | 77.05 | 437 | Titans |
| 67 | 4 | Bryan Bulaga | 73.23 | 61.60 | 76.81 | 444 | Chargers |
| 68 | 5 | Andrew Thomas | 73.16 | 62.40 | 76.16 | 978 | Giants |
| 69 | 6 | George Fant | 73.15 | 61.60 | 76.69 | 829 | Jets |
| 70 | 7 | Cam Robinson | 73.10 | 61.70 | 76.54 | 973 | Jaguars |
| 71 | 8 | Geron Christian | 73.00 | 62.30 | 75.96 | 398 | Commanders |
| 72 | 9 | Trent Scott | 72.91 | 60.20 | 77.21 | 347 | Panthers |
| 73 | 10 | Sam Young | 72.38 | 59.00 | 77.14 | 382 | Raiders |
| 74 | 11 | Cam Fleming | 72.38 | 58.40 | 77.54 | 913 | Giants |
| 75 | 12 | Chukwuma Okorafor | 71.05 | 57.50 | 75.91 | 1033 | Steelers |
| 76 | 13 | Jawaan Taylor | 70.69 | 56.50 | 75.99 | 1037 | Jaguars |
| 77 | 14 | Conor McDermott | 69.55 | 54.20 | 75.62 | 247 | Jets |
| 78 | 15 | Matt Gono | 69.40 | 54.90 | 74.90 | 336 | Falcons |
| 79 | 16 | Trey Pipkins III | 69.34 | 54.80 | 74.87 | 571 | Chargers |
| 80 | 17 | Sam Tevi | 68.05 | 52.90 | 73.98 | 1024 | Chargers |
| 81 | 18 | Matt Nelson | 67.50 | 55.80 | 71.14 | 242 | Lions |
| 82 | 19 | Austin Jackson | 67.07 | 52.50 | 72.61 | 848 | Dolphins |
| 83 | 20 | David Sharpe | 67.04 | 50.90 | 73.63 | 184 | Commanders |
| 84 | 21 | Hakeem Adeniji | 66.89 | 51.60 | 72.92 | 233 | Bengals |
| 85 | 22 | Greg Little | 66.56 | 44.10 | 77.36 | 134 | Panthers |
| 86 | 23 | Terence Steele | 66.17 | 50.30 | 72.58 | 970 | Cowboys |
| 87 | 24 | Brandon Parker | 65.81 | 48.90 | 72.91 | 345 | Raiders |
| 88 | 25 | Fred Johnson | 65.11 | 48.40 | 72.08 | 491 | Bengals |
| 89 | 26 | Brandon Knight | 65.07 | 48.50 | 71.95 | 774 | Cowboys |
| 90 | 27 | Le'Raven Clark | 64.96 | 49.10 | 71.36 | 148 | Colts |
| 91 | 28 | Calvin Anderson | 64.39 | 55.10 | 66.42 | 132 | Broncos |
| 92 | 29 | Justin Skule | 64.00 | 45.40 | 72.23 | 255 | 49ers |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | George Kittle | 85.30 | 84.90 | 81.40 | 262 | 49ers |
| 2 | 2 | Travis Kelce | 83.96 | 93.50 | 73.43 | 635 | Chiefs |
| 3 | 3 | Mark Andrews | 83.35 | 81.80 | 80.21 | 364 | Ravens |
| 4 | 4 | Mo Alie-Cox | 82.15 | 78.20 | 80.62 | 226 | Colts |
| 5 | 5 | Darren Waller | 82.03 | 86.50 | 74.89 | 620 | Raiders |
| 6 | 6 | Pharaoh Brown | 80.69 | 81.30 | 76.12 | 126 | Texans |

### Good (11 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Rob Gronkowski | 79.27 | 72.10 | 79.89 | 498 | Buccaneers |
| 8 | 2 | Richard Rodgers | 78.43 | 88.60 | 67.48 | 174 | Eagles |
| 9 | 3 | Adam Trautman | 77.12 | 79.40 | 71.44 | 172 | Saints |
| 10 | 4 | Dallas Goedert | 76.35 | 79.60 | 70.01 | 390 | Eagles |
| 11 | 5 | Anthony Firkser | 75.69 | 77.10 | 70.59 | 250 | Titans |
| 12 | 6 | Donald Parham Jr. | 75.59 | 64.50 | 78.82 | 136 | Chargers |
| 13 | 7 | Foster Moreau | 74.76 | 67.20 | 75.64 | 117 | Raiders |
| 14 | 8 | Will Dissly | 74.68 | 66.10 | 76.23 | 308 | Seahawks |
| 15 | 9 | Jordan Reed | 74.51 | 66.20 | 75.88 | 190 | 49ers |
| 16 | 10 | Mike Gesicki | 74.23 | 78.10 | 67.48 | 473 | Dolphins |
| 17 | 11 | Dan Arnold | 74.12 | 61.70 | 78.24 | 312 | Cardinals |

### Starter (55 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | T.J. Hockenson | 73.24 | 75.50 | 67.56 | 538 | Lions |
| 19 | 2 | Darren Fells | 73.06 | 69.50 | 71.27 | 313 | Texans |
| 20 | 3 | Hunter Henry | 72.78 | 69.30 | 70.94 | 586 | Chargers |
| 21 | 4 | Tyler Higbee | 72.61 | 69.20 | 70.71 | 424 | Rams |
| 22 | 5 | David Njoku | 72.60 | 66.90 | 72.23 | 216 | Browns |
| 23 | 6 | Jonnu Smith | 72.57 | 75.20 | 66.65 | 353 | Titans |
| 24 | 7 | Robert Tonyan | 72.43 | 68.00 | 71.21 | 426 | Packers |
| 25 | 8 | Jared Cook | 72.16 | 71.90 | 68.16 | 338 | Saints |
| 26 | 9 | Noah Fant | 72.13 | 71.20 | 68.58 | 466 | Broncos |
| 27 | 10 | Kyle Rudolph | 72.02 | 66.50 | 71.53 | 296 | Vikings |
| 28 | 11 | Trey Burton | 71.97 | 66.70 | 71.31 | 242 | Colts |
| 29 | 12 | Marcedes Lewis | 71.89 | 68.00 | 70.31 | 191 | Packers |
| 30 | 13 | Jack Doyle | 71.80 | 67.00 | 70.83 | 244 | Colts |
| 31 | 14 | Greg Olsen | 71.79 | 62.00 | 74.15 | 316 | Seahawks |
| 32 | 15 | Jordan Akins | 71.23 | 71.80 | 66.68 | 301 | Texans |
| 33 | 16 | Austin Hooper | 70.96 | 68.30 | 68.56 | 366 | Browns |
| 34 | 17 | Blake Bell | 70.38 | 63.10 | 71.07 | 154 | Cowboys |
| 35 | 18 | Adam Shaheen | 70.34 | 66.00 | 69.06 | 160 | Dolphins |
| 36 | 19 | Tyler Kroft | 70.13 | 64.30 | 69.85 | 135 | Bills |
| 37 | 20 | Durham Smythe | 69.88 | 68.70 | 66.50 | 207 | Dolphins |
| 38 | 21 | Jimmy Graham | 69.87 | 62.70 | 70.49 | 445 | Bears |
| 39 | 22 | Cameron Brate | 69.81 | 67.80 | 66.99 | 198 | Buccaneers |
| 40 | 23 | Zach Ertz | 69.80 | 57.30 | 73.97 | 424 | Eagles |
| 41 | 24 | Nick Boyle | 69.71 | 70.60 | 64.95 | 151 | Ravens |
| 42 | 25 | Irv Smith Jr. | 69.07 | 70.00 | 64.29 | 322 | Vikings |
| 43 | 26 | Chris Herndon | 68.81 | 57.60 | 72.11 | 438 | Jets |
| 44 | 27 | Gerald Everett | 68.47 | 62.80 | 68.09 | 353 | Rams |
| 45 | 28 | Dawson Knox | 68.47 | 61.40 | 69.02 | 249 | Bills |
| 46 | 29 | Evan Engram | 68.34 | 60.60 | 69.34 | 567 | Giants |
| 47 | 30 | Jesse James | 68.33 | 61.20 | 68.92 | 237 | Lions |
| 48 | 31 | Maxx Williams | 68.17 | 67.70 | 64.32 | 122 | Cardinals |
| 49 | 32 | Tyler Eifert | 67.48 | 62.20 | 66.84 | 417 | Jaguars |
| 50 | 33 | Harrison Bryant | 67.40 | 60.70 | 67.70 | 294 | Browns |
| 51 | 34 | Jason Witten | 67.15 | 60.80 | 67.21 | 179 | Raiders |
| 52 | 35 | Kaden Smith | 66.84 | 66.90 | 62.64 | 203 | Giants |
| 53 | 36 | Drew Sample | 66.67 | 61.40 | 66.02 | 513 | Bengals |
| 54 | 37 | Ross Dwelley | 66.43 | 61.30 | 65.69 | 234 | 49ers |
| 55 | 38 | Eric Ebron | 66.38 | 55.70 | 69.33 | 536 | Steelers |
| 56 | 39 | Dalton Schultz | 66.32 | 63.90 | 63.76 | 620 | Cowboys |
| 57 | 40 | Logan Thomas | 66.31 | 64.50 | 63.35 | 661 | Commanders |
| 58 | 41 | Darrell Daniels | 66.20 | 59.10 | 66.76 | 117 | Cardinals |
| 59 | 42 | James O'Shaughnessy | 66.00 | 57.70 | 67.37 | 264 | Jaguars |
| 60 | 43 | Hayden Hurst | 65.88 | 58.90 | 66.37 | 588 | Falcons |
| 61 | 44 | Geoff Swaim | 65.66 | 62.20 | 63.80 | 126 | Titans |
| 62 | 45 | Cole Kmet | 65.65 | 58.70 | 66.12 | 326 | Bears |
| 63 | 46 | Josh Hill | 65.43 | 62.70 | 63.08 | 141 | Saints |
| 64 | 47 | Jacob Hollister | 64.98 | 62.50 | 62.47 | 220 | Seahawks |
| 65 | 48 | Nick Vannett | 64.73 | 57.20 | 65.58 | 154 | Broncos |
| 66 | 49 | Devin Asiasi | 64.65 | 51.60 | 69.19 | 109 | Patriots |
| 67 | 50 | Chris Manhertz | 64.40 | 58.10 | 64.43 | 220 | Panthers |
| 68 | 51 | Ryan Izzo | 64.29 | 52.10 | 68.25 | 292 | Patriots |
| 69 | 52 | Demetrius Harris | 64.28 | 58.50 | 63.97 | 109 | Bears |
| 70 | 53 | Ryan Griffin | 64.10 | 56.10 | 65.26 | 157 | Jets |
| 71 | 54 | Levine Toilolo | 63.89 | 56.50 | 64.65 | 102 | Giants |
| 72 | 55 | Tyler Conklin | 62.55 | 52.60 | 65.01 | 257 | Vikings |

### Rotation/backup (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 73 | 1 | Nick Keizer | 59.95 | 47.50 | 64.09 | 127 | Chiefs |
| 74 | 2 | Ian Thomas | 59.60 | 42.80 | 66.64 | 501 | Panthers |
| 75 | 3 | Vance McDonald | 59.24 | 45.60 | 64.16 | 239 | Steelers |
| 76 | 4 | Luke Stocker | 57.43 | 48.10 | 59.49 | 212 | Falcons |

## WR — Wide Receiver

- **Season used:** `2020`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (28 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Justin Jefferson | 89.40 | 90.40 | 84.57 | 563 | Vikings |
| 2 | 2 | A.J. Brown | 89.25 | 90.10 | 84.51 | 429 | Titans |
| 3 | 3 | Davante Adams | 87.56 | 92.20 | 80.30 | 489 | Packers |
| 4 | 4 | Julio Jones | 86.06 | 86.30 | 81.74 | 314 | Falcons |
| 5 | 5 | Will Fuller V | 85.72 | 86.20 | 81.24 | 406 | Texans |
| 6 | 6 | Stefon Diggs | 85.38 | 89.40 | 78.53 | 658 | Bills |
| 7 | 7 | Antonio Brown | 85.22 | 85.80 | 80.66 | 249 | Buccaneers |
| 8 | 8 | DeAndre Hopkins | 84.75 | 87.10 | 79.01 | 663 | Cardinals |
| 9 | 9 | Corey Davis | 84.68 | 86.90 | 79.03 | 398 | Titans |
| 10 | 10 | Tyreek Hill | 84.06 | 83.20 | 80.46 | 621 | Chiefs |
| 11 | 11 | Michael Thomas | 84.05 | 85.20 | 79.11 | 215 | Saints |
| 12 | 12 | Kenny Golladay | 83.76 | 81.60 | 81.04 | 150 | Lions |
| 13 | 13 | Allen Robinson II | 83.69 | 88.40 | 76.38 | 625 | Bears |
| 14 | 14 | Chris Godwin | 83.27 | 80.30 | 81.09 | 462 | Buccaneers |
| 15 | 15 | Calvin Ridley | 82.85 | 84.90 | 77.32 | 593 | Falcons |
| 16 | 16 | Adam Thielen | 82.54 | 87.40 | 75.14 | 536 | Vikings |
| 17 | 17 | DJ Moore | 82.39 | 79.10 | 80.41 | 568 | Panthers |
| 18 | 18 | Keenan Allen | 81.87 | 84.90 | 75.68 | 554 | Chargers |
| 19 | 19 | Jarvis Landry | 81.84 | 84.20 | 76.10 | 428 | Browns |
| 20 | 20 | D.K. Metcalf | 81.73 | 82.70 | 76.91 | 667 | Seahawks |
| 21 | 21 | Donovan Peoples-Jones | 81.61 | 71.80 | 83.99 | 138 | Browns |
| 22 | 22 | Brandin Cooks | 81.51 | 81.10 | 77.61 | 583 | Texans |
| 23 | 23 | Cole Beasley | 81.36 | 84.90 | 74.83 | 497 | Bills |
| 24 | 24 | Cooper Kupp | 80.67 | 80.80 | 76.42 | 542 | Rams |
| 25 | 25 | T.Y. Hilton | 80.61 | 77.70 | 78.38 | 465 | Colts |
| 26 | 26 | Terry McLaurin | 80.48 | 78.00 | 77.96 | 624 | Commanders |
| 27 | 27 | Deebo Samuel | 80.21 | 79.80 | 76.31 | 177 | 49ers |
| 28 | 28 | Chase Claypool | 80.19 | 75.60 | 79.08 | 474 | Steelers |

### Good (45 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 29 | 1 | Rashard Higgins | 79.97 | 77.40 | 77.52 | 306 | Browns |
| 30 | 2 | Bryan Edwards | 79.57 | 69.50 | 82.12 | 143 | Raiders |
| 31 | 3 | Jakobi Meyers | 79.25 | 78.60 | 75.52 | 346 | Patriots |
| 32 | 4 | Mike Evans | 79.07 | 74.10 | 78.21 | 599 | Buccaneers |
| 33 | 5 | Odell Beckham Jr. | 79.02 | 75.30 | 77.34 | 189 | Browns |
| 34 | 6 | Tim Patrick | 79.01 | 74.50 | 77.85 | 455 | Broncos |
| 35 | 7 | Brandon Aiyuk | 78.98 | 80.10 | 74.07 | 455 | 49ers |
| 36 | 8 | Amari Cooper | 78.95 | 75.90 | 76.82 | 638 | Cowboys |
| 37 | 9 | Tee Higgins | 78.62 | 75.90 | 76.26 | 527 | Bengals |
| 38 | 10 | DeSean Jackson | 78.50 | 66.80 | 82.13 | 134 | Eagles |
| 39 | 11 | Mike Williams | 78.36 | 72.70 | 77.96 | 514 | Chargers |
| 40 | 12 | DeVante Parker | 78.20 | 76.40 | 75.23 | 484 | Dolphins |
| 41 | 13 | Hunter Renfrow | 77.88 | 74.80 | 75.77 | 401 | Raiders |
| 42 | 14 | Collin Johnson | 77.75 | 73.40 | 76.49 | 180 | Jaguars |
| 43 | 15 | Emmanuel Sanders | 77.57 | 74.60 | 75.39 | 378 | Saints |
| 44 | 16 | Tyler Boyd | 77.24 | 75.80 | 74.04 | 537 | Bengals |
| 45 | 17 | Tyler Lockett | 77.18 | 76.40 | 73.53 | 660 | Seahawks |
| 46 | 18 | Sterling Shepard | 76.92 | 79.60 | 70.97 | 395 | Giants |
| 47 | 19 | Robert Woods | 76.73 | 71.30 | 76.18 | 612 | Rams |
| 48 | 20 | Jamison Crowder | 76.67 | 75.30 | 73.41 | 411 | Jets |
| 49 | 21 | Denzel Mims | 76.64 | 69.40 | 77.30 | 264 | Jets |
| 50 | 22 | John Brown | 76.60 | 70.30 | 76.63 | 306 | Bills |
| 51 | 23 | Mecole Hardman Jr. | 76.54 | 69.80 | 76.86 | 361 | Chiefs |
| 52 | 24 | Danny Amendola | 76.34 | 74.90 | 73.14 | 352 | Lions |
| 53 | 25 | Nelson Agholor | 76.22 | 73.80 | 73.66 | 465 | Raiders |
| 54 | 26 | Braxton Berrios | 75.90 | 72.40 | 74.07 | 189 | Jets |
| 55 | 27 | Marvin Jones Jr. | 75.61 | 73.60 | 72.78 | 667 | Lions |
| 56 | 28 | DJ Chark Jr. | 75.47 | 71.30 | 74.09 | 497 | Jaguars |
| 57 | 29 | KhaDarel Hodge | 75.43 | 70.30 | 74.68 | 140 | Browns |
| 58 | 30 | Richie James | 75.37 | 64.50 | 78.45 | 247 | 49ers |
| 59 | 31 | Curtis Samuel | 75.33 | 77.10 | 69.99 | 435 | Panthers |
| 60 | 32 | Allen Lazard | 75.32 | 71.70 | 73.56 | 277 | Packers |
| 61 | 33 | Scott Miller | 75.11 | 66.90 | 76.41 | 336 | Buccaneers |
| 62 | 34 | Breshad Perriman | 75.08 | 63.60 | 78.56 | 393 | Jets |
| 63 | 35 | David Moore | 75.07 | 67.70 | 75.82 | 314 | Seahawks |
| 64 | 36 | Marquise Brown | 75.04 | 70.10 | 74.16 | 466 | Ravens |
| 65 | 37 | Randall Cobb | 74.96 | 71.60 | 73.03 | 277 | Texans |
| 66 | 38 | Auden Tate | 74.92 | 73.40 | 71.77 | 101 | Bengals |
| 67 | 39 | CeeDee Lamb | 74.75 | 71.60 | 72.68 | 539 | Cowboys |
| 68 | 40 | Quintez Cephus | 74.73 | 64.80 | 77.19 | 231 | Lions |
| 69 | 41 | Olamide Zaccheaus | 74.71 | 68.60 | 74.61 | 184 | Falcons |
| 70 | 42 | Kendrick Bourne | 74.40 | 72.00 | 71.84 | 476 | 49ers |
| 71 | 43 | Russell Gage | 74.30 | 76.40 | 68.74 | 548 | Falcons |
| 72 | 44 | Darius Slayton | 74.06 | 67.80 | 74.06 | 583 | Giants |
| 73 | 45 | Laviska Shenault Jr. | 74.01 | 71.50 | 71.51 | 392 | Jaguars |

### Starter (74 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 74 | 1 | Willie Snead IV | 73.77 | 71.60 | 71.05 | 296 | Ravens |
| 75 | 2 | Isaiah McKenzie | 73.70 | 72.10 | 70.60 | 149 | Bills |
| 76 | 3 | Travis Fulgham | 73.56 | 71.20 | 70.96 | 387 | Eagles |
| 77 | 4 | Jerry Jeudy | 73.47 | 65.20 | 74.82 | 545 | Broncos |
| 78 | 5 | Equanimeous St. Brown | 73.44 | 60.40 | 77.97 | 107 | Packers |
| 79 | 6 | Michael Gallup | 73.43 | 66.30 | 74.02 | 655 | Cowboys |
| 80 | 7 | Adam Humphries | 73.19 | 69.90 | 71.21 | 168 | Titans |
| 81 | 8 | Gabe Davis | 73.16 | 64.90 | 74.50 | 508 | Bills |
| 82 | 9 | Marquez Callaway | 73.15 | 69.10 | 71.69 | 150 | Saints |
| 83 | 10 | Deonte Harty | 73.12 | 71.60 | 69.96 | 125 | Saints |
| 84 | 11 | Julian Edelman | 73.03 | 68.80 | 71.68 | 175 | Patriots |
| 85 | 12 | Jalen Reagor | 72.99 | 64.00 | 74.82 | 321 | Eagles |
| 86 | 13 | Zach Pascal | 72.90 | 64.20 | 74.53 | 501 | Colts |
| 87 | 14 | Keke Coutee | 72.90 | 68.20 | 71.87 | 228 | Texans |
| 88 | 15 | Chris Conley | 72.83 | 70.10 | 70.48 | 315 | Jaguars |
| 89 | 16 | Josh Reynolds | 72.70 | 65.70 | 73.20 | 511 | Rams |
| 90 | 17 | JuJu Smith-Schuster | 72.60 | 68.20 | 71.36 | 687 | Steelers |
| 91 | 18 | Jakeem Grant Sr. | 72.54 | 70.90 | 69.46 | 247 | Dolphins |
| 92 | 19 | Golden Tate | 72.53 | 66.00 | 72.71 | 318 | Giants |
| 93 | 20 | A.J. Green | 72.52 | 66.30 | 72.50 | 538 | Bengals |
| 94 | 21 | Byron Pringle | 72.43 | 64.60 | 73.49 | 163 | Chiefs |
| 95 | 22 | Diontae Johnson | 72.28 | 69.40 | 70.04 | 557 | Steelers |
| 96 | 23 | Darnell Mooney | 72.14 | 68.70 | 70.27 | 537 | Bears |
| 97 | 24 | Cam Sims | 71.88 | 61.30 | 74.76 | 410 | Commanders |
| 98 | 25 | Preston Williams | 71.79 | 65.50 | 71.81 | 221 | Dolphins |
| 99 | 26 | Sammy Watkins | 71.78 | 64.40 | 72.53 | 361 | Chiefs |
| 100 | 27 | Marquez Valdes-Scantling | 71.68 | 57.70 | 76.84 | 497 | Packers |
| 101 | 28 | Michael Pittman Jr. | 71.61 | 62.60 | 73.45 | 385 | Colts |
| 102 | 29 | Miles Boykin | 71.61 | 62.50 | 73.51 | 272 | Ravens |
| 103 | 30 | Christian Kirk | 70.84 | 62.30 | 72.36 | 540 | Cardinals |
| 104 | 31 | James Washington | 70.83 | 60.90 | 73.29 | 322 | Steelers |
| 105 | 32 | Van Jefferson | 70.22 | 67.10 | 68.14 | 159 | Rams |
| 106 | 33 | Noah Brown | 70.19 | 66.70 | 68.35 | 143 | Cowboys |
| 107 | 34 | Olabisi Johnson | 70.18 | 67.00 | 68.13 | 148 | Vikings |
| 108 | 35 | Mohamed Sanu | 70.02 | 65.50 | 68.86 | 197 | Lions |
| 109 | 36 | Marvin Hall | 69.70 | 57.40 | 73.74 | 258 | Browns |
| 110 | 37 | Henry Ruggs III | 69.68 | 54.00 | 75.96 | 366 | Raiders |
| 111 | 38 | Tyler Johnson | 69.46 | 58.90 | 72.33 | 171 | Buccaneers |
| 112 | 39 | Keelan Cole Sr. | 69.42 | 64.40 | 68.60 | 601 | Jaguars |
| 113 | 40 | Damiere Byrd | 69.18 | 61.40 | 70.20 | 496 | Patriots |
| 114 | 41 | Demarcus Robinson | 69.05 | 62.90 | 68.98 | 523 | Chiefs |
| 115 | 42 | Tre'Quan Smith | 68.62 | 59.90 | 70.26 | 425 | Saints |
| 116 | 43 | Cedrick Wilson Jr. | 68.57 | 61.10 | 69.39 | 159 | Cowboys |
| 117 | 44 | Freddie Swain | 68.29 | 53.40 | 74.05 | 241 | Seahawks |
| 118 | 45 | Larry Fitzgerald | 68.14 | 59.60 | 69.66 | 473 | Cardinals |
| 119 | 46 | Kalif Raymond | 68.08 | 53.50 | 73.64 | 129 | Titans |
| 120 | 47 | Devin Duvernay | 67.87 | 60.50 | 68.61 | 202 | Ravens |
| 121 | 48 | Ray-Ray McCloud III | 67.74 | 70.20 | 61.94 | 105 | Steelers |
| 122 | 49 | Zay Jones | 67.72 | 62.90 | 66.77 | 170 | Raiders |
| 123 | 50 | Anthony Miller | 67.31 | 58.50 | 69.01 | 447 | Bears |
| 124 | 51 | Andy Isabella | 67.07 | 56.80 | 69.75 | 221 | Cardinals |
| 125 | 52 | C.J. Board | 66.93 | 67.40 | 62.45 | 116 | Giants |
| 126 | 53 | Jalen Guyton | 66.88 | 52.20 | 72.50 | 617 | Chargers |
| 127 | 54 | Chad Hansen | 66.82 | 60.70 | 66.74 | 209 | Texans |
| 128 | 55 | Isaiah Ford | 66.67 | 61.10 | 66.21 | 274 | Dolphins |
| 129 | 56 | Isaiah Wright | 66.59 | 59.00 | 67.49 | 247 | Commanders |
| 130 | 57 | Alex Erickson | 66.45 | 53.40 | 70.99 | 116 | Bengals |
| 131 | 58 | Chad Beebe | 66.37 | 57.20 | 68.32 | 233 | Vikings |
| 132 | 59 | Cameron Batson | 66.31 | 58.10 | 67.62 | 157 | Titans |
| 133 | 60 | Alshon Jeffery | 65.87 | 51.30 | 71.41 | 153 | Eagles |
| 134 | 61 | KeeSean Johnson | 65.74 | 61.20 | 64.60 | 145 | Cardinals |
| 135 | 62 | Dontrelle Inman | 65.72 | 61.50 | 64.36 | 240 | Commanders |
| 136 | 63 | Brandon Powell | 65.39 | 56.70 | 67.01 | 136 | Falcons |
| 137 | 64 | DaeSean Hamilton | 65.30 | 58.40 | 65.74 | 325 | Broncos |
| 138 | 65 | KJ Hamler | 64.72 | 55.90 | 66.44 | 351 | Broncos |
| 139 | 66 | N'Keal Harry | 64.60 | 57.80 | 64.96 | 332 | Patriots |
| 140 | 67 | Mike Thomas | 64.50 | 55.20 | 66.53 | 142 | Bengals |
| 141 | 68 | Mack Hollins | 64.43 | 58.20 | 64.41 | 162 | Dolphins |
| 142 | 69 | John Hightower | 64.32 | 50.50 | 69.37 | 250 | Eagles |
| 143 | 70 | Greg Ward | 64.27 | 58.80 | 63.75 | 543 | Eagles |
| 144 | 71 | Jamal Agnew | 64.07 | 56.70 | 64.82 | 128 | Lions |
| 145 | 72 | Steven Sims | 63.75 | 56.70 | 64.29 | 286 | Commanders |
| 146 | 73 | Trent Taylor | 63.34 | 53.90 | 65.46 | 154 | 49ers |
| 147 | 74 | K.J. Hill | 62.31 | 55.00 | 63.01 | 100 | Chargers |

### Rotation/backup (3 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 148 | 1 | Christian Blake | 61.84 | 51.30 | 64.70 | 156 | Falcons |
| 149 | 2 | Jeff Smith | 60.90 | 52.10 | 62.60 | 221 | Jets |
| 150 | 3 | Javon Wims | 60.53 | 53.90 | 60.79 | 153 | Bears |
