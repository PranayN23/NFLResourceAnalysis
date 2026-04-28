# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:37:51Z
- **Requested analysis_year:** 2021 (clamped to 2021)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Grade (composite):** same pipeline as FA `/evaluate` with **no team selected** — each position's `predict_performance` (PFF-style model grade + stats grade, weights vary by position; QB adds sample reliability and volume logic), then health and inactivity adjustments. **ED/DI** use the transformer ML `predicted_grade` as the model component.
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section header).
- **Eligibility:** players with **total snaps < 100** in that season row are omitted; composite still uses full history through that season (via `history_as_of_year`).

## C — Center

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Creed Humphrey | 97.31 | 91.40 | 97.08 | 1184 | Chiefs |
| 2 | 2 | Corey Linsley | 92.83 | 85.70 | 93.41 | 1076 | Chargers |
| 3 | 3 | Jason Kelce | 91.93 | 84.40 | 92.79 | 993 | Eagles |
| 4 | 4 | Chase Roullier | 89.08 | 79.51 | 91.30 | 490 | Commanders |
| 5 | 5 | Brian Allen | 88.54 | 79.34 | 90.50 | 903 | Rams |
| 6 | 6 | Frank Ragnow | 87.21 | 78.30 | 88.98 | 223 | Lions |
| 7 | 7 | J.C. Tretter | 86.54 | 78.70 | 87.60 | 1039 | Browns |
| 8 | 8 | Matt Hennessy | 86.33 | 76.40 | 88.78 | 988 | Falcons |
| 9 | 9 | David Andrews | 86.03 | 78.00 | 87.22 | 1087 | Patriots |
| 10 | 10 | Ben Jones | 85.95 | 77.80 | 87.22 | 1160 | Titans |
| 11 | 11 | Connor McGovern | 84.77 | 75.84 | 86.56 | 973 | Jets |
| 12 | 12 | Bradley Bozeman | 82.05 | 73.60 | 83.51 | 1125 | Ravens |
| 13 | 13 | Ryan Jensen | 81.36 | 69.90 | 84.84 | 1151 | Buccaneers |
| 14 | 14 | Alex Mack | 80.09 | 70.40 | 82.38 | 1088 | 49ers |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Ethan Pocic | 77.88 | 65.70 | 81.84 | 600 | Seahawks |
| 16 | 2 | Tyler Biadasz | 77.77 | 64.80 | 82.25 | 1202 | Cowboys |
| 17 | 3 | Mason Cole | 77.48 | 66.71 | 80.50 | 471 | Vikings |
| 18 | 4 | Evan Brown | 76.91 | 65.95 | 80.05 | 755 | Lions |
| 19 | 5 | Erik McCoy | 76.81 | 63.32 | 81.64 | 746 | Saints |
| 20 | 6 | Matt Paradis | 76.55 | 65.90 | 79.48 | 568 | Panthers |
| 21 | 7 | Justin Britt | 76.31 | 63.49 | 80.69 | 671 | Texans |
| 22 | 8 | Tyler Larsen | 75.35 | 62.47 | 79.77 | 185 | Commanders |
| 23 | 9 | Andre James | 75.21 | 64.10 | 78.45 | 1139 | Raiders |
| 24 | 10 | Lloyd Cushenberry III | 75.13 | 64.20 | 78.25 | 1039 | Broncos |
| 25 | 11 | Brandon Linder | 74.82 | 62.46 | 78.89 | 552 | Jaguars |
| 26 | 12 | Billy Price | 74.76 | 62.30 | 78.90 | 985 | Giants |
| 27 | 13 | Mitch Morse | 74.34 | 63.40 | 77.47 | 1167 | Bills |
| 28 | 14 | Keith Ismael | 74.29 | 63.05 | 77.62 | 382 | Commanders |

### Starter (21 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 29 | 1 | Danny Pinter | 73.55 | 67.04 | 73.73 | 226 | Colts |
| 30 | 2 | J.C. Hassenauer | 73.40 | 64.08 | 75.44 | 277 | Steelers |
| 31 | 3 | Rodney Hudson | 73.22 | 60.85 | 77.30 | 798 | Cardinals |
| 32 | 4 | Garrett Bradbury | 73.08 | 60.19 | 77.51 | 883 | Vikings |
| 33 | 5 | Michael Deiter | 72.72 | 60.45 | 76.73 | 546 | Dolphins |
| 34 | 6 | Greg Mancz | 72.58 | 61.25 | 75.97 | 185 | Dolphins |
| 35 | 7 | Scott Quessenberry | 72.52 | 61.72 | 75.56 | 115 | Chargers |
| 36 | 8 | Trystan Colon | 70.76 | 61.04 | 73.08 | 147 | Ravens |
| 37 | 9 | Tyler Shatley | 69.74 | 60.44 | 71.78 | 531 | Jaguars |
| 38 | 10 | Ryan Kelly | 69.56 | 56.98 | 73.78 | 907 | Colts |
| 39 | 11 | Will Clapp | 68.76 | 56.28 | 72.92 | 133 | Saints |
| 40 | 12 | Coleman Shelton | 67.40 | 58.92 | 68.88 | 216 | Rams |
| 41 | 13 | Sam Tecklenburg | 66.94 | 61.00 | 66.74 | 131 | Panthers |
| 42 | 14 | Kendrick Green | 66.93 | 52.44 | 72.42 | 975 | Steelers |
| 43 | 15 | Josh Myers | 66.93 | 59.07 | 68.00 | 293 | Packers |
| 44 | 16 | Sam Mustipher | 65.31 | 51.00 | 70.69 | 1121 | Bears |
| 45 | 17 | Trey Hopkins | 65.28 | 51.65 | 70.20 | 928 | Bengals |
| 46 | 18 | Trey Hill | 64.99 | 56.91 | 66.21 | 210 | Bengals |
| 47 | 19 | Kyle Fuller | 64.73 | 50.91 | 69.78 | 447 | Seahawks |
| 48 | 20 | Jimmy Morrissey | 63.66 | 52.32 | 67.05 | 258 | Texans |
| 49 | 21 | Ryan McCollum | 63.41 | 55.80 | 64.31 | 101 | Lions |

### Rotation/backup (0 players)

_None._

## CB — Cornerback

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jalen Ramsey | 88.40 | 86.30 | 87.26 | 1037 | Rams |
| 2 | 2 | J.C. Jackson | 86.99 | 83.00 | 85.48 | 944 | Patriots |
| 3 | 3 | A.J. Terrell | 86.88 | 85.60 | 84.96 | 1023 | Falcons |
| 4 | 4 | Darius Slay | 85.98 | 83.90 | 84.42 | 953 | Eagles |
| 5 | 5 | Chidobe Awuzie | 83.92 | 83.49 | 84.00 | 777 | Bengals |
| 6 | 6 | Marshon Lattimore | 83.15 | 76.40 | 85.02 | 999 | Saints |
| 7 | 7 | Rashad Fenton | 83.01 | 78.60 | 86.04 | 531 | Chiefs |
| 8 | 8 | Kendall Fuller | 82.39 | 78.70 | 82.84 | 1004 | Commanders |
| 9 | 9 | Casey Hayward Jr. | 81.56 | 75.00 | 82.40 | 1091 | Raiders |
| 10 | 10 | Jamel Dean | 81.02 | 75.81 | 83.18 | 685 | Buccaneers |
| 11 | 11 | Denzel Ward | 80.62 | 75.90 | 82.66 | 855 | Browns |
| 12 | 12 | Artie Burns | 80.46 | 78.40 | 85.35 | 254 | Bears |
| 13 | 13 | Stephon Gilmore | 80.31 | 74.83 | 85.77 | 304 | Panthers |

### Good (24 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Rasul Douglas | 79.85 | 76.34 | 81.10 | 680 | Packers |
| 15 | 2 | Mike Hughes | 79.59 | 74.77 | 82.80 | 509 | Chiefs |
| 16 | 3 | Adoree' Jackson | 79.37 | 80.82 | 81.31 | 815 | Giants |
| 17 | 4 | Avonte Maddox | 79.20 | 75.23 | 80.88 | 729 | Eagles |
| 18 | 5 | D.J. Reed | 78.23 | 75.40 | 80.85 | 1002 | Seahawks |
| 19 | 6 | Taron Johnson | 78.04 | 75.20 | 77.09 | 877 | Bills |
| 20 | 7 | A.J. Green III | 78.04 | 73.99 | 92.84 | 176 | Browns |
| 21 | 8 | Xavien Howard | 78.01 | 71.10 | 81.24 | 1026 | Dolphins |
| 22 | 9 | Mike Hilton | 77.90 | 73.49 | 77.93 | 803 | Bengals |
| 23 | 10 | Jaire Alexander | 77.62 | 72.20 | 83.75 | 219 | Packers |
| 24 | 11 | Nate Hobbs | 77.01 | 76.70 | 75.02 | 837 | Raiders |
| 25 | 12 | Thomas Graham Jr. | 76.34 | 72.24 | 93.40 | 112 | Bears |
| 26 | 13 | Shaquill Griffin | 76.24 | 71.10 | 78.63 | 872 | Jaguars |
| 27 | 14 | Tavierre Thomas | 75.91 | 74.17 | 79.30 | 639 | Texans |
| 28 | 15 | Trevon Diggs | 75.67 | 66.70 | 79.66 | 1013 | Cowboys |
| 29 | 16 | Levi Wallace | 75.61 | 70.00 | 76.44 | 993 | Bills |
| 30 | 17 | Ahkello Witherspoon | 75.57 | 75.05 | 79.12 | 368 | Steelers |
| 31 | 18 | Robert Alford | 75.22 | 67.68 | 78.05 | 580 | Cardinals |
| 32 | 19 | Rock Ya-Sin | 74.52 | 70.50 | 76.14 | 592 | Colts |
| 33 | 20 | Nik Needham | 74.50 | 66.52 | 77.11 | 608 | Dolphins |
| 34 | 21 | Tre'Davious White | 74.44 | 68.32 | 78.12 | 630 | Bills |
| 35 | 22 | Eric Stokes | 74.41 | 67.60 | 75.76 | 934 | Packers |
| 36 | 23 | Pat Surtain II | 74.27 | 66.30 | 76.40 | 900 | Broncos |
| 37 | 24 | Greg Newsome II | 74.08 | 69.70 | 77.73 | 691 | Browns |

### Starter (85 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 38 | 1 | Anthony Brown | 73.99 | 69.00 | 76.96 | 1046 | Cowboys |
| 39 | 2 | Carlton Davis III | 73.95 | 69.00 | 77.55 | 639 | Buccaneers |
| 40 | 3 | Sidney Jones IV | 73.92 | 68.65 | 79.79 | 730 | Seahawks |
| 41 | 4 | Emmanuel Moseley | 73.84 | 68.29 | 77.56 | 602 | 49ers |
| 42 | 5 | Ross Cockrell | 73.56 | 68.21 | 76.89 | 475 | Buccaneers |
| 43 | 6 | Bryce Hall | 73.26 | 64.50 | 78.06 | 1171 | Jets |
| 44 | 7 | James Bradberry | 73.15 | 65.00 | 74.94 | 1160 | Giants |
| 45 | 8 | Pierre Desir | 73.11 | 68.51 | 78.15 | 308 | Buccaneers |
| 46 | 9 | Darious Williams | 73.04 | 64.40 | 78.18 | 924 | Rams |
| 47 | 10 | Isaiah Oliver | 73.01 | 67.20 | 79.09 | 161 | Falcons |
| 48 | 11 | Greedy Williams | 72.90 | 67.45 | 74.18 | 591 | Browns |
| 49 | 12 | Isaiah Rodgers | 72.88 | 69.41 | 75.33 | 525 | Colts |
| 50 | 13 | Byron Jones | 72.84 | 63.50 | 76.22 | 976 | Dolphins |
| 51 | 14 | A.J. Bouye | 72.76 | 66.96 | 79.12 | 401 | Panthers |
| 52 | 15 | Kristian Fulton | 72.72 | 67.38 | 78.46 | 738 | Titans |
| 53 | 16 | Kevin King | 72.61 | 64.91 | 78.78 | 303 | Packers |
| 54 | 17 | Janoris Jenkins | 72.50 | 64.60 | 76.22 | 862 | Titans |
| 55 | 18 | Joe Haden | 72.12 | 64.62 | 76.03 | 631 | Steelers |
| 56 | 19 | Marlon Humphrey | 71.88 | 64.85 | 75.17 | 746 | Ravens |
| 57 | 20 | Cameron Dantzler | 71.67 | 66.93 | 75.07 | 685 | Vikings |
| 58 | 21 | Bryce Callahan | 71.44 | 66.86 | 75.14 | 504 | Broncos |
| 59 | 22 | Jerry Jacobs | 71.41 | 65.40 | 77.14 | 535 | Lions |
| 60 | 23 | Bradley Roby | 71.38 | 65.87 | 75.48 | 395 | Saints |
| 61 | 24 | Byron Murphy Jr. | 71.29 | 64.30 | 72.59 | 967 | Cardinals |
| 62 | 25 | Joejuan Williams | 71.24 | 60.89 | 80.07 | 254 | Patriots |
| 63 | 26 | Charvarius Ward | 71.12 | 61.91 | 75.67 | 751 | Chiefs |
| 64 | 27 | Nate Hairston | 71.00 | 66.78 | 79.12 | 148 | Broncos |
| 65 | 28 | Jalen Mills | 70.56 | 63.40 | 75.52 | 914 | Patriots |
| 66 | 29 | Jaylon Johnson | 70.51 | 60.50 | 75.41 | 933 | Bears |
| 67 | 30 | Steven Nelson | 70.42 | 60.40 | 73.95 | 981 | Eagles |
| 68 | 31 | Kenny Moore II | 70.38 | 62.10 | 72.77 | 1062 | Colts |
| 69 | 32 | Jourdan Lewis | 70.18 | 62.07 | 72.44 | 801 | Cowboys |
| 70 | 33 | Tyson Campbell | 70.10 | 59.90 | 74.70 | 864 | Jaguars |
| 71 | 34 | Trayvon Mullen | 69.69 | 62.27 | 76.36 | 229 | Raiders |
| 72 | 35 | Xavier Rhodes | 69.66 | 61.67 | 72.99 | 638 | Colts |
| 73 | 36 | Jimmy Smith | 69.43 | 65.38 | 74.42 | 293 | Ravens |
| 74 | 37 | Elijah Molden | 69.19 | 64.11 | 70.38 | 632 | Titans |
| 75 | 38 | Paulson Adebo | 69.14 | 61.40 | 70.13 | 851 | Saints |
| 76 | 39 | Desmond Trufant | 69.06 | 63.38 | 76.69 | 234 | Raiders |
| 77 | 40 | Kelvin Joseph | 68.80 | 65.81 | 80.41 | 165 | Cowboys |
| 78 | 41 | Eli Apple | 68.73 | 61.60 | 70.33 | 979 | Bengals |
| 79 | 42 | L'Jarius Sneed | 68.42 | 62.40 | 72.24 | 918 | Chiefs |
| 80 | 43 | Antonio Hamilton Sr. | 68.23 | 62.89 | 75.14 | 313 | Cardinals |
| 81 | 44 | Robert Rochell | 68.22 | 60.37 | 76.16 | 233 | Rams |
| 82 | 45 | Chris Harris Jr. | 67.97 | 60.39 | 72.50 | 747 | Chargers |
| 83 | 46 | Amani Oruwariye | 67.92 | 60.30 | 72.18 | 937 | Lions |
| 84 | 47 | William Jackson III | 67.88 | 59.51 | 72.80 | 748 | Commanders |
| 85 | 48 | Patrick Peterson | 67.86 | 61.00 | 71.48 | 884 | Vikings |
| 86 | 49 | Cameron Sutton | 67.86 | 58.90 | 70.15 | 1089 | Steelers |
| 87 | 50 | Tavon Young | 67.80 | 62.31 | 71.66 | 550 | Ravens |
| 88 | 51 | Ronald Darby | 67.74 | 58.77 | 73.53 | 675 | Broncos |
| 89 | 52 | Dont'e Deayon | 66.89 | 65.76 | 70.34 | 461 | Rams |
| 90 | 53 | Sean Murphy-Bunting | 66.76 | 60.75 | 70.73 | 462 | Buccaneers |
| 91 | 54 | K'Waun Williams | 66.54 | 60.75 | 70.42 | 647 | 49ers |
| 92 | 55 | Buster Skrine | 66.44 | 62.62 | 71.47 | 218 | Titans |
| 93 | 56 | Asante Samuel Jr. | 66.21 | 57.62 | 72.67 | 693 | Chargers |
| 94 | 57 | Myles Bryant | 66.16 | 58.25 | 70.33 | 405 | Patriots |
| 95 | 58 | Chandon Sullivan | 65.96 | 57.80 | 68.06 | 826 | Packers |
| 96 | 59 | Jarren Williams | 65.77 | 63.83 | 79.16 | 194 | Giants |
| 97 | 60 | A.J. Parker | 65.70 | 57.62 | 70.84 | 556 | Lions |
| 98 | 61 | Dane Jackson | 65.63 | 55.95 | 71.83 | 482 | Bills |
| 99 | 62 | Darnay Holmes | 65.56 | 60.23 | 70.18 | 282 | Giants |
| 100 | 63 | Michael Davis | 65.52 | 54.40 | 71.07 | 851 | Chargers |
| 101 | 64 | Dontae Johnson | 65.25 | 61.69 | 70.87 | 262 | 49ers |
| 102 | 65 | Donte Jackson | 65.24 | 56.36 | 70.91 | 717 | Panthers |
| 103 | 66 | Darryl Roberts | 65.05 | 59.16 | 74.38 | 203 | Commanders |
| 104 | 67 | T.J. Carrie | 65.00 | 56.89 | 70.99 | 142 | Colts |
| 105 | 68 | Danny Johnson | 64.99 | 63.32 | 72.48 | 336 | Commanders |
| 106 | 69 | Rashaan Melvin | 64.68 | 57.82 | 69.87 | 247 | Panthers |
| 107 | 70 | James Pierre | 64.60 | 58.87 | 71.62 | 415 | Steelers |
| 108 | 71 | Fabian Moreau | 64.59 | 55.20 | 69.25 | 1036 | Falcons |
| 109 | 72 | Justin Coleman | 64.48 | 56.23 | 68.37 | 371 | Dolphins |
| 110 | 73 | David Long Jr. | 64.31 | 58.73 | 68.57 | 517 | Rams |
| 111 | 74 | Troy Hill | 64.02 | 54.27 | 69.43 | 533 | Browns |
| 112 | 75 | Javelin Guidry | 63.95 | 57.54 | 67.58 | 487 | Jets |
| 113 | 76 | Aaron Robinson | 63.78 | 59.09 | 70.59 | 268 | Giants |
| 114 | 77 | Michael Carter II | 63.70 | 56.12 | 66.56 | 777 | Jets |
| 115 | 78 | Trae Waynes | 63.61 | 58.97 | 68.84 | 243 | Bengals |
| 116 | 79 | Keith Taylor Jr. | 63.53 | 55.43 | 65.74 | 448 | Panthers |
| 117 | 80 | Anthony Averett | 63.25 | 55.05 | 69.35 | 807 | Ravens |
| 118 | 81 | Jaycee Horn | 63.02 | 63.07 | 78.82 | 142 | Panthers |
| 119 | 82 | Isaiah Dunn | 62.83 | 60.00 | 74.33 | 115 | Jets |
| 120 | 83 | Terrance Mitchell | 62.80 | 51.85 | 69.07 | 796 | Texans |
| 121 | 84 | Vernon Hargreaves III | 62.08 | 56.01 | 65.60 | 390 | Bengals |
| 122 | 85 | Jonathan Jones | 62.07 | 53.94 | 68.92 | 224 | Patriots |

### Rotation/backup (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 123 | 1 | John Reid | 61.96 | 61.20 | 69.16 | 132 | Seahawks |
| 124 | 2 | Duke Shelley | 61.44 | 58.73 | 67.50 | 409 | Bears |
| 125 | 3 | Richard Sherman | 61.24 | 54.69 | 70.98 | 141 | Buccaneers |
| 126 | 4 | Zech McPhearson | 61.02 | 60.00 | 68.83 | 179 | Eagles |
| 127 | 5 | Chris Jackson | 60.88 | 54.66 | 67.27 | 386 | Titans |
| 128 | 6 | Tevaughn Campbell | 60.75 | 53.11 | 64.63 | 678 | Chargers |
| 129 | 7 | Tre Flowers | 60.28 | 53.05 | 64.67 | 391 | Bengals |
| 130 | 8 | Nevin Lawson | 60.03 | 53.03 | 65.83 | 325 | Jaguars |
| 131 | 9 | Josh Norman | 59.93 | 47.68 | 68.94 | 765 | 49ers |
| 132 | 10 | Brandin Echols | 59.88 | 48.47 | 66.26 | 762 | Jets |
| 133 | 11 | Desmond King II | 59.58 | 47.70 | 64.35 | 929 | Texans |
| 134 | 12 | Kyle Fuller | 59.43 | 41.57 | 67.66 | 719 | Broncos |
| 135 | 13 | Ifeatu Melifonwu | 59.20 | 60.38 | 65.54 | 242 | Lions |
| 136 | 14 | Chris Westry | 58.37 | 55.29 | 68.11 | 183 | Ravens |
| 137 | 15 | Arthur Maulet | 58.32 | 49.07 | 62.56 | 380 | Steelers |
| 138 | 16 | Dee Delaney | 58.13 | 56.65 | 66.32 | 213 | Buccaneers |
| 139 | 17 | Greg Mabin | 58.08 | 55.63 | 67.37 | 171 | Titans |
| 140 | 18 | Tre Brown | 57.97 | 60.50 | 68.39 | 255 | Seahawks |
| 141 | 19 | Benjamin St-Juste | 57.61 | 53.29 | 67.63 | 318 | Commanders |
| 142 | 20 | Mackensie Alexander | 57.56 | 42.08 | 65.77 | 689 | Vikings |
| 143 | 21 | Blessuan Austin | 57.50 | 56.56 | 63.78 | 149 | Seahawks |
| 144 | 22 | DeAndre Baker | 57.40 | 58.43 | 58.68 | 212 | Chiefs |
| 145 | 23 | Brandon Facyson | 57.29 | 47.78 | 66.68 | 602 | Raiders |
| 146 | 24 | Tremon Smith | 57.25 | 57.39 | 64.66 | 179 | Texans |
| 147 | 25 | Deommodore Lenoir | 56.35 | 59.36 | 61.48 | 238 | 49ers |
| 148 | 26 | Darren Hall | 55.92 | 52.44 | 59.96 | 283 | Falcons |
| 149 | 27 | Ugo Amadi | 55.84 | 42.78 | 63.29 | 692 | Seahawks |
| 150 | 28 | Daryl Worley | 55.60 | 48.44 | 65.15 | 100 | Ravens |
| 151 | 29 | Ambry Thomas | 54.80 | 51.79 | 60.49 | 334 | 49ers |
| 152 | 30 | Kevon Seymour | 54.56 | 47.91 | 64.60 | 247 | Ravens |
| 153 | 31 | CJ Henderson | 53.69 | 50.92 | 57.56 | 390 | Panthers |
| 154 | 32 | Chris Claybrooks | 53.43 | 50.42 | 57.74 | 199 | Jaguars |
| 155 | 33 | Tre Herndon | 52.23 | 47.83 | 56.59 | 207 | Jaguars |
| 156 | 34 | Kindle Vildor | 51.45 | 45.63 | 56.08 | 822 | Bears |
| 157 | 35 | Kris Boyd | 49.12 | 50.05 | 53.40 | 160 | Vikings |
| 158 | 36 | Amik Robertson | 46.46 | 51.81 | 49.37 | 137 | Raiders |
| 159 | 37 | Xavier Crawford | 45.00 | 53.99 | 49.86 | 140 | Bears |

## DI — Defensive Interior

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (15 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 92.53 | 87.59 | 91.66 | 1040 | Rams |
| 2 | 2 | Quinnen Williams | 86.89 | 85.84 | 85.98 | 613 | Jets |
| 3 | 3 | Zach Sieler | 85.97 | 79.42 | 88.05 | 518 | Dolphins |
| 4 | 4 | Jonathan Allen | 85.04 | 86.42 | 80.16 | 772 | Commanders |
| 5 | 5 | DeForest Buckner | 84.99 | 87.19 | 79.67 | 843 | Colts |
| 6 | 6 | Leonard Williams | 84.75 | 87.46 | 78.97 | 890 | Giants |
| 7 | 7 | Cameron Heyward | 83.71 | 81.63 | 81.24 | 955 | Steelers |
| 8 | 8 | Christian Wilkins | 82.90 | 84.60 | 78.23 | 734 | Dolphins |
| 9 | 9 | Grady Jarrett | 82.83 | 80.02 | 80.53 | 864 | Falcons |
| 10 | 10 | Chris Jones | 81.87 | 86.29 | 77.16 | 628 | Chiefs |
| 11 | 11 | Jeffery Simmons | 81.62 | 82.90 | 78.36 | 933 | Titans |
| 12 | 12 | Calais Campbell | 81.43 | 73.26 | 84.94 | 615 | Ravens |
| 13 | 13 | Kenny Clark | 81.37 | 79.56 | 79.85 | 781 | Packers |
| 14 | 14 | Dexter Lawrence | 80.37 | 80.71 | 76.46 | 759 | Giants |
| 15 | 15 | Ed Oliver | 80.21 | 69.77 | 83.00 | 622 | Bills |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Dalvin Tomlinson | 79.54 | 80.70 | 75.08 | 641 | Vikings |
| 17 | 2 | Poona Ford | 79.27 | 76.40 | 77.21 | 802 | Seahawks |
| 18 | 3 | Vita Vea | 79.10 | 83.95 | 75.63 | 608 | Buccaneers |
| 19 | 4 | B.J. Hill | 78.72 | 74.57 | 77.81 | 502 | Bengals |
| 20 | 5 | Fletcher Cox | 78.02 | 73.49 | 77.67 | 747 | Eagles |
| 21 | 6 | Daron Payne | 76.88 | 72.62 | 75.76 | 837 | Commanders |
| 22 | 7 | J.J. Watt | 76.86 | 59.83 | 96.59 | 341 | Cardinals |
| 23 | 8 | Michael Pierce | 76.78 | 74.30 | 79.71 | 251 | Vikings |
| 24 | 9 | Akiem Hicks | 76.62 | 72.56 | 81.70 | 304 | Bears |
| 25 | 10 | Tim Settle | 76.11 | 65.45 | 79.75 | 210 | Commanders |
| 26 | 11 | DJ Reader | 75.97 | 79.32 | 74.20 | 590 | Bengals |
| 27 | 12 | Shelby Harris | 75.80 | 65.01 | 80.87 | 564 | Broncos |
| 28 | 13 | David Onyemata | 75.76 | 75.71 | 75.10 | 430 | Saints |
| 29 | 14 | Derrick Brown | 75.42 | 71.87 | 74.24 | 631 | Panthers |
| 30 | 15 | Folorunso Fatukasi | 75.32 | 66.61 | 78.68 | 558 | Jets |
| 31 | 16 | D.J. Jones | 74.97 | 65.18 | 79.00 | 550 | 49ers |
| 32 | 17 | Javon Hargrave | 74.33 | 60.67 | 80.07 | 727 | Eagles |
| 33 | 18 | Sheldon Richardson | 74.30 | 61.01 | 79.00 | 688 | Vikings |

### Starter (81 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 34 | 1 | Grover Stewart | 73.92 | 71.07 | 71.65 | 643 | Colts |
| 35 | 2 | Greg Gaines | 73.80 | 67.16 | 75.31 | 780 | Rams |
| 36 | 3 | Matt Ioannidis | 73.77 | 69.08 | 77.28 | 608 | Commanders |
| 37 | 4 | Christian Barmore | 73.75 | 59.99 | 78.75 | 598 | Patriots |
| 38 | 5 | Sebastian Joseph-Day | 72.81 | 63.00 | 80.08 | 340 | Rams |
| 39 | 6 | Bilal Nichols | 72.58 | 61.21 | 76.63 | 679 | Bears |
| 40 | 7 | Dre'Mont Jones | 72.53 | 63.32 | 76.36 | 614 | Broncos |
| 41 | 8 | Shy Tuttle | 71.57 | 63.49 | 73.73 | 494 | Saints |
| 42 | 9 | Arik Armstead | 71.50 | 60.89 | 76.91 | 820 | 49ers |
| 43 | 10 | Linval Joseph | 71.47 | 60.27 | 76.87 | 550 | Chargers |
| 44 | 11 | Lawrence Guy Sr. | 71.45 | 55.45 | 78.58 | 532 | Patriots |
| 45 | 12 | Mario Edwards Jr. | 71.22 | 60.30 | 77.30 | 212 | Bears |
| 46 | 13 | Larry Ogunjobi | 71.04 | 53.25 | 79.75 | 724 | Bengals |
| 47 | 14 | William Gholston | 70.38 | 55.43 | 76.18 | 507 | Buccaneers |
| 48 | 15 | A'Shawn Robinson | 69.90 | 61.02 | 74.78 | 517 | Rams |
| 49 | 16 | Dean Lowry | 69.40 | 59.90 | 71.57 | 673 | Packers |
| 50 | 17 | Darius Philon | 69.29 | 58.44 | 75.29 | 277 | Raiders |
| 51 | 18 | Alim McNeill | 69.27 | 57.97 | 72.64 | 422 | Lions |
| 52 | 19 | Steve McLendon | 69.16 | 53.40 | 78.27 | 252 | Buccaneers |
| 53 | 20 | Harrison Phillips | 69.09 | 65.10 | 73.01 | 473 | Bills |
| 54 | 21 | Morgan Fox | 69.00 | 54.28 | 74.64 | 561 | Panthers |
| 55 | 22 | Chris Wormley | 68.92 | 67.42 | 67.67 | 729 | Steelers |
| 56 | 23 | Al Woods | 68.58 | 58.41 | 72.72 | 620 | Seahawks |
| 57 | 24 | Naquan Jones | 68.51 | 51.17 | 79.82 | 328 | Titans |
| 58 | 25 | DaQuan Jones | 68.42 | 61.56 | 68.82 | 640 | Panthers |
| 59 | 26 | Milton Williams | 68.33 | 50.89 | 75.79 | 456 | Eagles |
| 60 | 27 | Taven Bryan | 68.22 | 59.05 | 71.15 | 301 | Jaguars |
| 61 | 28 | Austin Johnson | 68.12 | 57.27 | 71.19 | 665 | Giants |
| 62 | 29 | Malcom Brown | 67.65 | 51.74 | 75.02 | 678 | Jaguars |
| 63 | 30 | Ndamukong Suh | 67.50 | 45.52 | 77.98 | 718 | Buccaneers |
| 64 | 31 | Roy Robertson-Harris | 67.42 | 55.66 | 75.28 | 547 | Jaguars |
| 65 | 32 | Kevin Givens | 67.39 | 56.34 | 76.60 | 230 | 49ers |
| 66 | 33 | Tyler Lancaster | 67.01 | 54.89 | 71.73 | 318 | Packers |
| 67 | 34 | Marquise Copeland | 66.76 | 60.90 | 74.35 | 108 | Rams |
| 68 | 35 | Neville Gallimore | 66.65 | 55.45 | 78.08 | 164 | Cowboys |
| 69 | 36 | Davon Godchaux | 66.64 | 57.49 | 72.00 | 640 | Patriots |
| 70 | 37 | Eddie Goldman | 66.50 | 49.93 | 75.16 | 336 | Bears |
| 71 | 38 | Michael Dogbe | 66.42 | 54.17 | 74.15 | 263 | Cardinals |
| 72 | 39 | Christian Ringo | 66.13 | 57.47 | 74.42 | 315 | Saints |
| 73 | 40 | Michael Brockers | 66.07 | 49.45 | 73.78 | 622 | Lions |
| 74 | 41 | Roy Lopez | 66.06 | 51.65 | 72.49 | 502 | Texans |
| 75 | 42 | Danny Shelton | 65.98 | 53.40 | 73.42 | 256 | Giants |
| 76 | 43 | Malik Jackson | 65.93 | 45.48 | 79.33 | 646 | Browns |
| 77 | 44 | Brandon Williams | 65.91 | 50.83 | 75.12 | 447 | Ravens |
| 78 | 45 | Derrick Nnadi | 65.75 | 55.42 | 68.79 | 449 | Chiefs |
| 79 | 46 | Jordan Phillips | 65.67 | 57.62 | 72.98 | 284 | Cardinals |
| 80 | 47 | Quinton Jefferson | 65.60 | 50.29 | 72.05 | 686 | Raiders |
| 81 | 48 | Justin Zimmer | 65.57 | 57.96 | 76.05 | 161 | Bills |
| 82 | 49 | Mike Purcell | 65.57 | 54.97 | 74.18 | 361 | Broncos |
| 83 | 50 | Jarran Reed | 65.41 | 49.03 | 73.42 | 711 | Chiefs |
| 84 | 51 | Christian Covington | 65.41 | 51.79 | 70.80 | 523 | Chargers |
| 85 | 52 | Adam Butler | 65.35 | 51.93 | 70.45 | 592 | Dolphins |
| 86 | 53 | Anthony Rush | 65.25 | 57.00 | 72.38 | 266 | Falcons |
| 87 | 54 | Johnathan Hankins | 65.12 | 48.80 | 73.30 | 568 | Raiders |
| 88 | 55 | Osa Odighizuwa | 65.09 | 47.70 | 73.50 | 614 | Cowboys |
| 89 | 56 | Carlos Watkins | 65.04 | 53.79 | 70.61 | 437 | Cowboys |
| 90 | 57 | John Jenkins | 64.88 | 58.55 | 71.40 | 176 | Dolphins |
| 91 | 58 | Hassan Ridgeway | 64.79 | 53.37 | 72.92 | 373 | Eagles |
| 92 | 59 | Armon Watts | 64.68 | 55.39 | 68.59 | 669 | Vikings |
| 93 | 60 | DeShawn Williams | 64.68 | 55.70 | 70.60 | 386 | Broncos |
| 94 | 61 | Tershawn Wharton | 64.57 | 50.91 | 69.51 | 501 | Chiefs |
| 95 | 62 | James Lynch | 64.56 | 53.09 | 73.22 | 304 | Vikings |
| 96 | 63 | Justin Jones | 64.38 | 56.24 | 70.35 | 486 | Chargers |
| 97 | 64 | Adam Gotsis | 64.28 | 47.39 | 74.05 | 443 | Jaguars |
| 98 | 65 | Kentavius Street | 64.05 | 50.83 | 71.72 | 352 | 49ers |
| 99 | 66 | Sheldon Rankins | 64.03 | 51.07 | 71.49 | 643 | Jets |
| 100 | 67 | DaVon Hamilton | 63.95 | 53.80 | 69.12 | 443 | Jaguars |
| 101 | 68 | Mike Pennel | 63.82 | 52.55 | 72.88 | 249 | Falcons |
| 102 | 69 | Khyiris Tonga | 63.72 | 54.62 | 67.59 | 217 | Bears |
| 103 | 70 | Jonathan Bullard | 63.48 | 56.13 | 72.71 | 224 | Falcons |
| 104 | 71 | Taylor Stallworth | 63.46 | 54.99 | 67.92 | 331 | Colts |
| 105 | 72 | Corey Peters | 63.42 | 51.99 | 70.53 | 362 | Cardinals |
| 106 | 73 | Robert Nkemdiche | 63.28 | 54.08 | 71.88 | 230 | Seahawks |
| 107 | 74 | Josh Tupou | 63.25 | 56.25 | 65.83 | 410 | Bengals |
| 108 | 75 | Brent Urban | 63.21 | 53.52 | 71.52 | 160 | Cowboys |
| 109 | 76 | Kyle Peko | 62.83 | 56.65 | 72.40 | 157 | Titans |
| 110 | 77 | Raekwon Davis | 62.69 | 54.13 | 66.06 | 424 | Dolphins |
| 111 | 78 | Maliek Collins | 62.50 | 54.43 | 65.95 | 628 | Texans |
| 112 | 79 | Vernon Butler | 62.13 | 52.09 | 69.13 | 285 | Bills |
| 113 | 80 | John Penisini | 62.09 | 53.47 | 64.29 | 276 | Lions |
| 114 | 81 | Bravvion Roy | 62.09 | 51.49 | 65.37 | 341 | Panthers |

### Rotation/backup (41 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 115 | 1 | Nathan Shepherd | 61.97 | 53.53 | 65.51 | 495 | Jets |
| 116 | 2 | Henry Mondeaux | 61.91 | 52.78 | 67.40 | 244 | Steelers |
| 117 | 3 | Montravius Adams | 61.90 | 55.70 | 68.21 | 286 | Steelers |
| 118 | 4 | Angelo Blackson | 61.89 | 47.47 | 67.54 | 584 | Bears |
| 119 | 5 | Javon Kinlaw | 61.58 | 56.14 | 69.79 | 149 | 49ers |
| 120 | 6 | Sheldon Day | 61.51 | 52.92 | 68.81 | 233 | Browns |
| 121 | 7 | Solomon Thomas | 61.47 | 49.92 | 70.47 | 554 | Raiders |
| 122 | 8 | Joe Gaziano | 61.20 | 52.65 | 65.66 | 214 | Chargers |
| 123 | 9 | L.J. Collier | 61.15 | 54.20 | 65.90 | 219 | Seahawks |
| 124 | 10 | Ross Blacklock | 61.09 | 51.73 | 65.39 | 457 | Texans |
| 125 | 11 | Leki Fotu | 61.08 | 49.71 | 66.44 | 371 | Cardinals |
| 126 | 12 | Rakeem Nunez-Roches | 60.89 | 50.48 | 64.14 | 415 | Buccaneers |
| 127 | 13 | Malik McDowell | 60.87 | 43.19 | 70.46 | 645 | Browns |
| 128 | 14 | Trysten Hill | 60.65 | 57.24 | 69.46 | 171 | Cowboys |
| 129 | 15 | Levi Onwuzurike | 60.42 | 50.07 | 64.13 | 396 | Lions |
| 130 | 16 | Jaleel Johnson | 60.11 | 49.35 | 65.57 | 322 | Texans |
| 131 | 17 | Star Lotulelei | 59.98 | 42.92 | 70.12 | 317 | Bills |
| 132 | 18 | Carl Davis Jr. | 59.83 | 52.97 | 67.00 | 277 | Patriots |
| 133 | 19 | Tyeler Davison | 59.69 | 49.46 | 64.80 | 358 | Falcons |
| 134 | 20 | Breiden Fehoko | 59.53 | 55.91 | 66.60 | 121 | Chargers |
| 135 | 21 | Margus Hunt | 59.14 | 44.97 | 68.79 | 151 | Bears |
| 136 | 22 | Bryan Mone | 59.10 | 54.64 | 63.76 | 395 | Seahawks |
| 137 | 23 | Isaiahh Loudermilk | 58.83 | 50.68 | 62.06 | 288 | Steelers |
| 138 | 24 | Jerry Tillery | 58.83 | 43.77 | 65.70 | 858 | Chargers |
| 139 | 25 | Shamar Stephen | 57.82 | 47.77 | 60.56 | 393 | Broncos |
| 140 | 26 | Justin Ellis | 57.40 | 45.68 | 64.48 | 381 | Ravens |
| 141 | 27 | Tommy Togiai | 57.32 | 54.77 | 68.64 | 125 | Browns |
| 142 | 28 | Ta'Quon Graham | 57.19 | 51.79 | 60.54 | 309 | Falcons |
| 143 | 29 | Rashard Lawrence | 57.15 | 55.31 | 60.63 | 219 | Cardinals |
| 144 | 30 | Khalen Saunders | 57.06 | 55.29 | 63.88 | 144 | Chiefs |
| 145 | 31 | Broderick Washington | 56.49 | 52.86 | 59.71 | 293 | Ravens |
| 146 | 32 | Raymond Johnson III | 55.88 | 53.17 | 55.49 | 166 | Giants |
| 147 | 33 | Michael Hoecht | 55.11 | 54.28 | 56.39 | 110 | Rams |
| 148 | 34 | Teair Tart | 54.83 | 52.80 | 59.22 | 344 | Titans |
| 149 | 35 | Larrell Murchison | 54.70 | 54.75 | 56.52 | 200 | Titans |
| 150 | 36 | Malcolm Roach | 54.11 | 52.61 | 59.81 | 194 | Saints |
| 151 | 37 | Jordan Elliott | 53.95 | 48.42 | 54.09 | 464 | Browns |
| 152 | 38 | Marlon Davidson | 53.75 | 54.66 | 55.78 | 270 | Falcons |
| 153 | 39 | Albert Huggins | 52.44 | 51.80 | 59.47 | 219 | Saints |
| 154 | 40 | Justin Hamilton | 51.80 | 50.57 | 56.75 | 249 | Broncos |
| 155 | 41 | Quinton Bohanna | 50.67 | 49.79 | 50.03 | 222 | Cowboys |

## ED — Edge

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Joey Bosa | 92.17 | 96.19 | 87.05 | 847 | Chargers |
| 2 | 2 | T.J. Watt | 91.90 | 92.82 | 88.42 | 758 | Steelers |
| 3 | 3 | Nick Bosa | 90.05 | 95.77 | 86.43 | 840 | 49ers |
| 4 | 4 | Myles Garrett | 89.24 | 95.35 | 82.88 | 866 | Browns |
| 5 | 5 | Rashan Gary | 87.70 | 88.52 | 83.79 | 681 | Packers |
| 6 | 6 | Von Miller | 87.33 | 81.24 | 88.40 | 762 | Rams |
| 7 | 7 | Shaquil Barrett | 83.85 | 79.89 | 83.63 | 768 | Buccaneers |
| 8 | 8 | Maxx Crosby | 83.17 | 88.43 | 75.50 | 926 | Raiders |
| 9 | 9 | Khalil Mack | 82.21 | 81.48 | 83.43 | 315 | Bears |
| 10 | 10 | Danielle Hunter | 81.64 | 79.21 | 84.00 | 384 | Vikings |
| 11 | 11 | Marcus Davenport | 81.01 | 86.88 | 78.07 | 437 | Saints |
| 12 | 12 | Cameron Jordan | 80.82 | 84.32 | 74.80 | 831 | Saints |
| 13 | 13 | DeMarcus Lawrence | 80.40 | 84.72 | 78.26 | 271 | Cowboys |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Trey Hendrickson | 78.80 | 71.42 | 80.99 | 717 | Bengals |
| 15 | 2 | Andrew Van Ginkel | 78.50 | 68.52 | 80.99 | 801 | Dolphins |
| 16 | 3 | Justin Houston | 77.29 | 61.18 | 84.84 | 577 | Ravens |
| 17 | 4 | Brian Burns | 76.58 | 67.32 | 78.91 | 838 | Panthers |
| 18 | 5 | Montez Sweat | 75.62 | 77.05 | 73.93 | 483 | Commanders |
| 19 | 6 | Matthew Judon | 74.20 | 56.77 | 82.28 | 878 | Patriots |

### Starter (77 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Jadeveon Clowney | 73.76 | 81.06 | 69.32 | 677 | Browns |
| 21 | 2 | Haason Reddick | 73.63 | 60.88 | 78.58 | 852 | Panthers |
| 22 | 3 | Trevis Gipson | 73.40 | 61.75 | 81.13 | 489 | Bears |
| 23 | 4 | Preston Smith | 73.11 | 68.54 | 72.48 | 688 | Packers |
| 24 | 5 | Greg Rousseau | 72.98 | 65.62 | 73.72 | 531 | Bills |
| 25 | 6 | Markus Golden | 72.53 | 54.75 | 80.70 | 681 | Cardinals |
| 26 | 7 | Josh Sweat | 72.35 | 68.23 | 72.05 | 654 | Eagles |
| 27 | 8 | Carlos Dunlap | 72.28 | 60.58 | 76.65 | 482 | Seahawks |
| 28 | 9 | Melvin Ingram III | 71.74 | 66.16 | 75.71 | 590 | Chiefs |
| 29 | 10 | Jonathan Greenard | 71.71 | 67.65 | 74.89 | 414 | Texans |
| 30 | 11 | Kaden Elliss | 71.64 | 59.52 | 83.47 | 192 | Saints |
| 31 | 12 | Ryan Kerrigan | 71.43 | 52.97 | 80.88 | 329 | Eagles |
| 32 | 13 | Jerry Hughes | 71.42 | 62.95 | 73.21 | 557 | Bills |
| 33 | 14 | Tyus Bowser | 71.26 | 60.82 | 74.05 | 832 | Ravens |
| 34 | 15 | Chandler Jones | 71.14 | 62.49 | 77.16 | 823 | Cardinals |
| 35 | 16 | Kemoko Turay | 71.07 | 60.32 | 81.34 | 224 | Colts |
| 36 | 17 | Trey Flowers | 71.06 | 71.68 | 74.39 | 302 | Lions |
| 37 | 18 | Harold Landry III | 70.83 | 61.00 | 73.21 | 981 | Titans |
| 38 | 19 | Randy Gregory | 70.35 | 65.95 | 73.45 | 436 | Cowboys |
| 39 | 20 | Robert Quinn | 70.33 | 54.94 | 77.64 | 755 | Bears |
| 40 | 21 | Darrell Taylor | 70.26 | 56.86 | 75.64 | 545 | Seahawks |
| 41 | 22 | Julian Okwara | 70.22 | 60.07 | 79.17 | 361 | Lions |
| 42 | 23 | Emmanuel Ogbah | 70.21 | 68.80 | 68.23 | 755 | Dolphins |
| 43 | 24 | Uchenna Nwosu | 70.08 | 63.59 | 71.40 | 781 | Chargers |
| 44 | 25 | Frank Clark | 69.91 | 60.25 | 74.38 | 657 | Chiefs |
| 45 | 26 | Chase Young | 69.85 | 85.13 | 60.80 | 477 | Commanders |
| 46 | 27 | Yannick Ngakoue | 69.85 | 58.71 | 73.62 | 835 | Raiders |
| 47 | 28 | John Franklin-Myers | 69.84 | 67.13 | 67.97 | 717 | Jets |
| 48 | 29 | Mario Addison | 69.66 | 51.94 | 77.83 | 481 | Bills |
| 49 | 30 | Odafe Oweh | 69.54 | 67.99 | 68.37 | 615 | Ravens |
| 50 | 31 | Jaelan Phillips | 69.54 | 59.60 | 72.00 | 603 | Dolphins |
| 51 | 32 | Takk McKinley | 69.43 | 62.43 | 73.70 | 320 | Browns |
| 52 | 33 | Ogbo Okoronkwo | 69.37 | 62.41 | 75.14 | 255 | Rams |
| 53 | 34 | Leonard Floyd | 69.22 | 59.96 | 71.23 | 932 | Rams |
| 54 | 35 | Bradley Chubb | 68.86 | 62.50 | 76.97 | 268 | Broncos |
| 55 | 36 | Chase Winovich | 68.52 | 59.74 | 74.12 | 112 | Patriots |
| 56 | 37 | Sam Hubbard | 68.24 | 61.76 | 70.03 | 877 | Bengals |
| 57 | 38 | Denico Autry | 67.97 | 50.72 | 75.30 | 710 | Titans |
| 58 | 39 | Samson Ebukam | 67.47 | 59.34 | 68.73 | 554 | 49ers |
| 59 | 40 | Dante Fowler Jr. | 67.14 | 59.48 | 70.18 | 508 | Falcons |
| 60 | 41 | Derek Barnett | 66.80 | 63.71 | 66.54 | 718 | Eagles |
| 61 | 42 | Deatrich Wise Jr. | 66.76 | 60.46 | 67.70 | 521 | Patriots |
| 62 | 43 | Pernell McPhee | 66.74 | 53.22 | 77.21 | 234 | Ravens |
| 63 | 44 | Dee Ford | 66.52 | 58.76 | 78.65 | 106 | 49ers |
| 64 | 45 | Jacob Martin | 66.49 | 58.43 | 68.73 | 700 | Texans |
| 65 | 46 | Alex Highsmith | 65.95 | 62.54 | 64.67 | 851 | Steelers |
| 66 | 47 | Josh Uche | 65.88 | 60.32 | 71.22 | 235 | Patriots |
| 67 | 48 | Carl Granderson | 65.75 | 61.09 | 68.07 | 448 | Saints |
| 68 | 49 | Genard Avery | 65.71 | 57.84 | 70.53 | 357 | Eagles |
| 69 | 50 | Everson Griffen | 65.68 | 53.28 | 74.53 | 457 | Vikings |
| 70 | 51 | Kwity Paye | 64.71 | 62.78 | 63.80 | 638 | Colts |
| 71 | 52 | Clelin Ferrell | 64.41 | 62.80 | 63.59 | 261 | Raiders |
| 72 | 53 | Terrell Lewis | 64.14 | 61.33 | 68.64 | 367 | Rams |
| 73 | 54 | Charles Omenihu | 64.14 | 58.89 | 64.46 | 355 | 49ers |
| 74 | 55 | Cam Gill | 64.05 | 59.47 | 72.14 | 100 | Buccaneers |
| 75 | 56 | Dawuane Smoot | 63.89 | 59.71 | 62.99 | 675 | Jaguars |
| 76 | 57 | Anthony Nelson | 63.88 | 65.79 | 59.91 | 359 | Buccaneers |
| 77 | 58 | Jordan Jenkins | 63.80 | 56.33 | 69.21 | 282 | Texans |
| 78 | 59 | Bud Dupree | 63.60 | 56.49 | 68.67 | 398 | Titans |
| 79 | 60 | Kerry Hyder Jr. | 63.55 | 53.96 | 66.76 | 508 | Seahawks |
| 80 | 61 | Charles Harris | 63.52 | 61.04 | 62.35 | 871 | Lions |
| 81 | 62 | Whitney Mercilus | 63.51 | 51.08 | 71.99 | 311 | Packers |
| 82 | 63 | Arden Key | 63.37 | 64.40 | 61.01 | 375 | 49ers |
| 83 | 64 | Efe Obada | 63.29 | 56.99 | 66.75 | 238 | Bills |
| 84 | 65 | Kenny Willekes | 63.25 | 59.11 | 71.57 | 202 | Vikings |
| 85 | 66 | Devon Kennard | 62.94 | 55.62 | 65.57 | 265 | Cardinals |
| 86 | 67 | Jonathon Cooper | 62.92 | 61.42 | 60.73 | 457 | Broncos |
| 87 | 68 | Lorenzo Carter | 62.90 | 59.32 | 66.23 | 617 | Giants |
| 88 | 69 | Yetur Gross-Matos | 62.88 | 61.17 | 63.25 | 349 | Panthers |
| 89 | 70 | A.J. Epenesa | 62.73 | 61.44 | 62.04 | 331 | Bills |
| 90 | 71 | Dennis Gardeck | 62.62 | 57.54 | 63.31 | 173 | Cardinals |
| 91 | 72 | Jeremiah Attaochu | 62.53 | 55.52 | 70.90 | 129 | Bears |
| 92 | 73 | Shaka Toney | 62.29 | 58.95 | 67.22 | 117 | Commanders |
| 93 | 74 | Romeo Okwara | 62.17 | 61.36 | 65.33 | 188 | Lions |
| 94 | 75 | Joe Tryon-Shoyinka | 62.13 | 57.88 | 60.79 | 560 | Buccaneers |
| 95 | 76 | Ifeadi Odenigbo | 62.12 | 57.34 | 65.37 | 162 | Browns |
| 96 | 77 | Jason Pierre-Paul | 62.00 | 50.02 | 69.52 | 601 | Buccaneers |

### Rotation/backup (54 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 97 | 1 | Marquis Haynes Sr. | 61.78 | 57.62 | 61.73 | 222 | Panthers |
| 98 | 2 | Alex Okafor | 61.71 | 51.19 | 67.38 | 463 | Chiefs |
| 99 | 3 | Kyler Fackrell | 61.58 | 52.46 | 66.71 | 382 | Chargers |
| 100 | 4 | Justin Hollins | 61.47 | 59.52 | 63.44 | 222 | Rams |
| 101 | 5 | Malik Reed | 60.82 | 55.86 | 61.64 | 737 | Broncos |
| 102 | 6 | Tarell Basham | 60.81 | 55.80 | 59.99 | 627 | Cowboys |
| 103 | 7 | Chauncey Golston | 60.78 | 59.32 | 59.55 | 414 | Cowboys |
| 104 | 8 | Quincy Roche | 60.78 | 59.59 | 60.34 | 401 | Giants |
| 105 | 9 | Dorance Armstrong | 60.73 | 59.69 | 59.42 | 507 | Cowboys |
| 106 | 10 | K'Lavon Chaisson | 60.48 | 57.99 | 59.21 | 384 | Jaguars |
| 107 | 11 | Carl Nassib | 60.41 | 58.62 | 60.44 | 251 | Raiders |
| 108 | 12 | Ronald Blair III | 60.14 | 56.57 | 64.95 | 315 | Jets |
| 109 | 13 | Cam Sample | 60.06 | 60.24 | 59.69 | 310 | Bengals |
| 110 | 14 | Jaylon Ferguson | 59.81 | 57.61 | 61.58 | 133 | Ravens |
| 111 | 15 | Bruce Irvin | 59.70 | 48.47 | 73.42 | 173 | Bears |
| 112 | 16 | Brennan Scarlett | 59.62 | 55.55 | 60.34 | 165 | Dolphins |
| 113 | 17 | Taco Charlton | 59.59 | 58.97 | 62.83 | 216 | Steelers |
| 114 | 18 | Benson Mayowa | 59.27 | 50.84 | 62.85 | 510 | Seahawks |
| 115 | 19 | Alton Robinson | 59.27 | 50.84 | 61.93 | 742 | Seahawks |
| 116 | 20 | Mike Danna | 59.12 | 58.17 | 56.76 | 534 | Chiefs |
| 117 | 21 | D.J. Wonnum | 59.11 | 54.92 | 58.52 | 951 | Vikings |
| 118 | 22 | Tarron Jackson | 59.07 | 59.49 | 54.62 | 253 | Eagles |
| 119 | 23 | Jihad Ward | 58.94 | 51.91 | 61.35 | 455 | Jaguars |
| 120 | 24 | Bryce Huff | 58.75 | 58.18 | 60.65 | 338 | Jets |
| 121 | 25 | Chris Rumph II | 58.72 | 58.99 | 56.34 | 176 | Chargers |
| 122 | 26 | Derrek Tuszka | 58.22 | 56.68 | 60.83 | 248 | Steelers |
| 123 | 27 | Stephen Weatherly | 58.13 | 54.66 | 59.44 | 344 | Broncos |
| 124 | 28 | Rasheem Green | 58.11 | 51.96 | 58.05 | 847 | Seahawks |
| 125 | 29 | Derek Rivers | 57.99 | 54.78 | 63.10 | 143 | Texans |
| 126 | 30 | Oshane Ximines | 57.88 | 58.41 | 61.02 | 183 | Giants |
| 127 | 31 | Al-Quadin Muhammad | 57.84 | 55.24 | 55.41 | 800 | Colts |
| 128 | 32 | Jonathan Garvin | 57.43 | 57.35 | 57.45 | 396 | Packers |
| 129 | 33 | James Vaughters | 57.39 | 55.61 | 59.47 | 210 | Falcons |
| 130 | 34 | Jessie Lemonier | 57.20 | 59.20 | 62.52 | 161 | Lions |
| 131 | 35 | Zach Allen | 57.08 | 53.79 | 57.02 | 684 | Cardinals |
| 132 | 36 | Tanoh Kpassagnon | 57.01 | 58.81 | 56.06 | 220 | Saints |
| 133 | 37 | Jordan Willis | 56.90 | 58.77 | 58.57 | 156 | 49ers |
| 134 | 38 | Brandon Copeland | 56.64 | 46.76 | 59.55 | 339 | Falcons |
| 135 | 39 | Dayo Odeyingbo | 56.56 | 61.09 | 56.24 | 173 | Colts |
| 136 | 40 | Olubunmi Rotimi | 56.52 | 56.33 | 60.33 | 204 | Commanders |
| 137 | 41 | Isaac Rochell | 56.52 | 57.29 | 54.29 | 177 | Colts |
| 138 | 42 | James Smith-Williams | 56.42 | 54.92 | 55.09 | 388 | Commanders |
| 139 | 43 | Khalid Kareem | 56.41 | 61.24 | 55.15 | 110 | Bengals |
| 140 | 44 | Casey Toohill | 56.16 | 57.28 | 55.38 | 361 | Commanders |
| 141 | 45 | Austin Bryant | 55.99 | 57.64 | 57.82 | 436 | Lions |
| 142 | 46 | Payton Turner | 55.89 | 63.62 | 62.83 | 144 | Saints |
| 143 | 47 | Tipa Galeai | 55.74 | 60.48 | 59.71 | 152 | Packers |
| 144 | 48 | Adetokunbo Ogundeji | 55.55 | 55.93 | 52.11 | 527 | Falcons |
| 145 | 49 | Kyle Phillips | 54.98 | 58.74 | 56.22 | 234 | Jets |
| 146 | 50 | Wyatt Ray | 54.64 | 57.99 | 54.16 | 219 | Bengals |
| 147 | 51 | Porter Gustin | 53.47 | 58.23 | 54.23 | 134 | Browns |
| 148 | 52 | Tim Ward | 52.07 | 57.59 | 51.86 | 191 | Jets |
| 149 | 53 | Elerson Smith | 51.97 | 59.70 | 56.44 | 107 | Giants |
| 150 | 54 | Steven Means | 51.96 | 46.11 | 53.16 | 693 | Falcons |

## G — Guard

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Zack Martin | 98.40 | 93.90 | 97.23 | 1101 | Cowboys |
| 2 | 2 | Joel Bitonio | 98.04 | 93.60 | 96.83 | 1107 | Browns |
| 3 | 3 | Wyatt Teller | 91.63 | 84.90 | 91.95 | 1107 | Browns |
| 4 | 4 | Ali Marpet | 91.36 | 83.60 | 92.37 | 1036 | Buccaneers |
| 5 | 5 | Shaq Mason | 91.21 | 84.97 | 91.20 | 955 | Patriots |
| 6 | 6 | Chris Lindstrom | 90.39 | 83.70 | 90.68 | 1034 | Falcons |
| 7 | 7 | Joe Thuney | 86.59 | 80.50 | 86.48 | 1184 | Chiefs |
| 8 | 8 | Wes Schweitzer | 84.97 | 71.93 | 89.50 | 401 | Commanders |
| 9 | 9 | Connor Williams | 84.96 | 75.79 | 86.91 | 948 | Cowboys |
| 10 | 10 | Laken Tomlinson | 83.95 | 75.90 | 85.15 | 1094 | 49ers |
| 11 | 11 | Kevin Zeitler | 82.79 | 75.10 | 83.75 | 1221 | Ravens |
| 12 | 12 | Isaac Seumalo | 82.00 | 66.11 | 88.42 | 168 | Eagles |
| 13 | 13 | Brandon Scherff | 81.76 | 72.30 | 83.90 | 697 | Commanders |
| 14 | 14 | Quinton Spain | 81.63 | 72.30 | 83.68 | 995 | Bengals |
| 15 | 15 | Alex Cappa | 81.34 | 73.40 | 82.47 | 1182 | Buccaneers |
| 16 | 16 | Trey Smith | 81.31 | 72.30 | 83.15 | 1194 | Chiefs |
| 17 | 17 | James Daniels | 80.67 | 71.80 | 82.41 | 1121 | Bears |

### Good (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Jonah Jackson | 79.87 | 69.30 | 82.75 | 1037 | Lions |
| 19 | 2 | Rodger Saffold | 79.25 | 68.91 | 81.97 | 853 | Titans |
| 20 | 3 | Ereck Flowers | 79.23 | 72.00 | 79.88 | 1061 | Commanders |
| 21 | 4 | Trai Turner | 79.05 | 69.40 | 81.31 | 1082 | Steelers |
| 22 | 5 | Quenton Nelson | 78.82 | 68.46 | 81.56 | 767 | Colts |
| 23 | 6 | Ezra Cleveland | 78.43 | 68.60 | 80.81 | 1140 | Vikings |
| 24 | 7 | Mark Glowinski | 78.29 | 69.65 | 79.88 | 843 | Colts |
| 25 | 8 | Greg Van Roten | 77.81 | 66.83 | 80.96 | 700 | Jets |
| 26 | 9 | Oday Aboushi | 77.79 | 64.51 | 82.47 | 298 | Chargers |
| 27 | 10 | Austin Corbett | 77.68 | 68.80 | 79.43 | 1081 | Rams |
| 28 | 11 | Sua Opeta | 77.68 | 65.74 | 81.47 | 163 | Eagles |
| 29 | 12 | Alijah Vera-Tucker | 77.63 | 66.80 | 80.69 | 1027 | Jets |
| 30 | 13 | Halapoulivaati Vaitai | 77.49 | 68.26 | 79.48 | 953 | Lions |
| 31 | 14 | Nate Davis | 77.39 | 68.75 | 78.98 | 951 | Titans |
| 32 | 15 | Robert Hunt | 77.16 | 67.40 | 79.50 | 1153 | Dolphins |
| 33 | 16 | Dalton Risner | 76.90 | 67.81 | 78.80 | 832 | Broncos |
| 34 | 17 | Landon Dickerson | 76.74 | 66.82 | 79.19 | 859 | Eagles |
| 35 | 18 | Jack Driscoll | 76.65 | 67.57 | 78.53 | 512 | Eagles |
| 36 | 19 | Andrew Norwell | 76.60 | 66.70 | 79.03 | 1077 | Jaguars |
| 37 | 20 | Connor McGovern | 76.50 | 66.19 | 79.20 | 499 | Cowboys |
| 38 | 21 | David Edwards | 76.22 | 66.90 | 78.26 | 1086 | Rams |
| 39 | 22 | Ben Powers | 75.77 | 65.83 | 78.23 | 844 | Ravens |
| 40 | 23 | Michael Schofield III | 75.55 | 66.53 | 77.40 | 907 | Chargers |
| 41 | 24 | Kevin Dotson | 75.08 | 63.41 | 78.70 | 565 | Steelers |
| 42 | 25 | Justin Pugh | 74.78 | 65.46 | 76.83 | 802 | Cardinals |
| 43 | 26 | Jon Runyan | 74.69 | 65.10 | 76.92 | 1053 | Packers |
| 44 | 27 | Graham Glasgow | 74.21 | 63.95 | 76.89 | 384 | Broncos |

### Starter (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 45 | 1 | Ben Bredeson | 73.74 | 57.92 | 80.12 | 294 | Giants |
| 46 | 2 | Gabe Jackson | 73.67 | 63.53 | 76.26 | 922 | Seahawks |
| 47 | 3 | Chris Reed | 72.92 | 65.24 | 73.87 | 522 | Colts |
| 48 | 4 | Ben Bartch | 72.65 | 61.78 | 75.73 | 705 | Jaguars |
| 49 | 5 | Michael Dunn | 72.45 | 63.16 | 74.48 | 128 | Browns |
| 50 | 6 | Daniel Brunskill | 72.26 | 61.40 | 75.33 | 1089 | 49ers |
| 51 | 7 | Solomon Kindley | 71.76 | 59.00 | 76.10 | 124 | Dolphins |
| 52 | 8 | Phil Haynes | 71.51 | 63.92 | 72.41 | 136 | Seahawks |
| 53 | 9 | Max Scharping | 71.32 | 59.92 | 74.76 | 689 | Texans |
| 54 | 10 | Jermaine Eluemunor | 70.66 | 59.90 | 73.66 | 266 | Raiders |
| 55 | 11 | Ike Boettger | 70.24 | 59.84 | 73.01 | 636 | Bills |
| 56 | 12 | Damien Lewis | 70.09 | 57.56 | 74.27 | 696 | Seahawks |
| 57 | 13 | Netane Muti | 69.74 | 59.43 | 72.45 | 317 | Broncos |
| 58 | 14 | Jon Feliciano | 69.49 | 57.79 | 73.13 | 442 | Bills |
| 59 | 15 | Will Hernandez | 68.98 | 55.90 | 73.53 | 1049 | Giants |
| 60 | 16 | Olisaemeka Udoh | 68.83 | 54.40 | 74.28 | 1075 | Vikings |
| 61 | 17 | Cesar Ruiz | 68.75 | 57.60 | 72.01 | 1091 | Saints |
| 62 | 18 | Andrus Peat | 68.34 | 55.62 | 72.66 | 303 | Saints |
| 63 | 19 | Dennis Daley | 68.29 | 53.75 | 73.81 | 573 | Panthers |
| 64 | 20 | John Leglue | 67.97 | 58.14 | 70.35 | 406 | Steelers |
| 65 | 21 | Jackson Carman | 67.79 | 57.47 | 70.51 | 462 | Bengals |
| 66 | 22 | Justin McCray | 67.47 | 53.31 | 72.75 | 545 | Texans |
| 67 | 23 | Royce Newman | 67.46 | 55.70 | 71.14 | 1084 | Packers |
| 68 | 24 | Aaron Brewer | 67.12 | 57.34 | 69.47 | 508 | Titans |
| 69 | 25 | Laurent Duvernay-Tardif | 67.05 | 54.71 | 71.11 | 390 | Jets |
| 70 | 26 | Sean Harlow | 66.93 | 57.12 | 69.30 | 441 | Cardinals |
| 71 | 27 | Ben Cleveland | 66.56 | 57.44 | 68.48 | 367 | Ravens |
| 72 | 28 | Xavier Su'a-Filo | 66.00 | 56.00 | 68.50 | 124 | Bengals |
| 73 | 29 | Michael Jordan | 65.73 | 52.14 | 70.62 | 703 | Panthers |
| 74 | 30 | Tommy Kraemer | 65.68 | 57.79 | 66.78 | 238 | Lions |
| 75 | 31 | John Simpson | 65.64 | 52.60 | 70.16 | 1112 | Raiders |
| 76 | 32 | John Miller | 64.79 | 53.55 | 68.11 | 656 | Panthers |
| 77 | 33 | A.J. Cann | 64.77 | 51.77 | 69.27 | 198 | Jaguars |
| 78 | 34 | Wes Martin | 63.56 | 52.12 | 67.02 | 130 | Giants |
| 79 | 35 | Senio Kelemete | 63.45 | 52.00 | 66.92 | 110 | Chargers |
| 80 | 36 | Cody Ford | 63.37 | 50.67 | 67.67 | 485 | Bills |
| 81 | 37 | Lane Taylor | 63.13 | 48.59 | 68.65 | 311 | Texans |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 82 | 1 | Calvin Throckmorton | 60.26 | 42.83 | 67.71 | 938 | Saints |

## HB — Running Back

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Nick Chubb | 84.19 | 77.76 | 84.31 | 185 | Browns |
| 2 | 2 | Jonathan Taylor | 84.12 | 85.09 | 79.31 | 292 | Colts |
| 3 | 3 | Tony Pollard | 83.16 | 77.42 | 82.82 | 168 | Cowboys |
| 4 | 4 | Aaron Jones | 81.20 | 81.59 | 76.78 | 316 | Packers |
| 5 | 5 | AJ Dillon | 80.39 | 79.32 | 76.93 | 199 | Packers |

### Good (14 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 6 | 1 | Javonte Williams | 79.76 | 73.20 | 79.96 | 262 | Broncos |
| 7 | 2 | Austin Ekeler | 78.87 | 75.30 | 77.08 | 417 | Chargers |
| 8 | 3 | Derrick Henry | 77.59 | 71.28 | 77.63 | 120 | Titans |
| 9 | 4 | Josh Jacobs | 77.38 | 76.47 | 73.82 | 297 | Raiders |
| 10 | 5 | Michael Carter | 77.13 | 68.37 | 78.81 | 220 | Jets |
| 11 | 6 | Kareem Hunt | 77.04 | 70.39 | 77.31 | 109 | Browns |
| 12 | 7 | Elijah Mitchell | 76.61 | 70.28 | 76.66 | 151 | 49ers |
| 13 | 8 | D'Ernest Johnson | 76.53 | 73.53 | 74.36 | 152 | Browns |
| 14 | 9 | Christian McCaffrey | 75.83 | 75.13 | 72.13 | 118 | Panthers |
| 15 | 10 | James Conner | 75.54 | 79.99 | 68.41 | 236 | Cardinals |
| 16 | 11 | Miles Sanders | 75.15 | 68.31 | 75.54 | 195 | Eagles |
| 17 | 12 | Cordarrelle Patterson | 75.03 | 77.10 | 69.49 | 245 | Falcons |
| 18 | 13 | Najee Harris | 75.01 | 70.70 | 73.72 | 467 | Steelers |
| 19 | 14 | Joe Mixon | 74.29 | 78.20 | 67.51 | 300 | Bengals |

### Starter (48 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | Melvin Gordon III | 73.93 | 75.73 | 68.56 | 247 | Broncos |
| 21 | 2 | Devin Singletary | 73.85 | 64.69 | 75.79 | 378 | Bills |
| 22 | 3 | Alvin Kamara | 73.84 | 62.90 | 76.96 | 271 | Saints |
| 23 | 4 | James Robinson | 73.57 | 66.64 | 74.02 | 261 | Jaguars |
| 24 | 5 | Dalvin Cook | 73.43 | 65.22 | 74.74 | 263 | Vikings |
| 25 | 6 | Kenyan Drake | 73.26 | 70.50 | 70.93 | 170 | Raiders |
| 26 | 7 | Damien Harris | 72.76 | 75.23 | 66.94 | 114 | Patriots |
| 27 | 8 | Brandon Bolden | 72.75 | 72.90 | 68.48 | 209 | Patriots |
| 28 | 9 | Justin Jackson | 72.74 | 61.77 | 75.88 | 124 | Chargers |
| 29 | 10 | Ezekiel Elliott | 72.36 | 68.90 | 70.50 | 420 | Cowboys |
| 30 | 11 | Leonard Fournette | 72.31 | 73.55 | 67.32 | 361 | Buccaneers |
| 31 | 12 | Khalil Herbert | 71.64 | 71.45 | 67.60 | 141 | Bears |
| 32 | 13 | Darrell Henderson | 70.88 | 67.47 | 68.98 | 280 | Rams |
| 33 | 14 | David Montgomery | 70.72 | 69.05 | 67.66 | 289 | Bears |
| 34 | 15 | Kenneth Gainwell | 70.68 | 67.41 | 68.70 | 182 | Eagles |
| 35 | 16 | Dontrell Hilliard | 70.53 | 58.88 | 74.13 | 109 | Titans |
| 36 | 17 | Nyheim Hines | 70.33 | 70.66 | 65.95 | 204 | Colts |
| 37 | 18 | Saquon Barkley | 70.11 | 59.20 | 73.21 | 257 | Giants |
| 38 | 19 | Chase Edmonds | 70.04 | 66.45 | 68.26 | 253 | Cardinals |
| 39 | 20 | Mark Ingram II | 69.74 | 64.74 | 68.91 | 148 | Saints |
| 40 | 21 | Clyde Edwards-Helaire | 69.07 | 63.49 | 68.62 | 178 | Chiefs |
| 41 | 22 | Antonio Gibson | 69.07 | 62.92 | 69.01 | 247 | Commanders |
| 42 | 23 | J.D. McKissic | 69.00 | 67.21 | 66.02 | 219 | Commanders |
| 43 | 24 | Alexander Mattison | 68.86 | 60.72 | 70.12 | 196 | Vikings |
| 44 | 25 | Jeremy McNichols | 68.20 | 60.25 | 69.34 | 149 | Titans |
| 45 | 26 | Devontae Booker | 68.17 | 64.08 | 66.73 | 253 | Giants |
| 46 | 27 | Devonta Freeman | 67.91 | 67.52 | 64.01 | 282 | Ravens |
| 47 | 28 | Sony Michel | 67.78 | 64.98 | 65.48 | 253 | Rams |
| 48 | 29 | Zack Moss | 67.55 | 64.97 | 65.10 | 176 | Bills |
| 49 | 30 | Jerick McKinnon | 67.35 | 64.84 | 64.85 | 125 | Chiefs |
| 50 | 31 | Samaje Perine | 67.05 | 61.45 | 66.61 | 164 | Bengals |
| 51 | 32 | Alex Collins | 67.03 | 63.18 | 65.43 | 107 | Seahawks |
| 52 | 33 | Jamaal Williams | 66.98 | 65.39 | 63.87 | 115 | Lions |
| 53 | 34 | Latavius Murray | 66.95 | 64.30 | 64.55 | 154 | Ravens |
| 54 | 35 | Myles Gaskin | 66.63 | 64.18 | 64.09 | 288 | Dolphins |
| 55 | 36 | Giovani Bernard | 66.31 | 61.87 | 65.10 | 108 | Buccaneers |
| 56 | 37 | Rex Burkhead | 66.12 | 63.81 | 63.50 | 189 | Texans |
| 57 | 38 | D'Andre Swift | 65.79 | 58.16 | 66.71 | 320 | Lions |
| 58 | 39 | DeeJay Dallas | 65.52 | 61.37 | 64.12 | 106 | Seahawks |
| 59 | 40 | David Johnson | 65.29 | 60.32 | 64.44 | 176 | Texans |
| 60 | 41 | Chuba Hubbard | 64.90 | 63.35 | 61.77 | 178 | Panthers |
| 61 | 42 | Carlos Hyde | 64.62 | 55.99 | 66.21 | 159 | Jaguars |
| 62 | 43 | Darrel Williams | 64.38 | 66.88 | 58.54 | 347 | Chiefs |
| 63 | 44 | Royce Freeman | 64.37 | 58.90 | 63.85 | 115 | Texans |
| 64 | 45 | Ameer Abdullah | 64.24 | 61.71 | 61.76 | 193 | Panthers |
| 65 | 46 | JaMycal Hasty | 63.94 | 60.86 | 61.83 | 110 | 49ers |
| 66 | 47 | Ty Johnson | 63.24 | 56.72 | 63.42 | 256 | Jets |
| 67 | 48 | Mike Davis | 62.90 | 56.32 | 63.12 | 313 | Falcons |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 68 | 1 | Dare Ogunbowale | 58.43 | 52.15 | 58.45 | 107 | Jaguars |

## LB — Linebacker

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | De'Vondre Campbell | 83.17 | 84.70 | 78.46 | 987 | Packers |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 2 | 1 | Micah Parsons | 78.23 | 89.80 | 67.34 | 902 | Cowboys |
| 3 | 2 | Lavonte David | 77.59 | 78.58 | 75.21 | 788 | Buccaneers |
| 4 | 3 | Demario Davis | 77.59 | 77.90 | 73.70 | 1038 | Saints |
| 5 | 4 | Jamie Collins Sr. | 76.43 | 78.55 | 73.93 | 301 | Patriots |
| 6 | 5 | Fred Warner | 75.36 | 75.20 | 71.78 | 977 | 49ers |
| 7 | 6 | Pete Werner | 75.35 | 73.79 | 75.15 | 394 | Saints |
| 8 | 7 | T.J. Edwards | 74.85 | 74.89 | 73.23 | 684 | Eagles |

### Starter (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Frankie Luvu | 73.66 | 73.67 | 69.97 | 249 | Panthers |
| 10 | 2 | Kyle Van Noy | 73.61 | 72.82 | 71.08 | 810 | Patriots |
| 11 | 3 | Bobby Wagner | 73.33 | 71.80 | 70.67 | 1129 | Seahawks |
| 12 | 4 | Josh Bynes | 72.39 | 73.11 | 70.04 | 537 | Ravens |
| 13 | 5 | Jeremiah Owusu-Koramoah | 72.16 | 74.08 | 69.65 | 597 | Browns |
| 14 | 6 | Shaq Thompson | 71.78 | 72.59 | 68.96 | 796 | Panthers |
| 15 | 7 | Nick Bolton | 71.30 | 68.02 | 70.30 | 623 | Chiefs |
| 16 | 8 | Mack Wilson Sr. | 71.19 | 67.42 | 72.63 | 193 | Browns |
| 17 | 9 | Matt Milano | 69.49 | 70.10 | 67.48 | 915 | Bills |
| 18 | 10 | Anthony Walker Jr. | 69.36 | 68.41 | 67.79 | 701 | Browns |
| 19 | 11 | Ja'Whaun Bentley | 68.97 | 67.54 | 67.19 | 693 | Patriots |
| 20 | 12 | Kyzir White | 68.92 | 66.50 | 67.93 | 979 | Chargers |
| 21 | 13 | Mykal Walker | 68.60 | 65.50 | 67.11 | 194 | Falcons |
| 22 | 14 | Jaylon Smith | 67.73 | 64.91 | 68.87 | 329 | Giants |
| 23 | 15 | Sione Takitaki | 67.64 | 64.42 | 68.58 | 285 | Browns |
| 24 | 16 | Jordan Hicks | 67.22 | 64.70 | 64.74 | 1053 | Cardinals |
| 25 | 17 | Zaire Franklin | 67.14 | 65.59 | 69.10 | 201 | Colts |
| 26 | 18 | K.J. Wright | 66.67 | 63.08 | 64.90 | 426 | Raiders |
| 27 | 19 | Zaven Collins | 66.19 | 64.82 | 66.85 | 220 | Cardinals |
| 28 | 20 | Denzel Perryman | 65.87 | 62.30 | 66.84 | 863 | Raiders |
| 29 | 21 | Duke Riley | 65.52 | 63.79 | 66.09 | 227 | Dolphins |
| 30 | 22 | Zach Cunningham | 65.36 | 60.18 | 66.11 | 646 | Titans |
| 31 | 23 | Reggie Ragland | 65.29 | 60.91 | 65.44 | 474 | Giants |
| 32 | 24 | Leighton Vander Esch | 65.20 | 63.29 | 65.64 | 661 | Cowboys |
| 33 | 25 | Azeez Al-Shaair | 64.86 | 64.53 | 65.06 | 730 | 49ers |
| 34 | 26 | Travin Howard | 64.85 | 63.92 | 69.06 | 103 | Rams |
| 35 | 27 | Jordyn Brooks | 64.56 | 58.40 | 65.29 | 1109 | Seahawks |
| 36 | 28 | Dre Greenlaw | 64.27 | 63.80 | 68.21 | 113 | 49ers |
| 37 | 29 | Jerome Baker | 64.19 | 60.90 | 62.70 | 971 | Dolphins |
| 38 | 30 | Drue Tranquill | 64.13 | 63.80 | 67.16 | 560 | Chargers |
| 39 | 31 | David Long Jr. | 63.52 | 66.51 | 64.54 | 634 | Titans |
| 40 | 32 | Eric Kendricks | 63.28 | 59.20 | 64.59 | 1032 | Vikings |
| 41 | 33 | Shaquille Quarterman | 63.22 | 61.22 | 64.67 | 144 | Jaguars |
| 42 | 34 | Sam Eguavoen | 63.21 | 61.55 | 61.13 | 181 | Dolphins |
| 43 | 35 | Bobby Okereke | 62.50 | 58.50 | 61.63 | 1072 | Colts |
| 44 | 36 | Blake Lynch | 62.17 | 62.37 | 67.24 | 218 | Vikings |

### Rotation/backup (75 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 45 | 1 | Akeem Davis-Gaither | 61.56 | 60.10 | 63.26 | 207 | Bengals |
| 46 | 2 | Divine Deablo | 61.55 | 61.93 | 65.95 | 297 | Raiders |
| 47 | 3 | Ernest Jones | 61.51 | 59.49 | 64.57 | 440 | Rams |
| 48 | 4 | Cole Holcomb | 61.17 | 56.70 | 62.35 | 1021 | Commanders |
| 49 | 5 | Jonas Griffith | 61.15 | 65.07 | 70.64 | 255 | Broncos |
| 50 | 6 | Anthony Barr | 60.63 | 62.86 | 62.71 | 783 | Vikings |
| 51 | 7 | Alex Singleton | 60.48 | 52.88 | 62.18 | 720 | Eagles |
| 52 | 8 | Dont'a Hightower | 60.30 | 54.16 | 61.72 | 634 | Patriots |
| 53 | 9 | Malcolm Smith | 60.26 | 57.50 | 62.15 | 425 | Browns |
| 54 | 10 | Roquan Smith | 60.17 | 47.80 | 65.09 | 1010 | Bears |
| 55 | 11 | Tremaine Edmunds | 60.01 | 50.40 | 63.55 | 872 | Bills |
| 56 | 12 | Chris Board | 59.76 | 55.90 | 60.54 | 337 | Ravens |
| 57 | 13 | Oren Burks | 59.57 | 56.89 | 59.89 | 206 | Packers |
| 58 | 14 | Cody Barton | 59.40 | 59.86 | 63.01 | 190 | Seahawks |
| 59 | 15 | Elandon Roberts | 59.37 | 54.87 | 59.56 | 620 | Dolphins |
| 60 | 16 | Markus Bailey | 59.36 | 57.54 | 62.75 | 256 | Bengals |
| 61 | 17 | Krys Barnes | 59.31 | 54.39 | 60.60 | 526 | Packers |
| 62 | 18 | Jacob Phillips | 59.24 | 60.96 | 64.63 | 123 | Browns |
| 63 | 19 | A.J. Klein | 59.04 | 56.11 | 59.98 | 277 | Bills |
| 64 | 20 | Willie Gay | 58.97 | 56.86 | 59.66 | 436 | Chiefs |
| 65 | 21 | Joe Schobert | 58.96 | 52.10 | 59.85 | 921 | Steelers |
| 66 | 22 | Neville Hewitt | 58.71 | 53.33 | 61.89 | 325 | Texans |
| 67 | 23 | Isaiah Simmons | 58.58 | 51.00 | 59.47 | 1005 | Cardinals |
| 68 | 24 | Blake Martinez | 58.31 | 58.57 | 60.84 | 142 | Giants |
| 69 | 25 | Kwon Alexander | 58.25 | 55.07 | 61.57 | 535 | Saints |
| 70 | 26 | Christian Jones | 58.11 | 54.12 | 60.17 | 116 | Bears |
| 71 | 27 | E.J. Speed | 57.86 | 56.46 | 63.50 | 146 | Colts |
| 72 | 28 | Logan Wilson | 57.82 | 54.34 | 59.99 | 707 | Bengals |
| 73 | 29 | Jalen Reeves-Maybin | 57.56 | 56.36 | 59.03 | 615 | Lions |
| 74 | 30 | Foyesade Oluokun | 57.54 | 47.00 | 60.72 | 1148 | Falcons |
| 75 | 31 | Baron Browning | 57.18 | 55.91 | 60.73 | 528 | Broncos |
| 76 | 32 | Kevin Minter | 56.89 | 56.00 | 58.27 | 331 | Buccaneers |
| 77 | 33 | Jayon Brown | 56.80 | 52.70 | 61.09 | 421 | Titans |
| 78 | 34 | Kenny Young | 56.72 | 51.66 | 60.40 | 645 | Broncos |
| 79 | 35 | Josh Woods | 56.61 | 58.36 | 61.40 | 113 | Lions |
| 80 | 36 | Christian Kirksey | 56.33 | 50.48 | 62.50 | 790 | Texans |
| 81 | 37 | Ben Niemann | 56.12 | 49.36 | 56.97 | 558 | Chiefs |
| 82 | 38 | Cory Littleton | 55.97 | 47.97 | 58.74 | 663 | Raiders |
| 83 | 39 | Anthony Hitchens | 55.75 | 45.50 | 60.23 | 597 | Chiefs |
| 84 | 40 | Robert Spillane | 55.54 | 53.36 | 59.66 | 347 | Steelers |
| 85 | 41 | Germaine Pratt | 55.34 | 48.06 | 57.21 | 692 | Bengals |
| 86 | 42 | Jamin Davis | 55.28 | 48.89 | 56.35 | 581 | Commanders |
| 87 | 43 | Rashaan Evans | 54.78 | 48.58 | 57.20 | 445 | Titans |
| 88 | 44 | Zack Baun | 54.69 | 52.51 | 58.44 | 194 | Saints |
| 89 | 45 | Malik Harrison | 54.58 | 51.83 | 57.76 | 171 | Ravens |
| 90 | 46 | Del'Shawn Phillips | 54.48 | 50.30 | 55.55 | 161 | Jets |
| 91 | 47 | Monty Rice | 54.26 | 58.18 | 61.26 | 179 | Titans |
| 92 | 48 | Troy Reeder | 54.03 | 47.87 | 56.78 | 682 | Rams |
| 93 | 49 | Patrick Queen | 54.01 | 43.50 | 56.85 | 826 | Ravens |
| 94 | 50 | David Mayo | 53.95 | 48.54 | 58.81 | 166 | Commanders |
| 95 | 51 | Nick Vigil | 53.75 | 43.25 | 58.32 | 718 | Vikings |
| 96 | 52 | Damien Wilson | 53.74 | 44.00 | 57.00 | 866 | Jaguars |
| 97 | 53 | Eric Wilson | 53.14 | 47.70 | 56.52 | 298 | Texans |
| 98 | 54 | Joe Bachie | 53.03 | 56.42 | 60.61 | 160 | Bengals |
| 99 | 55 | Derrick Barnes | 52.87 | 40.00 | 58.26 | 448 | Lions |
| 100 | 56 | C.J. Mosley | 52.43 | 42.00 | 58.61 | 1098 | Jets |
| 101 | 57 | Kamu Grugier-Hill | 52.36 | 44.61 | 57.33 | 778 | Texans |
| 102 | 58 | Jermaine Carter | 51.88 | 42.60 | 56.08 | 852 | Panthers |
| 103 | 59 | Devin White | 51.56 | 40.00 | 56.03 | 1080 | Buccaneers |
| 104 | 60 | Jon Bostic | 51.51 | 48.04 | 56.03 | 179 | Commanders |
| 105 | 61 | Myles Jack | 51.45 | 40.00 | 57.56 | 917 | Jaguars |
| 106 | 62 | Amen Ogbongbemiga | 51.40 | 49.32 | 55.48 | 111 | Chargers |
| 107 | 63 | Deion Jones | 51.18 | 40.00 | 54.95 | 1070 | Falcons |
| 108 | 64 | Devin Bush | 51.18 | 40.00 | 59.37 | 762 | Steelers |
| 109 | 65 | Tanner Vallejo | 50.89 | 50.00 | 56.97 | 121 | Cardinals |
| 110 | 66 | Tae Crowder | 50.83 | 40.00 | 56.23 | 1099 | Giants |
| 111 | 67 | Garret Wallow | 50.63 | 47.77 | 55.23 | 180 | Texans |
| 112 | 68 | Quincy Williams | 50.63 | 44.20 | 57.06 | 881 | Jets |
| 113 | 69 | Kenneth Murray Jr. | 50.48 | 42.70 | 55.18 | 363 | Chargers |
| 114 | 70 | Alec Ogletree | 50.15 | 40.00 | 54.80 | 697 | Bears |
| 115 | 71 | Keanu Neal | 50.03 | 40.00 | 55.49 | 579 | Cowboys |
| 116 | 72 | Justin Strnad | 49.78 | 42.67 | 55.26 | 314 | Broncos |
| 117 | 73 | Davion Taylor | 49.54 | 48.18 | 55.87 | 250 | Eagles |
| 118 | 74 | Jarrad Davis | 49.47 | 44.15 | 54.43 | 209 | Jets |
| 119 | 75 | Alex Anzalone | 48.87 | 40.00 | 55.31 | 827 | Lions |

## QB — Quarterback

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_pass` · **Snap column (volume filter):** `passing_snaps`

### Elite (8 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Rodgers | 86.11 | 88.23 | 81.36 | 609 | Packers |
| 2 | 2 | Joe Burrow | 85.40 | 87.32 | 81.68 | 629 | Bengals |
| 3 | 3 | Tom Brady | 84.99 | 89.75 | 76.55 | 781 | Buccaneers |
| 4 | 4 | Kirk Cousins | 82.62 | 85.22 | 77.23 | 647 | Vikings |
| 5 | 5 | Dak Prescott | 81.36 | 83.28 | 77.98 | 705 | Cowboys |
| 6 | 6 | Kyler Murray | 81.17 | 81.99 | 77.81 | 578 | Cardinals |
| 7 | 7 | Matthew Stafford | 80.93 | 79.25 | 78.93 | 670 | Rams |
| 8 | 8 | Justin Herbert | 80.56 | 83.91 | 73.28 | 793 | Chargers |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Derek Carr | 79.53 | 78.60 | 76.21 | 727 | Raiders |
| 10 | 2 | Russell Wilson | 78.56 | 77.61 | 78.19 | 486 | Seahawks |
| 11 | 3 | Patrick Mahomes | 78.51 | 77.91 | 74.69 | 787 | Chiefs |
| 12 | 4 | Josh Allen | 77.34 | 79.08 | 71.55 | 771 | Bills |
| 13 | 5 | Ryan Tannehill | 77.24 | 82.60 | 70.60 | 633 | Titans |
| 14 | 6 | Matt Ryan | 75.92 | 77.42 | 70.67 | 653 | Falcons |

### Starter (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Jimmy Garoppolo | 72.93 | 72.03 | 75.86 | 501 | 49ers |
| 16 | 2 | Lamar Jackson | 70.03 | 70.00 | 70.30 | 495 | Ravens |
| 17 | 3 | Teddy Bridgewater | 69.99 | 69.01 | 70.96 | 495 | Broncos |
| 18 | 4 | Mac Jones | 69.98 | 77.40 | 72.33 | 600 | Patriots |
| 19 | 5 | Carson Wentz | 69.04 | 67.02 | 68.19 | 622 | Colts |
| 20 | 6 | Baker Mayfield | 68.42 | 68.32 | 67.44 | 516 | Browns |
| 21 | 7 | Jared Goff | 68.09 | 65.59 | 67.59 | 574 | Lions |
| 22 | 8 | Jalen Hurts | 66.65 | 70.91 | 68.48 | 541 | Eagles |
| 23 | 9 | Daniel Jones | 66.23 | 70.07 | 63.86 | 439 | Giants |
| 24 | 10 | Jameis Winston | 66.13 | 64.43 | 75.55 | 199 | Saints |
| 25 | 11 | Tua Tagovailoa | 64.21 | 65.70 | 66.70 | 460 | Dolphins |
| 26 | 12 | Ben Roethlisberger | 63.23 | 58.34 | 64.31 | 692 | Steelers |
| 27 | 13 | Geno Smith | 63.17 | 65.02 | 77.80 | 118 | Seahawks |

### Rotation/backup (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Davis Mills | 61.48 | 59.12 | 67.69 | 469 | Texans |
| 29 | 2 | Taylor Heinicke | 60.95 | 58.95 | 65.87 | 616 | Commanders |
| 30 | 3 | Colt McCoy | 60.45 | 58.89 | 71.81 | 115 | Cardinals |
| 31 | 4 | Trevor Lawrence | 60.13 | 58.30 | 57.17 | 709 | Jaguars |
| 32 | 5 | Justin Fields | 59.96 | 60.64 | 63.03 | 378 | Bears |
| 33 | 6 | Andy Dalton | 59.90 | 64.71 | 60.19 | 279 | Bears |
| 34 | 7 | Trevor Siemian | 59.59 | 61.98 | 65.81 | 218 | Saints |
| 35 | 8 | Drew Lock | 59.23 | 60.77 | 64.94 | 138 | Broncos |
| 36 | 9 | Jacoby Brissett | 58.82 | 66.86 | 59.01 | 277 | Dolphins |
| 37 | 10 | Taysom Hill | 58.62 | 61.32 | 62.45 | 162 | Saints |
| 38 | 11 | Tyler Huntley | 58.01 | 61.28 | 58.03 | 245 | Ravens |
| 39 | 12 | Mike White | 57.70 | 55.67 | 63.35 | 147 | Jets |
| 40 | 13 | Zach Wilson | 57.66 | 55.37 | 56.31 | 469 | Jets |
| 41 | 14 | Tyrod Taylor | 56.96 | 56.91 | 59.33 | 188 | Texans |
| 42 | 15 | Sam Darnold | 56.16 | 55.21 | 57.48 | 486 | Panthers |
| 43 | 16 | Tim Boyle | 56.05 | 56.71 | 55.81 | 100 | Lions |
| 44 | 17 | Cam Newton | 55.29 | 58.37 | 57.25 | 147 | Panthers |
| 45 | 18 | Mike Glennon | 54.36 | 49.95 | 54.23 | 193 | Giants |

## S — Safety

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (10 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Kevin Byard | 93.79 | 90.70 | 91.69 | 1058 | Titans |
| 2 | 2 | Jevon Holland | 89.17 | 87.70 | 86.97 | 893 | Dolphins |
| 3 | 3 | Jordan Poyer | 88.33 | 89.20 | 84.07 | 983 | Bills |
| 4 | 4 | Micah Hyde | 87.55 | 88.30 | 83.20 | 1023 | Bills |
| 5 | 5 | Adrian Phillips | 86.52 | 86.30 | 84.38 | 883 | Patriots |
| 6 | 6 | Antoine Winfield Jr. | 85.48 | 87.20 | 82.61 | 876 | Buccaneers |
| 7 | 7 | Harrison Smith | 84.96 | 81.80 | 84.09 | 1048 | Vikings |
| 8 | 8 | Marcus Williams | 84.74 | 84.30 | 82.19 | 1037 | Saints |
| 9 | 9 | Amani Hooker | 82.27 | 81.63 | 80.98 | 705 | Titans |
| 10 | 10 | Jayron Kearse | 82.03 | 76.20 | 82.24 | 1012 | Cowboys |

### Good (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Devin McCourty | 79.86 | 76.20 | 78.14 | 1019 | Patriots |
| 12 | 2 | Adrian Amos | 79.84 | 74.10 | 79.50 | 1047 | Packers |
| 13 | 3 | Xavier McKinney | 79.59 | 78.40 | 80.11 | 1134 | Giants |
| 14 | 4 | Tre'von Moehrig | 79.31 | 77.70 | 76.22 | 1152 | Raiders |
| 15 | 5 | Jimmie Ward | 79.08 | 73.70 | 80.24 | 991 | 49ers |
| 16 | 6 | DeAndre Houston-Carson | 78.59 | 75.70 | 82.27 | 420 | Bears |
| 17 | 7 | Justin Simmons | 78.01 | 73.40 | 76.92 | 1082 | Broncos |
| 18 | 8 | Vonn Bell | 77.96 | 78.00 | 74.88 | 1004 | Bengals |
| 19 | 9 | Derwin James Jr. | 77.72 | 76.60 | 78.72 | 961 | Chargers |
| 20 | 10 | Tyrann Mathieu | 76.94 | 76.40 | 73.93 | 996 | Chiefs |
| 21 | 11 | Jeremy Chinn | 76.12 | 74.30 | 74.16 | 1015 | Panthers |
| 22 | 12 | Quandre Diggs | 75.74 | 72.30 | 75.12 | 1230 | Seahawks |
| 23 | 13 | Eric Rowe | 75.11 | 69.41 | 74.74 | 638 | Dolphins |
| 24 | 14 | Kyle Dugger | 74.75 | 74.80 | 72.55 | 733 | Patriots |
| 25 | 15 | Mike Edwards | 74.64 | 72.29 | 74.97 | 532 | Buccaneers |
| 26 | 16 | Jordan Fuller | 74.26 | 69.40 | 75.51 | 1028 | Rams |

### Starter (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 27 | 1 | Terrell Edmunds | 73.54 | 71.60 | 70.99 | 1145 | Steelers |
| 28 | 2 | Marcus Epps | 73.48 | 66.18 | 77.06 | 505 | Eagles |
| 29 | 3 | Juan Thornhill | 73.43 | 68.60 | 72.48 | 850 | Chiefs |
| 30 | 4 | Jalen Thompson | 73.34 | 72.90 | 73.74 | 986 | Cardinals |
| 31 | 5 | Jordan Whitehead | 73.31 | 72.07 | 71.86 | 795 | Buccaneers |
| 32 | 6 | Andrew Wingard | 72.40 | 75.10 | 69.91 | 930 | Jaguars |
| 33 | 7 | John Johnson III | 70.63 | 66.50 | 72.29 | 903 | Browns |
| 34 | 8 | Budda Baker | 69.87 | 65.60 | 68.86 | 1036 | Cardinals |
| 35 | 9 | Jahleel Addae | 69.29 | 64.61 | 75.99 | 132 | Colts |
| 36 | 10 | Xavier Woods | 68.95 | 58.30 | 72.40 | 1207 | Vikings |
| 37 | 11 | Andrew Adams | 68.24 | 64.07 | 75.86 | 214 | Buccaneers |
| 38 | 12 | Dean Marlowe | 67.64 | 62.76 | 70.52 | 700 | Lions |
| 39 | 13 | Grant Delpit | 67.50 | 63.83 | 67.02 | 599 | Browns |
| 40 | 14 | Chuck Clark | 67.32 | 61.10 | 68.00 | 1023 | Ravens |
| 41 | 15 | Tracy Walker III | 67.30 | 62.40 | 68.31 | 881 | Lions |
| 42 | 16 | Deon Bush | 67.12 | 65.68 | 69.15 | 377 | Bears |
| 43 | 17 | Anthony Harris | 66.90 | 59.60 | 69.49 | 834 | Eagles |
| 44 | 18 | Darnell Savage | 66.82 | 62.10 | 66.54 | 1037 | Packers |
| 45 | 19 | Taylor Rapp | 66.48 | 58.80 | 69.84 | 1113 | Rams |
| 46 | 20 | Malcolm Jenkins | 66.46 | 60.80 | 66.55 | 1042 | Saints |
| 47 | 21 | Damontae Kazee | 66.15 | 65.10 | 66.43 | 900 | Cowboys |
| 48 | 22 | Duron Harmon | 66.04 | 53.60 | 70.16 | 1071 | Falcons |
| 49 | 23 | Malik Hooker | 65.88 | 64.62 | 68.53 | 445 | Cowboys |
| 50 | 24 | Andrew Sendejo | 65.58 | 64.67 | 65.72 | 610 | Colts |
| 51 | 25 | Sean Chandler | 65.45 | 62.10 | 67.74 | 538 | Panthers |
| 52 | 26 | Ashtyn Davis | 65.40 | 65.80 | 66.55 | 745 | Jets |
| 53 | 27 | Andre Cisco | 65.38 | 61.48 | 69.70 | 247 | Jaguars |
| 54 | 28 | Talanoa Hufanga | 65.35 | 60.14 | 67.59 | 395 | 49ers |
| 55 | 29 | Rodney McLeod | 65.22 | 58.49 | 68.44 | 684 | Eagles |
| 56 | 30 | Jonathan Owens | 65.17 | 65.01 | 74.68 | 168 | Texans |
| 57 | 31 | Rayshawn Jenkins | 65.17 | 61.30 | 65.36 | 836 | Jaguars |
| 58 | 32 | Nasir Adderley | 65.02 | 62.90 | 66.48 | 987 | Chargers |
| 59 | 33 | Julian Love | 64.84 | 57.76 | 65.77 | 612 | Giants |
| 60 | 34 | George Odum | 64.34 | 60.53 | 67.87 | 472 | Colts |
| 61 | 35 | Ricardo Allen | 63.97 | 58.45 | 66.70 | 171 | Bengals |
| 62 | 36 | Eddie Jackson | 63.69 | 56.94 | 65.49 | 787 | Bears |
| 63 | 37 | Erik Harris | 63.33 | 60.29 | 64.47 | 702 | Falcons |
| 64 | 38 | Ronnie Harrison | 62.89 | 55.37 | 68.17 | 584 | Browns |
| 65 | 39 | Jaquiski Tartt | 62.63 | 62.02 | 63.98 | 727 | 49ers |
| 66 | 40 | Shawn Williams | 62.60 | 62.34 | 66.06 | 164 | Falcons |
| 67 | 41 | Geno Stone | 62.54 | 61.91 | 68.95 | 219 | Ravens |
| 68 | 42 | Jessie Bates III | 62.47 | 53.40 | 65.33 | 953 | Bengals |
| 69 | 43 | Daniel Thomas | 62.08 | 58.21 | 71.14 | 205 | Jaguars |

### Rotation/backup (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 70 | 1 | Donovan Wilson | 61.99 | 62.62 | 65.60 | 338 | Cowboys |
| 71 | 2 | Caden Sterns | 61.94 | 56.56 | 64.30 | 311 | Broncos |
| 72 | 3 | Minkah Fitzpatrick | 61.65 | 49.40 | 66.14 | 1083 | Steelers |
| 73 | 4 | Tavon Wilson | 61.37 | 60.90 | 65.76 | 102 | 49ers |
| 74 | 5 | Dallin Leavitt | 61.30 | 60.44 | 65.04 | 250 | Raiders |
| 75 | 6 | DeShon Elliott | 60.78 | 61.03 | 64.35 | 305 | Ravens |
| 76 | 7 | Jaylinn Hawkins | 60.57 | 60.67 | 62.47 | 462 | Falcons |
| 77 | 8 | Khari Willis | 60.36 | 51.81 | 65.87 | 564 | Colts |
| 78 | 9 | Jabrill Peppers | 59.62 | 55.13 | 65.20 | 229 | Giants |
| 79 | 10 | Juston Burris | 58.93 | 60.00 | 60.12 | 420 | Panthers |
| 80 | 11 | C.J. Moore | 58.70 | 61.01 | 62.30 | 158 | Lions |
| 81 | 12 | Jeremy Reaves | 58.67 | 55.62 | 67.21 | 195 | Commanders |
| 82 | 13 | Brandon Stephens | 58.58 | 47.01 | 62.12 | 742 | Ravens |
| 83 | 14 | Dane Cruikshank | 58.00 | 58.72 | 62.08 | 415 | Titans |
| 84 | 15 | Julian Blackmon | 57.94 | 55.41 | 62.59 | 376 | Colts |
| 85 | 16 | Tashaun Gipson Sr. | 57.89 | 51.38 | 60.93 | 660 | Bears |
| 86 | 17 | Terrence Brooks | 57.78 | 54.16 | 60.61 | 180 | Texans |
| 87 | 18 | K'Von Wallace | 57.77 | 56.47 | 63.38 | 183 | Eagles |
| 88 | 19 | Alohi Gilman | 57.55 | 59.74 | 60.29 | 355 | Chargers |
| 89 | 20 | Johnathan Abram | 57.29 | 54.20 | 60.71 | 955 | Raiders |
| 90 | 21 | Eric Murray | 57.29 | 54.92 | 57.28 | 759 | Texans |
| 91 | 22 | Marcus Maye | 57.07 | 51.95 | 61.71 | 362 | Jets |
| 92 | 23 | Daniel Sorensen | 57.04 | 52.95 | 55.91 | 699 | Chiefs |
| 93 | 24 | Jamal Adams | 56.20 | 47.40 | 62.02 | 872 | Seahawks |
| 94 | 25 | Nick Scott | 55.54 | 50.78 | 59.65 | 415 | Rams |
| 95 | 26 | Kareem Jackson | 55.42 | 48.50 | 57.46 | 895 | Broncos |
| 96 | 27 | Will Harris | 55.32 | 41.70 | 60.24 | 1011 | Lions |
| 97 | 28 | Kenny Robinson | 55.21 | 57.98 | 60.80 | 182 | Panthers |
| 98 | 29 | Brandon Jones | 54.74 | 49.93 | 55.02 | 644 | Dolphins |
| 99 | 30 | Justin Reid | 53.04 | 45.54 | 56.97 | 780 | Texans |
| 100 | 31 | Henry Black | 52.70 | 51.02 | 54.33 | 263 | Packers |
| 101 | 32 | Roderic Teamer | 51.93 | 56.11 | 55.23 | 195 | Raiders |
| 102 | 33 | Landon Collins | 51.16 | 40.01 | 59.41 | 675 | Commanders |
| 103 | 34 | Trey Marshall | 50.35 | 52.23 | 51.93 | 197 | Chargers |
| 104 | 35 | Adrian Colbert | 49.57 | 49.69 | 57.04 | 161 | Browns |
| 105 | 36 | Sam Franklin Jr. | 49.23 | 49.97 | 52.20 | 153 | Panthers |

## T — Tackle

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Trent Williams | 98.93 | 97.23 | 95.89 | 936 | 49ers |
| 2 | 2 | Tyron Smith | 95.62 | 88.87 | 95.96 | 738 | Cowboys |
| 3 | 3 | Jordan Mailata | 92.68 | 86.39 | 92.71 | 914 | Eagles |
| 4 | 4 | Tristan Wirfs | 90.32 | 84.60 | 89.97 | 1182 | Buccaneers |
| 5 | 5 | Andrew Whitworth | 90.03 | 85.62 | 88.80 | 926 | Rams |
| 6 | 6 | La'el Collins | 89.94 | 79.69 | 92.61 | 671 | Cowboys |
| 7 | 7 | Rashawn Slater | 89.69 | 83.60 | 89.58 | 1116 | Chargers |
| 8 | 8 | Kolton Miller | 89.03 | 84.00 | 88.22 | 1139 | Raiders |
| 9 | 9 | Ryan Ramczyk | 88.77 | 81.41 | 89.51 | 653 | Saints |
| 10 | 10 | Lane Johnson | 88.73 | 81.23 | 89.57 | 821 | Eagles |
| 11 | 11 | Donovan Smith | 87.92 | 83.30 | 86.83 | 1147 | Buccaneers |
| 12 | 12 | Rob Havenstein | 87.67 | 81.32 | 87.73 | 957 | Rams |
| 13 | 13 | David Quessenberry | 87.63 | 80.70 | 88.08 | 1184 | Titans |
| 14 | 14 | Braden Smith | 87.28 | 78.74 | 88.81 | 711 | Colts |
| 15 | 15 | Penei Sewell | 86.25 | 77.00 | 88.25 | 1039 | Lions |
| 16 | 16 | Elgton Jenkins | 85.56 | 75.68 | 87.98 | 496 | Packers |
| 17 | 17 | Jack Conklin | 85.34 | 74.35 | 88.50 | 361 | Browns |
| 18 | 18 | Charles Leno Jr. | 85.23 | 81.20 | 83.75 | 1121 | Commanders |
| 19 | 19 | Taylor Moton | 85.15 | 77.50 | 86.09 | 1149 | Panthers |
| 20 | 20 | Trent Brown | 84.94 | 75.88 | 86.81 | 489 | Patriots |
| 21 | 21 | Andrew Thomas | 84.80 | 77.03 | 85.82 | 800 | Giants |
| 22 | 22 | Dion Dawkins | 84.72 | 77.50 | 85.37 | 1089 | Bills |
| 23 | 23 | Jonah Williams | 84.69 | 77.10 | 85.58 | 1044 | Bengals |
| 24 | 24 | Jason Peters | 83.81 | 76.77 | 84.33 | 853 | Bears |
| 25 | 25 | Garett Bolles | 83.63 | 76.00 | 84.55 | 870 | Broncos |
| 26 | 26 | Isaiah Wynn | 83.45 | 74.58 | 85.19 | 915 | Patriots |
| 27 | 27 | Orlando Brown Jr. | 82.76 | 75.40 | 83.50 | 1127 | Chiefs |
| 28 | 28 | Brian O'Neill | 82.61 | 73.40 | 84.58 | 1140 | Vikings |
| 29 | 29 | Taylor Decker | 82.47 | 73.02 | 84.61 | 529 | Lions |
| 30 | 30 | Terron Armstead | 82.31 | 72.94 | 84.39 | 468 | Saints |
| 31 | 31 | Sam Cosmi | 81.97 | 70.34 | 85.56 | 474 | Commanders |
| 32 | 32 | Cornelius Lucas | 81.15 | 71.73 | 83.27 | 587 | Commanders |
| 33 | 33 | Morgan Moses | 80.95 | 71.00 | 83.42 | 1022 | Jets |
| 34 | 34 | Christian Darrisaw | 80.87 | 69.68 | 84.17 | 652 | Vikings |
| 35 | 35 | Duane Brown | 80.58 | 71.44 | 82.51 | 969 | Seahawks |
| 36 | 36 | Joe Noteboom | 80.18 | 66.72 | 84.98 | 174 | Rams |
| 37 | 37 | George Fant | 80.02 | 70.55 | 82.17 | 889 | Jets |

### Good (37 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 38 | 1 | Jake Matthews | 79.81 | 71.30 | 81.31 | 1029 | Falcons |
| 39 | 2 | Taylor Lewan | 79.71 | 70.33 | 81.79 | 846 | Titans |
| 40 | 3 | Bobby Massie | 79.43 | 68.99 | 82.22 | 796 | Broncos |
| 41 | 4 | Dennis Kelly | 79.26 | 65.79 | 84.08 | 305 | Packers |
| 42 | 5 | Cam Fleming | 79.04 | 66.29 | 83.38 | 285 | Broncos |
| 43 | 6 | Matt Pryor | 78.95 | 71.00 | 80.08 | 438 | Colts |
| 44 | 7 | D.J. Humphries | 78.89 | 67.80 | 82.12 | 1083 | Cardinals |
| 45 | 8 | James Hurst | 78.87 | 69.58 | 80.89 | 941 | Saints |
| 46 | 9 | Eric Fisher | 78.46 | 67.91 | 81.32 | 874 | Colts |
| 47 | 10 | Mike McGlinchey | 78.45 | 67.96 | 81.28 | 466 | 49ers |
| 48 | 11 | Trey Pipkins III | 78.16 | 63.56 | 83.73 | 173 | Chargers |
| 49 | 12 | Ty Nsekhe | 78.13 | 65.00 | 82.71 | 145 | Cowboys |
| 50 | 13 | Andrew Wylie | 77.95 | 65.27 | 82.24 | 527 | Chiefs |
| 51 | 14 | Riley Reiff | 77.90 | 66.64 | 81.24 | 711 | Bengals |
| 52 | 15 | Brandon Shell | 77.49 | 65.94 | 81.03 | 550 | Seahawks |
| 53 | 16 | Andre Dillard | 77.21 | 65.64 | 80.75 | 340 | Eagles |
| 54 | 17 | Chukwuma Okorafor | 77.05 | 63.60 | 81.85 | 1078 | Steelers |
| 55 | 18 | Terence Steele | 76.52 | 64.33 | 80.48 | 910 | Cowboys |
| 56 | 19 | Mike Remmers | 76.42 | 61.80 | 82.00 | 156 | Chiefs |
| 57 | 20 | Spencer Brown | 76.40 | 62.23 | 81.68 | 726 | Bills |
| 58 | 21 | Walker Little | 76.32 | 64.05 | 80.33 | 224 | Jaguars |
| 59 | 22 | Cam Robinson | 76.23 | 66.90 | 78.29 | 856 | Jaguars |
| 60 | 23 | Marcus Cannon | 76.20 | 64.48 | 79.84 | 213 | Texans |
| 61 | 24 | Jedrick Wills Jr. | 76.19 | 65.37 | 79.23 | 763 | Browns |
| 62 | 25 | Lucas Niang | 76.19 | 63.36 | 80.57 | 524 | Chiefs |
| 63 | 26 | Billy Turner | 76.11 | 65.62 | 78.93 | 810 | Packers |
| 64 | 27 | Alejandro Villanueva | 76.02 | 65.40 | 78.94 | 1205 | Ravens |
| 65 | 28 | Patrick Mekari | 75.89 | 65.37 | 78.74 | 762 | Ravens |
| 66 | 29 | Kaleb McGary | 75.56 | 62.80 | 79.90 | 986 | Falcons |
| 67 | 30 | Kelvin Beachum | 75.35 | 63.36 | 79.17 | 950 | Cardinals |
| 68 | 31 | Matt Peart | 75.13 | 61.96 | 79.74 | 421 | Giants |
| 69 | 32 | Germain Ifedi | 74.90 | 61.16 | 79.90 | 412 | Bears |
| 70 | 33 | Storm Norton | 74.82 | 60.30 | 80.33 | 1078 | Chargers |
| 71 | 34 | Yosh Nijman | 74.54 | 62.48 | 78.41 | 590 | Packers |
| 72 | 35 | Conor McDermott | 74.53 | 63.36 | 77.81 | 135 | Jets |
| 73 | 36 | Elijah Wilkinson | 74.38 | 62.00 | 78.47 | 120 | Bears |
| 74 | 37 | Josh Wells | 74.04 | 60.04 | 79.21 | 124 | Buccaneers |

### Starter (26 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 75 | 1 | Isaiah Prince | 73.91 | 58.75 | 79.85 | 384 | Bengals |
| 76 | 2 | Nate Solder | 73.88 | 60.29 | 78.78 | 927 | Giants |
| 77 | 3 | Chuma Edoga | 73.67 | 59.08 | 79.23 | 100 | Jets |
| 78 | 4 | Calvin Anderson | 73.29 | 65.22 | 74.51 | 172 | Broncos |
| 79 | 5 | Larry Borom | 73.20 | 61.12 | 77.09 | 633 | Bears |
| 80 | 6 | Laremy Tunsil | 73.09 | 60.57 | 77.27 | 262 | Texans |
| 81 | 7 | Geron Christian | 72.77 | 59.61 | 77.37 | 588 | Texans |
| 82 | 8 | Brady Christensen | 72.60 | 61.12 | 76.08 | 480 | Panthers |
| 83 | 9 | Jaylon Moore | 72.49 | 59.60 | 76.92 | 145 | 49ers |
| 84 | 10 | Jawaan Taylor | 72.36 | 60.40 | 76.16 | 1083 | Jaguars |
| 85 | 11 | James Hudson III | 72.21 | 58.50 | 77.18 | 303 | Browns |
| 86 | 12 | Justin Herron | 71.66 | 57.92 | 76.66 | 393 | Patriots |
| 87 | 13 | Dan Moore Jr. | 70.58 | 57.80 | 74.93 | 1079 | Steelers |
| 88 | 14 | Teven Jenkins | 70.19 | 54.96 | 76.18 | 160 | Bears |
| 89 | 15 | Cameron Erving | 70.18 | 56.91 | 74.86 | 589 | Panthers |
| 90 | 16 | Jake Curhan | 70.00 | 56.15 | 75.07 | 405 | Seahawks |
| 91 | 17 | Charlie Heck | 69.80 | 56.33 | 74.61 | 827 | Texans |
| 92 | 18 | Brandon Parker | 69.37 | 56.03 | 74.09 | 881 | Raiders |
| 93 | 19 | Tyre Phillips | 69.19 | 55.66 | 74.05 | 389 | Ravens |
| 94 | 20 | Jordan Mills | 68.94 | 54.22 | 74.58 | 221 | Saints |
| 95 | 21 | Korey Cunningham | 68.84 | 56.68 | 72.78 | 113 | Giants |
| 96 | 22 | Julie'n Davenport | 67.52 | 52.19 | 73.57 | 278 | Colts |
| 97 | 23 | Matt Nelson | 67.43 | 52.38 | 73.29 | 675 | Lions |
| 98 | 24 | Jesse Davis | 66.84 | 52.50 | 72.24 | 1063 | Dolphins |
| 99 | 25 | Rashod Hill | 64.75 | 49.81 | 70.54 | 342 | Vikings |
| 100 | 26 | Bobby Hart | 62.90 | 49.48 | 67.68 | 102 | Bills |

### Rotation/backup (0 players)

_None._

## TE — Tight End

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Mark Andrews | 87.20 | 91.50 | 80.16 | 669 | Ravens |
| 2 | 2 | George Kittle | 86.55 | 87.84 | 81.53 | 460 | 49ers |
| 3 | 3 | Kyle Pitts | 82.21 | 78.51 | 80.51 | 549 | Falcons |
| 4 | 4 | Travis Kelce | 82.20 | 81.90 | 78.23 | 672 | Chiefs |
| 5 | 5 | Rob Gronkowski | 81.47 | 77.16 | 80.17 | 422 | Buccaneers |
| 6 | 6 | Dallas Goedert | 81.16 | 85.26 | 74.26 | 412 | Eagles |

### Good (4 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Darren Waller | 76.17 | 67.24 | 77.95 | 427 | Raiders |
| 8 | 2 | Dalton Schultz | 75.87 | 77.74 | 70.45 | 634 | Cowboys |
| 9 | 3 | Marcedes Lewis | 75.62 | 73.82 | 72.66 | 206 | Packers |
| 10 | 4 | Zach Ertz | 74.61 | 66.55 | 75.82 | 554 | Cardinals |

### Starter (63 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Hunter Henry | 73.64 | 72.38 | 70.32 | 477 | Patriots |
| 12 | 2 | Pat Freiermuth | 73.58 | 70.18 | 71.68 | 452 | Steelers |
| 13 | 3 | David Njoku | 73.18 | 68.18 | 72.35 | 372 | Browns |
| 14 | 4 | Albert Okwuegbunam | 73.04 | 64.16 | 74.80 | 214 | Broncos |
| 15 | 5 | Mo Alie-Cox | 72.88 | 64.34 | 74.40 | 303 | Colts |
| 16 | 6 | Tyler Higbee | 72.44 | 66.86 | 72.00 | 559 | Rams |
| 17 | 7 | Maxx Williams | 72.10 | 67.54 | 70.98 | 117 | Cardinals |
| 18 | 8 | T.J. Hockenson | 72.08 | 67.23 | 71.14 | 444 | Lions |
| 19 | 9 | Jack Doyle | 72.05 | 69.15 | 69.82 | 319 | Colts |
| 20 | 10 | Gerald Everett | 71.51 | 62.75 | 73.18 | 408 | Seahawks |
| 21 | 11 | Zach Gentry | 71.30 | 64.06 | 71.96 | 222 | Steelers |
| 22 | 12 | Brevin Jordan | 71.30 | 63.26 | 72.49 | 161 | Texans |
| 23 | 13 | Jimmy Graham | 70.83 | 66.63 | 69.47 | 147 | Bears |
| 24 | 14 | Will Dissly | 70.82 | 62.37 | 72.29 | 257 | Seahawks |
| 25 | 15 | MyCole Pruitt | 70.57 | 66.17 | 69.34 | 190 | Titans |
| 26 | 16 | John Bates | 70.40 | 66.75 | 68.66 | 263 | Commanders |
| 27 | 17 | Donald Parham Jr. | 70.22 | 65.29 | 69.34 | 218 | Chargers |
| 28 | 18 | Blake Jarwin | 70.12 | 59.51 | 73.03 | 158 | Cowboys |
| 29 | 19 | Mike Gesicki | 70.05 | 68.40 | 66.99 | 587 | Dolphins |
| 30 | 20 | Stephen Anderson | 69.81 | 61.23 | 71.36 | 119 | Chargers |
| 31 | 21 | James O'Shaughnessy | 69.63 | 61.51 | 70.88 | 193 | Jaguars |
| 32 | 22 | Tyler Conklin | 69.52 | 66.38 | 67.45 | 598 | Vikings |
| 33 | 23 | Kylen Granson | 69.35 | 59.47 | 71.77 | 108 | Colts |
| 34 | 24 | Dan Arnold | 69.19 | 61.63 | 70.07 | 279 | Jaguars |
| 35 | 25 | Blake Bell | 69.17 | 65.48 | 67.47 | 121 | Chiefs |
| 36 | 26 | Ricky Seals-Jones | 69.02 | 61.58 | 69.81 | 313 | Commanders |
| 37 | 27 | Kyle Rudolph | 68.93 | 62.81 | 68.85 | 297 | Giants |
| 38 | 28 | Harrison Bryant | 68.65 | 62.41 | 68.64 | 167 | Browns |
| 39 | 29 | Jared Cook | 68.58 | 61.78 | 68.95 | 532 | Chargers |
| 40 | 30 | Jesse James | 68.54 | 59.73 | 70.25 | 136 | Bears |
| 41 | 31 | Austin Hooper | 68.30 | 63.58 | 67.28 | 376 | Browns |
| 42 | 32 | Noah Fant | 68.12 | 61.38 | 68.44 | 488 | Broncos |
| 43 | 33 | Nick Vannett | 68.06 | 59.13 | 69.85 | 112 | Saints |
| 44 | 34 | Chris Herndon | 67.80 | 57.04 | 70.81 | 104 | Vikings |
| 45 | 35 | Logan Thomas | 67.65 | 61.74 | 67.43 | 172 | Commanders |
| 46 | 36 | Adam Trautman | 67.60 | 61.92 | 67.22 | 333 | Saints |
| 47 | 37 | C.J. Uzomah | 67.43 | 61.74 | 67.05 | 498 | Bengals |
| 48 | 38 | Eric Ebron | 67.41 | 53.88 | 72.26 | 184 | Steelers |
| 49 | 39 | Foster Moreau | 67.23 | 59.04 | 68.52 | 424 | Raiders |
| 50 | 40 | Durham Smythe | 67.00 | 59.62 | 67.76 | 375 | Dolphins |
| 51 | 41 | O.J. Howard | 66.94 | 54.30 | 71.20 | 177 | Buccaneers |
| 52 | 42 | Adam Shaheen | 66.93 | 59.61 | 67.64 | 204 | Dolphins |
| 53 | 43 | Jonnu Smith | 66.79 | 59.62 | 67.40 | 196 | Patriots |
| 54 | 44 | Dawson Knox | 66.70 | 62.17 | 65.56 | 541 | Bills |
| 55 | 45 | Chris Manhertz | 66.67 | 61.33 | 66.07 | 173 | Jaguars |
| 56 | 46 | Cole Kmet | 66.15 | 63.20 | 63.95 | 583 | Bears |
| 57 | 47 | Eric Saubert | 66.06 | 63.21 | 63.79 | 112 | Broncos |
| 58 | 48 | Robert Tonyan | 65.61 | 56.95 | 67.21 | 211 | Packers |
| 59 | 49 | Lee Smith | 65.60 | 64.32 | 62.28 | 121 | Falcons |
| 60 | 50 | Drew Sample | 65.51 | 59.50 | 65.35 | 167 | Bengals |
| 61 | 51 | Cameron Brate | 65.16 | 57.54 | 66.07 | 323 | Buccaneers |
| 62 | 52 | Anthony Firkser | 64.86 | 56.57 | 66.22 | 287 | Titans |
| 63 | 53 | Hayden Hurst | 64.77 | 57.02 | 65.77 | 266 | Falcons |
| 64 | 54 | Tyler Kroft | 64.74 | 58.60 | 64.66 | 226 | Jets |
| 65 | 55 | Josiah Deguara | 64.62 | 58.82 | 64.32 | 228 | Packers |
| 66 | 56 | Evan Engram | 64.51 | 54.30 | 67.15 | 508 | Giants |
| 67 | 57 | Jacob Hollister | 64.13 | 56.39 | 65.12 | 106 | Jaguars |
| 68 | 58 | Ryan Griffin | 64.09 | 56.98 | 64.67 | 341 | Jets |
| 69 | 59 | Jordan Akins | 63.94 | 55.54 | 65.38 | 233 | Texans |
| 70 | 60 | Brock Wright | 63.85 | 57.25 | 64.09 | 118 | Lions |
| 71 | 61 | Geoff Swaim | 63.53 | 57.08 | 63.67 | 278 | Titans |
| 72 | 62 | Antony Auclair | 63.07 | 58.28 | 62.09 | 100 | Texans |
| 73 | 63 | Noah Gray | 62.43 | 55.17 | 63.10 | 157 | Chiefs |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 74 | 1 | Tommy Tremble | 61.92 | 56.73 | 61.21 | 306 | Panthers |
| 75 | 2 | Tommy Sweeney | 61.89 | 53.05 | 63.62 | 131 | Bills |
| 76 | 3 | Luke Farrell | 61.59 | 54.88 | 61.90 | 120 | Jaguars |
| 77 | 4 | Pharaoh Brown | 61.19 | 52.23 | 62.99 | 277 | Texans |
| 78 | 5 | Ian Thomas | 60.70 | 53.84 | 61.11 | 401 | Panthers |

## WR — Wide Receiver

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Davante Adams | 88.21 | 91.52 | 81.84 | 583 | Packers |
| 2 | 2 | Cooper Kupp | 88.15 | 92.14 | 81.32 | 649 | Rams |
| 3 | 3 | Justin Jefferson | 88.02 | 90.07 | 82.49 | 658 | Vikings |
| 4 | 4 | Ja'Marr Chase | 86.27 | 82.26 | 84.77 | 613 | Bengals |
| 5 | 5 | Deebo Samuel | 86.24 | 84.94 | 82.94 | 490 | 49ers |
| 6 | 6 | A.J. Brown | 83.70 | 80.68 | 81.54 | 367 | Titans |
| 7 | 7 | Tee Higgins | 82.47 | 82.75 | 78.12 | 512 | Bengals |
| 8 | 8 | Tyreek Hill | 81.89 | 84.74 | 75.82 | 629 | Chiefs |
| 9 | 9 | Deonte Harty | 81.86 | 75.83 | 81.72 | 214 | Saints |
| 10 | 10 | Stefon Diggs | 81.68 | 82.10 | 77.23 | 682 | Bills |
| 11 | 11 | Chris Godwin | 81.62 | 80.52 | 78.19 | 582 | Buccaneers |
| 12 | 12 | CeeDee Lamb | 81.21 | 82.82 | 75.97 | 582 | Cowboys |
| 13 | 13 | Tyler Lockett | 80.83 | 79.69 | 77.42 | 530 | Seahawks |
| 14 | 14 | D.K. Metcalf | 80.58 | 79.60 | 77.06 | 530 | Seahawks |
| 15 | 15 | Terry McLaurin | 80.54 | 77.97 | 78.09 | 621 | Commanders |
| 16 | 16 | Mike Williams | 80.46 | 77.27 | 78.42 | 620 | Chargers |

### Good (48 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | DeSean Jackson | 79.71 | 66.55 | 84.32 | 207 | Raiders |
| 18 | 2 | DJ Moore | 79.71 | 76.80 | 77.48 | 647 | Panthers |
| 19 | 3 | Lil'Jordan Humphrey | 79.49 | 65.15 | 84.89 | 142 | Saints |
| 20 | 4 | DeAndre Hopkins | 79.47 | 76.48 | 77.29 | 343 | Cardinals |
| 21 | 5 | Hunter Renfrow | 79.12 | 79.73 | 74.54 | 570 | Raiders |
| 22 | 6 | DeVonta Smith | 78.88 | 75.63 | 76.88 | 545 | Eagles |
| 23 | 7 | Brandin Cooks | 78.49 | 76.60 | 75.58 | 563 | Texans |
| 24 | 8 | Kendrick Bourne | 78.44 | 73.74 | 77.40 | 424 | Patriots |
| 25 | 9 | Julio Jones | 78.43 | 70.99 | 79.22 | 256 | Titans |
| 26 | 10 | Michael Pittman Jr. | 78.33 | 77.19 | 74.93 | 602 | Colts |
| 27 | 11 | Brandon Aiyuk | 77.94 | 73.41 | 76.79 | 517 | 49ers |
| 28 | 12 | Amon-Ra St. Brown | 77.75 | 77.56 | 73.71 | 541 | Lions |
| 29 | 13 | Mike Evans | 77.22 | 73.16 | 75.76 | 654 | Buccaneers |
| 30 | 14 | Quez Watkins | 77.17 | 68.52 | 78.77 | 435 | Eagles |
| 31 | 15 | Keenan Allen | 77.00 | 77.50 | 72.50 | 683 | Chargers |
| 32 | 16 | Darnell Mooney | 77.00 | 74.54 | 74.47 | 646 | Bears |
| 33 | 17 | Amari Cooper | 76.88 | 72.33 | 75.74 | 566 | Cowboys |
| 34 | 18 | Courtland Sutton | 76.87 | 70.75 | 76.78 | 574 | Broncos |
| 35 | 19 | Robert Woods | 76.52 | 73.02 | 74.69 | 338 | Rams |
| 36 | 20 | Kenny Golladay | 76.19 | 67.43 | 77.86 | 449 | Giants |
| 37 | 21 | Tyler Boyd | 76.19 | 72.02 | 74.81 | 562 | Bengals |
| 38 | 22 | Adam Thielen | 75.92 | 73.25 | 73.53 | 472 | Vikings |
| 39 | 23 | Tim Patrick | 75.89 | 70.48 | 75.33 | 526 | Broncos |
| 40 | 24 | Cedrick Wilson Jr. | 75.81 | 70.44 | 75.22 | 378 | Cowboys |
| 41 | 25 | Diontae Johnson | 75.73 | 74.19 | 72.59 | 659 | Steelers |
| 42 | 26 | Jakobi Meyers | 75.67 | 74.28 | 72.43 | 572 | Patriots |
| 43 | 27 | Quintez Cephus | 75.57 | 65.41 | 78.18 | 141 | Lions |
| 44 | 28 | Marquez Valdes-Scantling | 75.54 | 64.27 | 78.89 | 324 | Packers |
| 45 | 29 | Chase Claypool | 75.53 | 67.26 | 76.88 | 539 | Steelers |
| 46 | 30 | Gabe Davis | 75.52 | 70.16 | 74.92 | 363 | Bills |
| 47 | 31 | Randall Cobb | 75.47 | 69.38 | 75.37 | 267 | Packers |
| 48 | 32 | Jaylen Waddle | 75.43 | 77.64 | 69.79 | 613 | Dolphins |
| 49 | 33 | Corey Davis | 75.42 | 67.09 | 76.80 | 301 | Jets |
| 50 | 34 | DeVante Parker | 75.36 | 70.87 | 74.19 | 370 | Dolphins |
| 51 | 35 | Elijah Moore | 75.24 | 67.88 | 75.98 | 327 | Jets |
| 52 | 36 | Laquon Treadwell | 75.18 | 68.22 | 75.65 | 310 | Jaguars |
| 53 | 37 | Michael Gallup | 75.06 | 71.21 | 73.46 | 350 | Cowboys |
| 54 | 38 | Mecole Hardman Jr. | 74.94 | 67.03 | 76.05 | 451 | Chiefs |
| 55 | 39 | T.Y. Hilton | 74.92 | 67.68 | 75.58 | 238 | Colts |
| 56 | 40 | Donovan Peoples-Jones | 74.78 | 64.62 | 77.38 | 434 | Browns |
| 57 | 41 | John Ross | 74.74 | 63.38 | 78.15 | 154 | Giants |
| 58 | 42 | Christian Kirk | 74.73 | 71.90 | 72.45 | 579 | Cardinals |
| 59 | 43 | Bryan Edwards | 74.72 | 63.44 | 78.07 | 541 | Raiders |
| 60 | 44 | A.J. Green | 74.66 | 68.78 | 74.42 | 564 | Cardinals |
| 61 | 45 | Russell Gage | 74.43 | 73.23 | 71.06 | 406 | Falcons |
| 62 | 46 | Kadarius Toney | 74.40 | 67.32 | 74.95 | 216 | Giants |
| 63 | 47 | Jarvis Landry | 74.38 | 65.20 | 76.33 | 331 | Browns |
| 64 | 48 | Marquez Callaway | 74.27 | 67.94 | 74.33 | 481 | Saints |

### Starter (84 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 65 | 1 | Braxton Berrios | 73.87 | 69.05 | 72.92 | 276 | Jets |
| 66 | 2 | Brandon Zylstra | 73.63 | 62.05 | 77.19 | 192 | Panthers |
| 67 | 3 | Jerry Jeudy | 73.61 | 65.57 | 74.81 | 271 | Broncos |
| 68 | 4 | Marquise Brown | 73.58 | 68.58 | 72.74 | 657 | Ravens |
| 69 | 5 | Marquise Goodwin | 73.36 | 61.93 | 76.81 | 256 | Bears |
| 70 | 6 | Calvin Ridley | 73.25 | 63.10 | 75.85 | 209 | Falcons |
| 71 | 7 | Allen Robinson II | 73.21 | 65.93 | 73.90 | 386 | Bears |
| 72 | 8 | Nick Westbrook-Ikhine | 73.20 | 66.42 | 73.55 | 359 | Titans |
| 73 | 9 | Odell Beckham Jr. | 73.18 | 66.92 | 73.18 | 435 | Rams |
| 74 | 10 | Breshad Perriman | 73.18 | 60.53 | 77.44 | 121 | Buccaneers |
| 75 | 11 | DJ Chark Jr. | 73.01 | 63.86 | 74.95 | 119 | Jaguars |
| 76 | 12 | Emmanuel Sanders | 72.98 | 65.28 | 73.95 | 539 | Bills |
| 77 | 13 | Nelson Agholor | 72.51 | 65.23 | 73.19 | 434 | Patriots |
| 78 | 14 | Marvin Jones Jr. | 72.38 | 68.30 | 70.93 | 667 | Jaguars |
| 79 | 15 | Rondale Moore | 72.23 | 67.10 | 71.48 | 265 | Cardinals |
| 80 | 16 | K.J. Osborn | 72.21 | 64.42 | 73.23 | 536 | Vikings |
| 81 | 17 | Chris Moore | 71.96 | 63.81 | 73.22 | 139 | Texans |
| 82 | 18 | Nico Collins | 71.95 | 64.27 | 72.91 | 383 | Texans |
| 83 | 19 | Josh Reynolds | 71.92 | 63.92 | 73.08 | 291 | Lions |
| 84 | 20 | Zay Jones | 71.91 | 68.15 | 70.25 | 421 | Raiders |
| 85 | 21 | Cole Beasley | 71.76 | 66.19 | 71.30 | 531 | Bills |
| 86 | 22 | Danny Amendola | 71.74 | 62.16 | 73.96 | 162 | Texans |
| 87 | 23 | Sammy Watkins | 71.72 | 65.01 | 72.03 | 287 | Ravens |
| 88 | 24 | Allen Lazard | 71.58 | 64.57 | 72.09 | 455 | Packers |
| 89 | 25 | Byron Pringle | 71.54 | 65.26 | 71.56 | 432 | Chiefs |
| 90 | 26 | Marcus Johnson | 71.35 | 61.20 | 73.95 | 101 | Titans |
| 91 | 27 | Kendall Hinton | 71.32 | 63.29 | 72.51 | 169 | Broncos |
| 92 | 28 | Equanimeous St. Brown | 71.21 | 58.74 | 75.36 | 155 | Packers |
| 93 | 29 | Cam Sims | 70.99 | 61.37 | 73.23 | 182 | Commanders |
| 94 | 30 | KhaDarel Hodge | 70.78 | 64.06 | 71.10 | 166 | Lions |
| 95 | 31 | Jamison Crowder | 70.47 | 64.30 | 70.42 | 418 | Jets |
| 96 | 32 | Scott Miller | 70.28 | 57.92 | 74.35 | 104 | Buccaneers |
| 97 | 33 | James Proche II | 70.28 | 63.56 | 70.60 | 176 | Ravens |
| 98 | 34 | Keelan Cole Sr. | 70.09 | 61.10 | 71.92 | 367 | Jets |
| 99 | 35 | Chris Conley | 70.02 | 60.37 | 72.28 | 364 | Texans |
| 100 | 36 | Kalif Raymond | 70.00 | 60.43 | 72.22 | 496 | Lions |
| 101 | 37 | Van Jefferson | 69.96 | 59.81 | 72.56 | 580 | Rams |
| 102 | 38 | Olamide Zaccheaus | 69.91 | 62.65 | 70.59 | 400 | Falcons |
| 103 | 39 | Isaiah McKenzie | 69.84 | 64.48 | 69.25 | 150 | Bills |
| 104 | 40 | Collin Johnson | 69.81 | 61.06 | 71.47 | 141 | Giants |
| 105 | 41 | Rashod Bateman | 69.65 | 63.96 | 69.27 | 432 | Ravens |
| 106 | 42 | Sterling Shepard | 69.60 | 63.49 | 69.51 | 236 | Giants |
| 107 | 43 | Tre'Quan Smith | 69.54 | 61.89 | 70.47 | 323 | Saints |
| 108 | 44 | Ashton Dulin | 69.37 | 59.96 | 71.48 | 129 | Colts |
| 109 | 45 | Preston Williams | 69.00 | 59.68 | 71.04 | 108 | Dolphins |
| 110 | 46 | Joshua Palmer | 68.83 | 61.79 | 69.35 | 314 | Chargers |
| 111 | 47 | Laviska Shenault Jr. | 68.61 | 63.16 | 68.07 | 481 | Jaguars |
| 112 | 48 | N'Keal Harry | 68.44 | 64.34 | 67.01 | 150 | Patriots |
| 113 | 49 | Parris Campbell | 68.36 | 61.11 | 69.03 | 130 | Colts |
| 114 | 50 | Freddie Swain | 68.32 | 52.49 | 74.70 | 396 | Seahawks |
| 115 | 51 | Jalen Guyton | 68.32 | 57.83 | 71.15 | 461 | Chargers |
| 116 | 52 | DeAndre Carter | 68.30 | 62.41 | 68.06 | 281 | Commanders |
| 117 | 53 | Kenny Stills | 68.19 | 56.28 | 71.96 | 167 | Saints |
| 118 | 54 | Jauan Jennings | 67.89 | 63.28 | 66.79 | 211 | 49ers |
| 119 | 55 | Denzel Mims | 67.89 | 53.80 | 73.12 | 199 | Jets |
| 120 | 56 | Dyami Brown | 67.51 | 57.44 | 70.05 | 213 | Commanders |
| 121 | 57 | JuJu Smith-Schuster | 67.43 | 59.45 | 68.59 | 152 | Steelers |
| 122 | 58 | Antoine Wesley | 67.05 | 60.13 | 67.49 | 263 | Cardinals |
| 123 | 59 | Darius Slayton | 66.93 | 54.52 | 71.03 | 382 | Giants |
| 124 | 60 | Rashard Higgins | 66.88 | 56.35 | 69.73 | 313 | Browns |
| 125 | 61 | Anthony Schwartz | 66.81 | 58.68 | 68.07 | 169 | Browns |
| 126 | 62 | Jamal Agnew | 66.80 | 63.02 | 65.16 | 192 | Jaguars |
| 127 | 63 | James Washington | 66.68 | 53.01 | 71.62 | 357 | Steelers |
| 128 | 64 | Noah Brown | 66.68 | 60.31 | 66.76 | 173 | Cowboys |
| 129 | 65 | Albert Wilson | 66.43 | 59.47 | 66.91 | 238 | Dolphins |
| 130 | 66 | Damiere Byrd | 66.21 | 55.14 | 69.42 | 406 | Bears |
| 131 | 67 | Tavon Austin | 66.20 | 60.11 | 66.10 | 211 | Jaguars |
| 132 | 68 | Mack Hollins | 66.08 | 58.88 | 66.71 | 229 | Dolphins |
| 133 | 69 | Ben Skowronek | 65.92 | 58.88 | 66.45 | 103 | Rams |
| 134 | 70 | Mohamed Sanu | 65.90 | 62.15 | 64.23 | 184 | 49ers |
| 135 | 71 | Zach Pascal | 65.87 | 53.15 | 70.18 | 532 | Colts |
| 136 | 72 | Adam Humphries | 65.83 | 58.28 | 66.69 | 468 | Commanders |
| 137 | 73 | Dede Westbrook | 65.12 | 56.48 | 66.72 | 172 | Vikings |
| 138 | 74 | Jalen Reagor | 65.12 | 56.90 | 66.43 | 463 | Eagles |
| 139 | 75 | D'Wayne Eskridge | 65.09 | 59.30 | 64.79 | 112 | Seahawks |
| 140 | 76 | Trent Sherfield | 64.97 | 59.20 | 64.65 | 116 | 49ers |
| 141 | 77 | Tyler Johnson | 64.93 | 56.38 | 66.46 | 408 | Buccaneers |
| 142 | 78 | Tajae Sharpe | 64.90 | 55.97 | 66.69 | 341 | Falcons |
| 143 | 79 | Chester Rogers | 64.37 | 58.17 | 64.34 | 328 | Titans |
| 144 | 80 | Jeff Smith | 64.10 | 59.03 | 63.31 | 141 | Jets |
| 145 | 81 | Devin Duvernay | 63.95 | 56.63 | 64.67 | 405 | Ravens |
| 146 | 82 | Demarcus Robinson | 63.94 | 52.18 | 67.62 | 509 | Chiefs |
| 147 | 83 | Greg Ward | 63.82 | 57.57 | 63.82 | 150 | Eagles |
| 148 | 84 | Ray-Ray McCloud III | 62.36 | 57.62 | 61.35 | 366 | Steelers |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 149 | 1 | Terrace Marshall Jr. | 61.31 | 55.55 | 60.98 | 291 | Panthers |
| 150 | 2 | Trinity Benson | 60.83 | 52.95 | 61.92 | 191 | Lions |
