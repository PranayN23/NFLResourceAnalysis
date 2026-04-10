# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`, `--year-min 2020 --year-max 2025`, or `--all-analysis-years`.
- **Generated (UTC):** 2026-04-10 20:26:42Z
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
| 4 | 4 | Chase Roullier | 90.76 | 83.70 | 91.30 | 490 | Commanders |
| 5 | 5 | Frank Ragnow | 90.57 | 86.70 | 88.98 | 223 | Lions |
| 6 | 6 | Brian Allen | 88.88 | 80.20 | 90.50 | 903 | Rams |
| 7 | 7 | J.C. Tretter | 86.54 | 78.70 | 87.60 | 1039 | Browns |
| 8 | 8 | Matt Hennessy | 86.33 | 76.40 | 88.78 | 988 | Falcons |
| 9 | 9 | David Andrews | 86.03 | 78.00 | 87.22 | 1087 | Patriots |
| 10 | 10 | Ben Jones | 85.95 | 77.80 | 87.22 | 1160 | Titans |
| 11 | 11 | Connor McGovern | 84.80 | 75.90 | 86.56 | 973 | Jets |
| 12 | 12 | Bradley Bozeman | 82.05 | 73.60 | 83.51 | 1125 | Ravens |
| 13 | 13 | Ryan Jensen | 81.36 | 69.90 | 84.84 | 1151 | Buccaneers |
| 14 | 14 | Alex Mack | 80.09 | 70.40 | 82.38 | 1088 | 49ers |

### Good (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Mason Cole | 78.68 | 69.70 | 80.50 | 471 | Vikings |
| 16 | 2 | Ethan Pocic | 78.52 | 67.30 | 81.84 | 600 | Seahawks |
| 17 | 3 | Tyler Biadasz | 77.77 | 64.80 | 82.25 | 1202 | Cowboys |
| 18 | 4 | Evan Brown | 77.25 | 66.80 | 80.05 | 755 | Lions |
| 19 | 5 | Matt Paradis | 76.95 | 66.90 | 79.48 | 568 | Panthers |
| 20 | 6 | Erik McCoy | 76.92 | 63.60 | 81.64 | 746 | Saints |
| 21 | 7 | Tyler Larsen | 76.64 | 65.70 | 79.77 | 185 | Commanders |
| 22 | 8 | Danny Pinter | 76.62 | 74.70 | 73.73 | 226 | Colts |
| 23 | 9 | Justin Britt | 76.47 | 63.90 | 80.69 | 671 | Texans |
| 24 | 10 | Andre James | 75.21 | 64.10 | 78.45 | 1139 | Raiders |
| 25 | 11 | Lloyd Cushenberry III | 75.13 | 64.20 | 78.25 | 1039 | Broncos |
| 26 | 12 | Keith Ismael | 75.03 | 64.90 | 77.62 | 382 | Commanders |
| 27 | 13 | Brandon Linder | 74.99 | 62.90 | 78.89 | 552 | Jaguars |
| 28 | 14 | J.C. Hassenauer | 74.84 | 67.70 | 75.44 | 277 | Steelers |
| 29 | 15 | Billy Price | 74.76 | 62.30 | 78.90 | 985 | Giants |
| 30 | 16 | Mitch Morse | 74.34 | 63.40 | 77.47 | 1167 | Bills |

### Starter (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 31 | 1 | Scott Quessenberry | 73.56 | 64.30 | 75.56 | 115 | Chargers |
| 32 | 2 | Rodney Hudson | 73.24 | 60.90 | 77.30 | 798 | Cardinals |
| 33 | 3 | Garrett Bradbury | 73.09 | 60.20 | 77.51 | 883 | Vikings |
| 34 | 4 | Greg Mancz | 72.84 | 61.90 | 75.97 | 185 | Dolphins |
| 35 | 5 | Michael Deiter | 72.78 | 60.60 | 76.73 | 546 | Dolphins |
| 36 | 6 | Trystan Colon | 71.39 | 62.60 | 73.08 | 147 | Ravens |
| 37 | 7 | Tyler Shatley | 69.81 | 60.60 | 71.78 | 531 | Jaguars |
| 38 | 8 | Ryan Kelly | 69.53 | 56.90 | 73.78 | 907 | Colts |
| 39 | 9 | Sam Tecklenburg | 67.54 | 62.50 | 66.74 | 131 | Panthers |
| 40 | 10 | Kendrick Green | 66.91 | 52.40 | 72.42 | 975 | Steelers |
| 41 | 11 | Coleman Shelton | 66.91 | 57.70 | 68.88 | 216 | Rams |
| 42 | 12 | Josh Myers | 66.62 | 58.30 | 68.00 | 293 | Packers |
| 43 | 13 | Will Clapp | 66.53 | 50.70 | 72.92 | 133 | Saints |
| 44 | 14 | Sam Mustipher | 65.31 | 51.00 | 70.69 | 1121 | Bears |
| 45 | 15 | Trey Hopkins | 65.18 | 51.40 | 70.20 | 928 | Bengals |
| 46 | 16 | Trey Hill | 63.55 | 53.30 | 66.21 | 210 | Bengals |
| 47 | 17 | Kyle Fuller | 62.97 | 46.50 | 69.78 | 447 | Seahawks |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 48 | 1 | Ryan McCollum | 60.89 | 49.50 | 64.31 | 101 | Lions |
| 49 | 2 | Jimmy Morrissey | 60.73 | 45.00 | 67.05 | 258 | Texans |

## CB — Cornerback

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (17 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Jalen Ramsey | 88.40 | 86.30 | 87.26 | 1037 | Rams |
| 2 | 2 | J.C. Jackson | 86.99 | 83.00 | 85.48 | 944 | Patriots |
| 3 | 3 | A.J. Terrell | 86.88 | 85.60 | 84.96 | 1023 | Falcons |
| 4 | 4 | Darius Slay | 85.98 | 83.90 | 84.42 | 953 | Eagles |
| 5 | 5 | A.J. Green III | 84.56 | 90.30 | 92.84 | 176 | Browns |
| 6 | 6 | Chidobe Awuzie | 84.20 | 84.20 | 84.00 | 777 | Bengals |
| 7 | 7 | Rashad Fenton | 84.01 | 81.10 | 86.04 | 531 | Chiefs |
| 8 | 8 | Thomas Graham Jr. | 83.68 | 90.60 | 93.40 | 112 | Bears |
| 9 | 9 | Marshon Lattimore | 83.15 | 76.40 | 85.02 | 999 | Saints |
| 10 | 10 | Artie Burns | 83.14 | 85.10 | 85.35 | 254 | Bears |
| 11 | 11 | Kendall Fuller | 82.39 | 78.70 | 82.84 | 1004 | Commanders |
| 12 | 12 | Stephon Gilmore | 82.14 | 79.40 | 85.77 | 304 | Panthers |
| 13 | 13 | Casey Hayward Jr. | 81.56 | 75.00 | 82.40 | 1091 | Raiders |
| 14 | 14 | Jamel Dean | 81.38 | 76.70 | 83.18 | 685 | Buccaneers |
| 15 | 15 | Mike Hughes | 81.20 | 78.80 | 82.80 | 509 | Chiefs |
| 16 | 16 | Denzel Ward | 80.62 | 75.90 | 82.66 | 855 | Browns |
| 17 | 17 | Rasul Douglas | 80.51 | 78.00 | 81.10 | 680 | Packers |

### Good (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 18 | 1 | Jaire Alexander | 79.62 | 77.20 | 83.75 | 219 | Packers |
| 19 | 2 | Avonte Maddox | 79.59 | 76.20 | 80.88 | 729 | Eagles |
| 20 | 3 | Adoree' Jackson | 79.41 | 80.90 | 81.31 | 815 | Giants |
| 21 | 4 | D.J. Reed | 78.23 | 75.40 | 80.85 | 1002 | Seahawks |
| 22 | 5 | Taron Johnson | 78.04 | 75.20 | 77.09 | 877 | Bills |
| 23 | 6 | Xavien Howard | 78.01 | 71.10 | 81.24 | 1026 | Dolphins |
| 24 | 7 | Mike Hilton | 77.95 | 73.60 | 77.93 | 803 | Bengals |
| 25 | 8 | Ahkello Witherspoon | 77.07 | 78.80 | 79.12 | 368 | Steelers |
| 26 | 9 | Nate Hobbs | 77.01 | 76.70 | 75.02 | 837 | Raiders |
| 27 | 10 | Tavierre Thomas | 76.68 | 76.10 | 79.30 | 639 | Texans |
| 28 | 11 | Isaiah Oliver | 76.65 | 76.30 | 79.09 | 161 | Falcons |
| 29 | 12 | Shaquill Griffin | 76.24 | 71.10 | 78.63 | 872 | Jaguars |
| 30 | 13 | Trevon Diggs | 75.67 | 66.70 | 79.66 | 1013 | Cowboys |
| 31 | 14 | Levi Wallace | 75.61 | 70.00 | 76.44 | 993 | Bills |
| 32 | 15 | Robert Alford | 75.55 | 68.50 | 78.05 | 580 | Cardinals |
| 33 | 16 | Rock Ya-Sin | 75.28 | 72.40 | 76.14 | 592 | Colts |
| 34 | 17 | Nik Needham | 74.94 | 67.60 | 77.11 | 608 | Dolphins |
| 35 | 18 | Tre'Davious White | 74.71 | 69.00 | 78.12 | 630 | Bills |
| 36 | 19 | Nate Hairston | 74.69 | 76.00 | 79.12 | 148 | Broncos |
| 37 | 20 | Greg Newsome II | 74.44 | 70.60 | 77.73 | 691 | Browns |
| 38 | 21 | Emmanuel Moseley | 74.41 | 69.70 | 77.56 | 602 | 49ers |
| 39 | 22 | Eric Stokes | 74.41 | 67.60 | 75.76 | 934 | Packers |
| 40 | 23 | Pat Surtain II | 74.27 | 66.30 | 76.40 | 900 | Broncos |
| 41 | 24 | Carlton Davis III | 74.23 | 69.70 | 77.55 | 639 | Buccaneers |
| 42 | 25 | Pierre Desir | 74.15 | 71.10 | 78.15 | 308 | Buccaneers |
| 43 | 26 | Sidney Jones IV | 74.14 | 69.20 | 79.79 | 730 | Seahawks |
| 44 | 27 | Ross Cockrell | 74.11 | 69.60 | 76.89 | 475 | Buccaneers |

### Starter (77 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 45 | 1 | Anthony Brown | 73.99 | 69.00 | 76.96 | 1046 | Cowboys |
| 46 | 2 | Kevin King | 73.89 | 68.10 | 78.78 | 303 | Packers |
| 47 | 3 | Isaiah Rodgers | 73.84 | 71.80 | 75.33 | 525 | Colts |
| 48 | 4 | Greedy Williams | 73.44 | 68.80 | 74.18 | 591 | Browns |
| 49 | 5 | A.J. Bouye | 73.37 | 68.50 | 79.12 | 401 | Panthers |
| 50 | 6 | Bryce Hall | 73.26 | 64.50 | 78.06 | 1171 | Jets |
| 51 | 7 | James Bradberry | 73.15 | 65.00 | 74.94 | 1160 | Giants |
| 52 | 8 | Darious Williams | 73.04 | 64.40 | 78.18 | 924 | Rams |
| 53 | 9 | Kristian Fulton | 72.89 | 67.80 | 78.46 | 738 | Titans |
| 54 | 10 | Byron Jones | 72.84 | 63.50 | 76.22 | 976 | Dolphins |
| 55 | 11 | Janoris Jenkins | 72.50 | 64.60 | 76.22 | 862 | Titans |
| 56 | 12 | Joe Haden | 72.27 | 65.00 | 76.03 | 631 | Steelers |
| 57 | 13 | Cameron Dantzler | 71.94 | 67.60 | 75.07 | 685 | Vikings |
| 58 | 14 | Marlon Humphrey | 71.94 | 65.00 | 75.17 | 746 | Ravens |
| 59 | 15 | Jerry Jacobs | 71.93 | 66.70 | 77.14 | 535 | Lions |
| 60 | 16 | Bradley Roby | 71.91 | 67.20 | 75.48 | 395 | Saints |
| 61 | 17 | Bryce Callahan | 71.85 | 67.90 | 75.14 | 504 | Broncos |
| 62 | 18 | Kelvin Joseph | 71.68 | 73.00 | 80.41 | 165 | Cowboys |
| 63 | 19 | Joejuan Williams | 71.52 | 61.60 | 80.07 | 254 | Patriots |
| 64 | 20 | Byron Murphy Jr. | 71.29 | 64.30 | 72.59 | 967 | Cardinals |
| 65 | 21 | Charvarius Ward | 71.15 | 62.00 | 75.67 | 751 | Chiefs |
| 66 | 22 | Jalen Mills | 70.56 | 63.40 | 75.52 | 914 | Patriots |
| 67 | 23 | Trayvon Mullen | 70.51 | 64.30 | 76.36 | 229 | Raiders |
| 68 | 24 | Jaylon Johnson | 70.51 | 60.50 | 75.41 | 933 | Bears |
| 69 | 25 | Steven Nelson | 70.42 | 60.40 | 73.95 | 981 | Eagles |
| 70 | 26 | Kenny Moore II | 70.38 | 62.10 | 72.77 | 1062 | Colts |
| 71 | 27 | Jourdan Lewis | 70.19 | 62.10 | 72.44 | 801 | Cowboys |
| 72 | 28 | Jimmy Smith | 70.12 | 67.10 | 74.42 | 293 | Ravens |
| 73 | 29 | Tyson Campbell | 70.10 | 59.90 | 74.70 | 864 | Jaguars |
| 74 | 30 | Xavier Rhodes | 69.71 | 61.80 | 72.99 | 638 | Colts |
| 75 | 31 | Desmond Trufant | 69.58 | 64.70 | 76.69 | 234 | Raiders |
| 76 | 32 | Elijah Molden | 69.43 | 64.70 | 70.38 | 632 | Titans |
| 77 | 33 | Paulson Adebo | 69.14 | 61.40 | 70.13 | 851 | Saints |
| 78 | 34 | Antonio Hamilton Sr. | 68.95 | 64.70 | 75.14 | 313 | Cardinals |
| 79 | 35 | Eli Apple | 68.73 | 61.60 | 70.33 | 979 | Bengals |
| 80 | 36 | L'Jarius Sneed | 68.42 | 62.40 | 72.24 | 918 | Chiefs |
| 81 | 37 | Robert Rochell | 68.36 | 60.70 | 76.16 | 233 | Rams |
| 82 | 38 | Chris Harris Jr. | 67.97 | 60.40 | 72.50 | 747 | Chargers |
| 83 | 39 | Tavon Young | 67.92 | 62.60 | 71.66 | 550 | Ravens |
| 84 | 40 | Amani Oruwariye | 67.92 | 60.30 | 72.18 | 937 | Lions |
| 85 | 41 | William Jackson III | 67.88 | 59.50 | 72.80 | 748 | Commanders |
| 86 | 42 | Cameron Sutton | 67.86 | 58.90 | 70.15 | 1089 | Steelers |
| 87 | 43 | Patrick Peterson | 67.86 | 61.00 | 71.48 | 884 | Vikings |
| 88 | 44 | Ronald Darby | 67.71 | 58.70 | 73.53 | 675 | Broncos |
| 89 | 45 | Dont'e Deayon | 67.66 | 67.70 | 70.34 | 461 | Rams |
| 90 | 46 | Buster Skrine | 67.43 | 65.10 | 71.47 | 218 | Titans |
| 91 | 47 | Jarren Williams | 67.40 | 67.90 | 79.16 | 194 | Giants |
| 92 | 48 | Sean Murphy-Bunting | 66.86 | 61.00 | 70.73 | 462 | Buccaneers |
| 93 | 49 | K'Waun Williams | 66.56 | 60.80 | 70.42 | 647 | 49ers |
| 94 | 50 | Asante Samuel Jr. | 66.12 | 57.40 | 72.67 | 693 | Chargers |
| 95 | 51 | Chandon Sullivan | 65.96 | 57.80 | 68.06 | 826 | Packers |
| 96 | 52 | Myles Bryant | 65.86 | 57.50 | 70.33 | 405 | Patriots |
| 97 | 53 | Dontae Johnson | 65.77 | 63.00 | 70.87 | 262 | 49ers |
| 98 | 54 | Danny Johnson | 65.74 | 65.20 | 72.48 | 336 | Commanders |
| 99 | 55 | Darnay Holmes | 65.63 | 60.40 | 70.18 | 282 | Giants |
| 100 | 56 | Michael Davis | 65.52 | 54.40 | 71.07 | 851 | Chargers |
| 101 | 57 | A.J. Parker | 65.49 | 57.10 | 70.84 | 556 | Lions |
| 102 | 58 | Donte Jackson | 65.14 | 56.10 | 70.91 | 717 | Panthers |
| 103 | 59 | Dane Jackson | 65.13 | 54.70 | 71.83 | 482 | Bills |
| 104 | 60 | Jaycee Horn | 64.75 | 67.40 | 78.82 | 142 | Panthers |
| 105 | 61 | Darryl Roberts | 64.71 | 58.30 | 74.38 | 203 | Commanders |
| 106 | 62 | Fabian Moreau | 64.59 | 55.20 | 69.25 | 1036 | Falcons |
| 107 | 63 | James Pierre | 64.41 | 58.40 | 71.62 | 415 | Steelers |
| 108 | 64 | Rashaan Melvin | 64.35 | 57.00 | 69.87 | 247 | Panthers |
| 109 | 65 | T.J. Carrie | 64.32 | 55.20 | 70.99 | 142 | Colts |
| 110 | 66 | David Long Jr. | 64.18 | 58.40 | 68.57 | 517 | Rams |
| 111 | 67 | Justin Coleman | 64.11 | 55.30 | 68.37 | 371 | Dolphins |
| 112 | 68 | Troy Hill | 63.71 | 53.50 | 69.43 | 533 | Browns |
| 113 | 69 | Michael Carter II | 63.66 | 56.00 | 66.56 | 777 | Jets |
| 114 | 70 | Javelin Guidry | 63.66 | 56.80 | 67.58 | 487 | Jets |
| 115 | 71 | Aaron Robinson | 63.50 | 58.40 | 70.59 | 268 | Giants |
| 116 | 72 | Trae Waynes | 63.26 | 58.10 | 68.84 | 243 | Bengals |
| 117 | 73 | Anthony Averett | 63.23 | 55.00 | 69.35 | 807 | Ravens |
| 118 | 74 | Keith Taylor Jr. | 62.87 | 53.80 | 65.74 | 448 | Panthers |
| 119 | 75 | Isaiah Dunn | 62.83 | 60.00 | 74.33 | 115 | Jets |
| 120 | 76 | Terrance Mitchell | 62.74 | 51.70 | 69.07 | 796 | Texans |
| 121 | 77 | John Reid | 62.68 | 63.00 | 69.16 | 132 | Seahawks |

### Rotation/backup (38 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 122 | 1 | Vernon Hargreaves III | 61.36 | 54.20 | 65.60 | 390 | Bengals |
| 123 | 2 | Duke Shelley | 61.23 | 58.20 | 67.50 | 409 | Bears |
| 124 | 3 | Jonathan Jones | 61.09 | 51.50 | 68.92 | 224 | Patriots |
| 125 | 4 | Zech McPhearson | 61.02 | 60.00 | 68.83 | 179 | Eagles |
| 126 | 5 | Tevaughn Campbell | 60.47 | 52.40 | 64.63 | 678 | Chargers |
| 127 | 6 | Richard Sherman | 60.09 | 51.80 | 70.98 | 141 | Buccaneers |
| 128 | 7 | Chris Jackson | 59.89 | 52.20 | 67.27 | 386 | Titans |
| 129 | 8 | Josh Norman | 59.81 | 47.40 | 68.94 | 765 | 49ers |
| 130 | 9 | Brandin Echols | 59.70 | 48.00 | 66.26 | 762 | Jets |
| 131 | 10 | Desmond King II | 59.58 | 47.70 | 64.35 | 929 | Texans |
| 132 | 11 | Ifeatu Melifonwu | 59.32 | 60.70 | 65.54 | 242 | Lions |
| 133 | 12 | Kyle Fuller | 59.13 | 40.80 | 67.66 | 719 | Broncos |
| 134 | 13 | Tre Flowers | 59.02 | 49.90 | 64.67 | 391 | Bengals |
| 135 | 14 | Nevin Lawson | 58.38 | 48.90 | 65.83 | 325 | Jaguars |
| 136 | 15 | Tre Brown | 58.13 | 60.90 | 68.39 | 255 | Seahawks |
| 137 | 16 | Mackensie Alexander | 57.17 | 41.10 | 65.77 | 689 | Vikings |
| 138 | 17 | Dee Delaney | 56.83 | 53.40 | 66.32 | 213 | Buccaneers |
| 139 | 18 | DeAndre Baker | 56.79 | 56.90 | 58.68 | 212 | Chiefs |
| 140 | 19 | Brandon Facyson | 56.46 | 45.70 | 66.68 | 602 | Raiders |
| 141 | 20 | Chris Westry | 56.26 | 50.00 | 68.11 | 183 | Ravens |
| 142 | 21 | Arthur Maulet | 56.26 | 43.90 | 62.56 | 380 | Steelers |
| 143 | 22 | Deommodore Lenoir | 56.13 | 58.80 | 61.48 | 238 | 49ers |
| 144 | 23 | Tremon Smith | 56.06 | 54.40 | 64.66 | 179 | Texans |
| 145 | 24 | Greg Mabin | 55.99 | 50.40 | 67.37 | 171 | Titans |
| 146 | 25 | Benjamin St-Juste | 55.98 | 49.20 | 67.63 | 318 | Commanders |
| 147 | 26 | Blessuan Austin | 55.64 | 51.90 | 63.78 | 149 | Seahawks |
| 148 | 27 | Ugo Amadi | 55.20 | 41.20 | 63.29 | 692 | Seahawks |
| 149 | 28 | Darren Hall | 53.79 | 47.10 | 59.96 | 283 | Falcons |
| 150 | 29 | Ambry Thomas | 52.92 | 47.10 | 60.49 | 334 | 49ers |
| 151 | 30 | CJ Henderson | 52.05 | 46.80 | 57.56 | 390 | Panthers |
| 152 | 31 | Kindle Vildor | 51.44 | 45.60 | 56.08 | 822 | Bears |
| 153 | 32 | Kevon Seymour | 50.56 | 37.90 | 64.60 | 247 | Ravens |
| 154 | 33 | Chris Claybrooks | 49.46 | 40.50 | 57.74 | 199 | Jaguars |
| 155 | 34 | Daryl Worley | 48.66 | 31.10 | 65.15 | 100 | Ravens |
| 156 | 35 | Tre Herndon | 47.37 | 35.70 | 56.59 | 207 | Jaguars |
| 157 | 36 | Kris Boyd | 45.00 | 37.40 | 53.40 | 160 | Vikings |
| 158 | 37 | Amik Robertson | 45.00 | 39.90 | 49.37 | 137 | Raiders |
| 159 | 38 | Xavier Crawford | 45.00 | 45.40 | 49.86 | 140 | Bears |

## DI — Defensive Interior

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Aaron Donald | 92.53 | 87.59 | 91.66 | 1040 | Rams |
| 2 | 2 | Zach Sieler | 87.98 | 84.44 | 88.05 | 518 | Dolphins |
| 3 | 3 | Quinnen Williams | 87.81 | 88.13 | 85.98 | 613 | Jets |
| 4 | 4 | Jonathan Allen | 85.24 | 86.90 | 80.16 | 772 | Commanders |
| 5 | 5 | DeForest Buckner | 84.99 | 87.19 | 79.67 | 843 | Colts |
| 6 | 6 | Leonard Williams | 84.75 | 87.46 | 78.97 | 890 | Giants |
| 7 | 7 | Cameron Heyward | 83.71 | 81.63 | 81.24 | 955 | Steelers |
| 8 | 8 | Christian Wilkins | 83.46 | 86.00 | 78.23 | 734 | Dolphins |
| 9 | 9 | Grady Jarrett | 82.83 | 80.02 | 80.53 | 864 | Falcons |
| 10 | 10 | Chris Jones | 82.72 | 88.42 | 77.16 | 628 | Chiefs |
| 11 | 11 | Calais Campbell | 81.89 | 74.42 | 84.94 | 615 | Ravens |
| 12 | 12 | Jeffery Simmons | 81.62 | 82.90 | 78.36 | 933 | Titans |
| 13 | 13 | Kenny Clark | 81.49 | 79.85 | 79.85 | 781 | Packers |
| 14 | 14 | Ed Oliver | 80.79 | 71.22 | 83.00 | 622 | Bills |
| 15 | 15 | Dexter Lawrence | 80.56 | 81.19 | 76.46 | 759 | Giants |
| 16 | 16 | Dalvin Tomlinson | 80.16 | 82.25 | 75.08 | 641 | Vikings |

### Good (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 17 | 1 | Vita Vea | 79.97 | 86.13 | 75.63 | 608 | Buccaneers |
| 18 | 2 | B.J. Hill | 79.60 | 76.76 | 77.81 | 502 | Bengals |
| 19 | 3 | Poona Ford | 79.31 | 76.51 | 77.21 | 802 | Seahawks |
| 20 | 4 | Michael Pierce | 78.87 | 79.54 | 79.71 | 251 | Vikings |
| 21 | 5 | Tim Settle | 78.24 | 70.77 | 79.75 | 210 | Commanders |
| 22 | 6 | Fletcher Cox | 78.17 | 73.87 | 77.67 | 747 | Eagles |
| 23 | 7 | Akiem Hicks | 78.16 | 76.41 | 81.70 | 304 | Bears |
| 24 | 8 | David Onyemata | 77.01 | 78.83 | 75.10 | 430 | Saints |
| 25 | 9 | Daron Payne | 76.88 | 72.62 | 75.76 | 837 | Commanders |
| 26 | 10 | J.J. Watt | 76.84 | 59.79 | 96.59 | 341 | Cardinals |
| 27 | 11 | DJ Reader | 76.74 | 81.26 | 74.20 | 590 | Bengals |
| 28 | 12 | Derrick Brown | 76.09 | 73.53 | 74.24 | 631 | Panthers |
| 29 | 13 | Shelby Harris | 76.02 | 65.58 | 80.87 | 564 | Broncos |
| 30 | 14 | Folorunso Fatukasi | 75.63 | 67.39 | 78.68 | 558 | Jets |
| 31 | 15 | D.J. Jones | 75.43 | 66.33 | 79.00 | 550 | 49ers |
| 32 | 16 | Grover Stewart | 74.49 | 72.50 | 71.65 | 643 | Colts |
| 33 | 17 | Matt Ioannidis | 74.35 | 70.54 | 77.28 | 608 | Commanders |
| 34 | 18 | Javon Hargrave | 74.34 | 60.69 | 80.07 | 727 | Eagles |
| 35 | 19 | Sheldon Richardson | 74.32 | 61.06 | 79.00 | 688 | Vikings |

### Starter (69 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 36 | 1 | Greg Gaines | 73.87 | 67.34 | 75.31 | 780 | Rams |
| 37 | 2 | Christian Barmore | 73.75 | 59.99 | 78.75 | 598 | Patriots |
| 38 | 3 | Sebastian Joseph-Day | 73.13 | 63.81 | 80.08 | 340 | Rams |
| 39 | 4 | Dre'Mont Jones | 72.65 | 63.61 | 76.36 | 614 | Broncos |
| 40 | 5 | Bilal Nichols | 72.61 | 61.28 | 76.63 | 679 | Bears |
| 41 | 6 | Shy Tuttle | 71.97 | 64.49 | 73.73 | 494 | Saints |
| 42 | 7 | Arik Armstead | 71.50 | 60.89 | 76.91 | 820 | 49ers |
| 43 | 8 | Linval Joseph | 71.48 | 60.30 | 76.87 | 550 | Chargers |
| 44 | 9 | Mario Edwards Jr. | 71.27 | 60.42 | 77.30 | 212 | Bears |
| 45 | 10 | Lawrence Guy Sr. | 71.21 | 54.85 | 78.58 | 532 | Patriots |
| 46 | 11 | Larry Ogunjobi | 70.87 | 52.82 | 79.75 | 724 | Bengals |
| 47 | 12 | A'Shawn Robinson | 69.95 | 61.16 | 74.78 | 517 | Rams |
| 48 | 13 | William Gholston | 69.88 | 54.19 | 76.18 | 507 | Buccaneers |
| 49 | 14 | Harrison Phillips | 69.73 | 66.71 | 73.01 | 473 | Bills |
| 50 | 15 | Dean Lowry | 69.40 | 59.89 | 71.57 | 673 | Packers |
| 51 | 16 | Chris Wormley | 69.10 | 67.87 | 67.67 | 729 | Steelers |
| 52 | 17 | Alim McNeill | 68.95 | 57.17 | 72.64 | 422 | Lions |
| 53 | 18 | Darius Philon | 68.84 | 57.31 | 75.29 | 277 | Raiders |
| 54 | 19 | Morgan Fox | 68.52 | 53.09 | 74.64 | 561 | Panthers |
| 55 | 20 | Al Woods | 68.52 | 58.28 | 72.72 | 620 | Seahawks |
| 56 | 21 | DaQuan Jones | 68.46 | 61.68 | 68.82 | 640 | Panthers |
| 57 | 22 | Steve McLendon | 68.20 | 50.99 | 78.27 | 252 | Buccaneers |
| 58 | 23 | Taven Bryan | 68.10 | 58.76 | 71.15 | 301 | Jaguars |
| 59 | 24 | Austin Johnson | 68.00 | 56.97 | 71.19 | 665 | Giants |
| 60 | 25 | Malcom Brown | 67.46 | 51.26 | 75.02 | 678 | Jaguars |
| 61 | 26 | Marquise Copeland | 67.30 | 62.25 | 74.35 | 108 | Rams |
| 62 | 27 | Ndamukong Suh | 67.26 | 44.94 | 77.98 | 718 | Buccaneers |
| 63 | 28 | Milton Williams | 67.09 | 47.79 | 75.79 | 456 | Eagles |
| 64 | 29 | Roy Robertson-Harris | 67.03 | 54.69 | 75.28 | 547 | Jaguars |
| 65 | 30 | Davon Godchaux | 66.56 | 57.30 | 72.00 | 640 | Patriots |
| 66 | 31 | Naquan Jones | 66.46 | 46.04 | 79.82 | 328 | Titans |
| 67 | 32 | Kevin Givens | 66.08 | 53.08 | 76.60 | 230 | 49ers |
| 68 | 33 | Tyler Lancaster | 65.77 | 51.79 | 71.73 | 318 | Packers |
| 69 | 34 | Michael Brockers | 65.71 | 48.56 | 73.78 | 622 | Lions |
| 70 | 35 | Malik Jackson | 65.51 | 44.43 | 79.33 | 646 | Browns |
| 71 | 36 | Christian Ringo | 65.51 | 55.91 | 74.42 | 315 | Saints |
| 72 | 37 | Eddie Goldman | 65.39 | 47.16 | 75.16 | 336 | Bears |
| 73 | 38 | Quinton Jefferson | 65.39 | 49.77 | 72.05 | 686 | Raiders |
| 74 | 39 | Brandon Williams | 65.23 | 49.12 | 75.12 | 447 | Ravens |
| 75 | 40 | Roy Lopez | 65.14 | 49.33 | 72.49 | 502 | Texans |
| 76 | 41 | Derrick Nnadi | 65.11 | 53.81 | 68.79 | 449 | Chiefs |
| 77 | 42 | Jarran Reed | 65.09 | 48.22 | 73.42 | 711 | Chiefs |
| 78 | 43 | Danny Shelton | 65.03 | 51.03 | 73.42 | 256 | Giants |
| 79 | 44 | Jordan Phillips | 65.00 | 55.95 | 72.98 | 284 | Cardinals |
| 80 | 45 | Adam Butler | 64.78 | 50.50 | 70.45 | 592 | Dolphins |
| 81 | 46 | Michael Dogbe | 64.63 | 49.70 | 74.15 | 263 | Cardinals |
| 82 | 47 | Johnathan Hankins | 64.62 | 47.55 | 73.30 | 568 | Raiders |
| 83 | 48 | Christian Covington | 64.58 | 49.72 | 70.80 | 523 | Chargers |
| 84 | 49 | Justin Zimmer | 64.55 | 55.39 | 76.05 | 161 | Bills |
| 85 | 50 | Mike Purcell | 64.55 | 52.42 | 74.18 | 361 | Broncos |
| 86 | 51 | Armon Watts | 64.48 | 54.90 | 68.59 | 669 | Vikings |
| 87 | 52 | Neville Gallimore | 64.40 | 49.83 | 78.08 | 164 | Cowboys |
| 88 | 53 | Anthony Rush | 64.34 | 54.74 | 72.38 | 266 | Falcons |
| 89 | 54 | Osa Odighizuwa | 64.33 | 45.79 | 73.50 | 614 | Cowboys |
| 90 | 55 | John Jenkins | 64.21 | 56.87 | 71.40 | 176 | Dolphins |
| 91 | 56 | Carlos Watkins | 64.13 | 51.50 | 70.61 | 437 | Cowboys |
| 92 | 57 | Justin Jones | 63.92 | 55.11 | 70.35 | 486 | Chargers |
| 93 | 58 | DeShawn Williams | 63.89 | 53.73 | 70.60 | 386 | Broncos |
| 94 | 59 | Sheldon Rankins | 63.77 | 50.41 | 71.49 | 643 | Jets |
| 95 | 60 | Tershawn Wharton | 63.55 | 48.37 | 69.51 | 501 | Chiefs |
| 96 | 61 | Hassan Ridgeway | 63.51 | 50.17 | 72.92 | 373 | Eagles |
| 97 | 62 | DaVon Hamilton | 63.06 | 51.57 | 69.12 | 443 | Jaguars |
| 98 | 63 | James Lynch | 62.78 | 48.65 | 73.22 | 304 | Vikings |
| 99 | 64 | Josh Tupou | 62.63 | 54.70 | 65.83 | 410 | Bengals |
| 100 | 65 | Adam Gotsis | 62.46 | 42.84 | 74.05 | 443 | Jaguars |
| 101 | 66 | Taylor Stallworth | 62.31 | 52.12 | 67.92 | 331 | Colts |
| 102 | 67 | Maliek Collins | 62.19 | 53.64 | 65.95 | 628 | Texans |
| 103 | 68 | Kentavius Street | 62.13 | 46.01 | 71.72 | 352 | 49ers |
| 104 | 69 | Jonathan Bullard | 62.06 | 52.59 | 72.71 | 224 | Falcons |

### Rotation/backup (51 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 105 | 1 | Corey Peters | 61.80 | 47.94 | 70.53 | 362 | Cardinals |
| 106 | 2 | Raekwon Davis | 61.77 | 51.84 | 66.06 | 424 | Dolphins |
| 107 | 3 | Khyiris Tonga | 61.69 | 49.55 | 67.59 | 217 | Bears |
| 108 | 4 | Mike Pennel | 61.39 | 46.48 | 72.88 | 249 | Falcons |
| 109 | 5 | Nathan Shepherd | 61.22 | 51.67 | 65.51 | 495 | Jets |
| 110 | 6 | Robert Nkemdiche | 61.18 | 48.82 | 71.88 | 230 | Seahawks |
| 111 | 7 | Kyle Peko | 61.11 | 52.34 | 72.40 | 157 | Titans |
| 112 | 8 | Angelo Blackson | 60.96 | 45.15 | 67.54 | 584 | Bears |
| 113 | 9 | Montravius Adams | 60.70 | 52.72 | 68.21 | 286 | Steelers |
| 114 | 10 | Solomon Thomas | 60.60 | 47.74 | 70.47 | 554 | Raiders |
| 115 | 11 | Bravvion Roy | 60.21 | 46.80 | 65.37 | 341 | Panthers |
| 116 | 12 | John Penisini | 60.20 | 48.74 | 64.29 | 276 | Lions |
| 117 | 13 | Malik McDowell | 60.02 | 41.05 | 70.46 | 645 | Browns |
| 118 | 14 | Ross Blacklock | 59.96 | 48.92 | 65.39 | 457 | Texans |
| 119 | 15 | Brent Urban | 59.93 | 45.32 | 71.52 | 160 | Cowboys |
| 120 | 16 | Vernon Butler | 59.93 | 46.59 | 69.13 | 285 | Bills |
| 121 | 17 | Henry Mondeaux | 59.50 | 46.76 | 67.40 | 244 | Steelers |
| 122 | 18 | Javon Kinlaw | 59.50 | 50.95 | 69.79 | 149 | 49ers |
| 123 | 19 | Trysten Hill | 59.34 | 53.96 | 69.46 | 171 | Cowboys |
| 124 | 20 | Rakeem Nunez-Roches | 59.34 | 46.62 | 64.14 | 415 | Buccaneers |
| 125 | 21 | Leki Fotu | 59.07 | 44.70 | 66.44 | 371 | Cardinals |
| 126 | 22 | Sheldon Day | 59.03 | 46.71 | 68.81 | 233 | Browns |
| 127 | 23 | L.J. Collier | 58.98 | 48.78 | 65.90 | 219 | Seahawks |
| 128 | 24 | Jerry Tillery | 58.83 | 43.77 | 65.70 | 858 | Chargers |
| 129 | 25 | Levi Onwuzurike | 58.67 | 45.71 | 64.13 | 396 | Lions |
| 130 | 26 | Tyeler Davison | 58.62 | 46.77 | 64.80 | 358 | Falcons |
| 131 | 27 | Joe Gaziano | 58.38 | 45.62 | 65.66 | 214 | Chargers |
| 132 | 28 | Bryan Mone | 58.16 | 52.28 | 63.76 | 395 | Seahawks |
| 133 | 29 | Star Lotulelei | 57.98 | 37.91 | 70.12 | 317 | Bills |
| 134 | 30 | Carl Davis Jr. | 57.80 | 47.90 | 67.00 | 277 | Patriots |
| 135 | 31 | Jaleel Johnson | 57.57 | 43.00 | 65.57 | 322 | Texans |
| 136 | 32 | Breiden Fehoko | 57.08 | 49.78 | 66.60 | 121 | Chargers |
| 137 | 33 | Isaiahh Loudermilk | 56.27 | 44.28 | 62.06 | 288 | Steelers |
| 138 | 34 | Justin Ellis | 56.05 | 42.30 | 64.48 | 381 | Ravens |
| 139 | 35 | Margus Hunt | 56.01 | 37.14 | 68.79 | 151 | Bears |
| 140 | 36 | Shamar Stephen | 55.65 | 42.33 | 60.56 | 393 | Broncos |
| 141 | 37 | Rashard Lawrence | 55.40 | 50.93 | 60.63 | 219 | Cardinals |
| 142 | 38 | Ta'Quon Graham | 55.12 | 46.62 | 60.54 | 309 | Falcons |
| 143 | 39 | Broderick Washington | 54.57 | 48.06 | 59.71 | 293 | Ravens |
| 144 | 40 | Khalen Saunders | 54.46 | 48.77 | 63.88 | 144 | Chiefs |
| 145 | 41 | Tommy Togiai | 54.18 | 46.92 | 68.64 | 125 | Browns |
| 146 | 42 | Teair Tart | 53.27 | 48.89 | 59.22 | 344 | Titans |
| 147 | 43 | Larrell Murchison | 52.55 | 49.36 | 56.52 | 200 | Titans |
| 148 | 44 | Raymond Johnson III | 52.54 | 44.81 | 55.49 | 166 | Giants |
| 149 | 45 | Jordan Elliott | 52.42 | 44.60 | 54.09 | 464 | Browns |
| 150 | 46 | Marlon Davidson | 52.17 | 50.70 | 55.78 | 270 | Falcons |
| 151 | 47 | Michael Hoecht | 51.67 | 45.70 | 56.39 | 110 | Rams |
| 152 | 48 | Malcolm Roach | 50.99 | 44.81 | 59.81 | 194 | Saints |
| 153 | 49 | Albert Huggins | 49.38 | 44.14 | 59.47 | 219 | Saints |
| 154 | 50 | Justin Hamilton | 48.72 | 42.88 | 56.75 | 249 | Broncos |
| 155 | 51 | Quinton Bohanna | 46.91 | 40.38 | 50.03 | 222 | Cowboys |

## ED — Edge

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | T.J. Watt | 92.21 | 93.60 | 88.42 | 758 | Steelers |
| 2 | 2 | Joey Bosa | 92.17 | 96.19 | 87.05 | 847 | Chargers |
| 3 | 3 | Nick Bosa | 90.05 | 95.77 | 86.43 | 840 | 49ers |
| 4 | 4 | Myles Garrett | 89.24 | 95.35 | 82.88 | 866 | Browns |
| 5 | 5 | Rashan Gary | 88.81 | 91.30 | 83.79 | 681 | Packers |
| 6 | 6 | Von Miller | 87.51 | 81.71 | 88.40 | 762 | Rams |
| 7 | 7 | Khalil Mack | 84.75 | 87.83 | 83.43 | 315 | Bears |
| 8 | 8 | Shaquil Barrett | 84.01 | 80.28 | 83.63 | 768 | Buccaneers |
| 9 | 9 | DeMarcus Lawrence | 83.79 | 93.19 | 78.26 | 271 | Cowboys |
| 10 | 10 | Danielle Hunter | 83.44 | 83.70 | 84.00 | 384 | Vikings |
| 11 | 11 | Maxx Crosby | 83.17 | 88.43 | 75.50 | 926 | Raiders |
| 12 | 12 | Marcus Davenport | 83.09 | 92.07 | 78.07 | 437 | Saints |
| 13 | 13 | Cameron Jordan | 80.82 | 84.32 | 74.80 | 831 | Saints |

### Good (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 14 | 1 | Trey Hendrickson | 78.99 | 71.88 | 80.99 | 717 | Bengals |
| 15 | 2 | Andrew Van Ginkel | 78.53 | 68.58 | 80.99 | 801 | Dolphins |
| 16 | 3 | Justin Houston | 77.34 | 61.31 | 84.84 | 577 | Ravens |
| 17 | 4 | Montez Sweat | 76.72 | 79.81 | 73.93 | 483 | Commanders |
| 18 | 5 | Brian Burns | 76.58 | 67.32 | 78.91 | 838 | Panthers |
| 19 | 6 | Jadeveon Clowney | 74.24 | 82.28 | 69.32 | 677 | Browns |
| 20 | 7 | Matthew Judon | 74.20 | 56.77 | 82.28 | 878 | Patriots |

### Starter (70 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 21 | 1 | Haason Reddick | 73.63 | 60.88 | 78.58 | 852 | Panthers |
| 22 | 2 | Trevis Gipson | 73.61 | 62.27 | 81.13 | 489 | Bears |
| 23 | 3 | Greg Rousseau | 73.52 | 66.98 | 73.72 | 531 | Bills |
| 24 | 4 | Preston Smith | 73.29 | 68.99 | 72.48 | 688 | Packers |
| 25 | 5 | Jonathan Greenard | 72.96 | 70.77 | 74.89 | 414 | Texans |
| 26 | 6 | Josh Sweat | 72.74 | 69.21 | 72.05 | 654 | Eagles |
| 27 | 7 | Trey Flowers | 72.50 | 75.28 | 74.39 | 302 | Lions |
| 28 | 8 | Markus Golden | 72.41 | 54.45 | 80.70 | 681 | Cardinals |
| 29 | 9 | Carlos Dunlap | 72.32 | 60.67 | 76.65 | 482 | Seahawks |
| 30 | 10 | Melvin Ingram III | 71.99 | 66.78 | 75.71 | 590 | Chiefs |
| 31 | 11 | Jerry Hughes | 71.56 | 63.30 | 73.21 | 557 | Bills |
| 32 | 12 | Chase Young | 71.52 | 89.30 | 60.80 | 477 | Commanders |
| 33 | 13 | Kaden Elliss | 71.44 | 59.01 | 83.47 | 192 | Saints |
| 34 | 14 | Tyus Bowser | 71.26 | 60.82 | 74.05 | 832 | Ravens |
| 35 | 15 | Randy Gregory | 71.23 | 68.16 | 73.45 | 436 | Cowboys |
| 36 | 16 | Kemoko Turay | 71.19 | 60.61 | 81.34 | 224 | Colts |
| 37 | 17 | Chandler Jones | 71.14 | 62.49 | 77.16 | 823 | Cardinals |
| 38 | 18 | Harold Landry III | 70.83 | 61.00 | 73.21 | 981 | Titans |
| 39 | 19 | Ryan Kerrigan | 70.63 | 50.99 | 80.88 | 329 | Eagles |
| 40 | 20 | Emmanuel Ogbah | 70.36 | 69.17 | 68.23 | 755 | Dolphins |
| 41 | 21 | Robert Quinn | 70.28 | 54.81 | 77.64 | 755 | Bears |
| 42 | 22 | Julian Okwara | 70.24 | 60.11 | 79.17 | 361 | Lions |
| 43 | 23 | Ogbo Okoronkwo | 70.13 | 64.32 | 75.14 | 255 | Rams |
| 44 | 24 | Uchenna Nwosu | 70.11 | 63.68 | 71.40 | 781 | Chargers |
| 45 | 25 | John Franklin-Myers | 70.04 | 67.63 | 67.97 | 717 | Jets |
| 46 | 26 | Odafe Oweh | 70.03 | 69.23 | 68.37 | 615 | Ravens |
| 47 | 27 | Takk McKinley | 70.02 | 63.89 | 73.70 | 320 | Browns |
| 48 | 28 | Darrell Taylor | 69.97 | 56.15 | 75.64 | 545 | Seahawks |
| 49 | 29 | Frank Clark | 69.92 | 60.27 | 74.38 | 657 | Chiefs |
| 50 | 30 | Yannick Ngakoue | 69.85 | 58.71 | 73.62 | 835 | Raiders |
| 51 | 31 | Bradley Chubb | 69.61 | 64.37 | 76.97 | 268 | Broncos |
| 52 | 32 | Jaelan Phillips | 69.51 | 59.53 | 72.00 | 603 | Dolphins |
| 53 | 33 | Leonard Floyd | 69.22 | 59.96 | 71.23 | 932 | Rams |
| 54 | 34 | Mario Addison | 69.14 | 50.62 | 77.83 | 481 | Bills |
| 55 | 35 | Chase Winovich | 68.37 | 59.36 | 74.12 | 112 | Patriots |
| 56 | 36 | Sam Hubbard | 68.24 | 61.76 | 70.03 | 877 | Bengals |
| 57 | 37 | Denico Autry | 67.69 | 50.03 | 75.30 | 710 | Titans |
| 58 | 38 | Samson Ebukam | 67.42 | 59.20 | 68.73 | 554 | 49ers |
| 59 | 39 | Dante Fowler Jr. | 67.11 | 59.40 | 70.18 | 508 | Falcons |
| 60 | 40 | Derek Barnett | 66.90 | 63.97 | 66.54 | 718 | Eagles |
| 61 | 41 | Deatrich Wise Jr. | 66.81 | 60.58 | 67.70 | 521 | Patriots |
| 62 | 42 | Jacob Martin | 66.44 | 58.30 | 68.73 | 700 | Texans |
| 63 | 43 | Dee Ford | 66.24 | 58.06 | 78.65 | 106 | 49ers |
| 64 | 44 | Josh Uche | 65.99 | 60.60 | 71.22 | 235 | Patriots |
| 65 | 45 | Alex Highsmith | 65.95 | 62.54 | 64.67 | 851 | Steelers |
| 66 | 46 | Carl Granderson | 65.90 | 61.47 | 68.07 | 448 | Saints |
| 67 | 47 | Pernell McPhee | 65.69 | 50.59 | 77.21 | 234 | Ravens |
| 68 | 48 | Clelin Ferrell | 65.28 | 64.97 | 63.59 | 261 | Raiders |
| 69 | 49 | Genard Avery | 65.27 | 56.72 | 70.53 | 357 | Eagles |
| 70 | 50 | Everson Griffen | 65.20 | 52.08 | 74.53 | 457 | Vikings |
| 71 | 51 | Anthony Nelson | 65.07 | 68.75 | 59.91 | 359 | Buccaneers |
| 72 | 52 | Kwity Paye | 64.86 | 63.15 | 63.80 | 638 | Colts |
| 73 | 53 | Terrell Lewis | 64.40 | 61.99 | 68.64 | 367 | Rams |
| 74 | 54 | Arden Key | 64.21 | 66.50 | 61.01 | 375 | 49ers |
| 75 | 55 | Charles Omenihu | 63.91 | 58.31 | 64.46 | 355 | 49ers |
| 76 | 56 | Dawuane Smoot | 63.88 | 59.68 | 62.99 | 675 | Jaguars |
| 77 | 57 | Cam Gill | 63.73 | 58.67 | 72.14 | 100 | Buccaneers |
| 78 | 58 | Charles Harris | 63.52 | 61.04 | 62.35 | 871 | Lions |
| 79 | 59 | Bud Dupree | 63.29 | 55.71 | 68.67 | 398 | Titans |
| 80 | 60 | Yetur Gross-Matos | 63.13 | 61.79 | 63.25 | 349 | Panthers |
| 81 | 61 | Jonathon Cooper | 63.11 | 61.90 | 60.73 | 457 | Broncos |
| 82 | 62 | A.J. Epenesa | 63.06 | 62.27 | 62.04 | 331 | Bills |
| 83 | 63 | Kenny Willekes | 62.89 | 58.21 | 71.57 | 202 | Vikings |
| 84 | 64 | Kerry Hyder Jr. | 62.89 | 52.32 | 66.76 | 508 | Seahawks |
| 85 | 65 | Lorenzo Carter | 62.86 | 59.22 | 66.23 | 617 | Giants |
| 86 | 66 | Jordan Jenkins | 62.77 | 53.75 | 69.21 | 282 | Texans |
| 87 | 67 | Romeo Okwara | 62.76 | 62.83 | 65.33 | 188 | Lions |
| 88 | 68 | Whitney Mercilus | 62.44 | 48.41 | 71.99 | 311 | Packers |
| 89 | 69 | Devon Kennard | 62.33 | 54.09 | 65.57 | 265 | Cardinals |
| 90 | 70 | Efe Obada | 62.26 | 54.42 | 66.75 | 238 | Bills |

### Rotation/backup (60 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 91 | 1 | Joe Tryon-Shoyinka | 61.95 | 57.44 | 60.79 | 560 | Buccaneers |
| 92 | 2 | Shaka Toney | 61.66 | 57.38 | 67.22 | 117 | Commanders |
| 93 | 3 | Jason Pierre-Paul | 61.62 | 49.08 | 69.52 | 601 | Buccaneers |
| 94 | 4 | Ifeadi Odenigbo | 61.59 | 56.01 | 65.37 | 162 | Browns |
| 95 | 5 | Jeremiah Attaochu | 61.52 | 53.00 | 70.90 | 129 | Bears |
| 96 | 6 | Dennis Gardeck | 61.47 | 54.65 | 63.31 | 173 | Cardinals |
| 97 | 7 | Justin Hollins | 61.29 | 59.07 | 63.44 | 222 | Rams |
| 98 | 8 | Alex Okafor | 61.10 | 49.65 | 67.38 | 463 | Chiefs |
| 99 | 9 | Marquis Haynes Sr. | 60.90 | 55.43 | 61.73 | 222 | Panthers |
| 100 | 10 | Malik Reed | 60.73 | 55.63 | 61.64 | 737 | Broncos |
| 101 | 11 | Quincy Roche | 60.71 | 59.42 | 60.34 | 401 | Giants |
| 102 | 12 | Dorance Armstrong | 60.69 | 59.60 | 59.42 | 507 | Cowboys |
| 103 | 13 | Tarell Basham | 60.68 | 55.46 | 59.99 | 627 | Cowboys |
| 104 | 14 | Chauncey Golston | 60.67 | 59.04 | 59.55 | 414 | Cowboys |
| 105 | 15 | Kyler Fackrell | 60.18 | 48.95 | 66.71 | 382 | Chargers |
| 106 | 16 | Cam Sample | 60.12 | 60.39 | 59.69 | 310 | Bengals |
| 107 | 17 | K'Lavon Chaisson | 60.11 | 57.07 | 59.21 | 384 | Jaguars |
| 108 | 18 | Carl Nassib | 59.97 | 57.51 | 60.44 | 251 | Raiders |
| 109 | 19 | Ronald Blair III | 59.30 | 54.47 | 64.95 | 315 | Jets |
| 110 | 20 | Taco Charlton | 59.20 | 58.00 | 62.83 | 216 | Steelers |
| 111 | 21 | D.J. Wonnum | 59.11 | 54.92 | 58.52 | 951 | Vikings |
| 112 | 22 | Alton Robinson | 59.09 | 50.37 | 61.93 | 742 | Seahawks |
| 113 | 23 | Mike Danna | 58.95 | 57.73 | 56.76 | 534 | Chiefs |
| 114 | 24 | Tarron Jackson | 58.90 | 59.08 | 54.62 | 253 | Eagles |
| 115 | 25 | Benson Mayowa | 58.73 | 49.51 | 62.85 | 510 | Seahawks |
| 116 | 26 | Jaylon Ferguson | 58.40 | 54.07 | 61.58 | 133 | Ravens |
| 117 | 27 | Bryce Huff | 58.35 | 57.17 | 60.65 | 338 | Jets |
| 118 | 28 | Chris Rumph II | 58.25 | 57.81 | 56.34 | 176 | Chargers |
| 119 | 29 | Rasheem Green | 58.11 | 51.96 | 58.05 | 847 | Seahawks |
| 120 | 30 | Payton Turner | 57.90 | 68.65 | 62.83 | 144 | Saints |
| 121 | 31 | Jihad Ward | 57.84 | 49.14 | 61.35 | 455 | Jaguars |
| 122 | 32 | Al-Quadin Muhammad | 57.82 | 55.18 | 55.41 | 800 | Colts |
| 123 | 33 | Bruce Irvin | 57.48 | 42.93 | 73.42 | 173 | Bears |
| 124 | 34 | Brennan Scarlett | 57.44 | 50.09 | 60.34 | 165 | Dolphins |
| 125 | 35 | Oshane Ximines | 57.17 | 56.64 | 61.02 | 183 | Giants |
| 126 | 36 | Khalid Kareem | 57.15 | 63.09 | 55.15 | 110 | Bengals |
| 127 | 37 | Derrek Tuszka | 57.14 | 53.97 | 60.83 | 248 | Steelers |
| 128 | 38 | Dayo Odeyingbo | 57.07 | 62.37 | 56.24 | 173 | Colts |
| 129 | 39 | Stephen Weatherly | 56.97 | 51.76 | 59.44 | 344 | Broncos |
| 130 | 40 | Jonathan Garvin | 56.96 | 56.18 | 57.45 | 396 | Packers |
| 131 | 41 | Zach Allen | 56.84 | 53.20 | 57.02 | 684 | Cardinals |
| 132 | 42 | Jessie Lemonier | 56.80 | 58.20 | 62.52 | 161 | Lions |
| 133 | 43 | Tanoh Kpassagnon | 56.57 | 57.71 | 56.06 | 220 | Saints |
| 134 | 44 | Jordan Willis | 56.26 | 57.18 | 58.57 | 156 | 49ers |
| 135 | 45 | Tipa Galeai | 55.99 | 61.11 | 59.71 | 152 | Packers |
| 136 | 46 | James Vaughters | 55.67 | 51.33 | 59.47 | 210 | Falcons |
| 137 | 47 | Austin Bryant | 55.64 | 56.77 | 57.82 | 436 | Lions |
| 138 | 48 | Casey Toohill | 55.61 | 55.90 | 55.38 | 361 | Commanders |
| 139 | 49 | James Smith-Williams | 55.50 | 52.62 | 55.09 | 388 | Commanders |
| 140 | 50 | Isaac Rochell | 55.27 | 54.17 | 54.29 | 177 | Colts |
| 141 | 51 | Adetokunbo Ogundeji | 55.14 | 54.92 | 52.11 | 527 | Falcons |
| 142 | 52 | Derek Rivers | 55.08 | 47.49 | 63.10 | 143 | Texans |
| 143 | 53 | Olubunmi Rotimi | 55.04 | 52.64 | 60.33 | 204 | Commanders |
| 144 | 54 | Kyle Phillips | 54.54 | 57.65 | 56.22 | 234 | Jets |
| 145 | 55 | Wyatt Ray | 53.89 | 56.12 | 54.16 | 219 | Bengals |
| 146 | 56 | Brandon Copeland | 53.70 | 39.41 | 59.55 | 339 | Falcons |
| 147 | 57 | Porter Gustin | 52.42 | 55.61 | 54.23 | 134 | Browns |
| 148 | 58 | Elerson Smith | 51.79 | 59.24 | 56.44 | 107 | Giants |
| 149 | 59 | Steven Means | 51.47 | 44.89 | 53.16 | 693 | Falcons |
| 150 | 60 | Tim Ward | 51.04 | 55.01 | 51.86 | 191 | Jets |

## G — Guard

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Zack Martin | 98.40 | 93.90 | 97.23 | 1101 | Cowboys |
| 2 | 2 | Joel Bitonio | 98.04 | 93.60 | 96.83 | 1107 | Browns |
| 3 | 3 | Wyatt Teller | 91.63 | 84.90 | 91.95 | 1107 | Browns |
| 4 | 4 | Ali Marpet | 91.36 | 83.60 | 92.37 | 1036 | Buccaneers |
| 5 | 5 | Shaq Mason | 91.30 | 85.20 | 91.20 | 955 | Patriots |
| 6 | 6 | Chris Lindstrom | 90.39 | 83.70 | 90.68 | 1034 | Falcons |
| 7 | 7 | Wes Schweitzer | 87.68 | 78.70 | 89.50 | 401 | Commanders |
| 8 | 8 | Joe Thuney | 86.59 | 80.50 | 86.48 | 1184 | Chiefs |
| 9 | 9 | Isaac Seumalo | 85.47 | 74.80 | 88.42 | 168 | Eagles |
| 10 | 10 | Connor Williams | 85.09 | 76.10 | 86.91 | 948 | Cowboys |
| 11 | 11 | Laken Tomlinson | 83.95 | 75.90 | 85.15 | 1094 | 49ers |
| 12 | 12 | Kevin Zeitler | 82.79 | 75.10 | 83.75 | 1221 | Ravens |
| 13 | 13 | Brandon Scherff | 82.28 | 73.60 | 83.90 | 697 | Commanders |
| 14 | 14 | Quinton Spain | 81.63 | 72.30 | 83.68 | 995 | Bengals |
| 15 | 15 | Alex Cappa | 81.34 | 73.40 | 82.47 | 1182 | Buccaneers |
| 16 | 16 | Trey Smith | 81.31 | 72.30 | 83.15 | 1194 | Chiefs |
| 17 | 17 | Sua Opeta | 81.02 | 74.10 | 81.47 | 163 | Eagles |
| 18 | 18 | James Daniels | 80.67 | 71.80 | 82.41 | 1121 | Bears |

### Good (27 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 19 | 1 | Jonah Jackson | 79.87 | 69.30 | 82.75 | 1037 | Lions |
| 20 | 2 | Rodger Saffold | 79.40 | 69.30 | 81.97 | 853 | Titans |
| 21 | 3 | Oday Aboushi | 79.26 | 68.20 | 82.47 | 298 | Chargers |
| 22 | 4 | Ereck Flowers | 79.23 | 72.00 | 79.88 | 1061 | Commanders |
| 23 | 5 | Quenton Nelson | 79.08 | 69.10 | 81.56 | 767 | Colts |
| 24 | 6 | Trai Turner | 79.05 | 69.40 | 81.31 | 1082 | Steelers |
| 25 | 7 | Mark Glowinski | 78.47 | 70.10 | 79.88 | 843 | Colts |
| 26 | 8 | Ezra Cleveland | 78.43 | 68.60 | 80.81 | 1140 | Vikings |
| 27 | 9 | Greg Van Roten | 78.32 | 68.10 | 80.96 | 700 | Jets |
| 28 | 10 | Jack Driscoll | 77.82 | 70.50 | 78.53 | 512 | Eagles |
| 29 | 11 | Austin Corbett | 77.68 | 68.80 | 79.43 | 1081 | Rams |
| 30 | 12 | Alijah Vera-Tucker | 77.63 | 66.80 | 80.69 | 1027 | Jets |
| 31 | 13 | Halapoulivaati Vaitai | 77.55 | 68.40 | 79.48 | 953 | Lions |
| 32 | 14 | Connor McGovern | 77.50 | 68.70 | 79.20 | 499 | Cowboys |
| 33 | 15 | Nate Davis | 77.45 | 68.90 | 78.98 | 951 | Titans |
| 34 | 16 | Dalton Risner | 77.18 | 68.50 | 78.80 | 832 | Broncos |
| 35 | 17 | Robert Hunt | 77.16 | 67.40 | 79.50 | 1153 | Dolphins |
| 36 | 18 | Landon Dickerson | 76.93 | 67.30 | 79.19 | 859 | Eagles |
| 37 | 19 | Andrew Norwell | 76.60 | 66.70 | 79.03 | 1077 | Jaguars |
| 38 | 20 | David Edwards | 76.22 | 66.90 | 78.26 | 1086 | Rams |
| 39 | 21 | Ben Powers | 75.96 | 66.30 | 78.23 | 844 | Ravens |
| 40 | 22 | Michael Schofield III | 75.66 | 66.80 | 77.40 | 907 | Chargers |
| 41 | 23 | Kevin Dotson | 75.52 | 64.50 | 78.70 | 565 | Steelers |
| 42 | 24 | Justin Pugh | 74.92 | 65.80 | 76.83 | 802 | Cardinals |
| 43 | 25 | Jon Runyan | 74.69 | 65.10 | 76.92 | 1053 | Packers |
| 44 | 26 | Graham Glasgow | 74.67 | 65.10 | 76.89 | 384 | Broncos |
| 45 | 27 | Michael Dunn | 74.35 | 67.90 | 74.48 | 128 | Browns |

### Starter (32 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 46 | 1 | Phil Haynes | 73.87 | 69.80 | 72.41 | 136 | Seahawks |
| 47 | 2 | Gabe Jackson | 73.70 | 63.60 | 76.26 | 922 | Seahawks |
| 48 | 3 | Chris Reed | 73.70 | 67.20 | 73.87 | 522 | Colts |
| 49 | 4 | Ben Bredeson | 73.05 | 56.20 | 80.12 | 294 | Giants |
| 50 | 5 | Ben Bartch | 72.78 | 62.10 | 75.73 | 705 | Jaguars |
| 51 | 6 | Daniel Brunskill | 72.26 | 61.40 | 75.33 | 1089 | 49ers |
| 52 | 7 | Max Scharping | 71.32 | 59.90 | 74.76 | 689 | Texans |
| 53 | 8 | Solomon Kindley | 71.16 | 57.50 | 76.10 | 124 | Dolphins |
| 54 | 9 | Jermaine Eluemunor | 70.62 | 59.80 | 73.66 | 266 | Raiders |
| 55 | 10 | Ike Boettger | 70.23 | 59.80 | 73.01 | 636 | Bills |
| 56 | 11 | Damien Lewis | 69.90 | 57.10 | 74.27 | 696 | Seahawks |
| 57 | 12 | Netane Muti | 69.57 | 59.00 | 72.45 | 317 | Broncos |
| 58 | 13 | Jon Feliciano | 69.06 | 56.70 | 73.13 | 442 | Bills |
| 59 | 14 | Will Hernandez | 68.98 | 55.90 | 73.53 | 1049 | Giants |
| 60 | 15 | Olisaemeka Udoh | 68.83 | 54.40 | 74.28 | 1075 | Vikings |
| 61 | 16 | Cesar Ruiz | 68.75 | 57.60 | 72.01 | 1091 | Saints |
| 62 | 17 | John Leglue | 67.55 | 57.10 | 70.35 | 406 | Steelers |
| 63 | 18 | Dennis Daley | 67.51 | 51.80 | 73.81 | 573 | Panthers |
| 64 | 19 | Royce Newman | 67.46 | 55.70 | 71.14 | 1084 | Packers |
| 65 | 20 | Jackson Carman | 67.33 | 56.30 | 70.51 | 462 | Bengals |
| 66 | 21 | Andrus Peat | 66.94 | 52.10 | 72.66 | 303 | Saints |
| 67 | 22 | Aaron Brewer | 66.70 | 56.30 | 69.47 | 508 | Titans |
| 68 | 23 | Justin McCray | 66.55 | 51.00 | 72.75 | 545 | Texans |
| 69 | 24 | Laurent Duvernay-Tardif | 66.45 | 53.20 | 71.11 | 390 | Jets |
| 70 | 25 | Sean Harlow | 66.36 | 55.70 | 69.30 | 441 | Cardinals |
| 71 | 26 | Ben Cleveland | 65.91 | 55.80 | 68.48 | 367 | Ravens |
| 72 | 27 | John Simpson | 65.64 | 52.60 | 70.16 | 1112 | Raiders |
| 73 | 28 | Michael Jordan | 65.15 | 50.70 | 70.62 | 703 | Panthers |
| 74 | 29 | Tommy Kraemer | 64.77 | 55.50 | 66.78 | 238 | Lions |
| 75 | 30 | John Miller | 64.21 | 52.10 | 68.11 | 656 | Panthers |
| 76 | 31 | Xavier Su'a-Filo | 63.60 | 50.00 | 68.50 | 124 | Bengals |
| 77 | 32 | A.J. Cann | 63.14 | 47.70 | 69.27 | 198 | Jaguars |

### Rotation/backup (5 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 78 | 1 | Cody Ford | 61.78 | 46.70 | 67.67 | 485 | Bills |
| 79 | 2 | Calvin Throckmorton | 60.09 | 42.40 | 67.71 | 938 | Saints |
| 80 | 3 | Lane Taylor | 59.57 | 39.70 | 68.65 | 311 | Texans |
| 81 | 4 | Wes Martin | 58.83 | 40.30 | 67.02 | 130 | Giants |
| 82 | 5 | Senio Kelemete | 58.65 | 40.00 | 66.92 | 110 | Chargers |

## HB — Running Back

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `routes`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Tony Pollard | 86.67 | 86.20 | 82.82 | 168 | Cowboys |
| 2 | 2 | Nick Chubb | 85.77 | 81.70 | 84.31 | 185 | Browns |
| 3 | 3 | Jonathan Taylor | 84.93 | 87.10 | 79.31 | 292 | Colts |
| 4 | 4 | AJ Dillon | 83.34 | 86.70 | 76.93 | 199 | Packers |
| 5 | 5 | Aaron Jones | 81.69 | 82.80 | 76.78 | 316 | Packers |
| 6 | 6 | Javonte Williams | 80.84 | 75.90 | 79.96 | 262 | Broncos |

### Good (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | D'Ernest Johnson | 79.68 | 81.40 | 74.36 | 152 | Browns |
| 8 | 2 | Derrick Henry | 79.20 | 75.30 | 77.63 | 120 | Titans |
| 9 | 3 | Elijah Mitchell | 79.02 | 76.30 | 76.66 | 151 | 49ers |
| 10 | 4 | Austin Ekeler | 78.87 | 75.30 | 77.08 | 417 | Chargers |
| 11 | 5 | Kareem Hunt | 78.65 | 74.40 | 77.31 | 109 | Browns |
| 12 | 6 | Michael Carter | 78.19 | 71.00 | 78.81 | 220 | Jets |
| 13 | 7 | Christian McCaffrey | 78.02 | 80.60 | 72.13 | 118 | Panthers |
| 14 | 8 | Josh Jacobs | 77.87 | 77.70 | 73.82 | 297 | Raiders |
| 15 | 9 | Damien Harris | 77.78 | 87.80 | 66.94 | 114 | Patriots |
| 16 | 10 | James Conner | 76.71 | 82.90 | 68.41 | 236 | Cardinals |
| 17 | 11 | Cordarrelle Patterson | 76.71 | 81.30 | 69.49 | 245 | Falcons |
| 18 | 12 | Miles Sanders | 76.46 | 71.60 | 75.54 | 195 | Eagles |
| 19 | 13 | Najee Harris | 75.01 | 70.70 | 73.72 | 467 | Steelers |
| 20 | 14 | Joe Mixon | 74.81 | 79.50 | 67.51 | 300 | Bengals |
| 21 | 15 | Melvin Gordon III | 74.76 | 77.80 | 68.56 | 247 | Broncos |
| 22 | 16 | Khalil Herbert | 74.58 | 78.80 | 67.60 | 141 | Bears |
| 23 | 17 | Brandon Bolden | 74.55 | 77.40 | 68.48 | 209 | Patriots |
| 24 | 18 | Kenyan Drake | 74.30 | 73.10 | 70.93 | 170 | Raiders |

### Starter (43 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 25 | 1 | Alvin Kamara | 73.96 | 63.20 | 76.96 | 271 | Saints |
| 26 | 2 | James Robinson | 73.87 | 67.40 | 74.02 | 261 | Jaguars |
| 27 | 3 | Devin Singletary | 73.85 | 64.70 | 75.79 | 378 | Bills |
| 28 | 4 | Dalvin Cook | 73.66 | 65.80 | 74.74 | 263 | Vikings |
| 29 | 5 | Justin Jackson | 73.27 | 63.10 | 75.88 | 124 | Chargers |
| 30 | 6 | Leonard Fournette | 72.45 | 73.90 | 67.32 | 361 | Buccaneers |
| 31 | 7 | Ezekiel Elliott | 72.36 | 68.90 | 70.50 | 420 | Cowboys |
| 32 | 8 | Kenneth Gainwell | 72.00 | 70.70 | 68.70 | 182 | Eagles |
| 33 | 9 | Darrell Henderson | 71.37 | 68.70 | 68.98 | 280 | Rams |
| 34 | 10 | Nyheim Hines | 71.15 | 72.70 | 65.95 | 204 | Colts |
| 35 | 11 | David Montgomery | 71.02 | 69.80 | 67.66 | 289 | Bears |
| 36 | 12 | Mark Ingram II | 70.89 | 67.60 | 68.91 | 148 | Saints |
| 37 | 13 | Chase Edmonds | 70.62 | 67.90 | 68.26 | 253 | Cardinals |
| 38 | 14 | Dontrell Hilliard | 70.14 | 57.90 | 74.13 | 109 | Titans |
| 39 | 15 | Saquon Barkley | 70.07 | 59.10 | 73.21 | 257 | Giants |
| 40 | 16 | J.D. McKissic | 69.91 | 69.50 | 66.02 | 219 | Commanders |
| 41 | 17 | Clyde Edwards-Helaire | 69.39 | 64.30 | 68.62 | 178 | Chiefs |
| 42 | 18 | Antonio Gibson | 69.23 | 63.30 | 69.01 | 247 | Commanders |
| 43 | 19 | Alexander Mattison | 68.97 | 61.00 | 70.12 | 196 | Vikings |
| 44 | 20 | Jamaal Williams | 68.74 | 69.80 | 63.87 | 115 | Lions |
| 45 | 21 | Devontae Booker | 68.54 | 65.00 | 66.73 | 253 | Giants |
| 46 | 22 | Zack Moss | 68.48 | 67.30 | 65.10 | 176 | Bills |
| 47 | 23 | Jeremy McNichols | 68.26 | 60.40 | 69.34 | 149 | Titans |
| 48 | 24 | Sony Michel | 68.23 | 66.10 | 65.48 | 253 | Rams |
| 49 | 25 | Devonta Freeman | 68.19 | 68.20 | 64.01 | 282 | Ravens |
| 50 | 26 | Alex Collins | 68.16 | 66.00 | 65.43 | 107 | Seahawks |
| 51 | 27 | Jerick McKinnon | 68.01 | 66.50 | 64.85 | 125 | Chiefs |
| 52 | 28 | Latavius Murray | 67.43 | 65.50 | 64.55 | 154 | Ravens |
| 53 | 29 | Samaje Perine | 67.35 | 62.20 | 66.61 | 164 | Bengals |
| 54 | 30 | Myles Gaskin | 66.87 | 64.80 | 64.09 | 288 | Dolphins |
| 55 | 31 | Rex Burkhead | 66.76 | 65.40 | 63.50 | 189 | Texans |
| 56 | 32 | Giovani Bernard | 66.60 | 62.60 | 65.10 | 108 | Buccaneers |
| 57 | 33 | DeeJay Dallas | 66.01 | 62.60 | 64.12 | 106 | Seahawks |
| 58 | 34 | D'Andre Swift | 65.73 | 58.00 | 66.71 | 320 | Lions |
| 59 | 35 | Chuba Hubbard | 65.52 | 64.90 | 61.77 | 178 | Panthers |
| 60 | 36 | David Johnson | 65.32 | 60.40 | 64.44 | 176 | Texans |
| 61 | 37 | Ameer Abdullah | 64.52 | 62.40 | 61.76 | 193 | Panthers |
| 62 | 38 | Darrel Williams | 64.50 | 67.20 | 58.54 | 347 | Chiefs |
| 63 | 39 | JaMycal Hasty | 64.24 | 61.60 | 61.83 | 110 | 49ers |
| 64 | 40 | Royce Freeman | 64.01 | 58.00 | 63.85 | 115 | Texans |
| 65 | 41 | Carlos Hyde | 63.75 | 53.80 | 66.21 | 159 | Jaguars |
| 66 | 42 | Ty Johnson | 62.95 | 56.00 | 63.42 | 256 | Jets |
| 67 | 43 | Mike Davis | 62.81 | 56.10 | 63.12 | 313 | Falcons |

### Rotation/backup (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 68 | 1 | Dare Ogunbowale | 55.65 | 45.20 | 58.45 | 107 | Jaguars |

## LB — Linebacker

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (1 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | De'Vondre Campbell | 83.17 | 84.70 | 78.46 | 987 | Packers |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 2 | 1 | Jamie Collins Sr. | 78.73 | 84.30 | 73.93 | 301 | Patriots |
| 3 | 2 | Micah Parsons | 78.23 | 89.80 | 67.34 | 902 | Cowboys |
| 4 | 3 | Frankie Luvu | 78.11 | 84.80 | 69.97 | 249 | Panthers |
| 5 | 4 | Pete Werner | 77.79 | 79.90 | 75.15 | 394 | Saints |
| 6 | 5 | Lavonte David | 77.68 | 78.80 | 75.21 | 788 | Buccaneers |
| 7 | 6 | Demario Davis | 77.59 | 77.90 | 73.70 | 1038 | Saints |
| 8 | 7 | T.J. Edwards | 75.42 | 76.30 | 73.23 | 684 | Eagles |
| 9 | 8 | Fred Warner | 75.36 | 75.20 | 71.78 | 977 | 49ers |
| 10 | 9 | Mack Wilson Sr. | 74.34 | 75.30 | 72.63 | 193 | Browns |

### Starter (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 11 | 1 | Kyle Van Noy | 73.64 | 72.90 | 71.08 | 810 | Patriots |
| 12 | 2 | Bobby Wagner | 73.33 | 71.80 | 70.67 | 1129 | Seahawks |
| 13 | 3 | Jeremiah Owusu-Koramoah | 73.13 | 76.50 | 69.65 | 597 | Browns |
| 14 | 4 | Josh Bynes | 73.06 | 74.80 | 70.04 | 537 | Ravens |
| 15 | 5 | Shaq Thompson | 71.83 | 72.70 | 68.96 | 796 | Panthers |
| 16 | 6 | Nick Bolton | 71.77 | 69.20 | 70.30 | 623 | Chiefs |
| 17 | 7 | Mykal Walker | 70.92 | 71.30 | 67.11 | 194 | Falcons |
| 18 | 8 | Anthony Walker Jr. | 69.63 | 69.10 | 67.79 | 701 | Browns |
| 19 | 9 | Matt Milano | 69.49 | 70.10 | 67.48 | 915 | Bills |
| 20 | 10 | Zaire Franklin | 69.42 | 71.30 | 69.10 | 201 | Colts |
| 21 | 11 | Ja'Whaun Bentley | 69.23 | 68.20 | 67.19 | 693 | Patriots |
| 22 | 12 | Kyzir White | 68.92 | 66.50 | 67.93 | 979 | Chargers |
| 23 | 13 | Sione Takitaki | 68.87 | 67.50 | 68.58 | 285 | Browns |
| 24 | 14 | Jaylon Smith | 68.28 | 66.30 | 68.87 | 329 | Giants |
| 25 | 15 | Zaven Collins | 67.98 | 69.30 | 66.85 | 220 | Cardinals |
| 26 | 16 | Jordan Hicks | 67.22 | 64.70 | 64.74 | 1053 | Cardinals |
| 27 | 17 | Travin Howard | 67.21 | 69.80 | 69.06 | 103 | Rams |
| 28 | 18 | K.J. Wright | 66.92 | 63.70 | 64.90 | 426 | Raiders |
| 29 | 19 | Duke Riley | 66.88 | 67.20 | 66.09 | 227 | Dolphins |
| 30 | 20 | Dre Greenlaw | 66.55 | 69.50 | 68.21 | 113 | 49ers |
| 31 | 21 | Denzel Perryman | 65.87 | 62.30 | 66.84 | 863 | Raiders |
| 32 | 22 | Reggie Ragland | 65.40 | 61.20 | 65.44 | 474 | Giants |
| 33 | 23 | Zach Cunningham | 65.37 | 60.20 | 66.11 | 646 | Titans |
| 34 | 24 | Leighton Vander Esch | 65.28 | 63.50 | 65.64 | 661 | Cowboys |
| 35 | 25 | Azeez Al-Shaair | 64.97 | 64.80 | 65.06 | 730 | 49ers |
| 36 | 26 | Jordyn Brooks | 64.56 | 58.40 | 65.29 | 1109 | Seahawks |
| 37 | 27 | Drue Tranquill | 64.45 | 64.60 | 67.16 | 560 | Chargers |
| 38 | 28 | Jerome Baker | 64.19 | 60.90 | 62.70 | 971 | Dolphins |
| 39 | 29 | Sam Eguavoen | 63.91 | 63.30 | 61.13 | 181 | Dolphins |
| 40 | 30 | Shaquille Quarterman | 63.89 | 62.90 | 64.67 | 144 | Jaguars |
| 41 | 31 | David Long Jr. | 63.87 | 67.40 | 64.54 | 634 | Titans |
| 42 | 32 | Eric Kendricks | 63.28 | 59.20 | 64.59 | 1032 | Vikings |
| 43 | 33 | Blake Lynch | 63.06 | 64.60 | 67.24 | 218 | Vikings |
| 44 | 34 | Jonas Griffith | 62.76 | 69.10 | 70.64 | 255 | Broncos |
| 45 | 35 | Bobby Okereke | 62.50 | 58.50 | 61.63 | 1072 | Colts |
| 46 | 36 | Divine Deablo | 62.06 | 63.20 | 65.95 | 297 | Raiders |

### Rotation/backup (73 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 47 | 1 | Akeem Davis-Gaither | 61.60 | 60.20 | 63.26 | 207 | Bengals |
| 48 | 2 | Ernest Jones | 61.43 | 59.30 | 64.57 | 440 | Rams |
| 49 | 3 | Cole Holcomb | 61.17 | 56.70 | 62.35 | 1021 | Commanders |
| 50 | 4 | Anthony Barr | 60.65 | 62.90 | 62.71 | 783 | Vikings |
| 51 | 5 | Alex Singleton | 60.29 | 52.40 | 62.18 | 720 | Eagles |
| 52 | 6 | Roquan Smith | 60.17 | 47.80 | 65.09 | 1010 | Bears |
| 53 | 7 | Dont'a Hightower | 60.11 | 53.70 | 61.72 | 634 | Patriots |
| 54 | 8 | Malcolm Smith | 60.06 | 57.00 | 62.15 | 425 | Browns |
| 55 | 9 | Tremaine Edmunds | 60.01 | 50.40 | 63.55 | 872 | Bills |
| 56 | 10 | Jacob Phillips | 59.82 | 62.40 | 64.63 | 123 | Browns |
| 57 | 11 | Cody Barton | 59.34 | 59.70 | 63.01 | 190 | Seahawks |
| 58 | 12 | Elandon Roberts | 59.07 | 54.10 | 59.56 | 620 | Dolphins |
| 59 | 13 | Joe Schobert | 58.96 | 52.10 | 59.85 | 921 | Steelers |
| 60 | 14 | Chris Board | 58.84 | 53.60 | 60.54 | 337 | Ravens |
| 61 | 15 | Krys Barnes | 58.75 | 53.00 | 60.60 | 526 | Packers |
| 62 | 16 | Isaiah Simmons | 58.58 | 51.00 | 59.47 | 1005 | Cardinals |
| 63 | 17 | Markus Bailey | 58.58 | 55.60 | 62.75 | 256 | Bengals |
| 64 | 18 | Willie Gay | 58.51 | 55.70 | 59.66 | 436 | Chiefs |
| 65 | 19 | Oren Burks | 58.33 | 53.80 | 59.89 | 206 | Packers |
| 66 | 20 | Blake Martinez | 58.00 | 57.80 | 60.84 | 142 | Giants |
| 67 | 21 | A.J. Klein | 57.92 | 53.30 | 59.98 | 277 | Bills |
| 68 | 22 | Kwon Alexander | 57.78 | 53.90 | 61.57 | 535 | Saints |
| 69 | 23 | Logan Wilson | 57.64 | 53.90 | 59.99 | 707 | Bengals |
| 70 | 24 | Foyesade Oluokun | 57.54 | 47.00 | 60.72 | 1148 | Falcons |
| 71 | 25 | Jalen Reeves-Maybin | 57.34 | 55.80 | 59.03 | 615 | Lions |
| 72 | 26 | Neville Hewitt | 57.13 | 49.40 | 61.89 | 325 | Texans |
| 73 | 27 | Baron Browning | 56.78 | 54.90 | 60.73 | 528 | Broncos |
| 74 | 28 | Kenny Young | 56.30 | 50.60 | 60.40 | 645 | Broncos |
| 75 | 29 | Christian Kirksey | 56.26 | 50.30 | 62.50 | 790 | Texans |
| 76 | 30 | Jayon Brown | 56.20 | 51.20 | 61.09 | 421 | Titans |
| 77 | 31 | Kevin Minter | 55.97 | 53.70 | 58.27 | 331 | Buccaneers |
| 78 | 32 | E.J. Speed | 55.92 | 51.60 | 63.50 | 146 | Colts |
| 79 | 33 | Cory Littleton | 55.66 | 47.20 | 58.74 | 663 | Raiders |
| 80 | 34 | Josh Woods | 55.63 | 55.90 | 61.40 | 113 | Lions |
| 81 | 35 | Ben Niemann | 55.21 | 47.10 | 56.97 | 558 | Chiefs |
| 82 | 36 | Anthony Hitchens | 55.19 | 44.10 | 60.23 | 597 | Chiefs |
| 83 | 37 | Germaine Pratt | 54.92 | 47.00 | 57.21 | 692 | Bengals |
| 84 | 38 | Christian Jones | 54.58 | 45.30 | 60.17 | 116 | Bears |
| 85 | 39 | Jamin Davis | 54.44 | 46.80 | 56.35 | 581 | Commanders |
| 86 | 40 | Robert Spillane | 54.12 | 49.80 | 59.66 | 347 | Steelers |
| 87 | 41 | Patrick Queen | 54.01 | 43.50 | 56.85 | 826 | Ravens |
| 88 | 42 | Damien Wilson | 53.74 | 44.00 | 57.00 | 866 | Jaguars |
| 89 | 43 | Troy Reeder | 53.56 | 46.70 | 56.78 | 682 | Rams |
| 90 | 44 | Monty Rice | 53.43 | 56.10 | 61.26 | 179 | Titans |
| 91 | 45 | Nick Vigil | 53.29 | 42.10 | 58.32 | 718 | Vikings |
| 92 | 46 | Rashaan Evans | 53.15 | 44.50 | 57.20 | 445 | Titans |
| 93 | 47 | C.J. Mosley | 52.43 | 42.00 | 58.61 | 1098 | Jets |
| 94 | 48 | Kamu Grugier-Hill | 52.20 | 44.20 | 57.33 | 778 | Texans |
| 95 | 49 | Jermaine Carter | 51.88 | 42.60 | 56.08 | 852 | Panthers |
| 96 | 50 | David Mayo | 51.70 | 42.90 | 58.81 | 166 | Commanders |
| 97 | 51 | Zack Baun | 51.52 | 44.60 | 58.44 | 194 | Saints |
| 98 | 52 | Joe Bachie | 51.23 | 51.90 | 60.61 | 160 | Bengals |
| 99 | 53 | Malik Harrison | 50.69 | 42.10 | 57.76 | 171 | Ravens |
| 100 | 54 | Quincy Williams | 50.63 | 44.20 | 57.06 | 881 | Jets |
| 101 | 55 | Myles Jack | 50.53 | 37.70 | 57.56 | 917 | Jaguars |
| 102 | 56 | Devin White | 50.04 | 36.20 | 56.03 | 1080 | Buccaneers |
| 103 | 57 | Eric Wilson | 49.90 | 39.60 | 56.52 | 298 | Texans |
| 104 | 58 | Del'Shawn Phillips | 49.60 | 38.10 | 55.55 | 161 | Jets |
| 105 | 59 | Deion Jones | 49.02 | 34.60 | 54.95 | 1070 | Falcons |
| 106 | 60 | Devin Bush | 48.94 | 34.40 | 59.37 | 762 | Steelers |
| 107 | 61 | Derrick Barnes | 48.91 | 30.10 | 58.26 | 448 | Lions |
| 108 | 62 | Keanu Neal | 48.39 | 35.90 | 55.49 | 579 | Cowboys |
| 109 | 63 | Alex Anzalone | 47.03 | 35.40 | 55.31 | 827 | Lions |
| 110 | 64 | Kenneth Murray Jr. | 47.00 | 34.00 | 55.18 | 363 | Chargers |
| 111 | 65 | Tae Crowder | 46.47 | 29.10 | 56.23 | 1099 | Giants |
| 112 | 66 | Jon Bostic | 46.06 | 34.40 | 56.03 | 179 | Commanders |
| 113 | 67 | Alec Ogletree | 45.83 | 29.20 | 54.80 | 697 | Bears |
| 114 | 68 | Davion Taylor | 45.71 | 38.60 | 55.87 | 250 | Eagles |
| 115 | 69 | Justin Strnad | 45.52 | 32.00 | 55.26 | 314 | Broncos |
| 116 | 70 | Garret Wallow | 45.08 | 33.90 | 55.23 | 180 | Texans |
| 117 | 71 | Amen Ogbongbemiga | 45.00 | 33.30 | 55.48 | 111 | Chargers |
| 118 | 72 | Jarrad Davis | 45.00 | 28.60 | 54.43 | 209 | Jets |
| 119 | 73 | Tanner Vallejo | 45.00 | 35.00 | 56.97 | 121 | Cardinals |

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
| 6 | 6 | Kyler Murray | 81.25 | 82.17 | 77.81 | 578 | Cardinals |
| 7 | 7 | Matthew Stafford | 80.93 | 79.25 | 78.93 | 670 | Rams |
| 8 | 8 | Justin Herbert | 80.56 | 83.91 | 73.28 | 793 | Chargers |

### Good (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 9 | 1 | Derek Carr | 79.53 | 78.60 | 76.21 | 727 | Raiders |
| 10 | 2 | Russell Wilson | 79.03 | 78.66 | 78.19 | 486 | Seahawks |
| 11 | 3 | Patrick Mahomes | 78.51 | 77.91 | 74.69 | 787 | Chiefs |
| 12 | 4 | Josh Allen | 77.34 | 79.08 | 71.55 | 771 | Bills |
| 13 | 5 | Ryan Tannehill | 77.24 | 82.60 | 70.60 | 633 | Titans |
| 14 | 6 | Matt Ryan | 75.92 | 77.42 | 70.67 | 653 | Falcons |

### Starter (13 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 15 | 1 | Jimmy Garoppolo | 73.19 | 72.62 | 75.86 | 501 | 49ers |
| 16 | 2 | Lamar Jackson | 70.28 | 70.53 | 70.30 | 495 | Ravens |
| 17 | 3 | Teddy Bridgewater | 70.20 | 69.47 | 70.96 | 495 | Broncos |
| 18 | 4 | Mac Jones | 69.98 | 77.40 | 72.33 | 600 | Patriots |
| 19 | 5 | Carson Wentz | 69.04 | 67.02 | 68.19 | 622 | Colts |
| 20 | 6 | Baker Mayfield | 68.58 | 68.68 | 67.44 | 516 | Browns |
| 21 | 7 | Jared Goff | 68.11 | 65.65 | 67.59 | 574 | Lions |
| 22 | 8 | Jalen Hurts | 66.83 | 71.43 | 68.48 | 541 | Eagles |
| 23 | 9 | Jameis Winston | 66.79 | 65.91 | 75.55 | 199 | Saints |
| 24 | 10 | Daniel Jones | 66.68 | 71.05 | 63.86 | 439 | Giants |
| 25 | 11 | Tua Tagovailoa | 64.52 | 66.52 | 66.70 | 460 | Dolphins |
| 26 | 12 | Geno Smith | 64.43 | 71.22 | 77.80 | 118 | Seahawks |
| 27 | 13 | Ben Roethlisberger | 63.23 | 58.34 | 64.31 | 692 | Steelers |

### Rotation/backup (18 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Davis Mills | 61.45 | 59.00 | 67.69 | 469 | Texans |
| 29 | 2 | Taylor Heinicke | 60.95 | 58.95 | 65.87 | 616 | Commanders |
| 30 | 3 | Andy Dalton | 60.39 | 65.78 | 60.19 | 279 | Bears |
| 31 | 4 | Colt McCoy | 60.17 | 57.49 | 71.81 | 115 | Cardinals |
| 32 | 5 | Trevor Lawrence | 60.13 | 58.30 | 57.17 | 709 | Jaguars |
| 33 | 6 | Jacoby Brissett | 60.07 | 70.08 | 59.01 | 277 | Dolphins |
| 34 | 7 | Justin Fields | 60.00 | 60.80 | 63.03 | 378 | Bears |
| 35 | 8 | Trevor Siemian | 59.86 | 63.31 | 65.81 | 218 | Saints |
| 36 | 9 | Drew Lock | 59.56 | 61.62 | 64.94 | 138 | Broncos |
| 37 | 10 | Taysom Hill | 58.86 | 62.47 | 62.45 | 162 | Saints |
| 38 | 11 | Tyler Huntley | 58.16 | 62.00 | 58.03 | 245 | Ravens |
| 39 | 12 | Zach Wilson | 57.53 | 54.80 | 56.31 | 469 | Jets |
| 40 | 13 | Mike White | 56.81 | 51.25 | 63.35 | 147 | Jets |
| 41 | 14 | Tyrod Taylor | 56.74 | 55.81 | 59.33 | 188 | Texans |
| 42 | 15 | Sam Darnold | 55.94 | 54.73 | 57.48 | 486 | Panthers |
| 43 | 16 | Tim Boyle | 55.09 | 52.01 | 55.81 | 100 | Lions |
| 44 | 17 | Cam Newton | 55.07 | 57.67 | 57.25 | 147 | Panthers |
| 45 | 18 | Mike Glennon | 52.82 | 42.33 | 54.23 | 193 | Giants |

## S — Safety

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_defense` · **Snap column (volume filter):** `snap_counts_defense`

### Elite (11 players)

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
| 9 | 9 | Amani Hooker | 82.98 | 83.40 | 80.98 | 705 | Titans |
| 10 | 10 | Jayron Kearse | 82.03 | 76.20 | 82.24 | 1012 | Cowboys |
| 11 | 11 | DeAndre Houston-Carson | 81.11 | 82.00 | 82.27 | 420 | Bears |

### Good (16 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 12 | 1 | Devin McCourty | 79.86 | 76.20 | 78.14 | 1019 | Patriots |
| 13 | 2 | Adrian Amos | 79.84 | 74.10 | 79.50 | 1047 | Packers |
| 14 | 3 | Xavier McKinney | 79.59 | 78.40 | 80.11 | 1134 | Giants |
| 15 | 4 | Tre'von Moehrig | 79.31 | 77.70 | 76.22 | 1152 | Raiders |
| 16 | 5 | Jimmie Ward | 79.08 | 73.70 | 80.24 | 991 | 49ers |
| 17 | 6 | Justin Simmons | 78.01 | 73.40 | 76.92 | 1082 | Broncos |
| 18 | 7 | Vonn Bell | 77.96 | 78.00 | 74.88 | 1004 | Bengals |
| 19 | 8 | Derwin James Jr. | 77.72 | 76.60 | 78.72 | 961 | Chargers |
| 20 | 9 | Tyrann Mathieu | 76.94 | 76.40 | 73.93 | 996 | Chiefs |
| 21 | 10 | Jeremy Chinn | 76.12 | 74.30 | 74.16 | 1015 | Panthers |
| 22 | 11 | Mike Edwards | 75.84 | 75.30 | 74.97 | 532 | Buccaneers |
| 23 | 12 | Quandre Diggs | 75.74 | 72.30 | 75.12 | 1230 | Seahawks |
| 24 | 13 | Eric Rowe | 75.62 | 70.70 | 74.74 | 638 | Dolphins |
| 25 | 14 | Kyle Dugger | 75.11 | 75.70 | 72.55 | 733 | Patriots |
| 26 | 15 | Jordan Fuller | 74.26 | 69.40 | 75.51 | 1028 | Rams |
| 27 | 16 | Marcus Epps | 74.17 | 67.90 | 77.06 | 505 | Eagles |

### Starter (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 28 | 1 | Terrell Edmunds | 73.54 | 71.60 | 70.99 | 1145 | Steelers |
| 29 | 2 | Juan Thornhill | 73.43 | 68.60 | 72.48 | 850 | Chiefs |
| 30 | 3 | Jordan Whitehead | 73.37 | 72.20 | 71.86 | 795 | Buccaneers |
| 31 | 4 | Jalen Thompson | 73.34 | 72.90 | 73.74 | 986 | Cardinals |
| 32 | 5 | Andrew Wingard | 72.40 | 75.10 | 69.91 | 930 | Jaguars |
| 33 | 6 | John Johnson III | 70.63 | 66.50 | 72.29 | 903 | Browns |
| 34 | 7 | Jahleel Addae | 70.32 | 67.20 | 75.99 | 132 | Colts |
| 35 | 8 | Budda Baker | 69.87 | 65.60 | 68.86 | 1036 | Cardinals |
| 36 | 9 | Andrew Adams | 69.82 | 68.00 | 75.86 | 214 | Buccaneers |
| 37 | 10 | Xavier Woods | 68.95 | 58.30 | 72.40 | 1207 | Vikings |
| 38 | 11 | Deon Bush | 68.21 | 68.40 | 69.15 | 377 | Bears |
| 39 | 12 | Grant Delpit | 67.77 | 64.50 | 67.02 | 599 | Browns |
| 40 | 13 | Dean Marlowe | 67.73 | 63.00 | 70.52 | 700 | Lions |
| 41 | 14 | Jonathan Owens | 67.61 | 71.10 | 74.68 | 168 | Texans |
| 42 | 15 | Chuck Clark | 67.32 | 61.10 | 68.00 | 1023 | Ravens |
| 43 | 16 | Tracy Walker III | 67.30 | 62.40 | 68.31 | 881 | Lions |
| 44 | 17 | Anthony Harris | 66.90 | 59.60 | 69.49 | 834 | Eagles |
| 45 | 18 | Darnell Savage | 66.82 | 62.10 | 66.54 | 1037 | Packers |
| 46 | 19 | Taylor Rapp | 66.48 | 58.80 | 69.84 | 1113 | Rams |
| 47 | 20 | Malcolm Jenkins | 66.46 | 60.80 | 66.55 | 1042 | Saints |
| 48 | 21 | Malik Hooker | 66.23 | 65.50 | 68.53 | 445 | Cowboys |
| 49 | 22 | Damontae Kazee | 66.15 | 65.10 | 66.43 | 900 | Cowboys |
| 50 | 23 | Duron Harmon | 66.04 | 53.60 | 70.16 | 1071 | Falcons |
| 51 | 24 | Andre Cisco | 65.87 | 62.70 | 69.70 | 247 | Jaguars |
| 52 | 25 | Andrew Sendejo | 65.75 | 65.10 | 65.72 | 610 | Colts |
| 53 | 26 | Sean Chandler | 65.65 | 62.60 | 67.74 | 538 | Panthers |
| 54 | 27 | Ashtyn Davis | 65.52 | 66.10 | 66.55 | 745 | Jets |
| 55 | 28 | Talanoa Hufanga | 65.37 | 60.20 | 67.59 | 395 | 49ers |
| 56 | 29 | Rodney McLeod | 65.18 | 58.40 | 68.44 | 684 | Eagles |
| 57 | 30 | Rayshawn Jenkins | 65.17 | 61.30 | 65.36 | 836 | Jaguars |
| 58 | 31 | Nasir Adderley | 65.02 | 62.90 | 66.48 | 987 | Chargers |
| 59 | 32 | Julian Love | 64.69 | 57.40 | 65.77 | 612 | Giants |
| 60 | 33 | George Odum | 64.41 | 60.70 | 67.87 | 472 | Colts |
| 61 | 34 | Ricardo Allen | 63.67 | 57.70 | 66.70 | 171 | Bengals |
| 62 | 35 | Eddie Jackson | 63.67 | 56.90 | 65.49 | 787 | Bears |
| 63 | 36 | Erik Harris | 63.33 | 60.30 | 64.47 | 702 | Falcons |
| 64 | 37 | Geno Stone | 63.26 | 63.70 | 68.95 | 219 | Ravens |
| 65 | 38 | Shawn Williams | 63.07 | 63.50 | 66.06 | 164 | Falcons |
| 66 | 39 | Jaquiski Tartt | 62.66 | 62.10 | 63.98 | 727 | 49ers |
| 67 | 40 | Donovan Wilson | 62.58 | 64.10 | 65.60 | 338 | Cowboys |
| 68 | 41 | Ronnie Harrison | 62.54 | 54.50 | 68.17 | 584 | Browns |
| 69 | 42 | Jessie Bates III | 62.47 | 53.40 | 65.33 | 953 | Bengals |

### Rotation/backup (36 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 70 | 1 | Minkah Fitzpatrick | 61.65 | 49.40 | 66.14 | 1083 | Steelers |
| 71 | 2 | Tavon Wilson | 61.57 | 61.40 | 65.76 | 102 | 49ers |
| 72 | 3 | Dallin Leavitt | 61.44 | 60.80 | 65.04 | 250 | Raiders |
| 73 | 4 | Daniel Thomas | 61.35 | 56.40 | 71.14 | 205 | Jaguars |
| 74 | 5 | Caden Sterns | 61.08 | 54.40 | 64.30 | 311 | Broncos |
| 75 | 6 | DeShon Elliott | 61.05 | 61.70 | 64.35 | 305 | Ravens |
| 76 | 7 | Jaylinn Hawkins | 60.66 | 60.90 | 62.47 | 462 | Falcons |
| 77 | 8 | Khari Willis | 59.67 | 50.10 | 65.87 | 564 | Colts |
| 78 | 9 | C.J. Moore | 59.22 | 62.30 | 62.30 | 158 | Lions |
| 79 | 10 | Juston Burris | 58.93 | 60.00 | 60.12 | 420 | Panthers |
| 80 | 11 | Jabrill Peppers | 58.85 | 53.20 | 65.20 | 229 | Giants |
| 81 | 12 | Brandon Stephens | 58.29 | 46.30 | 62.12 | 742 | Ravens |
| 82 | 13 | Dane Cruikshank | 57.79 | 58.20 | 62.08 | 415 | Titans |
| 83 | 14 | Tashaun Gipson Sr. | 57.66 | 50.80 | 60.93 | 660 | Bears |
| 84 | 15 | Alohi Gilman | 57.49 | 59.60 | 60.29 | 355 | Chargers |
| 85 | 16 | Johnathan Abram | 57.29 | 54.20 | 60.71 | 955 | Raiders |
| 86 | 17 | Eric Murray | 57.20 | 54.70 | 57.28 | 759 | Texans |
| 87 | 18 | Julian Blackmon | 57.05 | 53.20 | 62.59 | 376 | Colts |
| 88 | 19 | Daniel Sorensen | 56.90 | 52.60 | 55.91 | 699 | Chiefs |
| 89 | 20 | Jeremy Reaves | 56.83 | 51.00 | 67.21 | 195 | Commanders |
| 90 | 21 | Marcus Maye | 56.25 | 49.90 | 61.71 | 362 | Jets |
| 91 | 22 | Jamal Adams | 56.20 | 47.40 | 62.02 | 872 | Seahawks |
| 92 | 23 | K'Von Wallace | 56.18 | 52.50 | 63.38 | 183 | Eagles |
| 93 | 24 | Kareem Jackson | 55.42 | 48.50 | 57.46 | 895 | Broncos |
| 94 | 25 | Will Harris | 55.32 | 41.70 | 60.24 | 1011 | Lions |
| 95 | 26 | Terrence Brooks | 55.12 | 47.50 | 60.61 | 180 | Texans |
| 96 | 27 | Kenny Robinson | 54.30 | 55.70 | 60.80 | 182 | Panthers |
| 97 | 28 | Brandon Jones | 54.21 | 48.60 | 55.02 | 644 | Dolphins |
| 98 | 29 | Nick Scott | 54.03 | 47.00 | 59.65 | 415 | Rams |
| 99 | 30 | Justin Reid | 52.94 | 45.30 | 56.97 | 780 | Texans |
| 100 | 31 | Landon Collins | 50.68 | 38.80 | 59.41 | 675 | Commanders |
| 101 | 32 | Roderic Teamer | 50.29 | 52.00 | 55.23 | 195 | Raiders |
| 102 | 33 | Henry Black | 49.93 | 44.10 | 54.33 | 263 | Packers |
| 103 | 34 | Adrian Colbert | 47.49 | 44.50 | 57.04 | 161 | Browns |
| 104 | 35 | Trey Marshall | 47.10 | 44.10 | 51.93 | 197 | Chargers |
| 105 | 36 | Sam Franklin Jr. | 45.00 | 36.70 | 52.20 | 153 | Panthers |

## T — Tackle

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `snap_counts_offense`

### Elite (42 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Trent Williams | 99.00 | 97.80 | 95.89 | 936 | 49ers |
| 2 | 2 | Tyron Smith | 96.64 | 91.40 | 95.96 | 738 | Cowboys |
| 3 | 3 | Jordan Mailata | 93.09 | 87.40 | 92.71 | 914 | Eagles |
| 4 | 4 | La'el Collins | 90.87 | 82.00 | 92.61 | 671 | Cowboys |
| 5 | 5 | Tristan Wirfs | 90.32 | 84.60 | 89.97 | 1182 | Buccaneers |
| 6 | 6 | Andrew Whitworth | 90.22 | 86.10 | 88.80 | 926 | Rams |
| 7 | 7 | Ryan Ramczyk | 89.85 | 84.10 | 89.51 | 653 | Saints |
| 8 | 8 | Rashawn Slater | 89.69 | 83.60 | 89.58 | 1116 | Chargers |
| 9 | 9 | Lane Johnson | 89.20 | 82.40 | 89.57 | 821 | Eagles |
| 10 | 10 | Kolton Miller | 89.03 | 84.00 | 88.22 | 1139 | Raiders |
| 11 | 11 | Elgton Jenkins | 88.13 | 82.10 | 87.98 | 496 | Packers |
| 12 | 12 | Braden Smith | 88.03 | 80.60 | 88.81 | 711 | Colts |
| 13 | 13 | Donovan Smith | 87.92 | 83.30 | 86.83 | 1147 | Buccaneers |
| 14 | 14 | Rob Havenstein | 87.74 | 81.50 | 87.73 | 957 | Rams |
| 15 | 15 | David Quessenberry | 87.63 | 80.70 | 88.08 | 1184 | Titans |
| 16 | 16 | Jack Conklin | 87.12 | 78.80 | 88.50 | 361 | Browns |
| 17 | 17 | Trent Brown | 86.31 | 79.30 | 86.81 | 489 | Patriots |
| 18 | 18 | Penei Sewell | 86.25 | 77.00 | 88.25 | 1039 | Lions |
| 19 | 19 | Andrew Thomas | 85.55 | 78.90 | 85.82 | 800 | Giants |
| 20 | 20 | Charles Leno Jr. | 85.23 | 81.20 | 83.75 | 1121 | Commanders |
| 21 | 21 | Taylor Moton | 85.15 | 77.50 | 86.09 | 1149 | Panthers |
| 22 | 22 | Dion Dawkins | 84.72 | 77.50 | 85.37 | 1089 | Bills |
| 23 | 23 | Jonah Williams | 84.69 | 77.10 | 85.58 | 1044 | Bengals |
| 24 | 24 | Jason Peters | 84.10 | 77.50 | 84.33 | 853 | Bears |
| 25 | 25 | Joe Noteboom | 83.89 | 76.00 | 84.98 | 174 | Rams |
| 26 | 26 | Garett Bolles | 83.87 | 76.60 | 84.55 | 870 | Broncos |
| 27 | 27 | Sam Cosmi | 83.80 | 74.90 | 85.56 | 474 | Commanders |
| 28 | 28 | Isaiah Wynn | 83.57 | 74.90 | 85.19 | 915 | Patriots |
| 29 | 29 | Terron Armstead | 83.49 | 75.90 | 84.39 | 468 | Saints |
| 30 | 30 | Taylor Decker | 83.47 | 75.50 | 84.61 | 529 | Lions |
| 31 | 31 | Orlando Brown Jr. | 82.76 | 75.40 | 83.50 | 1127 | Chiefs |
| 32 | 32 | Brian O'Neill | 82.61 | 73.40 | 84.58 | 1140 | Vikings |
| 33 | 33 | Cornelius Lucas | 82.54 | 75.20 | 83.27 | 587 | Commanders |
| 34 | 34 | Christian Darrisaw | 81.76 | 71.90 | 84.17 | 652 | Vikings |
| 35 | 35 | Cam Fleming | 81.21 | 71.70 | 83.38 | 285 | Broncos |
| 36 | 36 | Matt Pryor | 81.15 | 76.50 | 80.08 | 438 | Colts |
| 37 | 37 | Ty Nsekhe | 81.13 | 72.50 | 82.71 | 145 | Cowboys |
| 38 | 38 | Dennis Kelly | 81.11 | 70.40 | 84.08 | 305 | Packers |
| 39 | 39 | Morgan Moses | 80.95 | 71.00 | 83.42 | 1022 | Jets |
| 40 | 40 | Duane Brown | 80.61 | 71.50 | 82.51 | 969 | Seahawks |
| 41 | 41 | George Fant | 80.24 | 71.10 | 82.17 | 889 | Jets |
| 42 | 42 | Trey Pipkins III | 80.14 | 68.50 | 83.73 | 173 | Chargers |

### Good (33 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 43 | 1 | Taylor Lewan | 79.89 | 70.80 | 81.79 | 846 | Titans |
| 44 | 2 | Bobby Massie | 79.83 | 70.00 | 82.22 | 796 | Broncos |
| 45 | 3 | Jake Matthews | 79.81 | 71.30 | 81.31 | 1029 | Falcons |
| 46 | 4 | Mike McGlinchey | 79.19 | 69.80 | 81.28 | 466 | 49ers |
| 47 | 5 | James Hurst | 78.95 | 69.80 | 80.89 | 941 | Saints |
| 48 | 6 | D.J. Humphries | 78.89 | 67.80 | 82.12 | 1083 | Cardinals |
| 49 | 7 | Andre Dillard | 78.79 | 69.60 | 80.75 | 340 | Eagles |
| 50 | 8 | Andrew Wylie | 78.72 | 67.20 | 82.24 | 527 | Chiefs |
| 51 | 9 | Eric Fisher | 78.57 | 68.20 | 81.32 | 874 | Colts |
| 52 | 10 | Riley Reiff | 78.16 | 67.30 | 81.24 | 711 | Bengals |
| 53 | 11 | Walker Little | 78.10 | 68.50 | 80.33 | 224 | Jaguars |
| 54 | 12 | Brandon Shell | 77.92 | 67.00 | 81.03 | 550 | Seahawks |
| 55 | 13 | Mike Remmers | 77.50 | 64.50 | 82.00 | 156 | Chiefs |
| 56 | 14 | Chukwuma Okorafor | 77.05 | 63.60 | 81.85 | 1078 | Steelers |
| 57 | 15 | Marcus Cannon | 77.04 | 66.60 | 79.84 | 213 | Texans |
| 58 | 16 | Lucas Niang | 76.68 | 64.60 | 80.57 | 524 | Chiefs |
| 59 | 17 | Terence Steele | 76.59 | 64.50 | 80.48 | 910 | Cowboys |
| 60 | 18 | Spencer Brown | 76.55 | 62.60 | 81.68 | 726 | Bills |
| 61 | 19 | Conor McDermott | 76.55 | 68.40 | 77.81 | 135 | Jets |
| 62 | 20 | Jedrick Wills Jr. | 76.48 | 66.10 | 79.23 | 763 | Browns |
| 63 | 21 | Cam Robinson | 76.43 | 67.40 | 78.29 | 856 | Jaguars |
| 64 | 22 | Billy Turner | 76.34 | 66.20 | 78.93 | 810 | Packers |
| 65 | 23 | Calvin Anderson | 76.21 | 72.50 | 74.51 | 172 | Broncos |
| 66 | 24 | Patrick Mekari | 76.18 | 66.10 | 78.74 | 762 | Ravens |
| 67 | 25 | Alejandro Villanueva | 76.02 | 65.40 | 78.94 | 1205 | Ravens |
| 68 | 26 | Elijah Wilkinson | 75.58 | 65.00 | 78.47 | 120 | Bears |
| 69 | 27 | Kaleb McGary | 75.56 | 62.80 | 79.90 | 986 | Falcons |
| 70 | 28 | Matt Peart | 75.54 | 63.00 | 79.74 | 421 | Giants |
| 71 | 29 | Kelvin Beachum | 75.36 | 63.40 | 79.17 | 950 | Cardinals |
| 72 | 30 | Germain Ifedi | 75.16 | 61.80 | 79.90 | 412 | Bears |
| 73 | 31 | Yosh Nijman | 74.83 | 63.20 | 78.41 | 590 | Packers |
| 74 | 32 | Storm Norton | 74.82 | 60.30 | 80.33 | 1078 | Chargers |
| 75 | 33 | Josh Wells | 74.07 | 60.10 | 79.21 | 124 | Buccaneers |

### Starter (23 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 76 | 1 | Nate Solder | 73.89 | 60.30 | 78.78 | 927 | Giants |
| 77 | 2 | Isaiah Prince | 73.61 | 58.00 | 79.85 | 384 | Bengals |
| 78 | 3 | Larry Borom | 73.31 | 61.40 | 77.09 | 633 | Bears |
| 79 | 4 | Laremy Tunsil | 73.18 | 60.80 | 77.27 | 262 | Texans |
| 80 | 5 | Chuma Edoga | 73.12 | 57.70 | 79.23 | 100 | Jets |
| 81 | 6 | Brady Christensen | 72.79 | 61.60 | 76.08 | 480 | Panthers |
| 82 | 7 | Geron Christian | 72.72 | 59.50 | 77.37 | 588 | Texans |
| 83 | 8 | Jawaan Taylor | 72.36 | 60.40 | 76.16 | 1083 | Jaguars |
| 84 | 9 | Jaylon Moore | 72.25 | 59.00 | 76.92 | 145 | 49ers |
| 85 | 10 | James Hudson III | 71.73 | 57.30 | 77.18 | 303 | Browns |
| 86 | 11 | Justin Herron | 71.18 | 56.70 | 76.66 | 393 | Patriots |
| 87 | 12 | Dan Moore Jr. | 70.58 | 57.80 | 74.93 | 1079 | Steelers |
| 88 | 13 | Cameron Erving | 69.82 | 56.00 | 74.86 | 589 | Panthers |
| 89 | 14 | Charlie Heck | 69.67 | 56.00 | 74.61 | 827 | Texans |
| 90 | 15 | Brandon Parker | 69.27 | 55.80 | 74.09 | 881 | Raiders |
| 91 | 16 | Jake Curhan | 69.14 | 54.00 | 75.07 | 405 | Seahawks |
| 92 | 17 | Tyre Phillips | 68.17 | 53.10 | 74.05 | 389 | Ravens |
| 93 | 18 | Teven Jenkins | 67.21 | 47.50 | 76.18 | 160 | Bears |
| 94 | 19 | Korey Cunningham | 66.85 | 51.70 | 72.78 | 113 | Giants |
| 95 | 20 | Jesse Davis | 66.84 | 52.50 | 72.24 | 1063 | Dolphins |
| 96 | 21 | Matt Nelson | 66.79 | 50.80 | 73.29 | 675 | Lions |
| 97 | 22 | Jordan Mills | 66.37 | 47.80 | 74.58 | 221 | Saints |
| 98 | 23 | Julie'n Davenport | 64.76 | 45.30 | 73.57 | 278 | Colts |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 99 | 1 | Rashod Hill | 61.90 | 42.70 | 70.54 | 342 | Vikings |
| 100 | 2 | Bobby Hart | 56.59 | 33.70 | 67.68 | 102 | Bills |

## TE — Tight End

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (6 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | George Kittle | 87.78 | 90.90 | 81.53 | 460 | 49ers |
| 2 | 2 | Mark Andrews | 87.20 | 91.50 | 80.16 | 669 | Ravens |
| 3 | 3 | Kyle Pitts | 82.93 | 80.30 | 80.51 | 549 | Falcons |
| 4 | 4 | Dallas Goedert | 82.62 | 88.90 | 74.26 | 412 | Eagles |
| 5 | 5 | Rob Gronkowski | 82.40 | 79.50 | 80.17 | 422 | Buccaneers |
| 6 | 6 | Travis Kelce | 82.20 | 81.90 | 78.23 | 672 | Chiefs |

### Good (9 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 7 | 1 | Marcedes Lewis | 77.62 | 78.80 | 72.66 | 206 | Packers |
| 8 | 2 | Darren Waller | 76.55 | 68.20 | 77.95 | 427 | Raiders |
| 9 | 3 | Maxx Williams | 76.25 | 77.90 | 70.98 | 117 | Cardinals |
| 10 | 4 | Dalton Schultz | 76.01 | 78.10 | 70.45 | 634 | Cowboys |
| 11 | 5 | Zach Ertz | 74.75 | 66.90 | 75.82 | 554 | Cardinals |
| 12 | 6 | Pat Freiermuth | 74.43 | 72.30 | 71.68 | 452 | Steelers |
| 13 | 7 | Albert Okwuegbunam | 74.30 | 67.30 | 74.80 | 214 | Broncos |
| 14 | 8 | David Njoku | 74.27 | 70.90 | 72.35 | 372 | Browns |
| 15 | 9 | Hunter Henry | 74.13 | 73.60 | 70.32 | 477 | Patriots |

### Starter (56 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 16 | 1 | Mo Alie-Cox | 73.70 | 66.40 | 74.40 | 303 | Colts |
| 17 | 2 | Jack Doyle | 72.87 | 71.20 | 69.82 | 319 | Colts |
| 18 | 3 | MyCole Pruitt | 72.70 | 71.50 | 69.34 | 190 | Titans |
| 19 | 4 | Brevin Jordan | 72.63 | 66.60 | 72.49 | 161 | Texans |
| 20 | 5 | Tyler Higbee | 72.58 | 67.20 | 72.00 | 559 | Rams |
| 21 | 6 | Zach Gentry | 72.48 | 67.00 | 71.96 | 222 | Steelers |
| 22 | 7 | T.J. Hockenson | 72.42 | 68.10 | 71.14 | 444 | Lions |
| 23 | 8 | Blake Bell | 72.10 | 72.80 | 67.47 | 121 | Chiefs |
| 24 | 9 | Jimmy Graham | 72.06 | 69.70 | 69.47 | 147 | Bears |
| 25 | 10 | John Bates | 71.98 | 70.70 | 68.66 | 263 | Commanders |
| 26 | 11 | Gerald Everett | 71.81 | 63.50 | 73.18 | 408 | Seahawks |
| 27 | 12 | Donald Parham Jr. | 71.78 | 69.20 | 69.34 | 218 | Chargers |
| 28 | 13 | Will Dissly | 71.39 | 63.80 | 72.29 | 257 | Seahawks |
| 29 | 14 | Stephen Anderson | 70.48 | 62.90 | 71.36 | 119 | Chargers |
| 30 | 15 | Mike Gesicki | 70.17 | 68.70 | 66.99 | 587 | Dolphins |
| 31 | 16 | James O'Shaughnessy | 70.15 | 62.80 | 70.88 | 193 | Jaguars |
| 32 | 17 | Blake Jarwin | 69.92 | 59.00 | 73.03 | 158 | Cowboys |
| 33 | 18 | Tyler Conklin | 69.65 | 66.70 | 67.45 | 598 | Vikings |
| 34 | 19 | Harrison Bryant | 69.60 | 64.80 | 68.64 | 167 | Browns |
| 35 | 20 | Dan Arnold | 69.54 | 62.50 | 70.07 | 279 | Jaguars |
| 36 | 21 | Ricky Seals-Jones | 69.31 | 62.30 | 69.81 | 313 | Commanders |
| 37 | 22 | Kyle Rudolph | 69.21 | 63.50 | 68.85 | 297 | Giants |
| 38 | 23 | Kylen Granson | 69.04 | 58.70 | 71.77 | 108 | Colts |
| 39 | 24 | Jared Cook | 68.63 | 61.90 | 68.95 | 532 | Chargers |
| 40 | 25 | Austin Hooper | 68.55 | 64.20 | 67.28 | 376 | Browns |
| 41 | 26 | Jesse James | 68.41 | 59.40 | 70.25 | 136 | Bears |
| 42 | 27 | Logan Thomas | 68.32 | 63.40 | 67.43 | 172 | Commanders |
| 43 | 28 | Noah Fant | 68.20 | 61.60 | 68.44 | 488 | Broncos |
| 44 | 29 | Adam Trautman | 67.91 | 62.70 | 67.22 | 333 | Saints |
| 45 | 30 | Lee Smith | 67.91 | 70.10 | 62.28 | 121 | Falcons |
| 46 | 31 | Eric Saubert | 67.89 | 67.80 | 63.79 | 112 | Broncos |
| 47 | 32 | Nick Vannett | 67.57 | 57.90 | 69.85 | 112 | Saints |
| 48 | 33 | C.J. Uzomah | 67.53 | 62.00 | 67.05 | 498 | Bengals |
| 49 | 34 | Chris Manhertz | 67.18 | 62.60 | 66.07 | 173 | Jaguars |
| 50 | 35 | Foster Moreau | 67.13 | 58.80 | 68.52 | 424 | Raiders |
| 51 | 36 | Durham Smythe | 66.96 | 59.50 | 67.76 | 375 | Dolphins |
| 52 | 37 | Dawson Knox | 66.80 | 62.40 | 65.56 | 541 | Bills |
| 53 | 38 | Adam Shaheen | 66.80 | 59.30 | 67.64 | 204 | Dolphins |
| 54 | 39 | Jonnu Smith | 66.66 | 59.30 | 67.40 | 196 | Patriots |
| 55 | 40 | Cole Kmet | 66.23 | 63.40 | 63.95 | 583 | Bears |
| 56 | 41 | Chris Herndon | 66.03 | 52.60 | 70.81 | 104 | Vikings |
| 57 | 42 | Drew Sample | 65.31 | 59.00 | 65.35 | 167 | Bengals |
| 58 | 43 | Eric Ebron | 65.22 | 48.40 | 72.26 | 184 | Steelers |
| 59 | 44 | Cameron Brate | 64.94 | 57.00 | 66.07 | 323 | Buccaneers |
| 60 | 45 | O.J. Howard | 64.82 | 49.00 | 71.20 | 177 | Buccaneers |
| 61 | 46 | Robert Tonyan | 64.67 | 54.60 | 67.21 | 211 | Packers |
| 62 | 47 | Tyler Kroft | 64.34 | 57.60 | 64.66 | 226 | Jets |
| 63 | 48 | Josiah Deguara | 64.29 | 58.00 | 64.32 | 228 | Packers |
| 64 | 49 | Evan Engram | 64.19 | 53.50 | 67.15 | 508 | Giants |
| 65 | 50 | Anthony Firkser | 64.15 | 54.80 | 66.22 | 287 | Titans |
| 66 | 51 | Hayden Hurst | 64.08 | 55.30 | 65.77 | 266 | Falcons |
| 67 | 52 | Ryan Griffin | 63.62 | 55.80 | 64.67 | 341 | Jets |
| 68 | 53 | Geoff Swaim | 62.90 | 55.50 | 63.67 | 278 | Titans |
| 69 | 54 | Jordan Akins | 62.73 | 52.50 | 65.38 | 233 | Texans |
| 70 | 55 | Brock Wright | 62.35 | 53.50 | 64.09 | 118 | Lions |
| 71 | 56 | Antony Auclair | 62.03 | 55.70 | 62.09 | 100 | Texans |

### Rotation/backup (7 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 72 | 1 | Jacob Hollister | 61.97 | 51.00 | 65.12 | 106 | Jaguars |
| 73 | 2 | Tommy Tremble | 61.31 | 55.20 | 61.21 | 306 | Panthers |
| 74 | 3 | Noah Gray | 60.40 | 50.10 | 63.10 | 157 | Chiefs |
| 75 | 4 | Ian Thomas | 60.01 | 52.10 | 61.11 | 401 | Panthers |
| 76 | 5 | Pharaoh Brown | 59.49 | 48.00 | 62.99 | 277 | Texans |
| 77 | 6 | Luke Farrell | 58.84 | 48.00 | 61.90 | 120 | Jaguars |
| 78 | 7 | Tommy Sweeney | 58.43 | 44.40 | 63.62 | 131 | Bills |

## WR — Wide Receiver

- **Season used:** `2021`
- **PFF column (model anchor):** `grades_offense` · **Snap column (volume filter):** `total_snaps`

### Elite (19 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 1 | 1 | Davante Adams | 88.68 | 92.70 | 81.84 | 583 | Packers |
| 2 | 2 | Cooper Kupp | 88.21 | 92.30 | 81.32 | 649 | Rams |
| 3 | 3 | Justin Jefferson | 88.03 | 90.10 | 82.49 | 658 | Vikings |
| 4 | 4 | Deebo Samuel | 87.14 | 87.20 | 82.94 | 490 | 49ers |
| 5 | 5 | Deonte Harty | 86.65 | 87.80 | 81.72 | 214 | Saints |
| 6 | 6 | Ja'Marr Chase | 86.60 | 83.10 | 84.77 | 613 | Bengals |
| 7 | 7 | A.J. Brown | 85.18 | 84.40 | 81.54 | 367 | Titans |
| 8 | 8 | Tee Higgins | 83.17 | 84.50 | 78.12 | 512 | Bengals |
| 9 | 9 | Tyreek Hill | 82.03 | 85.10 | 75.82 | 629 | Chiefs |
| 10 | 10 | Chris Godwin | 81.93 | 81.30 | 78.19 | 582 | Buccaneers |
| 11 | 11 | Lil'Jordan Humphrey | 81.87 | 71.10 | 84.89 | 142 | Saints |
| 12 | 12 | CeeDee Lamb | 81.80 | 84.30 | 75.97 | 582 | Cowboys |
| 13 | 13 | Stefon Diggs | 81.68 | 82.10 | 77.23 | 682 | Bills |
| 14 | 14 | Tyler Lockett | 81.35 | 81.00 | 77.42 | 530 | Seahawks |
| 15 | 15 | D.K. Metcalf | 81.10 | 80.90 | 77.06 | 530 | Seahawks |
| 16 | 16 | DeAndre Hopkins | 80.79 | 79.80 | 77.29 | 343 | Cardinals |
| 17 | 17 | Terry McLaurin | 80.67 | 78.30 | 78.09 | 621 | Commanders |
| 18 | 18 | DeSean Jackson | 80.65 | 68.90 | 84.32 | 207 | Raiders |
| 19 | 19 | Mike Williams | 80.59 | 77.60 | 78.42 | 620 | Chargers |

### Good (49 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 20 | 1 | DJ Moore | 79.75 | 76.90 | 77.48 | 647 | Panthers |
| 21 | 2 | Julio Jones | 79.71 | 74.20 | 79.22 | 256 | Titans |
| 22 | 3 | DeVonta Smith | 79.51 | 77.20 | 76.88 | 545 | Eagles |
| 23 | 4 | Hunter Renfrow | 79.46 | 80.60 | 74.54 | 570 | Raiders |
| 24 | 5 | Kendrick Bourne | 79.18 | 75.60 | 77.40 | 424 | Patriots |
| 25 | 6 | Brandin Cooks | 78.81 | 77.40 | 75.58 | 563 | Texans |
| 26 | 7 | Michael Pittman Jr. | 78.66 | 78.00 | 74.93 | 602 | Colts |
| 27 | 8 | Amon-Ra St. Brown | 78.49 | 79.40 | 73.71 | 541 | Lions |
| 28 | 9 | Brandon Aiyuk | 78.33 | 74.40 | 76.79 | 517 | 49ers |
| 29 | 10 | Quintez Cephus | 78.09 | 71.70 | 78.18 | 141 | Lions |
| 30 | 11 | Quez Watkins | 77.96 | 70.50 | 78.77 | 435 | Eagles |
| 31 | 12 | Robert Woods | 77.59 | 75.70 | 74.69 | 338 | Rams |
| 32 | 13 | Mike Evans | 77.24 | 73.20 | 75.76 | 654 | Buccaneers |
| 33 | 14 | Cedrick Wilson Jr. | 77.15 | 73.80 | 75.22 | 378 | Cowboys |
| 34 | 15 | Amari Cooper | 77.10 | 72.90 | 75.74 | 566 | Cowboys |
| 35 | 16 | Darnell Mooney | 77.06 | 74.70 | 74.47 | 646 | Bears |
| 36 | 17 | Courtland Sutton | 77.05 | 71.20 | 76.78 | 574 | Broncos |
| 37 | 18 | Keenan Allen | 77.00 | 77.50 | 72.50 | 683 | Chargers |
| 38 | 19 | Gabe Davis | 76.93 | 73.70 | 74.92 | 363 | Bills |
| 39 | 20 | Laquon Treadwell | 76.69 | 72.00 | 75.65 | 310 | Jaguars |
| 40 | 21 | Kadarius Toney | 76.59 | 72.80 | 74.95 | 216 | Giants |
| 41 | 22 | Elijah Moore | 76.57 | 71.20 | 75.98 | 327 | Jets |
| 42 | 23 | Kenny Golladay | 76.54 | 68.30 | 77.86 | 449 | Giants |
| 43 | 24 | Randall Cobb | 76.52 | 72.00 | 75.37 | 267 | Packers |
| 44 | 25 | Adam Thielen | 76.46 | 74.60 | 73.53 | 472 | Vikings |
| 45 | 26 | Tyler Boyd | 76.43 | 72.60 | 74.81 | 562 | Bengals |
| 46 | 27 | Marquez Valdes-Scantling | 76.27 | 66.10 | 78.89 | 324 | Packers |
| 47 | 28 | John Ross | 76.19 | 67.00 | 78.15 | 154 | Giants |
| 48 | 29 | Tim Patrick | 76.18 | 71.20 | 75.33 | 526 | Broncos |
| 49 | 30 | DeVante Parker | 76.13 | 72.80 | 74.19 | 370 | Dolphins |
| 50 | 31 | Corey Davis | 76.10 | 68.80 | 76.80 | 301 | Jets |
| 51 | 32 | Michael Gallup | 75.94 | 73.40 | 73.46 | 350 | Cowboys |
| 52 | 33 | Jakobi Meyers | 75.92 | 74.90 | 72.43 | 572 | Patriots |
| 53 | 34 | T.Y. Hilton | 75.89 | 70.10 | 75.58 | 238 | Colts |
| 54 | 35 | Braxton Berrios | 75.85 | 74.00 | 72.92 | 276 | Jets |
| 55 | 36 | Diontae Johnson | 75.73 | 74.20 | 72.59 | 659 | Steelers |
| 56 | 37 | Chase Claypool | 75.71 | 67.70 | 76.88 | 539 | Steelers |
| 57 | 38 | Jaylen Waddle | 75.69 | 78.30 | 69.79 | 613 | Dolphins |
| 58 | 39 | Mecole Hardman Jr. | 75.53 | 68.50 | 76.05 | 451 | Chiefs |
| 59 | 40 | Russell Gage | 75.22 | 75.20 | 71.06 | 406 | Falcons |
| 60 | 41 | Donovan Peoples-Jones | 75.21 | 65.70 | 77.38 | 434 | Browns |
| 61 | 42 | Christian Kirk | 75.05 | 72.70 | 72.45 | 579 | Cardinals |
| 62 | 43 | Jerry Jeudy | 74.87 | 68.70 | 74.81 | 271 | Broncos |
| 63 | 44 | Bryan Edwards | 74.86 | 63.80 | 78.07 | 541 | Raiders |
| 64 | 45 | A.J. Green | 74.83 | 69.20 | 74.42 | 564 | Cardinals |
| 65 | 46 | Jarvis Landry | 74.82 | 66.30 | 76.33 | 331 | Browns |
| 66 | 47 | Marquez Callaway | 74.82 | 69.30 | 74.33 | 481 | Saints |
| 67 | 48 | Brandon Zylstra | 74.33 | 63.80 | 77.19 | 192 | Panthers |
| 68 | 49 | Nick Westbrook-Ikhine | 74.11 | 68.70 | 73.55 | 359 | Titans |

### Starter (80 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 69 | 1 | Rondale Moore | 73.87 | 71.20 | 71.48 | 265 | Cardinals |
| 70 | 2 | DJ Chark Jr. | 73.83 | 65.90 | 74.95 | 119 | Jaguars |
| 71 | 3 | Chris Moore | 73.75 | 68.30 | 73.22 | 139 | Texans |
| 72 | 4 | Calvin Ridley | 73.69 | 64.20 | 75.85 | 209 | Falcons |
| 73 | 5 | Allen Robinson II | 73.60 | 66.90 | 73.90 | 386 | Bears |
| 74 | 6 | Marquise Goodwin | 73.59 | 62.50 | 76.81 | 256 | Bears |
| 75 | 7 | Marquise Brown | 73.58 | 68.60 | 72.74 | 657 | Ravens |
| 76 | 8 | Odell Beckham Jr. | 73.53 | 67.80 | 73.18 | 435 | Rams |
| 77 | 9 | Breshad Perriman | 73.28 | 60.80 | 77.44 | 121 | Buccaneers |
| 78 | 10 | Emmanuel Sanders | 73.11 | 65.60 | 73.95 | 539 | Bills |
| 79 | 11 | Nelson Agholor | 72.77 | 65.90 | 73.19 | 434 | Patriots |
| 80 | 12 | Zay Jones | 72.73 | 70.20 | 70.25 | 421 | Raiders |
| 81 | 13 | Josh Reynolds | 72.71 | 65.90 | 73.08 | 291 | Lions |
| 82 | 14 | Kendall Hinton | 72.61 | 66.50 | 72.51 | 169 | Broncos |
| 83 | 15 | Nico Collins | 72.49 | 65.60 | 72.91 | 383 | Texans |
| 84 | 16 | K.J. Osborn | 72.40 | 64.90 | 73.23 | 536 | Vikings |
| 85 | 17 | KhaDarel Hodge | 72.40 | 68.10 | 71.10 | 166 | Lions |
| 86 | 18 | Marvin Jones Jr. | 72.38 | 68.30 | 70.93 | 667 | Jaguars |
| 87 | 19 | Sammy Watkins | 72.24 | 66.30 | 72.03 | 287 | Ravens |
| 88 | 20 | Danny Amendola | 72.12 | 63.10 | 73.96 | 162 | Texans |
| 89 | 21 | Marcus Johnson | 72.07 | 63.00 | 73.95 | 101 | Titans |
| 90 | 22 | Byron Pringle | 72.04 | 66.50 | 71.56 | 432 | Chiefs |
| 91 | 23 | Allen Lazard | 71.95 | 65.50 | 72.09 | 455 | Packers |
| 92 | 24 | Cole Beasley | 71.92 | 66.60 | 71.30 | 531 | Bills |
| 93 | 25 | Isaiah McKenzie | 71.81 | 69.40 | 69.25 | 150 | Bills |
| 94 | 26 | James Proche II | 71.62 | 66.90 | 70.60 | 176 | Ravens |
| 95 | 27 | Cam Sims | 71.48 | 62.60 | 73.23 | 182 | Commanders |
| 96 | 28 | Jamison Crowder | 70.71 | 64.90 | 70.42 | 418 | Jets |
| 97 | 29 | Equanimeous St. Brown | 70.68 | 57.40 | 75.36 | 155 | Packers |
| 98 | 30 | N'Keal Harry | 70.35 | 69.10 | 67.01 | 150 | Patriots |
| 99 | 31 | Collin Johnson | 70.30 | 62.30 | 71.47 | 141 | Giants |
| 100 | 32 | Olamide Zaccheaus | 70.21 | 63.40 | 70.59 | 400 | Falcons |
| 101 | 33 | Keelan Cole Sr. | 70.17 | 61.30 | 71.92 | 367 | Jets |
| 102 | 34 | Chris Conley | 70.07 | 60.50 | 72.28 | 364 | Texans |
| 103 | 35 | Sterling Shepard | 70.05 | 64.60 | 69.51 | 236 | Giants |
| 104 | 36 | Kalif Raymond | 70.03 | 60.50 | 72.22 | 496 | Lions |
| 105 | 37 | Rashod Bateman | 70.02 | 64.90 | 69.27 | 432 | Ravens |
| 106 | 38 | Van Jefferson | 69.96 | 59.80 | 72.56 | 580 | Rams |
| 107 | 39 | Tre'Quan Smith | 69.86 | 62.70 | 70.47 | 323 | Saints |
| 108 | 40 | Ashton Dulin | 69.35 | 59.90 | 71.48 | 129 | Colts |
| 109 | 41 | Joshua Palmer | 69.15 | 62.60 | 69.35 | 314 | Chargers |
| 110 | 42 | Scott Miller | 69.03 | 54.80 | 74.35 | 104 | Buccaneers |
| 111 | 43 | Parris Campbell | 68.92 | 62.50 | 69.03 | 130 | Colts |
| 112 | 44 | Jauan Jennings | 68.89 | 65.80 | 66.79 | 211 | 49ers |
| 113 | 45 | DeAndre Carter | 68.82 | 63.70 | 68.06 | 281 | Commanders |
| 114 | 46 | Laviska Shenault Jr. | 68.82 | 63.70 | 68.07 | 481 | Jaguars |
| 115 | 47 | Preston Williams | 68.80 | 59.20 | 71.04 | 108 | Dolphins |
| 116 | 48 | Jalen Guyton | 68.15 | 57.40 | 71.15 | 461 | Chargers |
| 117 | 49 | Jamal Agnew | 67.84 | 65.60 | 65.16 | 192 | Jaguars |
| 118 | 50 | Kenny Stills | 67.56 | 54.70 | 71.96 | 167 | Saints |
| 119 | 51 | Freddie Swain | 67.44 | 50.30 | 74.70 | 396 | Seahawks |
| 120 | 52 | JuJu Smith-Schuster | 67.33 | 59.20 | 68.59 | 152 | Steelers |
| 121 | 53 | Antoine Wesley | 67.07 | 60.20 | 67.49 | 263 | Cardinals |
| 122 | 54 | Noah Brown | 66.80 | 60.60 | 66.76 | 173 | Cowboys |
| 123 | 55 | Dyami Brown | 66.73 | 55.50 | 70.05 | 213 | Commanders |
| 124 | 56 | Albert Wilson | 66.37 | 59.30 | 66.91 | 238 | Dolphins |
| 125 | 57 | Anthony Schwartz | 66.30 | 57.40 | 68.07 | 169 | Browns |
| 126 | 58 | Mohamed Sanu | 66.24 | 63.00 | 64.23 | 184 | 49ers |
| 127 | 59 | Tavon Austin | 66.24 | 60.20 | 66.10 | 211 | Jaguars |
| 128 | 60 | Darius Slayton | 66.24 | 52.80 | 71.03 | 382 | Giants |
| 129 | 61 | Rashard Higgins | 66.22 | 54.70 | 69.73 | 313 | Browns |
| 130 | 62 | Denzel Mims | 65.85 | 48.70 | 73.12 | 199 | Jets |
| 131 | 63 | Mack Hollins | 65.77 | 58.10 | 66.71 | 229 | Dolphins |
| 132 | 64 | Adam Humphries | 65.75 | 58.10 | 66.69 | 468 | Commanders |
| 133 | 65 | Zach Pascal | 65.69 | 52.70 | 70.18 | 532 | Colts |
| 134 | 66 | Damiere Byrd | 65.67 | 53.80 | 69.42 | 406 | Bears |
| 135 | 67 | James Washington | 65.67 | 50.50 | 71.62 | 357 | Steelers |
| 136 | 68 | Ben Skowronek | 65.25 | 57.20 | 66.45 | 103 | Rams |
| 137 | 69 | Jalen Reagor | 64.88 | 56.30 | 66.43 | 463 | Eagles |
| 138 | 70 | D'Wayne Eskridge | 64.69 | 58.30 | 64.79 | 112 | Seahawks |
| 139 | 71 | Tyler Johnson | 64.54 | 55.40 | 66.46 | 408 | Buccaneers |
| 140 | 72 | Trent Sherfield | 64.53 | 58.10 | 64.65 | 116 | 49ers |
| 141 | 73 | Tajae Sharpe | 64.27 | 54.40 | 66.69 | 341 | Falcons |
| 142 | 74 | Chester Rogers | 64.06 | 57.40 | 64.34 | 328 | Titans |
| 143 | 75 | Dede Westbrook | 63.77 | 53.10 | 66.72 | 172 | Vikings |
| 144 | 76 | Jeff Smith | 63.65 | 57.90 | 63.31 | 141 | Jets |
| 145 | 77 | Devin Duvernay | 63.58 | 55.70 | 64.67 | 405 | Ravens |
| 146 | 78 | Demarcus Robinson | 63.51 | 51.10 | 67.62 | 509 | Chiefs |
| 147 | 79 | Greg Ward | 62.75 | 54.90 | 63.82 | 150 | Eagles |
| 148 | 80 | Ray-Ray McCloud III | 62.03 | 56.80 | 61.35 | 366 | Steelers |

### Rotation/backup (2 players)

| rank_pos | rank_in_tier | player | composite | model | stats | snaps | primary_team |
|---:|---:|---|---:|---:|---:|---:|---|
| 149 | 1 | Terrace Marshall Jr. | 60.41 | 53.30 | 60.98 | 291 | Panthers |
| 150 | 2 | Trinity Benson | 58.41 | 46.90 | 61.92 | 191 | Lions |
