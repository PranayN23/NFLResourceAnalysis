# Free Agency — position rankings by tier

- **Regenerate:** from repo root, `PYTHONPATH=. python backend/agent/export_fa_tier_rankings.py [--analysis-year 2025] [--min-snaps 100]`
- **Generated (UTC):** 2026-04-10 19:38:11Z
- **Requested analysis_year:** 2025 (clamped to 2025)
- **Tier cutoffs (same as FA UI):** Elite ≥80 · Good ≥74 · Starter ≥62 · else Rotation/backup
- **Season (`Year` in ML CSVs):** per position, latest season with data such that `Year ≤ analysis_year` (see each section; if 2025 rows are absent, this is **2024**).
- **Eligibility:** snap-weighted grade from listed snap column; players with **total snaps < 100** in that season are omitted.

## C — Center

- **Season used:** `2024`
- **Grade column:** `grades_offense` · **Snap column:** `snap_counts_offense`

### Elite (3 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | Erik McCoy | 94.20 | 293 | Saints |
| 2 | 2 | Creed Humphrey | 92.30 | 1232 | Chiefs |
| 3 | 3 | Frank Ragnow | 86.10 | 1129 | Lions |

### Good (4 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 4 | 1 | Tyler Linderbaum | 79.90 | 1227 | Ravens |
| 5 | 2 | Drew Dalman | 78.80 | 554 | Falcons |
| 6 | 3 | Zach Frazier | 77.90 | 1021 | Steelers |
| 7 | 4 | Hjalte Froholdt | 76.10 | 1078 | Cardinals |

### Starter (20 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 8 | 1 | Joe Tippmann | 73.40 | 1067 | Jets |
| 9 | 2 | Aaron Brewer | 73.30 | 1139 | Dolphins |
| 10 | 3 | Connor McGovern | 69.50 | 1164 | Bills |
| 11 | 4 | Danny Pinter | 68.60 | 138 | Colts |
| 12 | 5 | Cam Jurgens | 67.30 | 1217 | Eagles |
| 13 | 6 | Ryan Kelly | 67.00 | 601 | Colts |
| 14 | 7 | Alex Forsyth | 66.50 | 292 | Broncos |
| 15 | 8 | Coleman Shelton | 66.40 | 1121 | Bears |
| 16 | 9 | Cooper Beebe | 65.40 | 1059 | Cowboys |
| 17 | 10 | Jake Brendel | 65.00 | 1072 | 49ers |
| 18 | 11 | Luke Wattenberg | 64.30 | 864 | Broncos |
| 19 | 12 | Tyler Biadasz | 64.20 | 1166 | Commanders |
| 20 | 13 | Olusegun Oluwatimi | 64.20 | 435 | Seahawks |
| 21 | 14 | Ted Karras | 64.10 | 1136 | Bengals |
| 22 | 15 | Jarrett Patterson | 64.10 | 688 | Texans |
| 23 | 16 | Brady Christensen | 63.60 | 399 | Panthers |
| 24 | 17 | Ethan Pocic | 63.60 | 1073 | Browns |
| 25 | 18 | Juice Scruggs | 63.00 | 944 | Texans |
| 26 | 19 | Austin Corbett | 62.90 | 291 | Panthers |
| 27 | 20 | Garrett Bradbury | 62.80 | 1191 | Vikings |

### Rotation/backup (15 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 28 | 1 | John Michael Schmitz Jr. | 61.40 | 983 | Giants |
| 29 | 2 | Bradley Bozeman | 61.20 | 1112 | Chargers |
| 30 | 3 | David Andrews | 58.70 | 193 | Patriots |
| 31 | 4 | Ryan Neuzil | 58.50 | 578 | Falcons |
| 32 | 5 | Mitch Morse | 57.30 | 1021 | Jaguars |
| 33 | 6 | Andre James | 55.60 | 702 | Raiders |
| 34 | 7 | Graham Barton | 55.60 | 1111 | Buccaneers |
| 35 | 8 | Beaux Limmer | 55.50 | 1040 | Rams |
| 36 | 9 | Corey Levin | 55.50 | 133 | Titans |
| 37 | 10 | Lloyd Cushenberry III | 55.40 | 499 | Titans |
| 38 | 11 | Daniel Brunskill | 55.30 | 684 | Titans |
| 39 | 12 | Sedrick Van Pran-Granger | 54.60 | 125 | Bills |
| 40 | 13 | Josh Myers | 54.20 | 1067 | Packers |
| 41 | 14 | Shane Lemieux | 51.10 | 337 | Saints |
| 42 | 15 | Ryan McCollum | 50.30 | 153 | Steelers |

## CB — Cornerback

- **Season used:** `2024`
- **Grade column:** `grades_defense` · **Snap column:** `snap_counts_defense`

### Elite (5 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | Pat Surtain II | 83.80 | 1054 | Broncos |
| 2 | 2 | Trent McDuffie | 83.40 | 1132 | Chiefs |
| 3 | 3 | Cooper DeJean | 82.00 | 830 | Eagles |
| 4 | 4 | Garrett Williams | 82.00 | 778 | Cardinals |
| 5 | 5 | Derek Stingley Jr. | 80.40 | 1119 | Texans |

### Good (19 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 6 | 1 | Christian Benford | 79.30 | 1046 | Bills |
| 7 | 2 | Marlon Humphrey | 79.00 | 1000 | Ravens |
| 8 | 3 | Terell Smith | 78.50 | 207 | Bears |
| 9 | 4 | Quinyon Mitchell | 78.40 | 1104 | Eagles |
| 10 | 5 | Andru Phillips | 77.50 | 614 | Giants |
| 11 | 6 | Isaiah Bolden | 76.90 | 141 | Patriots |
| 12 | 7 | Jalen Ramsey | 76.90 | 1027 | Dolphins |
| 13 | 8 | Jaylon Johnson | 76.20 | 1031 | Bears |
| 14 | 9 | Devon Witherspoon | 76.10 | 1103 | Seahawks |
| 15 | 10 | Kyler Gordon | 76.00 | 724 | Bears |
| 16 | 11 | Christian Gonzalez | 76.00 | 978 | Patriots |
| 17 | 12 | Mike Hilton | 75.90 | 737 | Bengals |
| 18 | 13 | Jamel Dean | 75.70 | 745 | Buccaneers |
| 19 | 14 | Tarheeb Still | 75.20 | 826 | Chargers |
| 20 | 15 | Jaire Alexander | 75.20 | 361 | Packers |
| 21 | 16 | Kris Abrams-Draine | 75.10 | 123 | Broncos |
| 22 | 17 | Clark Phillips III | 74.90 | 409 | Falcons |
| 23 | 18 | Kamari Lassiter | 74.70 | 906 | Texans |
| 24 | 19 | Carlton Davis III | 74.50 | 697 | Lions |

### Starter (56 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 25 | 1 | Byron Murphy Jr. | 73.40 | 1109 | Vikings |
| 26 | 2 | Nate Wiggins | 72.60 | 769 | Ravens |
| 27 | 3 | Darius Slay | 72.00 | 897 | Eagles |
| 28 | 4 | Mike Hughes | 71.90 | 720 | Falcons |
| 29 | 5 | Samuel Womack III | 71.90 | 673 | Colts |
| 30 | 6 | Jourdan Lewis | 71.70 | 871 | Cowboys |
| 31 | 7 | Isaiah Rodgers | 71.50 | 413 | Eagles |
| 32 | 8 | Jaylen Watson | 71.00 | 433 | Chiefs |
| 33 | 9 | Kenny Moore II | 70.80 | 1013 | Colts |
| 34 | 10 | D.J. Reed | 70.70 | 880 | Jets |
| 35 | 11 | Marcus Jones | 70.20 | 586 | Patriots |
| 36 | 12 | Sauce Gardner | 70.20 | 879 | Jets |
| 37 | 13 | DaRon Bland | 70.10 | 436 | Cowboys |
| 38 | 14 | A.J. Terrell | 69.40 | 1085 | Falcons |
| 39 | 15 | Cory Trice Jr. | 69.40 | 194 | Steelers |
| 40 | 16 | Renardo Green | 69.20 | 675 | 49ers |
| 41 | 17 | Adoree' Jackson | 69.00 | 426 | Giants |
| 42 | 18 | Jarrian Jones | 69.00 | 699 | Jaguars |
| 43 | 19 | Kristian Fulton | 68.90 | 827 | Chargers |
| 44 | 20 | Denzel Ward | 68.40 | 757 | Browns |
| 45 | 21 | Dax Hill | 68.20 | 262 | Bengals |
| 46 | 22 | Kelee Ringo | 68.10 | 127 | Eagles |
| 47 | 23 | Chamarri Conner | 68.00 | 679 | Chiefs |
| 48 | 24 | Mike Jackson | 68.00 | 1204 | Panthers |
| 49 | 25 | Tariq Woolen | 67.90 | 889 | Seahawks |
| 50 | 26 | DJ Turner II | 67.80 | 508 | Bengals |
| 51 | 27 | Jaylon Jones | 67.40 | 1146 | Colts |
| 52 | 28 | Zyon McCollum | 67.40 | 1123 | Buccaneers |
| 53 | 29 | Deommodore Lenoir | 67.30 | 922 | 49ers |
| 54 | 30 | Kendall Fuller | 66.20 | 556 | Dolphins |
| 55 | 31 | Kool-Aid McKinstry | 66.10 | 680 | Saints |
| 56 | 32 | Mike Sainristil | 65.80 | 1158 | Commanders |
| 57 | 33 | Ahkello Witherspoon | 65.60 | 598 | Rams |
| 58 | 34 | D'Angelo Ross | 65.60 | 184 | Texans |
| 59 | 35 | Carrington Valentine | 65.50 | 606 | Packers |
| 60 | 36 | Darrell Baker Jr. | 65.50 | 626 | Titans |
| 61 | 37 | Myles Bryant | 65.00 | 156 | Texans |
| 62 | 38 | Cobie Durant | 64.90 | 843 | Rams |
| 63 | 39 | Amik Robertson | 64.70 | 630 | Lions |
| 64 | 40 | Kaiir Elam | 64.50 | 359 | Bills |
| 65 | 41 | Kader Kohou | 64.50 | 708 | Dolphins |
| 66 | 42 | Jaycee Horn | 64.50 | 1034 | Panthers |
| 67 | 43 | Chidobe Awuzie | 64.40 | 373 | Titans |
| 68 | 44 | Joshua Williams | 64.30 | 411 | Chiefs |
| 69 | 45 | Keisean Nixon | 64.00 | 1077 | Packers |
| 70 | 46 | Shaquill Griffin | 63.70 | 597 | Vikings |
| 71 | 47 | Cam Taylor-Britt | 63.60 | 1036 | Bengals |
| 72 | 48 | Paulson Adebo | 63.30 | 436 | Saints |
| 73 | 49 | Fabian Moreau | 63.20 | 104 | Vikings |
| 74 | 50 | Troy Hill | 63.20 | 236 | Buccaneers |
| 75 | 51 | Greg Stroman Jr. | 63.10 | 130 | Giants |
| 76 | 52 | Tyson Campbell | 62.90 | 767 | Jaguars |
| 77 | 53 | Montaric Brown | 62.30 | 855 | Jaguars |
| 78 | 54 | Christian Roland-Wallace | 62.30 | 197 | Chiefs |
| 79 | 55 | Cor'Dale Flott | 62.20 | 666 | Giants |
| 80 | 56 | Stephon Gilmore | 62.20 | 904 | Vikings |

### Rotation/backup (72 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 81 | 1 | Ja'Quan McMillian | 61.50 | 918 | Broncos |
| 82 | 2 | Nate Hobbs | 61.40 | 554 | Raiders |
| 83 | 3 | Eric Stokes | 61.30 | 588 | Packers |
| 84 | 4 | Roger McCreary | 61.30 | 652 | Titans |
| 85 | 5 | Alex Austin | 61.10 | 234 | Patriots |
| 86 | 6 | Starling Thomas V | 60.90 | 817 | Cardinals |
| 87 | 7 | Josh Newton | 60.90 | 504 | Bengals |
| 88 | 8 | Jonathan Jones | 60.70 | 712 | Patriots |
| 89 | 9 | Ronald Darby | 60.30 | 659 | Jaguars |
| 90 | 10 | Amani Oruwariye | 60.00 | 286 | Cowboys |
| 91 | 11 | Avonte Maddox | 59.80 | 352 | Eagles |
| 92 | 12 | Isaac Yiadom | 59.50 | 488 | 49ers |
| 93 | 13 | Jarvis Brownlee Jr. | 59.40 | 911 | Titans |
| 94 | 14 | Asante Samuel Jr. | 59.30 | 234 | Chargers |
| 95 | 15 | Darious Williams | 59.00 | 865 | Rams |
| 96 | 16 | Tyrique Stevenson | 58.90 | 810 | Bears |
| 97 | 17 | Taron Johnson | 58.80 | 785 | Bills |
| 98 | 18 | Jakorian Bennett | 58.60 | 459 | Raiders |
| 99 | 19 | Cam Hart | 58.60 | 502 | Chargers |
| 100 | 20 | Dee Alford | 58.20 | 724 | Falcons |
| 101 | 21 | Josh Blackwell | 58.10 | 102 | Bears |
| 102 | 22 | Riley Moss | 57.80 | 912 | Broncos |
| 103 | 23 | Marshon Lattimore | 56.80 | 687 | Commanders |
| 104 | 24 | Ja'Sir Taylor | 56.70 | 353 | Chargers |
| 105 | 25 | Trevon Diggs | 56.60 | 683 | Cowboys |
| 106 | 26 | Sean Murphy-Bunting | 56.50 | 725 | Cardinals |
| 107 | 27 | James Pierre | 56.20 | 207 | Steelers |
| 108 | 28 | Charvarius Ward | 56.20 | 694 | 49ers |
| 109 | 29 | Caleb Farley | 56.10 | 169 | Panthers |
| 110 | 30 | Brandin Echols | 56.00 | 406 | Jets |
| 111 | 31 | Ka'dar Hollman | 55.90 | 116 | Texans |
| 112 | 32 | Beanie Bishop Jr. | 55.60 | 550 | Steelers |
| 113 | 33 | Max Melton | 55.50 | 565 | Cardinals |
| 114 | 34 | Storm Duck | 55.50 | 359 | Dolphins |
| 115 | 35 | Joey Porter Jr. | 54.60 | 1038 | Steelers |
| 116 | 36 | Darnay Holmes | 54.10 | 298 | Raiders |
| 117 | 37 | Nazeeh Johnson | 53.90 | 547 | Chiefs |
| 118 | 38 | Rasul Douglas | 53.90 | 997 | Bills |
| 119 | 39 | Jack Jones | 53.90 | 1047 | Raiders |
| 120 | 40 | Tyrek Funderburk | 53.90 | 168 | Buccaneers |
| 121 | 41 | Ja'Marcus Ingram | 53.90 | 217 | Bills |
| 122 | 42 | Tre'Davious White | 53.60 | 445 | Ravens |
| 123 | 43 | Tre Brown | 53.50 | 290 | Seahawks |
| 124 | 44 | Shemar Jean-Charles | 53.40 | 143 | Saints |
| 125 | 45 | Cameron Mitchell | 52.90 | 371 | Browns |
| 126 | 46 | Greg Newsome II | 52.20 | 571 | Browns |
| 127 | 47 | Israel Mukuamu | 51.70 | 201 | Cowboys |
| 128 | 48 | Chau Smith-Wade | 51.60 | 301 | Panthers |
| 129 | 49 | Nick McCloud | 51.10 | 224 | 49ers |
| 130 | 50 | Josh Jobe | 51.00 | 443 | Seahawks |
| 131 | 51 | Deonte Banks | 50.90 | 788 | Giants |
| 132 | 52 | Terrion Arnold | 50.80 | 1021 | Lions |
| 133 | 53 | Michael Carter II | 50.70 | 285 | Jets |
| 134 | 54 | Deantre Prince | 50.60 | 101 | Jaguars |
| 135 | 55 | Cameron Sutton | 49.50 | 273 | Steelers |
| 136 | 56 | Donte Jackson | 49.40 | 832 | Steelers |
| 137 | 57 | Martin Emerson Jr. | 47.90 | 827 | Browns |
| 138 | 58 | Benjamin St-Juste | 47.70 | 859 | Commanders |
| 139 | 59 | Kindle Vildor | 46.70 | 316 | Lions |
| 140 | 60 | Decamerion Richardson | 45.60 | 559 | Raiders |
| 141 | 61 | Noah Igbinoghene | 45.30 | 971 | Commanders |
| 142 | 62 | Josh Wallace | 44.20 | 165 | Rams |
| 143 | 63 | Alontae Taylor | 44.00 | 1075 | Saints |
| 144 | 64 | Marco Wilson | 43.50 | 242 | Bengals |
| 145 | 65 | Michael Davis | 43.30 | 139 | Commanders |
| 146 | 66 | Caelen Carson | 39.70 | 252 | Cowboys |
| 147 | 67 | Emmanuel Forbes | 39.00 | 160 | Rams |
| 148 | 68 | L'Jarius Sneed | 36.30 | 301 | Titans |
| 149 | 69 | Andrew Booth Jr. | 35.60 | 118 | Cowboys |
| 150 | 70 | Dane Jackson | 33.80 | 282 | Panthers |
| 151 | 71 | Cam Smith | 33.60 | 133 | Dolphins |
| 152 | 72 | Nehemiah Pritchett | 29.50 | 151 | Seahawks |

## DI — Defensive Interior

- **Season used:** `2024`
- **Grade column:** `grades_defense` · **Snap column:** `snap_counts_defense`

### Elite (9 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | Chris Jones | 90.40 | 886 | Chiefs |
| 2 | 2 | Cameron Heyward | 90.10 | 838 | Steelers |
| 3 | 3 | Dexter Lawrence | 89.90 | 551 | Giants |
| 4 | 4 | Leonard Williams | 87.10 | 750 | Seahawks |
| 5 | 5 | Poona Ford | 85.30 | 652 | Chargers |
| 6 | 6 | Calais Campbell | 82.30 | 616 | Dolphins |
| 7 | 7 | DeForest Buckner | 81.90 | 579 | Colts |
| 8 | 8 | Michael Pierce | 80.70 | 254 | Ravens |
| 9 | 9 | Jeffery Simmons | 80.20 | 806 | Titans |

### Good (11 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 10 | 1 | Alim McNeill | 79.60 | 631 | Lions |
| 11 | 2 | Zach Sieler | 78.90 | 749 | Dolphins |
| 12 | 3 | John Franklin-Myers | 78.30 | 569 | Broncos |
| 13 | 4 | Teair Tart | 78.10 | 378 | Chargers |
| 14 | 5 | Grover Stewart | 77.00 | 690 | Colts |
| 15 | 6 | Jalen Carter | 76.80 | 1026 | Eagles |
| 16 | 7 | T'Vondre Sweat | 76.20 | 699 | Titans |
| 17 | 8 | Jalen Redmond | 75.60 | 236 | Vikings |
| 18 | 9 | Kobie Turner | 75.10 | 919 | Rams |
| 19 | 10 | Vita Vea | 74.90 | 756 | Buccaneers |
| 20 | 11 | Christian Wilkins | 74.80 | 246 | Raiders |

### Starter (35 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 21 | 1 | Milton Williams | 73.50 | 628 | Eagles |
| 22 | 2 | Ed Oliver | 73.10 | 727 | Bills |
| 23 | 3 | Jowon Briggs | 72.20 | 133 | Browns |
| 24 | 4 | Jarran Reed | 70.60 | 679 | Seahawks |
| 25 | 5 | Sebastian Joseph-Day | 70.30 | 483 | Titans |
| 26 | 6 | Gervon Dexter Sr. | 70.30 | 616 | Bears |
| 27 | 7 | Jordan Davis | 70.20 | 430 | Eagles |
| 28 | 8 | B.J. Hill | 70.20 | 710 | Bengals |
| 29 | 9 | Levi Onwuzurike | 70.20 | 697 | Lions |
| 30 | 10 | Travis Jones | 69.90 | 675 | Ravens |
| 31 | 11 | Quinnen Williams | 69.60 | 722 | Jets |
| 32 | 12 | Keeanu Benton | 69.50 | 671 | Steelers |
| 33 | 13 | Osa Odighizuwa | 68.10 | 859 | Cowboys |
| 34 | 14 | Moro Ojomo | 67.50 | 465 | Eagles |
| 35 | 15 | Dalvin Tomlinson | 67.40 | 609 | Browns |
| 36 | 16 | D.J. Jones | 67.30 | 510 | Broncos |
| 37 | 17 | Zach Harrison | 67.20 | 272 | Falcons |
| 38 | 18 | William Gholston | 67.00 | 205 | Buccaneers |
| 39 | 19 | Shelby Harris | 66.70 | 527 | Browns |
| 40 | 20 | Elijah Garcia | 66.50 | 143 | Giants |
| 41 | 21 | DJ Reader | 66.50 | 566 | Lions |
| 42 | 22 | Devonte Wyatt | 66.40 | 366 | Packers |
| 43 | 23 | David Onyemata | 66.20 | 567 | Falcons |
| 44 | 24 | DaQuan Jones | 65.30 | 629 | Bills |
| 45 | 25 | Zach Allen | 64.90 | 1031 | Broncos |
| 46 | 26 | Ta'Quon Graham | 63.90 | 193 | Falcons |
| 47 | 27 | Naquan Jones | 63.20 | 260 | Cardinals |
| 48 | 28 | Jeremiah Ledbetter | 63.00 | 441 | Jaguars |
| 49 | 29 | Desjuan Johnson | 62.80 | 155 | Rams |
| 50 | 30 | Leonard Taylor III | 62.70 | 261 | Jets |
| 51 | 31 | Jordan Jefferson | 62.70 | 151 | Jaguars |
| 52 | 32 | Mike Pennel | 62.60 | 365 | Chiefs |
| 53 | 33 | Andrew Billings | 62.50 | 297 | Bears |
| 54 | 34 | Grady Jarrett | 62.10 | 744 | Falcons |
| 55 | 35 | Bruce Hector | 62.00 | 118 | Jets |

### Rotation/backup (112 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 56 | 1 | Karl Brooks | 61.70 | 459 | Packers |
| 57 | 2 | Kentavius Street | 61.50 | 280 | Falcons |
| 58 | 3 | Bobby Brown III | 61.40 | 513 | Rams |
| 59 | 4 | Evan Anderson | 60.80 | 267 | 49ers |
| 60 | 5 | Javon Hargrave | 60.60 | 104 | 49ers |
| 61 | 6 | Da'Shawn Hand | 60.50 | 564 | Dolphins |
| 62 | 7 | Malcolm Roach | 60.40 | 524 | Broncos |
| 63 | 8 | Daniel Ekuale | 60.30 | 723 | Patriots |
| 64 | 9 | Carlos Watkins | 60.20 | 228 | Cowboys |
| 65 | 10 | James Lynch | 60.10 | 243 | Titans |
| 66 | 11 | Kenny Clark | 60.00 | 725 | Packers |
| 67 | 12 | Khyiris Tonga | 60.00 | 229 | Cardinals |
| 68 | 13 | Logan Hall | 59.30 | 571 | Buccaneers |
| 69 | 14 | Isaiahh Loudermilk | 59.30 | 255 | Steelers |
| 70 | 15 | Byron Cowart | 58.90 | 335 | Bears |
| 71 | 16 | Elijah Chatman | 58.90 | 423 | Giants |
| 72 | 17 | Tim Settle | 58.60 | 685 | Texans |
| 73 | 18 | Harrison Phillips | 58.30 | 701 | Vikings |
| 74 | 19 | Byron Murphy II | 58.20 | 457 | Seahawks |
| 75 | 20 | Tommy Togiai | 58.10 | 280 | Texans |
| 76 | 21 | Jeremiah Pharms Jr. | 58.10 | 457 | Patriots |
| 77 | 22 | Taven Bryan | 58.10 | 340 | Colts |
| 78 | 23 | Adetomiwa Adebawore | 57.90 | 137 | Colts |
| 79 | 24 | Mario Edwards Jr. | 57.90 | 519 | Texans |
| 80 | 25 | Maliek Collins | 57.90 | 715 | 49ers |
| 81 | 26 | Jaquelin Roy | 57.70 | 141 | Patriots |
| 82 | 27 | Austin Johnson | 57.50 | 353 | Bills |
| 83 | 28 | Braden Fiske | 57.50 | 700 | Rams |
| 84 | 29 | Dante Stills | 57.00 | 532 | Cardinals |
| 85 | 30 | Adam Butler | 56.90 | 856 | Raiders |
| 86 | 31 | Christian Barmore | 56.50 | 123 | Patriots |
| 87 | 32 | Tershawn Wharton | 56.30 | 733 | Chiefs |
| 88 | 33 | Kurt Hinish | 56.20 | 231 | Texans |
| 89 | 34 | Ruke Orhorhoro | 56.20 | 147 | Falcons |
| 90 | 35 | Greg Gaines | 56.00 | 421 | Buccaneers |
| 91 | 36 | Jonathan Allen | 56.00 | 421 | Commanders |
| 92 | 37 | Thomas Booker IV | 55.40 | 172 | Eagles |
| 93 | 38 | Matthew Butler | 55.40 | 101 | Raiders |
| 94 | 39 | Sheldon Rankins | 55.30 | 287 | Bengals |
| 95 | 40 | Jonathan Bullard | 55.10 | 590 | Vikings |
| 96 | 41 | Roy Lopez | 54.90 | 464 | Cardinals |
| 97 | 42 | Khalil Davis | 54.90 | 209 | 49ers |
| 98 | 43 | Sheldon Day | 54.70 | 339 | Commanders |
| 99 | 44 | Khalen Saunders | 54.70 | 460 | Saints |
| 100 | 45 | A'Shawn Robinson | 54.60 | 761 | Panthers |
| 101 | 46 | DaVon Hamilton | 54.50 | 626 | Jaguars |
| 102 | 47 | Eric Johnson | 54.20 | 178 | Patriots |
| 103 | 48 | Linval Joseph | 54.10 | 264 | Cowboys |
| 104 | 49 | Javon Kinlaw | 53.40 | 695 | Jets |
| 105 | 50 | John Jenkins | 53.10 | 606 | Raiders |
| 106 | 51 | McKinnley Jackson | 53.00 | 248 | Bengals |
| 107 | 52 | Roy Robertson-Harris | 52.80 | 398 | Seahawks |
| 108 | 53 | Jer'Zhan Newton | 52.70 | 586 | Commanders |
| 109 | 54 | Mekhi Wingo | 52.60 | 177 | Lions |
| 110 | 55 | Broderick Washington | 51.90 | 488 | Ravens |
| 111 | 56 | Daron Payne | 51.80 | 796 | Commanders |
| 112 | 57 | Davon Godchaux | 51.60 | 680 | Patriots |
| 113 | 58 | Morgan Fox | 51.50 | 619 | Chargers |
| 114 | 59 | Benito Jones | 51.30 | 481 | Dolphins |
| 115 | 60 | Quinton Jefferson | 50.30 | 258 | Bills |
| 116 | 61 | Jerry Tillery | 50.20 | 482 | Vikings |
| 117 | 62 | Colby Wooden | 50.00 | 260 | Packers |
| 118 | 63 | Neville Gallimore | 49.90 | 308 | Rams |
| 119 | 64 | Kevin Givens | 49.60 | 185 | 49ers |
| 120 | 65 | Jalyn Holmes | 49.30 | 337 | Commanders |
| 121 | 66 | Eddie Goldman | 49.30 | 330 | Falcons |
| 122 | 67 | L.J. Collier | 49.20 | 588 | Cardinals |
| 123 | 68 | John Ridgeway | 48.40 | 263 | Saints |
| 124 | 69 | Darius Robinson | 48.40 | 184 | Cardinals |
| 125 | 70 | Larry Ogunjobi | 48.30 | 550 | Steelers |
| 126 | 71 | Tyler Davis | 48.20 | 354 | Rams |
| 127 | 72 | DeShawn Williams | 47.90 | 338 | Panthers |
| 128 | 73 | Ben Stille | 47.70 | 120 | Cardinals |
| 129 | 74 | Jordan Elliott | 47.60 | 440 | 49ers |
| 130 | 75 | Jonah Laulu | 47.40 | 474 | Raiders |
| 131 | 76 | Kalia Davis | 47.30 | 259 | 49ers |
| 132 | 77 | Zach Carter | 47.20 | 263 | Raiders |
| 133 | 78 | Rakeem Nunez-Roches | 46.80 | 608 | Giants |
| 134 | 79 | Calijah Kancey | 46.00 | 595 | Buccaneers |
| 135 | 80 | Solomon Thomas | 45.60 | 458 | Jets |
| 136 | 81 | Chris Williams | 45.50 | 367 | Bears |
| 137 | 82 | Zacch Pickens | 45.50 | 228 | Bears |
| 138 | 83 | Kris Jenkins | 45.40 | 496 | Bengals |
| 139 | 84 | Maurice Hurst | 45.30 | 164 | Browns |
| 140 | 85 | Tyler Lacy | 45.20 | 340 | Jaguars |
| 141 | 86 | Keondre Coburn | 45.10 | 125 | Titans |
| 142 | 87 | Jay Tufele | 44.40 | 242 | Bengals |
| 143 | 88 | Montravius Adams | 44.20 | 207 | Steelers |
| 144 | 89 | Maason Smith | 43.70 | 384 | Jaguars |
| 145 | 90 | Patrick O'Connor | 43.50 | 235 | Lions |
| 146 | 91 | Nathan Shepherd | 43.40 | 567 | Saints |
| 147 | 92 | C.J. Brewer | 42.90 | 159 | Buccaneers |
| 148 | 93 | Shy Tuttle | 42.70 | 610 | Panthers |
| 149 | 94 | Jaden Crumedy | 42.50 | 121 | Panthers |
| 150 | 95 | Otito Ogbonnia | 41.50 | 538 | Chargers |
| 151 | 96 | DeWayne Carter | 41.40 | 315 | Bills |
| 152 | 97 | Jordan Jackson | 41.30 | 329 | Broncos |
| 153 | 98 | D.J. Davidson | 41.00 | 261 | Giants |
| 154 | 99 | Bilal Nichols | 39.60 | 173 | Cardinals |
| 155 | 100 | Jonah Williams | 38.20 | 108 | Lions |
| 156 | 101 | Derrick Nnadi | 37.60 | 248 | Chiefs |
| 157 | 102 | Raekwon Davis | 37.00 | 349 | Colts |
| 158 | 103 | Bryan Bresee | 36.50 | 708 | Saints |
| 159 | 104 | Johnathan Hankins | 35.90 | 389 | Seahawks |
| 160 | 105 | Folorunso Fatukasi | 35.60 | 366 | Texans |
| 161 | 106 | Jordan Phillips | 34.90 | 185 | Bills |
| 162 | 107 | Mazi Smith | 34.80 | 524 | Cowboys |
| 163 | 108 | Dean Lowry | 34.30 | 159 | Steelers |
| 164 | 109 | Jordon Riley | 34.30 | 248 | Giants |
| 165 | 110 | LaBryan Ray | 33.00 | 626 | Panthers |
| 166 | 111 | Phidarian Mathis | 32.30 | 257 | Jets |
| 167 | 112 | Justin Jones | 30.20 | 100 | Cardinals |

## ED — Edge

- **Season used:** `2024`
- **Grade column:** `grades_defense` · **Snap column:** `snap_counts_defense`

### Elite (16 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | Aidan Hutchinson | 94.90 | 280 | Lions |
| 2 | 2 | Myles Garrett | 92.30 | 822 | Browns |
| 3 | 3 | T.J. Watt | 91.70 | 1002 | Steelers |
| 4 | 4 | Nick Bosa | 91.00 | 693 | 49ers |
| 5 | 5 | Khalil Mack | 90.20 | 668 | Chargers |
| 6 | 6 | Micah Parsons | 90.00 | 694 | Cowboys |
| 7 | 7 | Jared Verse | 89.30 | 933 | Rams |
| 8 | 8 | Alex Highsmith | 89.10 | 592 | Steelers |
| 9 | 9 | Will Anderson Jr. | 88.80 | 645 | Texans |
| 10 | 10 | Trey Hendrickson | 88.10 | 823 | Bengals |
| 11 | 11 | Von Miller | 85.50 | 332 | Bills |
| 12 | 12 | Danielle Hunter | 84.60 | 859 | Texans |
| 13 | 13 | Isaiah McGuire | 83.30 | 469 | Browns |
| 14 | 14 | Greg Rousseau | 82.20 | 861 | Bills |
| 15 | 15 | Jonathan Greenard | 81.00 | 969 | Vikings |
| 16 | 16 | Nick Herbig | 80.70 | 433 | Steelers |

### Good (11 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 17 | 1 | Carl Granderson | 79.70 | 825 | Saints |
| 18 | 2 | Brian Burns | 79.20 | 865 | Giants |
| 19 | 3 | Nik Bonitto | 78.70 | 761 | Broncos |
| 20 | 4 | Brandon Graham | 78.60 | 311 | Eagles |
| 21 | 5 | Yaya Diaby | 78.00 | 841 | Buccaneers |
| 22 | 6 | Kyle Van Noy | 75.80 | 696 | Ravens |
| 23 | 7 | Boye Mafe | 75.20 | 607 | Seahawks |
| 24 | 8 | Victor Dimukeje | 75.10 | 157 | Cardinals |
| 25 | 9 | Nolan Smith | 74.40 | 725 | Eagles |
| 26 | 10 | Derek Barnett | 74.20 | 413 | Texans |
| 27 | 11 | Maxx Crosby | 74.10 | 766 | Raiders |

### Starter (49 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 28 | 1 | Rashan Gary | 73.00 | 670 | Packers |
| 29 | 2 | Dennis Gardeck | 72.00 | 206 | Cardinals |
| 30 | 3 | Zaven Collins | 72.00 | 600 | Cardinals |
| 31 | 4 | Za'Darius Smith | 71.70 | 655 | Lions |
| 32 | 5 | Laiatu Latu | 71.50 | 618 | Colts |
| 33 | 6 | Odafe Oweh | 71.50 | 683 | Ravens |
| 34 | 7 | Andrew Van Ginkel | 71.40 | 973 | Vikings |
| 35 | 8 | Josh Sweat | 71.00 | 775 | Eagles |
| 36 | 9 | Jadeveon Clowney | 70.80 | 650 | Panthers |
| 37 | 10 | Harold Landry III | 70.50 | 878 | Titans |
| 38 | 11 | George Karlaftis | 70.40 | 953 | Chiefs |
| 39 | 12 | Chop Robinson | 70.00 | 565 | Dolphins |
| 40 | 13 | Arden Key | 69.70 | 734 | Titans |
| 41 | 14 | Bryce Huff | 69.70 | 298 | Eagles |
| 42 | 15 | Dondrea Tillman | 69.10 | 275 | Broncos |
| 43 | 16 | Kayvon Thibodeaux | 69.00 | 593 | Giants |
| 44 | 17 | Jonathon Cooper | 68.80 | 882 | Broncos |
| 45 | 18 | Javon Solomon | 68.80 | 141 | Bills |
| 46 | 19 | Arik Armstead | 68.70 | 569 | Jaguars |
| 47 | 20 | Keion White | 68.50 | 830 | Patriots |
| 48 | 21 | Travon Walker | 68.20 | 911 | Jaguars |
| 49 | 22 | Arnold Ebiketie | 68.20 | 543 | Falcons |
| 50 | 23 | Brenton Cox Jr. | 67.20 | 187 | Packers |
| 51 | 24 | Jalyx Hunt | 67.10 | 320 | Eagles |
| 52 | 25 | Kwity Paye | 66.80 | 667 | Colts |
| 53 | 26 | Tyree Wilson | 66.70 | 524 | Raiders |
| 54 | 27 | Tuli Tuipulotu | 66.50 | 774 | Chargers |
| 55 | 28 | Anfernee Jennings | 66.40 | 831 | Patriots |
| 56 | 29 | Chase Young | 66.40 | 740 | Saints |
| 57 | 30 | Eric Watts | 66.10 | 231 | Jets |
| 58 | 31 | Jaelan Phillips | 66.10 | 134 | Dolphins |
| 59 | 32 | Dayo Odeyingbo | 66.10 | 746 | Colts |
| 60 | 33 | Charles Omenihu | 66.00 | 303 | Chiefs |
| 61 | 34 | Montez Sweat | 65.60 | 616 | Bears |
| 62 | 35 | Dorance Armstrong | 65.30 | 747 | Commanders |
| 63 | 36 | Tyquan Lewis | 65.00 | 355 | Colts |
| 64 | 37 | Quinton Bell | 64.80 | 258 | Dolphins |
| 65 | 38 | DeMarcus Walker | 64.70 | 738 | Bears |
| 66 | 39 | Michael Hoecht | 64.70 | 705 | Rams |
| 67 | 40 | Preston Smith | 64.20 | 469 | Steelers |
| 68 | 41 | Dallas Turner | 64.10 | 310 | Vikings |
| 69 | 42 | Jacob Martin | 63.50 | 222 | Bears |
| 70 | 43 | Demone Harris | 63.00 | 216 | Falcons |
| 71 | 44 | Uchenna Nwosu | 63.00 | 190 | Seahawks |
| 72 | 45 | K'Lavon Chaisson | 63.00 | 508 | Raiders |
| 73 | 46 | Arron Mosby | 62.50 | 150 | Packers |
| 74 | 47 | Tomon Fox | 62.20 | 207 | Giants |
| 75 | 48 | Chris Braswell | 62.10 | 335 | Buccaneers |
| 76 | 49 | Tavius Robinson | 62.00 | 548 | Ravens |

### Rotation/backup (66 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 77 | 1 | Mike Danna | 61.60 | 581 | Chiefs |
| 78 | 2 | Julian Okwara | 61.60 | 286 | Cardinals |
| 79 | 3 | Byron Young | 61.50 | 936 | Rams |
| 80 | 4 | Dylan Horton | 61.30 | 217 | Texans |
| 81 | 5 | Charles Snowden | 61.20 | 405 | Raiders |
| 82 | 6 | Joey Bosa | 61.10 | 503 | Chargers |
| 83 | 7 | James Smith-Williams | 60.90 | 306 | Falcons |
| 84 | 8 | Felix Anudike-Uzomah | 60.80 | 344 | Chiefs |
| 85 | 9 | Al-Quadin Muhammad | 60.70 | 293 | Lions |
| 86 | 10 | Dante Fowler Jr. | 60.50 | 642 | Commanders |
| 87 | 11 | Malik Herring | 60.20 | 193 | Chiefs |
| 88 | 12 | Alex Wright | 60.20 | 103 | Browns |
| 89 | 13 | Derick Hall | 60.00 | 673 | Seahawks |
| 90 | 14 | Darrell Taylor | 59.80 | 374 | Bears |
| 91 | 15 | Deatrich Wise Jr. | 59.70 | 409 | Patriots |
| 92 | 16 | Carl Lawson | 59.70 | 402 | Cowboys |
| 93 | 17 | Emmanuel Ogbah | 59.70 | 734 | Dolphins |
| 94 | 18 | Will McDonald IV | 59.40 | 756 | Jets |
| 95 | 19 | Sam Hubbard | 58.90 | 521 | Bengals |
| 96 | 20 | Joseph Ossai | 58.30 | 573 | Bengals |
| 97 | 21 | Baron Browning | 58.00 | 378 | Cardinals |
| 98 | 22 | Cameron Jordan | 57.40 | 565 | Saints |
| 99 | 23 | DeMarcus Lawrence | 56.80 | 167 | Cowboys |
| 100 | 24 | Kingsley Enagbare | 56.80 | 538 | Packers |
| 101 | 25 | Anthony Nelson | 56.60 | 624 | Buccaneers |
| 102 | 26 | Myles Murphy | 56.50 | 353 | Bengals |
| 103 | 27 | A.J. Epenesa | 56.50 | 712 | Bills |
| 104 | 28 | Azeez Ojulari | 56.40 | 391 | Giants |
| 105 | 29 | Dawuane Smoot | 56.30 | 386 | Bills |
| 106 | 30 | Robert Beal Jr. | 55.30 | 149 | 49ers |
| 107 | 31 | D.J. Wonnum | 55.00 | 453 | Panthers |
| 108 | 32 | Jonah Elliss | 54.60 | 441 | Broncos |
| 109 | 33 | Dre'Mont Jones | 54.30 | 617 | Seahawks |
| 110 | 34 | DJ Johnson | 54.20 | 392 | Panthers |
| 111 | 35 | Sam Okuayinonu | 54.10 | 451 | 49ers |
| 112 | 36 | Charles Harris | 54.10 | 474 | Eagles |
| 113 | 37 | Payton Turner | 54.00 | 335 | Saints |
| 114 | 38 | Brent Urban | 53.50 | 209 | Ravens |
| 115 | 39 | Leonard Floyd | 53.50 | 604 | 49ers |
| 116 | 40 | Lukas Van Ness | 53.50 | 458 | Packers |
| 117 | 41 | Haason Reddick | 53.50 | 392 | Jets |
| 118 | 42 | Joe Tryon-Shoyinka | 53.10 | 570 | Buccaneers |
| 119 | 43 | Josh Paschal | 53.00 | 613 | Lions |
| 120 | 44 | Tyus Bowser | 53.00 | 276 | Dolphins |
| 121 | 45 | Javontae Jean-Baptiste | 52.90 | 248 | Commanders |
| 122 | 46 | David Ojabo | 52.90 | 292 | Ravens |
| 123 | 47 | Austin Booker | 52.50 | 283 | Bears |
| 124 | 48 | Clelin Ferrell | 51.70 | 443 | Commanders |
| 125 | 49 | Yetur Gross-Matos | 51.50 | 367 | 49ers |
| 126 | 50 | Jamin Davis | 51.10 | 107 | Jets |
| 127 | 51 | Marshawn Kneeland | 50.80 | 255 | Cowboys |
| 128 | 52 | Micheal Clemons | 50.60 | 624 | Jets |
| 129 | 53 | Casey Toohill | 50.00 | 249 | Bills |
| 130 | 54 | Cam Gill | 49.10 | 222 | Panthers |
| 131 | 55 | Tyrus Wheat | 46.50 | 165 | Cowboys |
| 132 | 56 | Bud Dupree | 46.50 | 570 | Chargers |
| 133 | 57 | James Houston | 46.00 | 141 | Browns |
| 134 | 58 | Ogbo Okoronkwo | 45.30 | 464 | Browns |
| 135 | 59 | Jeremiah Moon | 44.20 | 117 | Steelers |
| 136 | 60 | Matthew Judon | 43.00 | 655 | Falcons |
| 137 | 61 | Lorenzo Carter | 43.00 | 410 | Falcons |
| 138 | 62 | Janarius Robinson | 42.60 | 109 | Raiders |
| 139 | 63 | Xavier Thomas | 39.40 | 208 | Cardinals |
| 140 | 64 | Jaylen Harrell | 38.90 | 286 | Titans |
| 141 | 65 | Ali Gaye | 38.30 | 177 | Titans |
| 142 | 66 | Myles Cole | 34.00 | 135 | Jaguars |

## G — Guard

- **Season used:** `2024`
- **Grade column:** `grades_offense` · **Snap column:** `snap_counts_offense`

### Elite (11 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | Chris Lindstrom | 93.50 | 1099 | Falcons |
| 2 | 2 | James Daniels | 92.90 | 209 | Steelers |
| 3 | 3 | Christian Mahogany | 91.50 | 144 | Lions |
| 4 | 4 | Will Fries | 86.90 | 268 | Colts |
| 5 | 5 | Quinn Meinerz | 86.90 | 1131 | Broncos |
| 6 | 6 | Kevin Zeitler | 86.80 | 1047 | Lions |
| 7 | 7 | Landon Dickerson | 82.30 | 1157 | Eagles |
| 8 | 8 | Quenton Nelson | 81.30 | 1083 | Colts |
| 9 | 9 | Jordan Meredith | 80.80 | 574 | Raiders |
| 10 | 10 | Dominick Puni | 80.50 | 1078 | 49ers |
| 11 | 11 | Joe Thuney | 80.20 | 1232 | Chiefs |

### Good (9 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 12 | 1 | Alijah Vera-Tucker | 77.70 | 916 | Jets |
| 13 | 2 | Kevin Dotson | 77.70 | 1145 | Rams |
| 14 | 3 | John Simpson | 77.30 | 1020 | Jets |
| 15 | 4 | Damien Lewis | 75.50 | 942 | Panthers |
| 16 | 5 | Teven Jenkins | 75.40 | 738 | Bears |
| 17 | 6 | Trey Smith | 75.30 | 1232 | Chiefs |
| 18 | 7 | Tyler Smith | 75.00 | 1052 | Cowboys |
| 19 | 8 | Cody Mauch | 74.60 | 1178 | Buccaneers |
| 20 | 9 | Dylan Parham | 74.30 | 882 | Raiders |

### Starter (28 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 21 | 1 | Mekhi Becton | 72.50 | 1097 | Eagles |
| 22 | 2 | Jack Driscoll | 71.30 | 110 | Eagles |
| 23 | 3 | Chandler Zavala | 71.20 | 198 | Panthers |
| 24 | 4 | Matthew Bergeron | 70.90 | 1106 | Falcons |
| 25 | 5 | Jake Hanson | 69.90 | 103 | Jets |
| 26 | 6 | Matt Pryor | 69.90 | 1005 | Bears |
| 27 | 7 | Will Hernandez | 69.30 | 280 | Cardinals |
| 28 | 8 | Dalton Risner | 68.10 | 611 | Vikings |
| 29 | 9 | Sam Cosmi | 67.80 | 1259 | Commanders |
| 30 | 10 | Robert Hunt | 67.70 | 966 | Panthers |
| 31 | 11 | Cesar Ruiz | 67.60 | 813 | Saints |
| 32 | 12 | Jonah Jackson | 67.60 | 267 | Rams |
| 33 | 13 | Isaac Seumalo | 66.30 | 872 | Steelers |
| 34 | 14 | David Edwards | 66.10 | 2360 | Bills |
| 35 | 15 | Evan Brown | 65.90 | 1070 | Cardinals |
| 36 | 16 | Zack Martin | 65.60 | 638 | Cowboys |
| 37 | 17 | Elgton Jenkins | 65.50 | 1073 | Packers |
| 38 | 18 | Aaron Banks | 65.40 | 775 | 49ers |
| 39 | 19 | Ezra Cleveland | 64.90 | 911 | Jaguars |
| 40 | 20 | Brandon Scherff | 64.70 | 1013 | Jaguars |
| 41 | 21 | Zion Johnson | 64.40 | 1102 | Chargers |
| 42 | 22 | Ben Powers | 64.40 | 1130 | Broncos |
| 43 | 23 | Joel Bitonio | 63.90 | 1178 | Browns |
| 44 | 24 | Jackson Powers-Johnson | 63.90 | 956 | Raiders |
| 45 | 25 | Greg Van Roten | 63.40 | 1121 | Giants |
| 46 | 26 | T.J. Bass | 63.00 | 315 | Cowboys |
| 47 | 27 | Wyatt Teller | 62.60 | 885 | Browns |
| 48 | 28 | Laken Tomlinson | 62.10 | 1094 | Seahawks |

### Rotation/backup (38 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 49 | 1 | Sean Rhyan | 61.30 | 1027 | Packers |
| 50 | 2 | Shaq Mason | 60.50 | 999 | Texans |
| 51 | 3 | Peter Skoronski | 60.30 | 1095 | Titans |
| 52 | 4 | Nick Allegretti | 59.40 | 1372 | Commanders |
| 53 | 5 | Cordell Volson | 59.30 | 984 | Bengals |
| 54 | 6 | Jordan Morgan | 59.20 | 186 | Packers |
| 55 | 7 | Patrick Mekari | 59.00 | 1131 | Ravens |
| 56 | 8 | Nick Zakelj | 58.70 | 162 | 49ers |
| 57 | 9 | Trey Pipkins III | 57.80 | 838 | Chargers |
| 58 | 10 | Mason McCormick | 57.70 | 936 | Steelers |
| 59 | 11 | Spencer Burford | 57.60 | 113 | 49ers |
| 60 | 12 | Graham Glasgow | 57.20 | 1149 | Lions |
| 61 | 13 | Andrew Vorhees | 57.20 | 268 | Ravens |
| 62 | 14 | Spencer Anderson | 56.70 | 357 | Steelers |
| 63 | 15 | Robert Jones | 56.10 | 1080 | Dolphins |
| 64 | 16 | Jon Runyan | 56.10 | 842 | Giants |
| 65 | 17 | Nick Saldiveri | 56.00 | 344 | Saints |
| 66 | 18 | Ben Bredeson | 56.00 | 1173 | Buccaneers |
| 67 | 19 | Blake Brandel | 55.70 | 1191 | Vikings |
| 68 | 20 | O'Cyrus Torrence | 55.50 | 1221 | Bills |
| 69 | 21 | Ed Ingram | 54.00 | 580 | Vikings |
| 70 | 22 | Mark Glowinski | 53.40 | 355 | Colts |
| 71 | 23 | Dalton Tucker | 53.30 | 464 | Colts |
| 72 | 24 | Jake Kubas | 52.20 | 197 | Giants |
| 73 | 25 | Kayode Awosika | 51.30 | 145 | Lions |
| 74 | 26 | Alex Cappa | 50.50 | 1132 | Bengals |
| 75 | 27 | Mike Caliendo | 49.40 | 354 | Chiefs |
| 76 | 28 | Isaiah Wynn | 49.00 | 103 | Dolphins |
| 77 | 29 | Anthony Bradford | 48.90 | 578 | Seahawks |
| 78 | 30 | Christian Haynes | 48.50 | 167 | Seahawks |
| 79 | 31 | Aaron Stinnie | 47.90 | 193 | Giants |
| 80 | 32 | Michael Dunn | 46.50 | 171 | Browns |
| 81 | 33 | Logan Bruss | 44.80 | 195 | Titans |
| 82 | 34 | Zak Zinter | 43.90 | 233 | Browns |
| 83 | 35 | Layden Robinson | 43.60 | 602 | Patriots |
| 84 | 36 | Kenyon Green | 38.60 | 582 | Texans |
| 85 | 37 | Sataoa Laumea | 36.90 | 355 | Seahawks |
| 86 | 38 | Sidy Sow | 29.80 | 155 | Patriots |

## HB — Running Back

- **Season used:** `2024`
- **Grade column:** `grades_offense` · **Snap column:** `routes`

### Elite (12 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | Derrick Henry | 94.20 | 197 | Ravens |
| 2 | 2 | Bijan Robinson | 92.80 | 389 | Falcons |
| 3 | 3 | Josh Jacobs | 92.30 | 265 | Packers |
| 4 | 4 | Bucky Irving | 90.80 | 246 | Buccaneers |
| 5 | 5 | James Conner | 90.40 | 269 | Cardinals |
| 6 | 6 | Jahmyr Gibbs | 90.10 | 347 | Lions |
| 7 | 7 | Kenneth Walker III | 88.40 | 224 | Seahawks |
| 8 | 8 | Saquon Barkley | 87.60 | 353 | Eagles |
| 9 | 9 | James Cook | 86.20 | 258 | Bills |
| 10 | 10 | David Montgomery | 85.90 | 158 | Lions |
| 11 | 11 | Emanuel Wilson | 82.90 | 109 | Packers |
| 12 | 12 | De'Von Achane | 81.60 | 408 | Dolphins |

### Good (8 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 13 | 1 | Justice Hill | 79.50 | 264 | Ravens |
| 14 | 2 | Zach Charbonnet | 77.50 | 284 | Seahawks |
| 15 | 3 | Najee Harris | 77.20 | 233 | Steelers |
| 16 | 4 | Joe Mixon | 76.60 | 273 | Texans |
| 17 | 5 | Chuba Hubbard | 75.90 | 336 | Panthers |
| 18 | 6 | Chase Brown | 75.70 | 339 | Bengals |
| 19 | 7 | Aaron Jones | 75.40 | 347 | Vikings |
| 20 | 8 | Kareem Hunt | 74.30 | 238 | Chiefs |

### Starter (30 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 21 | 1 | Rico Dowdle | 73.90 | 283 | Cowboys |
| 22 | 2 | Rachaad White | 73.80 | 311 | Buccaneers |
| 23 | 3 | Alvin Kamara | 73.70 | 311 | Saints |
| 24 | 4 | Braelon Allen | 73.60 | 134 | Jets |
| 25 | 5 | Ty Johnson | 73.30 | 231 | Bills |
| 26 | 6 | Jordan Mason | 72.70 | 170 | 49ers |
| 27 | 7 | Antonio Gibson | 72.60 | 164 | Patriots |
| 28 | 8 | Cam Akers | 72.20 | 126 | Vikings |
| 29 | 9 | Jeremy McNichols | 72.00 | 136 | Commanders |
| 30 | 10 | Brian Robinson | 72.00 | 237 | Commanders |
| 31 | 11 | Jerome Ford | 71.20 | 304 | Browns |
| 32 | 12 | Ray Davis | 70.40 | 112 | Bills |
| 33 | 13 | Austin Ekeler | 69.80 | 283 | Commanders |
| 34 | 14 | Rhamondre Stevenson | 69.60 | 273 | Patriots |
| 35 | 15 | Roschon Johnson | 69.50 | 136 | Bears |
| 36 | 16 | Raheem Mostert | 69.30 | 155 | Dolphins |
| 37 | 17 | Kyren Williams | 69.10 | 402 | Rams |
| 38 | 18 | Tony Pollard | 68.70 | 301 | Titans |
| 39 | 19 | Ameer Abdullah | 68.30 | 258 | Raiders |
| 40 | 20 | Tank Bigsby | 68.10 | 129 | Jaguars |
| 41 | 21 | Miles Sanders | 68.00 | 130 | Panthers |
| 42 | 22 | Tyjae Spears | 67.80 | 167 | Titans |
| 43 | 23 | Samaje Perine | 67.60 | 217 | Chiefs |
| 44 | 24 | J.K. Dobbins | 66.60 | 227 | Chargers |
| 45 | 25 | Jaleel McLaughlin | 65.20 | 146 | Broncos |
| 46 | 26 | Isaac Guerendo | 64.60 | 107 | 49ers |
| 47 | 27 | Jaylen Warren | 64.30 | 232 | Steelers |
| 48 | 28 | Isiah Pacheco | 64.20 | 115 | Chiefs |
| 49 | 29 | Devin Singletary | 62.10 | 164 | Giants |
| 50 | 30 | Breece Hall | 62.00 | 384 | Jets |

### Rotation/backup (12 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 51 | 1 | Javonte Williams | 61.70 | 300 | Broncos |
| 52 | 2 | Alexander Mattison | 61.40 | 218 | Raiders |
| 53 | 3 | D'Andre Swift | 61.30 | 353 | Bears |
| 54 | 4 | D'Ernest Johnson | 60.70 | 117 | Jaguars |
| 55 | 5 | Travis Etienne Jr. | 60.70 | 254 | Jaguars |
| 56 | 6 | Dare Ogunbowale | 60.60 | 213 | Texans |
| 57 | 7 | Pierre Strong Jr. | 58.40 | 134 | Browns |
| 58 | 8 | Tyrone Tracy | 58.40 | 310 | Giants |
| 59 | 9 | Zack Moss | 58.20 | 155 | Bengals |
| 60 | 10 | Kenneth Gainwell | 57.60 | 132 | Eagles |
| 61 | 11 | Jonathan Taylor | 56.90 | 270 | Colts |
| 62 | 12 | Trey Sermon | 54.10 | 122 | Colts |

## LB — Linebacker

- **Season used:** `2024`
- **Grade column:** `grades_defense` · **Snap column:** `snap_counts_defense`

### Elite (8 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | Zack Baun | 90.20 | 1150 | Eagles |
| 2 | 2 | Fred Warner | 89.20 | 997 | 49ers |
| 3 | 3 | Bobby Wagner | 88.30 | 1258 | Commanders |
| 4 | 4 | Edgerrin Cooper | 85.70 | 549 | Packers |
| 5 | 5 | Leo Chenal | 84.50 | 497 | Chiefs |
| 6 | 6 | Jack Gibbens | 83.20 | 234 | Titans |
| 7 | 7 | Oren Burks | 81.90 | 322 | Eagles |
| 8 | 8 | Jeremiah Owusu-Koramoah | 80.60 | 460 | Browns |

### Good (13 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 9 | 1 | Elandon Roberts | 79.70 | 525 | Steelers |
| 10 | 2 | Devin Bush | 79.20 | 497 | Browns |
| 11 | 3 | Jack Campbell | 78.70 | 1047 | Lions |
| 12 | 4 | Jake Hansen | 78.00 | 136 | Texans |
| 13 | 5 | Jordan Hicks | 77.40 | 602 | Browns |
| 14 | 6 | Devin Lloyd | 76.70 | 884 | Jaguars |
| 15 | 7 | Nakobe Dean | 76.60 | 880 | Eagles |
| 16 | 8 | Jeremiah Trotter Jr. | 75.40 | 109 | Eagles |
| 17 | 9 | Eric Kendricks | 75.20 | 918 | Cowboys |
| 18 | 10 | Bobby Okereke | 74.90 | 734 | Giants |
| 19 | 11 | C.J. Mosley | 74.70 | 110 | Jets |
| 20 | 12 | Payton Wilson | 74.70 | 520 | Steelers |
| 21 | 13 | Malcolm Rodriguez | 74.40 | 318 | Lions |

### Starter (48 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 22 | 1 | Demario Davis | 73.20 | 1090 | Saints |
| 23 | 2 | Chazz Surratt | 72.70 | 137 | Jets |
| 24 | 3 | Christian Elliss | 72.60 | 514 | Patriots |
| 25 | 4 | Logan Wilson | 72.30 | 743 | Bengals |
| 26 | 5 | Joe Andreessen | 72.10 | 116 | Bills |
| 27 | 6 | Blake Cashman | 72.00 | 947 | Vikings |
| 28 | 7 | Neville Hewitt | 71.90 | 351 | Texans |
| 29 | 8 | Derrick Barnes | 71.80 | 120 | Lions |
| 30 | 9 | Jordyn Brooks | 71.30 | 1039 | Dolphins |
| 31 | 10 | Kaden Elliss | 71.10 | 1097 | Falcons |
| 32 | 11 | Daiyan Henley | 69.90 | 1071 | Chargers |
| 33 | 12 | Grant Stuard | 69.40 | 229 | Colts |
| 34 | 13 | Omar Speights | 69.40 | 504 | Rams |
| 35 | 14 | J.J. Russell | 69.20 | 271 | Buccaneers |
| 36 | 15 | Pete Werner | 69.00 | 731 | Saints |
| 37 | 16 | Azeez Al-Shaair | 68.90 | 672 | Texans |
| 38 | 17 | Foyesade Oluokun | 68.50 | 815 | Jaguars |
| 39 | 18 | Robert Spillane | 68.40 | 1093 | Raiders |
| 40 | 19 | Damone Clark | 68.20 | 163 | Cowboys |
| 41 | 20 | Quincy Williams | 68.00 | 1136 | Jets |
| 42 | 21 | Lavonte David | 67.90 | 1149 | Buccaneers |
| 43 | 22 | Shaq Thompson | 67.40 | 245 | Panthers |
| 44 | 23 | SirVocea Dennis | 67.30 | 105 | Buccaneers |
| 45 | 24 | Tyrel Dodson | 67.30 | 854 | Dolphins |
| 46 | 25 | Alex Anzalone | 66.90 | 681 | Lions |
| 47 | 26 | Roquan Smith | 66.80 | 1099 | Ravens |
| 48 | 27 | Dee Winters | 66.40 | 398 | 49ers |
| 49 | 28 | Drue Tranquill | 66.00 | 902 | Chiefs |
| 50 | 29 | Nate Landman | 65.70 | 543 | Falcons |
| 51 | 30 | Troy Dye | 65.60 | 355 | Chargers |
| 52 | 31 | Chris Board | 65.50 | 213 | Ravens |
| 53 | 32 | Tyrice Knight | 65.40 | 550 | Seahawks |
| 54 | 33 | Eric Wilson | 64.30 | 559 | Packers |
| 55 | 34 | Frankie Luvu | 64.20 | 1239 | Commanders |
| 56 | 35 | Ty Summers | 64.20 | 113 | Giants |
| 57 | 36 | Mack Wilson Sr. | 63.80 | 760 | Cardinals |
| 58 | 37 | Cody Barton | 63.70 | 1129 | Broncos |
| 59 | 38 | Claudin Cherelus | 63.30 | 158 | Panthers |
| 60 | 39 | Jack Sanborn | 63.10 | 235 | Bears |
| 61 | 40 | Krys Barnes | 63.10 | 205 | Cardinals |
| 62 | 41 | Sione Takitaki | 63.00 | 194 | Patriots |
| 63 | 42 | Ivan Pace Jr. | 63.00 | 454 | Vikings |
| 64 | 43 | Owen Pappoe | 63.00 | 131 | Cardinals |
| 65 | 44 | Alex Singleton | 62.90 | 190 | Broncos |
| 66 | 45 | Micah McFadden | 62.80 | 668 | Giants |
| 67 | 46 | Nick Bolton | 62.50 | 1076 | Chiefs |
| 68 | 47 | Denzel Perryman | 62.30 | 343 | Chargers |
| 69 | 48 | Henry To'oTo'o | 62.20 | 936 | Texans |

### Rotation/backup (57 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 70 | 1 | DeMarvion Overshown | 61.60 | 708 | Cowboys |
| 71 | 2 | T.J. Edwards | 61.40 | 1054 | Bears |
| 72 | 3 | Jerome Baker | 61.00 | 566 | Titans |
| 73 | 4 | Ernest Jones | 60.70 | 995 | Seahawks |
| 74 | 5 | Troy Andersen | 60.40 | 287 | Falcons |
| 75 | 6 | Zaire Franklin | 60.30 | 1157 | Colts |
| 76 | 7 | Germaine Pratt | 60.20 | 1075 | Bengals |
| 77 | 8 | Luke Gifford | 60.20 | 203 | Titans |
| 78 | 9 | Tremaine Edmunds | 59.30 | 1055 | Bears |
| 79 | 10 | Akeem Davis-Gaither | 59.00 | 535 | Bengals |
| 80 | 11 | Trenton Simpson | 58.70 | 654 | Ravens |
| 81 | 12 | Dorian Williams | 58.50 | 680 | Bills |
| 82 | 13 | De'Vondre Campbell | 58.30 | 719 | 49ers |
| 83 | 14 | Troy Reeder | 57.90 | 372 | Rams |
| 84 | 15 | Jalen Reeves-Maybin | 57.50 | 165 | Lions |
| 85 | 16 | Chad Muma | 57.40 | 260 | Jaguars |
| 86 | 17 | Quay Walker | 57.40 | 804 | Packers |
| 87 | 18 | Divine Deablo | 57.30 | 689 | Raiders |
| 88 | 19 | Amari Burney | 57.20 | 101 | Raiders |
| 89 | 20 | Devin White | 57.00 | 176 | Texans |
| 90 | 21 | Darius Muasau | 56.80 | 435 | Giants |
| 91 | 22 | Patrick Queen | 56.80 | 1164 | Steelers |
| 92 | 23 | E.J. Speed | 56.70 | 1011 | Colts |
| 93 | 24 | Yasir Abdullah | 56.60 | 170 | Jaguars |
| 94 | 25 | Josey Jewell | 56.50 | 796 | Panthers |
| 95 | 26 | Trevin Wallace | 56.00 | 582 | Panthers |
| 96 | 27 | Jahlani Tavai | 55.50 | 916 | Patriots |
| 97 | 28 | Isaiah McDuffie | 55.40 | 728 | Packers |
| 98 | 29 | Anfernee Orji | 55.20 | 147 | Saints |
| 99 | 30 | Ben Niemann | 55.20 | 178 | Lions |
| 100 | 31 | Christian Rozeboom | 53.50 | 956 | Rams |
| 101 | 32 | Matt Milano | 53.30 | 333 | Bills |
| 102 | 33 | Mohamoud Diabate | 52.50 | 581 | Browns |
| 103 | 34 | Ventrell Miller | 52.00 | 482 | Jaguars |
| 104 | 35 | Winston Reid | 51.80 | 144 | Browns |
| 105 | 36 | Malik Harrison | 51.60 | 438 | Ravens |
| 106 | 37 | Jacoby Windmon | 50.50 | 128 | Panthers |
| 107 | 38 | Marist Liufau | 50.10 | 520 | Cowboys |
| 108 | 39 | Justin Strnad | 49.90 | 736 | Broncos |
| 109 | 40 | JD Bertrand | 49.80 | 157 | Falcons |
| 110 | 41 | Ezekiel Turner | 49.70 | 111 | Lions |
| 111 | 42 | Kyzir White | 48.80 | 1015 | Cardinals |
| 112 | 43 | Nick Vigil | 48.30 | 127 | Cowboys |
| 113 | 44 | Terrel Bernard | 48.20 | 917 | Bills |
| 114 | 45 | Anthony Walker Jr. | 48.00 | 516 | Dolphins |
| 115 | 46 | Luke Masterson | 46.70 | 102 | Raiders |
| 116 | 47 | Kenneth Murray Jr. | 45.90 | 815 | Titans |
| 117 | 48 | Isaiah Simmons | 45.60 | 181 | Giants |
| 118 | 49 | K.J. Britt | 45.50 | 632 | Buccaneers |
| 119 | 50 | Willie Gay | 43.90 | 277 | Saints |
| 120 | 51 | Raekwon McMillan | 40.80 | 267 | Titans |
| 121 | 52 | Christian Harris | 39.10 | 180 | Texans |
| 122 | 53 | Junior Colson | 36.70 | 234 | Chargers |
| 123 | 54 | Kamu Grugier-Hill | 36.40 | 182 | Vikings |
| 124 | 55 | Demetrius Flannigan-Fowles | 30.60 | 151 | 49ers |
| 125 | 56 | Baylon Spector | 30.10 | 291 | Bills |
| 126 | 57 | Chandler Wooten | 29.20 | 212 | Panthers |

## QB — Quarterback

- **Season used:** `2024`
- **Grade column:** `grades_pass` · **Snap column:** `passing_snaps`

### Elite (10 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | Lamar Jackson | 93.30 | 635 | Ravens |
| 2 | 2 | Joe Burrow | 92.80 | 775 | Bengals |
| 3 | 3 | Justin Herbert | 90.20 | 662 | Chargers |
| 4 | 4 | Michael Penix Jr. | 87.60 | 120 | Falcons |
| 5 | 5 | Derek Carr | 86.70 | 319 | Saints |
| 6 | 6 | Jayden Daniels | 84.70 | 781 | Commanders |
| 7 | 7 | Josh Allen | 84.10 | 686 | Bills |
| 8 | 8 | Baker Mayfield | 83.60 | 721 | Buccaneers |
| 9 | 9 | Geno Smith | 81.90 | 704 | Seahawks |
| 10 | 10 | Patrick Mahomes | 80.80 | 776 | Chiefs |

### Good (9 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 11 | 1 | Kyler Murray | 78.10 | 656 | Cardinals |
| 12 | 2 | C.J. Stroud | 77.50 | 742 | Texans |
| 13 | 3 | Sam Darnold | 77.50 | 725 | Vikings |
| 14 | 4 | Russell Wilson | 77.50 | 444 | Steelers |
| 15 | 5 | Aaron Rodgers | 76.30 | 684 | Jets |
| 16 | 6 | Brock Purdy | 76.30 | 567 | 49ers |
| 17 | 7 | Jared Goff | 76.30 | 648 | Lions |
| 18 | 8 | Bryce Young | 75.10 | 477 | Panthers |
| 19 | 9 | Jordan Love | 74.30 | 528 | Packers |

### Starter (16 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 20 | 1 | Bo Nix | 73.80 | 712 | Broncos |
| 21 | 2 | Trevor Lawrence | 73.80 | 332 | Jaguars |
| 22 | 3 | Matthew Stafford | 73.30 | 667 | Rams |
| 23 | 4 | Kirk Cousins | 72.30 | 521 | Falcons |
| 24 | 5 | Joe Flacco | 70.70 | 290 | Colts |
| 25 | 6 | Tua Tagovailoa | 70.20 | 460 | Dolphins |
| 26 | 7 | Jameis Winston | 69.90 | 347 | Browns |
| 27 | 8 | Jalen Hurts | 69.50 | 558 | Eagles |
| 28 | 9 | Daniel Jones | 67.50 | 418 | Vikings |
| 29 | 10 | Andy Dalton | 67.40 | 185 | Panthers |
| 30 | 11 | Dak Prescott | 67.20 | 344 | Cowboys |
| 31 | 12 | Justin Fields | 65.70 | 215 | Steelers |
| 32 | 13 | Drake Maye | 64.90 | 461 | Patriots |
| 33 | 14 | Deshaun Watson | 63.40 | 290 | Browns |
| 34 | 15 | Caleb Williams | 62.90 | 741 | Bears |
| 35 | 16 | Mac Jones | 62.10 | 309 | Jaguars |

### Rotation/backup (12 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 36 | 1 | Gardner Minshew | 61.70 | 370 | Raiders |
| 37 | 2 | Tyler Huntley | 61.60 | 182 | Dolphins |
| 38 | 3 | Mason Rudolph | 61.50 | 276 | Titans |
| 39 | 4 | Anthony Richardson | 59.80 | 317 | Colts |
| 40 | 5 | Cooper Rush | 59.80 | 352 | Cowboys |
| 41 | 6 | Jacoby Brissett | 58.50 | 200 | Patriots |
| 42 | 7 | Aidan O'Connell | 58.20 | 276 | Raiders |
| 43 | 8 | Will Levis | 54.60 | 384 | Titans |
| 44 | 9 | Spencer Rattler | 49.40 | 284 | Saints |
| 45 | 10 | Desmond Ridder | 49.00 | 105 | Raiders |
| 46 | 11 | Drew Lock | 48.20 | 216 | Giants |
| 47 | 12 | Dorian Thompson-Robinson | 39.30 | 142 | Browns |

## S — Safety

- **Season used:** `2024`
- **Grade column:** `grades_defense` · **Snap column:** `snap_counts_defense`

### Elite (12 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | Kerby Joseph | 90.40 | 1158 | Lions |
| 2 | 2 | Kyle Hamilton | 90.00 | 1150 | Ravens |
| 3 | 3 | Tony Jefferson | 89.00 | 261 | Chargers |
| 4 | 4 | Brandon Jones | 84.80 | 1042 | Broncos |
| 5 | 5 | Xavier McKinney | 84.70 | 1125 | Packers |
| 6 | 6 | Brian Branch | 83.70 | 982 | Lions |
| 7 | 7 | Jabrill Peppers | 83.10 | 372 | Patriots |
| 8 | 8 | Julian Love | 81.20 | 1079 | Seahawks |
| 9 | 9 | Ar'Darius Washington | 80.90 | 830 | Ravens |
| 10 | 10 | Derwin James Jr. | 80.80 | 1059 | Chargers |
| 11 | 11 | Zayne Anderson | 80.80 | 122 | Packers |
| 12 | 12 | Jessie Bates III | 80.20 | 1095 | Falcons |

### Good (8 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 13 | 1 | Andrew Wingard | 78.60 | 216 | Jaguars |
| 14 | 2 | Thomas Harper | 78.30 | 191 | Raiders |
| 15 | 3 | Budda Baker | 77.80 | 1064 | Cardinals |
| 16 | 4 | Justin Reid | 77.70 | 1112 | Chiefs |
| 17 | 5 | Ronnie Hickman Jr. | 77.20 | 463 | Browns |
| 18 | 6 | Jimmie Ward | 77.10 | 461 | Texans |
| 19 | 7 | Dadrion Taylor-Demerson | 75.50 | 258 | Cardinals |
| 20 | 8 | C.J. Gardner-Johnson | 75.40 | 1118 | Eagles |

### Starter (45 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 21 | 1 | Dell Pettus | 73.90 | 341 | Patriots |
| 22 | 2 | Jalen Pitre | 73.90 | 660 | Texans |
| 23 | 3 | Quandre Diggs | 73.20 | 419 | Titans |
| 24 | 4 | Jaden Hicks | 73.00 | 430 | Chiefs |
| 25 | 5 | Marcus Maye | 72.90 | 405 | Chargers |
| 26 | 6 | Evan Williams | 72.50 | 533 | Packers |
| 27 | 7 | Reed Blankenship | 72.00 | 1030 | Eagles |
| 28 | 8 | Kevin Byard | 72.00 | 1055 | Bears |
| 29 | 9 | Kamren Kinchens | 71.50 | 623 | Rams |
| 30 | 10 | DeShon Elliott | 71.10 | 895 | Steelers |
| 31 | 11 | Ashtyn Davis | 71.10 | 260 | Jets |
| 32 | 12 | George Odum | 70.50 | 139 | 49ers |
| 33 | 13 | Nick Cross | 70.30 | 1156 | Colts |
| 34 | 14 | Mike Edwards | 70.10 | 251 | Buccaneers |
| 35 | 15 | Julian Blackmon | 69.30 | 1084 | Colts |
| 36 | 16 | Mike Brown | 69.20 | 384 | Titans |
| 37 | 17 | Jalen Thompson | 68.80 | 941 | Cardinals |
| 38 | 18 | Kamren Curl | 68.40 | 1112 | Rams |
| 39 | 19 | Jordan Howden | 67.90 | 550 | Saints |
| 40 | 20 | Harrison Smith | 67.70 | 1062 | Vikings |
| 41 | 21 | Tre'von Moehrig | 67.50 | 1099 | Raiders |
| 42 | 22 | Kaevon Merriweather | 66.50 | 274 | Buccaneers |
| 43 | 23 | Jonathan Owens | 66.50 | 429 | Bears |
| 44 | 24 | Tony Adams | 66.40 | 764 | Jets |
| 45 | 25 | Tyler Nubin | 65.60 | 789 | Giants |
| 46 | 26 | Juan Thornhill | 65.50 | 401 | Browns |
| 47 | 27 | Malik Hooker | 65.30 | 1062 | Cowboys |
| 48 | 28 | Jaquan Brisker | 65.30 | 293 | Bears |
| 49 | 29 | Josh Metellus | 65.20 | 1030 | Vikings |
| 50 | 30 | Minkah Fitzpatrick | 65.20 | 1158 | Steelers |
| 51 | 31 | Jeremy Chinn | 65.20 | 1207 | Commanders |
| 52 | 32 | Grant Delpit | 65.20 | 976 | Browns |
| 53 | 33 | Amani Hooker | 65.10 | 848 | Titans |
| 54 | 34 | Ji'Ayir Brown | 64.80 | 886 | 49ers |
| 55 | 35 | Quentin Lake | 64.70 | 1207 | Rams |
| 56 | 36 | Alohi Gilman | 64.30 | 731 | Chargers |
| 57 | 37 | Malik Mustapha | 63.90 | 755 | 49ers |
| 58 | 38 | Eric Murray | 63.80 | 961 | Texans |
| 59 | 39 | Devon Key | 63.10 | 253 | Broncos |
| 60 | 40 | Jevon Holland | 63.00 | 854 | Dolphins |
| 61 | 41 | Jordan Poyer | 62.50 | 964 | Dolphins |
| 62 | 42 | Will Harris | 62.50 | 860 | Saints |
| 63 | 43 | Dane Belton | 62.50 | 460 | Giants |
| 64 | 44 | Donovan Wilson | 62.20 | 1008 | Cowboys |
| 65 | 45 | Vonn Bell | 62.10 | 705 | Bengals |

### Rotation/backup (38 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 66 | 1 | Camryn Bynum | 61.60 | 1056 | Vikings |
| 67 | 2 | Damontae Kazee | 60.80 | 313 | Steelers |
| 68 | 3 | Tyrann Mathieu | 60.50 | 1015 | Saints |
| 69 | 4 | Jaylinn Hawkins | 60.10 | 613 | Patriots |
| 70 | 5 | Demani Richardson | 60.10 | 403 | Panthers |
| 71 | 6 | Justin Simmons | 59.90 | 1017 | Falcons |
| 72 | 7 | Andre Cisco | 58.80 | 979 | Jaguars |
| 73 | 8 | Jaylen McCollough | 58.50 | 382 | Rams |
| 74 | 9 | Jordan Whitehead | 58.00 | 731 | Buccaneers |
| 75 | 10 | Talanoa Hufanga | 57.80 | 308 | 49ers |
| 76 | 11 | Antoine Winfield Jr. | 57.80 | 601 | Buccaneers |
| 77 | 12 | Taylor Rapp | 57.70 | 840 | Bills |
| 78 | 13 | Christian Izien | 57.60 | 697 | Buccaneers |
| 79 | 14 | Xavier Woods | 57.00 | 1216 | Panthers |
| 80 | 15 | Bryan Cook | 56.40 | 1056 | Chiefs |
| 81 | 16 | Jason Pinnock | 54.50 | 976 | Giants |
| 82 | 17 | Javon Bullard | 54.20 | 816 | Packers |
| 83 | 18 | Isaiah Pola-Mao | 54.20 | 952 | Raiders |
| 84 | 19 | Jordan Fuller | 53.80 | 574 | Panthers |
| 85 | 20 | Geno Stone | 53.10 | 1100 | Bengals |
| 86 | 21 | Jordan Battle | 53.10 | 464 | Bengals |
| 87 | 22 | Damar Hamlin | 52.70 | 1042 | Bills |
| 88 | 23 | Nick Scott | 52.60 | 324 | Panthers |
| 89 | 24 | Rayshawn Jenkins | 52.50 | 550 | Seahawks |
| 90 | 25 | Eddie Jackson | 52.00 | 390 | Chargers |
| 91 | 26 | Cole Bishop | 52.00 | 464 | Bills |
| 92 | 27 | P.J. Locke | 51.20 | 1076 | Broncos |
| 93 | 28 | Rodney McLeod | 50.90 | 565 | Browns |
| 94 | 29 | Chuck Clark | 50.80 | 709 | Jets |
| 95 | 30 | Antonio Johnson | 50.10 | 685 | Jaguars |
| 96 | 31 | Calen Bullock | 47.90 | 1083 | Texans |
| 97 | 32 | Marcus Epps | 46.90 | 176 | Raiders |
| 98 | 33 | Darnell Savage | 46.20 | 764 | Jaguars |
| 99 | 34 | Kyle Dugger | 44.30 | 759 | Patriots |
| 100 | 35 | Marcus Williams | 42.90 | 601 | Ravens |
| 101 | 36 | Percy Butler | 41.30 | 448 | Commanders |
| 102 | 37 | K'Von Wallace | 38.10 | 127 | Seahawks |
| 103 | 38 | Richie Grant | 37.70 | 165 | Falcons |

## T — Tackle

- **Season used:** `2024`
- **Grade column:** `grades_offense` · **Snap column:** `snap_counts_offense`

### Elite (14 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | Jordan Mailata | 95.20 | 995 | Eagles |
| 2 | 2 | Rashawn Slater | 90.90 | 959 | Chargers |
| 3 | 3 | Penei Sewell | 89.60 | 1213 | Lions |
| 4 | 4 | Terron Armstead | 89.40 | 821 | Dolphins |
| 5 | 5 | Lane Johnson | 87.50 | 1123 | Eagles |
| 6 | 6 | Zach Tom | 85.80 | 1134 | Packers |
| 7 | 7 | Trent Williams | 85.60 | 649 | 49ers |
| 8 | 8 | Bernhard Raimann | 85.10 | 856 | Colts |
| 9 | 9 | Tristan Wirfs | 82.50 | 1061 | Buccaneers |
| 10 | 10 | Charles Cross | 82.50 | 1094 | Seahawks |
| 11 | 11 | Christian Darrisaw | 81.40 | 392 | Vikings |
| 12 | 12 | Paris Johnson Jr. | 80.80 | 865 | Cardinals |
| 13 | 13 | Kolton Miller | 80.60 | 1075 | Raiders |
| 14 | 14 | Garett Bolles | 80.20 | 1111 | Broncos |

### Good (15 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 15 | 1 | Jake Matthews | 79.80 | 1119 | Falcons |
| 16 | 2 | Darnell Wright | 79.30 | 1021 | Bears |
| 17 | 3 | Brian O'Neill | 79.30 | 1151 | Vikings |
| 18 | 4 | Alaric Jackson | 78.40 | 1017 | Rams |
| 19 | 5 | Laremy Tunsil | 78.10 | 1167 | Texans |
| 20 | 6 | Spencer Brown | 77.90 | 1140 | Bills |
| 21 | 7 | Braxton Jones | 77.40 | 719 | Bears |
| 22 | 8 | Taylor Moton | 77.20 | 846 | Panthers |
| 23 | 9 | Taylor Decker | 77.20 | 963 | Lions |
| 24 | 10 | Joe Alt | 75.90 | 1066 | Chargers |
| 25 | 11 | Rob Havenstein | 75.80 | 805 | Rams |
| 26 | 12 | Andrew Thomas | 75.40 | 416 | Giants |
| 27 | 13 | Jaylon Moore | 74.90 | 271 | 49ers |
| 28 | 14 | Luke Goedeke | 74.20 | 952 | Buccaneers |
| 29 | 15 | Cornelius Lucas | 74.10 | 464 | Commanders |

### Starter (32 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 30 | 1 | Kaleb McGary | 73.90 | 1042 | Falcons |
| 31 | 2 | Tyron Smith | 73.70 | 592 | Jets |
| 32 | 3 | Walker Little | 72.80 | 508 | Jaguars |
| 33 | 4 | Kendall Lamm | 72.70 | 512 | Dolphins |
| 34 | 5 | Mike McGlinchey | 72.60 | 891 | Broncos |
| 35 | 6 | Dion Dawkins | 72.40 | 1164 | Bills |
| 36 | 7 | Colton McKivitz | 72.20 | 1062 | 49ers |
| 37 | 8 | Ikem Ekwonu | 71.70 | 909 | Panthers |
| 38 | 9 | Ronnie Stanley | 70.70 | 1221 | Ravens |
| 39 | 10 | Jonah Williams | 70.70 | 343 | Cardinals |
| 40 | 11 | Tytus Howard | 70.20 | 1157 | Texans |
| 41 | 12 | John Ojukwu | 69.40 | 264 | Titans |
| 42 | 13 | Justin Skule | 69.20 | 362 | Buccaneers |
| 43 | 14 | Rasheed Walker | 68.60 | 1139 | Packers |
| 44 | 15 | Matt Peart | 67.40 | 190 | Broncos |
| 45 | 16 | Dan Moore Jr. | 67.20 | 1128 | Steelers |
| 46 | 17 | Terence Steele | 67.00 | 1168 | Cowboys |
| 47 | 18 | Jack Conklin | 66.20 | 818 | Browns |
| 48 | 19 | Braden Smith | 66.20 | 731 | Colts |
| 49 | 20 | DJ Glaze | 66.10 | 998 | Raiders |
| 50 | 21 | Roger Rosengarten | 66.00 | 1066 | Ravens |
| 51 | 22 | Matt Goncalves | 65.90 | 566 | Colts |
| 52 | 23 | Taliese Fuaga | 65.70 | 1070 | Saints |
| 53 | 24 | Trent Brown | 65.20 | 139 | Bengals |
| 54 | 25 | Anton Harrison | 64.20 | 943 | Jaguars |
| 55 | 26 | Kelvin Beachum | 64.10 | 742 | Cardinals |
| 56 | 27 | Jackson Barton | 63.90 | 157 | Cardinals |
| 57 | 28 | Alex Palczewski | 63.40 | 179 | Broncos |
| 58 | 29 | Morgan Moses | 63.30 | 723 | Jets |
| 59 | 30 | Cam Robinson | 63.20 | 1073 | Vikings |
| 60 | 31 | Joshua Ezeudu | 62.70 | 182 | Giants |
| 61 | 32 | Cole Van Lanen | 62.30 | 252 | Jaguars |

### Rotation/backup (43 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 62 | 1 | Abraham Lucas | 61.90 | 406 | Seahawks |
| 63 | 2 | Storm Norton | 61.90 | 128 | Falcons |
| 64 | 3 | JC Latham | 61.80 | 1095 | Titans |
| 65 | 4 | Andrew Wylie | 61.70 | 1115 | Commanders |
| 66 | 5 | Olumuyiwa Fashanu | 61.20 | 534 | Jets |
| 67 | 6 | Evan Neal | 61.20 | 459 | Giants |
| 68 | 7 | Trevor Penning | 60.20 | 1081 | Saints |
| 69 | 8 | Joe Noteboom | 60.00 | 220 | Rams |
| 70 | 9 | Austin Jackson | 60.00 | 542 | Dolphins |
| 71 | 10 | Warren McClendon Jr. | 59.80 | 333 | Rams |
| 72 | 11 | Jawaan Taylor | 59.80 | 1209 | Chiefs |
| 73 | 12 | Brandon Coleman | 59.80 | 1013 | Commanders |
| 74 | 13 | Broderick Jones | 58.70 | 1117 | Steelers |
| 75 | 14 | Orlando Brown Jr. | 58.20 | 637 | Bengals |
| 76 | 15 | Yosh Nijman | 57.90 | 187 | Panthers |
| 77 | 16 | Amarius Mims | 57.80 | 835 | Bengals |
| 78 | 17 | David Quessenberry | 55.20 | 133 | Vikings |
| 79 | 18 | Dan Skipper | 55.20 | 324 | Lions |
| 80 | 19 | Vederian Lowe | 54.00 | 803 | Patriots |
| 81 | 20 | Larry Borom | 53.80 | 329 | Bears |
| 82 | 21 | Ryan Van Demark | 53.30 | 199 | Bills |
| 83 | 22 | Wanya Morris | 53.00 | 732 | Chiefs |
| 84 | 23 | Jedrick Wills Jr. | 52.90 | 245 | Browns |
| 85 | 24 | Chuma Edoga | 52.50 | 226 | Cowboys |
| 86 | 25 | Devin Cochran | 50.60 | 152 | Bengals |
| 87 | 26 | James Hudson III | 50.40 | 222 | Browns |
| 88 | 27 | Tyler Guyton | 49.40 | 668 | Cowboys |
| 89 | 28 | Fred Johnson | 49.30 | 490 | Eagles |
| 90 | 29 | Chris Hubbard | 46.60 | 257 | Giants |
| 91 | 30 | Nicholas Petit-Frere | 46.50 | 621 | Titans |
| 92 | 31 | Mike Jerrell | 46.40 | 250 | Seahawks |
| 93 | 32 | Dawand Jones | 46.40 | 511 | Browns |
| 94 | 33 | Trent Scott | 46.10 | 288 | Commanders |
| 95 | 34 | Thayer Munford Jr. | 45.90 | 201 | Raiders |
| 96 | 35 | Patrick Paul | 44.90 | 338 | Dolphins |
| 97 | 36 | Blake Fisher | 44.70 | 478 | Texans |
| 98 | 37 | Carter Warren | 44.60 | 141 | Jets |
| 99 | 38 | Caedan Wallace | 44.10 | 129 | Patriots |
| 100 | 39 | Stone Forsythe | 43.10 | 414 | Seahawks |
| 101 | 40 | Charlie Heck | 41.20 | 117 | 49ers |
| 102 | 41 | Kiran Amegadjie | 40.30 | 126 | Bears |
| 103 | 42 | Demontrey Jacobs | 38.40 | 867 | Patriots |
| 104 | 43 | Kingsley Suamataia | 37.90 | 198 | Chiefs |

## TE — Tight End

- **Season used:** `2024`
- **Grade column:** `grades_offense` · **Snap column:** `total_snaps`

### Elite (4 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | George Kittle | 92.10 | 489 | 49ers |
| 2 | 2 | Trey McBride | 86.80 | 581 | Cardinals |
| 3 | 3 | Brock Bowers | 85.10 | 654 | Raiders |
| 4 | 4 | Mark Andrews | 83.10 | 455 | Ravens |

### Good (5 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 5 | 1 | Jonnu Smith | 78.20 | 486 | Dolphins |
| 6 | 2 | Austin Hooper | 75.80 | 335 | Patriots |
| 7 | 3 | Isaiah Likely | 75.60 | 386 | Ravens |
| 8 | 4 | T.J. Hockenson | 74.80 | 366 | Vikings |
| 9 | 5 | Josh Oliver | 74.30 | 254 | Vikings |

### Starter (29 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 10 | 1 | Sam LaPorta | 73.60 | 546 | Lions |
| 11 | 2 | Dallas Goedert | 73.30 | 325 | Eagles |
| 12 | 3 | Travis Kelce | 72.70 | 698 | Chiefs |
| 13 | 4 | Evan Engram | 72.50 | 260 | Jaguars |
| 14 | 5 | Noah Gray | 71.80 | 380 | Chiefs |
| 15 | 6 | Taysom Hill | 71.80 | 103 | Saints |
| 16 | 7 | Mike Gesicki | 71.50 | 449 | Bengals |
| 17 | 8 | Dalton Kincaid | 71.50 | 365 | Bills |
| 18 | 9 | Darnell Washington | 71.20 | 257 | Steelers |
| 19 | 10 | Foster Moreau | 71.10 | 395 | Saints |
| 20 | 11 | Hunter Henry | 70.00 | 548 | Patriots |
| 21 | 12 | Chris Manhertz | 69.30 | 139 | Giants |
| 22 | 13 | Andrew Ogletree | 69.10 | 173 | Colts |
| 23 | 14 | Tucker Kraft | 67.80 | 538 | Packers |
| 24 | 15 | Jordan Akins | 67.20 | 348 | Browns |
| 25 | 16 | Pat Freiermuth | 67.20 | 516 | Steelers |
| 26 | 17 | Will Dissly | 67.10 | 361 | Chargers |
| 27 | 18 | Zach Ertz | 67.00 | 658 | Commanders |
| 28 | 19 | Juwan Johnson | 66.70 | 467 | Saints |
| 29 | 20 | Hunter Long | 66.60 | 110 | Rams |
| 30 | 21 | Stone Smartt | 66.40 | 138 | Chargers |
| 31 | 22 | Noah Fant | 66.00 | 426 | Seahawks |
| 32 | 23 | Brenton Strange | 66.00 | 320 | Jaguars |
| 33 | 24 | Josh Whyle | 64.80 | 204 | Titans |
| 34 | 25 | Mo Alie-Cox | 64.50 | 227 | Colts |
| 35 | 26 | Cade Otton | 64.10 | 573 | Buccaneers |
| 36 | 27 | David Njoku | 64.00 | 415 | Browns |
| 37 | 28 | Nate Adkins | 63.90 | 181 | Broncos |
| 38 | 29 | Colby Parkinson | 63.20 | 390 | Rams |

### Rotation/backup (41 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 39 | 1 | AJ Barner | 61.00 | 253 | Seahawks |
| 40 | 2 | Payne Durham | 60.80 | 191 | Buccaneers |
| 41 | 3 | Dalton Schultz | 60.80 | 648 | Texans |
| 42 | 4 | Cole Kmet | 60.60 | 637 | Bears |
| 43 | 5 | Harrison Bryant | 60.00 | 101 | Raiders |
| 44 | 6 | Erick All | 59.90 | 112 | Bengals |
| 45 | 7 | Chigoziem Okonkwo | 59.90 | 425 | Titans |
| 46 | 8 | Kyle Pitts | 59.60 | 511 | Falcons |
| 47 | 9 | Charlie Woerner | 58.90 | 131 | Falcons |
| 48 | 10 | Tyler Conklin | 58.80 | 556 | Jets |
| 49 | 11 | Kylen Granson | 58.30 | 268 | Colts |
| 50 | 12 | Luke Schoonmaker | 58.20 | 213 | Cowboys |
| 51 | 13 | Luke Farrell | 57.70 | 152 | Jaguars |
| 52 | 14 | Michael Mayer | 57.70 | 284 | Raiders |
| 53 | 15 | Daniel Bellinger | 57.20 | 207 | Giants |
| 54 | 16 | Johnny Mundt | 57.20 | 236 | Vikings |
| 55 | 17 | Dawson Knox | 57.10 | 392 | Bills |
| 56 | 18 | Adam Trautman | 56.90 | 288 | Broncos |
| 57 | 19 | Jake Ferguson | 54.50 | 427 | Cowboys |
| 58 | 20 | Tommy Tremble | 54.40 | 303 | Panthers |
| 59 | 21 | Nick Vannett | 54.30 | 145 | Titans |
| 60 | 22 | Brevyn Spann-Ford | 53.70 | 146 | Cowboys |
| 61 | 23 | Theo Johnson | 53.60 | 446 | Giants |
| 62 | 24 | Grant Calcaterra | 53.20 | 347 | Eagles |
| 63 | 25 | Ben Sinnott | 53.20 | 122 | Commanders |
| 64 | 26 | Ja'Tavion Sanders | 52.50 | 359 | Panthers |
| 65 | 27 | Lucas Krull | 52.40 | 252 | Broncos |
| 66 | 28 | Cade Stover | 52.10 | 192 | Texans |
| 67 | 29 | Brock Wright | 52.00 | 235 | Lions |
| 68 | 30 | Eric Saubert | 51.20 | 177 | 49ers |
| 69 | 31 | Tip Reiman | 49.30 | 169 | Cardinals |
| 70 | 32 | Drew Sample | 48.90 | 278 | Bengals |
| 71 | 33 | John Bates | 47.30 | 252 | Commanders |
| 72 | 34 | Jeremy Ruckert | 46.70 | 198 | Jets |
| 73 | 35 | Davis Allen | 46.40 | 176 | Rams |
| 74 | 36 | Pharaoh Brown | 45.60 | 107 | Seahawks |
| 75 | 37 | Durham Smythe | 43.30 | 156 | Dolphins |
| 76 | 38 | Gerald Everett | 42.90 | 133 | Bears |
| 77 | 39 | Greg Dulcich | 37.70 | 127 | Giants |
| 78 | 40 | Hayden Hurst | 37.30 | 103 | Chargers |
| 79 | 41 | Julian Hill | 37.20 | 228 | Dolphins |

## WR — Wide Receiver

- **Season used:** `2024`
- **Grade column:** `grades_offense` · **Snap column:** `total_snaps`

### Elite (20 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 1 | 1 | Puka Nacua | 92.50 | 370 | Rams |
| 2 | 2 | Nico Collins | 92.30 | 447 | Texans |
| 3 | 3 | Mike Evans | 90.40 | 463 | Buccaneers |
| 4 | 4 | A.J. Brown | 90.30 | 473 | Eagles |
| 5 | 5 | Amon-Ra St. Brown | 89.30 | 611 | Lions |
| 6 | 6 | Tee Higgins | 88.20 | 476 | Bengals |
| 7 | 7 | Drake London | 87.80 | 595 | Falcons |
| 8 | 8 | Malik Nabers | 86.70 | 600 | Giants |
| 9 | 9 | Justin Jefferson | 86.60 | 700 | Vikings |
| 10 | 10 | Chris Godwin | 86.30 | 265 | Buccaneers |
| 11 | 11 | Ja'Marr Chase | 85.80 | 745 | Bengals |
| 12 | 12 | Josh Downs | 84.80 | 381 | Colts |
| 13 | 13 | Ladd McConkey | 84.30 | 553 | Chargers |
| 14 | 14 | Jauan Jennings | 83.10 | 459 | 49ers |
| 15 | 15 | Chris Olave | 82.40 | 200 | Saints |
| 16 | 16 | Zay Flowers | 82.10 | 499 | Ravens |
| 17 | 17 | Terry McLaurin | 82.10 | 717 | Commanders |
| 18 | 18 | Brian Thomas Jr. | 82.00 | 552 | Jaguars |
| 19 | 19 | Jaxon Smith-Njigba | 81.00 | 666 | Seahawks |
| 20 | 20 | DeVonta Smith | 80.90 | 505 | Eagles |

### Good (17 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 21 | 1 | Stefon Diggs | 79.00 | 282 | Texans |
| 22 | 2 | Khalil Shakir | 78.90 | 495 | Bills |
| 23 | 3 | Garrett Wilson | 78.90 | 691 | Jets |
| 24 | 4 | George Pickens | 78.60 | 498 | Steelers |
| 25 | 5 | Jordan Whittington | 78.30 | 129 | Rams |
| 26 | 6 | Jakobi Meyers | 77.70 | 627 | Raiders |
| 27 | 7 | Marvin Harrison Jr. | 77.70 | 579 | Cardinals |
| 28 | 8 | DeAndre Hopkins | 77.70 | 419 | Chiefs |
| 29 | 9 | CeeDee Lamb | 77.30 | 566 | Cowboys |
| 30 | 10 | Adam Thielen | 76.40 | 319 | Panthers |
| 31 | 11 | Davante Adams | 75.80 | 557 | Jets |
| 32 | 12 | Courtland Sutton | 75.50 | 650 | Broncos |
| 33 | 13 | Brandon Aiyuk | 74.60 | 225 | 49ers |
| 34 | 14 | Jameson Williams | 74.50 | 535 | Lions |
| 35 | 15 | Alec Pierce | 74.30 | 485 | Colts |
| 36 | 16 | D.K. Metcalf | 74.30 | 590 | Seahawks |
| 37 | 17 | Darnell Mooney | 74.00 | 557 | Falcons |

### Starter (59 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 38 | 1 | DJ Moore | 73.50 | 722 | Bears |
| 39 | 2 | Jerry Jeudy | 73.50 | 757 | Browns |
| 40 | 3 | Calvin Ridley | 73.40 | 582 | Titans |
| 41 | 4 | Tank Dell | 73.10 | 467 | Texans |
| 42 | 5 | Jordan Addison | 72.90 | 600 | Vikings |
| 43 | 6 | Jalen Coker | 72.80 | 297 | Panthers |
| 44 | 7 | Tyreek Hill | 72.70 | 580 | Dolphins |
| 45 | 8 | Tutu Atwell | 72.70 | 277 | Rams |
| 46 | 9 | Michael Pittman Jr. | 72.20 | 511 | Colts |
| 47 | 10 | Jaylen Waddle | 72.10 | 513 | Dolphins |
| 48 | 11 | Rashod Bateman | 71.80 | 522 | Ravens |
| 49 | 12 | Jayden Reed | 71.70 | 430 | Packers |
| 50 | 13 | Cooper Kupp | 71.40 | 455 | Rams |
| 51 | 14 | KaVontae Turpin | 71.20 | 210 | Cowboys |
| 52 | 15 | Kalif Raymond | 71.10 | 158 | Lions |
| 53 | 16 | Deebo Samuel | 70.90 | 405 | 49ers |
| 54 | 17 | Noah Brown | 70.70 | 295 | Commanders |
| 55 | 18 | Dante Pettis | 70.20 | 104 | Saints |
| 56 | 19 | Demario Douglas | 70.00 | 477 | Patriots |
| 57 | 20 | Rashid Shaheed | 69.70 | 181 | Saints |
| 58 | 21 | Romeo Doubs | 69.70 | 406 | Packers |
| 59 | 22 | Devaughn Vele | 69.30 | 349 | Broncos |
| 60 | 23 | Olamide Zaccheaus | 69.20 | 403 | Commanders |
| 61 | 24 | Christian Watson | 69.10 | 295 | Packers |
| 62 | 25 | Amari Cooper | 68.70 | 451 | Bills |
| 63 | 26 | Xavier Worthy | 68.70 | 603 | Chiefs |
| 64 | 27 | Marvin Mims Jr. | 68.60 | 198 | Broncos |
| 65 | 28 | Keon Coleman | 68.30 | 404 | Bills |
| 66 | 29 | Christian Kirk | 67.90 | 230 | Jaguars |
| 67 | 30 | Quentin Johnston | 67.40 | 464 | Chargers |
| 68 | 31 | Tyler Johnson | 67.00 | 215 | Rams |
| 69 | 32 | Joshua Palmer | 67.00 | 428 | Chargers |
| 70 | 33 | Tim Patrick | 66.80 | 371 | Lions |
| 71 | 34 | Dyami Brown | 66.50 | 377 | Commanders |
| 72 | 35 | Tylan Wallace | 66.20 | 148 | Ravens |
| 73 | 36 | Curtis Samuel | 65.50 | 263 | Bills |
| 74 | 37 | Tyler Lockett | 65.20 | 589 | Seahawks |
| 75 | 38 | Dontayvion Wicks | 65.10 | 345 | Packers |
| 76 | 39 | Diontae Johnson | 65.10 | 277 | Texans |
| 77 | 40 | Demarcus Robinson | 65.00 | 618 | Rams |
| 78 | 41 | Keenan Allen | 64.40 | 596 | Bears |
| 79 | 42 | Ryan Flournoy | 63.90 | 103 | Cowboys |
| 80 | 43 | Greg Dortch | 63.90 | 299 | Cardinals |
| 81 | 44 | Ricky Pearsall | 63.90 | 324 | 49ers |
| 82 | 45 | Rome Odunze | 63.80 | 677 | Bears |
| 83 | 46 | Calvin Austin III | 63.80 | 423 | Steelers |
| 84 | 47 | Cedric Tillman | 63.60 | 300 | Browns |
| 85 | 48 | Cedrick Wilson Jr. | 63.50 | 220 | Saints |
| 86 | 49 | Nick Westbrook-Ikhine | 63.40 | 469 | Titans |
| 87 | 50 | Nelson Agholor | 63.40 | 250 | Ravens |
| 88 | 51 | Wan'Dale Robinson | 63.40 | 618 | Giants |
| 89 | 52 | Bo Melton | 63.20 | 118 | Packers |
| 90 | 53 | Brandin Cooks | 63.20 | 317 | Cowboys |
| 91 | 54 | David Moore | 63.10 | 358 | Panthers |
| 92 | 55 | Michael Wilson | 62.90 | 538 | Cardinals |
| 93 | 56 | KhaDarel Hodge | 62.80 | 121 | Falcons |
| 94 | 57 | Allen Lazard | 62.70 | 451 | Jets |
| 95 | 58 | Ray-Ray McCloud III | 62.70 | 598 | Falcons |
| 96 | 59 | Kendrick Bourne | 62.10 | 317 | Patriots |

### Rotation/backup (50 players)

| rank_pos | rank_in_tier | player | grade | snaps | primary_team |
|---:|---:|---|---:|---:|---|
| 97 | 1 | Marquez Valdes-Scantling | 61.70 | 315 | Saints |
| 98 | 2 | Mack Hollins | 61.60 | 495 | Bills |
| 99 | 3 | Kayshon Boutte | 61.40 | 507 | Patriots |
| 100 | 4 | Josh Reynolds | 61.20 | 220 | Jaguars |
| 101 | 5 | Jalen McMillan | 60.80 | 430 | Buccaneers |
| 102 | 6 | Jalen Tolbert | 60.70 | 597 | Cowboys |
| 103 | 7 | Tyler Boyd | 60.00 | 464 | Titans |
| 104 | 8 | JuJu Smith-Schuster | 60.00 | 321 | Chiefs |
| 105 | 9 | Chris Conley | 59.80 | 124 | 49ers |
| 106 | 10 | Parker Washington | 59.70 | 404 | Jaguars |
| 107 | 11 | Rakim Jarrett | 59.50 | 117 | Buccaneers |
| 108 | 12 | Jalen Nailor | 59.30 | 462 | Vikings |
| 109 | 13 | John Metchie III | 59.30 | 314 | Texans |
| 110 | 14 | Devin Duvernay | 59.30 | 141 | Jaguars |
| 111 | 15 | Xavier Legette | 59.30 | 441 | Panthers |
| 112 | 16 | Jalen Brooks | 59.20 | 236 | Cowboys |
| 113 | 17 | Darius Slayton | 59.00 | 575 | Giants |
| 114 | 18 | Ryan Miller | 59.00 | 151 | Buccaneers |
| 115 | 19 | Brandon Powell | 58.90 | 130 | Vikings |
| 116 | 20 | Robert Woods | 58.80 | 226 | Texans |
| 117 | 21 | Mike Williams | 58.80 | 366 | Steelers |
| 118 | 22 | Elijah Moore | 58.50 | 623 | Browns |
| 119 | 23 | Jermaine Burton | 58.40 | 100 | Bengals |
| 120 | 24 | Trey Palmer | 58.20 | 188 | Buccaneers |
| 121 | 25 | Adonai Mitchell | 57.90 | 221 | Colts |
| 122 | 26 | Xavier Hutchinson | 57.80 | 326 | Texans |
| 123 | 27 | Malik Washington | 57.70 | 270 | Dolphins |
| 124 | 28 | Tre Tucker | 57.50 | 683 | Raiders |
| 125 | 29 | Van Jefferson | 57.20 | 442 | Steelers |
| 126 | 30 | Kevin Austin Jr. | 56.00 | 210 | Saints |
| 127 | 31 | Jake Bobo | 55.40 | 163 | Seahawks |
| 128 | 32 | Simi Fehoko | 55.00 | 136 | Chargers |
| 129 | 33 | DJ Turner | 54.90 | 246 | Raiders |
| 130 | 34 | Jamison Crowder | 54.90 | 127 | Commanders |
| 131 | 35 | Sterling Shepard | 54.80 | 393 | Buccaneers |
| 132 | 36 | Luke McCaffrey | 54.30 | 283 | Commanders |
| 133 | 37 | Troy Franklin | 54.30 | 297 | Broncos |
| 134 | 38 | Jahan Dotson | 53.70 | 492 | Eagles |
| 135 | 39 | Justin Watson | 53.30 | 430 | Chiefs |
| 136 | 40 | Zay Jones | 53.10 | 184 | Cardinals |
| 137 | 41 | Andrei Iosivas | 52.70 | 620 | Bengals |
| 138 | 42 | Gabe Davis | 52.50 | 264 | Jaguars |
| 139 | 43 | K.J. Osborn | 52.20 | 143 | Commanders |
| 140 | 44 | Jonathan Mingo | 49.90 | 285 | Cowboys |
| 141 | 45 | Johnny Wilson | 49.80 | 177 | Eagles |
| 142 | 46 | Michael Woods II | 48.60 | 211 | Browns |
| 143 | 47 | Xavier Gipson | 48.60 | 138 | Jets |
| 144 | 48 | Mason Tipton | 48.00 | 253 | Saints |
| 145 | 49 | Jalin Hyatt | 48.00 | 230 | Giants |
| 146 | 50 | Ja'Lynn Polk | 43.10 | 272 | Patriots |
