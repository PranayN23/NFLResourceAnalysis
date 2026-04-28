# NFL GM Agent — Free Agency Evaluation Platform

A full-stack NFL analytics platform that evaluates free agent contracts using PFF grades, statistical projections, team context, and a LangGraph agentic pipeline. The tool covers all 12 offensive and defensive position groups, produces signed/passed recommendations with year-by-year salary breakdowns, and includes a class builder that lets you model an entire team's offseason simultaneously.

---

## Table of Contents

1. [What it does](#1-what-it-does)
2. [How to run](#2-how-to-run)
3. [System design](#3-system-design)
4. [Evaluation pipeline (deep dive)](#4-evaluation-pipeline-deep-dive)
5. [Salary curves — concrete numbers](#5-salary-curves--concrete-numbers)
6. [Grade thresholds & stat anchors](#6-grade-thresholds--stat-anchors)
7. [Team context & cap math](#7-team-context--cap-math)
8. [Free Agency class builder](#8-free-agency-class-builder)
9. [Scheme & personnel adjustments](#9-scheme--personnel-adjustments)
10. [Position GM APIs — port map](#10-position-gm-apis--port-map)
11. [Data sources](#11-data-sources)
12. [Repository map](#12-repository-map)
13. [Change log — last ~70 commits](#13-change-log--last-70-commits)
14. [Known issues & backlog](#14-known-issues--backlog)

---

## 1. What it does

- **12 position-specific GM agents** — Each position (QB, HB, WR, TE, T, G, C, ED, DI, LB, CB, S) runs through a four-node LangGraph pipeline: grade prediction → contract valuation → team fit → decision. You get a tiered recommendation (e.g. "Good Signing", "Overpay", "Must Sign"), a reasoning paragraph, fair AAV vs cap burden, and projected stats for the contract duration.
- **Team mode** — Select a team and year; the agent fetches the current roster, scores positional need league-relative (0–100), applies cap-percentage math, and upgrades or downgrades the decision accordingly.
- **FA class builder** — Add multiple signings and model departures together. The builder computes a weighted signing-class grade and a net-roster grade that accounts for talent lost, positional coverage gaps, and cap freed by exits.
- **Historical analysis** — Every position can be analyzed for any year 2010–2032, with real historical cap figures, allowing retrospective contract evaluation.
- **GM chat feed** — Append-only assistant feed where each evaluation is a structured card. Grades, tiers, projections, and reasoning are laid out per signing.

---

## 2. How to run

### Prerequisites

- Python 3.11+, Node 18+
- Install Python deps: `pip install -r requirements.txt` (also needs `fastapi`, `uvicorn`, `langgraph`, `langchain_core`, `joblib`)
- Install frontend deps: `cd frontend && npm install`

### Start all 12 backend APIs

From the **repo root**:

```bash
bash backend/agent/run_all_free_agency_apis.sh
```

This starts 12 Uvicorn processes on ports **8002–8013**, one per position. Each process loads its position's CSV at startup.

**Background (with log):**
```bash
nohup bash backend/agent/run_all_free_agency_apis.sh > /tmp/nfl-fa-apis.log 2>&1 &
```

**Stop all:**
```bash
for p in 8002 8003 8004 8005 8006 8007 8008 8009 8010 8011 8012 8013; do
  lsof -ti :$p | xargs kill -9 2>/dev/null || true
done
```

> **Important:** Python is loaded once at process start. After editing any `backend/agent/*.py` file you must restart the relevant Uvicorn process — refreshing the browser does not reload Python.

**Smoke-test a single API:**
```bash
curl -s "http://127.0.0.1:8004/qb-players" | python3 -m json.tool | head -20
# Expect 200 and a players list
```

### Start the frontend

```bash
cd frontend
npm run dev
# Opens http://localhost:5173
```

Navigate to the **Free Agency** route. Select a position, player, AAV, years, optionally a team, then hit Analyze.

### Legacy single API (optional)

`backend/agent/main_api.py` runs on port **8000** — a simpler single-CSV evaluator not used by the FA page.

### Environment

`.env` at repo root — set `ANTHROPIC_API_KEY` (used by LLM reasoning in the agent). MongoDB URI is hardcoded in `hermesServer.py` for the legacy player-lookup service.

---

## 3. System design

```
Browser (React / Vite :5173)
  │
  │  POST /evaluate   GET /team-roster   GET /teams
  ▼
┌──────────────────────────────────────────────────────┐
│  12 FastAPI / Uvicorn apps  (ports 8002–8013)         │
│                                                      │
│  *_main_api.py  →  *_agent_graph.py  (LangGraph)     │
│                         │                            │
│  ┌──────────────────────┴──────────────────────┐     │
│  │   Node 1: predict_performance               │     │
│  │   Node 2: evaluate_value                    │     │
│  │   Node 3: assess_team_fit                   │     │
│  │   Node 4: make_decision                     │     │
│  └─────────────────────────────────────────────┘     │
│                         │                            │
│  market_value_curves.py  team_context.py             │
│  stat_projection_utils.py  scheme_personnel.py       │
│  grade_projection.py                                 │
└──────────────────────────────────────────────────────┘
             │                        │
         ML/*.csv               ML/cap_data.csv
   (per-position PFF data)    (team cap allocations)
```

**Frontend config:** `frontend/src/config/freeAgencyPositionConfig.js` — maps each position key to port, stat columns, and market tier legend. This **must stay in sync** with Python when ports or grade anchors change.

**Key design decisions:**
- Each position is a **separate process** so the QB agent can run heavy pandas aggregation independently of, say, the safety agent, and CSV loading is paid once per startup, not per request.
- **LangGraph** compiles a static graph (`_workflow.compile()`) per position module, so the node execution order (`predict → evaluate → fit → decide`) is explicit and inspectable.
- **Cap scaling** is multiplicative: values are calibrated to `VALUE_ANCHOR_CALIBRATION_YEAR = 2026` then scaled backward/forward via `(cap_year / cap_2026) ** position_exponent`. This means a grade-80 QB worth $52M in 2026 is automatically re-priced to ~$44M in 2023 at the 2023 cap, without re-fitting the entire curve.

---

## 4. Evaluation pipeline (deep dive)

### Node 1 — `predict_performance`

1. Aggregates the player's career PFF rows by year (handles split-team seasons, infers missing game counts from dropbacks/attempts for QB).
2. Computes a **recency-weighted model grade** (last 3 seasons; weights `10% / 25% / 65%` or `28% / 72%` for two seasons).
3. Applies **snap-volume stress** via `shrink_model_grade_for_season_snap_volume` — if the latest season had fewer snaps than the full-season reference, the model grade is pulled toward an anchor (60 for QB, configurable per position). This prevents a 4-game injury season from driving an elite grade.
4. Applies **inactivity penalty** (`inactivity_retirement_penalty`) — players who haven't played in 3+ years take up to −22 pts.
5. Computes a **stats grade** (position-specific weighted formula) and blends it with the model grade for a **composite grade**.
6. For QB: also infers the **signing role** (starter / fringe / backup) based on the target team's incumbent QB and positional need, then scales dropback projections accordingly.

### Node 2 — `evaluate_value`

1. Runs `compute_contract_value` — projects grade year-by-year using age curves + player YoY trend; applies `fair_market_aav_millions` per year; discounts at **8%/yr** and inflates cap at **6.5%/yr** to get **effective cap burden**.
2. Front-weights the fair AAV using `1/year` weights so year-1 performance dominates the headline number.
3. Applies `snap_value_reliability_factor` — small recent sample sizes discount the fair value so tiny-sample elite grades don't appear as elite contracts.
4. Runs `project_stats` — year-by-year counting and efficiency stat projections for the contract duration.

### Node 3 — `assess_team_fit`

Re-computes the cap-percentage trajectory of the contract using `aav_to_cap_pcts` with the actual league cap for each year. Stored in `signing_cap_pcts` for the decision node.

### Node 4 — `make_decision`

1. Computes `surplus_pct = (fair_aav - cap_burden) / fair_aav * 100`.
2. Maps surplus % to a **base tier**:
   - ≥ +20% → Exceptional Value
   - ≥ +5% → Good Signing
   - ≥ −5% → Fair Deal
   - ≥ −15% → Slight Overpay
   - ≥ −30% → Overpay
   - < −30% → Poor Signing
3. When team context is present, applies **replacement overlap** — a fraction of the incumbent's market value is subtracted from the fair AAV so signing a QB to an already-QB-stocked team is appropriately penalized. Overlap weight by position: QB 88%, HB 82%, ED/DI/LB/CB ~38–40%, OL 26%.
4. Applies `assess_team_fit` — crosses `(need_label, base_tier)` through a 3×6 matrix (3 need labels × 6 tiers) to produce a team-adjusted tier with a narrative note. `Exceeds Cap` overrides all other tiers when year-1 cap % > available cap %.

---

## 5. Salary curves — concrete numbers

Calibrated to **2026 dollars** (`VALUE_ANCHOR_CALIBRATION_YEAR = 2026`). Piecewise nonlinear between grade anchors; segments use per-position convexity exponents (> 1 means elite-end premium rises steeply).

**Grade anchors:** `[45, 55, 60, 65, 70, 75, 80, 85, 88, 92, 96, 100]`

**Fair AAV ($M) at each anchor — 2026 dollars:**

| Grade → | 45 | 55 | 60 | 65 | 70 | 75 | 80 | 85 | 88 | 92 | 96 | 100 |
|---------|-----|-----|------|------|------|------|------|------|------|------|------|------|
| **QB** | 0.9 | 3.6 | 20.9 | 25.5 | 30.9 | 38.2 | 50.1 | 52.8 | 54.6 | 56.4 | 58.2 | 60.1 |
| **T** | 1.2 | 3.0 | 7.3 | 13.3 | 18.1 | 23.0 | 26.6 | 30.2 | 32.6 | 33.8 | 35.1 | 36.3 |
| **WR** | 1.5 | 3.7 | 7.5 | 13.0 | 20.0 | 31.0 | 39.5 | 42.5 | 44.0 | 45.0 | 46.0 | 46.5 |
| **ED** | 1.3 | 3.3 | 6.6 | 10.5 | 17.1 | 23.7 | 30.3 | 35.5 | 40.8 | 47.4 | 52.6 | 56.6 |
| **CB** | 1.2 | 2.5 | 6.2 | 9.9 | 13.7 | 18.6 | 23.6 | 28.5 | 31.0 | 33.5 | 36.0 | 38.5 |
| **HB** | 1.2 | 2.4 | 5.8 | 7.8 | 9.7 | 11.7 | 15.5 | 18.1 | 19.4 | 22.0 | 24.6 | 27.2 |
| **G** | 0.7 | 1.5 | 3.2 | 5.4 | 8.4 | 11.3 | 14.3 | 17.7 | 20.2 | 23.2 | 26.6 | 29.6 |

**Year scaling** (actual NFL caps used for 2010–2026; projected at 6.5%/yr beyond):

| Year | Cap ($M) | QB scale vs 2026 |
|------|----------|-----------------|
| 2010 | 102.0 | 0.34× |
| 2015 | 143.3 | 0.48× |
| 2020 | 198.2 | 0.66× |
| 2023 | 224.8 | 0.75× |
| 2024 | 255.4 | 0.85× |
| 2025 | 279.2 | 0.93× |
| 2026 | 301.2 | 1.00× |

QB exponent = 1.04, so the scaling is `(cap_year/301.2)^1.04`. A grade-80 QB worth $50.1M in 2026 is worth ~$41.8M in 2024 and ~$27.4M in 2015.

**Discount / inflation rates used in contract valuation:**
- Discount rate: **8%/yr** (present-value discounting on future contract years)
- Cap growth rate: **6.5%/yr** (used to make a fixed AAV cheaper in real terms over time)
- Effective cap burden = nominal AAV deflated at 6.5%/yr, so a $40M/yr deal costs effectively ~$37.4M/yr in year-2 real terms.

---

## 6. Grade thresholds & stat anchors

### Performance tiers (all positions)

| Grade | Tier |
|-------|------|
| ≥ 80 | Elite |
| ≥ 74 | Good |
| ≥ 62 | Starter |
| < 62 | Rotation/backup |

### QB stats grade anchors (empirical league-wide, 17-game basis)

| Stat | Reserve | Rotation | Starter | Elite |
|------|---------|----------|---------|-------|
| Passer rating | 72 | 82 | 92 | 105 |
| YPA | 5.8 | 6.8 | 7.5 | 8.5 |
| Completion % | 60% | 64% | 67% | 70% |
| EPA/dropback | −0.10 | 0.00 | +0.10 | +0.20 |

**QB stats grade weights:** Passer rating 35%, YPA 30%, EPA/dropback 25%, completion % 10%.

**QB composite grade blend:** Model PFF grade 40%, stats grade 60%.

### OL stats grade anchors

| Stat | Poor | Average | Good | Elite |
|------|------|---------|------|-------|
| PBE | < 91 | 94–96 | 96–98 | > 99 |
| Sack rate/pb-snap | 4% | 1.5% | 0.8% | ~0% |

**T composite grade blend:** Model grade 58%, stats grade 42%. Tackles weight pass-block 65% vs run-block 35%.

### QB dropback projections (starter workload)

| Role | 17-game dropback range |
|------|----------------------|
| Starter | 560–720 |
| Fringe starter | 300–580 |
| Backup | 85–280 |

---

## 7. Team context & cap math

### Positional need score (0–100)

Five components, weights differ by position type:

| Weight | QB/HB/TE ("starter-dominant") | WR/OL/DB/ED/DI |
|--------|-------------------------------|----------------|
| Star power | 25% | 20% |
| Starter strength | 32% | 28% |
| Team production | 28% | 24% |
| Depth | 8% | 23% |
| Age risk | 7% | 5% |

- **Star power** = best player's composite percentile (PFF grade 40% + per-snap production rate percentiles 60%).
- **Starter strength** = for QB/HB/TE, snap-weighted composite; for other positions, mean of top-2 composites.
- All scores are league-relative (percentile ranked vs all 32 teams that year).
- For HB/WR/TE, scheme/personnel data (11/12/13 rates) further nudges these weights.

**Need labels:**

| Score | Label |
|-------|-------|
| ≥ 75 | Well-Stocked |
| 40–74 | Average |
| < 40 | Weak |

### Team-adjusted tier matrix (3 × 6)

| Need ↓ / Value → | Exceptional Value | Good Signing | Fair Deal | Slight Overpay | Overpay | Poor Signing |
|------------------|-------------------|-------------|-----------|---------------|---------|-------------|
| **Weak** | Must Sign — Elite Value + Need | Priority Target | Fill the Gap | Justifiable Overpay | Overpay — But Consider | Desperation Overpay |
| **Average** | Exceptional Value | Good Signing | Fair Deal | Slight Overpay | Overpay | Poor Signing |
| **Well-Stocked** | Luxury Add — Great Value | Luxury Add | Unnecessary Spend | Wasteful Overpay | Poor Signing | Cap Mismanagement |

Hard override: if year-1 cap % > available cap % → **Exceeds Cap** (regardless of value tier).

Additional downgrade when cap burden ratio ≥ 50% of remaining room (e.g. "Fair Deal" → "Slight Overpay").

---

## 8. Free Agency class builder

### Signing class grade

Weighted mean of per-signing grades, where each weight = `FA_CLASS_POS_IMPORTANCE[position] × profile_weight(AAV)`.

**Position importance weights:**

| QB | T | C | G | ED | WR | CB | DI | LB | TE | HB | S |
|----|---|---|---|----|----|----|----|----|----|----|----|
| 1.45 | 1.28 | 1.24 | 1.22 | 1.25 | 1.18 | 1.15 | 1.08 | 1.00 | 0.95 | 0.90 | 0.98 |

### Departure-need bump on signing grades

When "Account for Departures" is on, each signing at a position that also had a departure gets an additive boost:

```
stress = 0.55 × min(1, total_departure_weight / 5.5) + 0.45 × min(1, n_departures / 7)
directShare = departure_weight_at_same_pos / total_departure_weight
samePosPts = (2.2 + 7.5 × directShare) × stress
churnPts = 1.1 × stress
boost = min(10, samePosPts + churnPts)
```

E.g. one elite QB departure (capW ≈ 1.4) with one QB signing: directShare = 1.0, stress ≈ 0.22 → boost ≈ **2.4 pts**. Multiple departures stack stress higher.

### Roster net score

Starts from the signing class grade, then applies:
1. **Talent-out penalty** — proportional to average departure grade × number of departures.
2. **Coverage gap penalty** — penalizes departure positions not replaced by a same-position signing (up to −26 pts for fully uncovered high-importance positions).
3. **Stress multiplier** — light ×(1 + 0.05 × stress) amplifier on the signing baseline.

---

## 9. Scheme & personnel adjustments

**File:** `backend/agent/scheme_personnel.py`

Data source: `backend/ML/scheme/data/{year}_schemes_with_personnel.csv` — one row per team per season with personnel rates (11/12/13/20/21/22 packages, shotgun rate).

**Currently active for HB, WR, TE only** (`SCHEME_PERSONNEL_POSITION_KEYS`).

| Position | Adjustment trigger | Effect |
|----------|-------------------|--------|
| TE | High 12/13 usage (> 22%) | Increases starter-strength and star-power weights; reduces depth weight |
| HB | Heavy run (20/21/22 + partial 12/13 > 22%) | Increases depth weight; reduces starter-strength weight |
| WR | High 11 usage (> 56%) | Increases depth weight (more WR snaps available); reduces starter-strength weight |

QB and T are not yet scheme-adjusted (tracked in backlog).

---

## 10. Position GM APIs — port map

| Port | Position | Module | CSV |
|------|----------|--------|-----|
| 8002 | ED | `ed_main_api` | `ML/ED.csv` |
| 8003 | DI | `di_main_api` | `ML/DI.csv` |
| 8004 | QB | `qb_main_api` | `ML/QB.csv` (~1,435 rows, 2010–2024) |
| 8005 | HB | `hb_main_api` | `ML/HB.csv` |
| 8006 | WR | `wr_main_api` | `ML/WR.csv` |
| 8007 | TE | `te_main_api` | `ML/TightEnds/TE.csv` |
| 8008 | T | `t_main_api` | `ML/T.csv` (~2,220 rows) |
| 8009 | G | `g_main_api` | `ML/G.csv` |
| 8010 | C | `c_main_api` | `ML/C.csv` |
| 8011 | LB | `lb_main_api` | `ML/LB.csv` |
| 8012 | CB | `cb_main_api` | `ML/CB.csv` |
| 8013 | S | `s_main_api` | `ML/S.csv` |

**API routes (common across positions):**
- `GET /{pos}-players` — player list + year range
- `GET /teams?analysis_year=N` — 32 teams
- `GET /team-roster?team=X&analysis_year=N` — roster with grades, snaps, cap %, need score
- `GET /team-summary?team=X&analysis_year=N` — full positional summary
- `POST /evaluate` — main evaluation endpoint

**`POST /evaluate` body:**
```json
{
  "player_name": "Patrick Mahomes",
  "salary_ask": 52.0,
  "contract_years": 3,
  "team": "Chiefs",
  "analysis_year": 2025,
  "is_extension": false,
  "years_remaining": 0,
  "current_aav": 0.0
}
```

---

## 11. Data sources

| File | Contents |
|------|----------|
| `backend/ML/QB.csv` | PFF QB grades + stats, 2010–2024, ~1,435 rows |
| `backend/ML/T.csv` | PFF OT grades + blocking metrics, ~2,220 rows |
| `backend/ML/{pos}.csv` | Equivalent files for all 12 positions |
| `backend/ML/cap_data.csv` | Team cap allocation % by year, used for positional need league percentiles |
| `backend/ML/scheme/data/{year}_schemes_with_personnel.csv` | Team offensive personnel rates per season |
| `backend/ML/data/nflpowerrankings.csv` | Team win totals for team summary displays |
| `rosters/` | Roster snapshots (used for team context queries) |

---

## 12. Repository map

```
NFLSalaryCap/
├── frontend/
│   ├── src/
│   │   ├── pages/FreeAgency.jsx         Main FA UI (chat, class builder, departures)
│   │   ├── pages/PlayerPage.jsx         Individual player detail
│   │   ├── pages/PlayerRanking.jsx      League-wide position rankings
│   │   ├── config/freeAgencyPositionConfig.js  Ports, stat columns, value anchors
│   └── package.json
├── backend/
│   ├── agent/
│   │   ├── agent_graph.py               QB LangGraph + extract/project functions
│   │   ├── ol_agent_graph.py            OL (T/G/C) shared LangGraph
│   │   ├── *_agent_graph.py             HB, WR, TE, ED, DI, LB, CB, S graphs
│   │   ├── *_main_api.py                FastAPI apps (one per position)
│   │   ├── market_value_curves.py       Piecewise salary curve + cap scaling
│   │   ├── team_context.py              Roster, need, cap, assess_team_fit
│   │   ├── stat_projection_utils.py     QB/defense snap/projection helpers
│   │   ├── grade_projection.py          Tier labels, age curves, YoY trend
│   │   ├── scheme_personnel.py          Personnel-based need weight adjustments
│   │   ├── team_summary.py              Team summaries, power rankings
│   │   ├── api_year_utils.py            Year clamping, history filtering
│   │   └── run_all_free_agency_apis.sh  Start all 12 Uvicorn processes
│   ├── ML/
│   │   ├── *.csv                        Per-position PFF data
│   │   ├── cap_data.csv
│   │   └── scheme/data/
│   └── hermesServer.py                  Legacy Flask/MongoDB player lookup
├── README.md                            This file
├── FREE_AGENCY_TOOL_REFERENCE.md        Detailed operational spec (last updated 2026-04-09)
└── requirements.txt
```

---

## 13. Change log — last ~70 commits

The `FREE_AGENCY_TOOL_REFERENCE.md` covers work up through **2026-04-09**. The sections below capture what has changed since then.

### Since FREE_AGENCY_TOOL_REFERENCE.md (Apr 10 – Apr 28, 2026)

#### Apr 10 — Contract extensions, breakout seasons, cap scaling

- **`2c45ab2` Contract extension mode** — New `is_extension` + `years_remaining` fields on the evaluate request. When set, fair AAV is computed starting at `analysis_year + years_remaining` so the valuation reflects what the contract will cost when it kicks in, not today's cap.
- **`6445f36` Snap-volume injury credit** — `shrink_model_grade_for_season_snap_volume` now checks whether any **prior** season had a full snap load + good grade. If so, the snap-volume stress factor is partially restored, preventing a one-year injury from permanently collapsing a player's grade.
- **`3d38c81` Breakout season detection** — When a player's most recent season is significantly above career baseline, the recency weighting now captures that breakout more aggressively rather than blending it down toward older mediocre seasons.
- **`6a096d0` Year-based value scaling** — All 12 positions now use `fair_market_aav_millions(grade, pos, analysis_year)` instead of a fixed-dollar lookup. Contracts evaluated in 2015 use 2015-dollar fair values; 2032 uses projected cap. The calibration year is `VALUE_ANCHOR_CALIBRATION_YEAR = 2026`.
- **`0b71f2a` Age bug fix** — Player age was computed incorrectly for multi-year contracts; now `current_age + (yr - 1)` per contract year. Acknowledged that pre-2025 salary ranges still need manual curve audit.

#### Apr 11 — Piecewise salary curves, OL differentiation, PFF/stats tuning

- **`8147777` Piecewise salary curves** — Replaced linear interpolation with nonlinear piecewise functions. Each grade segment has its own convexity exponent (`p > 1` at the top end means elite players earn disproportionately more). E.g. the QB grade-85→88 segment uses `p = 1.15`; grade-96→100 uses `p = 1.55`.
- **`f5841c2` Value anchor calibration** — Re-anchored all 12 positions' `VALUE_BY_POSITION` tables to match 2026 OTC contracts. Major changes: QB grade-80 moved to $50.1M; WR grade-75 to $31M (reflecting the CeeDee Lamb / Justin Jefferson era); ED grade-92 to $47.4M (Micah Parsons tier).
- **`d632f73` OL sub-position differentiation** — `_pass_block_grade` now uses different stat weights for T vs G vs C. Tackles: sacks weight 20%, PBE/PBG 25%/25%; Guards: sack weight 10%, PBE/PBG 30%/30% (sacks rarely attributed to guards); Centers: sack weight 0%, PBE/PBG 35%/35%.
- **`9069587` PFF/stats grade balance** — Adjusted the model-to-stats composite ratio. QB: 40% model / 60% stats. OL: 58% model / 42% stats. This shifts OL evaluation slightly more toward PFF grading vs raw blocking metrics.
- **`7614f60` Curve re-tuning** — Further adjustment of mid-grade value anchors after testing; commit notes "need more testing" still applies.
- **`cd1bcd5` Missing data handling** — Fixed a bug where players with missing `player_game_count` in certain years caused the evaluation to return a 404 instead of inferring games from dropbacks.
- **`11be008` Projected tier + stat cards** — Frontend now shows both **current tier** (based on last season) and **projected tier** (year-1 of contract). Added projected stats cards for HB, WR, and TE to match QB's existing stats projection display.
- **`5018932` 10-year contract cap** — Contract year input capped at 10 (from 7) to support very long extensions.

#### Apr 12 — Polish and rankings update

- **`8d854f9` Descriptive evaluator reasoning** — Each position's `make_decision` now outputs a more position-specific narrative. E.g. QB reasoning explicitly names the pass-grade, stats-grade, composite, health factor, surplus, and cap note in one cohesive paragraph rather than a generic template.
- **`7f6616d` Frontend fixes** — Minor layout tweaks to stat grid and tier badge positioning.
- **`a6fccef` Power rankings data update** — Updated `nflpowerrankings.csv` with 2025 season results.

#### Apr 20 — Running Back evaluator

- **`c4499d4` HB (Running Back) LangGraph evaluator** — Completed the full RB agent graph with position-specific stats (elusive rating, YCO, breakaway %, avoided tackles), health and inactivity model, and dedicated frontend page. This completes the offensive skill position set (QB, HB, WR, TE all have full LangGraph pipelines).
- **`2d34919` / `020beca`** — Merged siddarth-new and Ashwins-agentic-branch into main.

### Work captured by FREE_AGENCY_TOOL_REFERENCE.md (up to Apr 9)

Key milestones from the reference doc and earlier git history:

| Date | Milestone |
|------|-----------|
| Apr 2 | Initial LangGraph QB evaluator and chatbot |
| Apr 7 | DI and additional position agents; per-year cap growth |
| Apr 8 | Team context for all defensive positions; re-signing logic; PFF grade weighting |
| Apr 9 | FA class builder; departure modeling + roster net grade; positional need (league-relative); scheme/personnel hooks; historical year support (2010+); departure-need bump on signing grades |

---

## 14. Known issues & backlog

| Issue | Location | Notes |
|-------|----------|-------|
| **QB projected yards too high** | `agent_graph.py → project_stats` | YPA and attempt volume both scale with the grade-progress `scale` factor, compounding to unrealistic yardage totals. Fix: decouple YPA scaling from attempt volume. |
| **Pre-2025 salary curves off for some positions** | `market_value_curves.py → VALUE_BY_POSITION` | Cap exponents and value anchors are calibrated to 2026. Some mid-range grades produce too-high values in 2010–2020. Needs per-position audit vs OTC historical contracts. |
| **FA departure impact too weak for single elite-position departure** | `FreeAgency.jsx → departureImportanceBoostForSigning` | Signing a QB to replace a departed starting QB only adds ~2.4 pts of boost because `stress` normalizes by 5.5 (weight) and 7 (count). Should be 6–10 pts for an elite single departure. |
| **Scheme data untested** | `scheme_personnel.py` | Personnel adjustments for HB/WR/TE exist but have not been systematically validated. |
| **QB/T not scheme-aware** | `scheme_personnel.py` | QB and T are not in `SCHEME_PERSONNEL_POSITION_KEYS`. Shotgun rate / pass-heavy schemes should boost QB dropback projections; pass-heavy schemes should elevate T value. |
| **Scheme benefit not in stat projections** | `agent_graph.py` | Even where scheme data exists, it only adjusts positional need weights, not the player's projected stats or composite grade. Need to thread scheme cluster data into the model. |
| **Missing QB data early 2000s–2010s** | `ML/QB.csv` | Some starting QBs from 2010–2014 have missing `grades_pass` or `player_game_count`. Causes inaccurate career histories; some players may need manual data fills or a broader fallback. |
| **Cap data slightly off for some years** | `team_context.py` | Noted in commit `50496bb` — total cap hit numbers may not perfectly match published OTC snapshots for older years. |
