# MirrorLife Well-being Classification System

A multi-agent pipeline for binary preventive health classification. The system analyzes citizen health data (events, GPS, demographics) to identify individuals who need preventive support.

**Challenge**: Reply Challenge — Sandbox_2026_V3
**Task**: Binary classification per citizen → `0` (standard monitoring) or `1` (preventive support needed)
**Output**: List of CitizenIDs (one per line) flagged as needing preventive support

---

## Project Structure

```
MultiAgent-Demo/
├── .env                          # API keys and runtime config (not committed)
├── README.md                     # This file
├── Sandbox_2026_V3.pdf           # Challenge specification
│
├── data/                         # All datasets
│   ├── training/                 # Training datasets (one subfolder per level)
│   │   └── public_lev_1/         # Level 1 — Training dataset
│   │       ├── status.csv        # Health events (PAI, SQI, EEL per citizen)
│   │       ├── users.json        # Citizen profiles (demographics, home GPS)
│   │       ├── locations.json    # GPS location history
│   │       └── personas.md       # Free-text persona descriptions
│   └── evaluation/               # Evaluation datasets (one subfolder per level)
│       └── public_lev_1_ev/      # Level 1 — Evaluation dataset (same structure)
│
├── output/                       # Generated prediction files
│   └── predictions_<folder>.txt  # One file per evaluated dataset
│
├── solution/                     # Main application
│   ├── main.py                   # Pipeline entry point & orchestrator
│   ├── config.py                 # Paths, API keys, model settings
│   ├── requirements.txt          # Python dependencies
│   ├── agents/
│   │   ├── data_loader.py        # Agent 1 — Load all data sources
│   │   ├── feature_engineer.py   # Agent 2 — Compute quantitative risk features
│   │   ├── mobility_analyzer.py  # Agent 3 — GPS-based mobility analysis
│   │   ├── wellbeing_analyzer.py # Agent 4 — LLM classification (1 batched call)
│   │   └── output_writer.py      # Agent 5 — Write prediction file
│   └── utils/
│       └── langfuse_tracker.py   # Langfuse v3 observability integration

```

---

## Setup

### 1. Install dependencies

```bash
cd solution
pip install -r requirements.txt
```

### 2. Configure environment

Copy and fill in `.env` at the project root:

```env
OPENROUTER_API_KEY=sk-or-v1-...
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://challenges.reply.com/langfuse
TEAM_NAME=project-demo
DATA_FOLDER=public_lev_1_ev      # Folder to evaluate (searched in data/training/ then data/evaluation/)
```

### 3. Run

```bash
cd solution
python3 main.py
```

The output file is written to `output/predictions_<DATA_FOLDER>.txt`.

---

## Switching Datasets

Set `DATA_FOLDER` in `.env` to the name of the folder you want to evaluate:

```env
DATA_FOLDER=public_lev_1        # Self-check on training data
DATA_FOLDER=public_lev_1_ev     # Level 1 evaluation
DATA_FOLDER=public_lev_2_ev     # Level 2 evaluation (when available)
```

The system automatically looks for the folder in `data/training/` first, then `data/evaluation/`. Place new datasets in the appropriate subdirectory and update `DATA_FOLDER`.

---

## Pipeline Architecture

The system runs **5 agents in sequence** with a single LLM call in step 4.

```
┌─────────────────────────────────────────────────────────────┐
│                     run_pipeline()                          │
│                                                             │
│  ┌──────────────┐    ┌─────────────────┐                   │
│  │ DataLoader   │    │ FeatureEngineer │                   │
│  │  Agent [1]   │───▶│   Agent [2]     │                   │
│  │              │    │                 │                   │
│  │ Load:        │    │ Compute:        │                   │
│  │ - status.csv │    │ - PAI/SQI/EEL   │                   │
│  │ - users.json │    │   mean/std/slope│                   │
│  │ - locations  │    │ - Escalated     │                   │
│  │ - personas   │    │   event detect  │                   │
│  └──────────────┘    │ - Rule score    │                   │
│         │            │   [0–100]       │                   │
│         │            └────────┬────────┘                   │
│         │                     │                            │
│         ▼                     ▼                            │
│  ┌──────────────┐    ┌─────────────────┐                   │
│  │  Mobility    │    │   Wellbeing     │                   │
│  │  Analyzer    │───▶│   Analyzer [4]  │                   │
│  │  Agent [3]   │    │                 │                   │
│  │              │    │ Single batched  │                   │
│  │ Compute:     │    │ LLM call        │                   │
│  │ - Radius of  │    │ (train context  │                   │
│  │   gyration   │    │  + eval items)  │                   │
│  │ - Dist from  │    │                 │                   │
│  │   home       │    │ → {cid: 0|1}    │                   │
│  │ - Mobility   │    └────────┬────────┘                   │
│  │   level      │             │                            │
│  └──────────────┘             │                            │
│                               ▼                            │
│                     ┌─────────────────┐                   │
│                     │  Consensus      │                   │
│                     │  Decision Logic │                   │
│                     │                 │                   │
│                     │ rule≥70 → 1     │                   │
│                     │ rule≤10 → 0     │                   │
│                     │ else → LLM vote │                   │
│                     └────────┬────────┘                   │
│                               │                            │
│                               ▼                            │
│                     ┌─────────────────┐                   │
│                     │  OutputWriter   │                   │
│                     │  Agent [5]      │                   │
│                     │                 │                   │
│                     │ Write:          │                   │
│                     │ predictions_    │                   │
│                     │ <folder>.txt    │                   │
│                     └─────────────────┘                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Agent Details

### Agent 1 — DataLoaderAgent

Loads and normalizes all data sources. No LLM calls.

| Output | Description |
|--------|-------------|
| `StatusEvent` | EventID, CitizenID, EventType, PAI, SQI, EEL, Timestamp |
| `LocationRecord` | user_id, lat, lng, city, timestamp |
| `UserProfile` | name, birth_year, job, home coordinates |
| `DataBundle` | Container for all of the above + raw personas text |

---

### Agent 2 — FeatureEngineerAgent

Computes quantitative risk features per citizen via statistical analysis. No LLM calls.

**Risk Score Breakdown** (0–100):

| Signal | Max Points | Description |
|--------|-----------|-------------|
| Escalated events | 40 | specialist consultation, follow-up assessment, emergency visit, urgent care, hospitalization |
| High variability | 20 | PAI std × 0.8 + SQI std × 0.8 |
| Negative slopes | 20 | Declining PAI or SQI over time |
| High EEL mean | 10 | Environmental exposure > 55 |
| Rising EEL | 10 | Positive EEL slope |
| Low recent metrics | 10 | PAI < 35 or SQI < 40 in last 3 events |

**Key at-risk signals** (strongest → weakest):
1. Escalated event types (specialist consultations, emergency visits)
2. Declining PAI slope (e.g. −4.71 per event)
3. Rising EEL (Environmental Exposure Level > 55, positive slope)
4. High variability in PAI or SQI (std > 10)

---

### Agent 3 — MobilityAnalysisAgent

Extracts mobility patterns from GPS history. No LLM calls.

| Feature | Description |
|---------|-------------|
| `radius_of_gyration_km` | Spread of all locations from center of mass |
| `mean_distance_from_home_km` | Average distance from residential address |
| `mobility_level` | `very_low` / `low` / `moderate` / `high` / `very_high` |

Mobility thresholds (radius of gyration):
- < 200 km → `very_low`
- 200–1000 km → `low`
- 1000–2000 km → `moderate`
- 2000–3000 km → `high`
- \> 3000 km → `very_high`

---

### Agent 4 — WellbeingAnalysisAgent

The only agent that can call an LLM. Designed to minimize token usage while preserving accuracy.

- **Model**: `stepfun/step-3.5-flash:free` via OpenRouter (reasoning model)
- **Temperature**: 0.1 (near-deterministic output)
- **Max tokens**: 4000 (reasoning models consume tokens for internal chain-of-thought before producing output)
- **Fallback**: Rule-based predictions used if LLM response is empty or unparseable

---

### When and How the LLM Is Called

The LLM is **only called when the rule-based score cannot confidently decide** the outcome. Every citizen is first scored by the `FeatureEngineerAgent` on a 0–100 scale using pure Python statistics (no LLM cost). The result falls into one of three zones:

```
Rule score:   0 ────────── 20 ─────────────── 65 ──────────── 100
                   SAFE zone    AMBIGUOUS zone     RISK zone
              label=0, no LLM   ← LLM called →   label=1, no LLM
```

| Zone | Condition | Action | Tokens used |
|------|-----------|--------|-------------|
| Confident safe | `rule_score ≤ 20` | Label = 0 immediately | **0** |
| Ambiguous | `20 < rule_score < 65` | Single batched LLM call for all ambiguous citizens | Input + output tokens |
| Confident risk | `rule_score ≥ 65` | Label = 1 immediately | **0** |

**In practice, both Level 1 and Level 2 had all citizens outside the ambiguous zone → 0 LLM tokens used.**

#### What the LLM receives (when called)

Only the ambiguous citizens are included — no training examples, no mobility data. The prompt format is intentionally minimal:

```
[CITIZENID] esc=N(event_types) pai_slope=X sqi_slope=Y eel=Z(slope=W) score=S | health=... | social=...
Classify [CITIZENID, ...] → JSON only
```

- `esc` = number and types of escalated care events (strongest risk signal)
- `pai_slope` / `sqi_slope` = trend direction of physical activity and sleep quality
- `eel` + `slope` = current environmental exposure and its trajectory
- `health` / `social` = two-line persona summary extracted from the persona file
- The full training dataset is **not included** — the system prompt encodes the classification rule directly

#### LLM call decision tree

```
For each evaluation citizen:
  │
  ├─ rule_score ≥ 65? ──→ label=1 (skip LLM)
  │
  ├─ rule_score ≤ 20? ──→ label=0 (skip LLM)
  │
  └─ ambiguous (20–65)?
        │
        └─ collect all ambiguous citizens
              │
              ├─ none? ──→ return all rule-based results (0 tokens)
              │
              └─ some? ──→ one batched LLM call with minimal prompt
                                │
                                ├─ parse JSON response → {cid: 0|1}
                                └─ fallback to rule_prediction if parse fails
```

---

### Consensus Decision Logic

```python
if rule_score >= 70:
    prediction = 1   # Very high rule score → definite risk
elif rule_score <= 10:
    prediction = 0   # Very low rule score → definite safe
else:
    prediction = llm_prediction  # LLM gets the vote (nuanced cases)
```

---

### Further Optimizations: How to Reduce Tokens, Latency, and Cost

#### Input tokens

| Technique | Saving | Notes |
|-----------|--------|-------|
| **Current: skip LLM when no ambiguous citizens** | 100% | Already implemented. Zero tokens when all rule scores are extreme. |
| Widen the confidence bands | Reduces ambiguous pool | Raise `CONFIDENT_SAFE_THRESHOLD` (e.g. 25→30) or lower `CONFIDENT_RISK_THRESHOLD` (e.g. 65→55) if the rule score proves reliable across more levels. |
| Drop persona text from prompt | ~30–40% of prompt tokens | If the feature signals (escalated events, slopes) are already sufficient, the `health=` / `social=` lines add cost without changing the result. |
| Abbreviate feature keys | ~10–15% | Replace `pai_slope=` with `ps=`, `eel=` with `e=`, etc. Minimal readability cost. |
| Remove training examples | Already done | The current prompt does not include training citizens; the rule is encoded in the system prompt instead. |

#### Output tokens

| Technique | Saving | Notes |
|-----------|--------|-------|
| **Constrain response format** | Large | The current system prompt asks for compact JSON `{"ID":0,...}`. Tighter instruction (e.g. "respond with space-separated 0/1 values in the same order") removes JSON key overhead. |
| Reduce max_tokens | Moderate | `MAX_TOKENS=4000` is a ceiling, not a guarantee. For the reasoning model, the actual output is small (~30 tokens) — the rest is internal reasoning. Cannot be reduced without risking truncation. |
| Switch to a non-reasoning model | Large | A standard instruction-tuned model (e.g. `mistralai/mistral-7b-instruct:free`) does not use reasoning tokens, so `max_tokens=200` is sufficient. Saves ~400–500 tokens per call. Trade-off: may be less accurate on edge cases. |

#### Latency

| Technique | Impact | Notes |
|-----------|--------|-------|
| **Current: skip LLM when no ambiguous cases** | Eliminates LLM latency entirely (~14s → ~0.01s) | Already implemented. |
| Agents 1–3 run sequentially | Minor | DataLoader, FeatureEngineer, and MobilityAnalyzer are all fast pure-Python steps. Could be parallelized for very large datasets (e.g., run mobility analysis concurrently with feature engineering). |
| Cache feature results | Useful for repeated runs | Pickle `CitizenFeatures` per dataset after first computation; skip re-computation on subsequent runs with the same data. |
| Use a faster model | Large | Models with lower TTFT (time to first token) on OpenRouter reduce wall-clock time when LLM is needed. `stepfun/step-3.5-flash` is already optimized; alternatives: `google/gemma-3-4b-it:free` or `meta-llama/llama-3.2-3b-instruct:free`. |

---

### Agent 5 — OutputWriterAgent

Writes the final submission file.

- Filters citizens with `label = 1`
- Sorts alphabetically (deterministic)
- Writes one CitizenID per line to `output/predictions_<EVAL_FOLDER>.txt`

---

## Observability (Langfuse)

Every run is tracked in Langfuse with:

- **Session ID**: `<TEAM_NAME>-<EVAL_FOLDER>-<run_id>` (e.g. `project-demo-public_lev_1_ev-ad28bfcf`)
- **Spans**: One per agent (hierarchical trace)
- **LLM generation**: Prompt, response, token usage recorded
- **Post-run summary**: Token counts, costs, and LLM latency printed to console

Dashboard: configured via `LANGFUSE_HOST` in `.env`

---

## Model & API Configuration

| Setting | Value |
|---------|-------|
| Model | `stepfun/step-3.5-flash:free` |
| Provider | OpenRouter (`https://openrouter.ai/api/v1`) |
| Temperature | `0.1` |
| Max tokens | `4000` |

To switch models, update `MODEL_ID` in `solution/config.py`.

---

## Known Behaviors

- **"Context error: No active span"** at startup — cosmetic Langfuse OTEL initialization warning, non-blocking
- **Empty cost data** in trace summary — expected for free-tier models on OpenRouter
- **Reasoning model requirement** — models like stepfun require `MAX_TOKENS >= 4000` or the response will be empty
