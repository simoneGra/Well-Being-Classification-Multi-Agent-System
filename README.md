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
TRAIN_FOLDER=public_lev_1        # Folder inside data/training/ to use as training data
EVAL_FOLDER=public_lev_1_ev      # Folder inside data/evaluation/ to evaluate
```

### 3. Run

```bash
cd solution
python3 main.py
```

The output file is written to `output/predictions_<EVAL_FOLDER>.txt`.

---

## Switching Datasets

To use different datasets, set `TRAIN_FOLDER` and/or `EVAL_FOLDER` in `.env`:

```env
TRAIN_FOLDER=public_lev_1       # Folder under data/training/
EVAL_FOLDER=public_lev_1_ev     # Folder under data/evaluation/

# Level 2 example (when available):
TRAIN_FOLDER=public_lev_2
EVAL_FOLDER=public_lev_2_ev
```

Place training datasets inside `data/training/` and evaluation datasets inside `data/evaluation/`, then point the env vars to the subfolder name.

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

Single batched LLM call classifying all evaluation citizens at once. Token-optimized.

- **Model**: `stepfun/step-3.5-flash:free` via OpenRouter (reasoning model)
- **Temperature**: 0.1 (deterministic output)
- **Max tokens**: 4000 (reasoning models need buffer for chain-of-thought)
- **Prompt**: Training citizens used as calibration examples; LLM returns JSON `{cid: 0|1}`
- **Fallback**: Rule-based predictions used if LLM response is empty or unparseable

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
