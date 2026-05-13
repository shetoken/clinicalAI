# hh-note-features

Gemini-powered structured feature extraction from Home Health visit notes — the
"note-derived signals" layer of the CenterWell Clinical AI feature store.

This repo handles the step that, in the use-case walkthroughs, runs after a
nurse hits Save in HCHB: it takes the free-text narrative note and emits a
typed, evidence-cited JSON object that feeds `visit_note_embeddings_gold`,
`hosp_risk_indicators_gold`, and the visit-note QA pipeline.

> Per the platform roadmap: Gemini 2.5 Flash-Lite / Flash for fast/cheap
> extraction; Gemini 2.5 Pro for harder notes or retries. Both via
> **Vertex AI under the Google Cloud BAA**. Direct Gemini API mode is for
> dev-only with synthetic data.

---

## What gets extracted

A `VisitNoteFeatures` Pydantic object with these blocks:

| Block | What lives here |
|---|---|
| `vitals` | BP, HR, RR, SpO2 (+ room-air flag), weight + delta, pain |
| `cardiopulmonary` | Dyspnea level (on_exertion / at_rest / …), JVD, edema, crackles, wheezes |
| `neurocognitive` | AMS, new-onset confusion, lethargy |
| `medications` | PRN diuretic use, rescue inhaler use, adherence concerns, med changes, high-risk meds |
| `function_safety` | Falls since last visit, ambulation status, ADL concerns, home hazards |
| `wound` | Location, NPUAP stage, size, tissue %, exudate, infection signs |
| `care_context` | Caregiver present + role, education topics, SDOH concerns |
| `clinical_impression` | Primary dx cluster, overall status, escalation indicators, rationale |
| `doc_completeness` | OASIS-E items addressed, potential documentation gaps |
| `flat_signals` | Flat (name, value, evidence_quote, confidence) records for the feature store |

**Every populated finding carries a verbatim evidence quote.** This is the
provenance contract — no provenance, no landing in the feature store.

---

## Quickstart

```bash
git clone <this repo>
cd hh-note-features

python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env       # then edit credentials
```

Run it:

```bash
# Single note
hh-extract data/sample_notes/note_01_hfref_decompensation.txt

# Batch a whole directory of notes
hh-extract data/sample_notes/ --out data/extractions/ --concurrency 4

# Override the model (Pro for hard cases)
hh-extract data/sample_notes/note_02_copd_routine.txt --model gemini-2.5-pro
```

Or from Python:

```python
from hh_note_features import extract_features

with open("data/sample_notes/note_01_hfref_decompensation.txt") as f:
    note = f.read()

result = extract_features(note, model="gemini-2.5-flash")
feats = result.features

print(feats.cardiopulmonary.dyspnea)            # DyspneaLevel.ON_EXERTION
print(feats.vitals.weight_change_lbs)            # 5.3
print(feats.clinical_impression.escalation_indicators)
# ['rapid weight gain 5.3 lbs / 7 days', 'new-onset confusion x 2 days', ...]
```

---

## Configuration — Vertex (prod) vs direct API (dev)

The client picks its mode from env vars (see `.env.example`):

**Vertex AI (production, PHI-safe under BAA):**
```bash
USE_VERTEX_AI=true
GOOGLE_CLOUD_PROJECT=centerwell-clinical-ai-dev
GOOGLE_CLOUD_LOCATION=us-central1
# auth: gcloud auth application-default login
#   or: GOOGLE_APPLICATION_CREDENTIALS=/path/to/sa-key.json
#   or: GKE / Cloud Run workload identity
```

**Direct Gemini API (dev only, synthetic data only):**
```bash
USE_VERTEX_AI=false
GEMINI_API_KEY=...
```

---

## Architecture

```
                       ┌──────────────────────────────┐
   visit note text ──▶ │  hh_note_features.extractor  │ ──▶ VisitNoteFeatures
                       │                              │     (Pydantic, validated)
                       │  • SYSTEM_PROMPT (rules)     │
                       │  • Gemini structured output  │
                       │    via response_schema       │
                       │  • temperature = 0.0         │
                       └──────────────────────────────┘
                                    │
                                    ▼
                       feature-store ingest job
                       (writes visit_note_embeddings_gold
                        + hosp_risk_indicators_gold rows
                        with point-in-time correctness)
```

Key design decisions:

1. **Pydantic schema is the contract.** Gemini's `response_schema` enforces it
   on the model side; `model_validate` enforces it on our side. Garbage in,
   `ValidationError` out.
2. **Conservative extraction.** Defaults are null / False / `not_documented`.
   The prompt explicitly forbids inference beyond what the note states.
3. **Evidence quotes are verbatim.** Each clinical finding carries a span from
   the note. This is what makes the output auditable.
4. **Temperature 0.** Extraction is not a creative task.
5. **Two modes, one client.** Same code path runs against Vertex (BAA) or
   the direct API (dev). The `get_client` factory hides the difference.

---

## Sample notes

Five synthetic notes in `data/sample_notes/`, each mirroring a use case from
the roadmap walkthroughs:

| File | Scenario | Tests |
|---|---|---|
| `note_01_hfref_decompensation.txt` | HFrEF early decompensation (the Robert T. walkthrough) | weight delta, JVD, dyspnea, new confusion, PRN diuretic, escalation |
| `note_02_copd_routine.txt` | Stable COPD week-4 visit (the James W. walkthrough) | doc-completeness gap detection (no MRC scale) |
| `note_03_wound_followup.txt` | Stage 3 sacral pressure injury follow-up (Margaret K. walkthrough) | wound block: stage, size, tissue %, exudate, infection signs |
| `note_04_postop_fall.txt` | Post-TKA fall + home hazards | falls + fall_risk_concerns + ADL concerns + home_safety_hazards |
| `note_05_diabetes_sdoh.txt` | Uncontrolled T2DM with food insecurity, isolation, insulin gap | SDOH concerns, adherence concern, escalation indicators |

> ⚠️ All notes are clearly synthetic and contain no real PHI. Do not ever
> commit real notes to this repository.

---

## Testing

```bash
pytest                  # schema-only smoke tests (no API calls)
```

For integration testing against a real Gemini endpoint, write tests that mark
themselves `@pytest.mark.integration` and skip them by default in CI.

---

## Productionisation checklist

Before this leaves dev:

- [ ] Wire to `visit_notes_silver` change-data-capture stream so extraction
      fires automatically on note save.
- [ ] Land outputs as one row per `(note_id, asof_ts)` into
      `visit_note_features_gold` (new table) and append flat_signals to
      `visit_note_signals_gold`.
- [ ] Add MLflow tracing on every call (model, latency, token usage,
      validation errors, retry count).
- [ ] Retry policy: on `ValidationError`, retry once with `gemini-2.5-pro`
      and a slightly higher max_output_tokens. On a second failure, log and
      skip — never fabricate.
- [ ] PHI scrub before any external egress (gliner_multi_pii) — Vertex BAA
      covers Google's side, but the agent still needs to mask outside the
      enclave.
- [ ] Subgroup fairness audit on `dyspnea` and `confusion_new_onset` extraction
      precision across age, sex, race quartiles before model-serving features
      consume the output.

---

## License

Internal CenterWell use. Not for redistribution.
