# Changelog

## June 2026 — Evaluation & explainability (Phase C)

Additive changes. Core predict → modal → Supabase flow unchanged.

### New: Explainable AI section (classification report)

- **Component:** `frontend/src/components/ExplainableAIPanel.tsx`
- **Where:** Classification report modal (`AnalysisResults.tsx`)
- **Includes:** Plain-English narrative, Play Sound, confidence calibration, waveform/Mel/Grad-CAM visuals, router block (auto mode), top-3 softmax distribution

### New: Explainable AI blurb (comparison report)

- **Component:** `frontend/src/components/ComparisonExplainabilityBlurb.tsx`
- **Helper:** `frontend/src/lib/comparisonSummary.ts` (shared with `ComparisonWinnerCard`)
- **Where:** Multi-model comparison modal (`ModelComparisonPanel`)
- **Includes:** Per-model prediction summary, agreement %, fastest/highest-confidence notes, suggested pick, Play Sound

### New: Play Sound on all reports

- **Component:** `frontend/src/components/PlaySoundButton.tsx`
- **Where:** Explainable AI sections (classification + comparison)
- **Behaviour:** Upload/mic uses blob replay; dataset/showcase uses `dataset_domain` + `sample_id` stream URL
- **API:** `PredictResponse.dataset_domain`; `ModelCompareResponse.sample_id`, `dataset_domain`, `input_source`

### New: History ground-truth audit filters

- **Component:** `PredictionHistoryPanel.tsx` (filters + Ground truth / Audit columns)
- **Backend:** `enrich_prediction_row()` in `predictions_repo.py`; saves `ground_truth_label`, `sample_id`, `dataset_domain` on dataset predictions
- **Helper:** `label_matching.py` / `lib/labelMatching.ts` (`dog` ↔ `dog_bark` equivalence)
- **Filters:** All · Dataset audits · Correct · Mismatches

### Migration: ground-truth audit columns

- **File:** `supabase/migrations/003_ground_truth_audit.sql`
- **Columns:** `sample_id`, `ground_truth_label`, `dataset_domain` on `predictions`
- **Note:** History audit works without migration via sample lookup fallback on fetch

### Files added (Phase C)

```
frontend/src/components/
  ExplainableAIPanel.tsx
  ComparisonExplainabilityBlurb.tsx
  PlaySoundButton.tsx
frontend/src/lib/
  comparisonSummary.ts
  labelMatching.ts
backend/app/services/
  label_matching.py
supabase/migrations/
  003_ground_truth_audit.sql
```

---

## June 2026 — Phase B (session tools & router lab)

Additive changes. Core predict → modal → Supabase flow unchanged.

### New: Session Timeline tab

- **Nav:** Sidebar → **Session Timeline**
- **Component:** `frontend/src/components/SessionTimelinePanel.tsx`
- **Behaviour:** Merges prediction history with analytics summary metrics in a chronological timeline
- **Export:** **Export session ZIP** downloads `GET /api/reports/session-export` (summary JSON, predictions JSON/CSV, analytics snapshot)

### New: Router Lab tab

- **Nav:** Sidebar → **Router Lab**
- **Component:** `frontend/src/components/RouterLabPanel.tsx`
- **Behaviour:** Shows Smart Auto-Router explanation for the last auto-routed clip; **what-if** reruns force urban-only or animal-only on the same audio/sample
- **Context:** Captured from Analyze Live, Datasets, or Showcase when `router` telemetry is present

### New: Session export API

- **Endpoint:** `GET /api/reports/session-export` (header `X-Session-Id`)
- **Service:** `build_session_report_zip()` in `backend/app/services/export_report.py`

### New: Presentation mode

- **Toggle:** Sidebar → System Status → **Presentation mode**
- **Storage:** `localStorage` key `sap-presentation-mode`
- **Effect:** Larger typography, hides help banner, calmer panels for live demos (`index.css` `.presentation-mode`)

---

## June 2026 — Phase A (UI enhancements)

Additive frontend changes only. No changes to core inference API contracts (`/api/predict`, `/api/predict/sample`, etc.).

### New: Showcase tab

- **Nav:** Sidebar → **Showcase**
- **Component:** `frontend/src/components/ShowcasePanel.tsx`
- **Behaviour:** Five one-click scenarios on curated test clips (urban siren, jackhammer, animal dog, auto-router on urban `dog_bark` and animal `dog`)
- **API:** Reuses `POST /api/predict/sample` with `model_name=mobilenetv2`
- **Result:** Opens the same classification report modal as Analyze Live / Project Datasets
- **Loader:** WaveLoader while a scenario runs; panel hidden during load

### New: Comparison winner summary

- **Component:** `frontend/src/components/ComparisonWinnerCard.tsx`
- **Where:** Top of the multi-model comparison report (`ModelComparisonPanel`)
- **Shows:** Fastest model, highest confidence, label agreement %, suggested model pick

### New: Confidence calibration panel

- **Component:** `frontend/src/components/ConfidenceCalibrationPanel.tsx`
- **Where:** Classification report modal (`AnalysisResults`), below the assessment block
- **Shows:** Confidence bar with 40% / 70% thresholds, top-1 vs top-2 gap, normalized entropy, reliability label

### Updated: CNN Models tab

- **Component:** `frontend/src/components/ModelsPanel.tsx` (replaces inline cards in `App.tsx`)
- **Urban section:** Custom CNN, ResNet50, MobileNetV2 (from Supabase `model_benchmarks` or API fallback)
- **Animal section:** MobileNetV2 animal expert card (ESC-50, ~60% accuracy, 0.607 macro F1 — from training summary)
- **Deployment profiles:** Static cards mapping models to mobile edge, GPU server, and baseline use cases

### New: Help banner

- **Component:** `frontend/src/components/AppHelpBanner.tsx`
- **Where:** Below the header on all tabs until dismissed
- **Storage:** `localStorage` key `sap-help-dismissed`

---

## June 2026 — Platform fixes and polish (pre–Phase A)

### Analytics and history

- Fixed silent Supabase save when migration 002 columns were missing (`predictions_repo.py` fallback insert)
- `AnalyticsDashboardPanel`: loader, empty state, error retry
- `PredictionHistoryPanel`: loads via backend API only; model column; reliability fallback

### Datasets tab

- Processing Mode hidden (domain taken from sample)
- WaveLoader on Analyze / Compare
- Buttons disabled during inference

### Header controls

- Processing Mode, Backend Model, Grad-CAM hidden on Analytics, History, CNN Models, and Showcase tabs

### Other

- Microphone: WebM → WAV conversion (`lib/audio.ts`)
- Play Sound: fixed blob URL lifecycle in analysis report
- Smart Auto-Router: fixed serialization mismatch in inference service
- OpenAPI/Swagger documentation expanded

---

## Files added (Phase A)

```
frontend/src/components/
  ShowcasePanel.tsx
  ComparisonWinnerCard.tsx
  ConfidenceCalibrationPanel.tsx
  ModelsPanel.tsx
  AppHelpBanner.tsx
```

## Files modified (Phase A)

```
frontend/src/App.tsx
frontend/src/components/AnalysisResults.tsx
frontend/src/components/ModelComparisonPanel.tsx
```
