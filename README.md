---
title: Credit Scoring Model
emoji: 📊
colorFrom: purple
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
license: other
---

#  Credit Scoring Model

Logistic regression credit scoring model for the  unsecured personal loan
portfolio, exposed as a REST inference endpoint compatible with the
**Holistic AI Tracer → Artifacts → Custom API** connector.

**Synthetic data. Built for AI governance platform onboarding and assurance testing.
Not a production credit decisioning model and not fit for use on real applicants.**

## Model

| | |
|---|---|
| Algorithm | Logistic regression, L2 regularised (C = 1.0) |
| Target | `target_90dpd_12m` — 90+ days past due, or default treatment, within 12 months of drawdown |
| Training rows | 22,549 (TRAIN partition) |
| Validation rows | 7,451 |
| Out-of-time test rows | 10,000 |
| Training base rate | 5.87% |
| Input features | 46 raw fields (2 further features derived inside the service) |

### Performance

| Split | AUC | Gini | PR-AUC | Brier |
|---|---|---|---|---|
| Validation | 0.7904 | 0.5807 | 0.2286 | 0.04873 |
| Out-of-time test | 0.8119 | 0.6237 | 0.2572 | 0.04899 |

### Operating point

Decision threshold **0.1046**, the 85th percentile of the out-of-time score
distribution. This gives a 15% decline rate capturing 53.8% of 90dpd bads
(20.9% bad rate among declined, 3.18% among approved).

The threshold is deliberately not 0.5. At 0.5 the model flags well under 1% of
applications, which collapses the positive rate and makes disparate impact and
equal opportunity metrics unstable. Callers can override it per request.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/predict` | Tabular inference — probabilities, binary decisions, scorecard points |
| POST | `/predict_proba` | Probabilities only |
| GET | `/schema` | Input column order and field-level schema |
| GET | `/model_card` | Performance, operating point, feature list |
| GET | `/health` | Liveness |

### Request

```json
{"data": [[ ... 46 values in /schema column order ... ]]}
```

Positional arrays, arrays of objects (`dataframe_records`), and an explicit
`columns` override are all accepted. Missing values may be sent as `null` —
numeric fields are median-imputed and categorical fields mapped to `MISSING`,
using training-set statistics.

### Response

```json
{"probabilities": [0.0548], "predictions": [0], "scores": [764],
 "threshold": 0.1046, "n_rows": 1, "model_version": "1.0.0", "latency_ms": 2.1}
```

`scores` are scaled scorecard points: 600 at 1:1 odds, 40 points to double the odds.
Higher points mean lower risk.

## How this Space runs

Gradio SDK. `app.py` builds a FastAPI application, registers the REST routes on
it, and then mounts the Gradio Blocks UI at `/` with `gr.mount_gradio_app`.
Because the REST routes are registered before the mount, `/predict`, `/health`,
`/schema` and `/model_card` take precedence over the UI's catch-all — one
process on port 7860 serving both the interactive demo and the machine endpoint.

Dependencies are pandas and NumPy only. Gradio is supplied by the Space runtime
and is not pinned in `requirements.txt`. AUC in the UI is computed with a
tie-corrected rank statistic rather than pulling in scikit-learn, which keeps
the image small and cold starts short.

## Implementation note

The fitted pipeline is serialised as plain JSON coefficients
(`model_artifact.json`) and scored in NumPy rather than shipped as a pickle.
This removes scikit-learn version coupling between training and serving, keeps
cold start fast, and makes every preprocessing constant — imputation medians,
scaler parameters, one-hot category lists — directly auditable in the repo.
Parity against the fitted scikit-learn pipeline is exact to 1.1e-16.

## Preprocessing

- **Numeric** — median imputation (training statistics), then standardisation.
- **Categorical** — missing mapped to `MISSING`, then one-hot; unseen categories at
  inference map to an all-zero block rather than erroring.
- **Derived in-service** — `bur_no_delinquency_flag` (the 999 sentinel in
  `bur_months_since_delinquency`, recoded and the sentinel nulled) and
  `is_existing_customer` (whether internal banking fields are populated).
