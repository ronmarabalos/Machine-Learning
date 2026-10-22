# Phase 4B: Spending forecasts

**Date:** 2026-10-23
**Roadmap:** Phase 4, regression + gradient boosting, feature engineering, time-based splits, and the Ship line (`ml-roadmap.md`)
**Repos:** `machine-learning` (generator, notebook, `service/`) and `budgettracker/mobile` (app)

## Goal

Spendify tells the user two things about the future, per expense category: where this month will **end** at the current pace, and roughly what **next month** will cost. A gradient-boosting model trained in a notebook makes both predictions, served by the existing FastAPI service on Vercel. The app shows them on Budgets, Reports and through Ollie.

**Success criteria**
- The notebook compares each model with a simple baseline on a time-based split and records which one ships, with MAE in pesos.
- `POST /forecast` returns `spent`, `end_of_month` and `next_month` per category for categories with ≥ 3 full months of history.
- Budgets shows "On pace for ₱ X" per category budget; Reports shows a "Next month" section; Ollie warns when a category is on pace to go over its budget.
- Offline, or with the service down, the same screens show baseline numbers computed on the phone.
- Only daily expense totals per category leave the phone, and custom category names never do.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| What is forecast | End-of-month pace **and** next month's total | User choice |
| Where it runs | `/forecast` on the existing Vercel service | Real model serving; completes the roadmap Ship line |
| Model shape | One global model per forecast, all categories together, category as a feature | Thousands of rows instead of ~11 per category |
| Target | Ratio to the category's 3-month average, converted back to pesos | A model trained on fake data still fits users who spend more or less |
| Model type | HistGradientBoosting vs LightGBM, chosen by the notebook | LightGBM may not load on Vercel (`libgomp`); HistGB is already installed |
| Gate | A model ships only if its test MAE beats the baseline; otherwise the baseline ships | No ML for its own sake |
| Training data | New 3-year fake file; the 1-year file is unchanged | 12 months is too little; Week 1–4 results stay valid |
| Where shown | Budgets, Reports, Ollie | User choice |
| Privacy | Daily totals per category; default category names only, custom ones sent as `null` | Minimum data needed for the features |
| Setting | "Spending forecasts" switch, default on | Same pattern as smart category suggestions |

## Part 1: Data and notebook (`machine-learning`)

### Generator (`scripts/generate_fake_data.py`)

- New option `--years N` (default 1). `--years 1` keeps today's behaviour exactly: same rows, same file `data/fake_transactions.csv`, checked against `data/fake_transactions_before.csv` as in Week 4.
- `--years 3` writes `data/fake_transactions_3y.csv`, ending on the same pinned `END` (2026-10-21), and adds three patterns only in that mode:
  - **growth:** expense amounts rise about 5% a year (linear in time)
  - **December:** Shopping, Food & Drinks and Entertainment amounts ×1.3 in December
  - **weekends:** Food & Drinks, Coffee & Snacks and Entertainment get about 30% more purchases on Saturday and Sunday
- Spikes, sprees and bills (on the 5th) work as in the 1-year file. The same self-check asserts run for both files.

### Notebook `notebooks/05-forecasting.ipynb` (guided TODOs)

The user writes the TODO cells; the scaffold gives hints and checks, like Week 4.

- **Setup:** load the 3-year file, expenses only.
- **A. Daily table:** daily totals per category, every calendar day present (missing days = 0), dates as local dates.
- **B. Features.** One row per (category, month, cut-off day 1–28). Everything is computed from data **before or on** the cut-off day:
  - `spent`: spent this month through the cut-off day
  - `day`, `days_left`, `weekend_days_left`
  - `last_month`, `avg3`, `avg6`: totals of the previous 1, 3 and 6 full months
  - `last_month_share_by_day`: share of last month's total already spent by the same day last month (captures recurring payments such as bills on the 5th)
  - `month_of_year`, `category` (11 default names plus `other`)
  - Rows need `avg3 > 0` and 6 earlier full months.
  - A **look-ahead check** cell recomputes one row by hand from earlier data only.
- **Targets:**
  - end-of-month: `eom_ratio = (month total − spent) / avg3`
  - next month: one row per (category, month T) with features from full months up to T−1 (`last_month`, `avg3`, `avg6`, `month_of_year` of T+1, `category`); `next_ratio = total(T+1) / avg3`. At serving time T is the current month, so features use only completed months.
- **C. Time split:** the last 6 months are the test set, everything before is training. A prompt asks why a random split would cheat.
- **D. End-of-month model:** baseline `spent / day × days_in_month` vs HistGradientBoostingRegressor vs LightGBM. MAE in pesos on the test set, overall and **by cut-off day** (does day 5 look like a safe start for Ollie's pace warning?).
- **E. Next-month model:** baseline `avg3` vs the same two models, MAE in pesos.
- **F. Decide:** gate per forecast; the user records the winner and why, and the MAE table. The notebook saves:
  - `service/model/forecast_eom.joblib` and `service/model/forecast_next.joblib`, each a dict `{"model": <estimator or None>, "features": [<column names>]}`. `None` means the baseline ships.
  - `service/tests/fixtures/forecast_sample.json`: the daily totals for two categories over 7 months plus the feature rows the notebook computed for one cut-off date. The service test compares its own features against this file.

## Part 2: The `/forecast` endpoint (`machine-learning/service`)

### Request

```json
{ "today": "2026-10-23",
  "series": [ { "key": "<categoryId>", "category": "Food & Drinks",
                "daily": { "2026-04-01": 312.5, "2026-04-03": 98.0 } } ] }
```

- `today`: the phone's local date. `daily`: expense totals for local dates from the first day of the month 6 months before `today`'s month through `today`. Days with no spending may be left out (= 0).
- `category`: a default category name, or `null` for a custom category (→ `other`).
- `key`: echoed back; the server does not interpret it.
- **Validation (422):** `today` is a valid date; at most 50 series; every date is valid, within the window and not after `today`; every amount ≥ 0 and finite.

### Response

```json
{ "forecasts": [ { "key": "...", "spent": 4100.0, "end_of_month": 8200.0, "next_month": 7900.0 } ] }
```

- `end_of_month` ≥ `spent`; both forecasts ≥ 0; rounded to centavos.
- A series with fewer than 3 full months of history before this month, or `avg3 = 0`, is left out of `forecasts`.
- If `today` is past day 28, the end-of-month model uses cut-off day 28 features with the real `days_left` (training covered days 1–28).

### Code

- `service/forecast.py`: loads both joblib files once at import; `features(...)` builds the same columns as the notebook; `predict(request) -> list[dict]`. When `model` is `None`, it computes the baseline.
- `service/app.py`: `POST /forecast` with pydantic models. The request body is never logged.
- `service/requirements.txt`: add `lightgbm` (pinned) only if the notebook picks it.

### Tests (`service/tests/`, pytest)

- the 422 cases above
- a short-history category is left out
- `end_of_month ≥ spent`
- **feature parity:** `features()` on `forecast_sample.json` equals the notebook's rows
- a baseline-only joblib (`model: None`) returns the baseline numbers

After deploying, recheck the cold start (6.1 s before). If LightGBM fails to load on Vercel, switch to HistGB, retrain in the notebook, and record it as a decision.

## Part 3: The app (`budgettracker/mobile`)

### Files

- `src/services/forecast.ts`:
  - `buildForecastRequest(transactions, categories, now)`: pure; expenses only, local dates, the 7-month window, custom names → `null`, categories without spending left out
  - `fetchForecasts(...)`: POST with a timeout like `category-suggest.ts`; caches the last response in memory keyed by the request JSON, so an unchanged request makes no call
- `src/lib/forecast-fallback.ts`: `fallbackForecasts(request)` gives the same response shape from the two baselines (`spent / day × days_in_month`, `avg3`), with the same 3-month rule.
- `src/hooks/useForecasts.ts`: returns `Map<categoryId, Forecast>`; server result, else fallback; nothing when the setting is off.
- Settings: `spendingForecasts: boolean` (default `true`) in `types/index.ts` and `settings-repo.ts`; a switch on AccountScreen next to "Smart category suggestions"; one line in the app's privacy section saying what is sent.

### Screens

- **Budgets:** each **category** budget row shows "On pace for ₱ X" (`end_of_month`), in the warning colour when X > the budget. No line for the overall budget, or when the category has no forecast.
- **Reports:** a "Next month" section listing each category's `next_month`, largest first, with a total.
- **Ollie** (`ollie-messages.ts`): new input `pace: { categoryId, name, forecast, budget }[]`; new message:
  - id `pace:<categoryId>:<YYYY-MM>`, pose thinking, dismissible, View → `/budgets`
  - text: "At this pace, {category} will go over budget by about ₱ {forecast − budget} this month."
  - only for category budgets that are not already over 100%, only from **day 5** of the month (or the day the notebook shows is safe)
  - priority: over budget → unusual purchase → unusual week → **on pace to go over** → near budget → greetings

### Tests

- `scripts/check-forecast.mjs`: request building (local dates, window edges, custom name → `null`, income ignored) and fallback numbers, including the 3-month rule
- `scripts/check-ollie.mjs`: pace message text, the day-5 rule, the over-100% exclusion, and the new priority order
- typecheck and lint pass

**Device checklist** (test account with the fake CSV imported):
- [ ] Budgets shows "On pace for ₱ …" on category budgets
- [ ] A category budget set below its pace turns the line to the warning colour and Ollie shows the pace message
- [ ] Reports shows "Next month" with a total
- [ ] Airplane mode → the same screens still show numbers (fallback)
- [ ] Spending forecasts off → no pace lines, no Next month section, no pace message
- [ ] Light and dark themes look right

## Build order

1. Generator `--years 3` (machine-learning)
2. Notebook scaffold; the user completes it and saves the models and the fixture
3. `service/forecast.py`, route, tests; deploy; cold-start check
4. App: request builder and fallback, hook, setting, Budgets, Reports, Ollie
5. Device checklist; mark the roadmap

## Roadmap items this covers

- Regression + gradient boosting (notebook D–E)
- Feature engineering: rolling averages (`avg3`, `avg6`), day-of-week (`weekend_days_left`), recurring payments (`last_month_share_by_day`)
- Avoid data leakage with time-based splits (notebook C plus the look-ahead check)
- Ship line: `/forecast` endpoint and screens. Anomalies run on the phone by design (Phase 4A), noted when marking.

## Out of scope

- Overall-budget pace (forecasts are missing for short-history categories, so a sum would be too low)
- Confidence ranges, charts, per-user retraining
- K-means / PCA (separate roadmap item)
