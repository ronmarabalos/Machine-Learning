# Spending Forecasts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Forecast each expense category's end-of-month total and next month's total with gradient boosting, serve them from `POST /forecast`, and show them on Budgets, Reports and through Ollie.

**Architecture:** A 3-year fake dataset trains two global models (all categories together, ratio targets) in a guided notebook. `service/forecast.py` rebuilds the same features from daily totals sent by the phone and predicts. The app builds the request, falls back to on-phone baselines when offline, and renders the numbers.

**Tech Stack:** Python 3.14, pandas, scikit-learn 1.9.1 (HistGradientBoostingRegressor), LightGBM (notebook only unless it wins and deploys), FastAPI, pytest, Vercel; Expo / React Native / TypeScript, Node 22 `--experimental-strip-types` self-checks.

**Spec:** `docs/superpowers/specs/2026-10-23-spending-forecast-design.md`

## Global Constraints

- No AI attribution in commits. Never commit `data/`, `models/`, `CLAUDE.md`, `.claude/`, `.superpowers/`.
- In `budgettracker`: make changes, **do not stage or commit**. The user commits.
- Don't edit a notebook the user has open in Jupyter. Notebook TODO cells are the user's to write.
- `--years 1` (the default) must produce a byte-identical `data/fake_transactions.csv`.
- Expense categories, in this fixed order (index = `category_code`): `['Food & Drinks', 'Coffee & Snacks', 'Groceries', 'Transportation', 'Bills & Utilities', 'Shopping', 'Healthcare', 'Entertainment', 'other']`.
- **Feature definitions** (identical in notebook, service and fallback). For category c, month M, cut-off day d, where `total(X)` = the month total and full months = months before M, starting at c's first month with spending:
  - `avg3` = mean total of the last 3 full months (must be > 0). `avg6` = mean of the last min(6, available) full months. `last_month` = total(M−1).
  - End-of-month features: `category_code`, `day` = d, `days_left` = days_in_month(M) − d, `weekend_days_left` = Saturdays and Sundays in days d+1 to the end of the month, `spent_ratio` = spent(days 1..d) / avg3, `last_month_ratio` = last_month / avg3, `avg6_ratio` = avg6 / avg3, `last_month_share_by_day` = spent in M−1 through day min(d, days_in_month(M−1)) / total(M−1) (0 when the total is 0), `month_of_year` = M's month.
  - End-of-month target `eom_ratio` = (total(M) − spent) / avg3 → pesos: `end_of_month = spent + max(0, ratio) × avg3`.
  - Next-month features (M = the current month, only full months used): `category_code`, `last_month_ratio`, `avg6_ratio`, `month_of_year` = (M+1)'s month. Target `next_ratio` = total(M+1) / avg3 → pesos `max(0, ratio) × avg3`.
  - Baselines: end-of-month = `spent / d × days_in_month`; next month = `avg3`.
- Serving: at least 3 full months of history, otherwise the category is left out. Training rows: 6 full months of history, d in 1..28. For d > 28 the service computes features normally (trees treat `day` > 28 like 28).
- Request window: from the 1st of the month 6 months before `today`'s month, through `today`. At most 50 series. Amounts ≥ 0 and finite.
- Default category names are sent; anything else is sent as `null` (→ `other`).
- Ollie pace text: `At this pace, {category} will go over budget by about ₱ {forecast − budget} this month.` id `pace:<categoryId>:<YYYY-MM>`, thinking, dismissible, View → `/budgets`. Only from day `PACE_FROM_DAY = 5`, only category budgets with spent < budget.
- Ollie priority: over budget → unusual purchase → unusual week → pace → near budget → greetings.

## Review Focus

1. **The window crossing a year** (today in January–June): month arithmetic in the service, the request builder and the fallback must handle Dec → Jan. Tests use `today = 2027-02-10`.
2. **A category with a gap month** (spending in May, nothing in June, spending in July): June is a full month with total 0 and counts toward avg3. It's not skipped. Tested in service and fallback.
3. **Day 29–31 and February:** `days_left` and `weekend_days_left` for the 31st and for Feb 28 in a non-leap year. `end_of_month ≥ spent` always. Tested in the service.
4. **A transaction late on the last day of a month in local time** lands in that month, even though it falls on the 1st in UTC. Tested in check-forecast with a `T23:30` local timestamp.
5. **Server disagreement:** a malformed or partial server reply (missing keys, negative numbers, strings) must fall back instead of rendering `NaN`. Tested in check-forecast through `parseForecasts`.

---

### Task 1: Generator `--years 3`

**Files:**
- Modify: `scripts/generate_fake_data.py`

**Interfaces:**
- Produces: `data/fake_transactions_3y.csv` (same columns as the 1-year file), `generate(days=365, patterns=False)`

- [ ] **Step 1: Snapshot the current 1-year file**
Run: `cp data/fake_transactions.csv "$SCRATCH/fake_1y_before.csv"`
Expected: the file is copied.

- [ ] **Step 2: Add 3-year self-checks to `__main__`** (they run only for `--years 3`):
  - December means of Shopping, Food & Drinks and Entertainment are ≥ 1.2× their mean in the other months
  - the last 12 months' mean expense amount is ≥ 1.07× the first 12 months' mean (two years of 5% growth, minus noise)
  - Saturday + Sunday purchases per day in Food & Drinks are ≥ 1.2× the weekday rate
  - the existing checks still pass (spike share 0.5–2%; the spree-row bounds scale with `years`: `24*years ≤ n ≤ 36*years`, with sprees per year = 6)

- [ ] **Step 3: Run `--years 3` to watch it fail**
Run: `.venv/Scripts/python scripts/generate_fake_data.py --years 3`
Expected: an argparse error or a failed assert. The option doesn't exist yet.

- [ ] **Step 4: Implement** `argparse --years` (default 1). `days = 365 * years`, output `fake_transactions_3y.csv` when years ≠ 1. `patterns = years > 1` turns on:
  - growth `× (1 + 0.05 × years_since_start)`
  - the December ×1.3 for the 3 categories
  - weekend purchase count ×1.3 for Food & Drinks, Coffee & Snacks and Entertainment

  Every new random draw is guarded by `if patterns`, so the 1-year sequence is untouched. Sprees: `6 * years` Mondays.

- [ ] **Step 5: Verify**
Run: `.venv/Scripts/python scripts/generate_fake_data.py && cmp data/fake_transactions.csv "$SCRATCH/fake_1y_before.csv" && .venv/Scripts/python scripts/generate_fake_data.py --years 3 && echo OK`
Expected: `OK` (cmp silent = identical).

- [ ] **Step 6: Commit** `scripts/generate_fake_data.py`, message "Add 3-year option with growth, December and weekend patterns to generator".

### Task 2: Week 5 notebook scaffold

**Files:**
- Create: `notebooks/05-forecasting.ipynb` (built by a scratch script `mknb5.py OUT [--solved]`, like `mknb4.py`)

**Interfaces:**
- Consumes: `data/fake_transactions_3y.csv`
- Produces (when the user runs section F): `service/model/forecast_eom.joblib` and `service/model/forecast_next.joblib` = `{"model": estimator | None, "features": [column names]}`; `service/tests/fixtures/forecast_sample.json` = `{"request": <ForecastRequest for 2 categories, today 2026-09-15>, "eom": {key: {feature: value}}, "next": {key: {feature: value}}}`

- [ ] **Step 1: Write the scaffold** with sections Setup and A–F from the spec. The TODO cells hold hints only. These are **given cells, not TODOs:**
  - the `CATEGORIES` list
  - the look-ahead check (recompute one end-of-month row and one next-month row by hand from the daily table and compare them, within 1e-9)
  - the `mae_pesos(pred_ratio, rows)` helper
  - the F export cell, which writes the joblibs and the fixture from the user's feature tables. Before saving, it refits each chosen estimator (`sklearn.base.clone`, or a new LightGBM model with the same params) on `X_train.to_numpy()`, so the service can predict from plain NumPy arrays without pandas.
  - a gate cell that prints which forecast ships the model vs the baseline

  Prompts:
  - C: why a random split cheats
  - D: MAE by cut-off day, and is day 5 safe?
  - F: the user's decision and why

- [ ] **Step 2: Build `--solved` in the scratchpad and run it top to bottom**
Run: `.venv/Scripts/python -m nbconvert --to notebook --execute "$SCRATCH/05-solved.ipynb" --output 05-solved-run.ipynb`
Expected: it runs without errors, the look-ahead check passes, and it prints the MAE tables. Write down the baseline and model MAEs for both forecasts as review reference numbers. If LightGBM isn't installed, `pip install lightgbm` into `.venv` (notebook only; not added to `service/requirements.txt`).

- [ ] **Step 3: Copy the solved run's joblibs and fixture into `service/`, uncommitted.** Task 3 develops against them. The user's notebook run overwrites them later. Ruling: the service is built from the spec's feature definitions in parallel with the user's notebook; the parity test re-validates once the user's fixture lands.

- [ ] **Step 4: Build the TODO version into `notebooks/05-forecasting.ipynb` and commit** it, message "Add Week 5 forecasting notebook".

### Task 3: `service/forecast.py` and `POST /forecast`

**Files:**
- Create: `service/forecast.py`, `service/tests/test_forecast.py`
- Modify: `service/app.py`

**Interfaces:**
- Consumes: the joblibs and fixture from Task 2
- Produces: `POST /forecast` with request `{today, series: [{key, category, daily}]}` → `{"forecasts": [{"key", "spent", "end_of_month", "next_month"}]}`. Functions `eom_features(category: str | None, daily: dict[date, float], today: date) -> dict | None`, `next_features(...) -> dict | None`, `predict(today: date, series: list) -> list[dict]`.

- [ ] **Step 1: Write the failing tests** in `test_forecast.py`:
  - `test_feature_parity`: for each key in the fixture, `eom_features(...)` and `next_features(...)` equal the fixture rows (`pytest.approx`, 1e-9)
  - `test_rejects_bad_input` (parametrized, expects 422): `today` = `"2026-13-01"`; 51 series; a date after `today`; a date before the window; amount `-1`; amount `"nan"`
  - `test_short_history_left_out`: a category whose spending starts 2 full months ago is absent from `forecasts`
  - `test_eom_at_least_spent`: on the fixture request, `end_of_month >= spent`, and both forecasts are ≥ 0
  - `test_gap_month_counts_as_zero`: May 300, June nothing, July 300, `today` = Aug 10 → `avg3` = 200 (Review Focus 2)
  - `test_year_boundary`: `today = 2027-02-10` with 6 months of daily data → status 200 and every forecast has all three fields (Review Focus 1)
  - `test_month_end_days`: `eom_features` on 2026-10-31 gives `days_left == 0`, `weekend_days_left == 0`; on 2027-02-27 it gives `days_left == 1`, `weekend_days_left == 1` (Feb 28, 2027 is a Sunday) (Review Focus 3)
  - `test_baseline_when_model_none`: monkeypatch both bundles to `{"model": None, ...}` → `end_of_month == spent / d * days_in_month` and `next_month == avg3`

- [ ] **Step 2: Run them to watch them fail**
Run: `cd service && ../.venv/Scripts/python -m pytest tests/test_forecast.py -v`
Expected: FAIL (ImportError: no module `forecast`).

- [ ] **Step 3: Implement** `forecast.py` using pandas-free stdlib date maths (dicts of daily totals → month totals), so serving doesn't depend on pandas. Load the joblibs once at import into module globals `EOM` and `NEXT`, which the baseline test monkeypatches. Predict with a one-row `np.array` in `bundle["features"]` order. Round to 2 decimals. In `app.py`, add pydantic models with validators for the window, the series count and amounts (finite, ≥ 0), and the route. Never log the body.

- [ ] **Step 4: Run the whole suite**
Run: `cd service && ../.venv/Scripts/python -m pytest -v`
Expected: all pass, including the existing categorizer tests.

- [ ] **Step 5: Commit** `service/forecast.py`, `service/app.py`, `service/tests/test_forecast.py`, message "Add forecast endpoint". The joblibs and fixture wait for the user's notebook (Task 5).

### Task 4: App: request builder, fallback, client

**Files** (`budgettracker/mobile`):
- Create: `src/lib/forecast-fallback.ts`, `scripts/check-forecast.mjs`
- Create: `src/services/forecast.ts`

**Interfaces:**
- Produces:
  ```ts
  export type ForecastRequest = { today: string; series: { key: string; category: string | null; daily: Record<string, number> }[] };
  export type Forecast = { key: string; spent: number; end_of_month: number; next_month: number };
  buildForecastRequest(transactions: Transaction[], categories: Category[], now: Date): ForecastRequest   // services/forecast.ts
  parseForecasts(body: unknown): Forecast[] | null                                                     // services/forecast.ts
  fetchForecasts(req: ForecastRequest, signal?: AbortSignal): Promise<Forecast[] | null>               // services/forecast.ts
  fallbackForecasts(req: ForecastRequest): Forecast[]                                                  // lib/forecast-fallback.ts
  ```
  Both files import only types (so Node can run them). `fetchForecasts` uses `EXPO_PUBLIC_CATEGORIZER_URL`, an 8000 ms timeout and an in-memory cache keyed by `JSON.stringify(req)`.

- [ ] **Step 1: Write `check-forecast.mjs`** with these asserts:
  - income and uncategorized transactions are ignored
  - a custom category name → `category: null`, while default names are kept
  - the window starts on the 1st, 6 months back; a transaction one day earlier and a future one are excluded
  - a local `2026-09-30T23:30` expense lands under `"2026-09-30"` (Review Focus 4)
  - daily amounts on the same day are summed
  - fallback: 3 full months of 300/600/900 and `today` the 10th with 100 spent → `next_month = 600`, `end_of_month = 100 / 10 × days_in_month`
  - fallback leaves out a category with 2 full months
  - fallback with a gap month counts it as 0
  - `today = 2027-02-10` works across the year boundary
  - `parseForecasts` returns `null` for a non-object, a missing `forecasts`, an item with a string amount, or a negative amount, and returns the list for a valid body (Review Focus 5)

- [ ] **Step 2: Run it to watch it fail**
Run: `node --no-warnings --experimental-strip-types scripts/check-forecast.mjs`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** the four functions. Local dates use the same `isoDay` formatting as `anomalies.ts`. Fallback month maths follows the Global Constraints definitions.

- [ ] **Step 4: Run it to watch it pass, then typecheck and lint**
Run: `node --no-warnings --experimental-strip-types scripts/check-forecast.mjs && npx tsc --noEmit && npx eslint src/services/forecast.ts src/lib/forecast-fallback.ts scripts/check-forecast.mjs`
Expected: `check-forecast: all passed`, and no errors.

### Task 5: The user completes the notebook, then the service ships

- [ ] **Step 1: The user completes `notebooks/05-forecasting.ipynb`** and runs F. Review it per CLAUDE.md:
  - the look-ahead check passes
  - the MAEs are close to the Task 2 reference numbers
  - the gate decision is recorded
  - no TODOs are left

  Fix bugs first.

- [ ] **Step 2: If the user picked LightGBM:** add `lightgbm==<installed version>` to `service/requirements.txt`. Otherwise no change.

- [ ] **Step 3: Run the suite with the user's joblibs and fixture**
Run: `cd service && ../.venv/Scripts/python -m pytest -v`
Expected: all pass. A parity failure means the user's features differ from the spec: find which column, then fix the notebook (the user's code) or record a ruling.

- [ ] **Step 4: Commit** the notebook, `service/model/forecast_*.joblib`, the fixture and any requirements change, message "Complete Week 5 forecasting notebook and ship forecast models". Push (Vercel deploys).

- [ ] **Step 5: Verify production**
Run: `curl -s -w "\n%{time_total}s\n" https://machine-learning-rose.vercel.app/health`, then `curl -s -X POST .../forecast -H "Content-Type: application/json" -d @service/tests/fixtures/forecast_request_only.json` (a copy of `fixture["request"]` written to the scratchpad).
Expected: health is `ok`; the cold start is under 10 s; `/forecast` returns 2 forecasts. If LightGBM fails to import on Vercel: switch the notebook to HistGB, rerun F, recommit, and record it in the ledger as a ruling.

### Task 6: App: hook, setting, Budgets, Reports, Ollie

**Files** (`budgettracker/mobile`):
- Create: `src/hooks/use-forecasts.ts` (kebab-case, like the other hooks; the spec's `useForecasts.ts` is renamed to match the repo)
- Modify:
  - `src/types/index.ts` and `src/repositories/settings-repo.ts` (`spendingForecasts: boolean`, default `true`)
  - `src/features/account/AccountScreen.tsx` ("Spending forecasts" switch under "Smart category suggestions")
  - `src/features/budgets/BudgetsScreen.tsx`, `src/features/reports/ReportsScreen.tsx`
  - `src/features/companion/ollie-messages.ts`, `src/features/companion/OllieCard.tsx`, `scripts/check-ollie.mjs`
  - `../website/src/pages/Privacy.tsx` (one paragraph: what `/forecast` receives)

**Interfaces:**
- Consumes: Task 4 functions
- Produces:
  - `useForecasts(): Map<string, Forecast>` keyed by categoryId. It is empty when the setting is off. It uses the server result, falling back to `fallbackForecasts` when the server returns null.
  - `ollieMessages` input gains `pace: PaceItem[]`, with `export type PaceItem = { categoryId: string; forecast: number; budget: number; spent: number }` and `export const PACE_FROM_DAY = 5`

- [ ] **Step 1: Add the failing Ollie checks** to `check-ollie.mjs`:
  - on the 10th, pace `{shop, forecast 9200, budget 8000, spent 4000}` → id `pace:shop:2026-10`, the exact text with `₱ 1,200`, pose thinking, View → `/budgets`
  - on the 4th → no pace message
  - `spent >= budget` → no pace message
  - forecast ≤ budget → none
  - the priority test now expects `[..., 'week:food:2026-10-19', 'pace:shop:2026-10', 'budget:food:2026-10:near', 'greet:default']`

  `base` gains `pace: []`.

- [ ] **Step 2: Run them to watch them fail**
Run: `node --no-warnings --experimental-strip-types scripts/check-ollie.mjs`
Expected: FAIL on the pace assert.

- [ ] **Step 3: Implement**
  - the pace message in `ollieMessages`
  - `OllieCard` builds `pace` from category budgets × `useForecasts()`
  - the hook
  - the setting and switch
  - Budgets: under each category row's progress bar, `On pace for ₱ X`, in `text-error` when X > `budget.amount`, else `text-on-surface-variant`. Shown only when a forecast exists.
  - Reports: a "Next month" card after "Spending by category", with rows sorted by `next_month` descending and a "Total" row. Hidden when the map is empty.
  - the Privacy paragraph

- [ ] **Step 4: Verify**
Run: `node --no-warnings --experimental-strip-types scripts/check-ollie.mjs && node --no-warnings --experimental-strip-types scripts/check-forecast.mjs && npx tsc --noEmit && npx eslint src`
Expected: both checks pass; no type errors; 0 lint errors (the 5 old warnings elsewhere may stay).

### Task 7: Device checklist and roadmap

- [ ] **Step 1: The user runs `npx expo start -c`** with a test account and imports `data/fake_transactions_3y.csv`. The user also sets a Food & Drinks budget below its pace, then goes through the spec's device checklist and ticks it in the spec.
- [ ] **Step 2: Mark the roadmap per CLAUDE.md**, with the date and proof:
  - Regression + gradient boosting
  - Feature engineering (rolling averages, day-of-week, recurring payments)
  - Avoid data leakage with time-based splits
  - Ship line: `✅ date` with the note "anomalies on the phone by design (4A)"

  K-means/PCA stays unchecked, so the Phase 4 heading gets no ✅.
- [ ] **Step 3: Commit and push** the roadmap and the spec checklist in machine-learning. List the budgettracker files for the user.
