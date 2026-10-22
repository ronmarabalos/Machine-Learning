# Week Types Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cluster weeks with K-means (drawn with PCA) in a guided notebook, and show the user's week type on Reports and through Ollie, computed on the phone from exported centres.

**Architecture:** The notebook trains K-means on 9 relative weekly features and exports scaler numbers, centres, user-written type names and 3 sample weeks as JSON. The app copies that JSON. A pure TypeScript module rebuilds the same features and picks the nearest centre. The Reports card and an Ollie greeting render the result.

**Tech Stack:** Python 3.14, pandas, scikit-learn 1.9.1 (KMeans, StandardScaler, PCA, silhouette_score), matplotlib; Expo / React Native / TypeScript, Node 22 `--experimental-strip-types` self-checks.

**Spec:** `docs/superpowers/specs/2026-10-23-week-types-design.md`

## Global Constraints

- No AI attribution in commits. Never commit `data/`, `models/`, `CLAUDE.md`, `.claude/`, `.superpowers/`.
- In `budgettracker`: make changes, **do not stage or commit**. The user commits.
- Don't edit a notebook the user has open in Jupyter. Notebook TODO cells are the user's to write.
- Feature order (9): `share_Food & Drinks, share_Coffee & Snacks, share_Groceries, share_Transportation, share_Bills & Utilities, share_Shopping, share_Healthcare, share_Entertainment, total_ratio`. A share is that category's total ÷ the week's total over **all** expenses.
- `total_ratio` = the week's total ÷ the median of the previous weeks' totals (up to 8, counted from the first week with any expense, empty weeks = 0); `null` with fewer than 4 earlier weeks, a total of 0, or a median of 0. The current week's total is × 7 ÷ days so far (Mon = 1 … Sun = 7).
- Weeks start Monday at local midnight. `current` only from Wednesday.
- `KMeans(n_init=10, random_state=0)`; `classify` = nearest centre by squared Euclidean distance in scaled space.
- Ollie text: `So far this looks like a {name}. {description}`, id `week-type:<id>`, pose from the type, not dismissible.
- Ollie order: over budget → unusual purchase → unusual week → pace → near budget → week type → on-track greeting → default greeting.
- Setting `weekTypeGreeting` (default `true`), switch "Week type from Ollie", subtitle "Ollie says what kind of week it's been". It's placed after "Unusual spending alerts".
- Chip colours by type index: `primary`, `secondary`, `tertiary`, `error`, then repeat. Grey (`surface-container-high`) when the type is null.

## Review Focus

1. **Sunday:** `getDay()` is 0 on Sunday, but it must count as day 7 (× 7 ÷ 7), not day 0 (a division by zero, or the wrong week). Tested in check-week-types.
2. **Future-dated transactions inside the current week** (scheduled for Friday when it's Wednesday) must not count yet. Tested.
3. **Uncategorized or custom-category expenses** count toward the total and shrink the shares. They're not dropped. Tested.
4. **A model file whose `types`, `centers`, `mean` and `scale` lengths disagree** (a bad copy from the notebook) must fail the self-check loudly instead of returning `undefined` names. Tested in check-week-types with the real file.
5. **The year boundary:** a week starting Mon 2025-12-29 holds Jan 1–4 expenses, and "8 weeks earlier" reaches back into November. Tested.

---

### Task 1: Week 6 notebook scaffold

**Files:**
- Create: `notebooks/06-week-types.ipynb` (built by scratch `mknb6.py OUT [--solved]`, like `mknb5.py`)

**Interfaces:**
- Produces (when run): `models/week_types.json` = `{features: string[9], mean: number[9], scale: number[9], centers: number[k][9], types: {id: "t0".., name, description, pose}[k], samples: {features: number[9], type: string}[3]}`

- [ ] **Step 1: Write the scaffold** with sections A–G from the spec. TODO cells hold hints only. These are **given cells, not TODOs:**
  - Setup (constants `CATEGORIES`, `FEATURES`, load data, whole weeks only)
  - the look-ahead check after B (recompute one row's shares and `total_ratio` from raw expenses)
  - the PCA plotting helper's axis labels
  - G, the export cell. It asserts `len(TYPES) == k`, that each pose is valid, and that each name ends in "week". Samples are the week nearest each of 3 centres, from 3 different clusters.

  Prompts:
  - C: why scale
  - D: which k and why (silhouette ~0.2: what does that say?)
  - E: name the types
  - F: what PC1 and PC2 mean

- [ ] **Step 2: Build `--solved`, run it, and record reference numbers**
Run: `.venv/Scripts/python mknb6.py notebooks/_06-solved.ipynb --solved && .venv/Scripts/python -m nbconvert --to notebook --execute notebooks/_06-solved.ipynb --output _06-solved-run.ipynb`
Expected: no errors; the look-ahead check passes; it prints silhouette per k and the PC1/PC2 variance; it writes `models/week_types.json`. Record k, the silhouettes and the cluster sizes in the ledger. Copy the JSON to the scratchpad. Delete both `_06-*` notebooks.

- [ ] **Step 3: Build the TODO version and commit**
Run: `.venv/Scripts/python mknb6.py notebooks/06-week-types.ipynb` and validate with nbformat.
Expected: valid; TODO cells present. Commit "Add Week 6 week types notebook".

### Task 2: App week-type logic

**Files** (`budgettracker/mobile`):
- Create:
  - `src/lib/week-types.ts`
  - `src/lib/week-model.ts`
  - `src/lib/week-types.json` (the scratch export from Task 1 for now)
  - `scripts/check-week-types.mjs`
- Modify: `src/lib/anomalies.ts` (export the existing `mondayOf`; no behaviour change)

**Interfaces:**
- Consumes: `mondayOf(time: number): Date` from `anomalies.ts`; `OlliePose` type from `ollie-messages.ts`
- Produces:
  ```ts
  export type WeekType = { id: string; name: string; description: string; pose: OlliePose };
  export type WeekModel = { features: string[]; mean: number[]; scale: number[]; centers: number[][]; types: WeekType[] };
  weekFeatures(transactions: Transaction[], categoryNames: Map<string, string>, weekStart: Date, now: Date): number[] | null
  classify(model: WeekModel, features: number[]): WeekType
  weekTypes(model: WeekModel, transactions: Transaction[], categories: Category[], now: Date):
    { current: WeekType | null; recent: { weekStart: Date; type: WeekType | null }[] }
  ```
  `week-types.ts` takes the model as a parameter and has type-only imports (Node runs it). `week-model.ts` is one line, `import model from './week-types.json'`, exported as `WEEK_MODEL: WeekModel`. Metro handles the JSON import; Node never loads this file.

- [ ] **Step 1: Write `check-week-types.mjs`.** It reads the JSON with `fs` and sets `process.env.TZ = 'Asia/Manila'`. Asserts:
  - the model's lengths agree: features 9, mean 9, scale 9, every centre 9, and `types.length === centers.length` (Review Focus 4)
  - parity: `classify(model, s.features).id === s.type` for every sample
  - `classify` on a hand-made 2-centre toy model picks the nearer centre
  - partial week: with 6 earlier weeks of ₱700 (one ₱100 Food purchase a day), ₱300 of Food Mon–Wed of the current week, and `now` = Wednesday 18:00 → `total_ratio` = 1 (300 × 7 ÷ 3 = 700) and the Food share = 1
  - Sunday: the same history plus ₱700 Mon–Sun, `now` = Sunday → `total_ratio` = 1 (Review Focus 1)
  - a transaction dated Friday when `now` is Wednesday doesn't count (Review Focus 2)
  - a custom-category ₱300 next to ₱300 of Food → Food share 0.5, and the total includes both (Review Focus 3)
  - `null` for 3 earlier weeks; a 0 total; a 0 median (4 earlier weeks of all 0 after a first expense)
  - year boundary: `weekStart` Mon 2025-12-29 with Jan 2 spending and history from November → non-null (Review Focus 5)
  - `weekTypes`: `current` is null on Monday and Tuesday and set on Wednesday; `recent` has 8 entries, oldest first, the last one starting the previous Monday

- [ ] **Step 2: Run it to watch it fail**
Run: `node --no-warnings --experimental-strip-types scripts/check-week-types.mjs`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement** `week-types.ts`, `week-model.ts`, the JSON copy and the `mondayOf` export.

- [ ] **Step 4: Verify**
Run: `node --no-warnings --experimental-strip-types scripts/check-week-types.mjs && node --no-warnings --experimental-strip-types scripts/check-ollie.mjs && npx tsc --noEmit && npx eslint src/lib scripts/check-week-types.mjs`
Expected: `check-week-types: all passed`; `check-ollie` still passes; no errors.

### Task 3: Ollie greeting and setting

**Files** (`budgettracker/mobile`):
- Modify:
  - `src/features/companion/ollie-messages.ts`, `src/features/companion/OllieCard.tsx`, `scripts/check-ollie.mjs`
  - `src/types/index.ts`, `src/repositories/settings-repo.ts`, `src/features/account/AccountScreen.tsx`

**Interfaces:**
- Consumes: `weekTypes`, `WEEK_MODEL`, `WeekType`
- Produces: `ollieMessages` input gains `weekType: WeekType | null`

- [ ] **Step 1: Add the failing checks** to `check-ollie.mjs`. `base` gains `weekType: null`. With `quiet = { id: 't3', name: 'Quiet week', description: 'Mostly meals and commuting.', pose: 'celebrating' }`:
  - `ollieMessages({ ...base, weekType: quiet })[0]` deep-equals `{ id: 'week-type:t3', pose: 'celebrating', dismissible: false, text: 'So far this looks like a Quiet week. Mostly meals and commuting.' }`
  - with near-budget food at 85% → `['budget:food:2026-10:near', 'week-type:t3']`, then the greeting is not included
  - with all budgets on track → `week-type:t3` comes first, ahead of `greet:on-track`
  - `weekType: null` → the existing greetings are unchanged

- [ ] **Step 2: Run it to watch it fail**
Run: `node --no-warnings --experimental-strip-types scripts/check-ollie.mjs`
Expected: FAIL on the week-type assert.

- [ ] **Step 3: Implement**
  - `ollieMessages`: when `weekType` is set, its message replaces both greetings (the last element)
  - `OllieCard`: `weekTypes(WEEK_MODEL, …).current` when `settings.weekTypeGreeting` is on, otherwise `null`
  - the setting, its default and the switch

- [ ] **Step 4: Verify**
Run: `node --no-warnings --experimental-strip-types scripts/check-ollie.mjs && npx tsc --noEmit && npx eslint src/features scripts/check-ollie.mjs`
Expected: passes; no errors (the 2 old AccountScreen warnings may stay).

### Task 4: Reports "Your weeks" card

**Files:** Modify `src/features/reports/ReportsScreen.tsx`.

**Interfaces:** Consumes `weekTypes`, `WEEK_MODEL`.

- [ ] **Step 1: Implement the card** before "Next month":
  - title "Your weeks"
  - the headline is `current` ("This week so far: **{name}**") or, when `current` is null, the latest `recent` entry ("Last week: **{name}**"), then the description
  - 8 chips: a coloured dot or pill with a date label (`Oct 6`, `toLocaleDateString('en-US', { month: 'short', day: 'numeric' })`), grey when null
  - a legend of the types that appear
  - the whole card is hidden when neither `current` nor any `recent` entry has a type

  The data is a `useMemo` over transactions and categories. There's no separate test (it's layout only); the logic is covered by Task 2.

- [ ] **Step 2: Verify**
Run: `npx tsc --noEmit && npx eslint src/features/reports`
Expected: no errors.

### Task 5: The user completes the notebook

- [ ] **Step 1: The user completes `notebooks/06-week-types.ipynb`** and runs G. Review it per CLAUDE.md:
  - the look-ahead check passes
  - k is justified with the elbow and silhouette
  - every type is named, with a description and a pose
  - PC1 and PC2 are explained
  - no TODOs are left
- [ ] **Step 2: Copy `models/week_types.json` to `budgettracker/mobile/src/lib/week-types.json`**, then re-run Task 2 Step 4.
Expected: all pass. A parity failure means the user's features differ from the spec; find the column.
- [ ] **Step 3: Commit the notebook** ("Complete Week 6 week types notebook").

### Task 6: Device checklist and roadmap

- [ ] **Step 1:** The user goes through the spec's device checklist and ticks it.
- [ ] **Step 2:** Mark "Unsupervised learning: K-means clustering, PCA" with the date and proof (`notebooks/06-week-types.ipynb`, week types in the app). Every Phase 4 item and the Ship line are then done, so the heading gets ` ✅`.
- [ ] **Step 3:** Commit and push the roadmap and spec. List the budgettracker files for the user.
