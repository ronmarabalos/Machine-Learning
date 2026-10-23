# Phase 4C: Week types (K-means + PCA)

**Date:** 2026-10-23
**Roadmap:** Phase 4, "Unsupervised learning: K-means clustering, PCA" (`ml-roadmap.md`); the last open Phase 4 item
**Repos:** `machine-learning` (notebook) and `budgettracker/mobile` (app)

## Goal

K-means groups the user's weeks into a few spending "types" (for example *Big shopping week*, *Grocery run week*, *Bills week*, *Quiet week*), and PCA draws them in 2D. The user names the types in a notebook. The app tells the user which type this week looks like, on Reports and through Ollie, entirely on the phone.

**Success criteria**
- The notebook chooses k with the elbow and silhouette, names every cluster, and explains PC1 and PC2 from their loadings.
- The app assigns the notebook's 3 sample weeks to the same clusters the notebook did.
- Reports shows "Your weeks": this week's type (from Wednesday) or last week's, plus the last 8 weeks as coloured chips.
- With no alerts pending, Ollie says which type the week looks like, with the pose chosen in the notebook.
- Works offline; nothing leaves the phone.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Scope | Notebook + app feature | User choice |
| What is clustered | Monday-start weeks | ~150 weeks in the 3-year data; weeks have readable patterns |
| Where it runs | Notebook trains; app assigns the nearest centre from exported numbers | No server or network; private; a few lines of maths |
| Features | 8 category shares + `total_ratio` (week total ÷ median of the 8 earlier weeks) | Relative, so centres from fake data fit real users |
| Partial week | Total so far × 7 ÷ days so far; current type shown from Wednesday | Early in the week there's too little spending to judge |
| Where shown | Reports card **and** an Ollie greeting | User choice |
| Setting | "Week type from Ollie" switch (default on) for the Ollie line only | The card is information the user opens; the greeting appears unasked |

Probe on the 3-year data (k = 4, before the notebook): shopping-heavy (46 weeks), grocery run (47), bills week (33), quiet week (26); silhouette ≈ 0.21 for every k from 2 to 7. The notebook makes the real choice.

## Part 1: Notebook `notebooks/06-week-types.ipynb` (guided TODOs)

Data: `data/fake_transactions_3y.csv`, expenses only, dates used as local dates (as in Week 5).

- **A. Weekly table:** Monday-start weeks (`to_period('W-SUN')`), one column per category, every week present (empty = 0). Partial first and last weeks dropped.
- **B. Features:** for each week:
  - `share_<category>` for the 8 default expense categories, in the fixed order `Food & Drinks, Coffee & Snacks, Groceries, Transportation, Bills & Utilities, Shopping, Healthcare, Entertainment` = category total ÷ week total
  - `total_ratio` = week total ÷ median of the previous weeks' totals, up to 8 of them (empty weeks count as 0; `shift().rolling(8, min_periods=4).median()`)
  - Drop weeks with fewer than 4 earlier weeks, a week total of 0, or a median of 0. A given check recomputes one row from raw data (no look-ahead).
- **C. Scaling:** `StandardScaler` on the 9 features. Prompt: why K-means needs scaling.
- **D. Choosing k:** inertia (elbow) and silhouette for k = 2…8, `KMeans(n_init=10, random_state=0)`. The user picks k and writes why.
- **E. Read the clusters:** mean unscaled features per cluster and the 3 weeks nearest each centre. The user fills `TYPES = [{"name", "description", "pose"}]`, one per cluster in cluster order; `pose` is `happy`, `thinking` or `celebrating`. Names end in "week" (e.g. "Quiet week"); a description is one short sentence.
- **F. PCA:** 2 components on the scaled features; scatter coloured by cluster with the centres marked; explained variance of PC1 and PC2; a loadings table. Prompt: what does each component mean? PCA is for the picture only; K-means uses all 9 features.
- **G. Export (given cell):** writes `models/week_types.json` (not committed; `models/` is ignored), which the user copies into the app:
  ```json
  { "features": ["share_Food & Drinks", "...", "total_ratio"],
    "mean": [...], "scale": [...],
    "centers": [[...], ...],
    "types": [{ "id": "t0", "name": "...", "description": "...", "pose": "happy" }],
    "samples": [{ "features": [...], "type": "t2" }, ...] }
  ```
  `centers` are in scaled space. `samples` are 3 real weeks (one from each of 3 different clusters) with their unscaled features and the notebook's cluster, for the app's parity check.

## Part 2: App (`budgettracker/mobile`)

### Files

- `src/lib/week-types.json`: the notebook's export, copied as is.
- `src/lib/week-types.ts`: pure, type-only imports plus the JSON (so `scripts/check-week-types.mjs` runs it with Node):
  - `type WeekType = { id: string; name: string; description: string; pose: OlliePose }`
  - `weekFeatures(transactions: Transaction[], categoryNames: Map<string, string>, weekStart: Date, now: Date): number[] | null`: expenses in [weekStart, weekStart + 7 days) up to `now`, local dates. Shares are over **all** expenses that week (custom categories count toward the total but have no share). For the current week the total is scaled by 7 ÷ days so far (Monday = 1 … Sunday = 7). `total_ratio` uses the median of the weeks before `weekStart`, up to 8, counted from the week of the user's first expense, with empty weeks as 0 (same as the notebook). Returns `null` with fewer than 4 such earlier weeks, a week total of 0, or a median of 0.
  - `classify(features: number[]): WeekType`: scale with `mean`/`scale`, nearest centre by squared Euclidean distance.
  - `weekTypes(transactions, categories, now): { current: WeekType | null; recent: { weekStart: Date; type: WeekType | null }[] }`. `current` only from Wednesday (Monday and Tuesday give `null`). `recent` = the 8 completed weeks before this one, oldest first.
- **Reports** (`ReportsScreen.tsx`): a "Your weeks" card before "Next month":
  - "This week so far: **{name}**" and the description; on Monday and Tuesday, "Last week: **{name}**" from the latest `recent` entry instead
  - a row of 8 chips (Monday's date, "Oct 6"), coloured per type, grey when `type` is null, plus a legend of the types shown
  - type colours: fixed theme tokens by type index (`primary`, `secondary`, `tertiary`, `error`, then repeat)
  - hidden when no week has a type
- **Ollie** (`ollie-messages.ts`): new input `weekType: WeekType | null`; message id `week-type:<id>`, pose from the type, not dismissible, text `So far this looks like a {name}. {description}`. Order: over budget → unusual purchase → unusual week → pace → near budget → **week type** → on-track greeting → default greeting.
- **OllieCard:** passes `weekTypes(...).current` when `settings.weekTypeGreeting` is on, otherwise `null`.
- **Setting:** `weekTypeGreeting: boolean` (default `true`) in `types/index.ts` and `settings-repo.ts`; a "Week type from Ollie" switch on AccountScreen next to "Unusual spending alerts".

### Tests

- `scripts/check-week-types.mjs`:
  - parity: every `samples` entry classifies to its `type`
  - partial-week scaling (₱300 by Wednesday counts as ₱700)
  - Wednesday rule (`current` is null on Monday and Tuesday)
  - `null` for fewer than 4 earlier weeks since the first expense, a 0 total, or a 0 median
  - a custom category raises the total but has no share
  - a week from Mon Dec 29 to Sun Jan 4 (year boundary)
  - `recent` has 8 entries, oldest first
- `scripts/check-ollie.mjs`: exact text and pose; place in the order; absent when `weekType` is null
- typecheck and lint pass

**Device checklist** (test account with `fake_transactions_3y.csv` imported):
- [x] Reports shows "Your weeks" with this week's type and 8 coloured chips
- [x] On Monday or Tuesday (or with the phone's date changed) the card says "Last week: …"
- [x] With no alerts, Ollie shows the week-type line with the right pose
- [x] "Week type from Ollie" off → no Ollie line; the Reports card stays
- [x] Airplane mode → everything still works
- [x] Light and dark themes look right

## Build order

1. Notebook scaffold; the user completes it and exports `week_types.json`
2. App: `week-types.ts` + self-check (built against a scratch export, then re-checked with the user's file)
3. Reports card, Ollie greeting, setting
4. Device checklist; mark the roadmap (Phase 4 ✅)

## Out of scope

- Per-user clustering, retraining on the phone, or a server endpoint
- Notifications about week types
- Clustering days or months
