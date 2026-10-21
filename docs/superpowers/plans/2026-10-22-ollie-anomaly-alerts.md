# Ollie + Anomaly Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect unusual spending on the phone with a notebook-proven ratio rule and let Ollie the Otter deliver it, plus budget warnings, on the Home screen.

**Architecture:** The generator labels anomalies; a guided notebook compares ratio-to-median, robust z-score and Isolation Forest and fixes the rule's numbers. In the app, two pure TypeScript modules (detector, message builder) feed an `OllieCard` on Home; settings choose the companion and toggle alerts; the save toast stops showing budget warnings.

**Tech Stack:** Python 3.14, pandas, scikit-learn, Jupyter; Expo 57, React Native, TypeScript, AsyncStorage, Pillow (image resize), Node 22 `--experimental-strip-types` (self-check).

**Spec:** `docs/superpowers/specs/2026-10-22-ollie-anomaly-alerts-design.md`

## Global Constraints

- Detection runs on the phone. No transaction data is sent anywhere for this feature.
- Rule defaults until the notebook says otherwise: `MULTIPLIER = 3`, `MIN_AMOUNT = 500`, `MIN_PURCHASES = 5`, `MIN_WEEKS = 4`. Weeks start Monday. Expenses only; uncategorized ignored.
- "Usual" = median of **earlier** data only (no look-ahead).
- Message order: over budget → unusual transaction → unusual week → near budget (80–99%) → on-track greeting → default greeting. Copy is exactly as in the spec's Part 3 table.
- Settings: `companion: 'male' | 'female'` default `'male'`; `unusualSpendingAlerts: boolean` default `true`.
- Dismissed ids: AsyncStorage key `ollie-dismissed:<userId>`, JSON array, newest 200.
- Art: `mobile/assets/companion/ollie-{male|female}-{happy|thinking|celebrating}.png`, ~600 px tall.
- `anomalies.ts` and `ollie-messages.ts` have **no runtime imports** (only `import type`), so Node can run them.
- Commit only in `machine-learning`, plain messages, no AI attribution; never commit `data/`, `models/`, `.superpowers/`, `CLAUDE.md`. In `budgettracker`, make changes but do not stage or commit.
- Guided mode: the owner writes the notebook TODOs; Claude builds the generator change, the blank notebook, and all app code.

## Review Focus

1. A category whose earlier purchases are all the same amount (median = that amount) and a new purchase exactly 3× → flagged (≥, not >). Self-check in Task 3.
2. A transaction dated in the future or exactly 7 days ago → only the last 7 days, up to `now`, count. Self-check in Task 3.
3. Overall budget (`categoryId` null) over 100% → text reads "Your overall monthly budget is at …". Self-check in Task 4.
4. Two unusual transactions → both appear in turn, the bigger ratio first. Self-check in Task 4.
5. AsyncStorage empty, corrupt JSON, or throwing → no crash, nothing treated as dismissed. Manual check in Task 6.

---

### Task 1: Label anomalies in the generator

**Files:**
- Modify: `scripts/generate_fake_data.py`

**Interfaces:**
- Produces: `data/fake_transactions.csv` with a new last column `Anomaly` ∈ {`""`, `"spike"`, `"spree"`}; `data/fake_transactions_before.csv` (copy of the old file, for the notebook's equality check).

- [ ] **Step 1: Keep the old output**

Run: `cp data/fake_transactions.csv data/fake_transactions_before.csv`

- [ ] **Step 2: Label spikes without changing randomness**

In `generate()`, record `spike = random.random() < 0.01` (same single call as today) and pass `anomaly="spike" if spike else ""` into `row(...)`; `row()` gains an `anomaly=""` parameter written to `"Anomaly"`. Every other row gets `""`.

- [ ] **Step 3: Add spree weeks after the main loop**

After the daily loop (so the global `random` sequence for existing rows is untouched), use `rng = random.Random(7)`: pick 6 distinct Monday-start weeks inside the 365 days, for each pick a category from `["Groceries", "Shopping", "Entertainment"]`, and add `rng.randint(4, 6)` purchases on random days of that week with `rng.uniform(lo, hi)` amounts from that category's range, description `rng.choice(merchants)`, `anomaly="spree"`. Add `"Anomaly"` to the CSV `fieldnames`.

- [ ] **Step 4: Extend the self-check and run**

Add asserts: spikes are 0.5–2% of expenses; spree rows are 24–36. Run: `.venv/Scripts/python scripts/generate_fake_data.py`
Expected: `Wrote N transactions …` with no assertion error.

- [ ] **Step 5: Prove old rows are unchanged**

Run:
```
.venv/Scripts/python -c "import pandas as pd; a=pd.read_csv('data/fake_transactions_before.csv'); b=pd.read_csv('data/fake_transactions.csv'); b=b[b.Anomaly.fillna('')!='spree'].drop(columns='Anomaly').reset_index(drop=True); pd.testing.assert_frame_equal(a,b); print('unchanged')"
```
Expected: `unchanged`

- [ ] **Step 6: Commit**

```bash
git add scripts/generate_fake_data.py
git commit -m "Label spikes and add spree weeks in fake data generator"
```

### Task 2: Week 4 notebook (Claude builds blank; owner completes)

**Files:**
- Create: `notebooks/04-anomalies.ipynb`

**Interfaces:**
- Consumes: `data/fake_transactions.csv` with `Anomaly` (Task 1)
- Produces: wrap-up cell stating final `MULTIPLIER`, `MIN_AMOUNT`, `MIN_PURCHASES`, `MIN_WEEKS` (used by Task 3)

- [ ] **Step 1: Build the notebook with TODO cells and hints**

Sections from the spec Part 1: setup (load, expenses only, sort by date, assert old rows unchanged vs `fake_transactions_before.csv`); A. `usual` = median of earlier same-category amounts (hint: `groupby('Category')['Amount'].transform(lambda s: s.shift().expanding().median())`) with an assert that the first purchase per category has no usual; B. three methods vs `Anomaly == 'spike'` with `precision_recall_fscore_support` (hints: ratio ≥ 3 & amount ≥ 500 & earlier count ≥ 5; robust z with MAD; `IsolationForest(contamination=0.01, random_state=42)` on `[log(amount), category one-hot]`); C. sweep table multiplier 2–5 × min amount {0, 250, 500, 1000}; D. weekly totals per category (`dt.to_period('W-SUN')`), usual = median of earlier weeks incl. zero weeks, vs weeks containing `spree` rows; E. markdown wrap-up with the four numbers and why.

- [ ] **Step 2: Verify with a scratch solution**

Fill every TODO in a scratchpad copy and run it with `jupyter nbconvert --execute`. Expected: no errors; ratio F1 printed for B and D; the gate in E is decidable. Put only the blank notebook in the repo.

- [ ] **Step 3: Commit the blank notebook**

```bash
git add notebooks/04-anomalies.ipynb
git commit -m "Add Week 4 anomaly detection notebook"
```

- [ ] **Step 4: Owner completes the notebook; Claude reviews**

Per `CLAUDE.md`: read outputs, check asserts pass and numbers are justified. **Gate:** if ratio F1 is far below Isolation Forest (more than 0.15 lower), stop and revisit the approach with the owner. Commit the completed notebook (`git commit -m "Complete Week 4 anomaly notebook"`).

### Task 3: Detector (`budgettracker/mobile`, do not commit)

**Files:**
- Create: `mobile/src/lib/anomalies.ts`
- Test: `mobile/scripts/check-ollie.mjs`

**Interfaces:**
- Produces:
```ts
export const MULTIPLIER: number; export const MIN_AMOUNT: number; export const MIN_PURCHASES: number; export const MIN_WEEKS: number;
export type UnusualItem = { id: string; kind: 'transaction' | 'week'; categoryId: string; amount: number; usual: number; ratio: number; transactionId?: string };
export function unusualTransactions(transactions: Transaction[], now: Date): UnusualItem[];
export function unusualWeeks(transactions: Transaction[], now: Date): UnusualItem[];
```
Ids: `tx:<id>`, `week:<categoryId>:<YYYY-MM-DD of Monday>`. Results sorted by `ratio` descending.

- [ ] **Step 1: Write the failing self-check**

`check-ollie.mjs` uses `node:assert/strict` and imports `../src/lib/anomalies.ts`. Helper `tx(id, amount, daysAgo, categoryId = 'shop', type = 'expense')`. Cases (with `now = new Date('2026-10-22T12:00:00')`, a Thursday):
```js
// 5 earlier ₱1,200, one ₱4,800 today → flagged, ratio 4
assert.equal(unusualTransactions([...five(1200, 10), tx('big', 4800, 0)], now)[0].ratio, 4);
// exactly 3× → flagged (Review Focus 1)
assert.equal(unusualTransactions([...five(1000, 10), tx('x', 3000, 0)], now).length, 1);
// ₱90 vs ₱30 → not flagged (below MIN_AMOUNT)
assert.equal(unusualTransactions([...five(30, 10), tx('c', 90, 0)], now).length, 0);
// only 4 earlier → not flagged
assert.equal(unusualTransactions([...five(1200, 10).slice(1), tx('d', 4800, 0)], now).length, 0);
// later huge purchase does not change an earlier one's usual
assert.equal(unusualTransactions([...five(1200, 20), tx('e', 4800, 3), tx('f', 99999, 1)], now).find(i => i.transactionId === 'e').usual, 1200);
// income never flagged
assert.equal(unusualTransactions([...five(1200, 10, 'shop', 'income'), tx('g', 9000, 0, 'shop', 'income')], now).length, 0);
// older than 7 days or in the future → ignored (Review Focus 2)
assert.equal(unusualTransactions([...five(1000, 30), tx('old', 5000, 8), tx('fut', 5000, -2)], now).length, 0);
// week: 6 earlier weeks of ₱1,000, this week ₱4,000 → flagged; only 3 earlier weeks → not
```
(`five(amount, startDaysAgo, cat, type)` makes 5 transactions on consecutive earlier days; week cases build one ₱1,000 purchase per earlier week.) End with `console.log('check-ollie: all passed')`.

- [ ] **Step 2: Run it and see it fail**

Run: `cd "D:/Web Projects/budgettracker/mobile" && node --experimental-strip-types scripts/check-ollie.mjs`
Expected: `ERR_MODULE_NOT_FOUND` for `anomalies.ts`.

- [ ] **Step 3: Implement `mobile/src/lib/anomalies.ts`**

Only `import type { Transaction } from '@/types'`. Constants from the notebook wrap-up (defaults in Global Constraints). Window: `now − 7 days < date ≤ now`. Median of an even count = mean of the middle two. Weekly: Monday-start weeks in local time; earlier weeks run from the category's first expense week to last week, with zero weeks included; require `usual > 0`.

- [ ] **Step 4: Run it and see it pass**

Same command. Expected: `check-ollie: all passed`.

- [ ] **Step 5: Typecheck**

Run: `npm run typecheck`
Expected: no errors (`.mjs` is outside `tsconfig` include).

### Task 4: Message builder (`budgettracker/mobile`, do not commit)

**Files:**
- Create: `mobile/src/features/companion/ollie-messages.ts`
- Test: `mobile/scripts/check-ollie.mjs` (append)

**Interfaces:**
- Consumes: `UnusualItem` (Task 3), `BudgetProgress` (type from `@/lib/finance`: `{ budget, name, spent, remaining, percent }`)
- Produces:
```ts
export type OlliePose = 'happy' | 'thinking' | 'celebrating';
export type OllieMessage = { id: string; pose: OlliePose; text: string; dismissible: boolean; action?: { label: 'View'; href: string } };
export function ollieMessages(input: { budgets: BudgetProgress[]; unusual: UnusualItem[]; categoryNames: Map<string, string>; budgetWarnings: boolean; dismissed: Set<string>; now: Date }): OllieMessage[];
```

- [ ] **Step 1: Append failing checks**

```js
const b = (categoryId, percent, name = 'Shopping') => ({ budget: { categoryId }, name, spent: 0, remaining: 0, percent });
const names = new Map([['shop', 'Shopping']]);
const base = { budgets: [], unusual: [], categoryNames: names, budgetWarnings: true, dismissed: new Set(), now };
// order
const all = ollieMessages({ ...base, budgets: [b('shop', 112), b('food', 85, 'Food & Drinks')], unusual: [txItem, weekItem] });
assert.deepEqual(all.map(m => m.id.split(':')[0]), ['budget', 'tx', 'week', 'budget', 'greet']);
// overall wording (Review Focus 3)
assert.match(ollieMessages({ ...base, budgets: [b(null, 120, 'Overall monthly budget')] })[0].text, /^Your overall monthly budget is at 120%/);
// two unusual tx, bigger ratio first (Review Focus 4)
// dismissed skipped; next month's id differs so it shows again
// no alerts, all budgets < 80% → celebrating; no budgets → happy
// budgetWarnings false → no budget messages
// transaction text: "That ₱ 4,800 Shopping purchase is about 4× your usual. Was it planned?"
```

- [ ] **Step 2: Run and see it fail**

Run: `node --experimental-strip-types scripts/check-ollie.mjs`
Expected: `ERR_MODULE_NOT_FOUND` for `ollie-messages.ts`.

- [ ] **Step 3: Implement**

Only `import type`. Copy and ids exactly per the spec's Part 3 table; month key from `now` as `YYYY-MM`; amounts `₱ ${Math.round(n).toLocaleString('en-US')}`; ratio `Math.round`. Over/near budgets each sorted by percent descending. Greetings: `dismissible: false`, always last.

- [ ] **Step 4: Run and see it pass; typecheck**

Run: `node --experimental-strip-types scripts/check-ollie.mjs && npm run typecheck`
Expected: `check-ollie: all passed`, no type errors.

### Task 5: Ollie art (`budgettracker/mobile`, do not commit)

**Files:**
- Create: `mobile/assets/companion/ollie-{male|female}-{happy|thinking|celebrating}.png` (6 files)

- [ ] **Step 1: Resize and rename**

With Pillow (`.venv` in machine-learning, run with `-I`), open each source from `D:\Download\2026 Download\TURNITIN\` per the spec's art table, keep RGBA, resize to 600 px tall (Lanczos), `optimize=True`, save to the target name.

- [ ] **Step 2: Verify**

Check: 6 files, each RGBA, height 600, corner pixel alpha 0, each < 300 KB. View one to confirm quality.

### Task 6: OllieCard on Home (`budgettracker/mobile`, do not commit)

**Files:**
- Create: `mobile/src/features/companion/OllieCard.tsx`, `mobile/src/features/companion/dismissed-store.ts`
- Modify: `mobile/src/features/home/HomeScreen.tsx` (after the greeting block, ~line 139)

**Interfaces:**
- Consumes: `ollieMessages`, `unusualTransactions`, `unusualWeeks`, `budgetProgress`, `monthOf`, settings `companion`, `unusualSpendingAlerts`, `budgetLimitWarnings` (Task 7 adds the two new settings; build Task 7 Step 1 first if typecheck needs them)
- Produces: `export function OllieCard(): JSX.Element` (reads the store itself); `loadDismissed(userId: string): Promise<Set<string>>`, `saveDismissed(userId: string, ids: Set<string>): Promise<void>`

- [ ] **Step 1: Dismissed store**

AsyncStorage key `ollie-dismissed:<userId>`. `loadDismissed` returns an empty set on missing, non-array JSON, or a thrown error (Review Focus 5). `saveDismissed` keeps the newest 200 and swallows errors.

- [ ] **Step 2: OllieCard**

Static `require` map for the 6 images keyed by `${companion}-${pose}`. Compute messages with `useMemo` from store data and `now = new Date()`; `unusual = unusualSpendingAlerts ? [...unusualTransactions(...), ...unusualWeeks(...)] : []`. Show the first message: image (~150 px tall, `accessibilityLabel` "Ollie happy/thinking/celebrating") over a band using the theme's primary colour; bubble with "Ollie" label (primary text), message text, **View** (`router.push(action.href)`) and, if `dismissible`, **Got it** (adds id, saves, re-renders). Use existing `Text`, `Pressable`, theme tokens; no new dependencies.

- [ ] **Step 3: Add to Home**

Render `<OllieCard />` directly after the greeting `View` in `HomeScreen.tsx`.

- [ ] **Step 4: Typecheck and lint**

Run: `npm run typecheck && npx eslint src/features/companion src/features/home/HomeScreen.tsx`
Expected: no errors.

### Task 7: Settings, toast, cleanup (`budgettracker/mobile`, do not commit)

**Files:**
- Modify: `mobile/src/types/index.ts` (`UserSettings`), `mobile/src/repositories/settings-repo.ts` (`DEFAULT_USER_SETTINGS`), `mobile/src/features/account/AccountScreen.tsx` (Preferences group), `mobile/src/features/transactions/TransactionFormScreen.tsx` (submit toast), `mobile/docs/ARCHITECTURE.md` (~line 212)
- Create: `mobile/src/features/account/components/CompanionPicker.tsx`
- Delete: `mobile/src/features/budgets/lib/budget-alert.ts`

- [ ] **Step 1: Settings fields**

`companion: 'male' | 'female'` default `'male'`; `unusualSpendingAlerts: boolean` default `true`.

- [ ] **Step 2: Account rows**

"Companion" `SettingsRow` with `CompanionPicker` (two options Male/Female, modelled on `AppearancePicker.tsx`, calls `updateSettings({ companion })`). "Unusual spending alerts" switch row, subtitle "Ollie points out purchases far above your usual", below "Budget limit warnings".

- [ ] **Step 3: Plain save toast**

In `submit`, toast `isEdit ? 'Transaction updated' : \`Recorded ₱ ${formatNumber(amount)} ${category?.name ?? ''}\`.trim()`; remove the `budgetAlertFor` import and call.

- [ ] **Step 4: Delete and document**

Run: `grep -rn "budgetAlertFor\|budget-alert" src` → only the file itself; delete it. In `ARCHITECTURE.md` replace `UI->>UI: budgetAlertFor(...) → toast` with `UI->>UI: toast "Recorded…"; Home → Ollie shows budget/unusual alerts`.

- [ ] **Step 5: Full checks**

Run: `npm run typecheck && npm run lint && node --experimental-strip-types scripts/check-ollie.mjs`
Expected: no type errors, no new lint errors, `check-ollie: all passed`.

### Task 8: Device checklist and roadmap

**Files:**
- Modify: `ml-roadmap.md` (Phase 4)

- [ ] **Step 1: Owner runs the app** (`npx expo start -c`, test account with `data/fake_transactions.csv` imported) and goes through the spec's 10-item device checklist, including the corrupt-storage case (Review Focus 5) only if reproducible.

- [ ] **Step 2: Mark the roadmap (per `CLAUDE.md`)**

After the owner confirms: check "Anomaly detection (Isolation Forest)…" with date and proof (`notebooks/04-anomalies.ipynb`, the app files). Leave K-means/PCA and forecasting unchecked with a note. Commit and push in `machine-learning`:

```bash
git add ml-roadmap.md
git commit -m "Mark Phase 4 anomaly detection done"
git push origin main
```

- [ ] **Step 3: Tell the owner what to commit in budgettracker**

List every created, modified and deleted file from Tasks 3–7.
