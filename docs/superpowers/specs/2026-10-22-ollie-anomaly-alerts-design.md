# Phase 4A: Ollie the companion + unusual spending alerts

**Date:** 2026-10-22
**Roadmap:** Phase 4, anomaly detection (`ml-roadmap.md`). Forecasting (Phase 4B) is a separate, later spec.
**Repos:** `machine-learning` (notebook, generator) and `budgettracker/mobile` (app)

## Goal

Ollie the Otter, Spendify's companion, sits on the Home screen and tells the user one thing at a time: unusual spending, budget warnings, or encouragement. Unusual spending is detected on the phone with a simple rule that a notebook proves against real ML methods.

**Success criteria**
- The notebook scores ratio-to-median, robust z-score and Isolation Forest on labelled data and records the final rule numbers.
- A recent ₱4,800 Shopping purchase against a ₱1,200 usual makes Ollie (thinking) say it is about 4× your usual.
- Over-budget, unusual purchase, unusual week, near-budget and greeting messages appear one at a time in that order; "Got it" moves to the next.
- Everything works offline; no transaction data leaves the phone for this feature.
- The save toast no longer shows budget warnings.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Where detection runs | On the phone, plain TypeScript | Keeps the local-first promise; works offline |
| Method | Ratio to median of earlier purchases (approach A) | The message is the math ("about 4× your usual"); notebook proves it against Isolation Forest |
| What is unusual | Single transactions **and** weekly category totals | User choice |
| Defaults (confirmed or tuned by the notebook) | multiplier 3×, minimum ₱500, ≥5 earlier purchases, ≥4 earlier weeks | Avoid false alarms on cheap items and new users |
| Expenses only | Income is never flagged | Unusual income is not a warning |
| Delivery | Ollie on Home, one message at a time, "Got it" to dismiss | User choice |
| Budget toast | Removed; budget warnings move to Ollie | One voice for alerts |
| Character | Ollie the Otter, user's own art; male (overalls) and female (scarf) versions, both named "Ollie" | User's art; original character |
| Version choice | User picks under Account; **default male** | Respectful; no gender inference |
| Poses | happy (greetings), thinking (alerts), celebrating (good news) | Matches the art |

## Part 1: Data and notebook (`machine-learning`)

### Generator (`scripts/generate_fake_data.py`)

- New column `Anomaly`: `""`, `"spike"`, or `"spree"`. Existing spikes (the 1% ×4–8 amounts) are labelled `"spike"`.
- About 6 spree weeks: one expense category gets 4–6 extra purchases at normal amounts inside one Monday-start week; those rows are `"spree"`. Sprees go only into categories averaging ≤ 2 purchases a week (Groceries, Shopping, Entertainment), where 4–6 extra purchases are clearly unusual; in Food & Drinks (~8 a week) they would be normal.
- Sprees use a **separate** `random.Random` instance, so every existing row is identical to today's output (Week 1–2 results and the deployed model stay valid).
- The app's CSV import ignores unknown columns, so the file stays importable.

### Notebook (`notebooks/04-anomalies.ipynb`, guided TODOs)

- **A. No look-ahead:** for each expense, the "usual" is the median of **earlier** expenses in the same category.
- **B. Single transactions:** ratio-to-median, robust z-score (median + MAD), Isolation Forest. Precision, recall, F1 against `spike`.
- **C. Tuning:** sweep multiplier 2×–5× and minimum amount ₱0–₱1,000.
- **D. Weeks:** weekly category totals vs the median of earlier weeks; same comparison against `spree`.
- **E. Decide:** wrap-up cell states the final multiplier, minimum amount, minimum purchases, minimum weeks. These are copied into the app.
- **Gate:** if the ratio rule's F1 is far below Isolation Forest's, stop and revisit the approach before app work.

## Part 2: Detector (`mobile/src/lib/anomalies.ts`)

Pure functions. Only `import type` from other modules (no runtime imports), so the self-check can run it with Node directly.

```ts
export type UnusualItem = {
  id: string;            // 'tx:<transactionId>' or 'week:<categoryId>:<YYYY-MM-DD of Monday>'
  kind: 'transaction' | 'week';
  categoryId: string;
  amount: number;        // the purchase, or the week's total
  usual: number;         // median it was compared against
  ratio: number;         // amount / usual
  transactionId?: string;
};
export function unusualTransactions(transactions: Transaction[], now: Date): UnusualItem[];
export function unusualWeeks(transactions: Transaction[], now: Date): UnusualItem[];
```

- Constants: `MULTIPLIER`, `MIN_AMOUNT`, `MIN_PURCHASES`, `MIN_WEEKS`, values from the notebook.
- `unusualTransactions`: expenses dated in the last 7 days with a category; usual = median of that category's expenses dated **before** it; flag when ratio ≥ `MULTIPLIER`, amount ≥ `MIN_AMOUNT`, earlier count ≥ `MIN_PURCHASES`.
- `unusualWeeks`: current Monday-start week per category; usual = median of that category's totals for earlier weeks from its first expense week up to last week (weeks with no spending count as 0); flag when ratio ≥ `MULTIPLIER`, total ≥ `MIN_AMOUNT`, earlier weeks ≥ `MIN_WEEKS`, usual > 0.
- Uncategorized expenses and income are ignored.

## Part 3: Messages (`mobile/src/features/companion/ollie-messages.ts`)

Pure. Budget progress is passed in, not imported.

```ts
export type OlliePose = 'happy' | 'thinking' | 'celebrating';
export type OllieMessage = { id: string; pose: OlliePose; text: string; action?: { label: 'View'; href: string } };
export function ollieMessages(input: {
  budgets: BudgetProgress[];          // from budgetProgress(..., monthOf(now))
  unusual: UnusualItem[];             // [] when the setting is off
  categoryNames: Map<string, string>;
  budgetWarnings: boolean;            // settings.budgetLimitWarnings
  dismissed: Set<string>;
  now: Date;
}): OllieMessage[];
```

Order, skipping dismissed ids. Month key `YYYY-MM`. For the overall budget (`categoryId` null) `{name} budget` reads "overall monthly budget", so it never says "Overall monthly budget budget". Money formatted like the rest of the app (`₱ 4,800`), ratio rounded to a whole number.

| # | When | id | Pose | Text | Action |
|---|---|---|---|---|---|
| 1 | budget ≥ 100% (if `budgetWarnings`) | `budget:<categoryId\|overall>:<month>:over` | thinking | "Your {name} budget is at {percent}% this month. Want to review it?" | View → `/budgets` |
| 2 | unusual transaction | its `id` | thinking | "That ₱ {amount} {category} purchase is about {ratio}× your usual. Was it planned?" | View → `/transaction/<id>` |
| 3 | unusual week | its `id` | thinking | "{category} this week is about {ratio}× a usual week. Want to take a look?" | View → `/(tabs)/transactions` |
| 4 | budget 80–99% (if `budgetWarnings`) | `budget:<…>:<month>:near` | thinking | "Heads up: {percent}% of your {name} budget is used." | View → `/budgets` |
| 5 | no alerts, ≥1 budget, all < 80% | `greet:on-track` | celebrating | "All your budgets are on track this month. Nice work!" | — |
| 6 | otherwise | `greet:default` | happy | "Ready to log today's spending? Small steps, bigger tomorrow." | — |

Greetings (5, 6) have no "Got it" and are never dismissed. The Home card shows the first message.

## Part 4: Ollie on Home and settings

- `mobile/src/features/companion/OllieCard.tsx`: under the greeting in `src/features/home/HomeScreen.tsx`. Layout: Ollie image left (~150 px tall) over a green band; dark speech bubble with the label "Ollie", the text, **View** (if any) and **Got it** (alerts only). Works in light and dark themes. Image `accessibilityLabel` describes the pose ("Ollie thinking").
- Art: `mobile/assets/companion/ollie-{male|female}-{happy|thinking|celebrating}.png`, resized from the user's files to ~600 px tall (~150 KB each), transparent background. Source files:

| Pose | Male | Female |
|---|---|---|
| happy | Happy Ollie Otter Mascot.png | Joyful Ollie with raised paws.png |
| thinking | Ollie thinking in forest-green overalls-2.png | Thoughtful Ollie the Otter-1.png |
| celebrating | Ollie’s Triumphant Leap-2.png | Ollie’s joyful celebration leap-1.png |

- Dismissed ids: per device and per user in AsyncStorage key `ollie-dismissed:<userId>`, a JSON array capped at the newest 200.
- Settings (`UserSettings` + `DEFAULT_USER_SETTINGS`):
  - `companion: 'male' | 'female'`, default `'male'`. Account → Preferences row "Companion" with a two-option picker styled like `AppearancePicker`.
  - `unusualSpendingAlerts: boolean`, default `true`. Switch row "Unusual spending alerts", subtitle "Ollie points out purchases far above your usual".
- `TransactionFormScreen.tsx`: the save toast becomes only "Recorded ₱ {amount} {category}" / "Transaction updated". Remove the `budgetAlertFor` call.
- Delete `src/features/budgets/lib/budget-alert.ts` (its only caller was the form). Update the sequence diagram in `mobile/docs/ARCHITECTURE.md` (line ~212) to show Ollie on Home instead of the toast.

## Testing

**Notebook (assert cells)**
- Regenerated file without the `Anomaly` column equals the previous output.
- Each usual value uses only earlier rows.
- Label counts: spikes ≈ 1% of expenses; 5–7 spree weeks.

**Self-check script** `mobile/scripts/check-ollie.ts`, run with `node --experimental-strip-types scripts/check-ollie.ts` (Node 22.22, no new dependency):
- ₱4,800 vs usual ₱1,200 (≥5 earlier) → flagged, ratio 4
- ₱90 vs usual ₱30 → not flagged (below ₱500)
- 4 earlier purchases only → not flagged
- a large **later** purchase does not change an earlier purchase's usual
- income never flagged
- spree week flagged; a category with only 3 earlier weeks not flagged
- message order: over budget → unusual transaction → unusual week → near budget → greeting
- dismissed id skipped; same budget alert in the next month appears again
- no alerts and all budgets < 80% → celebrating

**Typecheck and lint:** `npm run typecheck`, `npm run lint` pass.

**Device checklist** (test account with the fake CSV imported):



- [x] Ollie appears under the greeting in the chosen version
- [x] Recent spike → thinking Ollie, "about N× your usual" → View opens the transaction
- [x] Got it → next message; reopening the app does not bring the dismissed one back
- [x] Over-budget message comes first
- [x] Unusual spending alerts off → only budget messages and greetings
- [x] Switching companion changes the art
- [x] No alerts → celebrating or happy greeting
- [x] Saving shows only "Recorded ₱ …"
- [x] Airplane mode → Ollie still works
- [x] Light and dark themes look right

## Build order

1. Generator labels + notebook (owner, guided) → owner confirms final numbers
2. Detector + self-check
3. Messages + self-check
4. Art + OllieCard on Home
5. Settings, toast change, delete `budget-alert.ts`, update ARCHITECTURE.md
6. Device checklist → mark Phase 4 anomaly items

**Commits:** `machine-learning` changes are committed. `budgettracker` changes are left uncommitted for the owner (their files carry unrelated uncommitted work).

## Out of scope

Forecasting (Phase 4B), push notifications, Ollie on other screens, animation, chatting with Ollie, filtered Transactions deep links, K-means/PCA (roadmap items left for a later notebook).
