# Week 3: Deploy the categorizer and connect it to Spendify

**Date:** 2026-10-21
**Roadmap:** Phase 3, Deployment (`ml-roadmap.md`)
**Repos:** `machine-learning` (service) and `budgettracker` (mobile app + website)

## Goal

When a user types a transaction description in Spendify, the app suggests a category from the Week 2 model. The user taps the suggestion to accept it.

**Audience:** a portfolio piece now, built so real Spendify users can use it later. Decisions must not need undoing when real users arrive.

**Success criteria**
- Typing "Jollibee" in a new expense shows "Suggested: Food & Drinks" and tapping it selects the category.
- The service runs on Vercel and is reachable while my computer is off.
- First request after 30+ minutes idle answers in under 8 seconds.
- Offline, server down, or feature switched off: no suggestion appears and saving still works.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Who uses it | Portfolio now, real users later | User choice |
| Privacy default | **On by default**, switch to turn off | User choice. Privacy page discloses it. |
| UX | Tap-to-accept chip, only when no category is chosen | User stays in control; never overrides a choice |
| Hosting | **Vercel** (free Hobby, no card) | Card-free, wakes in seconds. Hobby is non-commercial: move to a paid plan if Spendify earns money. |
| Docker | Local development and testing only | Learn Docker; Vercel does not use the Dockerfile |
| Confidence threshold | **0.5** | Held-out test on fake data: chip shown 97%, 100% correct when shown. Revisit with real data. |
| What is sent | Description text and type only | Never amount, date, account, or user ID |

## Architecture

```
Phone: Spendify (budgettracker/mobile)  --HTTPS-->  Vercel: FastAPI + categorizer.joblib
  knows only EXPO_PUBLIC_CATEGORIZER_URL  <--JSON--   (machine-learning/service)
```

The model stays out of the app: it is a Python object, and the phone runs TypeScript.

## Part 1: Service (`machine-learning/service/`)

```
service/
  app.py                     FastAPI app
  clean.py                   clean_description(), same logic as notebooks/02-categorizer.ipynb
  model/categorizer.joblib   committed (~140 KB), the released model
  requirements.txt           exact pins; scikit-learn==1.9.1 (the version that trained the model)
  Dockerfile                 local run and tests
  tests/
```

- `models/` (notebook output) stays gitignored. Releasing a model = copy it into `service/model/` and bump `MODEL_VERSION`.
- The model loads once at startup.

### API

`POST /categorize`

```json
// request
{ "description": "JOLLIBEE*4821", "type": "expense" }
// response, confident
{ "category": "Food & Drinks", "confidence": 0.91 }
// response, below threshold
{ "category": null, "confidence": 0.32 }
```

- `description`: required, 1–200 characters after trimming. `type`: `"expense"` or `"income"`. Invalid input → 422 (FastAPI validation).
- The server runs `clean_description` before predicting, the same as training.
- **Type filter:** income categories are `Salary`, `Freelance / Client`, `Other Income`. Probabilities of categories of the other type are zeroed and the rest renormalised before picking the top one.
- `category` is null when the top filtered probability is below `THRESHOLD = 0.5`.
- No logging of descriptions.

`GET /health` → `{ "status": "ok", "model_version": "2026-10-21" }`

## Part 2: App (`budgettracker/mobile/`)

1. **Setting:** add `smartCategorySuggestions: true` to `UserSettings` and `DEFAULT_USER_SETTINGS` (`src/repositories/settings-repo.ts`). Missing keys fall back to defaults, so existing users get it on. Add a `SettingsRow` + `Switch` under "Budget limit warnings" in `src/features/account/AccountScreen.tsx`:
   title "Smart category suggestions", subtitle "Sends only the description to suggest a category".
2. **Client:** new `src/services/category-suggest.ts`, modelled on `src/services/app-update.ts`:
   - reads `EXPO_PUBLIC_CATEGORIZER_URL`; empty → returns null (feature off)
   - 8 s timeout via `AbortController`
   - validates the response shape; any error, timeout, or bad shape → null
   - `suggestCategory(description: string, type: TransactionType): Promise<{ category: string; confidence: number } | null>`
3. **Chip:** in `src/features/transactions/TransactionFormScreen.tsx`:
   - request 600 ms after the description stops changing, only when: setting on, `useOnline()` true, no category chosen, trimmed description ≥ 3 characters
   - map the returned name to the user's category by `type` + lowercase name (same as CSV import); no match → no chip
   - show "✨ Suggested: <name>" under the Category picker; tap → `setCategoryId`, chip hides
   - ignore responses for text that is no longer current
   - applies in New and Edit, only while category is empty
4. **Env:** add `EXPO_PUBLIC_CATEGORIZER_URL` to the mobile env. `EXPO_PUBLIC_` values are baked in at build time, so the APK must be rebuilt.

## Part 3: Website

`budgettracker/website/src/pages/Privacy.tsx`: add a paragraph. When Smart category suggestions is on, only the transaction description is sent to a server to suggest a category. It is not stored. It can be turned off under Account.

## Deployment

1. Vercel project from the `machine-learning` GitHub repo, Root Directory `service/`.
2. Push to `main` deploys.
3. Put the production URL in the mobile env and rebuild the APK.

## Error handling

| Situation | Server | App |
|---|---|---|
| Low confidence | 200, `category: null` | no chip |
| Empty or >200 chars | 422 | no chip |
| Server down, slow, offline | — | no chip within 8 s; saving unaffected |
| Suggested category deleted/renamed by user | — | no chip |

## Security

- Endpoint is public. No API key: a key inside an APK is extractable, so it would not be secret.
- Protection: Vercel's built-in DDoS mitigation, 200-character cap, nothing stored.
- If abused later: add a Vercel Firewall rate limit.

## Testing

**Service (pytest, `service/tests/`)**
- `clean_description` matches the notebook: `JOLLIBEE*4821` → `jollibee`, `Jollibee Makati` → `jollibee`, `  Starbucks ` → `starbucks`
- `Jollibee` expense → `Food & Drinks`; `Payroll` income → `Salary`
- `UNKNOWN STORE` → `category: null`
- an expense never returns an income category
- empty and 201-character descriptions → 422
- `/health` returns `model_version`
- every model class is one of the app's 11 default category names

**Docker:** `docker build` and `docker run`, `curl /categorize`, tests pass in the container.

**Deployment gate:** live `/health` and `/categorize` work; first request after 30+ minutes idle takes under 8 s. **If not, stop and revisit hosting before any app work.**

**App:** `npm run typecheck` and `npm run lint` pass (no test framework exists; none added). Manual checklist on device:
- [ ] "Jollibee" → chip → tap → category selected
- [ ] category chosen first → no chip
- [ ] offline → no chip, save works
- [ ] setting off → no chip
- [ ] "asdfgh" → no chip
- [ ] typing fast and changing text → chip matches the final text

## Build order

1. Service + tests
2. Docker
3. Deploy + wake-up gate
4. App changes
5. Privacy page

## Out of scope

Learning from user corrections, offline suggestions, suggesting amounts or accounts, auth/API keys, rate limiting, retraining on real data.
