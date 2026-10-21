# Week 3 Categorizer Deployment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve the Week 2 categorizer from a FastAPI service on Vercel and show a tap-to-accept category suggestion in Spendify's transaction form.

**Architecture:** A small FastAPI app in `machine-learning/service/` loads the committed `categorizer.joblib` once and exposes `POST /categorize` and `GET /health`. Docker runs it locally; Vercel runs it in production. The Spendify app calls it through a fail-silent client modelled on `app-update.ts` and shows a chip only when no category is chosen.

**Tech Stack:** Python 3.14, FastAPI, scikit-learn 1.9.1, pytest, Docker, Vercel; Expo 57 / React Native / TypeScript.

**Spec:** `docs/superpowers/specs/2026-10-21-week3-categorizer-deployment-design.md`

## Global Constraints

- `scikit-learn==1.9.1` exactly (the version that trained the model). All service dependencies pinned with `==`.
- Confidence threshold `THRESHOLD = 0.5`. Description 1–200 characters after trimming. `type` is `"expense"` or `"income"`.
- Income categories: `Salary`, `Freelance / Client`, `Other Income`. All others are expense categories.
- Send only description and type. Never log descriptions.
- App client timeout 8 s; debounce 600 ms; minimum trimmed description length 3.
- Env var name: `EXPO_PUBLIC_CATEGORIZER_URL`. Setting key: `smartCategorySuggestions`, default `true`.
- Copy: switch title "Smart category suggestions", subtitle "Sends only the description to suggest a category"; chip "✨ Suggested: <name>".
- Commits: plain messages, **no AI attribution lines**. Never commit `data/`, `models/`, `CLAUDE.md`, `.claude/`.
- **Only commit in `machine-learning`.** In `budgettracker`, the touched files are untracked or already carry the owner's uncommitted edits: make the changes, do not stage or commit; the owner commits.
- Python commands run from `machine-learning/` with `.venv/Scripts/python`.

## Review Focus

1. Description that cleans to empty (`"12345"`, `"*1234"`, `"makati"`) → server returns `category: null`, not 500. Test in Task 2.
2. Non-ASCII text (`"Kape ☕ sa kanto"`, `"Ñaw"`) → 200 response, no crash. Test in Task 2.
3. User picks a category, or switches expense↔income, while a request is in flight → no chip appears for the old state. Manual check in Task 7.
4. Suggested name differs from the user's category only by case/whitespace (`"food & drinks "`) → still matches. Manual check in Task 7.
5. Server returns a category the user deleted → no chip, no crash. Manual check in Task 7.

---

### Task 1: Text cleaning module

**Files:**
- Create: `service/clean.py`
- Test: `service/tests/test_clean.py`

**Interfaces:**
- Produces: `clean_description(text: str) -> str`, `BRANCHES: list[str]`

- [ ] **Step 1: Install dev tools into the existing venv**

Run: `.venv/Scripts/python -m pip install fastapi httpx pytest`
Expected: installs without errors.

- [ ] **Step 2: Write the failing test**

```python
from clean import clean_description

def test_matches_notebook_cleaning():
    assert clean_description('JOLLIBEE*4821') == 'jollibee'
    assert clean_description('Jollibee Makati') == 'jollibee'
    assert clean_description('  Starbucks ') == 'starbucks'
```

- [ ] **Step 3: Run it and confirm it fails**

Run: `cd service && ../.venv/Scripts/python -m pytest tests/test_clean.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'clean'`.

- [ ] **Step 4: Implement `clean_description` in `service/clean.py`**

Copy the logic from `notebooks/02-categorizer.ipynb` cell B1 exactly (lowercase → `re.sub(r'\*?\d+', '', text)` → remove each of `BRANCHES = ['sm north', 'megamall', 'bgc', 'makati', 'qc', 'cubao', 'ortigas']` → strip). Add a one-line comment: must stay identical to the notebook (training/serving skew). Add `service/tests/conftest.py` that inserts the `service/` dir on `sys.path` if imports fail from `tests/`.

- [ ] **Step 5: Run it and confirm it passes**

Run: `cd service && ../.venv/Scripts/python -m pytest tests/test_clean.py -v`
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
git add service/clean.py service/tests/
git commit -m "Add description cleaning for categorizer service"
```

### Task 2: Categorize API

**Files:**
- Create: `service/app.py`, `service/requirements.txt`, `service/model/categorizer.joblib` (copied from `models/`)
- Test: `service/tests/test_app.py`

**Interfaces:**
- Consumes: `clean_description` (Task 1)
- Produces: FastAPI instance `app` in `service/app.py`; `POST /categorize` body `{description: str, type: "expense"|"income"}` → `{category: str | null, confidence: float}`; `GET /health` → `{status: "ok", model_version: "2026-10-21"}`

- [ ] **Step 1: Copy the released model**

Run: `mkdir -p service/model && cp models/categorizer.joblib service/model/`
Expected: `service/model/categorizer.joblib` exists (~140 KB).

- [ ] **Step 2: Write the failing tests**

```python
from fastapi.testclient import TestClient
from app import app, INCOME

client = TestClient(app)
APP_DEFAULTS = {'Food & Drinks', 'Transportation', 'Bills & Utilities', 'Shopping', 'Groceries',
                'Coffee & Snacks', 'Healthcare', 'Entertainment', 'Salary', 'Freelance / Client', 'Other Income'}

def post(description, type='expense'):
    return client.post('/categorize', json={'description': description, 'type': type})

def test_jollibee_is_food():
    r = post('Jollibee'); assert r.status_code == 200
    assert r.json()['category'] == 'Food & Drinks'

def test_payroll_income_is_salary():
    assert post('Payroll', 'income').json()['category'] == 'Salary'

def test_unknown_store_has_no_suggestion():
    assert post('UNKNOWN STORE').json()['category'] is None

def test_expense_never_gets_income_category():
    for text in ['Payroll', 'Upwork', 'Gift', 'Refund']:
        assert post(text).json()['category'] not in INCOME

def test_rejects_empty_and_too_long():
    assert post('   ').status_code == 422
    assert post('x' * 201).status_code == 422
    assert post('x' * 200).status_code == 200

def test_text_that_cleans_to_empty_returns_null():          # Review Focus 1
    for text in ['12345', '*1234', 'makati']:
        r = post(text); assert r.status_code == 200
        assert r.json()['category'] is None

def test_non_ascii_text_does_not_crash():                    # Review Focus 2
    for text in ['Kape ☕ sa kanto', 'Ñaw']:
        assert post(text).status_code == 200

def test_rejects_unknown_type():
    assert post('Jollibee', 'transfer').status_code == 422

def test_health():
    assert client.get('/health').json() == {'status': 'ok', 'model_version': '2026-10-21'}

def test_model_classes_match_app_defaults():
    from app import model
    assert set(model.classes_) <= APP_DEFAULTS
```

- [ ] **Step 3: Run and confirm they fail**

Run: `cd service && ../.venv/Scripts/python -m pytest tests/test_app.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'app'`.

- [ ] **Step 4: Implement `service/app.py`**

- Constants: `THRESHOLD = 0.5`, `MODEL_VERSION = "2026-10-21"`, `INCOME = {"Salary", "Freelance / Client", "Other Income"}`.
- Load once at import: `model = joblib.load(Path(__file__).parent / "model" / "categorizer.joblib")`.
- Request model (pydantic v2): `description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]`; `type: Literal["expense", "income"]`.
- If `clean_description(...)` is empty → `{"category": None, "confidence": 0.0}`.
- Type filter algorithm:

```python
probs = model.predict_proba([cleaned])[0]
keep = np.array([(c in INCOME) == (req.type == "income") for c in model.classes_])
probs = np.where(keep, probs, 0.0)
probs = probs / probs.sum()
best = int(probs.argmax()); confidence = float(probs[best])
category = str(model.classes_[best]) if confidence >= THRESHOLD else None
```

- No logging of the description.

- [ ] **Step 5: Write `service/requirements.txt` with exact pins**

Run: `.venv/Scripts/python -m pip show fastapi scikit-learn joblib numpy scipy | grep -E "^(Name|Version)"`
Write each as `name==version`; `scikit-learn==1.9.1` must match. Do not include pytest/httpx.

- [ ] **Step 6: Run all tests and confirm they pass**

Run: `cd service && ../.venv/Scripts/python -m pytest -v`
Expected: all passed (11 tests).

- [ ] **Step 7: Run locally and try it**

Run: `cd service && ../.venv/Scripts/python -m uvicorn app:app --port 8000` (install `uvicorn` into the venv if missing), then in another shell:
`curl -s -X POST localhost:8000/categorize -H "Content-Type: application/json" -d '{"description":"JOLLIBEE*4821","type":"expense"}'`
Expected: `{"category":"Food & Drinks","confidence":0.8...}`

- [ ] **Step 8: Commit**

```bash
git add service/app.py service/requirements.txt service/model/categorizer.joblib service/tests/test_app.py
git commit -m "Add categorize API with confidence threshold and type filter"
```

### Task 3: Docker

**Files:**
- Create: `service/Dockerfile`, `service/.dockerignore`

**Interfaces:**
- Consumes: `service/app.py`, `service/requirements.txt` (Task 2)

- [ ] **Step 1: Check Docker is installed and running**

Run: `docker version`
Expected: shows Client and Server versions. If not installed, stop and ask the owner to install Docker Desktop.

- [ ] **Step 2: Write `service/Dockerfile`**

Base `python:3.14-slim`; `WORKDIR /app`; install `requirements.txt` plus `uvicorn` and (for the test run) `pytest httpx`; copy the service; `CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]`. `.dockerignore`: `__pycache__/`, `.pytest_cache/`.

- [ ] **Step 3: Build**

Run: `docker build -t spendify-categorizer service`
Expected: `naming to docker.io/library/spendify-categorizer` with no errors.

- [ ] **Step 4: Run tests in the container**

Run: `docker run --rm spendify-categorizer python -m pytest -q`
Expected: `11 passed`.

- [ ] **Step 5: Run the server in the container and call it**

Run: `docker run --rm -d -p 8000:8000 --name cat spendify-categorizer`, then the same `curl` as Task 2 Step 7, then `docker stop cat`.
Expected: `"category":"Food & Drinks"`.

- [ ] **Step 6: Commit**

```bash
git add service/Dockerfile service/.dockerignore
git commit -m "Add Dockerfile for local categorizer runs"
```

### Task 4: Deploy to Vercel and pass the wake-up gate

**Files:**
- Modify: `ml-roadmap.md` (only if the gate passes)

**Interfaces:**
- Produces: production URL `https://<project>.vercel.app` used by Task 6 as `EXPO_PUBLIC_CATEGORIZER_URL`.

- [ ] **Step 1: Push `main` to GitHub**

Run: `git push origin main`
Expected: push succeeds.

- [ ] **Step 2: Owner creates the Vercel project (dashboard, needs their login)**

New Project → import `ronmarabalos/Machine-Learning` → **Root Directory `service`** → Deploy. Owner pastes the production URL.

- [ ] **Step 3: Check the live service**

Run: `curl -s https://<url>/health` and the Task 2 `curl` against `https://<url>/categorize`
Expected: `{"status":"ok","model_version":"2026-10-21"}` and `"category":"Food & Drinks"`. If the build log shows a Python version without a scikit-learn 1.9.1 wheel, fix the Python version before continuing.

- [ ] **Step 4: Wake-up gate**

Leave the service unused for 30+ minutes, then run: `curl -s -o /dev/null -w "%{time_total}\n" https://<url>/health`
Expected: under `8` seconds. **If 8 or more: stop. Report to the owner and revisit hosting before Task 5.**

- [ ] **Step 5: Mark roadmap progress (per CLAUDE.md rules)**

In `ml-roadmap.md` Phase 3, check `FastAPI: a /categorize endpoint`, `Docker: containerize the API`, `Deploy it (...)` with date and proof (`service/`, the URL). Commit:

```bash
git add ml-roadmap.md
git commit -m "Mark Phase 3 service and deploy items done"
git push origin main
```

### Task 5: Settings switch (budgettracker, do not commit)

**Files:**
- Modify: `budgettracker/mobile/src/types/index.ts:164-168` (`UserSettings`)
- Modify: `budgettracker/mobile/src/repositories/settings-repo.ts:4-8` (`DEFAULT_USER_SETTINGS`)
- Modify: `budgettracker/mobile/src/features/account/AccountScreen.tsx:247-259` (after the "Budget limit warnings" row)

**Interfaces:**
- Produces: `UserSettings.smartCategorySuggestions: boolean`, read via `useAppStore((s) => s.settings.smartCategorySuggestions)`

- [ ] **Step 1: Add the field and default**

`smartCategorySuggestions: boolean` in `UserSettings`; `smartCategorySuggestions: true` in `DEFAULT_USER_SETTINGS`.

- [ ] **Step 2: Add the switch row**

Copy the "Budget limit warnings" `SettingsRow` directly below it; title and subtitle from Global Constraints; `onValueChange={(smartCategorySuggestions) => updateSettings({ smartCategorySuggestions })}`; pick an existing icon name already used in the app (e.g. `category`).

- [ ] **Step 3: Typecheck**

Run: `cd "D:/Web Projects/budgettracker/mobile" && npm run typecheck`
Expected: no errors. If `backup-repo.ts:92` builds a `UserSettings` literal and now errors, add `smartCategorySuggestions` there from the parsed value with default `true`.

### Task 6: Suggestion client (budgettracker, do not commit)

**Files:**
- Create: `budgettracker/mobile/src/services/category-suggest.ts`
- Modify: `budgettracker/mobile/.env.example` (add `EXPO_PUBLIC_CATEGORIZER_URL=`), `budgettracker/mobile/.env.local` (add the Task 4 URL)

**Interfaces:**
- Produces: `export type CategorySuggestion = { category: string; confidence: number }` and `export async function suggestCategory(description: string, type: TransactionType, signal?: AbortSignal): Promise<CategorySuggestion | null>`

- [ ] **Step 1: Implement**

Follow `src/services/app-update.ts`: `const URL = process.env.EXPO_PUBLIC_CATEGORIZER_URL ?? ''`, empty → `null`; `TIMEOUT_MS = 8000` via its own `AbortController` (also abort when the caller's `signal` aborts); `POST ${URL}/categorize` with `{ description, type }`; return `null` on non-OK, thrown error, or a body where `category` is not a non-empty string or `confidence` is not a number. Header doc comment: fail-silent, must never block saving.

- [ ] **Step 2: Typecheck and lint**

Run: `npm run typecheck && npm run lint`
Expected: no errors.

### Task 7: Suggestion chip in the form (budgettracker, do not commit)

**Files:**
- Modify: `budgettracker/mobile/src/features/transactions/TransactionFormScreen.tsx` (state near lines 93-112; chip under the Category picker near line 191)

**Interfaces:**
- Consumes: `suggestCategory` (Task 6), `smartCategorySuggestions` (Task 5), `useOnline()` from `@/hooks/use-online`

- [ ] **Step 1: Fetch the suggestion**

`const [suggestion, setSuggestion] = useState<Category | null>(null)`. A `useEffect` on `[description, type, category, enabled, online]`: clear `suggestion`; return early unless `enabled && online && !category && description.trim().length >= 3`; after 600 ms call `suggestCategory(description.trim(), type, controller.signal)`; map the result with `typeCategories.find((c) => c.name.trim().toLowerCase() === result.category.trim().toLowerCase())`; set it only if found. Cleanup clears the timer and aborts the controller, so stale responses never land.

- [ ] **Step 2: Render the chip**

Under the Category picker, when `suggestion && !category`: a `Pressable` with text `✨ Suggested: {suggestion.name}` and `accessibilityLabel` of the same text; `onPress={() => setCategoryId(suggestion.id)}`. Style it like the existing "New category" pressable (line ~201).

- [ ] **Step 3: Typecheck and lint**

Run: `npm run typecheck && npm run lint`
Expected: no errors.

- [ ] **Step 4: Manual checks on device/emulator** (owner runs `npx expo start`)

- [ ] "Jollibee" → chip → tap → category selected, chip gone
- [ ] Choose a category first, then type → no chip
- [ ] Type "Jollibee", then quickly pick a category or switch to Income before the chip shows → no stale chip (Review Focus 3)
- [ ] Rename "Food & Drinks" to "food & drinks " → "Jollibee" still shows the chip (Review Focus 4)
- [ ] Delete "Healthcare", type "Mercury Drug" → no chip, no crash (Review Focus 5)
- [ ] Offline → no chip, saving works
- [ ] Setting off → no chip
- [ ] "asdfgh" → no chip
- [ ] Typing fast and changing text → chip matches the final text

### Task 8: Privacy page and wrap-up

**Files:**
- Modify: `budgettracker/website/src/pages/Privacy.tsx` (new `<H2>Smart category suggestions</H2>` section after "Cloud Sync (Google accounts)", ~line 88) — do not commit
- Modify: `ml-roadmap.md` (Phase 3)

- [ ] **Step 1: Add the section**

`<H2>Smart category suggestions</H2>` + `<p>`: When Smart category suggestions is on (it is on by default), the app sends only the description you type, never the amount, date, account, or your identity, to our server to suggest a category. The description is not stored. You can turn it off under Account.

- [ ] **Step 2: Build the website**

Run: `cd "D:/Web Projects/budgettracker/website" && npm run build`
Expected: build succeeds.

- [ ] **Step 3: Mark the roadmap (per CLAUDE.md rules)**

After the owner confirms the Task 7 checklist, check `Call it from the React Native app ...` and the Phase 3 **Ship** line with date and proof; add ` ✅` to the Phase 3 heading. Commit and push in `machine-learning` only:

```bash
git add ml-roadmap.md
git commit -m "Mark Phase 3 complete"
git push origin main
```

- [ ] **Step 4: Tell the owner what to commit in budgettracker**

List every budgettracker file changed in Tasks 5–8, and remind them to rebuild the APK so `EXPO_PUBLIC_CATEGORIZER_URL` is baked in.
