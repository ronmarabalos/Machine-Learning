# Machine Learning Roadmap

**Goal:** ML Engineer job
**Main project:** Add ML features to my Budget Tracker app (React Native + TypeScript)
**Time budget:** 12 hrs/day, 6 days/week (1 rest day to avoid burnout)
**Total:** ~8 weeks, then job applications

## How it fits together

```
React Native app (TypeScript)  --fetch()-->  Python ML API (FastAPI + Docker)  -->  trained models
```

The app stays in TypeScript. All ML lives in a separate Python service the app calls.

---

## Phase 1 — Python data tools (Week 1) ✅
- [x] NumPy (arrays and maths) — 2026-10-21, notebooks/02-categorizer.ipynb (A1)
- [x] Pandas (data manipulation): load, clean, group, merge — 2026-10-21, notebooks/01-explore-spending.ipynb + 02-categorizer.ipynb (A2)
- [x] Matplotlib (plots, graphs) — 2026-10-21, notebooks/01-explore-spending.ipynb
- [x] Export my own transactions as CSV (or use a public/fake dataset) — 2026-10-21, scripts/generate_fake_data.py

**Mini project:** Explore my spending in a Jupyter notebook. Top categories, monthly totals, charts. ✅ 2026-10-21

## Phase 2 — Core ML + Feature #1: Auto-categorize transactions (Week 2) ✅
- [x] scikit-learn basics: train/test split, fit, predict — 2026-10-21, notebooks/02-categorizer.ipynb (B2, B4)
- [x] Text features: TF-IDF on transaction descriptions — 2026-10-21, notebooks/02-categorizer.ipynb (B3)
- [x] Logistic Regression, Decision Trees, Random Forests — 2026-10-21, notebooks/02-categorizer.ipynb (B4, B6)
- [x] **Evaluation:** accuracy, precision/recall, F1, confusion matrix, cross-validation — 2026-10-21, notebooks/02-categorizer.ipynb (B5, B7)
- [x] Overfitting vs underfitting, bias–variance tradeoff — 2026-10-21, notebooks/02-categorizer.ipynb (B6 + shallow tree)

**Ship:** A model that turns "STARBUCKS #123" into "Food", saved to a file. ✅ 2026-10-21

## Phase 3 — Deployment: connect ML to the app (Week 3) ✅
- [x] FastAPI: a `/categorize` endpoint — 2026-10-22, service/app.py (11 tests passing)
- [x] Docker: containerize the API — 2026-10-22, service/Dockerfile (tests pass in container)
- [x] Deploy it (Render, Railway, Fly.io, or similar) — 2026-10-22, Vercel: https://machine-learning-rose.vercel.app (live /health + /categorize verified)
- [x] Call it from the React Native app with `fetch` + a TypeScript type for the response — 2026-10-22, budgettracker/mobile/src/services/category-suggest.ts (typecheck + lint pass, 9-item device checklist passed)

**Ship:** New transactions in the app get auto-categorized. ← This is what makes me an ML *engineer*. ✅ 2026-10-22

## Phase 4 — Feature #2 & #3: Anomalies + Forecasting (Weeks 4–5) ✅
- [x] Unsupervised learning: K-means clustering, PCA — 2026-10-23, notebooks/06-week-types.ipynb (k=5 by elbow + silhouette, PCA plot and loadings), week types on Reports and from Ollie
- [x] Anomaly detection (Isolation Forest): "you spent 3× your normal on X" — 2026-10-23, notebooks/04-anomalies.ipynb (ratio rule vs robust z vs Isolation Forest), Ollie alerts in the app
- [x] Regression + gradient boosting (XGBoost / LightGBM): predict next month's spending — 2026-10-23, notebooks/05-forecasting.ipynb (HistGB + LightGBM vs baselines; end-of-month HistGB ships, next month the 3-month average won the gate)
- [x] Feature engineering: rolling averages, day-of-week, recurring payments — 2026-10-23, notebooks/05-forecasting.ipynb (avg3/avg6, weekend_days_left, last_month_share_by_day)
- [x] Avoid data leakage with time-based splits — 2026-10-23, notebooks/05-forecasting.ipynb (last 6 months as test, look-ahead check), notebooks/04-anomalies.ipynb (earlier-only medians)

**Ship:** `/anomalies` and `/forecast` endpoints + screens in the app. ✅ 2026-10-23 (`/forecast` on Vercel + Budgets, Reports, Ollie; anomalies run on the phone by design, Phase 4A)

## Phase 5 — Deep Learning (Weeks 6–7) ✅
- [x] Neural network basics: neurons, layers, activation functions, loss functions, optimizers — 2026-10-24, notebooks/07-nn-from-scratch.ipynb (NumPy layers, ReLU, softmax, cross-entropy, SGD), notebooks/08-pytorch.ipynb (SGD vs Adam)
- [x] Forward pass / backpropagation — 2026-10-24, notebooks/07-nn-from-scratch.ipynb (backprop by hand, gradient check 1.35e-07), notebooks/08-pytorch.ipynb (autograd matches to ~1e-17)
- [x] PyTorch — 2026-10-24, notebooks/08-pytorch.ipynb (autograd, nn.Module, optimizers, GPU timing, dropout/weight decay, early stopping)
- [x] Transformers + Hugging Face: fine-tune a small model for categorization — 2026-10-24, notebooks/09-transformer-vs-sklearn.ipynb (DistilBERT fine-tuned with Trainer on the RTX 3060)
- [x] Compare it with the Phase 2 model (accuracy vs speed vs cost) — 2026-10-24, notebooks/09-transformer-vs-sklearn.ipynb, blog/results.json (seen/unseen/new merchants, size, CPU latency, training time)
- [x] Know what CNNs / RNNs / LSTMs are (concepts only, no need to build) — 2026-10-24, notebooks/08-pytorch.ipynb section F

**Ship:** Blog post: "scikit-learn vs a fine-tuned transformer for categorizing my transactions" ✅ 2026-10-24 (blog/sklearn-vs-transformer.md)

## Phase 6 — MLOps polish (Week 8)
- [x] Version control for code (Git) and data/models (DVC) — 2026-11-06, models.dvc + datasets.dvc (rollback tested in 5c76d5a)
- [x] CI/CD: GitHub Actions to test + retrain + redeploy — 2026-11-06, .github/workflows/test.yml + retrain.yml (CI run 42244709705 → commit 104debc → Vercel deploy)
- [x] Monitoring for data drift (spending patterns change over time) — 2026-11-06, notebooks/10-drift-monitoring.ipynb
- [ ] One cloud platform basics (AWS or GCP)
- [x] SQL for working with real data — 2026-11-07, notebooks/11-sql.ipynb

**Ship:** Polished GitHub repo + README with architecture diagram → start applying. ✅ 2026-11-07 (README.md, repo public)

## After Week 8 — Job hunt (keep building while applying)
- [ ] Apply to ML Engineer / junior ML roles every day
- [ ] Interview prep: ML fundamentals Q&A, ML system design ("design a fraud detector"), Python coding problems
- [ ] One Kaggle competition for extra portfolio proof
- [ ] Keep improving the Budget Tracker ML features

## Feature #4: Ask Spendif (questions about my money, no LLM)
The server only works out what the question is asking (intent) and the date range. The phone runs the SQL on its own data, so transactions never leave it. Online only.
- [x] Intent training data: 8 intents × 30–50 questions, incl. `out_of_scope` + Taglish (`datasets/intents.csv`) — 2026-11-07, datasets/intents.csv (317 rows, 34–45 per intent, 0 duplicates)
- [ ] Separate test set written later without looking at the training file (`datasets/intents_test.csv`) (left: current 64 rows were written by Claude right after the training file; add 3–5 of my own per intent)
- [ ] Train intent classifier: TF-IDF + LogisticRegression, confusion matrix, pick `THRESHOLD` (`notebooks/12-intents.ipynb`)
- [x] Date parser `parse_range(question, today)` with tests: ranges, this week, last month, today, 10 years ago, no date — 2026-11-07, service/dates.py + service/tests/test_dates.py (49 tests, 81/81 passing)
- [x] `/ask` endpoint: intent + confidence + start/end, `null` when unsure, tests in CI — 2026-11-07, service/app.py + service/tests/test_app.py (CI run 42388595114: 87 passed; live on Vercel)
- [x] Finance tips: `datasets/faq.csv` + TF-IDF matching to the closest answer — 2026-11-08, service/faq.csv (38 topics; in service/ so Vercel ships it) + service/faq.py + service/tests/test_faq.py (124/124 passing; 21/24 new wordings get a tip through /ask, 3 lost to the intent model)
- [ ] App: NetInfo online check → `/ask` → wallet match on phone → SQL → answer

**Mini project:** On my phone, "how much did i spend this week" returns the correct number (checked by hand).

---

## Math — learn it when you need it
- Linear algebra → when doing PCA and neural networks
- Probability & statistics → when doing evaluation metrics and anomaly detection
- Calculus (gradients) → when doing backpropagation in PyTorch

## For every algorithm, answer
1. What problem does it solve?
2. When to use it versus the alternatives?
3. How to evaluate it?

## How to learn effectively
- **70/30 rule:** 70% building, 30% theory and courses
- **Learn in public:** post projects on GitHub, write about what I learned
- **Daily 12 hrs ≈ 8.5 hrs building + 3.5 hrs theory**
  - Morning (3.5 hrs): theory, courses, reading — fresh brain for hard concepts
  - Afternoon + evening (8.5 hrs): build that phase's feature, with breaks
  - Last 15 min: write notes on what I learned (feeds the blog/GitHub)
- **1 rest day per week** — at 12 hrs/day, burnout is the biggest risk to finishing

## Later / optional
- Receipt scanning with a CNN (photo → amount + merchant)
- On-device models in React Native (ONNX / TensorFlow Lite)
