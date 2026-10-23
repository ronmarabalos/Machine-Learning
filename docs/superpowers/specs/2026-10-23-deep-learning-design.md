# Phase 5: Deep learning (NN from scratch, PyTorch, transformer vs scikit-learn)

**Date:** 2026-10-23
**Roadmap:** Phase 5, all items and the Ship line (`ml-roadmap.md`)
**Repo:** `machine-learning` only (no app change, no deploy)

## Goal

Learn neural networks from first principles, then PyTorch. Then answer one honest question with evidence: *would a fine-tuned transformer categorize Spendify transactions better than the Week 2 scikit-learn model, and is it worth it?* The answer becomes a blog post the user writes.

**Success criteria**
- Notebook 07: a NumPy 2-layer network whose hand-written backprop passes a numerical gradient check (relative error < 1e-6) and trains to useful accuracy.
- Notebook 08: the same network in PyTorch gives the same gradients as notebook 07 (to 1e-6). It covers optimizers, GPU timing and overfitting, and has written answers on CNN, RNN and LSTM concepts.
- Notebook 09: three models compared on three test sets with accuracy, macro F1, CPU latency, size and training time, saved to `blog/results.json` and `blog/figures/`.
- `blog/sklearn-vs-transformer.md` written by the user, with every number matching `results.json`.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Scope | Notebooks + blog; the app keeps scikit-learn | PyTorch is too big for the free Vercel tier; explaining that is part of the blog |
| Fundamentals | NumPy from scratch first, then PyTorch | Backprop is a common interview question |
| Contenders | Week 2 TF-IDF + LogReg; frozen `all-MiniLM-L6-v2` embeddings + LogReg; fine-tuned `distilbert-base-uncased` | Shows the full range: no pretraining, pretraining without fine-tuning, fine-tuning |
| Fair test | Seen merchants, unseen merchants (group split) and ~40 new real PH merchants | A random split rewards memorised names; Week 2 is already 96.3% there |
| Hardware | The user's RTX 3060 Laptop GPU (6 GB), CUDA PyTorch | Fine-tuning takes minutes; GPU vs CPU timing is a lesson |
| Blog | Markdown in the repo; the user publishes it externally | Lives in GitHub; the user controls going public |

## Data

- `data/fake_transactions.csv` (1-year file, as in Week 2), Description → Category, both expense and income categories (11), cleaned with `service/clean.py`'s `clean_description`. Rows with an empty cleaned description are dropped.
- **Merchant key** (given cell): each description is mapped to the generator's merchant name (the `EXPENSES`/`BILLS` lists plus the income descriptions) by matching the cleaned merchant name inside the cleaned description. Rows that match nothing are dropped from the group split only.
- `datasets/new_merchants.csv` (committed; not private): about 40 rows `description,category` of real Philippine merchants absent from the fake data, written in the same messy styles (upper case, branch names, `*1234` references). Written by Claude, reviewed by the user before notebook 09 uses it.

## Notebook 07: `notebooks/07-nn-from-scratch.ipynb` (NumPy only, guided TODOs)

Features: TF-IDF (`TfidfVectorizer` as in Week 2) → dense arrays; train/test split as Week 2.

- **A.** One layer `X @ W + b`; ReLU; softmax. Prompt: why a non-linearity (two linear layers collapse into one).
- **B.** Cross-entropy loss. Prompt: why not train on accuracy.
- **C.** Forward pass: input → 64 hidden ReLU → 11 softmax. Initialise with `rng = np.random.default_rng(0)`, `W * sqrt(2 / fan_in)`.
- **D.** Backprop by hand for `W2, b2, W1, b1`. Given hint: d(loss)/d(logits) = `(probs − one_hot) / n`.
- **E.** Gradient check (given cell): central differences with ε = 1e-5 on 20 random weights of each parameter, on a 32-row batch; asserts the relative error is < 1e-6.
- **F.** Mini-batch SGD (batch 64), loss and accuracy per epoch plotted. Prompt: learning rate ×10 and ÷10.

## Notebook 08: `notebooks/08-pytorch.ipynb` (guided TODOs)

- **A.** Tensors and autograd: the 07 network written with tensors and the same initial weights; `loss.backward()` gradients equal the NumPy gradients (given check, 1e-6, float64).
- **B.** `nn.Sequential(nn.Linear, nn.ReLU, nn.Linear)` + `nn.CrossEntropyLoss`. Prompt: why logits, not softmax outputs.
- **C.** SGD vs Adam, loss curves on one plot.
- **D.** CPU vs `cuda` time per epoch. Prompt: why the GPU may not win for a tiny network.
- **E.** Train vs validation curves; dropout and weight decay compared.
- **F.** Concepts (reading + 3 written answers, no code): what a CNN's filters slide over and why that suits images; what an RNN/LSTM keeps from step to step and why LSTMs were invented; what a transformer's attention does instead of reading in order.

## Notebook 09: `notebooks/09-transformer-vs-sklearn.ipynb` (guided TODOs)

- **Splits** (given cells, fixed seeds):
  - *seen*: random 80/20, stratified by category, `random_state=42`
  - *unseen*: `GroupShuffleSplit(test_size=0.2, random_state=42)` by merchant key
  - *new*: `datasets/new_merchants.csv`, test only

  The seen and unseen experiments each train all three models on their own training part. The *new* set is scored with the models from the unseen experiment.
- **Models:**
  1. TF-IDF + `LogisticRegression` with the Week 2 settings
  2. `SentenceTransformer('all-MiniLM-L6-v2').encode(...)` + `LogisticRegression(max_iter=2000)`
  3. `distilbert-base-uncased` + `AutoModelForSequenceClassification` via `Trainer`: 3 epochs, lr 5e-5, batch 32, `max_length=32`, `fp16` on GPU, seed 42
- **Measures** per model per test set: accuracy and macro F1; plus per model: CPU latency (median over 200 single descriptions after warm-up, `torch.set_num_threads(1)`), size on disk (MB), training time (s).
- **Outputs** (given export cell): `blog/results.json` (all numbers), `blog/figures/accuracy.png` (grouped bars: model × test set), `blog/figures/speed-size.png` (latency vs size, log scale). Models are saved under `models/` (not committed).
- **Prompts:** where each model fails on *new* (show its mistakes); is the accuracy gain worth the size and latency for this app?

## Blog: `blog/sklearn-vs-transformer.md`

The user writes it from a Claude outline:
1. The question
2. Setup and why one random split isn't enough
3. Results (table + figures)
4. What surprised you
5. The production view (size, speed, cost, why the app keeps or would switch)
6. What next

800–1,200 words; every number from `results.json`; no hype. Claude reviews it for number accuracy, claims backed by results, and clarity for a recruiter. The README links it; the user may publish it on dev.to, Hashnode or LinkedIn and add that link.

## Dependencies

Root `requirements.txt` only (the service is unchanged): `torch` (CUDA build from the PyTorch index), `transformers`, `accelerate`, `sentence-transformers`. Install first and check `torch.cuda.is_available()`.

## Build order

1. Install the dependencies; check the GPU
2. Notebook 07 scaffold (with a solved scratch run for reference numbers); the user completes it
3. Notebook 08 scaffold, the same way
4. `datasets/new_merchants.csv` (user reviews), notebook 09 scaffold; the user completes it
5. Blog outline; the user writes it; Claude reviews
6. Mark the roadmap (all Phase 5 items, Ship ✅, heading ✅)

## Out of scope

- Serving the transformer (ONNX, Hugging Face Spaces) or changing the app
- Training from scratch, larger models, hyperparameter search beyond the settings above
- Building CNNs, RNNs or LSTMs (concepts only)
