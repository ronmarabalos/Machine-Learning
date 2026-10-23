# TITLE: TODO

<!--
Your post, in your words: 800–1,200 words. Every number comes from blog/results.json (and the notebooks).
Write for a recruiter or engineer who has never seen Spendify: short sentences, no hype, define terms once.
Delete these comments when you're done.
-->

## The question

<!--
- What is Spendify, and what does the categorizer do for the user (one or two sentences)?
- Your Week 2 model was 96% accurate. Why ask whether a transformer would be better at all?
- What would "better" have to mean for this app? (accuracy on what, at what cost?)
-->

## The setup

<!--
- The three contenders, in one line each: TF-IDF + logistic regression, frozen MiniLM embeddings + logistic regression, fine-tuned DistilBERT.
- Why one random train/test split isn't enough here (the same merchant names appear on both sides).
- The three test sets: seen merchants, unseen merchants (held out completely), and 44 real Philippine merchants not in the data.
- How you kept it fair: same rows for every model, CPU timing for all, the leakage checks.
-->

## Results

<!-- Fill the table from blog/results.json (accuracy as %, size in MB, latency in ms on one CPU thread). -->

| Model | Seen | Unseen | New PH merchants | Size | CPU latency |
|---|---|---|---|---|---|
| TF-IDF + LogReg | | | | | |
| MiniLM + LogReg | | | | | |
| DistilBERT (fine-tuned) | | | | | |

![Accuracy by test set](figures/accuracy.png)

![Size vs speed](figures/speed-size.png)

<!--
- What does the table say in one sentence?
- Mention macro F1 where it tells a different story than accuracy.
-->

## What surprised me

<!--
- Which result did you not expect, and what is your best explanation for it? (Use your notebook 09 answers.)
- Show 2–3 concrete mistakes from the new merchants (e.g. a "letters only" one and a "reasonable" one).
- Be honest about what this experiment can't tell you (fake data, ~70 merchants, fixed settings).
-->

## The production view

<!--
- Size, speed and training time, and what each means for a free Vercel server and an instant suggestion chip.
- Your decision for Spendify, and what would have to change for you to switch.
-->

## What I'd try next

<!-- 3–4 bullets: ideas you would actually test next, each with why. -->
