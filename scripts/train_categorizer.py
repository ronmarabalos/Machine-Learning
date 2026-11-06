"""Train the transaction categorizer (same pipeline as notebooks/02-categorizer.ipynb).

Usage: python scripts/train_categorizer.py [--min-accuracy 0.9]
Reads data/fake_transactions.csv, writes service/model/categorizer.joblib.
Exits with an error (and saves nothing) if test accuracy is below --min-accuracy.
"""

import argparse
import csv
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "service"))
from clean import clean_description  # noqa: E402  same cleaning the app uses

parser = argparse.ArgumentParser()
parser.add_argument("--min-accuracy", type=float, default=0.9)
min_accuracy = parser.parse_args().min_accuracy

with (ROOT / "data" / "fake_transactions.csv").open(encoding="utf-8") as f:
    rows = [(clean_description(r["Description"] or ""), r["Category"]) for r in csv.DictReader(f)]
X, y = zip(*[(text, cat) for text, cat in rows if text])

def new_pipeline():
    return make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4)),
        LogisticRegression(max_iter=1000),
    )

# Quality gate: score on a held-out 20% before trusting the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
accuracy = new_pipeline().fit(X_train, y_train).score(X_test, y_test)
print(f"Test accuracy: {accuracy:.1%} on {len(X_test)} transactions (minimum {min_accuracy:.0%})")
if accuracy < min_accuracy:
    sys.exit(f"Accuracy below {min_accuracy:.0%}, not saving the model.")

# Passed: retrain on all data and save where the app loads it
out = ROOT / "service" / "model" / "categorizer.joblib"
joblib.dump(new_pipeline().fit(X, y), out)
print(f"Saved {out.relative_to(ROOT)} (trained on {len(X)} transactions)")
