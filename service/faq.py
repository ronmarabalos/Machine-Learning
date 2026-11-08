"""Finance tips: find the stored question closest to what the user asked.

faq.csv has one row per topic: several ways to ask (split by `|`) and one answer.
TF-IDF on character n-grams handles typos and Taglish; cosine similarity picks the closest phrasing.
"""
import csv
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

MIN_SCORE = 0.61  # tests/test_faq.py: held-out wordings scored 0.62+, off-topic up to 0.60; tight, so add wordings when users miss

with open(Path(__file__).parent / 'faq.csv', encoding='utf-8') as f:
    TOPICS = list(csv.DictReader(f))

PHRASES = [(i, q.strip().lower()) for i, t in enumerate(TOPICS) for q in t['questions'].split('|')]

# Fit once at import: ~200 short phrases take milliseconds
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True)
matrix = vectorizer.fit_transform([q for _, q in PHRASES])


def best_match(question):
    """(topic row, phrase, score) of the closest stored phrase. Rows are L2-normalised, so dot = cosine."""
    scores = (matrix @ vectorizer.transform([question.lower()]).T).toarray().ravel()
    best = int(scores.argmax())
    topic, phrase = PHRASES[best]
    return TOPICS[topic], phrase, float(scores[best])


def match(question, lang='en'):
    """{'topic', 'question', 'answer'} for a close enough match, else None. lang 'tl' gives the Taglish answer."""
    topic, phrase, score = best_match(question)
    if score < MIN_SCORE:
        return None
    return {'topic': topic['topic'], 'question': phrase, 'answer': topic['answer_tl' if lang == 'tl' else 'answer']}
