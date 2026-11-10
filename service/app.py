import re
from datetime import date
from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field, StringConstraints, model_validator

import dates
import faq
import forecast
from clean import clean_description

THRESHOLD = 0.5
INTENT_THRESHOLD = 0.35  # 114-row test set: 94% answered, 93.5% of those right (see notebooks/12-intents.ipynb)
DATED_INTENTS = {'spend_total', 'income_total', 'daily_summary', 'weekly_summary', 'list_transactions', 'report'}
# One of these words makes a question Taglish, so Ollie answers in Taglish. Left out on purpose:
# "may" (also the month) and short words that are English too.
# shortcut: a word list, not a language model; add words when Taglish questions get English replies
TAGALOG_WORDS = {
    'ako', 'ang', 'ano', 'anong', 'araw', 'ba', 'binayad', 'binili', 'buod', 'buwan', 'dapat', 'gaano', 'gastos',
    'ginastos', 'hanggang', 'ilan', 'ipakita', 'ipon', 'kahapon', 'kamusta', 'kinita', 'ko', 'kung', 'lahat',
    'linggo', 'listahan', 'magkano', 'makaipon', 'mga', 'mo', 'mula', 'nagastos', 'naman', 'natanggap', 'ng',
    'ngayon', 'ngayong', 'noong', 'nung', 'pakita', 'paano', 'para', 'po', 'pumasok', 'sa', 'sahod', 'sweldo',
    'taon', 'utang',
}
MODEL_VERSION = '2026-10-21'
INCOME = {'Salary', 'Freelance / Client', 'Other Income'}

# Load the saved pipeline once when this module is imported
model = joblib.load(
    Path(__file__).parent / 'model' / 'categorizer.joblib'
)
intent_model = joblib.load(Path(__file__).parent / 'model' / 'intents.joblib')

app = FastAPI()


class CategorizeRequest(BaseModel):
    description: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=200
        )
    ]
    type: Literal['expense', 'income']


@app.post('/categorize')
def categorize(req: CategorizeRequest):
    cleaned = clean_description(req.description)

    if not cleaned:
        return {'category': None, 'confidence': 0.0}

    probs = model.predict_proba([cleaned])[0]

    # Keep only categories matching the requested transaction type
    keep = np.array([
        (category in INCOME) == (req.type == 'income')
        for category in model.classes_
    ])
    probs = np.where(keep, probs, 0.0)

    total = probs.sum()
    if total == 0:
        return {'category': None, 'confidence': 0.0}

    # Rescale the remaining probabilities so they sum to 1
    probs = probs / total

    best_index = int(np.argmax(probs))
    confidence = float(probs[best_index])

    category = (
        str(model.classes_[best_index])
        if confidence >= THRESHOLD
        else None
    )

    return {'category': category, 'confidence': confidence}


class Series(BaseModel):
    key: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    category: Annotated[str, StringConstraints(max_length=40)] | None = None
    daily: dict[date, Annotated[float, Field(ge=0, allow_inf_nan=False)]]


class ForecastRequest(BaseModel):
    today: date
    series: Annotated[list[Series], Field(max_length=50)]

    @model_validator(mode='after')
    def dates_in_window(self):
        start = forecast.month_start(self.today, forecast.WINDOW_MONTHS)
        for s in self.series:
            if any(d < start or d > self.today for d in s.daily):
                raise ValueError(f'daily dates must be between {start} and {self.today}')
        return self


@app.post('/forecast')
def forecast_spending(req: ForecastRequest):
    # Only daily totals per category arrive here; the body is never logged
    series = [{'key': s.key, 'category': s.category, 'daily': s.daily} for s in req.series]
    return {'forecasts': forecast.predict(req.today, series)}


class AskRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
    ]
    today: date  # the phone's date, so "this week" matches the user's calendar


@app.post('/ask')
def ask(req: AskRequest):
    # Only the question arrives here, never transactions; the phone runs the query itself
    probs = intent_model.predict_proba([req.question.lower()])[0]  # same lowercase as the notebook
    best = int(np.argmax(probs))
    confidence = float(probs[best])
    intent = str(intent_model.classes_[best]) if confidence >= INTENT_THRESHOLD else None

    # Unsure, out_of_scope and finance_tip get no dates, even if the question says "today"
    span = None
    if intent in DATED_INTENTS:
        span = dates.parse_range(req.question, req.today) or dates.default_range(intent, req.today)

    lang = 'tl' if TAGALOG_WORDS & set(re.findall(r'[a-z]+', req.question.lower())) else 'en'

    # Finance tips are general text, so they're answered here; None when no stored question is close
    tip = faq.match(req.question, lang) if intent == 'finance_tip' else None

    return {
        'intent': intent,
        'confidence': confidence,
        'start': span[0] if span else None,
        'end': span[1] if span else None,
        'tip': tip and {'question': tip['question'], 'answer': tip['answer']},
        'lang': lang,
    }


@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': MODEL_VERSION}
