"""Spending forecasts from daily expense totals per category.

The features match notebooks/05-forecasting.ipynb exactly (tests/test_forecast.py checks this
against a sample the notebook saved). Plain dates and dicts, so serving needs no pandas.
"""
import calendar
from datetime import date
from pathlib import Path

import joblib
import numpy as np

CATEGORIES = ['Food & Drinks', 'Coffee & Snacks', 'Groceries', 'Transportation', 'Bills & Utilities',
              'Shopping', 'Healthcare', 'Entertainment', 'other']
WINDOW_MONTHS = 6   # full months the phone sends before the current one
MIN_MONTHS = 3      # fewer full months of history: no forecast

# Each bundle is {'model': fitted regressor or None (= the baseline won), 'features': [column names]}
EOM = joblib.load(Path(__file__).parent / 'model' / 'forecast_eom.joblib')
NEXT = joblib.load(Path(__file__).parent / 'model' / 'forecast_next.joblib')


def month_start(d: date, back: int = 0) -> date:
    """First day of d's month, `back` months earlier (negative = later)."""
    n = d.year * 12 + d.month - 1 - back
    return date(n // 12, n % 12 + 1, 1)


def days_in(m: date) -> int:
    return calendar.monthrange(m.year, m.month)[1]


def _spent(daily: dict[date, float], start: date, end: date) -> float:
    return sum(v for d, v in daily.items() if start <= d <= end)


def _history(category: str | None, daily: dict[date, float], today: date):
    """(code, full-month totals oldest first from the first month with spending, avg3, avg6), or None."""
    totals = [_spent(daily, m, date(m.year, m.month, days_in(m)))
              for m in (month_start(today, k) for k in range(WINDOW_MONTHS, 0, -1))]
    first = next((i for i, t in enumerate(totals) if t > 0), len(totals))
    hist = totals[first:]
    if len(hist) < MIN_MONTHS or sum(hist[-3:]) <= 0:
        return None
    code = CATEGORIES.index(category) if category in CATEGORIES[:-1] else len(CATEGORIES) - 1
    return code, hist, sum(hist[-3:]) / 3, sum(hist[-6:]) / len(hist[-6:])


def eom_features(category: str | None, daily: dict[date, float], today: date) -> dict | None:
    """End-of-month features at cut-off `today`, plus `spent` and `avg3` for converting back to pesos."""
    h = _history(category, daily, today)
    if h is None:
        return None
    code, hist, avg3, avg6 = h
    this, prev, d = month_start(today), month_start(today, 1), today.day
    dim, last = days_in(this), hist[-1]
    spent = _spent(daily, this, today)
    prev_spent = _spent(daily, prev, date(prev.year, prev.month, min(d, days_in(prev))))
    return {
        'category_code': code, 'day': d, 'days_left': dim - d,
        'weekend_days_left': sum(date(this.year, this.month, k).weekday() >= 5 for k in range(d + 1, dim + 1)),
        'spent_ratio': spent / avg3, 'last_month_ratio': last / avg3, 'avg6_ratio': avg6 / avg3,
        'last_month_share_by_day': prev_spent / last if last > 0 else 0.0,
        'month_of_year': this.month, 'spent': spent, 'avg3': avg3,
    }


def next_features(category: str | None, daily: dict[date, float], today: date) -> dict | None:
    """Next-month features from full months only, plus `avg3`."""
    h = _history(category, daily, today)
    if h is None:
        return None
    code, hist, avg3, avg6 = h
    return {'category_code': code, 'last_month_ratio': hist[-1] / avg3, 'avg6_ratio': avg6 / avg3,
            'month_of_year': month_start(today, -1).month, 'avg3': avg3}


def _ratio(bundle: dict, features: dict) -> float:
    return float(bundle['model'].predict(np.array([[features[f] for f in bundle['features']]], dtype=float))[0])


def predict(today: date, series: list[dict]) -> list[dict]:
    """[{key, spent, end_of_month, next_month}] for series with enough history; `daily` keys are dates."""
    out = []
    for s in series:
        eom = eom_features(s['category'], s['daily'], today)
        if eom is None:
            continue
        nxt = next_features(s['category'], s['daily'], today)
        spent, avg3 = eom['spent'], eom['avg3']
        if EOM['model'] is None:
            end = spent / today.day * days_in(today)
        else:
            end = spent + max(0.0, _ratio(EOM, eom)) * avg3
        following = avg3 if NEXT['model'] is None else max(0.0, _ratio(NEXT, nxt)) * avg3
        out.append({'key': s['key'], 'spent': round(spent, 2), 'end_of_month': round(max(end, spent), 2),
                    'next_month': round(following, 2)})
    return out
