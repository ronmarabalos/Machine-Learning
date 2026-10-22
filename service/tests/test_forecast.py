import calendar
import json
from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import forecast
from app import app
from forecast import eom_features, next_features

client = TestClient(app)
SAMPLE = json.loads((Path(__file__).parent / 'fixtures' / 'forecast_sample.json').read_text())


def as_dates(daily):
    return {date.fromisoformat(k): v for k, v in daily.items()}


def every_day(start, end, amount=100.0):
    """`amount` on every day from start to end, as the JSON `daily` dict."""
    return {str(start + timedelta(days=i)): amount for i in range((end - start).days + 1)}


def post(today, series):
    return client.post('/forecast', json={'today': str(today), 'series': series})


def test_feature_parity():
    today = date.fromisoformat(SAMPLE['request']['today'])
    for s in SAMPLE['request']['series']:
        daily = as_dates(s['daily'])
        eom, nxt = eom_features(s['category'], daily, today), next_features(s['category'], daily, today)
        for name, expected in SAMPLE['eom'][s['key']].items():
            assert eom[name] == pytest.approx(expected, rel=1e-9, abs=1e-9), name
        for name, expected in SAMPLE['next'][s['key']].items():
            assert nxt[name] == pytest.approx(expected, rel=1e-9, abs=1e-9), name


GOOD = {'key': 'k', 'category': 'Food & Drinks', 'daily': {'2026-08-01': 10.0}}


@pytest.mark.parametrize('body', [
    {'today': '2026-13-01', 'series': [GOOD]},
    {'today': '2026-08-15', 'series': [GOOD] * 51},
    {'today': '2026-08-15', 'series': [{**GOOD, 'daily': {'2026-08-16': 10.0}}]},   # after today
    {'today': '2026-08-15', 'series': [{**GOOD, 'daily': {'2026-01-31': 10.0}}]},   # before the window (Feb 1)
    {'today': '2026-08-15', 'series': [{**GOOD, 'daily': {'2026-08-01': -1}}]},
    {'today': '2026-08-15', 'series': [{**GOOD, 'daily': {'2026-08-01': 'nan'}}]},
])
def test_rejects_bad_input(body):
    assert client.post('/forecast', json=body).status_code == 422


def test_short_history_left_out():
    r = post(date(2026, 8, 15), [
        {'key': 'new', 'category': 'Shopping', 'daily': every_day(date(2026, 6, 1), date(2026, 8, 15))},
        {'key': 'old', 'category': 'Shopping', 'daily': every_day(date(2026, 4, 1), date(2026, 8, 15))},
    ])
    assert r.status_code == 200
    assert [f['key'] for f in r.json()['forecasts']] == ['old']


def test_eom_at_least_spent():
    r = client.post('/forecast', json=SAMPLE['request'])
    forecasts = r.json()['forecasts']
    assert len(forecasts) == 2
    for f in forecasts:
        assert f['end_of_month'] >= f['spent'] >= 0
        assert f['next_month'] >= 0


def test_gap_month_counts_as_zero():
    daily = {date(2026, 5, 10): 300.0, date(2026, 7, 10): 300.0, date(2026, 8, 5): 50.0}
    assert eom_features('Groceries', daily, date(2026, 8, 10))['avg3'] == 200


def test_year_boundary():
    r = post(date(2027, 2, 10), [{'key': 'k', 'category': None, 'daily': every_day(date(2026, 8, 1), date(2027, 2, 10))}])
    assert r.status_code == 200
    [f] = r.json()['forecasts']
    assert set(f) == {'key', 'spent', 'end_of_month', 'next_month'}
    assert f['spent'] == 1000


def test_month_end_days():
    daily = as_dates(every_day(date(2026, 4, 1), date(2027, 2, 27)))
    oct31 = eom_features('Food & Drinks', daily, date(2026, 10, 31))
    assert (oct31['days_left'], oct31['weekend_days_left']) == (0, 0)
    feb27 = eom_features('Food & Drinks', daily, date(2027, 2, 27))   # Feb 28, 2027 is a Sunday
    assert (feb27['days_left'], feb27['weekend_days_left']) == (1, 1)


def test_baseline_when_model_none(monkeypatch):
    monkeypatch.setattr(forecast, 'EOM', {'model': None, 'features': forecast.EOM['features']})
    monkeypatch.setattr(forecast, 'NEXT', {'model': None, 'features': forecast.NEXT['features']})
    today = date.fromisoformat(SAMPLE['request']['today'])
    for f, s in zip(client.post('/forecast', json=SAMPLE['request']).json()['forecasts'], SAMPLE['request']['series']):
        feats = eom_features(s['category'], as_dates(s['daily']), today)
        days = calendar.monthrange(today.year, today.month)[1]
        assert f['end_of_month'] == round(feats['spent'] / today.day * days, 2)
        assert f['next_month'] == round(feats['avg3'], 2)


def test_custom_category_uses_baseline():
    # The model never saw 'other' in training (its code would fall in Entertainment's leaf)
    today = date(2026, 8, 15)
    daily = every_day(date(2026, 2, 1), date(2026, 8, 15))
    [f] = post(today, [{'key': 'rent', 'category': None, 'daily': daily}]).json()['forecasts']
    assert f['end_of_month'] == round(1500 / 15 * 31, 2)
    assert f['next_month'] == round(sum(100 * n for n in (31, 31, 30)) / 3, 2)
