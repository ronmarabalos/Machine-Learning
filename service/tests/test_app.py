import pytest
from fastapi.testclient import TestClient

from app import INCOME, app

client = TestClient(app)
APP_DEFAULTS = {'Food & Drinks', 'Transportation', 'Bills & Utilities', 'Shopping', 'Groceries',
                'Coffee & Snacks', 'Healthcare', 'Entertainment', 'Salary', 'Freelance / Client', 'Other Income'}


def post(description, type='expense'):
    return client.post('/categorize', json={'description': description, 'type': type})


def test_jollibee_is_food():
    r = post('Jollibee')
    assert r.status_code == 200
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


def test_text_that_cleans_to_empty_returns_null():
    for text in ['12345', '*1234', 'makati']:
        r = post(text)
        assert r.status_code == 200
        assert r.json()['category'] is None


def test_non_ascii_text_does_not_crash():
    for text in ['Kape ☕ sa kanto', 'Ñaw']:
        assert post(text).status_code == 200


def test_rejects_unknown_type():
    assert post('Jollibee', 'transfer').status_code == 422


def test_health():
    assert client.get('/health').json() == {'status': 'ok', 'model_version': '2026-10-21'}


def test_model_classes_match_app_defaults():
    from app import model
    assert set(model.classes_) <= APP_DEFAULTS


def ask(question, today='2026-11-07'):
    return client.post('/ask', json={'question': question, 'today': today})


def test_ask_spend_with_date_range():
    body = ask('how much did i spend from aug 4 to aug 11').json()
    assert body['intent'] == 'spend_total'
    assert (body['start'], body['end']) == ('2026-08-04', '2026-08-11')


def test_ask_no_dates_uses_intent_default():
    body = ask('give me my weekly summary').json()
    assert body['intent'] == 'weekly_summary'
    assert (body['start'], body['end']) == ('2026-11-02', '2026-11-07')  # Monday to today


def test_ask_report_without_dates_means_all_time():
    body = ask('all transactions report').json()
    assert body['intent'] == 'report'
    assert body['start'] is None and body['end'] is None


def test_ask_unsure_returns_null_intent():
    body = ask('pera ko').json()  # too vague to pick an intent
    assert body['intent'] is None
    assert body['start'] is None and body['end'] is None


def test_ask_non_money_questions_get_no_dates():
    for q, intent in [("what's the weather today", 'out_of_scope'), ('banana', 'out_of_scope'),
                      ('how much is a jollibee chickenjoy', 'out_of_scope'),  # a price, not my spending
                      ('what is an emergency fund', 'finance_tip')]:
        body = ask(q).json()
        assert body['intent'] == intent, q
        assert body['start'] is None and body['end'] is None, q


def test_ask_finance_tip_comes_with_the_closest_answer():
    body = ask('what is an emergency fund').json()
    assert body['tip']['question'] == 'what is an emergency fund'
    assert 'surprises' in body['tip']['answer']


@pytest.mark.parametrize('question, lang', [
    ('how much did i spend this week', 'en'),
    ('what is an emergency fund', 'en'),
    ('report for may', 'en'),                    # "may" is the month here
    ('magkano nagastos ko this week', 'tl'),
    ('how much ang nagastos ko sa gcash', 'tl'),  # one Tagalog word is enough
    ('summary for kahapon', 'tl'),
    ('Paano mag-ipon?', 'tl'),
])
def test_ask_detects_taglish(question, lang):
    assert ask(question).json()['lang'] == lang


def test_ask_taglish_tip_is_in_taglish():
    body = ask('ano ang emergency fund').json()
    assert body['lang'] == 'tl'
    assert body['tip']['answer'].startswith('Ang emergency fund ay')


def test_ask_tip_only_for_finance_questions():
    assert ask('how much did i spend this week').json()['tip'] is None
    assert ask('banana').json()['tip'] is None


def test_ask_rejects_bad_input():
    assert ask('   ').status_code == 422
    assert ask('x' * 201).status_code == 422
    assert ask('weekly summary', today='not-a-date').status_code == 422
    assert client.post('/ask', json={'question': 'weekly summary'}).status_code == 422  # no today
