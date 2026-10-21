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
