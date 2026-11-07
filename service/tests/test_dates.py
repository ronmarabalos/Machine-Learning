from datetime import date

import pytest

from dates import default_range, parse_range

TODAY = date(2026, 11, 7)  # a Saturday


def d(s):
    return date.fromisoformat(s)


def test_today_is_saturday():
    assert TODAY.weekday() == 5


@pytest.mark.parametrize('question, start, end', [
    # explicit ranges, both ends included
    ('how much did i spend from aug 4 to aug 11', '2026-08-04', '2026-08-11'),
    ('how much did i spend between jan 1 and jan 15', '2026-01-01', '2026-01-15'),
    ('expenses total from may 1 to may 31', '2026-05-01', '2026-05-31'),
    ('dec 28 to jan 3', '2025-12-28', '2026-01-03'),
    ('report for january to june', '2026-01-01', '2026-06-30'),
    ('report from 2016 to 2026', '2016-01-01', '2026-11-07'),  # end capped at today
    # single days; no year = most recent one not in the future
    ('summary nov 5', '2026-11-05', '2026-11-05'),
    ('spent on dec 25', '2025-12-25', '2025-12-25'),
    ('summary for aug 4 2025', '2025-08-04', '2025-08-04'),
    ('how was my week of nov 4', '2026-11-02', '2026-11-07'),  # Mon–Sun week, capped at today
    ('how much did i spend today', '2026-11-07', '2026-11-07'),
    ('magkano nagastos ko kahapon', '2026-11-06', '2026-11-06'),
    ('summary for last friday', '2026-11-06', '2026-11-06'),
    ('day summary for monday', '2026-11-02', '2026-11-02'),
    ('how much did i spend 3 days ago', '2026-11-04', '2026-11-04'),
    # weeks start on Monday
    ('weekly summary this week', '2026-11-02', '2026-11-07'),
    ('magkano ginastos ko ngayong linggo', '2026-11-02', '2026-11-07'),
    ('summary for last week', '2026-10-26', '2026-11-01'),
    ('how much money did i spend last weekend', '2026-10-31', '2026-11-01'),
    ('what did i pay for 2 weeks ago', '2026-10-19', '2026-10-25'),
    ('recap of the past 7 days', '2026-11-01', '2026-11-07'),
    ('total spending since monday', '2026-11-02', '2026-11-07'),
    # months and years
    ('total expenses this month', '2026-11-01', '2026-11-07'),
    ('spending report last month', '2026-10-01', '2026-10-31'),
    ('report for october', '2026-10-01', '2026-10-31'),
    ('list all shopping purchases in december', '2025-12-01', '2025-12-31'),
    ('magkano lahat ng gastos ko noong march', '2026-03-01', '2026-03-31'),
    ('how did my finances look over the last 12 months', '2025-12-01', '2026-11-07'),
    ('how much did i spend in 2023', '2023-01-01', '2023-12-31'),
    ('transactions report 10 years ago', '2016-01-01', '2016-12-31'),
    ('total income since january', '2026-01-01', '2026-11-07'),
    ('overview of everything since 2015', '2015-01-01', '2026-11-07'),
    ('summary of the week before last', '2026-10-19', '2026-10-25'),
    ('how did my spending look on christmas day', '2025-12-25', '2025-12-25'),
    ("recap of my expenses on new year's day", '2026-01-01', '2026-01-01'),
    # period words with no date
    ('monthly report', '2026-11-01', '2026-11-07'),
    ('buwanang report', '2026-11-01', '2026-11-07'),
    ('yearly overview', '2026-01-01', '2026-11-07'),
    ('10 year report', '2017-01-01', '2026-11-07'),
    ('monthly breakdown for 2024', '2024-01-01', '2024-12-31'),  # a real year beats "monthly"
])
def test_parse_range(question, start, end):
    assert parse_range(question, TODAY) == (d(start), d(end))


@pytest.mark.parametrize('question', [
    'weekly report for maya',        # a wallet, not the month May
    'may pera pa ba ako',            # Tagalog "may", not the month
    'show transactions above 1000',  # not a year
    'show my last 10 transactions',  # "last 10" without a unit
    'how to make a budget',
    'all transactions report',
])
def test_no_dates(question):
    assert parse_range(question, TODAY) is None


@pytest.mark.parametrize('intent, expected', [
    ('daily_summary', (d('2026-11-07'), d('2026-11-07'))),
    ('weekly_summary', (d('2026-11-02'), d('2026-11-07'))),
    ('spend_total', (d('2026-11-01'), d('2026-11-07'))),
    ('income_total', (d('2026-11-01'), d('2026-11-07'))),
    ('report', None),
    ('list_transactions', None),
    ('finance_tip', None),
    (None, None),
])
def test_default_range(intent, expected):
    assert default_range(intent, TODAY) == expected
