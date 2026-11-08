"""Turn the date words in a question into an inclusive (start, end) date range.

Rules (the app relies on these):
- Both ends are included, and `end` is never after `today`.
- A month/day with no year means the most recent one that isn't in the future.
- Weeks start on Monday.
- "N years ago" means that whole calendar year; "last N months/years" counts the current one.
- No date words -> None, and /ask falls back to default_range(intent).
"""
import calendar
import re
from datetime import date, timedelta

from forecast import days_in, month_start

MONTH = (r'(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?'
         r'|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)')
DAY = r'(\d{1,2})(?:st|nd|rd|th)?'
YEAR = r'((?:19|20)\d\d)'
TO = r'\s*(?:to|-|–|and|until|hanggang)\s*'
UNIT = r'(day|week|month|year)s?'
WEEKDAYS = ['mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
# Tagalog day names; "linggo" (Sunday) is left out because it also means "week"
TL_WEEKDAYS = {'lunes': 'mon', 'martes': 'tue', 'miyerkules': 'wed', 'miyerkoles': 'wed',
               'huwebes': 'thu', 'biyernes': 'fri', 'sabado': 'sat'}
DAY_NAMES = '|'.join(['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', *TL_WEEKDAYS])
MAY_BEFORE = {'in', 'noong', 'of', 'for', 'sa', 'last', 'this'}  # "may" is also a Tagalog word


def _month_no(name: str) -> int:
    return ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'].index(name[:3]) + 1


def _day(y: int, m: int, d: int) -> date:
    return date(y, m, min(d, calendar.monthrange(y, m)[1]))  # "feb 30" -> feb 28/29


def _on_or_before(month: int, day: int, today: date, year: int | None = None) -> date:
    d = _day(year or today.year, month, day)
    return _day(today.year - 1, month, day) if year is None and d > today else d


def _month(m: date) -> tuple[date, date]:
    return m, date(m.year, m.month, days_in(m))


def _year(y: int) -> tuple[date, date]:
    return date(y, 1, 1), date(y, 12, 31)


def _week(d: date) -> tuple[date, date]:
    start = d - timedelta(days=d.weekday())
    return start, start + timedelta(days=6)


def _recent_month(month: int, today: date) -> date:
    return date(today.year if month <= today.month else today.year - 1, month, 1)


PHRASES = [
    (r'\b(today|ngayon)\b', lambda t: (t, t)),
    (r'\b(yesterday|kahapon)\b', lambda t: (t - timedelta(1), t - timedelta(1))),
    (r'\b(this (week|wk)|ngayong linggo|nitong linggo)\b', _week),
    (r'\b(last (week|wk)|noong isang linggo|nakaraang linggo)\b', lambda t: _week(t - timedelta(7))),
    (r'\bweek before last\b', lambda t: _week(t - timedelta(14))),
    (r'\bchristmas\b', lambda t: (_on_or_before(12, 25, t),) * 2),
    (r"\bnew year'?s? day\b", lambda t: (_on_or_before(1, 1, t),) * 2),
    (r'\blast weekend\b', lambda t: (_week(t - timedelta(7))[0] + timedelta(5), _week(t - timedelta(7))[1])),
    (r'\bpast week\b', lambda t: (t - timedelta(6), t)),
    (r'\b(this month|ngayong buwan|buwan na ito)\b', lambda t: _month(month_start(t))),
    (r'\b(last month|noong isang buwan|nakaraang buwan)\b', lambda t: _month(month_start(t, 1))),
    (r'\b(this year|ngayong taon)\b', lambda t: _year(t.year)),
    (r'\b(last year|noong isang taon|nakaraang taon)\b', lambda t: _year(t.year - 1)),
]


def parse_range(question: str, today: date) -> tuple[date, date] | None:
    r = _parse(question.lower(), today)
    if r is None or r[0] > today:
        return None
    return r[0], min(r[1], today)


def _parse(q: str, today: date) -> tuple[date, date] | None:
    # Most specific patterns first: "aug 4 to aug 11" must not be read as just "aug 4"
    if m := re.search(r'\b(?:since|mula)\s+(.+)', q):
        r = _parse(m[1], today)
        if r and re.search(r'\b(?:to|until|hanggang)\b', m[1]):
            return r  # "mula aug 4 hanggang aug 11" has its own end
        return (r[0], today) if r else None

    if m := re.search(rf'\b{MONTH}\s+{DAY}{TO}(?:{MONTH}\s+)?{DAY}\b', q):
        start = _on_or_before(_month_no(m[1]), int(m[2]), today)
        end = _day(start.year, _month_no(m[3] or m[1]), int(m[4]))
        return start, (_day(end.year + 1, end.month, end.day) if end < start else end)  # "dec 28 to jan 3"

    if m := re.search(rf'\b{YEAR}{TO}{YEAR}\b', q):
        return date(int(m[1]), 1, 1), date(int(m[2]), 12, 31)

    if m := re.search(rf'\b{MONTH}{TO}{MONTH}\b', q):
        start = _recent_month(_month_no(m[1]), today)
        end = date(start.year, _month_no(m[2]), 1)
        return start, _month(end if end >= start else date(start.year + 1, end.month, 1))[1]

    if m := re.search(rf'\b{MONTH}\s+{DAY}\b(?:,?\s+{YEAR})?', q):
        d = _on_or_before(_month_no(m[1]), int(m[2]), today, int(m[3]) if m[3] else None)
        return _week(d) if 'week of' in q else (d, d)

    if m := re.search(rf'\b(?:past|last|nakaraang)\s+(\d+)\s+{UNIT}', q):
        n, unit = int(m[1]), m[2]
        start = {'day': today - timedelta(n - 1), 'week': today - timedelta(7 * n - 1),
                 'month': month_start(today, n - 1), 'year': date(today.year - n + 1, 1, 1)}[unit]
        return start, today

    if m := re.search(rf'\b(\d+)\s+{UNIT}\s+ago\b', q):
        n, unit = int(m[1]), m[2]
        if unit == 'day':
            return today - timedelta(n), today - timedelta(n)
        if unit == 'week':
            return _week(today - timedelta(7 * n))
        return _month(month_start(today, n)) if unit == 'month' else _year(today.year - n)

    for pattern, fn in PHRASES:
        if re.search(pattern, q):
            return fn(today)

    if m := re.search(rf'\b(last\s+|nakaraang\s+)?({DAY_NAMES})\b', q):
        back = (today.weekday() - WEEKDAYS.index(TL_WEEKDAYS.get(m[2], m[2][:3]))) % 7
        d = today - timedelta(7 if m[1] and back == 0 else back)
        return d, d

    if m := re.search(rf'\b{YEAR}\b', q):
        return _year(int(m[1]))

    for m in re.finditer(rf'(\w+\s+)?\b{MONTH}\b', q):
        if m[2] == 'may' and (m[1] or '').strip() not in MAY_BEFORE:
            continue
        return _month(_recent_month(_month_no(m[2]), today))

    # Last resort: period words with no actual date ("monthly report", "10 year report")
    if m := re.search(r'\b(\d+)[ -]years?\b', q):
        return date(today.year - int(m[1]) + 1, 1, 1), today
    if re.search(r'\b(monthly|buwanang)\b', q):
        return _month(month_start(today))
    if re.search(r'\b(yearly|annual|year in review|buong taon)\b', q):
        return _year(today.year)

    return None


def default_range(intent: str | None, today: date) -> tuple[date, date] | None:
    """Range to use when the question has no date words. None = all time / no dates needed."""
    if intent == 'daily_summary':
        return today, today
    if intent == 'weekly_summary':
        return _week(today)[0], today
    if intent in ('spend_total', 'income_total'):
        return month_start(today), today
    return None
