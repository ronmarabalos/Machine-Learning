"""Generate fake Budget Tracker transactions in the app's CSV export format.

Usage: .venv\\Scripts\\python scripts\\generate_fake_data.py
Writes data/fake_transactions.csv (importable back into the app via Data Backup).
"""

import csv
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

random.seed(42)  # same data every run, so results are comparable

# category -> (merchants, typical min/max amount in PHP, avg transactions per day)
EXPENSES = {
    "Food & Drinks": (["Jollibee", "McDonald's", "Chowking", "Mang Inasal", "Greenwich", "KFC", "Max's Restaurant", "Shakey's", "Foodpanda", "GrabFood", "Carinderia lunch", "Lugaw"], 60, 600, 1.2),
    "Coffee & Snacks": (["Starbucks", "Tim Hortons", "Coffee Project", "But First Coffee", "Dunkin", "Potato Corner", "Zagu", "Bo's Coffee"], 40, 250, 0.6),
    "Groceries": (["SM Supermarket", "Puregold", "Robinsons Supermarket", "Savemore", "Landers", "S&R", "Palengke", "Alfamart"], 150, 3500, 0.35),
    "Transportation": (["Grab", "Angkas", "Joyride", "Jeep fare", "MRT Beep load", "LRT", "Bus fare", "Petron", "Shell", "Toll Autosweep"], 15, 900, 1.0),
    "Bills & Utilities": ([], 0, 0, 0),  # monthly recurring, handled below
    "Shopping": (["Shopee", "Lazada", "Uniqlo", "Penshoppe", "SM Department Store", "Miniso", "Daiso", "Ace Hardware"], 150, 4000, 0.2),
    "Healthcare": (["Mercury Drug", "Watsons", "The Generics Pharmacy", "Rose Pharmacy", "Clinic checkup", "Dental clinic"], 80, 2500, 0.08),
    "Entertainment": (["SM Cinema", "Steam", "Timezone", "Ayala Malls Cinema", "Concert ticket", "Videoke"], 150, 2000, 0.08),
}
# 7-Eleven sits in two categories on purpose: real data is ambiguous too.
EXPENSES["Coffee & Snacks"][0].append("7-Eleven")
EXPENSES["Groceries"][0].append("7-Eleven")

BILLS = [("Meralco", 1800, 3500), ("Maynilad", 300, 700), ("PLDT Home Fiber", 1699, 1699), ("Globe Postpaid", 999, 999), ("Netflix", 549, 549), ("Spotify Premium", 194, 194)]
ACCOUNTS = ["Cash Wallet", "GCash", "BDO Savings"]


def noisy(name):
    """Make descriptions look like real typed/bank text: case, branches, ref numbers."""
    r = random.random()
    if r < 0.15:
        return name.upper()
    if r < 0.25:
        return name.lower()
    if r < 0.40:
        return f"{name} {random.choice(['SM North', 'Megamall', 'BGC', 'Makati', 'QC', 'Cubao', 'Ortigas'])}"
    if r < 0.50:
        return f"{name.upper().replace(' ', '')}*{random.randint(1000, 9999)}"
    return name


def row(date, kind, amount, category, description, account=None, anomaly=""):
    ts = date.replace(hour=random.randint(7, 22), minute=random.randint(0, 59), tzinfo=timezone.utc)
    return {
        "Date": ts.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "Type": kind,
        "Amount": f"{amount:.2f}",
        "Category": category,
        "Account": account or random.choice(ACCOUNTS),
        "Description": description,
        "Anomaly": anomaly,
    }


END = datetime(2026, 10, 21)  # fixed (was "today") so every run produces the same file
SPREE_CATEGORIES = ["Groceries", "Shopping", "Entertainment"]  # ≤ 2 purchases/week, so a spree stands out


def generate(days=365):
    end = END
    rows = []
    for d in range(days):
        date = end - timedelta(days=days - d)
        for category, (merchants, lo, hi, per_day) in EXPENSES.items():
            if not merchants:
                continue
            for _ in range(int(per_day) + (random.random() < per_day % 1)):
                amount = random.uniform(lo, hi)
                spike = random.random() < 0.01
                if spike:
                    amount *= random.uniform(4, 8)  # rare spending spike, labelled for anomaly detection
                desc = "" if random.random() < 0.05 else noisy(random.choice(merchants))
                rows.append(row(date, "expense", amount, category, desc, anomaly="spike" if spike else ""))
        if date.day == 5:
            for name, lo, hi in BILLS:
                rows.append(row(date, "expense", random.uniform(lo, hi), "Bills & Utilities", noisy(name), "BDO Savings"))
        if date.day in (15, 30):
            rows.append(row(date, "income", 18000, "Salary", "Payroll", "BDO Savings"))
        if random.random() < 0.03:
            rows.append(row(date, "income", random.uniform(2000, 15000), "Freelance / Client", noisy(random.choice(["Upwork", "Client payment", "Fiverr"])), "GCash"))
        if random.random() < 0.02:
            rows.append(row(date, "income", random.uniform(200, 3000), "Other Income", random.choice(["Gift", "Refund", "Sold item"]), "GCash"))

    # Spree weeks: many normal-sized purchases that add up. A separate RNG, used after the
    # main loop, keeps every row above identical to earlier runs.
    rng = random.Random(7)
    first_monday = end - timedelta(days=days) + timedelta(days=(7 - (end - timedelta(days=days)).weekday()) % 7)
    mondays = [first_monday + timedelta(weeks=w) for w in range((end - first_monday).days // 7)]
    for monday in rng.sample(mondays[1:-1], 6):  # skip partial first/last weeks
        category = rng.choice(SPREE_CATEGORIES)
        merchants, lo, hi, _ = EXPENSES[category]
        for _ in range(rng.randint(4, 6)):
            day = monday + timedelta(days=rng.randint(0, 6))
            rows.append(row(day, "expense", rng.uniform(lo, hi), category, rng.choice(merchants), anomaly="spree"))
    rows.sort(key=lambda r: r["Date"], reverse=True)  # newest first, like the app
    return rows


if __name__ == "__main__":
    rows = generate()

    # self-check: matches the app's default categories and import rules
    defaults = set(EXPENSES) | {"Salary", "Freelance / Client", "Other Income"}
    assert {r["Category"] for r in rows} <= defaults
    assert all(float(r["Amount"]) > 0 for r in rows)
    assert len(rows) > 1000
    expenses = [r for r in rows if r["Type"] == "expense"]
    assert 0.005 <= sum(r["Anomaly"] == "spike" for r in expenses) / len(expenses) <= 0.02
    assert 24 <= sum(r["Anomaly"] == "spree" for r in rows) <= 36

    out = Path(__file__).resolve().parent.parent / "data" / "fake_transactions.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Type", "Amount", "Category", "Account", "Description", "Anomaly"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} transactions to {out}")
