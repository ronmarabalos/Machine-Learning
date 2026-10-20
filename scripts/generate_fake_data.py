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


def row(date, kind, amount, category, description, account=None):
    ts = date.replace(hour=random.randint(7, 22), minute=random.randint(0, 59), tzinfo=timezone.utc)
    return {
        "Date": ts.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "Type": kind,
        "Amount": f"{amount:.2f}",
        "Category": category,
        "Account": account or random.choice(ACCOUNTS),
        "Description": description,
    }


def generate(days=365):
    end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    rows = []
    for d in range(days):
        date = end - timedelta(days=days - d)
        for category, (merchants, lo, hi, per_day) in EXPENSES.items():
            if not merchants:
                continue
            for _ in range(int(per_day) + (random.random() < per_day % 1)):
                amount = random.uniform(lo, hi)
                if random.random() < 0.01:
                    amount *= random.uniform(4, 8)  # rare spending spike, for anomaly detection later
                desc = "" if random.random() < 0.05 else noisy(random.choice(merchants))
                rows.append(row(date, "expense", amount, category, desc))
        if date.day == 5:
            for name, lo, hi in BILLS:
                rows.append(row(date, "expense", random.uniform(lo, hi), "Bills & Utilities", noisy(name), "BDO Savings"))
        if date.day in (15, 30):
            rows.append(row(date, "income", 18000, "Salary", "Payroll", "BDO Savings"))
        if random.random() < 0.03:
            rows.append(row(date, "income", random.uniform(2000, 15000), "Freelance / Client", noisy(random.choice(["Upwork", "Client payment", "Fiverr"])), "GCash"))
        if random.random() < 0.02:
            rows.append(row(date, "income", random.uniform(200, 3000), "Other Income", random.choice(["Gift", "Refund", "Sold item"]), "GCash"))
    rows.sort(key=lambda r: r["Date"], reverse=True)  # newest first, like the app
    return rows


if __name__ == "__main__":
    rows = generate()

    # self-check: matches the app's default categories and import rules
    defaults = set(EXPENSES) | {"Salary", "Freelance / Client", "Other Income"}
    assert {r["Category"] for r in rows} <= defaults
    assert all(float(r["Amount"]) > 0 for r in rows)
    assert len(rows) > 1000

    out = Path(__file__).resolve().parent.parent / "data" / "fake_transactions.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Date", "Type", "Amount", "Category", "Account", "Description"], quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} transactions to {out}")
