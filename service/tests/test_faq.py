"""Finance tip matching: new wordings find the right topic, off-topic questions get nothing."""
import csv
from pathlib import Path

import pytest

import faq

HELD_OUT = [  # wordings not in faq.csv
 ("whats an emergency fund for", "emergency_fund"),
 ("how many months should my emergency fund cover", "emergency_fund_size"),
 ("paano ako makakaipon", "start_saving"),
 ("how much of my salary should i save", "savings_rate"),
 ("explain the 50 30 20 budget rule", "rule_50_30_20"),
 ("how do i make a monthly budget", "make_budget"),
 ("how to avoid overspending", "stick_to_budget"),
 ("how do i stop impulse buying", "impulse_buying"),
 ("what are sinking funds", "sinking_fund"),
 ("paano mag budget pag freelancer", "irregular_income"),
 ("lagi kulang sweldo ko", "paycheck_to_paycheck"),
 ("how to spend less on groceries", "groceries"),
 ("how does compound interest work for savings", "compound_interest"),
 ("what does inflation mean", "inflation"),
 ("should i pay my debt first or save", "debt_or_save"),
 ("how to get out of credit card debt fast", "pay_credit_card"),
 ("is mp2 worth it", "mp2"),
 ("how do i start investing with small money", "start_investing"),
 ("is a time deposit good", "time_deposit"),
 ("what to do with my 13th month pay", "thirteenth_month"),
 ("is a paluwagan a good idea", "paluwagan"),
 ("how do i compute my net worth", "net_worth"),
 ("should i buy vul insurance", "insurance"),
 ("how to know if an investment is a scam", "scams"),
 ]
OFF_TOPIC = ["what is bitcoin price today", "how to cook adobo", "who is the president",
       "how to lose weight", "what is a mortgage refinance", "how to file taxes online",
       "tell me a joke", "what is the meaning of life", "how to fix my phone", "best laptop 2026"]


@pytest.mark.parametrize('question,topic', HELD_OUT)
def test_new_wordings_find_the_right_tip(question, topic):
    tip = faq.match(question)
    assert tip and tip['topic'] == topic, (question, faq.best_match(question)[1:])


@pytest.mark.parametrize('question', OFF_TOPIC)
def test_off_topic_questions_get_no_tip(question):
    assert faq.match(question) is None, (question, faq.best_match(question)[1:])


def test_faq_csv_is_well_formed():
    with open(Path(faq.__file__).parent / 'faq.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    topics = [r['topic'] for r in rows]
    assert len(topics) == len(set(topics)), 'duplicate topic'
    phrases = [q.strip().lower() for r in rows for q in r['questions'].split('|')]
    assert len(phrases) == len(set(phrases)), 'same wording under two topics'
    for r in rows:
        assert r['answer'].strip() and len(r['answer']) <= 500, r['topic']
        assert r['answer_tl'].strip() and len(r['answer_tl']) <= 600, r['topic']
