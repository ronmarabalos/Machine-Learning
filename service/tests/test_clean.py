from clean import clean_description


def test_matches_notebook_cleaning():
    assert clean_description('JOLLIBEE*4821') == 'jollibee'
    assert clean_description('Jollibee Makati') == 'jollibee'
    assert clean_description('  Starbucks ') == 'starbucks'
