from clean import clean_description


def test_matches_notebook_cleaning():
    assert clean_description('JOLLIBEE*4821') == 'jollibee'
    assert clean_description('Jollibee Makati') == 'jollibee'
    assert clean_description('  Starbucks ') == 'starbucks'


def test_keeps_short_numbers_in_names():
    assert clean_description('7-Eleven') == '7-eleven'
    assert clean_description('7-ELEVEN*2231') == '7-eleven'
    assert clean_description('Grab 4821') == 'grab'
