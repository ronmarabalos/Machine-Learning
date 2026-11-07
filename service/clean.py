# Use the same cleaning as training so the model receives text in the format it learned.
import re

BRANCHES = [
    'sm north', 'megamall', 'bgc',
    'makati', 'qc', 'cubao', 'ortigas'
]

def clean_description(text):
    text = text.lower()
    text = re.sub(r'\*?\d{3,}', '', text)  # store codes like *4821; keeps the 7 in 7-Eleven

    for branch in BRANCHES:
        text = text.replace(branch, '')

    return text.strip()
