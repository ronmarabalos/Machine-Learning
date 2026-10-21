# Use the same cleaning as training so the model receives text in the format it learned.
import re

BRANCHES = [
    'sm north', 'megamall', 'bgc',
    'makati', 'qc', 'cubao', 'ortigas'
]

def clean_description(text):
    text = text.lower()
    text = re.sub(r'\*?\d+', '', text)

    for branch in BRANCHES:
        text = text.replace(branch, '')

    return text.strip()
