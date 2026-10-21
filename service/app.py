from pathlib import Path
from typing import Annotated, Literal

import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, StringConstraints

from clean import clean_description

THRESHOLD = 0.5
MODEL_VERSION = '2026-10-21'
INCOME = {'Salary', 'Freelance / Client', 'Other Income'}

# Load the saved pipeline once when this module is imported
model = joblib.load(
    Path(__file__).parent / 'model' / 'categorizer.joblib'
)

app = FastAPI()


class CategorizeRequest(BaseModel):
    description: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=200
        )
    ]
    type: Literal['expense', 'income']


@app.post('/categorize')
def categorize(req: CategorizeRequest):
    cleaned = clean_description(req.description)

    if not cleaned:
        return {'category': None, 'confidence': 0.0}

    probs = model.predict_proba([cleaned])[0]

    # Keep only categories matching the requested transaction type
    keep = np.array([
        (category in INCOME) == (req.type == 'income')
        for category in model.classes_
    ])
    probs = np.where(keep, probs, 0.0)

    total = probs.sum()
    if total == 0:
        return {'category': None, 'confidence': 0.0}

    # Rescale the remaining probabilities so they sum to 1
    probs = probs / total

    best_index = int(np.argmax(probs))
    confidence = float(probs[best_index])

    category = (
        str(model.classes_[best_index])
        if confidence >= THRESHOLD
        else None
    )

    return {'category': category, 'confidence': confidence}


@app.get('/health')
def health():
    return {'status': 'ok', 'model_version': MODEL_VERSION}
