import math

import numpy as np

from classification_model.predict import make_prediction
from sklearn.metrics import accuracy_score

def test_make_prediction(sample_input_data):
    # Given
    expected_no_predictions = 131

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    predictions = result.get("predictions")
    assert isinstance(predictions, np.ndarray)
    assert isinstance(predictions[0], np.int64)
    assert result.get("errors") is None
    assert len(predictions) == expected_no_predictions
    _predictions = list(predictions)
    y_true = sample_input_data["survived"]
    accuracy = accuracy_score(_predictions, y_true)
    assert accuracy > 0.7

