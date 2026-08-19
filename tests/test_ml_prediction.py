import joblib
import pandas as pd


MODEL_PATH = "ml/train_delay_model.pkl"
STATION_ENCODER_PATH = "ml/station_encoder.pkl"
REASON_ENCODER_PATH = "ml/reason_encoder.pkl"


def test_prediction_output():

    model = joblib.load(MODEL_PATH)
    station_encoder = joblib.load(STATION_ENCODER_PATH)
    reason_encoder = joblib.load(REASON_ENCODER_PATH)

    station = "CNB"
    reason = "Fog"

    station_value = station_encoder.transform([station])[0]
    reason_value = reason_encoder.transform([reason])[0]

    input_data = pd.DataFrame(
        [[12301, station_value, reason_value]],
        columns=[
            "train_no",
            "station_code",
            "reason"
        ]
    )

    prediction = model.predict(input_data)[0]

    assert prediction >= 0
    assert prediction < 180