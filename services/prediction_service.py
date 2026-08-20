import os
import joblib
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "train_delay_model.pkl"
)

STATION_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "station_encoder.pkl"
)

REASON_ENCODER_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "reason_encoder.pkl"
)


def load_prediction_model():

    model = joblib.load(MODEL_PATH)
    station_encoder = joblib.load(STATION_ENCODER_PATH)
    reason_encoder = joblib.load(REASON_ENCODER_PATH)

    return (
        model,
        station_encoder,
        reason_encoder
    )


def predict_delay(
    train_no,
    station_code,
    reason
):

    model, station_encoder, reason_encoder = (
        load_prediction_model()
    )

    if station_code not in station_encoder.classes_:
        raise ValueError(
            f"Unknown station code: {station_code}"
        )

    if reason not in reason_encoder.classes_:
        raise ValueError(
            f"Unknown delay reason: {reason}"
        )

    station_value = station_encoder.transform(
        [station_code]
    )[0]

    reason_value = reason_encoder.transform(
        [reason]
    )[0]

    input_data = pd.DataFrame(
        [[
            train_no,
            station_value,
            reason_value
        ]],
        columns=[
            "train_no",
            "station_code",
            "reason"
        ]
    )

    prediction = model.predict(
        input_data
    )[0]

    return max(
        0.0,
        float(prediction)
    )
