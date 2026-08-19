import os
import joblib


MODEL_PATH = "ml/train_delay_model.pkl"
STATION_ENCODER_PATH = "ml/station_encoder.pkl"
REASON_ENCODER_PATH = "ml/reason_encoder.pkl"


def test_ml_files_exist():
    assert os.path.exists(MODEL_PATH)
    assert os.path.exists(STATION_ENCODER_PATH)
    assert os.path.exists(REASON_ENCODER_PATH)


def test_ml_model_loads():
    model = joblib.load(MODEL_PATH)

    assert model is not None
    assert hasattr(model, "predict")


def test_encoders_load():
    station_encoder = joblib.load(STATION_ENCODER_PATH)
    reason_encoder = joblib.load(REASON_ENCODER_PATH)

    assert station_encoder is not None
    assert reason_encoder is not None
