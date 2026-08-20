from services.prediction_service import predict_delay


def test_prediction_service():

    prediction = predict_delay(
        12301,
        "CNB",
        "Fog"
    )

    assert isinstance(prediction, float)
    assert prediction >= 0
    assert prediction < 180


def test_invalid_station():

    try:

        predict_delay(
            12301,
            "INVALID",
            "Fog"
        )

        assert False

    except ValueError:

        assert True


def test_invalid_reason():

    try:

        predict_delay(
            12301,
            "CNB",
            "INVALID"
        )

        assert False

    except ValueError:

        assert True
