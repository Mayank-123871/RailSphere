import sys
import os

# Project root ko Python path me add karna
sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from database.db import get_connection


# =========================================================
# LOAD DATA FROM MYSQL
# =========================================================

print("🔄 Connecting to database...")

conn = get_connection()

query = """
SELECT
    train_no,
    station_code,
    delay_minutes,
    reason,
    delay_date
FROM delay_logs
"""

df = pd.read_sql(query, conn)

conn.close()

print("✅ Database data loaded successfully!")


# =========================================================
# CHECK DATA
# =========================================================

print("\n📊 Training Data:")
print(df)

print("\nTotal Records:", len(df))


if df.empty:

    print("❌ No data found in delay_logs table.")
    sys.exit()


# =========================================================
# ENCODE CATEGORICAL DATA
# =========================================================

station_encoder = LabelEncoder()

reason_encoder = LabelEncoder()


df["station_code"] = station_encoder.fit_transform(
    df["station_code"]
)

df["reason"] = reason_encoder.fit_transform(
    df["reason"]
)


# =========================================================
# FEATURES AND TARGET
# =========================================================

X = df[
    [
        "train_no",
        "station_code",
        "reason"
    ]
]

y = df["delay_minutes"]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("\n📚 Training Records:", len(X_train))
print("🧪 Testing Records:", len(X_test))


# =========================================================
# RANDOM FOREST MODEL
# =========================================================

print("\n🤖 Training Random Forest Model...")

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(
    X_train,
    y_train
)


# =========================================================
# PREDICTION
# =========================================================

predictions = model.predict(
    X_test
)


# =========================================================
# MODEL EVALUATION
# =========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5


# R² cannot be reliably calculated
# when there is only one test sample.

if len(y_test) > 1:

    r2 = r2_score(
        y_test,
        predictions
    )

else:

    r2 = None


# =========================================================
# DISPLAY RESULTS
# =========================================================

print("\n" + "=" * 50)

print("📊 MODEL EVALUATION RESULTS")

print("=" * 50)

print(
    f"📈 Mean Absolute Error (MAE): "
    f"{mae:.2f} minutes"
)

print(
    f"📉 Root Mean Squared Error (RMSE): "
    f"{rmse:.2f} minutes"
)

if r2 is not None:

    print(
        f"🎯 R² Score: "
        f"{r2:.4f}"
    )

else:

    print(
        "🎯 R² Score: Not available "
        "(only one test record)"
    )


print("=" * 50)


# =========================================================
# ACTUAL VS PREDICTED
# =========================================================

print("\n📋 Actual vs Predicted Delay:")

results = pd.DataFrame({

    "Actual Delay": y_test.values,

    "Predicted Delay": [
        round(value, 2)
        for value in predictions
    ]

})

print(results)


# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "ml/train_delay_model.pkl"
)

joblib.dump(
    station_encoder,
    "ml/station_encoder.pkl"
)

joblib.dump(
    reason_encoder,
    "ml/reason_encoder.pkl"
)


print("\n💾 Model saved successfully!")

print("\nCreated files:")

print("✅ ml/train_delay_model.pkl")

print("✅ ml/station_encoder.pkl")

print("✅ ml/reason_encoder.pkl")

print("\n🎉 Train Delay ML Model is ready!")