import sys
import os
import logging
import json
from datetime import datetime

# =========================================================
# PROJECT ROOT
# =========================================================

sys.path.append(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

import pandas as pd
import joblib

from sqlalchemy import create_engine
from urllib.parse import quote_plus

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    filename="railsphere.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# =========================================================
# MODEL CONFIGURATION
# =========================================================

MODEL_VERSION = "1.0.0"
MODEL_NAME = "Train Delay Prediction Model"
ALGORITHM = "Random Forest Regressor"

FEATURES = [
    "train_no",
    "station_code",
    "reason"
]

TARGET = "delay_minutes"


# =========================================================
# DATABASE CONFIGURATION
# =========================================================

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "railsphere")


# =========================================================
# SQLALCHEMY ENGINE
# =========================================================

try:

    encoded_password = quote_plus(DB_PASSWORD)

    database_url = (
        f"mysql+pymysql://"
        f"{DB_USER}:{encoded_password}@"
        f"{DB_HOST}:{DB_PORT}/"
        f"{DB_NAME}"
    )

    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        pool_recycle=280
    )

    print("🔄 Connecting to database...")

    logger.info(
        "SQLAlchemy database engine created successfully."
    )

except Exception as e:

    logger.exception(
        "Failed to create SQLAlchemy database engine."
    )

    print(
        f"❌ Database engine error: {e}"
    )

    sys.exit(1)


# =========================================================
# LOAD DATA FROM MYSQL
# =========================================================

query = """
SELECT
    train_no,
    station_code,
    delay_minutes,
    reason,
    delay_date
FROM delay_logs
"""


try:

    df = pd.read_sql(
        query,
        engine
    )

    print(
        "✅ Database data loaded successfully!"
    )

    logger.info(
        f"Training data loaded successfully. "
        f"Records: {len(df)}"
    )

except Exception as e:

    logger.exception(
        "Failed to load training data from database."
    )

    print(
        f"❌ Failed to load training data: {e}"
    )

    sys.exit(1)


# =========================================================
# CHECK DATA
# =========================================================

print("\n📊 Training Data:")
print(df)

print(
    "\nTotal Records:",
    len(df)
)


if df.empty:

    print(
        "❌ No data found in delay_logs table."
    )

    logger.warning(
        "Training stopped because delay_logs table is empty."
    )

    sys.exit(1)


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

X = df[FEATURES]

y = df[TARGET]


# =========================================================
# TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print(
    "\n📚 Training Records:",
    len(X_train)
)

print(
    "🧪 Testing Records:",
    len(X_test)
)


# =========================================================
# RANDOM FOREST MODEL
# =========================================================

print(
    "\n🤖 Training Random Forest Model..."
)

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

print(
    "📊 MODEL EVALUATION RESULTS"
)

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

print(
    "\n📋 Actual vs Predicted Delay:"
)

results = pd.DataFrame({

    "Actual Delay": y_test.values,

    "Predicted Delay": [
        round(value, 2)
        for value in predictions
    ]

})

print(results)


# =========================================================
# SAVE MODEL + ENCODERS
# =========================================================

try:

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

    logger.info(
        "ML model and encoders saved successfully."
    )

except Exception as e:

    logger.exception(
        "Failed to save ML model files."
    )

    print(
        f"❌ Model saving error: {e}"
    )

    sys.exit(1)


# =========================================================
# MODEL METADATA
# =========================================================

metadata = {

    "model_version": MODEL_VERSION,

    "model_name": MODEL_NAME,

    "algorithm": ALGORITHM,

    "training_timestamp": datetime.now().isoformat(),

    "dataset_records": int(len(df)),

    "training_records": int(len(X_train)),

    "testing_records": int(len(X_test)),

    "features": FEATURES,

    "target": TARGET,

    "metrics": {

        "mae_minutes": round(
            float(mae),
            4
        ),

        "rmse_minutes": round(
            float(rmse),
            4
        ),

        "r2_score": (
            round(
                float(r2),
                4
            )
            if r2 is not None
            else None
        )

    }

}


# =========================================================
# SAVE MODEL METADATA
# =========================================================

try:

    with open(
        "ml/model_metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    logger.info(
        "Model metadata saved successfully."
    )

except Exception as e:

    logger.exception(
        "Failed to save model metadata."
    )

    print(
        f"❌ Metadata saving error: {e}"
    )

    sys.exit(1)


# =========================================================
# FINAL STATUS
# =========================================================

print(
    "\n💾 Model saved successfully!"
)

print(
    "\nCreated files:"
)

print(
    "✅ ml/train_delay_model.pkl"
)

print(
    "✅ ml/station_encoder.pkl"
)

print(
    "✅ ml/reason_encoder.pkl"
)

print(
    "✅ ml/model_metadata.json"
)

print(
    "\n🎉 Train Delay ML Model is ready!"
)


logger.info(
    f"Model training completed successfully. "
    f"Version={MODEL_VERSION}, "
    f"MAE={mae:.2f}, "
    f"RMSE={rmse:.2f}, "
    f"R2={r2}"
)