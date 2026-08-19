import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import joblib
import folium

from streamlit_folium import st_folium
from database.db import get_connection

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =========================================================
# PROJECT BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="RailSphere",
    page_icon="🚆",
    layout="wide"
)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🚆 RailSphere")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Stations",
        "Trains",
        "Routes",
        "Analytics",
        "Simulation",
        "Prediction",
        "ML Evaluation"
    ]
)


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    try:

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM stations")
        stations = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM trains")
        trains = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM routes")
        routes = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM platforms")
        platforms = cursor.fetchone()[0]

        cursor.close()
        conn.close()

    except Exception as e:

        st.error(f"❌ Database Error: {e}")
        st.stop()

    st.title("🚆 RailSphere")

    st.subheader(
        "AI-Powered Railway Intelligence & Simulation Platform"
    )

    st.write(
        "Welcome to RailSphere!"
    )

    # -------------------------------------------------
    # METRICS
    # -------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🚉 Stations",
        stations
    )

    col2.metric(
        "🚆 Trains",
        trains
    )

    col3.metric(
        "🛤️ Routes",
        routes
    )

    col4.metric(
        "🚦 Platforms",
        platforms
    )

    # =================================================
    # INTERACTIVE RAILWAY MAP
    # =================================================

    st.markdown("---")

    st.subheader(
        "🗺️ Interactive Railway Network"
    )

    try:

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                station_code,
                station_name,
                city,
                state,
                latitude,
                longitude
            FROM stations
            WHERE latitude IS NOT NULL
            AND longitude IS NOT NULL
        """)

        station_data = cursor.fetchall()

        cursor.close()
        conn.close()

        if station_data:

            station_df = pd.DataFrame(
                station_data,
                columns=[
                    "Code",
                    "Station",
                    "City",
                    "State",
                    "Latitude",
                    "Longitude"
                ]
            )

            center_lat = station_df[
                "Latitude"
            ].mean()

            center_lon = station_df[
                "Longitude"
            ].mean()

            railway_map = folium.Map(
                location=[
                    center_lat,
                    center_lon
                ],
                zoom_start=5,
                tiles="OpenStreetMap"
            )

            # -----------------------------------------
            # STATION MARKERS
            # -----------------------------------------

            for _, row in station_df.iterrows():

                popup_text = f"""
                <b>🚉 {row['Station']}</b><br>
                Code: {row['Code']}<br>
                City: {row['City']}<br>
                State: {row['State']}
                """

                folium.Marker(
                    location=[
                        row["Latitude"],
                        row["Longitude"]
                    ],
                    popup=folium.Popup(
                        popup_text,
                        max_width=300
                    ),
                    tooltip=row["Code"],
                    icon=folium.Icon(
                        icon="train",
                        prefix="fa"
                    )
                ).add_to(railway_map)

            # -------------------------------------------------
            # LOAD ROUTE DATA
            # -------------------------------------------------

            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    train_no,
                    station_code,
                    stop_number
                FROM routes
                ORDER BY train_no, stop_number
            """)

            route_data = cursor.fetchall()

            cursor.close()
            conn.close()

            # Create station coordinate lookup
            station_coordinates = {
                row["Code"]: (
                    row["Latitude"],
                    row["Longitude"]
                )
                for _, row in station_df.iterrows()
            }

            # -------------------------------------------------
            # DRAW TRAIN ROUTES
            # -------------------------------------------------

            if route_data:

                route_df = pd.DataFrame(
                    route_data,
                    columns=[
                        "Train No",
                        "Station",
                        "Stop Number"
                    ]
                )

                for train_no, train_route in route_df.groupby(
                    "Train No"
                ):

                    train_route = train_route.sort_values(
                        "Stop Number"
                    )

                    route_coordinates = []

                    for station_code in train_route["Station"]:

                        if station_code in station_coordinates:

                            lat, lon = station_coordinates[
                                station_code
                            ]

                            route_coordinates.append(
                                [lat, lon]
                            )

                    if len(route_coordinates) >= 2:

                        folium.PolyLine(
                            locations=route_coordinates,
                            weight=4,
                            opacity=0.8,
                            tooltip=f"🚆 Train {train_no}"
                        ).add_to(railway_map)

                st.success(
                    f"🛤️ {route_df['Train No'].nunique()} "
                    "train route(s) loaded on the map."
                )

            else:

                st.info(
                    "ℹ️ No route data available for map visualization."
                )

            # -------------------------------------------------
            # DISPLAY MAP
            # -------------------------------------------------

            st_folium(
                railway_map,
                width=None,
                height=600
            )

        else:

            st.warning(
                "⚠️ No station coordinates available."
            )

    except Exception as e:

        st.error(
            f"❌ Railway Map Error: {e}"
        )

    # =================================================
    # LATEST UPDATES
    # =================================================

    st.markdown("---")

    st.subheader(
        "📢 Latest Updates"
    )

    st.success(
        "Rajdhani Express is running on time."
    )

    st.warning(
        "Vande Bharat delayed by 12 minutes."
    )

    st.error(
        "Platform congestion detected."
    )


# =========================================================
# STATIONS
# =========================================================

elif page == "Stations":

    st.title("🚉 Stations")

    try:

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM stations"
        )

        stations_data = cursor.fetchall()

        cursor.close()
        conn.close()

        st.dataframe(
            stations_data,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"❌ Database Error: {e}"
        )


# =========================================================
# TRAINS
# =========================================================

elif page == "Trains":

    st.title("🚆 Trains")

    try:

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM trains"
        )

        trains_data = cursor.fetchall()

        cursor.close()
        conn.close()

        st.dataframe(
            trains_data,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"❌ Database Error: {e}"
        )


# =========================================================
# ROUTES
# =========================================================

elif page == "Routes":

    st.title("🛤️ Routes")

    try:

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM routes"
        )

        routes_data = cursor.fetchall()

        cursor.close()
        conn.close()

        st.dataframe(
            routes_data,
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"❌ Database Error: {e}"
        )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "Analytics":

    st.title(
        "📊 Railway Analytics"
    )

    st.subheader(
        "Railway Performance & Delay Analysis"
    )

    try:

        conn = get_connection()
        cursor = conn.cursor()

        # ---------------------------------------------
        # DELAY DATA
        # ---------------------------------------------

        cursor.execute("""
            SELECT
                train_no,
                station_code,
                delay_minutes,
                reason,
                delay_date
            FROM delay_logs
        """)

        delay_data = cursor.fetchall()

        # ---------------------------------------------
        # PASSENGER DATA
        # ---------------------------------------------

        cursor.execute("""
            SELECT
                station_code,
                passenger_count,
                recorded_at
            FROM passenger_flow
        """)

        passenger_data = cursor.fetchall()

        cursor.close()
        conn.close()

        delay_df = pd.DataFrame(
            delay_data,
            columns=[
                "Train No",
                "Station",
                "Delay Minutes",
                "Reason",
                "Date"
            ]
        )

        passenger_df = pd.DataFrame(
            passenger_data,
            columns=[
                "Station",
                "Passenger Count",
                "Recorded At"
            ]
        )

        # ---------------------------------------------
        # DELAY STATISTICS
        # ---------------------------------------------

        st.markdown("---")

        st.subheader(
            "🚆 Delay Statistics"
        )

        if not delay_df.empty:

            col1, col2, col3 = st.columns(3)

            avg_delay = (
                delay_df["Delay Minutes"].mean()
            )

            max_delay = (
                delay_df["Delay Minutes"].max()
            )

            total_delays = len(delay_df)

            col1.metric(
                "Average Delay",
                f"{avg_delay:.1f} min"
            )

            col2.metric(
                "Maximum Delay",
                f"{max_delay} min"
            )

            col3.metric(
                "Delay Records",
                total_delays
            )

            # -----------------------------------------
            # DELAY RECORDS
            # -----------------------------------------

            st.markdown("---")

            st.subheader(
                "📋 Delay Records"
            )

            st.dataframe(
                delay_df,
                use_container_width=True
            )

            # -----------------------------------------
            # DELAY BY TRAIN
            # -----------------------------------------

            st.subheader(
                "🚆 Delay by Train"
            )

            train_delay = (
                delay_df
                .groupby("Train No")[
                    "Delay Minutes"
                ]
                .mean()
                .reset_index()
            )

            st.bar_chart(
                train_delay.set_index(
                    "Train No"
                )
            )

            # -----------------------------------------
            # DELAY REASONS
            # -----------------------------------------

            st.subheader(
                "⚠️ Delay Reasons"
            )

            reason_count = (
                delay_df["Reason"]
                .value_counts()
            )

            st.bar_chart(
                reason_count
            )

        else:

            st.warning(
                "No delay records available."
            )

        # ---------------------------------------------
        # PASSENGER FLOW
        # ---------------------------------------------

        st.markdown("---")

        st.subheader(
            "👥 Passenger Flow"
        )

        if not passenger_df.empty:

            st.dataframe(
                passenger_df,
                use_container_width=True
            )

            passenger_chart = (
                passenger_df
                .groupby("Station")[
                    "Passenger Count"
                ]
                .sum()
                .reset_index()
            )

            st.subheader(
                "👥 Passengers by Station"
            )

            st.bar_chart(
                passenger_chart.set_index(
                    "Station"
                )
            )

        else:

            st.warning(
                "No passenger flow records available."
            )

    except Exception as e:

        st.error(
            f"❌ Analytics Database Error: {e}"
        )


# =========================================================
# SIMULATION
# =========================================================

elif page == "Simulation":

    st.title(
        "🧠 Railway Digital Twin Simulation"
    )

    st.subheader(
        "🚆 Train Movement & Delay Impact Simulation"
    )

    st.info(
        "Simulation uses railway data from the MySQL "
        "database to estimate the impact of train delays."
    )

    try:

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT
                train_no,
                station_code,
                delay_minutes,
                reason,
                delay_date
            FROM delay_logs
        """)

        delay_data = cursor.fetchall()

        cursor.close()
        conn.close()

        if not delay_data:

            st.warning(
                "⚠️ No delay data available for simulation."
            )

            st.stop()

        delay_df = pd.DataFrame(
            delay_data,
            columns=[
                "Train No",
                "Station",
                "Delay Minutes",
                "Reason",
                "Date"
            ]
        )

        st.markdown("---")

        st.subheader(
            "⚙️ Simulation Parameters"
        )

        col1, col2 = st.columns(2)

        with col1:

            train_no = st.selectbox(
                "🚆 Select Train",
                sorted(
                    delay_df[
                        "Train No"
                    ].unique()
                )
            )

        with col2:

            extra_delay = st.slider(
                "⏱️ Simulated Additional Delay",
                min_value=0,
                max_value=60,
                value=10,
                step=5
            )

        train_data = delay_df[
            delay_df["Train No"] == train_no
        ]

        current_delay = int(
            train_data[
                "Delay Minutes"
            ].sum()
        )

        affected_stations = (
            train_data[
                "Station"
            ].nunique()
        )

        st.markdown("---")

        if st.button(
            "🚀 Run Simulation",
            use_container_width=True
        ):

            simulated_delay = (
                current_delay +
                extra_delay
            )

            if simulated_delay <= 15:

                impact = "Low"

            elif simulated_delay <= 30:

                impact = "Moderate"

            else:

                impact = "High"

            estimated_impact = (
                affected_stations +
                max(
                    1,
                    extra_delay // 15
                )
            )

            st.markdown("---")

            st.subheader(
                "📊 Simulation Results"
            )

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "🚆 Train",
                train_no
            )

            col2.metric(
                "Current Delay",
                f"{current_delay} min"
            )

            col3.metric(
                "Simulated Delay",
                f"{simulated_delay} min"
            )

            col4.metric(
                "Impact",
                impact
            )

            st.markdown("---")

            st.subheader(
                "🗺️ Affected Railway Network"
            )

            st.write(
                f"Estimated affected stations: "
                f"**{estimated_impact}**"
            )

            if impact == "Low":

                st.success(
                    "🟢 Railway network impact is "
                    "expected to remain low."
                )

            elif impact == "Moderate":

                st.warning(
                    "🟡 Moderate network disruption "
                    "may occur."
                )

            else:

                st.error(
                    "🔴 High network disruption detected. "
                    "Operational intervention may be required."
                )

            st.markdown("---")

            st.subheader(
                "📋 Train Delay History"
            )

            st.dataframe(
                train_data,
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"❌ Simulation Error: {e}"
        )


# =========================================================
# PREDICTION
# =========================================================

elif page == "Prediction":

    st.title(
        "🤖 Train Delay Prediction"
    )

    st.subheader(
        "Machine Learning Based Railway Delay Prediction"
    )

    try:

        model = joblib.load(
            "ml/train_delay_model.pkl"
        )

        station_encoder = joblib.load(
            "ml/station_encoder.pkl"
        )

        reason_encoder = joblib.load(
            "ml/reason_encoder.pkl"
        )

        st.success(
            "✅ ML Prediction Model Loaded Successfully"
        )

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT train_no FROM trains"
        )

        train_data = cursor.fetchall()

        cursor.execute("""
            SELECT DISTINCT station_code
            FROM delay_logs
        """)

        station_data = cursor.fetchall()

        cursor.execute("""
            SELECT DISTINCT reason
            FROM delay_logs
        """)

        reason_data = cursor.fetchall()

        cursor.close()
        conn.close()

        train_numbers = [
            row[0]
            for row in train_data
        ]

        stations = [
            row[0]
            for row in station_data
        ]

        reasons = [
            row[0]
            for row in reason_data
        ]

        if not train_numbers:

            st.warning(
                "No trains available."
            )

            st.stop()

        st.markdown("---")

        st.subheader(
            "🔮 Enter Prediction Details"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            train_no = st.selectbox(
                "🚆 Select Train",
                train_numbers
            )

        with col2:

            station_code = st.selectbox(
                "🚉 Select Station",
                stations
            )

        with col3:

            reason = st.selectbox(
                "⚠️ Delay Reason",
                reasons
            )

        st.markdown("---")

        if st.button(
            "🔮 Predict Train Delay",
            use_container_width=True
        ):

            try:

                station_encoded = (
                    station_encoder.transform(
                        [station_code]
                    )[0]
                )

                reason_encoded = (
                    reason_encoder.transform(
                        [reason]
                    )[0]
                )

            except ValueError:

                st.error(
                    "❌ Selected station or reason "
                    "was not present during model training."
                )

                st.stop()

            input_data = pd.DataFrame(
                [[
                    train_no,
                    station_encoded,
                    reason_encoded
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

            prediction = max(
                0,
                round(
                    float(prediction),
                    2
                )
            )

            st.markdown("---")

            st.subheader(
                "🎯 Prediction Result"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Train",
                train_no
            )

            col2.metric(
                "Station",
                station_code
            )

            col3.metric(
                "Predicted Delay",
                f"{prediction} min"
            )

            if prediction <= 5:

                st.success(
                    "🟢 Train is expected to run "
                    "almost on time."
                )

            elif prediction <= 15:

                st.warning(
                    "🟡 Moderate delay is expected."
                )

            else:

                st.error(
                    "🔴 Significant delay is expected."
                )

    except Exception as e:

        st.error(
            f"❌ Prediction Error: {e}"
        )


# =========================================================
# ML EVALUATION
# =========================================================

elif page == "ML Evaluation":

    st.title(
        "📊 Machine Learning Model Evaluation"
    )

    st.subheader(
        "Train Delay Prediction Model Performance"
    )

    st.info(
        "This module displays the performance of the "
        "trained Random Forest train delay prediction model."
    )

    try:

        # =================================================
        # MODEL FILES
        # =================================================

        model_path = os.path.join(
            BASE_DIR,
            "ml",
            "train_delay_model.pkl"
        )

        metadata_path = os.path.join(
            BASE_DIR,
            "ml",
            "model_metadata.json"
        )

        if not os.path.exists(model_path):

            st.error(
                "❌ Trained ML model not found."
            )

            st.info(
                "Run ml/train_delay_model.py first."
            )

            st.stop()

        if not os.path.exists(metadata_path):

            st.error(
                "❌ Model metadata file not found."
            )

            st.info(
                "Run ml/train_delay_model.py first."
            )

            st.stop()


        # =================================================
        # LOAD SAVED MODEL + METADATA
        # =================================================

        model = joblib.load(
            model_path
        )

        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)


        # =================================================
        # MODEL INFORMATION
        # =================================================

        st.markdown("---")

        st.subheader(
            "🤖 Model Information"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Model Version",
            metadata.get(
                "model_version",
                "N/A"
            )
        )

        col2.metric(
            "Algorithm",
            metadata.get(
                "algorithm",
                "N/A"
            )
        )

        col3.metric(
            "Dataset Records",
            metadata.get(
                "dataset_records",
                "N/A"
            )
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Training Records",
            metadata.get(
                "training_records",
                "N/A"
            )
        )

        col2.metric(
            "Testing Records",
            metadata.get(
                "testing_records",
                "N/A"
            )
        )

        col3.metric(
            "Target",
            metadata.get(
                "target",
                "N/A"
            )
        )


        # =================================================
        # TRAINING INFORMATION
        # =================================================

        st.markdown("---")

        st.subheader(
            "🕒 Training Information"
        )

        st.write(
            f"**Model Name:** "
            f"{metadata.get('model_name', 'N/A')}"
        )

        st.write(
            f"**Training Timestamp:** "
            f"{metadata.get('training_timestamp', 'N/A')}"
        )


        # =================================================
        # PERFORMANCE METRICS
        # =================================================

        metrics = metadata.get(
            "metrics",
            {}
        )

        mae = metrics.get(
            "mae_minutes"
        )

        rmse = metrics.get(
            "rmse_minutes"
        )

        r2 = metrics.get(
            "r2_score"
        )

        st.markdown("---")

        st.subheader(
            "📈 Performance Metrics"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE",
            f"{mae:.2f} min"
            if mae is not None
            else "N/A"
        )

        col2.metric(
            "RMSE",
            f"{rmse:.2f} min"
            if rmse is not None
            else "N/A"
        )

        col3.metric(
            "R² Score",
            f"{r2:.3f}"
            if r2 is not None
            else "N/A"
        )


        # =================================================
        # MODEL FEATURES
        # =================================================

        st.markdown("---")

        st.subheader(
            "🧩 Model Features"
        )

        features = metadata.get(
            "features",
            []
        )

        if features:

            feature_df = pd.DataFrame(
                {
                    "Feature": features
                }
            )

            st.dataframe(
                feature_df,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No feature information available."
            )


        # =================================================
        # METRIC EXPLANATION
        # =================================================

        st.markdown("---")

        st.subheader(
            "📚 Metric Explanation"
        )

        st.write(
            "**MAE:** Average absolute difference "
            "between actual and predicted delay."
        )

        st.write(
            "**RMSE:** Measures prediction error "
            "with greater weight for larger errors."
        )

        st.write(
            "**R² Score:** Shows how well the model "
            "explains variation in delay data."
        )


        # =================================================
        # MODEL STATUS
        # =================================================

        st.markdown("---")

        st.subheader(
            "🟢 Model Status"
        )

        if r2 is not None and r2 >= 0.80:

            st.success(
                f"✅ Strong evaluation performance. "
                f"Current R² score: {r2:.4f}"
            )

        elif r2 is not None:

            st.warning(
                f"⚠️ Current R² score is {r2:.4f}. "
                "More training data may improve the model."
            )

        else:

            st.info(
                "ℹ️ R² score is not available."
            )


        # =================================================
        # DATASET LIMITATION
        # =================================================

        dataset_records = metadata.get(
            "dataset_records",
            0
        )

        if dataset_records < 30:

            st.warning(
                f"⚠️ Current dataset contains only "
                f"{dataset_records} records. "
                "These metrics are suitable for "
                "demonstration and should not be treated "
                "as production-level accuracy."
            )


        # =================================================
        # FINAL STATUS
        # =================================================

        st.markdown("---")

        st.success(
            "✅ ML evaluation loaded successfully "
            "from the saved model metadata."
        )


    except Exception as e:

        st.error(
            f"❌ ML Evaluation Error: {e}"
        )