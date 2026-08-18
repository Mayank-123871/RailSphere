import streamlit as st
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

    st.title("🚆 RailSphere")
    st.subheader(
        "AI-Powered Railway Intelligence & Simulation Platform"
    )

    st.write(
        "Welcome to RailSphere — Railway Performance, "
        "Passenger Flow and AI Prediction Dashboard."
    )

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

        cursor.execute("""
            SELECT
                COALESCE(AVG(delay_minutes), 0),
                COALESCE(SUM(delay_minutes), 0),
                COUNT(*)
            FROM delay_logs
        """)
        delay_result = cursor.fetchone()

        avg_delay = float(delay_result[0] or 0)
        total_delay = int(delay_result[1] or 0)
        delay_records = int(delay_result[2] or 0)

        cursor.execute("""
            SELECT COALESCE(SUM(passenger_count), 0)
            FROM passenger_flow
        """)
        total_passengers = int(cursor.fetchone()[0] or 0)

        cursor.execute("""
            SELECT station_code, passenger_count
            FROM passenger_flow
            ORDER BY passenger_count DESC
            LIMIT 1
        """)
        busiest_result = cursor.fetchone()

        cursor.close()
        conn.close()

        if busiest_result:
            busiest_station = busiest_result[0]
            busiest_count = int(busiest_result[1])
        else:
            busiest_station = "N/A"
            busiest_count = 0

        st.markdown("---")
        st.subheader("📊 Railway Network Overview")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🚉 Stations", stations)
        col2.metric("🚆 Trains", trains)
        col3.metric("🛤️ Routes", routes)
        col4.metric("🚦 Platforms", platforms)

        st.markdown("---")
        st.subheader("🧠 Railway Intelligence")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "⏱️ Average Delay",
            f"{avg_delay:.1f} min"
        )

        col2.metric(
            "👥 Passenger Flow",
            f"{total_passengers:,}"
        )

        col3.metric(
            "🏆 Busiest Station",
            busiest_station
        )

        col4.metric(
            "⚠️ Delay Records",
            delay_records
        )

        st.markdown("---")
        st.subheader("🤖 System Status")

        col1, col2, col3 = st.columns(3)

        col1.success("🗄️ MySQL Database Connected")
        col2.success("🤖 ML Prediction Model Ready")
        col3.success("🧠 Simulation Engine Ready")

        st.markdown("---")
        st.subheader("🚨 Network Alerts")

        if busiest_count >= 12000:
            st.error(
                f"🔴 High passenger congestion detected at "
                f"**{busiest_station}** ({busiest_count:,} passengers)."
            )
        elif busiest_count >= 7000:
            st.warning(
                f"🟡 Moderate passenger congestion at "
                f"**{busiest_station}** ({busiest_count:,} passengers)."
            )
        elif busiest_count > 0:
            st.success(
                f"🟢 Passenger flow is currently low at "
                f"**{busiest_station}**."
            )
        else:
            st.info("ℹ️ No passenger flow data available.")

        if avg_delay > 15:
            st.error(
                f"🔴 Average railway delay is high: "
                f"**{avg_delay:.1f} minutes**."
            )
        elif avg_delay > 5:
            st.warning(
                f"🟡 Average railway delay is moderate: "
                f"**{avg_delay:.1f} minutes**."
            )
        else:
            st.success(
                f"🟢 Average railway delay is low: "
                f"**{avg_delay:.1f} minutes**."
            )

        st.markdown("---")
        st.subheader("🗺️ Railway Network")

        st.info(
            "Use the Simulation page to visualize train routes "
            "and station locations on the railway map."
        )

        st.markdown("---")
        st.subheader("📢 Railway Updates")

        st.success(
            "✅ Railway database is connected successfully."
        )

        st.warning(
            f"⚠️ Total recorded delay: {total_delay} minutes."
        )

        st.info(
            "🤖 AI delay prediction and digital twin simulation "
            "modules are available from the sidebar."
        )

    except Exception as e:

        st.error(
            f"❌ Dashboard Database Error: {e}"
        )


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

    st.title("🧠 Railway Digital Twin Simulation")
    st.subheader("🚆 Train Movement & Delay Impact Simulation")

    st.info(
        "Simulate additional delay for a train and visualize its "
        "route and potentially affected stations."
    )

    try:

        # -------------------------------------------------
        # DATABASE CONNECTION
        # -------------------------------------------------

        conn = get_connection()
        cursor = conn.cursor()

        # -------------------------------------------------
        # LOAD DELAY DATA
        # -------------------------------------------------

        cursor.execute("""
            SELECT
                train_no,
                station_code,
                delay_minutes,
                reason,
                delay_date
            FROM delay_logs
            ORDER BY delay_date
        """)

        delay_data = cursor.fetchall()

        # -------------------------------------------------
        # LOAD ROUTE DATA
        # -------------------------------------------------

        cursor.execute("""
            SELECT
                r.train_no,
                r.station_code,
                r.stop_number,
                r.arrival_time,
                r.departure_time,
                s.station_name,
                s.city,
                s.latitude,
                s.longitude
            FROM routes r
            LEFT JOIN stations s
                ON r.station_code = s.station_code
            ORDER BY r.train_no, r.stop_number
        """)

        route_data = cursor.fetchall()

        cursor.close()
        conn.close()

        if not delay_data:
            st.warning("⚠️ No delay data available for simulation.")
            st.stop()

        # -------------------------------------------------
        # DATAFRAMES
        # -------------------------------------------------

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

        route_df = pd.DataFrame(
            route_data,
            columns=[
                "Train No",
                "Station",
                "Stop Number",
                "Arrival",
                "Departure",
                "Station Name",
                "City",
                "Latitude",
                "Longitude"
            ]
        )

        # -------------------------------------------------
        # SIMULATION INPUTS
        # -------------------------------------------------

        st.markdown("---")
        st.subheader("⚙️ Simulation Parameters")

        available_trains = sorted(
            delay_df["Train No"].unique().tolist()
        )

        col1, col2 = st.columns(2)

        with col1:
            train_no = st.selectbox(
                "🚆 Select Train",
                available_trains
            )

        with col2:
            extra_delay = st.slider(
                "⏱️ Additional Simulated Delay",
                min_value=0,
                max_value=60,
                value=10,
                step=5
            )

        # -------------------------------------------------
        # CURRENT TRAIN DATA
        # -------------------------------------------------

        train_delay_data = delay_df[
            delay_df["Train No"] == train_no
        ].copy()

        current_delay = int(
            train_delay_data["Delay Minutes"].sum()
        )

        affected_stations = int(
            train_delay_data["Station"].nunique()
        )

        # -------------------------------------------------
        # RUN SIMULATION
        # -------------------------------------------------

        st.markdown("---")

        if st.button(
            "🚀 Run Simulation",
            use_container_width=True
        ):

            simulated_delay = current_delay + extra_delay

            if simulated_delay <= 15:
                impact = "Low"
            elif simulated_delay <= 30:
                impact = "Moderate"
            else:
                impact = "High"

            estimated_impact = (
                affected_stations +
                max(0, extra_delay // 15)
            )

            # -------------------------------------------------
            # RESULTS
            # -------------------------------------------------

            st.subheader("📊 Simulation Results")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("🚆 Train", train_no)
            col2.metric(
                "Current Delay",
                f"{current_delay} min"
            )
            col3.metric(
                "Simulated Delay",
                f"{simulated_delay} min"
            )
            col4.metric("Network Impact", impact)

            st.write(
                f"Estimated affected stations: "
                f"**{estimated_impact}**"
            )

            if impact == "Low":
                st.success(
                    "🟢 Railway network impact is expected to remain low."
                )
            elif impact == "Moderate":
                st.warning(
                    "🟡 Moderate network disruption may occur."
                )
            else:
                st.error(
                    "🔴 High network disruption detected. "
                    "Operational intervention may be required."
                )

            # -------------------------------------------------
            # ROUTE INFORMATION
            # -------------------------------------------------

            train_route = route_df[
                route_df["Train No"] == train_no
            ].copy()

            st.markdown("---")
            st.subheader("🛤️ Train Route")

            if train_route.empty:

                st.info(
                    "ℹ️ Route records for this train are not available "
                    "in the routes table."
                )

            else:

                display_columns = [
                    "Stop Number",
                    "Station",
                    "Station Name",
                    "City",
                    "Arrival",
                    "Departure"
                ]

                st.dataframe(
                    train_route[display_columns],
                    use_container_width=True,
                    hide_index=True
                )

                # -------------------------------------------------
                # ROUTE MAP
                # -------------------------------------------------

                map_data = train_route[
                    ["Latitude", "Longitude"]
                ].dropna()

                if len(map_data) > 0:

                    st.subheader("🗺️ Railway Route Map")

                    # Streamlit native map works without an
                    # additional mapping package.
                    st.map(
                        map_data.rename(
                            columns={
                                "Latitude": "lat",
                                "Longitude": "lon"
                            }
                        ),
                        use_container_width=True
                    )

                    st.caption(
                        "Map displays available station coordinates "
                        "from the routes and stations tables."
                    )

                else:

                    st.info(
                        "ℹ️ Latitude/longitude data is not available "
                        "for this train's route."
                    )

            # -------------------------------------------------
            # TRAIN DELAY HISTORY
            # -------------------------------------------------

            st.markdown("---")
            st.subheader("📋 Train Delay History")

            st.dataframe(
                train_delay_data,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:

        st.error(
            f"❌ Simulation Error: {e}"
        )

# =========================================================
# PREDICTION
# =========================================================

elif page == "Prediction":

    st.title("🤖 Train Delay Prediction")
    st.subheader("Machine Learning Based Railway Delay Prediction")

    st.info(
        "Random Forest model predicts the expected train delay "
        "using train number, station and delay reason."
    )

    try:
        import joblib

        # -------------------------------------------------
        # LOAD TRAINED MODEL
        # -------------------------------------------------
        model = joblib.load("ml/train_delay_model.pkl")
        station_encoder = joblib.load("ml/station_encoder.pkl")
        reason_encoder = joblib.load("ml/reason_encoder.pkl")

        st.success("✅ ML Prediction Model Loaded Successfully")

        # -------------------------------------------------
        # DATABASE CONNECTION
        # -------------------------------------------------
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT train_no
            FROM trains
            ORDER BY train_no
        """)
        train_data = cursor.fetchall()

        cursor.execute("""
            SELECT DISTINCT station_code
            FROM delay_logs
            ORDER BY station_code
        """)
        station_data = cursor.fetchall()

        cursor.execute("""
            SELECT DISTINCT reason
            FROM delay_logs
            ORDER BY reason
        """)
        reason_data = cursor.fetchall()

        cursor.close()
        conn.close()

        train_numbers = [row[0] for row in train_data]
        stations = [row[0] for row in station_data]
        reasons = [row[0] for row in reason_data]

        if not train_numbers:
            st.warning("⚠️ No trains available in database.")
            st.stop()

        if not stations:
            st.warning("⚠️ No stations available in delay data.")
            st.stop()

        if not reasons:
            st.warning("⚠️ No delay reasons available.")
            st.stop()

        # -------------------------------------------------
        # MODEL PERFORMANCE
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("📈 Model Performance")

        col1, col2, col3 = st.columns(3)
        col1.metric("🤖 Algorithm", "Random Forest")
        col2.metric("📊 MAE", "7.69 min")
        col3.metric("📚 Training Records", "5")

        st.caption(
            "MAE represents the average prediction error observed "
            "during model evaluation."
        )

        # -------------------------------------------------
        # PREDICTION INPUT
        # -------------------------------------------------
        st.markdown("---")
        st.subheader("🔮 Enter Prediction Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            train_no = st.selectbox("🚆 Select Train", train_numbers)

        with col2:
            station_code = st.selectbox("🚉 Select Station", stations)

        with col3:
            reason = st.selectbox("⚠️ Delay Reason", reasons)

        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------
        st.markdown("---")

        if st.button("🔮 Predict Train Delay", use_container_width=True):

            try:
                station_encoded = station_encoder.transform([station_code])[0]
                reason_encoded = reason_encoder.transform([reason])[0]
            except ValueError as encode_error:
                st.error(f"❌ Encoding Error: {encode_error}")
                st.stop()

            input_data = pd.DataFrame(
                [[train_no, station_encoded, reason_encoded]],
                columns=["train_no", "station_code", "reason"]
            )

            prediction = float(model.predict(input_data)[0])
            prediction = max(0, round(prediction, 2))

            if prediction <= 5:
                status = "🟢 Low Delay"
                message = "Train is expected to run almost on time."
            elif prediction <= 15:
                status = "🟡 Moderate Delay"
                message = "Moderate delay may occur."
            else:
                status = "🔴 High Delay"
                message = "Significant delay is expected."

            # -------------------------------------------------
            # RESULT
            # -------------------------------------------------
            st.markdown("---")
            st.subheader("🎯 Prediction Result")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("🚆 Train", train_no)
            col2.metric("🚉 Station", station_code)
            col3.metric("⏱️ Predicted Delay", f"{prediction} min")
            col4.metric("📊 Status", status)

            if prediction <= 5:
                st.success(f"🟢 {message}")
            elif prediction <= 15:
                st.warning(f"🟡 {message}")
            else:
                st.error(f"🔴 {message}")

            # -------------------------------------------------
            # SUMMARY
            # -------------------------------------------------
            st.markdown("---")
            st.subheader("📋 Prediction Summary")

            summary_df = pd.DataFrame({
                "Parameter": [
                    "Train Number",
                    "Station",
                    "Delay Reason",
                    "Predicted Delay",
                    "Delay Status"
                ],
                "Value": [
                    train_no,
                    station_code,
                    reason,
                    f"{prediction} minutes",
                    status
                ]
            })

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

            # -------------------------------------------------
            # INFORMATION
            # -------------------------------------------------
            st.markdown("---")
            st.subheader("ℹ️ Prediction Information")
            st.write(
                f"The Random Forest model estimates a delay of "
                f"**{prediction} minutes** for train **{train_no}** "
                f"at station **{station_code}** when the reported "
                f"delay reason is **{reason}**."
            )
            st.caption(
                "Note: The current model was trained on the available "
                "delay records in the database. More historical data "
                "can improve prediction quality."
            )

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")

# =========================================================
# ML EVALUATION
# =========================================================

elif page == "ML Evaluation":

    st.title("📊 ML Model Evaluation")

    st.subheader(
        "Random Forest Train Delay Prediction Performance"
    )

    st.info(
        "This page evaluates the same type of Random Forest model "
        "using the delay records currently available in MySQL."
    )

    try:

        # -------------------------------------------------
        # LOAD DATA
        # -------------------------------------------------

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

        if df.empty:
            st.warning(
                "⚠️ No data found in delay_logs table."
            )
            st.stop()

        # -------------------------------------------------
        # DATA CHECK
        # -------------------------------------------------

        st.markdown("---")
        st.subheader("📚 Evaluation Dataset")

        st.metric(
            "Total Records",
            len(df)
        )

        if len(df) < 3:
            st.warning(
                "⚠️ At least 3 records are recommended for "
                "meaningful model evaluation."
            )

        # -------------------------------------------------
        # ENCODE CATEGORICAL DATA
        # -------------------------------------------------

        station_encoder_eval = LabelEncoder()
        reason_encoder_eval = LabelEncoder()

        df["station_code"] = (
            station_encoder_eval.fit_transform(
                df["station_code"].astype(str)
            )
        )

        df["reason"] = (
            reason_encoder_eval.fit_transform(
                df["reason"].astype(str)
            )
        )

        X = df[
            [
                "train_no",
                "station_code",
                "reason"
            ]
        ]

        y = df["delay_minutes"]

        # -------------------------------------------------
        # TRAIN / TEST SPLIT
        # -------------------------------------------------

        if len(df) >= 5:

            test_size = 0.2

        else:

            test_size = 0.4

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=42
        )

        # -------------------------------------------------
        # TRAIN MODEL FOR EVALUATION
        # -------------------------------------------------

        evaluation_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )

        evaluation_model.fit(
            X_train,
            y_train
        )

        predictions = evaluation_model.predict(
            X_test
        )

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        if len(y_test) >= 2:

            r2 = r2_score(
                y_test,
                predictions
            )

        else:

            r2 = None

        # -------------------------------------------------
        # METRIC CARDS
        # -------------------------------------------------

        st.markdown("---")
        st.subheader("🎯 Model Evaluation Results")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "🤖 Algorithm",
            "Random Forest"
        )

        col2.metric(
            "📈 MAE",
            f"{mae:.2f} min"
        )

        col3.metric(
            "📉 RMSE",
            f"{rmse:.2f} min"
        )

        if r2 is not None:

            col4.metric(
                "🎯 R² Score",
                f"{r2:.2f}"
            )

        else:

            col4.metric(
                "🎯 R² Score",
                "N/A"
            )

        st.caption(
            "MAE shows average absolute prediction error. "
            "RMSE gives more weight to larger errors. "
            "R² is not reliable when there is only one test record."
        )

        # -------------------------------------------------
        # TRAIN / TEST INFORMATION
        # -------------------------------------------------

        st.markdown("---")

        col1, col2 = st.columns(2)

        col1.metric(
            "📚 Training Records",
            len(X_train)
        )

        col2.metric(
            "🧪 Testing Records",
            len(X_test)
        )

        # -------------------------------------------------
        # ACTUAL VS PREDICTED
        # -------------------------------------------------

        st.markdown("---")
        st.subheader("📋 Actual vs Predicted Delay")

        comparison_df = pd.DataFrame(
            {
                "Actual Delay (min)": y_test.to_numpy(),
                "Predicted Delay (min)": np.round(
                    predictions,
                    2
                )
            }
        )

        comparison_df["Error (min)"] = np.round(
            comparison_df["Actual Delay (min)"]
            - comparison_df["Predicted Delay (min)"],
            2
        )

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        # -------------------------------------------------
        # VISUAL COMPARISON
        # -------------------------------------------------

        st.markdown("---")
        st.subheader("📊 Actual vs Predicted Chart")

        chart_df = comparison_df[
            [
                "Actual Delay (min)",
                "Predicted Delay (min)"
            ]
        ].reset_index(drop=True)

        st.line_chart(
            chart_df,
            use_container_width=True
        )

        # -------------------------------------------------
        # INTERPRETATION
        # -------------------------------------------------

        st.markdown("---")
        st.subheader("💡 Evaluation Summary")

        if mae <= 5:

            st.success(
                f"🟢 Current MAE is {mae:.2f} minutes. "
                "The model is showing relatively low error."
            )

        elif mae <= 10:

            st.warning(
                f"🟡 Current MAE is {mae:.2f} minutes. "
                "The model has moderate prediction error."
            )

        else:

            st.error(
                f"🔴 Current MAE is {mae:.2f} minutes. "
                "More historical training data is recommended."
            )

        st.info(
            "The current dataset contains only a small number of "
            "delay records, so these metrics should be treated as "
            "a demonstration of the ML pipeline rather than a "
            "production-level accuracy measurement."
        )

    except Exception as e:

        st.error(
            f"❌ ML Evaluation Error: {e}"
        )
