# 🚆 RailSphere

## AI-Powered Railway Intelligence & Simulation Platform

RailSphere is a Python-based railway intelligence platform that combines railway network visualization, MySQL database management, analytics, simulation, and machine learning-based train delay prediction into a single interactive application.

The project demonstrates how database systems, data analytics, machine learning, and interactive visualization can be integrated into a railway-focused software platform.

---

## ✨ Key Features

### 🚉 Railway Network Dashboard

- Total stations, trains, routes and platforms
- Interactive railway network map
- Station markers with location information
- Train routes visualized on the map

### 🚆 Railway Data Management

- Station information
- Train information
- Route information
- Platform information
- Railway delay records

### 📊 Analytics

- Railway network statistics
- Data-driven operational insights
- Interactive analytical views
- Railway data exploration

### 🎯 Train Delay Prediction

RailSphere uses a **Random Forest Regressor** to predict expected train delay based on:

- Train number
- Station code
- Delay reason

The prediction system validates station and delay-reason inputs before generating predictions.

### 🔬 Machine Learning Evaluation

The application provides model evaluation metrics including:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### 🧪 Railway Simulation

The simulation module allows railway-related scenarios and operational data to be explored through an interactive interface.

### ✅ Automated Testing

The project includes automated tests for:

- Database connectivity
- Machine learning model
- Prediction output
- Prediction service validation

---

## 🤖 Machine Learning

### Algorithm

**Random Forest Regressor**

### Input Features

```text
train_no
station_code
reason