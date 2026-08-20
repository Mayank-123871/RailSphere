\# 🚆 RailSphere



\### AI-Powered Railway Intelligence, Delay Prediction \& Digital Twin Simulation Platform



RailSphere is a railway intelligence platform designed to combine railway network data, database management, machine learning and simulation into a single interactive application.



The platform provides railway network visualization, train and station information, operational analytics, train delay prediction and railway digital twin simulation.



\---



\## 🎯 Project Overview



Railway operations involve multiple interconnected components such as trains, stations, routes, platforms and delay events.



RailSphere provides a centralized platform to explore these components and analyze railway operations using data-driven techniques.



The system combines:



\- Railway database management

\- Interactive railway network visualization

\- Operational analytics

\- Train delay prediction using Machine Learning

\- Railway network simulation

\- ML model evaluation

\- Automated testing



\---



\## ✨ Key Features



\### 🗺️ Interactive Railway Network



\- Interactive map-based railway visualization

\- Station markers with station information

\- Train route visualization

\- Geographic railway network exploration



\### 🚆 Railway Management



\- Station information

\- Train information

\- Route information

\- Platform information

\- Railway operational data



\### 📊 Analytics



\- Railway operational statistics

\- Delay analysis

\- Train and route insights

\- Data-driven railway monitoring



\### 🧠 Digital Twin Simulation



RailSphere includes a railway network simulation module that models train movement and delay propagation across railway routes.



The simulation helps visualize how delays can affect connected railway operations.



\### 🤖 Train Delay Prediction



RailSphere uses a Machine Learning model to estimate train delays.



The prediction pipeline uses:



\- Train Number

\- Station Code

\- Delay Reason



The target variable is:



\- Delay Minutes



\### 📈 ML Model Evaluation



The application provides:



\- MAE

\- RMSE

\- R² Score

\- Actual vs Predicted comparison

\- Model information

\- Feature importance



\### 🧪 Automated Testing



The project includes automated tests for:



\- Database connectivity

\- ML model

\- Prediction output

\- Prediction service validation



\---



\# 🤖 Machine Learning



\## Model



\*\*Algorithm:\*\* Random Forest Regressor



Random Forest was selected because it can model non-linear relationships between railway features and delay values while providing feature importance information.



\### Input Features



| Feature | Description |

|---|---|

| `train\_no` | Train number |

| `station\_code` | Railway station code |

| `reason` | Delay reason |



\### Target



`delay\_minutes`



\---



\## 📈 Model Performance



Current prototype evaluation:



| Metric | Result |

|---|---:|

| MAE | 2.1358 minutes |

| RMSE | 2.6111 minutes |

| R² Score | 0.8848 |



\### Dataset



| Parameter | Value |

|---|---:|

| Total Records | 20 |

| Training Records | 16 |

| Testing Records | 4 |

| Model Version | 1.0.0 |



> \*\*Note:\*\* The current dataset is intentionally small and is suitable for prototype demonstration and system validation. The reported metrics should not be interpreted as production-level model accuracy.



\---



\# 🏗️ System Architecture



```text

&#x20;                   ┌─────────────────────┐

&#x20;                   │       User          │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │    Streamlit UI     │

&#x20;                   │      RailSphere      │

&#x20;                   └──────────┬──────────┘

&#x20;                              │

&#x20;             ┌────────────────┼────────────────┐

&#x20;             │                │                │

&#x20;             ▼                ▼                ▼

&#x20;      ┌─────────────┐  ┌─────────────┐  ┌─────────────┐

&#x20;      │  Analytics  │  │ Prediction  │  │ Simulation  │

&#x20;      └──────┬──────┘  └──────┬──────┘  └──────┬──────┘

&#x20;             │                │                │

&#x20;             │                ▼                │

&#x20;             │       ┌───────────────┐        │

&#x20;             │       │ Random Forest │        │

&#x20;             │       │     Model     │        │

&#x20;             │       └───────────────┘        │

&#x20;             │                │                │

&#x20;             └────────────────┼────────────────┘

&#x20;                              ▼

&#x20;                   ┌─────────────────────┐

&#x20;                   │      MySQL DB       │

&#x20;                   └─────────────────────┘

