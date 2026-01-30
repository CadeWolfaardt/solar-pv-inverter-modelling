# 📘 **Data-Driven Modelling of Solar PV Inverters**  
### *Digital Twin, Predictive Maintenance, Survival Analysis, Forecasting, and Anomaly Detection for Utility-Scale PV Systems*

**Authors:** Cade Wolfaardt, Payas Chatrath, Muhammad Ali, Trac Nguyen  
*Harvard University Extension School – Data Science Capstone (CSCI E-599a)*  

This repository contains the full implementation developed for the Harvard University Data Science Capstone project **Data-Driven Modelling of Solar PV Inverters**.  
The project builds a comprehensive analytical and modeling framework for monitoring, evaluating, and predicting the performance of solar photovoltaic (PV) inverters using real-world SCADA data.

The work integrates **five major components**:

---

## 🔹 1. Digital Twin (DT) Modeling
A data-driven hybrid Digital Twin capable of reconstructing and estimating inverter performance under varying environmental and operational conditions.

Includes:
- Physics-informed efficiency modeling  
- ML-based residual modeling  
- Feature engineering (POA, DC/AC metrics, thermal conditions, etc.)  
- Standardized, gap-filled, and cleaned data streams for downstream tasks  

---

## 🔹 2. Predictive Maintenance (PM)
A machine-learning pipeline designed to signal emerging inverter degradation and abnormal behaviour.

Key features:
- Ensemble models (XGBoost, Random Forest)  
- Density-based clustering (DBSCAN) for failure mode identification  
- Early-risk detection through engineered future-failure labels  

---

## 🔹 3. Survival Analysis
Statistical reliability modeling using:
- Kaplan–Meier estimators  
- Cox Proportional Hazards  
- Time-to-event analysis for inverter health under partially labeled conditions  

---

## 🔹 4. Time-Series Forecasting
Short-term and day-ahead power forecasting using:
- LSTM architectures  
- SARIMA  
- VAR with exogenous variables  
- Trend-decomposition approaches from the literature  

These methods support early-warning alerts for underperformance.

---

## 🔹 5. Anomaly Detection
A multi-signal anomaly detection framework combining:
- Digital Twin reconstruction errors  
- Power-curve deviation  
- Operational rule-based thresholds  

This system highlights potential issues such as sensor faults, panel soiling, environmental disruptions, or inverter misbehavior.

---

## 🛠 Technologies Used
- Python: Polars, NumPy, Pandas, Scikit-learn, XGBoost, TensorFlow/Keras  
- Machine Learning: LSTM, ensemble methods, linear models

---

## 🧩 Repository Structure

```.
├── src/
│   └── pv_inverter_modeling
│       ├── config/
│       │   ├── constants.py
│       │   ├── env.py
│       │   └── private_map.example.py
│       │
│       ├── data/
│       │   ├── loaders.py
│       │   ├── naming.py
│       │   └── schemas.py
│       │
│       ├── evaluation/
│       │   └── metrics.py
│       │
│       ├── models/
│       │   ├── forecasting.py
│       │   ├── io.py
│       │   ├── predictive_maintenance.py
│       │   └── survival_analysis.py
│       │
│       ├── preprocessing/
│       │   ├── astronomy.py
│       │   ├── interpolation.py
│       │   ├── outliers.py
│       │   └── reshape.py
│       │
│       ├── utils/
│       │   ├── attrs.py
│       │   ├── logging.py
│       │   ├── memory.py
│       │   ├── paths.py
│       │   ├── runtime.py
│       │   └── typing.py
│       │
│       ├── visualization/
│       │   └── timeseries.py
│       │
│       ├── __init__.py
│       └── py.typed
│
├── scripts/
│   ├── anomaly_detection.py
│   ├── failure_detection.py
│   ├── forecasting.py
│   ├── forecasting_decomposition_lstm.py
│   ├── predictive_maintenance.py
│   └── survival_analysis.py
│
├── .env.example
├── Data-Driven Modelling of Solar PV Inverters.pdf
├── pyproject.toml
└── README.md
```
---

## 🔒 Confidentiality Notice
This repository contains **no proprietary MN8 data**.  
All datasets have been removed or replaced with placeholders to comply with confidentiality requirements.  
Only reproducible code and methodological documentation are included.

The `.env.example` file contains placeholder values only.
All real site-specific configuration must be provided locally via a private `.env` file.

The `private_map.example.py` file contains placeholder values only.
All real site-specific configuration must be provided locally via a private `private_map.py` file.
---

## 📄 Citation
If you use this work or its components in research, please cite the capstone report.

---

