# Predicting EV Charging Station Demand: A KDD Approach

**Course:** Predictive Analytics, CA3 (End-to-End Predictive Model Development)
**Domain:** Electric-vehicle infrastructure planning / smart mobility

## Problem Statement

Charging operators and city planners must decide where to build EV charging stations and how to configure them (charger type, capacity, price, opening hours, parking). A badly placed station sits idle, and an under-provisioned one causes queues.

**Objective:** predict a station's demand, its **average number of users per day**, from its location, technical configuration, pricing and service attributes.

- **Primary task:** regression on `Usage Stats (avg users/day)`
- **Secondary task:** binary classification: is a station *high-demand* (usage ≥ median)?
- **Success criterion:** beat a naive mean or majority baseline on unseen test data

## Dataset

| | |
|---|---|
| File | `detailed_ev_charging_stations.csv` (included in this repo) |
| Source | *Detailed EV Charging Stations* dataset, Kaggle (2025) |
| Size | 5,000 stations × 17 attributes, no missing values |
| Attributes | Station ID, latitude/longitude, address, charger type, cost (USD/kWh), availability, distance to city, **usage (avg users/day)**, operator, charging capacity (kW), connector types, installation year, renewable energy source, rating, parking spots, maintenance frequency |

## Methodology — KDD Process

| Step | What was done |
|---|---|
| **1. Data selection** | Load the dataset and inspect it with `head`, `info` and `describe` |
| **2. Pre-processing** | Check missing values and duplicates (none), outliers with box plots (none), and data quality (charger type vs capacity, cost by type); EDA with a histogram, correlation heatmap and bar charts |
| **3. Transformation** | Drop ID, address and coordinates; create *Station Age*; one-hot encode categorical columns; 80/20 train/test split; StandardScaler |
| **4. Data mining** | Train Linear Regression, KNN, Decision Tree and Random Forest, compared with a mean baseline |
| **5. Evaluation** | R², MAE, RMSE; actual-vs-predicted plot; Random Forest feature importance; conclusion |

## Key Results

| Model | R² | MAE | RMSE |
|---|---|---|---|
| Baseline (mean) | -0.001 | 22.81 | 26.43 |
| Linear Regression | -0.002 | 22.81 | 26.44 |
| KNN | -0.040 | 23.03 | 26.94 |
| Decision Tree | -0.034 | 22.95 | 26.85 |
| **Random Forest** | **-0.000** | **22.75** | **26.42** |

**Main finding:** no model beats the simple average baseline in any meaningful way, so the features cannot predict station usage. The data quality checks show why: the dataset appears randomly generated (AC Level 1 chargers at 350 kW, identical prices for every charger type, placeholder addresses). The KDD process therefore shows that better, real-world data is needed. The same steps can be reused on a real charging-session dataset.

## Repository Structure

```
.
├── main.ipynb                          # KDD implementation, with outputs
├── detailed_ev_charging_stations.csv   # dataset
├── requirements.txt                    # Python dependencies
└── README.md
```

## How to Run

```bash
git clone <this-repo-url>
cd <repo-folder>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook main.ipynb      # then Kernel → Restart & Run All
```

The notebook runs in a few seconds. `random_state=42` is used everywhere, so results are reproducible.

**Tested with:** Python 3.13, pandas 3.0, NumPy 2.5, scikit-learn 1.9, Matplotlib 3.11.
