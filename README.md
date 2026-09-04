# Predictive System Health Dashboard

A data-engineering and machine-learning pipeline in Python that reads server
performance metrics (CPU, memory, and related signals), learns the patterns that
precede failures, and raises a real-time alert when a crash becomes likely.

## Features

- **Automatic log generation** — if no data exists, it creates a simulated
  server-log file (`server_logs.csv`) with realistic metrics, so the pipeline
  runs end to end out of the box.
- **Feature engineering** — uses Pandas to clean missing values and compute
  rolling five-minute trends such as moving averages and volatility, capturing
  how the system behaves over time rather than at a single instant.
- **Failure prediction** — trains a Random Forest classifier (Scikit-Learn) to
  recognise the metric patterns that tend to lead up to a system failure.
- **Real-time alerting** — computes a live failure-probability score and triggers
  a critical warning when it crosses 75%.

## Tech stack

| Tool | Role |
|------|------|
| Python | core language |
| Pandas | reading, cleaning, and feature-engineering the metrics |
| Scikit-Learn | training and running the Random Forest classifier |

## Project structure

```
predictive-system-health-dashboard/
└── predictive_health_dashboard.py   generates data, trains the model, raises alerts
```

## Getting started

**1. Clone the repository:**

```bash
git clone https://github.com/Abishnavi17/predictive-system-health-dashboard.git
cd predictive-system-health-dashboard
```

**2. Install the dependencies:**

```bash
pip install pandas scikit-learn
```

**3. Run the pipeline:**

```bash
python predictive_health_dashboard.py
```

On first run it generates `server_logs.csv` if none exists, engineers features,
trains the model, and prints failure-probability scores with alerts.

## How it works

1. **Data** — loads `server_logs.csv`, or generates a realistic simulated log if
   the file is missing.
2. **Feature engineering** — cleans the data and derives rolling five-minute
   statistics (moving averages, volatility) so the model sees temporal trends.
3. **Model** — trains a Random Forest classifier on those features to distinguish
   normal operation from the lead-up to a failure.
4. **Alerting** — scores incoming metrics in real time and emits a critical
   warning whenever the predicted failure probability exceeds 75%.

## Notes and limitations

- The metrics are **simulated**, so the model demonstrates the pipeline and
  method rather than production accuracy; the same pipeline would apply to real
  server logs.
- The 75% alert threshold is configurable and would be tuned against real
  incident data to balance false alarms against missed failures.
