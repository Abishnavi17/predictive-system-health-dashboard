# Predictive System Health Dashboard

A simple data engineering and machine learning pipeline built in Python. This project reads simulated server performance metrics (like CPU and memory usage) and predicts if a system crash or failure is likely to happen soon.

## 🌟 Key Features
* **Automatic Log Generator:** Automatically creates a simulated server log file (`server_logs.csv`) with realistic metrics if no data exists yet.
* **Smart Data Cleaning:** Uses Pandas to automatically clean up missing values and calculate rolling 5-minute behavior trends (like moving averages and volatility spikes).
* **Machine Learning Brain:** Trains a Random Forest Classifier using Scikit-Learn to spot patterns that lead up to system failures.
* **Proactive Live Alerts:** Calculates a real-time "Failure Probability Score." If the probability crosses 75%, it automatically triggers a critical warning alert.

## 🛠️ Requirements & Tech Stack
You only need Python installed along with two core data science libraries:
* **Pandas** (For reading, cleaning, and structuring data metrics)
* **Scikit-Learn** (For training the predictive machine learning model)

## 📁 File Structure
* `predictive_health_dashboard.py` — The main script that creates data, trains the model, and displays alerts.

## 🚀 How to Run It Locally

1. Clone this repository to your computer:
```bash
git clone [https://github.com/Abishnavi17/predictive-system-health-dashboard.git](https://github.com/Abishnavi17/predictive-system-health-dashboard.git)
cd predictive-system-health-dashboard
