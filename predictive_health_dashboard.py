import random
from datetime import datetime, timedelta

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


LOG_FILE = "server_logs.csv"
ALERT_THRESHOLD = 0.75


def create_sample_logs(file_name):
    """Create beginner-friendly sample server logs if no real log file exists."""
    rows = []
    start_time = datetime.now() - timedelta(minutes=240)

    cpu = 35
    memory = 45

    for minute in range(240):
        timestamp = start_time + timedelta(minutes=minute)

        cpu += random.uniform(-3, 4)
        memory += random.uniform(-2, 3)

        if minute > 170:
            cpu += random.uniform(0, 2.8)
            memory += random.uniform(0, 2.2)

        cpu_usage = min(max(cpu, 5), 99)
        memory_usage = min(max(memory, 10), 99)
        network_io = random.uniform(80, 600)
        disk_io = random.uniform(40, 350)

        failure_soon = int(
            cpu_usage > 88
            or memory_usage > 90
            or (cpu_usage > 78 and memory_usage > 82)
        )

        rows.append(
            {
                "timestamp": timestamp,
                "cpu_usage": round(cpu_usage, 2),
                "memory_usage": round(memory_usage, 2),
                "network_io": round(network_io, 2),
                "disk_io": round(disk_io, 2),
                "failure_soon": failure_soon,
            }
        )

    pd.DataFrame(rows).to_csv(file_name, index=False)
    print(f"Created sample log file: {file_name}")


def load_and_clean_logs(file_name):
    """Read logs, sort them by time, and handle missing values."""
    data = pd.read_csv(file_name)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")

    numeric_columns = ["cpu_usage", "memory_usage", "network_io", "disk_io"]
    data[numeric_columns] = data[numeric_columns].ffill()
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    return data


def add_features(data):
    """Create simple time-series features from recent system behavior."""
    data["cpu_5min_avg"] = data["cpu_usage"].rolling(window=5, min_periods=1).mean()
    data["memory_5min_avg"] = data["memory_usage"].rolling(window=5, min_periods=1).mean()
    data["network_5min_avg"] = data["network_io"].rolling(window=5, min_periods=1).mean()
    data["cpu_5min_variance"] = data["cpu_usage"].rolling(window=5, min_periods=1).var()
    data["memory_5min_variance"] = data["memory_usage"].rolling(window=5, min_periods=1).var()

    data = data.fillna(0)
    return data


def train_model(data):
    """Train a simple classifier that predicts whether failure may happen soon."""
    feature_columns = [
        "cpu_usage",
        "memory_usage",
        "network_io",
        "disk_io",
        "cpu_5min_avg",
        "memory_5min_avg",
        "network_5min_avg",
        "cpu_5min_variance",
        "memory_5min_variance",
    ]

    x = data[feature_columns]
    y = data["failure_soon"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(n_estimators=80, random_state=42)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    return model, feature_columns, accuracy


def show_recent_predictions(model, data, feature_columns):
    """Print the latest system health score and alert if risk is high."""
    recent_rows = data.tail(10).copy()
    probabilities = model.predict_proba(recent_rows[feature_columns])[:, 1]

    print("\nRecent System Health Predictions")
    print("-" * 72)

    for row, probability in zip(recent_rows.to_dict("records"), probabilities):
        status = "CRITICAL ALERT" if probability >= ALERT_THRESHOLD else "Normal"

        print(
            f"{row['timestamp']} | "
            f"CPU: {row['cpu_usage']:5.1f}% | "
            f"Memory: {row['memory_usage']:5.1f}% | "
            f"Failure Probability: {probability * 100:5.1f}% | "
            f"{status}"
        )

        if probability >= ALERT_THRESHOLD:
            trigger_alert(row, probability)


def trigger_alert(row, probability):
    """This is where a real system could reroute traffic or notify engineers."""
    print(
        "  Action: Proactive override triggered. "
        f"Server risk is {probability * 100:.1f}%."
    )


def main():
    try:
        open(LOG_FILE, "r").close()
    except FileNotFoundError:
        create_sample_logs(LOG_FILE)

    logs = load_and_clean_logs(LOG_FILE)
    logs = add_features(logs)

    model, feature_columns, accuracy = train_model(logs)

    print("\nPredictive System Health Dashboard")
    print("=" * 72)
    print(f"Rows processed: {len(logs)}")
    print(f"Model accuracy on test data: {accuracy * 100:.2f}%")

    show_recent_predictions(model, logs, feature_columns)


if __name__ == "__main__":
    main()
