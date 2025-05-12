import mlflow
from prometheus_client import Gauge, start_http_server
import time

# Khởi tạo các metric Prometheus
mlflow_metric = Gauge('mlflow_metrics', 'Metrics from MLflow experiments and models', labelnames=['metric_name'])

def collect_metrics():
    """Collect MLflow data and update Prometheus metrics."""
    # Tổng số Registered Models
    registered_models = mlflow.search_registered_models()
    num_registered_models = len(registered_models)

    # Tổng số version của tất cả models
    num_model_versions = sum(len(m.latest_versions) for m in registered_models)

    # Lấy tất cả experiment
    experiments = mlflow.search_experiments()
    num_experiments = len(experiments)

    # Lấy toàn bộ runs của tất cả experiments
    all_runs = []
    for exp in experiments:
        runs = mlflow.search_runs([exp.experiment_id])
        all_runs.extend(runs.to_dict(orient="records"))

    num_runs = len(all_runs)

    # Thống kê trạng thái runs
    num_runs_finished = sum(1 for run in all_runs if run["status"] == "FINISHED")
    num_runs_failed = sum(1 for run in all_runs if run["status"] == "FAILED")

    # === Cập nhật các metric tổng quan ===
    mlflow_metric.labels(metric_name="num_registered_models").set(num_registered_models)
    mlflow_metric.labels(metric_name="num_model_versions").set(num_model_versions)
    mlflow_metric.labels(metric_name="num_experiments").set(num_experiments)
    mlflow_metric.labels(metric_name="num_runs").set(num_runs)
    mlflow_metric.labels(metric_name="num_runs_finished").set(num_runs_finished)
    mlflow_metric.labels(metric_name="num_runs_failed").set(num_runs_failed)

    # === Cập nhật các metric trung bình cho mỗi experiment ===
    for exp in experiments:
        experiment_runs = mlflow.search_runs([exp.experiment_id])
        experiment_all_runs = experiment_runs.to_dict(orient="records")

        # Tính toán các metric trung bình
        experiment_metric_sums = {
            "accuracy": 0.0,
            "f1_score": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "val_loss": 0.0
        }
        experiment_metric_counts = {
            "accuracy": 0,
            "f1_score": 0,
            "precision": 0,
            "recall": 0,
            "val_loss": 0
        }

        for run in experiment_all_runs:
            for metric in experiment_metric_sums.keys():
                key = f"metrics.{metric}"
                if key in run:
                    experiment_metric_sums[metric] += run[key]
                    experiment_metric_counts[metric] += 1

        # Tính trung bình cho mỗi metric
        for metric, total in experiment_metric_sums.items():
            avg_value = total / experiment_metric_counts[metric] if experiment_metric_counts[metric] > 0 else 0.0
            mlflow_metric.labels(metric_name=f"avg_{metric}_experiment_{exp.experiment_id}").set(avg_value)

# Bắt đầu server Prometheus tại cổng 8000
start_http_server(8000)

# Gọi hàm collect_metrics để thu thập dữ liệu và cập nhật các metric
collect_metrics()

# Nếu muốn script chạy liên tục, có thể cho nó chạy trong một vòng lặp (ví dụ, mỗi phút)
while True:
    collect_metrics()
    time.sleep(60)  # Cập nhật dữ liệu mỗi 60 giây
