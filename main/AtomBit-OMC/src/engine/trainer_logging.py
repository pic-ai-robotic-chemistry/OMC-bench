import csv
import os


TRAIN_LOG_HEADERS = [
    "epoch", "step", "lr",
    "total_loss", "loss_e", "loss_f", "loss_s",
    "mae_e", "mae_f", "mae_s_gpa",
    "finite_pred_e", "finite_pred_f", "finite_pred_stress",
    "finite_loss_e", "finite_loss_f", "finite_loss_s", "finite_total_loss",
    "finite_grad", "finite_param", "grad_norm",
]

PERF_LOG_HEADERS = "step,memory_mb,time_sec,throughput_graphs_s,batch_size_graphs,num_atoms\n"


def init_csv_logs(*paths: str):
    for path in paths:
        with open(path, "w", newline="") as handle:
            csv.writer(handle).writerow(TRAIN_LOG_HEADERS)


def append_metrics_row(path: str, data: dict):
    with open(path, "a", newline="") as handle:
        csv.writer(handle).writerow([
            data["epoch"], data["step"], f"{data['lr']:.2e}",
            f"{data['total_loss']:.6f}", f"{data['loss_e']:.6f}",
            f"{data['loss_f']:.6f}", f"{data['loss_s']:.6f}",
            f"{data['mae_e'] * 1000:.6f}", f"{data['mae_f'] * 1000:.6f}", f"{data['mae_s_gpa']:.6f}",
            int(data.get("finite_pred_e", True)),
            int(data.get("finite_pred_f", True)),
            int(data.get("finite_pred_stress", True)),
            int(data.get("finite_loss_e", True)),
            int(data.get("finite_loss_f", True)),
            int(data.get("finite_loss_s", True)),
            int(data.get("finite_total_loss", True)),
            int(data.get("finite_grad", True)),
            int(data.get("finite_param", True)),
            f"{data.get('grad_norm', 0.0):.6f}",
        ])


def ensure_perf_log(path: str):
    if not os.path.exists(path):
        with open(path, "w") as handle:
            handle.write(PERF_LOG_HEADERS)


def append_perf_row(
    path: str,
    step: int,
    memory_mb: float,
    step_duration: float,
    throughput: float,
    batch_size_graphs: int,
    num_atoms: int,
):
    with open(path, "a") as handle:
        handle.write(
            f"{step},{memory_mb:.2f},{step_duration:.4f},{throughput:.2f},{batch_size_graphs},{num_atoms}\n"
        )
