"""Create latency plots from Locust CSV output."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

REPORT_DIR = Path("reports/locust")
stats_history = REPORT_DIR / "faultsense_stats_history.csv"
if not stats_history.exists():
    raise FileNotFoundError("Run locust with --csv=reports/locust/faultsense before plotting")

history = pd.read_csv(stats_history)
http_history = history[history["Name"] == "/predict"]

plt.figure(figsize=(8, 3))
plt.plot(http_history["Timestamp"], http_history["95%"], label="p95 latency")
plt.xlabel("Timestamp")
plt.ylabel("Latency (ms)")
plt.title("Locust /predict latency (p95)")
plt.grid(alpha=0.2)
plt.legend()
plt.tight_layout()
plt.savefig(REPORT_DIR / "locust_latency.png", dpi=200)
print("Saved reports/locust/locust_latency.png")
