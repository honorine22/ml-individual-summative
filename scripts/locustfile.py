from __future__ import annotations

from pathlib import Path

from locust import HttpUser, between, task

SAMPLE_AUDIO = (Path(__file__).resolve().parents[1] / "data/test/electrical_fault/1-21935-A-38.wav").resolve()


class FaultSenseUser(HttpUser):
    wait_time = between(0.5, 2)

    @task
    def predict(self):
        if not SAMPLE_AUDIO.exists():
            return
        files = {"file": (SAMPLE_AUDIO.name, SAMPLE_AUDIO.open("rb"), "audio/wav")}
        self.client.post("/predict", files=files)

