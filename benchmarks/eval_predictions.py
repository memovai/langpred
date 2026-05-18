"""
Offline evaluation of the predictor: build synthetic trajectories, hold out
20% for evaluation, ingest the rest, ask for cost/ETA at prefix length 25%
of full length, report MAE and p90 calibration.

Run:
    python benchmarks/eval_predictions.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, os.path.join(ROOT, "sdk-python"))

import random
import statistics
import uuid
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from langpred_server.main import app
from langpred_server import db, predict
from synth_trajectories import SynthTrace, make_dataset  # type: ignore  # noqa: E402
from langpred_server.ml.pricing import price_step


def _ingest(client: TestClient, trace: SynthTrace, prefix: int | None = None) -> None:
    base = datetime.now(timezone.utc)
    steps = trace.steps if prefix is None else trace.steps[:prefix]
    events = [
        {
            "id": str(uuid.uuid4()),
            "timestamp": base.isoformat(),
            "type": "trace-create",
            "body": {"id": trace.trace_id, "name": trace.name, "timestamp": base.isoformat()},
        }
    ]
    for k, step in enumerate(steps):
        start = (base + timedelta(seconds=k)).isoformat()
        end = (base + timedelta(seconds=k + 1)).isoformat()
        events.append({
            "id": str(uuid.uuid4()),
            "timestamp": start,
            "type": "generation-create",
            "body": {
                "id": "obs-" + uuid.uuid4().hex,
                "traceId": trace.trace_id,
                "name": f"step_{k}",
                "model": step.model,
                "startTime": start,
                "endTime": end,
                "usage": {
                    "input": step.prompt_tokens,
                    "output": step.completion_tokens,
                    "total": step.prompt_tokens + step.completion_tokens,
                },
            },
        })
    if prefix is None:
        events.append({
            "id": str(uuid.uuid4()),
            "timestamp": (base + timedelta(seconds=len(steps) + 1)).isoformat(),
            "type": "trace-create",
            "body": {"id": trace.trace_id, "name": trace.name, "output": "done"},
        })
    client.post("/api/public/ingestion", json={"batch": events})


def _true_cost(trace: SynthTrace) -> float:
    return sum(price_step(s.model, s.prompt_tokens, s.completion_tokens) for s in trace.steps)


def main() -> None:
    random.seed(7)
    db.reset_store_for_tests()
    predict.reset_service_for_tests()

    dataset = make_dataset(n_per_shape=60)
    random.shuffle(dataset)
    split = int(len(dataset) * 0.8)
    train, test = dataset[:split], dataset[split:]

    with TestClient(app) as client:
        for traj in train:
            _ingest(client, traj)
        client.post("/api/public/predict/rebuild")

        abs_errors: list[float] = []
        in_p90: list[int] = []
        for traj in test:
            prefix = max(1, len(traj.steps) // 4)
            _ingest(client, traj, prefix=prefix)
            r = client.get(f"/api/public/predict/{traj.trace_id}/cost").json()
            true_cost = _true_cost(traj)
            err = abs(r["p50"] - true_cost)
            abs_errors.append(err)
            in_p90.append(1 if true_cost <= r["p90"] else 0)

        mae = statistics.fmean(abs_errors)
        p90_cov = statistics.fmean(in_p90)
        print(f"n_train={len(train)}  n_test={len(test)}")
        print(f"cost MAE          : ${mae:.4f}")
        print(f"true cost ≤ p90   : {p90_cov:.2%}   (target ≈ 90%)")


if __name__ == "__main__":
    main()
