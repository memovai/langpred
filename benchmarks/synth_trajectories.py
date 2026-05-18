"""Generate synthetic trajectories that look agent-shaped: a noisy step count
with cost scaling roughly linearly per step, with two trace-name buckets."""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass


@dataclass
class SynthStep:
    model: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class SynthTrace:
    trace_id: str
    name: str
    steps: list[SynthStep]
    final_status: str = "ok"


def _step(model: str) -> SynthStep:
    base_prompt = {
        "claude-sonnet-4-6": (400, 1200),
        "claude-haiku-4-5": (200, 800),
        "claude-opus-4-7": (800, 2000),
    }[model]
    base_completion = {
        "claude-sonnet-4-6": (100, 400),
        "claude-haiku-4-5": (50, 200),
        "claude-opus-4-7": (200, 800),
    }[model]
    return SynthStep(
        model=model,
        prompt_tokens=random.randint(*base_prompt),
        completion_tokens=random.randint(*base_completion),
    )


def make_dataset(n_per_shape: int = 50) -> list[SynthTrace]:
    out: list[SynthTrace] = []
    for _ in range(n_per_shape):
        # "research_agent" shape: 4-10 sonnet steps
        n = random.randint(4, 10)
        out.append(
            SynthTrace(
                trace_id="t-" + uuid.uuid4().hex,
                name="research_agent",
                steps=[_step("claude-sonnet-4-6") for _ in range(n)],
            )
        )
        # "refactor_repo" shape: 8-25 mixed sonnet/opus steps
        n = random.randint(8, 25)
        steps = []
        for _ in range(n):
            steps.append(_step(random.choice(["claude-sonnet-4-6", "claude-opus-4-7"])))
        out.append(
            SynthTrace(
                trace_id="t-" + uuid.uuid4().hex,
                name="refactor_repo",
                steps=steps,
            )
        )
    return out
