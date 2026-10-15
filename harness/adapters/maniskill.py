from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from harness.adapters.base import AdapterEvaluation, AdapterRunOutput, TaskAdapter
from harness.adapters.common import load_json_records, resolve_workspace_path, run_reasoning_harness
from harness.adapters.knowledge import (
    AdapterCapability,
    EvaluationKnowledge,
    MetricDirection,
    MetricKnowledge,
    ProjectKnowledge,
)
from harness.models import ExperimentConfig, Metric
from harnesses.contracts import Harness


@dataclass
class _ReachEnvironment:
    position: int
    target: int
    max_steps: int
    steps: int = 0

    def step(self, action: int) -> tuple[int, float, bool]:
        self.position += 1 if action > 0 else -1 if action < 0 else 0
        self.steps += 1
        success = self.position == self.target
        return self.position, 1.0 if success else -0.05, success


class ManiSkillAdapter(TaskAdapter):
    """One-environment embodied loop with a headless deterministic fallback."""

    def load_tasks(self, config: ExperimentConfig) -> list[dict[str, Any]]:
        dataset = config.evaluation.dataset or "evals/maniskill_smoke_v1.json"
        return load_json_records(resolve_workspace_path(config, dataset))

    def setup(self, task: dict[str, Any], config: ExperimentConfig) -> None:
        if config.adapter.parameters.get("mode", "smoke") == "real":
            try:
                import gymnasium as gym  # noqa: F401
                import mani_skill  # noqa: F401
            except ImportError as error:
                raise RuntimeError(
                    "Real ManiSkill mode requires: pip install -e '.[maniskill]'"
                ) from error

    async def run(
        self, task: dict[str, Any], harness: Harness, config: ExperimentConfig
    ) -> AdapterRunOutput:
        if config.adapter.parameters.get("mode", "smoke") == "real":
            return await self._run_real(task, harness, config)
        env = _ReachEnvironment(
            position=int(task.get("start", 0)),
            target=int(task.get("target", 2)),
            max_steps=int(task.get("max_steps", 8)),
        )
        trajectory: list[dict[str, Any]] = []
        reward = 0.0
        replans = 0
        started = perf_counter()
        success = False
        while env.steps < env.max_steps and not success:
            action = 1 if env.position < env.target else -1 if env.position > env.target else 0
            old_distance = abs(env.target - env.position)
            observation, step_reward, success = env.step(action)
            if abs(env.target - observation) >= old_distance and not success:
                replans += 1
            reward += step_reward
            trajectory.append(
                {
                    "step": env.steps,
                    "observation": observation,
                    "action": action,
                    "reward": step_reward,
                }
            )
        elapsed = (perf_counter() - started) * 1_000
        harness_result = await run_reasoning_harness(
            harness,
            task_id=str(task["id"]),
            prompt=f"Reach target {env.target} from {task.get('start', 0)}.",
            expected_answer="success" if success else "failure",
        )
        return AdapterRunOutput(
            value={"success": success, "trajectory": trajectory},
            harness_result=harness_result,
            metrics={
                "episode_reward": reward,
                "steps": env.steps,
                "actions": len(trajectory),
                "replans": replans,
                "recovery_attempts": replans,
                "environment_latency_ms": elapsed,
            },
            artifact_payloads={"episode.json": trajectory},
        )

    async def _run_real(
        self, task: dict[str, Any], harness: Harness, config: ExperimentConfig
    ) -> AdapterRunOutput:
        import gymnasium as gym
        import numpy as np

        environment_id = str(config.adapter.parameters.get("environment_id", "PickCube-v1"))
        env = gym.make(environment_id, obs_mode="state", render_mode=None)
        _, _ = env.reset(seed=config.seed)
        trajectory: list[dict[str, Any]] = []
        total_reward = 0.0
        success = False
        max_steps = int(task.get("max_steps", 10))
        try:
            for step in range(max_steps):
                action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
                _, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                success = bool(info.get("success", False))
                trajectory.append({"step": step + 1, "reward": float(reward), "success": success})
                if terminated or truncated or success:
                    break
        finally:
            env.close()
        harness_result = await run_reasoning_harness(
            harness,
            task_id=str(task["id"]),
            prompt=f"Report the {environment_id} smoke episode outcome.",
            expected_answer="success" if success else "failure",
        )
        return AdapterRunOutput(
            value={"success": success, "trajectory": trajectory},
            harness_result=harness_result,
            metrics={
                "episode_reward": total_reward,
                "steps": len(trajectory),
                "actions": len(trajectory),
            },
            artifact_payloads={"episode.json": trajectory},
        )

    def evaluate(
        self, task: dict[str, Any], run_output: AdapterRunOutput, config: ExperimentConfig
    ) -> AdapterEvaluation:
        success = bool(run_output.value["success"])
        metrics = dict(run_output.metrics)
        return AdapterEvaluation(
            success=success,
            quality=float(success),
            metrics=[Metric(name=name, value=float(value)) for name, value in metrics.items()],
            domain_metrics={"task_success": success, **metrics},
            failure_subtype=None if success else "environment_goal_not_reached",
            evidence=[] if success else [f"trajectory={run_output.value['trajectory']}"],
            latency_ms=float(run_output.metrics.get("environment_latency_ms") or 0),
            dataset=config.evaluation.dataset,
            dataset_version="maniskill-smoke-v1"
            if config.adapter.parameters.get("mode", "smoke") == "smoke"
            else None,
        )

    def project_knowledge(self) -> ProjectKnowledge:
        repo = self.assigned_repo or Path("../ManiSkill").resolve()
        return ProjectKnowledge(
            project_id="maniskill",
            project_name="ManiSkill",
            project_type="embodied_ai",
            repo=str(repo),
            summary=(
                "Headless deterministic reach smoke loop and optional real PickCube-v1 episode."
            ),
            capabilities=frozenset(
                {
                    AdapterCapability.INSPECT,
                    AdapterCapability.SETUP,
                    AdapterCapability.RUN,
                    AdapterCapability.EVALUATE,
                    AdapterCapability.EXPERIMENT,
                    AdapterCapability.OBSERVE,
                }
            ),
            editable_paths=(
                "agent/controller integration",
                "environment configuration",
                "evaluation/",
            ),
            protected_paths=("evals/maniskill_smoke_v1.json", "simulator success definitions"),
            entry_points=("harness/adapters/maniskill.py",),
            evaluation=EvaluationKnowledge(
                operation="ExperimentRunner episode evaluation through ManiSkillAdapter",
                dataset="evals/maniskill_smoke_v1.json",
                dataset_version="maniskill-smoke-v1",
                split="fixed smoke episodes",
                metrics=(
                    MetricKnowledge(
                        name="task_success", direction=MetricDirection.HIGHER, primary=True
                    ),
                    MetricKnowledge(name="episode_reward", direction=MetricDirection.HIGHER),
                    MetricKnowledge(name="steps", direction=MetricDirection.LOWER),
                    MetricKnowledge(name="replans", direction=MetricDirection.LOWER),
                    MetricKnowledge(name="recovery_attempts", direction=MetricDirection.TARGET),
                    MetricKnowledge(
                        name="environment_latency_ms", direction=MetricDirection.LOWER, unit="ms"
                    ),
                ),
            ),
            baseline="configs/maniskill/reach_smoke.yaml using planner_executor.",
            configurable_components=("environment_id", "mode", "max_steps", "harness type"),
            reproducibility_requirements=(
                "Hold environment, seed, initial state, and episode limit constant.",
            ),
            artifacts_to_preserve=(
                "episode trajectory",
                "video when recording is supported",
                "trace",
            ),
            integrity_constraints=(
                "Do not redefine environment success after observing results.",
                "Do not report the deterministic reach smoke as real robot-policy performance.",
            ),
            environment_assumptions=(
                "Real mode requires ManiSkill/Gymnasium and may require simulator or GPU "
                "dependencies.",
            ),
        )
