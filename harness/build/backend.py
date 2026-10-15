from __future__ import annotations

from typing import Any, Protocol

from harness.build.models import (
    BuildDiagnosis,
    BuildPlan,
    BuildProposal,
    BuildTask,
    BuildValidation,
    RepositoryContext,
)


class CodingBackend(Protocol):
    """Produces reviewable change proposals; it does not decide acceptance."""

    name: str

    async def plan(self, task: BuildTask, context: RepositoryContext) -> BuildPlan: ...

    async def propose_changes(
        self,
        task: BuildTask,
        context: RepositoryContext,
        plan: BuildPlan,
        iteration: int,
        diagnosis: BuildDiagnosis | None,
    ) -> BuildProposal: ...

    async def diagnose(
        self,
        task: BuildTask,
        plan: BuildPlan,
        iteration: int,
        validation: BuildValidation,
    ) -> BuildDiagnosis: ...


class ScriptedBuildBackend:
    """Deterministic backend for tests and human-authored local build scripts.

    It proves build-loop control flow only. It is not a coding-model benchmark.
    """

    name = "scripted"

    def __init__(
        self,
        proposals: list[BuildProposal],
        *,
        plan_steps: tuple[str, ...] = ("Apply the approved scripted change",),
        major_architecture_change: bool = False,
    ) -> None:
        self.proposals = proposals
        self.plan_steps = plan_steps
        self.major_architecture_change = major_architecture_change

    @classmethod
    def from_task(cls, task: BuildTask) -> ScriptedBuildBackend:
        raw_proposals = task.metadata.get("scripted_proposals", [])
        if not isinstance(raw_proposals, list):
            raise ValueError("metadata.scripted_proposals must be a list")
        proposals = [BuildProposal.model_validate(item) for item in raw_proposals]
        raw_steps: Any = task.metadata.get("scripted_plan_steps", ())
        if not isinstance(raw_steps, (list, tuple)):
            raise ValueError("metadata.scripted_plan_steps must be a list")
        return cls(
            proposals,
            plan_steps=tuple(str(item) for item in raw_steps) or cls._default(),
            major_architecture_change=bool(task.metadata.get("major_architecture_change", False)),
        )

    @staticmethod
    def _default() -> tuple[str, ...]:
        return ("Apply the approved scripted change",)

    async def plan(self, task: BuildTask, context: RepositoryContext) -> BuildPlan:
        return BuildPlan(
            summary=f"Implement {task.goal}",
            steps=self.plan_steps,
            expected_files=tuple(
                dict.fromkeys(
                    change.path for proposal in self.proposals for change in proposal.changes
                )
            ),
            major_architecture_change=self.major_architecture_change,
        )

    async def propose_changes(
        self,
        task: BuildTask,
        context: RepositoryContext,
        plan: BuildPlan,
        iteration: int,
        diagnosis: BuildDiagnosis | None,
    ) -> BuildProposal:
        if iteration <= len(self.proposals):
            return self.proposals[iteration - 1]
        return BuildProposal(changes=(), actions=frozenset(), summary="No further scripted changes")

    async def diagnose(
        self,
        task: BuildTask,
        plan: BuildPlan,
        iteration: int,
        validation: BuildValidation,
    ) -> BuildDiagnosis:
        failed = [item.criterion_id for item in validation.criterion_results if not item.passed]
        more_changes = iteration < len(self.proposals)
        return BuildDiagnosis(
            summary=f"Failed criteria: {', '.join(failed) or 'none'}",
            evidence=tuple(validation.failures),
            next_actions=("Apply next scripted revision",) if more_changes else (),
            unrecoverable=not more_changes,
        )
