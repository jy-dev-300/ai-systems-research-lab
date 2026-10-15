from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from harness.adapters.knowledge import ProjectKnowledge
from harness.analysis.bottlenecks import analyze_bottlenecks
from harness.build.agent import BuildAgent
from harness.build.backend import ScriptedBuildBackend
from harness.build.loop import BuildLoop
from harness.build.models import (
    AcceptanceCriterion,
    BuildApprovalState,
    BuildTask,
    CriterionKind,
)
from harness.experiments.runner import ExperimentRunner
from harness.models import ExperimentConfig
from harness.programs.models import ProgramDefinition, ProgramDocument, Workflow, WorkflowDecision
from harness.research.advisor import ResearchAdvisor
from harness.research.ingestion import load_local_documents


@dataclass
class DispatchResult:
    summary: str
    experiment_run_ids: tuple[str, ...] = ()
    build_id: str | None = None
    candidate_experiment_ids: tuple[str, ...] = ()
    diff_artifact: str | None = None
    review_required: bool = False
    outputs: dict[str, Any] = field(default_factory=dict)


class WorkflowDispatcher(Protocol):
    async def dispatch(
        self,
        *,
        program_run_id: str,
        program: ProgramDocument,
        definition: ProgramDefinition,
        decision: WorkflowDecision,
        repo: Path,
        config_path: Path,
        knowledge: ProjectKnowledge,
    ) -> DispatchResult: ...


class ExistingLoopDispatcher:
    """Thin routing layer over the Lab's existing Build, Eval, and Research machinery."""

    def __init__(self, lab_root: str | Path | None = None) -> None:
        self.lab_root = Path(lab_root or Path.cwd()).resolve()

    @staticmethod
    def _lineage(
        config: ExperimentConfig,
        program_run_id: str,
        program: ProgramDocument,
        decision: WorkflowDecision,
        repo: Path,
    ) -> ExperimentConfig:
        updated = config.model_copy(deep=True)
        updated.adapter.parameters["repo"] = str(repo)
        updated.parameters.setdefault("lineage", {}).update(
            {
                "program_run_id": program_run_id,
                "program_filename": program.filename,
                "program_sha256": program.sha256,
                "program_contents": program.contents,
                "workflow": decision.workflow.value,
                "workflow_classifier": decision.classifier,
                "workflow_classifier_version": decision.classifier_version,
                "workflow_prompt_version": decision.prompt_version,
                "workflow_confidence": decision.confidence,
                "workflow_decision": decision.model_dump(mode="json"),
            }
        )
        return updated

    async def dispatch(
        self,
        *,
        program_run_id: str,
        program: ProgramDocument,
        definition: ProgramDefinition,
        decision: WorkflowDecision,
        repo: Path,
        config_path: Path,
        knowledge: ProjectKnowledge,
    ) -> DispatchResult:
        if decision.workflow is Workflow.BUILD:
            return await self._build(program_run_id, program, definition, repo, knowledge)
        config = self._lineage(
            ExperimentConfig.from_file(config_path), program_run_id, program, decision, repo
        )
        results = await ExperimentRunner().run_batch(config)
        run_ids = tuple(result.run_id for result in results)
        if decision.workflow is Workflow.EVAL:
            succeeded = sum(result.success for result in results)
            return DispatchResult(
                summary=f"Evaluation completed: {succeeded}/{len(results)} tasks succeeded.",
                experiment_run_ids=run_ids,
                outputs={"successes": succeeded, "total": len(results)},
            )

        documents = load_local_documents(
            self.lab_root / "research/fixtures/research_documents.json"
        )
        advisor = ResearchAdvisor(documents)
        candidates = []
        reports = []
        for result in results:
            report = analyze_bottlenecks(result)
            reports.append(report.model_dump(mode="json"))
            candidates.extend(advisor.propose(result, report))
        return DispatchResult(
            summary=(
                f"Research observation completed from {len(results)} baseline run(s); "
                f"{len(candidates)} externally supported candidate experiment(s) require review."
            ),
            experiment_run_ids=run_ids,
            candidate_experiment_ids=tuple(item.id for item in candidates),
            review_required=bool(candidates),
            outputs={
                "bottleneck_reports": reports,
                "candidate_experiments": [item.model_dump(mode="json") for item in candidates],
                "evidence_scope": "external_evidence_only",
                "experimental_surface": knowledge.classification_context(),
            },
        )

    async def _build(
        self,
        program_run_id: str,
        program: ProgramDocument,
        definition: ProgramDefinition,
        repo: Path,
        knowledge: ProjectKnowledge,
    ) -> DispatchResult:
        task = BuildTask(
            target_project=str(repo),
            adapter_name=definition.adapter,
            goal=program.contents.strip(),
            requirements=("Implement the approved Markdown program without redefining it.",),
            constraints=(
                "Acceptance criteria must be approved before code modification.",
                "Preserve the exact resulting diff.",
            ),
            acceptance_criteria=(
                AcceptanceCriterion(
                    id="human-approved-criteria",
                    description="A human approves concrete immutable acceptance criteria.",
                    kind=CriterionKind.HUMAN_REVIEW,
                ),
            ),
            approval_state=BuildApprovalState.PROPOSED,
            metadata={
                "program_run_id": program_run_id,
                "program_filename": program.filename,
                "program_sha256": program.sha256,
                "project_knowledge": knowledge.model_dump(mode="json"),
            },
        )
        result = await BuildLoop(BuildAgent(ScriptedBuildBackend([]))).run(task)
        return DispatchResult(
            summary=(
                "BUILD was routed to the existing BuildLoop and stopped at its human approval "
                "boundary. Approve concrete acceptance criteria and configure a coding backend."
            ),
            build_id=result.build_id,
            diff_artifact=result.diff_artifact,
            review_required=True,
            outputs={"build_task": task.model_dump(mode="json")},
        )
