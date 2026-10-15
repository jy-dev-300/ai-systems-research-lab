from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from harness.adapters.registry import adapter_registry, register_builtin_adapters
from harness.programs.classifier import (
    RuleWorkflowClassifier,
    WorkflowClassifier,
    validate_capabilities,
)
from harness.programs.dispatch import ExistingLoopDispatcher, WorkflowDispatcher
from harness.programs.isolation import WorktreeManager
from harness.programs.models import (
    ProgramDocument,
    ProgramRunResult,
    ProgramStatus,
    ProjectState,
    Workflow,
    WorkflowContext,
)
from harness.programs.registry import ProgramRegistry


class ProgramRunner:
    def __init__(
        self,
        classifier: WorkflowClassifier,
        *,
        lab_root: str | Path | None = None,
        registry: ProgramRegistry | None = None,
        dispatcher: WorkflowDispatcher | None = None,
        confidence_threshold: float = 0.70,
        output_root: str | Path = ".harness/programs",
        worktrees: WorktreeManager | None = None,
    ) -> None:
        self.lab_root = Path(lab_root or Path.cwd()).resolve()
        self.classifier = classifier
        self.registry = registry or ProgramRegistry(self.lab_root)
        self.dispatcher = dispatcher or ExistingLoopDispatcher(self.lab_root)
        self.confidence_threshold = confidence_threshold
        self.output_root = (self.lab_root / output_root).resolve()
        self.worktrees = worktrees or WorktreeManager(self.lab_root / ".harness/worktrees")

    async def run(self, path: str | Path) -> ProgramRunResult:
        program = ProgramDocument.load(path)
        program_run_id = str(uuid4())
        started = datetime.now(UTC)
        result: ProgramRunResult
        routing_history: list[dict[str, Any]] = []
        definition = None
        decision = None
        commit = None
        worktree = None
        branch = None
        try:
            definition = self.registry.resolve(program)
            repo = self.registry.repo_path(definition)
            register_builtin_adapters()
            project_adapter = adapter_registry.create(definition.adapter, repo=repo)
            knowledge = project_adapter.project_knowledge()
            capabilities = tuple(sorted(item.value for item in knowledge.capabilities))
            commit, dirty = WorktreeManager.inspect(repo)
            state = ProjectState(
                repo=str(repo),
                exists=repo.is_dir(),
                git_commit=commit,
                dirty=dirty,
                summary=(
                    f"{knowledge.summary} repo_exists={repo.is_dir()}, "
                    f"git_commit={commit or 'unavailable'}, dirty={dirty}."
                ),
            )
            context = WorkflowContext(
                program=program,
                project=definition.project,
                adapter=definition.adapter,
                capabilities=capabilities,
                project_state=state,
                project_knowledge=knowledge.classification_context(),
            )
            decision = self.classifier.classify(context)
            routing_history.append(decision.model_dump(mode="json"))
            if decision.confidence < self.confidence_threshold:
                fallback = RuleWorkflowClassifier().classify(context)
                routing_history.append(fallback.model_dump(mode="json"))
                if fallback.confidence < 0.90:
                    result = ProgramRunResult(
                        program_run_id=program_run_id,
                        status=ProgramStatus.REVIEW_REQUIRED,
                        program=program,
                        project=definition.project,
                        adapter=definition.adapter,
                        target_repo_commit=commit,
                        decision=decision,
                        summary=(
                            "Workflow intent is ambiguous; no loop was started. Clarify whether "
                            "the primary intent is BUILD, EVAL, or RESEARCH."
                        ),
                        outputs={"routing_history": routing_history},
                        project_knowledge=knowledge.model_dump(mode="json"),
                        started_at=started,
                        completed_at=datetime.now(UTC),
                    )
                    return self._persist(result)
                decision = fallback.model_copy(
                    update={
                        "classifier": f"{decision.classifier}+rules",
                        "classifier_version": (
                            f"{decision.classifier_version};{fallback.classifier_version}"
                        ),
                        "reason": (
                            "Low-confidence JEV decision "
                            f"({routing_history[0]['confidence']:.3f}); "
                            f"{fallback.reason}"
                        ),
                    }
                )
            missing = validate_capabilities(decision, capabilities)
            if missing:
                raise RuntimeError(f"Adapter lacks required capabilities: {', '.join(missing)}")
            execution_repo = repo
            if decision.workflow is Workflow.BUILD:
                execution_repo, branch, commit = self.worktrees.create(repo, program_run_id)
                worktree = str(execution_repo)
            execution_adapter = adapter_registry.create(definition.adapter, repo=execution_repo)
            execution_knowledge = execution_adapter.project_knowledge()
            dispatched = await self.dispatcher.dispatch(
                program_run_id=program_run_id,
                program=program,
                definition=definition,
                decision=decision,
                repo=execution_repo,
                config_path=self.registry.config_path(definition),
                knowledge=execution_knowledge,
            )
            result = ProgramRunResult(
                program_run_id=program_run_id,
                status=(
                    ProgramStatus.REVIEW_REQUIRED
                    if dispatched.review_required
                    else ProgramStatus.COMPLETED
                ),
                program=program,
                project=definition.project,
                adapter=definition.adapter,
                target_repo_commit=commit,
                worktree=worktree,
                branch=branch,
                decision=decision,
                experiment_run_ids=dispatched.experiment_run_ids,
                build_id=dispatched.build_id,
                candidate_experiment_ids=dispatched.candidate_experiment_ids,
                diff_artifact=dispatched.diff_artifact,
                summary=dispatched.summary,
                outputs={**dispatched.outputs, "routing_history": routing_history},
                project_knowledge=execution_knowledge.model_dump(mode="json"),
                started_at=started,
                completed_at=datetime.now(UTC),
            )
        except Exception as error:
            result = ProgramRunResult(
                program_run_id=program_run_id,
                status=ProgramStatus.FAILED,
                program=program,
                project=definition.project if definition else None,
                adapter=definition.adapter if definition else None,
                target_repo_commit=commit,
                worktree=worktree,
                branch=branch,
                decision=decision,
                summary="Program run failed explicitly before completion.",
                errors=(f"{type(error).__name__}: {error}",),
                outputs={"routing_history": routing_history},
                started_at=started,
                completed_at=datetime.now(UTC),
            )
        return self._persist(result)

    def _persist(self, result: ProgramRunResult) -> ProgramRunResult:
        directory = self.output_root / result.program_run_id
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "program.md").write_text(result.program.contents, encoding="utf-8")
        if result.decision is not None:
            (directory / "workflow_decision.json").write_text(
                result.decision.model_dump_json(indent=2), encoding="utf-8"
            )
        if result.project_knowledge:
            (directory / "adapter_knowledge.json").write_text(
                json.dumps(result.project_knowledge, indent=2), encoding="utf-8"
            )
        updated = result.model_copy(update={"artifact_dir": str(directory)})
        (directory / "program_result.json").write_text(
            updated.model_dump_json(indent=2), encoding="utf-8"
        )
        (directory / "routing_history.json").write_text(
            json.dumps(updated.outputs.get("routing_history", []), indent=2), encoding="utf-8"
        )
        return updated
