from __future__ import annotations

from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4

from harness.build.adapter import (
    BuildAdapterRegistry,
    BuildTargetAdapter,
    build_adapter_registry,
)
from harness.build.agent import BuildAgent
from harness.build.models import (
    AppliedChange,
    BuildApprovalState,
    BuildDiagnosis,
    BuildIteration,
    BuildPlan,
    BuildResult,
    BuildStatus,
    BuildTask,
    BuildValidation,
    RepositoryContext,
    TestExecutionRecord,
)
from harness.build.tracking import BuildTracker, LocalBuildTracker
from harness.models import ExperimentResult, TraceEvent


class BuildLoop:
    """Explicit SPEC -> PLAN -> IMPLEMENT -> EVALUATE -> DIAGNOSE loop."""

    def __init__(
        self,
        agent: BuildAgent,
        *,
        adapters: BuildAdapterRegistry = build_adapter_registry,
        tracker: BuildTracker | None = None,
    ) -> None:
        self.agent = agent
        self.adapters = adapters
        self.tracker = tracker or LocalBuildTracker()

    async def run(self, task: BuildTask) -> BuildResult:
        build_id = str(uuid4())
        started = perf_counter()
        fingerprint = task.spec_fingerprint()
        traces: list[TraceEvent] = []
        iterations: list[BuildIteration] = []
        all_changes: list[AppliedChange] = []
        failures: list[str] = []
        adapter: BuildTargetAdapter | None = None
        plan: BuildPlan | None = None
        context: RepositoryContext | None = None
        final_validation: BuildValidation | None = None
        total_tokens = 0
        total_cost = 0.0
        status = BuildStatus.FAILED
        approval = task.approval_state

        def trace(
            event_type: str, payload: dict[str, Any], duration_ms: float | None = None
        ) -> None:
            traces.append(
                TraceEvent(
                    sequence=len(traces),
                    type=event_type,
                    timestamp=datetime.now(UTC),
                    duration_ms=duration_ms,
                    payload=payload,
                )
            )

        if task.approval_state is not BuildApprovalState.APPROVED:
            trace("build_waiting_for_approval", {"state": task.approval_state.value})
            return self._persist(
                task=task,
                build_id=build_id,
                fingerprint=fingerprint,
                plan=None,
                iterations=iterations,
                changes=all_changes,
                traces=traces,
                status=BuildStatus.HUMAN_REVIEW_REQUIRED,
                approval=task.approval_state,
                failures=("BuildTask must be approved before execution",),
                validation=None,
                tokens=0,
                cost=0,
                latency_ms=(perf_counter() - started) * 1_000,
                git_commit=None,
            )

        approval = BuildApprovalState.RUNNING
        try:
            adapter = self.adapters.create(task.adapter_name)
            inspected = perf_counter()
            context = adapter.inspect(task)
            trace(
                "build_inspect",
                {"root": context.root, "files": len(context.files), "dirty": context.dirty},
                (perf_counter() - inspected) * 1_000,
            )
            planned = perf_counter()
            plan = await self.agent.plan(task, context)
            trace("build_plan", plan.model_dump(mode="json"), (perf_counter() - planned) * 1_000)
            if (
                plan.major_architecture_change
                and "architecture" not in task.approved_high_risk_actions
            ):
                trace("build_approval_required", {"reason": "major architecture change"})
                status = BuildStatus.HUMAN_REVIEW_REQUIRED
                approval = BuildApprovalState.REVIEW_REQUIRED
            else:
                diagnosis: BuildDiagnosis | None = None
                for number in range(1, task.max_iterations + 1):
                    if task.spec_fingerprint() != fingerprint:
                        raise RuntimeError("BuildTask changed after execution began")
                    proposed = perf_counter()
                    proposal = await self.agent.propose_changes(
                        task, context, plan, number, diagnosis
                    )
                    trace(
                        "build_implement_proposal",
                        {
                            "iteration": number,
                            "summary": proposal.summary,
                            "files": [change.path for change in proposal.changes],
                            "actions": sorted(proposal.actions),
                        },
                        (perf_counter() - proposed) * 1_000,
                    )
                    total_tokens += proposal.tokens_used
                    total_cost += proposal.cost_usd
                    limit_reason = self._limit_reason(task, total_tokens, total_cost)
                    if limit_reason:
                        failures.append(limit_reason)
                        trace("build_limit_reached", {"reason": limit_reason})
                        status = BuildStatus.LIMIT_REACHED
                        break
                    review_reason = self._review_reason(task, proposal.actions, proposal)
                    if review_reason:
                        failures.append(review_reason)
                        trace("build_approval_required", {"reason": review_reason})
                        status = BuildStatus.HUMAN_REVIEW_REQUIRED
                        approval = BuildApprovalState.REVIEW_REQUIRED
                        break

                    applied = self.agent.apply(task, proposal)
                    all_changes.extend(applied)
                    trace(
                        "build_changes_applied",
                        {"iteration": number, "files": [item.path for item in applied]},
                    )
                    evaluated = perf_counter()
                    validation = await adapter.evaluate(task, applied, build_id=build_id)
                    final_validation = validation
                    trace(
                        "build_validation",
                        {
                            "iteration": number,
                            "accepted": validation.accepted,
                            "criteria": [
                                item.model_dump(mode="json")
                                for item in validation.criterion_results
                            ],
                        },
                        (perf_counter() - evaluated) * 1_000,
                    )
                    iteration = BuildIteration(
                        number=number,
                        proposal=proposal,
                        applied_changes=applied,
                        validation=validation,
                        completed_at=datetime.now(UTC),
                    )
                    iterations.append(iteration)
                    if validation.requires_human_review:
                        status = BuildStatus.HUMAN_REVIEW_REQUIRED
                        approval = BuildApprovalState.REVIEW_REQUIRED
                        break
                    if validation.accepted:
                        status = BuildStatus.SUCCESS
                        approval = BuildApprovalState.REVIEW_REQUIRED
                        break
                    failures.extend(validation.failures)
                    diagnosed = perf_counter()
                    diagnosis = await self.agent.diagnose(task, plan, number, validation)
                    iteration.diagnosis = diagnosis
                    trace(
                        "build_diagnosis",
                        diagnosis.model_dump(mode="json"),
                        (perf_counter() - diagnosed) * 1_000,
                    )
                    if diagnosis.unrecoverable:
                        passed = any(item.passed for item in validation.criterion_results)
                        status = BuildStatus.PARTIAL if passed else BuildStatus.FAILED
                        break
                else:
                    status = BuildStatus.LIMIT_REACHED
                    failures.append(f"iteration limit reached: {task.max_iterations}")
        except Exception as error:
            failures.append(f"{type(error).__name__}: {error}")
            trace("build_error", {"type": type(error).__name__, "message": str(error)})
            status = BuildStatus.FAILED
        finally:
            if adapter is not None:
                try:
                    adapter.cleanup()
                except Exception as error:
                    failures.append(f"cleanup {type(error).__name__}: {error}")
                    trace("build_cleanup_error", {"message": str(error)})

        trace("build_termination", {"status": status.value, "approval": approval.value})
        return self._persist(
            task=task,
            build_id=build_id,
            fingerprint=fingerprint,
            plan=plan,
            iterations=iterations,
            changes=all_changes,
            traces=traces,
            status=status,
            approval=approval,
            failures=tuple(dict.fromkeys(failures)),
            validation=final_validation,
            tokens=total_tokens,
            cost=total_cost,
            latency_ms=(perf_counter() - started) * 1_000,
            git_commit=context.git_commit if context else None,
        )

    @staticmethod
    def _limit_reason(task: BuildTask, tokens: int, cost: float) -> str | None:
        if task.max_tokens is not None and tokens > task.max_tokens:
            return f"token limit exceeded: {tokens} > {task.max_tokens}"
        if task.max_cost_usd is not None and cost > task.max_cost_usd:
            return f"cost limit exceeded: {cost:.6f} > {task.max_cost_usd:.6f}"
        return None

    @staticmethod
    def _review_reason(task: BuildTask, actions: frozenset[str], proposal: Any) -> str | None:
        disallowed = actions - task.allowed_actions
        if disallowed:
            return f"proposal requested disallowed actions: {sorted(disallowed)}"
        if proposal.requires_human_approval:
            return "coding backend marked the proposal as requiring human approval"
        if any(change.operation == "delete" for change in proposal.changes) and (
            "delete" not in task.approved_high_risk_actions
        ):
            return "delete operations require explicit high-risk approval"
        high_risk = actions & frozenset({"architecture", "destructive", "delete"})
        unapproved = high_risk - task.approved_high_risk_actions
        if unapproved:
            return f"high-risk actions require approval: {sorted(unapproved)}"
        return None

    def _persist(
        self,
        *,
        task: BuildTask,
        build_id: str,
        fingerprint: str,
        plan: BuildPlan | None,
        iterations: list[BuildIteration],
        changes: list[AppliedChange],
        traces: list[TraceEvent],
        status: BuildStatus,
        approval: BuildApprovalState,
        failures: tuple[str, ...],
        validation: BuildValidation | None,
        tokens: int,
        cost: float,
        latency_ms: float,
        git_commit: str | None,
    ) -> BuildResult:
        tests: list[TestExecutionRecord] = []
        experiments: list[ExperimentResult] = []
        fixes: list[str] = []
        for iteration in iterations:
            fixes.append(iteration.proposal.summary)
            if iteration.validation:
                tests.extend(iteration.validation.tests)
                experiments.extend(iteration.validation.experiment_results)
        unresolved = tuple(
            result.criterion_id
            for result in (validation.criterion_results if validation else ())
            if not result.passed
        )
        result = BuildResult(
            build_id=build_id,
            task=task,
            spec_fingerprint=fingerprint,
            implementation_plan=plan,
            files_changed=tuple(dict.fromkeys(change.path for change in changes)),
            iterations=tuple(iterations),
            tests_executed=tuple(tests),
            evals_executed=tuple(
                str(result.config.get("experiment", {}).get("name", result.experiment_id))
                for result in experiments
            ),
            experiment_results=tuple(experiments),
            metrics=validation.metrics if validation else {},
            failures_encountered=failures,
            fixes_attempted=tuple(fixes),
            traces=tuple(traces),
            final_status=status,
            approval_state=approval,
            acceptance_criteria_results=(validation.criterion_results if validation else tuple()),
            unresolved_issues=unresolved or failures,
            tokens_used=tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
            git_commit=git_commit,
            research_hypothesis_id=task.research_hypothesis_id,
            experiment_plan_id=task.experiment_plan_id,
            completed_at=datetime.now(UTC),
        )
        return self.tracker.persist(result, tuple(changes))


def review_build(result: BuildResult, *, accept: bool) -> BuildResult:
    if result.approval_state is not BuildApprovalState.REVIEW_REQUIRED:
        raise ValueError("Only review-required builds can be accepted or rejected")
    decision = BuildApprovalState.ACCEPTED if accept else BuildApprovalState.REJECTED
    review_event = TraceEvent(
        sequence=len(result.traces),
        type="build_human_review",
        payload={"decision": decision.value},
    )
    return result.model_copy(
        update={
            "approval_state": decision,
            "traces": (*result.traces, review_event),
        }
    )
