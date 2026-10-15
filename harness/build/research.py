from __future__ import annotations

from harness.build.models import (
    AcceptanceCriterion,
    BuildApprovalState,
    BuildTask,
    ValidationCommand,
)
from harness.research.models import CandidateExperiment, ExperimentPlan, ResearchHypothesis


def hypothesis_from_candidate(candidate: CandidateExperiment) -> ResearchHypothesis:
    return ResearchHypothesis(
        statement=(
            f"{candidate.candidate_technique} may affect "
            f"{', '.join(candidate.expected_metric_affected)} for the observed bottleneck."
        ),
        mechanism=candidate.mechanism,
        expected_metrics=candidate.expected_metric_affected,
        source_candidate_id=candidate.id,
    )


def build_task_from_research(
    *,
    candidate: CandidateExperiment,
    hypothesis: ResearchHypothesis,
    experiment_plan: ExperimentPlan,
    target_project: str,
    adapter_name: str,
    requirements: tuple[str, ...],
    constraints: tuple[str, ...],
    acceptance_criteria: tuple[AcceptanceCriterion, ...],
    validation_config_paths: tuple[str, ...],
    validation_commands: tuple[ValidationCommand, ...] = (),
) -> BuildTask:
    """Create a proposed task; a human must approve it before the loop can run."""
    return BuildTask(
        target_project=target_project,
        adapter_name=adapter_name,
        goal=candidate.controlled_experiment,
        requirements=requirements,
        constraints=constraints,
        acceptance_criteria=acceptance_criteria,
        research_hypothesis_id=hypothesis.id,
        experiment_plan_id=experiment_plan.id,
        validation_config_paths=validation_config_paths,
        validation_commands=validation_commands,
        approval_state=BuildApprovalState.PROPOSED,
        metadata={"candidate_experiment_id": candidate.id},
    )
