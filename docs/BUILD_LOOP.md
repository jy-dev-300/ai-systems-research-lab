# Build Agent and Build Loop

The Lab now builds and researches AI systems. Building remains human-directed: a `BuildTask` is proposed, reviewed, and explicitly approved before any implementation step can run.

## The three loops

```text
EXECUTION LOOP
model -> tool/action -> observation -> model -> repeat
Purpose: perform one agent task. Implemented by harnesses/.

BUILD LOOP
spec -> plan -> implement -> test/evaluate -> diagnose -> modify -> repeat
Purpose: build or modify an application. Implemented by harness/build/.

RESEARCH LOOP
observe -> research -> hypothesize -> experiment -> analyze -> verify
        -> retain/reject -> repeat
Purpose: discover and validate improvements. Implemented by the Research Brain,
ExperimentRunner, bottleneck analysis, and research evidence models.
```

The loops call one another without merging responsibilities. A build validation may run an execution harness through a domain adapter. A research hypothesis may create a BuildTask and controlled experiment. Only verified internal experiment evidence can become a retained ResearchFinding.

## Build architecture

```text
Human-approved BuildTask
        |
        v
BuildLoop ----> one general BuildAgent ----> pluggable CodingBackend
   |                    |                           |
   |                    +-- plan/diagnose           +-- change proposals
   |                                                (never success criteria)
   v
BuildTargetAdapter bridge
   |
   +-- approved test commands
   +-- existing TaskAdapter + ExperimentRunner
   +-- fixed AcceptanceEvaluator
   v
BuildResult + patch + traces + linked ExperimentResults
        |
        v
Human review -> ACCEPTED or REJECTED
```

`BuildAgent` never declares success. The immutable `BuildTask.acceptance_criteria` are evaluated by `AcceptanceEvaluator` against command exits, experiment results, recorded metrics, files, and applied diffs. Missing evidence fails a criterion. A human-review criterion never passes automatically.

## Human control and safety

Lifecycle states are:

```text
PROPOSED -> APPROVED -> RUNNING -> REVIEW_REQUIRED -> ACCEPTED / REJECTED
```

Technical outcomes are separate: `SUCCESS`, `PARTIAL`, `FAILED`, `LIMIT_REACHED`, or `HUMAN_REVIEW_REQUIRED`.

- Unapproved tasks do not inspect, edit, or test the target.
- Allowed actions are fixed in the approved task.
- Deletes require both an allowed `delete` action and explicit high-risk approval.
- Major architecture/destructive actions require explicit high-risk approval.
- File changes are path-contained inside the target project.
- `create` cannot overwrite an existing file; `update` requires an existing file.
- Token, cost, and iteration limits stop the loop explicitly.
- Held-out data and criteria live outside the coding backend and cannot be rewritten by it.
- Existing dirty target changes are never reset or reverted.

## Backends

`CodingBackend` is the narrow future integration point for Codex or another coding system. It returns a plan, bounded file-change proposals, usage, and diagnoses. The Lab applies approved changes itself.

`ScriptedBuildBackend` is available now for deterministic tests and human-authored local scripts. It proves loop behavior only and must not be described as autonomous coding or model performance.

BuildTask scope is deliberately neutral: the same loop can represent a bug fix, feature, subsystem, model integration, evaluation addition, paper implementation, optimization, architecture change, or new experimental application.

## Adapter and experiment integration

`ExperimentBuildAdapter` wraps the existing adapter registry and `ExperimentRunner`; it does not reproduce Bo, MyVoice, SWE-bench, or ManiSkill evaluation logic. A BuildTask supplies approved validation commands and experiment configs. Result lineage records:

```text
ResearchHypothesis <-> ExperimentPlan <-> BuildTask <-> BuildResult
                   <-> ExperimentResult <-> VerificationResult <-> ResearchFinding
```

## CLI

Review a task YAML first. Change `approval_state` to `approved` only after approving its spec, criteria, commands, actions, limits, and proposed changes.

```powershell
py -m harness.cli build configs/build/bo_hybrid_example.yaml
py -m harness.cli review-build <build-id> accept
py -m harness.cli review-build <build-id> reject
```

The example remains `proposed` and therefore performs no target mutation. A real coding backend is intentionally not impersonated; it should implement `CodingBackend` when an authenticated, sandboxed Codex execution interface is available.
