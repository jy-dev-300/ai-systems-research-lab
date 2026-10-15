# Architecture

## Data flow

```text
YAML config + task
        |
        v
  ExperimentRunner -----> adapter registry -----> TaskAdapter
        |                                           |
        |                                  sibling domain system
        v                                           |
 harness registry -> simple_loop/planner_executor <-+
        |
        v
 evaluation + failure taxonomy + trace
        |
        v
 ExperimentResult -> local artifacts + MLflow + SQLite knowledge index
        |
        +-> BottleneckReport -> research retrieval -> CandidateExperiment
        +-> historical aggregate selector (optional)
```

## Program control surface

```text
Researcher -> programs/*.md -> ProgramRegistry -> bound TaskAdapter
                                                    |
                                             ProjectKnowledge
                                      |
                                      v
                         JevWorkflowClassifier
                         BUILD | EVAL | RESEARCH
                                      |
                                      v
                           ExistingLoopDispatcher
                    BuildLoop | ExperimentRunner | ResearchAdvisor
                                      |
                                      v
                    TaskAdapter -> independent project/worktree
                                      |
                                      v
                         MLflow artifacts + exact diff
```

The program says what. JEV classifies the top-level intent. The Lab runs an existing loop. The adapter knows how to operate and evaluate the project. JEV never plans, executes tools, edits code, or decides research conclusions.

The Build Agent is a peer capability, not another execution harness:

```text
Human/Research Brain -> approved BuildTask -> BuildLoop -> BuildAgent backend
                                             -> BuildTargetAdapter
                                             -> ExperimentRunner/TaskAdapter
                                             -> fixed acceptance criteria
                                             -> BuildResult -> human review
```

`harnesses/` remains the model/tool control-loop layer. `harness/` composes that layer with domain adapters. A domain repository is imported or invoked across a narrow boundary; its internals are not copied into Optimo.

## Contracts

`TaskAdapter` provides `load_tasks`, `setup`, async `run`, `evaluate`, `artifacts`, optional failure subtype classification, `cleanup`, repository binding, and `project_knowledge`. `ProjectKnowledge` is the discoverable capability and integrity contract for the bound checkout. `ExperimentResult` keeps stable common fields while `domain_metrics` and artifact metadata remain extensible. `Metric` and `TraceEvent` are typed, but payload dictionaries intentionally allow domain evolution.

The runner creates a fresh adapter bound to the configured repository/worktree, harness, trace recorder, run ID, and output directory per task. Cleanup runs in `finally`. BUILD creates its worktree before creating the execution adapter, so inspection, tests, and evaluation address the same candidate checkout. Exceptions are classified and persisted as failed results. Config fingerprints are SHA-256 hashes over canonical JSON. Git commit capture is best-effort because this workspace copy may not itself be a Git worktree.

Adapters return evidence and domain metrics; they never own MLflow or experiment history. `ExperimentRunner` normalizes and persists those results, then sends them to tracking and the knowledge index.

## Trust and reproducibility

Model output is still untrusted at the existing `ToolRegistry` boundary. Trace payloads use the existing redaction and size limits. Config snapshots and dataset versions should be retained with artifacts. Deterministic mocks establish orchestration correctness only; they are not evidence of model, GPU, simulator, or official benchmark performance.

## Selection

`FixedHarnessPolicy` is the control. `HistoricalAggregatePolicy` chooses the best observed success rate and mean quality for matching domain/task history, falling back when no history exists. `compare_selection_policies` records fixed versus task-conditioned choices; it does not claim optimality.

## Loop boundaries

- The **execution loop** performs one model/tool task and lives in `harnesses/`.
- The **build loop** changes an application until fixed evidence criteria pass or a limit/approval boundary stops it.
- The **research loop** proposes and verifies hypotheses, retaining or rejecting them only from internal evidence.

Program routing is a control surface over these loops, not a fourth loop. A RESEARCH program may invoke Build and Eval internally while retaining RESEARCH as its top-level classification.

See `docs/BUILD_LOOP.md` for lifecycle, safety, backend, and lineage details.
