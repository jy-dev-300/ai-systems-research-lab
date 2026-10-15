# Lab programs and JEV routing

`programs/*.md` are natural-language units of work owned by the Lab. They contain no adapter names, commands, repository paths, or workflow fields. Multiple files may target the same project; filename prefixes or the first Markdown heading resolve them through `ProgramRegistry`.

```powershell
py -m pip install -e ".[jev]"
$env:TYPESAFE_API_KEY = "..."
lab run programs/bo-program.md
```

For a dependency-free routing smoke test:

```powershell
lab run programs/myvoice-program.md --classifier rules
lab eval-routing --classifier rules
```

The rules classifier is a measured baseline, not a substitute reported as JEV. The default `JevWorkflowClassifier` uses the official TypeSafe Python API with one typed `Choice`: `TypeSafeClient.system_one(...)`, `response.choices["workflow"].choice`, confidence, and class probabilities. Its prompt, SDK constraint, model alias, decision, confidence, latency, and probabilities are versioned in every run.

## Resolution and context

The program registry owns only project identity, adapter name, sibling repository path, evaluation config, and aliases. A fresh adapter bound to that checkout owns capabilities and the short operational summary through `ProjectKnowledge`. JEV receives only the exact program, project/adapter names, concise adapter knowledge, commit, and dirty state. It receives neither repository contents nor execution traces.

Default mappings are:

| Program prefix | Adapter | Repository | Evaluation config |
|---|---|---|---|
| `bo-` | `bo` | `../natural-language-file-search-rag` | `configs/bo/hybrid.yaml` |
| `myvoice-` | `myvoice` | `../myvoice` | `configs/myvoice/retrieval.yaml` |
| `swe-` | `swebench` | `../swe-bench` | `configs/swebench/simple_loop.yaml` |
| `maniskill-` | `maniskill` | `../ManiSkill` | `configs/maniskill/reach_smoke.yaml` |
| `harness-` | `harness` | this repository | `configs/harness/baseline.yaml` |

## Dispatch and safety

- EVAL loads the registered fixed experiment config and calls `ExperimentRunner`.
- RESEARCH runs the fixed baseline, creates evidence-based bottleneck reports, and asks the existing Research Advisor for externally supported candidate experiments. Candidates remain unverified and require review.
- BUILD creates an immutable proposed `BuildTask` and invokes the existing BuildLoop approval boundary. The current repository has no model-backed coding backend, so no source edit occurs until a human approves concrete acceptance criteria and configures one.

Low-confidence routing never starts a modifying workflow. A strong deterministic rule may resolve an obvious case; otherwise the result is `review_required`. The raw routing history is retained.

`WorktreeManager` allocates a unique detached Git worktree before a BUILD dispatch. A future approved candidate build inside RESEARCH must use the same boundary; the read-only baseline phase does not need a worktree. The source workspace is never reused for simultaneous modifications. This exported workspace currently has no `.git` metadata, so BUILD fails safely until run from real Git clones. Worktrees are intentionally preserved for review rather than automatically deleted.

## Records and evaluation

`.harness/programs/<id>/` contains the exact `program.md`, `adapter_knowledge.json`, `workflow_decision.json`, `routing_history.json`, and `program_result.json`. Child experiment configs receive program lineage, which MLflow logs as parameters and tags. Build tracking preserves `changes.patch`; summaries never replace the patch.

`evals/workflow_routing_v1.json` is the versioned protected routing set. `lab eval-routing` reports overall accuracy, per-class precision/recall, a confusion matrix, mean Brier score, latency, and each failure. Compare classifier changes on this set before changing the default routing prompt or model.
