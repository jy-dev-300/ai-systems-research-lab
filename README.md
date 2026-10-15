# AI Systems Research Lab

AI Systems Research Lab is a shared R&D platform for four independent systems: SWE-bench coding agents, Natural Language File Search + Grounded RAG, ManiSkill embodied AI, and MyVoice personalized generation. It connects reproducible experiments, evaluation, build loops, tracking, bottleneck analysis, and research recommendations while each system's domain code stays in its own repository. `harness/` owns the cross-system orchestration; `harnesses/` owns explicit agent control loops.

The primary human control surface is a Lab-owned Markdown file in `programs/`. It states what the researcher wants in natural language. JEV classifies only its top-level intent as BUILD, EVAL, or RESEARCH; the Lab then invokes the existing loop and project adapter.

```text
Task -> TaskAdapter -> ExperimentRunner -> Agent Harness -> Domain System
     -> Evaluation -> ExperimentResult -> MLflow -> BottleneckReport
     -> ResearchAdvisor -> CandidateExperiment

Human-approved BuildTask -> BuildAgent -> Build Loop -> target adapter
                         -> tests/experiments -> BuildResult -> human review

program.md -> project registry -> JEV workflow decision -> existing Lab loop
           -> adapter -> sibling repo/worktree -> MLflow + exact git diff
```

## Repository responsibilities

- `harness/`: domain-independent experiment runner, adapters, schemas, tracking, analysis, research, knowledge, CLI.
- `harness/build/`: one general Build Agent, backend contract, approval boundary, iterative Build Loop, acceptance evaluation, and build records.
- `harnesses/`: existing `simple_loop` and `planner_executor` reasoning/execution loops.
- `../natural-language-file-search-rag/`: Natural Language File Search retrieval, ingestion, grounding, and evaluation implementation.
- `../swe-bench/`: external SWE-bench checkout/evaluator location (currently empty).
- `../ManiSkill/`: external simulator location (currently only a project prompt).
- `../myvoice/`: independent MyVoice product location (currently only a project prompt).

Adapters own domain setup, execution, evaluation, artifacts, cleanup, and a discoverable `ProjectKnowledge` contract describing capabilities, commands, paths, fixed evaluations, metric directions, reproducibility, and integrity constraints. Each run gets a fresh adapter bound to its selected checkout or isolated worktree. `ExperimentRunner` owns lifecycle, IDs, timestamps, config fingerprints, Git metadata, exception conversion, trace collection, persistence, MLflow, and experiment history. Adding an adapter or harness uses a registry; the runner does not change.

## Install and verify

Requires Python 3.12 (the smoke suite also runs on the current Python 3.11 environment).

```powershell
cd ai-systems-research-lab
py -m pip install -e ".[dev]"
py -m pytest -q
py -m ruff check .
```

Optional integrations:

```powershell
py -m pip install -e ".[tracking]"   # MLflow
py -m pip install -e ".[maniskill]"  # real ManiSkill smoke episode
py -m pip install -e ".[jev]"        # JEV workflow routing
```

## CLI

```powershell
py -m harness.cli --help
py -m harness.cli run configs/bo/bm25.yaml
py -m harness.cli batch configs/bo/bm25.yaml
py -m harness.cli sweep configs/bo/reranker_sweep.yaml
py -m harness.cli analyze <run-id>
py -m harness.cli research <run-id>
py -m harness.cli compare-selection configs/swebench/simple_loop.yaml
py -m harness.cli build configs/build/bo_hybrid_example.yaml
lab run programs/bo-program.md
lab eval-routing --classifier rules
```

`lab run` defaults to JEV and requires `TYPESAFE_API_KEY`. `--classifier rules` is the explicit offline baseline/smoke path; it is never silently presented as JEV evidence. Every program run preserves the exact Markdown, SHA-256, routing decision/version/confidence, project state, child experiment IDs, and outputs under `.harness/programs/<program-run-id>/`. BUILD programs stop for approval of immutable acceptance criteria until a real coding backend is configured.

Example domain runs:

```powershell
py -m harness.cli run configs/swebench/simple_loop.yaml
py -m harness.cli run configs/bo/bm25.yaml
py -m harness.cli run configs/maniskill/reach_smoke.yaml
py -m harness.cli run configs/myvoice/retrieval.yaml
```

Every run writes immutable config, result, trace, evaluation, and domain artifacts under `.harness/runs/<run-id>/`. Failed setup, model, execution, or evaluation steps become explicit failed results whenever the local artifact store remains writable.

## MLflow

Set `tracking.type: mlflow` in a config, install the tracking extra, and start the UI:

```powershell
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

MLflow receives flattened parameters, common and domain metrics, domain/Git/config/failure tags, and all run artifacts. The Lab UI summarizes research workflow state but does not reimplement MLflow's experiment-tracking views. Local tracking is the dependency-free default in smoke configs.

## Research Lab UI

The central UI is a research cockpit over the Lab's existing records—not a second experiment tracker. It reads adapter knowledge and local `.harness` experiment, program, build, bottleneck, trace, artifact, and research-document data through `GET /api/lab/overview`. MLflow remains the deep tracking UI and is linked from every Lab screen.

Run locally in three terminals:

```powershell
py -m uvicorn backend.main:app --reload --port 8000
cd frontend
npm.cmd install
npm.cmd run dev
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Open `http://localhost:5173`. Set `VITE_API_URL` to move the API and `MLFLOW_UI_URL` to change the MLflow link. Alternatively, `docker compose up --build` starts PostgreSQL, the API, the UI, and MLflow together.

## Domain status

- **Natural Language File Search + Grounded RAG:** the real sibling-repository BM25 path is runnable now. Its BGE, hybrid, RRF, and reranking stages are selectable, but model-backed stages require the repository's declared dependencies and assets.
- **SWE-bench:** representative local patch/evaluation smoke is runnable. Official mode requires installing the SWE-bench checkout, dataset, task repositories/containers, and an agent/evaluator that emits patch/test fields.
- **ManiSkill:** deterministic headless reach loop is runnable. Real `PickCube-v1` smoke uses the installed ManiSkill/Gymnasium API, a zero-action inference check, and needs simulator dependencies/GPU as required by that installation. It does not train a policy.
- **MyVoice:** versioned, leakage-checked held-out evaluation and base/prompt/retrieval strategies are runnable. The current repository has no trained model or service; LoRA is an intentionally open adapter boundary, not a claimed implementation.

The shared taxonomy distinguishes reasoning, tool, retrieval, planning, execution, timeout, resource, evaluation, model, configuration, and unknown failures, with adapter subtypes and evidence. Bottleneck rules use recorded thresholds and trace timings. The Research Advisor retrieves curated/local evidence and labels every proposal `external_evidence_only` and `unverified` until an internal run verifies it. The Build Loop keeps its immutable acceptance criteria outside the coding backend, so writing code is never treated as proof of success.

See [programs](docs/PROGRAMS.md), [architecture](docs/ARCHITECTURE.md), [Build Loop](docs/BUILD_LOOP.md), [experiment guide](docs/EXPERIMENTS.md), [adapter guide](docs/ADDING_ADAPTERS.md), and [Research Brain](docs/RESEARCH_BRAIN.md).
