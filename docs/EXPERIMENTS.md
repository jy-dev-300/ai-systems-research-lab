# Running Experiments

Configs are versioned YAML and contain experiment identity, harness, model/backend, adapter, domain parameters, evaluation dataset, tracking, seed, and output directory. Run one task with `py -m harness.cli run <config>`, every task with `batch`, or a Cartesian parameter grid with `sweep`.

To compare harnesses fairly, hold task IDs, dataset version, seed, model/backend, and adapter parameters constant. `configs/swebench/simple_loop.yaml` and `planner_executor.yaml` are paired examples. `configs/bo/reranker_sweep.yaml` demonstrates dotted-path grid expansion.

Local results are under `.harness/runs`; `.harness/knowledge.sqlite3` indexes only queryable run summaries. MLflow remains the primary UI and full artifact tracker when enabled. Start it with:

```powershell
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```

Use `analyze <run-id>` for threshold-based bottlenecks and `research <run-id>` for evidence-linked, controlled candidate experiments. A proposal is a hypothesis until the proposed internal comparison is run.

After both harnesses have historical runs, compare the fixed control with historical routing:

```powershell
py -m harness.cli compare-selection configs/swebench/simple_loop.yaml
```

The output reports choices only; it does not call them optimal or infer a performance gain.

Research-driven implementation uses the separate Build Loop. A BuildTask may reference control/treatment configs; each resulting `ExperimentResult` records its build, hypothesis, and experiment-plan lineage. The coding backend cannot modify the evaluation criteria or held-out dataset after seeing results. See `docs/BUILD_LOOP.md`.

External activation requirements:

- SWE-bench official: populate `../swe-bench`, install its evaluation dependencies/dataset and task execution environment, then use `mode: official` with evaluator-produced patches/logs.
- Natural Language File Search model stages: install `../natural-language-file-search-rag` dependencies and cached BGE/cross-encoder model artifacts; set the real stage exposed by the adapter.
- ManiSkill real: install the optional dependency and simulator assets, then set `mode: real`; video recording can be added through ManiSkill's recorder wrapper when the installed API supports it.
- MyVoice model-backed generation: implement the provider behind MyVoice, keep held-out references out of profiles/training, and return usage/cost fields through the adapter.
