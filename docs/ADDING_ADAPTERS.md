# Adding an Adapter

1. Implement `TaskAdapter` in `harness/adapters/<domain>.py`.
2. Override `project_knowledge()` with the adapter's supported capabilities and concise operational contract.
3. Keep domain logic in its own repository; import its public API or invoke a documented command.
4. Return `AdapterRunOutput` with status, errors, traces, duration, domain output, measured values, and artifact payloads where available.
5. Return `AdapterEvaluation` with success, quality, latency/cost/usage where measured, typed metrics, extensible domain metrics, dataset identity, and an evidence-backed failure subtype.
6. Register the factory in `register_builtin_adapters` (or register it at application startup).
7. Add a versioned evaluation fixture, YAML config, unit tests, and one dependency-light smoke test.

`ProjectKnowledge` is the machine-readable project manual. Declare only capabilities the adapter can actually perform. Include editable and protected paths, entry points, setup/run/test commands, fixed evaluation data, metric directions, baseline, configurable components, reproducibility requirements, retained artifacts, integrity constraints, and environment assumptions. Missing capabilities are valid and cause an explicit workflow rejection instead of a fabricated implementation path.

Adapters are instantiated through `AdapterRegistry.create(name, repo=...)`. The binding is per instance and immutable in practice for the run: BUILD uses the isolated worktree, while EVAL uses the selected checkout. Never retain repository state in a module singleton. Build inspection receives this same knowledge as `RepositoryContext.project_knowledge`; program routing receives only its concise classification view.

Do not catch errors merely to report success, use held-out answers to generate candidates, or label deterministic fixtures as official benchmark/model performance. Cleanup must be idempotent.

To add a harness, implement the existing `Harness` protocol and register a factory in `harness_registry`; `ExperimentRunner` requires no changes.

Adapters do not log to MLflow. They return structured evidence; `ExperimentRunner` owns local artifacts, experiment history, and MLflow tracking.

Build Agents use `ExperimentBuildAdapter`, which runs approved validation commands and invokes `ExperimentRunner` with the same domain config bound to the BuildTask checkout. A new domain therefore normally needs only its `TaskAdapter`, evaluation config, and build-adapter registry name—not a second evaluation implementation.
