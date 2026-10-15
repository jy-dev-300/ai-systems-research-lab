# Results

This is a living results record. Never mix deterministic mock numbers with hardware measurements.

## Implementation verification conditions (2026-10-01)

- Environment: Windows, CPython 3.11 available locally; the declared Python 3.12 minimum was not verified.
- Correctness workload: `deterministic-tools` version `1.0.0`, three fixed single-tool tasks.
- Model: `DeterministicMockBackend`; temperature and sampling are not applicable.
- Sample count: one deterministic trajectory per focused assertion, plus explicit failure fixtures.
- Judge: exact final-answer equality plus exact tool name and argument equality.
- Evidence: `py -m pytest -q --run-student` and Ruff. These prove deterministic control flow only.
- Hardware inference conditions: not run. GPU latency, throughput, memory, prefix-cache benefit,
  batching behavior, and quantization quality remain unmeasured.

The cache, concurrency, and quantization modules were verified with synthetic callback profiles.
Those checks prove ordering, aggregation, workload consistency, and quality-gate behavior; their
numbers are deliberately not copied into the inference table below.

### Representative deterministic failure evidence

The failure fixtures include a repeated identical action trajectory:

1. model proposes a valid tool call;
2. the same structured action appears again and produces a `repeated_action` recovery event;
3. a third occurrence produces `loop_detected`;
4. policy returns `terminate`, and the run returns `FAILED` with a termination event.

This is synthetic policy evidence, not evidence that a particular real model loops at this rate.

## Harness results

| Harness | Benchmark/version | Success | Correct tool | Correct args | Model calls/task | Tool calls/task | Retries/task | Tokens/task | Main failures |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Simple Tool Loop | deterministic-tools/1.0.0 | 100% (3/3) | 100% | 100% | 2 | 1 | 0 | Not aggregated | None in nominal fixture |
| Planner / Executor | deterministic-tools/1.0.0 | 100% (3/3) | 100% | 100% | 2 | 1 | 0 | Not aggregated | None in nominal fixture |
| Stateful Retry | deterministic-tools/1.0.0 | 100% (3/3) | 100% | 100% | 2 | 1 | 0 | Not aggregated | None in nominal fixture |

Document task definitions, dataset version, model, decoding settings, success judge, sample count, and confidence/variance where meaningful.

## Inference results

| Hardware | Model | Backend/config | p50/p95/p99 | TTFT | Output tok/s | Throughput | GPU memory | Cache hit | Cost/task | Failures/OOM |
|---|---|---|---|---|---|---|---|---|---|---|
| Not run | Not run | Not run | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |

Record driver/CUDA versions, precision or quantization method, context distribution, concurrency, arrival pattern, warmup, and measurement window.

## End-to-end results

- Best harness for quality: three-way tie on the three-task deterministic mock suite only.
- Cheapest harness: not measured.
- Fastest quality-acceptable serving configuration: not measured on real hardware.
- Bottleneck breakdown: accounting behavior tested synthetically; no real-server result recorded.
- Quality/performance tradeoffs: not measurable from the deterministic mock.
- Pareto frontier: algorithm tested with dominated, nondominated, tied, and missing-value fixtures;
  no real hardware frontier recorded.

## Evidence policy

A mock run proves deterministic control-flow correctness. Unit tests prove the assertions they contain. Only a repeated real-server experiment on named hardware supports latency, throughput, cache, utilization, or memory claims.

For profiling, generation duration means backend-reported total inference latency minus
backend-reported queue time. Output tokens/second uses this duration, not end-to-end wall time.
If a backend does not report TTFT, queue time, cache status, or memory, the field remains `None`.
