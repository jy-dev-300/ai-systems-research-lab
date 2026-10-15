# Research Brain

The Research Brain is an experiment recommender, not a paper chatbot.

```text
ExperimentResult + BottleneckReport
        -> hybrid local research retrieval
        -> evidence records
        -> controlled CandidateExperiment
```

`ResearchDocument` normalizes title, authors, date, venue, text/abstract, identifiers, implementation links, tags, provenance, and evidence type. Local JSON ingestion works offline. `ArxivMetadataSource` is a bounded metadata-only connector (maximum 20 results); callers should cache results and avoid aggressive requests. Additional sources implement ingestion into the same document schema.

Retrieval combines token overlap with a small hashed semantic vector, avoiding a vector database. The advisor maps measured bottlenecks to mechanisms and controlled comparisons, attaches retrieved sources, states transfer assumptions, and marks evidence as external/unverified. An internal run may later be stored as `internal_experiment`, but external and internal evidence must never be conflated.

To add a source, write a bounded connector returning `ResearchDocument`, preserve source URL and query/file provenance, test it with fixtures, and merge documents into the advisor corpus. Do not turn abstracts or engineering posts into claims about Optimo without a matching internal experiment.

## From hypothesis to implementation

`hypothesis_from_candidate` converts a candidate into a proposed `ResearchHypothesis`. `build_task_from_research` links an approved `ExperimentPlan` to a still-proposed `BuildTask`; a human must approve the task before the Build Loop runs. Experiment results carry optional build-task, build-result, hypothesis, and experiment-plan IDs. `VerificationResult` and `ResearchFinding` close the lineage without converting external evidence into an internal claim.
