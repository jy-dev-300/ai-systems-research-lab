export type LabMetric = {
  name: string
  value: number
  unit: string | null
}

export type LabSystem = {
  id: string
  name: string
  project_type: string
  summary: string
  status: string
  repo: string
  repo_exists: boolean
  capabilities: string[]
  metrics: LabMetric[]
  run_count: number
  success_count: number
  last_run_at: string | null
}

export type LabOpportunity = {
  id: string
  system_id: string
  title: string
  description: string
  severity: string
  status: string
  run_id: string | null
  affected_metrics: string[]
  evidence: string[]
  candidate_experiment_id: string | null
}

export type LabExperiment = {
  run_id: string
  task_id: string
  domain: string
  name: string
  timestamp: string
  success: boolean
  is_mock: boolean
  harness_type: string
  model: string
  latency_ms: number
  cost_usd: number
  metrics: LabMetric[]
  failure: Record<string, unknown> | null
  trace: Array<Record<string, unknown>>
  artifacts: Array<Record<string, unknown>>
  output: Record<string, unknown>
  config: Record<string, unknown>
  error: string | null
}

export type ResearchDocument = {
  id: string
  title: string
  authors: string[]
  publication_date: string | null
  venue: string | null
  abstract: string
  url: string
  implementation_url: string | null
  tags: string[]
  evidence_type: string
}

export type LabActivity = {
  id: string
  kind: string
  title: string
  status: string
  timestamp: string | null
  summary: string
  details: Record<string, unknown>
}

export type LabOverview = {
  generated_at: string
  artifact_root: string
  mlflow_url: string
  systems: LabSystem[]
  opportunities: LabOpportunity[]
  experiments: LabExperiment[]
  research_documents: ResearchDocument[]
  activity: LabActivity[]
  counts: Record<string, number>
}

export type LabView = 'lab' | 'systems' | 'experiments' | 'evidence' | 'activity'
