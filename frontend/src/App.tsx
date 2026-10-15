import { useEffect, useState } from 'react'

import { fetchLabOverview } from './api'
import { Icon } from './components/Icon'
import { MetricValue } from './components/MetricValue'
import type {
  LabExperiment,
  LabOpportunity,
  LabOverview,
  LabSystem,
  LabView,
  ResearchDocument,
} from './types'

const NAV: Array<{ id: LabView; label: string; icon: 'lab' | 'systems' | 'experiments' | 'evidence' | 'activity' }> = [
  { id: 'lab', label: 'Research lab', icon: 'lab' },
  { id: 'systems', label: 'Systems', icon: 'systems' },
  { id: 'experiments', label: 'Experiments', icon: 'experiments' },
  { id: 'evidence', label: 'Evidence', icon: 'evidence' },
  { id: 'activity', label: 'Activity', icon: 'activity' },
]

const VIEW_TITLES: Record<LabView, [string, string]> = {
  lab: ['Research lab', 'Decide what is worth testing next.'],
  systems: ['Systems', 'Current evidence and experimental surfaces.'],
  experiments: ['Experiments', 'Runs, metrics, traces, and artifacts.'],
  evidence: ['Evidence library', 'External evidence stays separate from internal results.'],
  activity: ['Lab activity', 'Programs, builds, approvals, and review boundaries.'],
}

function pretty(value: string) {
  return value.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())
}

function shortDate(value: string | null) {
  if (!value) return 'No runs yet'
  return new Intl.DateTimeFormat(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(new Date(value))
}

function systemFor(opportunity: LabOpportunity, systems: LabSystem[]) {
  return systems.find((system) => system.id === opportunity.system_id)
}

function relatedDocuments(system: LabSystem, documents: ResearchDocument[]) {
  const vocabulary: Record<string, string[]> = {
    bo: ['retrieval', 'rank fusion', 'rrf'],
    myvoice: ['inference', 'latency'],
    swebench: ['agents', 'planning', 'replanning'],
    maniskill: ['planning', 'environment feedback'],
    harness: ['agents', 'inference', 'batching', 'prefix cache'],
  }
  const tags = new Set(vocabulary[system.id] ?? [])
  return documents.filter((document) => document.tags.some((tag) => tags.has(tag))).slice(0, 3)
}

function Status({ value }: { value: string }) {
  return <span className={`status status-${value}`}><i />{pretty(value)}</span>
}

function SectionTitle({ kicker, title, action }: { kicker: string; title: string; action?: React.ReactNode }) {
  return <div className="section-title"><div><span>{kicker}</span><h2>{title}</h2></div>{action}</div>
}

function SystemCard({ system, onOpen }: { system: LabSystem; onOpen: () => void }) {
  return <button className="system-card" onClick={onOpen}>
    <div className="system-card-head"><span className="system-glyph">{system.name.slice(0, 2).toUpperCase()}</span><Status value={system.status} /></div>
    <h3>{system.name}</h3>
    <p>{system.summary}</p>
    <div className="system-card-foot"><span>{system.run_count} runs</span><Icon name="arrow" size={16} /></div>
  </button>
}

function OpportunityRow({ opportunity, systems, onOpen }: { opportunity: LabOpportunity; systems: LabSystem[]; onOpen: () => void }) {
  const system = systemFor(opportunity, systems)
  return <button className="opportunity-row" onClick={onOpen}>
    <span className={`severity severity-${opportunity.severity}`} />
    <span className="opportunity-copy"><small>{system?.name ?? opportunity.system_id} · {opportunity.status}</small><strong>{opportunity.title}</strong><em>{opportunity.description}</em></span>
    <Icon name="arrow" size={16} />
  </button>
}

function LabHome({ data, openSystem, openExperiments }: { data: LabOverview; openSystem: (id: string) => void; openExperiments: () => void }) {
  const papers = data.research_documents.slice(0, 3)
  const activeSystems = data.systems.filter((system) => system.id !== 'harness')
  return <div className="view-stack">
    <section className="research-brief">
      <div><span className="kicker">Today in the lab</span><h2>{data.counts.opportunities ?? 0} opportunities are waiting for judgment.</h2><p>The machinery is running. Your job is to choose the question that deserves a controlled test.</p></div>
      <div className="loop-path" aria-label="Research loop"><span className="active">Notice</span><i /><span>Hypothesize</span><i /><span>Test</span><i /><span>Inspect</span><i /><span>Explain</span></div>
    </section>

    <section className="quick-stats" aria-label="Lab totals">
      <div><strong>{data.counts.experiments ?? 0}</strong><span>experiment runs</span></div>
      <div><strong>{data.counts.candidates ?? 0}</strong><span>candidates to review</span></div>
      <div><strong>{data.counts.papers ?? 0}</strong><span>research sources</span></div>
      <div><strong>{data.counts.builds ?? 0}</strong><span>build records</span></div>
    </section>

    <div className="home-grid">
      <div className="view-stack">
        <section className="panel">
          <SectionTitle kicker="Research inbox" title="Recent findings" />
          <div className="paper-list">
            {papers.map((paper, index) => <a className="paper-row" href={paper.url} target="_blank" rel="noreferrer" key={paper.id}>
              <span>0{index + 1}</span><div><small>{paper.venue} · {paper.publication_date?.slice(0, 4)}</small><strong>{paper.title}</strong><p>{paper.abstract}</p></div><Icon name="external" size={15} />
            </a>)}
            {papers.length === 0 ? <Empty text="No research documents have been ingested." /> : null}
          </div>
        </section>

        <section className="panel">
          <SectionTitle kicker="Experimental surfaces" title="Systems" />
          <div className="system-grid">{activeSystems.map((system) => <SystemCard key={system.id} system={system} onOpen={() => openSystem(system.id)} />)}</div>
        </section>
      </div>

      <section className="panel opportunity-panel">
        <SectionTitle kicker="Measured, not guessed" title="Interesting opportunities" action={<button className="text-button" onClick={openExperiments}>All runs <Icon name="arrow" size={14} /></button>} />
        <div className="opportunity-list">{data.opportunities.slice(0, 8).map((item) => <OpportunityRow key={item.id} opportunity={item} systems={data.systems} onOpen={() => openSystem(item.system_id)} />)}</div>
        {data.opportunities.length === 0 ? <Empty text="No configured bottleneck threshold has been crossed." /> : null}
      </section>
    </div>
  </div>
}

function SystemsView({ data, selectedId, onSelect }: { data: LabOverview; selectedId: string; onSelect: (id: string) => void }) {
  const system = data.systems.find((item) => item.id === selectedId) ?? data.systems[0]
  const opportunities = data.opportunities.filter((item) => item.system_id === system?.id)
  const papers = system ? relatedDocuments(system, data.research_documents) : []
  const [draftOpen, setDraftOpen] = useState(false)
  const [copied, setCopied] = useState(false)
  const primaryOpportunity = opportunities[0]
  const experimentBrief = primaryOpportunity && system
    ? `Target: ${system.name}\nObserved: ${primaryOpportunity.title}\nHypothesis: ${primaryOpportunity.description}\nMetrics: ${primaryOpportunity.affected_metrics.join(', ') || 'define before approval'}\nControl: current fixed baseline\nAcceptance criteria: define thresholds before implementation\nEvidence status: unverified`
    : ''

  async function copyBrief() {
    await navigator.clipboard.writeText(experimentBrief)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1800)
  }

  if (!system) return <Empty text="No registered systems were returned by the Lab API." />
  return <div className="systems-layout">
    <aside className="system-index panel"><span className="kicker">Projects</span>{data.systems.map((item) => <button className={item.id === system.id ? 'system-index-item active' : 'system-index-item'} onClick={() => { onSelect(item.id); setDraftOpen(false) }} key={item.id}><span>{item.name}</span><Status value={item.status} /><small>{pretty(item.project_type)}</small></button>)}</aside>
    <div className="view-stack">
      <section className="panel system-hero"><div className="system-title"><span className="system-glyph large">{system.name.slice(0, 2).toUpperCase()}</span><div><span className="kicker">{pretty(system.project_type)}</span><h2>{system.name}</h2><p>{system.summary}</p><div className="capability-row">{system.capabilities.map((capability) => <span key={capability}>{capability}</span>)}</div></div></div><div className="system-proof"><div><strong>{system.success_count}/{system.run_count}</strong><span>successful runs</span></div><div><strong>{system.capabilities.length}</strong><span>adapter capabilities</span></div></div></section>
      <section className="metric-strip">{system.metrics.length ? system.metrics.map((metric) => <MetricValue metric={metric} key={metric.name} />) : <Empty text="Run the first fixed evaluation to establish a baseline." />}</section>
      <div className="detail-grid">
        <section className="panel"><SectionTitle kicker="Observed" title="Bottlenecks & candidates" />{opportunities.map((item) => <OpportunityRow key={item.id} opportunity={item} systems={data.systems} onOpen={() => setDraftOpen(true)} />)}{opportunities.length === 0 ? <Empty text="No evidence-backed opportunity recorded for this system." /> : null}</section>
        <section className="panel"><SectionTitle kicker="External evidence" title="Research matches" />{papers.map((paper, index) => <a className="match-row" href={paper.url} target="_blank" rel="noreferrer" key={paper.id}><span>{index + 1}</span><div><strong>{paper.title}</strong><small>{paper.venue} · {paper.tags.slice(0, 3).join(' / ')}</small></div><Icon name="external" size={14} /></a>)}{papers.length === 0 ? <Empty text="No tagged research match yet." /> : null}</section>
      </div>
      {primaryOpportunity ? <section className="hypothesis-card"><div><span className="kicker">Suggested hypothesis · unverified</span><h2>{primaryOpportunity.title}</h2><p>{primaryOpportunity.description} Test against the current fixed baseline while holding the task set, model, seed, and evaluation criteria constant.</p><div className="hypothesis-metrics">{primaryOpportunity.affected_metrics.map((metric) => <span key={metric}>{pretty(metric)}</span>)}</div></div><button className="primary-button" onClick={() => setDraftOpen((value) => !value)}>{draftOpen ? 'Close brief' : 'Draft experiment'} <Icon name="arrow" size={16} /></button></section> : null}
      {draftOpen && primaryOpportunity ? <section className="panel experiment-draft"><SectionTitle kicker="Human approval boundary" title="Controlled experiment brief" action={<button className="outline-button" onClick={copyBrief}><Icon name={copied ? 'check' : 'copy'} size={15} /> {copied ? 'Copied' : 'Copy brief'}</button>} /><pre>{experimentBrief}</pre><p>This creates a reviewable specification only. Acceptance thresholds must be fixed before the Build Loop modifies code.</p></section> : null}
    </div>
  </div>
}

function ExperimentDetail({ experiment }: { experiment: LabExperiment }) {
  return <div className="experiment-detail">
    <div className="experiment-heading"><div><span className="kicker">{experiment.domain} / {experiment.run_id.slice(0, 8)}</span><h2>{experiment.name}</h2><p>{experiment.task_id} · {experiment.harness_type} · {experiment.model}</p></div><div className="run-badges"><Status value={experiment.success ? 'healthy' : 'attention'} />{experiment.is_mock ? <span className="mock-badge">Mock evidence</span> : <span className="real-badge">Live evidence</span>}</div></div>
    <section className="metric-strip compact">{experiment.metrics.map((metric) => <MetricValue key={metric.name} metric={metric} />)}</section>
    {experiment.failure ? <section className="failure-box"><Icon name="alert" /><div><strong>Failure classification</strong><pre>{JSON.stringify(experiment.failure, null, 2)}</pre></div></section> : null}
    {experiment.error ? <section className="failure-box"><Icon name="alert" /><div><strong>Run error</strong><pre>{experiment.error}</pre></div></section> : null}
    <section className="trace-section"><SectionTitle kicker="Execution evidence" title={`Trace · ${experiment.trace.length} events`} />{experiment.trace.length ? experiment.trace.map((event, index) => <details className="trace-event" key={`${String(event.sequence)}-${index}`} open={index === 0}><summary><span>{String(index + 1).padStart(2, '0')}</span><strong>{pretty(String(event.type ?? 'event'))}</strong><small>{typeof event.duration_ms === 'number' ? `${event.duration_ms.toFixed(1)}ms` : ''}</small></summary><pre>{JSON.stringify(event.payload ?? event, null, 2)}</pre></details>) : <Empty text="This run did not emit trace events." />}</section>
    <section className="artifact-section"><SectionTitle kicker="Reproducibility" title="Artifacts" /><div className="artifact-grid">{experiment.artifacts.map((artifact, index) => <div className="artifact" key={`${String(artifact.name)}-${index}`}><Icon name="evidence" size={16} /><div><strong>{String(artifact.name ?? 'artifact')}</strong><small>{String(artifact.media_type ?? '')}</small></div></div>)}</div></section>
    <section className="raw-records"><details><summary>Run output</summary><pre>{JSON.stringify(experiment.output, null, 2)}</pre></details><details><summary>Immutable config snapshot</summary><pre>{JSON.stringify(experiment.config, null, 2)}</pre></details></section>
  </div>
}

function ExperimentsView({ data, selectedRun, onSelect }: { data: LabOverview; selectedRun: string | null; onSelect: (id: string) => void }) {
  const experiment = data.experiments.find((item) => item.run_id === selectedRun) ?? data.experiments[0]
  return <div className="experiments-layout"><section className="run-list panel"><div className="run-list-title"><span className="kicker">Recent runs</span><strong>{data.experiments.length}</strong></div>{data.experiments.map((run) => <button className={experiment?.run_id === run.run_id ? 'run-row active' : 'run-row'} onClick={() => onSelect(run.run_id)} key={run.run_id}><span className={run.success ? 'run-dot success' : 'run-dot failed'} /><div><strong>{run.name}</strong><small>{run.domain} · {run.task_id}</small></div><time>{shortDate(run.timestamp)}</time></button>)}{!data.experiments.length ? <Empty text="No experiment results are stored yet." /> : null}</section><section className="panel">{experiment ? <ExperimentDetail experiment={experiment} /> : <Empty text="Choose an experiment run." />}</section></div>
}

function EvidenceView({ data }: { data: LabOverview }) {
  return <div className="evidence-grid">{data.research_documents.map((paper) => <article className="evidence-card" key={paper.id}><div className="evidence-meta"><span>{paper.evidence_type}</span><span>{paper.venue} · {paper.publication_date?.slice(0, 4)}</span></div><h2>{paper.title}</h2><p className="authors">{paper.authors.join(', ')}</p><p>{paper.abstract}</p><div className="tag-row">{paper.tags.map((tag) => <span key={tag}>{tag}</span>)}</div><div className="evidence-links"><a href={paper.url} target="_blank" rel="noreferrer">Read source <Icon name="external" size={14} /></a>{paper.implementation_url ? <a href={paper.implementation_url} target="_blank" rel="noreferrer">Implementation <Icon name="external" size={14} /></a> : null}</div><footer>External evidence · not yet verified in this Lab</footer></article>)}</div>
}

function ActivityView({ data }: { data: LabOverview }) {
  return <section className="panel activity-panel"><SectionTitle kicker="Traceable history" title="Programs & builds" /><div className="activity-list">{data.activity.map((item) => <article className="activity-row" key={`${item.kind}-${item.id}`}><span className={`activity-icon ${item.kind}`}><Icon name={item.kind === 'build' ? 'systems' : 'lab'} size={17} /></span><div><small>{item.kind} · {shortDate(item.timestamp)}</small><strong>{item.title}</strong><p>{item.summary}</p>{Object.keys(item.details).length ? <details className="activity-details"><summary>Inspect record</summary><pre>{JSON.stringify(item.details, null, 2)}</pre></details> : null}</div><Status value={item.status} /></article>)}</div>{!data.activity.length ? <Empty text="No program or build activity has been recorded." /> : null}</section>
}

function Empty({ text }: { text: string }) {
  return <div className="empty-state">{text}</div>
}

export function App() {
  const [data, setData] = useState<LabOverview | null>(null)
  const [view, setView] = useState<LabView>('lab')
  const [selectedSystem, setSelectedSystem] = useState('bo')
  const [selectedRun, setSelectedRun] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  async function load(signal?: AbortSignal) {
    setLoading(true)
    setError(null)
    try {
      const overview = await fetchLabOverview(signal)
      setData(overview)
      setSelectedRun((current) => current ?? overview.experiments[0]?.run_id ?? null)
    } catch (cause) {
      if (cause instanceof DOMException && cause.name === 'AbortError') return
      setError(cause instanceof Error ? cause.message : 'Could not load the Lab API')
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }

  useEffect(() => {
    const controller = new AbortController()
    void load(controller.signal)
    return () => controller.abort()
  }, [])

  const title = VIEW_TITLES[view]
  const successRate = data?.counts.experiments
    ? Math.round(((data.counts.successful_experiments ?? 0) / data.counts.experiments) * 100)
    : 0

  function openSystem(id: string) {
    setSelectedSystem(id)
    setView('systems')
  }

  return <div className="app-shell">
    <aside className="sidebar">
      <div className="brand"><span className="brand-mark"><Icon name="lab" size={20} /></span><div><strong>AI Systems</strong><small>Research Lab</small></div></div>
      <nav aria-label="Lab navigation">{NAV.map((item) => <button key={item.id} className={view === item.id ? 'nav-item active' : 'nav-item'} onClick={() => setView(item.id)} aria-current={view === item.id ? 'page' : undefined}><Icon name={item.icon} /><span>{item.label}</span>{item.id === 'lab' && data?.counts.opportunities ? <b>{data.counts.opportunities}</b> : null}</button>)}</nav>
      <div className="sidebar-foot"><div className="evidence-health"><span>Internal evidence</span><strong>{successRate}% run success</strong><i><b style={{ width: `${successRate}%` }} /></i></div><span className="api-state"><i className={error ? 'offline' : 'online'} />{error ? 'API unavailable' : 'Lab connected'}</span></div>
    </aside>

    <main className="main-content">
      <header className="topbar"><div><p>{title[1]}</p><h1>{title[0]}</h1></div><div className="top-actions"><button className="icon-button" aria-label="Refresh lab data" onClick={() => void load()} disabled={loading}><Icon name="refresh" /></button>{data ? <a className="mlflow-button" href={data.mlflow_url} target="_blank" rel="noreferrer"><span className="mlflow-mark">M</span> Open MLflow <Icon name="external" size={14} /></a> : null}</div></header>

      {loading && !data ? <div className="loading-screen"><span className="loader" /><p>Reading experiment history…</p></div> : null}
      {error && !data ? <div className="error-screen"><Icon name="alert" size={24} /><h2>The Lab API is not reachable.</h2><p>{error}</p><code>py -m uvicorn backend.main:app --reload</code><button className="primary-button" onClick={() => void load()}>Try again</button></div> : null}
      {data ? <>
        {view === 'lab' ? <LabHome data={data} openSystem={openSystem} openExperiments={() => setView('experiments')} /> : null}
        {view === 'systems' ? <SystemsView data={data} selectedId={selectedSystem} onSelect={setSelectedSystem} /> : null}
        {view === 'experiments' ? <ExperimentsView data={data} selectedRun={selectedRun} onSelect={setSelectedRun} /> : null}
        {view === 'evidence' ? <EvidenceView data={data} /> : null}
        {view === 'activity' ? <ActivityView data={data} /> : null}
      </> : null}
    </main>
  </div>
}
