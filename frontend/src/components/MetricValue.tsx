import type { LabMetric } from '../types'

function label(name: string) {
  return name.replace(/^domain\./, '').replaceAll('_', ' ')
}

function value(metric: LabMetric) {
  if (metric.unit === 'ms' || metric.name.endsWith('_ms')) {
    return metric.value >= 1000
      ? `${(metric.value / 1000).toFixed(1)}s`
      : `${metric.value.toFixed(metric.value < 10 ? 1 : 0)}ms`
  }
  if (metric.unit === 'USD' || metric.name === 'cost_usd') return `$${metric.value.toFixed(3)}`
  if (metric.value >= 0 && metric.value <= 1) return metric.value.toFixed(3).replace(/^0/, '')
  return metric.value.toLocaleString(undefined, { maximumFractionDigits: 2 })
}

export function MetricValue({ metric }: { metric: LabMetric }) {
  return <div className="metric-value"><span>{label(metric.name)}</span><strong>{value(metric)}</strong></div>
}
