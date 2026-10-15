import type { LabOverview } from './types'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export async function fetchLabOverview(signal?: AbortSignal): Promise<LabOverview> {
  const response = await fetch(`${API_URL}/api/lab/overview`, { signal })
  if (!response.ok) throw new Error(`Lab API returned ${response.status}`)
  return response.json() as Promise<LabOverview>
}
