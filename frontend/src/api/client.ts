import type { PatientOverview, PipelineStatus, QueryResponse, SummaryResponse } from '../types/api'

const BASE_URL =
  (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000'

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, init)
  if (!res.ok) {
    const err = (await res.json().catch(() => ({}))) as { detail?: string }
    throw new Error(err.detail ?? `Request failed (${res.status})`)
  }
  return res.json() as Promise<T>
}

export const fetchStatus = (): Promise<PipelineStatus> =>
  apiFetch('/status')

export const fetchPatients = (): Promise<string[]> =>
  apiFetch('/patients')

export const fetchOverview = (hadmId: string): Promise<PatientOverview> =>
  apiFetch(`/patients/${hadmId}/overview`)

export const fetchSummary = (hadmId: string): Promise<SummaryResponse> =>
  apiFetch(`/patients/${hadmId}/summary`, { method: 'POST' })

export const sendQuery = (
  hadmId: string,
  query: string,
  nResults: number,
): Promise<QueryResponse> =>
  apiFetch('/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ hadm_id: hadmId, query, n_results: nResults }),
  })

export const reloadPipeline = (): Promise<void> =>
  apiFetch('/pipeline/reload', { method: 'POST' })
