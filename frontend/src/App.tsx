import { useState, useEffect, useRef, type ReactNode } from 'react'
import {
  LayoutDashboard, Pill, Stethoscope, MessageSquare, Loader2, LucideIcon,
} from 'lucide-react'

import { fetchPatients, fetchOverview, fetchSummary, fetchStatus, sendQuery, reloadPipeline } from './api/client'
import type { Message, PatientOverview, PipelineStatus } from './types/api'

import Sidebar from './components/Sidebar'
import StatusBanner from './components/StatusBanner'
import OverviewTab from './components/tabs/OverviewTab'
import MedicationsTab from './components/tabs/MedicationsTab'
import DiagnosesTab from './components/tabs/DiagnosesTab'
import ChatTab from './components/tabs/ChatTab'

// ── Tab bar ────────────────────────────────────────────────────────────────────

type TabId = 'overview' | 'medications' | 'diagnoses' | 'chat'

const TABS: { id: TabId; label: string; icon: LucideIcon }[] = [
  { id: 'overview',    label: 'Overview',    icon: LayoutDashboard },
  { id: 'medications', label: 'Medications', icon: Pill },
  { id: 'diagnoses',   label: 'Diagnoses',   icon: Stethoscope },
  { id: 'chat',        label: 'Chat',        icon: MessageSquare },
]

function TabBar({
  active, onChange,
}: {
  active: TabId
  onChange: (id: TabId) => void
}) {
  return (
    <div className="flex border-b border-slate-200 bg-white shrink-0 px-2">
      {TABS.map(({ id, label, icon: Icon }) => (
        <button
          key={id}
          onClick={() => onChange(id)}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
            active === id
              ? 'border-blue-600 text-blue-600'
              : 'border-transparent text-slate-500 hover:text-slate-700 hover:border-slate-300'
          }`}
        >
          <Icon className="w-4 h-4" />
          {label}
        </button>
      ))}
    </div>
  )
}

function LoadingOverlay({ label }: { label: string }) {
  return (
    <div className="flex flex-col items-center justify-center flex-1 gap-3 text-slate-400">
      <Loader2 className="w-7 h-7 animate-spin text-blue-400" />
      <p className="text-sm">{label}</p>
    </div>
  )
}

// ── App ────────────────────────────────────────────────────────────────────────

export default function App() {
  const [patients, setPatients]               = useState<string[]>([])
  const [selectedPatient, setSelectedPatient] = useState<string>('')
  const [activeTab, setActiveTab]             = useState<TabId>('overview')
  const [nResults, setNResults]               = useState(5)
  const [isReloading, setIsReloading]         = useState(false)

  // Pipeline status
  const [pipelineStatus, setPipelineStatus]   = useState<PipelineStatus | null>(null)
  const [justReady, setJustReady]             = useState(false)
  const patientsLoaded                        = useRef(false)

  // Per-patient data
  const [overview, setOverview]               = useState<PatientOverview | null>(null)
  const [overviewLoading, setOverviewLoading] = useState(false)
  const [summary, setSummary]                 = useState<string | null>(null)
  const [summaryLoading, setSummaryLoading]   = useState(false)

  // Chat
  const [messages, setMessages]   = useState<Message[]>([])
  const [isQuerying, setIsQuerying] = useState(false)
  const [chatError, setChatError]   = useState<string | null>(null)
  const [prefill, setPrefill]       = useState<string | undefined>()

  // Poll /status until ready, then load patients
  useEffect(() => {
    let intervalId: ReturnType<typeof setInterval>

    const poll = async () => {
      try {
        const s = await fetchStatus()
        setPipelineStatus(s)

        if (s.ready && !s.building && !patientsLoaded.current) {
          patientsLoaded.current = true
          clearInterval(intervalId)
          setJustReady(true)
          setTimeout(() => setJustReady(false), 3000)

          const ids = await fetchPatients()
          setPatients(ids)
          if (ids.length > 0) setSelectedPatient(ids[0])
        }
      } catch {
        // Server not yet accepting connections — keep polling
      }
    }

    poll()
    intervalId = setInterval(poll, 2000)
    return () => clearInterval(intervalId)
  }, [])

  // When patient changes: reset + fetch overview + start summary
  useEffect(() => {
    if (!selectedPatient) return
    setMessages([])
    setChatError(null)
    setOverview(null)
    setSummary(null)

    setOverviewLoading(true)
    fetchOverview(selectedPatient)
      .then(setOverview)
      .catch(console.error)
      .finally(() => setOverviewLoading(false))

    setSummaryLoading(true)
    fetchSummary(selectedPatient)
      .then((r) => setSummary(r.summary))
      .catch(console.error)
      .finally(() => setSummaryLoading(false))
  }, [selectedPatient])

  async function handleQuery(query: string) {
    if (!selectedPatient || isQuerying) return
    const userMsg: Message = { id: crypto.randomUUID(), role: 'user', content: query }
    const loadingMsg: Message = { id: crypto.randomUUID(), role: 'assistant', content: '', loading: true }
    setMessages((prev) => [...prev, userMsg, loadingMsg])
    setIsQuerying(true)
    try {
      const result = await sendQuery(selectedPatient, query, nResults)
      setMessages((prev) =>
        prev.map((m) =>
          m.loading
            ? { ...m, content: result.answer, sources: result.sources,
                elapsed_s: result.elapsed_s, n_chunks: result.n_chunks, loading: false }
            : m,
        ),
      )
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setChatError(msg)
      setMessages((prev) => prev.map((m) => (m.loading ? { ...m, content: `Error: ${msg}`, loading: false } : m)))
    } finally {
      setIsQuerying(false)
    }
  }

  async function handleReload() {
    setIsReloading(true)
    patientsLoaded.current = false
    setPatients([])
    setSelectedPatient('')
    try {
      await reloadPipeline()
      // Pipeline is now rebuilding in background — polling loop will pick it up
    } catch (err) {
      console.error(err)
    } finally {
      setIsReloading(false)
    }
  }

  // Determine tab content
  let tabContent: ReactNode

  if (activeTab === 'chat') {
    tabContent = (
      <ChatTab
        key={selectedPatient}
        patientId={selectedPatient}
        messages={messages}
        isQuerying={isQuerying}
        error={chatError}
        nResults={nResults}
        prefill={prefill}
        onSend={handleQuery}
        onPrefillConsumed={() => setPrefill(undefined)}
      />
    )
  } else if (!selectedPatient) {
    tabContent = <LoadingOverlay label="Select a patient to begin" />
  } else if (overviewLoading) {
    tabContent = <LoadingOverlay label="Loading patient data…" />
  } else if (!overview) {
    tabContent = <LoadingOverlay label="No data available" />
  } else if (activeTab === 'overview') {
    tabContent = (
      <div className="flex-1 overflow-y-auto">
        <OverviewTab overview={overview} summary={summary} summaryLoading={summaryLoading} />
      </div>
    )
  } else if (activeTab === 'medications') {
    tabContent = (
      <div className="flex-1 overflow-y-auto">
        <MedicationsTab
          medications={overview.medications}
          admissionTime={overview.admission.admission_time}
        />
      </div>
    )
  } else if (activeTab === 'diagnoses') {
    tabContent = (
      <div className="flex-1 overflow-y-auto">
        <DiagnosesTab diagnoses={overview.diagnoses} procedures={overview.procedures} />
      </div>
    )
  }

  return (
    <div className="flex h-screen bg-slate-50 overflow-hidden">
      <Sidebar
        patients={patients}
        selectedPatient={selectedPatient}
        onSelectPatient={setSelectedPatient}
        nResults={nResults}
        onNResultsChange={setNResults}
        onReload={handleReload}
        isReloading={isReloading}
      />

      <main className="flex flex-col flex-1 overflow-hidden">
        {/* Header */}
        <header className="flex items-center gap-3 px-6 py-4 bg-white border-b border-slate-200 shrink-0">
          <div className="min-w-0">
            <h1 className="text-base font-semibold text-slate-900 truncate">
              {selectedPatient ? `Patient ${selectedPatient}` : 'No patient selected'}
            </h1>
            <p className="text-xs text-slate-500">
              MIMIC-IV &middot; llama3.2:3b &middot; fully local
            </p>
          </div>
        </header>

        {/* Pipeline status banner */}
        <StatusBanner status={pipelineStatus} justReady={justReady} />

        {/* Tabs */}
        <TabBar active={activeTab} onChange={setActiveTab} />

        {/* Tab content */}
        <div className="flex flex-col flex-1 overflow-hidden">
          {tabContent}
        </div>
      </main>
    </div>
  )
}
