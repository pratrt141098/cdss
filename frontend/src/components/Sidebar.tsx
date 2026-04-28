import { RefreshCw, Activity, Shield } from 'lucide-react'

interface SidebarProps {
  patients: string[]
  selectedPatient: string
  onSelectPatient: (id: string) => void
  nResults: number
  onNResultsChange: (n: number) => void
  onReload: () => void
  isReloading: boolean
}

export default function Sidebar({
  patients,
  selectedPatient,
  onSelectPatient,
  nResults,
  onNResultsChange,
  onReload,
  isReloading,
}: SidebarProps) {
  return (
    <aside className="w-72 flex flex-col bg-slate-900 text-slate-100 border-r border-slate-800 overflow-hidden shrink-0">
      {/* Logo */}
      <div className="px-5 py-5 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          <Activity className="w-5 h-5 text-blue-400" />
          <span className="font-semibold text-white tracking-tight">CDSS</span>
          <span className="ml-auto text-xs text-slate-500 font-mono">v1.0</span>
        </div>
        <p className="mt-1 text-xs text-slate-500">Clinical Decision Support</p>
      </div>

      <div className="flex-1 overflow-y-auto px-5 py-5 space-y-6">
        {/* Patient selector */}
        <div className="space-y-2">
          <label
            htmlFor="patient-select"
            className="text-xs font-medium text-slate-400 uppercase tracking-wider"
          >
            Patient (HADM ID)
          </label>
          <select
            id="patient-select"
            value={selectedPatient}
            onChange={(e) => onSelectPatient(e.target.value)}
            className="w-full rounded-md bg-slate-800 border border-slate-700 text-slate-100 text-sm px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent cursor-pointer"
          >
            {patients.length === 0 && (
              <option value="" disabled>
                Loading patients…
              </option>
            )}
            {patients.map((id) => (
              <option key={id} value={id}>
                {id}
              </option>
            ))}
          </select>
          {patients.length > 0 && (
            <p className="text-xs text-slate-500">{patients.length} admissions loaded</p>
          )}
        </div>

        {/* Retrieved chunks slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <label
              htmlFor="n-results"
              className="text-xs font-medium text-slate-400 uppercase tracking-wider"
            >
              Retrieved chunks
            </label>
            <span className="text-sm font-mono text-blue-400 tabular-nums">{nResults}</span>
          </div>
          <input
            id="n-results"
            type="range"
            min={3}
            max={10}
            value={nResults}
            onChange={(e) => onNResultsChange(Number(e.target.value))}
            className="w-full accent-blue-500"
          />
          <div className="flex justify-between text-xs text-slate-600 select-none">
            <span>3</span>
            <span>10</span>
          </div>
        </div>

        {/* Reload button */}
        <div className="pt-1">
          <button
            onClick={onReload}
            disabled={isReloading}
            className="w-full flex items-center justify-center gap-2 rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 text-sm px-3 py-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <RefreshCw className={`w-4 h-4 ${isReloading ? 'animate-spin' : ''}`} />
            {isReloading ? 'Reloading…' : 'Reload pipeline'}
          </button>
        </div>
      </div>

      {/* Footer */}
      <div className="px-5 py-4 border-t border-slate-800">
        <div className="flex items-center gap-2 text-xs text-slate-500">
          <Shield className="w-3.5 h-3.5 text-green-500 shrink-0" />
          <span>Fully local — no data leaves this machine</span>
        </div>
      </div>
    </aside>
  )
}
