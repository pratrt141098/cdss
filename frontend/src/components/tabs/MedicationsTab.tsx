import { useMemo } from 'react'
import { Pill } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend,
} from 'recharts'
import type { Medication } from '../../types/api'
import { routeColor, ROUTE_COLORS } from '../../lib/icd'

interface Props {
  medications: Medication[]
  admissionTime: string | null
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function fmtDate(s: string | null): string {
  if (!s) return '—'
  try {
    return new Date(s).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })
  } catch {
    return s.slice(0, 10)
  }
}

function RouteTag({ route }: { route: string | null }) {
  if (!route) return null
  const bg: Record<string, string> = {
    PO: 'bg-blue-50 text-blue-700', IV: 'bg-rose-50 text-rose-700',
    SC: 'bg-amber-50 text-amber-700', IM: 'bg-purple-50 text-purple-700',
    SL: 'bg-teal-50 text-teal-700', TOP: 'bg-green-50 text-green-700',
    INH: 'bg-sky-50 text-sky-700',
  }
  const key = route.toUpperCase().split(/[\s/]/)[0]
  return (
    <span className={`inline-block text-xs font-semibold px-2 py-0.5 rounded-full ${bg[key] ?? 'bg-slate-100 text-slate-600'}`}>
      {route}
    </span>
  )
}

// Custom Gantt tooltip
function GanttTooltip({ active, payload }: { active?: boolean; payload?: any[] }) {
  if (!active || !payload?.length) return null
  const row = payload[0]?.payload
  const dur = payload.find((p: any) => p.dataKey === 'duration')
  const off = payload.find((p: any) => p.dataKey === 'startOffset')
  if (!row) return null
  return (
    <div className="bg-white border border-slate-200 rounded-lg px-3 py-2 text-xs shadow-sm">
      <p className="font-semibold text-slate-800 mb-1">{row.name}</p>
      {row.route && <p className="text-slate-500">Route: {row.route}</p>}
      <p className="text-slate-500">Starts day {(off?.value ?? 0).toFixed(1)}</p>
      <p className="text-slate-500">Duration {(dur?.value ?? 0).toFixed(1)} d</p>
    </div>
  )
}

// ── Sub-charts ────────────────────────────────────────────────────────────────

function RouteDonut({ medications }: { medications: Medication[] }) {
  const data = useMemo(() => {
    const counts = new Map<string, number>()
    for (const m of medications) {
      const key = m.route?.toUpperCase().split(/[\s/]/)[0] ?? 'OTHER'
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .map(([name, value]) => ({ name, value }))
  }, [medications])

  if (data.length === 0) return <p className="text-xs text-slate-400 text-center py-8">No route data</p>

  return (
    <ResponsiveContainer width="100%" height={200}>
      <PieChart>
        <Pie data={data} cx="50%" cy="45%" innerRadius={42} outerRadius={68}
          paddingAngle={3} dataKey="value">
          {data.map((d, i) => (
            <Cell key={i} fill={ROUTE_COLORS[d.name] ?? '#94a3b8'} strokeWidth={0} />
          ))}
        </Pie>
        <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v) => [v, 'orders']} />
        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
      </PieChart>
    </ResponsiveContainer>
  )
}

function FrequencyChart({ medications }: { medications: Medication[] }) {
  const data = useMemo(() => {
    const counts = new Map<string, number>()
    for (const m of medications) counts.set(m.drug, (counts.get(m.drug) ?? 0) + 1)
    return Array.from(counts.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 12)
      .map(([name, count]) => ({ name, count }))
  }, [medications])

  if (data.length === 0) return null

  return (
    <ResponsiveContainer width="100%" height={data.length * 32 + 24}>
      <BarChart data={data} layout="vertical" margin={{ top: 0, right: 16, bottom: 0, left: 148 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
        <XAxis type="number" tick={{ fontSize: 11 }} tickLine={false} axisLine={false} />
        <YAxis type="category" dataKey="name" width={143} tick={{ fontSize: 11, fill: '#475569' }}
          tickLine={false} axisLine={false} />
        <Tooltip contentStyle={{ fontSize: 12, borderRadius: 8 }} formatter={(v) => [v, 'orders']} />
        <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={16}>
          {data.map((_, i) => <Cell key={i} fill={i < 3 ? '#7c3aed' : '#a78bfa'} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

function MedGantt({ medications, admissionTime }: { medications: Medication[]; admissionTime: string | null }) {
  const ganttData = useMemo(() => {
    if (!admissionTime) return []
    const admitMs = new Date(admissionTime).getTime()
    if (isNaN(admitMs)) return []

    // Deduplicate: earliest start, latest stop per drug
    const drugMap = new Map<string, { start: number; stop: number; route: string }>()
    for (const m of medications) {
      if (!m.starttime) continue
      const startMs = new Date(m.starttime).getTime()
      if (isNaN(startMs)) continue
      const stopMs = m.stoptime ? new Date(m.stoptime).getTime() : startMs + 86_400_000
      const existing = drugMap.get(m.drug)
      if (!existing) {
        drugMap.set(m.drug, { start: startMs, stop: stopMs, route: m.route ?? '' })
      } else {
        if (startMs < existing.start) existing.start = startMs
        if (stopMs > existing.stop) existing.stop = stopMs
      }
    }

    return Array.from(drugMap.entries())
      .map(([name, { start, stop, route }]) => ({
        name,
        startOffset: parseFloat(Math.max(0, (start - admitMs) / 86_400_000).toFixed(1)),
        duration:    parseFloat(Math.max(0.3, (stop - start) / 86_400_000).toFixed(1)),
        route,
      }))
      .filter(d => d.startOffset >= -0.5)
      .sort((a, b) => a.startOffset - b.startOffset)
      .slice(0, 20)
  }, [medications, admissionTime])

  if (ganttData.length === 0) {
    return <p className="text-xs text-slate-400 text-center py-6">No date data available for timeline.</p>
  }

  const chartH = ganttData.length * 26 + 56

  return (
    <ResponsiveContainer width="100%" height={chartH}>
      <BarChart data={ganttData} layout="vertical"
        margin={{ top: 4, right: 16, bottom: 24, left: 148 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
        <XAxis type="number" unit="d" tick={{ fontSize: 11 }} tickLine={false} axisLine={false}
          label={{ value: 'Days from admission', position: 'insideBottom', offset: -12, fontSize: 11, fill: '#94a3b8' }} />
        <YAxis type="category" dataKey="name" width={143} tick={{ fontSize: 11, fill: '#475569' }}
          tickLine={false} axisLine={false} />
        <Tooltip content={<GanttTooltip />} />
        {/* Transparent offset bar pushes coloured bar to the correct x position */}
        <Bar dataKey="startOffset" stackId="g" fill="transparent" isAnimationActive={false} legendType="none" />
        <Bar dataKey="duration" stackId="g" radius={[0, 3, 3, 0]} maxBarSize={14} isAnimationActive={false}>
          {ganttData.map((d, i) => <Cell key={i} fill={routeColor(d.route)} />)}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}

// ── Main component ────────────────────────────────────────────────────────────

export default function MedicationsTab({ medications, admissionTime }: Props) {
  const uniqueMeds = useMemo(() => {
    const seen = new Map<string, Medication>()
    for (const m of medications) {
      if (!seen.has(m.drug)) seen.set(m.drug, m)
    }
    return Array.from(seen.values())
  }, [medications])

  if (medications.length === 0) {
    return (
      <div className="p-6 flex items-center justify-center h-64 text-slate-400 text-sm">
        No medication records for this admission.
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      {/* Header */}
      <div className="flex items-center gap-2">
        <Pill className="w-5 h-5 text-violet-500" />
        <h2 className="text-base font-semibold text-slate-800">
          {uniqueMeds.length} unique medications
        </h2>
        <span className="text-sm text-slate-400">({medications.length} total orders)</span>
      </div>

      {/* Row 1: route donut + frequency bar side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
        <div className="bg-white rounded-xl border border-slate-200 p-4 lg:col-span-2">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
            Orders by route
          </p>
          <RouteDonut medications={medications} />
        </div>

        <div className="bg-white rounded-xl border border-slate-200 p-4 lg:col-span-3">
          <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
            Top medications by order frequency
          </p>
          <FrequencyChart medications={medications} />
        </div>
      </div>

      {/* Row 2: Gantt timeline */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
          Medication timeline (coloured by route)
        </p>
        <MedGantt medications={medications} admissionTime={admissionTime} />
      </div>

      {/* Row 3: Medication cards */}
      <div>
        <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
          All prescribed medications
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {uniqueMeds.map((med, i) => (
            <div key={i} className="bg-white rounded-xl border border-slate-200 p-4 flex items-start gap-3">
              <div className="w-8 h-8 rounded-full bg-violet-50 flex items-center justify-center shrink-0 mt-0.5">
                <Pill className="w-4 h-4 text-violet-500" />
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2 flex-wrap">
                  <p className="text-sm font-semibold text-slate-800 truncate">{med.drug}</p>
                  <RouteTag route={med.route} />
                </div>
                {(med.dose_val_rx || med.dose_unit_rx) && (
                  <p className="text-xs text-slate-500 mt-0.5">
                    {[med.dose_val_rx, med.dose_unit_rx].filter(Boolean).join(' ')}
                  </p>
                )}
                <p className="text-xs text-slate-400 mt-1">
                  {fmtDate(med.starttime)} → {fmtDate(med.stoptime)}
                </p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
