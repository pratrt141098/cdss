import { Loader2, User, Calendar, ArrowRight, Clock, Pill, Stethoscope, Scissors, Brain } from 'lucide-react'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import type { PatientOverview } from '../../types/api'

interface Props {
  overview: PatientOverview
  summary: string | null
  summaryLoading: boolean
}

const ENTITY_COLORS = ['#ef4444', '#3b82f6', '#8b5cf6', '#10b981']

function StatCard({
  label, value, sub, icon: Icon, color,
}: {
  label: string
  value: string | number
  sub?: string
  icon: React.ElementType
  color: string
}) {
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-5 flex items-start gap-4">
      <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${color}`}>
        <Icon className="w-5 h-5" />
      </div>
      <div className="min-w-0">
        <p className="text-2xl font-bold text-slate-900 leading-tight">{value}</p>
        <p className="text-sm font-medium text-slate-600">{label}</p>
        {sub && <p className="text-xs text-slate-400 mt-0.5 truncate">{sub}</p>}
      </div>
    </div>
  )
}

function fmt(dateStr: string | null): string {
  if (!dateStr) return '—'
  try {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short', day: 'numeric', year: 'numeric',
    })
  } catch {
    return dateStr
  }
}

export default function OverviewTab({ overview, summary, summaryLoading }: Props) {
  const { admission, demographics, diagnoses, medications, procedures, ner_entities } = overview

  const entityData = [
    { name: 'Diseases',    value: ner_entities.diseases.length,    color: ENTITY_COLORS[0] },
    { name: 'Medications', value: ner_entities.medications.length, color: ENTITY_COLORS[1] },
    { name: 'Procedures',  value: ner_entities.procedures.length,  color: ENTITY_COLORS[2] },
    { name: 'Anatomy',     value: ner_entities.anatomy.length,     color: ENTITY_COLORS[3] },
  ].filter(d => d.value > 0)

  const uniqueMedCount = new Set(medications.map(m => m.drug)).size

  return (
    <div className="p-6 space-y-6 max-w-5xl">
      {/* Stat cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label="Length of Stay"
          value={admission.length_of_stay_days != null ? `${admission.length_of_stay_days}d` : '—'}
          sub={`${fmt(admission.admission_time)} → ${fmt(admission.discharge_time)}`}
          icon={Clock}
          color="bg-blue-50 text-blue-600"
        />
        <StatCard
          label="Diagnoses"
          value={diagnoses.length}
          sub={diagnoses[0]?.description ?? '—'}
          icon={Stethoscope}
          color="bg-rose-50 text-rose-600"
        />
        <StatCard
          label="Medications"
          value={uniqueMedCount}
          sub={`${medications.length} total orders`}
          icon={Pill}
          color="bg-violet-50 text-violet-600"
        />
        <StatCard
          label="Procedures"
          value={procedures.length}
          sub={procedures[0]?.description ?? '—'}
          icon={Scissors}
          color="bg-emerald-50 text-emerald-600"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Patient & admission details */}
        <div className="bg-white rounded-xl border border-slate-200 p-5 space-y-4">
          <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">
            Patient &amp; Admission
          </h2>
          <div className="space-y-3">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-full bg-slate-100 flex items-center justify-center">
                <User className="w-4 h-4 text-slate-500" />
              </div>
              <div>
                <p className="text-sm font-semibold text-slate-800">
                  {demographics.gender === 'M' ? 'Male' : demographics.gender === 'F' ? 'Female' : demographics.gender}
                  {demographics.anchor_age != null && `, ${demographics.anchor_age} yrs`}
                </p>
                <p className="text-xs text-slate-400">Patient ID {overview.patient_id}</p>
              </div>
            </div>

            {[
              { label: 'Admission type',  value: admission.admission_type },
              { label: 'Admitted from',   value: admission.admission_location },
              { label: 'Discharged to',   value: admission.discharge_location },
            ].map(({ label, value }) => value && (
              <div key={label} className="flex items-start gap-2 text-sm">
                <span className="text-slate-400 w-32 shrink-0">{label}</span>
                <span className="text-slate-700 font-medium leading-snug">{value}</span>
              </div>
            ))}

            <div className="flex items-center gap-2 pt-1">
              <Calendar className="w-4 h-4 text-slate-400" />
              <span className="text-sm text-slate-500">
                {fmt(admission.admission_time)}
              </span>
              <ArrowRight className="w-3 h-3 text-slate-300" />
              <span className="text-sm text-slate-500">
                {fmt(admission.discharge_time)}
              </span>
            </div>
          </div>
        </div>

        {/* NER entity breakdown */}
        {entityData.length > 0 ? (
          <div className="bg-white rounded-xl border border-slate-200 p-5">
            <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider mb-3">
              Entities extracted from notes
            </h2>
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie
                  data={entityData}
                  cx="50%"
                  cy="50%"
                  innerRadius={45}
                  outerRadius={72}
                  paddingAngle={3}
                  dataKey="value"
                >
                  {entityData.map((entry, i) => (
                    <Cell key={i} fill={entry.color} strokeWidth={0} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v, name) => [v, name]}
                  contentStyle={{ fontSize: 12, borderRadius: 8 }}
                />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 12 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="bg-white rounded-xl border border-slate-200 p-5 flex items-center justify-center text-sm text-slate-400">
            No NER entities found — run the pipeline to populate.
          </div>
        )}
      </div>

      {/* AI clinical summary */}
      <div className="bg-white rounded-xl border border-slate-200 p-5">
        <div className="flex items-center gap-2 mb-3">
          <Brain className="w-4 h-4 text-blue-500" />
          <h2 className="text-sm font-semibold text-slate-500 uppercase tracking-wider">
            AI Clinical Summary
          </h2>
          <span className="text-xs text-slate-400 ml-auto">llama3.2:3b</span>
        </div>
        {summaryLoading ? (
          <div className="flex items-center gap-2 text-sm text-slate-400 py-2">
            <Loader2 className="w-4 h-4 animate-spin" />
            <span>Generating summary…</span>
          </div>
        ) : summary ? (
          <p className="text-sm text-slate-700 leading-relaxed">{summary}</p>
        ) : (
          <p className="text-sm text-slate-400 italic">Summary unavailable.</p>
        )}
      </div>
    </div>
  )
}
