import { useMemo } from 'react'
import { Stethoscope, Scissors } from 'lucide-react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import type { Diagnosis, Procedure } from '../../types/api'
import { getICDCategory } from '../../lib/icd'

interface Props {
  diagnoses: Diagnosis[]
  procedures: Procedure[]
}

export default function DiagnosesTab({ diagnoses, procedures }: Props) {
  // Category bar chart data
  const categoryData = useMemo(() => {
    const counts = new Map<string, { count: number; hex: string }>()
    for (const d of diagnoses) {
      const cat = getICDCategory(d.icd_code, d.icd_version)
      const existing = counts.get(cat.label)
      if (existing) {
        existing.count++
      } else {
        counts.set(cat.label, { count: 1, hex: cat.hex })
      }
    }
    return Array.from(counts.entries())
      .sort((a, b) => b[1].count - a[1].count)
      .map(([label, { count, hex }]) => ({ label, count, hex }))
  }, [diagnoses])

  // Grouped diagnoses for the card list
  const grouped = useMemo(() => {
    const map = new Map<string, { cat: ReturnType<typeof getICDCategory>; items: Diagnosis[] }>()
    for (const d of diagnoses) {
      const cat = getICDCategory(d.icd_code, d.icd_version)
      if (!map.has(cat.label)) map.set(cat.label, { cat, items: [] })
      map.get(cat.label)!.items.push(d)
    }
    return Array.from(map.values()).sort(
      (a, b) => a.items[0].seq_num - b.items[0].seq_num,
    )
  }, [diagnoses])

  return (
    <div className="p-6 space-y-8 max-w-5xl">
      {/* ── Diagnoses section ── */}
      <section>
        <div className="flex items-center gap-2 mb-4">
          <Stethoscope className="w-5 h-5 text-rose-500" />
          <h2 className="text-base font-semibold text-slate-800">Diagnoses</h2>
          <span className="text-sm text-slate-400">({diagnoses.length})</span>
        </div>

        {diagnoses.length === 0 ? (
          <p className="text-sm text-slate-400">No diagnoses recorded.</p>
        ) : (
          <div className="space-y-6">
            {/* Category bar chart */}
            <div className="bg-white rounded-xl border border-slate-200 p-5">
              <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-4">
                Diagnoses by ICD system
              </p>
              <ResponsiveContainer width="100%" height={categoryData.length * 32 + 20}>
                <BarChart
                  data={categoryData}
                  layout="vertical"
                  margin={{ top: 0, right: 20, bottom: 0, left: 160 }}
                >
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11 }}
                    tickLine={false} axisLine={false} />
                  <YAxis type="category" dataKey="label" width={155}
                    tick={{ fontSize: 11, fill: '#475569' }} tickLine={false} axisLine={false} />
                  <Tooltip
                    contentStyle={{ fontSize: 12, borderRadius: 8, border: '1px solid #e2e8f0' }}
                    formatter={(v) => [v, 'diagnoses']}
                  />
                  <Bar dataKey="count" radius={[0, 4, 4, 0]} maxBarSize={18}>
                    {categoryData.map((d, i) => (
                      <Cell key={i} fill={d.hex} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Grouped card list */}
            <div className="space-y-5">
              {grouped.map(({ cat, items }) => (
                <div key={cat.label}>
                  <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">
                    {cat.label}
                  </p>
                  <div className="space-y-2">
                    {items.map((d) => (
                      <div
                        key={`${d.icd_code}-${d.seq_num}`}
                        className={`bg-white rounded-lg border border-slate-200 border-l-4 ${cat.card} px-4 py-3 flex items-start gap-3`}
                      >
                        <span className={`text-xs font-mono font-bold px-2 py-0.5 rounded shrink-0 mt-0.5 ${cat.badge}`}>
                          {d.icd_code}
                        </span>
                        <div className="min-w-0">
                          <p className="text-sm text-slate-800 leading-snug">{d.description}</p>
                          <p className="text-xs text-slate-400 mt-0.5">
                            ICD-{d.icd_version} · sequence #{d.seq_num}
                          </p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* ── Procedures section ── */}
      <section>
        <div className="flex items-center gap-2 mb-4">
          <Scissors className="w-5 h-5 text-emerald-500" />
          <h2 className="text-base font-semibold text-slate-800">Procedures</h2>
          <span className="text-sm text-slate-400">({procedures.length})</span>
        </div>

        {procedures.length === 0 ? (
          <p className="text-sm text-slate-400">No procedures recorded.</p>
        ) : (
          <div className="space-y-2">
            {procedures.map((p) => (
              <div
                key={`${p.icd_code}-${p.seq_num}`}
                className="bg-white rounded-lg border border-slate-200 border-l-4 border-l-emerald-400 px-4 py-3 flex items-start gap-3"
              >
                <span className="text-xs font-mono font-bold px-2 py-0.5 rounded shrink-0 mt-0.5 bg-emerald-50 text-emerald-700">
                  {p.icd_code}
                </span>
                <div className="min-w-0">
                  <p className="text-sm text-slate-800 leading-snug">{p.description}</p>
                  <p className="text-xs text-slate-400 mt-0.5">
                    ICD-{p.icd_version} · sequence #{p.seq_num}
                  </p>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  )
}
