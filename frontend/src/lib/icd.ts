// Shared ICD classification utilities used by DiagnosesTab and the category chart.

export interface ICDCategory {
  label: string
  badge: string   // Tailwind classes for the inline badge
  card: string    // Tailwind classes for the card left-border
  hex: string     // Hex colour for Recharts
}

const C = (label: string, badge: string, card: string, hex: string): ICDCategory =>
  ({ label, badge, card, hex })

export function getICDCategory(code: string, version: number): ICDCategory {
  if (version === 10) {
    const p = code[0]?.toUpperCase() ?? ''
    if ('AB'.includes(p)) return C('Infectious & Parasitic',  'bg-yellow-100 text-yellow-800', 'border-l-yellow-400', '#ca8a04')
    if ('CD'.includes(p)) return C('Neoplasms',               'bg-red-100 text-red-800',       'border-l-red-400',    '#ef4444')
    if (p === 'E')        return C('Endocrine & Metabolic',   'bg-orange-100 text-orange-800', 'border-l-orange-400', '#f97316')
    if (p === 'F')        return C('Mental & Behavioural',    'bg-purple-100 text-purple-800', 'border-l-purple-400', '#a855f7')
    if (p === 'G')        return C('Nervous System',          'bg-indigo-100 text-indigo-800', 'border-l-indigo-400', '#6366f1')
    if (p === 'I')        return C('Circulatory System',      'bg-rose-100 text-rose-800',     'border-l-rose-400',   '#f43f5e')
    if (p === 'J')        return C('Respiratory System',      'bg-blue-100 text-blue-800',     'border-l-blue-400',   '#3b82f6')
    if (p === 'K')        return C('Digestive System',        'bg-amber-100 text-amber-800',   'border-l-amber-400',  '#f59e0b')
    if (p === 'M')        return C('Musculoskeletal',         'bg-stone-100 text-stone-700',   'border-l-stone-400',  '#78716c')
    if (p === 'N')        return C('Genitourinary',           'bg-teal-100 text-teal-800',     'border-l-teal-400',   '#14b8a6')
    if ('ST'.includes(p)) return C('Injury & Poisoning',      'bg-orange-100 text-orange-800', 'border-l-orange-400', '#f97316')
  } else {
    if (code.startsWith('V')) return C('Supplementary Factors', 'bg-slate-100 text-slate-600',   'border-l-slate-300',  '#94a3b8')
    if (code.startsWith('E')) return C('External Causes',       'bg-slate-100 text-slate-600',   'border-l-slate-300',  '#94a3b8')
    const n = parseInt(code)
    if (n <= 139)  return C('Infectious & Parasitic',  'bg-yellow-100 text-yellow-800', 'border-l-yellow-400', '#ca8a04')
    if (n <= 239)  return C('Neoplasms',               'bg-red-100 text-red-800',       'border-l-red-400',    '#ef4444')
    if (n <= 279)  return C('Endocrine & Metabolic',   'bg-orange-100 text-orange-800', 'border-l-orange-400', '#f97316')
    if (n <= 289)  return C('Blood Disorders',         'bg-pink-100 text-pink-800',     'border-l-pink-400',   '#ec4899')
    if (n <= 319)  return C('Mental & Behavioural',    'bg-purple-100 text-purple-800', 'border-l-purple-400', '#a855f7')
    if (n <= 389)  return C('Nervous System',          'bg-indigo-100 text-indigo-800', 'border-l-indigo-400', '#6366f1')
    if (n <= 459)  return C('Circulatory System',      'bg-rose-100 text-rose-800',     'border-l-rose-400',   '#f43f5e')
    if (n <= 519)  return C('Respiratory System',      'bg-blue-100 text-blue-800',     'border-l-blue-400',   '#3b82f6')
    if (n <= 579)  return C('Digestive System',        'bg-amber-100 text-amber-800',   'border-l-amber-400',  '#f59e0b')
    if (n <= 629)  return C('Genitourinary',           'bg-teal-100 text-teal-800',     'border-l-teal-400',   '#14b8a6')
    if (n <= 709)  return C('Skin & Tissue',           'bg-lime-100 text-lime-800',     'border-l-lime-400',   '#84cc16')
    if (n <= 739)  return C('Musculoskeletal',         'bg-stone-100 text-stone-700',   'border-l-stone-400',  '#78716c')
    if (n <= 799)  return C('Symptoms & Signs',        'bg-gray-100 text-gray-700',     'border-l-gray-300',   '#9ca3af')
    return C('Injury & Poisoning', 'bg-orange-100 text-orange-800', 'border-l-orange-400', '#f97316')
  }
  return C('Other', 'bg-slate-100 text-slate-600', 'border-l-slate-300', '#94a3b8')
}

// Route colours shared between the donut chart and Gantt bars
export const ROUTE_COLORS: Record<string, string> = {
  PO:  '#3b82f6',
  IV:  '#ef4444',
  SC:  '#f59e0b',
  IM:  '#8b5cf6',
  SL:  '#14b8a6',
  TOP: '#10b981',
  INH: '#06b6d4',
  PR:  '#ec4899',
  NG:  '#f97316',
}

export function routeColor(route: string | null): string {
  if (!route) return '#94a3b8'
  const key = route.toUpperCase().split(/[\s/]/)[0]
  return ROUTE_COLORS[key] ?? '#94a3b8'
}
