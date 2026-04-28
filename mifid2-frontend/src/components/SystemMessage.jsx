const SYSTEM_CONFIG = {
  precheck: {
    icon: '◆',
    color: 'text-blue-500',
    bg: 'bg-blue-50 border-blue-100',
  },
  audit: {
    icon: '◆',
    color: 'text-blue-500',
    bg: 'bg-blue-50 border-blue-100',
  },
  escalation: {
    icon: '▲',
    color: 'text-amber-500',
    bg: 'bg-amber-50 border-amber-100',
  },
  halt: {
    icon: '■',
    color: 'text-red-500',
    bg: 'bg-red-50 border-red-100',
  },
  // Verifier passed — green tick
  verification_pass: {
    icon: '✓',
    color: 'text-green-600',
    bg: 'bg-green-50 border-green-100',
  },
  // Verifier flagged issues — amber warning
  verification_warn: {
    icon: '⚠',
    color: 'text-amber-600',
    bg: 'bg-amber-50 border-amber-100',
  },
  // Deterministic consistency issues found before LLM verifier
  consistency: {
    icon: '⚑',
    color: 'text-orange-500',
    bg: 'bg-orange-50 border-orange-100',
  },
  // Corrector successfully applied fixes
  correction_applied: {
    icon: '✎',
    color: 'text-teal-600',
    bg: 'bg-teal-50 border-teal-100',
  },
  // Corrector errored or produced no output
  correction_error: {
    icon: '✖',
    color: 'text-red-500',
    bg: 'bg-red-50 border-red-100',
  },
}

export default function SystemMessage({ type, text }) {
  const config = SYSTEM_CONFIG[type]

  return (
    <div className={`flex items-center gap-2 px-4 py-2 rounded-lg border text-xs ${config.bg}`}>
      <span className={`text-xs ${config.color}`}>{config.icon}</span>
      <span className="text-gray-500 font-medium">{text}</span>
    </div>
  )
}