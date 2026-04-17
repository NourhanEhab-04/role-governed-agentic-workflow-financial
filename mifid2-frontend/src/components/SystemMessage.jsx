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