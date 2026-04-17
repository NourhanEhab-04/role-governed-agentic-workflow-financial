import { useState } from 'react'
const AGENT_CONFIG = {
  A0: { label: 'A0 — Orchestrator',      color: 'bg-gray-100 border-gray-200',   dot: 'bg-gray-400',   text: 'text-gray-500'  },
  A1: { label: 'A1 — Client Profiler',   color: 'bg-blue-50 border-blue-100',    dot: 'bg-blue-400',   text: 'text-blue-600'  },
  A2: { label: 'A2 — Product Classifier',color: 'bg-purple-50 border-purple-100',dot: 'bg-purple-400', text: 'text-purple-600'},
  A3: { label: 'A3 — Rule Engine Agent', color: 'bg-orange-50 border-orange-100',dot: 'bg-orange-400', text: 'text-orange-600'},
  A4: { label: 'A4 — Conflict Detector', color: 'bg-red-50 border-red-100',      dot: 'bg-red-400',    text: 'text-red-600'   },
  A5: { label: 'A5 — Disclosure Agent',  color: 'bg-green-50 border-green-100',  dot: 'bg-green-400',  text: 'text-green-600' },
}

export default function MessageBubble({ agentId, content, structuredOutput, stepNumber }) {
  const [expanded, setExpanded] = useState(false)
  const config = AGENT_CONFIG[agentId]

  return (
    <div className={`rounded-lg border p-4 ${config.color}`}>
      
      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full ${config.dot}`} />
          <span className={`text-sm font-medium ${config.text}`}>
            {config.label}
          </span>
        </div>
        {stepNumber !== undefined && (
          <span className="text-xs text-gray-400">Step {stepNumber}</span>
        )}
      </div>

      {/* Message content */}
      <p className="text-sm text-gray-700 leading-relaxed">
        {content}
      </p>

      {/* Expandable structured output */}
      {structuredOutput && (
        <div className="mt-3">
          <button
            onClick={() => setExpanded(v => !v)}
            className="text-xs text-gray-400 hover:text-gray-600 transition-colors flex items-center gap-1"
          >
            <span>{expanded ? '▲' : '▼'}</span>
            <span>{expanded ? 'Hide structured output' : 'View structured output'}</span>
          </button>
          {expanded && (
            <pre className="mt-2 p-3 bg-white rounded border border-gray-200 text-xs text-gray-600 overflow-x-auto">
              {JSON.stringify(structuredOutput, null, 2)}
            </pre>
          )}
        </div>
      )}

    </div>
  )
}