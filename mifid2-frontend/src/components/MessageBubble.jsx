import { useState } from 'react'

// ── Feedback loop inline components ──────────────────────────────────────────

function ConfidencePill({ confidence, passed }) {
  if (confidence == null) return null
  const pct = Math.round(confidence * 100)
  const color = passed ? 'bg-green-100 text-green-700' : pct >= 60 ? 'bg-amber-100 text-amber-700' : 'bg-red-100 text-red-700'
  return (
    <span className={`text-xs font-semibold px-1.5 py-0.5 rounded ${color}`}>{pct}%</span>
  )
}

function VerifyBadge({ verification, label }) {
  const [showDetails, setShowDetails] = useState(false)
  if (!verification || verification.passed === null) return null
  const passed = verification.passed
  const fieldEntries = Object.entries(verification.field_checks ?? {})
  const failedFields = fieldEntries.filter(([, v]) => v.supported === false).map(([k]) => k)

  return (
    <div className={`rounded border px-3 py-2 space-y-1.5 ${passed ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
      {/* Header row */}
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-gray-500">{label}</span>
        <span className={`text-xs font-bold ${passed ? 'text-green-700' : 'text-red-700'}`}>
          {passed ? '✓ Passed' : '✗ Failed'}
        </span>
        <ConfidencePill confidence={verification.confidence} passed={passed} />
        {fieldEntries.length > 0 && (
          <button
            onClick={() => setShowDetails(v => !v)}
            className="ml-auto text-xs text-gray-400 hover:text-gray-600 transition-colors"
          >
            {showDetails ? '▲ hide fields' : '▼ field checks'}
          </button>
        )}
      </div>

      {/* Summary of failed fields */}
      {!passed && failedFields.length > 0 && (
        <div className="text-xs text-red-600">
          Flagged: <span className="font-mono">{failedFields.join(', ')}</span>
        </div>
      )}

      {/* Verifier issues */}
      {verification.issues?.length > 0 && (
        <div className="space-y-0.5">
          {verification.issues.map((issue, i) => (
            <div key={i} className="text-xs text-gray-500 italic">{issue}</div>
          ))}
        </div>
      )}

      {/* Per-field checks table */}
      {showDetails && fieldEntries.length > 0 && (
        <div className="mt-2 rounded border border-gray-200 overflow-hidden">
          <table className="w-full text-xs">
            <thead>
              <tr className="bg-gray-50 border-b border-gray-200">
                <th className="text-left px-2 py-1 font-semibold text-gray-500 w-1/3">Field</th>
                <th className="text-left px-2 py-1 font-semibold text-gray-500 w-12">Status</th>
                <th className="text-left px-2 py-1 font-semibold text-gray-500">Reason</th>
              </tr>
            </thead>
            <tbody>
              {fieldEntries.map(([field, check], i) => (
                <tr key={field} className={`border-b border-gray-100 last:border-0 ${i % 2 === 0 ? 'bg-white' : 'bg-gray-50/50'}`}>
                  <td className="px-2 py-1 font-mono text-gray-700">{field}</td>
                  <td className="px-2 py-1">
                    <span className={`font-bold ${check.supported ? 'text-green-600' : 'text-red-600'}`}>
                      {check.supported ? '✓' : '✗'}
                    </span>
                  </td>
                  <td className="px-2 py-1 text-gray-500 leading-relaxed">{check.reason}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function JsonToggle({ label, data }) {
  const [open, setOpen] = useState(false)
  if (!data) return null
  return (
    <div>
      <button
        onClick={() => setOpen(v => !v)}
        className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1 transition-colors"
      >
        <span>{open ? '▲' : '▼'}</span>
        <span>{label}</span>
      </button>
      {open && (
        <pre className="mt-1 p-2 bg-white rounded border border-gray-200 text-xs text-gray-600 overflow-x-auto leading-relaxed">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  )
}

function FeedbackLoopSection({ loopData }) {
  const [expanded, setExpanded] = useState(false)
  const { initialOutput, consistencyIssues, verification, correction, finalVerification } = loopData

  const corrected = correction?.corrected
  const hasCorrected = !!corrected && !correction?.error
  const effectiveVerify = finalVerification ?? verification
  const loopPassed = effectiveVerify?.passed === true
  const wasCorrected = hasCorrected

  const summaryColor = loopPassed
    ? 'text-green-700 bg-green-50 border-green-200'
    : 'text-amber-700 bg-amber-50 border-amber-200'
  const summaryDot = loopPassed ? 'bg-green-500' : 'bg-amber-400'

  return (
    <div className="mt-3 border-t border-gray-100 pt-2">
      <button
        onClick={() => setExpanded(v => !v)}
        className={`w-full flex items-center gap-2 text-xs font-medium px-2 py-1.5 rounded border transition-colors ${summaryColor}`}
      >
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${summaryDot}`} />
        <span className="flex-1 text-left">
          AV Verification
          {wasCorrected ? ' → Correction Applied' : ''}
          {loopPassed ? ' → Passed' : ' → Issues Remain'}
        </span>
        <span className="text-gray-400">{expanded ? '▲' : '▼'}</span>
      </button>

      {expanded && (
        <div className="mt-3 space-y-3 pl-1">

          {consistencyIssues?.length > 0 && (
            <div className="space-y-1">
              <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide">Consistency flags</div>
              {consistencyIssues.map((issue, i) => (
                <div key={i} className="text-xs bg-orange-50 text-orange-700 rounded px-2 py-1">{issue}</div>
              ))}
            </div>
          )}

          <div className="space-y-2">
            <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide">① First extraction</div>
            <JsonToggle label="View initial output" data={initialOutput} />
          </div>

          {verification && (
            <VerifyBadge verification={verification} label="AV — first check" />
          )}

          {hasCorrected && (
            <div className="space-y-2">
              <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide">② Correction applied</div>
              {correction.fields_fixed?.length > 0 && (
                <div className="text-xs text-teal-700 bg-teal-50 rounded px-2 py-1">
                  Fixed fields: <span className="font-mono">{correction.fields_fixed.join(', ')}</span>
                </div>
              )}
              {correction.error && (
                <div className="text-xs text-red-700 bg-red-50 rounded px-2 py-1">
                  Corrector error: {correction.error}
                </div>
              )}
              <JsonToggle label="View corrected output" data={corrected} />
            </div>
          )}

          {finalVerification && (
            <VerifyBadge verification={finalVerification} label="AV — re-verify after correction" />
          )}

          {!finalVerification && verification?.passed === true && (
            <div className="text-xs text-green-700 bg-green-50 rounded px-2 py-1">
              No correction needed — passed on first check.
            </div>
          )}

        </div>
      )}
    </div>
  )
}

// ─────────────────────────────────────────────────────────────────────────────

const AGENT_CONFIG = {
  A0: { label: 'A0 — Orchestrator',       color: 'bg-gray-100 border-gray-200',    dot: 'bg-gray-400',    text: 'text-gray-500'   },
  A1: { label: 'A1 — Client Profiler',    color: 'bg-blue-50 border-blue-100',     dot: 'bg-blue-400',    text: 'text-blue-600'   },
  A2: { label: 'A2 — Product Classifier', color: 'bg-purple-50 border-purple-100', dot: 'bg-purple-400',  text: 'text-purple-600' },
  AV: { label: 'AV — Verifier',           color: 'bg-indigo-50 border-indigo-100', dot: 'bg-indigo-400',  text: 'text-indigo-600' },
  A3: { label: 'A3 — Rule Engine Agent',  color: 'bg-orange-50 border-orange-100', dot: 'bg-orange-400',  text: 'text-orange-600' },
  A4: { label: 'A4 — Conflict Detector',  color: 'bg-red-50 border-red-100',       dot: 'bg-red-400',     text: 'text-red-600'    },
  A5: { label: 'A5 — Disclosure Agent',   color: 'bg-green-50 border-green-100',   dot: 'bg-green-400',   text: 'text-green-600'  },
}

export default function MessageBubble({ agentId, content, structuredOutput, stepNumber, isRetry, loopData }) {
  const [expanded, setExpanded] = useState(false)
  const config = AGENT_CONFIG[agentId] ?? AGENT_CONFIG['A0']

  return (
    <div className={`rounded-lg border p-4 ${config.color}`}>

      {/* Header row */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className={`w-2.5 h-2.5 rounded-full ${config.dot}`} />
          <span className={`text-sm font-medium ${config.text}`}>
            {config.label}
          </span>
          {isRetry && (
            <span className="text-xs font-medium px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 border border-amber-200">
              correction
            </span>
          )}
        </div>
        {stepNumber !== undefined && (
          <span className="text-xs text-gray-400">Step {stepNumber}</span>
        )}
      </div>

      {/* Message content */}
      <p className="text-sm text-gray-700 leading-relaxed">
        {content}
      </p>

      {/* Feedback loop — verifier + correction inline */}
      {loopData && <FeedbackLoopSection loopData={loopData} />}

      {/* Expandable structured output — always shows the final output */}
      {structuredOutput && (
        <div className="mt-3">
          <button
            onClick={() => setExpanded(v => !v)}
            className="text-xs text-gray-400 hover:text-gray-600 transition-colors flex items-center gap-1"
          >
            <span>{expanded ? '▲' : '▼'}</span>
            <span>{expanded ? 'Hide' : 'View'} {loopData ? 'final output' : 'structured output'}</span>
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