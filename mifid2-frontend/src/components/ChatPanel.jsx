import MessageBubble from './MessageBubble'
import SystemMessage from './SystemMessage'

// Maps the last-completed LangGraph node name to what the pipeline is doing next
const NEXT_STAGE_LABEL = {
  start:      'Profiling client…',
  A1:         'Classifying product…',
  A2:         'Running deterministic pre-check…',
  pre_check:  'Running rule engine (A3)…',
  A3:         'Running audit check…',
  audit:      'Running conflict detector (A4)…',
  A4:         'Generating suitability report (A5)…',
}

export default function ChatPanel({ messages, isLoading, currentStage }) {
  const loadingLabel = isLoading ? (NEXT_STAGE_LABEL[currentStage] ?? 'Processing…') : null

  if ((!messages || messages.length === 0) && !loadingLabel) {
    return (
      <div className="flex-1 flex items-center justify-center">
        <p className="text-sm text-gray-300">
          Run an assessment to see the agent pipeline here
        </p>
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-6 flex flex-col gap-3">
      {(messages ?? []).map((msg, i) => {
        if (msg.type === 'agent') {
          return (
            <MessageBubble
              key={i}
              agentId={msg.agentId}
              content={msg.content}
              structuredOutput={msg.structuredOutput}
              stepNumber={msg.stepNumber}
              isRetry={msg.isRetry}
              loopData={msg.loopData}
            />
          )
        }
        if (msg.type === 'system') {
          return (
            <SystemMessage
              key={i}
              type={msg.systemType}
              text={msg.text}
            />
          )
        }
        return null
      })}

      {loadingLabel && (
        <div className="flex items-center gap-2 px-1 py-1 text-sm text-gray-400">
          <span className="inline-flex gap-0.5">
            <span className="animate-bounce [animation-delay:-0.3s]">●</span>
            <span className="animate-bounce [animation-delay:-0.15s]">●</span>
            <span className="animate-bounce">●</span>
          </span>
          <span>{loadingLabel}</span>
        </div>
      )}
    </div>
  )
}
