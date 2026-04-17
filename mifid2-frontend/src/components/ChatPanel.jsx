import MessageBubble from './MessageBubble'
import SystemMessage from './SystemMessage'

export default function ChatPanel({ messages }) {
  if (!messages || messages.length === 0) {
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
      {messages.map((msg, i) => {
        if (msg.type === 'agent') {
          return (
            <MessageBubble
              key={i}
              agentId={msg.agentId}
              content={msg.content}
              structuredOutput={msg.structuredOutput}
              stepNumber={msg.stepNumber}
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
    </div>
  )
}