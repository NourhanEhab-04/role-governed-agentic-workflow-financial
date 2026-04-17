import { useState } from 'react'
import InputPanel from './components/InputPanel'
import ChatPanel from './components/ChatPanel'
import SummaryStrip from './components/SummaryStrip'

const TEST_MESSAGES = [
  { type: 'agent',  agentId: 'A0', stepNumber: 1, content: 'Pipeline started. Running pre-validation checks on inputs.', structuredOutput: null },
  { type: 'agent',  agentId: 'A1', stepNumber: 2, content: 'Client profile extracted. Conservative investor, basic equity knowledge, 5-year horizon, €50,000 liquid assets. No vulnerability flags.', structuredOutput: { knowledge_level: 'basic', risk_tolerance: 3, investment_horizon: 5, liquid_assets: 50000, vulnerability: 'NONE' } },
  { type: 'agent',  agentId: 'A2', stepNumber: 3, content: 'Product classified. Leveraged ETF, risk class 6, requires moderate knowledge, minimum 3-year horizon, total loss potential.', structuredOutput: { product_type: 'Leveraged ETF', risk_class: 6, required_knowledge: 'moderate', min_horizon: 3, total_loss_potential: true } },
  { type: 'system', systemType: 'precheck', text: 'Pre-check verdict: UNSUITABLE (score: 35)' },
  { type: 'agent',  agentId: 'A3', stepNumber: 4, content: 'Rule engine executed. R1 failed (knowledge gap), R2 failed (risk class exceeds tolerance), R6 failed (leverage requires risk tolerance ≥ 7). Score: 35 — UNSUITABLE.', structuredOutput: { score: 35, decision: 'UNSUITABLE', failed_rules: ['R1', 'R2', 'R6'] } },
  { type: 'system', systemType: 'audit', text: 'Audit verdict: AGREED ✅ — all three checkpoints consistent' },
  { type: 'agent',  agentId: 'A4', stepNumber: 5, content: 'Conflict detector audited verdict. Rule engine agreement confirmed. No escalation required.', structuredOutput: { agreed: true, escalate: false } },
  { type: 'agent',  agentId: 'A5', stepNumber: 6, content: 'Suitability report generated. Product is UNSUITABLE for this client under MiFID II Article 25(2). Three rule violations recorded.', structuredOutput: { decision: 'UNSUITABLE', score: 35, article: '25(2)' } },
]
const MOCK_STATE_FULL = {
  pre_check_verdict: "SUITABLE",
  rule_verdict: {
    score: 71,
    decision: "SUITABLE",
    rules: [
      { rule: "R1_knowledge",  pass_: true,  penalty: 0,   detail: "Client holds CFA Level II." },
      { rule: "R2_risk",       pass_: true,  penalty: 0,   detail: "Risk appetite matches product tier." },
      { rule: "R3_horizon",    pass_: true,  penalty: 0,   detail: "5-year horizon aligns with product." },
      { rule: "R4_afford",     pass_: false, penalty: -15, detail: "Liquid assets below recommended threshold." },
      { rule: "R5_vuln",       pass_: true,  penalty: 0,   detail: "No vulnerability flags detected." },
      { rule: "R6_leverage",   pass_: true,  penalty: 0,   detail: "No leverage exposure in product." },
      { rule: "R7_complexity", pass_: false, penalty: -14, detail: "Product complexity exceeds client experience." },
    ],
  },
  audit_verdict: "SUITABLE",
  suitability_report: "Client is broadly suitable. Two minor rule failures (R4, R7) were noted and disclosed. Proceed with written acknowledgment.",
  escalated: false,
  halt: false,
  halt_reason: null,
};

const MOCK_STATE_HALT = {
  pre_check_verdict: "UNSUITABLE",
  rule_verdict: null,
  audit_verdict: null,
  suitability_report: null,
  escalated: false,
  halt: true,
  halt_reason: "Client explicitly stated zero risk tolerance for equities.",
};

const MOCK_STATE_ESCALATED = {
  pre_check_verdict: "CONDITIONAL",
  rule_verdict: {
    score: 31,
    decision: "UNSUITABLE",
    rules: [
      { rule: "R1_knowledge",  pass_: false, penalty: -20, detail: "No relevant product knowledge declared." },
      { rule: "R2_risk",       pass_: false, penalty: -18, detail: "Risk tolerance well below product risk class." },
      { rule: "R3_horizon",    pass_: true,  penalty: 0,   detail: "Horizon acceptable." },
      { rule: "R4_afford",     pass_: true,  penalty: 0,   detail: "Financial situation adequate." },
      { rule: "R5_vuln",       pass_: false, penalty: -16, detail: "Client flagged as potentially vulnerable." },
      { rule: "R6_leverage",   pass_: true,  penalty: 0,   detail: "No leverage exposure." },
      { rule: "R7_complexity", pass_: true,  penalty: 0,   detail: "Complexity within tolerance." },
    ],
  },
  audit_verdict: "UNSUITABLE",
  suitability_report: "Multiple critical rule failures. Escalating to compliance for manual review.",
  escalated: true,
  halt: false,
  halt_reason: null,
};
export default function App() {
  const [isLoading, setIsLoading] = useState(false)

  function handleRun(clientInput, productInput) {
    console.log('Run clicked:', { clientInput, productInput })
    setIsLoading(true)
    setTimeout(() => setIsLoading(false), 3000)
  }

  return (
    <div className="min-h-screen bg-gray-100 flex">
      <InputPanel onRun={handleRun} isLoading={isLoading} />

      <div className="flex-1 flex flex-col">
        <div className="border-b border-gray-200 bg-white px-6 py-4">
          <h2 className="text-base font-medium text-gray-700">Agent Pipeline</h2>
        </div>
        <ChatPanel messages={TEST_MESSAGES} />
        <SummaryStrip state={MOCK_STATE_ESCALATED} />
      </div>
    </div>
  )
}