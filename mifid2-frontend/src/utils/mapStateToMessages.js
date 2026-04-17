export function mapStateToMessages(state) {
  const messages = []
  let step = 1

  // A0 — always first
  messages.push({
    type: 'agent',
    agentId: 'A0',
    stepNumber: step++,
    content: state.halt && !state.client_profile
      ? `Pipeline halted before profiling. Reason: ${state.halt_reason}`
      : 'Pipeline started. Running pre-validation checks on inputs.',
    structuredOutput: null,
  })

  // A1 — client profiler
  if (state.client_profile) {
    const p = state.client_profile
    messages.push({
      type: 'agent',
      agentId: 'A1',
      stepNumber: step++,
      content: `Client profile extracted. ${p.knowledge_level} knowledge level, risk tolerance ${p.risk_tolerance}/10, ${p.investment_horizon}-year horizon, €${p.liquid_assets?.toLocaleString()} liquid assets. Vulnerability: ${p.vulnerability ?? 'NONE'}.`,
      structuredOutput: p,
    })
  }

  if (state.halt && !state.product_profile) {
    messages.push({ type: 'system', systemType: 'halt', text: `Pipeline halted: ${state.halt_reason}` })
    return messages
  }

  // A2 — product classifier
  if (state.product_profile) {
    const p = state.product_profile
    messages.push({
      type: 'agent',
      agentId: 'A2',
      stepNumber: step++,
      content: `Product classified. ${p.product_type ?? 'Product'}, risk class ${p.risk_class}, requires ${p.required_knowledge} knowledge, minimum ${p.min_horizon}-year horizon.${p.total_loss_potential ? ' Has total loss potential.' : ''}${p.is_leveraged ? ' Leveraged product.' : ''}`,
      structuredOutput: p,
    })
  }

  if (state.halt && !state.pre_check_verdict) {
    messages.push({ type: 'system', systemType: 'halt', text: `Pipeline halted: ${state.halt_reason}` })
    return messages
  }

  // Pre-check system message
  if (state.pre_check_verdict) {
    const v = state.pre_check_verdict
    messages.push({
      type: 'system',
      systemType: 'precheck',
      text: `Pre-check verdict: ${v.decision} (score: ${v.score})`,
    })
  }

  // A3 — rule engine agent
  if (state.rule_verdict) {
    const v = state.rule_verdict
    const failed = v.rules
      ? Object.entries(v.rules).filter(([, r]) => !r.passed).map(([k]) => k).join(', ')
      : 'none'
    messages.push({
      type: 'agent',
      agentId: 'A3',
      stepNumber: step++,
      content: `Rule engine executed. Score: ${v.score} — ${v.decision}.${failed !== 'none' ? ` Failed rules: ${failed}.` : ' All rules passed.'}`,
      structuredOutput: v,
    })
  }

  if (state.halt && !state.audit_verdict) {
    messages.push({ type: 'system', systemType: 'halt', text: `Pipeline halted: ${state.halt_reason}` })
    return messages
  }

  // Audit system message
  if (state.audit_verdict) {
    const a = state.audit_verdict
    messages.push({
      type: 'system',
      systemType: 'audit',
      text: a.agreed
        ? 'Audit verdict: AGREED ✅ — all three checkpoints consistent'
        : `Audit verdict: DISAGREEMENT ❌ — ${a.detail ?? 'possible bypass detected'}`,
    })
  }

  // A4 — conflict detector
  if (state.conflict_report) {
    const c = state.conflict_report
    messages.push({
      type: 'agent',
      agentId: 'A4',
      stepNumber: step++,
      content: c.escalate
        ? `Conflict detected. ${c.detail ?? 'Rule engine disagreement flagged.'} Escalating to compliance review.`
        : 'Conflict detector audited verdict. Rule engine agreement confirmed. No escalation required.',
      structuredOutput: c,
    })
  }

  // Escalation system message
  if (state.escalated) {
    messages.push({
      type: 'system',
      systemType: 'escalation',
      text: 'Escalation triggered — flagged for compliance review',
    })
  }

  // Halt after A4
  if (state.halt && !state.suitability_report) {
    messages.push({ type: 'system', systemType: 'halt', text: `Pipeline halted: ${state.halt_reason}` })
    return messages
  }

  // A5 — disclosure agent
  if (state.suitability_report) {
    const r = state.suitability_report
    messages.push({
      type: 'agent',
      agentId: 'A5',
      stepNumber: step++,
      content: `Suitability report generated. Decision: ${r.decision} under MiFID II Article 25(2).${state.escalated ? ' Compliance escalation noted in report.' : ''}`,
      structuredOutput: r,
    })
  }

  return messages
}
