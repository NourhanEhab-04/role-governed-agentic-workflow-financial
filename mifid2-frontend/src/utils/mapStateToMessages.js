// ── Verifier helpers ──────────────────────────────────────────────────────────

/**
 * Build the AV bubble content string from a verification result object.
 * Returns null if the result is missing or errored (passed === null).
 */
function _avContent(verification, agentLabel) {
  if (!verification) return null
  // Errored verifier
  if (verification.passed === null || verification.passed === undefined) {
    const err = verification.issues?.[0] ?? 'unknown error'
    return `Verifier for ${agentLabel} encountered an error: ${err}`
  }
  const pct = verification.confidence != null
    ? ` (confidence ${Math.round(verification.confidence * 100)}%)`
    : ''
  if (verification.passed) {
    return `Verifier confirmed ${agentLabel} output is faithful to the input${pct}. All field checks passed.`
  }
  const issueCount = (verification.issues ?? []).length
  const fieldsFailed = Object.entries(verification.field_checks ?? {})
    .filter(([, v]) => v.supported === false)
    .map(([k]) => k)
    .join(', ')
  return `Verifier flagged ${agentLabel} output${pct}. ${issueCount} issue(s) detected.${fieldsFailed ? ` Unsupported fields: ${fieldsFailed}.` : ''}`
}

/**
 * Build the retry bubble content string shown after a feedback-driven retry.
 */
function _retryContent(correction, agentLabel) {
  if (!correction) return null
  if (correction.error) {
    return `${agentLabel} corrector failed: ${correction.error}`
  }
  if (!correction.corrected) {
    return `${agentLabel} corrector produced no output.`
  }
  const fields = (correction.fields_fixed ?? []).join(', ') || 'none'
  return `${agentLabel} corrected with verifier feedback. Fields fixed: ${fields}.`
}

/**
 * Emit consistency + verifier + correction + re-verify messages.
 *
 * consistencyIssues  : string[]
 * verification       : object | undefined  (first AV run)
 * retry              : object | undefined  (corrector output, only if AV failed)
 * finalVerification  : object | undefined  (second AV run after correction)
 * agentLabel         : e.g. "A1" or "A2"
 * stepRef            : { n }
 */
function _emitVerifierMessages(
  messages,
  consistencyIssues,
  verification,
  retry,
  finalVerification,
  agentLabel,
  stepRef,
) {
  // 1. Consistency issues (deterministic, no LLM)
  if (consistencyIssues && consistencyIssues.length > 0) {
    messages.push({
      type: 'system',
      systemType: 'consistency',
      text: `${agentLabel} consistency: ${consistencyIssues.length} issue(s) — ${consistencyIssues[0]}${consistencyIssues.length > 1 ? ` (+${consistencyIssues.length - 1} more)` : ''}`,
    })
  }

  // 2. First AV bubble
  if (verification !== undefined && verification !== null) {
    const content = _avContent(verification, agentLabel)
    if (content) {
      const isWarn = verification.passed === false || verification.passed === null
      messages.push({
        type: 'agent',
        agentId: 'AV',
        stepNumber: stepRef.n++,
        content,
        structuredOutput: verification,
        verifierPassed: verification.passed,
        _systemType: isWarn ? 'verification_warn' : 'verification_pass',
      })
    }
  }

  // 3. Correction bubble — only when AV failed and corrector ran
  if (retry !== undefined && retry !== null) {
    const content = _retryContent(retry, agentLabel)
    if (content) {
      const isError = !!retry.error || !retry.corrected
      messages.push({
        type: 'agent',
        agentId: agentLabel,
        stepNumber: stepRef.n++,
        content,
        structuredOutput: retry.corrected ?? null,
        _systemType: isError ? 'retry_error' : 'retry_applied',
        isRetry: true,
      })
    }
  }

  // 4. Re-verification AV bubble — only present when correction was applied
  if (finalVerification !== undefined && finalVerification !== null) {
    const content = _avContent(finalVerification, agentLabel)
    if (content) {
      const isWarn = finalVerification.passed === false || finalVerification.passed === null
      messages.push({
        type: 'agent',
        agentId: 'AV',
        stepNumber: stepRef.n++,
        content: `[Re-verify] ${content}`,
        structuredOutput: finalVerification,
        verifierPassed: finalVerification.passed,
        _systemType: isWarn ? 'verification_warn' : 'verification_pass',
      })
    }
  }
}

export function mapStateToMessages(state) {
  const messages = []
  // Use an object so _emitVerifierMessages can mutate the counter
  const stepRef = { n: 1 }

  // A0 — always first
  messages.push({
    type: 'agent',
    agentId: 'A0',
    stepNumber: stepRef.n++,
    content: state.halt && !state.client_profile
      ? `Pipeline halted before profiling. Reason: ${state.halt_reason}`
      : 'Pipeline started. Running pre-validation checks on inputs.',
    structuredOutput: null,
  })

  // A1 — client profiler
  if (state.client_profile) {
    // init_p is the raw first parse; p is the final (possibly corrected) output.
    const init_p = state.a1_initial_profile ?? state.client_profile
    messages.push({
      type: 'agent',
      agentId: 'A1',
      stepNumber: stepRef.n++,
      content: `Client profiler parsed input. ${init_p.financial_knowledge} knowledge, risk ${init_p.risk_tolerance_score}/10, ${init_p.investment_horizon}-year horizon, €${init_p.liquid_assets?.toLocaleString()} liquid assets. Vulnerability: ${init_p.financial_vulnerability ?? 'UNKNOWN'}.`,
      structuredOutput: init_p,
    })

    // AV check → correction → AV re-check — each appears as its own visible bubble.
    _emitVerifierMessages(
      messages,
      state.a1_consistency_issues,
      state.a1_verification,
      state.a1_correction,
      state.a1_final_verification,
      'A1',
      stepRef,
    )
  }

  if (state.halt && !state.product_profile) {
    messages.push({ type: 'system', systemType: 'halt', text: `Pipeline halted: ${state.halt_reason}` })
    return messages
  }

  // A2 — product classifier
  if (state.product_profile) {
    const init_p2 = state.a2_initial_profile ?? state.product_profile
    messages.push({
      type: 'agent',
      agentId: 'A2',
      stepNumber: stepRef.n++,
      content: `Product classifier parsed input. ${init_p2.product_name ?? 'Product'}, risk class ${init_p2.risk_class}, requires ${init_p2.requires_knowledge_level} knowledge, minimum ${init_p2.minimum_horizon}-year horizon.${init_p2.potential_loss === 'total' ? ' Has total loss potential.' : ''}${init_p2.leverage ? ' Leveraged product.' : ''}`,
      structuredOutput: init_p2,
    })

    // AV check → correction → AV re-check — each appears as its own visible bubble.
    _emitVerifierMessages(
      messages,
      state.a2_consistency_issues,
      state.a2_verification,
      state.a2_correction,
      state.a2_final_verification,
      'A2',
      stepRef,
    )

    // Cross-consistency check (A1 × A2)
    if (state.cross_consistency_issues && state.cross_consistency_issues.length > 0) {
      messages.push({
        type: 'system',
        systemType: 'consistency',
        text: `A1×A2 cross-check: ${state.cross_consistency_issues.length} issue(s) — ${state.cross_consistency_issues[0]}${state.cross_consistency_issues.length > 1 ? ` (+${state.cross_consistency_issues.length - 1} more)` : ''}`,
      })
    }
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
      ? Object.entries(v.rules)
          .filter(([, r]) => typeof r === 'string' ? r === 'FAIL' : !(r.pass_ ?? r.passed ?? r.pass))
          .map(([k]) => k)
          .join(', ')
      : 'none'
    messages.push({
      type: 'agent',
      agentId: 'A3',
      stepNumber: stepRef.n++,
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
      stepNumber: stepRef.n++,
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
      stepNumber: stepRef.n++,
      content: `Suitability report generated. Decision: ${r.decision} under MiFID II Article 25(2).${state.escalated ? ' Compliance escalation noted in report.' : ''}`,
      structuredOutput: r,
    })
  }

  return messages
}
