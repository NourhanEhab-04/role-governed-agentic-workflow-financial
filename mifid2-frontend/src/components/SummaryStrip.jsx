// src/components/SummaryStrip.jsx

const RULE_LABELS = {
  R1: "Knowledge & Experience",
  R2: "Risk Tolerance",
  R3: "Investment Horizon",
  R4: "Affordability",
  R5: "Vulnerability",
  R6: "Leverage",
  R7: "Product Complexity",
};

// Normalize any verdict string to a display-safe lowercase key
function normalizeVerdict(v) {
  if (!v) return null;
  return v.toString().toLowerCase(); // "SUITABLE" → "suitable", "CONDITIONAL" → "conditional"
}

function verdictColor(verdict) {
  const v = normalizeVerdict(verdict);
  if (!v) return "bg-gray-100 text-gray-400 border-gray-200";
  if (v === "suitable") return "bg-green-50 text-green-700 border-green-200";
  if (v === "unsuitable") return "bg-red-50 text-red-700 border-red-200";
  if (v === "conditional") return "bg-amber-50 text-amber-700 border-amber-200";
  if (v === "escalate" || v === "escalated") return "bg-amber-50 text-amber-700 border-amber-200";
  if (v === "halt") return "bg-red-100 text-red-800 border-red-300";
  return "bg-gray-50 text-gray-600 border-gray-200";
}

function verdictDot(verdict) {
  const v = normalizeVerdict(verdict);
  if (!v) return "bg-gray-300";
  if (v === "suitable") return "bg-green-500";
  if (v === "unsuitable") return "bg-red-500";
  if (v === "conditional") return "bg-amber-500";
  if (v === "escalate" || v === "escalated") return "bg-amber-500";
  if (v === "halt") return "bg-red-700";
  return "bg-gray-400";
}

function VerdictColumn({ label, agentLabel, verdict }) {
  return (
    <div className={`flex-1 rounded-lg border p-3 ${verdictColor(verdict)}`}>
      <div className="text-xs font-semibold uppercase tracking-wide opacity-60 mb-1">{label}</div>
      <div className="text-xs opacity-50 mb-2">{agentLabel}</div>
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${verdictDot(verdict)}`} />
        <span className="font-semibold text-sm capitalize">
          {verdict ? verdict.toString().toLowerCase() : "—"}
        </span>
      </div>
    </div>
  );
}

function AgreementBadge({ verdicts }) {
  const filled = verdicts.filter(Boolean);
  if (filled.length < 2) return null;

  // Normalize all to lowercase for comparison
  const normed = filled.map(normalizeVerdict);
  const allSame = normed.every((v) => v === normed[0]);

  return (
    <div
      className={`text-xs font-medium px-2 py-0.5 rounded-full border ${
        allSame
          ? "bg-green-50 text-green-700 border-green-200"
          : "bg-amber-50 text-amber-700 border-amber-200"
      }`}
    >
      {allSame ? "✓ All agents agree" : "⚠ Agents disagree"}
    </div>
  );
}

function ScoreBar({ score }) {
  const pct = Math.max(0, Math.min(100, score ?? 0));
  const color = pct >= 70 ? "bg-green-500" : pct >= 40 ? "bg-amber-500" : "bg-red-500";

  return (
    <div>
      <div className="flex justify-between text-xs text-gray-500 mb-1">
        <span>Rule Engine Score</span>
        <span className="font-semibold text-gray-700">{pct}/100</span>
      </div>
      <div className="h-2 rounded-full bg-gray-100 overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-500 ${color}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

// rules: List[RuleResult] where each is { rule, pass_, penalty, detail }
// Also handles legacy dict shape: { R1: { passed, penalty, detail }, ... }
function normalizeRules(rules) {
  if (Array.isArray(rules)) return rules;
  if (rules && typeof rules === 'object') {
    return Object.entries(rules).map(([id, r]) => ({
      rule: id,
      pass_: typeof r === 'string' ? r === 'PASS' : (r.pass_ ?? r.passed ?? r.pass ?? false),
      penalty: typeof r === 'object' ? (r.penalty ?? 0) : 0,
      detail: typeof r === 'object' ? (r.detail ?? '') : '',
    }));
  }
  return [];
}

function RuleChecklist({ rules }) {
  const normalized = normalizeRules(rules);
  if (!normalized.length) return null;

  return (
    <div>
      <div className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">
        Rule Breakdown
      </div>
      <div className="space-y-1">
        {normalized.map((r) => {
          const label = RULE_LABELS[r.rule] ?? r.rule;
          const passed = r.pass_ ?? r.pass ?? r.passed;
          return (
            <div
              key={r.rule}
              className={`flex items-start gap-2 text-xs px-2 py-1.5 rounded ${
                passed ? "text-green-700 bg-green-50" : "text-red-700 bg-red-50"
              }`}
            >
              {/* Rule ID badge */}
              <span className="font-mono font-bold flex-shrink-0 mt-0.5">
                {r.rule.split("_")[0]}
              </span>
              {/* Label + detail */}
              <div className="flex-1 min-w-0">
                <div className="font-medium">{label}</div>
                {r.detail && (
                  <div className="opacity-70 mt-0.5 leading-snug">{r.detail}</div>
                )}
              </div>
              {/* Pass/fail + penalty */}
              <div className="flex-shrink-0 flex flex-col items-end gap-0.5">
                <span>{passed ? "✓" : "✗"}</span>
                {r.penalty < 0 && (
                  <span className="text-red-600 font-semibold">{r.penalty}</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ── VerificationPanel ────────────────────────────────────────────────────────

function ConfidenceBar({ confidence }) {
  if (confidence == null) return null
  const pct = Math.round(confidence * 100)
  const color = confidence >= 0.8 ? 'bg-green-400' : confidence >= 0.6 ? 'bg-amber-400' : 'bg-red-400'
  return (
    <div className="flex items-center gap-2">
      <span className="text-xs text-gray-400 w-20 flex-shrink-0">Confidence</span>
      <div className="flex-1 h-1.5 rounded-full bg-gray-100 overflow-hidden">
        <div className={`h-full rounded-full ${color}`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs text-gray-500 w-8 text-right flex-shrink-0">{pct}%</span>
    </div>
  )
}

function VerificationRow({ label, consistencyIssues, verification, correction, finalVerification }) {
  const hasConsistency   = consistencyIssues && consistencyIssues.length > 0
  const hasVerification  = verification !== undefined && verification !== null
  const hasCorrection    = correction   !== undefined && correction   !== null
  const hasFinalVerify   = finalVerification !== undefined && finalVerification !== null
  const correctionOk     = hasCorrection && correction.corrected != null && !correction.error

  // Determine overall status — final verification wins if present
  const effectiveVerification = hasFinalVerify ? finalVerification : verification
  let statusColor, statusDot, statusText
  if (!hasVerification && !hasConsistency) {
    statusColor = 'text-gray-400'; statusDot = 'bg-gray-300'; statusText = 'Not sampled'
  } else if (effectiveVerification?.passed === null) {
    statusColor = 'text-gray-500'; statusDot = 'bg-gray-400'; statusText = 'Verifier error'
  } else if (effectiveVerification?.passed === false || hasConsistency) {
    statusColor = 'text-amber-700'; statusDot = 'bg-amber-400'
    statusText = effectiveVerification?.passed === false ? 'Issues found' : 'Consistency flags'
  } else {
    statusColor = 'text-green-700'; statusDot = 'bg-green-500'
    statusText = correctionOk ? 'Corrected ✓' : 'Verified'
  }

  return (
    <div className="space-y-1">
      {/* Header */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-gray-600">{label}</span>
        <div className="flex items-center gap-1.5">
          <span className={`w-2 h-2 rounded-full flex-shrink-0 ${statusDot}`} />
          <span className={`text-xs font-medium ${statusColor}`}>{statusText}</span>
        </div>
      </div>

      {/* Consistency issues */}
      {hasConsistency && (
        <div className="space-y-0.5 pl-2">
          {consistencyIssues.slice(0, 3).map((issue, i) => (
            <div key={i} className="text-xs text-orange-700 bg-orange-50 rounded px-2 py-0.5 leading-snug">{issue}</div>
          ))}
          {consistencyIssues.length > 3 && (
            <div className="text-xs text-orange-500 pl-2">+{consistencyIssues.length - 3} more</div>
          )}
        </div>
      )}

      {/* First verifier result */}
      {hasVerification && verification.passed !== null && (
        <div className="pl-2 space-y-0.5">
          <ConfidenceBar confidence={verification.confidence} />
          {verification.issues?.slice(0, 2).map((issue, i) => (
            <div key={i} className="text-xs text-amber-700 bg-amber-50 rounded px-2 py-0.5 leading-snug">{issue}</div>
          ))}
          {verification.issues?.length > 2 && (
            <div className="text-xs text-amber-500 pl-2">+{verification.issues.length - 2} more</div>
          )}
        </div>
      )}

      {/* Correction block */}
      {hasCorrection && (
        <div className="pl-2 space-y-0.5">
          {correction.error ? (
            <div className="text-xs text-red-700 bg-red-50 rounded px-2 py-0.5">
              Corrector error: {correction.error}
            </div>
          ) : correctionOk ? (
            <div className="text-xs text-teal-700 bg-teal-50 rounded px-2 py-1 space-y-0.5">
              <div className="font-medium">✎ Correction applied</div>
              {correction.fields_fixed?.length > 0 && (
                <div className="opacity-80">Fixed: {correction.fields_fixed.join(', ')}</div>
              )}
            </div>
          ) : null}
        </div>
      )}

      {/* Re-verification result after correction */}
      {hasFinalVerify && finalVerification.passed !== null && (
        <div className="pl-2 space-y-0.5">
          <div className="text-xs text-gray-400 font-medium">Re-verify after correction</div>
          <ConfidenceBar confidence={finalVerification.confidence} />
          {finalVerification.issues?.slice(0, 2).map((issue, i) => (
            <div key={i} className="text-xs text-amber-700 bg-amber-50 rounded px-2 py-0.5 leading-snug">{issue}</div>
          ))}
        </div>
      )}
    </div>
  )
}

function VerificationPanel({ state }) {
  const hasAny =
    state.a1_consistency_issues?.length > 0 ||
    state.a1_verification != null ||
    state.a1_correction != null ||
    state.a2_consistency_issues?.length > 0 ||
    state.a2_verification != null ||
    state.a2_correction != null ||
    state.cross_consistency_issues?.length > 0

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold uppercase tracking-wide text-gray-400">
        Verification Layer
      </div>

      {!hasAny ? (
        <div className="text-xs text-gray-400 italic">
          No consistency issues detected. Verifier not sampled.
        </div>
      ) : (
        <div className="space-y-3">
          <VerificationRow
            label="A1 — Client Profile"
            consistencyIssues={state.a1_consistency_issues}
            verification={state.a1_verification}
            correction={state.a1_correction}
            finalVerification={state.a1_final_verification}
          />
          <VerificationRow
            label="A2 — Product Profile"
            consistencyIssues={state.a2_consistency_issues}
            verification={state.a2_verification}
            correction={state.a2_correction}
            finalVerification={state.a2_final_verification}
          />

          {/* Cross-check issues */}
          {state.cross_consistency_issues?.length > 0 && (
            <div className="space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium text-gray-600">A1×A2 Cross-check</span>
                <div className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full flex-shrink-0 bg-amber-400" />
                  <span className="text-xs font-medium text-amber-700">
                    {state.cross_consistency_issues.length} flag(s)
                  </span>
                </div>
              </div>
              <div className="space-y-0.5 pl-2">
                {state.cross_consistency_issues.slice(0, 3).map((issue, i) => (
                  <div key={i} className="text-xs text-amber-700 bg-amber-50 rounded px-2 py-0.5 leading-snug">
                    {issue}
                  </div>
                ))}
                {state.cross_consistency_issues.length > 3 && (
                  <div className="text-xs text-amber-500 pl-2">
                    +{state.cross_consistency_issues.length - 3} more
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function SummaryStrip({ state }) {
  if (!state) {
    return (
      <div className="p-4 text-sm text-gray-400 italic">
        Run an assessment to see the summary.
      </div>
    );
  }

  const preVerdict = state.pre_check_verdict?.decision ?? state.pre_check_verdict ?? null;
  const ruleVerdict = state.rule_verdict ?? null;          // RuleVerdict object
  const overallRuleVerdict = ruleVerdict?.decision ?? null; // Decision enum value
  const auditVerdict = state.audit_verdict?.a4_decision ?? null;
  const explanation = state.suitability_report ?? null;    // A5 free text string

  const finalVerdict = state.halt
    ? "Halt"
    : overallRuleVerdict ?? preVerdict ?? null;

  return (
    <div className="space-y-4 p-4">

      {/* Three-column verdict panel */}
      <div>
        <div className="flex items-center justify-between mb-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-gray-400">
            Three-Point Verdict Agreement
          </span>
          <AgreementBadge verdicts={[preVerdict, overallRuleVerdict, auditVerdict]} />
        </div>
        <div className="flex gap-2">
          <VerdictColumn label="Pre-Check"  agentLabel="A1 / A2"           verdict={preVerdict} />
          <VerdictColumn label="Rule Engine" agentLabel="A3 deterministic" verdict={overallRuleVerdict} />
          <VerdictColumn label="Audit"       agentLabel="A4 override check" verdict={auditVerdict} />
        </div>
      </div>

      {/* Final decision card */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 space-y-4 shadow-sm">

        {/* Halt banner */}
        {state.halt && (
          <div className="rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700 font-medium">
            🛑 Pipeline halted — {state.halt_reason ?? "no reason provided"}
          </div>
        )}

        {/* Escalation badge */}
        {state.escalated && !state.halt && (
          <div className="rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-sm text-amber-700 font-medium">
            ⚠ Escalated for human review
          </div>
        )}

        {/* Final verdict headline */}
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-400 uppercase tracking-wide font-semibold">
            Final Decision
          </span>
          {finalVerdict && (
            <span className={`px-2.5 py-0.5 rounded-full text-sm font-semibold border ${verdictColor(finalVerdict)}`}>
              {finalVerdict.toString().toLowerCase()}
            </span>
          )}
        </div>

        {/* Score bar — from rule_verdict.score */}
        {ruleVerdict?.score != null && <ScoreBar score={ruleVerdict.score} />}

        {/* Rule checklist — prefer pre_check_verdict.rules (full objects with penalty+detail);
            fall back to rule_verdict.rules (A3 LLM strings) when deterministic engine didn't run */}
        {(state.pre_check_verdict?.rules ?? ruleVerdict?.rules) && (
          <RuleChecklist rules={state.pre_check_verdict?.rules ?? ruleVerdict?.rules} />
        )}

        {/* Verification layer — AV results */}
        <VerificationPanel state={state} />

        {/* A5 explanation — suitability_report is an object */}
        {explanation && (
          <div>
            <div className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-1">
              A5 Disclosure
            </div>
            {typeof explanation === 'string' ? (
              <p className="text-sm text-gray-700 leading-relaxed">{explanation}</p>
            ) : (
              <div className="space-y-1 text-sm text-gray-700">
                {explanation.client_facing_summary && (
                  <p className="leading-relaxed">{explanation.client_facing_summary}</p>
                )}
                {explanation.summary && explanation.summary !== explanation.client_facing_summary && (
                  <p className="leading-relaxed text-gray-500">{explanation.summary}</p>
                )}
                {explanation.regulatory_basis && (
                  <p className="text-xs text-gray-400 italic">{explanation.regulatory_basis}</p>
                )}
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}