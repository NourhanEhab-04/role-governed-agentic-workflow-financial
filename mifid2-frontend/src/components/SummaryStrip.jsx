// src/components/SummaryStrip.jsx
import { useState } from 'react'

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

// ── ClientProfilePanel ────────────────────────────────────────────────────────

function ClientProfilePanel({ profile }) {
  if (!profile) return null
  const fields = [
    { label: 'Knowledge',    value: profile.financial_knowledge },
    { label: 'Risk',         value: profile.risk_tolerance_score != null ? `${profile.risk_tolerance_score}/10` : null },
    { label: 'Horizon',      value: profile.investment_horizon  != null ? `${profile.investment_horizon} yr`  : null },
    { label: 'Liquid assets',value: profile.liquid_assets        != null ? `€${Number(profile.liquid_assets).toLocaleString()}`       : null },
    { label: 'Income',       value: profile.income               != null ? `€${Number(profile.income).toLocaleString()}`              : null },
    { label: 'Investment',   value: profile.investment_amount    != null ? `€${Number(profile.investment_amount).toLocaleString()}`   : null },
    { label: 'Age',          value: profile.age                  != null ? `${profile.age} yr`                : null },
    { label: 'Vulnerability',value: profile.financial_vulnerability },
    { label: 'Afford loss',  value: profile.can_afford_total_loss != null ? (profile.can_afford_total_loss ? 'Yes' : 'No') : null },
    { label: 'Concentration',value: profile.portfolio_concentration_pct != null ? `${profile.portfolio_concentration_pct}%` : null },
  ].filter(f => f.value != null)

  if (!fields.length) return null

  return (
    <div>
      <div className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">
        Client Profile
      </div>
      <div className="flex flex-wrap gap-1.5">
        {fields.map(f => (
          <span
            key={f.label}
            className="text-xs px-2 py-1 rounded-full bg-blue-50 text-blue-700 border border-blue-100"
          >
            <span className="text-blue-400 font-medium">{f.label}:</span> {f.value}
          </span>
        ))}
      </div>
    </div>
  )
}

// ── ConflictFlagsPanel ────────────────────────────────────────────────────────

function ConflictFlagsPanel({ flags }) {
  if (!flags || flags.length === 0) return null
  const triggered    = flags.filter(f => f.triggered)
  const notTriggered = flags.filter(f => !f.triggered)

  return (
    <div>
      <div className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-2">
        Conflict Flags
      </div>
      <div className="space-y-1">
        {triggered.map(f => (
          <div
            key={f.rule_id}
            className={`flex items-start gap-2 text-xs px-2 py-1.5 rounded border ${
              f.severity === 'HIGH'
                ? 'bg-red-50 text-red-700 border-red-200'
                : 'bg-amber-50 text-amber-700 border-amber-200'
            }`}
          >
            <span className="font-mono font-bold flex-shrink-0 mt-0.5">{f.rule_id}</span>
            <span className="flex-1 leading-snug">{f.message}</span>
            <span className={`flex-shrink-0 text-xs font-bold px-1.5 py-0.5 rounded ${
              f.severity === 'HIGH' ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'
            }`}>{f.severity}</span>
          </div>
        ))}
        {notTriggered.map(f => (
          <div
            key={f.rule_id}
            className="flex items-center gap-2 text-xs px-2 py-1.5 rounded bg-gray-50 text-gray-400 border border-gray-100"
          >
            <span className="font-mono font-bold flex-shrink-0">{f.rule_id}</span>
            <span className="flex-1 opacity-60">{f.message}</span>
            <span className="text-xs opacity-50">{f.severity}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

// ── AuditDetailsPanel ─────────────────────────────────────────────────────────

function AuditDetailsPanel({ auditVerdict }) {
  if (!auditVerdict) return null
  const { agreed, a3_decision, a4_decision, a3_failed_rules, a4_failed_rules } = auditVerdict
  const a3Failed = (a3_failed_rules ?? []).join(', ') || 'none'
  const a4Failed = (a4_failed_rules ?? []).join(', ') || 'none'

  return (
    <div
      className={`rounded-lg border px-3 py-2 text-xs space-y-2 ${
        agreed ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'
      }`}
    >
      <div className={`font-semibold ${agreed ? 'text-green-700' : 'text-red-700'}`}>
        {agreed ? '✓ Three-point audit agreed' : '✗ Three-point audit disagreed'}
      </div>
      <div className="grid grid-cols-2 gap-3 text-gray-600">
        <div className="space-y-0.5">
          <div className="font-medium text-gray-400 uppercase tracking-wide" style={{fontSize:'10px'}}>A3 — LLM tool call</div>
          <div className="font-semibold">{a3_decision ?? '—'}</div>
          <div className="text-gray-400">Failed: {a3Failed}</div>
        </div>
        <div className="space-y-0.5">
          <div className="font-medium text-gray-400 uppercase tracking-wide" style={{fontSize:'10px'}}>A4 — deterministic re-run</div>
          <div className="font-semibold">{a4_decision ?? '—'}</div>
          <div className="text-gray-400">Failed: {a4Failed}</div>
        </div>
      </div>
    </div>
  )
}

// ── RuleFindingsTable ─────────────────────────────────────────────────────────

function RuleFindingsTable({ findings }) {
  const [open, setOpen] = useState(false)
  if (!findings || findings.length === 0) return null

  return (
    <div>
      <button
        onClick={() => setOpen(v => !v)}
        className="text-xs text-gray-400 hover:text-gray-600 flex items-center gap-1 transition-colors"
      >
        <span>{open ? '▲' : '▼'}</span>
        <span>Per-rule explanations ({findings.length} rules)</span>
      </button>
      {open && (
        <div className="mt-2 space-y-1">
          {findings.map(f => (
            <div
              key={f.rule_id}
              className={`text-xs px-2 py-1.5 rounded border ${
                f.status === 'PASS'
                  ? 'bg-green-50 text-green-800 border-green-200'
                  : 'bg-red-50 text-red-800 border-red-200'
              }`}
            >
              <div className="flex items-center gap-2 mb-0.5">
                <span className="font-mono font-bold">{f.rule_id}</span>
                <span className={`font-bold ${f.status === 'PASS' ? 'text-green-600' : 'text-red-600'}`}>
                  {f.status === 'PASS' ? '✓' : '✗'} {f.status}
                </span>
                <span className="opacity-60">{RULE_LABELS[f.rule_id] ?? f.rule_id}</span>
              </div>
              <div className="opacity-80 leading-snug">{f.explanation}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ── FlagsAddressedPanel ───────────────────────────────────────────────────────

function FlagsAddressedPanel({ flagsAddressed }) {
  if (!flagsAddressed || flagsAddressed.length === 0) return null
  return (
    <div className="space-y-1">
      <div className="text-xs font-semibold text-gray-400 uppercase tracking-wide">
        Conflict flags addressed
      </div>
      {flagsAddressed.map((f, i) => (
        <div
          key={i}
          className="text-xs px-2 py-1.5 rounded bg-amber-50 border border-amber-200 text-amber-800 leading-snug"
        >
          <span className="font-mono font-bold mr-1.5">{f.rule_id}</span>
          {f.explanation}
        </div>
      ))}
    </div>
  )
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
  // Verifier always runs — show rows as soon as A1/A2 verification data arrives OR pipeline completed.
  const pipelineComplete = state.rule_verdict != null
  const hasAny =
    pipelineComplete ||
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
          Verification pending.
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

  const preVerdict        = state.pre_check_verdict?.decision ?? state.pre_check_verdict ?? null;
  const ruleVerdict       = state.rule_verdict ?? null;
  const overallRuleVerdict = ruleVerdict?.decision ?? null;
  const auditVerdictObj   = state.audit_verdict ?? null;
  const auditDecision     = auditVerdictObj?.a4_decision ?? null;
  const explanation       = state.suitability_report ?? null;
  const conflictFlags     = state.conflict_report?.flags ?? null;

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
          <AgreementBadge verdicts={[preVerdict, overallRuleVerdict, auditDecision]} />
        </div>
        <div className="flex gap-2">
          <VerdictColumn label="Pre-Check"   agentLabel="A1 / A2"            verdict={preVerdict} />
          <VerdictColumn label="Rule Engine" agentLabel="A3 deterministic"   verdict={overallRuleVerdict} />
          <VerdictColumn label="Audit"       agentLabel="A4 override check"  verdict={auditDecision} />
        </div>
      </div>

      {/* Audit details — A3 vs A4 comparison */}
      {auditVerdictObj && <AuditDetailsPanel auditVerdict={auditVerdictObj} />}

      {/* Client profile — all fields including hidden ones */}
      {state.client_profile && <ClientProfilePanel profile={state.client_profile} />}

      {/* Final decision card */}
      <div className="rounded-xl border border-gray-200 bg-white p-4 space-y-4 shadow-sm">

        {/* Halt banner */}
        {state.halt && (
          <div className="rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-sm text-red-700 font-medium">
            Pipeline halted — {state.halt_reason ?? "no reason provided"}
          </div>
        )}

        {/* Escalation badge */}
        {state.escalated && !state.halt && (
          <div className="rounded-lg bg-amber-50 border border-amber-200 px-3 py-2 text-sm text-amber-700 font-medium">
            ⚠ Escalated for human review
          </div>
        )}

        {/* Conflict flags breakdown */}
        {conflictFlags && <ConflictFlagsPanel flags={conflictFlags} />}

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

        {/* Score bar */}
        {ruleVerdict?.score != null && <ScoreBar score={ruleVerdict.score} />}

        {/* Rule checklist */}
        {(state.pre_check_verdict?.rules ?? ruleVerdict?.rules) && (
          <RuleChecklist rules={state.pre_check_verdict?.rules ?? ruleVerdict?.rules} />
        )}

        {/* Verification layer */}
        <VerificationPanel state={state} />

        {/* A5 Disclosure */}
        {explanation && (
          <div className="space-y-2">
            <div className="text-xs font-semibold uppercase tracking-wide text-gray-400">
              A5 Disclosure
            </div>
            {typeof explanation === 'string' ? (
              <p className="text-sm text-gray-700 leading-relaxed">{explanation}</p>
            ) : (
              <div className="space-y-2 text-sm text-gray-700">
                {explanation.client_facing_summary && (
                  <p className="leading-relaxed">{explanation.client_facing_summary}</p>
                )}
                {explanation.summary && explanation.summary !== explanation.client_facing_summary && (
                  <p className="leading-relaxed text-gray-500">{explanation.summary}</p>
                )}
                {/* Per-rule explanations (collapsible) */}
                <RuleFindingsTable findings={explanation.rule_findings} />
                {/* Conflict flags addressed by A5 */}
                <FlagsAddressedPanel flagsAddressed={explanation.flags_addressed} />
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