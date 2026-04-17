
import { mapStateToMessages } from "../src/utils/mapStateToMessages.js"
const state1 = {
  client_profile: { knowledge_level: 'basic', risk_tolerance: 3, investment_horizon: 5, liquid_assets: 50000, vulnerability: 'NONE' },
  product_profile: { product_type: 'Leveraged ETF', risk_class: 6, required_knowledge: 'moderate', min_horizon: 3, total_loss_potential: true, is_leveraged: true },
  pre_check_verdict: { decision: 'UNSUITABLE', score: 35 },
  rule_verdict: { score: 35, decision: 'UNSUITABLE', rules: { R1: { passed: false }, R2: { passed: false }, R6: { passed: false } } },
  audit_verdict: { agreed: true },
  conflict_report: { escalate: false },
  escalated: false,
  suitability_report: { decision: 'UNSUITABLE', score: 35 },
  halt: false,
}
console.log(mapStateToMessages(state1))
const state2 = {
  ...state1,
  audit_verdict: { agreed: false, detail: 'A3 reported SUITABLE but audit found UNSUITABLE' },
  conflict_report: { escalate: true, detail: 'Rule engine disagreement detected' },
  escalated: true,
  suitability_report: { decision: 'UNSUITABLE', score: 35 },
}
console.log(mapStateToMessages(state2))
const state3 = {
  client_profile: { knowledge_level: 'basic', risk_tolerance: 3, investment_horizon: 5, liquid_assets: 50000 },
  product_profile: { product_type: 'ETF', risk_class: 4, required_knowledge: 'basic', min_horizon: 2, total_loss_potential: false },
  pre_check_verdict: null,
  rule_verdict: null,
  audit_verdict: null,
  conflict_report: null,
  escalated: false,
  suitability_report: null,
  halt: true,
  halt_reason: 'A2 failed after 2 attempts',
}
console.log(mapStateToMessages(state3))
const state4 = {
  client_profile: { knowledge_level: 'moderate', risk_tolerance: 5, investment_horizon: 3, liquid_assets: 20000 },
  product_profile: null,
  pre_check_verdict: null,
  rule_verdict: null,
  audit_verdict: null,
  conflict_report: null,
  escalated: false,
  suitability_report: null,
  halt: true,
  halt_reason: 'A1 failed after 2 attempts',
}
console.log(mapStateToMessages(state4))