/**
 * mapStateToMessages.test.mjs
 * Plain Node.js tests — no framework needed.
 * Run: node src/utils/mapStateToMessages.test.mjs
 */

import assert from 'node:assert/strict'
import { mapStateToMessages } from './mapStateToMessages.js'

// ── helpers ───────────────────────────────────────────────────────────────────

let passed = 0
let failed = 0

function test(name, fn) {
  try {
    fn()
    console.log(`  ✓ ${name}`)
    passed++
  } catch (err) {
    console.error(`  ✗ ${name}`)
    console.error(`      ${err.message}`)
    failed++
  }
}

function msgs(state) { return mapStateToMessages(state) }

function byAgent(messages, agentId) {
  return messages.filter(m => m.type === 'agent' && m.agentId === agentId)
}
function bySystemType(messages, systemType) {
  return messages.filter(m => m.type === 'system' && m.systemType === systemType)
}

// ── Fixtures ──────────────────────────────────────────────────────────────────

const CLIENT = {
  financial_knowledge: 'moderate',
  risk_tolerance_score: 5,
  investment_horizon: 5,
  liquid_assets: 20000,
  income: 50000,
  investment_amount: 5000,
  can_afford_total_loss: false,
  financial_vulnerability: 'LOW',
}

const PRODUCT = {
  product_name: 'Global Equity ETF',
  risk_class: 4,
  complexity_tier: 'NON-COMPLEX',
  requires_knowledge_level: 'basic',
  minimum_horizon: 3,
  potential_loss: 'partial',
  leverage: false,
}

const VERIFICATION_PASS = {
  passed: true,
  confidence: 0.95,
  issues: [],
  field_checks: { financial_knowledge: { supported: true, reason: 'Clear from input.' } },
}

const VERIFICATION_WARN = {
  passed: false,
  confidence: 0.55,
  issues: ['risk_tolerance_score seems too high given input'],
  field_checks: {
    risk_tolerance_score: { supported: false, reason: 'Client said cautious but 8 returned.' },
  },
}

const VERIFICATION_ERROR = {
  passed: null,
  confidence: null,
  issues: ['Verifier error: API timeout'],
  field_checks: {},
}

// ── Suite: baseline (no verifier state) ──────────────────────────────────────

console.log('\nBaseline — no verifier fields in state')

test('A0 is always first message', () => {
  const m = msgs({})
  assert.equal(m[0].agentId, 'A0')
})

test('A1 bubble appears when client_profile is present', () => {
  const m = msgs({ client_profile: CLIENT })
  assert.equal(byAgent(m, 'A1').length, 1)
})

test('No AV bubble when a1_verification is absent', () => {
  const m = msgs({ client_profile: CLIENT })
  assert.equal(byAgent(m, 'AV').length, 0)
})

test('No consistency message when a1_consistency_issues is absent', () => {
  const m = msgs({ client_profile: CLIENT })
  assert.equal(bySystemType(m, 'consistency').length, 0)
})

// ── Suite: consistency issues ─────────────────────────────────────────────────

console.log('\nConsistency issues')

test('A1 consistency system message emitted when issues list is non-empty', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_consistency_issues: ['income is 0 but vulnerability is LOW'],
  })
  const sys = bySystemType(m, 'consistency')
  assert.equal(sys.length, 1)
  assert.ok(sys[0].text.includes('A1 consistency'))
})

test('A1 consistency message shows first issue text', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_consistency_issues: ['income is 0 but vulnerability is LOW'],
  })
  assert.ok(bySystemType(m, 'consistency')[0].text.includes('income is 0'))
})

test('A1 consistency message shows +N more when multiple issues', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_consistency_issues: ['issue one', 'issue two', 'issue three'],
  })
  assert.ok(bySystemType(m, 'consistency')[0].text.includes('+2 more'))
})

test('A1 consistency message NOT emitted when issues list is empty', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_consistency_issues: [],
  })
  assert.equal(bySystemType(m, 'consistency').length, 0)
})

test('A2 consistency message emitted when a2_consistency_issues is non-empty', () => {
  const m = msgs({
    client_profile: CLIENT,
    product_profile: PRODUCT,
    a2_consistency_issues: ['risk_class 7 must have leverage=true'],
  })
  const sys = bySystemType(m, 'consistency')
  assert.equal(sys.length, 1)
  assert.ok(sys[0].text.includes('A2 consistency'))
})

test('cross_consistency_issues message emitted when non-empty', () => {
  const m = msgs({
    client_profile: CLIENT,
    product_profile: PRODUCT,
    cross_consistency_issues: ['client knowledge below product requirement'],
  })
  const sys = bySystemType(m, 'consistency')
  assert.ok(sys.some(s => s.text.includes('A1×A2 cross-check')))
})

test('cross_consistency_issues NOT emitted when empty', () => {
  const m = msgs({
    client_profile: CLIENT,
    product_profile: PRODUCT,
    cross_consistency_issues: [],
  })
  assert.ok(!bySystemType(m, 'consistency').some(s => s.text.includes('cross-check')))
})

// ── Suite: AV bubble — passed ─────────────────────────────────────────────────

console.log('\nAV bubble — verifier passed')

test('AV bubble appears when a1_verification is present', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_PASS })
  assert.equal(byAgent(m, 'AV').length, 1)
})

test('AV bubble content says "confirmed" when passed=true', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_PASS })
  assert.ok(byAgent(m, 'AV')[0].content.includes('confirmed'))
})

test('AV bubble content includes confidence percentage', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_PASS })
  assert.ok(byAgent(m, 'AV')[0].content.includes('95%'))
})

test('AV bubble structuredOutput is the verification object', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_PASS })
  assert.deepEqual(byAgent(m, 'AV')[0].structuredOutput, VERIFICATION_PASS)
})

test('AV bubble has a stepNumber', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_PASS })
  assert.ok(typeof byAgent(m, 'AV')[0].stepNumber === 'number')
})

// ── Suite: AV bubble — warned ─────────────────────────────────────────────────

console.log('\nAV bubble — verifier flagged issues')

test('AV bubble appears when a1_verification has passed=false', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_WARN })
  assert.equal(byAgent(m, 'AV').length, 1)
})

test('AV bubble content says "flagged" when passed=false', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_WARN })
  assert.ok(byAgent(m, 'AV')[0].content.includes('flagged'))
})

test('AV bubble lists unsupported fields when passed=false', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_WARN })
  assert.ok(byAgent(m, 'AV')[0].content.includes('risk_tolerance_score'))
})

test('AV bubble _systemType is verification_warn when flagged', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_WARN })
  assert.equal(byAgent(m, 'AV')[0]._systemType, 'verification_warn')
})

test('AV bubble _systemType is verification_pass when passed=true', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_PASS })
  assert.equal(byAgent(m, 'AV')[0]._systemType, 'verification_pass')
})

// ── Suite: AV bubble — error ──────────────────────────────────────────────────

console.log('\nAV bubble — verifier errored')

test('AV bubble appears even when verifier errored (passed=null)', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_ERROR })
  assert.equal(byAgent(m, 'AV').length, 1)
})

test('AV error bubble content contains error message', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_ERROR })
  assert.ok(byAgent(m, 'AV')[0].content.includes('API timeout'))
})

// ── Suite: A2 verifier ────────────────────────────────────────────────────────

console.log('\nA2 verifier')

test('AV bubble for A2 appears when a2_verification is present', () => {
  const m = msgs({
    client_profile: CLIENT,
    product_profile: PRODUCT,
    a2_verification: VERIFICATION_PASS,
  })
  // There should be exactly one AV bubble (A1 has no verification here)
  assert.equal(byAgent(m, 'AV').length, 1)
  assert.ok(byAgent(m, 'AV')[0].content.includes('A2'))
})

test('Both A1 and A2 AV bubbles appear when both verifications present', () => {
  const m = msgs({
    client_profile: CLIENT,
    product_profile: PRODUCT,
    a1_verification: VERIFICATION_PASS,
    a2_verification: VERIFICATION_WARN,
  })
  assert.equal(byAgent(m, 'AV').length, 2)
})

// ── Suite: ordering ───────────────────────────────────────────────────────────

console.log('\nMessage ordering')

test('A1 bubble comes before AV bubble', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_PASS })
  const a1Idx = m.findIndex(x => x.type === 'agent' && x.agentId === 'A1')
  const avIdx = m.findIndex(x => x.type === 'agent' && x.agentId === 'AV')
  assert.ok(a1Idx < avIdx)
})

test('A1 consistency message comes after A1 bubble and before AV bubble', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_consistency_issues: ['some issue'],
    a1_verification: VERIFICATION_PASS,
  })
  const a1Idx = m.findIndex(x => x.type === 'agent' && x.agentId === 'A1')
  const consIdx = m.findIndex(x => x.type === 'system' && x.systemType === 'consistency')
  const avIdx = m.findIndex(x => x.type === 'agent' && x.agentId === 'AV')
  assert.ok(a1Idx < consIdx, 'A1 before consistency')
  assert.ok(consIdx < avIdx, 'consistency before AV')
})

test('A2 bubble comes after AV-for-A1 bubble', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_PASS,
    product_profile: PRODUCT,
  })
  const avIdx = m.findIndex(x => x.type === 'agent' && x.agentId === 'AV')
  const a2Idx = m.findIndex(x => x.type === 'agent' && x.agentId === 'A2')
  assert.ok(avIdx < a2Idx)
})

test('cross_consistency message comes after A2-AV bubble', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_PASS,
    product_profile: PRODUCT,
    a2_verification: VERIFICATION_WARN,
    cross_consistency_issues: ['knowledge mismatch'],
  })
  const avIdxes = m.reduce((acc, x, i) => (x.type === 'agent' && x.agentId === 'AV' ? [...acc, i] : acc), [])
  const lastAvIdx = avIdxes[avIdxes.length - 1]
  const crossIdx = m.findIndex(x => x.type === 'system' && x.text.includes('A1×A2'))
  assert.ok(lastAvIdx < crossIdx, `AV at ${lastAvIdx}, cross at ${crossIdx}`)
})

test('step numbers are monotonically increasing across all agent bubbles', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_PASS,
    product_profile: PRODUCT,
    a2_verification: VERIFICATION_PASS,
  })
  const steps = m.filter(x => x.type === 'agent').map(x => x.stepNumber)
  for (let i = 1; i < steps.length; i++) {
    assert.ok(steps[i] > steps[i - 1], `step ${steps[i]} not > ${steps[i - 1]}`)
  }
})

// ── Suite: AC corrector bubble ────────────────────────────────────────────────

console.log('\nAC bubble — corrector')

const CORRECTION_OK = {
  original: { risk_tolerance_score: 8 },
  corrected: { risk_tolerance_score: 3 },
  fields_fixed: ['risk_tolerance_score'],
}

const CORRECTION_ERROR = {
  original: { risk_tolerance_score: 8 },
  corrected: null,
  fields_fixed: [],
  error: 'LLM timeout',
}

test('AC bubble appears when a1_correction is present and AV failed', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_OK,
  })
  assert.equal(byAgent(m, 'AC').length, 1)
})

test('AC bubble content mentions fields fixed', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_OK,
  })
  assert.ok(byAgent(m, 'AC')[0].content.includes('risk_tolerance_score'))
})

test('AC bubble _systemType is correction_applied on success', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_OK,
  })
  assert.equal(byAgent(m, 'AC')[0]._systemType, 'correction_applied')
})

test('AC bubble _systemType is correction_error when corrector failed', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_ERROR,
  })
  assert.equal(byAgent(m, 'AC')[0]._systemType, 'correction_error')
})

test('AC error bubble content contains error message', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_ERROR,
  })
  assert.ok(byAgent(m, 'AC')[0].content.includes('LLM timeout'))
})

test('AC bubble NOT emitted when no correction in state', () => {
  const m = msgs({ client_profile: CLIENT, a1_verification: VERIFICATION_PASS })
  assert.equal(byAgent(m, 'AC').length, 0)
})

test('AC bubble structuredOutput is the correction object', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_OK,
  })
  assert.deepEqual(byAgent(m, 'AC')[0].structuredOutput, CORRECTION_OK)
})

// ── Suite: re-verification bubble after correction ────────────────────────────

console.log('\nAV re-verify bubble after correction')

const FINAL_VERIFICATION_PASS = {
  passed: true,
  confidence: 0.92,
  issues: [],
  field_checks: { risk_tolerance_score: { supported: true, reason: 'Now correct.' } },
}

test('Second AV bubble appears when a1_final_verification is present', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_OK,
    a1_final_verification: FINAL_VERIFICATION_PASS,
  })
  assert.equal(byAgent(m, 'AV').length, 2)
})

test('Second AV bubble content mentions re-verify', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_OK,
    a1_final_verification: FINAL_VERIFICATION_PASS,
  })
  const avBubbles = byAgent(m, 'AV')
  assert.ok(avBubbles[1].content.includes('re-verify'))
})

test('Second AV bubble is verification_pass when re-verify passed', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_OK,
    a1_final_verification: FINAL_VERIFICATION_PASS,
  })
  assert.equal(byAgent(m, 'AV')[1]._systemType, 'verification_pass')
})

test('No second AV bubble when no final_verification in state', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_OK,
  })
  assert.equal(byAgent(m, 'AV').length, 1)
})

// ── Suite: full correction ordering ──────────────────────────────────────────

console.log('\nFull correction ordering')

test('AV → AC → AV-reverify order is maintained', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_OK,
    a1_final_verification: FINAL_VERIFICATION_PASS,
  })
  const avIdx1  = m.findIndex(x => x.type === 'agent' && x.agentId === 'AV')
  const acIdx   = m.findIndex(x => x.type === 'agent' && x.agentId === 'AC')
  const avIdxes = m.reduce((acc, x, i) => (x.type === 'agent' && x.agentId === 'AV' ? [...acc, i] : acc), [])
  const avIdx2  = avIdxes[1]
  assert.ok(avIdx1 < acIdx,  `AV(${avIdx1}) must come before AC(${acIdx})`)
  assert.ok(acIdx  < avIdx2, `AC(${acIdx}) must come before AV-reverify(${avIdx2})`)
})

test('step numbers still monotonically increasing with full correction sequence', () => {
  const m = msgs({
    client_profile: CLIENT,
    a1_verification: VERIFICATION_WARN,
    a1_correction: CORRECTION_OK,
    a1_final_verification: FINAL_VERIFICATION_PASS,
    product_profile: PRODUCT,
    a2_verification: VERIFICATION_PASS,
  })
  const steps = m.filter(x => x.type === 'agent').map(x => x.stepNumber)
  for (let i = 1; i < steps.length; i++) {
    assert.ok(steps[i] > steps[i - 1], `step ${steps[i]} not > ${steps[i - 1]}`)
  }
})

// ── Summary ───────────────────────────────────────────────────────────────────

console.log(`\n${'─'.repeat(50)}`)
console.log(`${passed + failed} tests: ${passed} passed, ${failed} failed`)
if (failed > 0) process.exit(1)
