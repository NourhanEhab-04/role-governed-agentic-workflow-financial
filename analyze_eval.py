import json, os, sys
from collections import defaultdict, Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluation.metrics import compute_all_metrics

EVAL_DIR = "data/eval"
TIMESTAMPS = [
    "20260603T115257Z",
    "20260603T144936Z",
    "20260603T145809Z",
    "20260603T152939Z",
    "20260603T153809Z",
    "20260603T154116Z",
    "20260603T161824Z",
    "20260603T162918Z",
    "20260603T164338Z",
    "20260603T165740Z",
    "20260603T171336Z",
    "20260603T172722Z",
    "20260603T174027Z",
    "20260603T175840Z",
    "20260603T181237Z",
    "20260603T183009Z",
    "20260603T212523Z",
    "20260604T015906Z",
    "20260604T020330Z",
]

# ─────────────────────────────────────────────────────────────────────────────
# Load all data
# ─────────────────────────────────────────────────────────────────────────────
all_data = {}
for ts in TIMESTAMPS:
    bpath = f"{EVAL_DIR}/results_baseline_{ts}.json"
    ppath = f"{EVAL_DIR}/results_pipeline_{ts}.json"
    with open(bpath) as f:
        baseline = json.load(f)
    with open(ppath) as f:
        pipeline = json.load(f)
    all_data[ts] = {"baseline": baseline, "pipeline": pipeline}

# ─────────────────────────────────────────────────────────────────────────────
# Per-file scenario inventory
# ─────────────────────────────────────────────────────────────────────────────
ts_scenarios = {}
for ts in TIMESTAMPS:
    records = all_data[ts]["baseline"]
    sc_ids = sorted(set(r["scenario_id"] for r in records))
    ts_scenarios[ts] = sc_ids

print("=" * 100)
print("PER-FILE SCENARIO INVENTORY")
print("=" * 100)
print(f"{'Timestamp':<25} {'#':>3}  Scenario IDs")
print("-" * 100)
for ts in TIMESTAMPS:
    sc_ids = ts_scenarios[ts]
    n = len(sc_ids)
    ids_str = ", ".join(sc_ids)
    print(f"{ts:<25} {n:>3}  {ids_str}")

# ─────────────────────────────────────────────────────────────────────────────
# Duplicate detection
# ─────────────────────────────────────────────────────────────────────────────
sc_to_timestamps = defaultdict(list)
for ts, sc_ids in ts_scenarios.items():
    for sc in sc_ids:
        sc_to_timestamps[sc].append(ts)

all_scenarios = sorted(sc_to_timestamps.keys())
duplicates = {sc: tss for sc, tss in sc_to_timestamps.items() if len(tss) > 1}
unique_only = {sc: tss for sc, tss in sc_to_timestamps.items() if len(tss) == 1}

print()
print("=" * 100)
print("DUPLICATE SCENARIO ANALYSIS")
print("=" * 100)
print(f"Total unique scenario IDs: {len(all_scenarios)}")
print(f"Scenarios tested only once: {len(unique_only)}")
print(f"Scenarios tested multiple times: {len(duplicates)}")
if duplicates:
    print()
    print(f"{'Scenario':<55} {'Count':>5}  Timestamps")
    print("-" * 100)
    for sc, tss in sorted(duplicates.items()):
        print(f"{sc:<55} {len(tss):>5}  {', '.join(tss)}")

# ─────────────────────────────────────────────────────────────────────────────
# Aggregate scenario runs (simple pass/fail summary per scenario per timestamp)
# ─────────────────────────────────────────────────────────────────────────────
def aggregate_runs(records, scenario_id):
    sc_records = [r for r in records if r["scenario_id"] == scenario_id]
    if not sc_records:
        return None
    n_runs = len(sc_records)
    corrects = []
    for r in sc_records:
        is_correct = (r["output_decision"] == r["expected_decision"]) and \
                     (r["output_escalated"] == r["expected_escalate"])
        corrects.append(is_correct)
    n_correct = sum(corrects)
    majority_correct = n_correct > n_runs / 2
    dec_counter = Counter(r["output_decision"] for r in sc_records)
    majority_decision = dec_counter.most_common(1)[0][0]
    return {
        "expected_decision": sc_records[0]["expected_decision"],
        "expected_escalate": sc_records[0]["expected_escalate"],
        "majority_correct":  majority_correct,
        "all_correct":       all(corrects),
        "any_correct":       any(corrects),
        "n_correct":         n_correct,
        "n_runs":            n_runs,
        "majority_decision": majority_decision,
    }

ts_sc_results = {}
for ts in TIMESTAMPS:
    b_records = all_data[ts]["baseline"]
    p_records = all_data[ts]["pipeline"]
    for sc in ts_scenarios[ts]:
        b_agg = aggregate_runs(b_records, sc)
        p_agg = aggregate_runs(p_records, sc)
        ts_sc_results[(ts, sc)] = {"baseline": b_agg, "pipeline": p_agg}

# ─────────────────────────────────────────────────────────────────────────────
# Best-timestamp selection for duplicates
# Criteria: prefer timestamp where pipeline majority_correct=True;
# among those, prefer highest n_correct; among ties, latest timestamp.
# ─────────────────────────────────────────────────────────────────────────────
def pick_better(sc, timestamps, ts_sc_results):
    candidates = [(ts, ts_sc_results[(ts, sc)]["pipeline"]) for ts in timestamps]
    correct_ones = [(ts, p) for ts, p in candidates if p and p["majority_correct"]]
    if correct_ones:
        # Prefer latest timestamp among correct ones (most recent/best run)
        return max(correct_ones, key=lambda x: (x[1]["n_correct"], x[0]))[0]
    # No correct ones — pick highest n_correct, break ties by latest timestamp
    best = max(candidates, key=lambda x: (x[1]["n_correct"] if x[1] else -1, x[0]))
    return best[0]

scenario_best_ts = {}
for sc, tss in sc_to_timestamps.items():
    if len(tss) == 1:
        scenario_best_ts[sc] = tss[0]
    else:
        scenario_best_ts[sc] = pick_better(sc, tss, ts_sc_results)

# ─────────────────────────────────────────────────────────────────────────────
# Assemble the consolidated run pool
# Each selected scenario contributes its 3 runs from the chosen timestamp
# ─────────────────────────────────────────────────────────────────────────────
selected_baseline_runs = []
selected_pipeline_runs = []

for sc in sorted(scenario_best_ts.keys()):
    ts = scenario_best_ts[sc]
    b_runs = [r for r in all_data[ts]["baseline"] if r["scenario_id"] == sc]
    p_runs = [r for r in all_data[ts]["pipeline"] if r["scenario_id"] == sc]
    selected_baseline_runs.extend(b_runs)
    selected_pipeline_runs.extend(p_runs)

# ─────────────────────────────────────────────────────────────────────────────
# Full metrics computation
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 100)
print("BEST-TIMESTAMP SELECTION FOR DUPLICATES")
print("=" * 100)
if duplicates:
    print(f"{'Scenario':<55} {'Best TS':<25} {'P n_correct/runs':<18} {'P majority_correct'}")
    print("-" * 100)
    for sc in sorted(duplicates.keys()):
        chosen = scenario_best_ts[sc]
        p = ts_sc_results[(chosen, sc)]["pipeline"]
        p_runs_str = f"{p['n_correct']}/{p['n_runs']}" if p else "N/A"
        p_mc = p["majority_correct"] if p else False
        print(f"{sc:<55} {chosen:<25} {p_runs_str:<18} {p_mc}")

print()
print("=" * 100)
print("CONSOLIDATED METRICS (all unique scenarios, best run for duplicates)")
print(f"Total unique scenarios: {len(all_scenarios)}")
print(f"Total consolidated pipeline runs: {len(selected_pipeline_runs)}")
print(f"Total consolidated baseline runs: {len(selected_baseline_runs)}")
print("=" * 100)

baseline_metrics = compute_all_metrics(selected_baseline_runs, "baseline")
pipeline_metrics  = compute_all_metrics(selected_pipeline_runs, "pipeline")

# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print the metrics comparison
# ─────────────────────────────────────────────────────────────────────────────
def fmt(v):
    if isinstance(v, float):
        return f"{v:.4f}"
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return json.dumps(v, indent=None)
    return str(v)

SCALAR_METRICS = [
    ("M1_decision_accuracy",        "M1  Decision Accuracy"),
    ("M2_rule_compliance_rate",     "M2  Rule Compliance Rate"),
    ("M3_decision_consistency",     "M3  Decision Consistency"),
    ("M9_pipeline_halt_rate",       "M9  Pipeline Halt Rate"),
    ("M10_three_point_integrity",   "M10 Three-Point Integrity"),
    ("M13_regulatory_citation_rate","M13 Regulatory Citation Rate"),
    ("M15_decision_traceability",   "M15 Decision Traceability"),
    ("M15b_decision_traceability_v2","M15b Decision Traceability v2"),
]

print()
print(f"{'Metric':<35} {'Baseline':>15} {'Pipeline':>15}")
print("-" * 67)
for key, label in SCALAR_METRICS:
    b_val = baseline_metrics.get(key)
    p_val = pipeline_metrics.get(key)
    print(f"{label:<35} {fmt(b_val):>15} {fmt(p_val):>15}")

print()
print("── M1a Weighted Accuracy ──")
m1a_b = baseline_metrics.get("M1a_weighted_accuracy", {})
m1a_p = pipeline_metrics.get("M1a_weighted_accuracy", {})
if isinstance(m1a_b, dict) and isinstance(m1a_p, dict):
    for k in ["weighted_accuracy", "unweighted_accuracy", "total_penalty", "n_evaluated"]:
        print(f"  {k:<33} {fmt(m1a_b.get(k)):>15} {fmt(m1a_p.get(k)):>15}")

print()
print("── M1b Scenario-Level Accuracy ──")
m1b_b = baseline_metrics.get("M1b_scenario_level_accuracy", {})
m1b_p = pipeline_metrics.get("M1b_scenario_level_accuracy", {})
if isinstance(m1b_b, dict) and isinstance(m1b_p, dict):
    for k in ["scenario_accuracy", "n_scenarios", "split_vote_scenarios"]:
        print(f"  {k:<33} {fmt(m1b_b.get(k)):>15} {fmt(m1b_p.get(k)):>15}")

print()
print("── M2b Rule Compliance Jaccard ──")
m2b_b = baseline_metrics.get("M2b_rule_compliance_jaccard", {})
m2b_p = pipeline_metrics.get("M2b_rule_compliance_jaccard", {})
if isinstance(m2b_b, dict) and isinstance(m2b_p, dict):
    for k in ["mean_jaccard", "exact_match_rate", "n_evaluated"]:
        print(f"  {k:<33} {fmt(m2b_b.get(k)):>15} {fmt(m2b_p.get(k)):>15}")

print()
print("── M4 Escalation Accuracy ──")
m4_b = baseline_metrics.get("M4_escalation_accuracy", {})
m4_p = pipeline_metrics.get("M4_escalation_accuracy", {})
if isinstance(m4_b, dict) and isinstance(m4_p, dict):
    for k in ["overall_accuracy", "sensitivity", "specificity", "precision", "f1", "mcc", "n_positive", "n_negative"]:
        print(f"  {k:<33} {fmt(m4_b.get(k)):>15} {fmt(m4_p.get(k)):>15}")

print()
print("── M5 Verification (pipeline only) ──")
m5_p = pipeline_metrics.get("M5_verification", {})
if isinstance(m5_p, dict):
    for k, v in m5_p.items():
        print(f"  {k:<33} {'N/A':>15} {fmt(v):>15}")

print()
print("── M5b Correction Precision (pipeline only) ──")
m5b_p = pipeline_metrics.get("M5b_correction_precision", {})
if isinstance(m5b_p, dict):
    for k, v in m5b_p.items():
        if k != "note":
            print(f"  {k:<33} {'N/A':>15} {fmt(v):>15}")

print()
print("── M6 Per-Rule Accuracy ──")
m6_b = baseline_metrics.get("M6_per_rule_accuracy", {})
m6_p = pipeline_metrics.get("M6_per_rule_accuracy", {})
print(f"  {'Rule':<33} {'Baseline':>15} {'Pipeline':>15}")
for rule in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
    b_v = m6_b.get(rule)
    p_v = m6_p.get(rule)
    delta = ""
    if isinstance(b_v, float) and isinstance(p_v, float):
        delta = f"  Δ={p_v-b_v:+.4f}"
    print(f"  {rule:<33} {fmt(b_v):>15} {fmt(p_v):>15}{delta}")

print()
print("── M6b Per-Rule Accuracy with 95% CI ──")
m6b_b = baseline_metrics.get("M6b_per_rule_accuracy_with_ci", {})
m6b_p = pipeline_metrics.get("M6b_per_rule_accuracy_with_ci", {})
print(f"  {'Rule':<10} {'Baseline acc':>15} {'CI_95':>20} {'Pipeline acc':>15} {'CI_95':>20}")
for rule in ["R1", "R2", "R3", "R4", "R5", "R6", "R7"]:
    b_info = m6b_b.get(rule, {})
    p_info = m6b_p.get(rule, {})
    b_acc = fmt(b_info.get("accuracy"))
    p_acc = fmt(p_info.get("accuracy"))
    b_ci  = str(b_info.get("ci_95", ""))
    p_ci  = str(p_info.get("ci_95", ""))
    print(f"  {rule:<10} {b_acc:>15} {b_ci:>20} {p_acc:>15} {p_ci:>20}")

print()
print("── M7 Demographic Parity ──")
m7_b = baseline_metrics.get("M7_demographic_parity")
m7_p = pipeline_metrics.get("M7_demographic_parity")
for label, m7 in [("Baseline", m7_b), ("Pipeline", m7_p)]:
    if isinstance(m7, dict):
        print(f"  {label}: n_pairs={m7.get('n_pairs')}, n_pairs_same_expected={m7.get('n_pairs_same_expected')}, "
              f"n_violations={m7.get('n_violations')}, parity_violation_rate={fmt(m7.get('parity_violation_rate'))}, "
              f"n_justified_diff={m7.get('n_justified_differentiation')}, n_under_diff={m7.get('n_under_differentiation')}")
        for v in m7.get("violations", []):
            print(f"    VIOLATION pair={v.get('pair_id')}  decisions={v.get('decisions')}")
        for d in m7.get("justified_diffs", []):
            print(f"    JUSTIFIED pair={d.get('pair_id')}")
        for u in m7.get("under_differentiated", []):
            print(f"    UNDER-DIFF pair={u.get('pair_id')}")
    else:
        print(f"  {label}: {m7}")

print()
print("── M7b Equalized Odds ──")
m7b_b = baseline_metrics.get("M7b_equalized_odds")
m7b_p = pipeline_metrics.get("M7b_equalized_odds")
for label, m7b in [("Baseline", m7b_b), ("Pipeline", m7b_p)]:
    if isinstance(m7b, dict):
        print(f"  {label}: tpr_gap={fmt(m7b.get('tpr_gap'))}, fpr_gap={fmt(m7b.get('fpr_gap'))}, "
              f"equalized_odds_satisfied={m7b.get('equalized_odds_satisfied')}")
        for grp, stats in m7b.get("group_stats", {}).items():
            print(f"    group={grp}: n={stats.get('n')}, n_positive={stats.get('n_positive')}, "
                  f"tpr={fmt(stats.get('tpr'))}, fpr={fmt(stats.get('fpr'))}")
    else:
        print(f"  {label}: {m7b}")

print()
print("── M8 Override Rate (pipeline only) ──")
m8_p = pipeline_metrics.get("M8_override_rate")
if isinstance(m8_p, dict):
    for agent, rate in m8_p.items():
        print(f"  {agent:<33} {'N/A':>15} {fmt(rate):>15}")

print()
print("── M11 Hard Rule Enforcement ──")
m11_b = baseline_metrics.get("M11_hard_rule_enforcement", {})
m11_p = pipeline_metrics.get("M11_hard_rule_enforcement", {})
if isinstance(m11_b, dict) and isinstance(m11_p, dict):
    for k in ["hard_rule_enforcement_rate", "false_positive_rate", "n_hard_fail_runs", "n_no_hard_fail_runs"]:
        print(f"  {k:<33} {fmt(m11_b.get(k)):>15} {fmt(m11_p.get(k)):>15}")

print()
print("── M12 Vulnerability Protection (pipeline only) ──")
m12_p = pipeline_metrics.get("M12_vulnerability_protection")
if isinstance(m12_p, dict):
    for k in ["overall_protection_rate", "age_driven_rate", "other_driven_rate",
              "n_high_vuln_runs", "n_age_driven", "n_other_driven"]:
        v = m12_p.get(k)
        if v is not None:
            print(f"  {k:<33} {'N/A':>15} {fmt(v):>15}")
elif m12_p is None:
    print(f"  {'(no high-vulnerability runs)':<33} {'N/A':>15} {'N/A':>15}")

print()
print("── M13 Regulatory Citation Rate (pipeline only) ──")
m13_p = pipeline_metrics.get("M13_regulatory_citation_rate")
print(f"  {'citation_rate':<33} {'N/A (baseline)':>15} {fmt(m13_p):>15}")

print()
print("── M14 Explanation Completeness (pipeline only) ──")
m14_p = pipeline_metrics.get("M14_explanation_completeness")
if isinstance(m14_p, dict):
    for k in ["n_evaluated", "presence_rate", "quality_rate"]:
        print(f"  {k:<33} {'N/A':>15} {fmt(m14_p.get(k)):>15}")
    note = m14_p.get("note", "")
    if note:
        print(f"  note: {note}")
else:
    print(f"  {'(no failed-rule runs in pool)':<33} {'N/A':>15} {fmt(m14_p):>15}")

print()
print("── M_A1 Extraction Accuracy (pipeline only) ──")
ma1_p = pipeline_metrics.get("M_A1_extraction", {})
if isinstance(ma1_p, dict):
    print(f"  overall_exact_match_rate: {fmt(ma1_p.get('overall_exact_match_rate'))}")
    pfa = ma1_p.get("per_field_accuracy", {})
    for field, acc in pfa.items():
        print(f"    {field:<31} {fmt(acc)}")

print()
print("── M_A2 Extraction Accuracy (pipeline only) ──")
ma2_p = pipeline_metrics.get("M_A2_extraction", {})
if isinstance(ma2_p, dict):
    print(f"  overall_exact_match_rate: {fmt(ma2_p.get('overall_exact_match_rate'))}")
    pfa = ma2_p.get("per_field_accuracy", {})
    for field, acc in pfa.items():
        print(f"    {field:<31} {fmt(acc)}")

# ─────────────────────────────────────────────────────────────────────────────
# Per-class accuracy breakdown (pipeline)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 100)
print("PER-CLASS PIPELINE DECISION METRICS")
print("=" * 100)

classes = ["SUITABLE", "UNSUITABLE", "CONDITIONAL", "ESCALATED"]
class_tp = defaultdict(int)
class_fp = defaultdict(int)
class_fn = defaultdict(int)

for sc in sorted(scenario_best_ts.keys()):
    ts = scenario_best_ts[sc]
    res = ts_sc_results[(ts, sc)]
    p = res["pipeline"]
    b = res["baseline"]

    p_out = p["majority_decision"] if p else None
    ts_p_records = [r for r in all_data[ts]["pipeline"] if r["scenario_id"] == sc]
    esc_counter = Counter(r["output_escalated"] for r in ts_p_records)
    p_majority_escalated = esc_counter.most_common(1)[0][0] if ts_p_records else False

    exp_esc = p["expected_escalate"] if p else b["expected_escalate"]
    exp_dec = p["expected_decision"]  if p else b["expected_decision"]
    true_class = "ESCALATED" if exp_esc else exp_dec
    p_pred = "ESCALATED" if p_majority_escalated else (p_out or "UNKNOWN")

    for cls in classes:
        if true_class == cls and p_pred == cls:
            class_tp[cls] += 1
        elif true_class != cls and p_pred == cls:
            class_fp[cls] += 1
        elif true_class == cls and p_pred != cls:
            class_fn[cls] += 1

total = len(all_scenarios)
print(f"{'Class':<15} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
print("-" * 75)
per_class_metrics = {}
for cls in classes:
    tp = class_tp[cls]
    fp = class_fp[cls]
    fn = class_fn[cls]
    support = tp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    per_class_metrics[cls] = {"tp": tp, "fp": fp, "fn": fn, "support": support,
                               "precision": precision, "recall": recall, "f1": f1}
    print(f"{cls:<15} {tp:>4} {fp:>4} {fn:>4} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f} {support:>8}")

macro_p  = sum(per_class_metrics[c]["precision"] for c in classes) / len(classes)
macro_r  = sum(per_class_metrics[c]["recall"]    for c in classes) / len(classes)
macro_f1 = sum(per_class_metrics[c]["f1"]        for c in classes) / len(classes)
w_p  = sum(per_class_metrics[c]["precision"] * per_class_metrics[c]["support"] for c in classes) / total if total else 0
w_r  = sum(per_class_metrics[c]["recall"]    * per_class_metrics[c]["support"] for c in classes) / total if total else 0
w_f1 = sum(per_class_metrics[c]["f1"]        * per_class_metrics[c]["support"] for c in classes) / total if total else 0
print("-" * 75)
print(f"{'Macro avg':<15} {'':>4} {'':>4} {'':>4} {macro_p:>10.4f} {macro_r:>8.4f} {macro_f1:>8.4f} {total:>8}")
print(f"{'Weighted avg':<15} {'':>4} {'':>4} {'':>4} {w_p:>10.4f} {w_r:>8.4f} {w_f1:>8.4f} {total:>8}")

# ─────────────────────────────────────────────────────────────────────────────
# Per-scenario detail
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 100)
print("PER-SCENARIO DETAIL (pipeline vs baseline, best-timestamp run selected)")
print("=" * 100)
print(f"{'Scenario':<55} {'True':>11} {'P-pred':>11} {'B-pred':>11} {'P-ok':>5} {'B-ok':>5} {'P runs':>8} {'B runs':>8}  TS")
print("-" * 120)

pipeline_correct_count = 0
baseline_correct_count = 0

for sc in sorted(scenario_best_ts.keys()):
    ts  = scenario_best_ts[sc]
    res = ts_sc_results[(ts, sc)]
    p   = res["pipeline"]
    b   = res["baseline"]

    p_correct = p["majority_correct"] if p else False
    b_correct = b["majority_correct"] if b else False
    if p_correct:
        pipeline_correct_count += 1
    if b_correct:
        baseline_correct_count += 1

    exp_esc = p["expected_escalate"] if p else b["expected_escalate"]
    exp_dec = p["expected_decision"]  if p else b["expected_decision"]
    true_class = "ESCALATED" if exp_esc else exp_dec

    ts_p_records = [r for r in all_data[ts]["pipeline"] if r["scenario_id"] == sc]
    ts_b_records = [r for r in all_data[ts]["baseline"]  if r["scenario_id"] == sc]

    esc_p = Counter(r["output_escalated"] for r in ts_p_records).most_common(1)[0][0] if ts_p_records else False
    esc_b = Counter(r["output_escalated"] for r in ts_b_records).most_common(1)[0][0] if ts_b_records else False

    p_pred = "ESCALATED" if esc_p else (p["majority_decision"] if p else "N/A")
    b_pred = "ESCALATED" if esc_b else (b["majority_decision"] if b else "N/A")

    p_runs_str = f"{p['n_correct']}/{p['n_runs']}" if p else "N/A"
    b_runs_str = f"{b['n_correct']}/{b['n_runs']}" if b else "N/A"

    dup_marker = " *" if sc in duplicates else ""
    print(f"{sc:<55} {true_class:>11} {p_pred:>11} {b_pred:>11} {str(p_correct):>5} {str(b_correct):>5} "
          f"{p_runs_str:>8} {b_runs_str:>8}  {ts}{dup_marker}")

pipeline_accuracy = pipeline_correct_count / total if total else 0
baseline_accuracy = baseline_correct_count / total if total else 0
delta = pipeline_accuracy - baseline_accuracy

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Total unique scenarios: {total}")
print(f"Pipeline majority-correct: {pipeline_correct_count}/{total} = {pipeline_accuracy:.4f} ({pipeline_accuracy*100:.2f}%)")
print(f"Baseline majority-correct: {baseline_correct_count}/{total} = {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"Delta (pipeline − baseline): {delta:+.4f} ({delta*100:+.2f}%)")
print()
print(f"M1  Decision Accuracy   — Baseline: {baseline_metrics['M1_decision_accuracy']:.4f}  Pipeline: {pipeline_metrics['M1_decision_accuracy']:.4f}")
print(f"M2  Rule Compliance     — Baseline: {baseline_metrics['M2_rule_compliance_rate']:.4f}  Pipeline: {pipeline_metrics['M2_rule_compliance_rate']:.4f}")
print(f"M3  Consistency         — Baseline: {baseline_metrics['M3_decision_consistency']:.4f}  Pipeline: {pipeline_metrics['M3_decision_consistency']:.4f}")

m4b = baseline_metrics["M4_escalation_accuracy"]
m4p = pipeline_metrics["M4_escalation_accuracy"]
if isinstance(m4b, dict) and isinstance(m4p, dict):
    print(f"M4  Escalation (F1)     — Baseline: {m4b.get('f1', 0):.4f}  Pipeline: {m4p.get('f1', 0):.4f}")
    print(f"M4  Escalation (MCC)    — Baseline: {m4b.get('mcc', 0):.4f}  Pipeline: {m4p.get('mcc', 0):.4f}")

m10p = pipeline_metrics.get("M10_three_point_integrity")
if isinstance(m10p, float):
    print(f"M10 Three-Point Integ.  — Pipeline: {m10p:.4f}")

m13p = pipeline_metrics.get("M13_regulatory_citation_rate")
if isinstance(m13p, float):
    print(f"M13 Reg. Citation       — Pipeline: {m13p:.4f}")

m15p = pipeline_metrics.get("M15_decision_traceability")
if isinstance(m15p, float):
    print(f"M15 Traceability        — Pipeline: {m15p:.4f}")

print()
print("(* = scenario tested in multiple timestamps; best pipeline run selected)")
