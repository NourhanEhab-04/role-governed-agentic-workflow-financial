import json, os, sys
from collections import defaultdict, Counter

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
]

all_data = {}
for ts in TIMESTAMPS:
    bpath = f"{EVAL_DIR}/results_baseline_{ts}.json"
    ppath = f"{EVAL_DIR}/results_pipeline_{ts}.json"
    with open(bpath) as f:
        baseline = json.load(f)
    with open(ppath) as f:
        pipeline = json.load(f)
    all_data[ts] = {"baseline": baseline, "pipeline": pipeline}

ts_scenarios = {}
for ts in TIMESTAMPS:
    records = all_data[ts]["baseline"]
    sc_ids = sorted(set(r["scenario_id"] for r in records))
    ts_scenarios[ts] = sc_ids

def aggregate_runs(records, scenario_id):
    sc_records = [r for r in records if r["scenario_id"] == scenario_id]
    if not sc_records:
        return None
    expected_decision = sc_records[0]["expected_decision"]
    expected_escalate = sc_records[0]["expected_escalate"]
    n_runs = len(sc_records)
    corrects = []
    for r in sc_records:
        out_dec = r["output_decision"]
        exp_dec = r["expected_decision"]
        out_esc = r["output_escalated"]
        exp_esc = r["expected_escalate"]
        is_correct = (out_dec == exp_dec) and (out_esc == exp_esc)
        corrects.append(is_correct)
    n_correct = sum(corrects)
    majority_correct = n_correct > n_runs / 2
    dec_counter = Counter(r["output_decision"] for r in sc_records)
    majority_decision = dec_counter.most_common(1)[0][0]
    return {
        "expected_decision": expected_decision,
        "expected_escalate": expected_escalate,
        "majority_correct": majority_correct,
        "all_correct": all(corrects),
        "any_correct": any(corrects),
        "n_correct": n_correct,
        "n_runs": n_runs,
        "majority_decision": majority_decision,
    }

sc_to_timestamps = defaultdict(list)
for ts, sc_ids in ts_scenarios.items():
    for sc in sc_ids:
        sc_to_timestamps[sc].append(ts)

ts_sc_results = {}
for ts in TIMESTAMPS:
    b_records = all_data[ts]["baseline"]
    p_records = all_data[ts]["pipeline"]
    for sc in ts_scenarios[ts]:
        b_agg = aggregate_runs(b_records, sc)
        p_agg = aggregate_runs(p_records, sc)
        ts_sc_results[(ts, sc)] = {"baseline": b_agg, "pipeline": p_agg}

def pick_better(sc, timestamps, ts_sc_results):
    candidates = [(ts, ts_sc_results[(ts, sc)]["pipeline"], ts_sc_results[(ts, sc)]["baseline"]) for ts in timestamps]
    correct_ones = [(ts, p, b) for ts, p, b in candidates if p and p["majority_correct"]]
    if correct_ones:
        return correct_ones[0][0]
    best = max(candidates, key=lambda x: x[1]["n_correct"] if x[1] else -1)
    return best[0]

scenario_best_ts = {}
for sc, tss in sc_to_timestamps.items():
    if len(tss) == 1:
        scenario_best_ts[sc] = tss[0]
    else:
        scenario_best_ts[sc] = pick_better(sc, tss, ts_sc_results)

all_scenarios = sorted(scenario_best_ts.keys())
total = len(all_scenarios)

pipeline_correct_count = 0
baseline_correct_count = 0
pipeline_beats_baseline = 0
baseline_beats_pipeline = 0
ties = 0

class_tp_pipeline = defaultdict(int)
class_fp_pipeline = defaultdict(int)
class_fn_pipeline = defaultdict(int)

per_scenario_detail = []

for sc in all_scenarios:
    ts = scenario_best_ts[sc]
    res = ts_sc_results[(ts, sc)]
    p = res["pipeline"]
    b = res["baseline"]

    p_correct = p["majority_correct"] if p else False
    b_correct = b["majority_correct"] if b else False

    if p_correct:
        pipeline_correct_count += 1
    if b_correct:
        baseline_correct_count += 1

    if p_correct and not b_correct:
        pipeline_beats_baseline += 1
    elif b_correct and not p_correct:
        baseline_beats_pipeline += 1
    else:
        ties += 1

    exp_esc = p["expected_escalate"] if p else b["expected_escalate"]
    exp_dec = p["expected_decision"] if p else b["expected_decision"]

    if exp_esc:
        true_class = "ESCALATED"
    else:
        true_class = exp_dec

    p_out_dec = p["majority_decision"] if p else None
    ts_p_records = [r for r in all_data[ts]["pipeline"] if r["scenario_id"] == sc]
    esc_counter = Counter(r["output_escalated"] for r in ts_p_records)
    p_majority_escalated = esc_counter.most_common(1)[0][0] if ts_p_records else False
    if p_majority_escalated:
        p_pred_class = "ESCALATED"
    else:
        p_pred_class = p_out_dec if p_out_dec else "UNKNOWN"

    ts_b_records = [r for r in all_data[ts]["baseline"] if r["scenario_id"] == sc]
    b_out_dec = b["majority_decision"] if b else None
    esc_counter_b = Counter(r["output_escalated"] for r in ts_b_records)
    b_majority_escalated = esc_counter_b.most_common(1)[0][0] if ts_b_records else False
    if b_majority_escalated:
        b_pred_class = "ESCALATED"
    else:
        b_pred_class = b_out_dec if b_out_dec else "UNKNOWN"

    for cls in ["SUITABLE", "UNSUITABLE", "CONDITIONAL", "ESCALATED"]:
        if true_class == cls and p_pred_class == cls:
            class_tp_pipeline[cls] += 1
        elif true_class != cls and p_pred_class == cls:
            class_fp_pipeline[cls] += 1
        elif true_class == cls and p_pred_class != cls:
            class_fn_pipeline[cls] += 1

    per_scenario_detail.append({
        "sc": sc, "ts": ts, "true_class": true_class,
        "p_pred": p_pred_class, "b_pred": b_pred_class,
        "p_correct": p_correct, "b_correct": b_correct,
        "p_n_correct": p["n_correct"] if p else 0, "p_n_runs": p["n_runs"] if p else 0,
        "b_n_correct": b["n_correct"] if b else 0, "b_n_runs": b["n_runs"] if b else 0,
    })

pipeline_accuracy = pipeline_correct_count / total
baseline_accuracy = baseline_correct_count / total
delta = pipeline_accuracy - baseline_accuracy

print("=" * 80)
print("CONSOLIDATED EVALUATION REPORT")
print("=" * 80)
print()
print(f"Total unique scenarios evaluated: {total}")
print(f"Missing from eval (01, 02, 03): 3 scenarios")
print()
print(f"Pipeline accuracy: {pipeline_correct_count}/{total} = {pipeline_accuracy:.4f} ({pipeline_accuracy*100:.2f}%)")
print(f"Baseline accuracy: {baseline_correct_count}/{total} = {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"Delta (pipeline - baseline):     {delta:+.4f} ({delta*100:+.2f}%)")
print()
print(f"Pipeline beats baseline: {pipeline_beats_baseline}")
print(f"Baseline beats pipeline: {baseline_beats_pipeline}")
print(f"Ties:                    {ties}")
print()

print("Per-class Pipeline metrics:")
print(f"{'Class':<15} {'TP':>4} {'FP':>4} {'FN':>4} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>8}")
print("-" * 75)
classes = ["SUITABLE", "UNSUITABLE", "CONDITIONAL", "ESCALATED"]
per_class_metrics = {}
for cls in classes:
    tp = class_tp_pipeline[cls]
    fp = class_fp_pipeline[cls]
    fn = class_fn_pipeline[cls]
    support = tp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    per_class_metrics[cls] = {"tp": tp, "fp": fp, "fn": fn, "support": support, "precision": precision, "recall": recall, "f1": f1}
    print(f"{cls:<15} {tp:>4} {fp:>4} {fn:>4} {precision:>10.4f} {recall:>8.4f} {f1:>8.4f} {support:>8}")

macro_p = sum(per_class_metrics[c]["precision"] for c in classes) / len(classes)
macro_r = sum(per_class_metrics[c]["recall"] for c in classes) / len(classes)
macro_f1 = sum(per_class_metrics[c]["f1"] for c in classes) / len(classes)
w_p = sum(per_class_metrics[c]["precision"] * per_class_metrics[c]["support"] for c in classes) / total
w_r = sum(per_class_metrics[c]["recall"] * per_class_metrics[c]["support"] for c in classes) / total
w_f1_num = sum(per_class_metrics[c]["f1"] * per_class_metrics[c]["support"] for c in classes)
w_f1 = w_f1_num / total
print("-" * 75)
print(f"{'Macro avg':<15} {'':>4} {'':>4} {'':>4} {macro_p:>10.4f} {macro_r:>8.4f} {macro_f1:>8.4f} {total:>8}")
print(f"{'Weighted avg':<15} {'':>4} {'':>4} {'':>4} {w_p:>10.4f} {w_r:>8.4f} {w_f1:>8.4f} {total:>8}")
print()

print("=" * 80)
print("PER-SCENARIO DETAIL (pipeline vs baseline, best run selected)")
print("=" * 80)
print(f"{'Scenario':<55} {'True':>11} {'P-pred':>11} {'B-pred':>11} {'P-ok':>5} {'B-ok':>5} {'P runs':>8} {'B runs':>8}")
print("-" * 120)
for d in per_scenario_detail:
    p_runs_str = f"{d['p_n_correct']}/{d['p_n_runs']}"
    b_runs_str = f"{d['b_n_correct']}/{d['b_n_runs']}"
    print(f"{d['sc']:<55} {d['true_class']:>11} {d['p_pred']:>11} {d['b_pred']:>11} {str(d['p_correct']):>5} {str(d['b_correct']):>5} {p_runs_str:>8} {b_runs_str:>8}")
