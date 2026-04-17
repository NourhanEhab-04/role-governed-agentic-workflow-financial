# schemas/suitability_report.py

VALID_DECISIONS = {"SUITABLE", "CONDITIONAL", "UNSUITABLE", "ESCALATED"}

VALID_RULE_IDS = {"R1", "R2", "R3", "R4", "R5", "R6", "R7"}

REQUIRED_RULE_FINDING_KEYS = {"rule_id", "status", "explanation"}

REQUIRED_REPORT_KEYS = {
    "decision",
    "summary",
    "rule_findings",
    "flags_addressed",
    "regulatory_basis",
    "client_facing_summary",
}