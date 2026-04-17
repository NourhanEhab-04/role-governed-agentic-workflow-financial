from enum import Enum
from dataclasses import dataclass, field
from typing import List

REQUIRED_VERDICT_KEYS = {"score", "decision", "rules"}

VALID_DECISIONS = {"SUITABLE", "UNSUITABLE", "CONDITIONAL"}

VALID_RULE_IDS = {
    "R1_knowledge",
    "R2_risk",
    "R3_horizon",
    "R4_afford",
    "R5_vuln",
    "R6_leverage",
    "R7_complexity",
}


class Decision(str, Enum):
    SUITABLE = "SUITABLE"
    CONDITIONAL = "CONDITIONAL"
    UNSUITABLE = "UNSUITABLE"


@dataclass
class RuleResult:
    rule: str       # "R1" .. "R7"
    pass_: bool     # True = PASS, False = FAIL  (pass_ avoids the keyword clash)
    penalty: int    # 0 or negative int
    detail: str     # human-readable explanation


@dataclass
class RuleVerdict:
    score: int
    decision: Decision
    rules: List[RuleResult] = field(default_factory=list)
