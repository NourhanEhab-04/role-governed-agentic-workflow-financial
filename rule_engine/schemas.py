# rule_engine/schemas.py
# Re-exports for backward compatibility.
from rule_engine.rule_engine import (
    HARD_FAIL_RULES,
    SOFT_FAIL_RULES,
    PILLAR_1_KNOWLEDGE,
    PILLAR_2_FINANCIAL,
    PILLAR_3_OBJECTIVES,
    SUITABLE_THRESHOLD,
    CONDITIONAL_THRESHOLD,
    KNOWLEDGE_LEVELS,
)

__all__ = [
    "HARD_FAIL_RULES",
    "SOFT_FAIL_RULES",
    "PILLAR_1_KNOWLEDGE",
    "PILLAR_2_FINANCIAL",
    "PILLAR_3_OBJECTIVES",
    "SUITABLE_THRESHOLD",
    "CONDITIONAL_THRESHOLD",
    "KNOWLEDGE_LEVELS",
]
