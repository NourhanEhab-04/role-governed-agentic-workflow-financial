from enum import Enum
from dataclasses import dataclass

from schemas.client_profile import FinancialKnowledge

REQUIRED_PRODUCT_KEYS = {
    "product_name",
    "risk_class",
    "complexity_tier",
    "requires_knowledge_level",
    "minimum_horizon",
    "potential_loss",
    "leverage",
}


VALID_COMPLEXITY_TIERS = {"NON-COMPLEX", "COMPLEX"}
VALID_KNOWLEDGE_LEVELS = {"none", "basic", "moderate", "advanced"}
VALID_POTENTIAL_LOSS = {"partial", "total"}


class ComplexityTier(str, Enum):
    NON_COMPLEX = "NON-COMPLEX"
    COMPLEX = "COMPLEX"


class PotentialLoss(str, Enum):
    PARTIAL = "partial"
    TOTAL = "total"


@dataclass
class ProductProfile:
    product_name: str
    risk_class: int                        # 1–7
    complexity_tier: ComplexityTier
    requires_knowledge_level: FinancialKnowledge
    minimum_horizon: int                   # years
    potential_loss: PotentialLoss
    leverage: bool
