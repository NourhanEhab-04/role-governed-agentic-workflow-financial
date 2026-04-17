from enum import Enum
from dataclasses import dataclass

REQUIRED_CLIENT_KEYS = {
    "financial_knowledge",
    "risk_tolerance_score",
    "investment_horizon",
    "liquid_assets",
    "income",
    "investment_amount",
    "can_afford_total_loss",
    "financial_vulnerability",
}


class FinancialKnowledge(str, Enum):
    NONE = "none"
    BASIC = "basic"
    MODERATE = "moderate"
    ADVANCED = "advanced"


class FinancialVulnerability(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"

#verifier agent
@dataclass
class ClientProfile:
    age: int
    financial_knowledge: FinancialKnowledge
    risk_tolerance_score: int          # 1–10
    investment_horizon: int            # years
    liquid_assets: float
    income: float
    investment_amount: float
    can_afford_total_loss: bool
    financial_vulnerability: FinancialVulnerability
