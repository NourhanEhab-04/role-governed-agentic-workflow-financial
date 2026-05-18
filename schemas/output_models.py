# schemas/output_models.py
"""
Pydantic v2 output models for all six agent outputs.

Each model is the single source of truth for schema validation.
The parse_X() functions in agents/ delegate directly to these models,
replacing all manual key-checking, enum validation, and type coercion.

Coercion policy:
  - LLM outputs (A1, A2, A5, AV): accept "true"/"false" strings for booleans,
    accept quoted numeric strings where the field is numeric.
  - Deterministic outputs (A3, A4): strict bool validation — non-bool types raise.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ── helpers ────────────────────────────────────────────────────────────────────

def _coerce_llm_bool(v: Any) -> bool:
    """Accept Python bool or 'true'/'false' strings (LLM output coercion)."""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.lower()
        if low in ("true", "1"):
            return True
        if low in ("false", "0"):
            return False
        raise ValueError(f"cannot coerce string '{v}' to bool — use true or false")
    if isinstance(v, int):
        return bool(v)
    raise ValueError(f"expected bool, got {type(v).__name__}: {v!r}")


def _strict_bool(field_name: str):
    """Return a field_validator that rejects non-bool types (deterministic data)."""
    def validator(v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError(
                f"conflict_report '{field_name}' must be a bool, got {type(v).__name__}: {v!r}"
            )
        return v
    return validator


# ── A1 — ClientProfileModel ────────────────────────────────────────────────────

class ClientProfileModel(BaseModel):
    financial_knowledge: Literal["none", "basic", "moderate", "advanced"]
    risk_tolerance_score: int = Field(ge=1, le=10)
    investment_horizon: int = Field(ge=1)
    liquid_assets: float
    income: float = Field(ge=0.0)
    investment_amount: float = Field(gt=0.0)
    can_afford_total_loss: bool
    financial_vulnerability: Literal["LOW", "MEDIUM", "HIGH"]
    age: Optional[int] = Field(default=None, ge=0)

    @field_validator("can_afford_total_loss", mode="before")
    @classmethod
    def coerce_afford_bool(cls, v: Any) -> bool:
        return _coerce_llm_bool(v)

    @field_validator("liquid_assets", "income", "investment_amount", mode="before")
    @classmethod
    def coerce_numeric(cls, v: Any) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            cleaned = v.replace(",", "").replace("€", "").replace("$", "").strip()
            try:
                return float(cleaned)
            except ValueError:
                raise ValueError(f"cannot convert '{v}' to a number")
        raise ValueError(f"expected a number, got {type(v).__name__}: {v!r}")

    @field_validator("financial_knowledge", mode="before")
    @classmethod
    def validate_knowledge(cls, v: Any) -> str:
        valid = {"none", "basic", "moderate", "advanced"}
        if isinstance(v, str) and v in valid:
            return v
        raise ValueError(
            f"Invalid financial_knowledge: '{v}'. Must be one of {valid}"
        )

    @field_validator("financial_vulnerability", mode="before")
    @classmethod
    def validate_vulnerability(cls, v: Any) -> str:
        valid = {"LOW", "MEDIUM", "HIGH"}
        if isinstance(v, str) and v in valid:
            return v
        raise ValueError(
            f"Invalid financial_vulnerability: '{v}'. Must be one of {valid}"
        )


# ── A2 — ProductProfileModel ───────────────────────────────────────────────────

class ProductProfileModel(BaseModel):
    product_name: str
    risk_class: int = Field(ge=1, le=7)
    complexity_tier: Literal["NON-COMPLEX", "COMPLEX"]
    requires_knowledge_level: Literal["none", "basic", "moderate", "advanced"]
    minimum_horizon: int = Field(ge=1)
    potential_loss: Literal["partial", "total"]
    leverage: bool

    @field_validator("risk_class", "minimum_horizon", mode="before")
    @classmethod
    def reject_float_for_int(cls, v: Any, info) -> int:
        if isinstance(v, float):
            if v != int(v):
                raise ValueError(
                    f"{info.field_name} must be an integer, got float: {v}"
                )
            return int(v)
        return v

    @field_validator("leverage", mode="before")
    @classmethod
    def coerce_leverage_bool(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            low = v.lower()
            if low in ("true", "false"):
                return low == "true"
            raise ValueError(f"leverage must be a boolean, got string: '{v}'")
        raise ValueError(f"leverage must be a boolean, got: {type(v).__name__}: {v!r}")

    @field_validator("complexity_tier", mode="before")
    @classmethod
    def validate_complexity(cls, v: Any) -> str:
        valid = {"NON-COMPLEX", "COMPLEX"}
        if isinstance(v, str) and v in valid:
            return v
        raise ValueError(f"Invalid complexity_tier: '{v}'. Must be one of {valid}")

    @field_validator("requires_knowledge_level", mode="before")
    @classmethod
    def validate_knowledge_level(cls, v: Any) -> str:
        valid = {"none", "basic", "moderate", "advanced"}
        if isinstance(v, str) and v in valid:
            return v
        raise ValueError(
            f"Invalid requires_knowledge_level: '{v}'. Must be one of {valid}"
        )

    @field_validator("potential_loss", mode="before")
    @classmethod
    def validate_potential_loss(cls, v: Any) -> str:
        valid = {"partial", "total"}
        if isinstance(v, str) and v in valid:
            return v
        raise ValueError(f"Invalid potential_loss: '{v}'. Must be one of {valid}")


# ── A3 — RuleVerdictModel ──────────────────────────────────────────────────────

class RuleVerdictModel(BaseModel):
    score: int
    decision: Literal["SUITABLE", "CONDITIONAL", "UNSUITABLE"]
    rules: Dict[str, Literal["PASS", "FAIL"]]

    @field_validator("score", mode="before")
    @classmethod
    def validate_score_int(cls, v: Any) -> int:
        if not isinstance(v, int) or isinstance(v, bool):
            raise ValueError(f"score must be an integer, got: {type(v).__name__}: {v!r}")
        return v

    @field_validator("decision", mode="before")
    @classmethod
    def validate_decision(cls, v: Any) -> str:
        valid = {"SUITABLE", "CONDITIONAL", "UNSUITABLE"}
        if v not in valid:
            raise ValueError(f"Invalid decision: '{v}'. Must be one of {valid}")
        return v

    @field_validator("rules", mode="before")
    @classmethod
    def validate_rules_is_dict(cls, v: Any) -> dict:
        if not isinstance(v, dict):
            raise ValueError("rules must be a dict")
        return v

    @model_validator(mode="after")
    def validate_all_rule_ids_present(self) -> "RuleVerdictModel":
        from schemas.rule_verdict import VALID_RULE_IDS
        missing = VALID_RULE_IDS - self.rules.keys()
        if missing:
            raise ValueError(f"rules dict missing rule IDs: {missing}")
        for rule_id, result in self.rules.items():
            if result not in ("PASS", "FAIL"):
                raise ValueError(
                    f"Rule {rule_id} has invalid result '{result}'. Must be PASS or FAIL"
                )
        return self


# ── AV — VerificationResultModel ──────────────────────────────────────────────

class FieldCheckModel(BaseModel):
    supported: bool
    reason: str

    @field_validator("supported", mode="before")
    @classmethod
    def coerce_supported(cls, v: Any) -> bool:
        return _coerce_llm_bool(v)


class VerificationResultModel(BaseModel):
    passed: bool
    confidence: float = Field(ge=0.0, le=1.0)
    issues: List[str] = Field(default_factory=list)
    field_checks: Dict[str, FieldCheckModel] = Field(default_factory=dict)

    @field_validator("passed", mode="before")
    @classmethod
    def coerce_passed(cls, v: Any) -> bool:
        return _coerce_llm_bool(v)

    @model_validator(mode="after")
    def enforce_confidence_threshold(self) -> "VerificationResultModel":
        from schemas.verification_result import CONFIDENCE_THRESHOLD
        if self.confidence < CONFIDENCE_THRESHOLD and self.passed is True:
            self.passed = False
            self.issues.append(
                f"passed forced to False: confidence {self.confidence:.2f} "
                f"is below threshold {CONFIDENCE_THRESHOLD}"
            )
        return self


# ── A4 — ConflictReportModel ───────────────────────────────────────────────────

class FlagModel(BaseModel):
    rule_id: str
    triggered: bool
    severity: Literal["LOW", "HIGH"]
    message: str

    @field_validator("triggered", mode="before")
    @classmethod
    def strict_triggered(cls, v: Any) -> bool:
        return _strict_bool("triggered")(v)

    @field_validator("severity", mode="before")
    @classmethod
    def validate_severity(cls, v: Any) -> str:
        valid = {"LOW", "HIGH"}
        if v not in valid:
            raise ValueError(f"severity '{v}' invalid. Must be one of {valid}")
        return v


class ConflictReportModel(BaseModel):
    flags: List[FlagModel]
    escalate: bool
    summary: str

    @field_validator("escalate", mode="before")
    @classmethod
    def strict_escalate(cls, v: Any) -> bool:
        if not isinstance(v, bool):
            raise ValueError(
                f"conflict_report 'escalate' must be a bool, got {type(v).__name__}: {v!r}"
            )
        return v

    @model_validator(mode="after")
    def validate_escalate_consistency(self) -> "ConflictReportModel":
        if self.escalate and len(self.flags) == 0:
            raise ValueError("escalate=True requires at least one flag")
        return self


# ── A5 — SuitabilityReportModel ───────────────────────────────────────────────

class RuleFindingModel(BaseModel):
    rule_id: Literal["R1", "R2", "R3", "R4", "R5", "R6", "R7"]
    status: Literal["PASS", "FAIL"]
    explanation: str

    @field_validator("rule_id", mode="before")
    @classmethod
    def normalise_long_rule_id(cls, v: Any) -> str:
        _LONG_TO_SHORT = {
            "R1_knowledge": "R1", "R2_risk": "R2", "R3_horizon": "R3",
            "R4_afford": "R4", "R5_vuln": "R5", "R6_leverage": "R6",
            "R7_complexity": "R7",
        }
        return _LONG_TO_SHORT.get(v, v)


class FlagAddressedModel(BaseModel):
    rule_id: str
    explanation: str


class SuitabilityReportModel(BaseModel):
    decision: Literal["SUITABLE", "CONDITIONAL", "UNSUITABLE", "ESCALATED"]
    summary: str
    rule_findings: Annotated[List[RuleFindingModel], Field(min_length=7, max_length=7)]
    flags_addressed: List[FlagAddressedModel] = Field(default_factory=list)
    regulatory_basis: str
    client_facing_summary: str

    @field_validator("decision", mode="before")
    @classmethod
    def validate_decision(cls, v: Any) -> str:
        valid = {"SUITABLE", "CONDITIONAL", "UNSUITABLE", "ESCALATED"}
        if v not in valid:
            raise ValueError(f"Invalid decision '{v}'. Must be one of {valid}")
        return v

    @model_validator(mode="after")
    def validate_unique_rule_ids(self) -> "SuitabilityReportModel":
        ids = [f.rule_id for f in self.rule_findings]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate rule_id in rule_findings: {ids}")
        return self

    @model_validator(mode="after")
    def ensure_regulatory_basis(self) -> "SuitabilityReportModel":
        if not self.regulatory_basis or "Article 25" not in self.regulatory_basis:
            passed = [f.rule_id for f in self.rule_findings if f.status == "PASS"]
            failed = [f.rule_id for f in self.rule_findings if f.status == "FAIL"]
            all_rules = ", ".join(passed + failed) or "R1–R7"
            self.regulatory_basis = (
                f"MiFID II Article 25(2) — suitability assessed under {all_rules}. "
                f"Decision: {self.decision}."
            )
        return self
