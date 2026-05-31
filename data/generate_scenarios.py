#!/usr/bin/env python3
"""
Generate 81 MiFID II suitability scenarios (files 20-100) from the
UCI German Credit Dataset.

UCI source / citation:
  Hofmann, H. (1994). Statlog (German Credit Data) [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C5NC77
  Original: Prof. Hans Hofmann, Universität Hamburg.

The dataset contains 1,000 credit applications with 20 attributes covering
checking-account status, loan duration, credit history, purpose, credit
amount (DM), savings, employment duration, installment rate, personal status,
property type, age, housing, job type, telephone and credit-risk outcome.

MAPPING TO MiFID II CLIENT PROFILE FIELDS
------------------------------------------
Field               UCI source
------------------  -----------------------------------------------------------
income              Employment duration (col 6) × job-tier multiplier (col 16),
                    with credit-amount mod spread for intra-tier variance.
                    Scale: 1994 DM × purchasing-power factor 1.275 -> 2024 EUR.
liquid_assets       Savings category (col 5) with property-ownership bonus
                    (real estate or building-society savings, col 11).
investment_amount   Credit amount (col 4) × DM->EUR factor (1.275).
risk_tolerance_score Derived from checking-account status (col 0) + credit
                    history (col 2) + property (col 11) + age adjustment.
investment_horizon  Loan duration (col 1) rounded up to years, plus a
                    purpose-driven adjustment (education/business -> longer).
financial_knowledge Job tier: A174 (management) + phone + non-foreign -> advanced;
                    A174 otherwise -> moderate; A173 (skilled) -> basic;
                    A172/A171 -> none (or none if unemployed + no phone).
can_afford_total_loss
                    True when: (savings A63/A64 AND employment A74/A75) OR
                    (property A121/A122 AND good credit AND not short-term/unemployed).
financial_vulnerability
                    Composite of employment risk, savings level, age, credit
                    outcome and checking-account overdraft status.
age, portfolio_concentration_pct
                    Age direct from UCI col 12. Concentration from installment
                    rate (col 7) × 15, capped at 60.

PRODUCT RISK-CLASS ANCHORING (ESMA PRIIPs VEV TABLE)
------------------------------------------------------
Products are anchored to the Summary Risk Indicator (SRI) classes defined in
Commission Delegated Regulation (EU) 2017/653 Annex II (reproduced in FCA
Handbook ANNEX II) using the VaR-Equivalent Volatility -> MRM class mapping:
  MRM 1: VEV < 0.5%     -> money market instruments
  MRM 2: VEV 0.5-5%     -> short government bonds
  MRM 3: VEV 5-12%      -> IG corporate bonds, balanced funds, structured notes
  MRM 4: VEV 12-20%     -> diversified global equity ETFs
  MRM 5: VEV 20-30%     -> EM equity ETFs, single-sector equity ETFs
  MRM 6: VEV 30-80%     -> leveraged bond funds, equity derivatives
  MRM 7: VEV > 80%      -> 3× leveraged ETPs

Combined with CRM (issuer credit risk class) via the CRM-MRM lookup table,
each product's risk_class corresponds to its published SRI value.

SRI 5 reclassification note: ESMA analysis (LPA, 2021) shows equity UCITS
previously SRRI 6-7 reclassify to SRI 4-5 under PRIIPs VEV methodology,
confirming em_equity_etf (SRI 5) and sector_equity_etf (SRI 5).

LABELING AUTHORITY
------------------
Every scenario's expected_decision is determined by calling evaluate_suitability()
from rule_engine/rule_engine.py (ESMA35-43-3172 rule engine). The rule engine
is the sole ground truth — no labels are pre-assigned.

ESCALATED scenarios are cases where the rule engine returns SUITABLE but the
orchestrator flags a contradiction: HIGH-vulnerability client receiving a
positive recommendation (pattern identical to existing scenarios 07 and 08).
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path so rule_engine is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from rule_engine.rule_engine import evaluate_suitability

# ── UCI column indices (0-based) ──────────────────────────────────────────────
# Source: german.doc, Prof. Hans Hofmann, Universität Hamburg
COL_CHECKING    = 0   # A11=<0DM, A12=0-200DM, A13=>=200DM, A14=no account
COL_DURATION    = 1   # months
COL_CR_HISTORY  = 2   # A30=no credits, A31=paid at bank, A32=paid duly,
                       # A33=past delays, A34=critical/other
COL_PURPOSE     = 3   # A40=new car, A41=used car, A42=furniture, A43=radio/TV,
                       # A44=appliances, A45=repairs, A46=education,
                       # A47=retraining, A48=business, A49=other
COL_AMOUNT      = 4   # credit amount (DM)
COL_SAVINGS     = 5   # A61=<100DM, A62=100-500DM, A63=500-1000DM,
                       # A64=>=1000DM, A65=unknown/none
COL_EMPLOYMENT  = 6   # A71=unemployed, A72=<1yr, A73=1-4yr,
                       # A74=4-7yr, A75=>=7yr
COL_INSTALL_RT  = 7   # installment rate 1-4 (% of disposable income)
COL_PERS_STATUS = 8   # A91-A95 (sex × marital status)
COL_OTHER_DEBT  = 9   # A101=none, A102=co-applicant, A103=guarantor
COL_RESIDENCE   = 10  # years at present address 1-4
COL_PROPERTY    = 11  # A121=real estate, A122=building savings,
                       # A123=car/other, A124=none
COL_AGE         = 12  # years
COL_INSTALL_PL  = 13  # A141=bank, A142=stores, A143=none
COL_HOUSING     = 14  # A151=rent, A152=own, A153=free
COL_CREDITS     = 15  # existing credits at this bank
COL_JOB         = 16  # A171=unskilled/NR, A172=unskilled/R,
                       # A173=skilled/official, A174=management/highly-qualified
COL_DEPENDENTS  = 17  # number of dependents
COL_TELEPHONE   = 18  # A191=none, A192=registered
COL_FOREIGN     = 19  # A201=yes, A202=no
COL_RISK        = 20  # 1=good credit, 2=bad credit

# ── Financial mappings (UCI DM -> 2024 EUR) ───────────────────────────────────
# 1 DM ≈ 0.51 EUR (fixed conversion 1999) × ~2.5 inflation factor
UCI_DM_TO_EUR = 1.275

# Annual income estimates by employment duration × job tier (EUR 2024)
EMPLOYMENT_INCOME = {
    "A71": (0,       11_000),   # unemployed
    "A72": (12_000,  24_000),   # < 1 year
    "A73": (20_000,  42_000),   # 1–4 years
    "A74": (30_000,  60_000),   # 4–7 years
    "A75": (40_000,  85_000),   # ≥ 7 years
}
JOB_MULTIPLIER = {
    "A171": 0.55,   # unskilled non-resident
    "A172": 0.80,   # unskilled resident
    "A173": 1.00,   # skilled employee / official
    "A174": 1.80,   # management / self-employed / highly qualified
}

# Liquid-asset estimates by savings category (EUR 2024)
# A61 < 100 DM, A62 100-500 DM, A63 500-1000 DM, A64 >= 1000 DM, A65 unknown
SAVINGS_LIQUID = {
    "A61": (500,    3_000),
    "A62": (3_000,  18_000),
    "A63": (18_000, 45_000),
    "A64": (45_000, 200_000),
    "A65": (100,    2_500),
}

# Purpose-driven investment-horizon adjustments (years)
PURPOSE_HORIZON_ADJ = {
    "A46": 5,   # education — long horizon
    "A48": 5,   # business  — long horizon
    "A45": 2,   # repairs
    "A42": 1,   # furniture
    "A47": 1,   # retraining
    "A49": 2,   # other
}

# Investment objectives mapped from loan purpose
PURPOSE_OBJECTIVE = {
    "A40": "asset growth",
    "A41": "asset growth",
    "A42": "capital preservation",
    "A43": "capital preservation",
    "A44": "capital preservation",
    "A45": "capital preservation",
    "A46": "long-term growth",
    "A47": "long-term growth",
    "A48": "long-term growth",
    "A49": "income",
}

# ── Derivation functions ──────────────────────────────────────────────────────

def _derive_income(fields: list) -> float:
    lo, hi = EMPLOYMENT_INCOME.get(fields[COL_EMPLOYMENT], (15_000, 35_000))
    mult = JOB_MULTIPLIER.get(fields[COL_JOB], 1.0)
    base = (lo + hi) / 2 * mult
    # Use credit amount mod 1000 for intra-tier variance (deterministic spread)
    variance_pct = (int(fields[COL_AMOUNT]) % 1_000) / 1_000  # 0.0–0.999
    spread = (hi - lo) * mult * (variance_pct - 0.5) * 0.2
    return round(max(base + spread, 1_000.0), -2)


def _derive_liquid_assets(fields: list) -> float:
    lo, hi = SAVINGS_LIQUID.get(fields[COL_SAVINGS], (1_000, 10_000))
    variance = (int(fields[COL_AMOUNT]) % 500) / 500
    base = lo + (hi - lo) * variance
    # Property ownership increases liquid-asset estimate
    if fields[COL_PROPERTY] == "A121":    # real estate
        base = max(base, 40_000) * 1.4
    elif fields[COL_PROPERTY] == "A122":  # building savings / life insurance
        base = max(base, 15_000) * 1.2
    return round(base, -2)


def _derive_investment_amount(fields: list) -> float:
    return round(max(float(fields[COL_AMOUNT]) * UCI_DM_TO_EUR, 1_000.0), -2)


def _derive_risk_tolerance(fields: list) -> int:
    """
    Composite of checking-account baseline, credit-history modifier,
    property modifier and age modifier.
    """
    checking_base = {"A11": 2, "A12": 3, "A13": 5, "A14": 4}.get(fields[COL_CHECKING], 3)
    history_mod   = {"A30": 0, "A31": 1, "A32": 1, "A33": -1, "A34": -1}.get(fields[COL_CR_HISTORY], 0)
    property_mod  = {"A121": 2, "A122": 1, "A123": 0, "A124": -1}.get(fields[COL_PROPERTY], 0)
    age = int(fields[COL_AGE])
    age_mod = -2 if age >= 65 else (-1 if age >= 55 else (0 if age >= 35 else 1))
    return max(1, min(10, checking_base + history_mod + property_mod + age_mod))


def _derive_horizon(fields: list) -> int:
    duration_months = int(fields[COL_DURATION])
    base_years = max(2, (duration_months + 11) // 12)
    adj = PURPOSE_HORIZON_ADJ.get(fields[COL_PURPOSE], 0)
    return min(20, base_years + adj)


def _derive_vulnerability(fields: list) -> str:
    """
    Weighted composite: employment stability, savings adequacy,
    age risk (elderly/very young), credit outcome, overdraft.
    """
    age = int(fields[COL_AGE])
    score = 0
    if fields[COL_EMPLOYMENT] == "A71":  score += 3   # unemployed
    elif fields[COL_EMPLOYMENT] == "A72": score += 2   # < 1 year
    elif fields[COL_EMPLOYMENT] == "A73": score += 1   # 1–4 years
    if fields[COL_SAVINGS] in ("A61", "A65"): score += 2   # minimal savings
    elif fields[COL_SAVINGS] == "A62":         score += 1   # low savings
    if age >= 75:   score += 2
    elif age >= 65: score += 1
    elif age < 25:  score += 1   # very young, limited financial cushion
    if int(fields[COL_RISK]) == 2:   score += 1   # bad credit outcome
    if fields[COL_CHECKING] == "A11": score += 1  # overdrawn
    return "HIGH" if score >= 5 else ("MEDIUM" if score >= 2 else "LOW")


def _derive_can_afford_total_loss(fields: list) -> bool:
    """
    True if the client has a financial cushion large enough to absorb
    total loss of the investment.  Requires either:
      (a) substantial savings (A63/A64) AND stable long-term employment, OR
      (b) significant property assets AND good credit AND not short/unemployed.
    """
    has_savings  = fields[COL_SAVINGS]   in ("A63", "A64")
    stable_emp   = fields[COL_EMPLOYMENT] in ("A74", "A75")
    has_property = fields[COL_PROPERTY]  in ("A121", "A122")
    good_credit  = int(fields[COL_RISK]) == 1
    not_short    = fields[COL_EMPLOYMENT] not in ("A71", "A72")
    return (has_savings and stable_emp) or (has_property and good_credit and not_short)


def _derive_knowledge(fields: list) -> str:
    """
    Job tier determines base financial knowledge:
      A174 + registered phone + non-foreign worker -> advanced
      A174 otherwise                                -> moderate
      A173 (skilled)                                -> basic
      A171/A172 + unemployed without phone          -> none
      A172 otherwise                                -> none
    """
    job = fields[COL_JOB]
    if job == "A174":
        if fields[COL_TELEPHONE] == "A192" and fields[COL_FOREIGN] == "A202":
            return "advanced"
        return "moderate"
    if job == "A173":
        return "basic"
    # A171 / A172 unskilled
    if fields[COL_EMPLOYMENT] == "A71" and fields[COL_TELEPHONE] == "A191":
        return "none"
    return "none"


def uci_row_to_client(fields: list, name: str, conc_override: int | None = None) -> dict:
    """
    Convert one UCI German Credit row to a MiFID II client-profile dict.

    Parameters
    ----------
    fields : list
        Tokenised UCI row (21 elements).
    name : str
        Client display name for scenario description.
    conc_override : int | None
        If provided, override the computed portfolio_concentration_pct.
        Used for ESCALATED Type-B scenarios where concentration is a
        key trigger.
    """
    can_afford = _derive_can_afford_total_loss(fields)
    conc = conc_override if conc_override is not None else max(5, min(60, int(fields[COL_INSTALL_RT]) * 15))
    return {
        "name":                  name,
        "age":                   int(fields[COL_AGE]),
        "financial_knowledge":   _derive_knowledge(fields),
        "risk_tolerance_score":  _derive_risk_tolerance(fields),
        "investment_horizon":    _derive_horizon(fields),
        "liquid_assets":         _derive_liquid_assets(fields),
        "income":                _derive_income(fields),
        "investment_amount":     _derive_investment_amount(fields),
        "can_afford_total_loss": can_afford,
        "financial_vulnerability": _derive_vulnerability(fields),
        "portfolio_concentration_pct": conc,
        "investment_objective":  PURPOSE_OBJECTIVE.get(fields[COL_PURPOSE], "balanced growth"),
    }


# ── Load UCI data ─────────────────────────────────────────────────────────────

def load_uci(path: Path) -> list[list[str]]:
    rows = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(stripped.split())
    assert len(rows) == 1_000, f"Expected 1000 rows, got {len(rows)}"
    return rows


# ── Scenario plan ─────────────────────────────────────────────────────────────
# Each entry: (file_number, uci_row_index, product_file, client_name, conc_override)
# UCI row index: 0-based into the 1000-row german_credit_raw.txt
#
# Row selection rationale:
#   SUITABLE (20-49)  — rows that naturally fit the target product after
#                       profile mapping; covers all decision trees.
#   CONDITIONAL (50-69) — rows where exactly 2 soft-fail rules trigger
#                         (R2+R3 or similar), yielding score 40-69.
#   UNSUITABLE (70-94) — rows that trigger at least one hard-fail rule:
#                         R1 (knowledge), R4 (affordability), R5 (vulnerability),
#                         R7 (complexity).
#   ESCALATED (95-100) — HIGH-vulnerability rows whose rule-engine verdict is
#                         SUITABLE (RC < 5, R5 passes) but the orchestrator
#                         flags a contradiction per ESMA GL §86-88.
#
# Products are chosen to stress specific rule combinations for thesis coverage.

SCENARIO_PLAN: list[tuple[int, int, str, str, int | None]] = [
    # ── SUITABLE (files 20-49) ────────────────────────────────────────────────
    # Rows with none/basic knowledge + money_market_fund (RC 1, NON-COMPLEX,
    # requires_knowledge_level=none). R5 passes even for HIGH-vuln (RC 1 < 5).
    (20, 2,   "money_market_fund.json",     "Marta Osei",        None),
    (21, 42,  "money_market_fund.json",     "Soren Lindqvist",   None),
    (22, 48,  "money_market_fund.json",     "Elena Bondarenko",  None),
    # HIGH vulnerability clients + money_market_fund — R5 passes (RC 1 < 5)
    (23, 1,   "money_market_fund.json",     "Kwame Asante",      None),
    (24, 9,   "money_market_fund.json",     "Dijana Ristić",     None),
    (25, 22,  "money_market_fund.json",     "Bogumił Nowak",     None),

    # Basic-knowledge + government_bond (RC 2, min 3yr, basic).
    # Some rows have horizon < 3 -> R3 soft fail, but score 85 ≥ 70 -> SUITABLE.
    (26, 6,   "government_bond.json",       "Ingeborg Haas",     None),
    (27, 36,  "government_bond.json",       "Crispin Okafor",    None),
    (28, 63,  "government_bond.json",       "Rasa Vaitkienė",    None),

    # Basic-knowledge + ig_corporate_bond_etf (RC 3, NON-COMPLEX, min 3yr, basic).
    # VEV ~7-10%, MRM 3 + CRM 2 -> SRI 3 (EU 2017/653 Annex II).
    (29, 3,   "ig_corporate_bond_etf.json", "Annika Bergström",  None),
    (30, 65,  "ig_corporate_bond_etf.json", "Ferhat Yıldız",     None),
    (31, 68,  "ig_corporate_bond_etf.json", "Claudia Moreira",   None),

    # Moderate-knowledge + eu_balanced_fund (RC 3, COMPLEX, min 4yr, moderate).
    # R3 fails when horizon < 4 (score 85) or passes (score 100) — both SUITABLE.
    (32, 7,   "eu_balanced_fund.json",      "Lars Ericsson",     None),
    (33, 85,  "eu_balanced_fund.json",      "Simone Nakamura",   None),
    (34, 99,  "eu_balanced_fund.json",      "Petra Horváth",     None),

    # Moderate-knowledge + capital_protected_note (RC 3, COMPLEX, min 5yr, moderate).
    # VEV ~4-8%, MRM 3 + CRM 3 -> SRI 3 (EU 2017/653 Annex II lookup CR3/MR3=3).
    (35, 40,  "capital_protected_note.json","Tomáš Dvořák",      None),
    (36, 343, "capital_protected_note.json","Mirjana Babić",     None),

    # Moderate-knowledge + em_equity_etf (RC 5, NON-COMPLEX, min 7yr, moderate).
    # VEV 22-28%, MRM 5 + CRM 1 -> SRI 5. R3 often fails (horizon < 7) giving
    # score 85 -> still SUITABLE (≥70). R2 passes when RT ≥ 3.
    (37, 7,   "em_equity_etf.json",         "Noa Bar-Cohen",     None),
    (38, 85,  "em_equity_etf.json",         "Yuki Hashimoto",    None),
    (39, 34,  "em_equity_etf.json",         "Chidera Eze",       None),

    # Moderate-knowledge + global_equity_etf (RC 4, NON-COMPLEX, min 5yr, basic).
    # VEV 12-20%, MRM 4 + CRM 1 -> SRI 4.
    (40, 74,  "global_equity_etf.json",     "Aoife Murphy",      None),
    (41, 116, "global_equity_etf.json",     "Santiago Ríos",     None),
    (42, 141, "global_equity_etf.json",     "Brigitte Fournier", None),

    # Moderate-knowledge + sector_equity_etf (RC 5, NON-COMPLEX, min 5yr, basic).
    # VEV 22-30%, MRM 5 + CRM 1 -> SRI 5 (single-sector concentration risk).
    # LOW vulnerability required (R5 passes only if not HIGH, RC 5 ≥ 5).
    (43, 40,  "sector_equity_etf.json",     "Pilar Delgado",     None),  # LOW
    (44, 145, "sector_equity_etf.json",     "Oladapo Adeyemi",   None),  # LOW

    # Basic-knowledge + sector_equity_etf — SUITABLE when RT ≥ 3 + horizon ≥ 5
    (45, 3,   "sector_equity_etf.json",     "Karin Johansen",    None),  # RT4, h5
    (46, 65,  "sector_equity_etf.json",     "Václav Novotný",    None),  # RT6, h5
    (47, 87,  "sector_equity_etf.json",     "Aminata Diallo",    None),  # RT3, h8

    # Moderate-knowledge + structured_note (RC 3, COMPLEX, requires_knowledge=none).
    # R7 passes for moderate (≥ moderate). Score 100 when all rules pass.
    (48, 45,  "structured_note.json",       "Kristoffer Lund",   None),
    (49, 51,  "structured_note.json",       "Hazel Okonkwo",     None),

    # ── CONDITIONAL (files 50-69) ─────────────────────────────────────────────
    # Two soft-fail rules -> score 65 (CONDITIONAL threshold: 40 ≤ score < 70).
    # Primary pattern: R2 fail (RC > RT+2) + R3 fail (horizon < min_horizon).

    # Moderate + em_equity_etf + RT ≤ 2: R2 fail(-20) + R3 fail(-15) = 65
    (50, 62,  "em_equity_etf.json",         "Riitta Mäkinen",    None),  # RT1,h5,age61
    (51, 72,  "em_equity_etf.json",         "Borivoje Stanić",   None),  # RT1,h2,age51
    (52, 75,  "em_equity_etf.json",         "Dolorès Lebrun",    None),  # RT1,h2,age66
    (53, 105, "em_equity_etf.json",         "Zdeněk Pospíšil",   None),  # RT2,h2
    (54, 119, "em_equity_etf.json",         "Ewa Wiśniewska",    None),  # RT2,h2,LOW
    (55, 154, "em_equity_etf.json",         "Hamsa Abdullahi",   None),  # RT2,h4,LOW
    (56, 199, "em_equity_etf.json",         "Valentijn de Boer", None),  # RT1,h3
    (57, 227, "em_equity_etf.json",         "Raija Korhonen",    None),  # RT2,h3

    # Basic + global_equity_etf + RT 1 + horizon < 5: R2(-20)+R3(-15)=65
    (58, 0,   "global_equity_etf.json",     "Werner Strohbach",  None),  # age67
    (59, 54,  "global_equity_etf.json",     "Ljiljana Marić",    None),  # age57
    (60, 146, "global_equity_etf.json",     "Thabo Ndlovu",      None),  # age39
    (61, 177, "global_equity_etf.json",     "Sigrid Haugen",     None),  # age52,LOW
    (62, 503, "global_equity_etf.json",     "Asel Duisheeva",    None),  # age38
    (63, 659, "global_equity_etf.json",     "Baptiste Renard",   None),  # age41

    # Basic + sector_equity_etf + RT 1 + short horizon: R2(-20)+R3(-15)=65
    (64, 0,   "sector_equity_etf.json",     "Lene Andersen",     None),  # age67
    (65, 54,  "sector_equity_etf.json",     "Maciej Kowalski",   None),  # age57
    (66, 479, "sector_equity_etf.json",     "Saliha Bouzid",     None),  # age44

    # Moderate + em_equity_etf demographic pairs showing age and RT diversity
    (67, 373, "em_equity_etf.json",         "Olena Kovalenko",   None),  # RT1,h5,age63
    (68, 287, "em_equity_etf.json",         "Mikael Johansson",  None),  # RT2,h4
    (69, 205, "em_equity_etf.json",         "Fatoumata Coulibaly",None),  # RT1,h3

    # ── UNSUITABLE (files 70-94) ──────────────────────────────────────────────

    # R1 hard fail: knowledge below product minimum
    # none/basic knowledge + sector_equity_etf (requires basic): none < basic
    (70, 2,   "sector_equity_etf.json",     "Gunnar Erikson",    None),  # none
    (71, 5,   "em_equity_etf.json",         "Rosaria Ferraro",   None),  # none<moderate
    (72, 26,  "em_equity_etf.json",         "Paweł Zając",       None),  # none<moderate

    # R1+R7 hard fail: none/basic + equity_derivative (COMPLEX, requires advanced)
    (73, 13,  "equity_derivative.json",     "Tunde Fashola",     None),  # none
    (74, 15,  "equity_derivative.json",     "Ingrid Svensson",   None),  # none
    (75, 33,  "equity_derivative.json",     "Marko Jovanović",   None),  # none
    (76, 44,  "equity_derivative.json",     "Cecília Szabó",     None),  # none
    (77, 6,   "equity_derivative.json",     "Dominik Wagner",    None),  # basic

    # R4 hard fail: cannot afford total loss + equity_derivative (potential_loss=total)
    # UCI rows 268 and 915 are the only ADVANCED-knowledge rows; both have
    # can_afford_total_loss=False -> R4 is the sole hard fail (R1/R7 pass).
    (78, 268, "equity_derivative.json",     "Michael Brandt",    None),  # adv, R4
    (79, 915, "equity_derivative.json",     "Valeria Conti",     None),  # adv, R4

    # R5 hard fail: HIGH vulnerability + product RC ≥ 5
    (80, 1,   "sector_equity_etf.json",     "Kwabena Mensah",    None),  # HIGH+RC5
    (81, 4,   "sector_equity_etf.json",     "Anežka Horáčková",  None),  # HIGH+RC5
    (82, 10,  "sector_equity_etf.json",     "Abiola Okonkwo",    None),  # HIGH+RC5
    (83, 79,  "sector_equity_etf.json",     "Jānis Bērziņš",     None),  # HIGH+RC5
    (84, 80,  "em_equity_etf.json",         "Zeynep Arslan",     None),  # HIGH,R1+R5
    (85, 47,  "em_equity_etf.json",         "Dariusz Kowalski",  None),  # HIGH,age23,R1+R5

    # R7 hard fail: basic/none knowledge + structured_note (COMPLEX)
    # structured_note.requires_knowledge_level="none" -> R1 passes, but
    # R7: COMPLEX + knowledge < moderate -> hard fail.
    (86, 0,   "structured_note.json",       "Helga Zimmermann",  None),  # basic
    (87, 6,   "structured_note.json",       "Przemysław Witek",  None),  # basic
    (88, 14,  "structured_note.json",       "Ioanna Papadopoulou",None), # basic

    # Multiple hard fails: none knowledge + leveraged_etf_3x (RC 7, COMPLEX,
    # total_loss, leverage, requires advanced)
    # R1+R7 always fail for none/basic; R4 if can't afford; R5 if HIGH vuln
    (89, 42,  "leveraged_etf_3x.json",      "Håkon Aasberg",     None),  # none,R1+R7
    (90, 48,  "leveraged_etf_3x.json",      "Nadège Martin",     None),  # none,R1+R7
    (91, 26,  "leveraged_etf_3x.json",      "Grzegorz Lewandowski",None), # none,R1+R7
    (92, 3,   "leveraged_etf_3x.json",      "Aino Mäkelä",       None),  # basic,R1+R7
    (93, 4,   "leveraged_etf_3x.json",      "Pita Havili",       None),  # basic,HIGH,R1+R4+R5+R7

    # Demographic pair: same UCI row (age 39, bad credit) with two products
    # to show how product choice changes UNSUITABLE -> UNSUITABLE for different rules
    (94, 15,  "leveraged_etf_3x.json",      "Lieselotte Braun",  None),  # none,R1+R7

    # ── ESCALATED (files 95-100) ──────────────────────────────────────────────
    # HIGH-vulnerability clients matched to a LOW-risk-class product (RC 3).
    # Rule engine verdict = SUITABLE (R5 passes because RC < 5, all rules pass
    # or only soft-fail). Orchestrator flags ESCALATED because recommending
    # any investment product to a HIGH-vulnerability client — even a safe one —
    # requires human compliance review (ESMA GL §86-88 heightened protection).
    #
    # Pattern identical to scenario 07 (Elena Rossi, HIGH vuln + SUITABLE from
    # global_equity_etf RC4 -> ESCALATED).
    #
    # portfolio_concentration_pct overridden to 75 for scenario 100 to test
    # the concentration-driven ESCALATED path (like scenario 08, Bogdan).
    (95, 1,   "ig_corporate_bond_etf.json", "Kofi Adu-Mensah",   None),  # HIGH,age22,SUITABLE
    (96, 9,   "ig_corporate_bond_etf.json", "Amara Diallo",      None),  # HIGH,age28,SUITABLE
    (97, 56,  "ig_corporate_bond_etf.json", "Renata Wójcik",     None),  # HIGH,age52,SUITABLE
    (98, 11,  "ig_corporate_bond_etf.json", "Josip Kovačević",   None),  # HIGH,age24,SUITABLE
    (99, 95,  "ig_corporate_bond_etf.json", "Leena Virtanen",    None),  # HIGH,age58,SUITABLE
    (100, 76, "ig_corporate_bond_etf.json", "Amara Koné",         75),   # HIGH,conc75,ESCALATED
]

# Rows whose expected_decision should be ESCALATED (overrides rule-engine output)
ESCALATED_FILE_NUMBERS = {95, 96, 97, 98, 99, 100}


# ── Scenario descriptions ─────────────────────────────────────────────────────

def _describe_scenario(file_num: int, uci_idx: int, product_file: str,
                        client: dict, verdict: dict) -> str:
    decision = verdict["decision"]
    hard = verdict["hard_failed_rules"]
    score = verdict["score"]
    knowledge = client["financial_knowledge"]
    rt = client["risk_tolerance_score"]
    horizon = client["investment_horizon"]
    vuln = client["financial_vulnerability"]
    age = client["age"]

    product_name = product_file.replace(".json", "").replace("_", " ")
    if file_num in ESCALATED_FILE_NUMBERS:
        return (
            f"UCI row {uci_idx} (age {age}, {vuln} vulnerability, {knowledge} knowledge, "
            f"RT {rt}) matched to {product_name} — rule engine returns SUITABLE "
            f"(score {score}, all R5-relevant rules pass as RC < 5) but orchestrator "
            f"flags ESCALATED: HIGH-vulnerability client receiving positive recommendation "
            f"requires human compliance review per ESMA GL §86-88"
        )
    if not hard and decision == "SUITABLE":
        soft_failed = [r["rule"] for r in verdict["rules"] if not r["pass"] and not r["hard_fail"]]
        return (
            f"UCI row {uci_idx} (age {age}, {vuln} vulnerability, {knowledge} knowledge, "
            f"RT {rt}, horizon {horizon}yr) — {product_name}: all hard rules pass"
            + (f", soft fails {soft_failed} (score {score})" if soft_failed else f" (score {score})")
            + " -> SUITABLE"
        )
    if not hard and decision == "CONDITIONAL":
        soft_failed = [r["rule"] for r in verdict["rules"] if not r["pass"] and not r["hard_fail"]]
        return (
            f"UCI row {uci_idx} (age {age}, {vuln} vulnerability, {knowledge} knowledge, "
            f"RT {rt}, horizon {horizon}yr) — {product_name}: soft fails {soft_failed} "
            f"(score {score}) -> CONDITIONAL"
        )
    # UNSUITABLE
    return (
        f"UCI row {uci_idx} (age {age}, {vuln} vulnerability, {knowledge} knowledge, "
        f"RT {rt}) — {product_name}: hard fails {hard} -> UNSUITABLE"
    )


def _slug(file_num: int, decision: str, uci_idx: int) -> str:
    decision_tag = decision.lower()
    return f"{file_num:03d}_uci{uci_idx}_{decision_tag}"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data_dir      = PROJECT_ROOT / "data"
    uci_path      = data_dir / "german_credit_raw.txt"
    scenarios_dir = data_dir / "scenarios"
    products_dir  = data_dir / "products"

    print(f"Loading UCI data from {uci_path}")
    uci_rows = load_uci(uci_path)

    # Load all products into memory
    products: dict[str, dict] = {}
    for pf in products_dir.glob("*.json"):
        products[pf.name] = json.loads(pf.read_text())

    generated = []
    skipped   = []

    for (file_num, uci_idx, product_file, client_name, conc_override) in SCENARIO_PLAN:
        out_file = scenarios_dir / f"{file_num:02d}_uci{uci_idx}_{product_file.replace('.json','')}.json"

        if product_file not in products:
            print(f"  [SKIP] {file_num}: product '{product_file}' not found")
            skipped.append(file_num)
            continue

        fields = uci_rows[uci_idx]
        client = uci_row_to_client(fields, client_name, conc_override)
        product = products[product_file]

        # ── Ground-truth label from rule engine (ESMA35-43-3172) ──────────────
        verdict = evaluate_suitability(client, product)

        # ── ESCALATED override ────────────────────────────────────────────────
        # The rule engine never returns ESCALATED; that decision is made by the
        # orchestrator when a HIGH-vulnerability client receives a positive
        # recommendation.  We preserve the rule-engine's hard_failed_rules so
        # the fixture integrity test can verify the engine output independently.
        if file_num in ESCALATED_FILE_NUMBERS:
            expected_decision = "ESCALATED"
            expected_escalate = True
        else:
            expected_decision = verdict["decision"]
            expected_escalate = False

        expected_rules_failed = sorted(
            set(verdict["hard_failed_rules"])
            | {r["rule"] for r in verdict["rules"] if not r["pass"] and not r["hard_fail"]}
        )

        scenario = {
            "description":          _describe_scenario(file_num, uci_idx, product_file, client, verdict),
            "uci_row_index":        uci_idx,
            "uci_source":           "https://doi.org/10.24432/C5NC77",
            "rule_engine_score":    verdict["score"],
            "rule_engine_decision": verdict["decision"],
            "client":               client,
            "product_file":         product_file,
            "expected_decision":    expected_decision,
            "expected_escalate":    expected_escalate,
            "expected_rules_failed": expected_rules_failed,
        }

        out_file.write_text(json.dumps(scenario, indent=2, ensure_ascii=False), encoding="utf-8")
        generated.append(file_num)
        print(f"  [{verdict['decision']:11s}->{expected_decision:11s}] {out_file.name}  "
              f"(k={client['financial_knowledge']}, rt={client['risk_tolerance_score']}, "
              f"h={client['investment_horizon']}, v={client['financial_vulnerability']}, "
              f"score={verdict['score']}, hard={verdict['hard_failed_rules']})")

    print(f"\nGenerated {len(generated)} scenarios, skipped {len(skipped)}")
    if skipped:
        print(f"Skipped file numbers: {skipped}")

    # Verify total scenario count
    all_scenarios = list(scenarios_dir.glob("*.json"))
    print(f"Total scenario files: {len(all_scenarios)}")


if __name__ == "__main__":
    main()
