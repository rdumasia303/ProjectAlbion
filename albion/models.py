"""
Core data models for ALBION Engine

All models use Pydantic for validation against JSON schemas.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Policy Levers
# ============================================================================

class IncomeTaxLevers(BaseModel):
    """Income tax rate/band adjustments"""
    basic_rate_pp: float = Field(0.0, ge=-10, le=10, description="Percentage points")
    higher_rate_pp: float = Field(0.0, ge=-10, le=10)
    additional_rate_pp: float = Field(0.0, ge=-10, le=10)
    personal_allowance_delta_gbp: float = Field(0.0)


class NICSLevers(BaseModel):
    """National Insurance Contributions adjustments"""
    class1_main_pp: float = Field(0.0, ge=-5, le=5)
    class1_additional_pp: float = Field(0.0, ge=-5, le=5)
    class4_main_pp: float = Field(0.0, ge=-5, le=5)
    threshold_shift_gbp: float = Field(0.0)


class VATLevers(BaseModel):
    """Value Added Tax adjustments"""
    standard_rate_pp: float = Field(0.0, ge=-10, le=10)
    reduced_rate_pp: float = Field(0.0, ge=-10, le=10)
    exemption_changes: List[Dict[str, str]] = Field(default_factory=list)


class CorpTaxLevers(BaseModel):
    """Corporation tax adjustments"""
    main_rate_pp: float = Field(0.0, ge=-10, le=10)
    small_profits_rate_pp: float = Field(0.0, ge=-10, le=10)


class CGTLevers(BaseModel):
    """Capital Gains Tax adjustments"""
    align_to_income_tax: bool = Field(False)
    higher_rate_pp: float = Field(0.0, ge=-20, le=20)
    annual_exempt_amount_delta_gbp: float = Field(0.0)


class CarbonPriceLevers(BaseModel):
    """Emissions Trading Scheme / Carbon Price"""
    start_gbp_per_tco2e: float = Field(55.0, ge=0)
    ramp_ppy: float = Field(10.0, description="£ increase per year")


class DepartmentalLever(BaseModel):
    """Single department spending adjustment"""
    dept: str = Field(..., pattern=r"^[A-Z]{2,10}$")
    rdel_pct: Optional[float] = Field(None, ge=-50, le=50)
    cdel_pct: Optional[float] = Field(None, ge=-50, le=50)


class BenefitsLevers(BaseModel):
    """Benefit uplift adjustments"""
    uc_uplift_pct: float = Field(0.0, ge=-20, le=50)
    state_pension_uplift_pct: float = Field(0.0, ge=-10, le=20)
    pension_credit_uplift_pct: float = Field(0.0, ge=-10, le=20)


class PolicyLevers(BaseModel):
    """Complete set of policy instruments"""
    income_tax: IncomeTaxLevers = Field(default_factory=IncomeTaxLevers)
    nics: NICSLevers = Field(default_factory=NICSLevers)
    vat: VATLevers = Field(default_factory=VATLevers)
    corp_tax: CorpTaxLevers = Field(default_factory=CorpTaxLevers)
    cgt: CGTLevers = Field(default_factory=CGTLevers)
    ets_carbon_price: CarbonPriceLevers = Field(default_factory=CarbonPriceLevers)
    departmental: List[DepartmentalLever] = Field(default_factory=list)
    benefits: BenefitsLevers = Field(default_factory=BenefitsLevers)


# ============================================================================
# Policy Targets
# ============================================================================

class PolicyTargets(BaseModel):
    """What the policy aims to achieve"""
    revenue_delta_gbp_bny: Optional[float] = None
    deficit_delta_gbp_bny: Optional[float] = None
    emissions_delta_mtco2e: Optional[float] = None


# ============================================================================
# Impact Results
# ============================================================================

class DistributionalImpact(BaseModel):
    """Distributional analysis across income deciles"""
    decile_deltas_pct: List[float] = Field(..., min_length=10, max_length=10)
    gini_delta: Optional[float] = None
    poverty_rate_delta_pp: Optional[float] = None


class RegionalImpact(BaseModel):
    """Impact on a single region"""
    gdp_delta_pct: float
    employment_delta_thousands: float


class ClimateImpact(BaseModel):
    """Climate and emissions impact"""
    emissions_delta_mtco2e: float
    trajectory_2024_2037: List[float] = Field(description="Annual emissions MtCO2e")
    budget_compliance: Dict[str, Any]


class MacroImpact(BaseModel):
    """Macroeconomic impact"""
    gdp_delta_pct: float
    employment_delta_thousands: float
    debt_ratio_year5_pct: float


class ImpactResult(BaseModel):
    """Complete impact assessment from all agents"""
    distribution: DistributionalImpact
    regional: Dict[str, RegionalImpact]  # ITL1 code -> impact
    climate: ClimateImpact
    macro: MacroImpact
    protected_groups: Dict[str, float]


# ============================================================================
# Policy Proposal
# ============================================================================

class Policy(BaseModel):
    """
    Complete policy proposal

    Conforms to proposal.v1.schema.json
    """
    schema: str = Field("proposal/v1", const=True)
    jurisdiction: str = Field("UK")
    horizon_years: int = Field(5, ge=1, le=10)
    levers: PolicyLevers
    targets: PolicyTargets
    impacts: Optional[ImpactResult] = None
    meta: Dict[str, Any] = Field(default_factory=dict)

    # Runtime attributes (not in schema)
    id: UUID = Field(default_factory=uuid4)
    objective_value: Optional[float] = None
    gate_results: Dict[str, Any] = Field(default_factory=dict)
    signature: Optional[Dict[str, Any]] = None

    def __hash__(self) -> int:
        """Make Policy hashable for set operations"""
        return hash(self.id)


# ============================================================================
# Constitution
# ============================================================================

class FiscalRules(BaseModel):
    """Fiscal policy constraints"""
    source: str
    debt_rule: str
    borrowing_rule: str
    debt_threshold_pct_gdp: Optional[float] = None
    borrowing_envelope_bn_gbp: Optional[float] = None
    headroom_required_bn_gbp: float = Field(5.0)


class CarbonBudget(BaseModel):
    """Single carbon budget period"""
    total_mtco2e: float
    years: str  # e.g., "2033-2037"


class DevolutionRules(BaseModel):
    """Devolved competences"""
    scotland: Dict[str, str] = Field(default_factory=dict)
    wales: Dict[str, str] = Field(default_factory=dict)
    northern_ireland: Dict[str, str] = Field(default_factory=dict)
    ukwide: List[str] = Field(default_factory=list)
    barnett_formula: Optional[Dict[str, Any]] = None


class EqualityRules(BaseModel):
    """Equality Act / PSED requirements"""
    psed: Dict[str, Any]
    require_eqia_for: List[str] = Field(default_factory=list)


class DiversityParams(BaseModel):
    """DNOS selector configuration"""
    k: int = Field(5, ge=3, le=10)
    epsilon_additive: float = Field(0.02, ge=0.0, le=0.1)
    weights: Dict[str, float]
    quotas: Dict[str, int] = Field(default_factory=dict)
    distance_metric: str = Field("euclidean")
    kernel_bandwidth: Optional[float] = None


class Constitution(BaseModel):
    """
    System constitution: all constraints and parameters

    Conforms to constitution.v1.schema.json
    """
    version: str
    fiscal_rules: FiscalRules
    carbon_budgets: Dict[str, CarbonBudget]
    devolution: DevolutionRules
    equality: EqualityRules
    diversity: DiversityParams
    security: Optional[Dict[str, Any]] = None
    data_sources: Optional[Dict[str, str]] = None
    effective_from: Optional[datetime] = None


# ============================================================================
# Diversity Certificate
# ============================================================================

class DiversityCertificate(BaseModel):
    """
    Certificate proving diversity of selected plans

    Includes mathematical proof and signatures
    """
    version: str = Field("diversity_cert/v1")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    selection: Dict[str, Any]
    proof: Dict[str, Any]
    explanation: str
    signature: Optional[Dict[str, Any]] = None
    id: UUID = Field(default_factory=uuid4)


# ============================================================================
# Household Cohort (for microsimulation)
# ============================================================================

class HouseholdCohort(BaseModel):
    """
    Single household cohort in synthetic population

    Represents a weighted group of similar households
    """
    cohort_id: int
    income_decile: int = Field(..., ge=1, le=10)
    household_type: str
    region: str
    gross_income: float
    equivalised_income: float
    benefit_flags: Dict[str, bool]
    tax_profile: Dict[str, float]
    consumption_basket: Dict[str, float]
    weight: float  # Number of real households represented

    @field_validator("weight")
    @classmethod
    def weight_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("weight must be positive")
        return v


# ============================================================================
# Economic Sector (for I/O modeling)
# ============================================================================

class EconomicSector(BaseModel):
    """Single sector in the economy"""
    sector_code: str
    sector_name: str
    parent_sector: Optional[str] = None
    gva_million_gbp: float
    employment_thousands: float
    emissions_tco2e: float
    emissions_intensity: float  # tCO2e per £million GVA
    year: int
