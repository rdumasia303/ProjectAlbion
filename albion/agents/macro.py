"""
MacroAgent: Input-Output Economic Modeling

Implements:
1. Leontief inverse matrix multiplication for sectoral spillovers
2. Tax elasticities from OBR ready-reckoners
3. Regional disaggregation
4. Employment multipliers

Based on:
- ONS Supply-Use & Input-Output Tables
- OBR Economic & Fiscal Outlook elasticities
- Regional economic accounts
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from albion.models import Policy, RegionalImpact

logger = logging.getLogger(__name__)


@dataclass
class OBRElasticities:
    """
    Tax elasticities from OBR ready-reckoners

    These capture behavioural responses to tax changes
    Source: OBR Economic & Fiscal Outlook (March 2025)
    """

    # Income tax elasticities (GDP impact per 1pp rate change)
    income_tax_basic_rate: float = -0.08  # 1pp increase → -0.08% GDP
    income_tax_higher_rate: float = -0.12  # Higher earners more responsive
    income_tax_additional_rate: float = -0.15

    # NICs elasticities
    nics_employee: float = -0.05
    nics_employer: float = -0.10  # Employer NICs hit employment more

    # Consumption taxes
    vat_standard_rate: float = -0.15
    vat_reduced_rate: float = -0.08

    # Business taxes
    corporation_tax: float = -0.20  # Investment-sensitive
    capital_gains_tax: float = -0.10

    # Multipliers (second-round effects)
    government_spending_multiplier: float = 1.35  # OBR central estimate


class MacroAgent:
    """
    Macroeconomic impact modeling via Input-Output analysis

    Uses Leontief inverse matrix to model how shocks propagate through
    the economy via inter-sectoral linkages.
    """

    def __init__(
        self,
        leontief_matrix: np.ndarray,
        sector_names: List[str],
        sector_data: pd.DataFrame,
        obr_baselines: Dict[str, any],
    ):
        """
        Args:
            leontief_matrix: (I-A)^-1 matrix (n_sectors × n_sectors)
            sector_names: List of sector labels (length n_sectors)
            sector_data: DataFrame with columns: sector_code, gva_million_gbp,
                        employment_thousands, region
            obr_baselines: OBR baseline forecasts (GDP, employment, etc.)
        """
        self.L = leontief_matrix  # Leontief inverse
        self.sector_names = sector_names
        self.sector_data = sector_data
        self.baselines = obr_baselines
        self.elasticities = OBRElasticities()

        self.n_sectors = len(sector_names)

        # Validate matrix dimensions
        assert self.L.shape == (self.n_sectors, self.n_sectors), \
            f"Leontief matrix shape {self.L.shape} doesn't match {self.n_sectors} sectors"

        # Build sector → department mapping
        self.dept_sector_mapping = self._build_dept_sector_mapping()

        logger.info(f"Initialized MacroAgent with {self.n_sectors} sectors")

    def simulate_macro_impact(
        self,
        policy: Policy,
        tax_benefit_results: Dict[str, any]
    ) -> Dict[str, any]:
        """
        Calculate macroeconomic impacts

        Steps:
        1. Tax behavioural responses (via elasticities)
        2. Government spending shocks (via Leontief multipliers)
        3. Regional disaggregation
        4. Employment effects

        Args:
            policy: Policy proposal
            tax_benefit_results: Output from TaxBenefitAgent

        Returns:
            {
                'gdp_delta_pct': float,
                'gdp_delta_bn_gbp': float,
                'employment_delta_thousands': float,
                'sector_output_changes': np.ndarray,
                'regional_impacts': Dict[str, RegionalImpact],
                'debt_ratio_year5_pct': float
            }
        """
        logger.info(f"Calculating macro impacts for policy {policy.id}")

        # 1. Tax elasticity effects on GDP
        gdp_tax_effect_pct = self._calculate_tax_elasticity_effects(policy)

        # 2. Departmental spending shocks
        sector_shocks = self._calculate_spending_shocks(policy)

        # 3. Apply Leontief multiplier
        total_output_changes = self.L @ sector_shocks

        # Convert output changes to GDP impact
        # (Simplified: assume output changes map linearly to GVA)
        total_gva_change_m = np.sum(
            total_output_changes * self.sector_data['gva_million_gbp'].values
        )
        baseline_gdp_bn = self.baselines.get('gdp_bn_gbp', 2800)
        gdp_spending_effect_pct = (total_gva_change_m / 1000) / baseline_gdp_bn * 100

        # 4. Total GDP effect
        gdp_delta_pct = gdp_tax_effect_pct + gdp_spending_effect_pct
        gdp_delta_bn = gdp_delta_pct / 100 * baseline_gdp_bn

        # 5. Employment effect
        employment_delta = self._calculate_employment_effect(
            total_output_changes,
            gdp_delta_pct
        )

        # 6. Regional disaggregation
        regional_impacts = self._disaggregate_to_regions(
            total_output_changes,
            employment_delta
        )

        # 7. Debt dynamics
        debt_ratio_year5 = self._calculate_debt_trajectory(
            policy,
            tax_benefit_results,
            gdp_delta_pct
        )

        results = {
            'gdp_delta_pct': gdp_delta_pct,
            'gdp_delta_bn_gbp': gdp_delta_bn,
            'employment_delta_thousands': employment_delta,
            'sector_output_changes': total_output_changes.tolist(),
            'regional_impacts': regional_impacts,
            'debt_ratio_year5_pct': debt_ratio_year5,
            'decomposition': {
                'tax_effect_pct': gdp_tax_effect_pct,
                'spending_effect_pct': gdp_spending_effect_pct
            }
        }

        logger.info(
            f"Macro impact: GDP Δ = {gdp_delta_pct:+.2f}%, "
            f"Employment Δ = {employment_delta:+,.0f}k"
        )

        return results

    def _calculate_tax_elasticity_effects(self, policy: Policy) -> float:
        """
        Calculate GDP impact from tax changes using OBR elasticities

        Returns:
            GDP delta in percentage points
        """
        gdp_effect = 0.0
        levers = policy.levers

        # Income tax
        gdp_effect += levers.income_tax.basic_rate_pp * self.elasticities.income_tax_basic_rate
        gdp_effect += levers.income_tax.higher_rate_pp * self.elasticities.income_tax_higher_rate
        gdp_effect += levers.income_tax.additional_rate_pp * self.elasticities.income_tax_additional_rate

        # NICs
        gdp_effect += levers.nics.class1_main_pp * self.elasticities.nics_employee

        # VAT
        gdp_effect += levers.vat.standard_rate_pp * self.elasticities.vat_standard_rate

        # Corporation tax
        gdp_effect += levers.corp_tax.main_rate_pp * self.elasticities.corporation_tax

        return gdp_effect

    def _calculate_spending_shocks(self, policy: Policy) -> np.ndarray:
        """
        Map departmental spending changes to sector shocks

        Returns:
            sector_shocks: np.ndarray of length n_sectors (as % of sector output)
        """
        sector_shocks = np.zeros(self.n_sectors)

        for dept_lever in policy.levers.departmental:
            dept_code = dept_lever.dept
            rdel_pct_change = dept_lever.rdel_pct or 0.0

            # Map department to sectors
            sector_weights = self.dept_sector_mapping.get(dept_code, {})

            for sector_idx, weight in sector_weights.items():
                # weight = proportion of department spending going to this sector
                # Apply the percentage change weighted
                sector_shocks[sector_idx] += rdel_pct_change * weight

        return sector_shocks

    def _build_dept_sector_mapping(self) -> Dict[str, Dict[int, float]]:
        """
        Build mapping from government departments to economic sectors

        Returns:
            {
                'DfT': {sector_idx: weight, ...},  # Transport dept → construction, services
                'DfE': {sector_idx: weight, ...},  # Education → education sector
                ...
            }

        This would ideally be calibrated from PESA data showing where each
        department actually spends money. For now, using stylised mappings.
        """
        mappings = {}

        # Create sector name → index lookup
        sector_idx_map = {name: idx for idx, name in enumerate(self.sector_names)}

        # Stylised mappings (in production, calibrate from PESA)
        mappings['DfT'] = {  # Transport
            sector_idx_map.get('Construction', 0): 0.40,
            sector_idx_map.get('Transport & storage', 1): 0.35,
            sector_idx_map.get('Professional services', 2): 0.25,
        }

        mappings['DfE'] = {  # Education
            sector_idx_map.get('Education', 3): 0.80,
            sector_idx_map.get('Professional services', 2): 0.20,
        }

        mappings['DHSC'] = {  # Health
            sector_idx_map.get('Health & social work', 4): 0.70,
            sector_idx_map.get('Pharmaceuticals', 5): 0.15,
            sector_idx_map.get('Professional services', 2): 0.15,
        }

        mappings['HO'] = {  # Home Office (police, etc.)
            sector_idx_map.get('Public admin & defence', 6): 0.60,
            sector_idx_map.get('Professional services', 2): 0.40,
        }

        mappings['MoD'] = {  # Defence
            sector_idx_map.get('Public admin & defence', 6): 0.50,
            sector_idx_map.get('Manufacturing', 7): 0.30,
            sector_idx_map.get('Professional services', 2): 0.20,
        }

        # Fill in with uniform distribution if sector not found
        for dept, mapping in mappings.items():
            cleaned = {}
            for sector_idx, weight in mapping.items():
                if sector_idx < self.n_sectors:
                    cleaned[sector_idx] = weight
            mappings[dept] = cleaned

        return mappings

    def _calculate_employment_effect(
        self,
        output_changes: np.ndarray,
        gdp_delta_pct: float
    ) -> float:
        """
        Calculate employment impact in thousands

        Uses sector-specific employment multipliers from I/O tables
        """
        # Sector employment intensities (jobs per £million GVA)
        # From ONS I/O tables
        employment_intensities = self.sector_data.get('employment_per_million_gva', np.ones(self.n_sectors) * 15)

        # Employment change from output changes
        employment_from_output = np.sum(
            output_changes * self.sector_data['gva_million_gbp'].values * employment_intensities
        )

        # Convert to thousands
        employment_delta_thousands = employment_from_output / 1000

        # Also add direct GDP elasticity effect
        # Rule of thumb: 1% GDP growth → 0.7% employment growth (Okun's law variant)
        baseline_employment_thousands = self.baselines.get('employment_thousands', 33_000)
        employment_from_gdp = gdp_delta_pct * 0.7 / 100 * baseline_employment_thousands

        total_employment = employment_from_output / 1000 + employment_from_gdp

        return total_employment

    def _disaggregate_to_regions(
        self,
        output_changes: np.ndarray,
        employment_delta: float
    ) -> Dict[str, RegionalImpact]:
        """
        Disaggregate national impacts to ITL1 regions

        Uses regional sector composition data
        """
        regions = ['UKC', 'UKD', 'UKE', 'UKF', 'UKG', 'UKH', 'UKI', 'UKJ', 'UKK', 'UKL', 'UKM', 'UKN']
        region_names = {
            'UKC': 'North East',
            'UKD': 'North West',
            'UKE': 'Yorkshire and The Humber',
            'UKF': 'East Midlands',
            'UKG': 'West Midlands',
            'UKH': 'East of England',
            'UKI': 'London',
            'UKJ': 'South East',
            'UKK': 'South West',
            'UKL': 'Wales',
            'UKM': 'Scotland',
            'UKN': 'Northern Ireland'
        }

        regional_impacts = {}

        for region_code in regions:
            # Get region's share of each sector
            # (In production, from ONS regional accounts)
            # Stylised: assume proportional to population/GVA
            region_shares = self._get_region_sector_shares(region_code)

            # Regional output change
            regional_output = np.sum(output_changes * region_shares)

            # Simplified regional GDP calculation
            # Assume output maps to GVA and region has baseline GVA share
            region_gdp_share = self._get_region_gdp_share(region_code)
            baseline_gdp_bn = self.baselines.get('gdp_bn_gbp', 2800)
            region_baseline_gdp = baseline_gdp_bn * region_gdp_share

            regional_gdp_delta_pct = (regional_output / 100) / region_baseline_gdp * 100

            # Employment (proportional)
            regional_employment_delta = employment_delta * region_gdp_share

            regional_impacts[region_code] = RegionalImpact(
                gdp_delta_pct=regional_gdp_delta_pct,
                employment_delta_thousands=regional_employment_delta
            )

        return regional_impacts

    def _get_region_sector_shares(self, region_code: str) -> np.ndarray:
        """
        Get region's share of each sector's output

        Returns: array of length n_sectors (values sum to ~1 across regions)
        """
        # Stylised shares (in production, from ONS regional accounts)
        # For now, assume uniform except for known specialisations

        shares = np.ones(self.n_sectors) * (1.0 / 12)  # 12 regions

        # London has higher share of financial services
        if region_code == 'UKI':
            shares[self.sector_names.index('Financial services')] = 0.40
            shares[self.sector_names.index('Professional services')] = 0.30

        # North East has lower overall share
        elif region_code == 'UKC':
            shares *= 0.6

        return shares

    def _get_region_gdp_share(self, region_code: str) -> float:
        """Regional share of UK GDP (stylised, from ONS data)"""
        shares = {
            'UKC': 0.031,  # North East (3.1%)
            'UKD': 0.093,  # North West
            'UKE': 0.071,  # Yorkshire
            'UKF': 0.061,  # East Midlands
            'UKG': 0.078,  # West Midlands
            'UKH': 0.099,  # East of England
            'UKI': 0.236,  # London (largest)
            'UKJ': 0.147,  # South East
            'UKK': 0.072,  # South West
            'UKL': 0.032,  # Wales
            'UKM': 0.070,  # Scotland
            'UKN': 0.021,  # Northern Ireland
        }
        return shares.get(region_code, 1.0 / 12)

    def _calculate_debt_trajectory(
        self,
        policy: Policy,
        tax_benefit_results: Dict[str, any],
        gdp_delta_pct: float
    ) -> float:
        """
        Calculate debt-to-GDP ratio in year 5

        Simple debt dynamics:
        debt_{t+1} = debt_t + deficit_t

        Args:
            policy: Policy proposal
            tax_benefit_results: Revenue impacts
            gdp_delta_pct: GDP growth impact

        Returns:
            debt_ratio_year5_pct: Debt as % of GDP in year 5
        """
        # Baseline debt trajectory from OBR
        baseline_debt_ratio = self.baselines.get('debt_ratio_pct', [95, 94, 93, 92, 91])

        # Revenue impact (£bn per year)
        annual_revenue_bn = tax_benefit_results.get('total_revenue_delta_bn', 0.0)

        # GDP impact compounds
        baseline_gdp_bn = self.baselines.get('gdp_bn_gbp', 2800)
        gdp_year5_bn = baseline_gdp_bn * ((1 + gdp_delta_pct / 100) ** 5)

        # Simple debt accumulation
        # debt_year5 = debt_year0 - (annual_revenue × 5)
        baseline_debt_year5_bn = baseline_gdp_bn * baseline_debt_ratio[4] / 100
        debt_year5_bn = baseline_debt_year5_bn - (annual_revenue_bn * 5)

        # Debt ratio
        debt_ratio_year5_pct = debt_year5_bn / gdp_year5_bn * 100

        return debt_ratio_year5_pct
