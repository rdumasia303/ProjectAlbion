"""
ClimateAgent: Integrated Assessment Modeling

Implements:
1. Emissions intensity linkage to economic sectors
2. Marginal Abatement Cost Curves (MACC)
3. Carbon Budget compliance checking
4. Carbon price revenue calculation

Based on:
- DESNZ UK greenhouse gas emissions statistics
- Climate Change Committee Marginal Abatement Cost data
- Carbon Budget Orders (legal limits)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from albion.models import ClimateImpact, Policy

logger = logging.getLogger(__name__)


@dataclass
class CarbonBudgetLimit:
    """Legal carbon budget from Carbon Budget Orders"""
    budget_name: str  # e.g., "Sixth"
    start_year: int
    end_year: int
    total_mtco2e: float  # Total emissions allowed over period
    annual_allocation: List[float]  # Allocation by year (if available)


class MACCurve:
    """
    Marginal Abatement Cost Curve for a sector

    Maps carbon price → emissions abatement

    Based on CCC data showing technology-specific abatement potentials
    """

    def __init__(
        self,
        sector_code: str,
        baseline_emissions_mtco2e: float,
        abatement_potentials: List[Tuple[float, float]]  # [(price, abatement), ...]
    ):
        """
        Args:
            sector_code: Sector identifier
            baseline_emissions_mtco2e: Current annual emissions (MtCO2e)
            abatement_potentials: List of (carbon_price, cumulative_abatement_mtco2e)
                                 Sorted by price ascending
        """
        self.sector_code = sector_code
        self.baseline_emissions = baseline_emissions_mtco2e

        # Sort by price
        self.abatement_potentials = sorted(abatement_potentials, key=lambda x: x[0])

        # Extract prices and abatements
        self.prices = np.array([p for p, _ in self.abatement_potentials])
        self.abatements = np.array([a for _, a in self.abatement_potentials])

    def abatement_at_price(self, carbon_price: float) -> float:
        """
        Calculate emissions abatement at given carbon price

        Uses linear interpolation between MACC points

        Args:
            carbon_price: £ per tCO2e

        Returns:
            abatement_mtco2e: Annual emissions reduction (MtCO2e)
        """
        if carbon_price <= self.prices[0]:
            return 0.0

        if carbon_price >= self.prices[-1]:
            return self.abatements[-1]  # Maximum abatement

        # Linear interpolation
        abatement = np.interp(carbon_price, self.prices, self.abatements)

        return float(abatement)

    def revenue_at_price(self, carbon_price: float) -> float:
        """
        Calculate carbon price revenue

        Revenue = price × residual_emissions
        where residual_emissions = baseline - abatement

        Note: This is simplified. In reality, revenue depends on whether
        it's a tax (gov revenue) or ETS (auction revenue).

        Args:
            carbon_price: £ per tCO2e

        Returns:
            revenue_million_gbp: Annual revenue
        """
        abatement = self.abatement_at_price(carbon_price)
        residual_emissions = max(0, self.baseline_emissions - abatement)

        # Convert MtCO2e to tCO2e and calculate revenue
        revenue_million = residual_emissions * 1_000_000 * carbon_price / 1_000_000

        return revenue_million


class ClimateAgent:
    """
    Climate and emissions impact modeling

    Links economic activity to emissions via sector intensities,
    and models carbon price abatement via MACC curves.
    """

    def __init__(
        self,
        emissions_intensities: pd.DataFrame,
        macc_data: Dict[str, MACCurve],
        carbon_budgets: List[CarbonBudgetLimit],
        baseline_trajectory: Optional[Dict[int, float]] = None
    ):
        """
        Args:
            emissions_intensities: DataFrame with columns:
                                  sector_code, emissions_tco2e, gva_million_gbp, intensity
            macc_data: Dict mapping sector_code → MACCurve
            carbon_budgets: List of legal carbon budget limits
            baseline_trajectory: Dict {year: emissions_mtco2e} for baseline scenario
        """
        self.intensities = emissions_intensities
        self.macc_data = macc_data
        self.carbon_budgets = carbon_budgets

        # Default baseline trajectory if not provided
        self.baseline_trajectory = baseline_trajectory or self._default_baseline_trajectory()

        logger.info(
            f"Initialized ClimateAgent with {len(macc_data)} sector MACC curves, "
            f"{len(carbon_budgets)} carbon budgets"
        )

    def simulate_emissions_impact(
        self,
        macro_results: Dict[str, any],
        policy: Policy
    ) -> Dict[str, any]:
        """
        Calculate emissions impact

        Two components:
        1. Economic activity effect (via sector output changes × intensities)
        2. Carbon price abatement effect (via MACC)

        Args:
            macro_results: Output from MacroAgent (sector output changes)
            policy: Policy proposal (with carbon price lever)

        Returns:
            {
                'emissions_delta_mtco2e': float,
                'trajectory_2024_2037': List[float],
                'budget_compliance': Dict,
                'carbon_revenue_bn_gbp': float,
                'abatement_by_sector': Dict[str, float]
            }
        """
        logger.info(f"Calculating climate impact for policy {policy.id}")

        # 1. Extract carbon price from policy
        carbon_price_start = policy.levers.ets_carbon_price.start_gbp_per_tco2e
        carbon_price_ramp = policy.levers.ets_carbon_price.ramp_ppy

        # 2. Economic activity effect on emissions
        sector_output_changes = np.array(macro_results.get('sector_output_changes', []))
        activity_emissions_delta = self._calculate_activity_emissions(sector_output_changes)

        # 3. Carbon price abatement effect
        abatement_results = self._calculate_carbon_price_abatement(
            carbon_price_start,
            carbon_price_ramp
        )

        # 4. Net emissions change
        net_emissions_delta = activity_emissions_delta - abatement_results['total_abatement_mtco2e']

        # 5. Build emissions trajectory (2024-2037)
        trajectory = self._build_emissions_trajectory(
            net_emissions_delta,
            carbon_price_start,
            carbon_price_ramp
        )

        # 6. Check carbon budget compliance
        budget_compliance = self._check_carbon_budget_compliance(trajectory)

        # 7. Calculate carbon revenue
        carbon_revenue_bn = abatement_results['revenue_million_gbp'] / 1000

        results = {
            'emissions_delta_mtco2e': net_emissions_delta,
            'trajectory_2024_2037': trajectory,
            'budget_compliance': budget_compliance,
            'carbon_revenue_bn_gbp': carbon_revenue_bn,
            'abatement_by_sector': abatement_results['abatement_by_sector'],
            'decomposition': {
                'activity_effect_mtco2e': activity_emissions_delta,
                'abatement_effect_mtco2e': -abatement_results['total_abatement_mtco2e']
            }
        }

        logger.info(
            f"Climate impact: Emissions Δ = {net_emissions_delta:+.1f} MtCO2e, "
            f"Budget status = {budget_compliance['status']}"
        )

        return results

    def _calculate_activity_emissions(self, sector_output_changes: np.ndarray) -> float:
        """
        Calculate emissions change from economic activity changes

        emissions_delta = Σ (output_change_i × intensity_i)

        Args:
            sector_output_changes: % changes in sector outputs

        Returns:
            emissions_delta_mtco2e: Change in annual emissions
        """
        if len(sector_output_changes) == 0:
            return 0.0

        # Match sector output changes to intensity data
        # (Assumes same ordering)
        intensities = self.intensities['intensity'].values[:len(sector_output_changes)]
        base_gva = self.intensities['gva_million_gbp'].values[:len(sector_output_changes)]

        # emissions_change = intensity × (output_change% / 100) × base_GVA
        emissions_changes = intensities * (sector_output_changes / 100) * base_gva

        total_emissions_delta_t = np.sum(emissions_changes)  # In tCO2e

        # Convert to MtCO2e
        emissions_delta_mt = total_emissions_delta_t / 1_000_000

        return emissions_delta_mt

    def _calculate_carbon_price_abatement(
        self,
        carbon_price_start: float,
        carbon_price_ramp: float
    ) -> Dict[str, any]:
        """
        Calculate emissions abatement from carbon pricing via MACC

        Args:
            carbon_price_start: Starting carbon price (£/tCO2e)
            carbon_price_ramp: Annual price increase (£/tCO2e/year)

        Returns:
            {
                'total_abatement_mtco2e': float,
                'revenue_million_gbp': float,
                'abatement_by_sector': Dict[str, float]
            }
        """
        # Use year 1 carbon price for steady-state calculation
        # (In full model, would iterate year-by-year)
        carbon_price = carbon_price_start

        total_abatement = 0.0
        total_revenue = 0.0
        abatement_by_sector = {}

        for sector_code, macc in self.macc_data.items():
            sector_abatement = macc.abatement_at_price(carbon_price)
            sector_revenue = macc.revenue_at_price(carbon_price)

            total_abatement += sector_abatement
            total_revenue += sector_revenue

            abatement_by_sector[sector_code] = sector_abatement

        return {
            'total_abatement_mtco2e': total_abatement,
            'revenue_million_gbp': total_revenue,
            'abatement_by_sector': abatement_by_sector
        }

    def _build_emissions_trajectory(
        self,
        net_emissions_delta: float,
        carbon_price_start: float,
        carbon_price_ramp: float
    ) -> List[float]:
        """
        Build annual emissions trajectory 2024-2037

        Assumes:
        - Policy implemented from 2025
        - Carbon price ramps linearly
        - Abatement increases with carbon price
        - Baseline trajectory from CBGDP

        Returns:
            List of 14 annual emissions values (MtCO2e)
        """
        trajectory = []

        for year_offset in range(14):  # 2024-2037
            year = 2024 + year_offset

            # Baseline emissions
            baseline_emissions = self.baseline_trajectory.get(year, 400.0)

            if year < 2025:
                # Pre-policy
                trajectory.append(baseline_emissions)
            else:
                # Policy active
                years_active = year - 2025

                # Carbon price increases over time
                carbon_price = carbon_price_start + (years_active * carbon_price_ramp)

                # Recalculate abatement at this price
                abatement_results = self._calculate_carbon_price_abatement(carbon_price, 0)
                abatement = abatement_results['total_abatement_mtco2e']

                # Net emissions
                net_emissions = baseline_emissions - abatement

                trajectory.append(net_emissions)

        return trajectory

    def _check_carbon_budget_compliance(
        self,
        trajectory: List[float]
    ) -> Dict[str, any]:
        """
        Check compliance with legal carbon budgets

        Args:
            trajectory: Annual emissions 2024-2037 (MtCO2e)

        Returns:
            {
                'status': 'COMPLIANT' | 'TIGHT' | 'BREACH',
                'headroom_or_overshoot_mtco2e': float,
                'budget_name': str,
                'budget_limit_mtco2e': float,
                'projected_total_mtco2e': float
            }
        """
        # Check Sixth Carbon Budget (2033-2037)
        sixth_budget = next(
            (b for b in self.carbon_budgets if b.budget_name == "Sixth"),
            None
        )

        if not sixth_budget:
            logger.warning("Sixth Carbon Budget not defined")
            return {'status': 'UNKNOWN'}

        # Sum emissions over budget period
        budget_start_idx = sixth_budget.start_year - 2024
        budget_end_idx = sixth_budget.end_year - 2024 + 1

        projected_total = sum(trajectory[budget_start_idx:budget_end_idx])
        budget_limit = sixth_budget.total_mtco2e

        headroom = budget_limit - projected_total

        if headroom < 0:
            status = 'BREACH'
        elif headroom < 50:  # Less than 50 MtCO2e headroom
            status = 'TIGHT'
        else:
            status = 'COMPLIANT'

        return {
            'status': status,
            'headroom_or_overshoot_mtco2e': headroom,
            'budget_name': sixth_budget.budget_name,
            'budget_limit_mtco2e': budget_limit,
            'projected_total_mtco2e': projected_total,
            'budget_years': f"{sixth_budget.start_year}-{sixth_budget.end_year}"
        }

    def _default_baseline_trajectory(self) -> Dict[int, float]:
        """
        Default baseline emissions trajectory

        Based on CBGDP central scenario (simplified)
        Declining from ~420 MtCO2e (2024) to ~290 MtCO2e (2037)
        """
        # Linear decline from 420 to 290 over 2024-2037
        start_year = 2024
        end_year = 2037
        start_emissions = 420.0
        end_emissions = 290.0

        trajectory = {}
        for year in range(start_year, end_year + 1):
            progress = (year - start_year) / (end_year - start_year)
            emissions = start_emissions + progress * (end_emissions - start_emissions)
            trajectory[year] = emissions

        return trajectory


# ============================================================================
# MACC Data Loading Utilities
# ============================================================================

def build_default_macc_curves() -> Dict[str, MACCurve]:
    """
    Build default MACC curves for key sectors

    Based on CCC Sixth Carbon Budget technical annex

    Returns:
        Dict mapping sector_code → MACCurve
    """
    macc_curves = {}

    # Power sector MACC
    # Technology potentials: wind, solar, nuclear, CCS
    macc_curves['power'] = MACCurve(
        sector_code='power',
        baseline_emissions_mtco2e=100.0,
        abatement_potentials=[
            (40, 10),   # £40/tCO2e → 10 MtCO2e abatement (wind)
            (60, 25),   # £60/tCO2e → 25 MtCO2e (solar)
            (80, 45),   # £80/tCO2e → 45 MtCO2e (wind+solar+nuclear)
            (120, 70),  # £120/tCO2e → 70 MtCO2e (+ CCS)
            (200, 85),  # £200/tCO2e → 85 MtCO2e (deep decarbonisation)
        ]
    )

    # Industry MACC
    # Technology potentials: fuel switching, electrification, CCS
    macc_curves['industry'] = MACCurve(
        sector_code='industry',
        baseline_emissions_mtco2e=80.0,
        abatement_potentials=[
            (50, 5),    # Efficiency improvements
            (80, 15),   # Fuel switching (gas → hydrogen/electric)
            (120, 30),  # Electrification + CCS
            (180, 50),  # Deep decarbonisation
        ]
    )

    # Transport MACC
    # Technology potentials: EVs, efficiency, modal shift
    macc_curves['transport'] = MACCurve(
        sector_code='transport',
        baseline_emissions_mtco2e=110.0,
        abatement_potentials=[
            (30, 10),   # Efficiency (ICE improvements)
            (60, 30),   # EV uptake (cars)
            (100, 55),  # EV uptake (cars + vans)
            (150, 75),  # EVs + HGV transition + modal shift
        ]
    )

    # Buildings MACC
    # Technology potentials: insulation, heat pumps
    macc_curves['buildings'] = MACCurve(
        sector_code='buildings',
        baseline_emissions_mtco2e=70.0,
        abatement_potentials=[
            (40, 10),   # Insulation
            (70, 25),   # Heat pumps (partial)
            (110, 45),  # Heat pumps (widespread)
            (160, 60),  # Deep retrofit
        ]
    )

    # Agriculture MACC
    # Technology potentials: methane reduction, efficiency
    macc_curves['agriculture'] = MACCurve(
        sector_code='agriculture',
        baseline_emissions_mtco2e=45.0,
        abatement_potentials=[
            (60, 5),    # Efficiency improvements
            (100, 12),  # Methane reduction (livestock)
            (150, 20),  # Land use changes
        ]
    )

    return macc_curves


def load_carbon_budgets() -> List[CarbonBudgetLimit]:
    """
    Load legal carbon budgets from Carbon Budget Orders

    Returns:
        List of CarbonBudgetLimit objects
    """
    budgets = [
        CarbonBudgetLimit(
            budget_name="Sixth",
            start_year=2033,
            end_year=2037,
            total_mtco2e=965.0,  # Legal limit from Sixth Carbon Budget Order
            annual_allocation=[193] * 5  # Approximate even split
        ),
    ]

    return budgets
