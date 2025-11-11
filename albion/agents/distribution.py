"""
DistributionAgent: Distributional Impact Analysis

Aggregates results from other agents and performs:
1. Decile analysis (income distribution)
2. Regional impact aggregation
3. Protected characteristics analysis (EqIA proxies)
4. Gini coefficient calculation
5. Poverty impact estimation
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from albion.models import DistributionalImpact, ImpactResult, Policy, RegionalImpact

logger = logging.getLogger(__name__)


class DistributionAgent:
    """
    Distributional analysis across income, geography, and protected groups
    """

    def __init__(self, households: pd.DataFrame):
        """
        Args:
            households: DataFrame of household cohorts with columns:
                       cohort_id, income_decile, household_type, region,
                       gross_income, weight, benefit_flags
        """
        self.households = households

        # Pre-compute distributional baselines
        self._compute_baseline_metrics()

        logger.info(f"Initialized DistributionAgent with {len(households):,} cohorts")

    def analyze_distribution(
        self,
        policy: Policy,
        tax_benefit_results: Dict[str, any],
        macro_results: Dict[str, any],
        climate_results: Dict[str, any]
    ) -> ImpactResult:
        """
        Complete distributional analysis

        Args:
            policy: Policy proposal
            tax_benefit_results: Output from TaxBenefitAgent
            macro_results: Output from MacroAgent
            climate_results: Output from ClimateAgent

        Returns:
            ImpactResult with all distributional metrics
        """
        logger.info(f"Analyzing distribution for policy {policy.id}")

        # 1. Decile analysis
        distributional_impact = self._analyze_deciles(tax_benefit_results)

        # 2. Regional analysis
        regional_impacts = macro_results.get('regional_impacts', {})

        # 3. Protected groups analysis (EqIA)
        protected_group_impacts = self._analyze_protected_groups(tax_benefit_results)

        # 4. Climate impact (already computed)
        climate_impact = self._format_climate_impact(climate_results)

        # 5. Macro impact (already computed)
        macro_impact = self._format_macro_impact(macro_results)

        result = ImpactResult(
            distribution=distributional_impact,
            regional=regional_impacts,
            climate=climate_impact,
            macro=macro_impact,
            protected_groups=protected_group_impacts
        )

        logger.info(
            f"Distribution analysis complete. "
            f"Gini Δ = {distributional_impact.gini_delta:+.3f}"
        )

        return result

    def _analyze_deciles(self, tax_benefit_results: Dict[str, any]) -> DistributionalImpact:
        """
        Analyze impact across income deciles

        Returns:
            DistributionalImpact with decile_deltas_pct, gini_delta, poverty_rate_delta
        """
        # Decile impacts as % of income (from TaxBenefitAgent)
        decile_impacts_pct = tax_benefit_results.get('decile_impacts_pct', [0.0] * 10)

        # Calculate Gini coefficient change
        gini_delta = self._calculate_gini_delta(tax_benefit_results)

        # Calculate poverty rate change
        poverty_delta = self._calculate_poverty_delta(tax_benefit_results)

        return DistributionalImpact(
            decile_deltas_pct=decile_impacts_pct,
            gini_delta=gini_delta,
            poverty_rate_delta_pp=poverty_delta
        )

    def _calculate_gini_delta(self, tax_benefit_results: Dict[str, any]) -> float:
        """
        Calculate change in Gini coefficient

        Gini measures income inequality (0 = perfect equality, 1 = perfect inequality)

        This is a simplified calculation. In production, would use full
        household-level income distributions.
        """
        # Get cohort-level results
        cohort_details = tax_benefit_results.get('detail_by_cohort')

        if cohort_details is None or len(cohort_details) == 0:
            return 0.0

        # Baseline Gini (pre-policy)
        baseline_incomes = cohort_details['gross_income'].values
        baseline_weights = cohort_details['weight'].values
        baseline_gini = self._gini_coefficient(baseline_incomes, baseline_weights)

        # Post-policy Gini
        net_deltas = cohort_details['net_delta'].values  # Household perspective
        post_policy_incomes = baseline_incomes - net_deltas  # Negative delta = better off
        post_policy_gini = self._gini_coefficient(post_policy_incomes, baseline_weights)

        gini_delta = post_policy_gini - baseline_gini

        return gini_delta

    def _gini_coefficient(self, incomes: np.ndarray, weights: np.ndarray) -> float:
        """
        Calculate weighted Gini coefficient

        Args:
            incomes: Array of income values
            weights: Array of weights (household counts)

        Returns:
            gini: Value between 0 and 1
        """
        # Sort by income
        sorted_indices = np.argsort(incomes)
        sorted_incomes = incomes[sorted_indices]
        sorted_weights = weights[sorted_indices]

        # Cumulative proportion of population and income
        cumulative_pop = np.cumsum(sorted_weights) / np.sum(sorted_weights)
        cumulative_income = np.cumsum(sorted_incomes * sorted_weights) / np.sum(sorted_incomes * sorted_weights)

        # Gini = (A) / (A + B) where A is area between Lorenz curve and diagonal
        # Approximated using trapezoidal rule
        n = len(cumulative_pop)
        gini = 1 - np.sum((cumulative_pop[1:] - cumulative_pop[:-1]) * (cumulative_income[1:] + cumulative_income[:-1]))

        return float(gini)

    def _calculate_poverty_delta(self, tax_benefit_results: Dict[str, any]) -> float:
        """
        Calculate change in poverty rate (percentage points)

        Poverty threshold: 60% of median equivalised income (standard UK measure)

        Returns:
            poverty_delta_pp: Change in poverty rate (percentage points)
        """
        cohort_details = tax_benefit_results.get('detail_by_cohort')

        if cohort_details is None or len(cohort_details) == 0:
            return 0.0

        # Find median equivalised income (baseline)
        # (In production, would recalculate using actual household data)
        # Simplified: use income distribution

        # Baseline poverty rate
        poverty_threshold = self._calculate_poverty_threshold(cohort_details['gross_income'].values)

        baseline_poor = np.sum(
            cohort_details['weight'].values[cohort_details['gross_income'].values < poverty_threshold]
        )
        total_households = np.sum(cohort_details['weight'].values)
        baseline_poverty_rate = baseline_poor / total_households * 100

        # Post-policy poverty rate
        net_deltas = cohort_details['net_delta'].values
        post_policy_incomes = cohort_details['gross_income'].values - net_deltas

        post_policy_poor = np.sum(
            cohort_details['weight'].values[post_policy_incomes < poverty_threshold]
        )
        post_policy_poverty_rate = post_policy_poor / total_households * 100

        poverty_delta = post_policy_poverty_rate - baseline_poverty_rate

        return poverty_delta

    def _calculate_poverty_threshold(self, incomes: np.ndarray) -> float:
        """60% of median income"""
        median_income = np.median(incomes)
        return 0.6 * median_income

    def _analyze_protected_groups(self, tax_benefit_results: Dict[str, any]) -> Dict[str, float]:
        """
        Analyze impact on protected groups (Equality Act 2010)

        Uses proxy indicators from benefit flags and household composition:
        - Disabled: households receiving disability benefits
        - Pensioners: households receiving state pension
        - Families with children: households with child-related benefits

        Returns:
            Dict mapping group_name → average impact (£/year)
        """
        cohort_details = tax_benefit_results.get('detail_by_cohort')

        if cohort_details is None or len(cohort_details) == 0:
            return {}

        protected_impacts = {}

        # Merge with household data to get benefit flags
        cohort_details_with_flags = pd.merge(
            cohort_details,
            self.households[['cohort_id', 'benefit_flags', 'household_type']],
            on='cohort_id',
            how='left'
        )

        # Disabled (proxy: disability benefits)
        disabled_mask = cohort_details_with_flags['benefit_flags'].apply(
            lambda x: x.get('disability_benefits', False) if isinstance(x, dict) else False
        )
        if disabled_mask.any():
            disabled_avg_impact = self._weighted_average_impact(
                cohort_details_with_flags[disabled_mask]
            )
            protected_impacts['disabled'] = disabled_avg_impact

        # Pensioners (proxy: state pension)
        pensioner_mask = cohort_details_with_flags['benefit_flags'].apply(
            lambda x: x.get('state_pension', False) if isinstance(x, dict) else False
        )
        if pensioner_mask.any():
            pensioner_avg_impact = self._weighted_average_impact(
                cohort_details_with_flags[pensioner_mask]
            )
            protected_impacts['pensioners'] = pensioner_avg_impact

        # Families with children (proxy: household type)
        family_mask = cohort_details_with_flags['household_type'].str.contains('child', case=False, na=False)
        if family_mask.any():
            family_avg_impact = self._weighted_average_impact(
                cohort_details_with_flags[family_mask]
            )
            protected_impacts['families_with_children'] = family_avg_impact

        return protected_impacts

    def _weighted_average_impact(self, group_data: pd.DataFrame) -> float:
        """Calculate weighted average impact for a group"""
        if len(group_data) == 0:
            return 0.0

        total_impact = np.sum(group_data['net_delta'].values * group_data['weight'].values)
        total_weight = np.sum(group_data['weight'].values)

        return float(total_impact / total_weight) if total_weight > 0 else 0.0

    def _format_climate_impact(self, climate_results: Dict[str, any]) -> any:
        """Format climate results into ClimateImpact model"""
        from albion.models import ClimateImpact

        return ClimateImpact(
            emissions_delta_mtco2e=climate_results.get('emissions_delta_mtco2e', 0.0),
            trajectory_2024_2037=climate_results.get('trajectory_2024_2037', []),
            budget_compliance=climate_results.get('budget_compliance', {})
        )

    def _format_macro_impact(self, macro_results: Dict[str, any]) -> any:
        """Format macro results into MacroImpact model"""
        from albion.models import MacroImpact

        return MacroImpact(
            gdp_delta_pct=macro_results.get('gdp_delta_pct', 0.0),
            employment_delta_thousands=macro_results.get('employment_delta_thousands', 0.0),
            debt_ratio_year5_pct=macro_results.get('debt_ratio_year5_pct', 90.0)
        )

    def _compute_baseline_metrics(self):
        """Pre-compute baseline distributional metrics for reference"""
        # Calculate baseline Gini
        baseline_incomes = self.households['gross_income'].values
        baseline_weights = self.households['weight'].values

        self.baseline_gini = self._gini_coefficient(baseline_incomes, baseline_weights)

        logger.info(f"Baseline Gini coefficient: {self.baseline_gini:.3f}")
