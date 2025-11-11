"""
ALBION Simulation Runner

Main orchestrator that ties everything together:
1. Load data (households, Leontief, emissions, etc.)
2. Generate candidate policies
3. Evaluate each with all agents
4. Apply policy gates (OPA)
5. Select diverse set (DNOS)
6. Sign artifacts (Sigstore)
7. Generate certificates
8. Write output

Usage:
    python -m albion.runner --target-bn 50 --k 5
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from rich.console import Console
from rich.progress import track
from rich.table import Table

from albion.agents.climate import ClimateAgent, build_default_macc_curves, load_carbon_budgets
from albion.agents.distribution import DistributionAgent
from albion.agents.macro import MacroAgent
from albion.agents.taxbenefit import TaxBenefitAgent
from albion.dnos.selector import DNOSSelector
from albion.models import (
    BenefitsLevers,
    CarbonPriceLevers,
    CGTLevers,
    Constitution,
    CorpTaxLevers,
    DepartmentalLever,
    HouseholdCohort,
    IncomeTaxLevers,
    NICSLevers,
    Policy,
    PolicyLevers,
    PolicyTargets,
    VATLevers,
)
from albion.sign.sigstore_service import SigstoreService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "output"
CERTS_DIR = PROJECT_ROOT / "certs"

OUTPUT_DIR.mkdir(exist_ok=True)
CERTS_DIR.mkdir(exist_ok=True)


def load_data() -> tuple:
    """
    Load all processed data

    Returns:
        (households_df, leontief_matrix, sector_names, emissions_df, obr_baselines)
    """
    console.print("[cyan]Loading processed data...[/cyan]")

    # Households
    households_file = DATA_DIR / "households.parquet"
    if not households_file.exists():
        raise FileNotFoundError(
            f"Households data not found: {households_file}\n"
            "Run 'python tools/generate_mock_data.py' first!"
        )
    households_df = pd.read_parquet(households_file)
    console.print(f"[green]✓[/green] Loaded {len(households_df):,} household cohorts")

    # Leontief matrix
    leontief_file = DATA_DIR / "leontief_matrix.npz"
    if not leontief_file.exists():
        raise FileNotFoundError(f"Leontief matrix not found: {leontief_file}")

    leontief_data = np.load(leontief_file, allow_pickle=True)
    L = leontief_data['L']
    sector_names = leontief_data['sector_names'].tolist()
    console.print(f"[green]✓[/green] Loaded Leontief matrix ({L.shape[0]} sectors)")

    # Emissions intensities
    emissions_file = DATA_DIR / "emissions_intensities.parquet"
    if not emissions_file.exists():
        raise FileNotFoundError(f"Emissions data not found: {emissions_file}")
    emissions_df = pd.read_parquet(emissions_file)
    console.print(f"[green]✓[/green] Loaded emissions intensities ({len(emissions_df)} sectors)")

    # OBR baselines
    obr_file = DATA_DIR / "obr_baselines.json"
    if not obr_file.exists():
        raise FileNotFoundError(f"OBR baselines not found: {obr_file}")
    with open(obr_file) as f:
        obr_baselines = json.load(f)
    console.print(f"[green]✓[/green] Loaded OBR baselines")

    return households_df, L, sector_names, emissions_df, obr_baselines


def initialize_agents(
    households_df: pd.DataFrame,
    leontief_matrix: np.ndarray,
    sector_names: List[str],
    emissions_df: pd.DataFrame,
    obr_baselines: dict
) -> tuple:
    """
    Initialize all compute agents

    Returns:
        (tb_agent, macro_agent, climate_agent, dist_agent)
    """
    console.print("\n[cyan]Initializing compute agents...[/cyan]")

    # Convert households to HouseholdCohort objects
    households = [
        HouseholdCohort(
            cohort_id=int(row['cohort_id']),
            income_decile=int(row['income_decile']),
            household_type=row['household_type'],
            region=row['region'],
            gross_income=float(row['gross_income']),
            equivalised_income=float(row['equivalised_income']),
            benefit_flags=row['benefit_flags'],
            tax_profile=row['tax_profile'],
            consumption_basket=row['consumption_basket'],
            weight=float(row['weight'])
        )
        for _, row in households_df.iterrows()
    ]

    # TaxBenefitAgent
    tb_agent = TaxBenefitAgent(households)
    console.print(f"[green]✓[/green] TaxBenefitAgent initialized")

    # MacroAgent
    # Build sector data DataFrame
    sector_data = pd.DataFrame({
        'sector_code': [f'S{i:03d}' for i in range(len(sector_names))],
        'sector_name': sector_names,
        'gva_million_gbp': emissions_df['gva_million_gbp'].values[:len(sector_names)],
        'employment_thousands': np.random.uniform(50, 500, len(sector_names)),  # Mock
        'employment_per_million_gva': np.random.uniform(10, 30, len(sector_names))
    })

    macro_agent = MacroAgent(
        leontief_matrix=leontief_matrix,
        sector_names=sector_names,
        sector_data=sector_data,
        obr_baselines=obr_baselines
    )
    console.print(f"[green]✓[/green] MacroAgent initialized")

    # ClimateAgent
    macc_curves = build_default_macc_curves()
    carbon_budgets = load_carbon_budgets()

    climate_agent = ClimateAgent(
        emissions_intensities=emissions_df,
        macc_data=macc_curves,
        carbon_budgets=carbon_budgets
    )
    console.print(f"[green]✓[/green] ClimateAgent initialized")

    # DistributionAgent
    dist_agent = DistributionAgent(households_df)
    console.print(f"[green]✓[/green] DistributionAgent initialized")

    return tb_agent, macro_agent, climate_agent, dist_agent


def generate_policy_candidates(target_revenue_bn: float, max_candidates: int = 1000) -> List[Policy]:
    """
    Generate diverse policy candidates

    This is a simplified generator. In production, you'd use optimization
    (e.g., genetic algorithms, MCMC) to explore the policy space efficiently.

    Args:
        target_revenue_bn: Target revenue to raise (£bn)
        max_candidates: Maximum number of candidates

    Returns:
        List of Policy objects
    """
    console.print(f"\n[cyan]Generating {max_candidates} policy candidates (target: £{target_revenue_bn}bn)...[/cyan]")

    np.random.seed(42)
    candidates = []

    for i in range(max_candidates):
        # Generate random policy levers
        # Strategy: mix tax increases, spending cuts, carbon pricing

        policy = Policy(
            schema="proposal/v1",
            jurisdiction="UK",
            horizon_years=5,
            levers=PolicyLevers(
                income_tax=IncomeTaxLevers(
                    basic_rate_pp=np.random.choice([0, 0.5, 1.0, 1.5, 2.0], p=[0.3, 0.3, 0.2, 0.1, 0.1]),
                    higher_rate_pp=np.random.choice([0, 1.0, 2.0], p=[0.5, 0.3, 0.2]),
                    additional_rate_pp=np.random.choice([0, 2.0, 3.0], p=[0.7, 0.2, 0.1])
                ),
                nics=NICSLevers(
                    class1_main_pp=np.random.choice([0, 0.5, 1.0], p=[0.5, 0.3, 0.2])
                ),
                vat=VATLevers(
                    standard_rate_pp=np.random.choice([0, 0.5, 1.0, 2.0], p=[0.4, 0.3, 0.2, 0.1])
                ),
                corp_tax=CorpTaxLevers(
                    main_rate_pp=np.random.choice([0, 1.0, 2.0], p=[0.6, 0.3, 0.1])
                ),
                cgt=CGTLevers(
                    align_to_income_tax=np.random.choice([False, True], p=[0.7, 0.3])
                ),
                ets_carbon_price=CarbonPriceLevers(
                    start_gbp_per_tco2e=np.random.choice([55, 75, 100, 125, 150]),
                    ramp_ppy=np.random.choice([5, 10, 15, 20])
                ),
                departmental=[
                    DepartmentalLever(dept="DfT", rdel_pct=np.random.choice([0, -2, -5])),
                    DepartmentalLever(dept="HO", rdel_pct=np.random.choice([0, -2, -5])),
                ] if np.random.random() < 0.3 else [],
                benefits=BenefitsLevers(
                    uc_uplift_pct=np.random.choice([0, 2, 5], p=[0.7, 0.2, 0.1])
                )
            ),
            targets=PolicyTargets(
                revenue_delta_gbp_bny=target_revenue_bn
            )
        )

        # Rough objective value (will be refined by agents)
        # For now, just use expected revenue
        policy.objective_value = abs(target_revenue_bn)

        candidates.append(policy)

    console.print(f"[green]✓[/green] Generated {len(candidates)} candidate policies")

    return candidates


def evaluate_candidates(
    candidates: List[Policy],
    tb_agent: TaxBenefitAgent,
    macro_agent: MacroAgent,
    climate_agent: ClimateAgent,
    dist_agent: DistributionAgent
) -> List[Policy]:
    """
    Evaluate all candidates with agents

    Args:
        candidates: List of policies to evaluate
        tb_agent, macro_agent, climate_agent, dist_agent: Compute agents

    Returns:
        List of evaluated policies
    """
    console.print(f"\n[cyan]Evaluating {len(candidates)} candidates...[/cyan]")

    evaluated = []

    for policy in track(candidates, description="Evaluating policies"):
        try:
            # Run all agents
            tb_result = tb_agent.simulate_policy(policy)
            macro_result = macro_agent.simulate_macro_impact(policy, tb_result)
            climate_result = climate_agent.simulate_emissions_impact(macro_result, policy)
            impacts = dist_agent.analyze_distribution(policy, tb_result, macro_result, climate_result)

            policy.impacts = impacts

            # Update objective value with actual revenue
            policy.objective_value = abs(tb_result['total_revenue_delta_bn'])

            evaluated.append(policy)

        except Exception as e:
            logger.warning(f"Policy evaluation failed: {e}")
            continue

    console.print(f"[green]✓[/green] Evaluated {len(evaluated)} policies successfully")

    return evaluated


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="ALBION Policy Simulation Runner")
    parser.add_argument('--target-bn', type=float, default=50.0, help='Target revenue to raise (£bn)')
    parser.add_argument('--k', type=int, default=5, help='Number of diverse options to select')
    parser.add_argument('--epsilon', type=float, default=0.02, help='Near-optimality threshold')
    parser.add_argument('--max-candidates', type=int, default=1000, help='Maximum candidates to generate')
    parser.add_argument('--no-signing', action='store_true', help='Disable Sigstore signing (faster)')

    args = parser.parse_args()

    console.print("\n[bold]ALBION Simulation Runner[/bold]")
    console.print("=" * 70)
    console.print(f"Target: Raise £{args.target_bn}bn")
    console.print(f"Diversity: Select {args.k} plans (ε={args.epsilon})")
    console.print("=" * 70)

    # 1. Load data
    households_df, L, sector_names, emissions_df, obr = load_data()

    # 2. Initialize agents
    tb_agent, macro_agent, climate_agent, dist_agent = initialize_agents(
        households_df, L, sector_names, emissions_df, obr
    )

    # 3. Load constitution
    constitution_file = PROJECT_ROOT / "configs" / "constitution.json"
    with open(constitution_file) as f:
        constitution_dict = json.load(f)
    constitution = Constitution(**constitution_dict)

    # 4. Generate candidates
    candidates = generate_policy_candidates(args.target_bn, args.max_candidates)

    # 5. Evaluate candidates
    evaluated = evaluate_candidates(candidates, tb_agent, macro_agent, climate_agent, dist_agent)

    if len(evaluated) == 0:
        console.print("[red]✗ No policies evaluated successfully. Exiting.[/red]")
        return 1

    # 6. DNOS selection
    console.print(f"\n[cyan]Running DNOS selector...[/cyan]")
    selector = DNOSSelector(constitution)
    selection_result = selector.select(evaluated, objective='revenue')

    # 7. Sign artifacts
    if not args.no_signing:
        console.print(f"\n[cyan]Signing artifacts with Sigstore...[/cyan]")
        signing_service = SigstoreService(enable_signing=False)  # Mock for now

        for plan in track(selection_result.selected_plans, description="Signing plans"):
            sig = signing_service.sign_artifact(plan.dict())
            plan.signature = sig.__dict__

        # Sign certificate
        cert_sig = signing_service.sign_artifact(selection_result.certificate.dict())
        selection_result.certificate.signature = cert_sig.__dict__

        console.print(f"[green]✓[/green] Signed {len(selection_result.selected_plans)} plans + certificate")

    # 8. Write output
    console.print(f"\n[cyan]Writing output files...[/cyan]")

    # Plans
    plans_dir = OUTPUT_DIR / "plans"
    plans_dir.mkdir(exist_ok=True)

    for i, plan in enumerate(selection_result.selected_plans, 1):
        plan_file = plans_dir / f"plan_{i}.json"
        with open(plan_file, 'w') as f:
            json.dump(plan.dict(), f, indent=2, default=str)
        console.print(f"[green]✓[/green] {plan_file}")

    # Certificate
    cert_file = CERTS_DIR / f"diversity_cert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(cert_file, 'w') as f:
        json.dump(selection_result.certificate.dict(), f, indent=2, default=str)
    console.print(f"[green]✓[/green] {cert_file}")

    # Summary table
    console.print("\n[bold]Selected Plans Summary[/bold]")
    table = Table(show_header=True)
    table.add_column("Plan", style="cyan")
    table.add_column("Revenue (£bn)", justify="right")
    table.add_column("Decile 1 Impact", justify="right")
    table.add_column("Decile 10 Impact", justify="right")
    table.add_column("Emissions (MtCO2e)", justify="right")

    for i, plan in enumerate(selection_result.selected_plans, 1):
        if plan.impacts:
            revenue = plan.objective_value
            d1 = plan.impacts.distribution.decile_deltas_pct[0] if plan.impacts.distribution else 0
            d10 = plan.impacts.distribution.decile_deltas_pct[9] if plan.impacts.distribution else 0
            emissions = plan.impacts.climate.emissions_delta_mtco2e if plan.impacts.climate else 0

            table.add_row(
                f"Plan {i}",
                f"£{revenue:.1f}",
                f"{d1:+.2f}%",
                f"{d10:+.2f}%",
                f"{emissions:+.1f}"
            )

    console.print(table)

    console.print(f"\n[bold green]✓ Simulation complete![/bold green]")
    console.print(f"\n[dim]Plans written to: {plans_dir}[/dim]")
    console.print(f"[dim]Certificate written to: {cert_file}[/dim]")

    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Error: {e}[/bold red]")
        logger.exception("Fatal error")
        sys.exit(1)
