"""
Mock Data Generator

Generates realistic synthetic data for local testing/demo.

This allows the system to run end-to-end without downloading real UK data.

Generates:
- Synthetic household population (~100k cohorts)
- Leontief inverse matrix
- Emissions intensities
- OBR baselines
- MACC curves
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Directories
DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def generate_synthetic_households(n_cohorts: int = 100_000) -> pd.DataFrame:
    """
    Generate synthetic household population

    Creates realistic distributions based on ONS ETB patterns

    Args:
        n_cohorts: Number of household cohorts to generate

    Returns:
        DataFrame with household cohorts
    """
    console.print(f"[cyan]Generating {n_cohorts:,} synthetic household cohorts...[/cyan]")

    np.random.seed(42)  # Reproducible

    # Income deciles (equal probability)
    income_deciles = np.random.choice(range(1, 11), size=n_cohorts)

    # Household types
    household_types = np.random.choice(
        ['single_under_65', 'single_over_65', 'couple_no_children',
         'couple_with_children', 'lone_parent', 'other'],
        size=n_cohorts,
        p=[0.15, 0.12, 0.20, 0.25, 0.08, 0.20]
    )

    # Regions (ITL1)
    regions = np.random.choice(
        ['UKC', 'UKD', 'UKE', 'UKF', 'UKG', 'UKH',
         'UKI', 'UKJ', 'UKK', 'UKL', 'UKM', 'UKN'],
        size=n_cohorts,
        p=[0.031, 0.093, 0.071, 0.061, 0.078, 0.099,
           0.236, 0.147, 0.072, 0.032, 0.070, 0.021]
    )

    # Gross incomes (realistic distribution by decile)
    # Decile 1: £0-15k, Decile 10: £80k+
    decile_means = [10_000, 18_000, 24_000, 30_000, 36_000,
                    43_000, 51_000, 62_000, 78_000, 120_000]
    decile_stds = [3_000, 2_000, 2_500, 3_000, 3_500,
                   4_000, 5_000, 7_000, 10_000, 40_000]

    gross_incomes = np.array([
        max(0, np.random.normal(decile_means[d-1], decile_stds[d-1]))
        for d in income_deciles
    ])

    # Equivalised incomes (slightly lower than gross due to household composition)
    equivalised_incomes = gross_incomes * np.random.uniform(0.7, 0.95, size=n_cohorts)

    # Benefit flags (higher probability in lower deciles)
    uc_prob = 1.0 - (income_deciles - 1) / 9 * 0.9  # Decile 1: 100%, Decile 10: 10%
    pension_prob = np.where(
        (household_types == 'single_over_65') | (household_types == 'couple_no_children'),
        0.8, 0.1
    )

    benefit_flags = [
        {
            'uc': np.random.random() < uc_prob[i],
            'state_pension': np.random.random() < pension_prob[i],
            'disability_benefits': np.random.random() < 0.15,
            'pension_credit': np.random.random() < (0.1 if income_deciles[i] <= 3 else 0.01)
        }
        for i in range(n_cohorts)
    ]

    # Tax profiles (simplied - actual calculation done by TaxBenefitAgent)
    tax_profiles = [
        {
            'income_tax': gross_incomes[i] * 0.15 if gross_incomes[i] > 12_570 else 0,
            'nics': gross_incomes[i] * 0.10 if gross_incomes[i] > 12_570 else 0,
            'vat': gross_incomes[i] * 0.12  # Rough VAT burden
        }
        for i in range(n_cohorts)
    ]

    # Consumption baskets (VAT modeling)
    consumption_baskets = [
        {
            'standard_rated': gross_incomes[i] * 0.40,  # 40% on standard-rated goods
            'reduced_rated': gross_incomes[i] * 0.05,   # 5% on reduced-rated
            'zero_rated': gross_incomes[i] * 0.25       # 25% on zero-rated (food, etc.)
        }
        for i in range(n_cohorts)
    ]

    # Weights (how many real households this cohort represents)
    # Total UK households ~28 million
    total_households = 28_000_000
    weights = np.random.gamma(shape=2, scale=total_households / n_cohorts / 2, size=n_cohorts)
    weights = weights / weights.sum() * total_households  # Normalize to total

    # Build DataFrame
    df = pd.DataFrame({
        'cohort_id': range(n_cohorts),
        'income_decile': income_deciles,
        'household_type': household_types,
        'region': regions,
        'gross_income': gross_incomes,
        'equivalised_income': equivalised_incomes,
        'benefit_flags': benefit_flags,
        'tax_profile': tax_profiles,
        'consumption_basket': consumption_baskets,
        'weight': weights,
        'data_version': '2024_mock'
    })

    console.print(f"[green]✓[/green] Generated {len(df):,} household cohorts")
    console.print(f"[dim]Total weighted households: {df['weight'].sum():,.0f}[/dim]")

    return df


def generate_leontief_matrix(n_sectors: int = 105) -> tuple:
    """
    Generate synthetic Leontief inverse matrix

    Based on realistic I/O structure

    Args:
        n_sectors: Number of sectors (ONS has 105)

    Returns:
        (leontief_matrix, sector_names)
    """
    console.print(f"[cyan]Generating Leontief matrix ({n_sectors} sectors)...[/cyan]")

    np.random.seed(42)

    # Sector names (simplified for demo)
    sector_names = [
        'Agriculture', 'Mining', 'Manufacturing', 'Electricity',
        'Water', 'Construction', 'Retail', 'Transport',
        'Accommodation', 'Information', 'Financial services',
        'Real estate', 'Professional services', 'Admin services',
        'Public admin & defence', 'Education', 'Health & social work',
        'Arts & recreation', 'Other services'
    ] + [f'Sector_{i}' for i in range(20, n_sectors)]

    # Generate realistic technical coefficients matrix A
    # Diagonal dominance (sectors use their own output)
    A = np.random.uniform(0, 0.2, size=(n_sectors, n_sectors))
    np.fill_diagonal(A, np.random.uniform(0.3, 0.6, size=n_sectors))

    # Ensure A is economically viable (spectral radius < 1)
    eigenvalues = np.linalg.eigvals(A)
    max_eigenvalue = np.max(np.abs(eigenvalues))

    if max_eigenvalue >= 1:
        A = A / (max_eigenvalue * 1.1)  # Scale down

    # Compute Leontief inverse L = (I - A)^-1
    I = np.eye(n_sectors)
    L = np.linalg.inv(I - A)

    console.print(f"[green]✓[/green] Generated Leontief matrix")
    console.print(f"[dim]Spectral radius of A: {max_eigenvalue:.3f}[/dim]")

    return L, sector_names


def generate_emissions_intensities(sector_names: list) -> pd.DataFrame:
    """
    Generate emissions intensities by sector

    Args:
        sector_names: List of sector names

    Returns:
        DataFrame with sector emissions data
    """
    console.print(f"[cyan]Generating emissions intensities for {len(sector_names)} sectors...[/cyan]")

    np.random.seed(42)

    # Realistic emissions intensities (tCO2e per £million GVA)
    # Higher for energy/industry, lower for services
    intensities = []
    gvas = []

    for sector in sector_names:
        if 'Electricity' in sector or 'Mining' in sector:
            intensity = np.random.uniform(200, 500)
            gva = np.random.uniform(5_000, 20_000)
        elif 'Manufacturing' in sector or 'Transport' in sector:
            intensity = np.random.uniform(50, 200)
            gva = np.random.uniform(10_000, 50_000)
        elif 'Agriculture' in sector:
            intensity = np.random.uniform(100, 250)
            gva = np.random.uniform(8_000, 25_000)
        else:  # Services
            intensity = np.random.uniform(10, 80)
            gva = np.random.uniform(15_000, 100_000)

        intensities.append(intensity)
        gvas.append(gva)

    emissions = np.array(intensities) * np.array(gvas) / 1_000_000  # MtCO2e

    df = pd.DataFrame({
        'sector_code': [f'S{i:03d}' for i in range(len(sector_names))],
        'sector_name': sector_names,
        'emissions_tco2e': emissions,
        'gva_million_gbp': gvas,
        'intensity': intensities,
        'year': 2024
    })

    console.print(f"[green]✓[/green] Generated emissions intensities")
    console.print(f"[dim]Total emissions: {df['emissions_tco2e'].sum():.1f} MtCO2e[/dim]")

    return df


def generate_obr_baselines() -> dict:
    """
    Generate OBR baseline forecasts

    Returns:
        Dict with baseline data
    """
    console.print(f"[cyan]Generating OBR baseline forecasts...[/cyan]")

    baselines = {
        'gdp_bn_gbp': 2800,  # £2.8 trillion
        'employment_thousands': 33_000,
        'debt_ratio_pct': [95.0, 94.5, 93.8, 92.5, 91.0],  # Years 1-5
        'deficit_bn_gbp': [80, 75, 70, 65, 60],
        'receipts_bn_gbp': [1050, 1080, 1110, 1145, 1180],
        'spending_bn_gbp': [1130, 1155, 1180, 1210, 1240],
    }

    console.print(f"[green]✓[/green] Generated OBR baselines")
    console.print(f"[dim]Baseline GDP: £{baselines['gdp_bn_gbp']:.0f}bn[/dim]")

    return baselines


def save_all_data(households: pd.DataFrame, leontief: tuple, emissions: pd.DataFrame, obr: dict):
    """Save all generated data"""
    console.print("\n[cyan]Saving generated data...[/cyan]")

    # Households
    households_file = PROCESSED_DIR / "households.parquet"
    households.to_parquet(households_file, index=False)
    console.print(f"[green]✓[/green] Saved: {households_file}")

    # Leontief matrix
    L, sector_names = leontief
    leontief_file = PROCESSED_DIR / "leontief_matrix.npz"
    np.savez(leontief_file, L=L, sector_names=sector_names)
    console.print(f"[green]✓[/green] Saved: {leontief_file}")

    # Emissions
    emissions_file = PROCESSED_DIR / "emissions_intensities.parquet"
    emissions.to_parquet(emissions_file, index=False)
    console.print(f"[green]✓[/green] Saved: {emissions_file}")

    # OBR baselines
    obr_file = PROCESSED_DIR / "obr_baselines.json"
    with open(obr_file, 'w') as f:
        json.dump(obr, f, indent=2)
    console.print(f"[green]✓[/green] Saved: {obr_file}")


def main():
    """Main entry point"""
    console.print("\n[bold]ALBION Mock Data Generator[/bold]")
    console.print("=" * 60)
    console.print("Generating realistic synthetic data for local demo\n")

    # Generate all data
    households = generate_synthetic_households(n_cohorts=100_000)
    leontief = generate_leontief_matrix(n_sectors=105)
    emissions = generate_emissions_intensities(leontief[1])
    obr = generate_obr_baselines()

    # Save
    save_all_data(households, leontief, emissions, obr)

    console.print("\n[bold green]✓ Mock data generation complete![/bold green]")
    console.print(f"\n[dim]Data saved to: {PROCESSED_DIR}[/dim]")
    console.print("\n[yellow]Note:[/yellow] This is synthetic data for demonstration purposes.")
    console.print("[dim]For production use, run 'make data' to fetch real UK data.[/dim]")


if __name__ == "__main__":
    main()
