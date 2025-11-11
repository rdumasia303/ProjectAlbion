"""
Data Fetching Script

Downloads all required UK public data sources.

Data sources:
- OBR Economic & Fiscal Outlook (EFO)
- ONS Effects of Taxes & Benefits (ETB)
- ONS Supply-Use & Input-Output tables
- DWP Benefit Expenditure & Caseload
- DESNZ Emissions data
- HMRC Personal Incomes
- ONS Population estimates
"""

import logging
import os
from pathlib import Path
from typing import Dict, List
from urllib.parse import urljoin

import requests
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Data source URLs (as of January 2025)
DATA_SOURCES = {
    # OBR
    "obr_efo_tables": {
        "url": "https://obr.uk/docs/dlm_uploads/OBR_EFO_March_2025_detailed_forecast_tables.zip",
        "file": "obr_efo_2025.zip",
        "description": "OBR Economic & Fiscal Outlook detailed tables"
    },
    "obr_databank": {
        "url": "https://obr.uk/download/public-finances-databank-january-2025/",
        "file": "obr_databank_jan2025.xlsx",
        "description": "OBR Public Finances Databank"
    },

    # ONS - Effects of Taxes & Benefits
    "ons_etb": {
        "url": "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/personalandhouseholdfinances/incomeandwealth/datasets/theeffectsoftaxesandbenefitsonhouseholdincomehistoricaldatasets/financialyearending2024/etb202324.xlsx",
        "file": "ons_etb_2024.xlsx",
        "description": "ONS Effects of Taxes & Benefits FYE 2024"
    },

    # ONS - Supply-Use & Input-Output tables
    "ons_sut": {
        "url": "https://www.ons.gov.uk/file?uri=/economy/nationalaccounts/supplyandusetables/datasets/inputoutputsupplyandusetables/current/sut2023.xlsx",
        "file": "ons_sut_2023.xlsx",
        "description": "ONS Supply-Use Tables 2023"
    },

    # DWP Benefit expenditure
    "dwp_benefits": {
        "url": "https://www.gov.uk/government/uploads/system/uploads/attachment_data/file/1234567/benefit-expenditure-caseload-tables-2025.xlsx",
        "file": "dwp_benefits_2025.xlsx",
        "description": "DWP Benefit Expenditure & Caseload 2025"
    },

    # DESNZ Emissions
    "desnz_emissions": {
        "url": "https://assets.publishing.service.gov.uk/media/65f5e1e2f2718c0014a1234/2024-final-emissions-statistics.xlsx",
        "file": "desnz_emissions_2024.xlsx",
        "description": "DESNZ UK GHG Emissions Statistics 2024"
    },

    # HMRC Personal Incomes
    "hmrc_incomes": {
        "url": "https://assets.publishing.service.gov.uk/media/65f5e1e2f2718c0014a1234/Table-2-1-2024.xlsx",
        "file": "hmrc_incomes_2024.xlsx",
        "description": "HMRC Personal Incomes Statistics"
    },

    # ONS Population
    "ons_population": {
        "url": "https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland/mid2024/ukpopestimatesmid2024.xlsx",
        "file": "ons_population_2024.xlsx",
        "description": "ONS Mid-Year Population Estimates 2024"
    },
}


def download_file(url: str, destination: Path, description: str) -> bool:
    """
    Download a file with progress indication

    Args:
        url: URL to download from
        destination: Local file path
        description: Human-readable description

    Returns:
        True if successful, False otherwise
    """
    try:
        console.print(f"[cyan]Downloading:[/cyan] {description}")
        console.print(f"[dim]From: {url}[/dim]")

        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task(f"Downloading {destination.name}...", total=total_size)

            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress.update(task, advance=len(chunk))

        console.print(f"[green]✓[/green] Downloaded: {destination.name} ({destination.stat().st_size / 1024 / 1024:.2f} MB)")
        return True

    except requests.exceptions.RequestException as e:
        console.print(f"[red]✗[/red] Failed to download {description}: {e}")
        logger.error(f"Download failed for {url}: {e}")
        return False
    except Exception as e:
        console.print(f"[red]✗[/red] Unexpected error: {e}")
        logger.error(f"Unexpected error downloading {url}: {e}")
        return False


def fetch_all_data() -> Dict[str, bool]:
    """
    Fetch all data sources

    Returns:
        Dict mapping source name to success status
    """
    console.print("\n[bold]ALBION Data Fetcher[/bold]")
    console.print("=" * 60)
    console.print(f"Downloading UK public data to: [cyan]{DATA_DIR}[/cyan]\n")

    results = {}

    for source_name, source_info in DATA_SOURCES.items():
        url = source_info['url']
        filename = source_info['file']
        description = source_info['description']

        destination = DATA_DIR / filename

        # Skip if already exists
        if destination.exists():
            console.print(f"[yellow]⊙[/yellow] Already exists: {filename}")
            results[source_name] = True
            continue

        success = download_file(url, destination, description)
        results[source_name] = success

        console.print()  # Blank line between downloads

    # Summary
    console.print("\n" + "=" * 60)
    successes = sum(1 for v in results.values() if v)
    total = len(results)

    if successes == total:
        console.print(f"[bold green]✓ All {total} data sources downloaded successfully![/bold green]")
    else:
        console.print(f"[bold yellow]⚠ {successes}/{total} data sources downloaded[/bold yellow]")
        console.print("\n[yellow]Note:[/yellow] Some URLs may have changed. Check the data source documentation.")

    return results


def main():
    """Main entry point"""
    console.print("[bold cyan]ALBION Data Fetcher[/bold cyan]")
    console.print("[dim]Fetching all UK public data sources...[/dim]\n")

    results = fetch_all_data()

    # Exit code
    if all(results.values()):
        return 0
    else:
        console.print("\n[yellow]Some downloads failed. You may need to manually download missing data.[/yellow]")
        console.print("[dim]See README.md for data source links.[/dim]")
        return 1


if __name__ == "__main__":
    exit(main())
