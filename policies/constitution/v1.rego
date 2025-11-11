# ALBION Constitution Policy Gate v1
#
# Enforces all constitutional constraints:
# - Fiscal rules (OBR targets)
# - Carbon budgets (legal limits)
# - Devolution (reserved vs devolved)
# - Equality (PSED requirements)

package constitution.v1

import future.keywords.contains
import future.keywords.if
import future.keywords.in

# Default decision: deny unless explicitly allowed
default allow := false
default decision := {"allow": false, "reasons": ["No matching allow rule"]}

# ============================================================================
# Schema Validation
# ============================================================================

deny contains msg if {
    input.schema != "proposal/v1"
    msg := "Invalid schema: must be 'proposal/v1'"
}

deny contains msg if {
    not input.levers
    msg := "Missing required field: levers"
}

# ============================================================================
# Fiscal Rules Gate
# ============================================================================

# Debt must be falling by year 5 (OBR fiscal rule)
deny contains msg if {
    data.constitution.fiscal_rules.debt_rule == "falling_by_year_5"
    input.impacts.macro.debt_ratio_year5_pct >= data.constitution.baseline_debt_year1_pct
    msg := sprintf(
        "Fiscal rule breach: Debt ratio not falling (Year 5: %.1f%% vs Year 1: %.1f%%)",
        [input.impacts.macro.debt_ratio_year5_pct, data.constitution.baseline_debt_year1_pct]
    )
}

# Borrowing must stay within envelope
deny contains msg if {
    data.constitution.fiscal_rules.borrowing_envelope_bn_gbp
    annual_borrowing := -input.impacts.macro.gdp_delta_bn_gbp / 5  # Simplified
    annual_borrowing > data.constitution.fiscal_rules.borrowing_envelope_bn_gbp
    msg := sprintf(
        "Fiscal rule breach: Borrowing (£%.1fbn) exceeds envelope (£%.1fbn)",
        [annual_borrowing, data.constitution.fiscal_rules.borrowing_envelope_bn_gbp]
    )
}

# Warn if headroom is tight
warn contains msg if {
    data.constitution.fiscal_rules.headroom_required_bn_gbp
    input.impacts.macro.debt_ratio_year5_pct
    # Simplified headroom check
    headroom := 100 - input.impacts.macro.debt_ratio_year5_pct
    headroom < 5
    msg := sprintf(
        "Fiscal warning: Tight headroom (%.1f%% of GDP)",
        [headroom]
    )
}

# ============================================================================
# Carbon Budget Gate
# ============================================================================

# Sixth Carbon Budget compliance (965 MtCO2e over 2033-2037)
deny contains msg if {
    input.impacts.climate.budget_compliance.status == "BREACH"
    overshoot := -input.impacts.climate.budget_compliance.headroom_or_overshoot_mtco2e
    msg := sprintf(
        "Carbon Budget breach: Sixth Budget exceeded by %.1f MtCO2e",
        [overshoot]
    )
}

warn contains msg if {
    input.impacts.climate.budget_compliance.status == "TIGHT"
    headroom := input.impacts.climate.budget_compliance.headroom_or_overshoot_mtco2e
    msg := sprintf(
        "Carbon Budget warning: Low headroom (%.1f MtCO2e remaining)",
        [headroom]
    )
}

# Carbon price cannot be negative
deny contains msg if {
    input.levers.ets_carbon_price.start_gbp_per_tco2e < 0
    msg := "Invalid carbon price: cannot be negative"
}

# ============================================================================
# Devolution Gate
# ============================================================================

# Income tax rates in Scotland are devolved - UK-wide changes not allowed
deny contains msg if {
    data.constitution.devolution.scotland.income_tax_rates == "devolved"
    input.jurisdiction == "UK"

    # Check if income tax levers are used
    some_it_lever_used

    msg := "Devolution breach: Income tax rates are devolved to Scotland; UK-wide changes not permitted"
}

some_it_lever_used if {
    input.levers.income_tax.basic_rate_pp != 0
}

some_it_lever_used if {
    input.levers.income_tax.higher_rate_pp != 0
}

some_it_lever_used if {
    input.levers.income_tax.additional_rate_pp != 0
}

# Corporation tax in Northern Ireland is devolved
deny contains msg if {
    data.constitution.devolution.northern_ireland.corporation_tax == "devolved"
    input.jurisdiction == "UK"
    input.levers.corp_tax.main_rate_pp != 0
    msg := "Devolution breach: Corporation tax is devolved to Northern Ireland"
}

# VAT, NICs must be UK-wide
deny contains msg if {
    input.jurisdiction != "UK"
    input.jurisdiction != "England"  # England = de facto UK-wide for reserved matters
    input.levers.vat.standard_rate_pp != 0
    msg := "Devolution error: VAT is a UK-wide (reserved) matter"
}

# Barnett consequentials required for DEL changes
warn contains msg if {
    data.constitution.devolution.barnett_formula.enabled == true
    count([d | d := input.levers.departmental[_]; d.rdel_pct != 0]) > 0
    msg := "Barnett formula: DEL changes will trigger consequentials for Scotland, Wales, NI"
}

# ============================================================================
# Equality (PSED) Gate
# ============================================================================

# Public Sector Equality Duty: Warn if protected groups disproportionately affected
warn contains msg if {
    data.constitution.equality.psed.enabled == true
    warn_ratio := data.constitution.equality.psed.warn_ratio

    # Calculate general population impact
    general_impact := avg_decile_impact(input.impacts.distribution.decile_deltas_pct)

    # Check each protected group
    some group, impact in input.impacts.protected_groups
    abs(impact) > abs(general_impact) * warn_ratio

    msg := sprintf(
        "EqIA required: '%s' group disproportionately affected (%.1f%% vs %.1f%% general)",
        [group, impact, general_impact]
    )
}

# Helper: calculate average impact across deciles
avg_decile_impact(deltas) := avg if {
    sum_deltas := sum(deltas)
    count_deltas := count(deltas)
    avg := sum_deltas / count_deltas
}

# Specific EqIA requirements for benefit changes
warn contains msg if {
    "benefits" in data.constitution.equality.require_eqia_for
    input.levers.benefits.uc_uplift_pct != 0
    msg := "EqIA required: Universal Credit changes affect protected groups"
}

# ============================================================================
# Sanity Checks & Bounds
# ============================================================================

# Rate changes must be reasonable
deny contains msg if {
    input.levers.income_tax.basic_rate_pp > 10
    msg := "Invalid: Basic rate increase >10pp is unrealistic"
}

deny contains msg if {
    input.levers.income_tax.basic_rate_pp < -10
    msg := "Invalid: Basic rate decrease >10pp is unrealistic"
}

deny contains msg if {
    input.levers.vat.standard_rate_pp > 10
    msg := "Invalid: VAT increase >10pp is unrealistic"
}

# Cannot both raise taxes massively and cut spending massively
# (This is a policy coherence check, not a legal constraint)
warn contains msg if {
    total_tax_increase_pp := (
        input.levers.income_tax.basic_rate_pp +
        input.levers.nics.class1_main_pp +
        input.levers.vat.standard_rate_pp
    )

    total_spending_cuts := sum([
        d.rdel_pct |
        d := input.levers.departmental[_]
        d.rdel_pct < 0
    ])

    total_tax_increase_pp > 5
    total_spending_cuts < -20

    msg := "Policy coherence warning: Large tax rises + large spending cuts may be politically difficult"
}

# ============================================================================
# Decision Logic
# ============================================================================

# Allow if no deny rules triggered
allow if {
    count(deny) == 0
}

# Construct decision object
decision := result if {
    allow
    result := {
        "allow": true,
        "warnings": warn,
        "denials": [],
        "timestamp": time.now_ns()
    }
}

decision := result if {
    not allow
    result := {
        "allow": false,
        "warnings": warn,
        "denials": deny,
        "timestamp": time.now_ns()
    }
}

# ============================================================================
# Test Data Helpers (for unit testing)
# ============================================================================

# Expected constitution data structure
constitution_schema := {
    "fiscal_rules": {
        "debt_rule": "falling_by_year_5",
        "borrowing_envelope_bn_gbp": 50
    },
    "baseline_debt_year1_pct": 95,
    "carbon_budgets": {
        "sixth": {
            "total_mtco2e": 965
        }
    },
    "devolution": {
        "scotland": {
            "income_tax_rates": "devolved"
        },
        "northern_ireland": {
            "corporation_tax": "devolved"
        },
        "barnett_formula": {
            "enabled": true
        }
    },
    "equality": {
        "psed": {
            "enabled": true,
            "warn_ratio": 1.5
        },
        "require_eqia_for": ["benefits", "healthcare", "education"]
    }
}
