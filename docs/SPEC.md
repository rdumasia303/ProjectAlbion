# ALBION: The National Options Engine

A buildable platform for lawful, diverse, near-optimal national policy menus — with public receipts
*("If it isn't explorable and explainable, it isn't allowed.")*

## 0) Ground rules

1. **Humans decide.** Engine proposes; Cabinet/Parliament disposes.
2. **Law before loss.** Statutory/fiscal/climate/devolution constraints are hard gates. Plans that breach them don't render.
3. **Few good options.** Always output k (3–7) diverse, near-optimal plans with explicit guarantees and a "why these?" certificate.
4. **Cohorts, not people.** No PII; cohort modelling + published methods; DP if/when microdata required.
5. **Receipts or retract.** Every artefact is signed, hashed, and logged to a public transparency ledger.
6. **Portability.** Constitutions/policies/schemas are versioned, open, and vendor-portable.

## 1) What it is (and isn't)

Albion manufactures options, not answers. It's a lawful choice architect:

- turns fiscal rules, carbon budgets, and devolved/reserved competences into executable policy gates;
- enumerates near-optimal candidates across taxes/benefits/departmental spend/green levies;
- proves diversity across value-axes (distributional, regional, climate, system dynamics);
- ships an audit trail any journalist, MP, or civil servant can verify.

It does not replace ministers, Parliament, or the OBR. It forces trade-offs into sunlight.

## 2) Big-picture architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ HUMAN LAYER                                                                 │
│  Web app (Next.js) · "People's Budget" explorer · Consultation/Quorum app   │
└────────────▲────────────────────────────────────────────────────────────────┘
             │ mTLS
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ POLICY LAYER (Enforcement)                                                  │
│  Gatekeeper API · Open Policy Agent (Rego) · Devolution & EqIA gates        │
│  Fiscal-Rule Gate · Carbon-Budget Gate · Kill-switch (TLA+ proven)          │
└────────────▲────────────────────────────────────────────────────────────────┘
             │ gRPC + protobuf
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ COMPUTE LAYER (Agents + Selector)                                           │
│  TaxBenefitAgent     (rate/band/threshold levers, UC uplifts, NICs)         │
│  MacroAgent          (elasticities + SUT multipliers; OBR paths)            │
│  ClimateAgent        (ETS/carbon price scenarios vs. legal budgets)         │
│  DistributionAgent   (decile/region/cohort impacts; EqIA flags)             │
│  DNOS Selector       (Diverse Near-Optimal Set: FL + DPP with quotas)       │
└────────────▲────────────────────────────────────────────────────────────────┘
             │ Append-only I/O
             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ DATA & LOG LAYER                                                            │
│  Postgres (append-only tables) · Parquet lake · Sigstore/Rekor public log   │
│  "Constitution" registry (versioned params & gates)                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **OPA/Rego** is the policy brain; we keep constraints declarative and testable.
- **Rekor/Sigstore** signs and time-stamps every option & certificate.

## 3) The "constitution" at national scale

### 3.1 Fiscal targets
Parameterised from the latest OBR/EFO targets (debt path, borrowing envelopes). Plans must not breach the live rule set; rules are versioned in the constitution package.

### 3.2 Carbon budgets
Legal caps from The Carbon Budget Orders (e.g., Sixth Carbon Budget = 965 MtCO₂e over 2033-37). ClimateAgent checks scenario consistency with the Carbon Budget & Growth Delivery Plan baselines.

### 3.3 Devolution
Reserved vs. devolved matters (e.g., Income Tax in Scotland bands; Barnett consequentials for UK-wide DEL moves). Constitution ships competent levers per nation, with Barnett logic sourced from HMT/PESA and Statement of Funding Policy.

### 3.4 Equality
EqIA prompts when protected-group proxies show disproportionate impact; duty references Equality Act 2010 and PSED guidance. (We use cohort outputs only; no PII.)

## 4) Data sources (all free)

### Core fiscal & macro
- **OBR Economic & Fiscal Outlook (EFO)** tables & databank – receipts, spend, borrowing, headroom, ready-reckoners
- **PESA 2025** & Public Spending releases – departmental spend, DEL/CDEL/RDEL history & plans
- **DWP Benefit Expenditure & Caseload 2025** – historic & forecast benefits by instrument

### Distributional/household
- **ONS "Effects of taxes & benefits" (FYE 2024)** – income/benefit/tax by decile & household type
- **HMRC Personal Incomes & Income Tax distributions** – taxpayer counts by ranges, sources

### Climate
- **DESNZ UK greenhouse-gas emissions statistics** (national & by sector); Carbon Budget Delivery Plan tech annex for baselines

### Regional
- **ONS Supply-Use & Input-Output tables (SUT/IO)** – sector multipliers (Type I employment etc.), Blue Book-consistent
- **Open Geography Portal** – Parliamentary constituencies 2024; ITL (ex-NUTS) regions & lookups

### Population
- **ONS mid-year population estimates (mid-2024)** – national & sub-national denominators

## 5) Schemas (portable & versioned)

### 5.1 proposal.v1.json

```json
{
  "schema": "proposal/v1",
  "jurisdiction": "UK",
  "horizon_years": 5,
  "levers": {
    "income_tax": {"basic_rate_pp": 1.0, "higher_rate_pp": 0.0, "add_rate_pp": 0.0},
    "nics": {"class1_main_pp": 0.5, "threshold_shift_gbp": 0},
    "vat": {"standard_rate_pp": 1.0, "exemption_changes": []},
    "corp_tax": {"main_rate_pp": 1.0},
    "cgt": {"align_to_income_tax": true},
    "ets_carbon_price": {"start_gbp_per_tCO2e": 55, "ramp_ppy": 10},
    "departmental": [{"dept":"HO","rdel_pct":-2.0},{"dept":"DfT","rdel_pct":-2.0}],
    "benefits": {"uc_uplift_pct": 0.0}
  },
  "targets": {"revenue_delta_gbp_bny": 50},
  "impacts": {
    "distribution": {"decile_deltas_pct":[...]},
    "regional": {"itl1":{"UKC":[...], "UKD":[...]}}
  },
  "meta": {"created_at":"RFC3339","agent":"Albion@1.0","notes":["..."]}
}
```

### 5.2 constitution.v1.json

```json
{
  "version":"constitution/v1",
  "fiscal_rules": {"source":"OBR/EFO/2025-03","debt_rule":"param", "borrowing_rule":"param"},
  "carbon_budgets": {"sixth": {"total_mtco2e": 965}},
  "devolution": {"scotland":{"income_tax_bands":"devolved"}, "ukwide":["VAT","NICs","CorpTax"]},
  "equality": {"psed":{"enabled": true, "warn_ratio": 1.5}},
  "diversity": {
    "k":5,
    "epsilon_additive":0.02,
    "weights":{"dist":0.35,"regional":0.25,"climate":0.25,"system":0.15},
    "quotas":{"low_income_advantaged":1,"north_east_prioritised":1}
  }
}
```

## 6) Policy gates (Rego)

```rego
package constitution.v1

default decision := "allow"

deny[msg] { input.schema != "proposal/v1"; msg := "Bad schema" }

# Carbon budget check
deny[msg] {
  input.levers.ets_carbon_price.start_gbp_per_tCO2e < 0
  msg := "Negative carbon price not allowed"
}

# EqIA prompt: group disproportionality
warn[msg] {
  r := data.params.eq_warn_ratio
  g := input.impacts.distribution.general
  some k
  input.impacts.distribution.by_group[k] > g * r
  msg := "EqIA sign-off required"
}
```

## 7) Compute layer

### TaxBenefitAgent
Parameterises tax-benefit levers (rates, bands, thresholds, UC uprates). Uses HMRC distributions + ONS ETB deciles to approximate static yields & incidence.

### MacroAgent
Applies lightweight elasticities & multipliers anchored to OBR/EFO baselines; supports sensitivity toggles (low/central/high). For place-based views, folds through SUT/IO multipliers.

### ClimateAgent
Computes revenue from carbon pricing paths; checks legal consistency against carbon budgets and the current Delivery Plan baseline.

### DistributionAgent
Maps households by decile & type to show direct+indirect effects (inc. VAT shifts), plus regional slices via ITL mapping.

### DNOS Selector
Selects k near-optimal plans using Facility Location + DPP with representation quotas. Emits a signed Diversity Certificate explaining "why these five?".

## 8) Security, integrity, audit

- **Hash → Sign → Log** every plan & certificate using Sigstore/Rekor; verify on read
- **Append-only Postgres** with hashing triggers; no UPDATE/DELETE on artefacts
- **Kill-switch**: 2-of-N quorum freeze (TLA+ model checked) if gates/feeds drift

## 9) The UI

- Five cards side-by-side with mini-sparklines for receipts, debt, emissions
- "Impact on a typical household": decile waterfall
- Regional map (constituency & ITL toggles)
- Gates drawer: shows pass/warn/consult/block with rationale
- Diversity dial (read-only for public; editable for authorised users)

## 10) Deployment

- Containers per service; non-root; read-only FS
- OPA runs sidecar or central with bundle updates
- Signing: cosign sign --recursive to Rekor; verify on read
- SLOs: Gate P99 < 50ms; end-to-end P95 < 2s; Rekor backlog age < 15m

## 11) Testing

- Unit tests for policy packs
- Property tests for DNOS diminishing returns; quota enforcement
- Backtests: replay past fiscal events with actuals from OBR Databank & PESA
- Red-team: try illegal lever combos

## Key References

- OBR Economic & Fiscal Outlook
- PESA (Public Expenditure Statistical Analyses)
- ONS Effects of Taxes & Benefits
- DESNZ Carbon Budget & Growth Delivery Plan
- ONS Supply-Use & Input-Output Tables
- Open Policy Agent documentation
- Sigstore/Rekor documentation
