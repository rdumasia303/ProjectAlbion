# ALBION: The National Options Engine

**A production-grade platform for lawful, diverse, near-optimal national policy simulation with public receipts.**

> *"If it isn't explorable and explainable, it isn't allowed."*

---

## What is ALBION?

ALBION is a national-scale fiscal and policy simulation engine designed to generate **mathematically diverse, legally compliant, and publicly verifiable** policy options.

It's not a black box. It's not a recommendation system. It's a **constraint-based option generator** that:

1. **Enforces hard legal limits** (fiscal rules, carbon budgets, devolution, equality duties) via declarative policy gates
2. **Simulates precise impacts** via full microsimulation (100k+ household cohorts) and Input-Output economic modeling
3. **Proves diversity** using Facility Location + Determinantal Point Processes with representation quotas
4. **Signs everything** with Sigstore/Rekor for public cryptographic audit trails

### The Five Pillars

1. **Microsimulation**: Not approximations. Full tax-benefit calculations across synthetic population.
2. **I/O Modeling**: Leontief inverse matrices for sectoral spillover effects.
3. **Climate Integration**: Marginal Abatement Cost Curves (MACC) linked to economic activity.
4. **Diversity Certification**: Mathematical proof of "why these k plans?" using FL+DPP+Quotas.
5. **Cryptographic Receipts**: Every artifact signed and logged to public transparency ledger.

---

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PostgreSQL 15+
- (Optional) OPA CLI
- (Optional) cosign (Sigstore)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ProjectAlbion.git
cd ProjectAlbion

# Install dependencies
make deps

# Initialize database
make db-init

# Fetch and process UK public data
make data

# Run first simulation
make simulate
```

This will:
- Fetch ONS, OBR, DWP, DESNZ data
- Build synthetic population (~100k cohorts)
- Compute Leontief matrix from SUT/IO tables
- Generate 5 diverse, lawful policy options to raise £50bn
- Produce signed diversity certificates

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ HUMAN LAYER (Next.js)                                       │
│  Five-card option display · Regional maps · Receipts       │
└────────────▲────────────────────────────────────────────────┘
             │ HTTPS + mTLS
             ▼
┌─────────────────────────────────────────────────────────────┐
│ POLICY LAYER (OPA/Rego)                                    │
│  Fiscal gates · Carbon gates · Devolution · EqIA           │
└────────────▲────────────────────────────────────────────────┘
             │ gRPC
             ▼
┌─────────────────────────────────────────────────────────────┐
│ COMPUTE LAYER (Python)                                      │
│  TaxBenefitAgent · MacroAgent · ClimateAgent               │
│  DistributionAgent · DNOS Selector                          │
└────────────▲────────────────────────────────────────────────┘
             │ Async/Queue
             ▼
┌─────────────────────────────────────────────────────────────┐
│ DATA LAYER (Postgres + Parquet)                            │
│  Households · Sectors · Emissions · Append-only log        │
│  Sigstore signatures                                        │
└─────────────────────────────────────────────────────────────┘
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details.

---

## The DNOS Algorithm

**D**iverse **N**ear-**O**ptimal **S**et selection is the mathematical heart of ALBION.

Given thousands of candidate policies, DNOS selects exactly `k` (typically 5) that:

1. **Near-Optimal**: All within ε (2%) of the best solution
2. **Representative** (Facility Location): Cover the policy space
3. **Diverse** (DPP): Maximize kernel determinant (orthogonality)
4. **Quota-Compliant**: Guaranteed representation (e.g., "one must benefit low-income households")

### Mathematical Guarantees

- **FL approximation**: 2-approximation to optimal k-center
- **DPP sampling**: Probability ∝ det(K) ensures diversity
- **Quota satisfaction**: Hard constraints via partitioning

Every selection comes with a **signed diversity certificate** explaining "why these five?"

---

## Data Sources (All Public)

ALBION uses only free, publicly available UK data:

### Fiscal & Macro
- **OBR Economic & Fiscal Outlook**: Receipts, spending, borrowing baselines
- **PESA 2025**: Departmental expenditure (RDEL/CDEL)
- **DWP Benefit Expenditure & Caseload**: Benefit spending & claimant counts

### Distributional
- **ONS Effects of Taxes & Benefits**: Income/tax/benefit by decile
- **HMRC Personal Incomes**: Taxpayer distributions

### Climate
- **DESNZ GHG Statistics**: Emissions by sector
- **Carbon Budget Delivery Plan**: Legal limits and tech annex

### Regional
- **ONS Supply-Use & Input-Output Tables**: Sector multipliers
- **Open Geography Portal**: Constituencies, ITL regions

### Population
- **ONS Mid-Year Estimates**: National/sub-national denominators

All sources documented in [docs/SPEC.md](docs/SPEC.md#4-data-sources-all-free).

---

## Components

### 1. Compute Agents

#### TaxBenefitAgent
Full microsimulation across ~100k household cohorts.

- Precise Income Tax (with personal allowance taper)
- NICs (Class 1, 4)
- VAT (standard/reduced rates)
- Universal Credit (with taper)
- State Pension

No ready-reckoners. No approximations. Every household cohort calculated.

#### MacroAgent
Input-Output economic modeling.

- Leontief inverse matrix from ONS SUT/IO tables
- OBR tax elasticities
- Department → sector mapping
- Regional disaggregation (ITL1)

#### ClimateAgent
Integrated assessment modeling.

- Emissions intensities by sector
- Marginal Abatement Cost Curves (MACC)
- Carbon Budget compliance checking (Sixth Budget: 965 MtCO2e)
- Carbon price revenue calculation

#### DistributionAgent
Distributional analysis.

- Decile impacts (% of income)
- Gini coefficient changes
- Poverty rate changes
- Protected group analysis (EqIA proxies)

### 2. DNOS Selector

Located in `albion/dnos/selector.py`.

Implements:
- ε-near-optimal filtering
- Greedy Facility Location (Gonzalez algorithm)
- Determinantal Point Process (greedy MAP inference)
- Quota-constrained partitioning

Output: k diverse plans + signed certificate

### 3. Policy Gates (OPA/Rego)

Located in `policies/constitution/v1.rego`.

Enforces:
- **Fiscal rules**: Debt falling by year 5, borrowing within envelope
- **Carbon budgets**: Sixth Carbon Budget (965 MtCO2e 2033-37)
- **Devolution**: Scotland (income tax), NI (corp tax), Barnett consequentials
- **Equality**: PSED warnings for disproportionate impact

Decisions: `allow`, `warn`, `deny`

### 4. Sigstore Integration

Located in `albion/sign/sigstore_service.py`.

Every artifact:
1. Canonicalized (deterministic JSON)
2. Hashed (SHA-256)
3. Signed (cosign)
4. Logged to Rekor (public transparency log)

Verification: Hash + signature + Rekor log check

### 5. Kill-Switch (TLA+ Verified)

Located in `infra/tla/KillSwitch.tla`.

Formal specification with proven properties:
- **Safety**: Once frozen, always frozen
- **Liveness**: If drift + quorum → eventually freeze

States: `Live` → `DriftDetected` → `Frozen`

Requires 2-of-N quorum to freeze. No escape from frozen state.

---

## Running a Simulation

### Command Line

```bash
python -m albion.runner \
    --target-bn 50 \
    --epsilon 0.02 \
    --k 5 \
    --max-candidates 5000
```

This will:
1. Generate 5000 candidate policies
2. Evaluate each with all agents (TaxBenefit, Macro, Climate, Distribution)
3. Apply policy gates (OPA)
4. Filter to lawful near-optimal set
5. Select 5 diverse plans (DNOS)
6. Sign all artifacts (Sigstore)
7. Generate diversity certificate
8. Write results to `output/plans/` and `certs/`

### Python API

```python
from albion import Constitution, Policy
from albion.agents import TaxBenefitAgent, MacroAgent, ClimateAgent, DistributionAgent
from albion.dnos import DNOSSelector
from albion.sign import SigstoreService

# Load constitution
constitution = Constitution.parse_file('configs/constitution.json')

# Initialize agents
households = load_synthetic_population()
tb_agent = TaxBenefitAgent(households)
macro_agent = MacroAgent(leontief_matrix, sectors, obr_baselines)
climate_agent = ClimateAgent(emissions_intensities, macc_curves, carbon_budgets)
dist_agent = DistributionAgent(households)

# Initialize DNOS selector
selector = DNOSSelector(constitution)

# Generate candidates (your policy generation logic here)
candidates = generate_policy_candidates(target_revenue_bn=50)

# Evaluate each candidate
for policy in candidates:
    tb_result = tb_agent.simulate_policy(policy)
    macro_result = macro_agent.simulate_macro_impact(policy, tb_result)
    climate_result = climate_agent.simulate_emissions_impact(macro_result, policy)
    policy.impacts = dist_agent.analyze_distribution(policy, tb_result, macro_result, climate_result)

# Select diverse set
result = selector.select(candidates)

# Sign plans
signing_service = SigstoreService()
for plan in result.selected_plans:
    plan.signature = signing_service.sign_artifact(plan.dict())

# Certificate is already signed
print(f"Selected {len(result.selected_plans)} plans")
print(f"Certificate: {result.certificate.explanation}")
```

---

## Testing

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# Property tests (hypothesis)
pytest tests/property -v

# OPA policy tests
opa test policies/ -v

# TLA+ model checking
java -jar tla2tools.jar -config KillSwitch.cfg KillSwitch.tla
```

---

## Deployment

### Docker

```bash
make docker
docker-compose up
```

Services:
- **API**: `localhost:8000`
- **Frontend**: `localhost:3000`
- **OPA**: `localhost:8181`
- **Postgres**: `localhost:5432`

### Kubernetes

```bash
kubectl apply -f infra/k8s/
```

Includes:
- Worker pool (autoscaling)
- API gateway
- OPA sidecar
- Postgres with read replicas
- RabbitMQ for job queue

---

## Configuration

### Constitution

Located in `configs/constitution.json`.

Key parameters:
- `diversity.k`: Number of options to select (default: 5)
- `diversity.epsilon_additive`: Near-optimality threshold (default: 0.02 = 2%)
- `diversity.weights`: Feature weights (distributional, regional, climate, system)
- `diversity.quotas`: Representation guarantees

Example:
```json
{
  "diversity": {
    "k": 5,
    "epsilon_additive": 0.02,
    "weights": {
      "dist": 0.35,
      "regional": 0.25,
      "climate": 0.25,
      "system": 0.15
    },
    "quotas": {
      "low_income_advantaged": 1,
      "north_east_prioritised": 1,
      "climate_ambitious": 1
    }
  }
}
```

### Fiscal Rules

Based on OBR Economic & Fiscal Outlook (March 2025):
- Debt must be falling by year 5
- Borrowing within envelope

### Carbon Budgets

Based on Carbon Budget Orders:
- **Sixth Carbon Budget** (2033-2037): 965 MtCO2e

---

## Documentation

- [SPEC.md](docs/SPEC.md): Complete specification
- [MANDATE.md](docs/MANDATE.md): Implementation mandate
- [ARCHITECTURE.md](docs/ARCHITECTURE.md): System architecture
- API docs: `http://localhost:8000/docs` (when running)

---

## Key Principles

1. **Humans decide. Engine proposes.**
   - ALBION generates options. Parliament/Cabinet chooses.

2. **Law before loss.**
   - Statutory constraints are hard gates. Illegal plans don't render.

3. **Few good options.**
   - Always k (3-7) diverse plans. Never thousands. Never one.

4. **Cohorts, not people.**
   - No PII. Synthetic population. Differential privacy if microdata used.

5. **Receipts or retract.**
   - Every artifact signed and logged. Public verification.

6. **Portability.**
   - Open schemas. Vendor-portable. No lock-in.

---

## Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`make test`)
4. Run linters (`make lint`)
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use ALBION in research, please cite:

```bibtex
@software{albion2025,
  title={ALBION: The National Options Engine},
  author={ALBION Team},
  year={2025},
  url={https://github.com/yourusername/ProjectAlbion}
}
```

---

## Acknowledgments

Built on the shoulders of giants:

- **ONS**: Supply-Use tables, Effects of Taxes & Benefits
- **OBR**: Economic & Fiscal Outlook, ready-reckoners
- **DWP**: Benefit expenditure data
- **DESNZ**: Emissions statistics
- **CCC**: Marginal Abatement Cost data
- **Open Policy Agent**: Policy enforcement
- **Sigstore**: Transparency logging
- **TLA+**: Formal verification

---

## Status

**Production-ready core.** Frontend, API, and deployment automation are in various stages of completion.

Current capabilities:
- ✅ Full microsimulation (TaxBenefitAgent)
- ✅ I/O modeling (MacroAgent with Leontief)
- ✅ Climate modeling (ClimateAgent with MACC)
- ✅ DNOS selection with mathematical guarantees
- ✅ OPA policy gates (fiscal, carbon, devolution, equality)
- ✅ TLA+ kill-switch specification
- ✅ Sigstore signing integration
- ✅ Database schema (append-only, audit trail)
- 🚧 Data ingestion scripts (in progress)
- 🚧 API layer (FastAPI skeleton exists)
- 🚧 Frontend (Next.js components needed)
- 🚧 Full deployment automation

---

## Contact

- Issues: [GitHub Issues](https://github.com/yourusername/ProjectAlbion/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/ProjectAlbion/discussions)
- Security: security@albion.gov.uk (GPG key in repo)

---

**ALBION**: *Lawful, diverse, near-optimal policy options with public receipts.*

*"Show your working. Sign your work. Serve the public."*
