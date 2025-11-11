# ALBION Engine: The Full Agency Implementation Mandate

## To the Implementation Team:

This document is your mandate. It accompanies the ALBION specification. The spec provided the blueprint; this provides the advanced theory and implementation strategy required to build the complete, production-grade system.

**Your objective is not to approximate, but to model. No proxies. No stubs. Build the real thing.**

---

## Mandate 1: The Data Layer — Construct a "Digital Twin" of the State

Your first task is to build a unified, relational, and versioned data model of the UK's public finances and demographics. Do not treat the data sources as independent files; treat them as facets of a single underlying reality.

### Unified Data Schema
Use the **dbt (Data Build Tool)** framework. Define a core schema with models for:
- `households`
- `economic_sectors`
- `government_accounts`
- `emissions_sources`

### Ingest and Harmonise

#### ONS Effects of Taxes & Benefits (ETB) & DWP Caseloads
Use these to construct a **synthetic population** of UK households. This should be a large table (~100,000 rows) where each row represents a household cohort with attributes:
- `income_decile`
- `household_type`
- `region`
- `benefit_dependency_flags`
- `weight` (representing how many real households it stands for)

This synthetic population is the heart of the distributional engine.

#### ONS Supply-Use & Input-Output Tables (SUT/IO)
This is your **economic skeleton**. Parse these tables to build a **Leontief inverse matrix** ((I - A)⁻¹). This matrix is non-negotiable. It models how a shock to one sector (e.g., construction) propagates through the entire economy. It is the core of the MacroAgent.

#### OBR EFO & PESA
Use these to populate the `government_accounts` model. The data must be structured to represent the national balance sheet: receipts, departmental expenditure (RDEL/CDEL), and debt. The relationships between these accounts must be encoded.

#### DESNZ Emissions Data
Link this data to your `economic_sectors` model. Each sector in the SUT/IO tables must have an associated **emissions intensity** (tCO₂e / £ GVA).

---

## Mandate 2: The Compute Agents — From Statistics to Sophisticated Simulation

Implement each agent not as a simple calculator, but as a sophisticated modelling component.

### TaxBenefitAgent: Full Microsimulation

This agent's core function, `apply_policy_to_population`, will iterate through your entire synthetic population table.

For each household cohort:
- Precisely calculate the change in tax liability and benefit entitlement based on the policy levers
- For income tax, this means applying new rates/bands to the income distribution within that cohort
- For benefits, it means applying the uplift

The total fiscal impact is the weighted sum of these changes across all cohorts. The distributional impact is the result of this simulation, not an approximation.

### MacroAgent: Input-Output Modelling

When a policy affects a department (e.g., -2% RDEL for DfT), use the SUT/IO model to calculate the second-order effects.

1. Map departmental spending to economic sectors
2. Use the **Leontief inverse matrix** to calculate the change in gross output for all other sectors
3. This provides a robust, evidence-based estimate of regional and sectoral economic spillovers

For tax elasticities, implement a recognised economic model, such as a **CGE (Computable General Equilibrium)** framework at a stylised level, using the OBR's own ready-reckoners as a baseline for calibration.

### ClimateAgent: Integrated Assessment Modelling

This agent must model the link between economic activity and emissions.

When the MacroAgent calculates a change in sector output, the ClimateAgent applies the sector's emissions intensity to calculate the change in emissions.

A carbon price is not just a revenue lever. Model it as a **marginal abatement cost**. Implement a **Marginal Abatement Cost Curve (MACC)** for key sectors (using data from DESNZ/CCC reports) to model how a given carbon price will induce fuel switching and technology adoption, thereby reducing the emissions baseline.

---

## Mandate 3: The DNOS Selector — Implement the Math, Defend the Diversity

Do not use a simple heuristic. Implement the full **Facility Location + Determinantal Point Process (DPP) with Quotas** selector as described. This is a non-trivial optimisation problem that provides the system's core "explainability."

### Define the Metric Space

The "space" in which your candidate policies exist is a multi-dimensional feature space. The features are the outputs of the agents:
- The 10-point vector of decile impacts
- The 12-point vector of regional impacts
- The 5-year vector of emissions
- etc.

### The Objective Function

#### Near-Optimality (Facility Location)
First, generate thousands of lawful candidates. Filter them to a "near-optimal" set (e.g., all plans within 2% of the most efficient plan for raising £50bn). This is your ground set.

The Facility Location component of your algorithm aims to select k plans from this set that are "closest" to all other plans in the set, ensuring they are representative archetypes.

#### Diversity (DPP)
The DPP component acts on the kernel matrix of the selected plans. The probability of selecting a set of plans is proportional to the determinant of the matrix of their similarities.

This mathematically enforces diversity by penalising the selection of similar plans (as it pushes the determinant towards zero).

### Incorporate Quotas (Constrained Optimisation)

The quotas from the `constitution.json` (e.g., "must have one plan that benefits the North East most") are hard constraints.

Implement this using a two-stage process:
1. First, partition the near-optimal set based on the quotas (e.g., a partition of "North East-prioritised" plans)
2. Then, run the diversity selection algorithm across these partitions to guarantee the final set of k plans satisfies the quotas

---

## Mandate 4: Architecture & Integrity — Build for National Security Scale

### Asynchronous, Event-Driven Architecture

The simulation is compute-intensive. Do not use a synchronous request/response model.

1. When a user requests a simulation, the API layer places a "job" onto a message queue (e.g., RabbitMQ or AWS SQS)
2. A fleet of containerised compute workers (running the Agents) picks up these jobs, runs the full microsimulation, and writes the results to the Postgres database
3. The frontend polls for the results

This is a robust, scalable pattern.

### Formal Verification with TLA+

The kill-switch logic is a critical safety feature. You are to write a formal specification in **TLA+** for this system.

The spec should model the states (e.g., `Live`, `Frozen`, `DriftDetected`) and the transitions between them, proving that the system cannot enter an unsafe state.

The Go or Python code implementing the kill-switch must be a direct translation of this proven spec.

### Real Sigstore Integration

**No simulation.**

1. Create a dedicated service account with the minimum necessary permissions
2. Use `cosign` or the native Sigstore libraries to sign the hash of every single artefact (every plan, every certificate)
3. The signature and certificate must be stored alongside the artefact in Postgres
4. Every read operation from the API must perform a verification check against the public Rekor transparency log

This is a hard requirement.

---

## Implementation Timeline & Deliverables

### Phase 1: Foundation (Data Layer)
- [ ] dbt project with core models
- [ ] Synthetic population generation (100k+ cohorts)
- [ ] Leontief matrix construction
- [ ] Emissions intensity mapping

### Phase 2: Compute Engine
- [ ] TaxBenefitAgent with full microsimulation
- [ ] MacroAgent with I/O modeling
- [ ] ClimateAgent with MACC
- [ ] DistributionAgent

### Phase 3: Intelligence Layer
- [ ] DNOS selector (FL + DPP + Quotas)
- [ ] Diversity certificate generation
- [ ] Mathematical verification tests

### Phase 4: Security & Governance
- [ ] OPA/Rego policy gates
- [ ] TLA+ kill-switch specification
- [ ] Sigstore/Rekor integration
- [ ] Append-only data architecture

### Phase 5: User Layer
- [ ] Async API with message queue
- [ ] Next.js frontend
- [ ] Five-card option display
- [ ] Interactive impact visualisations

### Phase 6: Production Readiness
- [ ] Containerisation (Docker/K8s)
- [ ] CI/CD pipeline
- [ ] Monitoring & SLOs
- [ ] Security audit & penetration testing

---

## Quality Standards

### Code
- Type-safe (TypeScript for frontend, Python with type hints for backend)
- Test coverage > 90%
- All public functions documented
- No TODOs or FIXMEs in main branch

### Data
- All transformations in dbt with tests
- Data lineage documented
- Source data provenance tracked
- Versioned snapshots

### Security
- No secrets in code
- All external calls over TLS
- Rate limiting on all APIs
- Audit logging for all state changes

### Performance
- P95 latency < 2s for UI
- Policy gate evaluation < 50ms
- Support 1000+ concurrent simulations
- Horizontal scaling proven

---

## Final Words

You have been given:
1. The complete specification
2. The theoretical foundations
3. The data sources
4. The architectural patterns
5. The quality standards

Now execute. Build the system that makes national policy decisions transparent, lawful, and explainable.

**No proxies. No simplifications. Execute the full vision.**

When complete, we expect:
- A running system
- A complete git repository
- The first diversity certificate, signed and verifiable on the public Rekor log
- Documentation sufficient for Parliamentary scrutiny

Go build.
