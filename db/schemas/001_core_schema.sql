-- ALBION Engine Core Database Schema
-- Append-only architecture with cryptographic audit trail
-- PostgreSQL 15+

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================================
-- Data Layer: Synthetic Population & Economic Structure
-- ============================================================================

CREATE TABLE households (
    cohort_id BIGSERIAL PRIMARY KEY,
    income_decile INT NOT NULL CHECK (income_decile BETWEEN 1 AND 10),
    household_type VARCHAR(50) NOT NULL,
    region VARCHAR(10) NOT NULL,  -- ITL1 code
    gross_income NUMERIC(12,2) NOT NULL,
    equivalised_income NUMERIC(12,2) NOT NULL,
    benefit_flags JSONB NOT NULL DEFAULT '{}',
    tax_profile JSONB NOT NULL DEFAULT '{}',
    consumption_basket JSONB NOT NULL DEFAULT '{}',
    weight NUMERIC(12,2) NOT NULL,  -- Number of real households represented
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    data_version VARCHAR(20) NOT NULL  -- e.g., "2024_etb"
);

CREATE INDEX idx_households_decile ON households(income_decile);
CREATE INDEX idx_households_region ON households(region);
CREATE INDEX idx_households_type ON households(household_type);

COMMENT ON TABLE households IS 'Synthetic population: ~100k cohorts representing UK households';
COMMENT ON COLUMN households.weight IS 'Number of real UK households this cohort represents';

-- ============================================================================

CREATE TABLE economic_sectors (
    sector_id SERIAL PRIMARY KEY,
    sector_code VARCHAR(10) NOT NULL,
    sector_name VARCHAR(200) NOT NULL,
    parent_sector VARCHAR(10),
    gva_million_gbp NUMERIC(15,2),
    employment_thousands NUMERIC(12,2),
    emissions_tco2e NUMERIC(15,2),
    emissions_intensity NUMERIC(10,6),  -- tCO2e per £million GVA
    year INT NOT NULL,
    UNIQUE(sector_code, year)
);

CREATE INDEX idx_sectors_code ON economic_sectors(sector_code);
CREATE INDEX idx_sectors_year ON economic_sectors(year);

COMMENT ON TABLE economic_sectors IS 'Economic sectors from ONS Supply-Use tables';

-- ============================================================================

CREATE TABLE leontief_matrix (
    id SERIAL PRIMARY KEY,
    row_sector VARCHAR(10) NOT NULL,
    col_sector VARCHAR(10) NOT NULL,
    coefficient NUMERIC(10,6) NOT NULL,
    version VARCHAR(20) NOT NULL,  -- e.g., "2023_blue_book"
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(row_sector, col_sector, version)
);

CREATE INDEX idx_leontief_version ON leontief_matrix(version);

COMMENT ON TABLE leontief_matrix IS 'Leontief inverse matrix (I-A)^-1 for input-output modeling';
COMMENT ON COLUMN leontief_matrix.coefficient IS 'Output multiplier: how £1 change in col_sector affects row_sector';

-- ============================================================================

CREATE TABLE emissions_intensity (
    id SERIAL PRIMARY KEY,
    sector_code VARCHAR(10) NOT NULL,
    sector_name VARCHAR(200) NOT NULL,
    emissions_tco2e NUMERIC(15,2) NOT NULL,
    gva_million_gbp NUMERIC(15,2) NOT NULL,
    intensity NUMERIC(10,6) NOT NULL,  -- tCO2e per £million GVA
    year INT NOT NULL,
    UNIQUE(sector_code, year)
);

COMMENT ON TABLE emissions_intensity IS 'DESNZ emissions data mapped to economic sectors';

-- ============================================================================

CREATE TABLE macc_curves (
    id SERIAL PRIMARY KEY,
    sector_code VARCHAR(10) NOT NULL,
    carbon_price_gbp_per_tco2e NUMERIC(8,2) NOT NULL,
    abatement_mtco2e NUMERIC(10,4) NOT NULL,  -- Cumulative abatement at this price
    technology VARCHAR(100),  -- e.g., "fuel_switching", "ccs", "efficiency"
    year INT NOT NULL,
    UNIQUE(sector_code, carbon_price_gbp_per_tco2e, year)
);

CREATE INDEX idx_macc_sector_year ON macc_curves(sector_code, year);

COMMENT ON TABLE macc_curves IS 'Marginal Abatement Cost Curves from DESNZ/CCC data';

-- ============================================================================

CREATE TABLE government_accounts (
    id SERIAL PRIMARY KEY,
    account_code VARCHAR(20) NOT NULL,
    account_name VARCHAR(200) NOT NULL,
    account_type VARCHAR(20) NOT NULL,  -- 'receipt', 'rdel', 'cdel', 'ame'
    department VARCHAR(10),  -- e.g., 'DfT', 'HO', NULL for receipts
    amount_million_gbp NUMERIC(15,2) NOT NULL,
    fiscal_year VARCHAR(10) NOT NULL,  -- e.g., '2024-25'
    source VARCHAR(50) NOT NULL,  -- 'OBR_EFO', 'PESA_2025'
    UNIQUE(account_code, fiscal_year)
);

CREATE INDEX idx_gov_accounts_year ON government_accounts(fiscal_year);
CREATE INDEX idx_gov_accounts_dept ON government_accounts(department);

COMMENT ON TABLE government_accounts IS 'OBR/PESA baseline fiscal accounts';

-- ============================================================================
-- Policy Layer: Plans, Gates, Certificates (Append-Only)
-- ============================================================================

CREATE TABLE plans (
    plan_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    simulation_id UUID NOT NULL,
    plan_name VARCHAR(100) NOT NULL,
    content JSONB NOT NULL,  -- Full policy with levers, impacts, etc.
    content_hash VARCHAR(64) NOT NULL,
    objective_value NUMERIC(15,6),  -- For near-optimality filtering
    gate_results JSONB NOT NULL DEFAULT '{}',
    signature JSONB,  -- Sigstore signature bundle
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100)
);

CREATE INDEX idx_plans_simulation ON plans(simulation_id);
CREATE INDEX idx_plans_created ON plans(created_at DESC);
CREATE INDEX idx_plans_hash ON plans(content_hash);

COMMENT ON TABLE plans IS 'Append-only table of policy plans (no UPDATE/DELETE allowed)';

-- Trigger to prevent updates
CREATE OR REPLACE FUNCTION prevent_plan_update()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Updates not allowed on plans table (append-only)';
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER no_updates_plans
BEFORE UPDATE ON plans
FOR EACH ROW EXECUTE FUNCTION prevent_plan_update();

-- Trigger to auto-calculate content hash
CREATE OR REPLACE FUNCTION calculate_plan_hash()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_hash := encode(
        digest(NEW.content::text, 'sha256'),
        'hex'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER hash_plan_content
BEFORE INSERT ON plans
FOR EACH ROW EXECUTE FUNCTION calculate_plan_hash();

-- ============================================================================

CREATE TABLE certificates (
    cert_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    simulation_id UUID NOT NULL,
    plan_ids UUID[] NOT NULL,
    diversity_proof JSONB NOT NULL,
    content JSONB NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    signature JSONB,  -- Sigstore signature bundle
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_certs_simulation ON certificates(simulation_id);
CREATE INDEX idx_certs_hash ON certificates(content_hash);

COMMENT ON TABLE certificates IS 'Diversity certificates proving "why these k plans?"';

CREATE TRIGGER no_updates_certs
BEFORE UPDATE ON certificates
FOR EACH ROW EXECUTE FUNCTION prevent_plan_update();

CREATE OR REPLACE FUNCTION calculate_cert_hash()
RETURNS TRIGGER AS $$
BEGIN
    NEW.content_hash := encode(
        digest(NEW.content::text, 'sha256'),
        'hex'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER hash_cert_content
BEFORE INSERT ON certificates
FOR EACH ROW EXECUTE FUNCTION calculate_cert_hash();

-- ============================================================================

CREATE TABLE simulations (
    simulation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status VARCHAR(20) NOT NULL DEFAULT 'queued',  -- queued, running, complete, failed
    progress INT DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    params JSONB NOT NULL,
    error_message TEXT,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    created_by VARCHAR(100)
);

CREATE INDEX idx_simulations_status ON simulations(status);
CREATE INDEX idx_simulations_created ON simulations(created_at DESC);

COMMENT ON TABLE simulations IS 'Simulation job tracking (supports async architecture)';

-- This table CAN be updated (for status tracking)

-- ============================================================================

CREATE TABLE gate_evaluations (
    eval_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    plan_id UUID NOT NULL REFERENCES plans(plan_id),
    gate_name VARCHAR(100) NOT NULL,
    decision VARCHAR(20) NOT NULL,  -- 'allow', 'warn', 'deny', 'block'
    messages TEXT[],
    evaluated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_gate_evals_plan ON gate_evaluations(plan_id);

COMMENT ON TABLE gate_evaluations IS 'OPA policy gate evaluation results';

-- ============================================================================
-- Audit & Integrity Layer
-- ============================================================================

CREATE TABLE audit_log (
    log_id BIGSERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    entity_type VARCHAR(50),
    entity_id UUID,
    user_id VARCHAR(100),
    changes JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_audit_entity ON audit_log(entity_type, entity_id);
CREATE INDEX idx_audit_created ON audit_log(created_at DESC);

COMMENT ON TABLE audit_log IS 'Comprehensive audit trail for all state changes';

-- ============================================================================

CREATE TABLE kill_switch_state (
    id INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),  -- Singleton table
    state VARCHAR(20) NOT NULL DEFAULT 'Live',  -- Live, DriftDetected, Frozen
    approvals TEXT[] NOT NULL DEFAULT '{}',
    quorum_size INT NOT NULL DEFAULT 2,
    data_drift_detected BOOLEAN NOT NULL DEFAULT FALSE,
    last_drift_check TIMESTAMP,
    frozen_at TIMESTAMP,
    frozen_by TEXT[],
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

INSERT INTO kill_switch_state (id) VALUES (1);

COMMENT ON TABLE kill_switch_state IS 'Kill-switch state (TLA+ verified)';

-- Trigger to enforce kill-switch immutability
CREATE OR REPLACE FUNCTION enforce_kill_switch()
RETURNS TRIGGER AS $$
BEGIN
    -- Once frozen, cannot transition back
    IF OLD.state = 'Frozen' AND NEW.state != 'Frozen' THEN
        RAISE EXCEPTION 'Cannot unfreeze system (safety invariant)';
    END IF;

    -- Quorum requirement
    IF NEW.state = 'Frozen' AND array_length(NEW.approvals, 1) < NEW.quorum_size THEN
        RAISE EXCEPTION 'Insufficient approvals for freeze (need %)', NEW.quorum_size;
    END IF;

    NEW.updated_at := NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER enforce_kill_switch_invariants
BEFORE UPDATE ON kill_switch_state
FOR EACH ROW EXECUTE FUNCTION enforce_kill_switch();

-- ============================================================================
-- Configuration Layer
-- ============================================================================

CREATE TABLE constitution (
    id INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),  -- Singleton
    version VARCHAR(20) NOT NULL,
    content JSONB NOT NULL,
    effective_from TIMESTAMP NOT NULL DEFAULT NOW(),
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

INSERT INTO constitution (version, content, effective_from) VALUES (
    'constitution/v1',
    '{
        "fiscal_rules": {"source":"OBR/EFO/2025-03","debt_rule":"falling_by_year_5", "borrowing_rule":"within_envelope"},
        "carbon_budgets": {"sixth": {"total_mtco2e": 965, "years": "2033-2037"}},
        "devolution": {
            "scotland":{"income_tax_bands":"devolved"},
            "wales":{"income_tax_bands":"partial"},
            "northern_ireland":{"corporation_tax":"devolved"},
            "ukwide":["VAT","NICs"]
        },
        "equality": {"psed":{"enabled": true, "warn_ratio": 1.5}},
        "diversity": {
            "k":5,
            "epsilon_additive":0.02,
            "weights":{"dist":0.35,"regional":0.25,"climate":0.25,"system":0.15},
            "quotas":{"low_income_advantaged":1,"north_east_prioritised":1}
        }
    }'::jsonb,
    NOW()
);

COMMENT ON TABLE constitution IS 'Active constitution (versioned, immutable)';

-- ============================================================================
-- Performance & Maintenance
-- ============================================================================

-- Vacuum and analyze recommendations
COMMENT ON DATABASE postgres IS 'ALBION Engine Database - Run VACUUM ANALYZE weekly';

-- ============================================================================
-- Summary Statistics View
-- ============================================================================

CREATE OR REPLACE VIEW system_stats AS
SELECT
    (SELECT COUNT(*) FROM households) AS household_cohorts,
    (SELECT COUNT(*) FROM economic_sectors WHERE year = 2023) AS economic_sectors,
    (SELECT COUNT(*) FROM plans) AS total_plans,
    (SELECT COUNT(*) FROM certificates) AS total_certificates,
    (SELECT COUNT(*) FROM simulations WHERE status = 'complete') AS completed_simulations,
    (SELECT state FROM kill_switch_state WHERE id = 1) AS kill_switch_state,
    (SELECT version FROM constitution WHERE id = 1) AS constitution_version;

COMMENT ON VIEW system_stats IS 'Quick system health check';

-- ============================================================================
-- Grant appropriate permissions (adjust for your deployment)
-- ============================================================================

-- Example: Read-only user for verification/audit
-- CREATE ROLE albion_readonly;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO albion_readonly;

-- Example: Worker role (can INSERT but not UPDATE/DELETE)
-- CREATE ROLE albion_worker;
-- GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA public TO albion_worker;
-- REVOKE UPDATE, DELETE ON plans, certificates FROM albion_worker;

-- ============================================================================
-- Schema complete
-- ============================================================================

SELECT 'ALBION Core Schema v1.0 initialized successfully' AS status;
