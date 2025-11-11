# ALBION Build & Deployment Guide

Complete guide to building, testing, and deploying ALBION.

---

## Table of Contents

1. [Local Development Setup](#local-development-setup)
2. [Using Real UK Data](#using-real-uk-data)
3. [Running Tests](#running-tests)
4. [Docker Deployment](#docker-deployment)
5. [Production Deployment](#production-deployment)
6. [Database Setup](#database-setup)
7. [Continuous Integration](#continuous-integration)

---

## Local Development Setup

### Prerequisites

- **Python 3.11+**
- **Node.js 18+**
- **PostgreSQL 15+** (optional for local dev)
- **Git**

### Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/ProjectAlbion.git
cd ProjectAlbion
```

### Step 2: Python Environment

```bash
# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install --upgrade pip
pip install -e ".[dev]"
```

### Step 3: Node.js Setup

```bash
cd apps/web
npm install
cd ../..
```

### Step 4: Generate Mock Data

```bash
python tools/generate_mock_data.py
```

### Step 5: Verify Installation

```bash
# Run a quick simulation
python -m albion.runner --target-bn 50 --k 5 --max-candidates 100

# Should output 5 plans to output/plans/
```

---

## Using Real UK Data

### Data Sources

ALBION uses publicly available UK government data:

1. **OBR Economic & Fiscal Outlook** - https://obr.uk
2. **ONS Effects of Taxes & Benefits** - https://www.ons.gov.uk
3. **ONS Supply-Use Tables** - https://www.ons.gov.uk
4. **DWP Benefit Expenditure** - https://www.gov.uk
5. **DESNZ Emissions Data** - https://www.gov.uk
6. **HMRC Personal Incomes** - https://www.gov.uk

### Automated Download

```bash
# Download all data sources
python tools/fetch_data.py
```

**Note**: URLs may change. If downloads fail, manually download from source websites and place in `data/raw/`.

### Manual Processing

```bash
# Transform raw data to processed format
python tools/transform.py

# Build synthetic population from ONS ETB
python tools/build_population.py

# Construct Leontief matrix from SUT/IO tables
python tools/build_leontief.py
```

### Data Quality Checks

```bash
# Validate processed data
python tools/validate_data.py
```

---

## Running Tests

### Unit Tests

```bash
pytest tests/unit -v
```

### Integration Tests

```bash
pytest tests/integration -v
```

### Property Tests (Hypothesis)

```bash
pytest tests/property -v
```

### OPA Policy Tests

```bash
# Install OPA CLI first
curl -L -o /usr/local/bin/opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64_static
chmod +x /usr/local/bin/opa

# Run policy tests
opa test policies/ -v
```

### TLA+ Model Checking

```bash
# Install TLA+ tools
# Download from: https://github.com/tlaplus/tlaplus/releases

# Run model checker
java -jar tla2tools.jar -config infra/tla/KillSwitch.cfg infra/tla/KillSwitch.tla
```

### Coverage Report

```bash
pytest tests/ --cov=albion --cov-report=html
open htmlcov/index.html
```

---

## Docker Deployment

### Build Images

```bash
# Build all images
docker-compose build

# Or build individually
docker build -t albion-api:latest -f Dockerfile.api .
docker build -t albion-web:latest -f apps/web/Dockerfile apps/web/
```

### Development Mode

```bash
# Start all services with hot reload
docker-compose up

# Or start in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Initialize Database

```bash
# Run migrations
docker-compose exec api alembic upgrade head

# Or run SQL directly
docker-compose exec postgres psql -U albion -d albion -f /docker-entrypoint-initdb.d/001_core_schema.sql
```

### Generate Mock Data in Docker

```bash
docker-compose exec api python tools/generate_mock_data.py
```

### Run Simulation in Docker

```bash
docker-compose exec api python -m albion.runner --target-bn 50 --k 5
```

---

## Production Deployment

### Environment Variables

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/albion

# API
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Frontend
NEXT_PUBLIC_API_URL=https://api.albion.gov.uk

# Security
SECRET_KEY=<generate-strong-secret>
ALLOWED_HOSTS=albion.gov.uk,www.albion.gov.uk

# Sigstore
SIGSTORE_IDENTITY=albion-engine@gov.uk
ENABLE_SIGNING=true

# Monitoring
SENTRY_DSN=<your-sentry-dsn>
LOG_LEVEL=INFO
```

### Kubernetes Deployment

```bash
# Apply configurations
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/configmap.yaml
kubectl apply -f infra/k8s/secrets.yaml
kubectl apply -f infra/k8s/postgres.yaml
kubectl apply -f infra/k8s/api.yaml
kubectl apply -f infra/k8s/web.yaml
kubectl apply -f infra/k8s/ingress.yaml

# Check status
kubectl get pods -n albion
kubectl get svc -n albion
```

### Health Checks

```bash
# API health
curl https://api.albion.gov.uk/health

# Database health
kubectl exec -it postgres-0 -n albion -- pg_isready
```

### Monitoring

```bash
# Prometheus metrics
curl https://api.albion.gov.uk/metrics

# Logs (if using ELK stack)
kubectl logs -f -l app=albion-api -n albion
```

---

## Database Setup

### Local PostgreSQL

```bash
# Install PostgreSQL 15
brew install postgresql@15  # macOS
apt-get install postgresql-15  # Ubuntu

# Start service
brew services start postgresql@15
systemctl start postgresql

# Create database
createdb albion

# Run schema
psql albion < db/schemas/001_core_schema.sql
```

### Alembic Migrations

```bash
# Initialize Alembic (already done)
alembic init db/migrations

# Create migration
alembic revision -m "Add new feature"

# Run migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Database Backups

```bash
# Backup
pg_dump -U albion albion > backup_$(date +%Y%m%d).sql

# Restore
psql -U albion albion < backup_20250111.sql
```

---

## Continuous Integration

### GitHub Actions

`.github/workflows/ci.yml`:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -e ".[dev]"

    - name: Run tests
      run: |
        pytest tests/ --cov=albion

    - name: Lint
      run: |
        ruff check .
        mypy albion/

    - name: OPA policy tests
      run: |
        curl -L -o opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64_static
        chmod +x opa
        ./opa test policies/ -v
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

---

## Performance Tuning

### Python Optimization

```bash
# Use PyPy for faster execution
pypy3.11 -m pip install -e ".[dev]"
pypy3.11 -m albion.runner --target-bn 50 --k 5
```

### Parallel Processing

```python
# In albion/runner.py, use multiprocessing
from multiprocessing import Pool

with Pool(processes=8) as pool:
    results = pool.map(evaluate_policy, candidates)
```

### Database Indexing

```sql
-- Add indexes for common queries
CREATE INDEX idx_plans_created ON plans(created_at DESC);
CREATE INDEX idx_plans_objective ON plans(objective_value DESC);
```

---

## Security Hardening

### 1. Use Strong Secrets

```bash
# Generate secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### 2. Enable HTTPS

```yaml
# In ingress.yaml
spec:
  tls:
    - secretName: albion-tls
      hosts:
        - albion.gov.uk
```

### 3. Rate Limiting

```python
# In API
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter

@app.post("/simulations", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def create_simulation(...):
    ...
```

### 4. Input Validation

```python
# All inputs validated via Pydantic models
# SQL injection prevented (using SQLAlchemy ORM)
# XSS prevented (React escapes by default)
```

---

## Monitoring & Alerting

### Metrics to Track

- **API**: Request rate, latency, error rate
- **Simulations**: Success rate, runtime, queue depth
- **Database**: Connection pool, query performance
- **System**: CPU, memory, disk usage

### Prometheus + Grafana

```bash
# Install Prometheus
helm install prometheus prometheus-community/prometheus

# Install Grafana
helm install grafana grafana/grafana

# Import ALBION dashboard
kubectl apply -f infra/monitoring/grafana-dashboard.json
```

---

## Troubleshooting

### Simulation Fails

**Check logs**:
```bash
docker-compose logs api
```

**Common issues**:
- Missing data: Run `python tools/generate_mock_data.py`
- Memory: Increase Docker memory limit (Docker Desktop → Preferences → Resources)
- Timeout: Increase timeout in `albion/api/main.py`

### Database Connection Errors

```bash
# Check database is running
docker-compose ps postgres

# Check connection
docker-compose exec postgres psql -U albion -c "SELECT 1"
```

### Frontend Not Updating

```bash
# Clear Next.js cache
cd apps/web
rm -rf .next
npm run dev
```

---

## Advanced Topics

### Custom Policy Generation

Create custom policy generator in `albion/generators/`:

```python
from albion.models import Policy, PolicyLevers

class CustomGenerator:
    def generate(self, n: int) -> List[Policy]:
        # Your custom logic here
        pass
```

### Custom Agents

Add custom agent in `albion/agents/`:

```python
class CustomAgent:
    def simulate(self, policy: Policy) -> dict:
        # Your analysis here
        return {'custom_metric': 123}
```

### Custom OPA Policies

Add new policy file in `policies/`:

```rego
package custom

deny[msg] {
    # Your custom rule
    msg := "Custom constraint violated"
}
```

---

## Resources

- **Documentation**: `docs/` directory
- **Examples**: `examples/` directory
- **API Docs**: http://localhost:8000/docs
- **Community**: GitHub Discussions

---

**Questions?** Open an issue or check the documentation!
