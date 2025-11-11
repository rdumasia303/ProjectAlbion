# ALBION Quick Start Guide

Get ALBION running locally in **5 minutes**!

## Prerequisites

- Python 3.11+
- Node.js 18+
- (Optional) Docker & Docker Compose

## Option 1: Quick Start with Mock Data (Recommended)

### 1. Clone and Install

```bash
git clone https://github.com/your-repo/ProjectAlbion.git
cd ProjectAlbion

# Install Python dependencies
pip install -e ".[dev]"

# Install Node dependencies (for frontend)
cd apps/web
npm install
cd ../..
```

### 2. Generate Mock Data

```bash
# Generate synthetic data for demo (~30 seconds)
python tools/generate_mock_data.py
```

This creates:
- ✅ 100,000 synthetic household cohorts
- ✅ Leontief inverse matrix (105 sectors)
- ✅ Emissions intensities
- ✅ OBR baseline forecasts

### 3. Run Your First Simulation

```bash
# Run simulation to generate 5 diverse policy options
python -m albion.runner --target-bn 50 --k 5
```

This will:
- Generate 1,000 candidate policies
- Evaluate each with all 4 agents (TaxBenefit, Macro, Climate, Distribution)
- Select 5 diverse, near-optimal plans
- Output results to `output/plans/` and `certs/`

**Expected runtime**: ~2-3 minutes

### 4. Start the API (Optional)

```bash
# In one terminal
python -m albion.api.main
```

API will be available at: http://localhost:8000

API documentation: http://localhost:8000/docs

### 5. Start the Frontend (Optional)

```bash
# In another terminal
cd apps/web
npm run dev
```

Frontend will be available at: http://localhost:3000

---

## Option 2: Docker Compose (Complete Stack)

### 1. Build and Start

```bash
# Build images
docker-compose build

# Start all services
docker-compose up -d
```

### 2. Generate Mock Data

```bash
# Run inside the API container
docker-compose exec api python tools/generate_mock_data.py
```

### 3. Run Simulation

```bash
docker-compose exec api python -m albion.runner --target-bn 50 --k 5
```

### 4. Access Services

- **Frontend**: http://localhost:3000
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Database**: localhost:5432 (user: albion, password: albion_dev_password)

### 5. Stop Services

```bash
docker-compose down
```

---

## What You Get

After running a simulation, you'll have:

### 1. Five Diverse Policy Options

Located in `output/plans/`:
- `plan_1.json` - Progressive Tax
- `plan_2.json` - Broad-Based
- `plan_3.json` - Green Priority
- `plan_4.json` - Spending Efficiency
- `plan_5.json` - Balanced Approach

Each plan includes:
- ✅ Complete policy levers (tax rates, spending cuts, carbon price)
- ✅ Distributional impacts (all 10 deciles)
- ✅ Regional impacts (12 ITL1 regions)
- ✅ Climate impacts (emissions trajectory vs carbon budget)
- ✅ Macro impacts (GDP, employment, debt)
- ✅ Cryptographic signature (mock)

### 2. Diversity Certificate

Located in `certs/`:
- Mathematical proof of "why these five?"
- Facility Location coverage radius
- DPP kernel determinant
- Quota satisfaction proof

---

## Example: Running Different Scenarios

```bash
# Raise £100bn (higher target)
python -m albion.runner --target-bn 100 --k 5

# Generate only 3 options
python -m albion.runner --target-bn 50 --k 3

# Use more candidates (slower, better quality)
python -m albion.runner --target-bn 50 --k 5 --max-candidates 5000
```

---

## Viewing Results

### Command Line

```bash
# View a plan
cat output/plans/plan_1.json | python -m json.tool

# View certificate
cat certs/diversity_cert_*.json | python -m json.tool
```

### Web Interface

1. Start API: `python -m albion.api.main`
2. Start frontend: `cd apps/web && npm run dev`
3. Visit: http://localhost:3000
4. Click "Run Simulation" to generate new plans
5. View five cards with all impacts

---

## Troubleshooting

### "Module not found"

```bash
# Make sure you installed in development mode
pip install -e ".[dev]"
```

### "Data not found"

```bash
# Generate mock data first
python tools/generate_mock_data.py
```

### "API connection failed"

```bash
# Make sure API is running
python -m albion.api.main

# Check it's responding
curl http://localhost:8000/health
```

### "Port already in use"

```bash
# Find and kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Or use different port
uvicorn albion.api.main:app --port 8001
```

---

## Next Steps

1. **Explore the Code**: See [ARCHITECTURE.md](docs/ARCHITECTURE.md)
2. **Understand the Math**: See [SPEC.md](docs/SPEC.md)
3. **Run with Real Data**: See [BUILD.md](BUILD.md#using-real-uk-data)
4. **Deploy to Production**: See [BUILD.md](BUILD.md#production-deployment)

---

## Getting Help

- **Documentation**: See `docs/` directory
- **API Docs**: http://localhost:8000/docs (when API is running)
- **Issues**: GitHub Issues
- **Examples**: See `examples/` directory

---

**Congratulations!** 🎉

You now have a working national-scale policy simulation engine running locally.

**ALBION**: Lawful, diverse, near-optimal policy options with public receipts.

*"Show your working. Sign your work. Serve the public."*
