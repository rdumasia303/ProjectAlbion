.PHONY: help deps data simulate serve test lint clean docker all

help:
	@echo "ALBION Engine - Available Commands"
	@echo "===================================="
	@echo "  make deps        - Install all dependencies (Python + Node + OPA + cosign)"
	@echo "  make data        - Fetch and process all UK public data sources"
	@echo "  make db-init     - Initialize database schema"
	@echo "  make simulate    - Run policy simulation (generates 5 diverse plans)"
	@echo "  make serve       - Start API and frontend"
	@echo "  make test        - Run full test suite"
	@echo "  make lint        - Run linters and type checkers"
	@echo "  make clean       - Remove generated files"
	@echo "  make docker      - Build all Docker images"
	@echo "  make all         - Complete setup: deps + data + db + simulate"

deps:
	@echo "📦 Installing Python dependencies..."
	pip install -e ".[dev]"
	@echo "📦 Installing Node dependencies..."
	cd apps/web && npm install
	@echo "📦 Installing OPA..."
	curl -L -o /usr/local/bin/opa https://openpolicyagent.org/downloads/latest/opa_linux_amd64_static
	chmod +x /usr/local/bin/opa
	@echo "📦 Installing cosign..."
	curl -L -o /usr/local/bin/cosign https://github.com/sigstore/cosign/releases/latest/download/cosign-linux-amd64
	chmod +x /usr/local/bin/cosign
	@echo "✅ All dependencies installed"

data:
	@echo "🌐 Fetching UK public data sources..."
	python tools/fetch_data.py
	@echo "🔧 Processing and harmonizing data..."
	python tools/transform.py
	@echo "📊 Building synthetic population..."
	python tools/build_population.py
	@echo "🧮 Computing Leontief matrix..."
	python tools/build_leontief.py
	@echo "✅ Data layer complete"

db-init:
	@echo "🗄️  Initializing database schema..."
	alembic upgrade head
	@echo "✅ Database ready"

simulate:
	@echo "🎯 Running ALBION simulation..."
	python -m albion.runner \
		--target-bn 50 \
		--epsilon 0.02 \
		--k 5 \
		--max-candidates 5000
	@echo "✅ Simulation complete - check output/plans/"

serve:
	@echo "🚀 Starting ALBION services..."
	@echo "  - API on http://localhost:8000"
	@echo "  - Frontend on http://localhost:3000"
	@echo "  - OPA on http://localhost:8181"
	docker-compose up -d
	@echo "✅ Services running"

test:
	@echo "🧪 Running unit tests..."
	pytest tests/unit -v
	@echo "🧪 Running integration tests..."
	pytest tests/integration -v
	@echo "🧪 Running property tests..."
	pytest tests/property -v
	@echo "🧪 Running backtests..."
	python tests/backtests/run_backtests.py
	@echo "✅ All tests passed"

lint:
	@echo "🔍 Running ruff..."
	ruff check .
	@echo "🔍 Running mypy..."
	mypy albion/
	@echo "🔍 Running black..."
	black --check albion/ tests/
	@echo "🔍 Testing OPA policies..."
	opa test policies/ -v
	@echo "✅ All checks passed"

clean:
	@echo "🧹 Cleaning generated files..."
	rm -rf data/processed/*
	rm -rf data/models/*
	rm -rf output/*
	rm -rf .pytest_cache
	rm -rf .coverage htmlcov
	rm -rf **/__pycache__
	rm -rf *.egg-info
	@echo "✅ Cleaned"

docker:
	@echo "🐳 Building Docker images..."
	docker build -t albion-api:latest -f infra/docker/Dockerfile.api .
	docker build -t albion-worker:latest -f infra/docker/Dockerfile.worker .
	docker build -t albion-web:latest -f infra/docker/Dockerfile.web apps/web/
	@echo "✅ Docker images built"

all: deps data db-init simulate
	@echo "🎉 ALBION Engine fully initialized and ready!"
	@echo ""
	@echo "Next steps:"
	@echo "  1. Run 'make serve' to start services"
	@echo "  2. Visit http://localhost:3000 to see the 5 diverse options"
	@echo "  3. Check certs/ directory for signed diversity certificates"
