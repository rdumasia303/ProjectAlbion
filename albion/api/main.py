"""
ALBION FastAPI Backend

Async API for running simulations and retrieving results.

Endpoints:
- POST /simulations - Create new simulation job
- GET /simulations/{id} - Get simulation status/results
- GET /plans/{id} - Get specific plan details
- GET /certificates/{id} - Get diversity certificate
- GET /health - Health check
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rich.console import Console

app = FastAPI(
    title="ALBION National Options Engine API",
    description="Production-grade policy simulation with legal compliance and diversity guarantees",
    version="1.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

console = Console()

# In-memory storage (in production, use Postgres)
simulations_db: Dict[str, dict] = {}
plans_db: Dict[str, dict] = {}
certificates_db: Dict[str, dict] = {}

# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "plans"
CERTS_DIR = PROJECT_ROOT / "certs"


# Request/Response models
class SimulationRequest(BaseModel):
    """Request to create a new simulation"""
    target_revenue_bn: float = 50.0
    k: int = 5
    epsilon: float = 0.02
    max_candidates: int = 1000


class SimulationStatus(BaseModel):
    """Simulation status response"""
    simulation_id: str
    status: str  # queued, running, complete, failed
    progress: int  # 0-100
    created_at: str
    completed_at: Optional[str] = None
    error: Optional[str] = None
    result: Optional[dict] = None


class PlanSummary(BaseModel):
    """Summary of a plan"""
    plan_id: str
    name: str
    revenue_bn: float
    emissions_delta_mtco2e: float
    decile_impacts: List[float]
    regional_impacts: Dict[str, dict]


# Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "ALBION National Options Engine",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "simulations_count": len(simulations_db),
        "plans_count": len(plans_db)
    }


@app.post("/simulations", response_model=SimulationStatus)
async def create_simulation(
    request: SimulationRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new simulation job

    This queues a background task to run the full simulation.
    """
    simulation_id = str(uuid.uuid4())

    # Create simulation record
    simulation = {
        "simulation_id": simulation_id,
        "status": "queued",
        "progress": 0,
        "params": request.dict(),
        "created_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "error": None,
        "result": None
    }

    simulations_db[simulation_id] = simulation

    # Queue background task
    background_tasks.add_task(run_simulation, simulation_id, request)

    console.print(f"[cyan]Created simulation {simulation_id}[/cyan]")

    return SimulationStatus(**simulation)


@app.get("/simulations/{simulation_id}", response_model=SimulationStatus)
async def get_simulation(simulation_id: str):
    """Get simulation status and results"""
    if simulation_id not in simulations_db:
        raise HTTPException(status_code=404, detail="Simulation not found")

    simulation = simulations_db[simulation_id]

    return SimulationStatus(**simulation)


@app.get("/simulations")
async def list_simulations(limit: int = 10):
    """List recent simulations"""
    sims = sorted(
        simulations_db.values(),
        key=lambda x: x['created_at'],
        reverse=True
    )[:limit]

    return {"simulations": sims}


@app.get("/plans/{plan_id}")
async def get_plan(plan_id: str):
    """Get plan details"""
    # Try to load from disk
    plan_file = OUTPUT_DIR / f"{plan_id}.json"

    if plan_file.exists():
        with open(plan_file) as f:
            return json.load(f)

    # Try memory cache
    if plan_id in plans_db:
        return plans_db[plan_id]

    raise HTTPException(status_code=404, detail="Plan not found")


@app.get("/plans")
async def list_plans():
    """List all available plans"""
    plans = []

    # Load from disk
    if OUTPUT_DIR.exists():
        for plan_file in OUTPUT_DIR.glob("plan_*.json"):
            with open(plan_file) as f:
                plan = json.load(f)
                plans.append({
                    "plan_id": plan.get('id', plan_file.stem),
                    "file": plan_file.name,
                    "created": plan.get('meta', {}).get('created_at', '')
                })

    return {"plans": plans}


@app.get("/certificates/{cert_id}")
async def get_certificate(cert_id: str):
    """Get diversity certificate"""
    # Try memory cache
    if cert_id in certificates_db:
        return certificates_db[cert_id]

    # Try to load from disk
    if CERTS_DIR.exists():
        for cert_file in CERTS_DIR.glob("*.json"):
            with open(cert_file) as f:
                cert = json.load(f)
                if cert.get('id') == cert_id or cert_file.stem == cert_id:
                    return cert

    raise HTTPException(status_code=404, detail="Certificate not found")


@app.get("/certificates")
async def list_certificates():
    """List all diversity certificates"""
    certs = []

    if CERTS_DIR.exists():
        for cert_file in CERTS_DIR.glob("*.json"):
            with open(cert_file) as f:
                cert = json.load(f)
                certs.append({
                    "cert_id": cert.get('id', cert_file.stem),
                    "file": cert_file.name,
                    "timestamp": cert.get('timestamp', '')
                })

    return {"certificates": certs}


# Background task runner
async def run_simulation(simulation_id: str, request: SimulationRequest):
    """
    Run simulation in background

    In production, this would be a separate worker process consuming
    from a message queue (RabbitMQ, SQS, etc.)
    """
    import subprocess
    import sys

    try:
        # Update status
        simulations_db[simulation_id]["status"] = "running"
        simulations_db[simulation_id]["progress"] = 10

        console.print(f"[cyan]Running simulation {simulation_id}...[/cyan]")

        # Run the actual simulation
        cmd = [
            sys.executable, "-m", "albion.runner",
            "--target-bn", str(request.target_revenue_bn),
            "--k", str(request.k),
            "--epsilon", str(request.epsilon),
            "--max-candidates", str(request.max_candidates),
            "--no-signing"  # Faster for demo
        ]

        simulations_db[simulation_id]["progress"] = 20

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            # Success
            simulations_db[simulation_id]["status"] = "complete"
            simulations_db[simulation_id]["progress"] = 100
            simulations_db[simulation_id]["completed_at"] = datetime.utcnow().isoformat()

            # Load results
            plans = []
            if OUTPUT_DIR.exists():
                for plan_file in sorted(OUTPUT_DIR.glob("plan_*.json")):
                    with open(plan_file) as f:
                        plan = json.load(f)
                        plans.append(plan)

            simulations_db[simulation_id]["result"] = {
                "plans": plans,
                "count": len(plans)
            }

            console.print(f"[green]✓ Simulation {simulation_id} complete![/green]")

        else:
            # Failed
            simulations_db[simulation_id]["status"] = "failed"
            simulations_db[simulation_id]["error"] = result.stderr
            console.print(f"[red]✗ Simulation {simulation_id} failed[/red]")

    except subprocess.TimeoutExpired:
        simulations_db[simulation_id]["status"] = "failed"
        simulations_db[simulation_id]["error"] = "Simulation timeout (5 minutes)"
        console.print(f"[red]✗ Simulation {simulation_id} timed out[/red]")

    except Exception as e:
        simulations_db[simulation_id]["status"] = "failed"
        simulations_db[simulation_id]["error"] = str(e)
        console.print(f"[red]✗ Simulation {simulation_id} error: {e}[/red]")


# Startup
@app.on_event("startup")
async def startup():
    """Startup tasks"""
    console.print("[bold]ALBION API Starting...[/bold]")
    console.print(f"[dim]Output directory: {OUTPUT_DIR}[/dim]")
    console.print(f"[dim]Certificates directory: {CERTS_DIR}[/dim]")

    # Ensure directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CERTS_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("shutdown")
async def shutdown():
    """Shutdown tasks"""
    console.print("[bold]ALBION API Shutting down...[/bold]")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "albion.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
