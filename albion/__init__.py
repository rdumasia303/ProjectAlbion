"""
ALBION Engine: The National Options Engine

A production-grade platform for lawful, diverse, near-optimal national policy simulation.

Components:
- agents: TaxBenefitAgent, MacroAgent, ClimateAgent, DistributionAgent
- dnos: Diverse Near-Optimal Set selector (Facility Location + DPP)
- gates: OPA policy gate integration
- sign: Sigstore/Rekor signing and verification
- api: FastAPI async interface
"""

__version__ = "1.0.0"
__author__ = "ALBION Team"

from albion.models import Policy, Constitution, ImpactResult, DiversityCertificate

__all__ = ["Policy", "Constitution", "ImpactResult", "DiversityCertificate"]
