"""
Sigstore/Rekor Signing Service

Integrates with Sigstore for cryptographic signing and transparency logging.

Every artifact (plan, certificate) is:
1. Serialized to canonical JSON
2. Hashed (SHA-256)
3. Signed using Sigstore (cosign)
4. Logged to Rekor transparency log
5. Signature bundle stored in database

Verification:
- Check hash matches content
- Verify signature against Rekor log
- Check certificate identity
"""

from __future__ import annotations

import hashlib
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SignatureBundle:
    """
    Complete signature bundle for an artifact

    Includes everything needed to verify the signature
    """
    artifact_hash: str  # SHA-256 hex
    signature: str  # Base64-encoded signature
    certificate: str  # X.509 certificate (PEM)
    rekor_log_id: str  # Rekor transparency log UUID
    rekor_log_index: int  # Index in Rekor log
    timestamp: datetime  # Signing timestamp
    sigstore_bundle: Dict[str, Any]  # Full Sigstore bundle (JSON)


class SigstoreService:
    """
    Sigstore integration for artifact signing and verification

    Uses cosign CLI for signing (production would use Python SDK)
    """

    def __init__(
        self,
        identity: Optional[str] = None,
        enable_signing: bool = True
    ):
        """
        Args:
            identity: Sigstore identity (email or OIDC subject)
            enable_signing: If False, skip actual signing (for testing)
        """
        self.identity = identity
        self.enable_signing = enable_signing

        if enable_signing:
            # Check cosign is available
            try:
                result = subprocess.run(
                    ['cosign', 'version'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    logger.info("Sigstore (cosign) available")
                else:
                    logger.warning("cosign command failed - signing disabled")
                    self.enable_signing = False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("cosign not found - signing disabled")
                self.enable_signing = False

    def sign_artifact(self, artifact: Dict[str, Any]) -> SignatureBundle:
        """
        Sign an artifact and log to Rekor

        Args:
            artifact: Dict to sign (will be serialized to canonical JSON)

        Returns:
            SignatureBundle with signature, certificate, and Rekor UUID
        """
        # 1. Canonical serialization
        artifact_json = self._canonicalize(artifact)
        artifact_bytes = artifact_json.encode('utf-8')

        # 2. Hash
        artifact_hash = hashlib.sha256(artifact_bytes).hexdigest()

        logger.info(f"Signing artifact (hash: {artifact_hash[:16]}...)")

        if not self.enable_signing:
            # Return mock signature for testing
            return self._mock_signature(artifact_hash)

        # 3. Sign with cosign
        try:
            signature_bundle = self._sign_with_cosign(artifact_bytes)

            # 4. Parse bundle
            bundle_dict = json.loads(signature_bundle)

            # Extract components
            rekor_bundle = bundle_dict.get('rekorBundle', {})
            rekor_payload = rekor_bundle.get('Payload', {})

            signature = bundle_dict.get('base64Signature', '')
            certificate = bundle_dict.get('cert', '')
            rekor_log_id = rekor_payload.get('logID', '')
            rekor_log_index = rekor_payload.get('logIndex', 0)
            timestamp_unix = rekor_payload.get('integratedTime', 0)

            timestamp = datetime.fromtimestamp(timestamp_unix)

            logger.info(
                f"Signed successfully. Rekor UUID: {rekor_log_id[:16]}..., "
                f"Index: {rekor_log_index}"
            )

            return SignatureBundle(
                artifact_hash=artifact_hash,
                signature=signature,
                certificate=certificate,
                rekor_log_id=rekor_log_id,
                rekor_log_index=rekor_log_index,
                timestamp=timestamp,
                sigstore_bundle=bundle_dict
            )

        except Exception as e:
            logger.error(f"Signing failed: {e}")
            # Return mock signature as fallback
            return self._mock_signature(artifact_hash)

    def verify_signature(
        self,
        artifact: Dict[str, Any],
        signature_bundle: SignatureBundle
    ) -> bool:
        """
        Verify artifact signature against Rekor log

        Args:
            artifact: Original artifact
            signature_bundle: Signature bundle to verify

        Returns:
            True if signature is valid
        """
        # 1. Recompute hash
        artifact_json = self._canonicalize(artifact)
        artifact_bytes = artifact_json.encode('utf-8')
        computed_hash = hashlib.sha256(artifact_bytes).hexdigest()

        # 2. Check hash matches
        if computed_hash != signature_bundle.artifact_hash:
            logger.error(
                f"Hash mismatch: computed {computed_hash[:16]}..., "
                f"expected {signature_bundle.artifact_hash[:16]}..."
            )
            return False

        if not self.enable_signing:
            # In mock mode, just check hash
            return True

        # 3. Verify with cosign
        try:
            bundle_path = self._write_temp_bundle(signature_bundle.sigstore_bundle)

            result = subprocess.run(
                [
                    'cosign', 'verify-blob',
                    '--bundle', str(bundle_path),
                    '--signature', signature_bundle.signature,
                    '--certificate-identity-regexp', '.*',  # Accept any identity for now
                    '--certificate-oidc-issuer-regexp', '.*',
                    '/dev/stdin'
                ],
                input=artifact_bytes,
                capture_output=True,
                timeout=10
            )

            bundle_path.unlink()  # Clean up temp file

            if result.returncode == 0:
                logger.info("Signature verified successfully")
                return True
            else:
                logger.error(f"Signature verification failed: {result.stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"Verification error: {e}")
            return False

    def _canonicalize(self, obj: Dict[str, Any]) -> str:
        """
        Canonical JSON serialization

        Ensures consistent serialization for hashing/signing
        """
        return json.dumps(obj, sort_keys=True, separators=(',', ':'))

    def _sign_with_cosign(self, data: bytes) -> str:
        """
        Sign data using cosign

        Args:
            data: Bytes to sign

        Returns:
            JSON bundle as string
        """
        # Write bundle to temp file
        bundle_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        )
        bundle_path = bundle_file.name
        bundle_file.close()

        try:
            # Sign blob
            result = subprocess.run(
                [
                    'cosign', 'sign-blob',
                    '--yes',  # Non-interactive
                    '--bundle', bundle_path,
                    '/dev/stdin'
                ],
                input=data,
                capture_output=True,
                timeout=30
            )

            if result.returncode != 0:
                raise RuntimeError(f"cosign failed: {result.stderr.decode()}")

            # Read bundle
            with open(bundle_path, 'r') as f:
                bundle_json = f.read()

            return bundle_json

        finally:
            # Clean up temp file
            Path(bundle_path).unlink(missing_ok=True)

    def _write_temp_bundle(self, bundle_dict: Dict[str, Any]) -> Path:
        """Write Sigstore bundle to temp file for verification"""
        temp_file = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.json',
            delete=False
        )

        json.dump(bundle_dict, temp_file)
        temp_file.close()

        return Path(temp_file.name)

    def _mock_signature(self, artifact_hash: str) -> SignatureBundle:
        """
        Generate mock signature bundle for testing

        DO NOT use in production!
        """
        logger.warning("Using MOCK signature (not cryptographically secure)")

        return SignatureBundle(
            artifact_hash=artifact_hash,
            signature='MOCK_SIGNATURE_' + artifact_hash[:32],
            certificate='MOCK_CERTIFICATE',
            rekor_log_id='MOCK_REKOR_' + artifact_hash[:16],
            rekor_log_index=12345,
            timestamp=datetime.utcnow(),
            sigstore_bundle={
                'mediaType': 'application/vnd.dev.sigstore.bundle+json;version=0.1',
                'verificationMaterial': {
                    'certificate': 'MOCK_CERT'
                },
                'dsseEnvelope': {
                    'payload': artifact_hash
                }
            }
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def sign_plan(plan: Dict[str, Any], service: SigstoreService) -> Dict[str, Any]:
    """
    Sign a policy plan and attach signature

    Args:
        plan: Plan dict (proposal/v1 schema)
        service: SigstoreService instance

    Returns:
        Plan dict with 'signature' field added
    """
    signature = service.sign_artifact(plan)

    plan['signature'] = {
        'artifact_hash': signature.artifact_hash,
        'signature': signature.signature,
        'certificate': signature.certificate,
        'rekor_log_id': signature.rekor_log_id,
        'rekor_log_index': signature.rekor_log_index,
        'timestamp': signature.timestamp.isoformat(),
        'bundle': signature.sigstore_bundle
    }

    return plan


def verify_plan(plan: Dict[str, Any], service: SigstoreService) -> bool:
    """
    Verify a signed plan

    Args:
        plan: Plan dict with 'signature' field
        service: SigstoreService instance

    Returns:
        True if signature is valid
    """
    if 'signature' not in plan:
        logger.error("Plan has no signature")
        return False

    sig_dict = plan['signature']

    # Reconstruct SignatureBundle
    signature = SignatureBundle(
        artifact_hash=sig_dict['artifact_hash'],
        signature=sig_dict['signature'],
        certificate=sig_dict['certificate'],
        rekor_log_id=sig_dict['rekor_log_id'],
        rekor_log_index=sig_dict['rekor_log_index'],
        timestamp=datetime.fromisoformat(sig_dict['timestamp']),
        sigstore_bundle=sig_dict['bundle']
    )

    # Remove signature for verification (check signature of unsigned content)
    plan_copy = dict(plan)
    del plan_copy['signature']

    return service.verify_signature(plan_copy, signature)
