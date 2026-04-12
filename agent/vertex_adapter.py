"""Vertex AI (Google Cloud) adapter for Hermes Agent.

Provides authentication and configuration for Vertex AI's OpenAI-compatible
endpoint. This allows Hermes to use Gemini models via Google Cloud with
enterprise-grade rate limits and quotas.

Requires: pip install google-auth
"""

import logging
import os
from typing import Optional, Tuple

try:
    import google.auth
    import google.auth.transport.requests
    from google.oauth2 import service_account
except ImportError:
    google = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

# Default Vertex AI settings
DEFAULT_REGION = "us-central1"

def get_vertex_credentials(credentials_path: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
    """Get a Google Cloud access token and project ID.
    
    Returns (access_token, project_id) or (None, None) if auth fails.
    """
    if google is None:
        logger.warning("google-auth package not installed. Cannot use Vertex AI.")
        return None, None

    try:
        if credentials_path and os.path.exists(credentials_path):
            creds = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            project_id = creds.project_id
        else:
            creds, project_id = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )

        auth_req = google.auth.transport.requests.Request()
        creds.refresh(auth_req)
        return creds.token, project_id
    except Exception as e:
        logger.error(f"Failed to resolve Vertex AI credentials: {e}")
        return None, None

def build_vertex_base_url(project_id: str, region: str = DEFAULT_REGION) -> str:
    """Build the OpenAI-compatible base URL for Vertex AI."""
    return f"https://{region}-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{region}/endpoints/openapi"

def get_vertex_config(
    credentials_path: Optional[str] = None, 
    region: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Resolve API key (token) and base URL for Vertex AI.
    
    Returns (access_token, base_url) or (None, None).
    """
    token, project_id = get_vertex_credentials(credentials_path)
    if not token or not project_id:
        return None, None
        
    effective_region = region or DEFAULT_REGION
    base_url = build_vertex_base_url(project_id, effective_region)
    return token, base_url
