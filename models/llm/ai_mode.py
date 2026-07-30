"""Resolve a parsed Flyfus AI Mode reference."""

import requests

from models.llm.model_route import ModelRouteResult


def apply_ai_mode(
    model: str,
    model_parameters: dict,
    reference: str | None,
    credentials: dict,
) -> ModelRouteResult:
    """Resolve and apply a previously parsed AI Mode reference."""
    fallback = ModelRouteResult(model=model, parameters=dict(model_parameters))
    if reference is None:
        return fallback

    base_url = str(credentials.get("geo_prompt_render_url") or "").strip().rstrip("/")
    if not base_url:
        return fallback

    try:
        response = requests.post(
            f"{base_url}/dify_admin/ai_mode/resolve_reference",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {str(credentials.get('geo_prompt_api_key') or '').strip()}",
            },
            json={"reference": reference},
            timeout=(10, 60),
        )
        payload = response.json()
        config = payload["data"]["config"]
        target_model = config["model"]
        target_parameters = config["parameters"]
    except (requests.RequestException, KeyError, TypeError, ValueError):
        return fallback

    if (
        response.status_code != 200
        or payload.get("code") != 200
        or not isinstance(target_model, str)
        or not isinstance(target_parameters, dict)
    ):
        return fallback
    return ModelRouteResult(model=target_model, parameters=target_parameters, applied=True)
