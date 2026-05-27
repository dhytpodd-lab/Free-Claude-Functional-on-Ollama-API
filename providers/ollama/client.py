"""Ollama provider implementation."""

import httpx

from providers.anthropic_messages import AnthropicMessagesTransport
from providers.base import ProviderConfig
from providers.defaults import OLLAMA_DEFAULT_BASE
from providers.model_listing import extract_ollama_model_ids


class OllamaProvider(AnthropicMessagesTransport):
    """Ollama provider using native Anthropic Messages API."""

    def __init__(self, config: ProviderConfig):
        super().__init__(
            config,
            provider_name="OLLAMA",
            default_base_url=OLLAMA_DEFAULT_BASE,
        )
        self._api_key = config.api_key or "ollama"

    def _auth_headers(self) -> dict[str, str]:
        """Return optional proxy/cloud auth headers for remote Ollama hosts."""
        if not self._api_key or self._api_key == "ollama":
            return {}
        return {"Authorization": f"Bearer {self._api_key}"}

    def _request_headers(self) -> dict[str, str]:
        """Return headers for Ollama's native Anthropic messages endpoint."""
        headers = super()._request_headers()
        headers.update(self._auth_headers())
        return headers

    def _model_list_headers(self) -> dict[str, str]:
        """Return headers for Ollama's native model-list endpoint."""
        return self._auth_headers()

    async def _send_stream_request(self, body: dict) -> httpx.Response:
        """Create a streaming native Anthropic messages response."""
        request = self._client.build_request(
            "POST",
            "/v1/messages",
            json=body,
            headers=self._request_headers(),
        )
        return await self._client.send(request, stream=True)

    async def _send_model_list_request(self) -> httpx.Response:
        """Query Ollama's native local model-list endpoint."""
        return await self._client.get(
            f"{self._base_url}/api/tags",
            headers=self._model_list_headers(),
        )

    def _extract_model_ids_from_model_list_payload(
        self, payload: object
    ) -> frozenset[str]:
        return extract_ollama_model_ids(payload, provider_name=self._provider_name)
