"""Async REST transport for the game server.

Ports the semantics of the seed client's ``app/api_client.py`` (bearer auth,
reauth-once on token 401/403, retry-once on connection error) onto a single
pooled ``httpx.AsyncClient`` with explicit timeouts and structured results.

Unlike the seed client this returns a typed ``ApiResult`` instead of a
``(status, json)`` tuple, and distinguishes error *kinds* (auth/timeout/
network/http/decode) rather than substring-matching error text at the call site.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable

import httpx

log = logging.getLogger("agent.transport.rest")

# Reauth is triggered on 401/403 only when the error text looks token-related,
# matching the seed client's heuristic.
_TOKEN_HINTS = ("token", "expired", "invalid")


class ErrorKind(str, Enum):
    NONE = "none"
    HTTP = "http"          # non-2xx with a normal body
    AUTH = "auth"          # 401/403 that reauth could not resolve
    TIMEOUT = "timeout"    # connect/read timeout after retry
    NETWORK = "network"    # connection error after retry
    DECODE = "decode"      # body was not valid JSON


@dataclass(frozen=True)
class ApiResult:
    status: int
    data: dict[str, Any]
    error_kind: ErrorKind = ErrorKind.NONE

    @property
    def ok(self) -> bool:
        return self.error_kind is ErrorKind.NONE and 200 <= self.status < 300

    @property
    def error_message(self) -> str:
        err = self.data.get("error")
        if isinstance(err, dict):
            return str(err.get("message", ""))
        if err is not None:
            return str(err)
        return ""


def _extract_error_message(body: Any) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            return str(err.get("message", ""))
        if err is not None:
            return str(err)
    return ""


def _looks_like_token_error(body: Any) -> bool:
    msg = _extract_error_message(body).lower()
    return any(h in msg for h in _TOKEN_HINTS)


class GameRest:
    """Pooled async REST client with bearer auth and reauth/retry.

    ``reauth`` is an optional coroutine that re-logs-in and returns a fresh
    token (or None on failure). It is injected so this module has no dependency
    on the auth/login flow.
    """

    def __init__(
        self,
        api_base: str,
        token: str | None = None,
        *,
        reauth: Callable[[], Awaitable[str | None]] | None = None,
        connect_timeout: float = 5.0,
        read_timeout: float = 10.0,
        retry_backoff_s: float = 1.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_base = api_base.rstrip("/")
        self._token = token
        self._reauth = reauth
        self._retry_backoff_s = retry_backoff_s
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=40),
        )
        self._owns_client = client is None

    @property
    def token(self) -> str | None:
        return self._token

    def set_token(self, token: str | None) -> None:
        self._token = token

    def set_reauth(self, reauth: Callable[[], Awaitable[str | None]] | None) -> None:
        """Install (or replace) the reauth callback after construction.

        Useful when the callback needs a reference to this same client (e.g. a
        login coroutine that posts through it).
        """
        self._reauth = reauth

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> "GameRest":
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    @staticmethod
    def _safe_json(resp: httpx.Response) -> tuple[dict[str, Any], bool]:
        """Return (body, decoded_ok). On decode failure, synthesize an error body."""
        try:
            body = resp.json()
        except ValueError:
            text = (resp.text or "")[:300]
            return {"error": {"message": text or f"HTTP {resp.status_code} (no body)"}}, False
        if not isinstance(body, dict):
            # Some endpoints return bare lists; wrap so callers always see a dict.
            return {"data": body}, True
        return body, True

    async def _reauth_and_retry_headers(self) -> bool:
        """Attempt reauth; on success update the stored token. Returns success."""
        if self._reauth is None:
            return False
        log.warning("Token expired/invalid, attempting reauth...")
        try:
            new_token = await self._reauth()
        except Exception as exc:  # reauth must never crash the request path
            log.error("Reauth raised: %s", exc)
            return False
        if new_token:
            self._token = new_token
            return True
        return False

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> ApiResult:
        url = f"{self._api_base}{path}"
        last_exc: Exception | None = None

        for attempt in range(2):  # one connection-error retry, like the seed client
            try:
                resp = await self._client.request(
                    method, url, params=params, json=json, headers=self._headers()
                )
            except httpx.TimeoutException as exc:
                last_exc = exc
                if attempt == 0:
                    log.warning("Request timeout, retrying in %ss: %s %s", self._retry_backoff_s, method, path)
                    await asyncio.sleep(self._retry_backoff_s)
                    continue
                log.error("Request timed out after retry: %s %s", method, path)
                return ApiResult(504, {"error": {"message": f"timeout: {exc}"}}, ErrorKind.TIMEOUT)
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt == 0:
                    log.warning(
                        "Request failed (%s), retrying in %ss: %s %s",
                        exc.__class__.__name__, self._retry_backoff_s, method, path,
                    )
                    await asyncio.sleep(self._retry_backoff_s)
                    continue
                log.error("Request failed after retry: %s %s: %s", method, path, exc)
                return ApiResult(
                    502,
                    {"error": {"message": f"unreachable ({exc.__class__.__name__}): {str(exc)[:200]}"}},
                    ErrorKind.NETWORK,
                )

            body, decoded = self._safe_json(resp)

            # Token 401/403 → reauth once and replay this same request immediately.
            if resp.status_code in (401, 403) and _looks_like_token_error(body):
                if await self._reauth_and_retry_headers():
                    try:
                        resp = await self._client.request(
                            method, url, params=params, json=json, headers=self._headers()
                        )
                    except httpx.HTTPError as exc:
                        return ApiResult(
                            502,
                            {"error": {"message": f"post-reauth failure: {str(exc)[:200]}"}},
                            ErrorKind.NETWORK,
                        )
                    body, decoded = self._safe_json(resp)

            if resp.status_code in (401, 403):
                return ApiResult(resp.status_code, body, ErrorKind.AUTH)
            if not decoded:
                return ApiResult(resp.status_code, body, ErrorKind.DECODE)
            if not (200 <= resp.status_code < 300):
                return ApiResult(resp.status_code, body, ErrorKind.HTTP)
            return ApiResult(resp.status_code, body, ErrorKind.NONE)

        # Unreachable in practice; both loop exits return above.
        return ApiResult(502, {"error": {"message": f"exhausted: {last_exc}"}}, ErrorKind.NETWORK)

    async def get(self, path: str, params: dict[str, Any] | None = None) -> ApiResult:
        return await self.request("GET", path, params=params)

    async def post(self, path: str, json: dict[str, Any] | None = None) -> ApiResult:
        return await self.request("POST", path, json=json or {})

    async def delete(self, path: str, json: dict[str, Any] | None = None) -> ApiResult:
        return await self.request("DELETE", path, json=json)
