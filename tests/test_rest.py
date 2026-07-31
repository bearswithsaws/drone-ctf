"""Tests for the async REST transport — network-free via respx."""

from __future__ import annotations

import httpx
import pytest
import respx

from agent.transport.rest import ApiResult, ErrorKind, GameRest

BASE = "https://game.test/api/v1"


async def _make(token: str = "t0", **kw) -> GameRest:
    return GameRest(BASE, token=token, retry_backoff_s=0.0, **kw)


@respx.mock
async def test_get_success() -> None:
    respx.get(f"{BASE}/drones").mock(return_value=httpx.Response(200, json={"drones": [1, 2]}))
    rest = await _make()
    res = await rest.get("/drones")
    assert res.ok
    assert res.status == 200
    assert res.data == {"drones": [1, 2]}
    assert res.error_kind is ErrorKind.NONE
    # Bearer token is attached.
    assert respx.calls.last.request.headers["Authorization"] == "Bearer t0"
    await rest.aclose()


@respx.mock
async def test_bare_list_body_is_wrapped() -> None:
    respx.get(f"{BASE}/list").mock(return_value=httpx.Response(200, json=[1, 2, 3]))
    rest = await _make()
    res = await rest.get("/list")
    assert res.ok
    assert res.data == {"data": [1, 2, 3]}
    await rest.aclose()


@respx.mock
async def test_http_error_is_classified() -> None:
    respx.post(f"{BASE}/x").mock(
        return_value=httpx.Response(400, json={"error": {"message": "bad request"}})
    )
    rest = await _make()
    res = await rest.post("/x", json={})
    assert not res.ok
    assert res.status == 400
    assert res.error_kind is ErrorKind.HTTP
    assert res.error_message == "bad request"
    await rest.aclose()


@respx.mock
async def test_decode_error_on_non_json() -> None:
    respx.get(f"{BASE}/html").mock(return_value=httpx.Response(200, text="<html>oops</html>"))
    rest = await _make()
    res = await rest.get("/html")
    assert res.error_kind is ErrorKind.DECODE
    assert "oops" in res.error_message
    await rest.aclose()


@respx.mock
async def test_timeout_then_success_retry() -> None:
    route = respx.get(f"{BASE}/flaky")
    route.side_effect = [
        httpx.ConnectTimeout("slow"),
        httpx.Response(200, json={"ok": True}),
    ]
    rest = await _make()
    res = await rest.get("/flaky")
    assert res.ok
    assert route.call_count == 2
    await rest.aclose()


@respx.mock
async def test_timeout_twice_gives_timeout_kind() -> None:
    respx.get(f"{BASE}/dead").mock(side_effect=httpx.ReadTimeout("gone"))
    rest = await _make()
    res = await rest.get("/dead")
    assert res.error_kind is ErrorKind.TIMEOUT
    assert res.status == 504
    await rest.aclose()


@respx.mock
async def test_network_error_twice_gives_network_kind() -> None:
    respx.get(f"{BASE}/dead").mock(side_effect=httpx.ConnectError("no route"))
    rest = await _make()
    res = await rest.get("/dead")
    assert res.error_kind is ErrorKind.NETWORK
    assert res.status == 502
    await rest.aclose()


@respx.mock
async def test_reauth_on_token_401_then_replay() -> None:
    calls = {"n": 0}

    def responder(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            assert request.headers["Authorization"] == "Bearer stale"
            return httpx.Response(401, json={"error": {"message": "token expired"}})
        # After reauth the fresh token must be used.
        assert request.headers["Authorization"] == "Bearer fresh"
        return httpx.Response(200, json={"ok": True})

    respx.get(f"{BASE}/secure").mock(side_effect=responder)

    async def reauth() -> str:
        return "fresh"

    rest = await _make(token="stale", reauth=reauth)
    res = await rest.get("/secure")
    assert res.ok
    assert calls["n"] == 2
    assert rest.token == "fresh"
    await rest.aclose()


@respx.mock
async def test_auth_error_when_reauth_fails() -> None:
    respx.get(f"{BASE}/secure").mock(
        return_value=httpx.Response(403, json={"error": {"message": "invalid token"}})
    )

    async def reauth() -> None:
        return None  # reauth failed

    rest = await _make(token="stale", reauth=reauth)
    res = await rest.get("/secure")
    assert res.error_kind is ErrorKind.AUTH
    assert res.status == 403
    await rest.aclose()


@respx.mock
async def test_non_token_403_is_not_reauthed() -> None:
    reauthed = {"called": False}

    async def reauth() -> str:
        reauthed["called"] = True
        return "fresh"

    respx.get(f"{BASE}/forbidden").mock(
        return_value=httpx.Response(403, json={"error": {"message": "not your team"}})
    )
    rest = await _make(token="t0", reauth=reauth)
    res = await rest.get("/forbidden")
    assert res.error_kind is ErrorKind.AUTH
    assert reauthed["called"] is False
    await rest.aclose()


def test_apiresult_helpers() -> None:
    r = ApiResult(200, {"x": 1})
    assert r.ok
    assert r.error_message == ""
    r2 = ApiResult(500, {"error": "boom"}, ErrorKind.HTTP)
    assert not r2.ok
    assert r2.error_message == "boom"
