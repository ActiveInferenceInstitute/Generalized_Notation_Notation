"""Two-listener startup and shutdown through the CLI."""

from __future__ import annotations

import http.client
import json
import socket
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any

import pytest

from gnn.cli import EXIT_ERROR, EXIT_SUCCESS, main

HOST = "127.0.0.1"


@pytest.fixture
def api_ports() -> Iterator[tuple[socket.socket, socket.socket]]:
    for _ in range(20):
        with socket.socket() as runs, socket.socket() as jobs:
            runs.bind((HOST, 0))
            try:
                jobs.bind((HOST, runs.getsockname()[1] + 1))
            except (OSError, OverflowError):
                continue
            yield runs, jobs
            return
    pytest.fail("Could not reserve consecutive loopback ports")


def test_both_rejects_port_without_room_for_jobs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert main(["serve", "--surface", "both", "--port", "65535"]) == EXIT_ERROR
    assert "base port from 1 to 65534" in caplog.text


@pytest.mark.parametrize("blocked", [0, 1], ids=["runs", "jobs"])
def test_both_releases_peer_on_bind_failure(
    api_ports: tuple[socket.socket, socket.socket],
    blocked: int,
    caplog: pytest.LogCaptureFixture,
) -> None:
    port = api_ports[0].getsockname()[1]
    api_ports[blocked].listen()
    peer = api_ports[1 - blocked]
    peer_port = peer.getsockname()[1]
    peer.close()

    assert main(["serve", "--surface", "both", "--port", str(port)]) == EXIT_ERROR
    assert f"Could not bind both API ports {port} and {port + 1}" in caplog.text
    with socket.socket() as probe:
        probe.bind((HOST, peer_port))


def test_both_stops_runs_when_jobs_fails(
    api_ports: tuple[socket.socket, socket.socket],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import uvicorn

    port = api_ports[0].getsockname()[1]
    for reserved in api_ports:
        reserved.close()
    run_server = uvicorn.Server.run

    def run(server: uvicorn.Server, **kwargs: Any) -> None:
        if server.config.port == port + 1:
            raise SystemExit(1)
        run_server(server, **kwargs)

    monkeypatch.setattr(uvicorn.Server, "run", run)
    assert main(["serve", "--surface", "both", "--port", str(port)]) == EXIT_ERROR
    assert "Jobs API listener failed" in caplog.text


def test_both_serves_and_stops_each_api(
    api_ports: tuple[socket.socket, socket.socket], monkeypatch: pytest.MonkeyPatch
) -> None:
    import uvicorn

    port = api_ports[0].getsockname()[1]
    for reserved in api_ports:
        reserved.close()
    server_type = uvicorn.Server
    servers: dict[int, uvicorn.Server] = {}
    constructed = Event()

    def capture(config: uvicorn.Config) -> uvicorn.Server:
        server = server_type(config)
        servers[config.port] = server
        if len(servers) == 2:
            constructed.set()
        return server

    monkeypatch.setattr(uvicorn, "Server", capture)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            main, ["serve", "--surface", "both", "--port", str(port)]
        )
        try:
            assert constructed.wait(timeout=5)
            for service_port, route in [
                (port, "/api/v1/run"),
                (port + 1, "/api/v1/process"),
            ]:
                deadline = time.monotonic() + 5.0
                while True:
                    connection = http.client.HTTPConnection(
                        HOST, service_port, timeout=0.5
                    )
                    try:
                        connection.request("GET", "/openapi.json")
                        response = connection.getresponse()
                        assert response.status == 200
                        assert route in json.loads(response.read())["paths"]
                        break
                    except OSError:
                        assert not result.done(), result.result()
                        assert time.monotonic() < deadline, (
                            f"Port {service_port} did not start"
                        )
                        time.sleep(0.05)
                    finally:
                        connection.close()
        finally:
            for server in list(servers.values()):
                if server.config.port == port:
                    server.should_exit = True
        assert result.result(timeout=5) == EXIT_SUCCESS

    for service_port in (port, port + 1):
        with (
            pytest.raises(OSError),
            socket.create_connection((HOST, service_port), timeout=0.5),
        ):
            pytest.fail(f"Port {service_port} was still serving after CLI shutdown")
