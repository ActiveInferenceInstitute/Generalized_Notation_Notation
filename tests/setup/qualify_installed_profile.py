"""Standalone wheel witnesses; run with python -I outside the checkout."""

import argparse
import importlib
import json
import os
import pathlib
import subprocess
import sys


def qualify(profile: str) -> None:
    import gnn
    import gnn.importers
    import gnn.parsers
    import gnn.types
    import gnn.validation.semantic_witness

    assert sys.flags.isolated
    assert (
        pathlib.Path(gnn.__file__)
        .resolve()
        .is_relative_to(pathlib.Path(sys.prefix).resolve())
    )
    if profile == "base":
        for module in ("app", "auth", "parity", "rate_limit", "responses"):
            try:
                importlib.import_module(f"gnn.api.{module}")
            except ImportError as exc:
                assert "generalized-notation-notation[api]" in str(exc), str(exc)
            else:
                raise AssertionError(f"base unexpectedly loaded api.{module}")
    if profile == "api":
        from fastapi.testclient import TestClient

        from gnn.api.app import create_app as create_run_app
        from gnn.api.server import create_app as create_job_app

        os.environ["GNN_API_ROOT"] = str(pathlib.Path.cwd().resolve())
        os.environ["GNN_RATE_LIMIT"] = "0"
        parity = {
            "/api/v1/validate",
            "/api/v1/parse",
            "/api/v1/extract",
            "/api/v1/render",
            "/api/v1/graph",
            "/api/v1/templates",
            "/api/v1/models",
            "/api/v1/preflight",
            "/api/v1/report",
            "/api/v1/reproduce",
            "/api/v1/pull",
        }
        for factory in (create_run_app, create_job_app):
            app = factory()
            paths = {route.path for route in app.routes}
            assert parity <= paths
            with TestClient(app) as client:
                response = client.get("/api/v1/health")
                assert response.status_code == 200, response.text
                response = client.get("/api/v1/templates")
                assert response.status_code == 200, response.text
                assert response.json()["status"] == "success"
                response = client.post(
                    "/api/v1/parse", json={"file_path": "missing.gnn.md"}
                )
                assert response.status_code == 404, response.text
                assert response.json()["status"] == "error"
            print(json.dumps(sorted(paths)))
    elif profile == "lsp":
        from gnn.lsp import PYGLS_AVAILABLE

        assert PYGLS_AVAILABLE
        requests = [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {"processId": None, "rootUri": None, "capabilities": {}},
            },
            {"jsonrpc": "2.0", "method": "initialized", "params": {}},
            {"jsonrpc": "2.0", "id": 2, "method": "shutdown"},
            {"jsonrpc": "2.0", "method": "exit"},
        ]
        wire = b""
        for request in requests:
            body = json.dumps(request).encode()
            wire += f"Content-Length: {len(body)}\r\n\r\n".encode() + body
        child = subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "from gnn.lsp import start_server; start_server()",
            ],
            input=wire,
            capture_output=True,
            timeout=30,
            check=True,
        )
        responses = {}
        remaining = child.stdout
        while remaining:
            header, separator, body = remaining.partition(b"\r\n\r\n")
            assert separator
            headers = dict(line.split(b":", 1) for line in header.split(b"\r\n"))
            length = int(
                next(
                    value
                    for key, value in headers.items()
                    if key.lower() == b"content-length"
                )
            )
            response = json.loads(body[:length])
            if "id" in response:
                responses[response["id"]] = response
            remaining = body[length:]
        assert "capabilities" in responses[1]["result"]
        assert responses[2] == {"jsonrpc": "2.0", "id": 2, "result": None}
        print(json.dumps(responses))
    print(f"{profile}: installed public imports passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("profile", choices=["base", "api", "lsp"])
    qualify(parser.parse_args().profile)
