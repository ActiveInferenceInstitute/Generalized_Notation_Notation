"""Local-only WebSocket message contracts for reactive GUI synchronization."""

from __future__ import annotations

import asyncio
import copy
import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

GUI_WEBSOCKET_MESSAGE_TYPES = frozenset(
    {
        "model.load",
        "matrix.patch",
        "validation.result",
        "model.export",
        "simulation.tick",
        "state.update",
        "matrix.heatmap",
        "error",
    }
)


@dataclass(frozen=True)
class GUIWebSocketMessage:
    """Validated JSON message exchanged by the Step 22 GUI bridge."""

    type: str
    payload: Dict[str, Any] = field(default_factory=dict)
    request_id: str | None = None

    def to_json(self) -> str:
        """Serialize to a JSON message."""
        validate_gui_message_type(self.type)
        data: Dict[str, Any] = {"type": self.type, "payload": self.payload}
        if self.request_id is not None:
            data["request_id"] = self.request_id
        return json.dumps(data, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> GUIWebSocketMessage:
        """Parse and validate a JSON message."""
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("GUI WebSocket message must be a JSON object")
        msg_type = data.get("type")
        if not isinstance(msg_type, str):
            raise ValueError("GUI WebSocket message requires a string 'type'")
        validate_gui_message_type(msg_type)
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            raise ValueError("GUI WebSocket message payload must be an object")
        request_id = data.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise ValueError("GUI WebSocket request_id must be a string when present")
        return cls(type=msg_type, payload=payload, request_id=request_id)


def validate_gui_message_type(message_type: str) -> None:
    """Raise if ``message_type`` is not part of the public GUI contract."""
    if message_type not in GUI_WEBSOCKET_MESSAGE_TYPES:
        allowed = ", ".join(sorted(GUI_WEBSOCKET_MESSAGE_TYPES))
        raise ValueError(
            f"Unsupported GUI WebSocket message type '{message_type}'. Allowed: {allowed}"
        )


def build_initial_messages(
    model_payloads: Iterable[Dict[str, Any]],
) -> List[GUIWebSocketMessage]:
    """Build initial ``model.load`` messages for converted oxdraw/Mermaid models."""
    return [
        GUIWebSocketMessage(type="model.load", payload=dict(payload))
        for payload in model_payloads
    ]


@dataclass
class GUIBridgeState:
    """In-memory local bridge state for Step 22 reactive GUI sessions."""

    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    patches: List[Dict[str, Any]] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[Dict[str, Any]] = field(default_factory=list)

    def apply_message(self, message: GUIWebSocketMessage) -> GUIWebSocketMessage | None:
        """Apply a validated GUI message and return an optional response."""
        if message.type == "model.load":
            model_id = _model_id_from_payload(message.payload, len(self.models) + 1)
            payload = copy.deepcopy(message.payload)
            payload.setdefault("model_id", model_id)
            self.models[model_id] = payload
            return GUIWebSocketMessage(
                type="validation.result",
                request_id=message.request_id,
                payload={"model_id": model_id, "valid": True, "errors": []},
            )
        if message.type == "matrix.patch":
            return self._apply_matrix_patch(message)
        if message.type == "validation.result":
            self.validation_results.append(copy.deepcopy(message.payload))
            return None
        if message.type == "model.export":
            requested_model = message.payload.get("model_id")
            models = (
                {requested_model: self.models[requested_model]}
                if requested_model in self.models
                else self.models
            )
            return GUIWebSocketMessage(
                type="model.export",
                request_id=message.request_id,
                payload={
                    "format": message.payload.get("format", "json"),
                    "models": copy.deepcopy(models),
                },
            )
        if message.type == "error":
            self.errors.append(copy.deepcopy(message.payload))
            return None
        return None

    def _apply_matrix_patch(self, message: GUIWebSocketMessage) -> GUIWebSocketMessage:
        payload = message.payload
        model_id = payload.get("model_id")
        if model_id is None and len(self.models) == 1:
            model_id = next(iter(self.models))
        matrix_name = payload.get("matrix")
        path = payload.get("path")
        if model_id not in self.models:
            return _bridge_error(
                "Unknown model_id for matrix.patch", message.request_id
            )
        if not isinstance(matrix_name, str) or not isinstance(path, list):
            return _bridge_error(
                "matrix.patch requires string matrix and list path",
                message.request_id,
            )
        model = self.models[model_id]
        matrices = model.setdefault("matrices", {})
        if not isinstance(matrices, dict):
            return _bridge_error(
                "model matrices payload must be an object", message.request_id
            )
        if matrix_name not in matrices:
            matrices[matrix_name] = []
        try:
            _patch_nested_value(matrices, [matrix_name, *path], payload.get("value"))
        except (IndexError, TypeError, ValueError) as exc:
            return _bridge_error(str(exc), message.request_id)
        patch = copy.deepcopy(payload)
        patch["model_id"] = model_id
        self.patches.append(patch)
        result = {
            "model_id": model_id,
            "matrix": matrix_name,
            "valid": True,
            "patch_count": len(self.patches),
            "errors": [],
        }
        self.validation_results.append(result)
        return GUIWebSocketMessage(
            type="validation.result", request_id=message.request_id, payload=result
        )


async def run_local_gui_bridge(
    host: str, port: int, initial_messages: Iterable[GUIWebSocketMessage]
) -> None:
    """Run an optional local-only WebSocket bridge.

    The optional ``websockets`` package is imported only here so headless tests and
    artifact generation do not require a live WebSocket runtime.
    """
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError(
            "GUI WebSocket bridge is local-only; bind to 127.0.0.1, localhost, or ::1"
        )
    try:
        import websockets
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional websockets package to launch the GUI bridge"
        ) from exc

    initial = list(initial_messages)
    state = GUIBridgeState()
    for message in initial:
        state.apply_message(message)
    messages = [message.to_json() for message in initial]

    async def handler(websocket: Any) -> None:
        for message in messages:
            await websocket.send(message)
        async for raw in websocket:
            response = state.apply_message(GUIWebSocketMessage.from_json(raw))
            if response is not None:
                await websocket.send(response.to_json())

    async with websockets.serve(handler, host, port):
        await asyncio.Future()


def _model_id_from_payload(payload: Dict[str, Any], fallback_index: int) -> str:
    """Derive a stable model id from a model.load payload."""
    for key in ("model_id", "model_name", "name", "path"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return f"model-{fallback_index}"


def _bridge_error(message: str, request_id: str | None = None) -> GUIWebSocketMessage:
    """Build a GUI bridge error response."""
    return GUIWebSocketMessage(
        type="error",
        request_id=request_id,
        payload={"message": message},
    )


def _patch_nested_value(container: Dict[str, Any], path: List[Any], value: Any) -> None:
    """Patch a nested dict/list matrix value in place."""
    if not path:
        raise ValueError("Patch path cannot be empty")
    current: Any = container
    for key in path[:-1]:
        if isinstance(current, dict):
            current = current[key]
        elif isinstance(current, list) and isinstance(key, int):
            current = current[key]
        else:
            raise TypeError(f"Cannot traverse patch segment {key!r}")
    final_key = path[-1]
    if isinstance(current, dict):
        current[final_key] = value
    elif isinstance(current, list) and isinstance(final_key, int):
        current[final_key] = value
    else:
        raise TypeError(f"Cannot set patch segment {final_key!r}")
