"""Websocket policy server, wire-compatible with ``openpi_client.WebsocketClientPolicy``.

Protocol (binary msgpack frames, NumPy arrays encoded as in openpi):
    server -> client on connect : metadata dict
    client -> server            : observation dict, either
                                    {"images": {"base_0_rgb": u8[H,W,3], "left_wrist_0_rgb": u8[H,W,3]},
                                     "state": f32[8], "instruction": "flip the fried egg"}
                                  or openpi-style {"observation/base_0_rgb": ..., "observation/state": ..., "prompt": ...}
    server -> client            : {"actions": f32[chunk, action_dim], "targets": f32[chunk, action_dim],
                                   "server_timing": {"infer_ms": float}}
    GET /healthz                : "OK"
"""
from __future__ import annotations

import asyncio
import http
import logging
import time
import traceback

import numpy as np

from . import msgpack_numpy
from .inference import Pi05Runner, obs_from_message

log = logging.getLogger(__name__)


class PolicyServer:
    def __init__(self, runner: Pi05Runner, host: str = "0.0.0.0", port: int = 8000):
        self.runner = runner
        self.host = host
        self.port = port
        self.metadata = {"protocol": "vla_pi05/1", **runner.metadata()}

    def infer(self, obs: dict) -> dict:
        images, state, instruction = obs_from_message(obs)
        chunk = self.runner.predict(images, state, instruction)
        return {
            "actions": chunk.astype(np.float32),
            "targets": self.runner.chunk_to_targets(chunk, state).astype(np.float32),
        }

    async def _handler(self, websocket):
        import websockets

        log.info("connection from %s", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        await websocket.send(packer.pack(self.metadata))
        while True:
            try:
                t0 = time.monotonic()
                obs = msgpack_numpy.unpackb(await websocket.recv())
                if isinstance(obs, dict) and obs.get("type") == "reset":
                    self.runner.reset()
                    await websocket.send(packer.pack({"ok": True}))
                    continue
                t1 = time.monotonic()
                out = self.infer(obs)
                out["server_timing"] = {"infer_ms": (time.monotonic() - t1) * 1000, "total_ms": (time.monotonic() - t0) * 1000}
                await websocket.send(packer.pack(out))
            except websockets.ConnectionClosed:
                log.info("connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error. Traceback included in previous frame.")
                raise

    async def run(self):
        import websockets.asyncio.server as ws_server

        def health(connection, request):
            if request.path == "/healthz":
                return connection.respond(http.HTTPStatus.OK, "OK\n")
            return None

        async with ws_server.serve(self._handler, self.host, self.port, compression=None, max_size=None, process_request=health) as server:
            log.info("serving on ws://%s:%d", self.host, self.port)
            await server.serve_forever()

    def serve_forever(self):
        asyncio.run(self.run())
