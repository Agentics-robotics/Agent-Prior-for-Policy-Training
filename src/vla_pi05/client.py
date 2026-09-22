"""Minimal synchronous websocket client for ``vla_pi05.serve`` (no torch/lerobot needed).

Copy this file together with ``msgpack_numpy.py`` to the robot PC, or use
``openpi_client.WebsocketClientPolicy`` which speaks the same protocol.

    from vla_pi05.client import PolicyClient
    c = PolicyClient("cluster-host", 8000)
    print(c.metadata)
    out = c.infer({"images": {"base_0_rgb": img_third, "left_wrist_0_rgb": img_wrist},
                   "state": state8, "instruction": "flip the fried egg"})
    out["targets"]  # (50, 8) absolute joint targets + gripper opening
"""
from __future__ import annotations

import logging
import time

from . import msgpack_numpy

log = logging.getLogger(__name__)


class PolicyClient:
    def __init__(self, host: str = "127.0.0.1", port: int = 8000, connect_timeout_s: float = 60.0):
        self.uri = f"ws://{host}:{port}"
        self._ws = None
        self.metadata = {}
        self._packer = msgpack_numpy.Packer()
        self._connect(connect_timeout_s)

    def _connect(self, timeout_s: float):
        from websockets.sync.client import connect

        deadline = time.monotonic() + timeout_s
        last = None
        while True:
            try:
                self._ws = connect(self.uri, compression=None, max_size=None)
                self.metadata = msgpack_numpy.unpackb(self._ws.recv())
                return
            except ConnectionRefusedError as e:
                last = e
                if time.monotonic() > deadline:
                    raise TimeoutError(f"could not connect to {self.uri}: {last}")
                time.sleep(1.0)

    def infer(self, obs: dict) -> dict:
        self._ws.send(self._packer.pack(obs))
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"server error:\n{resp}")
        return msgpack_numpy.unpackb(resp)

    def reset(self) -> None:
        self.infer({"type": "reset"})

    def close(self) -> None:
        if self._ws is not None:
            self._ws.close()
            self._ws = None
