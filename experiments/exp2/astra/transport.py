"""Bound provider concurrency without changing or retrying any API request."""
from contextlib import contextmanager
import fcntl
from functools import wraps
from pathlib import Path
import time

from appl.agent import ResponsesClient
from appl.io import event


@contextmanager
def capacity(root, slots):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    handles = [(root / f'slot_{i}.lock').open('a') for i in range(slots)]
    owned = None
    try:
        while owned is None:
            for index, handle in enumerate(handles):
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    owned = index
                    break
                except BlockingIOError:
                    continue
            if owned is None:
                time.sleep(0.1)
        yield owned
    finally:
        if owned is not None:
            fcntl.flock(handles[owned], fcntl.LOCK_UN)
        for handle in handles:
            handle.close()


def install_episode_gate(study_root, episode_root, slots=4):
    original = ResponsesClient.respond
    if getattr(original, '_appl_capacity_gate', False):
        raise RuntimeError('Provider capacity gate already installed')

    @wraps(original)
    def respond(client, request):
        waiting = time.monotonic()
        with capacity(Path(study_root) / 'api_capacity', slots) as slot:
            event(Path(episode_root) / 'provider_queue.jsonl', 'admitted',
                  slot=slot, wait_seconds=time.monotonic()-waiting, max_concurrent_requests=slots)
            # Forward the exact request once. Preserve the original HTTP status,
            # response, exceptions and uncertainty; never retry or rewrite output.
            return original(client, request)

    respond._appl_capacity_gate = True
    ResponsesClient.respond = respond
