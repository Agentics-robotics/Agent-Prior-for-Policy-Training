"""Numerical package audit and kernel isolation, reusing frozen APPL primitives."""

import ast
import os
from pathlib import Path
import resource
import sys
import time

from appl.io import ROOT, atomic
from appl.kernel import _landlock, _seccomp

ALLOWED = {
    "torch",
    "numpy",
    "cv2",
    "scipy",
    "skimage",
    "PIL",
    "diffusers",
    "safetensors",
    "math",
    "json",
    "collections",
    "itertools",
    "functools",
    "dataclasses",
    "typing",
    "copy",
    "transformers",
    "torchvision",
    "sam2",
    "hydra",
    "omegaconf",
}


def audit(source):
    source = Path(source)
    local = {p.stem for p in source.glob("*.py")}
    forbidden = {
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "__import__",
        "breakpoint",
    }
    for path in source.glob("*.py"):
        if path.is_symlink():
            raise ValueError("No source symlinks")
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [a.name for a in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                if isinstance(node, ast.ImportFrom) and node.level:
                    raise ValueError("Use explicit flat module imports")
                for name in names:
                    if name.split(".")[0] not in ALLOWED | local and name not in {
                        "real_robot.policy_training_v4",
                        "real_robot.policy_training_v4.public",
                    }:
                        raise ValueError(
                            "Import outside declared numerical capabilities: " + name
                        )
            if isinstance(node, ast.Name) and node.id in forbidden:
                raise ValueError("Dynamic IO/reflection is not a policy capability")
            if (
                isinstance(node, ast.Attribute)
                and node.attr.startswith("__")
                and node.attr != "__init__"
            ):
                raise ValueError("Private reflection is forbidden")
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                raise ValueError("Global mutation is forbidden")


def lockdown(source, output, readonly, gpu):
    import torch
    from appl.gpu import verify_cuda

    source, output = Path(source).resolve(), Path(output).resolve()
    audit(source)
    identity = verify_cuda() if gpu else None
    started = time.monotonic()
    if gpu:
        layer = torch.nn.Conv2d(3, 8, 3).cuda()
        optimizer = torch.optim.AdamW(layer.parameters())
        loss = layer(torch.ones(2, 3, 16, 16, device="cuda")).square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        del layer, optimizer, loss
    paths = [
        Path(sys.prefix),
        Path("/usr/lib"),
        Path("/usr/lib64"),
        Path("/etc/ld.so.cache"),
        Path("/dev/null"),
        Path("/dev/zero"),
        Path("/dev/urandom"),
        Path("/dev/random"),
        ROOT / "real_robot/policy_training_v4",
        ROOT / "real_robot/data.py",
        ROOT / "src/appl/io.py",
        source,
        *map(Path, readonly),
    ]
    devices = (
        []
        if not gpu
        else [
            Path("/dev") / name
            for name in (
                f"nvidia{identity['minor']}",
                "nvidiactl",
                "nvidia-uvm",
                "nvidia-uvm-tools",
            )
            if (Path("/dev") / name).exists()
        ]
    )
    environment = {
        k: os.environ[k]
        for k in (
            "APPL_GPU_UUID",
            "APPL_GPU_MINOR",
            "APPL_PHYSICAL_GPU",
            "CUDA_VISIBLE_DEVICES",
        )
        if k in os.environ
    }
    os.environ.clear()
    os.environ.update(
        environment,
        PATH="/usr/bin:/bin",
        LANG="C.UTF-8",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        CUDA_CACHE_DISABLE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        TMPDIR=str(output),
        XDG_CACHE_HOME=str(output / "cache"),
    )
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    abi = _landlock([p.resolve() for p in paths if p.exists()], [output], devices)
    _seccomp()
    atomic(
        output / "enforcement.json",
        dict(
            landlock_abi=abi,
            seccomp=True,
            network=False,
            credentials=False,
            device=identity,
            source_imported_after_lockdown=True,
            readonly_grants=[str(p) for p in paths if p.exists()],
            write_dirs=[str(output)],
            initialization_seconds=time.monotonic() - started,
            trusted_warmup_optimizer_steps=1 if gpu else 0,
            trusted_warmup_is_candidate_training=False,
        ),
    )
    sys.path.insert(0, str(source))
    return identity
