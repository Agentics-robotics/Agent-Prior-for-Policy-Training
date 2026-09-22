"""CUDA namespace launcher adapted from the preserved v3 host."""
import os
from pathlib import Path
import sys


def launch(gpu, args):
    # Reuse recorded UUID/minor verification. This CUDA-only launcher does not
    # require graphics/render nodes used by the frozen simulator launcher.
    from appl.gpu import identity

    device = identity(gpu)
    if device["free_mib"] < 2048:
        raise RuntimeError(
            "Selected GPU has less than 2 GiB free; no alternate selected"
        )
    nodes = [
        Path("/dev/nvidiactl"),
        Path("/dev/nvidia-uvm"),
        Path("/dev") / f"nvidia{device['minor']}",
    ]
    nodes += [p for p in [Path("/dev/nvidia-uvm-tools")] if p.exists()]
    for p in nodes:
        if not p.is_char_device():
            raise RuntimeError(f"Missing NVIDIA character device: {p}")
    command = [
        "bwrap",
        "--die-with-parent",
        "--bind",
        "/",
        "/",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--unsetenv",
        "DISPLAY",
        "--unsetenv",
        "WAYLAND_DISPLAY",
    ]
    env = dict(
        APPL_PHYSICAL_GPU=gpu,
        APPL_GPU_UUID=device["uuid"],
        APPL_GPU_MINOR=device["minor"],
        CUDA_VISIBLE_DEVICES=device["uuid"],
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        LD_LIBRARY_PATH=str(Path(sys.prefix) / "lib"),
    )
    for name, value in env.items():
        command += ["--setenv", name, str(value)]
    for p in nodes:
        command += ["--dev-bind", str(p), str(p)]
    os.execvp(
        command[0],
        command
        + [
            "--",
            sys.executable,
            "-u",
            "-m",
            "real_robot.deployment_pipeline.worker",
            *args,
            "--isolated",
        ],
    )

