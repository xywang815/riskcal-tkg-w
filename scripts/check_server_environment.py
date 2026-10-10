from argparse import ArgumentParser
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


def _run(command: list[str]) -> dict[str, Any]:
    if shutil.which(command[0]) is None:
        return {"available": False, "command": command, "stdout": "", "stderr": ""}
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "available": True,
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def _torch_summary() -> dict[str, Any]:
    try:
        import torch
    except Exception as error:  # pragma: no cover - diagnostic path
        return {"installed": False, "error": repr(error)}
    summary: dict[str, Any] = {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        summary["devices"] = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
            }
            for index in range(torch.cuda.device_count())
        ]
    else:
        summary["devices"] = []
    return summary


def collect_environment() -> dict[str, Any]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "cwd": str(Path.cwd()),
        "torch": _torch_summary(),
        "nvidia_smi": _run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
    }


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    summary = collect_environment()
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
    if args.require_cuda and not summary["torch"].get("cuda_available", False):
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false")


if __name__ == "__main__":
    main()
