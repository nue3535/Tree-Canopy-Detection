from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class JobState:
    method: str
    profile: str
    started_at: float
    command: list[str]
    log_path: Path
    pid: int
    status: str
    return_code: int | None = None


class TrainingManager:
    SUPPORTED_PROFILES = {"fast", "balanced", "best-quality"}

    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.logs_dir = self.project_root / "local_outputs" / "training_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, JobState] = {}
        self.processes: dict[str, subprocess.Popen[Any]] = {}

    @staticmethod
    def _first_existing_file(candidates: list[Path]) -> str | None:
        """Return first existing file path from candidates."""
        for path in candidates:
            if path.is_file():
                return str(path)
        return None

    def _profile_args_for_method(self, method: str, profile: str) -> list[str]:
        """Profile presets for each model training workflow."""
        presets: dict[str, dict[str, list[str]]] = {
            "deeplabv3plus": {
                "fast": ["--epochs", "20", "--img-size", "512", "--lr", "1e-4"],
                "balanced": ["--epochs", "60", "--img-size", "640", "--lr", "1e-4"],
                "best-quality": ["--epochs", "90", "--img-size", "640", "--lr", "8e-5"],
            },
            "sam2": {
                "fast": [],
                "balanced": [],
                "best-quality": [],
            },
            "unet": {
                "fast": ["--epochs", "12", "--lr", "2e-4"],
                "balanced": ["--epochs", "30", "--lr", "2e-4"],
                "best-quality": ["--epochs", "50", "--lr", "1.5e-4"],
            },
            "maskrcnn": {
                "fast": ["--epochs", "4", "--img-size", "320", "--max-instances-per-image", "60", "--lr", "1e-4"],
                "balanced": ["--epochs", "10", "--img-size", "384", "--max-instances-per-image", "80", "--lr", "1e-4"],
                "best-quality": ["--epochs", "16", "--img-size", "448", "--max-instances-per-image", "100", "--lr", "8e-5"],
            },
            "segformer": {
                "fast": ["--epochs", "8", "--img-size", "448", "--lr", "7e-5"],
                "balanced": ["--epochs", "20", "--img-size", "512", "--lr", "6e-5"],
                "best-quality": ["--epochs", "35", "--img-size", "640", "--lr", "4e-5"],
            },
        }
        if method == "sam2":
            # SAM2 uses module-level constants; tune via env vars.
            return []
        method_presets = presets.get(method, {})
        return method_presets.get(profile, [])

    def _command_for_method(self, method: str, profile: str) -> list[str]:
        if method == "deeplabv3plus":
            return [
                sys.executable,
                "-u",
                "-m",
                "backend.scripts.deeplab_v3plus_workflow",
                "train",
            ] + self._profile_args_for_method(method, profile)
        if method == "sam2":
            return [
                sys.executable,
                "-u",
                "-m",
                "backend.scripts.sam2_workflow",
            ]
        if method == "unet":
            return [
                sys.executable,
                "-u",
                "-m",
                "backend.scripts.unet_workflow",
                "train",
            ] + self._profile_args_for_method(method, profile)
        if method == "maskrcnn":
            return [
                sys.executable,
                "-u",
                "-m",
                "backend.scripts.mask_rcnn_workflow",
                "train",
            ] + self._profile_args_for_method(method, profile)
        if method == "segformer":
            return [
                sys.executable,
                "-u",
                "-m",
                "backend.scripts.segformer_workflow",
                "train",
            ] + self._profile_args_for_method(method, profile)
        raise ValueError("Unsupported method.")

    def _runtime_env_for_method(self, method: str, profile: str) -> dict[str, str]:
        """Build child process env and inject model asset vars when applicable."""
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        if method != "sam2":
            return env

        # SAM2 profile control is env-driven because workflow reads module constants.
        sam2_profile = profile if profile in self.SUPPORTED_PROFILES else "balanced"
        if sam2_profile == "fast":
            env["SAM2_STEPS"] = "1200"
            env["SAM2_SAVE_EVERY"] = "100"
            env["SAM2_LR"] = "1e-5"
        elif sam2_profile == "best-quality":
            env["SAM2_STEPS"] = "6000"
            env["SAM2_SAVE_EVERY"] = "250"
            env["SAM2_LR"] = "8e-6"
        else:
            env["SAM2_STEPS"] = "3000"
            env["SAM2_SAVE_EVERY"] = "200"
            env["SAM2_LR"] = "1e-5"

        cfg_env = env.get("SAM2_MODEL_CFG") or env.get("MODEL_CFG")
        ckpt_env = env.get("SAM2_CHECKPOINT")
        if cfg_env and ckpt_env:
            return env

        cfg_candidates = [
            self.project_root / "checkpoints_sam2" / "sam2_hiera_s.yaml",
            self.project_root / "sam2_hiera_s.yaml",
            self.project_root / "configs" / "sam2_hiera_s.yaml",
            self.project_root / "backend" / "configs" / "sam2_hiera_s.yaml",
        ]
        ckpt_candidates = [
            self.project_root / "checkpoints_sam2" / "sam2_hiera_small.pt",
            self.project_root / "sam2_hiera_small.pt",
            self.project_root / "checkpoints" / "sam2_hiera_small.pt",
            self.project_root / "backend" / "checkpoints" / "sam2_hiera_small.pt",
        ]
        resolved_cfg = self._first_existing_file(cfg_candidates)
        resolved_ckpt = self._first_existing_file(ckpt_candidates)

        if not cfg_env and resolved_cfg:
            env["SAM2_MODEL_CFG"] = resolved_cfg
        if not ckpt_env and resolved_ckpt:
            env["SAM2_CHECKPOINT"] = resolved_ckpt

        return env

    def start_training(self, method: str, profile: str = "balanced") -> JobState:
        method = method.strip().lower()
        profile = profile.strip().lower()
        if profile not in self.SUPPORTED_PROFILES:
            raise ValueError("Unsupported training profile.")
        self.refresh_status(method)
        existing = self.jobs.get(method)
        if existing and existing.status == "running":
            return existing

        command = self._command_for_method(method, profile)
        runtime_env = self._runtime_env_for_method(method, profile)
        ts = int(time.time())
        log_path = self.logs_dir / f"{method}_{ts}.log"
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)
        proc = subprocess.Popen(
            command,
            cwd=self.project_root,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=runtime_env,
        )
        self.processes[method] = proc
        state = JobState(
            method=method,
            profile=profile,
            started_at=time.time(),
            command=command,
            log_path=log_path,
            pid=proc.pid,
            status="running",
            return_code=None,
        )
        self.jobs[method] = state
        return state

    def refresh_status(self, method: str) -> JobState | None:
        method = method.strip().lower()
        state = self.jobs.get(method)
        proc = self.processes.get(method)
        if not state or not proc:
            return state
        code = proc.poll()
        if code is None:
            state.status = "running"
            state.return_code = None
        else:
            state.status = "completed" if code == 0 else "failed"
            state.return_code = int(code)
        return state

    def get_status(self, method: str) -> dict[str, Any]:
        method = method.strip().lower()
        state = self.refresh_status(method)
        if not state:
            return {
                "method": method,
                "profile": None,
                "status": "not_started",
                "started_at": 0,
                "pid": None,
                "return_code": None,
                "command": [],
                "log_path": "",
                "log_tail": "",
            }

        log_tail = ""
        if state.log_path.exists():
            with open(state.log_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
                log_tail = "".join(lines[-40:])

        return {
            "method": state.method,
            "profile": state.profile,
            "status": state.status,
            "started_at": int(state.started_at),
            "pid": state.pid,
            "return_code": state.return_code,
            "command": state.command,
            "log_path": str(state.log_path),
            "log_tail": log_tail,
        }

    def stop_training(self, method: str) -> dict[str, Any]:
        method = method.strip().lower()
        state = self.refresh_status(method)
        if not state:
            return {
                "method": method,
                "status": "not_started",
                "stopped": False,
                "message": "Training has not been started for this method.",
            }

        proc = self.processes.get(method)
        if proc is None:
            return {
                "method": method,
                "status": state.status,
                "stopped": False,
                "message": "No active process handle was found.",
            }

        code = proc.poll()
        if code is not None:
            state.status = "completed" if code == 0 else "failed"
            state.return_code = int(code)
            return {
                "method": method,
                "status": state.status,
                "stopped": False,
                "message": "Process already finished.",
            }

        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=8)

        state.status = "stopped"
        state.return_code = int(proc.returncode) if proc.returncode is not None else None
        return {
            "method": method,
            "status": state.status,
            "stopped": True,
            "message": "Training process has been stopped.",
        }
