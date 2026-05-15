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
    """Launches workflow scripts; hyperparameters match the multimodel Colab notebook defaults in code."""
    SUPPORTED_PROFILES = {"best-quality"}

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
        """Workflow scripts embed multimodel-notebook defaults; do not override from the API."""
        return []

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
        # Training children must see the same third-party installs as an interactive interpreter.
        # IDEs sometimes set PYTHONNOUSERSITE=1, which hides user-site packages (where `pip install`
        # lands when global site-packages is not writable — e.g. SAM2 from GitHub).
        env.pop("PYTHONNOUSERSITE", None)
        if method == "maskrcnn":
            # Defaults: longer runs + val early stopping; Windows workers=0 avoids DataLoader MemoryError.
            env.setdefault("MASK_RCNN_EPOCHS", "500")
            env.setdefault("MASK_RCNN_EARLY_STOPPING_PATIENCE", "50")
            env.setdefault("MASK_RCNN_EARLY_STOPPING_MIN_DELTA", "1e-4")
            if sys.platform == "win32":
                env.setdefault("MASK_RCNN_DATALOADER_WORKERS", "0")
            return env
        if method == "sam2":
            # SAM2: optional env overrides; defaults match tree_canopy_multimodel notebook spirit (batch 2, lr 1e-4).
            env.setdefault("SAM2_MIN_MASK_SCORE", "0.3")
            env.setdefault("SAM2_BATCH_SIZE", "2")
            env.setdefault("SAM2_LR", "1e-4")
            env.setdefault("SAM2_EARLY_STOPPING_PATIENCE", "50")

            cfg_env = env.get("SAM2_MODEL_CFG") or env.get("MODEL_CFG")
            if cfg_env:
                return env

            cfg_candidates = [
                self.project_root / "checkpoints_sam2" / "sam2_hiera_l.yaml",
                self.project_root / "sam2_hiera_l.yaml",
                self.project_root / "configs" / "sam2_hiera_l.yaml",
                self.project_root / "backend" / "configs" / "sam2_hiera_l.yaml",
                self.project_root / "checkpoints_sam2" / "sam2_hiera_s.yaml",
                self.project_root / "sam2_hiera_s.yaml",
                self.project_root / "configs" / "sam2_hiera_s.yaml",
                self.project_root / "backend" / "configs" / "sam2_hiera_s.yaml",
            ]
            resolved_cfg = self._first_existing_file(cfg_candidates)
            if resolved_cfg:
                env["SAM2_MODEL_CFG"] = resolved_cfg

            return env

        return env

    def start_training(self, method: str, profile: str = "best-quality") -> JobState:
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
        if method == "sam2":
            from backend.app.sam2_assets import prepare_sam2_checkpoints_dir

            yaml_path, pt_path = prepare_sam2_checkpoints_dir(self.project_root, runtime_env)
            runtime_env = dict(runtime_env)
            runtime_env["SAM2_MODEL_CFG"] = str(yaml_path)
            runtime_env["SAM2_CHECKPOINT"] = str(pt_path)

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
                log_tail = "".join(lines[-120:])

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
