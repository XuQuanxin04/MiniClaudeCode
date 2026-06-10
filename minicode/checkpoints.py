from __future__ import annotations

import base64
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHECKPOINT_DIR_NAME = ".mini-code-checkpoints"


@dataclass(frozen=True)
class CheckpointFile:
    path: str
    existed: bool
    kind: str
    sha256: str | None = None
    size: int = 0


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_id: str
    timestamp: float
    reason: str
    tool_name: str
    files: list[CheckpointFile]

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "reason": self.reason,
            "tool_name": self.tool_name,
            "files": [file.__dict__ for file in self.files],
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class CheckpointManager:
    """Workspace-local checkpoints for MiniCode-managed file changes.

    The manager records the file state immediately before a MiniCode tool
    modifies it. Rollback restores those previous states. Shell command side
    effects are intentionally not captured because they can touch arbitrary
    external state; those still require permission review.
    """

    def __init__(self, workspace: str | Path):
        self.workspace = Path(workspace).resolve()
        self.root = self.workspace / CHECKPOINT_DIR_NAME

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        return self.root / checkpoint_id

    def _metadata_path(self, checkpoint_id: str) -> Path:
        return self._checkpoint_path(checkpoint_id) / "metadata.json"

    def _resolve_workspace_path(self, target: str | Path) -> Path:
        path = Path(target)
        if path.is_absolute():
            return path.resolve()

        from_cwd = path.resolve()
        if _is_relative_to(from_cwd, self.workspace):
            return from_cwd
        return (self.workspace / path).resolve()

    def _relative_key(self, path: Path) -> str:
        try:
            return path.relative_to(self.workspace).as_posix()
        except ValueError as exc:
            raise ValueError(f"Checkpoint path is outside workspace: {path}") from exc

    def _should_skip(self, path: Path) -> bool:
        return _is_relative_to(path, self.root)

    def create(
        self,
        paths: list[str | Path],
        *,
        reason: str,
        tool_name: str = "",
    ) -> CheckpointRecord | None:
        resolved_paths = []
        for item in paths:
            path = self._resolve_workspace_path(item)
            if self._should_skip(path):
                continue
            self._relative_key(path)
            resolved_paths.append(path)

        if not resolved_paths:
            return None

        checkpoint_id = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
        checkpoint_path = self._checkpoint_path(checkpoint_id)
        files_dir = checkpoint_path / "files"
        files_dir.mkdir(parents=True, exist_ok=False)

        entries: list[CheckpointFile] = []
        seen: set[str] = set()
        for path in resolved_paths:
            rel = self._relative_key(path)
            if rel in seen:
                continue
            seen.add(rel)
            snapshot_path = files_dir / (base64.urlsafe_b64encode(rel.encode("utf-8")).decode("ascii") + ".bin")

            if not path.exists():
                entries.append(CheckpointFile(path=rel, existed=False, kind="missing"))
                continue

            if path.is_dir():
                shutil.make_archive(str(snapshot_path.with_suffix("")), "zip", root_dir=path)
                data = snapshot_path.with_suffix(".zip").read_bytes()
                entries.append(
                    CheckpointFile(path=rel, existed=True, kind="directory", sha256=_sha256_bytes(data), size=len(data))
                )
                continue

            data = path.read_bytes()
            snapshot_path.write_bytes(data)
            entries.append(
                CheckpointFile(path=rel, existed=True, kind="file", sha256=_sha256_bytes(data), size=len(data))
            )

        record = CheckpointRecord(
            checkpoint_id=checkpoint_id,
            timestamp=time.time(),
            reason=reason,
            tool_name=tool_name,
            files=entries,
        )
        self._metadata_path(checkpoint_id).write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return record

    def list_records(self) -> list[CheckpointRecord]:
        if not self.root.exists():
            return []
        records: list[CheckpointRecord] = []
        for meta in sorted(self.root.glob("*/metadata.json"), reverse=True):
            try:
                records.append(self._load_metadata(meta))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        return records

    def resolve_id(self, checkpoint_id: str) -> str:
        checkpoint_id = checkpoint_id.strip()
        if not checkpoint_id:
            raise ValueError("Checkpoint id is required")
        matches = [
            path.name for path in self.root.glob(f"{checkpoint_id}*")
            if path.is_dir() and (path / "metadata.json").exists()
        ]
        if not matches:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        if len(matches) > 1:
            raise ValueError(f"Checkpoint id is ambiguous: {checkpoint_id}")
        return matches[0]

    def load(self, checkpoint_id: str) -> CheckpointRecord:
        resolved = self.resolve_id(checkpoint_id)
        return self._load_metadata(self._metadata_path(resolved))

    def _load_metadata(self, path: Path) -> CheckpointRecord:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return CheckpointRecord(
            checkpoint_id=str(raw["checkpoint_id"]),
            timestamp=float(raw["timestamp"]),
            reason=str(raw.get("reason", "")),
            tool_name=str(raw.get("tool_name", "")),
            files=[CheckpointFile(**item) for item in raw.get("files", [])],
        )

    def rollback(self, checkpoint_id: str) -> CheckpointRecord:
        record = self.load(checkpoint_id)
        files_dir = self._checkpoint_path(record.checkpoint_id) / "files"

        for entry in record.files:
            target = (self.workspace / entry.path).resolve()
            if not _is_relative_to(target, self.workspace):
                raise ValueError(f"Refusing to rollback outside workspace: {target}")
            encoded = base64.urlsafe_b64encode(entry.path.encode("utf-8")).decode("ascii")

            if not entry.existed:
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            if entry.kind == "directory":
                snapshot_zip = files_dir / f"{encoded}.zip"
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                target.mkdir(parents=True, exist_ok=True)
                shutil.unpack_archive(str(snapshot_zip), str(target), "zip")
            elif entry.kind == "file":
                snapshot = files_dir / f"{encoded}.bin"
                target.write_bytes(snapshot.read_bytes())

        return record


def create_checkpoint_for_paths(
    workspace: str | Path,
    paths: list[str | Path],
    *,
    reason: str,
    tool_name: str = "",
) -> CheckpointRecord | None:
    return CheckpointManager(workspace).create(paths, reason=reason, tool_name=tool_name)


def format_checkpoint_list(workspace: str | Path, *, limit: int = 10) -> str:
    records = CheckpointManager(workspace).list_records()
    if not records:
        return "No checkpoints found."
    lines = ["Checkpoints", "=" * 50]
    for record in records[:limit]:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.timestamp))
        files = ", ".join(file.path for file in record.files[:3])
        if len(record.files) > 3:
            files += f", ... (+{len(record.files) - 3})"
        lines.append(f"{record.checkpoint_id}  {ts}  {record.tool_name or '-'}")
        lines.append(f"  reason: {record.reason}")
        lines.append(f"  files: {files}")
    return "\n".join(lines)


def format_checkpoint_show(workspace: str | Path, checkpoint_id: str) -> str:
    record = CheckpointManager(workspace).load(checkpoint_id)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record.timestamp))
    lines = [
        f"Checkpoint {record.checkpoint_id}",
        "=" * 50,
        f"time: {ts}",
        f"tool: {record.tool_name or '-'}",
        f"reason: {record.reason}",
        "",
        "files:",
    ]
    for file in record.files:
        status = "existed" if file.existed else "new file placeholder"
        lines.append(f"  - {file.path} ({file.kind}, {status}, {file.size} bytes)")
    return "\n".join(lines)
