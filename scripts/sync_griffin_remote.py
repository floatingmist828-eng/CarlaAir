#!/usr/bin/env python3
"""Sync only the isolated Griffin reproduction package to a remote CarlaAir tree."""

from __future__ import annotations

import argparse
import io
import json
import os
import posixpath
import tarfile
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_FILES = {
    ".gitignore",
    "docs/griffin_reproduction_design.md",
    "scripts/griffin_repro.py",
    "scripts/sync_griffin_remote.py",
    "tests/test_griffin_repro.py",
}
INCLUDE_DIRS = {
    "griffin_repro",
}
EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
}
EXCLUDED_PREFIXES = (
    "griffin_repro/artifacts/",
    "griffin_repro/official/ckpts/",
    "griffin_repro/official/datasets/",
    "griffin_repro/official/outputs/",
    "griffin_repro/official/result_vis/",
)


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def is_excluded(relative: str) -> bool:
    if any(relative.startswith(prefix) for prefix in EXCLUDED_PREFIXES):
        return True
    parts = set(Path(relative).parts)
    if parts & EXCLUDED_PARTS:
        return True
    if "/work_dirs" in relative or relative.startswith("griffin_repro/official/work_dirs"):
        return True
    if relative.endswith((".pyc", ".pyo", ".tar.gz")):
        return True
    return False


def included_files() -> list[str]:
    files = set()
    for name in INCLUDE_FILES:
        path = REPO_ROOT / name
        if path.exists() and not is_excluded(name):
            files.add(name)
    for dirname in INCLUDE_DIRS:
        root = REPO_ROOT / dirname
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                name = rel(path)
                if not is_excluded(name):
                    files.add(name)
    return sorted(files)


def build_tar(files: list[str]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name in files:
            archive.add(REPO_ROOT / name, arcname=name)
    buffer.seek(0)
    return buffer.read()


def run_remote_command(client: Any, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command)
    del stdin
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def sync(args: argparse.Namespace, files: list[str]) -> dict[str, Any]:
    try:
        import paramiko
    except ImportError as exc:
        raise SystemExit("paramiko is required for password-based sync: python -m pip install paramiko") from exc

    password = os.environ.get(args.password_env)
    if not password:
        raise SystemExit(f"Set {args.password_env} before running non-dry sync")

    archive_name = f"carlaair_griffin_repro_{int(time.time())}.tar.gz"
    remote_archive = posixpath.join("/tmp", archive_name)
    tar_bytes = build_tar(files)

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(args.host, port=args.port, username=args.user, password=password, timeout=args.timeout)
    try:
        sftp = client.open_sftp()
        try:
            with sftp.file(remote_archive, "wb") as remote_file:
                remote_file.write(tar_bytes)
        finally:
            sftp.close()

        mkdir = f"mkdir -p {sh_quote(args.remote_dir)}"
        extract = f"tar -xzf {sh_quote(remote_archive)} -C {sh_quote(args.remote_dir)}"
        cleanup = f"rm -f {sh_quote(remote_archive)}"
        code, out, err = run_remote_command(client, f"{mkdir} && {extract} && {cleanup}")
        if code != 0:
            raise SystemExit(f"remote extract failed with {code}\n{out}\n{err}")

        verify_cmd = f"cd {sh_quote(args.remote_dir)} && python3 scripts/griffin_repro.py verify-layout --json"
        verify_code, verify_out, verify_err = run_remote_command(client, verify_cmd)
        return {
            "host": args.host,
            "remote_dir": args.remote_dir,
            "files_synced": len(files),
            "remote_verify_code": verify_code,
            "remote_verify_stdout": verify_out.strip(),
            "remote_verify_stderr": verify_err.strip(),
        }
    finally:
        client.close()


def sh_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--remote-dir", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--password-env", default="GRIFFIN_REMOTE_PASSWORD")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    files = included_files()
    if args.dry_run:
        payload = {"files": files, "file_count": len(files), "remote_dir": args.remote_dir}
    else:
        payload = sync(args, files)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
