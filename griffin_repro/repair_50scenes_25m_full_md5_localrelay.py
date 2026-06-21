#!/usr/bin/env python3
"""Repair Griffin-50scenes-25m archives through a local download relay.

The remote A100 server may have poor connectivity to Hugging Face mirrors. This
script downloads only MD5-mismatched 25m archives on the local machine with
parallel HTTP range requests, verifies them locally, uploads them to the remote
archive directory, verifies them again on the remote host, and only then swaps
the corrupt archive out.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, Iterable, List, Optional

import paramiko


DEFAULT_REMOTE_ROOT = "/home/fp/CARLA/CarlaAir-v0.1.7/code"
DEFAULT_BASE_URL = "https://hf-mirror.com/datasets/wjh-svm/Griffin/resolve/main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair MD5-mismatched Griffin-50scenes-25m archives via local download and SFTP."
    )
    parser.add_argument("--remote-host", default=os.environ.get("GRIFFIN_REMOTE_HOST", "10.2.14.120"))
    parser.add_argument("--remote-user", default=os.environ.get("GRIFFIN_REMOTE_USER", "fp"))
    parser.add_argument("--remote-password", default=os.environ.get("GRIFFIN_REMOTE_PASSWORD"))
    parser.add_argument("--remote-root", default=os.environ.get("GRIFFIN_REMOTE_ROOT", DEFAULT_REMOTE_ROOT))
    parser.add_argument("--base-url", default=os.environ.get("GRIFFIN_REPAIR_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--parts", type=int, default=int(os.environ.get("GRIFFIN_LOCAL_REPAIR_PARTS", "24")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("GRIFFIN_LOCAL_REPAIR_WORKERS", "16")))
    parser.add_argument(
        "--cache-dir",
        default=os.environ.get(
            "GRIFFIN_LOCAL_REPAIR_CACHE",
            r"D:\griffin_25m_md5_repair_cache" if os.name == "nt" else "/tmp/griffin_25m_md5_repair_cache",
        ),
    )
    parser.add_argument("--stamp", default=os.environ.get("GRIFFIN_LOCAL_REPAIR_STAMP"))
    parser.add_argument("--limit", type=int, default=0, help="Repair at most N mismatched archives.")
    parser.add_argument("--only-name", action="append", default=[], help="Repair only this archive basename.")
    parser.add_argument("--no-final-verify", action="store_true", help="Skip final full MD5 and extraction.")
    parser.add_argument(
        "--keep-local-archives",
        action="store_true",
        help="Keep locally downloaded zip files after verified remote replacement.",
    )
    return parser.parse_args()


def run_remote(client: paramiko.SSHClient, command: str, timeout: Optional[int] = None) -> str:
    stdin, stdout, stderr = client.exec_command(command, get_pty=False, timeout=timeout)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    if code != 0:
        raise RuntimeError(f"remote command failed ({code}): {command}\nSTDOUT:\n{out}\nSTDERR:\n{err}")
    return out


def connect(args: argparse.Namespace) -> paramiko.SSHClient:
    if not args.remote_password:
        raise SystemExit("Set GRIFFIN_REMOTE_PASSWORD or pass --remote-password.")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        args.remote_host,
        username=args.remote_user,
        password=args.remote_password,
        timeout=20,
        banner_timeout=20,
        auth_timeout=20,
    )
    return client


def fetch_mismatches(client: paramiko.SSHClient, remote_root: str) -> List[Dict[str, Any]]:
    command = f"""
set -euo pipefail
cd {shlex.quote(remote_root)}
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py verify-data-md5 \\
  --dataset 50scenes_25m \\
  --package-profile full \\
  --json
"""
    payload = json.loads(run_remote(client, command, timeout=1200))
    return [item for item in payload.get("checks", []) if item.get("status") == "mismatch"]


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curl_executable() -> str:
    exe = shutil.which("curl.exe") or shutil.which("curl")
    if not exe:
        raise SystemExit("curl/curl.exe not found on local machine.")
    return exe


def download_part(
    curl: str,
    url: str,
    part_path: Path,
    start: int,
    end: int,
    expected_size: int,
) -> None:
    if part_path.exists() and part_path.stat().st_size == expected_size:
        return
    part_path.unlink(missing_ok=True)
    cmd = [
        curl,
        "--silent",
        "--show-error",
        "--fail",
        "--retry",
        "8",
        "--retry-all-errors",
        "--connect-timeout",
        "30",
        "--max-time",
        "900",
        "--speed-limit",
        "10240",
        "--speed-time",
        "90",
        "-L",
        "-r",
        f"{start}-{end}",
        "-o",
        str(part_path),
        url,
    ]
    subprocess.run(cmd, check=True)
    actual = part_path.stat().st_size
    if actual != expected_size:
        raise RuntimeError(f"part size mismatch for {part_path.name}: expected {expected_size}, got {actual}")


def assemble(parts_dir: Path, out_path: Path, part_count: int) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".assembling")
    tmp.unlink(missing_ok=True)
    with tmp.open("wb") as out:
        for idx in range(part_count):
            part_path = parts_dir / f"part_{idx:03d}"
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, out, length=8 * 1024 * 1024)
    tmp.replace(out_path)


def download_archive(item: Dict[str, Any], base_url: str, cache_dir: Path, parts: int, workers: int) -> Path:
    name = Path(item["path"]).name
    expected_size = int(item["expected_size_bytes"])
    expected_md5 = item["expected_md5"]
    zip_path = cache_dir / name
    if zip_path.exists() and zip_path.stat().st_size == expected_size and md5_file(zip_path) == expected_md5:
        print(f"[local] already verified {name}", flush=True)
        return zip_path

    zip_path.unlink(missing_ok=True)
    parts_dir = cache_dir / f"{name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url.rstrip('/')}/{item['path']}"
    curl = curl_executable()
    print(f"[local] downloading {name} as {parts} range parts with {workers} workers", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = []
        for idx in range(parts):
            start = expected_size * idx // parts
            end = expected_size * (idx + 1) // parts - 1
            part_size = end - start + 1
            futures.append(pool.submit(download_part, curl, url, parts_dir / f"part_{idx:03d}", start, end, part_size))
        for future in concurrent.futures.as_completed(futures):
            future.result()

    assemble(parts_dir, zip_path, parts)
    actual_size = zip_path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(f"archive size mismatch for {name}: expected {expected_size}, got {actual_size}")
    actual_md5 = md5_file(zip_path)
    if actual_md5 != expected_md5:
        raise RuntimeError(f"archive md5 mismatch for {name}: expected {expected_md5}, got {actual_md5}")
    shutil.rmtree(parts_dir)
    print(f"[local] verified {name}", flush=True)
    return zip_path


def upload_and_replace(
    client: paramiko.SSHClient,
    local_path: Path,
    item: Dict[str, Any],
    remote_root: str,
    stamp: str,
) -> None:
    name = local_path.name
    expected_size = int(item["expected_size_bytes"])
    expected_md5 = item["expected_md5"]
    archive_dir = f"{remote_root}/griffin_repro/official/datasets/griffin_50scenes_25m/archives"
    remote_tmp = f"{archive_dir}/{name}.redownload.{stamp}.localrelay"
    remote_target = f"{archive_dir}/{name}"
    print(f"[remote] uploading {name}", flush=True)
    sftp = client.open_sftp()
    try:
        sftp.put(str(local_path), remote_tmp)
    finally:
        sftp.close()
    verify = f"""
set -euo pipefail
test "$(stat -c%s {shlex.quote(remote_tmp)})" = {expected_size}
test "$(md5sum {shlex.quote(remote_tmp)} | awk '{{print $1}}')" = {shlex.quote(expected_md5)}
if [ -f {shlex.quote(remote_target)} ]; then
  mv -f {shlex.quote(remote_target)} {shlex.quote(remote_target + '.corrupt.' + stamp)}
fi
mv -f {shlex.quote(remote_tmp)} {shlex.quote(remote_target)}
rm -f {shlex.quote(remote_target + '.extracted.to-data-parent')}
"""
    run_remote(client, verify, timeout=1200)
    print(f"[remote] replaced {name}", flush=True)


def final_remote_verify(client: paramiko.SSHClient, remote_root: str, stamp: str, repaired_names: Iterable[str]) -> None:
    names_json = json.dumps(list(repaired_names))
    command = f"""
set -euo pipefail
cd {shlex.quote(remote_root)}
after_json=griffin_repro/artifacts/logs/official_25m_md5_repair_after_{shlex.quote(stamp)}.json
/home/fp/miniconda3/envs/griffin/bin/python scripts/griffin_repro.py verify-data-md5 \\
  --dataset 50scenes_25m \\
  --package-profile full \\
  --json > "$after_json"
/home/fp/miniconda3/envs/griffin/bin/python - "$after_json" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
if not payload.get("ready"):
    bad = [item for item in payload.get("checks", []) if item.get("status") != "matched"]
    raise SystemExit(f"full MD5 still not ready: {{bad}}")
PY
/home/fp/miniconda3/envs/griffin/bin/python - <<'PY'
import json
import subprocess
names = json.loads({names_json!r})
archive_dir = "griffin_repro/official/datasets/griffin_50scenes_25m/archives"
data_parent = "griffin_repro/official/datasets"
for name in names:
    print(f"[remote] extracting {{name}}", flush=True)
    subprocess.run(["unzip", "-oq", f"{{archive_dir}}/{{name}}", "-d", data_parent], check=True)
    open(f"{{archive_dir}}/{{name}}.extracted.to-data-parent", "a").close()
PY
"""
    run_remote(client, command, timeout=7200)


def main() -> int:
    args = parse_args()
    if args.parts < 1:
        raise SystemExit("--parts must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    stamp = args.stamp or time.strftime("localrelay_%Y%m%d_%H%M%S")
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = connect(args)
    repaired: List[str] = []
    try:
        mismatches = fetch_mismatches(client, args.remote_root)
        if args.only_name:
            wanted = set(args.only_name)
            mismatches = [item for item in mismatches if Path(item["path"]).name in wanted]
        if args.limit:
            mismatches = mismatches[: args.limit]
        print(f"[remote] mismatches={len(mismatches)}", flush=True)
        for item in mismatches:
            local_zip = download_archive(item, args.base_url, cache_dir, args.parts, args.workers)
            upload_and_replace(client, local_zip, item, args.remote_root, stamp)
            repaired.append(local_zip.name)
            if not args.keep_local_archives:
                local_zip.unlink(missing_ok=True)
                print(f"[local] removed {local_zip.name}", flush=True)
        if repaired and not args.no_final_verify:
            final_remote_verify(client, args.remote_root, stamp, repaired)
    finally:
        client.close()
    print(f"[done] repaired={len(repaired)} stamp={stamp}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
