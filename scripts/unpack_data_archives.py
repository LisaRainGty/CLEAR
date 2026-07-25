"""Verify and restore the archived raw and processed data layers."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Iterator

try:
    import zstandard as zstd
except ModuleNotFoundError:  # pragma: no cover - exercised on minimal hosts
    zstd = None


@contextmanager
def decompressed_reader(path: Path) -> Iterator[BinaryIO]:
    """Use the Python package when installed, otherwise the zstd executable."""
    if zstd is not None:
        with path.open("rb") as compressed:
            with zstd.ZstdDecompressor().stream_reader(compressed) as stream:
                yield stream
        return

    executable = shutil.which("zstd")
    if not executable:
        raise RuntimeError("install zstandard>=0.23 or provide a zstd executable")
    process = subprocess.Popen([executable, "-dc", str(path)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    try:
        yield process.stdout
    finally:
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, process.args)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_member(member: tarfile.TarInfo, layer: str) -> None:
    path = PurePosixPath(member.name)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != layer:
        raise ValueError(f"unsafe archive member: {member.name}")
    if not (member.isdir() or member.isfile()):
        raise ValueError(f"unsupported archive member: {member.name}")


def tree_record(root: Path) -> tuple[str, int, int]:
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("._")
    )
    digest = hashlib.sha256()
    logical_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        logical_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest(), logical_bytes, len(files)


def archive_tree(archive_path: Path, layer: str) -> tuple[str, int, int]:
    """Hash archive payloads without extracting them to disk."""
    tree_digest = hashlib.sha256()
    logical_bytes = 0
    file_count = 0
    with decompressed_reader(archive_path) as stream:
        with tarfile.open(fileobj=stream, mode="r|") as archive:
            for member in archive:
                safe_member(member, layer)
                if member.isdir():
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError(f"missing file payload: {member.name}")
                file_digest = hashlib.sha256()
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    file_digest.update(chunk)
                relative = PurePosixPath(member.name).relative_to(layer).as_posix()
                tree_digest.update(relative.encode("utf-8"))
                tree_digest.update(b"\0")
                tree_digest.update(str(member.size).encode("ascii"))
                tree_digest.update(b"\0")
                tree_digest.update(file_digest.hexdigest().encode("ascii"))
                tree_digest.update(b"\n")
                logical_bytes += member.size
                file_count += 1
    return tree_digest.hexdigest(), logical_bytes, file_count


def verify_archive(data_root: Path, layer: str, expected: dict) -> None:
    archive_path = data_root.parent / expected["archive"]
    actual_archive_sha = sha256(archive_path)
    if actual_archive_sha != expected["archive_sha256"]:
        raise ValueError(f"archive checksum mismatch: {archive_path}")
    tree_sha, logical_bytes, file_count = archive_tree(archive_path, layer)
    actual = {
        "tree_sha256": tree_sha,
        "logical_bytes": logical_bytes,
        "file_count": file_count,
    }
    wanted = {key: expected[key] for key in actual}
    if actual != wanted:
        raise ValueError(f"archive tree mismatch for {layer}: {actual} != {wanted}")
    print(f"[verified archive] {layer}: {file_count} files, {logical_bytes} bytes, {tree_sha}")


def unpack(data_root: Path, layer: str, expected: dict) -> None:
    destination = data_root / layer
    if destination.exists():
        raise FileExistsError(
            f"refusing to merge into existing {destination}; move it aside before unpacking"
        )
    archive_path = data_root.parent / expected["archive"]
    actual_archive_sha = sha256(archive_path)
    if actual_archive_sha != expected["archive_sha256"]:
        raise ValueError(f"archive checksum mismatch: {archive_path}")

    try:
        with decompressed_reader(archive_path) as stream:
            with tarfile.open(fileobj=stream, mode="r|") as archive:
                for member in archive:
                    safe_member(member, layer)
                    archive.extract(member, path=data_root)
        tree_sha, logical_bytes, file_count = tree_record(destination)
        actual = {
            "tree_sha256": tree_sha,
            "logical_bytes": logical_bytes,
            "file_count": file_count,
        }
        wanted = {key: expected[key] for key in actual}
        if actual != wanted:
            raise ValueError(f"extracted tree mismatch for {layer}: {actual} != {wanted}")
    except Exception:
        if destination.exists():
            shutil.rmtree(destination)
        raise
    print(f"[verified] {layer}: {file_count} files, {logical_bytes} bytes, {tree_sha}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--layers", nargs="*", default=None)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="stream-verify archives without extracting them",
    )
    args = parser.parse_args()

    manifest_path = args.data_root / "archives" / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    layers = args.layers or list(manifest["layers"])
    unknown = sorted(set(layers) - set(manifest["layers"]))
    if unknown:
        raise ValueError(f"unknown archived layers: {unknown}")
    for layer in layers:
        if args.verify_only:
            verify_archive(args.data_root, layer, manifest["layers"][layer])
        else:
            unpack(args.data_root, layer, manifest["layers"][layer])


if __name__ == "__main__":
    main()
