"""Create compact, checksummed archives for high-file-count data layers.

The archives preserve every regular file while normalising filesystem-only
metadata (owner, group, permissions, and modification time).  Scientific
identity is recorded separately as a SHA-256 tree digest over sorted relative
paths, file sizes, and file contents.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import BinaryIO, Iterator

try:
    import zstandard as zstd
except ModuleNotFoundError:  # pragma: no cover - exercised on minimal hosts
    zstd = None


DEFAULT_LAYERS = ("raw", "processed")


@contextmanager
def compressed_writer(target: Path) -> Iterator[BinaryIO]:
    """Use the Python package when installed, otherwise the zstd executable."""
    if zstd is not None:
        compressor = zstd.ZstdCompressor(
            level=10,
            threads=-1,
            write_checksum=True,
            write_content_size=False,
        )
        with target.open("wb") as compressed:
            with compressor.stream_writer(compressed, closefd=False) as stream:
                yield stream
        return

    executable = shutil.which("zstd")
    if not executable:
        raise RuntimeError("install zstandard>=0.23 or provide a zstd executable")
    with target.open("wb") as compressed:
        process = subprocess.Popen(
            [executable, "-T0", "-10", "--stdout"],
            stdin=subprocess.PIPE,
            stdout=compressed,
        )
        assert process.stdin is not None
        try:
            yield process.stdin
        finally:
            process.stdin.close()
            return_code = process.wait()
            if return_code:
                raise subprocess.CalledProcessError(return_code, process.args)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file() and not path.name.startswith("._")
    )


def tree_record(root: Path, files: list[Path]) -> tuple[str, int]:
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
    return digest.hexdigest(), logical_bytes


def normalised_info(path: Path, arcname: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arcname)
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mtime = 0
    if path.is_dir():
        info.type = tarfile.DIRTYPE
        info.mode = 0o755
    else:
        info.type = tarfile.REGTYPE
        info.mode = 0o644
        info.size = path.stat().st_size
    return info


def pack_layer(data_root: Path, output_root: Path, layer: str, overwrite: bool) -> dict:
    source = data_root / layer
    target = output_root / f"{layer}.tar.zst"
    if not source.is_dir():
        raise FileNotFoundError(source)
    if target.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {target}; pass --overwrite")

    files = regular_files(source)
    tree_sha, logical_bytes = tree_record(source, files)
    output_root.mkdir(parents=True, exist_ok=True)
    with compressed_writer(target) as stream:
        with tarfile.open(fileobj=stream, mode="w|") as archive:
            archive.addfile(normalised_info(source, layer))
            for path in files:
                arcname = f"{layer}/{path.relative_to(source).as_posix()}"
                info = normalised_info(path, arcname)
                with path.open("rb") as handle:
                    archive.addfile(info, handle)

    return {
        "archive": target.relative_to(data_root.parent).as_posix(),
        "archive_sha256": sha256(target),
        "compressed_bytes": target.stat().st_size,
        "file_count": len(files),
        "logical_bytes": logical_bytes,
        "tree_sha256": tree_sha,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--layers", nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_root = args.data_root / "archives"
    records = {
        layer: pack_layer(args.data_root, output_root, layer, args.overwrite)
        for layer in args.layers
    }
    manifest = {
        "schema_version": 1,
        "format": "tar+zstandard",
        "metadata_normalisation": {
            "uid": 0,
            "gid": 0,
            "mtime": 0,
            "directory_mode": "0755",
            "file_mode": "0644",
        },
        "layers": records,
    }
    manifest_path = output_root / "MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
