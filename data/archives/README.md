# Archived data layers

`raw.tar.zst` and `processed.tar.zst` store the complete high-file-count data
layers without duplicating thousands of small Git/LFS objects. Their archive
and content-tree checksums are recorded in `MANIFEST.json`.

Restore and verify both directories from the repository root:

```bash
python scripts/unpack_data_archives.py
```

Verify every archived payload without extracting it:

```bash
python scripts/unpack_data_archives.py --verify-only
```

The command refuses to merge into an existing `data/raw` or `data/processed`
directory. This protects the frozen snapshot from accidental overwrites.
