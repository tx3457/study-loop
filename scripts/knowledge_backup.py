#!/usr/bin/env python3
"""Offline backup/restore of the dedicated knowledge database and material directory.

Stop the knowledge service before backup. Restore requires an empty destination
database and directory and never drops existing tables or overwrites user files.
Database credentials stay in the database container, not in the archive manifest.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def docker(args, program, *options, **kwargs):
    return subprocess.run(
        ["docker", "exec", "-i", args.container, program, "-U", args.user,
         "-d", args.database, *options], check=True, **kwargs,
    )


def backup(args):
    if not args.service_stopped:
        raise SystemExit("Stop the knowledge service and pass --service-stopped first")
    if (not args.embedding_dimension or args.embedding_dimension <= 0
            or not args.embedding_model.strip() or not args.llm_model.strip()):
        raise SystemExit("backup requires embedding dimension, embedding model and LLM model")
    source = args.materials.resolve(strict=True)
    if not source.is_dir():
        raise SystemExit("materials must be a directory")
    target = args.archive.resolve()
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise SystemExit("backup destination must be empty")
    if target == source or source in target.parents:
        raise SystemExit("backup destination must be outside the material directory")
    with (target / "knowledge.dump").open("wb") as output:
        docker(args, "pg_dump", "-Fc", "--no-owner", stdout=output)
    with tarfile.open(target / "materials.tar.gz", "w:gz") as archive:
        archive.add(source, arcname="materials", recursive=True)
    config_rows = docker(
        args, "psql", "-At", "-v", "ON_ERROR_STOP=1", "-c",
        "SELECT COALESCE(json_agg(DISTINCT index_config_hash), '[]'::json) FROM sl_knowledge_bases",
        capture_output=True, text=True,
    )
    manifest = {
        "format": 1,
        "lightrag_version": "1.5.7",
        "database_image": "pgvector/pgvector:0.8.6-pg16",
        "database_image_digest": "sha256:ccc6e83d6e35e931dc7c5def2022729d5a6c370318d099181995567ff1fb4d6b",
        "index_config_hashes": json.loads(config_rows.stdout),
        "files": {name: digest(target / name) for name in ("knowledge.dump", "materials.tar.gz")},
        "configuration": {"embedding_dimension": args.embedding_dimension,
                          "embedding_model": args.embedding_model, "llm_model": args.llm_model},
    }
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("knowledge backup complete")


def restore(args):
    archive = args.archive.resolve(strict=True)
    manifest = json.loads((archive / "manifest.json").read_text())
    if manifest.get("format") != 1:
        raise SystemExit("unsupported backup format")
    for name in ("knowledge.dump", "materials.tar.gz"):
        if manifest.get("files", {}).get(name) != digest(archive / name):
            raise SystemExit("backup checksum mismatch")
    destination = args.materials.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit("restore material destination must be empty")
    result = docker(
        args, "psql", "-At", "-v", "ON_ERROR_STOP=1", "-c",
        "SELECT count(*) FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname NOT IN ('pg_catalog','information_schema') "
        "AND n.nspname NOT LIKE 'pg_toast%' AND c.relkind IN ('r','p')",
        capture_output=True, text=True,
    )
    if result.stdout.strip() != "0":
        raise SystemExit("restore database must contain no user tables")
    # Validate all archive names and links before restoring either resource.
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".knowledge-restore-", dir=destination.parent))
    with tarfile.open(archive / "materials.tar.gz", "r:gz") as bundle:
        for member in bundle.getmembers():
            name = Path(member.name)
            if (name.is_absolute() or ".." in name.parts or not name.parts
                    or name.parts[0] != "materials" or member.issym() or member.islnk()
                    or not (member.isfile() or member.isdir())):
                raise SystemExit("unsafe material archive")
        # Remove only the fixed archive root so callers choose the exact destination.
        for member in bundle.getmembers():
            relative = Path(member.name).parts[1:]
            if not relative:
                continue
            target = staging.joinpath(*relative)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)
    try:
        with (archive / "knowledge.dump").open("rb") as source:
            docker(args, "pg_restore", "--exit-on-error", "--single-transaction", "--no-owner", stdin=source)
    except BaseException:
        shutil.rmtree(staging)
        raise
    try:
        os.replace(staging, destination)
    except OSError:
        # Database committed atomically; retain extracted bytes for a publication-only retry.
        print(f"Database restored. Material publication failed; verified staging retained: {staging}")
        raise
    print("knowledge restore complete; verify model configuration before starting the service")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("backup", "restore"))
    parser.add_argument("--container", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--user", default="studyloop_graph")
    parser.add_argument("--materials", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--service-stopped", action="store_true")
    parser.add_argument("--embedding-dimension", type=int)
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--llm-model", default="")
    args = parser.parse_args()
    (backup if args.operation == "backup" else restore)(args)


if __name__ == "__main__":
    main()
