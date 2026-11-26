#!/usr/bin/env python3
"""Publish prebuilt models to Hugging Face, then prove every byte arrived.

The files to publish, the repo each belongs to and its SHA-256 all come from
the registry in scripts/fetch_models.sh, the same table users fetch from, so
the two cannot drift apart. The model card for a repo is
model_zoo/hf/<repo name>/README.md.

    python model_zoo/publish_hf.py --list                # what would go where
    python model_zoo/publish_hf.py rcdl-yolo             # dry run: diff local against the repo
    python model_zoo/publish_hf.py rcdl-yolo --upload    # create/update the repo, then verify
    python model_zoo/publish_hf.py --all --upload

Files are read from models/ (RCDL_MODELS overrides it), and the OCR dictionaries
from data/. A file whose SHA-256 differs from the registry is refused before
anything is uploaded. Change the registry first: it is what users will check
their downloads against.

Each file is its own commit. A large-folder upload that stalls loses
everything, and one file per commit means a retry only redoes what failed. A
file already on the repo with the same SHA-256 is skipped. After uploading,
every file's size and SHA-256 are read back from the repo. A partly uploaded
repo looks fine from outside, and an upload call does not always report a
failure.

Needs `huggingface_hub`, and a write token where it looks for one (`hf auth
login`, HF_TOKEN, or HF_TOKEN_PATH).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "scripts" / "fetch_models.sh"
CARDS = ROOT / "model_zoo" / "hf"
MODELS = Path(os.environ.get("RCDL_MODELS", ROOT / "models"))

# Files a repo carries that are not models in the registry: they live in data/,
# where the tests read them, and ship beside the models that need them.
EXTRA = {
    "rcdl-ppocr": ["data/ppocr_keys_v1_6625.txt",
                   "data/ppocr_keys_v5_18385.txt",
                   "data/ppocr_keys_v6_18710.txt"],
}

# .rknn is not in Hugging Face's default LFS patterns.
LFS_LINE = "*.rknn filter=lfs diff=lfs merge=lfs -text"

ENTRY = re.compile(r'^\s*"([^"|]+)\|hf\|([^"|]+)\|([0-9a-f]{64})"\s*$')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_registry() -> "OrderedDict[str, list[tuple[str, Path, str | None]]]":
    """repo id -> [(name in repo, local path, expected sha256 or None)]."""
    repos: "OrderedDict[str, list]" = OrderedDict()
    for line in REGISTRY.read_text().splitlines():
        m = ENTRY.match(line)
        if m:
            name, repo, digest = m.groups()
            repos.setdefault(repo, []).append((name, MODELS / name, digest))
    for repo, files in repos.items():
        for rel in EXTRA.get(repo.split("/")[-1], []):
            p = ROOT / rel
            files.append((p.name, p, None))
    return repos


def check_local(repo: str, files) -> list[str]:
    problems = []
    card = CARDS / repo.split("/")[-1] / "README.md"
    if not card.is_file():
        problems.append(f"no model card at {card.relative_to(ROOT)}")
    for name, path, want in files:
        if not path.is_file():
            problems.append(f"{name}: not found at {path}")
        elif want and sha256(path) != want:
            problems.append(f"{name}: SHA-256 differs from the registry")
    return problems


def remote_state(api, repo: str) -> dict:
    """name -> (size, sha256 or None) for what the repo holds now."""
    from huggingface_hub.errors import RepositoryNotFoundError
    try:
        info = api.model_info(repo, files_metadata=True)
    except RepositoryNotFoundError:
        return {}
    out = {}
    for s in info.siblings or []:
        lfs = getattr(s, "lfs", None)
        out[s.rfilename] = (s.size, lfs.sha256 if lfs else None)
    return out


def remote_sha(api, repo: str, name: str, state: dict) -> str | None:
    size, digest = state.get(name, (None, None))
    if digest or size is None:
        return digest
    # A small file stored as plain git has no LFS hash: fetch it and hash it.
    with tempfile.TemporaryDirectory() as tmp:
        p = api.hf_hub_download(repo, name, local_dir=tmp, force_download=True)
        return sha256(Path(p))


def ensure_lfs_rule(api, repo: str) -> None:
    from huggingface_hub.errors import EntryNotFoundError
    try:
        with tempfile.TemporaryDirectory() as tmp:
            p = api.hf_hub_download(repo, ".gitattributes", local_dir=tmp, force_download=True)
            text = Path(p).read_text()
    except EntryNotFoundError:
        text = ""
    if LFS_LINE in text:
        return
    text = text.rstrip("\n") + ("\n" if text else "") + LFS_LINE + "\n"
    api.upload_file(path_or_fileobj=text.encode(), path_in_repo=".gitattributes",
                    repo_id=repo, commit_message="Store .rknn files in LFS")


def publish(api, repo: str, files, upload: bool) -> bool:
    card = CARDS / repo.split("/")[-1] / "README.md"
    state = remote_state(api, repo)
    plan = []
    for name, path, _ in files + [("README.md", card, None)]:
        local = sha256(path)
        have = remote_sha(api, repo, name, state) if name in state else None
        plan.append((name, path, local, have == local))
    for name, path, _, same in plan:
        mb = path.stat().st_size / 1048576
        print(f"  {'same' if same else 'SEND'}  {name:<46} {mb:8.1f} MB")
    if not upload:
        return True

    api.create_repo(repo, repo_type="model", exist_ok=True)
    ensure_lfs_rule(api, repo)
    for name, path, _, same in plan:
        if same:
            continue
        print(f"  uploading {name} ...", flush=True)
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name, repo_id=repo,
                        commit_message=f"Add {name}" if name != "README.md" else "Update the model card")

    # Read everything back: sizes and hashes, not the upload calls' word for it.
    state = remote_state(api, repo)
    ok = True
    for name, path, local, _ in plan:
        size, _ = state.get(name, (None, None))
        got = remote_sha(api, repo, name, state) if name in state else None
        good = size == path.stat().st_size and got == local
        ok &= good
        print(f"  {'ok  ' if good else 'BAD '}  {name}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("repos", nargs="*", help="repo names, e.g. rcdl-yolo (owner optional)")
    ap.add_argument("--all", action="store_true", help="every repo in the registry")
    ap.add_argument("--list", action="store_true", help="print the registry's repos and files")
    ap.add_argument("--upload", action="store_true", help="upload; without it, only compare")
    args = ap.parse_args()

    registry = read_registry()
    if args.list:
        for repo, files in registry.items():
            print(repo)
            for name, _, _ in files:
                print(f"  {name}")
        return 0

    wanted = list(registry) if args.all else [
        next((r for r in registry if r == n or r.split("/")[-1] == n), n) for n in args.repos]
    if not wanted:
        ap.error("name a repo, or pass --all")
    unknown = [r for r in wanted if r not in registry]
    if unknown:
        ap.error(f"not in the registry: {', '.join(unknown)}")

    bad = {r: p for r in wanted if (p := check_local(r, registry[r]))}
    for repo, problems in bad.items():
        print(f"{repo}:", *problems, sep="\n  ")
    if bad:
        return 1

    from huggingface_hub import HfApi
    api = HfApi()
    print(f">> as {api.whoami()['name']}{'' if args.upload else ' (dry run)'}")
    ok = True
    for repo in wanted:
        print(repo)
        ok &= publish(api, repo, registry[repo], args.upload)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
