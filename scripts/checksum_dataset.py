"""Generate / verify dataset/CHECKSUMS.txt -- the only file under dataset/ that
is tracked in git (see .gitignore). `dataset/` itself is not tracked (too
large, differs per platform), so this file is the sole proof that two
platforms (e.g. local vs Kaggle) ran on identical bytes -- PROJECT_CORE.md §0.
"""

import argparse
import hashlib
from pathlib import Path


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def generate(root: Path, out_path: Path) -> int:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.name != "CHECKSUMS.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        for p in files:
            rel = p.relative_to(root).as_posix()
            f.write(f"{_hash_file(p)}  {rel}\n")
    return len(files)


def verify(root: Path, checksums_path: Path) -> list[str]:
    """Returns a list of mismatch/missing descriptions; empty means clean."""
    problems = []
    expected = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, _, rel = line.partition("  ")
        expected[rel] = digest

    for rel, digest in expected.items():
        p = root / rel
        if not p.exists():
            problems.append(f"MISSING: {rel}")
            continue
        actual = _hash_file(p)
        if actual != digest:
            problems.append(f"MISMATCH: {rel} (expected {digest}, got {actual})")
    return problems


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset")
    ap.add_argument("--mode", choices=["generate", "verify"], default="verify")
    args = ap.parse_args()

    root = Path(args.root)
    out = root / "CHECKSUMS.txt"

    if args.mode == "generate":
        n = generate(root, out)
        print(f"wrote {n} checksums -> {out}")
    else:
        problems = verify(root, out)
        if problems:
            for p in problems:
                print(p)
            raise SystemExit(f"{len(problems)} problem(s) -- dataset does not match CHECKSUMS.txt")
        print("OK -- dataset matches CHECKSUMS.txt")
