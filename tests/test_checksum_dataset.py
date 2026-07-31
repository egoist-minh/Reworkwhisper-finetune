"""Tests for scripts/checksum_dataset.py, using a small synthetic tree so this
doesn't depend on the real (2 GB) dataset being present."""

from pathlib import Path

from scripts.checksum_dataset import generate, verify


def _make_tree(root: Path) -> None:
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("hello", encoding="utf-8")
    (root / "sub" / "b.txt").write_text("world", encoding="utf-8")


def test_generate_then_verify_clean(tmp_path):
    _make_tree(tmp_path)
    n = generate(tmp_path, tmp_path / "CHECKSUMS.txt")
    assert n == 2
    assert verify(tmp_path, tmp_path / "CHECKSUMS.txt") == []


def test_verify_detects_mismatch(tmp_path):
    _make_tree(tmp_path)
    generate(tmp_path, tmp_path / "CHECKSUMS.txt")
    (tmp_path / "a.txt").write_text("tampered", encoding="utf-8")
    problems = verify(tmp_path, tmp_path / "CHECKSUMS.txt")
    assert any("MISMATCH: a.txt" in p for p in problems)


def test_verify_detects_missing_file(tmp_path):
    _make_tree(tmp_path)
    generate(tmp_path, tmp_path / "CHECKSUMS.txt")
    (tmp_path / "a.txt").unlink()
    problems = verify(tmp_path, tmp_path / "CHECKSUMS.txt")
    assert any("MISSING: a.txt" in p for p in problems)
