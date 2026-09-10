"""Zip a manifest corpus for upload as a Kaggle Dataset, with POSIX separators.

Written because Windows zip tools have shipped entry names with `\\` separators
instead of `/`. The ZIP spec requires `/`; an archive built the other way extracts
on Kaggle as ONE flat file literally named `IGZYBrbDUEw\\seg_0000.wav`, so
`audio_root / r["audio_filepath"]` -- which joins with `/` -- finds nothing, and the
failure surfaces hours later as a FileNotFoundError mid-decode rather than at upload.

Every arcname is written explicitly with `/`, then the finished archive is reopened
and refused if any entry still carries a backslash.

Only `manifest.*.jsonl` and `audio/` go in. Everything else in a corpus directory
(`raw/`, `review/`, worksheets, notes) is either large or not needed to score.

    python -m scripts.zip_for_kaggle --src dataset/youtube-data-pilot-package \\
        --out Outputs/cross-domain-bench.zip
"""

import argparse
import sys
import zipfile
from pathlib import Path


def collect(src: Path) -> list[tuple[Path, str]]:
    """(absolute file, arcname). Arcnames are relative to `src` with `/` separators."""
    manifests = sorted(src.glob("manifest.*.jsonl"))
    if not manifests:
        raise FileNotFoundError(f"no manifest.*.jsonl at the root of {src}")
    audio_dir = src / "audio"
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"{audio_dir} missing")

    items = [(m, m.name) for m in manifests]
    wavs = sorted(p for p in audio_dir.rglob("*.wav") if p.is_file())
    if not wavs:
        raise FileNotFoundError(f"no .wav under {audio_dir}")
    items += [(w, w.relative_to(src).as_posix()) for w in wavs]
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="corpus directory (manifest.*.jsonl + audio/)")
    ap.add_argument("--out", required=True, help="zip file to write")
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    src = Path(args.src)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    items = collect(src)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, arcname in items:
            zf.write(path, arcname)

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    bad = [n for n in names if "\\" in n]
    if bad:
        raise RuntimeError(f"{len(bad)} entries use a backslash separator, e.g. {bad[:3]}")

    tops = sorted({n.split("/")[0] for n in names if "/" in n})
    print(f"wrote {out} | {len(names)} entries | {out.stat().st_size / 1e6:.1f} MB")
    print(f"manifests: {sum(1 for n in names if n.startswith('manifest.'))} | "
          f"audio dirs: {tops}")
    print("separator check: OK (every entry uses `/`)")
    print(f"\nUpload {out} as a Kaggle Dataset. It extracts to "
          f"manifest.*.jsonl + audio/<meeting_id>/*.wav -- which is what "
          f"--path <suite>=<dir> expects.")


if __name__ == "__main__":
    main()
