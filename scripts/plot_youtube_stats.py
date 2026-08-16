"""G3 (SESSIONS.md "YouTube data report" 2026-08-12) -- 6 charts for the YouTube
pilot corpus (`dataset/youtube-meetings/`, 7 meetings, 790 segments).

Runs on this repo's own Python (numpy + matplotlib only, no torch). Reuses
`is_vietnamese_shaped` (scripts/inspect_errors.py) and `LEXICAL_PARTICLES`
(src/config.py) rather than re-deriving a whitelist -- PROJECT_CORE.md notes
an English-word whitelist has a 72% false-positive rate on this data. The
non-Vietnamese-shaped filter (`_bare` + len>1 + isalpha()) is
`scripts/probe_youtube_captions.text_stats`'s, reused so this matches the
number already used to screen these videos in F1.

Charts 1-4 (code-switch, vocab, corpus comparison, segment duration) are
computed live from the 7 manifests every run. Charts 5-6 (overlap, speaker
similarity) need GPU models (pyannote/segmentation-3.0, speechbrain ECAPA)
this script's environment doesn't have, so their numbers are embedded as
constants below, measured once on 2026-08-12 (see SESSIONS.md G1/G2) --
`dataset/` is gitignored, so without embedding these two charts would break
for anyone who re-runs this script without the raw audio.
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.inspect_errors import is_vietnamese_shaped  # noqa: E402
from scripts.probe_youtube_captions import _bare  # noqa: E402
from src.config import LEXICAL_PARTICLES  # noqa: E402

OUT_DIR = ROOT / "docs" / "youtube-data-charts"

YOUTUBE_MANIFESTS = "dataset/youtube-meetings/manifest.*.jsonl"
PAID_V2_MANIFESTS = "dataset/paid-dataset-v2/manifest.*.jsonl"
REAL_BENCH_MANIFESTS = "dataset/real-meetings-bench/manifest.*.jsonl"
TEST_MEETINGS = ["3nuCdzuyqng", "7B24A9GfHAo"]
MEETING_ORDER = ["3nuCdzuyqng", "7B24A9GfHAo", "dGT3YW0AdD8", "iyeFAuuEBl4",
                  "rCd8DSMk3-c", "rIFrrmm8ILY", "xKDHjUoUN54"]

# --- G1 constants: overlap detection, pyannote/segmentation-3.0, measured
# 2026-08-12 on all 790 segment wavs.
#
# TWO metrics, because the reference points are not what a naive reading
# suggests. viet-speech's 16.4%-detected-vs-17.2%-ground-truth pair comes from
# osd_rate.py, which computes `sum(duration of VAD units whose
# OverlapLabel.overlapping is True) / sum(duration of all VAD speech units)` --
# the duration of FLAGGED UNITS, not the overlapping seconds themselves. Only
# G1_REFERENCE_FORMULA_PCT below reproduces that formula (silero-VAD units
# inside each clip, flagged against the same segmentation-3.0 timeline), so
# only it may be compared against those two reference lines.
# G1_OVERLAPPING_SEC_PCT is the stricter quantity -- seconds actually carrying
# >=2 simultaneous speakers, over clip duration -- and has no reference point.
G1_REFERENCE_FORMULA_PCT = {
    "3nuCdzuyqng": 3.73,
    "7B24A9GfHAo": 8.29,
    "dGT3YW0AdD8": 17.12,
    "iyeFAuuEBl4": 14.58,
    "rCd8DSMk3-c": 3.92,
    "rIFrrmm8ILY": 21.39,
    "xKDHjUoUN54": 6.49,
}
G1_OVERLAPPING_SEC_PCT = {
    "3nuCdzuyqng": 0.21,
    "7B24A9GfHAo": 0.65,
    "dGT3YW0AdD8": 2.13,
    "iyeFAuuEBl4": 1.38,
    "rCd8DSMk3-c": 0.32,
    "rIFrrmm8ILY": 1.21,
    "xKDHjUoUN54": 0.43,
}
G1_CORPUS_REFERENCE_FORMULA_PCT = 10.57
G1_CORPUS_OVERLAPPING_SEC_PCT = 0.89
G1_REFERENCE_DETECTED_PCT = 16.4
G1_REFERENCE_GROUND_TRUTH_PCT = 17.2

# --- G2 constants: speaker similarity, speechbrain/spkrec-ecapa-voxceleb,
# measured 2026-08-12 on the 40 least-overlapping segments per meeting (280
# total, selected using G1's per-segment output; scratchpad/speaker_sim.json
# this session). Mean cosine similarity per meeting pair.
G2_SIMILARITY_MEAN = {
    "3nuCdzuyqng": {"3nuCdzuyqng": 0.5110, "7B24A9GfHAo": 0.4019, "dGT3YW0AdD8": 0.3460,
                    "iyeFAuuEBl4": 0.3598, "rCd8DSMk3-c": 0.2239, "rIFrrmm8ILY": 0.2498,
                    "xKDHjUoUN54": 0.4899},
    "7B24A9GfHAo": {"3nuCdzuyqng": 0.4019, "7B24A9GfHAo": 0.4796, "dGT3YW0AdD8": 0.3700,
                     "iyeFAuuEBl4": 0.3600, "rCd8DSMk3-c": 0.2517, "rIFrrmm8ILY": 0.3072,
                     "xKDHjUoUN54": 0.3901},
    "dGT3YW0AdD8": {"3nuCdzuyqng": 0.3460, "7B24A9GfHAo": 0.3700, "dGT3YW0AdD8": 0.6545,
                     "iyeFAuuEBl4": 0.2840, "rCd8DSMk3-c": 0.2290, "rIFrrmm8ILY": 0.2830,
                     "xKDHjUoUN54": 0.3365},
    "iyeFAuuEBl4": {"3nuCdzuyqng": 0.3598, "7B24A9GfHAo": 0.3600, "dGT3YW0AdD8": 0.2840,
                     "iyeFAuuEBl4": 0.5650, "rCd8DSMk3-c": 0.3435, "rIFrrmm8ILY": 0.3582,
                     "xKDHjUoUN54": 0.3497},
    "rCd8DSMk3-c": {"3nuCdzuyqng": 0.2239, "7B24A9GfHAo": 0.2517, "dGT3YW0AdD8": 0.2290,
                     "iyeFAuuEBl4": 0.3435, "rCd8DSMk3-c": 0.4827, "rIFrrmm8ILY": 0.2841,
                     "xKDHjUoUN54": 0.1971},
    "rIFrrmm8ILY": {"3nuCdzuyqng": 0.2498, "7B24A9GfHAo": 0.3072, "dGT3YW0AdD8": 0.2830,
                     "iyeFAuuEBl4": 0.3582, "rCd8DSMk3-c": 0.2841, "rIFrrmm8ILY": 0.5280,
                     "xKDHjUoUN54": 0.2213},
    "xKDHjUoUN54": {"3nuCdzuyqng": 0.4899, "7B24A9GfHAo": 0.3901, "dGT3YW0AdD8": 0.3365,
                     "iyeFAuuEBl4": 0.3497, "rCd8DSMk3-c": 0.1971, "rIFrrmm8ILY": 0.2213,
                     "xKDHjUoUN54": 0.5672},
}
G2_GLOBAL_WITHIN_MEAN = 0.5411
G2_GLOBAL_CROSS_MEAN = 0.3160


def load_manifest_records(pattern: str) -> list[dict]:
    records = []
    for path in sorted(glob.glob(str(ROOT / pattern))):
        with open(path, encoding="utf-8") as f:
            records.extend(json.loads(line) for line in f)
    return records


def nonvn_tokens(text: str) -> list[str]:
    """Same filter as probe_youtube_captions.text_stats's `english` list."""
    bare = [_bare(t) for t in text.split()]
    return [b for b in bare if len(b) > 1 and b.isalpha() and not is_vietnamese_shaped(b)]


def word_level_stats(records: list[dict]) -> dict:
    n_words = 0
    n_nonvn = 0
    n_particle = 0
    for r in records:
        toks = r["text"].split()
        bare = [_bare(t) for t in toks]
        n_words += len(toks)
        n_nonvn += sum(1 for b in bare if len(b) > 1 and b.isalpha() and not is_vietnamese_shaped(b))
        n_particle += sum(1 for b in bare if b in LEXICAL_PARTICLES)
    return {
        "n_words": n_words,
        "nonvn_rate": n_nonvn / n_words if n_words else 0.0,
        "particle_rate": n_particle / n_words if n_words else 0.0,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    yt_records = load_manifest_records(YOUTUBE_MANIFESTS)
    n_meetings = len({r["meeting_id"] for r in yt_records})
    assert n_meetings == 7, f"expected 7 meetings, found {n_meetings}"
    assert len(yt_records) == 790, f"expected 790 segments, found {len(yt_records)}"

    total_min = sum(r["duration"] for r in yt_records) / 60
    yt_stats = word_level_stats(yt_records)
    n_seg_with_codeswitch = sum(1 for r in yt_records if nonvn_tokens(r["text"]))

    print(f"segments: {len(yt_records)}")
    print(f"total duration: {total_min:.2f} min")
    print(f"words: {yt_stats['n_words']}")
    print(f"non-Vietnamese-shaped rate: {yt_stats['nonvn_rate']:.2%}")
    print(f"segments with >=1 code-switch token: {n_seg_with_codeswitch}/{len(yt_records)} "
          f"({n_seg_with_codeswitch / len(yt_records):.1%})")
    print("(SESSIONS.md 'So da do' quotes 6.70% / 89.7% off a 41,043-word denominator this "
          "script cannot reconstruct from the current manifests -- word count 41,210, "
          "non-Vietnamese-shaped token count 2,748 and segment share 709/790 all match "
          "exactly, so the 0.03pp gap is a stale denominator in that note, not a script bug.")

    # ---- Chart 1: code-switch tokens per segment (histogram, bins 0..16)
    per_seg_counts = [len(nonvn_tokens(r["text"])) for r in yt_records]
    max_count = max(per_seg_counts)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_seg_counts, bins=range(0, max_count + 2), align="left",
            color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Số token không mang hình dạng tiếng Việt / segment")
    ax.set_ylabel("Số segment")
    ax.set_title(f"Code-switch trên mỗi segment (n={len(yt_records)} segment, 7 meeting)")
    ax.set_xticks(range(0, max_count + 1))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "codeswitch-per-segment.png", dpi=150)
    plt.close(fig)

    # ---- Chart 2: top-25 code-switch vocabulary (types in >=3/7 meetings)
    type_counts: Counter = Counter()
    type_meetings: dict[str, set[str]] = {}
    for r in yt_records:
        for tok in nonvn_tokens(r["text"]):
            type_counts[tok] += 1
            type_meetings.setdefault(tok, set()).add(r["meeting_id"])
    ge3 = {t: c for t, c in type_counts.items() if len(type_meetings[t]) >= 3}
    top25 = sorted(ge3.items(), key=lambda kv: -kv[1])[:25]
    labels = [f"{t} ({len(type_meetings[t])} meeting)" for t, _ in top25]
    counts = [c for _, c in top25]
    fig, ax = plt.subplots(figsize=(8, 9))
    y = np.arange(len(top25))
    ax.barh(y, counts, color="#55A868")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Số lần xuất hiện")
    ax.set_title(f"Top 25 từ code-switch xuất hiện ở >=3/7 meeting\n"
                 f"({len(ge3)} type, {sum(ge3.values())} lần)")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "codeswitch-vocab.png", dpi=150)
    plt.close(fig)

    # ---- Chart 3: corpus comparison (3 corpus x 2 metrics)
    paid_v2_stats = word_level_stats(load_manifest_records(PAID_V2_MANIFESTS))
    real_bench_stats = word_level_stats(load_manifest_records(REAL_BENCH_MANIFESTS))
    corpora = [
        ("YouTube pilot\n(790 seg)", yt_stats),
        ("paid-dataset-v2\n(synthetic)", paid_v2_stats),
        ("real-meetings-bench\n(real, post-edited)", real_bench_stats),
    ]
    x = np.arange(len(corpora))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, [c["nonvn_rate"] * 100 for _, c in corpora], width,
           label="Non-Vietnamese-shaped (%)", color="#4C72B0")
    ax.bar(x + width / 2, [c["particle_rate"] * 100 for _, c in corpora], width,
           label="Lexical-particle (%)", color="#C44E52")
    ax.set_xticks(x)
    ax.set_xticklabels([name for name, _ in corpora])
    ax.set_ylabel("% token")
    ax.set_title("Đối chiếu 3 corpus: code-switch & particle rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "corpus-comparison.png", dpi=150)
    plt.close(fig)

    # ---- Chart 4: segment duration histogram (1s bins, 3-23s)
    durations = [r["duration"] for r in yt_records]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(durations, bins=np.arange(3, 24, 1), color="#8172B2", edgecolor="white")
    ax.set_xlabel("Thời lượng segment (s)")
    ax.set_ylabel("Số segment")
    ax.set_title(f"Phân phối thời lượng segment (min={min(durations):.1f}s, "
                 f"max={max(durations):.1f}s, n={len(durations)})")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "segment-duration.png", dpi=150)
    plt.close(fig)

    # ---- Chart 5: overlap per meeting (G1 constants), both metrics side by
    # side -- only the reference-formula series is comparable to the two
    # reference lines (see the constants block above).
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(MEETING_ORDER))
    width = 0.38
    ax.bar(x - width / 2, [G1_REFERENCE_FORMULA_PCT[m] for m in MEETING_ORDER], width,
           label="Đơn vị VAD bị gắn cờ / tổng tiếng nói (công thức tham chiếu)",
           color="#4C72B0")
    ax.bar(x + width / 2, [G1_OVERLAPPING_SEC_PCT[m] for m in MEETING_ORDER], width,
           label="Giây thật sự chồng tiếng / thời lượng clip", color="#DD8452")
    ax.axhline(G1_REFERENCE_DETECTED_PCT, color="gray", linestyle="--",
               label=f"Tham chiếu viet-speech: model đo {G1_REFERENCE_DETECTED_PCT}%")
    ax.axhline(G1_REFERENCE_GROUND_TRUTH_PCT, color="black", linestyle=":",
               label=f"Tham chiếu viet-speech: ground truth {G1_REFERENCE_GROUND_TRUTH_PCT}%")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(test)" if m in TEST_MEETINGS else m for m in MEETING_ORDER])
    ax.set_ylabel("%")
    ax.set_title(f"Chồng tiếng theo meeting (pyannote/segmentation-3.0)\n"
                 f"Toàn corpus: {G1_CORPUS_REFERENCE_FORMULA_PCT}% (công thức tham chiếu) "
                 f"· {G1_CORPUS_OVERLAPPING_SEC_PCT}% (giây chồng tiếng)", fontsize=11)
    ax.legend(fontsize=7.5)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "overlap-per-meeting.png", dpi=150)
    plt.close(fig)

    # ---- Chart 6: speaker similarity heatmap (G2 constants)
    matrix = np.array([[G2_SIMILARITY_MEAN[a][b] for b in MEETING_ORDER] for a in MEETING_ORDER])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=matrix.max())
    ax.set_xticks(range(len(MEETING_ORDER)))
    ax.set_yticks(range(len(MEETING_ORDER)))
    ax.set_xticklabels(MEETING_ORDER, rotation=45, ha="right")
    ax.set_yticklabels(MEETING_ORDER)
    for i in range(len(MEETING_ORDER)):
        for j in range(len(MEETING_ORDER)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                     color="white" if matrix[i, j] < matrix.max() * 0.6 else "black", fontsize=8)
    for m in TEST_MEETINGS:
        idx = MEETING_ORDER.index(m)
        ax.add_patch(plt.Rectangle((idx - 0.5, -0.5), 1, len(MEETING_ORDER),
                                    fill=False, edgecolor="red", linewidth=2))
        ax.add_patch(plt.Rectangle((-0.5, idx - 0.5), len(MEETING_ORDER), 1,
                                    fill=False, edgecolor="red", linewidth=2))
    fig.colorbar(im, ax=ax, label="cosine similarity (mean)")
    ax.set_title(f"Tương đồng giọng nói giữa 7 meeting (ECAPA)\n"
                 f"within-meeting mean {G2_GLOBAL_WITHIN_MEAN:.2f} > "
                 f"cross-meeting mean {G2_GLOBAL_CROSS_MEAN:.2f} -- viền đỏ = meeting split test",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "speaker-similarity.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote 6 PNGs to {OUT_DIR}")


if __name__ == "__main__":
    main()
