"""Bảng soát chất lượng cho phần YouTube đi vào train của `dataset/v6-corpus`.

`scripts/review_youtube.py` là vòng sửa nhãn cho **một** cuộc họp của bộ pilot:
mọi đoạn đều hiện ra, người soát gõ lại từng câu. Ở quy mô 11.586 đoạn / 47
cuộc họp thì cách đó không dùng được -- việc cần làm không phải gõ lại toàn bộ
mà là khoanh vùng chỗ đáng ngờ rồi nghe kiểm chứng. Nên tool này gắn cờ theo
luật, cho lọc theo cờ / cuộc họp / người soát, và chỉ dựng một thẻ <audio> duy
nhất dùng lại cho mọi đoạn (11.586 thẻ <audio> cùng lúc thì trình duyệt chết).

Phạm vi: `source == "youtube"` và `split_orig` không phải val/test -- val/test
là phần quyết định con số CER báo cáo, soát theo quy trình khác.

Đường dẫn audio là file:/// tuyệt đối, cùng cách `review_youtube.build_html`
làm, nên trang mở thẳng bằng file:// không cần server.

Chữ hoa bị gắn cờ vì `youtube-data-pilot/style-guide.md` chốt nhãn tiếng nói
thật viết thường toàn bộ; Google ASR viết hoa giữa câu không theo luật nào.
Từ ngoài tiếng Việt đếm bằng `is_vietnamese_shaped` (PROJECT_CORE.md §4), không
dùng whitelist tiếng Anh -- whitelist sai 72%.

Kết quả chấm nằm trong localStorage của trình duyệt; nút "Xuất JSONL" tải về
một dòng cho mỗi đoạn đã chấm.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.plot_youtube_stats import nonvn_tokens  # noqa: E402

CORPUS_DIR = ROOT / "dataset" / "v6-corpus"
OUT_PATH = ROOT / "youtube-data-pilot" / "review" / "qc-v6-youtube.html"

# Ngưỡng: 130/11.586 đoạn có mật độ chữ dưới 4 ký tự/giây, trong khi bách phân
# vị 1% của cả bộ là 3,65 -- dưới ngưỡng này gần như luôn là caption thiếu chữ
# so với phần audio, không phải người nói chậm.
MIN_CHARS_PER_SEC = 4.0
LONG_SEGMENT_SEC = 25.0
NONVN_TOKEN_THRESHOLD = 3

FLAG_LABELS = {
    "chua_soat": "chưa ai soát",
    "chi_soat_text": "soát text, không nghe",
    "may_soat": "máy soát",
    "overlap": "chồng tiếng",
    "text_rong": "text rỗng",
    "text_thua_thoi": "text quá ít so với audio",
    "lap_tu": "lặp từ liên tiếp",
    "trung_lap": "trùng đoạn khác cùng cuộc họp",
    "co_chu_hoa": "có chữ hoa",
    "co_chu_so": "có chữ số",
    "tu_ngoai_lai": f"≥{NONVN_TOKEN_THRESHOLD} từ ngoài tiếng Việt",
    "dai_bat_thuong": f"dài hơn {LONG_SEGMENT_SEC:.0f}s",
}


def load_youtube_train(corpus_dir: Path) -> list[dict]:
    records = []
    for path in sorted(corpus_dir.glob("manifest.*.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                if r.get("source") == "youtube" and r.get("split_orig") not in ("val", "test"):
                    records.append(r)
    if not records:
        raise FileNotFoundError(f"không có bản ghi youtube nào trong {corpus_dir}")
    return records


def _has_token_loop(text: str) -> bool:
    words = text.split()
    return any(words[i] == words[i + 1] == words[i + 2] == words[i + 3]
               for i in range(len(words) - 3))


def flags_for(record: dict, duplicate_texts: set[tuple[str, str]]) -> list[str]:
    text = record["text"]
    stripped = text.strip()
    reviewer = record.get("reviewed_by", "") or ""
    duration = record["duration"]
    flags = []

    if not reviewer:
        flags.append("chua_soat")
    elif "text-only" in reviewer:
        flags.append("chi_soat_text")
    elif reviewer.startswith("claude-"):
        flags.append("may_soat")

    if record.get("quality") == "overlap":
        flags.append("overlap")
    if not stripped:
        flags.append("text_rong")
    elif duration > 0 and len(text) / duration < MIN_CHARS_PER_SEC:
        flags.append("text_thua_thoi")
    if _has_token_loop(text):
        flags.append("lap_tu")
    if stripped and (record["meeting_id"], stripped) in duplicate_texts:
        flags.append("trung_lap")
    if any(c.isupper() for c in text):
        flags.append("co_chu_hoa")
    if re.search(r"\d", text):
        flags.append("co_chu_so")
    if len(nonvn_tokens(text)) >= NONVN_TOKEN_THRESHOLD:
        flags.append("tu_ngoai_lai")
    if duration > LONG_SEGMENT_SEC:
        flags.append("dai_bat_thuong")
    return flags


def build_payload(records: list[dict], audio_root: Path) -> dict:
    counts = collections.Counter((r["meeting_id"], r["text"].strip()) for r in records)
    duplicate_texts = {k for k, c in counts.items() if c > 1 and k[1]}

    segments = []
    for r in records:
        segments.append({
            "m": r["meeting_id"],
            "s": r["segment_id"],
            "t": r["text"],
            "d": round(r["duration"], 2),
            "a": r["audio_filepath"].replace("\\", "/"),
            "r": r.get("reviewed_by", "") or "",
            "v": bool(r.get("verified")),
            "u": r.get("video_url", ""),
            "y": round(r.get("yt_start", 0.0)),
            "f": flags_for(r, duplicate_texts),
        })
    segments.sort(key=lambda s: (s["m"], s["s"]))

    meetings = collections.OrderedDict()
    for s in segments:
        m = meetings.setdefault(s["m"], {
            "id": s["m"], "n": 0, "sec": 0.0, "reviewers": collections.Counter(),
            "verified": 0, "flags": collections.Counter(),
        })
        m["n"] += 1
        m["sec"] += s["d"]
        m["reviewers"][s["r"] or "(rỗng)"] += 1
        m["verified"] += int(s["v"])
        for f in s["f"]:
            m["flags"][f] += 1

    return {
        "audio_root": audio_root.resolve().as_uri().rstrip("/") + "/",
        "flag_labels": FLAG_LABELS,
        "segments": segments,
        "meetings": [
            {**m, "reviewers": dict(m["reviewers"]), "flags": dict(m["flags"]),
             "sec": round(m["sec"], 1)}
            for m in meetings.values()
        ],
    }


def build_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    data = data.replace("</", r"<\/")     # payload nằm trong <script>, đừng để "</script" trong text cắt thẻ
    template = (Path(__file__).parent / "qc_v6_youtube.template.html").read_text(encoding="utf-8")
    return template.replace("/*__PAYLOAD__*/null", data)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")     # tiếng Việt trên console cp1252
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", type=Path, default=CORPUS_DIR)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    records = load_youtube_train(args.corpus_dir)
    payload = build_payload(records, args.corpus_dir / "audio")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(payload), encoding="utf-8")

    total_flags = collections.Counter(f for s in payload["segments"] for f in s["f"])
    hours = sum(s["d"] for s in payload["segments"]) / 3600
    print(f"{len(payload['segments'])} đoạn / {len(payload['meetings'])} cuộc họp / {hours:.2f} h")
    for flag, n in total_flags.most_common():
        print(f"  {FLAG_LABELS[flag]:<34} {n:5d}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
