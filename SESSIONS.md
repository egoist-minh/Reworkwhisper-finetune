# SESSIONS

Nhật ký task của pipeline PhoWhisper. Spec ổn định (module contract, data flow, config schema,
luật gate) ở `PROJECT_CORE.md`; luật hành vi ở `CLAUDE.md`; cách chạy từng bước ở `docs/runbook.md`.

**Bản đầy đủ nằm ở commit `192c4ef`** (`git show 192c4ef:SESSIONS.md`, 114 KB). File này là bản
co gọn 2026-09-09: giữ nguyên **mọi row ID** (35 file khác trỏ vào đây theo ID), mọi số đo, mọi
quyết định và mọi cổng dừng — cắt phần tường thuật "đã kiểm những gì". Cần chi tiết một row thì
đọc bản đầy đủ.

## Status legend
`todo` → chưa bắt đầu · `in-progress` → dở · `done` → deliverable tồn tại và đạt Acceptance ·
`blocked` → chờ dependency · `deferred` → cố ý cắt khỏi deadline, không tự mở lại

---

## Deadline scope cut (2026-08-02) — đọc trước khi nhận bất kỳ row `todo` nào

Chốt 2026-07-31, người dùng xác nhận. Không mở lại nếu không có chữ ký của người dùng.

- **`lora.rank: 16`** cho run đầu (không phải 64 — ví dụ ở PROJECT_CORE §3 là minh hoạ).
  `configs/experiment.yaml` là nguồn sự thật.
- **Tier 3 (RTF) bỏ, tier 4b (long-form) hoãn.** Gate ship cho deadline này là tier **1, 2, 4a**.
  Không thêm `rtf_threshold`/`longform_*` lại mà không bàn scope trước.
- **Tier 4a (real-meeting bench) BẮT BUỘC**, không phải cơ hội. `data.real_bench_path` =
  `dataset/real-meetings-bench`, ingest từ `done/` qua `scripts/ingest_real_bench.py`.
- **`training.full_finetune: false`** cho run đầu → đường SVD (§6 Stage 3 Path B,
  `min_retained_energy`) không cần để ship. Vẫn là đường có tài liệu nhưng chưa dựng.
- **Chỉ dùng `paid-dataset`.** `unpaid-dataset` và `dataset_by_task` ngoài scope deadline này.
- **`paid-dataset` đã migrate vào `Fine_tune_wf/dataset/`**, 2.0 GB, checksum-verified qua
  `scripts/checksum_dataset.py` → `dataset/CHECKSUMS.txt` (2.237 file, file duy nhất dưới
  `dataset/` được track trong git).
- **Pipeline phải push và lưu bằng chứng thật, không chỉ in số.** `src/hub.py:push_adapter` chỉ
  bắn khi `overall_pass` và `hub.push: true`; `src/gate.py` ghi `audit/predictions_{tier}.csv`
  cho mọi lần eval, cả baseline lẫn gate.
- **Ranh giới module lệch so với bản plan gốc, và đó là chốt:** `normalize.py` (không phải
  `metrics.py`) giữ contract §6 Stage 4; λ scaling ở `lora.py` (Path A, không có `model.py`);
  chọn λ inline trong `pipeline.py:stage_sweep_gate` (không có `sweep.py`); `report.py` **không
  dựng**.

---

## Track A/B/D — nền móng, bring-up, tích hợp (2026-07-31 → 08-01)

Track A = module thuần Python, CPU-testable, no GPU. Track B = bring-up GPU. Track D = tích hợp.
Track C của plan gốc gộp vào A/D: logic gate + sweep đã là hàm đủ thuần trong `gate.py`/`pipeline.py`.

| # | Việc | Trạng thái |
|---|---|---|
| **A1** | Bootstrap — `git init`, `.gitignore`, `requirements.txt`, remote GitHub, migrate `paid-dataset` + `CHECKSUMS.txt` | **done** — repo `github.com/egoist-minh/Reworkwhisper-finetune` (private). Split resolve xác nhận trên bản local: **1316/250/236** |
| **A3** | Config schema — `src/config.py`, `configs/experiment.yaml` | **done**. Từ chối key lạ, rank/alpha sai, filler token là lexical particle, rò tier-4, sweep grid rỗng. `tests/test_config.py` viết sau |
| **A4** | Data module — `src/data.py`: merge manifest, resolve split, resample 24k→16k | **done**, 4 test. Val meetings resolve `[0001, 0002, 0011]`; `meeting_id` trùng thì raise |
| **A5** | Normalization + metrics — `src/normalize.py` (§6 Stage 4, word→digit), `src/metrics.py` (Levenshtein, CER/WER, bootstrap CI, paired delta CI) | **done**, 9 test. CER/WER tính **corpus-level**, không phải mean per-segment; delta CI có ghép cặp. Bug `hai ba nghìn`→`23000` sửa ở đây. `code-switch tagging` và `measure_rtf()` **không dựng** — hoãn |
| **A6** | LoRA + λ (Path A) — `src/lora.py`: `set_lambda` lọc `LoraLayer` (không dùng `hasattr`), `save_with_lambda` bake vào `adapter_config.json` và verify bằng reload trong 1e-5 | **done**. Không có literal rank ở đâu, chỉ `cfg.lora.rank`. Path B (SVD full-FT) **deferred** |
| **A2** | VIVOS (OOD) — `scripts/fetch_vivos.py`, đường parquet chính + tarball dự phòng | **done** sau khi sửa bug #6 dưới |
| **A2b** | Real-meetings benchmark — `scripts/ingest_real_bench.py`, cắt lại span >30 s | **done** — 196 segment thô → **264** sau khi cắt lại 20 span quá dài (real_0001: 146, real_0002: 118), 29.579 ký tự. **Caveat mang theo mãi:** text/audio bên trong span cắt lại là **xấp xỉ theo tỉ lệ thời lượng**, không phải forced alignment — đủ cho CER gộp tier 4a, không đủ cho bất cứ gì nói về mốc thời gian mức từ |
| **A2c** | Reference audit + freeze | **deferred** — 6 caveat §4 (reference là bản post-edit output PhoWhisper-small, quy ước số lẫn lộn, segment tới 212 s) vẫn áp dụng, chỉ chưa verify lại trên bản đã cắt |
| **A7** | Hub push + prediction evidence — `src/hub.py:push_adapter` (model card có bảng gate CER, không token trong artifact nào) | **done** |
| **B1** | Kaggle smoke — `--stage smoke` trên Kaggle T4 | **done** 2026-08-01 |
| **B2** | Baseline eval | **done** 2026-08-01. `cer_test=0.1117`, `cer_ood=0.0067`, `cer_real=0.4676` |
| **B3** | Train — LoRA SFT rank 16, 3 epoch | **done** 2026-08-01, `train_loss=2.388`. Bug còn lại: `metric_for_best_model="val_cer"` không khớp key có prefix của multi-eval-dataset → `EarlyStoppingCallback` tự tắt. Vô hại run này vì `patience=3 == epochs=3` |
| **D1** | Sweep + Gate + Push | **done** 2026-08-01. tier2 pass (2.00% ≤ 2.6676%), tier1 pass (2.28%), **tier4a FAIL** (55.81% so với bound 46.76%, zero-tolerance) → `overall_pass: false` → push tự động **đúng là không bắn**. Người dùng quyết định push tay qua `src.hub.push_adapter` ngoài gate, model card vẫn hiện bảng FAIL trung thực, thay vì nới ngưỡng |
| **D2** | E2E confirmation — chứng minh đường halt + no-push | **done, do tai nạn thật chứ không phải test dựng** — run D1 fail tier4a thật và hành xử đúng |

**9 bug Kaggle của đợt bring-up** (đánh số ở đây là số được trích ở nơi khác — "bug #7" trong
`youtube-data-pilot/README.md` là mục 7 dưới đây)**, tất cả chỉ lộ ra từ traceback thật, không cái
nào từ review tĩnh ở local (máy dev không có torch/transformers/peft):** (1) tên file manifest
VIVOS không khớp glob
của `load_manifests`; (2) `ManifestDataset.__getitem__` đòi key mà record VIVOS không có; (3)
`compat.apply()` chỉ được gọi trong `stage_smoke`; (4) `Trainer(remove_unused_columns=True)` bóc
mất key input của collator; (5) thiếu `enable_input_require_grads()` cho PEFT + gradient
checkpointing; (6) `fetch_vivos.py` chọn nhầm cột `speaker_id` làm text thay vì `sentence` —
baseline `cer_ood` đọc ra **358%** cho tới khi sửa; (7) vòng lặp vô hạn trong
`normalize.py:words_to_digits` khi gặp một từ zero-filler đơn lẻ như tên "Linh"; (8)
`PeftModel.generate()` từ chối positional arg; (9) tensor giả của `_verify_saved_adapter` mặc định
fp32 chọi lại model fp16. Xem memory `kaggle-code-never-works-first-try`.

---

## E1–E5 — sửa pipeline sau grill v3-r16 (2026-08-04)

Nguồn: một phiên grill kết quả v3-r16, tìm ra bằng cách đọc tay predictions CSV chứ không phải
bằng tooling có sẵn — chính khoảng trống đó là E5.

| # | Việc | Trạng thái |
|---|---|---|
| **E1** | Kiểm tốc độ nói lúc ingest — `len(text)/duration` so với ngưỡng hợp lý, raise chứ không âm thầm bỏ | **done** — `_check_speech_rate`, `MAX_CHARS_PER_SEC = 60.0` (dư trên 39.5 tệ nhất mà thật, dưới xa 475.7 của ca bug). Raise đúng ở `real_0002/seg_0074`. **Bug rejoin/chia-theo-tỉ-lệ gây ra seg_0074 KHÔNG sửa ở đây** — E1 chỉ có nhiệm vụ phát hiện |
| **E2** | Chọn λ theo cost/benefit — thay "λ lớn nhất trong ngân sách" bằng luật elbow | **done** — `select_lambda()` trong `pipeline.py`, `Sweep.elbow_ratio_threshold: float = 10.0`. Trên sweep thật của v3-r16: jump 8.9× ở λ=0.5 (nhận) so với 12.6× ở λ=0.75 (loại) → chọn 0.5, không phải 1.0, dù 1.0 vẫn trong ngân sách (0.0186 < 0.02). Con số "~31× ở 1.0" của bản grill tay **không tái hiện được** dưới bất kỳ công thức adjacent-step nào (gần nhất 2.55×) — nhiều khả năng là artifact làm tròn |
| **E3** | Wire `audit_conversions` cho tier2_ood | todo |
| **E4** | Lưu predictions từng λ trong sweep, không chỉ λ được chọn | todo |
| **E5** | Công cụ soi lỗi dùng lại được — `scripts/inspect_errors.py` | **done, có khoảng hở trung thực ở 2/3 số**. `rank_by_edit_chars`, `top_segment_edit_share`, `loanword_dropped_segments`, `digit_mismatch_segments`. **Digit-mismatch tái hiện chính xác: 42/183** trên VIVOS. Hai số kia không tái hiện bit-exact vì code scratch gốc đã mất: top-segment share đo **19.0%** (không phải 23%), loanword-drop đo **59/266 = 22.2%** (ghi chú cũ 83/361). **Ghi chú quan trọng:** pattern 1-phụ-âm/1-nguyên-âm nguyên văn của PROJECT_CORE §4 gắn cờ ~99% từ Việt thật (`không`, `được`) là non-Vietnamese — phải thêm digraph onset + nucleus đôi/ba mới dùng được |

Thứ tự đề xuất: E1 + E5 trước (CPU thuần), E2 tiếp (cũng CPU, chạy trên sweep data đã tải),
E3 + E4 gộp vào phiên Kaggle của run train tiếp theo.

---

## F1–F4 — YouTube data pilot, thu thập (2026-08-11 → 12)

What/how-much/why chốt ở `youtube-data-pilot/README.md`. Cách chạy ở `docs/runbook.md` Phần A.
Cả 4 row chạy **local** (CPU + network), không phải Kaggle.

**Hai delta so với README, chốt 2026-08-11:**

- **Nhãn nháp lấy từ auto-caption của Google, không phải ElevenLabs Scribe.** Miễn phí, và json3
  mang **mốc thời gian từng từ** — xoá nguyên lớp bug mà README bước 2 cảnh báo (`done/` chỉ có mốc
  cấp segment nên `ingest_real_bench.py` phải chia text theo tỉ lệ, chính xấp xỉ đó đẻ ra
  `real_0002/seg_0074` ở 475.7 ký tự/giây). Scribe vẫn swap được bất cứ lúc nào qua `--draft-source`.
- **Quy ước số lật sang chữ số cho giá trị rõ ràng là số** (README §5 cũ ghi "viết bằng chữ"). Đo
  trên `paid-dataset-v2`: 529/4946 segment (10.7%) có chữ số, 1022 (20.7%) có số viết chữ, 183 có
  cả hai. CER không đổi (`number_convention: word_to_digit` áp đối xứng); khác biệt thuần là công
  gõ của người soát.

| # | Việc | Trạng thái |
|---|---|---|
| **F1** | Caption probe + lớp draft-source — `scripts/draft_sources.py` (`Word(text, start, end)`, `parse_json3`/`parse_scribe`), `scripts/probe_youtube_captions.py` | **done** → `youtube-data-pilot/caption-probe.md`. **Ba phát hiện đo được, không đoán:** (a) bản nhận dạng gốc là key riêng hậu tố `-orig` (`vi-orig`), `name` kết thúc " (Original)" — có key `vi` không chứng minh gì vì YouTube tự dịch ra ~100 thứ tiếng; (b) seg đầu của mỗi event **không** có `tOffsetMs`, ngầm hiểu là 0; (c) endpoint caption trả **revision khác nhau mỗi lần extract** → F3 phải parse file đã lưu, không fetch lại |
| **F2** | Fetch audio + captions — `scripts/fetch_youtube.py` | **done**, chạy thật trên cả 7 nguồn. Mono 16 kHz **ngay lúc tách** (`load_audio_16k` raise trên stereo, mà YouTube mặc định stereo). `provenance.json` đủ 11 field README §5, không field nào null, `label_source: "google_asr"`, `asr_draft_model: "vi-orig"`. Chạy lần hai **không** tải lại mà verify sha256 rồi skip. File raw luôn là bản **đầy đủ chưa cắt cho mọi meeting** (vật liệu tier 4b) — quyết định cố ý, "giữ đoạn giữa" là quyết định của bước cắt |
| **F3** | Draft → segment + manifest — `scripts/ingest_youtube.py` | **done**. Cắt dùng lại `_choose_splits`/`_check_speech_rate` của `ingest_real_bench.py` nguyên vẹn; phần mới duy nhất là `gap_candidates`, feed khoảng cách **start-to-start** giữa hai từ (`MIN_GAP_SEC = 0.3`) — json3 không có mốc kết thúc nên `next.start − cur.end` bằng 0 trong cùng event, không mang tin |
| **F4** | Review round-trip + guard — `scripts/review_youtube.py` (`--emit`/`--apply`/`--check`) + `youtube-data-pilot/style-guide.md` | **done (tooling); người soát chạy thật sau đó, xem mục H.** `--check` **raise** khi còn bất kỳ record nào `verified: false` — cơ giới hoá "held-out test set phải được verify đầy đủ" thay vì trông vào kỷ luật. **Luật casing quyết từ đo, không đoán:** `real-meetings-bench` (speech thật) **toàn chữ thường, không có sentence casing nào**, trong khi `paid-dataset-v2` (synthetic) giữ casing bình thường → `normalize_for_review` hạ chữ thường để khớp speech thật |

**Cập nhật F3, 2026-08-12 — 2 quyết định của người dùng, đã code và chạy thật:**

1. **Trim mỗi meeting về cửa sổ giữa 30 phút** trước khi cắt (`TRIM_TARGET_SEC = 1800.0`).
   `middle_window()` + `window_words()`. `raw/<meeting_id>/audio.wav` **không đổi** (vẫn đủ cho
   tier 4b). `yt_start`/`yt_end` mỗi record vẫn là toạ độ **tuyệt đối trong video gốc**.
2. **Test = 2 meeting khó nhì/khó ba** (`7B24A9GfHAo`, `3nuCdzuyqng`), xếp theo composite rank 3
   chỉ số của `caption-probe.md` (EN words/min, tỉ lệ non-VN-shaped, particle rate nghịch) trên
   cả 7 — ngay dưới val (`rCd8DSMk3-c`, khó nhất).

**Số đo sau khi chạy lại:** **790 segment** (giảm từ 1816 vì đã trim) — 105 (`dGT3YW0AdD8`, gốc đã
<30 phút) + 113–115 mỗi meeting còn lại. **207.4 phút ≈ 3.46 giờ.** Max duration 22.92 s, median
15.64 s, 99.6% nằm trong 10–25 s. Full suite **117 pass**.

**Bug tìm và sửa 2026-08-12:** `ingest_meeting` không xoá `audio/<meeting_id>/*.wav` cũ trước khi
ghi — mỗi lần chạy lại với ít segment hơn để lại file mồ côi. Đo thật: `audio/` nặng **889 MB**,
đĩa có 335–421 file/meeting trong khi manifest chỉ còn 105–115. Sau khi sửa: **382 MB**, số file
khớp đúng manifest. Full suite **131 pass**.

---

## G1–G4 — YouTube data report (2026-08-12), viết báo cáo, không thu thập thêm

Scope là **viết báo cáo**: không fetch thêm, không sửa manifest, không train, không đo CER/WER.
Người dùng quyết: báo cáo **trung tính, không có mục khuyến nghị**, tiếng Việt, Markdown, **6 biểu
đồ PNG**, **không** `stats.json`.

**Số đã đo, dùng trực tiếp — đo trên nhãn nháp, TRƯỚC lượt soát tay:**

- Quy mô: 7 meeting (`meeting_id == video_id` mọi row), 790 segment, **207.42 phút = 3.457 giờ**,
  **41.210 từ**. Duration min 3.024 s, max 22.920 s, mean 15.753 s, median 15.640 s, **0 segment
  vượt `MAX_SEGMENT_SEC`**. Words/segment 8–97, mean 52.16.
- Split: `test` 228 segment (`3nuCdzuyqng`, `7B24A9GfHAo`, 114 mỗi cái) / `demo` 562. Không meeting
  nào nằm hai bên.
- Code-switch (đo bằng `inspect_errors.is_vietnamese_shaped`): **2.748 / 41.043 token = 6.70%**,
  708 type; **709/790 = 89.7% segment** có ít nhất một. Lexical-particle **1.22%** (500),
  digit-bearing **0.53%** (217). *(`plot_youtube_stats.py` tính lại ra 6.67% vì mẫu số thật là
  41.210 chứ không phải 41.043 — 41.043 là số cũ còn sót trong ghi chú grill.)*
- Top type ≥3 meeting: 81 type, 1.365 lần (3.33% token). `ok(140,6) design(118,7) code(96,7)
  interview(76,6) system(60,6) team(44,5) level(40,7) … grap(24,4) … grab(9,4) th(14,6)`.
  `grap(24)` so với `grab(9)`, và `th(14)`, là lỗi nhãn `google_asr` thấy được ngay.
- Đối chiếu 3 corpus (non-VN-shaped / particle): YouTube **6.70% / 1.22%** · `paid-dataset-v2`
  (74.542 từ) **7.23% / 7.12%** · `real-meetings-bench` (7.168 từ) **8.75% / 2.87%**. YouTube thấp
  hơn cả hai trên cả hai chỉ số.
- Chủ đề: cả 7 tiêu đề đều là **phỏng vấn tuyển dụng kỹ thuật / nghề Big Tech**, cùng hệ sinh thái
  EngineerPro/MentorPro. Corpus chỉ có một chủ đề → không có phân phối chủ đề để vẽ.

**Môi trường G1/G2 — đã dò, không dò lại:** `d:\viet-speech\.venv\Scripts\python.exe`, Python
3.10.9, torch 2.13.0+cu126 CUDA khả dụng 1 GPU, `pyannote.audio` 4.0.7, `speechbrain`. `HF_TOKEN`
ở `d:\viet-speech\.env`. Trọng số gated đã cache. **Gotcha: `torchcodec` cài lỗi** nên đường đọc
file mặc định của pyannote fail — nạp wav bằng `soundfile`, truyền
`{'waveform': tensor(channel,time), 'sample_rate': int}`. `pyannote.audio` 4.0.7 đã bỏ
`OverlappedSpeechDetection` nên overlap dẫn từ `segmentation-3.0`. **Không sửa file nào trong
`d:\viet-speech`, không cài gì vào Fine_tune_wf.**

| # | Việc | Trạng thái |
|---|---|---|
| **G1** | Đo chồng tiếng — module 2 của viet-speech (`PyannoteSegmentationOverlapDetector`, `pyannote/segmentation-3.0`) trên **790 segment wav**, không chạy trên 7 `raw/*/audio.wav` (raw là video đầy đủ tới 110 phút, corpus chỉ là cửa sổ 30 phút giữa) | **done — gate trip là BÁO ĐỘNG GIẢ, đã truy ra nguyên nhân.** 790/790 dòng, 0 lỗi. Số đầu 0.89% (giây chồng tiếng / thời lượng clip) trip cổng "gần 0%". Nguyên nhân: số tham chiếu 16.4%/17.2% sinh bằng **công thức khác hẳn** — `OverlapLabel.overlapping` là **boolean**, nên tử số của nó là *thời lượng đơn vị VAD bị gắn cờ*, không phải số giây chồng tiếng. Chạy lại đúng công thức tham chiếu (silero-VAD từng clip, 3217 đơn vị / 10366 s): **corpus 10.57%**, per-meeting 3.73%–21.39% (`rIFrrmm8ILY` 21.39% và `dGT3YW0AdD8` 17.12% **vượt cả ground truth 17.2%**). Báo cáo đưa **cả hai** chỉ số kèm giải thích |
| **G2** | Đo trùng người nói `test` vs `demo` — ECAPA (`speechbrain/spkrec-ecapa-voxceleb`), cosine theo cách của `backend/core/speaker_id.py`, mẫu 40 segment ít chồng tiếng nhất mỗi meeting | **done, cổng kiểm ĐẠT** — within-meeting **0.5411** > cross-meeting **0.3160**, ma trận 7×7 đủ, 0 ô lỗi. Phát hiện: `xKDHjUoUN54` (demo) × `3nuCdzuyqng` (test) = **0.4899**, cặp cross cao nhất; 3/18 cụm chứa cả segment test lẫn `xKDHjUoUN54`. **Tự kiểm chứng phát hiện lỗi chọn mẫu:** 49–103 segment/meeting có overlap đúng 0 nên sort ổn định hoà theo thứ tự manifest → mẫu "40 ít chồng tiếng nhất" thật ra là ~40 segment ĐẦU theo thời gian. Chạy lại với mẫu ngẫu nhiên seed 42: within 0.5389 > cross 0.3101 (vẫn đạt), kết luận bền |
| **G3** | Script sinh biểu đồ — `scripts/plot_youtube_stats.py`, chỉ `numpy` + `matplotlib`, dùng lại `is_vietnamese_shaped` và `LEXICAL_PARTICLES` (**không** tự viết bộ nhận diện, **không** whitelist từ tiếng Anh — 72% dương tính giả). Kết quả G1/G2 nhúng thành hằng số đầu file kèm ngày đo | **done** — 6 PNG trong `docs/youtube-data-charts/`. Số in ra khớp mục "Số đã đo". Phát hiện thêm: phân phối duration cực hẹp, **739/790 (93.5%) nằm trong 15–17 s** — có thật, kiểm bằng percentile |
| **G4** | Viết `docs/youtube-data-report.md`, 11 mục, không mục khuyến nghị | **done** |

**Bốn tầng độc lập train/test — chỉ một tầng đạt.** Phát biểu đúng: test đo *"cùng chủ đề, cùng từ
vựng, giọng chưa rõ, session khác"*, **không** phải một tập test độc lập.

| Tầng | Trạng thái | Bằng chứng |
|---|---|---|
| `meeting_id` / file audio | **đạt** | test = `3nuCdzuyqng`, `7B24A9GfHAo`; không meeting nào nằm hai bên, kiểm trên cả 790 record |
| Người nói | **chưa kiểm ở thời điểm G4** | không có trường `speaker` trong 18 key — đây đúng là việc của G2, xem kết quả ở trên |
| Chủ đề | **trùng hoàn toàn** | cả 7 tiêu đề đều là phỏng vấn tuyển dụng kỹ thuật |
| Từ vựng | **trùng, đo được** | `design`, `code`, `level` có ở **cả 7 meeting**; 81 type ở ≥3 meeting |

> ⚠ **Mục 11 của `docs/youtube-data-report.md` viết "0/790 verified" — đúng ở thời điểm G4, SAI kể
> từ 2026-08-13/14.** Xem mục H dưới: cả 790 record đã soát tay. Bản manifest trong repo này vẫn là
> snapshot trước soát. **Bài học: đọc số từ artifact, không từ tài liệu mô tả artifact.**

**Hai cổng dừng của round này, không được đi vòng:** (1) nếu cách vòng `soundfile` cho `torchcodec`
không chạy, dừng và báo — **không tự đổi sang model khác**; (2) nếu cổng kiểm G2 thất bại, bỏ mục
10 + biểu đồ 6 và ghi thành câu hỏi mở, **không** hạ ngưỡng cho tới khi ra kết quả đẹp.

---

## Hồi quy Reworkwhisper-large-v5 trong production (2026-08-17 → 18)

Mục dài nhất và quan trọng nhất cho benchmark. Bản kiểm điểm quy trình ở
`docs/kiem-diem-v5-production-regression.md`.

**Ánh xạ tên:** `Reworkwhisper-large-v5` = run `v4-mixed-r16` · `Reworkwhisper-large-v4` = run
`v3-r16`. Tên repo HF lệch một bậc so với run id.

### Triệu chứng

Người dùng đổi model ASR production từ v4 sang v5, pipeline 8 module ở `d:\viet-speech` cho kết quả
**tệ hơn**. Hai run so được vì chỉ khác model ASR (module 1–3, 5–6 hành vi trùng, điểm định danh
giống hệt 0.7856 / 0.7341 / 0.8433). Trên `04_asr.json`: chữ hoa **15 → 1**, `team` 7 → 1 (thành
`tim` ×5), `build` 2 → 0 (thành `bill` ×2), `developer`/`standard`/`realize`/`nus` 1/1/1/2 →
**0/0/0/0**. Timestamp lành mạnh ở cả hai run → **không phải** truncation.

> **Đọc triệu chứng đúng những gì nó là.** Phép so sánh production là **đếm token và chữ hoa,
> không có reference nào**. Nó giả định ngầm rằng những từ tiếng Anh đó ở v4 là **đúng** — nhưng
> không ai phiên âm audio production. Phép đo này chứng minh hai model **khác nhau**, chưa chứng
> minh v5 **tệ hơn**.

### Nguyên nhân chắc chắn: quy trình, không phải weights

**Gate không bao giờ so run mới với model đang chạy production**; nó chỉ so mỗi run với baseline
của base model. Nên v3-r16 và v4-mixed-r16 đều `overall_pass: true` trên tiêu chí riêng, trong khi
trên **đúng 426 segment synthetic dùng chung, `ref` trùng từng byte**, CER là **0.0171 (v3-r16) so
với 0.0258 (v4-mixed-r16)** — tệ hơn **51% tương đối**, và **không cơ chế nào trong pipeline có
thể phát hiện**. → row **H4**.

**Bốn điểm mù khiến gate báo pass.** Ba điểm đầu mang tính định nghĩa, không phải ngưỡng đặt sai:
1. `normalization.lowercase: true` áp cho **cả hyp lẫn ref** → 14 ký tự hoa mất đóng góp **đúng 0**
   vào CER.
2. CER cân đều mọi ký tự. Toàn bộ khác biệt handoff ước ~30–36 ký tự edit distance trên 8.515 →
   **~0.4 pp**, nằm gọn trong bề rộng CI tier1-youtube 1.35 pp. *(Tính tay, đúng bậc độ lớn, đừng
   trích như số đo.)*
3. **Gate code-switch chưa từng được xây.** `style-guide.md` quy ước 4 tuyên bố chính tả tiếng Anh
   là "ground truth cho gate code-switch", nhưng `src/gate.py` grep `code_switch|english` ra **0
   kết quả**.
4. **Không có phép so nào giữa run mới và model đang chạy production.** Ba điểm trên là mù về *chỉ
   số*; điểm này mù về *đối tượng so sánh*, và đúng bất kể nguyên nhân hành vi là gì.

### Đã loại trừ, không điều tra lại

- **Preprocessing borrowed từ PhoWhisper-large** — bác ở phiên trước. `processor_config.json` khớp
  từng giá trị; `generation_config.json` khớp byte-for-byte trừ `transformers_version`;
  `tokenizer.json` là whisper-large-v2 gốc, **0 added token**.
- **Nhãn YouTube chưa soát — SAI, đã bác.** Cả 790 record ghi `verified: true`, `reviewed_by: Quang`,
  `review_date: 2026-08-13/14`. `build_mixed_dataset.py` raise nếu bất kỳ record nào
  `verified != true`, nên `mixed-noisy-v1` build được đã chứng minh nhãn đã soát. Sai sót gốc: đọc
  `docs/youtube-data-report.md` mục 11 (viết **trước** khi soát xong) thay vì đo manifest.
- **λ / rank** — v5 dùng λ=0.25, **thấp hơn** v4 (λ=0.5), tức ít adapter hơn mà vẫn tệ hơn trên
  synthetic. *(Tỉ lệ trộn thì KHÔNG loại trừ — xem D3.)*
- **Chất lượng nội dung nhãn YouTube — đã đo, nhãn tốt.** So token non-VN-shaped giữa
  `raw/*/captions.json3` và nhãn đã soát, khớp theo `yt_start`/`yt_end` trên cả 790 segment: người
  soát **bỏ 491** token lỗi của Google (`grap` 21, `prom` 10, `btech` 5) và **thêm 788** token
  tiếng Anh đúng chính tả (`system` 38, `grab` 34, `leetcode` 34, `design` 21, `backend` 14,
  `engineer` 13). Việc sửa data còn lại **chỉ là casing**, hẹp hơn dự đoán: 788 token người soát tự
  gõ **không có nguồn casing**, và phần lớn vốn viết thường trong tiếng Anh đúng — chỉ tên riêng
  cần hoa (`Grab`, `LeetCode`, `Shopee`, `Axon`), ước 20–40 mục.

**Casing — truy được đến dòng code, nhưng độ lớn nhỏ, đừng đề cao.** Nhãn YouTube có **0 chữ hoa
trên 178.075 ký tự** (đo trực tiếp 7 manifest) so với 6.610/333.272 của paid-dataset-v2, và YouTube
chiếm ~60% ký tự (49.260 vs 32.329). Nguồn: `normalize_for_review` tại `scripts/review_youtube.py:78`
là `text.lower()`. Đây là **quy ước cố ý** (caption Google viết hoa ngẫu nhiên giữa câu). **Giới
hạn:** v4 cũng chỉ xuất **15 ký tự hoa trên 8.515** — casing khó là toàn bộ nguyên nhân.

### Nghi vấn N1–N4

| # | Nghi vấn | Trạng thái |
|---|---|---|
| **N1** | Chưa chứng minh v5 được publish bằng `scripts/merge_and_push.py` | **ĐÓNG** qua H1 |
| **N2** | Chưa giải thích được cơ chế `team`→`tim`. Nhãn chứa `team` **32 lần**, `tim` **0 lần** | **vẫn mở** — xem "Bốn kiểu lỗi" dưới |
| **N3** | Chưa tự đo số nào của v4/v5 trên audio production | **ĐÓNG** qua H2(a). `04_asr.json` ghi `model_id` lồng trong từng speaker, khớp `07_transcript.json:asr_model_id`. Hai id trỏ cùng URL ngrok nhưng output khác thật (8515→8608 ký tự) → weights trên server **đã đổi**, phép so hợp lệ |
| **N4** | Retention cho phán quyết **trái chiều theo domain** — v5 tệ hơn v4 trên synthetic (0.7599 vs 0.8643) nhưng **tốt hơn** trên real-meetings-bench (0.4545 vs 0.4083) | **giải quyết bởi phát hiện λ dưới** |

### Row H0–H6, D1–D3

| # | Việc | Trạng thái |
|---|---|---|
| **H0** | Rollback production | **Tiền đề sai, đã sửa sau khi đọc file:** `config/models.yaml` **không** chọn v5 ở đâu cả; `default: true` đã nằm trên `reworkwhisper-large-v4`. Nặng hơn: `-v4-remote` và `-v5-remote` trỏ **cùng một URL ngrok**, nên model thật chạy do `ASR_MODEL_SIZE` phía Kaggle quyết định, **không** do id trong config. Rollback thật = khởi động lại server Kaggle |
| **H1** | Đóng N1 — xác minh repo HF v5 do `merge_and_push.py` tạo | **done, N1 ĐÓNG.** README của v5 ghi đúng ``Pipeline commit `9fea278`, run `v4-mixed-r16` `` + bảng gate 3 tier khớp `gate_results.json` (0.0564 / 0.0226 / 0.2890). **Merge từ NF4 và double-merge bị loại trừ dứt điểm** |
| **H6** | **PHÉP ĐO DỨT ĐIỂM** — chạy v3-r16 (=HF v4) trên đúng 228 segment `source: youtube` của tier1 test. Bộ test duy nhất vừa là speech thật, vừa có nhãn người soát, vừa cùng domain với audio production | **done, KẾT QUẢ PHÂN ĐÔI.** CER v3=**0.1205** vs v5@0.25=**0.0764** (v5 tốt hơn); retention v3=**0.7746** vs v5=**0.7094** (v5 tệ hơn, đúng chiều triệu chứng). Người dùng quyết: **retention là tín hiệu chính** vì nó khớp triệu chứng production, còn CER không bắt được kiểu lỗi này |
| **H2** | (a) tự đếm lại 4 chỉ số handoff; (b) A/B `prompt_ids` với glossary có casing | **(a) done, kèm phát hiện quan trọng hơn acceptance ban đầu.** Đếm lại trên đúng 2 file `04_asr.json`: **8515 vs 8608** ký tự, chữ hoa **15→1**, `team` **7→1**, `build` **2→0**. `01_vad.json`/`02_overlap.json` byte-identical → cùng audio đầu vào. **Phát hiện mới:** `first10.wav` (video `E5dAymt68-0`, **không** nằm trong 7 meeting train) có transcript người soát tay thật, cắt đúng 596.0 s. Qua đúng `Normalizer`/`score`/`english_token_retention` của repo này: **CER v4=0.2461 vs v5=0.2818**, retention **v4=0.7179 vs v5=0.5128** — CER **và** retention cùng chiều "v5 tệ hơn" trên audio thật ngoài tập train của cả hai. **(b) chưa chạy** |
| **H3** | Xây thang đo còn thiếu — (1) tỉ lệ giữ token tiếng Anh; (2) CER phân biệt hoa thường | **done cho (1).** `english_token_retention(refs, hyps)` trong `src/metrics.py`: đếm token trong ref có `len>1`, `isalpha()`, **không** khớp `is_vietnamese_shaped`; khớp theo multiset trên cả segment. Cùng bộ lọc với `plot_youtube_stats.py`. Cổng ĐẠT (phân biệt được v4/v5) nhưng kết quả **bác một phần** chẩn đoán → sinh ra N4 |
| **H4** | **Bịt điểm mù số 4** — (a) `merge_and_push.py` nhận tham chiếu tới run đang chạy production và raise nếu ứng viên hồi quy; (b) tier mới dùng `english_token_retention` | **(a) done** — `check_no_regression_vs_production`. **(b)** phụ thuộc H6 |
| **D1** | Phục hồi casing tên riêng trong nhãn YouTube (~20–40 mục) | todo. **Lợi ích nhỏ, đừng đề cao:** v4 cũng chỉ xuất 15 hoa/8.515. Danh sách tên riêng phải do người xác nhận, **không** tự suy từ tần suất |
| **D2** | Tăng lượng dữ liệu thật — YouTube hiện chỉ **3.46 giờ / 790 segment** mà chiếm 60% ký tự train | todo. `meeting_id` và thư mục audio phải disjoint với 7 meeting hiện có. **Không** đưa nhãn chưa soát vào train |
| **D3** | Đổi tỉ lệ trộn, train lại (~6.5 h T4) | todo, **quyết định không chạy lúc này** — xem dưới |

### Bốn bộ đo audio thật, bốn kết quả

| Bộ đo | Thuộc train v5? | CER | retention |
|---|---|---|---|
| H6 (`youtube-meetings` test, 228 seg) | có | v5 tốt hơn | v5 tệ hơn |
| tier4a (`real-meetings-bench`, ML/VAE, 264 chunk) | không | v5 tốt hơn | v5 tốt hơn |
| `first10.wav` (webinar career) | không | v5 tệ hơn | v5 tệ hơn |
| `NZiW4QH83CI` (RPA/lập trình, 20 seg) | không | hoà | v5 tệ hơn 12.8 pp |

**retention tệ hơn ở 3/4 bộ, chỉ tier4a là ngoại lệ.** CER lộn xộn (2 tốt hơn, 1 tệ hơn, 1 hoà) —
đúng như H3 cảnh báo: CER không đủ nhạy để bắt lỗi mất từ mượn.

**Giả thuyết "đồng âm ngắn thông dụng vs thuật ngữ dài hiếm" — ĐÃ KIỂM, BỊ BÁC.** Chạy
`english_token_retention` trên đúng cặp `ref`/`hyp` của cả 4 bộ (join `(meeting_id, segment_id)`,
`ref` khớp byte-for-byte cả 4 — 228/264/20/1 khoá, 0 mismatch), lấy phần chênh `missing[token]`:

- **tier4a** (bộ duy nhất v5 tốt hơn): `task research attention size transformer tokenize mask vae
  llm sublayer embedding` — thuật ngữ dài/hiếm, không đồng âm. Giả thuyết đúng.
- **`first10.wav`** (v5 tệ nhất): `team google build client webinar gmail nus` — đồng âm ngắn thông
  dụng. Giả thuyết đúng.
- **H6**: `team` (+14) là token đơn lẻ lớn nhất, nhưng **76% phần chênh còn lại** (`scope version
  meeting location staff outsource design service master interviewer database qps mock latency
  backend`) không phải đồng âm ngắn. Giả thuyết đúng 1 phần.
- **`NZiW4QH83CI`** — **phản ví dụ trực tiếp**: 4 token hồi quy toàn bộ là `rpa rule chatbot ocr`,
  đúng loại "hiếm, không đồng âm" mà giả thuyết dự đoán sẽ **không** hồi quy, nhưng bộ này lại hồi
  quy mạnh nhì.

**Bốn kiểu lỗi trộn lẫn, không phải một cơ chế** (đọc trực tiếp transcript): (1) thay bằng từ tiếng
Việt khác nghĩa — `team`→`phim`/`tim`; (2) nghe nhầm **tên riêng** thành rác — `Engineer Pro`→`nga`;
(3) hallucinate từ lạ — `client`→`lion`/`ờjohn`; (4) lỗi chính tả nhẹ bị metric tính nhầm thành
"mất" — `webinar`→`webina`, `rule`→`ru`. Thêm một dạng ở H6 `seg_0051`: đánh vần lại từ mượn thành
âm tiết tiếng Việt — `service`→`sờ vít`, `scope`→`sờ cốp`, `aws`→`android s`. Và `rpa` bị cả hai
model nghe nhầm thành đánh vần chữ cái, nhưng v5 **rời rạc và biến thiên hơn** (`a pa` / `ây ai` /
`a ti a` so với `apa` ổn định của v3).

**Kết luận:** trục "đồng âm vs không đồng âm" giải thích đúng 2/4 bộ, bị `NZiW4QH83CI` bác thẳng.
Trục thật có thể gần cơ chế phát âm/đánh vần hơn — cần nghe lại audio, không chỉ đọc text. **Không
đủ cơ sở để chọn D2 qua D3 chỉ từ kết quả này.** Không có "nút sửa" đơn nhất, nên **không chạy D3
lúc này** — 6.5 h GPU đặt cược vào một nguyên nhân chưa xác định.

### PHÁT HIỆN NẶNG NHẤT: v5 publish ở λ=0.25 do LỖI DẤU trong `select_lambda`

Sửa được không cần train lại. Replay `src/pipeline.py:select_lambda` trên
`Outputs/v4-mixed-r16/metrics/lambda_sweep.csv`:

```
lam=0.0   val_cer=0.10337 ood_cer=0.02283  in_budget=True
lam=0.25  val_cer=0.05483 ood_cer=0.02261  in_budget=True  ratio=-0.005   <-- ÂM
lam=0.5   val_cer=0.03636 ood_cer=0.02619  in_budget=True  ratio=0.193  jump=-42x  <-- bị loại oan
lam=0.75  val_cer=0.03316 ood_cer=0.03343  in_budget=True  ratio=2.260  jump=11.68x
lam=1.0   val_cer=0.03257 ood_cer=0.04257  in_budget=True  ratio=15.360
```

Bước 0.0→0.25 làm OOD CER **giảm** (0.022833→0.022612) nên `delta_ood` âm, `prev_ratio = −0.005`.
Điều kiện dừng `ratio > prev_ratio * elbow_ratio_threshold` thành `0.193 > −0.045` — **mọi ratio
dương đều vượt một cái ngưỡng âm**, nên vòng lặp `break` ngay tại λ=0.5. Trên sweep của v3-r16 lỗi
này không kích hoạt vì ratio ở λ=0.25 dương (+0.017) — đó là lý do nó sống sót qua E2 và toàn bộ
test cũ.

**Đã sửa:** chỉ bước có ratio **dương** mới được làm mốc so cho bước sau (bước "mua val CER mà
không tốn OOD" không mang thông tin cost/benefit nên không định nghĩa được elbow). Test mới
`test_select_lambda_a_free_step_does_not_reject_every_later_lambda` chạy đúng sweep row thật của
v4-mixed-r16, đỏ trước khi sửa (`assert 0.25 == 0.5`), xanh sau. Full suite **170 pass**.

**Bằng chứng phụ, 0 GPU — retention TĂNG theo độ mạnh adapter, base model là tệ nhất.** Trên
`predictions_baseline_*.csv`: base PhoWhisper (λ=0) **0.5574** synthetic / **0.3638** youtube /
**0.3158** real, so với v5@λ=0.25 đạt 0.7599 / 0.7094 / 0.4545. Fine-tune **cải thiện** retention
chứ không phá — nên hạ λ để "an toàn" là đi sai chiều.

### Row Lλ — chấm lại cùng bộ weights ở nhiều λ (inference thuần, không train)

`scripts/eval_v4_mixed_at_lambda.py` + `notebooks/eval-v4-mixed-at-lambda05.ipynb`, chấm
`Outputs/v4-mixed-r16/checkpoints/best` (adapter thô — `adapter/` đã bake 0.25, không scale ngược
lên được). Dùng lại chính `_score_by_source`/`score_real` của gate nên `pass`/`retention_pass` mang
đúng nghĩa gate thật.

**Cổng đọc kết quả đặt ra TRƯỚC khi chạy:** CER giữ/tốt lên **và** retention tăng ở mọi slice →
λ=0.5 là điểm vận hành tốt hơn. Retention không tăng → λ không phải đòn bẩy, nói thẳng và dừng.
**Không** đọc bằng CER một mình — đó đúng là lỗi đã cho v5 ra production.

**ĐƯỜNG CONG ĐẦY ĐỦ (2026-08-17). CER chạm đáy ở λ=0.75 rồi QUAY ĐẦU:**

| λ | youtube CER | synthetic CER | real CER | youtube ret | synthetic ret | real ret | OOD CER |
|---|---|---|---|---|---|---|---|
| 0.25 (đã publish) | 0.0764 | 0.0258 | 0.2890 | 0.7094 | 0.7599 | 0.4545 | 0.0226 |
| 0.5 (luật chọn) | 0.0624 | 0.0185 | 0.2453 | 0.8227 | 0.8309 | 0.5359 | 0.0262 |
| **0.75 (chọn ship)** | **0.0593** | **0.0161** | **0.2440** | **0.8547** | 0.8622 | 0.5678 | 0.0334 |
| 1.0 | 0.0610 | 0.0173 | 0.2504 | 0.8490 | **0.8706** | **0.5710** | 0.0426 |

*(OOD CER từ `lambda_sweep.csv`, bound 0.0428. `pass` và `retention_pass` đều `true` ở mọi λ đo.)*

- **CER chạm đáy ở λ=0.75 trên cả ba slice rồi tệ đi ở λ=1.0.** Ba slice cùng quay đầu một lúc —
  overfit thật ở cường độ adapter tối đa, không phải nhiễu một bộ.
- **Lo ngại "λ cao kéo về nhãn 0 chữ hoa nên retention giảm" — SAI, đã bác bằng đo.** Retention
  tăng đều tới 0.75 rồi gần phẳng. Ghi lại vì tôi đã nêu nó như rủi ro trước khi đo.
- **λ=1.0 bị loại:** CER tệ hơn 0.75 ở cả ba slice, hơn retention không đáng kể, và OOD 0.0426 dư
  bound đúng 0.00027.
- **Triệu chứng production gốc đã sửa, đo trực tiếp trên token** (228 segment youtube, λ=0.25→0.5):
  `team` mất **14 → 1**; `meeting` 6→0, `version` 6→0, `staff` 4→0, `service` 3→0, `scope` 6→1.
  Hồi quy ngược lại chỉ 3 token lẻ: `sql` −2, `off` −1, `six` −1.

**QUYẾT ĐỊNH (người dùng, 2026-08-17): ship λ=0.75.** Tốt hơn λ=0.5 ở **cả 6 ô**, giá là OOD CER
0.0262→0.0334 (+0.72 pp tuyệt đối, +27% tương đối), vẫn trong bound với biên 0.0094. OOD là VIVOS —
speech đọc, tồn tại như chốt chặn catastrophic forgetting chứ không đại diện production; mọi slice
**giống production** đều tốt hơn ở 0.75.

**Đây là người override `select_lambda`, phải ghi rõ chứ không được trình bày như kết quả của
luật.** Luật sau khi sửa vẫn chọn 0.5 (bước 0.75 có jump 11.68× > ngưỡng 10). Lý do luật hụt: nó
chỉ tối ưu val CER so với OOD CER, mà **val CER không nhìn thấy retention** — đúng cái tăng 3.2 pp.
**Chưa sửa `elbow_ratio_threshold`** — đổi ngưỡng để luật tự ra 0.75 là fit ngưỡng theo một kết quả
đã biết, đúng thứ cổng dừng cấm. Tiền lệ ngược chiều: gate của v3-r16 chọn λ=1.0, người override
xuống 0.5 sau khi grill `docs/finetune-results-report-v3.md`.

**Đính chính nhãn λ.** `Outputs/v3-r16/audit/predictions_*.csv` được chấm ở **λ=1.0**, không phải
λ=0.5. Bằng chứng: `adapter_config.json` có `lora_alpha=32.0` (rsLoRA, r=16 → scaling 8.0 = λ=1.0),
và `gate_results.json` có `tier2_ood.cer = 0.041428164827171814`, trùng **từng chữ số** với dòng
λ=1.0 của `lambda_sweep.csv`. Các số 0.0171 / 0.8643 / 0.4801 / 0.4083 vẫn đúng, chỉ nhãn λ sai.

### GATE λ=0.75 ĐÃ CHẠY, QUA HẾT (2026-08-18, `Outputs/lambda075-metrics/`)

`overall_pass: true`, `_lambda: 0.75`, `_lambda_source` ghi rõ đây là người override.

| tier | CER | bound | pass |
|---|---|---|---|
| tier1_in_domain | 0.0422 | 0.1018 | true |
| — youtube | 0.0593 (ret 0.8547, base 0.3638) | — | pass + retention_pass |
| — synthetic | 0.0161 (ret 0.8622, base 0.5574) | — | pass + retention_pass |
| tier2_ood | 0.033427152841466114 | 0.0428 | true |
| tier4a_real | 0.2440 | 0.3157 | true, verdict **IMPROVED** |

- **tier2 OOD khớp row sweep λ=0.75 với delta đúng 0.0** — giả định "greedy decode cùng split OOD
  tái hiện đúng số sweep" được xác nhận chứ không phải chỉ hy vọng.
- **Production check qua sạch trên đủ 654 segment chung:** candidate **0.0422** so với production
  **0.0805**; retention **0.8574** so với 0.7945 (+6.3 pp).
- **Mốc kiểm H6 tái hiện chính xác:** v3-r16@λ=0.5 trên 228 youtube ra CER **0.1205**, đúng số H6.
- **Nghi ngờ về tham chiếu λ=1.0 được xác nhận đúng.** v3-r16@λ=0.5 trên 426 synthetic đạt
  **0.0196**, không phải 0.0171 của bản λ=1.0. Tham chiếu cũ khắt khe hơn model thật đang chạy, và
  v5@0.75 (0.0161) **thắng cả slice synthetic**. Nếu vẫn dùng file λ=1.0 thì check đã đánh trượt
  một ứng viên đáng lẽ phải đậu.

Predictions của v3-r16@λ=0.5 (654 dòng) giữ lại làm tham chiếu production cho mọi ứng viên sau,
đừng tính lại mỗi lần.

**Đường ship:** `scripts/gate_at_lambda.py` — chạy gate đầy đủ ở λ chỉ định rồi bake adapter, tạo
run dir `merge_and_push.py` nhận nguyên trạng. Cố ý **không** làm cờ trên `stage_sweep_gate`, để
một run bình thường không bao giờ lặng lẽ đi vòng `select_lambda`. Script raise nếu λ không phải
row trong `lambda_sweep.csv`, và gate **đo lại** tier2 OOD ở λ đó thay vì chép số sweep.

**Bốn cổng dừng của round này:** (1) ~~H1 không tìm được dòng provenance~~ — đã đóng; (2) H2(b) ra
output lặp/hallucinate → đó là phép thử fail, **không** được đọc thành "prompt không cứu được";
(3) H3 chỉ số không phân biệt được v4/v5 → thiết kế lại chỉ số, **không** hạ ngưỡng cho tới khi ra
kết quả đẹp; (4) ~~H6 cho thấy v5 tốt hơn → dừng D3~~ — không áp được nguyên văn vì H6 ra kết quả
phân đôi; người dùng quyết retention là tín hiệu chính.

**Môi trường — đã dò, không dò lại.** `HF_TOKEN` **không** có trong shell và `huggingface_hub`
**không** cài trong `/c/Program Files/Python310/python`. Dùng `d:\viet-speech\.venv\Scripts\python.exe`
(`huggingface_hub` 1.23.0) + `HF_TOKEN` trong `d:\viet-speech\.env`. Repo HF v4/v5 là
**private/gated** — `curl` tới `/raw/main/` trả HTTP 401. `dataset/mixed-noisy-v1` **không** có
trên máy này (build trên Kaggle). Console cp1252 → đặt `PYTHONIOENCODING=utf-8` cho mọi script in
tiếng Việt.

---

## 2026-08-19 — Đo cổng kiểm bằng đúng module 4 của production

**Trục thứ ba của sự cố v5: đường decode.** Các round trước sửa "so sai đối tượng" và "chỉ số không
bắt được lỗi", nhưng cả hai vẫn đo bằng `src/asr.py:transcribe_batch` — greedy, cắt cứng 30 s, fp16,
chấm từng segment rời — trong khi module 4 của `d:\viet-speech` decode với thang temperature
`(0.0 … 1.0)`, `condition_on_prev_tokens=True`, long-form `truncation=False`, fp32. Hai thuật toán
khác nhau, nên CER cổng **không chặn được** CER production, và `check_no_regression_vs_production`
đang xếp hạng hai model trên một đường decode không model nào chạy khi phục vụ người dùng. Chi
tiết: `docs/kiem-diem-v5-production-regression.md` §2.7.

**Không viết lại module 4 — chép nguyên văn.** Kế hoạch ban đầu là gọi HTTP tới
`d:\viet-speech\scripts\remote_asr_server.py`, nhưng repo đó private nên máy GPU không pull được.
`vendor/viet_speech/` giữ bản sao **byte-identical** của 9 file, chạy được bằng
`python vendor/viet_speech/remote_asr_server.py`. `backend/adapters/asr/phowhisper.py` và
`remote_asr_server.py` **trùng sha256 với đúng entry trong `viet-speech-remote.zip`** — bản chép là
chính xác code máy GPU đang chạy. Hai file đó có thay đổi chưa commit so với `9795707`; phần chênh
chỉ thêm `transcribe_batch` và tách hàm, **không đổi tham số decode nào**. Ghi đầy đủ trong
`vendor/viet_speech/VENDORED.md`.

**Fine_tune_wf vẫn không cần torch.** Phía local chỉ ghép audio, gọi HTTP, chấm điểm. Chấm bằng
`Normalizer` của repo này chứ không phải `normalize_vi` của viet-speech (bản kia không có quy ước
số — dùng nó sẽ thêm một trục lệch nữa vào phép đo sinh ra để cô lập một trục).

| File | Việc |
|---|---|
| `vendor/viet_speech/` | 9 file chép nguyên văn + `VENDORED.md` (nguồn, commit, sha256 từng file, profile decode) |
| `scripts/eval_module4.py` | 2 split, một clip một POST, `gen_params: null`; ghi `predictions_*.csv` đúng 4 cột của `write_predictions`, `hyp_raw_*.csv`, `module4_results.json`, `responses_*.jsonl` (resume được) |
| `tests/test_module4_profile.py` | 13 test: hash 9 file so với `d:\viet-speech`, assert từng default decode, assert `_gen_kwargs` không override, assert bảng hash trong `VENDORED.md` còn đúng |
| `scripts/merge_and_push.py` | `--module4-dir` + `--module4-production-dir` **bắt buộc**; `check_module4_evidence` chạy CER + retention trên **cả hai split** |

**Xác minh đã chạy ở máy này (không cần GPU):**

- `build_real_clips`: **196 parent segment, 43.2 phút**, clip dài nhất **212.0 s**. Chunk mỗi
  parent: 1–10.
- **Ref trùng từng byte với `rejoin_real_chunks` của cổng trên cả 196 parent** (0 lệch). Đây là
  điều kiện để bước verify có nghĩa: cùng tập, cùng tham chiếu, chỉ khác đường decode.
- `build_test_clips`: **654 segment (426 synthetic + 228 youtube), 94.95 phút**. Lưu ý đã bắt được:
  audio `paid-dataset-v2` là **24 kHz** nên phải qua `load_audio_16k` — gửi thẳng 24 kHz thì server
  đọc sample rate từ header WAV và feature extractor sẽ tạo mel sai.
- Chạy đủ đường bằng server giả: POST → resume jsonl → CSV → JSON đều đúng; `--expect-model-id` sai
  thì raise; `--probe-determinism` chạy.
- `pytest tests/` — **189 pass**.

**Chưa chạy: bộ bằng chứng thật.** Cần máy GPU, ~4.6 h (138 phút audio × 2 model, fp32 large
long-form, RTF ước ~1). Thứ tự: real-bench trước cho cả hai model (~1.5 h), vì triệu chứng
production nằm ở đó và `--probe-determinism` phải trả lời trước khi tiêu 3 h còn lại. **Cho tới lúc
đó, độ lệch giữa cổng và production vẫn là con số chưa ai biết**; số cổng để đối chiếu là tier4a
**0.2440** và tier1 **0.0422** (synthetic 0.0161 / youtube 0.0593) của v4-mixed-r16@λ=0.75.

**Phần dư có ý thức:** production chạy đủ 8 module nên module 4 nhận splice đã tách nguồn + trim im
lặng, còn bằng chứng này dùng audio real-bench thô. Đây là bằng chứng về *đường decode*, không phải
về toàn pipeline.

---

## Ghi chú còn hiệu lực

- Tier 4's reference là bản post-edit output PhoWhisper-small — nó **ưu ái** checkpoint PhoWhisper.
  Đọc `PROJECT_CORE.md` §4 trước khi báo cáo bất kỳ số tier-4 nào.
- `unpaid-dataset` / `dataset_by_task` vẫn chưa profile, vẫn ngoài scope.
- Bản manifest `dataset/youtube-meetings/` trong repo này là snapshot **trước** lượt soát tay (cả
  829 record `verified: false`). Nhãn đã soát chỉ tồn tại trong corpus dựng `mixed-noisy-v1`.
