# Sổ thí nghiệm

Một dòng một lượt gate/sweep (không phải một lượt train — một run train có thể sinh
nhiều dòng nếu benchmark ở nhiều λ). Số **đọc từ artifact liệt kê ở cột cuối**, không
chép từ doc tổng hợp — `docs/so-lieu-tong-hop.md` đã ghi nhận là lệch khỏi manifest.
Điền ngược 15/09/2026 (phiên grill) cho các lượt đã chạy trước khi sổ này tồn tại.

Cột `corpus` là seg **train** (không tính val/test); giờ và mật độ token ngoại lai đọc
từ manifest gốc của run đó, không phải từ bảng tổng hợp.

## Các lượt

### v3-r16 (base run, paid-dataset-v2, 100% synthetic)

| λ báo cáo | production? | tier2 OOD CER | tier1 in-domain | cross-domain | artifact |
|---|---|---:|---:|---|---|
| 1,0 | không | 4,14% | — | chưa đo (bench chưa tồn tại) | `Outputs/v3-r16/metrics/gate_results.json` |
| 0,5 | **có — Reworkwhisper-large-v4** | không có trong summary | 8,05% (youtube 12,05%/retention 77,5%; synthetic 1,96%/retention 83,1%) | chưa đo | `experiments/v3-r16-lambda0.5/summary.json` |

- ngày: 2026-08-04 (ngày rescore rejoin-fix; ngày train gốc không xác định lại được từ
  artifact hiện có)
- corpus: 4.270 seg train, giờ chưa đo (không tìm thấy artifact đo hours cho
  paid-dataset-v2)
- rank/α: 16/32 · epoch 3 · LR 2e-4 · init_adapter: không
- λ=1,0 là kết quả của rule cũ "λ lớn nhất trong ngân sách" (`select_lambda` đã thay,
  xem docstring `src/pipeline.py:44`) — không phải số production.
- **Kết luận:** v4 production (λ=0,5) đo tier1 8,05% CER; tier2 OOD và cross-domain
  chưa từng được đo cho lượt này (cả hai bench đều ra đời sau).

### v4-mixed-r16 (mixed-noisy-v1, 4.717 seg / 6,98 h / 7,13% mật độ)

Nguồn corpus: `docs/v6-analysis-brief.md` §2 (bảng đo lại từ artifact của chính run).
rank/α: 16/32 · epoch 3 · LR 2e-4 · init_adapter: không.

| λ báo cáo | production? | ngày | tier2 OOD CER | cross-domain CER | retention | no_loanword | artifact |
|---|---|---|---:|---:|---:|---:|---|
| 0,25 | không | — | 2,26% | chưa đo | chưa đo | chưa đo | `experiments/v4-mixed-r16/metrics/gate_results.json` |
| 0,75 | **có — Reworkwhisper-large-v5, production từ 2026-08-18** | 2026-08-18 | 3,34% | 7,40%\* | 66,48%\* | 6,13%\* | `experiments/v4-mixed-r16-lambda0.75/metrics/gate_results.json` |

\* Số ngưỡng gate hiện tại (`configs/experiment.yaml:120-122`), đo lại 2026-09-14 dưới
đường decode chống lặp (`temperature fallback` + `compression_ratio_threshold`,
`eval.batch_size=64`, 299 segment) tại `outputs/cross-domain-v5-newdecode.json` — file
đó không có trên máy này, chỉ còn comment trong config ghi lại số. Đường decode **cũ**
(trước fix) cho 8,19% CER / 66,6% retention (`Outputs/bench-cross-v6/scores.json`,
key `winhsss_reworkwhisper_large_v5|cross-domain`) — chênh 0,79pp do một segment lặp,
đừng trộn hai đường decode khi so sánh.

- λ=0,25 là kết quả của elbow rule khi `sweep.retention_floor` còn null: rule tối ưu
  val CER so OOD CER, mù trước `english_token_retention` — dừng ở bước đầu tiên có
  lợi, bỏ qua chỗ retention còn tiếp tục lên ở λ cao hơn ([[lambda-raises-retention-not-lowers]]).
  λ=0,75 là override tay, ghi lại trong SESSIONS.md ([[v5-lambda-is-human-override]]).
- **Kết luận:** production v5 (λ=0,75) là ngưỡng cross-domain dùng làm mốc so mọi lượt
  v6 sau này. λ=0,25 (tự động) không phải số production — sổ này không dùng nó làm mốc.

### v6-corpus-r32 (v6-corpus nguyên, 34.972 seg / 101,23 h / 2,48% mật độ)

- ngày: 2026-09-12 · rank/α: 32/64 · epoch: 1 · LR: 2e-4 · init_adapter: không
- λ báo cáo: 0,5 (khớp `ood_cer` trong `lambda_sweep.csv` tại hàng λ=0,5)
- tier2 OOD CER: 2,82%
- cross-domain: CER 10,81% / retention 48,64% / no_loanword 5,77% — số đọc trong phiên
  15/09 từ `Outputs/bench-cross-v6/scores.json` (`rework_whisper_v6|cross-domain`,
  biến thể chuẩn hoá N3); file đó tự báo CER 10,85%/retention 48,56% (chênh <0,1pp,
  không tìm được nguồn `no_loanword` riêng trong chính file — không có field đó).
- artifact chính: `Outputs/v6-corpus-r32.zip` → `v6-corpus-r32/metrics/gate_results.json`
  (cross_domain_bench không nằm trong file này — cross-domain-bench cho v6-corpus-r32
  chỉ đo qua script ngoài, `Outputs/bench-cross-v6/scores.json`)
- **Kết luận:** thắng trên miền được train (tier2 OOD 2,82% tốt hơn v5's 3,34%), thua
  nặng ngoài miền (cross-domain retention 48,6% so với v5's 66,5%) — nguyên nhân hàng
  đầu là mật độ token ngoại lai trong corpus loãng gấp gần 3 lần v5 (2,48% so 7,13%),
  không phải thiếu giờ ([[v6-corpus-loanword-dilution]]).

### v6-dense-1 / -3 / -4 (curriculum hai pha, pha 1 = v6-corpus-r32, pha 2 = tập lọc mật độ)

rank/α: 32/64 (kế thừa từ adapter pha 1, `cfg.lora` bị bỏ qua) · epoch pha 2: 2 ·
LR pha 2: 1e-4 · init_adapter: `outputs/v6-corpus-r32/checkpoints/best` · λ ghim cứng
0,5 (để bốn nấc so được trực tiếp) · ngày: 2026-09-13 · nguồn corpus (giờ, mật độ):
`docs/v6-density-ladder-plan.md` §3.

| run_id | filter N | seg train | giờ | mật độ | tier2 OOD CER | cross-domain CER | retention | no_loanword | artifact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v6-dense-1` | ≥1 | 12.880 | 39,76 | 6,515% | 2,87% | 10,59% | 52,69% | 5,72% | `Outputs/ladder-phaseA/v6-dense-1.zip` |
| `v6-dense-3` | ≥3 | 4.188 | 15,81 | 10,442% | 2,73% | 9,14% | 55,96% | 5,55% | `Outputs/ladder-phaseA/v6-dense-3.zip` |
| `v6-dense-4` | ≥4 | 2.028 | 8,13 | 14,050% | 2,83% | 9,87% | 53,55% | 5,53% | `Outputs/ladder-phaseA/v6-dense-4.zip` |

(cross-domain CER/retention/no_loanword đọc từ mỗi run's
`metrics/gate_results.json.cross_domain_bench`, khớp với bảng đã đối chiếu phiên 15/09.)

- **Kết luận (chốt hướng đi, không re-litigate):** curriculum hai pha thua một pha.
  Cả ba lượt đều thua `v6-ondomain-15h` (một pha LoRA tươi, không kế thừa pha 1) ở cả
  CER lẫn retention. Lượt có pha 2 **lớn nhất** (`-1`, 12.880 seg) lại cho retention
  **tệ nhất** trong ba — vì pha 1 trên corpus loãng 2,48% đã đặt trần, pha 2 không gỡ
  được.

### v6-ondomain-15h (lọc nguyên cuộc mật độ ≥3%, một pha LoRA tươi)

- ngày: 2026-09-15 · corpus: 7.022 seg train, ~15,50 h (đo lại từ manifest,
  `docs/v6-ondomain-plan.md` bảng đầu file, khớp trong sai số làm tròn với tổng giờ
  toàn manifest 17,87 h × tỷ lệ seg train/tổng 87,3%), mật độ 7,46%
- rank/α: 16/32 · epoch 3 · LR 2e-4 · init_adapter: không
- tier2 OOD CER (tại λ=0,25, số gate tự chọn — xem ghi chú dưới): 2,27%

| λ báo cáo | cross-domain CER | retention | no_loanword CER | artifact |
|---|---:|---:|---:|---|
| 1,0 (checkpoint gốc, không rescale) | 9,19% | 61,57% | 7,31% | `Outputs/v6-ondomain-15h/metrics/cross_domain.json` |
| 0,75 (rescale) | 8,75% | 61,89% | 6,28% | `Outputs/v6-ondomain-15h/metrics/cross_domain_lambda075.json` |
| 0,25 (gate tự chọn, elbow rule) | 9,26% | 50,12% | 5,75% | `Outputs/v6-ondomain-15h/metrics/gate_results.json` (`cross_domain_bench`) |

⚠ λ=0,25 là số **gate thật sự đã chọn** (tier2 OOD CER 2,27% ở trên là của chính lượt
này) nhưng cho retention tệ nhất trong ba — bằng chứng trực tiếp cho
[[lambda-raises-retention-not-lowers]]: `sweep.retention_floor` vẫn null nên elbow rule
không thấy được đánh đổi này. Hai dòng λ=1,0/0,75 là rescore tay sau đó để so với v5.

- **Kết luận:** một pha thắng hai pha (so `v6-dense-*` ở trên). Nhưng ở cả λ=1,0 lẫn
  0,75 — hai điểm tốt nhất đo được — retention (61,6–61,9%) vẫn thua xa v5's 66,5% dù
  mật độ đã khớp gần đúng (7,46% so 7,13%). Biến còn lại nằm ở **thành phần corpus**,
  không phải mật độ hay số pha: YouTube của v5 là 2,46 h người soát toàn bộ, 0 segment
  toàn tiếng Anh; YouTube của `-15h` phần lớn nhãn máy và có segment toàn tiếng Anh.

### v6-ondomain-29h (quota theo cuộc θ=4,5%, so đối chứng với -15h)

- ngày: 2026-09-15 · corpus: 11.504 seg train, ~29,45 h (đo lại từ manifest, cùng cách
  ước lượng như trên: tổng 31,82 h × tỷ lệ train/tổng 91,9%), mật độ 7,42%
- rank/α: 16/32 · epoch 3 · LR 2e-4 · init_adapter: không
- λ báo cáo: 1,0 (không rescale) — **không có `gate_results.json`**, lượt này chỉ chạy
  qua script chấm cross-domain riêng, chưa qua gate/sweep đầy đủ
- tier2 OOD CER: không đo (không có sweep cho lượt này)
- cross-domain: CER 11,01% / retention 59,47% / no_loanword 10,83%
- artifact: `Outputs/v6-ondomain-29h.zip` → `metrics/cross_domain.json`
- **Kết luận:** `-15h` (retention 61,6%, 15,50 h) **thắng** `-29h` (retention 59,5%,
  29,45 h) dù ít hơn gần một nửa số giờ — giờ ở cùng dải mật độ (~7,4%) không phải đòn
  bẩy. `-29h` có 86,5% segment mang từ ngoại lai (quota theo cuộc dồn nhiều cuộc lệch
  vào), so với `-15h`'s lọc nguyên cuộc — độ đậm ép quá tay đổi lấy số lần tiếp xúc
  không đều, không phải chỉ tổng giờ. Đóng hướng: giờ ở cùng mật độ không phải đòn bẩy;
  biến thắng là **cách chọn cuộc** (lọc nguyên cuộc > quota theo cuộc).

---

### v6-100h-steps (v6-corpus + add-on TTS, 42.794 seg / 106,69 h / 3,55% mật độ)

- ngày: 2026-09-16 · máy H200 thuê theo giờ · commit `38f06e5`
- rank/α: 16/32 · epoch 2 · LR 2e-4 · batch 16 × accum 1 · init_adapter: không
- 5.350 bước, 88 phút (train 65 + eval 23), 0,730 giây/bước, đỉnh VRAM 77,1 GB
- val mở rộng 7 cuộc / 1.179 seg: ValCER 0,1754 (b.1600) → 0,1542 (b.3200) → 0,1293
  (b.4800), giảm đơn điệu, **chưa hội tụ** lúc dừng. Không so được với lượt cũ (val 4 cuộc)
- không chạy `baseline`/`sweep-gate` · không có cổng tự động

Đường cong retention theo số bước (`score_cross_domain_model`, batch 64, λ=1,0;
`Outputs/v6-100h-steps-metrics/cross_domain.step-*.json`):

| bước | 400 | **800** | 1200 | 1600 | 5350 |
|---|---:|---:|---:|---:|---:|
| cross-domain CER | 11,36% | 9,69% | 13,03% | 12,79% | 11,93% |
| retention | 48,95% | **62,82%** | 47,70% | 48,25% | 46,30% |

λ trên `step-800` (cùng đường đo): λ=0,75 → CER 8,80% / retention 61,18%; λ=0,5 →
CER 8,77% / retention 57,29%. λ=0,5 bị λ=0,75 chi phối hoàn toàn.

Bảng benchmark N3 (`Outputs/bench-merged-v6-100h/benchmark-table.N3.md`, CER %, so theo cột):

| model | cross-domain | synthetic-test | vimedcss-hard | vimedcss-test | youtube-test |
|---|---:|---:|---:|---:|---:|
| `scribe_v2` | **6,25** | — | **13,22** | **11,19** | 6,44 |
| v5 | 8,19 | **1,61** | 16,42 | 13,79 | **5,77** |
| step-800 λ=1,0 | 9,67 | 2,41 | 20,94 | — | 8,31 |
| step-800 λ=0,75 | 8,81 | — | — | — | — |
| base | 13,32 | 4,26 | 19,43 | 15,38 | 15,92 |

- **Kết luận:** đường cong retention **không giảm đơn điệu** — nó có đỉnh nhọn ở bước
  800, đúng ngân sách bước của v5 (801), và tụt về ~47–48% ở mọi bước khác. Dự đoán của
  `docs/v6-100h-steps-plan.md` §5 (`step-400` cao nhất, giảm đều) sai ở cả hai vế: 400
  bước cho retention 48,95%, thấp hơn 800 bước tới 13,9 điểm. Nhưng `step-800` vẫn thua
  v5 ở **cả bốn suite** nó chạy, ở mọi λ đã thử; nặng nhất là `vimedcss-hard` 20,94% —
  *tệ hơn cả base model* (19,43%). Corpus 106 giờ ở điểm dừng tốt nhất của nó không
  sinh ra model thay được v5.
- **Đóng hướng:** "dừng sớm theo số bước" không phải đòn bẩy còn lại — đỉnh tồn tại
  nhưng nằm đúng chỗ v5 đã đứng, và vẫn thấp hơn v5 5,1–7,3 điểm retention. Biến chưa
  bị loại vẫn là **thành phần corpus** (nhãn máy so với người soát), đúng kết luận 2 ở dưới.

## Hai kết luận đóng hướng đi (không re-litigate, xem `docs/plan-cat-pipeline-va-so-thi-nghiem.md` §2.3)

1. **Curriculum hai pha thua một pha.** `v6-dense-1/3/4` (nối từ
   `v6-corpus-r32/checkpoints/best`) đều thua `v6-ondomain-15h` (một pha LoRA tươi) ở
   cả CER lẫn retention. Lượt pha 2 lớn nhất cho retention tệ nhất — pha 1 loãng
   2,48% đặt trần trước khi pha 2 kịp sửa.
2. **Giờ ở cùng mật độ không phải đòn bẩy.** `v5` (7,77 h theo tính riêng, mật độ
   7,13%) đạt retention 66,5%; `-15h` (15,50 h, 7,46%) chỉ 61,6%; `-29h` (29,45 h,
   7,42%) còn thấp hơn, 59,5%. Mật độ đã khớp mà vẫn thua — và thêm giờ ở cùng mật độ
   còn **thua thêm**, không phải đứng yên. Biến còn lại nằm ở thành phần corpus: v5's
   YouTube là 2,46 h người soát toàn bộ, 0 segment toàn tiếng Anh; corpus v6 phần lớn
   nhãn máy (`google_asr`) và có segment toàn tiếng Anh.

## Việc chưa xong (đừng lặp lại nếu đọc sổ này trước khi chạy lượt mới)

- Tier1 in-domain và cross-domain của `v3-r16` (cả hai λ) chưa từng đo — bench
  cross-domain ra đời sau lượt này, không cần chạy lại để lấp chỗ trống, chỉ cần biết
  là "chưa đo" chứ không phải "đo được 0".
- Giờ corpus của `paid-dataset-v2` (v3-r16) không có artifact đo sẵn trên máy này.
- `v6-ondomain-29h` chưa từng qua gate/sweep — chỉ có số cross-domain đơn lẻ tại λ=1,0.
  Nếu cần tier1/tier2/λ khác cho nó, phải chạy `sweep-gate` thật, không suy ra được từ
  những gì đang có trên đĩa.
