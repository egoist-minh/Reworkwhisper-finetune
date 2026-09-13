# v6: vì sao 100 giờ dữ liệu không làm mô hình tốt hơn

Hồ sơ phân tích cho lượt tinh chỉnh tiếp theo. Mọi số trong file này đo lại từ artifact
của chính run, không trích từ bảng tổng hợp hay trí nhớ — đường dẫn ghi kèm từng mục.

## 0. Câu hỏi

`v6-corpus-r32` train trên 101,24 giờ, gấp 7,4 lần corpus của v5 (9,35 giờ theo manifest
gốc, 6,98 giờ theo phần train sau khi tách val/test). Kết quả: **thắng trên miền được
train, thua trên mọi thứ ngoài miền đó**. Cần biết vì sao trước khi tiêu thêm giờ GPU.

## 1. Kết quả, gọn

| Bộ đo | n | PhoWhisper-large | v5 | v6 (λ=0,5) | nguồn |
|---|---:|---:|---:|---:|---|
| cross-domain-bench | 299 | 13,32 | **8,19** | 10,85 | `Outputs/bench-cross-v6/scores.json` |
| Giữ từ ngoại lai, cross-domain | 1283 từ | 27,8% | **66,6%** | 48,6% | cùng file |
| YouTube test | 228 | 16,25 | 7,64 | **6,63** | `metrics/gate_results.json` mỗi run |
| Tổng hợp (TTS) | 426 | 4,24 | 2,58 | **1,98** | như trên |
| VIVOS | 760 | **2,26** | 2,26 | 2,82 | như trên |
| ViMedCSS hard / test | 658 / 1614 | 22,65 / 18,60 | 18,25 / 15,72 | chưa đo | `Outputs/benchmark-2026-09-10 (3)/scores.N0.json` |

Chỉ hàng cross-domain là v5 và v6 đo hoàn toàn cùng cách (cùng 299 segment, cùng chuẩn hoá
N3, cùng harness `scripts/benchmark_run.py` batch 8). Ba hàng giữa lấy từ gate của từng
run: cùng mã, cùng segment, nhưng v5 chạy `eval.batch_size=8` còn v6 chạy 64 — chênh dưới
~1 điểm ở các hàng đó chưa chắc là thật.

## 2. Nguyên nhân hàng đầu: corpus mới loãng từ ngoại lai gấp 2,8 lần

Đo trên chính nhãn dùng để train, chuẩn hoá bằng `src.normalize.Normalizer`, đếm token
bằng đúng bộ lọc của `src.metrics.english_token_retention` (dài > 1, `isalpha()`, trượt
phép thử hình dạng âm tiết tiếng Việt):

| Corpus train | seg | giờ | % token ngoại lai | token ngoại lai / giờ | dur trung vị |
|---|---:|---:|---:|---:|---:|
| **v5 — `mixed-noisy-v1`** | 4.717 | 6,98 | **7,13%** | **886,7** | 4,4 s |
| &nbsp;&nbsp;synthetic | 4.270 | 5,02 | 7,22% | 912,2 | 4,1 s |
| &nbsp;&nbsp;youtube | 447 | 1,96 | 6,88% | 821,2 | 15,7 s |
| **v6 — `v6-corpus`** | 34.972 | 101,23 | **2,48%** | **313,2** | — |
| &nbsp;&nbsp;synthetic (ElevenLabs 50 h) | 23.386 | 49,98 | 2,81% | 352,7 | 4,8 s |
| &nbsp;&nbsp;youtube (`google_asr`) | 11.586 | 51,25 | 2,17% | 274,7 | 15,7 s |

Đối chiếu với bộ đo đang thua: **cross-domain-bench có 7,41% token ngoại lai**
(1.283 / 17.312, tham chiếu do người soát). Corpus của v5 trùng mật độ đó gần như chính
xác (7,13%); corpus của v6 loãng hơn ba lần.

Mô hình học đúng cái được cho ăn: v6 phiên âm từ tiếng Anh thành tiếng Việt vô nghĩa, còn
v5 giữ nguyên. Ba câu tụt nặng nhất (`Outputs/bench-cross-v6/*.persegment.jsonl`):

| tham chiếu | v6 | v5 |
|---|---|---|
| `devops` | `để báo`, `đáp ốc` | `dev off` |
| `customer journey` | `cách sử dụng` | `customer journey` |
| `awareness` | `quen thế` | `awareness` |

Phân bố cũng hợp với cách hỏng này: 144/299 câu v6 tệ hơn v5 quá 2 điểm, 35 câu tốt hơn,
120 hoà; dCER trung vị +1,85 điểm. Tụt nặng nhất ở **câu ngắn** (CER trung bình mỗi câu
17,03 so với 8,82), nơi một từ tiếng Anh hỏng chiếm tỉ trọng lớn trong câu.

Một chi tiết nên biết: 447 segment YouTube trong corpus của v5 chính là 4 cuộc
`dGT3YW0AdD8`, `iyeFAuuEBl4`, `rIFrrmm8ILY`, `xKDHjUoUN54` — cùng 4 cuộc vừa thêm vào
v6-corpus ngày 12/09/2026. Phần YouTube của hai corpus không mâu thuẫn nhau; khác biệt nằm
ở 49,98 giờ synthetic mới và 49,29 giờ YouTube `google_asr` mới.

## 3. Nguyên nhân phụ, chưa loại trừ

**(a) Nhãn `google_asr` chưa soát.** Nửa corpus (50,70 giờ) mang nhãn máy; trong đó
**13,87 giờ có `verified: false`** (3.128 segment), 37,38 giờ đã soát. `src/data.py` train
trên trường `text`; các bản sửa QC nằm ở `qc_corrected_text` **không được dùng** — đúng với
mọi run trước đó, không phải lỗi mới, nhưng nghĩa là công soát chỉ vào corpus khi ai đó ghi
đè `text`. Cần kiểm: trong drop `youtube-meeting-1`, bao nhiêu dòng có
`qc_corrected_text != text`.

**(b) λ\* = 0,5 với rank 32 để lại lượng thích nghi lớn hơn λ\* = 0,25 với rank 16 của v5.**
Sweep chỉ ràng buộc OOD trên VIVOS — giọng đọc rõ, câu có kịch bản, **0% token ngoại lai** —
nên nó không hề chặn kiểu hỏng đang xảy ra. VIVOS 2,82% nằm trong ngân sách 4,26% trong khi
cross-domain tụt 2,66 điểm: **ngân sách gate hiện tại mù với lỗi này.**

**(c) Một epoch trên 101 giờ so với ba epoch trên 6,98 giờ.** 2.186 step so với 801.
Loss train rơi hết đà trong ~200 step đầu rồi phẳng ở 0,20–0,25 suốt 1.900 step còn lại
(`metrics/training.csv`) — sàn đó nhiều khả năng là sàn nhiễu nhãn, không phải giới hạn
năng lực. Thêm epoch sẽ học thuộc nhiễu, không hạ được sàn.

**(d) Độ dài segment lệch.** YouTube 15,7 s trung vị so với synthetic 4,8 s. v5 train chủ
yếu trên câu 4,4 s. Chưa đo ảnh hưởng.

## 4. Phép kiểm rẻ nhất cho từng giả thuyết

Xếp theo tỉ lệ thông tin thu được trên giờ GPU:

| # | Giả thuyết | Cách kiểm | Chi phí |
|---|---|---|---|
| 1 | λ là thủ phạm, không phải dữ liệu | Decode cross-domain ở λ=0,25 và 0,75 từ chính adapter đã có | ~3 phút GPU mỗi λ (merge + 299 segment) |
| 2 | Mật độ từ ngoại lai quyết định | Train lại chỉ trên phần synthetic 50 h (mật độ 2,81%) rồi chỉ trên `mixed-noisy-v1` + 4 cuộc YouTube (7,13%), so retention | ~30 phút GPU mỗi lượt |
| 3 | 13,87 giờ nhãn chưa soát làm hỏng | Train lại loại `verified: false`, còn 87,4 giờ | ~45 phút GPU |
| 4 | Gate mù với lỗi này | Thêm cross-domain-bench thành tier của gate, hoặc thêm ràng buộc retention vào `sweep` | chỉ code, 0 giờ GPU |
| 5 | ViMedCSS thì sao | Decode v6 trên `vimedcss-hard` + `vimedcss-test` | ~15 phút GPU, 2.272 segment |

Việc số 4 nên làm trước khi thuê máy: mọi lượt tinh chỉnh sau đó sẽ tự bị chặn nếu tái
phạm, thay vì phát hiện sau khi đã push.

## 5. Cấu hình và chi phí của run này

Commit `367d88a`, nhánh `h200-server-run`, chạy trên H200 thuê theo `docs/server-finetune.md`.

| | |
|---|---|
| override | `lora.rank=32`, `alpha=64`, `batch_size=16`, `grad_accum=1`, `epochs=1`, `eval_steps=364`, `eval.batch_size=64`, `gradient_checkpointing=false`, `real_bench_path=null` |
| step | 2.186 (1 epoch) |
| thời gian train thực tế | 51 phút 30 (gồm 6 lượt eval; bị giết ở step 1144, resume từ `checkpoint-1092`) |
| tính toán thuần | 0,723 s/step (trung vị), khớp mốc đo 11/09 |
| eval giữa chừng | 23,3 phút — gần bằng 26,3 phút tính toán train |
| đỉnh VRAM | 77,7 GB allocated / 90,8 GB reserved trên 143,8 |
| toàn pipeline | 68 phút 50 đồng hồ thật (baseline 2'50 + train + sweep-gate 9'30 + 5 phút chết) |

Gate: tier1 4,79% (trần 10,34%), tier2 2,82% (trần 4,26%), tier4a tắt → `overall_pass: true`,
λ\* = 0,5, đã push.

## 6. Tài liệu và dữ liệu nằm ở đâu

| Thứ | Đường dẫn |
|---|---|
| Artifact của run (adapter, best, metrics, audit, manifest) | `Outputs/v6-corpus-r32.zip` (19 file, 432 MB) |
| Log train (lần đầu + resume) | `Outputs/v6-corpus-r32.baseline-train-partial.log`, `Outputs/v6-corpus-r32.run.log` |
| Hypothesis từng segment, cross-domain, 4 model | `Outputs/bench-cross-v6/*.persegment.jsonl` + `scores.json` + `benchmark-table.N3.md` |
| Ma trận benchmark v5 / Scribe / base (09/09/2026) | `Outputs/benchmark-2026-09-10 (3)/scores.json` (N3) và `scores.N0.json` (ViMedCSS) |
| Bảng so sánh v5–v6 để dán vào sheet | `Outputs/bang-tong-hop/v5-vs-v6-2026-09-12.tsv` |
| Corpus | HF `rework-whisper-v6-org/v6-corpus`, commit `c5951ae` (private, 14,9 GB, 35.991 record) |
| Adapter đã publish | HF `rework-whisper-v6-org/Reworkwhisper-large-v6` — **adapter-only**, khác v5 (model đã merge), nên `from_pretrained` thẳng repo sẽ lỗi thiếu `preprocessor_config.json` |
| Kế hoạch run, override, ngân sách | `docs/v6-finetune-plan.md` |
| Quy trình chạy trên máy thuê | `docs/server-finetune.md` |
| Chi phí step đo trên H200 | `docs/h200-timing-probe-2026-09-11.md` |

Chưa ghi: `experiments/task_ledger.md` và `provenance.md` cho run này.

## 7. Đề xuất cho lượt tinh chỉnh tới

1. Thêm ràng buộc giữ từ ngoại lai vào `sweep`/gate (việc 4 ở §4) — không tốn GPU, chặn
   được đúng kiểu hỏng đã xảy ra.
2. Chạy việc 1 (λ=0,25 / 0,75) để tách hẳn "lỗi chọn λ" khỏi "lỗi dữ liệu" trước khi quyết
   định đụng vào corpus.
3. Nếu đúng là dữ liệu: **không thêm giờ, thêm mật độ** — sinh thêm dữ liệu synthetic có
   code-switch dày như `mixed-noisy-v1` (7,1%), hoặc soát lại 13,87 giờ nhãn `verified:
   false` và ghi bản sửa vào `text`.
4. Hạ `eval_steps` còn 4 lượt cho lần chạy tới: tiết kiệm ~8 phút GPU mà vẫn đủ điểm dừng.
