# Số liệu tổng hợp — PhoWhisper-large và các bản fine-tune

> **Phạm vi tài liệu.** Đây là bản tập hợp số liệu thuần tuý. Không có mục nhận xét, không
> có kết luận, không có khuyến nghị. Mỗi khối ghi kèm nguồn để tra ngược.
>
> **Quy ước đặt tên.** Gọi mô hình bằng tên đã publish trên HuggingFace, không dùng run ID.
> Ánh xạ ở mục 1.
>
> **Cách sinh số.** Các bảng ở mục 5, 6, 7, 11 sinh bằng `python -m scripts.paper_tables` và
> `python -m scripts.rescore_pilot_benchmark`. Ô trống nghĩa là **chưa đo**, không phải 0.

---

## 1. Ánh xạ tên mô hình

| Tên mô hình | Adapter (run ID) | λ | Trạng thái |
|---|---|---:|---|
| `vinai/PhoWhisper-large` | — | — | mô hình nền, không fine-tune |
| `Reworkwhisper-large-v4` | `v3-r16` | 0,5 | đã publish |
| `Reworkwhisper-large-v5` | `v4-mixed-r16` | 0,75 | đã publish, đang chạy production từ 2026-08-18 |

Adapter `v4-mixed-r16` còn được chấm ở hai mức λ khác. Trong tài liệu này ghi là
**`Reworkwhisper-large-v5` @ λ=0,25** và **@ λ=0,5** — cùng một adapter, khác hệ số hợp nhất,
không phải mô hình khác.

Dữ liệu huấn luyện của hai adapter khác nhau: `v3-r16` train trên `paid-dataset-v2` (chỉ
giọng tổng hợp); `v4-mixed-r16` train trên `mixed-noisy-v1` (giọng tổng hợp + họp thật).

---

## 2. Số liệu corpus

### 2.1 Corpus họp YouTube

| Chỉ số | Giá trị |
|---|---:|
| Số buổi | 7 |
| Số đoạn | 790 |
| Thời lượng | 207,42 phút |
| Số từ | 41.210 |
| Code-switch — token | 6,67% |
| Code-switch — đoạn có ít nhất 1 token | 89,7% |
| Chồng tiếng — số giây thực có 2 người nói | 0,89% |
| Chồng tiếng — theo công thức của mốc tham chiếu | 10,57% |
| Tương đồng người nói ECAPA — trong-buổi | 0,54 |
| Tương đồng người nói ECAPA — giữa-buổi | 0,32 |

Sàng lọc: 3 luật chạy trên phụ đề tự động trước khi tải audio; 7 nhận, 0 loại.
Cắt đoạn: cửa sổ 30 phút liên tục căn giữa video; 6/7 buổi đúng 1800,0 giây.
Điểm cắt lấy từ mốc thời gian từng từ của phụ đề, không từ khoảng lặng trên dạng sóng.

### 2.2 Trạng thái nhãn corpus YouTube

| Buổi | Split | n | `correct` | `minor_edit` | Trường trạng thái để trống | Người soát |
|---|---|---:|---:|---:|---:|---|
| `3nuCdzuyqng` | test | 114 | 95 | 19 | 0 | Quang |
| `7B24A9GfHAo` | test | 114 | 0 | 0 | 114 | Quang |
| `dGT3YW0AdD8` | train | 105 | 0 | 0 | 105 | Quang |
| `iyeFAuuEBl4` | train | 115 | 0 | 0 | 115 | quang |
| `rCd8DSMk3-c` | val | 115 | 74 | 41 | 0 | Nguyên, Quang |
| `rIFrrmm8ILY` | train | 113 | 0 | 0 | 113 | Quang |
| `xKDHjUoUN54` | train | 114 | 0 | 0 | 114 | Quang |
| **Tổng** | | **790** | **169** | **60** | **561** | |

`label_source` = `google_asr` trên cả 790 bản ghi. `verified` = `true` trên cả 790.
229 đoạn có ghi trạng thái ở mức từng đoạn (`correct` hoặc `minor_edit`).

Cột "Trường trạng thái để trống" đếm số đoạn **không điền trường trạng thái**, không phải số
đoạn chưa soát. Nhóm soát xác nhận (2026-09-08) toàn bộ 790 đoạn đều đã được soát và hiệu
đính; 561 đoạn còn lại chỉ thiếu thao tác cập nhật trường trạng thái. Số liệu độc lập nhất
quán với xác nhận này: text khác nhau giữa nhãn nháp và nhãn dùng thật ở 779/790 đoạn.
Nguồn của đoạn này là xác nhận của nhóm soát, không suy ra được từ manifest.

Đối chiếu nhãn nháp và nhãn dùng thật, chuỗi `grap`/`grab`:

| Chuỗi | Nhãn nháp | Nhãn dùng thật |
|---|---:|---:|
| `grap` | 21 | 0 |
| `grab` | 9 | 43 |

Text khác nhau giữa nhãn nháp và nhãn dùng thật ở **779/790** đoạn.

### 2.3 Bốn tầng độc lập giữa train và test (corpus YouTube)

| Tầng | Trạng thái |
|---|---|
| `meeting_id` và file audio | đạt — không buổi nào chung |
| Người nói | chưa xác định độc lập; `xKDHjUoUN54` (train) tương đồng ECAPA cao với cả 2 buổi test |
| Chủ đề | trùng — cả 7 buổi cùng chủ đề phỏng vấn tuyển dụng kỹ thuật |
| Từ vựng | trùng — 81 type xuất hiện ở ≥3/7 buổi |

### 2.4 VIVOS

Tập test chuẩn, 760 câu. Số token không mang hình dạng âm tiết tiếng Việt: **0**.

### 2.5 Benchmark audio thu thật

2 bản ghi, 43,2 phút, 196 đoạn gốc. Transcript là bản nháp PhoWhisper-small đã người sửa.

### 2.6 Ba buổi giữ riêng

| Buổi | Lĩnh vực | Đoạn | Từ (tham chiếu) | Thời lượng |
|---|---|---:|---:|---:|
| `IGZYBrbDUEw` | tuyển dụng nhân sự | 114 | 6.809 | 30 phút |
| `ySdJ3sg_2lk` | hội chẩn y khoa | 69 | 4.010 | 18 phút |
| `z-nODwLyhA0` | webinar marketing | 116 | 6.520 | 30 phút |
| **Tổng** | | **299** | | **78 phút** |

Cả 299 đoạn `verified: true`.

---

## 3. Phân chia dữ liệu của `Reworkwhisper-large-v5`

| Split | Họp thật | Tổng hợp | Tổng đoạn |
|---|---|---|---:|
| train | 447 đoạn · 117,4 phút | 4.270 đoạn · 301,1 phút | 4.717 |
| val | 115 đoạn · 30,0 phút | 250 đoạn · 17,5 phút | 365 |
| test | 228 đoạn · 60,0 phút | 426 đoạn · 34,9 phút | 654 |

Họp thật trong train: **1,96 giờ**.

Buổi trong tập test:

| `meeting_id` | Nguồn | Đoạn |
|---|---|---:|
| `3nuCdzuyqng` | họp thật | 114 |
| `7B24A9GfHAo` | họp thật | 114 |
| `paid_meeting_test_0001` | tổng hợp | 161 |
| `paid_meeting_test_0002` | tổng hợp | 159 |
| `paid_meeting_test_0003` | tổng hợp | 106 |

Giọng tổng hợp: 10 giọng ở test, không giao với 10 giọng ở train.

---

## 4. Cấu hình huấn luyện `Reworkwhisper-large-v5`

| Mục | Giá trị |
|---|---|
| Mô hình nền | `vinai/PhoWhisper-large` |
| Phương pháp | LoRA, `use_rslora: true` |
| Hạng | 16 |
| Alpha | 32 |
| Dropout | 0,05 |
| Ma trận đích | `q_proj`, `k_proj`, `v_proj`, `out_proj`, `fc1`, `fc2` |
| Tham số huấn luyện | 28,8 triệu / 1,64 tỷ = 1,76% |
| Learning rate | 2×10⁻⁴ |
| Epoch | 3 |
| Warmup ratio | 0,1 |
| Seed | 42 |
| Gradient checkpointing | bật |
| Số bước cập nhật | 885 |
| Phần cứng | 1× GPU T4 |

Cấu hình của `Reworkwhisper-large-v4` giống hệt trừ ba mục: `dataset_path`, `val_meetings`
(thiếu `rCd8DSMk3-c`), và số bước cập nhật (801).

**Chuẩn hoá khi chấm điểm**, áp giống hệt cho giả thuyết và tham chiếu: NFC, hạ chữ thường,
bỏ dấu câu, gộp khoảng trắng, loại từ đệm `ừm ờm ehm uhm hmm`, quy đổi số viết bằng chữ về
chữ số. Giải mã: greedy (`num_beams: 1`), `language: vi`, batch 8, fp16.

---

## 5. Kết quả trên tập test 654 đoạn

| Chỉ số | `PhoWhisper-large` | `Reworkwhisper-large-v4` | `-v5` @ λ=0,5 | `Reworkwhisper-large-v5` |
|---|---:|---:|---:|---:|
| Họp thật · CER | 15,934% | 12,052% | 6,242% | 5,930% |
| Tổng hợp · CER | 4,259% | 1,958% | 1,853% | 1,608% |
| Gộp 654 · CER | 11,308% | 8,053% | 4,503% | 4,217% |
| Họp thật · WER | 22,687% | 16,098% | 10,107% | 9,371% |
| Tổng hợp · WER | 7,404% | 2,967% | 2,887% | 2,366% |
| Gộp 654 · WER | 16,721% | 10,972% | 7,289% | 6,636% |
| Họp thật · giữ từ mượn | 36,384% | 77,460% | 82,265% | 85,469% |
| Tổng hợp · giữ từ mượn | 55,741% | 83,090% | 83,090% | 86,221% |
| Gộp 654 · giữ từ mượn | 43,237% | 79,453% | 82,557% | 85,735% |

Mẫu số của chỉ số giữ từ mượn: họp thật 874 lượt, tổng hợp 479 lượt, gộp 1.353 lượt
(7,1% tổng số từ của tập test).

Số lượt giữ đúng: họp thật 318 / 677 / 719 / 747 · tổng hợp 267 / 398 / 398 / 413 ·
gộp 585 / 1.075 / 1.117 / 1.160, theo thứ tự bốn cột.

Khoảng tin cậy bootstrap ở mức đoạn, `Reworkwhisper-large-v5`: họp thật
[5,419%; 6,451%] · tổng hợp [1,314%; 1,892%]. Mốc nền tương ứng: họp thật
[12,049%; 23,345%] · tổng hợp [3,780%; 4,731%].

Cả bốn cột chấm trên cùng 654 khoá `(meeting_id, segment_id)`; tham chiếu lệch nhau **0 dòng**.

---

## 6. Kết quả theo từng buổi trong tập test

Mỗi ô: CER / tỷ lệ giữ từ mượn.

| Buổi | n | `PhoWhisper-large` | `Reworkwhisper-large-v4` | `-v5` @ λ=0,5 | `Reworkwhisper-large-v5` |
|---|---:|---|---|---|---|
| `3nuCdzuyqng` | 114 | 12,125% / 30,2% | 11,211% / 79,2% | 5,786% / 83,8% | 5,366% / 85,7% |
| `7B24A9GfHAo` | 114 | 20,019% / 41,0% | 12,955% / 76,1% | 6,732% / 81,1% | 6,534% / 85,3% |
| `paid_meeting_test_0001` | 161 | 5,016% / 57,8% | 2,165% / 81,0% | 2,165% / 80,2% | 1,949% / 83,6% |
| `paid_meeting_test_0002` | 159 | 3,193% / 58,5% | 1,878% / 83,1% | 1,758% / 84,6% | 1,495% / 85,4% |
| `paid_meeting_test_0003` | 106 | 4,983% / 48,7% | 1,689% / 87,2% | 1,391% / 87,2% | 1,126% / 92,3% |

---

## 7. VIVOS — tập test chuẩn 760 câu

| Mô hình | CER | WER |
|---|---:|---:|
| `vinai/PhoWhisper-large` | 2,28% | 4,73% |
| `Reworkwhisper-large-v5` @ λ=0,25 | 2,26% | 4,64% |
| `Reworkwhisper-large-v5` @ λ=0,5 | 2,62% | — |
| `Reworkwhisper-large-v5` | 3,34% | 6,83% |
| `Reworkwhisper-large-v4` | — | — |

Ngân sách VIVOS CER dùng trong luật chọn λ: 0,0428.

---

## 8. Benchmark audio thu thật

| Mô hình | CER | Khoảng tin cậy | Giữ từ mượn |
|---|---:|---|---:|
| `vinai/PhoWhisper-large` | 31,572% | — | — |
| `Reworkwhisper-large-v5` @ λ=0,25 | 28,902% | [17,415%; 43,538%] | 45,45% |
| `Reworkwhisper-large-v5` @ λ=0,5 | 24,534% | — | 53,59% |
| `Reworkwhisper-large-v5` | 24,401% | [14,040%; 38,200%] | 56,78% |

Số của `Reworkwhisper-large-v4` trên benchmark này đo trên phân chia khác (196 đoạn,
base 31,57 → 29,06, khoảng tin cậy của hiệu là [−0,010; +0,076]) nên không xếp cùng bảng.

---

## 9. Đường cong hệ số hợp nhất — adapter `v4-mixed-r16`

### 9.1 Trên tập validation

| λ | val gộp | val tổng hợp | val họp thật | VIVOS |
|---:|---:|---:|---:|---:|
| 0,00 | 0,10337 | 0,05830 | 0,12899 | 0,02283 |
| 0,25 | 0,05483 | 0,02855 | 0,06977 | 0,02261 |
| 0,50 | 0,03636 | 0,01510 | 0,04845 | 0,02619 |
| 0,75 | 0,03316 | 0,00998 | 0,04633 | 0,03343 |
| 1,00 | 0,03257 | 0,00954 | 0,04565 | 0,04257 |

### 9.2 Trên tập test

| λ | Họp thật CER | Tổng hợp CER | Audio thu thật CER | Giữ từ mượn (họp thật) | VIVOS CER |
|---:|---:|---:|---:|---:|---:|
| 0,00 | 15,934% | 4,259% | 31,572% | 36,38% | 2,28% |
| 0,25 | 7,643% | 2,583% | 28,902% | 70,94% | 2,26% |
| 0,50 | 6,242% | 1,853% | 24,534% | 82,27% | 2,62% |
| 0,75 | 5,930% | 1,608% | 24,401% | 85,47% | 3,34% |
| 1,00 | — | — | — | — | 4,26% |

WER tương ứng: λ=0,25 họp thật 12,127% / tổng hợp 4,197%; λ=0,5 họp thật 10,107% /
tổng hợp 2,887%; λ=0,75 họp thật 9,371% / tổng hợp 2,366%.

Hàng λ=1,0 trên tập test **không có artifact trên đĩa**. Riêng VIVOS ở λ=1,0 có trong
`lambda_sweep.csv`.

Hình: `docs/training-curves/paper-fig1-lambda.png`.

---

## 10. Giữ từ mượn tách theo độ phủ từ vựng huấn luyện

Từ vựng huấn luyện: **1.129 type, 6.185 lượt** token không mang hình dạng âm tiết tiếng
Việt, lấy từ tham chiếu của split train.

### 10.1 Trên tập test 654 đoạn

| Mô hình | Slice | Đã thấy | Chưa thấy |
|---|---|---|---|
| `vinai/PhoWhisper-large` | họp thật | 37,0% (255/689) | 34,1% (63/185) |
| `vinai/PhoWhisper-large` | tổng hợp | 53,6% (180/336) | 60,8% (87/143) |
| `Reworkwhisper-large-v5` | họp thật | 90,1% (621/689) | 68,1% (126/185) |
| `Reworkwhisper-large-v5` | tổng hợp | 90,2% (303/336) | 76,9% (110/143) |

### 10.2 Trên 3 buổi giữ riêng

| Buổi | Mô hình | Đã thấy | Chưa thấy |
|---|---|---|---|
| `IGZYBrbDUEw` | ElevenLabs Scribe v2 | 92,1% (255/277) | 54,4% (68/125) |
| `IGZYBrbDUEw` | `Reworkwhisper-large-v5` | 77,6% (215/277) | 31,2% (39/125) |
| `ySdJ3sg_2lk` | ElevenLabs Scribe v2 | 100,0% (1/1) | 49,3% (36/73) |
| `ySdJ3sg_2lk` | `Reworkwhisper-large-v5` | 100,0% (1/1) | 21,9% (16/73) |
| `z-nODwLyhA0` | ElevenLabs Scribe v2 | 90,3% (399/442) | 86,6% (316/365) |
| `z-nODwLyhA0` | `Reworkwhisper-large-v5` | 85,1% (376/442) | 53,7% (196/365) |
| **Gộp** | ElevenLabs Scribe v2 | **91,0%** (655/720) | **74,6%** (420/563) |
| **Gộp** | `Reworkwhisper-large-v5` | **82,2%** (592/720) | **44,6%** (251/563) |

Token chưa thấy mà `Reworkwhisper-large-v5` mất nhiều lượt nhất: `jd`(26) · `vinpearl`(24) ·
`funnel`(15) · `hunt`(10) · `acid`(10) · `uric`(10) · `headhunt`(9) · `own`(9) ·
`digital`(6) · `myg`(6) · `mg`(5) · `salesforce`(5) · `loyalty`(5) · `monta`(5) ·
`touchpoint`(5) · `purchase`(5) · `helpdesk`(4) · `pmax`(4) · `pearl`(4) · `linkedin`(3).

---

## 11. Ba buổi giữ riêng — chấm lại bằng chuẩn hoá của repo này

| Buổi | n | Mô hình | CER | WER | Giữ từ mượn |
|---|---:|---|---:|---:|---:|
| `IGZYBrbDUEw` | 114 | ElevenLabs Scribe v2 | 5,295% | 8,747% | 80,35% (323/402) |
| `IGZYBrbDUEw` | 114 | `Reworkwhisper-large-v5` | 6,487% | 10,453% | 63,18% (254/402) |
| `ySdJ3sg_2lk` | 69 | ElevenLabs Scribe v2 | 8,213% | 12,912% | 50,00% (37/74) |
| `ySdJ3sg_2lk` | 69 | `Reworkwhisper-large-v5` | 11,673% | 17,083% | 22,97% (17/74) |
| `z-nODwLyhA0` | 116 | ElevenLabs Scribe v2 | 6,783% | 12,319% | 88,60% (715/807) |
| `z-nODwLyhA0` | 116 | `Reworkwhisper-large-v5` | 8,479% | 14,949% | 70,88% (572/807) |
| **Gộp** | **299** | ElevenLabs Scribe v2 | **6,535%** | **11,053%** | **83,79%** (1075/1283) |
| **Gộp** | **299** | `Reworkwhisper-large-v5` | **8,436%** | **13,676%** | **65,71%** (843/1283) |

Số của cùng bộ dữ liệu khi chấm bằng `normalize_vi` của `viet-speech` (không có quy ước số):

| Mô hình | CER | WER |
|---|---:|---:|
| ElevenLabs Scribe v2 | 8,1% | 11,6% |
| `Reworkwhisper-large-v5` | 9,7% | 14,0% |

RTF ghi nhận trong lần chấm gốc: ElevenLabs 0,136–0,173 (API); `Reworkwhisper-large-v5`
0,335–0,365 (GPU T4).

**Điều kiện của lần chấm này:** chạy 2026-08-27 trên Google Colab; cấu hình giải mã có
`no_repeat_ngram_size`, `no_speech_threshold`, `condition_on_prev_tokens=False` — khác cấu
hình giải mã dùng ở mục 5–10. Bản `Reworkwhisper-large-v5` được tải từ HuggingFace tại thời
điểm đó; revision hash chưa được ghi lại.

---

## 12. Số kiểm chứng phụ

### 12.1 Quy ước viết số

| Cách chấm | `Reworkwhisper-large-v5` @ λ=0,25 | `Reworkwhisper-large-v5` |
|---|---:|---:|
| Quy đổi chữ về số | 5,638% | 4,217% |
| Giữ nguyên cách viết | 6,085% | 4,379% |
| Chênh lệch | 0,447 điểm | **0,161 điểm** |

### 12.2 Đoạn dị thường của mô hình nền

| Mục | Giá trị |
|---|---|
| Đoạn | `7B24A9GfHAo/seg_0079` |
| CER | 2543,8% |
| Số phép sửa | 1.628 |
| Độ dài tham chiếu | 64 ký tự |
| CER họp thật của mô hình nền, đủ 228 đoạn | 15,934% |
| CER họp thật của mô hình nền, loại đoạn này | 12,645% |
| Tần suất | 1/228 đoạn |

Ba đoạn CER cao nhất của mô hình nền trên slice họp thật: 2543,8% · 90,9% · 67,2%.

### 12.3 Bộ nhận diện từ mượn

| Mục | Giá trị |
|---|---:|
| Type được đánh dấu trên tập test | 362 |
| Lượt được đánh dấu | 1.353 |
| Type đáng ngờ khi rà soát | 7 |
| Lượt đáng ngờ | 15 |
| Tỷ lệ đáng ngờ theo lượt | 1,1% |

Type đáng ngờ: `politic`(7) · `cashing`(2) · `catching`(2) · `mili`(1) · `massage`(1) ·
`ht`(1) · `nonfunction`(1). Rà soát do nhóm tác giả thực hiện, không phải kiểm định độc lập.

Điều kiện chọn ứng viên: độ dài > 1 ký tự, `isalpha()`, và trượt phép thử hình dạng âm tiết
tiếng Việt. Danh sách trắng từ tiếng Anh không được dùng — cách đó đo được 72% dương tính giả.

---

## 13. Số công bố của PhoWhisper

Le, Nguyen & Nguyen. *PhoWhisper: Automatic Speech Recognition for Vietnamese.*
ICLR 2024 Tiny Papers Track. arXiv:2406.02555.

### 13.1 Dữ liệu huấn luyện (Bảng 1 của bài gốc)

| Bộ | Train (giờ) | Val (giờ) | Test (giờ) | Âm tiết/phát ngôn (min–max \| trung bình) |
|---|---:|---:|---:|---|
| CMV–Vi | 143,04 | 0,41 | 1,35 | 1–14 \| 7,55 |
| VIVOS | 13,94 | 0,98 | 0,75 | 2–30 \| 13,25 |
| VLSP 2020 Task-1 | 240,91 | 2,53 | 7,50 | 1–349 \| 17,52 |
| VLSP 2020 Task-2 | — | — | 6,01 | — |
| Dữ liệu riêng | 585,90 | — | — | 11–24 \| 16,90 |
| **Tổng** | **843,79** | **3,92** | **15,61** | |

Dữ liệu riêng: 26 nghìn người, 63 tỉnh thành. Tăng cường nhiễu bằng ESC-50 qua
`audiomentations`, áp cho một nửa tập train.

### 13.2 Kết quả (Bảng 2 của bài gốc) — WER

| Mô hình | Tham số | CMV–Vi | VIVOS | VLSP T1 | VLSP T2 |
|---|---:|---:|---:|---:|---:|
| wav2vec2-base-vietnamese-250h | 95M | 102,04 | 10,83 | 21,02 | 50,35 |
| wav2vec2-base-vi-vlsp2020 | 95M | 103,71 | 9,90 | 16,82 | 44,91 |
| wav2vec2-large-vi-vlsp2020 | 317M | 101,41 | 8,61 | 15,18 | 36,75 |
| PhoWhisper-tiny | 39M | 19,05 | 10,41 | 20,74 | 49,85 |
| PhoWhisper-base | 74M | 16,19 | 8,46 | 19,70 | 43,01 |
| PhoWhisper-small | 244M | 11,08 | 6,33 | 15,93 | 32,96 |
| PhoWhisper-medium | 769M | 8,27 | 4,97 | 14,12 | 26,85 |
| PhoWhisper-large | 1,55B | 8,14 | 4,67 | 13,75 | 26,68 |

### 13.3 Cấu hình huấn luyện của bài gốc

Fine-tune đầy đủ, không LoRA. 8× A100 40GB, batch mỗi thiết bị 4, tích luỹ gradient 2,
batch toàn cục 64. Learning rate đỉnh cho bản large: 5×10⁻⁶. Tổng 48.000 bước ≈ 5 epoch.

---

## 14. Nguồn từng khối

| Khối | Nguồn |
|---|---|
| Mục 2.1, 2.3 | `docs/youtube-data-report.md` |
| Mục 2.2, 3 | `Outputs/v4-mixed-r16/validated_manifest.jsonl` |
| Mục 2.2 (nhãn nháp) | `dataset/youtube-meetings/manifest.*.jsonl` |
| Mục 2.6, 11 | `Outputs/review_youtube_data_pilot-20260827T054523Z-1-001.zip` · `docs/bao-cao-benchmark-elevenlabs-scribe2-vs-reworkwhisper-v5-2026-08-27.md` |
| Mục 4 | `Outputs/v4-mixed-r16/config.json` · `Outputs/v3-r16/config.json` |
| Mục 5, 6, 7, 10.1 | `python -m scripts.paper_tables` |
| Mục 5 (khoảng tin cậy) | `Outputs/lambda075-metrics/v4-mixed-r16-lambda0.75/metrics/gate_results.json` |
| Mục 8, 9.2, 12.1 | `Outputs/v4-mixed-r16/metrics/gate_results.json` · `Outputs/v4-mixed-r16-lambda0.5/summary.json` · `Outputs/lambda075-metrics/v4-mixed-r16-lambda0.75/metrics/gate_results.json` |
| Mục 8 (`Reworkwhisper-large-v4`) | `docs/score-table.md` |
| Mục 9.1 | `Outputs/v4-mixed-r16/metrics/lambda_sweep.csv` |
| Mục 10.2, 11 | `python -m scripts.rescore_pilot_benchmark` |
| Mục 12.2, 12.3 | tính lại từ `audit/predictions_*.csv` bằng `src/metrics.py` |
| Mục 13 | arXiv:2406.02555 |

---

## 15. Ô trống — chưa đo

| Ô | Lý do |
|---|---|
| Hàng λ=1,0 trên tập test (mục 9.2) | Không có artifact trên đĩa; chỉ có bản ghi văn xuôi trong `SESSIONS.md` |
| VIVOS WER ở λ=0,5 (mục 7) | Không có file predictions cho slice này |
| VIVOS của `Reworkwhisper-large-v4` (mục 7) | Chưa chấm trên phân chia của lần chạy này |
| `vinai/PhoWhisper-large` trên 3 buổi giữ riêng (mục 10.2, 11) | Chưa chạy |
| `Reworkwhisper-large-v4` trên 3 buổi giữ riêng (mục 10.2, 11) | Chưa chạy |
| Revision hash của lần chấm 2026-08-27 (mục 11) | Chưa ghi lại trong log |
| Precision của bộ nhận diện từ mượn theo kiểm định độc lập (mục 12.3) | Chưa có người soát thứ hai |
| CER theo đường giải mã của hệ thống production | Chưa chạy; công cụ ở `scripts/eval_module4.py` |
